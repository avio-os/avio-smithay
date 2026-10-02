use std::{
    fmt,
    os::fd::{AsFd, OwnedFd},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

use ash::{khr, vk};
use tracing::warn;

use crate::backend::renderer::sync::{Fence, Interrupted};

use super::{device::DeviceHandle, VulkanRendererError};

pub(crate) struct VulkanFence {
    inner: VulkanFenceInner,
}

struct VulkanFenceInner {
    device: Arc<DeviceHandle>,
    fence: vk::Fence,
    /// SYNC_FD exported from the submission's dedicated export semaphore, set
    /// by the device immediately after a successful `vkQueueSubmit` while the
    /// signal operation is provably pending.
    ///
    /// The VkFence itself is NEVER exported: `vkGetFenceFdKHR(SYNC_FD)` has
    /// move semantics (it consumes the fence payload) and was observed on
    /// NVIDIA to race fence completion, producing a valid fd bound to an
    /// already-consumed payload — a sync_file that never signals while the
    /// queue stays healthy (the fourth manifestation of SYNC_FD fence-export
    /// fragility in this stack). Keeping the fence un-exported means
    /// `is_signaled`/`wait` always reflect true completion state, and
    /// `export` is an idempotent dup of this fd.
    sync_file: Mutex<Option<OwnedFd>>,
    submitted_native: AtomicBool,
    native_attempted: AtomicBool,
    retirement: Option<Box<super::retirement::RetirementNode<super::device_handle::DeviceRetirement>>>,
}

impl fmt::Debug for VulkanFence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanFence")
            .field("fence", &self.inner.fence)
            .field("has_sync_file", &self.inner.sync_file.lock().unwrap().is_some())
            .finish()
    }
}

impl VulkanFence {
    pub(crate) fn create(device: Arc<DeviceHandle>) -> Result<Self, VulkanRendererError> {
        if device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        let create_info = vk::FenceCreateInfo::default();

        // SAFETY: Device is valid and create info references live memory.
        let fence = device.observe_result(unsafe { device.handle().create_fence(&create_info, None) })?;

        Ok(Self {
            inner: VulkanFenceInner {
                device,
                fence,
                sync_file: Mutex::new(None),
                submitted_native: AtomicBool::new(false),
                native_attempted: AtomicBool::new(false),
                retirement: Some(super::retirement::RetirementNode::new(
                    super::device_handle::DeviceRetirement::Fence(fence),
                )),
            },
        })
    }

    pub(super) fn belongs_to(&self, device: &Arc<DeviceHandle>) -> bool {
        self.inner.submitted_native.load(Ordering::Acquire) && Arc::ptr_eq(&self.inner.device, device)
    }

    pub(super) fn native_attempted(&self) -> bool {
        self.inner.native_attempted.load(Ordering::Acquire)
    }
    pub(super) fn mark_native_attempt(&self) {
        self.inner.native_attempted.store(true, Ordering::Release);
    }
    pub(super) fn cancel_unsubmitted_attempt(&self) {
        self.inner.native_attempted.store(false, Ordering::Release);
    }

    pub(super) fn mark_submitted(&self) {
        self.inner.submitted_native.store(true, Ordering::Release);
    }

    pub(crate) fn handle(&self) -> vk::Fence {
        self.inner.fence
    }

    /// Attach the SYNC_FD exported from the submission's export semaphore.
    /// Called exactly once by the device right after a successful submit.
    pub(crate) fn set_exported_sync_file(&self, fd: OwnedFd) {
        let mut exported = self.inner.sync_file.lock().unwrap();
        if exported.is_some() {
            warn!("submission sync_file was already attached; ignoring duplicate");
        } else {
            *exported = Some(fd);
        }
    }

    pub(crate) fn status(&self) -> Result<bool, vk::Result> {
        // A fence on a lost device can't be meaningfully polled (the driver call faults/UB on
        // NVIDIA). A lost device will never signal again, so report it as signaled to keep
        // teardown and callers from blocking on a dead device.
        if self.inner.device.is_lost() {
            return Ok(true);
        }
        // SAFETY: Fence belongs to this device and remains valid while `self` is alive.
        self.inner
            .device
            .observe_result(unsafe { self.inner.device.handle().get_fence_status(self.inner.fence) })
    }

    pub(crate) fn wait_vk(&self) -> Result<(), vk::Result> {
        // A fence on a lost device must not be waited on (the driver call faults/UB on NVIDIA);
        // a lost device never signals, so treat the wait as immediately complete.
        if self.inner.device.is_lost() {
            return Ok(());
        }
        // SAFETY: Fence belongs to this device and remains valid while `self` is alive.
        self.inner.device.observe_result(unsafe {
            self.inner
                .device
                .handle()
                .wait_for_fences(&[self.inner.fence], true, u64::MAX)
        })
    }

    pub(crate) fn export_sync_file(&self) -> Option<OwnedFd> {
        let exported = self.inner.sync_file.lock().unwrap();
        let fd = exported.as_ref()?;
        match fd.as_fd().try_clone_to_owned() {
            Ok(fd) => Some(fd),
            Err(err) => {
                warn!(?err, "failed to dup submission sync_file fd");
                None
            }
        }
    }
}

impl Fence for VulkanFence {
    fn is_signaled(&self) -> bool {
        self.status().unwrap_or(false)
    }

    fn wait(&self) -> Result<(), Interrupted> {
        self.wait_vk().map_err(|_| Interrupted)
    }

    fn is_exportable(&self) -> bool {
        self.inner.sync_file.lock().unwrap().is_some()
    }

    fn export(&self) -> Option<OwnedFd> {
        self.export_sync_file()
    }
}

impl VulkanFence {
    /// Called only with the slot's sole logical reader and proven completion.
    pub(super) fn reset_for_reuse(&self) -> Result<(), VulkanRendererError> {
        self.inner
            .device
            .observe_result(unsafe { self.inner.device.handle().reset_fences(&[self.inner.fence]) })?;
        self.inner.sync_file.lock().unwrap().take();
        self.inner.submitted_native.store(false, Ordering::Release);
        self.inner.native_attempted.store(false, Ordering::Release);
        Ok(())
    }
}
impl Drop for VulkanFenceInner {
    fn drop(&mut self) {
        if let Some(node) = self.retirement.take() {
            self.device.retire_resource(node);
        }
    }
}

pub(crate) fn import_sync_file_to_fence(
    device: &DeviceHandle,
    external_fence_fd: &khr::external_fence_fd::Device,
    sync_file: OwnedFd,
) -> Result<vk::Fence, VulkanRendererError> {
    let fence_info = vk::FenceCreateInfo::default();

    // SAFETY: Device is valid and create info references no borrowed memory.
    let fence = match device.observe_result(unsafe { device.handle().create_fence(&fence_info, None) }) {
        Ok(fence) => fence,
        Err(err) => return Err(err.into()),
    };

    let sync_file_raw = std::os::fd::IntoRawFd::into_raw_fd(sync_file);
    // SAFETY: `fd` is owned by this guard and closed exactly once on early-return paths.
    let sync_file_guard = scopeguard::guard(sync_file_raw, |fd| unsafe {
        libc::close(fd);
    });

    let import_info = vk::ImportFenceFdInfoKHR::default()
        .fence(fence)
        .flags(vk::FenceImportFlags::TEMPORARY)
        .handle_type(vk::ExternalFenceHandleTypeFlags::SYNC_FD)
        .fd(*sync_file_guard);

    // SAFETY: Fence and device are valid and import info references live memory.
    if let Err(err) = device.observe_result(unsafe { external_fence_fd.import_fence_fd(&import_info) }) {
        device.destroy_with(|device| unsafe { device.destroy_fence(fence, None) });
        return Err(err.into());
    }

    // Ownership moved to Vulkan on successful import.
    let _ = scopeguard::ScopeGuard::into_inner(sync_file_guard);

    Ok(fence)
}

pub(crate) fn import_sync_file_to_semaphore(
    device: &DeviceHandle,
    external_semaphore_fd: &khr::external_semaphore_fd::Device,
    sync_file: OwnedFd,
) -> Result<vk::Semaphore, VulkanRendererError> {
    let semaphore_info = vk::SemaphoreCreateInfo::default();

    // SAFETY: Device is valid and create info references no borrowed memory.
    let semaphore =
        match device.observe_result(unsafe { device.handle().create_semaphore(&semaphore_info, None) }) {
            Ok(semaphore) => semaphore,
            Err(err) => return Err(err.into()),
        };

    if let Err(error) = import_sync_file_into_semaphore(device, external_semaphore_fd, semaphore, sync_file) {
        device.destroy_with(|device| unsafe { device.destroy_semaphore(semaphore, None) });
        return Err(error);
    }
    Ok(semaphore)
}

/// Import into a cold-owned binary semaphore. A failed import retains the
/// old payload and closes the exact untransferred FD. A successful TEMPORARY
/// import replaces any previous temporary payload and transfers the FD.
pub(crate) fn import_sync_file_into_semaphore(
    device: &DeviceHandle,
    external_semaphore_fd: &khr::external_semaphore_fd::Device,
    semaphore: vk::Semaphore,
    sync_file: OwnedFd,
) -> Result<(), VulkanRendererError> {
    let sync_file_raw = std::os::fd::IntoRawFd::into_raw_fd(sync_file);
    // SAFETY: `fd` is owned by this guard and closed exactly once on early-return paths.
    let sync_file_guard = scopeguard::guard(sync_file_raw, |fd| unsafe {
        libc::close(fd);
    });

    let import_info = vk::ImportSemaphoreFdInfoKHR::default()
        .semaphore(semaphore)
        .flags(vk::SemaphoreImportFlags::TEMPORARY)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
        .fd(*sync_file_guard);

    // SAFETY: Semaphore and device are valid and import info references live memory.
    if let Err(err) =
        device.observe_result(unsafe { external_semaphore_fd.import_semaphore_fd(&import_info) })
    {
        return Err(err.into());
    }

    // Ownership moved to Vulkan on successful import.
    let _ = scopeguard::ScopeGuard::into_inner(sync_file_guard);

    Ok(())
}
