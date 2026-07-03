use std::{
    fmt,
    os::fd::{AsFd, OwnedFd},
    sync::{Arc, OnceLock},
};

use ash::{khr, vk};
use tracing::warn;

use crate::backend::renderer::sync::{Fence, Interrupted};

use super::{device::DeviceHandle, VulkanRendererError};

#[derive(Clone)]
pub(crate) struct VulkanFence {
    inner: Arc<VulkanFenceInner>,
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
    sync_file: OnceLock<Arc<OwnedFd>>,
}

impl fmt::Debug for VulkanFence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanFence")
            .field("fence", &self.inner.fence)
            .field("has_sync_file", &self.inner.sync_file.get().is_some())
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
            inner: Arc::new(VulkanFenceInner {
                device,
                fence,
                sync_file: OnceLock::new(),
            }),
        })
    }

    pub(crate) fn handle(&self) -> vk::Fence {
        self.inner.fence
    }

    /// Attach the SYNC_FD exported from the submission's export semaphore.
    /// Called exactly once by the device right after a successful submit.
    pub(crate) fn set_exported_sync_file(&self, fd: OwnedFd) {
        if self.inner.sync_file.set(Arc::new(fd)).is_err() {
            warn!("submission sync_file was already attached; ignoring duplicate");
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
        let fd = self.inner.sync_file.get()?;
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
        self.inner.sync_file.get().is_some()
    }

    fn export(&self) -> Option<OwnedFd> {
        self.export_sync_file()
    }
}

impl Drop for VulkanFenceInner {
    fn drop(&mut self) {
        // SAFETY: Fence belongs to this device and is only destroyed once when the final owner drops.
        // Skipped on a lost device: destroying a fence on a lost VkDevice faults on NVIDIA
        // (destroy_fence → libnvidia-eglcore SIGSEGV — the device-loss teardown crash this guards).
        // `destroy_with` is the single ownership-encoded teardown gate; a no-op when lost.
        self.device
            .destroy_with(|device| unsafe { device.destroy_fence(self.fence, None) });
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
        device.destroy_with(|device| unsafe { device.destroy_semaphore(semaphore, None) });
        return Err(err.into());
    }

    // Ownership moved to Vulkan on successful import.
    let _ = scopeguard::ScopeGuard::into_inner(sync_file_guard);

    Ok(semaphore)
}
