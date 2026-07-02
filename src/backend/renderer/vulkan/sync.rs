use std::{
    fmt,
    os::fd::{FromRawFd, OwnedFd},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use ash::{khr, vk};
use tracing::{trace, warn};

use crate::backend::renderer::sync::{Fence, Interrupted};

use super::{device::DeviceHandle, VulkanRendererError};

#[derive(Clone)]
pub(crate) struct VulkanFence {
    inner: Arc<VulkanFenceInner>,
}

struct VulkanFenceInner {
    device: Arc<DeviceHandle>,
    fence: vk::Fence,
    external_fence_fd: Option<Arc<khr::external_fence_fd::Device>>,
    exportable_sync_file: bool,
    /// Set after a successful `vkGetFenceFdKHR(SYNC_FD)` call. SYNC_FD export
    /// has move/transfer semantics: the VkFence is reset to unsignaled after
    /// export, even when the returned fd is -1 (already-signaled sentinel).
    /// Once exported, the VkFence must not be waited on or re-exported.
    exported: AtomicBool,
}

impl fmt::Debug for VulkanFence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanFence")
            .field("fence", &self.inner.fence)
            .field("exportable_sync_file", &self.inner.exportable_sync_file)
            .finish()
    }
}

impl fmt::Debug for VulkanFenceInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanFenceInner")
            .field("device", &self.device.handle().handle())
            .field("fence", &self.fence)
            .field("exportable_sync_file", &self.exportable_sync_file)
            .field("exported", &self.exported.load(Ordering::Relaxed))
            .finish()
    }
}

impl VulkanFence {
    pub(crate) fn create(
        device: Arc<DeviceHandle>,
        external_fence_fd: Option<Arc<khr::external_fence_fd::Device>>,
        exportable_sync_file: bool,
    ) -> Result<Self, VulkanRendererError> {
        if device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        let mut export_info;
        let mut create_info = vk::FenceCreateInfo::default();

        if exportable_sync_file && external_fence_fd.is_some() {
            export_info =
                vk::ExportFenceCreateInfo::default().handle_types(vk::ExternalFenceHandleTypeFlags::SYNC_FD);
            create_info = create_info.push_next(&mut export_info);
        }

        // SAFETY: Device is valid and create info references live memory.
        let fence = device.observe_result(unsafe { device.handle().create_fence(&create_info, None) })?;

        Ok(Self {
            inner: Arc::new(VulkanFenceInner {
                device,
                fence,
                external_fence_fd,
                exportable_sync_file,
                exported: AtomicBool::new(false),
            }),
        })
    }

    pub(crate) fn handle(&self) -> vk::Fence {
        self.inner.fence
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
        if !self.inner.exportable_sync_file {
            return None;
        }

        // SYNC_FD export has move semantics — the VkFence is reset after export.
        // Prevent double-export which would wait/export on an undefined fence.
        if self.inner.exported.load(Ordering::Acquire) {
            return None;
        }

        let loader = self.inner.external_fence_fd.as_ref()?;
        let get_info = vk::FenceGetFdInfoKHR::default()
            .fence(self.inner.fence)
            .handle_type(vk::ExternalFenceHandleTypeFlags::SYNC_FD);

        // SAFETY: Fence belongs to the same device as `loader` and remains valid for this call.
        let fd = match self
            .inner
            .device
            .observe_result(unsafe { loader.get_fence_fd(&get_info) })
        {
            Ok(fd) => fd,
            Err(err) => {
                warn!(?err, "failed to export Vulkan fence as sync_file fd");
                return None;
            }
        };

        // Mark as exported BEFORE checking fd value. vkGetFenceFdKHR(SYNC_FD)
        // transfers the fence payload on success — the VkFence is now reset to
        // unsignaled regardless of whether fd is -1 or a valid fd.
        self.inner.exported.store(true, Ordering::Release);

        if fd == -1 {
            trace!("Vulkan fence export returned already-signaled sync_file sentinel (fence consumed)");
            return None;
        }

        if fd < -1 {
            warn!(
                fd,
                "Vulkan fence export returned invalid negative fd for sync_file handle"
            );
            return None;
        }

        // SAFETY: Vulkan returns ownership of a valid fd on success. `-1` is handled above.
        Some(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}

impl Fence for VulkanFence {
    fn is_signaled(&self) -> bool {
        // An exported SYNC_FD fence has been consumed — treat as signaled so
        // callers don't poll a reset fence that will never be signaled again.
        if self.inner.exported.load(Ordering::Acquire) {
            return true;
        }
        self.status().unwrap_or(false)
    }

    fn wait(&self) -> Result<(), Interrupted> {
        // An exported SYNC_FD fence has been consumed — the payload was moved
        // to the sync_file fd. Waiting on the reset VkFence would block forever.
        if self.inner.exported.load(Ordering::Acquire) {
            return Ok(());
        }
        self.wait_vk().map_err(|_| Interrupted)
    }

    fn is_exportable(&self) -> bool {
        if self.inner.exported.load(Ordering::Acquire) {
            return false;
        }
        self.inner.exportable_sync_file && self.inner.external_fence_fd.is_some()
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
