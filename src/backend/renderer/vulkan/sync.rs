use std::{
    fmt,
    os::fd::{FromRawFd, OwnedFd},
    sync::Arc,
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
    external_fence_fd: Option<Arc<khr::external_fence_fd::Device>>,
    exportable_sync_file: bool,
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
            .finish()
    }
}

impl VulkanFence {
    pub(crate) fn create(
        device: Arc<DeviceHandle>,
        external_fence_fd: Option<Arc<khr::external_fence_fd::Device>>,
        exportable_sync_file: bool,
    ) -> Result<Self, VulkanRendererError> {
        let mut export_info;
        let mut create_info = vk::FenceCreateInfo::default();

        if exportable_sync_file && external_fence_fd.is_some() {
            export_info =
                vk::ExportFenceCreateInfo::default().handle_types(vk::ExternalFenceHandleTypeFlags::SYNC_FD);
            create_info = create_info.push_next(&mut export_info);
        }

        // SAFETY: Device is valid and create info references live memory.
        let fence = unsafe { device.handle().create_fence(&create_info, None) }?;

        Ok(Self {
            inner: Arc::new(VulkanFenceInner {
                device,
                fence,
                external_fence_fd,
                exportable_sync_file,
            }),
        })
    }

    pub(crate) fn handle(&self) -> vk::Fence {
        self.inner.fence
    }

    pub(crate) fn status(&self) -> Result<bool, vk::Result> {
        // SAFETY: Fence belongs to this device and remains valid while `self` is alive.
        unsafe { self.inner.device.handle().get_fence_status(self.inner.fence) }
    }

    pub(crate) fn wait_vk(&self) -> Result<(), vk::Result> {
        // SAFETY: Fence belongs to this device and remains valid while `self` is alive.
        unsafe {
            self.inner
                .device
                .handle()
                .wait_for_fences(&[self.inner.fence], true, u64::MAX)
        }
    }

    pub(crate) fn export_sync_file(&self) -> Option<OwnedFd> {
        if !self.inner.exportable_sync_file {
            return None;
        }

        let loader = self.inner.external_fence_fd.as_ref()?;
        let get_info = vk::FenceGetFdInfoKHR::default()
            .fence(self.inner.fence)
            .handle_type(vk::ExternalFenceHandleTypeFlags::SYNC_FD);

        // SAFETY: Fence belongs to the same device as `loader` and remains valid for this call.
        let fd = match unsafe { loader.get_fence_fd(&get_info) } {
            Ok(fd) => fd,
            Err(err) => {
                warn!(?err, "failed to export Vulkan fence as sync_file fd");
                return None;
            }
        };

        // SAFETY: Vulkan returns ownership of a valid fd on success.
        Some(unsafe { OwnedFd::from_raw_fd(fd) })
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
        self.inner.exportable_sync_file && self.inner.external_fence_fd.is_some()
    }

    fn export(&self) -> Option<OwnedFd> {
        self.export_sync_file()
    }
}

impl Drop for VulkanFenceInner {
    fn drop(&mut self) {
        // SAFETY: Fence belongs to this device and is only destroyed once when the final owner drops.
        unsafe { self.device.handle().destroy_fence(self.fence, None) };
    }
}

pub(crate) fn import_sync_file_to_fence(
    device: &ash::Device,
    external_fence_fd: &khr::external_fence_fd::Device,
    sync_file: OwnedFd,
) -> Result<vk::Fence, VulkanRendererError> {
    let fence_info = vk::FenceCreateInfo::default();

    // SAFETY: Device is valid and create info references no borrowed memory.
    let fence = match unsafe { device.create_fence(&fence_info, None) } {
        Ok(fence) => fence,
        Err(err) => return Err(err.into()),
    };

    let sync_file_raw = std::os::fd::IntoRawFd::into_raw_fd(sync_file);
    let sync_file_guard = scopeguard::guard(sync_file_raw, |fd| unsafe {
        libc::close(fd);
    });

    let import_info = vk::ImportFenceFdInfoKHR::default()
        .fence(fence)
        .flags(vk::FenceImportFlags::TEMPORARY)
        .handle_type(vk::ExternalFenceHandleTypeFlags::SYNC_FD)
        .fd(*sync_file_guard);

    // SAFETY: Fence and device are valid and import info references live memory.
    if let Err(err) = unsafe { external_fence_fd.import_fence_fd(&import_info) } {
        unsafe { device.destroy_fence(fence, None) };
        return Err(err.into());
    }

    // Ownership moved to Vulkan on successful import.
    let _ = scopeguard::ScopeGuard::into_inner(sync_file_guard);

    Ok(fence)
}
