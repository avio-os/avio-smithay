//! GPU-owned logical-device origin; render contexts own independent command pools.
use super::{
    device::{DeviceCapabilities, DeviceState},
    device_handle::DeviceHandle,
};
use crate::backend::{
    allocator::vulkan::{Error, ImageUsageFlags, VulkanAllocator},
    vulkan::PhysicalDevice,
};
use ash::vk;
use std::{ffi::CStr, sync::Arc};

/// Immutable, cloneable ownership of one Vulkan device and its synchronized queue.
/// Construct this once per GPU and create every output renderer from it. Export
/// allocation uses that same device without sharing command pools or mutable caches.
#[derive(Clone, Debug)]
pub struct VulkanDeviceOrigin {
    pub(super) physical_device: PhysicalDevice,
    pub(super) enabled_extensions: Vec<&'static CStr>,
    pub(super) capabilities: DeviceCapabilities,
    pub(super) queue_family_index: u32,
    pub(super) queue: vk::Queue,
    pub(super) device: Arc<DeviceHandle>,
}

impl VulkanDeviceOrigin {
    pub(super) fn from_state(state: &DeviceState) -> Self {
        Self {
            physical_device: state.physical_device().clone(),
            enabled_extensions: state.enabled_extensions().to_vec(),
            capabilities: state.capabilities(),
            queue_family_index: state.queue_family_index(),
            queue: state.queue(),
            device: state.shared_device(),
        }
    }

    /// Create a modifier-aware export allocator on this origin's existing device.
    /// Allocation and export must run on a cold/helper turn; no queue is accessed.
    pub fn dmabuf_allocator(&self, usage: ImageUsageFlags) -> Result<VulkanAllocator, Error> {
        VulkanAllocator::from_renderer_device(
            &self.physical_device,
            self.device.clone(),
            &self.enabled_extensions,
            usage,
        )
    }

    /// Tag a cold resource preparation operation on the calling helper thread.
    pub fn allocation_phase_scope(
        &self,
        phase: super::VulkanAllocationPhase,
    ) -> super::VulkanAllocationPhaseGuard {
        self.device.allocation_ledger().enter_phase(phase)
    }

    /// Weak, allocation-only census provider; it cannot keep this device alive.
    pub fn allocation_observer(&self) -> super::VulkanAllocationObserver {
        super::VulkanAllocationObserver::new(self.device.allocation_ledger())
    }

    /// Reserve one allocation-free final-drop handoff on a cold owner turn.
    /// The resource protocol, not this slot, must prove that all readers ended.
    pub fn retirement_slot<T: Send + Sync + 'static>(&self) -> super::VulkanRetirementSlot<T> {
        super::VulkanRetirementSlot::new(self.device.clone())
    }

    /// Whether two contexts originate from exactly the same logical device.
    pub fn same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.device, &other.device)
    }
}
