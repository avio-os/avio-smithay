//! Opaque, exact-origin command storage prepared by an untagged cold owner.
use super::{
    device::{DeviceHandle, InFlightSubmission, RetiredCommands},
    retirement::RetirementNode,
    VulkanCommandStorageLimits, VulkanDeviceOrigin, VulkanRendererError,
};
use ash::vk;
use std::{fmt, sync::Arc};

/// Native command/fence and CPU workspace bank, prepared before adoption.
/// Dropping an unused token returns its existing node to the owning actor.
pub struct VulkanCommandStorage {
    pub(super) node: Option<Box<RetirementNode<RetiredCommands>>>,
    pub(super) device: Arc<DeviceHandle>,
    pub(super) family: u32,
    pub(super) limits: VulkanCommandStorageLimits,
}
impl fmt::Debug for VulkanCommandStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanCommandStorage")
            .field("limits", &self.limits)
            .field("prepared", &self.node.is_some())
            .finish()
    }
}
impl VulkanCommandStorage {
    /// Exact CPU/native slot bounds represented by this immutable token.
    pub fn limits(&self) -> VulkanCommandStorageLimits {
        self.limits
    }
    pub(super) fn prepare(
        device: Arc<DeviceHandle>,
        family: u32,
        exports: bool,
        limits: VulkanCommandStorageLimits,
    ) -> Result<Self, VulkanRendererError> {
        let limits = limits.validate()?;
        if device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = device.observe_result(unsafe { device.handle().create_command_pool(&info, None) })?;
        let mut commands = RetiredCommands::cold_cpu_storage(pool, limits);
        commands.device = Some(device.clone());
        let mut token = Self {
            node: Some(RetirementNode::new(commands)),
            device,
            family,
            limits,
        };
        let commands = token.node.as_mut().unwrap().value_mut();
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(limits.command_buffers() as u32);
        commands.reusable_command_buffers = token
            .device
            .observe_result(unsafe { token.device.handle().allocate_command_buffers(&info) })?;
        for _ in 0..limits.submission_slots {
            let semaphore = if exports {
                let mut export = vk::ExportSemaphoreCreateInfo::default()
                    .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
                let info = vk::SemaphoreCreateInfo::default().push_next(&mut export);
                Some(
                    token
                        .device
                        .observe_result(unsafe { token.device.handle().create_semaphore(&info, None) })?,
                )
            } else {
                None
            };
            match InFlightSubmission::cold(token.device.clone(), semaphore, limits) {
                Ok(slot) => commands.free_submissions.push(slot),
                Err(error) => {
                    if let Some(semaphore) = semaphore {
                        token
                            .device
                            .destroy_with(|raw| unsafe { raw.destroy_semaphore(semaphore, None) });
                    }
                    return Err(error);
                }
            }
        }
        commands.bank_return = Some(super::bank_return::cold_bank_return(&commands.free_submissions));
        Ok(token)
    }
}
impl Drop for VulkanCommandStorage {
    fn drop(&mut self) {
        if let Some(node) = self.node.take() {
            self.device.retire_commands(node);
        }
    }
}
impl VulkanDeviceOrigin {
    /// Allocate an exclusive command bank on this exact native device/queue
    /// family. Call only on the cold resource owner, before renderer adoption.
    pub fn prepare_command_storage(
        &self,
        limits: VulkanCommandStorageLimits,
    ) -> Result<VulkanCommandStorage, VulkanRendererError> {
        let _phase = self.allocation_phase_scope(super::VulkanAllocationPhase::Warmup);
        VulkanCommandStorage::prepare(
            self.device.clone(),
            self.queue_family_index,
            self.capabilities.sync_file_semaphore_export(),
            limits,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::device_handle::retirement_tests::{device, next, Operation};
    use super::*;
    #[test]
    fn unused_cold_bank_returns_its_native_pool_and_fences_on_existing_actor() {
        let (device, events) = device();
        let caller = std::thread::current().id();
        let limits = VulkanCommandStorageLimits {
            submission_slots: 2,
            images_per_submission: 259,
            framebuffers_per_submission: 2,
            waits_per_submission: 2,
        };
        let bank = VulkanCommandStorage::prepare(device.clone(), 3, false, limits).unwrap();
        assert_eq!(bank.limits(), limits);
        let created = next(&events).0;
        assert!(matches!(created, Operation::CommandPoolCreated(_)));
        assert_eq!(next(&events).0, Operation::CommandBuffersAllocated(6));
        let native = bank.node.as_ref().unwrap().value();
        assert_eq!(native.free_submissions.len(), 2);
        assert_eq!(native.reusable_command_buffers.len(), 6);
        assert_ne!(
            native.free_submissions[0].native_fence().handle(),
            native.free_submissions[1].native_fence().handle()
        );
        drop(bank);
        drop(device);
        let destroyed = next(&events);
        assert!(matches!(destroyed.0, Operation::CommandPool(_)));
        assert_ne!(destroyed.1, caller);
        for _ in 0..2 {
            let fence = next(&events);
            assert!(matches!(fence.0, Operation::Fence(_)));
            assert_ne!(fence.1, caller);
        }
        assert_eq!(next(&events).0, Operation::Device);
        assert_eq!(next(&events).0, Operation::Parent);
    }
    #[test]
    fn invalid_cold_limit_creates_no_native_object() {
        let (device, events) = device();
        let limits = VulkanCommandStorageLimits {
            submission_slots: 0,
            ..Default::default()
        };
        assert!(VulkanCommandStorage::prepare(device.clone(), 3, false, limits).is_err());
        assert!(events.try_recv().is_err());
        drop(device);
        assert_eq!(next(&events).0, Operation::Device);
        assert_eq!(next(&events).0, Operation::Parent);
    }
}
