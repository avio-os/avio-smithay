//! Cold limits for context-local command recording and exact native readers.
//! No submitted owner can migrate to another slot before native completion.
use super::{
    device::{DeviceHandle, InFlightSubmission},
    fence_return::{FenceReaderReturn, FenceReaders},
    sync::VulkanFence,
    VulkanRendererError,
};
use crate::backend::renderer::sync::SyncPoint;
use ash::vk;
use std::sync::Arc;

/// CPU storage limits allocated when the renderer context is created.
/// These bounds are independent of GPU image and upload-storage budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanCommandStorageLimits {
    /// Maximum simultaneous native submissions or externally held fence epochs.
    pub submission_slots: usize,
    /// Maximum distinct retained images in one submitted command scope.
    pub images_per_submission: usize,
    /// Maximum temporary framebuffers in one submitted command scope.
    pub framebuffers_per_submission: usize,
    /// Maximum imported waits in one command scope.
    pub waits_per_submission: usize,
}
impl Default for VulkanCommandStorageLimits {
    fn default() -> Self {
        Self {
            submission_slots: 64,
            images_per_submission: 1024,
            framebuffers_per_submission: 1024,
            waits_per_submission: 256,
        }
    }
}
impl VulkanCommandStorageLimits {
    pub(super) fn validate(self) -> Result<Self, VulkanRendererError> {
        if self.submission_slots == 0
            || self.images_per_submission < 258
            || self.framebuffers_per_submission == 0
            || self.waits_per_submission == 0
            || self.waits_per_submission > 256
            || self.submission_slots > 256
            || self.images_per_submission > 65536
            || self.framebuffers_per_submission > 65536
        {
            return Err(VulkanRendererError::TemporaryFailure(
                "invalid cold command storage limits",
            ));
        }
        Ok(self)
    }
    pub(super) fn command_buffers(self) -> usize {
        self.submission_slots * 2 + 2
    }
}
impl InFlightSubmission {
    pub(super) fn cold(
        device: Arc<DeviceHandle>,
        export_semaphore: Option<vk::Semaphore>,
        limits: VulkanCommandStorageLimits,
    ) -> Result<Self, VulkanRendererError> {
        let readers = Arc::new(FenceReaders::default());
        let native = Arc::new(VulkanFence::create(device)?);
        let fence = SyncPoint::from_shared_fence_with_owner(native.clone(), readers.clone());
        let reader_return = SyncPoint::from_shared_fence(Arc::new(FenceReaderReturn(readers.clone())));
        Ok(Self {
            id: super::device::SubmissionId(0),
            submission_id_known: false,
            fence,
            native,
            readers,
            reader_return,
            export_semaphore,
            export_unconsumed: false,
            command_buffers: Vec::with_capacity(2),
            framebuffers: Vec::with_capacity(limits.framebuffers_per_submission),
            retained_images: Vec::with_capacity(limits.images_per_submission),
            upload_sources: Vec::with_capacity(256),
            _readback: None,
            wait_semaphores: Vec::with_capacity(limits.waits_per_submission),
            submitted_at: std::time::Instant::now(),
        })
    }
    pub(super) fn prepare_for_submit(&self) -> Result<(), VulkanRendererError> {
        let Some(_epoch) = self.readers.try_native_epoch() else {
            return Err(VulkanRendererError::CommandCapacityExhausted { slots: 1 });
        };
        if !self.readers.count_returned() {
            return Err(VulkanRendererError::CommandCapacityExhausted { slots: 1 });
        }
        self.native_fence().reset_for_reuse()
    }
    pub(super) fn native_fence(&self) -> &VulkanFence {
        &self.native
    }
    pub(super) fn readers_returned(&self) -> bool {
        self.readers.returned()
    }
}

impl Drop for InFlightSubmission {
    fn drop(&mut self) {
        self.readers.close_slot();
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        device_handle::retirement_tests::{device_with_wait_result, drain_until_parent, image},
        storage_heap_probe::measure,
    };
    use super::*;
    use std::sync::atomic::{AtomicI32, Ordering};
    #[test]
    fn actual_native_device_loss_is_preserved_but_unknown_status_has_no_loss_proof() {
        for (result, known_loss) in [
            (vk::Result::ERROR_OUT_OF_HOST_MEMORY, false),
            (vk::Result::ERROR_DEVICE_LOST, true),
        ] {
            let status = Arc::new(AtomicI32::new(result.as_raw()));
            let (device, events) = device_with_wait_result(status);
            let slot = InFlightSubmission::cold(device.clone(), None, Default::default()).unwrap();
            let raw = slot.native_fence().status().unwrap_err();
            let error = VulkanRendererError::from(raw).completion_failure();
            assert_eq!(device.is_lost(), known_loss);
            assert_eq!(error.is_device_lost(), known_loss);
            assert_eq!(error.is_command_completion_unavailable(), !known_loss);
            assert!(!error.is_command_deferred());
            assert!(
                events.try_recv().is_err(),
                "failed observation releases no native owner"
            );
            drop(slot);
            drop(device);
            drain_until_parent(&events);
        }
    }
    #[test]
    fn cold_native_slot_returns_no_epoch_while_external_reader_remains() {
        let wait = Arc::new(AtomicI32::new(vk::Result::SUCCESS.as_raw()));
        let (device, events) = device_with_wait_result(wait);
        let slot = InFlightSubmission::cold(device, None, Default::default()).unwrap();
        let external = slot.fence.clone();
        assert!(!slot.readers_returned());
        assert!(!slot.reader_return.is_reached());
        drop(external);
        assert!(slot.readers_returned());
        assert!(slot.reader_return.is_reached());
        drop(slot);
        drain_until_parent(&events);
    }
    #[test]
    fn failed_native_wait_keeps_exact_slot_image_reader_until_real_success() {
        let wait = Arc::new(AtomicI32::new(vk::Result::ERROR_OUT_OF_HOST_MEMORY.as_raw()));
        let (device, events) = device_with_wait_result(wait.clone());
        let source = image(device.clone(), 910);
        let weak = Arc::downgrade(&source);
        let mut slot = InFlightSubmission::cold(device, None, Default::default()).unwrap();
        slot.retained_images.push(source);
        assert_eq!(
            slot.native_fence().wait_vk(),
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        );
        assert!(weak.upgrade().is_some());
        assert_eq!(slot.retained_images.len(), 1);
        wait.store(vk::Result::SUCCESS.as_raw(), Ordering::Release);
        slot.native_fence().wait_vk().unwrap();
        slot.retained_images.clear();
        assert!(weak.upgrade().is_none());
        drop(slot);
        drain_until_parent(&events);
    }
    #[test]
    fn reused_cold_submission_vectors_never_grow_or_free_on_warm_operations() {
        let wait = Arc::new(AtomicI32::new(vk::Result::SUCCESS.as_raw()));
        let (device, events) = device_with_wait_result(wait);
        let image = image(device.clone(), 911);
        let mut slot = InFlightSubmission::cold(device, None, Default::default()).unwrap();
        let (_, operations) = measure(|| {
            for _ in 0..4096 {
                slot.command_buffers.push(vk::CommandBuffer::null());
                slot.framebuffers.push(vk::Framebuffer::null());
                slot.retained_images.push(image.clone());
                slot.wait_semaphores
                    .push(super::super::external_wait_storage::ImportedWaitSemaphore {
                        handle: vk::Semaphore::null(),
                        pooled: false,
                        index: 0,
                    });
                let reader = slot.fence.clone();
                assert!(!slot.readers_returned());
                drop(reader);
                assert!(slot.readers_returned());
                slot.command_buffers.clear();
                slot.framebuffers.clear();
                slot.retained_images.clear();
                slot.wait_semaphores.clear();
            }
        });
        assert_eq!(operations, [0; 4]);
        drop(slot);
        drop(image);
        drain_until_parent(&events);
    }
    #[test]
    fn slot_rejection_never_resets_a_native_epoch_still_read_by_the_caller() {
        use super::super::device_handle::retirement_tests::{next, Operation};
        let wait = Arc::new(AtomicI32::new(vk::Result::SUCCESS.as_raw()));
        let (device, events) = device_with_wait_result(wait);
        let slot = InFlightSubmission::cold(device, None, Default::default()).unwrap();
        let external = slot.fence.clone();
        assert!(slot.prepare_for_submit().unwrap_err().is_command_deferred());
        assert!(
            events.try_recv().is_err(),
            "no native reset while the old epoch is borrowed"
        );
        drop(external);
        slot.prepare_for_submit().unwrap();
        assert!(matches!(next(&events).0, Operation::FenceReset(_)));
        drop(slot);
        drain_until_parent(&events);
    }
}
