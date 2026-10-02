//! One exclusive, cold-allocated CPU workspace for a mutable renderer frame.
//! The atomic handoff only transfers a previously allocated Box; its capacity
//! survives every segment, successful submission, and rejected recording.
use super::{image::VulkanImage, submission_storage::VulkanCommandStorageLimits, VulkanRendererError};
use ash::vk;
use indexmap::IndexMap;
use std::{
    fmt,
    ops::{Deref, DerefMut},
    ptr,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc,
    },
};

#[derive(Debug)]
pub(super) struct RecordingStorage {
    pub(super) pending_layouts: IndexMap<u64, (Arc<VulkanImage>, vk::ImageLayout)>,
    pub(super) foreign_release_images: IndexMap<u64, Arc<VulkanImage>>,
    pub(super) unsubmitted_foreign_acquires: IndexMap<u64, Arc<VulkanImage>>,
    pub(super) effect_framebuffers: Vec<vk::Framebuffer>,
    pub(super) submitted_framebuffers: Vec<vk::Framebuffer>,
    pub(super) retained_images: Vec<Arc<VulkanImage>>,
    pub(super) blit_steps: Vec<super::blit::ResolvedBlitChainStep>,
    pub(super) blit_layouts: IndexMap<u64, super::blit::TrackedBlitImageLayout>,
    pub(super) resolved_kawase: Vec<super::kawase::ResolvedKawasePass>,
    pub(super) barriers: Vec<vk::ImageMemoryBarrier<'static>>,
    pub(super) image_limit: usize,
    pub(super) framebuffer_limit: usize,
}
impl RecordingStorage {
    fn new(limits: VulkanCommandStorageLimits) -> Self {
        // Reserve upload readers and the parent separately in each submission.
        let image_limit = limits.images_per_submission - 257;
        Self {
            pending_layouts: IndexMap::with_capacity(image_limit),
            foreign_release_images: IndexMap::with_capacity(image_limit),
            unsubmitted_foreign_acquires: IndexMap::with_capacity(image_limit),
            effect_framebuffers: Vec::with_capacity(limits.framebuffers_per_submission),
            submitted_framebuffers: Vec::with_capacity(limits.framebuffers_per_submission),
            retained_images: Vec::with_capacity(limits.images_per_submission),
            blit_steps: Vec::with_capacity(image_limit / 2),
            blit_layouts: IndexMap::with_capacity(image_limit),
            resolved_kawase: Vec::with_capacity(limits.framebuffers_per_submission - 1),
            barriers: Vec::with_capacity(image_limit),
            image_limit,
            framebuffer_limit: limits.framebuffers_per_submission,
        }
    }
    fn clear(&mut self) {
        self.pending_layouts.clear();
        self.foreign_release_images.clear();
        self.unsubmitted_foreign_acquires.clear();
        self.effect_framebuffers.clear();
        self.submitted_framebuffers.clear();
        self.retained_images.clear();
        self.barriers.clear();
        self.resolved_kawase.clear();
        self.blit_steps.clear();
        self.blit_layouts.clear();
    }
    pub(super) fn admit_image(&self, id: u64) -> Result<(), VulkanRendererError> {
        if !self.pending_layouts.contains_key(&id) && self.pending_layouts.len() >= self.image_limit {
            return Err(VulkanRendererError::CommandStorageLimitExceeded {
                resource: "recording images",
                requested: self.pending_layouts.len() + 1,
                limit: self.image_limit,
            });
        }
        Ok(())
    }
    pub(super) fn admit_framebuffer(&self) -> Result<(), VulkanRendererError> {
        let requested = self.effect_framebuffers.len() + 2;
        if requested > self.framebuffer_limit {
            return Err(VulkanRendererError::CommandStorageLimitExceeded {
                resource: "recording framebuffers",
                requested,
                limit: self.framebuffer_limit,
            });
        }
        Ok(())
    }
}

pub(super) struct RecordingStorageBank {
    storage: AtomicPtr<RecordingStorage>,
}
impl fmt::Debug for RecordingStorageBank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecordingStorageBank")
            .field("available", &(!self.storage.load(Ordering::Acquire).is_null()))
            .finish()
    }
}
// Every pointer transfer is unique. Only the owning frame accesses its Box;
// the bank and its destructor access storage after the frame returns it.
unsafe impl Send for RecordingStorageBank {}
unsafe impl Sync for RecordingStorageBank {}
impl RecordingStorageBank {
    pub(super) fn cold(limits: VulkanCommandStorageLimits) -> Arc<Self> {
        Arc::new(Self {
            storage: AtomicPtr::new(Box::into_raw(Box::new(RecordingStorage::new(limits)))),
        })
    }
    pub(super) fn acquire(self: &Arc<Self>) -> Result<RecordingStorageLease, VulkanRendererError> {
        let ptr = self.storage.swap(ptr::null_mut(), Ordering::AcqRel);
        if ptr.is_null() {
            return Err(VulkanRendererError::TemporaryFailure(
                "exclusive recording storage is already in use",
            ));
        }
        Ok(RecordingStorageLease {
            storage: Some(unsafe { Box::from_raw(ptr) }),
            bank: self.clone(),
        })
    }
}
impl Drop for RecordingStorageBank {
    fn drop(&mut self) {
        let ptr = *self.storage.get_mut();
        if !ptr.is_null() {
            drop(unsafe { Box::from_raw(ptr) });
        }
    }
}
#[derive(Debug)]
pub(super) struct RecordingStorageLease {
    storage: Option<Box<RecordingStorage>>,
    bank: Arc<RecordingStorageBank>,
}
impl Deref for RecordingStorageLease {
    type Target = RecordingStorage;
    fn deref(&self) -> &RecordingStorage {
        self.storage.as_ref().unwrap()
    }
}
impl DerefMut for RecordingStorageLease {
    fn deref_mut(&mut self) -> &mut RecordingStorage {
        self.storage.as_mut().unwrap()
    }
}
impl Drop for RecordingStorageLease {
    fn drop(&mut self) {
        if let Some(mut storage) = self.storage.take() {
            storage.clear();
            let ptr = Box::into_raw(storage);
            let prior = self.bank.storage.swap(ptr, Ordering::AcqRel);
            assert!(prior.is_null(), "exclusive recording storage returned twice");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        device_handle::retirement_tests::{device, drain_until_parent, image},
        storage_heap_probe::measure,
    };
    use super::*;
    #[test]
    fn warm_recording_maps_and_vectors_return_without_allocator_operations() {
        let (device, events) = device();
        let image = image(device.clone(), 900);
        let bank = RecordingStorageBank::cold(VulkanCommandStorageLimits::default());
        let (_, operations) = measure(|| {
            for _ in 0..4096 {
                let mut lease = bank.acquire().unwrap();
                lease.admit_image(image.id()).unwrap();
                lease
                    .pending_layouts
                    .insert(image.id(), (image.clone(), vk::ImageLayout::GENERAL));
                lease.foreign_release_images.insert(image.id(), image.clone());
                lease
                    .unsubmitted_foreign_acquires
                    .insert(image.id(), image.clone());
                lease.effect_framebuffers.push(vk::Framebuffer::null());
                lease.retained_images.push(image.clone());
                lease.barriers.push(vk::ImageMemoryBarrier::default());
                drop(lease);
            }
        });
        assert_eq!(
            operations, [0; 4],
            "alloc/zero/realloc/free on actual warm storage path"
        );
        assert!(bank.acquire().unwrap().pending_layouts.is_empty());
        drop(bank);
        drop(image);
        drop(device);
        drain_until_parent(&events);
    }
    #[test]
    fn exclusive_cold_storage_refuses_alias_and_preserves_capacity_after_rejection() {
        let bank = RecordingStorageBank::cold(VulkanCommandStorageLimits::default());
        let lease = bank.acquire().unwrap();
        assert!(bank.acquire().is_err());
        let capacity = lease.pending_layouts.capacity();
        drop(lease);
        assert_eq!(bank.acquire().unwrap().pending_layouts.capacity(), capacity);
    }
    #[test]
    fn bounds_are_checked_before_another_image_or_framebuffer_is_adopted() {
        let (device, events) = device();
        let image = image(device, 901);
        let limits = VulkanCommandStorageLimits {
            images_per_submission: 258,
            framebuffers_per_submission: 1,
            ..Default::default()
        };
        let bank = RecordingStorageBank::cold(limits);
        let mut lease = bank.acquire().unwrap();
        lease
            .pending_layouts
            .insert(image.id(), (image.clone(), vk::ImageLayout::GENERAL));
        assert!(lease.admit_image(image.id()).is_ok());
        assert!(matches!(
            lease.admit_image(902),
            Err(VulkanRendererError::CommandStorageLimitExceeded { .. })
        ));
        assert!(lease.admit_framebuffer().is_err());
        assert_eq!(lease.pending_layouts.len(), 1);
        assert!(lease.effect_framebuffers.is_empty());
        drop(lease);
        drop(bank);
        drop(image);
        drain_until_parent(&events);
    }
}
