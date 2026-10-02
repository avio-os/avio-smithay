//! Cold rectangle loans for nested framebuffer readers. These CPU packets are
//! consumed synchronously; no GPU resource or completion identity is stored.
use super::VulkanRendererError;
use crate::utils::{Physical, Rectangle};
use std::{
    fmt,
    ops::{Deref, DerefMut},
    ptr,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc,
    },
};

type Rectangles = Vec<Rectangle<i32, Physical>>;

pub(super) struct DamageScratchBank {
    slots: Box<[AtomicPtr<Rectangles>]>,
    rectangles: usize,
}
impl fmt::Debug for DamageScratchBank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DamageScratchBank")
            .field("slots", &self.slots.len())
            .field("rectangles", &self.rectangles)
            .finish()
    }
}
impl DamageScratchBank {
    pub(super) fn bytes_per_rectangle(slots: usize) -> Option<usize> {
        slots.checked_mul(std::mem::size_of::<Rectangle<i32, Physical>>())
    }
    pub(super) fn cold(slots: usize, rectangles: usize) -> Result<Arc<Self>, VulkanRendererError> {
        let valid = Self::bytes_per_rectangle(slots)
            .and_then(|per_rectangle| per_rectangle.checked_mul(rectangles))
            .is_some_and(|bytes| bytes <= isize::MAX as usize);
        if !valid
            || slots
                .checked_mul(std::mem::size_of::<AtomicPtr<Rectangles>>())
                .is_none()
        {
            return Err(VulkanRendererError::TemporaryFailure(
                "damage scratch declaration overflow",
            ));
        }
        Ok(Arc::new(Self {
            slots: (0..slots)
                .map(|_| AtomicPtr::new(Box::into_raw(Box::new(Vec::with_capacity(rectangles)))))
                .collect(),
            rectangles,
        }))
    }
    pub(super) fn limits(&self) -> (usize, usize) {
        (self.slots.len(), self.rectangles)
    }
    pub(super) fn acquire(
        self: &Arc<Self>,
        required: usize,
    ) -> Result<DamageScratchLoan, VulkanRendererError> {
        if required > self.rectangles {
            return Err(VulkanRendererError::CommandStorageLimitExceeded {
                resource: "damage scratch rectangles",
                requested: required,
                limit: self.rectangles,
            });
        }
        for (index, slot) in self.slots.iter().enumerate() {
            let storage = slot.swap(ptr::null_mut(), Ordering::AcqRel);
            if !storage.is_null() {
                return Ok(DamageScratchLoan {
                    index,
                    bank: self.clone(),
                    storage: Some(unsafe { Box::from_raw(storage) }),
                });
            }
        }
        Err(VulkanRendererError::CommandStorageLimitExceeded {
            resource: "damage scratch nesting",
            requested: self.slots.len().saturating_add(1),
            limit: self.slots.len(),
        })
    }
}
impl Drop for DamageScratchBank {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            let storage = *slot.get_mut();
            if !storage.is_null() {
                drop(unsafe { Box::from_raw(storage) });
            }
        }
    }
}
#[derive(Debug)]
pub(super) struct DamageScratchLoan {
    index: usize,
    bank: Arc<DamageScratchBank>,
    storage: Option<Box<Rectangles>>,
}
impl Deref for DamageScratchLoan {
    type Target = Rectangles;
    fn deref(&self) -> &Self::Target {
        self.storage.as_ref().unwrap()
    }
}
impl DerefMut for DamageScratchLoan {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.storage.as_mut().unwrap()
    }
}
impl Drop for DamageScratchLoan {
    fn drop(&mut self) {
        if let Some(mut storage) = self.storage.take() {
            storage.clear();
            let prior = self.bank.slots[self.index].swap(Box::into_raw(storage), Ordering::AcqRel);
            assert!(prior.is_null(), "damage scratch returned to an occupied slot");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::storage_heap_probe::measure;
    use super::*;
    #[test]
    fn nested_batch_loans_preserve_outer_damage_and_return_after_error_or_panic() {
        let bank = DamageScratchBank::cold(2, 4).unwrap();
        let outer_rect = Rectangle::new((1, 2).into(), (3, 4).into());
        let mut outer = bank.acquire(4).unwrap();
        outer.push(outer_rect);
        let refusal = bank.acquire(5).unwrap_err();
        assert!(matches!(
            refusal,
            VulkanRendererError::CommandStorageLimitExceeded { .. }
        ));
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut inner = bank.acquire(2).unwrap();
            inner.push(Rectangle::from_size((2, 2).into()));
            assert!(bank.acquire(1).is_err());
            panic!("interrupted nested drawing");
        }))
        .unwrap_err();
        assert_eq!(outer.as_slice(), [outer_rect]);
        assert!(bank.acquire(4).unwrap().is_empty());
        drop(outer);
        assert!(bank.acquire(4).unwrap().is_empty());
    }
    #[test]
    fn warmed_nested_packet_loans_use_no_allocator_operation() {
        let bank = DamageScratchBank::cold(2, 16).unwrap();
        let (_, operations) = measure(|| {
            for _ in 0..4096 {
                let mut outer = bank.acquire(16).unwrap();
                outer.push(Rectangle::from_size((20, 20).into()));
                let mut inner = bank.acquire(16).unwrap();
                inner.extend_from_slice(&outer);
                assert!(bank.acquire(1).is_err());
                drop(inner);
                assert_eq!(outer.len(), 1);
                drop(outer);
            }
        });
        assert_eq!(operations, [0; 4]);
    }
    #[test]
    fn bounds_are_explicit_and_overflow_refused_before_storage_creation() {
        assert_eq!(DamageScratchBank::bytes_per_rectangle(2), Some(32));
        assert!(DamageScratchBank::cold(usize::MAX, 2).is_err());
        let bank = DamageScratchBank::cold(0, 0).unwrap();
        assert_eq!(bank.limits(), (0, 0));
        assert!(bank.acquire(0).is_err());
    }
}
