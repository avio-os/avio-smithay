//! Numeric-only access to the existing successful first-import counter.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
struct Cell {
    id: u64,
    total: AtomicU64,
    complete: AtomicBool,
}

#[derive(Debug)]
pub(super) struct ClientImportCounter(Arc<Cell>);

impl Default for ClientImportCounter {
    fn default() -> Self {
        let id = NEXT_ID.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1));
        Self(Arc::new(Cell {
            id: id.unwrap_or(0),
            total: AtomicU64::new(0),
            complete: AtomicBool::new(id.is_ok()),
        }))
    }
}

impl ClientImportCounter {
    pub(super) fn created(&self) {
        if self
            .0
            .total
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .is_err()
        {
            self.0.complete.store(false, Ordering::Release);
        }
    }
    pub(super) fn total(&self) -> u64 {
        self.0.total.load(Ordering::Relaxed)
    }
    pub(super) fn observer(&self) -> VulkanClientImportObserver {
        VulkanClientImportObserver(Arc::downgrade(&self.0))
    }
}

/// Weak counter access. This never retains a renderer, image or device.
#[derive(Debug, Clone)]
pub struct VulkanClientImportObserver(Weak<Cell>);

impl VulkanClientImportObserver {
    /// Exact owner-lifetime identity and successful first Frame imports.
    /// Retired owners and overflow are unavailable, never zero.
    pub fn snapshot(&self) -> Option<(u64, u64)> {
        let cell = self.0.upgrade()?;
        let total = cell.total.load(Ordering::Relaxed);
        cell.complete.load(Ordering::Acquire).then_some((cell.id, total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weak_observer_reads_the_same_counter_and_cannot_retain_its_owner() {
        let owner = ClientImportCounter::default();
        let reader = owner.observer();
        let id = reader.snapshot().unwrap().0;
        owner.created();
        assert_eq!(owner.total(), 1);
        assert_eq!(reader.snapshot(), Some((id, 1)));
        drop(owner);
        assert_eq!(reader.snapshot(), None);
    }
    #[test]
    fn overflow_does_not_wrap_or_claim_a_complete_count() {
        let owner = ClientImportCounter::default();
        owner.0.total.store(u64::MAX, Ordering::Relaxed);
        owner.created();
        assert_eq!(owner.total(), u64::MAX);
        assert_eq!(owner.observer().snapshot(), None);
    }
}
