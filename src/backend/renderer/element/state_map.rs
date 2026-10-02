//! Returned state-map storage. Each receipt owns one whole map until its drop.
use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicBool, AtomicPtr, Ordering},
        Arc,
    },
};

use super::{Id, RenderElementState};

/// A bounded workspace was not cold-prepared for this operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("{resource} requires {required} entries, admitted capacity is {capacity}")]
pub struct FrameWorkspaceError {
    /// Owning storage which needs cold admission or an exact receipt release.
    pub resource: &'static str,
    /// Required entry count.
    pub required: usize,
    /// Currently admitted entry count.
    pub capacity: usize,
}

#[derive(Debug)]
struct StateMapBankInner {
    entries: usize,
    slots: Vec<AtomicPtr<HashMap<Id, RenderElementState>>>,
    returned: Option<StateReceiptReturnWakeup>,
}

/// Cold-registered exact receipt-return notification. The callback must only
/// signal/enqueue a bounded owning control; it must not wait or reclaim readers.
#[derive(Clone)]
pub(crate) struct StateReceiptReturnWakeup {
    active: Option<Arc<dyn Fn() + Send + Sync>>,
    retirement: Option<(Arc<AtomicBool>, Arc<dyn Fn() + Send + Sync>)>,
}
impl StateReceiptReturnWakeup {
    pub(crate) fn with_retirement(
        active: Option<Arc<dyn Fn() + Send + Sync>>,
        armed: Arc<AtomicBool>,
        retired: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> Self {
        Self {
            active,
            retirement: retired.map(|callback| (armed, callback)),
        }
    }
    pub(crate) fn notification(&self) -> Option<Arc<dyn Fn() + Send + Sync>> {
        // Only clone the independently retained owning callback. No temporary
        // Arc owns the bank or its retirement gate when the callback runs.
        if let Some((armed, callback)) = &self.retirement {
            if armed.load(Ordering::Acquire) {
                return Some(callback.clone());
            }
        }
        self.active.clone()
    }
}
impl std::fmt::Debug for StateReceiptReturnWakeup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StateReceiptReturnWakeup")
    }
}

/// Cold-reserved maps for independent returned render receipts.
#[derive(Debug, Clone)]
pub(crate) struct StateMapBank(Arc<StateMapBankInner>);

impl StateMapBank {
    #[cfg(test)]
    pub(crate) fn new(entries: usize, receipts: usize) -> Self {
        Self::new_with_wakeup(entries, receipts, None)
    }
    pub(crate) fn new_with_wakeup(
        entries: usize,
        receipts: usize,
        returned: Option<StateReceiptReturnWakeup>,
    ) -> Self {
        Self(Arc::new(StateMapBankInner {
            entries,
            slots: (0..receipts)
                .map(|_| AtomicPtr::new(Box::into_raw(Box::new(HashMap::with_capacity(entries)))))
                .collect(),
            returned,
        }))
    }

    pub(crate) fn entries(&self) -> usize {
        self.0.entries
    }

    pub(crate) fn receipt_capacity(&self) -> usize {
        self.0.slots.len()
    }

    pub(crate) fn is_reclaimable(&self) -> bool {
        Arc::strong_count(&self.0) == 1
            && self
                .0
                .slots
                .iter()
                .all(|slot| !slot.load(Ordering::Acquire).is_null())
    }

    pub(crate) fn acquire(&self, required: usize) -> Result<RenderElementStateMap, FrameWorkspaceError> {
        if required > self.0.entries {
            return Err(FrameWorkspaceError {
                resource: "render element states",
                required,
                capacity: self.0.entries,
            });
        }
        for (slot, entries) in self.0.slots.iter().enumerate() {
            let entries = entries.swap(std::ptr::null_mut(), Ordering::Acquire);
            if !entries.is_null() {
                // SAFETY: atomic exchange transfers the cold box to this one
                // receipt. Null is occupied, so no other claimant can read it.
                let entries = unsafe { Box::from_raw(entries) };
                return Ok(RenderElementStateMap {
                    entries: Some(StateMapStorage::Reserved(entries)),
                    return_to: Some((self.clone(), slot)),
                });
            }
        }
        Err(FrameWorkspaceError {
            resource: "retained render receipts",
            required: self.0.slots.len().saturating_add(1),
            capacity: self.0.slots.len(),
        })
    }
}

impl Drop for StateMapBankInner {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            let entries = *slot.get_mut();
            if !entries.is_null() {
                // SAFETY: the last bank owner is gone. Live receipts retain a
                // bank Arc, so every non-null slot is exclusively owned here.
                drop(unsafe { Box::from_raw(entries) });
            }
        }
    }
}

#[derive(Debug)]
enum StateMapStorage {
    Detached(HashMap<Id, RenderElementState>),
    Reserved(Box<HashMap<Id, RenderElementState>>),
}
impl StateMapStorage {
    fn get(&self) -> &HashMap<Id, RenderElementState> {
        match self {
            Self::Detached(map) => map,
            Self::Reserved(map) => map,
        }
    }
    fn get_mut(&mut self) -> &mut HashMap<Id, RenderElementState> {
        match self {
            Self::Detached(map) => map,
            Self::Reserved(map) => map,
        }
    }
}

/// An independently owned map, optionally leased from a cold-prepared bank.
///
/// Its storage cannot be reused while a receipt is alive. Warm paths move
/// receipts and use [`Self::try_clone_reserved`] when an independent copy is
/// required. There is no implicit allocating `Clone` implementation.
#[derive(Debug)]
pub struct RenderElementStateMap {
    entries: Option<StateMapStorage>,
    return_to: Option<(StateMapBank, usize)>,
}

impl Default for RenderElementStateMap {
    fn default() -> Self {
        HashMap::new().into()
    }
}

impl From<HashMap<Id, RenderElementState>> for RenderElementStateMap {
    fn from(entries: HashMap<Id, RenderElementState>) -> Self {
        Self {
            entries: Some(StateMapStorage::Detached(entries)),
            return_to: None,
        }
    }
}

impl RenderElementStateMap {
    /// Clone into another admitted receipt slot, refusing held-slot exhaustion.
    pub fn try_clone_reserved(&self) -> Result<Self, FrameWorkspaceError> {
        let Some((bank, _)) = &self.return_to else {
            return Err(FrameWorkspaceError {
                resource: "unprepared render receipt clone",
                required: self.len(),
                capacity: 0,
            });
        };
        let mut copy = bank.acquire(self.len())?;
        copy.extend(self.iter().map(|(id, state)| (id.clone(), *state)));
        Ok(copy)
    }
}

impl Deref for RenderElementStateMap {
    type Target = HashMap<Id, RenderElementState>;
    fn deref(&self) -> &Self::Target {
        self.entries.as_ref().expect("live state map").get()
    }
}
impl DerefMut for RenderElementStateMap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.entries.as_mut().expect("live state map").get_mut()
    }
}
impl Drop for RenderElementStateMap {
    fn drop(&mut self) {
        if let Some((bank, slot)) = self.return_to.take() {
            let Some(StateMapStorage::Reserved(mut entries)) = self.entries.take() else {
                unreachable!("leased state map")
            };
            // Drop IDs before publishing storage back to the cold bank.
            entries.clear();
            let old = bank.0.slots[slot].swap(Box::into_raw(entries), Ordering::Release);
            debug_assert!(old.is_null(), "exact receipt slot returned once");
            let notify = bank
                .0
                .returned
                .as_ref()
                .and_then(StateReceiptReturnWakeup::notification);
            drop(bank);
            if let Some(notify) = notify {
                notify();
            }
        }
    }
}

/// Owning compatibility iterator. Dropping it returns the exact receipt map.
#[derive(Debug)]
pub struct RenderElementStateMapIntoIter(RenderElementStateMap);

impl Iterator for RenderElementStateMapIntoIter {
    type Item = (Id, RenderElementState);
    fn next(&mut self) -> Option<Self::Item> {
        let id = self.0.keys().next()?.clone();
        self.0.remove_entry(&id)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.len();
        (n, Some(n))
    }
}
impl ExactSizeIterator for RenderElementStateMapIntoIter {}
impl IntoIterator for RenderElementStateMap {
    type Item = (Id, RenderElementState);
    type IntoIter = RenderElementStateMapIntoIter;
    fn into_iter(self) -> Self::IntoIter {
        RenderElementStateMapIntoIter(self)
    }
}

#[cfg(test)]
#[path = "state_map_tests.rs"]
mod tests;
