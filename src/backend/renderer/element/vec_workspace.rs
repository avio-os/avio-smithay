//! Independently leased numeric/descriptor vectors reserved by the cold owner.
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc, Mutex,
    },
};

use super::{FrameWorkspaceError, StateReceiptReturnWakeup};

#[derive(Debug)]
struct VecBankInner<T> {
    entries: usize,
    resource: &'static str,
    slots: Vec<AtomicPtr<Vec<T>>>,
    // AtomicPtr alone does not express the uniquely transferred T ownership.
    // A slot may cross threads only when its contents are Send; no mutex is
    // actually constructed or waited on.
    ownership: PhantomData<Mutex<T>>,
    returned: Option<StateReceiptReturnWakeup>,
}

#[derive(Debug)]
pub(crate) struct VecStorageBank<T>(Arc<VecBankInner<T>>);

impl<T> Clone for VecStorageBank<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> VecStorageBank<T> {
    #[cfg(test)]
    pub(crate) fn new(resource: &'static str, entries: usize, receipts: usize) -> Self {
        Self::new_with_wakeup(resource, entries, receipts, None)
    }

    pub(crate) fn new_with_wakeup(
        resource: &'static str,
        entries: usize,
        receipts: usize,
        returned: Option<StateReceiptReturnWakeup>,
    ) -> Self {
        Self(Arc::new(VecBankInner {
            entries,
            resource,
            slots: (0..receipts)
                .map(|_| AtomicPtr::new(Box::into_raw(Box::new(Vec::with_capacity(entries)))))
                .collect(),
            ownership: PhantomData,
            returned,
        }))
    }

    pub(crate) fn entries(&self) -> usize {
        self.0.entries
    }

    pub(crate) fn is_reclaimable(&self) -> bool {
        Arc::strong_count(&self.0) == 1
            && self
                .0
                .slots
                .iter()
                .all(|slot| !slot.load(Ordering::Acquire).is_null())
    }

    pub(crate) fn acquire(&self, required: usize) -> Result<WorkspaceVec<T>, FrameWorkspaceError> {
        if required > self.0.entries {
            return Err(FrameWorkspaceError {
                resource: self.0.resource,
                required,
                capacity: self.0.entries,
            });
        }
        for (slot, entries) in self.0.slots.iter().enumerate() {
            let entries = entries.swap(std::ptr::null_mut(), Ordering::Acquire);
            if !entries.is_null() {
                // SAFETY: the atomic exchange gives this receipt exclusive
                // ownership of the one cold Box; occupied slots contain null.
                let entries = unsafe { Box::from_raw(entries) };
                return Ok(WorkspaceVec {
                    entries: Some(entries),
                    return_to: Some((self.clone(), slot)),
                });
            }
        }
        Err(FrameWorkspaceError {
            resource: self.0.resource,
            required: self.0.slots.len().saturating_add(1),
            capacity: self.0.slots.len(),
        })
    }
}

impl<T> Drop for VecBankInner<T> {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            let entries = *slot.get_mut();
            if !entries.is_null() {
                // SAFETY: live receipts hold a bank Arc, so the final bank
                // owner exclusively owns every non-null cold Box.
                drop(unsafe { Box::from_raw(entries) });
            }
        }
    }
}

/// A unique vector receipt. Reserved mode never silently grows or clones.
#[derive(Debug)]
pub struct WorkspaceVec<T> {
    entries: Option<Box<Vec<T>>>,
    return_to: Option<(VecStorageBank<T>, usize)>,
}

impl<T> WorkspaceVec<T> {
    pub(crate) fn legacy(entries: Vec<T>) -> Self {
        // A detached legacy vector is explicit and may allocate. Reserved
        // frame paths always acquire their storage from VecStorageBank.
        Self {
            entries: Some(Box::new(entries)),
            return_to: None,
        }
    }

    pub(crate) fn push(&mut self, value: T) -> Result<(), FrameWorkspaceError> {
        if let Some((bank, _)) = &self.return_to {
            if self.len() == bank.0.entries {
                return Err(FrameWorkspaceError {
                    resource: bank.0.resource,
                    required: self.len().saturating_add(1),
                    capacity: bank.0.entries,
                });
            }
        }
        self.entries.as_mut().expect("live vector receipt").push(value);
        Ok(())
    }

    pub(crate) fn extend(&mut self, values: impl IntoIterator<Item = T>) -> Result<(), FrameWorkspaceError> {
        for value in values {
            self.push(value)?;
        }
        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        self.entries.as_mut().expect("live vector receipt").clear();
    }
    pub(crate) fn remove(&mut self, index: usize) -> T {
        self.entries.as_mut().expect("live vector receipt").remove(index)
    }
    pub(crate) fn insert(&mut self, index: usize, value: T) -> Result<(), FrameWorkspaceError> {
        if let Some((bank, _)) = &self.return_to {
            if self.len() == bank.0.entries {
                return Err(FrameWorkspaceError {
                    resource: bank.0.resource,
                    required: self.len().saturating_add(1),
                    capacity: bank.0.entries,
                });
            }
        }
        self.entries
            .as_mut()
            .expect("live vector receipt")
            .insert(index, value);
        Ok(())
    }

    pub(crate) fn swap_remove(&mut self, index: usize) -> T {
        self.entries
            .as_mut()
            .expect("live vector receipt")
            .swap_remove(index)
    }
    pub(crate) fn admitted_capacity(&self) -> Option<usize> {
        self.return_to.as_ref().map(|(bank, _)| bank.0.entries)
    }

    #[cfg(test)]
    pub(crate) fn try_clone_reserved(&self) -> Result<Self, FrameWorkspaceError>
    where
        T: Clone,
    {
        let Some((bank, _)) = &self.return_to else {
            return Err(FrameWorkspaceError {
                resource: "unprepared vector receipt copy",
                required: self.len(),
                capacity: 0,
            });
        };
        let mut copy = bank.acquire(self.len())?;
        copy.extend(self.iter().cloned())?;
        Ok(copy)
    }
}

impl<T> Deref for WorkspaceVec<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.entries.as_ref().expect("live vector receipt")
    }
}
impl<T> DerefMut for WorkspaceVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.entries.as_mut().expect("live vector receipt")
    }
}
impl<'a, T> IntoIterator for &'a WorkspaceVec<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<'a, T> IntoIterator for &'a mut WorkspaceVec<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
impl<T> Drop for WorkspaceVec<T> {
    fn drop(&mut self) {
        if let Some((bank, slot)) = self.return_to.take() {
            let mut entries = self.entries.take().expect("live vector receipt");
            // Final native/resource owners are dropped before slot reuse.
            entries.clear();
            let old = bank.0.slots[slot].swap(Box::into_raw(entries), Ordering::Release);
            debug_assert!(old.is_null(), "exact vector receipt returned once");
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

#[cfg(test)]
#[path = "vec_workspace_tests.rs"]
mod tests;
