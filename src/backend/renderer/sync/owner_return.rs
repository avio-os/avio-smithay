//! Exact logical reader returns for cold shared fence storage.

use std::{fmt, sync::Arc};

/// A cold owner observes each live `SyncPoint` reader of its shared fence.
///
/// Hooks must be bounded and must not allocate, wait or call a graphics driver.
/// The release hook runs after that reader's fence Arc has been released. It
/// may wake an existing cold completion worker; it does not prove GPU completion.
pub trait SyncPointOwnerReturn: fmt::Debug + Send + Sync {
    /// One actual reader was constructed or cloned.
    fn acquire_reader(&self);
    /// That reader released its exact fence reference.
    fn release_reader(&self);
}

#[derive(Debug)]
pub(super) struct OwnerReturnGuard(Arc<dyn SyncPointOwnerReturn>);

impl OwnerReturnGuard {
    pub(super) fn new(owner: Arc<dyn SyncPointOwnerReturn>) -> Self {
        owner.acquire_reader();
        Self(owner)
    }
}

impl Clone for OwnerReturnGuard {
    fn clone(&self) -> Self {
        Self::new(self.0.clone())
    }
}

impl Drop for OwnerReturnGuard {
    fn drop(&mut self) {
        self.0.release_reader();
    }
}
