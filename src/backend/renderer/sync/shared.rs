//! Bounded shared-fence custody, including exact logical reader returns.

use std::{os::fd::OwnedFd, sync::Arc};

use super::{owner_return::OwnerReturnGuard, Fence, Interrupted, SyncPointOwnerReturn};

#[derive(Debug, Clone)]
struct SharedFence {
    // Field order is the protocol: release the actual native reference before
    // publishing its logical owner-return notification.
    fence: Arc<dyn Fence>,
    _owner_return: Option<OwnerReturnGuard>,
}

/// A synchronization point with at most two exact constituent fences.
///
/// A compound point is the AND of both children. It has no synthetic GPU
/// completion receipt and is not exportable as one native descriptor.
#[derive(Debug, Clone)]
#[must_use = "this SyncPoint may contain a fence that must be awaited"]
pub struct SyncPoint {
    fences: [Option<SharedFence>; 2],
}

impl Default for SyncPoint {
    fn default() -> Self {
        Self::signaled()
    }
}

impl SyncPoint {
    /// Create an already signaled synchronization point.
    pub fn signaled() -> Self {
        Self { fences: [None, None] }
    }

    /// Borrow a cold shared fence without allocating another owner container.
    pub fn from_shared_fence(fence: Arc<dyn Fence>) -> Self {
        Self {
            fences: [
                Some(SharedFence {
                    fence,
                    _owner_return: None,
                }),
                None,
            ],
        }
    }

    /// Borrow a cold shared fence and retain its exact logical reader witness.
    ///
    /// Clone acquires another reader. Drop first releases its native reference,
    /// then notifies the owner. The hook is not a native completion proof.
    pub fn from_shared_fence_with_owner(fence: Arc<dyn Fence>, owner: Arc<dyn SyncPointOwnerReturn>) -> Self {
        Self {
            fences: [
                Some(SharedFence {
                    fence,
                    _owner_return: Some(OwnerReturnGuard::new(owner)),
                }),
                None,
            ],
        }
    }

    /// Join two points without allocating or discarding any constituent owner.
    ///
    /// More than two actual fences returns both original inputs unchanged.
    /// Even a presently signaled child retains its exact reader witness.
    pub fn try_join(self, other: Self) -> Result<Self, (Self, Self)> {
        let count = self.fences.iter().filter(|f| f.is_some()).count()
            + other.fences.iter().filter(|f| f.is_some()).count();
        if count > 2 {
            return Err((self, other));
        }
        let mut joined = Self::signaled();
        for (index, fence) in self.fences.into_iter().chain(other.fences).flatten().enumerate() {
            joined.fences[index] = Some(fence);
        }
        Ok(joined)
    }

    /// Whether this point contains any fence.
    pub fn contains_fence(&self) -> bool {
        self.fences[0].is_some()
    }

    /// Whether both points retain the very same single native fence owner.
    /// Signaled points and compound points have no single native identity.
    /// This does not compare borrowed addresses or descriptor numbers.
    pub fn same_native_owner(&self, other: &Self) -> bool {
        match (self.single_fence(), other.single_fence()) {
            (Some(left), Some(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }

    /// Borrow the underlying single fence, preserving the existing downcast API.
    /// A compound point has no single fence and returns None.
    pub fn get<F: Fence + 'static>(&self) -> Option<&F> {
        self.single_fence().and_then(|f| f.downcast_ref())
    }

    /// Query whether every constituent fence has actually signaled.
    pub fn is_reached(&self) -> bool {
        self.fences.iter().flatten().all(|f| f.fence.is_signaled())
    }

    /// Wait for every constituent fence. Call only from a permitted cold waiter.
    #[profiling::function]
    pub fn wait(&self) -> Result<(), Interrupted> {
        for fence in self.fences.iter().flatten() {
            fence.fence.wait()?;
        }
        Ok(())
    }

    /// Whether this single fence can be exported as a native descriptor.
    pub fn is_exportable(&self) -> bool {
        self.single_fence().is_some_and(|f| f.is_exportable())
    }

    /// Export a single native fence; compound points remain nonexportable.
    #[profiling::function]
    pub fn export(&self) -> Option<OwnedFd> {
        self.single_fence().and_then(|f| f.export())
    }

    fn single_fence(&self) -> Option<&Arc<dyn Fence>> {
        self.fences[1].is_none().then_some(())?;
        self.fences[0].as_ref().map(|f| &f.fence)
    }
}

impl<T: Fence + 'static> From<T> for SyncPoint {
    #[inline]
    fn from(value: T) -> Self {
        Self::from_shared_fence(Arc::new(value))
    }
}

#[cfg(test)]
#[path = "shared_tests.rs"]
mod tests;
