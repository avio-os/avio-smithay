//! Logical reader return is separate from native GPU completion. This cold
//! object remains owned by its submission slot; reader clone/drop only changes
//! an atomic and wakes a completion worker, never allocates or waits.

use crate::backend::renderer::sync::{Fence, Interrupted, SyncPointOwnerReturn};
use std::{
    fmt,
    os::fd::OwnedFd,
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc, Mutex, MutexGuard,
    },
};

pub(super) struct FenceReaders {
    readers: AtomicU32,
    native_epoch: Mutex<()>,
    slot_alive: AtomicBool,
}
impl Default for FenceReaders {
    fn default() -> Self {
        Self {
            readers: AtomicU32::new(0),
            native_epoch: Mutex::new(()),
            slot_alive: AtomicBool::new(true),
        }
    }
}
impl fmt::Debug for FenceReaders {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FenceReaders")
            .field("readers", &self.readers.load(Ordering::Acquire))
            .finish()
    }
}
impl FenceReaders {
    pub(super) fn try_native_epoch(&self) -> Option<MutexGuard<'_, ()>> {
        self.native_epoch.try_lock().ok()
    }
    pub(super) fn wait_native_epoch(&self) -> MutexGuard<'_, ()> {
        self.native_epoch
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
    pub(super) fn close_slot(&self) {
        // The slot's baseline SyncPoint drops immediately after this flag.
        // Until then its counted guard still makes readiness conservative.
        self.slot_alive.store(false, Ordering::Release);
    }
    fn baseline(&self) -> u32 {
        u32::from(self.slot_alive.load(Ordering::Acquire))
    }
    pub(super) fn count_returned(&self) -> bool {
        self.readers.load(Ordering::Acquire) <= self.baseline()
    }
    pub(super) fn only_internal_reader(&self) -> bool {
        self.readers.load(Ordering::Acquire) == self.baseline() + 1
    }

    pub(super) fn returned(&self) -> bool {
        self.try_native_epoch().is_some() && self.count_returned()
    }
    pub(super) fn wait_returned(&self) -> Result<(), Interrupted> {
        loop {
            // A native observer may have obtained the epoch lock before its
            // reader count became visible. Observe its release first; this
            // wait is confined to the cold completion actor.
            let epoch = self.wait_native_epoch();
            let expected = self.readers.load(Ordering::Acquire);
            if expected <= self.baseline() {
                return Ok(());
            }
            drop(epoch);
            // A concurrent return changes the compared word before sleeping,
            // so futex returns EAGAIN instead of losing its exact wake.
            let result = unsafe {
                libc::syscall(
                    libc::SYS_futex,
                    self.readers.as_ptr(),
                    libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                    expected,
                    std::ptr::null::<libc::timespec>(),
                )
            };
            if result < 0 {
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::EAGAIN) {
                    return Err(Interrupted);
                }
            }
        }
    }
}
impl SyncPointOwnerReturn for FenceReaders {
    fn acquire_reader(&self) {
        self.readers.fetch_add(1, Ordering::AcqRel);
    }
    fn release_reader(&self) {
        let previous = self.readers.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
        if previous <= 2 {
            unsafe {
                libc::syscall(
                    libc::SYS_futex,
                    self.readers.as_ptr(),
                    libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                    i32::MAX,
                );
            }
        }
    }
}
#[derive(Debug)]
pub(super) struct FenceReaderReturn(pub(super) Arc<FenceReaders>);
impl Fence for FenceReaderReturn {
    fn is_signaled(&self) -> bool {
        self.0.returned()
    }
    fn wait(&self) -> Result<(), Interrupted> {
        self.0.wait_returned()
    }
    fn is_exportable(&self) -> bool {
        false
    }
    fn export(&self) -> Option<OwnedFd> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::super::storage_heap_probe::measure;
    use super::*;
    use crate::backend::renderer::sync::SyncPoint;
    #[derive(Debug)]
    struct NativeCompletion;
    impl Fence for NativeCompletion {
        fn is_signaled(&self) -> bool {
            true
        }
        fn wait(&self) -> Result<(), Interrupted> {
            Ok(())
        }
        fn is_exportable(&self) -> bool {
            false
        }
        fn export(&self) -> Option<OwnedFd> {
            None
        }
    }
    #[test]
    fn ready_native_fence_does_not_prove_external_reader_return() {
        let readers = Arc::new(FenceReaders::default());
        let slot = SyncPoint::from_shared_fence_with_owner(Arc::new(NativeCompletion), readers.clone());
        let returned = SyncPoint::from_shared_fence(Arc::new(FenceReaderReturn(readers.clone())));
        let external = slot.clone();
        assert!(external.is_reached());
        assert!(!returned.is_reached());
        assert!(!returned.is_exportable());
        let last = external.clone();
        drop(external);
        assert!(!returned.is_reached());
        drop(last);
        assert!(returned.is_reached());
    }
    #[test]
    fn cold_waiter_observes_final_reader_drop_without_timer_or_lost_wake() {
        let readers = Arc::new(FenceReaders::default());
        let slot = SyncPoint::from_shared_fence_with_owner(Arc::new(NativeCompletion), readers.clone());
        let external = slot.clone();
        let waiter = readers.clone();
        let join = std::thread::spawn(move || waiter.wait_returned());
        drop(external);
        assert!(join.join().unwrap().is_ok());
    }
    #[test]
    fn reader_clone_and_return_edge_use_zero_allocator_operations() {
        let readers = Arc::new(FenceReaders::default());
        let slot = SyncPoint::from_shared_fence_with_owner(Arc::new(NativeCompletion), readers.clone());
        let returned = SyncPoint::from_shared_fence(Arc::new(FenceReaderReturn(readers.clone())));
        let (_, operations) = measure(|| {
            for _ in 0..4096 {
                let external = slot.clone();
                let edge = returned.clone();
                assert!(!edge.is_reached());
                drop(external);
                assert!(edge.is_reached());
                drop(edge);
            }
        });
        assert_eq!(operations, [0; 4]);
    }
}
