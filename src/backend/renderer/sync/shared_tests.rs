use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Weak,
};

use super::{SyncPoint, SyncPointOwnerReturn};
use crate::backend::renderer::sync::{Fence, Interrupted};

#[derive(Debug)]
struct CpuFence(AtomicBool);
impl Fence for CpuFence {
    fn is_signaled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    fn wait(&self) -> Result<(), Interrupted> {
        self.is_signaled().then_some(()).ok_or(Interrupted)
    }
    fn is_exportable(&self) -> bool {
        false
    }
    fn export(&self) -> Option<std::os::fd::OwnedFd> {
        None
    }
}

#[derive(Debug)]
struct ReaderOwner {
    readers: AtomicUsize,
    native: Weak<dyn Fence>,
    native_refs_on_final_return: AtomicUsize,
}
impl SyncPointOwnerReturn for ReaderOwner {
    fn acquire_reader(&self) {
        self.readers.fetch_add(1, Ordering::AcqRel);
    }
    fn release_reader(&self) {
        if self.readers.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.native_refs_on_final_return
                .store(self.native.strong_count(), Ordering::Release);
        }
    }
}
fn owned(ready: bool) -> (SyncPoint, Arc<ReaderOwner>) {
    let fence: Arc<dyn Fence> = Arc::new(CpuFence(AtomicBool::new(ready)));
    let owner = Arc::new(ReaderOwner {
        readers: AtomicUsize::new(0),
        native: Arc::downgrade(&fence),
        native_refs_on_final_return: AtomicUsize::new(usize::MAX),
    });
    (
        SyncPoint::from_shared_fence_with_owner(fence, owner.clone()),
        owner,
    )
}

#[test]
fn final_reader_notification_follows_actual_native_arc_release() {
    let (point, owner) = owned(false);
    let clone = point.clone();
    assert_eq!(owner.readers.load(Ordering::Acquire), 2);
    drop(point);
    assert_eq!(owner.readers.load(Ordering::Acquire), 1);
    assert_eq!(owner.native.strong_count(), 1);
    drop(clone);
    assert_eq!(owner.readers.load(Ordering::Acquire), 0);
    assert_eq!(owner.native_refs_on_final_return.load(Ordering::Acquire), 0);
}

#[test]
fn native_identity_compares_exact_shared_owner_without_collapsing_compound_proofs() {
    let (first, owner) = owned(false);
    let same = first.clone();
    let (different, _) = owned(false);
    assert!(first.same_native_owner(&same));
    assert!(!first.same_native_owner(&different));
    assert!(!first.same_native_owner(&SyncPoint::signaled()));
    assert!(!SyncPoint::signaled().same_native_owner(&SyncPoint::signaled()));
    let pair = same.try_join(different).unwrap();
    assert!(!pair.same_native_owner(&pair.clone()));
    assert!(!first.same_native_owner(&pair));
    assert_eq!(owner.readers.load(Ordering::Acquire), 2);
}

#[test]
fn inline_join_keeps_both_actual_owners_and_requires_both_fences() {
    let (first, first_owner) = owned(true);
    let (second, second_owner) = owned(false);
    let joined = first.try_join(second).unwrap();
    assert!(joined.contains_fence());
    assert!(!joined.is_reached());
    assert!(joined.get::<CpuFence>().is_none());
    assert!(!joined.is_exportable());
    assert!(joined.export().is_none());
    assert!(joined.wait().is_err());
    assert_eq!(first_owner.readers.load(Ordering::Acquire), 1);
    assert_eq!(second_owner.readers.load(Ordering::Acquire), 1);
    let clone = joined.clone();
    drop(joined);
    assert_eq!(first_owner.readers.load(Ordering::Acquire), 1);
    assert_eq!(second_owner.readers.load(Ordering::Acquire), 1);
    drop(clone);
    assert_eq!(first_owner.native_refs_on_final_return.load(Ordering::Acquire), 0);
    assert_eq!(
        second_owner.native_refs_on_final_return.load(Ordering::Acquire),
        0
    );
}

#[test]
fn overflowing_join_returns_every_original_reader_unchanged() {
    let (first, a) = owned(true);
    let (second, b) = owned(false);
    let (third, c) = owned(false);
    let pair = first.try_join(second).unwrap();
    let (pair, third) = pair.try_join(third).unwrap_err();
    for owner in [&a, &b, &c] {
        assert_eq!(owner.readers.load(Ordering::Acquire), 1);
    }
    assert!(!pair.is_reached());
    assert!(third.get::<CpuFence>().is_some());
    drop(pair);
    assert_eq!(c.readers.load(Ordering::Acquire), 1);
    drop(third);
    for owner in [&a, &b, &c] {
        assert_eq!(owner.native_refs_on_final_return.load(Ordering::Acquire), 0);
    }
}

#[test]
fn empty_join_preserves_single_fence_get_and_ready_semantics() {
    let (point, owner) = owned(true);
    let point = SyncPoint::signaled().try_join(point).unwrap();
    assert!(point.get::<CpuFence>().is_some());
    assert!(point.is_reached());
    point.wait().unwrap();
    drop(point);
    assert_eq!(owner.native_refs_on_final_return.load(Ordering::Acquire), 0);
}
