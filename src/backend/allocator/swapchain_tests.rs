use super::*;
use crate::utils::{Buffer as BufferCoords, Size};
use std::convert::Infallible;
use std::sync::atomic::AtomicUsize;

#[derive(Debug)]
struct TestBuffer {
    size: Size<i32, BufferCoords>,
    format: super::super::Format,
    live: Arc<AtomicUsize>,
}
impl Drop for TestBuffer {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}
impl Buffer for TestBuffer {
    fn size(&self) -> Size<i32, BufferCoords> {
        self.size
    }
    fn format(&self) -> super::super::Format {
        self.format
    }
}
#[derive(Default)]
struct TestAllocator {
    allocations: usize,
    live: Arc<AtomicUsize>,
}
impl Allocator for TestAllocator {
    type Buffer = TestBuffer;
    type Error = Infallible;
    fn create_buffer(
        &mut self,
        width: u32,
        height: u32,
        code: Fourcc,
        modifiers: &[Modifier],
    ) -> Result<TestBuffer, Infallible> {
        self.allocations += 1;
        self.live.fetch_add(1, Ordering::SeqCst);
        Ok(TestBuffer {
            size: (width as i32, height as i32).into(),
            format: super::super::Format {
                code,
                modifier: modifiers[0],
            },
            live: self.live.clone(),
        })
    }
}
fn chain() -> Swapchain<TestAllocator> {
    Swapchain::new(
        TestAllocator::default(),
        640,
        480,
        Fourcc::Argb8888,
        vec![Modifier::Linear],
    )
}

#[test]
fn adoption_preserves_off_thread_userdata_without_an_allocator_call() {
    let mut chain = chain();
    let buffer = chain
        .allocator
        .create_buffer(640, 480, Fourcc::Argb8888, &[Modifier::Linear])
        .unwrap();
    let slot = Slot::new(buffer);
    slot.userdata().insert_if_missing_threadsafe(|| 37u64);
    chain.adopt(vec![slot]).unwrap();
    let slot = chain.acquire().unwrap().unwrap();
    assert_eq!(chain.allocator.allocations, 1);
    assert_eq!(slot.age(), 0);
    assert_eq!(slot.userdata().get::<u64>(), Some(&37));
}

#[test]
fn retirement_keeps_submitted_owners_and_frees_only_unused_slots() {
    let mut chain = chain();
    let submitted = chain.acquire().unwrap().unwrap();
    chain.submitted(&submitted);
    let unused = chain.acquire().unwrap().unwrap();
    drop(unused);
    assert_eq!(chain.retire_unreferenced(), 1);
    assert_eq!(chain.allocator.live.load(Ordering::SeqCst), 1);
    assert_eq!(submitted.size().w, 640);
    assert_eq!(chain.retire_unreferenced(), 0);
    drop(submitted);
    assert_eq!(chain.retire_unreferenced(), 1);
    assert_eq!(chain.allocator.live.load(Ordering::SeqCst), 0);
}

#[test]
fn rejected_adoption_is_atomic_and_returns_prepared_resources() {
    let mut chain = chain();
    let first = Slot::new(
        chain
            .allocator
            .create_buffer(640, 480, Fourcc::Argb8888, &[Modifier::Linear])
            .unwrap(),
    );
    let wrong = Slot::new(
        chain
            .allocator
            .create_buffer(639, 480, Fourcc::Argb8888, &[Modifier::Linear])
            .unwrap(),
    );
    let rejected = chain.adopt(vec![first, wrong]).unwrap_err();
    assert_eq!(rejected.reason, AdoptionFailure::WrongSize);
    assert_eq!(rejected.slots.len(), 2);
    assert!(chain.slots.iter().all(|slot| slot.buffer.is_none()));
    drop(rejected);
    assert_eq!(chain.allocator.live.load(Ordering::SeqCst), 0);
}

#[test]
fn adoption_never_replaces_an_acquired_or_cached_buffer() {
    let mut chain = chain();
    let held = chain.acquire().unwrap().unwrap();
    let prepared = (0..SLOT_CAP)
        .map(|_| {
            Slot::new(
                chain
                    .allocator
                    .create_buffer(640, 480, Fourcc::Argb8888, &[Modifier::Linear])
                    .unwrap(),
            )
        })
        .collect();
    let rejected = chain.adopt(prepared).unwrap_err();
    assert_eq!(rejected.reason, AdoptionFailure::NoVacancies);
    assert_eq!(held.size().w, 640);
    assert_eq!(chain.slots.iter().filter(|slot| slot.buffer.is_some()).count(), 1);
}

#[test]
fn adoption_rejects_slots_still_owned_by_another_swapchain() {
    let mut source = chain();
    let mut destination = chain();
    let owned = source.acquire().unwrap().unwrap();
    let rejected = destination.adopt(vec![owned]).unwrap_err();
    assert_eq!(rejected.reason, AdoptionFailure::AlreadyOwned);
    assert!(destination.slots.iter().all(|slot| slot.buffer.is_none()));
    assert_eq!(source.allocator.live.load(Ordering::SeqCst), 1);
}
