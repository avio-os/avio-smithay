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
fn prepared_only_acquisition_never_allocates_on_a_miss_or_full_pool() {
    let mut chain = chain();
    assert!(chain.acquire_existing().is_none());
    assert_eq!(chain.allocator.allocations, 0);
    let cold = chain.acquire().unwrap().unwrap();
    assert!(chain.acquire_existing().is_none());
    assert_eq!(chain.allocator.allocations, 1);
    drop(cold);
    let warm = chain.acquire_existing().unwrap();
    assert_eq!(chain.allocator.allocations, 1);
    assert_eq!(chain.allocated_slots(), 1);
    drop(warm);
}

#[test]
fn completion_extraction_preserves_frame_owners_and_defers_destruction() {
    let mut chain = chain();
    let presented = chain.acquire().unwrap().unwrap();
    let unused = chain.acquire().unwrap().unwrap();
    unused.userdata().insert_if_missing_threadsafe(|| 41u64);
    drop(unused);
    let retired = chain.take_unreferenced();
    assert_eq!(retired.iter().flatten().count(), 1);
    assert_eq!(chain.allocated_slots(), 1);
    assert_eq!(chain.allocator.live.load(Ordering::SeqCst), 2);
    assert_eq!(
        retired.iter().flatten().next().unwrap().userdata().get::<u64>(),
        Some(&41)
    );
    assert_eq!(presented.size().w, 640);
    drop(retired);
    assert_eq!(chain.allocator.live.load(Ordering::SeqCst), 1);
    drop(presented);
    let retired = chain.take_unreferenced();
    assert_eq!(chain.allocated_slots(), 0);
    assert!(chain.acquire_existing().is_none());
    assert_eq!(chain.allocator.allocations, 2);
    drop(retired);
    assert_eq!(chain.allocator.live.load(Ordering::SeqCst), 0);
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

#[test]
fn immutable_shield_does_not_consume_a_composition_slot_or_release_readers() {
    let mut chain = chain();
    let shield = Arc::new(chain.acquire().unwrap().unwrap());
    let displayed = shield.clone();
    chain.submitted(&shield);
    assert!(chain.detach(&shield));
    assert!(!chain.detach(&shield));
    let composition = (0..SLOT_CAP)
        .map(|_| {
            chain
                .acquire()
                .unwrap()
                .expect("all ordinary slots remain available")
        })
        .collect::<Vec<_>>();
    assert!(chain.acquire().unwrap().is_none());
    assert_eq!(chain.allocator.live.load(Ordering::SeqCst), SLOT_CAP + 1);
    drop(shield);
    assert_eq!(displayed.size().w, 640);
    assert_eq!(chain.allocator.live.load(Ordering::SeqCst), SLOT_CAP + 1);
    drop(displayed);
    assert_eq!(chain.allocator.live.load(Ordering::SeqCst), SLOT_CAP);
    drop(composition);
    assert_eq!(chain.retire_unreferenced(), SLOT_CAP);
    assert_eq!(chain.allocator.live.load(Ordering::SeqCst), 0);
}

#[test]
fn prepared_mode_and_modifier_adoption_retains_old_reader_and_displaced_controls() {
    let mut chain = chain();
    let front = chain.acquire().unwrap().unwrap();
    let live = chain.allocator.live.clone();
    let mut successor = PreparedSwapchainResize::new(800, 600);
    let mut modifiers = vec![Modifier::Invalid];
    let (_, measured) = crate::backend::renderer::storage_heap_probe::measure(|| {
        chain.adopt_prepared_modifiers(&mut successor, &mut modifiers);
    });
    assert_eq!(measured, [0; 4]);
    assert_eq!(chain.dimensions(), (800, 600));
    assert_eq!(chain.modifiers(), [Modifier::Invalid]);
    assert_eq!(front.size(), (640, 480).into());
    assert_eq!(modifiers, [Modifier::Linear]);
    assert_eq!(live.load(Ordering::SeqCst), 1);
    drop(front);
    assert_eq!(
        live.load(Ordering::SeqCst),
        1,
        "packet owns displaced slot until cold disposal"
    );
    drop(successor);
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[test]
fn same_extent_modifier_restoration_replaces_controls_without_reinterpreting_front() {
    let mut chain = chain();
    let front = chain.acquire().unwrap().unwrap();
    let mut successor = PreparedSwapchainResize::new(640, 480);
    let mut modifiers = vec![Modifier::Invalid];
    chain.adopt_prepared_modifiers(&mut successor, &mut modifiers);
    assert_eq!(chain.allocated_slots(), 0);
    assert_eq!(front.format().modifier, Modifier::Linear);
    assert_eq!(chain.modifiers(), [Modifier::Invalid]);
    assert_eq!(chain.allocator.allocations, 1);
}

#[test]
fn age_overflow_preserves_admitted_backing_and_held_reader_until_cold_retirement() {
    let mut chain = chain();
    let held = chain.acquire().unwrap().unwrap();
    held.userdata().insert_if_missing_threadsafe(|| 37u64);
    let selected = chain.acquire().unwrap().unwrap();
    let control = Arc::as_ptr(&held.0);
    let buffer = &*held as *const TestBuffer;
    chain.submitted(&held);
    for _ in 0..254 {
        chain.submitted(&selected);
    }
    assert_eq!(held.age(), u8::MAX);
    chain.submitted(&selected);
    assert_eq!(held.age(), 0, "unknown age requires full-buffer repair");
    assert!(chain.slots.iter().any(|slot| Arc::as_ptr(slot) == control));
    assert_eq!(&*held as *const TestBuffer, buffer);
    assert_eq!(held.userdata().get::<u64>(), Some(&37));
    assert_eq!(chain.allocated_slots(), 2);
    assert_eq!(chain.allocator.live.load(Ordering::SeqCst), 2);
    assert_eq!(chain.retire_unreferenced(), 0, "both readers are still held");
    drop(held);
    let reused = chain.acquire_existing().unwrap();
    assert_eq!(Arc::as_ptr(&reused.0), control);
    assert_eq!(&*reused as *const TestBuffer, buffer);
    assert_eq!(reused.age(), 0);
    assert_eq!(chain.allocator.allocations, 2);
    drop((reused, selected));
    assert_eq!(chain.retire_unreferenced(), 2);
    assert_eq!(chain.allocator.live.load(Ordering::SeqCst), 0);
}

#[test]
fn repeated_age_overflows_are_heap_quiet_for_4096_warm_submissions() {
    let mut chain = chain();
    let held = chain.acquire().unwrap().unwrap();
    let selected = chain.acquire().unwrap().unwrap();
    let control = Arc::as_ptr(&held.0);
    let mut overflows = 0;
    let (_, heap) = crate::backend::renderer::storage_heap_probe::measure(|| {
        for submission in 0..4096 {
            if submission % 256 == 0 {
                chain.submitted(&held);
            }
            let previous_age = held.age();
            chain.submitted(&selected);
            if previous_age == u8::MAX {
                assert_eq!(held.age(), 0);
                overflows += 1;
            }
            assert!(chain.slots.iter().any(|slot| Arc::as_ptr(slot) == control));
            assert_eq!(chain.allocator.live.load(Ordering::SeqCst), 2);
        }
    });
    assert_eq!(overflows, 16);
    assert_eq!(heap, [0; 4]);
    assert_eq!(chain.allocator.allocations, 2);
    assert_eq!(held.age(), 0);
    assert_eq!(selected.age(), 1);
}
