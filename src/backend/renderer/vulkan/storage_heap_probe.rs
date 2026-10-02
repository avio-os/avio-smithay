//! Test-only thread-local allocator accounting shared by storage fixtures.
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
thread_local! { static ENABLED: Cell<bool> = const { Cell::new(false) }; static OPERATIONS: Cell<[usize;4]> = const { Cell::new([0;4]) }; }
struct StorageTestAllocator;
fn note(index: usize) {
    ENABLED
        .try_with(|enabled| {
            if enabled.get() {
                let _ = OPERATIONS.try_with(|operations| {
                    let mut counts = operations.get();
                    counts[index] += 1;
                    operations.set(counts);
                });
            }
        })
        .ok();
}
unsafe impl GlobalAlloc for StorageTestAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(0);
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(1);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        note(2);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        note(3);
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: StorageTestAllocator = StorageTestAllocator;
pub(crate) fn measure<T>(operation: impl FnOnce() -> T) -> (T, [usize; 4]) {
    OPERATIONS.with(|counts| counts.set([0; 4]));
    ENABLED.with(|enabled| assert!(!enabled.replace(true)));
    struct Disable;
    impl Drop for Disable {
        fn drop(&mut self) {
            ENABLED.with(|enabled| enabled.set(false));
        }
    }
    let disable = Disable;
    let result = operation();
    drop(disable);
    (result, OPERATIONS.with(Cell::get))
}
