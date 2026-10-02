//! Restore cold native-cache table ownership on every selector result.
use super::*;

pub(super) fn finish_selection<T, E, F: Framebuffer>(
    current: &mut IndexMap<Id, ElementState<F>>,
    previous: &mut IndexMap<Id, ElementState<F>>,
    selected: IndexMap<Id, ElementState<F>>,
    result: Result<T, E>,
) -> Result<T, E> {
    // current is the empty placeholder left by mem::take. Never let a native
    // preparation/capacity error drop the admitted table on the render thread.
    // Partially transferred caches stay in one of these two owning tables.
    *current = selected;
    if result.is_ok() {
        previous.clear();
    }
    result
}
