//! Allocation-free attribution at the existing successful native creation edge.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::GpuAllocationKind;

const KINDS: usize = 6;
const EXCLUSION_CAPACITY: usize = 8;
static OBSERVED: AtomicBool = AtomicBool::new(false);
static COMPLETE: AtomicBool = AtomicBool::new(true);
static COUNTED: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
static CLIENT: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
static SHM: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];

/// Successful imports excluded by the frame-path acceptance definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFrameAllocationExclusion {
    /// The first native binding of an explicitly prepared client source.
    ClientFirstImport,
    /// Creation of a native texture for a Wayland SHM generation.
    ShmTextureCreation,
}

#[derive(Clone, Copy)]
struct Pending {
    id: u64,
    reason: GpuFrameAllocationExclusion,
    creations: [u64; KINDS],
}

struct State {
    frames: usize,
    next_id: u64,
    pending: [Option<Pending>; EXCLUSION_CAPACITY],
}

thread_local! {
    static STATE: RefCell<State> = const { RefCell::new(State {
        frames: 0, next_id: 0, pending: [None; EXCLUSION_CAPACITY],
    }) };
}

/// Explicit accepted producer draw scope. Guards cannot move between threads.
#[derive(Debug)]
pub struct GpuFrameAllocationScope(PhantomData<Rc<()>>);

impl GpuFrameAllocationScope {
    /// Bracket actual frame/cursor/capture work, after cold source preparation.
    pub fn enter() -> Self {
        OBSERVED.store(true, Ordering::Release);
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            if let Some(depth) = state.frames.checked_add(1) {
                state.frames = depth;
            } else {
                COMPLETE.store(false, Ordering::Release);
            }
        });
        Self(PhantomData)
    }
}

impl Drop for GpuFrameAllocationScope {
    fn drop(&mut self) {
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.frames = state.frames.saturating_sub(1);
            if state.frames == 0 && state.pending.iter().any(Option::is_some) {
                // An escaped exclusion cannot classify operations reliably.
                COMPLETE.store(false, Ordering::Release);
            }
        });
    }
}

/// Transactional attribution of native creations by one import attempt.
/// Failed attempts count any successfully created partial resources as frame
/// allocations; only an accepted texture/binding commits an exclusion.
#[derive(Debug)]
pub struct GpuFrameAllocationExclusionScope {
    id: Option<u64>,
    accepted: bool,
    _thread: PhantomData<Rc<()>>,
}

impl GpuFrameAllocationExclusionScope {
    /// Outside an accepted draw scope this is an allocation-free no-op.
    pub fn enter(reason: GpuFrameAllocationExclusion) -> Self {
        let id = STATE.with(|state| {
            let mut state = state.borrow_mut();
            if state.frames == 0 {
                return None;
            }
            let Some(id) = state.next_id.checked_add(1) else {
                COMPLETE.store(false, Ordering::Release);
                return None;
            };
            let Some(index) = state.pending.iter().position(Option::is_none) else {
                COMPLETE.store(false, Ordering::Release);
                return None;
            };
            state.next_id = id;
            state.pending[index] = Some(Pending {
                id,
                reason,
                creations: [0; KINDS],
            });
            Some(id)
        });
        Self {
            id,
            accepted: false,
            _thread: PhantomData,
        }
    }

    /// The exact import returned a successfully created texture/binding.
    pub fn accepted(&mut self) {
        self.accepted = true;
    }
}

impl Drop for GpuFrameAllocationExclusionScope {
    fn drop(&mut self) {
        let Some(id) = self.id else { return };
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            let Some(index) = state.pending.iter().position(|p| p.is_some_and(|p| p.id == id)) else {
                COMPLETE.store(false, Ordering::Release);
                return;
            };
            let pending = state.pending[index].take().unwrap();
            if state.pending.iter().flatten().any(|p| p.id > id) {
                COMPLETE.store(false, Ordering::Release);
            }
            let counters = if self.accepted {
                match pending.reason {
                    GpuFrameAllocationExclusion::ClientFirstImport => &CLIENT,
                    GpuFrameAllocationExclusion::ShmTextureCreation => &SHM,
                }
            } else {
                &COUNTED
            };
            for (counter, value) in counters.iter().zip(pending.creations) {
                add(counter, value);
            }
        });
    }
}

fn add(counter: &AtomicU64, value: u64) {
    if counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(value))
        .is_err()
    {
        COMPLETE.store(false, Ordering::Release);
    }
}

pub(super) fn note(kind: GpuAllocationKind) {
    let index = match kind {
        GpuAllocationKind::VulkanImage => 0,
        GpuAllocationKind::VulkanBuffer => 1,
        GpuAllocationKind::VulkanDeviceMemory => 2,
        GpuAllocationKind::GbmBuffer => 3,
        GpuAllocationKind::DrmDumbBuffer => 4,
        GpuAllocationKind::DrmFramebuffer => 5,
    };
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if state.frames == 0 {
            return;
        }
        if let Some(pending) = state.pending.iter_mut().flatten().max_by_key(|p| p.id) {
            if let Some(value) = pending.creations[index].checked_add(1) {
                pending.creations[index] = value;
            } else {
                COMPLETE.store(false, Ordering::Release);
            }
        } else {
            add(&COUNTED[index], 1);
        }
    });
}

/// Process-lifetime successful native operations, read without Vulkan calls.
/// An image and its memory binding are distinct API operations, never aliases
/// or physical-byte estimates. Array order is image, buffer, memory, GBM, GEM,
/// framebuffer. Unsupported/overflowed attribution returns `None`.
#[derive(Debug, Clone, Copy)]
pub struct GpuFrameAllocationSnapshot {
    /// Counted native creations by operation kind.
    pub counted: [u64; KINDS],
    /// Actual native operations committed by first client imports.
    pub client_first_import: [u64; KINDS],
    /// Actual native operations committed by SHM texture creations.
    pub shm_texture_creation: [u64; KINDS],
}

/// Observe the same native event authority that drives the installed observer.
pub fn gpu_frame_allocation_snapshot() -> Option<GpuFrameAllocationSnapshot> {
    if !OBSERVED.load(Ordering::Acquire) || !COMPLETE.load(Ordering::Acquire) {
        return None;
    }
    let read = |values: &[AtomicU64; KINDS]| std::array::from_fn(|i| values[i].load(Ordering::Relaxed));
    let result = GpuFrameAllocationSnapshot {
        counted: read(&COUNTED),
        client_first_import: read(&CLIENT),
        shm_texture_creation: read(&SHM),
    };
    COMPLETE.load(Ordering::Acquire).then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_operations_exclude_only_successful_import_transactions() {
        let _frame = GpuFrameAllocationScope::enter();
        let before = gpu_frame_allocation_snapshot().unwrap();
        note(GpuAllocationKind::VulkanBuffer);
        {
            let mut client =
                GpuFrameAllocationExclusionScope::enter(GpuFrameAllocationExclusion::ClientFirstImport);
            note(GpuAllocationKind::VulkanImage);
            note(GpuAllocationKind::VulkanDeviceMemory);
            note(GpuAllocationKind::VulkanDeviceMemory); // disjoint two-plane binding
            client.accepted();
        }
        {
            let _failed =
                GpuFrameAllocationExclusionScope::enter(GpuFrameAllocationExclusion::ShmTextureCreation);
            note(GpuAllocationKind::VulkanImage);
        }
        {
            let mut shm =
                GpuFrameAllocationExclusionScope::enter(GpuFrameAllocationExclusion::ShmTextureCreation);
            note(GpuAllocationKind::VulkanImage);
            note(GpuAllocationKind::VulkanDeviceMemory);
            shm.accepted();
        }
        let after = gpu_frame_allocation_snapshot().unwrap();
        assert_eq!(after.counted[0] - before.counted[0], 1);
        assert_eq!(after.counted[1] - before.counted[1], 1);
        assert_eq!(after.counted[2] - before.counted[2], 0);
        assert_eq!(after.client_first_import[0] - before.client_first_import[0], 1);
        assert_eq!(after.client_first_import[2] - before.client_first_import[2], 2);
        assert_eq!(after.shm_texture_creation[0] - before.shm_texture_creation[0], 1);
    }
}
