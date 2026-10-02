//! Device-memory ownership accounting. Each successful Vulkan allocation owns
//! one non-cloneable guard; texture clones and submission custody share it.

use std::{
    cell::RefCell,
    marker::PhantomData,
    rc::Rc,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    thread::{self, ThreadId},
};

use tracing::trace;

/// The operation that owns a Vulkan device-memory allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanAllocationReason {
    /// Memory backing a renderer-local sampled texture.
    Texture,
    /// Memory backing a renderer-local offscreen render target.
    RenderTarget,
    /// A Vulkan memory binding imported from an external dma-buf.
    Import,
    /// Temporary readback or other operation-owned scratch memory.
    Scratch,
    /// Persistently mapped memory backing an upload arena chunk.
    Upload,
}

impl VulkanAllocationReason {
    /// Stable order of the reason entries in an allocation census.
    pub const ALL: [Self; 5] = [
        Self::Texture,
        Self::RenderTarget,
        Self::Import,
        Self::Scratch,
        Self::Upload,
    ];

    fn index(self) -> usize {
        self as usize
    }
}

/// The caller-owned phase in which device memory was allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanAllocationPhase {
    /// No explicit scope covers this allocation, or thread scope storage overflowed.
    Unspecified,
    /// Renderer/device initialization.
    Initialization,
    /// Explicit resource preparation before frame work.
    Warmup,
    /// Frame preparation, recording, or submission.
    Frame,
    /// Explicit cleanup or background resource maintenance.
    Maintenance,
}

impl VulkanAllocationPhase {
    /// Stable order of the phase entries in an allocation census.
    pub const ALL: [Self; 5] = [
        Self::Unspecified,
        Self::Initialization,
        Self::Warmup,
        Self::Frame,
        Self::Maintenance,
    ];

    fn index(self) -> usize {
        self as usize
    }
}

/// Accounting for successful device-memory allocation owners.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VulkanAllocationStats {
    /// Currently live successful allocation owners.
    pub live_allocations: u64,
    /// Sum of their exact `VkMemoryAllocateInfo::allocation_size` values.
    pub live_bytes: u64,
    /// Successful allocations since this device ledger was created.
    pub total_allocations: u64,
    /// Exact bytes requested by those successful allocations.
    pub total_bytes: u64,
    /// Greatest observed number of live allocation bytes in this category.
    pub high_water_bytes: u64,
}

/// Read-only device-memory census. Imports count Vulkan bindings, not new
/// physical RAM: an external buffer may also be counted by its producer.
/// Driver-internal allocations are outside this explicit-allocation ledger.
/// After device loss, owner retirement does not prove that the driver freed
/// memory because the renderer deliberately suppresses unsafe destroy calls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VulkanAllocationSnapshot {
    /// Process-local identity matching the `device_ledger` TRACE field.
    pub device_ledger_id: u64,
    /// Allocation owners grouped by immutable allocation reason.
    pub by_reason: [VulkanAllocationStats; 5],
    /// Allocation owners grouped by the phase that created them.
    pub by_phase: [VulkanAllocationStats; 5],
}

impl VulkanAllocationSnapshot {
    /// Returns the counters for an allocation reason.
    pub fn reason(&self, reason: VulkanAllocationReason) -> VulkanAllocationStats {
        self.by_reason[reason.index()]
    }

    /// Returns the counters for the phase that created an allocation.
    pub fn phase(&self, phase: VulkanAllocationPhase) -> VulkanAllocationStats {
        self.by_phase[phase.index()]
    }
}

#[derive(Debug, Default)]
struct Counters {
    live_allocations: AtomicU64,
    live_bytes: AtomicU64,
    total_allocations: AtomicU64,
    total_bytes: AtomicU64,
    high_water_bytes: AtomicU64,
}

impl Counters {
    fn acquire(&self, bytes: u64) {
        self.live_allocations.fetch_add(1, Ordering::Relaxed);
        let live_bytes = self.live_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.high_water_bytes.fetch_max(live_bytes, Ordering::Relaxed);
    }

    fn release(&self, bytes: u64) {
        self.live_allocations.fetch_sub(1, Ordering::Relaxed);
        self.live_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    fn snapshot(&self) -> VulkanAllocationStats {
        VulkanAllocationStats {
            live_allocations: self.live_allocations.load(Ordering::Relaxed),
            live_bytes: self.live_bytes.load(Ordering::Relaxed),
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            total_bytes: self.total_bytes.load(Ordering::Relaxed),
            high_water_bytes: self.high_water_bytes.load(Ordering::Relaxed),
        }
    }
}

static NEXT_LEDGER_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_SCOPE_ID: AtomicU64 = AtomicU64::new(1);

const PHASE_SCOPE_CAPACITY: usize = 64;

#[derive(Debug, Clone, Copy)]
struct PhaseScope {
    ledger: u64,
    scope: u64,
    phase: VulkanAllocationPhase,
}

struct PhaseScopes {
    entries: [Option<PhaseScope>; PHASE_SCOPE_CAPACITY],
    overflow_scopes: usize,
}

impl PhaseScopes {
    const fn new() -> Self {
        Self {
            entries: [None; PHASE_SCOPE_CAPACITY],
            overflow_scopes: 0,
        }
    }

    /// Return true if there was no slot. Overflow cannot retain device/scope
    /// identities without allocating, so it suppresses all inferred phases
    /// on this thread until the last overflow guard retires.
    fn enter(&mut self, entry: PhaseScope) -> bool {
        if let Some(slot) = self.entries.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(entry);
            false
        } else {
            self.overflow_scopes = self.overflow_scopes.saturating_add(1);
            true
        }
    }

    fn phase(&self, ledger: u64) -> VulkanAllocationPhase {
        if self.overflow_scopes != 0 {
            return VulkanAllocationPhase::Unspecified;
        }
        // Slots are reused after out-of-order drops. Scope identity, rather
        // than array position, selects the newest surviving device scope.
        self.entries
            .iter()
            .flatten()
            .filter(|entry| entry.ledger == ledger)
            .max_by_key(|entry| entry.scope)
            .map(|entry| entry.phase)
            .unwrap_or(VulkanAllocationPhase::Unspecified)
    }

    fn leave(&mut self, ledger: u64, scope: u64, overflow: bool) {
        if overflow {
            // Saturation can only be reached after an impossible number of
            // simultaneously live guards, but keep that case conservative:
            // do not restore an inferred phase after losing the count.
            if self.overflow_scopes != usize::MAX {
                self.overflow_scopes = self.overflow_scopes.saturating_sub(1);
            }
        } else if let Some(slot) = self.entries.iter_mut().find(|slot| {
            slot.as_ref()
                .is_some_and(|entry| (entry.ledger, entry.scope) == (ledger, scope))
        }) {
            *slot = None;
        }
    }
}

thread_local! {
    static PHASES: RefCell<PhaseScopes> = const { RefCell::new(PhaseScopes::new()) };
}

/// One independent logical device's allocation counters.
#[derive(Debug)]
pub(super) struct AllocationLedger {
    id: u64,
    by_reason: [Counters; 5],
    by_phase: [Counters; 5],
}

impl Default for AllocationLedger {
    fn default() -> Self {
        Self {
            id: NEXT_LEDGER_ID.fetch_add(1, Ordering::Relaxed),
            by_reason: std::array::from_fn(|_| Counters::default()),
            by_phase: std::array::from_fn(|_| Counters::default()),
        }
    }
}

impl AllocationLedger {
    /// Register only after `vkAllocateMemory` succeeds, and move the returned
    /// guard into the actual memory owner. Drops happen after Vulkan teardown,
    /// outside any cache or allocation-custody mutex.
    pub(super) fn record(self: &Arc<Self>, reason: VulkanAllocationReason, bytes: u64) -> AllocationGuard {
        let phase = PHASES.with(|phases| phases.borrow().phase(self.id));
        self.by_reason[reason.index()].acquire(bytes);
        self.by_phase[phase.index()].acquire(bytes);
        let current_thread = thread::current();
        let thread = current_thread.id();
        trace!(
            device_ledger = self.id,
            ?reason,
            ?phase,
            ?thread,
            thread_name = current_thread.name().unwrap_or("unnamed"),
            allocation_bytes = bytes,
            "Vulkan device-memory allocation acquired"
        );
        AllocationGuard {
            ledger: self.clone(),
            reason,
            phase,
            bytes,
            thread,
        }
    }

    /// Atomic counters avoid adding a contended lock to allocation/free paths.
    /// A concurrent allocation or retirement can straddle snapshot reads; at a
    /// quiescent point the snapshot is exact. Reading it changes no counters.
    pub(super) fn snapshot(&self) -> VulkanAllocationSnapshot {
        VulkanAllocationSnapshot {
            device_ledger_id: self.id,
            by_reason: std::array::from_fn(|i| self.by_reason[i].snapshot()),
            by_phase: std::array::from_fn(|i| self.by_phase[i].snapshot()),
        }
    }

    pub(super) fn enter_phase(&self, phase: VulkanAllocationPhase) -> VulkanAllocationPhaseGuard {
        let scope = NEXT_SCOPE_ID.fetch_add(1, Ordering::Relaxed);
        let overflow = PHASES.with(|phases| {
            phases.borrow_mut().enter(PhaseScope {
                ledger: self.id,
                scope,
                phase,
            })
        });
        if overflow {
            trace!(
                device_ledger = self.id,
                ?scope,
                requested_phase = ?phase,
                effective_phase = ?VulkanAllocationPhase::Unspecified,
                ?PHASE_SCOPE_CAPACITY,
                thread = ?thread::current().id(),
                "Vulkan allocation phase scope capacity exceeded"
            );
        }
        VulkanAllocationPhaseGuard {
            ledger: self.id,
            scope,
            overflow,
            _thread: PhantomData,
        }
    }
}

/// A caller-owned phase scope for one device on the current thread. It holds
/// no renderer borrow, so callers can keep it around the operation they tag.
/// Nested devices remain independent, and dropping an outer scope early does
/// not erase a live inner scope. It must be dropped on the thread that enters it.
/// Scope storage is fixed at 64 entries per thread and never grows. While any
/// overflow guard is live, all devices on that thread use `Unspecified`;
/// existing scopes are restored when the final overflow guard is dropped.
#[derive(Debug)]
pub struct VulkanAllocationPhaseGuard {
    ledger: u64,
    scope: u64,
    overflow: bool,
    _thread: PhantomData<Rc<()>>,
}

impl Drop for VulkanAllocationPhaseGuard {
    fn drop(&mut self) {
        // A guard stored in another thread-local can be dropped after PHASES
        // was destroyed during thread teardown; its phase has already ended.
        let _ = PHASES.try_with(|phases| {
            phases.borrow_mut().leave(self.ledger, self.scope, self.overflow);
        });
    }
}

/// Non-cloneable custody for one successful Vulkan memory allocation. Sharing
/// the image/chunk via Arc shares this guard instead of recording another one.
#[derive(Debug)]
pub(super) struct AllocationGuard {
    ledger: Arc<AllocationLedger>,
    reason: VulkanAllocationReason,
    phase: VulkanAllocationPhase,
    bytes: u64,
    thread: ThreadId,
}

impl Drop for AllocationGuard {
    fn drop(&mut self) {
        self.ledger.by_reason[self.reason.index()].release(self.bytes);
        self.ledger.by_phase[self.phase.index()].release(self.bytes);
        trace!(
            device_ledger = self.ledger.id,
            reason = ?self.reason,
            phase = ?self.phase,
            allocated_thread = ?self.thread,
            retired_thread = ?thread::current().id(),
            allocation_bytes = self.bytes,
            "Vulkan device-memory allocation retired"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_and_submitted_owners_count_once_until_final_retirement() {
        let ledger = Arc::new(AllocationLedger::default());
        let image = Arc::new(ledger.record(VulkanAllocationReason::Import, 4096));
        let submitted = image.clone();
        drop(image);
        assert_eq!(
            ledger
                .snapshot()
                .reason(VulkanAllocationReason::Import)
                .live_bytes,
            4096
        );
        drop(submitted);
        let stats = ledger.snapshot().reason(VulkanAllocationReason::Import);
        assert_eq!(stats.live_bytes, 0);
        assert_eq!(stats.live_allocations, 0);
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_bytes, 4096);
        assert_eq!(stats.high_water_bytes, 4096);
    }

    #[test]
    fn successful_allocation_then_failed_binding_retires_its_guard() {
        let ledger = Arc::new(AllocationLedger::default());
        let guard = ledger.record(VulkanAllocationReason::Texture, 8192);
        assert_eq!(
            ledger
                .snapshot()
                .reason(VulkanAllocationReason::Texture)
                .live_allocations,
            1
        );
        drop(guard);
        assert_eq!(
            ledger
                .snapshot()
                .reason(VulkanAllocationReason::Texture)
                .live_allocations,
            0
        );
    }

    #[test]
    fn independent_devices_and_nested_phase_scopes_do_not_alias() {
        let a = Arc::new(AllocationLedger::default());
        let b = Arc::new(AllocationLedger::default());
        let outer = a.enter_phase(VulkanAllocationPhase::Initialization);
        let other = b.enter_phase(VulkanAllocationPhase::Warmup);
        let inner = a.enter_phase(VulkanAllocationPhase::Frame);
        drop(outer);
        let allocation = a.record(VulkanAllocationReason::RenderTarget, 1024);
        let other_allocation = b.record(VulkanAllocationReason::Upload, 2048);
        assert_eq!(allocation.phase, VulkanAllocationPhase::Frame);
        assert_eq!(other_allocation.phase, VulkanAllocationPhase::Warmup);
        drop((inner, other));
        let unscoped = a.record(VulkanAllocationReason::Scratch, 32);
        assert_eq!(unscoped.phase, VulkanAllocationPhase::Unspecified);
        assert_eq!(a.snapshot().phase(VulkanAllocationPhase::Frame).live_bytes, 1024);
        assert_eq!(b.snapshot().phase(VulkanAllocationPhase::Frame).live_bytes, 0);
    }

    #[test]
    fn phase_storage_is_fixed_and_reused_on_first_frame_worker() {
        let ledger = Arc::new(AllocationLedger::default());
        let initialized = ledger.enter_phase(VulkanAllocationPhase::Initialization);
        drop(initialized);
        std::thread::spawn(move || {
            // The device moved from its initialization thread. This worker's
            // first scope must not lazily create/grow a vector for frame work.
            let frame = ledger.enter_phase(VulkanAllocationPhase::Frame);
            assert_eq!(
                ledger.record(VulkanAllocationReason::Texture, 16).phase,
                VulkanAllocationPhase::Frame
            );
            let storage = PHASES.with(|phases| phases.borrow().entries.as_ptr());
            drop(frame);
            for _ in 0..128 {
                let scopes: [_; PHASE_SCOPE_CAPACITY] =
                    std::array::from_fn(|_| ledger.enter_phase(VulkanAllocationPhase::Frame));
                PHASES.with(|phases| {
                    let phases = phases.borrow();
                    assert_eq!(phases.entries.as_ptr(), storage);
                    assert_eq!(phases.entries.len(), PHASE_SCOPE_CAPACITY);
                    assert_eq!(phases.entries.iter().flatten().count(), PHASE_SCOPE_CAPACITY);
                    assert_eq!(phases.overflow_scopes, 0);
                });
                assert_eq!(
                    ledger.record(VulkanAllocationReason::Upload, 32).phase,
                    VulkanAllocationPhase::Frame
                );
                drop(scopes);
                PHASES.with(|phases| {
                    let phases = phases.borrow();
                    assert_eq!(phases.entries.as_ptr(), storage);
                    assert!(phases.entries.iter().all(Option::is_none));
                });
            }
        })
        .join()
        .unwrap();
    }

    #[test]
    fn overflow_suppresses_phases_until_final_guard_then_restores_newest_scope() {
        let a = Arc::new(AllocationLedger::default());
        let b = Arc::new(AllocationLedger::default());
        let mut scopes: [_; PHASE_SCOPE_CAPACITY] = std::array::from_fn(|i| {
            Some(if i % 2 == 0 {
                a.enter_phase(VulkanAllocationPhase::Frame)
            } else {
                b.enter_phase(VulkanAllocationPhase::Warmup)
            })
        });
        let overflow_a = a.enter_phase(VulkanAllocationPhase::Maintenance);
        let overflow_b = b.enter_phase(VulkanAllocationPhase::Frame);
        assert!(overflow_a.overflow);
        assert!(overflow_b.overflow);
        let check_phase = |ledger: &Arc<AllocationLedger>, expected| {
            assert_eq!(ledger.record(VulkanAllocationReason::Scratch, 16).phase, expected);
        };
        check_phase(&a, VulkanAllocationPhase::Unspecified);
        check_phase(&b, VulkanAllocationPhase::Unspecified);

        // Reuse the earliest slot while a newer scope remains active. Array
        // order must not resurrect that older scope, including after overflow.
        drop(scopes[0].take());
        let newest_a = a.enter_phase(VulkanAllocationPhase::Maintenance);
        assert!(!newest_a.overflow);
        drop(overflow_a);
        check_phase(&a, VulkanAllocationPhase::Unspecified);
        check_phase(&b, VulkanAllocationPhase::Unspecified);
        drop(scopes[1].take());
        let newest_b = b.enter_phase(VulkanAllocationPhase::Initialization);
        assert!(!newest_b.overflow);
        check_phase(&b, VulkanAllocationPhase::Unspecified);

        drop(overflow_b);
        check_phase(&a, VulkanAllocationPhase::Maintenance);
        check_phase(&b, VulkanAllocationPhase::Initialization);
        drop(newest_a);
        check_phase(&a, VulkanAllocationPhase::Frame);
        check_phase(&b, VulkanAllocationPhase::Initialization);
        drop(newest_b);
        check_phase(&b, VulkanAllocationPhase::Warmup);
        drop(scopes);
        check_phase(&a, VulkanAllocationPhase::Unspecified);
        check_phase(&b, VulkanAllocationPhase::Unspecified);
    }

    #[test]
    fn phase_belongs_to_allocating_thread_and_guard_can_retire_elsewhere() {
        let ledger = Arc::new(AllocationLedger::default());
        let _phase = ledger.enter_phase(VulkanAllocationPhase::Frame);
        let current = ledger.record(VulkanAllocationReason::Texture, 128);
        let other_ledger = ledger.clone();
        let other = std::thread::spawn(move || {
            let unscoped = other_ledger.record(VulkanAllocationReason::Upload, 256);
            assert_eq!(unscoped.phase, VulkanAllocationPhase::Unspecified);
            drop(current);
            unscoped
        })
        .join()
        .unwrap();
        assert_eq!(
            ledger.snapshot().phase(VulkanAllocationPhase::Frame).live_bytes,
            0
        );
        drop(other);
        assert_eq!(
            ledger
                .snapshot()
                .reason(VulkanAllocationReason::Upload)
                .live_bytes,
            0
        );
    }

    #[test]
    fn concurrent_allocation_and_retirement_preserve_cumulative_totals() {
        let ledger = Arc::new(AllocationLedger::default());
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let ledger = ledger.clone();
                std::thread::spawn(move || {
                    let _phase = ledger.enter_phase(VulkanAllocationPhase::Frame);
                    for _ in 0..128 {
                        let guard = ledger.record(VulkanAllocationReason::Texture, 4096);
                        drop(guard);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        let stats = ledger.snapshot().reason(VulkanAllocationReason::Texture);
        assert_eq!(stats.live_allocations, 0);
        assert_eq!(stats.live_bytes, 0);
        assert_eq!(stats.total_allocations, 512);
        assert_eq!(stats.total_bytes, 512 * 4096);
        let phase = ledger.snapshot().phase(VulkanAllocationPhase::Frame);
        assert_eq!(phase.live_allocations, 0);
        assert_eq!(phase.live_bytes, 0);
        assert_eq!(phase.total_allocations, stats.total_allocations);
        assert_eq!(phase.total_bytes, stats.total_bytes);
        assert!((4096..=4 * 4096).contains(&stats.high_water_bytes));
        assert!((4096..=4 * 4096).contains(&phase.high_water_bytes));
    }

    #[test]
    fn every_reason_counts_exact_bytes_and_snapshot_is_read_only() {
        let ledger = Arc::new(AllocationLedger::default());
        let allocations: Vec<_> = VulkanAllocationReason::ALL
            .into_iter()
            .map(|reason| ledger.record(reason, 4097))
            .collect();
        let before = ledger.snapshot();
        assert_eq!(before, ledger.snapshot());
        for reason in VulkanAllocationReason::ALL {
            assert_eq!(before.reason(reason).live_bytes, 4097);
            assert_eq!(before.reason(reason).live_allocations, 1);
        }
        drop(allocations);
        for reason in VulkanAllocationReason::ALL {
            assert_eq!(ledger.snapshot().reason(reason).live_bytes, 0);
        }
    }
}
