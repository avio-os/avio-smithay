//! Immutable damage blobs with cold-reserved CPU backing and reader ownership.
use super::*;
use crate::backend::renderer::element::{FrameWorkspaceError, StateReceiptReturnWakeup};
use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize},
    mpsc::{self, Receiver, SyncSender},
    Mutex,
};

static CONSERVATIVE_CLIP_UNIONS: AtomicU64 = AtomicU64::new(0);
/// Process-lifetime streamed clip requests coalesced after actual overflow
/// under cold-admitted conservative repair. Counts collection, including any
/// subsequent blob-ioctl failure; it is not a commit/presentation receipt.
pub fn conservative_damage_clip_unions() -> u64 {
    CONSERVATIVE_CLIP_UNIONS.load(Ordering::Relaxed)
}

#[derive(Debug)]
struct LegacyDamage {
    drm: DrmDeviceFd,
    blob: u64,
}
impl Drop for LegacyDamage {
    fn drop(&mut self) {
        let _ = self.drm.destroy_property_blob(self.blob);
    }
}
#[derive(Debug)]
struct Slot {
    blob: AtomicU32,
    payload: Mutex<Payload>,
}
#[derive(Debug)]
struct Payload {
    drm: Option<DrmDeviceFd>,
    rectangles: Vec<drm_ffi::drm_mode_rect>,
}
#[derive(Debug)]
enum DamageOwner {
    Legacy(Arc<LegacyDamage>),
    Pooled { slot: Arc<Slot>, wake: SyncSender<()> },
}

/// Immutable `FB_DAMAGE_CLIPS` owner retained by every real plane configuration.
#[derive(Debug)]
pub struct PlaneDamageClips {
    owner: Option<DamageOwner>,
}
impl PlaneDamageClips {
    /// The actual immutable native property blob belonging to this reader.
    pub fn blob(&self) -> drm::control::property::Value<'_> {
        let id = match self.owner.as_ref().expect("live damage reader") {
            DamageOwner::Legacy(owner) => owner.blob,
            DamageOwner::Pooled { slot, .. } => u64::from(slot.blob.load(Ordering::Acquire)),
        };
        drm::control::property::Value::Blob(id)
    }
    /// Allocating compatibility path for callers without cold-admitted storage.
    pub fn from_damage(
        device: &DrmDeviceFd,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: impl IntoIterator<Item = Rectangle<i32, Physical>>,
    ) -> io::Result<Option<Self>> {
        let mut rectangles: Vec<_> = damage
            .into_iter()
            .map(|rect| mapped_rect(src, dst, rect))
            .collect();
        let Some(blob) = create_blob(device, &mut rectangles)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            owner: Some(DamageOwner::Legacy(Arc::new(LegacyDamage {
                drm: device.clone(),
                blob: u64::from(blob),
            }))),
        }))
    }
}
impl Clone for PlaneDamageClips {
    fn clone(&self) -> Self {
        let owner = match self.owner.as_ref().expect("live damage reader") {
            DamageOwner::Legacy(owner) => DamageOwner::Legacy(owner.clone()),
            DamageOwner::Pooled { slot, wake } => DamageOwner::Pooled {
                slot: slot.clone(),
                wake: wake.clone(),
            },
        };
        Self { owner: Some(owner) }
    }
}
impl Drop for PlaneDamageClips {
    fn drop(&mut self) {
        let Some(owner) = self.owner.take() else {
            return;
        };
        match owner {
            DamageOwner::Legacy(owner) => drop(owner),
            DamageOwner::Pooled { slot, wake } => {
                // Decrement the actual reader before signaling. Concurrent
                // final drops each wake, so a scan cannot miss the last edge.
                drop(slot);
                let _ = wake.try_send(());
            }
        }
    }
}

fn mapped_rect(
    src: Rectangle<f64, Buffer>,
    dst: Rectangle<i32, Physical>,
    rect: Rectangle<i32, Physical>,
) -> drm_ffi::drm_mode_rect {
    let scale = src.size / dst.size.to_logical(1).to_buffer(1, Transform::Normal).to_f64();
    let mut rect = rect
        .to_f64()
        .to_logical(1f64)
        .to_buffer(
            1f64,
            Transform::Normal,
            &src.size.to_logical(1f64, Transform::Normal),
        )
        .upscale(scale);
    rect.loc += src.loc;
    let rect = rect.to_i32_up();
    drm_ffi::drm_mode_rect {
        x1: rect.loc.x,
        y1: rect.loc.y,
        x2: rect.loc.x.saturating_add(rect.size.w),
        y2: rect.loc.y.saturating_add(rect.size.h),
    }
}
fn create_blob(device: &DrmDeviceFd, rectangles: &mut [drm_ffi::drm_mode_rect]) -> io::Result<Option<u32>> {
    if rectangles.is_empty() {
        return Ok(None);
    }
    // Native drm_mode_rect is the ABI POD array expected by CREATEPROPBLOB.
    let data = unsafe {
        std::slice::from_raw_parts_mut(
            rectangles.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(rectangles),
        )
    };
    Ok(Some(
        drm_ffi::mode::create_property_blob(device.as_fd(), data)?.blob_id,
    ))
}

fn fill_damage(
    rectangles: &mut Vec<drm_ffi::drm_mode_rect>,
    capacity: usize,
    conservative: bool,
    src: Rectangle<f64, Buffer>,
    dst: Rectangle<i32, Physical>,
    produce: impl FnOnce(&mut dyn FnMut(Rectangle<i32, Physical>)),
) -> Result<(), FrameWorkspaceError> {
    rectangles.clear();
    let mut coalesced = false;
    crate::backend::renderer::damage::workspace::try_visit(produce, |rect| {
        let mapped = mapped_rect(src, dst, rect);
        if coalesced {
            merge_rect(&mut rectangles[0], mapped);
        } else if rectangles.len() < capacity && rectangles.len() < rectangles.capacity() {
            rectangles.push(mapped);
        } else if conservative && !rectangles.is_empty() {
            // This policy was selected cold. Only the advisory native hint
            // expands: every source/renderer damage rectangle stays exact.
            let mut union = mapped;
            for existing in rectangles.iter().copied() {
                merge_rect(&mut union, existing);
            }
            rectangles.clear();
            rectangles.push(union);
            coalesced = true;
            CONSERVATIVE_CLIP_UNIONS.fetch_add(1, Ordering::Relaxed);
        } else {
            return Err(FrameWorkspaceError {
                resource: "DRM damage clip rectangles",
                required: rectangles.len().saturating_add(1),
                capacity,
            });
        }
        Ok(())
    })
}
fn merge_rect(union: &mut drm_ffi::drm_mode_rect, rect: drm_ffi::drm_mode_rect) {
    union.x1 = union.x1.min(rect.x1);
    union.y1 = union.y1.min(rect.y1);
    union.x2 = union.x2.max(rect.x2);
    union.y2 = union.y2.max(rect.y2);
}

#[derive(Debug)]
struct PoolState {
    roots: Vec<Arc<Slot>>,
    waiting: Mutex<Vec<Arc<Slot>>>,
    returned: Mutex<Receiver<Arc<Slot>>>,
    ready: SyncSender<Arc<Slot>>,
    available: Arc<AtomicUsize>,
    closed: AtomicBool,
    quarantined: AtomicBool,
    notification: Option<StateReceiptReturnWakeup>,
}
#[derive(Debug)]
struct DamageRegistry(Arc<PoolState>);
impl super::blob_actor::BlobRegistryRoot for DamageRegistry {
    fn reconcile(&self) -> bool {
        let pool = &self.0;
        reconcile(pool);
        !pool.closed.load(Ordering::Acquire)
            || pool.quarantined.load(Ordering::Acquire)
            || pool.available.load(Ordering::Acquire) != pool.roots.len()
            || Arc::strong_count(pool) != 1
    }
}

fn reconcile(pool: &PoolState) {
    let returned = pool.returned.lock().unwrap_or_else(|p| p.into_inner());
    let mut waiting = pool.waiting.lock().unwrap_or_else(|p| p.into_inner());
    while let Ok(slot) = returned.try_recv() {
        // Every slot has one owner distributed among ready/claim/return/wait.
        // Capacity was reserved cold for all slots; this never grows.
        debug_assert!(waiting.len() < waiting.capacity());
        waiting.push(slot);
    }
    let mut index = 0;
    while index < waiting.len() {
        if Arc::strong_count(&waiting[index]) != 2 {
            index += 1;
            continue;
        }
        let slot = &waiting[index];
        let mut payload = slot.payload.lock().unwrap_or_else(|p| p.into_inner());
        let blob = slot.blob.load(Ordering::Acquire);
        if blob != 0 {
            let result = payload
                .drm
                .as_ref()
                .expect("native blob retains its exact FD")
                .destroy_property_blob(u64::from(blob));
            if result.is_err() {
                pool.quarantined.store(true, Ordering::Release);
                index += 1;
                continue;
            }
            slot.blob.store(0, Ordering::Release);
        }
        payload.rectangles.clear();
        payload.drm = None;
        drop(payload);
        let slot = waiting.swap_remove(index);
        // Publish availability before the ready owner, avoiding an underflow
        // if a concurrent claimant consumes the newly enqueued slot at once.
        pool.available.fetch_add(1, Ordering::Release);
        match pool.ready.try_send(slot) {
            Ok(()) => {}
            Err(mpsc::TrySendError::Disconnected(slot)) if pool.closed.load(Ordering::Acquire) => drop(slot),
            Err(mpsc::TrySendError::Full(slot) | mpsc::TrySendError::Disconnected(slot)) => {
                // Preserve conservation even after an impossible transport failure.
                waiting.push(slot);
                pool.available.fetch_sub(1, Ordering::AcqRel);
                pool.quarantined.store(true, Ordering::Release);
                index += 1;
                continue;
            }
        }
        if let Some(callback) = pool
            .notification
            .as_ref()
            .and_then(StateReceiptReturnWakeup::notification)
        {
            callback();
        }
    }
}

/// Each warm reference releases the state before its final wake. The actor
/// keeps the last registry root until all such handles are gone, so even pool
/// metadata and std channel backing have their final free on the cold actor.
#[derive(Debug, Clone)]
struct PoolHandle {
    state: Option<Arc<PoolState>>,
    wake: SyncSender<()>,
}
impl std::ops::Deref for PoolHandle {
    type Target = PoolState;
    fn deref(&self) -> &Self::Target {
        self.state.as_deref().expect("live pool handle")
    }
}
impl Drop for PoolHandle {
    fn drop(&mut self) {
        drop(self.state.take());
        let _ = self.wake.try_send(());
    }
}

/// Numeric and immutable-owner pool. Only cold factories construct it.
#[derive(Debug)]
pub(crate) struct PlaneDamageClipBank {
    ready: Mutex<Option<Receiver<Arc<Slot>>>>,
    returned: SyncSender<Arc<Slot>>,
    wake: SyncSender<()>,
    rectangle_capacity: usize,
    conservative: bool,
    // Declared last: transport handles drop before the state-reference wake.
    state: PoolHandle,
}
impl PlaneDamageClipBank {
    pub(crate) fn owner_slots(planes: usize) -> Option<usize> {
        planes.checked_mul(4)?.checked_add(1)
    }
    pub(crate) fn bytes_per_rectangle(planes: usize) -> Option<usize> {
        Self::owner_slots(planes)?.checked_mul(std::mem::size_of::<drm_ffi::drm_mode_rect>())
    }
    pub(crate) fn fixed_bytes(planes: usize) -> Option<usize> {
        Self::owner_slots(planes)?
            .checked_mul(std::mem::size_of::<Slot>() + std::mem::size_of::<Arc<Slot>>() * 4)?
            .checked_add(std::mem::size_of::<PoolState>())
    }
    pub(crate) fn new(
        planes: usize,
        rectangles: usize,
        conservative: bool,
        notification: Option<StateReceiptReturnWakeup>,
    ) -> Result<Self, FrameWorkspaceError> {
        let slots = Self::owner_slots(planes).ok_or(FrameWorkspaceError {
            resource: "DRM damage blob owners",
            required: usize::MAX,
            capacity: planes,
        })?;
        let (ready_tx, ready) = mpsc::sync_channel(slots);
        let (returned, returned_rx) = mpsc::sync_channel(slots);
        let mut roots = Vec::with_capacity(slots);
        for _ in 0..slots {
            let slot = Arc::new(Slot {
                blob: AtomicU32::new(0),
                payload: Mutex::new(Payload {
                    drm: None,
                    rectangles: Vec::with_capacity(rectangles),
                }),
            });
            roots.push(slot.clone());
            ready_tx.try_send(slot).expect("cold ready queue has every slot");
        }
        let state = Arc::new(PoolState {
            roots,
            waiting: Mutex::new(Vec::with_capacity(slots)),
            returned: Mutex::new(returned_rx),
            ready: ready_tx,
            available: Arc::new(AtomicUsize::new(slots)),
            closed: AtomicBool::new(false),
            quarantined: AtomicBool::new(false),
            notification,
        });
        let wake = super::blob_actor::register(DamageRegistry(state.clone()))?;
        Ok(Self {
            state: PoolHandle {
                state: Some(state),
                wake: wake.clone(),
            },
            ready: Mutex::new(Some(ready)),
            returned,
            wake: wake.clone(),
            rectangle_capacity: rectangles,
            conservative,
        })
    }
    pub(crate) fn is_reclaimable(&self) -> bool {
        !self.state.quarantined.load(Ordering::Acquire)
            && self.state.available.load(Ordering::Acquire) == self.state.roots.len()
    }
    pub(crate) fn try_claim(&self) -> Result<PlaneDamageWriter, FrameWorkspaceError> {
        let error = || FrameWorkspaceError {
            resource: "DRM damage blob owners",
            required: self.state.roots.len().saturating_add(1),
            capacity: self.state.roots.len(),
        };
        let receiver = self.ready.try_lock().map_err(|_| error())?;
        if self.state.quarantined.load(Ordering::Acquire) {
            return Err(error());
        }
        let slot = receiver
            .as_ref()
            .ok_or_else(error)?
            .try_recv()
            .map_err(|_| error())?;
        self.state.available.fetch_sub(1, Ordering::AcqRel);
        Ok(PlaneDamageWriter {
            slot: Some(slot),
            state: self.state.clone(),
            returned: self.returned.clone(),
            wake: self.wake.clone(),
            rectangle_capacity: self.rectangle_capacity,
            conservative: self.conservative,
        })
    }
}
impl Drop for PlaneDamageClipBank {
    fn drop(&mut self) {
        // Drain ready owners before the actor is allowed to release its roots.
        // No Slot is final here: the registered cold roots still retain each.
        drop(self.ready.get_mut().unwrap_or_else(|p| p.into_inner()).take());
        self.state.closed.store(true, Ordering::Release);
        let _ = self.wake.try_send(());
    }
}

#[derive(Debug)]
pub(crate) struct PlaneDamageWriter {
    slot: Option<Arc<Slot>>,
    returned: SyncSender<Arc<Slot>>,
    wake: SyncSender<()>,
    rectangle_capacity: usize,
    conservative: bool,
    // As with the bank, this must drop after every per-pool transport handle.
    state: PoolHandle,
}
impl PlaneDamageWriter {
    /// Create the exact accepted-generation blob. Capacity failure precedes
    /// the native ioctl; this never publishes a partial damage list.
    pub(crate) fn create(
        &mut self,
        device: &DrmDeviceFd,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        produce: impl FnOnce(&mut dyn FnMut(Rectangle<i32, Physical>)),
    ) -> Result<io::Result<Option<PlaneDamageClips>>, FrameWorkspaceError> {
        let slot = self.slot.as_ref().expect("unconsumed blob writer");
        let mut payload = slot.payload.try_lock().map_err(|_| FrameWorkspaceError {
            resource: "DRM damage scratch is owned",
            required: 1,
            capacity: 0,
        })?;
        fill_damage(
            &mut payload.rectangles,
            self.rectangle_capacity,
            self.conservative,
            src,
            dst,
            produce,
        )?;
        let blob = match create_blob(device, &mut payload.rectangles) {
            Ok(Some(blob)) => blob,
            other => return Ok(other.map(|_| None)),
        };
        payload.drm = Some(device.clone());
        slot.blob.store(blob, Ordering::Release);
        let reader = PlaneDamageClips {
            owner: Some(DamageOwner::Pooled {
                slot: slot.clone(),
                wake: self.wake.clone(),
            }),
        };
        drop(payload);
        self.return_slot();
        Ok(Ok(Some(reader)))
    }
    fn return_slot(&mut self) {
        if let Some(slot) = self.slot.take() {
            // Each finite slot has exactly one transferable owner. The bounded
            // return queue cannot be full under that conservation invariant.
            if let Err(error) = self.returned.try_send(slot) {
                // Preserve exact unknown owner instead of freeing on this lane.
                self.state.quarantined.store(true, Ordering::Release);
                let slot = match error {
                    mpsc::TrySendError::Full(slot) | mpsc::TrySendError::Disconnected(slot) => slot,
                };
                std::mem::forget(slot);
            }
            let _ = self.wake.try_send(());
        }
    }
}
impl Drop for PlaneDamageWriter {
    fn drop(&mut self) {
        self.return_slot();
    }
}

#[cfg(test)]
#[path = "plane_damage_tests.rs"]
mod tests;
