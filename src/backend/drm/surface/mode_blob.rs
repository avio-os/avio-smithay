//! KMS mode blobs owned by the atomic surface state that names them.
//!
//! `MODE_ID` takes a property blob. The kernel keeps every blob a file
//! creates on that file's blob list until the file destroys it or closes, and
//! each `DESTROYPROPBLOB` walks the list linearly. A blob nothing destroys is
//! therefore not only leaked memory: it slows every later blob destroy on the
//! same file, including the per-frame `FB_DAMAGE_CLIPS` blob.
//!
//! A [`ModeBlob`] retires its blob when the last state that names it is
//! dropped; the existing DRM blob actor destroys it outside render/input paths. The current and pending states of an atomic surface share one blob
//! after a commit, so replacing either state releases the blob exactly when
//! neither names it any more. The kernel holds its own reference to a blob a
//! committed CRTC state uses, so destroying our handle never disturbs the
//! active mode.

use std::fmt;
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use drm::control::{property, Device as ControlDevice, Mode};
use tracing::warn;

use crate::backend::drm::error::{AccessError, Error};
use crate::utils::DevPath;

static LIVE_MODE_BLOBS: AtomicUsize = AtomicUsize::new(0);

/// Number of KMS mode blobs this process's atomic DRM surfaces currently own.
///
/// Includes active, pending, validated, retired and quarantined native blobs.
/// One creating DRM file admits four owner slots per actual CRTC; recreating a
/// surface reuses that inventory. Unknown native destruction retains its slot
/// and accounting, and cannot mint an unbounded replacement history.
pub fn live_mode_blobs() -> usize {
    LIVE_MODE_BLOBS.load(Ordering::Relaxed)
}

/// The file that created a blob, and must destroy it.
trait BlobOwner: fmt::Debug + Send + Sync {
    fn destroy_blob(&self, id: u64) -> io::Result<()>;
}

impl<D: ControlDevice + fmt::Debug + Send + Sync> BlobOwner for D {
    fn destroy_blob(&self, id: u64) -> io::Result<()> {
        self.destroy_property_blob(id)
    }
}

const MODE_OWNER_SLOTS: usize = 4;

struct ModePayload {
    owner: Box<dyn BlobOwner>,
    id: u64,
}
impl fmt::Debug for ModePayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The creating file owns this inventory; formatting its child owner
        // would recurse back through that same file and inventory.
        f.debug_struct("ModePayload")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}
#[derive(Debug)]
struct ModeSlot {
    id: std::sync::atomic::AtomicU64,
    payload: std::sync::Mutex<Option<ModePayload>>,
    quarantined: std::sync::atomic::AtomicBool,
}
/// Registered cold with the existing DRM blob actor. Each root is reusable
/// only after its exact readers are gone and native destruction succeeded.
#[derive(Debug)]
pub(super) struct ModePool {
    slots: Vec<Arc<ModeSlot>>,
    closed: std::sync::atomic::AtomicBool,
}
impl ModePool {
    fn reconcile(&self) -> bool {
        for slot in &self.slots {
            // Empty roots need no native reconciliation. Do not contend with
            // cold admission on a reusable root merely because another reader
            // published an unrelated retirement wake.
            if slot.id.load(Ordering::Acquire) == 0 || Arc::strong_count(slot) != 1 {
                continue;
            }
            let Ok(mut payload) = slot.payload.try_lock() else {
                continue;
            };
            if Arc::strong_count(slot) != 1 || slot.quarantined.load(Ordering::Acquire) {
                continue;
            }
            if let Some(native) = payload.as_ref() {
                if let Err(err) = native.owner.destroy_blob(native.id) {
                    warn!(blob = native.id, "Quarantining undestroyed mode blob: {}", err);
                    slot.quarantined.store(true, Ordering::Release);
                    continue;
                }
                LIVE_MODE_BLOBS.fetch_sub(1, Ordering::Relaxed);
                slot.id.store(0, Ordering::Release);
                // The exact creating FD and payload are disposed on this actor.
                *payload = None;
            }
        }
        !self.closed.load(Ordering::Acquire)
            || self
                .slots
                .iter()
                .any(|slot| Arc::strong_count(slot) != 1 || slot.id.load(Ordering::Acquire) != 0)
    }
}
#[derive(Debug)]
struct ModeRegistry(Arc<ModePool>);
impl super::blob_actor::BlobRegistryRoot for ModeRegistry {
    fn reconcile(&self) -> bool {
        self.0.reconcile() || Arc::strong_count(&self.0) != 1
    }
}
/// Two active states, one validated candidate, one completed retirement edge.
/// Extra outstanding candidates or quarantined history refuse before CREATE.
#[derive(Debug)]
pub(in crate::backend::drm) struct ModeBlobBank {
    pool: Option<Arc<ModePool>>,
    wake: std::sync::mpsc::SyncSender<()>,
    parent: bool,
}
impl ModeBlobBank {
    pub(super) fn cold() -> Result<Self, Error> {
        let pool = Arc::new(ModePool {
            slots: (0..MODE_OWNER_SLOTS)
                .map(|_| {
                    Arc::new(ModeSlot {
                        id: std::sync::atomic::AtomicU64::new(0),
                        payload: std::sync::Mutex::new(None),
                        quarantined: std::sync::atomic::AtomicBool::new(false),
                    })
                })
                .collect(),
            closed: std::sync::atomic::AtomicBool::new(false),
        });
        let wake =
            super::blob_actor::register(ModeRegistry(pool.clone())).map_err(|_| Error::ModeBlobCapacity {
                capacity: MODE_OWNER_SLOTS,
            })?;
        Ok(Self {
            pool: Some(pool),
            wake,
            parent: true,
        })
    }
    fn reserve(&self) -> Result<ModeBlob, Error> {
        let pool = self.pool.as_ref().ok_or(Error::DeviceInactive)?;
        if pool.closed.load(Ordering::Acquire) {
            return Err(Error::DeviceInactive);
        }
        for slot in &pool.slots {
            let Ok(payload) = slot.payload.try_lock() else {
                continue;
            };
            if payload.is_none() && !slot.quarantined.load(Ordering::Acquire) && Arc::strong_count(slot) == 1
            {
                let blob = ModeBlob {
                    slot: Some(slot.clone()),
                    wake: self.wake.clone(),
                };
                drop(payload);
                return Ok(blob);
            }
        }
        Err(Error::ModeBlobCapacity {
            capacity: MODE_OWNER_SLOTS,
        })
    }
}
impl Clone for ModeBlobBank {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            wake: self.wake.clone(),
            parent: false,
        }
    }
}
/// Exact CRTC inventory of one creating-file Arc, preserved across surface generations.
#[derive(Debug)]
pub(in crate::backend::drm) struct ModeFileBanks {
    banks: Vec<(drm::control::crtc::Handle, ModeBlobBank)>,
}
impl ModeFileBanks {
    pub(in crate::backend::drm) fn cold(crtcs: &[drm::control::crtc::Handle]) -> Result<Self, Error> {
        let mut banks = Vec::with_capacity(crtcs.len());
        for crtc in crtcs {
            banks.push((*crtc, ModeBlobBank::cold()?));
        }
        banks.sort_unstable_by_key(|entry| u32::from(entry.0));
        banks.dedup_by_key(|entry| entry.0);
        Ok(Self { banks })
    }
    pub(in crate::backend::drm) fn contains(&self, crtc: drm::control::crtc::Handle) -> bool {
        self.banks
            .binary_search_by_key(&u32::from(crtc), |entry| u32::from(entry.0))
            .is_ok()
    }
    pub(in crate::backend::drm) fn lease(
        &self,
        crtc: drm::control::crtc::Handle,
    ) -> Result<ModeBlobBank, Error> {
        let index = self
            .banks
            .binary_search_by_key(&u32::from(crtc), |entry| u32::from(entry.0))
            .map_err(|_| Error::UnknownCrtc(crtc))?;
        Ok(self.banks[index].1.clone())
    }
}
impl Drop for ModeBlobBank {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.take() {
            if self.parent {
                pool.closed.store(true, Ordering::Release);
            }
            drop(pool);
            let _ = self.wake.try_send(());
        }
    }
}
/// A reader of one native mode blob. The cold actor retains the final root.
#[derive(Clone)]
pub(crate) struct ModeBlob {
    slot: Option<Arc<ModeSlot>>,
    wake: std::sync::mpsc::SyncSender<()>,
}
impl fmt::Debug for ModeBlob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ModeBlob").field(&self.id()).finish()
    }
}
impl Drop for ModeBlob {
    fn drop(&mut self) {
        drop(self.slot.take());
        let _ = self.wake.try_send(());
    }
}
impl ModeBlob {
    pub(super) fn new_in<D>(bank: &ModeBlobBank, device: &D, mode: &Mode) -> Result<Self, Error>
    where
        D: ControlDevice + DevPath + Clone + fmt::Debug + Send + Sync + 'static,
    {
        // Admission is before both the native blob and its creating-device owner.
        let blob = bank.reserve()?;
        let value = device.create_property_blob(mode).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Failed to create Property Blob for mode",
                dev: device.dev_path(),
                source,
            })
        })?;
        let slot = blob.slot.as_ref().expect("reserved mode reader");
        let id = value.into();
        *slot.payload.lock().unwrap_or_else(|p| p.into_inner()) = Some(ModePayload {
            owner: Box::new(device.clone()),
            id,
        });
        LIVE_MODE_BLOBS.fetch_add(1, Ordering::Relaxed);
        slot.id.store(id, Ordering::Release);
        Ok(blob)
    }
    #[cfg(test)]
    pub(crate) fn new<D>(device: &D, mode: &Mode) -> Result<Self, Error>
    where
        D: ControlDevice + DevPath + Clone + fmt::Debug + Send + Sync + 'static,
    {
        Self::new_in(&ModeBlobBank::cold()?, device, mode)
    }
    pub(crate) fn value(&self) -> property::Value<'static> {
        property::Value::Blob(self.id())
    }
    pub(crate) fn same_owner(&self, other: &Self) -> bool {
        match (&self.slot, &other.slot) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
    pub(crate) fn id(&self) -> u64 {
        self.slot
            .as_ref()
            .expect("live mode reader")
            .id
            .load(Ordering::Acquire)
    }
}

#[cfg(test)]
pub(crate) mod test_device {
    //! A [`ControlDevice`] whose blob calls are recorded instead of reaching a
    //! kernel. Every other `ControlDevice` call would reach `/dev/null` and
    //! fail, so a test that passes never touched a real device.

    use std::collections::HashMap;
    use std::fs::File;
    use std::io;
    use std::os::unix::io::{AsFd, BorrowedFd};
    use std::sync::{Arc, Mutex, MutexGuard};

    use drm::control::{property, Device as ControlDevice};

    /// Serializes the tests that read the process-wide blob counter.
    pub(crate) fn serial() -> MutexGuard<'static, ()> {
        static SERIAL: Mutex<()> = Mutex::new(());
        let guard = SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        super::super::blob_actor::drain();
        guard
    }

    #[derive(Debug, Default)]
    struct Ledger {
        next_id: u64,
        destroys: HashMap<u64, usize>,
        created: usize,
        threads: Vec<std::thread::ThreadId>,
        fail_destroy: bool,
    }

    /// A device that hands out blob ids and counts each id's destroys.
    #[derive(Debug, Clone)]
    pub(crate) struct BlobLedgerDevice {
        null: Arc<File>,
        ledger: Arc<Mutex<Ledger>>,
    }

    impl BlobLedgerDevice {
        pub(crate) fn new() -> Self {
            BlobLedgerDevice {
                null: Arc::new(File::open("/dev/null").expect("open /dev/null")),
                ledger: Arc::new(Mutex::new(Ledger {
                    next_id: 1,
                    ..Ledger::default()
                })),
            }
        }

        pub(crate) fn destruction_threads(&self) -> Vec<std::thread::ThreadId> {
            super::super::blob_actor::drain();
            self.ledger.lock().unwrap().threads.clone()
        }
        pub(crate) fn fail_destroy(&self) {
            self.ledger.lock().unwrap().fail_destroy = true;
        }
        pub(crate) fn created(&self) -> usize {
            self.ledger.lock().unwrap().created
        }
        /// How many times `id` was destroyed.
        pub(crate) fn destroys(&self, id: u64) -> usize {
            super::super::blob_actor::drain();
            self.ledger
                .lock()
                .unwrap()
                .destroys
                .get(&id)
                .copied()
                .unwrap_or(0)
        }

        /// Blobs created and not yet destroyed on this device.
        pub(crate) fn alive(&self) -> usize {
            super::super::blob_actor::drain();
            let ledger = self.ledger.lock().unwrap();
            ledger.created - ledger.destroys.values().sum::<usize>()
        }
    }

    impl AsFd for BlobLedgerDevice {
        fn as_fd(&self) -> BorrowedFd<'_> {
            self.null.as_fd()
        }
    }

    impl drm::Device for BlobLedgerDevice {}

    impl ControlDevice for BlobLedgerDevice {
        fn create_property_blob<T>(&self, _data: &T) -> io::Result<property::Value<'static>> {
            let mut ledger = self.ledger.lock().unwrap();
            let id = ledger.next_id;
            ledger.next_id += 1;
            ledger.created += 1;
            Ok(property::Value::Blob(id))
        }

        fn destroy_property_blob(&self, blob: u64) -> io::Result<()> {
            let mut ledger = self.ledger.lock().unwrap();
            ledger.threads.push(std::thread::current().id());
            if ledger.fail_destroy {
                return Err(io::Error::from_raw_os_error(5));
            }
            *ledger.destroys.entry(blob).or_default() += 1;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_device::{serial, BlobLedgerDevice};
    use super::{live_mode_blobs, ModeBlob, ModeBlobBank, MODE_OWNER_SLOTS};
    use crate::backend::drm::error::Error;
    use drm::control::Mode;

    fn mode() -> Mode {
        // SAFETY: drm_mode_modeinfo is plain old data; a zeroed mode is valid.
        unsafe { std::mem::zeroed() }
    }

    #[test]
    fn the_last_clone_destroys_the_blob_once() {
        let _serial = serial();
        let start = live_mode_blobs();
        let device = BlobLedgerDevice::new();
        let blob = ModeBlob::new(&device, &mode()).unwrap();
        let id = blob.id();
        let shared = blob.clone();
        assert_eq!(live_mode_blobs(), start + 1);
        drop(blob);
        assert_eq!(device.destroys(id), 0);
        drop(shared);
        assert_eq!(device.destroys(id), 1);
        assert_eq!(live_mode_blobs(), start);
    }

    #[test]
    fn the_value_names_the_created_blob() {
        let _serial = serial();
        let device = BlobLedgerDevice::new();
        let blob = ModeBlob::new(&device, &mode()).unwrap();
        assert_eq!(blob.value(), drm::control::property::Value::Blob(blob.id()));
    }
    #[test]
    fn final_reader_drop_runs_destroy_on_existing_actor() {
        let _serial = serial();
        let device = BlobLedgerDevice::new();
        let bank = ModeBlobBank::cold().unwrap();
        let blob = ModeBlob::new_in(&bank, &device, &mode()).unwrap();
        let id = blob.id();
        let dropped_on = std::thread::spawn(move || {
            let thread = std::thread::current().id();
            drop(blob);
            thread
        })
        .join()
        .unwrap();
        assert_eq!(device.destroys(id), 1);
        assert_eq!(device.destruction_threads().len(), 1);
        assert_ne!(device.destruction_threads()[0], dropped_on);
    }
    #[test]
    fn exact_reader_capacity_precedes_native_creation_and_reuses_retired_slot() {
        let _serial = serial();
        let device = BlobLedgerDevice::new();
        let bank = ModeBlobBank::cold().unwrap();
        let mut readers: Vec<_> = (0..MODE_OWNER_SLOTS)
            .map(|_| ModeBlob::new_in(&bank, &device, &mode()).unwrap())
            .collect();
        assert!(matches!(
            ModeBlob::new_in(&bank, &device, &mode()),
            Err(Error::ModeBlobCapacity {
                capacity: MODE_OWNER_SLOTS
            })
        ));
        assert_eq!(device.created(), MODE_OWNER_SLOTS);
        let old = readers.pop().unwrap();
        let id = old.id();
        drop(old);
        assert_eq!(device.destroys(id), 1);
        let successor = ModeBlob::new_in(&bank, &device, &mode()).unwrap();
        assert_ne!(successor.id(), id);
        drop(readers);
        drop(successor);
        assert_eq!(device.alive(), 0);
    }
    #[test]
    fn unknown_native_destroy_quarantines_exact_owners_at_finite_bound() {
        let _serial = serial();
        let start = live_mode_blobs();
        let device = BlobLedgerDevice::new();
        let bank = ModeBlobBank::cold().unwrap();
        device.fail_destroy();
        for _ in 0..MODE_OWNER_SLOTS {
            let blob = ModeBlob::new_in(&bank, &device, &mode()).unwrap();
            drop(blob);
            super::super::blob_actor::drain();
        }
        assert!(matches!(
            ModeBlob::new_in(&bank, &device, &mode()),
            Err(Error::ModeBlobCapacity {
                capacity: MODE_OWNER_SLOTS
            })
        ));
        assert_eq!(device.created(), MODE_OWNER_SLOTS);
        assert_eq!(device.alive(), MODE_OWNER_SLOTS);
        assert_eq!(live_mode_blobs(), start + MODE_OWNER_SLOTS);
    }
    #[test]
    #[cfg(feature = "backend_vulkan")]
    fn warmed_reader_clone_and_final_drop_do_not_allocate_or_free() {
        let _serial = serial();
        let device = BlobLedgerDevice::new();
        let bank = ModeBlobBank::cold().unwrap();
        let blob = ModeBlob::new_in(&bank, &device, &mode()).unwrap();
        let id = blob.id();
        let (_, calls) = crate::backend::renderer::vulkan::storage_heap_probe::measure(|| {
            for _ in 0..4096 {
                drop(blob.clone());
            }
            drop(blob);
        });
        assert_eq!(calls, [0; 4]);
        assert_eq!(device.destroys(id), 1);
    }
    #[test]
    fn creating_file_inventory_preserves_quarantine_across_surface_generations() {
        let _serial = serial();
        let device = BlobLedgerDevice::new();
        device.fail_destroy();
        let crtc = drm::control::from_u32(12).unwrap();
        let file = super::ModeFileBanks::cold(&[crtc]).unwrap();
        {
            let first_surface = file.lease(crtc).unwrap();
            for _ in 0..MODE_OWNER_SLOTS {
                drop(ModeBlob::new_in(&first_surface, &device, &mode()).unwrap());
                super::super::blob_actor::drain();
            }
        }
        let recreated_surface = file.lease(crtc).unwrap();
        assert!(matches!(
            ModeBlob::new_in(&recreated_surface, &device, &mode()),
            Err(Error::ModeBlobCapacity {
                capacity: MODE_OWNER_SLOTS
            })
        ));
        assert_eq!(device.created(), MODE_OWNER_SLOTS);
        assert!(matches!(
            file.lease(drm::control::from_u32(13).unwrap()),
            Err(Error::UnknownCrtc(_))
        ));
    }
}
