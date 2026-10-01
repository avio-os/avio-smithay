//! KMS mode blobs owned by the atomic surface state that names them.
//!
//! `MODE_ID` takes a property blob. The kernel keeps every blob a file
//! creates on that file's blob list until the file destroys it or closes, and
//! each `DESTROYPROPBLOB` walks the list linearly. A blob nothing destroys is
//! therefore not only leaked memory: it slows every later blob destroy on the
//! same file, including the per-frame `FB_DAMAGE_CLIPS` blob.
//!
//! A [`ModeBlob`] destroys its blob when the last state that names it is
//! dropped. The current and pending states of an atomic surface share one blob
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
/// An atomic surface owns at most two: the blob of its pending mode, and the
/// blob of its committed mode until the next commit or state reset releases it.
/// A count that grows with uptime is a leak.
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

#[derive(Debug)]
struct ModeBlobInner {
    owner: Box<dyn BlobOwner>,
    id: u64,
}

impl Drop for ModeBlobInner {
    fn drop(&mut self) {
        if let Err(err) = self.owner.destroy_blob(self.id) {
            warn!(blob = self.id, "Failed to destroy mode property blob: {}", err);
        }
        LIVE_MODE_BLOBS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// A KMS property blob holding one [`Mode`], destroyed with its last clone.
#[derive(Clone)]
pub(crate) struct ModeBlob(Arc<ModeBlobInner>);

impl fmt::Debug for ModeBlob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ModeBlob").field(&self.0.id).finish()
    }
}

impl ModeBlob {
    /// Create the blob for `mode` on `device`, which destroys it again.
    pub(crate) fn new<D>(device: &D, mode: &Mode) -> Result<Self, Error>
    where
        D: ControlDevice + DevPath + Clone + fmt::Debug + Send + Sync + 'static,
    {
        let value = device.create_property_blob(mode).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Failed to create Property Blob for mode",
                dev: device.dev_path(),
                source,
            })
        })?;
        LIVE_MODE_BLOBS.fetch_add(1, Ordering::Relaxed);
        Ok(ModeBlob(Arc::new(ModeBlobInner {
            owner: Box::new(device.clone()),
            id: value.into(),
        })))
    }

    /// The `MODE_ID` property value naming this blob.
    pub(crate) fn value(&self) -> property::Value<'static> {
        property::Value::Blob(self.0.id)
    }

    #[cfg(test)]
    pub(crate) fn id(&self) -> u64 {
        self.0.id
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
        SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[derive(Debug, Default)]
    struct Ledger {
        next_id: u64,
        destroys: HashMap<u64, usize>,
        created: usize,
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

        /// How many times `id` was destroyed.
        pub(crate) fn destroys(&self, id: u64) -> usize {
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
            *self.ledger.lock().unwrap().destroys.entry(blob).or_default() += 1;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_device::{serial, BlobLedgerDevice};
    use super::{live_mode_blobs, ModeBlob};
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
}
