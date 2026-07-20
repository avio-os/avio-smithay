use calloop::generic::Generic;
use calloop::{EventSource, Interest, Mode, Poll, PostAction, Readiness, Token, TokenFactory};
use drm::control::Device;
use std::sync::{Mutex, Weak};
use std::{
    io,
    os::unix::io::{AsFd, BorrowedFd, OwnedFd},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use crate::backend::drm::{DrmDeviceFd, WeakDrmDeviceFd};
use crate::backend::renderer::sync::{Fence, Interrupted};
use crate::wayland::compositor::{Blocker, BlockerState};

#[derive(Debug)]
pub(super) struct DrmTimelineInner {
    timeline_fd: OwnedFd,
    dev_ctx: Mutex<DrmTimelineDeviceSpecific>,
}

impl DrmTimelineInner {
    pub(super) fn update_device(&self, device: &DrmDeviceFd) -> io::Result<()> {
        let mut ctx = self.dev_ctx.lock().unwrap();
        let mut new = DrmTimelineDeviceSpecific::import(self.timeline_fd.as_fd(), device)?;
        for (point, eventfd) in ctx
            .event_fds
            .iter()
            .flat_map(|(p, fd)| fd.upgrade().map(|fd| (p, fd)))
        {
            device.syncobj_eventfd(new.syncobj, *point, eventfd.as_fd(), false)?;
            new.event_fds.push((*point, Arc::downgrade(&eventfd)));
        }
        *ctx = new;
        Ok(())
    }

    pub(super) fn invalidate(&self) {
        self.dev_ctx.lock().unwrap().invalidate()
    }
}

#[derive(Debug)]
struct DrmTimelineDeviceSpecific {
    device: WeakDrmDeviceFd,
    syncobj: drm::control::syncobj::Handle,
    event_fds: Vec<(u64, Weak<OwnedFd>)>,
}

impl DrmTimelineDeviceSpecific {
    fn import(fd: BorrowedFd<'_>, device: &DrmDeviceFd) -> io::Result<Self> {
        let syncobj = device.fd_to_syncobj(fd, false)?;
        Ok(DrmTimelineDeviceSpecific {
            device: device.downgrade(),
            syncobj,
            event_fds: Vec::new(),
        })
    }

    fn invalidate(&mut self) {
        if let Some(device) = self.device.upgrade() {
            let _ = device.destroy_syncobj(self.syncobj);
        }
        self.device = WeakDrmDeviceFd::new();
        // trigger event fds
        for eventfd in self.event_fds.drain(..).filter_map(|(_, x)| Weak::upgrade(&x)) {
            let _ = rustix::io::write(&eventfd, &[1]);
        }
    }
}

/// DRM timeline syncobj
#[derive(Clone, Debug)]
pub struct DrmTimeline(pub(super) Arc<DrmTimelineInner>);

impl PartialEq for DrmTimeline {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl DrmTimeline {
    /// Import DRM timeline from file descriptor
    pub fn new(device: &DrmDeviceFd, fd: OwnedFd) -> io::Result<Self> {
        let dev_ctx = Mutex::new(DrmTimelineDeviceSpecific::import(fd.as_fd(), device)?);
        Ok(Self(Arc::new(DrmTimelineInner {
            timeline_fd: fd,
            dev_ctx,
        })))
    }

    /// Query the last signalled timeline point
    pub fn query_signalled_point(&self) -> io::Result<u64> {
        let ctx = self.0.dev_ctx.lock().unwrap();
        let device = ctx
            .device
            .upgrade()
            .ok_or::<io::Error>(io::ErrorKind::InvalidInput.into())?;

        let mut points = [0];
        device.syncobj_timeline_query(&[ctx.syncobj], &mut points, false)?;
        Ok(points[0])
    }
}

/// Point on a DRM timeline syncobj
#[derive(Clone, Debug)]
pub struct DrmSyncPoint {
    pub(super) timeline: DrmTimeline,
    pub(super) point: u64,
}

impl DrmSyncPoint {
    /// Create an eventfd that will be signaled by the syncpoint
    pub fn eventfd(&self) -> io::Result<Arc<OwnedFd>> {
        let fd = rustix::event::eventfd(
            0,
            rustix::event::EventfdFlags::CLOEXEC | rustix::event::EventfdFlags::NONBLOCK,
        )?;
        let mut ctx = self.timeline.0.dev_ctx.lock().unwrap();
        ctx.device
            .upgrade()
            .ok_or::<io::Error>(io::ErrorKind::InvalidInput.into())?
            .syncobj_eventfd(ctx.syncobj, self.point, fd.as_fd(), false)?;

        let fd = Arc::new(fd);
        ctx.event_fds.retain(|(_, fd)| fd.upgrade().is_some());
        ctx.event_fds.push((self.point, Arc::downgrade(&fd)));
        Ok(fd)
    }

    /// Signal the sync point.
    pub fn signal(&self) -> io::Result<()> {
        let ctx = self.timeline.0.dev_ctx.lock().unwrap();
        ctx.device
            .upgrade()
            .ok_or::<io::Error>(io::ErrorKind::InvalidInput.into())?
            .syncobj_timeline_signal(&[ctx.syncobj], &[self.point])
    }

    /// Signal the sync point when the provided sync_file fence is reached.
    pub fn signal_with_sync_file(&self, sync_file: OwnedFd) -> io::Result<()> {
        let ctx = self.timeline.0.dev_ctx.lock().unwrap();
        let device = ctx
            .device
            .upgrade()
            .ok_or::<io::Error>(io::ErrorKind::InvalidInput.into())?;

        let imported_syncobj = import_sync_file_as_syncobj(&device, sync_file.as_fd())?;
        let transfer_result = device.syncobj_timeline_transfer(imported_syncobj, ctx.syncobj, 0, self.point);
        let destroy_result = device.destroy_syncobj(imported_syncobj);

        if let Err(err) = transfer_result {
            return Err(err);
        }
        // The transfer is the release-semantic operation. Failure to destroy
        // its temporary source handle must not make callers retry by manually
        // signaling a point whose completion dependency was already installed.
        if let Err(err) = destroy_result {
            tracing::warn!(
                ?err,
                "failed to destroy temporary syncobj after successful timeline transfer"
            );
        }
        Ok(())
    }

    /// Wait for sync point.
    pub fn wait(&self, timeout_nsec: i64) -> io::Result<()> {
        let ctx = self.timeline.0.dev_ctx.lock().unwrap();
        ctx.device
            .upgrade()
            .ok_or::<io::Error>(io::ErrorKind::InvalidInput.into())?
            .syncobj_timeline_wait(&[ctx.syncobj], &[self.point], timeout_nsec, false, false, false)?;
        Ok(())
    }

    /// Export DRM sync file for sync point.
    pub fn export_sync_file(&self) -> io::Result<OwnedFd> {
        let ctx = self.timeline.0.dev_ctx.lock().unwrap();
        let Some(device) = ctx.device.upgrade() else {
            return Err(io::ErrorKind::InvalidInput.into());
        };

        let syncobj = device.create_syncobj(false)?;
        if let Err(err) = device.syncobj_timeline_transfer(ctx.syncobj, syncobj, self.point, 0) {
            let _ = device.destroy_syncobj(syncobj);
            return Err(err);
        };

        let res = device.syncobj_to_fd(syncobj, true);
        let _ = device.destroy_syncobj(syncobj);
        res
    }

    /// Create an [`calloop::EventSource`] and [`Blocker`] for this sync point.
    ///
    /// This will fail if `drmSyncobjEventfd` isn't supported by the device. See
    /// [`supports_syncobj_eventfd`](super::supports_syncobj_eventfd).
    pub fn generate_blocker(&self) -> io::Result<(DrmSyncPointBlocker, DrmSyncPointSource)> {
        let fd = self.eventfd()?;
        let signal = Arc::new(AtomicBool::new(false));
        let blocker = DrmSyncPointBlocker {
            signal: signal.clone(),
            sync_point: self.clone(),
        };
        let source = DrmSyncPointSource {
            source: Generic::new(fd, Interest::READ, Mode::Level),
            signal,
        };
        Ok((blocker, source))
    }

    /// Create an authoritative blocker without an event-source fast path.
    ///
    /// [`DrmSyncPointBlocker::state`] queries the DRM timeline point itself,
    /// so readiness does not depend on successful eventfd creation or calloop
    /// registration. Compositors using this constructor must arrange a
    /// deterministic transaction-queue re-evaluation when the blocker is
    /// pending.
    pub fn blocker(&self) -> DrmSyncPointBlocker {
        DrmSyncPointBlocker {
            signal: Arc::new(AtomicBool::new(false)),
            sync_point: self.clone(),
        }
    }
}

/// Import a sync_file fence into a freshly created binary syncobj.
///
/// `DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE` with `IMPORT_SYNC_FILE` attaches the
/// sync_file's fence to an *existing* syncobj passed in `handle` — the kernel
/// does not create one, and handle 0 fails with `ENOENT`. drm-rs's
/// `fd_to_syncobj` hardcodes handle 0, so the ioctl is issued directly here
/// against a syncobj created first. The caller owns (and must destroy) the
/// returned handle.
fn import_sync_file_as_syncobj(
    device: &DrmDeviceFd,
    sync_file: BorrowedFd<'_>,
) -> io::Result<drm::control::syncobj::Handle> {
    use drm_ffi::drm_syncobj_handle;
    use std::os::unix::io::AsRawFd;

    const DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE: u32 = 1;
    // DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, per include/uapi/drm/drm.h (base 'd',
    // nr 0xC2, read-write on struct drm_syncobj_handle) — matching
    // drm-ffi's own binding of this ioctl.
    const DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE: rustix::ioctl::Opcode =
        rustix::ioctl::opcode::read_write::<drm_syncobj_handle>(b'd', 0xC2);

    let syncobj = device.create_syncobj(false)?;
    let mut args = drm_syncobj_handle {
        handle: syncobj.into(),
        flags: DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE,
        fd: sync_file.as_raw_fd(),
        pad: 0,
    };
    // SAFETY: `device` is a live DRM fd and `args` matches the kernel's
    // `struct drm_syncobj_handle` layout for this opcode.
    let import_result = unsafe {
        rustix::ioctl::ioctl(
            device,
            rustix::ioctl::Updater::<DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, drm_syncobj_handle>::new(&mut args),
        )
    };
    if let Err(err) = import_result {
        let _ = device.destroy_syncobj(syncobj);
        return Err(io::Error::from(err));
    }
    Ok(syncobj)
}

impl Fence for DrmSyncPoint {
    fn is_signaled(&self) -> bool {
        self.timeline
            .query_signalled_point()
            .ok()
            .is_some_and(|point| point >= self.point)
    }

    fn wait(&self) -> Result<(), Interrupted> {
        self.wait(i64::MAX).map_err(|_| Interrupted)
    }

    fn is_exportable(&self) -> bool {
        true
    }

    fn export(&self) -> Option<OwnedFd> {
        self.export_sync_file().ok()
    }
}

/// Event source generating an event when a [`DrmSyncPoint`] is signalled..
#[derive(Debug)]
pub struct DrmSyncPointSource {
    source: Generic<Arc<OwnedFd>>,
    signal: Arc<AtomicBool>,
}

impl EventSource for DrmSyncPointSource {
    type Event = ();
    type Metadata = ();
    type Ret = Result<(), std::io::Error>;
    type Error = io::Error;

    fn process_events<C>(
        &mut self,
        readiness: Readiness,
        token: Token,
        mut callback: C,
    ) -> Result<PostAction, Self::Error>
    where
        C: FnMut(Self::Event, &mut Self::Metadata) -> Self::Ret,
    {
        self.signal.store(true, Ordering::SeqCst);
        self.source
            .process_events(readiness, token, |_, _| Ok(PostAction::Remove))?;
        callback((), &mut ())?;
        Ok(PostAction::Remove)
    }

    fn register(&mut self, poll: &mut Poll, token_factory: &mut TokenFactory) -> calloop::Result<()> {
        self.source.register(poll, token_factory)?;
        Ok(())
    }

    fn reregister(&mut self, poll: &mut Poll, token_factory: &mut TokenFactory) -> calloop::Result<()> {
        self.source.reregister(poll, token_factory)?;
        Ok(())
    }

    fn unregister(&mut self, poll: &mut Poll) -> calloop::Result<()> {
        self.source.unregister(poll)?;
        Ok(())
    }
}

/// [`Blocker`] implementation for an accompaning [`DrmSyncPointSource`]
#[derive(Debug)]
pub struct DrmSyncPointBlocker {
    signal: Arc<AtomicBool>,
    sync_point: DrmSyncPoint,
}

impl Blocker for DrmSyncPointBlocker {
    fn state(&self) -> BlockerState {
        // The eventfd callback is the fast path. Querying the timeline point
        // makes blocker state authoritative even if that one-shot wakeup is
        // lost before the compositor re-enters its transaction queue.
        if self.signal.load(Ordering::SeqCst) || self.sync_point.is_signaled() {
            BlockerState::Released
        } else {
            BlockerState::Pending
        }
    }
}
