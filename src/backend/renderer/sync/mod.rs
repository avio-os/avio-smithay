//! Helper for synchronizing rendering operations
use std::{error::Error, fmt, os::unix::io::OwnedFd, sync::Arc};

use downcast_rs::{impl_downcast, Downcast};

#[cfg(feature = "backend_egl")]
mod egl;

/// A native Linux `sync_file` fence FD wrapped as a [`Fence`].
///
/// This is intended as a small glue type for embedders/compositors that already
/// operate on `sync_file` FDs (e.g. DMA-BUF acquire fences) and want to feed
/// them into Smithay's [`SyncPoint`] API without implementing a bespoke fence
/// wrapper in their compositor.
///
/// On Vulkan, renderers can import the exported FD into a GPU wait primitive
/// (e.g. `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`) to avoid host-side
/// polling/waiting in steady-state.
#[derive(Debug)]
pub struct SyncFileFence {
    sync_file: OwnedFd,
}

impl SyncFileFence {
    /// Wrap a `sync_file` FD.
    pub fn new(sync_file: OwnedFd) -> Self {
        Self { sync_file }
    }
}

/// Waiting for the fence was interrupted for an unknown reason.
///
/// This does not mean that the fence is signalled or not, neither that
/// any timeout was reached. Waiting should be attempted again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Interrupted;

impl fmt::Display for Interrupted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Wait for Fence was interrupted")
    }
}
impl Error for Interrupted {}

/// A fence that will be signaled in finite time
pub trait Fence: std::fmt::Debug + Send + Sync + Downcast {
    /// Queries the state of the fence
    fn is_signaled(&self) -> bool;

    /// Blocks the current thread until the fence is signaled
    fn wait(&self) -> Result<(), Interrupted>;

    /// Returns whether this fence can be exported
    /// as a native fence fd
    fn is_exportable(&self) -> bool;

    /// Export this fence as a native fence fd
    fn export(&self) -> Option<OwnedFd>;
}
impl_downcast!(Fence);

impl Fence for SyncFileFence {
    fn is_signaled(&self) -> bool {
        let requested = rustix::event::PollFlags::IN
            | rustix::event::PollFlags::ERR
            | rustix::event::PollFlags::HUP
            | rustix::event::PollFlags::NVAL;
        let mut poll_fd = [rustix::event::PollFd::new(&self.sync_file, requested)];
        let Ok(ready) = rustix::event::poll(
            &mut poll_fd,
            Some(&rustix::time::Timespec {
                tv_sec: 0,
                tv_nsec: 0,
            }),
        ) else {
            return false;
        };
        ready > 0
            && !poll_fd[0].revents().intersects(
                rustix::event::PollFlags::ERR
                    | rustix::event::PollFlags::HUP
                    | rustix::event::PollFlags::NVAL,
            )
            && poll_fd[0].revents().contains(rustix::event::PollFlags::IN)
    }

    fn wait(&self) -> Result<(), Interrupted> {
        loop {
            let requested = rustix::event::PollFlags::IN
                | rustix::event::PollFlags::ERR
                | rustix::event::PollFlags::HUP
                | rustix::event::PollFlags::NVAL;
            let mut poll_fd = [rustix::event::PollFd::new(&self.sync_file, requested)];
            match rustix::event::poll(&mut poll_fd, None) {
                Ok(ready)
                    if ready > 0
                        && poll_fd[0].revents().contains(rustix::event::PollFlags::IN)
                        && !poll_fd[0].revents().intersects(
                            rustix::event::PollFlags::ERR
                                | rustix::event::PollFlags::HUP
                                | rustix::event::PollFlags::NVAL,
                        ) =>
                {
                    return Ok(())
                }
                Ok(ready) if ready > 0 => return Err(Interrupted),
                Ok(_) => continue,
                Err(rustix::io::Errno::INTR) => continue,
                Err(_) => return Err(Interrupted),
            }
        }
    }

    fn is_exportable(&self) -> bool {
        true
    }

    fn export(&self) -> Option<OwnedFd> {
        rustix::io::fcntl_dupfd_cloexec(
            &self.sync_file,
            3, // Avoid stdio fds.
        )
        .ok()
    }
}

/// A sync point the will be signaled in finite time
#[derive(Debug, Clone)]
#[must_use = "this `SyncPoint` may contain a fence that should be awaited, failing to do so may result in unexpected rendering artifacts"]
pub struct SyncPoint {
    fence: Option<Arc<dyn Fence>>,
}

impl Default for SyncPoint {
    fn default() -> Self {
        Self::signaled()
    }
}

impl SyncPoint {
    /// Create an already signaled sync point
    pub fn signaled() -> Self {
        Self {
            fence: Default::default(),
        }
    }

    /// Returns `true` if `SyncPoint` contains a [`Fence`]
    pub fn contains_fence(&self) -> bool {
        self.fence.is_some()
    }

    /// Get a reference to the underlying [`Fence`] if any
    ///
    /// Returns `None` if the sync point does not contain a fence
    /// or contains a different type of fence
    pub fn get<F: Fence + 'static>(&self) -> Option<&F> {
        self.fence.as_ref().and_then(|f| f.downcast_ref())
    }

    /// Queries the state of the sync point
    ///
    /// Will always return `true` in case the sync point does not contain a fence
    pub fn is_reached(&self) -> bool {
        self.fence.as_ref().map(|f| f.is_signaled()).unwrap_or(true)
    }

    /// Blocks the current thread until the sync point is signaled
    ///
    /// If the sync point does not contain a fence this will never block.
    #[profiling::function]
    pub fn wait(&self) -> Result<(), Interrupted> {
        if let Some(fence) = self.fence.as_ref() {
            fence.wait()
        } else {
            Ok(())
        }
    }

    /// Returns whether this sync point can be exported as a native fence fd
    ///
    /// Will always return `false` in case the sync point does not contain a fence
    pub fn is_exportable(&self) -> bool {
        self.fence.as_ref().map(|f| f.is_exportable()).unwrap_or(false)
    }

    /// Export this [`SyncPoint`] as a native fence fd
    ///
    /// Will always return `None` in case the sync point does not contain a fence
    #[profiling::function]
    pub fn export(&self) -> Option<OwnedFd> {
        self.fence.as_ref().and_then(|f| f.export())
    }
}

impl<T: Fence + 'static> From<T> for SyncPoint {
    #[inline]
    fn from(value: T) -> Self {
        SyncPoint {
            fence: Some(Arc::new(value)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Fence, SyncFileFence};

    #[test]
    fn sync_file_fence_requires_the_requested_ready_event() {
        let sync_file = rustix::event::eventfd(
            0,
            rustix::event::EventfdFlags::CLOEXEC | rustix::event::EventfdFlags::NONBLOCK,
        )
        .unwrap();
        let signal_fd = rustix::io::fcntl_dupfd_cloexec(&sync_file, 3).unwrap();
        let fence = SyncFileFence::new(sync_file);

        assert!(!fence.is_signaled());
        rustix::io::write(&signal_fd, &1_u64.to_ne_bytes()).unwrap();
        assert!(fence.is_signaled());
        fence.wait().unwrap();
    }
}
