//! Release ownership for one applied Wayland buffer attachment.
//!
//! A release is an AND-join of two independent facts:
//!
//! 1. the attachment has been superseded, so no new compositor use can start;
//! 2. every use which already started has supplied completion evidence.
//!
//! `BufferReleaseMerger` is the single authority for that join. Render fences
//! are exported once and merged in the kernel as they arrive. The normal path
//! forwards that merged fence directly into implicit reservation state or the
//! explicit release timeline, just like KWin's `SyncReleasePoint` model. Rare
//! merge/import failures are held by one process-wide, fd-readiness-driven
//! executor. There is no periodic polling and elapsed time is never completion
//! evidence.

use std::{
    io,
    os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd},
    sync::{mpsc::TrySendError, Arc, Mutex, OnceLock},
    thread,
};

use calloop::{
    channel::{self, Channel, Event, SyncSender},
    generic::Generic,
    Interest, LoopHandle, Mode, PostAction,
};
use wayland_server::protocol::wl_buffer::WlBuffer;

use crate::{backend::allocator::dmabuf::Dmabuf, wayland::drm_syncobj::DrmSyncPoint};

/// Exceptional release work is bounded. Normal release never enters this
/// queue: it imports the merged fence and returns immediately. Filling this
/// queue therefore means synchronization forwarding is systematically broken;
/// additional work is failed closed by withholding that generation's release.
const RELEASE_WORK_QUEUE_CAPACITY: usize = 256;

#[derive(Debug)]
enum ReleaseTarget {
    Implicit {
        buffer: WlBuffer,
        dmabuf: Option<Dmabuf>,
    },
    Explicit(DrmSyncPoint),
    #[cfg(test)]
    Probe {
        attachment: u64,
        completed: Arc<Mutex<Vec<u64>>>,
    },
}

#[derive(Debug)]
struct ReleaseState {
    target: Option<ReleaseTarget>,
    merged_sync_file: Option<OwnedFd>,
    merge_disabled: bool,
    pending_waits: usize,
    retired: bool,
    finalize_queued: bool,
}

#[derive(Debug)]
struct BufferReleaseMergerInner {
    state: Mutex<ReleaseState>,
}

/// One attachment generation's release authority.
///
/// Registration is open while a `Buffer` for this generation exists. The
/// `InnerBuffer` owner closes registration exactly once through [`Self::retire`].
/// Completion callbacks can outlive that edge, but the release target cannot.
#[derive(Debug)]
pub(super) struct BufferReleaseMerger {
    inner: Arc<BufferReleaseMergerInner>,
}

impl BufferReleaseMerger {
    pub(super) fn implicit(buffer: WlBuffer) -> Self {
        let dmabuf = crate::wayland::dmabuf::get_dmabuf(&buffer).ok().cloned();
        Self::new(ReleaseTarget::Implicit { buffer, dmabuf })
    }

    pub(super) fn explicit(release_point: DrmSyncPoint) -> Self {
        Self::new(ReleaseTarget::Explicit(release_point))
    }

    #[cfg(test)]
    fn probe(attachment: u64, completed: Arc<Mutex<Vec<u64>>>) -> Self {
        Self::new(ReleaseTarget::Probe {
            attachment,
            completed,
        })
    }

    fn new(target: ReleaseTarget) -> Self {
        Self {
            inner: Arc::new(BufferReleaseMergerInner {
                state: Mutex::new(ReleaseState {
                    target: Some(target),
                    merged_sync_file: None,
                    merge_disabled: false,
                    pending_waits: 0,
                    retired: false,
                    finalize_queued: false,
                }),
            }),
        }
    }

    pub(super) fn is_explicit(&self) -> bool {
        let state = self
            .inner
            .state
            .lock()
            .expect("Wayland buffer release state poisoned");
        matches!(
            state
                .target
                .as_ref()
                .expect("live Wayland buffer lost its release target"),
            ReleaseTarget::Explicit(_)
        )
    }

    /// Register one renderer read of this attachment generation.
    ///
    /// The cached export must be shared by every consumer of the submission:
    /// exporting some renderer fences twice is destructive. If the fence
    /// cannot join the kernel accumulator, the executor watches that exact
    /// completion and holds one merger reference until it is reached.
    pub(super) fn add_render_completion(&self, exported_sync_file: Arc<OwnedFd>) {
        let mut unmerged_completion = None;
        {
            let mut state = self
                .inner
                .state
                .lock()
                .expect("Wayland buffer release state poisoned");
            // `add_render_completion` borrows a live `Buffer`, while retirement
            // happens only from the exclusively-owned `InnerBuffer::drop`.
            // Reaching this state would therefore be an ownership bug, not a
            // runtime condition that can be recovered by inventing more state.
            assert!(
                !state.retired,
                "Wayland buffer use registered after attachment retirement"
            );

            let exported = exported_sync_file.as_fd().try_clone_to_owned();
            match exported {
                Ok(new_sync_file) if state.merged_sync_file.is_none() => {
                    state.merged_sync_file = Some(new_sync_file);
                }
                Ok(new_sync_file) if !state.merge_disabled => {
                    let existing = state
                        .merged_sync_file
                        .as_ref()
                        .expect("merged completion disappeared while registering a render use");
                    match merge_sync_files(existing.as_fd(), new_sync_file.as_fd()) {
                        Ok(merged) => state.merged_sync_file = Some(merged),
                        Err(err) => {
                            tracing::warn!(
                                ?err,
                                "Failed to merge Wayland render completions; waiting on the exact fence event"
                            );
                            state.merge_disabled = true;
                            unmerged_completion = Some(Arc::new(new_sync_file));
                        }
                    }
                }
                Ok(new_sync_file) => {
                    unmerged_completion = Some(Arc::new(new_sync_file));
                }
                Err(err) => {
                    tracing::warn!(
                        ?err,
                        "Failed to duplicate Wayland render completion; waiting on the shared exact fence event"
                    );
                    state.merge_disabled = true;
                    unmerged_completion = Some(exported_sync_file);
                }
            }

            if unmerged_completion.is_some() {
                state.pending_waits = state
                    .pending_waits
                    .checked_add(1)
                    .expect("Wayland buffer completion count overflowed");
            }
        }

        if let Some(completion) = unmerged_completion {
            enqueue_release_work(ReleaseWork::Wait {
                merger: Arc::clone(&self.inner),
                completion,
            });
        }
    }

    /// Close completion registration for this attachment generation.
    ///
    /// This is called only by `InnerBuffer::drop`, which proves that the
    /// surface-current owner and every compositor read lease have retired.
    /// The normal path forwards the merged completion immediately. Only an
    /// exceptional fence-forwarding failure enters the readiness executor.
    pub(super) fn retire(&self) {
        let finalization = {
            let mut state = self
                .inner
                .state
                .lock()
                .expect("Wayland buffer release state poisoned");
            assert!(!state.retired, "Wayland buffer attachment retired more than once");
            state.retired = true;
            take_finalization_if_ready(&mut state)
        };
        if let Some(finalization) = finalization {
            start_finalization(finalization);
        }
    }
}

fn take_finalization_if_ready(state: &mut ReleaseState) -> Option<PendingFinalization> {
    if finalization_is_ready(state) {
        state.finalize_queued = true;
        Some(PendingFinalization {
            target: state
                .target
                .take()
                .expect("Wayland buffer release target finalized more than once"),
            completion: state.merged_sync_file.take().map(Arc::new),
            forwarded_completion: false,
            forward_error_reported: false,
            completion_error_reported: false,
            signal_error_reported: false,
        })
    } else {
        None
    }
}

fn finalization_is_ready(state: &ReleaseState) -> bool {
    state.retired && state.pending_waits == 0 && !state.finalize_queued
}

impl BufferReleaseMergerInner {
    fn complete_wait(self: &Arc<Self>) {
        let finalization = {
            let mut state = self.state.lock().expect("Wayland buffer release state poisoned");
            assert!(
                state.pending_waits > 0,
                "Wayland buffer release completion underflow"
            );
            state.pending_waits -= 1;
            take_finalization_if_ready(&mut state)
        };
        if let Some(finalization) = finalization {
            start_finalization(finalization);
        }
    }
}

enum ReleaseWork {
    Wait {
        merger: Arc<BufferReleaseMergerInner>,
        completion: Arc<OwnedFd>,
    },
    Finalize(PendingFinalization),
}

struct PendingWait {
    merger: Arc<BufferReleaseMergerInner>,
    completion: Arc<OwnedFd>,
}

struct PendingFinalization {
    target: ReleaseTarget,
    completion: Option<Arc<OwnedFd>>,
    forwarded_completion: bool,
    forward_error_reported: bool,
    completion_error_reported: bool,
    signal_error_reported: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FinalizationProgress {
    Complete,
    WaitForCompletion,
    TerminalFailure,
}

impl PendingFinalization {
    /// Advance without blocking. A pending result is armed on the completion
    /// fd by the executor; it is never revisited by a periodic timer.
    fn progress(&mut self) -> FinalizationProgress {
        match &self.target {
            ReleaseTarget::Implicit { buffer, dmabuf } => {
                if let Some(completion) = self.completion.as_ref() {
                    if !self.forwarded_completion {
                        if let Some(dmabuf) = dmabuf {
                            match dmabuf.import_sync_file_read(completion.as_fd()) {
                                Ok(()) => {
                                    buffer.release();
                                    return FinalizationProgress::Complete;
                                }
                                Err(err) if !self.forward_error_reported => {
                                    tracing::warn!(
                                        ?err,
                                        "Failed to publish compositor read completion to implicit-sync buffer; waiting asynchronously"
                                    );
                                    self.forward_error_reported = true;
                                }
                                Err(_) => {}
                            }
                        }
                        self.forwarded_completion = true;
                    }
                    match sync_file_state(completion.as_fd()) {
                        CompletionState::Ready => {}
                        CompletionState::Pending => {
                            return FinalizationProgress::WaitForCompletion;
                        }
                        CompletionState::Invalid => {
                            if !self.completion_error_reported {
                                tracing::error!(
                                    "Wayland implicit buffer completion fence became invalid; withholding release"
                                );
                                self.completion_error_reported = true;
                            }
                            return FinalizationProgress::TerminalFailure;
                        }
                    }
                }
                buffer.release();
                FinalizationProgress::Complete
            }
            ReleaseTarget::Explicit(release_point) => {
                if let Some(completion) = self.completion.as_ref() {
                    if !self.forwarded_completion {
                        match completion
                            .as_fd()
                            .try_clone_to_owned()
                            .and_then(|completion| release_point.signal_with_sync_file(completion))
                        {
                            Ok(()) => return FinalizationProgress::Complete,
                            Err(err) if !self.forward_error_reported => {
                                tracing::warn!(
                                    ?err,
                                    "Failed to forward compositor completion to explicit release point; waiting asynchronously"
                                );
                                self.forward_error_reported = true;
                            }
                            Err(_) => {}
                        }
                        self.forwarded_completion = true;
                    }
                    match sync_file_state(completion.as_fd()) {
                        CompletionState::Ready => {}
                        CompletionState::Pending => {
                            return FinalizationProgress::WaitForCompletion;
                        }
                        CompletionState::Invalid => {
                            if !self.completion_error_reported {
                                tracing::error!(
                                    "Wayland explicit buffer completion fence became invalid; withholding release"
                                );
                                self.completion_error_reported = true;
                            }
                            return FinalizationProgress::TerminalFailure;
                        }
                    }
                }

                match release_point.signal() {
                    Ok(()) => FinalizationProgress::Complete,
                    Err(err) => {
                        if !self.signal_error_reported {
                            tracing::error!(?err, "Failed to signal explicit Wayland buffer release point");
                            self.signal_error_reported = true;
                        }
                        FinalizationProgress::TerminalFailure
                    }
                }
            }
            #[cfg(test)]
            ReleaseTarget::Probe {
                attachment,
                completed,
            } => {
                completed
                    .lock()
                    .expect("test release probe poisoned")
                    .push(*attachment);
                FinalizationProgress::Complete
            }
        }
    }
}

fn start_finalization(mut finalization: PendingFinalization) {
    match finalization.progress() {
        FinalizationProgress::Complete | FinalizationProgress::TerminalFailure => {}
        FinalizationProgress::WaitForCompletion => {
            enqueue_release_work(ReleaseWork::Finalize(finalization));
        }
    }
}

fn release_sender() -> &'static SyncSender<ReleaseWork> {
    static RELEASE_SENDER: OnceLock<SyncSender<ReleaseWork>> = OnceLock::new();
    RELEASE_SENDER.get_or_init(|| {
        let (sender, receiver) = channel::sync_channel(RELEASE_WORK_QUEUE_CAPACITY);
        thread::Builder::new()
            .name("smithay-buffer-release".into())
            .spawn(move || run_release_executor(receiver))
            .expect("failed to start the Smithay Wayland buffer release executor");
        sender
    })
}

fn enqueue_release_work(work: ReleaseWork) {
    match release_sender().try_send(work) {
        Ok(()) => {}
        Err(TrySendError::Full(_work)) => {
            tracing::error!(
                capacity = RELEASE_WORK_QUEUE_CAPACITY,
                "Wayland buffer release exception queue is full; withholding this generation's release"
            );
        }
        Err(TrySendError::Disconnected(_work)) => {
            tracing::error!("Wayland buffer release executor stopped; withholding this generation's release");
        }
    }
}

fn run_release_executor(receiver: Channel<ReleaseWork>) {
    let mut event_loop = calloop::EventLoop::<()>::try_new()
        .expect("failed to create the Smithay Wayland buffer release event loop");
    let handle = event_loop.handle();
    let callback_handle = handle.clone();
    let signal = event_loop.get_signal();
    handle
        .insert_source(receiver, move |event, _, _| match event {
            Event::Msg(work) => arm_release_work(work, &callback_handle),
            Event::Closed => signal.stop(),
        })
        .expect("failed to register the Smithay Wayland buffer release channel");
    event_loop
        .run(None, &mut (), |_| {})
        .expect("Smithay Wayland buffer release event loop failed");
}

fn arm_release_work(work: ReleaseWork, handle: &LoopHandle<'_, ()>) {
    match work {
        ReleaseWork::Wait { merger, completion } => {
            arm_pending_wait(PendingWait { merger, completion }, handle);
        }
        ReleaseWork::Finalize(finalization) => {
            arm_pending_finalization(finalization, handle);
        }
    }
}

fn arm_pending_wait(wait: PendingWait, handle: &LoopHandle<'_, ()>) {
    match sync_file_state(wait.completion.as_fd()) {
        CompletionState::Ready => {
            wait.merger.complete_wait();
            return;
        }
        CompletionState::Pending => {}
        CompletionState::Invalid => {
            tracing::error!("Unmerged Wayland render completion was invalid; withholding release");
            return;
        }
    }

    let source_fd = Arc::clone(&wait.completion);
    let mut wait = Some(wait);
    if let Err(err) = handle.insert_source(
        Generic::new(source_fd, Interest::READ, Mode::Level),
        move |_, fd, _| match sync_file_state(fd.as_fd()) {
            CompletionState::Ready => {
                wait.take()
                    .expect("Wayland completion source fired after removal")
                    .merger
                    .complete_wait();
                Ok(PostAction::Remove)
            }
            CompletionState::Pending => Ok(PostAction::Continue),
            CompletionState::Invalid => {
                tracing::error!("Unmerged Wayland render completion became invalid; withholding release");
                let _ = wait.take();
                Ok(PostAction::Remove)
            }
        },
    ) {
        tracing::error!(
            ?err,
            "Failed to arm Wayland render-completion readiness; withholding release"
        );
    }
}

fn arm_pending_finalization(finalization: PendingFinalization, handle: &LoopHandle<'_, ()>) {
    let Some(source_fd) = finalization.completion.as_ref().map(Arc::clone) else {
        tracing::error!(
            "Wayland release requested asynchronous finalization without a completion fence; withholding release"
        );
        return;
    };
    let mut finalization = Some(finalization);
    if let Err(err) = handle.insert_source(
        Generic::new(source_fd, Interest::READ, Mode::Level),
        move |_, _, _| {
            let pending = finalization
                .as_mut()
                .expect("Wayland finalization source fired after removal");
            match pending.progress() {
                FinalizationProgress::Complete | FinalizationProgress::TerminalFailure => {
                    let _ = finalization.take();
                    Ok(PostAction::Remove)
                }
                FinalizationProgress::WaitForCompletion => Ok(PostAction::Continue),
            }
        },
    ) {
        tracing::error!(
            ?err,
            "Failed to arm Wayland release-fence readiness; withholding release"
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompletionState {
    Pending,
    Ready,
    Invalid,
}

fn sync_file_state(sync_file: BorrowedFd<'_>) -> CompletionState {
    let requested = rustix::event::PollFlags::IN
        | rustix::event::PollFlags::ERR
        | rustix::event::PollFlags::HUP
        | rustix::event::PollFlags::NVAL;
    let mut poll_fd = [rustix::event::PollFd::new(&sync_file, requested)];
    let Ok(ready) = rustix::event::poll(
        &mut poll_fd,
        Some(&rustix::time::Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        }),
    ) else {
        return CompletionState::Invalid;
    };
    if ready == 0 {
        return CompletionState::Pending;
    }
    let returned = poll_fd[0].revents();
    if returned.intersects(
        rustix::event::PollFlags::ERR | rustix::event::PollFlags::HUP | rustix::event::PollFlags::NVAL,
    ) {
        CompletionState::Invalid
    } else if returned.contains(rustix::event::PollFlags::IN) {
        CompletionState::Ready
    } else {
        CompletionState::Invalid
    }
}

fn merge_sync_files(first: BorrowedFd<'_>, second: BorrowedFd<'_>) -> io::Result<OwnedFd> {
    #[repr(C)]
    struct SyncMergeData {
        name: [u8; 32],
        fd2: i32,
        fence: i32,
        flags: u32,
        pad: u32,
    }

    const SYNC_IOC_MERGE: rustix::ioctl::Opcode = rustix::ioctl::opcode::read_write::<SyncMergeData>(b'>', 3);
    let mut data = SyncMergeData {
        name: [0; 32],
        fd2: second.as_raw_fd(),
        fence: -1,
        flags: 0,
        pad: 0,
    };
    // SAFETY: `first` and `second` are live sync_file descriptors and `data`
    // matches the kernel's `struct sync_merge_data` layout.
    unsafe {
        rustix::ioctl::ioctl(
            first,
            rustix::ioctl::Updater::<SYNC_IOC_MERGE, SyncMergeData>::new(&mut data),
        )
    }
    .map_err(io::Error::from)?;
    if data.fence < 0 {
        return Err(io::Error::other("SYNC_IOC_MERGE returned no fence"));
    }
    // SAFETY: a successful SYNC_IOC_MERGE returns a new owned descriptor.
    Ok(unsafe { OwnedFd::from_raw_fd(data.fence) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "use_system_lib"))]
    mod rust_server_egress {
        use std::{
            collections::HashSet,
            io::{self, Read},
            os::{fd::OwnedFd, unix::net::UnixStream},
            sync::{Arc, Barrier},
            thread,
        };

        use wayland_server::{
            backend::{protocol::Message, ClientId, Handle, ObjectData, ObjectId},
            protocol::{wl_buffer::WlBuffer, wl_callback::WlCallback},
            Display, Resource,
        };

        use super::super::BufferReleaseMerger;

        const EVENT_COUNT: usize = 64;

        #[derive(Debug)]
        struct InertObjectData;

        impl ObjectData<()> for InertObjectData {
            fn request(
                self: Arc<Self>,
                _handle: &Handle,
                _data: &mut (),
                _client_id: ClientId,
                _message: Message<ObjectId, OwnedFd>,
            ) -> Option<Arc<dyn ObjectData<()>>> {
                None
            }

            fn destroyed(
                self: Arc<Self>,
                _handle: &Handle,
                _data: &mut (),
                _client_id: ClientId,
                _object_id: ObjectId,
            ) {
            }
        }

        #[test]
        fn concurrent_callback_and_buffer_release_preserve_wire_framing() {
            let (server_stream, mut client_stream) = UnixStream::pair().unwrap();
            let mut display = Display::<()>::new().unwrap();
            let mut display_handle = display.handle();
            let client = display_handle.insert_client(server_stream, Arc::new(())).unwrap();

            let buffer: WlBuffer = client
                .create_resource_from_objdata(&display_handle, 1, Arc::new(InertObjectData))
                .unwrap();
            let callbacks: Vec<WlCallback> = (0..EVENT_COUNT)
                .map(|_| {
                    client
                        .create_resource_from_objdata(&display_handle, 1, Arc::new(InertObjectData))
                        .unwrap()
                })
                .collect();
            let buffer_id = buffer.id().protocol_id();
            let callback_ids: HashSet<u32> = callbacks
                .iter()
                .map(|callback| callback.id().protocol_id())
                .collect();

            let barrier = Barrier::new(2);
            thread::scope(|scope| {
                let worker_barrier = &barrier;
                let worker_buffer = buffer.clone();
                scope.spawn(move || {
                    worker_barrier.wait();
                    for _ in 0..EVENT_COUNT {
                        BufferReleaseMerger::implicit(worker_buffer.clone()).retire();
                        thread::yield_now();
                    }
                });

                barrier.wait();
                for (serial, callback) in callbacks.iter().enumerate() {
                    callback.done(serial as u32);
                    thread::yield_now();
                }
            });

            display.flush_clients().unwrap();
            client_stream.set_nonblocking(true).unwrap();
            let mut wire = Vec::new();
            let mut chunk = [0_u8; 4096];
            loop {
                match client_stream.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(read) => wire.extend_from_slice(&chunk[..read]),
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => panic!("failed to read test Wayland stream: {error}"),
                }
            }

            let mut offset = 0;
            let mut releases = 0;
            let mut callbacks_done = 0;
            while offset < wire.len() {
                assert!(wire.len() - offset >= 8, "truncated Wayland header at {offset}");
                let sender = u32::from_ne_bytes(wire[offset..offset + 4].try_into().unwrap());
                let header = u32::from_ne_bytes(wire[offset + 4..offset + 8].try_into().unwrap());
                let opcode = (header & 0xffff) as u16;
                let size = (header >> 16) as usize;
                assert!(
                    size >= 8 && size % 4 == 0,
                    "invalid Wayland size {size} at {offset}"
                );
                assert!(
                    size <= wire.len() - offset,
                    "Wayland message at {offset} overruns captured bytes"
                );

                if sender == buffer_id {
                    assert_eq!((opcode, size), (0, 8));
                    releases += 1;
                } else if callback_ids.contains(&sender) {
                    assert_eq!((opcode, size), (0, 12));
                    callbacks_done += 1;
                } else if sender == 1 {
                    assert_eq!((opcode, size), (1, 12));
                    let deleted = u32::from_ne_bytes(wire[offset + 8..offset + 12].try_into().unwrap());
                    assert!(callback_ids.contains(&deleted));
                } else {
                    panic!("unexpected Wayland sender {sender} at {offset}");
                }
                offset += size;
            }

            assert_eq!(releases, EVENT_COUNT);
            assert_eq!(callbacks_done, EVENT_COUNT);
        }
    }

    #[test]
    fn sync_file_readiness_distinguishes_pending_and_ready() {
        let sync_file = rustix::event::eventfd(
            0,
            rustix::event::EventfdFlags::CLOEXEC | rustix::event::EventfdFlags::NONBLOCK,
        )
        .unwrap();
        let signal_fd = sync_file.as_fd().try_clone_to_owned().unwrap();

        assert_eq!(sync_file_state(sync_file.as_fd()), CompletionState::Pending);
        rustix::io::write(&signal_fd, &1_u64.to_ne_bytes()).unwrap();
        assert_eq!(sync_file_state(sync_file.as_fd()), CompletionState::Ready);
    }

    #[test]
    fn exceptional_completion_wait_is_fd_driven() {
        let mut event_loop = calloop::EventLoop::<()>::try_new().unwrap();
        let completion = Arc::new(
            rustix::event::eventfd(
                0,
                rustix::event::EventfdFlags::CLOEXEC | rustix::event::EventfdFlags::NONBLOCK,
            )
            .unwrap(),
        );
        let signal_fd = completion.as_fd().try_clone_to_owned().unwrap();
        let merger = Arc::new(BufferReleaseMergerInner {
            state: Mutex::new(ReleaseState {
                target: None,
                merged_sync_file: None,
                merge_disabled: true,
                pending_waits: 1,
                retired: false,
                finalize_queued: false,
            }),
        });

        arm_pending_wait(
            PendingWait {
                merger: Arc::clone(&merger),
                completion,
            },
            &event_loop.handle(),
        );
        rustix::io::write(&signal_fd, &1_u64.to_ne_bytes()).unwrap();
        event_loop
            .dispatch(Some(std::time::Duration::from_secs(1)), &mut ())
            .unwrap();

        assert_eq!(merger.state.lock().unwrap().pending_waits, 0);
    }

    #[test]
    fn finalization_gate_requires_retirement_and_every_unmerged_completion() {
        let mut state = ReleaseState {
            target: None,
            merged_sync_file: None,
            merge_disabled: false,
            pending_waits: 1,
            retired: false,
            finalize_queued: false,
        };

        assert!(!finalization_is_ready(&state));
        state.retired = true;
        assert!(!finalization_is_ready(&state));
        state.pending_waits = 0;
        assert!(finalization_is_ready(&state));
        state.finalize_queued = true;
        assert!(!finalization_is_ready(&state));
    }

    #[test]
    fn separate_attachment_release_targets_complete_out_of_order() {
        let completed = Arc::new(Mutex::new(Vec::new()));
        let first = BufferReleaseMerger::probe(11, Arc::clone(&completed));
        let second = BufferReleaseMerger::probe(22, Arc::clone(&completed));
        first.inner.state.lock().unwrap().pending_waits = 1;
        second.inner.state.lock().unwrap().pending_waits = 1;

        first.retire();
        second.retire();
        assert!(completed.lock().unwrap().is_empty());

        second.inner.complete_wait();
        assert_eq!(*completed.lock().unwrap(), vec![22]);
        first.inner.complete_wait();
        assert_eq!(*completed.lock().unwrap(), vec![22, 11]);
    }

    #[test]
    fn attachment_release_waits_for_every_retained_reader() {
        let completed = Arc::new(Mutex::new(Vec::new()));
        let attachment = BufferReleaseMerger::probe(31, Arc::clone(&completed));
        attachment.inner.state.lock().unwrap().pending_waits = 2;

        attachment.retire();
        attachment.inner.complete_wait();
        assert!(completed.lock().unwrap().is_empty());

        attachment.inner.complete_wait();
        assert_eq!(*completed.lock().unwrap(), vec![31]);
    }
}
