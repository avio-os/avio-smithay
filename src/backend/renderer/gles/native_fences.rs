//! Strict completion policy. A poisoned epoch never resumes or becomes signaled.

use std::{
    cell::{Cell, RefCell},
    os::fd::{BorrowedFd, FromRawFd, IntoRawFd, OwnedFd},
    sync::{
        atomic::{AtomicI32, AtomicU64, Ordering},
        Arc,
    },
};

static NEXT_NATIVE_EPOCH: AtomicU64 = AtomicU64::new(1);

use super::{Capability, GlesError, GlesRenderer, GlesTexture};
use crate::backend::renderer::Renderer;
use crate::backend::{
    egl::{
        fence::{EGLFence, OwnedEGLFence},
        EGLContext,
    },
    renderer::sync::{Fence, Interrupted, SyncFileFence, SyncPoint},
};

#[derive(Debug)]
struct PublishedNativeFence {
    native: AtomicI32,
}
impl Default for PublishedNativeFence {
    fn default() -> Self {
        Self {
            native: AtomicI32::new(-1),
        }
    }
}
impl PublishedNativeFence {
    fn publish(&self, native: OwnedFd) {
        // The owner calls this only after every external Arc has returned and
        // the previous kernel fence signaled. No Weak handle is ever exposed.
        let old = self.native.swap(native.into_raw_fd(), Ordering::AcqRel);
        if old >= 0 {
            unsafe {
                drop(OwnedFd::from_raw_fd(old));
            }
        }
    }
    fn empty(&self) -> bool {
        self.native.load(Ordering::Acquire) < 0
    }
}
impl Drop for PublishedNativeFence {
    fn drop(&mut self) {
        let native = *self.native.get_mut();
        if native >= 0 {
            unsafe {
                drop(OwnedFd::from_raw_fd(native));
            }
        }
    }
}
impl Fence for PublishedNativeFence {
    fn is_signaled(&self) -> bool {
        let native = self.native.load(Ordering::Acquire);
        if native < 0 {
            return false;
        }
        // SAFETY: Every external call retains its Arc. Publication cannot
        // replace this descriptor until all such readers have returned.
        let fd = unsafe { BorrowedFd::borrow_raw(native) };
        let mut poll = [rustix::event::PollFd::new(&fd, rustix::event::PollFlags::IN)];
        rustix::event::poll(
            &mut poll,
            Some(&rustix::time::Timespec {
                tv_sec: 0,
                tv_nsec: 0,
            }),
        )
        .is_ok_and(|ready| ready > 0)
            && poll[0].revents() == rustix::event::PollFlags::IN
    }
    fn wait(&self) -> Result<(), Interrupted> {
        SyncFileFence::new(self.export().ok_or(Interrupted)?).wait()
    }
    fn is_exportable(&self) -> bool {
        !self.empty()
    }
    fn export(&self) -> Option<OwnedFd> {
        let native = self.native.load(Ordering::Acquire);
        if native < 0 {
            return None;
        }
        // SAFETY: The same exact Arc return condition pins this live fd.
        rustix::io::fcntl_dupfd_cloexec(unsafe { BorrowedFd::borrow_raw(native) }, 3).ok()
    }
}

#[derive(Debug, Default)]
pub(super) struct NativeFencePolicy {
    required: bool,
    epoch: u64,
    poisoned: Cell<bool>,
    sequence: Cell<u64>,
    latest: RefCell<Option<SyncPoint>>,
    completions: Vec<Arc<PublishedNativeFence>>,
}

impl NativeFencePolicy {
    pub(super) fn required() -> Self {
        Self {
            required: true,
            epoch: NEXT_NATIVE_EPOCH
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| value.checked_add(1))
                .expect("native renderer epoch identity exhausted"),
            poisoned: Cell::new(false),
            sequence: Cell::new(0),
            latest: RefCell::new(None),
            completions: (0..32)
                .map(|_| Arc::new(PublishedNativeFence::default()))
                .collect(),
        }
    }

    fn available_completion(&self) -> Option<&Arc<PublishedNativeFence>> {
        self.completions.iter().find(|slot| {
            let known_latest = self
                .latest
                .borrow()
                .as_ref()
                .and_then(|point| point.get::<PublishedNativeFence>())
                .is_some_and(|native| std::ptr::eq(native, &***slot));
            if Arc::strong_count(slot) != 1 + usize::from(known_latest) {
                return false;
            }
            slot.empty() || slot.is_signaled()
        })
    }

    pub(super) fn is_required(&self) -> bool {
        self.required
    }

    pub(super) fn check(&self) -> Result<(), GlesError> {
        if self.poisoned.get() {
            Err(GlesError::UnobservableCompletion)
        } else {
            Ok(())
        }
    }

    pub(super) fn poison(&self) {
        if self.required {
            self.poisoned.set(true);
        }
    }

    pub(super) fn observe<T>(&self, result: Result<T, GlesError>) -> Result<T, GlesError> {
        self.check()?;
        result.inspect_err(|_| self.poison())
    }
}

/// Position in one native context's ordered completion history.
#[derive(Debug, Clone, Copy)]
pub struct GlesSubmissionSnapshot {
    epoch: u64,
    sequence: u64,
}

impl GlesRenderer {
    /// Reserve exact native completion owner slots before the first submission.
    /// The strict constructor initially admits 32 slots; workers can declare
    /// their complete bounded plan here. Slot reuse requires both a signaled
    /// kernel fence and return of every external SyncPoint reader.
    pub fn reserve_native_completions(&mut self, capacity: usize) -> Result<(), GlesError> {
        if !self.requires_native_fences() || self.native_fences.sequence.get() != 0 || capacity == 0 {
            return Err(GlesError::Composition(
                "native completion reservation must precede submissions",
            ));
        }
        self.native_fences.completions = (0..capacity)
            .map(|_| Arc::new(PublishedNativeFence::default()))
            .collect();
        Ok(())
    }

    /// Number of preallocated native completion ownership slots.
    pub fn native_completion_capacity(&self) -> usize {
        self.native_fences.completions.len()
    }

    /// Snapshot the actual successful native submissions in this context.
    pub fn submission_snapshot(&self) -> GlesSubmissionSnapshot {
        GlesSubmissionSnapshot {
            epoch: self.native_fences.epoch,
            sequence: self.native_fences.sequence.get(),
        }
    }

    /// Latest real native completion covering work issued since the snapshot.
    /// Failed fence exports never advance this history or synthesize receipts.
    pub fn completion_since(&self, snapshot: GlesSubmissionSnapshot) -> Option<SyncPoint> {
        (self.native_fences.epoch == snapshot.epoch && self.native_fences.sequence.get() > snapshot.sequence)
            .then(|| self.native_fences.latest.borrow().clone())
            .flatten()
    }

    /// Create a GLES 3 renderer requiring native completion and GPU server waits.
    ///
    /// This constructor checks the required GL/EGL APIs before creating renderer
    /// resources. Actual allocation, import, sampling and presentation admission
    /// still belongs to the caller.
    ///
    /// Unlike [`Self::new`], frame completion, blits and dependency waits never
    /// substitute `glFinish`, a CPU wait, or a signaled completion on failure.
    /// A failed synchronization operation or dropped unfinished frame poisons
    /// this renderer epoch. [`Self::check_completion_health`] then reports an
    /// error and subsequent rendering is refused.
    ///
    /// The caller must retain/quarantine every externally owned allocation and
    /// sampled source whose completion is unobservable. Destroying this renderer
    /// is not proof that the GPU finished: do not reuse those allocations in a
    /// replacement epoch. Native object deletion follows GL/driver deferred
    /// destruction; dispose of the old allocation owners without returning them
    /// to a reusable pool. Context-affine teardown must run on its owner thread.
    ///
    /// # Safety
    ///
    /// The context must not be current on another thread, as for [`Self::new`].
    pub unsafe fn new_with_native_fences(context: EGLContext) -> Result<Self, GlesError> {
        let capabilities = Self::supported_capabilities(&context)?;
        if !capabilities.contains(&Capability::Fencing) {
            return Err(GlesError::GLVersionNotSupported(super::version::GLES_3_0));
        }
        if !capabilities.contains(&Capability::ExportFence) {
            return Err(GlesError::GLExtensionNotSupported(&["GL_OES_EGL_sync"]));
        }
        if !EGLFence::supports_importing(context.display()) {
            return Err(GlesError::EGLExtensionNotSupported(&[
                "EGL_ANDROID_native_fence_sync",
            ]));
        }
        let display = context.display();
        if display.get_egl_version() < (1, 5)
            && !display.extensions().iter().any(|ext| ext == "EGL_KHR_wait_sync")
        {
            return Err(GlesError::EGLExtensionNotSupported(&["EGL_KHR_wait_sync"]));
        }
        let mut renderer = unsafe { Self::with_capabilities(context, capabilities)? };
        renderer.native_fences = NativeFencePolicy::required();
        Ok(renderer)
    }

    /// Whether this renderer was constructed with mandatory native fences.
    pub fn requires_native_fences(&self) -> bool {
        self.native_fences.is_required()
    }

    /// Check whether all submitted work still has observable completion.
    ///
    /// An error is permanent for this renderer epoch. A successful return does
    /// not mean pending work is complete; the returned native fences remain the
    /// only completion evidence.
    pub fn check_completion_health(&self) -> Result<(), GlesError> {
        self.native_fences.check()?;
        if let Some(origin) = &self.origin {
            self.native_fences.observe(origin.check_health())?;
        }
        Ok(())
    }

    /// Publish all already-issued helper work as a real native completion.
    /// The caller retains its immutable resource packet through owner import.
    pub fn publish_shared_work(&mut self) -> Result<SyncPoint, GlesError> {
        self.check_completion_health()?;
        unsafe {
            self.egl.make_current()?;
        }
        self.finish_native()
    }

    /// Publish cold texture creation/import to shared contexts with actual
    /// GL ordering and an exportable native receipt. No CPU wait is used.
    pub fn publish_texture_for_sharing(&mut self, texture: &GlesTexture) -> Result<SyncPoint, GlesError> {
        self.check_completion_health()?;
        if texture.context_id() != self.context_id() {
            return Err(GlesError::Composition(
                "published texture belongs to another share group",
            ));
        }
        unsafe {
            self.egl.make_current()?;
        }
        self.native_fences
            .observe(texture.0.sync.write().unwrap().update_write(&self.gl))?;
        self.finish_native()
    }

    pub(super) fn finish_native(&self) -> Result<SyncPoint, GlesError> {
        self.native_fences.check()?;
        self.native_fences.observe((|| {
            let slot = self
                .native_fences
                .available_completion()
                .ok_or(GlesError::UnobservableCompletion)?;
            let fence = OwnedEGLFence::create_native(self.egl.display()).map_err(GlesError::NativeFence)?;
            // Native fence export requires flushing after creation. Export now
            // so a later caller cannot discover an unobservable completion.
            unsafe { self.gl.Flush() };
            let native = fence.export().map_err(GlesError::NativeFence)?;
            slot.publish(native);
            let completion = SyncPoint::from_shared_fence(slot.clone());
            self.native_fences.sequence.set(
                self.native_fences
                    .sequence
                    .get()
                    .checked_add(1)
                    .ok_or(GlesError::UnobservableCompletion)?,
            );
            *self.native_fences.latest.borrow_mut() = Some(completion.clone());
            Ok(completion)
        })())
    }

    pub(super) fn wait_native(&self, sync: &SyncPoint) -> Result<(), GlesError> {
        self.native_fences.check()?;
        self.native_fences.observe(sync.try_for_each_fence(|dependency| {
            let native = dependency.export().ok_or(GlesError::NativeFenceUnavailable)?;
            let fence =
                OwnedEGLFence::import_native(self.egl.display(), native).map_err(GlesError::NativeFence)?;
            fence.wait().map_err(GlesError::NativeFence)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_required_completion_permanently_quarantines_epoch() {
        let policy = NativeFencePolicy::required();
        assert!(policy.check().is_ok());
        assert!(policy
            .observe::<()>(Err(GlesError::NativeFenceUnavailable))
            .is_err());
        assert!(matches!(policy.check(), Err(GlesError::UnobservableCompletion)));
        assert!(policy.observe(Ok(())).is_err());
        assert!(matches!(policy.check(), Err(GlesError::UnobservableCompletion)));
    }

    #[test]
    fn native_completion_slot_requires_gpu_signal_and_all_external_returns() {
        let mut policy = NativeFencePolicy::required();
        policy.completions.truncate(1);
        let native = rustix::event::eventfd(
            0,
            rustix::event::EventfdFlags::CLOEXEC | rustix::event::EventfdFlags::NONBLOCK,
        )
        .unwrap();
        let signal = rustix::io::fcntl_dupfd_cloexec(&native, 3).unwrap();
        policy.completions[0].publish(native);
        let reader = SyncPoint::from_shared_fence(policy.completions[0].clone());
        *policy.latest.borrow_mut() = Some(reader.clone());
        assert!(policy.available_completion().is_none());
        rustix::io::write(&signal, &1u64.to_ne_bytes()).unwrap();
        assert!(
            policy.available_completion().is_none(),
            "GPU completion cannot erase a live native reader"
        );
        drop(reader);
        assert!(
            policy.available_completion().is_some(),
            "internal history owner is reusable only after all external readers return"
        );
        let next = NativeFencePolicy::required();
        assert_ne!(policy.epoch, next.epoch);
    }

    #[test]
    fn compatibility_policy_does_not_acquire_strict_epoch_state() {
        let policy = NativeFencePolicy::default();
        assert!(!policy.is_required());
        policy.poison();
        assert!(policy.check().is_ok());
    }
}
