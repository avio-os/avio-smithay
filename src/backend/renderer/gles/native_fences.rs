//! Strict completion policy. A poisoned epoch never resumes or becomes signaled.

use std::cell::Cell;

use super::{Capability, GlesError, GlesRenderer};
use crate::backend::{
    egl::{fence::EGLFence, EGLContext},
    renderer::sync::{SyncFileFence, SyncPoint},
};

#[derive(Debug, Default)]
pub(super) struct NativeFencePolicy {
    required: bool,
    poisoned: Cell<bool>,
}

impl NativeFencePolicy {
    pub(super) fn required() -> Self {
        Self {
            required: true,
            poisoned: Cell::new(false),
        }
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

impl GlesRenderer {
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
        self.native_fences.check()
    }

    pub(super) fn finish_native(&self) -> Result<SyncPoint, GlesError> {
        self.native_fences.check()?;
        self.native_fences.observe((|| {
            let fence = EGLFence::create(self.egl.display()).map_err(GlesError::NativeFence)?;
            // Native fence export requires flushing after creation. Export now
            // so a later caller cannot discover an unobservable completion.
            unsafe { self.gl.Flush() };
            let native = fence.export().map_err(GlesError::NativeFence)?;
            Ok(SyncPoint::from(SyncFileFence::new(native)))
        })())
    }

    pub(super) fn wait_native(&self, sync: &SyncPoint) -> Result<(), GlesError> {
        self.native_fences.check()?;
        self.native_fences.observe(sync.try_for_each_fence(|dependency| {
            let native = dependency.export().ok_or(GlesError::NativeFenceUnavailable)?;
            let fence = EGLFence::import(self.egl.display(), native).map_err(GlesError::NativeFence)?;
            fence.wait(self.egl.display()).map_err(GlesError::NativeFence)
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
    fn compatibility_policy_does_not_acquire_strict_epoch_state() {
        let policy = NativeFencePolicy::default();
        assert!(!policy.is_required());
        policy.poison();
        assert!(policy.check().is_ok());
    }
}
