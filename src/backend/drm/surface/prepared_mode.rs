//! Cold validation, followed by exact owner-only pending-mode adoption.
use super::{atomic, DrmSurface, DrmSurfaceInternal, PlaneState};
use crate::backend::drm::error::Error;
use drm::control::{connector, Mode};
use std::sync::Arc;

#[derive(Debug)]
pub(super) enum ModeState {
    Atomic {
        current: atomic::State,
        expected: atomic::State,
        candidate: atomic::State,
    },
}

/// A validated mode of one exact surface. Preparing it does not mutate that
/// surface or submit KMS state. After adoption it owns the displaced pending
/// state/blob, so the caller returns the packet to its cold disposal owner.
#[derive(Debug)]
pub struct PreparedSurfaceMode {
    owner: Arc<DrmSurfaceInternal>,
    pub(super) state: ModeState,
    adopted: bool,
}
impl PreparedSurfaceMode {
    /// Desired mode before adoption; displaced mode after successful adoption.
    pub fn mode(&self) -> Mode {
        match &self.state {
            ModeState::Atomic { candidate, .. } => candidate.mode,
        }
    }
    pub(crate) fn connectors(&self) -> &std::collections::HashSet<connector::Handle> {
        match &self.state {
            ModeState::Atomic { candidate, .. } => &candidate.connectors,
        }
    }
    pub(crate) fn requires_modeset(&self) -> bool {
        match &self.state {
            ModeState::Atomic {
                current, candidate, ..
            } => current != candidate,
        }
    }
}
impl DrmSurface {
    /// Build the real mode blob/test request outside render/input threads.
    /// Only immutable snapshots cross native validation; no pending-state
    /// write lock or compositor/controller lock is held during its ioctl.
    pub fn prepare_mode(&self, mode: Mode) -> Result<PreparedSurfaceMode, Error> {
        let state = match &*self.internal {
            DrmSurfaceInternal::Atomic(surface) => surface.prepare_mode(mode)?,
            DrmSurfaceInternal::Legacy(_) => return Err(Error::PreparedModeUnsupported),
        };
        Ok(PreparedSurfaceMode {
            owner: self.internal.clone(),
            state,
            adopted: false,
        })
    }
    pub(crate) fn test_prepared_mode<'a>(
        &self,
        prepared: &PreparedSurfaceMode,
        planes: impl IntoIterator<Item = PlaneState<'a>>,
        allow_modeset: bool,
    ) -> Result<(), Error> {
        if prepared.adopted || !Arc::ptr_eq(&self.internal, &prepared.owner) {
            return Err(Error::TestFailed(self.crtc));
        }
        match (&*self.internal, &prepared.state) {
            (
                DrmSurfaceInternal::Atomic(surface),
                ModeState::Atomic {
                    current,
                    expected,
                    candidate,
                },
            ) => surface.test_prepared_mode(current, expected, candidate, planes, allow_modeset),
            _ => Err(Error::TestFailed(self.crtc)),
        }
    }
    /// Adopt once, only if the original surface still has the exact expected
    /// pending/current state. Busy or stale state leaves both owners unchanged.
    /// No allocation, native validation, blob destruction or GPU wait occurs.
    pub fn adopt_prepared_mode(&self, prepared: &mut PreparedSurfaceMode) -> bool {
        if prepared.adopted || !Arc::ptr_eq(&self.internal, &prepared.owner) {
            return false;
        }
        let adopted = match (&*self.internal, &mut prepared.state) {
            (
                DrmSurfaceInternal::Atomic(surface),
                ModeState::Atomic {
                    current,
                    expected,
                    candidate,
                },
            ) => surface.adopt_prepared_mode(current, expected, candidate),
            _ => false,
        };
        prepared.adopted = adopted;
        adopted
    }
}
