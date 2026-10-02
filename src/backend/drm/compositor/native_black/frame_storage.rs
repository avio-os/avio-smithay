//! Four native state receipts mirror current, pending, queued and candidate.
//! The bank stays with the compositor, so target replacement cannot orphan it.
use super::*;

#[derive(Debug)]
pub(in crate::backend::drm::compositor) struct NativeBlackFrameStorage<B: Buffer, F: Framebuffer> {
    planes: VecStorageBank<(plane::Handle, PlaneState<B, F>)>,
    resets: VecStorageBank<PlaneClaim>,
}
impl<B: Buffer, F: Framebuffer> NativeBlackFrameStorage<B, F> {
    pub(super) fn cold(planes: usize) -> Self {
        Self {
            planes: VecStorageBank::new_with_wakeup("native-black plane receipts", planes, 4, None),
            resets: VecStorageBank::new_with_wakeup("native-black reset receipts", planes, 4, None),
        }
    }
    pub(super) fn empty_frame(&self) -> Result<FrameState<B, F>, FrameWorkspaceError> {
        Ok(FrameState {
            planes: self.planes.acquire(0)?,
            reset_plane_claims: self.resets.acquire(0)?,
            opaque_black: false,
            native_black: true,
            damage_clip_error: None,
        })
    }
}

fn add_reset<B: Buffer, F: Framebuffer>(
    frame: &mut FrameState<B, F>,
    claim: &PlaneClaim,
) -> Result<(), FrameWorkspaceError> {
    if frame.plane_state(claim.plane()).is_some() {
        return Ok(());
    }
    frame.planes.push((
        claim.plane(),
        PlaneState {
            skip: false,
            ..Default::default()
        },
    ))?;
    frame.reset_plane_claims.push(claim.clone())
}

pub(super) fn assemble_native_frame<'a, B: Buffer + 'a, F: Framebuffer + 'a>(
    storage: &NativeBlackFrameStorage<B, F>,
    standing: &[PlaneClaim],
    previous: impl IntoIterator<Item = &'a FrameState<B, F>>,
) -> Result<FrameState<B, F>, FrameWorkspaceError> {
    let mut frame = storage.empty_frame()?;
    for claim in standing {
        add_reset(&mut frame, claim)?;
    }
    for prior in previous {
        for (_, state) in &prior.planes {
            if let Some(config) = &state.config {
                add_reset(&mut frame, &config.plane_claim)?;
            }
        }
    }
    Ok(frame)
}
