//! Cold admitted DRM selections and independently owned plane snapshots.
use super::*;

type VisibleElement = (usize, Rectangle<i32, Physical>, usize, bool);
type OverlaySelection = (plane::Handle, usize, PlaneAssignmentInfo);

#[derive(Debug)]
pub(super) struct DrmFrameStorage<B: Buffer, F: Framebuffer> {
    pub output: VecStorageBank<VisibleElement>,
    pub primary: VecStorageBank<usize>,
    pub overlays: VecStorageBank<OverlaySelection>,
    pub removed: VecStorageBank<(usize, usize)>,
    pub fake: VecStorageBank<PlaneElementDescriptor>,
    pub returned_indices: VecStorageBank<usize>,
    pub returned_assignments: VecStorageBank<PlaneAssignmentInfo>,
    pub opaque: VecStorageBank<Rectangle<i32, Physical>>,
    pub opaque_work: VecStorageBank<Rectangle<i32, Physical>>,
    pub damage_clips: crate::backend::drm::surface::PlaneDamageClipBank,
    planes: VecStorageBank<(plane::Handle, PlaneState<B, F>)>,
    resets: VecStorageBank<PlaneClaim>,
}

impl<B: Buffer, F: Framebuffer> DrmFrameStorage<B, F> {
    #[cfg(test)]
    pub fn new(
        elements: usize,
        planes: usize,
        rectangles: usize,
        receipts: usize,
    ) -> Result<Self, FrameWorkspaceError> {
        Self::new_with_wakeup(
            elements,
            planes,
            rectangles,
            receipts,
            DamageStoragePolicy::Exact,
            None,
        )
    }

    pub fn new_with_wakeup(
        elements: usize,
        planes: usize,
        rectangles: usize,
        receipts: usize,
        policy: DamageStoragePolicy,
        returned: Option<StateReceiptReturnWakeup>,
    ) -> Result<Self, FrameWorkspaceError> {
        let selected_receipts = receipts.checked_mul(2).ok_or(FrameWorkspaceError {
            resource: "DRM selected receipts",
            required: usize::MAX,
            capacity: receipts,
        })?;
        Ok(Self {
            damage_clips: crate::backend::drm::surface::PlaneDamageClipBank::new(
                planes,
                rectangles,
                policy == DamageStoragePolicy::ConservativeFullOutput,
                returned.clone(),
            )?,
            output: VecStorageBank::new_with_wakeup("DRM visibility indices", elements, 1, returned.clone()),
            primary: VecStorageBank::new_with_wakeup("DRM primary indices", elements, 1, returned.clone()),
            overlays: VecStorageBank::new_with_wakeup("DRM overlay indices", planes, 1, returned.clone()),
            removed: VecStorageBank::new_with_wakeup("DRM failed plane indices", planes, 1, returned.clone()),
            fake: VecStorageBank::new_with_wakeup("DRM fake plane descriptors", planes, 1, returned.clone()),
            returned_indices: VecStorageBank::new_with_wakeup(
                "DRM selected result indices",
                elements,
                selected_receipts,
                returned.clone(),
            ),
            returned_assignments: VecStorageBank::new_with_wakeup(
                "DRM selected plane receipts",
                planes,
                receipts,
                returned.clone(),
            ),
            // Exact native state lane: current, pending, queued and candidate.
            // A newer candidate displaces next before taking its own slot.
            opaque: VecStorageBank::new_with_wakeup("DRM opaque rectangles", rectangles, 1, returned.clone()),
            opaque_work: VecStorageBank::new_with_wakeup(
                "DRM visibility rectangles",
                rectangles,
                1,
                returned.clone(),
            ),
            planes: VecStorageBank::new_with_wakeup("DRM plane state receipts", planes, 4, returned.clone()),
            resets: VecStorageBank::new_with_wakeup("DRM reset claim receipts", planes, 4, returned.clone()),
        })
    }

    pub fn plane_capacity(&self) -> usize {
        self.planes.entries()
    }
    pub fn is_reclaimable(&self) -> bool {
        self.output.is_reclaimable()
            && self.primary.is_reclaimable()
            && self.overlays.is_reclaimable()
            && self.removed.is_reclaimable()
            && self.fake.is_reclaimable()
            && self.returned_indices.is_reclaimable()
            && self.returned_assignments.is_reclaimable()
            && self.opaque.is_reclaimable()
            && self.opaque_work.is_reclaimable()
            && self.damage_clips.is_reclaimable()
            && self.planes.is_reclaimable()
            && self.resets.is_reclaimable()
    }

    pub fn empty_frame(&self, required: usize) -> Result<FrameState<B, F>, FrameWorkspaceError> {
        Ok(FrameState {
            planes: self.planes.acquire(required)?,
            reset_plane_claims: self.resets.acquire(required)?,
            opaque_black: false,
            native_black: false,
            damage_clip_error: None,
        })
    }
}

impl<B: Buffer, F: Framebuffer> FrameState<B, F> {
    pub(super) fn from_planes_reserved(
        primary: plane::Handle,
        planes: &Planes,
        storage: Option<&DrmFrameStorage<B, F>>,
    ) -> Result<Self, FrameWorkspaceError> {
        let Some(storage) = storage else {
            return Ok(Self::from_planes(primary, planes));
        };
        let mut frame = storage.empty_frame(1 + planes.cursor.len() + planes.overlay.len())?;
        frame.planes.push((primary, PlaneState::default()))?;
        frame.planes.extend(
            planes
                .cursor
                .iter()
                .chain(&planes.overlay)
                .map(|info| (info.handle, PlaneState::default())),
        )?;
        Ok(frame)
    }

    pub(super) fn copy_for_update(
        &self,
        storage: Option<&DrmFrameStorage<B, F>>,
    ) -> Result<Self, FrameWorkspaceError> {
        let mut frame = match storage {
            Some(storage) => storage.empty_frame(self.planes.len())?,
            None => FrameState {
                planes: WorkspaceVec::legacy(Vec::with_capacity(self.planes.len())),
                reset_plane_claims: WorkspaceVec::legacy(Vec::with_capacity(self.reset_plane_claims.len())),
                opaque_black: false,
                native_black: self.native_black,
                damage_clip_error: None,
            },
        };
        frame
            .planes
            .extend(self.planes.iter().map(|(handle, state)| (*handle, state.clone())))?;
        frame
            .reset_plane_claims
            .extend(self.reset_plane_claims.iter().cloned())?;
        frame.native_black = self.native_black;
        Ok(frame)
    }
}
