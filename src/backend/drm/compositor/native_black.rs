//! Native security frames use the ordinary KMS pending/queued completion lane.

use super::*;
use std::collections::HashSet;

/// Capability and resource policy for one output's native black shield.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeBlackKind {
    /// TEST_ONLY accepted an active CRTC with every output-owned plane disabled.
    Planeless,
    /// The output retains its previously committed opaque-black target.
    Buffered,
}

/// Native black could not be prepared under its exact output configuration.
#[derive(Debug, thiserror::Error)]
pub enum NativeBlackError {
    /// No committed initial opaque-black target was retained for this output.
    #[error("native black has no committed opaque-black target")]
    MissingBlackTarget,
    /// A mode or connector change requires a fresh black target/capability.
    #[error("native black belongs to an earlier output configuration")]
    ConfigurationChanged,
    /// DRM plane enumeration failed; no plane may be silently left visible.
    #[error("could not enumerate output planes for native black: {0}")]
    PlaneEnumeration(#[source] std::io::Error),
    /// A reset-plane belongs to another output; the frame cannot omit it.
    #[error("could not claim every output plane for native black")]
    PlaneClaimDenied,
    /// The exact all-planes-off or buffered-black atomic request was rejected.
    #[error(transparent)]
    Drm(#[from] DrmError),
}

#[derive(Debug)]
pub(super) struct NativeBlackTarget<B: Buffer, F: Framebuffer> {
    config: Option<PlaneConfig<B, F>>,
    mode: drm::control::Mode,
    connectors: HashSet<connector::Handle>,
}

fn disabled_planes<B: Buffer, F: Framebuffer>(
    handles: impl IntoIterator<Item = plane::Handle>,
) -> FrameState<B, F> {
    let mut frame = FrameState {
        planes: SmallVec::new(),
        reset_plane_claims: Vec::new(),
    };
    for handle in handles {
        if frame.plane_state(handle).is_none() {
            frame.planes.push((
                handle,
                PlaneState {
                    skip: false,
                    ..Default::default()
                },
            ));
        }
    }
    frame
}

fn shield_configuration<C>(accepted: bool, candidate: Option<C>) -> Result<Option<C>, NativeBlackError> {
    if accepted {
        Ok(None)
    } else {
        candidate.map(Some).ok_or(NativeBlackError::MissingBlackTarget)
    }
}

impl<A, F, U, G> DrmCompositor<A, F, U, G>
where
    A: Allocator,
    A::Error: std::error::Error + Send + Sync,
    A::Buffer: AsDmabuf,
    <A::Buffer as AsDmabuf>::Error: std::error::Error + Send + Sync + std::fmt::Debug,
    F: ExportFramebuffer<A::Buffer>,
    F::Framebuffer: std::fmt::Debug + Send + Sync + 'static,
    F::Error: std::error::Error + Send + Sync,
    G: AsFd + Clone,
{
    /// Enable native-black target custody before the initial black modeset.
    /// This is an output-owner policy; ordinary compositors keep it disabled.
    pub fn enable_native_black(&mut self) {
        self.native_black_enabled = true;
    }

    // Snapshot only planes actually assigned to this CRTC or held by its frame
    // pipeline. Potential overlay candidates may belong to another output.
    fn native_black_plane_handles(&self) -> Result<Vec<plane::Handle>, NativeBlackError> {
        let fd = self.surface.device_fd();
        let mut handles = vec![self.surface.plane()];
        for handle in fd.plane_handles().map_err(NativeBlackError::PlaneEnumeration)? {
            let info = fd.get_plane(handle).map_err(NativeBlackError::PlaneEnumeration)?;
            if info.crtc() == Some(self.surface.crtc()) && !handles.contains(&handle) {
                handles.push(handle);
            }
        }
        for frame in [
            Some(&self.current_frame),
            self.pending_frame.as_ref().map(|pending| &pending.frame),
            self.queued_frame
                .as_ref()
                .map(|queued| &queued.prepared_frame.frame),
        ]
        .into_iter()
        .flatten()
        {
            for (handle, state) in &frame.planes {
                if state.config.is_some() && !handles.contains(handle) {
                    handles.push(*handle);
                }
            }
        }
        Ok(handles)
    }

    fn native_black_frame(&self) -> Result<FrameState<A::Buffer, F::Framebuffer>, NativeBlackError> {
        let mut frame = disabled_planes(self.native_black_plane_handles()?);
        // Claims cover the exact request until the matching physical flip.
        // They also prevent Full's ordinary claim filter silently omitting a
        // cursor/old scanout plane that the capability test disabled.
        for (handle, _) in &frame.planes {
            frame.reset_plane_claims.push(
                self.surface
                    .claim_plane(*handle)
                    .ok_or(NativeBlackError::PlaneClaimDenied)?,
            );
        }
        Ok(frame)
    }

    /// Probe native planeless black after committing the output's initial
    /// opaque-black modeset. TEST_ONLY establishes capability only; it does
    /// not establish physical presentation or security authority.
    ///
    /// A refused/legacy output reserves the exact committed black target,
    /// making future shields independent of allocation, import and rendering.
    pub fn initialize_native_black(&mut self) -> Result<NativeBlackKind, NativeBlackError> {
        let mut frame = self.native_black_frame()?;
        let accepted = !self.surface.is_legacy()
            && self
                .surface
                .test_state(
                    frame.build_planes(
                        &self.surface,
                        self.supports_fencing,
                        true,
                        PlaneSyncMode::TestOnly,
                    ),
                    true,
                )
                .is_ok();
        let config = shield_configuration(accepted, self.native_black_candidate.take())?;
        let kind = if accepted {
            NativeBlackKind::Planeless
        } else {
            NativeBlackKind::Buffered
        };
        tracing::info!(
            crtc = ?self.surface.crtc(),
            shield_planeless = u8::from(accepted),
            "DRM native black capability"
        );
        self.native_black = Some(NativeBlackTarget {
            config,
            mode: self.surface.pending_mode(),
            connectors: self.surface.pending_connectors().into_iter().collect(),
        });
        Ok(kind)
    }

    /// Prepare a full native-black frame for ordinary [`Self::queue_frame`].
    /// Every output-owned cursor/overlay is disabled in that exact request.
    /// A successful return is preparation, never a presentation receipt.
    /// Existing pending/queued resources retain their normal completion custody.
    pub fn prepare_native_black(&mut self) -> Result<NativeBlackKind, NativeBlackError> {
        let black = self
            .native_black
            .as_ref()
            .ok_or(NativeBlackError::MissingBlackTarget)?;
        if black.mode != self.surface.pending_mode()
            || black.connectors != self.surface.pending_connectors().into_iter().collect()
        {
            self.native_black = None;
            self.native_black_candidate = None;
            return Err(NativeBlackError::ConfigurationChanged);
        }
        let mut frame = self.native_black_frame()?;
        let kind = if let Some(config) = &black.config {
            let mut primary = PlaneState {
                skip: false,
                ..Default::default()
            };
            primary.config = Some(config.clone());
            frame.set_state(self.surface.plane(), primary);
            NativeBlackKind::Buffered
        } else {
            NativeBlackKind::Planeless
        };
        self.surface.test_state(
            frame.build_planes(
                &self.surface,
                self.supports_fencing,
                true,
                PlaneSyncMode::TestOnly,
            ),
            true,
        )?;
        self.next_frame = Some(PreparedFrame {
            kind: PreparedFrameKind::Full,
            frame,
        });
        Ok(kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::drm::device::PlaneClaimStorage;
    use std::num::NonZeroU32;

    #[derive(Debug)]
    struct TestBuffer;
    impl Buffer for TestBuffer {
        fn size(&self) -> Size<i32, BufferCoords> {
            (1, 1).into()
        }
        fn format(&self) -> DrmFormat {
            DrmFormat {
                code: DrmFourcc::Xrgb8888,
                modifier: DrmModifier::Linear,
            }
        }
    }
    #[derive(Debug)]
    struct TestFramebuffer(framebuffer::Handle);
    impl AsRef<framebuffer::Handle> for TestFramebuffer {
        fn as_ref(&self) -> &framebuffer::Handle {
            &self.0
        }
    }
    impl Framebuffer for TestFramebuffer {
        fn format(&self) -> DrmFormat {
            TestBuffer.format()
        }
    }
    fn handle<T: From<NonZeroU32>>(id: u32) -> T {
        NonZeroU32::new(id).unwrap().into()
    }

    #[test]
    fn native_frame_disables_primary_cursor_and_old_overlay_without_duplicates() {
        let primary = handle(10);
        let cursor = handle(11);
        let old_overlay = handle(12);
        let frame = disabled_planes::<TestBuffer, TestFramebuffer>([primary, cursor, old_overlay, cursor]);
        assert_eq!(frame.planes.len(), 3);
        for plane in [primary, cursor, old_overlay] {
            let state = frame.plane_state(plane).unwrap();
            assert!(!state.skip);
            assert!(state.config.is_none());
        }
    }

    #[test]
    fn reset_claims_remain_owned_until_the_frame_retires() {
        let claims = PlaneClaimStorage::default();
        let crtc = handle(2);
        let other_crtc = handle(3);
        let plane = handle(10);
        let mut frame = disabled_planes::<TestBuffer, TestFramebuffer>([plane]);
        frame.reset_plane_claims.push(claims.claim(plane, crtc).unwrap());
        assert!(claims.claim(plane, other_crtc).is_none());
        assert!(claims.claim(plane, crtc).is_some());
        drop(frame);
        assert!(claims.claim(plane, other_crtc).is_some());
    }

    #[test]
    fn capable_outputs_release_black_target_and_refused_outputs_keep_it() {
        let target = Arc::new(());
        let config = shield_configuration(true, Some(target.clone())).unwrap();
        assert!(config.is_none());
        assert_eq!(Arc::strong_count(&target), 1);
        let config = shield_configuration(false, Some(target.clone())).unwrap();
        assert!(config.is_some());
        assert_eq!(Arc::strong_count(&target), 2);
        assert!(matches!(
            shield_configuration::<()>(false, None),
            Err(NativeBlackError::MissingBlackTarget)
        ));
    }
}
