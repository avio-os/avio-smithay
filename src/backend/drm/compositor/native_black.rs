//! Native security frames use the ordinary KMS pending/queued completion lane.
//!
//! Configuration owners prepare capability and immutable rendering-complete
//! targets on cold turns. Mode/connector changes require that preparation
//! before readiness. No speculative frame pins a candidate, and no flip
//! callback probes or allocates one. Capability and rendering completion are
//! separate from the actual KMS presentation receipt.

use super::*;
use std::collections::HashSet;

/// Capability and resource policy for one output's native black shield.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeBlackKind {
    /// TEST_ONLY accepted an active CRTC with every output-owned plane disabled.
    Planeless,
    /// The output retains a rendering-complete immutable opaque-black target.
    Buffered,
}

/// Native black could not be prepared under its exact output configuration.
#[derive(Debug, thiserror::Error)]
pub enum NativeBlackError {
    /// No rendering-complete opaque-black target was retained for this output.
    #[error("native black has no completed opaque-black target")]
    MissingBlackTarget,
    /// A mode or connector change requires a fresh black target/capability.
    #[error("native black belongs to an earlier output configuration")]
    ConfigurationChanged,
    /// A previously accepted planeless request was refused. Keep the current
    /// front and ask the cold configuration owner for a completed replacement.
    #[error("planeless black was refused; a cold opaque-black fallback is required")]
    ColdFallbackRequired,
    /// A configuration target's rendering could not be observed complete.
    #[error("native-black rendering completion was interrupted")]
    RenderCompletionInterrupted,
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
    planeless: bool,
    modeset_fallback: bool,
    mode: drm::control::Mode,
    connectors: HashSet<connector::Handle>,
}

impl<B: Buffer, F: Framebuffer> NativeBlackTarget<B, F> {
    fn matches(&self, mode: drm::control::Mode, connectors: &HashSet<connector::Handle>) -> bool {
        self.mode == mode && self.connectors == *connectors
    }

    fn retained_configuration(
        accepted: bool,
        commit_pending: bool,
        candidate: Option<PlaneConfig<B, F>>,
    ) -> Result<Option<PlaneConfig<B, F>>, NativeBlackError> {
        shield_configuration(!needs_buffered_fallback(accepted, commit_pending), candidate)
    }
}

fn disabled_planes<B: Buffer, F: Framebuffer>(
    handles: impl IntoIterator<Item = plane::Handle>,
) -> FrameState<B, F> {
    let mut frame = FrameState {
        planes: SmallVec::new(),
        reset_plane_claims: Vec::new(),
        opaque_black: false,
        native_black: true,
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

// The same modeset allowance selects the real commit/page-flip path. A
// driver may pass atomic_check but reject modeset-only state afterwards.
fn probe_planeless<E>(legacy: bool, commit_pending: bool, test: impl FnOnce(bool) -> Result<(), E>) -> bool {
    !legacy && test(commit_pending).is_ok()
}

// Configuration owners wait before dropping render fences. Keeping a fence or
// damage blob on an immutable shield would replay a previous frame's metadata.
fn completed_black_config<B: Buffer, F: Framebuffer>(
    mut config: PlaneConfig<B, F>,
) -> Result<PlaneConfig<B, F>, NativeBlackError> {
    if let Some((sync, _)) = &config.sync {
        sync.wait()
            .map_err(|_| NativeBlackError::RenderCompletionInterrupted)?;
    }
    config.sync = None;
    config.damage_clips = None;
    Ok(config)
}

fn needs_buffered_fallback(accepted: bool, commit_pending: bool) -> bool {
    // ALLOW_MODESET capability does not prove a subsequent ordinary page flip.
    // Keep a fallback until a cold owner probes the committed configuration.
    !accepted || commit_pending
}

#[derive(Debug, Default)]
pub(super) struct NativeBlackRepaint {
    pending: bool,
}

impl NativeBlackRepaint {
    pub(super) fn request(&mut self) {
        self.pending = true;
    }

    pub(super) fn accepted(&mut self, full: bool, native_black: bool) {
        if full {
            self.pending = native_black;
        }
    }

    pub(super) fn pending(&self) -> bool {
        self.pending
    }

    pub(super) fn age(&self, age: usize) -> usize {
        if self.pending {
            0
        } else {
            age
        }
    }
}

pub(super) fn opaque_black_rendered(rendered: bool, empty: bool, color: Color32F, diagnostic: bool) -> bool {
    rendered && empty && color == Color32F::BLACK && !diagnostic
}

fn configuration_candidate<B: Buffer, F: Framebuffer>(
    current: &FrameState<B, F>,
    retained: Option<&PlaneConfig<B, F>>,
    primary: plane::Handle,
    size: Size<i32, Physical>,
) -> Option<PlaneConfig<B, F>> {
    current
        .opaque_black
        .then(|| current.plane_state(primary)?.config.clone())
        .flatten()
        .filter(|config| config.properties.dst.size == size)
        .or_else(|| {
            retained
                .filter(|config| config.properties.dst.size == size)
                .cloned()
        })
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

    /// Whether this output owner requested native-black configuration custody.
    pub fn native_black_enabled(&self) -> bool {
        self.native_black_enabled
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
        if let Some(config) = self.native_black_uncompleted.as_ref() {
            if let Some((sync, _)) = &config.sync {
                sync.wait()
                    .map_err(|_| NativeBlackError::RenderCompletionInterrupted)?;
            }
        }
        let (w, h) = self.surface.pending_mode().size();
        let size = Size::from((i32::from(w), i32::from(h)));
        let candidate = self.completed_black_candidate().or_else(|| {
            self.native_black_uncompleted
                .take()
                .filter(|config| config.properties.dst.size == size)
        });
        self.native_black_uncompleted = None;
        self.install_native_black(candidate, true)
    }

    fn completed_black_candidate(&self) -> Option<PlaneConfig<A::Buffer, F::Framebuffer>> {
        let (w, h) = self.surface.pending_mode().size();
        let size = Size::from((i32::from(w), i32::from(h)));
        configuration_candidate(
            &self.current_frame,
            self.native_black
                .as_ref()
                .and_then(|target| target.config.as_ref()),
            self.surface.plane(),
            size,
        )
    }

    fn complete_black_candidate(
        &mut self,
        config: PlaneConfig<A::Buffer, F::Framebuffer>,
    ) -> Result<PlaneConfig<A::Buffer, F::Framebuffer>, NativeBlackError> {
        if let ScanoutBuffer::Swapchain(slot) = &config.buffer.buffer {
            self.swapchain.detach(slot);
        }
        match completed_black_config(config.clone()) {
            Ok(config) => Ok(config),
            Err(error) => {
                self.native_black_uncompleted = Some(config);
                Err(error)
            }
        }
    }

    fn install_native_black(
        &mut self,
        candidate: Option<PlaneConfig<A::Buffer, F::Framebuffer>>,
        allow_planeless: bool,
    ) -> Result<NativeBlackKind, NativeBlackError> {
        // Legacy outputs need a buffered target and must not attempt a
        // planeless atomic request during bring-up.
        let accepted = allow_planeless
            && probe_planeless(
                self.surface.is_legacy(),
                self.surface.commit_pending(),
                |allow_modeset| {
                    let mut frame = self.native_black_frame()?;
                    self.surface
                        .test_state(
                            frame.build_planes(
                                &self.surface,
                                self.supports_fencing,
                                true,
                                PlaneSyncMode::TestOnly,
                            ),
                            allow_modeset,
                        )
                        .map_err(NativeBlackError::Drm)
                },
            );
        let config =
            NativeBlackTarget::retained_configuration(accepted, self.surface.commit_pending(), candidate)?
                .map(|config| self.complete_black_candidate(config))
                .transpose()?;
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
            planeless: accepted,
            modeset_fallback: self.surface.commit_pending(),
            mode: self.surface.pending_mode(),
            connectors: self.surface.pending_connectors().into_iter().collect(),
        });
        Ok(kind)
    }

    /// A modeset-only probe needs one committed-state cold refresh. The owner
    /// invokes this only after a real full flip completion, never from elapsed
    /// time or a speculative capability test.
    pub fn native_black_needs_committed_refresh(&self) -> bool {
        self.native_black
            .as_ref()
            .is_some_and(|black| black.modeset_fallback)
            && !self.surface.commit_pending()
    }

    /// Probe the completed configuration without ALLOW_MODESET and release its
    /// modeset fallback when this exact steady-state request is accepted.
    /// This cold-only API neither allocates nor submits a physical frame.
    pub fn refresh_committed_native_black(&mut self) -> Result<(), NativeBlackError> {
        if self.native_black_needs_committed_refresh() {
            self.initialize_native_black()?;
        }
        Ok(())
    }

    /// Prepare capability and, when necessary, a completed immutable shield
    /// for the pending mode/connectors using this output's existing renderer.
    ///
    /// Call on a cold configuration turn after [`Self::use_mode`] or connector
    /// changes, before publishing configuration readiness. This can allocate,
    /// render and wait; it must never run in a realtime render/flip callback.
    /// It does not submit KMS state or create a presentation receipt. The next
    /// [`Self::prepare_native_black`] still uses ordinary commit custody.
    pub fn configure_native_black<R>(
        &mut self,
        renderer: &mut R,
    ) -> Result<NativeBlackKind, RenderFrameErrorType<A, F, R>>
    where
        R: Renderer + Bind<Dmabuf>,
        R::TextureId: Texture + 'static,
    {
        self.configure_native_black_impl(renderer, false)
    }

    /// Cold replacement after a real planeless preparation was rejected.
    /// Retains a completed buffered shield even if a later speculative probe
    /// would accept planeless again; no speculative test retires this fallback.
    pub fn configure_native_black_fallback<R>(
        &mut self,
        renderer: &mut R,
    ) -> Result<NativeBlackKind, RenderFrameErrorType<A, F, R>>
    where
        R: Renderer + Bind<Dmabuf>,
        R::TextureId: Texture + 'static,
    {
        self.configure_native_black_impl(renderer, true)
    }

    fn configure_native_black_impl<R>(
        &mut self,
        renderer: &mut R,
        force_buffered: bool,
    ) -> Result<NativeBlackKind, RenderFrameErrorType<A, F, R>>
    where
        R: Renderer + Bind<Dmabuf>,
        R::TextureId: Texture + 'static,
    {
        self.native_black_enabled = true;
        if force_buffered {
            if let Some(candidate) = self.completed_black_candidate() {
                return self
                    .install_native_black(Some(candidate), false)
                    .map_err(|error| FrameError::NativeBlack(error).into());
            }
        } else {
            match self.initialize_native_black() {
                Ok(kind) => return Ok(kind),
                Err(NativeBlackError::MissingBlackTarget) => {}
                Err(error) => return Err(FrameError::NativeBlack(error).into()),
            }
        }
        // A configuration transaction must not replace work accepted by KMS.
        if !self.is_frame_pipeline_idle() {
            return Err(FrameError::NoFreeSlotsError.into());
        }
        if primary_clear_red_diag() {
            // A diagnostic render is never an opaque-black configuration
            // target. Reject before submitting GPU work that cannot qualify.
            return Err(FrameError::NativeBlack(NativeBlackError::MissingBlackTarget).into());
        }
        let (w, h) = self.surface.pending_mode().size();
        let source = self.output_mode_source.clone();
        self.set_output_mode_source(OutputModeSource::Static {
            size: (i32::from(w), i32::from(h)).into(),
            scale: 1.0.into(),
            transform: Transform::Normal,
        });
        self.native_black_repaint.request();
        // This API is cold-only: exact-mode shield provisioning is independent
        // of the ordinary realtime prepared-target admission policy.
        let prepared_only = std::mem::replace(&mut self.composition_prepared_only, false);
        let result = self
            .render_frame(
                renderer,
                &[] as &[crate::backend::renderer::element::solid::SolidColorRenderElement],
                Color32F::BLACK,
                FrameFlags::empty(),
            )
            .map(|_| ());
        self.composition_prepared_only = prepared_only;
        self.set_output_mode_source(source);
        result?;
        let prepared = self.next_frame.take().ok_or(FrameError::EmptyFrame)?;
        let candidate = prepared
            .frame
            .opaque_black
            .then(|| prepared.frame.plane_state(self.surface.plane())?.config.clone())
            .flatten();
        // Even if capability changes between the two tests, the freshly
        // rendered target may not be freed until its real GPU work completes.
        let candidate = candidate
            .map(|config| self.complete_black_candidate(config))
            .transpose()
            .map_err(FrameError::NativeBlack)?;
        self.install_native_black(candidate, !force_buffered)
            .map_err(|error| FrameError::NativeBlack(error).into())
    }

    /// Already-exported immutable shield allocation, if this exact configuration
    /// needs one. Census registration must occur on a cold owner turn; no buffer
    /// is allocated or exported by this read-only accessor.
    pub fn native_black_dmabuf(&self) -> Option<Dmabuf> {
        let config = self.native_black.as_ref()?.config.as_ref()?;
        let ScanoutBuffer::Swapchain(slot) = &config.buffer.buffer else {
            return None;
        };
        slot.userdata().get::<Dmabuf>().cloned()
    }

    /// Read the accepted cold shield capability without a TEST_ONLY operation,
    /// allocating connectors, or creating/exporting a shield. Missing cold
    /// configuration remains unavailable. A pending reconfiguration owner must
    /// invalidate its own prior census until that transaction completes.
    pub fn configured_native_black_kind(&self) -> Option<NativeBlackKind> {
        self.native_black.as_ref().map(|black| {
            if black.planeless {
                NativeBlackKind::Planeless
            } else {
                NativeBlackKind::Buffered
            }
        })
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
        if !black.matches(
            self.surface.pending_mode(),
            &self.surface.pending_connectors().into_iter().collect(),
        ) {
            return Err(NativeBlackError::ConfigurationChanged);
        }
        let mut frame = self.native_black_frame()?;
        let planeless = black.planeless
            && self
                .surface
                .test_state(
                    frame.build_planes(
                        &self.surface,
                        self.supports_fencing,
                        true,
                        PlaneSyncMode::TestOnly,
                    ),
                    self.surface.commit_pending(),
                )
                .is_ok();
        let kind = if !planeless {
            let config = black
                .config
                .as_ref()
                .ok_or(NativeBlackError::ColdFallbackRequired)?;
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
        if !planeless {
            self.surface.test_state(
                frame.build_planes(
                    &self.surface,
                    self.supports_fencing,
                    true,
                    PlaneSyncMode::TestOnly,
                ),
                self.surface.commit_pending(),
            )?;
        }
        self.next_frame = Some(PreparedFrame {
            kind: PreparedFrameKind::Full,
            frame,
        });
        self.native_black_repaint.request();
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

    fn config() -> PlaneConfig<TestBuffer, TestFramebuffer> {
        let claims = PlaneClaimStorage::default();
        PlaneConfig {
            properties: PlaneProperties {
                src: Rectangle::from_size((1.0, 1.0).into()),
                dst: Rectangle::from_size((1, 1).into()),
                transform: Transform::Normal,
                alpha: 1.0,
                format: TestBuffer.format(),
            },
            buffer: DrmScanoutBuffer {
                buffer: ScanoutBuffer::Swapchain(Arc::new(Slot::new(TestBuffer))),
                fb: CachedDrmFramebuffer::new(DrmFramebuffer::Exporter(TestFramebuffer(handle(20)))),
            },
            damage_clips: None,
            plane_claim: claims.claim(handle(10), handle(2)).unwrap(),
            sync: Some((SyncPoint::signaled(), None)),
        }
    }

    #[derive(Debug)]
    struct ConfigurationFence {
        waited: Arc<std::sync::atomic::AtomicUsize>,
        interrupted: bool,
    }
    impl crate::backend::renderer::sync::Fence for ConfigurationFence {
        fn is_signaled(&self) -> bool {
            false
        }
        fn wait(&self) -> Result<(), crate::backend::renderer::sync::Interrupted> {
            self.waited.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.interrupted {
                Err(crate::backend::renderer::sync::Interrupted)
            } else {
                Ok(())
            }
        }
        fn is_exportable(&self) -> bool {
            false
        }
        fn export(&self) -> Option<OwnedFd> {
            None
        }
    }

    #[test]
    fn configuration_observes_gpu_completion_before_discarding_sync_metadata() {
        let waited = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut candidate = config();
        candidate.sync = Some((
            SyncPoint::from(ConfigurationFence {
                waited: waited.clone(),
                interrupted: false,
            }),
            None,
        ));
        let target = completed_black_config(candidate).unwrap();
        assert_eq!(waited.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(target.sync.is_none());
        assert!(target.damage_clips.is_none());

        let mut interrupted = config();
        interrupted.sync = Some((
            SyncPoint::from(ConfigurationFence {
                waited: waited.clone(),
                interrupted: true,
            }),
            None,
        ));
        assert!(matches!(
            completed_black_config(interrupted.clone()),
            Err(NativeBlackError::RenderCompletionInterrupted)
        ));
        assert!(
            interrupted.sync.is_some(),
            "the cold owner retains an unobservable candidate for retry"
        );
        assert_eq!(waited.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn legacy_and_unfenced_black_render_can_supply_the_configuration_target() {
        let accepted = probe_planeless(true, false, |_| {
            panic!("legacy configuration must not submit a planeless atomic probe");
            #[allow(unreachable_code)]
            Ok::<_, ()>(())
        });
        let mut current = disabled_planes([handle(10)]);
        // Real rendering sets provenance from damage, even without a fence FD.
        current.opaque_black = opaque_black_rendered(true, true, Color32F::BLACK, false);
        current.set_state(
            handle(10),
            PlaneState {
                config: Some(config()),
                ..Default::default()
            },
        );
        let candidate = configuration_candidate(&current, None, handle(10), (1, 1).into());
        let target = shield_configuration(!needs_buffered_fallback(accepted, false), candidate)
            .unwrap()
            .unwrap();
        let target = completed_black_config(target).unwrap();
        assert!(target.sync.is_none());
        assert!(target.damage_clips.is_none());
        assert!(!opaque_black_rendered(false, true, Color32F::BLACK, false));
    }

    #[test]
    fn same_size_configuration_reuses_black_but_a_new_extent_requires_cold_rendering() {
        let mode = |refresh| {
            drm::control::Mode::from(drm_ffi::drm_mode_modeinfo {
                hdisplay: 1,
                vdisplay: 1,
                vrefresh: refresh,
                // SAFETY: this plain DRM ABI record accepts zero for other fields.
                ..unsafe { std::mem::zeroed() }
            })
        };
        let old_connectors = HashSet::from([handle(30)]);
        let old = NativeBlackTarget {
            mode: mode(60),
            connectors: old_connectors.clone(),
            config: Some(config()),
            planeless: false,
            modeset_fallback: false,
        };
        assert!(
            !old.matches(mode(144), &old_connectors),
            "old capability cannot serve a same-size new mode"
        );
        let target = old.config.as_ref().unwrap();
        let current = disabled_planes([handle(10)]);
        // Mode refresh-rate/connector changes do not change opaque pixels.
        let reused = configuration_candidate(&current, Some(&target), handle(10), (1, 1).into()).unwrap();
        assert_eq!(reused.buffer.fb, target.buffer.fb);
        if let (ScanoutBuffer::Swapchain(a), ScanoutBuffer::Swapchain(b)) =
            (&reused.buffer.buffer, &target.buffer.buffer)
        {
            assert!(Arc::ptr_eq(a, b));
        } else {
            panic!("expected the same immutable target");
        }
        assert!(configuration_candidate(&current, Some(&target), handle(10), (2, 1).into()).is_none());
        assert!(
            needs_buffered_fallback(true, true),
            "modeset-only capability retains this fallback"
        );
        let refreshed = NativeBlackTarget {
            mode: mode(144),
            connectors: old_connectors.clone(),
            config: Some(reused),
            planeless: false,
            modeset_fallback: false,
        };
        assert!(refreshed.matches(mode(144), &old_connectors));
        assert!(!refreshed.matches(mode(144), &HashSet::from([handle(31)])));
    }

    #[test]
    fn black_then_cursor_then_composition_repaints_the_entire_primary() {
        let mut repaint = NativeBlackRepaint::default();
        assert_eq!(repaint.age(3), 3);
        repaint.request();
        repaint.accepted(true, true); // Actual full native-black commit.
        repaint.accepted(false, false); // Cursor-only commit cannot consume it.
        assert_eq!(repaint.age(3), 0);
        assert!(
            repaint.pending(),
            "omit FB_DAMAGE_CLIPS for the first composition"
        );
        // A failed composition never reaches accepted(), so its retry is full.
        assert_eq!(repaint.age(2), 0);
        repaint.accepted(true, false);
        assert_eq!(repaint.age(2), 2);
        assert!(!repaint.pending());
    }

    #[test]
    fn modeset_only_planeless_state_is_not_adopted_for_an_ordinary_flip() {
        let target = Arc::new(());
        let mut tested_allowance = None;
        let accepted = probe_planeless(false, false, |allow_modeset| {
            tested_allowance = Some(allow_modeset);
            if allow_modeset {
                Ok(())
            } else {
                Err("kernel requires ALLOW_MODESET")
            }
        });
        let config = shield_configuration(accepted, Some(target.clone())).unwrap();
        assert_eq!(tested_allowance, Some(false));
        assert!(
            config.is_some(),
            "steady-state page flip must keep the buffered shield"
        );
        assert_eq!(Arc::strong_count(&target), 2);

        let accepted = probe_planeless(false, true, |allow_modeset| {
            assert!(allow_modeset, "the real pending modeset also allows modesetting");
            Ok::<_, ()>(())
        });
        assert!(accepted);
        let fallback =
            shield_configuration(!needs_buffered_fallback(accepted, true), Some(target.clone())).unwrap();
        assert!(
            fallback.is_some(),
            "pending modeset acceptance does not prove the next ordinary flip"
        );
    }

    #[test]
    fn completed_same_size_modeset_releases_only_the_retained_fallback_after_steady_probe() {
        let candidate = config();
        let ScanoutBuffer::Swapchain(slot) = &candidate.buffer.buffer else {
            panic!("expected slot");
        };
        let weak = Arc::downgrade(slot);
        let mut retained = NativeBlackTarget::<TestBuffer, TestFramebuffer>::retained_configuration(
            true,
            true,
            Some(candidate.clone()),
        )
        .unwrap();
        // The initial probe only permits a modeset; its shield remains owned.
        drop(candidate);
        assert!(weak.upgrade().is_some());
        let modeset_still_pending = true;
        let first =
            NativeBlackTarget::retained_configuration(true, modeset_still_pending, retained.take()).unwrap();
        assert!(first.is_some());
        // An ordinary committed probe now accepted the exact all-planes-off
        // request. It releases the configuration owner, without inventing a
        // physical black frame or revoking another current/pending reader.
        let scanned_out = first.as_ref().unwrap().clone();
        let refreshed = NativeBlackTarget::retained_configuration(true, false, first).unwrap();
        assert!(refreshed.is_none());
        assert!(
            weak.upgrade().is_some(),
            "actual scanout reader still pins its shield"
        );
        drop(scanned_out);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn refused_committed_probe_preserves_exact_fallback_while_accepted_probe_needs_no_candidate() {
        let candidate = config();
        let framebuffer = candidate.buffer.fb.clone();
        let fallback = NativeBlackTarget::retained_configuration(false, false, Some(candidate))
            .unwrap()
            .unwrap();
        assert_eq!(fallback.buffer.fb, framebuffer);
        assert!(
            NativeBlackTarget::<TestBuffer, TestFramebuffer>::retained_configuration(true, false, None)
                .unwrap()
                .is_none()
        );
        assert!(matches!(
            NativeBlackTarget::<TestBuffer, TestFramebuffer>::retained_configuration(false, false, None),
            Err(NativeBlackError::MissingBlackTarget)
        ));
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
