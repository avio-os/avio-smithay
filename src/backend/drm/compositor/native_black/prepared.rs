//! Same-output cold native configuration, with no mutable controller handoff.
use super::*;
use crate::backend::allocator::PreparedSwapchainResize;
use crate::backend::drm::surface::PreparedSurfaceMode;

/// Immutable allocator/native surface lineage captured before realtime work.
/// It owns no mutable DRM compositor and creates no renderer/device/context.
#[derive(Debug)]
pub struct NativeBlackAllocator<A: Allocator, F> {
    composition: CompositionAllocator<A, F>,
    layer: Option<CompositionAllocator<A, F>>,
    surface: Arc<DrmSurface>,
    supports_fencing: bool,
    preferred_modifiers: Vec<DrmModifier>,
}

/// Cold mode/shield preparation. After adoption this packet retains every
/// displaced mode blob, shield and swapchain control for cold disposal.
#[derive(Debug)]
pub struct PreparedNativeBlack<B: Buffer, F: Framebuffer> {
    surface: Arc<DrmSurface>,
    mode: drm::control::Mode,
    prepared_mode: PreparedSurfaceMode,
    resize: PreparedSwapchainResize<B>,
    layer_resize: Option<PreparedSwapchainResize<B>>,
    target: Option<NativeBlackTarget<B, F>>,
    format: DrmFourcc,
    expected_modifiers: Vec<DrmModifier>,
    desired_modifiers: Vec<DrmModifier>,
    expected_layer: Option<(DrmFourcc, Vec<DrmModifier>)>,
    rendering_complete: bool,
    tested: bool,
    adopted: bool,
    supports_fencing: bool,
}

/// Exact nonblocking adoption outcome. Busy/stale retains the entire packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedNativeBlackAdoption {
    /// Pending mode and completed shield were adopted together.
    Adopted(NativeBlackKind),
    /// An actual pending/queued frame owns the configuration lane.
    Busy,
    /// Surface, pending state or format no longer matches.
    Stale,
    /// The packet lacks completed rendering or an exact capability test.
    Incomplete,
}

impl<B: Buffer, F: Framebuffer> PreparedNativeBlack<B, F> {
    /// Exact physical target extent.
    pub fn size(&self) -> Size<i32, Physical> {
        (i32::from(self.mode.size().0), i32::from(self.mode.size().1)).into()
    }
    /// Already-exported metadata; this performs no export or allocation.
    pub fn dmabuf(&self) -> Option<Dmabuf> {
        let config = self.target.as_ref()?.config.as_ref()?;
        let ScanoutBuffer::Swapchain(slot) = &config.buffer.buffer else {
            return None;
        };
        slot.userdata().get::<Dmabuf>().cloned()
    }
    /// Install the real clear completion before observing it. Interruption
    /// leaves the exact BO and original proof together in this packet.
    pub fn complete_opaque_black_render(&mut self, sync: SyncPoint) -> Result<(), NativeBlackError> {
        let config = self
            .target
            .as_mut()
            .and_then(|target| target.config.as_mut())
            .ok_or(NativeBlackError::MissingBlackTarget)?;
        config.sync = Some((sync, None));
        config
            .sync
            .as_ref()
            .unwrap()
            .0
            .wait()
            .map_err(|_| NativeBlackError::RenderCompletionInterrupted)?;
        config.sync = None;
        self.rendering_complete = true;
        Ok(())
    }
    /// Cold TEST_ONLY against the original surface and prepared mode. Every
    /// plane assigned to this CRTC is reset; no sibling output is committed.
    pub fn test_configuration(&mut self, force_buffered: bool) -> Result<NativeBlackKind, NativeBlackError> {
        if !self.rendering_complete {
            return Err(NativeBlackError::MissingBlackTarget);
        }
        let fd = self.surface.device_fd();
        let mut handles = vec![self.surface.plane()];
        for handle in fd.plane_handles().map_err(NativeBlackError::PlaneEnumeration)? {
            let info = fd.get_plane(handle).map_err(NativeBlackError::PlaneEnumeration)?;
            if info.crtc() == Some(self.surface.crtc()) && !handles.contains(&handle) {
                handles.push(handle);
            }
        }
        let mut frame = disabled_planes(handles);
        for (handle, _) in &frame.planes {
            frame
                .reset_plane_claims
                .push(
                    self.surface
                        .claim_plane(*handle)
                        .ok_or(NativeBlackError::PlaneClaimDenied)?,
                )
                .expect("cold native configuration plane list");
        }
        let modeset = self.prepared_mode.requires_modeset();
        frame.set_state(
            self.surface.plane(),
            PlaneState {
                skip: false,
                config: self.target.as_ref().and_then(|target| target.config.clone()),
                ..Default::default()
            },
        );
        // A modesetting planeless success alone does not prove the fallback
        // that a later ordinary page flip may need. Test the actual BO too.
        self.surface.test_prepared_mode(
            &self.prepared_mode,
            frame.build_planes(
                &self.surface,
                self.supports_fencing,
                true,
                PlaneSyncMode::TestOnly,
            ),
            modeset,
        )?;
        frame.set_state(
            self.surface.plane(),
            PlaneState {
                skip: false,
                ..Default::default()
            },
        );
        let planeless = !force_buffered
            && self
                .surface
                .test_prepared_mode(
                    &self.prepared_mode,
                    frame.build_planes(
                        &self.surface,
                        self.supports_fencing,
                        true,
                        PlaneSyncMode::TestOnly,
                    ),
                    modeset,
                )
                .is_ok();
        let target = self.target.as_mut().unwrap();
        target.planeless = planeless;
        target.modeset_fallback = planeless && modeset;
        if planeless && !modeset {
            target.config = None;
        }
        target.reset_claims = frame.reset_plane_claims.iter().cloned().collect();
        self.tested = true;
        Ok(if planeless {
            NativeBlackKind::Planeless
        } else {
            NativeBlackKind::Buffered
        })
    }
}

impl<A, F> NativeBlackAllocator<A, F>
where
    A: Allocator + Clone,
    A::Buffer: AsDmabuf,
    A::Error: std::error::Error + Send + Sync + 'static,
    <A::Buffer as AsDmabuf>::Error: std::error::Error + Send + Sync + 'static,
    F: ExportFramebuffer<A::Buffer> + Clone,
    F::Framebuffer: std::fmt::Debug + Send + Sync + 'static,
    F::Error: std::error::Error + Send + Sync + 'static,
{
    /// Cold validation/allocation/export; no live compositor is borrowed.
    pub fn prepare(
        &mut self,
        mode: drm::control::Mode,
        restore_modifiers: bool,
    ) -> FrameResult<PreparedNativeBlack<A::Buffer, F::Framebuffer>, A, F> {
        let prepared_mode = self.surface.prepare_mode(mode).map_err(FrameError::DrmError)?;
        let connectors = prepared_mode.connectors().clone();
        let expected_modifiers = self.composition.modifiers().to_vec();
        let desired_modifiers = if restore_modifiers && !self.preferred_modifiers.is_empty() {
            self.preferred_modifiers.clone()
        } else {
            expected_modifiers.clone()
        };
        let expected_layer = self
            .layer
            .as_ref()
            .map(|layer| (layer.format(), layer.modifiers().to_vec()));
        let (width, height) = mode.size();
        let mut composition = self.composition.clone();
        composition.set_extent(u32::from(width), u32::from(height));
        composition.set_modifiers(desired_modifiers.clone());
        let mut buffers = composition.prepare(1)?;
        let slot = Arc::new(buffers.slots.pop().ok_or(FrameError::NoFreeSlotsError)?);
        let fb = slot
            .userdata()
            .get::<CachedDrmFramebuffer<F::Framebuffer>>()
            .ok_or(FrameError::NoFramebuffer)?
            .clone();
        let config = PlaneConfig {
            properties: PlaneProperties {
                src: Rectangle::from_size((f64::from(width), f64::from(height)).into()),
                dst: Rectangle::from_size((i32::from(width), i32::from(height)).into()),
                transform: Transform::Normal,
                alpha: 1.0,
                format: slot.format(),
            },
            buffer: DrmScanoutBuffer {
                buffer: ScanoutBuffer::Swapchain(slot),
                fb,
            },
            damage_clips: None,
            plane_claim: self
                .surface
                .claim_plane(self.surface.plane())
                .ok_or(FrameError::NativeBlack(NativeBlackError::PlaneClaimDenied))?,
            sync: None,
        };
        Ok(PreparedNativeBlack {
            surface: self.surface.clone(),
            mode,
            prepared_mode,
            resize: PreparedSwapchainResize::new(u32::from(width), u32::from(height)),
            layer_resize: self
                .layer
                .as_ref()
                .map(|_| PreparedSwapchainResize::new(u32::from(width), u32::from(height))),
            target: Some(NativeBlackTarget {
                config: Some(config),
                planeless: false,
                modeset_fallback: false,
                mode,
                connectors,
                reset_claims: Vec::new(),
            }),
            format: self.composition.format(),
            expected_modifiers,
            desired_modifiers,
            expected_layer,
            rendering_complete: false,
            tested: false,
            adopted: false,
            supports_fencing: self.supports_fencing,
        })
    }
    /// Cold-clone allocator successors for the existing target helper.
    pub fn composition_configuration(
        &self,
        packet: &PreparedNativeBlack<A::Buffer, F::Framebuffer>,
    ) -> (CompositionAllocator<A, F>, Option<CompositionAllocator<A, F>>) {
        let (width, height) = packet.mode.size();
        let mut primary = self.composition.clone();
        primary.set_extent(u32::from(width), u32::from(height));
        primary.set_modifiers(packet.desired_modifiers.clone());
        let layer = self.layer.as_ref().map(|layer| {
            let mut layer = layer.clone();
            layer.set_extent(u32::from(width), u32::from(height));
            layer
        });
        (primary, layer)
    }
    /// Adopt the same cold-created successor; displaced metadata stays in the
    /// returned task until its cold owner disposes it.
    pub fn adopt_composition_configuration(
        &mut self,
        primary: &mut CompositionAllocator<A, F>,
        layer: &mut Option<CompositionAllocator<A, F>>,
    ) {
        std::mem::swap(&mut self.composition, primary);
        std::mem::swap(&mut self.layer, layer);
    }
}

impl<A, F, U, G> DrmCompositor<A, F, U, G>
where
    A: Allocator + Clone,
    A::Error: std::error::Error + Send + Sync + 'static,
    A::Buffer: AsDmabuf,
    <A::Buffer as AsDmabuf>::Error: std::error::Error + Send + Sync + 'static,
    F: ExportFramebuffer<A::Buffer> + Clone,
    F::Framebuffer: std::fmt::Debug + Send + Sync + 'static,
    F::Error: std::error::Error + Send + Sync + 'static,
    G: AsFd + Clone,
{
    /// Capture actual plane/renderer compatibility once before realtime work.
    pub fn native_black_allocator(&self, renderer_modifiers: &[DrmModifier]) -> NativeBlackAllocator<A, F> {
        let code = self.swapchain.format();
        let preferred_modifiers = self
            .planes
            .primary
            .iter()
            .find(|plane| plane.handle == self.surface.plane())
            .map(|plane| {
                plane
                    .formats
                    .iter()
                    .filter(|format| {
                        format.code == code
                            && format.modifier != DrmModifier::Invalid
                            && renderer_modifiers.contains(&format.modifier)
                    })
                    .map(|format| format.modifier)
                    .collect()
            })
            .unwrap_or_default();
        NativeBlackAllocator {
            composition: self.composition_allocator(),
            layer: self.output_layer_composition_allocator(),
            surface: self.surface.clone(),
            supports_fencing: self.supports_fencing,
            preferred_modifiers,
        }
    }
    /// Stage-only adoption. Stale/busy never mutates live configuration.
    /// Success swaps displaced native ownership into the exact cold packet.
    pub fn adopt_native_black_configuration(
        &mut self,
        packet: &mut PreparedNativeBlack<A::Buffer, F::Framebuffer>,
    ) -> PreparedNativeBlackAdoption {
        if !packet.rendering_complete || !packet.tested || packet.adopted {
            return PreparedNativeBlackAdoption::Incomplete;
        }
        if !self.is_frame_pipeline_idle() {
            return PreparedNativeBlackAdoption::Busy;
        }
        let layer_matches = match (&self.output_layer_swapchain, &packet.expected_layer) {
            (None, None) => true,
            (Some(chain), Some((code, modifiers))) => {
                chain.format() == *code && chain.modifiers() == modifiers
            }
            _ => false,
        };
        if !Arc::ptr_eq(&self.surface, &packet.surface)
            || self.swapchain.format() != packet.format
            || self.swapchain.modifiers() != packet.expected_modifiers
            || !layer_matches
            || !self.surface.adopt_prepared_mode(&mut packet.prepared_mode)
        {
            return PreparedNativeBlackAdoption::Stale;
        }
        self.swapchain
            .adopt_prepared_modifiers(&mut packet.resize, &mut packet.desired_modifiers);
        if let (Some(chain), Some(resize)) = (&mut self.output_layer_swapchain, &mut packet.layer_resize) {
            chain.adopt_prepared_resize(resize);
        }
        let kind = if packet.target.as_ref().unwrap().planeless {
            NativeBlackKind::Planeless
        } else {
            NativeBlackKind::Buffered
        };
        std::mem::swap(&mut self.native_black, &mut packet.target);
        self.native_black_enabled = true;
        self.native_black_repaint.request();
        packet.adopted = true;
        PreparedNativeBlackAdoption::Adopted(kind)
    }
}
