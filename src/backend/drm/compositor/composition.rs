//! Demand-owned preparation of the compositor's primary target.

use super::*;
use crate::backend::allocator::{Fourcc, Modifier};

/// Fixed, allocation-free retirement handoff for both composition pools.
#[derive(Debug)]
pub struct RetiredCompositionBuffers<B: Buffer> {
    primary: [Option<crate::backend::allocator::RetiredSlot<B>>; crate::backend::allocator::SLOT_CAP],
    output_layer: [Option<crate::backend::allocator::RetiredSlot<B>>; crate::backend::allocator::SLOT_CAP],
}

impl<B: Buffer> RetiredCompositionBuffers<B> {
    /// Actual allocations in this disposal handoff.
    pub fn len(&self) -> usize {
        self.primary
            .iter()
            .chain(self.output_layer.iter())
            .flatten()
            .count()
    }

    /// Whether no allocation owners were detached.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Actual reusable composition allocations, excluding the immutable shield.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompositionBufferCounts {
    /// Actual primary composition allocations.
    pub primary: usize,
    /// Actual above-primary composition allocations.
    pub output_layer: usize,
}

// Direct ARGB scanout can be fully opaque. Its alpha-channel capability is
// not evidence of a compositor-rendered transparent hole over an underlay.
pub(super) fn underlay_preserves_visibility(
    element_is_opaque: bool,
    primary_has_alpha: bool,
    primary_can_holepunch: bool,
) -> bool {
    element_is_opaque && primary_has_alpha && primary_can_holepunch
}

pub(super) fn composition_primary<A, F>(
    cached: &mut Option<PlaneState<A::Buffer, F::Framebuffer>>,
    swapchain: &mut Swapchain<A>,
    surface: &DrmSurface,
    framebuffer_exporter: &F,
    primary_is_opaque: bool,
    current_size: Size<i32, Physical>,
    prepared_only: bool,
) -> FrameResult<PlaneState<A::Buffer, F::Framebuffer>, A, F>
where
    A: Allocator,
    A::Buffer: AsDmabuf,
    A::Error: std::error::Error + Send + Sync + 'static,
    <A::Buffer as AsDmabuf>::Error: std::error::Error + Send + Sync + 'static,
    F: ExportFramebuffer<A::Buffer>,
    F::Framebuffer: std::fmt::Debug + Send + Sync + 'static,
    F::Error: std::error::Error + Send + Sync + 'static,
{
    if let Some(state) = cached {
        return Ok(state.clone());
    }
    let primary_plane_buffer = if prepared_only {
        swapchain.acquire_existing().ok_or_else(|| {
            if swapchain.allocated_slots() == 0 {
                FrameError::CompositionTargetUnavailable
            } else {
                FrameError::NoFreeSlotsError
            }
        })?
    } else {
        swapchain
            .acquire()
            .map_err(FrameError::Allocator)?
            .ok_or(FrameError::NoFreeSlotsError)?
    };

    // It is safe to call export multiple times as the Slot will cache the dmabuf for us
    let dmabuf = primary_plane_buffer.export().map_err(FrameError::AsDmabufError)?;

    // Let's check if we already have a cached framebuffer for this Slot, if not try to export
    // it and use the Slot userdata to cache it
    let maybe_buffer = primary_plane_buffer
        .userdata()
        .get::<CachedDrmFramebuffer<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>();
    if maybe_buffer.is_none() {
        if prepared_only {
            return Err(FrameError::CompositionTargetUnavailable);
        }
        let fb_buffer = framebuffer_exporter
            .add_framebuffer(
                surface.device_fd(),
                ExportBuffer::Allocator(&primary_plane_buffer),
                primary_is_opaque,
            )
            .map_err(FrameError::FramebufferExport)?
            .ok_or(FrameError::NoFramebuffer)?;
        primary_plane_buffer
            .userdata()
            .insert_if_missing_threadsafe(|| CachedDrmFramebuffer::new(DrmFramebuffer::Exporter(fb_buffer)));
    }

    // This unwrap is safe as we error out above if we were unable to export a framebuffer
    let fb = primary_plane_buffer
        .userdata()
        .get::<CachedDrmFramebuffer<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>()
        .unwrap()
        .clone();

    // We want to make sure we can actually scan-out the primary plane, so
    // explicitly set skip to false
    let plane_claim = surface.claim_plane(surface.plane()).ok_or_else(|| {
        error!("failed to claim primary plane");
        FrameError::PrimaryPlaneClaimFailed
    })?;
    let primary_plane_state: PlaneState<
        <A as Allocator>::Buffer,
        <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer,
    > = PlaneState {
        skip: false,
        needs_test: false,
        element_state: None,
        config: Some(PlaneConfig {
            properties: PlaneProperties {
                src: Rectangle::from_size(dmabuf.size()).to_f64(),
                dst: Rectangle::from_size(current_size),
                // NOTE: We do not apply the transform to the primary plane as this is handled by the dtr/renderer
                transform: Transform::Normal,
                alpha: 1.0,
                format: primary_plane_buffer.format(),
            },
            buffer: DrmScanoutBuffer {
                buffer: ScanoutBuffer::Swapchain(Arc::new(primary_plane_buffer)),
                fb,
            },
            damage_clips: None,
            plane_claim,
            sync: None,
        }),
    };

    *cached = Some(primary_plane_state.clone());
    Ok(primary_plane_state)
}

/// One tested primary candidate or the composition target acquired after it.
/// The closure boundary keeps target allocation out of a successful direct path.
pub(super) enum PrimaryPreparation<D, C> {
    Direct(D),
    Composition(C),
}

pub(super) fn select_primary<D, C, E>(
    tested_direct: Option<D>,
    compose: impl FnOnce() -> Result<C, E>,
) -> Result<PrimaryPreparation<D, C>, E> {
    match tested_direct {
        Some(direct) => Ok(PrimaryPreparation::Direct(direct)),
        None => compose().map(PrimaryPreparation::Composition),
    }
}

#[cfg(test)]
mod tests {
    use super::{select_primary, underlay_preserves_visibility, PrimaryPreparation};
    use std::cell::Cell;

    #[test]
    fn direct_opaque_argb_primary_cannot_hide_an_auxiliary_underlay() {
        // The direct-first primary TEST_ONLY succeeded; it is an opaque
        // producer ARGB buffer rather than a target that Stage may hole punch.
        assert!(!underlay_preserves_visibility(true, true, false));
        assert!(underlay_preserves_visibility(true, true, true));
        assert!(!underlay_preserves_visibility(false, true, true));
        assert!(!underlay_preserves_visibility(true, false, true));
    }

    #[test]
    fn accepted_plane_test_never_acquires_a_composition_target() {
        let allocations = Cell::new(0);
        let driver_test = Ok::<_, ()>(41u64);
        let result = select_primary(driver_test.ok(), || {
            allocations.set(allocations.get() + 1);
            Err::<u64, _>("allocation must not happen")
        })
        .unwrap();
        assert!(matches!(result, PrimaryPreparation::Direct(41)));
        assert_eq!(allocations.get(), 0);
    }

    #[test]
    fn rejected_plane_test_acquires_once_and_preserves_allocation_failure() {
        let allocations = Cell::new(0);
        let driver_test = Err::<u64, _>("unsupported plane configuration");
        let result = select_primary(driver_test.ok(), || {
            allocations.set(allocations.get() + 1);
            Err::<u64, _>("owner has no target ready")
        });
        assert!(matches!(result, Err("owner has no target ready")));
        assert_eq!(allocations.get(), 1);
    }
}

/// Framebuffer-ready composition buffers with exact DRM output custody.
pub struct PreparedCompositionBuffers<B: Buffer> {
    pub(super) device_fd: crate::backend::drm::DrmDeviceFd,
    pub(super) crtc: crtc::Handle,
    pub(super) slots: Vec<Slot<B>>,
    output_layer: bool,
}

impl<B: Buffer> PreparedCompositionBuffers<B> {
    /// Whether these allocations belong to the above-primary output layer.
    pub fn is_output_layer(&self) -> bool {
        self.output_layer
    }
    /// Already-exported DMA-BUFs, for renderer import preparation on a cold
    /// owner turn before the batch becomes available to realtime rendering.
    pub fn dmabufs(&self) -> impl Iterator<Item = crate::backend::allocator::dmabuf::Dmabuf> + '_ {
        self.slots.iter().map(|slot| {
            slot.userdata()
                .get::<crate::backend::allocator::dmabuf::Dmabuf>()
                .expect("prepared composition buffer was exported")
                .clone()
        })
    }
}

impl<B: Buffer> std::fmt::Debug for PreparedCompositionBuffers<B> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedCompositionBuffers")
            .field("crtc", &self.crtc)
            .field("slot_count", &self.slots.len())
            .finish()
    }
}

/// An output owner's off-thread composition allocator and framebuffer exporter.
///
/// Construct on the configuration path, move to an allocation thread, and adopt
/// its prepared slots back into the same output. Dropping it retains no targets.
#[derive(Debug, Clone)]
pub struct CompositionAllocator<A: Allocator, F> {
    allocator: A,
    framebuffer_exporter: F,
    device_fd: crate::backend::drm::DrmDeviceFd,
    crtc: crtc::Handle,
    width: u32,
    height: u32,
    fourcc: Fourcc,
    modifiers: Vec<Modifier>,
    primary_is_opaque: bool,
    output_layer: bool,
}

impl<A, F> CompositionAllocator<A, F>
where
    A: Allocator,
    A::Buffer: AsDmabuf,
    A::Error: std::error::Error + Send + Sync + 'static,
    <A::Buffer as AsDmabuf>::Error: std::error::Error + Send + Sync + 'static,
    F: ExportFramebuffer<A::Buffer>,
    F::Framebuffer: std::fmt::Debug + Send + Sync + 'static,
    F::Error: std::error::Error + Send + Sync + 'static,
{
    pub(super) fn set_extent(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
    pub(super) fn format(&self) -> Fourcc {
        self.fourcc
    }
    pub(super) fn modifiers(&self) -> &[Modifier] {
        &self.modifiers
    }
    pub(super) fn set_modifiers(&mut self, modifiers: Vec<Modifier>) {
        self.modifiers = modifiers;
    }
    /// Allocate, export, and create DRM framebuffers outside frame rendering.
    ///
    /// A partial failure drops every target prepared by this call. The resulting
    /// slots carry both their exported DMA-BUF and framebuffer userdata across
    /// the thread handoff; composition only acquires those prepared objects.
    pub fn prepare(&mut self, count: usize) -> FrameResult<PreparedCompositionBuffers<A::Buffer>, A, F> {
        if count > crate::backend::allocator::SLOT_CAP {
            return Err(FrameError::NoFreeSlotsError);
        }
        let mut slots = Vec::with_capacity(count);
        for _ in 0..count {
            let buffer = self
                .allocator
                .create_buffer(self.width, self.height, self.fourcc, &self.modifiers)
                .map_err(FrameError::Allocator)?;
            let slot = Slot::new(buffer);
            let _ = slot.export().map_err(FrameError::AsDmabufError)?;
            let framebuffer = self
                .framebuffer_exporter
                .add_framebuffer(
                    &self.device_fd,
                    ExportBuffer::Allocator(&slot),
                    self.primary_is_opaque,
                )
                .map_err(FrameError::FramebufferExport)?
                .ok_or(FrameError::NoFramebuffer)?;
            slot.userdata().insert_if_missing_threadsafe(|| {
                CachedDrmFramebuffer::new(DrmFramebuffer::Exporter(framebuffer))
            });
            slots.push(slot);
        }
        Ok(PreparedCompositionBuffers {
            device_fd: self.device_fd.clone(),
            crtc: self.crtc,
            slots,
            output_layer: self.output_layer,
        })
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
    /// Snapshot this output's allocator/exporter contract for off-thread work.
    /// This call creates neither buffers nor framebuffers.
    pub fn composition_allocator(&self) -> CompositionAllocator<A, F> {
        let (width, height) = self.swapchain.dimensions();
        CompositionAllocator {
            allocator: self.swapchain.allocator.clone(),
            framebuffer_exporter: self.framebuffer_exporter.clone(),
            device_fd: self.surface.device_fd().clone(),
            crtc: self.surface.crtc(),
            width,
            height,
            fourcc: self.swapchain.format(),
            modifiers: self.swapchain.modifiers().to_vec(),
            primary_is_opaque: self.primary_is_opaque,
            output_layer: false,
        }
    }

    /// Snapshot the independently formatted above-primary composition target.
    pub fn output_layer_composition_allocator(&self) -> Option<CompositionAllocator<A, F>> {
        let swapchain = self.output_layer_swapchain.as_ref()?;
        let (width, height) = swapchain.dimensions();
        Some(CompositionAllocator {
            allocator: swapchain.allocator.clone(),
            framebuffer_exporter: self.framebuffer_exporter.clone(),
            device_fd: self.surface.device_fd().clone(),
            crtc: self.surface.crtc(),
            width,
            height,
            fourcc: swapchain.format(),
            modifiers: swapchain.modifiers().to_vec(),
            primary_is_opaque: false,
            output_layer: true,
        })
    }

    /// Adopt slots prepared by this output's composition allocator.
    /// A mode/format change rejects stale preparation atomically.
    pub fn adopt_composition_buffers(
        &mut self,
        buffers: PreparedCompositionBuffers<A::Buffer>,
    ) -> Result<(), PreparedCompositionBuffers<A::Buffer>> {
        if buffers.device_fd != *self.surface.device_fd() || buffers.crtc != self.surface.crtc() {
            return Err(buffers);
        }
        let PreparedCompositionBuffers {
            device_fd,
            crtc,
            slots,
            output_layer,
        } = buffers;
        let swapchain = if output_layer {
            let Some(layer) = self.output_layer_swapchain.as_mut() else {
                return Err(PreparedCompositionBuffers {
                    device_fd,
                    crtc,
                    slots,
                    output_layer,
                });
            };
            layer
        } else {
            &mut self.swapchain
        };
        swapchain
            .adopt(slots)
            .map_err(|rejected| PreparedCompositionBuffers {
                device_fd,
                crtc,
                slots: rejected.slots,
                output_layer,
            })
    }

    /// Retire unused composition targets at an owner-observed completion edge.
    /// Current, pending, queued, and prepared frame owners retain their slots.
    pub fn retire_unreferenced_composition_buffers(&mut self) -> usize {
        let mut retired = self.swapchain.retire_unreferenced();
        if let Some(layer) = self.output_layer_swapchain.as_mut() {
            retired += layer.retire_unreferenced();
        }
        retired
    }

    /// Enable allocation-free composition acquisition after the cold owner has
    /// prepared its allocator and wakeup lane. An unexpected plane-test miss
    /// returns a typed request for preparation; it never allocates or AddFB2s.
    pub fn use_prepared_composition_buffers(&mut self, enabled: bool) {
        self.composition_prepared_only = enabled;
    }

    /// Actual reusable targets retained by this output.
    pub fn composition_buffer_counts(&self) -> CompositionBufferCounts {
        CompositionBufferCounts {
            primary: self.swapchain.allocated_slots(),
            output_layer: self
                .output_layer_swapchain
                .as_ref()
                .map_or(0, Swapchain::allocated_slots),
        }
    }

    /// Whether the current physical frame still uses a composition allocation.
    /// Immutable opaque-black shield buffers belong to separate custody.
    pub fn current_frame_is_composited(&self) -> bool {
        !self.current_frame.opaque_black
            && !self.current_frame.native_black
            && self.current_frame.planes.iter().any(|(_, state)| {
                state
                    .config
                    .as_ref()
                    .is_some_and(|config| matches!(config.buffer.buffer, ScanoutBuffer::Swapchain(_)))
            })
    }

    /// Detach only targets no prepared/current/pending/queued frame owns. Call
    /// after the actual direct-scanout flip retires the previous frame; move
    /// the returned value to the output's off-thread disposal owner.
    pub fn take_unreferenced_composition_buffers(&mut self) -> RetiredCompositionBuffers<A::Buffer> {
        RetiredCompositionBuffers {
            primary: self.swapchain.take_unreferenced(),
            output_layer: self
                .output_layer_swapchain
                .as_mut()
                .map_or_else(|| std::array::from_fn(|_| None), Swapchain::take_unreferenced),
        }
    }
}
