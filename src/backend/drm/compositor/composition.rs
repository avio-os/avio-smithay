//! Demand-owned preparation of the compositor's primary target.

use super::*;
use crate::backend::allocator::{Fourcc, Modifier};

pub(super) fn composition_primary<A, F>(
    cached: &mut Option<PlaneState<A::Buffer, F::Framebuffer>>,
    swapchain: &mut Swapchain<A>,
    surface: &DrmSurface,
    framebuffer_exporter: &F,
    primary_is_opaque: bool,
    current_size: Size<i32, Physical>,
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
    let primary_plane_buffer = swapchain
        .acquire()
        .map_err(FrameError::Allocator)?
        .ok_or(FrameError::NoFreeSlotsError)?;

    // It is safe to call export multiple times as the Slot will cache the dmabuf for us
    let dmabuf = primary_plane_buffer.export().map_err(FrameError::AsDmabufError)?;

    // Let's check if we already have a cached framebuffer for this Slot, if not try to export
    // it and use the Slot userdata to cache it
    let maybe_buffer = primary_plane_buffer
        .userdata()
        .get::<CachedDrmFramebuffer<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>();
    if maybe_buffer.is_none() {
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
    use super::{select_primary, PrimaryPreparation};
    use std::cell::Cell;

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
    device_fd: crate::backend::drm::DrmDeviceFd,
    crtc: crtc::Handle,
    slots: Vec<Slot<B>>,
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
#[derive(Debug)]
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
        }
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
        } = buffers;
        self.swapchain
            .adopt(slots)
            .map_err(|rejected| PreparedCompositionBuffers {
                device_fd,
                crtc,
                slots: rejected.slots,
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
}
