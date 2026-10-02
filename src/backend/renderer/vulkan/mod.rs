//! Vulkan renderer implementation for Smithay composition paths.
//!
//! This module exposes [`VulkanRenderer`], [`VulkanTexture`], and [`VulkanTarget`], and
//! wires them into Smithay renderer traits for dma-buf import, memory upload/readback,
//! offscreen rendering, framebuffer blit, and explicit sync interop.
//!
//! # Renderer Lifecycle
//!
//! 1. Probe device extension requirements with [`VulkanRenderer::required_extensions`].
//! 2. Create the renderer using [`VulkanRenderer::new`].
//! 3. Inspect supported format/modifier combinations via [`VulkanRenderer::dmabuf_import_formats`]
//!    and [`VulkanRenderer::dmabuf_render_formats`] before allocator/compositor setup.
//! 4. Bind targets with [`crate::backend::renderer::Bind::bind`] and record drawing through
//!    [`crate::backend::renderer::Renderer::render`].
//! 5. Finalize frames with [`crate::backend::renderer::Frame::finish`] and hand returned
//!    [`crate::backend::renderer::sync::SyncPoint`] objects to presentation/scheduling code.
//! 6. Periodically call [`VulkanRenderer::cleanup_dmabuf_cache`] (or
//!    [`crate::backend::renderer::Renderer::cleanup_texture_cache`]) in long-running compositors.
//! 7. Optionally bound what the device keeps bound by calling
//!    [`VulkanRenderer::evict_idle_sampled_dmabuf_imports`] after a frame is submitted. Imported
//!    textures otherwise stay cached, used or not, until their dma-buf is dropped or the cache
//!    overflows.
//!
//! # Trait Behavior Notes
//!
//! - [`crate::backend::renderer::Renderer::render`] opens Vulkan command recording for one frame and
//!   returns a [`crate::backend::renderer::Frame`] implementation (`VulkanFrame`).
//! - [`crate::backend::renderer::Frame::finish`] submits recorded commands and returns a `SyncPoint`.
//!   Dropping a frame without `finish` aborts recording and never submits partial work.
//! - [`crate::backend::renderer::Bind`] currently accepts dma-buf render targets.
//! - [`crate::backend::renderer::ImportDma`] imports dma-bufs as sampled textures with strict
//!   format/modifier validation.
//! - [`crate::backend::renderer::ImportMem`] copies into a bounded, persistently mapped arena,
//!   batches upload commands into the next real render submission, and creates writable Vulkan
//!   textures for shm/memory-backed client paths. Bounded-capacity relief is a separate typed
//!   submission whose completion point is the only staging-reuse edge.
//! - [`crate::backend::renderer::ExportMem`] performs readback through transfer buffers and returns
//!   deterministic linear pixel data for supported formats.
//!
//! # Explicit Device-Memory Census
//!
//! [`VulkanRenderer::diagnostics`] reports successful explicit device-memory
//! allocation owners by texture, render target, import, scratch, and upload
//! reason. Bytes are the exact Vulkan allocation size rather than pixel-size
//! estimates. Each image/chunk owns one guard per actual memory binding;
//! cache aliases, texture clones, and submitted readers share that guard.
//! Error cleanup and final memory-owner destruction retire it after memory
//! teardown. Imported bytes describe Vulkan bindings to externally owned
//! storage and must not be added again to a producer's physical-buffer count.
//! Driver-internal memory is outside this explicit-allocation census.
//!
//! Use [`VulkanRenderer::allocation_phase_scope`] around initialization,
//! warmup, frame preparation/recording, or maintenance work on its owner
//! thread. The phase and allocating thread are included in per-event TRACE
//! records; retirement also records the retiring thread. Phase scopes are
//! device-specific, nested, and bound to their entering thread. Their fixed
//! 64-entry thread-local storage never grows; while an overflow scope is live,
//! all allocations on that thread are tagged `Unspecified`. Snapshots are
//! read-only atomic samples: a concurrent allocation/free can straddle their
//! reads, while a quiescent sample is exact. Device-loss teardown intentionally
//! skips unsafe Vulkan frees, so owner retirement after loss does not establish
//! that the driver reclaimed the corresponding physical memory.
//!
//! # Format And Modifier Expectations
//!
//! - Explicit modifier support is queried from Vulkan (`VK_EXT_image_drm_format_modifier`) and cached.
//! - Implicit modifier support (`Modifier::Invalid`) is advertised only when Vulkan reports support
//!   without explicit modifier metadata.
//! - Explicit modifier imports validate plane count, plane strides, plane offsets, and usage before
//!   creating Vulkan images.
//! - Plane file-descriptor numbers are not allocation identities. Imports compare dma-buf object
//!   identity, bind duplicated descriptors once for shared allocations, and use disjoint Vulkan
//!   plane bindings only when the modifier and requested usage explicitly support them.
//! - Compositor integration should intersect renderer-supported modifiers with plane/allocator
//!   capabilities instead of assuming one global modifier set.
//!
//! # Sync Expectations
//!
//! - [`crate::backend::renderer::Renderer::wait`] first resolves renderer-native fence payloads.
//! - If sync-file import support is available, `SyncPoint` FDs are imported into Vulkan wait paths.
//! - If explicit sync import/export is unavailable (or import fails with recoverable errors), behavior
//!   degrades to host-side blocking waits to preserve correctness.
//! - Capability probes are exposed through [`VulkanRenderer::supports_explicit_sync_import`],
//!   [`VulkanRenderer::supports_explicit_sync_export`], and
//!   [`VulkanRenderer::supports_timeline_semaphore`].
//!
//! # DRM Composition Integration Snippet
//!
//! ```ignore
//! use smithay::backend::{
//!     allocator::gbm::{GbmAllocator, GbmDevice},
//!     drm::{
//!         compositor::{DrmCompositor, FrameFlags},
//!         exporter::gbm::GbmFramebufferExporter,
//!         DrmDevice, DrmDeviceFd, DrmSurface,
//!     },
//!     renderer::{
//!         element::surface::WaylandSurfaceRenderElement,
//!         vulkan::VulkanRenderer,
//!     },
//!     vulkan::{version::Version, Instance, PhysicalDevice},
//! };
//! use smithay::output::Output;
//! use std::collections::HashSet;
//!
//! let instance = Instance::new(Version::VERSION_1_3, None)?;
//! let physical_device = PhysicalDevice::enumerate(&instance)?.next().expect("no Vulkan device");
//! let mut renderer = VulkanRenderer::new(&physical_device)?;
//!
//! let renderer_formats = renderer
//!     .dmabuf_render_formats()
//!     .iter()
//!     .copied()
//!     .collect::<HashSet<_>>();
//!
//! # let output: Output = todo!();
//! # let surface: DrmSurface = todo!();
//! # let allocator: GbmAllocator<DrmDeviceFd> = todo!();
//! # let exporter: GbmFramebufferExporter<DrmDeviceFd> = todo!();
//! # let drm_device: DrmDevice = todo!();
//! # let gbm: GbmDevice<DrmDeviceFd> = todo!();
//! let mut compositor: DrmCompositor<_, _, (), _> = DrmCompositor::new(
//!     &output,
//!     surface,
//!     None,
//!     allocator,
//!     exporter,
//!     [drm_fourcc::DrmFourcc::Argb8888],
//!     renderer_formats,
//!     drm_device.cursor_size(),
//!     Some(gbm),
//! )?;
//!
//! let elements: Vec<WaylandSurfaceRenderElement<VulkanRenderer>> = Vec::new();
//! let frame_result = compositor.render_frame::<_, _>(
//!     &mut renderer,
//!     &elements,
//!     [0.0, 0.0, 0.0, 1.0],
//!     FrameFlags::DEFAULT,
//! )?;
//!
//! if !frame_result.is_empty {
//!     compositor.queue_frame(())?;
//!     // ...wait for VBlank/page-flip event...
//!     let _user_data = compositor.frame_submitted()?;
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![allow(dead_code)]

pub(crate) mod allocation;
mod client_import_census;
pub use client_import_census::VulkanClientImportObserver;
mod blit;
mod descriptor;
mod device;
pub(crate) mod device_handle;
mod device_origin;
mod ordered_queue;
pub use device_origin::VulkanDeviceOrigin;
mod dmabuf;
mod error;
mod format;
mod frame;
mod material_tint;
pub use material_tint::VulkanMaterialTint;
#[cfg(feature = "wayland_frontend")]
mod host_memory;
mod image;
mod kawase;
#[cfg(test)]
mod kawase_calibration;
mod offscreen;
mod pipeline;
mod readback;
mod retired_views;
mod retirement_slot;
pub use retirement_slot::VulkanRetirementSlot;
mod retirement;
mod staging;
pub use staging::VulkanUploadStorage;
mod sync;
mod target;
mod texture;
mod upload;

pub use allocation::{
    VulkanAllocationObserver, VulkanAllocationPhase, VulkanAllocationPhaseGuard, VulkanAllocationReason,
    VulkanAllocationSnapshot, VulkanAllocationStats, VulkanImportedBackingSnapshot,
};
pub use blit::VulkanBlitChainStep;
pub use error::{VulkanRendererError, VulkanRendererErrorKind};
pub use frame::VulkanFrame;
pub use kawase::{VulkanKawaseEncoding, VulkanKawaseOutput, VulkanKawasePass};
pub use offscreen::VulkanOffscreenAllocator;
pub use target::VulkanTarget;
pub use texture::VulkanTexture;

use std::{ffi::CStr, time::Instant};

use crate::backend::{
    allocator::{dmabuf::Dmabuf, format::FormatSet, Format, Fourcc, Modifier},
    renderer::{ContextId, DebugFlags, TextureFilter},
    vulkan::PhysicalDevice,
};

use self::blit::BlitState;
use self::descriptor::DescriptorState;
use self::device::{DeviceDiagnostics, DeviceState};
use self::dmabuf::DmabufState;
use self::format::FormatCapabilities;
use self::pipeline::PipelineState;
use self::readback::ReadbackState;
use self::upload::UploadState;

/// Cache diagnostics for renderer-internal resource caches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VulkanCacheStats {
    /// Number of successful cache lookups.
    pub hits: u64,
    /// Number of cache misses requiring allocation or import work.
    pub misses: u64,
    /// Number of entries evicted from the cache.
    pub evictions: u64,
    /// Number of entries retired because their texture's image view was
    /// destroyed (the death-edge reclaim; these free capacity without any
    /// quiescence requirement beyond their own last use).
    pub dead_view_reclaims: u64,
}

/// Submission and timing diagnostics for renderer command dispatch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VulkanSubmissionStats {
    /// Number of queue submissions (asynchronous and blocking).
    pub total_submissions: u64,
    /// Number of blocking submissions.
    pub blocking_submissions: u64,
    /// Number of asynchronously submitted command buffers reclaimed after completion.
    pub reclaimed_submissions: u64,
    /// Average host-side CPU time spent in `vkQueueSubmit` for async submissions (nanoseconds).
    pub avg_submit_cpu_ns: u64,
    /// Maximum host-side CPU time spent in `vkQueueSubmit` for async submissions (nanoseconds).
    pub max_submit_cpu_ns: u64,
    /// Average async submission completion latency from submit to reclaim (nanoseconds).
    pub avg_completion_ns: u64,
    /// Maximum async submission completion latency from submit to reclaim (nanoseconds).
    pub max_completion_ns: u64,
}

/// Persistent memory-upload arena and batching diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VulkanUploadStats {
    /// Capacity of the sole owner-sized chunk serving slice and mapped-row uploads.
    pub owner_capacity_bytes: usize,
    /// Structural capacity declared by the live source/output owner.
    pub owner_structural_bytes: usize,
    /// Bytes above the structural extent required by the largest live generation.
    pub owner_extent_exception_bytes: usize,
    /// Successful transitions into a distinct larger-source extent exception.
    pub owner_extent_exceptions_total: u64,
    /// Bytes mapped by the sole owner-provisioned arena chunk.
    pub arena_capacity_bytes: usize,
    /// Bytes retained by pending or in-flight upload operations.
    pub arena_in_use_bytes: usize,
    /// Largest observed live reservation total.
    pub arena_high_water_bytes: usize,
    /// Number of persistent mapped chunks.
    pub arena_chunk_count: usize,
    /// Frame-path growth count; always zero for owner-provisioned storage.
    pub arena_growth_count: u64,
    /// Number of reservations deferred by the arena byte/chunk bounds.
    pub arena_deferred_count: u64,
    /// Operations currently waiting for the next render or capacity-edge submission.
    pub pending_operations: usize,
    /// Packed bytes currently waiting for submission.
    pub pending_bytes: usize,
    /// Number of upload batches attached to queue submissions.
    pub submitted_batches: u64,
    /// Number of upload operations attached to queue submissions.
    pub submitted_operations: u64,
    /// Packed bytes attached to queue submissions.
    pub submitted_bytes: u64,
}

/// Bounded descriptor-arena diagnostics. The stable-key cache target is a
/// performance policy; `arena_max_sets` is the independent safety bound.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VulkanDescriptorStats {
    /// Sets allocated across all live descriptor-pool pages.
    pub arena_capacity_sets: usize,
    /// Hard renderer-local set limit.
    pub arena_max_sets: usize,
    /// Number of lazily allocated descriptor-pool pages.
    pub arena_pool_count: usize,
    /// Stable texture keys currently cached.
    pub cached_sets: usize,
    /// Dead-view sets awaiting their last GPU completion.
    pub retired_sets: usize,
    /// Immediately rewriteable sets outside the cache.
    pub free_sets: usize,
    /// Sets bound by the current, not-yet-submitted recording.
    pub recording_sets: usize,
    /// Largest observed cache plus retired-set working set.
    pub arena_high_water_sets: usize,
    /// Number of bounded page growth operations.
    pub arena_growth_count: u64,
    /// Number of admissions deferred by incomplete GPU work at the hard bound.
    pub arena_deferred_count: u64,
}

/// Opaque queue frontier used to determine whether a fallible render scope
/// actually submitted work. Equality is renderer-local and wrap-safe for the
/// practical lifetime of a renderer epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanSubmissionSnapshot {
    next_submission_id: u64,
}

/// Aggregated runtime diagnostics for the Vulkan renderer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VulkanRendererDiagnostics {
    /// Exact successful explicit device-memory allocation owners by reason and creation phase.
    pub allocations: VulkanAllocationSnapshot,
    /// dma-buf import/bind cache diagnostics.
    pub dmabuf_cache: VulkanCacheStats,
    /// Texture descriptor cache diagnostics.
    pub descriptor_cache: VulkanCacheStats,
    /// Bounded descriptor-pool and transactional-use diagnostics.
    pub descriptors: VulkanDescriptorStats,
    /// Submission and completion timing diagnostics.
    pub submissions: VulkanSubmissionStats,
    /// Persistent staging and upload-batch diagnostics.
    pub uploads: VulkanUploadStats,
    /// Whether Vulkan debug markers are active for command-buffer labels.
    pub debug_markers_enabled: bool,
}

/// Vulkan renderer implementation under active phased development.
#[derive(Debug)]
pub struct VulkanRenderer {
    context_id: ContextId<VulkanTexture>,
    downscale_filter: TextureFilter,
    upscale_filter: TextureFilter,
    debug_flags: DebugFlags,
    device: DeviceState,
    formats: FormatCapabilities,
    dmabuf: DmabufState,
    upload: UploadState,
    readback: ReadbackState,
    blit: BlitState,
    descriptors: std::mem::ManuallyDrop<DescriptorState>,
    pipelines: std::mem::ManuallyDrop<PipelineState>,
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        // Both states are used by submitted command buffers. Their exact
        // objects follow the command pool to its proven-completion retiree.
        // SAFETY: Each field is taken once here and ManuallyDrop suppresses
        // the subsequent field destructor.
        let pipelines = unsafe { std::mem::ManuallyDrop::take(&mut self.pipelines) };
        let descriptors = unsafe { std::mem::ManuallyDrop::take(&mut self.descriptors) };
        self.device.retain_context_state(pipelines, descriptors);
    }
}

impl VulkanRenderer {
    /// Actual foreign-host transfer-source support queried from this physical
    /// device. The returned alignment is a requirement, not a claimed platform
    /// constant. Each source still needs seal, extent and memory-type validation.
    #[cfg(feature = "wayland_frontend")]
    pub fn host_memory_import_alignment(
        &self,
    ) -> Result<usize, crate::backend::renderer::MemoryHostUnavailable> {
        self.device.host_memory_alignment()
    }
    /// Returns the required device extensions for this renderer.
    pub fn required_extensions(physical_device: &PhysicalDevice) -> Vec<&'static CStr> {
        DeviceState::required_extensions(physical_device)
    }

    /// Creates a new Vulkan renderer and initializes device/queue infrastructure.
    pub fn new(physical_device: &PhysicalDevice) -> Result<Self, VulkanRendererError> {
        Self::from_device_state(DeviceState::new(physical_device)?)
    }

    /// Creates an output context without creating another logical device.
    pub fn from_device_origin(origin: &VulkanDeviceOrigin) -> Result<Self, VulkanRendererError> {
        Self::from_device_state(DeviceState::from_origin(origin)?)
    }

    /// Retains this context's GPU origin for other output contexts/export allocation.
    pub fn device_origin(&self) -> VulkanDeviceOrigin {
        VulkanDeviceOrigin::from_state(&self.device)
    }

    fn from_device_state(device: DeviceState) -> Result<Self, VulkanRendererError> {
        let physical_device = device.physical_device();
        let _initialization = device
            .shared_device()
            .allocation_ledger()
            .enter_phase(VulkanAllocationPhase::Initialization);
        let descriptors = DescriptorState::new(device.shared_device())?;
        let pipelines = PipelineState::new(device.shared_device(), descriptors.texture_layout())?;

        let formats = FormatCapabilities::new(physical_device)?;
        let readback = ReadbackState::new(device.shared_device().offscreen_ids());
        let context_id = ContextId::new();
        let dmabuf = DmabufState::new(context_id.erased());
        Ok(Self {
            context_id,
            downscale_filter: TextureFilter::Linear,
            upscale_filter: TextureFilter::Linear,
            debug_flags: DebugFlags::empty(),
            device,
            formats,
            dmabuf,
            upload: UploadState::default(),
            readback,
            blit: BlitState,
            descriptors: std::mem::ManuallyDrop::new(descriptors),
            pipelines: std::mem::ManuallyDrop::new(pipelines),
        })
    }

    /// Sets runtime debug flags.
    pub fn set_debug_flags(&mut self, flags: DebugFlags) {
        self.debug_flags = flags;
    }

    /// Returns the currently active runtime debug flags.
    pub fn debug_flags(&self) -> DebugFlags {
        self.debug_flags
    }

    /// Marks the renderer's Vulkan device lost and propagates the mark to the
    /// owning Vulkan instance teardown gate.
    pub fn mark_device_lost(&self) {
        self.device.mark_lost();
    }

    /// Returns whether the renderer's Vulkan device or owning instance has been
    /// marked lost.
    pub fn is_device_lost(&self) -> bool {
        self.device.is_lost()
    }

    /// The queue family selected for renderer command submissions.
    pub fn queue_family_index(&self) -> u32 {
        self.device.queue_family_index()
    }

    /// Returns whether timeline semaphore support was detected.
    pub fn supports_timeline_semaphore(&self) -> bool {
        self.device.capabilities().timeline_semaphore()
    }

    /// Returns whether importing native sync-file fds into Vulkan queue wait semaphores is supported.
    ///
    /// When this is `false`, `Renderer::wait` and `Frame::wait` fall back to blocking on
    /// the provided `SyncPoint` at the host level.
    pub fn supports_explicit_sync_import(&self) -> bool {
        self.device.supports_sync_file_import()
    }

    /// Returns whether submitted frame fences can be exported as native sync-file fds.
    pub fn supports_explicit_sync_export(&self) -> bool {
        self.device.supports_sync_file_export()
    }

    /// Returns formats accepted by [`crate::backend::renderer::ImportMem`] uploads.
    pub fn memory_upload_formats(&self) -> &[Fourcc] {
        self.upload.supported_formats()
    }

    /// Returns the enabled Vulkan device extensions.
    pub fn enabled_extensions(&self) -> &[&'static CStr] {
        self.device.enabled_extensions()
    }

    /// Supported dma-buf format+modifier combinations for texture import.
    pub fn dmabuf_import_formats(&self) -> &FormatSet {
        self.formats.import_formats()
    }

    /// Supported dma-buf format+modifier combinations for render-target binding.
    pub fn dmabuf_render_formats(&self) -> &FormatSet {
        self.formats.render_formats()
    }

    /// Supported dma-buf format+modifier combinations for render targets
    /// whose completed lower prefix may be copied by an inline framebuffer
    /// effect.
    pub fn dmabuf_framebuffer_effect_formats(&self) -> &FormatSet {
        self.formats.framebuffer_effect_formats()
    }

    /// Supported dma-buf format+modifier combinations for direct capture
    /// rendering and terminal transport blits.
    pub fn dmabuf_capture_formats(&self) -> &FormatSet {
        self.formats.capture_formats()
    }

    /// Returns whether a format+modifier is supported for dma-buf texture import.
    pub fn has_dmabuf_import_format(&self, format: Format) -> bool {
        self.formats.has_import_format(format)
    }

    /// Returns whether a format+modifier is supported for dma-buf render-target binding.
    pub fn has_dmabuf_render_format(&self, format: Format) -> bool {
        self.formats.has_render_format(format)
    }

    /// Returns whether a format+modifier is supported for a dma-buf
    /// framebuffer-effect target.
    pub fn has_dmabuf_framebuffer_effect_format(&self, format: Format) -> bool {
        self.formats.has_framebuffer_effect_format(format)
    }

    /// Supported import modifiers for the given DRM format code.
    pub fn import_modifiers(&self, code: Fourcc) -> &[Modifier] {
        self.formats.import_modifiers(code)
    }

    /// Supported render-target modifiers for the given DRM format code.
    pub fn render_modifiers(&self, code: Fourcc) -> &[Modifier] {
        self.formats.render_modifiers(code)
    }

    /// Supported framebuffer-effect target modifiers for the DRM format.
    pub fn framebuffer_effect_modifiers(&self, code: Fourcc) -> &[Modifier] {
        self.formats.framebuffer_effect_modifiers(code)
    }

    /// Intersects renderer import capabilities with a caller-provided modifier preference list.
    pub fn intersect_import_modifiers(&self, code: Fourcc, requested: &[Modifier]) -> Vec<Modifier> {
        self.formats.intersect_import_modifiers(code, requested)
    }

    /// Intersects renderer render-target capabilities with a caller-provided modifier preference list.
    pub fn intersect_render_modifiers(&self, code: Fourcc, requested: &[Modifier]) -> Vec<Modifier> {
        self.formats.intersect_render_modifiers(code, requested)
    }

    /// Intersects framebuffer-effect target capabilities with a
    /// caller-provided modifier preference list.
    pub fn intersect_framebuffer_effect_modifiers(
        &self,
        code: Fourcc,
        requested: &[Modifier],
    ) -> Vec<Modifier> {
        self.formats
            .intersect_framebuffer_effect_modifiers(code, requested)
    }

    /// Returns whether implicit modifier support (`Modifier::Invalid`) exists for import.
    pub fn supports_implicit_import_modifier(&self, code: Fourcc) -> bool {
        self.formats.supports_implicit_import_modifier(code)
    }

    /// Returns whether implicit modifier support (`Modifier::Invalid`) exists for render-target use.
    pub fn supports_implicit_render_modifier(&self, code: Fourcc) -> bool {
        self.formats.supports_implicit_render_modifier(code)
    }

    /// Import a dma-buf as a sampled texture.
    pub fn import_dmabuf_texture(&mut self, dmabuf: &Dmabuf) -> Result<VulkanTexture, VulkanRendererError> {
        self.dmabuf.import_texture(&self.device, &self.formats, dmabuf)
    }

    /// Bind a dma-buf for render-target usage.
    pub fn bind_dmabuf_target(&mut self, dmabuf: &Dmabuf) -> Result<VulkanTarget, VulkanRendererError> {
        self.dmabuf
            .bind_render_target(&self.device, &self.formats, dmabuf)
    }

    /// Bind immutable-image copy storage without the compositing pass's sRGB
    /// conversion. An unscaled source drawn with alpha one over transparent
    /// black retains its encoded premultiplied channels, including alpha.
    ///
    /// The destination must have an alpha channel: opaque DRM formats use a
    /// sampled-view component swizzle, which is not a legal attachment view.
    /// This is a resource-copy target, not a different display blend policy.
    pub fn bind_dmabuf_storage_copy_target(
        &mut self,
        dmabuf: &Dmabuf,
    ) -> Result<VulkanTarget, VulkanRendererError> {
        use crate::backend::allocator::Buffer;
        if !crate::backend::allocator::format::has_alpha(dmabuf.format().code) {
            return Err(VulkanRendererError::InvalidDmabuf(
                "storage-copy target requires identity alpha components",
            ));
        }
        let mut target = self.bind_dmabuf_target(dmabuf)?;
        target.encoding = target::VulkanTargetEncoding::PreserveStorage;
        Ok(target)
    }

    /// Whether every advertised framebuffer-effect/capture modifier for this
    /// format supports combined sampled and target usage. Mixed sets use copy capture.
    pub fn framebuffer_sampling_supported(&self, format: crate::backend::allocator::Fourcc) -> bool {
        self.formats.framebuffer_sampling_supported(format)
    }

    /// Clone renderer-origin image allocation for an off-frame provisioning helper.
    pub fn offscreen_allocator(&self) -> VulkanOffscreenAllocator {
        self.readback.offscreen_allocator(&self.device)
    }

    /// Bind a dma-buf as the active accumulator for inline framebuffer
    /// effects. The imported Vulkan image declares both color-attachment and
    /// transfer-source usage.
    pub fn bind_dmabuf_framebuffer_effect_target(
        &mut self,
        dmabuf: &Dmabuf,
    ) -> Result<VulkanTarget, VulkanRendererError> {
        self.dmabuf
            .bind_framebuffer_effect_target(&self.device, &self.formats, dmabuf)
    }

    /// Bind a dma-buf with the complete direct-render and terminal-blit usage
    /// contract required by compositor capture.
    pub fn bind_dmabuf_capture_target(
        &mut self,
        dmabuf: &Dmabuf,
    ) -> Result<VulkanTarget, VulkanRendererError> {
        self.dmabuf
            .bind_capture_target(&self.device, &self.formats, dmabuf)
    }

    /// Drop stale cached dma-buf imports.
    pub fn cleanup_dmabuf_cache(&mut self) {
        self.dmabuf.cleanup();
    }

    /// Drop up to `max` cached dma-buf texture imports that were last imported
    /// before `used_before`, least recently used first, and return how many
    /// were dropped.
    ///
    /// Dropping an import destroys the renderer's Vulkan image and imported
    /// memory for that dma-buf, so the device no longer keeps it bound; the
    /// dma-buf itself is untouched. A later [`Self::import_dmabuf_texture`] of
    /// the same dma-buf imports it again, exactly like a first import.
    ///
    /// Only imports made solely through [`Self::import_dmabuf_texture`] (or
    /// [`crate::backend::renderer::ImportDma`]) are candidates. An import that
    /// has been bound as a render, storage-copy, framebuffer-effect or capture
    /// target is kept: its contents are renderer-authored and callers may track
    /// their age. An import still referenced by a [`VulkanTexture`], a
    /// [`VulkanTarget`] or submitted GPU work is kept as well.
    ///
    /// This is opt-in residency control for a caller that owns a retention
    /// policy. The renderer never calls it, and [`Self::cleanup_dmabuf_cache`]
    /// is unchanged.
    pub fn evict_idle_sampled_dmabuf_imports(&mut self, used_before: Instant, max: usize) -> usize {
        self.dmabuf.evict_idle_sampled(used_before, max)
    }

    /// Import a pool/member buffer and pin its renderer cache entry until one
    /// matching [`Self::unpin_dmabuf_import`]. Pins survive usage upgrades and
    /// exempt metadata from capacity, idle and explicit sampled retirement.
    /// The caller owns allocation and membership; a texture handle alone is
    /// submitted-reader custody and does not imply a membership pin.
    pub fn pin_dmabuf_import(&mut self, dmabuf: &Dmabuf) -> Result<VulkanTexture, VulkanRendererError> {
        self.dmabuf.pin_texture(&self.device, &self.formats, dmabuf)
    }

    /// End one exact membership pin. Returns false for an absent/unpinned entry.
    pub fn unpin_dmabuf_import(&mut self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.unpin(dmabuf)
    }

    /// End one membership pin by exact weak source identity without extending
    /// the DMA-BUF lifetime. Used by owners handling source destruction.
    pub fn unpin_weak_dmabuf_import(&mut self, key: &crate::backend::allocator::dmabuf::WeakDmabuf) -> bool {
        self.dmabuf.unpin_weak(key)
    }

    /// Retire exactly the unpinned sampled-only imports named by their source
    /// identities. No clock or implicit capacity threshold chooses these.
    /// Targets retain their authored contents; submitted readers retain the
    /// old imported image independently through their GPU completion.
    pub fn retire_sampled_dmabuf_imports(
        &mut self,
        buffers: &[crate::backend::allocator::dmabuf::WeakDmabuf],
    ) -> usize {
        self.dmabuf.retire_sampled(buffers)
    }

    /// Prepare exact client source identities off-frame for attribution. Pool,
    /// owned-copy and target imports never count as client first imports.
    pub fn prepare_frame_client_sources(
        &mut self,
        sources: &[crate::backend::allocator::dmabuf::WeakDmabuf],
    ) {
        self.dmabuf.prepare_frame_client_sources(sources);
    }

    /// Bracket only actual frame work, including unsuccessful frame attempts.
    /// Control/preparation turns leave this false.
    pub fn set_frame_client_import_scope(&mut self, active: bool) {
        self.dmabuf.set_frame_client_scope(active);
    }

    /// Successful first client image creations made inside frame work. Cache
    /// hits, usage upgrades, failed imports and off-frame creations are omitted.
    pub fn client_first_imports_on_frame(&self) -> u64 {
        self.dmabuf.client_first_imports_on_frame()
    }

    /// Weak, numeric-only access to the exact first-client-import owner.
    pub fn client_import_observer(&self) -> VulkanClientImportObserver {
        self.dmabuf.client_import_observer()
    }

    /// Take (and drop) any wait semaphores staged via [`Renderer::wait`] that
    /// no submission has consumed yet, returning how many there were.
    ///
    /// Staged waits are drained by the next render submission — the first
    /// pass that samples the fenced buffers. They must not outlive the
    /// logical frame that staged them: a frame that ends without a
    /// submission (no damage, abort) leaves them for an unrelated later
    /// submission to inherit, and with them any fence that never signals.
    /// Callers enforce that boundary by draining here at the end of each
    /// frame-producing scope; a nonzero return means the frame aborted after
    /// staging, which is expected on abort paths and must stay confined.
    pub fn take_pending_acquire_waits(&mut self) -> usize {
        self.device.take_pending_wait_semaphores()
    }

    /// Preallocate descriptor pages for a conservative recording bound and
    /// reject pressure before command recording begins.
    pub fn reserve_texture_descriptors(&mut self, requested_sets: usize) -> Result<(), VulkanRendererError> {
        self.device.reclaim_completed_submissions()?;
        self.descriptors.reserve_texture_descriptors(requested_sets)
    }

    /// Capture the renderer's queue frontier immediately before a fallible
    /// scope whose submission behavior must later be classified exactly.
    pub fn submission_snapshot(&self) -> VulkanSubmissionSnapshot {
        self.device.submission_snapshot()
    }

    /// Return the completion edge of the newest submission after `snapshot`,
    /// or `None` when the scope provably submitted nothing.
    pub fn completion_since(
        &self,
        snapshot: VulkanSubmissionSnapshot,
    ) -> Option<crate::backend::renderer::sync::SyncPoint> {
        self.device.completion_since(snapshot)
    }

    /// Exports the current Vulkan pipeline cache blob for persistence by the caller.
    pub fn pipeline_cache_data(&self) -> Result<Vec<u8>, VulkanRendererError> {
        self.pipelines.pipeline_cache_data()
    }

    /// Merges caller-provided cache data into the current Vulkan pipeline cache.
    pub fn merge_pipeline_cache_data(&mut self, cache_data: &[u8]) -> Result<(), VulkanRendererError> {
        self.pipelines.merge_pipeline_cache_data(cache_data)
    }

    /// Returns a standardized "not implemented" error.
    pub fn not_yet_implemented(operation: &'static str) -> VulkanRendererError {
        VulkanRendererError::not_implemented(operation)
    }

    /// Configure one owner-sized staging chunk, or retire it with zero bytes.
    ///
    /// Call only on an upload-owner lifecycle turn, outside render frame work.
    /// Successful reservations in this mode never allocate or grow storage.
    /// `Ok(false)` means a queued upload, GPU submission, or detached writer
    /// still owns the current storage. Its completion/return must trigger a
    /// later lifecycle retry; this method never waits or cancels live work.
    /// Allocation failure preserves the previous mode and storage.
    /// All memory imports and updates, including slices, require this one
    /// chunk. A new renderer starts with zero upload storage; no upload path
    /// provisions storage or falls back to a separate arena. The owner must
    /// account for every live SHM, cursor, and raster source before frame work.
    pub fn configure_memory_upload_capacity(&mut self, capacity: usize) -> Result<bool, VulkanRendererError> {
        self.device.configure_memory_upload_capacity(capacity)
    }

    /// Provision the maximum of a structural extent and a real admitted
    /// generation. Larger-source exceptions are explicit in upload diagnostics
    /// and TRACE; the reserve path never changes the configured extent.
    pub fn configure_memory_upload_capacity_for_extent(
        &mut self,
        structural_bytes: usize,
        generation_bytes: usize,
    ) -> Result<bool, VulkanRendererError> {
        self.device
            .configure_memory_upload_capacity_for_extent(structural_bytes, generation_bytes)
    }

    /// Exact owner metadata/capacity comparison; performs no GPU work.
    pub fn memory_upload_storage_matches(&self, structural: usize, generation: usize) -> bool {
        self.device.memory_upload_storage_matches(structural, generation)
    }
    /// Adopt an exact helper result after prior reservations/readers retire.
    /// No mapping, native allocation, destruction or wait occurs in adoption.
    pub fn adopt_memory_upload_storage(
        &mut self,
        storage: &mut VulkanUploadStorage,
        structural: usize,
        generation: usize,
    ) -> Result<bool, VulkanRendererError> {
        self.device
            .adopt_memory_upload_storage(storage, structural, generation)
    }

    /// Tags allocations on this thread for this renderer's device until the
    /// returned guard is dropped. The guard holds no renderer borrow and must
    /// stay on this thread. This thread has 64 fixed scope slots. While any
    /// overflow guard is live, all allocations on this thread conservatively
    /// use `Unspecified`; existing device phases return after overflow retires.
    pub fn allocation_phase_scope(&self, phase: VulkanAllocationPhase) -> VulkanAllocationPhaseGuard {
        self.device.shared_device().allocation_ledger().enter_phase(phase)
    }

    /// Returns live diagnostics for cache behavior and command submission timing.
    pub fn diagnostics(&self) -> VulkanRendererDiagnostics {
        let submissions: DeviceDiagnostics = self.device.diagnostics();
        let arena = self.device.upload_arena_stats();
        let (
            owner_capacity_bytes,
            owner_structural_bytes,
            owner_extent_exception_bytes,
            owner_extent_exceptions_total,
        ) = self.device.owner_upload_extent_stats();
        let (pending_operations, pending_bytes) = self.device.pending_upload_stats();
        VulkanRendererDiagnostics {
            allocations: self.device.shared_device().allocation_ledger().snapshot(),
            dmabuf_cache: self.dmabuf.cache_stats(),
            descriptor_cache: self.descriptors.cache_stats(),
            descriptors: self.descriptors.arena_stats(),
            submissions: VulkanSubmissionStats {
                total_submissions: submissions.total_submissions,
                blocking_submissions: submissions.blocking_submissions,
                reclaimed_submissions: submissions.reclaimed_submissions,
                avg_submit_cpu_ns: avg_nanos(submissions.total_submit_cpu_ns, submissions.total_submissions),
                max_submit_cpu_ns: submissions.max_submit_cpu_ns,
                avg_completion_ns: avg_nanos(
                    submissions.total_completion_ns,
                    submissions.reclaimed_submissions,
                ),
                max_completion_ns: submissions.max_completion_ns,
            },
            uploads: VulkanUploadStats {
                owner_capacity_bytes,
                owner_structural_bytes,
                owner_extent_exception_bytes,
                owner_extent_exceptions_total,
                arena_capacity_bytes: arena.capacity_bytes,
                arena_in_use_bytes: arena.in_use_bytes,
                arena_high_water_bytes: arena.high_water_bytes,
                arena_chunk_count: arena.chunk_count,
                arena_growth_count: arena.growth_count,
                arena_deferred_count: arena.deferred_count,
                pending_operations,
                pending_bytes,
                submitted_batches: submissions.upload_batches,
                submitted_operations: submissions.upload_operations,
                submitted_bytes: submissions.upload_bytes,
            },
            debug_markers_enabled: submissions.debug_markers_enabled,
        }
    }
}

fn avg_nanos(total_ns: u64, count: u64) -> u64 {
    if count == 0 {
        return 0;
    }
    total_ns / count
}

#[cfg(test)]
mod test_support;

#[cfg(test)]
mod tests {
    use super::VulkanRenderer;

    #[test]
    fn renderer_create_drop_loop() {
        let Some(physical_device) = crate::backend::renderer::vulkan::test_support::physical_device() else {
            return;
        };
        let Some(renderer) = crate::backend::renderer::vulkan::test_support::renderer(&physical_device)
        else {
            return;
        };
        drop(renderer);

        for _ in 0..32 {
            let renderer = VulkanRenderer::new(&physical_device)
                .expect("renderer initialization should remain stable across repeated create/drop");
            drop(renderer);
        }
    }
}
