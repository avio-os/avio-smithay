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
//! - [`crate::backend::renderer::ImportMem`] uses host-visible staging uploads and creates writable
//!   Vulkan textures for shm/memory-backed client paths.
//! - [`crate::backend::renderer::ExportMem`] performs readback through transfer buffers and returns
//!   deterministic linear pixel data for supported formats.
//!
//! # Format And Modifier Expectations
//!
//! - Explicit modifier support is queried from Vulkan (`VK_EXT_image_drm_format_modifier`) and cached.
//! - Implicit modifier support (`Modifier::Invalid`) is advertised only when Vulkan reports support
//!   without explicit modifier metadata.
//! - Explicit modifier imports validate plane count, plane strides, plane offsets, and usage before
//!   creating Vulkan images.
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

mod blit;
mod descriptor;
mod device;
mod dmabuf;
mod error;
mod format;
mod frame;
mod pipeline;
mod readback;
mod sync;
mod target;
mod texture;
mod upload;

pub use error::{VulkanRendererError, VulkanRendererErrorKind};
pub use target::VulkanTarget;
pub use texture::VulkanTexture;

use std::ffi::CStr;

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

/// Aggregated runtime diagnostics for the Vulkan renderer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VulkanRendererDiagnostics {
    /// dma-buf import/bind cache diagnostics.
    pub dmabuf_cache: VulkanCacheStats,
    /// Texture descriptor cache diagnostics.
    pub descriptor_cache: VulkanCacheStats,
    /// Submission and completion timing diagnostics.
    pub submissions: VulkanSubmissionStats,
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
    descriptors: DescriptorState,
    pipelines: PipelineState,
    vk_render_probe_count: u64,
}

impl VulkanRenderer {
    /// Returns the required device extensions for this renderer.
    pub fn required_extensions(physical_device: &PhysicalDevice) -> Vec<&'static CStr> {
        DeviceState::required_extensions(physical_device)
    }

    /// Creates a new Vulkan renderer and initializes device/queue infrastructure.
    pub fn new(physical_device: &PhysicalDevice) -> Result<Self, VulkanRendererError> {
        let device = DeviceState::new(physical_device)?;
        let descriptors = DescriptorState::new(device.shared_device())?;
        let pipelines = PipelineState::new(device.shared_device(), descriptors.texture_layout())?;

        Ok(Self {
            context_id: ContextId::new(),
            downscale_filter: TextureFilter::Linear,
            upscale_filter: TextureFilter::Linear,
            debug_flags: DebugFlags::empty(),
            device,
            formats: FormatCapabilities::new(physical_device)?,
            dmabuf: DmabufState::default(),
            upload: UploadState::default(),
            readback: ReadbackState::default(),
            blit: BlitState::default(),
            descriptors,
            pipelines,
            vk_render_probe_count: 0,
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

    /// Returns whether a format+modifier is supported for dma-buf texture import.
    pub fn has_dmabuf_import_format(&self, format: Format) -> bool {
        self.formats.has_import_format(format)
    }

    /// Returns whether a format+modifier is supported for dma-buf render-target binding.
    pub fn has_dmabuf_render_format(&self, format: Format) -> bool {
        self.formats.has_render_format(format)
    }

    /// Supported import modifiers for the given DRM format code.
    pub fn import_modifiers(&self, code: Fourcc) -> &[Modifier] {
        self.formats.import_modifiers(code)
    }

    /// Supported render-target modifiers for the given DRM format code.
    pub fn render_modifiers(&self, code: Fourcc) -> &[Modifier] {
        self.formats.render_modifiers(code)
    }

    /// Intersects renderer import capabilities with a caller-provided modifier preference list.
    pub fn intersect_import_modifiers(&self, code: Fourcc, requested: &[Modifier]) -> Vec<Modifier> {
        self.formats.intersect_import_modifiers(code, requested)
    }

    /// Intersects renderer render-target capabilities with a caller-provided modifier preference list.
    pub fn intersect_render_modifiers(&self, code: Fourcc, requested: &[Modifier]) -> Vec<Modifier> {
        self.formats.intersect_render_modifiers(code, requested)
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

    /// Drop stale cached dma-buf imports.
    pub fn cleanup_dmabuf_cache(&mut self) {
        self.dmabuf.cleanup();
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

    /// Returns live diagnostics for cache behavior and command submission timing.
    pub fn diagnostics(&self) -> VulkanRendererDiagnostics {
        let submissions: DeviceDiagnostics = self.device.diagnostics();
        VulkanRendererDiagnostics {
            dmabuf_cache: self.dmabuf.cache_stats(),
            descriptor_cache: self.descriptors.cache_stats(),
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
mod tests {
    use crate::backend::vulkan::{version::Version, Instance, PhysicalDevice};

    use super::{VulkanRenderer, VulkanRendererError};

    #[test]
    fn renderer_create_drop_loop() {
        let instance = match Instance::new(Version::VERSION_1_3, None) {
            Ok(instance) => instance,
            Err(_) => return,
        };

        let physical_device = match PhysicalDevice::enumerate(&instance) {
            Ok(mut iter) => match iter.next() {
                Some(phd) => phd,
                None => return,
            },
            Err(_) => return,
        };

        match VulkanRenderer::new(&physical_device) {
            Ok(renderer) => drop(renderer),
            Err(
                VulkanRendererError::MissingDeviceExtensions(_)
                | VulkanRendererError::MissingDeviceFeature(_)
                | VulkanRendererError::MissingQueueFamily { .. },
            ) => {
                return;
            }
            Err(err) => panic!("unexpected initialization failure for Vulkan renderer: {err}"),
        }

        for _ in 0..32 {
            let renderer = VulkanRenderer::new(&physical_device)
                .expect("renderer initialization should remain stable across repeated create/drop");
            drop(renderer);
        }
    }
}
