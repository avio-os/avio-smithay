//! Vulkan renderer scaffolding.
//!
//! This module currently includes phase-7 device, dma-buf import/bind, descriptor,
//! pipeline, frame-recording infrastructure, explicit sync bridge support, and
//! memory upload support for shared-memory client paths.
//! Readback paths are implemented in later phases.

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

use self::descriptor::DescriptorState;
use self::device::DeviceState;
use self::dmabuf::DmabufState;
use self::format::FormatCapabilities;
use self::pipeline::PipelineState;
use self::upload::UploadState;

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
    descriptors: DescriptorState,
    pipelines: PipelineState,
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
            descriptors,
            pipelines,
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

    /// Returns whether importing native sync-file fds into Vulkan wait paths is supported.
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
