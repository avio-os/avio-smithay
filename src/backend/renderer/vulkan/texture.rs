use crate::{
    backend::{allocator::Fourcc, renderer::Texture},
    utils::{Buffer as BufferCoord, Size},
};

use super::image::VulkanImage;
use std::sync::Arc;

/// Placeholder Vulkan texture handle for phase-0 scaffolding.
#[derive(Debug, Clone)]
pub struct VulkanTexture {
    size: Size<i32, BufferCoord>,
    format: Option<Fourcc>,
    y_inverted: bool,
    image: Option<Arc<VulkanImage>>,
    memory_writable: bool,
}

impl VulkanTexture {
    /// Creates a placeholder texture description.
    pub fn new(size: Size<i32, BufferCoord>, format: Option<Fourcc>) -> Self {
        Self {
            size,
            format,
            y_inverted: false,
            image: None,
            memory_writable: false,
        }
    }

    pub(crate) fn from_dmabuf_import(
        image: Arc<VulkanImage>,
        size: Size<i32, BufferCoord>,
        format: Option<Fourcc>,
        y_inverted: bool,
    ) -> Self {
        debug_assert!(image.uses_foreign_queue());
        Self {
            size,
            format,
            y_inverted,
            image: Some(image),
            memory_writable: false,
        }
    }

    pub(crate) fn from_renderer_image(
        image: Arc<VulkanImage>,
        size: Size<i32, BufferCoord>,
        format: Fourcc,
        y_inverted: bool,
        memory_writable: bool,
    ) -> Self {
        debug_assert!(image.is_renderer_local());
        Self {
            size,
            format: Some(format),
            y_inverted,
            image: Some(image),
            memory_writable,
        }
    }

    pub(super) fn from_framebuffer_image(image: Arc<VulkanImage>) -> Self {
        Self {
            size: image.size(),
            format: Some(image.format().code),
            y_inverted: false,
            image: Some(image),
            memory_writable: false,
        }
    }

    /// Returns if this texture originates from y-inverted dma-buf contents.
    pub fn y_inverted(&self) -> bool {
        self.y_inverted
    }

    pub(crate) fn image_resource_id(&self) -> Option<u64> {
        self.image.as_ref().map(|image| image.id())
    }

    pub(crate) fn image_resource(&self) -> Option<&Arc<VulkanImage>> {
        self.image.as_ref()
    }

    /// Returns true when no other owner can currently submit work touching this
    /// imported image. Compositors use this before reusing or evicting retained
    /// offscreen targets.
    pub fn is_externally_idle(&self) -> bool {
        self.image
            .as_ref()
            .map(|image| Arc::strong_count(image) <= 1)
            .unwrap_or(true)
    }

    pub(crate) fn memory_writable(&self) -> bool {
        self.memory_writable
    }
}

impl Texture for VulkanTexture {
    fn width(&self) -> u32 {
        self.size.w as u32
    }

    fn height(&self) -> u32 {
        self.size.h as u32
    }

    fn format(&self) -> Option<Fourcc> {
        self.format
    }

    fn size(&self) -> Size<i32, BufferCoord> {
        self.size
    }
}
