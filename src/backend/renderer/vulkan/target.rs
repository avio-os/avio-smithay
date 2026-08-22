use crate::{
    backend::{allocator::Fourcc, renderer::Texture},
    utils::{Buffer as BufferCoord, Size},
};

use super::image::VulkanImage;
use std::sync::Arc;

/// Placeholder Vulkan render target handle for phase-0 scaffolding.
#[derive(Debug, Clone)]
pub struct VulkanTarget {
    size: Size<i32, BufferCoord>,
    format: Option<Fourcc>,
    image: Option<Arc<VulkanImage>>,
}

impl VulkanTarget {
    /// Creates a placeholder render-target description.
    pub fn new(size: Size<i32, BufferCoord>, format: Option<Fourcc>) -> Self {
        Self {
            size,
            format,
            image: None,
        }
    }

    pub(crate) fn from_image_resource(
        image: Arc<VulkanImage>,
        size: Size<i32, BufferCoord>,
        format: Option<Fourcc>,
    ) -> Self {
        Self {
            size,
            format,
            image: Some(image),
        }
    }

    pub(crate) fn image_resource_id(&self) -> Option<u64> {
        self.image.as_ref().map(|image| image.id())
    }

    pub(crate) fn image_resource(&self) -> Option<&Arc<VulkanImage>> {
        self.image.as_ref()
    }
}

impl Texture for VulkanTarget {
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
