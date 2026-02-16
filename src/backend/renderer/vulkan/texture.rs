use crate::{
    backend::{allocator::Fourcc, renderer::Texture},
    utils::{Buffer as BufferCoord, Size},
};

use super::dmabuf::ImportedDmabufImage;
use std::sync::Arc;

/// Placeholder Vulkan texture handle for phase-0 scaffolding.
#[derive(Debug, Clone)]
pub struct VulkanTexture {
    size: Size<i32, BufferCoord>,
    format: Option<Fourcc>,
    y_inverted: bool,
    imported: Option<Arc<ImportedDmabufImage>>,
}

impl VulkanTexture {
    /// Creates a placeholder texture description.
    pub fn new(size: Size<i32, BufferCoord>, format: Option<Fourcc>) -> Self {
        Self {
            size,
            format,
            y_inverted: false,
            imported: None,
        }
    }

    pub(crate) fn from_dmabuf_import(
        imported: Arc<ImportedDmabufImage>,
        size: Size<i32, BufferCoord>,
        format: Option<Fourcc>,
        y_inverted: bool,
    ) -> Self {
        Self {
            size,
            format,
            y_inverted,
            imported: Some(imported),
        }
    }

    /// Returns if this texture originates from y-inverted dma-buf contents.
    pub fn y_inverted(&self) -> bool {
        self.y_inverted
    }

    pub(crate) fn imported_image_id(&self) -> Option<u64> {
        self.imported.as_ref().map(|image| image.id())
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
