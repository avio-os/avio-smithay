use crate::{
    backend::{allocator::Fourcc, renderer::Texture},
    utils::{Buffer as BufferCoord, Size},
};

use super::dmabuf::ImportedDmabufImage;
use std::sync::Arc;

/// Placeholder Vulkan render target handle for phase-0 scaffolding.
#[derive(Debug, Clone)]
pub struct VulkanTarget {
    size: Size<i32, BufferCoord>,
    format: Option<Fourcc>,
    imported: Option<Arc<ImportedDmabufImage>>,
    release_to_foreign_on_finish: bool,
}

impl VulkanTarget {
    /// Creates a placeholder render-target description.
    pub fn new(size: Size<i32, BufferCoord>, format: Option<Fourcc>) -> Self {
        Self {
            size,
            format,
            imported: None,
            release_to_foreign_on_finish: false,
        }
    }

    pub(crate) fn from_dmabuf_import(
        imported: Arc<ImportedDmabufImage>,
        size: Size<i32, BufferCoord>,
        format: Option<Fourcc>,
    ) -> Self {
        Self {
            size,
            format,
            imported: Some(imported),
            release_to_foreign_on_finish: true,
        }
    }

    pub(crate) fn from_imported_image(
        imported: Arc<ImportedDmabufImage>,
        size: Size<i32, BufferCoord>,
        format: Option<Fourcc>,
    ) -> Self {
        Self {
            size,
            format,
            imported: Some(imported),
            release_to_foreign_on_finish: false,
        }
    }

    pub(crate) fn imported_image_id(&self) -> Option<u64> {
        self.imported.as_ref().map(|image| image.id())
    }

    pub(crate) fn imported_image(&self) -> Option<&Arc<ImportedDmabufImage>> {
        self.imported.as_ref()
    }

    pub(crate) fn release_to_foreign_on_finish(&self) -> bool {
        self.release_to_foreign_on_finish
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
