use crate::{
    backend::{allocator::Fourcc, renderer::Texture},
    utils::{Buffer as BufferCoord, Size},
};

/// Placeholder Vulkan texture handle for phase-0 scaffolding.
#[derive(Debug, Clone)]
pub struct VulkanTexture {
    size: Size<i32, BufferCoord>,
    format: Option<Fourcc>,
}

impl VulkanTexture {
    /// Creates a placeholder texture description.
    pub fn new(size: Size<i32, BufferCoord>, format: Option<Fourcc>) -> Self {
        Self { size, format }
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
