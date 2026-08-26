use crate::{
    backend::{allocator::Fourcc, renderer::Texture},
    utils::{Buffer as BufferCoord, Size},
};

use super::{format::render_view_format, image::VulkanImage};
use std::sync::Arc;

/// Whether a pass composites in linear light or preserves sampled storage.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) enum VulkanTargetEncoding {
    #[default]
    LinearBlend,
    PreserveStorage,
}

impl VulkanTargetEncoding {
    pub(crate) fn format(self, storage: ash::vk::Format) -> ash::vk::Format {
        match self {
            Self::LinearBlend => render_view_format(storage),
            Self::PreserveStorage => storage,
        }
    }

    pub(crate) fn view(self, image: &VulkanImage) -> ash::vk::ImageView {
        match self {
            Self::LinearBlend => image.render_view(),
            Self::PreserveStorage => image.view(),
        }
    }

    pub(crate) fn blends_in_linear_light(self, image: &VulkanImage) -> bool {
        self.format(image.vk_format()) != image.vk_format()
    }
}

/// Vulkan render-target handle with explicit attachment encoding.
#[derive(Debug, Clone)]
pub struct VulkanTarget {
    size: Size<i32, BufferCoord>,
    format: Option<Fourcc>,
    image: Option<Arc<VulkanImage>>,
    pub(crate) encoding: VulkanTargetEncoding,
}

impl VulkanTarget {
    /// Creates a placeholder render-target description.
    pub fn new(size: Size<i32, BufferCoord>, format: Option<Fourcc>) -> Self {
        Self {
            size,
            format,
            image: None,
            encoding: VulkanTargetEncoding::LinearBlend,
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
            encoding: VulkanTargetEncoding::LinearBlend,
        }
    }

    pub(crate) fn image_resource_id(&self) -> Option<u64> {
        self.image.as_ref().map(|image| image.id())
    }

    pub(crate) fn image_resource(&self) -> Option<&Arc<VulkanImage>> {
        self.image.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::VulkanTargetEncoding;
    use ash::vk;

    #[test]
    fn storage_copy_does_not_linearize_or_reencode_translucent_pixels() {
        for (storage, composited) in [
            (vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB),
            (vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_SRGB),
            (
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::Format::A2B10G10R10_UNORM_PACK32,
            ),
        ] {
            assert_eq!(VulkanTargetEncoding::PreserveStorage.format(storage), storage);
            assert_eq!(VulkanTargetEncoding::LinearBlend.format(storage), composited);
        }
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
