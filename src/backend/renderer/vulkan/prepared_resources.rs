//! Opaque cold-prepared resources and allocation-free adoption outcomes.
pub use super::dmabuf::preparation::PreparedSourceImport;
pub use super::pipeline::preparation::PreparedPipelineBank;

/// The actual native attachment semantics, rather than a source colour format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedAttachmentFormat {
    /// Attachment of the negotiated framebuffer format, blending in
    /// [`VulkanRenderer::BLEND_SPACE`](super::VulkanRenderer::BLEND_SPACE).
    /// Display, capture and material passes on that format share this bank.
    Framebuffer(crate::backend::allocator::Fourcc),
    /// The existing F16 material working target.
    MaterialWorking,
}
impl PreparedAttachmentFormat {
    pub(super) fn native(self) -> Result<ash::vk::Format, super::VulkanRendererError> {
        let storage = match self {
            Self::MaterialWorking => ash::vk::Format::R16G16B16A16_SFLOAT,
            Self::Framebuffer(fourcc) => crate::backend::allocator::vulkan::format::get_vk_format(fourcc)
                .ok_or(super::VulkanRendererError::UnsupportedMemoryFormat(fourcc))?,
        };
        Ok(super::VulkanRenderer::BLEND_SPACE.attachment_format(storage))
    }
}

/// No outcome authorizes drawing an unprepared resource or retiring readers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedResourceAdoption {
    /// Exact native resource and its membership pin were installed.
    Adopted,
    /// The existing owner import was retained and its pin was admitted.
    AlreadyPrepared,
    /// A source registry is busy; the exact prepared object is unchanged.
    Busy,
    /// Owner cache capacity is exhausted; no cache entry was evicted or grown.
    CapacityDeferred,
}

#[cfg(test)]
mod tests {
    use super::PreparedAttachmentFormat;
    use crate::backend::allocator::vulkan::format::{get_vk_format, known_formats};

    /// The blend space is a renderer parameter, not a property of the format:
    /// every storage format, 8-bit and 2101010 alike, attaches through itself.
    #[test]
    fn every_framebuffer_format_attaches_through_its_own_storage_format() {
        for fourcc in known_formats() {
            assert_eq!(
                PreparedAttachmentFormat::Framebuffer(*fourcc).native().ok(),
                get_vk_format(*fourcc),
                "{fourcc:?}"
            );
        }
        assert_eq!(
            PreparedAttachmentFormat::MaterialWorking.native().ok(),
            Some(ash::vk::Format::R16G16B16A16_SFLOAT)
        );
    }
}
