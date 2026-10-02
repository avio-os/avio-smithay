//! Opaque cold-prepared resources and allocation-free adoption outcomes.
pub use super::dmabuf::preparation::PreparedSourceImport;
pub use super::pipeline::preparation::PreparedPipelineBank;

/// The actual native attachment semantics, rather than a source colour format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedAttachmentFormat {
    /// Linear blending in the negotiated framebuffer format.
    Framebuffer(crate::backend::allocator::Fourcc),
    /// Preserve the negotiated encoded storage, used by explicit captures.
    PreserveStorage(crate::backend::allocator::Fourcc),
    /// The existing F16 material working target.
    MaterialWorking,
}
impl PreparedAttachmentFormat {
    pub(super) fn native(self) -> Result<ash::vk::Format, super::VulkanRendererError> {
        match self {
            Self::MaterialWorking => Ok(ash::vk::Format::R16G16B16A16_SFLOAT),
            Self::Framebuffer(fourcc) | Self::PreserveStorage(fourcc) => {
                let storage = crate::backend::allocator::vulkan::format::get_vk_format(fourcc)
                    .ok_or(super::VulkanRendererError::UnsupportedMemoryFormat(fourcc))?;
                Ok(if matches!(self, Self::Framebuffer(_)) {
                    super::format::render_view_format(storage)
                } else {
                    storage
                })
            }
        }
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
