use std::sync::{
    atomic::{AtomicBool, AtomicI32, Ordering},
    Arc,
};

use ash::vk;
use indexmap::IndexMap;

use crate::{
    backend::allocator::Format,
    utils::{Buffer as BufferCoord, Size},
};

use super::{
    device::DeviceHandle,
    format::{render_view_format, ColorEncoding},
};

/// Immutable origin of a Vulkan image resource.
///
/// Queue-family ownership is meaningful only for memory imported from an
/// external producer. Renderer-local images stay on this device's queue for
/// their complete lifetime and must never be transferred to
/// `VK_QUEUE_FAMILY_FOREIGN_EXT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VulkanImageOrigin {
    RendererLocal,
    ExternalDmabuf,
}

impl VulkanImageOrigin {
    fn uses_foreign_queue(self) -> bool {
        matches!(self, Self::ExternalDmabuf)
    }
}

/// One Vulkan image together with the immutable facts that govern every use.
///
/// Storage provenance, color encoding, layout, and external queue custody live
/// on the resource itself. Callers select an operation; they do not guess how
/// the image was allocated or whether a FOREIGN transfer is legal.
pub(crate) struct VulkanImage {
    resource_id: u64,
    image: vk::Image,
    memories: Vec<vk::DeviceMemory>,
    sampled_view: vk::ImageView,
    render_view: vk::ImageView,
    size: Size<i32, BufferCoord>,
    format: Format,
    vk_format: vk::Format,
    color_encoding: ColorEncoding,
    usage: vk::ImageUsageFlags,
    y_inverted: bool,
    origin: VulkanImageOrigin,
    layout: AtomicI32,
    owned_by_foreign: AtomicBool,
    device: Arc<DeviceHandle>,
}

impl std::fmt::Debug for VulkanImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanImage")
            .field("resource_id", &self.resource_id)
            .field("image", &self.image)
            .field("memories", &self.memories)
            .field("sampled_view", &self.sampled_view)
            .field("render_view", &self.render_view)
            .field("size", &self.size)
            .field("format", &self.format)
            .field("vk_format", &self.vk_format)
            .field("usage", &self.usage)
            .field("y_inverted", &self.y_inverted)
            .field("origin", &self.origin)
            .finish()
    }
}

impl VulkanImage {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new_renderer_local(
        resource_id: u64,
        image: vk::Image,
        memory: vk::DeviceMemory,
        sampled_view: vk::ImageView,
        render_view: vk::ImageView,
        size: Size<i32, BufferCoord>,
        format: Format,
        vk_format: vk::Format,
        color_encoding: ColorEncoding,
        usage: vk::ImageUsageFlags,
        y_inverted: bool,
        initial_layout: vk::ImageLayout,
        device: Arc<DeviceHandle>,
    ) -> Self {
        Self::new(
            resource_id,
            image,
            vec![memory],
            sampled_view,
            render_view,
            size,
            format,
            vk_format,
            color_encoding,
            usage,
            y_inverted,
            VulkanImageOrigin::RendererLocal,
            initial_layout,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn new_external_dmabuf(
        resource_id: u64,
        image: vk::Image,
        memories: Vec<vk::DeviceMemory>,
        sampled_view: vk::ImageView,
        render_view: vk::ImageView,
        size: Size<i32, BufferCoord>,
        format: Format,
        vk_format: vk::Format,
        color_encoding: ColorEncoding,
        usage: vk::ImageUsageFlags,
        y_inverted: bool,
        initial_layout: vk::ImageLayout,
        device: Arc<DeviceHandle>,
    ) -> Self {
        Self::new(
            resource_id,
            image,
            memories,
            sampled_view,
            render_view,
            size,
            format,
            vk_format,
            color_encoding,
            usage,
            y_inverted,
            VulkanImageOrigin::ExternalDmabuf,
            initial_layout,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        resource_id: u64,
        image: vk::Image,
        memories: Vec<vk::DeviceMemory>,
        sampled_view: vk::ImageView,
        render_view: vk::ImageView,
        size: Size<i32, BufferCoord>,
        format: Format,
        vk_format: vk::Format,
        color_encoding: ColorEncoding,
        usage: vk::ImageUsageFlags,
        y_inverted: bool,
        origin: VulkanImageOrigin,
        initial_layout: vk::ImageLayout,
        device: Arc<DeviceHandle>,
    ) -> Self {
        Self {
            resource_id,
            image,
            memories,
            sampled_view,
            render_view,
            size,
            format,
            vk_format,
            color_encoding,
            usage,
            y_inverted,
            origin,
            layout: AtomicI32::new(initial_layout.as_raw()),
            owned_by_foreign: AtomicBool::new(origin.uses_foreign_queue()),
            device,
        }
    }

    pub(crate) fn id(&self) -> u64 {
        self.resource_id
    }

    pub(crate) fn image(&self) -> vk::Image {
        self.image
    }

    pub(crate) fn view(&self) -> vk::ImageView {
        self.sampled_view
    }

    /// Format of the colour attachment view — the `_SRGB` sibling of the
    /// storage format wherever one exists.
    pub(crate) fn render_format(&self) -> vk::Format {
        render_view_format(self.vk_format)
    }

    pub(crate) fn color_encoding(&self) -> ColorEncoding {
        self.color_encoding
    }

    pub(crate) fn blends_in_linear_light(&self) -> bool {
        self.render_format() != self.vk_format
    }

    pub(crate) fn render_view(&self) -> vk::ImageView {
        self.render_view
    }

    pub(crate) fn vk_format(&self) -> vk::Format {
        self.vk_format
    }

    pub(crate) fn format(&self) -> Format {
        self.format
    }

    pub(crate) fn size(&self) -> Size<i32, BufferCoord> {
        self.size
    }

    pub(crate) fn usage(&self) -> vk::ImageUsageFlags {
        self.usage
    }

    pub(crate) fn y_inverted(&self) -> bool {
        self.y_inverted
    }

    pub(crate) fn is_renderer_local(&self) -> bool {
        self.origin == VulkanImageOrigin::RendererLocal
    }

    pub(crate) fn uses_foreign_queue(&self) -> bool {
        self.origin.uses_foreign_queue()
    }

    pub(crate) fn current_layout(&self) -> vk::ImageLayout {
        vk::ImageLayout::from_raw(self.layout.load(Ordering::Relaxed))
    }

    pub(crate) fn set_layout(&self, layout: vk::ImageLayout) {
        self.layout.store(layout.as_raw(), Ordering::Relaxed);
    }

    /// Claims an externally backed image for this renderer queue.
    /// Renderer-local images return false without mutating state.
    pub(crate) fn take_foreign_ownership(&self) -> bool {
        self.uses_foreign_queue() && self.owned_by_foreign.swap(false, Ordering::AcqRel)
    }

    pub(crate) fn set_foreign_ownership(&self) {
        debug_assert!(
            self.uses_foreign_queue(),
            "renderer-local images must never be released to FOREIGN"
        );
        if self.uses_foreign_queue() {
            self.owned_by_foreign.store(true, Ordering::Release);
        }
    }

    #[cfg(test)]
    pub(crate) fn is_owned_by_foreign(&self) -> bool {
        self.owned_by_foreign.load(Ordering::Acquire)
    }
}

impl Drop for VulkanImage {
    fn drop(&mut self) {
        self.device.note_view_retired(self.sampled_view);
        self.device.destroy_with(|device| unsafe {
            device.destroy_image_view(self.sampled_view, None);
            device.destroy_image_view(self.render_view, None);
            device.destroy_image(self.image, None);
            for memory in &self.memories {
                device.free_memory(*memory, None);
            }
        });
    }
}

pub(crate) type ForeignImageAccesses = IndexMap<u64, (Arc<VulkanImage>, vk::ImageLayout)>;

pub(crate) fn acquire_images_from_foreign(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    queue_family_index: u32,
    images: impl IntoIterator<Item = (Arc<VulkanImage>, vk::ImageLayout)>,
) -> ForeignImageAccesses {
    let mut acquired = ForeignImageAccesses::new();
    for (image, layout) in images {
        if acquired.contains_key(&image.id()) || !image.take_foreign_ownership() {
            continue;
        }
        acquired.insert(image.id(), (image, layout));
    }
    if acquired.is_empty() {
        return acquired;
    }

    let mut dst_stage_mask = vk::PipelineStageFlags::empty();
    let barriers = acquired
        .values()
        .map(|(image, layout)| {
            let (dst_stage, dst_access) = stage_access_for_layout(*layout);
            dst_stage_mask |= dst_stage;
            vk::ImageMemoryBarrier::default()
                .old_layout(*layout)
                .new_layout(*layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .dst_queue_family_index(queue_family_index)
                .image(image.image())
                .subresource_range(color_subresource_range())
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(dst_access)
        })
        .collect::<Vec<_>>();

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            if dst_stage_mask.is_empty() {
                vk::PipelineStageFlags::ALL_COMMANDS
            } else {
                dst_stage_mask
            },
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barriers,
        );
    }
    acquired
}

pub(crate) fn release_images_to_foreign(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    queue_family_index: u32,
    images: &ForeignImageAccesses,
) {
    if images.is_empty() {
        return;
    }

    let mut src_stage_mask = vk::PipelineStageFlags::empty();
    let barriers = images
        .values()
        .map(|(image, layout)| {
            let (src_stage, src_access) = stage_access_for_layout(*layout);
            src_stage_mask |= src_stage;
            vk::ImageMemoryBarrier::default()
                .old_layout(*layout)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(queue_family_index)
                .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .image(image.image())
                .subresource_range(color_subresource_range())
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::empty())
        })
        .collect::<Vec<_>>();

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            if src_stage_mask.is_empty() {
                vk::PipelineStageFlags::ALL_COMMANDS
            } else {
                src_stage_mask
            },
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barriers,
        );
    }
}

pub(crate) fn restore_unsubmitted_foreign_acquires(images: &ForeignImageAccesses) {
    for (image, _) in images.values() {
        image.set_foreign_ownership();
    }
}

pub(crate) fn commit_foreign_releases(images: &ForeignImageAccesses) {
    for (image, _) in images.values() {
        image.set_layout(vk::ImageLayout::GENERAL);
        image.set_foreign_ownership();
    }
}

pub(crate) fn transition_image_layout(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let (src_stage, src_access) = stage_access_for_layout(old_layout);
    let (dst_stage, dst_access) = stage_access_for_layout(new_layout);
    let barrier = [vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(color_subresource_range())
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)];

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }
}

pub(crate) fn stage_access_for_layout(layout: vk::ImageLayout) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    match layout {
        vk::ImageLayout::UNDEFINED => (vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty()),
        vk::ImageLayout::GENERAL => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_READ,
        ),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => {
            (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ)
        }
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => {
            (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE)
        }
        _ => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ),
    }
}

fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

#[cfg(test)]
mod tests {
    use super::VulkanImageOrigin;

    #[test]
    fn only_external_dmabufs_participate_in_foreign_queue_custody() {
        assert!(!VulkanImageOrigin::RendererLocal.uses_foreign_queue());
        assert!(VulkanImageOrigin::ExternalDmabuf.uses_foreign_queue());
    }
}
