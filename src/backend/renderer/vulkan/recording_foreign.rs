//! Foreign layout barriers use the same exclusive cold recording workspace.
use super::{
    image::{stage_access_for_layout, VulkanImage},
    recording_storage::RecordingStorage,
};
use ash::vk;
use std::sync::Arc;
impl RecordingStorage {
    pub(super) fn acquire_foreign_images(
        &mut self,
        device: &ash::Device,
        command: vk::CommandBuffer,
        family: u32,
        images: impl IntoIterator<Item = (Arc<VulkanImage>, vk::ImageLayout)>,
    ) {
        for (image, layout) in images {
            if self.pending_layouts.contains_key(&image.id()) || !image.take_foreign_ownership() {
                continue;
            }
            // The complete graph's conservative entry count was admitted
            // before recording or queue-family ownership was changed.
            debug_assert!(self.pending_layouts.len() < self.image_limit);
            self.pending_layouts.insert(image.id(), (image, layout));
        }
        self.barriers.clear();
        let mut stages = vk::PipelineStageFlags::empty();
        for (image, layout) in self.pending_layouts.values() {
            let (stage, access) = stage_access_for_layout(*layout);
            stages |= stage;
            self.barriers.push(
                vk::ImageMemoryBarrier::default()
                    .old_layout(*layout)
                    .new_layout(*layout)
                    .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                    .dst_queue_family_index(family)
                    .image(image.image())
                    .subresource_range(range())
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(access),
            );
        }
        if !self.barriers.is_empty() {
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    if stages.is_empty() {
                        vk::PipelineStageFlags::ALL_COMMANDS
                    } else {
                        stages
                    },
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &self.barriers,
                );
            }
        }
    }
    pub(super) fn release_foreign_images(
        &mut self,
        device: &ash::Device,
        command: vk::CommandBuffer,
        family: u32,
    ) {
        self.barriers.clear();
        let mut stages = vk::PipelineStageFlags::empty();
        for (image, layout) in self.pending_layouts.values() {
            let (stage, access) = stage_access_for_layout(*layout);
            stages |= stage;
            self.barriers.push(
                vk::ImageMemoryBarrier::default()
                    .old_layout(*layout)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(family)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                    .image(image.image())
                    .subresource_range(range())
                    .src_access_mask(access)
                    .dst_access_mask(vk::AccessFlags::empty()),
            );
        }
        if !self.barriers.is_empty() {
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    if stages.is_empty() {
                        vk::PipelineStageFlags::ALL_COMMANDS
                    } else {
                        stages
                    },
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &self.barriers,
                );
            }
        }
    }
}
fn range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}
