use std::sync::Arc;

use ash::vk;
use indexmap::IndexMap;
use tracing::{instrument, trace};

use crate::{
    backend::renderer::{sync::SyncPoint, Blit, TextureFilter},
    utils::{Buffer as BufferCoord, Physical, Rectangle, Size},
};

use super::{
    device::DeviceState,
    image::{
        acquire_images_from_foreign, commit_foreign_releases, release_images_to_foreign,
        restore_unsubmitted_foreign_acquires, transition_image_layout, VulkanImage,
    },
    VulkanRenderer, VulkanRendererError, VulkanTarget, VulkanTexture,
};

/// One image blit operation in a Vulkan batch.
#[derive(Debug, Clone)]
pub struct VulkanBlitChainStep {
    source: VulkanTexture,
    destination: VulkanTexture,
    source_rect: Rectangle<i32, Physical>,
    destination_rect: Rectangle<i32, Physical>,
    filter: TextureFilter,
}

impl VulkanBlitChainStep {
    /// Creates a blit step using cloned texture handles.
    pub fn new(
        source: &VulkanTexture,
        destination: &VulkanTexture,
        source_rect: Rectangle<i32, Physical>,
        destination_rect: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Self {
        Self {
            source: source.clone(),
            destination: destination.clone(),
            source_rect,
            destination_rect,
            filter,
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct BlitState {
    format_features: IndexMap<vk::Format, vk::FormatFeatureFlags>,
}

#[derive(Debug)]
struct ResolvedBlitChainStep {
    source: Arc<VulkanImage>,
    destination: Arc<VulkanImage>,
    source_rect: Rectangle<i32, Physical>,
    destination_rect: Rectangle<i32, Physical>,
    filter: TextureFilter,
}

#[derive(Debug)]
pub(crate) struct TrackedBlitImageLayout {
    pub(crate) image: Arc<VulkanImage>,
    pub(crate) current_layout: vk::ImageLayout,
    pub(crate) restore_layout: vk::ImageLayout,
}

impl BlitState {
    #[instrument(level = "trace", skip(self, device, from, to))]
    #[profiling::function]
    fn blit_images(
        &mut self,
        device: &mut DeviceState,
        from: Arc<VulkanImage>,
        to: Arc<VulkanImage>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, VulkanRendererError> {
        trace!(?src, ?dst, ?filter, "recording vulkan blit");
        self.validate_blit_images(device, &from, &to, src, dst, filter)?;

        let from_layout = from.current_layout();
        if from_layout == vk::ImageLayout::UNDEFINED {
            return Err(VulkanRendererError::TemporaryFailure(
                "source image contents are undefined and cannot be blitted",
            ));
        }

        let to_layout = to.current_layout();
        let restore_to_layout = if to_layout == vk::ImageLayout::UNDEFINED {
            vk::ImageLayout::GENERAL
        } else {
            to_layout
        };

        let command_buffer = device.acquire_command_buffer()?;
        let vk_device = device.device_handle();
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: Command buffer belongs to this device command pool and is not currently in-flight.
        if let Err(err) = unsafe { vk_device.begin_command_buffer(command_buffer, &begin_info) } {
            let _ = device.discard_command_buffer(command_buffer);
            return Err(err.into());
        }
        device.insert_debug_label(command_buffer, c"vulkan.blit", [0.92, 0.74, 0.13, 1.0]);
        let mut foreign_images = acquire_images_from_foreign(
            vk_device,
            command_buffer,
            device.queue_family_index(),
            [(from.clone(), from_layout), (to.clone(), to_layout)],
        );

        transition_image_layout(
            vk_device,
            command_buffer,
            from.image(),
            from_layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        transition_image_layout(
            vk_device,
            command_buffer,
            to.image(),
            to_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        record_image_blit(vk_device, command_buffer, &from, &to, src, dst, filter);

        transition_image_layout(
            vk_device,
            command_buffer,
            from.image(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            from_layout,
        );
        transition_image_layout(
            vk_device,
            command_buffer,
            to.image(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            restore_to_layout,
        );
        if let Some((_, layout)) = foreign_images.get_mut(&from.id()) {
            *layout = from_layout;
        }
        if let Some((_, layout)) = foreign_images.get_mut(&to.id()) {
            *layout = restore_to_layout;
        }
        release_images_to_foreign(
            vk_device,
            command_buffer,
            device.queue_family_index(),
            &foreign_images,
        );

        // SAFETY: Command buffer recording is valid and all commands were encoded above.
        if let Err(err) = unsafe { vk_device.end_command_buffer(command_buffer) } {
            let _ = device.discard_command_buffer(command_buffer);
            restore_unsubmitted_foreign_acquires(&foreign_images);
            return Err(err.into());
        }

        let (_, submission_fence) = match device.submit_with_resources_and_fence(
            command_buffer,
            Vec::new(),
            vec![from.clone(), to.clone()],
        ) {
            Ok(submission) => submission,
            Err(err) => {
                let _ = device.discard_command_buffer(command_buffer);
                restore_unsubmitted_foreign_acquires(&foreign_images);
                return Err(err);
            }
        };

        from.set_layout(from_layout);
        to.set_layout(restore_to_layout);
        commit_foreign_releases(&foreign_images);

        Ok(SyncPoint::from(submission_fence))
    }

    #[instrument(level = "trace", skip(self, device, steps))]
    #[profiling::function]
    fn blit_texture_chain(
        &mut self,
        device: &mut DeviceState,
        steps: &[VulkanBlitChainStep],
    ) -> Result<SyncPoint, VulkanRendererError> {
        trace!(step_count = steps.len(), "recording vulkan blit chain");
        if steps.is_empty() {
            return Ok(SyncPoint::signaled());
        }

        let mut resolved_steps = Vec::with_capacity(steps.len());
        for step in steps {
            let Some(source) = step.source.image_resource().cloned() else {
                return Err(VulkanRendererError::NotImplemented(
                    "blit chain currently requires image-backed Vulkan source textures",
                ));
            };
            let Some(destination) = step.destination.image_resource().cloned() else {
                return Err(VulkanRendererError::NotImplemented(
                    "blit chain currently requires image-backed Vulkan destination textures",
                ));
            };
            self.validate_blit_images(
                device,
                &source,
                &destination,
                step.source_rect,
                step.destination_rect,
                step.filter,
            )?;
            resolved_steps.push(ResolvedBlitChainStep {
                source,
                destination,
                source_rect: step.source_rect,
                destination_rect: step.destination_rect,
                filter: step.filter,
            });
        }

        validate_blit_chain_source_layouts(&resolved_steps)?;

        let command_buffer = device.acquire_command_buffer()?;
        let vk_device = device.device_handle();
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: Command buffer belongs to this device command pool and is not currently in-flight.
        if let Err(err) = unsafe { vk_device.begin_command_buffer(command_buffer, &begin_info) } {
            let _ = device.discard_command_buffer(command_buffer);
            return Err(err.into());
        }
        device.insert_debug_label(command_buffer, c"vulkan.blit_chain", [0.94, 0.56, 0.18, 1.0]);
        let mut foreign_images = acquire_images_from_foreign(
            vk_device,
            command_buffer,
            device.queue_family_index(),
            resolved_steps.iter().flat_map(|step| {
                [
                    (step.source.clone(), step.source.current_layout()),
                    (step.destination.clone(), step.destination.current_layout()),
                ]
            }),
        );

        let mut layouts = IndexMap::new();
        for step in &resolved_steps {
            transition_tracked_image_layout(
                vk_device,
                command_buffer,
                &mut layouts,
                step.source.clone(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            transition_tracked_image_layout(
                vk_device,
                command_buffer,
                &mut layouts,
                step.destination.clone(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            record_image_blit(
                vk_device,
                command_buffer,
                &step.source,
                &step.destination,
                step.source_rect,
                step.destination_rect,
                step.filter,
            );
        }

        for tracked in layouts.values_mut() {
            transition_image_layout(
                vk_device,
                command_buffer,
                tracked.image.image(),
                tracked.current_layout,
                tracked.restore_layout,
            );
            tracked.current_layout = tracked.restore_layout;
        }
        for (id, (_, layout)) in foreign_images.iter_mut() {
            if let Some(tracked) = layouts.get(id) {
                *layout = tracked.current_layout;
            }
        }
        release_images_to_foreign(
            vk_device,
            command_buffer,
            device.queue_family_index(),
            &foreign_images,
        );

        // SAFETY: Command buffer recording is valid and all commands were encoded above.
        if let Err(err) = unsafe { vk_device.end_command_buffer(command_buffer) } {
            let _ = device.discard_command_buffer(command_buffer);
            restore_unsubmitted_foreign_acquires(&foreign_images);
            return Err(err.into());
        }

        let retained_images = resolved_steps
            .iter()
            .flat_map(|step| [step.source.clone(), step.destination.clone()])
            .collect::<Vec<_>>();
        let (_, submission_fence) =
            match device.submit_with_resources_and_fence(command_buffer, Vec::new(), retained_images) {
                Ok(submission) => submission,
                Err(err) => {
                    let _ = device.discard_command_buffer(command_buffer);
                    restore_unsubmitted_foreign_acquires(&foreign_images);
                    return Err(err);
                }
            };

        for tracked in layouts.values() {
            tracked.image.set_layout(tracked.restore_layout);
        }
        commit_foreign_releases(&foreign_images);

        Ok(SyncPoint::from(submission_fence))
    }

    pub(super) fn validate_blit_images(
        &mut self,
        device: &DeviceState,
        from: &VulkanImage,
        to: &VulkanImage,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<(), VulkanRendererError> {
        if from.id() == to.id() {
            return Err(VulkanRendererError::TemporaryFailure(
                "blit source and destination must be different images",
            ));
        }

        validate_rect(from.size(), src, "source")?;
        validate_rect(to.size(), dst, "destination")?;

        if !from.usage().contains(vk::ImageUsageFlags::TRANSFER_SRC) {
            return Err(VulkanRendererError::TemporaryFailure(
                "source image does not support transfer-source usage for blit",
            ));
        }
        if !to.usage().contains(vk::ImageUsageFlags::TRANSFER_DST) {
            return Err(VulkanRendererError::TemporaryFailure(
                "destination image does not support transfer-destination usage for blit",
            ));
        }

        if image_blit_required(from, to, src, dst) {
            let source_features = self.query_format_features(device, from.vk_format());
            let destination_features = self.query_format_features(device, to.vk_format());
            if !source_features.contains(vk::FormatFeatureFlags::BLIT_SRC)
                || !destination_features.contains(vk::FormatFeatureFlags::BLIT_DST)
            {
                return Err(VulkanRendererError::TemporaryFailure(
                    "scaled or format-converting blit is unsupported for the image formats on this device",
                ));
            }
            if filter == TextureFilter::Linear
                && !source_features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
            {
                return Err(VulkanRendererError::TemporaryFailure(
                    "linear filtered blit is unsupported for the source format",
                ));
            }
        }

        Ok(())
    }

    fn query_format_features(&mut self, device: &DeviceState, format: vk::Format) -> vk::FormatFeatureFlags {
        if let Some(features) = self.format_features.get(&format).copied() {
            return features;
        }

        // SAFETY: Instance/physical-device handles are valid for the lifetime of the renderer.
        let properties = unsafe {
            device
                .physical_device()
                .instance()
                .handle()
                .get_physical_device_format_properties(device.physical_device().handle(), format)
        };

        let features = properties.optimal_tiling_features;
        self.format_features.insert(format, features);
        features
    }
}

impl Blit for VulkanRenderer {
    #[instrument(level = "trace", skip(self, from, to))]
    #[profiling::function]
    fn blit(
        &mut self,
        from: &VulkanTarget,
        to: &mut VulkanTarget,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        let Some(from_image) = from.image_resource().cloned() else {
            return Err(VulkanRendererError::NotImplemented(
                "blit currently requires image-backed Vulkan source targets",
            ));
        };
        let Some(to_image) = to.image_resource().cloned() else {
            return Err(VulkanRendererError::NotImplemented(
                "blit currently requires image-backed Vulkan destination targets",
            ));
        };

        self.blit
            .blit_images(&mut self.device, from_image, to_image, src, dst, filter)
    }
}

impl VulkanRenderer {
    /// Blits a sequence of Vulkan textures in one command-buffer submission.
    #[instrument(level = "trace", skip(self, steps))]
    #[profiling::function]
    pub fn blit_texture_chain(
        &mut self,
        steps: &[VulkanBlitChainStep],
    ) -> Result<SyncPoint, VulkanRendererError> {
        self.blit.blit_texture_chain(&mut self.device, steps)
    }
}

fn validate_rect(
    bounds: Size<i32, BufferCoord>,
    rect: Rectangle<i32, Physical>,
    label: &'static str,
) -> Result<(), VulkanRendererError> {
    if rect.loc.x < 0 || rect.loc.y < 0 || rect.size.w <= 0 || rect.size.h <= 0 {
        return Err(VulkanRendererError::TemporaryFailure(match label {
            "source" => "source blit rectangle must be non-empty and non-negative",
            _ => "destination blit rectangle must be non-empty and non-negative",
        }));
    }

    let end_x = rect
        .loc
        .x
        .checked_add(rect.size.w)
        .ok_or(VulkanRendererError::TemporaryFailure(match label {
            "source" => "source blit x extent overflowed",
            _ => "destination blit x extent overflowed",
        }))?;
    let end_y = rect
        .loc
        .y
        .checked_add(rect.size.h)
        .ok_or(VulkanRendererError::TemporaryFailure(match label {
            "source" => "source blit y extent overflowed",
            _ => "destination blit y extent overflowed",
        }))?;

    if end_x > bounds.w || end_y > bounds.h {
        return Err(VulkanRendererError::TemporaryFailure(match label {
            "source" => "source blit rectangle exceeds source bounds",
            _ => "destination blit rectangle exceeds destination bounds",
        }));
    }

    Ok(())
}

fn validate_blit_chain_source_layouts(steps: &[ResolvedBlitChainStep]) -> Result<(), VulkanRendererError> {
    let mut layouts = IndexMap::new();
    for step in steps {
        let source_layout = layouts
            .get(&step.source.id())
            .copied()
            .unwrap_or_else(|| step.source.current_layout());
        if source_layout == vk::ImageLayout::UNDEFINED {
            return Err(VulkanRendererError::TemporaryFailure(
                "source image contents are undefined and cannot be blitted",
            ));
        }
        layouts.insert(step.source.id(), vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        layouts.insert(step.destination.id(), vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    }
    Ok(())
}

pub(crate) fn transition_tracked_image_layout(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    layouts: &mut IndexMap<u64, TrackedBlitImageLayout>,
    image: Arc<VulkanImage>,
    new_layout: vk::ImageLayout,
) {
    let entry = layouts.entry(image.id()).or_insert_with(|| {
        let current_layout = image.current_layout();
        let restore_layout = if current_layout == vk::ImageLayout::UNDEFINED {
            vk::ImageLayout::GENERAL
        } else {
            current_layout
        };
        TrackedBlitImageLayout {
            image,
            current_layout,
            restore_layout,
        }
    });
    if entry.current_layout == new_layout {
        return;
    }

    transition_image_layout(
        device,
        command_buffer,
        entry.image.image(),
        entry.current_layout,
        new_layout,
    );
    entry.current_layout = new_layout;
}

pub(super) fn record_image_blit(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    from: &VulkanImage,
    to: &VulkanImage,
    src: Rectangle<i32, Physical>,
    dst: Rectangle<i32, Physical>,
    filter: TextureFilter,
) {
    // vkCmdCopyImage copies storage bytes and therefore swaps logical red/blue
    // between BGRA and RGBA images. A Vulkan blit performs component-aware
    // format conversion, even when source and destination extents are equal.
    if image_blit_required(from, to, src, dst) {
        let blit_regions = [vk::ImageBlit::default()
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_offsets([
                vk::Offset3D {
                    x: src.loc.x,
                    y: src.loc.y,
                    z: 0,
                },
                vk::Offset3D {
                    x: src.loc.x + src.size.w,
                    y: src.loc.y + src.size.h,
                    z: 1,
                },
            ])
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .dst_offsets([
                vk::Offset3D {
                    x: dst.loc.x,
                    y: dst.loc.y,
                    z: 0,
                },
                vk::Offset3D {
                    x: dst.loc.x + dst.size.w,
                    y: dst.loc.y + dst.size.h,
                    z: 1,
                },
            ])];

        // SAFETY: Command buffer recording is active and source/destination images belong to this device.
        unsafe {
            device.cmd_blit_image(
                command_buffer,
                from.image(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                to.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &blit_regions,
                match filter {
                    TextureFilter::Linear => vk::Filter::LINEAR,
                    TextureFilter::Nearest => vk::Filter::NEAREST,
                },
            );
        }
    } else {
        let copy_regions = [vk::ImageCopy::default()
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_offset(vk::Offset3D {
                x: src.loc.x,
                y: src.loc.y,
                z: 0,
            })
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .dst_offset(vk::Offset3D {
                x: dst.loc.x,
                y: dst.loc.y,
                z: 0,
            })
            .extent(vk::Extent3D {
                width: src.size.w as u32,
                height: src.size.h as u32,
                depth: 1,
            })];

        // SAFETY: Command buffer recording is active and source/destination images belong to this device.
        unsafe {
            device.cmd_copy_image(
                command_buffer,
                from.image(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                to.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy_regions,
            );
        }
    }
}

fn image_blit_required(
    from: &VulkanImage,
    to: &VulkanImage,
    src: Rectangle<i32, Physical>,
    dst: Rectangle<i32, Physical>,
) -> bool {
    src.size != dst.size || from.vk_format() != to.vk_format()
}

#[cfg(test)]
mod tests {
    use ash::vk;

    use crate::{
        backend::{
            allocator::Fourcc,
            renderer::{
                vulkan::VulkanBlitChainStep, Bind, Blit, Color32F, ExportMem, Frame, Offscreen, Renderer,
                TextureFilter,
            },
            vulkan::{version::Version, Instance, PhysicalDevice},
        },
        utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform},
    };

    use super::super::image::{
        acquire_images_from_foreign, commit_foreign_releases, release_images_to_foreign,
    };
    use super::{VulkanRenderer, VulkanRendererError};

    fn init_renderer() -> Option<VulkanRenderer> {
        let instance = Instance::new(Version::VERSION_1_3, None).ok()?;
        let physical_device = PhysicalDevice::enumerate(&instance).ok()?.next()?;
        VulkanRenderer::new(&physical_device).ok()
    }

    fn first_working_offscreen_format(renderer: &mut VulkanRenderer) -> Option<Fourcc> {
        [
            Fourcc::Argb8888,
            Fourcc::Abgr8888,
            Fourcc::Xrgb8888,
            Fourcc::Xbgr8888,
        ]
        .into_iter()
        .find(|format| renderer.create_buffer(*format, Size::from((4, 4))).is_ok())
    }

    fn expected_red_pixel(format: Fourcc) -> [u8; 4] {
        match format {
            Fourcc::Argb8888 | Fourcc::Xrgb8888 => [0, 0, 255, 255],
            Fourcc::Abgr8888 | Fourcc::Xbgr8888 => [255, 0, 0, 255],
            _ => [255, 0, 0, 255],
        }
    }

    fn expected_blue_pixel(format: Fourcc) -> [u8; 4] {
        match format {
            Fourcc::Argb8888 | Fourcc::Xrgb8888 => [255, 0, 0, 255],
            Fourcc::Abgr8888 | Fourcc::Xbgr8888 => [0, 0, 255, 255],
            _ => [0, 0, 255, 255],
        }
    }

    #[test]
    fn renderer_local_image_is_excluded_from_foreign_barriers() {
        let Some(mut renderer) = init_renderer() else {
            return;
        };
        let Some(format) = first_working_offscreen_format(&mut renderer) else {
            return;
        };
        let texture = match renderer.create_buffer(format, Size::from((8, 8))) {
            Ok(texture) => texture,
            Err(_) => return,
        };
        let image = texture
            .image_resource()
            .expect("offscreen allocation should retain its imported image")
            .clone();
        assert!(image.is_renderer_local());
        assert!(!image.is_owned_by_foreign());

        let command_buffer = renderer
            .device
            .acquire_command_buffer()
            .expect("test command buffer should be available");
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: The command buffer belongs to this renderer and is not in use.
        unsafe {
            renderer
                .device
                .device_handle()
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("test command buffer should begin");
        }
        let acquired = acquire_images_from_foreign(
            renderer.device.device_handle(),
            command_buffer,
            renderer.device.queue_family_index(),
            [(image.clone(), image.current_layout())],
        );
        assert!(acquired.is_empty());
        assert!(!image.is_owned_by_foreign());
        release_images_to_foreign(
            renderer.device.device_handle(),
            command_buffer,
            renderer.device.queue_family_index(),
            &acquired,
        );
        // SAFETY: The ownership barriers above form a complete command buffer.
        unsafe {
            renderer
                .device
                .device_handle()
                .end_command_buffer(command_buffer)
                .expect("test command buffer should end");
        }
        renderer
            .device
            .submit_with_resources_and_fence(command_buffer, Vec::new(), vec![image.clone()])
            .expect("empty local ownership batch should submit");
        commit_foreign_releases(&acquired);
        assert!(!image.is_owned_by_foreign());
        assert_eq!(image.current_layout(), vk::ImageLayout::UNDEFINED);
        renderer
            .device
            .wait_for_all_submissions()
            .expect("local ownership test submission should complete");
    }

    #[test]
    fn blit_between_offscreen_targets_is_readback_correct() {
        let Some(mut renderer) = init_renderer() else {
            return;
        };

        let Some(format) = first_working_offscreen_format(&mut renderer) else {
            return;
        };

        let buffer_size: Size<i32, BufferCoord> = Size::from((16, 16));
        let physical_size: Size<i32, Physical> = Size::from((16, 16));
        let buffer_region: Rectangle<i32, BufferCoord> = Rectangle::from_size(buffer_size);
        let physical_region: Rectangle<i32, Physical> = Rectangle::from_size(physical_size);

        let mut src = match renderer.create_buffer(format, buffer_size) {
            Ok(texture) => texture,
            Err(_) => return,
        };
        let mut dst = match renderer.create_buffer(format, buffer_size) {
            Ok(texture) => texture,
            Err(_) => return,
        };

        let mut src_target = match renderer.bind(&mut src) {
            Ok(target) => target,
            Err(_) => return,
        };
        let mut dst_target = match renderer.bind(&mut dst) {
            Ok(target) => target,
            Err(_) => return,
        };

        {
            let mut frame = match renderer.render(&mut src_target, physical_size, Transform::Normal) {
                Ok(frame) => frame,
                Err(_) => return,
            };
            frame
                .clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &[physical_region])
                .expect("clear source target should succeed");
            let sync = frame.finish().expect("source frame finish should succeed");
            let _ = sync.wait();
        }

        {
            let mut frame = match renderer.render(&mut dst_target, physical_size, Transform::Normal) {
                Ok(frame) => frame,
                Err(_) => return,
            };
            frame
                .clear(Color32F::new(0.0, 0.0, 0.0, 1.0), &[physical_region])
                .expect("clear destination target should succeed");
            let sync = frame.finish().expect("destination frame finish should succeed");
            let _ = sync.wait();
        }

        let sync = renderer
            .blit(
                &src_target,
                &mut dst_target,
                physical_region,
                physical_region,
                TextureFilter::Nearest,
            )
            .expect("blit should succeed for offscreen targets");
        let _ = sync.wait();

        let mapping = renderer
            .copy_framebuffer(&dst_target, buffer_region, format)
            .expect("readback after blit should succeed");
        let bytes = renderer
            .map_texture(&mapping)
            .expect("map_texture should expose readback bytes");

        assert_eq!(&bytes[0..4], &expected_red_pixel(format));
    }

    #[test]
    fn blit_texture_chain_uses_one_submission_for_dependent_steps() {
        let Some(mut renderer) = init_renderer() else {
            return;
        };

        let Some(format) = first_working_offscreen_format(&mut renderer) else {
            return;
        };

        let src_size: Size<i32, BufferCoord> = Size::from((16, 16));
        let mid_size: Size<i32, BufferCoord> = Size::from((8, 8));
        let dst_size: Size<i32, BufferCoord> = Size::from((16, 16));
        let src_region: Rectangle<i32, Physical> =
            Rectangle::from_size(Size::<i32, Physical>::from((16, 16)));
        let mid_region: Rectangle<i32, Physical> = Rectangle::from_size(Size::<i32, Physical>::from((8, 8)));
        let dst_region: Rectangle<i32, Physical> =
            Rectangle::from_size(Size::<i32, Physical>::from((16, 16)));

        let mut src = match renderer.create_buffer(format, src_size) {
            Ok(texture) => texture,
            Err(_) => return,
        };
        let mid = match renderer.create_buffer(format, mid_size) {
            Ok(texture) => texture,
            Err(_) => return,
        };
        let mut dst = match renderer.create_buffer(format, dst_size) {
            Ok(texture) => texture,
            Err(_) => return,
        };

        let mut src_target = match renderer.bind(&mut src) {
            Ok(target) => target,
            Err(_) => return,
        };
        let dst_target = match renderer.bind(&mut dst) {
            Ok(target) => target,
            Err(_) => return,
        };

        {
            let mut frame = match renderer.render(&mut src_target, src_region.size, Transform::Normal) {
                Ok(frame) => frame,
                Err(_) => return,
            };
            frame
                .clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &[src_region])
                .expect("clear source target should succeed");
            let sync = frame.finish().expect("source frame finish should succeed");
            let _ = sync.wait();
        }

        let sync = renderer
            .blit_texture_chain(&[
                VulkanBlitChainStep::new(&src, &mid, src_region, mid_region, TextureFilter::Linear),
                VulkanBlitChainStep::new(&mid, &dst, mid_region, dst_region, TextureFilter::Linear),
            ])
            .expect("dependent blit chain should submit");
        let _ = sync.wait();

        let mapping = renderer
            .copy_framebuffer(&dst_target, Rectangle::from_size(dst_size), format)
            .expect("readback after blit chain should succeed");
        let bytes = renderer
            .map_texture(&mapping)
            .expect("map_texture should expose readback bytes");

        assert_eq!(&bytes[0..4], &expected_red_pixel(format));
    }

    #[test]
    fn equal_extent_cross_format_blit_preserves_logical_blue() {
        let Some(mut renderer) = init_renderer() else {
            return;
        };
        let size = Size::from((16, 16));
        let physical_size = Size::<i32, Physical>::from((16, 16));
        let region = Rectangle::from_size(physical_size);
        let buffer_region = Rectangle::<i32, BufferCoord>::from_size(size);

        let Ok(mut source) = renderer.create_buffer(Fourcc::Argb8888, size) else {
            return;
        };
        let Ok(mut destination) = renderer.create_buffer(Fourcc::Abgr8888, size) else {
            return;
        };
        let Ok(mut source_target) = renderer.bind(&mut source) else {
            return;
        };
        let Ok(mut destination_target) = renderer.bind(&mut destination) else {
            return;
        };

        {
            let mut frame = renderer
                .render(&mut source_target, physical_size, Transform::Normal)
                .expect("source frame");
            frame
                .clear(Color32F::new(0.0, 0.0, 1.0, 1.0), &[region])
                .expect("clear source blue");
            let _ = frame.finish().expect("finish source").wait();
        }

        let sync = renderer
            .blit(
                &source_target,
                &mut destination_target,
                region,
                region,
                TextureFilter::Nearest,
            )
            .expect("cross-format blit");
        let _ = sync.wait();

        let mapping = renderer
            .copy_framebuffer(&destination_target, buffer_region, Fourcc::Abgr8888)
            .expect("read destination");
        let bytes = renderer.map_texture(&mapping).expect("map destination");
        assert_eq!(&bytes[0..4], &expected_blue_pixel(Fourcc::Abgr8888));
    }

    #[test]
    fn blit_rejects_same_image_source_and_destination() {
        let Some(mut renderer) = init_renderer() else {
            return;
        };

        let Some(format) = first_working_offscreen_format(&mut renderer) else {
            return;
        };

        let buffer_size: Size<i32, BufferCoord> = Size::from((8, 8));
        let mut texture = match renderer.create_buffer(format, buffer_size) {
            Ok(texture) => texture,
            Err(_) => return,
        };
        let mut target = match renderer.bind(&mut texture) {
            Ok(target) => target,
            Err(_) => return,
        };

        let source = target.clone();
        let physical_region = Rectangle::from_size(Size::<i32, Physical>::from((8, 8)));
        let err = renderer
            .blit(
                &source,
                &mut target,
                physical_region,
                physical_region,
                TextureFilter::Nearest,
            )
            .expect_err("blit should reject same underlying image");

        assert!(matches!(err, VulkanRendererError::TemporaryFailure(_)));
    }
}
