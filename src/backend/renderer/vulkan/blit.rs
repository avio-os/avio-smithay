use ash::vk;
use indexmap::IndexMap;

use crate::{
    backend::renderer::{sync::SyncPoint, Blit, TextureFilter},
    utils::{Buffer as BufferCoord, Physical, Rectangle, Size},
};

use super::{
    device::DeviceState, dmabuf::ImportedDmabufImage, VulkanRenderer, VulkanRendererError, VulkanTarget,
};

#[derive(Debug, Default)]
pub(crate) struct BlitState {
    format_features: IndexMap<vk::Format, vk::FormatFeatureFlags>,
}

impl BlitState {
    fn blit_images(
        &mut self,
        device: &mut DeviceState,
        from: &ImportedDmabufImage,
        to: &ImportedDmabufImage,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, VulkanRendererError> {
        if from.id() == to.id() {
            return Err(VulkanRendererError::TemporaryFailure(
                "blit source and destination must be different images",
            ));
        }

        validate_rect(from.size(), src, "source")?;
        validate_rect(to.size(), dst, "destination")?;

        if from.vk_format() != to.vk_format() {
            return Err(VulkanRendererError::TemporaryFailure(
                "blit currently requires matching source and destination formats",
            ));
        }

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

        let scaled = src.size != dst.size;
        if scaled {
            let features = self.query_format_features(device, from.vk_format());
            if !features.contains(vk::FormatFeatureFlags::BLIT_SRC)
                || !features.contains(vk::FormatFeatureFlags::BLIT_DST)
            {
                return Err(VulkanRendererError::TemporaryFailure(
                    "scaled blit is unsupported for the image format on this device",
                ));
            }
            if filter == TextureFilter::Linear
                && !features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
            {
                return Err(VulkanRendererError::TemporaryFailure(
                    "linear filtered blit is unsupported for the source format",
                ));
            }
        }

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

        if scaled {
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
                vk_device.cmd_blit_image(
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
                vk_device.cmd_copy_image(
                    command_buffer,
                    from.image(),
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    to.image(),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &copy_regions,
                );
            }
        }

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

        // SAFETY: Command buffer recording is valid and all commands were encoded above.
        if let Err(err) = unsafe { vk_device.end_command_buffer(command_buffer) } {
            let _ = device.discard_command_buffer(command_buffer);
            return Err(err.into());
        }

        let (_, submission_fence) =
            match device.submit_with_framebuffers_and_fence(command_buffer, Vec::new()) {
                Ok(submission) => submission,
                Err(err) => {
                    let _ = device.discard_command_buffer(command_buffer);
                    return Err(err);
                }
            };

        from.set_layout(from_layout);
        to.set_layout(restore_to_layout);

        Ok(SyncPoint::from(submission_fence))
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
    fn blit(
        &mut self,
        from: &VulkanTarget,
        to: &mut VulkanTarget,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        let Some(from_image) = from.imported_image() else {
            return Err(VulkanRendererError::NotImplemented(
                "blit currently requires image-backed Vulkan source targets",
            ));
        };
        let Some(to_image) = to.imported_image() else {
            return Err(VulkanRendererError::NotImplemented(
                "blit currently requires image-backed Vulkan destination targets",
            ));
        };

        self.blit
            .blit_images(&mut self.device, from_image, to_image, src, dst, filter)
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

fn transition_image_layout(
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
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        )
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)];

    // SAFETY: Command buffer recording is active and image handle belongs to this device.
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

fn stage_access_for_layout(layout: vk::ImageLayout) -> (vk::PipelineStageFlags, vk::AccessFlags) {
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

#[cfg(test)]
mod tests {
    use crate::{
        backend::{
            allocator::Fourcc,
            renderer::{Bind, Blit, Color32F, ExportMem, Frame, Offscreen, Renderer, TextureFilter},
            vulkan::{version::Version, Instance, PhysicalDevice},
        },
        utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform},
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
