use std::sync::Arc;

use ash::vk;
use tracing::{instrument, trace};

use crate::{
    backend::{
        allocator::{format::get_bpp, Format, Fourcc, Modifier},
        renderer::{Bind, ExportMem, Offscreen, Texture, TextureMapping},
    },
    utils::{Buffer as BufferCoord, Rectangle, Size},
};

use super::{
    device::DeviceState,
    dmabuf::ImportedDmabufImage,
    format::{render_view_format, srgb_view_format_list, texture_view_components, ColorEncoding},
    VulkanRenderer, VulkanRendererError, VulkanTarget, VulkanTexture,
};

#[derive(Debug)]
pub(crate) struct ReadbackState {
    next_offscreen_id: u64,
}

impl Default for ReadbackState {
    fn default() -> Self {
        Self {
            next_offscreen_id: 1u64 << 61,
        }
    }
}

impl ReadbackState {
    fn next_offscreen_id(&mut self) -> u64 {
        let id = self.next_offscreen_id;
        self.next_offscreen_id = self.next_offscreen_id.wrapping_add(1);
        id
    }

    #[instrument(level = "trace", skip(self, device))]
    #[profiling::function]
    pub(crate) fn create_offscreen_texture(
        &mut self,
        device: &DeviceState,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
    ) -> Result<VulkanTexture, VulkanRendererError> {
        trace!(?format, ?size, "creating vulkan offscreen texture");
        if size.w <= 0 || size.h <= 0 {
            return Err(VulkanRendererError::TemporaryFailure(
                "offscreen buffer dimensions must be positive",
            ));
        }

        let vk_format = crate::backend::allocator::vulkan::format::get_vk_format(format)
            .ok_or(VulkanRendererError::UnsupportedMemoryFormat(format))?;
        if !format_supports_offscreen_usage(device, vk_format) {
            return Err(VulkanRendererError::UnsupportedMemoryFormat(format));
        }

        let device_handle = device.shared_device();
        let vk_device = device_handle.handle();
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;

        // Offscreens are both rendered into and sampled back, so they carry an `_SRGB`
        // attachment view alongside the encoded UNORM sampled view.
        let view_formats = srgb_view_format_list(vk_format);
        let mut format_list_info;
        let mut create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width: size.w as u32,
                height: size.h as u32,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .flags(match view_formats {
                Some(_) => vk::ImageCreateFlags::MUTABLE_FORMAT,
                None => vk::ImageCreateFlags::empty(),
            })
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        if let Some(formats) = view_formats.as_ref() {
            format_list_info = vk::ImageFormatListCreateInfo::default().view_formats(formats);
            create_info = create_info.push_next(&mut format_list_info);
        }

        // SAFETY: Device is valid and create info references live memory.
        let image = device_handle.observe_result(unsafe { vk_device.create_image(&create_info, None) })?;
        // SAFETY: Image belongs to this device and remains valid until explicit destruction.
        let memory_requirements = unsafe { vk_device.get_image_memory_requirements(image) };
        let memory_type_index = pick_image_memory_type(device, memory_requirements.memory_type_bits)
            .ok_or(VulkanRendererError::NoCompatibleMemoryType)?;

        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);
        // SAFETY: Device is valid and allocation info references live memory.
        let memory =
            match device_handle.observe_result(unsafe { vk_device.allocate_memory(&allocate_info, None) }) {
                Ok(memory) => memory,
                Err(err) => {
                    // SAFETY: Image belongs to this device and has not been bound.
                    device_handle.destroy_with(|vk_device| unsafe { vk_device.destroy_image(image, None) });
                    return Err(err.into());
                }
            };

        // SAFETY: Image and memory belong to this device and offset 0 is valid.
        if let Err(err) =
            device_handle.observe_result(unsafe { vk_device.bind_image_memory(image, memory, 0) })
        {
            // SAFETY: Handles belong to this device and were created above.
            device_handle.destroy_with(|vk_device| unsafe {
                vk_device.free_memory(memory, None);
                vk_device.destroy_image(image, None);
            });
            return Err(err.into());
        }

        let sampled_view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk_format)
            .components(texture_view_components(format, usage))
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        // SAFETY: Image view create info references a live image handle.
        let sampled_view = match device_handle
            .observe_result(unsafe { vk_device.create_image_view(&sampled_view_info, None) })
        {
            Ok(view) => view,
            Err(err) => {
                // SAFETY: Handles belong to this device and were created above.
                device_handle.destroy_with(|vk_device| unsafe {
                    vk_device.free_memory(memory, None);
                    vk_device.destroy_image(image, None);
                });
                return Err(err.into());
            }
        };

        let render_view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            // Linear-light blending, same rule as every other colour attachment.
            .format(render_view_format(vk_format))
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        // SAFETY: Image view create info references a live image handle.
        let render_view = match device_handle
            .observe_result(unsafe { vk_device.create_image_view(&render_view_info, None) })
        {
            Ok(view) => view,
            Err(err) => {
                // SAFETY: Handles belong to this device and were created above.
                device_handle.destroy_with(|vk_device| unsafe {
                    vk_device.destroy_image_view(sampled_view, None);
                    vk_device.free_memory(memory, None);
                    vk_device.destroy_image(image, None);
                });
                return Err(err.into());
            }
        };

        let imported = Arc::new(ImportedDmabufImage::new(
            self.next_offscreen_id(),
            image,
            memory,
            sampled_view,
            render_view,
            size,
            Format {
                code: format,
                modifier: Modifier::Invalid,
            },
            vk_format,
            // Offscreens are filled by our own linear-blending render passes, so their
            // stored bytes are the sRGB encoding of a premultiplied *linear* value.
            ColorEncoding::LinearPremultiplied,
            usage,
            false,
            vk::ImageLayout::UNDEFINED,
            device.shared_device(),
        ));

        Ok(VulkanTexture::from_dmabuf_import(
            imported,
            size,
            Some(format),
            false,
        ))
    }

    #[instrument(level = "trace", skip(self, device, image))]
    #[profiling::function]
    fn copy_image_to_mapping(
        &mut self,
        device: &mut DeviceState,
        image: &ImportedDmabufImage,
        region: Rectangle<i32, BufferCoord>,
        dst_format: Fourcc,
    ) -> Result<VulkanMapping, VulkanRendererError> {
        trace!(?region, src = ?image.format().code, dst = ?dst_format, "copying image to cpu mapping");
        validate_region(image.size(), region)?;

        let src_format = image.format().code;
        let src_bpp = bytes_per_pixel(src_format)?;
        let src_len = region
            .size
            .w
            .try_into()
            .ok()
            .and_then(|w: usize| {
                region
                    .size
                    .h
                    .try_into()
                    .ok()
                    .and_then(move |h: usize| w.checked_mul(h))
            })
            .and_then(|pixel_count| pixel_count.checked_mul(src_bpp))
            .ok_or(VulkanRendererError::TemporaryFailure(
                "readback buffer size overflowed",
            ))?;

        let (staging_buffer, staging_memory, coherent) = create_readback_buffer(device, src_len)?;
        let cleanup_device = device.shared_device();
        let cleanup_buffer_device = cleanup_device.clone();
        let cleanup_buffer = scopeguard::guard(staging_buffer, |buffer| {
            // SAFETY: Buffer belongs to this device and is no longer referenced after readback completion.
            cleanup_buffer_device.destroy_with(|device| unsafe { device.destroy_buffer(buffer, None) });
        });
        let cleanup_memory_device = cleanup_device.clone();
        let cleanup_memory = scopeguard::guard(staging_memory, |memory| {
            // SAFETY: Memory belongs to this device and is no longer referenced after readback completion.
            cleanup_memory_device.destroy_with(|device| unsafe { device.free_memory(memory, None) });
        });

        let vk_device = cleanup_device.handle();
        let command_buffer = device.acquire_command_buffer()?;
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: Command buffer belongs to this device command pool and is not in-flight.
        if let Err(err) = unsafe { vk_device.begin_command_buffer(command_buffer, &begin_info) } {
            let _ = device.discard_command_buffer(command_buffer);
            return Err(err.into());
        }
        device.insert_debug_label(command_buffer, c"vulkan.readback", [0.52, 0.47, 0.91, 1.0]);

        let old_layout = image.current_layout();
        transition_image_layout(
            vk_device,
            command_buffer,
            image.image(),
            old_layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );

        let copy_region = [vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D {
                x: region.loc.x,
                y: region.loc.y,
                z: 0,
            })
            .image_extent(vk::Extent3D {
                width: region.size.w as u32,
                height: region.size.h as u32,
                depth: 1,
            })];

        // SAFETY: Command buffer recording is active and all handles belong to this device.
        unsafe {
            vk_device.cmd_copy_image_to_buffer(
                command_buffer,
                image.image(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                *cleanup_buffer,
                &copy_region,
            );
        }

        transition_image_layout(
            vk_device,
            command_buffer,
            image.image(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            old_layout,
        );

        // SAFETY: Command buffer recording is valid and ready for submission.
        if let Err(err) = unsafe { vk_device.end_command_buffer(command_buffer) } {
            let _ = device.discard_command_buffer(command_buffer);
            return Err(err.into());
        }

        device.submit_blocking(command_buffer)?;
        image.set_layout(old_layout);

        let mut raw = vec![0u8; src_len];
        // SAFETY: Memory belongs to this device and the mapped range is within allocation bounds.
        let mapped = unsafe {
            vk_device.map_memory(
                *cleanup_memory,
                0,
                src_len as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }?;

        if !coherent {
            let ranges = [vk::MappedMemoryRange::default()
                .memory(*cleanup_memory)
                .offset(0)
                .size(src_len as vk::DeviceSize)];
            // SAFETY: The mapped range belongs to this memory allocation.
            if let Err(err) = unsafe { vk_device.invalidate_mapped_memory_ranges(&ranges) } {
                // SAFETY: Memory is currently mapped and must be unmapped before returning.
                unsafe { vk_device.unmap_memory(*cleanup_memory) };
                return Err(err.into());
            }
        }

        // SAFETY: Source pointer is valid for `src_len` bytes and destination vec has exact capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(mapped as *const u8, raw.as_mut_ptr(), src_len);
            vk_device.unmap_memory(*cleanup_memory);
        }

        let converted = convert_pixels(src_format, dst_format, &raw)?;
        Ok(VulkanMapping {
            data: converted,
            size: region.size,
            format: dst_format,
            flipped: false,
        })
    }
}

#[derive(Debug, Clone)]
pub struct VulkanMapping {
    data: Vec<u8>,
    size: Size<i32, BufferCoord>,
    format: Fourcc,
    flipped: bool,
}

impl Texture for VulkanMapping {
    fn width(&self) -> u32 {
        self.size.w as u32
    }

    fn height(&self) -> u32 {
        self.size.h as u32
    }

    fn size(&self) -> Size<i32, BufferCoord> {
        self.size
    }

    fn format(&self) -> Option<Fourcc> {
        Some(self.format)
    }
}

impl TextureMapping for VulkanMapping {
    fn flipped(&self) -> bool {
        self.flipped
    }

    fn format(&self) -> Fourcc {
        self.format
    }
}

impl Offscreen<VulkanTexture> for VulkanRenderer {
    fn create_buffer(
        &mut self,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
    ) -> Result<VulkanTexture, Self::Error> {
        self.readback.create_offscreen_texture(&self.device, format, size)
    }
}

impl Bind<VulkanTexture> for VulkanRenderer {
    fn bind<'a>(&mut self, target: &'a mut VulkanTexture) -> Result<Self::Framebuffer<'a>, Self::Error> {
        let Some(imported) = target.imported_image().cloned() else {
            return Err(VulkanRendererError::NotImplemented(
                "binding VulkanTexture currently requires an image-backed texture",
            ));
        };

        Ok(VulkanTarget::from_imported_image(
            imported,
            target.size(),
            target.format(),
        ))
    }
}

impl ExportMem for VulkanRenderer {
    type TextureMapping = VulkanMapping;

    fn copy_framebuffer(
        &mut self,
        target: &Self::Framebuffer<'_>,
        region: Rectangle<i32, BufferCoord>,
        format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        let Some(image) = target.imported_image() else {
            return Err(VulkanRendererError::NotImplemented(
                "copy_framebuffer currently requires image-backed VulkanTarget",
            ));
        };

        self.readback
            .copy_image_to_mapping(&mut self.device, image, region, format)
    }

    fn copy_texture(
        &mut self,
        texture: &Self::TextureId,
        region: Rectangle<i32, BufferCoord>,
        format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        let Some(image) = texture.imported_image() else {
            return Err(VulkanRendererError::NotImplemented(
                "copy_texture currently requires image-backed VulkanTexture",
            ));
        };

        self.readback
            .copy_image_to_mapping(&mut self.device, image, region, format)
    }

    fn can_read_texture(&mut self, texture: &Self::TextureId) -> Result<bool, Self::Error> {
        let Some(image) = texture.imported_image() else {
            return Ok(false);
        };

        Ok(bytes_per_pixel(image.format().code).is_ok())
    }

    fn map_texture<'a>(
        &mut self,
        texture_mapping: &'a Self::TextureMapping,
    ) -> Result<&'a [u8], Self::Error> {
        Ok(&texture_mapping.data)
    }
}

fn format_supports_offscreen_usage(device: &DeviceState, format: vk::Format) -> bool {
    let properties = unsafe {
        device
            .physical_device()
            .instance()
            .handle()
            .get_physical_device_format_properties(device.physical_device().handle(), format)
    };

    let required = vk::FormatFeatureFlags::COLOR_ATTACHMENT
        | vk::FormatFeatureFlags::SAMPLED_IMAGE
        | vk::FormatFeatureFlags::TRANSFER_SRC
        | vk::FormatFeatureFlags::TRANSFER_DST;
    properties.optimal_tiling_features.contains(required)
}

fn validate_region(
    image_size: Size<i32, BufferCoord>,
    region: Rectangle<i32, BufferCoord>,
) -> Result<(), VulkanRendererError> {
    if region.loc.x < 0 || region.loc.y < 0 || region.size.w <= 0 || region.size.h <= 0 {
        return Err(VulkanRendererError::TemporaryFailure(
            "readback region must be positive and non-empty",
        ));
    }

    let end_x = region
        .loc
        .x
        .checked_add(region.size.w)
        .ok_or(VulkanRendererError::TemporaryFailure(
            "readback x extent overflowed",
        ))?;
    let end_y = region
        .loc
        .y
        .checked_add(region.size.h)
        .ok_or(VulkanRendererError::TemporaryFailure(
            "readback y extent overflowed",
        ))?;

    if end_x > image_size.w || end_y > image_size.h {
        return Err(VulkanRendererError::TemporaryFailure(
            "readback region exceeds source bounds",
        ));
    }

    Ok(())
}

fn create_readback_buffer(
    device: &DeviceState,
    size: usize,
) -> Result<(vk::Buffer, vk::DeviceMemory, bool), VulkanRendererError> {
    let device_handle = device.shared_device();
    let vk_device = device_handle.handle();
    let create_info = vk::BufferCreateInfo::default()
        .size(size as u64)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    // SAFETY: Device is valid and create info references live data.
    let buffer = device_handle.observe_result(unsafe { vk_device.create_buffer(&create_info, None) })?;
    // SAFETY: Buffer belongs to this device and remains valid until destroyed.
    let memory_requirements = unsafe { vk_device.get_buffer_memory_requirements(buffer) };

    let (memory_type_index, coherent) =
        pick_host_visible_memory_type(device, memory_requirements.memory_type_bits)
            .ok_or(VulkanRendererError::NoCompatibleMemoryType)?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(memory_requirements.size)
        .memory_type_index(memory_type_index);
    // SAFETY: Device is valid and allocation info references live data.
    let memory = match device_handle.observe_result(unsafe { vk_device.allocate_memory(&alloc_info, None) }) {
        Ok(memory) => memory,
        Err(err) => {
            // SAFETY: Buffer belongs to this device and allocation failed before binding.
            device_handle.destroy_with(|vk_device| unsafe { vk_device.destroy_buffer(buffer, None) });
            return Err(err.into());
        }
    };

    // SAFETY: Buffer and memory belong to this device and offset 0 is valid for this allocation.
    if let Err(err) = device_handle.observe_result(unsafe { vk_device.bind_buffer_memory(buffer, memory, 0) })
    {
        // SAFETY: Handles belong to this device and were created above.
        device_handle.destroy_with(|vk_device| unsafe {
            vk_device.free_memory(memory, None);
            vk_device.destroy_buffer(buffer, None);
        });
        return Err(err.into());
    }

    Ok((buffer, memory, coherent))
}

fn pick_image_memory_type(device: &DeviceState, memory_type_bits: u32) -> Option<u32> {
    let memory_properties = unsafe {
        device
            .physical_device()
            .instance()
            .handle()
            .get_physical_device_memory_properties(device.physical_device().handle())
    };

    pick_memory_type_index(
        &memory_properties,
        memory_type_bits,
        vk::MemoryPropertyFlags::empty(),
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .map(|(index, _)| index)
}

fn pick_host_visible_memory_type(device: &DeviceState, memory_type_bits: u32) -> Option<(u32, bool)> {
    let memory_properties = unsafe {
        device
            .physical_device()
            .instance()
            .handle()
            .get_physical_device_memory_properties(device.physical_device().handle())
    };

    pick_memory_type_index(
        &memory_properties,
        memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .map(|(index, flags)| (index, flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)))
}

fn pick_memory_type_index(
    properties: &vk::PhysicalDeviceMemoryProperties,
    memory_type_bits: u32,
    required: vk::MemoryPropertyFlags,
    preferred: vk::MemoryPropertyFlags,
) -> Option<(u32, vk::MemoryPropertyFlags)> {
    let mut fallback = None;

    for index in 0..properties.memory_type_count {
        let mask = 1u32 << index;
        if memory_type_bits & mask == 0 {
            continue;
        }

        let flags = properties.memory_types[index as usize].property_flags;
        if !flags.contains(required) {
            continue;
        }

        if flags.contains(preferred) {
            return Some((index, flags));
        }

        if fallback.is_none() {
            fallback = Some((index, flags));
        }
    }

    fallback
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

    // SAFETY: Command buffer recording is active and barrier references a live image.
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

fn bytes_per_pixel(format: Fourcc) -> Result<usize, VulkanRendererError> {
    let bits = get_bpp(format).ok_or(VulkanRendererError::UnsupportedMemoryFormat(format))?;
    if bits % 8 != 0 {
        return Err(VulkanRendererError::UnsupportedMemoryFormat(format));
    }
    Ok(bits / 8)
}

fn is_alpha_equivalent_pair(src: Fourcc, dst: Fourcc) -> bool {
    matches!(
        (src, dst),
        (Fourcc::Argb8888, Fourcc::Xrgb8888)
            | (Fourcc::Xrgb8888, Fourcc::Argb8888)
            | (Fourcc::Abgr8888, Fourcc::Xbgr8888)
            | (Fourcc::Xbgr8888, Fourcc::Abgr8888)
            | (Fourcc::Bgra8888, Fourcc::Bgrx8888)
            | (Fourcc::Bgrx8888, Fourcc::Bgra8888)
            | (Fourcc::Rgba8888, Fourcc::Rgbx8888)
            | (Fourcc::Rgbx8888, Fourcc::Rgba8888)
    )
}

fn convert_pixels(src: Fourcc, dst: Fourcc, pixels: &[u8]) -> Result<Vec<u8>, VulkanRendererError> {
    let src_bpp = bytes_per_pixel(src)?;
    let dst_bpp = bytes_per_pixel(dst)?;
    if src_bpp != dst_bpp {
        return Err(VulkanRendererError::UnsupportedMemoryFormat(dst));
    }

    if src == dst {
        return Ok(pixels.to_vec());
    }

    if !is_alpha_equivalent_pair(src, dst) {
        return Err(VulkanRendererError::UnsupportedMemoryFormat(dst));
    }

    let mut converted = pixels.to_vec();
    if src_bpp == 4 {
        for pixel in converted.chunks_exact_mut(4) {
            pixel[3] = 0xFF;
        }
    }

    Ok(converted)
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::{
            allocator::Fourcc,
            renderer::{
                damage::OutputDamageTracker,
                element::{solid::SolidColorBuffer, solid::SolidColorRenderElement, Kind},
                Bind, Color32F, ExportMem, Frame, Offscreen, Renderer, Texture,
            },
            vulkan::{version::Version, Instance, PhysicalDevice},
        },
        utils::{Buffer as BufferCoord, Physical, Point, Rectangle, Size, Transform},
    };

    use super::VulkanRenderer;

    fn init_renderer() -> Option<VulkanRenderer> {
        let instance = Instance::new(Version::VERSION_1_3, None).ok()?;
        let physical_device = PhysicalDevice::enumerate(&instance).ok()?.next()?;
        VulkanRenderer::new(&physical_device).ok()
    }

    fn first_working_offscreen_format(renderer: &mut VulkanRenderer) -> Option<Fourcc> {
        for format in [
            Fourcc::Argb8888,
            Fourcc::Abgr8888,
            Fourcc::Xrgb8888,
            Fourcc::Xbgr8888,
        ] {
            if renderer.create_buffer(format, Size::from((4, 4))).is_ok() {
                return Some(format);
            }
        }
        None
    }

    fn expected_clear_pixel(format: Fourcc) -> [u8; 4] {
        // clear color is r=1.0, g=0.5, b=0.25, a=1.0
        match format {
            Fourcc::Argb8888 | Fourcc::Xrgb8888 => [64, 128, 255, 255],
            Fourcc::Abgr8888 | Fourcc::Xbgr8888 => [255, 128, 64, 255],
            _ => [255, 128, 64, 255],
        }
    }

    fn assert_clear_pixel_matches_unorm_quantization(actual: &[u8], format: Fourcc) {
        let expected = expected_clear_pixel(format);
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().copied().zip(expected) {
            // 0.5 * 255 is exactly halfway between 127 and 128. Vulkan asks
            // implementations to round normalized fixed-point conversions to
            // nearest but does not prescribe the halfway direction. All other
            // components in this clear have one exact expected encoding.
            if expected == 128 {
                assert!(matches!(actual, 127 | 128));
            } else {
                assert_eq!(actual, expected);
            }
        }
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

    fn pixel_at(bytes: &[u8], width: usize, x: usize, y: usize) -> [u8; 4] {
        let offset = ((y * width) + x) * 4;
        [
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]
    }

    #[test]
    fn offscreen_render_and_export_mapping() {
        let Some(mut renderer) = init_renderer() else {
            return;
        };

        let Some(format) = first_working_offscreen_format(&mut renderer) else {
            return;
        };

        let buffer_size: Size<i32, BufferCoord> = Size::from((16, 16));
        let physical_size: Size<i32, Physical> = Size::from((16, 16));
        let mut offscreen = match renderer.create_buffer(format, buffer_size) {
            Ok(texture) => texture,
            Err(_) => return,
        };

        let mut target = match renderer.bind(&mut offscreen) {
            Ok(target) => target,
            Err(_) => return,
        };

        {
            let mut frame = match renderer.render(&mut target, physical_size, Transform::Normal) {
                Ok(frame) => frame,
                Err(_) => return,
            };
            frame
                .clear(
                    Color32F::new(1.0, 0.5, 0.25, 1.0),
                    &[Rectangle::from_size(physical_size)],
                )
                .expect("clear should succeed on offscreen target");
            let sync = frame.finish().expect("finish should succeed");
            let _ = sync.wait();
        }

        let region = Rectangle::from_size(buffer_size);
        let mapping = renderer
            .copy_texture(&offscreen, region, format)
            .expect("copy_texture should succeed for offscreen texture");
        let bytes = renderer
            .map_texture(&mapping)
            .expect("map_texture should expose readback bytes");

        assert_eq!(bytes.len(), (buffer_size.w * buffer_size.h * 4) as usize);
        assert_clear_pixel_matches_unorm_quantization(&bytes[0..4], format);

        let fallback_format = match format {
            Fourcc::Argb8888 => Fourcc::Xrgb8888,
            Fourcc::Abgr8888 => Fourcc::Xbgr8888,
            Fourcc::Xrgb8888 => Fourcc::Argb8888,
            Fourcc::Xbgr8888 => Fourcc::Abgr8888,
            _ => format,
        };

        let converted = renderer
            .copy_framebuffer(&target, region, fallback_format)
            .expect("copy_framebuffer should support alpha-equivalent format conversions");
        let converted_bytes = renderer
            .map_texture(&converted)
            .expect("map_texture should expose converted bytes");

        assert_eq!(Texture::format(&converted), Some(fallback_format));
        assert_eq!(converted_bytes[3], 0xFF);
    }

    #[test]
    fn damage_tracker_scene_composition_tracks_multi_frame_updates() {
        let Some(mut renderer) = init_renderer() else {
            return;
        };

        let Some(format) = first_working_offscreen_format(&mut renderer) else {
            return;
        };

        let buffer_size: Size<i32, BufferCoord> = Size::from((64, 64));
        let physical_size: Size<i32, Physical> = Size::from((64, 64));
        let mut offscreen = match renderer.create_buffer(format, buffer_size) {
            Ok(texture) => texture,
            Err(_) => return,
        };
        let mut target = match renderer.bind(&mut offscreen) {
            Ok(target) => target,
            Err(_) => return,
        };

        let mut damage_tracker = OutputDamageTracker::new(physical_size, 1.0, Transform::Normal);
        let background = SolidColorBuffer::new((64, 64), Color32F::new(0.0, 0.0, 1.0, 1.0));
        let foreground = SolidColorBuffer::new((16, 64), Color32F::new(1.0, 0.0, 0.0, 1.0));

        {
            let foreground_element = SolidColorRenderElement::from_buffer(
                &foreground,
                Point::from((8, 0)),
                1.0,
                1.0,
                Kind::Unspecified,
            );
            let background_element = SolidColorRenderElement::from_buffer(
                &background,
                Point::from((0, 0)),
                1.0,
                1.0,
                Kind::Unspecified,
            );
            let elements = vec![foreground_element, background_element];

            let result = damage_tracker
                .render_output(
                    &mut renderer,
                    &mut target,
                    0,
                    &elements,
                    Color32F::new(0.0, 0.0, 0.0, 1.0),
                )
                .expect("initial scene composition should render");
            assert!(
                result.damage.is_some(),
                "initial frame should include full output damage"
            );
            let _ = result.sync.wait();
        }

        let full_region = Rectangle::from_size(buffer_size);
        let first_frame = renderer
            .copy_framebuffer(&target, full_region, format)
            .expect("first scene readback should succeed");
        let first_bytes = renderer
            .map_texture(&first_frame)
            .expect("first scene bytes should be mappable");
        let width = buffer_size.w as usize;
        assert_eq!(
            pixel_at(first_bytes, width, 4, 10),
            expected_blue_pixel(format),
            "background region should stay blue",
        );
        assert_eq!(
            pixel_at(first_bytes, width, 10, 10),
            expected_red_pixel(format),
            "foreground stripe should render red",
        );

        {
            let foreground_element = SolidColorRenderElement::from_buffer(
                &foreground,
                Point::from((8, 0)),
                1.0,
                1.0,
                Kind::Unspecified,
            );
            let background_element = SolidColorRenderElement::from_buffer(
                &background,
                Point::from((0, 0)),
                1.0,
                1.0,
                Kind::Unspecified,
            );
            let elements = vec![foreground_element, background_element];

            let result = damage_tracker
                .render_output(
                    &mut renderer,
                    &mut target,
                    1,
                    &elements,
                    Color32F::new(0.0, 0.0, 0.0, 1.0),
                )
                .expect("unchanged scene should be accepted");
            assert!(
                result.damage.is_none(),
                "unchanged frame should skip rendering damage",
            );
            let _ = result.sync.wait();
        }

        {
            let foreground_element = SolidColorRenderElement::from_buffer(
                &foreground,
                Point::from((24, 0)),
                1.0,
                1.0,
                Kind::Unspecified,
            );
            let background_element = SolidColorRenderElement::from_buffer(
                &background,
                Point::from((0, 0)),
                1.0,
                1.0,
                Kind::Unspecified,
            );
            let elements = vec![foreground_element, background_element];

            let result = damage_tracker
                .render_output(
                    &mut renderer,
                    &mut target,
                    1,
                    &elements,
                    Color32F::new(0.0, 0.0, 0.0, 1.0),
                )
                .expect("moved scene should produce incremental damage");
            assert!(result.damage.is_some(), "moved element should generate damage",);
            let _ = result.sync.wait();
        }

        let moved_frame = renderer
            .copy_framebuffer(&target, full_region, format)
            .expect("moved scene readback should succeed");
        let moved_bytes = renderer
            .map_texture(&moved_frame)
            .expect("moved scene bytes should be mappable");
        assert_eq!(
            pixel_at(moved_bytes, width, 10, 10),
            expected_blue_pixel(format),
            "old stripe position should be restored by fallback composition",
        );
        assert_eq!(
            pixel_at(moved_bytes, width, 26, 10),
            expected_red_pixel(format),
            "new stripe position should be red after composition update",
        );
    }
}
