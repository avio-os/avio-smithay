use ash::vk;
use tracing::{instrument, trace};

use crate::{
    backend::{
        allocator::{format::get_bpp, Format, Fourcc, Modifier},
        renderer::{ImportMem, MemoryUploadCapacityEdge, MemoryUploadErrorKind, Texture},
    },
    utils::{Buffer as BufferCoord, Rectangle, Size},
};

#[cfg(feature = "wayland_frontend")]
use crate::{
    backend::renderer::ImportMemWl,
    reexports::wayland_server::protocol::wl_buffer,
    wayland::{compositor::SurfaceData, shm},
};

use super::{
    device::DeviceState,
    dmabuf::ImportedDmabufImage,
    format::{texture_view_components, ColorEncoding},
    VulkanRenderer, VulkanRendererError, VulkanTexture,
};

const SUPPORTED_MEMORY_FORMATS: &[Fourcc] = &[
    Fourcc::Abgr8888,
    Fourcc::Xbgr8888,
    Fourcc::Argb8888,
    Fourcc::Xrgb8888,
];

#[derive(Debug)]
pub(crate) struct UploadState {
    next_upload_id: u64,
}

impl Default for UploadState {
    fn default() -> Self {
        Self {
            next_upload_id: 1u64 << 63,
        }
    }
}

impl UploadState {
    pub(crate) fn supported_formats(&self) -> &'static [Fourcc] {
        SUPPORTED_MEMORY_FORMATS
    }

    fn next_upload_id(&mut self) -> u64 {
        let id = self.next_upload_id;
        self.next_upload_id = self.next_upload_id.wrapping_add(1);
        id
    }

    #[instrument(level = "trace", skip(self, device, data))]
    #[profiling::function]
    pub(crate) fn import_memory(
        &mut self,
        device: &mut DeviceState,
        data: &[u8],
        format: Fourcc,
        size: Size<i32, BufferCoord>,
        flipped: bool,
    ) -> Result<VulkanTexture, VulkanRendererError> {
        trace!(?format, ?size, flipped, "importing memory into vulkan texture");
        let _ = validate_memory_format(format)?;
        let expected_len = expected_len_for_size(format, size)?;
        if data.len() < expected_len {
            return Err(VulkanRendererError::InvalidMemoryUpload(
                "memory buffer is smaller than the declared import dimensions",
            ));
        }

        let image = create_upload_image(device, self.next_upload_id(), format, size, flipped)?;
        upload_region_to_image(device, &image, format, data, Rectangle::from_size(size))?;

        Ok(VulkanTexture::from_memory_import(image, size, format, flipped))
    }

    #[instrument(level = "trace", skip(self, device, texture, data))]
    #[profiling::function]
    pub(crate) fn update_memory(
        &mut self,
        device: &mut DeviceState,
        texture: &VulkanTexture,
        data: &[u8],
        region: Rectangle<i32, BufferCoord>,
    ) -> Result<(), VulkanRendererError> {
        trace!(?region, "updating vulkan texture memory region");
        if !texture.memory_writable() {
            return Err(VulkanRendererError::InvalidMemoryUpload(
                "texture is not writable through ImportMem::update_memory",
            ));
        }

        let Some(format) = texture.format() else {
            return Err(VulkanRendererError::InvalidMemoryUpload(
                "memory-backed texture format metadata is missing",
            ));
        };
        let _ = validate_memory_format(format)?;

        let texture_size = texture.size();
        let expected_len = expected_len_for_size(format, texture_size)?;
        if data.len() < expected_len {
            return Err(VulkanRendererError::InvalidMemoryUpload(
                "memory buffer is smaller than the texture size required for partial updates",
            ));
        }

        if region.is_empty() {
            return Ok(());
        }
        validate_region(texture_size, region)?;

        let Some(image) = texture.imported_image() else {
            return Err(VulkanRendererError::InvalidMemoryUpload(
                "memory-backed texture image is missing",
            ));
        };

        upload_region_to_image(device, image, format, data, region)
    }
}

impl ImportMem for VulkanRenderer {
    fn import_memory(
        &mut self,
        data: &[u8],
        format: Fourcc,
        size: Size<i32, BufferCoord>,
        flipped: bool,
    ) -> Result<Self::TextureId, Self::Error> {
        self.upload
            .import_memory(&mut self.device, data, format, size, flipped)
    }

    fn update_memory(
        &mut self,
        texture: &Self::TextureId,
        data: &[u8],
        region: Rectangle<i32, BufferCoord>,
    ) -> Result<(), Self::Error> {
        self.upload.update_memory(&mut self.device, texture, data, region)
    }

    fn memory_upload_error_kind(error: &Self::Error) -> MemoryUploadErrorKind {
        if error.is_upload_deferred() {
            MemoryUploadErrorKind::DeferredCapacity
        } else {
            MemoryUploadErrorKind::Other
        }
    }

    fn memory_upload_capacity_edge(&mut self) -> Result<MemoryUploadCapacityEdge, Self::Error> {
        self.device.memory_upload_capacity_edge()
    }

    fn mem_formats(&self) -> Box<dyn Iterator<Item = Fourcc>> {
        Box::new(self.upload.supported_formats().iter().copied())
    }
}

#[cfg(feature = "wayland_frontend")]
impl ImportMemWl for VulkanRenderer {
    fn import_shm_buffer(
        &mut self,
        buffer: &wl_buffer::WlBuffer,
        _surface: Option<&SurfaceData>,
        _damage: &[Rectangle<i32, BufferCoord>],
    ) -> Result<Self::TextureId, Self::Error> {
        with_packed_shm_buffer(buffer, |packed, format, size| {
            self.upload
                .import_memory(&mut self.device, packed, format, size, false)
        })
    }
}

#[cfg(feature = "wayland_frontend")]
impl VulkanRenderer {
    pub(crate) fn update_shm_texture(
        &mut self,
        texture: &VulkanTexture,
        buffer: &wl_buffer::WlBuffer,
        damage: &[Rectangle<i32, BufferCoord>],
    ) -> Result<VulkanTexture, VulkanRendererError> {
        if damage.is_empty() {
            return Ok(texture.clone());
        }

        with_packed_shm_buffer(buffer, |packed, format, size| {
            let update_region = damage
                .iter()
                .copied()
                .reduce(|a, b| a.merge(b))
                .unwrap_or_else(|| Rectangle::from_size(size));

            if update_region.is_empty() {
                return Ok(texture.clone());
            }

            match self
                .upload
                .update_memory(&mut self.device, texture, packed, update_region)
            {
                Ok(()) => Ok(texture.clone()),
                Err(error) if error.is_upload_deferred() => Err(error),
                Err(_) => self
                    .upload
                    .import_memory(&mut self.device, packed, format, size, false),
            }
        })
    }
}

#[cfg(feature = "wayland_frontend")]
fn with_packed_shm_buffer<T, F>(buffer: &wl_buffer::WlBuffer, f: F) -> Result<T, VulkanRendererError>
where
    F: FnOnce(&[u8], Fourcc, Size<i32, BufferCoord>) -> Result<T, VulkanRendererError>,
{
    shm::with_buffer_contents(buffer, |ptr, len, data| {
        let format = shm::shm_format_to_fourcc(data.format).ok_or(
            VulkanRendererError::InvalidMemoryUpload("wl_shm buffer format is unsupported"),
        )?;
        let bytes_per_pixel = bytes_per_pixel(format)?;

        if data.width <= 0 || data.height <= 0 {
            return Err(VulkanRendererError::InvalidMemoryUpload(
                "wl_shm buffer dimensions must be positive",
            ));
        }
        if data.stride <= 0 {
            return Err(VulkanRendererError::InvalidMemoryUpload(
                "wl_shm buffer stride must be positive",
            ));
        }

        let row_bytes = (data.width as usize).checked_mul(bytes_per_pixel).ok_or(
            VulkanRendererError::InvalidMemoryUpload("wl_shm row byte count overflowed"),
        )?;
        if (data.stride as usize) < row_bytes {
            return Err(VulkanRendererError::InvalidMemoryUpload(
                "wl_shm stride is smaller than width * bytes_per_pixel",
            ));
        }

        let src_offset = usize::try_from(data.offset).map_err(|_| {
            VulkanRendererError::InvalidMemoryUpload("wl_shm offset could not be represented")
        })?;
        let src_stride = usize::try_from(data.stride).map_err(|_| {
            VulkanRendererError::InvalidMemoryUpload("wl_shm stride could not be represented")
        })?;
        let height = usize::try_from(data.height).map_err(|_| {
            VulkanRendererError::InvalidMemoryUpload("wl_shm height could not be represented")
        })?;

        let expected_len = src_offset
            .checked_add((height - 1).checked_mul(src_stride).ok_or(
                VulkanRendererError::InvalidMemoryUpload("wl_shm payload size overflowed"),
            )?)
            .and_then(|base| base.checked_add(row_bytes))
            .ok_or(VulkanRendererError::InvalidMemoryUpload(
                "wl_shm payload size overflowed",
            ))?;

        if len < expected_len {
            return Err(VulkanRendererError::InvalidMemoryUpload(
                "wl_shm payload is smaller than declared dimensions",
            ));
        }

        let mut packed = vec![
            0u8;
            row_bytes.checked_mul(height).ok_or(
                VulkanRendererError::InvalidMemoryUpload("packed wl_shm upload size overflowed"),
            )?
        ];

        for row in 0..height {
            let src_row_offset = src_offset + row * src_stride;
            let dst_row_offset = row * row_bytes;
            // SAFETY: Buffer bounds are validated above and we copy exactly `row_bytes` bytes per row.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ptr.add(src_row_offset),
                    packed.as_mut_ptr().add(dst_row_offset),
                    row_bytes,
                );
            }
        }

        f(&packed, format, Size::from((data.width, data.height)))
    })
    .map_err(|_| VulkanRendererError::TemporaryFailure("failed to access wl_shm buffer contents"))?
}

fn upload_region_to_image(
    device: &mut DeviceState,
    image: &std::sync::Arc<ImportedDmabufImage>,
    format: Fourcc,
    data: &[u8],
    region: Rectangle<i32, BufferCoord>,
) -> Result<(), VulkanRendererError> {
    trace!(?region, ?format, "recording upload-to-image copy");
    if region.is_empty() {
        return Ok(());
    }

    let bytes_per_pixel = bytes_per_pixel(format)?;
    let upload_width = usize::try_from(region.size.w)
        .map_err(|_| VulkanRendererError::InvalidMemoryUpload("upload width could not be represented"))?;
    let upload_height = usize::try_from(region.size.h)
        .map_err(|_| VulkanRendererError::InvalidMemoryUpload("upload height could not be represented"))?;
    let upload_row_bytes =
        upload_width
            .checked_mul(bytes_per_pixel)
            .ok_or(VulkanRendererError::InvalidMemoryUpload(
                "upload row byte count overflowed",
            ))?;
    let texture_width = usize::try_from(image.size().w)
        .map_err(|_| VulkanRendererError::InvalidMemoryUpload("texture width conversion failed"))?;
    let region_x = usize::try_from(region.loc.x)
        .map_err(|_| VulkanRendererError::InvalidMemoryUpload("region x conversion failed"))?;
    let region_y = usize::try_from(region.loc.y)
        .map_err(|_| VulkanRendererError::InvalidMemoryUpload("region y conversion failed"))?;
    let src_stride =
        texture_width
            .checked_mul(bytes_per_pixel)
            .ok_or(VulkanRendererError::InvalidMemoryUpload(
                "source stride overflowed",
            ))?;
    let src_offset = region_y
        .checked_mul(src_stride)
        .and_then(|base| base.checked_add(region_x.saturating_mul(bytes_per_pixel)))
        .ok_or(VulkanRendererError::InvalidMemoryUpload(
            "source upload offset overflowed",
        ))?;

    let copy_region = vk::BufferImageCopy::default()
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
        });

    device.queue_image_upload(
        std::sync::Arc::clone(image),
        super::device::ImageUpload {
            data,
            source_offset: src_offset,
            source_stride: src_stride,
            row_bytes: upload_row_bytes,
            rows: upload_height,
            region: copy_region,
        },
    )
}

fn validate_memory_format(format: Fourcc) -> Result<vk::Format, VulkanRendererError> {
    if !SUPPORTED_MEMORY_FORMATS.contains(&format) {
        return Err(VulkanRendererError::UnsupportedMemoryFormat(format));
    }

    crate::backend::allocator::vulkan::format::get_vk_format(format)
        .ok_or(VulkanRendererError::UnsupportedMemoryFormat(format))
}

fn bytes_per_pixel(format: Fourcc) -> Result<usize, VulkanRendererError> {
    let bits = get_bpp(format).ok_or(VulkanRendererError::UnsupportedMemoryFormat(format))?;
    if bits % 8 != 0 {
        return Err(VulkanRendererError::UnsupportedMemoryFormat(format));
    }

    let bytes = bits / 8;
    if bytes == 0 {
        return Err(VulkanRendererError::UnsupportedMemoryFormat(format));
    }

    Ok(bytes)
}

fn expected_len_for_size(format: Fourcc, size: Size<i32, BufferCoord>) -> Result<usize, VulkanRendererError> {
    if size.w <= 0 || size.h <= 0 {
        return Err(VulkanRendererError::InvalidMemoryUpload(
            "texture dimensions must be positive",
        ));
    }

    let width = usize::try_from(size.w)
        .map_err(|_| VulkanRendererError::InvalidMemoryUpload("texture width conversion failed"))?;
    let height = usize::try_from(size.h)
        .map_err(|_| VulkanRendererError::InvalidMemoryUpload("texture height conversion failed"))?;
    let bpp = bytes_per_pixel(format)?;

    width
        .checked_mul(height)
        .and_then(|pixel_count| pixel_count.checked_mul(bpp))
        .ok_or(VulkanRendererError::InvalidMemoryUpload(
            "texture byte size overflowed",
        ))
}

fn validate_region(
    texture_size: Size<i32, BufferCoord>,
    region: Rectangle<i32, BufferCoord>,
) -> Result<(), VulkanRendererError> {
    if region.loc.x < 0 || region.loc.y < 0 || region.size.w <= 0 || region.size.h <= 0 {
        return Err(VulkanRendererError::InvalidMemoryUpload(
            "region must be non-empty and fully non-negative",
        ));
    }

    let end_x = region
        .loc
        .x
        .checked_add(region.size.w)
        .ok_or(VulkanRendererError::InvalidMemoryUpload(
            "region x extent overflowed",
        ))?;
    let end_y = region
        .loc
        .y
        .checked_add(region.size.h)
        .ok_or(VulkanRendererError::InvalidMemoryUpload(
            "region y extent overflowed",
        ))?;

    if end_x > texture_size.w || end_y > texture_size.h {
        return Err(VulkanRendererError::InvalidMemoryUpload(
            "region exceeds texture bounds",
        ));
    }

    Ok(())
}

fn create_upload_image(
    device: &DeviceState,
    import_id: u64,
    format: Fourcc,
    size: Size<i32, BufferCoord>,
    y_inverted: bool,
) -> Result<std::sync::Arc<ImportedDmabufImage>, VulkanRendererError> {
    let vk_format = validate_memory_format(format)?;
    let device_handle = device.shared_device();
    let vk_device = device_handle.handle();
    let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;

    let create_info = vk::ImageCreateInfo::default()
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
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    // SAFETY: Device is valid and image create info references live data for the duration of the call.
    let image = match device_handle.observe_result(unsafe { vk_device.create_image(&create_info, None) }) {
        Ok(image) => image,
        Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => {
            return Err(VulkanRendererError::UnsupportedMemoryFormat(format))
        }
        Err(err) => return Err(err.into()),
    };

    // SAFETY: Image belongs to this device and remains valid until explicitly destroyed below.
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
                // SAFETY: Image belongs to this device and allocation failed before binding.
                device_handle.destroy_with(|vk_device| unsafe { vk_device.destroy_image(image, None) });
                return Err(err.into());
            }
        };

    // SAFETY: Image and memory belong to this device and memory offset 0 is valid for the allocation.
    if let Err(err) = device_handle.observe_result(unsafe { vk_device.bind_image_memory(image, memory, 0) }) {
        // SAFETY: Both handles belong to this device and were created above.
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

    // SAFETY: Image view create info references a live image handle for this device.
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
        .format(vk_format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );

    // SAFETY: Image view create info references a live image handle for this device.
    let render_view =
        match device_handle.observe_result(unsafe { vk_device.create_image_view(&render_view_info, None) }) {
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

    Ok(std::sync::Arc::new(ImportedDmabufImage::new(
        import_id,
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
        // shm buffers and CPU-rasterized chrome are premultiplied in electrical values,
        // exactly like client DMA-BUFs. These images are SAMPLED-only, so they keep a
        // storage-format render view: it is never usable as a colour attachment.
        ColorEncoding::ElectricalPremultiplied,
        usage,
        y_inverted,
        vk::ImageLayout::UNDEFINED,
        device.shared_device(),
    )))
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

#[cfg(test)]
mod tests {
    use crate::{
        backend::{
            allocator::{
                dmabuf::AsDmabuf,
                vulkan::{ImageUsageFlags, VulkanAllocator},
                Allocator,
            },
            renderer::{Color32F, Frame, ImportMem, MemoryUploadCapacityEdge, Renderer},
            vulkan::{version::Version, Instance, PhysicalDevice},
        },
        utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform},
    };

    use super::VulkanRenderer;

    fn init_renderer_and_allocator() -> Option<(VulkanRenderer, VulkanAllocator)> {
        let instance = Instance::new(Version::VERSION_1_3, None).ok()?;
        let physical_device = PhysicalDevice::enumerate(&instance).ok()?.next()?;
        let renderer = VulkanRenderer::new(&physical_device).ok()?;
        let allocator = VulkanAllocator::new(
            &physical_device,
            ImageUsageFlags::SAMPLED | ImageUsageFlags::COLOR_ATTACHMENT,
        )
        .ok()?;
        Some((renderer, allocator))
    }

    #[test]
    fn import_memory_update_and_render_path_succeeds() {
        let Some((mut renderer, mut allocator)) = init_renderer_and_allocator() else {
            return;
        };

        let format = renderer
            .mem_formats()
            .next()
            .expect("phase-7 must expose mem formats");
        let size: Size<i32, BufferCoord> = Size::from((16, 16));
        let data = vec![255u8; (size.w * size.h * 4) as usize];

        let texture = match renderer.import_memory(&data, format, size, false) {
            Ok(texture) => texture,
            Err(_) => return,
        };

        let damage = Rectangle::<i32, BufferCoord>::new((4, 4).into(), Size::from((8, 8)));
        renderer
            .update_memory(&texture, &data, damage)
            .expect("partial memory update should succeed for in-bounds regions");

        let out_of_bounds = Rectangle::<i32, BufferCoord>::new((12, 12).into(), Size::from((8, 8)));
        assert!(
            renderer.update_memory(&texture, &data, out_of_bounds).is_err(),
            "out-of-bounds region must be rejected by update_memory",
        );

        let Some(target_format) = renderer
            .dmabuf_render_formats()
            .iter()
            .copied()
            .find(|candidate| renderer.has_dmabuf_import_format(*candidate))
        else {
            return;
        };

        let buffer = match allocator.create_buffer(32, 32, target_format.code, &[target_format.modifier]) {
            Ok(buffer) => buffer,
            Err(_) => return,
        };
        let dmabuf = match buffer.export() {
            Ok(dmabuf) => dmabuf,
            Err(_) => return,
        };
        let mut target = match renderer.bind_dmabuf_target(&dmabuf) {
            Ok(target) => target,
            Err(_) => return,
        };

        let before_submit = renderer.diagnostics();
        assert_eq!(before_submit.uploads.pending_operations, 2);
        assert_eq!(before_submit.uploads.submitted_batches, 0);
        assert_eq!(before_submit.submissions.total_submissions, 0);

        let mut frame = match renderer.render(&mut target, Size::from((32, 32)), Transform::Normal) {
            Ok(frame) => frame,
            Err(_) => return,
        };
        let full = Rectangle::<i32, Physical>::from_size(Size::from((32, 32)));
        frame
            .clear(Color32F::new(0.0, 0.0, 0.0, 1.0), &[full])
            .expect("clear should succeed");
        frame
            .render_texture_from_to(
                &texture,
                Rectangle::new((0.0, 0.0).into(), (size.w as f64, size.h as f64).into()),
                Rectangle::new((0, 0).into(), Size::from((16, 16))),
                &[Rectangle::new((0, 0).into(), Size::from((16, 16)))],
                &[],
                Transform::Normal,
                1.0,
            )
            .expect("rendering memory-backed texture should succeed");

        let sync = frame.finish().expect("finish should submit successfully");
        let capacity_edge = match renderer
            .memory_upload_capacity_edge()
            .expect("submitted upload capacity must expose its exact completion edge")
        {
            MemoryUploadCapacityEdge::InFlight(edge) => edge,
            other => panic!("render-submitted uploads must remain attributable: {other:?}"),
        };
        let _ = sync.wait();
        let _ = capacity_edge.wait();
        let after_submit = renderer.diagnostics();
        assert_eq!(after_submit.uploads.pending_operations, 0);
        assert_eq!(after_submit.uploads.submitted_batches, 1);
        assert_eq!(after_submit.uploads.submitted_operations, 2);
        assert_eq!(after_submit.submissions.total_submissions, 1);
    }
}
