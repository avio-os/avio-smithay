use std::sync::Arc;

use ash::vk;
use tracing::{instrument, trace};

use crate::{
    backend::{
        allocator::{format::get_bpp, Format, Fourcc, Modifier},
        renderer::{
            ImportMem, MemoryRowUpload, MemoryUploadCapacityEdge, MemoryUploadErrorKind, StagedMemoryRows,
            StagedMemoryUpdate, Texture,
        },
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
    allocation::VulkanAllocationReason,
    device::{DeviceHandle, DeviceState},
    format::{optimal_tiling_features, texture_view_components, ColorEncoding},
    image::VulkanImage,
    staging::StagingReservation,
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

        Ok(VulkanTexture::from_renderer_image(
            image, size, format, flipped, true,
        ))
    }

    /// Admit a complete initial generation before allocating its image, then
    /// expose its exclusive mapped rows to a guarded source copy.
    pub(crate) fn stage_memory_import(
        &mut self,
        device: &mut DeviceState,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
        flipped: bool,
    ) -> Result<(VulkanTexture, StagedMemoryUpdate, StagedMemoryRows), VulkanRendererError> {
        validate_memory_format(format)?;
        let len = expected_len_for_size(format, size)?;
        // Dimension and format validation above proves both values. Derive
        // them before reserving so no fallible arithmetic can leak capacity.
        let rows = usize::try_from(size.h)
            .map_err(|_| VulkanRendererError::InvalidMemoryUpload("invalid row count"))?;
        let row_bytes = len / rows;
        // Admit the complete generation before creating its image. Capacity
        // pressure cannot create and immediately discard a full-size texture.
        let (reservation, ptr, memory) = device.reserve_staged_upload(len)?;
        let image = match create_upload_image(device, self.next_upload_id(), format, size, flipped) {
            Ok(image) => image,
            Err(error) => {
                device.release_staged_image_upload(reservation);
                return Err(error);
            }
        };
        let region = Rectangle::from_size(size);
        let ticket = self.next_upload_id();
        let staged = VulkanStagedUpdate {
            device: Arc::as_ptr(&device.shared_device()),
            image: Arc::clone(&image),
            reservation,
            copy: buffer_image_copy(region),
        };
        // SAFETY: The whole-generation reservation bounds these rows. Only
        // this exclusive writer can access them; `memory` owns the mapping.
        let rows = unsafe { StagedMemoryRows::new(ptr, row_bytes, rows, ticket, memory) };
        Ok((
            VulkanTexture::from_renderer_image(image, size, format, flipped, true),
            StagedMemoryUpdate::new(ticket, region, Box::new(staged)),
            rows,
        ))
    }

    pub(crate) fn import_memory_rows(
        &mut self,
        device: &mut DeviceState,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
        flipped: bool,
        fill: &mut dyn FnMut(&mut StagedMemoryRows) -> bool,
    ) -> Result<MemoryRowUpload<VulkanTexture>, VulkanRendererError> {
        validate_memory_format(format)?;
        let len = expected_len_for_size(format, size)?;
        let rows = size.h as usize;
        let (reservation, ptr, memory) = device.reserve_staged_upload(len)?;
        // SAFETY: The complete reservation bounds these exclusive rows.
        let rows = unsafe { StagedMemoryRows::new(ptr, len / rows, rows, 0, memory) };
        if !fill_reserved_rows(device, reservation, rows, fill) {
            return Ok(MemoryRowUpload::SourceFailed);
        }
        let image = match create_upload_image(device, self.next_upload_id(), format, size, flipped) {
            Ok(image) => image,
            Err(error) => {
                device.release_staged_image_upload(reservation);
                return Err(error);
            }
        };
        let texture = VulkanTexture::from_renderer_image(Arc::clone(&image), size, format, flipped, true);
        device.queue_staged_image_upload(
            image,
            reservation,
            buffer_image_copy(Rectangle::from_size(size)),
        )?;
        Ok(MemoryRowUpload::Queued(texture))
    }

    pub(crate) fn update_memory_rows(
        &mut self,
        device: &mut DeviceState,
        texture: &VulkanTexture,
        region: Rectangle<i32, BufferCoord>,
        fill: &mut dyn FnMut(&mut StagedMemoryRows) -> bool,
    ) -> Result<MemoryRowUpload<()>, VulkanRendererError> {
        let (image, format) = writable_memory_image(texture)?;
        validate_region(texture.size(), region)?;
        let rows = region.size.h as usize;
        let len = expected_len_for_size(format, region.size)?;
        let row_bytes = len / rows;
        let (reservation, ptr, memory) = device.stage_image_upload(image, len)?;
        // SAFETY: The region reservation bounds these exclusive rows.
        let rows = unsafe { StagedMemoryRows::new(ptr, row_bytes, rows, 0, memory) };
        if !fill_reserved_rows(device, reservation, rows, fill) {
            return Ok(MemoryRowUpload::SourceFailed);
        }
        device.queue_staged_image_upload(Arc::clone(image), reservation, buffer_image_copy(region))?;
        Ok(MemoryRowUpload::Queued(()))
    }

    /// Reserve staging for `region` of `texture`, to be written off-thread.
    pub(crate) fn stage_memory_update(
        &mut self,
        device: &mut DeviceState,
        texture: &VulkanTexture,
        region: Rectangle<i32, BufferCoord>,
    ) -> Result<(StagedMemoryUpdate, StagedMemoryRows), VulkanRendererError> {
        let (image, format) = writable_memory_image(texture)?;
        validate_region(texture.size(), region)?;
        let row_bytes = usize::try_from(region.size.w)
            .ok()
            .and_then(|width| width.checked_mul(bytes_per_pixel(format).ok()?))
            .ok_or(VulkanRendererError::InvalidMemoryUpload(
                "staged row size overflowed",
            ))?;
        let rows = usize::try_from(region.size.h)
            .map_err(|_| VulkanRendererError::InvalidMemoryUpload("staged row count overflowed"))?;
        let len = row_bytes
            .checked_mul(rows)
            .ok_or(VulkanRendererError::InvalidMemoryUpload(
                "staged byte count overflowed",
            ))?;
        let (reservation, ptr, memory) = device.stage_image_upload(image, len)?;
        let ticket = self.next_upload_id();
        let staged = VulkanStagedUpdate {
            device: Arc::as_ptr(&device.shared_device()),
            image: Arc::clone(image),
            reservation,
            copy: buffer_image_copy(region),
        };
        // SAFETY: The reservation is `len` bytes of the chunk's persistent
        // mapping, owned by this update until it is submitted or cancelled;
        // `memory` keeps that mapping alive wherever the rows go.
        let rows = unsafe { StagedMemoryRows::new(ptr, row_bytes, rows, ticket, memory) };
        Ok((StagedMemoryUpdate::new(ticket, region, Box::new(staged)), rows))
    }

    fn staged_record(
        device: &DeviceState,
        update: StagedMemoryUpdate,
    ) -> Result<VulkanStagedUpdate, VulkanRendererError> {
        let staged = update
            .into_inner()
            .downcast::<VulkanStagedUpdate>()
            .map_err(|_| {
                VulkanRendererError::InvalidMemoryUpload("staged update was not staged by a Vulkan renderer")
            })?;
        if staged.device != Arc::as_ptr(&device.shared_device()) {
            return Err(VulkanRendererError::InvalidMemoryUpload(
                "staged update belongs to another Vulkan renderer",
            ));
        }
        Ok(*staged)
    }

    /// Queue a staged update whose rows are written.
    pub(crate) fn submit_staged_memory_update(
        &mut self,
        device: &mut DeviceState,
        update: StagedMemoryUpdate,
        rows: StagedMemoryRows,
    ) -> Result<(), VulkanRendererError> {
        if rows.ticket() != update.ticket() {
            return Err(VulkanRendererError::InvalidMemoryUpload(
                "staged rows belong to another update",
            ));
        }
        let staged = Self::staged_record(device, update)?;
        // The rows are back: no other thread writes the reservation any more.
        drop(rows);
        device.queue_staged_image_upload(staged.image, staged.reservation, staged.copy)
    }

    /// Release a staged update's reservation.
    pub(crate) fn cancel_staged_memory_update(
        &mut self,
        device: &mut DeviceState,
        update: StagedMemoryUpdate,
    ) {
        if let Ok(staged) = Self::staged_record(device, update) {
            device.release_staged_image_upload(staged.reservation);
        }
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

        let Some(image) = texture.image_resource() else {
            return Err(VulkanRendererError::InvalidMemoryUpload(
                "memory-backed texture image is missing",
            ));
        };

        upload_region_to_image(device, image, format, data, region)
    }
}

fn fill_reserved_rows(
    device: &mut DeviceState,
    reservation: StagingReservation,
    mut rows: StagedMemoryRows,
    fill: &mut dyn FnMut(&mut StagedMemoryRows) -> bool,
) -> bool {
    let copied = fill(&mut rows);
    drop(rows);
    if !copied {
        device.release_staged_image_upload(reservation);
        return false;
    }
    true
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

    fn stage_memory_update(
        &mut self,
        texture: &Self::TextureId,
        region: Rectangle<i32, BufferCoord>,
    ) -> Result<Option<(StagedMemoryUpdate, StagedMemoryRows)>, Self::Error> {
        self.upload
            .stage_memory_update(&mut self.device, texture, region)
            .map(Some)
    }

    fn stage_memory_import(
        &mut self,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
        flipped: bool,
    ) -> Result<Option<(Self::TextureId, StagedMemoryUpdate, StagedMemoryRows)>, Self::Error> {
        self.upload
            .stage_memory_import(&mut self.device, format, size, flipped)
            .map(Some)
    }

    fn import_memory_rows(
        &mut self,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
        flipped: bool,
        fill: &mut dyn FnMut(&mut StagedMemoryRows) -> bool,
    ) -> Result<MemoryRowUpload<Self::TextureId>, Self::Error> {
        self.upload
            .import_memory_rows(&mut self.device, format, size, flipped, fill)
    }

    fn update_memory_rows(
        &mut self,
        texture: &Self::TextureId,
        region: Rectangle<i32, BufferCoord>,
        fill: &mut dyn FnMut(&mut StagedMemoryRows) -> bool,
    ) -> Result<MemoryRowUpload<()>, Self::Error> {
        self.upload
            .update_memory_rows(&mut self.device, texture, region, fill)
    }

    fn submit_staged_memory_update(
        &mut self,
        update: StagedMemoryUpdate,
        rows: StagedMemoryRows,
    ) -> Result<(), Self::Error> {
        self.upload
            .submit_staged_memory_update(&mut self.device, update, rows)
    }

    fn cancel_staged_memory_update(&mut self, update: StagedMemoryUpdate) {
        self.upload.cancel_staged_memory_update(&mut self.device, update);
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

/// A Vulkan renderer's record of one staged update.
struct VulkanStagedUpdate {
    /// Identity of the staging device; checked on submit and cancel.
    device: *const DeviceHandle,
    image: Arc<VulkanImage>,
    reservation: StagingReservation,
    copy: vk::BufferImageCopy,
}

// SAFETY: `device` is only compared as an identity, never dereferenced; the
// rest are owned handles the renderer's own records already move between threads.
unsafe impl Send for VulkanStagedUpdate {}

fn writable_memory_image(
    texture: &VulkanTexture,
) -> Result<(&Arc<VulkanImage>, Fourcc), VulkanRendererError> {
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
    let Some(image) = texture.image_resource() else {
        return Err(VulkanRendererError::InvalidMemoryUpload(
            "memory-backed texture image is missing",
        ));
    };
    Ok((image, format))
}

fn buffer_image_copy(region: Rectangle<i32, BufferCoord>) -> vk::BufferImageCopy {
    vk::BufferImageCopy::default()
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
        })
}

fn upload_region_to_image(
    device: &mut DeviceState,
    image: &std::sync::Arc<VulkanImage>,
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
) -> Result<std::sync::Arc<VulkanImage>, VulkanRendererError> {
    let vk_format = validate_memory_format(format)?;
    let format_features = optimal_tiling_features(device.physical_device(), vk_format);
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

    let allocation = device_handle
        .allocation_ledger()
        .record(VulkanAllocationReason::Texture, memory_requirements.size);

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

    Ok(std::sync::Arc::new(VulkanImage::new_renderer_local(
        import_id,
        image,
        memory,
        allocation,
        sampled_view,
        render_view,
        size,
        Format {
            code: format,
            modifier: Modifier::Invalid,
        },
        vk_format,
        format_features,
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

    fn read_pixel(renderer: &mut VulkanRenderer, texture: &super::VulkanTexture, x: i32, y: i32) -> [u8; 4] {
        use crate::backend::renderer::ExportMem;
        let mapping = renderer
            .copy_texture(
                texture,
                Rectangle::new((x, y).into(), Size::from((1, 1))),
                crate::backend::allocator::Fourcc::Argb8888,
            )
            .expect("texture readback");
        let bytes = renderer.map_texture(&mapping).expect("mapped readback");
        [bytes[0], bytes[1], bytes[2], bytes[3]]
    }

    /// Rows written on another thread land atomically at the next
    /// submission after they are handed back; until then the texture keeps
    /// its pixels. Cancelling releases the reservation.
    #[test]
    fn staged_memory_update_lands_rows_written_on_another_thread() {
        let Some((mut renderer, _allocator)) = init_renderer_and_allocator() else {
            return;
        };
        let format = crate::backend::allocator::Fourcc::Argb8888;
        let size: Size<i32, BufferCoord> = Size::from((16, 16));
        let texture = match renderer.import_memory(&vec![0u8; 16 * 16 * 4], format, size, false) {
            Ok(texture) => texture,
            Err(_) => return,
        };
        let region = Rectangle::<i32, BufferCoord>::new((4, 4).into(), Size::from((8, 8)));
        let (update, rows) = renderer
            .stage_memory_update(&texture, region)
            .expect("in-bounds staged update")
            .expect("the Vulkan renderer stages");
        assert_eq!((rows.rows(), rows.row_bytes()), (8, 32));
        assert_eq!(update.region(), region);

        let rows = std::thread::spawn(move || {
            let mut rows = rows;
            for row in 0..rows.rows() {
                rows.row_mut(row).fill(0xff);
            }
            rows
        })
        .join()
        .expect("writer thread");
        assert_eq!(
            read_pixel(&mut renderer, &texture, 5, 5),
            [0; 4],
            "not yet submitted"
        );

        renderer
            .submit_staged_memory_update(update, rows)
            .expect("staged update queued");
        assert_eq!(read_pixel(&mut renderer, &texture, 5, 5), [0xff; 4]);
        assert_eq!(read_pixel(&mut renderer, &texture, 11, 11), [0xff; 4]);
        assert_eq!(read_pixel(&mut renderer, &texture, 3, 3), [0; 4]);
        assert_eq!(read_pixel(&mut renderer, &texture, 12, 12), [0; 4]);

        let in_use = renderer.diagnostics().uploads.arena_in_use_bytes;
        let (update, _rows) = renderer
            .stage_memory_update(&texture, region)
            .expect("in-bounds staged update")
            .expect("the Vulkan renderer stages");
        assert!(renderer.diagnostics().uploads.arena_in_use_bytes > in_use);
        renderer.cancel_staged_memory_update(update);
        assert_eq!(renderer.diagnostics().uploads.arena_in_use_bytes, in_use);

        let outside = Rectangle::<i32, BufferCoord>::new((12, 12).into(), Size::from((8, 8)));
        assert!(renderer.stage_memory_update(&texture, outside).is_err());
    }

    /// A queued upload whose texture is gone before any submission is
    /// dropped: a run of replaced images (a cursor bitmap changing while an
    /// output scans out without composition) holds neither batch slots nor
    /// staging, and never reaches the GPU. A live texture keeps its upload.
    #[test]
    fn an_upload_no_texture_can_sample_is_dropped_before_submission() {
        let Some((mut renderer, _allocator)) = init_renderer_and_allocator() else {
            return;
        };
        let format = crate::backend::allocator::Fourcc::Argb8888;
        let size: Size<i32, BufferCoord> = Size::from((16, 16));
        let bytes = 16 * 16 * 4;
        let in_use = renderer.diagnostics().uploads.arena_in_use_bytes;
        // More than one batch (256 operations): before the drop, import 257
        // failed with UploadBatchFull, with every replaced image allocated.
        for shade in 0..300_u32 {
            let texture = renderer
                .import_memory(&vec![shade as u8; bytes], format, size, false)
                .expect("a replaced upload frees its batch slot");
            drop(texture);
        }
        // Only dead work is pending: nothing is submitted for it.
        assert!(!matches!(
            renderer.memory_upload_capacity_edge(),
            Ok(MemoryUploadCapacityEdge::Submitted(_))
        ));
        let uploads = renderer.diagnostics().uploads;
        assert_eq!(uploads.pending_operations, 0);
        assert_eq!(uploads.arena_in_use_bytes, in_use);
        assert_eq!(renderer.diagnostics().submissions.total_submissions, 0);

        let live = renderer
            .import_memory(&vec![0xff; bytes], format, size, false)
            .expect("live import");
        let held = renderer
            .import_memory(&vec![0x40; bytes], format, size, false)
            .expect("second live import");
        let held_image = held.image_resource().expect("upload image").clone();
        drop(held);
        let replaced = renderer
            .import_memory(&vec![0x80; bytes], format, size, false)
            .expect("replaced import");
        drop(replaced);
        // The submission carries the live uploads, including the one whose
        // image is still held without a texture, and not the replaced one.
        assert_eq!(read_pixel(&mut renderer, &live, 3, 3), [0xff; 4]);
        let uploads = renderer.diagnostics().uploads;
        assert_eq!(uploads.pending_operations, 0);
        assert_eq!(uploads.submitted_batches, 1);
        assert_eq!(uploads.submitted_operations, 2);
        drop(held_image);
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
        let upload_image = texture
            .image_resource()
            .expect("memory upload must create an image resource")
            .clone();
        assert!(upload_image.is_renderer_local());
        assert!(!upload_image.is_owned_by_foreign());

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
        assert!(!upload_image.is_owned_by_foreign());
    }
}
