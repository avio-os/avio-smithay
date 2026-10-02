//! Cloneable renderer-origin offscreen allocation, without queue or command-pool access.
use super::{
    allocation::VulkanAllocationReason,
    device::DeviceState,
    device_handle::DeviceHandle,
    format::{
        optimal_tiling_features, render_view_format, srgb_view_format_list, texture_view_components,
        ColorEncoding,
    },
    image::VulkanImage,
    readback::format_supports_offscreen_usage,
    VulkanRendererError, VulkanTexture,
};
use crate::{
    backend::{
        allocator::{Format, Fourcc, Modifier},
        vulkan::PhysicalDevice,
    },
    utils::{Buffer as BufferCoord, Size},
};
use ash::vk;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tracing::{instrument, trace};

/// Creates native images on its originating renderer's existing Vulkan device.
/// Clones share the same resource-id namespace and device retirement owner.
/// Allocation helpers may use this handle without accessing renderer queues;
/// returned images still require normal ordered bind/initialization before use.
#[derive(Clone, Debug)]
pub struct VulkanOffscreenAllocator {
    device: Arc<DeviceHandle>,
    physical_device: PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    ids: Arc<AtomicU64>,
}
impl VulkanOffscreenAllocator {
    pub(super) fn new(device: &DeviceState, ids: Arc<AtomicU64>) -> Self {
        let physical_device = device.physical_device().clone();
        let memory_properties = unsafe {
            physical_device
                .instance()
                .handle()
                .get_physical_device_memory_properties(physical_device.handle())
        };
        Self {
            device: device.shared_device(),
            physical_device,
            memory_properties,
            ids,
        }
    }
    /// Explicitly tag a helper's allocation turn without overriding a legacy
    /// frame caller's phase. Scope belongs to this calling thread only.
    pub fn allocation_phase_scope(
        &self,
        phase: super::VulkanAllocationPhase,
    ) -> super::VulkanAllocationPhaseGuard {
        self.device.allocation_ledger().enter_phase(phase)
    }

    /// Allocate/map the sole fixed upload arena off the queue-owner thread.
    pub fn prepare_upload_storage(
        &self,
        bytes: usize,
    ) -> Result<super::VulkanUploadStorage, VulkanRendererError> {
        super::VulkanUploadStorage::prepare(&self.physical_device, self.device.clone(), bytes)
    }

    /// Allocate a native colour attachment/sampleable image without submitting work.
    #[instrument(level = "trace", skip(self))]
    #[profiling::function]
    pub fn create_buffer(
        &self,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
    ) -> Result<VulkanTexture, VulkanRendererError> {
        self.create_image(format, size, false)
    }

    /// Allocate a sampled image whose immutable encoded pixels will be
    /// uploaded later through the renderer's already provisioned upload ring.
    /// Call on an owning off-frame provisioning turn, never during a draw.
    pub fn create_memory_buffer(
        &self,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
    ) -> Result<VulkanTexture, VulkanRendererError> {
        self.create_image(format, size, true)
    }

    fn create_image(
        &self,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
        memory_writable: bool,
    ) -> Result<VulkanTexture, VulkanRendererError> {
        trace!(?format, ?size, memory_writable, "creating vulkan native image");
        if size.w <= 0 || size.h <= 0 {
            return Err(VulkanRendererError::TemporaryFailure(
                "offscreen buffer dimensions must be positive",
            ));
        }

        let id = self
            .ids
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |next| {
                next.checked_add(1).filter(|next| *next < (1u64 << 62))
            })
            .map_err(|_| VulkanRendererError::TemporaryFailure("offscreen image namespace exhausted"))?;
        let vk_format = crate::backend::allocator::vulkan::format::get_vk_format(format)
            .ok_or(VulkanRendererError::UnsupportedMemoryFormat(format))?;
        let format_features = optimal_tiling_features(&self.physical_device, vk_format);
        if !format_supports_offscreen_usage(format_features) {
            return Err(VulkanRendererError::UnsupportedMemoryFormat(format));
        }

        let device_handle = self.device.clone();
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
        let image = device_handle.observe_result(crate::backend::allocator::observe_gpu_allocation(
            unsafe { vk_device.create_image(&create_info, None) },
            crate::backend::allocator::GpuAllocationKind::VulkanImage,
        ))?;
        // SAFETY: Image belongs to this device and remains valid until explicit destruction.
        let memory_requirements = unsafe { vk_device.get_image_memory_requirements(image) };
        let Some(memory_type_index) =
            pick_image_memory_type(&self.memory_properties, memory_requirements.memory_type_bits)
        else {
            device_handle.destroy_with(|vk_device| unsafe { vk_device.destroy_image(image, None) });
            return Err(VulkanRendererError::NoCompatibleMemoryType);
        };

        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);
        // SAFETY: Device is valid and allocation info references live memory.
        let memory = match device_handle.observe_result(crate::backend::allocator::observe_gpu_allocation(
            unsafe { vk_device.allocate_memory(&allocate_info, None) },
            crate::backend::allocator::GpuAllocationKind::VulkanDeviceMemory,
        )) {
            Ok(memory) => memory,
            Err(err) => {
                // SAFETY: Image belongs to this device and has not been bound.
                device_handle.destroy_with(|vk_device| unsafe { vk_device.destroy_image(image, None) });
                return Err(err.into());
            }
        };

        let allocation = device_handle
            .allocation_ledger()
            .record(VulkanAllocationReason::RenderTarget, memory_requirements.size);

        // SAFETY: Image and memory belong to this device and offset 0 is valid.
        if let Err(err) =
            device_handle.observe_result(unsafe { vk_device.bind_image_memory(image, memory, 0) })
        {
            // SAFETY: Handles belong to this device and were created above.
            device_handle.destroy_with(|vk_device| unsafe {
                vk_device.destroy_image(image, None);
                vk_device.free_memory(memory, None);
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
                    vk_device.destroy_image(image, None);
                    vk_device.free_memory(memory, None);
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
                    vk_device.destroy_image(image, None);
                    vk_device.free_memory(memory, None);
                });
                return Err(err.into());
            }
        };

        let imported = Arc::new(VulkanImage::new_renderer_local(
            id,
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
            // Offscreens are filled by our own linear-blending render passes, so their
            // stored bytes are the sRGB encoding of a premultiplied *linear* value.
            if memory_writable {
                ColorEncoding::ElectricalPremultiplied
            } else {
                ColorEncoding::LinearPremultiplied
            },
            usage,
            false,
            vk::ImageLayout::UNDEFINED,
            self.device.clone(),
        )?);

        Ok(VulkanTexture::from_renderer_image(
            imported,
            size,
            format,
            false,
            memory_writable,
        ))
    }
}

fn pick_image_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    memory_type_bits: u32,
) -> Option<u32> {
    super::readback::pick_memory_type_index(
        properties,
        memory_type_bits,
        vk::MemoryPropertyFlags::empty(),
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .map(|(index, _)| index)
}
