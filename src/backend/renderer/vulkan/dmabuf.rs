use std::{
    os::fd::{AsRawFd, BorrowedFd, IntoRawFd},
    sync::Arc,
};

use ash::{khr, vk};
use indexmap::IndexMap;
use scopeguard::ScopeGuard;
use tracing::{info, trace};

use crate::{
    backend::allocator::{
        dmabuf::{Dmabuf, WeakDmabuf, MAX_PLANES},
        Buffer, Format, Modifier,
    },
    utils::{Buffer as BufferCoord, Size},
};

use super::{
    device::{DeviceHandle, DeviceState},
    format::{
        optimal_tiling_features, render_view_format, srgb_view_format_list, texture_view_components,
        ColorEncoding, FormatCapabilities, ModifierCapability,
    },
    image::VulkanImage,
    VulkanCacheStats, VulkanRendererError, VulkanTarget, VulkanTexture,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DmabufRole {
    Texture,
    RenderTarget,
    FramebufferEffectTarget,
    CaptureTarget,
}

impl DmabufRole {
    fn required_usage(self) -> vk::ImageUsageFlags {
        match self {
            DmabufRole::Texture => vk::ImageUsageFlags::SAMPLED,
            DmabufRole::RenderTarget => vk::ImageUsageFlags::COLOR_ATTACHMENT,
            DmabufRole::FramebufferEffectTarget => {
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC
            }
            DmabufRole::CaptureTarget => {
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
            }
        }
    }

    fn format_supported(self, formats: &FormatCapabilities, format: Format) -> bool {
        match self {
            DmabufRole::Texture => formats.has_import_format(format),
            DmabufRole::RenderTarget => formats.has_render_format(format),
            DmabufRole::FramebufferEffectTarget => formats.has_framebuffer_effect_format(format),
            DmabufRole::CaptureTarget => formats.has_capture_format(format),
        }
    }

    fn supports_implicit_modifier(
        self,
        formats: &FormatCapabilities,
        code: crate::backend::allocator::Fourcc,
    ) -> bool {
        match self {
            DmabufRole::Texture => formats.supports_implicit_import_modifier(code),
            DmabufRole::RenderTarget => formats.supports_implicit_render_modifier(code),
            DmabufRole::FramebufferEffectTarget => {
                formats.supports_implicit_framebuffer_effect_modifier(code)
            }
            DmabufRole::CaptureTarget => formats.supports_implicit_capture_modifier(code),
        }
    }

    fn supports_disjoint(self, capability: &ModifierCapability) -> bool {
        match self {
            DmabufRole::Texture => capability.supports_disjoint_import,
            DmabufRole::RenderTarget => capability.supports_disjoint_render,
            DmabufRole::FramebufferEffectTarget => capability.supports_disjoint_framebuffer_effect,
            DmabufRole::CaptureTarget => capability.supports_disjoint_capture,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DmabufSignature {
    size: Size<i32, BufferCoord>,
    format: Format,
    num_planes: usize,
    offsets: Vec<u32>,
    strides: Vec<u32>,
    disjoint: bool,
    y_inverted: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct CachedDmabuf {
    pub(crate) handle: WeakDmabuf,
    signature: DmabufSignature,
    imported: Arc<VulkanImage>,
}

#[derive(Debug, Clone)]
struct DmabufImportDescriptor {
    signature: DmabufSignature,
    vk_format: vk::Format,
    format_features: vk::FormatFeatureFlags,
}

#[derive(Debug, Default)]
pub(crate) struct DmabufState {
    cache: IndexMap<WeakDmabuf, CachedDmabuf>,
    next_import_id: u64,
    imports_since_cleanup: u32,
    import_attempts_total: u64,
    cleanup_runs: u64,
    cleanup_scanned: u64,
    cleanup_stale_evictions: u64,
    capacity_evictions: u64,
    max_cache_len: usize,
    cache_stats: VulkanCacheStats,
}

const MAX_DMABUF_CACHE_ENTRIES: usize = 256;
const DMABUF_CLEANUP_INTERVAL_IMPORTS: u32 = 64;
const DMABUF_CLEANUP_SCAN_LIMIT: usize = 64;
const DMABUF_DIAG_LOG_INTERVAL_IMPORTS: u64 = 256;

fn dmabuf_is_disjoint(dmabuf: &Dmabuf) -> Result<bool, VulkanRendererError> {
    let mut handles = dmabuf.handles();
    let first = handles
        .next()
        .ok_or(VulkanRendererError::InvalidDmabuf("dma-buf has no fds"))?;
    let first_stat = rustix::fs::fstat(first).map_err(std::io::Error::from)?;

    for handle in handles {
        let stat = rustix::fs::fstat(handle).map_err(std::io::Error::from)?;
        if (stat.st_dev, stat.st_ino) != (first_stat.st_dev, first_stat.st_ino) {
            return Ok(true);
        }
    }

    Ok(false)
}

impl DmabufState {
    pub(crate) fn import_texture(
        &mut self,
        device: &DeviceState,
        formats: &FormatCapabilities,
        dmabuf: &Dmabuf,
    ) -> Result<VulkanTexture, VulkanRendererError> {
        let imported = self.import_or_reuse(device, formats, dmabuf, DmabufRole::Texture)?;
        Ok(VulkanTexture::from_dmabuf_import(
            imported,
            dmabuf.size(),
            Some(dmabuf.format().code),
            dmabuf.y_inverted(),
        ))
    }

    pub(crate) fn bind_render_target(
        &mut self,
        device: &DeviceState,
        formats: &FormatCapabilities,
        dmabuf: &Dmabuf,
    ) -> Result<VulkanTarget, VulkanRendererError> {
        let imported = self.import_or_reuse(device, formats, dmabuf, DmabufRole::RenderTarget)?;
        Ok(VulkanTarget::from_image_resource(
            imported,
            dmabuf.size(),
            Some(dmabuf.format().code),
        ))
    }

    pub(crate) fn bind_framebuffer_effect_target(
        &mut self,
        device: &DeviceState,
        formats: &FormatCapabilities,
        dmabuf: &Dmabuf,
    ) -> Result<VulkanTarget, VulkanRendererError> {
        let imported = self.import_or_reuse(device, formats, dmabuf, DmabufRole::FramebufferEffectTarget)?;
        Ok(VulkanTarget::from_image_resource(
            imported,
            dmabuf.size(),
            Some(dmabuf.format().code),
        ))
    }

    pub(crate) fn bind_capture_target(
        &mut self,
        device: &DeviceState,
        formats: &FormatCapabilities,
        dmabuf: &Dmabuf,
    ) -> Result<VulkanTarget, VulkanRendererError> {
        let imported = self.import_or_reuse(device, formats, dmabuf, DmabufRole::CaptureTarget)?;
        Ok(VulkanTarget::from_image_resource(
            imported,
            dmabuf.size(),
            Some(dmabuf.format().code),
        ))
    }

    pub(crate) fn cleanup(&mut self) {
        self.cleanup_runs = self.cleanup_runs.saturating_add(1);
        self.cleanup_stale_entries(usize::MAX);
        self.evict_to_capacity();
    }

    pub(crate) fn cache_stats(&self) -> VulkanCacheStats {
        self.cache_stats
    }

    fn import_or_reuse(
        &mut self,
        device: &DeviceState,
        formats: &FormatCapabilities,
        dmabuf: &Dmabuf,
        role: DmabufRole,
    ) -> Result<Arc<VulkanImage>, VulkanRendererError> {
        self.import_attempts_total = self.import_attempts_total.saturating_add(1);
        self.maybe_cleanup();

        let requested_usage = role.required_usage();
        let key = dmabuf.weak();

        if let Some(cached) = self.cache.get(&key) {
            // Dmabuf plane metadata is immutable after construction and the weak key
            // identifies that exact allocation. Avoid repeating fstat topology checks
            // for every frame once this buffer and usage have been validated.
            if cached.imported.usage().contains(requested_usage) {
                let imported = cached.imported.clone();
                self.promote_entry(&key);
                self.cache_stats.hits = self.cache_stats.hits.saturating_add(1);
                trace!(
                    hits = self.cache_stats.hits,
                    misses = self.cache_stats.misses,
                    evictions = self.cache_stats.evictions,
                    "vulkan dmabuf cache hit"
                );
                self.maybe_log_cache_diagnostics();
                return Ok(imported);
            }
        }

        let descriptor = Self::validate_dmabuf(device, dmabuf, formats, role)?;
        self.cache_stats.misses = self.cache_stats.misses.saturating_add(1);
        let usage = self
            .cache
            .get(&key)
            .filter(|cached| cached.signature == descriptor.signature)
            .map(|cached| cached.imported.usage() | requested_usage)
            .unwrap_or(requested_usage);

        let imported = self.create_image_resource(device, dmabuf, &descriptor, usage)?;
        let _ = self.cache.shift_remove(&key);
        self.cache.insert(
            key.clone(),
            CachedDmabuf {
                handle: key,
                signature: descriptor.signature,
                imported: imported.clone(),
            },
        );
        self.update_max_cache_len();
        self.evict_to_capacity();
        trace!(
            hits = self.cache_stats.hits,
            misses = self.cache_stats.misses,
            evictions = self.cache_stats.evictions,
            "vulkan dmabuf cache miss"
        );
        self.maybe_log_cache_diagnostics();

        Ok(imported)
    }

    fn maybe_cleanup(&mut self) {
        self.imports_since_cleanup = self.imports_since_cleanup.saturating_add(1);
        let needs_cleanup = self.imports_since_cleanup >= DMABUF_CLEANUP_INTERVAL_IMPORTS
            || self.cache.len() > MAX_DMABUF_CACHE_ENTRIES;
        if !needs_cleanup {
            return;
        }

        self.imports_since_cleanup = 0;
        self.cleanup_runs = self.cleanup_runs.saturating_add(1);
        self.cleanup_stale_entries(DMABUF_CLEANUP_SCAN_LIMIT);
        self.evict_to_capacity();
    }

    fn cleanup_stale_entries(&mut self, max_scan: usize) {
        let mut scanned = 0usize;
        let mut index = 0usize;

        while index < self.cache.len() && scanned < max_scan {
            let remove = self
                .cache
                .get_index(index)
                .map(|(_, cached)| cached.handle.is_gone() && Arc::strong_count(&cached.imported) <= 1)
                .unwrap_or(false);
            scanned = scanned.saturating_add(1);

            if remove {
                let _ = self.cache.shift_remove_index(index);
                self.cache_stats.evictions = self.cache_stats.evictions.saturating_add(1);
                self.cleanup_stale_evictions = self.cleanup_stale_evictions.saturating_add(1);
            } else {
                index = index.saturating_add(1);
            }
        }
        self.cleanup_scanned = self.cleanup_scanned.saturating_add(scanned as u64);
    }

    fn evict_to_capacity(&mut self) {
        while self.cache.len() > MAX_DMABUF_CACHE_ENTRIES {
            if self.cache.shift_remove_index(0).is_none() {
                break;
            }
            self.cache_stats.evictions = self.cache_stats.evictions.saturating_add(1);
            self.capacity_evictions = self.capacity_evictions.saturating_add(1);
        }
    }

    fn promote_entry(&mut self, key: &WeakDmabuf) {
        let Some(entry) = self.cache.shift_remove(key) else {
            return;
        };
        self.cache.insert(key.clone(), entry);
    }

    fn update_max_cache_len(&mut self) {
        self.max_cache_len = self.max_cache_len.max(self.cache.len());
    }

    fn maybe_log_cache_diagnostics(&self) {
        if self.import_attempts_total == 0
            || (self.import_attempts_total % DMABUF_DIAG_LOG_INTERVAL_IMPORTS) != 0
        {
            return;
        }
        info!(
            imports_total = self.import_attempts_total,
            cache_len = self.cache.len(),
            cache_max_len = self.max_cache_len,
            hits = self.cache_stats.hits,
            misses = self.cache_stats.misses,
            evictions = self.cache_stats.evictions,
            cleanup_runs = self.cleanup_runs,
            cleanup_scanned = self.cleanup_scanned,
            cleanup_stale_evictions = self.cleanup_stale_evictions,
            capacity_evictions = self.capacity_evictions,
            "smithay vulkan dmabuf cache diagnostics"
        );
    }

    fn validate_dmabuf(
        device: &DeviceState,
        dmabuf: &Dmabuf,
        formats: &FormatCapabilities,
        role: DmabufRole,
    ) -> Result<DmabufImportDescriptor, VulkanRendererError> {
        let size = dmabuf.size();
        if size.w <= 0 || size.h <= 0 {
            return Err(VulkanRendererError::InvalidDmabuf(
                "dma-buf dimensions must be positive",
            ));
        }

        let format = dmabuf.format();
        if !role.format_supported(formats, format) {
            return Err(VulkanRendererError::UnsupportedDmabufFormat(format));
        }

        let num_planes = dmabuf.num_planes();
        if num_planes == 0 || num_planes > MAX_PLANES {
            return Err(VulkanRendererError::InvalidDmabuf(
                "dma-buf plane count is outside supported range",
            ));
        }

        let offsets = dmabuf.offsets().collect::<Vec<_>>();
        let strides = dmabuf.strides().collect::<Vec<_>>();
        if offsets.len() != num_planes || strides.len() != num_planes {
            return Err(VulkanRendererError::InvalidDmabuf(
                "dma-buf plane metadata is inconsistent",
            ));
        }

        if strides.contains(&0) {
            return Err(VulkanRendererError::InvalidDmabuf(
                "dma-buf stride must be non-zero for all planes",
            ));
        }

        if dmabuf.handles().count() != num_planes {
            return Err(VulkanRendererError::InvalidDmabuf(
                "dma-buf fd count does not match plane count",
            ));
        }
        let disjoint = dmabuf_is_disjoint(dmabuf)?;

        let Some(vk_format) = crate::backend::allocator::vulkan::format::get_vk_format(format.code) else {
            return Err(VulkanRendererError::UnsupportedDmabufFormat(format));
        };

        let format_features = if format.modifier == Modifier::Invalid {
            if !role.supports_implicit_modifier(formats, format.code) {
                return Err(VulkanRendererError::UnsupportedDmabufFormat(format));
            }
            if num_planes != 1 {
                return Err(VulkanRendererError::InvalidDmabuf(
                    "implicit-modifier dma-bufs must contain exactly one memory plane",
                ));
            }
            optimal_tiling_features(device.physical_device(), vk_format)
        } else {
            let modifier_caps = formats.modifier_capabilities(format.code);
            let Some(modifier_cap) = modifier_caps.iter().find(|cap| cap.modifier == format.modifier) else {
                return Err(VulkanRendererError::UnsupportedDmabufFormat(format));
            };

            if modifier_cap.drm_format_modifier_plane_count as usize != num_planes {
                return Err(VulkanRendererError::DmabufPlaneCountMismatch {
                    modifier: format.modifier,
                    expected: modifier_cap.drm_format_modifier_plane_count,
                    actual: num_planes,
                });
            }

            if disjoint && !role.supports_disjoint(modifier_cap) {
                return Err(VulkanRendererError::UnsupportedDmabufDisjoint);
            }
            modifier_cap.drm_format_modifier_tiling_features
        };

        Ok(DmabufImportDescriptor {
            signature: DmabufSignature {
                size,
                format,
                num_planes,
                offsets,
                strides,
                disjoint,
                y_inverted: dmabuf.y_inverted(),
            },
            vk_format,
            format_features,
        })
    }

    fn create_image_resource(
        &mut self,
        device: &DeviceState,
        dmabuf: &Dmabuf,
        descriptor: &DmabufImportDescriptor,
        usage: vk::ImageUsageFlags,
    ) -> Result<Arc<VulkanImage>, VulkanRendererError> {
        let device_handle = device.shared_device();
        let vk_device = device_handle.handle();
        let format = descriptor.signature.format;
        let size = descriptor.signature.size;
        let disjoint = descriptor.signature.disjoint;
        let external_memory_fd =
            khr::external_memory_fd::Device::new(device.physical_device().instance().handle(), vk_device);

        let mut external_memory_image_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let mut explicit_modifier_info;
        let plane_layouts;
        // The image keeps its encoded storage format; MUTABLE_FORMAT only lets the
        // colour attachment view reinterpret those same bytes as `_SRGB` so blending
        // happens in linear light. Storage, stride and DRM modifier are untouched.
        let view_formats = srgb_view_format_list(descriptor.vk_format);
        let mut format_list_info;
        let mut create_flags = if disjoint {
            vk::ImageCreateFlags::DISJOINT
        } else {
            vk::ImageCreateFlags::empty()
        };
        if view_formats.is_some() {
            create_flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
        }
        let mut image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(descriptor.vk_format)
            .extent(vk::Extent3D {
                width: size.w as u32,
                height: size.h as u32,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(usage)
            .flags(create_flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        if let Some(formats) = view_formats.as_ref() {
            format_list_info = vk::ImageFormatListCreateInfo::default().view_formats(formats);
            image_create_info = image_create_info.push_next(&mut format_list_info);
        }

        if format.modifier == Modifier::Invalid {
            image_create_info = image_create_info.tiling(vk::ImageTiling::OPTIMAL);
        } else {
            plane_layouts = descriptor
                .signature
                .offsets
                .iter()
                .zip(descriptor.signature.strides.iter())
                .map(|(offset, stride)| {
                    vk::SubresourceLayout::default()
                        .offset(*offset as u64)
                        .row_pitch(*stride as u64)
                })
                .collect::<Vec<_>>();

            explicit_modifier_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
                .drm_format_modifier(format.modifier.into())
                .plane_layouts(&plane_layouts);

            image_create_info = image_create_info
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .push_next(&mut explicit_modifier_info);
        }

        image_create_info = image_create_info.push_next(&mut external_memory_image_info);

        let image =
            match device_handle.observe_result(unsafe { vk_device.create_image(&image_create_info, None) }) {
                Ok(image) => image,
                Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => {
                    return Err(VulkanRendererError::UnsupportedDmabufFormat(format))
                }
                Err(err) => return Err(err.into()),
            };

        let handles = dmabuf.handles().collect::<Vec<_>>();
        let memory_count = if disjoint {
            descriptor.signature.num_planes
        } else {
            1
        };
        let mut memories = Vec::with_capacity(memory_count);

        for plane_index in 0..memory_count {
            let requirements =
                match Self::image_memory_requirements(vk_device, image, disjoint.then_some(plane_index)) {
                    Ok(requirements) => requirements,
                    Err(err) => {
                        Self::destroy_image_and_memories(&device_handle, image, &memories);
                        return Err(err);
                    }
                };
            let Some(fd) = handles.get(plane_index).copied() else {
                Self::destroy_image_and_memories(&device_handle, image, &memories);
                return Err(VulkanRendererError::InvalidDmabuf(
                    "dma-buf fd count does not match memory binding count",
                ));
            };

            match Self::allocate_imported_memory(&device_handle, &external_memory_fd, image, fd, requirements)
            {
                Ok(memory) => memories.push(memory),
                Err(err) => {
                    Self::destroy_image_and_memories(&device_handle, image, &memories);
                    return Err(err);
                }
            }
        }

        trace!(
            plane_count = descriptor.signature.num_planes,
            memory_bindings = memories.len(),
            disjoint,
            ?format,
            ?usage,
            "binding imported dma-buf memory to Vulkan image"
        );

        let bind_result = if disjoint {
            let mut plane_infos = match (0..memories.len())
                .map(|plane_index| {
                    Self::memory_plane_aspect(plane_index)
                        .map(|aspect| vk::BindImagePlaneMemoryInfo::default().plane_aspect(aspect))
                })
                .collect::<Result<Vec<_>, _>>()
            {
                Ok(plane_infos) => plane_infos,
                Err(err) => {
                    Self::destroy_image_and_memories(&device_handle, image, &memories);
                    return Err(err);
                }
            };
            let bind_infos = plane_infos
                .iter_mut()
                .zip(memories.iter().copied())
                .map(|(plane_info, memory)| {
                    vk::BindImageMemoryInfo::default()
                        .image(image)
                        .memory(memory)
                        .memory_offset(0)
                        .push_next(plane_info)
                })
                .collect::<Vec<_>>();
            device_handle.observe_result(unsafe { vk_device.bind_image_memory2(&bind_infos) })
        } else {
            device_handle.observe_result(unsafe { vk_device.bind_image_memory(image, memories[0], 0) })
        };

        if let Err(err) = bind_result {
            Self::destroy_image_and_memories(&device_handle, image, &memories);
            return Err(err.into());
        }

        let sampled_view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(descriptor.vk_format)
            .components(texture_view_components(format.code, usage))
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let sampled_view = match device_handle
            .observe_result(unsafe { vk_device.create_image_view(&sampled_view_info, None) })
        {
            Ok(view) => view,
            Err(err) => {
                Self::destroy_image_and_memories(&device_handle, image, &memories);
                return Err(err.into());
            }
        };

        let render_view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            // Linear-light blending: the hardware decodes the destination through this
            // view and re-encodes the blended result on store.
            .format(render_view_format(descriptor.vk_format))
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let render_view = match device_handle
            .observe_result(unsafe { vk_device.create_image_view(&render_view_info, None) })
        {
            Ok(view) => view,
            Err(err) => {
                device_handle.destroy_with(|vk_device| unsafe {
                    vk_device.destroy_image_view(sampled_view, None);
                });
                Self::destroy_image_and_memories(&device_handle, image, &memories);
                return Err(err.into());
            }
        };

        let import_id = self.next_import_id;
        self.next_import_id = self.next_import_id.wrapping_add(1);

        Ok(Arc::new(VulkanImage::new_external_dmabuf(
            import_id,
            image,
            memories,
            sampled_view,
            render_view,
            size,
            format,
            descriptor.vk_format,
            descriptor.format_features,
            // Imported buffers are authored outside the compositor: Wayland clients and
            // the Flutter shell both premultiply in electrical values.
            ColorEncoding::ElectricalPremultiplied,
            usage,
            descriptor.signature.y_inverted,
            vk::ImageLayout::UNDEFINED,
            device_handle,
        )))
    }

    fn image_memory_requirements(
        device: &ash::Device,
        image: vk::Image,
        plane_index: Option<usize>,
    ) -> Result<vk::MemoryRequirements, VulkanRendererError> {
        let Some(plane_index) = plane_index else {
            return Ok(unsafe { device.get_image_memory_requirements(image) });
        };

        let mut plane_info = vk::ImagePlaneMemoryRequirementsInfo::default()
            .plane_aspect(Self::memory_plane_aspect(plane_index)?);
        let image_info = vk::ImageMemoryRequirementsInfo2::default()
            .image(image)
            .push_next(&mut plane_info);
        let mut requirements = vk::MemoryRequirements2::default();
        unsafe { device.get_image_memory_requirements2(&image_info, &mut requirements) };
        Ok(requirements.memory_requirements)
    }

    fn allocate_imported_memory(
        device: &DeviceHandle,
        external_memory_fd: &khr::external_memory_fd::Device,
        image: vk::Image,
        fd: BorrowedFd<'_>,
        requirements: vk::MemoryRequirements,
    ) -> Result<vk::DeviceMemory, VulkanRendererError> {
        let mut fd_properties = vk::MemoryFdPropertiesKHR::default();
        device.observe_result(unsafe {
            external_memory_fd.get_memory_fd_properties(
                vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                fd.as_raw_fd(),
                &mut fd_properties,
            )
        })?;

        let memory_type_index =
            Self::pick_memory_type(requirements.memory_type_bits & fd_properties.memory_type_bits)?;
        let import_fd = fd.try_clone_to_owned()?;
        let import_fd_raw = import_fd.into_raw_fd();
        let import_fd_guard = scopeguard::guard(import_fd_raw, |raw_fd| unsafe {
            libc::close(raw_fd);
        });

        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(*import_fd_guard);
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut import_info)
            .push_next(&mut dedicated_info);

        let memory = device.observe_result(unsafe { device.handle().allocate_memory(&alloc_info, None) })?;
        // A successful import transfers ownership of the duplicated FD to Vulkan.
        let _ = ScopeGuard::into_inner(import_fd_guard);
        Ok(memory)
    }

    fn memory_plane_aspect(plane_index: usize) -> Result<vk::ImageAspectFlags, VulkanRendererError> {
        match plane_index {
            0 => Ok(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT),
            1 => Ok(vk::ImageAspectFlags::MEMORY_PLANE_1_EXT),
            2 => Ok(vk::ImageAspectFlags::MEMORY_PLANE_2_EXT),
            3 => Ok(vk::ImageAspectFlags::MEMORY_PLANE_3_EXT),
            _ => Err(VulkanRendererError::InvalidDmabuf(
                "dma-buf memory plane index is outside Vulkan's supported range",
            )),
        }
    }

    fn destroy_image_and_memories(device: &DeviceHandle, image: vk::Image, memories: &[vk::DeviceMemory]) {
        device.destroy_with(|vk_device| unsafe {
            vk_device.destroy_image(image, None);
            for memory in memories {
                vk_device.free_memory(*memory, None);
            }
        });
    }

    fn pick_memory_type(memory_type_bits: u32) -> Result<u32, VulkanRendererError> {
        if memory_type_bits == 0 {
            return Err(VulkanRendererError::NoCompatibleMemoryType);
        }

        let index = memory_type_bits.trailing_zeros();
        if index >= 32 {
            return Err(VulkanRendererError::NoCompatibleMemoryType);
        }

        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use std::os::fd::{AsFd, AsRawFd, OwnedFd};

    use crate::backend::{
        allocator::{
            dmabuf::{AsDmabuf, Dmabuf, DmabufFlags},
            vulkan::{ImageUsageFlags, VulkanAllocator},
            Allocator, Fourcc, Modifier,
        },
        renderer::vulkan::{VulkanRenderer, VulkanRendererError},
        vulkan::{version::Version, Instance, PhysicalDevice},
    };

    use super::{dmabuf_is_disjoint, DmabufRole, DmabufState};

    #[test]
    fn capture_target_usage_covers_direct_materials_and_terminal_blits() {
        let usage = DmabufRole::CaptureTarget.required_usage();
        assert!(usage.contains(ash::vk::ImageUsageFlags::COLOR_ATTACHMENT));
        assert!(usage.contains(ash::vk::ImageUsageFlags::TRANSFER_SRC));
        assert!(usage.contains(ash::vk::ImageUsageFlags::TRANSFER_DST));
    }

    fn backing_object(name: &str) -> OwnedFd {
        rustix::fs::memfd_create(name, rustix::fs::MemfdFlags::CLOEXEC).expect("temporary backing object")
    }

    fn two_plane_dmabuf(first: OwnedFd, second: OwnedFd) -> Dmabuf {
        let mut builder = Dmabuf::builder((64, 64), Fourcc::Xrgb8888, Modifier::Linear, DmabufFlags::empty());
        assert!(builder.add_plane(first, 0, 0, 256));
        assert!(builder.add_plane(second, 1, 4096, 256));
        builder.build().expect("two-plane dma-buf")
    }

    fn renderer_and_device() -> Option<(PhysicalDevice, VulkanRenderer)> {
        let instance = Instance::new(Version::VERSION_1_3, None).ok()?;
        let physical_device = PhysicalDevice::enumerate(&instance).ok()?.next()?;
        let renderer = match VulkanRenderer::new(&physical_device) {
            Ok(renderer) => renderer,
            Err(
                VulkanRendererError::MissingDeviceExtensions(_)
                | VulkanRendererError::MissingDeviceFeature(_)
                | VulkanRendererError::MissingQueueFamily { .. },
            ) => return None,
            Err(err) => panic!("unexpected Vulkan renderer init failure: {err}"),
        };
        Some((physical_device, renderer))
    }

    #[test]
    fn duplicated_descriptors_are_one_shared_allocation() {
        let first = backing_object("smithay-vulkan-shared");
        let second = first.as_fd().try_clone_to_owned().expect("duplicate descriptor");
        assert_ne!(
            first.as_raw_fd(),
            second.as_raw_fd(),
            "the test requires different descriptor numbers"
        );

        let dmabuf = two_plane_dmabuf(first, second);
        assert!(!dmabuf_is_disjoint(&dmabuf).expect("inspect backing identity"));
    }

    #[test]
    fn distinct_backing_objects_are_disjoint() {
        let first = backing_object("smithay-vulkan-disjoint-first");
        let second = backing_object("smithay-vulkan-disjoint-second");

        let dmabuf = two_plane_dmabuf(first, second);
        assert!(dmabuf_is_disjoint(&dmabuf).expect("inspect backing identity"));
    }

    #[test]
    fn memory_type_selection_uses_the_image_and_fd_intersection() {
        let image_memory_types = 0b0110;
        let fd_memory_types = 0b1100;
        let compatible = image_memory_types & fd_memory_types;
        assert_eq!(DmabufState::pick_memory_type(compatible).unwrap(), 2);

        let incompatible = 0b0010 & fd_memory_types;
        assert!(matches!(
            DmabufState::pick_memory_type(incompatible),
            Err(VulkanRendererError::NoCompatibleMemoryType)
        ));
    }

    #[test]
    fn shared_multiplane_import_when_available() {
        let Some((physical_device, mut renderer)) = renderer_and_device() else {
            return;
        };
        let candidates = renderer
            .dmabuf_import_formats()
            .iter()
            .copied()
            .filter(|format| {
                renderer.has_dmabuf_render_format(*format) && format.modifier != Modifier::Invalid
            })
            .collect::<Vec<_>>();
        let mut allocator = match VulkanAllocator::new(
            &physical_device,
            ImageUsageFlags::SAMPLED | ImageUsageFlags::COLOR_ATTACHMENT,
        ) {
            Ok(allocator) => allocator,
            Err(_) => return,
        };

        for format in candidates {
            let buffer = match allocator.create_buffer(64, 64, format.code, &[format.modifier]) {
                Ok(buffer) => buffer,
                Err(_) => continue,
            };
            let dmabuf = match buffer.export() {
                Ok(dmabuf) if dmabuf.num_planes() > 1 => dmabuf,
                Ok(_) | Err(_) => continue,
            };

            assert!(
                !dmabuf_is_disjoint(&dmabuf).expect("inspect Vulkan allocation identity"),
                "VulkanAllocator exports modifier planes from one shared allocation"
            );
            renderer
                .import_dmabuf_texture(&dmabuf)
                .expect("shared multi-plane texture import should succeed");
            renderer
                .bind_dmabuf_target(&dmabuf)
                .expect("shared multi-plane target bind should succeed");
            return;
        }
    }

    /// Exercises the allocator and a separate import device, including drivers
    /// whose preferred modifier requires compressed dedicated image memory.
    #[test]
    #[ignore = "requires a hardware Vulkan render node; run explicitly"]
    fn exported_modifier_images_roundtrip_between_devices() {
        use crate::backend::renderer::{ExportMem, Frame, Renderer};
        use crate::utils::{Rectangle, Transform};

        let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
        let physical = PhysicalDevice::enumerate(&instance)
            .unwrap()
            .find(|device| device.render_node().ok().flatten().is_some())
            .expect("hardware Vulkan render node required");
        let mut renderer = VulkanRenderer::new(&physical).unwrap();
        let mut allocator = VulkanAllocator::new(
            &physical,
            ImageUsageFlags::SAMPLED | ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSFER_SRC,
        )
        .unwrap();
        let modifiers = renderer
            .dmabuf_import_formats()
            .iter()
            .filter(|format| {
                format.code == Fourcc::Argb8888
                    && format.modifier != Modifier::Invalid
                    && renderer.has_dmabuf_render_format(**format)
            })
            .map(|format| format.modifier)
            .collect::<Vec<_>>();
        assert!(!modifiers.is_empty());

        // Keep the driver's preference: forcing LINEAR would hide a mismatch
        // between a compressed modifier and the allocation's memory attributes.
        for (width, height) in [(64, 64), (1600, 1200), (2400, 1600), (64, 64)] {
            let buffer = allocator
                .create_buffer(width, height, Fourcc::Argb8888, &modifiers)
                .unwrap();
            let dmabuf = buffer.export().unwrap();
            let mut target = renderer.bind_dmabuf_target(&dmabuf).unwrap();
            let size = (width as i32, height as i32).into();
            let mut frame = renderer.render(&mut target, size, Transform::Normal).unwrap();
            frame
                .clear([0.0, 0.0, 1.0, 1.0].into(), &[Rectangle::from_size(size)])
                .unwrap();
            let ready = frame.finish().unwrap();
            renderer.wait(&ready).unwrap();
            let texture = renderer.import_dmabuf_texture(&dmabuf).unwrap();
            let mapping = renderer
                .copy_texture(&texture, Rectangle::from_size((1, 1).into()), Fourcc::Argb8888)
                .unwrap();
            assert_eq!(renderer.map_texture(&mapping).unwrap(), &[255, 0, 0, 255]);
        }
    }

    #[test]
    fn import_bind_reimport_stress() {
        let Some((physical_device, mut renderer)) = renderer_and_device() else {
            return;
        };

        let candidate = renderer
            .dmabuf_import_formats()
            .iter()
            .copied()
            .find(|format| renderer.has_dmabuf_render_format(*format) && format.modifier != Modifier::Invalid)
            .or_else(|| {
                renderer
                    .dmabuf_import_formats()
                    .iter()
                    .copied()
                    .find(|format| renderer.has_dmabuf_render_format(*format))
            });

        let Some(format) = candidate else {
            return;
        };

        let mut allocator = match VulkanAllocator::new(
            &physical_device,
            ImageUsageFlags::SAMPLED | ImageUsageFlags::COLOR_ATTACHMENT,
        ) {
            Ok(allocator) => allocator,
            Err(_) => return,
        };

        let buffer = match allocator.create_buffer(64, 64, format.code, &[format.modifier]) {
            Ok(buffer) => buffer,
            Err(_) => return,
        };

        let dmabuf = match buffer.export() {
            Ok(dmabuf) => dmabuf,
            Err(_) => return,
        };

        let texture_first = renderer
            .import_dmabuf_texture(&dmabuf)
            .expect("initial texture import should succeed");
        let target = renderer
            .bind_dmabuf_target(&dmabuf)
            .expect("dmabuf target bind should succeed");
        let texture_second = renderer
            .import_dmabuf_texture(&dmabuf)
            .expect("re-import should succeed");

        assert_eq!(
            texture_second.image_resource_id(),
            target.image_resource_id(),
            "cache should converge to one imported image after usage upgrade"
        );
        assert_ne!(
            texture_first.image_resource_id(),
            texture_second.image_resource_id(),
            "upgrading usage from sampled-only to sampled+render should re-import once"
        );

        for _ in 0..32 {
            let buffer = match allocator.create_buffer(64, 64, format.code, &[format.modifier]) {
                Ok(buffer) => buffer,
                Err(_) => return,
            };
            let dmabuf = match buffer.export() {
                Ok(dmabuf) => dmabuf,
                Err(_) => return,
            };

            let _texture = renderer
                .import_dmabuf_texture(&dmabuf)
                .expect("stress texture import should succeed");
            let _target = renderer
                .bind_dmabuf_target(&dmabuf)
                .expect("stress target bind should succeed");
        }
    }
}
