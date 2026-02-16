use std::{
    collections::HashMap,
    os::fd::{AsRawFd, IntoRawFd},
    sync::{
        atomic::{AtomicI32, Ordering},
        Arc,
    },
};

use ash::vk;
use scopeguard::ScopeGuard;

use crate::{
    backend::allocator::{
        dmabuf::{Dmabuf, WeakDmabuf, MAX_PLANES},
        Buffer, Format, Modifier,
    },
    utils::{Buffer as BufferCoord, Size},
};

use super::{
    device::{DeviceHandle, DeviceState},
    format::FormatCapabilities,
    VulkanRendererError, VulkanTarget, VulkanTexture,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DmabufRole {
    Texture,
    RenderTarget,
}

impl DmabufRole {
    fn required_usage(self) -> vk::ImageUsageFlags {
        match self {
            DmabufRole::Texture => vk::ImageUsageFlags::SAMPLED,
            DmabufRole::RenderTarget => vk::ImageUsageFlags::COLOR_ATTACHMENT,
        }
    }

    fn format_supported(self, formats: &FormatCapabilities, format: Format) -> bool {
        match self {
            DmabufRole::Texture => formats.has_import_format(format),
            DmabufRole::RenderTarget => formats.has_render_format(format),
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
    y_inverted: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct CachedDmabuf {
    pub(crate) handle: WeakDmabuf,
    signature: DmabufSignature,
    imported: Arc<ImportedDmabufImage>,
}

#[derive(Debug, Clone)]
struct DmabufImportDescriptor {
    signature: DmabufSignature,
    vk_format: vk::Format,
}

#[derive(Debug, Default)]
pub(crate) struct DmabufState {
    cache: HashMap<WeakDmabuf, CachedDmabuf>,
    next_import_id: u64,
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
        Ok(VulkanTarget::from_dmabuf_import(
            imported,
            dmabuf.size(),
            Some(dmabuf.format().code),
        ))
    }

    pub(crate) fn cleanup(&mut self) {
        self.cache
            .retain(|_, cached| !cached.handle.is_gone() || Arc::strong_count(&cached.imported) > 1);
    }

    fn import_or_reuse(
        &mut self,
        device: &DeviceState,
        formats: &FormatCapabilities,
        dmabuf: &Dmabuf,
        role: DmabufRole,
    ) -> Result<Arc<ImportedDmabufImage>, VulkanRendererError> {
        self.cleanup();

        let descriptor = Self::validate_dmabuf(dmabuf, formats, role)?;
        let requested_usage = role.required_usage();
        let key = dmabuf.weak();

        if let Some(cached) = self.cache.get(&key) {
            if cached.signature == descriptor.signature && cached.imported.usage().contains(requested_usage) {
                return Ok(cached.imported.clone());
            }
        }

        let usage = self
            .cache
            .get(&key)
            .filter(|cached| cached.signature == descriptor.signature)
            .map(|cached| cached.imported.usage() | requested_usage)
            .unwrap_or(requested_usage);

        let imported = self.create_imported_image(device, dmabuf, &descriptor, usage)?;
        self.cache.insert(
            key.clone(),
            CachedDmabuf {
                handle: key,
                signature: descriptor.signature,
                imported: imported.clone(),
            },
        );

        Ok(imported)
    }

    fn validate_dmabuf(
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

        if strides.iter().any(|stride| *stride == 0) {
            return Err(VulkanRendererError::InvalidDmabuf(
                "dma-buf stride must be non-zero for all planes",
            ));
        }

        let raw_fds = dmabuf.handles().map(|fd| fd.as_raw_fd()).collect::<Vec<_>>();
        if raw_fds.len() != num_planes {
            return Err(VulkanRendererError::InvalidDmabuf(
                "dma-buf fd count does not match plane count",
            ));
        }
        if raw_fds.iter().skip(1).any(|fd| *fd != raw_fds[0]) {
            return Err(VulkanRendererError::UnsupportedDmabufDisjoint);
        }

        if format.modifier == Modifier::Invalid {
            if !role.supports_implicit_modifier(formats, format.code) {
                return Err(VulkanRendererError::UnsupportedDmabufFormat(format));
            }
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
        }

        let Some(vk_format) = crate::backend::allocator::vulkan::format::get_vk_format(format.code) else {
            return Err(VulkanRendererError::UnsupportedDmabufFormat(format));
        };

        Ok(DmabufImportDescriptor {
            signature: DmabufSignature {
                size,
                format,
                num_planes,
                offsets,
                strides,
                y_inverted: dmabuf.y_inverted(),
            },
            vk_format,
        })
    }

    fn create_imported_image(
        &mut self,
        device: &DeviceState,
        dmabuf: &Dmabuf,
        descriptor: &DmabufImportDescriptor,
        usage: vk::ImageUsageFlags,
    ) -> Result<Arc<ImportedDmabufImage>, VulkanRendererError> {
        let device_handle = device.shared_device();
        let vk_device = device_handle.handle();
        let format = descriptor.signature.format;
        let size = descriptor.signature.size;

        let mut external_memory_image_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let mut explicit_modifier_info;
        let plane_layouts;
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
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

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

        let image = match unsafe { vk_device.create_image(&image_create_info, None) } {
            Ok(image) => image,
            Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => {
                return Err(VulkanRendererError::UnsupportedDmabufFormat(format))
            }
            Err(err) => return Err(err.into()),
        };

        let memory_requirements = unsafe { vk_device.get_image_memory_requirements(image) };
        let memory_type_index = Self::pick_memory_type(memory_requirements.memory_type_bits)?;

        let import_fd = dmabuf
            .handles()
            .next()
            .ok_or(VulkanRendererError::InvalidDmabuf("dma-buf has no fds"))?
            .try_clone_to_owned()?;
        let import_fd_raw = import_fd.into_raw_fd();
        let import_fd_guard = scopeguard::guard(import_fd_raw, |fd| unsafe {
            libc::close(fd);
        });

        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(*import_fd_guard);
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut import_info)
            .push_next(&mut dedicated_info);

        let memory = match unsafe { vk_device.allocate_memory(&alloc_info, None) } {
            Ok(memory) => {
                // Ownership of the import fd has moved to Vulkan.
                let _ = ScopeGuard::into_inner(import_fd_guard);
                memory
            }
            Err(err) => {
                unsafe { vk_device.destroy_image(image, None) };
                return Err(err.into());
            }
        };

        if let Err(err) = unsafe { vk_device.bind_image_memory(image, memory, 0) } {
            unsafe {
                vk_device.free_memory(memory, None);
                vk_device.destroy_image(image, None);
            }
            return Err(err.into());
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(descriptor.vk_format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let view = match unsafe { vk_device.create_image_view(&view_info, None) } {
            Ok(view) => view,
            Err(err) => {
                unsafe {
                    vk_device.free_memory(memory, None);
                    vk_device.destroy_image(image, None);
                }
                return Err(err.into());
            }
        };

        let import_id = self.next_import_id;
        self.next_import_id = self.next_import_id.wrapping_add(1);

        Ok(Arc::new(ImportedDmabufImage::new(
            import_id,
            image,
            memory,
            view,
            size,
            format,
            descriptor.vk_format,
            usage,
            descriptor.signature.y_inverted,
            vk::ImageLayout::UNDEFINED,
            device_handle,
        )))
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

pub(crate) struct ImportedDmabufImage {
    import_id: u64,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    size: Size<i32, BufferCoord>,
    format: Format,
    vk_format: vk::Format,
    usage: vk::ImageUsageFlags,
    y_inverted: bool,
    layout: AtomicI32,
    device: Arc<DeviceHandle>,
}

impl std::fmt::Debug for ImportedDmabufImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportedDmabufImage")
            .field("import_id", &self.import_id)
            .field("image", &self.image)
            .field("memory", &self.memory)
            .field("view", &self.view)
            .field("size", &self.size)
            .field("format", &self.format)
            .field("vk_format", &self.vk_format)
            .field("usage", &self.usage)
            .field("y_inverted", &self.y_inverted)
            .finish()
    }
}

impl ImportedDmabufImage {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        import_id: u64,
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: vk::ImageView,
        size: Size<i32, BufferCoord>,
        format: Format,
        vk_format: vk::Format,
        usage: vk::ImageUsageFlags,
        y_inverted: bool,
        initial_layout: vk::ImageLayout,
        device: Arc<DeviceHandle>,
    ) -> Self {
        Self {
            import_id,
            image,
            memory,
            view,
            size,
            format,
            vk_format,
            usage,
            y_inverted,
            layout: AtomicI32::new(initial_layout.as_raw()),
            device,
        }
    }

    pub(crate) fn id(&self) -> u64 {
        self.import_id
    }

    pub(crate) fn image(&self) -> vk::Image {
        self.image
    }

    pub(crate) fn view(&self) -> vk::ImageView {
        self.view
    }

    pub(crate) fn vk_format(&self) -> vk::Format {
        self.vk_format
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

    pub(crate) fn current_layout(&self) -> vk::ImageLayout {
        vk::ImageLayout::from_raw(self.layout.load(Ordering::Relaxed))
    }

    pub(crate) fn set_layout(&self, layout: vk::ImageLayout) {
        self.layout.store(layout.as_raw(), Ordering::Relaxed);
    }
}

impl Drop for ImportedDmabufImage {
    fn drop(&mut self) {
        let device = self.device.handle();
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::{
        allocator::{
            dmabuf::AsDmabuf,
            vulkan::{ImageUsageFlags, VulkanAllocator},
            Allocator, Modifier,
        },
        renderer::vulkan::{VulkanRenderer, VulkanRendererError},
        vulkan::{version::Version, Instance, PhysicalDevice},
    };

    #[test]
    fn import_bind_reimport_stress() {
        let instance = match Instance::new(Version::VERSION_1_3, None) {
            Ok(instance) => instance,
            Err(_) => return,
        };

        let physical_device = match PhysicalDevice::enumerate(&instance) {
            Ok(mut iter) => match iter.next() {
                Some(phd) => phd,
                None => return,
            },
            Err(_) => return,
        };

        let mut renderer = match VulkanRenderer::new(&physical_device) {
            Ok(renderer) => renderer,
            Err(
                VulkanRendererError::MissingDeviceExtensions(_)
                | VulkanRendererError::MissingDeviceFeature(_)
                | VulkanRendererError::MissingQueueFamily { .. },
            ) => {
                return;
            }
            Err(err) => panic!("unexpected Vulkan renderer init failure: {err}"),
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
            texture_second.imported_image_id(),
            target.imported_image_id(),
            "cache should converge to one imported image after usage upgrade"
        );
        assert_ne!(
            texture_first.imported_image_id(),
            texture_second.imported_image_id(),
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
