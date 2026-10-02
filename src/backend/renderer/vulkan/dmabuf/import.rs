//! Native import construction shared by the exact owner and its cold factory.
use super::*;

impl DmabufState {
    pub(super) fn validate_dmabuf(
        device: &impl super::super::resource_factory::ImportDevice,
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

    pub(super) fn create_image_resource(
        import_ids: &std::sync::atomic::AtomicU64,
        device: &impl super::super::resource_factory::ImportDevice,
        dmabuf: &Dmabuf,
        descriptor: &DmabufImportDescriptor,
        usage: vk::ImageUsageFlags,
    ) -> Result<Arc<VulkanImage>, VulkanRendererError> {
        let import_id = import_ids
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |id| id.checked_add(1),
            )
            .map_err(|_| VulkanRendererError::TemporaryFailure("import image identity exhausted"))?;
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

        let image = match device_handle.observe_result(crate::backend::allocator::observe_gpu_allocation(
            unsafe { vk_device.create_image(&image_create_info, None) },
            crate::backend::allocator::GpuAllocationKind::VulkanImage,
        )) {
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
        let mut allocations = Vec::with_capacity(memory_count);

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

            match Self::allocate_imported_memory(
                &device_handle,
                &external_memory_fd,
                image,
                fd,
                requirements,
                dmabuf.backing_metadata(),
            ) {
                Ok((memory, allocation)) => {
                    memories.push(memory);
                    allocations.push(allocation);
                }
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

        Ok(Arc::new(VulkanImage::new_external_dmabuf(
            import_id,
            image,
            memories,
            allocations,
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
}
