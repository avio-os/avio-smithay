use ash::{ext, vk};
use indexmap::{IndexMap, IndexSet};

use crate::backend::{
    allocator::{
        format::FormatSet,
        vulkan::format::{get_vk_format, known_formats},
        Format, Fourcc, Modifier,
    },
    vulkan::{PhysicalDevice, UnsupportedProperty},
};

use super::VulkanRendererError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FormatUsage {
    Import,
    RenderTarget,
}

impl FormatUsage {
    fn image_usage(self) -> vk::ImageUsageFlags {
        match self {
            FormatUsage::Import => vk::ImageUsageFlags::SAMPLED,
            FormatUsage::RenderTarget => vk::ImageUsageFlags::COLOR_ATTACHMENT,
        }
    }
}

/// Cached DRM modifier capabilities for a Vulkan format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ModifierCapability {
    /// DRM modifier.
    pub(crate) modifier: Modifier,
    /// Number of planes required by this modifier.
    pub(crate) drm_format_modifier_plane_count: u32,
    /// Vulkan tiling features advertised for this modifier.
    pub(crate) drm_format_modifier_tiling_features: vk::FormatFeatureFlags,
}

#[derive(Debug, Default)]
pub(crate) struct FormatCapabilities {
    import_formats: FormatSet,
    render_formats: FormatSet,
    modifier_query_cache: IndexMap<Fourcc, Vec<ModifierCapability>>,
    import_modifiers_by_code: IndexMap<Fourcc, Vec<Modifier>>,
    render_modifiers_by_code: IndexMap<Fourcc, Vec<Modifier>>,
}

impl FormatCapabilities {
    pub(crate) fn new(physical_device: &PhysicalDevice) -> Result<Self, VulkanRendererError> {
        if !physical_device.has_device_extension(ext::image_drm_format_modifier::NAME) {
            return Err(VulkanRendererError::MissingDeviceExtensions(vec![
                ext::image_drm_format_modifier::NAME,
            ]));
        }

        let mut import_formats = IndexSet::new();
        let mut render_formats = IndexSet::new();
        let mut modifier_query_cache = IndexMap::new();
        let mut import_modifiers_by_code: IndexMap<Fourcc, IndexSet<Modifier>> = IndexMap::new();
        let mut render_modifiers_by_code: IndexMap<Fourcc, IndexSet<Modifier>> = IndexMap::new();

        for &fourcc in known_formats() {
            let Some(vk_format) = get_vk_format(fourcc) else {
                continue;
            };

            let mut modifier_properties = physical_device
                .get_format_modifier_properties(vk_format)
                .map_err(Self::map_unsupported_property)?;
            modifier_properties.sort_unstable_by_key(|props| props.drm_format_modifier);
            let mut cached_modifiers = Vec::with_capacity(modifier_properties.len());

            for properties in modifier_properties {
                let modifier = Modifier::from(properties.drm_format_modifier);
                cached_modifiers.push(ModifierCapability {
                    modifier,
                    drm_format_modifier_plane_count: properties.drm_format_modifier_plane_count,
                    drm_format_modifier_tiling_features: properties.drm_format_modifier_tiling_features,
                });

                if Self::is_explicit_modifier_supported(
                    physical_device,
                    vk_format,
                    modifier,
                    FormatUsage::Import,
                )? {
                    Self::insert_supported_format(
                        &mut import_formats,
                        &mut import_modifiers_by_code,
                        fourcc,
                        modifier,
                    );
                }

                if Self::is_explicit_modifier_supported(
                    physical_device,
                    vk_format,
                    modifier,
                    FormatUsage::RenderTarget,
                )? {
                    Self::insert_supported_format(
                        &mut render_formats,
                        &mut render_modifiers_by_code,
                        fourcc,
                        modifier,
                    );
                }
            }

            if !cached_modifiers.is_empty() {
                modifier_query_cache.insert(fourcc, cached_modifiers);
            }

            // Explicitly and conservatively probe "implicit modifier" support. We only advertise
            // Modifier::Invalid when Vulkan reports support without explicit DRM modifier metadata.
            if Self::is_implicit_modifier_supported(physical_device, vk_format, FormatUsage::Import)? {
                Self::insert_supported_format(
                    &mut import_formats,
                    &mut import_modifiers_by_code,
                    fourcc,
                    Modifier::Invalid,
                );
            }

            if Self::is_implicit_modifier_supported(physical_device, vk_format, FormatUsage::RenderTarget)? {
                Self::insert_supported_format(
                    &mut render_formats,
                    &mut render_modifiers_by_code,
                    fourcc,
                    Modifier::Invalid,
                );
            }
        }

        Ok(FormatCapabilities {
            import_formats: import_formats.into_iter().collect(),
            render_formats: render_formats.into_iter().collect(),
            modifier_query_cache,
            import_modifiers_by_code: Self::finalize_modifier_map(import_modifiers_by_code),
            render_modifiers_by_code: Self::finalize_modifier_map(render_modifiers_by_code),
        })
    }

    pub(crate) fn import_formats(&self) -> &FormatSet {
        &self.import_formats
    }

    pub(crate) fn render_formats(&self) -> &FormatSet {
        &self.render_formats
    }

    pub(crate) fn has_import_format(&self, format: Format) -> bool {
        self.import_formats.contains(&format)
    }

    pub(crate) fn has_render_format(&self, format: Format) -> bool {
        self.render_formats.contains(&format)
    }

    pub(crate) fn import_modifiers(&self, code: Fourcc) -> &[Modifier] {
        self.import_modifiers_by_code
            .get(&code)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub(crate) fn render_modifiers(&self, code: Fourcc) -> &[Modifier] {
        self.render_modifiers_by_code
            .get(&code)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub(crate) fn supports_implicit_import_modifier(&self, code: Fourcc) -> bool {
        self.import_modifiers(code).contains(&Modifier::Invalid)
    }

    pub(crate) fn supports_implicit_render_modifier(&self, code: Fourcc) -> bool {
        self.render_modifiers(code).contains(&Modifier::Invalid)
    }

    pub(crate) fn intersect_import_formats(&self, formats: &FormatSet) -> FormatSet {
        self.import_formats.intersection(formats).copied().collect()
    }

    pub(crate) fn intersect_render_formats(&self, formats: &FormatSet) -> FormatSet {
        self.render_formats.intersection(formats).copied().collect()
    }

    pub(crate) fn intersect_import_modifiers(&self, code: Fourcc, requested: &[Modifier]) -> Vec<Modifier> {
        Self::intersect_modifiers(self.import_modifiers(code), requested)
    }

    pub(crate) fn intersect_render_modifiers(&self, code: Fourcc, requested: &[Modifier]) -> Vec<Modifier> {
        Self::intersect_modifiers(self.render_modifiers(code), requested)
    }

    pub(crate) fn modifier_capabilities(&self, code: Fourcc) -> &[ModifierCapability] {
        self.modifier_query_cache
            .get(&code)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn insert_supported_format(
        formats: &mut IndexSet<Format>,
        modifiers_by_code: &mut IndexMap<Fourcc, IndexSet<Modifier>>,
        code: Fourcc,
        modifier: Modifier,
    ) {
        formats.insert(Format { code, modifier });
        modifiers_by_code.entry(code).or_default().insert(modifier);
    }

    fn finalize_modifier_map(map: IndexMap<Fourcc, IndexSet<Modifier>>) -> IndexMap<Fourcc, Vec<Modifier>> {
        map.into_iter()
            .map(|(code, modifiers)| (code, modifiers.into_iter().collect()))
            .collect()
    }

    fn intersect_modifiers(supported: &[Modifier], requested: &[Modifier]) -> Vec<Modifier> {
        if requested.is_empty() {
            return supported.to_vec();
        }

        let requested_set = requested.iter().copied().collect::<IndexSet<_>>();
        supported
            .iter()
            .copied()
            .filter(|modifier| requested_set.contains(modifier))
            .collect()
    }

    fn map_unsupported_property(error: UnsupportedProperty) -> VulkanRendererError {
        match error {
            UnsupportedProperty::Extensions(extensions) => {
                VulkanRendererError::MissingDeviceExtensions(extensions.to_vec())
            }
        }
    }

    fn is_implicit_modifier_supported(
        physical_device: &PhysicalDevice,
        vk_format: vk::Format,
        usage: FormatUsage,
    ) -> Result<bool, VulkanRendererError> {
        Self::query_external_format(physical_device, vk_format, usage, None)
    }

    fn is_explicit_modifier_supported(
        physical_device: &PhysicalDevice,
        vk_format: vk::Format,
        modifier: Modifier,
        usage: FormatUsage,
    ) -> Result<bool, VulkanRendererError> {
        Self::query_external_format(physical_device, vk_format, usage, Some(modifier))
    }

    fn query_external_format(
        physical_device: &PhysicalDevice,
        vk_format: vk::Format,
        usage: FormatUsage,
        modifier: Option<Modifier>,
    ) -> Result<bool, VulkanRendererError> {
        let mut external_image_format_info = vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let mut drm_modifier_info = modifier.map(|modifier| {
            vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
                .drm_format_modifier(modifier.into())
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
        });

        let mut format_info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(vk_format)
            .ty(vk::ImageType::TYPE_2D)
            .usage(usage.image_usage())
            .flags(vk::ImageCreateFlags::empty())
            .push_next(&mut external_image_format_info);

        match drm_modifier_info.as_mut() {
            Some(drm_modifier_info) => {
                format_info = format_info
                    .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                    .push_next(drm_modifier_info);
            }
            None => {
                format_info = format_info.tiling(vk::ImageTiling::OPTIMAL);
            }
        }

        let mut external_image_format_properties = vk::ExternalImageFormatProperties::default();
        let mut image_format_properties =
            vk::ImageFormatProperties2::default().push_next(&mut external_image_format_properties);

        let instance = physical_device.instance().handle();
        // SAFETY: The physical device belongs to `instance` and pointers in `format_info` and
        // `image_format_properties` are valid for this call.
        let result = unsafe {
            instance.get_physical_device_image_format_properties2(
                physical_device.handle(),
                &format_info,
                &mut image_format_properties,
            )
        };

        match result {
            Ok(()) => {
                let external_features = external_image_format_properties
                    .external_memory_properties
                    .external_memory_features;
                Ok(external_features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE))
            }
            Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => Ok(false),
            Err(err) => Err(err.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FormatCapabilities;
    use crate::backend::allocator::{Format, Fourcc, Modifier};
    use indexmap::IndexMap;

    fn sample_capabilities() -> FormatCapabilities {
        let import_formats = [
            Format {
                code: Fourcc::Argb8888,
                modifier: Modifier::Invalid,
            },
            Format {
                code: Fourcc::Argb8888,
                modifier: Modifier::Linear,
            },
            Format {
                code: Fourcc::Argb8888,
                modifier: Modifier::from(0xdead_beef_u64),
            },
            Format {
                code: Fourcc::Xrgb8888,
                modifier: Modifier::Linear,
            },
        ]
        .into_iter()
        .collect();

        let render_formats = [
            Format {
                code: Fourcc::Argb8888,
                modifier: Modifier::Linear,
            },
            Format {
                code: Fourcc::Argb8888,
                modifier: Modifier::from(0xdead_beef_u64),
            },
        ]
        .into_iter()
        .collect();

        let mut import_modifiers_by_code = IndexMap::new();
        import_modifiers_by_code.insert(
            Fourcc::Argb8888,
            vec![
                Modifier::Invalid,
                Modifier::Linear,
                Modifier::from(0xdead_beef_u64),
            ],
        );
        import_modifiers_by_code.insert(Fourcc::Xrgb8888, vec![Modifier::Linear]);

        let mut render_modifiers_by_code = IndexMap::new();
        render_modifiers_by_code.insert(
            Fourcc::Argb8888,
            vec![Modifier::Linear, Modifier::from(0xdead_beef_u64)],
        );

        FormatCapabilities {
            import_formats,
            render_formats,
            modifier_query_cache: IndexMap::new(),
            import_modifiers_by_code,
            render_modifiers_by_code,
        }
    }

    #[test]
    fn implicit_modifier_is_explicitly_detectable() {
        let caps = sample_capabilities();
        assert!(caps.supports_implicit_import_modifier(Fourcc::Argb8888));
        assert!(!caps.supports_implicit_render_modifier(Fourcc::Argb8888));
        assert!(!caps.supports_implicit_import_modifier(Fourcc::Xrgb8888));
    }

    #[test]
    fn modifier_intersection_is_stable_and_ordered_by_supported_priority() {
        let caps = sample_capabilities();
        let requested = [
            Modifier::from(0xdead_beef_u64),
            Modifier::Invalid,
            Modifier::from(0xbead_u64),
        ];
        let result = caps.intersect_import_modifiers(Fourcc::Argb8888, &requested);
        assert_eq!(result, vec![Modifier::Invalid, Modifier::from(0xdead_beef_u64)]);
    }

    #[test]
    fn empty_modifier_request_returns_all_supported() {
        let caps = sample_capabilities();
        let result = caps.intersect_import_modifiers(Fourcc::Argb8888, &[]);
        assert_eq!(
            result,
            vec![
                Modifier::Invalid,
                Modifier::Linear,
                Modifier::from(0xdead_beef_u64),
            ]
        );
    }
}
