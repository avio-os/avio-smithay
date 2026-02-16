use std::sync::Arc;

use ash::vk;
use indexmap::IndexMap;
use tracing::trace;

use super::{device::DeviceHandle, VulkanCacheStats, VulkanRendererError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TextureDescriptorKey {
    image_view: vk::ImageView,
}

#[derive(Debug)]
pub(crate) struct DescriptorState {
    device: Arc<DeviceHandle>,
    texture_layout: vk::DescriptorSetLayout,
    texture_sampler: vk::Sampler,
    pool: vk::DescriptorPool,
    texture_sets: IndexMap<TextureDescriptorKey, vk::DescriptorSet>,
    max_texture_sets: usize,
    cache_stats: VulkanCacheStats,
}

impl DescriptorState {
    pub(crate) const DEFAULT_MAX_TEXTURE_SETS: usize = 256;

    pub(crate) fn new(device: Arc<DeviceHandle>) -> Result<Self, VulkanRendererError> {
        Self::with_capacity(device, Self::DEFAULT_MAX_TEXTURE_SETS)
    }

    pub(crate) fn with_capacity(
        device: Arc<DeviceHandle>,
        max_texture_sets: usize,
    ) -> Result<Self, VulkanRendererError> {
        if max_texture_sets == 0 {
            return Err(VulkanRendererError::TemporaryFailure(
                "descriptor cache capacity must be greater than zero",
            ));
        }

        let vk_device = device.handle();

        let texture_binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let texture_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&texture_binding);

        // SAFETY: Device is valid and create-info points to live memory.
        let texture_layout = unsafe { vk_device.create_descriptor_set_layout(&texture_layout_info, None) }?;

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(0.0);

        // SAFETY: Device is valid and create-info points to live memory.
        let texture_sampler = match unsafe { vk_device.create_sampler(&sampler_info, None) } {
            Ok(sampler) => sampler,
            Err(err) => {
                // SAFETY: Descriptor set layout was created by this device and is not used after this point.
                unsafe { vk_device.destroy_descriptor_set_layout(texture_layout, None) };
                return Err(err.into());
            }
        };

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(max_texture_sets as u32)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .pool_sizes(&pool_sizes)
            .max_sets(max_texture_sets as u32);

        // SAFETY: Device is valid and create-info points to live memory.
        let pool = match unsafe { vk_device.create_descriptor_pool(&pool_info, None) } {
            Ok(pool) => pool,
            Err(err) => {
                unsafe {
                    vk_device.destroy_sampler(texture_sampler, None);
                    vk_device.destroy_descriptor_set_layout(texture_layout, None);
                }
                return Err(err.into());
            }
        };

        Ok(Self {
            device,
            texture_layout,
            texture_sampler,
            pool,
            texture_sets: IndexMap::new(),
            max_texture_sets,
            cache_stats: VulkanCacheStats::default(),
        })
    }

    pub(crate) fn texture_layout(&self) -> vk::DescriptorSetLayout {
        self.texture_layout
    }

    pub(crate) fn texture_sampler(&self) -> vk::Sampler {
        self.texture_sampler
    }

    pub(crate) fn texture_descriptor_set(
        &mut self,
        image_view: vk::ImageView,
    ) -> Result<vk::DescriptorSet, VulkanRendererError> {
        let key = TextureDescriptorKey { image_view };

        if let Some(existing) = self.texture_sets.shift_remove(&key) {
            // Keep hot entries toward the end of insertion order so old entries are evicted first.
            self.texture_sets.insert(key, existing);
            self.cache_stats.hits = self.cache_stats.hits.saturating_add(1);
            trace!(
                hits = self.cache_stats.hits,
                misses = self.cache_stats.misses,
                evictions = self.cache_stats.evictions,
                "vulkan descriptor cache hit"
            );
            return Ok(existing);
        }

        self.cache_stats.misses = self.cache_stats.misses.saturating_add(1);
        self.evict_if_needed()?;

        let layouts = [self.texture_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);

        // SAFETY: Descriptor pool and layout belong to this device and are valid.
        let descriptor_set = unsafe { self.device.handle().allocate_descriptor_sets(&alloc_info) }?
            .into_iter()
            .next()
            .ok_or(VulkanRendererError::TemporaryFailure(
                "Vulkan did not return an allocated descriptor set",
            ))?;

        self.write_texture_descriptor(descriptor_set, image_view);
        self.texture_sets.insert(key, descriptor_set);
        trace!(
            hits = self.cache_stats.hits,
            misses = self.cache_stats.misses,
            evictions = self.cache_stats.evictions,
            "vulkan descriptor cache miss"
        );

        Ok(descriptor_set)
    }

    pub(crate) fn cache_stats(&self) -> VulkanCacheStats {
        self.cache_stats
    }

    pub(crate) fn clear_texture_cache(&mut self) -> Result<(), VulkanRendererError> {
        if self.texture_sets.is_empty() {
            return Ok(());
        }

        let sets = self.texture_sets.values().copied().collect::<Vec<_>>();
        self.cache_stats.evictions = self.cache_stats.evictions.saturating_add(sets.len() as u64);
        self.texture_sets.clear();

        // SAFETY: All descriptor sets originate from this pool and are no longer referenced after cache clear.
        unsafe { self.device.handle().free_descriptor_sets(self.pool, &sets) }?;

        Ok(())
    }

    fn evict_if_needed(&mut self) -> Result<(), VulkanRendererError> {
        if self.texture_sets.len() < self.max_texture_sets {
            return Ok(());
        }

        let Some((_, descriptor_set)) = self.texture_sets.shift_remove_index(0) else {
            return Ok(());
        };
        self.cache_stats.evictions = self.cache_stats.evictions.saturating_add(1);

        // SAFETY: Descriptor set originates from this pool and has been removed from the cache.
        unsafe {
            self.device
                .handle()
                .free_descriptor_sets(self.pool, &[descriptor_set])
        }?;

        Ok(())
    }

    fn write_texture_descriptor(&self, descriptor_set: vk::DescriptorSet, image_view: vk::ImageView) {
        let image_info = [vk::DescriptorImageInfo::default()
            .sampler(self.texture_sampler)
            .image_view(image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info)];

        // SAFETY: Descriptor set and image view are valid handles managed by the renderer.
        unsafe { self.device.handle().update_descriptor_sets(&writes, &[]) };
    }
}

impl Drop for DescriptorState {
    fn drop(&mut self) {
        let device = self.device.handle();
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_sampler(self.texture_sampler, None);
            device.destroy_descriptor_set_layout(self.texture_layout, None);
        }
    }
}
