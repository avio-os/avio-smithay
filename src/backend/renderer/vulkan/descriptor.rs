use std::sync::Arc;

use ash::vk;
use indexmap::IndexMap;
use tracing::trace;

use super::{device::DeviceHandle, VulkanCacheStats, VulkanRendererError};
use crate::backend::renderer::TextureFilter;

const BASE_LEVEL_MAX_LOD: f32 = 0.25;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TextureSampler {
    min_filter: TextureFilter,
    mag_filter: TextureFilter,
}

impl TextureSampler {
    pub(crate) const LINEAR: Self = Self {
        min_filter: TextureFilter::Linear,
        mag_filter: TextureFilter::Linear,
    };
    pub(crate) const NEAREST: Self = Self {
        min_filter: TextureFilter::Nearest,
        mag_filter: TextureFilter::Nearest,
    };

    pub(crate) fn new(min_filter: TextureFilter, mag_filter: TextureFilter) -> Self {
        Self {
            min_filter,
            mag_filter,
        }
    }

    fn filters(self) -> (vk::Filter, vk::Filter) {
        (filter_to_vk(self.min_filter), filter_to_vk(self.mag_filter))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TextureDescriptorKey {
    image_view: vk::ImageView,
    sampler: TextureSampler,
}

#[derive(Debug)]
pub(crate) struct DescriptorState {
    device: Arc<DeviceHandle>,
    texture_layout: vk::DescriptorSetLayout,
    texture_samplers: IndexMap<TextureSampler, vk::Sampler>,
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

        let texture_samplers = match create_texture_samplers(vk_device) {
            Ok(samplers) => samplers,
            Err(err) => {
                // SAFETY: Descriptor set layout was created by this device and is not used after this point.
                unsafe { vk_device.destroy_descriptor_set_layout(texture_layout, None) };
                return Err(err);
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
                    for sampler in texture_samplers.values() {
                        vk_device.destroy_sampler(*sampler, None);
                    }
                    vk_device.destroy_descriptor_set_layout(texture_layout, None);
                }
                return Err(err.into());
            }
        };

        Ok(Self {
            device,
            texture_layout,
            texture_samplers,
            pool,
            texture_sets: IndexMap::new(),
            max_texture_sets,
            cache_stats: VulkanCacheStats::default(),
        })
    }

    pub(crate) fn texture_layout(&self) -> vk::DescriptorSetLayout {
        self.texture_layout
    }

    pub(crate) fn texture_descriptor_set(
        &mut self,
        image_view: vk::ImageView,
        sampler: TextureSampler,
    ) -> Result<vk::DescriptorSet, VulkanRendererError> {
        let key = TextureDescriptorKey { image_view, sampler };

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

        self.write_texture_descriptor(descriptor_set, image_view, sampler);
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

        if self.device.has_pending_submissions() {
            trace!(
                cached_sets = self.texture_sets.len(),
                "skipping vulkan descriptor cache clear while submissions are pending"
            );
            return Ok(());
        }

        let sets = self.texture_sets.values().copied().collect::<Vec<_>>();
        self.cache_stats.evictions = self.cache_stats.evictions.saturating_add(sets.len() as u64);
        self.texture_sets.clear();

        // SAFETY: All descriptor sets originate from this pool, have been removed from the
        // cache, and no command buffer using them is pending.
        unsafe { self.device.handle().free_descriptor_sets(self.pool, &sets) }?;

        Ok(())
    }

    fn evict_if_needed(&mut self) -> Result<(), VulkanRendererError> {
        if self.texture_sets.len() < self.max_texture_sets {
            return Ok(());
        }

        if self.device.has_pending_submissions() {
            return Err(VulkanRendererError::TemporaryFailure(
                "descriptor cache is full while submissions are pending",
            ));
        }

        let Some((_, descriptor_set)) = self.texture_sets.shift_remove_index(0) else {
            return Ok(());
        };
        self.cache_stats.evictions = self.cache_stats.evictions.saturating_add(1);

        // SAFETY: Descriptor set originates from this pool, has been removed from the cache,
        // and no command buffer using it is pending.
        unsafe {
            self.device
                .handle()
                .free_descriptor_sets(self.pool, &[descriptor_set])
        }?;

        Ok(())
    }

    fn write_texture_descriptor(
        &self,
        descriptor_set: vk::DescriptorSet,
        image_view: vk::ImageView,
        sampler: TextureSampler,
    ) {
        let image_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler_handle(sampler))
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

    fn sampler_handle(&self, sampler: TextureSampler) -> vk::Sampler {
        self.texture_samplers[&sampler]
    }
}

impl Drop for DescriptorState {
    fn drop(&mut self) {
        // Skipped on a lost device: destroying these objects on a lost VkDevice faults on NVIDIA.
        // `destroy_with` is the single ownership-encoded teardown gate; a no-op when lost.
        self.device.destroy_with(|device| unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            for sampler in self.texture_samplers.values() {
                device.destroy_sampler(*sampler, None);
            }
            device.destroy_descriptor_set_layout(self.texture_layout, None);
        });
    }
}

fn create_texture_samplers(
    device: &ash::Device,
) -> Result<IndexMap<TextureSampler, vk::Sampler>, VulkanRendererError> {
    let mut samplers = IndexMap::new();
    for min_filter in [TextureFilter::Linear, TextureFilter::Nearest] {
        for mag_filter in [TextureFilter::Linear, TextureFilter::Nearest] {
            let sampler = TextureSampler::new(min_filter, mag_filter);
            match create_texture_sampler(device, sampler) {
                Ok(handle) => {
                    samplers.insert(sampler, handle);
                }
                Err(err) => {
                    unsafe {
                        for handle in samplers.values() {
                            device.destroy_sampler(*handle, None);
                        }
                    }
                    return Err(err);
                }
            }
        }
    }
    Ok(samplers)
}

fn create_texture_sampler(
    device: &ash::Device,
    sampler: TextureSampler,
) -> Result<vk::Sampler, VulkanRendererError> {
    let (min_filter, mag_filter) = sampler.filters();
    // Vulkan has no direct GL_LINEAR/GL_NEAREST minification mode for a
    // single-mip texture. Per the spec mapping, clamp to a small non-zero
    // LOD with nearest mip selection so minFilter can still be selected
    // without sampling another mip level.
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(mag_filter)
        .min_filter(min_filter)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .max_lod(BASE_LEVEL_MAX_LOD);

    // SAFETY: Device is valid and create-info points to live memory.
    unsafe { device.create_sampler(&sampler_info, None) }.map_err(Into::into)
}

fn filter_to_vk(filter: TextureFilter) -> vk::Filter {
    match filter {
        TextureFilter::Linear => vk::Filter::LINEAR,
        TextureFilter::Nearest => vk::Filter::NEAREST,
    }
}
