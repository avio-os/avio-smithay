use std::collections::VecDeque;
use std::sync::Arc;

use ash::vk;
use indexmap::IndexMap;
use tracing::trace;

use super::device::SubmissionId;
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
    texture_sets: IndexMap<TextureDescriptorKey, CachedTextureSet>,
    /// Sets whose texture died; each is reusable once its last-use
    /// submission has retired. Reuse rewrites the set in place — never a
    /// pool free/alloc cycle on the hot path.
    recycled_sets: VecDeque<CachedTextureSet>,
    max_texture_sets: usize,
    cache_stats: VulkanCacheStats,
}

/// One cached combined-image-sampler set with the submission frontier that
/// last referenced it. `last_used` proves quiescence: once that submission
/// retires, no pending command buffer can reference this set, so it may be
/// rewritten or evicted regardless of other in-flight work. This is what
/// replaced the old all-or-nothing rule ("refuse eviction while ANY
/// submission is pending"), which at high refresh refused essentially
/// always and turned a full cache into a frame failure.
#[derive(Debug, Clone, Copy)]
struct CachedTextureSet {
    set: vk::DescriptorSet,
    last_used: SubmissionId,
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
            recycled_sets: VecDeque::new(),
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
        self.retire_dead_views();
        let key = TextureDescriptorKey { image_view, sampler };
        let stamp = self.device.upcoming_submission();

        if let Some(mut existing) = self.texture_sets.shift_remove(&key) {
            // Keep hot entries toward the end of insertion order so old entries are evicted first.
            existing.last_used = stamp;
            self.texture_sets.insert(key, existing);
            self.cache_stats.hits = self.cache_stats.hits.saturating_add(1);
            trace!(
                hits = self.cache_stats.hits,
                misses = self.cache_stats.misses,
                evictions = self.cache_stats.evictions,
                "vulkan descriptor cache hit"
            );
            return Ok(existing.set);
        }

        self.cache_stats.misses = self.cache_stats.misses.saturating_add(1);
        let descriptor_set = self.acquire_set()?;
        self.write_texture_descriptor(descriptor_set, image_view, sampler);
        self.texture_sets.insert(
            key,
            CachedTextureSet {
                set: descriptor_set,
                last_used: stamp,
            },
        );
        trace!(
            hits = self.cache_stats.hits,
            misses = self.cache_stats.misses,
            evictions = self.cache_stats.evictions,
            "vulkan descriptor cache miss"
        );

        Ok(descriptor_set)
    }

    /// Drop cache entries whose image view was destroyed. Their sets move
    /// to the recycle queue and become reusable once their last-use
    /// submission retires — the cache therefore tracks the *live* working
    /// set, and capacity pressure only ever means "this many textures are
    /// genuinely in flight right now".
    fn retire_dead_views(&mut self) {
        let retired = self.device.take_retired_views();
        if retired.is_empty() {
            return;
        }
        for view in retired {
            let dead: Vec<TextureDescriptorKey> = self
                .texture_sets
                .keys()
                .filter(|key| key.image_view == view)
                .copied()
                .collect();
            for key in dead {
                if let Some(entry) = self.texture_sets.shift_remove(&key) {
                    self.cache_stats.dead_view_reclaims =
                        self.cache_stats.dead_view_reclaims.saturating_add(1);
                    self.recycled_sets.push_back(entry);
                }
            }
        }
    }

    /// Produce a writable descriptor set: a recycled quiescent set, a fresh
    /// pool allocation while capacity remains, or the oldest quiescent
    /// cache entry. Refuses only when every set in the pool is still
    /// referenced by pending submissions — a genuine live working set at
    /// capacity, not mere concurrency.
    fn acquire_set(&mut self) -> Result<vk::DescriptorSet, VulkanRendererError> {
        if let Some(index) = self
            .recycled_sets
            .iter()
            .position(|entry| self.device.submission_completed(entry.last_used))
        {
            let entry = self
                .recycled_sets
                .remove(index)
                .expect("position came from this queue");
            return Ok(entry.set);
        }
        if self.texture_sets.len() + self.recycled_sets.len() < self.max_texture_sets {
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
            return Ok(descriptor_set);
        }
        if let Some(index) = self
            .texture_sets
            .values()
            .position(|entry| self.device.submission_completed(entry.last_used))
        {
            let (_, entry) = self
                .texture_sets
                .shift_remove_index(index)
                .expect("position came from this map");
            self.cache_stats.evictions = self.cache_stats.evictions.saturating_add(1);
            return Ok(entry.set);
        }
        Err(VulkanRendererError::TemporaryFailure(
            "every cached descriptor set is referenced by pending submissions",
        ))
    }

    pub(crate) fn cache_stats(&self) -> VulkanCacheStats {
        self.cache_stats
    }

    pub(crate) fn clear_texture_cache(&mut self) -> Result<(), VulkanRendererError> {
        self.retire_dead_views();
        if self.texture_sets.is_empty() && self.recycled_sets.is_empty() {
            return Ok(());
        }

        if self.device.has_pending_submissions() {
            trace!(
                cached_sets = self.texture_sets.len(),
                "skipping vulkan descriptor cache clear while submissions are pending"
            );
            return Ok(());
        }

        let mut sets = self
            .texture_sets
            .values()
            .map(|entry| entry.set)
            .collect::<Vec<_>>();
        sets.extend(self.recycled_sets.iter().map(|entry| entry.set));
        self.cache_stats.evictions = self.cache_stats.evictions.saturating_add(sets.len() as u64);
        self.texture_sets.clear();
        self.recycled_sets.clear();

        // SAFETY: All descriptor sets originate from this pool, have been removed from the
        // cache, and no command buffer using them is pending.
        unsafe { self.device.handle().free_descriptor_sets(self.pool, &sets) }?;

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

#[cfg(test)]
mod tests {
    use ash::vk;

    use super::super::device::{DeviceState, SubmissionId};
    use crate::backend::vulkan::{version::Version, Instance, PhysicalDevice};

    use super::{DescriptorState, TextureSampler};

    struct TestView {
        device: std::sync::Arc<super::super::device::DeviceHandle>,
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: vk::ImageView,
    }

    impl Drop for TestView {
        fn drop(&mut self) {
            self.device.destroy_with(|device| unsafe {
                device.destroy_image_view(self.view, None);
                device.destroy_image(self.image, None);
                device.free_memory(self.memory, None);
            });
        }
    }

    fn test_view(device: &DeviceState) -> Option<TestView> {
        let handle = device.shared_device();
        let vk_device = handle.handle();
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width: 4,
                height: 4,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { vk_device.create_image(&image_info, None) }.ok()?;
        let requirements = unsafe { vk_device.get_image_memory_requirements(image) };
        let memory_type = (0..32).find(|index| requirements.memory_type_bits & (1 << index) != 0)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = match unsafe { vk_device.allocate_memory(&alloc, None) } {
            Ok(memory) => memory,
            Err(_) => {
                unsafe { vk_device.destroy_image(image, None) };
                return None;
            }
        };
        if unsafe { vk_device.bind_image_memory(image, memory, 0) }.is_err() {
            unsafe {
                vk_device.destroy_image(image, None);
                vk_device.free_memory(memory, None);
            }
            return None;
        }
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );
        let view = match unsafe { vk_device.create_image_view(&view_info, None) } {
            Ok(view) => view,
            Err(_) => {
                unsafe {
                    vk_device.destroy_image(image, None);
                    vk_device.free_memory(memory, None);
                }
                return None;
            }
        };
        Some(TestView {
            device: handle,
            image,
            memory,
            view,
        })
    }

    /// Two concurrent `Instance::new` calls can spin inside the ICD on
    /// some drivers; these tests serialize on one lock instead.
    static DEVICE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn test_state() -> Option<(DeviceState, DescriptorState)> {
        let instance = Instance::new(Version::VERSION_1_3, None).ok()?;
        let physical_device = PhysicalDevice::enumerate(&instance).ok()?.next()?;
        let device = DeviceState::new(&physical_device).ok()?;
        let descriptors = DescriptorState::with_capacity(device.shared_device(), 2).ok()?;
        Some((device, descriptors))
    }

    #[test]
    fn a_dead_view_frees_descriptor_capacity_without_global_quiescence() {
        let _serial = DEVICE_TEST_LOCK.lock().expect("device test lock");
        let Some((device, mut descriptors)) = test_state() else {
            return;
        };
        let first = test_view(&device).expect("test image");
        let second = test_view(&device).expect("test image");
        descriptors
            .texture_descriptor_set(first.view, TextureSampler::LINEAR)
            .expect("first set");
        descriptors
            .texture_descriptor_set(second.view, TextureSampler::LINEAR)
            .expect("second set");
        assert_eq!(descriptors.texture_sets.len(), 2);

        // The first texture dies. Its set must leave the live cache on the
        // next drain and, once its last-use submission has retired, serve a
        // brand-new texture — no global no-pending-submissions requirement.
        // The third view is created BEFORE the first dies so the driver
        // cannot reuse the dead handle value for it (that reuse is real,
        // and the drain-before-insert ordering is what keeps it safe in
        // production).
        let third = test_view(&device).expect("test image");
        let dead_view = first.view;
        let handle = device.shared_device();
        // Production textures notify through VulkanImage::drop; the
        // bare test image notifies explicitly.
        handle.note_view_retired(dead_view);
        drop(first);
        handle.note_submission_completed(SubmissionId::for_tests(0));
        let set = descriptors
            .texture_descriptor_set(third.view, TextureSampler::LINEAR)
            .expect("recycled set");
        assert!(descriptors
            .texture_sets
            .keys()
            .all(|key| key.image_view != dead_view));
        assert_eq!(descriptors.texture_sets.len(), 2);
        assert_eq!(descriptors.cache_stats().dead_view_reclaims, 1);
        let _ = set;
    }

    #[test]
    fn full_cache_evicts_quiescent_entries_and_refuses_only_live_ones() {
        let _serial = DEVICE_TEST_LOCK.lock().expect("device test lock");
        let Some((device, mut descriptors)) = test_state() else {
            return;
        };
        let handle = device.shared_device();
        let first = test_view(&device).expect("test image");
        let second = test_view(&device).expect("test image");
        descriptors
            .texture_descriptor_set(first.view, TextureSampler::LINEAR)
            .expect("first set");
        descriptors
            .texture_descriptor_set(second.view, TextureSampler::LINEAR)
            .expect("second set");

        // Nothing has completed: every cached set is (conservatively) still
        // referenced by the upcoming submission, so a third texture refuses.
        let third = test_view(&device).expect("test image");
        assert!(descriptors
            .texture_descriptor_set(third.view, TextureSampler::LINEAR)
            .is_err());

        // The recording frontier retires: the oldest quiescent entry evicts
        // in place of a refusal, live entries stay.
        handle.note_submission_completed(handle.upcoming_submission());
        descriptors
            .texture_descriptor_set(third.view, TextureSampler::LINEAR)
            .expect("evicted a quiescent set");
        assert_eq!(descriptors.texture_sets.len(), 2);
        assert_eq!(descriptors.cache_stats().evictions, 1);
    }
}
