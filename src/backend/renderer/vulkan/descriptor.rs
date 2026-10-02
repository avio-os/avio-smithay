use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use ash::vk;
use indexmap::IndexMap;
use tracing::trace;

use super::device::SubmissionId;
use super::{device::DeviceHandle, VulkanCacheStats, VulkanDescriptorStats, VulkanRendererError};
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
    incarnation: std::num::NonZeroU64,
    sampler: TextureSampler,
}

#[derive(Debug)]
pub(crate) struct DescriptorState {
    device: Arc<DeviceHandle>,
    texture_layout: vk::DescriptorSetLayout,
    texture_samplers: IndexMap<TextureSampler, vk::Sampler>,
    pools: Vec<vk::DescriptorPool>,
    texture_sets: IndexMap<TextureDescriptorKey, CachedTextureSet>,
    retired_sets: VecDeque<vk::DescriptorSet>,
    free_sets: Vec<vk::DescriptorSet>,
    last_submissions: HashMap<vk::DescriptorSet, Option<SubmissionId>>,
    recording_sets: HashSet<vk::DescriptorSet>,
    cache_target: usize,
    page_size: usize,
    max_sets: usize,
    cache_stats: VulkanCacheStats,
    arena_high_water_sets: usize,
    arena_growth_count: u64,
    arena_deferred_count: u64,
}

#[derive(Debug, Clone, Copy)]
struct CachedTextureSet {
    set: vk::DescriptorSet,
}

impl DescriptorState {
    /// Stable texture identities retained for lookup performance. This is a
    /// cache policy, deliberately separate from allocator capacity.
    pub(crate) const DEFAULT_TEXTURE_CACHE_TARGET: usize = 256;
    /// Descriptor pools grow in bounded pages. Sets are recycled in place;
    /// pages are destroyed only with the renderer.
    pub(crate) const DEFAULT_PAGE_SIZE: usize = 256;
    /// Safety limit for one renderer. This covers the legal Avio material
    /// graph at several frames in flight while keeping memory use bounded.
    pub(crate) const DEFAULT_MAX_TEXTURE_SETS: usize = 4096;

    pub(crate) fn new(device: Arc<DeviceHandle>) -> Result<Self, VulkanRendererError> {
        Self::with_limits(
            device,
            Self::DEFAULT_TEXTURE_CACHE_TARGET,
            Self::DEFAULT_PAGE_SIZE,
            Self::DEFAULT_MAX_TEXTURE_SETS,
        )
    }

    fn with_limits(
        device: Arc<DeviceHandle>,
        cache_target: usize,
        page_size: usize,
        max_sets: usize,
    ) -> Result<Self, VulkanRendererError> {
        if cache_target == 0 || page_size == 0 || max_sets == 0 {
            return Err(VulkanRendererError::TemporaryFailure(
                "descriptor cache and arena limits must be greater than zero",
            ));
        }
        if cache_target > max_sets || page_size > max_sets {
            return Err(VulkanRendererError::TemporaryFailure(
                "descriptor cache target and page size must fit the arena limit",
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

        Ok(Self {
            device,
            texture_layout,
            texture_samplers,
            pools: Vec::new(),
            texture_sets: IndexMap::with_capacity(max_sets),
            retired_sets: VecDeque::with_capacity(max_sets),
            free_sets: Vec::with_capacity(max_sets),
            last_submissions: HashMap::with_capacity(max_sets),
            recording_sets: HashSet::with_capacity(max_sets),
            cache_target,
            page_size,
            max_sets,
            cache_stats: VulkanCacheStats::default(),
            arena_high_water_sets: 0,
            arena_growth_count: 0,
            arena_deferred_count: 0,
        })
    }

    pub(crate) fn texture_layout(&self) -> vk::DescriptorSetLayout {
        self.texture_layout
    }

    pub(crate) fn texture_descriptor_set(
        &mut self,
        image_view: vk::ImageView,
        incarnation: std::num::NonZeroU64,
        sampler: TextureSampler,
    ) -> Result<vk::DescriptorSet, VulkanRendererError> {
        self.retire_previous_incarnations(image_view, incarnation);
        self.reclaim_rewriteable_sets();
        let key = TextureDescriptorKey {
            image_view,
            incarnation,
            sampler,
        };

        if let Some(existing) = self.texture_sets.shift_remove(&key) {
            // Keep hot entries toward the end of insertion order so old entries are evicted first.
            self.texture_sets.insert(key, existing);
            self.recording_sets.insert(existing.set);
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
        self.texture_sets
            .insert(key, CachedTextureSet { set: descriptor_set });
        self.recording_sets.insert(descriptor_set);
        self.note_live_high_water();
        trace!(
            hits = self.cache_stats.hits,
            misses = self.cache_stats.misses,
            evictions = self.cache_stats.evictions,
            "vulkan descriptor cache miss"
        );

        Ok(descriptor_set)
    }

    /// Reserve enough rewriteable or unallocated sets for a conservative
    /// upper bound before command recording begins. Pool growth therefore
    /// stays off the draw path, and impossible requests fail without creating
    /// a partially recorded frame.
    pub(crate) fn reserve_texture_descriptors(
        &mut self,
        requested_sets: usize,
    ) -> Result<(), VulkanRendererError> {
        if requested_sets > self.max_sets {
            return Err(VulkanRendererError::DescriptorRequestExceedsLimit {
                requested_sets,
                max_sets: self.max_sets,
            });
        }

        self.reclaim_rewriteable_sets();
        while self.rewriteable_set_count() < requested_sets && self.allocated_set_count() < self.max_sets {
            self.grow_arena()?;
        }
        if self.rewriteable_set_count() < requested_sets {
            self.arena_deferred_count = self.arena_deferred_count.saturating_add(1);
            return Err(VulkanRendererError::DescriptorCapacityExhausted {
                requested_sets,
                capacity_sets: self.allocated_set_count(),
                in_use_sets: self.in_use_set_count(),
            });
        }
        Ok(())
    }

    /// Commit only the sets actually bound by the successfully submitted
    /// recording. A later use advances the set to the later submission.
    pub(crate) fn commit_submission(&mut self, submission: SubmissionId) {
        for set in self.recording_sets.drain() {
            self.last_submissions.insert(set, Some(submission));
        }
    }

    /// Roll back descriptor-use bookkeeping for a recording that never
    /// reached the queue. Previous successful-submission stamps remain intact.
    pub(crate) fn abort_recording(&mut self) {
        self.recording_sets.clear();
        self.reclaim_rewriteable_sets();
    }

    /// A native handle may be recycled, but the cold shared image incarnation
    /// cannot be. Retire every sampler of the old incarnation before looking
    /// up the new one. Native/recording stamps still gate descriptor rewrites.
    /// Other stale keys need no notification: ordinary bounded LRU eviction
    /// reuses their sets after the exact last submitted reader completes.
    fn retire_previous_incarnations(&mut self, view: vk::ImageView, incarnation: std::num::NonZeroU64) {
        let mut index = 0;
        while index < self.texture_sets.len() {
            if self
                .texture_sets
                .get_index(index)
                .is_some_and(|(key, _)| key.image_view == view && key.incarnation != incarnation)
            {
                let (_, entry) = self
                    .texture_sets
                    .shift_remove_index(index)
                    .expect("observed descriptor entry");
                self.retired_sets.push_back(entry.set);
                self.cache_stats.dead_view_reclaims = self.cache_stats.dead_view_reclaims.saturating_add(1);
            } else {
                index += 1;
            }
        }
    }

    fn reclaim_rewriteable_sets(&mut self) {
        for _ in 0..self.retired_sets.len() {
            let set = self
                .retired_sets
                .pop_front()
                .expect("bounded initial queue length");
            if self.is_rewriteable(set) {
                self.free_sets.push(set);
            } else {
                self.retired_sets.push_back(set);
            }
        }

        while self.texture_sets.len() > self.cache_target {
            let Some(set) = self.evict_oldest_rewriteable_cache_entry() else {
                break;
            };
            self.free_sets.push(set);
        }
    }

    fn acquire_set(&mut self) -> Result<vk::DescriptorSet, VulkanRendererError> {
        self.reclaim_rewriteable_sets();
        if self.texture_sets.len() >= self.cache_target {
            if let Some(set) = self.evict_oldest_rewriteable_cache_entry() {
                return Ok(set);
            }
        }
        if let Some(set) = self.free_sets.pop() {
            return Ok(set);
        }
        if self.allocated_set_count() < self.max_sets {
            self.grow_arena()?;
            return self.free_sets.pop().ok_or(VulkanRendererError::TemporaryFailure(
                "Vulkan descriptor page contained no sets",
            ));
        }
        if let Some(set) = self.evict_oldest_rewriteable_cache_entry() {
            return Ok(set);
        }

        self.arena_deferred_count = self.arena_deferred_count.saturating_add(1);
        Err(VulkanRendererError::DescriptorCapacityExhausted {
            requested_sets: 1,
            capacity_sets: self.allocated_set_count(),
            in_use_sets: self.in_use_set_count(),
        })
    }

    fn grow_arena(&mut self) -> Result<(), VulkanRendererError> {
        let remaining = self.max_sets.saturating_sub(self.allocated_set_count());
        if remaining == 0 {
            return Ok(());
        }
        let count = remaining.min(self.page_size);
        let count_u32 = u32::try_from(count).map_err(|_| {
            VulkanRendererError::TemporaryFailure("descriptor page size exceeds Vulkan limits")
        })?;
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(count_u32)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(count_u32);
        let pool = unsafe { self.device.handle().create_descriptor_pool(&pool_info, None) }?;
        let layouts = vec![self.texture_layout; count];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let sets = match unsafe { self.device.handle().allocate_descriptor_sets(&alloc_info) } {
            Ok(sets) => sets,
            Err(error) => {
                self.device
                    .destroy_with(|device| unsafe { device.destroy_descriptor_pool(pool, None) });
                return Err(error.into());
            }
        };
        if sets.len() != count {
            self.device
                .destroy_with(|device| unsafe { device.destroy_descriptor_pool(pool, None) });
            return Err(VulkanRendererError::TemporaryFailure(
                "Vulkan returned an incomplete descriptor page",
            ));
        }
        for set in &sets {
            self.last_submissions.insert(*set, None);
        }
        self.free_sets.extend(sets);
        self.pools.push(pool);
        self.arena_growth_count = self.arena_growth_count.saturating_add(1);
        trace!(
            page_sets = count,
            arena_sets = self.allocated_set_count(),
            pool_count = self.pools.len(),
            "grew bounded Vulkan descriptor arena"
        );
        Ok(())
    }

    fn evict_oldest_rewriteable_cache_entry(&mut self) -> Option<vk::DescriptorSet> {
        let index = self
            .texture_sets
            .values()
            .position(|entry| self.is_rewriteable(entry.set))?;
        let (_, entry) = self
            .texture_sets
            .shift_remove_index(index)
            .expect("position came from this map");
        self.cache_stats.evictions = self.cache_stats.evictions.saturating_add(1);
        Some(entry.set)
    }

    fn is_rewriteable(&self, set: vk::DescriptorSet) -> bool {
        if self.recording_sets.contains(&set) {
            return false;
        }
        self.last_submissions
            .get(&set)
            .copied()
            .flatten()
            .is_none_or(|submission| self.device.submission_completed(submission))
    }

    fn allocated_set_count(&self) -> usize {
        self.last_submissions.len()
    }

    fn in_use_set_count(&self) -> usize {
        self.last_submissions
            .keys()
            .filter(|set| !self.is_rewriteable(**set))
            .count()
    }

    fn rewriteable_set_count(&self) -> usize {
        self.free_sets.len()
            + self
                .texture_sets
                .values()
                .filter(|entry| self.is_rewriteable(entry.set))
                .count()
    }

    fn note_live_high_water(&mut self) {
        let live = self.texture_sets.len().saturating_add(self.retired_sets.len());
        self.arena_high_water_sets = self.arena_high_water_sets.max(live);
    }

    pub(crate) fn cache_stats(&self) -> VulkanCacheStats {
        self.cache_stats
    }

    pub(crate) fn arena_stats(&self) -> VulkanDescriptorStats {
        VulkanDescriptorStats {
            arena_capacity_sets: self.allocated_set_count(),
            arena_max_sets: self.max_sets,
            arena_pool_count: self.pools.len(),
            cached_sets: self.texture_sets.len(),
            retired_sets: self.retired_sets.len(),
            free_sets: self.free_sets.len(),
            recording_sets: self.recording_sets.len(),
            arena_high_water_sets: self.arena_high_water_sets,
            arena_growth_count: self.arena_growth_count,
            arena_deferred_count: self.arena_deferred_count,
        }
    }

    pub(crate) fn clear_texture_cache(&mut self) -> Result<(), VulkanRendererError> {
        if self.texture_sets.is_empty() && self.retired_sets.is_empty() {
            return Ok(());
        }

        if self.device.has_pending_submissions() || !self.recording_sets.is_empty() {
            trace!(
                cached_sets = self.texture_sets.len(),
                "skipping Vulkan descriptor cache clear while sets are in use"
            );
            return Ok(());
        }

        let removed = self.texture_sets.len() + self.retired_sets.len();
        self.cache_stats.evictions = self.cache_stats.evictions.saturating_add(removed as u64);
        while let Some((_, entry)) = self.texture_sets.pop() {
            self.last_submissions.insert(entry.set, None);
            self.free_sets.push(entry.set);
        }
        while let Some(set) = self.retired_sets.pop_front() {
            self.last_submissions.insert(set, None);
            self.free_sets.push(set);
        }

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
            for pool in self.pools.drain(..) {
                device.destroy_descriptor_pool(pool, None);
            }
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

    use super::{DescriptorState, TextureSampler};

    struct TestView {
        device: std::sync::Arc<super::super::device::DeviceHandle>,
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: vk::ImageView,
        incarnation: std::num::NonZeroU64,
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
        let incarnation = device.shared_device().reserve_image_incarnation().ok()?;
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
        let image = crate::backend::allocator::observe_gpu_allocation(
            unsafe { vk_device.create_image(&image_info, None) },
            crate::backend::allocator::GpuAllocationKind::VulkanImage,
        )
        .ok()?;
        let requirements = unsafe { vk_device.get_image_memory_requirements(image) };
        let memory_type = (0..32).find(|index| requirements.memory_type_bits & (1 << index) != 0)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = match crate::backend::allocator::observe_gpu_allocation(
            unsafe { vk_device.allocate_memory(&alloc, None) },
            crate::backend::allocator::GpuAllocationKind::VulkanDeviceMemory,
        ) {
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
            incarnation,
        })
    }

    /// Two concurrent `Instance::new` calls can spin inside the ICD on
    /// some drivers; these tests serialize on one lock instead.
    static DEVICE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn test_state_with_limits(
        cache_target: usize,
        page_size: usize,
        max_sets: usize,
    ) -> Option<(DeviceState, DescriptorState)> {
        let physical_device = crate::backend::renderer::vulkan::test_support::physical_device()?;
        let device = super::super::test_support::available(
            DeviceState::new(&physical_device),
            "descriptor test device",
        )?;
        let descriptors = super::super::test_support::available(
            DescriptorState::with_limits(device.shared_device(), cache_target, page_size, max_sets),
            "descriptor pools",
        )?;
        Some((device, descriptors))
    }

    fn test_state() -> Option<(DeviceState, DescriptorState)> {
        test_state_with_limits(2, 2, 2)
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
            .texture_descriptor_set(first.view, first.incarnation, TextureSampler::LINEAR)
            .expect("first set");
        descriptors
            .texture_descriptor_set(second.view, second.incarnation, TextureSampler::LINEAR)
            .expect("second set");
        descriptors.commit_submission(SubmissionId::for_tests(0));
        assert_eq!(descriptors.texture_sets.len(), 2);

        // Bounded LRU eviction reuses dead entries after their native stamp,
        // even when another cached texture remains resident.
        let third = test_view(&device).expect("test image");
        let dead_view = first.view;
        let handle = device.shared_device();
        drop(first);
        handle.note_submission_completed(SubmissionId::for_tests(0));
        let set = descriptors
            .texture_descriptor_set(third.view, third.incarnation, TextureSampler::LINEAR)
            .expect("recycled set");
        assert!(descriptors
            .texture_sets
            .keys()
            .all(|key| key.image_view != dead_view));
        assert_eq!(descriptors.texture_sets.len(), 2);
        assert_eq!(descriptors.cache_stats().evictions, 1);
        let _ = set;
    }

    #[test]
    fn committed_sets_wait_for_completion_before_rewrite() {
        let _serial = DEVICE_TEST_LOCK.lock().expect("device test lock");
        let Some((device, mut descriptors)) = test_state_with_limits(1, 1, 1) else {
            return;
        };
        let handle = device.shared_device();
        let first = test_view(&device).expect("test image");
        descriptors
            .texture_descriptor_set(first.view, first.incarnation, TextureSampler::LINEAR)
            .expect("first set");
        descriptors.commit_submission(SubmissionId::for_tests(7));

        let second = test_view(&device).expect("test image");
        assert!(descriptors
            .texture_descriptor_set(second.view, second.incarnation, TextureSampler::LINEAR)
            .is_err());

        handle.note_submission_completed(SubmissionId::for_tests(7));
        descriptors
            .texture_descriptor_set(second.view, second.incarnation, TextureSampler::LINEAR)
            .expect("evicted a quiescent set");
        assert_eq!(descriptors.texture_sets.len(), 1);
        assert_eq!(descriptors.cache_stats().evictions, 1);
    }

    #[test]
    fn abort_makes_never_submitted_sets_immediately_rewriteable() {
        let _serial = DEVICE_TEST_LOCK.lock().expect("device test lock");
        let Some((device, mut descriptors)) = test_state_with_limits(1, 1, 1) else {
            return;
        };
        let first = test_view(&device).expect("test image");
        let second = test_view(&device).expect("test image");
        descriptors
            .texture_descriptor_set(first.view, first.incarnation, TextureSampler::LINEAR)
            .expect("first set");
        assert!(descriptors
            .texture_descriptor_set(second.view, second.incarnation, TextureSampler::LINEAR)
            .is_err());

        descriptors.abort_recording();
        descriptors
            .texture_descriptor_set(second.view, second.incarnation, TextureSampler::LINEAR)
            .expect("aborted use must not invent an in-flight submission");
    }

    #[test]
    fn cache_target_can_spill_into_bounded_pool_pages() {
        let _serial = DEVICE_TEST_LOCK.lock().expect("device test lock");
        let Some((device, mut descriptors)) = test_state_with_limits(2, 2, 4) else {
            return;
        };
        let views = (0..3)
            .map(|_| test_view(&device).expect("test image"))
            .collect::<Vec<_>>();
        for view in &views {
            descriptors
                .texture_descriptor_set(view.view, view.incarnation, TextureSampler::LINEAR)
                .expect("recording may exceed the stable cache target");
        }

        let stats = descriptors.arena_stats();
        assert_eq!(stats.cached_sets, 3);
        assert_eq!(stats.arena_capacity_sets, 4);
        assert_eq!(stats.arena_pool_count, 2);
    }

    #[test]
    fn preflight_is_bounded_and_completion_aware() {
        let _serial = DEVICE_TEST_LOCK.lock().expect("device test lock");
        let Some((device, mut descriptors)) = test_state_with_limits(2, 2, 4) else {
            return;
        };
        descriptors
            .reserve_texture_descriptors(3)
            .expect("bounded arena grows before recording");
        assert_eq!(descriptors.arena_stats().arena_capacity_sets, 4);
        assert!(matches!(
            descriptors.reserve_texture_descriptors(5),
            Err(super::VulkanRendererError::DescriptorRequestExceedsLimit {
                requested_sets: 5,
                max_sets: 4
            })
        ));

        let views = (0..4)
            .map(|_| test_view(&device).expect("test image"))
            .collect::<Vec<_>>();
        for view in &views {
            descriptors
                .texture_descriptor_set(view.view, view.incarnation, TextureSampler::LINEAR)
                .expect("reserved descriptor");
        }
        descriptors.commit_submission(SubmissionId::for_tests(11));
        assert!(matches!(
            descriptors.reserve_texture_descriptors(1),
            Err(super::VulkanRendererError::DescriptorCapacityExhausted { .. })
        ));
        device
            .shared_device()
            .note_submission_completed(SubmissionId::for_tests(11));
        descriptors
            .reserve_texture_descriptors(1)
            .expect("completion returns rewrite capacity");
    }

    #[test]
    fn submission_snapshot_distinguishes_no_submit_from_tracked_submit() {
        let _serial = DEVICE_TEST_LOCK.lock().expect("device test lock");
        let Some((mut device, _descriptors)) = test_state() else {
            return;
        };
        let snapshot = device.submission_snapshot();
        assert!(device.completion_since(snapshot).is_none());

        let command_buffer = device.acquire_command_buffer().expect("command buffer");
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .device_handle()
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("begin command buffer");
            device
                .device_handle()
                .end_command_buffer(command_buffer)
                .expect("end command buffer");
        }
        device.submit(command_buffer).expect("tracked submit");
        assert!(device.completion_since(snapshot).is_some());
        device.wait_for_all_submissions().expect("submission completion");
        assert!(device.completion_since(snapshot).is_some());
    }
}

#[cfg(test)]
#[path = "descriptor/incarnation_tests.rs"]
mod incarnation_tests;
