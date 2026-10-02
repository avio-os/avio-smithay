//! Exact context adoption; retired metadata returns with the cold result owner.
use super::super::{prepared_resources::PreparedResourceAdoption, resource_factory::VulkanResourceFactory};
use super::*;

/// Strong ownership of the prepared source and image. After successful adoption
/// the caller returns this object to its cold helper, including displaced cache
/// signatures and the previous source-custody Arc; their final drops never run
/// in the warm adopter. Submitted readers keep their independent image owners.
#[derive(Debug)]
pub struct PreparedSourceImport {
    factory: VulkanResourceFactory,
    source: Dmabuf,
    generation: u64,
    image: Arc<VulkanImage>,
    custody: Arc<ImportCustody<VulkanImage>>,
    cached: Option<CachedDmabuf>,
    retired_cache: Option<CachedDmabuf>,
    retired_stale: Vec<CachedDmabuf>,
    retired_image: Option<Arc<VulkanImage>>,
    consumed: bool,
}
impl PreparedSourceImport {
    /// Exact backing identity retained through preparation and adoption.
    pub fn source(&self) -> &Dmabuf {
        &self.source
    }
    /// The caller's exact operation generation; not a source-format hash.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

pub(in crate::backend::renderer::vulkan) fn prepare(
    factory: &VulkanResourceFactory,
    source: &Dmabuf,
    generation: u64,
) -> Result<PreparedSourceImport, VulkanRendererError> {
    if generation == 0 {
        return Err(VulkanRendererError::InvalidDmabuf(
            "zero source preparation generation",
        ));
    }
    let descriptor =
        DmabufState::validate_dmabuf(factory.origin(), source, factory.formats(), DmabufRole::Texture)?;
    let image = DmabufState::create_image_resource(
        factory.import_ids(),
        factory.origin(),
        source,
        &descriptor,
        DmabufRole::Texture.required_usage(),
    )?;
    let custody = source.resource_custody::<ImportCustody<VulkanImage>>();
    // Reserving may allocate, but does not replace the old actual image.
    custody.reserve(factory.context().clone());
    let key = source.weak();
    let cached = CachedDmabuf {
        handle: key,
        signature: descriptor.signature,
        imported: Arc::downgrade(&image),
        custody: Arc::downgrade(&custody),
        context: factory.context().clone(),
        pins: 1,
        last_used: Instant::now(),
    };
    Ok(PreparedSourceImport {
        factory: factory.clone(),
        source: source.clone(),
        generation,
        image,
        custody,
        cached: Some(cached),
        retired_cache: None,
        retired_stale: Vec::with_capacity(MAX_DMABUF_CACHE_ENTRIES),
        retired_image: None,
        consumed: false,
    })
}

impl DmabufState {
    pub(in crate::backend::renderer::vulkan) fn resource_factory(
        &self,
        origin: super::super::VulkanDeviceOrigin,
        formats: Arc<FormatCapabilities>,
        pipelines: Arc<super::super::pipeline::PipelineCreationAuthority>,
    ) -> VulkanResourceFactory {
        VulkanResourceFactory::new(
            self.context.clone(),
            origin,
            formats,
            self.import_ids.clone(),
            pipelines,
        )
    }

    pub(in crate::backend::renderer::vulkan) fn adopt_prepared(
        &mut self,
        device: &DeviceState,
        prepared: &mut PreparedSourceImport,
        expected_source: &Dmabuf,
        expected_generation: u64,
    ) -> Result<PreparedResourceAdoption, VulkanRendererError> {
        self.adopt_prepared_with_custody(device, prepared, expected_source, expected_generation, false)
    }

    pub(in crate::backend::renderer::vulkan) fn adopt_prepared_cold(
        &mut self,
        device: &DeviceState,
        prepared: &mut PreparedSourceImport,
        expected_source: &Dmabuf,
        expected_generation: u64,
    ) -> Result<PreparedResourceAdoption, VulkanRendererError> {
        self.adopt_prepared_with_custody(device, prepared, expected_source, expected_generation, true)
    }

    fn adopt_prepared_with_custody(
        &mut self,
        device: &DeviceState,
        prepared: &mut PreparedSourceImport,
        expected_source: &Dmabuf,
        expected_generation: u64,
        cold_owner: bool,
    ) -> Result<PreparedResourceAdoption, VulkanRendererError> {
        if prepared.consumed
            || prepared.generation != expected_generation
            || prepared.source.weak() != expected_source.weak()
            || prepared.factory.context() != &self.context
            || !Arc::ptr_eq(&prepared.factory.origin().device, &device.shared_device())
        {
            return Err(VulkanRendererError::InvalidDmabuf(
                "prepared import authority or identity mismatch",
            ));
        }
        let key = expected_source.weak();
        if self.sampled_prepared(expected_source) {
            let cached = self.cache.get_mut(&key).expect("prepared cache entry exists");
            cached.pins = cached
                .pins
                .checked_add(1)
                .ok_or(VulkanRendererError::TemporaryFailure(
                    "import membership exhausted",
                ))?;
            prepared.consumed = true;
            return Ok(PreparedResourceAdoption::AlreadyPrepared);
        }
        // Return dead metadata in the already-reserved cold result payload.
        // Removing a key cannot free its weak control because CachedDmabuf
        // retains the same backing until the helper/actor drops this batch.
        let mut index = 0;
        while index < self.cache.len() && prepared.retired_stale.len() < prepared.retired_stale.capacity() {
            let stale = self
                .cache
                .get_index(index)
                .is_some_and(|(_, entry)| entry.pins == 0 && entry.handle.is_gone());
            if stale {
                let (_, cached) = self
                    .cache
                    .shift_remove_index(index)
                    .expect("observed cache entry");
                prepared.retired_stale.push(cached);
                self.cache_stats.evictions = self.cache_stats.evictions.saturating_add(1);
                self.cleanup_stale_evictions = self.cleanup_stale_evictions.saturating_add(1);
            } else {
                index += 1;
            }
        }
        if !self.cache.contains_key(&key)
            && (self.cache.len() >= MAX_DMABUF_CACHE_ENTRIES || self.cache.len() >= self.cache.capacity())
        {
            return Ok(PreparedResourceAdoption::CapacityDeferred);
        }
        // A target import cannot lose its renderer-authored layout/usage history.
        if self
            .cache
            .get(&key)
            .and_then(|entry| entry.imported.upgrade())
            .is_some_and(|image| !idle_evictable_usage(image.usage()))
        {
            return Err(VulkanRendererError::InvalidDmabuf(
                "prepared sampled import cannot replace an authored target",
            ));
        }
        let pins = self.cache.get(&key).map_or(Ok(1), |old| {
            old.pins
                .checked_add(1)
                .ok_or(VulkanRendererError::TemporaryFailure(
                    "import membership exhausted",
                ))
        })?;
        let replacement = if cold_owner {
            prepared
                .custody
                .replace_reserved_cold(&self.context, prepared.image.clone())
                .map_err(|_| VulkanRendererError::InvalidDmabuf("reserved cold import custody unavailable"))
        } else {
            match prepared
                .custody
                .try_replace_reserved(&self.context, prepared.image.clone())
            {
                Ok(previous) => Ok(previous),
                Err(()) => return Ok(PreparedResourceAdoption::Busy),
            }
        };
        let previous = replacement?;
        let mut cached = prepared.cached.take().expect("unconsumed prepared metadata");
        cached.pins = pins;
        prepared.retired_cache = self.cache.shift_remove(&key);
        prepared.retired_image = previous;
        self.cache.insert(key, cached);
        self.cache_stats.misses = self.cache_stats.misses.saturating_add(1);
        self.update_max_cache_len();
        prepared.consumed = true;
        Ok(PreparedResourceAdoption::Adopted)
    }
}
