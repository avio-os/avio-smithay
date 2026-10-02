//! Exact format-bank preparation and allocation-free owner adoption.
use super::super::{prepared_resources::PreparedResourceAdoption, resource_factory::VulkanResourceFactory};
use super::*;

/// Prepared native handles for one exact renderer, operation and attachment
/// format. Rejected or duplicate banks must return to their cold result owner.
#[derive(Debug)]
pub struct PreparedPipelineBank {
    authority: Arc<PipelineCreationAuthority>,
    context: crate::backend::renderer::ErasedContextId,
    generation: u64,
    format: vk::Format,
    set: Option<FormatPipelineSet>,
}
impl PreparedPipelineBank {
    /// Exact operation generation retained until owner adoption.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}
impl Drop for PreparedPipelineBank {
    fn drop(&mut self) {
        if let Some(set) = self.set.take() {
            self.authority
                .device
                .destroy_with(|device| destroy_format_set(device, set));
        }
    }
}

pub(in crate::backend::renderer::vulkan) fn prepare(
    factory: &VulkanResourceFactory,
    format: vk::Format,
    generation: u64,
) -> Result<PreparedPipelineBank, VulkanRendererError> {
    if generation == 0 {
        return Err(VulkanRendererError::TemporaryFailure(
            "zero pipeline preparation generation",
        ));
    }
    let set = factory.pipeline_authority().create_format_pipeline_set(format)?;
    Ok(PreparedPipelineBank {
        authority: factory.pipeline_authority().clone(),
        context: factory.context().clone(),
        generation,
        format,
        set: Some(set),
    })
}
impl PipelineState {
    pub(in crate::backend::renderer::vulkan) fn creation_authority(&self) -> Arc<PipelineCreationAuthority> {
        self.authority.clone()
    }

    pub(in crate::backend::renderer::vulkan) fn adopt_prepared(
        &mut self,
        prepared: &mut PreparedPipelineBank,
        expected_factory: &VulkanResourceFactory,
        generation: u64,
        format: vk::Format,
    ) -> Result<PreparedResourceAdoption, VulkanRendererError> {
        self.adopt_prepared_for_context(prepared, expected_factory.context(), generation, format)
    }

    fn adopt_prepared_for_context(
        &mut self,
        prepared: &mut PreparedPipelineBank,
        expected_context: &crate::backend::renderer::ErasedContextId,
        generation: u64,
        format: vk::Format,
    ) -> Result<PreparedResourceAdoption, VulkanRendererError> {
        if prepared.generation != generation
            || prepared.format != format
            || prepared.set.is_none()
            || &prepared.context != expected_context
            || !Arc::ptr_eq(&prepared.authority, &self.authority)
        {
            return Err(VulkanRendererError::TemporaryFailure(
                "prepared pipeline authority or identity mismatch",
            ));
        }
        if self.per_format.contains_key(&format) {
            return Ok(PreparedResourceAdoption::AlreadyPrepared);
        }
        if self.per_format.len() >= self.per_format.capacity() {
            return Ok(PreparedResourceAdoption::CapacityDeferred);
        }
        self.per_format.insert(
            format,
            prepared.set.take().expect("validated prepared pipeline bank"),
        );
        Ok(PreparedResourceAdoption::Adopted)
    }
}

#[cfg(test)]
#[path = "preparation_tests.rs"]
mod tests;
