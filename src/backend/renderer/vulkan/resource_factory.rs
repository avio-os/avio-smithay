//! Immutable preparation authority minted by one exact renderer context.
use std::sync::{atomic::AtomicU64, Arc};

use super::{
    device::DeviceState,
    device_handle::DeviceHandle,
    device_origin::VulkanDeviceOrigin,
    format::FormatCapabilities,
    pipeline::PipelineCreationAuthority,
    prepared_resources::{PreparedAttachmentFormat, PreparedPipelineBank, PreparedSourceImport},
    VulkanRendererError,
};
use crate::backend::{allocator::dmabuf::Dmabuf, renderer::ErasedContextId, vulkan::PhysicalDevice};

/// Queue-free native image creation on an existing logical device.
pub(super) trait ImportDevice {
    fn physical_device(&self) -> &PhysicalDevice;
    fn shared_device(&self) -> Arc<DeviceHandle>;
}
impl ImportDevice for DeviceState {
    fn physical_device(&self) -> &PhysicalDevice {
        self.physical_device()
    }
    fn shared_device(&self) -> Arc<DeviceHandle> {
        self.shared_device()
    }
}
impl ImportDevice for VulkanDeviceOrigin {
    fn physical_device(&self) -> &PhysicalDevice {
        &self.physical_device
    }
    fn shared_device(&self) -> Arc<DeviceHandle> {
        self.device.clone()
    }
}

#[derive(Debug)]
struct Authority {
    context: ErasedContextId,
    origin: VulkanDeviceOrigin,
    formats: Arc<FormatCapabilities>,
    import_ids: Arc<AtomicU64>,
    pipelines: Arc<PipelineCreationAuthority>,
}

/// Cloneable, immutable authority for cold preparation. It grants no access to
/// the owner's queue, command pool, mutable cache or submitted readers. Another
/// renderer on the same device has a distinct authority and cannot adopt it.
#[derive(Clone, Debug)]
pub struct VulkanResourceFactory(Arc<Authority>);
impl VulkanResourceFactory {
    pub(super) fn new(
        context: ErasedContextId,
        origin: VulkanDeviceOrigin,
        formats: Arc<FormatCapabilities>,
        import_ids: Arc<AtomicU64>,
        pipelines: Arc<PipelineCreationAuthority>,
    ) -> Self {
        Self(Arc::new(Authority {
            context,
            origin,
            formats,
            import_ids,
            pipelines,
        }))
    }
    pub(super) fn context(&self) -> &ErasedContextId {
        &self.0.context
    }
    pub(super) fn origin(&self) -> &VulkanDeviceOrigin {
        &self.0.origin
    }
    pub(super) fn formats(&self) -> &FormatCapabilities {
        &self.0.formats
    }
    pub(super) fn import_ids(&self) -> &AtomicU64 {
        &self.0.import_ids
    }

    pub(super) fn pipeline_authority(&self) -> &Arc<PipelineCreationAuthority> {
        &self.0.pipelines
    }

    /// Create the exact owner's native format bank on its untagged helper.
    pub fn prepare_pipeline_bank(
        &self,
        format: PreparedAttachmentFormat,
        generation: u64,
    ) -> Result<PreparedPipelineBank, VulkanRendererError> {
        super::pipeline::preparation::prepare(self, format.native()?, generation)
    }

    /// Prepare one exact immutable backing on an untagged helper. Native image,
    /// memory, views, signature storage and source-custody capacity are created
    /// here. No existing source image is replaced until owner adoption.
    pub fn prepare_sampled_source(
        &self,
        source: &Dmabuf,
        generation: u64,
    ) -> Result<PreparedSourceImport, VulkanRendererError> {
        let _phase = self
            .0
            .origin
            .allocation_phase_scope(super::VulkanAllocationPhase::Warmup);
        super::dmabuf::preparation::prepare(self, source, generation)
    }
}
