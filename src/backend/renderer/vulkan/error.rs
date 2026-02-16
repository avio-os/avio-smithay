use std::ffi::CStr;

use ash::vk;

use crate::backend::SwapBuffersError;

/// High-level error classes used for `SwapBuffersError` mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VulkanRendererErrorKind {
    /// The renderer context is not recoverable and must be recreated.
    ContextLost,
    /// The failure is recoverable and rendering can be retried.
    TemporaryFailure,
}

/// Error type for phase-3 Vulkan renderer scaffolding.
#[derive(Debug, thiserror::Error)]
pub enum VulkanRendererError {
    /// Required Vulkan device extensions are unavailable.
    #[error("required Vulkan device extensions are unavailable: {0:?}")]
    MissingDeviceExtensions(Vec<&'static CStr>),

    /// Required Vulkan device feature is unavailable.
    #[error("required Vulkan device feature is unavailable: {0}")]
    MissingDeviceFeature(&'static str),

    /// No queue family satisfies the renderer requirements.
    #[error("no queue family satisfies required flags: {required:?}")]
    MissingQueueFamily {
        /// Queue flags required by the renderer.
        required: vk::QueueFlags,
    },

    /// Vulkan API error.
    #[error(transparent)]
    Vk(#[from] vk::Result),

    /// Unsupported dma-buf format/modifier for the requested usage.
    #[error("unsupported dma-buf format for Vulkan import/bind: {0:?}")]
    UnsupportedDmabufFormat(crate::backend::allocator::Format),

    /// dma-buf metadata did not pass strict validation.
    #[error("invalid dma-buf metadata: {0}")]
    InvalidDmabuf(&'static str),

    /// dma-buf multi-fd/disjoint imports are currently unsupported.
    #[error("dma-buf disjoint/multi-fd imports are currently unsupported")]
    UnsupportedDmabufDisjoint,

    /// dma-buf plane count does not match modifier requirements.
    #[error("dma-buf plane count mismatch for modifier {modifier:?}: expected {expected}, got {actual}")]
    DmabufPlaneCountMismatch {
        /// Modifier that was validated.
        modifier: crate::backend::allocator::Modifier,
        /// Expected number of planes.
        expected: u32,
        /// Actual number of planes.
        actual: usize,
    },

    /// No compatible memory type for an imported image.
    #[error("no compatible Vulkan memory type for dma-buf import")]
    NoCompatibleMemoryType,

    /// Operating system I/O error during dma-buf import.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// The Vulkan renderer context has been lost and must be recreated.
    #[error("vulkan renderer context lost: {0}")]
    ContextLost(&'static str),

    /// A temporary rendering failure occurred.
    #[error("vulkan renderer temporary failure: {0}")]
    TemporaryFailure(&'static str),

    /// The requested operation is not implemented in the current renderer phase.
    #[error("vulkan renderer operation is not implemented yet: {0}")]
    NotImplemented(&'static str),
}

impl VulkanRendererError {
    /// Returns the coarse error class for this error value.
    pub const fn kind(&self) -> VulkanRendererErrorKind {
        match self {
            VulkanRendererError::MissingDeviceExtensions(_)
            | VulkanRendererError::MissingDeviceFeature(_)
            | VulkanRendererError::MissingQueueFamily { .. }
            | VulkanRendererError::Vk(_)
            | VulkanRendererError::ContextLost(_) => VulkanRendererErrorKind::ContextLost,
            VulkanRendererError::UnsupportedDmabufFormat(_)
            | VulkanRendererError::InvalidDmabuf(_)
            | VulkanRendererError::UnsupportedDmabufDisjoint
            | VulkanRendererError::DmabufPlaneCountMismatch { .. }
            | VulkanRendererError::NoCompatibleMemoryType
            | VulkanRendererError::Io(_)
            | VulkanRendererError::TemporaryFailure(_)
            | VulkanRendererError::NotImplemented(_) => VulkanRendererErrorKind::TemporaryFailure,
        }
    }

    /// Creates a standardized "not implemented" error for a named operation.
    pub const fn not_implemented(operation: &'static str) -> Self {
        VulkanRendererError::NotImplemented(operation)
    }
}

impl From<VulkanRendererError> for SwapBuffersError {
    fn from(err: VulkanRendererError) -> Self {
        match err.kind() {
            VulkanRendererErrorKind::ContextLost => SwapBuffersError::ContextLost(Box::new(err)),
            VulkanRendererErrorKind::TemporaryFailure => SwapBuffersError::TemporaryFailure(Box::new(err)),
        }
    }
}
