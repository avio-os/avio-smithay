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

    /// The requested format/modifier/usage cannot bind disjoint dma-buf memory planes.
    #[error("dma-buf format/modifier does not support disjoint memory-plane import for this usage")]
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

    /// Unsupported memory-upload format.
    #[error("unsupported memory-upload format for Vulkan import: {0:?}")]
    UnsupportedMemoryFormat(crate::backend::allocator::Fourcc),

    /// Memory upload metadata did not pass validation.
    #[error("invalid memory upload metadata: {0}")]
    InvalidMemoryUpload(&'static str),

    /// The bounded persistent upload arena has no completion-retired span
    /// large enough for this exact upload.
    #[error(
        "vulkan upload arena capacity exhausted: requested {requested_bytes} bytes, \
         {in_use_bytes}/{capacity_bytes} bytes in use"
    )]
    UploadCapacityExhausted {
        /// Exact packed byte count requested by the upload.
        requested_bytes: usize,
        /// Aggregate allocated arena capacity for this renderer.
        capacity_bytes: usize,
        /// Capacity still retained by pending or in-flight submissions.
        in_use_bytes: usize,
    },

    /// One exact upload can never fit in any bounded arena chunk, even after
    /// every in-flight reservation retires.
    #[error(
        "vulkan memory upload is larger than the bounded staging arena can represent: \
         requested {requested_bytes} bytes, maximum contiguous capacity {max_contiguous_bytes} bytes"
    )]
    UploadExceedsArenaLimit {
        /// Exact packed byte count requested by the upload.
        requested_bytes: usize,
        /// Largest contiguous reservation the bounded arena can ever provide.
        max_contiguous_bytes: usize,
    },

    /// One render opportunity attempted to enqueue more upload operations
    /// than the renderer's bounded batch can represent.
    #[error("vulkan upload batch operation limit reached: {limit}")]
    UploadBatchFull {
        /// Maximum operations retained before the next queue submission.
        limit: usize,
    },

    /// The bounded descriptor arena cannot admit the requested recording
    /// until one or more submitted descriptor uses complete.
    #[error(
        "vulkan descriptor arena capacity exhausted: requested {requested_sets} sets, \
         {in_use_sets}/{capacity_sets} sets in use"
    )]
    DescriptorCapacityExhausted {
        /// Conservative number of distinct sampled descriptors requested by
        /// the next recording scope.
        requested_sets: usize,
        /// Total descriptor sets currently allocated across bounded pages.
        capacity_sets: usize,
        /// Sets that cannot currently be rewritten because they are recording
        /// or referenced by an incomplete submission.
        in_use_sets: usize,
    },

    /// One recording's conservative descriptor bound exceeds the renderer's
    /// configured arena limit and therefore cannot succeed after waiting.
    #[error(
        "vulkan descriptor request exceeds the bounded arena limit: \
         requested {requested_sets} sets, maximum {max_sets} sets"
    )]
    DescriptorRequestExceedsLimit {
        /// Conservative number of distinct sampled descriptors requested.
        requested_sets: usize,
        /// Hard renderer-local arena bound.
        max_sets: usize,
    },

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
    /// Returns `true` when this error represents an unrecoverable loss of the
    /// underlying Vulkan device (as opposed to a merely temporary or
    /// allocation failure).
    ///
    /// This is deliberately distinct from [`kind`](Self::kind): `kind` lumps
    /// every [`VulkanRendererError::Vk`] together with
    /// [`VulkanRendererError::ContextLost`] under
    /// [`VulkanRendererErrorKind::ContextLost`], so it cannot distinguish
    /// `VK_ERROR_DEVICE_LOST` (renderer must be recreated) from, e.g.,
    /// `VK_ERROR_OUT_OF_DEVICE_MEMORY` (an allocation failure the caller may
    /// recover from by evicting cached resources). Device-loss recovery and
    /// out-of-memory eviction are different policies; only the former should
    /// trigger renderer re-initialization.
    pub fn is_device_lost(&self) -> bool {
        matches!(
            self,
            VulkanRendererError::Vk(vk::Result::ERROR_DEVICE_LOST) | VulkanRendererError::ContextLost(_)
        )
    }

    /// Returns `true` when exact upload work should remain retained until the
    /// renderer's submission-completion edge returns staging capacity.
    pub const fn is_upload_deferred(&self) -> bool {
        matches!(
            self,
            VulkanRendererError::UploadCapacityExhausted { .. } | VulkanRendererError::UploadBatchFull { .. }
        )
    }

    /// Returns `true` only for descriptor pressure that completion can
    /// relieve. An over-limit request is a permanent contract violation, not
    /// a reason to wait for an unrelated submission.
    pub const fn is_descriptor_deferred(&self) -> bool {
        matches!(self, VulkanRendererError::DescriptorCapacityExhausted { .. })
    }

    /// Allocation refusal alone does not prove whether queue submission began.
    /// Callers may reject an attempt only with independent pre-submission proof.
    pub const fn is_resource_allocation_failure(&self) -> bool {
        matches!(
            self,
            Self::Vk(
                vk::Result::ERROR_OUT_OF_HOST_MEMORY
                    | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                    | vk::Result::ERROR_OUT_OF_POOL_MEMORY
                    | vk::Result::ERROR_FRAGMENTED_POOL
            )
        )
    }

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
            | VulkanRendererError::UnsupportedMemoryFormat(_)
            | VulkanRendererError::InvalidMemoryUpload(_)
            | VulkanRendererError::UploadExceedsArenaLimit { .. }
            | VulkanRendererError::UploadCapacityExhausted { .. }
            | VulkanRendererError::UploadBatchFull { .. }
            | VulkanRendererError::DescriptorCapacityExhausted { .. }
            | VulkanRendererError::DescriptorRequestExceedsLimit { .. }
            | VulkanRendererError::TemporaryFailure(_)
            | VulkanRendererError::NotImplemented(_) => VulkanRendererErrorKind::TemporaryFailure,
        }
    }

    /// Creates a standardized "not implemented" error for a named operation.
    pub const fn not_implemented(operation: &'static str) -> Self {
        VulkanRendererError::NotImplemented(operation)
    }
}

impl crate::backend::renderer::damage::MaybeDeviceLost for VulkanRendererError {
    #[inline]
    fn is_device_lost(&self) -> bool {
        VulkanRendererError::is_device_lost(self)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_device_lost_distinguishes_loss_from_oom_and_other() {
        // Genuine device loss: the renderer must be recreated.
        assert!(VulkanRendererError::Vk(vk::Result::ERROR_DEVICE_LOST).is_device_lost());
        assert!(VulkanRendererError::ContextLost("surface lost").is_device_lost());

        // Out-of-device-memory is an allocation failure, NOT a device loss:
        // the caller recovers by evicting cached resources, never by tearing
        // down and re-initializing the renderer.
        assert!(!VulkanRendererError::Vk(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY).is_device_lost());
        assert!(!VulkanRendererError::Vk(vk::Result::ERROR_OUT_OF_HOST_MEMORY).is_device_lost());

        // Other transient / classification errors are not device loss.
        assert!(!VulkanRendererError::TemporaryFailure("retry").is_device_lost());
        assert!(!VulkanRendererError::NoCompatibleMemoryType.is_device_lost());

        // `kind()` cannot make this distinction: it lumps every `Vk(_)` and
        // `ContextLost(_)` under `ContextLost`, which is exactly why
        // `is_device_lost()` exists as a separate predicate.
        assert_eq!(
            VulkanRendererError::Vk(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY).kind(),
            VulkanRendererErrorKind::ContextLost
        );
    }

    #[test]
    fn bounded_upload_pressure_is_typed_and_recoverable() {
        let capacity = VulkanRendererError::UploadCapacityExhausted {
            requested_bytes: 4096,
            capacity_bytes: 8192,
            in_use_bytes: 8192,
        };
        let operations = VulkanRendererError::UploadBatchFull { limit: 256 };

        for error in [capacity, operations] {
            assert!(error.is_upload_deferred());
            assert!(!error.is_device_lost());
            assert_eq!(error.kind(), VulkanRendererErrorKind::TemporaryFailure);
        }
        assert!(!VulkanRendererError::InvalidMemoryUpload("bad layout").is_upload_deferred());
        assert!(!VulkanRendererError::UploadExceedsArenaLimit {
            requested_bytes: 8192,
            max_contiguous_bytes: 4096,
        }
        .is_upload_deferred());
    }

    #[test]
    fn descriptor_pressure_distinguishes_waitable_from_impossible() {
        let pressure = VulkanRendererError::DescriptorCapacityExhausted {
            requested_sets: 768,
            capacity_sets: 1024,
            in_use_sets: 512,
        };
        let impossible = VulkanRendererError::DescriptorRequestExceedsLimit {
            requested_sets: 4097,
            max_sets: 4096,
        };

        assert!(pressure.is_descriptor_deferred());
        assert!(!impossible.is_descriptor_deferred());
        assert!(!pressure.is_device_lost());
        assert!(!impossible.is_device_lost());
        assert_eq!(pressure.kind(), VulkanRendererErrorKind::TemporaryFailure);
        assert_eq!(impossible.kind(), VulkanRendererErrorKind::TemporaryFailure);
    }
}

#[cfg(test)]
mod resource_allocation_tests {
    use super::*;
    #[test]
    fn allocation_refusals_are_distinct_from_device_loss_and_unknown_errors() {
        for result in [
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            vk::Result::ERROR_OUT_OF_POOL_MEMORY,
            vk::Result::ERROR_FRAGMENTED_POOL,
        ] {
            let error = VulkanRendererError::Vk(result);
            assert!(error.is_resource_allocation_failure());
            assert!(!error.is_device_lost());
        }
        for result in [vk::Result::ERROR_DEVICE_LOST, vk::Result::ERROR_UNKNOWN] {
            assert!(!VulkanRendererError::Vk(result).is_resource_allocation_failure());
        }
    }
}
