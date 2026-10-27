//! Vulkan uses the shared preallocated native retirement queue.
pub use super::super::retirement::RetirementCompletion as VulkanRetirementCompletion;
pub(super) use super::super::retirement::{RetirementNode, RetirementQueue};
