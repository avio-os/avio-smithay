//! Optional successful native-resource observation, installed once on cold startup.

use std::sync::OnceLock;
#[path = "observation/frame.rs"]
mod frame;
pub use frame::{
    gpu_frame_allocation_snapshot, GpuFrameAllocationExclusion, GpuFrameAllocationExclusionScope,
    GpuFrameAllocationScope, GpuFrameAllocationSnapshot,
};

/// Native creation unit. Imports are Vulkan bindings, never new physical RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuAllocationKind {
    /// Successful `vkCreateImage`.
    VulkanImage,
    /// Successful `vkCreateBuffer`.
    VulkanBuffer,
    /// Successful `vkAllocateMemory`, including explicit imported bindings.
    VulkanDeviceMemory,
    /// Successful GBM buffer creation.
    GbmBuffer,
    /// Successful DRM dumb-buffer creation (GEM backing).
    DrmDumbBuffer,
    /// Successful DRM framebuffer creation, including AddFB2.
    DrmFramebuffer,
}

static OBSERVER: OnceLock<fn(GpuAllocationKind)> = OnceLock::new();

/// Install a process-local observer once before graphics initialization. The
/// callback must allocate/lock/log nothing and decide its own thread tagging.
/// The `Err` contains the uninstalled callback; an existing observer stays put.
pub fn set_gpu_allocation_observer(observer: fn(GpuAllocationKind)) -> Result<(), fn(GpuAllocationKind)> {
    OBSERVER.set(observer)
}

/// Observe only successful native creations. Error values and caller custody
/// remain unchanged, and callbacks perform no driver or thread-policy query.
pub(crate) fn observe_gpu_allocation<T, E>(result: Result<T, E>, kind: GpuAllocationKind) -> Result<T, E> {
    if result.is_ok() {
        note_gpu_allocation(kind);
    }
    result
}

/// Call immediately after an unwrapped native creation has succeeded.
pub(crate) fn note_gpu_allocation(kind: GpuAllocationKind) {
    frame::note(kind);
    if let Some(observer) = OBSERVER.get() {
        observer(kind);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    thread_local! { static CREATED: Cell<u64> = const { Cell::new(0) }; }
    fn observe(_: GpuAllocationKind) {
        CREATED.set(CREATED.get() + 1);
    }
    #[test]
    fn rejected_native_creations_are_not_successful_allocations() {
        set_gpu_allocation_observer(observe).unwrap();
        assert_eq!(
            observe_gpu_allocation(Err::<(), _>("oom"), GpuAllocationKind::VulkanImage),
            Err("oom")
        );
        assert_eq!(CREATED.get(), 0);
        assert_eq!(
            observe_gpu_allocation(Ok::<_, &str>(17), GpuAllocationKind::DrmFramebuffer),
            Ok(17)
        );
        assert_eq!(CREATED.get(), 1);
        assert!(set_gpu_allocation_observer(observe).is_err());
    }
}
