//! Shared logical-device custody and off-thread image/device retirement.

use std::{
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use super::{
    allocation::AllocationLedger,
    device::SubmissionId,
    image::RetiredImage,
    retirement::{RetirementNode, RetirementQueue},
};
use crate::backend::vulkan::{Instance, QueuePriorityGrant};
use ash::vk;

pub(super) enum DeviceRetirement {
    Fence(vk::Fence),
    Image(RetiredImage),
    /// A context returns the exact actor-allocated notification node here.
    ViewNotification(vk::ImageView),
    #[cfg(test)]
    Drain(std::sync::mpsc::SyncSender<()>),
    OpaqueCustody(Box<dyn std::any::Any + Send + Sync>),
    ReadbackBuffer(super::readback::RetiredReadbackBuffer),
    StagingBuffer(super::staging::RetiredStagingBuffer),
    ExternalWaitSemaphores(super::external_wait_storage::RetiredExternalWaitSemaphores),
    ExternalWaitOwner(super::external_wait_storage::ExternalWaitOwnerReturn),
    #[cfg(feature = "wayland_frontend")]
    HostBuffer(super::host_memory::RetiredHostBuffer),
}

pub(crate) struct DeviceHandle {
    device: ash::Device,
    /// Immutable grant from the call which created this exact native queue.
    /// Every renderer originating here shares this authority.
    queue_priority: QueuePriorityGrant,
    queue_order: super::ordered_queue::OrderedQueue,
    offscreen_ids: Arc<std::sync::atomic::AtomicU64>,
    allocation_ledger: Arc<AllocationLedger>,
    /// The executor owns the parent Instance and logical-device destruction.
    /// Every child keeps this endpoint alive through its DeviceHandle Arc;
    /// endpoint closure drains images before destroying device and instance.
    retirement: RetirementQueue<DeviceRetirement>,
    command_retirement: RetirementQueue<super::device::RetiredCommands>,
    /// Set once the device has been observed to be lost (any Vulkan call returning
    /// `VK_ERROR_DEVICE_LOST`). Owned by the device abstraction so that teardown paths
    /// can consult a single source of truth instead of scattering guards at call sites.
    ///
    /// Vulkan keeps lost-device child handles valid and requires normal parent-before-child
    /// cleanup, but NVIDIA can fault in these destroy paths after some device-loss cascades
    /// (`destroy_fence`/`vkDestroyInstance` → `libnvidia-eglcore` SIGSEGV). Once loss is
    /// observed, this flag intentionally chooses a crash-prevention leak over strict teardown
    /// cleanup.
    lost: Arc<AtomicBool>,
    instance_lost: Arc<AtomicBool>,
    pending_submissions: std::sync::atomic::AtomicUsize,
    /// Ids strictly below this watermark have completed on the queue.
    /// Submissions retire in FIFO order, so a single monotonic frontier is
    /// total. Written only by submission reclaim; read by the descriptor
    /// cache to prove a cached set is no longer referenced by pending work.
    completed_submission_watermark: std::sync::atomic::AtomicU64,
    /// Image views destroyed since the descriptor cache last drained. A
    /// dead view's descriptor set must leave the cache promptly — leaving
    /// it to capacity-triggered eviction let ordinary client-buffer churn
    /// fill the cache in under a minute and then refuse under load, and a
    /// driver reusing the raw handle value could even alias a stale set
    /// onto a new texture.
    retired_texture_views: Arc<super::retired_views::RetirementSubscribers<DeviceRetirement>>,
}

#[cfg(test)]
#[path = "device_handle/retirement_tests.rs"]
pub(super) mod retirement_tests;

impl fmt::Debug for DeviceHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceHandle")
            .field("device", &self.device.handle())
            .field("queue_priority", &self.queue_priority)
            .field("lost", &self.is_lost())
            .finish()
    }
}

impl DeviceHandle {
    pub(super) fn new(
        device: ash::Device,
        instance: Instance,
        queue_priority: QueuePriorityGrant,
    ) -> std::io::Result<Self> {
        let instance_lost = instance.lost_flag();
        Self::with_retirement(device, instance_lost, instance, queue_priority)
    }

    pub(super) fn queue_priority(&self) -> QueuePriorityGrant {
        self.queue_priority
    }

    fn with_retirement(
        device: ash::Device,
        instance_lost: Arc<AtomicBool>,
        parent: impl Send + Sync + 'static,
        queue_priority: QueuePriorityGrant,
    ) -> std::io::Result<Self> {
        let lost = Arc::new(AtomicBool::new(false));
        let retired_texture_views = Arc::new(super::retired_views::RetirementSubscribers::default());
        let destroy_device = device.clone();
        let destroy_lost = lost.clone();
        let destroy_instance_lost = instance_lost.clone();
        let retired_views = retired_texture_views.clone();
        let finish_device = device.clone();
        let finish_lost = lost.clone();
        let finish_instance_lost = instance_lost.clone();
        // Keep a local parent owner through a possible spawn failure, so even
        // initialization cleanup destroys the device before its parent.
        let parent = Arc::new(parent);
        let commands_device = device.clone();
        let commands_lost = lost.clone();
        let commands_instance_lost = instance_lost.clone();
        let command_retirement = match RetirementQueue::start(
            move |commands: super::device::RetiredCommands| {
                let mut reported = false;
                loop {
                    let valid = !commands_lost.load(Ordering::Acquire)
                        && !commands_instance_lost.load(Ordering::Acquire);
                    if valid && commands.wait_complete(&commands_device).is_ok() {
                        commands.destroy(&commands_device);
                        break;
                    }
                    if !reported {
                        tracing::error!("Vulkan command completion is unobservable; retaining exact GPU/host reader custody");
                        reported = true;
                    }
                    // A concrete image retirement or another command retirement
                    // may restore enough driver memory to observe completion. No
                    // timer, fabricated signal or inline-free fallback exists.
                    std::thread::park();
                }
            },
            || {},
        ) {
            Ok(retirement) => retirement,
            Err(error) => {
                unsafe { device.destroy_device(None) };
                return Err(error);
            }
        };
        let command_wake = command_retirement.worker_thread();
        let executor_parent = parent.clone();
        let retirement = match RetirementQueue::start_with_nodes(
            move |node: Box<RetirementNode<DeviceRetirement>>| {
                if matches!(node.value(), DeviceRetirement::ExternalWaitOwner(_)) {
                    super::external_wait_storage::ExternalWaitOwnerReturn::reset(node);
                    command_wake.unpark();
                    return;
                }
                let resource = node.into_value();
                let valid =
                    !destroy_lost.load(Ordering::Acquire) && !destroy_instance_lost.load(Ordering::Acquire);
                match resource {
                    DeviceRetirement::ViewNotification(_) => {}
                    DeviceRetirement::ExternalWaitOwner(_) => unreachable!("wait owner node recycled above"),
                    DeviceRetirement::Fence(fence) => {
                        if valid {
                            unsafe {
                                destroy_device.destroy_fence(fence, None);
                            }
                        }
                    }
                    #[cfg(test)]
                    DeviceRetirement::Drain(completed) => {
                        let _ = completed.send(());
                    }
                    DeviceRetirement::OpaqueCustody(custody) => {
                        if valid {
                            drop(custody);
                        } else {
                            // Foreign export owners can contain their own native
                            // device bindings. Preserve the exact owner after
                            // loss instead of running an unknown destructor.
                            std::mem::forget(custody);
                        }
                    }
                    DeviceRetirement::ReadbackBuffer(buffer) => {
                        if valid {
                            buffer.destroy(&destroy_device);
                        } else {
                            std::mem::forget(buffer);
                        }
                    }
                    DeviceRetirement::StagingBuffer(buffer) => {
                        if valid {
                            buffer.destroy(&destroy_device);
                        } else {
                            // Detached rows may be the last owner of this
                            // persistent mapping. Loss cannot authorize an
                            // inline unmap/free on their releasing thread.
                            std::mem::forget(buffer);
                        }
                    }
                    DeviceRetirement::ExternalWaitSemaphores(semaphores) => {
                        if valid {
                            semaphores.destroy(&destroy_device);
                        } else {
                            std::mem::forget(semaphores);
                        }
                    }
                    DeviceRetirement::Image(image) => {
                        retired_views.retired(|| DeviceRetirement::ViewNotification(image.sampled_view));
                        if valid {
                            image.destroy(&destroy_device);
                        }
                    }
                    #[cfg(feature = "wayland_frontend")]
                    DeviceRetirement::HostBuffer(buffer) => {
                        if valid {
                            buffer.destroy(&destroy_device);
                        } else {
                            // A lost device can still retain an imported pointer.
                            // Keep its exact mapping/FD/source quarantined with
                            // the skipped driver binding, rather than unmapping it.
                            std::mem::forget(buffer);
                        }
                    }
                }
                command_wake.unpark();
            },
            move || {
                if !finish_lost.load(Ordering::Acquire) && !finish_instance_lost.load(Ordering::Acquire) {
                    // SAFETY: Closing the endpoint proves every device child
                    // dropped; all queued images have already been destroyed.
                    unsafe { finish_device.destroy_device(None) };
                }
                // Parent-before-child lifetime holds even when the last
                // DeviceHandle Arc disappeared on a non-renderer thread.
                drop(executor_parent);
            },
        ) {
            Ok(retirement) => retirement,
            Err(error) => {
                // No device child exists yet. Fail initialization rather than
                // installing an inline-free fallback for later resource drops.
                unsafe { device.destroy_device(None) };
                return Err(error);
            }
        };
        Ok(Self {
            device,
            queue_priority,
            queue_order: Default::default(),
            offscreen_ids: Arc::new(std::sync::atomic::AtomicU64::new(1u64 << 61)),
            allocation_ledger: Arc::new(AllocationLedger::default()),
            retirement,
            command_retirement,
            lost,
            instance_lost,
            pending_submissions: std::sync::atomic::AtomicUsize::new(0),
            completed_submission_watermark: std::sync::atomic::AtomicU64::new(0),
            retired_texture_views,
        })
    }

    pub(super) fn offscreen_ids(&self) -> Arc<std::sync::atomic::AtomicU64> {
        self.offscreen_ids.clone()
    }

    pub(super) fn submit_queue(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
    ) -> Result<SubmissionId, super::VulkanRendererError> {
        self.queue_order
            .submit(|| self.observe_result(unsafe { self.device.queue_submit(queue, submits, fence) }))
            .map(SubmissionId)
            .map_err(queue_error)
    }

    pub(super) fn wait_queue_idle(&self, queue: vk::Queue) -> Result<(), super::VulkanRendererError> {
        self.queue_order
            .access(|| self.observe_result(unsafe { self.device.queue_wait_idle(queue) }))
            .map_err(queue_error)
    }

    pub(super) fn retire_commands(&self, commands: Box<RetirementNode<super::device::RetiredCommands>>) {
        self.command_retirement.retire(commands);
    }

    pub(super) fn retire_resource(&self, resource: Box<RetirementNode<DeviceRetirement>>) {
        self.retirement.retire(resource);
    }

    #[cfg(test)]
    pub(super) fn for_retirement_test(device: ash::Device, parent: impl Send + Sync + 'static) -> Self {
        Self::with_retirement(
            device,
            Arc::new(AtomicBool::new(false)),
            parent,
            QueuePriorityGrant::NOT_REQUESTED,
        )
        .unwrap()
    }

    pub(crate) fn allocation_ledger(&self) -> &Arc<AllocationLedger> {
        &self.allocation_ledger
    }

    /// Live-operation accessor. Always returns the device regardless of validity — callers on
    /// the live render path must keep using this so a single observed loss does not silently
    /// disable in-flight work that the caller is already prepared to error out of.
    pub(crate) fn handle(&self) -> &ash::Device {
        &self.device
    }

    /// Marks the device as lost. Idempotent; must only be called after a Vulkan call returns
    /// `VK_ERROR_DEVICE_LOST` (never for `VK_ERROR_OUT_OF_DEVICE_MEMORY`, which is an
    /// allocator/eviction event handled elsewhere).
    pub(super) fn mark_lost(&self) {
        self.lost.store(true, Ordering::Release);
        self.instance_lost.store(true, Ordering::Release);
    }

    /// Records `VK_ERROR_DEVICE_LOST` at the device-ownership layer and returns the
    /// original result. Callers keep their normal error flow, while every path that
    /// can newly observe loss flips the shared validity bit before teardown begins.
    pub(super) fn observe_result<T>(&self, result: Result<T, vk::Result>) -> Result<T, vk::Result> {
        if matches!(result, Err(vk::Result::ERROR_DEVICE_LOST)) {
            self.mark_lost();
        }
        result
    }

    /// Returns whether the device has been observed lost.
    pub(super) fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire) || self.instance_lost.load(Ordering::Acquire)
    }

    pub(super) fn has_pending_submissions(&self) -> bool {
        self.pending_submissions.load(Ordering::Acquire) != 0
    }

    /// Record one destroyed (or about-to-be-destroyed) sampled image view so
    /// the descriptor cache can retire its set on the next drain. Views that
    /// never had a cached set drain as no-ops.
    #[cfg(test)]
    pub(super) fn note_view_retired(&self, view: vk::ImageView) {
        self.retired_texture_views
            .retired(|| DeviceRetirement::ViewNotification(view));
    }

    /// Cold test-only completion edge for all previously published native
    /// resource nodes. Never polls a driver or relies on a timing delay.
    #[cfg(test)]
    pub(super) fn wait_retirement_drained(&self) {
        let (completed, received) = std::sync::mpsc::sync_channel(1);
        self.retire_resource(RetirementNode::new(DeviceRetirement::Drain(completed)));
        received.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
    }

    pub(super) fn retired_view_subscription(
        &self,
    ) -> super::retired_views::RetirementSubscription<DeviceRetirement> {
        self.retired_texture_views.subscribe()
    }

    pub(super) fn note_submission_completed(&self, id: SubmissionId) {
        // FIFO retirement makes this monotonic; max() guards the
        // wait-for-all path racing an ordinary reclaim.
        self.completed_submission_watermark
            .fetch_max(id.0.wrapping_add(1), Ordering::AcqRel);
    }

    pub(super) fn submission_completed(&self, id: SubmissionId) -> bool {
        id.0 < self.completed_submission_watermark.load(Ordering::Acquire)
    }

    pub(super) fn mark_submission_pending(&self) {
        self.pending_submissions.fetch_add(1, Ordering::AcqRel);
    }

    pub(super) fn mark_submission_completed(&self) {
        self.pending_submissions.fetch_sub(1, Ordering::AcqRel);
    }

    /// Teardown/Drop accessor. Returns the device only while it has not been marked lost,
    /// so NVIDIA-sensitive destroy/wait calls are skipped after a device-loss observation.
    pub(super) fn handle_for_destroy(&self) -> Option<&ash::Device> {
        (!self.is_lost()).then_some(&self.device)
    }

    /// Runs `f` with the device only while it is still valid. The single teardown pattern:
    /// a no-op on a lost device, otherwise a normal destroy. Healthy-path cost is one
    /// relaxed atomic load.
    pub(crate) fn destroy_with(&self, f: impl FnOnce(&ash::Device)) {
        if let Some(device) = self.handle_for_destroy() {
            f(device);
        }
    }

    /// Test-only: clears the lost flag so a device sabotaged into the lost state for assertions
    /// can still be torn down through the normal (healthy) destroy path, avoiding a leaked live
    /// `VkDevice` whose surviving instance destruction faults on some drivers.
    #[cfg(test)]
    pub(super) fn clear_lost_for_test(&self) {
        self.lost.store(false, Ordering::Release);
        self.instance_lost.store(false, Ordering::Release);
        self.pending_submissions.store(0, Ordering::Release);
    }
}

fn queue_error(error: super::ordered_queue::QueueError<vk::Result>) -> super::VulkanRendererError {
    match error {
        super::ordered_queue::QueueError::Backend(error) => error.into(),
        super::ordered_queue::QueueError::Poisoned => {
            super::VulkanRendererError::TemporaryFailure("shared queue owner panicked")
        }
        super::ordered_queue::QueueError::Exhausted => {
            super::VulkanRendererError::TemporaryFailure("shared queue submission identities exhausted")
        }
    }
}
