use std::{
    collections::VecDeque,
    ffi::CStr,
    fmt,
    os::fd::{FromRawFd, OwnedFd},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

use ash::{ext, khr, vk};
use tracing::{instrument, trace, warn};

use crate::backend::vulkan::{version::Version, Instance, PhysicalDevice};

use super::{
    dmabuf::ImportedDmabufImage,
    sync::{import_sync_file_to_fence, import_sync_file_to_semaphore, VulkanFence},
    VulkanRendererError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SubmissionId(u64);

impl SubmissionId {
    #[cfg(test)]
    pub(crate) fn for_tests(id: u64) -> Self {
        Self(id)
    }
}

struct InFlightSubmission {
    id: SubmissionId,
    fence: VulkanFence,
    /// Binary semaphore signaled by this submission whose SYNC_FD was
    /// exported right after submit (the caller-visible sync_file). The
    /// VkFence is never exported, so it faithfully tracks completion and is
    /// polled directly for reclamation; the semaphore is kept alive until
    /// the submission retires (a semaphore referenced by pending GPU work
    /// must not be destroyed) and destroyed on recycle.
    export_semaphore: Option<vk::Semaphore>,
    command_buffer: vk::CommandBuffer,
    framebuffers: Vec<vk::Framebuffer>,
    retained_images: Vec<Arc<ImportedDmabufImage>>,
    transient_buffers: Vec<TransientBufferAllocation>,
    wait_semaphores: Vec<vk::Semaphore>,
    submitted_at: Instant,
}

/// Buffer allocation whose lifetime is exactly one GPU submission.
///
/// Upload staging must survive `vkQueueSubmit`, but retaining it until a
/// host-side wait defeats asynchronous queueing. The tracked submission owns
/// these handles and destroys them only after its real VkFence signals.
pub(crate) struct TransientBufferAllocation {
    device: Arc<DeviceHandle>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl TransientBufferAllocation {
    pub(crate) fn new(device: &DeviceState, buffer: vk::Buffer, memory: vk::DeviceMemory) -> Self {
        Self {
            device: device.shared_device(),
            buffer,
            memory,
        }
    }
}

impl Drop for TransientBufferAllocation {
    fn drop(&mut self) {
        self.device.destroy_with(|device| {
            // SAFETY: This allocation is constructed only after both handles
            // exist and is dropped either before submission or after its
            // tracked fence signals.
            unsafe {
                device.destroy_buffer(self.buffer, None);
                device.free_memory(self.memory, None);
            }
        });
    }
}

/// The two legal queue-submission contracts.
///
/// An enum keeps acquire-wait consumption and completion export coupled to
/// the operation that owns them; callers cannot construct a mixed policy
/// that silently steals synchronization from a later render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubmissionKind {
    Render,
    Upload,
}

impl SubmissionKind {
    fn consumes_pending_waits(self) -> bool {
        matches!(self, Self::Render)
    }

    fn exports_completion(self) -> bool {
        matches!(self, Self::Render)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DeviceCapabilities {
    timeline_semaphore: bool,
    sync_file_import: bool,
    sync_file_export: bool,
    sync_file_semaphore_import: bool,
    sync_file_semaphore_export: bool,
}

impl DeviceCapabilities {
    pub(crate) fn timeline_semaphore(self) -> bool {
        self.timeline_semaphore
    }

    pub(crate) fn sync_file_import(self) -> bool {
        self.sync_file_import
    }

    pub(crate) fn sync_file_export(self) -> bool {
        self.sync_file_export
    }

    pub(crate) fn sync_file_semaphore_import(self) -> bool {
        self.sync_file_semaphore_import
    }

    pub(crate) fn sync_file_semaphore_export(self) -> bool {
        self.sync_file_semaphore_export
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct DeviceDiagnostics {
    pub(crate) total_submissions: u64,
    pub(crate) blocking_submissions: u64,
    pub(crate) upload_submissions: u64,
    pub(crate) reclaimed_submissions: u64,
    pub(crate) total_submit_cpu_ns: u64,
    pub(crate) max_submit_cpu_ns: u64,
    pub(crate) total_completion_ns: u64,
    pub(crate) max_completion_ns: u64,
    pub(crate) debug_markers_enabled: bool,
}

pub(crate) struct DeviceState {
    physical_device: PhysicalDevice,
    enabled_extensions: Vec<&'static CStr>,
    capabilities: DeviceCapabilities,
    queue_family_index: u32,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    reusable_command_buffers: Vec<vk::CommandBuffer>,
    in_flight_submissions: VecDeque<InFlightSubmission>,
    pending_waits: Vec<(vk::Semaphore, vk::PipelineStageFlags)>,
    next_submission_id: u64,
    device: Arc<DeviceHandle>,
    external_fence_fd: Option<Arc<khr::external_fence_fd::Device>>,
    external_semaphore_fd: Option<Arc<khr::external_semaphore_fd::Device>>,
    debug_utils: Option<Arc<ext::debug_utils::Device>>,
    diagnostics: DeviceDiagnostics,
}

impl fmt::Debug for DeviceState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceState")
            .field("physical_device", &self.physical_device)
            .field("enabled_extensions", &self.enabled_extensions)
            .field("capabilities", &self.capabilities)
            .field("queue_family_index", &self.queue_family_index)
            .field("queue", &self.queue)
            .field("command_pool", &self.command_pool)
            .field("reusable_command_buffers", &self.reusable_command_buffers.len())
            .field("in_flight_submissions", &self.in_flight_submissions.len())
            .field("next_submission_id", &self.next_submission_id)
            .field("device", &self.device.handle().handle())
            .field("supports_sync_file_import", &self.supports_sync_file_import())
            .field(
                "supports_sync_file_fence_import",
                &self.supports_sync_file_fence_import(),
            )
            .field("supports_sync_file_export", &self.supports_sync_file_export())
            .field("debug_markers_enabled", &self.diagnostics.debug_markers_enabled)
            .field("total_submissions", &self.diagnostics.total_submissions)
            .field("blocking_submissions", &self.diagnostics.blocking_submissions)
            .field("upload_submissions", &self.diagnostics.upload_submissions)
            .field("reclaimed_submissions", &self.diagnostics.reclaimed_submissions)
            .finish()
    }
}

pub(super) struct DeviceHandle {
    device: ash::Device,
    /// Keeps the Vulkan parent instance alive until the logical device and
    /// every child object sharing this handle have been destroyed.
    ///
    /// `DeviceState` is only one owner of this handle: descriptor, pipeline,
    /// image, fence, and transient-allocation state can outlive it during
    /// renderer field teardown. Retaining only the instance-loss flag allowed
    /// the last `PhysicalDevice`/`Instance` owner to drop before these device
    /// children, violating Vulkan's parent-before-child lifetime contract.
    _instance: Instance,
    /// Set once the device has been observed to be lost (any Vulkan call returning
    /// `VK_ERROR_DEVICE_LOST`). Owned by the device abstraction so that teardown paths
    /// can consult a single source of truth instead of scattering guards at call sites.
    ///
    /// Vulkan keeps lost-device child handles valid and requires normal parent-before-child
    /// cleanup, but NVIDIA can fault in these destroy paths after some device-loss cascades
    /// (`destroy_fence`/`vkDestroyInstance` → `libnvidia-eglcore` SIGSEGV). Once loss is
    /// observed, this flag intentionally chooses a crash-prevention leak over strict teardown
    /// cleanup.
    lost: AtomicBool,
    instance_lost: Arc<AtomicBool>,
    pending_submissions: std::sync::atomic::AtomicUsize,
    /// Ids strictly below this watermark have completed on the queue.
    /// Submissions retire in FIFO order, so a single monotonic frontier is
    /// total. Written only by submission reclaim; read by the descriptor
    /// cache to prove a cached set is no longer referenced by pending work.
    completed_submission_watermark: std::sync::atomic::AtomicU64,
    /// The id the next queue submission will carry. Consumers stamping
    /// resources "used by the recording frame" use this id; the stamp
    /// completes when that submission (or any later one) retires.
    upcoming_submission: std::sync::atomic::AtomicU64,
    /// Image views destroyed since the descriptor cache last drained. A
    /// dead view's descriptor set must leave the cache promptly — leaving
    /// it to capacity-triggered eviction let ordinary client-buffer churn
    /// fill the cache in under a minute and then refuse under load, and a
    /// driver reusing the raw handle value could even alias a stale set
    /// onto a new texture.
    retired_texture_views: std::sync::Mutex<Vec<vk::ImageView>>,
}

impl fmt::Debug for DeviceHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceHandle")
            .field("device", &self.device.handle())
            .field("lost", &self.is_lost())
            .finish()
    }
}

impl DeviceHandle {
    /// Live-operation accessor. Always returns the device regardless of validity — callers on
    /// the live render path must keep using this so a single observed loss does not silently
    /// disable in-flight work that the caller is already prepared to error out of.
    pub(super) fn handle(&self) -> &ash::Device {
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
    pub(super) fn note_view_retired(&self, view: vk::ImageView) {
        self.retired_texture_views
            .lock()
            .expect("retired-view queue poisoned")
            .push(view);
    }

    pub(super) fn take_retired_views(&self) -> Vec<vk::ImageView> {
        std::mem::take(
            &mut *self
                .retired_texture_views
                .lock()
                .expect("retired-view queue poisoned"),
        )
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

    pub(super) fn upcoming_submission(&self) -> SubmissionId {
        SubmissionId(self.upcoming_submission.load(Ordering::Acquire))
    }

    fn store_upcoming_submission(&self, next: u64) {
        self.upcoming_submission.store(next, Ordering::Release);
    }

    fn mark_submission_pending(&self) {
        self.pending_submissions.fetch_add(1, Ordering::AcqRel);
    }

    fn mark_submission_completed(&self) {
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
    pub(super) fn destroy_with(&self, f: impl FnOnce(&ash::Device)) {
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

impl Drop for DeviceHandle {
    fn drop(&mut self) {
        // Device destruction happens once, after dependent resources are dropped. Vulkan
        // permits this even after loss; skipping it here is the NVIDIA crash workaround.
        if let Some(device) = self.handle_for_destroy() {
            unsafe { device.destroy_device(None) };
        }
    }
}

impl DeviceState {
    pub(crate) fn required_extensions(physical_device: &PhysicalDevice) -> Vec<&'static CStr> {
        // Base requirements for modifier-aware dmabuf import/export workflows.
        let mut extensions = vec![
            ext::image_drm_format_modifier::NAME,
            ext::external_memory_dma_buf::NAME,
            ext::queue_family_foreign::NAME,
            khr::external_memory_fd::NAME,
        ];

        if physical_device.api_version() < Version::VERSION_1_2 {
            // VK_EXT_image_drm_format_modifier requires VK_KHR_image_format_list on Vulkan < 1.2.
            extensions.push(khr::image_format_list::NAME);
        }

        extensions
    }

    pub(crate) fn new(physical_device: &PhysicalDevice) -> Result<Self, VulkanRendererError> {
        let enabled_extensions = Self::validate_required_extensions(physical_device)?;
        let capabilities = Self::query_capabilities(physical_device, &enabled_extensions);
        Self::validate_required_features(physical_device)?;

        let queue_family_index = Self::select_queue_family(physical_device)?;
        let queue_priority = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priority)];

        let extension_ptrs = enabled_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();
        let features = vk::PhysicalDeviceFeatures {
            robust_buffer_access: vk::TRUE,
            ..Default::default()
        };

        let create_info = vk::DeviceCreateInfo::default()
            .enabled_extension_names(&extension_ptrs)
            .enabled_features(&features)
            .queue_create_infos(&queue_info);

        let instance = physical_device.instance().handle();
        // SAFETY: The physical device belongs to this instance and all pointers in create_info
        // are valid for the duration of this call.
        let raw_device = unsafe { instance.create_device(physical_device.handle(), &create_info, None) }?;

        let queue = {
            // SAFETY: Queue family/index are valid for this device by construction in select_queue_family.
            unsafe { raw_device.get_device_queue(queue_family_index, 0) }
        };

        let device = Arc::new(DeviceHandle {
            device: raw_device,
            _instance: physical_device.instance().clone(),
            lost: AtomicBool::new(false),
            instance_lost: physical_device.instance().lost_flag(),
            pending_submissions: std::sync::atomic::AtomicUsize::new(0),
            completed_submission_watermark: std::sync::atomic::AtomicU64::new(0),
            upcoming_submission: std::sync::atomic::AtomicU64::new(0),
            retired_texture_views: std::sync::Mutex::new(Vec::new()),
        });
        let external_fence_fd = enabled_extensions
            .contains(&khr::external_fence_fd::NAME)
            .then(|| Arc::new(khr::external_fence_fd::Device::new(instance, device.handle())));
        let external_semaphore_fd = enabled_extensions
            .contains(&khr::external_semaphore_fd::NAME)
            .then(|| Arc::new(khr::external_semaphore_fd::Device::new(instance, device.handle())));
        let debug_utils = if cfg!(debug_assertions)
            && physical_device
                .instance()
                .enabled_extensions()
                .any(|name| name == ext::debug_utils::NAME)
        {
            Some(Arc::new(ext::debug_utils::Device::new(instance, device.handle())))
        } else {
            None
        };

        if cfg!(debug_assertions) && debug_utils.is_none() {
            trace!("vulkan debug markers are unavailable (VK_EXT_debug_utils not enabled)");
        }
        let debug_markers_enabled = debug_utils.is_some();

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        // SAFETY: Device is valid and create info references live data.
        let command_pool = match unsafe { device.handle().create_command_pool(&pool_info, None) } {
            Ok(pool) => pool,
            Err(err) => {
                return Err(err.into());
            }
        };

        Ok(DeviceState {
            physical_device: physical_device.clone(),
            enabled_extensions,
            capabilities,
            queue_family_index,
            queue,
            command_pool,
            reusable_command_buffers: Vec::new(),
            in_flight_submissions: VecDeque::new(),
            pending_waits: Vec::new(),
            next_submission_id: 0,
            device,
            external_fence_fd,
            external_semaphore_fd,
            debug_utils,
            diagnostics: DeviceDiagnostics {
                debug_markers_enabled,
                ..DeviceDiagnostics::default()
            },
        })
    }

    pub(crate) fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub(crate) fn capabilities(&self) -> DeviceCapabilities {
        self.capabilities
    }

    pub(crate) fn enabled_extensions(&self) -> &[&'static CStr] {
        &self.enabled_extensions
    }

    pub(crate) fn physical_device(&self) -> &PhysicalDevice {
        &self.physical_device
    }

    pub(crate) fn device_handle(&self) -> &ash::Device {
        self.device.handle()
    }

    pub(crate) fn shared_device(&self) -> Arc<DeviceHandle> {
        self.device.clone()
    }

    pub(crate) fn mark_lost(&self) {
        self.device.mark_lost();
    }

    pub(crate) fn is_lost(&self) -> bool {
        self.device.is_lost()
    }

    pub(crate) fn supports_sync_file_import(&self) -> bool {
        self.capabilities.sync_file_semaphore_import() && self.external_semaphore_fd.is_some()
    }

    pub(crate) fn supports_sync_file_fence_import(&self) -> bool {
        self.capabilities.sync_file_import() && self.external_fence_fd.is_some()
    }

    pub(crate) fn supports_sync_file_export(&self) -> bool {
        // Render-completion sync_files are exported from a per-submission
        // binary semaphore (never from the VkFence — see VulkanFenceInner).
        self.capabilities.sync_file_semaphore_export() && self.external_semaphore_fd.is_some()
    }

    pub(crate) fn debug_markers_enabled(&self) -> bool {
        self.diagnostics.debug_markers_enabled
    }

    pub(crate) fn diagnostics(&self) -> DeviceDiagnostics {
        self.diagnostics
    }

    pub(crate) fn insert_debug_label(
        &self,
        command_buffer: vk::CommandBuffer,
        label_name: &'static CStr,
        color: [f32; 4],
    ) {
        let Some(debug_utils) = self.debug_utils.as_ref() else {
            return;
        };

        let label_info = vk::DebugUtilsLabelEXT::default()
            .label_name(label_name)
            .color(color);

        // SAFETY: Command buffer belongs to this device; debug label data points to static/living memory.
        unsafe { debug_utils.cmd_insert_debug_utils_label(command_buffer, &label_info) };
    }

    pub(crate) fn wait_on_sync_file(&self, sync_file: OwnedFd) -> Result<(), VulkanRendererError> {
        if self.device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        if !self.supports_sync_file_fence_import() {
            return Err(VulkanRendererError::NotImplemented(
                "sync_file fence import is not available on this Vulkan device",
            ));
        }

        let external_fence_fd = self
            .external_fence_fd
            .as_ref()
            .expect("checked by supports_sync_file_fence_import");
        let fence = import_sync_file_to_fence(self.device.as_ref(), external_fence_fd, sync_file)?;

        // SAFETY: Fence was created/imported on this device and is valid until we destroy it below.
        let wait_result = self
            .device
            .observe_result(unsafe { self.device.handle().wait_for_fences(&[fence], true, u64::MAX) });

        // SAFETY: Fence belongs to this device and is no longer needed after the host wait attempt.
        self.device
            .destroy_with(|device| unsafe { device.destroy_fence(fence, None) });

        wait_result.map_err(Into::into)
    }

    pub(crate) fn queue_wait_on_sync_file(&mut self, sync_file: OwnedFd) -> Result<(), VulkanRendererError> {
        self.queue_wait_on_sync_file_with_stage(sync_file, vk::PipelineStageFlags::ALL_COMMANDS)
    }

    pub(crate) fn queue_wait_on_sync_file_with_stage(
        &mut self,
        sync_file: OwnedFd,
        wait_stage_mask: vk::PipelineStageFlags,
    ) -> Result<(), VulkanRendererError> {
        if self.device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        if !self.supports_sync_file_import() {
            return Err(VulkanRendererError::NotImplemented(
                "sync_file semaphore import is not available on this Vulkan device",
            ));
        }

        let external_semaphore_fd = self
            .external_semaphore_fd
            .as_ref()
            .expect("checked by supports_sync_file_import");
        let semaphore =
            import_sync_file_to_semaphore(self.device.as_ref(), external_semaphore_fd, sync_file)?;

        if wait_stage_mask.is_empty() {
            // SAFETY: Semaphore was just imported on this device and was never submitted.
            // Routed through the teardown accessor so a lost device is not touched.
            self.device
                .destroy_with(|device| unsafe { device.destroy_semaphore(semaphore, None) });
            return Err(VulkanRendererError::TemporaryFailure(
                "sync_file wait stage mask must not be empty",
            ));
        }

        self.pending_waits.push((semaphore, wait_stage_mask));
        Ok(())
    }

    pub(crate) fn take_pending_wait_semaphores(&mut self) -> usize {
        let count = self.pending_waits.len();
        self.clear_pending_wait_semaphores();
        count
    }

    pub(crate) fn clear_pending_wait_semaphores(&mut self) {
        let pending = std::mem::take(&mut self.pending_waits);
        // On a lost device these never-submitted semaphores cannot be destroyed without
        // touching the dead driver; drop the handles and skip the destroy.
        self.device.destroy_with(|device| {
            for (semaphore, _) in pending {
                // SAFETY: Semaphore belongs to this device and is not in-flight because it was never submitted.
                unsafe { device.destroy_semaphore(semaphore, None) };
            }
        });
    }

    #[instrument(level = "trace", skip(self))]
    #[profiling::function]
    pub(crate) fn acquire_command_buffer(&mut self) -> Result<vk::CommandBuffer, VulkanRendererError> {
        if self.device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        self.reclaim_completed_submissions()?;

        if let Some(command_buffer) = self.reusable_command_buffers.pop() {
            return Ok(command_buffer);
        }

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        // SAFETY: Device and command pool are valid, allocation info references live data.
        let command_buffers = unsafe { self.device.handle().allocate_command_buffers(&alloc_info) }?;
        command_buffers
            .into_iter()
            .next()
            .ok_or(VulkanRendererError::TemporaryFailure(
                "Vulkan did not return a command buffer allocation",
            ))
    }

    pub(crate) fn submit(
        &mut self,
        command_buffer: vk::CommandBuffer,
    ) -> Result<SubmissionId, VulkanRendererError> {
        self.submit_with_framebuffers(command_buffer, Vec::new())
    }

    pub(crate) fn submit_with_framebuffers(
        &mut self,
        command_buffer: vk::CommandBuffer,
        framebuffers: Vec<vk::Framebuffer>,
    ) -> Result<SubmissionId, VulkanRendererError> {
        let (id, _) = self.submit_with_resources_and_fence(command_buffer, framebuffers, Vec::new())?;
        Ok(id)
    }

    pub(crate) fn submit_with_resources(
        &mut self,
        command_buffer: vk::CommandBuffer,
        framebuffers: Vec<vk::Framebuffer>,
        retained_images: Vec<Arc<ImportedDmabufImage>>,
    ) -> Result<SubmissionId, VulkanRendererError> {
        let (id, _) = self.submit_with_resources_and_fence(command_buffer, framebuffers, retained_images)?;
        Ok(id)
    }

    #[instrument(level = "trace", skip(self, command_buffer, framebuffers))]
    #[profiling::function]
    pub(crate) fn submit_with_framebuffers_and_fence(
        &mut self,
        command_buffer: vk::CommandBuffer,
        framebuffers: Vec<vk::Framebuffer>,
    ) -> Result<(SubmissionId, VulkanFence), VulkanRendererError> {
        self.submit_with_resources_and_fence(command_buffer, framebuffers, Vec::new())
    }

    pub(crate) fn submit_with_resources_and_fence(
        &mut self,
        command_buffer: vk::CommandBuffer,
        framebuffers: Vec<vk::Framebuffer>,
        retained_images: Vec<Arc<ImportedDmabufImage>>,
    ) -> Result<(SubmissionId, VulkanFence), VulkanRendererError> {
        self.submit_tracked(
            command_buffer,
            framebuffers,
            retained_images,
            Vec::new(),
            SubmissionKind::Render,
        )
    }

    /// Submit a staging upload without consuming render acquire waits or
    /// blocking the worker thread. Queue order makes every later render see
    /// the upload; the tracked VkFence owns the staging allocation and target
    /// image until execution really completes.
    pub(crate) fn submit_upload(
        &mut self,
        command_buffer: vk::CommandBuffer,
        retained_image: Arc<ImportedDmabufImage>,
        staging: TransientBufferAllocation,
    ) -> Result<SubmissionId, VulkanRendererError> {
        let (id, _) = self.submit_tracked(
            command_buffer,
            Vec::new(),
            vec![retained_image],
            vec![staging],
            SubmissionKind::Upload,
        )?;
        self.diagnostics.upload_submissions = self.diagnostics.upload_submissions.saturating_add(1);
        Ok(id)
    }

    #[instrument(
        level = "trace",
        skip(self, command_buffer, framebuffers, retained_images, transient_buffers)
    )]
    #[profiling::function]
    fn submit_tracked(
        &mut self,
        command_buffer: vk::CommandBuffer,
        framebuffers: Vec<vk::Framebuffer>,
        retained_images: Vec<Arc<ImportedDmabufImage>>,
        transient_buffers: Vec<TransientBufferAllocation>,
        kind: SubmissionKind,
    ) -> Result<(SubmissionId, VulkanFence), VulkanRendererError> {
        if self.device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        let submit_started_at = Instant::now();
        let fence = VulkanFence::create(self.shared_device())?;

        // Render-completion export rides a dedicated binary semaphore signaled
        // by this submission, exported exactly once immediately after submit
        // while the signal operation is provably pending. Exporting the
        // VkFence instead (vkGetFenceFdKHR has move semantics) was observed
        // racing fence completion on NVIDIA, yielding valid fds bound to a
        // consumed payload — sync_files that never signal.
        let export_semaphore = if kind.exports_completion() && self.supports_sync_file_export() {
            match self.create_export_semaphore() {
                Ok(semaphore) => Some(semaphore),
                Err(err) => {
                    warn!(
                        ?err,
                        "failed to create export semaphore; submission completes without an exportable sync_file"
                    );
                    None
                }
            }
        } else {
            None
        };

        let pending_waits = if kind.consumes_pending_waits() {
            std::mem::take(&mut self.pending_waits)
        } else {
            Vec::new()
        };
        let (wait_semaphores, wait_dst_stage_mask): (Vec<vk::Semaphore>, Vec<vk::PipelineStageFlags>) =
            pending_waits.into_iter().unzip();

        let command_buffers = [command_buffer];
        let mut submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
        if !wait_semaphores.is_empty() {
            submit = submit
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_dst_stage_mask);
        }
        let signal_semaphores = export_semaphore.map(|semaphore| [semaphore]);
        if let Some(signal_semaphores) = signal_semaphores.as_ref() {
            submit = submit.signal_semaphores(signal_semaphores);
        }
        let submit_info = [submit];

        // SAFETY: Queue, fence, and command buffers are valid; host-side synchronization is upheld by
        // requiring &mut self for submissions.
        if let Err(err) = self.device.observe_result(unsafe {
            self.device
                .handle()
                .queue_submit(self.queue, &submit_info, fence.handle())
        }) {
            // Cleanup runs through the teardown accessor: once the device is lost these destroys
            // must not touch the dead driver, otherwise they fault on NVIDIA.
            self.device.destroy_with(|device| {
                for semaphore in wait_semaphores {
                    // SAFETY: Semaphore belongs to this device and the submission did not succeed.
                    unsafe { device.destroy_semaphore(semaphore, None) };
                }
                if let Some(semaphore) = export_semaphore {
                    // SAFETY: Semaphore belongs to this device and was never submitted.
                    unsafe { device.destroy_semaphore(semaphore, None) };
                }
                for framebuffer in framebuffers {
                    // SAFETY: Framebuffer belongs to this device and is not referenced by a failed submission.
                    unsafe { device.destroy_framebuffer(framebuffer, None) };
                }
            });
            return Err(err.into());
        }

        if let Some(semaphore) = export_semaphore {
            match self.export_semaphore_sync_file(semaphore) {
                Ok(fd) => fence.set_exported_sync_file(fd),
                Err(err) => {
                    // The semaphore stays in the submission for deferred
                    // destruction; callers see a non-exportable sync point and
                    // fall back to genuine host waits (the fence is truthful).
                    warn!(?err, "failed to export submission sync_file from semaphore");
                }
            }
        }

        let submit_cpu_ns = duration_to_ns(submit_started_at.elapsed());
        self.diagnostics.total_submissions = self.diagnostics.total_submissions.saturating_add(1);
        self.diagnostics.total_submit_cpu_ns =
            self.diagnostics.total_submit_cpu_ns.saturating_add(submit_cpu_ns);
        self.diagnostics.max_submit_cpu_ns = self.diagnostics.max_submit_cpu_ns.max(submit_cpu_ns);

        let id = SubmissionId(self.next_submission_id);
        self.next_submission_id = self.next_submission_id.wrapping_add(1);
        self.device.store_upcoming_submission(self.next_submission_id);
        self.device.mark_submission_pending();
        self.in_flight_submissions.push_back(InFlightSubmission {
            id,
            fence: fence.clone(),
            export_semaphore,
            command_buffer,
            framebuffers,
            retained_images,
            transient_buffers,
            wait_semaphores,
            submitted_at: submit_started_at,
        });
        trace!(
            submission = ?id,
            submit_cpu_ns,
            in_flight = self.in_flight_submissions.len(),
            "submitted vulkan command buffer"
        );

        Ok((id, fence))
    }

    #[instrument(level = "trace", skip(self, command_buffer))]
    #[profiling::function]
    pub(crate) fn submit_blocking(
        &mut self,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), VulkanRendererError> {
        if self.device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        let submit_started_at = Instant::now();
        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: Device is valid and create info references no borrowed resources.
        let fence = self
            .device
            .observe_result(unsafe { self.device.handle().create_fence(&fence_info, None) })?;

        // Blocking submission remains only for synchronous GPU-to-CPU
        // readback. It must not consume `pending_waits`: those waits belong to
        // the next render submission, the first pass that samples the fenced
        // client buffers. Draining them here would both host-block readback on
        // unrelated producers and strip synchronization from the guarded
        // render.
        let command_buffers = [command_buffer];
        let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];

        // SAFETY: Queue, fence, and command buffers are valid; queue access is serialized by `&mut self`.
        if let Err(err) = self
            .device
            .observe_result(unsafe { self.device.handle().queue_submit(self.queue, &submit_info, fence) })
        {
            self.device.destroy_with(|device| {
                // SAFETY: Fence belongs to this device and is not in-flight after failed submission.
                unsafe { device.destroy_fence(fence, None) };
            });
            let _ = self.discard_command_buffer(command_buffer);
            return Err(err.into());
        }

        // SAFETY: Fence belongs to this device and was submitted by the queue_submit call above.
        let wait_result = self
            .device
            .observe_result(unsafe { self.device.handle().wait_for_fences(&[fence], true, u64::MAX) });
        self.device.destroy_with(|device| {
            // SAFETY: Fence belongs to this device and is no longer needed after wait completes/errors.
            unsafe { device.destroy_fence(fence, None) };
        });
        wait_result?;

        // SAFETY: Command buffer belongs to this command pool and execution completed after host wait.
        self.device.observe_result(unsafe {
            self.device
                .handle()
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
        })?;
        self.reusable_command_buffers.push(command_buffer);
        let submit_cpu_ns = duration_to_ns(submit_started_at.elapsed());
        self.diagnostics.total_submissions = self.diagnostics.total_submissions.saturating_add(1);
        self.diagnostics.blocking_submissions = self.diagnostics.blocking_submissions.saturating_add(1);
        self.diagnostics.total_submit_cpu_ns =
            self.diagnostics.total_submit_cpu_ns.saturating_add(submit_cpu_ns);
        self.diagnostics.max_submit_cpu_ns = self.diagnostics.max_submit_cpu_ns.max(submit_cpu_ns);
        Ok(())
    }

    pub(crate) fn discard_command_buffer(
        &mut self,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), VulkanRendererError> {
        if self.device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        // SAFETY: Command buffer belongs to `self.command_pool` and is not in-flight because
        // it was never submitted.
        self.device.observe_result(unsafe {
            self.device
                .handle()
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
        })?;
        self.reusable_command_buffers.push(command_buffer);
        Ok(())
    }

    pub(crate) fn in_flight_submission_count(&self) -> usize {
        self.in_flight_submissions.len()
    }

    /// Create the binary semaphore that carries a submission's exportable
    /// SYNC_FD payload.
    fn create_export_semaphore(&self) -> Result<vk::Semaphore, VulkanRendererError> {
        if self.device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        let mut export_info = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut export_info);
        // SAFETY: Device is valid and create info references live memory.
        let semaphore = self
            .device
            .observe_result(unsafe { self.device.handle().create_semaphore(&create_info, None) })?;
        Ok(semaphore)
    }

    /// Export the SYNC_FD from a submission's export semaphore. Must be
    /// called immediately after the successful `vkQueueSubmit` that signals
    /// the semaphore, while the signal operation is pending — the export has
    /// move semantics and binds the fd to that pending operation.
    fn export_semaphore_sync_file(&self, semaphore: vk::Semaphore) -> Result<OwnedFd, VulkanRendererError> {
        let Some(loader) = self.external_semaphore_fd.as_ref() else {
            return Err(VulkanRendererError::ContextLost(
                "external semaphore fd extension unavailable",
            ));
        };
        let get_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        // SAFETY: Semaphore belongs to the same device as `loader` and has a pending signal op.
        let fd = self
            .device
            .observe_result(unsafe { loader.get_semaphore_fd(&get_info) })?;
        if fd < 0 {
            warn!(fd, "semaphore sync_file export returned an invalid fd");
            return Err(VulkanRendererError::ContextLost(
                "semaphore sync_file export returned an invalid fd",
            ));
        }
        // SAFETY: Vulkan returns ownership of a valid fd on success; negative values handled above.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    #[instrument(level = "trace", skip(self))]
    #[profiling::function]
    pub(crate) fn reclaim_completed_submissions(&mut self) -> Result<(), VulkanRendererError> {
        if self.device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        loop {
            let Some(front) = self.in_flight_submissions.front() else {
                break;
            };

            // The caller-visible fence is never exported (the sync_file rides
            // a dedicated semaphore), so it faithfully tracks completion and
            // can be polled directly.
            let poll_fence = front.fence.handle();
            // SAFETY: Fence was created by this device and remains valid while tracked.
            let signaled = match self
                .device
                .observe_result(unsafe { self.device.handle().get_fence_status(poll_fence) })
            {
                Ok(signaled) => signaled,
                Err(err) => return Err(err.into()),
            };
            if !signaled {
                break;
            }

            let completed = self
                .in_flight_submissions
                .pop_front()
                .expect("front element exists");
            self.recycle_submission(completed)?;
        }

        Ok(())
    }

    #[instrument(level = "trace", skip(self))]
    #[profiling::function]
    pub(crate) fn wait_for_all_submissions(&mut self) -> Result<(), VulkanRendererError> {
        if self.device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        while let Some(submission) = self.in_flight_submissions.pop_front() {
            // The caller-visible fence is never exported, so it faithfully
            // tracks completion and can be waited on directly.
            // SAFETY: Fence was created by this device and remains valid while tracked.
            if let Err(err) = self.device.observe_result(unsafe {
                self.device
                    .handle()
                    .wait_for_fences(&[submission.fence.handle()], true, u64::MAX)
            }) {
                return Err(err.into());
            }
            self.recycle_submission(submission)?;
        }

        Ok(())
    }

    fn recycle_submission(&mut self, submission: InFlightSubmission) -> Result<(), VulkanRendererError> {
        let InFlightSubmission {
            id,
            fence: _fence,
            export_semaphore,
            command_buffer,
            framebuffers,
            transient_buffers,
            wait_semaphores,
            submitted_at,
            ..
        } = submission;

        if let Some(semaphore) = export_semaphore {
            // The submission completed, so no pending operation references the
            // semaphore anymore; its exported payload lives on in the fd.
            self.device.destroy_with(|device| {
                // SAFETY: Semaphore belongs to this device and is no longer in use.
                unsafe { device.destroy_semaphore(semaphore, None) };
            });
        }

        let completion_ns = duration_to_ns(submitted_at.elapsed());
        self.device.mark_submission_completed();
        self.device.note_submission_completed(id);
        self.diagnostics.reclaimed_submissions = self.diagnostics.reclaimed_submissions.saturating_add(1);
        self.diagnostics.total_completion_ns =
            self.diagnostics.total_completion_ns.saturating_add(completion_ns);
        self.diagnostics.max_completion_ns = self.diagnostics.max_completion_ns.max(completion_ns);
        trace!(
            submission = ?id,
            completion_ns,
            reclaimed = self.diagnostics.reclaimed_submissions,
            "reclaimed completed vulkan submission"
        );

        // Routed through the teardown accessor: a device lost mid-drain must not have its
        // framebuffers/semaphores destroyed on the dead driver (faults on NVIDIA).
        self.device.destroy_with(|device| {
            for framebuffer in framebuffers {
                // SAFETY: The submission fence is already signaled when this method is called,
                // so command buffer execution is complete and framebuffer handles may be destroyed.
                unsafe { device.destroy_framebuffer(framebuffer, None) };
            }

            for semaphore in wait_semaphores {
                // SAFETY: Submission completion implies this semaphore is no longer referenced by the queue.
                unsafe { device.destroy_semaphore(semaphore, None) };
            }
        });

        // Each transient allocation owns its teardown capability. The fence
        // is signaled, so dropping now is the exact staging-resource edge.
        drop(transient_buffers);

        // SAFETY: Command buffer belongs to `self.command_pool` and can be reset because the associated fence
        // is known to be signaled before this method is called.
        self.device.observe_result(unsafe {
            self.device
                .handle()
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
        })?;

        self.reusable_command_buffers.push(command_buffer);
        Ok(())
    }

    fn validate_required_extensions(
        physical_device: &PhysicalDevice,
    ) -> Result<Vec<&'static CStr>, VulkanRendererError> {
        let required = Self::required_extensions(physical_device);
        let missing = required
            .iter()
            .copied()
            .filter(|extension| !physical_device.has_device_extension(extension))
            .collect::<Vec<_>>();

        if missing.is_empty() {
            let mut enabled = required;
            if physical_device.has_device_extension(khr::external_fence_fd::NAME) {
                enabled.push(khr::external_fence_fd::NAME);
            }
            if physical_device.has_device_extension(khr::external_semaphore_fd::NAME) {
                enabled.push(khr::external_semaphore_fd::NAME);
            }
            Ok(enabled)
        } else {
            Err(VulkanRendererError::MissingDeviceExtensions(missing))
        }
    }

    fn validate_required_features(physical_device: &PhysicalDevice) -> Result<(), VulkanRendererError> {
        let features = physical_device.features();

        // Phase-1 uses robust buffer access as the baseline safety feature for command/resource management.
        if features.robust_buffer_access != vk::TRUE {
            return Err(VulkanRendererError::MissingDeviceFeature("robust_buffer_access"));
        }

        Ok(())
    }

    fn select_queue_family(physical_device: &PhysicalDevice) -> Result<u32, VulkanRendererError> {
        // SAFETY: Physical device belongs to this instance and query only reads driver-provided immutable properties.
        let queue_families = unsafe {
            physical_device
                .instance()
                .handle()
                .get_physical_device_queue_family_properties(physical_device.handle())
        };

        queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| {
                family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            })
            .map(|(index, _)| index as u32)
            .ok_or(VulkanRendererError::MissingQueueFamily {
                required: vk::QueueFlags::GRAPHICS,
            })
    }

    fn query_capabilities(
        physical_device: &PhysicalDevice,
        enabled_extensions: &[&'static CStr],
    ) -> DeviceCapabilities {
        let instance = physical_device.instance().handle();
        let mut timeline = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut timeline);

        // SAFETY: `features2` points to valid writable memory and the physical device belongs to this instance.
        unsafe { instance.get_physical_device_features2(physical_device.handle(), &mut features2) };

        let (sync_file_import, sync_file_export) =
            if enabled_extensions.contains(&khr::external_fence_fd::NAME) {
                let fence_info = vk::PhysicalDeviceExternalFenceInfo::default()
                    .handle_type(vk::ExternalFenceHandleTypeFlags::SYNC_FD);
                let mut fence_properties = vk::ExternalFenceProperties::default();

                // SAFETY: `fence_properties` points to valid writable memory and `fence_info` outlives the call.
                unsafe {
                    instance.get_physical_device_external_fence_properties(
                        physical_device.handle(),
                        &fence_info,
                        &mut fence_properties,
                    )
                };

                (
                    fence_properties
                        .external_fence_features
                        .contains(vk::ExternalFenceFeatureFlags::IMPORTABLE),
                    fence_properties
                        .external_fence_features
                        .contains(vk::ExternalFenceFeatureFlags::EXPORTABLE),
                )
            } else {
                (false, false)
            };

        let (sync_file_semaphore_import, sync_file_semaphore_export) =
            if enabled_extensions.contains(&khr::external_semaphore_fd::NAME) {
                let semaphore_info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
                let mut semaphore_properties = vk::ExternalSemaphoreProperties::default();

                // SAFETY: `semaphore_properties` points to valid writable memory and `semaphore_info` outlives the call.
                unsafe {
                    instance.get_physical_device_external_semaphore_properties(
                        physical_device.handle(),
                        &semaphore_info,
                        &mut semaphore_properties,
                    )
                };

                (
                    semaphore_properties
                        .external_semaphore_features
                        .contains(vk::ExternalSemaphoreFeatureFlags::IMPORTABLE),
                    semaphore_properties
                        .external_semaphore_features
                        .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE),
                )
            } else {
                (false, false)
            };

        DeviceCapabilities {
            timeline_semaphore: timeline.timeline_semaphore == vk::TRUE,
            sync_file_import,
            sync_file_export,
            sync_file_semaphore_import,
            sync_file_semaphore_export,
        }
    }
}

#[cfg(test)]
mod submission_kind_tests {
    use super::SubmissionKind;

    #[test]
    fn render_and_upload_submission_contracts_cannot_be_mixed() {
        assert!(SubmissionKind::Render.consumes_pending_waits());
        assert!(SubmissionKind::Render.exports_completion());
        assert!(!SubmissionKind::Upload.consumes_pending_waits());
        assert!(!SubmissionKind::Upload.exports_completion());
    }
}

fn duration_to_ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

impl Drop for DeviceState {
    fn drop(&mut self) {
        // Vulkan permits teardown after device loss, but this driver path has been observed
        // to fault in object destroys. Once loss is marked, intentionally leak device children
        // rather than calling back into the wedged driver during Drop.
        if self.device.is_lost() {
            return;
        }

        if let Err(err) = self.wait_for_all_submissions() {
            warn!(
                ?err,
                "failed to drain Vulkan submissions during renderer teardown"
            );
        }
        if self.device.is_lost() {
            return;
        }
        let pending_waits = std::mem::take(&mut self.pending_waits);

        // `wait_for_all_submissions` marks the device lost on `ERROR_DEVICE_LOST`; re-check so a
        // loss observed mid-drain still skips the queue wait and the destroys below.
        self.device.destroy_with(|device| {
            // SAFETY: Synchronization for queue operations is handled by `&mut self` in all queue-touching APIs.
            if let Err(err) = self
                .device
                .observe_result(unsafe { device.queue_wait_idle(self.queue) })
            {
                warn!(
                    ?err,
                    "failed to wait for Vulkan queue idle during renderer teardown"
                );
                if self.device.is_lost() {
                    return;
                }
            }

            for (semaphore, _) in pending_waits {
                // SAFETY: Semaphore belongs to this device and was never submitted.
                unsafe { device.destroy_semaphore(semaphore, None) };
            }

            // SAFETY: Command pool belongs to this device and may be destroyed after queue idle.
            unsafe { device.destroy_command_pool(self.command_pool, None) };
        });
    }
}

#[cfg(test)]
mod tests {
    use super::DeviceState;
    use crate::backend::vulkan::{version::Version, Instance, PhysicalDevice};

    /// The Tier-3 device-validity invariant: once a device is observed lost, the single
    /// teardown accessor stops handing out the device so every destroy/wait becomes a no-op.
    /// Healthy state keeps the device available. Skips silently when no GPU is present.
    #[test]
    fn handle_for_destroy_is_none_after_mark_lost() {
        let instance = match Instance::new(Version::VERSION_1_3, None) {
            Ok(instance) => instance,
            Err(_) => return,
        };
        let physical_device = match PhysicalDevice::enumerate(&instance) {
            Ok(mut iter) => match iter.next() {
                Some(phd) => phd,
                None => return,
            },
            Err(_) => return,
        };
        let device = match DeviceState::new(&physical_device) {
            Ok(device) => device,
            Err(_) => return,
        };

        let handle = device.shared_device();
        assert!(!handle.is_lost(), "device starts healthy");
        assert!(
            handle.handle_for_destroy().is_some(),
            "healthy device is available for teardown destroys"
        );

        handle.mark_lost();

        assert!(handle.is_lost(), "mark_lost is observed");
        assert!(
            handle.handle_for_destroy().is_none(),
            "a lost device must not be handed out for destroys"
        );

        let mut ran = false;
        handle.destroy_with(|_| ran = true);
        assert!(!ran, "destroy_with is a no-op on a lost device");

        // The device is genuinely healthy; restore the flag so it (and the owning instance)
        // tear down through the normal destroy path instead of leaking a live VkDevice.
        handle.clear_lost_for_test();
        assert!(
            handle.handle_for_destroy().is_some(),
            "cleared flag restores availability"
        );
    }

    /// End-to-end check of the render-completion sync_file contract: the
    /// exported fd comes from the submission's dedicated export semaphore,
    /// signals when the submission completes, export is an idempotent dup,
    /// and the never-exported VkFence tracks true completion state for
    /// reclamation. Skips silently when no GPU is present.
    #[test]
    fn submission_sync_file_export_signals_on_completion() {
        use ash::vk;

        let instance = match Instance::new(Version::VERSION_1_3, None) {
            Ok(instance) => instance,
            Err(_) => return,
        };
        let physical_device = match PhysicalDevice::enumerate(&instance) {
            Ok(mut iter) => match iter.next() {
                Some(phd) => phd,
                None => return,
            },
            Err(_) => return,
        };
        let mut device = match DeviceState::new(&physical_device) {
            Ok(device) => device,
            Err(_) => return,
        };
        if !device.supports_sync_file_export() {
            return;
        }

        let command_buffer = device
            .acquire_command_buffer()
            .expect("command buffer acquisition should succeed");
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: Command buffer and device are valid; the recording is empty.
        unsafe {
            device
                .shared_device()
                .handle()
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("begin_command_buffer should succeed");
            device
                .shared_device()
                .handle()
                .end_command_buffer(command_buffer)
                .expect("end_command_buffer should succeed");
        }

        let (_id, fence) = device
            .submit_with_framebuffers_and_fence(command_buffer, Vec::new())
            .expect("submission should succeed");

        let sync_file = fence
            .export_sync_file()
            .expect("submission must carry an exportable sync_file");
        assert!(
            fence.export_sync_file().is_some(),
            "sync_file export must be an idempotent dup, not a consuming operation"
        );

        fence.wait_vk().expect("fence wait should succeed");
        assert!(
            fence.status().unwrap_or(false),
            "the never-exported fence must report true completion state"
        );

        let mut poll_fd = [rustix::event::PollFd::new(
            &sync_file,
            rustix::event::PollFlags::IN,
        )];
        let ready = rustix::event::poll(
            &mut poll_fd,
            Some(&rustix::time::Timespec {
                tv_sec: 0,
                tv_nsec: 0,
            }),
        )
        .expect("sync_file poll should succeed");
        assert!(
            ready > 0,
            "the exported sync_file must be signaled once the submission completed"
        );

        device
            .reclaim_completed_submissions()
            .expect("reclaim should succeed");
        assert_eq!(
            device.in_flight_submission_count(),
            0,
            "the completed submission must reclaim via the un-exported fence"
        );
    }
}
