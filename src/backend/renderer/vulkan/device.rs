use std::{collections::VecDeque, ffi::CStr, fmt, os::fd::OwnedFd, sync::Arc, time::Instant};

use ash::{ext, khr, vk};
use tracing::{instrument, trace, warn};

use crate::backend::vulkan::{version::Version, PhysicalDevice};

use super::{
    sync::{import_sync_file_to_fence, import_sync_file_to_semaphore, VulkanFence},
    VulkanRendererError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SubmissionId(u64);

struct InFlightSubmission {
    id: SubmissionId,
    fence: VulkanFence,
    /// Non-exportable fence used exclusively by `reclaim_completed_submissions` to detect
    /// completion.  When `supports_sync_file_export` is true, the caller-visible `VulkanFence`
    /// may be exported as a SYNC_FD — which per the Vulkan spec resets the VkFence to
    /// unsignaled, making it unsuitable for host-side polling.  This dedicated reclaim fence
    /// is never exported and therefore always reflects the true completion state.
    reclaim_fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
    framebuffers: Vec<vk::Framebuffer>,
    wait_semaphores: Vec<vk::Semaphore>,
    submitted_at: Instant,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DeviceCapabilities {
    timeline_semaphore: bool,
    sync_file_import: bool,
    sync_file_export: bool,
    sync_file_semaphore_import: bool,
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
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct DeviceDiagnostics {
    pub(crate) total_submissions: u64,
    pub(crate) blocking_submissions: u64,
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
    reusable_reclaim_fences: Vec<vk::Fence>,
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
            .field("reusable_reclaim_fences", &self.reusable_reclaim_fences.len())
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
            .field("reclaimed_submissions", &self.diagnostics.reclaimed_submissions)
            .finish()
    }
}

pub(super) struct DeviceHandle {
    device: ash::Device,
}

impl fmt::Debug for DeviceHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceHandle")
            .field("device", &self.device.handle())
            .finish()
    }
}

impl DeviceHandle {
    pub(super) fn handle(&self) -> &ash::Device {
        &self.device
    }
}

impl Drop for DeviceHandle {
    fn drop(&mut self) {
        // SAFETY: Device destruction happens once, after all dependent resources are dropped.
        unsafe { self.device.destroy_device(None) };
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

        let device = Arc::new(DeviceHandle { device: raw_device });
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
            reusable_reclaim_fences: Vec::new(),
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

    pub(crate) fn supports_sync_file_import(&self) -> bool {
        self.capabilities.sync_file_semaphore_import() && self.external_semaphore_fd.is_some()
    }

    pub(crate) fn supports_sync_file_fence_import(&self) -> bool {
        self.capabilities.sync_file_import() && self.external_fence_fd.is_some()
    }

    pub(crate) fn supports_sync_file_export(&self) -> bool {
        self.capabilities.sync_file_export() && self.external_fence_fd.is_some()
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
        if !self.supports_sync_file_fence_import() {
            return Err(VulkanRendererError::NotImplemented(
                "sync_file fence import is not available on this Vulkan device",
            ));
        }

        let external_fence_fd = self
            .external_fence_fd
            .as_ref()
            .expect("checked by supports_sync_file_fence_import");
        let fence = import_sync_file_to_fence(self.device.handle(), external_fence_fd, sync_file)?;

        // SAFETY: Fence was created/imported on this device and is valid until we destroy it below.
        let wait_result = unsafe { self.device.handle().wait_for_fences(&[fence], true, u64::MAX) };

        // SAFETY: Fence belongs to this device and is no longer needed after the host wait attempt.
        unsafe { self.device.handle().destroy_fence(fence, None) };

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
            import_sync_file_to_semaphore(self.device.handle(), external_semaphore_fd, sync_file)?;

        if wait_stage_mask.is_empty() {
            unsafe { self.device.handle().destroy_semaphore(semaphore, None) };
            return Err(VulkanRendererError::TemporaryFailure(
                "sync_file wait stage mask must not be empty",
            ));
        }

        self.pending_waits.push((semaphore, wait_stage_mask));
        Ok(())
    }

    pub(crate) fn clear_pending_wait_semaphores(&mut self) {
        for (semaphore, _) in self.pending_waits.drain(..) {
            // SAFETY: Semaphore belongs to this device and is not in-flight because it was never submitted.
            unsafe { self.device.handle().destroy_semaphore(semaphore, None) };
        }
    }

    #[instrument(level = "trace", skip(self))]
    #[profiling::function]
    pub(crate) fn acquire_command_buffer(&mut self) -> Result<vk::CommandBuffer, VulkanRendererError> {
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
        let (id, _) = self.submit_with_framebuffers_and_fence(command_buffer, framebuffers)?;
        Ok(id)
    }

    #[instrument(level = "trace", skip(self, command_buffer, framebuffers))]
    #[profiling::function]
    pub(crate) fn submit_with_framebuffers_and_fence(
        &mut self,
        command_buffer: vk::CommandBuffer,
        framebuffers: Vec<vk::Framebuffer>,
    ) -> Result<(SubmissionId, VulkanFence), VulkanRendererError> {
        let submit_started_at = Instant::now();
        let fence = VulkanFence::create(
            self.shared_device(),
            self.external_fence_fd.clone(),
            self.supports_sync_file_export(),
        )?;

        let pending_waits = std::mem::take(&mut self.pending_waits);
        let (wait_semaphores, wait_dst_stage_mask): (Vec<vk::Semaphore>, Vec<vk::PipelineStageFlags>) =
            pending_waits.into_iter().unzip();

        let command_buffers = [command_buffer];
        let mut submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
        if !wait_semaphores.is_empty() {
            submit = submit
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_dst_stage_mask);
        }
        let submit_info = [submit];

        // SAFETY: Queue, fence, and command buffers are valid; host-side synchronization is upheld by
        // requiring &mut self for submissions.
        if let Err(err) = unsafe {
            self.device
                .handle()
                .queue_submit(self.queue, &submit_info, fence.handle())
        } {
            for semaphore in wait_semaphores {
                // SAFETY: Semaphore belongs to this device and the submission did not succeed.
                unsafe { self.device.handle().destroy_semaphore(semaphore, None) };
            }
            for framebuffer in framebuffers {
                // SAFETY: Framebuffer belongs to this device and is not referenced by a failed submission.
                unsafe { self.device.handle().destroy_framebuffer(framebuffer, None) };
            }
            return Err(err.into());
        }

        // When the caller-visible fence is exportable as SYNC_FD, the Vulkan spec mandates that
        // exporting resets the VkFence to unsignaled.  Since the DRM compositor routinely exports
        // the fence for KMS in-fencing, `get_fence_status` on the caller-visible fence would
        // always return false — preventing reclaim.  We solve this by submitting a lightweight
        // empty batch with a separate non-exportable fence that faithfully tracks completion.
        //
        // When SYNC_FD export is not supported, the caller-visible fence is never exported and
        // can be polled directly, so the reclaim fence is redundant — we use vk::Fence::null()
        // as a sentinel to skip the extra submit.
        let reclaim_fence = if self.supports_sync_file_export() {
            match self.acquire_reclaim_fence() {
                Ok(rf) => {
                    // SAFETY: Empty submit; fence signals when all prior queue work completes.
                    match unsafe { self.device.handle().queue_submit(self.queue, &[], rf) } {
                        Ok(()) => rf,
                        Err(err) => {
                            // The real work was already submitted — we cannot un-submit it.
                            // Fall back to null (polling the caller-visible fence, which may
                            // not work if exported) rather than losing track of the submission.
                            warn!(?err, "failed to submit reclaim fence; reclaim may be delayed");
                            self.recycle_reclaim_fence(rf);
                            vk::Fence::null()
                        }
                    }
                }
                Err(err) => {
                    warn!(?err, "failed to create reclaim fence; reclaim may be delayed");
                    vk::Fence::null()
                }
            }
        } else {
            vk::Fence::null()
        };

        let submit_cpu_ns = duration_to_ns(submit_started_at.elapsed());
        self.diagnostics.total_submissions = self.diagnostics.total_submissions.saturating_add(1);
        self.diagnostics.total_submit_cpu_ns =
            self.diagnostics.total_submit_cpu_ns.saturating_add(submit_cpu_ns);
        self.diagnostics.max_submit_cpu_ns = self.diagnostics.max_submit_cpu_ns.max(submit_cpu_ns);

        let id = SubmissionId(self.next_submission_id);
        self.next_submission_id = self.next_submission_id.wrapping_add(1);
        self.in_flight_submissions.push_back(InFlightSubmission {
            id,
            fence: fence.clone(),
            reclaim_fence,
            command_buffer,
            framebuffers,
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
        let submit_started_at = Instant::now();
        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: Device is valid and create info references no borrowed resources.
        let fence = unsafe { self.device.handle().create_fence(&fence_info, None) }?;

        let pending_waits = std::mem::take(&mut self.pending_waits);
        let (wait_semaphores, wait_dst_stage_mask): (Vec<vk::Semaphore>, Vec<vk::PipelineStageFlags>) =
            pending_waits.into_iter().unzip();

        let command_buffers = [command_buffer];
        let mut submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
        if !wait_semaphores.is_empty() {
            submit = submit
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_dst_stage_mask);
        }
        let submit_info = [submit];

        // SAFETY: Queue, fence, and command buffers are valid; queue access is serialized by `&mut self`.
        if let Err(err) = unsafe { self.device.handle().queue_submit(self.queue, &submit_info, fence) } {
            for semaphore in wait_semaphores {
                // SAFETY: Semaphore belongs to this device and the submission did not succeed.
                unsafe { self.device.handle().destroy_semaphore(semaphore, None) };
            }
            // SAFETY: Fence belongs to this device and is not in-flight after failed submission.
            unsafe { self.device.handle().destroy_fence(fence, None) };
            let _ = self.discard_command_buffer(command_buffer);
            return Err(err.into());
        }

        // SAFETY: Fence belongs to this device and was submitted by the queue_submit call above.
        let wait_result = unsafe { self.device.handle().wait_for_fences(&[fence], true, u64::MAX) };
        // SAFETY: Fence belongs to this device and is no longer needed after wait completes/errors.
        unsafe { self.device.handle().destroy_fence(fence, None) };
        for semaphore in wait_semaphores {
            // SAFETY: Submission completion is determined by the host fence above; semaphore can be released.
            unsafe { self.device.handle().destroy_semaphore(semaphore, None) };
        }
        wait_result?;

        // SAFETY: Command buffer belongs to this command pool and execution completed after host wait.
        unsafe {
            self.device
                .handle()
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
        }?;
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
        // SAFETY: Command buffer belongs to `self.command_pool` and is not in-flight because
        // it was never submitted.
        unsafe {
            self.device
                .handle()
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
        }?;
        self.reusable_command_buffers.push(command_buffer);
        Ok(())
    }

    pub(crate) fn in_flight_submission_count(&self) -> usize {
        self.in_flight_submissions.len()
    }

    fn acquire_reclaim_fence(&mut self) -> Result<vk::Fence, VulkanRendererError> {
        if let Some(fence) = self.reusable_reclaim_fences.pop() {
            return Ok(fence);
        }
        let create_info = vk::FenceCreateInfo::default();
        // SAFETY: Device is valid and create info references no borrowed resources.
        let fence = unsafe { self.device.handle().create_fence(&create_info, None) }?;
        Ok(fence)
    }

    fn recycle_reclaim_fence(&mut self, fence: vk::Fence) {
        // SAFETY: Fence was signaled (or never submitted) and belongs to this device.
        if let Err(err) = unsafe { self.device.handle().reset_fences(&[fence]) } {
            warn!(?err, "failed to reset reclaim fence, destroying instead");
            unsafe { self.device.handle().destroy_fence(fence, None) };
            return;
        }
        self.reusable_reclaim_fences.push(fence);
    }

    #[instrument(level = "trace", skip(self))]
    #[profiling::function]
    pub(crate) fn reclaim_completed_submissions(&mut self) -> Result<(), VulkanRendererError> {
        loop {
            let Some(front) = self.in_flight_submissions.front() else {
                break;
            };

            // When a dedicated reclaim fence exists, poll it instead of the caller-visible
            // fence — the latter may have been exported as SYNC_FD (resetting it to unsignaled).
            // When reclaim_fence is null, SYNC_FD export is not supported, so the caller-visible
            // fence is safe to poll directly.
            let poll_fence = if front.reclaim_fence != vk::Fence::null() {
                front.reclaim_fence
            } else {
                front.fence.handle()
            };
            // SAFETY: Fence was created by this device and remains valid while tracked.
            let signaled = unsafe { self.device.handle().get_fence_status(poll_fence) }?;
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
        while let Some(submission) = self.in_flight_submissions.pop_front() {
            let wait_fence = if submission.reclaim_fence != vk::Fence::null() {
                submission.reclaim_fence
            } else {
                submission.fence.handle()
            };
            // SAFETY: Fence was created by this device and remains valid while tracked.
            unsafe {
                self.device
                    .handle()
                    .wait_for_fences(&[wait_fence], true, u64::MAX)
            }?;
            self.recycle_submission(submission)?;
        }

        Ok(())
    }

    fn recycle_submission(&mut self, submission: InFlightSubmission) -> Result<(), VulkanRendererError> {
        let InFlightSubmission {
            id,
            fence: _fence,
            reclaim_fence,
            command_buffer,
            framebuffers,
            wait_semaphores,
            submitted_at,
            ..
        } = submission;

        if reclaim_fence != vk::Fence::null() {
            self.recycle_reclaim_fence(reclaim_fence);
        }

        let completion_ns = duration_to_ns(submitted_at.elapsed());
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

        for framebuffer in framebuffers {
            // SAFETY: The submission fence is already signaled when this method is called,
            // so command buffer execution is complete and framebuffer handles may be destroyed.
            unsafe { self.device.handle().destroy_framebuffer(framebuffer, None) };
        }

        for semaphore in wait_semaphores {
            // SAFETY: Submission completion implies this semaphore is no longer referenced by the queue.
            unsafe { self.device.handle().destroy_semaphore(semaphore, None) };
        }

        // SAFETY: Command buffer belongs to `self.command_pool` and can be reset because the associated fence
        // is known to be signaled before this method is called.
        unsafe {
            self.device
                .handle()
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
        }?;

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

        let sync_file_semaphore_import = if enabled_extensions.contains(&khr::external_semaphore_fd::NAME) {
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

            semaphore_properties
                .external_semaphore_features
                .contains(vk::ExternalSemaphoreFeatureFlags::IMPORTABLE)
        } else {
            false
        };

        DeviceCapabilities {
            timeline_semaphore: timeline.timeline_semaphore == vk::TRUE,
            sync_file_import,
            sync_file_export,
            sync_file_semaphore_import,
        }
    }
}

fn duration_to_ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

impl Drop for DeviceState {
    fn drop(&mut self) {
        self.clear_pending_wait_semaphores();

        if let Err(err) = self.wait_for_all_submissions() {
            warn!(
                ?err,
                "failed to drain Vulkan submissions during renderer teardown"
            );
        }

        // SAFETY: Synchronization for queue operations is handled by `&mut self` in all queue-touching APIs.
        if let Err(err) = unsafe { self.device.handle().queue_wait_idle(self.queue) } {
            warn!(
                ?err,
                "failed to wait for Vulkan queue idle during renderer teardown"
            );
        }

        // SAFETY: Command pool belongs to this device and may be destroyed after queue idle.
        unsafe { self.device.handle().destroy_command_pool(self.command_pool, None) };

        for fence in self.reusable_reclaim_fences.drain(..) {
            // SAFETY: Fence belongs to this device and is not in-flight (all submissions drained above).
            unsafe { self.device.handle().destroy_fence(fence, None) };
        }
    }
}
