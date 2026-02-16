use std::{collections::VecDeque, ffi::CStr, fmt, sync::Arc};

use ash::{ext, khr, vk};
use tracing::warn;

use crate::backend::vulkan::{version::Version, PhysicalDevice};

use super::VulkanRendererError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SubmissionId(u64);

#[derive(Debug)]
struct InFlightSubmission {
    id: SubmissionId,
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
    framebuffers: Vec<vk::Framebuffer>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DeviceCapabilities {
    timeline_semaphore: bool,
}

impl DeviceCapabilities {
    pub(crate) fn timeline_semaphore(self) -> bool {
        self.timeline_semaphore
    }
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
    next_submission_id: u64,
    device: Arc<DeviceHandle>,
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
        let capabilities = Self::query_capabilities(physical_device);
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
            next_submission_id: 0,
            device,
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
        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: Device is valid, create info references no borrowed resources.
        let fence = unsafe { self.device.handle().create_fence(&fence_info, None) }?;

        let command_buffers = [command_buffer];
        let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];

        // SAFETY: Queue, fence, and command buffers are valid; host-side synchronization is upheld by
        // requiring &mut self for submissions.
        if let Err(err) = unsafe { self.device.handle().queue_submit(self.queue, &submit_info, fence) } {
            // SAFETY: Fence was created by this device and has not been submitted on failure path.
            unsafe { self.device.handle().destroy_fence(fence, None) };
            for framebuffer in framebuffers {
                // SAFETY: Framebuffer belongs to this device and is not referenced by a failed submission.
                unsafe { self.device.handle().destroy_framebuffer(framebuffer, None) };
            }
            return Err(err.into());
        }

        let id = SubmissionId(self.next_submission_id);
        self.next_submission_id = self.next_submission_id.wrapping_add(1);
        self.in_flight_submissions.push_back(InFlightSubmission {
            id,
            fence,
            command_buffer,
            framebuffers,
        });

        Ok(id)
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

    pub(crate) fn reclaim_completed_submissions(&mut self) -> Result<(), VulkanRendererError> {
        loop {
            let Some(front) = self.in_flight_submissions.front() else {
                break;
            };

            // SAFETY: Fence was created by this device and remains valid while tracked in `in_flight_submissions`.
            let signaled = unsafe { self.device.handle().get_fence_status(front.fence) }?;
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

    pub(crate) fn wait_for_all_submissions(&mut self) -> Result<(), VulkanRendererError> {
        while let Some(submission) = self.in_flight_submissions.pop_front() {
            // SAFETY: Fence was created by this device and remains valid while tracked.
            unsafe {
                self.device
                    .handle()
                    .wait_for_fences(&[submission.fence], true, u64::MAX)
            }?;
            self.recycle_submission(submission)?;
        }

        Ok(())
    }

    fn recycle_submission(&mut self, submission: InFlightSubmission) -> Result<(), VulkanRendererError> {
        let InFlightSubmission {
            fence,
            command_buffer,
            framebuffers,
            ..
        } = submission;

        for framebuffer in framebuffers {
            // SAFETY: The submission fence is already signaled when this method is called,
            // so command buffer execution is complete and framebuffer handles may be destroyed.
            unsafe { self.device.handle().destroy_framebuffer(framebuffer, None) };
        }

        // SAFETY: Command buffer belongs to `self.command_pool` and can be reset because the associated fence
        // is known to be signaled before this method is called.
        unsafe {
            self.device
                .handle()
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
        }?;

        // SAFETY: Fence was created by this device and is no longer needed after completion.
        unsafe { self.device.handle().destroy_fence(fence, None) };
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
            Ok(required)
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

    fn query_capabilities(physical_device: &PhysicalDevice) -> DeviceCapabilities {
        let instance = physical_device.instance().handle();
        let mut timeline = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut timeline);

        // SAFETY: `features2` points to valid writable memory and the physical device belongs to this instance.
        unsafe { instance.get_physical_device_features2(physical_device.handle(), &mut features2) };

        DeviceCapabilities {
            timeline_semaphore: timeline.timeline_semaphore == vk::TRUE,
        }
    }
}

impl Drop for DeviceState {
    fn drop(&mut self) {
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
    }
}
