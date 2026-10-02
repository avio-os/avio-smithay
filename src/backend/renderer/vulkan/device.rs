use std::{
    collections::VecDeque,
    ffi::CStr,
    fmt,
    os::fd::{FromRawFd, OwnedFd},
    sync::Arc,
    time::Instant,
};

use ash::{ext, khr, vk};
use tracing::{instrument, trace, warn};

use crate::backend::{
    renderer::{sync::SyncPoint, MemoryUploadCapacityEdge},
    vulkan::{
        create_device_with_queue_priority, version::Version, PhysicalDevice, QueueGlobalPriority,
        QueuePriorityGrant, QueuePriorityOutcome,
    },
};

use super::{
    external_wait_storage::{ExternalWaitBank, ExternalWaitBatch, ImportedWaitSemaphore},
    image::{transition_image_layout, VulkanImage},
    staging::{ReservationWriter, StagingReservation, UploadArena, UploadArenaStats},
    sync::{
        import_sync_file_into_semaphore, import_sync_file_to_fence, import_sync_file_to_semaphore,
        VulkanFence,
    },
    VulkanRendererError, VulkanSubmissionSnapshot,
};

const MAX_UPLOAD_BATCH_OPERATIONS: usize = 256;
use super::submission_storage::VulkanCommandStorageLimits;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SubmissionId(pub(super) u64);

impl SubmissionId {
    #[cfg(test)]
    pub(crate) fn for_tests(id: u64) -> Self {
        Self(id)
    }
}

pub(super) struct InFlightSubmission {
    pub(super) id: SubmissionId,
    pub(super) submission_id_known: bool,
    pub(super) fence: SyncPoint,
    pub(super) native: Arc<VulkanFence>,
    pub(super) readers: Arc<super::fence_return::FenceReaders>,
    pub(super) reader_return: SyncPoint,
    /// Binary semaphore signaled by this submission whose SYNC_FD was
    /// exported right after submit (the caller-visible sync_file). The
    /// VkFence is never exported, so it faithfully tracks completion and is
    /// polled directly for reclamation; the semaphore is kept alive until
    /// the submission retires (a semaphore referenced by pending GPU work
    /// must not be destroyed) and destroyed on recycle.
    pub(super) export_semaphore: Option<vk::Semaphore>,
    pub(super) export_unconsumed: bool,
    pub(super) command_buffers: Vec<vk::CommandBuffer>,
    pub(super) framebuffers: Vec<vk::Framebuffer>,
    pub(super) retained_images: Vec<Arc<VulkanImage>>,
    pub(super) upload_sources: Vec<UploadSource>,
    pub(super) _readback: Option<Arc<super::readback::ReadbackBuffer>>,
    pub(super) wait_semaphores: Vec<ImportedWaitSemaphore>,
    pub(super) submitted_at: Instant,
}

/// Exact queue readers moved intact to the device's off-thread command
/// retirement owner. A failed wait retains this whole object, including every
/// foreign host mapping, instead of asserting that GPU access ended.
pub(super) struct RetiredCommands {
    pub(super) command_pool: vk::CommandPool,
    pub(super) submissions: VecDeque<InFlightSubmission>,
    pub(super) free_submissions: Vec<InFlightSubmission>,
    pub(super) reusable_command_buffers: Vec<vk::CommandBuffer>,
    pub(super) recording_storage: Option<Arc<super::recording_storage::RecordingStorageBank>>,
    pub(super) bank_return: Option<SyncPoint>,
    damage_scratch: Option<Arc<super::damage_scratch::DamageScratchBank>>,
    failed_recording: Option<super::recording_storage::RecordingStorageLease>,
    pending_waits: Vec<(ImportedWaitSemaphore, vk::PipelineStageFlags)>,
    pub(super) external_wait_bank: Option<ExternalWaitBank>,
    pending_uploads: PendingUploadBatch,
    upload_arena: Option<UploadArena>,
    context_state: Option<(super::pipeline::PipelineState, super::descriptor::DescriptorState)>,
    pub(super) device: Option<Arc<DeviceHandle>>,
}

impl RetiredCommands {
    pub(super) fn empty(command_pool: vk::CommandPool) -> Self {
        Self {
            command_pool,
            submissions: VecDeque::new(),
            free_submissions: Vec::new(),
            reusable_command_buffers: Vec::new(),
            recording_storage: None,
            bank_return: None,
            damage_scratch: None,
            failed_recording: None,
            pending_waits: Vec::new(),
            external_wait_bank: None,
            pending_uploads: Default::default(),
            upload_arena: None,
            context_state: None,
            device: None,
        }
    }

    pub(super) fn cold_cpu_storage(
        command_pool: vk::CommandPool,
        limits: VulkanCommandStorageLimits,
    ) -> Self {
        let mut commands = Self::empty(command_pool);
        commands.submissions = VecDeque::with_capacity(limits.submission_slots);
        commands.free_submissions = Vec::with_capacity(limits.submission_slots);
        commands.recording_storage = Some(super::recording_storage::RecordingStorageBank::cold(limits));
        commands.pending_waits = Vec::with_capacity(limits.waits_per_submission);
        commands.pending_uploads.operations = Vec::with_capacity(MAX_UPLOAD_BATCH_OPERATIONS);
        commands
    }

    /// The native VkFence is never exported/consumed. Success for every
    /// reader is the only proof that allows destruction of this command pool.
    pub(super) fn wait_complete(&self, raw: &ash::Device) -> Result<(), vk::Result> {
        for submission in &self.submissions {
            unsafe { raw.wait_for_fences(&[submission.native_fence().handle()], true, u64::MAX) }?;
        }
        Ok(())
    }

    pub(super) fn destroy(mut self, raw: &ash::Device) {
        for submission in &self.submissions {
            if let Some(semaphore) = submission.export_semaphore {
                unsafe { raw.destroy_semaphore(semaphore, None) };
            }
            for semaphore in &submission.wait_semaphores {
                if !semaphore.pooled {
                    unsafe { raw.destroy_semaphore(semaphore.handle, None) };
                }
            }
            for framebuffer in &submission.framebuffers {
                unsafe { raw.destroy_framebuffer(*framebuffer, None) };
            }
            if let Some(device) = &self.device {
                device.mark_submission_completed();
                if submission.submission_id_known {
                    device.note_submission_completed(submission.id);
                }
            }
        }
        for slot in &self.free_submissions {
            if let Some(semaphore) = slot.export_semaphore {
                unsafe {
                    raw.destroy_semaphore(semaphore, None);
                }
            }
        }
        for (semaphore, _) in &self.pending_waits {
            if !semaphore.pooled {
                unsafe { raw.destroy_semaphore(semaphore.handle, None) };
            }
        }
        unsafe { raw.destroy_command_pool(self.command_pool, None) };
        // Drop GPU readers and host mappings only after their exact fences and
        // command-pool destruction; device ownership is released last.
        self.submissions.clear();
        self.pending_uploads.operations.clear();
        self.upload_arena.take();
        if let Some(recording) = self.failed_recording.as_mut() {
            for framebuffer in recording.submitted_framebuffers.drain(..) {
                unsafe {
                    raw.destroy_framebuffer(framebuffer, None);
                }
            }
        }
        self.failed_recording.take();
        self.recording_storage.take();
        self.context_state.take();
        self.external_wait_bank.take();
        self.device.take();
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
    MemoryUploadBatch,
    Readback,
}

impl SubmissionKind {
    fn consumes_pending_waits(self) -> bool {
        matches!(self, Self::Render)
    }

    fn exports_completion(self) -> bool {
        !matches!(self, Self::Readback)
    }
}

pub(crate) enum BlockingSubmitError {
    NotSubmitted(VulkanRendererError),
    Submitted(VulkanRendererError),
}

pub(super) enum UploadSource {
    Staging(StagingReservation),
    #[cfg(feature = "wayland_frontend")]
    Host(Arc<super::host_memory::HostBuffer>),
}

impl UploadSource {
    fn release(&self, arena: &mut UploadArena) {
        if let Self::Staging(reservation) = self {
            arena.release(*reservation);
        }
    }
}

struct PendingUpload {
    image: Arc<VulkanImage>,
    source: UploadSource,
    bytes: usize,
    region: vk::BufferImageCopy,
    old_layout: vk::ImageLayout,
}

pub(crate) struct ImageUpload<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) source_offset: usize,
    pub(crate) source_stride: usize,
    pub(crate) row_bytes: usize,
    pub(crate) rows: usize,
    pub(crate) region: vk::BufferImageCopy,
}

impl ImageUpload<'_> {
    fn byte_len(&self) -> Result<usize, VulkanRendererError> {
        self.row_bytes
            .checked_mul(self.rows)
            .ok_or(VulkanRendererError::InvalidMemoryUpload(
                "upload byte count overflowed",
            ))
    }
}

#[derive(Default)]
struct PendingUploadBatch {
    operations: Vec<PendingUpload>,
    bytes: usize,
}

impl PendingUploadBatch {
    fn layout_after_pending(&self, image_id: u64) -> Option<vk::ImageLayout> {
        self.operations
            .iter()
            .rev()
            .find(|operation| operation.image.id() == image_id)
            .map(|_| vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
    }

    /// Drop every operation whose image only this batch still holds, and
    /// return its staging to `arena`.
    ///
    /// Such an image has no texture, no frame recording, no blit and no
    /// submission that could sample it, so its copy is dead work. Nothing
    /// was submitted for it: the GPU has never read the image or the
    /// staging span, and dropping the last reference destroys the image
    /// unused. Without this, a renderer that queues uploads but submits
    /// nothing (an output scanned out without composition while a cursor
    /// bitmap changes) keeps every superseded image and its staging until
    /// the batch is full, and the next upload has to wait on a capacity
    /// edge.
    fn drop_unsampleable(&mut self, arena: &mut UploadArena) {
        let mut index = 0;
        while let Some(operation) = self.operations.get(index) {
            let held_here = self
                .operations
                .iter()
                .filter(|other| Arc::ptr_eq(&other.image, &operation.image))
                .count();
            if Arc::strong_count(&operation.image) > held_here {
                index += 1;
                continue;
            }
            // Every operation on this image is at or after `index`: an
            // earlier one would have been dropped with the same decision.
            let image = Arc::clone(&operation.image);
            let bytes = &mut self.bytes;
            self.operations.retain(|other| {
                if !Arc::ptr_eq(&other.image, &image) {
                    return true;
                }
                *bytes = bytes.saturating_sub(other.bytes);
                if let UploadSource::Staging(reservation) = &other.source {
                    arena.release(*reservation);
                }
                false
            });
        }
    }
}

struct RecordedUploadBatch {
    command_buffer: vk::CommandBuffer,
    batch: PendingUploadBatch,
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
    pub(crate) upload_batches: u64,
    pub(crate) upload_operations: u64,
    pub(crate) upload_bytes: u64,
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
    command_retirement: Option<Box<super::retirement::RetirementNode<RetiredCommands>>>,
    reusable_command_buffers: Vec<vk::CommandBuffer>,
    in_flight_submissions: VecDeque<InFlightSubmission>,
    free_submissions: Vec<InFlightSubmission>,
    command_limits: VulkanCommandStorageLimits,
    recording_storage: Arc<super::recording_storage::RecordingStorageBank>,
    bank_return: SyncPoint,
    damage_scratch: Arc<super::damage_scratch::DamageScratchBank>,
    completion_unknown: bool,
    failed_recording: Option<super::recording_storage::RecordingStorageLease>,
    upload_arena: UploadArena,
    owner_upload_structural_bytes: usize,
    owner_upload_exception_bytes: usize,
    owner_upload_extent_exceptions: u64,
    pending_uploads: PendingUploadBatch,
    pending_waits: Vec<(ImportedWaitSemaphore, vk::PipelineStageFlags)>,
    external_wait_bank: Option<ExternalWaitBank>,
    external_wait_batch: ExternalWaitBatch,
    external_wait_pressure: usize,
    next_submission_id: u64,
    device: Arc<DeviceHandle>,
    #[cfg(feature = "wayland_frontend")]
    host_memory: super::host_memory::HostMemorySupport,
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
            .field("queue_priority", &self.queue_priority())
            .field("command_pool", &self.command_pool)
            .field("reusable_command_buffers", &self.reusable_command_buffers.len())
            .field("in_flight_submissions", &self.in_flight_submissions.len())
            .field("upload_arena", &self.upload_arena)
            .field(
                "pending_upload_operations",
                &self.pending_uploads.operations.len(),
            )
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
            .field("upload_batches", &self.diagnostics.upload_batches)
            .field("upload_operations", &self.diagnostics.upload_operations)
            .field("reclaimed_submissions", &self.diagnostics.reclaimed_submissions)
            .finish()
    }
}

pub(super) use super::device_handle::DeviceHandle;

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

    #[cfg(test)]
    pub(crate) fn new(physical_device: &PhysicalDevice) -> Result<Self, VulkanRendererError> {
        Self::with_queue_priority(physical_device, None)
    }

    /// Creates the renderer's logical device. Its one queue asks for `requested_priority`, if
    /// any; a driver refusal falls back once to the default priority (see
    /// [`create_device_with_queue_priority`]).
    pub(crate) fn with_queue_priority(
        physical_device: &PhysicalDevice,
        requested_priority: Option<QueueGlobalPriority>,
    ) -> Result<Self, VulkanRendererError> {
        let mut enabled_extensions = Self::validate_required_extensions(physical_device)?;
        let capabilities = Self::query_capabilities(physical_device, &enabled_extensions);
        Self::validate_required_features(physical_device)?;

        let queue_family_index = Self::select_queue_family(physical_device)?;
        let queue_weights = [1.0f32];
        let features = vk::PhysicalDeviceFeatures {
            robust_buffer_access: vk::TRUE,
            ..Default::default()
        };

        let instance = physical_device.instance().handle();
        let (raw_device, queue_priority) =
            create_device_with_queue_priority(physical_device, requested_priority, |request| {
                let mut extension_ptrs = enabled_extensions
                    .iter()
                    .map(|ext| ext.as_ptr())
                    .collect::<Vec<_>>();
                let mut global_priority = request.map(|request| request.create_info());
                let mut queue_info = vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .queue_priorities(&queue_weights);
                if let (Some(request), Some(global_priority)) = (request, global_priority.as_mut()) {
                    extension_ptrs.push(request.extension().as_ptr());
                    queue_info = queue_info.push_next(global_priority);
                }
                let queue_infos = [queue_info];
                let create_info = vk::DeviceCreateInfo::default()
                    .enabled_extension_names(&extension_ptrs)
                    .enabled_features(&features)
                    .queue_create_infos(&queue_infos);
                // SAFETY: The physical device belongs to this instance and all pointers in
                // create_info are valid for the duration of this call.
                unsafe { instance.create_device(physical_device.handle(), &create_info, None) }
            })?;
        if queue_priority.outcome == QueuePriorityOutcome::Granted {
            if let Some(extension) = physical_device.queue_global_priority_extension() {
                enabled_extensions.push(extension);
            }
        }

        let queue = {
            // SAFETY: Queue family/index are valid for this device by construction in select_queue_family.
            unsafe { raw_device.get_device_queue(queue_family_index, 0) }
        };

        let device = Arc::new(DeviceHandle::new(
            raw_device,
            physical_device.instance().clone(),
            queue_priority,
        )?);
        Self::from_parts(
            physical_device,
            enabled_extensions,
            capabilities,
            queue_family_index,
            queue,
            device,
            VulkanCommandStorageLimits::default(),
        )
    }

    pub(crate) fn from_origin(origin: &super::VulkanDeviceOrigin) -> Result<Self, VulkanRendererError> {
        Self::from_origin_with_limits(origin, VulkanCommandStorageLimits::default())
    }
    pub(crate) fn from_origin_with_limits(
        origin: &super::VulkanDeviceOrigin,
        limits: VulkanCommandStorageLimits,
    ) -> Result<Self, VulkanRendererError> {
        Self::from_parts(
            &origin.physical_device,
            origin.enabled_extensions.clone(),
            origin.capabilities,
            origin.queue_family_index,
            origin.queue,
            origin.device.clone(),
            limits,
        )
    }

    fn from_parts(
        physical_device: &PhysicalDevice,
        enabled_extensions: Vec<&'static CStr>,
        capabilities: DeviceCapabilities,
        queue_family_index: u32,
        queue: vk::Queue,
        device: Arc<DeviceHandle>,
        command_limits: VulkanCommandStorageLimits,
    ) -> Result<Self, VulkanRendererError> {
        let instance = physical_device.instance().handle();
        #[cfg(feature = "wayland_frontend")]
        let host_memory = super::host_memory::HostMemorySupport::new(
            physical_device,
            device.handle(),
            enabled_extensions.contains(&ext::external_memory_host::NAME),
        );
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

        // Host-visible staging is allocated lazily on the first memory upload.
        // Renderers that only import DMA-BUFs retain no idle upload allocation.
        let upload_arena = UploadArena::new(physical_device);

        let mut prepared = super::VulkanCommandStorage::prepare(
            device.clone(),
            queue_family_index,
            capabilities.sync_file_semaphore_export(),
            command_limits,
        )?;
        let command_limits = prepared.limits;
        let mut command_retirement = prepared.node.take().expect("prepared command bank");
        let commands = command_retirement.value_mut();
        let command_pool = commands.command_pool;
        let reusable_command_buffers = std::mem::take(&mut commands.reusable_command_buffers);
        let free_submissions = std::mem::take(&mut commands.free_submissions);
        let in_flight_submissions = std::mem::take(&mut commands.submissions);
        let recording_storage = commands.recording_storage.take().unwrap();
        let bank_return = commands.bank_return.take().unwrap();
        let pending_uploads = std::mem::take(&mut commands.pending_uploads);
        let pending_waits = std::mem::take(&mut commands.pending_waits);
        commands.device.take();
        Ok(DeviceState {
            physical_device: physical_device.clone(),
            enabled_extensions,
            capabilities,
            queue_family_index,
            queue,
            command_pool,
            command_retirement: Some(command_retirement),
            reusable_command_buffers,
            in_flight_submissions,
            free_submissions,
            recording_storage,
            bank_return,
            damage_scratch: super::damage_scratch::DamageScratchBank::cold(0, 0)?,
            command_limits,
            completion_unknown: false,
            failed_recording: None,
            upload_arena,
            owner_upload_structural_bytes: 0,
            owner_upload_exception_bytes: 0,
            owner_upload_extent_exceptions: 0,
            pending_uploads,
            pending_waits,
            external_wait_bank: None,
            external_wait_batch: Default::default(),
            external_wait_pressure: 0,
            next_submission_id: 0,
            device,
            #[cfg(feature = "wayland_frontend")]
            host_memory,
            external_fence_fd,
            external_semaphore_fd,
            debug_utils,
            diagnostics: DeviceDiagnostics {
                debug_markers_enabled,
                ..DeviceDiagnostics::default()
            },
        })
    }

    pub(super) fn retain_context_state(
        &mut self,
        pipelines: super::pipeline::PipelineState,
        descriptors: super::descriptor::DescriptorState,
    ) {
        self.command_retirement
            .as_mut()
            .expect("live renderer has command retirement custody")
            .value_mut()
            .context_state = Some((pipelines, descriptors));
    }

    pub(super) fn queue(&self) -> vk::Queue {
        self.queue
    }

    pub(crate) fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub(crate) fn queue_priority(&self) -> QueuePriorityGrant {
        self.device.queue_priority()
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

    #[cfg(test)]
    pub(super) fn wait_retirement_drained(&self) {
        self.device.wait_retirement_drained();
    }

    pub(crate) fn shared_device(&self) -> Arc<DeviceHandle> {
        self.device.clone()
    }

    #[cfg(feature = "wayland_frontend")]
    pub(super) fn host_memory_alignment(
        &self,
    ) -> Result<usize, crate::backend::renderer::MemoryHostUnavailable> {
        self.host_memory.alignment
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

    pub(crate) fn upload_arena_stats(&self) -> UploadArenaStats {
        self.upload_arena.stats()
    }

    pub(crate) fn configure_memory_upload_capacity(
        &mut self,
        capacity: usize,
    ) -> Result<bool, VulkanRendererError> {
        self.configure_memory_upload_capacity_for_extent(capacity, 0)
    }

    pub(crate) fn owner_upload_extent_stats(&self) -> (usize, usize, usize, u64) {
        (
            self.upload_arena.stats().capacity_bytes,
            self.owner_upload_structural_bytes,
            self.owner_upload_exception_bytes,
            self.owner_upload_extent_exceptions,
        )
    }

    pub(crate) fn configure_memory_upload_capacity_for_extent(
        &mut self,
        structural_bytes: usize,
        generation_bytes: usize,
    ) -> Result<bool, VulkanRendererError> {
        self.reclaim_completed_submissions()?;
        self.drop_unsampleable_uploads();
        let capacity = structural_bytes.max(generation_bytes);
        let configured =
            self.upload_arena
                .configure_fixed(&self.physical_device, self.device.clone(), capacity)?;
        if configured {
            let exception_bytes = capacity.saturating_sub(structural_bytes);
            if exception_bytes != 0
                && (self.owner_upload_structural_bytes != structural_bytes
                    || self.owner_upload_exception_bytes != exception_bytes)
            {
                self.owner_upload_extent_exceptions = self.owner_upload_extent_exceptions.saturating_add(1);
                trace!(
                    structural_bytes,
                    generation_bytes,
                    exception_bytes,
                    "owner-sized upload extent exception"
                );
            }
            self.owner_upload_structural_bytes = structural_bytes;
            self.owner_upload_exception_bytes = exception_bytes;
        }
        Ok(configured)
    }

    pub(crate) fn memory_upload_storage_matches(&self, structural: usize, generation: usize) -> bool {
        self.upload_arena.capacity_matches(structural.max(generation))
            && self.owner_upload_structural_bytes == structural
            && self.owner_upload_exception_bytes == generation.saturating_sub(structural)
    }

    pub(crate) fn adopt_memory_upload_storage(
        &mut self,
        storage: &mut super::VulkanUploadStorage,
        structural: usize,
        generation: usize,
    ) -> Result<bool, VulkanRendererError> {
        if storage.requested_bytes() != structural.max(generation) {
            return Err(VulkanRendererError::InvalidMemoryUpload(
                "prepared arena differs from the exact owner extent",
            ));
        }
        self.reclaim_completed_submissions()?;
        self.drop_unsampleable_uploads();
        if !self.upload_arena.adopt(&self.device, storage)? {
            return Ok(false);
        }
        let exception = generation.saturating_sub(structural);
        if exception != 0
            && (self.owner_upload_structural_bytes != structural
                || self.owner_upload_exception_bytes != exception)
        {
            self.owner_upload_extent_exceptions = self.owner_upload_extent_exceptions.saturating_add(1);
        }
        self.owner_upload_structural_bytes = structural;
        self.owner_upload_exception_bytes = exception;
        Ok(true)
    }

    fn release_upload(&mut self, reservation: StagingReservation) {
        self.upload_arena.release(reservation);
    }

    pub(crate) fn pending_upload_stats(&self) -> (usize, usize) {
        (self.pending_uploads.operations.len(), self.pending_uploads.bytes)
    }

    /// Copy exact CPU pixels into renderer-owned staging and retain one upload
    /// operation for the next real queue submission. This method records no
    /// command buffer and performs no queue submission.
    pub(crate) fn queue_image_upload(
        &mut self,
        image: Arc<VulkanImage>,
        upload: ImageUpload<'_>,
    ) -> Result<(), VulkanRendererError> {
        if !image.is_renderer_local() {
            return Err(VulkanRendererError::TemporaryFailure(
                "memory uploads require a renderer-local Vulkan image",
            ));
        }
        let reservation = self.reserve_image_upload(upload.byte_len()?)?;
        self.queue_reserved_image_upload(image, reservation, upload)
    }

    #[cfg(feature = "wayland_frontend")]
    pub(super) fn prepare_host_buffer(
        &mut self,
        source: &crate::backend::renderer::utils::Buffer,
    ) -> Result<
        Result<Arc<super::host_memory::HostBuffer>, crate::backend::renderer::MemoryHostUnavailable>,
        VulkanRendererError,
    > {
        self.reclaim_completed_submissions()?;
        self.drop_unsampleable_uploads();
        if self.pending_uploads.operations.len() >= MAX_UPLOAD_BATCH_OPERATIONS {
            return Err(VulkanRendererError::UploadBatchFull {
                limit: MAX_UPLOAD_BATCH_OPERATIONS,
            });
        }
        self.host_memory
            .import(&self.physical_device, self.shared_device(), source)
    }

    #[cfg(feature = "wayland_frontend")]
    pub(super) fn queue_host_image_upload(
        &mut self,
        image: Arc<VulkanImage>,
        source: Arc<super::host_memory::HostBuffer>,
        region: vk::BufferImageCopy,
        bytes: usize,
    ) -> Result<(), VulkanRendererError> {
        self.drop_unsampleable_uploads();
        if self.pending_uploads.operations.len() >= MAX_UPLOAD_BATCH_OPERATIONS {
            return Err(VulkanRendererError::UploadBatchFull {
                limit: MAX_UPLOAD_BATCH_OPERATIONS,
            });
        }
        let old_layout = self
            .pending_uploads
            .layout_after_pending(image.id())
            .unwrap_or_else(|| image.current_layout());
        self.pending_uploads.bytes = self.pending_uploads.bytes.saturating_add(bytes);
        image.set_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        self.pending_uploads.operations.push(PendingUpload {
            image,
            source: UploadSource::Host(source),
            bytes,
            region,
            old_layout,
        });
        Ok(())
    }

    /// Whole-generation admission before an initial import allocates its image.
    /// Only the owner's existing chunk may satisfy it.
    pub(crate) fn reserve_image_upload(
        &mut self,
        len: usize,
    ) -> Result<StagingReservation, VulkanRendererError> {
        self.reclaim_completed_submissions()?;
        self.drop_unsampleable_uploads();
        if self.pending_uploads.operations.len() >= MAX_UPLOAD_BATCH_OPERATIONS {
            return Err(VulkanRendererError::UploadBatchFull {
                limit: MAX_UPLOAD_BATCH_OPERATIONS,
            });
        }
        self.upload_arena.reserve(len)
    }

    pub(crate) fn queue_reserved_image_upload(
        &mut self,
        image: Arc<VulkanImage>,
        reservation: StagingReservation,
        upload: ImageUpload<'_>,
    ) -> Result<(), VulkanRendererError> {
        if let Err(error) = self.upload_arena.write_rows(
            reservation,
            upload.data,
            upload.source_offset,
            upload.source_stride,
            upload.row_bytes,
            upload.rows,
        ) {
            self.upload_arena.release(reservation);
            return Err(error);
        }
        self.queue_staged_image_upload(image, reservation, upload.region)
    }

    /// Reserve staging bytes for an upload into `image` whose rows are
    /// written later, possibly on another thread ([`Self::queue_staged_image_upload`]).
    pub(crate) fn stage_image_upload(
        &mut self,
        image: &Arc<VulkanImage>,
        len: usize,
    ) -> Result<(StagingReservation, *mut u8, Arc<ReservationWriter>), VulkanRendererError> {
        if !image.is_renderer_local() {
            return Err(VulkanRendererError::TemporaryFailure(
                "memory uploads require a renderer-local Vulkan image",
            ));
        }
        self.reserve_staged_upload(len)
    }

    pub(crate) fn reserve_staged_upload(
        &mut self,
        len: usize,
    ) -> Result<(StagingReservation, *mut u8, Arc<ReservationWriter>), VulkanRendererError> {
        let reservation = self.reserve_image_upload(len)?;
        let arena = &mut self.upload_arena;
        match arena.detach(reservation) {
            Ok((ptr, memory)) => Ok((reservation, ptr, memory)),
            Err(error) => {
                arena.release(reservation);
                Err(error)
            }
        }
    }

    /// Queue a staged upload whose rows are all written. It joins the pending
    /// batch that the next submission carries first. On error the reservation
    /// is released and `image` keeps its pixels.
    pub(crate) fn queue_staged_image_upload(
        &mut self,
        image: Arc<VulkanImage>,
        reservation: StagingReservation,
        region: vk::BufferImageCopy,
    ) -> Result<(), VulkanRendererError> {
        self.drop_unsampleable_uploads();
        if self.pending_uploads.operations.len() >= MAX_UPLOAD_BATCH_OPERATIONS {
            self.release_upload(reservation);
            return Err(VulkanRendererError::UploadBatchFull {
                limit: MAX_UPLOAD_BATCH_OPERATIONS,
            });
        }
        if let Err(error) = self.upload_arena.flush(reservation) {
            self.release_upload(reservation);
            return Err(error);
        }
        let old_layout = self
            .pending_uploads
            .layout_after_pending(image.id())
            .unwrap_or_else(|| image.current_layout());
        self.pending_uploads.bytes = self.pending_uploads.bytes.saturating_add(reservation.len());
        image.set_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        self.pending_uploads.operations.push(PendingUpload {
            image,
            source: UploadSource::Staging(reservation),
            bytes: reservation.len(),
            region,
            old_layout,
        });
        Ok(())
    }

    /// Release a staged reservation that will not be queued.
    pub(crate) fn release_staged_image_upload(&mut self, reservation: StagingReservation) {
        self.release_upload(reservation);
    }

    /// Drop the pending uploads whose image nothing can sample any more
    /// (see [`PendingUploadBatch::drop_unsampleable`]). Runs before an
    /// upload claims batch or staging capacity and before a batch is
    /// recorded, so dead uploads neither fill the batch nor reach the GPU.
    fn drop_unsampleable_uploads(&mut self) {
        self.upload_arena.reap_parked();
        self.pending_uploads.drop_unsampleable(&mut self.upload_arena);
    }

    fn record_pending_uploads(&mut self) -> Result<Option<RecordedUploadBatch>, VulkanRendererError> {
        self.drop_unsampleable_uploads();
        if self.pending_uploads.operations.is_empty() {
            return Ok(None);
        }
        let command_buffer = self.acquire_command_buffer()?;
        let batch = std::mem::take(&mut self.pending_uploads);
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        if let Err(error) = self.device.observe_result(unsafe {
            self.device
                .handle()
                .begin_command_buffer(command_buffer, &begin_info)
        }) {
            let _ = self.discard_command_buffer(command_buffer);
            self.restore_pending_uploads(batch);
            return Err(error.into());
        }
        self.insert_debug_label(command_buffer, c"vulkan.upload_batch", [0.11, 0.78, 0.86, 1.0]);

        for operation in &batch.operations {
            transition_image_layout(
                self.device.handle(),
                command_buffer,
                operation.image.image(),
                operation.old_layout,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            let mut copy_region = operation.region;
            let buffer = match &operation.source {
                UploadSource::Staging(reservation) => {
                    copy_region.buffer_offset += reservation.offset();
                    self.upload_arena.buffer(*reservation)
                }
                #[cfg(feature = "wayland_frontend")]
                UploadSource::Host(buffer) => {
                    copy_region.buffer_offset += buffer.source_offset;
                    // Coherent imported host pages need execution/visibility
                    // ordering, but no CPU copy or noncoherent flush.
                    let barrier = vk::BufferMemoryBarrier::default()
                        .buffer(buffer.buffer)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .src_access_mask(vk::AccessFlags::HOST_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                    unsafe {
                        self.device.handle().cmd_pipeline_barrier(
                            command_buffer,
                            vk::PipelineStageFlags::HOST,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[barrier],
                            &[],
                        );
                    }
                    buffer.buffer
                }
            };
            unsafe {
                self.device.handle().cmd_copy_buffer_to_image(
                    command_buffer,
                    buffer,
                    operation.image.image(),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy_region],
                );
            }
            transition_image_layout(
                self.device.handle(),
                command_buffer,
                operation.image.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        }

        if let Err(error) = self
            .device
            .observe_result(unsafe { self.device.handle().end_command_buffer(command_buffer) })
        {
            let _ = self.discard_command_buffer(command_buffer);
            self.restore_pending_uploads(batch);
            return Err(error.into());
        }
        Ok(Some(RecordedUploadBatch {
            command_buffer,
            batch,
        }))
    }

    fn restore_pending_uploads(&mut self, mut batch: PendingUploadBatch) {
        if self.pending_uploads.operations.is_empty() {
            self.pending_uploads = batch;
            return;
        }
        batch.bytes = batch.bytes.saturating_add(self.pending_uploads.bytes);
        batch.operations.append(&mut self.pending_uploads.operations);
        self.pending_uploads = batch;
    }

    /// Name the exact completion edge that returns bounded upload capacity.
    ///
    /// Pending uploads are sealed into one submission. If all capacity is
    /// already submitted, the newest staging-owning submission is sufficient:
    /// Vulkan queue ordering guarantees that its completion also completes all
    /// earlier submissions retaining arena spans.
    pub(crate) fn memory_upload_capacity_edge(
        &mut self,
    ) -> Result<MemoryUploadCapacityEdge, VulkanRendererError> {
        self.drop_unsampleable_uploads();
        if self.pending_uploads.operations.is_empty() {
            return Ok(self
                .in_flight_submissions
                .iter()
                .rev()
                .find(|submission| !submission.upload_sources.is_empty())
                .map(|submission| MemoryUploadCapacityEdge::InFlight(submission.fence.clone()))
                .unwrap_or_else(|| {
                    if let Some(completion) = self.upload_arena.cpu_completion() {
                        MemoryUploadCapacityEdge::CpuWriterPending(completion)
                    } else if self.upload_arena.stats().in_use_bytes == 0 {
                        MemoryUploadCapacityEdge::Available
                    } else {
                        MemoryUploadCapacityEdge::NotApplicable
                    }
                }));
        }

        let command_buffer = self.acquire_command_buffer()?;
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        if let Err(error) = self.device.observe_result(unsafe {
            self.device
                .handle()
                .begin_command_buffer(command_buffer, &begin_info)
        }) {
            let _ = self.discard_command_buffer(command_buffer);
            return Err(error.into());
        }
        self.insert_debug_label(
            command_buffer,
            c"vulkan.upload_capacity_edge",
            [0.88, 0.45, 0.12, 1.0],
        );
        if let Err(error) = self
            .device
            .observe_result(unsafe { self.device.handle().end_command_buffer(command_buffer) })
        {
            let _ = self.discard_command_buffer(command_buffer);
            return Err(error.into());
        }

        match self.submit_tracked(
            command_buffer,
            &mut Vec::new(),
            &mut Vec::new(),
            None,
            SubmissionKind::MemoryUploadBatch,
        ) {
            Ok((_, fence)) => Ok(MemoryUploadCapacityEdge::Submitted(fence)),
            Err(error) => Err(error),
        }
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
        self.queue_wait_on_sync_file_with_stage(sync_file, vk::PipelineStageFlags::ALL_COMMANDS, None)
    }

    pub(crate) fn external_wait_already_staged(&self, source: &SyncPoint) -> bool {
        self.external_wait_bank
            .as_ref()
            .is_some_and(|bank| bank.contains_owner(source))
    }

    pub(crate) fn queue_wait_on_sync_file_with_stage(
        &mut self,
        sync_file: OwnedFd,
        wait_stage_mask: vk::PipelineStageFlags,
        source: Option<&SyncPoint>,
    ) -> Result<(), VulkanRendererError> {
        if self.device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        if !self.supports_sync_file_import() {
            return Err(VulkanRendererError::NotImplemented(
                "sync_file semaphore import is not available on this Vulkan device",
            ));
        }
        if self.completion_unknown {
            return Err(VulkanRendererError::CommandCompletionUnavailable);
        }
        if wait_stage_mask.is_empty() {
            return Err(VulkanRendererError::TemporaryFailure(
                "sync_file wait stage mask must not be empty",
            ));
        }

        if self.pending_waits.len() >= self.command_limits.waits_per_submission {
            return Err(VulkanRendererError::CommandStorageLimitExceeded {
                resource: "imported waits",
                requested: self.pending_waits.len() + 1,
                limit: self.command_limits.waits_per_submission,
            });
        }
        let external_semaphore_fd = self
            .external_semaphore_fd
            .as_ref()
            .expect("checked by supports_sync_file_import");
        let semaphore = if let Some(bank) = self.external_wait_bank.as_mut() {
            // The selected frame preclaims its complete import inventory. A
            // caller undercount is refused before the extra native import.
            self.external_wait_batch.check()?;
            let source = source.ok_or(VulkanRendererError::TemporaryFailure(
                "prepared wait requires its original source proof",
            ))?;
            let Some(semaphore) = bank.take(source) else {
                return Err(VulkanRendererError::CommandStorageLimitExceeded {
                    resource: "admitted external wait loans",
                    requested: self.pending_waits.len().saturating_add(1),
                    limit: bank.capacity(),
                });
            };
            if let Err(error) = import_sync_file_into_semaphore(
                self.device.as_ref(),
                external_semaphore_fd,
                semaphore.handle,
                sync_file,
            ) {
                // Import failure did not submit a wait or transfer the FD.
                bank.release(semaphore);
                return Err(error);
            }
            self.external_wait_batch.imported();
            semaphore
        } else {
            ImportedWaitSemaphore {
                handle: import_sync_file_to_semaphore(
                    self.device.as_ref(),
                    external_semaphore_fd,
                    sync_file,
                )?,
                pooled: false,
                index: 0,
            }
        };

        // A repeated proof can cover any subsequent queue read, irrespective
        // of the narrower stage mask of its first importer.
        self.pending_waits.push((
            semaphore,
            if semaphore.pooled {
                vk::PipelineStageFlags::ALL_COMMANDS
            } else {
                wait_stage_mask
            },
        ));
        Ok(())
    }

    pub(crate) fn take_pending_wait_semaphores(&mut self) -> usize {
        let count = self.pending_waits.len();
        self.clear_pending_wait_semaphores();
        count
    }

    pub(crate) fn clear_pending_wait_semaphores(&mut self) {
        // These loans were never part of a successful or uncertain native
        // submit. TEMPORARY import replacement discards their old payload;
        // no CPU signal/reset is fabricated, and the original source FD stays
        // owned by the cached source proof for a retry.
        for (semaphore, _) in self.pending_waits.drain(..) {
            if semaphore.pooled {
                self.external_wait_bank
                    .as_mut()
                    .expect("native wait loan bank")
                    .release(semaphore);
            } else {
                self.device
                    .destroy_with(|device| unsafe { device.destroy_semaphore(semaphore.handle, None) });
            }
        }
        self.external_wait_batch.begin(0);
    }

    #[instrument(level = "trace", skip(self))]
    #[profiling::function]
    pub(super) fn completion_unknown(&self) -> bool {
        self.completion_unknown
    }
    pub(super) fn preserve_failed_recording(
        &mut self,
        storage: super::recording_storage::RecordingStorageLease,
    ) {
        assert!(self.failed_recording.is_none(), "one exclusive failed recording");
        self.completion_unknown = true;
        self.failed_recording = Some(storage);
    }

    pub(super) fn acquire_recording_storage(
        &self,
    ) -> Result<super::recording_storage::RecordingStorageLease, VulkanRendererError> {
        self.recording_storage.acquire()
    }

    pub(crate) fn acquire_command_buffer(&mut self) -> Result<vk::CommandBuffer, VulkanRendererError> {
        if self.device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        self.reclaim_completed_submissions()?;

        if let Some(command_buffer) = self.reusable_command_buffers.pop() {
            return Ok(command_buffer);
        }

        Err(VulkanRendererError::CommandCapacityExhausted {
            slots: self.command_limits.submission_slots,
        })
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
        mut framebuffers: Vec<vk::Framebuffer>,
    ) -> Result<SubmissionId, VulkanRendererError> {
        let (id, _) =
            self.submit_with_resources_and_fence(command_buffer, &mut framebuffers, &mut Vec::new())?;
        Ok(id)
    }

    pub(crate) fn submit_with_resources(
        &mut self,
        command_buffer: vk::CommandBuffer,
        framebuffers: &mut Vec<vk::Framebuffer>,
        retained_images: &mut Vec<Arc<VulkanImage>>,
    ) -> Result<SubmissionId, VulkanRendererError> {
        let (id, _) = self.submit_with_resources_and_fence(command_buffer, framebuffers, retained_images)?;
        Ok(id)
    }

    #[instrument(level = "trace", skip(self, command_buffer, framebuffers))]
    #[profiling::function]
    pub(crate) fn submit_with_framebuffers_and_fence(
        &mut self,
        command_buffer: vk::CommandBuffer,
        mut framebuffers: Vec<vk::Framebuffer>,
    ) -> Result<(SubmissionId, SyncPoint), VulkanRendererError> {
        self.submit_with_resources_and_fence(command_buffer, &mut framebuffers, &mut Vec::new())
    }

    pub(crate) fn submit_with_resources_and_fence(
        &mut self,
        command_buffer: vk::CommandBuffer,
        framebuffers: &mut Vec<vk::Framebuffer>,
        retained_images: &mut Vec<Arc<VulkanImage>>,
    ) -> Result<(SubmissionId, SyncPoint), VulkanRendererError> {
        self.submit_tracked(
            command_buffer,
            framebuffers,
            retained_images,
            None,
            SubmissionKind::Render,
        )
    }

    #[instrument(
        level = "trace",
        skip(self, command_buffer, framebuffers, retained_images, readback)
    )]
    #[profiling::function]
    fn submit_tracked(
        &mut self,
        command_buffer: vk::CommandBuffer,
        framebuffers: &mut Vec<vk::Framebuffer>,
        retained_images: &mut Vec<Arc<VulkanImage>>,
        readback: Option<Arc<super::readback::ReadbackBuffer>>,
        kind: SubmissionKind,
    ) -> Result<(SubmissionId, SyncPoint), VulkanRendererError> {
        if self.device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        if self.completion_unknown {
            return Err(VulkanRendererError::CommandCompletionUnavailable);
        }
        self.reclaim_completed_submissions()?;
        let image_count = retained_images.len() + self.pending_uploads.operations.len();
        if image_count > self.command_limits.images_per_submission
            || framebuffers.len() > self.command_limits.framebuffers_per_submission
        {
            let _ = self.discard_recording_resources(command_buffer, framebuffers);
            return Err(VulkanRendererError::CommandStorageLimitExceeded {
                resource: "submitted resources",
                requested: image_count,
                limit: self.command_limits.images_per_submission,
            });
        }
        let Some(index) = self
            .free_submissions
            .iter()
            .position(InFlightSubmission::readers_returned)
        else {
            let _ = self.discard_recording_resources(command_buffer, framebuffers);
            return Err(VulkanRendererError::CommandCapacityExhausted {
                slots: self.command_limits.submission_slots,
            });
        };
        let mut slot = self.free_submissions.swap_remove(index);
        if let Err(error) = slot.prepare_for_submit() {
            self.free_submissions.push(slot);
            let _ = self.discard_recording_resources(command_buffer, framebuffers);
            return Err(error);
        }
        let recorded_uploads = match self.record_pending_uploads() {
            Ok(recorded) => recorded,
            Err(error) => {
                self.free_submissions.push(slot);
                let _ = self.discard_recording_resources(command_buffer, framebuffers);
                return Err(error);
            }
        };
        let submit_started_at = Instant::now();
        let export_semaphore = if kind.exports_completion() {
            slot.export_semaphore
        } else {
            None
        };
        let mut waits = [vk::Semaphore::null(); 256];
        let mut stages = [vk::PipelineStageFlags::empty(); 256];
        let wait_count = if kind.consumes_pending_waits() {
            self.pending_waits.len()
        } else {
            0
        };
        for (index, (semaphore, stage)) in self.pending_waits.iter().take(wait_count).enumerate() {
            waits[index] = semaphore.handle;
            stages[index] = *stage;
        }
        if let Some(upload) = recorded_uploads.as_ref() {
            slot.command_buffers.push(upload.command_buffer);
        }
        slot.command_buffers.push(command_buffer);
        let mut submit = vk::SubmitInfo::default()
            .command_buffers(&slot.command_buffers)
            .wait_semaphores(&waits[..wait_count])
            .wait_dst_stage_mask(&stages[..wait_count]);
        let signals = export_semaphore.map(|semaphore| [semaphore]);
        if let Some(signals) = signals.as_ref() {
            submit = submit.signal_semaphores(signals);
        }
        slot.native_fence().mark_native_attempt();
        let id = match self
            .device
            .submit_queue(self.queue, &[submit], slot.native_fence().handle())
        {
            Ok(id) => id,
            Err(error) => {
                if !error.is_resource_allocation_failure() {
                    // Unknown/device-loss submit outcomes cannot authorize
                    // resetting encoded commands or releasing their mappings.
                    self.completion_unknown = true;
                    slot.submission_id_known = false;
                    slot.submitted_at = submit_started_at;
                    slot.export_unconsumed = export_semaphore.is_some();
                    slot.framebuffers.append(framebuffers);
                    slot.retained_images.append(retained_images);
                    slot._readback = readback;
                    if kind.consumes_pending_waits() {
                        slot.wait_semaphores
                            .extend(self.pending_waits.drain(..).map(|(semaphore, _)| semaphore));
                    }
                    if let Some(mut upload) = recorded_uploads {
                        for operation in upload.batch.operations.drain(..) {
                            slot.retained_images.push(operation.image);
                            slot.upload_sources.push(operation.source);
                        }
                        upload.batch.bytes = 0;
                        self.pending_uploads = upload.batch;
                    }
                    self.next_submission_id = self.next_submission_id.wrapping_add(1);
                    self.device.mark_submission_pending();
                    self.in_flight_submissions.push_back(slot);
                    return Err(error.completion_failure());
                }
                if let Some(upload) = recorded_uploads {
                    if self.discard_command_buffer(upload.command_buffer).is_ok() {
                        self.restore_pending_uploads(upload.batch);
                    }
                }
                let _ = self.discard_recording_resources(command_buffer, framebuffers);
                slot.native_fence().cancel_unsubmitted_attempt();
                slot.command_buffers.clear();
                self.free_submissions.push(slot);
                // Imported acquire waits were never consumed. Keep them in
                // their exact pending owner for the caller's rollback/retry.
                return Err(error);
            }
        };
        // Native submission succeeded. Every vector below was cold-reserved
        // and the admission above proves draining cannot grow any container.
        slot.id = id;
        slot.native_fence().mark_submitted();
        slot.submission_id_known = true;
        slot.submitted_at = submit_started_at;
        slot.framebuffers.append(framebuffers);
        slot.retained_images.append(retained_images);
        slot._readback = readback;
        if kind.consumes_pending_waits() {
            slot.wait_semaphores
                .extend(self.pending_waits.drain(..).map(|(semaphore, _)| semaphore));
        }
        let mut upload_operation_count = 0;
        let mut upload_bytes = 0;
        if let Some(mut upload) = recorded_uploads {
            upload_operation_count = upload.batch.operations.len();
            upload_bytes = upload.batch.bytes;
            for operation in upload.batch.operations.drain(..) {
                slot.retained_images.push(operation.image);
                slot.upload_sources.push(operation.source);
            }
            upload.batch.bytes = 0;
            self.pending_uploads = upload.batch;
            self.diagnostics.upload_batches = self.diagnostics.upload_batches.saturating_add(1);
            self.diagnostics.upload_operations = self
                .diagnostics
                .upload_operations
                .saturating_add(upload_operation_count as u64);
            self.diagnostics.upload_bytes = self.diagnostics.upload_bytes.saturating_add(upload_bytes as u64);
        }
        if let Some(semaphore) = export_semaphore {
            slot.export_unconsumed = true;
            match self.export_semaphore_sync_file(semaphore) {
                Ok(fd) => {
                    slot.native_fence().set_exported_sync_file(fd);
                    slot.export_unconsumed = false;
                }
                Err(error) => {
                    // Without a successful SYNC_FD export its binary payload
                    // cannot be signaled again. Retire it after native proof;
                    // this slot then truthfully supplies a nonexportable fence.
                    warn!(?error, "submission completion export failed");
                }
            }
        }
        let submit_cpu_ns = duration_to_ns(submit_started_at.elapsed());
        self.diagnostics.total_submissions = self.diagnostics.total_submissions.saturating_add(1);
        self.diagnostics.total_submit_cpu_ns =
            self.diagnostics.total_submit_cpu_ns.saturating_add(submit_cpu_ns);
        self.diagnostics.max_submit_cpu_ns = self.diagnostics.max_submit_cpu_ns.max(submit_cpu_ns);
        self.next_submission_id = self.next_submission_id.wrapping_add(1);
        self.device.mark_submission_pending();
        let fence = slot.fence.clone();
        self.in_flight_submissions.push_back(slot);
        trace!(
            ?id,
            submit_cpu_ns,
            upload_operation_count,
            upload_bytes,
            "submitted bounded vulkan command scope"
        );
        Ok((id, fence))
    }

    #[instrument(level = "trace", skip(self, command_buffer, image, readback))]
    #[profiling::function]
    pub(crate) fn submit_blocking(
        &mut self,
        command_buffer: vk::CommandBuffer,
        image: Arc<VulkanImage>,
        readback: Arc<super::readback::ReadbackBuffer>,
    ) -> Result<(), BlockingSubmitError> {
        // Tracking happens before any host wait. Once queue submission
        // succeeds, every native fence, upload source and destination owner
        // stays in the exact submission even if completion is unobservable.
        let mut storage = self
            .acquire_recording_storage()
            .map_err(BlockingSubmitError::NotSubmitted)?;
        storage.retained_images.push(image);
        let recording = &mut *storage;
        let submitted = self.submit_tracked(
            command_buffer,
            &mut recording.submitted_framebuffers,
            &mut recording.retained_images,
            Some(readback),
            SubmissionKind::Readback,
        );
        let (id, fence) = match submitted {
            Ok(submitted) => submitted,
            Err(error) if self.completion_unknown => return Err(BlockingSubmitError::Submitted(error)),
            Err(error) => return Err(BlockingSubmitError::NotSubmitted(error)),
        };
        self.diagnostics.blocking_submissions = self.diagnostics.blocking_submissions.saturating_add(1);
        if self.device.is_lost() {
            return Err(BlockingSubmitError::Submitted(VulkanRendererError::ContextLost(
                "readback completion became unobservable on lost device",
            )));
        }
        if let Err(error) = self.device.observe_result(unsafe {
            self.device.handle().wait_for_fences(
                &[fence
                    .get::<VulkanFence>()
                    .expect("native submission fence")
                    .handle()],
                true,
                u64::MAX,
            )
        }) {
            self.completion_unknown = true;
            return Err(BlockingSubmitError::Submitted(
                VulkanRendererError::from(error).completion_failure(),
            ));
        }
        let index = self
            .in_flight_submissions
            .iter()
            .position(|submission| submission.id == id)
            .expect("successful readback remains tracked until its exact fence completes");
        let completed = self
            .in_flight_submissions
            .remove(index)
            .expect("located submission");
        self.recycle_submission(completed)
            .map_err(BlockingSubmitError::Submitted)
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
        if let Err(error) = self.device.observe_result(unsafe {
            self.device
                .handle()
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
        }) {
            self.completion_unknown = true;
            return Err(VulkanRendererError::from(error).completion_failure());
        }
        self.reusable_command_buffers.push(command_buffer);
        Ok(())
    }

    /// Reset an unsubmitted command buffer before destroying objects encoded
    /// into it. On reset failure the child handles intentionally survive until
    /// device teardown rather than being destroyed while still referenced by
    /// an executable command buffer.
    pub(crate) fn discard_recording_resources(
        &mut self,
        command_buffer: vk::CommandBuffer,
        framebuffers: &mut Vec<vk::Framebuffer>,
    ) -> Result<(), VulkanRendererError> {
        self.discard_command_buffer(command_buffer)?;
        self.device.destroy_with(|device| {
            for framebuffer in framebuffers.drain(..) {
                // SAFETY: Reset removed every command-buffer reference and no
                // queue submission ever consumed this framebuffer.
                unsafe { device.destroy_framebuffer(framebuffer, None) };
            }
        });
        Ok(())
    }

    pub(crate) fn in_flight_submission_count(&self) -> usize {
        self.in_flight_submissions.len()
    }

    pub(crate) fn submission_snapshot(&self) -> VulkanSubmissionSnapshot {
        VulkanSubmissionSnapshot {
            next_submission_id: self.next_submission_id,
        }
    }

    pub(crate) fn completion_since(&self, snapshot: VulkanSubmissionSnapshot) -> Option<SyncPoint> {
        if snapshot.next_submission_id == self.next_submission_id {
            return None;
        }
        self.in_flight_submissions
            .back()
            .map(|submission| submission.fence.clone())
            .or_else(|| Some(SyncPoint::signaled()))
    }

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
        if self.completion_unknown {
            return Err(VulkanRendererError::CommandCompletionUnavailable);
        }
        loop {
            let Some(front) = self.in_flight_submissions.front() else {
                break;
            };

            // The caller-visible fence is never exported (the sync_file rides
            // a dedicated semaphore), so it faithfully tracks completion and
            // can be polled directly.
            let poll_fence = front.native_fence().handle();
            // SAFETY: Fence was created by this device and remains valid while tracked.
            let signaled = match self
                .device
                .observe_result(unsafe { self.device.handle().get_fence_status(poll_fence) })
            {
                Ok(signaled) => signaled,
                Err(error) => {
                    self.completion_unknown = true;
                    return Err(VulkanRendererError::from(error).completion_failure());
                }
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
            if let Err(error) = self.device.observe_result(unsafe {
                self.device
                    .handle()
                    .wait_for_fences(&[submission.native_fence().handle()], true, u64::MAX)
            }) {
                self.in_flight_submissions.push_front(submission);
                self.completion_unknown = true;
                return Err(VulkanRendererError::from(error).completion_failure());
            }
            self.recycle_submission(submission)?;
        }

        Ok(())
    }

    fn recycle_submission(&mut self, mut submission: InFlightSubmission) -> Result<(), VulkanRendererError> {
        let completion_ns = duration_to_ns(submission.submitted_at.elapsed());
        // Preserve the complete slot on any reset failure. Native completion
        // is proven, but executable command references still need teardown.
        for command_buffer in &submission.command_buffers {
            if let Err(error) = self.device.observe_result(unsafe {
                self.device
                    .handle()
                    .reset_command_buffer(*command_buffer, vk::CommandBufferResetFlags::empty())
            }) {
                self.completion_unknown = true;
                self.in_flight_submissions.push_front(submission);
                return Err(VulkanRendererError::from(error).completion_failure());
            }
        }
        self.reusable_command_buffers
            .extend(submission.command_buffers.drain(..));
        // Only the exact submitted fence, plus successful command reset above,
        // authorizes reuse. Unknown completion retains these loans in the slot.
        for semaphore in submission.wait_semaphores.drain(..) {
            if semaphore.pooled {
                self.external_wait_bank
                    .as_mut()
                    .expect("submitted native wait bank")
                    .release(semaphore);
            } else {
                self.device
                    .destroy_with(|device| unsafe { device.destroy_semaphore(semaphore.handle, None) });
            }
        }
        self.device.destroy_with(|device| {
            for framebuffer in submission.framebuffers.drain(..) {
                unsafe {
                    device.destroy_framebuffer(framebuffer, None);
                }
            }
            if submission.export_unconsumed {
                if let Some(semaphore) = submission.export_semaphore.take() {
                    unsafe {
                        device.destroy_semaphore(semaphore, None);
                    }
                }
            }
        });
        for source in submission.upload_sources.drain(..) {
            source.release(&mut self.upload_arena);
        }
        submission.retained_images.clear();
        submission._readback.take();
        self.device.mark_submission_completed();
        if submission.submission_id_known {
            self.device.note_submission_completed(submission.id);
        }
        self.diagnostics.reclaimed_submissions = self.diagnostics.reclaimed_submissions.saturating_add(1);
        self.diagnostics.total_completion_ns =
            self.diagnostics.total_completion_ns.saturating_add(completion_ns);
        self.diagnostics.max_completion_ns = self.diagnostics.max_completion_ns.max(completion_ns);
        self.free_submissions.push(submission);
        Ok(())
    }

    pub(crate) fn adopt_command_storage(
        &mut self,
        prepared: &mut super::VulkanCommandStorage,
    ) -> Result<bool, VulkanRendererError> {
        if !Arc::ptr_eq(&self.device, &prepared.device) || self.queue_family_index != prepared.family {
            return Err(VulkanRendererError::TemporaryFailure(
                "prepared command storage belongs to another native origin",
            ));
        }
        if prepared.node.is_none() {
            return Err(VulkanRendererError::TemporaryFailure(
                "prepared command storage already adopted",
            ));
        }
        self.reclaim_completed_submissions()?;
        if !self.in_flight_submissions.is_empty()
            || !self.pending_uploads.operations.is_empty()
            || !self.pending_waits.is_empty()
            || self.free_submissions.iter().any(|slot| !slot.readers_returned())
        {
            return Ok(false);
        }
        let mut new_node = prepared.node.take().unwrap();
        let new = new_node.value_mut();
        let mut old_node = self.command_retirement.take().unwrap();
        let old = old_node.value_mut();
        old.device = Some(self.device.clone());
        old.free_submissions = std::mem::replace(
            &mut self.free_submissions,
            std::mem::take(&mut new.free_submissions),
        );
        old.submissions = std::mem::replace(
            &mut self.in_flight_submissions,
            std::mem::take(&mut new.submissions),
        );
        old.reusable_command_buffers = std::mem::replace(
            &mut self.reusable_command_buffers,
            std::mem::take(&mut new.reusable_command_buffers),
        );
        old.bank_return = Some(std::mem::replace(
            &mut self.bank_return,
            new.bank_return.take().unwrap(),
        ));
        old.recording_storage = Some(std::mem::replace(
            &mut self.recording_storage,
            new.recording_storage.take().unwrap(),
        ));
        old.pending_uploads = std::mem::replace(
            &mut self.pending_uploads,
            std::mem::take(&mut new.pending_uploads),
        );
        old.pending_waits =
            std::mem::replace(&mut self.pending_waits, std::mem::take(&mut new.pending_waits));
        self.command_pool = new.command_pool;
        self.command_limits = prepared.limits;
        new.device.take();
        self.command_retirement = Some(new_node);
        self.device.retire_commands(old_node);
        Ok(true)
    }

    pub(crate) fn command_storage_adoption_edge(&mut self) -> Result<SyncPoint, VulkanRendererError> {
        self.reclaim_completed_submissions()?;
        Ok(self.bank_return.clone())
    }

    pub(crate) fn prepare_damage_scratch_storage(
        &mut self,
        slots: usize,
        rectangles: usize,
    ) -> Result<(), VulkanRendererError> {
        if self.damage_scratch.limits() != (slots, rectangles) {
            self.damage_scratch = super::damage_scratch::DamageScratchBank::cold(slots, rectangles)?;
        }
        Ok(())
    }
    pub(super) fn acquire_damage_scratch(
        &self,
        required: usize,
    ) -> Result<super::damage_scratch::DamageScratchLoan, VulkanRendererError> {
        self.damage_scratch.acquire(required)
    }

    pub(crate) fn command_storage_limits(&self) -> VulkanCommandStorageLimits {
        self.command_limits
    }

    pub(crate) fn external_wait_storage_capacity(&self) -> usize {
        self.external_wait_bank
            .as_ref()
            .map_or(0, ExternalWaitBank::capacity)
    }

    pub(crate) fn external_wait_storage_is_prepared(&self) -> bool {
        self.external_wait_bank.is_some()
    }

    pub(crate) fn prepare_external_wait_storage(&mut self, count: usize) -> Result<(), VulkanRendererError> {
        if self.device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        if self.completion_unknown {
            return Err(VulkanRendererError::CommandCompletionUnavailable);
        }
        if count > self.command_limits.waits_per_submission {
            return Err(VulkanRendererError::CommandStorageLimitExceeded {
                resource: "prepared external waits",
                requested: count,
                limit: self.command_limits.waits_per_submission,
            });
        }
        if count != 0 && !self.supports_sync_file_import() {
            return Err(VulkanRendererError::NotImplemented(
                "prepared external waits require SYNC_FD semaphore import",
            ));
        }
        self.clear_pending_wait_semaphores();
        self.reclaim_completed_submissions()?;
        if let Some(bank) = &self.external_wait_bank {
            if bank.capacity() >= count {
                return Ok(());
            }
            if bank.available() != bank.capacity() {
                self.external_wait_pressure = bank.capacity();
                return Err(VulkanRendererError::CommandCapacityExhausted {
                    slots: bank.capacity(),
                });
            }
        }
        // The old bank remains intact if cold creation fails. Both replacement
        // and eventual old native destruction happen outside selected draws.
        let replacement = ExternalWaitBank::cold(self.device.clone(), count)?;
        self.external_wait_bank = Some(replacement);
        self.external_wait_pressure = 0;
        Ok(())
    }

    pub(crate) fn begin_external_wait_batch(&mut self, required: usize) -> Result<(), VulkanRendererError> {
        if self.device.is_lost() {
            return Err(VulkanRendererError::ContextLost("vulkan device already lost"));
        }
        if self.completion_unknown {
            return Err(VulkanRendererError::CommandCompletionUnavailable);
        }
        let capacity = self.external_wait_storage_capacity();
        if self.external_wait_bank.is_none() || required > capacity {
            return Err(VulkanRendererError::CommandStorageLimitExceeded {
                resource: "prepared external waits",
                requested: required,
                limit: capacity,
            });
        }
        // A failed pre-GPU importer may leave exact unsubmitted loans. Cancel
        // them before retrying the original frame rather than duplicating them.
        self.clear_pending_wait_semaphores();
        self.reclaim_completed_submissions()?;
        let bank = self.external_wait_bank.as_ref().unwrap();
        if let Err(error) = bank.admit_batch(required) {
            self.external_wait_pressure = required;
            return Err(error);
        }
        self.external_wait_pressure = 0;
        self.external_wait_batch.begin(required);
        Ok(())
    }

    pub(crate) fn command_capacity_edge(&mut self) -> Result<Option<SyncPoint>, VulkanRendererError> {
        self.reclaim_completed_submissions()?;
        if self.external_wait_pressure != 0 {
            let bank = self
                .external_wait_bank
                .as_ref()
                .expect("native wait pressure bank");
            if bank.available() < self.external_wait_pressure {
                // Every unavailable handle was transferred to a native slot;
                // a free command slot is unrelated to this source-fence loan.
                let native = self
                    .in_flight_submissions
                    .iter()
                    .find(|slot| slot.wait_semaphores.iter().any(|wait| wait.pooled))
                    .map(|slot| slot.fence.clone());
                return Ok(Some(native.unwrap_or_else(|| bank.all_returned_edge())));
            }
            // Reclaim may race the error observer to readiness. Return the
            // same real CPU capacity predicate rather than losing the retry.
            return Ok(Some(bank.all_returned_edge()));
        }
        if self
            .free_submissions
            .iter()
            .any(InFlightSubmission::readers_returned)
            && !self.reusable_command_buffers.is_empty()
        {
            return Ok(None);
        }
        if let Some(submission) = self.in_flight_submissions.front() {
            return Ok(Some(submission.fence.clone()));
        }
        // Native fences here are already ready. An exact logical-return fence
        // prevents an immediate-ready FD from producing a busy retry loop.
        Ok(self
            .free_submissions
            .iter()
            .find(|slot| !slot.readers_returned())
            .map(|slot| slot.reader_return.clone()))
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
            if physical_device.has_device_extension(ext::external_memory_host::NAME) {
                enabled.push(ext::external_memory_host::NAME);
            }
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
    fn renderer_submission_owns_acquire_waits_and_completion_export() {
        assert!(SubmissionKind::Render.consumes_pending_waits());
        assert!(SubmissionKind::Render.exports_completion());
        assert!(!SubmissionKind::MemoryUploadBatch.consumes_pending_waits());
        assert!(SubmissionKind::MemoryUploadBatch.exports_completion());
        assert!(!SubmissionKind::Readback.consumes_pending_waits());
        assert!(!SubmissionKind::Readback.exports_completion());
    }
}

#[cfg(test)]
#[path = "device/readback_custody_tests.rs"]
mod readback_custody_tests;

fn duration_to_ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

impl Drop for DeviceState {
    fn drop(&mut self) {
        let Some(mut node) = self.command_retirement.take() else {
            return;
        };
        let commands = node.value_mut();
        commands.submissions = std::mem::take(&mut self.in_flight_submissions);
        commands.free_submissions = std::mem::take(&mut self.free_submissions);
        commands.reusable_command_buffers = std::mem::take(&mut self.reusable_command_buffers);
        commands.recording_storage = Some(self.recording_storage.clone());
        commands.bank_return = Some(self.bank_return.clone());
        commands.damage_scratch = Some(self.damage_scratch.clone());
        commands.failed_recording = self.failed_recording.take();
        commands.pending_waits = std::mem::take(&mut self.pending_waits);
        commands.external_wait_bank = self.external_wait_bank.take();
        commands.pending_uploads = std::mem::take(&mut self.pending_uploads);
        commands.upload_arena = Some(std::mem::replace(
            &mut self.upload_arena,
            UploadArena::new(&self.physical_device),
        ));
        commands.device = Some(self.device.clone());
        // This final drop may run on Wayland or a frame worker. Publishing a
        // node allocated at construction performs no wait or driver call.
        self.device.retire_commands(node);
    }
}

#[cfg(test)]
mod tests {
    use super::DeviceState;

    /// The Tier-3 device-validity invariant: once a device is observed lost, the single
    /// teardown accessor stops handing out the device so every destroy/wait becomes a no-op.
    /// Healthy state keeps the device available. Fails on missing prerequisites when AVIO_REQUIRE_VK_DEVICE=1.
    #[test]
    fn handle_for_destroy_is_none_after_mark_lost() {
        let Some(physical_device) = super::super::test_support::physical_device() else {
            return;
        };
        let device = match DeviceState::new(&physical_device) {
            Ok(device) => device,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
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
    /// reclamation. Fails on missing prerequisites when AVIO_REQUIRE_VK_DEVICE=1.
    #[test]
    fn submission_sync_file_export_signals_on_completion() {
        use ash::vk;

        let Some(physical_device) = super::super::test_support::physical_device() else {
            return;
        };
        let mut device = match DeviceState::new(&physical_device) {
            Ok(device) => device,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        if !device.supports_sync_file_export() {
            super::super::test_support::unavailable("sync_file export");
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
            .export()
            .expect("submission must carry an exportable sync_file");
        assert!(
            fence.export().is_some(),
            "sync_file export must be an idempotent dup, not a consuming operation"
        );

        fence.wait().expect("fence wait should succeed");
        assert!(
            fence.is_reached(),
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
