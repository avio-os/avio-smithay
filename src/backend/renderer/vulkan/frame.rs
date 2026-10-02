use std::sync::Arc;

use ash::vk;
use tracing::{instrument, trace, warn};

use crate::{
    backend::{
        allocator::{
            dmabuf::Dmabuf,
            format::{has_alpha, FormatSet},
        },
        renderer::{
            sync::SyncPoint, Bind, Blit, BlitFrame, BottomEdgeClip, Color32F, ContextId, Frame, ImportDma,
            RenderTargetAccess, Renderer, RendererSuper, RoundedClip, Texture, TextureFilter,
            TextureRenderEffect,
        },
    },
    utils::{Buffer as BufferCoord, Physical, Point, Rectangle, Size, Transform},
};

#[cfg(feature = "wayland_frontend")]
use crate::backend::renderer::ImportDmaWl;

use super::{
    blit::record_image_blit,
    descriptor::TextureSampler,
    format::srgb_channel_to_linear,
    image::{stage_access_for_layout, VulkanImage},
    kawase::ResolvedKawasePass,
    pipeline::{
        push_constants_bytes, PipelineHandles, SolidPushConstants, TexturePushConstants, TextureTransform,
    },
    sync::VulkanFence,
    target::VulkanTargetEncoding,
    VulkanKawasePass, VulkanRenderer, VulkanRendererError, VulkanRendererErrorKind, VulkanTarget,
    VulkanTexture,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum VulkanFrameState {
    #[default]
    Idle,
    Recording,
    Finished,
    Aborted,
}

#[derive(Debug)]
struct FrameRecording {
    command_buffer: vk::CommandBuffer,
    framebuffer: vk::Framebuffer,
    storage: super::recording_storage::RecordingStorageLease,
    target: Arc<VulkanImage>,
    encoding: VulkanTargetEncoding,
    pipelines: PipelineHandles,
    transform: Transform,
    output_size: Size<i32, Physical>,
    size: Size<i32, Physical>,
}

impl std::ops::Deref for FrameRecording {
    type Target = super::recording_storage::RecordingStorage;
    fn deref(&self) -> &Self::Target {
        &self.storage
    }
}
impl std::ops::DerefMut for FrameRecording {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.storage
    }
}
impl FrameRecording {
    fn prepare_framebuffers(&mut self) {
        let framebuffer = self.framebuffer;
        let storage = &mut *self.storage;
        storage.submitted_framebuffers.push(framebuffer);
        storage
            .submitted_framebuffers
            .append(&mut storage.effect_framebuffers);
    }
}

#[derive(Debug)]
struct FrameResumeContext {
    target: Arc<VulkanImage>,
    encoding: VulkanTargetEncoding,
    pipelines: PipelineHandles,
    transform: Transform,
    output_size: Size<i32, Physical>,
    size: Size<i32, Physical>,
}

/// In-flight Vulkan renderer frame recording context.
#[derive(Debug)]
pub struct VulkanFrame<'frame> {
    renderer: &'frame mut VulkanRenderer,
    state: VulkanFrameState,
    recording: Option<FrameRecording>,
    draw_alpha: f32,
    outer_rounded_clip: Option<RoundedClip>,
    owner_sample_replay: Option<OwnerSampleReplay>,
    resolved_sample_texture: bool,
    submitted_segment: bool,
}

impl RendererSuper for VulkanRenderer {
    type Error = VulkanRendererError;
    type TextureId = VulkanTexture;
    type Framebuffer<'buffer> = VulkanTarget;
    type Frame<'frame, 'buffer>
        = VulkanFrame<'frame>
    where
        'buffer: 'frame,
        Self: 'frame;
}

impl Renderer for VulkanRenderer {
    fn context_id(&self) -> ContextId<VulkanTexture> {
        self.context_id.clone()
    }

    fn downscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        self.downscale_filter = filter;
        Ok(())
    }

    fn upscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        self.upscale_filter = filter;
        Ok(())
    }

    fn set_debug_flags(&mut self, flags: crate::backend::renderer::DebugFlags) {
        self.debug_flags = flags;
    }

    fn debug_flags(&self) -> crate::backend::renderer::DebugFlags {
        self.debug_flags
    }

    #[instrument(level = "trace", skip(self, target))]
    #[profiling::function]
    fn render<'frame, 'buffer>(
        &'frame mut self,
        target: &'frame mut Self::Framebuffer<'buffer>,
        output_size: Size<i32, Physical>,
        dst_transform: Transform,
    ) -> Result<Self::Frame<'frame, 'buffer>, Self::Error>
    where
        'buffer: 'frame,
    {
        trace!(
            ?output_size,
            ?dst_transform,
            "starting vulkan render pass recording"
        );
        self.device.reclaim_completed_submissions()?;

        if output_size.w <= 0 || output_size.h <= 0 {
            return Err(VulkanRendererError::TemporaryFailure(
                "render target size must be positive",
            ));
        }

        let Some(target_image) = target.image_resource().cloned() else {
            return Err(VulkanRendererError::NotImplemented(
                "Renderer::render currently requires dma-buf-backed VulkanTarget",
            ));
        };

        let target_size = target_image.size();
        let transformed_size = dst_transform.transform_size(output_size);
        if transformed_size.w <= 0 || transformed_size.h <= 0 {
            return Err(VulkanRendererError::TemporaryFailure(
                "transformed render size must be positive",
            ));
        }

        if transformed_size.w > target_size.w || transformed_size.h > target_size.h {
            return Err(VulkanRendererError::TemporaryFailure(
                "requested render size exceeds bound Vulkan target dimensions",
            ));
        }

        // Keyed on the attachment view's format, not the image's storage format: the
        // render pass and pipelines must agree with the `_SRGB` view that makes the
        // fixed-function blend operate in linear light.
        let pipelines = self
            .pipelines
            .pipelines_for_format(target.encoding.format(target_image.vk_format()))?;

        let storage = self.device.acquire_recording_storage()?;
        let command_buffer = self.device.acquire_command_buffer()?;
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: Command buffer belongs to this device command pool and is not currently in use.
        if let Err(error) = unsafe {
            self.device
                .device_handle()
                .begin_command_buffer(command_buffer, &begin_info)
        } {
            let _ = self.device.discard_command_buffer(command_buffer);
            return Err(error.into());
        }
        self.device
            .insert_debug_label(command_buffer, c"vulkan.render.begin", [0.17, 0.42, 0.86, 1.0]);

        let framebuffer = match create_framebuffer(
            self.device.device_handle(),
            pipelines.render_pass,
            target.encoding.view(&target_image),
            transformed_size,
        ) {
            Ok(framebuffer) => framebuffer,
            Err(error) => {
                let _ = self.device.discard_command_buffer(command_buffer);
                return Err(error);
            }
        };

        let mut frame = VulkanFrame {
            renderer: self,
            state: VulkanFrameState::Recording,
            submitted_segment: false,
            draw_alpha: 1.0,
            outer_rounded_clip: None,
            owner_sample_replay: None,
            resolved_sample_texture: false,
            recording: Some(FrameRecording {
                command_buffer,
                framebuffer,
                storage,
                target: target_image,
                encoding: target.encoding,
                pipelines,
                transform: dst_transform,
                output_size,
                size: transformed_size,
            }),
        };

        {
            let recording = frame.recording.as_ref().expect("recording initialized");
            let target = recording.target.clone();
            frame.transition_image_layout(&target, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)?;
        }

        let render_pass_begin_info = {
            let recording = frame.recording.as_ref().expect("recording initialized");
            vk::RenderPassBeginInfo::default()
                .render_pass(recording.pipelines.render_pass)
                .framebuffer(recording.framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: recording.size.w as u32,
                        height: recording.size.h as u32,
                    },
                })
        };

        // SAFETY: All render pass/framebuffer handles are valid for this command buffer.
        unsafe {
            frame.renderer.device.insert_debug_label(
                command_buffer,
                c"vulkan.render.begin_pass",
                [0.20, 0.58, 0.95, 1.0],
            );
            frame.renderer.device.device_handle().cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
        }

        Ok(frame)
    }

    #[instrument(level = "trace", skip(self, sync))]
    #[profiling::function]
    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        wait_on_sync_point(self, sync, vk::PipelineStageFlags::ALL_COMMANDS)
    }

    #[instrument(level = "trace", skip(self))]
    #[profiling::function]
    fn cleanup_texture_cache(&mut self) -> Result<(), Self::Error> {
        self.device.reclaim_completed_submissions()?;
        self.dmabuf.cleanup();
        self.descriptors.clear_texture_cache()?;
        Ok(())
    }
}

impl Bind<Dmabuf> for VulkanRenderer {
    fn bind<'a>(&mut self, target: &'a mut Dmabuf) -> Result<Self::Framebuffer<'a>, Self::Error> {
        self.bind_dmabuf_target(target)
    }

    fn bind_with_access<'a>(
        &mut self,
        target: &'a mut Dmabuf,
        access: RenderTargetAccess,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        match access {
            RenderTargetAccess::Render => self.bind_dmabuf_target(target),
            RenderTargetAccess::FramebufferEffectSource => self.bind_dmabuf_framebuffer_effect_target(target),
            RenderTargetAccess::CaptureTarget => self.bind_dmabuf_capture_target(target),
        }
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Some(self.dmabuf_render_formats().iter().copied().collect())
    }
}

impl ImportDma for VulkanRenderer {
    fn dmabuf_formats(&self) -> FormatSet {
        self.dmabuf_import_formats().iter().copied().collect()
    }

    fn import_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        _damage: Option<&[Rectangle<i32, BufferCoord>]>,
    ) -> Result<Self::TextureId, Self::Error> {
        self.import_dmabuf_texture(dmabuf)
    }
}

#[cfg(feature = "wayland_frontend")]
impl ImportDmaWl for VulkanRenderer {}

impl Frame for VulkanFrame<'_> {
    type Error = VulkanRendererError;
    type TextureId = VulkanTexture;

    fn context_id(&self) -> ContextId<Self::TextureId> {
        self.renderer.context_id.clone()
    }

    #[instrument(level = "trace", skip(self, at))]
    #[profiling::function]
    fn clear(&mut self, color: Color32F, at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        if self.owner_sample_replay.is_some() {
            return Err(VulkanRendererError::TemporaryFailure(
                "owner replay cannot clear its frozen prefix",
            ));
        }
        let (command_buffer, transform, output_size, size, linear_blending) = {
            let recording = self.recording()?;
            (
                recording.command_buffer,
                recording.transform,
                recording.output_size,
                recording.size,
                recording.encoding.blends_in_linear_light(&recording.target),
            )
        };

        let mut clear_regions = Self::transformed_damage_rects(
            transform,
            output_size,
            size,
            Rectangle::from_size(output_size),
            at,
        )
        .peekable();
        if clear_regions.peek().is_none() {
            return Ok(());
        }

        let clear_attachments = [vk::ClearAttachment::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .color_attachment(0)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    // Vulkan treats a clear value as linear and encodes it for an `_SRGB`
                    // attachment, so the caller's sRGB-encoded colour is linearized here
                    // to land on exactly the same stored bytes as before.
                    float32: encode_color_for_target(color, linear_blending),
                },
            })];

        // SAFETY: Command buffer recording is active and clear regions are inside the current render area.
        unsafe {
            self.renderer.device.insert_debug_label(
                command_buffer,
                c"vulkan.frame.clear",
                [0.26, 0.73, 0.31, 1.0],
            );
            for rect in clear_regions {
                let clear = vk::ClearRect::default()
                    .rect(to_vk_rect(rect))
                    .base_array_layer(0)
                    .layer_count(1);
                self.renderer.device.device_handle().cmd_clear_attachments(
                    command_buffer,
                    &clear_attachments,
                    &[clear],
                );
            }
        }

        Ok(())
    }

    #[instrument(level = "trace", skip(self, damage))]
    #[profiling::function]
    fn draw_solid(
        &mut self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        color: Color32F,
    ) -> Result<(), Self::Error> {
        if self.outer_rounded_clip.is_some() && self.owner_sample_replay.is_none() {
            return Err(VulkanRendererError::NotImplemented(
                "solid draws require an explicit owner-mask packet",
            ));
        }
        if damage.is_empty() {
            return Ok(());
        }

        let (command_buffer, pipelines, transform, output_size, size) = {
            let recording = self.recording()?;
            (
                recording.command_buffer,
                recording.pipelines,
                recording.transform,
                recording.output_size,
                recording.size,
            )
        };

        let size = self.geometry_frame_size(size);
        let Some(viewport_rect) = self.draw_viewport_rect(transform, output_size, size, dst) else {
            return Ok(());
        };

        let draw_damage = Self::transformed_damage_rects(transform, output_size, size, dst, damage);
        let replay = self.owner_sample_replay;
        let mut draw_damage = draw_damage
            .filter_map(move |mut rect| {
                if let Some(replay) = replay {
                    rect.loc += replay.translation;
                    rect.intersection(replay.bounds)
                } else {
                    Some(rect)
                }
            })
            .peekable();
        if draw_damage.peek().is_none() {
            return Ok(());
        }

        let color = color * self.draw_alpha;
        let pipeline = if color.is_opaque() {
            pipelines.solid_opaque_pipeline
        } else {
            pipelines.solid_pipeline
        };

        // The shader premultiplies by alpha; linearizing the channels first makes that
        // a premultiplied-*linear* value, which is what a linear-light blend expects.
        let mut constants = SolidPushConstants {
            color: encode_color_for_target(
                color,
                self.recording()?
                    .encoding
                    .blends_in_linear_light(&self.recording()?.target),
            ),
            owner_radius: -1.0,
            ..SolidPushConstants::default()
        };
        if let Some(TransformedAnalyticClip::Rounded(clip)) = self.outer_rounded_clip.map(|clip| {
            self.shift_analytic_clip(transform_analytic_clip(
                transform,
                output_size,
                AnalyticClip::Rounded(clip),
            ))
        }) {
            constants.owner_rect = analytic_clip_rect_push_constant(clip.rect);
            constants.owner_radius = clip.radius;
        }

        // SAFETY: Command buffer recording is active and all pipeline/layout handles are valid.
        unsafe {
            self.renderer.device.insert_debug_label(
                command_buffer,
                c"vulkan.frame.draw_solid",
                [0.90, 0.56, 0.22, 1.0],
            );
            self.renderer.device.device_handle().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );

            self.renderer.device.device_handle().cmd_set_viewport(
                command_buffer,
                0,
                &[self.sample_viewport(viewport_rect)],
            );

            self.renderer.device.device_handle().cmd_push_constants(
                command_buffer,
                pipelines.solid_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                push_constants_bytes(&constants),
            );

            for rect in draw_damage {
                self.renderer
                    .device
                    .device_handle()
                    .cmd_set_scissor(command_buffer, 0, &[to_vk_rect(rect)]);
                self.renderer
                    .device
                    .device_handle()
                    .cmd_draw(command_buffer, 4, 1, 0, 0);
            }
        }

        Ok(())
    }

    fn canonical_coverage_lane(&self) -> Option<usize> {
        VulkanFrame::canonical_coverage_lane(self)
    }

    fn render_texture_from_to_resolved_sample_lane(
        &mut self,
        texture: &Self::TextureId,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
    ) -> Result<(), Self::Error> {
        let previous = self.resolved_sample_texture;
        self.resolved_sample_texture = true;
        let result = self.render_texture_from_to_internal(
            texture,
            src,
            dst,
            damage,
            &[],
            src_transform,
            alpha,
            None,
            TextureRenderEffect::NONE,
            None,
            None,
        );
        self.resolved_sample_texture = previous;
        result
    }

    #[instrument(level = "trace", skip(self, texture, damage, opaque_regions))]
    #[profiling::function]
    fn render_texture_from_to(
        &mut self,
        texture: &Self::TextureId,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
    ) -> Result<(), Self::Error> {
        self.render_texture_from_to_internal(
            texture,
            src,
            dst,
            damage,
            opaque_regions,
            src_transform,
            alpha,
            None,
            TextureRenderEffect::NONE,
            None,
            None,
        )
    }

    #[instrument(level = "trace", skip(self, texture, damage, opaque_regions))]
    #[profiling::function]
    fn render_texture_from_to_with_rounded_clip(
        &mut self,
        texture: &Self::TextureId,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
        rounded_clip: RoundedClip,
    ) -> Result<(), Self::Error> {
        self.render_texture_from_to_internal(
            texture,
            src,
            dst,
            damage,
            opaque_regions,
            src_transform,
            alpha,
            Some(AnalyticClip::Rounded(rounded_clip)),
            TextureRenderEffect::NONE,
            None,
            None,
        )
    }

    #[instrument(level = "trace", skip(self, texture, damage, opaque_regions))]
    #[profiling::function]
    fn render_texture_from_to_with_bottom_edge_clip(
        &mut self,
        texture: &VulkanTexture,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
        bottom_edge_clip: BottomEdgeClip,
    ) -> Result<(), Self::Error> {
        self.render_texture_from_to_internal(
            texture,
            src,
            dst,
            damage,
            opaque_regions,
            src_transform,
            alpha,
            Some(AnalyticClip::BottomEdge(bottom_edge_clip)),
            TextureRenderEffect::NONE,
            None,
            None,
        )
    }

    #[instrument(level = "trace", skip(self, texture, damage, opaque_regions))]
    #[profiling::function]
    fn render_texture_from_to_with_effect(
        &mut self,
        texture: &Self::TextureId,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
        effect: TextureRenderEffect,
    ) -> Result<(), Self::Error> {
        self.render_texture_from_to_internal(
            texture,
            src,
            dst,
            damage,
            opaque_regions,
            src_transform,
            alpha,
            None,
            effect,
            None,
            None,
        )
    }

    #[instrument(level = "trace", skip(self, texture, damage, opaque_regions))]
    #[profiling::function]
    fn render_texture_from_to_with_rounded_clip_and_effect(
        &mut self,
        texture: &Self::TextureId,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
        rounded_clip: RoundedClip,
        effect: TextureRenderEffect,
    ) -> Result<(), Self::Error> {
        self.render_texture_from_to_internal(
            texture,
            src,
            dst,
            damage,
            opaque_regions,
            src_transform,
            alpha,
            Some(AnalyticClip::Rounded(rounded_clip)),
            effect,
            None,
            None,
        )
    }

    fn transformation(&self) -> Transform {
        self.recording
            .as_ref()
            .map(|recording| recording.transform)
            .unwrap_or(Transform::Normal)
    }

    #[instrument(level = "trace", skip(self, sync))]
    #[profiling::function]
    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        wait_on_sync_point(
            self.renderer,
            sync,
            vk::PipelineStageFlags::ALL_GRAPHICS | vk::PipelineStageFlags::TRANSFER,
        )
    }

    fn completion_unobservable_on_error(&self) -> bool {
        self.submitted_segment
    }

    #[instrument(level = "trace", skip(self))]
    #[profiling::function]
    fn finish(mut self) -> Result<SyncPoint, Self::Error> {
        self.finish_internal()
    }
}

fn wait_on_sync_point(
    renderer: &mut VulkanRenderer,
    sync: &SyncPoint,
    wait_stage_mask: vk::PipelineStageFlags,
) -> Result<(), VulkanRendererError> {
    if sync.is_reached() {
        return Ok(());
    }
    if renderer.device.external_wait_already_staged(sync) {
        return Ok(());
    }

    if let Some(vulkan_fence) = sync.get::<VulkanFence>() {
        if vulkan_fence.belongs_to(&renderer.device.shared_device()) {
            // All contexts of this origin submit to one serialized native
            // queue. A completion exists only after its producer submitted.
            return Ok(());
        }
        // Exported render completions are queue dependencies, including when
        // a sibling renderer reads an immutable image copy. Never turn that
        // acquire into a host wait, or silently block after an import error.
        if sync.is_exportable() {
            let sync_file = sync.export().ok_or(VulkanRendererError::TemporaryFailure(
                "could not retain exported Vulkan completion",
            ))?;
            return renderer.device.queue_wait_on_sync_file_with_stage(
                sync_file,
                wait_stage_mask,
                Some(sync),
            );
        }
        if renderer.device.external_wait_storage_is_prepared() {
            return Err(VulkanRendererError::NotImplemented(
                "prepared external waits require an exportable foreign fence",
            ));
        }
        return vulkan_fence.wait_vk().map_err(Into::into);
    }

    if renderer.device.supports_sync_file_import() {
        if let Some(sync_file) = sync.export() {
            match renderer
                .device
                .queue_wait_on_sync_file_with_stage(sync_file, wait_stage_mask, Some(sync))
            {
                Ok(()) => return Ok(()),
                Err(err) => {
                    if renderer.device.external_wait_storage_is_prepared()
                        || err.kind() == VulkanRendererErrorKind::ContextLost
                    {
                        return Err(err);
                    }

                    warn!(
                        ?err,
                        "failed to import SyncPoint fd into Vulkan wait semaphore; falling back to blocking wait"
                    );
                }
            }
        }
    }

    if renderer.device.external_wait_storage_is_prepared() {
        return Err(VulkanRendererError::NotImplemented(
            "prepared external waits require an exportable native fence",
        ));
    }

    if renderer.device.supports_sync_file_fence_import() {
        if let Some(sync_file) = sync.export() {
            match renderer.device.wait_on_sync_file(sync_file) {
                Ok(()) => return Ok(()),
                Err(err) => {
                    if err.kind() == VulkanRendererErrorKind::ContextLost {
                        return Err(err);
                    }

                    warn!(
                        ?err,
                        "failed to import SyncPoint fd into Vulkan host wait; falling back to blocking wait"
                    );
                }
            }
        }
    }

    sync.wait()
        .map_err(|_| VulkanRendererError::TemporaryFailure("sync wait was interrupted"))
}

impl BlitFrame<VulkanTarget> for VulkanFrame<'_> {
    #[instrument(level = "trace", skip(self, to))]
    #[profiling::function]
    fn blit_to(
        &mut self,
        to: &mut VulkanTarget,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<(), Self::Error> {
        if self.owner_sample_replay.is_some() {
            return Err(VulkanRendererError::TemporaryFailure(
                "owner replay cannot submit a partial recording",
            ));
        }
        let resume_context = self.flush_recording_segment()?;
        let resume_target = Self::frame_target_from_context(&resume_context);
        let blit_result = self.renderer.blit(&resume_target, to, src, dst, filter);
        let resume_result = self.begin_recording_segment(resume_context);

        if let Err(err) = resume_result {
            return Err(err);
        }

        match blit_result {
            Ok(_) => Ok(()),
            Err(err) => Err(err),
        }
    }

    #[instrument(level = "trace", skip(self, from))]
    #[profiling::function]
    fn blit_from(
        &mut self,
        from: &VulkanTarget,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<(), Self::Error> {
        if self.owner_sample_replay.is_some() {
            return Err(VulkanRendererError::TemporaryFailure(
                "owner replay cannot submit a partial recording",
            ));
        }
        let resume_context = self.flush_recording_segment()?;
        let mut resume_target = Self::frame_target_from_context(&resume_context);
        let blit_result = self.renderer.blit(from, &mut resume_target, src, dst, filter);
        let resume_result = self.begin_recording_segment(resume_context);

        if let Err(err) = resume_result {
            return Err(err);
        }

        match blit_result {
            Ok(_) => Ok(()),
            Err(err) => Err(err),
        }
    }
}

impl VulkanFrame<'_> {
    /// Borrow one cold rectangle packet while allowing nested frame drawing.
    /// Bounds are checked before the callback; backing returns on every error
    /// and panic without changing draw order or splitting damage into draws.
    pub fn with_damage_scratch<T>(
        &mut self,
        required: usize,
        draw: impl FnOnce(&mut Self, &mut Vec<Rectangle<i32, Physical>>) -> Result<T, VulkanRendererError>,
    ) -> Result<T, VulkanRendererError> {
        let mut scratch = self.renderer.device.acquire_damage_scratch(required)?;
        draw(self, &mut scratch)
    }

    /// Interpolate the completed accumulator with its saved lower prefix.
    ///
    /// `group_opacity` weights the current attachment; the saved prefix gets
    /// its complement. The dedicated pipeline interpolates premultiplied RGBA,
    /// including alpha, so this also works for transparent capture targets.
    /// Call after drawing the complete group with its ordinary member alpha.
    ///
    /// `prefix` is the image [`Self::capture_and_filter_framebuffer`] filled
    /// for the same `region`, which must lie inside the output. Only the
    /// captured extent at the image's origin is sampled, so a reused image
    /// larger than the region is valid and its spare texels are never read.
    pub fn interpolate_framebuffer_prefix(
        &mut self,
        prefix: &VulkanTexture,
        region: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        group_opacity: f32,
    ) -> Result<(), VulkanRendererError> {
        if !group_opacity.is_finite() || !(0.0..=1.0).contains(&group_opacity) {
            return Err(VulkanRendererError::TemporaryFailure("invalid group opacity"));
        }
        let Some(region) = self
            .owner_sample_replay
            .map_or(Some(region), |replay| replay.clip_output_region(region))
        else {
            return Ok(());
        };
        // The capture blitted exactly this framebuffer area to the origin.
        let Some(captured) = self.effect_framebuffer_rect(region)? else {
            return Ok(());
        };
        let prefix_size = prefix.size();
        if prefix_size.w < captured.size.w || prefix_size.h < captured.size.h {
            return Err(VulkanRendererError::TemporaryFailure(
                "framebuffer prefix image is smaller than its captured extent",
            ));
        }
        let source =
            Rectangle::from_size(Size::<i32, BufferCoord>::from((captured.size.w, captured.size.h)).to_f64());
        let previous = self.resolved_sample_texture;
        self.resolved_sample_texture = self.owner_sample_replay.is_some();
        let result = self.render_texture_from_to_internal(
            prefix,
            source,
            region,
            damage,
            &[],
            self.transformation(),
            1.0,
            None,
            TextureRenderEffect::NONE,
            Some(1.0 - group_opacity),
            None,
        );
        self.resolved_sample_texture = previous;
        result
    }

    fn frame_target_from_context(context: &FrameResumeContext) -> VulkanTarget {
        let format = Some(context.target.format().code);
        let mut target =
            VulkanTarget::from_image_resource(context.target.clone(), context.target.size(), format);
        target.encoding = context.encoding;
        target
    }

    fn recording(&self) -> Result<&FrameRecording, VulkanRendererError> {
        if self.state != VulkanFrameState::Recording {
            return Err(VulkanRendererError::TemporaryFailure(
                "frame is no longer recording",
            ));
        }

        self.recording
            .as_ref()
            .ok_or(VulkanRendererError::TemporaryFailure(
                "frame recording context is unavailable",
            ))
    }

    fn recording_mut(&mut self) -> Result<&mut FrameRecording, VulkanRendererError> {
        if self.state != VulkanFrameState::Recording {
            return Err(VulkanRendererError::TemporaryFailure(
                "frame is no longer recording",
            ));
        }

        self.recording
            .as_mut()
            .ok_or(VulkanRendererError::TemporaryFailure(
                "frame recording context is unavailable",
            ))
    }

    fn transition_image_layout(
        &mut self,
        image: &Arc<VulkanImage>,
        new_layout: vk::ImageLayout,
    ) -> Result<(), VulkanRendererError> {
        let (command_buffer, old_layout) = {
            let recording = self.recording_mut()?;
            recording.admit_image(image.id())?;
            if image.uses_foreign_queue() {
                recording
                    .foreign_release_images
                    .entry(image.id())
                    .or_insert_with(|| image.clone());
            }
            let old_layout = recording
                .pending_layouts
                .get(&image.id())
                .map(|(_, layout)| *layout)
                .unwrap_or_else(|| image.current_layout());
            (recording.command_buffer, old_layout)
        };
        let acquire_from_foreign = image.take_foreign_ownership();
        if acquire_from_foreign {
            self.recording_mut()?
                .unsubmitted_foreign_acquires
                .entry(image.id())
                .or_insert_with(|| image.clone());
        }

        if old_layout == new_layout && !acquire_from_foreign {
            self.recording_mut()?
                .pending_layouts
                .entry(image.id())
                .or_insert_with(|| (image.clone(), new_layout));
            return Ok(());
        }

        let (mut src_stage_mask, mut src_access_mask) = stage_access_for_layout(old_layout);
        let (dst_stage_mask, dst_access_mask) = stage_access_for_layout(new_layout);
        let (src_queue_family_index, dst_queue_family_index) = if acquire_from_foreign {
            src_stage_mask = vk::PipelineStageFlags::TOP_OF_PIPE;
            src_access_mask = vk::AccessFlags::empty();
            (
                vk::QUEUE_FAMILY_FOREIGN_EXT,
                self.renderer.device.queue_family_index(),
            )
        } else {
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        };

        let barrier = [vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(src_queue_family_index)
            .dst_queue_family_index(dst_queue_family_index)
            .image(image.image())
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)];

        // SAFETY: Command buffer recording is active; barrier references a valid image owned by this renderer.
        unsafe {
            self.renderer.device.device_handle().cmd_pipeline_barrier(
                command_buffer,
                src_stage_mask,
                dst_stage_mask,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
        }

        self.recording_mut()?
            .pending_layouts
            .insert(image.id(), (image.clone(), new_layout));

        Ok(())
    }

    fn release_recording_images_to_foreign(&mut self, recording: &mut FrameRecording) {
        if recording.foreign_release_images.is_empty() {
            return;
        }

        let new_layout = vk::ImageLayout::GENERAL;
        let storage = &mut *recording.storage;
        storage.barriers.clear();
        let layouts = &storage.pending_layouts;
        let foreign = &storage.foreign_release_images;
        storage.barriers.extend(foreign.values().map(|image| {
            let old_layout = layouts
                .get(&image.id())
                .map(|(_, layout)| *layout)
                .unwrap_or_else(|| image.current_layout());
            let (_, src_access_mask) = stage_access_for_layout(old_layout);
            vk::ImageMemoryBarrier::default()
                .old_layout(old_layout)
                .new_layout(new_layout)
                .src_queue_family_index(self.renderer.device.queue_family_index())
                .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .image(image.image())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .src_access_mask(src_access_mask)
                .dst_access_mask(vk::AccessFlags::empty())
        }));

        // SAFETY: Command buffer recording is active after the render pass has ended.
        // All imported images remain alive through submission. The release
        // barriers return exclusive queue-family ownership after this frame's
        // final compositor access.
        unsafe {
            self.renderer.device.device_handle().cmd_pipeline_barrier(
                recording.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &storage.barriers,
            );
        }

        for image in storage.foreign_release_images.values() {
            storage
                .pending_layouts
                .insert(image.id(), (image.clone(), new_layout));
        }
    }

    fn restore_unsubmitted_foreign_acquires(recording: &mut FrameRecording) {
        for (_, image) in recording.unsubmitted_foreign_acquires.drain(..) {
            image.set_foreign_ownership();
        }
    }

    /// Jointly evaluate an immutable owner mask with a draw's existing clip
    /// at Vulkan's canonical four sample identities, before resolving coverage.
    /// This scope records no image, descriptor, or allocation of its own.
    pub fn with_outer_rounded_clip<T>(
        &mut self,
        clip: Option<RoundedClip>,
        draw: impl FnOnce(&mut Self) -> Result<T, VulkanRendererError>,
    ) -> Result<T, VulkanRendererError> {
        let previous = self.outer_rounded_clip;
        if previous.is_some() && clip.is_some() && previous != clip {
            return Err(VulkanRendererError::TemporaryFailure(
                "nested distinct outer clips exceed the declared sample-mask packet",
            ));
        }
        self.outer_rounded_clip = clip.or(previous);
        let result = draw(self);
        self.outer_rounded_clip = previous;
        result
    }

    /// Apply a group opacity to independent source-over primitives whose
    /// destination supports are disjoint. Framebuffer effects must retain their
    /// ordinary prefix-interpolation path; they cannot use this operation.
    pub fn with_draw_alpha<T>(
        &mut self,
        alpha: f32,
        draw: impl FnOnce(&mut Self) -> Result<T, VulkanRendererError>,
    ) -> Result<T, VulkanRendererError> {
        let previous = self.draw_alpha;
        self.draw_alpha *= alpha;
        let result = draw(self);
        self.draw_alpha = previous;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn render_texture_from_to_internal(
        &mut self,
        texture: &VulkanTexture,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
        analytic_clip: Option<AnalyticClip>,
        effect: TextureRenderEffect,
        prefix_weight: Option<f32>,
        material_operation: Option<MaterialTextureOperation>,
    ) -> Result<(), VulkanRendererError> {
        let alpha = alpha * self.draw_alpha;
        if damage.is_empty() {
            return Ok(());
        }

        let Some(texture_image) = texture.image_resource().cloned() else {
            return Err(VulkanRendererError::NotImplemented(
                "render_texture_from_to currently requires dma-buf-backed VulkanTexture",
            ));
        };

        self.transition_image_layout(&texture_image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;

        let (command_buffer, pipelines, transform, output_size, size, linear_blending) = {
            let recording = self.recording()?;
            (
                recording.command_buffer,
                recording.pipelines,
                recording.transform,
                recording.output_size,
                recording.size,
                recording.encoding.blends_in_linear_light(&recording.target),
            )
        };

        let size = self.geometry_frame_size(size);
        let Some(viewport_rect) = self.draw_viewport_rect(transform, output_size, size, dst) else {
            return Ok(());
        };

        if prefix_weight.is_some() && self.outer_rounded_clip.is_some() && self.owner_sample_replay.is_none()
        {
            return Err(VulkanRendererError::TemporaryFailure(
                "owner mask cannot clip a constant-factor prefix resolve",
            ));
        }
        let outer_clip = self
            .outer_rounded_clip
            .filter(|_| prefix_weight.is_none())
            .map(|clip| {
                self.shift_analytic_clip(transform_analytic_clip(
                    transform,
                    output_size,
                    AnalyticClip::Rounded(clip),
                ))
            });
        let analytic_clip = analytic_clip
            .map(|clip| self.shift_analytic_clip(transform_analytic_clip(transform, output_size, clip)));
        let has_analytic_clip = analytic_clip
            .map(TransformedAnalyticClip::has_coverage_mask)
            .unwrap_or(false)
            || outer_clip.is_some();

        let draw_damage = Self::transformed_damage_rects(transform, output_size, size, dst, damage);
        let replay = self.owner_sample_replay;
        let mut draw_damage = draw_damage
            .filter_map(move |mut rect| {
                if let Some(replay) = replay {
                    rect.loc += replay.translation;
                    rect.intersection(replay.bounds)
                } else {
                    Some(rect)
                }
            })
            .peekable();
        if draw_damage.peek().is_none() {
            return Ok(());
        }

        let texture_size = texture.size();
        if texture_size.w <= 0 || texture_size.h <= 0 {
            return Err(VulkanRendererError::TemporaryFailure(
                "texture dimensions must be positive",
            ));
        }

        if let Some(MaterialTextureOperation::Kawase(output)) = material_operation {
            if output.extent.w <= 0
                || output.extent.h <= 0
                || output.extent.w > texture_size.w
                || output.extent.h > texture_size.h
                || !output.offset.is_finite()
                || output.offset < 0.0
            {
                return Err(VulkanRendererError::TemporaryFailure(
                    "invalid fused Kawase source extent",
                ));
            }
        }
        if src.size.w <= 0.0 || src.size.h <= 0.0 {
            return Ok(());
        }

        if src.loc.x < 0.0
            || src.loc.y < 0.0
            || src.loc.x + src.size.w > texture_size.w as f64
            || src.loc.y + src.size.h > texture_size.h as f64
        {
            return Err(VulkanRendererError::TemporaryFailure(
                "source rectangle must remain within the source texture bounds",
            ));
        }

        let texture_transform = if prefix_weight.is_some()
            || material_operation.is_some_and(MaterialTextureOperation::framebuffer_coordinates)
        {
            Transform::Normal
        } else {
            combine_image_transform(src_transform, transform)
        };
        let src_offset = [
            src.loc.x as f32 / texture_size.w as f32,
            src.loc.y as f32 / texture_size.h as f32,
        ];
        let src_scale = [
            src.size.w as f32 / texture_size.w as f32,
            src.size.h as f32 / texture_size.h as f32,
        ];

        let mut push_constants = TexturePushConstants::new(
            alpha,
            TextureTransform::from(texture_transform),
            texture.y_inverted(),
        )
        .with_src_rect(src_offset, src_scale)
        .with_source_encoding(texture_image.color_encoding(), linear_blending)
        .with_effect(effect);

        if let Some(TransformedAnalyticClip::Rounded(clip)) = outer_clip {
            push_constants = push_constants.with_outer_rounded_clip(
                [clip.rect.loc.x as f32, clip.rect.loc.y as f32],
                [clip.rect.size.w as f32, clip.rect.size.h as f32],
                clip.radius,
            );
        }
        if let Some(operation) = material_operation {
            push_constants = match operation {
                MaterialTextureOperation::Kawase(output) => {
                    push_constants.with_kawase_output(output, texture_size)
                }
                MaterialTextureOperation::FramebufferCopy => push_constants,
                MaterialTextureOperation::OwnerResolve(extent) => {
                    push_constants.with_owner_resolve(extent, texture_size)
                }
                MaterialTextureOperation::Tint(tint) => {
                    push_constants.with_material_tone(tint).with_source_encoding(
                        super::format::ColorEncoding::ElectricalPremultiplied,
                        linear_blending,
                    )
                }
            };
        }
        if let Some(clip) = analytic_clip {
            push_constants = match clip {
                TransformedAnalyticClip::Rounded(clip) => push_constants.with_rounded_clip(
                    clip.corner_mask,
                    analytic_clip_rect_push_constant(clip.rect),
                    [
                        clip.radius.max(0.0),
                        clip.exponent.max(2.0),
                        clip.aa_width.max(0.001),
                        0.0,
                    ],
                ),
                TransformedAnalyticClip::BottomEdge { clip, transform } => push_constants
                    .with_bottom_edge_clip(
                        transform,
                        analytic_clip_rect_push_constant(clip.rect),
                        [
                            clip.content_width.max(0.0),
                            clip.progress.max(0.0),
                            clip.edge_height.max(0.0),
                            clip.plateau_inset.max(0.0),
                        ],
                        clip.geometry_scale.max(0.001),
                    ),
            };
        }

        if self.owner_sample_replay.is_some() {
            push_constants.rounded_clip_flags |= 1 << 30;
        }
        let sampler = if matches!(
            material_operation,
            Some(MaterialTextureOperation::OwnerResolve(_))
        ) || self.resolved_sample_texture
        {
            TextureSampler::NEAREST
        } else if material_operation.is_some() || self.owner_sample_replay.is_some() {
            TextureSampler::LINEAR
        } else {
            texture_sampler_for_render(
                src,
                dst,
                src_transform,
                self.renderer.downscale_filter,
                self.renderer.upscale_filter,
            )
        };
        let descriptor_set = self.renderer.descriptors.texture_descriptor_set(
            texture_image.view(),
            texture_image.incarnation(),
            sampler,
        )?;

        let texture_has_alpha = texture.format().map(has_alpha).unwrap_or(true);
        let has_shader_effect = !effect.is_none() || material_operation.is_some();
        let use_opaque_only = prefix_weight.is_none()
            && alpha >= 1.0
            && !texture_has_alpha
            && !has_analytic_clip
            && !has_shader_effect;

        let can_use_opaque =
            prefix_weight.is_none() && alpha >= 1.0 && !has_analytic_clip && !has_shader_effect;
        let classify_opaque = |rect: Rectangle<i32, Physical>| {
            use_opaque_only
                || (can_use_opaque
                    && Self::transformed_damage_rects(transform, output_size, size, dst, opaque_regions)
                        .any(|opaque| opaque.contains_rect(rect)))
        };
        let mut opaque_draws = draw_damage
            .clone()
            .filter(|rect| classify_opaque(*rect))
            .peekable();
        let mut blended_draws = draw_damage.filter(|rect| !classify_opaque(*rect)).peekable();

        // SAFETY: Command buffer recording is active and all handles belong to this renderer device.
        unsafe {
            self.renderer.device.insert_debug_label(
                command_buffer,
                c"vulkan.frame.render_texture",
                [0.74, 0.34, 0.89, 1.0],
            );
            self.renderer.device.device_handle().cmd_set_viewport(
                command_buffer,
                0,
                &[self.sample_viewport(viewport_rect)],
            );

            if opaque_draws.peek().is_some() {
                self.renderer.device.device_handle().cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipelines.textured_opaque_pipeline,
                );
                self.renderer.device.device_handle().cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipelines.textured_layout,
                    0,
                    &[descriptor_set],
                    &[],
                );
                self.renderer.device.device_handle().cmd_push_constants(
                    command_buffer,
                    pipelines.textured_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    push_constants_bytes(&push_constants),
                );

                for rect in opaque_draws {
                    self.renderer.device.device_handle().cmd_set_scissor(
                        command_buffer,
                        0,
                        &[to_vk_rect(rect)],
                    );
                    self.renderer
                        .device
                        .device_handle()
                        .cmd_draw(command_buffer, 4, 1, 0, 0);
                }
            }

            if blended_draws.peek().is_some() {
                if let Some(weight) = prefix_weight {
                    self.renderer
                        .device
                        .device_handle()
                        .cmd_set_blend_constants(command_buffer, &[weight; 4]);
                }
                self.renderer.device.device_handle().cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    if prefix_weight.is_some() {
                        pipelines.prefix_mix_pipeline
                    } else {
                        pipelines.textured_pipeline
                    },
                );
                self.renderer.device.device_handle().cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipelines.textured_layout,
                    0,
                    &[descriptor_set],
                    &[],
                );
                self.renderer.device.device_handle().cmd_push_constants(
                    command_buffer,
                    pipelines.textured_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    push_constants_bytes(&push_constants),
                );

                for rect in blended_draws {
                    self.renderer.device.device_handle().cmd_set_scissor(
                        command_buffer,
                        0,
                        &[to_vk_rect(rect)],
                    );
                    self.renderer
                        .device
                        .device_handle()
                        .cmd_draw(command_buffer, 4, 1, 0, 0);
                }
            }
        }

        Ok(())
    }

    fn transformed_damage_rects<'a>(
        transform: Transform,
        output_size: Size<i32, Physical>,
        target_size: Size<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        damage: &'a [Rectangle<i32, Physical>],
    ) -> impl Iterator<Item = Rectangle<i32, Physical>> + Clone + 'a {
        let dst_bounds = dst;

        damage.iter().filter_map(move |rect| {
            let absolute = Rectangle::new(
                (
                    dst.loc.x.saturating_add(rect.loc.x),
                    dst.loc.y.saturating_add(rect.loc.y),
                )
                    .into(),
                rect.size,
            );

            absolute
                .intersection(dst_bounds)
                .and_then(|clipped| framebuffer_rect(transform, output_size, target_size, clipped))
                .filter(|region| region.size.w > 0 && region.size.h > 0)
        })
    }

    #[instrument(level = "trace", skip(self))]
    #[profiling::function]
    fn finish_internal(&mut self) -> Result<SyncPoint, VulkanRendererError> {
        if self.state != VulkanFrameState::Recording {
            return Ok(SyncPoint::signaled());
        }

        let mut recording = self
            .recording
            .take()
            .ok_or(VulkanRendererError::TemporaryFailure(
                "frame recording context was already consumed",
            ))?;

        // SAFETY: Render pass was begun when entering frame recording.
        unsafe {
            self.renderer.device.insert_debug_label(
                recording.command_buffer,
                c"vulkan.frame.finish",
                [0.95, 0.25, 0.35, 1.0],
            );
            self.renderer
                .device
                .device_handle()
                .cmd_end_render_pass(recording.command_buffer);
        }
        self.release_recording_images_to_foreign(&mut recording);
        recording.prepare_framebuffers();

        // SAFETY: Command buffer recording is valid and render pass has been ended.
        if let Err(err) = self.renderer.device.shared_device().observe_result(unsafe {
            self.renderer
                .device
                .device_handle()
                .end_command_buffer(recording.command_buffer)
        }) {
            let _ = self.renderer.device.discard_recording_resources(
                recording.command_buffer,
                &mut recording.storage.submitted_framebuffers,
            );
            self.renderer.descriptors.abort_recording();
            Self::restore_unsubmitted_foreign_acquires(&mut recording);
            self.renderer.device.clear_pending_wait_semaphores();
            self.state = VulkanFrameState::Aborted;
            if self.renderer.device.completion_unknown() {
                self.renderer.device.preserve_failed_recording(recording.storage);
            }
            return Err(err.into());
        }

        retained_recording_images(&mut recording);
        let storage = &mut *recording.storage;
        let (submission_id, submission_fence) = match self.renderer.device.submit_with_resources_and_fence(
            recording.command_buffer,
            &mut storage.submitted_framebuffers,
            &mut storage.retained_images,
        ) {
            Ok(submission) => submission,
            Err(err) => {
                self.renderer.descriptors.abort_recording();
                Self::restore_unsubmitted_foreign_acquires(&mut recording);
                self.renderer.device.clear_pending_wait_semaphores();
                self.state = VulkanFrameState::Aborted;
                if self.renderer.device.completion_unknown() {
                    self.renderer.device.preserve_failed_recording(recording.storage);
                }
                return Err(err);
            }
        };
        self.renderer.descriptors.commit_submission(submission_id);

        for (_, (image, layout)) in recording.pending_layouts.drain(..) {
            image.set_layout(layout);
        }
        for (_, image) in recording.foreign_release_images.drain(..) {
            image.set_foreign_ownership();
        }
        recording.unsubmitted_foreign_acquires.clear();

        self.state = VulkanFrameState::Finished;
        Ok(submission_fence)
    }

    fn flush_recording_segment(&mut self) -> Result<FrameResumeContext, VulkanRendererError> {
        let mut recording = self
            .recording
            .take()
            .ok_or(VulkanRendererError::TemporaryFailure(
                "frame recording context was already consumed",
            ))?;

        // SAFETY: Render pass was begun when entering frame recording.
        unsafe {
            self.renderer.device.insert_debug_label(
                recording.command_buffer,
                c"vulkan.frame.flush_segment",
                [0.86, 0.49, 0.18, 1.0],
            );
            self.renderer
                .device
                .device_handle()
                .cmd_end_render_pass(recording.command_buffer);
        }
        // A flushed segment is a complete queue submission. Return every
        // imported image to FOREIGN here rather than carrying local ownership
        // across the standalone blit between segments. This makes every
        // submitted segment independently safe if the blit or resume fails.
        self.release_recording_images_to_foreign(&mut recording);
        recording.prepare_framebuffers();

        // SAFETY: Command buffer recording is valid and render pass has been ended.
        if let Err(err) = self.renderer.device.shared_device().observe_result(unsafe {
            self.renderer
                .device
                .device_handle()
                .end_command_buffer(recording.command_buffer)
        }) {
            let _ = self.renderer.device.discard_recording_resources(
                recording.command_buffer,
                &mut recording.storage.submitted_framebuffers,
            );
            self.renderer.descriptors.abort_recording();
            Self::restore_unsubmitted_foreign_acquires(&mut recording);
            self.renderer.device.clear_pending_wait_semaphores();
            self.state = VulkanFrameState::Aborted;
            if self.renderer.device.completion_unknown() {
                self.renderer.device.preserve_failed_recording(recording.storage);
            }
            return Err(err.into());
        }

        retained_recording_images(&mut recording);
        let storage = &mut *recording.storage;
        let submission_id = match self.renderer.device.submit_with_resources(
            recording.command_buffer,
            &mut storage.submitted_framebuffers,
            &mut storage.retained_images,
        ) {
            Ok(submission_id) => submission_id,
            Err(err) => {
                self.renderer.descriptors.abort_recording();
                Self::restore_unsubmitted_foreign_acquires(&mut recording);
                self.renderer.device.clear_pending_wait_semaphores();
                self.state = VulkanFrameState::Aborted;
                if self.renderer.device.completion_unknown() {
                    self.renderer.device.preserve_failed_recording(recording.storage);
                }
                return Err(err);
            }
        };
        self.submitted_segment = true;
        self.renderer.descriptors.commit_submission(submission_id);

        for (_, (image, layout)) in recording.pending_layouts.drain(..) {
            image.set_layout(layout);
        }
        for (_, image) in recording.foreign_release_images.drain(..) {
            image.set_foreign_ownership();
        }
        recording.unsubmitted_foreign_acquires.clear();

        self.state = VulkanFrameState::Idle;
        Ok(FrameResumeContext {
            target: recording.target,
            encoding: recording.encoding,
            pipelines: recording.pipelines,
            transform: recording.transform,
            output_size: recording.output_size,
            size: recording.size,
        })
    }

    fn begin_recording_segment(&mut self, context: FrameResumeContext) -> Result<(), VulkanRendererError> {
        let storage = self.renderer.device.acquire_recording_storage()?;
        let command_buffer = self.renderer.device.acquire_command_buffer()?;
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: Command buffer belongs to this device command pool and is not currently in use.
        if let Err(err) = unsafe {
            self.renderer
                .device
                .device_handle()
                .begin_command_buffer(command_buffer, &begin_info)
        } {
            let _ = self.renderer.device.discard_command_buffer(command_buffer);
            self.renderer.device.clear_pending_wait_semaphores();
            self.state = VulkanFrameState::Aborted;
            return Err(err.into());
        }
        self.renderer.device.insert_debug_label(
            command_buffer,
            c"vulkan.render.resume",
            [0.17, 0.42, 0.86, 1.0],
        );

        let framebuffer = match create_framebuffer(
            self.renderer.device.device_handle(),
            context.pipelines.render_pass,
            context.encoding.view(&context.target),
            context.size,
        ) {
            Ok(framebuffer) => framebuffer,
            Err(err) => {
                let _ = self.renderer.device.discard_command_buffer(command_buffer);
                self.renderer.device.clear_pending_wait_semaphores();
                self.state = VulkanFrameState::Aborted;
                return Err(err);
            }
        };

        self.recording = Some(FrameRecording {
            command_buffer,
            framebuffer,
            storage,
            target: context.target.clone(),
            encoding: context.encoding,
            pipelines: context.pipelines,
            transform: context.transform,
            output_size: context.output_size,
            size: context.size,
        });
        self.state = VulkanFrameState::Recording;

        if let Err(err) =
            self.transition_image_layout(&context.target, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        {
            self.abort_recording();
            return Err(err);
        }

        let render_pass_begin_info = {
            let recording = self.recording.as_ref().expect("recording initialized");
            vk::RenderPassBeginInfo::default()
                .render_pass(recording.pipelines.render_pass)
                .framebuffer(recording.framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: recording.size.w as u32,
                        height: recording.size.h as u32,
                    },
                })
        };

        // SAFETY: All render pass/framebuffer handles are valid for this command buffer.
        unsafe {
            self.renderer.device.insert_debug_label(
                command_buffer,
                c"vulkan.render.resume_pass",
                [0.20, 0.58, 0.95, 1.0],
            );
            self.renderer.device.device_handle().cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
        }

        Ok(())
    }

    fn abort_recording(&mut self) {
        self.renderer.device.clear_pending_wait_semaphores();
        self.renderer.descriptors.abort_recording();

        if let Some(mut recording) = self.recording.take() {
            recording.prepare_framebuffers();
            if let Err(err) = self.renderer.device.discard_recording_resources(
                recording.command_buffer,
                &mut recording.storage.submitted_framebuffers,
            ) {
                warn!(?err, "failed to discard Vulkan frame resources during abort");
            }
            Self::restore_unsubmitted_foreign_acquires(&mut recording);
            if self.renderer.device.completion_unknown() {
                self.renderer.device.preserve_failed_recording(recording.storage);
            }
        }

        self.state = VulkanFrameState::Aborted;
    }
}

/// Shared mapping for draws, damage and backdrop capture. Clip in target space
/// only after rotating around the original output extent.
fn framebuffer_rect(
    transform: Transform,
    output_size: Size<i32, Physical>,
    target_size: Size<i32, Physical>,
    output_rect: Rectangle<i32, Physical>,
) -> Option<Rectangle<i32, Physical>> {
    transform
        .transform_rect_in(output_rect, &output_size)
        .intersection(Rectangle::from_size(target_size))
}

include!("frame_material.rs");
include!("frame_owner.rs");

impl Drop for VulkanFrame<'_> {
    fn drop(&mut self) {
        if self.state != VulkanFrameState::Recording {
            self.renderer.device.clear_pending_wait_semaphores();
            return;
        }

        self.abort_recording();
    }
}

fn retained_recording_images(recording: &mut FrameRecording) {
    let storage = &mut *recording.storage;
    storage.retained_images.push(recording.target.clone());
    storage.retained_images.extend(
        storage
            .pending_layouts
            .values()
            .filter(|(image, _)| image.id() != recording.target.id())
            .map(|(image, _)| image.clone()),
    );
    // Every foreign image was admitted to pending_layouts at its first access.
    // Retain each image once rather than doubling both maps' native custody.
}

fn create_framebuffer(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    image_view: vk::ImageView,
    size: Size<i32, Physical>,
) -> Result<vk::Framebuffer, VulkanRendererError> {
    let attachments = [image_view];
    let framebuffer_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&attachments)
        .width(size.w as u32)
        .height(size.h as u32)
        .layers(1);

    // SAFETY: Device is valid and create info references live handles.
    Ok(unsafe { device.create_framebuffer(&framebuffer_info, None) }?)
}

fn to_vk_rect(rect: Rectangle<i32, Physical>) -> vk::Rect2D {
    vk::Rect2D {
        offset: vk::Offset2D {
            x: rect.loc.x,
            y: rect.loc.y,
        },
        extent: vk::Extent2D {
            width: rect.size.w as u32,
            height: rect.size.h as u32,
        },
    }
}

/// Cancels the `_SRGB` attachment's encode-on-store for an API-supplied colour.
///
/// Against a linear-blending target each colour channel is linearized, so the
/// hardware's encode lands on exactly the byte the caller named — a pure round-trip
/// that holds whatever colour space or premultiplication the caller meant, which is
/// what keeps [`Frame::clear`] writing the same bytes it always did. For the solid
/// pipeline it additionally puts the channels in linear light before the shader
/// multiplies by alpha, so that premultiplication happens in linear too.
///
/// Against a gamma-blending target (formats with no `_SRGB` sibling) the channels
/// pass through untouched. Alpha is a coverage term and is never transformed.
fn encode_color_for_target(color: Color32F, linear_blending: bool) -> [f32; 4] {
    if linear_blending {
        [
            srgb_channel_to_linear(color.r()),
            srgb_channel_to_linear(color.g()),
            srgb_channel_to_linear(color.b()),
            color.a(),
        ]
    } else {
        [color.r(), color.g(), color.b(), color.a()]
    }
}

fn to_vk_viewport(rect: Rectangle<i32, Physical>) -> vk::Viewport {
    vk::Viewport {
        x: rect.loc.x as f32,
        y: rect.loc.y as f32,
        width: rect.size.w as f32,
        height: rect.size.h as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}

#[derive(Debug, Clone, Copy)]
enum AnalyticClip {
    Rounded(RoundedClip),
    BottomEdge(BottomEdgeClip),
}

#[derive(Debug, Clone, Copy)]
enum TransformedAnalyticClip {
    Rounded(RoundedClip),
    BottomEdge {
        clip: BottomEdgeClip,
        transform: TextureTransform,
    },
}

impl TransformedAnalyticClip {
    fn has_coverage_mask(self) -> bool {
        match self {
            Self::Rounded(clip) => clip.corner_mask != 0 && clip.radius > 0.0,
            Self::BottomEdge { .. } => true,
        }
    }
}

fn analytic_clip_rect_push_constant(rect: Rectangle<f64, Physical>) -> [f32; 4] {
    [
        rect.loc.x as f32,
        rect.loc.y as f32,
        rect.size.w as f32,
        rect.size.h as f32,
    ]
}

fn transform_analytic_clip(
    transform: Transform,
    frame_size: Size<i32, Physical>,
    clip: AnalyticClip,
) -> TransformedAnalyticClip {
    match clip {
        AnalyticClip::Rounded(clip) => {
            TransformedAnalyticClip::Rounded(transform_rounded_clip(transform, frame_size, clip))
        }
        AnalyticClip::BottomEdge(clip) => {
            let frame_size = frame_size.to_f64();
            TransformedAnalyticClip::BottomEdge {
                clip: BottomEdgeClip {
                    rect: transform.transform_rect_in(clip.rect, &frame_size),
                    ..clip
                },
                transform: TextureTransform::from(transform),
            }
        }
    }
}

fn transform_rounded_clip(
    transform: Transform,
    frame_size: Size<i32, Physical>,
    clip: RoundedClip,
) -> RoundedClip {
    let frame_size = frame_size.to_f64();
    let rect = transform.transform_rect_in(clip.rect, &frame_size);
    RoundedClip {
        rect,
        corner_mask: transform_corner_mask(transform, frame_size, clip.rect, rect, clip.corner_mask),
        ..clip
    }
}

fn transform_corner_mask(
    transform: Transform,
    frame_size: Size<f64, Physical>,
    source_rect: Rectangle<f64, Physical>,
    transformed_rect: Rectangle<f64, Physical>,
    source_mask: u32,
) -> u32 {
    let corners = [
        (
            RoundedClip::TOP_LEFT,
            Point::from((source_rect.loc.x, source_rect.loc.y)),
        ),
        (
            RoundedClip::TOP_RIGHT,
            Point::from((source_rect.loc.x + source_rect.size.w, source_rect.loc.y)),
        ),
        (
            RoundedClip::BOTTOM_RIGHT,
            Point::from((
                source_rect.loc.x + source_rect.size.w,
                source_rect.loc.y + source_rect.size.h,
            )),
        ),
        (
            RoundedClip::BOTTOM_LEFT,
            Point::from((source_rect.loc.x, source_rect.loc.y + source_rect.size.h)),
        ),
    ];

    corners
        .into_iter()
        .filter(|(flag, _)| source_mask & *flag != 0)
        .map(|(_, point)| {
            transformed_corner_flag(transform.transform_point_in(point, &frame_size), transformed_rect)
        })
        .fold(0, |mask, flag| mask | flag)
}

fn transformed_corner_flag(point: Point<f64, Physical>, rect: Rectangle<f64, Physical>) -> u32 {
    let center_x = rect.loc.x + rect.size.w * 0.5;
    let center_y = rect.loc.y + rect.size.h * 0.5;
    match (point.x <= center_x, point.y <= center_y) {
        (true, true) => RoundedClip::TOP_LEFT,
        (false, true) => RoundedClip::TOP_RIGHT,
        (false, false) => RoundedClip::BOTTOM_RIGHT,
        (true, false) => RoundedClip::BOTTOM_LEFT,
    }
}

fn texture_sampler_for_render(
    src: Rectangle<f64, BufferCoord>,
    dst: Rectangle<i32, Physical>,
    src_transform: Transform,
    downscale_filter: TextureFilter,
    upscale_filter: TextureFilter,
) -> TextureSampler {
    let source_size = source_size_in_destination_axes(src, src_transform);
    let destination_size = (dst.size.w.max(0) as f64, dst.size.h.max(0) as f64);
    if source_size.0 <= 0.0 || source_size.1 <= 0.0 || destination_size.0 <= 0.0 || destination_size.1 <= 0.0
    {
        return TextureSampler::LINEAR;
    }

    if source_rect_is_texel_aligned(src)
        && nearly_equal(source_size.0, destination_size.0)
        && nearly_equal(source_size.1, destination_size.1)
    {
        return TextureSampler::NEAREST;
    }

    TextureSampler::new(downscale_filter, upscale_filter)
}

fn source_size_in_destination_axes(src: Rectangle<f64, BufferCoord>, src_transform: Transform) -> (f64, f64) {
    if transform_swaps_axes(src_transform) {
        (src.size.h, src.size.w)
    } else {
        (src.size.w, src.size.h)
    }
}

fn transform_swaps_axes(transform: Transform) -> bool {
    matches!(
        transform,
        Transform::_90 | Transform::_270 | Transform::Flipped90 | Transform::Flipped270
    )
}

fn nearly_equal(lhs: f64, rhs: f64) -> bool {
    (lhs - rhs).abs() <= 0.001
}

fn source_rect_is_texel_aligned(src: Rectangle<f64, BufferCoord>) -> bool {
    nearly_equal(src.loc.x, src.loc.x.round()) && nearly_equal(src.loc.y, src.loc.y.round())
}

fn combine_image_transform(src_transform: Transform, output_transform: Transform) -> Transform {
    let src_transform = src_transform.invert();

    match (src_transform, output_transform) {
        (Transform::Normal, output_transform) => output_transform,

        (Transform::_90, Transform::Normal) => Transform::_270,
        (Transform::_90, Transform::_90) => Transform::Normal,
        (Transform::_90, Transform::_180) => Transform::_90,
        (Transform::_90, Transform::_270) => Transform::_180,
        (Transform::_90, Transform::Flipped) => Transform::Flipped90,
        (Transform::_90, Transform::Flipped90) => Transform::Flipped180,
        (Transform::_90, Transform::Flipped180) => Transform::Flipped270,
        (Transform::_90, Transform::Flipped270) => Transform::Flipped,

        (Transform::_180, Transform::Normal) => Transform::_180,
        (Transform::_180, Transform::_90) => Transform::_270,
        (Transform::_180, Transform::_180) => Transform::Normal,
        (Transform::_180, Transform::_270) => Transform::_90,
        (Transform::_180, Transform::Flipped) => Transform::Flipped180,
        (Transform::_180, Transform::Flipped90) => Transform::Flipped270,
        (Transform::_180, Transform::Flipped180) => Transform::Flipped,
        (Transform::_180, Transform::Flipped270) => Transform::Flipped90,

        (Transform::_270, Transform::Normal) => Transform::_90,
        (Transform::_270, Transform::_90) => Transform::_180,
        (Transform::_270, Transform::_180) => Transform::_270,
        (Transform::_270, Transform::_270) => Transform::Normal,
        (Transform::_270, Transform::Flipped) => Transform::Flipped270,
        (Transform::_270, Transform::Flipped90) => Transform::Flipped,
        (Transform::_270, Transform::Flipped180) => Transform::Flipped90,
        (Transform::_270, Transform::Flipped270) => Transform::Flipped180,

        (Transform::Flipped, Transform::Normal) => Transform::Flipped,
        (Transform::Flipped, Transform::_90) => Transform::Flipped90,
        (Transform::Flipped, Transform::_180) => Transform::Flipped180,
        (Transform::Flipped, Transform::_270) => Transform::Flipped270,
        (Transform::Flipped, Transform::Flipped) => Transform::Normal,
        (Transform::Flipped, Transform::Flipped90) => Transform::_90,
        (Transform::Flipped, Transform::Flipped180) => Transform::_180,
        (Transform::Flipped, Transform::Flipped270) => Transform::_270,

        (Transform::Flipped90, Transform::Normal) => Transform::Flipped90,
        (Transform::Flipped90, Transform::_90) => Transform::Flipped180,
        (Transform::Flipped90, Transform::_180) => Transform::Flipped270,
        (Transform::Flipped90, Transform::_270) => Transform::Flipped,
        (Transform::Flipped90, Transform::Flipped) => Transform::_270,
        (Transform::Flipped90, Transform::Flipped90) => Transform::Normal,
        (Transform::Flipped90, Transform::Flipped180) => Transform::_90,
        (Transform::Flipped90, Transform::Flipped270) => Transform::_180,

        (Transform::Flipped180, Transform::Normal) => Transform::Flipped180,
        (Transform::Flipped180, Transform::_90) => Transform::Flipped270,
        (Transform::Flipped180, Transform::_180) => Transform::Flipped,
        (Transform::Flipped180, Transform::_270) => Transform::Flipped90,
        (Transform::Flipped180, Transform::Flipped) => Transform::_180,
        (Transform::Flipped180, Transform::Flipped90) => Transform::_270,
        (Transform::Flipped180, Transform::Flipped180) => Transform::Normal,
        (Transform::Flipped180, Transform::Flipped270) => Transform::_90,

        (Transform::Flipped270, Transform::Normal) => Transform::Flipped270,
        (Transform::Flipped270, Transform::_90) => Transform::Flipped,
        (Transform::Flipped270, Transform::_180) => Transform::Flipped90,
        (Transform::Flipped270, Transform::_270) => Transform::Flipped180,
        (Transform::Flipped270, Transform::Flipped) => Transform::_90,
        (Transform::Flipped270, Transform::Flipped90) => Transform::_180,
        (Transform::Flipped270, Transform::Flipped180) => Transform::_270,
        (Transform::Flipped270, Transform::Flipped270) => Transform::Normal,
    }
}

#[cfg(test)]
mod tests {
    include!("frame/window_group_tests.rs");
    include!("frame/owner_group_tests.rs");
    use super::{
        combine_image_transform, framebuffer_rect, texture_sampler_for_render, transform_analytic_clip,
        AnalyticClip, TextureSampler, TextureTransform, TransformedAnalyticClip, VulkanFrame, VulkanRenderer,
    };
    use crate::{
        backend::{
            allocator::{
                dmabuf::AsDmabuf,
                vulkan::{ImageUsageFlags, VulkanAllocator},
                Allocator, Fourcc,
            },
            renderer::{
                vulkan::VulkanTexture, Bind, Blit, BlitFrame, BottomEdgeClip, Color32F, ExportMem, Frame,
                Offscreen, RenderTargetAccess, Renderer, Texture, TextureFilter,
            },
            vulkan::{version::Version, Instance, PhysicalDevice},
        },
        utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform},
    };

    fn init_renderer_and_allocator() -> Option<(VulkanRenderer, VulkanAllocator)> {
        let physical_device = crate::backend::renderer::vulkan::test_support::physical_device()?;

        let renderer = crate::backend::renderer::vulkan::test_support::renderer(&physical_device)?;

        let allocator = super::super::test_support::available(
            VulkanAllocator::new(
                &physical_device,
                ImageUsageFlags::SAMPLED | ImageUsageFlags::COLOR_ATTACHMENT,
            ),
            "frame test allocator",
        )?;

        Some((renderer, allocator))
    }

    #[test]
    #[ignore = "requires a hardware Vulkan render node; run explicitly with --ignored"]
    fn exported_vulkan_acquire_failure_does_not_fall_back_to_host_wait() {
        let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
        let physical = PhysicalDevice::enumerate(&instance)
            .unwrap()
            .find(|device| device.render_node().ok().flatten().is_some())
            .expect("hardware render node required");
        let mut renderer = VulkanRenderer::new(&physical).unwrap();
        // This unsignaled native fence has no submission, so a host wait would
        // never finish. Reject the GPU acquire with an empty wait-stage mask;
        // the result must not fall back to that host wait. The descriptor is
        // only a test export and is never submitted to a queue.
        let fence = super::VulkanFence::create(renderer.device.shared_device()).unwrap();
        fence.set_exported_sync_file(std::fs::File::open("/dev/null").unwrap().into());
        let sync = crate::backend::renderer::sync::SyncPoint::from(fence);
        assert!(!sync.is_reached());
        assert!(
            super::wait_on_sync_point(&mut renderer, &sync, ash::vk::PipelineStageFlags::empty(),).is_err()
        );
        assert_eq!(renderer.device.in_flight_submission_count(), 0);
    }

    fn init_renderer_and_effect_allocator() -> Option<(VulkanRenderer, VulkanAllocator)> {
        let instance = super::super::test_support::available(
            Instance::new(Version::VERSION_1_3, None),
            "effect test instance",
        )?;
        // Avio's Vulkan-allocated external targets are the Venus/virtio path.
        // NVIDIA's proprietary driver loses the consumer device when a DMA-BUF
        // exported by a second VkDevice with TRANSFER_SRC usage is first used;
        // physical NVIDIA outputs use GBM allocation instead. Prefer a device
        // that can exercise the Vulkan-export contract this test owns.
        let physical_device = super::super::test_support::present(
            super::super::test_support::available(
                PhysicalDevice::enumerate(&instance),
                "effect test devices",
            )?
            .find(|device| device.properties().vendor_id != 0x10de),
            "effect test requires a Vulkan-export-compatible non-NVIDIA device",
        )?;
        let renderer = crate::backend::renderer::vulkan::test_support::renderer(&physical_device)?;
        let allocator = super::super::test_support::available(
            VulkanAllocator::new(
                &physical_device,
                ImageUsageFlags::COLOR_ATTACHMENT
                    | ImageUsageFlags::TRANSFER_SRC
                    | ImageUsageFlags::TRANSFER_DST,
            ),
            "effect test allocator",
        )?;
        Some((renderer, allocator))
    }

    fn expected_blue_pixel(format: Fourcc) -> [u8; 4] {
        match format {
            Fourcc::Argb8888 | Fourcc::Xrgb8888 => [255, 0, 0, 255],
            Fourcc::Abgr8888 | Fourcc::Xbgr8888 => [0, 0, 255, 255],
            _ => [0, 0, 255, 255],
        }
    }

    fn output_size_for_target(transform: Transform, target_size: Size<i32, Physical>) -> Size<i32, Physical> {
        transform.invert().transform_size(target_size)
    }

    fn src_rect(width: f64, height: f64) -> Rectangle<f64, BufferCoord> {
        Rectangle::new((0.0, 0.0).into(), (width, height).into())
    }

    fn dst_rect(width: i32, height: i32) -> Rectangle<i32, Physical> {
        Rectangle::new((0, 0).into(), (width, height).into())
    }

    #[test]
    fn non_square_output_clear_covers_the_entire_transformed_target() {
        let output_size = Size::<i32, Physical>::from((97, 81));
        for transform in [
            Transform::Normal,
            Transform::_90,
            Transform::_180,
            Transform::_270,
            Transform::Flipped,
            Transform::Flipped90,
            Transform::Flipped180,
            Transform::Flipped270,
        ] {
            let target_size = transform.transform_size(output_size);
            let source_bounds = Rectangle::from_size(output_size);
            let clear = VulkanFrame::transformed_damage_rects(
                transform,
                output_size,
                target_size,
                source_bounds,
                &[Rectangle::from_size(output_size)],
            )
            .collect::<Vec<_>>();
            assert_eq!(clear, vec![Rectangle::from_size(target_size)], "{transform:?}");
        }
    }

    #[test]
    fn non_square_draw_damage_capture_and_clip_share_source_extent() {
        let output_size = Size::<i32, Physical>::from((97, 81));
        let dst = Rectangle::new((19, 7).into(), (41, 65).into());
        // Hand-derived locations in each transformed target; all use the
        // original 97x81 output as the rotation/flip extent.
        for (transform, location, size) in [
            (Transform::Normal, (19, 7), (41, 65)),
            (Transform::_90, (9, 19), (65, 41)),
            (Transform::_180, (37, 9), (41, 65)),
            (Transform::_270, (7, 37), (65, 41)),
            (Transform::Flipped, (37, 7), (41, 65)),
            (Transform::Flipped90, (9, 37), (65, 41)),
            (Transform::Flipped180, (19, 9), (41, 65)),
            (Transform::Flipped270, (7, 19), (65, 41)),
        ] {
            let target_size = transform.transform_size(output_size);
            let expected = Rectangle::new(location.into(), size.into());
            assert_eq!(
                framebuffer_rect(transform, output_size, target_size, dst),
                Some(expected)
            );
            let damage = VulkanFrame::transformed_damage_rects(
                transform,
                output_size,
                target_size,
                dst,
                &[Rectangle::from_size(dst.size)],
            )
            .collect::<Vec<_>>();
            assert_eq!(damage, vec![expected]);
            let clip = BottomEdgeClip {
                rect: dst.to_f64(),
                content_width: 30.0,
                progress: 1.0,
                edge_height: 8.0,
                plateau_inset: 2.0,
                geometry_scale: 1.0,
            };
            let TransformedAnalyticClip::BottomEdge { clip, .. } =
                transform_analytic_clip(transform, output_size, AnalyticClip::BottomEdge(clip))
            else {
                panic!("expected bottom-edge clip");
            };
            assert_eq!(clip.rect, expected.to_f64());
        }
    }

    #[test]
    fn framebuffer_capture_uses_the_pre_transform_output_extent() {
        let output_size = Size::<i32, Physical>::from((300, 200));
        let read = Rectangle::new((20, 30).into(), (180, 120).into());

        for transform in [
            Transform::Normal,
            Transform::_90,
            Transform::_180,
            Transform::_270,
            Transform::Flipped,
            Transform::Flipped90,
            Transform::Flipped180,
            Transform::Flipped270,
        ] {
            let target_size = transform.transform_size(output_size);
            let capture = framebuffer_rect(transform, output_size, target_size, read)
                .expect("read area remains inside the transformed target");
            assert!(Rectangle::from_size(target_size).contains_rect(capture));
            assert_eq!(capture.size, transform.transform_size(read.size));
        }
    }

    #[test]
    fn bottom_edge_clip_keeps_source_orientation_across_output_transforms() {
        let frame_size = Size::<i32, Physical>::from((300, 200));
        let source_rect = Rectangle::new((20.0, 30.0).into(), (100.0, 80.0).into());
        let source = BottomEdgeClip {
            rect: source_rect,
            content_width: 72.0,
            progress: 1.05,
            edge_height: 62.0,
            plateau_inset: 8.0,
            geometry_scale: 1.25,
        };

        for transform in [
            Transform::Normal,
            Transform::_90,
            Transform::_180,
            Transform::_270,
            Transform::Flipped,
            Transform::Flipped90,
            Transform::Flipped180,
            Transform::Flipped270,
        ] {
            let transformed =
                transform_analytic_clip(transform, frame_size, AnalyticClip::BottomEdge(source));
            let TransformedAnalyticClip::BottomEdge {
                clip,
                transform: clip_transform,
            } = transformed
            else {
                panic!("bottom-edge clip kind changed while transforming");
            };
            assert_eq!(
                clip.rect,
                transform.transform_rect_in(source_rect, &frame_size.to_f64())
            );
            assert_eq!(clip_transform, TextureTransform::from(transform));
        }
    }

    #[test]
    fn exact_texture_render_uses_nearest_sampler() {
        assert_eq!(
            texture_sampler_for_render(
                src_rect(64.0, 32.0),
                dst_rect(64, 32),
                Transform::Normal,
                TextureFilter::Linear,
                TextureFilter::Linear,
            ),
            TextureSampler::NEAREST
        );
    }

    #[test]
    fn scaled_texture_render_uses_renderer_filters() {
        assert_eq!(
            texture_sampler_for_render(
                src_rect(64.0, 32.0),
                dst_rect(128, 64),
                Transform::Normal,
                TextureFilter::Linear,
                TextureFilter::Nearest,
            ),
            TextureSampler::new(TextureFilter::Linear, TextureFilter::Nearest)
        );
    }

    #[test]
    fn rotated_exact_texture_render_compares_swapped_axes() {
        assert_eq!(
            texture_sampler_for_render(
                src_rect(64.0, 32.0),
                dst_rect(32, 64),
                Transform::_90,
                TextureFilter::Linear,
                TextureFilter::Linear,
            ),
            TextureSampler::NEAREST
        );
    }

    #[test]
    fn fractional_source_origin_keeps_renderer_filters() {
        let src = Rectangle::new((0.5, 0.0).into(), (64.0, 32.0).into());
        assert_eq!(
            texture_sampler_for_render(
                src,
                dst_rect(64, 32),
                Transform::Normal,
                TextureFilter::Linear,
                TextureFilter::Linear,
            ),
            TextureSampler::LINEAR
        );
    }

    #[test]
    fn frame_drop_without_finish_does_not_submit() {
        let Some((mut renderer, mut allocator)) = init_renderer_and_allocator() else {
            return;
        };

        let Some(format) = renderer
            .dmabuf_render_formats()
            .iter()
            .copied()
            .find(|format| renderer.has_dmabuf_import_format(*format))
        else {
            return;
        };

        let buffer = match allocator.create_buffer(64, 64, format.code, &[format.modifier]) {
            Ok(buffer) => buffer,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        let dmabuf = match buffer.export() {
            Ok(dmabuf) => dmabuf,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };

        let mut target = renderer
            .bind_dmabuf_target(&dmabuf)
            .expect("binding dmabuf target should succeed");
        let target_image = target
            .image_resource()
            .expect("dmabuf target should retain its imported image")
            .clone();
        assert!(target_image.is_owned_by_foreign());

        {
            let mut frame = renderer
                .render(&mut target, Size::from((64, 64)), Transform::Normal)
                .expect("frame creation should succeed");
            assert!(!target_image.is_owned_by_foreign());

            frame
                .draw_solid(
                    Rectangle::new((0, 0).into(), Size::from((32, 32))),
                    &[Rectangle::new((0, 0).into(), Size::from((32, 32)))],
                    crate::backend::renderer::Color32F::new(1.0, 0.0, 0.0, 1.0),
                )
                .expect("recording draw_solid should succeed");

            // drop without finish
        }
        assert!(
            target_image.is_owned_by_foreign(),
            "an unsubmitted acquire must restore foreign ownership"
        );

        assert_eq!(
            renderer.device.in_flight_submission_count(),
            0,
            "dropping a frame without finish must not submit work",
        );

        let frame = renderer
            .render(&mut target, Size::from((64, 64)), Transform::Normal)
            .expect("frame creation should succeed");
        let sync = frame
            .finish()
            .expect("finishing frame should submit work successfully");
        assert!(
            sync.contains_fence(),
            "finished Vulkan frame should return a fence-backed sync point",
        );
        assert_eq!(
            sync.is_exportable(),
            renderer.supports_explicit_sync_export(),
            "sync point exportability should match device explicit-sync export capability",
        );
        renderer
            .wait(&sync)
            .expect("renderer wait should accept Vulkan fence-backed sync points");
        let _ = sync.wait();

        assert_eq!(
            renderer.device.in_flight_submission_count(),
            1,
            "finishing exactly once must produce exactly one submission",
        );

        renderer
            .device
            .wait_for_all_submissions()
            .expect("all submissions should complete cleanly");

        assert_eq!(renderer.device.in_flight_submission_count(), 0);
    }

    #[test]
    fn frame_methods_handle_all_output_transforms_with_damage() {
        let Some((mut renderer, mut allocator)) = init_renderer_and_allocator() else {
            return;
        };

        let Some(format) = renderer
            .dmabuf_render_formats()
            .iter()
            .copied()
            .find(|format| renderer.has_dmabuf_import_format(*format))
        else {
            return;
        };

        let texture_buffer = match allocator.create_buffer(32, 32, format.code, &[format.modifier]) {
            Ok(buffer) => buffer,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        let texture_dmabuf = match texture_buffer.export() {
            Ok(dmabuf) => dmabuf,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        let texture: VulkanTexture = match renderer.import_dmabuf_texture(&texture_dmabuf) {
            Ok(texture) => texture,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };

        let transforms = [
            Transform::Normal,
            Transform::_90,
            Transform::_180,
            Transform::_270,
            Transform::Flipped,
            Transform::Flipped90,
            Transform::Flipped180,
            Transform::Flipped270,
        ];

        for transform in transforms {
            let target_buffer = match allocator.create_buffer(96, 64, format.code, &[format.modifier]) {
                Ok(buffer) => buffer,
                Err(error) => {
                    super::super::test_support::unavailable(error);
                    return;
                }
            };
            let target_dmabuf = match target_buffer.export() {
                Ok(dmabuf) => dmabuf,
                Err(error) => {
                    super::super::test_support::unavailable(error);
                    return;
                }
            };
            let mut target = match renderer.bind_dmabuf_target(&target_dmabuf) {
                Ok(target) => target,
                Err(error) => {
                    super::super::test_support::unavailable(error);
                    return;
                }
            };

            let target_size = Size::from((target.width() as i32, target.height() as i32));
            let output_size = output_size_for_target(transform, target_size);

            let mut frame = match renderer.render(&mut target, output_size, transform) {
                Ok(frame) => frame,
                Err(error) => {
                    super::super::test_support::unavailable(error);
                    return;
                }
            };

            frame
                .clear(
                    crate::backend::renderer::Color32F::new(0.0, 0.0, 1.0, 1.0),
                    &[Rectangle::new((0, 0).into(), output_size)],
                )
                .expect("clear should record for all transforms");

            frame
                .draw_solid(
                    Rectangle::new((8, 6).into(), Size::from((20, 18))),
                    &[
                        Rectangle::new((2, 2).into(), Size::from((8, 7))),
                        Rectangle::new((11, 4).into(), Size::from((6, 6))),
                    ],
                    crate::backend::renderer::Color32F::new(0.0, 1.0, 0.0, 1.0),
                )
                .expect("draw_solid should record with transformed damage");

            frame
                .render_texture_from_to(
                    &texture,
                    Rectangle::new((0.0, 0.0).into(), texture.size().to_f64()),
                    Rectangle::new((24, 10).into(), Size::from((30, 24))),
                    &[Rectangle::new((4, 3).into(), Size::from((14, 10)))],
                    &[Rectangle::new((4, 3).into(), Size::from((14, 10)))],
                    Transform::Flipped90,
                    1.0,
                )
                .expect("render_texture_from_to should record with transform and damage");

            let sync = frame
                .finish()
                .expect("finish should submit once per completed frame");
            let _ = sync.wait();
        }

        renderer
            .device
            .wait_for_all_submissions()
            .expect("all transform submissions should complete");
    }

    #[test]
    fn renderer_diagnostics_track_cache_and_submission_metrics() {
        let Some((mut renderer, mut allocator)) = init_renderer_and_allocator() else {
            return;
        };

        let Some(format) = renderer
            .dmabuf_render_formats()
            .iter()
            .copied()
            .find(|format| renderer.has_dmabuf_import_format(*format))
        else {
            return;
        };

        let texture_buffer = match allocator.create_buffer(32, 32, format.code, &[format.modifier]) {
            Ok(buffer) => buffer,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        let texture_dmabuf = match texture_buffer.export() {
            Ok(dmabuf) => dmabuf,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        let texture: VulkanTexture = match renderer.import_dmabuf_texture(&texture_dmabuf) {
            Ok(texture) => texture,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        let texture_image = texture
            .image_resource()
            .expect("dmabuf texture should retain its imported image")
            .clone();

        let target_buffer = match allocator.create_buffer(96, 64, format.code, &[format.modifier]) {
            Ok(buffer) => buffer,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        let target_dmabuf = match target_buffer.export() {
            Ok(dmabuf) => dmabuf,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        let mut target = match renderer.bind_dmabuf_target(&target_dmabuf) {
            Ok(target) => target,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };

        for _ in 0..2 {
            let mut frame = match renderer.render(&mut target, Size::from((96, 64)), Transform::Normal) {
                Ok(frame) => frame,
                Err(error) => {
                    super::super::test_support::unavailable(error);
                    return;
                }
            };

            frame
                .render_texture_from_to(
                    &texture,
                    Rectangle::new((0.0, 0.0).into(), texture.size().to_f64()),
                    Rectangle::new((12, 10).into(), Size::from((36, 24))),
                    &[Rectangle::new((0, 0).into(), Size::from((96, 64)))],
                    &[],
                    Transform::Normal,
                    1.0,
                )
                .expect("texture rendering should succeed");
            assert!(!texture_image.is_owned_by_foreign());
            let sync = frame.finish().expect("finish should submit the frame");
            assert!(
                texture_image.is_owned_by_foreign(),
                "a sampled DMA-BUF must be released to FOREIGN after final access"
            );
            let _ = sync.wait();
        }

        renderer
            .device
            .wait_for_all_submissions()
            .expect("submissions should complete");

        let diagnostics = renderer.diagnostics();
        assert!(
            diagnostics.dmabuf_cache.misses >= 2,
            "importing texture and binding target should populate dmabuf cache misses"
        );
        assert!(
            diagnostics.descriptor_cache.misses >= 1,
            "first textured draw should allocate a descriptor"
        );
        assert!(
            diagnostics.descriptor_cache.hits >= 1,
            "second textured draw should reuse descriptor cache entry"
        );
        assert!(
            diagnostics.submissions.total_submissions >= 2,
            "two finished frames should record at least two submissions"
        );
        assert!(
            diagnostics.submissions.reclaimed_submissions >= 1,
            "completed submissions should be reclaimed into diagnostics"
        );
    }

    #[test]
    fn combine_image_transform_matches_reference_cases() {
        assert_eq!(
            combine_image_transform(Transform::Normal, Transform::_90),
            Transform::_90
        );
        assert_eq!(
            combine_image_transform(Transform::_90, Transform::Normal),
            Transform::_90
        );
        assert_eq!(
            combine_image_transform(Transform::Flipped, Transform::Flipped),
            Transform::Normal
        );
        assert_eq!(
            combine_image_transform(Transform::Flipped90, Transform::_270),
            Transform::Flipped180
        );
    }

    #[test]
    fn frame_blit_to_and_from_resume_recording() {
        let Some((mut renderer, _)) = init_renderer_and_allocator() else {
            return;
        };

        let Some(format) = [
            Fourcc::Argb8888,
            Fourcc::Abgr8888,
            Fourcc::Xrgb8888,
            Fourcc::Xbgr8888,
        ]
        .into_iter()
        .find(|format| renderer.create_buffer(*format, Size::from((4, 4))).is_ok()) else {
            super::super::test_support::unavailable("frame offscreen format");
            return;
        };

        let size = Size::from((32, 32));
        let physical_size = Size::<i32, Physical>::from((32, 32));
        let full_damage = Rectangle::from_size(physical_size);

        let mut frame_tex = match renderer.create_buffer(format, size) {
            Ok(buffer) => buffer,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        let mut aux_tex = match renderer.create_buffer(format, size) {
            Ok(buffer) => buffer,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };

        let mut frame_target = match renderer.bind(&mut frame_tex) {
            Ok(target) => target,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        let mut aux_target = match renderer.bind(&mut aux_tex) {
            Ok(target) => target,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };

        let mut frame = match renderer.render(&mut frame_target, physical_size, Transform::Normal) {
            Ok(frame) => frame,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };

        frame
            .clear(Color32F::new(0.1, 0.2, 0.3, 1.0), &[full_damage])
            .expect("clear should succeed before frame blit operations");

        frame
            .blit_to(&mut aux_target, full_damage, full_damage, TextureFilter::Nearest)
            .expect("blit_to should succeed and keep frame usable");

        frame
            .blit_from(&aux_target, full_damage, full_damage, TextureFilter::Nearest)
            .expect("blit_from should succeed and keep frame usable");

        frame
            .draw_solid(
                Rectangle::new((8, 8).into(), Size::from((8, 8))),
                &[Rectangle::new((0, 0).into(), Size::from((8, 8)))],
                Color32F::new(1.0, 0.0, 0.0, 1.0),
            )
            .expect("frame should continue recording after blit operations");

        let sync = frame
            .finish()
            .expect("finish should submit frame after frame blits");
        let _ = sync.wait();
    }

    #[test]
    fn framebuffer_effect_capture_preserves_target_storage_format() {
        let Some((mut renderer, _)) = init_renderer_and_allocator() else {
            return;
        };
        let size = Size::from((16, 16));
        let physical_size = Size::<i32, Physical>::from((16, 16));
        let region = Rectangle::from_size(physical_size);
        let buffer_region = Rectangle::<i32, BufferCoord>::from_size(size);

        let Some(mut frame_texture) = super::super::test_support::available(
            renderer.create_buffer(Fourcc::Argb8888, size),
            "test allocation or binding",
        ) else {
            return;
        };
        let Some(capture) = super::super::test_support::available(
            renderer.create_buffer(Fourcc::Argb8888, size),
            "test allocation or binding",
        ) else {
            return;
        };
        let Some(mut target) = super::super::test_support::available(
            renderer.bind(&mut frame_texture),
            "test allocation or binding",
        ) else {
            return;
        };

        let mut frame = renderer
            .render(&mut target, physical_size, Transform::Normal)
            .expect("frame");
        frame
            .clear(Color32F::new(0.0, 0.0, 1.0, 1.0), &[region])
            .expect("clear blue accumulator");
        frame
            .capture_and_filter_framebuffer(region, &capture, &[])
            .expect("same-format framebuffer capture");
        frame
            .finish()
            .expect("finish capture")
            .wait()
            .expect("same-format capture submission completes");

        let mapping = renderer
            .copy_texture(&capture, buffer_region, Fourcc::Argb8888)
            .expect("read capture");
        let bytes = renderer.map_texture(&mapping).expect("map capture");
        assert_eq!(&bytes[0..4], &expected_blue_pixel(Fourcc::Argb8888));
    }

    #[test]
    fn framebuffer_effect_capture_rejects_a_different_channel_layout() {
        let Some((mut renderer, _)) = init_renderer_and_allocator() else {
            return;
        };
        let size = Size::from((16, 16));
        let physical_size = Size::<i32, Physical>::from((16, 16));
        let region = Rectangle::from_size(physical_size);
        let Some(mut frame_texture) = super::super::test_support::available(
            renderer.create_buffer(Fourcc::Argb8888, size),
            "test allocation or binding",
        ) else {
            return;
        };
        let Some(capture) = super::super::test_support::available(
            renderer.create_buffer(Fourcc::Abgr8888, size),
            "test allocation or binding",
        ) else {
            return;
        };
        let Some(mut target) = super::super::test_support::available(
            renderer.bind(&mut frame_texture),
            "test allocation or binding",
        ) else {
            return;
        };

        let mut frame = renderer
            .render(&mut target, physical_size, Transform::Normal)
            .expect("frame");
        frame
            .clear(Color32F::new(0.0, 0.0, 1.0, 1.0), &[region])
            .expect("clear blue accumulator");
        let error = frame
            .capture_and_filter_framebuffer(region, &capture, &[])
            .expect_err("material capture must not convert channel layouts");
        assert!(error
            .to_string()
            .contains("capture format must match the active target format"));
    }

    #[test]
    fn external_dmabuf_effect_target_captures_logical_blue_without_conversion() {
        let Some((mut renderer, mut allocator)) = init_renderer_and_effect_allocator() else {
            return;
        };
        let Some(format) = renderer
            .dmabuf_framebuffer_effect_formats()
            .iter()
            .copied()
            .find(|format| {
                format.code == Fourcc::Argb8888
                    && format.modifier == crate::backend::allocator::Modifier::Linear
            })
            .or_else(|| {
                renderer
                    .dmabuf_framebuffer_effect_formats()
                    .iter()
                    .copied()
                    .find(|format| format.code == Fourcc::Argb8888)
            })
            .or_else(|| {
                renderer
                    .dmabuf_framebuffer_effect_formats()
                    .iter()
                    .copied()
                    .next()
            })
        else {
            super::super::test_support::unavailable("framebuffer-effect DMA-BUF format");
            return;
        };
        let size = Size::from((16, 16));
        let physical_size = Size::<i32, Physical>::from((16, 16));
        let region = Rectangle::from_size(physical_size);
        let buffer_region = Rectangle::<i32, BufferCoord>::from_size(size);
        let Some(buffer) = super::super::test_support::available(
            allocator.create_buffer(16, 16, format.code, &[format.modifier]),
            "test allocation or binding",
        ) else {
            return;
        };
        let Some(mut dmabuf) =
            super::super::test_support::available(buffer.export(), "test allocation or binding")
        else {
            return;
        };
        let Some(capture) = super::super::test_support::available(
            renderer.create_buffer(format.code, size),
            "test allocation or binding",
        ) else {
            return;
        };
        let mut target = renderer
            .bind_with_access(&mut dmabuf, RenderTargetAccess::FramebufferEffectSource)
            .expect("effect-capable external target bind");
        assert!(target
            .image_resource()
            .expect("image-backed target")
            .usage()
            .contains(ash::vk::ImageUsageFlags::TRANSFER_SRC));

        let mut frame = renderer
            .render(&mut target, physical_size, Transform::Normal)
            .expect("external target frame");
        frame
            .clear(Color32F::new(0.0, 0.0, 1.0, 1.0), &[region])
            .expect("clear blue accumulator");
        frame
            .capture_and_filter_framebuffer(region, &capture, &[])
            .expect("external same-format framebuffer capture");
        frame
            .finish()
            .expect("finish capture")
            .wait()
            .expect("external capture submission completes");

        let mapping = renderer
            .copy_texture(&capture, buffer_region, format.code)
            .expect("read capture");
        let bytes = renderer.map_texture(&mapping).expect("map capture");
        assert_eq!(&bytes[0..4], &expected_blue_pixel(format.code));
    }

    #[test]
    fn external_dmabuf_capture_target_accepts_a_terminal_blit() {
        let Some((mut renderer, mut allocator)) = init_renderer_and_effect_allocator() else {
            return;
        };
        let Some(format) = renderer
            .dmabuf_capture_formats()
            .iter()
            .copied()
            .find(|format| format.code == Fourcc::Argb8888)
        else {
            super::super::test_support::unavailable("framebuffer-effect DMA-BUF format");
            return;
        };
        let size = Size::from((16, 16));
        let rect = Rectangle::<i32, Physical>::from_size((16, 16).into());
        let Some(buffer) = super::super::test_support::available(
            allocator.create_buffer(16, 16, format.code, &[format.modifier]),
            "test allocation or binding",
        ) else {
            return;
        };
        let Some(mut dmabuf) =
            super::super::test_support::available(buffer.export(), "test allocation or binding")
        else {
            return;
        };
        let Some(mut source_texture) = super::super::test_support::available(
            renderer.create_buffer(format.code, size),
            "test allocation or binding",
        ) else {
            return;
        };
        let mut source = renderer.bind(&mut source_texture).expect("source target");
        let mut source_frame = renderer
            .render(&mut source, (16, 16).into(), Transform::Normal)
            .expect("source frame");
        source_frame
            .clear(Color32F::new(0.0, 0.0, 1.0, 1.0), &[rect])
            .expect("clear source blue");
        source_frame
            .finish()
            .expect("finish source")
            .wait()
            .expect("source submission completes");

        let mut destination = renderer
            .bind_with_access(&mut dmabuf, RenderTargetAccess::CaptureTarget)
            .expect("capture target bind");
        let usage = destination
            .image_resource()
            .expect("image-backed capture target")
            .usage();
        assert!(usage.contains(ash::vk::ImageUsageFlags::TRANSFER_SRC));
        assert!(usage.contains(ash::vk::ImageUsageFlags::TRANSFER_DST));
        Blit::blit(
            &mut renderer,
            &source,
            &mut destination,
            rect,
            rect,
            TextureFilter::Linear,
        )
        .expect("terminal capture blit")
        .wait()
        .expect("terminal capture blit completes");
    }

    #[test]
    fn flushed_segment_releases_sampled_dmabuf_before_blit() {
        let Some((mut renderer, mut allocator)) = init_renderer_and_allocator() else {
            return;
        };
        let Some(format) = renderer
            .dmabuf_render_formats()
            .iter()
            .copied()
            .find(|format| renderer.has_dmabuf_import_format(*format))
        else {
            return;
        };

        let texture_buffer = match allocator.create_buffer(32, 32, format.code, &[format.modifier]) {
            Ok(buffer) => buffer,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        let texture_dmabuf = match texture_buffer.export() {
            Ok(dmabuf) => dmabuf,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        let texture = match renderer.import_dmabuf_texture(&texture_dmabuf) {
            Ok(texture) => texture,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        let texture_image = texture
            .image_resource()
            .expect("dmabuf texture should retain its imported image")
            .clone();

        let size = Size::from((32, 32));
        let physical_size = Size::<i32, Physical>::from((32, 32));
        let full_damage = Rectangle::from_size(physical_size);
        let mut frame_tex = match renderer.create_buffer(format.code, size) {
            Ok(buffer) => buffer,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        let mut aux_tex = match renderer.create_buffer(format.code, size) {
            Ok(buffer) => buffer,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        let mut frame_target = match renderer.bind(&mut frame_tex) {
            Ok(target) => target,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        let mut aux_target = match renderer.bind(&mut aux_tex) {
            Ok(target) => target,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };

        let mut frame = match renderer.render(&mut frame_target, physical_size, Transform::Normal) {
            Ok(frame) => frame,
            Err(error) => {
                super::super::test_support::unavailable(error);
                return;
            }
        };
        frame
            .render_texture_from_to(
                &texture,
                Rectangle::new((0.0, 0.0).into(), texture.size().to_f64()),
                full_damage,
                &[full_damage],
                &[],
                Transform::Normal,
                1.0,
            )
            .expect("sampled DMA-BUF should record before the segment flush");
        assert!(!texture_image.is_owned_by_foreign());

        frame
            .blit_to(&mut aux_target, full_damage, full_damage, TextureFilter::Nearest)
            .expect("segment blit should flush and resume the frame");
        assert!(
            texture_image.is_owned_by_foreign(),
            "a completed segment must not strand sampled DMA-BUF ownership"
        );

        let sync = frame.finish().expect("resumed frame should remain finishable");
        let _ = sync.wait();
    }
}
