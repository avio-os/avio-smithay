use std::sync::Arc;

use ash::vk;
use indexmap::IndexMap;
use tracing::{instrument, trace, warn};

use crate::{
    backend::{
        allocator::{
            dmabuf::Dmabuf,
            format::{has_alpha, FormatSet},
        },
        renderer::{
            sync::SyncPoint, Bind, Color32F, ContextId, Frame, ImportDma, Renderer, RendererSuper, Texture,
            TextureFilter,
        },
    },
    utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform},
};

use super::{
    dmabuf::ImportedDmabufImage,
    pipeline::{
        push_constants_bytes, PipelineHandles, SolidPushConstants, TexturePushConstants, TextureTransform,
    },
    sync::VulkanFence,
    VulkanRenderer, VulkanRendererError, VulkanRendererErrorKind, VulkanTarget, VulkanTexture,
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
    target: Arc<ImportedDmabufImage>,
    pipelines: PipelineHandles,
    transform: Transform,
    output_size: Size<i32, Physical>,
    size: Size<i32, Physical>,
    pending_layouts: IndexMap<u64, (Arc<ImportedDmabufImage>, vk::ImageLayout)>,
}

/// In-flight Vulkan renderer frame recording context.
#[derive(Debug)]
pub struct VulkanFrame<'frame> {
    renderer: &'frame mut VulkanRenderer,
    state: VulkanFrameState,
    recording: Option<FrameRecording>,
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

        let Some(target_image) = target.imported_image().cloned() else {
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

        let pipelines = self.pipelines.pipelines_for_format(target_image.vk_format())?;

        let command_buffer = self.device.acquire_command_buffer()?;
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: Command buffer belongs to this device command pool and is not currently in use.
        unsafe {
            self.device
                .device_handle()
                .begin_command_buffer(command_buffer, &begin_info)
        }?;
        self.device
            .insert_debug_label(command_buffer, c"vulkan.render.begin", [0.17, 0.42, 0.86, 1.0]);

        let framebuffer = create_framebuffer(
            self.device.device_handle(),
            pipelines.render_pass,
            target_image.view(),
            transformed_size,
        )?;

        let mut frame = VulkanFrame {
            renderer: self,
            state: VulkanFrameState::Recording,
            recording: Some(FrameRecording {
                command_buffer,
                framebuffer,
                target: target_image,
                pipelines,
                transform: dst_transform,
                output_size,
                size: transformed_size,
                pending_layouts: IndexMap::new(),
            }),
        };

        {
            let target = frame
                .recording
                .as_ref()
                .expect("recording initialized")
                .target
                .clone();
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
        if let Some(vulkan_fence) = sync.get::<VulkanFence>() {
            return vulkan_fence
                .wait_vk()
                .map_err(|_| VulkanRendererError::TemporaryFailure("sync wait was interrupted"));
        }

        if self.device.supports_sync_file_import() {
            if let Some(sync_file) = sync.export() {
                match self.device.wait_on_sync_file(sync_file) {
                    Ok(()) => return Ok(()),
                    Err(err) => {
                        if err.kind() == VulkanRendererErrorKind::ContextLost {
                            return Err(err);
                        }

                        warn!(
                            ?err,
                            "failed to import SyncPoint fd into Vulkan wait path; falling back to blocking wait"
                        );
                    }
                }
            }
        }

        sync.wait()
            .map_err(|_| VulkanRendererError::TemporaryFailure("sync wait was interrupted"))
    }

    #[instrument(level = "trace", skip(self))]
    #[profiling::function]
    fn cleanup_texture_cache(&mut self) -> Result<(), Self::Error> {
        self.dmabuf.cleanup();
        self.descriptors.clear_texture_cache()?;
        self.device.reclaim_completed_submissions()?;
        Ok(())
    }
}

impl Bind<Dmabuf> for VulkanRenderer {
    fn bind<'a>(&mut self, target: &'a mut Dmabuf) -> Result<Self::Framebuffer<'a>, Self::Error> {
        self.bind_dmabuf_target(target)
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

impl Frame for VulkanFrame<'_> {
    type Error = VulkanRendererError;
    type TextureId = VulkanTexture;

    fn context_id(&self) -> ContextId<Self::TextureId> {
        self.renderer.context_id.clone()
    }

    #[instrument(level = "trace", skip(self, at))]
    #[profiling::function]
    fn clear(&mut self, color: Color32F, at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        let (command_buffer, transform, size) = {
            let recording = self.recording()?;
            (recording.command_buffer, recording.transform, recording.size)
        };

        let clear_regions = Self::transformed_damage_rects(transform, size, Rectangle::from_size(size), at);
        if clear_regions.is_empty() {
            return Ok(());
        }

        let clear_attachments = [vk::ClearAttachment::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .color_attachment(0)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [color.r(), color.g(), color.b(), color.a()],
                },
            })];

        let clear_rects = clear_regions
            .iter()
            .map(|rect| {
                vk::ClearRect::default()
                    .rect(to_vk_rect(*rect))
                    .base_array_layer(0)
                    .layer_count(1)
            })
            .collect::<Vec<_>>();

        // SAFETY: Command buffer recording is active and clear regions are inside the current render area.
        unsafe {
            self.renderer.device.insert_debug_label(
                command_buffer,
                c"vulkan.frame.clear",
                [0.26, 0.73, 0.31, 1.0],
            );
            self.renderer.device.device_handle().cmd_clear_attachments(
                command_buffer,
                &clear_attachments,
                &clear_rects,
            );
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
        if damage.is_empty() {
            return Ok(());
        }

        let (command_buffer, pipelines, transform, size) = {
            let recording = self.recording()?;
            (
                recording.command_buffer,
                recording.pipelines,
                recording.transform,
                recording.size,
            )
        };

        let frame_bounds = Rectangle::from_size(size);
        let Some(viewport_rect) = transform.transform_rect_in(dst, &size).intersection(frame_bounds) else {
            return Ok(());
        };

        let draw_damage = Self::transformed_damage_rects(transform, size, dst, damage);
        if draw_damage.is_empty() {
            return Ok(());
        }

        let pipeline = if color.is_opaque() {
            pipelines.solid_opaque_pipeline
        } else {
            pipelines.solid_pipeline
        };

        let constants = SolidPushConstants {
            color: [color.r(), color.g(), color.b(), color.a()],
        };

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
                &[to_vk_viewport(viewport_rect)],
            );

            self.renderer.device.device_handle().cmd_push_constants(
                command_buffer,
                pipelines.solid_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                push_constants_bytes(&constants),
            );

            for rect in &draw_damage {
                self.renderer
                    .device
                    .device_handle()
                    .cmd_set_scissor(command_buffer, 0, &[to_vk_rect(*rect)]);
                self.renderer
                    .device
                    .device_handle()
                    .cmd_draw(command_buffer, 4, 1, 0, 0);
            }
        }

        Ok(())
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
        if damage.is_empty() {
            return Ok(());
        }

        let Some(texture_image) = texture.imported_image().cloned() else {
            return Err(VulkanRendererError::NotImplemented(
                "render_texture_from_to currently requires dma-buf-backed VulkanTexture",
            ));
        };

        self.transition_image_layout(&texture_image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;

        let (command_buffer, pipelines, transform, size) = {
            let recording = self.recording()?;
            (
                recording.command_buffer,
                recording.pipelines,
                recording.transform,
                recording.size,
            )
        };

        let frame_bounds = Rectangle::from_size(size);
        let Some(viewport_rect) = transform.transform_rect_in(dst, &size).intersection(frame_bounds) else {
            return Ok(());
        };

        let draw_damage = Self::transformed_damage_rects(transform, size, dst, damage);
        if draw_damage.is_empty() {
            return Ok(());
        }

        let texture_size = texture.size();
        if texture_size.w <= 0 || texture_size.h <= 0 {
            return Err(VulkanRendererError::TemporaryFailure(
                "texture dimensions must be positive",
            ));
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

        let texture_transform = combine_image_transform(src_transform, transform);
        let src_offset = [
            src.loc.x as f32 / texture_size.w as f32,
            src.loc.y as f32 / texture_size.h as f32,
        ];
        let src_scale = [
            src.size.w as f32 / texture_size.w as f32,
            src.size.h as f32 / texture_size.h as f32,
        ];

        let push_constants = TexturePushConstants::new(
            alpha,
            TextureTransform::from(texture_transform),
            texture.y_inverted(),
        )
        .with_src_rect(src_offset, src_scale);

        let descriptor_set = self
            .renderer
            .descriptors
            .texture_descriptor_set(texture_image.view())?;

        let texture_has_alpha = texture.format().map(has_alpha).unwrap_or(true);
        let use_opaque_only = alpha >= 1.0 && !texture_has_alpha;

        let transformed_opaque = if alpha >= 1.0 {
            Self::transformed_damage_rects(transform, size, dst, opaque_regions)
        } else {
            Vec::new()
        };

        let (opaque_draws, blended_draws) = if use_opaque_only {
            (draw_damage, Vec::new())
        } else if !transformed_opaque.is_empty() {
            let mut opaque = Vec::new();
            let mut blended = Vec::new();

            for rect in draw_damage {
                if transformed_opaque
                    .iter()
                    .any(|opaque_rect| opaque_rect.contains_rect(rect))
                {
                    opaque.push(rect);
                } else {
                    blended.push(rect);
                }
            }

            (opaque, blended)
        } else {
            (Vec::new(), draw_damage)
        };

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
                &[to_vk_viewport(viewport_rect)],
            );

            if !opaque_draws.is_empty() {
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

                for rect in &opaque_draws {
                    self.renderer.device.device_handle().cmd_set_scissor(
                        command_buffer,
                        0,
                        &[to_vk_rect(*rect)],
                    );
                    self.renderer
                        .device
                        .device_handle()
                        .cmd_draw(command_buffer, 4, 1, 0, 0);
                }
            }

            if !blended_draws.is_empty() {
                self.renderer.device.device_handle().cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipelines.textured_pipeline,
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

                for rect in &blended_draws {
                    self.renderer.device.device_handle().cmd_set_scissor(
                        command_buffer,
                        0,
                        &[to_vk_rect(*rect)],
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

    fn transformation(&self) -> Transform {
        self.recording
            .as_ref()
            .map(|recording| recording.transform)
            .unwrap_or(Transform::Normal)
    }

    #[instrument(level = "trace", skip(self, sync))]
    #[profiling::function]
    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        self.renderer.wait(sync)
    }

    #[instrument(level = "trace", skip(self))]
    #[profiling::function]
    fn finish(mut self) -> Result<SyncPoint, Self::Error> {
        self.finish_internal()
    }
}

impl VulkanFrame<'_> {
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
        image: &Arc<ImportedDmabufImage>,
        new_layout: vk::ImageLayout,
    ) -> Result<(), VulkanRendererError> {
        let (command_buffer, old_layout) = {
            let recording = self.recording_mut()?;
            let old_layout = recording
                .pending_layouts
                .get(&image.id())
                .map(|(_, layout)| *layout)
                .unwrap_or_else(|| image.current_layout());
            (recording.command_buffer, old_layout)
        };

        if old_layout == new_layout {
            return Ok(());
        }

        let (src_stage_mask, src_access_mask) = stage_access_for_layout(old_layout);
        let (dst_stage_mask, dst_access_mask) = stage_access_for_layout(new_layout);

        let barrier = [vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
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

    fn transformed_damage_rects(
        transform: Transform,
        size: Size<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
    ) -> Vec<Rectangle<i32, Physical>> {
        let dst_bounds = dst;
        let frame_bounds = Rectangle::from_size(size);

        damage
            .iter()
            .filter_map(|rect| {
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
                    .map(|clipped| transform.transform_rect_in(clipped, &size))
                    .and_then(|transformed| transformed.intersection(frame_bounds))
                    .filter(|region| region.size.w > 0 && region.size.h > 0)
            })
            .collect()
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

        // SAFETY: Command buffer recording is valid and render pass has been ended.
        if let Err(err) = unsafe {
            self.renderer
                .device
                .device_handle()
                .end_command_buffer(recording.command_buffer)
        } {
            // SAFETY: Framebuffer was created for this device and command buffer will not be submitted.
            unsafe {
                self.renderer
                    .device
                    .device_handle()
                    .destroy_framebuffer(recording.framebuffer, None)
            };
            let _ = self
                .renderer
                .device
                .discard_command_buffer(recording.command_buffer);
            self.state = VulkanFrameState::Aborted;
            return Err(err.into());
        }

        let (_, submission_fence) = match self
            .renderer
            .device
            .submit_with_framebuffers_and_fence(recording.command_buffer, vec![recording.framebuffer])
        {
            Ok(submission) => submission,
            Err(err) => {
                let _ = self
                    .renderer
                    .device
                    .discard_command_buffer(recording.command_buffer);
                self.state = VulkanFrameState::Aborted;
                return Err(err);
            }
        };

        for (_, (image, layout)) in recording.pending_layouts.drain(..) {
            image.set_layout(layout);
        }

        self.state = VulkanFrameState::Finished;
        Ok(SyncPoint::from(submission_fence))
    }
}

impl Drop for VulkanFrame<'_> {
    fn drop(&mut self) {
        if self.state != VulkanFrameState::Recording {
            return;
        }

        if let Some(recording) = self.recording.take() {
            // SAFETY: Framebuffer was created by this device and command buffer is not submitted on drop.
            unsafe {
                self.renderer
                    .device
                    .device_handle()
                    .destroy_framebuffer(recording.framebuffer, None)
            };

            if let Err(err) = self
                .renderer
                .device
                .discard_command_buffer(recording.command_buffer)
            {
                warn!(?err, "failed to discard Vulkan frame command buffer on drop");
            }
        }

        self.state = VulkanFrameState::Aborted;
    }
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

fn stage_access_for_layout(layout: vk::ImageLayout) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    match layout {
        vk::ImageLayout::UNDEFINED => (vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty()),
        vk::ImageLayout::GENERAL => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_READ,
        ),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => {
            (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ)
        }
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => {
            (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE)
        }
        _ => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ),
    }
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
    use super::{combine_image_transform, VulkanRenderer, VulkanRendererError};
    use crate::{
        backend::{
            allocator::{
                dmabuf::AsDmabuf,
                vulkan::{ImageUsageFlags, VulkanAllocator},
                Allocator,
            },
            renderer::{vulkan::VulkanTexture, Frame, Renderer, Texture},
            vulkan::{version::Version, Instance, PhysicalDevice},
        },
        utils::{Physical, Rectangle, Size, Transform},
    };

    fn init_renderer_and_allocator() -> Option<(VulkanRenderer, VulkanAllocator)> {
        let instance = Instance::new(Version::VERSION_1_3, None).ok()?;
        let physical_device = PhysicalDevice::enumerate(&instance).ok()?.next()?;

        let renderer = match VulkanRenderer::new(&physical_device) {
            Ok(renderer) => renderer,
            Err(
                VulkanRendererError::MissingDeviceExtensions(_)
                | VulkanRendererError::MissingDeviceFeature(_)
                | VulkanRendererError::MissingQueueFamily { .. },
            ) => {
                return None;
            }
            Err(_) => return None,
        };

        let allocator = VulkanAllocator::new(
            &physical_device,
            ImageUsageFlags::SAMPLED | ImageUsageFlags::COLOR_ATTACHMENT,
        )
        .ok()?;

        Some((renderer, allocator))
    }

    fn output_size_for_target(transform: Transform, target_size: Size<i32, Physical>) -> Size<i32, Physical> {
        transform.invert().transform_size(target_size)
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
            Err(_) => return,
        };
        let dmabuf = match buffer.export() {
            Ok(dmabuf) => dmabuf,
            Err(_) => return,
        };

        let mut target = renderer
            .bind_dmabuf_target(&dmabuf)
            .expect("binding dmabuf target should succeed");

        {
            let mut frame = renderer
                .render(&mut target, Size::from((64, 64)), Transform::Normal)
                .expect("frame creation should succeed");

            frame
                .draw_solid(
                    Rectangle::new((0, 0).into(), Size::from((32, 32))),
                    &[Rectangle::new((0, 0).into(), Size::from((32, 32)))],
                    crate::backend::renderer::Color32F::new(1.0, 0.0, 0.0, 1.0),
                )
                .expect("recording draw_solid should succeed");

            // drop without finish
        }

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
            Err(_) => return,
        };
        let texture_dmabuf = match texture_buffer.export() {
            Ok(dmabuf) => dmabuf,
            Err(_) => return,
        };
        let texture: VulkanTexture = match renderer.import_dmabuf_texture(&texture_dmabuf) {
            Ok(texture) => texture,
            Err(_) => return,
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
                Err(_) => return,
            };
            let target_dmabuf = match target_buffer.export() {
                Ok(dmabuf) => dmabuf,
                Err(_) => return,
            };
            let mut target = match renderer.bind_dmabuf_target(&target_dmabuf) {
                Ok(target) => target,
                Err(_) => return,
            };

            let target_size = Size::from((target.width() as i32, target.height() as i32));
            let output_size = output_size_for_target(transform, target_size);

            let mut frame = match renderer.render(&mut target, output_size, transform) {
                Ok(frame) => frame,
                Err(_) => return,
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
            Err(_) => return,
        };
        let texture_dmabuf = match texture_buffer.export() {
            Ok(dmabuf) => dmabuf,
            Err(_) => return,
        };
        let texture: VulkanTexture = match renderer.import_dmabuf_texture(&texture_dmabuf) {
            Ok(texture) => texture,
            Err(_) => return,
        };

        let target_buffer = match allocator.create_buffer(96, 64, format.code, &[format.modifier]) {
            Ok(buffer) => buffer,
            Err(_) => return,
        };
        let target_dmabuf = match target_buffer.export() {
            Ok(dmabuf) => dmabuf,
            Err(_) => return,
        };
        let mut target = match renderer.bind_dmabuf_target(&target_dmabuf) {
            Ok(target) => target,
            Err(_) => return,
        };

        for _ in 0..2 {
            let mut frame = match renderer.render(&mut target, Size::from((96, 64)), Transform::Normal) {
                Ok(frame) => frame,
                Err(_) => return,
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
            let sync = frame.finish().expect("finish should submit the frame");
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
}
