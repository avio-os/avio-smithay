// Ordered prefix capture and fused material lowering; one frame owns all resources.

#[derive(Clone, Copy)]
enum MaterialTextureOperation {
    Kawase(super::kawase::VulkanKawaseOutput),
    Tint(super::VulkanMaterialTint),
    OwnerResolve(Size<i32, Physical>),
    FramebufferCopy,
}

impl MaterialTextureOperation {
    fn framebuffer_coordinates(self) -> bool {
        matches!(
            self,
            Self::Kawase(_) | Self::OwnerResolve(_) | Self::FramebufferCopy
        )
    }
}

impl VulkanFrame<'_> {
    /// Capture a bounded rectangle from the current output prefix and execute
    /// a Kawase filter graph in this frame's active command buffer.
    ///
    /// The main render pass is paused and resumed with `LOAD`; no queue submit,
    /// host wait, or second renderer authority is introduced. Every image and
    /// transient framebuffer is retained by the final frame submission.
    pub fn capture_and_filter_framebuffer(
        &mut self,
        backdrop_read_area: Rectangle<i32, Physical>,
        capture: &VulkanTexture,
        passes: &[VulkanKawasePass],
    ) -> Result<(), VulkanRendererError> {
        self.capture_material_prefix(backdrop_read_area, capture, passes, None)
    }

    /// Freeze the completed lower scene directly into the first downsample,
    /// before any member of this material group is drawn. No full-size copy exists.
    pub fn capture_and_downsample_framebuffer(
        &mut self,
        backdrop_read_area: Rectangle<i32, Physical>,
        first_level: &VulkanTexture,
        first_extent: Size<i32, BufferCoord>,
        offset: f32,
        passes: &[VulkanKawasePass],
    ) -> Result<(), VulkanRendererError> {
        self.capture_material_prefix(
            backdrop_read_area,
            first_level,
            passes,
            Some((first_extent, offset)),
        )
    }

    fn capture_material_prefix(
        &mut self,
        backdrop_read_area: Rectangle<i32, Physical>,
        capture: &VulkanTexture,
        passes: &[VulkanKawasePass],
        direct_downsample: Option<(Size<i32, BufferCoord>, f32)>,
    ) -> Result<(), VulkanRendererError> {
        let Some(capture_image) = capture.image_resource().cloned() else {
            return Err(VulkanRendererError::NotImplemented(
                "framebuffer-effect capture requires an image-backed Vulkan texture",
            ));
        };
        if !capture_image.usage().contains(vk::ImageUsageFlags::TRANSFER_DST) {
            return Err(VulkanRendererError::TemporaryFailure(
                "framebuffer-effect capture texture lacks transfer-destination usage",
            ));
        }

        let (command_buffer, target, frame_size, main_render_pass, main_framebuffer) = {
            let recording = self.recording()?;
            (
                recording.command_buffer,
                recording.target.clone(),
                recording.size,
                recording.pipelines.render_pass,
                recording.framebuffer,
            )
        };
        let Some(source_area) = self.effect_framebuffer_rect(backdrop_read_area)? else {
            return Ok(());
        };
        let capture_area: Rectangle<i32, Physical> = Rectangle::from_size(source_area.size);
        let mut first_pass = None;
        if let Some((extent, offset)) = direct_downsample {
            if !target.usage().contains(vk::ImageUsageFlags::SAMPLED) {
                return Err(VulkanRendererError::TemporaryFailure(
                    "active framebuffer cannot be sampled",
                ));
            }
            first_pass = Some(
                VulkanKawasePass::new(
                    &VulkanTexture::from_framebuffer_image(target.clone()),
                    capture,
                    false,
                    offset,
                )
                .with_extents(Size::from((source_area.size.w, source_area.size.h)), extent)
                .with_source_origin((source_area.loc.x, source_area.loc.y).into())
                .with_encoded_srgb(),
            );
        } else {
            let capture_size = capture_image.size();
            if capture_size.w < source_area.size.w || capture_size.h < source_area.size.h {
                return Err(VulkanRendererError::TemporaryFailure(
                    "framebuffer-effect capture capacity is smaller than transformed read area",
                ));
            }
            if target.vk_format() != capture_image.vk_format() {
                return Err(VulkanRendererError::TemporaryFailure(
                    "framebuffer-effect capture format must match active target",
                ));
            }
            self.renderer.blit.validate_blit_images(
                &target,
                &capture_image,
                source_area,
                capture_area,
                TextureFilter::Linear,
            )?;
        }
        // Resolve all resources before ending the main pass; no invalid graph can
        // leave an otherwise usable frame in a paused render-pass state.
        let first_resolved = self.renderer.resolve_kawase_passes(first_pass.as_slice())?;
        let resolved = self.renderer.resolve_kawase_passes(passes)?;

        // SAFETY: The frame owns an active render pass in this command buffer.
        unsafe {
            self.renderer.device.insert_debug_label(
                command_buffer,
                c"vulkan.frame.framebuffer_effect",
                [0.67, 0.34, 0.91, 1.0],
            );
            self.renderer
                .device
                .device_handle()
                .cmd_end_render_pass(command_buffer);
        }

        if direct_downsample.is_some() {
            for pass in &first_resolved {
                self.record_kawase_pass(command_buffer, pass)?;
            }
        } else {
            self.transition_image_layout(&target, vk::ImageLayout::TRANSFER_SRC_OPTIMAL)?;
            self.transition_image_layout(&capture_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL)?;
            record_image_blit(
                self.renderer.device.device_handle(),
                command_buffer,
                &target,
                &capture_image,
                source_area,
                capture_area,
                TextureFilter::Linear,
            );
        }

        for pass in &resolved {
            self.record_kawase_pass(command_buffer, pass)?;
        }

        self.transition_image_layout(&target, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)?;
        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(main_render_pass)
            .framebuffer(main_framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: frame_size.w.max(1) as u32,
                    height: frame_size.h.max(1) as u32,
                },
            });
        // SAFETY: Main framebuffer/render pass are live for the frame and use
        // LOAD, preserving the completed lower scene captured above.
        unsafe {
            self.renderer.device.device_handle().cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin,
                vk::SubpassContents::INLINE,
            );
        }
        Ok(())
    }

    fn record_kawase_pass(
        &mut self,
        command_buffer: vk::CommandBuffer,
        pass: &ResolvedKawasePass,
    ) -> Result<(), VulkanRendererError> {
        self.transition_image_layout(&pass.source, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
        self.transition_image_layout(&pass.destination, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)?;

        let destination_size = pass.destination_extent;
        let extent = vk::Extent2D {
            width: destination_size.w.max(1) as u32,
            height: destination_size.h.max(1) as u32,
        };
        let attachments = [pass.destination_view];
        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(pass.render_pass)
            .attachments(&attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(1);
        // SAFETY: Device and render-pass handles are live for this renderer.
        let framebuffer = unsafe {
            self.renderer
                .device
                .device_handle()
                .create_framebuffer(&framebuffer_info, None)
        }?;
        self.recording_mut()?.effect_framebuffers.push(framebuffer);

        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(pass.render_pass)
            .framebuffer(framebuffer)
            .render_area(render_area);
        // SAFETY: All handles are valid, recording is active, and every pass
        // overwrites its complete destination.
        unsafe {
            let device = self.renderer.device.device_handle();
            device.cmd_begin_render_pass(command_buffer, &render_pass_begin, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pass.pipeline);
            device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_set_scissor(command_buffer, 0, &[render_area]);
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pass.layout,
                0,
                &[pass.descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                command_buffer,
                pass.layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                push_constants_bytes(&pass.constants),
            );
            device.cmd_draw(command_buffer, 4, 1, 0, 0);
            device.cmd_end_render_pass(command_buffer);
        }
        Ok(())
    }

    /// Read an already filtered native-framebuffer prefix with its exact
    /// captured orientation. It does not pass through client UV transforms.
    #[allow(clippy::too_many_arguments)]
    pub fn render_framebuffer_texture_with_rounded_clip(
        &mut self,
        texture: &VulkanTexture,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        alpha: f32,
        clip: RoundedClip,
    ) -> Result<(), VulkanRendererError> {
        self.render_texture_from_to_internal(
            texture,
            src,
            dst,
            damage,
            &[],
            self.transformation(),
            alpha,
            Some(AnalyticClip::Rounded(clip)),
            TextureRenderEffect::NONE,
            None,
            Some(MaterialTextureOperation::FramebufferCopy),
        )
    }

    /// Bottom-edge variant of the same native captured-prefix operation.
    #[allow(clippy::too_many_arguments)]
    pub fn render_framebuffer_texture_with_bottom_edge_clip(
        &mut self,
        texture: &VulkanTexture,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        alpha: f32,
        clip: BottomEdgeClip,
    ) -> Result<(), VulkanRendererError> {
        self.render_texture_from_to_internal(
            texture,
            src,
            dst,
            damage,
            &[],
            self.transformation(),
            alpha,
            Some(AnalyticClip::BottomEdge(clip)),
            TextureRenderEffect::NONE,
            None,
            Some(MaterialTextureOperation::FramebufferCopy),
        )
    }

    /// Draw the final encoded-sRGB upsample with a rounded material clip.
    #[allow(clippy::too_many_arguments)]
    pub fn render_kawase_with_rounded_clip(
        &mut self,
        texture: &VulkanTexture,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
        clip: RoundedClip,
        output: super::kawase::VulkanKawaseOutput,
    ) -> Result<(), VulkanRendererError> {
        self.render_texture_from_to_internal(
            texture,
            src,
            dst,
            damage,
            &[],
            src_transform,
            alpha,
            Some(AnalyticClip::Rounded(clip)),
            TextureRenderEffect::NONE,
            None,
            Some(MaterialTextureOperation::Kawase(output)),
        )
    }

    /// Draw the final encoded-sRGB upsample with the authored bottom-edge clip.
    #[allow(clippy::too_many_arguments)]
    pub fn render_kawase_with_bottom_edge_clip(
        &mut self,
        texture: &VulkanTexture,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
        clip: BottomEdgeClip,
        output: super::kawase::VulkanKawaseOutput,
    ) -> Result<(), VulkanRendererError> {
        self.render_texture_from_to_internal(
            texture,
            src,
            dst,
            damage,
            &[],
            src_transform,
            alpha,
            Some(AnalyticClip::BottomEdge(clip)),
            TextureRenderEffect::NONE,
            None,
            Some(MaterialTextureOperation::Kawase(output)),
        )
    }
    /// Draw electrical-sRGB tint through the same authored bottom-edge clip
    /// as the blurred pane. `texture` supplies a live immutable descriptor,
    /// but this shader branch emits the colour without sampling/uploading it.
    #[allow(clippy::too_many_arguments)]
    pub fn render_bottom_edge_tint(
        &mut self,
        texture: &VulkanTexture,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        alpha: f32,
        clip: BottomEdgeClip,
        color: [f32; 4],
    ) -> Result<(), VulkanRendererError> {
        self.render_bottom_edge_material(
            texture,
            src,
            dst,
            damage,
            alpha,
            clip,
            super::VulkanMaterialTint::new(color),
        )
    }

    /// Draw the typed tint/body packet, then resolve its analytic clip once.
    #[allow(clippy::too_many_arguments)]
    pub fn render_bottom_edge_material(
        &mut self,
        texture: &VulkanTexture,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        alpha: f32,
        clip: BottomEdgeClip,
        tint: super::VulkanMaterialTint,
    ) -> Result<(), VulkanRendererError> {
        if !tint.valid() {
            return Err(VulkanRendererError::TemporaryFailure(
                "invalid material tint/body packet",
            ));
        }
        self.render_texture_from_to_internal(
            texture,
            src,
            dst,
            damage,
            &[],
            self.transformation(),
            alpha,
            Some(AnalyticClip::BottomEdge(clip)),
            TextureRenderEffect::NONE,
            None,
            Some(MaterialTextureOperation::Tint(tint)),
        )
    }
}
