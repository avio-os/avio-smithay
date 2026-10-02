// One owner mask stays unresolved across the complete ordered item group.
// The atlas and all effect images are admitted before recording starts.

include!("owner_sample.rs");

impl VulkanFrame<'_> {
    /// Current canonical physical sample during a complete owner-group replay.
    pub fn canonical_coverage_lane(&self) -> Option<usize> {
        self.owner_sample_replay.map(|replay| replay.lane)
    }

    fn draw_viewport_rect(
        &self,
        transform: Transform,
        output_size: Size<i32, Physical>,
        size: Size<i32, Physical>,
        dst: Rectangle<i32, Physical>,
    ) -> Option<Rectangle<i32, Physical>> {
        if self.owner_sample_replay.is_some() {
            // The scissor clips writes. Cropping a viewport would stretch its
            // whole source UV range and lose phase at a partial-output owner.
            Some(transform.transform_rect_in(dst, &output_size))
        } else {
            framebuffer_rect(transform, output_size, size, dst)
        }
    }

    fn sample_viewport(&self, rect: Rectangle<i32, Physical>) -> vk::Viewport {
        let mut viewport = to_vk_viewport(rect);
        if let Some(replay) = self.owner_sample_replay {
            let shift = replay.shift(self.resolved_sample_texture);
            viewport.x += shift[0];
            viewport.y += shift[1];
        }
        viewport
    }

    fn shift_analytic_clip(&self, mut clip: TransformedAnalyticClip) -> TransformedAnalyticClip {
        if let Some(replay) = self.owner_sample_replay {
            let shift = replay.shift(false);
            let rect = match &mut clip {
                TransformedAnalyticClip::Rounded(clip) => &mut clip.rect,
                TransformedAnalyticClip::BottomEdge { clip, .. } => &mut clip.rect,
            };
            rect.loc.x += f64::from(shift[0]);
            rect.loc.y += f64::from(shift[1]);
        }
        clip
    }

    fn geometry_frame_size(&self, size: Size<i32, Physical>) -> Size<i32, Physical> {
        self.owner_sample_replay.map_or(size, |replay| replay.frame_size)
    }

    fn effect_framebuffer_rect(
        &self,
        rect: Rectangle<i32, Physical>,
    ) -> Result<Option<Rectangle<i32, Physical>>, VulkanRendererError> {
        let recording = self.recording()?;
        let mut area = framebuffer_rect(
            recording.transform,
            recording.output_size,
            self.geometry_frame_size(recording.size),
            rect,
        );
        if let Some(replay) = self.owner_sample_replay {
            area = area.and_then(|mut area| {
                area.loc += replay.translation;
                area.intersection(replay.bounds)
            });
        }
        Ok(area)
    }

    /// Render four ordered 1x sample lanes, then resolve the complete group once.
    /// The parent remains the exact frozen lower prefix until all lanes finish.
    /// The bounded atlas includes the group's complete framebuffer-read support.
    /// Texture inputs are accepted resolved RGBA content, not reconstructed
    /// engine sample colours. No queue submission or host wait is introduced.
    pub fn render_owner_clipped_group(
        &mut self,
        atlas: &VulkanTexture,
        read: Rectangle<i32, Physical>,
        paint: Rectangle<i32, Physical>,
        owner: RoundedClip,
        damage: &[Rectangle<i32, Physical>],
        mut replay: impl FnMut(&mut Self) -> Result<(), VulkanRendererError>,
    ) -> Result<(), VulkanRendererError> {
        self.recording()?.admit_framebuffer()?;
        if self.owner_sample_replay.is_some() || self.outer_rounded_clip.is_some() || self.draw_alpha != 1.0 {
            return Err(VulkanRendererError::TemporaryFailure(
                "owner groups require an independent neutral-alpha replay epoch",
            ));
        }
        let image = atlas
            .image_resource()
            .cloned()
            .ok_or(VulkanRendererError::TemporaryFailure(
                "owner lane atlas is not image backed",
            ))?;
        let (command_buffer, parent, encoding, pipelines, framebuffer, transform, output_size, frame_size) = {
            let recording = self.recording()?;
            (
                recording.command_buffer,
                recording.target.clone(),
                recording.encoding,
                recording.pipelines,
                recording.framebuffer,
                recording.transform,
                recording.output_size,
                recording.size,
            )
        };
        let Some(source) = framebuffer_rect(transform, output_size, frame_size, read) else {
            return Ok(());
        };
        let Some(paint_native) = framebuffer_rect(transform, output_size, frame_size, paint) else {
            return Ok(());
        };
        if !source.contains_rect(paint_native) {
            return Err(VulkanRendererError::TemporaryFailure(
                "owner paint exceeds frozen prefix support",
            ));
        }
        let extent = source.size;
        if image.size().w < extent.w.saturating_mul(2)
            || image.size().h < extent.h.saturating_mul(2)
            || parent.vk_format() != image.vk_format()
            || !image.usage().contains(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
        {
            return Err(VulkanRendererError::TemporaryFailure(
                "owner atlas lacks exact four-lane capacity/format/usage",
            ));
        }
        for lane in 0..4 {
            let destination = Rectangle::new(((lane % 2) * extent.w, (lane / 2) * extent.h).into(), extent);
            self.renderer.blit.validate_blit_images(
                &parent,
                &image,
                source,
                destination,
                TextureFilter::Nearest,
            )?;
        }
        let atlas_size = Size::from((image.size().w, image.size().h));
        let lane_framebuffer = create_framebuffer(
            self.renderer.device.device_handle(),
            pipelines.render_pass,
            encoding.view(&image),
            atlas_size,
        )?;
        self.recording_mut()?.effect_framebuffers.push(lane_framebuffer);
        let result = (|| {
            unsafe {
                self.renderer
                    .device
                    .device_handle()
                    .cmd_end_render_pass(command_buffer);
            }
            self.transition_image_layout(&parent, vk::ImageLayout::TRANSFER_SRC_OPTIMAL)?;
            self.transition_image_layout(&image, vk::ImageLayout::TRANSFER_DST_OPTIMAL)?;
            for lane in 0..4 {
                let destination = Rectangle::new(
                    ((lane % 2) as i32 * extent.w, (lane / 2) as i32 * extent.h).into(),
                    extent,
                );
                record_image_blit(
                    self.renderer.device.device_handle(),
                    command_buffer,
                    &parent,
                    &image,
                    source,
                    destination,
                    TextureFilter::Nearest,
                );
            }
            self.transition_image_layout(&image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)?;
            {
                let recording = self.recording_mut()?;
                recording.target = image.clone();
                recording.framebuffer = lane_framebuffer;
                recording.size = atlas_size;
            }
            self.outer_rounded_clip = Some(owner);
            for lane in 0..4 {
                let origin = Point::from(((lane % 2) as i32 * extent.w, (lane / 2) as i32 * extent.h));
                let bounds = Rectangle::new(origin, extent);
                self.owner_sample_replay = OwnerSampleReplay::new(frame_size, lane, source, read);
                let begin = vk::RenderPassBeginInfo::default()
                    .render_pass(pipelines.render_pass)
                    .framebuffer(lane_framebuffer)
                    .render_area(to_vk_rect(bounds));
                unsafe {
                    self.renderer.device.device_handle().cmd_begin_render_pass(
                        command_buffer,
                        &begin,
                        vk::SubpassContents::INLINE,
                    );
                }
                replay(self)?;
                unsafe {
                    self.renderer
                        .device
                        .device_handle()
                        .cmd_end_render_pass(command_buffer);
                }
            }
            Ok(())
        })();
        self.owner_sample_replay = None;
        self.outer_rounded_clip = None;
        self.resolved_sample_texture = false;
        if let Some(recording) = self.recording.as_mut() {
            recording.target = parent.clone();
            recording.framebuffer = framebuffer;
            recording.size = frame_size;
        }
        if let Err(error) = result {
            // No partial lane or fabricated completion may escape. The original
            // and atlas resources remain in recording custody until rollback.
            self.abort_recording();
            return Err(error);
        }
        let result = (|| {
            self.transition_image_layout(&image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
            self.transition_image_layout(&parent, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)?;
            let begin = vk::RenderPassBeginInfo::default()
                .render_pass(pipelines.render_pass)
                .framebuffer(framebuffer)
                .render_area(to_vk_rect(Rectangle::from_size(frame_size)));
            unsafe {
                self.renderer.device.device_handle().cmd_begin_render_pass(
                    command_buffer,
                    &begin,
                    vk::SubpassContents::INLINE,
                );
            }
            let src = Rectangle::<i32, BufferCoord>::new(
                (
                    paint_native.loc.x - source.loc.x,
                    paint_native.loc.y - source.loc.y,
                )
                    .into(),
                (paint_native.size.w, paint_native.size.h).into(),
            )
            .to_f64();
            self.render_texture_from_to_internal(
                atlas,
                src,
                paint,
                damage,
                &[],
                transform,
                1.0,
                None,
                TextureRenderEffect::NONE,
                Some(1.0),
                Some(MaterialTextureOperation::OwnerResolve(extent)),
            )
        })();
        if result.is_err() {
            self.abort_recording();
        }
        result
    }
}
