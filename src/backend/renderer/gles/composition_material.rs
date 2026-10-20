#[repr(C)]
#[derive(Clone, Copy)]
struct KawaseConstants {
    halfpixel: [f32; 2],
    offset: f32,
    mode: u32,
    encode_output: u32,
    saturation: f32,
    encoded_srgb: u32,
    contrast: f32,
    brightness: f32,
    pad: u32,
    source_uv_scale: [f32; 2],
    source_uv_min: [f32; 2],
    source_uv_max: [f32; 2],
    source_uv_offset: [f32; 2],
    std140_padding: [u32; 2],
}

impl GlesRenderer {
    fn execute_kawase(&self, pass: &GlesKawasePass, source_in_frame: bool) -> Result<(), GlesError> {
        self.check_completion_health()?;
        let source_size = pass.source.size();
        let destination_size = pass.destination.size();
        if Arc::ptr_eq(&pass.source.0, &pass.destination.0)
            || pass.source_extent.is_empty()
            || pass.destination_extent.is_empty()
            || pass.source_origin.x < 0
            || pass.source_origin.y < 0
            || pass.source_origin.x.saturating_add(pass.source_extent.w) > source_size.w
            || pass.source_origin.y.saturating_add(pass.source_extent.h) > source_size.h
            || pass.destination_extent.w > destination_size.w
            || pass.destination_extent.h > destination_size.h
            || !pass.offset.is_finite()
            || pass.offset < 0.0
        {
            return Err(GlesError::Composition(
                "invalid Kawase source/destination extent or alias",
            ));
        }
        let state = self
            .composition
            .as_ref()
            .ok_or(GlesError::Composition("composition not prepared"))?;
        let framebuffer = state.attachment(&pass.destination)?;
        let source_sync = if source_in_frame {
            None
        } else {
            Some(
                pass.source
                    .0
                    .sync
                    .try_read()
                    .map_err(|_| GlesError::Composition("Kawase source is still bound for writing"))?,
            )
        };
        if let Some(sync) = &source_sync {
            self.native_fences.observe(sync.wait_for_upload(&self.gl))?;
        }
        let mut destination_sync = pass
            .destination
            .0
            .sync
            .try_write()
            .map_err(|_| GlesError::Composition("Kawase destination is already bound or sampled"))?;
        self.native_fences
            .observe(destination_sync.wait_for_all(&self.gl))?;
        let constants = KawaseConstants {
            halfpixel: [
                0.5 / pass.source_extent.w.min(pass.destination_extent.w) as f32,
                0.5 / pass.source_extent.h.min(pass.destination_extent.h) as f32,
            ],
            offset: pass.offset,
            mode: pass.upsample as u32,
            encode_output: (!pass.destination.0.storage_is_srgb) as u32,
            saturation: pass.transform.saturation,
            encoded_srgb: (pass.encoding == GlesKawaseEncoding::EncodedSrgb) as u32,
            contrast: pass.transform.contrast,
            brightness: pass.transform.brightness,
            pad: 0,
            source_uv_scale: [
                pass.source_extent.w as f32 / source_size.w as f32,
                pass.source_extent.h as f32 / source_size.h as f32,
            ],
            source_uv_min: [
                (pass.source_origin.x as f32 + 0.5) / source_size.w as f32,
                (pass.source_origin.y as f32 + 0.5) / source_size.h as f32,
            ],
            source_uv_max: [
                (pass.source_origin.x as f32 + pass.source_extent.w as f32 - 0.5) / source_size.w as f32,
                (pass.source_origin.y as f32 + pass.source_extent.h as f32 - 0.5) / source_size.h as f32,
            ],
            source_uv_offset: [
                pass.source_origin.x as f32 / source_size.w as f32,
                pass.source_origin.y as f32 / source_size.h as f32,
            ],
            std140_padding: [0; 2],
        };
        unsafe {
            let gl = &self.gl;
            let program = state.kawase_program;
            gl.BindFramebuffer(ffi::FRAMEBUFFER, framebuffer);
            gl.UseProgram(program);
            gl.BindBuffer(ffi::UNIFORM_BUFFER, state.uniform_buffer);
            gl.BufferSubData(
                ffi::UNIFORM_BUFFER,
                0,
                mem::size_of::<KawaseConstants>() as isize,
                (&constants as *const KawaseConstants).cast(),
            );
            gl.BindBufferBase(ffi::UNIFORM_BUFFER, 0, state.uniform_buffer);
            state.bind_frame(
                gl,
                [
                    0.0,
                    0.0,
                    pass.destination_extent.w as f32,
                    pass.destination_extent.h as f32,
                ],
                (destination_size.w, destination_size.h).into(),
                pass.source.0.storage_is_srgb,
                false,
                false,
                pass.destination.0.storage_is_srgb,
                source_size,
            );
            gl.ActiveTexture(ffi::TEXTURE0);
            gl.BindTexture(ffi::TEXTURE_2D, pass.source.tex_id());
            gl.Viewport(0, 0, destination_size.w, destination_size.h);
            gl.Enable(ffi::SCISSOR_TEST);
            gl.Scissor(0, 0, pass.destination_extent.w, pass.destination_extent.h);
            gl.Disable(ffi::BLEND);
            gl.DrawArrays(ffi::TRIANGLE_STRIP, 0, 4);
            gl.Enable(ffi::BLEND);
            gl.BlendFunc(ffi::ONE, ffi::ONE_MINUS_SRC_ALPHA);
            gl.BindBuffer(ffi::UNIFORM_BUFFER, 0);
            gl.BindSampler(0, 0);
            let code = gl.GetError();
            if code != ffi::NO_ERROR {
                return self.native_fences.observe(Err(GlesError::GlSync {
                    operation: "Kawase pass",
                    code,
                }));
            }
        }
        if let Some(sync) = source_sync {
            self.native_fences.observe(sync.update_read(&self.gl))?;
        }
        self.native_fences
            .observe(destination_sync.update_write(&self.gl))
    }
}

impl GlesFrame<'_, '_> {
    fn copy_prefix_to(
        &self,
        source: Rectangle<i32, Physical>,
        destination: &GlesTexture,
        offset: Point<i32, Physical>,
    ) -> Result<(), GlesError> {
        let current = self.composition_source()?;
        if Arc::ptr_eq(&current.0, &destination.0)
            || offset.x < 0
            || offset.y < 0
            || offset.x.saturating_add(source.size.w) > destination.size().w
            || offset.y.saturating_add(source.size.h) > destination.size().h
        {
            return Err(GlesError::Composition("invalid prefix capture capacity or alias"));
        }
        let state = self
            .renderer
            .composition
            .as_ref()
            .ok_or(GlesError::Composition("composition not prepared"))?;
        let destination_fbo = state.attachment(destination)?;
        let mut sync = destination
            .0
            .sync
            .try_write()
            .map_err(|_| GlesError::Composition("prefix destination is already bound or sampled"))?;
        self.renderer
            .native_fences
            .observe(sync.wait_for_all(&self.renderer.gl))?;
        self.bind_composition_target()?;
        unsafe {
            let gl = &self.renderer.gl;
            gl.ReadBuffer(ffi::COLOR_ATTACHMENT0);
            gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, destination_fbo);
            gl.Disable(ffi::SCISSOR_TEST);
            gl.BlitFramebuffer(
                source.loc.x,
                source.loc.y,
                source.loc.x + source.size.w,
                source.loc.y + source.size.h,
                offset.x,
                offset.y,
                offset.x + source.size.w,
                offset.y + source.size.h,
                ffi::COLOR_BUFFER_BIT,
                ffi::NEAREST,
            );
            gl.Enable(ffi::SCISSOR_TEST);
            let code = gl.GetError();
            if code != ffi::NO_ERROR {
                return self.renderer.native_fences.observe(Err(GlesError::GlSync {
                    operation: "prefix capture",
                    code,
                }));
            }
        }
        self.renderer
            .native_fences
            .observe(sync.update_write(&self.renderer.gl))?;
        self.bind_composition_target()
    }

    /// Freeze the exact bounded lower prefix, then run its admitted filter graph.
    pub fn capture_and_filter_framebuffer(
        &mut self,
        region: Rectangle<i32, Physical>,
        capture: &GlesTexture,
        passes: &[GlesKawasePass],
    ) -> Result<(), GlesError> {
        let Some(source) = self.native_region(region) else {
            return Ok(());
        };
        let result = (|| {
            self.copy_prefix_to(source, capture, (0, 0).into())?;
            for pass in passes {
                self.renderer.execute_kawase(pass, false)?;
            }
            Ok(())
        })();
        let restored = self.bind_composition_target();
        result.and(restored)
    }

    /// Directly filter the active prefix into the first pyramid level; no full
    /// size intermediate or CPU transfer is introduced.
    pub fn capture_and_downsample_framebuffer(
        &mut self,
        region: Rectangle<i32, Physical>,
        first_level: &GlesTexture,
        first_extent: Size<i32, BufferCoord>,
        offset: f32,
        passes: &[GlesKawasePass],
    ) -> Result<(), GlesError> {
        let Some(source) = self.native_region(region) else {
            return Ok(());
        };
        let current = self.composition_source()?;
        let first = GlesKawasePass::new(&current, first_level, false, offset)
            .with_extents((source.size.w, source.size.h).into(), first_extent)
            .with_source_origin((source.loc.x, source.loc.y).into())
            .with_encoded_srgb();
        let result = (|| {
            self.renderer.execute_kawase(&first, true)?;
            for pass in passes {
                self.renderer.execute_kawase(pass, false)?;
            }
            Ok(())
        })();
        let restored = self.bind_composition_target();
        result.and(restored)
    }

    /// Mix the completed group against its frozen lower prefix, including alpha.
    pub fn interpolate_framebuffer_prefix(
        &mut self,
        prefix: &GlesTexture,
        region: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        group_opacity: f32,
    ) -> Result<(), GlesError> {
        if !group_opacity.is_finite() || !(0.0..=1.0).contains(&group_opacity) {
            return Err(GlesError::Composition("invalid group opacity"));
        }
        if self.outer_rounded_clip.is_some() && self.owner_sample_replay.is_none() {
            return Err(GlesError::Composition(
                "owner mask cannot clip a prefix interpolation",
            ));
        }
        let region = match self.owner_sample_replay {
            Some(r) => match r.clip_output_region(region) {
                Some(region) => region,
                None => return Ok(()),
            },
            None => region,
        };
        let Some(source) = self.native_region(region) else {
            return Ok(());
        };
        let src = Rectangle::<i32, BufferCoord>::from_size((source.size.w, source.size.h).into()).to_f64();
        let previous = self.resolved_sample_texture;
        self.resolved_sample_texture = self.owner_sample_replay.is_some();
        let result = self.draw_composition_texture(
            prefix,
            src,
            region,
            damage,
            self.transform,
            1.0,
            None,
            TextureRenderEffect::NONE,
            Some(MaterialOperation::Copy),
            Some(1.0 - group_opacity),
        );
        self.resolved_sample_texture = previous;
        result
    }

    /// Current unresolved canonical sample identity.
    pub fn canonical_coverage_lane(&self) -> Option<usize> {
        self.owner_sample_replay.map(|r| r.lane)
    }

    /// Replay the complete group at all four canonical samples over one frozen
    /// prefix, then resolve once. Every attachment must be admitted in advance.
    #[allow(clippy::too_many_arguments)]
    pub fn render_owner_clipped_group(
        &mut self,
        atlas: &GlesTexture,
        read: Rectangle<i32, Physical>,
        paint: Rectangle<i32, Physical>,
        owner: RoundedClip,
        damage: &[Rectangle<i32, Physical>],
        mut replay: impl FnMut(&mut Self) -> Result<(), GlesError>,
    ) -> Result<(), GlesError> {
        if self.owner_sample_replay.is_some() || self.outer_rounded_clip.is_some() || self.draw_alpha != 1.0 {
            return Err(GlesError::Composition(
                "owner group requires independent neutral replay",
            ));
        }
        let Some(source) = self.native_region(read) else {
            return Ok(());
        };
        let Some(paint_native) = self.native_region(paint) else {
            return Ok(());
        };
        if !source.contains_rect(paint_native)
            || atlas.size().w < source.size.w.saturating_mul(2)
            || atlas.size().h < source.size.h.saturating_mul(2)
        {
            return Err(GlesError::Composition(
                "owner atlas lacks complete four-lane read support",
            ));
        }
        let framebuffer = self.renderer.composition.as_ref().unwrap().attachment(atlas)?;
        for lane in 0..4 {
            self.copy_prefix_to(
                source,
                atlas,
                ((lane % 2) * source.size.w, (lane / 2) * source.size.h).into(),
            )?;
        }
        let parent = self.composition_target.take();
        let parent_size = self.composition_target_size();
        self.composition_target = Some((framebuffer, atlas.clone()));
        self.outer_rounded_clip = Some(owner);
        let result = (|| {
            for lane in 0..4 {
                self.owner_sample_replay = OwnerSampleReplay::new(parent_size, lane, source, read);
                self.bind_composition_target()?;
                replay(self)?;
            }
            Ok(())
        })();
        self.owner_sample_replay = None;
        self.outer_rounded_clip = None;
        self.resolved_sample_texture = false;
        self.composition_target = parent;
        let restore = self.bind_composition_target();
        if let Err(error) = result.and(restore) {
            self.renderer.native_fences.poison();
            return Err(error);
        }
        self.renderer
            .native_fences
            .observe(atlas.0.sync.write().unwrap().update_write(&self.renderer.gl))?;
        let src = Rectangle::<i32, BufferCoord>::new(
            (
                paint_native.loc.x - source.loc.x,
                paint_native.loc.y - source.loc.y,
            )
                .into(),
            (paint_native.size.w, paint_native.size.h).into(),
        )
        .to_f64();
        self.draw_composition_texture(
            atlas,
            src,
            paint,
            damage,
            self.transform,
            1.0,
            None,
            TextureRenderEffect::NONE,
            Some(MaterialOperation::OwnerResolve(source.size)),
            Some(1.0),
        )
    }
}

impl GlesFrame<'_, '_> {
    /// Draw a frozen prefix with its native captured orientation and analytic clip.
    #[allow(clippy::too_many_arguments)]
    pub fn render_framebuffer_texture_with_rounded_clip(
        &mut self,
        texture: &GlesTexture,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        alpha: f32,
        clip: RoundedClip,
    ) -> Result<(), GlesError> {
        self.draw_composition_texture(
            texture,
            src,
            dst,
            damage,
            self.transform,
            alpha,
            Some(AnalyticClip::Rounded(clip)),
            TextureRenderEffect::NONE,
            Some(MaterialOperation::Copy),
            None,
        )
    }
    /// Fuse the final encoded Kawase pass with its analytic material clip.
    #[allow(clippy::too_many_arguments)]
    pub fn render_kawase_with_rounded_clip(
        &mut self,
        texture: &GlesTexture,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        transform: Transform,
        alpha: f32,
        clip: RoundedClip,
        output: GlesKawaseOutput,
    ) -> Result<(), GlesError> {
        self.draw_composition_texture(
            texture,
            src,
            dst,
            damage,
            transform,
            alpha,
            Some(AnalyticClip::Rounded(clip)),
            TextureRenderEffect::NONE,
            Some(MaterialOperation::Kawase(output)),
            None,
        )
    }
    /// Draw a frozen prefix with its native captured orientation and analytic clip.
    #[allow(clippy::too_many_arguments)]
    pub fn render_framebuffer_texture_with_bottom_edge_clip(
        &mut self,
        texture: &GlesTexture,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        alpha: f32,
        clip: BottomEdgeClip,
    ) -> Result<(), GlesError> {
        self.draw_composition_texture(
            texture,
            src,
            dst,
            damage,
            self.transform,
            alpha,
            Some(AnalyticClip::BottomEdge(clip)),
            TextureRenderEffect::NONE,
            Some(MaterialOperation::Copy),
            None,
        )
    }
    /// Fuse the final encoded Kawase pass with its analytic material clip.
    #[allow(clippy::too_many_arguments)]
    pub fn render_kawase_with_bottom_edge_clip(
        &mut self,
        texture: &GlesTexture,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        transform: Transform,
        alpha: f32,
        clip: BottomEdgeClip,
        output: GlesKawaseOutput,
    ) -> Result<(), GlesError> {
        self.draw_composition_texture(
            texture,
            src,
            dst,
            damage,
            transform,
            alpha,
            Some(AnalyticClip::BottomEdge(clip)),
            TextureRenderEffect::NONE,
            Some(MaterialOperation::Kawase(output)),
            None,
        )
    }
    /// Draw the material tint/body packet through the authored bottom edge.
    #[allow(clippy::too_many_arguments)]
    pub fn render_bottom_edge_material(
        &mut self,
        texture: &GlesTexture,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        alpha: f32,
        clip: BottomEdgeClip,
        tint: GlesMaterialTint,
    ) -> Result<(), GlesError> {
        if !tint.valid() {
            return Err(GlesError::Composition("invalid material tint"));
        }
        self.draw_composition_texture(
            texture,
            src,
            dst,
            damage,
            self.transform,
            alpha,
            Some(AnalyticClip::BottomEdge(clip)),
            TextureRenderEffect::NONE,
            Some(MaterialOperation::Tint(tint)),
            None,
        )
    }
}
