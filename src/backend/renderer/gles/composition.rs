//! Native lowering of the compositor's bounded GLES draw and effect packets.
include!("composition_types.rs");

#[derive(Debug)]
struct PreparedAttachment {
    texture: std::sync::Weak<GlesTextureInternal>,
    framebuffer: u32,
}

#[derive(Debug)]
pub(super) struct CompositionPrograms {
    program: u32,
    external_program: Option<u32>,
    kawase_program: u32,
    context: ContextId<GlesTexture>,
    ready: SyncPoint,
    cleanup: Sender<CleanupResource>,
    retirement: Option<GlesRetirementSlot<GlesSharedCleanup>>,
}

impl Drop for CompositionPrograms {
    fn drop(&mut self) {
        let cleanup = GlesSharedCleanup {
            resources: [
                Some(CleanupResource::Program(self.program)),
                Some(CleanupResource::Program(self.kawase_program)),
                self.external_program.map(CleanupResource::Program),
            ],
            images: None,
            sender: self.cleanup.clone(),
        };
        if let Some(retirement) = self.retirement.take() {
            retirement.retire(cleanup);
        } else {
            drop(cleanup);
        }
    }
}

/// Immutable shared native programs compiled on a cold helper context.
/// All changing draw data lives in each renderer's private uniform buffers.
#[derive(Debug, Clone)]
pub struct GlesPreparedCompositionPrograms(Arc<CompositionPrograms>);

impl GlesPreparedCompositionPrograms {
    /// Whether this immutable bank contains a successfully linked external
    /// image sampler using the same clipping and material equations.
    pub fn supports_external_sources(&self) -> bool {
        self.0.external_program.is_some()
    }
}

pub(super) struct CompositionState {
    programs: GlesPreparedCompositionPrograms,
    uniform_buffer: u32,
    frame_buffer: u32,
    samplers: [u32; 2],
    white: GlesTexture,
    attachments: Vec<PreparedAttachment>,
    spare_framebuffers: Vec<u32>,
    damage: Vec<Vec<Rectangle<i32, Physical>>>,
}

/// Helper-allocated bounded backing for nested native damage operations.
#[derive(Debug)]
pub struct GlesPreparedDamageScratch(Vec<Vec<Rectangle<i32, Physical>>>);
impl GlesPreparedDamageScratch {
    /// Allocate the exact declared backing on a helper thread.
    pub fn new(slots: usize, rectangles: usize) -> Result<Self, GlesError> {
        GlesRenderer::damage_scratch_bytes(slots, rectangles)
            .ok_or(GlesError::Composition("damage backing cost overflows"))?;
        let mut backing = Vec::new();
        backing
            .try_reserve_exact(slots)
            .map_err(|_| GlesError::Composition("damage backing allocation failed"))?;
        for _ in 0..slots {
            let mut rectangles_backing = Vec::new();
            rectangles_backing
                .try_reserve_exact(rectangles)
                .map_err(|_| GlesError::Composition("damage rectangle backing allocation failed"))?;
            backing.push(rectangles_backing);
        }
        Ok(Self(backing))
    }
}

#[repr(C)]
struct FrameUniforms {
    destination: [f32; 4],
    target_size: [f32; 2],
    storage_is_srgb: u32,
    sample_nearest: u32,
    source_size: [f32; 2],
    padding: [u32; 2],
}

impl std::ops::Deref for CompositionState {
    type Target = CompositionPrograms;
    fn deref(&self) -> &Self::Target {
        &self.programs.0
    }
}

fn shader_source(source: &str, kawase: bool, external: bool) -> String {
    let mut source = source
        .replace(
            "#version 450",
            "#version 300 es\nprecision highp float;\nprecision highp int;",
        )
        .replace("layout(set = 0, binding = 0)", "")
        .replace("layout(push_constant)", "layout(std140)")
        .replace("layout(location = 0) in", "in")
        .replace("bitCount(", "count_mask(");
    // The common Vulkan source is the sole authority for clip, material and
    // texture-effect equations. Only native declaration/sampling is lowered.
    source = source.replace("texture(texture_sampler, ", "sample_encoded(");
    let sampling = include_str!("shaders/encoded_sampling.glsl");
    let insert = source.find("layout(std140)").expect("common shader constants");
    source.insert_str(insert, sampling);
    if !kawase {
        source = source.replace("if (material_kind == 5) {", "if (material_kind == 8) {\n        sampled = vec4(constants.effect.yzw, constants.effect_params.x);\n    } else if (material_kind == 5) {");
    }
    if external {
        source = source
            .replace(
                "#version 300 es",
                "#version 300 es\n#extension GL_OES_EGL_image_external_essl3 : require",
            )
            .replace(
                "uniform sampler2D texture_sampler",
                "uniform highp samplerExternalOES texture_sampler",
            )
            .replace("textureSize(texture_sampler, 0)", "ivec2(frame.source_size)");
        let start = source.find("vec4 encoded_texel(").unwrap();
        let end = source[start..].find("uint count_mask(").unwrap() + start;
        source.replace_range(
            start..end,
            "vec4 sample_encoded(vec2 uv) { return texture(texture_sampler, uv); }\n",
        );
    }
    source
}

impl CompositionState {
    fn new(renderer: &GlesRenderer, programs: GlesPreparedCompositionPrograms) -> Result<Self, GlesError> {
        if programs.0.context != renderer.context_id() {
            return Err(GlesError::Composition(
                "program bank belongs to another share group",
            ));
        }
        let gl = &renderer.gl;
        unsafe {
            let mut uniform_buffer = 0;
            let mut frame_buffer = 0;
            gl.GenBuffers(1, &mut uniform_buffer);
            gl.BindBuffer(ffi::UNIFORM_BUFFER, uniform_buffer);
            gl.BufferData(ffi::UNIFORM_BUFFER, 128, ptr::null(), ffi::DYNAMIC_DRAW);
            gl.GenBuffers(1, &mut frame_buffer);
            gl.BindBuffer(ffi::UNIFORM_BUFFER, frame_buffer);
            gl.BufferData(
                ffi::UNIFORM_BUFFER,
                mem::size_of::<FrameUniforms>() as isize,
                ptr::null(),
                ffi::DYNAMIC_DRAW,
            );
            gl.BindBuffer(ffi::UNIFORM_BUFFER, 0);
            let mut samplers = [0; 2];
            gl.GenSamplers(2, samplers.as_mut_ptr());
            for (sampler, filter) in samplers.iter().zip([ffi::NEAREST, ffi::LINEAR]) {
                gl.SamplerParameteri(*sampler, ffi::TEXTURE_MIN_FILTER, filter as i32);
                gl.SamplerParameteri(*sampler, ffi::TEXTURE_MAG_FILTER, filter as i32);
                gl.SamplerParameteri(*sampler, ffi::TEXTURE_WRAP_S, ffi::CLAMP_TO_EDGE as i32);
                gl.SamplerParameteri(*sampler, ffi::TEXTURE_WRAP_T, ffi::CLAMP_TO_EDGE as i32);
            }
            let mut white = 0;
            gl.GenTextures(1, &mut white);
            gl.BindTexture(ffi::TEXTURE_2D, white);
            gl.TexImage2D(
                ffi::TEXTURE_2D,
                0,
                ffi::RGBA8 as i32,
                1,
                1,
                0,
                ffi::RGBA,
                ffi::UNSIGNED_BYTE,
                [255u8; 4].as_ptr().cast(),
            );
            let white = GlesTexture::from_raw(renderer, Some(ffi::RGBA8), true, white, (1, 1).into());
            let code = gl.GetError();
            if code != ffi::NO_ERROR {
                gl.DeleteBuffers(1, &uniform_buffer);
                gl.DeleteBuffers(1, &frame_buffer);
                gl.DeleteSamplers(2, samplers.as_ptr());
                return Err(GlesError::GlSync {
                    operation: "composition context state creation",
                    code,
                });
            }
            Ok(Self {
                programs,
                uniform_buffer,
                frame_buffer,
                samplers,
                white,
                attachments: Vec::new(),
                spare_framebuffers: Vec::new(),
                damage: Vec::new(),
            })
        }
    }

    pub(super) fn destroy(self, gl: &ffi::Gles2) -> Result<(), GlesError> {
        unsafe {
            for attachment in &self.attachments {
                gl.DeleteFramebuffers(1, &attachment.framebuffer);
                check_cleanup_error(gl, "composition framebuffer deletion")?;
            }
            gl.DeleteFramebuffers(
                self.spare_framebuffers.len() as i32,
                self.spare_framebuffers.as_ptr(),
            );
            check_cleanup_error(gl, "composition spare framebuffer deletion")?;
            gl.DeleteBuffers(1, &self.uniform_buffer);
            check_cleanup_error(gl, "composition uniform buffer deletion")?;
            gl.DeleteBuffers(1, &self.frame_buffer);
            check_cleanup_error(gl, "composition frame buffer deletion")?;
            gl.DeleteSamplers(2, self.samplers.as_ptr());
            check_cleanup_error(gl, "composition sampler deletion")?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn bind_frame(
        &self,
        gl: &ffi::Gles2,
        destination: [f32; 4],
        target_size: Size<i32, Physical>,
        storage_is_srgb: bool,
        nearest: bool,
        source_size: Size<i32, BufferCoord>,
    ) {
        let data = FrameUniforms {
            destination,
            target_size: [target_size.w as f32, target_size.h as f32],
            storage_is_srgb: storage_is_srgb as u32,
            sample_nearest: nearest as u32,
            source_size: [source_size.w as f32, source_size.h as f32],
            padding: [0; 2],
        };
        gl.BindBuffer(ffi::UNIFORM_BUFFER, self.frame_buffer);
        gl.BufferSubData(
            ffi::UNIFORM_BUFFER,
            0,
            mem::size_of::<FrameUniforms>() as isize,
            (&data as *const FrameUniforms).cast(),
        );
        gl.BindBufferBase(ffi::UNIFORM_BUFFER, 1, self.frame_buffer);
        gl.BindSampler(0, self.samplers[if nearest { 0 } else { 1 }]);
    }

    pub(super) fn attachment(&self, texture: &GlesTexture) -> Result<u32, GlesError> {
        self.attachments
            .iter()
            .find(|a| a.texture.as_ptr() == Arc::as_ptr(&texture.0))
            .map(|a| a.framebuffer)
            .ok_or(GlesError::Composition(
                "attachment was not admitted on a cold resource turn",
            ))
    }
}

impl GlesRenderer {
    /// Copy prepared framebuffer textures with explicit compositor encoding.
    /// Source and destination store encoded premultiplied values. The
    /// destination rectangle is replaced, including alpha, without a transfer.
    pub fn blit_composition_texture(
        &mut self,
        source: &GlesTarget<'_>,
        destination: &mut GlesTarget<'_>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, GlesError> {
        self.check_completion_health()?;
        source.validate_context(self)?;
        destination.make_current(self)?;
        let GlesTargetInternal::Texture {
            texture, sync_lock, ..
        } = &source.0
        else {
            return Err(GlesError::Composition(
                "color transfer requires a prepared source texture",
            ));
        };
        let GlesTargetInternal::Texture { texture: target, .. } = &destination.0 else {
            return Err(GlesError::Composition(
                "color transfer requires a prepared destination texture",
            ));
        };
        if Arc::ptr_eq(&texture.0, &target.0)
            || !Rectangle::from_size((texture.width() as i32, texture.height() as i32).into())
                .contains_rect(src)
            || !Rectangle::from_size((target.width() as i32, target.height() as i32).into())
                .contains_rect(dst)
            || src.is_empty()
            || dst.is_empty()
        {
            return Err(GlesError::Composition("invalid color transfer extent or alias"));
        }
        let constants = TexturePushConstants::default().with_src_rect(
            [
                src.loc.x as f32 / texture.width() as f32,
                src.loc.y as f32 / texture.height() as f32,
            ],
            [
                src.size.w as f32 / texture.width() as f32,
                src.size.h as f32 / texture.height() as f32,
            ],
        );
        self.native_fences.observe(sync_lock.wait_for_upload(&self.gl))?;
        self.composition_draw(
            texture,
            (target.width() as i32, target.height() as i32).into(),
            analytic_clip_rect_push_constant(dst.to_f64()),
            &constants,
            filter == TextureFilter::Nearest,
            true,
            true,
            None,
            std::iter::once(dst),
        )?;
        self.native_fences.observe(sync_lock.update_read(&self.gl))?;
        if let GlesTargetInternal::Texture { sync_lock, .. } = &mut destination.0 {
            self.native_fences.observe(sync_lock.update_write(&self.gl))?;
        }
        self.finish_native()
    }

    /// Whether the adopted native composition bank accepts external images.
    /// This is false until the cold shader preparation/adoption succeeds.
    pub fn supports_composition_external_sources(&self) -> bool {
        self.composition
            .as_ref()
            .is_some_and(|state| state.programs.supports_external_sources())
    }

    /// Compile one immutable native program bank on a cold helper context.
    pub fn prepare_composition_programs(&mut self) -> Result<GlesPreparedCompositionPrograms, GlesError> {
        self.check_completion_health()?;
        if self.gl_version < version::GLES_3_0 {
            return Err(GlesError::GLVersionNotSupported(version::GLES_3_0));
        }
        unsafe {
            self.egl.make_current()?;
        }
        if let Some(state) = &self.composition {
            return Ok(state.programs.clone());
        }
        let texture = shader_source(include_str!("../vulkan/shaders/texture.frag"), false, false);
        let kawase = shader_source(include_str!("../vulkan/shaders/kawase.frag"), true, false);
        let external = self
            .extensions
            .iter()
            .any(|extension| extension == "GL_OES_EGL_image_external_essl3")
            .then(|| shader_source(include_str!("../vulkan/shaders/texture.frag"), false, true));
        let vertex = include_str!("shaders/composition.vert");
        unsafe {
            let gl = &self.gl;
            let program = link_program(gl, vertex, &texture)?;
            let kawase_program = match link_program(gl, vertex, &kawase) {
                Ok(program) => program,
                Err(error) => {
                    gl.DeleteProgram(program);
                    return Err(error);
                }
            };
            let external_program = match external
                .map(|source| link_program(gl, vertex, &source))
                .transpose()
            {
                Ok(program) => program,
                Err(error) => {
                    gl.DeleteProgram(program);
                    gl.DeleteProgram(kawase_program);
                    return Err(error);
                }
            };
            for (program, block) in [
                Some((program, c"TexturePushConstants")),
                Some((kawase_program, c"KawasePushConstants")),
                external_program.map(|program| (program, c"TexturePushConstants")),
            ]
            .into_iter()
            .flatten()
            {
                gl.UniformBlockBinding(program, gl.GetUniformBlockIndex(program, block.as_ptr()), 0);
                gl.UniformBlockBinding(
                    program,
                    gl.GetUniformBlockIndex(program, c"FrameUniforms".as_ptr()),
                    1,
                );
                gl.UseProgram(program);
                gl.Uniform1i(gl.GetUniformLocation(program, c"texture_sampler".as_ptr()), 0);
            }
            gl.UseProgram(0);
            gl.Flush();
            let code = gl.GetError();
            if code != ffi::NO_ERROR {
                gl.DeleteProgram(program);
                gl.DeleteProgram(kawase_program);
                if let Some(program) = external_program {
                    gl.DeleteProgram(program);
                }
                return Err(GlesError::GlSync {
                    operation: "composition program initialization",
                    code,
                });
            }
            let ready = match self.finish_native() {
                Ok(ready) => ready,
                Err(error) => {
                    gl.DeleteProgram(program);
                    gl.DeleteProgram(kawase_program);
                    if let Some(program) = external_program {
                        gl.DeleteProgram(program);
                    }
                    return Err(error);
                }
            };
            Ok(GlesPreparedCompositionPrograms(Arc::new(CompositionPrograms {
                program,
                external_program,
                kawase_program,
                context: self.context_id(),
                ready,
                cleanup: self.gles_cleanup().sender.clone(),
                retirement: self.origin.as_ref().map(GlesDeviceOrigin::retirement_slot),
            })))
        }
    }

    /// Adopt an immutable shared program bank and create only context-local
    /// uniform/sampler state on this renderer's cold owner turn.
    pub fn adopt_composition_programs(
        &mut self,
        programs: GlesPreparedCompositionPrograms,
    ) -> Result<(), GlesError> {
        self.check_completion_health()?;
        unsafe {
            self.egl.make_current()?;
        }
        if programs.0.context != self.context_id() {
            return Err(GlesError::Composition(
                "program bank belongs to another share group",
            ));
        }
        // Compile/link and immutable uniform initialization occurred on the
        // helper context. Rust publication does not order shared GL work.
        self.wait_native(&programs.0.ready)?;
        if let Some(state) = self.composition.as_mut() {
            state.programs = programs;
            return Ok(());
        }
        self.composition = Some(CompositionState::new(self, programs)?);
        Ok(())
    }

    /// Prepare programs for encoded-premultiplied compositing. This may compile shaders
    /// and must run before opening a frame, on the context's owning thread.
    pub fn prepare_composition(&mut self) -> Result<(), GlesError> {
        self.check_completion_health()?;
        if self.gl_version < version::GLES_3_0 {
            return Err(GlesError::GLVersionNotSupported(version::GLES_3_0));
        }
        unsafe {
            self.egl.make_current()?;
        }
        if self.composition.is_none() {
            let programs = self.prepare_composition_programs()?;
            self.adopt_composition_programs(programs)?;
        }
        Ok(())
    }

    /// Create a compositor attachment with explicit storage and encoding.
    /// All formats use UNORM/float attachment storage, matching the shared
    /// encoded blend space. Client imports retain their separate role.
    pub fn create_composition_buffer(
        &mut self,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
    ) -> Result<GlesTexture, GlesError> {
        self.check_completion_health()?;
        unsafe {
            self.egl.make_current()?;
        }
        let (storage, metadata, storage_is_srgb) = match format {
            Fourcc::Argb8888 | Fourcc::Xrgb8888 | Fourcc::Abgr8888 | Fourcc::Xbgr8888 => {
                (ffi::RGBA8, ffi::RGBA8, false)
            }
            Fourcc::Abgr16161616f | Fourcc::Xbgr16161616f => (ffi::RGBA16F, ffi::RGBA16F, false),
            Fourcc::Abgr2101010 | Fourcc::Xbgr2101010 => (ffi::RGB10_A2, ffi::RGB10_A2, false),
            _ => return Err(GlesError::UnsupportedPixelFormat(format)),
        };
        if size.is_empty() {
            return Err(GlesError::UnsupportedPixelFormat(format));
        }
        let mut name = 0;
        unsafe {
            self.gl.GenTextures(1, &mut name);
            self.gl.BindTexture(ffi::TEXTURE_2D, name);
            self.gl.TexStorage2D(ffi::TEXTURE_2D, 1, storage, size.w, size.h);
            self.gl
                .TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_S, ffi::CLAMP_TO_EDGE as i32);
            self.gl
                .TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_T, ffi::CLAMP_TO_EDGE as i32);
            let error = self.gl.GetError();
            if error != ffi::NO_ERROR {
                self.gl.DeleteTextures(1, &name);
                return Err(GlesError::GlSync {
                    operation: "composition allocation",
                    code: error,
                });
            }
        }
        let mut texture =
            unsafe { GlesTexture::from_raw(self, Some(metadata), !has_alpha(format), name, size) };
        let inner = Arc::get_mut(&mut texture.0).expect("new native texture");
        inner.encoding = GlesTextureEncoding::ElectricalPremultiplied;
        inner.storage_is_srgb = storage_is_srgb;
        inner.is_composition = true;
        Ok(texture)
    }

    /// Reserve context-local framebuffer names at initial cold startup.
    /// Later texture admission only attaches images to these existing names.
    pub fn reserve_composition_framebuffers(&mut self, capacity: usize) -> Result<(), GlesError> {
        self.prepare_composition()?;
        let state = self.composition.as_mut().unwrap();
        if !state.attachments.is_empty() || !state.spare_framebuffers.is_empty() {
            return Err(GlesError::Composition("framebuffer inventory already reserved"));
        }
        let count: i32 = capacity
            .try_into()
            .map_err(|_| GlesError::Composition("framebuffer inventory exceeds native count"))?;
        state.attachments = Vec::with_capacity(capacity);
        state.spare_framebuffers = vec![0; capacity];
        unsafe {
            self.gl
                .GenFramebuffers(count, state.spare_framebuffers.as_mut_ptr());
            let code = self.gl.GetError();
            if code != ffi::NO_ERROR {
                return Err(GlesError::GlSync {
                    operation: "framebuffer inventory reservation",
                    code,
                });
            }
        }
        if state.spare_framebuffers.contains(&0) {
            return Err(GlesError::Composition("native framebuffer reservation failed"));
        }
        Ok(())
    }

    /// Adopt helper-allocated Rust damage backing without allocating on the
    /// owner worker. Call at a frame-resource epoch boundary.
    pub fn adopt_damage_scratch(&mut self, backing: &mut GlesPreparedDamageScratch) -> Result<(), GlesError> {
        let state = self
            .composition
            .as_mut()
            .ok_or(GlesError::Composition("composition state not initialized"))?;
        std::mem::swap(&mut state.damage, &mut backing.0);
        Ok(())
    }

    /// Admit exact sampled color attachments before rendering. A frame never
    /// creates a framebuffer or grows this inventory.
    pub fn prepare_composition_textures(&mut self, textures: &[&GlesTexture]) -> Result<(), GlesError> {
        if textures.iter().any(|texture| !texture.0.is_composition) {
            return Err(GlesError::Composition(
                "attachment lacks an explicit compositor storage role",
            ));
        }
        self.prepare_framebuffer_textures(textures)
    }

    /// Validate a complete inventory handoff without allocating or changing GL.
    pub fn can_prepare_framebuffer_textures(&self, textures: &[&GlesTexture]) -> Result<(), GlesError> {
        self.check_completion_health()?;
        let state = self
            .composition
            .as_ref()
            .ok_or(GlesError::Composition("composition state not initialized"))?;
        if textures
            .iter()
            .any(|texture| texture.0.is_external || texture.context_id() != self.context_id())
        {
            return Err(GlesError::Composition(
                "attachment is not renderable or belongs to another share group",
            ));
        }
        let missing = textures
            .iter()
            .enumerate()
            .filter(|(index, texture)| {
                state.attachment(texture).is_err()
                    && !textures[..*index]
                        .iter()
                        .any(|previous| Arc::ptr_eq(&previous.0, &texture.0))
            })
            .count();
        let available = state.spare_framebuffers.len()
            + state
                .attachments
                .iter()
                .filter(|a| a.texture.strong_count() == 0)
                .count();
        if missing > available {
            return Err(GlesError::Composition("framebuffer inventory capacity exhausted"));
        }
        Ok(())
    }

    /// Attach helper-imported renderable textures to startup-reserved FBO names.
    /// This performs no shader compilation, framebuffer creation, or Vec growth.
    pub fn prepare_framebuffer_textures(&mut self, textures: &[&GlesTexture]) -> Result<(), GlesError> {
        self.can_prepare_framebuffer_textures(textures)?;
        unsafe {
            self.egl.make_current()?;
        }
        let state = self.composition.as_mut().unwrap();
        let mut index = 0;
        while index < state.attachments.len() {
            if state.attachments[index].texture.strong_count() != 0 {
                index += 1;
            } else {
                let old = state.attachments.swap_remove(index);
                unsafe {
                    self.gl.BindFramebuffer(ffi::FRAMEBUFFER, old.framebuffer);
                    self.gl.FramebufferTexture2D(
                        ffi::FRAMEBUFFER,
                        ffi::COLOR_ATTACHMENT0,
                        ffi::TEXTURE_2D,
                        0,
                        0,
                    );
                    self.gl.BindFramebuffer(ffi::FRAMEBUFFER, 0);
                }
                state.spare_framebuffers.push(old.framebuffer);
            }
        }
        for texture in textures {
            let state = self.composition.as_mut().unwrap();
            if state.attachment(texture).is_ok() {
                continue;
            }
            let framebuffer = state
                .spare_framebuffers
                .pop()
                .ok_or(GlesError::Composition("framebuffer inventory capacity exhausted"))?;
            unsafe {
                self.gl.BindFramebuffer(ffi::FRAMEBUFFER, framebuffer);
                self.gl.FramebufferTexture2D(
                    ffi::FRAMEBUFFER,
                    ffi::COLOR_ATTACHMENT0,
                    ffi::TEXTURE_2D,
                    texture.tex_id(),
                    0,
                );
                let complete = self.gl.CheckFramebufferStatus(ffi::FRAMEBUFFER) == ffi::FRAMEBUFFER_COMPLETE;
                let code = self.gl.GetError();
                if !complete || code != ffi::NO_ERROR {
                    self.gl.FramebufferTexture2D(
                        ffi::FRAMEBUFFER,
                        ffi::COLOR_ATTACHMENT0,
                        ffi::TEXTURE_2D,
                        0,
                        0,
                    );
                }
                self.gl.BindFramebuffer(ffi::FRAMEBUFFER, 0);
                if !complete || code != ffi::NO_ERROR {
                    state.spare_framebuffers.push(framebuffer);
                    return Err(GlesError::FramebufferBindingError);
                }
            }
            state.attachments.push(PreparedAttachment {
                texture: Arc::downgrade(&texture.0),
                framebuffer,
            });
        }
        Ok(())
    }

    /// Exact requested Rust backing bytes for the bounded damage bank.
    pub fn damage_scratch_bytes(slots: usize, rectangles: usize) -> Option<usize> {
        slots.checked_mul(
            mem::size_of::<Vec<Rectangle<i32, Physical>>>()
                .checked_add(rectangles.checked_mul(mem::size_of::<Rectangle<i32, Physical>>())?)?,
        )
    }

    /// Replace the admitted FBO inventory on a cold resource epoch boundary.
    /// Remaining old users must retain their own completion and image custody.
    pub fn replace_composition_textures(&mut self, textures: &[&GlesTexture]) -> Result<(), GlesError> {
        self.check_completion_health()?;
        unsafe {
            self.egl.make_current()?;
        }
        let state = self
            .composition
            .as_mut()
            .ok_or(GlesError::Composition("composition state not initialized"))?;
        let mut index = 0;
        while index < state.attachments.len() {
            if textures
                .iter()
                .any(|texture| Arc::as_ptr(&texture.0) == state.attachments[index].texture.as_ptr())
            {
                index += 1;
            } else {
                let old = state.attachments.swap_remove(index);
                unsafe {
                    self.gl.BindFramebuffer(ffi::FRAMEBUFFER, old.framebuffer);
                    self.gl.FramebufferTexture2D(
                        ffi::FRAMEBUFFER,
                        ffi::COLOR_ATTACHMENT0,
                        ffi::TEXTURE_2D,
                        0,
                        0,
                    );
                    self.gl.BindFramebuffer(ffi::FRAMEBUFFER, 0);
                }
                state.spare_framebuffers.push(old.framebuffer);
            }
        }
        self.prepare_framebuffer_textures(textures)
    }

    /// Reserve bounded nested damage workspaces on a cold resource turn.
    pub fn prepare_damage_scratch(&mut self, slots: usize, rectangles: usize) -> Result<(), GlesError> {
        self.prepare_composition()?;
        self.adopt_damage_scratch(&mut GlesPreparedDamageScratch::new(slots, rectangles)?)
    }

    /// Prepare every exact destination in a material filter graph.
    pub fn prepare_kawase_passes(&mut self, passes: &[GlesKawasePass]) -> Result<(), GlesError> {
        for pass in passes {
            self.prepare_composition_textures(&[&pass.destination])?;
        }
        Ok(())
    }

    /// Prepare a direct first downsample destination on a cold turn.
    pub fn prepare_kawase_capture(&mut self, destination: &GlesTexture) -> Result<(), GlesError> {
        self.prepare_composition_textures(&[destination])
    }

    /// Bind an admitted texture while retaining its native custody and write
    /// guard inside the target. No framebuffer or owner container is allocated.
    pub fn bind_prepared_texture(&mut self, texture: &GlesTexture) -> Result<GlesTarget<'static>, GlesError> {
        self.check_completion_health()?;
        if texture.context_id() != self.context_id() {
            return Err(GlesError::Composition(
                "target belongs to another native share group",
            ));
        }
        unsafe {
            self.egl.make_current()?;
        }
        let fbo = self
            .composition
            .as_ref()
            .ok_or(GlesError::Composition("framebuffer inventory not initialized"))?
            .attachment(texture)?;
        let mut sync_lock = TextureSyncGuard::Owned(OwnedTextureSync::new(texture)?);
        self.native_fences.observe(sync_lock.wait_for_all(&self.gl))?;
        Ok(GlesTarget(
            GlesTargetInternal::Texture {
                texture: texture.clone(),
                sync_lock,
                fbo,
                framebuffer_sender: None,
            },
            self.framebuffer_cleanup.identity.clone(),
        ))
    }

    /// Copy a completed encoded composition into an imported scanout
    /// attachment, preserving pixels outside the exact damage rectangles.
    pub fn resolve_composition(
        &mut self,
        source: &GlesTexture,
        destination: &mut GlesTarget<'_>,
        damage: &[Rectangle<i32, Physical>],
    ) -> Result<SyncPoint, GlesError> {
        self.resolve_composition_internal(source, destination, damage.iter().copied(), false)
    }

    /// Resolve composition damage expressed in transformed output coordinates.
    ///
    /// `output_size` is the extent before `transform` maps each damage rectangle
    /// into destination storage coordinates. Mapping is streamed directly into
    /// the draw; it needs no intermediate rectangle allocation.
    pub fn resolve_composition_transformed(
        &mut self,
        source: &GlesTexture,
        destination: &mut GlesTarget<'_>,
        damage: &[Rectangle<i32, Physical>],
        transform: Transform,
        output_size: Size<i32, Physical>,
    ) -> Result<SyncPoint, GlesError> {
        self.resolve_composition_internal(
            source,
            destination,
            damage
                .iter()
                .map(|rect| transform.transform_rect_in(*rect, &output_size)),
            false,
        )
    }

    fn resolve_composition_internal(
        &mut self,
        source: &GlesTexture,
        destination: &mut GlesTarget<'_>,
        damage: impl Iterator<Item = Rectangle<i32, Physical>>,
        source_in_frame: bool,
    ) -> Result<SyncPoint, GlesError> {
        self.check_completion_health()?;
        destination.make_current(self)?;
        if !source.0.is_composition {
            return Err(GlesError::Composition(
                "terminal source is not a composition attachment",
            ));
        }
        if let GlesTargetInternal::Texture { texture, .. } = &destination.0 {
            if texture.0.is_composition || Arc::ptr_eq(&texture.0, &source.0) {
                return Err(GlesError::Composition(
                    "terminal output must be distinct encoded storage",
                ));
            }
        }
        let size: Size<i32, Physical> = (destination.size().w, destination.size().h).into();
        if source.size().w != size.w || source.size().h != size.h {
            return Err(GlesError::Composition(
                "terminal source and destination extent differ",
            ));
        }
        let constants = TexturePushConstants::default();
        self.composition_draw(
            source,
            size,
            [0.0, 0.0, size.w as f32, size.h as f32],
            &constants,
            true,
            true,
            source_in_frame,
            None,
            damage,
        )?;
        if let GlesTargetInternal::Texture { sync_lock, .. } = &mut destination.0 {
            self.native_fences.observe(sync_lock.update_write(&self.gl))?;
        }
        self.finish_native()
    }

    #[allow(clippy::too_many_arguments)]
    fn composition_draw(
        &self,
        texture: &GlesTexture,
        target_size: Size<i32, Physical>,
        destination: [f32; 4],
        constants: &TexturePushConstants,
        nearest: bool,
        replace: bool,
        source_in_frame: bool,
        prefix_weight: Option<f32>,
        damage: impl Iterator<Item = Rectangle<i32, Physical>>,
    ) -> Result<(), GlesError> {
        self.check_completion_health()?;
        if texture.context_id() != self.context_id() {
            return Err(GlesError::Composition(
                "source texture belongs to another share group",
            ));
        }
        let state = self
            .composition
            .as_ref()
            .ok_or(GlesError::Composition("composition programs were not prepared"))?;
        let program = if texture.0.is_external {
            state
                .external_program
                .ok_or(GlesError::GLExtensionNotSupported(&[
                    "GL_OES_EGL_image_external_essl3",
                ]))?
        } else {
            state.program
        };
        let sync = if source_in_frame {
            None
        } else {
            Some(
                texture
                    .0
                    .sync
                    .try_read()
                    .map_err(|_| GlesError::Composition("source texture is still bound for writing"))?,
            )
        };
        if let Some(sync) = &sync {
            self.native_fences.observe(sync.wait_for_upload(&self.gl))?;
        }
        unsafe {
            let gl = &self.gl;
            gl.UseProgram(program);
            gl.BindBuffer(ffi::UNIFORM_BUFFER, state.uniform_buffer);
            gl.BufferSubData(
                ffi::UNIFORM_BUFFER,
                0,
                mem::size_of::<TexturePushConstants>() as isize,
                (constants as *const TexturePushConstants).cast(),
            );
            gl.BindBufferBase(ffi::UNIFORM_BUFFER, 0, state.uniform_buffer);
            state.bind_frame(
                gl,
                destination,
                target_size,
                texture.0.storage_is_srgb,
                nearest,
                texture.size(),
            );
            gl.ActiveTexture(ffi::TEXTURE0);
            gl.BindTexture(
                if texture.0.is_external {
                    ffi::TEXTURE_EXTERNAL_OES
                } else {
                    ffi::TEXTURE_2D
                },
                texture.tex_id(),
            );
            gl.Viewport(0, 0, target_size.w, target_size.h);
            gl.Enable(ffi::SCISSOR_TEST);
            if replace {
                gl.Disable(ffi::BLEND);
            } else {
                gl.Enable(ffi::BLEND);
                if let Some(weight) = prefix_weight {
                    gl.BlendColor(weight, weight, weight, weight);
                    gl.BlendFunc(ffi::CONSTANT_ALPHA, ffi::ONE_MINUS_CONSTANT_ALPHA);
                } else {
                    gl.BlendFunc(ffi::ONE, ffi::ONE_MINUS_SRC_ALPHA);
                }
            }
            for rect in damage {
                gl.Scissor(rect.loc.x, rect.loc.y, rect.size.w, rect.size.h);
                gl.DrawArrays(ffi::TRIANGLE_STRIP, 0, 4);
            }
            gl.BlendFunc(ffi::ONE, ffi::ONE_MINUS_SRC_ALPHA);
            gl.BindBuffer(ffi::UNIFORM_BUFFER, 0);
            gl.BindSampler(0, 0);
            let code = gl.GetError();
            if code != ffi::NO_ERROR {
                return self.native_fences.observe(Err(GlesError::GlSync {
                    operation: "composition draw",
                    code,
                }));
            }
        }
        if let Some(sync) = &sync {
            self.native_fences.observe(sync.update_read(&self.gl))?;
        }
        Ok(())
    }
}

impl GlesFrame<'_, '_> {
    fn composition_target_size(&self) -> Size<i32, Physical> {
        let size = self
            .composition_target
            .as_ref()
            .map_or_else(|| self.target.size(), |(_, texture)| texture.size());
        (size.w, size.h).into()
    }

    fn composition_source(&self) -> Result<GlesTexture, GlesError> {
        if let Some((_, texture)) = &self.composition_target {
            return Ok(texture.clone());
        }
        match &self.target.0 {
            GlesTargetInternal::Texture { texture, .. } if texture.0.is_composition => Ok(texture.clone()),
            _ => Err(GlesError::Composition(
                "framebuffer reads require a prepared sampled composition attachment",
            )),
        }
    }

    fn bind_composition_target(&self) -> Result<(), GlesError> {
        if let Some((framebuffer, _)) = self.composition_target {
            unsafe {
                self.renderer.gl.BindFramebuffer(ffi::FRAMEBUFFER, framebuffer);
            }
        } else {
            self.target.make_current(self.renderer)?;
        }
        Ok(())
    }

    fn native_region(&self, rect: Rectangle<i32, Physical>) -> Option<Rectangle<i32, Physical>> {
        let target_size = self
            .owner_sample_replay
            .map_or(self.composition_target_size(), |r| r.frame_size);
        let mut native = self
            .transform
            .transform_rect_in(rect, &self.size)
            .intersection(Rectangle::from_size(target_size))?;
        if let Some(replay) = self.owner_sample_replay {
            native.loc += replay.translation;
            native = native.intersection(replay.bounds)?;
        }
        Some(native)
    }

    fn shift_clip(&self, mut clip: TransformedAnalyticClip) -> TransformedAnalyticClip {
        if let Some(replay) = self.owner_sample_replay {
            let shift = replay.shift(false);
            let rect = match &mut clip {
                TransformedAnalyticClip::Rounded(clip) => &mut clip.rect,
                TransformedAnalyticClip::BottomEdge { clip, .. } => &mut clip.rect,
            };
            rect.loc.x += shift[0] as f64;
            rect.loc.y += shift[1] as f64;
        }
        clip
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_composition_texture(
        &mut self,
        texture: &GlesTexture,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        transform: Transform,
        alpha: f32,
        clip: Option<AnalyticClip>,
        effect: TextureRenderEffect,
        operation: Option<MaterialOperation>,
        prefix_weight: Option<f32>,
    ) -> Result<(), GlesError> {
        self.renderer.check_completion_health()?;
        self.composition_source()?;
        if damage.is_empty() || src.size.is_empty() || dst.size.is_empty() {
            return Ok(());
        }
        let capacity = texture.size();
        if let Some(MaterialOperation::Kawase(output)) = operation {
            if output.extent.is_empty()
                || output.extent.w > capacity.w
                || output.extent.h > capacity.h
                || !output.offset.is_finite()
                || output.offset < 0.0
            {
                return Err(GlesError::Composition("invalid fused Kawase extent or offset"));
            }
        }
        if src.loc.x < 0.0
            || src.loc.y < 0.0
            || src.loc.x + src.size.w > capacity.w as f64
            || src.loc.y + src.size.h > capacity.h as f64
        {
            return Err(GlesError::Composition(
                "source rectangle exceeds initialized texture capacity",
            ));
        }
        let mut native = self.transform.transform_rect_in(dst, &self.size).to_f64();
        if let Some(replay) = self.owner_sample_replay {
            let shift = replay.shift(self.resolved_sample_texture);
            native.loc.x += shift[0] as f64;
            native.loc.y += shift[1] as f64;
        }
        let framebuffer_coordinates = operation.is_some() || prefix_weight.is_some();
        let mut constants = TexturePushConstants::new(
            alpha * self.draw_alpha,
            TextureTransform::from(if framebuffer_coordinates {
                Transform::Normal
            } else {
                combine_image_transform(transform, self.transform)
            }),
            texture.0.y_inverted,
        )
        .with_src_rect(
            [
                src.loc.x as f32 / capacity.w as f32,
                src.loc.y as f32 / capacity.h as f32,
            ],
            [
                src.size.w as f32 / capacity.w as f32,
                src.size.h as f32 / capacity.h as f32,
            ],
        )
        .with_effect(effect);
        if let Some(clip) =
            clip.map(|c| self.shift_clip(transform_analytic_clip(self.transform, self.size, c)))
        {
            constants = match clip {
                TransformedAnalyticClip::Rounded(c) => constants.with_rounded_clip(
                    c.corner_mask,
                    analytic_clip_rect_push_constant(c.rect),
                    [c.radius.max(0.0), c.exponent.max(2.0), c.aa_width.max(0.001), 0.0],
                ),
                TransformedAnalyticClip::BottomEdge { clip: c, transform } => constants
                    .with_bottom_edge_clip(
                        transform,
                        analytic_clip_rect_push_constant(c.rect),
                        [
                            c.content_width.max(0.0),
                            c.progress.max(0.0),
                            c.edge_height.max(0.0),
                            c.plateau_inset.max(0.0),
                        ],
                        c.geometry_scale.max(0.001),
                    ),
            };
        }
        if let Some(clip) = self.outer_rounded_clip.filter(|_| prefix_weight.is_none()) {
            if let TransformedAnalyticClip::Rounded(c) = self.shift_clip(transform_analytic_clip(
                self.transform,
                self.size,
                AnalyticClip::Rounded(clip),
            )) {
                constants = constants.with_outer_rounded_clip(
                    [c.rect.loc.x as f32, c.rect.loc.y as f32],
                    [c.rect.size.w as f32, c.rect.size.h as f32],
                    c.radius,
                );
            }
        }
        if let Some(operation) = operation {
            constants = match operation {
                MaterialOperation::Copy => constants,
                MaterialOperation::Kawase(output) => constants.with_kawase_output(output, capacity),
                MaterialOperation::Tint(tint) => constants.with_material_tone(tint),
                MaterialOperation::OwnerResolve(extent) => constants.with_owner_resolve(extent, capacity),
                MaterialOperation::Solid(color) => {
                    constants.effect = [8.0, color.r(), color.g(), color.b()];
                    constants.effect_params = [color.a(), 0.0, 0.0, 0.0];
                    constants
                }
            };
        }
        if self.owner_sample_replay.is_some() {
            constants.rounded_clip_flags |= 1 << 30;
        }
        self.bind_composition_target()?;
        let nearest = self.resolved_sample_texture
            || matches!(operation, Some(MaterialOperation::OwnerResolve(_)))
            || (operation.is_none()
                && src.size.w == dst.size.w as f64
                && src.size.h == dst.size.h as f64
                && src.loc.x.fract() == 0.0
                && src.loc.y.fract() == 0.0);
        let damage = damage.iter().filter_map(|rect| {
            Rectangle::new(dst.loc + rect.loc, rect.size)
                .intersection(dst)
                .and_then(|rect| self.native_region(rect))
        });
        self.renderer.composition_draw(
            texture,
            self.composition_target_size(),
            analytic_clip_rect_push_constant(native),
            &constants,
            nearest,
            false,
            false,
            prefix_weight,
            damage,
        )
    }

    /// Borrow bounded damage storage admitted before rendering.
    pub fn with_damage_scratch<T>(
        &mut self,
        required: usize,
        draw: impl FnOnce(&mut Self, &mut Vec<Rectangle<i32, Physical>>) -> Result<T, GlesError>,
    ) -> Result<T, GlesError> {
        let state = self
            .renderer
            .composition
            .as_mut()
            .ok_or(GlesError::Composition("composition was not prepared"))?;
        let mut scratch = state.damage.pop().ok_or(GlesError::Composition(
            "nested damage workspace capacity exhausted",
        ))?;
        if scratch.capacity() < required {
            state.damage.push(scratch);
            return Err(GlesError::Composition("damage workspace capacity exhausted"));
        }
        scratch.clear();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| draw(self, &mut scratch)));
        scratch.clear();
        self.renderer.composition.as_mut().unwrap().damage.push(scratch);
        match result {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }

    /// Jointly evaluate the owner clip with inner coverage at canonical samples.
    pub fn with_outer_rounded_clip<T>(
        &mut self,
        clip: Option<RoundedClip>,
        draw: impl FnOnce(&mut Self) -> Result<T, GlesError>,
    ) -> Result<T, GlesError> {
        let previous = self.outer_rounded_clip;
        if previous.is_some() && clip.is_some() && previous != clip {
            return Err(GlesError::Composition(
                "nested distinct owner clips exceed admitted packet",
            ));
        }
        self.outer_rounded_clip = clip.or(previous);
        let result = draw(self);
        self.outer_rounded_clip = previous;
        result
    }

    /// Apply group alpha to disjoint source-over primitives.
    pub fn with_draw_alpha<T>(
        &mut self,
        alpha: f32,
        draw: impl FnOnce(&mut Self) -> Result<T, GlesError>,
    ) -> Result<T, GlesError> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(GlesError::Composition("invalid draw opacity"));
        }
        let previous = self.draw_alpha;
        self.draw_alpha *= alpha;
        let result = draw(self);
        self.draw_alpha = previous;
        result
    }
}

#[derive(Clone, Copy)]
pub(super) enum MaterialOperation {
    Copy,
    Kawase(GlesKawaseOutput),
    Tint(GlesMaterialTint),
    OwnerResolve(Size<i32, Physical>),
    Solid(Color32F),
}

include!("composition_material.rs");

impl GlesFrame<'_, '_> {
    pub(super) fn clear_composition(
        &mut self,
        color: Color32F,
        damage: &[Rectangle<i32, Physical>],
    ) -> Result<(), GlesError> {
        self.composition_source()?;
        self.bind_composition_target()?;
        unsafe {
            let gl = &self.renderer.gl;
            gl.Enable(ffi::SCISSOR_TEST);
            gl.ClearColor(color.r(), color.g(), color.b(), color.a());
            for rect in damage.iter().filter_map(|r| self.native_region(*r)) {
                gl.Scissor(rect.loc.x, rect.loc.y, rect.size.w, rect.size.h);
                gl.Clear(ffi::COLOR_BUFFER_BIT);
            }
            let code = gl.GetError();
            if code != ffi::NO_ERROR {
                return self.renderer.native_fences.observe(Err(GlesError::GlSync {
                    operation: "composition clear",
                    code,
                }));
            }
        }
        Ok(())
    }

    pub(super) fn solid_composition(
        &mut self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        color: Color32F,
    ) -> Result<(), GlesError> {
        let texture = self
            .renderer
            .composition
            .as_ref()
            .ok_or(GlesError::Composition("composition was not prepared"))?
            .white
            .clone();
        let alpha = self.draw_alpha;
        self.draw_alpha = 1.0;
        let result = self.draw_composition_texture(
            &texture,
            Rectangle::from_size((1.0, 1.0).into()),
            dst,
            damage,
            self.transform,
            1.0,
            None,
            TextureRenderEffect::NONE,
            Some(MaterialOperation::Solid(color * alpha)),
            None,
        );
        self.draw_alpha = alpha;
        result
    }
}

impl GlesFrame<'_, '_> {
    /// Finish a sampled sRGB composition and encode it into the selected output.
    /// The returned final native fence covers all scene reads and output writes.
    pub fn finish_into(
        self,
        destination: &mut GlesTarget<'_>,
        damage: &[Rectangle<i32, Physical>],
    ) -> Result<SyncPoint, GlesError> {
        self.finish_into_internal(destination, damage.iter().copied())
    }

    /// Finish and encode damage mapped from output coordinates into storage.
    /// The mapping streams directly into the terminal draw without allocation.
    pub fn finish_into_transformed(
        self,
        destination: &mut GlesTarget<'_>,
        damage: &[Rectangle<i32, Physical>],
        transform: Transform,
        output_size: Size<i32, Physical>,
    ) -> Result<SyncPoint, GlesError> {
        self.finish_into_internal(
            destination,
            damage
                .iter()
                .map(|rect| transform.transform_rect_in(*rect, &output_size)),
        )
    }

    fn finish_into_internal(
        mut self,
        destination: &mut GlesTarget<'_>,
        damage: impl Iterator<Item = Rectangle<i32, Physical>>,
    ) -> Result<SyncPoint, GlesError> {
        let source = self.composition_source()?;
        self.finish_drawing()?;
        let result = self
            .renderer
            .resolve_composition_internal(&source, destination, damage, true);
        if result.is_ok() {
            if let GlesTargetInternal::Texture { sync_lock, .. } = &mut self.target.0 {
                self.renderer
                    .native_fences
                    .observe(sync_lock.update_read(&self.renderer.gl))?;
            }
        } else {
            self.renderer.native_fences.poison();
        }
        result
    }
}
