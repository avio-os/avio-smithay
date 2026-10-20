use ffi::Gles2;

use super::*;
use std::sync::Arc;

/// Meaning of the stored sRGB channel values before composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlesTextureEncoding {
    /// Client/shell channels were premultiplied in electrical sRGB.
    ElectricalPremultiplied,
    /// Compositor channels were premultiplied in linear light before encoding.
    LinearPremultiplied,
}

/// A handle to a GLES texture
///
/// The texture can be used with the same [`GlesRenderer`] it was created with, or one using a
/// shared [`EGLContext`].
#[derive(Debug, Clone)]
pub struct GlesTexture(pub(super) Arc<GlesTextureInternal>);

impl GlesTexture {
    /// Create a GlesTexture from a raw gl texture id.
    ///
    /// It is expected to not be external or y_inverted.
    ///
    /// Ownership over the texture is taken by the renderer, you should not free the texture yourself.
    ///
    /// # Safety
    ///
    /// The renderer cannot make sure `tex` is a valid texture id.
    pub unsafe fn from_raw(
        renderer: &GlesRenderer,
        internal_format: Option<ffi::types::GLenum>,
        opaque: bool,
        tex: ffi::types::GLuint,
        size: Size<i32, BufferCoord>,
    ) -> GlesTexture {
        GlesTexture(Arc::new(GlesTextureInternal {
            context: renderer.context_id(),
            texture: tex,
            sync: RwLock::default(),
            format: internal_format,
            has_alpha: !opaque,
            is_external: false,
            y_inverted: false,
            size,
            retirement: renderer.origin.as_ref().map(GlesDeviceOrigin::retirement_slot),
            encoding: GlesTextureEncoding::ElectricalPremultiplied,
            storage_is_srgb: false,
            is_composition: false,
            egl_images: None,
            destruction_callback_sender: renderer.gles_cleanup().sender.clone(),
        }))
    }

    /// Take ownership of an existing `GL_TEXTURE_EXTERNAL_OES` texture.
    ///
    /// Its pixels are client electrical-premultiplied values. Imported
    /// allocation custody must outlive every reader's native completion.
    ///
    /// # Safety
    ///
    /// `tex` must be a valid external texture in this renderer's share group,
    /// with the stated extent and alpha convention. Ownership transfers to
    /// this handle; no caller may delete it or change its attached image.
    pub unsafe fn from_raw_external(
        renderer: &GlesRenderer,
        opaque: bool,
        tex: ffi::types::GLuint,
        size: Size<i32, BufferCoord>,
        y_inverted: bool,
    ) -> GlesTexture {
        let mut texture = unsafe { Self::from_raw(renderer, None, opaque, tex, size) };
        let internal = Arc::get_mut(&mut texture.0).expect("new external texture");
        internal.is_external = true;
        internal.y_inverted = y_inverted;
        texture
    }

    /// Identity of the exact native share group that owns this texture.
    pub fn context_id(&self) -> ContextId<GlesTexture> {
        self.0.context.clone()
    }

    /// OpenGL texture id of this texture
    ///
    /// This id will become invalid, when the GlesTexture is dropped and does not transfer ownership.
    pub fn tex_id(&self) -> ffi::types::GLuint {
        self.0.texture
    }

    /// Whether the texture is upside down
    pub fn is_y_inverted(&self) -> bool {
        self.0.y_inverted
    }

    /// Whether this handle has no other strong native texture readers.
    /// Weak FBO inventory identities do not count. Reuse additionally requires
    /// the caller's complete GPU reader fences and external allocation custody.
    pub fn is_externally_idle(&self) -> bool {
        Arc::strong_count(&self.0) == 1
    }

    /// Whether this is the only reference to this texture (strong or weak)
    ///
    /// Note that this tracks only references to this Smithay object (that you can get by cloning
    /// it). If you make a reference via OpenGL directly somehow, you need to keep track of it on
    /// your own.
    pub fn is_unique_reference(&mut self) -> bool {
        Arc::get_mut(&mut self.0).is_some()
    }
}

#[derive(Debug, Default)]
pub(super) struct TextureSync {
    pub(super) read_sync: Mutex<Option<ffi::types::GLsync>>,
    pub(super) write_sync: Mutex<Option<ffi::types::GLsync>>,
}

unsafe fn wait_for_syncpoint(sync: &mut Option<ffi::types::GLsync>, gl: &Gles2) -> Result<(), GlesError> {
    if let Some(sync_obj) = *sync {
        match gl.ClientWaitSync(sync_obj, 0, 0) {
            ffi::ALREADY_SIGNALED | ffi::CONDITION_SATISFIED => {
                let _ = sync.take();
                gl.DeleteSync(sync_obj);
            }
            ffi::TIMEOUT_EXPIRED => {
                gl.WaitSync(sync_obj, 0, ffi::TIMEOUT_IGNORED);
                let code = gl.GetError();
                if code != ffi::NO_ERROR {
                    return Err(GlesError::GlSync {
                        operation: "texture server wait",
                        code,
                    });
                }
            }
            _ => {
                return Err(GlesError::GlSync {
                    operation: "texture readiness query",
                    code: gl.GetError(),
                })
            }
        };
    }
    Ok(())
}

impl TextureSync {
    pub(super) fn wait_for_upload(&self, gl: &Gles2) -> Result<(), GlesError> {
        unsafe { wait_for_syncpoint(&mut self.write_sync.lock().unwrap(), gl) }
    }

    pub(super) fn update_read(&self, gl: &Gles2) -> Result<(), GlesError> {
        let mut read_sync = self.read_sync.lock().unwrap();
        replace_syncpoint(&mut read_sync, gl)
    }

    pub(super) fn wait_for_all(&mut self, gl: &Gles2) -> Result<(), GlesError> {
        unsafe {
            wait_for_syncpoint(self.read_sync.get_mut().unwrap(), gl)?;
            wait_for_syncpoint(self.write_sync.get_mut().unwrap(), gl)
        }
    }

    pub(super) fn update_write(&mut self, gl: &Gles2) -> Result<(), GlesError> {
        replace_syncpoint(self.write_sync.get_mut().unwrap(), gl)
    }
}

fn replace_syncpoint(sync: &mut Option<ffi::types::GLsync>, gl: &Gles2) -> Result<(), GlesError> {
    // Retain the old dependency until both its wait and the replacement fence
    // succeeded. A null fence must never become an apparently valid receipt.
    unsafe {
        if let Some(old) = *sync {
            gl.WaitSync(old, 0, ffi::TIMEOUT_IGNORED);
            let code = gl.GetError();
            if code != ffi::NO_ERROR {
                return Err(GlesError::GlSync {
                    operation: "texture dependency wait",
                    code,
                });
            }
        }
        let next = gl.FenceSync(ffi::SYNC_GPU_COMMANDS_COMPLETE, 0);
        if next.is_null() {
            return Err(GlesError::GlSync {
                operation: "texture completion fence",
                code: gl.GetError(),
            });
        }
        if let Some(old) = sync.replace(next) {
            gl.DeleteSync(old);
        }
        // Shared-context consumers must be able to make forward progress.
        gl.Flush();
    }
    Ok(())
}

#[derive(Debug)]
pub(super) struct GlesTextureInternal {
    pub(super) context: ContextId<GlesTexture>,
    pub(super) encoding: GlesTextureEncoding,
    pub(super) storage_is_srgb: bool,
    pub(super) is_composition: bool,
    pub(super) retirement: Option<GlesRetirementSlot<GlesSharedCleanup>>,
    pub(super) texture: ffi::types::GLuint,
    pub(super) sync: RwLock<TextureSync>,
    pub(super) format: Option<ffi::types::GLenum>,
    pub(super) has_alpha: bool,
    pub(super) is_external: bool,
    pub(super) y_inverted: bool,
    pub(super) size: Size<i32, BufferCoord>,
    pub(super) egl_images: Option<Vec<EGLImage>>,
    pub(super) destruction_callback_sender: Sender<CleanupResource>,
}
unsafe impl Send for GlesTextureInternal {}
unsafe impl Sync for GlesTextureInternal {}

impl Drop for GlesTextureInternal {
    fn drop(&mut self) {
        // The final Arc owns this lock exclusively: no lock acquisition, driver
        // call, or native cleanup queue allocation runs on the dropping thread.
        let poisoned = self.sync.is_poisoned()
            || self
                .sync
                .get_mut()
                .map(|sync| sync.read_sync.is_poisoned() || sync.write_sync.is_poisoned())
                .unwrap_or(true);
        if poisoned {
            if let Some(retirement) = self.retirement.take() {
                retirement.quarantine();
            }
            // Neither recovering a poisoned guard nor unwinding is GPU
            // completion. Leave every native handle untouched, including EGL
            // image siblings, and never cause a second panic during final Drop.
            if let Some(images) = self.egl_images.take() {
                std::mem::forget(images);
            }
            return;
        }
        let sync = self.sync.get_mut().expect("exclusive unpoisoned texture");
        let cleanup = GlesSharedCleanup {
            resources: [
                Some(CleanupResource::Texture(self.texture)),
                sync.read_sync
                    .get_mut()
                    .unwrap()
                    .take()
                    .map(CleanupResource::Sync),
                sync.write_sync
                    .get_mut()
                    .unwrap()
                    .take()
                    .map(CleanupResource::Sync),
            ],
            images: self.egl_images.take(),
            sender: self.destruction_callback_sender.clone(),
        };
        if let Some(retirement) = self.retirement.take() {
            retirement.retire(cleanup);
        } else {
            // Legacy renderers retain their existing shared cleanup queue.
            drop(cleanup);
        }
    }
}

impl Texture for GlesTexture {
    fn width(&self) -> u32 {
        self.0.size.w as u32
    }
    fn height(&self) -> u32 {
        self.0.size.h as u32
    }
    fn size(&self) -> Size<i32, BufferCoord> {
        self.0.size
    }
    fn format(&self) -> Option<Fourcc> {
        let fmt = gl_internal_format_to_fourcc(self.0.format?);
        if self.0.has_alpha {
            fmt
        } else {
            fmt.and_then(get_opaque)
        }
    }
}

/// Texture mapping of a GLES2 texture
#[derive(Debug)]
pub struct GlesMapping {
    pub(super) context: ContextId<GlesTexture>,
    pub(super) pbo: ffi::types::GLuint,
    pub(super) format: ffi::types::GLenum,
    pub(super) layout: ffi::types::GLenum,
    pub(super) has_alpha: bool,
    pub(super) size: Size<i32, BufferCoord>,
    pub(super) mapping: AtomicPtr<std::ffi::c_void>,
    pub(super) destruction_callback_sender: Sender<CleanupResource>,
    pub(super) completion: Option<SyncPoint>,
    pub(super) retirement: Option<GlesRetirementSlot<GlesSharedCleanup>>,
}

impl GlesMapping {
    /// Actual native completion of the GPU readback, when produced by a strict
    /// renderer. Callers retain external source custody through this receipt.
    pub fn completion(&self) -> Option<&SyncPoint> {
        self.completion.as_ref()
    }
}

impl Texture for GlesMapping {
    fn width(&self) -> u32 {
        self.size.w as u32
    }
    fn height(&self) -> u32 {
        self.size.h as u32
    }
    fn size(&self) -> Size<i32, BufferCoord> {
        self.size
    }
    fn format(&self) -> Option<Fourcc> {
        let fmt = gl_read_format_to_fourcc(self.format, self.layout);
        if self.has_alpha {
            fmt
        } else {
            fmt.and_then(get_opaque)
        }
    }
}
impl TextureMapping for GlesMapping {
    fn flipped(&self) -> bool {
        true
    }
    fn format(&self) -> Fourcc {
        Texture::format(self).expect("Should never happen")
    }
}

impl Drop for GlesMapping {
    fn drop(&mut self) {
        let cleanup = GlesSharedCleanup {
            resources: [
                Some(CleanupResource::Mapping(
                    self.pbo,
                    self.mapping.load(Ordering::SeqCst),
                )),
                None,
                None,
            ],
            images: None,
            sender: self.destruction_callback_sender.clone(),
        };
        if let Some(retirement) = self.retirement.take() {
            retirement.retire(cleanup);
        } else {
            drop(cleanup);
        }
    }
}

/// Owns the Arc which pins the lock for the guard's entire lifetime. The guard
/// remains !Send and is explicitly released before dropping that Arc.
#[derive(Debug)]
pub(super) struct OwnedTextureSync {
    guard: std::mem::ManuallyDrop<RwLockWriteGuard<'static, TextureSync>>,
    _texture: GlesTexture,
}
impl OwnedTextureSync {
    pub(super) fn new(texture: &GlesTexture) -> Result<Self, GlesError> {
        let texture = texture.clone();
        // SAFETY: Arc's inner allocation never moves. `_texture` retains that
        // exact allocation until Drop explicitly releases `guard` first. No
        // reference to this extended lifetime escapes the owning wrapper.
        let lock: &'static RwLock<TextureSync> = unsafe { &(*Arc::as_ptr(&texture.0)).sync };
        let guard = lock
            .try_write()
            .map_err(|_| GlesError::Composition("texture is already bound or sampled"))?;
        Ok(Self {
            guard: std::mem::ManuallyDrop::new(guard),
            _texture: texture,
        })
    }
}
impl Drop for OwnedTextureSync {
    fn drop(&mut self) {
        // SAFETY: The guard is initialized exactly once and is not otherwise
        // moved or dropped. The Arc keeping the lock live is still retained.
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.guard);
        }
    }
}
#[derive(Debug)]
pub(super) enum TextureSyncGuard<'a> {
    Borrowed(RwLockWriteGuard<'a, TextureSync>),
    Owned(OwnedTextureSync),
}
impl std::ops::Deref for TextureSyncGuard<'_> {
    type Target = TextureSync;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(guard) => guard,
            Self::Owned(guard) => &guard.guard,
        }
    }
}
impl std::ops::DerefMut for TextureSyncGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Borrowed(guard) => guard,
            Self::Owned(guard) => &mut guard.guard,
        }
    }
}
