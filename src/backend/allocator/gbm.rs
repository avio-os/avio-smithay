//! Module for Buffers created using [libgbm](gbm).
//!
//! The re-exported [`GbmDevice`](gbm::Device) implements the [`Allocator`] trait
//! and [`GbmBuffer`](gbm::BufferObject) satisfies the [`Buffer`] trait while also allowing
//! conversions to and from [dmabufs](super::dmabuf).

use super::{
    dmabuf::{AsDmabuf, Dmabuf, DmabufFlags, MAX_PLANES},
    Allocator, Buffer, Format, Fourcc, Modifier,
};
use crate::backend::drm::DrmNode;
use crate::utils::{Buffer as BufferCoords, Size};
use drm::buffer::PlanarBuffer;
use gbm::BufferObject;
pub use gbm::{BufferObjectFlags as GbmBufferFlags, Device as GbmDevice};
use std::os::unix::io::{AsFd, BorrowedFd};
use tracing::instrument;

/// A GBM buffer object
#[derive(Debug)]
pub struct GbmBuffer {
    drm_node: Option<DrmNode>,
    bo: BufferObject<()>,
    size: Size<i32, BufferCoords>,
    format: Format,
}

impl PlanarBuffer for GbmBuffer {
    #[inline]
    fn size(&self) -> (u32, u32) {
        (self.size.w as u32, self.size.h as u32)
    }

    #[inline]
    fn format(&self) -> Fourcc {
        self.format.code
    }

    #[inline]
    fn modifier(&self) -> Option<Modifier> {
        match self.format.modifier {
            Modifier::Invalid => None,
            x => Some(x),
        }
    }

    #[inline]
    fn pitches(&self) -> [u32; 4] {
        self.bo.pitches()
    }

    #[inline]
    fn handles(&self) -> [Option<drm::buffer::Handle>; 4] {
        self.bo.handles()
    }

    #[inline]
    fn offsets(&self) -> [u32; 4] {
        self.bo.offsets()
    }
}

impl drm::buffer::Buffer for GbmBuffer {
    #[inline]
    fn size(&self) -> (u32, u32) {
        (self.size.w as u32, self.size.h as u32)
    }

    #[inline]
    fn format(&self) -> Fourcc {
        self.format.code
    }

    #[inline]
    fn pitch(&self) -> u32 {
        self.bo.pitch()
    }

    #[inline]
    fn handle(&self) -> drm::buffer::Handle {
        drm::buffer::Buffer::handle(&self.bo)
    }
}

impl GbmBuffer {
    /// Create a [`GbmBuffer`] from an existing [`BufferObject`]
    ///
    /// `implicit` forces the object to assume the modifier is `Invalid` for cases,
    /// where the buffer was allocated with an older api, that doesn't support modifiers.
    ///
    /// Gbm might otherwise give us the underlying or a non-sensical modifier,
    /// which can fail in various other apis.
    pub fn from_bo(bo: BufferObject<()>, implicit: bool) -> Self {
        let drm_node = DrmNode::from_file(bo.device_fd()).ok();
        Self::from_bo_with_node(bo, implicit, drm_node)
    }

    /// Create a [`GbmBuffer`] from an existing [`BufferObject`] explicitly defining the device node
    ///
    /// `implicit` forces the object to assume the modifier is `Invalid` for cases,
    /// where the buffer was allocated with an older api, that doesn't support modifiers.
    ///
    /// Gbm might otherwise give us the underlying or a non-sensical modifier,
    /// which can fail in various other apis.
    pub fn from_bo_with_node(bo: BufferObject<()>, implicit: bool, drm_node: Option<DrmNode>) -> Self {
        let modifier = if implicit {
            Modifier::Invalid
        } else {
            bo.modifier()
        };
        Self::from_bo_with_explicit_modifier(bo, drm_node, modifier)
    }

    /// Create a [`GbmBuffer`] from an existing [`BufferObject`] with the modifier
    /// the caller asserts the bytes-on-disk layout actually conforms to.
    ///
    /// This exists for the NVIDIA-proprietary case where `gbm_bo_create()` was
    /// invoked via the legacy entrypoint with `GBM_BO_USE_LINEAR`: the driver
    /// honors the flag at the byte-layout level but still reports
    /// `Modifier::Invalid` from `gbm_bo_get_modifier()`. Any importer that
    /// trusts the reported modifier will misinterpret the buffer; using this
    /// constructor lets the allocator pin the *requested* modifier (e.g.
    /// `Modifier::Linear`) onto the resulting `GbmBuffer` so downstream
    /// consumers see what the producer actually intended.
    ///
    /// See KWin `src/core/gbmgraphicsbufferallocator.cpp:159–205`,
    /// Mozilla bug 1688851, and xdpw's `force_mod_linear` for the same
    /// pattern in other Wayland producers.
    pub fn from_bo_with_explicit_modifier(
        bo: BufferObject<()>,
        drm_node: Option<DrmNode>,
        modifier: Modifier,
    ) -> Self {
        let size = (bo.width() as i32, bo.height() as i32).into();
        let format = Format {
            code: bo.format(),
            modifier,
        };
        Self {
            drm_node,
            bo,
            size,
            format,
        }
    }

    /// Get the [`DrmNode`] of the device the buffer was allocated when available
    pub fn device_node(&self) -> Option<DrmNode> {
        self.drm_node
    }
}

impl std::ops::Deref for GbmBuffer {
    type Target = BufferObject<()>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.bo
    }
}

impl std::ops::DerefMut for GbmBuffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.bo
    }
}

/// Light wrapper around an [`GbmDevice`] to implement the [`Allocator`]-trait
#[derive(Clone, Debug)]
pub struct GbmAllocator<A: AsFd + 'static> {
    drm_node: Option<DrmNode>,
    device: GbmDevice<A>,
    default_flags: GbmBufferFlags,
}

impl<A: AsFd + 'static> AsRef<GbmDevice<A>> for GbmAllocator<A> {
    #[inline]
    fn as_ref(&self) -> &GbmDevice<A> {
        &self.device
    }
}

impl<A: AsFd + 'static> AsMut<GbmDevice<A>> for GbmAllocator<A> {
    #[inline]
    fn as_mut(&mut self) -> &mut GbmDevice<A> {
        &mut self.device
    }
}

impl<A: AsFd + 'static> AsFd for GbmAllocator<A> {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.device.as_fd()
    }
}

impl<A: AsFd + 'static> GbmAllocator<A> {
    /// Create a new [`GbmAllocator`] from a [`GbmDevice`] with some default usage flags,
    /// to be used when [`Allocator::create_buffer`] is invoked.
    pub fn new(device: GbmDevice<A>, default_flags: GbmBufferFlags) -> GbmAllocator<A> {
        let drm_node = DrmNode::from_file(device.as_fd()).ok();
        GbmAllocator {
            drm_node,
            device,
            default_flags,
        }
    }

    /// Alternative to [`Allocator::create_buffer`], if you need a one-off buffer with
    /// a different set of usage flags.
    #[instrument(level = "trace", skip(self), fields(self.device = ?self.device, err))]
    #[profiling::function]
    pub fn create_buffer_with_flags(
        &mut self,
        width: u32,
        height: u32,
        fourcc: Fourcc,
        modifiers: &[Modifier],
        flags: GbmBufferFlags,
    ) -> Result<GbmBuffer, std::io::Error> {
        #[cfg(feature = "backend_gbm_has_create_with_modifiers2")]
        let result = self
            .device
            .create_buffer_object_with_modifiers2(width, height, fourcc, modifiers.iter().copied(), flags)
            .map_err(std::io::Error::from)
            .and_then(|bo| Self::wrap_with_modifiers_result(bo, modifiers, self.drm_node));

        #[cfg(not(feature = "backend_gbm_has_create_with_modifiers2"))]
        let result = if (flags & !(GbmBufferFlags::SCANOUT | GbmBufferFlags::RENDERING)).is_empty() {
            self.device
                .create_buffer_object_with_modifiers(width, height, fourcc, modifiers.iter().copied())
                .map_err(std::io::Error::from)
                .and_then(|bo| Self::wrap_with_modifiers_result(bo, modifiers, self.drm_node))
        } else if modifiers.contains(&Modifier::Invalid) || modifiers.contains(&Modifier::Linear) {
            return Self::create_legacy_with_linear_quirk(
                &self.device,
                width,
                height,
                fourcc,
                modifiers,
                flags,
                self.drm_node,
            );
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "unsupported combination of flags and modifiers",
            ));
        };

        match result {
            Ok(bo) => Ok(bo),
            Err(err) => {
                if modifiers.contains(&Modifier::Invalid) || modifiers.contains(&Modifier::Linear) {
                    Self::create_legacy_with_linear_quirk(
                        &self.device,
                        width,
                        height,
                        fourcc,
                        modifiers,
                        flags,
                        self.drm_node,
                    )
                } else {
                    Err(err)
                }
            }
        }
    }

    /// Disposition of the modern `gbm_bo_create_with_modifiers2` result.
    ///
    /// On NVIDIA proprietary, requesting `[Linear]` succeeds but the
    /// returned bo reports `Modifier::Invalid`. The actual byte layout in
    /// that case is *not* linear — it's whatever NVIDIA picked — so
    /// trusting the bo (or trusting the caller's request) would mislead
    /// downstream consumers either way. We discard such a bo and let the
    /// caller fall through to the legacy `gbm_bo_create` path with
    /// `GBM_BO_USE_LINEAR` injected, which is the only entrypoint NVIDIA
    /// honors at the bytes-on-disk level.
    fn wrap_with_modifiers_result(
        bo: BufferObject<()>,
        requested: &[Modifier],
        drm_node: Option<DrmNode>,
    ) -> Result<GbmBuffer, std::io::Error> {
        // If the only thing we asked for was `Linear` and the driver
        // returned `Invalid`, this is the NVIDIA quirk. Force a fall
        // through to the legacy path by returning an error.
        let actual = bo.modifier();
        if requested == [Modifier::Linear] && actual == Modifier::Invalid {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "driver returned Modifier::Invalid for explicit Linear request; fall back to legacy gbm_bo_create + GBM_BO_USE_LINEAR",
            ));
        }
        Ok(GbmBuffer::from_bo_with_node(bo, false, drm_node))
    }

    /// Legacy `gbm_bo_create` path with the KWin/Mozilla NVIDIA workaround.
    ///
    /// When the caller asked for *only* `Modifier::Linear` we inject
    /// `GbmBufferFlags::LINEAR` into the legacy create call. NVIDIA honors
    /// that flag at the bytes-on-disk level even though
    /// `gbm_bo_get_modifier()` will still return `Modifier::Invalid` on the
    /// resulting bo. We therefore stamp the wrapping `GbmBuffer` with the
    /// modifier we *actually* produced (Linear) so downstream consumers
    /// don't misinterpret it.
    ///
    /// For all other fallback cases (caller included `Invalid`, or
    /// `[Linear, Invalid]`, or similar) we keep the historical "treat as
    /// implicit modifier" behavior — the consumer is opting into "whatever
    /// the driver picks."
    fn create_legacy_with_linear_quirk(
        device: &GbmDevice<A>,
        width: u32,
        height: u32,
        fourcc: Fourcc,
        modifiers: &[Modifier],
        flags: GbmBufferFlags,
        drm_node: Option<DrmNode>,
    ) -> Result<GbmBuffer, std::io::Error> {
        let force_linear = modifiers == [Modifier::Linear];
        if !force_linear {
            return device
                .create_buffer_object(width, height, fourcc, flags)
                .map(|bo| GbmBuffer::from_bo_with_node(bo, true, drm_node));
        }

        // KWin / xdpw flag-combination ladder for the LINEAR-on-NVIDIA path.
        //
        // KWin (`gbmgraphicsbufferallocator.cpp:178–203`) uses
        // `SCANOUT | RENDERING | LINEAR` as its primary attempt. NVIDIA's
        // proprietary GBM, in some driver versions, refuses
        // `RENDERING | LINEAR` alone (returns EINVAL) — apparently because
        // the linear-render-target path requires the buffer to also be
        // scanout-capable for layout selection. xdpw historically used
        // `RENDERING | LINEAR` and that worked on older drivers; newer
        // NVIDIA needs SCANOUT included even for non-scanout consumers.
        //
        // We try in descending preference:
        //   1. caller_flags | SCANOUT | LINEAR — KWin's choice, broadest
        //      acceptance on NVIDIA. SCANOUT does not prevent the buffer
        //      from being used as a render target or being read by another
        //      process; it just hints layout.
        //   2. caller_flags | LINEAR — xdpw's choice, works on Mesa drivers
        //      and older NVIDIA. Smaller alignment requirements.
        //   3. just LINEAR — last-resort minimum.
        let attempts: [GbmBufferFlags; 3] = [
            flags | GbmBufferFlags::SCANOUT | GbmBufferFlags::LINEAR,
            flags | GbmBufferFlags::LINEAR,
            GbmBufferFlags::LINEAR,
        ];
        let mut last_err: Option<std::io::Error> = None;
        for attempt_flags in attempts {
            match device.create_buffer_object(width, height, fourcc, attempt_flags) {
                Ok(bo) => {
                    return Ok(GbmBuffer::from_bo_with_explicit_modifier(
                        bo,
                        drm_node,
                        Modifier::Linear,
                    ));
                }
                Err(err) => {
                    last_err = Some(err.into());
                }
            }
        }
        Err(last_err.unwrap_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                "no LINEAR allocation flag combination succeeded",
            )
        }))
    }
}

impl<A: AsFd + 'static> Allocator for GbmAllocator<A> {
    type Buffer = GbmBuffer;
    type Error = std::io::Error;

    #[profiling::function]
    fn create_buffer(
        &mut self,
        width: u32,
        height: u32,
        fourcc: Fourcc,
        modifiers: &[Modifier],
    ) -> Result<GbmBuffer, Self::Error> {
        self.create_buffer_with_flags(width, height, fourcc, modifiers, self.default_flags)
    }
}

impl Buffer for GbmBuffer {
    #[inline]
    fn size(&self) -> Size<i32, BufferCoords> {
        self.size
    }

    #[inline]
    fn format(&self) -> Format {
        self.format
    }
}

/// Errors during conversion to a dmabuf handle from a gbm buffer object
#[derive(thiserror::Error, Debug)]
pub enum GbmConvertError {
    /// The buffer consists out of multiple file descriptions, which is currently unsupported
    #[error("Buffer consists out of multiple file descriptors, which is currently unsupported")]
    UnsupportedBuffer,
    /// The conversion returned an invalid file descriptor
    #[error("Buffer returned invalid file descriptor")]
    InvalidFD(#[from] gbm::InvalidFdError),
}

impl AsDmabuf for GbmBuffer {
    type Error = GbmConvertError;

    #[cfg(feature = "backend_gbm_has_fd_for_plane")]
    #[profiling::function]
    fn export(&self) -> Result<Dmabuf, GbmConvertError> {
        let planes = self.plane_count() as i32;

        let mut builder = Dmabuf::builder_from_buffer(self, DmabufFlags::empty());
        for idx in 0..planes {
            let fd = self.fd_for_plane(idx)?;

            builder.add_plane(
                // SAFETY: `gbm_bo_get_fd_for_plane` returns a new fd owned by the caller.
                fd,
                idx as u32,
                self.offset(idx),
                self.stride_for_plane(idx),
            );
        }

        if let Some(node) = self.device_node() {
            builder.set_node(node);
        }

        Ok(builder.build().unwrap())
    }

    #[cfg(not(feature = "backend_gbm_has_fd_for_plane"))]
    #[profiling::function]
    fn export(&self) -> Result<Dmabuf, GbmConvertError> {
        let planes = self.plane_count() as i32;

        let mut iter = (0i32..planes).map(|i| self.handle_for_plane(i));
        let first = iter.next().expect("Encountered a buffer with zero planes");
        // check that all handles are the same
        let handle = iter.try_fold(first, |first, next| {
            if unsafe { next.u64_ == first.u64_ } {
                return Some(first);
            }
            None
        });
        if handle.is_none() {
            // GBM is lacking a function to get a FD for a given plane. Instead,
            // check all planes have the same handle. We can't use
            // drmPrimeHandleToFD because that messes up handle ref'counting in
            // the user-space driver.
            return Err(GbmConvertError::UnsupportedBuffer);
        }

        let mut builder = Dmabuf::builder_from_buffer(self, DmabufFlags::empty());
        for idx in 0..planes {
            let fd = self.fd()?;

            builder.add_plane(
                // SAFETY: `gbm_bo_get_fd` returns a new fd owned by the caller.
                fd,
                idx as u32,
                self.offset(idx),
                self.stride_for_plane(idx),
            );
        }

        if let Some(node) = self.device_node() {
            builder.set_node(node);
        }

        Ok(builder.build().unwrap())
    }
}

impl Dmabuf {
    /// Import a Dmabuf using libgbm, creating a gbm Buffer Object to the same underlying data.
    #[profiling::function]
    pub fn import_to<A: AsFd + 'static>(
        &self,
        gbm: &GbmDevice<A>,
        usage: GbmBufferFlags,
    ) -> std::io::Result<GbmBuffer> {
        let mut handles = [None; MAX_PLANES];
        for (i, handle) in self.handles().take(MAX_PLANES).enumerate() {
            handles[i] = Some(handle);
        }
        let mut strides = [0i32; MAX_PLANES];
        for (i, stride) in self.strides().take(MAX_PLANES).enumerate() {
            strides[i] = stride as i32;
        }
        let mut offsets = [0i32; MAX_PLANES];
        for (i, offset) in self.offsets().take(MAX_PLANES).enumerate() {
            offsets[i] = offset as i32;
        }

        if self.has_modifier() || self.num_planes() > 1 || self.offsets().next().unwrap() != 0 {
            gbm.import_buffer_object_from_dma_buf_with_modifiers(
                self.num_planes() as u32,
                handles,
                self.width(),
                self.height(),
                self.format().code,
                usage,
                strides,
                offsets,
                self.format().modifier,
            )
            .map(|bo| GbmBuffer::from_bo(bo, false))
        } else {
            gbm.import_buffer_object_from_dma_buf(
                handles[0].unwrap(),
                self.width(),
                self.height(),
                strides[0] as u32,
                self.format().code,
                if self.format().modifier == Modifier::Linear {
                    usage | GbmBufferFlags::LINEAR
                } else {
                    usage
                },
            )
            .map(|bo| GbmBuffer::from_bo(bo, true))
        }
    }
}
