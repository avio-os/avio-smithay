//! [`ExportFramebuffer`] implementation that imports dmabufs into the KMS
//! device via raw PRIME ioctls — deliberately no GBM/Mesa involvement.
//!
//! This matters on split-driver stacks such as virtio-gpu with Venus: buffers
//! allocated by Vulkan (Venus blob resources) cannot be re-imported through
//! Mesa's GBM (a virgl/GL context on such guests), but the kernel PRIME path
//! works for any dmabuf that is already a GEM object of the scanout device.

use drm::buffer::PlanarBuffer;
use drm::control::{framebuffer, Device as ControlDevice, FbCmd2Flags};
use drm::Device as BasicDevice;
use drm::DriverCapability;
use tracing::warn;

use super::{ExportBuffer, ExportFramebuffer, FramebufferPolicy};
use crate::backend::allocator::format::get_opaque;
use crate::backend::allocator::{dmabuf::Dmabuf, Buffer as AllocBuffer, Fourcc, Modifier};
use crate::backend::drm::{DrmDeviceFd, Framebuffer};

/// Errors raised by [`PrimeFramebufferExporter`].
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Another raw PRIME transaction owns this DRM file's GEM handles.
    #[error("PRIME handle transaction is busy")]
    Busy,
    /// A previous raw PRIME transaction unwound with uncertain handle custody.
    #[error("PRIME handle transaction is poisoned")]
    TransactionPoisoned,
    /// The buffer type is not supported by this exporter
    #[error("Unsupported buffer supplied")]
    Unsupported,
    /// PRIME import of a dmabuf plane failed
    #[error("PRIME fd import failed: {0}")]
    Import(#[source] std::io::Error),
    /// Framebuffer creation failed
    #[error("AddFB2 failed: {0}")]
    AddFb(#[source] std::io::Error),
    /// The buffer carries a non-linear explicit modifier but the device
    /// cannot take modifiers in AddFB2
    #[error("device lacks AddFB2 modifier support for modifier {0:?}")]
    ModifierUnsupported(Modifier),
}

/// Export framebuffers via raw PRIME import, without GBM.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrimeFramebufferExporter;

/// Framebuffer API selected from the KMS device's capability before use.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExplicitFramebufferApi {
    /// AddFB2 with the buffer's exact explicit modifier.
    Modifiers,
    /// AddFB2 without modifier flags, only for a proven single-plane linear BO.
    Linear,
}

/// PRIME exporter for an explicitly admitted layout and framebuffer API.
///
/// The KMS file must not be the GBM allocation file: PRIME handle imports are
/// not reference-counted against other users of that same file description.
#[derive(Debug, Clone, Copy)]
pub struct ExplicitPrimeFramebufferExporter(pub ExplicitFramebufferApi);

/// Framebuffer created by [`PrimeFramebufferExporter`].
#[derive(Debug)]
pub struct PrimeFramebuffer {
    drm: DrmDeviceFd,
    fb: framebuffer::Handle,
    format: drm_fourcc::DrmFormat,
}

impl Drop for PrimeFramebuffer {
    fn drop(&mut self) {
        if let Err(err) = self.drm.destroy_framebuffer(self.fb) {
            warn!(?err, "failed to destroy prime framebuffer");
        }
    }
}

impl AsRef<framebuffer::Handle> for PrimeFramebuffer {
    fn as_ref(&self) -> &framebuffer::Handle {
        &self.fb
    }
}

impl Framebuffer for PrimeFramebuffer {
    fn format(&self) -> drm_fourcc::DrmFormat {
        self.format
    }
}

struct PrimePlanarBuffer {
    size: (u32, u32),
    format: Fourcc,
    modifier: Option<drm::buffer::DrmModifier>,
    pitches: [u32; 4],
    handles: [Option<drm::buffer::Handle>; 4],
    offsets: [u32; 4],
}

impl PlanarBuffer for PrimePlanarBuffer {
    fn size(&self) -> (u32, u32) {
        self.size
    }
    fn format(&self) -> Fourcc {
        self.format
    }
    fn modifier(&self) -> Option<drm::buffer::DrmModifier> {
        self.modifier
    }
    fn pitches(&self) -> [u32; 4] {
        self.pitches
    }
    fn handles(&self) -> [Option<drm::buffer::Handle>; 4] {
        self.handles
    }
    fn offsets(&self) -> [u32; 4] {
        self.offsets
    }
}

fn framebuffer_from_prime(
    drm: &DrmDeviceFd,
    dmabuf: &Dmabuf,
    use_opaque: bool,
) -> Result<PrimeFramebuffer, Error> {
    framebuffer_from_prime_with_api(drm, dmabuf, use_opaque, None)
}

fn explicit_flags(
    api: ExplicitFramebufferApi,
    modifier: Modifier,
    planes: usize,
) -> Result<(Option<drm::buffer::DrmModifier>, FbCmd2Flags), Error> {
    match api {
        ExplicitFramebufferApi::Modifiers if modifier != Modifier::Invalid => Ok((
            Some(drm::buffer::DrmModifier::from(u64::from(modifier))),
            FbCmd2Flags::MODIFIERS,
        )),
        ExplicitFramebufferApi::Linear if modifier == Modifier::Linear && planes == 1 => {
            Ok((None, FbCmd2Flags::empty()))
        }
        _ => Err(Error::ModifierUnsupported(modifier)),
    }
}

struct ImportedHandles<'a> {
    drm: &'a DrmDeviceFd,
    handles: [Option<drm::buffer::Handle>; 4],
}

impl Drop for ImportedHandles<'_> {
    fn drop(&mut self) {
        for handle in self.handles.iter().flatten() {
            let _ = self.drm.close_buffer(*handle);
        }
    }
}

fn framebuffer_from_prime_with_api(
    drm: &DrmDeviceFd,
    dmabuf: &Dmabuf,
    use_opaque: bool,
    api: Option<ExplicitFramebufferApi>,
) -> Result<PrimeFramebuffer, Error> {
    with_prime_transaction(drm, || {
        framebuffer_from_prime_locked(drm, dmabuf, use_opaque, api)
    })
}

fn with_prime_transaction<T>(
    drm: &DrmDeviceFd,
    operation: impl FnOnce() -> Result<T, Error>,
) -> Result<T, Error> {
    let _custody = drm.try_prime_transaction().map_err(|error| match error {
        std::sync::TryLockError::WouldBlock => Error::Busy,
        std::sync::TryLockError::Poisoned(_) => Error::TransactionPoisoned,
    })?;
    operation()
}

fn framebuffer_from_prime_locked(
    drm: &DrmDeviceFd,
    dmabuf: &Dmabuf,
    use_opaque: bool,
    api: Option<ExplicitFramebufferApi>,
) -> Result<PrimeFramebuffer, Error> {
    let size = dmabuf.size();
    let format = dmabuf.format();
    let code = if use_opaque {
        get_opaque(format.code).unwrap_or(format.code)
    } else {
        format.code
    };

    let (modifier, flags) = if let Some(api) = api {
        explicit_flags(api, format.modifier, dmabuf.handles().count())?
    } else if format.modifier == Modifier::Invalid {
        (None, FbCmd2Flags::empty())
    } else if drm
        .get_driver_capability(DriverCapability::AddFB2Modifiers)
        .map(|value| value != 0)
        .unwrap_or(false)
    {
        (
            Some(drm::buffer::DrmModifier::from(u64::from(format.modifier))),
            FbCmd2Flags::MODIFIERS,
        )
    } else if format.modifier == Modifier::Linear {
        // A linear buffer scans out correctly through the implicit path.
        (None, FbCmd2Flags::empty())
    } else {
        return Err(Error::ModifierUnsupported(format.modifier));
    };

    let mut pitches = [0u32; 4];
    let mut offsets = [0u32; 4];
    let mut handles: [Option<drm::buffer::Handle>; 4] = [None; 4];
    let mut imported = ImportedHandles {
        drm,
        handles: [None; 4],
    };
    let mut imported_fds: [Option<(std::os::fd::RawFd, drm::buffer::Handle)>; 4] = [None; 4];
    for (idx, ((fd, stride), offset)) in dmabuf
        .handles()
        .zip(dmabuf.strides())
        .zip(dmabuf.offsets())
        .enumerate()
        .take(4)
    {
        use std::os::fd::AsRawFd;
        let raw = fd.as_raw_fd();
        let handle = match imported_fds.iter().flatten().find(|(f, _)| *f == raw) {
            Some((_, handle)) => *handle,
            None => {
                let handle = drm.prime_fd_to_buffer(fd).map_err(Error::Import)?;
                imported_fds[idx] = Some((raw, handle));
                // Distinct FDs may refer to the same GEM object. The ioctl
                // returns the existing per-file handle, not another reference.
                if !imported.handles.contains(&Some(handle)) {
                    imported.handles[idx] = Some(handle);
                }
                handle
            }
        };
        pitches[idx] = stride;
        offsets[idx] = offset;
        handles[idx] = Some(handle);
    }

    let fb = drm
        .add_planar_framebuffer(
            &PrimePlanarBuffer {
                size: (size.w as u32, size.h as u32),
                format: code,
                modifier,
                pitches,
                handles,
                offsets,
            },
            flags,
        )
        .inspect(|_| {
            crate::backend::allocator::note_gpu_allocation(
                crate::backend::allocator::GpuAllocationKind::DrmFramebuffer,
            )
        })
        .map_err(Error::AddFb);

    // The framebuffer holds its own reference to the underlying object;
    // release the userspace GEM handles regardless of the outcome.
    drop(imported);

    Ok(PrimeFramebuffer {
        drm: drm.clone(),
        fb: fb?,
        format: drm_fourcc::DrmFormat {
            code,
            modifier: format.modifier,
        },
    })
}

impl ExportFramebuffer<Dmabuf> for ExplicitPrimeFramebufferExporter {
    type Framebuffer = PrimeFramebuffer;
    type Error = Error;

    fn framebuffer_policy(&self) -> FramebufferPolicy {
        match self.0 {
            ExplicitFramebufferApi::Modifiers => FramebufferPolicy::ExplicitModifiers,
            ExplicitFramebufferApi::Linear => FramebufferPolicy::ExplicitLinear,
        }
    }

    fn add_framebuffer(
        &self,
        drm: &DrmDeviceFd,
        buffer: ExportBuffer<'_, Dmabuf>,
        use_opaque: bool,
    ) -> Result<Option<Self::Framebuffer>, Self::Error> {
        let dmabuf = match buffer {
            #[cfg(feature = "wayland_frontend")]
            ExportBuffer::Wayland(buffer) => match crate::wayland::dmabuf::get_dmabuf(buffer) {
                Ok(dmabuf) => dmabuf,
                Err(_) => return Ok(None),
            },
            ExportBuffer::Allocator(dmabuf) | ExportBuffer::Dmabuf(dmabuf) => dmabuf,
        };
        framebuffer_from_prime_with_api(drm, dmabuf, use_opaque, Some(self.0)).map(Some)
    }

    fn can_add_framebuffer(&self, buffer: &ExportBuffer<'_, Dmabuf>) -> bool {
        let dmabuf = match buffer {
            #[cfg(feature = "wayland_frontend")]
            ExportBuffer::Wayland(buffer) => match crate::wayland::dmabuf::get_dmabuf(buffer) {
                Ok(dmabuf) => dmabuf,
                Err(_) => return false,
            },
            ExportBuffer::Allocator(dmabuf) | ExportBuffer::Dmabuf(dmabuf) => dmabuf,
        };
        explicit_flags(self.0, dmabuf.format().modifier, dmabuf.handles().count()).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_file() -> DrmDeviceFd {
        // A non-DRM fd: the constructor's harmless master ioctl fails with
        // ENOTTY. The fake operations below never issue any GPU ioctl.
        let file = std::fs::File::open("/dev/null").unwrap();
        DrmDeviceFd::new(crate::utils::DeviceFd::from(std::os::fd::OwnedFd::from(file)))
    }

    #[test]
    fn overlapping_exporters_do_not_touch_another_transactions_handle() {
        let drm = fake_file();
        let cloned_file = drm.clone();
        let mut ioctls = Vec::new();
        with_prime_transaction(&drm, || {
            ioctls.push("import handle 7");
            // Deterministic interleaving: a second exporter reaches import
            // while the first owns the shared GEM handle, before its AddFB2.
            let competing: Result<(), Error> = with_prime_transaction(&cloned_file, || {
                panic!("busy exporter reached the import/close operations")
            });
            assert!(matches!(competing, Err(Error::Busy)));
            ioctls.push("add framebuffer with handle 7");
            ioctls.push("close handle 7");
            Ok(())
        })
        .unwrap();
        with_prime_transaction(&cloned_file, || {
            ioctls.push("import handle 8");
            ioctls.push("add framebuffer with handle 8");
            ioctls.push("close handle 8");
            Ok(())
        })
        .unwrap();
        assert_eq!(
            ioctls,
            [
                "import handle 7",
                "add framebuffer with handle 7",
                "close handle 7",
                "import handle 8",
                "add framebuffer with handle 8",
                "close handle 8"
            ]
        );
    }

    #[test]
    fn ordinary_import_failure_releases_transaction_custody() {
        let drm = fake_file();
        assert!(matches!(
            with_prime_transaction::<()>(&drm, || Err(Error::Unsupported)),
            Err(Error::Unsupported)
        ));
        with_prime_transaction(&drm, || Ok(())).unwrap();
    }

    #[test]
    fn unwound_transaction_cannot_be_reused() {
        let drm = fake_file();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = with_prime_transaction::<()>(&drm, || panic!("fake import panic"));
        }));
        assert!(panic.is_err());
        assert!(matches!(
            with_prime_transaction(&drm, || Ok(())),
            Err(Error::TransactionPoisoned)
        ));
    }

    #[test]
    fn declared_linear_api_keeps_explicit_layout_and_rejects_unknown_layouts() {
        assert_eq!(
            explicit_flags(ExplicitFramebufferApi::Linear, Modifier::Linear, 1).unwrap(),
            (None, FbCmd2Flags::empty())
        );
        assert!(explicit_flags(ExplicitFramebufferApi::Linear, Modifier::Invalid, 1).is_err());
        assert!(explicit_flags(ExplicitFramebufferApi::Linear, Modifier::Linear, 2).is_err());
        assert!(explicit_flags(ExplicitFramebufferApi::Linear, Modifier::from(7_u64), 1).is_err());
    }

    #[test]
    fn explicit_modifier_api_never_substitutes_implicit_layout() {
        let (modifier, flags) =
            explicit_flags(ExplicitFramebufferApi::Modifiers, Modifier::Linear, 1).unwrap();
        assert_eq!(modifier, Some(drm::buffer::DrmModifier::from(0)));
        assert_eq!(flags, FbCmd2Flags::MODIFIERS);
        assert!(explicit_flags(ExplicitFramebufferApi::Modifiers, Modifier::Invalid, 1).is_err());
    }
}

impl ExportFramebuffer<Dmabuf> for PrimeFramebufferExporter {
    type Framebuffer = PrimeFramebuffer;
    type Error = Error;

    #[profiling::function]
    fn add_framebuffer(
        &self,
        drm: &DrmDeviceFd,
        buffer: ExportBuffer<'_, Dmabuf>,
        use_opaque: bool,
    ) -> Result<Option<Self::Framebuffer>, Self::Error> {
        match buffer {
            #[cfg(feature = "wayland_frontend")]
            ExportBuffer::Wayland(wl_buffer) => match crate::wayland::dmabuf::get_dmabuf(wl_buffer) {
                Ok(dmabuf) => framebuffer_from_prime(drm, dmabuf, use_opaque).map(Some),
                Err(_) => Ok(None),
            },
            ExportBuffer::Allocator(dmabuf) => framebuffer_from_prime(drm, dmabuf, use_opaque).map(Some),
            ExportBuffer::Dmabuf(dmabuf) => framebuffer_from_prime(drm, dmabuf, use_opaque).map(Some),
        }
    }

    #[inline]
    fn can_add_framebuffer(&self, buffer: &ExportBuffer<'_, Dmabuf>) -> bool {
        match buffer {
            #[cfg(feature = "wayland_frontend")]
            ExportBuffer::Wayland(wl_buffer) => crate::wayland::dmabuf::get_dmabuf(wl_buffer).is_ok(),
            ExportBuffer::Allocator(_) => true,
            ExportBuffer::Dmabuf(_) => true,
        }
    }
}
