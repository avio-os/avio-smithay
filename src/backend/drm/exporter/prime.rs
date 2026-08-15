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

use super::{ExportBuffer, ExportFramebuffer};
use crate::backend::allocator::format::get_opaque;
use crate::backend::allocator::{dmabuf::Dmabuf, Buffer as AllocBuffer, Fourcc, Modifier};
use crate::backend::drm::{DrmDeviceFd, Framebuffer};

/// Errors raised by [`PrimeFramebufferExporter`].
#[derive(thiserror::Error, Debug)]
pub enum Error {
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
    let size = dmabuf.size();
    let format = dmabuf.format();
    let code = if use_opaque {
        get_opaque(format.code).unwrap_or(format.code)
    } else {
        format.code
    };

    let supports_modifiers = drm
        .get_driver_capability(DriverCapability::AddFB2Modifiers)
        .map(|val| val != 0)
        .unwrap_or(false);
    let (modifier, flags) = if format.modifier == Modifier::Invalid {
        (None, FbCmd2Flags::empty())
    } else if supports_modifiers {
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
    let mut imported: Vec<(std::os::fd::RawFd, drm::buffer::Handle)> = Vec::new();
    for (idx, ((fd, stride), offset)) in dmabuf
        .handles()
        .zip(dmabuf.strides())
        .zip(dmabuf.offsets())
        .enumerate()
        .take(4)
    {
        use std::os::fd::AsRawFd;
        let raw = fd.as_raw_fd();
        let handle = match imported.iter().find(|(f, _)| *f == raw) {
            Some((_, handle)) => *handle,
            None => {
                let handle = drm.prime_fd_to_buffer(fd).map_err(Error::Import)?;
                imported.push((raw, handle));
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
        .map_err(Error::AddFb);

    // The framebuffer holds its own reference to the underlying object;
    // release the userspace GEM handles regardless of the outcome.
    for (_, handle) in imported {
        let _ = drm.close_buffer(handle);
    }

    Ok(PrimeFramebuffer {
        drm: drm.clone(),
        fb: fb?,
        format: drm_fourcc::DrmFormat {
            code,
            modifier: format.modifier,
        },
    })
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
