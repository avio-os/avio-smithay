//! Cold output preparation separated from renderer-thread initialization.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    os::fd::AsFd,
    sync::{Arc, Mutex},
};

use drm::control::crtc;
use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};

use crate::{
    backend::{
        allocator::{
            dmabuf::{AsDmabuf, Dmabuf},
            gbm::GbmDevice,
            Allocator,
        },
        drm::{
            compositor::{DrmCompositor, FrameError, FrameResult},
            exporter::{ExportFramebuffer, FramebufferPolicy},
            DrmError, DrmSurface, Planes,
        },
        renderer::{element::RenderElement, Bind, Renderer, Texture},
    },
    output::OutputModeSource,
    utils::{Buffer, Size},
};

use super::{
    CompositorList, DrmOutput, DrmOutputManagerError, DrmOutputManagerResult, DrmOutputRenderElements,
};

/// Errors from preparing an output before its renderer exists.
#[derive(Debug, thiserror::Error)]
pub enum DrmOutputPrepareError {
    /// The CRTC already has an output or an outstanding initializer.
    #[error("The specified CRTC {0:?} is already in use or being initialized.")]
    DuplicateCrtc(crtc::Handle),
    /// The underlying DRM device rejected surface preparation.
    #[error(transparent)]
    Drm(#[from] DrmError),
}

#[derive(Debug)]
pub(super) struct OutputReservation {
    crtc: crtc::Handle,
    pending: Arc<Mutex<HashSet<crtc::Handle>>>,
}

impl OutputReservation {
    pub(super) fn new(
        pending: Arc<Mutex<HashSet<crtc::Handle>>>,
        crtc: crtc::Handle,
    ) -> Result<Self, DrmOutputPrepareError> {
        if !pending.lock().unwrap().insert(crtc) {
            return Err(DrmOutputPrepareError::DuplicateCrtc(crtc));
        }
        Ok(Self { crtc, pending })
    }
}

impl Drop for OutputReservation {
    fn drop(&mut self) {
        self.pending.lock().unwrap().remove(&self.crtc);
    }
}

/// An owning cold-start handoff for a DRM output.
///
/// Preparation does not retain a manager lock or a renderer. Initialize it on
/// the renderer's owning thread, which is required for thread-affine EGL/GLES
/// contexts. This type is `Send` when its allocator, exporter, and output state
/// are `Send`; it adds no unsafe thread-transfer implementation.
///
/// Dropping an unused or failed initializer releases its CRTC reservation.
#[must_use = "the prepared output must be initialized on its renderer thread"]
pub struct DrmOutputInitializer<A, F, U, G>
where
    A: Allocator,
    F: ExportFramebuffer<A::Buffer>,
    F::Framebuffer: fmt::Debug + Send + Sync + 'static,
    G: AsFd + 'static,
{
    pub(super) surface: Arc<DrmSurface>,
    pub(super) output_mode_source: OutputModeSource,
    pub(super) planes: Option<Planes>,
    pub(super) allocator: A,
    pub(super) exporter: F,
    pub(super) gbm: Option<GbmDevice<G>>,
    pub(super) compositor: CompositorList<A, F, U, G>,
    pub(super) color_formats: Vec<DrmFourcc>,
    pub(super) renderer_formats: Vec<DrmFormat>,
    pub(super) cursor_size: Size<u32, Buffer>,
    pub(super) native_black_enabled: bool,
    pub(super) reservation: OutputReservation,
}

impl<A, F, U, G> fmt::Debug for DrmOutputInitializer<A, F, U, G>
where
    A: Allocator,
    F: ExportFramebuffer<A::Buffer>,
    F::Framebuffer: fmt::Debug + Send + Sync + 'static,
    G: AsFd + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DrmOutputInitializer")
            .field("crtc", &self.reservation.crtc)
            .finish_non_exhaustive()
    }
}

impl<A, F, U, G> DrmOutputInitializer<A, F, U, G>
where
    A: Allocator + Clone + fmt::Debug,
    A::Buffer: AsDmabuf,
    A::Error: Send + Sync + 'static,
    <A::Buffer as AsDmabuf>::Error: Send + Sync + 'static,
    F: ExportFramebuffer<A::Buffer> + Clone,
    F::Framebuffer: fmt::Debug + Send + Sync + 'static,
    F::Error: Send + Sync + 'static,
    G: AsFd + Clone + 'static,
    U: 'static,
{
    /// Initialize and render the first frame on the renderer's owning thread.
    ///
    /// This takes the device-wide compositor lock only for initialization,
    /// preserving bandwidth fallbacks and native-black admission. The caller
    /// must release any [`super::LockedDrmOutputManager`] guard before calling.
    pub fn initialize<R, E>(
        self,
        renderer: &mut R,
        render_elements: &DrmOutputRenderElements<R, E>,
    ) -> DrmOutputManagerResult<DrmOutput<A, F, U, G>, A, F, R>
    where
        E: RenderElement<R>,
        R: Renderer + Bind<Dmabuf>,
        R::TextureId: Texture + 'static,
        R::Error: Send + Sync + 'static,
    {
        let compositors = self.compositor.clone();
        let mut compositors = compositors.write().unwrap();
        self.initialize_locked(&mut compositors, renderer, render_elements)
    }

    pub(super) fn initialize_locked<R, E>(
        self,
        compositors: &mut HashMap<crtc::Handle, Mutex<DrmCompositor<A, F, U, G>>>,
        renderer: &mut R,
        render_elements: &DrmOutputRenderElements<R, E>,
    ) -> DrmOutputManagerResult<DrmOutput<A, F, U, G>, A, F, R>
    where
        E: RenderElement<R>,
        R: Renderer + Bind<Dmabuf>,
        R::TextureId: Texture + 'static,
        R::Error: Send + Sync + 'static,
    {
        let output_mode_source = self.output_mode_source.clone();
        let crtc = self.reservation.crtc;
        let planes = &self.planes;

        if compositors.contains_key(&crtc) {
            return Err(DrmOutputManagerError::DuplicateCrtc(crtc));
        }

        let create_compositor = |implicit_modifiers: bool| -> FrameResult<DrmCompositor<A, F, U, G>, A, F> {
            let surface = self.surface.clone();

            if implicit_modifiers {
                DrmCompositor::<A, F, U, G>::new_with_output_layer_allocator(
                    output_mode_source.clone(),
                    surface,
                    planes.clone(),
                    self.allocator.clone(),
                    Some(self.allocator.clone()),
                    self.exporter.clone(),
                    self.color_formats.iter().copied(),
                    self.renderer_formats
                        .iter()
                        .filter(|f| f.modifier == DrmModifier::Invalid)
                        .copied(),
                    self.cursor_size,
                    self.gbm.clone(),
                )
            } else {
                DrmCompositor::<A, F, U, G>::new_with_output_layer_allocator(
                    output_mode_source.clone(),
                    surface,
                    planes.clone(),
                    self.allocator.clone(),
                    Some(self.allocator.clone()),
                    self.exporter.clone(),
                    self.color_formats.iter().copied(),
                    self.renderer_formats.iter().copied(),
                    self.cursor_size,
                    self.gbm.clone(),
                )
            }
            .map(|mut compositor| {
                if self.native_black_enabled {
                    compositor.enable_native_black();
                }
                compositor
            })
        };

        let compositor = create_compositor(false);

        // Okay, so this can fail for various reasons...
        //
        //  Enabling an additional CRTC might fail because the bandwidth
        //  requirement is higher then supported with the current configuration.
        //
        // * Bandwidth limitation caused by overlay plane usage:
        //   Each overlay plane requires some certain bandwidth and we only
        //   test that during plane assignment implicitly through an atomic test.
        //   When trying to enable an additional CRTC we might hit some limit and the
        //   only way to resolve this might be to disable all overlay planes and
        //   retry enabling the CRTC.
        //
        // * Bandwidth limitation caused by the primary plane format:
        //   Different formats (might) require a higher memory bandwidth than others.
        //   This also applies to the same fourcc with different modifiers. For example
        //   the Intel CCS Formats use an additional plane to transport meta-data.
        //   So if we fail to enable an additional CRTC we might be able to resolve
        //   the issue by using a different format. Again the only way to know is by
        //   trying out what works.
        //
        // So for now we try to disable overlay planes first. If that doesn't work,
        // we set the modifiers to `Invalid` (for now) to give the driver the opportunity
        // to re-allocate in the background. (TODO: Do this ourselves).

        match compositor {
            Ok(compositor) => {
                compositors.insert(crtc, Mutex::new(compositor));
            }
            Err(err) => {
                // An admitted explicit layout/fence contract cannot retry a
                // different modifier stack or render other outputs through
                // this context. Preserve the original setup failure.
                if self.exporter.framebuffer_policy() != FramebufferPolicy::Legacy {
                    return Err(DrmOutputManagerError::Frame(err));
                }
                tracing::warn!(
                    ?crtc,
                    ?err,
                    "failed to initialize crtc, trying to lower bandwidth"
                );

                for compositor in compositors.values_mut() {
                    let compositor = compositor.get_mut().unwrap();
                    if let Err(err) = render_elements.submit_composited_frame(&mut *compositor, renderer) {
                        if !matches!(err, DrmOutputManagerError::Frame(FrameError::EmptyFrame)) {
                            return Err(err);
                        }
                    }
                }

                let compositor = create_compositor(false);

                match compositor {
                    Ok(compositor) => {
                        compositors.insert(crtc, Mutex::new(compositor));
                    }
                    Err(err) => {
                        tracing::warn!(
                            ?crtc,
                            ?err,
                            "failed to initialize crtc, trying implicit modifiers"
                        );

                        for compositor in compositors.values_mut() {
                            let compositor = compositor.get_mut().unwrap();

                            let current_format = compositor.format();
                            if let Err(err) = compositor.set_format(
                                self.allocator.clone(),
                                current_format,
                                [DrmModifier::Invalid],
                            ) {
                                tracing::warn!(?err, "failed to set new format");
                                continue;
                            }

                            render_elements.submit_composited_frame(&mut *compositor, renderer)?;
                        }

                        let compositor = create_compositor(true);

                        match compositor {
                            Ok(compositor) => {
                                compositors.insert(crtc, Mutex::new(compositor));
                            }
                            Err(err) => {
                                // try to reset formats
                                for compositor in compositors.values_mut() {
                                    let compositor = compositor.get_mut().unwrap();

                                    let current_format = compositor.format();
                                    if let Err(err) = compositor.set_format(
                                        self.allocator.clone(),
                                        current_format,
                                        self.renderer_formats
                                            .iter()
                                            .filter(|f| f.code == current_format)
                                            .map(|f| f.modifier),
                                    ) {
                                        tracing::warn!(?err, "failed to reset format");
                                        continue;
                                    }

                                    render_elements.submit_composited_frame(&mut *compositor, renderer)?;
                                }

                                return Err(DrmOutputManagerError::Frame(err));
                            }
                        }
                    }
                }
            }
        };

        // We need to render the new output once to lock in the primary plane as used with the new format, so we don't hit the bandwidth issue,
        // when downstream potentially uses `FrameFlags::DEFAULT` immediately after this.

        let compositor = compositors.get_mut(&crtc).unwrap();
        let compositor = compositor.get_mut().unwrap();
        if let Err(err) = render_elements.submit_composited_frame(&mut *compositor, renderer) {
            compositors.remove(&crtc);
            return Err(err);
        }

        if self.native_black_enabled {
            if let Err(error) = compositor.configure_native_black(renderer) {
                compositors.remove(&crtc);
                return Err(DrmOutputManagerError::RenderFrame(error));
            }
        }

        Ok(DrmOutput {
            compositor: self.compositor.clone(),
            crtc,
            allocator: self.allocator.clone(),
            renderer_formats: self.renderer_formats.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;

    #[test]
    fn pending_output_reserves_crtc_until_owning_handoff_is_dropped() {
        let pending = Arc::new(Mutex::new(HashSet::new()));
        let crtc = crtc::Handle::from(NonZeroU32::new(1).unwrap());
        let reservation = OutputReservation::new(pending.clone(), crtc).unwrap();
        assert!(matches!(
            OutputReservation::new(pending.clone(), crtc),
            Err(DrmOutputPrepareError::DuplicateCrtc(handle)) if handle == crtc
        ));
        std::thread::spawn(move || drop(reservation)).join().unwrap();
        let retry = OutputReservation::new(pending.clone(), crtc).unwrap();
        assert!(pending.lock().unwrap().contains(&crtc));
        drop(retry);
        assert!(pending.lock().unwrap().is_empty());
    }

    #[test]
    fn gbm_output_initializer_can_transfer_without_a_renderer() {
        use crate::backend::{
            allocator::gbm::GbmAllocator,
            drm::{exporter::gbm::GbmFramebufferExporter, DrmDeviceFd},
        };
        fn assert_send<T: Send>() {}
        assert_send::<
            DrmOutputInitializer<
                GbmAllocator<DrmDeviceFd>,
                GbmFramebufferExporter<DrmDeviceFd>,
                (),
                DrmDeviceFd,
            >,
        >();
    }
}
