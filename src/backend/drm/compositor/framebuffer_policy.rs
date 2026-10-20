//! Strict layout negotiation and pending-fence admission.
use super::*;
use crate::backend::drm::exporter::FramebufferPolicy;
use std::os::fd::OwnedFd;

pub(super) fn select_formats(
    code: DrmFourcc,
    planes: &IndexSet<DrmFormat>,
    renderer: &[DrmFormat],
    policy: FramebufferPolicy,
) -> Vec<DrmFormat> {
    if policy == FramebufferPolicy::Legacy {
        return select_compositor_formats(code, planes, renderer);
    }
    renderer
        .iter()
        .filter(|format| {
            format.modifier != DrmModifier::Invalid
                && match policy {
                    FramebufferPolicy::ExplicitLinear => {
                        format.modifier == DrmModifier::Linear
                            && planes.iter().any(|plane| {
                                matches!(plane.modifier, DrmModifier::Linear | DrmModifier::Invalid)
                            })
                    }
                    FramebufferPolicy::ExplicitModifiers => {
                        planes.iter().any(|plane| plane.modifier == format.modifier)
                    }
                    FramebufferPolicy::Legacy => unreachable!(),
                }
        })
        .map(|format| DrmFormat {
            code,
            modifier: format.modifier,
        })
        .collect()
}

/// Validate before constructing a real atomic request. A failed export is not
/// a reason to wait on the CPU or to submit an unsynchronized plane.
pub(super) fn admit_fence(
    sync: &SyncPoint,
    fence: &mut Option<Arc<OwnedFd>>,
    supports_fencing: bool,
) -> Result<(), DrmError> {
    if sync.is_reached() {
        return Ok(());
    }
    if !supports_fencing {
        return Err(DrmError::NativeFenceUnavailable);
    }
    if fence.is_none() {
        *fence = sync.export().map(Arc::new);
    }
    if fence.is_none() {
        Err(DrmError::NativeFenceUnavailable)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::renderer::sync::{Fence, Interrupted};

    #[derive(Debug)]
    struct Unexportable {
        reached: bool,
    }
    impl Fence for Unexportable {
        fn is_signaled(&self) -> bool {
            self.reached
        }
        fn wait(&self) -> Result<(), Interrupted> {
            panic!("strict KMS submission must never host-wait")
        }
        fn is_exportable(&self) -> bool {
            false
        }
        fn export(&self) -> Option<OwnedFd> {
            None
        }
    }

    #[test]
    fn pending_unexportable_fence_prevents_submission_without_cpu_wait() {
        let sync = SyncPoint::from(Unexportable { reached: false });
        for supports_fencing in [false, true] {
            let mut fence = None;
            let mut submitted = false;
            let result = admit_fence(&sync, &mut fence, supports_fencing).map(|()| submitted = true);
            assert!(matches!(result, Err(DrmError::NativeFenceUnavailable)));
            assert!(!submitted);
            assert!(fence.is_none());
        }
    }

    #[test]
    fn completed_fence_requires_no_export_or_host_wait() {
        let sync = SyncPoint::from(Unexportable { reached: true });
        assert!(admit_fence(&sync, &mut None, false).is_ok());
    }

    #[test]
    fn explicit_linear_survives_implicit_only_plane_negotiation() {
        let code = DrmFourcc::Argb8888;
        let plane = IndexSet::from([DrmFormat {
            code,
            modifier: DrmModifier::Invalid,
        }]);
        let renderer = [DrmFormat {
            code,
            modifier: DrmModifier::Linear,
        }];
        let selected = select_formats(code, &plane, &renderer, FramebufferPolicy::ExplicitLinear);
        assert_eq!(selected, renderer);
        // The subsequent fixed-format recreation applies the same contract.
        assert_eq!(
            select_formats(code, &plane, &selected, FramebufferPolicy::ExplicitLinear),
            renderer
        );
        assert!(select_formats(code, &plane, &renderer, FramebufferPolicy::ExplicitModifiers).is_empty());
        assert_eq!(
            select_formats(code, &plane, &renderer, FramebufferPolicy::Legacy)[0].modifier,
            DrmModifier::Invalid
        );
    }
}
