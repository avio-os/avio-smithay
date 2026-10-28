//! Per-surface accepted property metadata and bounded submission storage.
//!
//! Discovery is device-wide and may run for a sibling output. A prepared
//! request borrows only this owner's snapshot, never the discovery map.
//! Connector changes replace metadata and admit its arrays under the same
//! ordered surface guard. Cold mode probes use separate request storage.

use super::request::{AtomicRequest, AtomicRequestStorage};
use crate::backend::drm::{device::atomic::PropMapping, Planes};
use drm::control::crtc;

#[derive(Debug)]
pub(super) struct AtomicRequestWorkspace {
    mapping: PropMapping,
    storage: AtomicRequestStorage,
}

impl AtomicRequestWorkspace {
    pub(super) fn new(mapping: &PropMapping, crtc: crtc::Handle, planes: &Planes) -> Self {
        let mut accepted = mapping.clone();
        accepted.crtcs.retain(|handle, _| *handle == crtc);
        accepted.planes.retain(|handle, _| {
            planes
                .primary
                .iter()
                .chain(&planes.cursor)
                .chain(&planes.overlay)
                .any(|plane| plane.handle == *handle)
        });
        let storage = AtomicRequestStorage::for_surface(&accepted, crtc, planes);
        Self {
            mapping: accepted,
            storage,
        }
    }

    /// Called only at ordered connector admission or hardware resynchronization.
    /// Previously admitted connector handles stay valid for disconnect requests.
    pub(super) fn refresh_connectors(&mut self, discovered: &PropMapping) {
        self.mapping.connectors.extend(discovered.connectors.clone());
        self.storage.admit_connectors_cold(&self.mapping);
    }

    pub(super) fn plane_capacity(&self) -> usize {
        self.storage.plane_capacity()
    }
    pub(super) fn connector_capacity(&self) -> usize {
        self.storage.connector_capacity()
    }

    pub(super) fn request(&mut self) -> AtomicRequest<'_> {
        AtomicRequest::prepared(&self.mapping, &mut self.storage)
    }
}

#[cfg(test)]
#[path = "workspace_tests.rs"]
mod tests;
