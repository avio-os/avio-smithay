//! One checked native request encoder for cold configuration and warm submission.
use super::super::PlaneState;
use super::{to_fixed, DrmRotation};
use crate::backend::drm::{device::atomic::PropMapping, error::Error, Planes};
use crate::utils::Transform;
use drm::control::{connector, crtc, plane, property, AtomicCommitFlags, ResourceHandle};
use std::{
    fmt,
    ops::{Deref, DerefMut},
    os::unix::io::{AsFd, AsRawFd},
};
#[path = "request_storage.rs"]
mod storage;
pub(super) use storage::AtomicRequestStorage;
use storage::ObjectRoster;

fn raw_handle<H: ResourceHandle>(handle: H) -> u32 {
    let handle: drm::control::RawResourceHandle = handle.into();
    handle.get()
}
const CRTC_PROPERTIES: &[&str] = &["ACTIVE", "MODE_ID", "VRR_ENABLED", "OUT_FENCE_PTR"];
const CONNECTOR_PROPERTIES: &[&str] = &["CRTC_ID"];
const PLANE_PROPERTIES: &[&str] = &[
    "CRTC_ID",
    "FB_ID",
    "SRC_X",
    "SRC_Y",
    "SRC_W",
    "SRC_H",
    "CRTC_X",
    "CRTC_Y",
    "CRTC_W",
    "CRTC_H",
    "rotation",
    "alpha",
    "FB_DAMAGE_CLIPS",
    "IN_FENCE_FD",
];
fn object_roster<H: ResourceHandle>(
    handle: H,
    mapping: &std::collections::HashMap<String, property::Handle>,
    names: &[&str],
) -> ObjectRoster {
    ObjectRoster::cold(
        raw_handle(handle),
        names
            .iter()
            .filter_map(|name| mapping.get(*name).map(|p| u32::from(*p))),
    )
}
impl AtomicRequestStorage {
    pub(super) fn admit_connectors_cold(&mut self, mapping: &PropMapping) {
        self.admit_objects_cold(
            mapping
                .connectors
                .iter()
                .map(|(h, p)| object_roster(*h, p, CONNECTOR_PROPERTIES)),
        );
    }
    /// The actual compatible plane roster and known connector inventory at cold surface creation.
    pub(super) fn for_surface(mapping: &PropMapping, crtc: crtc::Handle, planes: &Planes) -> Self {
        let mut roster = Vec::new();
        if let Some(props) = mapping.crtcs.get(&crtc) {
            roster.push(object_roster(crtc, props, CRTC_PROPERTIES));
        }
        roster.extend(
            mapping
                .connectors
                .iter()
                .map(|(h, p)| object_roster(*h, p, CONNECTOR_PROPERTIES)),
        );
        let mut plane_count = 0;
        for plane in planes.primary.iter().chain(&planes.cursor).chain(&planes.overlay) {
            if let Some(props) = mapping.planes.get(&plane.handle) {
                roster.push(object_roster(plane.handle, props, PLANE_PROPERTIES));
                plane_count += 1;
            }
        }
        Self::cold(roster, plane_count)
    }
    fn for_cold_request(mapping: &PropMapping) -> Self {
        let mut roster = Vec::new();
        roster.extend(
            mapping
                .crtcs
                .iter()
                .map(|(h, p)| object_roster(*h, p, CRTC_PROPERTIES)),
        );
        roster.extend(
            mapping
                .connectors
                .iter()
                .map(|(h, p)| object_roster(*h, p, CONNECTOR_PROPERTIES)),
        );
        roster.extend(
            mapping
                .planes
                .iter()
                .map(|(h, p)| object_roster(*h, p, PLANE_PROPERTIES)),
        );
        Self::cold(roster, mapping.planes.len())
    }
}
enum Storage<'a> {
    Cold(AtomicRequestStorage),
    Prepared(&'a mut AtomicRequestStorage),
}
impl Deref for Storage<'_> {
    type Target = AtomicRequestStorage;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Cold(s) => s,
            Self::Prepared(s) => s,
        }
    }
}
impl DerefMut for Storage<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Cold(s) => s,
            Self::Prepared(s) => s,
        }
    }
}
pub(super) struct AtomicRequest<'a> {
    mapping: &'a PropMapping,
    storage: Storage<'a>,
}
impl fmt::Debug for AtomicRequest<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.storage.fmt(f)
    }
}
impl Drop for AtomicRequest<'_> {
    fn drop(&mut self) {
        self.storage.clear();
    }
}
impl<'a> AtomicRequest<'a> {
    /// Cold-only compatibility construction; warm callers borrow their admitted surface storage.
    pub(super) fn new(mapping: &'a PropMapping) -> Self {
        Self {
            mapping,
            storage: Storage::Cold(AtomicRequestStorage::for_cold_request(mapping)),
        }
    }
    pub(super) fn prepared(mapping: &'a PropMapping, storage: &'a mut AtomicRequestStorage) -> Self {
        storage.clear();
        Self {
            mapping,
            storage: Storage::Prepared(storage),
        }
    }
    fn add_property<H: ResourceHandle>(
        &mut self,
        handle: H,
        prop: property::Handle,
        value: property::Value<'_>,
    ) -> Result<(), Error> {
        self.storage.add(raw_handle(handle), prop.into(), value.into())
    }
    pub(super) fn commit(&mut self, device: &impl AsFd, flags: AtomicCommitFlags) -> std::io::Result<()> {
        self.storage.commit(device.as_fd(), flags.bits())
    }
    #[cfg(test)]
    pub(super) fn value<H: ResourceHandle>(&self, handle: H, name: &'static str) -> Option<u64> {
        let object = raw_handle(handle);
        let property = self
            .mapping
            .crtcs
            .values()
            .chain(self.mapping.connectors.values())
            .chain(self.mapping.planes.values())
            .find_map(|props| {
                props
                    .get(name)
                    .filter(|p| self.storage.value(object, (**p).into()).is_some())
            })?;
        self.storage.value(object, (*property).into())
    }
    pub(super) fn plane_edits(&self) -> &[(plane::Handle, bool)] {
        self.storage.plane_edits()
    }
    pub(super) fn set_connector(&mut self, conn: connector::Handle, crtc: crtc::Handle) -> Result<(), Error> {
        self.add_property(
            conn,
            self.mapping.conn_prop_handle(conn, "CRTC_ID")?,
            property::Value::CRTC(Some(crtc)),
        )?;
        Ok(())
    }

    pub(super) fn reset_connector(&mut self, conn: connector::Handle) -> Result<(), Error> {
        self.add_property(
            conn,
            self.mapping.conn_prop_handle(conn, "CRTC_ID")?,
            property::Value::CRTC(None),
        )?;
        Ok(())
    }

    pub(super) fn set_crtc(
        &mut self,
        crtc: crtc::Handle,
        mode: Option<property::Value<'static>>,
        vrr: bool,
    ) -> Result<(), Error> {
        if let Some(blob) = mode {
            self.add_property(crtc, self.mapping.crtc_prop_handle(crtc, "MODE_ID")?, blob)?;
        }

        self.add_property(
            crtc,
            self.mapping.crtc_prop_handle(crtc, "ACTIVE")?,
            property::Value::Boolean(true),
        )?;

        if let Ok(vrr_prop) = self.mapping.crtc_prop_handle(crtc, "VRR_ENABLED") {
            self.add_property(crtc, vrr_prop, property::Value::Boolean(vrr))?;
        } else if vrr {
            return Err(Error::UnknownProperty {
                handle: crtc.into(),
                name: "VRR_ENABLED",
            });
        }

        Ok(())
    }

    pub(super) fn set_crtc_out_fence_ptr(
        &mut self,
        crtc: crtc::Handle,
        out_fence_fd: &mut i32,
    ) -> Result<bool, Error> {
        let Ok(prop) = self.mapping.crtc_prop_handle(crtc, "OUT_FENCE_PTR") else {
            return Ok(false);
        };
        self.add_property(
            crtc,
            prop,
            property::Value::UnsignedRange((out_fence_fd as *mut i32 as usize) as u64),
        )?;
        Ok(true)
    }

    pub(super) fn reset_crtc(&mut self, crtc: crtc::Handle) -> Result<(), Error> {
        self.add_property(
            crtc,
            self.mapping.crtc_prop_handle(crtc, "ACTIVE")?,
            property::Value::Boolean(false),
        )?;
        self.add_property(
            crtc,
            self.mapping.crtc_prop_handle(crtc, "MODE_ID")?,
            property::Value::Blob(0),
        )?;
        if let Ok(prop) = self.mapping.crtc_prop_handle(crtc, "VRR_ENABLED") {
            self.add_property(crtc, prop, property::Value::Boolean(false))?;
        }
        Ok(())
    }

    pub(super) fn set_plane(
        &mut self,
        crtc: crtc::Handle,
        plane_state: &PlaneState<'_>,
    ) -> Result<(), Error> {
        let handle = plane_state.handle;
        self.storage
            .remember_plane(handle, plane_state.config.is_some())?;
        if let Some(config) = plane_state.config.as_ref() {
            // connect the plane to the CRTC
            self.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "CRTC_ID")?,
                property::Value::CRTC(Some(crtc)),
            )?;

            // Set the fb for the plane
            self.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "FB_ID")?,
                property::Value::Framebuffer(Some(config.fb)),
            )?;

            self.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "SRC_X")?,
                // these are 16.16. fixed point
                property::Value::UnsignedRange(to_fixed(config.src.loc.x) as u64),
            )?;
            self.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "SRC_Y")?,
                // these are 16.16. fixed point
                property::Value::UnsignedRange(to_fixed(config.src.loc.y) as u64),
            )?;
            self.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "SRC_W")?,
                // these are 16.16. fixed point
                property::Value::UnsignedRange(to_fixed(config.src.size.w) as u64),
            )?;
            self.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "SRC_H")?,
                // these are 16.16. fixed point
                property::Value::UnsignedRange(to_fixed(config.src.size.h) as u64),
            )?;

            self.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "CRTC_X")?,
                property::Value::SignedRange(config.dst.loc.x as i64),
            )?;
            self.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "CRTC_Y")?,
                property::Value::SignedRange(config.dst.loc.y as i64),
            )?;
            self.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "CRTC_W")?,
                property::Value::UnsignedRange(config.dst.size.w as u64),
            )?;
            self.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "CRTC_H")?,
                property::Value::UnsignedRange(config.dst.size.h as u64),
            )?;
            if let Ok(prop) = self.mapping.plane_prop_handle(handle, "rotation") {
                self.add_property(
                    handle,
                    prop,
                    property::Value::Bitmask(DrmRotation::from(config.transform).bits() as u64),
                )?;
            } else if config.transform != Transform::Normal {
                // if we are missing the rotation property we can no rely on
                // the driver to report a non working configuration and can
                // only guarantee that Transform::Normal (no rotation) will
                // work
                return Err(Error::UnknownProperty {
                    handle: handle.into(),
                    name: "rotation",
                });
            }
            if let Ok(prop) = self.mapping.plane_prop_handle(handle, "alpha") {
                self.add_property(
                    handle,
                    prop,
                    property::Value::UnsignedRange((config.alpha * u16::MAX as f32).round() as u64),
                )?;
            } else if config.alpha != 1.0 {
                // if we are missing the alpha property we can not display any transparent alpha values
                return Err(Error::UnknownProperty {
                    handle: handle.into(),
                    name: "alpha",
                });
            }
            if let Ok(prop) = self.mapping.plane_prop_handle(handle, "FB_DAMAGE_CLIPS") {
                if let Some(damage) = config.damage_clips.as_ref() {
                    self.add_property(handle, prop, *damage)?;
                } else {
                    self.add_property(handle, prop, property::Value::Blob(0))?;
                }
            }
            if let Ok(prop) = self.mapping.plane_prop_handle(handle, "IN_FENCE_FD") {
                if let Some(fence) = config.fence.as_ref().map(|f| f.as_raw_fd()) {
                    self.add_property(handle, prop, property::Value::SignedRange(fence as i64))?;
                } else {
                    self.add_property(handle, prop, property::Value::SignedRange(-1))?;
                }
            } else if config.fence.is_some() {
                return Err(Error::UnknownProperty {
                    handle: handle.into(),
                    name: "IN_FENCE_FD",
                });
            }
        } else {
            self.reset_plane(handle)?;
        }

        Ok(())
    }

    pub(super) fn reset_plane(&mut self, plane: plane::Handle) -> Result<(), Error> {
        self.storage.remember_plane(plane, false)?;
        self.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "CRTC_ID")?,
            property::Value::CRTC(None),
        )?;

        self.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "FB_ID")?,
            property::Value::Framebuffer(None),
        )?;

        // reset the plane properties
        self.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "SRC_X")?,
            // these are 16.16. fixed point
            property::Value::UnsignedRange(0u64),
        )?;
        self.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "SRC_Y")?,
            // these are 16.16. fixed point
            property::Value::UnsignedRange(0u64),
        )?;
        self.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "SRC_W")?,
            // these are 16.16. fixed point
            property::Value::UnsignedRange(0u64),
        )?;
        self.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "SRC_H")?,
            // these are 16.16. fixed point
            property::Value::UnsignedRange(0u64),
        )?;

        self.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "CRTC_X")?,
            property::Value::SignedRange(0i64),
        )?;
        self.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "CRTC_Y")?,
            property::Value::SignedRange(0i64),
        )?;
        self.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "CRTC_W")?,
            property::Value::UnsignedRange(0u64),
        )?;
        self.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "CRTC_H")?,
            property::Value::UnsignedRange(0u64),
        )?;
        if let Ok(prop) = self.mapping.plane_prop_handle(plane, "rotation") {
            self.add_property(
                plane,
                prop,
                property::Value::Bitmask(DrmRotation::from(Transform::Normal).bits() as u64),
            )?;
        }
        if let Ok(prop) = self.mapping.plane_prop_handle(plane, "alpha") {
            self.add_property(plane, prop, property::Value::UnsignedRange(0xffff))?;
        }
        if let Ok(prop) = self.mapping.plane_prop_handle(plane, "FB_DAMAGE_CLIPS") {
            self.add_property(plane, prop, property::Value::Blob(0))?;
        }
        if let Ok(prop) = self.mapping.plane_prop_handle(plane, "IN_FENCE_FD") {
            self.add_property(plane, prop, property::Value::SignedRange(-1))?;
        }
        Ok(())
    }

    pub(super) fn fill_request<'c, 'p>(
        &mut self,
        crtc: crtc::Handle,
        blob: Option<property::Value<'static>>,
        vrr: bool,
        connectors: impl IntoIterator<Item = &'c connector::Handle>,
        removed: impl IntoIterator<Item = &'c connector::Handle>,
        planes: impl IntoIterator<Item = PlaneState<'p>>,
    ) -> Result<(), Error> {
        for conn in connectors {
            self.set_connector(*conn, crtc)?;
        }
        for conn in removed {
            self.reset_connector(*conn)?;
        }
        self.set_crtc(crtc, blob, vrr)?;
        for plane in planes {
            self.set_plane(crtc, &plane)?;
        }
        Ok(())
    }
    pub(super) fn build_request<'c, 'p, 'q: 'p>(
        mapping: &'a PropMapping,
        crtc: crtc::Handle,
        blob: Option<property::Value<'static>>,
        vrr: bool,
        connectors: impl IntoIterator<Item = &'c connector::Handle>,
        removed: impl IntoIterator<Item = &'c connector::Handle>,
        planes: impl IntoIterator<Item = &'p PlaneState<'q>>,
    ) -> Result<Self, Error> {
        let mut request = Self::new(mapping);
        for conn in connectors {
            request.set_connector(*conn, crtc)?;
        }
        for conn in removed {
            request.reset_connector(*conn)?;
        }
        request.set_crtc(crtc, blob, vrr)?;
        for plane in planes {
            request.set_plane(crtc, plane)?;
        }
        Ok(request)
    }
}
