use drm::control::connector::Interface;
use drm::control::property::ValueType;
use drm::control::Device as ControlDevice;
use drm::control::{
    connector, crtc, dumbbuffer::DumbBuffer, framebuffer, plane, property, AtomicCommitFlags, Mode, PlaneType,
};

use std::collections::HashSet;
use std::os::unix::io::{FromRawFd, OwnedFd};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, RwLock,
};

use crate::backend::drm::error::AccessError;
use crate::utils::{Coordinate, Rectangle, Transform};
use crate::{
    backend::{
        allocator::format::{get_bpp, get_depth},
        drm::{
            device::atomic::{map_props, PropMapping},
            device::DrmDeviceInternal,
            error::Error,
            plane_type, DrmDeviceFd,
        },
    },
    utils::DevPath,
};

use tracing::{debug, info, info_span, instrument, trace, warn};

use super::{
    mode_blob::{ModeBlob, ModeBlobBank},
    PlaneConfig, PlaneState, VrrSupport,
};
mod request;
use request::{AtomicRequest, AtomicRequestStorage};

#[derive(Debug, Clone)]
pub struct State {
    pub active: bool,
    pub mode: Mode,
    /// The blob that sets `mode`. A committed state shares the pending state's
    /// blob; a state read back from KMS owns none, because no request names it.
    pub blob: Option<ModeBlob>,
    pub vrr: bool,
    pub connectors: HashSet<connector::Handle>,
}

impl PartialEq for State {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.active == other.active
            && self.mode == other.mode
            && self.vrr == other.vrr
            && self.connectors == other.connectors
    }
}

fn adopt_pending_mode(
    live_current: &State,
    live_pending: &mut State,
    current: &State,
    expected: &State,
    candidate: &mut State,
) -> bool {
    if !live_current.same_mode_owner(current) || !live_pending.same_mode_owner(expected) {
        return false;
    }
    std::mem::swap(live_pending, candidate);
    true
}

impl State {
    fn same_mode_owner(&self, other: &Self) -> bool {
        self == other
            && match (&self.blob, &other.blob) {
                (Some(a), Some(b)) => a.same_owner(b),
                (None, None) => true,
                _ => false,
            }
    }
    fn current_state<A: DevPath + ControlDevice>(
        fd: &A,
        crtc: crtc::Handle,
        prop_mapping: &mut PropMapping,
    ) -> Result<Self, Error> {
        let crtc_info = fd.get_crtc(crtc).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Error loading crtc info",
                dev: fd.dev_path(),
                source,
            })
        })?;

        // If we have no current mode, we create a fake one, which will not match (and thus gets overridden on the commit below).
        // A better fix would probably be making mode an `Option`, but that would mean
        // we need to be sure, we require a mode to always be set without relying on the compiler.
        // So we cheat, because it works and is easier to handle later.
        let current_mode = crtc_info.mode().unwrap_or_else(|| unsafe { std::mem::zeroed() });

        let res_handles = fd.resource_handles().map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Error loading drm resources",
                dev: fd.dev_path(),
                source,
            })
        })?;

        // the current set of connectors are those, that already have the correct `CRTC_ID` set.
        // so we collect them for `current_state` and set the user-given once in `pending_state`.
        //
        // If they don't match, `commit_pending` will return true and they will be changed on the next `commit`.
        let mut current_connectors = HashSet::new();
        // make sure the mapping is up to date
        map_props(fd, res_handles.connectors(), &mut prop_mapping.connectors)?;
        for conn in res_handles.connectors() {
            let crtc_prop = prop_mapping.conn_prop_handle(*conn, "CRTC_ID")?;
            if let (Ok(crtc_prop_info), Ok(props)) = (fd.get_property(crtc_prop), fd.get_properties(*conn)) {
                let (ids, vals) = props.as_props_and_values();
                for (&id, &val) in ids.iter().zip(vals.iter()) {
                    if id == crtc_prop {
                        if let property::Value::CRTC(Some(conn_crtc)) =
                            crtc_prop_info.value_type().convert_value(val)
                        {
                            if conn_crtc == crtc {
                                current_connectors.insert(*conn);
                            }
                        }
                        break;
                    }
                }
            }
        }

        // Get the current active (dpms) state and vrr state of the CRTC
        //
        // Changing a CRTC to active might require a modeset
        let mut active = None;
        let mut vrr = None;
        if let Ok(props) = fd.get_properties(crtc) {
            let active_prop = prop_mapping.crtcs.get(&crtc).and_then(|m| m.get("ACTIVE"));
            let vrr_prop = prop_mapping.crtcs.get(&crtc).and_then(|m| m.get("VRR_ENABLED"));
            let (ids, vals) = props.as_props_and_values();
            for (&id, &val) in ids.iter().zip(vals.iter()) {
                if Some(&id) == active_prop {
                    active = property::ValueType::Boolean.convert_value(val).as_boolean();
                    break;
                }
                if Some(&id) == vrr_prop {
                    vrr = property::ValueType::Boolean.convert_value(val).as_boolean();
                    break;
                }
            }
        }

        Ok(State::read_back(
            current_mode,
            // If we don't know the active state we just assume off.
            // This is highly unlikely, but having a false negative should do no harm.
            active.unwrap_or(false),
            // If we don't know the VRR state, the driver doesn't support the property
            vrr.unwrap_or(false),
            current_connectors,
        ))
    }

    /// The state KMS reports for a CRTC. It owns no blob: requests only ever
    /// name the pending blob, so a state read back from KMS needs none.
    /// Creating one here leaked a blob on every `reset_state`.
    fn read_back(mode: Mode, active: bool, vrr: bool, connectors: HashSet<connector::Handle>) -> Self {
        State {
            active,
            mode,
            blob: None,
            vrr,
            connectors,
        }
    }

    /// A pending state that sets `mode` through its own `blob`.
    fn with_mode(mode: Mode, blob: ModeBlob, connectors: &[connector::Handle]) -> Self {
        State {
            active: true,
            mode,
            blob: Some(blob),
            vrr: false,
            connectors: connectors.iter().copied().collect(),
        }
    }

    /// Take `mode`, set through `blob`. The superseded blob is destroyed,
    /// unless the committed state still shares it.
    fn set_mode(&mut self, mode: Mode, blob: ModeBlob) {
        self.mode = mode;
        self.blob = Some(blob);
    }

    /// The `MODE_ID` value a request sets for this state.
    fn mode_id(&self) -> Option<property::Value<'static>> {
        self.blob.as_ref().map(ModeBlob::value)
    }

    fn clear(&mut self) {
        self.mode = unsafe { std::mem::zeroed() };
        self.blob = None;
        self.connectors.clear();
        self.active = false;
        self.vrr = false;
    }
}

#[derive(Debug)]
pub struct AtomicDrmSurface {
    pub(in crate::backend::drm) fd: Arc<DrmDeviceInternal>,
    pub(super) active: Arc<AtomicBool>,
    crtc: crtc::Handle,
    plane: plane::Handle,
    used_planes: Mutex<HashSet<plane::Handle>>,
    request_storage: Mutex<AtomicRequestStorage>,
    mode_blobs: ModeBlobBank,
    prop_mapping: Arc<RwLock<PropMapping>>,
    state: RwLock<State>,
    pending: RwLock<State>,
    last_out_fence: Mutex<Option<OwnedFd>>,
    pub(super) span: tracing::Span,
}

impl AtomicDrmSurface {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fd: Arc<DrmDeviceInternal>,
        active: Arc<AtomicBool>,
        crtc: crtc::Handle,
        plane: plane::Handle,
        prop_mapping: Arc<RwLock<PropMapping>>,
        mode: Mode,
        connectors: &[connector::Handle],
        planes: &super::Planes,
    ) -> Result<Self, Error> {
        let span = info_span!("drm_atomic", crtc = ?crtc);
        let _guard = span.enter();
        info!(
            "Initializing drm surface ({:?}:{:?}) with mode {:?} and connectors {:?}",
            crtc, plane, mode, connectors
        );

        let mut state = State::current_state(&*fd, crtc, &mut prop_mapping.write().unwrap())?;
        let request_storage = AtomicRequestStorage::for_surface(&prop_mapping.read().unwrap(), crtc, planes);
        state.connectors.reserve(
            request_storage
                .connector_capacity()
                .saturating_sub(state.connectors.len()),
        );
        let used_planes = HashSet::with_capacity(request_storage.plane_capacity());
        let mode_blobs = fd.device_fd().mode_blob_bank(crtc)?;
        let pending = State::with_mode(
            mode,
            ModeBlob::new_in(&mode_blobs, fd.device_fd(), &mode)?,
            connectors,
        );

        drop(_guard);
        let surface = AtomicDrmSurface {
            fd,
            active,
            crtc,
            plane,
            used_planes: Mutex::new(used_planes),
            request_storage: Mutex::new(request_storage),
            mode_blobs,
            prop_mapping,
            state: RwLock::new(state),
            pending: RwLock::new(pending),
            last_out_fence: Mutex::new(None),
            span,
        };

        Ok(surface)
    }

    // we need a framebuffer to do test commits, which we use to verify our pending state.
    // here we create a dumbbuffer for that purpose.
    #[profiling::function]
    fn create_test_buffer(&self, size: (u16, u16), plane: plane::Handle) -> Result<TestBuffer, Error> {
        let (w, h) = size;
        let needs_alpha = plane_type(&*self.fd, plane)? != PlaneType::Primary;
        let format = if needs_alpha {
            crate::backend::allocator::Fourcc::Argb8888
        } else {
            crate::backend::allocator::Fourcc::Xrgb8888
        };

        let db = self
            .fd
            .create_dumb_buffer((w as u32, h as u32), format, get_bpp(format).unwrap() as u32)
            .inspect(|_| {
                crate::backend::allocator::note_gpu_allocation(
                    crate::backend::allocator::GpuAllocationKind::DrmDumbBuffer,
                )
            })
            .map_err(|source| {
                Error::Access(AccessError {
                    errmsg: "Failed to create dumb buffer",
                    dev: self.fd.dev_path(),
                    source,
                })
            })?;
        let fb_result = self
            .fd
            .add_framebuffer(
                &db,
                get_depth(format).unwrap() as u32,
                get_bpp(format).unwrap() as u32,
            )
            .inspect(|_| {
                crate::backend::allocator::note_gpu_allocation(
                    crate::backend::allocator::GpuAllocationKind::DrmFramebuffer,
                )
            })
            .map_err(|source| {
                Error::Access(AccessError {
                    errmsg: "Failed to create framebuffer",
                    dev: self.fd.dev_path(),
                    source,
                })
            });

        match fb_result {
            Ok(fb) => Ok(TestBuffer {
                fd: self.fd.clone(),
                db,
                fb,
            }),
            Err(err) => {
                let _ = self.fd.destroy_dumb_buffer(db);
                Err(err)
            }
        }
    }

    pub fn current_connectors(&self) -> HashSet<connector::Handle> {
        self.state.read().unwrap().connectors.clone()
    }

    pub(super) fn pending_configuration_matches(
        &self,
        mode: Mode,
        connectors: &HashSet<connector::Handle>,
    ) -> bool {
        self.pending
            .try_read()
            .is_ok_and(|pending| pending.mode == mode && pending.connectors == *connectors)
    }
    pub fn pending_connectors(&self) -> HashSet<connector::Handle> {
        self.pending.read().unwrap().connectors.clone()
    }

    pub fn current_mode(&self) -> Mode {
        self.state.read().unwrap().mode
    }

    pub fn pending_mode(&self) -> Mode {
        self.pending.read().unwrap().mode
    }

    fn ensure_props_known(&self, conns: &[connector::Handle]) -> Result<(), Error> {
        let mapping_exists = {
            let prop_mapping = self.prop_mapping.read().unwrap();
            conns
                .iter()
                .all(|conn| prop_mapping.connectors.contains_key(conn))
        };
        if !mapping_exists {
            map_props(
                &*self.fd,
                self.fd
                    .resource_handles()
                    .map_err(|source| {
                        Error::Access(AccessError {
                            errmsg: "Error loading connector info",
                            dev: self.fd.dev_path(),
                            source,
                        })
                    })?
                    .connectors(),
                &mut self.prop_mapping.write().unwrap().connectors,
            )?;
        }
        let connector_capacity = {
            let mapping = self.prop_mapping.read().unwrap();
            let mut storage = self.request_storage.lock().unwrap();
            storage.admit_connectors_cold(&mapping);
            storage.connector_capacity()
        };
        // Release map/storage before taking state: native commit takes state first.
        let mut current = self.state.write().unwrap();
        let additional = connector_capacity.saturating_sub(current.connectors.len());
        current.connectors.reserve(additional);
        Ok(())
    }

    #[instrument(parent = &self.span, skip(self))]
    pub fn add_connector(&self, conn: connector::Handle) -> Result<(), Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        self.ensure_props_known(&[conn])?;
        let info = self.fd.get_connector(conn, false).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Error loading connector info",
                dev: self.fd.dev_path(),
                source,
            })
        })?;

        let mut pending = self.pending.write().unwrap();

        // check if the connector can handle the current mode
        if info.modes().contains(&pending.mode) {
            let test_buffer = self.create_test_buffer(pending.mode.size(), self.plane)?;

            // check if config is supported
            let prop_mapping = self.prop_mapping.read().unwrap();
            let plane_state = PlaneState {
                handle: self.plane,
                config: Some(PlaneConfig {
                    src: Rectangle::from_size(pending.mode.size().into()).to_f64(),
                    dst: Rectangle::from_size(
                        (pending.mode.size().0 as i32, pending.mode.size().1 as i32).into(),
                    ),
                    transform: Transform::Normal,
                    alpha: 1.0,
                    damage_clips: None,
                    fb: test_buffer.fb,
                    fence: None,
                }),
            };

            let mut connectors = pending.connectors.clone();
            connectors.insert(conn);
            let mut req = AtomicRequest::build_request(
                &prop_mapping,
                self.crtc,
                pending.mode_id(),
                pending.vrr,
                &connectors,
                [],
                [&plane_state],
            )?;
            req.commit(
                &*self.fd,
                AtomicCommitFlags::ALLOW_MODESET | AtomicCommitFlags::TEST_ONLY,
            )
            .map_err(|_| Error::TestFailed(self.crtc))?;

            // seems to be, lets add the connector
            pending.connectors.insert(conn);

            Ok(())
        } else {
            Err(Error::ModeNotSuitable(pending.mode))
        }
    }

    #[instrument(parent = &self.span, skip(self))]
    pub fn remove_connector(&self, conn: connector::Handle) -> Result<(), Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let mut pending = self.pending.write().unwrap();

        // the test would also prevent this, but the error message is far less helpful
        if pending.connectors.contains(&conn) && pending.connectors.len() == 1 {
            return Err(Error::SurfaceWithoutConnectors(self.crtc));
        }

        // check if new config is supported (should be)
        let test_buffer = self.create_test_buffer(pending.mode.size(), self.plane)?;

        let prop_mapping = self.prop_mapping.read().unwrap();
        let plane_state = PlaneState {
            handle: self.plane,
            config: Some(PlaneConfig {
                src: Rectangle::from_size(pending.mode.size().into()).to_f64(),
                dst: Rectangle::from_size(
                    (pending.mode.size().0 as i32, pending.mode.size().1 as i32).into(),
                ),
                transform: Transform::Normal,
                alpha: 1.0,
                damage_clips: None,
                fb: test_buffer.fb,
                fence: None,
            }),
        };

        let mut connectors = pending.connectors.clone();
        connectors.remove(&conn);
        let mut req = AtomicRequest::build_request(
            &prop_mapping,
            self.crtc,
            pending.mode_id(),
            pending.vrr,
            &connectors,
            [&conn],
            [&plane_state],
        )?;
        req.commit(
            &*self.fd,
            AtomicCommitFlags::ALLOW_MODESET | AtomicCommitFlags::TEST_ONLY,
        )
        .map_err(|_| Error::TestFailed(self.crtc))?;

        // seems to be, lets remove the connector
        pending.connectors.remove(&conn);

        Ok(())
    }

    #[instrument(parent = &self.span, skip(self))]
    pub fn set_connectors(&self, connectors: &[connector::Handle]) -> Result<(), Error> {
        // the test would also prevent this, but the error message is far less helpful
        if connectors.is_empty() {
            return Err(Error::SurfaceWithoutConnectors(self.crtc));
        }

        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        self.ensure_props_known(connectors)?;
        let current = self.state.read().unwrap();
        let mut pending = self.pending.write().unwrap();
        let conns = connectors.iter().cloned().collect::<HashSet<_>>();
        let removed = current.connectors.difference(&conns);

        let test_buffer = self.create_test_buffer(pending.mode.size(), self.plane)?;

        let prop_mapping = self.prop_mapping.read().unwrap();
        let plane_state = PlaneState {
            handle: self.plane,
            config: Some(PlaneConfig {
                src: Rectangle::from_size(pending.mode.size().into()).to_f64(),
                dst: Rectangle::from_size(
                    (pending.mode.size().0 as i32, pending.mode.size().1 as i32).into(),
                ),
                transform: Transform::Normal,
                alpha: 1.0,
                damage_clips: None,
                fb: test_buffer.fb,
                fence: None,
            }),
        };
        let mut req = AtomicRequest::build_request(
            &prop_mapping,
            self.crtc,
            pending.mode_id(),
            pending.vrr,
            &conns,
            removed,
            [&plane_state],
        )?;

        req.commit(
            &*self.fd,
            AtomicCommitFlags::ALLOW_MODESET | AtomicCommitFlags::TEST_ONLY,
        )
        .map_err(|_| Error::TestFailed(self.crtc))?;

        pending.connectors = conns;

        Ok(())
    }

    #[instrument(level = "debug", parent = &self.span, skip(self))]
    pub fn use_mode(&self, mode: Mode) -> Result<(), Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let mut pending = self.pending.write().unwrap();

        // check if new config is supported; a failed test drops (destroys) the new blob
        let new_blob = ModeBlob::new_in(&self.mode_blobs, self.fd.device_fd(), &mode)?;

        let test_buffer = self.create_test_buffer(mode.size(), self.plane)?;

        let prop_mapping = self.prop_mapping.read().unwrap();
        let plane_state = PlaneState {
            handle: self.plane,
            config: Some(PlaneConfig {
                src: Rectangle::from_size(mode.size().into()).to_f64(),
                dst: Rectangle::from_size((mode.size().0 as i32, mode.size().1 as i32).into()),
                transform: Transform::Normal,
                alpha: 1.0,
                damage_clips: None,
                fb: test_buffer.fb,
                fence: None,
            }),
        };
        let mut req = AtomicRequest::build_request(
            &prop_mapping,
            self.crtc,
            Some(new_blob.value()),
            pending.vrr,
            pending.connectors.iter(),
            [],
            [&plane_state],
        )?;
        req.commit(
            &*self.fd,
            AtomicCommitFlags::ALLOW_MODESET | AtomicCommitFlags::TEST_ONLY,
        )
        .map_err(|_| Error::TestFailed(self.crtc))?;

        // seems to be, lets change the mode
        pending.set_mode(mode, new_blob);

        Ok(())
    }

    pub(super) fn prepare_mode(&self, mode: Mode) -> Result<super::prepared_mode::ModeState, Error> {
        if !self.active.load(Ordering::Acquire) {
            return Err(Error::DeviceInactive);
        }
        let current = self.state.read().unwrap().clone();
        let expected = self.pending.read().unwrap().clone();
        let mut candidate = expected.clone();
        if mode == expected.mode {
            return Ok(super::prepared_mode::ModeState::Atomic {
                current,
                expected,
                candidate,
            });
        }
        candidate.set_mode(
            mode,
            ModeBlob::new_in(&self.mode_blobs, self.fd.device_fd(), &mode)?,
        );
        let test_buffer = self.create_test_buffer(mode.size(), self.plane)?;
        let plane = PlaneState {
            handle: self.plane,
            config: Some(PlaneConfig {
                src: Rectangle::from_size(mode.size().into()).to_f64(),
                dst: Rectangle::from_size((i32::from(mode.size().0), i32::from(mode.size().1)).into()),
                transform: Transform::Normal,
                alpha: 1.0,
                damage_clips: None,
                fb: test_buffer.fb,
                fence: None,
            }),
        };
        self.test_state_cold([plane], true, &current, &candidate)?;
        Ok(super::prepared_mode::ModeState::Atomic {
            current,
            expected,
            candidate,
        })
    }
    pub(super) fn test_prepared_mode<'a>(
        &self,
        current: &State,
        expected: &State,
        candidate: &State,
        planes: impl IntoIterator<Item = PlaneState<'a>>,
        allow_modeset: bool,
    ) -> Result<(), Error> {
        if !self.active.load(Ordering::Acquire)
            || !self.state.read().unwrap().same_mode_owner(current)
            || !self.pending.read().unwrap().same_mode_owner(expected)
        {
            return Err(Error::TestFailed(self.crtc));
        }
        self.test_state_cold(planes, allow_modeset, current, candidate)
    }
    pub(super) fn adopt_prepared_mode(
        &self,
        current: &State,
        expected: &State,
        candidate: &mut State,
    ) -> bool {
        if !self.active.load(Ordering::Acquire) {
            return false;
        }
        let Ok(live_current) = self.state.try_read() else {
            return false;
        };
        let Ok(mut pending) = self.pending.try_write() else {
            return false;
        };
        adopt_pending_mode(&live_current, &mut pending, current, expected, candidate)
    }

    pub fn vrr_supported(&self, conn: connector::Handle) -> Result<VrrSupport, Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let props = self.prop_mapping.read().unwrap();
        if self
            .prop_mapping
            .read()
            .unwrap()
            .crtc_prop_handle(self.crtc, "VRR_ENABLED")
            .is_err()
        {
            return Ok(VrrSupport::NotSupported);
        }

        if let Some(vrr_prop) = props
            .connectors
            .get(&conn)
            .and_then(|props| props.get("vrr_capable"))
        {
            for (prop, value) in self.fd.get_properties(conn).map_err(|source| {
                Error::Access(AccessError {
                    errmsg: "Error querying properties",
                    dev: self.fd.dev_path(),
                    source,
                })
            })? {
                if prop == *vrr_prop {
                    let interface = self
                        .fd
                        .get_connector(conn, false)
                        .map_err(|source| {
                            Error::Access(AccessError {
                                errmsg: "Error querying connector",
                                dev: self.fd.dev_path(),
                                source,
                            })
                        })?
                        .interface();

                    // see: https://gitlab.freedesktop.org/drm/amd/-/issues/2200#note_2159982
                    // Currently setting VRR for HDMI connectors will cause flickering despite not needing `ALLOW_MODESET`
                    // TODO: Once the kernel is fixed, do actual test commits with and without `ALLOW_MODESET`.
                    return Ok(
                        match ValueType::Boolean.convert_value(value).as_boolean().unwrap() {
                            true if interface == Interface::HDMIA || interface == Interface::HDMIB => {
                                VrrSupport::RequiresModeset
                            }
                            true => VrrSupport::Supported,
                            false => VrrSupport::NotSupported,
                        },
                    );
                }
            }
        }

        Ok(VrrSupport::NotSupported)
    }

    pub fn vrr_enabled(&self) -> bool {
        self.pending.read().unwrap().vrr
    }

    pub fn use_vrr(&self, value: bool) -> Result<(), Error> {
        let mut current = self.state.write().unwrap();
        let mut pending = self.pending.write().unwrap();
        if pending.vrr == value {
            return Ok(());
        }
        let prop_mapping = self.prop_mapping.read().unwrap();

        if value && prop_mapping.crtc_prop_handle(self.crtc, "VRR_ENABLED").is_err() {
            return Err(Error::UnknownProperty {
                handle: self.crtc.into(),
                name: "VRR_ENABLED",
            });
        }

        let test_buffer = self.create_test_buffer(pending.mode.size(), self.plane)?;
        let plane_config = PlaneState {
            handle: self.plane,
            config: Some(PlaneConfig {
                src: Rectangle::from_size(pending.mode.size().into()).to_f64(),
                dst: Rectangle::from_size(
                    (pending.mode.size().0 as i32, pending.mode.size().1 as i32).into(),
                ),
                transform: Transform::Normal,
                alpha: 1.0,
                damage_clips: None,
                fb: test_buffer.fb,
                fence: None,
            }),
        };

        let mut req = AtomicRequest::build_request(
            &prop_mapping,
            self.crtc,
            pending.mode_id(),
            value,
            &pending.connectors,
            &[],
            [&plane_config],
        )?;

        if *current == *pending {
            // Try a non modesetting commit
            if req.commit(&*self.fd, AtomicCommitFlags::TEST_ONLY).is_ok() {
                pending.vrr = value;
                current.vrr = value;
                return Ok(());
            }
        }

        // Try a modeset commit
        req.commit(
            &*self.fd,
            AtomicCommitFlags::ALLOW_MODESET | AtomicCommitFlags::TEST_ONLY,
        )
        .map_err(|_| Error::TestFailed(self.crtc))?;

        pending.vrr = value;
        Ok(())
    }

    pub fn commit_pending(&self) -> bool {
        *self.pending.read().unwrap() != *self.state.read().unwrap()
    }

    #[instrument(level = "trace", parent = &self.span, skip(self, planes))]
    #[profiling::function]
    pub fn test_state<'a>(
        &self,
        planes: impl IntoIterator<Item = PlaneState<'a>>,
        allow_modeset: bool,
    ) -> Result<(), Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let current = self.state.try_read().map_err(|_| Error::AtomicRequestBusy)?;
        let pending = self.pending.try_read().map_err(|_| Error::AtomicRequestBusy)?;
        self.test_state_internal(planes, allow_modeset, &current, &pending)
    }

    fn test_state_internal<'a>(
        &self,
        planes: impl IntoIterator<Item = PlaneState<'a>>,
        allow_modeset: bool,
        current: &State,
        pending: &State,
    ) -> Result<(), Error> {
        let mapping = self
            .prop_mapping
            .try_read()
            .map_err(|_| Error::AtomicRequestBusy)?;
        let mut storage = self
            .request_storage
            .try_lock()
            .map_err(|_| Error::AtomicRequestBusy)?;
        let mut req = AtomicRequest::prepared(&mapping, &mut storage);
        self.test_state_request(&mut req, planes, allow_modeset, current, pending)
    }

    /// Prepared-mode validation executes on the cold output helper. Its native
    /// TEST_ONLY operation must not borrow the worker's request workspace:
    /// authority withdrawal/power-off can clear that worker concurrently.
    fn test_state_cold<'a>(
        &self,
        planes: impl IntoIterator<Item = PlaneState<'a>>,
        allow_modeset: bool,
        current: &State,
        pending: &State,
    ) -> Result<(), Error> {
        let mapping = self.prop_mapping.read().unwrap();
        let mut req = AtomicRequest::new(&mapping);
        self.test_state_request(&mut req, planes, allow_modeset, current, pending)
    }

    fn test_state_request<'a>(
        &self,
        req: &mut AtomicRequest<'_>,
        planes: impl IntoIterator<Item = PlaneState<'a>>,
        allow_modeset: bool,
        current: &State,
        pending: &State,
    ) -> Result<(), Error> {
        req.fill_request(
            self.crtc,
            pending.mode_id(),
            pending.vrr,
            &pending.connectors,
            current.connectors.difference(&pending.connectors),
            planes,
        )?;
        let flags = if allow_modeset {
            AtomicCommitFlags::ALLOW_MODESET | AtomicCommitFlags::TEST_ONLY
        } else {
            AtomicCommitFlags::TEST_ONLY
        };
        req.commit(&*self.fd, flags).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Error testing state",
                dev: self.fd.dev_path(),
                source,
            })
        })
    }

    #[instrument(level = "trace", parent = &self.span, skip(self, planes))]
    #[profiling::function]
    pub fn commit<'a>(
        &self,
        planes: impl IntoIterator<Item = PlaneState<'a>>,
        event: bool,
    ) -> Result<(), Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let mut current = self.state.try_write().map_err(|_| Error::AtomicRequestBusy)?;
        let mut used_planes = self
            .used_planes
            .try_lock()
            .map_err(|_| Error::AtomicRequestBusy)?;
        let pending = self.pending.try_read().map_err(|_| Error::AtomicRequestBusy)?;
        let mapping = self
            .prop_mapping
            .try_read()
            .map_err(|_| Error::AtomicRequestBusy)?;
        let mut storage = self
            .request_storage
            .try_lock()
            .map_err(|_| Error::AtomicRequestBusy)?;
        let mut last_out_fence = self
            .last_out_fence
            .try_lock()
            .map_err(|_| Error::AtomicRequestBusy)?;
        if pending.connectors.len() > current.connectors.capacity() {
            return Err(Error::AtomicRequestCapacity {
                resource: "DRM committed connectors",
                required: pending.connectors.len(),
                capacity: current.connectors.capacity(),
            });
        }
        debug!(current = ?*current, pending = ?*pending, "Preparing Commit");
        let mut req = AtomicRequest::prepared(&mapping, &mut storage);
        req.fill_request(
            self.crtc,
            pending.mode_id(),
            pending.vrr,
            &pending.connectors,
            current.connectors.difference(&pending.connectors),
            planes,
        )?;
        if let Err(err) = req.commit(
            &*self.fd,
            AtomicCommitFlags::ALLOW_MODESET | AtomicCommitFlags::TEST_ONLY,
        ) {
            warn!("New screen configuration invalid: {:?}: {}", req, err);
            return Err(Error::TestFailed(self.crtc));
        }
        debug!("Setting screen: {:?}", req);
        let mut out_fence_fd = -1;
        let requested_out_fence = req.set_crtc_out_fence_ptr(self.crtc, &mut out_fence_fd)?;
        let flags = if event {
            // Keep the existing blocking modeset contract: amdgpu may fail with NONBLOCK here.
            AtomicCommitFlags::PAGE_FLIP_EVENT | AtomicCommitFlags::ALLOW_MODESET
        } else {
            AtomicCommitFlags::ALLOW_MODESET
        };
        let result = req.commit(&*self.fd, flags).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Error setting crtc",
                dev: self.fd.dev_path(),
                source,
            })
        });
        if result.is_ok() {
            *last_out_fence = consume_out_fence(requested_out_fence, out_fence_fd);
            current.active = pending.active;
            current.mode = pending.mode;
            current.vrr = pending.vrr;
            current.blob.clone_from(&pending.blob);
            current.connectors.clear();
            current.connectors.extend(pending.connectors.iter().copied());
            for &(plane, configured) in req.plane_edits() {
                if configured {
                    used_planes.insert(plane);
                } else {
                    used_planes.remove(&plane);
                }
            }
        } else {
            let _ = consume_out_fence(requested_out_fence, out_fence_fd);
        }
        result
    }

    #[instrument(level = "trace", parent = &self.span, skip(self, planes))]
    #[profiling::function]
    pub fn page_flip<'a>(
        &self,
        planes: impl IntoIterator<Item = PlaneState<'a>>,
        event: bool,
    ) -> Result<(), Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let mut used_planes = self
            .used_planes
            .try_lock()
            .map_err(|_| Error::AtomicRequestBusy)?;
        let mapping = self
            .prop_mapping
            .try_read()
            .map_err(|_| Error::AtomicRequestBusy)?;
        let state = self.state.try_read().map_err(|_| Error::AtomicRequestBusy)?;
        let mut storage = self
            .request_storage
            .try_lock()
            .map_err(|_| Error::AtomicRequestBusy)?;
        let mut last_out_fence = self
            .last_out_fence
            .try_lock()
            .map_err(|_| Error::AtomicRequestBusy)?;
        let mut req = AtomicRequest::prepared(&mapping, &mut storage);
        req.fill_request(self.crtc, None, state.vrr, [], [], planes)?;
        let mut out_fence_fd = -1;
        let requested_out_fence = req.set_crtc_out_fence_ptr(self.crtc, &mut out_fence_fd)?;
        trace!("Queueing page flip: {:?}", req);
        let flags = if event {
            AtomicCommitFlags::PAGE_FLIP_EVENT | AtomicCommitFlags::NONBLOCK
        } else {
            AtomicCommitFlags::NONBLOCK
        };
        let result = req.commit(&*self.fd, flags).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Page flip commit failed",
                dev: self.fd.dev_path(),
                source,
            })
        });
        if result.is_ok() {
            *last_out_fence = consume_out_fence(requested_out_fence, out_fence_fd);
            for &(plane, configured) in req.plane_edits() {
                if configured {
                    used_planes.insert(plane);
                } else {
                    used_planes.remove(&plane);
                }
            }
        } else {
            let _ = consume_out_fence(requested_out_fence, out_fence_fd);
        }
        result
    }

    // this helper function disconnects the plane.
    // this is mostly used to remove the contents quickly, e.g. on tty switch,
    // as other compositors might not make use of other planes,
    // leaving our e.g. cursor or overlays as a relict of a better time on the screen.
    pub fn clear_plane(&self, plane: plane::Handle) -> Result<(), Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let mut used_planes = self
            .used_planes
            .try_lock()
            .map_err(|_| Error::AtomicRequestBusy)?;
        let mapping = self
            .prop_mapping
            .try_read()
            .map_err(|_| Error::AtomicRequestBusy)?;
        let mut storage = self
            .request_storage
            .try_lock()
            .map_err(|_| Error::AtomicRequestBusy)?;
        let mut req = AtomicRequest::prepared(&mapping, &mut storage);
        req.reset_plane(plane)?;
        let result = req
            .commit(&*self.fd, AtomicCommitFlags::empty())
            .map_err(|source| {
                Error::Access(AccessError {
                    errmsg: "Failed to commit on clear_plane",
                    dev: self.fd.dev_path(),
                    source,
                })
            });
        if result.is_ok() {
            used_planes.remove(&plane);
        }
        result
    }
    #[profiling::function]
    fn clear_state(&self) -> Result<(), Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }
        let _guard = self.span.enter();
        // Every owning guard is acquired before even a partial native request is dispatched.
        let mut current = self.state.try_write().map_err(|_| Error::AtomicRequestBusy)?;
        let mut used_planes = self
            .used_planes
            .try_lock()
            .map_err(|_| Error::AtomicRequestBusy)?;
        let mapping = self
            .prop_mapping
            .try_read()
            .map_err(|_| Error::AtomicRequestBusy)?;
        let mut storage = self
            .request_storage
            .try_lock()
            .map_err(|_| Error::AtomicRequestBusy)?;
        let mut req = AtomicRequest::prepared(&mapping, &mut storage);
        for plane in used_planes.iter() {
            req.reset_plane(*plane)?;
        }
        for connector in current.connectors.iter() {
            req.reset_connector(*connector)?;
        }
        req.reset_crtc(self.crtc)?;
        let result = req
            .commit(&*self.fd, AtomicCommitFlags::ALLOW_MODESET)
            .map_err(|source| {
                Error::Access(AccessError {
                    errmsg: "Failed to commit on clear_state",
                    dev: self.fd.dev_path(),
                    source,
                })
            });
        if result.is_ok() {
            used_planes.clear();
            current.clear();
        }
        result
    }

    /// Re-read the current state from KMS. The re-read state owns no mode
    /// blob, so a reset creates none; the replaced state's blob is destroyed
    /// unless the pending state still names it.
    pub(crate) fn reset_state<B: DevPath + ControlDevice + 'static>(
        &self,
        fd: Option<&B>,
    ) -> Result<(), Error> {
        let mut state = if let Some(fd) = fd {
            State::current_state(fd, self.crtc, &mut self.prop_mapping.write().unwrap())?
        } else {
            State::current_state(&*self.fd, self.crtc, &mut self.prop_mapping.write().unwrap())?
        };
        let capacity = self.request_storage.lock().unwrap().connector_capacity();
        state
            .connectors
            .reserve(capacity.saturating_sub(state.connectors.len()));
        *self.state.write().unwrap() = state;
        Ok(())
    }

    pub(crate) fn device_fd(&self) -> &DrmDeviceFd {
        self.fd.device_fd()
    }

    pub fn clear(&self) -> Result<(), Error> {
        self.clear_state()
    }

    pub fn take_out_fence(&self) -> Option<OwnedFd> {
        self.last_out_fence.lock().unwrap().take()
    }
}

fn consume_out_fence(requested: bool, out_fence_fd: i32) -> Option<OwnedFd> {
    if !requested || out_fence_fd < 0 {
        return None;
    }

    Some(unsafe { OwnedFd::from_raw_fd(out_fence_fd) })
}

struct TestBuffer {
    fd: Arc<DrmDeviceInternal>,
    db: DumbBuffer,
    fb: framebuffer::Handle,
}

impl AsRef<framebuffer::Handle> for TestBuffer {
    fn as_ref(&self) -> &framebuffer::Handle {
        &self.fb
    }
}

impl Drop for TestBuffer {
    fn drop(&mut self) {
        let _ = self.fd.destroy_framebuffer(self.fb);
        let _ = self.fd.destroy_dumb_buffer(self.db);
    }
}

impl Drop for AtomicDrmSurface {
    fn drop(&mut self) {
        if !self.active.load(Ordering::SeqCst) {
            // the device is gone or we are on another tty
            // old state has been restored, we shouldn't touch it.
            // if we are on another tty the connectors will get disabled
            // by the device, when switching back
            return;
        }

        let _guard = self.span.enter();
        if let Err(err) = self.clear_state() {
            warn!("Unable to clear state: {}", err);
        }
    }
}

#[inline]
fn to_fixed<N: Coordinate>(n: N) -> u32 {
    f64::round(n.to_f64() * (1 << 16) as f64) as u32
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct DrmRotation: u8 {
        const ROTATE_0      =   0b00000001;
        const ROTATE_90     =   0b00000010;
        const ROTATE_180    =   0b00000100;
        const ROTATE_270    =   0b00001000;
        const REFLECT_X     =   0b00010000;
        const REFLECT_Y     =   0b00100000;
    }
}

impl From<Transform> for DrmRotation {
    #[inline]
    fn from(transform: Transform) -> Self {
        match transform {
            Transform::Normal => DrmRotation::ROTATE_0,
            Transform::_90 => DrmRotation::ROTATE_90,
            Transform::_180 => DrmRotation::ROTATE_180,
            Transform::_270 => DrmRotation::ROTATE_270,
            Transform::Flipped => DrmRotation::REFLECT_Y,
            Transform::Flipped90 => DrmRotation::REFLECT_Y | DrmRotation::ROTATE_90,
            Transform::Flipped180 => DrmRotation::REFLECT_Y | DrmRotation::ROTATE_180,
            Transform::Flipped270 => DrmRotation::REFLECT_Y | DrmRotation::ROTATE_270,
        }
    }
}

/// The mode-blob ownership of [`State`] through the transitions an
/// [`AtomicDrmSurface`] performs: `new` (current read back, pending with its
/// own blob), `use_mode` (`set_mode`, or a dropped blob when the test commit
/// fails), `commit` (`*current = pending.clone()`), `reset_state` (current
/// replaced by a state read back from KMS, which owns no blob), `clear` and
/// drop. The device records each blob's destroys instead of reaching a kernel.
#[cfg(test)]
mod mode_blob_ownership {
    use std::collections::HashSet;

    use drm::control::{property, Mode};

    use super::State;
    use crate::backend::drm::surface::mode_blob::{
        live_mode_blobs,
        test_device::{serial, BlobLedgerDevice},
        ModeBlob,
    };

    fn mode(hdisplay: u16) -> Mode {
        Mode::from(drm_ffi::drm_mode_modeinfo {
            hdisplay,
            // SAFETY: drm_mode_modeinfo is plain old data; zero is valid.
            ..unsafe { std::mem::zeroed() }
        })
    }

    fn blob(device: &BlobLedgerDevice, mode: &Mode) -> ModeBlob {
        ModeBlob::new(device, mode).unwrap()
    }

    fn blob_id(state: &State) -> Option<u64> {
        state.blob.as_ref().map(ModeBlob::id)
    }

    /// The state `State::current_state` builds from KMS, through the same
    /// constructor.
    fn read_back(mode: Mode) -> State {
        State::read_back(mode, true, false, HashSet::new())
    }

    /// `commit` after the kernel accepted the request.
    fn commit(current: &mut State, pending: &State) {
        *current = pending.clone();
    }

    /// `reset_state`.
    fn reset(current: &mut State) {
        *current = read_back(current.mode);
    }

    /// `AtomicDrmSurface::new` followed by its first successful commit.
    fn committed_surface(device: &BlobLedgerDevice, mode: Mode) -> (State, State) {
        let pending = State::with_mode(mode, blob(device, &mode), &[]);
        // `new` reads the current state back from KMS, owning no blob; the
        // commit replaces it with a clone of pending.
        assert_eq!(blob_id(&read_back(mode)), None);
        (pending.clone(), pending)
    }

    #[test]
    fn prepared_pending_adoption_retains_displaced_blob_until_packet_disposal() {
        let _serial = serial();
        let device = BlobLedgerDevice::new();
        let (current, mut pending) = committed_surface(&device, mode(1920));
        let expected_current = current.clone();
        let expected_pending = pending.clone();
        let old_blob = blob_id(&pending).unwrap();
        let mut candidate = State::with_mode(mode(2560), blob(&device, &mode(2560)), &[]);
        let new_blob = blob_id(&candidate).unwrap();
        let (_, heap) = crate::backend::renderer::storage_heap_probe::measure(|| {
            assert!(super::adopt_pending_mode(
                &current,
                &mut pending,
                &expected_current,
                &expected_pending,
                &mut candidate
            ));
        });
        assert_eq!(heap, [0; 4]);
        assert_eq!(blob_id(&pending), Some(new_blob));
        assert_eq!(blob_id(&candidate), Some(old_blob));
        assert_eq!(device.destroys(old_blob), 0);
        drop((current, expected_current, expected_pending));
        assert_eq!(
            device.destroys(old_blob),
            0,
            "displaced packet remains a native owner"
        );
        drop(candidate);
        assert_eq!(device.destroys(old_blob), 1);
        assert_eq!(device.destroys(new_blob), 0);
    }
    #[test]
    fn same_logical_mode_with_replaced_blob_rejects_stale_preparation() {
        let _serial = serial();
        let device = BlobLedgerDevice::new();
        let (current, expected) = committed_surface(&device, mode(1920));
        let mut pending = State::with_mode(mode(1920), blob(&device, &mode(1920)), &[]);
        let mut candidate = State::with_mode(mode(2560), blob(&device, &mode(2560)), &[]);
        let live_blob = blob_id(&pending);
        let candidate_blob = blob_id(&candidate);
        assert_eq!(
            pending, expected,
            "ordinary State equality deliberately omits blob identity"
        );
        assert!(!super::adopt_pending_mode(
            &current,
            &mut pending,
            &current,
            &expected,
            &mut candidate
        ));
        assert_eq!(blob_id(&pending), live_blob);
        assert_eq!(blob_id(&candidate), candidate_blob);
    }
    #[test]
    fn changed_committed_owner_rejects_candidate_without_pending_mutation() {
        let _serial = serial();
        let device = BlobLedgerDevice::new();
        let (expected_current, mut pending) = committed_surface(&device, mode(1920));
        let expected_pending = pending.clone();
        let current = read_back(mode(1920));
        let mut candidate = State::with_mode(mode(2560), blob(&device, &mode(2560)), &[]);
        let old_blob = blob_id(&pending);
        assert!(!super::adopt_pending_mode(
            &current,
            &mut pending,
            &expected_current,
            &expected_pending,
            &mut candidate
        ));
        assert_eq!(blob_id(&pending), old_blob);
    }

    #[test]
    fn resets_and_commits_leave_only_the_pending_blob_alive() {
        let _serial = serial();
        let start = live_mode_blobs();
        let device = BlobLedgerDevice::new();
        let (mut current, pending) = committed_surface(&device, mode(1920));
        let pending_blob = blob_id(&pending).unwrap();

        for _ in 0..64 {
            reset(&mut current);
            assert_eq!(blob_id(&current), None);
            commit(&mut current, &pending);
        }

        assert_eq!(device.alive(), 1);
        assert_eq!(live_mode_blobs(), start + 1);
        assert_eq!(device.destroys(pending_blob), 0);
        assert_eq!(pending.mode_id(), Some(property::Value::Blob(pending_blob)));

        drop(current);
        drop(pending);
        assert_eq!(device.destroys(pending_blob), 1);
        assert_eq!(live_mode_blobs(), start);
    }

    #[test]
    fn a_mode_change_destroys_the_old_blob_exactly_once() {
        let _serial = serial();
        let device = BlobLedgerDevice::new();
        let (mut current, mut pending) = committed_surface(&device, mode(1920));
        let old = blob_id(&pending).unwrap();

        pending.set_mode(mode(1280), blob(&device, &mode(1280)));
        let new = blob_id(&pending).unwrap();
        // The committed state still names the old blob.
        assert_eq!(device.destroys(old), 0);

        commit(&mut current, &pending);
        assert_eq!(device.destroys(old), 1);
        assert_eq!(device.destroys(new), 0);

        reset(&mut current);
        assert_eq!(device.destroys(new), 0);
        commit(&mut current, &pending);
        assert_eq!(device.destroys(old), 1);
        assert_eq!(device.alive(), 1);
        drop((current, pending));
        assert_eq!(device.destroys(new), 1);
    }

    #[test]
    fn a_failed_test_commit_destroys_its_blob() {
        let _serial = serial();
        let device = BlobLedgerDevice::new();
        let (current, pending) = committed_surface(&device, mode(1920));
        let kept = blob_id(&pending).unwrap();

        // `use_mode` returns the test commit's error before `set_mode`.
        let candidate = blob(&device, &mode(1280));
        let candidate_id = candidate.id();
        drop(candidate);

        assert_eq!(device.destroys(candidate_id), 1);
        assert_eq!(blob_id(&pending), Some(kept));
        assert_eq!(device.destroys(kept), 0);
        drop((current, pending));
        assert_eq!(device.alive(), 0);
    }

    #[test]
    fn clear_then_drop_never_destroys_twice() {
        let _serial = serial();
        let device = BlobLedgerDevice::new();

        // Current shares pending's blob.
        let (mut current, pending) = committed_surface(&device, mode(1920));
        let shared = blob_id(&pending).unwrap();
        current.clear();
        assert_eq!(device.destroys(shared), 0);
        drop((current, pending));
        assert_eq!(device.destroys(shared), 1);

        // Current owns the blob pending has moved away from.
        let (mut current, mut pending) = committed_surface(&device, mode(1920));
        let old = blob_id(&current).unwrap();
        pending.set_mode(mode(1280), blob(&device, &mode(1280)));
        current.clear();
        assert_eq!(device.destroys(old), 1);
        drop((current, pending));
        assert_eq!(device.destroys(old), 1);
        assert_eq!(device.alive(), 0);
    }

    #[test]
    fn dropping_the_surface_states_destroys_every_blob() {
        let _serial = serial();
        let start = live_mode_blobs();
        let device = BlobLedgerDevice::new();
        let (current, mut pending) = committed_surface(&device, mode(1920));
        let old = blob_id(&current).unwrap();
        pending.set_mode(mode(1280), blob(&device, &mode(1280)));
        let new = blob_id(&pending).unwrap();
        assert_eq!(live_mode_blobs(), start + 2);

        drop((current, pending));
        assert_eq!(device.destroys(old), 1);
        assert_eq!(device.destroys(new), 1);
        assert_eq!(live_mode_blobs(), start);
    }

    #[test]
    fn using_the_current_mode_then_committing_leaves_one_live_blob() {
        let _serial = serial();
        let device = BlobLedgerDevice::new();
        let (mut current, mut pending) = committed_surface(&device, mode(1920));
        let old = blob_id(&pending).unwrap();

        pending.set_mode(mode(1920), blob(&device, &mode(1920)));
        commit(&mut current, &pending);

        assert_eq!(device.destroys(old), 1);
        assert_eq!(device.alive(), 1);
        drop((current, pending));
        assert_eq!(device.alive(), 0);
    }

    #[test]
    fn using_a_mode_twice_before_commit_destroys_the_superseded_pending_blob() {
        let _serial = serial();
        let device = BlobLedgerDevice::new();
        let (mut current, mut pending) = committed_surface(&device, mode(1920));
        let committed = blob_id(&pending).unwrap();

        pending.set_mode(mode(1280), blob(&device, &mode(1280)));
        let superseded = blob_id(&pending).unwrap();
        pending.set_mode(mode(800), blob(&device, &mode(800)));

        assert_eq!(device.destroys(superseded), 1);
        assert_eq!(device.destroys(committed), 0);
        assert_eq!(device.alive(), 2);

        commit(&mut current, &pending);
        assert_eq!(device.destroys(committed), 1);
        assert_eq!(device.alive(), 1);
        drop((current, pending));
        assert_eq!(device.alive(), 0);
    }
}

#[cfg(test)]
mod test {
    use crate::{
        backend::drm::surface::atomic::to_fixed,
        utils::{Physical, Rectangle},
    };

    use super::AtomicDrmSurface;

    fn is_send<S: Send>() {}

    #[test]
    fn surface_is_send() {
        is_send::<AtomicDrmSurface>();
    }

    #[test]
    fn test_fixed_point() {
        let geometry: Rectangle<f64, Physical> = Rectangle::from_size((1920.0, 1080.0).into());
        let fixed = to_fixed(geometry.size.w) as u64;
        assert_eq!(125829120, fixed);
    }

    #[test]
    fn test_fractional_fixed_point() {
        let geometry: Rectangle<f64, Physical> = Rectangle::from_size((1920.1, 1080.0).into());
        let fixed = to_fixed(geometry.size.w) as u64;
        assert_eq!(125835674, fixed);
    }
}

/// Exercise the real atomic request builder, without a DRM device: a
/// planeless security frame keeps the exact mode active and resets every
/// primary/cursor/overlay property. The configuration TEST_ONLY request and
/// the actual full modeset use this builder; an ordinary page flip intentionally
/// omits connector and MODE_ID properties and disallows modesetting.
#[cfg(test)]
mod planeless_black_request {
    use super::*;
    use std::num::NonZeroU32;
    fn handle<T: From<NonZeroU32>>(id: u32) -> T {
        NonZeroU32::new(id).unwrap().into()
    }
    #[test]
    fn exact_active_mode_and_every_owned_plane_are_in_the_request() {
        let crtc = handle(2);
        let connector = handle(3);
        let mut mapping = PropMapping::default();
        mapping.crtcs.insert(
            crtc,
            [("ACTIVE".into(), handle(101)), ("MODE_ID".into(), handle(102))].into(),
        );
        mapping
            .connectors
            .insert(connector, [("CRTC_ID".into(), handle(103))].into());
        let names = [
            "CRTC_ID", "FB_ID", "SRC_X", "SRC_Y", "SRC_W", "SRC_H", "CRTC_X", "CRTC_Y", "CRTC_W", "CRTC_H",
        ];
        let planes: Vec<_> = [10, 11, 12]
            .into_iter()
            .map(|id| {
                let plane = handle(id);
                mapping.planes.insert(
                    plane,
                    names
                        .into_iter()
                        .enumerate()
                        .map(|(index, name)| (name.into(), handle(200 + index as u32)))
                        .collect(),
                );
                PlaneState {
                    handle: plane,
                    config: None,
                }
            })
            .collect();
        let req = AtomicRequest::build_request(
            &mapping,
            crtc,
            Some(property::Value::Blob(77)),
            false,
            [&connector],
            [],
            &planes,
        )
        .unwrap();
        assert_eq!(req.value(crtc, "ACTIVE"), Some(1));
        assert_eq!(req.value(crtc, "MODE_ID"), Some(77));
        assert_eq!(req.value(connector, "CRTC_ID"), Some(2));
        assert_eq!(req.plane_edits().len(), 3);
        for plane in &planes {
            for name in names {
                assert_eq!(req.value(plane.handle, name), Some(0));
            }
        }
    }
}
