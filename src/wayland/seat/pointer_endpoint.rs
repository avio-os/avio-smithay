//! Endpoint identity outlives wl_pointer when a derived protocol object does.
use super::PointerUserData;
use crate::input::{pointer::PointerHandle, Seat, SeatHandler};
use std::sync::{Arc, Mutex, Weak};
use wayland_server::{backend::ClientId, protocol::wl_surface::WlSurface, Client, Resource};

#[derive(Debug)]
pub(crate) struct PointerEndpoint<D: SeatHandler> {
    handle: Mutex<Option<PointerHandle<D>>>,
    client: ClientId,
    constraints: Mutex<Vec<wayland_server::Weak<WlSurface>>>,
}
impl<D: SeatHandler> std::ops::Deref for PointerEndpoint<D> {
    type Target = Mutex<Option<PointerHandle<D>>>;
    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}
struct Endpoints<D: SeatHandler>(Mutex<Vec<Weak<PointerEndpoint<D>>>>);
impl<D: SeatHandler + 'static> PointerEndpoint<D> {
    pub(crate) fn new(declared: &Seat<D>, client: &Client, handle: Option<PointerHandle<D>>) -> Arc<Self> {
        declared
            .user_data()
            .insert_if_missing(|| Endpoints::<D>(Mutex::new(Vec::new())));
        let endpoint = Arc::new(Self {
            handle: Mutex::new(handle),
            client: client.id(),
            constraints: Mutex::new(Vec::new()),
        });
        let mut endpoints = declared
            .user_data()
            .get::<Endpoints<D>>()
            .unwrap()
            .0
            .lock()
            .unwrap();
        endpoints.retain(|weak| weak.strong_count() > 0);
        endpoints.push(Arc::downgrade(&endpoint));
        endpoint
    }
    pub(crate) fn track_constraint(&self, surface: &WlSurface) {
        let mut surfaces = self.constraints.lock().unwrap();
        surfaces.retain(|weak| weak.upgrade().is_ok());
        if !surfaces.iter().any(|weak| weak.upgrade().as_ref() == Ok(surface)) {
            surfaces.push(surface.downgrade());
        }
    }
}
pub(crate) fn route_endpoints<D: SeatHandler + 'static>(
    from: &Seat<D>,
    to: &Seat<D>,
    declared: &Seat<D>,
    client: &Client,
) {
    let (Some(from), Some(to)) = (from.get_pointer(), to.get_pointer()) else {
        return;
    };
    let Some(endpoints) = declared.user_data().get::<Endpoints<D>>() else {
        return;
    };
    for endpoint in endpoints.0.lock().unwrap().iter().filter_map(Weak::upgrade) {
        if endpoint.client != client.id() || endpoint.lock().unwrap().as_ref() != Some(&from) {
            continue;
        }
        let mut moved = Vec::new();
        from.wl_pointer.known_pointers.lock().unwrap().retain(|weak| {
            let Ok(resource) = weak.upgrade() else {
                return false;
            };
            if Arc::ptr_eq(&resource.data::<PointerUserData<D>>().unwrap().handle, &endpoint) {
                moved.push(weak.clone());
                false
            } else {
                true
            }
        });
        to.wl_pointer.known_pointers.lock().unwrap().extend(moved);
        from.wp_relative.route_endpoint(&to.wp_relative, &endpoint);
        from.wp_pointer_gestures
            .route_endpoint(&to.wp_pointer_gestures, &endpoint);
        for surface in endpoint
            .constraints
            .lock()
            .unwrap()
            .iter()
            .filter_map(|weak| weak.upgrade().ok())
        {
            crate::wayland::pointer_constraints::route_pointer_constraint(&surface, &from, &to, &endpoint);
        }
        *endpoint.lock().unwrap() = Some(to.clone());
    }
}

pub(crate) fn client_has_endpoint<D: SeatHandler + 'static>(seat: &Seat<D>, client: &Client) -> bool {
    seat.user_data().get::<Endpoints<D>>().is_some_and(|endpoints| {
        endpoints
            .0
            .lock()
            .unwrap()
            .iter()
            .filter_map(Weak::upgrade)
            .any(|endpoint| endpoint.client == client.id() && endpoint.lock().unwrap().is_some())
    })
}
