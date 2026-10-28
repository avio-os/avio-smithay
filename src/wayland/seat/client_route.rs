//! Reversible protocol-resource custody for clients which bind only one wl_seat.
use super::{KeyboardUserData, WaylandFocus};
use crate::input::{Seat, SeatHandler, WeakSeat};
use crate::utils::Serial;
use crate::wayland::{input_method::InputMethodSeat, text_input::TextInputSeat};
use std::sync::{Arc, Mutex};
use wayland_server::{backend::ClientId, Client, Resource};
mod passive_focus;

struct Entry<D: SeatHandler> {
    client: ClientId,
    target: WeakSeat<D>,
    identity: Arc<()>,
}
struct Routes<D: SeatHandler> {
    entries: Mutex<Vec<Entry<D>>>,
}
impl<D: SeatHandler> Default for Routes<D> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }
}

/// Owns protocol routing, never permission, task identity, focus or a popup grant.
///
/// The caller must retire held input and logical focus before dropping this
/// lease. Drop only moves registrations; it never locks either keyboard's
/// internal state or generates a physical enter. This permits retirement from
/// inside the physical seat's enter callback, before its normal event delivery.
#[derive(Debug)]
pub struct ClientInputRouteLease<D: SeatHandler + 'static> {
    source: Seat<D>,
    target: Seat<D>,
    client: Client,
    identity: Arc<()>,
}

impl<D: SeatHandler + 'static> ClientInputRouteLease<D> {
    /// Restore the client's physical protocol focus after its logical holds
    /// have been retired. Call outside either physical seat handle's lock.
    /// The physical focus and pressed-key authority are unchanged.
    pub fn release(self, data: &mut D, serial: Serial, time: u32)
    where
        D::KeyboardFocus: WaylandFocus,
        D::PointerFocus: WaylandFocus,
    {
        let source = self.source.clone();
        let client = self.client.clone();
        drop(self);
        passive_focus::resume(&source, data, &client, serial, time);
    }

    /// The exact connection whose declared first-seat resources are routed.
    pub fn client(&self) -> &Client {
        &self.client
    }
}

impl<D: SeatHandler + 'static> Seat<D> {
    /// Return the logical input seat selected for this exact connection.
    /// Declared wl_seat identity and non-input protocols remain unchanged.
    pub fn input_route_for_client(&self, client: &Client) -> Option<Seat<D>> {
        let routes = self.user_data().get::<Routes<D>>()?;
        routes
            .entries
            .lock()
            .unwrap()
            .iter()
            .find(|e| e.client == client.id())?
            .target
            .upgrade()
    }

    /// Route this client's keyboard, pointer and text-input resources to an
    /// existing logical seat. New bindings follow the same route until Drop.
    /// Admission and exact-root authorization remain the compositor's job.
    pub fn route_client_input(
        &self,
        client: &Client,
        target: &Self,
    ) -> Result<ClientInputRouteLease<D>, &'static str>
    where
        D::KeyboardFocus: WaylandFocus,
        D::PointerFocus: WaylandFocus,
    {
        self.route_client_input_inner(client, target, None)
    }

    /// Lease an idle client's input resources even when it has passive physical
    /// focus. The compositor must suppress passive physical protocol delivery
    /// while routed and release the lease before actual human content input.
    /// Held keys/grabs remain exclusive. No physical focus is moved.
    pub fn route_client_input_with_passive_focus(
        &self,
        data: &mut D,
        client: &Client,
        target: &Self,
        serial: Serial,
        time: u32,
    ) -> Result<ClientInputRouteLease<D>, &'static str>
    where
        D::KeyboardFocus: WaylandFocus,
        D::PointerFocus: WaylandFocus,
    {
        self.route_client_input_inner(client, target, Some((data, serial, time)))
    }

    fn route_client_input_inner(
        &self,
        client: &Client,
        target: &Self,
        passive: Option<(&mut D, Serial, u32)>,
    ) -> Result<ClientInputRouteLease<D>, &'static str>
    where
        D::KeyboardFocus: WaylandFocus,
        D::PointerFocus: WaylandFocus,
    {
        if self == target || target.input_route_for_client(client).is_some() {
            return Err("invalid or chained input route");
        }
        let keyboard = self.get_keyboard().ok_or("source keyboard unavailable")?;
        let pointer = self.get_pointer().ok_or("source pointer unavailable")?;
        let target_keyboard = target.get_keyboard().ok_or("target keyboard unavailable")?;
        let target_pointer = target.get_pointer().ok_or("target pointer unavailable")?;
        if keyboard.arc.keymap.lock().unwrap().id() != target_keyboard.arc.keymap.lock().unwrap().id() {
            return Err("input route keymaps differ");
        }
        let matches = |surface: Option<wayland_server::protocol::wl_surface::WlSurface>| {
            surface.is_some_and(|s| s.client().as_ref() == Some(client))
        };
        let keyboard_focused = matches(
            keyboard
                .current_focus()
                .and_then(|f| f.wl_surface().map(|s| s.into_owned())),
        );
        let pointer_focused = matches(
            pointer
                .current_focus()
                .and_then(|f| f.wl_surface().map(|s| s.into_owned())),
        );
        let held_keys = keyboard.has_forwarded_pressed_keys();
        if (keyboard_focused && (passive.is_none() || held_keys))
            || (pointer_focused && passive.is_none())
            || matches(
                keyboard
                    .grab_start_data()
                    .and_then(|g| g.focus)
                    .and_then(|f| f.wl_surface().map(|s| s.into_owned())),
            )
            || matches(
                pointer
                    .grab_start_data()
                    .and_then(|g| g.focus)
                    .and_then(|(f, _)| f.wl_surface().map(|s| s.into_owned())),
            )
        {
            return Err("physical input owns client");
        }
        if super::pointer_endpoint::client_has_endpoint(target, client)
            || target_keyboard.client_keyboards(client).next().is_some()
            || target_pointer.client_pointers(client).next().is_some()
            || !target.text_input().client_text_inputs(client).is_empty()
        {
            return Err("target already has client bindings");
        }
        self.user_data().insert_if_missing(Routes::<D>::default);
        let identity = Arc::new(());
        {
            let mut entries = self
                .user_data()
                .get::<Routes<D>>()
                .unwrap()
                .entries
                .lock()
                .unwrap();
            if entries.iter().any(|e| e.client == client.id()) {
                return Err("client input already routed");
            }
            entries.push(Entry {
                client: client.id(),
                target: target.downgrade(),
                identity: identity.clone(),
            });
        }
        // Validate and reserve first; suspension cannot fail. Resources have
        // not moved yet, so these leave events reach their original endpoint.
        if let Some((data, serial, time)) = passive {
            passive_focus::suspend(self, data, client, serial, time);
        }
        move_keyboards(self, target, self, client);
        move_pointers(self, target, self, client);
        self.text_input().route_client_instances(
            target.text_input(),
            self.text_input(),
            client,
            target.input_method().clone(),
        );
        Ok(ClientInputRouteLease {
            source: self.clone(),
            target: target.clone(),
            client: client.clone(),
            identity,
        })
    }
}

fn move_keyboards<D: SeatHandler + 'static>(
    from: &Seat<D>,
    to: &Seat<D>,
    declared: &Seat<D>,
    client: &Client,
) {
    let (Some(from), Some(to)) = (from.get_keyboard(), to.get_keyboard()) else {
        return;
    };
    let mut moved = Vec::new();
    from.arc.known_kbds.lock().unwrap().retain(|weak| {
        let Ok(resource) = weak.upgrade() else {
            return false;
        };
        let data = resource.data::<KeyboardUserData<D>>().unwrap();
        if resource.client().as_ref() == Some(client)
            && data.declared_seat.upgrade().as_ref() == Some(declared)
        {
            *data.handle.lock().unwrap() = Some(to.clone());
            moved.push(weak.clone());
            false
        } else {
            true
        }
    });
    // The compositor keeps the physical/logical keymaps synchronized. Moving
    // registrations is deliberately silent: ordinary seat focus emits enter,
    // modifiers and repeat state without impersonating a physical interaction.
    to.arc.known_kbds.lock().unwrap().extend(moved);
}

fn move_pointers<D: SeatHandler + 'static>(
    from: &Seat<D>,
    to: &Seat<D>,
    declared: &Seat<D>,
    client: &Client,
) {
    super::pointer_endpoint::route_endpoints(from, to, declared, client);
}

impl<D: SeatHandler + 'static> Drop for ClientInputRouteLease<D> {
    fn drop(&mut self) {
        let Some(routes) = self.source.user_data().get::<Routes<D>>() else {
            return;
        };
        let owned = {
            let mut entries = routes.entries.lock().unwrap();
            let old = entries.len();
            entries.retain(|e| !Arc::ptr_eq(&e.identity, &self.identity));
            old != entries.len()
        };
        if !owned {
            return;
        }
        move_keyboards(&self.target, &self.source, &self.source, &self.client);
        move_pointers(&self.target, &self.source, &self.source, &self.client);
        self.target.text_input().route_client_instances(
            self.source.text_input(),
            self.source.text_input(),
            &self.client,
            self.source.input_method().clone(),
        );
    }
}
