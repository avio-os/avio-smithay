//! Upgrade an existing pointer route, fenced by its original lease identity.
use super::*;

/// A short-lived transition request, not a lease or a source of input authority.
/// The original lease remains the owner throughout protocol callbacks.
#[derive(Debug)]
#[must_use]
pub struct ClientInputRouteKeyboardUpgrade<D: SeatHandler + 'static> {
    pub(super) source: Seat<D>,
    pub(super) target: Seat<D>,
    pub(super) client: Client,
    pub(super) identity: Arc<()>,
}

impl<D: SelectionHandler + 'static> ClientInputRouteKeyboardUpgrade<D>
where
    D::KeyboardFocus: WaylandFocus,
    D::PointerFocus: WaylandFocus,
{
    /// Borrow keyboard/text-input resources for this exact still-live route.
    /// Repeated application is inert; retired or replaced routes fail closed.
    /// Call outside physical seat locks, after compositor content admission.
    pub fn apply(self, data: &mut D, serial: Serial) -> Result<(), &'static str> {
        let routes = self
            .source
            .user_data()
            .get::<Routes<D>>()
            .ok_or("input route retired")?;
        {
            let entries = routes.entries.lock().unwrap();
            let entry = entries
                .iter()
                .find(|e| Arc::ptr_eq(&e.identity, &self.identity))
                .ok_or("input route retired")?;
            if entry.keyboard {
                return Ok(());
            }
        }
        let keyboard = self.source.get_keyboard().ok_or("source keyboard unavailable")?;
        let target_keyboard = self.target.get_keyboard().ok_or("target keyboard unavailable")?;
        if keyboard.arc.keymap.lock().unwrap().id() != target_keyboard.arc.keymap.lock().unwrap().id() {
            return Err("input route keymaps differ");
        }
        let matches = |surface: Option<wayland_server::protocol::wl_surface::WlSurface>| {
            surface.is_some_and(|surface| surface.client().as_ref() == Some(&self.client))
        };
        if (keyboard.has_forwarded_pressed_keys()
            && matches(
                keyboard
                    .current_focus()
                    .and_then(|f| f.wl_surface().map(|s| s.into_owned())),
            ))
            || matches(
                keyboard
                    .grab_start_data()
                    .and_then(|g| g.focus)
                    .and_then(|f| f.wl_surface().map(|s| s.into_owned())),
            )
            || self.source.get_pointer().is_some_and(|pointer| {
                matches(
                    pointer
                        .grab_start_data()
                        .and_then(|g| g.focus)
                        .and_then(|(f, _)| f.wl_surface().map(|s| s.into_owned())),
                )
            })
        {
            return Err("physical input owns client");
        }
        if target_keyboard.client_keyboards(&self.client).next().is_some()
            || !self
                .target
                .text_input()
                .client_text_inputs(&self.client)
                .is_empty()
        {
            return Err("target already has client bindings");
        }
        // Reserve the transition before protocol leave. Never hold the route
        // registry lock across target callbacks or another seat lock.
        {
            let mut entries = routes.entries.lock().unwrap();
            let entry = entries
                .iter_mut()
                .find(|e| Arc::ptr_eq(&e.identity, &self.identity))
                .ok_or("input route retired")?;
            if entry.keyboard {
                return Ok(());
            }
            entry.keyboard = true;
        }
        passive_focus::suspend_keyboard(&self.source, data, &self.client, serial);
        // IME popup dismissal may call back into the compositor and retire the
        // lease. Never move resources after that retirement or its replacement.
        if !routes
            .entries
            .lock()
            .unwrap()
            .iter()
            .any(|e| Arc::ptr_eq(&e.identity, &self.identity))
        {
            if self
                .source
                .keyboard_input_route_for_client(&self.client)
                .is_none()
            {
                passive_focus::resume_keyboard(&self.source, data, &self.client, serial);
            }
            return Err("input route retired during keyboard transition");
        }
        move_keyboards(&self.source, &self.target, &self.source, &self.client);
        self.source.text_input().route_client_instances(
            self.target.text_input(),
            self.source.text_input(),
            &self.client,
            self.target.input_method().clone(),
        );
        // A popup may already own logical focus. Newly transferred keyboards
        // need its enter before any key, without resetting the popup grab.
        passive_focus::resume_keyboard(&self.target, data, &self.client, serial);
        if !routes
            .entries
            .lock()
            .unwrap()
            .iter()
            .any(|e| Arc::ptr_eq(&e.identity, &self.identity))
        {
            return Err("input route retired during keyboard transition");
        }
        Ok(())
    }
}
