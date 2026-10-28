//! Protocol focus is suspended independently of physical seat focus.
use super::*;
use crate::input::{
    keyboard::KeyboardTarget,
    pointer::{MotionEvent, PointerTarget},
};

pub(super) fn suspend<D: SeatHandler + 'static>(
    seat: &Seat<D>,
    data: &mut D,
    client: &Client,
    serial: Serial,
    time: u32,
) where
    D::KeyboardFocus: WaylandFocus,
    D::PointerFocus: WaylandFocus,
{
    if let Some(surface) = seat
        .get_keyboard()
        .and_then(|k| k.current_focus())
        .and_then(|f| f.wl_surface().map(|s| s.into_owned()))
        .filter(|s| s.client().as_ref() == Some(client))
    {
        // Call the underlying protocol target: the compositor may already
        // suppress passive delivery because the route has been reserved.
        KeyboardTarget::leave(&surface, seat, data, serial);
    }
    if let Some(surface) = seat
        .get_pointer()
        .and_then(|p| p.current_focus())
        .and_then(|f| f.wl_surface().map(|s| s.into_owned()))
        .filter(|s| s.client().as_ref() == Some(client))
    {
        PointerTarget::leave(&surface, seat, data, serial, time);
        PointerTarget::frame(&surface, seat, data);
    }
}

pub(super) fn resume<D: SeatHandler + 'static>(
    seat: &Seat<D>,
    data: &mut D,
    client: &Client,
    serial: Serial,
    time: u32,
) where
    D::KeyboardFocus: WaylandFocus,
    D::PointerFocus: WaylandFocus,
{
    if let Some(keyboard) = seat.get_keyboard() {
        let snapshot = {
            let mut inner = keyboard.arc.internal.lock().unwrap();
            let surface = inner
                .focus
                .as_ref()
                .and_then(|(focus, _)| focus.wl_surface().map(|s| s.into_owned()))
                .filter(|s| s.client().as_ref() == Some(client));
            surface.map(|surface| {
                if let Some((_, entered)) = inner.focus.as_mut() {
                    *entered = serial;
                }
                (surface, inner.forwarded_pressed_keys.clone(), inner.mods_state)
            })
        };
        if let Some((surface, keys, mods)) = snapshot {
            // Only forwarded holds belong in enter, never a just-intercepted
            // takeover press or a consumed compositor shortcut.
            super::super::keyboard::enter_internal(&surface, seat, data, keys.into_iter(), serial);
            KeyboardTarget::modifiers(&surface, seat, data, mods, serial);
        }
    }
    if let Some(pointer) = seat.get_pointer() {
        let focus = pointer.inner.lock().unwrap().focus.clone();
        if let Some((focus, origin)) =
            focus.filter(|(focus, _)| focus.wl_surface().and_then(|s| s.client()).as_ref() == Some(client))
        {
            focus.enter(
                seat,
                data,
                &MotionEvent {
                    location: pointer.current_location() - origin,
                    serial,
                    time,
                },
            );
            focus.frame(seat, data);
        }
    }
}
