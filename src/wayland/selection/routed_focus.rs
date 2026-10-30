//! First-seat resource borrowing withdraws selection authority, not its source.
use super::{seat_data::SeatData, SelectionHandler};
use crate::input::Seat;
use std::cell::RefCell;
use wayland_server::{Client, DisplayHandle};

pub(crate) fn suspend<D: SelectionHandler + 'static>(seat: &Seat<D>, client: &Client) {
    if let Some(data) = seat.user_data().get::<RefCell<SeatData<D::SelectionUserData>>>() {
        data.borrow_mut().suspend_client(client);
    }
}

pub(crate) fn resume<D: SelectionHandler + 'static>(dh: &DisplayHandle, seat: &Seat<D>, client: &Client) {
    // The caller has restored routing and verified the actual physical focus.
    if let Some(data) = seat.user_data().get::<RefCell<SeatData<D::SelectionUserData>>>() {
        let mut data = data.borrow_mut();
        data.set_clipboard_focus::<D>(dh, Some(client.clone()));
        data.set_primary_focus::<D>(dh, Some(client.clone()));
    }
}
