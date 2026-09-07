//! Exclusive ownership for a compositor's built-in text-input provider.
use super::{input_method_handle::InputMethodProvider, InputMethodHandle};
use crate::wayland::text_input::TextInputHandle;
use std::sync::Arc;

/// An exclusive seat input-method lease held by the compositor itself.
///
/// This participates in the same owner slot as a Wayland input-method client.
/// It enables the standard text-input focus/enable/commit handshake without a
/// synthetic Wayland peer. Dropping it leaves text input and releases the slot.
#[derive(Debug)]
pub struct InProcessTextInput {
    input_method: InputMethodHandle,
    text_input: TextInputHandle,
    identity: Arc<()>,
}

impl InProcessTextInput {
    pub(super) fn acquire(input_method: InputMethodHandle, text_input: TextInputHandle) -> Option<Self> {
        let identity = Arc::new(());
        {
            let mut inner = input_method.inner.lock().unwrap();
            if inner.provider.is_some() {
                return None;
            }
            inner.provider = Some(InputMethodProvider::InProcess(identity.clone()));
            // Owner transitions serialize their text-input notifications under
            // the input-method lock, before a successor can acquire the seat.
            text_input.enter();
        }
        Some(Self {
            input_method,
            text_input,
            identity,
        })
    }

    /// The standard text-input handle associated with the leased seat.
    pub fn text_input(&self) -> &TextInputHandle {
        &self.text_input
    }
}

impl Drop for InProcessTextInput {
    fn drop(&mut self) {
        let mut inner = self.input_method.inner.lock().unwrap();
        if matches!(&inner.provider, Some(InputMethodProvider::InProcess(identity)) if Arc::ptr_eq(identity, &self.identity))
        {
            inner.provider = None;
            self.text_input.leave();
        }
    }
}
