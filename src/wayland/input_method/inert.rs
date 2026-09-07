//! Children requested from an unavailable input method must remain inert.
use std::{os::fd::OwnedFd, sync::Arc};
use wayland_server::backend::{protocol::Message, ClientId, Handle, ObjectData, ObjectId};

#[derive(Debug)]
pub(super) struct InertChild;
impl<D: 'static> ObjectData<D> for InertChild {
    fn request(
        self: Arc<Self>,
        _: &Handle,
        _: &mut D,
        _: ClientId,
        _: Message<ObjectId, OwnedFd>,
    ) -> Option<Arc<dyn ObjectData<D>>> {
        None
    }
    fn destroyed(self: Arc<Self>, _: &Handle, _: &mut D, _: ClientId, _: ObjectId) {}
}
