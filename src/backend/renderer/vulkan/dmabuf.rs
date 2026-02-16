use crate::backend::allocator::dmabuf::WeakDmabuf;

#[derive(Debug, Clone)]
pub(crate) struct CachedDmabuf {
    pub(crate) handle: WeakDmabuf,
}
