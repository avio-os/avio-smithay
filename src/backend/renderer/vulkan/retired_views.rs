//! Each renderer observes retirement independently without locking its draw path.
//! Only the fair destruction executor allocates notification nodes. A renderer
//! detaches its own stack and returns every node to that executor for disposal.
use super::retirement::RetirementNode;
use std::{
    ptr,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc, Mutex, Weak,
    },
};

pub(super) struct RetirementSubscribers<T>(Mutex<Vec<Weak<Notifications<T>>>>);
impl<T> Default for RetirementSubscribers<T> {
    fn default() -> Self {
        Self(Mutex::new(Vec::new()))
    }
}
impl<T> std::fmt::Debug for RetirementSubscribers<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetirementSubscribers").finish_non_exhaustive()
    }
}
struct Notifications<T> {
    head: AtomicPtr<RetirementNode<T>>,
}
// SAFETY: Only atomic publication is shared. Exclusive atomic detach transfers
// ownership to one context; the value needs Send, never concurrent access.
unsafe impl<T: Send> Send for Notifications<T> {}
unsafe impl<T: Send> Sync for Notifications<T> {}
impl<T> Notifications<T> {
    fn drain(&self, mut consume: impl FnMut(Box<RetirementNode<T>>)) {
        let mut node = self.head.swap(ptr::null_mut(), Ordering::Acquire);
        while !node.is_null() {
            // SAFETY: Detach transfers each exclusively owned node once. The
            // consumer owns the Box and must return it to the fair executor.
            let owned = unsafe { Box::from_raw(node) };
            node = owned.next_detached();
            consume(owned);
        }
    }
}
impl<T> Drop for Notifications<T> {
    fn drop(&mut self) {
        // DescriptorState itself retires on the fair command executor. A cold
        // context teardown may therefore dispose notifications it never read.
        self.drain(drop);
    }
}

pub(super) struct RetirementSubscription<T>(Arc<Notifications<T>>);
impl<T> std::fmt::Debug for RetirementSubscription<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetirementSubscription").finish_non_exhaustive()
    }
}
impl<T> RetirementSubscription<T> {
    pub(super) fn drain(&self, consume: impl FnMut(Box<RetirementNode<T>>)) {
        self.0.drain(consume);
    }
    #[cfg(test)]
    fn take(&self) -> Vec<T> {
        let mut values = Vec::new();
        self.drain(|node| values.push(node.into_value()));
        values.reverse();
        values
    }
}
impl<T> RetirementSubscribers<T> {
    #[cfg(test)]
    pub(super) fn poison_registry(&self) {
        let _subscribers = self.0.lock().unwrap();
        panic!("injected renderer registry fault");
    }
    pub(super) fn subscribe(&self) -> RetirementSubscription<T> {
        // Called during cold renderer initialization, never on the draw path.
        let queue = Arc::new(Notifications {
            head: AtomicPtr::new(ptr::null_mut()),
        });
        let mut subscribers = self.0.lock().unwrap_or_else(|poison| poison.into_inner());
        subscribers.retain(|subscriber| subscriber.strong_count() != 0);
        subscribers.push(Arc::downgrade(&queue));
        RetirementSubscription(queue)
    }
    pub(super) fn retired(&self, mut make_value: impl FnMut() -> T) {
        // One native executor broadcasts to every live context. Registry locks
        // and node allocation run only here; the contexts touch neither.
        self.0
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .retain(|subscriber| {
                let Some(queue) = subscriber.upgrade() else {
                    return false;
                };
                let node = Box::into_raw(RetirementNode::new(make_value()));
                let mut head = queue.head.load(Ordering::Relaxed);
                loop {
                    // SAFETY: Node is producer-owned until the successful CAS.
                    unsafe {
                        (*node).set_next_detached(head);
                    }
                    match queue
                        .head
                        .compare_exchange_weak(head, node, Ordering::Release, Ordering::Relaxed)
                    {
                        Ok(_) => break,
                        Err(current) => head = current,
                    }
                }
                true
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn one_output_draining_cannot_hide_retired_views_from_another() {
        let device = RetirementSubscribers::<u64>::default();
        let first = device.subscribe();
        let second = device.subscribe();
        device.retired(|| 12);
        assert_eq!(first.take(), [12]);
        assert!(first.take().is_empty());
        assert_eq!(second.take(), [12]);
        device.retired(|| 24);
        assert_eq!(second.take(), [24]);
        assert_eq!(first.take(), [24]);
    }
    #[test]
    fn retired_contexts_are_unsubscribed_without_extending_resource_lifetime() {
        let device = RetirementSubscribers::<u64>::default();
        let first = device.subscribe();
        drop(first);
        let second = device.subscribe();
        assert_eq!(device.0.lock().unwrap().len(), 1);
        device.retired(|| 8);
        assert_eq!(second.take(), [8]);
        drop(second);
        device.retired(|| 9);
        assert!(device.0.lock().unwrap().is_empty());
    }
    #[test]
    fn draw_drain_does_not_wait_for_the_cold_registry_lock() {
        let device = RetirementSubscribers::<u64>::default();
        let output = device.subscribe();
        device.retired(|| 8);
        let _cold_registry = device.0.lock().unwrap();
        assert_eq!(output.take(), [8]);
    }
    #[test]
    fn drain_returns_the_exact_preallocated_node_without_freeing_it() {
        let device = RetirementSubscribers::<u64>::default();
        let output = device.subscribe();
        device.retired(|| 8);
        let published = output.0.head.load(Ordering::Acquire);
        let mut returned = None;
        output.drain(|node| returned = Some(node));
        let returned = returned.unwrap();
        assert_eq!(&*returned as *const _ as *mut _, published);
        assert_eq!(*returned.value(), 8);
        assert!(output.0.head.load(Ordering::Acquire).is_null());
    }
}
