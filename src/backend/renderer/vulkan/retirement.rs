//! Device-owned, event-driven destruction. A resource reserves its queue node
//! when it is created; final drop only publishes that node and unparks the
//! executor. No driver operation, allocation, mutex, or completion wait runs there.

use std::{
    marker::PhantomData,
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicPtr, Ordering},
        Arc,
    },
    thread,
};

pub(super) struct RetirementNode<T> {
    value: T,
    next: AtomicPtr<Self>,
}

impl<T> RetirementNode<T> {
    pub(super) fn value(&self) -> &T {
        &self.value
    }

    pub(super) fn next_detached(&self) -> *mut Self {
        self.next.load(Ordering::Relaxed)
    }

    pub(super) fn set_next_detached(&self, next: *mut Self) {
        self.next.store(next, Ordering::Relaxed);
    }

    pub(super) fn into_value(self: Box<Self>) -> T {
        self.value
    }
    pub(super) fn value_mut(&mut self) -> &mut T {
        &mut self.value
    }

    pub(super) fn new(value: T) -> Box<Self> {
        Box::new(Self {
            value,
            next: AtomicPtr::new(ptr::null_mut()),
        })
    }
}

struct Shared<T> {
    head: AtomicPtr<RetirementNode<T>>,
    closed: AtomicBool,
    _owns: PhantomData<Box<RetirementNode<T>>>,
}

// SAFETY: Producers touch only atomics and their own unpublished node. One
// consumer detaches the stack with acquire ordering before reading any T.
// Sharing this ownership-transfer container requires Send, not Sync, from T.
unsafe impl<T: Send> Sync for Shared<T> {}

/// One producer endpoint, shared through the device handle rather than cloned.
/// The executor owns no endpoint: closing the final device handle therefore
/// drains all queued children and then destroys the device, without an Arc cycle.
pub(super) struct RetirementQueue<T> {
    shared: Arc<Shared<T>>,
    executor: thread::Thread,
}

impl<T: Send + 'static> RetirementQueue<T> {
    pub(super) fn start(
        mut destroy: impl FnMut(T) + Send + 'static,
        finish: impl FnOnce() + Send + 'static,
    ) -> std::io::Result<Self> {
        Self::start_with_nodes(move |node| destroy(node.into_value()), finish)
    }

    /// A native owner may return this exact cold-allocated node to its fixed
    /// bank after disposal, avoiding a release-path allocation or replacement.
    pub(super) fn start_with_nodes(
        mut destroy: impl FnMut(Box<RetirementNode<T>>) + Send + 'static,
        finish: impl FnOnce() + Send + 'static,
    ) -> std::io::Result<Self> {
        let shared = Arc::new(Shared {
            head: AtomicPtr::new(ptr::null_mut()),
            closed: AtomicBool::new(false),
            _owns: PhantomData,
        });
        let executor = shared.clone();
        // There is no join in a final resource drop: it may occur on Wayland or
        // frame/input work. Endpoint closure is the deterministic shutdown edge.
        let worker = thread::Builder::new()
            .name("vulkan-retire".into())
            .spawn(move || {
                loop {
                    // Read closure before detaching: observing closed proves
                    // every producer already published its last node. Reading
                    // it after an empty detach could miss a concurrent final
                    // publication and prematurely destroy the logical device.
                    let closed = executor.closed.load(Ordering::Acquire);
                    let mut batch = executor.head.swap(ptr::null_mut(), Ordering::Acquire);
                    if batch.is_null() {
                        if closed {
                            break;
                        }
                        // Unpark carries a permit. A publication between the
                        // detach and park cannot lose its idle wake.
                        thread::park();
                        continue;
                    }
                    // Reverse the detached producer stack once. FIFO native
                    // disposal makes a cold drain marker an exact edge for
                    // everything published before that marker.
                    let mut fifo = ptr::null_mut();
                    while !batch.is_null() {
                        // SAFETY: Detach grants exclusive list ownership.
                        let next = unsafe { (*batch).next.load(Ordering::Relaxed) };
                        unsafe { (*batch).next.store(fifo, Ordering::Relaxed) };
                        fifo = batch;
                        batch = next;
                    }
                    batch = fifo;
                    while !batch.is_null() {
                        // SAFETY: Atomic detach transferred exclusive ownership
                        // of the entire list. Each node was Box::into_raw once,
                        // and each is reconstructed exactly once here.
                        let node = unsafe { Box::from_raw(batch) };
                        batch = node.next.swap(ptr::null_mut(), Ordering::Relaxed);
                        destroy(node);
                    }
                }
                finish();
            })?;
        Ok(Self {
            shared,
            executor: worker.thread().clone(),
        })
    }
}

impl<T> RetirementQueue<T> {
    pub(super) fn worker_thread(&self) -> thread::Thread {
        self.executor.clone()
    }

    pub(super) fn retire(&self, node: Box<RetirementNode<T>>) {
        let node = Box::into_raw(node);
        let mut head = self.shared.head.load(Ordering::Relaxed);
        loop {
            // SAFETY: This producer still exclusively owns the unpublished
            // node. A failed CAS publishes nothing and permits updating its
            // link; a successful release-CAS transfers ownership permanently.
            unsafe {
                (*node).next.store(head, Ordering::Relaxed);
            }
            match self
                .shared
                .head
                .compare_exchange_weak(head, node, Ordering::Release, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(current) => head = current,
            }
        }
        // Closing requires exclusive endpoint drop; a borrowed endpoint cannot
        // race it. Enqueue cannot fail or use an inline driver-free fallback.
        self.executor.unpark();
    }
}

impl<T> Drop for RetirementQueue<T> {
    fn drop(&mut self) {
        self.shared.closed.store(true, Ordering::Release);
        self.executor.unpark();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn idle_executor_retires_on_its_thread_before_device_shutdown() {
        let (events, received) = mpsc::channel();
        let retired = events.clone();
        let queue = RetirementQueue::start(
            move |id| retired.send((id, thread::current().id())).unwrap(),
            move || events.send((0, thread::current().id())).unwrap(),
        )
        .unwrap();
        let caller = thread::current().id();
        queue.retire(RetirementNode::new(1));
        // No endpoint closure, frame, import, poll or timeout initiates this.
        let (id, executor) = received.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(id, 1);
        assert_ne!(executor, caller);
        queue.retire(RetirementNode::new(2));
        queue.retire(RetirementNode::new(3));
        drop(queue);
        let mut ids = Vec::new();
        loop {
            let (id, thread) = received.recv_timeout(Duration::from_secs(2)).unwrap();
            assert_eq!(thread, executor);
            if id == 0 {
                break;
            }
            ids.push(id);
        }
        ids.sort_unstable();
        assert_eq!(ids, [2, 3]);
    }

    #[test]
    fn blocked_driver_does_not_block_enqueue_or_endpoint_drop() {
        let (entered, started) = mpsc::channel();
        let (resume, stalled) = mpsc::channel();
        let (events, received) = mpsc::channel();
        let retired = events.clone();
        let queue = RetirementQueue::start(
            move |id| {
                if id == 1 {
                    entered.send(()).unwrap();
                    stalled.recv().unwrap();
                }
                retired.send(id).unwrap();
            },
            move || events.send(0).unwrap(),
        )
        .unwrap();
        queue.retire(RetirementNode::new(1));
        started.recv_timeout(Duration::from_secs(2)).unwrap();
        // Returning before resume proves no driver wait is under the lock.
        let (enqueued, accepted) = mpsc::channel();
        let producer = thread::spawn(move || {
            queue.retire(RetirementNode::new(2));
            drop(queue);
            enqueued.send(()).unwrap();
        });
        accepted.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(received.try_recv().is_err());
        resume.send(()).unwrap();
        producer.join().unwrap();
        assert_eq!(received.recv_timeout(Duration::from_secs(2)).unwrap(), 1);
        assert_eq!(received.recv_timeout(Duration::from_secs(2)).unwrap(), 2);
        assert_eq!(received.recv_timeout(Duration::from_secs(2)).unwrap(), 0);
    }

    #[test]
    fn concurrent_producers_transfer_every_preallocated_node_before_shutdown() {
        let (events, received) = mpsc::channel();
        let retired = events.clone();
        let queue = Arc::new(
            RetirementQueue::start(
                move |id| retired.send(id).unwrap(),
                move || events.send(0).unwrap(),
            )
            .unwrap(),
        );
        let mut producers = Vec::new();
        for producer in 0..8 {
            let queue = queue.clone();
            producers.push(thread::spawn(move || {
                for index in 0..100 {
                    queue.retire(RetirementNode::new(producer * 100 + index + 1));
                }
            }));
        }
        drop(queue);
        for producer in producers {
            producer.join().unwrap();
        }
        let mut ids = Vec::new();
        loop {
            let id = received.recv_timeout(Duration::from_secs(2)).unwrap();
            if id == 0 {
                break;
            }
            ids.push(id);
        }
        ids.sort_unstable();
        assert_eq!(ids, (1..=800).collect::<Vec<_>>());
    }
}
