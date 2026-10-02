//! One host-synchronization and submission-order domain for a shared VkQueue.
use std::sync::Mutex;

#[derive(Debug, Default)]
pub(super) struct OrderedQueue(Mutex<u64>);

#[derive(Debug)]
pub(super) enum QueueError<E> {
    Poisoned,
    Exhausted,
    Backend(E),
}

impl OrderedQueue {
    pub(super) fn submit<E>(&self, submit: impl FnOnce() -> Result<(), E>) -> Result<u64, QueueError<E>> {
        let mut next = self.0.lock().map_err(|_| QueueError::Poisoned)?;
        let successor = next.checked_add(1).ok_or(QueueError::Exhausted)?;
        submit().map_err(QueueError::Backend)?;
        let id = *next;
        *next = successor;
        Ok(id)
    }

    pub(super) fn access<T, E>(&self, operation: impl FnOnce() -> Result<T, E>) -> Result<T, QueueError<E>> {
        let _access = self.0.lock().map_err(|_| QueueError::Poisoned)?;
        operation().map_err(QueueError::Backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[test]
    fn failed_submissions_do_not_create_completion_identities() {
        let queue = OrderedQueue::default();
        assert!(matches!(
            queue.submit(|| Err("rejected")),
            Err(QueueError::Backend("rejected"))
        ));
        assert_eq!(queue.submit(|| Ok::<_, ()>(())).unwrap(), 0);
        assert_eq!(queue.submit(|| Ok::<_, ()>(())).unwrap(), 1);
        queue.access(|| Ok::<_, ()>(())).unwrap();
        assert_eq!(queue.submit(|| Ok::<_, ()>(())).unwrap(), 2);
    }

    #[test]
    fn output_workers_share_exact_queue_order_and_never_enter_concurrently() {
        let queue = Arc::new(OrderedQueue::default());
        let entered = Arc::new(AtomicUsize::new(0));
        let order = Arc::new(Mutex::new(Vec::new()));
        let workers = (0..8)
            .map(|_| {
                let queue = queue.clone();
                let entered = entered.clone();
                let order = order.clone();
                std::thread::spawn(move || {
                    (0..32)
                        .map(|_| {
                            queue
                                .submit(|| {
                                    assert_eq!(entered.fetch_add(1, Ordering::SeqCst), 0);
                                    std::thread::yield_now();
                                    order.lock().unwrap().push(());
                                    assert_eq!(entered.fetch_sub(1, Ordering::SeqCst), 1);
                                    Ok::<_, ()>(())
                                })
                                .unwrap()
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        let mut ids = workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        ids.sort_unstable();
        assert_eq!(ids, (0..256).collect::<Vec<_>>());
        assert_eq!(order.lock().unwrap().len(), 256);
    }

    #[test]
    fn exhausted_identity_domain_submits_no_untracked_gpu_work() {
        let queue = OrderedQueue(Mutex::new(u64::MAX));
        let entered = AtomicUsize::new(0);
        assert!(matches!(
            queue.submit(|| {
                entered.fetch_add(1, Ordering::Relaxed);
                Ok::<_, ()>(())
            }),
            Err(QueueError::Exhausted)
        ));
        assert_eq!(entered.load(Ordering::Relaxed), 0);
    }
}
