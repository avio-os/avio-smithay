//! CPU writer completion is independent from the GPU submission fence.

use std::{
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

type Wake = Arc<dyn Fn() + Send + Sync>;

/// Exact CPU writers whose returned rows permit an upload-owner retry.
///
/// This snapshot retains only readiness signals, never source pixels, mapped
/// storage or Vulkan objects. A ready signal permits nonblocking collection of
/// returned staged rows; it does not submit their upload or complete GPU work.
#[derive(Clone)]
pub struct MemoryUploadCpuCompletion {
    signals: Vec<Arc<MemoryUploadCpuSignal>>,
}

impl fmt::Debug for MemoryUploadCpuCompletion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryUploadCpuCompletion")
            .field("writers", &self.signals.len())
            .field("ready", &self.is_ready())
            .finish()
    }
}

impl MemoryUploadCpuCompletion {
    pub(crate) fn new(signals: impl IntoIterator<Item = Arc<MemoryUploadCpuSignal>>) -> Self {
        Self {
            signals: signals.into_iter().collect(),
        }
    }

    /// Whether every writer captured by this completion has returned its rows.
    pub fn is_ready(&self) -> bool {
        self.signals.iter().all(|signal| signal.is_ready())
    }

    /// Notify once after every captured writer returns, including if they
    /// returned before registration. Install accepted-work custody before
    /// registering: a completion that is already ready invokes this inline.
    /// The callback may run on any writer thread and must only enqueue work.
    pub fn on_ready(&self, wake: Wake) {
        if self.signals.is_empty() {
            wake();
            return;
        }
        let remaining = Arc::new(AtomicUsize::new(self.signals.len()));
        let one_returned: Wake = Arc::new(move || {
            if remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                wake();
            }
        });
        for signal in &self.signals {
            signal.on_ready(one_returned.clone());
        }
    }
}

#[derive(Default)]
pub(crate) struct MemoryUploadCpuSignal {
    ready: AtomicBool,
    observers: Mutex<Vec<Wake>>,
}

impl MemoryUploadCpuSignal {
    pub(crate) fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    fn on_ready(&self, wake: Wake) {
        let mut observers = self.observers.lock().unwrap();
        if self.is_ready() {
            drop(observers);
            wake();
        } else {
            observers.push(wake);
        }
    }

    /// Called only after the exact row keepalive has been released.
    pub(crate) fn complete(&self) {
        let observers = {
            let mut observers = self.observers.lock().unwrap();
            assert!(
                !self.ready.swap(true, Ordering::Release),
                "CPU writer returns once"
            );
            std::mem::take(&mut *observers)
        };
        for observer in observers {
            observer();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn registration_after_return_fires_and_multiple_writers_wait_whole() {
        let first = Arc::new(MemoryUploadCpuSignal::default());
        let second = Arc::new(MemoryUploadCpuSignal::default());
        let completion = MemoryUploadCpuCompletion::new([first.clone(), second.clone()]);
        let calls = Arc::new(AtomicUsize::new(0));
        first.complete();
        let observed = calls.clone();
        completion.on_ready(Arc::new(move || {
            observed.fetch_add(1, Ordering::Relaxed);
        }));
        assert!(!completion.is_ready());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        second.complete();
        assert!(completion.is_ready());
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        let observed = calls.clone();
        completion.on_ready(Arc::new(move || {
            observed.fetch_add(1, Ordering::Relaxed);
        }));
        assert_eq!(
            calls.load(Ordering::Relaxed),
            2,
            "already ready registration fires once"
        );
    }

    #[test]
    fn concurrent_return_and_registration_never_lose_the_exact_wake() {
        for _ in 0..64 {
            let signal = Arc::new(MemoryUploadCpuSignal::default());
            let completion = MemoryUploadCpuCompletion::new([signal.clone()]);
            let barrier = Arc::new(Barrier::new(2));
            let calls = Arc::new(AtomicUsize::new(0));
            let waiter_barrier = barrier.clone();
            let observed = calls.clone();
            let waiter = std::thread::spawn(move || {
                waiter_barrier.wait();
                completion.on_ready(Arc::new(move || {
                    observed.fetch_add(1, Ordering::Relaxed);
                }));
            });
            barrier.wait();
            signal.complete();
            waiter.join().unwrap();
            assert_eq!(calls.load(Ordering::Relaxed), 1);
        }
    }
}
