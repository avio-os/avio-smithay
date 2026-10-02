//! The existing process-lifetime DRM blob actor; only cold factories register roots.
use crate::backend::renderer::element::FrameWorkspaceError;
use std::{
    fmt, io,
    sync::{
        mpsc::{self, SyncSender},
        Mutex, OnceLock,
    },
};

pub(super) trait BlobRegistryRoot: fmt::Debug + Send {
    /// Reconcile exact returned readers; true retains this native/control root.
    fn reconcile(&self) -> bool;
}
#[derive(Debug)]
struct Executor {
    roots: Mutex<Vec<Box<dyn BlobRegistryRoot>>>,
    wake: SyncSender<()>,
}
fn executor() -> Result<&'static Executor, FrameWorkspaceError> {
    static EXECUTOR: OnceLock<Result<Executor, io::Error>> = OnceLock::new();
    EXECUTOR
        .get_or_init(|| {
            let (wake, events) = mpsc::sync_channel(1);
            // The actor accesses the one static registry only after startup publishes it.
            let actor = std::thread::Builder::new()
                .name("drm-blob-retire".into())
                .spawn(move || {
                    while events.recv().is_ok() {
                        #[cfg(test)]
                        let barriers = std::mem::take(&mut *BARRIERS.lock().unwrap());
                        let actor = executor().expect("published DRM blob actor");
                        let mut roots = actor.roots.lock().unwrap_or_else(|p| p.into_inner());
                        roots.retain(|root| root.reconcile());
                        drop(roots);
                        #[cfg(test)]
                        for barrier in barriers {
                            let _ = barrier.send(());
                        }
                    }
                })?;
            // The static sender/registry is the process owner, as in the original actor.
            drop(actor);
            Ok(Executor {
                roots: Mutex::new(Vec::new()),
                wake,
            })
        })
        .as_ref()
        .map_err(|_| FrameWorkspaceError {
            resource: "DRM blob retirement executor",
            required: 1,
            capacity: 0,
        })
}
/// The only allocating operation is cold registration, never reader return.
pub(super) fn register(root: impl BlobRegistryRoot + 'static) -> Result<SyncSender<()>, FrameWorkspaceError> {
    let actor = executor()?;
    actor
        .roots
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .push(Box::new(root));
    let _ = actor.wake.try_send(());
    Ok(actor.wake.clone())
}
#[cfg(test)]
static BARRIERS: Mutex<Vec<mpsc::Sender<()>>> = Mutex::new(Vec::new());
/// Test-only barrier through the real actor, never caller-thread reconciliation.
#[cfg(test)]
pub(super) fn drain() {
    let actor = executor().expect("cold DRM blob actor");
    let (answer, receive) = mpsc::channel();
    BARRIERS.lock().unwrap().push(answer);
    let _ = actor.wake.try_send(());
    receive
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("DRM blob actor drained");
}
