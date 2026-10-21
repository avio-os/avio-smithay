//! One GLES share group, with context-affine construction and cold retirement.
use std::{
    any::Any,
    marker::PhantomData,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex, OnceLock,
    },
};

use super::{GlesError, GlesRenderer, GlesTexture};
use crate::{
    backend::{
        allocator::{format::FormatSet, Fourcc},
        egl::{EGLContext, EGLDisplay},
        renderer::{
            retirement::{RetirementNode, RetirementQueue},
            Offscreen,
        },
    },
    utils::{Buffer, Size},
};

type Custody = Box<dyn Any + Send + Sync>;

#[derive(Debug)]
struct Shared {
    // This root is never made current. Serializing child creation does not
    // transfer a live renderer or any current context across threads.
    root: Mutex<EGLContext>,
    available: AtomicBool,
    operational_2d_formats: OnceLock<FormatSet>,
}

/// Failure to construct a complete cold GLES device owner.
#[derive(Debug, thiserror::Error)]
pub enum GlesDeviceError {
    /// The native EGL/GLES operation failed.
    #[error(transparent)]
    Native(#[from] GlesError),
    /// The retirement executor could not start.
    #[error("could not start GLES retirement executor: {0}")]
    Executor(#[from] std::io::Error),
    /// The retirement executor did not terminate normally during cold setup.
    #[error(transparent)]
    TerminalJoin(#[from] GlesDeviceJoinError),
    /// The executor's native context could not be initialized.
    #[error("could not initialize GLES retirement context: {0}")]
    Initialization(String),
}

/// One cold terminal observer for the actual GLES retirement thread.
///
/// It owns only a join handle, never an origin or producer endpoint. Drop every
/// productive origin, renderer, allocator and retirement slot before joining.
/// Queued custody may publish nested retirement nodes before the final endpoint
/// closes. Joining waits for all of them, native state, closure captures and
/// thread-local destructors. It is not GPU completion or a device-health proof;
/// quarantined owners can prevent closure indefinitely. Use a supervised cold
/// process-terminal path, never a frame/input turn or ordinary resource Drop.
#[derive(Debug)]
pub struct GlesDeviceRetirementJoin {
    worker: std::thread::JoinHandle<()>,
}

/// Failure of an explicit cold retirement-thread join.
#[derive(Debug, thiserror::Error)]
pub enum GlesDeviceJoinError {
    /// Joining from the worker itself is refused without consuming authority.
    #[error("cannot join the GLES retirement thread from itself")]
    CurrentThread(GlesDeviceRetirementJoin),
    /// The worker unwound instead of completing its terminal teardown.
    #[error("GLES retirement thread panicked")]
    WorkerPanicked,
}

impl GlesDeviceRetirementJoin {
    /// Query the actual join handle's readiness for cold registry reaping.
    /// This may become true before thread-local destruction ends. Only join
    /// supplies the thread-termination receipt; neither is GPU completion.
    pub fn is_finished(&self) -> bool {
        self.worker.is_finished()
    }

    /// Wait for actual thread termination after all productive owners close.
    /// A self-worker refusal returns this exact one-shot observer to the caller.
    pub fn join(self) -> Result<(), GlesDeviceJoinError> {
        if self.worker.thread().id() == std::thread::current().id() {
            return Err(GlesDeviceJoinError::CurrentThread(self));
        }
        self.worker
            .join()
            .map_err(|_| GlesDeviceJoinError::WorkerPanicked)
    }
}

/// Sendable authority to create contexts on their actual owning threads.
///
/// The never-current root keeps one exact share group alive. A dedicated
/// executor owns its own context and destroys shared native objects after
/// their protocol has proved completion. No renderer itself is Send.
#[derive(Clone)]
pub struct GlesDeviceOrigin {
    shared: Arc<Shared>,
    retirement: Arc<RetirementQueue<Custody>>,
}

impl std::fmt::Debug for GlesDeviceOrigin {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GlesDeviceOrigin")
            .field("available", &self.shared.available.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

fn renderer_for(shared: &Shared) -> Result<GlesRenderer, GlesError> {
    if !shared.available.load(Ordering::Acquire) {
        return Err(GlesError::UnobservableCompletion);
    }
    let context = {
        let root = shared
            .root
            .lock()
            .map_err(|_| GlesError::UnobservableCompletion)?;
        EGLContext::new_shared(root.display(), &root).map_err(GlesError::ContextCreationError)?
    };
    // SAFETY: This fresh child has never been current on any thread. Its
    // renderer stays on the calling thread for its entire lifetime.
    unsafe { GlesRenderer::new_with_native_fences(context) }
}

impl GlesDeviceOrigin {
    /// Seal exact formats proved by actual import, rendering, sampling and
    /// native-fence operations on a renderer created from this same origin.
    /// Call only after those operations succeed and before admitting workers.
    /// This does not rewrite EGL query results or infer support from a format.
    pub fn seal_operational_2d_formats(
        &self,
        witness: &GlesRenderer,
        formats: FormatSet,
    ) -> Result<(), GlesError> {
        self.check_health()?;
        witness.check_completion_health()?;
        if !witness
            .device_origin()
            .is_some_and(|origin| self.same_device(&origin))
        {
            return Err(GlesError::Composition(
                "operational role witness belongs to another GLES origin",
            ));
        }
        self.shared
            .operational_2d_formats
            .set(formats)
            .map_err(|_| GlesError::Composition("GLES operational roles already sealed"))
    }

    /// The immutable operation-proved 2D roles of this exact share group.
    /// None means admission has not yet sealed a result, including an empty one.
    pub fn operational_2d_formats(&self) -> Option<&FormatSet> {
        self.shared.operational_2d_formats.get()
    }

    /// Create the exact display's share group and initialize its retirement
    /// context. This cold constructor waits only for its own initialization.
    pub fn new(display: &EGLDisplay) -> Result<Self, GlesDeviceError> {
        let (origin, terminal) = Self::new_with_terminal_join(display)?;
        // Existing callers retain nonblocking final-drop behavior. A process
        // terminal owner must instead retain and explicitly join the observer.
        drop(terminal);
        Ok(origin)
    }

    /// Create a cold origin with its one endpoint-free terminal join observer.
    /// Failed retirement-context initialization closes and joins the started
    /// worker before returning an error; no observer escapes a failed setup.
    pub fn new_with_terminal_join(
        display: &EGLDisplay,
    ) -> Result<(Self, GlesDeviceRetirementJoin), GlesDeviceError> {
        let root = EGLContext::new(display).map_err(GlesError::ContextCreationError)?;
        let shared = Arc::new(Shared {
            root: Mutex::new(root),
            available: AtomicBool::new(true),
            operational_2d_formats: OnceLock::new(),
        });
        let initialize_shared = shared.clone();
        let destroy_shared = shared.clone();
        let (initialized, ready) = mpsc::sync_channel(1);
        let mut retirement = RetirementQueue::start_with_state(
            move || match renderer_for(&initialize_shared) {
                Ok(renderer) => {
                    let _ = initialized.send(Ok(()));
                    Some(renderer)
                }
                Err(error) => {
                    initialize_shared.available.store(false, Ordering::Release);
                    let _ = initialized.send(Err(error.to_string()));
                    None
                }
            },
            move |renderer, node| {
                let Some(renderer) = renderer
                    .as_mut()
                    .filter(|_| destroy_shared.available.load(Ordering::Acquire))
                else {
                    // A failure is not reader completion or safe reuse.
                    std::mem::forget(node);
                    return;
                };
                drop(node);
                // Only shareable GL objects enter the shared cleanup queue.
                // Context-local framebuffer cleanup stays with its renderer.
                if renderer.cleanup_retired_resources().is_err() {
                    destroy_shared.available.store(false, Ordering::Release);
                }
            },
            drop,
        )?;
        let terminal = GlesDeviceRetirementJoin {
            worker: retirement
                .take_terminal_join()
                .expect("new retirement worker has one terminal observer"),
        };
        let initialization_error = match ready.recv() {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(error),
            Err(error) => Some(error.to_string()),
        };
        if let Some(error) = initialization_error {
            drop(retirement);
            drop(shared);
            terminal.join()?;
            return Err(GlesDeviceError::Initialization(error));
        }
        Ok((
            Self {
                shared,
                retirement: Arc::new(retirement),
            },
            terminal,
        ))
    }

    /// Construct the worker's strict renderer on the calling owning thread.
    pub fn create_renderer(&self) -> Result<GlesRenderer, GlesError> {
        let mut renderer = renderer_for(&self.shared)?;
        renderer.origin = Some(self.clone());
        Ok(renderer)
    }

    /// Whether two owners name the same immutable share group.
    pub fn same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.shared, &other.shared)
    }

    /// Check the cold native retirement owner's health.
    pub fn check_health(&self) -> Result<(), GlesError> {
        if self.shared.available.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(GlesError::UnobservableCompletion)
        }
    }

    pub(super) fn mark_unavailable(&self) {
        self.shared.available.store(false, Ordering::Release);
    }

    /// Cold allocation authority; each operation creates its helper context
    /// on the calling helper thread and returns only shareable textures.
    pub fn offscreen_allocator(&self) -> GlesOffscreenAllocator {
        GlesOffscreenAllocator(self.clone())
    }

    /// Reserve one final-drop handoff. Completion remains the caller's proof.
    pub fn retirement_slot<T: Send + Sync + 'static>(&self) -> GlesRetirementSlot<T> {
        GlesRetirementSlot {
            origin: self.clone(),
            node: Some(RetirementNode::new(Box::new(None::<T>))),
            value: PhantomData,
        }
    }
}

impl GlesRenderer {
    /// The share-group owner when this renderer was created through an origin.
    /// Legacy standalone renderers deliberately have no cold-device authority.
    pub fn device_origin(&self) -> Option<GlesDeviceOrigin> {
        self.origin.clone()
    }
}

/// A cold-reserved publication to the exact GLES device's retirement context.
pub struct GlesRetirementSlot<T: Send + Sync + 'static> {
    origin: GlesDeviceOrigin,
    node: Option<Box<RetirementNode<Custody>>>,
    value: PhantomData<T>,
}

impl<T: Send + Sync + 'static> std::fmt::Debug for GlesRetirementSlot<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GlesRetirementSlot")
            .field("reserved", &self.node.is_some())
            .finish_non_exhaustive()
    }
}

impl<T: Send + Sync + 'static> GlesRetirementSlot<T> {
    pub(super) fn quarantine(self) {
        self.origin.mark_unavailable();
        // Keep the exact share root and retirement context alive. An unwound
        // native resource owner supplies no completion evidence.
        std::mem::forget(self);
    }

    /// Publish without allocating, locking, waiting, or calling a driver.
    pub fn retire(mut self, value: T) {
        let node = self.node.as_mut().expect("unused retirement publication");
        *node
            .value_mut()
            .downcast_mut::<Option<T>>()
            .expect("immutable retirement custody type") = Some(value);
    }
}

impl<T: Send + Sync + 'static> Drop for GlesRetirementSlot<T> {
    fn drop(&mut self) {
        if let Some(node) = self.node.take() {
            self.origin.retirement.retire(node);
        }
    }
}

/// Share-group native image creation confined to a cold helper turn.
#[derive(Debug, Clone)]
pub struct GlesOffscreenAllocator(GlesDeviceOrigin);

impl GlesOffscreenAllocator {
    /// Allocate the explicit sRGB composition attachment. Memory-source
    /// textures use `create_memory_buffer` and retain their client encoding.
    pub fn create_composition_buffer(
        &self,
        format: Fourcc,
        size: Size<i32, Buffer>,
    ) -> Result<GlesTexture, GlesError> {
        let mut renderer = self.0.create_renderer()?;
        let texture = renderer.create_composition_buffer(format, size)?;
        texture.0.sync.write().unwrap().update_write(&renderer.gl)?;
        Ok(texture)
    }

    /// Allocate a sampleable color attachment and publish its creation to
    /// other contexts through a real shared GL fence.
    pub fn create_buffer(&self, format: Fourcc, size: Size<i32, Buffer>) -> Result<GlesTexture, GlesError> {
        let mut renderer = self.0.create_renderer()?;
        let texture: GlesTexture = renderer.create_buffer(format, size)?;
        texture.0.sync.write().unwrap().update_write(&renderer.gl)?;
        Ok(texture)
    }

    /// Allocate a writable sampled image on a cold helper turn.
    pub fn create_memory_buffer(
        &self,
        format: Fourcc,
        size: Size<i32, Buffer>,
    ) -> Result<GlesTexture, GlesError> {
        self.create_buffer(format, size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn terminal_join_refuses_self_worker_without_losing_the_handle() {
        let (publish, receive) = mpsc::channel();
        let (returned, observer) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let terminal = GlesDeviceRetirementJoin {
                worker: receive.recv().unwrap(),
            };
            match terminal.join() {
                Err(GlesDeviceJoinError::CurrentThread(terminal)) => returned.send(terminal).unwrap(),
                other => panic!("self-worker join did not return its authority: {other:?}"),
            }
        });
        publish.send(worker).unwrap();
        observer
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn terminal_join_reports_worker_panic_as_failure() {
        let terminal = GlesDeviceRetirementJoin {
            worker: std::thread::spawn(|| panic!("worker failure")),
        };
        assert!(matches!(
            terminal.join(),
            Err(GlesDeviceJoinError::WorkerPanicked)
        ));
    }
}
