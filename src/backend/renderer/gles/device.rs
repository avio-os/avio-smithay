//! One GLES share group, with context-affine construction and cold retirement.
use std::{
    any::Any,
    marker::PhantomData,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
};

use super::{GlesError, GlesRenderer, GlesTexture};
use crate::{
    backend::{
        allocator::Fourcc,
        egl::{EGLContext, EGLDisplay},
        renderer::{
            retirement::{RetirementNode, RetirementQueue},
            Offscreen, Renderer,
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
    /// The executor's native context could not be initialized.
    #[error("could not initialize GLES retirement context: {0}")]
    Initialization(String),
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
        let root = shared.root.lock().map_err(|_| GlesError::UnobservableCompletion)?;
        EGLContext::new_shared(root.display(), &root).map_err(GlesError::ContextCreationError)?
    };
    // SAFETY: This fresh child has never been current on any thread. Its
    // renderer stays on the calling thread for its entire lifetime.
    unsafe { GlesRenderer::new_with_native_fences(context) }
}

impl GlesDeviceOrigin {
    /// Create the exact display's share group and initialize its retirement
    /// context. This cold constructor waits only for its own initialization.
    pub fn new(display: &EGLDisplay) -> Result<Self, GlesDeviceError> {
        let root = EGLContext::new(display).map_err(GlesError::ContextCreationError)?;
        let shared = Arc::new(Shared {
            root: Mutex::new(root),
            available: AtomicBool::new(true),
        });
        let initialize_shared = shared.clone();
        let destroy_shared = shared.clone();
        let (initialized, ready) = mpsc::sync_channel(1);
        let retirement = RetirementQueue::start_with_state(
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
                let Some(renderer) = renderer.as_mut().filter(|_| {
                    destroy_shared.available.load(Ordering::Acquire)
                }) else {
                    // A failure is not reader completion or safe reuse.
                    std::mem::forget(node);
                    return;
                };
                drop(node);
                // Only shareable GL objects enter the shared cleanup queue.
                // Context-local framebuffer cleanup stays with its renderer.
                if renderer.cleanup_texture_cache().is_err() {
                    destroy_shared.available.store(false, Ordering::Release);
                }
            },
            drop,
        )?;
        match ready.recv() {
            Ok(Ok(())) => Ok(Self {
                shared,
                retirement: Arc::new(retirement),
            }),
            Ok(Err(error)) => Err(GlesDeviceError::Initialization(error)),
            Err(error) => Err(GlesDeviceError::Initialization(error.to_string())),
        }
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
    /// Allocate a sampleable color attachment and publish its creation to
    /// other contexts through a real shared GL fence.
    pub fn create_buffer(&self, format: Fourcc, size: Size<i32, Buffer>) -> Result<GlesTexture, GlesError> {
        let mut renderer = self.0.create_renderer()?;
        let texture: GlesTexture = renderer.create_buffer(format, size)?;
        texture.0.sync.write().unwrap().update_write(&renderer.gl)?;
        Ok(texture)
    }

    /// Allocate a writable sampled image on a cold helper turn.
    pub fn create_memory_buffer(&self, format: Fourcc, size: Size<i32, Buffer>) -> Result<GlesTexture, GlesError> {
        self.create_buffer(format, size)
    }
}
