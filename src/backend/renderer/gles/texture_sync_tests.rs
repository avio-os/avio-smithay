use super::{ffi, texture::*, GlesError};
use std::{cell::RefCell, ffi::c_void};

#[derive(Default)]
struct Calls {
    query: u32,
    error: u32,
    next_fence: usize,
    events: Vec<&'static str>,
}

thread_local! {
    static CALLS: RefCell<Calls> = RefCell::new(Calls::default());
}

extern "system" fn client_wait(_sync: *const c_void, _flags: u32, timeout: u64) -> u32 {
    assert_eq!(timeout, 0, "readiness queries must never wait on the CPU");
    CALLS.with_borrow_mut(|state| {
        state.events.push("query");
        state.query
    })
}
extern "system" fn wait(_sync: *const c_void, _flags: u32, timeout: u64) {
    assert_eq!(timeout, ffi::TIMEOUT_IGNORED);
    CALLS.with_borrow_mut(|state| state.events.push("gpu_wait"));
}
extern "system" fn fence(_condition: u32, _flags: u32) -> *const c_void {
    CALLS.with_borrow_mut(|state| {
        state.events.push("fence");
        state.next_fence as *const c_void
    })
}
extern "system" fn delete(_sync: *const c_void) {
    CALLS.with_borrow_mut(|state| state.events.push("delete"));
}
extern "system" fn flush() {
    CALLS.with_borrow_mut(|state| state.events.push("flush"));
}
extern "system" fn error() -> u32 {
    CALLS.with_borrow_mut(|state| std::mem::take(&mut state.error))
}

fn gl(calls: Calls) -> ffi::Gles2 {
    CALLS.with_borrow_mut(|state| *state = calls);
    ffi::Gles2::load_with(|name| match name {
        "glClientWaitSync" => client_wait as *const c_void,
        "glWaitSync" => wait as *const c_void,
        "glFenceSync" => fence as *const c_void,
        "glDeleteSync" => delete as *const c_void,
        "glFlush" => flush as *const c_void,
        "glGetError" => error as *const c_void,
        _ => std::ptr::null(),
    })
}

#[test]
fn failed_readiness_query_keeps_dependency_without_wait_or_deletion() {
    let gl = gl(Calls {
        query: ffi::WAIT_FAILED,
        error: ffi::INVALID_VALUE,
        ..Calls::default()
    });
    let sync = TextureSync::default();
    *sync.write_sync.lock().unwrap() = Some(1 as _);
    assert!(matches!(sync.wait_for_upload(&gl), Err(GlesError::GlSync { .. })));
    assert_eq!(*sync.write_sync.lock().unwrap(), Some(1 as _));
    CALLS.with_borrow(|state| assert_eq!(state.events, ["query"]));
}

#[test]
fn pending_texture_uses_gpu_wait_without_blocking_or_discarding_fence() {
    let gl = gl(Calls {
        query: ffi::TIMEOUT_EXPIRED,
        ..Calls::default()
    });
    let sync = TextureSync::default();
    *sync.write_sync.lock().unwrap() = Some(1 as _);
    sync.wait_for_upload(&gl).unwrap();
    assert_eq!(*sync.write_sync.lock().unwrap(), Some(1 as _));
    CALLS.with_borrow(|state| assert_eq!(state.events, ["query", "gpu_wait"]));
}

#[test]
fn null_replacement_fence_keeps_prior_dependency() {
    let gl = gl(Calls::default());
    let mut sync = TextureSync::default();
    *sync.write_sync.get_mut().unwrap() = Some(1 as _);
    assert!(matches!(sync.update_write(&gl), Err(GlesError::GlSync { .. })));
    assert_eq!(*sync.write_sync.get_mut().unwrap(), Some(1 as _));
    CALLS.with_borrow(|state| assert_eq!(state.events, ["gpu_wait", "fence"]));
}

#[test]
fn replacement_wait_precedes_fence_and_old_handle_retirement() {
    let gl = gl(Calls {
        next_fence: 2,
        ..Calls::default()
    });
    let mut sync = TextureSync::default();
    *sync.write_sync.get_mut().unwrap() = Some(1 as _);
    sync.update_write(&gl).unwrap();
    assert_eq!(*sync.write_sync.get_mut().unwrap(), Some(2 as _));
    CALLS.with_borrow(|state| assert_eq!(state.events, ["gpu_wait", "fence", "delete", "flush"]));
}

#[test]
fn poisoned_texture_final_drop_never_panics_or_publishes_native_deletion() {
    use std::sync::{mpsc, Arc, RwLock};
    for poison_inner in [false, true] {
        let (sender, receiver) = mpsc::channel();
        let texture = GlesTexture(Arc::new(GlesTextureInternal {
            context: super::ContextId::new(),
            encoding: GlesTextureEncoding::ElectricalPremultiplied,
            storage_is_srgb: false,
            is_composition: false,
            retirement: None,
            texture: 17,
            sync: RwLock::new(TextureSync::default()),
            format: Some(ffi::RGBA8),
            has_alpha: true,
            is_external: false,
            y_inverted: false,
            size: (1, 1).into(),
            egl_images: Some(vec![std::ptr::dangling_mut()]),
            destruction_callback_sender: sender,
        }));
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if poison_inner {
                let sync = texture.0.sync.read().unwrap();
                let _held = sync.read_sync.lock().unwrap();
                panic!("unwound native read synchronization");
            } else {
                let _held = texture.0.sync.write().unwrap();
                panic!("unwound native framebuffer write");
            }
        }));
        assert!(panic.is_err());
        drop(texture);
        assert!(
            receiver.try_recv().is_err(),
            "uncertain native handles escaped quarantine"
        );
    }
}
