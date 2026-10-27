use super::{ffi, CleanupResource, FramebufferCleanup, GlesCleanup, GlesError};
use crate::backend::egl::EGLError;
use std::{cell::RefCell, ffi::c_void, sync::Arc};

#[derive(Default)]
struct Calls {
    error: u32,
    deletion_error: u32,
    textures: Vec<u32>,
}

thread_local! {
    static CALLS: RefCell<Calls> = RefCell::new(Calls::default());
}

extern "system" fn delete_textures(count: i32, textures: *const u32) {
    assert_eq!(count, 1);
    CALLS.with_borrow_mut(|calls| {
        calls.textures.push(unsafe { *textures });
        calls.error = calls.deletion_error;
    });
}

extern "system" fn get_error() -> u32 {
    CALLS.with_borrow_mut(|calls| std::mem::take(&mut calls.error))
}

fn mock_gl(calls: Calls) -> ffi::Gles2 {
    CALLS.with_borrow_mut(|state| *state = calls);
    ffi::Gles2::load_with(|name| match name {
        "glDeleteTextures" => delete_textures as *const c_void,
        "glDeleteFramebuffers" => delete_textures as *const c_void,
        "glGetError" => get_error as *const c_void,
        _ => std::ptr::null(),
    })
}

#[test]
fn failed_native_delete_is_observed_and_never_retried() {
    let cleanup = GlesCleanup::default();
    cleanup.sender.send(CleanupResource::Texture(17)).unwrap();
    cleanup.sender.send(CleanupResource::Texture(18)).unwrap();
    let gl = mock_gl(Calls {
        deletion_error: ffi::INVALID_OPERATION,
        ..Calls::default()
    });
    let error = cleanup.cleanup_with(&gl, false, |_| unreachable!()).unwrap_err();
    assert!(error.is_completion_unobservable());
    assert!(!error.is_device_lost());
    assert!(matches!(
        cleanup.cleanup_with(&gl, true, |_| unreachable!()),
        Err(GlesError::UnobservableCompletion)
    ));
    CALLS.with_borrow(|calls| assert_eq!(calls.textures, [17]));
}

#[test]
fn failed_context_local_delete_cannot_resume_teardown() {
    let cleanup = FramebufferCleanup::default();
    cleanup.sender.send(17).unwrap();
    cleanup.sender.send(18).unwrap();
    let gl = mock_gl(Calls {
        deletion_error: ffi::INVALID_OPERATION,
        ..Calls::default()
    });
    assert!(cleanup.cleanup(&gl).is_err());
    assert!(matches!(
        cleanup.cleanup(&gl),
        Err(GlesError::UnobservableCompletion)
    ));
    CALLS.with_borrow(|calls| assert_eq!(calls.textures, [17]));
}

#[test]
fn failed_egl_image_delete_stops_share_group_cleanup() {
    let cleanup = GlesCleanup::default();
    cleanup
        .sender
        .send(CleanupResource::EGLImage(std::ptr::dangling_mut()))
        .unwrap();
    cleanup.sender.send(CleanupResource::Texture(18)).unwrap();
    let gl = mock_gl(Calls::default());
    let error = cleanup
        .cleanup_with(&gl, false, |_| Err(EGLError::BadParameter))
        .unwrap_err();
    assert!(matches!(error, GlesError::NativeCleanup(EGLError::BadParameter)));
    assert!(error.is_completion_unobservable());
    assert!(!error.is_device_lost());
    assert!(GlesError::NativeCleanup(EGLError::ContextLost).is_device_lost());
    assert!(cleanup.cleanup_with(&gl, true, |_| unreachable!()).is_err());
    CALLS.with_borrow(|calls| assert!(calls.textures.is_empty()));
}

#[test]
fn prior_native_error_prevents_cleanup_in_uncertain_context() {
    let cleanup = GlesCleanup::default();
    cleanup.sender.send(CleanupResource::Texture(17)).unwrap();
    let gl = mock_gl(Calls {
        error: ffi::CONTEXT_LOST,
        ..Calls::default()
    });
    assert!(cleanup
        .cleanup_with(&gl, false, |_| unreachable!())
        .unwrap_err()
        .is_device_lost());
    CALLS.with_borrow(|calls| assert!(calls.textures.is_empty()));
}

#[test]
fn cold_final_drain_survives_concurrent_queue_ownership() {
    let cleanup = Arc::new(GlesCleanup::default());
    cleanup.sender.send(CleanupResource::Texture(17)).unwrap();
    let held = cleanup.receiver.lock().unwrap();
    let gl = mock_gl(Calls::default());
    cleanup.cleanup_with(&gl, false, |_| unreachable!()).unwrap();
    CALLS.with_borrow(|calls| assert!(calls.textures.is_empty()));

    let (started, ready) = std::sync::mpsc::channel();
    let worker = {
        let cleanup = cleanup.clone();
        std::thread::spawn(move || {
            let gl = mock_gl(Calls::default());
            started.send(()).unwrap();
            cleanup.cleanup_with(&gl, true, |_| unreachable!()).unwrap();
            CALLS.with_borrow(|calls| assert_eq!(calls.textures, [17]));
        })
    };
    ready.recv().unwrap();
    drop(held);
    worker.join().unwrap();
}

#[test]
fn unwound_cleanup_cannot_resume_native_deletion() {
    let cleanup = GlesCleanup::default();
    let _ = std::panic::catch_unwind(|| {
        let _held = cleanup.receiver.lock().unwrap();
        panic!("interrupted native cleanup");
    });
    cleanup.sender.send(CleanupResource::Texture(17)).unwrap();
    let gl = mock_gl(Calls::default());
    assert!(matches!(
        cleanup.cleanup_with(&gl, false, |_| unreachable!()),
        Err(GlesError::UnobservableCompletion)
    ));
    CALLS.with_borrow(|calls| assert!(calls.textures.is_empty()));
}
