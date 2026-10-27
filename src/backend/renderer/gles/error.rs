use super::*;
use crate::backend::SwapBuffersError;

#[cfg(feature = "wayland_frontend")]
use wayland_server::protocol::wl_shm;

/// Error returned during rendering using GL ES
#[derive(thiserror::Error, Debug)]
pub enum GlesError {
    /// A composition packet exceeds its admitted format, storage or geometry.
    #[error("GLES composition refused: {0}")]
    Composition(&'static str),
    /// An EGL context could not be created on its owning thread.
    #[error("Failed to create EGL context")]
    ContextCreationError(#[source] crate::backend::egl::Error),
    /// A context-local framebuffer was used by another renderer.
    #[error("Framebuffer belongs to another GLES context")]
    FramebufferContextMismatch,
    /// A shader could not be compiled
    #[error("Failed to compile Shader")]
    ShaderCompileError,
    /// A program could not be linked
    #[error("Failed to link Program")]
    ProgramLinkError,
    /// A framebuffer could not be bound
    #[error("Failed to bind Framebuffer")]
    FramebufferBindingError,
    /// Required GL functions could not be loaded
    #[error("Failed to load GL functions from EGL")]
    GLFunctionLoaderError,
    /// Required GL extension are not supported by the underlying implementation
    #[error("None of the following GL extensions is supported by the underlying GL implementation, at least one is required: {0:?}")]
    GLExtensionNotSupported(&'static [&'static str]),
    /// Required EGL extension are not supported by the underlying implementation
    #[error("None of the following EGL extensions is supported by the underlying implementation, at least one is required: {0:?}")]
    EGLExtensionNotSupported(&'static [&'static str]),
    /// Required GL version is not available by the underlying implementation
    #[error(
        "The OpenGL ES version of the underlying GL implementation is too low, at least required: {0:?}"
    )]
    GLVersionNotSupported(version::GlVersion),
    /// The underlying egl context could not be activated
    #[error("Failed to active egl context")]
    ContextActivationError(#[from] crate::backend::egl::MakeCurrentError),
    ///The given dmabuf could not be converted to an EGLImage for framebuffer use
    #[error("Failed to convert between dmabuf and EGLImage")]
    BindBufferEGLError(#[source] crate::backend::egl::Error),
    /// The given buffer has an unknown pixel format
    #[error("Unknown pixel format")]
    UnknownPixelFormat,
    /// The given buffer has an unsupported pixel format
    #[error("Unsupported pixel format: {0:?}")]
    UnsupportedPixelFormat(Fourcc),
    /// The given buffer has an unknown pixel layout
    #[error("Unsupported pixel layout")]
    UnsupportedPixelLayout,
    /// The given wl buffer has an unsupported pixel format
    #[error("Unsupported wl_shm format: {0:?}")]
    #[cfg(feature = "wayland_frontend")]
    UnsupportedWlPixelFormat(wl_shm::Format),
    /// The given buffer was not accessible
    #[error("Error accessing the buffer ({0:?})")]
    #[cfg(feature = "wayland_frontend")]
    BufferAccessError(crate::wayland::shm::BufferAccessError),
    /// The given egl buffer was not accessible
    #[error("Error accessing the buffer ({0:?})")]
    #[cfg(feature = "wayland_frontend")]
    EGLBufferAccessError(crate::backend::egl::BufferAccessError),
    /// There was an error mapping the buffer
    #[error("Error mapping the buffer")]
    MappingError,
    /// The provided buffer's size did not match the requested one.
    #[error("Error reading buffer, size is too small for the given dimensions")]
    UnexpectedSize,
    /// Unable to determine the size of the framebuffer
    #[error("Error determining the size of the provided framebuffer")]
    UnknownSize,
    /// The blitting operation was unsuccessful
    #[error("Error blitting between framebuffers")]
    BlitError,
    /// An error occured while creating the shader object.
    #[error("An error occured while creating the shader object.")]
    CreateShaderObject,
    /// Uniform was not declared when compiling shader
    #[error("Uniform {0:?} was not declared when compiling the provided shader")]
    UnknownUniform(String),
    /// The provided uniform has a different type then was provided when compiling the shader
    #[error("Uniform with different type (got {provided:?}, expected: {declared:?})")]
    UniformTypeMismatch {
        /// Uniform type that was provided during the call
        provided: UniformType,
        /// Uniform type that was declared when compiling
        declared: UniformType,
    },
    /// Blocking for a synchronization primitive failed
    #[error("Blocking for a synchronization primitive got interrupted")]
    SyncInterrupted,
    /// The required native fence operation failed.
    ///
    /// Submitted work must remain quarantined; this error is not a completion
    /// receipt and the strict renderer refuses further submissions.
    #[error("Required native fence operation failed")]
    NativeFence(#[source] crate::backend::egl::Error),
    /// Native resource deletion failed. The share group must be quarantined.
    #[error("Native EGL resource cleanup failed")]
    NativeCleanup(#[source] crate::backend::egl::EGLError),
    /// A retained dependency could not produce a native fence descriptor.
    #[error("Required dependency has no exportable native fence")]
    NativeFenceUnavailable,
    /// A GL synchronization command failed.
    #[error("GL synchronization failed during {operation} (GL error {code:#x})")]
    GlSync {
        /// Operation whose completion could not be established.
        operation: &'static str,
        /// Raw GL error, or zero when the driver returned a null fence without
        /// a GL error.
        code: u32,
    },
    /// An earlier submission or abandoned frame has no observable completion.
    #[error("Renderer completion is unobservable; quarantine this renderer epoch")]
    UnobservableCompletion,
}

impl From<GlesError> for SwapBuffersError {
    #[cfg(feature = "wayland_frontend")]
    #[inline]
    fn from(err: GlesError) -> SwapBuffersError {
        match err {
            x @ GlesError::ContextCreationError(_)
            | x @ GlesError::ShaderCompileError
            | x @ GlesError::ProgramLinkError
            | x @ GlesError::GLFunctionLoaderError
            | x @ GlesError::GLExtensionNotSupported(_)
            | x @ GlesError::EGLExtensionNotSupported(_)
            | x @ GlesError::GLVersionNotSupported(_)
            | x @ GlesError::NativeFence(_)
            | x @ GlesError::NativeCleanup(_)
            | x @ GlesError::NativeFenceUnavailable
            | x @ GlesError::GlSync { .. }
            | x @ GlesError::UnobservableCompletion => SwapBuffersError::ContextLost(Box::new(x)),
            GlesError::ContextActivationError(err) => err.into(),
            x @ GlesError::Composition(_)
            | x @ GlesError::FramebufferContextMismatch
            | x @ GlesError::FramebufferBindingError
            | x @ GlesError::BindBufferEGLError(_)
            | x @ GlesError::UnknownPixelFormat
            | x @ GlesError::UnsupportedPixelFormat(_)
            | x @ GlesError::UnsupportedWlPixelFormat(_)
            | x @ GlesError::UnsupportedPixelLayout
            | x @ GlesError::BufferAccessError(_)
            | x @ GlesError::MappingError
            | x @ GlesError::UnexpectedSize
            | x @ GlesError::UnknownSize
            | x @ GlesError::BlitError
            | x @ GlesError::CreateShaderObject
            | x @ GlesError::UniformTypeMismatch { .. }
            | x @ GlesError::UnknownUniform(_)
            | x @ GlesError::EGLBufferAccessError(_)
            | x @ GlesError::SyncInterrupted => SwapBuffersError::TemporaryFailure(Box::new(x)),
        }
    }
    #[cfg(not(feature = "wayland_frontend"))]
    #[inline]
    fn from(err: GlesError) -> SwapBuffersError {
        match err {
            x @ GlesError::ContextCreationError(_)
            | x @ GlesError::ShaderCompileError
            | x @ GlesError::ProgramLinkError
            | x @ GlesError::GLFunctionLoaderError
            | x @ GlesError::GLExtensionNotSupported(_)
            | x @ GlesError::EGLExtensionNotSupported(_)
            | x @ GlesError::GLVersionNotSupported(_)
            | x @ GlesError::NativeFence(_)
            | x @ GlesError::NativeCleanup(_)
            | x @ GlesError::NativeFenceUnavailable
            | x @ GlesError::GlSync { .. }
            | x @ GlesError::UnobservableCompletion => SwapBuffersError::ContextLost(Box::new(x)),
            GlesError::ContextActivationError(err) => err.into(),
            x @ GlesError::Composition(_)
            | x @ GlesError::FramebufferContextMismatch
            | x @ GlesError::FramebufferBindingError
            | x @ GlesError::MappingError
            | x @ GlesError::UnknownPixelFormat
            | x @ GlesError::UnsupportedPixelFormat(_)
            | x @ GlesError::UnsupportedPixelLayout
            | x @ GlesError::UnexpectedSize
            | x @ GlesError::UnknownSize
            | x @ GlesError::BlitError
            | x @ GlesError::CreateShaderObject
            | x @ GlesError::UniformTypeMismatch { .. }
            | x @ GlesError::UnknownUniform(_)
            | x @ GlesError::BindBufferEGLError(_)
            | x @ GlesError::SyncInterrupted => SwapBuffersError::TemporaryFailure(Box::new(x)),
        }
    }
}

impl GlesError {
    /// True only for an actual EGL/GL context-loss report.
    ///
    /// An unobservable fence does not establish device loss or authorize
    /// reclaiming submitted resources.
    pub fn is_device_lost(&self) -> bool {
        if matches!(self, Self::GlSync { code: 0x0507, .. }) {
            return true;
        }
        let mut source = std::error::Error::source(self);
        while let Some(error) = source {
            if matches!(
                error.downcast_ref::<crate::backend::egl::EGLError>(),
                Some(crate::backend::egl::EGLError::ContextLost)
            ) {
                return true;
            }
            source = error.source();
        }
        false
    }

    /// Whether native work may lack an observable completion edge.
    pub fn is_completion_unobservable(&self) -> bool {
        matches!(
            self,
            Self::NativeFence(_)
                | Self::NativeCleanup(_)
                | Self::NativeFenceUnavailable
                | Self::GlSync { .. }
                | Self::UnobservableCompletion
        )
    }
}

impl crate::backend::renderer::damage::MaybeDeviceLost for GlesError {
    fn is_device_lost(&self) -> bool {
        self.is_device_lost()
    }

    fn is_completion_unobservable(&self) -> bool {
        self.is_completion_unobservable()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::egl::{EGLError, Error};

    #[test]
    fn lost_completion_is_not_device_loss_or_reclamation_evidence() {
        for error in [
            GlesError::UnobservableCompletion,
            GlesError::NativeFenceUnavailable,
            GlesError::NativeFence(Error::CreationFailed(EGLError::BadAlloc)),
        ] {
            assert!(error.is_completion_unobservable());
            assert!(!error.is_device_lost());
        }
        assert!(GlesError::NativeFence(Error::CreationFailed(EGLError::ContextLost)).is_device_lost());
        assert!(GlesError::ContextActivationError(EGLError::ContextLost.into()).is_device_lost());
    }
}
