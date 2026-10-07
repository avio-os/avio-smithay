//! Application-supplied Vulkan drivers (`VK_LUNARG_direct_driver_loading`).
//!
//! The Vulkan loader normally discovers every installed driver (ICD) and keeps each one that
//! reports a physical device. A process that already knows which driver serves its device can
//! instead load that driver itself and hand the loader its `vk_icdGetInstanceProcAddr` through
//! [`Instance::with_direct_drivers`](super::Instance::with_direct_drivers). In the extension's
//! exclusive mode the loader then performs no system or environment driver search for the
//! instance. Layers keep working.
//!
//! This module only loads the library it is given. Deciding which driver serves a device, and
//! where its library lives, belongs to the caller.

use std::{
    ffi::CStr,
    fmt,
    path::{Path, PathBuf},
};

use ash::vk;
use libloading::os::unix::{Library, RTLD_LOCAL, RTLD_NOW};

/// The entry point the loader-ICD interface requires every driver to export.
const ICD_GET_INSTANCE_PROC_ADDR: &CStr = c"vk_icdGetInstanceProcAddr";

/// Error loading a [`DirectDriver`].
#[derive(Debug, thiserror::Error)]
pub enum DirectDriverError {
    /// The driver library could not be loaded.
    #[error("failed to load Vulkan driver {path}: {source}")]
    Load {
        /// The library path or soname given to the dynamic linker.
        path: PathBuf,
        /// The dynamic linker's error.
        source: libloading::Error,
    },
    /// The library does not export `vk_icdGetInstanceProcAddr`, so it is not a Vulkan driver.
    #[error("{path} does not export vk_icdGetInstanceProcAddr")]
    MissingEntryPoint {
        /// The library path or soname given to the dynamic linker.
        path: PathBuf,
    },
}

/// One loaded Vulkan driver library and its loader entry point.
///
/// The library stays loaded for as long as this value lives. An [`Instance`](super::Instance)
/// created from it owns it until after `vkDestroyInstance`.
pub struct DirectDriver {
    path: PathBuf,
    get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddrLUNARG,
    _library: Library,
}

impl DirectDriver {
    /// Loads a Vulkan driver library and resolves its `vk_icdGetInstanceProcAddr`.
    ///
    /// The library is opened with `RTLD_NOW | RTLD_LOCAL`, as the Vulkan loader opens drivers.
    /// `path` is passed to the dynamic linker unchanged: an absolute path, or a bare soname that
    /// the linker resolves through its own search path.
    ///
    /// # Safety
    ///
    /// Loading a library runs its initialisers. `path` must name a trusted Vulkan driver.
    pub unsafe fn load(path: &Path) -> Result<Self, DirectDriverError> {
        let library = unsafe { Library::open(Some(path), RTLD_NOW | RTLD_LOCAL) }.map_err(|source| {
            DirectDriverError::Load {
                path: path.to_owned(),
                source,
            }
        })?;
        let get_instance_proc_addr = unsafe {
            library.get::<unsafe extern "system" fn(
                    vk::Instance,
                    *const std::ffi::c_char,
                ) -> vk::PFN_vkVoidFunction>(
                    ICD_GET_INSTANCE_PROC_ADDR.to_bytes_with_nul()
                )
        }
        .map(|symbol| *symbol)
        .map_err(|_| DirectDriverError::MissingEntryPoint {
            path: path.to_owned(),
        })?;
        Ok(Self {
            path: path.to_owned(),
            get_instance_proc_addr: Some(get_instance_proc_addr),
            _library: library,
        })
    }

    /// The library path or soname this driver was loaded from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn loading_info(&self) -> vk::DirectDriverLoadingInfoLUNARG<'static> {
        vk::DirectDriverLoadingInfoLUNARG::default().pfn_get_instance_proc_addr(self.get_instance_proc_addr)
    }
}

impl fmt::Debug for DirectDriver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DirectDriver")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_library_is_a_typed_load_error() {
        let error = unsafe { DirectDriver::load(Path::new("/nonexistent/libvulkan_missing.so")) }
            .expect_err("no such library");
        assert!(matches!(error, DirectDriverError::Load { .. }));
    }

    #[test]
    fn a_library_without_the_icd_entry_point_is_not_a_driver() {
        let error = unsafe { DirectDriver::load(Path::new("libc.so.6")) }.expect_err("libc is no driver");
        assert!(matches!(error, DirectDriverError::MissingEntryPoint { .. }));
    }
}
