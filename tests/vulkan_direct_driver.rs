//! An exclusive direct-driver instance loads only the driver it was given.
//!
//! This is its own test binary with a single test: the Vulkan loader never unloads a driver that
//! links LLVM, so the process maps must not have seen a system-loader instance before.

#![cfg(all(feature = "backend_vulkan", feature = "backend_drm"))]

use std::{
    fs,
    path::{Path, PathBuf},
};

use smithay::backend::{
    drm::DrmNode,
    vulkan::{version::Version, DirectDriver, Instance, PhysicalDevice},
};

/// Every DRM render node and its kernel driver.
fn render_nodes() -> Vec<(DrmNode, String)> {
    let mut nodes = fs::read_dir("/dev/dri")
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("renderD"))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    nodes.sort();
    nodes
        .into_iter()
        .filter_map(|path| Some((DrmNode::from_path(&path).ok()?, kernel_driver(&path)?)))
        .collect()
}

fn kernel_driver(node: &Path) -> Option<String> {
    let name = node.file_name()?;
    let driver = fs::read_link(Path::new("/sys/class/drm").join(name).join("device/driver")).ok()?;
    Some(driver.file_name()?.to_string_lossy().into_owned())
}

/// The library of the system manifest for `driver`, read with just enough parsing for a test.
fn driver_library(driver: &str) -> Option<PathBuf> {
    let stem = match driver {
        "nvidia" => "nvidia_icd",
        "amdgpu" => "radeon_icd",
        "i915" | "xe" => "intel_icd",
        _ => return None,
    };
    let manifest = ["/etc/vulkan/icd.d", "/usr/share/vulkan/icd.d"]
        .iter()
        .flat_map(|dir| {
            [
                Path::new(dir).join(format!("{stem}.{}.json", std::env::consts::ARCH)),
                Path::new(dir).join(format!("{stem}.json")),
            ]
        })
        .find_map(|path| fs::read_to_string(path).ok())?;
    let value = manifest.split("\"library_path\"").nth(1)?;
    let value = value.split('"').nth(1)?;
    Some(PathBuf::from(value))
}

#[test]
fn an_exclusive_instance_enumerates_and_maps_only_its_driver() {
    let mut routes = render_nodes()
        .into_iter()
        .filter_map(|(node, driver)| Some((node, driver_library(&driver)?, driver)))
        .collect::<Vec<_>>();
    // radv links LLVM, which can never be unloaded; probe it last so every other driver is
    // checked against a process that has never loaded LLVM.
    routes.sort_by_key(|(_, _, driver)| driver == "amdgpu");
    if routes.is_empty() {
        // No DRM render node with a known Vulkan manifest: nothing to prove here.
        return;
    }
    for (node, library, driver) in &routes {
        let loaded = unsafe { DirectDriver::load(library) }.expect("the system driver loads");
        let instance =
            unsafe { Instance::with_direct_drivers(Version::VERSION_1_3, None, &[], vec![loaded]) }
                .expect("exclusive instance");
        assert!(instance.is_extension_enabled(c"VK_LUNARG_direct_driver_loading"));
        assert_eq!(instance.direct_drivers().collect::<Vec<_>>(), [library.as_path()]);

        let devices = PhysicalDevice::enumerate(&instance)
            .expect("enumeration")
            .collect::<Vec<_>>();
        assert!(
            devices
                .iter()
                .any(|device| device.render_node().ok().flatten() == Some(*node)),
            "{driver} exposes its own render node {node}"
        );
        for device in &devices {
            let device_driver = device
                .render_node()
                .ok()
                .flatten()
                .and_then(|node| node.dev_path())
                .and_then(|path| kernel_driver(&path));
            assert_eq!(
                device_driver.as_deref(),
                Some(driver.as_str()),
                "{} belongs to another driver",
                device.name()
            );
        }

        // Neither a pre-instance extension enumeration nor the instance loaded another
        // installed driver. Of these, only radv itself links LLVM.
        let maps = fs::read_to_string("/proc/self/maps").expect("own maps");
        for foreign in ["libvulkan_lvp.so", "libvulkan_dzn.so"] {
            assert!(!maps.contains(foreign), "{foreign} was loaded beside {driver}");
        }
        if driver != "amdgpu" {
            assert!(
                !maps.contains("libvulkan_radeon.so"),
                "radv was loaded beside {driver}"
            );
            assert!(!maps.contains("libLLVM"), "LLVM was loaded beside {driver}");
        }
    }
}
