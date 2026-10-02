//! Make GPU prerequisites explicit in required-device test runs.

use std::fmt::Display;

use super::{VulkanRenderer, VulkanRendererError};
use crate::backend::vulkan::{version::Version, Instance, PhysicalDevice};

fn device_required() -> bool {
    std::env::var_os("AVIO_REQUIRE_VK_DEVICE").is_some_and(|value| value == "1")
}

#[track_caller]
pub(super) fn unavailable(reason: impl Display) {
    check_available(false, device_required(), reason);
}

#[track_caller]
pub(super) fn available<T, E: Display>(result: Result<T, E>, requirement: &str) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(error) => {
            unavailable(format_args!("{requirement}: {error}"));
            None
        }
    }
}

#[track_caller]
pub(super) fn present<T>(value: Option<T>, requirement: &str) -> Option<T> {
    if value.is_none() {
        unavailable(requirement);
    }
    value
}

pub(super) fn physical_device() -> Option<PhysicalDevice> {
    let instance = available(Instance::new(Version::VERSION_1_3, None), "Vulkan instance")?;
    let mut devices = available(PhysicalDevice::enumerate(&instance), "Vulkan device enumeration")?;
    present(devices.next(), "no Vulkan physical device")
}

pub(super) fn renderer(physical_device: &PhysicalDevice) -> Option<VulkanRenderer> {
    match VulkanRenderer::new(physical_device) {
        Ok(renderer) => Some(renderer),
        Err(
            error @ (VulkanRendererError::MissingDeviceExtensions(_)
            | VulkanRendererError::MissingDeviceFeature(_)
            | VulkanRendererError::MissingQueueFamily { .. }),
        ) => {
            unavailable(error);
            None
        }
        Err(error) => panic!("unexpected Vulkan renderer init failure: {error}"),
    }
}

#[track_caller]
fn check_available(available: bool, required: bool, reason: impl Display) {
    assert!(
        available || !required,
        "required Vulkan test prerequisite unavailable: {reason}"
    );
}

#[test]
#[should_panic(expected = "VK_EXT_external_memory_dma_buf")]
fn required_device_failure_names_the_missing_extension() {
    check_available(false, true, "missing VK_EXT_external_memory_dma_buf");
}

#[test]
fn optional_device_failure_and_available_device_are_allowed() {
    check_available(false, false, "no Vulkan device");
    check_available(true, true, "device available");
}
