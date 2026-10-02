//! No Vulkan loader/device is used: actual image/handle Drop invokes these
//! Vulkan ABI functions through an ash dispatch table. This proves execution
//! ownership, not real driver latency or GPU completion.

use super::DeviceHandle;
use crate::backend::allocator::{Format, Fourcc, Modifier};
use crate::backend::renderer::vulkan::{
    allocation::VulkanAllocationReason, format::ColorEncoding, image::VulkanImage,
};
use ash::{vk, vk::Handle};
use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc, Mutex, OnceLock,
    },
    thread::{self, ThreadId},
    time::Duration,
};

#[derive(Debug, PartialEq, Eq)]
pub(in super::super) enum Operation {
    View(u64),
    Image(u64),
    Memory(u64),
    Device,
    Parent,
}

type Event = (Operation, ThreadId);
static DEVICES: OnceLock<Mutex<HashMap<u64, Sender<Event>>>> = OnceLock::new();
static NEXT_DEVICE: AtomicU64 = AtomicU64::new(1);

fn record(device: vk::Device, operation: Operation) {
    DEVICES
        .get()
        .unwrap()
        .lock()
        .unwrap()
        .get(&device.as_raw())
        .unwrap()
        .send((operation, thread::current().id()))
        .unwrap();
}

unsafe extern "system" fn destroy_view(
    device: vk::Device,
    view: vk::ImageView,
    _: *const vk::AllocationCallbacks<'_>,
) {
    record(device, Operation::View(view.as_raw()));
}

unsafe extern "system" fn destroy_image(
    device: vk::Device,
    image: vk::Image,
    _: *const vk::AllocationCallbacks<'_>,
) {
    record(device, Operation::Image(image.as_raw()));
}

unsafe extern "system" fn free_memory(
    device: vk::Device,
    memory: vk::DeviceMemory,
    _: *const vk::AllocationCallbacks<'_>,
) {
    record(device, Operation::Memory(memory.as_raw()));
}

unsafe extern "system" fn destroy_device(device: vk::Device, _: *const vk::AllocationCallbacks<'_>) {
    record(device, Operation::Device);
}

struct Parent {
    device: u64,
    events: Sender<Event>,
}

impl Drop for Parent {
    fn drop(&mut self) {
        DEVICES.get().unwrap().lock().unwrap().remove(&self.device);
        self.events
            .send((Operation::Parent, thread::current().id()))
            .unwrap();
    }
}

pub(in super::super) fn device() -> (Arc<DeviceHandle>, Receiver<Event>) {
    let id = NEXT_DEVICE.fetch_add(1, Ordering::Relaxed);
    let (events, received) = mpsc::channel();
    DEVICES
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .insert(id, events.clone());
    // SAFETY: Only the supplied destructor functions are invoked by this
    // fixture. All handles are test identities, never handed to a real driver.
    let raw = unsafe {
        ash::Device::load_with(
            |name| match name.to_bytes() {
                b"vkDestroyImageView" => destroy_view as *const c_void,
                b"vkDestroyImage" => destroy_image as *const c_void,
                b"vkFreeMemory" => free_memory as *const c_void,
                b"vkDestroyDevice" => destroy_device as *const c_void,
                _ => std::ptr::null(),
            },
            vk::Device::from_raw(id),
        )
    };
    (
        Arc::new(DeviceHandle::for_retirement_test(
            raw,
            Parent { device: id, events },
        )),
        received,
    )
}

pub(in super::super) fn image(device: Arc<DeviceHandle>, id: u64) -> Arc<VulkanImage> {
    let allocation = device
        .allocation_ledger()
        .record(VulkanAllocationReason::RenderTarget, 4096);
    Arc::new(VulkanImage::new_renderer_local(
        id,
        vk::Image::from_raw(id),
        vk::DeviceMemory::from_raw(id + 1),
        allocation,
        vk::ImageView::from_raw(id + 2),
        vk::ImageView::from_raw(id + 3),
        (1, 1).into(),
        Format {
            code: Fourcc::Argb8888,
            modifier: Modifier::Linear,
        },
        vk::Format::B8G8R8A8_UNORM,
        vk::FormatFeatureFlags::SAMPLED_IMAGE,
        ColorEncoding::LinearPremultiplied,
        vk::ImageUsageFlags::SAMPLED,
        false,
        vk::ImageLayout::UNDEFINED,
        device,
    ))
}

pub(in super::super) fn next(events: &Receiver<Event>) -> Event {
    events.recv_timeout(Duration::from_secs(2)).unwrap()
}

#[test]
fn final_image_drop_wakes_idle_device_and_frees_on_executor() {
    let (device, events) = device();
    let ledger = device.allocation_ledger().clone();
    let image = image(device.clone(), 10);
    let weak = Arc::downgrade(&image);
    let dropping_thread = thread::spawn(move || {
        let id = thread::current().id();
        drop(image);
        id
    })
    .join()
    .unwrap();
    assert!(weak.upgrade().is_none());
    let (operation, executor) = next(&events);
    assert_eq!(operation, Operation::View(12));
    assert_ne!(executor, dropping_thread);
    for expected in [Operation::View(13), Operation::Image(10), Operation::Memory(11)] {
        assert_eq!(next(&events), (expected, executor));
    }
    // The executor woke and freed memory while the owner remained idle/live.
    assert!(events.try_recv().is_err());
    assert_eq!(device.take_retired_views(), [vk::ImageView::from_raw(12)]);
    drop(device);
    assert_eq!(next(&events), (Operation::Device, executor));
    assert_eq!(next(&events), (Operation::Parent, executor));
    assert_eq!(
        ledger
            .snapshot()
            .reason(VulkanAllocationReason::RenderTarget)
            .live_bytes,
        0
    );
}

#[test]
fn submitted_image_custody_precedes_retirement_and_parent_teardown() {
    let (device, events) = device();
    let image = image(device.clone(), 20);
    let submitted_reader = image.clone();
    drop(image);
    drop(device);
    assert!(
        events.try_recv().is_err(),
        "a submitted reader owns the image and device"
    );
    let completed_thread = thread::spawn(move || {
        let id = thread::current().id();
        drop(submitted_reader);
        id
    })
    .join()
    .unwrap();
    let expected = [
        Operation::View(22),
        Operation::View(23),
        Operation::Image(20),
        Operation::Memory(21),
        Operation::Device,
        Operation::Parent,
    ];
    let mut executor = None;
    for operation in expected {
        let (actual, thread) = next(&events);
        assert_eq!(actual, operation);
        assert_ne!(thread, completed_thread);
        assert_eq!(*executor.get_or_insert(thread), thread);
    }
}

#[test]
fn poisoned_view_notifications_cannot_make_final_drop_panic() {
    let (device, events) = device();
    let image = image(device.clone(), 30);
    let poison = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _views = device.retired_texture_views.lock().unwrap();
        panic!("renderer fault");
    }));
    assert!(poison.is_err());
    drop(image);
    drop(device);
    for expected in [
        Operation::View(32),
        Operation::View(33),
        Operation::Image(30),
        Operation::Memory(31),
        Operation::Device,
        Operation::Parent,
    ] {
        assert_eq!(next(&events).0, expected);
    }
}

#[test]
fn lost_device_retires_owners_without_claiming_driver_frees() {
    let (device, events) = device();
    let ledger = device.allocation_ledger().clone();
    let image = image(device.clone(), 40);
    device.mark_lost();
    drop(image);
    drop(device);
    // A lost device's driver handles intentionally remain quarantined.
    assert_eq!(next(&events).0, Operation::Parent);
    assert!(events.try_recv().is_err());
    assert_eq!(
        ledger
            .snapshot()
            .reason(VulkanAllocationReason::RenderTarget)
            .live_bytes,
        0
    );
}
