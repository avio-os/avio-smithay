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
        atomic::{AtomicI32, AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc, Mutex, OnceLock,
    },
    thread::{self, ThreadId},
    time::Duration,
};

#[derive(Debug, PartialEq, Eq)]
pub(in super::super) enum Operation {
    DescriptorWrite(u64),
    Buffer(u64),
    Unmap(u64),
    Fence(u64),
    FenceReset(u64),
    CommandPool(u64),
    CommandPoolCreated(u64),
    CommandBuffersAllocated(u32),
    SemaphoreCreated(u64),
    Semaphore(u64),
    SemaphoreImported(u64),
    View(u64),
    Image(u64),
    Memory(u64),
    Device,
    Parent,
}

// CPU native-dispatch fixtures: descriptors have no driver allocation here.
unsafe extern "system" fn update_descriptor_sets(
    device: vk::Device,
    count: u32,
    writes: *const vk::WriteDescriptorSet<'_>,
    _: u32,
    _: *const vk::CopyDescriptorSet<'_>,
) {
    for write in unsafe { std::slice::from_raw_parts(writes, count as usize) } {
        record(device, Operation::DescriptorWrite(write.dst_set.as_raw()));
    }
}
unsafe extern "system" fn destroy_descriptor_set_layout(
    _: vk::Device,
    _: vk::DescriptorSetLayout,
    _: *const vk::AllocationCallbacks<'_>,
) {
}
unsafe extern "system" fn destroy_sampler(
    _: vk::Device,
    _: vk::Sampler,
    _: *const vk::AllocationCallbacks<'_>,
) {
}

unsafe extern "system" fn destroy_buffer(
    device: vk::Device,
    buffer: vk::Buffer,
    _: *const vk::AllocationCallbacks<'_>,
) {
    record(device, Operation::Buffer(buffer.as_raw()));
}

unsafe extern "system" fn unmap_memory(device: vk::Device, memory: vk::DeviceMemory) {
    record(device, Operation::Unmap(memory.as_raw()));
}

type Event = (Operation, ThreadId);
struct TestDispatch {
    events: Sender<Event>,
    wait_result: Arc<AtomicI32>,
    import_result: AtomicI32,
}
static DEVICES: OnceLock<Mutex<HashMap<u64, TestDispatch>>> = OnceLock::new();
static NEXT_DEVICE: AtomicU64 = AtomicU64::new(1);
static NEXT_NATIVE: AtomicU64 = AtomicU64::new(10000);

fn record(device: vk::Device, operation: Operation) {
    DEVICES
        .get()
        .unwrap()
        .lock()
        .unwrap()
        .get(&device.as_raw())
        .unwrap()
        .events
        .send((operation, thread::current().id()))
        .unwrap();
}

unsafe extern "system" fn create_fence(
    _device: vk::Device,
    _: *const vk::FenceCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    fence: *mut vk::Fence,
) -> vk::Result {
    unsafe {
        fence.write(vk::Fence::from_raw(NEXT_NATIVE.fetch_add(1, Ordering::Relaxed)));
    }
    vk::Result::SUCCESS
}
unsafe extern "system" fn reset_fences(
    device: vk::Device,
    count: u32,
    fences: *const vk::Fence,
) -> vk::Result {
    for fence in unsafe { std::slice::from_raw_parts(fences, count as usize) } {
        record(device, Operation::FenceReset(fence.as_raw()));
    }
    vk::Result::SUCCESS
}
unsafe extern "system" fn destroy_fence(
    device: vk::Device,
    fence: vk::Fence,
    _: *const vk::AllocationCallbacks<'_>,
) {
    record(device, Operation::Fence(fence.as_raw()));
}

unsafe extern "system" fn create_semaphore(
    device: vk::Device,
    _: *const vk::SemaphoreCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    semaphore: *mut vk::Semaphore,
) -> vk::Result {
    let id = NEXT_NATIVE.fetch_add(1, Ordering::Relaxed);
    unsafe { semaphore.write(vk::Semaphore::from_raw(id)) };
    record(device, Operation::SemaphoreCreated(id));
    vk::Result::SUCCESS
}
unsafe extern "system" fn destroy_semaphore(
    device: vk::Device,
    semaphore: vk::Semaphore,
    _: *const vk::AllocationCallbacks<'_>,
) {
    record(device, Operation::Semaphore(semaphore.as_raw()));
}
unsafe extern "system" fn import_semaphore_fd(
    device: vk::Device,
    info: *const vk::ImportSemaphoreFdInfoKHR<'_>,
) -> vk::Result {
    let result = {
        let devices = DEVICES.get().unwrap().lock().unwrap();
        vk::Result::from_raw(
            devices
                .get(&device.as_raw())
                .unwrap()
                .import_result
                .load(Ordering::Acquire),
        )
    };
    let info = unsafe { &*info };
    assert_eq!(info.flags, vk::SemaphoreImportFlags::TEMPORARY);
    assert_eq!(info.handle_type, vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    assert!(unsafe { libc::fcntl(info.fd, libc::F_GETFD) } >= 0);
    if result == vk::Result::SUCCESS {
        // Mirror the native ABI ownership transfer, without claiming that the
        // eventfd fixture is a real driver-accepted Linux sync_file.
        unsafe { libc::close(info.fd) };
        record(device, Operation::SemaphoreImported(info.semaphore.as_raw()));
    }
    result
}

unsafe extern "system" fn get_device_proc_addr(
    _: vk::Device,
    name: *const std::ffi::c_char,
) -> vk::PFN_vkVoidFunction {
    if unsafe { std::ffi::CStr::from_ptr(name) }.to_bytes() == b"vkImportSemaphoreFdKHR" {
        Some(unsafe {
            std::mem::transmute::<vk::PFN_vkImportSemaphoreFdKHR, unsafe extern "system" fn()>(
                import_semaphore_fd,
            )
        })
    } else {
        None
    }
}

pub(in super::super) fn external_semaphore_fd(
    device: &DeviceHandle,
) -> ash::khr::external_semaphore_fd::Device {
    let instance = unsafe {
        ash::Instance::load_with(
            |name| {
                if name.to_bytes() == b"vkGetDeviceProcAddr" {
                    get_device_proc_addr as *const c_void
                } else {
                    std::ptr::null()
                }
            },
            vk::Instance::null(),
        )
    };
    ash::khr::external_semaphore_fd::Device::new(&instance, device.handle())
}

pub(in super::super) fn set_import_result(device: &DeviceHandle, result: vk::Result) {
    DEVICES
        .get()
        .unwrap()
        .lock()
        .unwrap()
        .get(&device.handle().handle().as_raw())
        .unwrap()
        .import_result
        .store(result.as_raw(), Ordering::Release);
}
unsafe extern "system" fn get_fence_status(device: vk::Device, _: vk::Fence) -> vk::Result {
    let devices = DEVICES.get().unwrap().lock().unwrap();
    vk::Result::from_raw(
        devices
            .get(&device.as_raw())
            .unwrap()
            .wait_result
            .load(Ordering::Acquire),
    )
}
unsafe extern "system" fn wait_for_fences(
    device: vk::Device,
    _: u32,
    _: *const vk::Fence,
    _: vk::Bool32,
    _: u64,
) -> vk::Result {
    let devices = DEVICES.get().unwrap().lock().unwrap();
    vk::Result::from_raw(
        devices
            .get(&device.as_raw())
            .unwrap()
            .wait_result
            .load(Ordering::Acquire),
    )
}
unsafe extern "system" fn create_command_pool(
    device: vk::Device,
    _: *const vk::CommandPoolCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    pool: *mut vk::CommandPool,
) -> vk::Result {
    let id = NEXT_NATIVE.fetch_add(1, Ordering::Relaxed);
    unsafe {
        pool.write(vk::CommandPool::from_raw(id));
    }
    record(device, Operation::CommandPoolCreated(id));
    vk::Result::SUCCESS
}
unsafe extern "system" fn allocate_command_buffers(
    device: vk::Device,
    info: *const vk::CommandBufferAllocateInfo<'_>,
    buffers: *mut vk::CommandBuffer,
) -> vk::Result {
    let count = unsafe { (*info).command_buffer_count };
    for index in 0..count {
        unsafe {
            buffers.add(index as usize).write(vk::CommandBuffer::from_raw(
                NEXT_NATIVE.fetch_add(1, Ordering::Relaxed),
            ));
        }
    }
    record(device, Operation::CommandBuffersAllocated(count));
    vk::Result::SUCCESS
}
unsafe extern "system" fn destroy_command_pool(
    device: vk::Device,
    pool: vk::CommandPool,
    _: *const vk::AllocationCallbacks<'_>,
) {
    record(device, Operation::CommandPool(pool.as_raw()));
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
    device_with_wait_result(Arc::new(AtomicI32::new(vk::Result::SUCCESS.as_raw())))
}

pub(in super::super) fn device_with_wait_result(
    wait_result: Arc<AtomicI32>,
) -> (Arc<DeviceHandle>, Receiver<Event>) {
    let id = NEXT_DEVICE.fetch_add(1, Ordering::Relaxed);
    let (events, received) = mpsc::channel();
    DEVICES.get_or_init(Default::default).lock().unwrap().insert(
        id,
        TestDispatch {
            events: events.clone(),
            wait_result,
            import_result: AtomicI32::new(vk::Result::SUCCESS.as_raw()),
        },
    );
    // SAFETY: Only the supplied destructor functions are invoked by this
    // fixture. All handles are test identities, never handed to a real driver.
    let raw = unsafe {
        ash::Device::load_with(
            |name| match name.to_bytes() {
                b"vkUpdateDescriptorSets" => update_descriptor_sets as *const c_void,
                b"vkDestroyDescriptorSetLayout" => destroy_descriptor_set_layout as *const c_void,
                b"vkDestroySampler" => destroy_sampler as *const c_void,
                b"vkDestroyImageView" => destroy_view as *const c_void,
                b"vkDestroyImage" => destroy_image as *const c_void,
                b"vkDestroyBuffer" => destroy_buffer as *const c_void,
                b"vkUnmapMemory" => unmap_memory as *const c_void,
                b"vkCreateFence" => create_fence as *const c_void,
                b"vkCreateSemaphore" => create_semaphore as *const c_void,
                b"vkDestroySemaphore" => destroy_semaphore as *const c_void,
                b"vkDestroyFence" => destroy_fence as *const c_void,
                b"vkResetFences" => reset_fences as *const c_void,
                b"vkWaitForFences" => wait_for_fences as *const c_void,
                b"vkGetFenceStatus" => get_fence_status as *const c_void,
                b"vkDestroyCommandPool" => destroy_command_pool as *const c_void,
                b"vkCreateCommandPool" => create_command_pool as *const c_void,
                b"vkAllocateCommandBuffers" => allocate_command_buffers as *const c_void,
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
    Arc::new(
        VulkanImage::new_renderer_local(
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
        )
        .expect("cold image incarnation"),
    )
}

pub(in super::super) fn drain_until_parent(events: &Receiver<Event>) {
    loop {
        if next(events).0 == Operation::Parent {
            break;
        }
    }
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
