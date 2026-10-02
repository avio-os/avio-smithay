//! Actual submitted resource records and Vulkan ABI waits, with no driver.
//! An allocation error is not a completion proof or DEVICE_LOST.
use super::super::{
    allocation::VulkanAllocationReason,
    device_handle::retirement_tests::{device_with_wait_result, image, next, Operation},
    readback::ReadbackBuffer,
};
use super::*;
use ash::vk::Handle;
use std::sync::atomic::{AtomicI32, Ordering};

#[test]
fn failed_native_readback_wait_retains_exact_fence_source_and_destination_until_completion() {
    let wait = Arc::new(AtomicI32::new(vk::Result::ERROR_OUT_OF_HOST_MEMORY.as_raw()));
    let (device, events) = device_with_wait_result(wait.clone());
    let census = device.allocation_ledger().clone();
    let source = image(device.clone(), 400);
    let source_weak = Arc::downgrade(&source);
    let destination = ReadbackBuffer::new(
        device.clone(),
        vk::Buffer::from_raw(500),
        vk::DeviceMemory::from_raw(501),
        true,
        census.record(VulkanAllocationReason::Scratch, 4096),
    );
    let destination_weak = Arc::downgrade(&destination);
    let fence = VulkanFence::create(device.clone()).unwrap();
    let fence_handle = fence.handle().as_raw();
    device.mark_submission_pending();
    let mut commands = RetiredCommands::empty(vk::CommandPool::from_raw(600));
    commands.device = Some(device.clone());
    commands.submissions.push_back(InFlightSubmission {
        id: SubmissionId(1),
        fence,
        export_semaphore: None,
        command_buffers: Vec::new(),
        framebuffers: Vec::new(),
        retained_images: vec![source],
        upload_sources: Vec::new(),
        _readback: Some(destination),
        wait_semaphores: Vec::new(),
        submitted_at: Instant::now(),
    });
    assert_eq!(
        device.observe_result(commands.wait_complete(device.handle())),
        Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
    );
    assert!(
        !device.is_lost(),
        "allocation failure must not invent device loss"
    );
    assert!(
        events.try_recv().is_err(),
        "no fence or GPU resource was destroyed on failed wait"
    );
    assert!(source_weak.upgrade().is_some());
    assert!(destination_weak.upgrade().is_some());
    assert_eq!(
        census
            .snapshot()
            .reason(VulkanAllocationReason::Scratch)
            .live_bytes,
        4096
    );
    // A subsequent actual success is the only release gate. Destroy the
    // native command references before dropping either resource owner.
    wait.store(vk::Result::SUCCESS.as_raw(), Ordering::Release);
    commands.wait_complete(device.handle()).unwrap();
    commands.destroy(device.handle());
    assert!(source_weak.upgrade().is_none());
    assert!(destination_weak.upgrade().is_none());
    let mut seen = Vec::new();
    for _ in 0..8 {
        seen.push(next(&events).0);
    }
    assert_eq!(seen[0], Operation::CommandPool(600));
    assert!(seen.contains(&Operation::Fence(fence_handle)));
    assert!(seen.contains(&Operation::Image(400)));
    assert!(seen.contains(&Operation::Buffer(500)));
    let buffer = seen.iter().position(|e| *e == Operation::Buffer(500)).unwrap();
    let memory = seen.iter().position(|e| *e == Operation::Memory(501)).unwrap();
    assert!(buffer < memory);
    drop(device);
    assert_eq!(next(&events).0, Operation::Device);
    assert_eq!(next(&events).0, Operation::Parent);
    assert_eq!(
        census
            .snapshot()
            .reason(VulkanAllocationReason::Scratch)
            .live_bytes,
        0
    );
}
