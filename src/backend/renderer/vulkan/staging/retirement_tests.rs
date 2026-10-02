//! Actual arena/row Drop paths through native ash dispatch. No GPU completion
//! or driver timing is inferred from this CPU ownership fixture.

use super::super::device_handle::retirement_tests::{device, next, Operation};
use super::*;
use crate::backend::renderer::{MemoryUploadCpuCompletion, StagedMemoryRows};
use ash::vk::Handle;

fn arena(device: Arc<DeviceHandle>, bytes: &mut [u8; 16]) -> UploadArena {
    let allocation = device
        .allocation_ledger()
        .record(VulkanAllocationReason::Upload, 16);
    UploadArena {
        chunks: vec![StagingChunk {
            memory: Arc::new(ChunkMemory::new(
                device,
                vk::Buffer::from_raw(41),
                vk::DeviceMemory::from_raw(42),
                MappedAddress(bytes.as_mut_ptr()),
                true,
                allocation,
            )),
            ranges: ReservationRanges::new(16),
        }],
        atom_size: 4,
        stats: UploadArenaStats {
            capacity_bytes: 16,
            chunk_count: 1,
            ..Default::default()
        },
    }
}

fn rows(arena: &mut UploadArena) -> (StagedMemoryRows, MemoryUploadCpuCompletion) {
    let reservation = arena.reserve(16).unwrap();
    let (pointer, writer) = arena.detach(reservation).unwrap();
    let signal = writer.completion();
    let completion = MemoryUploadCpuCompletion::new([signal.clone()]);
    // SAFETY: The real arena detached these exclusive sixteen writable bytes;
    // its exact ReservationWriter keeps the native mapping valid.
    let rows = unsafe { StagedMemoryRows::new_with_completion(pointer, 4, 4, 1, writer, signal) };
    (rows, completion)
}

#[test]
fn returned_rows_outlive_arena_and_destroy_native_mapping_on_device_actor() {
    let (device, events) = device();
    let census = device.allocation_ledger().clone();
    let mut bytes = Box::new([0; 16]);
    let mut arena = arena(device.clone(), &mut bytes);
    let (mut rows, completion) = rows(&mut arena);
    rows.row_mut(0).copy_from_slice(&[1, 2, 3, 4]);
    let old_arena = super::super::VulkanRetirementSlot::new(device.clone());
    old_arena.retire(arena);
    device.wait_retirement_drained();
    assert!(events.try_recv().is_err(), "returned rows still pin the mapping");
    assert!(!completion.is_ready());
    assert_eq!(
        census
            .snapshot()
            .reason(VulkanAllocationReason::Upload)
            .live_bytes,
        16
    );
    // DeviceState/renderer can go first. The exact row mapping owner retains
    // the device endpoint until the cache's final returned rows are dropped.
    drop(device);
    rows.row_mut(3).copy_from_slice(&[5, 6, 7, 8]);
    let dropping_thread = std::thread::spawn(move || {
        let caller = std::thread::current().id();
        drop(rows);
        caller
    })
    .join()
    .unwrap();
    assert!(completion.is_ready());
    let (operation, executor) = next(&events);
    assert_eq!(operation, Operation::Unmap(42));
    assert_ne!(executor, dropping_thread);
    for expected in [
        Operation::Buffer(41),
        Operation::Memory(42),
        Operation::Device,
        Operation::Parent,
    ] {
        assert_eq!(next(&events), (expected, executor));
    }
    assert_eq!(&bytes[..4], &[1, 2, 3, 4]);
    assert_eq!(&bytes[12..], &[5, 6, 7, 8]);
    assert_eq!(
        census
            .snapshot()
            .reason(VulkanAllocationReason::Upload)
            .live_bytes,
        0
    );
}

#[test]
fn final_chunk_release_publishes_before_device_endpoint_closure() {
    let (device, events) = device();
    let mut bytes = Box::new([0; 16]);
    let arena = arena(device.clone(), &mut bytes);
    drop(device);
    let caller = std::thread::current().id();
    drop(arena);
    let (operation, executor) = next(&events);
    assert_eq!(operation, Operation::Unmap(42));
    assert_ne!(executor, caller);
    for expected in [
        Operation::Buffer(41),
        Operation::Memory(42),
        Operation::Device,
        Operation::Parent,
    ] {
        assert_eq!(next(&events), (expected, executor));
    }
}

#[test]
fn lost_device_keeps_exact_staging_mapping_and_allocation_quarantined() {
    let (device, events) = device();
    let census = device.allocation_ledger().clone();
    let mut bytes = Box::new([0; 16]);
    let mut arena = arena(device.clone(), &mut bytes);
    let (mut rows, completion) = rows(&mut arena);
    device.mark_lost();
    drop(arena);
    drop(device);
    assert!(events.try_recv().is_err(), "live rows retain the device endpoint");
    rows.row_mut(0).copy_from_slice(&[9, 8, 7, 6]);
    drop(rows);
    assert!(
        completion.is_ready(),
        "CPU row return is independent of native device loss"
    );
    assert_eq!(next(&events).0, Operation::Parent);
    assert!(
        events.try_recv().is_err(),
        "loss cannot manufacture native unmap/free"
    );
    assert_eq!(
        census
            .snapshot()
            .reason(VulkanAllocationReason::Upload)
            .live_bytes,
        16
    );
    assert_eq!(&bytes[..4], &[9, 8, 7, 6]);
}
