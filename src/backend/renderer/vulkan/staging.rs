use std::{fmt, ops::Range, sync::Arc};

use ash::vk;

use crate::backend::vulkan::PhysicalDevice;

use super::{device::DeviceHandle, VulkanRendererError};

const INITIAL_UPLOAD_ARENA_BYTES: usize = 16 * 1024 * 1024;
const MAX_UPLOAD_ARENA_BYTES: usize = 256 * 1024 * 1024;
const MAX_UPLOAD_ARENA_CHUNKS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StagingReservation {
    chunk: usize,
    offset: usize,
    len: usize,
    reserved_len: usize,
}

impl StagingReservation {
    pub(crate) fn offset(self) -> vk::DeviceSize {
        self.offset as vk::DeviceSize
    }

    pub(crate) fn len(self) -> usize {
        self.len
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct UploadArenaStats {
    pub(crate) capacity_bytes: usize,
    pub(crate) in_use_bytes: usize,
    pub(crate) high_water_bytes: usize,
    pub(crate) chunk_count: usize,
    pub(crate) growth_count: u64,
    pub(crate) deferred_count: u64,
}

/// Renderer-local, persistently mapped staging memory.
///
/// All allocation and release happens on the renderer's one queue-owner
/// thread. Reservations remain unavailable until the tracked Vulkan
/// submission that consumed them retires.
pub(crate) struct UploadArena {
    chunks: Vec<StagingChunk>,
    atom_size: usize,
    stats: UploadArenaStats,
}

impl fmt::Debug for UploadArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UploadArena")
            .field("atom_size", &self.atom_size)
            .field("stats", &self.stats)
            .finish()
    }
}

impl UploadArena {
    pub(crate) fn new(physical_device: &PhysicalDevice) -> Self {
        let atom_size = usize::try_from(physical_device.limits().non_coherent_atom_size)
            .unwrap_or(usize::MAX)
            .max(4);
        Self {
            chunks: Vec::new(),
            atom_size,
            stats: UploadArenaStats::default(),
        }
    }

    pub(crate) fn reserve(
        &mut self,
        physical_device: &PhysicalDevice,
        device: Arc<DeviceHandle>,
        len: usize,
    ) -> Result<StagingReservation, VulkanRendererError> {
        let reserved_len = align_up(len, self.atom_size).ok_or(VulkanRendererError::InvalidMemoryUpload(
            "staging reservation size overflowed",
        ))?;
        if reserved_len == 0 {
            return Err(VulkanRendererError::InvalidMemoryUpload(
                "staging reservation must be non-empty",
            ));
        }

        if let Some(reservation) = self.reserve_existing(len, reserved_len) {
            return Ok(reservation);
        }

        if self.chunks.len() < MAX_UPLOAD_ARENA_CHUNKS {
            let remaining = MAX_UPLOAD_ARENA_BYTES.saturating_sub(self.stats.capacity_bytes);
            let previous = self.chunks.last().map(StagingChunk::capacity);
            let chunk_size = bounded_chunk_size(previous, reserved_len, remaining, self.chunks.len());
            if chunk_size >= reserved_len {
                let chunk = StagingChunk::new(
                    physical_device,
                    device,
                    align_up(chunk_size, self.atom_size).ok_or(VulkanRendererError::InvalidMemoryUpload(
                        "upload arena growth size overflowed",
                    ))?,
                    self.atom_size,
                )?;
                self.stats.capacity_bytes = self.stats.capacity_bytes.saturating_add(chunk.capacity());
                self.stats.chunk_count = self.stats.chunk_count.saturating_add(1);
                self.stats.growth_count = self.stats.growth_count.saturating_add(1);
                self.chunks.push(chunk);
                return self.reserve_existing(len, reserved_len).ok_or(
                    VulkanRendererError::TemporaryFailure(
                        "new upload arena chunk could not satisfy its triggering reservation",
                    ),
                );
            }
        }

        let remaining = MAX_UPLOAD_ARENA_BYTES.saturating_sub(self.stats.capacity_bytes);
        let max_existing = self.chunks.iter().map(StagingChunk::capacity).max().unwrap_or(0);
        let max_future = if self.chunks.len() < MAX_UPLOAD_ARENA_CHUNKS {
            remaining
        } else {
            0
        };
        let max_contiguous_bytes = max_existing.max(max_future);
        if reserved_len > max_contiguous_bytes {
            return Err(VulkanRendererError::UploadExceedsArenaLimit {
                requested_bytes: len,
                max_contiguous_bytes,
            });
        }

        self.stats.deferred_count = self.stats.deferred_count.saturating_add(1);
        Err(VulkanRendererError::UploadCapacityExhausted {
            requested_bytes: len,
            capacity_bytes: self.stats.capacity_bytes,
            in_use_bytes: self.stats.in_use_bytes,
        })
    }

    fn reserve_existing(&mut self, len: usize, reserved_len: usize) -> Option<StagingReservation> {
        for (chunk_index, chunk) in self.chunks.iter_mut().enumerate() {
            let Some(offset) = chunk.ranges.reserve(reserved_len) else {
                continue;
            };
            self.stats.in_use_bytes = self.stats.in_use_bytes.saturating_add(reserved_len);
            self.stats.high_water_bytes = self.stats.high_water_bytes.max(self.stats.in_use_bytes);
            return Some(StagingReservation {
                chunk: chunk_index,
                offset,
                len,
                reserved_len,
            });
        }
        None
    }

    pub(crate) fn write_rows(
        &self,
        reservation: StagingReservation,
        data: &[u8],
        src_offset: usize,
        src_stride: usize,
        row_bytes: usize,
        rows: usize,
    ) -> Result<(), VulkanRendererError> {
        let copied_len = row_bytes
            .checked_mul(rows)
            .ok_or(VulkanRendererError::InvalidMemoryUpload(
                "staging row copy size overflowed",
            ))?;
        if copied_len != reservation.len {
            return Err(VulkanRendererError::InvalidMemoryUpload(
                "staging reservation does not match row copy extent",
            ));
        }
        let chunk = self
            .chunks
            .get(reservation.chunk)
            .ok_or(VulkanRendererError::TemporaryFailure(
                "staging reservation names an unknown chunk",
            ))?;
        chunk.write_rows(reservation.offset, data, src_offset, src_stride, row_bytes, rows)?;
        chunk.flush(reservation.offset, reservation.reserved_len)
    }

    pub(crate) fn buffer(&self, reservation: StagingReservation) -> vk::Buffer {
        self.chunks[reservation.chunk].buffer
    }

    pub(crate) fn release(&mut self, reservation: StagingReservation) {
        let chunk = self
            .chunks
            .get_mut(reservation.chunk)
            .expect("tracked staging reservation names a live arena chunk");
        chunk.ranges.release(reservation.offset, reservation.reserved_len);
        self.stats.in_use_bytes = self.stats.in_use_bytes.saturating_sub(reservation.reserved_len);
    }

    pub(crate) fn stats(&self) -> UploadArenaStats {
        self.stats
    }
}

struct StagingChunk {
    device: Arc<DeviceHandle>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: MappedAddress,
    coherent: bool,
    ranges: RangeAllocator,
}

impl StagingChunk {
    fn new(
        physical_device: &PhysicalDevice,
        device: Arc<DeviceHandle>,
        size: usize,
        atom_size: usize,
    ) -> Result<Self, VulkanRendererError> {
        let create_info = vk::BufferCreateInfo::default()
            .size(size as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = device.observe_result(unsafe { device.handle().create_buffer(&create_info, None) })?;
        let memory_requirements = unsafe { device.handle().get_buffer_memory_requirements(buffer) };
        let (memory_type_index, coherent) =
            pick_host_visible_memory_type(physical_device, memory_requirements.memory_type_bits)
                .ok_or(VulkanRendererError::NoCompatibleMemoryType)?;
        let allocation_size = memory_requirements.size.max(size as vk::DeviceSize);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(allocation_size)
            .memory_type_index(memory_type_index);
        let memory =
            match device.observe_result(unsafe { device.handle().allocate_memory(&allocate_info, None) }) {
                Ok(memory) => memory,
                Err(error) => {
                    device.destroy_with(|vk_device| unsafe { vk_device.destroy_buffer(buffer, None) });
                    return Err(error.into());
                }
            };
        if let Err(error) =
            device.observe_result(unsafe { device.handle().bind_buffer_memory(buffer, memory, 0) })
        {
            device.destroy_with(|vk_device| unsafe {
                vk_device.free_memory(memory, None);
                vk_device.destroy_buffer(buffer, None);
            });
            return Err(error.into());
        }
        let mapped = match device.observe_result(unsafe {
            device
                .handle()
                .map_memory(memory, 0, allocation_size, vk::MemoryMapFlags::empty())
        }) {
            Ok(mapped) => MappedAddress(mapped.cast()),
            Err(error) => {
                device.destroy_with(|vk_device| unsafe {
                    vk_device.free_memory(memory, None);
                    vk_device.destroy_buffer(buffer, None);
                });
                return Err(error.into());
            }
        };

        let capacity = usize::try_from(allocation_size).unwrap_or(usize::MAX).min(size);
        let capacity = capacity - (capacity % atom_size);
        Ok(Self {
            device,
            buffer,
            memory,
            mapped,
            coherent,
            ranges: RangeAllocator::new(capacity),
        })
    }

    fn capacity(&self) -> usize {
        self.ranges.capacity
    }

    fn write_rows(
        &self,
        dst_offset: usize,
        data: &[u8],
        src_offset: usize,
        src_stride: usize,
        row_bytes: usize,
        rows: usize,
    ) -> Result<(), VulkanRendererError> {
        for row in 0..rows {
            let src_start =
                src_offset
                    .checked_add(row.checked_mul(src_stride).ok_or(
                        VulkanRendererError::InvalidMemoryUpload("source row offset overflowed"),
                    )?)
                    .ok_or(VulkanRendererError::InvalidMemoryUpload(
                        "source row offset overflowed",
                    ))?;
            let src_end =
                src_start
                    .checked_add(row_bytes)
                    .ok_or(VulkanRendererError::InvalidMemoryUpload(
                        "source row range overflowed",
                    ))?;
            let src = data
                .get(src_start..src_end)
                .ok_or(VulkanRendererError::InvalidMemoryUpload(
                    "source data slice is out of bounds",
                ))?;
            let dst =
                dst_offset
                    .checked_add(row.checked_mul(row_bytes).ok_or(
                        VulkanRendererError::InvalidMemoryUpload("staging row offset overflowed"),
                    )?)
                    .ok_or(VulkanRendererError::InvalidMemoryUpload(
                        "staging row offset overflowed",
                    ))?;
            // SAFETY: The arena owns a persistent mapping for the whole chunk;
            // the reservation bounds and source slice are validated above,
            // and atom-aligned reservations never overlap while live.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), self.mapped.0.add(dst), row_bytes);
            }
        }
        Ok(())
    }

    fn flush(&self, offset: usize, len: usize) -> Result<(), VulkanRendererError> {
        if self.coherent {
            return Ok(());
        }
        let ranges = [vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(offset as vk::DeviceSize)
            .size(len as vk::DeviceSize)];
        self.device
            .observe_result(unsafe { self.device.handle().flush_mapped_memory_ranges(&ranges) })?;
        Ok(())
    }
}

impl Drop for StagingChunk {
    fn drop(&mut self) {
        self.device.destroy_with(|device| unsafe {
            device.unmap_memory(self.memory);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        });
    }
}

/// The Vulkan renderer is moved between worker setup and its final queue-owner
/// thread, but a mapped address is only dereferenced through that exclusive
/// `&mut VulkanRenderer` authority. Vulkan host synchronization for the memory
/// allocation is therefore preserved across the move.
struct MappedAddress(*mut u8);

// SAFETY: See the type-level ownership argument above. The pointer is never
// shared across queue owners and is unmapped only by its owning chunk.
unsafe impl Send for MappedAddress {}

struct RangeAllocator {
    capacity: usize,
    free: Vec<Range<usize>>,
}

impl RangeAllocator {
    fn new(capacity: usize) -> Self {
        let mut free = Vec::with_capacity(1);
        free.push(0..capacity);
        Self { capacity, free }
    }

    fn reserve(&mut self, len: usize) -> Option<usize> {
        let index = self
            .free
            .iter()
            .position(|span| span.end.saturating_sub(span.start) >= len)?;
        let offset = self.free[index].start;
        self.free[index].start = self.free[index].start.saturating_add(len);
        if self.free[index].is_empty() {
            self.free.remove(index);
        }
        Some(offset)
    }

    fn release(&mut self, offset: usize, len: usize) {
        let released = offset..offset.saturating_add(len);
        assert!(
            released.end <= self.capacity,
            "released staging span exceeds chunk"
        );
        let index = self.free.partition_point(|span| span.start < released.start);
        if index > 0 {
            assert!(
                self.free[index - 1].end <= released.start,
                "released staging span overlaps a free predecessor"
            );
        }
        if index < self.free.len() {
            assert!(
                released.end <= self.free[index].start,
                "released staging span overlaps a free successor"
            );
        }
        self.free.insert(index, released);
        let mut merge_index = index.saturating_sub(1);
        while merge_index + 1 < self.free.len() {
            if self.free[merge_index].end != self.free[merge_index + 1].start {
                merge_index += 1;
                continue;
            }
            let end = self.free[merge_index + 1].end;
            self.free[merge_index].end = end;
            self.free.remove(merge_index + 1);
        }
    }
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    debug_assert!(alignment > 0);
    value
        .checked_add(alignment.saturating_sub(1))
        .map(|rounded| rounded / alignment * alignment)
}

fn next_chunk_size(previous: Option<usize>, requested: usize) -> usize {
    previous
        .map_or(INITIAL_UPLOAD_ARENA_BYTES, |size| size.saturating_mul(2))
        .max(requested)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX)
}

fn bounded_chunk_size(
    previous: Option<usize>,
    requested: usize,
    remaining: usize,
    existing_chunks: usize,
) -> usize {
    if existing_chunks + 1 == MAX_UPLOAD_ARENA_CHUNKS {
        return remaining;
    }
    next_chunk_size(previous, requested).min(remaining)
}

fn pick_host_visible_memory_type(
    physical_device: &PhysicalDevice,
    memory_type_bits: u32,
) -> Option<(u32, bool)> {
    let properties = unsafe {
        physical_device
            .instance()
            .handle()
            .get_physical_device_memory_properties(physical_device.handle())
    };
    let mut fallback = None;
    for index in 0..properties.memory_type_count {
        if memory_type_bits & (1u32 << index) == 0 {
            continue;
        }
        let flags = properties.memory_types[index as usize].property_flags;
        if !flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            continue;
        }
        let coherent = flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        if coherent {
            return Some((index, true));
        }
        fallback.get_or_insert((index, false));
    }
    fallback
}

#[cfg(test)]
mod tests {
    use super::{
        align_up, bounded_chunk_size, next_chunk_size, RangeAllocator, INITIAL_UPLOAD_ARENA_BYTES,
        MAX_UPLOAD_ARENA_BYTES,
    };

    #[test]
    fn allocator_never_reuses_a_live_span_and_coalesces_retired_neighbors() {
        let mut allocator = RangeAllocator::new(64);
        let first = allocator.reserve(16).unwrap();
        let second = allocator.reserve(16).unwrap();
        let third = allocator.reserve(16).unwrap();
        assert_eq!((first, second, third), (0, 16, 32));

        allocator.release(second, 16);
        assert_eq!(allocator.reserve(8), Some(16));
        assert_eq!(allocator.reserve(8), Some(24));
        assert_eq!(allocator.reserve(16), Some(48));
        assert_eq!(allocator.reserve(1), None);

        allocator.release(first, 16);
        allocator.release(16, 8);
        allocator.release(24, 8);
        allocator.release(third, 16);
        allocator.release(48, 16);
        assert_eq!(allocator.free, vec![0..64]);
    }

    #[test]
    fn alignment_rounds_to_non_coherent_atom_domains() {
        assert_eq!(align_up(1, 256), Some(256));
        assert_eq!(align_up(256, 256), Some(256));
        assert_eq!(align_up(257, 256), Some(512));
        assert_eq!(align_up(usize::MAX, 256), None);
    }

    #[test]
    fn first_upload_allocates_only_the_initial_chunk() {
        assert_eq!(next_chunk_size(None, 4_096), INITIAL_UPLOAD_ARENA_BYTES);
        assert_eq!(
            next_chunk_size(Some(INITIAL_UPLOAD_ARENA_BYTES), 4_096),
            INITIAL_UPLOAD_ARENA_BYTES * 2
        );
    }

    #[test]
    fn final_chunk_uses_the_remaining_bounded_budget() {
        let first = bounded_chunk_size(None, 4_096, MAX_UPLOAD_ARENA_BYTES, 0);
        let second = bounded_chunk_size(Some(first), 4_096, MAX_UPLOAD_ARENA_BYTES - first, 1);
        let third = bounded_chunk_size(Some(second), 4_096, MAX_UPLOAD_ARENA_BYTES - first - second, 2);
        let fourth = bounded_chunk_size(
            Some(third),
            4_096,
            MAX_UPLOAD_ARENA_BYTES - first - second - third,
            3,
        );

        assert_eq!(
            [first, second, third, fourth],
            [16, 32, 64, 144].map(|mib| mib * 1024 * 1024)
        );
        assert_eq!(first + second + third + fourth, MAX_UPLOAD_ARENA_BYTES);
    }

    #[test]
    fn randomized_reservation_and_fifo_completion_preserve_capacity() {
        let mut allocator = RangeAllocator::new(4096);
        let mut live = std::collections::VecDeque::new();
        let mut state = 0x4d59_5df4_d0f3_3173_u64;
        for _ in 0..20_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let len = (((state as usize) % 16) + 1) * 16;
            if let Some(offset) = allocator.reserve(len) {
                assert!(live.iter().all(|(other_offset, other_len)| {
                    offset + len <= *other_offset || *other_offset + *other_len <= offset
                }));
                live.push_back((offset, len));
            } else if let Some((offset, len)) = live.pop_front() {
                allocator.release(offset, len);
            }
            if state & 3 == 0 {
                if let Some((offset, len)) = live.pop_front() {
                    allocator.release(offset, len);
                }
            }
        }
        while let Some((offset, len)) = live.pop_front() {
            allocator.release(offset, len);
        }
        assert_eq!(allocator.free, vec![0..4096]);
    }
}
