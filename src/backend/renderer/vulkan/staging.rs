use std::{fmt, ops::Range, sync::Arc};

use ash::vk;

use crate::backend::vulkan::PhysicalDevice;

use super::{
    allocation::{AllocationGuard, VulkanAllocationReason},
    device::DeviceHandle,
    VulkanRendererError,
};

const INITIAL_UPLOAD_ARENA_BYTES: usize = 16 * 1024 * 1024;
const MAX_UPLOAD_ARENA_BYTES: usize = 256 * 1024 * 1024;
const MAX_UPLOAD_ARENA_CHUNKS: usize = 4;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum ArenaMode {
    #[default]
    Growing,
    /// The owner provisions the one chunk outside frame work. Reservations
    /// never allocate, even when the chunk is empty or exhausted.
    Fixed { capacity: usize },
}

impl ArenaMode {
    fn next_chunk(
        self,
        previous: Option<usize>,
        requested: usize,
        capacity: usize,
        chunks: usize,
    ) -> Option<usize> {
        match self {
            Self::Growing if chunks < MAX_UPLOAD_ARENA_CHUNKS => Some(bounded_chunk_size(
                previous,
                requested,
                MAX_UPLOAD_ARENA_BYTES.saturating_sub(capacity),
                chunks,
            )),
            _ => None,
        }
    }

    fn max_contiguous(self, existing: usize, capacity: usize, chunks: usize) -> usize {
        match self {
            Self::Growing if chunks < MAX_UPLOAD_ARENA_CHUNKS => {
                existing.max(MAX_UPLOAD_ARENA_BYTES.saturating_sub(capacity))
            }
            Self::Fixed { capacity } => capacity,
            _ => existing,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StagingReservation {
    owner_sized: bool,
    chunk: usize,
    offset: usize,
    len: usize,
    reserved_len: usize,
}

impl StagingReservation {
    pub(crate) fn owner_sized(self) -> bool {
        self.owner_sized
    }
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
    mode: ArenaMode,
    owner_sized: bool,
}

impl fmt::Debug for UploadArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UploadArena")
            .field("atom_size", &self.atom_size)
            .field("stats", &self.stats)
            .field("mode", &self.mode)
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
            mode: ArenaMode::default(),
            owner_sized: false,
        }
    }

    pub(crate) fn owner_sized(physical_device: &PhysicalDevice) -> Self {
        Self {
            mode: ArenaMode::Fixed { capacity: 0 },
            owner_sized: true,
            ..Self::new(physical_device)
        }
    }

    /// Provision, shrink, or retire owner-sized storage. This is a lifecycle
    /// operation, never a reservation-path fallback. A detached writer keeps
    /// the old mapping alive even after cancellation; it must return before
    /// storage can be replaced. No completion source is waited on here.
    pub(crate) fn configure_fixed(
        &mut self,
        physical_device: &PhysicalDevice,
        device: Arc<DeviceHandle>,
        capacity: usize,
    ) -> Result<bool, VulkanRendererError> {
        let capacity = align_up(capacity, self.atom_size).ok_or(VulkanRendererError::InvalidMemoryUpload(
            "fixed upload capacity overflowed",
        ))?;
        if self.mode == (ArenaMode::Fixed { capacity }) {
            return Ok(true);
        }
        if self.stats.in_use_bytes != 0
            || self
                .chunks
                .iter()
                .any(|chunk| Arc::strong_count(&chunk.memory) != 1)
        {
            return Ok(false);
        }
        // Failure preserves the old storage and its mode. The replacement is
        // allocated before the old chunk is destroyed, outside frame work.
        let replacement = if capacity == 0 {
            None
        } else {
            Some(StagingChunk::new(
                physical_device,
                device,
                capacity,
                self.atom_size,
            )?)
        };
        self.chunks.clear();
        self.chunks.extend(replacement);
        self.mode = ArenaMode::Fixed { capacity };
        self.stats.capacity_bytes = capacity;
        self.stats.chunk_count = usize::from(capacity != 0);
        Ok(true)
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

        if let Some(chunk_size) = self.mode.next_chunk(
            self.chunks.last().map(StagingChunk::capacity),
            reserved_len,
            self.stats.capacity_bytes,
            self.chunks.len(),
        ) {
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

        let max_existing = self.chunks.iter().map(StagingChunk::capacity).max().unwrap_or(0);
        let max_contiguous_bytes =
            self.mode
                .max_contiguous(max_existing, self.stats.capacity_bytes, self.chunks.len());
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
            // Detached rows are an exclusive writer. Even a prematurely
            // cancelled ticket cannot make its mapping writable by another
            // reservation before that writer returns. A fixed owner waits
            // whole; it never creates a second chunk around this custody.
            if Arc::strong_count(&chunk.memory) != 1 {
                continue;
            }
            let Some(offset) = chunk.ranges.reserve(reserved_len) else {
                continue;
            };
            self.stats.in_use_bytes = self.stats.in_use_bytes.saturating_add(reserved_len);
            self.stats.high_water_bytes = self.stats.high_water_bytes.max(self.stats.in_use_bytes);
            return Some(StagingReservation {
                owner_sized: self.owner_sized,
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
        chunk
            .memory
            .write_rows(reservation.offset, data, src_offset, src_stride, row_bytes, rows)?;
        chunk.memory.flush(reservation.offset, reservation.reserved_len)
    }

    pub(crate) fn buffer(&self, reservation: StagingReservation) -> vk::Buffer {
        self.chunks[reservation.chunk].memory.buffer
    }

    /// The reservation's bytes as an exclusive writable view that may move to
    /// another thread. The returned mapping owner keeps the bytes valid even if
    /// the arena is dropped first.
    pub(crate) fn detach(
        &self,
        reservation: StagingReservation,
    ) -> Result<(*mut u8, Arc<ChunkMemory>), VulkanRendererError> {
        let chunk = self
            .chunks
            .get(reservation.chunk)
            .ok_or(VulkanRendererError::TemporaryFailure(
                "staging reservation names an unknown chunk",
            ))?;
        // SAFETY: The reservation lies inside the chunk's persistent mapping.
        let ptr = unsafe { chunk.memory.mapped.0.add(reservation.offset) };
        Ok((ptr, Arc::clone(&chunk.memory)))
    }

    /// Make host writes to the reservation visible to the device.
    pub(crate) fn flush(&self, reservation: StagingReservation) -> Result<(), VulkanRendererError> {
        let chunk = self
            .chunks
            .get(reservation.chunk)
            .ok_or(VulkanRendererError::TemporaryFailure(
                "staging reservation names an unknown chunk",
            ))?;
        chunk.memory.flush(reservation.offset, reservation.reserved_len)
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
    memory: Arc<ChunkMemory>,
    ranges: RangeAllocator,
}

/// One chunk's buffer, memory and persistent mapping. Shared with detached
/// writers so a reservation being written off-thread outlives the arena.
pub(crate) struct ChunkMemory {
    device: Arc<DeviceHandle>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: MappedAddress,
    coherent: bool,
    _allocation: AllocationGuard,
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
            match pick_host_visible_memory_type(physical_device, memory_requirements.memory_type_bits) {
                Some(memory_type) => memory_type,
                None => {
                    device.destroy_with(|vk_device| unsafe { vk_device.destroy_buffer(buffer, None) });
                    return Err(VulkanRendererError::NoCompatibleMemoryType);
                }
            };
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
        let allocation = device
            .allocation_ledger()
            .record(VulkanAllocationReason::Upload, allocation_size);
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
            memory: Arc::new(ChunkMemory {
                device,
                buffer,
                memory,
                mapped,
                coherent,
                _allocation: allocation,
            }),
            ranges: RangeAllocator::new(capacity),
        })
    }

    fn capacity(&self) -> usize {
        self.ranges.capacity
    }
}

impl ChunkMemory {
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

impl Drop for ChunkMemory {
    fn drop(&mut self) {
        self.device.destroy_with(|device| unsafe {
            device.unmap_memory(self.memory);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        });
    }
}

/// The Vulkan renderer is moved between worker setup and its final queue-owner
/// thread, and a detached reservation is written on another thread. Every
/// write goes to a reserved range no one else touches until it is handed
/// back; mapping and unmapping stay with the owning chunk memory.
struct MappedAddress(*mut u8);

// SAFETY: See the type-level ownership argument above. The address itself is
// immutable; disjoint reservations never alias, and the mapping is unmapped
// only when the last owner of the chunk memory drops it.
unsafe impl Send for MappedAddress {}
// SAFETY: As above: shared references only read the base address.
unsafe impl Sync for MappedAddress {}

struct RangeAllocator {
    capacity: usize,
    free: Vec<Range<usize>>,
}

impl RangeAllocator {
    fn new(capacity: usize) -> Self {
        let free = std::iter::once(0..capacity).collect();
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
        align_up, bounded_chunk_size, next_chunk_size, ArenaMode, RangeAllocator, INITIAL_UPLOAD_ARENA_BYTES,
        MAX_UPLOAD_ARENA_BYTES,
    };

    #[test]
    fn owner_sized_ring_never_grows_when_a_whole_generation_waits() {
        let mode = ArenaMode::Fixed { capacity: 4096 };
        let mut ring = RangeAllocator::new(4096);
        let first = ring.reserve(3072).unwrap();
        assert_eq!(
            ring.reserve(2048),
            None,
            "a generation is never partially admitted"
        );
        assert_eq!(mode.next_chunk(Some(4096), 2048, 4096, 1), None);
        assert_eq!(mode.max_contiguous(4096, 4096, 1), 4096);
        ring.release(first, 3072);
        assert_eq!(
            ring.reserve(2048),
            Some(0),
            "completion admits the whole generation"
        );
    }

    #[test]
    fn generation_does_not_land_in_bands_across_fragmented_free_spans() {
        let mut ring = RangeAllocator::new(4096);
        let first = ring.reserve(1024).unwrap();
        let second = ring.reserve(1024).unwrap();
        let third = ring.reserve(2048).unwrap();
        ring.release(first, 1024);
        ring.release(third, 2048);
        assert_eq!(
            ring.reserve(3072),
            None,
            "enough total free bytes cannot authorize a banded upload"
        );
        ring.release(second, 1024);
        assert_eq!(ring.reserve(3072), Some(0));
    }

    #[test]
    #[ignore = "[laptop] requires the renderer Vulkan device extensions"]
    fn owner_ring_shrinks_and_retires_only_after_detached_writer_returns() {
        use crate::backend::allocator::Fourcc;
        use crate::backend::renderer::ImportMem;
        let Some(physical) = super::super::test_support::physical_device() else {
            return;
        };
        let Some(mut renderer) = super::super::test_support::renderer(&physical) else {
            return;
        };
        assert!(renderer.configure_memory_upload_capacity(4096).unwrap());
        assert_eq!(renderer.diagnostics().uploads.arena_capacity_bytes, 4096);
        let (_, update, rows) = renderer
            .stage_memory_import(Fourcc::Argb8888, (16, 16).into(), false)
            .unwrap()
            .unwrap();
        assert!(
            !renderer.configure_memory_upload_capacity(2048).unwrap(),
            "a live generation owns its reservation"
        );
        renderer.cancel_staged_memory_update(update);
        assert!(
            !renderer.configure_memory_upload_capacity(2048).unwrap(),
            "a detached writer still owns the mapping"
        );
        assert!(
            renderer
                .stage_memory_import(Fourcc::Argb8888, (16, 16).into(), false)
                .unwrap_err()
                .is_upload_deferred(),
            "cancelled bytes cannot be reused while the detached writer exists"
        );
        drop(rows);
        assert!(renderer.configure_memory_upload_capacity(2048).unwrap());
        assert_eq!(renderer.diagnostics().uploads.arena_capacity_bytes, 2048);
        assert!(renderer.configure_memory_upload_capacity(0).unwrap());
        assert_eq!(renderer.diagnostics().uploads.arena_capacity_bytes, 0);
        assert_eq!(renderer.diagnostics().uploads.arena_chunk_count, 0);
    }

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
