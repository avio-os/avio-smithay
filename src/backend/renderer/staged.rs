//! Memory texture updates whose pixels are written off the renderer thread.
//!
//! [`ImportMem::update_memory`](super::ImportMem::update_memory) copies the
//! caller's pixels into renderer staging memory on the calling thread. For a
//! large update that copy is the whole cost, and it lands in the frame being
//! built. A staged update splits it: the renderer reserves the staging bytes
//! ([`ImportMem::stage_memory_update`](super::ImportMem::stage_memory_update)),
//! any thread writes the rows through [`StagedMemoryRows`], and the renderer
//! applies the whole region at its next submission
//! ([`ImportMem::submit_staged_memory_update`](super::ImportMem::submit_staged_memory_update)).
//! Until then the texture keeps its previous pixels.

use std::{any::Any, fmt, sync::Arc};

use crate::utils::{Buffer as BufferCoord, Rectangle};

/// The renderer-side half of a staged memory update.
///
/// It names the texture region and the renderer's reservation. Hand it back to
/// the renderer that staged it, with its rows, to apply it, or cancel it there
/// to release the reservation. Dropping it elsewhere keeps the reservation
/// until the renderer itself is dropped.
pub struct StagedMemoryUpdate {
    ticket: u64,
    region: Rectangle<i32, BufferCoord>,
    inner: Box<dyn Any + Send>,
}

impl fmt::Debug for StagedMemoryUpdate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StagedMemoryUpdate")
            .field("ticket", &self.ticket)
            .field("region", &self.region)
            .finish_non_exhaustive()
    }
}

impl StagedMemoryUpdate {
    /// Wrap a renderer's own staging record. `ticket` pairs it with its rows.
    pub fn new(ticket: u64, region: Rectangle<i32, BufferCoord>, inner: Box<dyn Any + Send>) -> Self {
        Self {
            ticket,
            region,
            inner,
        }
    }

    /// The texture region this update replaces.
    pub fn region(&self) -> Rectangle<i32, BufferCoord> {
        self.region
    }

    /// The ticket pairing this update with its [`StagedMemoryRows`].
    pub fn ticket(&self) -> u64 {
        self.ticket
    }

    /// The renderer's own staging record.
    pub fn into_inner(self) -> Box<dyn Any + Send> {
        self.inner
    }
}

/// The pixel rows of a staged memory update, writable on any thread.
///
/// Rows are tightly packed in the texture's format, `row_bytes` each, top to
/// bottom across the staged region. Every row must be written before the rows
/// are handed back with their update.
pub struct StagedMemoryRows {
    ptr: *mut u8,
    row_bytes: usize,
    rows: usize,
    ticket: u64,
    _keepalive: Arc<dyn Any + Send + Sync>,
}

impl fmt::Debug for StagedMemoryRows {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StagedMemoryRows")
            .field("row_bytes", &self.row_bytes)
            .field("rows", &self.rows)
            .field("ticket", &self.ticket)
            .finish_non_exhaustive()
    }
}

// SAFETY: The rows are an exclusive view of a reserved staging range. No other
// code reads or writes that range until the rows are handed back, and
// `_keepalive` keeps its mapping alive on whichever thread holds the rows.
unsafe impl Send for StagedMemoryRows {}

impl StagedMemoryRows {
    /// Wrap `rows * row_bytes` writable bytes at `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for writes of `rows * row_bytes` bytes for as long
    /// as `keepalive` lives, and nothing else may access that range until the
    /// rows are handed back to the renderer that created them.
    pub unsafe fn new(
        ptr: *mut u8,
        row_bytes: usize,
        rows: usize,
        ticket: u64,
        keepalive: Arc<dyn Any + Send + Sync>,
    ) -> Self {
        Self {
            ptr,
            row_bytes,
            rows,
            ticket,
            _keepalive: keepalive,
        }
    }

    /// Bytes in one row.
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The ticket pairing these rows with their [`StagedMemoryUpdate`].
    pub fn ticket(&self) -> u64 {
        self.ticket
    }

    /// Row `row`, top to bottom.
    ///
    /// # Panics
    ///
    /// If `row` is not below [`Self::rows`].
    pub fn row_mut(&mut self, row: usize) -> &mut [u8] {
        assert!(row < self.rows, "staged row {row} out of {} rows", self.rows);
        // SAFETY: `new`'s contract makes the whole range writable and
        // exclusive to these rows; `row` is in bounds, and `&mut self` makes
        // the returned slice the only live reference into it.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(row * self.row_bytes), self.row_bytes) }
    }
}
