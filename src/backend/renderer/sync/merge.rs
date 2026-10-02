//! Kernel AND-join of Linux sync_file completion fences.

use std::{
    io,
    os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd},
};

/// Return a fence that becomes readable only after both input fences signal.
/// This never waits; an ioctl failure returns no weaker completion evidence.
pub fn merge_sync_files(first: BorrowedFd<'_>, second: BorrowedFd<'_>) -> io::Result<OwnedFd> {
    #[repr(C)]
    struct SyncMergeData {
        name: [u8; 32],
        fd2: i32,
        fence: i32,
        flags: u32,
        pad: u32,
    }

    const SYNC_IOC_MERGE: rustix::ioctl::Opcode = rustix::ioctl::opcode::read_write::<SyncMergeData>(b'>', 3);
    let mut data = SyncMergeData {
        name: [0; 32],
        fd2: second.as_raw_fd(),
        fence: -1,
        flags: 0,
        pad: 0,
    };
    // SAFETY: `first` and `second` are live sync_file descriptors and `data`
    // matches the kernel's `struct sync_merge_data` layout.
    unsafe {
        rustix::ioctl::ioctl(
            first,
            rustix::ioctl::Updater::<SYNC_IOC_MERGE, SyncMergeData>::new(&mut data),
        )
    }
    .map_err(io::Error::from)?;
    if data.fence < 0 {
        return Err(io::Error::other("SYNC_IOC_MERGE returned no fence"));
    }
    // SAFETY: a successful SYNC_IOC_MERGE returns a new owned descriptor.
    Ok(unsafe { OwnedFd::from_raw_fd(data.fence) })
}
