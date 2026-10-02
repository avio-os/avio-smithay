//! Independent read-only mappings for asynchronous device reads. A callback
//! address from `Pool::with_data` is never suitable: pool resize may move it.

use std::{
    io,
    os::fd::{AsRawFd, OwnedFd},
    ptr::NonNull,
};

use crate::backend::renderer::MemoryHostUnavailable;

/// No Rust reference to externally mutable pixels is ever constructed.
#[derive(Debug)]
pub(crate) struct HostMapping {
    pointer: NonNull<libc::c_void>,
    len: usize,
    _fd: OwnedFd,
    pub(crate) source_offset: usize,
}

// SAFETY: This owns a stable independent mapping. The kernel shrink seal
// prevents truncation. Pixel accesses belong to the external device, not Rust
// references; external writes may change pixels but cannot invalidate the map.
unsafe impl Send for HostMapping {}
// SAFETY: Sharing this owner exposes only an address for externally synchronized
// device reads, never shared Rust references to client-controlled bytes.
unsafe impl Sync for HostMapping {}

impl HostMapping {
    pub(crate) fn new(
        fd: OwnedFd,
        source_offset: usize,
        source_end: usize,
        pool_len: usize,
        alignment: usize,
    ) -> io::Result<Result<Self, MemoryHostUnavailable>> {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let Ok(page) = usize::try_from(page) else {
            return Ok(Err(MemoryHostUnavailable::Alignment));
        };
        if !page.is_power_of_two() || !alignment.is_power_of_two() {
            return Ok(Err(MemoryHostUnavailable::Alignment));
        }
        let alignment = alignment.max(page);
        let seals = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GET_SEALS) };
        if seals < 0 || seals & libc::F_SEAL_SHRINK == 0 {
            return Ok(Err(MemoryHostUnavailable::Unsealed));
        }
        if source_end <= source_offset || source_end > pool_len {
            return Ok(Err(MemoryHostUnavailable::Layout));
        }
        let start = source_offset & !(alignment - 1);
        let Some(end) = source_end
            .checked_add(alignment - 1)
            .map(|end| end & !(alignment - 1))
        else {
            return Ok(Err(MemoryHostUnavailable::Extent));
        };
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        if unsafe { libc::fstat(fd.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // Check after the monotonic seal: neither a concurrent client nor a
        // later pool resize can shrink the backing file below this extent.
        let stat = unsafe { stat.assume_init() };
        if u64::try_from(stat.st_size)
            .ok()
            .is_none_or(|size| size < end as u64)
        {
            return Ok(Err(MemoryHostUnavailable::Extent));
        }
        let len = end - start;
        let Ok(offset) = libc::off_t::try_from(start) else {
            return Ok(Err(MemoryHostUnavailable::Extent));
        };
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                offset,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let Some(pointer) = NonNull::new(pointer) else {
            unsafe {
                libc::munmap(pointer, len);
            }
            return Ok(Err(MemoryHostUnavailable::Pointer));
        };
        let mapping = Self {
            pointer,
            len,
            _fd: fd,
            source_offset: source_offset - start,
        };
        if pointer.as_ptr() as usize % alignment != 0 {
            return Ok(Err(MemoryHostUnavailable::Alignment));
        }
        Ok(Ok(mapping))
    }

    pub(crate) fn pointer(&self) -> *mut libc::c_void {
        self.pointer.as_ptr()
    }
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

impl Drop for HostMapping {
    fn drop(&mut self) {
        // A Vulkan owner releases this only after vkFreeMemory on the device
        // retirement executor. Failure cleanup happens on the cold prep turn.
        unsafe {
            libc::munmap(self.pointer.as_ptr(), self.len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::FromRawFd;

    fn backing(bytes: usize, sealed: bool) -> OwnedFd {
        let fd = unsafe {
            libc::memfd_create(
                c"host-map-test".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            )
        };
        assert!(fd >= 0);
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        assert_eq!(unsafe { libc::ftruncate(fd.as_raw_fd(), bytes as _) }, 0);
        if sealed {
            assert_eq!(
                unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, libc::F_SEAL_SHRINK) },
                0
            );
        }
        fd
    }

    #[test]
    fn unsealed_and_rounded_out_of_file_maps_are_rejected() {
        assert!(matches!(
            HostMapping::new(backing(8192, false), 0, 4096, 8192, 4096).unwrap(),
            Err(MemoryHostUnavailable::Unsealed)
        ));
        assert!(matches!(
            HostMapping::new(backing(4100, true), 0, 4100, 4100, 4096).unwrap(),
            Err(MemoryHostUnavailable::Extent)
        ));
        assert!(matches!(
            HostMapping::new(backing(8192, true), 0, usize::MAX, usize::MAX, 4096).unwrap(),
            Err(MemoryHostUnavailable::Extent)
        ));
    }

    #[test]
    fn mapping_owns_its_address_and_sealed_backing_after_original_fd_drops() {
        let fd = backing(8192, true);
        let copied = fd.try_clone().unwrap();
        let mapping = HostMapping::new(copied, 4096, 4100, 8192, 4096).unwrap().unwrap();
        assert_eq!(mapping.len(), 4096);
        assert_eq!(mapping.source_offset, 0);
        assert_eq!(unsafe { libc::ftruncate(fd.as_raw_fd(), 4096) }, -1);
        drop(fd);
        assert_eq!(unsafe { mapping.pointer().cast::<u8>().read_volatile() }, 0);
        assert_eq!(
            unsafe { libc::fcntl(mapping._fd.as_raw_fd(), libc::F_GET_SEALS) } & libc::F_SEAL_SHRINK,
            libc::F_SEAL_SHRINK
        );
    }
}
