//! Optional sealed foreign-host-memory uploads, prepared before a render turn.

use super::{
    allocation::{AllocationGuard, VulkanAllocationReason},
    device::DeviceHandle,
    device_handle::DeviceRetirement,
    retirement::RetirementNode,
    VulkanRendererError,
};
use crate::reexports::wayland_server::Resource;
use crate::{
    backend::{
        renderer::{utils::Buffer, MemoryHostUnavailable},
        vulkan::PhysicalDevice,
    },
    wayland::shm::{host_memory::HostMapping, ShmBufferUserData},
};
use ash::{ext, vk};
use std::sync::Arc;

pub(super) struct HostMemorySupport {
    pub(super) alignment: Result<usize, MemoryHostUnavailable>,
    dedicated: bool,
    loader: Option<ext::external_memory_host::Device>,
}

impl HostMemorySupport {
    pub(super) fn new(physical: &PhysicalDevice, device: &ash::Device, enabled: bool) -> Self {
        let instance = physical.instance().handle();
        let mut result = Self {
            alignment: Err(MemoryHostUnavailable::Extension),
            dedicated: false,
            loader: None,
        };
        if !enabled {
            return result;
        }
        let info = vk::PhysicalDeviceExternalBufferInfo::default()
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_MAPPED_FOREIGN_MEMORY_EXT);
        let mut properties = vk::ExternalBufferProperties::default();
        unsafe {
            instance.get_physical_device_external_buffer_properties(
                physical.handle(),
                &info,
                &mut properties,
            );
        }
        let properties = properties.external_memory_properties;
        if !properties
            .external_memory_features
            .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
            || !properties
                .compatible_handle_types
                .contains(vk::ExternalMemoryHandleTypeFlags::HOST_MAPPED_FOREIGN_MEMORY_EXT)
        {
            result.alignment = Err(MemoryHostUnavailable::BufferUsage);
            return result;
        }
        result.dedicated = properties
            .external_memory_features
            .contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY);
        let mut host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut host);
        unsafe {
            instance.get_physical_device_properties2(physical.handle(), &mut properties);
        }
        result.alignment = usize::try_from(host.min_imported_host_pointer_alignment)
            .ok()
            .filter(|alignment| alignment.is_power_of_two())
            .ok_or(MemoryHostUnavailable::Alignment);
        result.loader = Some(ext::external_memory_host::Device::new(instance, device));
        result
    }

    pub(super) fn import(
        &self,
        physical: &PhysicalDevice,
        device: Arc<DeviceHandle>,
        source: &Buffer,
    ) -> Result<Result<Arc<HostBuffer>, MemoryHostUnavailable>, VulkanRendererError> {
        let alignment = match self.alignment {
            Ok(value) => value,
            Err(reason) => return Ok(Err(reason)),
        };
        let Some(data) = source.data::<ShmBufferUserData>() else {
            return Ok(Err(MemoryHostUnavailable::Layout));
        };
        let layout = data.data;
        let Some((offset, end)) = checked_extent(layout.offset, layout.width, layout.height, layout.stride)
        else {
            return Ok(Err(MemoryHostUnavailable::Layout));
        };
        let mapping = match HostMapping::new(
            data.pool.duplicate_fd()?,
            offset,
            end,
            data.pool.size(),
            alignment,
        )? {
            Ok(mapping) => mapping,
            Err(reason) => return Ok(Err(reason)),
        };
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_MAPPED_FOREIGN_MEMORY_EXT);
        let info = vk::BufferCreateInfo::default()
            .size(mapping.len() as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        let buffer = device.observe_result(crate::backend::allocator::observe_gpu_allocation(
            unsafe { device.handle().create_buffer(&info, None) },
            crate::backend::allocator::GpuAllocationKind::VulkanBuffer,
        ))?;
        let requirements = unsafe { device.handle().get_buffer_memory_requirements(buffer) };
        if requirements.size > mapping.len() as u64 {
            device.destroy_with(|raw| unsafe {
                raw.destroy_buffer(buffer, None);
            });
            return Ok(Err(MemoryHostUnavailable::Alignment));
        }
        let mut pointer_properties = vk::MemoryHostPointerPropertiesEXT::default();
        let loader = self.loader.as_ref().expect("supported host-memory loader");
        let queried = unsafe {
            (loader.fp().get_memory_host_pointer_properties_ext)(
                device.handle().handle(),
                vk::ExternalMemoryHandleTypeFlags::HOST_MAPPED_FOREIGN_MEMORY_EXT,
                mapping.pointer(),
                &mut pointer_properties,
            )
        };
        if queried != vk::Result::SUCCESS {
            let _ = device.observe_result(Err::<(), _>(queried));
            device.destroy_with(|raw| unsafe {
                raw.destroy_buffer(buffer, None);
            });
            return if queried == vk::Result::ERROR_INVALID_EXTERNAL_HANDLE {
                Ok(Err(MemoryHostUnavailable::Pointer))
            } else {
                Err(queried.into())
            };
        }
        let properties = unsafe {
            physical
                .instance()
                .handle()
                .get_physical_device_memory_properties(physical.handle())
        };
        let bits = requirements.memory_type_bits & pointer_properties.memory_type_bits;
        let Some(memory_type) = coherent_memory_type(bits, &properties) else {
            device.destroy_with(|raw| unsafe {
                raw.destroy_buffer(buffer, None);
            });
            return Ok(Err(MemoryHostUnavailable::MemoryType));
        };
        let allocation_size = if self.dedicated {
            requirements.size
        } else {
            mapping.len() as u64
        };
        if allocation_size % alignment as u64 != 0 {
            device.destroy_with(|raw| unsafe {
                raw.destroy_buffer(buffer, None);
            });
            return Ok(Err(MemoryHostUnavailable::Alignment));
        }
        let mut host = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_MAPPED_FOREIGN_MEMORY_EXT)
            .host_pointer(mapping.pointer());
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let mut allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(allocation_size)
            .memory_type_index(memory_type)
            .push_next(&mut host);
        if self.dedicated {
            allocation = allocation.push_next(&mut dedicated);
        }
        let memory = match device.observe_result(crate::backend::allocator::observe_gpu_allocation(
            unsafe { device.handle().allocate_memory(&allocation, None) },
            crate::backend::allocator::GpuAllocationKind::VulkanDeviceMemory,
        )) {
            Ok(memory) => memory,
            Err(error) => {
                device.destroy_with(|raw| unsafe {
                    raw.destroy_buffer(buffer, None);
                });
                return if error == vk::Result::ERROR_INVALID_EXTERNAL_HANDLE {
                    Ok(Err(MemoryHostUnavailable::Pointer))
                } else {
                    Err(error.into())
                };
            }
        };
        let guard = device
            .allocation_ledger()
            .record(VulkanAllocationReason::Import, allocation_size);
        let source_offset = mapping.source_offset as u64;
        let retirement = RetirementNode::new(DeviceRetirement::HostBuffer(RetiredHostBuffer {
            buffer,
            memory,
            _allocation: guard,
            _mapping: mapping,
            _source: source.clone(),
        }));
        if let Err(error) =
            device.observe_result(unsafe { device.handle().bind_buffer_memory(buffer, memory, 0) })
        {
            // vkAllocateMemory already imported this pointer. Even a failed
            // bind must retain the mapping until memory destruction; on
            // DEVICE_LOST the executor quarantines the exact native owner.
            device.retire_resource(retirement);
            return Err(error.into());
        }
        Ok(Ok(Arc::new(HostBuffer {
            device,
            buffer,
            source_offset,
            retirement: Some(retirement),
        })))
    }
}

fn coherent_memory_type(bits: u32, properties: &vk::PhysicalDeviceMemoryProperties) -> Option<u32> {
    (0..properties.memory_type_count).find(|index| {
        bits & (1 << index) != 0
            && properties.memory_types[*index as usize]
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
    })
}

pub(super) fn checked_extent(offset: i32, width: i32, height: i32, stride: i32) -> Option<(usize, usize)> {
    let (offset, width, height, stride) = (
        usize::try_from(offset).ok()?,
        usize::try_from(width).ok()?,
        usize::try_from(height).ok()?,
        usize::try_from(stride).ok()?,
    );
    if width == 0 || height == 0 || offset % 4 != 0 || stride % 4 != 0 || stride < width.checked_mul(4)? {
        return None;
    }
    let end = offset
        .checked_add((height - 1).checked_mul(stride)?)?
        .checked_add(width.checked_mul(4)?)?;
    Some((offset, end))
}

pub(super) struct HostBuffer {
    device: Arc<DeviceHandle>,
    pub(super) buffer: vk::Buffer,
    pub(super) source_offset: u64,
    retirement: Option<Box<RetirementNode<DeviceRetirement>>>,
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        if let Some(node) = self.retirement.take() {
            self.device.retire_resource(node);
        }
    }
}

pub(super) struct RetiredHostBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    _allocation: AllocationGuard,
    _mapping: HostMapping,
    _source: Buffer,
}

impl RetiredHostBuffer {
    pub(super) fn destroy(self, device: &ash::Device) {
        // The mapping, FD and exact attachment owner drop after the Vulkan
        // binding is freed, on this executor, never on the final reader's turn.
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn foreign_layout_requires_copy_compatible_pitch_and_checked_extent() {
        assert_eq!(checked_extent(4, 2, 3, 16), Some((4, 44)));
        assert_eq!(checked_extent(1, 2, 3, 16), None);
        assert_eq!(checked_extent(0, 2, 3, 9), None);
        assert_eq!(checked_extent(0, 3, 3, 8), None);
        assert_eq!(checked_extent(-1, 2, 3, 8), None);
        assert_eq!(checked_extent(0, 0, 3, 8), None);
    }
    #[test]
    fn pointer_and_buffer_memory_types_must_intersect_and_be_coherent() {
        let mut properties = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 3,
            ..Default::default()
        };
        properties.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        properties.memory_types[1].property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE;
        properties.memory_types[2].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        assert_eq!(coherent_memory_type(0b011, &properties), None);
        assert_eq!(coherent_memory_type(0b100, &properties), Some(2));
        assert_eq!(coherent_memory_type(0, &properties), None);
    }

    #[test]
    fn submitted_reader_keeps_host_binding_mapping_and_device_until_executor_free() {
        use super::super::device_handle::retirement_tests::{device, Operation};
        use crate::reexports::wayland_server::{
            protocol::wl_buffer, Client, DataInit, Dispatch, Display, DisplayHandle,
        };
        use ash::vk::Handle;
        use std::{
            os::fd::{AsRawFd, FromRawFd, OwnedFd},
            os::unix::net::UnixStream,
            time::Duration,
        };
        struct State;
        impl Dispatch<wl_buffer::WlBuffer, ()> for State {
            fn request(
                _: &mut Self,
                _: &Client,
                _: &wl_buffer::WlBuffer,
                _: wl_buffer::Request,
                _: &(),
                _: &DisplayHandle,
                _: &mut DataInit<'_, Self>,
            ) {
            }
        }
        let display = Display::<State>::new().unwrap();
        let mut handle = display.handle();
        let (server, _client) = UnixStream::pair().unwrap();
        let client = handle.insert_client(server, Arc::new(())).unwrap();
        let resource = client
            .create_resource::<wl_buffer::WlBuffer, (), State>(&handle, 1, ())
            .unwrap();
        let source = Buffer::with_implicit(resource);
        let raw = unsafe {
            libc::memfd_create(
                c"retired-host-test".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            )
        };
        assert!(raw >= 0);
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        assert_eq!(unsafe { libc::ftruncate(fd.as_raw_fd(), 4096) }, 0);
        assert_eq!(
            unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, libc::F_SEAL_SHRINK) },
            0
        );
        let mapping = HostMapping::new(fd, 0, 4, 4096, 4096).unwrap().unwrap();
        let pointer = mapping.pointer();
        let (device, events) = device();
        let census = device.allocation_ledger().clone();
        let guard = census.record(VulkanAllocationReason::Import, 4096);
        let buffer = Arc::new(HostBuffer {
            device: device.clone(),
            buffer: vk::Buffer::from_raw(101),
            source_offset: 0,
            retirement: Some(RetirementNode::new(DeviceRetirement::HostBuffer(
                RetiredHostBuffer {
                    buffer: vk::Buffer::from_raw(101),
                    memory: vk::DeviceMemory::from_raw(102),
                    _allocation: guard,
                    _mapping: mapping,
                    _source: source,
                },
            ))),
        });
        let submitted = buffer.clone();
        drop(buffer);
        drop(device);
        assert!(
            events.try_recv().is_err(),
            "submitted reader owns exact mapping and logical device"
        );
        assert_eq!(unsafe { pointer.cast::<u8>().read_volatile() }, 0);
        let caller = std::thread::current().id();
        std::thread::spawn(move || drop(submitted)).join().unwrap();
        let (event, executor) = events.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(event, Operation::Buffer(101));
        assert_ne!(executor, caller);
        for expected in [Operation::Memory(102), Operation::Device, Operation::Parent] {
            let (event, thread) = events.recv_timeout(Duration::from_secs(2)).unwrap();
            assert_eq!(event, expected);
            assert_eq!(thread, executor);
        }
        assert_eq!(
            census
                .snapshot()
                .reason(VulkanAllocationReason::Import)
                .live_allocations,
            0
        );
    }

    #[test]
    fn gpu_sealed_host_copy_preserves_offset_pitch_without_staging() {
        use crate::backend::{
            allocator::Fourcc,
            renderer::{ExportMem, ImportMem, MemoryHostUpload},
        };
        use crate::reexports::wayland_server::{
            protocol::{wl_buffer, wl_shm},
            Client, DataInit, Dispatch, Display, DisplayHandle,
        };
        use crate::{utils::Rectangle, wayland::shm::BufferData};
        use std::{
            os::fd::{AsRawFd, FromRawFd, OwnedFd},
            os::unix::net::UnixStream,
        };
        let Some(physical) = super::super::test_support::physical_device() else {
            return;
        };
        let Some(mut renderer) = super::super::test_support::renderer(&physical) else {
            return;
        };
        if let Err(reason) = renderer.host_memory_import_alignment() {
            super::super::test_support::capability_unavailable(format_args!(
                "foreign host-memory transfer source: {reason:?}"
            ));
            return;
        }
        struct State;
        impl Dispatch<wl_buffer::WlBuffer, ShmBufferUserData> for State {
            fn request(
                _: &mut Self,
                _: &Client,
                _: &wl_buffer::WlBuffer,
                _: wl_buffer::Request,
                _: &ShmBufferUserData,
                _: &DisplayHandle,
                _: &mut DataInit<'_, Self>,
            ) {
            }
        }
        let display = Display::<State>::new().unwrap();
        let mut handle = display.handle();
        let (server, _client) = UnixStream::pair().unwrap();
        let client = handle.insert_client(server, Arc::new(())).unwrap();
        let raw = unsafe {
            libc::memfd_create(
                c"gpu-host-copy-test".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            )
        };
        assert!(raw >= 0);
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        assert_eq!(unsafe { libc::ftruncate(fd.as_raw_fd(), 4096) }, 0);
        let first = [12, 34, 56, 255, 78, 90, 123, 255];
        let second = [145, 167, 189, 255, 201, 223, 245, 255];
        assert_eq!(
            unsafe { libc::pwrite(fd.as_raw_fd(), first.as_ptr().cast(), 8, 4) },
            8
        );
        assert_eq!(
            unsafe { libc::pwrite(fd.as_raw_fd(), second.as_ptr().cast(), 8, 20) },
            8
        );
        let metadata = BufferData {
            offset: 4,
            width: 2,
            height: 2,
            stride: 16,
            format: wl_shm::Format::Argb8888,
        };
        let unsealed = ShmBufferUserData::test_data(fd.try_clone().unwrap(), 4096, metadata);
        let unsealed = Buffer::with_implicit(
            client
                .create_resource::<wl_buffer::WlBuffer, _, State>(&handle, 1, unsealed)
                .unwrap(),
        );
        assert!(matches!(
            renderer
                .import_host_shm(&unsealed, Fourcc::Argb8888, (2, 2).into())
                .unwrap(),
            MemoryHostUpload::Unavailable(MemoryHostUnavailable::Unsealed)
        ));
        drop(unsealed);
        assert_eq!(
            unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, libc::F_SEAL_SHRINK) },
            0
        );
        let data = ShmBufferUserData::test_data(fd, 4096, metadata);
        let source = Buffer::with_implicit(
            client
                .create_resource::<wl_buffer::WlBuffer, _, State>(&handle, 1, data)
                .unwrap(),
        );
        let before = renderer.diagnostics().uploads;
        let texture = match renderer
            .import_host_shm(&source, Fourcc::Argb8888, (2, 2).into())
            .unwrap()
        {
            MemoryHostUpload::Queued(texture) => texture,
            MemoryHostUpload::Unavailable(reason) => {
                super::super::test_support::capability_unavailable(format_args!(
                    "sealed host source import: {reason:?}"
                ));
                return;
            }
        };
        drop(source); // pending/submitted source custody now belongs to the GPU operation
        let after = renderer.diagnostics().uploads;
        assert_eq!(after.arena_capacity_bytes, before.arena_capacity_bytes);
        assert_eq!(after.arena_in_use_bytes, before.arena_in_use_bytes);
        assert_eq!(after.pending_operations, before.pending_operations + 1);
        let mapping = renderer
            .copy_texture(&texture, Rectangle::from_size((2, 2).into()), Fourcc::Argb8888)
            .unwrap();
        let bytes = renderer.map_texture(&mapping).unwrap();
        assert_eq!(bytes, [first, second].concat());
    }
}
