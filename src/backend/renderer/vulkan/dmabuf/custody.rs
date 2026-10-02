//! Imports live with their dma-buf, while submitted readers keep their own Arc.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, Weak},
};

#[derive(Debug)]
pub(super) struct ImportCustody<T> {
    resources: Mutex<HashMap<usize, Arc<T>>>,
}

impl<T> Default for ImportCustody<T> {
    fn default() -> Self {
        Self {
            resources: Mutex::new(HashMap::new()),
        }
    }
}

impl<T> ImportCustody<T> {
    pub(super) fn insert(&self, device: usize, resource: Arc<T>) {
        let previous = self.resources.lock().unwrap().insert(device, resource);
        // Driver destruction must happen after releasing the custody lock.
        drop(previous);
    }

    pub(super) fn remove(&self, device: usize, expected: &Weak<T>) {
        let removed = {
            let mut resources = self.resources.lock().unwrap();
            if resources
                .get(&device)
                .is_some_and(|resource| Weak::ptr_eq(&Arc::downgrade(resource), expected))
            {
                resources.remove(&device)
            } else {
                None
            }
        };
        drop(removed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::allocator::{
        dmabuf::{Dmabuf, DmabufFlags},
        Fourcc, Modifier,
    };

    fn buffer() -> Dmabuf {
        let fd = rustix::fs::memfd_create("import-custody", rustix::fs::MemfdFlags::CLOEXEC).unwrap();
        let mut buffer = Dmabuf::builder((1, 1), Fourcc::Argb8888, Modifier::Linear, DmabufFlags::empty());
        assert!(buffer.add_plane(fd, 0, 0, 4));
        buffer.build().unwrap()
    }

    #[test]
    fn last_buffer_drop_retires_import_without_another_import_or_cleanup() {
        let buffer = buffer();
        let resource = Arc::new(1u8);
        let weak = Arc::downgrade(&resource);
        buffer.resource_custody::<ImportCustody<u8>>().insert(1, resource);
        assert_eq!(weak.strong_count(), 1);
        drop(buffer);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn submitted_reader_keeps_retired_import_until_completion() {
        let buffer = buffer();
        let reader = Arc::new(1u8);
        let weak = Arc::downgrade(&reader);
        buffer
            .resource_custody::<ImportCustody<u8>>()
            .insert(1, reader.clone());
        drop(buffer);
        assert_eq!(weak.strong_count(), 1);
        drop(reader);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn renderer_retirement_removes_only_its_own_exact_import() {
        let buffer = buffer();
        let custody = buffer.resource_custody::<ImportCustody<u8>>();
        let first = Arc::new(1u8);
        let first_weak = Arc::downgrade(&first);
        let other = Arc::new(2u8);
        let other_weak = Arc::downgrade(&other);
        custody.insert(1, first);
        custody.insert(2, other);
        let replacement = Arc::new(3u8);
        let replacement_weak = Arc::downgrade(&replacement);
        custody.insert(1, replacement);
        custody.remove(1, &first_weak);
        assert!(replacement_weak.upgrade().is_some());
        custody.remove(1, &replacement_weak);
        assert!(replacement_weak.upgrade().is_none());
        assert!(other_weak.upgrade().is_some());
        custody.remove(2, &other_weak);
        assert!(other_weak.upgrade().is_none());
    }

    #[test]
    fn concurrent_renderers_share_one_buffer_custody_container() {
        let buffer = buffer();
        std::thread::scope(|scope| {
            let first = scope.spawn(|| buffer.resource_custody::<ImportCustody<u8>>());
            let second = scope.spawn(|| buffer.resource_custody::<ImportCustody<u8>>());
            assert!(Arc::ptr_eq(&first.join().unwrap(), &second.join().unwrap()));
        });
    }

    #[test]
    fn last_dmabuf_drop_queues_actual_image_destruction_on_idle_device() {
        use crate::backend::renderer::vulkan::{
            device_handle::retirement_tests::{device, image, next, Operation},
            image::VulkanImage,
        };

        let buffer = buffer();
        let (device, events) = device();
        buffer
            .resource_custody::<ImportCustody<VulkanImage>>()
            .insert(1, image(device.clone(), 50));
        let dropping_thread = std::thread::spawn(move || {
            let thread = std::thread::current().id();
            drop(buffer);
            thread
        })
        .join()
        .unwrap();
        let (operation, executor) = next(&events);
        assert_eq!(operation, Operation::View(52));
        assert_ne!(executor, dropping_thread);
        for operation in [Operation::View(53), Operation::Image(50), Operation::Memory(51)] {
            assert_eq!(next(&events), (operation, executor));
        }
        // No later import, frame or metadata cleanup was needed.
        drop(device);
        assert_eq!(next(&events), (Operation::Device, executor));
        assert_eq!(next(&events), (Operation::Parent, executor));
    }
}
