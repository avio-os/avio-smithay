//! Imports live with their dma-buf, while submitted readers keep their own Arc.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, Weak},
};

use crate::backend::renderer::ErasedContextId;

#[derive(Debug)]
pub(super) struct ImportCustody<T> {
    // Contexts can share a device while importing distinct images. The
    // existing cold context token also prevents identity reuse while this
    // source still owns an import, without retaining a renderer or client.
    resources: Mutex<HashMap<ErasedContextId, Option<Arc<T>>>>,
}

impl<T> Default for ImportCustody<T> {
    fn default() -> Self {
        Self {
            resources: Mutex::new(HashMap::new()),
        }
    }
}

impl<T> ImportCustody<T> {
    pub(super) fn insert(&self, context: ErasedContextId, resource: Arc<T>) {
        let previous = self.resources.lock().unwrap().insert(context, Some(resource));
        // Driver destruction must happen after releasing the custody lock.
        drop(previous);
    }

    /// Cold preparation reserves the context slot without replacing its image.
    pub(super) fn reserve(&self, context: ErasedContextId) {
        self.resources.lock().unwrap().entry(context).or_insert(None);
    }

    /// The warm owner changes only a previously reserved slot. A busy registry
    /// defers adoption; old readers remain valid and no allocation is permitted.
    pub(super) fn try_replace_reserved(
        &self,
        context: &ErasedContextId,
        resource: Arc<T>,
    ) -> Result<Option<Arc<T>>, ()> {
        let mut resources = self.resources.try_lock().map_err(|_| ())?;
        let slot = resources.get_mut(context).ok_or(())?;
        Ok(slot.replace(resource))
    }

    /// The unique cold renderer owner may wait for the creating buffer's
    /// registry. Failure still preserves the prepared and previous images;
    /// it never invents a readiness edge for a poisoned or unreserved slot.
    pub(super) fn replace_reserved_cold(
        &self,
        context: &ErasedContextId,
        resource: Arc<T>,
    ) -> Result<Option<Arc<T>>, ()> {
        let mut resources = self.resources.lock().map_err(|_| ())?;
        let slot = resources.get_mut(context).ok_or(())?;
        Ok(slot.replace(resource))
    }

    pub(super) fn remove(&self, context: &ErasedContextId, expected: &Weak<T>) {
        let removed = {
            let mut resources = self.resources.lock().unwrap();
            if resources
                .get(context)
                .and_then(Option::as_ref)
                .is_some_and(|resource| Weak::ptr_eq(&Arc::downgrade(resource), expected))
            {
                resources.remove(context)
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

    fn context() -> ErasedContextId {
        crate::backend::renderer::ContextId::<crate::backend::renderer::vulkan::VulkanTexture>::new().erased()
    }

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
        buffer
            .resource_custody::<ImportCustody<u8>>()
            .insert(context(), resource);
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
            .insert(context(), reader.clone());
        drop(buffer);
        assert_eq!(weak.strong_count(), 1);
        drop(reader);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn preparation_reserves_without_replacing_the_current_image() {
        let registry = ImportCustody::default();
        let context = context();
        let original = Arc::new(7u8);
        registry.insert(context.clone(), original.clone());
        registry.reserve(context.clone());
        assert_eq!(Arc::strong_count(&original), 2);
        let previous = registry
            .try_replace_reserved(&context, Arc::new(9u8))
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&original, &previous));
        assert_eq!(
            Arc::strong_count(&original),
            2,
            "old reader and returned cold owner remain"
        );
    }

    #[test]
    fn warm_adoption_refuses_unreserved_context_and_busy_registry() {
        let registry = ImportCustody::<u8>::default();
        let context = context();
        let original = Arc::new(7u8);
        assert!(registry.try_replace_reserved(&context, original.clone()).is_err());
        registry.reserve(context.clone());
        let held = registry.resources.lock().unwrap();
        assert!(registry.try_replace_reserved(&context, original.clone()).is_err());
        drop(held);
        assert!(registry
            .try_replace_reserved(&context, original.clone())
            .unwrap()
            .is_none());
        let current = registry.resources.lock().unwrap();
        assert!(Arc::ptr_eq(
            current.get(&context).unwrap().as_ref().unwrap(),
            &original
        ));
    }

    #[test]
    fn cold_adoption_waits_for_the_exact_reserved_registry_without_losing_readers() {
        let registry = Arc::new(ImportCustody::default());
        let context = context();
        let original = Arc::new(7u8);
        registry.insert(context.clone(), original.clone());
        let held = registry.resources.lock().unwrap();
        let (started, received_start) = std::sync::mpsc::sync_channel(1);
        let (finished, received_finish) = std::sync::mpsc::sync_channel(1);
        let cold_registry = registry.clone();
        let cold_context = context.clone();
        let worker = std::thread::spawn(move || {
            started.send(()).unwrap();
            let previous = cold_registry
                .replace_reserved_cold(&cold_context, Arc::new(9u8))
                .unwrap();
            finished.send(()).unwrap();
            previous
        });
        received_start.recv().unwrap();
        assert!(matches!(
            received_finish.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        assert_eq!(Arc::strong_count(&original), 2);
        drop(held);
        let previous = worker.join().unwrap().unwrap();
        assert!(Arc::ptr_eq(&previous, &original));
        received_finish.recv().unwrap();
        assert_eq!(Arc::strong_count(&original), 2);
        let current = registry.resources.lock().unwrap();
        assert_eq!(**current.get(&context).unwrap().as_ref().unwrap(), 9);
    }

    #[test]
    fn cold_adoption_rejects_an_unreserved_slot_without_inserting_it() {
        let registry = ImportCustody::<u8>::default();
        let context = context();
        let prepared = Arc::new(7u8);
        assert!(registry
            .replace_reserved_cold(&context, prepared.clone())
            .is_err());
        assert!(registry.resources.lock().unwrap().is_empty());
        assert_eq!(Arc::strong_count(&prepared), 1);
    }

    #[test]
    fn renderer_retirement_removes_only_its_own_exact_import() {
        let buffer = buffer();
        let custody = buffer.resource_custody::<ImportCustody<u8>>();
        let first_context = context();
        let other_context = context();
        let first = Arc::new(1u8);
        let first_weak = Arc::downgrade(&first);
        let other = Arc::new(2u8);
        let other_weak = Arc::downgrade(&other);
        custody.insert(first_context.clone(), first);
        custody.insert(other_context.clone(), other);
        let replacement = Arc::new(3u8);
        let replacement_weak = Arc::downgrade(&replacement);
        custody.insert(first_context.clone(), replacement);
        custody.remove(&first_context, &first_weak);
        assert!(replacement_weak.upgrade().is_some());
        custody.remove(&first_context, &replacement_weak);
        assert!(replacement_weak.upgrade().is_none());
        assert!(other_weak.upgrade().is_some());
        custody.remove(&other_context, &other_weak);
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
            .insert(context(), image(device.clone(), 50));
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
