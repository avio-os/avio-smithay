// Actual DmabufState, source custody and VulkanImage lifetimes. The fixture
// dispatches only native destructors; it does not load a Vulkan driver.
use crate::backend::renderer::vulkan::{
    device_handle::retirement_tests::{device, image, next, Operation},
    image::VulkanImage,
};
use std::sync::{mpsc::Receiver, Arc};

fn install_context_import(state: &mut DmabufState, buffer: &Dmabuf, image: &Arc<VulkanImage>) {
    let custody = buffer.resource_custody::<super::ImportCustody<VulkanImage>>();
    // Match production ordering: install the new strong owner first, then
    // drop stale metadata. Its exact weak-image check cannot remove the new
    // image or another renderer's image.
    custody.insert(state.context.clone(), image.clone());
    state.cache.shift_remove(&buffer.weak());
    let mut entry = metadata_entry(buffer, 1, state.context.clone());
    entry.imported = Arc::downgrade(image);
    entry.custody = Arc::downgrade(&custody);
    state.cache.insert(buffer.weak(), entry);
}

fn expect_context_image_retired(events: &Receiver<(Operation, std::thread::ThreadId)>, id: u64) {
    let (first, executor) = next(events);
    assert_eq!(first, Operation::View(id + 2));
    for expected in [
        Operation::View(id + 3),
        Operation::Image(id),
        Operation::Memory(id + 1),
    ] {
        assert_eq!(next(events), (expected, executor));
    }
}

fn expect_context_device_retired(events: &Receiver<(Operation, std::thread::ThreadId)>) {
    let (operation, executor) = next(events);
    assert_eq!(operation, Operation::Device);
    assert_eq!(next(events), (Operation::Parent, executor));
}

#[test]
fn same_device_contexts_keep_both_pinned_images_after_readers_drain() {
    let buffer = metadata_buffer();
    let (device, events) = device();
    let mut first = metadata_state();
    let mut second = metadata_state();
    assert_ne!(first.context, second.context);
    let first_image = image(device.clone(), 100);
    let second_image = image(device.clone(), 200);
    let first_weak = Arc::downgrade(&first_image);
    let second_weak = Arc::downgrade(&second_image);
    install_context_import(&mut first, &buffer, &first_image);
    install_context_import(&mut second, &buffer, &second_image);
    drop(first_image);
    drop(second_image);

    // Both imports use the exact same device. Repeated alternating readers
    // must still find their own warm image after every reader drops.
    for _ in 0..120 {
        for (state, expected) in [(&mut first, 100), (&mut second, 200)] {
            state.evict_to_capacity();
            assert_eq!(state.retire_sampled(&[buffer.weak()]), 0);
            let reader = state.cache[&buffer.weak()]
                .imported
                .upgrade()
                .expect("pinned import resident");
            assert_eq!(reader.id(), expected);
            drop(reader);
        }
    }
    assert_eq!(first_weak.strong_count(), 1);
    assert_eq!(second_weak.strong_count(), 1);

    // Moving and destroying the first context does not change or revoke the
    // other context's owner. No future frame is needed to retire its image.
    let context = first.context.clone();
    let moved = (first, ()).0;
    assert_eq!(moved.context, context);
    drop(moved);
    expect_context_image_retired(&events, 100);
    assert!(first_weak.upgrade().is_none());
    assert!(second_weak.upgrade().is_some());
    drop(buffer);
    expect_context_image_retired(&events, 200);
    assert!(second_weak.upgrade().is_none());
    drop(second);
    drop(device);
    expect_context_device_retired(&events);
}

#[test]
fn same_device_replacement_and_stale_entry_drop_preserve_other_context() {
    let buffer = metadata_buffer();
    let (device, events) = device();
    let mut first = metadata_state();
    let mut second = metadata_state();
    let old_reader = image(device.clone(), 300);
    let other_reader = image(device.clone(), 400);
    install_context_import(&mut first, &buffer, &old_reader);
    install_context_import(&mut second, &buffer, &other_reader);
    let replacement = image(device.clone(), 500);
    install_context_import(&mut first, &buffer, &replacement);
    drop(replacement);
    assert_eq!(first.cache[&buffer.weak()].imported.upgrade().unwrap().id(), 500);
    assert_eq!(second.cache[&buffer.weak()].imported.upgrade().unwrap().id(), 400);
    assert_eq!(
        old_reader.id(),
        300,
        "submitted readers retain the replaced image"
    );
    drop(old_reader);
    expect_context_image_retired(&events, 300);
    drop(first);
    expect_context_image_retired(&events, 500);
    assert_eq!(second.cache[&buffer.weak()].imported.upgrade().unwrap().id(), 400);

    drop(buffer);
    assert_eq!(
        other_reader.id(),
        400,
        "source retirement does not revoke a reader"
    );
    assert!(events.try_recv().is_err());
    drop(other_reader);
    expect_context_image_retired(&events, 400);
    drop(second);
    drop(device);
    expect_context_device_retired(&events);
}

#[test]
fn source_drop_retires_all_context_imports_without_revoking_submitted_reader() {
    let buffer = metadata_buffer();
    let (device, events) = device();
    let mut first = metadata_state();
    let mut second = metadata_state();
    let reader = image(device.clone(), 600);
    let idle = image(device.clone(), 700);
    install_context_import(&mut first, &buffer, &reader);
    install_context_import(&mut second, &buffer, &idle);
    drop(idle);
    drop(buffer);
    expect_context_image_retired(&events, 700);
    assert_eq!(
        first
            .cache
            .values()
            .next()
            .unwrap()
            .imported
            .upgrade()
            .unwrap()
            .id(),
        600
    );
    assert!(second.cache.values().next().unwrap().imported.upgrade().is_none());
    drop(first);
    drop(second);
    assert!(events.try_recv().is_err());
    drop(reader);
    expect_context_image_retired(&events, 600);
    drop(device);
    expect_context_device_retired(&events);
}

#[test]
#[ignore = "requires a native Vulkan device and DMA-BUF support"]
fn same_origin_real_context_pins_survive_alternating_hot_imports() {
    let (mut first, mut allocator, format) =
        idle_eviction_setup().expect("requires a native Vulkan device and DMA-BUF format");
    let mut second = super::super::test_support::available(
        VulkanRenderer::from_device_origin(&first.device_origin()),
        "second same-origin Vulkan renderer",
    )
    .expect("requires second same-origin Vulkan renderer");
    let (_allocation, buffer) =
        exported_buffer(&mut allocator, format).expect("requires actual DMA-BUF allocation and export");
    let first_id = first.pin_dmabuf_import(&buffer).unwrap().image_resource_id();
    let second_id = second.pin_dmabuf_import(&buffer).unwrap().image_resource_id();
    let first_resource = first.dmabuf.cache[&buffer.weak()].imported.clone();
    let second_resource = second.dmabuf.cache[&buffer.weak()].imported.clone();
    assert_ne!(first.context_id.erased(), second.context_id.erased());
    assert!(first_resource.upgrade().is_some());
    assert!(second_resource.upgrade().is_some());
    first.set_frame_client_import_scope(true);
    second.set_frame_client_import_scope(true);
    for _ in 0..120 {
        assert_eq!(
            first.import_dmabuf_texture(&buffer).unwrap().image_resource_id(),
            first_id
        );
        assert_eq!(
            second.import_dmabuf_texture(&buffer).unwrap().image_resource_id(),
            second_id
        );
    }
    assert_eq!(first.client_first_imports_on_frame(), 0);
    assert_eq!(second.client_first_imports_on_frame(), 0);
    drop(first);
    assert!(first_resource.upgrade().is_none());
    assert!(second_resource.upgrade().is_some());
    assert!(second.unpin_dmabuf_import(&buffer));
    assert_eq!(second.retire_sampled_dmabuf_imports(&[buffer.weak()]), 1);
    assert!(second_resource.upgrade().is_none());
}
