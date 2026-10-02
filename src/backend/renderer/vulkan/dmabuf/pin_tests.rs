// Included into dmabuf::tests: pure metadata checks run without a device;
// image/usage checks require AVIO_REQUIRE_VK_DEVICE=1 for acceptance.
use crate::backend::allocator::Buffer as _;

fn metadata_buffer() -> Dmabuf {
    let fd = rustix::fs::memfd_create("import-pin-test", rustix::fs::MemfdFlags::CLOEXEC).unwrap();
    rustix::fs::ftruncate(&fd, 4096).unwrap();
    let mut builder = Dmabuf::builder(
        (1, 1),
        Fourcc::Argb8888,
        Modifier::Linear,
        crate::backend::allocator::dmabuf::DmabufFlags::empty(),
    );
    assert!(builder.add_plane(fd.into(), 0, 0, 4));
    builder.build().unwrap()
}

fn metadata_entry(buffer: &Dmabuf, pins: usize) -> super::CachedDmabuf {
    super::CachedDmabuf {
        handle: buffer.weak(),
        signature: super::DmabufSignature {
            size: buffer.size(),
            format: buffer.format(),
            num_planes: 1,
            offsets: vec![0],
            strides: vec![4],
            disjoint: false,
            y_inverted: false,
        },
        imported: std::sync::Weak::new(),
        custody: std::sync::Weak::new(),
        device: 0,
        pins,
        last_used: std::time::Instant::now(),
    }
}

#[test]
fn capacity_eviction_skips_pinned_metadata_even_above_the_capacity() {
    let buffers = (0..super::MAX_DMABUF_CACHE_ENTRIES + 2)
        .map(|_| metadata_buffer())
        .collect::<Vec<_>>();
    let mut state = DmabufState::default();
    for buffer in &buffers {
        state.cache.insert(buffer.weak(), metadata_entry(buffer, 1));
    }
    state.evict_to_capacity();
    assert_eq!(
        state.cache.len(),
        buffers.len(),
        "the working set determines pinned membership"
    );
    assert!(state.unpin(&buffers[3]));
    state.evict_to_capacity();
    assert!(!state.cache.contains_key(&buffers[3].weak()));
    assert_eq!(state.cache.len(), buffers.len() - 1);
}

#[test]
fn pins_balance_exact_source_identity_without_underflow() {
    let buffer = metadata_buffer();
    let other = metadata_buffer();
    let mut state = DmabufState::default();
    state.cache.insert(buffer.weak(), metadata_entry(&buffer, 2));
    assert!(!state.unpin(&other));
    assert!(state.unpin_weak(&buffer.weak()));
    assert_eq!(state.cache[&buffer.weak()].pins, 1);
    assert!(state.unpin(&buffer));
    assert!(!state.unpin(&buffer));
    assert_eq!(state.cache[&buffer.weak()].pins, 0);
}

#[test]
fn membership_pin_survives_usage_upgrade_and_all_retirement_paths() {
    let Some((mut renderer, mut allocator, format)) = idle_eviction_setup() else {
        return;
    };
    let Some((_buffer, dmabuf)) = exported_buffer(&mut allocator, format) else {
        return;
    };
    renderer.pin_dmabuf_import(&dmabuf).unwrap();
    renderer.pin_dmabuf_import(&dmabuf).unwrap();
    renderer.bind_dmabuf_target(&dmabuf).unwrap();
    assert_eq!(renderer.dmabuf.cache[&dmabuf.weak()].pins, 2);
    assert_eq!(
        renderer.evict_idle_sampled_dmabuf_imports(instant_after_now(), usize::MAX),
        0
    );
    assert_eq!(renderer.retire_sampled_dmabuf_imports(&[dmabuf.weak()]), 0);
    assert!(renderer.unpin_dmabuf_import(&dmabuf));
    assert_eq!(renderer.dmabuf.cache[&dmabuf.weak()].pins, 1);
    assert!(renderer.unpin_dmabuf_import(&dmabuf));
    assert!(!renderer.unpin_dmabuf_import(&dmabuf));
    assert_eq!(
        renderer.retire_sampled_dmabuf_imports(&[dmabuf.weak()]),
        0,
        "authored targets preserve contents"
    );
}

#[test]
fn explicit_retirement_releases_owner_while_a_reader_keeps_the_old_image() {
    let Some((mut renderer, mut allocator, format)) = idle_eviction_setup() else {
        return;
    };
    let Some((_buffer, dmabuf)) = exported_buffer(&mut allocator, format) else {
        return;
    };
    let reader = renderer.pin_dmabuf_import(&dmabuf).unwrap();
    let old = reader.image_resource_id();
    assert_eq!(renderer.retire_sampled_dmabuf_imports(&[dmabuf.weak()]), 0);
    assert!(renderer.unpin_dmabuf_import(&dmabuf));
    assert_eq!(renderer.retire_sampled_dmabuf_imports(&[dmabuf.weak()]), 1);
    assert!(!renderer.dmabuf.cache.contains_key(&dmabuf.weak()));
    assert_eq!(
        reader.image_resource_id(),
        old,
        "reader custody is independent of residency"
    );
    assert_ne!(
        renderer
            .import_dmabuf_texture(&dmabuf)
            .unwrap()
            .image_resource_id(),
        old
    );
}

#[test]
fn client_first_import_counter_uses_successful_frame_creations_only() {
    let Some((mut renderer, mut allocator, format)) = idle_eviction_setup() else {
        return;
    };
    let (Some((_first_buffer, first)), Some((_second_buffer, second))) = (
        exported_buffer(&mut allocator, format),
        exported_buffer(&mut allocator, format),
    ) else {
        return;
    };
    renderer.prepare_frame_client_sources(&[first.weak(), second.weak()]);
    renderer.pin_dmabuf_import(&first).unwrap();
    assert_eq!(
        renderer.client_first_imports_on_frame(),
        0,
        "commit acceptance is off-frame"
    );
    renderer.set_frame_client_import_scope(true);
    renderer.import_dmabuf_texture(&first).unwrap();
    assert_eq!(
        renderer.client_first_imports_on_frame(),
        0,
        "cache hit is not a first import"
    );
    renderer.import_dmabuf_texture(&second).unwrap();
    assert_eq!(renderer.client_first_imports_on_frame(), 1);
    renderer.bind_dmabuf_target(&second).unwrap();
    assert_eq!(
        renderer.client_first_imports_on_frame(),
        1,
        "usage upgrade is not first import"
    );
    renderer.set_frame_client_import_scope(false);
}
