#[test]
fn sampled_preparation_requires_a_live_import_in_this_exact_context() {
    let buffer = metadata_buffer();
    let other = metadata_buffer();
    let mut state = metadata_state();
    assert!(!state.sampled_prepared(&buffer));
    state
        .cache
        .insert(buffer.weak(), metadata_entry(&buffer, 1, state.context.clone()));
    assert!(
        !state.sampled_prepared(&buffer),
        "a pin and immutable descriptor are not an image"
    );

    let (device, events) = device();
    let imported = image(device.clone(), 1000);
    install_context_import(&mut state, &buffer, &imported);
    assert!(state.sampled_prepared(&buffer));
    assert!(
        !state.sampled_prepared(&other),
        "equal geometry is not the exact backing"
    );
    let before = (
        state.cache_stats,
        state.import_attempts_total,
        state.cache[&buffer.weak()].last_used,
    );
    for _ in 0..120 {
        assert!(state.sampled_prepared(&buffer));
    }
    assert_eq!(state.cache_stats.hits, before.0.hits);
    assert_eq!(state.cache_stats.misses, before.0.misses);
    assert_eq!(state.import_attempts_total, before.1);
    assert_eq!(state.cache[&buffer.weak()].last_used, before.2);
    let original_context = state.cache[&buffer.weak()].context.clone();
    state.cache.get_mut(&buffer.weak()).unwrap().context = metadata_state().context;
    assert!(
        !state.sampled_prepared(&buffer),
        "shared device does not grant another context's import"
    );
    state.cache.get_mut(&buffer.weak()).unwrap().context = original_context;
    state.cache.get_mut(&buffer.weak()).unwrap().signature.size = (2, 1).into();
    assert!(
        !state.sampled_prepared(&buffer),
        "a live view for another extent is not admitted"
    );
    state.cache.get_mut(&buffer.weak()).unwrap().signature.size = (1, 1).into();
    state.cache.get_mut(&buffer.weak()).unwrap().signature.format.code = Fourcc::Abgr8888;
    assert!(
        !state.sampled_prepared(&buffer),
        "a live view for another format is not admitted"
    );
    state.cache.get_mut(&buffer.weak()).unwrap().signature.format.code = Fourcc::Argb8888;
    drop(imported);
    assert!(
        state.sampled_prepared(&buffer),
        "actual source custody preserves its image"
    );
    let custody = buffer.resource_custody::<super::ImportCustody<VulkanImage>>();
    let weak = state.cache[&buffer.weak()].imported.clone();
    custody.remove(&state.context, &weak);
    assert!(
        !state.sampled_prepared(&buffer),
        "stale metadata cannot establish readiness"
    );
    expect_context_image_retired(&events, 1000);
    drop(state);
    drop(buffer);
    drop(other);
    drop(custody);
    drop(device);
    expect_context_device_retired(&events);
}

#[test]
#[ignore = "requires a real Vulkan dma-buf export/import device"]
fn three_slot_sampled_rotation_stays_prepared_without_new_imports() {
    let Some((mut renderer, mut allocator, format)) = idle_eviction_setup() else {
        return;
    };
    let buffers: Vec<_> = (0..3)
        .map(|_| exported_buffer(&mut allocator, format).unwrap())
        .collect();
    for (_, dmabuf) in &buffers {
        assert!(!renderer.sampled_dmabuf_is_prepared(dmabuf));
        renderer.bind_dmabuf_target(dmabuf).unwrap();
        assert!(
            !renderer.sampled_dmabuf_is_prepared(dmabuf),
            "an attachment-only image/view does not admit sampled usage"
        );
        renderer.import_dmabuf_texture(dmabuf).unwrap();
        assert!(renderer.sampled_dmabuf_is_prepared(dmabuf));
    }
    let misses = renderer.dmabuf.cache_stats.misses;
    for index in 0..120 {
        let dmabuf = &buffers[index % 3].1;
        assert!(renderer.sampled_dmabuf_is_prepared(dmabuf));
        renderer.import_dmabuf_texture(dmabuf).unwrap();
    }
    assert_eq!(renderer.dmabuf.cache_stats.misses, misses);
}
