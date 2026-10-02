// Idle eviction of texture imports against a real device. Included into
// `dmabuf::tests`; optional runs skip unavailable hardware. Required runs set
// AVIO_REQUIRE_VK_DEVICE=1 and fail with the missing prerequisite.

/// An instant strictly after every recency stamp taken before this call.
fn instant_after_now() -> std::time::Instant {
    let start = std::time::Instant::now();
    loop {
        let now = std::time::Instant::now();
        if now > start {
            return now;
        }
        std::hint::spin_loop();
    }
}

/// A renderer and an allocator with one format usable as a texture, a render
/// target and a framebuffer-effect target.
fn idle_eviction_setup() -> Option<(VulkanRenderer, VulkanAllocator, crate::backend::allocator::Format)> {
    let (physical_device, renderer) = renderer_and_device()?;
    let format = renderer.dmabuf_import_formats().iter().copied().find(|format| {
        format.modifier != Modifier::Invalid
            && renderer.has_dmabuf_render_format(*format)
            && renderer.has_dmabuf_framebuffer_effect_format(*format)
    });
    let format = super::super::test_support::present(format, "sampled/render/effect DMA-BUF format")?;
    let allocator = VulkanAllocator::new(
        &physical_device,
        ImageUsageFlags::SAMPLED | ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSFER_SRC,
    );
    let allocator = super::super::test_support::available(allocator, "DMA-BUF allocator")?;
    Some((renderer, allocator, format))
}

/// A small buffer and its dma-buf. Callers keep the buffer next to the
/// dma-buf, so the allocation outlives every import made from it.
fn exported_buffer(
    allocator: &mut VulkanAllocator,
    format: crate::backend::allocator::Format,
) -> Option<(crate::backend::allocator::vulkan::VulkanImage, Dmabuf)> {
    let buffer = super::super::test_support::available(
        allocator.create_buffer(64, 64, format.code, &[format.modifier]),
        "DMA-BUF allocation",
    )?;
    let dmabuf = super::super::test_support::available(buffer.export(), "DMA-BUF export")?;
    Some((buffer, dmabuf))
}

#[test]
fn idle_eviction_keeps_imports_referenced_outside_the_cache() {
    let Some((mut renderer, mut allocator, format)) = idle_eviction_setup() else {
        return;
    };
    let (Some((_held_buffer, held)), Some((_idle_buffer, idle))) = (
        exported_buffer(&mut allocator, format),
        exported_buffer(&mut allocator, format),
    ) else {
        return;
    };

    let held_texture = renderer.import_dmabuf_texture(&held).expect("texture import");
    let idle_import = renderer
        .import_dmabuf_texture(&idle)
        .expect("texture import")
        .image_resource_id();
    let cutoff = instant_after_now();

    assert_eq!(renderer.evict_idle_sampled_dmabuf_imports(cutoff, usize::MAX), 1);
    assert!(renderer.dmabuf.cache.contains_key(&held.weak()));
    assert!(!renderer.dmabuf.cache.contains_key(&idle.weak()));
    assert_eq!(renderer.dmabuf.idle_evictions, 1);
    assert_eq!(renderer.diagnostics().dmabuf_cache.evictions, 1);

    // The held texture still shares the cached import; the evicted dma-buf is
    // imported anew, like a first import.
    let held_again = renderer.import_dmabuf_texture(&held).expect("cache hit");
    assert_eq!(held_again.image_resource_id(), held_texture.image_resource_id());
    let reimported = renderer.import_dmabuf_texture(&idle).expect("re-import");
    assert_ne!(reimported.image_resource_id(), idle_import);
}

#[test]
fn idle_eviction_never_drops_target_imports() {
    let Some((mut renderer, mut allocator, format)) = idle_eviction_setup() else {
        return;
    };
    let (Some((_effect_buffer, effect_target)), Some((_upgraded_buffer, upgraded))) = (
        exported_buffer(&mut allocator, format),
        exported_buffer(&mut allocator, format),
    ) else {
        return;
    };

    // Each returned handle is dropped at once, so only the cache holds the
    // imports: a framebuffer-effect target, and a texture later bound as a
    // render target.
    renderer
        .bind_dmabuf_framebuffer_effect_target(&effect_target)
        .expect("framebuffer-effect bind");
    renderer.import_dmabuf_texture(&upgraded).expect("texture import");
    renderer
        .bind_dmabuf_target(&upgraded)
        .expect("render target bind");
    let cutoff = instant_after_now();

    assert_eq!(renderer.evict_idle_sampled_dmabuf_imports(cutoff, usize::MAX), 0);
    assert!(renderer.dmabuf.cache.contains_key(&effect_target.weak()));
    assert!(renderer.dmabuf.cache.contains_key(&upgraded.weak()));
    assert_eq!(renderer.dmabuf.idle_evictions, 0);
}

#[test]
fn idle_eviction_takes_at_most_max_oldest_first() {
    let Some((mut renderer, mut allocator, format)) = idle_eviction_setup() else {
        return;
    };
    let (Some((_oldest_buffer, oldest)), Some((_middle_buffer, middle)), Some((_newest_buffer, newest))) = (
        exported_buffer(&mut allocator, format),
        exported_buffer(&mut allocator, format),
        exported_buffer(&mut allocator, format),
    ) else {
        return;
    };

    let before_imports = std::time::Instant::now();
    for dmabuf in [&oldest, &middle, &newest] {
        renderer.import_dmabuf_texture(dmabuf).expect("texture import");
    }
    assert_eq!(
        renderer.evict_idle_sampled_dmabuf_imports(before_imports, usize::MAX),
        0,
        "nothing was used before the imports"
    );

    let cutoff = instant_after_now();
    assert_eq!(renderer.evict_idle_sampled_dmabuf_imports(cutoff, 0), 0);
    assert_eq!(renderer.evict_idle_sampled_dmabuf_imports(cutoff, 2), 2);
    assert!(!renderer.dmabuf.cache.contains_key(&oldest.weak()));
    assert!(!renderer.dmabuf.cache.contains_key(&middle.weak()));
    assert!(renderer.dmabuf.cache.contains_key(&newest.weak()));
    assert_eq!(renderer.dmabuf.idle_evictions, 2);
}

#[test]
fn idle_eviction_keeps_an_import_hit_since_the_cutoff() {
    let Some((mut renderer, mut allocator, format)) = idle_eviction_setup() else {
        return;
    };
    let (Some((_first_buffer, first)), Some((_second_buffer, second))) = (
        exported_buffer(&mut allocator, format),
        exported_buffer(&mut allocator, format),
    ) else {
        return;
    };

    renderer.import_dmabuf_texture(&first).expect("texture import");
    renderer.import_dmabuf_texture(&second).expect("texture import");
    let cutoff = instant_after_now();

    // A hit after the cutoff marks `first` used and moves it behind `second`.
    let hits = renderer.diagnostics().dmabuf_cache.hits;
    renderer.import_dmabuf_texture(&first).expect("cache hit");
    assert_eq!(renderer.diagnostics().dmabuf_cache.hits, hits + 1);
    assert_eq!(renderer.dmabuf.cache.get_index_of(&first.weak()), Some(1));

    assert_eq!(renderer.evict_idle_sampled_dmabuf_imports(cutoff, usize::MAX), 1);
    assert!(renderer.dmabuf.cache.contains_key(&first.weak()));
    assert!(!renderer.dmabuf.cache.contains_key(&second.weak()));
}

#[test]
fn last_buffer_drop_retires_cached_vulkan_import_without_cleanup() {
    let Some((mut renderer, mut allocator, format)) = idle_eviction_setup() else {
        return;
    };
    let Some((buffer, dmabuf)) = exported_buffer(&mut allocator, format) else {
        return;
    };
    let texture = renderer.import_dmabuf_texture(&dmabuf).expect("texture import");
    let weak = std::sync::Arc::downgrade(texture.image_resource().unwrap());
    let key = dmabuf.weak();
    drop(texture);
    assert!(weak.upgrade().is_some(), "a live buffer permits cache reuse");
    drop((dmabuf, buffer));
    assert!(
        renderer.dmabuf.cache.contains_key(&key),
        "metadata cleanup has not run"
    );
    assert!(
        weak.upgrade().is_none(),
        "stale metadata must not retain GPU memory"
    );
}
