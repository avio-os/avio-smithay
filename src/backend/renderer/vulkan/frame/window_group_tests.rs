/// Exercise the actual Vulkan constant-factor pipeline, not a CPU imitation.
/// Transparent captures are deliberately included: source-over cannot implement
/// prefix interpolation when the saved lower framebuffer has alpha below one.
#[test]
#[ignore = "requires Vulkan; run explicitly to prevent silently skipped pixel evidence"]
fn framebuffer_group_prefix_interpolates_premultiplied_rgba() {
    let instance = Instance::new(Version::VERSION_1_3, None).expect("Vulkan instance");
    let physical = PhysicalDevice::enumerate(&instance)
        .expect("devices")
        .find(|device| device.render_node().ok().flatten().is_some())
        .expect("hardware render node");
    let mut renderer = VulkanRenderer::new(&physical).expect("renderer");
    for transform in [
        Transform::Normal,
        Transform::_90,
        Transform::_180,
        Transform::_270,
        Transform::Flipped,
        Transform::Flipped90,
        Transform::Flipped180,
        Transform::Flipped270,
    ] {
        for background_alpha in [0.0_f32, 0.35, 1.0] {
            for opacity in [0.0_f32, 0.25, 0.7, 1.0] {
                let output = Size::<i32, Physical>::from((12, 8));
                let storage = transform.transform_size(output);
                let size = Size::from((storage.w, storage.h));
                let mut accumulator = renderer.create_buffer(Fourcc::Abgr8888, size).expect("target");
                let prefix = renderer.create_buffer(Fourcc::Abgr8888, size).expect("prefix");
                let full = Rectangle::from_size(output);
                let mut target = renderer.bind(&mut accumulator).expect("bind");
                let mut frame = renderer.render(&mut target, output, transform).expect("frame");
                frame
                    .clear(
                        Color32F::new(0.0, 0.0, background_alpha, background_alpha),
                        &[full],
                    )
                    .unwrap();
                frame.capture_and_filter_framebuffer(full, &prefix, &[]).unwrap();
                // Two overlapping members must receive group opacity once.
                frame
                    .draw_solid(full, &[full], Color32F::new(0.5, 0.0, 0.0, 0.5))
                    .unwrap();
                let upper = Rectangle::new((4, 0).into(), (8, 8).into());
                frame
                    .draw_solid(
                        upper,
                        &[Rectangle::from_size(upper.size)],
                        Color32F::new(0.0, 0.5, 0.0, 0.5),
                    )
                    .unwrap();
                frame
                    .interpolate_framebuffer_prefix(&prefix, full, &[full], opacity)
                    .unwrap();
                frame.finish().expect("submit").wait().expect("completed");
                drop(target);
                let mapping = renderer
                    .copy_texture(&accumulator, Rectangle::from_size(size), Fourcc::Abgr8888)
                    .expect("readback");
                let bytes = renderer.map_texture(&mapping).expect("map");
                for (location, covered_twice) in [((2, 3), false), ((7, 3), true)] {
                    let pixel = transform
                        .transform_rect_in(Rectangle::new(location.into(), (1, 1).into()), &output)
                        .loc;
                    let index = ((pixel.y * storage.w + pixel.x) * 4) as usize;
                    // Color32F solids are straight electrical colors; the
                    // attachment blends premultiplied linear light and stores
                    // sRGB. Alpha is linear throughout.
                    let decode = |c: f32| {
                        if c <= 0.04045 {
                            c / 12.92
                        } else {
                            ((c + 0.055) / 1.055).powf(2.4)
                        }
                    };
                    let encode = |c: f32| {
                        if c <= 0.0031308 {
                            c * 12.92
                        } else {
                            1.055 * c.powf(1.0 / 2.4) - 0.055
                        }
                    };
                    let half = decode(0.5) * 0.5;
                    let blue = decode(background_alpha);
                    let completed = if covered_twice {
                        [half * 0.5, half, blue * 0.25, 0.75 + background_alpha * 0.25]
                    } else {
                        [half, 0.0, blue * 0.5, 0.5 + background_alpha * 0.5]
                    };
                    let lower = [0.0, 0.0, blue, background_alpha];
                    for channel in 0..4 {
                        let mixed = lower[channel] * (1.0 - opacity) + completed[channel] * opacity;
                        let expected =
                            ((if channel == 3 { mixed } else { encode(mixed) }) * 255.0).round() as i32;
                        assert!((bytes[index+channel] as i32 - expected).abs() <= 3,
                            "{transform:?}, background={background_alpha}, opacity={opacity}, overlap={covered_twice}, channel={channel}: got {}, expected {expected}", bytes[index+channel]);
                    }
                }
            }
        }
    }
}
