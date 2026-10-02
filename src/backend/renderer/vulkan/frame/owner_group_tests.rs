/// Exercise the real same-CB lane renderer and one final RGBA resolve. This
/// is authored pixel evidence; it must be explicitly run on a Vulkan device.
#[test]
#[ignore = "requires Vulkan; never count an unavailable device as pixel evidence"]
fn owner_group_resolves_overlapping_paint_and_opacity_once() {
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
        for background_alpha in [0.0, 1.0] {
            for opacity in [0.0, 0.5, 1.0] {
                let output = Size::<i32, Physical>::from((4, 4));
                let full = Rectangle::from_size(output);
                let mut accumulator = renderer
                    .create_buffer(Fourcc::Abgr8888, (4, 4).into())
                    .expect("parent");
                let atlas = renderer
                    .create_buffer(Fourcc::Abgr8888, (10, 12).into())
                    .expect("bounded reusable lanes");
                let prefix = renderer
                    .create_buffer(Fourcc::Abgr8888, (5, 6).into())
                    .expect("bounded root opacity prefix");
                let clip = crate::backend::renderer::RoundedClip {
                    rect: full.to_f64(),
                    radius: 2.0,
                    exponent: 2.0,
                    aa_width: 1.0,
                    corner_mask: 15,
                };
                let mut target = renderer.bind(&mut accumulator).expect("bind");
                let mut frame = renderer.render(&mut target, output, transform).expect("frame");
                frame
                    .clear(
                        Color32F::new(0.0, 0.0, 0.4 * background_alpha, background_alpha),
                        &[full],
                    )
                    .unwrap();
                frame
                    .render_owner_clipped_group(&atlas, full, full, clip, &[full], |frame| {
                        frame.capture_and_filter_framebuffer(full, &prefix, &[])?;
                        frame.draw_solid(full, &[full], Color32F::new(1.0, 0.0, 0.0, 0.5))?;
                        frame.draw_solid(full, &[full], Color32F::new(0.0, 1.0, 0.0, 0.5))?;
                        frame.interpolate_framebuffer_prefix(&prefix, full, &[full], opacity)
                    })
                    .expect("all ordered sample lanes");
                frame.finish().expect("submit").wait().expect("completed");
                drop(target);
                let mapping = renderer
                    .copy_texture(
                        &accumulator,
                        Rectangle::from_size((4, 4).into()),
                        Fourcc::Abgr8888,
                    )
                    .expect("readback");
                let bytes = renderer.map_texture(&mapping).expect("map");
                for (location, coverage) in [((0, 0), 0.5), ((1, 1), 1.0)] {
                    let pixel = transform
                        .transform_rect_in(Rectangle::new(location.into(), (1, 1).into()), &output)
                        .loc;
                    let index = ((pixel.y * 4 + pixel.x) * 4) as usize;
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
                    let lower = [0.0, 0.0, decode(0.4 * background_alpha), background_alpha];
                    let group = [0.25, 0.5, lower[2] * 0.25, 0.75 + background_alpha * 0.25];
                    for channel in 0..4 {
                        let weight = coverage * opacity;
                        let mixed = lower[channel] * (1.0 - weight) + group[channel] * weight;
                        let expected =
                            ((if channel == 3 { mixed } else { encode(mixed) }) * 255.0).round() as i32;
                        assert!((bytes[index+channel] as i32-expected).abs()<=3,
                            "{transform:?}, background={background_alpha}, opacity={opacity}, location={location:?}, channel={channel}: got {}, expected {expected}",bytes[index+channel]);
                    }
                }
            }
        }
    }
}
