// Encoded-sRGB blend-space evidence through the production frame path: every
// authored value (token alpha, draw opacity, solid colour, material filter and
// tint) must store the CSS SourceOver result on 8-bit and 2101010 targets alike.

/// One renderer plus the small upload ring its memory sources need.
struct BlendProbe {
    renderer: VulkanRenderer,
}

/// A source drawn over a solid backdrop, in the three shapes the compositor
/// feeds the blend: a premultiplied client texel, an opaque texel faded by
/// draw opacity, and a straight solid colour.
#[derive(Debug, Clone, Copy)]
enum SourceShape {
    PremultipliedTexel,
    FadedOpaqueTexel,
    Solid,
}

const PROBE_EXTENT: i32 = 8;

impl BlendProbe {
    fn new() -> Option<Self> {
        let physical = crate::backend::renderer::vulkan::test_support::physical_device()?;
        let mut renderer = crate::backend::renderer::vulkan::test_support::renderer(&physical)?;
        assert!(renderer
            .configure_memory_upload_capacity(64 * 1024)
            .expect("provision probe upload ring"));
        Some(Self { renderer })
    }

    /// Target formats whose offscreens this device supports. 8-bit storage is
    /// required; 2101010 is an optional device capability.
    fn target_formats(&mut self) -> Vec<Fourcc> {
        let mut formats = Vec::new();
        for format in [
            Fourcc::Abgr8888,
            Fourcc::Argb8888,
            Fourcc::Argb2101010,
            Fourcc::Abgr2101010,
        ] {
            if self.renderer.create_buffer(format, (1, 1).into()).is_ok() {
                formats.push(format);
            } else if is_ten_bit(format) {
                super::super::test_support::capability_unavailable(format_args!(
                    "{format:?} offscreen targets"
                ));
            }
        }
        assert!(
            formats.iter().any(|format| !is_ten_bit(*format)),
            "no 8-bit offscreen target format"
        );
        formats
    }

    fn import_abgr(&mut self, size: Size<i32, BufferCoord>, bytes: &[u8]) -> VulkanTexture {
        use crate::backend::renderer::ImportMem;
        self.renderer
            .import_memory(bytes, Fourcc::Abgr8888, size, false)
            .expect("memory source")
    }

    /// Clears a `format` target to `backdrop`, lets `draw` record into it and
    /// returns every stored pixel, normalized to straight channel values.
    fn render(
        &mut self,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
        backdrop: [f32; 4],
        draw: impl FnOnce(&mut VulkanFrame<'_>, Rectangle<i32, Physical>),
    ) -> Vec<[f64; 4]> {
        let mut target_texture = self.renderer.create_buffer(format, size).expect("target");
        let physical = Size::<i32, Physical>::from((size.w, size.h));
        let full = Rectangle::from_size(physical);
        {
            let mut target = self.renderer.bind(&mut target_texture).expect("bind");
            let mut frame = self
                .renderer
                .render(&mut target, physical, Transform::Normal)
                .expect("frame");
            frame
                .clear(
                    Color32F::new(backdrop[0], backdrop[1], backdrop[2], backdrop[3]),
                    &[full],
                )
                .expect("clear");
            draw(&mut frame, full);
            frame.finish().expect("submit").wait().expect("complete");
        }
        self.read(&target_texture, format)
    }

    fn read(&mut self, texture: &VulkanTexture, format: Fourcc) -> Vec<[f64; 4]> {
        let mapping = self
            .renderer
            .copy_texture(texture, Rectangle::from_size(texture.size()), format)
            .expect("readback");
        let bytes = self.renderer.map_texture(&mapping).expect("map");
        bytes
            .chunks_exact(4)
            .map(|pixel| stored_rgba(format, pixel))
            .collect()
    }

    /// Composites one ink at `alpha` over an opaque backdrop through `shape`.
    fn composite(
        &mut self,
        format: Fourcc,
        shape: SourceShape,
        ink: [u8; 3],
        alpha: f32,
        backdrop: [u8; 3],
    ) -> Vec<[f64; 4]> {
        let size = Size::<i32, BufferCoord>::from((PROBE_EXTENT, PROBE_EXTENT));
        let texel = match shape {
            SourceShape::PremultipliedTexel => {
                let premultiply = |channel: u8| (f32::from(channel) * alpha).round() as u8;
                [
                    premultiply(ink[0]),
                    premultiply(ink[1]),
                    premultiply(ink[2]),
                    (alpha * 255.0).round() as u8,
                ]
            }
            SourceShape::FadedOpaqueTexel | SourceShape::Solid => [ink[0], ink[1], ink[2], 255],
        };
        let source = self.import_abgr(size, &texel.repeat((size.w * size.h) as usize));
        let encoded = |channel: u8| f32::from(channel) / 255.0;
        self.render(
            format,
            size,
            [
                encoded(backdrop[0]),
                encoded(backdrop[1]),
                encoded(backdrop[2]),
                1.0,
            ],
            |frame, full| match shape {
                SourceShape::PremultipliedTexel | SourceShape::FadedOpaqueTexel => frame
                    .render_texture_from_to(
                        &source,
                        Rectangle::from_size((f64::from(size.w), f64::from(size.h)).into()),
                        full,
                        &[full],
                        &[],
                        Transform::Normal,
                        if matches!(shape, SourceShape::FadedOpaqueTexel) {
                            alpha
                        } else {
                            1.0
                        },
                    )
                    .expect("texture draw"),
                SourceShape::Solid => frame
                    .draw_solid(
                        full,
                        &[full],
                        Color32F::new(encoded(ink[0]), encoded(ink[1]), encoded(ink[2]), alpha),
                    )
                    .expect("solid draw"),
            },
        )
    }
}

fn is_ten_bit(format: Fourcc) -> bool {
    matches!(
        format,
        Fourcc::Argb2101010 | Fourcc::Xrgb2101010 | Fourcc::Abgr2101010 | Fourcc::Xbgr2101010
    )
}

/// Straight `[r, g, b, a]` in 0..=255 units, undoing each fourcc's packing.
fn stored_rgba(format: Fourcc, pixel: &[u8]) -> [f64; 4] {
    let byte = |index: usize| f64::from(pixel[index]);
    let packed = u32::from_le_bytes([pixel[0], pixel[1], pixel[2], pixel[3]]);
    let ten = |shift: u32| f64::from((packed >> shift) & 0x3ff) * 255.0 / 1023.0;
    let two = f64::from(packed >> 30) * 255.0 / 3.0;
    match format {
        Fourcc::Abgr8888 | Fourcc::Xbgr8888 => [byte(0), byte(1), byte(2), byte(3)],
        Fourcc::Argb8888 | Fourcc::Xrgb8888 => [byte(2), byte(1), byte(0), byte(3)],
        Fourcc::Argb2101010 | Fourcc::Xrgb2101010 => [ten(20), ten(10), ten(0), two],
        Fourcc::Abgr2101010 | Fourcc::Xbgr2101010 => [ten(0), ten(10), ten(20), two],
        other => panic!("unexpected probe format {other:?}"),
    }
}

/// CSS SourceOver of a straight `ink` at `alpha` over an opaque backdrop, in
/// encoded sRGB: what Chromium, GTK and the Avio prototype store.
fn css_source_over(ink: [u8; 3], alpha: f64, backdrop: [u8; 3]) -> [f64; 3] {
    std::array::from_fn(|channel| {
        f64::from(ink[channel]) * alpha + f64::from(backdrop[channel]) * (1.0 - alpha)
    })
}

#[track_caller]
fn assert_rgb_near(pixels: &[[f64; 4]], expected: [f64; 3], tolerance: f64, what: &str) {
    for (index, pixel) in pixels.iter().enumerate() {
        for channel in 0..3 {
            assert!(
                (pixel[channel] - expected[channel]).abs() <= tolerance,
                "{what}: pixel {index} channel {channel} stored {:.2}, CSS {:.2} +-{tolerance}",
                pixel[channel],
                expected[channel]
            );
        }
    }
}

/// D1 acceptance: Avio ink tokens over paper and night backdrops store the
/// CSS SourceOver bytes within +-2, whatever shape carries the alpha and
/// whatever the target's storage depth.
#[test]
fn css_ink_tokens_store_css_source_over_bytes_on_every_target_format() {
    let Some(mut probe) = BlendProbe::new() else {
        return;
    };
    const PAPER: [u8; 3] = [254, 254, 253];
    const NIGHT: [u8; 3] = [27, 52, 63];
    let cases: [([u8; 3], f32, [u8; 3]); 5] = [
        ([0x0A, 0x0D, 0x13], 0.78, PAPER),
        ([0x0A, 0x0D, 0x13], 0.64, PAPER),
        ([0x00, 0x00, 0x00], 0.86, PAPER),
        ([0xEE, 0xF2, 0xF7], 0.86, NIGHT),
        ([0xEE, 0xF2, 0xF7], 0.76, NIGHT),
    ];
    // Black at .86 over paper is the notification-title control: CSS stores
    // 36, linear-light blending stored 104.
    assert_eq!(css_source_over([0, 0, 0], 0.86, PAPER)[0].round(), 36.0);
    for format in probe.target_formats() {
        for (ink, alpha, backdrop) in cases {
            let expected = css_source_over(ink, f64::from(alpha), backdrop);
            for shape in [
                SourceShape::PremultipliedTexel,
                SourceShape::FadedOpaqueTexel,
                SourceShape::Solid,
            ] {
                let pixels = probe.composite(format, shape, ink, alpha, backdrop);
                assert_rgb_near(
                    &pixels,
                    expected,
                    2.0,
                    &format!("{format:?} {shape:?} {ink:02X?}@{alpha} over {backdrop:?}"),
                );
            }
        }
    }
}

/// The 256 byte values of one channel, with the other two channels permuted so
/// every pixel differs, fully opaque.
fn every_byte_value() -> Vec<u8> {
    (0..=255u8)
        .flat_map(|value| [value, 255 - value, value.wrapping_mul(37), 255])
        .collect()
}

/// An opaque source composited 1:1 stores exactly its bytes, directly and
/// after a hop through a compositor offscreen, so composited and directly
/// scanned-out buffers cannot drift apart.
#[test]
fn opaque_one_to_one_composite_stores_its_source_bytes() {
    let Some(mut probe) = BlendProbe::new() else {
        return;
    };
    let size = Size::<i32, BufferCoord>::from((16, 16));
    let source_bytes = every_byte_value();
    let source = probe.import_abgr(size, &source_bytes);
    let expected = source_bytes
        .chunks_exact(4)
        .map(|pixel| stored_rgba(Fourcc::Abgr8888, pixel))
        .collect::<Vec<_>>();
    let src = Rectangle::from_size((16.0, 16.0).into());
    for format in [Fourcc::Abgr8888, Fourcc::Argb8888] {
        let direct = probe.render(format, size, [0.3, 0.6, 0.9, 1.0], |frame, full| {
            frame
                .render_texture_from_to(&source, src, full, &[full], &[], Transform::Normal, 1.0)
                .expect("opaque draw");
        });
        assert_eq!(
            direct, expected,
            "{format:?}: opaque 1:1 composite must store the source bytes"
        );
    }

    let mut hop = probe
        .renderer
        .create_buffer(Fourcc::Abgr8888, size)
        .expect("offscreen");
    {
        let physical = Size::<i32, Physical>::from((16, 16));
        let full = Rectangle::from_size(physical);
        let mut target = probe.renderer.bind(&mut hop).expect("bind hop");
        let mut frame = probe
            .renderer
            .render(&mut target, physical, Transform::Normal)
            .unwrap();
        frame.clear(Color32F::new(0.0, 0.0, 0.0, 0.0), &[full]).unwrap();
        frame
            .render_texture_from_to(&source, src, full, &[full], &[], Transform::Normal, 1.0)
            .unwrap();
        frame.finish().unwrap().wait().unwrap();
    }
    let hopped = probe.render(Fourcc::Argb8888, size, [0.9, 0.2, 0.1, 1.0], |frame, full| {
        frame
            .render_texture_from_to(&hop, src, full, &[full], &[], Transform::Normal, 1.0)
            .expect("offscreen draw");
    });
    assert_eq!(
        hopped, expected,
        "compositor offscreens sample their stored bytes plainly"
    );
}

/// A translucent source rendered into a cleared compositor offscreen keeps
/// its encoded premultiplied value, so drawing that offscreen over a backdrop
/// equals drawing the source there directly (SourceOver is associative).
#[test]
fn translucent_offscreen_hop_equals_direct_composition() {
    let Some(mut probe) = BlendProbe::new() else {
        return;
    };
    let size = Size::<i32, BufferCoord>::from((PROBE_EXTENT, PROBE_EXTENT));
    let src = Rectangle::from_size((f64::from(PROBE_EXTENT), f64::from(PROBE_EXTENT)).into());
    let texel = [6u8, 8, 12, 163];
    let source = probe.import_abgr(size, &texel.repeat(64));
    let mut hop = probe
        .renderer
        .create_buffer(Fourcc::Abgr8888, size)
        .expect("offscreen");
    {
        let physical = Size::<i32, Physical>::from((PROBE_EXTENT, PROBE_EXTENT));
        let full = Rectangle::from_size(physical);
        let mut target = probe.renderer.bind(&mut hop).expect("bind hop");
        let mut frame = probe
            .renderer
            .render(&mut target, physical, Transform::Normal)
            .unwrap();
        frame.clear(Color32F::new(0.0, 0.0, 0.0, 0.0), &[full]).unwrap();
        frame
            .render_texture_from_to(&source, src, full, &[full], &[], Transform::Normal, 1.0)
            .unwrap();
        frame.finish().unwrap().wait().unwrap();
    }
    let stored = probe.read(&hop, Fourcc::Abgr8888);
    let texel_rgba = stored_rgba(Fourcc::Abgr8888, &texel);
    assert!(
        stored.iter().all(|pixel| *pixel == texel_rgba),
        "offscreen stores the texel unconverted"
    );

    let backdrop = [0.996, 0.996, 0.992, 1.0];
    let direct = probe.render(Fourcc::Abgr8888, size, backdrop, |frame, full| {
        frame
            .render_texture_from_to(&source, src, full, &[full], &[], Transform::Normal, 1.0)
            .unwrap();
    });
    let hopped = probe.render(Fourcc::Abgr8888, size, backdrop, |frame, full| {
        frame
            .render_texture_from_to(&hop, src, full, &[full], &[], Transform::Normal, 1.0)
            .unwrap();
    });
    assert_eq!(direct, hopped);
}

/// CSS `contrast() brightness() saturate()` of an encoded colour, each clamped.
fn css_filter(rgb: [u8; 3], contrast: f64, brightness: f64, saturation: f64) -> [f64; 3] {
    let channels = rgb.map(|channel| {
        let contrasted = ((f64::from(channel) / 255.0 - 0.5) * contrast + 0.5).clamp(0.0, 1.0);
        (contrasted * brightness).clamp(0.0, 1.0)
    });
    let luma = 0.213 * channels[0] + 0.715 * channels[1] + 0.072 * channels[2];
    channels.map(|channel| (luma + (channel - luma) * saturation).clamp(0.0, 1.0) * 255.0)
}

/// The material bridge: the filtered backdrop, captured in-frame and returned
/// to the accumulator through the production capture, pyramid and fused
/// final-pass paths, lands on CSS `backdrop-filter` of that backdrop. At a
/// partial strength it mixes with the unfiltered backdrop in encoded sRGB.
#[test]
fn material_filter_bridge_stores_the_css_filtered_backdrop() {
    use crate::backend::renderer::{
        vulkan::{VulkanKawaseOutput, VulkanKawasePass},
        RoundedClip,
    };
    let Some(mut probe) = BlendProbe::new() else {
        return;
    };
    let size = Size::<i32, BufferCoord>::from((32, 24));
    let half = Size::<i32, BufferCoord>::from((16, 12));
    let quarter = Size::<i32, BufferCoord>::from((8, 6));
    let level_one = probe.renderer.create_buffer(Fourcc::Abgr8888, half).unwrap();
    let level_two = probe.renderer.create_buffer(Fourcc::Abgr8888, quarter).unwrap();
    let recipes: [([u8; 3], [f32; 3]); 2] = [
        // Light glass over the sky wallpaper.
        ([141, 185, 232], [1.10, 1.02, 1.15]),
        // Dark glass: the prototype's pale sky to its documented deep blue.
        ([150, 190, 230], [1.35, 0.40, 2.9]),
    ];
    for (backdrop, [contrast, brightness, saturation]) in recipes {
        let filtered = css_filter(backdrop, contrast.into(), brightness.into(), saturation.into());
        let backdrop_color = backdrop.map(|channel| f32::from(channel) / 255.0);
        for fused in [false, true] {
            let passes = if fused {
                vec![VulkanKawasePass::new(&level_one, &level_two, false, 1.5)]
            } else {
                vec![
                    VulkanKawasePass::new(&level_one, &level_two, false, 1.5),
                    VulkanKawasePass::new(&level_two, &level_one, true, 1.5)
                        .with_contrast(contrast)
                        .with_brightness(brightness)
                        .with_saturation(saturation),
                ]
            };
            probe.renderer.prepare_kawase_capture(&level_one).unwrap();
            probe.renderer.prepare_kawase_passes(&passes).unwrap();
            for strength in [1.0_f32, 0.2] {
                let pixels = probe.render(
                    Fourcc::Abgr8888,
                    size,
                    [backdrop_color[0], backdrop_color[1], backdrop_color[2], 1.0],
                    |frame, full| {
                        frame
                            .capture_and_downsample_framebuffer(full, &level_one, half, 1.5, &passes)
                            .expect("capture");
                        let clip = RoundedClip {
                            rect: full.to_f64(),
                            radius: 0.0,
                            exponent: 2.0,
                            aa_width: 1.0,
                            corner_mask: 0,
                        };
                        if fused {
                            frame
                                .render_kawase_with_rounded_clip(
                                    &level_two,
                                    Rectangle::from_size((8.0, 6.0).into()),
                                    full,
                                    &[full],
                                    Transform::Normal,
                                    strength,
                                    clip,
                                    VulkanKawaseOutput::new(quarter, 1.5)
                                        .with_filter(contrast, brightness, saturation),
                                )
                                .expect("fused material");
                        } else {
                            frame
                                .render_framebuffer_texture_with_rounded_clip(
                                    &level_one,
                                    Rectangle::from_size((16.0, 12.0).into()),
                                    full,
                                    &[full],
                                    strength,
                                    clip,
                                )
                                .expect("material bridge");
                        }
                    },
                );
                let strength = f64::from(strength);
                let expected: [f64; 3] = std::array::from_fn(|channel| {
                    filtered[channel] * strength + f64::from(backdrop[channel]) * (1.0 - strength)
                });
                assert_rgb_near(
                    &pixels,
                    expected,
                    3.0,
                    &format!("{backdrop:?} fused={fused} strength={strength}"),
                );
            }
        }
    }
}

/// The analytic material tint composites like CSS `background: rgba()` over
/// the filtered backdrop: light glass white@.64 over 54 stores 183 (linear
/// light stored 211), dark tint #12151B@.604 over 102 stores 51 (was 67).
#[test]
fn material_tint_composites_as_css_background_rgba() {
    use crate::backend::renderer::vulkan::VulkanMaterialTint;
    let Some(mut probe) = BlendProbe::new() else {
        return;
    };
    let size = Size::<i32, BufferCoord>::from((32, 32));
    let descriptor_source = probe.renderer.create_buffer(Fourcc::Abgr8888, size).unwrap();
    let cases: [([u8; 3], f32, [u8; 3]); 2] = [
        ([255, 255, 255], 0.64, [54, 54, 54]),
        ([0x12, 0x15, 0x1B], 0.604, [102, 102, 102]),
    ];
    for (tint, alpha, backdrop) in cases {
        let backdrop_color = backdrop.map(|channel| f32::from(channel) / 255.0);
        let pixels = probe.render(
            Fourcc::Abgr8888,
            size,
            [backdrop_color[0], backdrop_color[1], backdrop_color[2], 1.0],
            |frame, full| {
                let clip = BottomEdgeClip {
                    rect: full.to_f64(),
                    content_width: 24.0,
                    progress: 1.0,
                    edge_height: 32.0,
                    plateau_inset: 0.0,
                    geometry_scale: 1.0,
                };
                let color = tint.map(|channel| f32::from(channel) / 255.0);
                frame
                    .render_bottom_edge_material(
                        &descriptor_source,
                        Rectangle::from_size((32.0, 32.0).into()),
                        full,
                        &[full],
                        1.0,
                        clip,
                        VulkanMaterialTint::new([color[0], color[1], color[2], alpha]),
                    )
                    .expect("tint");
            },
        );
        // The plateau interior is fully covered; its centre carries the tint.
        let centre = &pixels[(16 * 32 + 16) as usize..(16 * 32 + 17) as usize];
        assert_rgb_near(
            centre,
            css_source_over(tint, f64::from(alpha), backdrop),
            2.0,
            &format!("tint {tint:?}@{alpha} over {backdrop:?}"),
        );
    }
}
