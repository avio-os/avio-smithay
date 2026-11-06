// A color-managed draw through the production frame path: what the shader stores
// must be what the equations say, and nothing else may change.

mod source_color {
    use super::*;
    use crate::backend::renderer::{RoundedClip, SourceColor, SourcePrimaries, SourceTransfer};

    /// The chromaticities of each named set, as the standards publish them.
    const BT709: [[f64; 2]; 3] = [[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]];
    const BT2020: [[f64; 2]; 3] = [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]];
    const P3: [[f64; 2]; 3] = [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]];
    const D65: [f64; 2] = [0.3127, 0.3290];

    /// `shaders/texture.frag`, by columns.
    const BT2020_TO_BT709: [[f32; 3]; 3] = [
        [1.660_491, -0.1245505, -0.0181508],
        [-0.5876411, 1.1328999, -0.1005789],
        [-0.0728499, -0.0083494, 1.1187297],
    ];
    const P3_TO_BT709: [[f32; 3]; 3] = [
        [1.2249402, -0.0420570, -0.0196376],
        [-0.2249402, 1.042_057, -0.0786361],
        [0.0, 0.0, 1.0982737],
    ];

    type Matrix = [[f64; 3]; 3];

    fn invert(m: Matrix) -> Matrix {
        let [[a, b, c], [d, e, f], [g, h, i]] = m;
        let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        [
            [e * i - f * h, c * h - b * i, b * f - c * e],
            [f * g - d * i, a * i - c * g, c * d - a * f],
            [d * h - e * g, b * g - a * h, a * e - b * d],
        ]
        .map(|row| row.map(|value| value / determinant))
    }

    fn apply(m: Matrix, v: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|row| (0..3).map(|column| m[row][column] * v[column]).sum())
    }

    fn multiply(a: Matrix, b: Matrix) -> Matrix {
        std::array::from_fn(|row| {
            std::array::from_fn(|column| (0..3).map(|k| a[row][k] * b[k][column]).sum())
        })
    }

    /// RGB to XYZ, rows first, from xy chromaticities.
    fn rgb_to_xyz(primaries: [[f64; 2]; 3], white: [f64; 2]) -> Matrix {
        let xyz = |[x, y]: [f64; 2]| [x / y, 1.0, (1.0 - x - y) / y];
        let [r, g, b] = primaries.map(xyz);
        let basis = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
        let scale = apply(invert(basis), xyz(white));
        basis.map(|row| std::array::from_fn(|column| row[column] * scale[column]))
    }

    #[test]
    fn the_shader_matrices_are_what_the_chromaticities_derive() {
        for (primaries, columns) in [(BT2020, BT2020_TO_BT709), (P3, P3_TO_BT709)] {
            let derived = multiply(invert(rgb_to_xyz(BT709, D65)), rgb_to_xyz(primaries, D65));
            for column in 0..3 {
                for row in 0..3 {
                    assert!(
                        (derived[row][column] - f64::from(columns[column][row])).abs() < 0.0005,
                        "{primaries:?} row {row} column {column}: {} against {}",
                        derived[row][column],
                        columns[column][row],
                    );
                }
            }
        }
    }

    fn pq_to_nits(signal: f32) -> f32 {
        let power = signal.powf(1.0 / 78.84375);
        10_000.0
            * ((power - 0.8359375).max(0.0) / (18.851_563 - 18.6875 * power)).powf(1.0 / 0.159_301_76)
    }

    fn hlg_to_scene(signal: f32) -> f32 {
        if signal <= 0.5 {
            signal * signal / 3.0
        } else {
            (((signal - 0.559_910_7) / 0.17883277).exp() + 0.28466892) / 12.0
        }
    }

    fn srgb_decode(signal: f32) -> f32 {
        if signal <= 0.04045 {
            signal / 12.92
        } else {
            ((signal + 0.055) / 1.055).powf(2.4)
        }
    }

    fn srgb_encode(light: f32) -> f32 {
        if light <= 0.0031308 {
            light * 12.92
        } else {
            1.055 * light.powf(1.0 / 2.4) - 0.055
        }
    }

    /// The conversion's definition, written out once more on the CPU.
    fn expected(color: SourceColor, texel: [f32; 4]) -> [f32; 4] {
        let alpha = texel[3];
        if alpha <= 0.0 {
            return texel;
        }
        let signal: [f32; 3] = std::array::from_fn(|i| (texel[i] / alpha).clamp(0.0, 1.0));
        let mut light: [f32; 3] = match color.transfer {
            SourceTransfer::Srgb => signal.map(srgb_decode),
            SourceTransfer::Gamma22 => signal.map(|value| value.powf(2.2)),
            SourceTransfer::Bt1886 => signal.map(|value| value.powf(2.4)),
            SourceTransfer::Pq => signal.map(|value| pq_to_nits(value) / color.reference_luminance),
            SourceTransfer::Hlg => {
                let scene = signal.map(hlg_to_scene);
                let luma = 0.2627 * scene[0] + 0.6780 * scene[1] + 0.0593 * scene[2];
                let gain = luma.max(0.000001).powf(0.2) * 1000.0 / color.reference_luminance;
                scene.map(|value| value * gain)
            }
        };
        let matrix = match color.primaries {
            SourcePrimaries::Bt709 => None,
            SourcePrimaries::Bt2020 => Some(BT2020_TO_BT709),
            SourcePrimaries::DisplayP3 => Some(P3_TO_BT709),
        };
        if let Some(columns) = matrix {
            light = std::array::from_fn(|row| (0..3).map(|c| columns[c][row] * light[c]).sum());
        }
        let light = light.map(|value| value.max(0.0));
        let peak = light[0].max(light[1]).max(light[2]);
        let limit = color.relative_peak();
        const KNEE: f32 = 0.75;
        let light = if limit <= 1.0 || peak <= KNEE {
            light
        } else {
            let over = (peak - KNEE) / (1.0 - KNEE);
            let span = (limit - KNEE) / (1.0 - KNEE);
            let mapped = KNEE + (1.0 - KNEE) * over * (1.0 + over / (span * span)) / (1.0 + over);
            light.map(|value| value * mapped / peak)
        };
        let encoded = light.map(|value| srgb_encode(value.clamp(0.0, 1.0)));
        [
            encoded[0] * alpha,
            encoded[1] * alpha,
            encoded[2] * alpha,
            alpha,
        ]
    }

    const HDR10: SourceColor = SourceColor {
        transfer: SourceTransfer::Pq,
        primaries: SourcePrimaries::Bt2020,
        reference_luminance: 203.0,
        peak_luminance: 1000.0,
    };

    fn nits_to_pq(nits: f32) -> f32 {
        let scaled = (nits / 10_000.0).powf(0.159_301_76);
        ((0.8359375 + 18.851_563 * scaled) / (1.0 + 18.6875 * scaled)).powf(78.84375)
    }

    #[test]
    fn reference_white_stays_bright_and_the_peak_reaches_exactly_white() {
        let white = |nits: f32| {
            let signal = nits_to_pq(nits);
            expected(HDR10, [signal, signal, signal, 1.0])
        };
        assert!(white(0.0)[0] < 0.001, "black stays black");
        let reference = white(203.0)[0];
        assert!(
            (0.90..0.99).contains(&reference),
            "reference white maps near SDR white, got {reference}"
        );
        assert!(white(1000.0)[0] > 0.999, "the claimed peak is white");
        assert!(white(4000.0)[0] > 0.999, "beyond the claim clamps");
        let mut previous = -1.0;
        for nits in (0..=1000).step_by(25) {
            let value = white(nits as f32)[0];
            assert!(value >= previous, "the curve never folds at {nits}");
            previous = value;
        }
    }

    #[test]
    fn dim_content_keeps_its_light() {
        // Well below the knee, tone mapping changes nothing: 100 cd/m² against a
        // reference of 203 is half of white, in any peak.
        let signal = nits_to_pq(100.0);
        for peak_luminance in [1000.0, 4000.0] {
            let color = SourceColor {
                peak_luminance,
                ..HDR10
            };
            let [value, ..] = expected(color, [signal, signal, signal, 1.0]);
            assert!((value - srgb_encode(100.0 / 203.0)).abs() < 0.002, "{value}");
        }
    }

    /// One texel drawn through the frame, under `color`.
    fn draw(
        probe: &mut BlendProbe,
        format: Fourcc,
        color: Option<SourceColor>,
        texel: [u8; 4],
    ) -> Vec<[f64; 4]> {
        let size = Size::<i32, BufferCoord>::from((PROBE_EXTENT, PROBE_EXTENT));
        let source = probe.import_abgr(size, &texel.repeat((size.w * size.h) as usize));
        probe.render(format, size, [0.0, 0.0, 0.0, 1.0], |frame, full| {
            frame.set_source_color(color);
            frame
                .render_texture_from_to(
                    &source,
                    Rectangle::from_size((f64::from(size.w), f64::from(size.h)).into()),
                    full,
                    &[full],
                    &[],
                    Transform::Normal,
                    1.0,
                )
                .expect("texture draw");
            frame.set_source_color(None);
        })
    }

    #[test]
    fn every_transfer_and_primaries_stores_what_the_conversion_says() {
        let Some(mut probe) = BlendProbe::new() else {
            return;
        };
        let colors = [
            HDR10,
            SourceColor {
                primaries: SourcePrimaries::DisplayP3,
                ..HDR10
            },
            SourceColor {
                transfer: SourceTransfer::Hlg,
                ..HDR10
            },
            SourceColor {
                transfer: SourceTransfer::Gamma22,
                primaries: SourcePrimaries::Bt709,
                reference_luminance: 80.0,
                peak_luminance: 80.0,
            },
            SourceColor {
                transfer: SourceTransfer::Bt1886,
                primaries: SourcePrimaries::Bt709,
                reference_luminance: 100.0,
                peak_luminance: 100.0,
            },
            SourceColor {
                transfer: SourceTransfer::Srgb,
                primaries: SourcePrimaries::DisplayP3,
                reference_luminance: 80.0,
                peak_luminance: 80.0,
            },
        ];
        let texels: [[u8; 4]; 6] = [
            [0, 0, 0, 255],
            [128, 128, 128, 255],
            [200, 200, 200, 255],
            [255, 255, 255, 255],
            [220, 90, 40, 255],
            // Premultiplied at half alpha.
            [100, 50, 25, 128],
        ];
        for format in probe.target_formats() {
            for color in colors {
                for texel in texels {
                    let input = texel.map(|channel| f32::from(channel) / 255.0);
                    let want = expected(color, input).map(|value| f64::from(value) * 255.0);
                    let pixels = draw(&mut probe, format, Some(color), texel);
                    // The target is opaque black, so the blend adds nothing.
                    assert_rgb_near(
                        &pixels,
                        [want[0], want[1], want[2]],
                        2.0,
                        &format!("{format:?} {color:?} {texel:?}"),
                    );
                }
            }
        }
    }

    #[test]
    fn a_draw_without_a_source_color_keeps_its_exact_bytes() {
        let Some(mut probe) = BlendProbe::new() else {
            return;
        };
        let texel = [220, 90, 40, 255];
        let pixels = draw(&mut probe, Fourcc::Abgr8888, None, texel);
        assert_rgb_near(&pixels, [220.0, 90.0, 40.0], 0.0, "untouched");
    }

    #[test]
    fn the_declaration_ends_when_the_frame_says_so() {
        let Some(mut probe) = BlendProbe::new() else {
            return;
        };
        let size = Size::<i32, BufferCoord>::from((PROBE_EXTENT, PROBE_EXTENT));
        let texel = [128, 128, 128, 255];
        let source = probe.import_abgr(size, &texel.repeat((size.w * size.h) as usize));
        let pixels = probe.render(Fourcc::Abgr8888, size, [0.0, 0.0, 0.0, 1.0], |frame, full| {
            frame.set_source_color(Some(HDR10));
            frame.set_source_color(None);
            frame
                .render_texture_from_to(
                    &source,
                    Rectangle::from_size((f64::from(size.w), f64::from(size.h)).into()),
                    full,
                    &[full],
                    &[],
                    Transform::Normal,
                    1.0,
                )
                .expect("texture draw");
        });
        assert_rgb_near(&pixels, [128.0, 128.0, 128.0], 0.0, "back to encoded sRGB");
    }

    /// How an HDR10 client reaches an HDR10 target: no declaration, so its 10-bit
    /// signal lands exactly where it was, which is what direct scanout would show.
    #[test]
    fn ten_bit_pq_passes_into_a_ten_bit_target_with_every_code_intact() {
        let Some(mut probe) = BlendProbe::new() else {
            return;
        };
        if !probe.target_formats().contains(&Fourcc::Argb2101010) {
            return;
        }
        let size = Size::<i32, BufferCoord>::from((PROBE_EXTENT, PROBE_EXTENT));
        let (red, green, blue) = (614u32, 301u32, 1000u32);
        let packed: u32 = (3 << 30) | (red << 20) | (green << 10) | blue;
        let bytes = packed.to_le_bytes().repeat((size.w * size.h) as usize);
        let source = {
            use crate::backend::renderer::ImportMem;
            match probe
                .renderer
                .import_memory(&bytes, Fourcc::Argb2101010, size, false)
            {
                Ok(source) => source,
                Err(error) => {
                    crate::backend::renderer::vulkan::test_support::capability_unavailable(
                        format_args!("Argb2101010 memory sources: {error}"),
                    );
                    return;
                }
            }
        };
        let pixels = probe.render(Fourcc::Argb2101010, size, [0.0, 0.0, 0.0, 1.0], |frame, full| {
            frame
                .render_texture_from_to(
                    &source,
                    Rectangle::from_size((f64::from(size.w), f64::from(size.h)).into()),
                    full,
                    &[full],
                    &[],
                    Transform::Normal,
                    1.0,
                )
                .expect("texture draw");
        });
        // `stored_rgba` scales ten bits to 0..=255, so half a code is 0.125.
        let scale = 255.0 / 1023.0;
        assert_rgb_near(
            &pixels,
            [
                f64::from(red) * scale,
                f64::from(green) * scale,
                f64::from(blue) * scale,
            ],
            0.125,
            "ten-bit signal",
        );
    }

    #[test]
    fn the_conversion_keeps_its_clip() {
        let Some(mut probe) = BlendProbe::new() else {
            return;
        };
        // A clip replaces the flag word, so this fails if the color bits go first.
        let size = Size::<i32, BufferCoord>::from((PROBE_EXTENT, PROBE_EXTENT));
        let texel = [128, 128, 128, 255];
        let want = expected(HDR10, texel.map(|channel| f32::from(channel) / 255.0))
            .map(|value| f64::from(value) * 255.0);
        let source = probe.import_abgr(size, &texel.repeat((size.w * size.h) as usize));
        let pixels = probe.render(Fourcc::Abgr8888, size, [0.0, 0.0, 0.0, 1.0], |frame, full| {
            frame.set_source_color(Some(HDR10));
            frame
                .render_texture_from_to_with_rounded_clip(
                    &source,
                    Rectangle::from_size((f64::from(size.w), f64::from(size.h)).into()),
                    full,
                    &[full],
                    &[],
                    Transform::Normal,
                    1.0,
                    RoundedClip {
                        rect: full.to_f64(),
                        radius: 0.0,
                        exponent: 2.0,
                        aa_width: 1.0,
                        corner_mask: 0xF,
                    },
                )
                .expect("clipped draw");
            frame.set_source_color(None);
        });
        assert_rgb_near(&pixels, [want[0], want[1], want[2]], 2.0, "clipped");
    }
}
