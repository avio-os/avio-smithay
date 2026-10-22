// Compositor-authored clip coverage through the production frame path: plain
// encoded coverage with a linear ramp 1 device pixel wide, centred on the
// edge, whatever the authored geometry scale.

/// Draws opaque white through `draw` over opaque black, and returns every
/// stored pixel's coverage in 0..=1 (its red channel).
fn white_through_clip(
    probe: &mut BlendProbe,
    size: Size<i32, BufferCoord>,
    draw: impl FnOnce(&mut VulkanFrame<'_>, &VulkanTexture, Rectangle<f64, BufferCoord>, Rectangle<i32, Physical>),
) -> Vec<f64> {
    let white = probe.import_abgr(size, &[255u8; 4].repeat((size.w * size.h) as usize));
    let src = Rectangle::from_size((f64::from(size.w), f64::from(size.h)).into());
    probe
        .render(Fourcc::Abgr8888, size, [0.0, 0.0, 0.0, 1.0], |frame, full| {
            draw(frame, &white, src, full)
        })
        .into_iter()
        .map(|pixel| pixel[0] / 255.0)
        .collect()
}

#[track_caller]
fn assert_coverage_near(stored: f64, expected: f64, what: &str) {
    assert!(
        (stored - expected).abs() <= 2.0 / 255.0,
        "{what}: coverage {stored:.3}, expected {expected:.3}"
    );
}

/// A rounded corner's coverage is `clamp(0.5 - distance / aa_width)`: at the
/// compositor's `aa_width` of 1 the edge ramps over one device pixel, a pixel
/// centre half a pixel inside the curve is fully covered, and one half a pixel
/// outside is empty. The corner meets its straight sides without a seam.
#[test]
fn rounded_clip_corners_ramp_linearly_across_their_aa_width() {
    use crate::backend::renderer::RoundedClip;
    let Some(mut probe) = BlendProbe::new() else {
        return;
    };
    const EXTENT: i32 = 24;
    const RADIUS: f64 = 9.0;
    let size = Size::<i32, BufferCoord>::from((EXTENT, EXTENT));
    for aa_width in [1.0_f32, 2.0] {
        let coverage = white_through_clip(&mut probe, size, |frame, white, src, full| {
            frame
                .render_texture_from_to_with_rounded_clip(
                    white,
                    src,
                    full,
                    &[full],
                    &[],
                    Transform::Normal,
                    1.0,
                    RoundedClip {
                        rect: full.to_f64(),
                        radius: RADIUS as f32,
                        exponent: 2.0,
                        aa_width,
                        corner_mask: RoundedClip::ALL_CORNERS,
                    },
                )
                .expect("rounded clip draw");
        });
        // Each pixel centre's distance to its nearest edge, folding all four
        // corners onto the top-left one.
        let fold = |index: i32| {
            let centre = f64::from(index) + 0.5;
            centre.min(f64::from(EXTENT) - centre)
        };
        for y in 0..EXTENT {
            for x in 0..EXTENT {
                let (from_side, from_top) = (fold(x), fold(y));
                let expected = if from_side < RADIUS && from_top < RADIUS {
                    let distance = (RADIUS - from_side).hypot(RADIUS - from_top) - RADIUS;
                    (0.5 - distance / f64::from(aa_width)).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                assert_coverage_near(
                    coverage[(y * EXTENT + x) as usize],
                    expected,
                    &format!("aa_width {aa_width}, pixel ({x}, {y})"),
                );
            }
        }
    }
}

/// The Dock's bottom-edge pull ramps over one device pixel centred on its
/// plateau top at every geometry scale (its old ramp grew with the scale and
/// covered only the inner half of the edge).
#[test]
fn bottom_edge_pull_ramps_over_one_device_pixel_at_every_geometry_scale() {
    let Some(mut probe) = BlendProbe::new() else {
        return;
    };
    const WIDTH: i32 = 64;
    const HEIGHT: i32 = 32;
    const EDGE_HEIGHT: f32 = 20.3;
    // Settled, the plateau top sits at HEIGHT - EDGE_HEIGHT = 11.7: the row
    // centred at 11.5 lies 0.2 px outside the edge, the row at 12.5 0.8 px in.
    let plateau_top = f64::from(HEIGHT) - f64::from(EDGE_HEIGHT);
    let size = Size::<i32, BufferCoord>::from((WIDTH, HEIGHT));
    for geometry_scale in [1.0_f32, 1.75, 2.0] {
        let coverage = white_through_clip(&mut probe, size, |frame, white, src, full| {
            frame
                .render_texture_from_to_with_bottom_edge_clip(
                    white,
                    src,
                    full,
                    &[full],
                    &[],
                    Transform::Normal,
                    1.0,
                    BottomEdgeClip {
                        rect: full.to_f64(),
                        content_width: 24.0,
                        progress: 1.0,
                        edge_height: EDGE_HEIGHT,
                        plateau_inset: 0.0,
                        geometry_scale,
                    },
                )
                .expect("bottom-edge clip draw");
        });
        // The centre column crosses the flat plateau, clear of its corners.
        let column = WIDTH / 2;
        for y in 0..HEIGHT {
            let signed_inside = f64::from(y) + 0.5 - plateau_top;
            assert_coverage_near(
                coverage[(y * WIDTH + column) as usize],
                (0.5 + signed_inside).clamp(0.0, 1.0),
                &format!("geometry_scale {geometry_scale}, row {y}"),
            );
        }
    }
}
