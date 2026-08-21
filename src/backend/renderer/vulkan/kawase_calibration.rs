//! Effective-Gaussian calibration for the dual-Kawase chain.
//!
//! The shell drives [`super::kawase`] from a `blur_radius` in pixels; the design
//! language it implements is written in Gaussian sigma. Nothing maps one to the
//! other, so this module measures it on a real device: render a hard step edge,
//! run the pyramid exactly the way the shell runs it, read the blurred edge back
//! and fit the effective sigma of the composite kernel.
//!
//! The whole module is test-only. The table lives in
//! [`kawase_sigma_calibration_table`]; run it with
//! `cargo test -p smithay --lib kawase_sigma_calibration_table -- --nocapture`
//! to print the surface rather than just assert on it.

// The tables are the product here, and the test harness only surfaces stdout.
// `tracing` remains the rule for library logging; a calibration report is not
// library logging.
#![allow(clippy::disallowed_macros)]

use crate::{
    backend::{
        allocator::Fourcc,
        renderer::{
            vulkan::{VulkanKawasePass, VulkanTexture},
            Bind, Color32F, ExportMem, Frame, Offscreen, Renderer,
        },
        vulkan::{version::Version, Instance, PhysicalDevice},
    },
    utils::{Buffer as BufferCoord, Physical, Point, Rectangle, Size, Transform},
};

use super::VulkanRenderer;

/// The shell's fixed kawase spread (`SHELL_MATERIAL_KAWASE_OFFSET`).
const SHELL_OFFSET: f32 = 1.5;

/// The shell's pyramid depth cap (`SHELL_MATERIAL_MAX_DOWNSAMPLE_LEVELS`).
const MAX_LEVELS: usize = 5;

/// Spreads swept for the sigma(offset) response. The overhaul wants per-region
/// sigma control through the per-pass offset, so this axis is a deliverable and
/// not just a sanity check.
const OFFSETS: [f32; 5] = [0.75, 1.0, 1.5, 2.0, 2.5];

/// Bracket the spread search runs in. Below 0.5 the taps collapse onto the texel
/// they started from; above 3.0 the 5-tap downsample starts to alias.
const OFFSET_SEARCH: (f32, f32) = (0.5, 3.0);

/// Sigmas the design language asks for, in logical pixels.
const SPEC_SIGMAS: [f64; 6] = [12.0, 18.0, 28.0, 30.0, 32.0, 36.0];

/// Root sizes swept for size dependence. 500 is deliberately not a power of two,
/// so the shell's `(w + 1) / 2` halving rounds and any size coupling shows up.
const ROOT_SIZES: [i32; 3] = [512, 1024, 500];

/// How many step-edge positions each configuration is measured at. The pyramid
/// samples on a `2^levels` grid, so the edge's phase against that grid changes
/// the kernel; averaging over phases is what the shell actually experiences.
const PHASE_SAMPLES: usize = 4;

/// Reference to the shell's own level-count rule, mirrored here so the table can
/// be keyed on `blur_radius` instead of on an abstract level count.
///
/// `SHELL_MATERIAL_BLUR_DOWNSAMPLE_RADIUS_STEP_PX` is 8.0 and the cap is 5.
fn shell_level_count(blur_radius_physical: f64) -> usize {
    if blur_radius_physical <= 2.0 {
        return 0;
    }
    (((blur_radius_physical / 8.0).max(1.0).log2().ceil() as usize) + 1).clamp(1, MAX_LEVELS)
}

/// The shell rounds every material offscreen up to this quantum
/// (`SHELL_MATERIAL_OFFSCREEN_SIZE_QUANTUM`).
const SHELL_OFFSCREEN_QUANTUM: i32 = 64;

fn shell_canonical_extent(size: Size<i32, BufferCoord>) -> Size<i32, BufferCoord> {
    let round_up = |value: i32| {
        if value <= 0 {
            value
        } else {
            (value + SHELL_OFFSCREEN_QUANTUM - 1) / SHELL_OFFSCREEN_QUANTUM * SHELL_OFFSCREEN_QUANTUM
        }
    };
    Size::from((round_up(size.w), round_up(size.h)))
}

/// The offscreen sizes the shell actually reserves for a material atlas.
///
/// Level 0 is the canonicalised atlas allocation; every deeper level halves the
/// *content* and is then canonicalised on its own. Since the passes map the full
/// source allocation onto the full destination allocation, what scales the
/// content between levels is the ratio of those rounded allocations — not two.
fn shell_pyramid_allocations(
    content: Size<i32, BufferCoord>,
    blur_radius_physical: f64,
) -> Vec<Size<i32, BufferCoord>> {
    let mut sizes = vec![shell_canonical_extent(content)];
    let mut level = content;
    for _ in 0..shell_level_count(blur_radius_physical) {
        let next = Size::<i32, BufferCoord>::from(((level.w + 1) / 2, (level.h + 1) / 2));
        if next == level || next.w < 2 || next.h < 2 {
            break;
        }
        sizes.push(shell_canonical_extent(next));
        level = next;
    }
    sizes
}

fn transpose(sizes: &[Size<i32, BufferCoord>]) -> Vec<Size<i32, BufferCoord>> {
    sizes.iter().map(|size| Size::from((size.h, size.w))).collect()
}

fn srgb_byte_to_linear(byte: u8) -> f64 {
    let c = f64::from(byte) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Abramowitz & Stegun 7.1.26; |error| < 1.5e-7, far below the 8-bit floor the
/// measurement itself sits on.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = ((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736) * t
        + 0.254_829_592)
        * t;
    sign * (1.0 - poly * (-x * x).exp())
}

/// Step response of a unit Gaussian: the shape a perfectly Gaussian blur would
/// leave behind on a hard edge.
fn gaussian_step(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Where a rising profile first reaches `level`, in sample units, interpolated
/// linearly between the bracketing samples.
fn first_crossing(profile: &[f64], level: f64) -> Option<f64> {
    for index in 1..profile.len() {
        let (lower, upper) = (profile[index - 1], profile[index]);
        if lower < level && upper >= level {
            let span = upper - lower;
            if span <= f64::EPSILON {
                return Some(index as f64);
            }
            return Some((index - 1) as f64 + (level - lower) / span);
        }
    }
    None
}

/// Centre and RMS width of the blur kernel.
///
/// For an input step that flips between texels `b - 1` and `b`, the step
/// response is `p[i] = sum(K[k], k <= i - b)`, so `p[i] - p[i - 1]` *is* `K[i - b]`
/// — the discrete kernel, recovered exactly, with no least squares in sight. RMS
/// width is the number a spec wants because variance composes additively: two
/// blurs in series give `sqrt(s1^2 + s2^2)` whatever their shapes.
fn kernel_moments(profile: &[f64]) -> Option<(f64, f64)> {
    let mut mass = 0.0;
    let mut first = 0.0;
    let mut second = 0.0;
    for index in 1..profile.len() {
        let weight = profile[index] - profile[index - 1];
        let position = index as f64;
        mass += weight;
        first += weight * position;
        second += weight * position * position;
    }
    if mass <= 1e-6 {
        return None;
    }
    let mean = first / mass;
    Some((mean, (second / mass - mean * mean).max(0.0).sqrt()))
}

/// Sigma from the 25%-75% span, which is exactly 1.349 sigma for a Gaussian.
/// Blind to the tails, so it is the cross-check that quantisation noise in the
/// far tail cannot move.
fn quartile_sigma(profile: &[f64]) -> Option<f64> {
    let low = first_crossing(profile, 0.25)?;
    let high = first_crossing(profile, 0.75)?;
    Some((high - low) / 1.349)
}

/// The continuous position a profile sample stands for.
///
/// `p[i]` accumulates every kernel tap up to and including offset `i`, so as a
/// cumulative distribution it is sampled half a texel to the right of the tap
/// positions [`kernel_moments`] works in. Everything that compares a profile
/// level against a kernel centre has to cross that half texel.
fn profile_position(index: f64) -> f64 {
    index + 0.5
}

/// Largest absolute gap between the measured profile and the Gaussian step
/// response with the same centre and sigma — how Gaussian the fit really is.
///
/// A *perfect* discrete Gaussian does not score zero here: summing samples
/// instead of integrating leaves an Euler-Maclaurin gap of `0.0101 / sigma^2`
/// against the continuous erf. That floor is 7e-5 at sigma 12 and falls from
/// there, so anything an actual measurement reports above it is shape, not
/// sampling.
fn erf_residual(profile: &[f64], mean: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return f64::INFINITY;
    }
    profile
        .iter()
        .enumerate()
        .map(|(index, measured)| {
            (measured - gaussian_step((profile_position(index as f64) - mean) / sigma)).abs()
        })
        .fold(0.0, f64::max)
}

/// How far below the edge centre the profile is still lifted off black by
/// `threshold`. This is the padding a blurred region actually needs; the "3
/// sigma" rule of thumb only approximates it, and dual-kawase tails are not
/// Gaussian tails.
fn tail_support(profile: &[f64], mean: f64, threshold: f64) -> Option<f64> {
    first_crossing(profile, threshold).map(|crossing| mean - profile_position(crossing))
}

/// One fitted edge profile.
#[derive(Debug, Clone, Copy)]
struct Fit {
    sigma_rms: f64,
    sigma_quartile: f64,
    residual: f64,
    support_1pct: f64,
    support_0p1pct: f64,
    /// Where the response first lifts off exactly zero. A dual-kawase kernel is a
    /// finite sum of bilinear taps, so unlike a Gaussian it has genuinely compact
    /// support: past this distance a blurred region contributes nothing at all.
    support_hard: f64,
    /// False when the profile is still climbing at the image border, which means
    /// the support outran the measurement window and the tail numbers are floors.
    settled: bool,
}

fn fit_profile(profile: &[f64]) -> Option<Fit> {
    let (mean, sigma_rms) = kernel_moments(profile)?;
    let sigma_quartile = quartile_sigma(profile)?;
    let last = profile.len() - 1;
    Some(Fit {
        sigma_rms,
        sigma_quartile,
        residual: erf_residual(profile, mean, sigma_rms),
        support_1pct: tail_support(profile, mean, 0.01)?,
        support_0p1pct: tail_support(profile, mean, 0.001)?,
        // One 8-bit sRGB code is 3.04e-4 of linear range, so this threshold finds
        // the last sample the chain left exactly black.
        support_hard: tail_support(profile, mean, 1e-5)?,
        settled: (profile[1] - profile[0]).abs() < 1e-9 && (profile[last] - profile[last - 1]).abs() < 1e-9,
    })
}

/// A pyramid of offscreens shaped like the one the shell reserves: level 0 at
/// the root size, each further level halved with the shell's `(w + 1) / 2` rule.
struct Pyramid {
    levels: Vec<VulkanTexture>,
    sizes: Vec<Size<i32, BufferCoord>>,
}

struct EdgeProbe {
    renderer: VulkanRenderer,
    format: Fourcc,
}

impl EdgeProbe {
    /// Returns `None` when no usable Vulkan device is present, so the calibration
    /// skips on hosted runners rather than failing there.
    fn new() -> Option<Self> {
        let instance = Instance::new(Version::VERSION_1_3, None).ok()?;
        let physical_device = PhysicalDevice::enumerate(&instance).ok()?.next()?;
        let mut renderer = VulkanRenderer::new(&physical_device).ok()?;
        let format = [
            Fourcc::Argb8888,
            Fourcc::Abgr8888,
            Fourcc::Xrgb8888,
            Fourcc::Xbgr8888,
        ]
        .into_iter()
        .find(|format| {
            Offscreen::<VulkanTexture>::create_buffer(&mut renderer, *format, Size::from((4, 4))).is_ok()
        })?;
        Some(Self { renderer, format })
    }

    fn pyramid_from_sizes(&mut self, sizes: Vec<Size<i32, BufferCoord>>) -> Option<Pyramid> {
        let mut levels = Vec::with_capacity(sizes.len());
        for size in &sizes {
            levels.push(
                Offscreen::<VulkanTexture>::create_buffer(&mut self.renderer, self.format, *size).ok()?,
            );
        }
        Some(Pyramid { levels, sizes })
    }

    fn pyramid(&mut self, root: i32, depth: usize) -> Option<Pyramid> {
        let mut sizes = vec![Size::<i32, BufferCoord>::from((root, root))];
        for _ in 0..depth {
            let previous = *sizes.last()?;
            sizes.push(Size::from(((previous.w + 1) / 2, (previous.h + 1) / 2)));
        }
        self.pyramid_from_sizes(sizes)
    }

    /// Paints a hard black|white step into level 0 with the boundary at
    /// `edge_x`, so the continuous edge sits at `edge_x - 0.5` in texel-centre
    /// coordinates. Both clears land on exact bytes (0 and 255) whichever way the
    /// attachment encodes, so the input carries no encoding error of its own.
    fn paint_step(&mut self, pyramid: &mut Pyramid, edge_x: i32) -> Option<()> {
        let root = pyramid.sizes[0];
        let physical = Size::<i32, Physical>::from((root.w, root.h));
        let mut target = self.renderer.bind(&mut pyramid.levels[0]).ok()?;
        let mut frame = self
            .renderer
            .render(&mut target, physical, Transform::Normal)
            .ok()?;
        frame
            .clear(
                Color32F::new(0.0, 0.0, 0.0, 1.0),
                &[Rectangle::from_size(physical)],
            )
            .ok()?;
        frame
            .clear(
                Color32F::new(1.0, 1.0, 1.0, 1.0),
                &[Rectangle::new(
                    Point::from((edge_x, 0)),
                    Size::from((physical.w - edge_x, physical.h)),
                )],
            )
            .ok()?;
        let _ = frame.finish().ok()?.wait();
        Some(())
    }

    /// Reads the middle scanline of level 0 back and decodes it to linear light.
    /// Storage is sRGB-encoded (the attachment view does the encode), so the
    /// decode here is the exact inverse of what the chain wrote.
    fn read_profile(&mut self, pyramid: &mut Pyramid) -> Option<Vec<f64>> {
        let root = pyramid.sizes[0];
        let target = self.renderer.bind(&mut pyramid.levels[0]).ok()?;
        let mapping = self
            .renderer
            .copy_framebuffer(&target, Rectangle::from_size(root), self.format)
            .ok()?;
        let data = self.renderer.map_texture(&mapping).ok()?;
        let stride = root.w as usize * 4;
        let row = (root.h / 2) as usize * stride;
        Some(
            (0..root.w as usize)
                .map(|x| srgb_byte_to_linear(data[row + x * 4]))
                .collect(),
        )
    }

    /// Runs the shell's chain — `levels` downsamples followed by the mirrored
    /// upsamples, every pass at the same `offset` — and returns the normalised
    /// edge profile in linear light.
    fn measure(
        &mut self,
        pyramid: &mut Pyramid,
        levels: usize,
        offset: f32,
        edge_x: i32,
    ) -> Option<Vec<f64>> {
        self.paint_step(pyramid, edge_x)?;

        let mut passes = Vec::with_capacity(levels * 2);
        for level in 0..levels {
            passes.push(VulkanKawasePass::new(
                &pyramid.levels[level],
                &pyramid.levels[level + 1],
                false,
                offset,
            ));
        }
        for level in (0..levels).rev() {
            passes.push(VulkanKawasePass::new(
                &pyramid.levels[level + 1],
                &pyramid.levels[level],
                true,
                offset,
            ));
        }
        let _ = self.renderer.kawase_texture_chain(&passes).ok()?.wait();

        let raw = self.read_profile(pyramid)?;
        let (low, high) = (raw[0], raw[raw.len() - 1]);
        if high - low < 0.5 {
            return None;
        }
        Some(
            raw.into_iter()
                .map(|value| (value - low) / (high - low))
                .collect(),
        )
    }

    /// Averages a configuration over `PHASE_SAMPLES` edge phases against the
    /// `2^levels` pyramid grid, and reports the sigma spread across them.
    fn sweep(&mut self, root: i32, levels: usize, offset: f32) -> Option<(Fit, f64)> {
        let pyramid = self.pyramid(root, levels)?;
        self.sweep_pyramid(pyramid, offset)
    }

    /// The same sweep over a pyramid whose level sizes were chosen by the caller,
    /// so the shell's real allocation sizes can be measured and not just its
    /// idealised halving.
    fn sweep_pyramid(&mut self, mut pyramid: Pyramid, offset: f32) -> Option<(Fit, f64)> {
        let levels = pyramid.sizes.len() - 1;
        let root = pyramid.sizes[0].w;
        let stride = ((1usize << levels) / PHASE_SAMPLES).max(1);
        let phases = (1usize << levels).min(PHASE_SAMPLES);
        let mut fits = Vec::with_capacity(phases);
        for phase in 0..phases {
            let edge_x = root / 2 + (phase * stride) as i32;
            let profile = self.measure(&mut pyramid, levels, offset, edge_x)?;
            fits.push(fit_profile(&profile)?);
        }
        let count = fits.len() as f64;
        let mean = Fit {
            sigma_rms: fits.iter().map(|fit| fit.sigma_rms).sum::<f64>() / count,
            sigma_quartile: fits.iter().map(|fit| fit.sigma_quartile).sum::<f64>() / count,
            residual: fits.iter().map(|fit| fit.residual).fold(0.0, f64::max),
            support_1pct: fits.iter().map(|fit| fit.support_1pct).fold(0.0, f64::max),
            support_0p1pct: fits.iter().map(|fit| fit.support_0p1pct).fold(0.0, f64::max),
            support_hard: fits.iter().map(|fit| fit.support_hard).fold(0.0, f64::max),
            settled: fits.iter().all(|fit| fit.settled),
        };
        let spread = fits.iter().map(|fit| fit.sigma_rms).fold(0.0, f64::max)
            - fits.iter().map(|fit| fit.sigma_rms).fold(f64::MAX, f64::min);
        Some((mean, spread))
    }

    /// The spread that lands `target` sigma at this pyramid depth, or `None` when
    /// the depth cannot reach it inside [`OFFSET_SEARCH`].
    ///
    /// Sigma rises monotonically with the spread at every depth measured, so a
    /// bisection is enough; there is no analytic inverse because bilinear taps
    /// contribute a variance term that depends on where each tap falls between
    /// texels, and that term is not smooth in the spread.
    fn solve_offset(&mut self, root: i32, levels: usize, target: f64) -> Option<(f32, Fit)> {
        let (mut low, mut high) = OFFSET_SEARCH;
        if self.sweep(root, levels, low)?.0.sigma_rms > target
            || self.sweep(root, levels, high)?.0.sigma_rms < target
        {
            return None;
        }
        let mut fit = self.sweep(root, levels, high)?.0;
        for _ in 0..12 {
            let middle = 0.5 * (low + high);
            let (candidate, _) = self.sweep(root, levels, middle)?;
            if candidate.sigma_rms < target {
                low = middle;
            } else {
                high = middle;
            }
            fit = candidate;
        }
        Some((0.5 * (low + high), fit))
    }
}

/// Step response of a *discrete* Gaussian kernel of the given sigma, with the
/// input step flipping between texels `edge - 1` and `edge`. This is the shape
/// the estimators are supposed to see, so it is the shape they get validated on.
fn synthetic_gaussian_profile(width: usize, edge: i64, sigma: f64) -> Vec<f64> {
    let half = (6.0 * sigma).ceil() as i64;
    let taps = (-half..=half)
        .map(|k| (-((k * k) as f64) / (2.0 * sigma * sigma)).exp())
        .collect::<Vec<_>>();
    let mass = taps.iter().sum::<f64>();
    (0..width)
        .map(|index| {
            let reach = index as i64 - edge;
            taps.iter()
                .enumerate()
                .filter(|(tap, _)| (*tap as i64 - half) <= reach)
                .map(|(_, weight)| weight)
                .sum::<f64>()
                / mass
        })
        .collect()
}

/// The estimators must be right before any measurement made with them means
/// anything, so feed them profiles whose sigma is known by construction. This is
/// the hand-checked case: exact discrete-Gaussian step responses.
#[test]
fn sigma_estimators_recover_a_synthetic_gaussian_edge() {
    for truth in [3.0_f64, 7.0, 17.5] {
        let profile = synthetic_gaussian_profile(512, 256, truth);
        let fit = fit_profile(&profile).expect("synthetic profile must fit");

        assert!(
            (fit.sigma_rms - truth).abs() < 0.02 * truth,
            "moment sigma {} should recover {truth}",
            fit.sigma_rms
        );
        assert!(
            (fit.sigma_quartile - truth).abs() < 0.02 * truth,
            "quartile sigma {} should recover {truth}",
            fit.sigma_quartile
        );
        // Only the sampling floor documented on `erf_residual` may survive.
        assert!(
            fit.residual < 0.015 / (truth * truth),
            "a Gaussian profile must fit a Gaussian down to the sampling floor, got {} at {truth}",
            fit.residual
        );
        // 1% and 0.1% of a Gaussian's step response sit at 2.326 and 3.090 sigma.
        assert!(
            (fit.support_1pct / truth - 2.326).abs() < 0.05,
            "1% support {} should land at 2.326 sigma for {truth}",
            fit.support_1pct
        );
        assert!(
            (fit.support_0p1pct / truth - 3.090).abs() < 0.05,
            "0.1% support {} should land at 3.090 sigma for {truth}",
            fit.support_0p1pct
        );
    }
}

/// The second hand-checked case, this one through the real device.
///
/// A downsample pass between two images of the *same* size puts every tap on a
/// position the kernel can be written out by hand. The destination texel `i`
/// centre maps to source coordinate `i`; taps land at `i` (weight 4) and at
/// `i ± offset` (weight 2 each), and at offset 1.5 those bilinear taps split
/// 0.25/0.75 across neighbouring texels. The kernel is therefore
/// `[1.5, 5, 1.5] / 8` over `[-1, 0, +1]`, whose variance is `3 / 8 = 0.375`
/// and whose sigma is 0.6124 — with no resampling in the way, the measured
/// moment must reproduce that exactly.
///
/// The shell never runs a same-size pass; this configuration exists only to pin
/// the measurement chain against arithmetic done on paper.
#[test]
fn same_size_downsample_matches_the_hand_derived_kernel() {
    let Some(mut probe) = EdgeProbe::new() else {
        return;
    };
    let root = 256;
    let size = Size::<i32, BufferCoord>::from((root, root));
    let mut pyramid = Pyramid {
        levels: vec![
            Offscreen::<VulkanTexture>::create_buffer(&mut probe.renderer, probe.format, size)
                .expect("offscreen alloc"),
            Offscreen::<VulkanTexture>::create_buffer(&mut probe.renderer, probe.format, size)
                .expect("offscreen alloc"),
        ],
        sizes: vec![size, size],
    };

    probe.paint_step(&mut pyramid, root / 2).expect("paint step");
    let passes = [VulkanKawasePass::new(
        &pyramid.levels[0],
        &pyramid.levels[1],
        false,
        SHELL_OFFSET,
    )];
    let _ = probe
        .renderer
        .kawase_texture_chain(&passes)
        .expect("kawase chain should record and submit")
        .wait();

    // Read the destination, not level 0.
    pyramid.levels.swap(0, 1);
    let raw = probe.read_profile(&mut pyramid).expect("readback");
    let profile = raw
        .iter()
        .map(|value| (value - raw[0]) / (raw[raw.len() - 1] - raw[0]))
        .collect::<Vec<_>>();
    let (_, sigma) = kernel_moments(&profile).expect("kernel moments");

    let expected = 0.375_f64.sqrt();
    assert!(
        (sigma - expected).abs() < 0.02,
        "same-size downsample sigma {sigma} should match the hand-derived {expected}"
    );
}

/// The calibration itself: the sigma(levels, offset, size) surface plus the tail
/// support that sets the real padding requirement.
///
/// Assertions here are the invariants the mapping has to satisfy for downstream
/// numbers to be trustworthy; the numbers themselves are printed, since pinning
/// them to a tolerance would only re-encode this run's driver.
#[test]
fn kawase_sigma_calibration_table() {
    let Some(mut probe) = EdgeProbe::new() else {
        return;
    };

    println!(
        "\n{:>5} {:>7} {:>6} {:>8} {:>8} {:>7} {:>8} {:>7} {:>7} {:>8} {:>8}",
        "root",
        "levels",
        "offset",
        "radius",
        "sigmaRMS",
        "spread",
        "sigmaQ",
        "resid",
        "sup1%",
        "sup0.1%",
        "supHard"
    );

    let mut shell_offset_fits: Vec<(i32, usize, Fit)> = Vec::new();
    for root in ROOT_SIZES {
        for levels in 1..=MAX_LEVELS {
            for offset in OFFSETS {
                let Some((fit, spread)) = probe.sweep(root, levels, offset) else {
                    continue;
                };
                // The blur_radius band that selects this level count.
                let radius = if levels == 1 {
                    "<=8".to_string()
                } else {
                    format!("{}-{}", (1 << (levels - 2)) * 8 + 1, (1 << (levels - 1)) * 8)
                };
                println!(
                    "{root:>5} {levels:>7} {offset:>6.2} {radius:>8} {:>8.3} {spread:>7.3} {:>7.3} {:>7.4} {:>7.1} {:>8.1} {:>8.1}{}",
                    fit.sigma_rms,
                    fit.sigma_quartile,
                    fit.residual,
                    fit.support_1pct,
                    fit.support_0p1pct,
                    fit.support_hard,
                    if fit.settled { "" } else { "  (support clipped)" }
                );
                if (offset - SHELL_OFFSET).abs() < f32::EPSILON {
                    shell_offset_fits.push((root, levels, fit));
                }

                assert!(
                    fit.sigma_rms > 0.0 && fit.sigma_rms.is_finite(),
                    "sigma must be finite and positive at root {root}, {levels} levels, offset {offset}"
                );
                assert!(
                    (fit.sigma_quartile - fit.sigma_rms).abs() < 0.35 * fit.sigma_rms,
                    "the two estimators must agree to within 35% at root {root}, {levels} levels, \
                     offset {offset}: rms {} vs quartile {}",
                    fit.sigma_rms,
                    fit.sigma_quartile
                );
            }
        }
    }

    assert!(
        !shell_offset_fits.is_empty(),
        "the shell's own offset must be measurable"
    );

    // Sigma has to grow with pyramid depth, or the level-count rule is not a
    // blur-strength control at all.
    for root in ROOT_SIZES {
        let series = shell_offset_fits
            .iter()
            .filter(|(size, _, _)| *size == root)
            .map(|(_, levels, fit)| (*levels, fit.sigma_rms))
            .collect::<Vec<_>>();
        for window in series.windows(2) {
            assert!(
                window[1].1 > window[0].1 * 1.5,
                "sigma must grow with pyramid depth at root {root}: {:?}",
                series
            );
        }
    }

    // Level 0 is the only place a size could enter: every tap offset is derived
    // from the smaller level's own allocation, so sigma in level-0 pixels must
    // not depend on how big level 0 is.
    for levels in 1..=MAX_LEVELS {
        let at = |root: i32| {
            shell_offset_fits
                .iter()
                .find(|(size, depth, _)| *size == root && *depth == levels)
                .map(|(_, _, fit)| fit.sigma_rms)
        };
        if let (Some(small), Some(large)) = (at(512), at(1024)) {
            assert!(
                (large - small).abs() < 0.02 * small,
                "sigma at {levels} levels must not depend on the root size: 512 gave {small}, \
                 1024 gave {large}"
            );
        }
    }

    // What the shell's own `blur_radius` values actually buy, at the shell's own
    // fixed spread. `blur_radius` only reaches the chain through a level count,
    // so it is a five-rung staircase, not a continuous control: neighbouring
    // radii inside one band are indistinguishable.
    println!(
        "\n{:>12} {:>7} {:>9} {:>9}",
        "blur_radius", "levels", "sigmaRMS", "sup0.1%"
    );
    for radius in [8.0_f64, 12.0, 16.0, 18.0, 24.0, 32.0, 36.0, 48.0, 58.0, 78.0] {
        let levels = shell_level_count(radius);
        let Some(fit) = shell_offset_fits
            .iter()
            .find(|(root, depth, _)| *root == 512 && *depth == levels)
            .map(|(_, _, fit)| *fit)
        else {
            continue;
        };
        println!(
            "{radius:>12.0} {levels:>7} {:>9.2} {:>9.1}",
            fit.sigma_rms, fit.support_0p1pct
        );
    }
}

/// The practical half of the calibration: for every sigma the design language
/// names, which pyramid depth and spread actually produce it.
///
/// The shell's fixed 1.5 spread only reaches a five-rung ladder — roughly 2.1,
/// 4.8, 9.8, 19.6, 39.2 — so none of the spec's sigmas are reachable without
/// moving the spread. Each row here is the depth/spread pair that lands one.
#[test]
fn spec_sigmas_resolve_to_a_depth_and_spread() {
    let Some(mut probe) = EdgeProbe::new() else {
        return;
    };
    let root = 512;

    println!(
        "\n{:>6} {:>7} {:>7} {:>9} {:>7} {:>8} {:>8}",
        "target", "levels", "offset", "sigmaRMS", "resid", "sup0.1%", "supHard"
    );
    let mut solved = 0usize;
    for target in SPEC_SIGMAS {
        for levels in 1..=MAX_LEVELS {
            let Some((offset, fit)) = probe.solve_offset(root, levels, target) else {
                continue;
            };
            println!(
                "{target:>6.1} {levels:>7} {offset:>7.3} {:>9.3} {:>7.4} {:>8.1} {:>8.1}",
                fit.sigma_rms, fit.residual, fit.support_0p1pct, fit.support_hard
            );
            assert!(
                (fit.sigma_rms - target).abs() < 0.05 * target,
                "solved spread {offset} at {levels} levels should land sigma {target}, got {}",
                fit.sigma_rms
            );
            solved += 1;
        }
    }
    assert!(
        solved >= SPEC_SIGMAS.len(),
        "every spec sigma must be reachable by some depth/spread pair"
    );
}

/// What the shell's 64 px allocation quantum does to the blur it asked for.
///
/// Every pass maps the whole source allocation onto the whole destination
/// allocation, so the factor the content shrinks by between two levels is the
/// ratio of their *rounded* allocations. On a wide, short atlas — a dock, a
/// title bar, a toast — the short axis hits the quantum floor after one or two
/// levels and stops halving, while the long axis keeps going. The chain then
/// blurs the two axes by different amounts from one `blur_radius`.
///
/// Measured on the transposed geometry: the chain is separable per axis, so
/// running the existing vertical-edge probe against transposed allocations
/// measures the vertical response of the original.
#[test]
fn the_offscreen_quantum_blurs_the_two_axes_unequally() {
    let Some(mut probe) = EdgeProbe::new() else {
        return;
    };

    // A dock-shaped material atlas at the recipe's own radius.
    let content = Size::<i32, BufferCoord>::from((800, 120));
    let radius = 48.0;
    let sizes = shell_pyramid_allocations(content, radius);
    let described = sizes
        .iter()
        .map(|size| format!("{}x{}", size.w, size.h))
        .collect::<Vec<_>>()
        .join(" -> ");
    println!(
        "\nshell allocations for {}x{} content at blur_radius {radius}: {described}",
        content.w, content.h
    );

    let horizontal = probe.pyramid_from_sizes(sizes.clone()).expect("pyramid alloc");
    let (across, _) = probe
        .sweep_pyramid(horizontal, SHELL_OFFSET)
        .expect("horizontal sweep");
    let vertical = probe
        .pyramid_from_sizes(transpose(&sizes))
        .expect("pyramid alloc");
    let (down, _) = probe
        .sweep_pyramid(vertical, SHELL_OFFSET)
        .expect("vertical sweep");

    // What the same depth would give with nothing rounded, for reference.
    let (ideal, _) = probe
        .sweep(512, sizes.len() - 1, SHELL_OFFSET)
        .expect("reference sweep");

    println!(
        "sigma across {:.3}, sigma down {:.3}, unrounded reference {:.3}",
        across.sigma_rms, down.sigma_rms, ideal.sigma_rms
    );

    assert!(
        across.sigma_rms > down.sigma_rms * 2.0,
        "the quantum floor must show up as anisotropy: across {} vs down {}",
        across.sigma_rms,
        down.sigma_rms
    );
    assert!(
        down.sigma_rms < ideal.sigma_rms * 0.5,
        "the short axis must fall far short of the depth it was allocated for: \
         down {} vs unrounded {}",
        down.sigma_rms,
        ideal.sigma_rms
    );
}
