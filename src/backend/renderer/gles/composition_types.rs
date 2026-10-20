// GLES lowering for the same composition packets and shader equations as Vulkan.
// Native programs, framebuffers and damage storage are admitted on cold turns.
use super::*;
use crate::backend::renderer::{BottomEdgeClip, RoundedClip, TextureRenderEffect, TextureRenderEffectKind};
use crate::utils::Point;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub(crate) enum TextureTransform {
    #[default]
    Normal = 0,
    Rotate90 = 1,
    Rotate180 = 2,
    Rotate270 = 3,
    Flipped = 4,
    Flipped90 = 5,
    Flipped180 = 6,
    Flipped270 = 7,
}

impl From<crate::utils::Transform> for TextureTransform {
    fn from(value: crate::utils::Transform) -> Self {
        match value {
            crate::utils::Transform::Normal => TextureTransform::Normal,
            crate::utils::Transform::_90 => TextureTransform::Rotate90,
            crate::utils::Transform::_180 => TextureTransform::Rotate180,
            crate::utils::Transform::_270 => TextureTransform::Rotate270,
            crate::utils::Transform::Flipped => TextureTransform::Flipped,
            crate::utils::Transform::Flipped90 => TextureTransform::Flipped90,
            crate::utils::Transform::Flipped180 => TextureTransform::Flipped180,
            crate::utils::Transform::Flipped270 => TextureTransform::Flipped270,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct TexturePushConstants {
    pub(crate) alpha: f32,
    pub(crate) transform: u32,
    pub(crate) y_inverted: u32,
    pub(crate) rounded_clip_flags: u32,
    pub(crate) src_offset: [f32; 2],
    pub(crate) src_scale: [f32; 2],
    pub(crate) clip_rect: [f32; 4],
    pub(crate) clip_params: [f32; 4],
    pub(crate) effect: [f32; 4],
    pub(crate) effect_params: [f32; 4],
    /// How the sampled texels relate to linear light, and therefore which conversion
    /// the shader applies before the blend. See [`GlesTextureEncoding`].
    pub(crate) source_encoding: u32,
    /// Physical pixels per authored logical pixel for parametric clips.
    pub(crate) clip_scale: f32,
    // vec2 spelling in GLSL avoids vec4 alignment padding past the minimum
    // 128-byte Vulkan push-constant allowance.
    pub(crate) outer_origin: [f32; 2],
    pub(crate) outer_size: [f32; 2],
    pub(crate) outer_radius: f32,
    _std140_padding: f32,
}

const BOTTOM_EDGE_CLIP_FLAG: u32 = 1 << 31;
const CLIP_TRANSFORM_SHIFT: u32 = 8;

/// Shader-side spelling of [`GlesTextureEncoding`]. Kept next to the push-constant struct so
/// the two stay in step; `texture.frag` reads the same numbering.
pub(crate) const SOURCE_ENCODING_ELECTRICAL_PREMULTIPLIED: u32 = 0;
pub(crate) const SOURCE_ENCODING_LINEAR_PREMULTIPLIED: u32 = 1;
/// The target blends in gamma space (no `_SRGB` sibling), so no conversion happens.
pub(crate) const SOURCE_ENCODING_PASSTHROUGH: u32 = 2;

impl From<GlesTextureEncoding> for u32 {
    fn from(encoding: GlesTextureEncoding) -> Self {
        match encoding {
            GlesTextureEncoding::ElectricalPremultiplied => SOURCE_ENCODING_ELECTRICAL_PREMULTIPLIED,
            GlesTextureEncoding::LinearPremultiplied => SOURCE_ENCODING_LINEAR_PREMULTIPLIED,
        }
    }
}

impl Default for TexturePushConstants {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            transform: TextureTransform::Normal as u32,
            y_inverted: 0,
            rounded_clip_flags: 0,
            src_offset: [0.0, 0.0],
            src_scale: [1.0, 1.0],
            clip_rect: [0.0, 0.0, 0.0, 0.0],
            clip_params: [0.0, 2.0, 0.5, 0.0],
            effect: [TextureRenderEffectKind::None as u32 as f32, 0.0, 0.5, 0.5],
            effect_params: [0.0, 0.0, 0.0, 0.0],
            source_encoding: SOURCE_ENCODING_PASSTHROUGH,
            clip_scale: 1.0,
            outer_origin: [0.0; 2],
            outer_size: [0.0; 2],
            outer_radius: -1.0,
            _std140_padding: 0.0,
        }
    }
}

impl TexturePushConstants {
    pub(crate) fn new(alpha: f32, transform: TextureTransform, y_inverted: bool) -> Self {
        Self {
            alpha,
            transform: transform as u32,
            y_inverted: u32::from(y_inverted),
            ..Self::default()
        }
    }

    /// Declares how the sampled texels relate to linear light.
    ///
    /// `linear_blending` is a property of the bound target: when it is false the pass
    /// blends in gamma space and texels must reach the blend exactly as stored.
    pub(crate) fn with_source_encoding(
        mut self,
        encoding: GlesTextureEncoding,
        linear_blending: bool,
    ) -> Self {
        self.source_encoding = if linear_blending {
            encoding.into()
        } else {
            SOURCE_ENCODING_PASSTHROUGH
        };
        self
    }

    pub(crate) fn with_src_rect(mut self, offset: [f32; 2], scale: [f32; 2]) -> Self {
        self.src_offset = offset;
        self.src_scale = scale;
        self
    }

    pub(crate) fn with_rounded_clip(mut self, flags: u32, rect: [f32; 4], params: [f32; 4]) -> Self {
        self.rounded_clip_flags = flags;
        self.clip_rect = rect;
        self.clip_params = params;
        self
    }

    pub(crate) fn with_outer_rounded_clip(mut self, origin: [f32; 2], size: [f32; 2], radius: f32) -> Self {
        self.outer_origin = origin;
        self.outer_size = size;
        self.outer_radius = radius;
        self
    }

    pub(crate) fn with_bottom_edge_clip(
        mut self,
        transform: TextureTransform,
        rect: [f32; 4],
        params: [f32; 4],
        geometry_scale: f32,
    ) -> Self {
        self.rounded_clip_flags = BOTTOM_EDGE_CLIP_FLAG | ((transform as u32) << CLIP_TRANSFORM_SHIFT);
        self.clip_rect = rect;
        self.clip_params = params;
        self.clip_scale = geometry_scale;
        self
    }

    pub(crate) fn with_owner_resolve(
        mut self,
        extent: crate::utils::Size<i32, crate::utils::Physical>,
        capacity: crate::utils::Size<i32, crate::utils::Buffer>,
    ) -> Self {
        self.effect = [
            7.0,
            extent.w as f32 / capacity.w as f32,
            extent.h as f32 / capacity.h as f32,
            0.0,
        ];
        self
    }

    pub(crate) fn with_material_tone(mut self, tint: GlesMaterialTint) -> Self {
        self.effect = [5.0, tint.color[0], tint.color[1], tint.color[2]];
        self.effect_params = [
            tint.color[3],
            tint.body_lights[0],
            tint.body_lights[1],
            tint.body_lights[2],
        ];
        self
    }

    pub(crate) fn with_kawase_output(
        mut self,
        output: GlesKawaseOutput,
        capacity: crate::utils::Size<i32, crate::utils::Buffer>,
    ) -> Self {
        self.effect = [
            if output.upsample { 4.0 } else { 6.0 },
            output.offset,
            output.transform.saturation,
            output.transform.contrast,
        ];
        self.effect_params = [
            output.transform.brightness,
            output.extent.w as f32 / capacity.w as f32,
            output.extent.h as f32 / capacity.h as f32,
            0.0,
        ];
        self
    }

    pub(crate) fn with_effect(mut self, effect: TextureRenderEffect) -> Self {
        self.effect = [
            effect.kind as u32 as f32,
            effect.progress.clamp(0.0, 1.0),
            effect.anchor[0],
            effect.anchor[1],
        ];
        self.effect_params = effect.params;
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum AnalyticClip {
    Rounded(RoundedClip),
    BottomEdge(BottomEdgeClip),
}

#[derive(Debug, Clone, Copy)]
enum TransformedAnalyticClip {
    Rounded(RoundedClip),
    BottomEdge {
        clip: BottomEdgeClip,
        transform: TextureTransform,
    },
}

fn analytic_clip_rect_push_constant(rect: Rectangle<f64, Physical>) -> [f32; 4] {
    [
        rect.loc.x as f32,
        rect.loc.y as f32,
        rect.size.w as f32,
        rect.size.h as f32,
    ]
}

fn transform_analytic_clip(
    transform: Transform,
    frame_size: Size<i32, Physical>,
    clip: AnalyticClip,
) -> TransformedAnalyticClip {
    match clip {
        AnalyticClip::Rounded(clip) => {
            TransformedAnalyticClip::Rounded(transform_rounded_clip(transform, frame_size, clip))
        }
        AnalyticClip::BottomEdge(clip) => {
            let frame_size = frame_size.to_f64();
            TransformedAnalyticClip::BottomEdge {
                clip: BottomEdgeClip {
                    rect: transform.transform_rect_in(clip.rect, &frame_size),
                    ..clip
                },
                transform: TextureTransform::from(transform),
            }
        }
    }
}

fn transform_rounded_clip(
    transform: Transform,
    frame_size: Size<i32, Physical>,
    clip: RoundedClip,
) -> RoundedClip {
    let frame_size = frame_size.to_f64();
    let rect = transform.transform_rect_in(clip.rect, &frame_size);
    RoundedClip {
        rect,
        corner_mask: transform_corner_mask(transform, frame_size, clip.rect, rect, clip.corner_mask),
        ..clip
    }
}

fn transform_corner_mask(
    transform: Transform,
    frame_size: Size<f64, Physical>,
    source_rect: Rectangle<f64, Physical>,
    transformed_rect: Rectangle<f64, Physical>,
    source_mask: u32,
) -> u32 {
    let corners = [
        (
            RoundedClip::TOP_LEFT,
            Point::from((source_rect.loc.x, source_rect.loc.y)),
        ),
        (
            RoundedClip::TOP_RIGHT,
            Point::from((source_rect.loc.x + source_rect.size.w, source_rect.loc.y)),
        ),
        (
            RoundedClip::BOTTOM_RIGHT,
            Point::from((
                source_rect.loc.x + source_rect.size.w,
                source_rect.loc.y + source_rect.size.h,
            )),
        ),
        (
            RoundedClip::BOTTOM_LEFT,
            Point::from((source_rect.loc.x, source_rect.loc.y + source_rect.size.h)),
        ),
    ];

    corners
        .into_iter()
        .filter(|(flag, _)| source_mask & *flag != 0)
        .map(|(_, point)| {
            transformed_corner_flag(transform.transform_point_in(point, &frame_size), transformed_rect)
        })
        .fold(0, |mask, flag| mask | flag)
}

fn transformed_corner_flag(point: Point<f64, Physical>, rect: Rectangle<f64, Physical>) -> u32 {
    let center_x = rect.loc.x + rect.size.w * 0.5;
    let center_y = rect.loc.y + rect.size.h * 0.5;
    match (point.x <= center_x, point.y <= center_y) {
        (true, true) => RoundedClip::TOP_LEFT,
        (false, true) => RoundedClip::TOP_RIGHT,
        (false, false) => RoundedClip::BOTTOM_RIGHT,
        (true, false) => RoundedClip::BOTTOM_LEFT,
    }
}

fn combine_image_transform(src_transform: Transform, output_transform: Transform) -> Transform {
    let src_transform = src_transform.invert();

    match (src_transform, output_transform) {
        (Transform::Normal, output_transform) => output_transform,

        (Transform::_90, Transform::Normal) => Transform::_270,
        (Transform::_90, Transform::_90) => Transform::Normal,
        (Transform::_90, Transform::_180) => Transform::_90,
        (Transform::_90, Transform::_270) => Transform::_180,
        (Transform::_90, Transform::Flipped) => Transform::Flipped90,
        (Transform::_90, Transform::Flipped90) => Transform::Flipped180,
        (Transform::_90, Transform::Flipped180) => Transform::Flipped270,
        (Transform::_90, Transform::Flipped270) => Transform::Flipped,

        (Transform::_180, Transform::Normal) => Transform::_180,
        (Transform::_180, Transform::_90) => Transform::_270,
        (Transform::_180, Transform::_180) => Transform::Normal,
        (Transform::_180, Transform::_270) => Transform::_90,
        (Transform::_180, Transform::Flipped) => Transform::Flipped180,
        (Transform::_180, Transform::Flipped90) => Transform::Flipped270,
        (Transform::_180, Transform::Flipped180) => Transform::Flipped,
        (Transform::_180, Transform::Flipped270) => Transform::Flipped90,

        (Transform::_270, Transform::Normal) => Transform::_90,
        (Transform::_270, Transform::_90) => Transform::_180,
        (Transform::_270, Transform::_180) => Transform::_270,
        (Transform::_270, Transform::_270) => Transform::Normal,
        (Transform::_270, Transform::Flipped) => Transform::Flipped270,
        (Transform::_270, Transform::Flipped90) => Transform::Flipped,
        (Transform::_270, Transform::Flipped180) => Transform::Flipped90,
        (Transform::_270, Transform::Flipped270) => Transform::Flipped180,

        (Transform::Flipped, Transform::Normal) => Transform::Flipped,
        (Transform::Flipped, Transform::_90) => Transform::Flipped90,
        (Transform::Flipped, Transform::_180) => Transform::Flipped180,
        (Transform::Flipped, Transform::_270) => Transform::Flipped270,
        (Transform::Flipped, Transform::Flipped) => Transform::Normal,
        (Transform::Flipped, Transform::Flipped90) => Transform::_90,
        (Transform::Flipped, Transform::Flipped180) => Transform::_180,
        (Transform::Flipped, Transform::Flipped270) => Transform::_270,

        (Transform::Flipped90, Transform::Normal) => Transform::Flipped90,
        (Transform::Flipped90, Transform::_90) => Transform::Flipped180,
        (Transform::Flipped90, Transform::_180) => Transform::Flipped270,
        (Transform::Flipped90, Transform::_270) => Transform::Flipped,
        (Transform::Flipped90, Transform::Flipped) => Transform::_270,
        (Transform::Flipped90, Transform::Flipped90) => Transform::Normal,
        (Transform::Flipped90, Transform::Flipped180) => Transform::_90,
        (Transform::Flipped90, Transform::Flipped270) => Transform::_180,

        (Transform::Flipped180, Transform::Normal) => Transform::Flipped180,
        (Transform::Flipped180, Transform::_90) => Transform::Flipped270,
        (Transform::Flipped180, Transform::_180) => Transform::Flipped,
        (Transform::Flipped180, Transform::_270) => Transform::Flipped90,
        (Transform::Flipped180, Transform::Flipped) => Transform::_180,
        (Transform::Flipped180, Transform::Flipped90) => Transform::_270,
        (Transform::Flipped180, Transform::Flipped180) => Transform::Normal,
        (Transform::Flipped180, Transform::Flipped270) => Transform::_90,

        (Transform::Flipped270, Transform::Normal) => Transform::Flipped270,
        (Transform::Flipped270, Transform::_90) => Transform::Flipped,
        (Transform::Flipped270, Transform::_180) => Transform::Flipped90,
        (Transform::Flipped270, Transform::_270) => Transform::Flipped180,
        (Transform::Flipped270, Transform::Flipped) => Transform::_90,
        (Transform::Flipped270, Transform::Flipped90) => Transform::_180,
        (Transform::Flipped270, Transform::Flipped180) => Transform::_270,
        (Transform::Flipped270, Transform::Flipped270) => Transform::Normal,
    }
}
#[derive(Debug, Clone, Copy)]
pub(super) struct OwnerSampleReplay {
    frame_size: Size<i32, Physical>,
    lane: usize,
    bounds: Rectangle<i32, Physical>,
    translation: Point<i32, Physical>,
    sample: [f32; 2],
    output_read: Rectangle<i32, Physical>,
}

impl OwnerSampleReplay {
    fn new(
        frame_size: Size<i32, Physical>,
        lane: usize,
        source: Rectangle<i32, Physical>,
        output_read: Rectangle<i32, Physical>,
    ) -> Option<Self> {
        let samples = [[0.375, 0.125], [0.875, 0.375], [0.125, 0.625], [0.625, 0.875]];
        let sample = *samples.get(lane)?;
        let origin = Point::from((
            (lane % 2) as i32 * source.size.w,
            (lane / 2) as i32 * source.size.h,
        ));
        Some(Self {
            frame_size,
            lane,
            bounds: Rectangle::new(origin, source.size),
            translation: origin - source.loc,
            sample,
            output_read,
        })
    }

    fn clip_output_region(self, region: Rectangle<i32, Physical>) -> Option<Rectangle<i32, Physical>> {
        region.intersection(self.output_read)
    }

    fn shift(self, resolved_texture: bool) -> [f32; 2] {
        [
            self.translation.x as f32
                + if resolved_texture {
                    0.0
                } else {
                    0.5 - self.sample[0]
                },
            self.translation.y as f32
                + if resolved_texture {
                    0.0
                } else {
                    0.5 - self.sample[1]
                },
        ]
    }

    #[cfg(test)]
    fn damage(self, rects: Vec<Rectangle<i32, Physical>>) -> Vec<Rectangle<i32, Physical>> {
        rects
            .into_iter()
            .filter_map(|mut rect| {
                rect.loc += self.translation;
                rect.intersection(self.bounds)
            })
            .collect()
    }
}

#[cfg(test)]
mod owner_sample_tests {
    use super::*;
    #[test]
    fn each_lane_preserves_native_sample_and_exact_resolved_prefix_grid() {
        let source = Rectangle::new((17, 29).into(), (12, 10).into());
        for lane in 0..4 {
            let replay = OwnerSampleReplay::new((100, 80).into(), lane, source, source).unwrap();
            assert_eq!(replay.frame_size, (100, 80).into());
            assert_eq!(replay.lane, lane);
            assert_eq!(
                replay.clip_output_region(Rectangle::from_size((100, 80).into())),
                Some(source)
            );
            let origin = replay.bounds.loc;
            // A destination fragment centre minus the analytic viewport shift
            // is precisely its source pixel plus the canonical sample.
            let shift = replay.shift(false);
            assert_eq!(
                origin.x as f32 + 0.5 - shift[0],
                source.loc.x as f32 + replay.sample[0]
            );
            assert_eq!(
                origin.y as f32 + 0.5 - shift[1],
                source.loc.y as f32 + replay.sample[1]
            );
            let shift = replay.shift(true);
            assert_eq!(origin.x as f32 + 0.5 - shift[0], source.loc.x as f32 + 0.5);
        }
        assert!(OwnerSampleReplay::new((100, 80).into(), 4, source, source).is_none());
    }

    #[test]
    fn lane_damage_cannot_touch_an_adjacent_atlas_quadrant() {
        let replay = OwnerSampleReplay {
            frame_size: (100, 80).into(),
            lane: 3,
            bounds: Rectangle::new((12, 10).into(), (12, 10).into()),
            translation: (7, 3).into(),
            sample: [0.625, 0.875],
            output_read: Rectangle::from_size((100, 80).into()),
        };
        assert_eq!(
            replay.damage(vec![Rectangle::from_size((100, 80).into())]),
            vec![replay.bounds]
        );
        assert_eq!(replay.shift(false), [6.875, 2.625]);
        assert_eq!(replay.shift(true), [7.0, 3.0]);
    }
}
