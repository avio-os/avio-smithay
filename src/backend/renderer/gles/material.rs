//! Typed material packets used by the GLES composition lowering.
use super::{GlesTexture, Texture};
use crate::utils::{Buffer as BufferCoord, Point, Size};

#[derive(Debug, Clone, Copy)]
pub(crate) struct GlesKawaseColorTransform {
    pub contrast: f32,
    pub brightness: f32,
    pub saturation: f32,
}
impl GlesKawaseColorTransform {
    const IDENTITY: Self = Self {
        contrast: 1.0,
        brightness: 1.0,
        saturation: 1.0,
    };
}

/// One dual-Kawase pyramid step: renders `source` into `destination` with the
/// down- or upsample kernel.
///
/// Like the shared Vulkan equations, every pass filters and applies CSS
/// transforms directly to encoded premultiplied channel values.
#[derive(Debug, Clone)]
pub struct GlesKawasePass {
    pub(crate) source: GlesTexture,
    pub(crate) destination: GlesTexture,
    pub(crate) upsample: bool,
    pub(crate) offset: f32,
    pub(crate) transform: GlesKawaseColorTransform,
    pub(crate) source_origin: Point<i32, BufferCoord>,
    pub(crate) source_extent: Size<i32, BufferCoord>,
    pub(crate) destination_extent: Size<i32, BufferCoord>,
}

impl GlesKawasePass {
    /// Describes a kawase pass between two renderer textures.
    pub fn new(source: &GlesTexture, destination: &GlesTexture, upsample: bool, offset: f32) -> Self {
        Self {
            source: source.clone(),
            destination: destination.clone(),
            upsample,
            offset,
            transform: GlesKawaseColorTransform::IDENTITY,
            source_origin: (0, 0).into(),
            source_extent: source.size(),
            destination_extent: destination.size(),
        }
    }

    /// Restrict this pass to initialized, origin-aligned regions of reusable
    /// backing images. Sampling clamps at the source region's texel centers;
    /// pixels beyond the destination extent are neither read nor written.
    /// Extents are validated against the image capacities before recording.
    pub fn with_extents(
        mut self,
        source: Size<i32, BufferCoord>,
        destination: Size<i32, BufferCoord>,
    ) -> Self {
        self.source_extent = source;
        self.destination_extent = destination;
        self
    }

    pub(crate) fn with_source_origin(mut self, origin: Point<i32, BufferCoord>) -> Self {
        self.source_origin = origin;
        self
    }

    /// Applies a post-blur saturation transform in the destination colour
    /// space. Material graphs should set this on their final pass only.
    ///
    /// Saturation runs last, after [`with_contrast`](Self::with_contrast) and
    /// [`with_brightness`](Self::with_brightness), matching the CSS
    /// `backdrop-filter: blur() contrast() brightness() saturate()` order
    /// regardless of the order the builders are called in.
    pub fn with_saturation(mut self, saturation: f32) -> Self {
        self.transform.saturation = saturation.clamp(0.0, 4.0);
        self
    }

    /// Applies a post-blur CSS `contrast()` transform in the destination
    /// colour space: `(x - 0.5) * contrast + 0.5`, clamped. It runs first
    /// after the blur, before brightness and saturation. `1.0` is the identity.
    pub fn with_contrast(mut self, contrast: f32) -> Self {
        self.transform.contrast = contrast.clamp(0.0, 4.0);
        self
    }

    /// Applies a post-blur CSS `brightness()` multiply in the destination
    /// colour space, clamped. It runs after contrast and before saturation.
    /// `1.0` is the identity.
    pub fn with_brightness(mut self, brightness: f32) -> Self {
        self.transform.brightness = brightness.clamp(0.0, 4.0);
        self
    }
}

/// Final encoded-sRGB eight-tap upsample fused into a clipped material draw.
/// The source remains the immutable half-resolution prefix captured before all cards.
#[derive(Debug, Clone, Copy)]
pub struct GlesKawaseOutput {
    pub(crate) extent: Size<i32, BufferCoord>,
    pub(crate) upsample: bool,
    pub(crate) offset: f32,
    pub(crate) transform: GlesKawaseColorTransform,
}
impl GlesKawaseOutput {
    /// Create a fused final pass over the initialized origin-aligned source region.
    pub fn new(extent: Size<i32, BufferCoord>, offset: f32) -> Self {
        Self {
            extent,
            offset,
            upsample: true,
            transform: GlesKawaseColorTransform::IDENTITY,
        }
    }
    /// Fuse the original five-tap zero-depth blur over one frozen full-size prefix.
    pub fn new_downsample(extent: Size<i32, BufferCoord>, offset: f32) -> Self {
        Self {
            extent,
            offset,
            upsample: false,
            transform: GlesKawaseColorTransform::IDENTITY,
        }
    }
    /// Set the CSS filter-list transform, in contrast/brightness/saturation order.
    pub fn with_filter(mut self, contrast: f32, brightness: f32, saturation: f32) -> Self {
        self.transform = GlesKawaseColorTransform {
            contrast: contrast.clamp(0.0, 4.0),
            brightness: brightness.clamp(0.0, 4.0),
            saturation: saturation.clamp(0.0, 4.0),
        };
        self
    }
}

/// Electrical-sRGB tint followed by three white SourceOver body lights.
///
/// The descriptor fixes their normalized geometry: radial centre `(0.5, 0)`,
/// elliptical radii `(0.58, 1.3)` with zero at radius `0.68`; bottom fade over
/// `0.22` of height; top fade over `0.26`. The three independent opacities are
/// authored values, not inferred from a texture, view identity, or clip shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlesMaterialTint {
    /// Straight electrical-sRGB RGBA colour.
    pub color: [f32; 4],
    /// Radial, bottom, and top white-light opacities, in SourceOver order.
    pub body_lights: [f32; 3],
}
impl GlesMaterialTint {
    /// Plain tint retains the existing no-body-light material behavior.
    pub const fn new(color: [f32; 4]) -> Self {
        Self {
            color,
            body_lights: [0.0; 3],
        }
    }
    /// Attach the explicit fixed-geometry lights without another image.
    pub const fn with_body_lights(mut self, opacities: [f32; 3]) -> Self {
        self.body_lights = opacities;
        self
    }
    pub(crate) fn valid(self) -> bool {
        self.color
            .into_iter()
            .chain(self.body_lights)
            .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_light_packet_rejects_nonfinite_and_nonunit_authored_values() {
        let plain = GlesMaterialTint::new([0.2, 0.3, 0.4, 0.5]);
        assert!(plain.valid());
        assert_eq!(plain.body_lights, [0.0; 3]);
        for invalid in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
            assert!(!plain.with_body_lights([0.0, invalid, 0.0]).valid());
        }
        assert!(plain
            .with_body_lights([41.0 / 255.0, 56.0 / 255.0, 128.0 / 255.0])
            .valid());
    }
}
