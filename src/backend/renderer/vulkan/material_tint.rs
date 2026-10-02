//! Bounded material paint packet; no image or upload is needed for its body.

/// Electrical-sRGB tint followed by three white SourceOver body lights.
///
/// The descriptor fixes their normalized geometry: radial centre `(0.5, 0)`,
/// elliptical radii `(0.58, 1.3)` with zero at radius `0.68`; bottom fade over
/// `0.22` of height; top fade over `0.26`. The three independent opacities are
/// authored values, not inferred from a texture, view identity, or clip shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VulkanMaterialTint {
    /// Straight electrical-sRGB RGBA colour.
    pub color: [f32; 4],
    /// Radial, bottom, and top white-light opacities, in SourceOver order.
    pub body_lights: [f32; 3],
}
impl VulkanMaterialTint {
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
        let plain = VulkanMaterialTint::new([0.2, 0.3, 0.4, 0.5]);
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
