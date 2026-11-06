use std::ops::Mul;

/// The electro-optical transfer function of a draw source that is not encoded
/// sRGB. The values are the shader's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SourceTransfer {
    /// IEC 61966-2-1, for wide-gamut content that is sRGB in every other way.
    Srgb = 1,
    /// A pure power of 2.2.
    Gamma22 = 2,
    /// BT.1886, taken as a pure power of 2.4.
    Bt1886 = 3,
    /// SMPTE ST 2084, absolute up to 10000 cd/m².
    Pq = 4,
    /// BT.2100 hybrid log-gamma, on a 1000 cd/m² display.
    Hlg = 5,
}

/// The primaries of a draw source. The values are the shader's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SourcePrimaries {
    /// BT.709, the primaries of sRGB.
    Bt709 = 0,
    /// BT.2020 and BT.2100.
    Bt2020 = 1,
    /// Display P3.
    DisplayP3 = 2,
}

/// How a client texture is encoded, when that is not what the renderer's encoded
/// sRGB blend expects. A draw under this decodes each texel, maps it into BT.709
/// and the SDR range, and encodes it as sRGB before it blends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceColor {
    /// The source's transfer function.
    pub transfer: SourceTransfer,
    /// The source's primaries.
    pub primaries: SourcePrimaries,
    /// The cd/m² that composition shows as its white.
    pub reference_luminance: f32,
    /// The brightest cd/m² the source claims, which tone mapping brings to white.
    pub peak_luminance: f32,
}

impl SourceColor {
    const TRANSFER_SHIFT: u32 = 16;
    const PRIMARIES_SHIFT: u32 = 20;

    /// The bits this source sets in the shader's flag word.
    pub(crate) fn shader_flags(self) -> u32 {
        ((self.transfer as u32) << Self::TRANSFER_SHIFT) | ((self.primaries as u32) << Self::PRIMARIES_SHIFT)
    }

    /// The source's peak relative to its reference white, never below it.
    pub(crate) fn relative_peak(self) -> f32 {
        (self.peak_luminance / self.reference_luminance.max(1.0)).max(1.0)
    }
}

/// A four-component color representing pre-multiplied RGBA color values
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Color32F([f32; 4]);

impl Color32F {
    /// Initialize a new [`Color32F`]
    #[inline]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self([r, g, b, a])
    }
}

impl Color32F {
    /// Transparent color
    pub const TRANSPARENT: Color32F = Color32F::new(0.0, 0.0, 0.0, 0.0);

    /// Solid black color
    pub const BLACK: Color32F = Color32F::new(0f32, 0f32, 0f32, 1f32);
}

impl Color32F {
    /// Red color component
    #[inline]
    pub fn r(&self) -> f32 {
        self.0[0]
    }

    /// Green color component
    #[inline]
    pub fn g(&self) -> f32 {
        self.0[1]
    }

    /// Blue color component
    #[inline]
    pub fn b(&self) -> f32 {
        self.0[2]
    }

    /// Alpha color component
    #[inline]
    pub fn a(&self) -> f32 {
        self.0[3]
    }

    /// Color components
    #[inline]
    pub fn components(self) -> [f32; 4] {
        self.0
    }
}

impl Color32F {
    /// Test if the color represents a opaque color
    #[inline]
    pub fn is_opaque(&self) -> bool {
        self.a() == 1f32
    }
}

impl From<[f32; 4]> for Color32F {
    #[inline]
    fn from(value: [f32; 4]) -> Self {
        Self(value)
    }
}

impl Mul<f32> for Color32F {
    type Output = Color32F;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.r() * rhs, self.g() * rhs, self.b() * rhs, self.a() * rhs)
    }
}
