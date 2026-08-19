//! Format conversions between Vulkan and DRM formats.

/// Macro to generate format conversions between Vulkan and FourCC format codes.
///
/// Any entry in this table may have attributes associated with a conversion. This is needed for `PACK` Vulkan
/// formats which may only have an alternative given a specific host endian.
///
/// See the module documentation for usage details.
macro_rules! vk_format_table {
    (
        $(
            // This meta specifier is used for format conversions for PACK formats.
            $(#[$conv_meta:meta])*
            $fourcc: ident => $vk: ident
        ),* $(,)?
    ) => {
        /// Converts a FourCC format code to a Vulkan format code.
        ///
        /// This will return [`None`] if the format is not known.
        ///
        /// These format conversions will return all known FourCC and Vulkan format conversions. However a
        /// Vulkan implementation may not support some Vulkan format. One notable example of this are the
        /// formats introduced in `VK_EXT_4444_formats`. The corresponding FourCC codes will return the
        /// formats from `VK_EXT_4444_formats`, but the caller is responsible for testing that a Vulkan device
        /// supports these formats.
        pub const fn get_vk_format(fourcc: $crate::backend::allocator::Fourcc) -> Option<ash::vk::Format> {
            // FIXME: Use reexport for ash::vk::Format
            match fourcc {
                $(
                    $(#[$conv_meta])*
                    $crate::backend::allocator::Fourcc::$fourcc => Some(ash::vk::Format::$vk),
                )*

                _ => None,
            }
        }

        /// Returns all the known format conversions.
        ///
        /// The list contains FourCC format codes that may be converted using [`get_vk_format`].
        pub const fn known_formats() -> &'static [$crate::backend::allocator::Fourcc] {
            &[
                $(
                    $crate::backend::allocator::Fourcc::$fourcc
                ),*
            ]
        }
    };
}

// Vulkan classifies formats by both channel sizes and colorspace. FourCC format codes do not classify formats
// based on colorspace.
//
// Buffer STORAGE stays encoded UNORM pixels: this table is what every image, copy, blit and DRM fourcc keys
// off, and none of those bytes change meaning. Wayland specifies alpha-bearing buffers as premultiplied in
// electrical values, so a sampled view must never linearize them behind the shader's back — the shader owns
// that conversion, because it has to unpremultiply first. See `get_vk_srgb_format` for the one place the
// _SRGB sibling is used: colour attachment views, where the hardware decodes the destination and re-encodes
// the blended result so compositing arithmetic happens in linear light.
vk_format_table! {
    Argb8888 => B8G8R8A8_UNORM,
    Xrgb8888 => B8G8R8A8_UNORM,

    Abgr8888 => R8G8B8A8_UNORM,
    Xbgr8888 => R8G8B8A8_UNORM,

    // PACK32 formats are equivalent to u32 instead of [u8; 4] and thus depend their layout depends the host
    // endian.
    #[cfg(target_endian = "little")]
    Rgba8888 => A8B8G8R8_UNORM_PACK32,
    #[cfg(target_endian = "little")]
    Rgbx8888 => A8B8G8R8_UNORM_PACK32,

    #[cfg(target_endian = "little")]
    Argb2101010 => A2R10G10B10_UNORM_PACK32,
    #[cfg(target_endian = "little")]
    Xrgb2101010 => A2R10G10B10_UNORM_PACK32,

    #[cfg(target_endian = "little")]
    Abgr2101010 => A2B10G10R10_UNORM_PACK32,
    #[cfg(target_endian = "little")]
    Xbgr2101010 => A2B10G10R10_UNORM_PACK32,
}

/// Returns the sRGB-transfer sibling of an encoded UNORM colour format.
///
/// The compositor blends in linear light. A colour attachment viewed through this
/// sibling makes the hardware decode the destination before blending and re-encode
/// the result on store, so the blend arithmetic is linear while the stored bytes
/// stay exactly as sRGB-encoded as they are today. The image itself keeps the
/// [`get_vk_format`] format; only the attachment *view* uses this one, which is why
/// images that are both sampled and rendered are created `MUTABLE_FORMAT` with both
/// formats in their view-format list.
///
/// Returns [`None`] for formats with no `_SRGB` sibling — the 2101010 family has
/// none, so those targets keep blending in gamma space.
pub const fn get_vk_srgb_format(format: ash::vk::Format) -> Option<ash::vk::Format> {
    match format {
        ash::vk::Format::B8G8R8A8_UNORM => Some(ash::vk::Format::B8G8R8A8_SRGB),
        ash::vk::Format::R8G8B8A8_UNORM => Some(ash::vk::Format::R8G8B8A8_SRGB),
        ash::vk::Format::A8B8G8R8_UNORM_PACK32 => Some(ash::vk::Format::A8B8G8R8_SRGB_PACK32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use ash::vk;

    use super::{get_vk_format, get_vk_srgb_format};
    use crate::backend::allocator::Fourcc;

    #[test]
    fn srgb_siblings_exist_for_every_eight_bit_storage_format() {
        for fourcc in [
            Fourcc::Argb8888,
            Fourcc::Xrgb8888,
            Fourcc::Abgr8888,
            Fourcc::Xbgr8888,
            Fourcc::Rgba8888,
            Fourcc::Rgbx8888,
        ] {
            let storage = get_vk_format(fourcc).expect("8-bit format is mapped");
            assert!(
                get_vk_srgb_format(storage).is_some(),
                "{fourcc:?} maps to {storage:?}, which has no _SRGB sibling to render through"
            );
        }
    }

    #[test]
    fn ten_bit_formats_have_no_srgb_sibling_and_stay_gamma_blended() {
        let storage = get_vk_format(Fourcc::Argb2101010).expect("10-bit format is mapped");
        assert_eq!(get_vk_srgb_format(storage), None);
    }

    #[test]
    fn eight_bit_color_formats_are_encoded_unorm() {
        assert_eq!(get_vk_format(Fourcc::Argb8888), Some(vk::Format::B8G8R8A8_UNORM));
        assert_eq!(get_vk_format(Fourcc::Xrgb8888), Some(vk::Format::B8G8R8A8_UNORM));
        assert_eq!(get_vk_format(Fourcc::Abgr8888), Some(vk::Format::R8G8B8A8_UNORM));
        assert_eq!(get_vk_format(Fourcc::Xbgr8888), Some(vk::Format::R8G8B8A8_UNORM));
    }
}
