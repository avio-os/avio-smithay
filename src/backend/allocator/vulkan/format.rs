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
// The compositor's current SDR render contract keeps Wayland/DRM buffers as encoded UNORM pixels. Wayland
// specifies alpha-bearing buffers as premultiplied in electrical values, so automatic SRGB sampling would
// linearize already-premultiplied channels before the shader can repair that relationship.
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

#[cfg(test)]
mod tests {
    use ash::vk;

    use super::get_vk_format;
    use crate::backend::allocator::Fourcc;

    #[test]
    fn eight_bit_color_formats_are_encoded_unorm() {
        assert_eq!(get_vk_format(Fourcc::Argb8888), Some(vk::Format::B8G8R8A8_UNORM));
        assert_eq!(get_vk_format(Fourcc::Xrgb8888), Some(vk::Format::B8G8R8A8_UNORM));
        assert_eq!(get_vk_format(Fourcc::Abgr8888), Some(vk::Format::R8G8B8A8_UNORM));
        assert_eq!(get_vk_format(Fourcc::Xbgr8888), Some(vk::Format::R8G8B8A8_UNORM));
    }
}
