//! Connector signalling for an HDR10 output: `Colorspace`, `HDR_OUTPUT_METADATA`
//! and `max bpc`.
//!
//! These are connector properties a frame never changes, so they follow the same
//! shape as `VRR_ENABLED`: a selection is tested and recorded on the pending state,
//! and the next commit carries it with its first pixels in that encoding. A flip
//! after that writes none of them. Blobs are created and destroyed only on that
//! cold edge, which the mode-blob actor's per-frame reasoning does not cover.

use std::{collections::HashSet, sync::Arc};

use drm::control::{connector, property, Device as ControlDevice};

use super::super::device::atomic::PropMapping;
use super::super::device::DrmDeviceInternal;
use super::super::error::{AccessError, Error};
use crate::utils::DevPath;

const EOTF_SMPTE_ST2084: u8 = 2;
const STATIC_METADATA_TYPE1: u8 = 0;
/// `DRM_MODE_COLORIMETRY_DEFAULT`. The kernel's enum value, on every driver.
const COLORSPACE_DEFAULT: u64 = 0;
/// A 10-bit signal, or PQ quantizes visibly.
const MIN_HDR_BPC: u64 = 10;

/// CTA-861.3 Static Metadata Type 1, in the units of the InfoFrame: chromaticity
/// in 0.00002 steps, maximum levels in cd/m², minimum mastering in 0.0001 cd/m².
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdrOutputMetadata {
    /// Mastering display primaries, red, green then blue, as (x, y).
    pub primaries: [(u16, u16); 3],
    /// Mastering display white point as (x, y).
    pub white: (u16, u16),
    /// Maximum mastering display luminance.
    pub max_mastering_luminance: u16,
    /// Minimum mastering display luminance.
    pub min_mastering_luminance: u16,
    /// Maximum content light level, zero when unknown.
    pub max_content_light_level: u16,
    /// Maximum frame-average light level, zero when unknown.
    pub max_frame_average_light_level: u16,
}

/// `struct hdr_output_metadata` from `drm_mode.h`.
#[repr(C)]
struct KernelHdrOutputMetadata {
    metadata_type: u32,
    eotf: u8,
    descriptor_type: u8,
    primaries: [[u16; 2]; 3],
    white: [u16; 2],
    max_mastering: u16,
    min_mastering: u16,
    max_cll: u16,
    max_fall: u16,
}

const _: () = assert!(std::mem::size_of::<KernelHdrOutputMetadata>() == 32);

impl HdrOutputMetadata {
    fn to_kernel(self) -> KernelHdrOutputMetadata {
        KernelHdrOutputMetadata {
            metadata_type: u32::from(STATIC_METADATA_TYPE1),
            eotf: EOTF_SMPTE_ST2084,
            descriptor_type: STATIC_METADATA_TYPE1,
            primaries: self.primaries.map(|(x, y)| [x, y]),
            white: [self.white.0, self.white.1],
            max_mastering: self.max_mastering_luminance,
            min_mastering: self.min_mastering_luminance,
            max_cll: self.max_content_light_level,
            max_fall: self.max_frame_average_light_level,
        }
    }
}

/// How a selection reached the pending state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdrOutputSelection {
    /// The kernel took it without a modeset, so the panel does not blank.
    pub seamless: bool,
}

/// One created metadata blob. Dropping the last holder destroys it, always on
/// the cold edge that replaced it.
pub(super) struct HdrBlob {
    fd: Arc<DrmDeviceInternal>,
    id: u64,
}

impl std::fmt::Debug for HdrBlob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("HdrBlob").field(&self.id).finish()
    }
}

impl Drop for HdrBlob {
    fn drop(&mut self) {
        if let Err(error) = self.fd.destroy_property_blob(self.id) {
            tracing::warn!(blob = self.id, %error, "Failed to destroy an HDR metadata blob");
        }
    }
}

/// What a connector's color signalling is, as an atomic state records it.
#[derive(Debug, Clone, Default)]
pub(super) enum ConnectorColor {
    /// Default colorimetry and no metadata.
    #[default]
    Sdr,
    /// A previous owner of the connector left it signalling something else.
    /// Only a state read from KMS holds this, so the next commit corrects it.
    Inherited,
    Hdr {
        metadata: HdrOutputMetadata,
        blob: Arc<HdrBlob>,
        colorspace: u64,
        /// Each connector's `max bpc` to go back to, when it has the property.
        restore_max_bpc: Vec<(connector::Handle, u64)>,
    },
}

impl PartialEq for ConnectorColor {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Sdr, Self::Sdr) | (Self::Inherited, Self::Inherited) => true,
            (Self::Hdr { metadata: a, .. }, Self::Hdr { metadata: b, .. }) => a == b,
            _ => false,
        }
    }
}

impl ConnectorColor {
    pub(super) fn metadata(&self) -> Option<HdrOutputMetadata> {
        match self {
            Self::Hdr { metadata, .. } => Some(*metadata),
            _ => None,
        }
    }

    /// What a state in HDR goes back to, so a new selection can carry it forward.
    pub(super) fn restore_max_bpc(&self) -> Option<&[(connector::Handle, u64)]> {
        match self {
            Self::Hdr { restore_max_bpc, .. } => Some(restore_max_bpc),
            _ => None,
        }
    }
}

/// The one property write set a color change needs, for one connector.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct ConnectorColorWrite {
    pub(super) connector: connector::Handle,
    pub(super) colorspace: u64,
    pub(super) metadata_blob: u64,
    pub(super) max_bpc: Option<u64>,
}

/// What `from` to `to` writes. Nothing when they agree, which is every ordinary
/// commit, so SDR output sends exactly the request it always sent.
pub(super) fn connector_color_writes<'a>(
    from: &ConnectorColor,
    to: &ConnectorColor,
    connectors: impl IntoIterator<Item = &'a connector::Handle>,
) -> Vec<ConnectorColorWrite> {
    if from == to {
        return Vec::new();
    }
    connectors
        .into_iter()
        .map(|connector| match to {
            ConnectorColor::Hdr { blob, colorspace, .. } => ConnectorColorWrite {
                connector: *connector,
                colorspace: *colorspace,
                metadata_blob: blob.id,
                max_bpc: Some(MIN_HDR_BPC),
            },
            ConnectorColor::Sdr | ConnectorColor::Inherited => ConnectorColorWrite {
                connector: *connector,
                colorspace: COLORSPACE_DEFAULT,
                metadata_blob: 0,
                max_bpc: match from {
                    ConnectorColor::Hdr { restore_max_bpc, .. } => restore_max_bpc
                        .iter()
                        .find(|(known, _)| known == connector)
                        .map(|(_, value)| *value),
                    _ => None,
                },
            },
        })
        .collect()
}

fn access_error(fd: &DrmDeviceInternal, errmsg: &'static str, source: std::io::Error) -> Error {
    Error::Access(AccessError {
        errmsg,
        dev: fd.dev_path(),
        source,
    })
}

/// Whether KMS on this connector can signal HDR10, asked by property name. Reads
/// the device and needs no surface, so an output owner can answer it cold before
/// any worker holds the output. This says nothing about the panel.
pub fn connector_can_signal_hdr10<A: ControlDevice>(
    device: &A,
    connector: connector::Handle,
) -> std::io::Result<bool> {
    let values = device.get_properties(connector)?;
    let (ids, _) = values.as_props_and_values();
    let mut metadata = false;
    let mut bt2020 = false;
    let mut bpc = true;
    for id in ids {
        let info = device.get_property(*id)?;
        match info.name().to_bytes() {
            b"HDR_OUTPUT_METADATA" => metadata = true,
            b"Colorspace" => {
                if let property::ValueType::Enum(values) = info.value_type() {
                    let (_, names) = values.values();
                    bt2020 = names.iter().any(|entry| entry.name().to_bytes() == b"BT2020_RGB");
                }
            }
            b"max bpc" => {
                if let property::ValueType::UnsignedRange(_, max) = info.value_type() {
                    bpc = max >= MIN_HDR_BPC;
                }
            }
            _ => {}
        }
    }
    Ok(metadata && bt2020 && bpc)
}

/// The enum value of `BT2020_RGB`, the one colorimetry HDR10 pixels need.
fn bt2020_rgb_value(
    fd: &DrmDeviceInternal,
    mapping: &PropMapping,
    connector: connector::Handle,
) -> Result<Option<u64>, Error> {
    let Ok(prop) = mapping.conn_prop_handle(connector, "Colorspace") else {
        return Ok(None);
    };
    let info = fd
        .get_property(prop)
        .map_err(|source| access_error(fd, "Error querying Colorspace", source))?;
    let property::ValueType::Enum(values) = info.value_type() else {
        return Ok(None);
    };
    let (_, names) = values.values();
    Ok(names
        .iter()
        .find(|entry| entry.name().to_bytes() == b"BT2020_RGB")
        .map(|entry| entry.value()))
}

/// Whether this connector's `max bpc` can reach 10, when it has the property.
fn max_bpc_reaches_hdr(
    fd: &DrmDeviceInternal,
    mapping: &PropMapping,
    connector: connector::Handle,
) -> Result<bool, Error> {
    let Ok(prop) = mapping.conn_prop_handle(connector, "max bpc") else {
        return Ok(true);
    };
    let info = fd
        .get_property(prop)
        .map_err(|source| access_error(fd, "Error querying max bpc", source))?;
    Ok(match info.value_type() {
        property::ValueType::UnsignedRange(_, max) => max >= MIN_HDR_BPC,
        _ => true,
    })
}

/// Whether KMS on this connector can signal HDR10 at all. This says nothing
/// about the panel, whose EDID is the caller's to read, or about the mode.
pub(super) fn connector_can_signal_hdr(
    fd: &DrmDeviceInternal,
    mapping: &PropMapping,
    connector: connector::Handle,
) -> Result<bool, Error> {
    Ok(mapping.conn_prop_handle(connector, "HDR_OUTPUT_METADATA").is_ok()
        && bt2020_rgb_value(fd, mapping, connector)?.is_some()
        && max_bpc_reaches_hdr(fd, mapping, connector)?)
}

fn current_max_bpc(
    fd: &DrmDeviceInternal,
    mapping: &PropMapping,
    connector: connector::Handle,
) -> Result<Option<u64>, Error> {
    let Ok(prop) = mapping.conn_prop_handle(connector, "max bpc") else {
        return Ok(None);
    };
    let values = fd
        .get_properties(connector)
        .map_err(|source| access_error(fd, "Error querying connector properties", source))?;
    let (ids, raw) = values.as_props_and_values();
    Ok(ids
        .iter()
        .zip(raw)
        .find(|(id, _)| **id == prop)
        .map(|(_, value)| *value))
}

/// Everything a new HDR state needs, or `None` when a connector cannot signal it.
pub(super) fn hdr_state(
    fd: &Arc<DrmDeviceInternal>,
    mapping: &PropMapping,
    connectors: &HashSet<connector::Handle>,
    metadata: HdrOutputMetadata,
    already_in_hdr: Option<&[(connector::Handle, u64)]>,
) -> Result<Option<ConnectorColor>, Error> {
    let mut colorspace = None;
    let mut restore_max_bpc = Vec::with_capacity(connectors.len());
    for connector in connectors {
        if !connector_can_signal_hdr(fd, mapping, *connector)? {
            return Ok(None);
        }
        let Some(value) = bt2020_rgb_value(fd, mapping, *connector)? else {
            return Ok(None);
        };
        // A connector set shares one write set, so it must share this value.
        if colorspace.is_some_and(|known| known != value) {
            return Ok(None);
        }
        colorspace = Some(value);
        // Changing metadata in HDR must still go back to what was there before HDR,
        // not to the 10 HDR set.
        let before_hdr = already_in_hdr.and_then(|restore| {
            restore
                .iter()
                .find(|(known, _)| known == connector)
                .map(|(_, value)| *value)
        });
        let restore = match before_hdr {
            Some(value) => Some(value),
            None => current_max_bpc(fd, mapping, *connector)?,
        };
        if let Some(bpc) = restore {
            restore_max_bpc.push((*connector, bpc));
        }
    }
    let Some(colorspace) = colorspace else {
        return Ok(None);
    };
    let kernel = metadata.to_kernel();
    let blob = match fd
        .create_property_blob(&kernel)
        .map_err(|source| access_error(fd, "Failed to create an HDR metadata blob", source))?
    {
        property::Value::Blob(id) => id,
        _ => unreachable!("create_property_blob returns a blob"),
    };
    Ok(Some(ConnectorColor::Hdr {
        metadata,
        blob: Arc::new(HdrBlob {
            fd: fd.clone(),
            id: blob,
        }),
        colorspace,
        restore_max_bpc,
    }))
}

/// What a connector set read back from KMS signals.
pub(super) fn read_back_color<'a, A: ControlDevice>(
    fd: &A,
    mapping: &PropMapping,
    connectors: impl IntoIterator<Item = &'a connector::Handle>,
) -> ConnectorColor {
    for connector in connectors {
        let Ok(values) = fd.get_properties(*connector) else {
            continue;
        };
        let (ids, raw) = values.as_props_and_values();
        let value_of = |name: &'static str| {
            let prop = mapping.conn_prop_handle(*connector, name).ok()?;
            ids.iter().zip(raw).find(|(id, _)| **id == prop).map(|(_, v)| *v)
        };
        if value_of("HDR_OUTPUT_METADATA").is_some_and(|blob| blob != 0)
            || value_of("Colorspace").is_some_and(|value| value != COLORSPACE_DEFAULT)
        {
            return ConnectorColor::Inherited;
        }
    }
    ConnectorColor::Sdr
}

#[cfg(test)]
mod tests {
    use super::*;

    const METADATA: HdrOutputMetadata = HdrOutputMetadata {
        primaries: [(34_000, 16_000), (13_250, 34_500), (7_500, 3_000)],
        white: (15_635, 16_450),
        max_mastering_luminance: 1_000,
        min_mastering_luminance: 50,
        max_content_light_level: 800,
        max_frame_average_light_level: 200,
    };

    #[test]
    fn the_blob_is_the_kernels_layout_with_pq_and_type_1() {
        let kernel = METADATA.to_kernel();
        let bytes: [u8; 32] = unsafe { std::mem::transmute(kernel) };
        let read = |offset: usize| u16::from_ne_bytes([bytes[offset], bytes[offset + 1]]);
        assert_eq!(&bytes[..4], &[0, 0, 0, 0], "HDMI_STATIC_METADATA_TYPE1");
        assert_eq!(bytes[4], 2, "EOTF SMPTE ST 2084");
        assert_eq!(bytes[5], 0, "static metadata type 1");
        assert_eq!(read(6), 34_000, "red x");
        assert_eq!(read(8), 16_000, "red y");
        assert_eq!(read(10), 13_250, "green x");
        assert_eq!(read(12), 34_500, "green y");
        assert_eq!(read(14), 7_500, "blue x");
        assert_eq!(read(16), 3_000, "blue y");
        assert_eq!(read(18), 15_635, "white x");
        assert_eq!(read(20), 16_450, "white y");
        assert_eq!(read(22), 1_000, "max mastering");
        assert_eq!(read(24), 50, "min mastering");
        assert_eq!(read(26), 800, "max cll");
        assert_eq!(read(28), 200, "max fall");
    }

    #[test]
    fn an_agreeing_state_writes_nothing_so_sdr_output_is_untouched() {
        let connector = connector::Handle::from(std::num::NonZeroU32::new(7).unwrap());
        assert!(connector_color_writes(&ConnectorColor::Sdr, &ConnectorColor::Sdr, [&connector]).is_empty());
    }

    #[test]
    fn inherited_signalling_is_corrected_to_sdr_and_leaves_max_bpc_alone() {
        let connector = connector::Handle::from(std::num::NonZeroU32::new(7).unwrap());
        assert_eq!(
            connector_color_writes(&ConnectorColor::Inherited, &ConnectorColor::Sdr, [&connector]),
            [ConnectorColorWrite {
                connector,
                colorspace: COLORSPACE_DEFAULT,
                metadata_blob: 0,
                max_bpc: None,
            }]
        );
    }
}
