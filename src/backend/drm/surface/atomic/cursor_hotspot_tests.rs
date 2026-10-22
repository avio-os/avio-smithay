use super::*;
use crate::backend::drm::surface::PlaneConfig;
use crate::utils::Rectangle;
use drm::control::from_u32;

fn mapping(hotspots: bool) -> (PropMapping, plane::Handle) {
    let plane = from_u32(2).unwrap();
    let mut mapping = PropMapping::default();
    mapping.planes.insert(
        plane,
        PLANE_PROPERTIES
            .iter()
            .enumerate()
            .filter(|(_, name)| hotspots || !name.starts_with("HOTSPOT_"))
            .map(|(i, name)| ((*name).into(), from_u32(10 + i as u32).unwrap()))
            .collect(),
    );
    (mapping, plane)
}

fn state(plane: plane::Handle, hotspot: Option<(i32, i32)>) -> PlaneState<'static> {
    PlaneState {
        handle: plane,
        config: Some(PlaneConfig {
            src: Rectangle::from_size((64.0, 64.0).into()),
            dst: Rectangle::new((120, 240).into(), (64, 64).into()),
            transform: Transform::Normal,
            alpha: 1.0,
            damage_clips: None,
            fb: from_u32(3).unwrap(),
            fence: None,
            cursor_hotspot: hotspot.map(Into::into),
        }),
    }
}

#[test]
fn virtual_cursor_encodes_signed_hotspot_and_clear_with_its_native_state() {
    let (mapping, plane) = mapping(true);
    let mut request = AtomicRequest::new(&mapping);
    request
        .set_plane(from_u32(1).unwrap(), &state(plane, Some((-3, 24))))
        .unwrap();
    assert_eq!(request.value(plane, "HOTSPOT_X"), Some((-3i64) as u64));
    assert_eq!(request.value(plane, "HOTSPOT_Y"), Some(24));
    assert_eq!(request.value(plane, "FB_ID"), Some(3));
    request.reset_plane(plane).unwrap();
    assert_eq!(request.value(plane, "HOTSPOT_X"), Some(0));
    assert_eq!(request.value(plane, "HOTSPOT_Y"), Some(0));
    assert_eq!(request.value(plane, "FB_ID"), Some(0));
}

#[test]
fn virtual_cursor_refuses_missing_or_half_advertised_hotspots() {
    let (mut mapping, plane) = mapping(true);
    assert!(matches!(AtomicRequest::new(&mapping)
        .set_plane(from_u32(1).unwrap(), &state(plane, None)),
        Err(Error::UnsupportedPlaneConfiguration(p)) if p == plane));
    mapping.planes.get_mut(&plane).unwrap().remove("HOTSPOT_Y");
    assert!(matches!(
        AtomicRequest::new(&mapping).set_plane(from_u32(1).unwrap(), &state(plane, Some((4, 5)))),
        Err(Error::UnknownProperty {
            name: "HOTSPOT_Y",
            ..
        })
    ));
}

#[test]
fn ordinary_plane_without_hotspot_properties_keeps_its_existing_contract() {
    let (mapping, plane) = mapping(false);
    let mut request = AtomicRequest::new(&mapping);
    request
        .set_plane(from_u32(1).unwrap(), &state(plane, None))
        .unwrap();
    assert_eq!(request.value(plane, "FB_ID"), Some(3));
    assert_eq!(request.value(plane, "HOTSPOT_X"), None);
    assert_eq!(request.value(plane, "HOTSPOT_Y"), None);
}
