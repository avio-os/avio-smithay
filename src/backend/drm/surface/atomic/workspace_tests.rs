use super::*;
use drm::control::{connector, from_u32};
use std::sync::RwLock;

fn fixture() -> (PropMapping, crtc::Handle, connector::Handle) {
    let crtc = from_u32(1).unwrap();
    let connector = from_u32(2).unwrap();
    let mut mapping = PropMapping::default();
    mapping.crtcs.insert(
        crtc,
        [
            ("ACTIVE".into(), from_u32(10).unwrap()),
            ("MODE_ID".into(), from_u32(11).unwrap()),
        ]
        .into(),
    );
    mapping
        .connectors
        .insert(connector, [("CRTC_ID".into(), from_u32(12).unwrap())].into());
    (mapping, crtc, connector)
}
fn no_planes() -> Planes {
    Planes {
        primary: vec![],
        cursor: vec![],
        overlay: vec![],
    }
}

#[test]
fn sibling_discovery_writer_cannot_block_or_reinterpret_prepared_requests() {
    let (mapping, crtc, connector) = fixture();
    let discovery = RwLock::new(mapping);
    let mut first = AtomicRequestWorkspace::new(&discovery.read().unwrap(), crtc, &no_planes());
    let mut second = AtomicRequestWorkspace::new(&discovery.read().unwrap(), crtc, &no_planes());
    // Deliberately keep device discovery write-locked across both warm turns.
    let mut writer = discovery.write().unwrap();
    writer
        .connectors
        .get_mut(&connector)
        .unwrap()
        .insert("CRTC_ID".into(), from_u32(99).unwrap());
    for workspace in [&mut first, &mut second] {
        let mut request = workspace.request();
        request.reset_connector(connector).unwrap();
        request.reset_crtc(crtc).unwrap();
        assert_eq!(request.value(connector, "CRTC_ID"), Some(0));
        assert_eq!(request.value(crtc, "ACTIVE"), Some(0));
        drop(request);
        assert_eq!(
            workspace.mapping.conn_prop_handle(connector, "CRTC_ID").unwrap(),
            from_u32(12).unwrap()
        );
    }
    // Only this output's ordered cold adoption changes its accepted metadata.
    first.refresh_connectors(&writer);
    assert_eq!(
        first.mapping.conn_prop_handle(connector, "CRTC_ID").unwrap(),
        from_u32(99).unwrap()
    );
    assert_eq!(
        second.mapping.conn_prop_handle(connector, "CRTC_ID").unwrap(),
        from_u32(12).unwrap()
    );
    first.request().reset_connector(connector).unwrap();
}

#[test]
fn connector_admission_updates_metadata_and_capacity_together_and_preserves_disconnects() {
    let (mapping, crtc, old) = fixture();
    let mut workspace = AtomicRequestWorkspace::new(&mapping, crtc, &no_planes());
    let new: connector::Handle = from_u32(3).unwrap();
    assert!(workspace.request().reset_connector(new).is_err());
    let mut discovery = mapping;
    discovery.connectors.remove(&old);
    discovery
        .connectors
        .insert(new, [("CRTC_ID".into(), from_u32(13).unwrap())].into());
    workspace.refresh_connectors(&discovery);
    let capacity = workspace.connector_capacity();
    for _ in 0..120 {
        let mut request = workspace.request();
        request.reset_connector(old).unwrap();
        request.reset_connector(new).unwrap();
        assert_eq!(request.value(old, "CRTC_ID"), Some(0));
        assert_eq!(request.value(new, "CRTC_ID"), Some(0));
    }
    assert_eq!(workspace.connector_capacity(), capacity);
}
