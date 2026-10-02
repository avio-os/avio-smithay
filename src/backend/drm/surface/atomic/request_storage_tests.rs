use super::*;
use drm::control::{atomic::AtomicModeReq, from_u32, property, AtomicCommitFlags};
use std::os::unix::io::AsFd;

fn storage() -> AtomicRequestStorage {
    AtomicRequestStorage::cold(
        vec![ObjectRoster::cold(7, [1, 3, 8]), ObjectRoster::cold(2, [2, 4])],
        2,
    )
}
fn upstream_debug(storage: &AtomicRequestStorage) -> String {
    let objects: Vec<drm::control::RawResourceHandle> = storage
        .objects
        .iter()
        .map(|id| std::num::NonZeroU32::new(*id).unwrap())
        .collect();
    let props: Vec<property::Handle> = storage
        .properties
        .iter()
        .map(|id| from_u32(*id).unwrap())
        .collect();
    format!(
        "AtomicModeReq {{ objects: {:?}, count_props_per_object: {:?}, props: {:?}, values: {:?} }}",
        objects, storage.counts, props, storage.values
    )
}
#[test]
fn insertion_and_fifo_overwrite_match_actual_drm_request() {
    let mut request = storage();
    let mut upstream = AtomicModeReq::new();
    for (object, prop, value) in [
        (7, 8, 100),
        (2, 4, 200),
        (7, 1, 300),
        (2, 2, 400),
        (7, 3, 500),
        (7, 8, 600),
        (2, 4, 700),
    ] {
        request.add(object, prop, value).unwrap();
        upstream.add_raw_property(
            std::num::NonZeroU32::new(object).unwrap(),
            from_u32(prop).unwrap(),
            value,
        );
        assert_eq!(upstream_debug(&request), format!("{:?}", upstream));
    }
    assert_eq!(request.objects, [2, 7]);
    assert_eq!(request.counts, [2, 3]);
    assert_eq!(request.properties, [2, 4, 1, 3, 8]);
    assert_eq!(request.values, [400, 700, 300, 500, 600]);
}
#[test]
fn unknown_roster_never_mutates_native_arrays() {
    let mut request = storage();
    request.add(7, 1, 10).unwrap();
    for (object, property) in [(9, 1), (7, 9)] {
        assert!(matches!(
            request.add(object, property, 123),
            Err(Error::AtomicRequestCapacity { .. })
        ));
        assert_eq!(request.objects, [7]);
        assert_eq!(request.counts, [1]);
        assert_eq!(request.values, [10]);
    }
}
#[test]
fn exhaustion_is_typed_before_partial_insert_and_reuse_keeps_backing() {
    let mut request = storage();
    request.object_limit = 1;
    request.property_limit = 1;
    let ptrs = (
        request.objects.as_ptr(),
        request.counts.as_ptr(),
        request.properties.as_ptr(),
        request.values.as_ptr(),
    );
    request.add(7, 1, 10).unwrap();
    assert!(matches!(
        request.add(2, 2, 20),
        Err(Error::AtomicRequestCapacity { .. })
    ));
    assert!(matches!(
        request.add(7, 3, 30),
        Err(Error::AtomicRequestCapacity { .. })
    ));
    assert_eq!(request.objects, [7]);
    assert_eq!(request.properties, [1]);
    request.add(7, 1, 44).unwrap();
    assert_eq!(request.values, [44]);
    request.clear();
    request.add(2, 2, 99).unwrap();
    assert_eq!(request.values, [99]);
    assert_eq!(
        ptrs,
        (
            request.objects.as_ptr(),
            request.counts.as_ptr(),
            request.properties.as_ptr(),
            request.values.as_ptr()
        )
    );
}
#[test]
fn repeated_plane_updates_keep_exact_last_intent_and_bound() {
    let mut request = storage();
    let a = from_u32(2).unwrap();
    let b = from_u32(7).unwrap();
    request.remember_plane(a, true).unwrap();
    request.remember_plane(b, true).unwrap();
    request.remember_plane(a, false).unwrap();
    assert_eq!(request.plane_edits(), &[(a, false), (b, true)]);
    assert!(matches!(
        request.remember_plane(from_u32(9).unwrap(), true),
        Err(Error::AtomicRequestCapacity { .. })
    ));
    assert_eq!(request.plane_edits(), &[(a, false), (b, true)]);
}
#[test]
fn real_borrowed_ffi_dispatch_keeps_array_owners_on_native_error() {
    let null = std::fs::File::open("/dev/null").unwrap();
    let mut request = storage();
    request.add(7, 1, 55).unwrap();
    let pointer = request.values.as_ptr();
    let error = request
        .commit(
            null.as_fd(),
            (AtomicCommitFlags::TEST_ONLY | AtomicCommitFlags::NONBLOCK).bits(),
        )
        .unwrap_err();
    assert_eq!(error.raw_os_error(), Some(25)); // genuine ioctl ENOTTY, not a missing-device success
    assert_eq!(request.values, [55]);
    assert_eq!(request.values.as_ptr(), pointer);
    request.clear();
    request.add(2, 4, 66).unwrap();
    assert_eq!(request.values, [66]);
}
#[test]
#[cfg(feature = "backend_vulkan")]
fn warmed_native_request_refill_has_no_heap_operations() {
    let mut request = storage();
    let (_, calls) = crate::backend::renderer::vulkan::storage_heap_probe::measure(|| {
        for generation in 0..4096 {
            request.clear();
            request.add(7, 8, generation).unwrap();
            request.add(2, 4, generation + 1).unwrap();
            request.add(7, 1, generation + 2).unwrap();
            request.add(7, 8, generation + 3).unwrap();
            request.remember_plane(from_u32(7).unwrap(), true).unwrap();
            request.remember_plane(from_u32(7).unwrap(), false).unwrap();
        }
        request.clear();
    });
    assert_eq!(calls, [0; 4]);
}
