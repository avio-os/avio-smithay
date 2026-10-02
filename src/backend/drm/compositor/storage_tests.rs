use super::native_black::tests::{config, TestBuffer, TestFramebuffer};
use super::*;
use std::num::NonZeroU32;

#[test]
fn twelve_plane_snapshots_and_native_owners_survive_independent_delayed_receipts() {
    let storage = DrmFrameStorage::<TestBuffer, TestFramebuffer>::new(12, 12, 32, 2).unwrap();
    let mut current = storage.empty_frame(12).unwrap();
    for number in 1..=12 {
        current
            .planes
            .push((
                NonZeroU32::new(number).unwrap().into(),
                PlaneState {
                    config: Some(config()),
                    skip: false,
                    ..Default::default()
                },
            ))
            .unwrap();
    }
    let pending = current.copy_for_update(Some(&storage)).unwrap();
    let queued = current.copy_for_update(Some(&storage)).unwrap();
    let mut candidate = current.copy_for_update(Some(&storage)).unwrap();
    assert!(current.copy_for_update(Some(&storage)).is_err());
    candidate.planes[0].1.config = None;
    assert!(pending.planes[0].1.config.is_some());
    assert!(queued.planes[0].1.config.is_some());
    drop(candidate);
    let replacement = current.copy_for_update(Some(&storage)).unwrap();
    assert_eq!(replacement.planes.len(), 12);
    assert!(
        pending.planes[0].1.config.is_some(),
        "no reuse merely because candidate wasdiscarded"
    );
}

#[test]
fn returned_selections_borrow_original_source_and_keep_their_own_index_storage() {
    let storage = DrmFrameStorage::<TestBuffer, TestFramebuffer>::new(3, 1, 4, 1).unwrap();
    let source = [11, 22, 33];
    let mut indices = storage.returned_indices.acquire(3).unwrap();
    indices.extend([2, 0]).unwrap();
    let selection = SelectedElements::new(&source, indices);
    assert_eq!(selection.iter().copied().collect::<Vec<_>>(), [33, 11]);
    let held = storage.returned_indices.acquire(3).unwrap();
    assert!(storage.returned_indices.acquire(0).is_err());
    drop(held);
    let mut replacement = storage.returned_indices.acquire(3).unwrap();
    replacement.push(1).unwrap();
    assert_eq!(selection.iter().copied().collect::<Vec<_>>(), [33, 11]);
    assert!(std::ptr::eq(selection.iter().next().unwrap(), &source[2]));
}

#[test]
fn warmed_120_twelve_plane_native_configuration_copies_allocate_and_free_nothing() {
    let storage = DrmFrameStorage::<TestBuffer, TestFramebuffer>::new(12, 12, 32, 2).unwrap();
    let mut current = storage.empty_frame(12).unwrap();
    for number in 1..=12 {
        current
            .planes
            .push((
                NonZeroU32::new(number).unwrap().into(),
                PlaneState {
                    config: Some(config()),
                    skip: false,
                    ..Default::default()
                },
            ))
            .unwrap();
    }
    let pending = current.copy_for_update(Some(&storage)).unwrap();
    let (_, counts) = crate::backend::renderer::storage_heap_probe::measure(|| {
        for _ in 0..120 {
            let mut cursor_update = current.copy_for_update(Some(&storage)).unwrap();
            cursor_update.planes[0]
                .1
                .config
                .as_mut()
                .unwrap()
                .properties
                .dst
                .loc
                .x = 10;
            let newer = current.copy_for_update(Some(&storage)).unwrap();
            assert!(current.copy_for_update(Some(&storage)).is_err());
            assert_eq!(
                pending.planes[0].1.config.as_ref().unwrap().properties.dst.loc.x,
                0
            );
            drop(cursor_update);
            drop(newer);
        }
    });
    assert_eq!(
        counts, [0; 4],
        "actual native owner/plane configuration containers; no kernel/GPU submission in fixture"
    );
}

fn prepared_storage(
    receipts: usize,
    returned: Option<Arc<dyn Fn() + Send + Sync>>,
) -> PreparedDrmFrameStorage<TestBuffer, TestFramebuffer> {
    PreparedDrmFrameStorage::new(
        OutputModeSource::Static {
            size: (128, 128).into(),
            scale: 1.0.into(),
            transform: Transform::Normal,
        },
        12,
        32,
        receipts,
        DamageStoragePolicy::Exact,
        12,
        2,
        None,
        returned,
    )
    .unwrap()
}

#[test]
fn cold_packet_waits_for_actual_main_selection_and_native_reader_returns() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let signals = Arc::new(AtomicUsize::new(0));
    let observe = signals.clone();
    let returned: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
        observe.fetch_add(1, Ordering::Relaxed);
    });
    let mut packet = prepared_storage(2, Some(returned.clone()));
    assert_eq!(packet.state_receipt_capacity(), 2);
    let map = packet.states.as_ref().unwrap().acquire(12).unwrap();
    let selection = packet
        .selection
        .as_ref()
        .unwrap()
        .returned_indices
        .acquire(12)
        .unwrap();
    let current = packet.selection.as_ref().unwrap().empty_frame(12).unwrap();
    let pending = current.copy_for_update(packet.selection.as_ref()).unwrap();
    packet.old_current = Some(current);
    assert!(
        !packet.is_reclaimable(),
        "active or parked storage cannot be reclaimed"
    );
    packet.arm_retirement();
    assert_eq!(
        signals.load(Ordering::Relaxed),
        0,
        "cold parked-current drop needs no wake"
    );
    assert!(!packet.is_reclaimable());
    drop(map);
    assert_eq!(signals.load(Ordering::Relaxed), 1);
    assert!(
        !packet.is_reclaimable(),
        "Main map alone does not release native or selection custody"
    );
    drop(selection);
    assert_eq!(signals.load(Ordering::Relaxed), 2);
    assert!(
        !packet.is_reclaimable(),
        "pending native snapshot still names the old bank"
    );
    drop(pending);
    assert_eq!(
        signals.load(Ordering::Relaxed),
        4,
        "both native vector slots returned"
    );
    assert!(packet.is_reclaimable());
    assert!(
        Arc::strong_count(&returned) > 1,
        "cold actor retains notification control through disposal"
    );
}

#[test]
fn sixty_four_is_real_admitted_map_capacity_not_an_inferred_kms_depth() {
    let mut packet = prepared_storage(64, None);
    let mut held = Vec::with_capacity(64);
    let bank = packet.states.as_ref().unwrap();
    for _ in 0..64 {
        held.push(bank.acquire(12).unwrap());
    }
    assert_eq!(packet.state_receipt_capacity(), 64);
    let failure = bank.acquire(1).unwrap_err();
    assert_eq!(failure.capacity, 64);
    assert_eq!(failure.required, 65);
    drop(held.pop());
    let restored = bank.acquire(12).unwrap();
    assert!(bank.acquire(1).is_err());
    drop(restored);
    drop(held);
    packet.arm_retirement();
    assert!(packet.is_reclaimable());
}

#[test]
fn last_warm_reader_returns_do_not_free_displaced_cold_banks() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let wakes = Arc::new(AtomicUsize::new(0));
    let observed = wakes.clone();
    let wake: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
        observed.fetch_add(1, Ordering::Relaxed);
    });
    let mut old = prepared_storage(2, Some(wake.clone()));
    let replacement = prepared_storage(2, None);
    let mut current = old.selection.as_ref().unwrap().empty_frame(12).unwrap();
    for number in 1..=12 {
        current
            .planes
            .push((
                NonZeroU32::new(number).unwrap().into(),
                PlaneState {
                    config: Some(config()),
                    skip: false,
                    ..Default::default()
                },
            ))
            .unwrap();
    }
    // Adoption retains exactly the same strong native config resources in
    // its distinct new bank. This is real production copy_for_update, not a
    // fake completion or timer claiming that native readers went away.
    let active = current.copy_for_update(replacement.selection.as_ref()).unwrap();
    let pending = current.copy_for_update(old.selection.as_ref()).unwrap();
    let map = old.states.as_ref().unwrap().acquire(12).unwrap();
    let selection = old
        .selection
        .as_ref()
        .unwrap()
        .returned_indices
        .acquire(12)
        .unwrap();
    old.old_current = Some(current);
    old.arm_retirement();
    let (_, counts) = crate::backend::renderer::storage_heap_probe::measure(|| {
        drop(pending);
        drop(map);
        drop(selection);
        assert!(old.is_reclaimable());
    });
    assert_eq!(
        counts, [0; 4],
        "final warm leases return storage; cold actor alone destroys banks"
    );
    assert_eq!(wakes.load(Ordering::Relaxed), 4);
    assert_eq!(active.planes.len(), 12);
    drop(old); // The owning untagged actor disposes cold vector/map backing.
}

#[test]
fn physical_overlay_identity_preserves_repair_when_sources_alternate_and_readd() {
    use crate::backend::renderer::element::solid::{SolidColorBuffer, SolidColorRenderElement};
    let plane: plane::Handle = NonZeroU32::new(1).unwrap().into();
    let mut identities = OverlayPlaneElementIds::from_handles(std::iter::once(plane));
    let buffers = [
        SolidColorBuffer::new((12, 10), [1.0; 4]),
        SolidColorBuffer::new((12, 10), [1.0; 4]),
    ];
    let sources = buffers
        .map(|buffer| SolidColorRenderElement::from_buffer(&buffer, (4, 5), 1.0, 1.0, Kind::Unspecified));
    let mut tracker = OutputDamageTracker::new((64, 64), 1.0, Transform::Normal);
    tracker.prepare_frame_storage(1, 32, 2).unwrap();
    let (id, first_commit) = identities
        .plane_id_for_element_id(&plane, sources[0].id())
        .unwrap();
    let mut descriptor = PlaneElementDescriptor {
        id: id.clone(),
        commit: first_commit,
        geometry: Rectangle::new((4, 5).into(), (12, 10).into()),
        source_index: 0,
        opaque_offset: (0, 0).into(),
        holepunch: true,
    };
    let fake = BorrowedPlaneElement {
        descriptor: &descriptor,
        source: &sources[0],
        scale: 1.0.into(),
    };
    let (_, initial) = tracker.damage_output(0, &[fake]).unwrap();
    drop(initial);
    let (_, same_commit) = identities
        .plane_id_for_element_id(&plane, sources[0].id())
        .unwrap();
    assert_eq!(same_commit, first_commit);
    let fake = BorrowedPlaneElement {
        descriptor: &descriptor,
        source: &sources[0],
        scale: 1.0.into(),
    };
    let (damage, unchanged) = tracker.damage_output(1, &[fake]).unwrap();
    assert!(damage.is_none());
    drop(unchanged);
    let (same_id, next_commit) = identities
        .plane_id_for_element_id(&plane, sources[1].id())
        .unwrap();
    assert_eq!(id, same_id);
    assert!(next_commit > first_commit);
    descriptor.commit = next_commit;
    let fake = BorrowedPlaneElement {
        descriptor: &descriptor,
        source: &sources[1],
        scale: 1.0.into(),
    };
    let (damage, changed) = tracker.damage_output(1, &[fake]).unwrap();
    assert!(damage
        .unwrap()
        .iter()
        .any(|rectangle| rectangle.contains_rect(descriptor.geometry)));
    drop(changed);
    identities.remove_plane(&plane).unwrap();
    let (damage, removed) = tracker
        .damage_output(1, &[] as &[SolidColorRenderElement])
        .unwrap();
    assert!(damage
        .unwrap()
        .iter()
        .any(|rectangle| rectangle.contains_rect(descriptor.geometry)));
    drop(removed);
    let (same_id, readded) = identities
        .plane_id_for_element_id(&plane, sources[0].id())
        .unwrap();
    assert_eq!(id, same_id);
    assert!(readded > next_commit);
    descriptor.commit = readded;
    let fake = BorrowedPlaneElement {
        descriptor: &descriptor,
        source: &sources[0],
        scale: 1.0.into(),
    };
    let (damage, readded_state) = tracker.damage_output(1, &[fake]).unwrap();
    assert!(damage
        .unwrap()
        .iter()
        .any(|rectangle| rectangle.contains_rect(descriptor.geometry)));
    drop(readded_state);
    let (_, counts) = crate::backend::renderer::storage_heap_probe::measure(|| {
        for cycle in 0..120 {
            let source = &sources[cycle % sources.len()];
            let (same_id, commit) = identities.plane_id_for_element_id(&plane, source.id()).unwrap();
            assert_eq!(same_id, id);
            descriptor.commit = commit;
            let fake = BorrowedPlaneElement {
                descriptor: &descriptor,
                source,
                scale: 1.0.into(),
            };
            drop(tracker.damage_output(1, &[fake]).unwrap());
            identities.remove_plane(&plane).unwrap();
            drop(
                tracker
                    .damage_output(1, &[] as &[SolidColorRenderElement])
                    .unwrap(),
            );
        }
    });
    assert_eq!(counts, [0;4], "source IDs and synthetic identities were admitted cold; warm change/removal repairs do not allocate or free");
}

#[test]
fn overlay_revision_exhaustion_preserves_current_source_and_never_wraps() {
    let plane: plane::Handle = NonZeroU32::new(1).unwrap().into();
    let mut identities = OverlayPlaneElementIds::from_handles(std::iter::once(plane));
    let old = Id::new();
    let new = Id::new();
    identities.plane_id_for_element_id(&plane, &old).unwrap();
    identities.plane_ids[0].revision = usize::MAX;
    assert!(identities.plane_id_for_element_id(&plane, &new).is_err());
    assert!(identities.remove_plane(&plane).is_err());
    assert_eq!(identities.plane_ids[0].source.as_ref(), Some(&old));
    assert_eq!(identities.plane_ids[0].revision, usize::MAX);
}

fn cache_state() -> ElementState<TestFramebuffer> {
    ElementState {
        instances: SmallVec::new(),
        fb_cache: ElementFramebufferCache::default(),
    }
}

#[test]
fn native_selection_failure_restores_admitted_table_and_preserves_partial_cache_transfers() {
    let id = Id::new();
    let other = Id::new();
    let mut current = IndexMap::with_capacity(12);
    let mut previous = IndexMap::with_capacity(12);
    previous.insert(id.clone(), cache_state());
    previous.insert(other.clone(), cache_state());
    let mut selected = std::mem::take(&mut current);
    // Real selection transfers cache metadata from the previous accepted lane.
    let transferred = previous.swap_remove(&id).unwrap();
    selected.insert(id.clone(), transferred);
    let failure = FrameWorkspaceError {
        resource: "test native preparation refusal",
        required: 13,
        capacity: 12,
    };
    assert_eq!(
        native_cache::finish_selection(&mut current, &mut previous, selected, Err::<(), _>(failure)),
        Err(failure)
    );
    assert!(current.capacity() >= 12);
    assert!(previous.capacity() >= 12);
    assert!(current.contains_key(&id));
    assert!(
        previous.contains_key(&other),
        "failed selection does not discard still-owned native caches"
    );
}

#[test]
fn warmed_native_selection_errors_and_retries_never_drop_or_reallocate_cold_tables() {
    let id = Id::new();
    let previous_id = Id::new();
    let mut current = IndexMap::with_capacity(12);
    let mut previous = IndexMap::with_capacity(12);
    previous.insert(previous_id.clone(), cache_state());
    let error = FrameWorkspaceError {
        resource: "actual selector capacity refusal",
        required: 13,
        capacity: 12,
    };
    let (_, counts) = crate::backend::renderer::storage_heap_probe::measure(|| {
        for _ in 0..120 {
            let mut selected = std::mem::take(&mut current);
            selected.entry(id.clone()).or_insert_with(cache_state);
            assert!(native_cache::finish_selection(
                &mut current,
                &mut previous,
                selected,
                Err::<(), _>(error)
            )
            .is_err());
            assert!(previous.contains_key(&previous_id));
            assert!(current.capacity() >= 12);
        }
        let selected = std::mem::take(&mut current);
        native_cache::finish_selection(
            &mut current,
            &mut previous,
            selected,
            Ok::<(), FrameWorkspaceError>(()),
        )
        .unwrap();
        assert!(previous.is_empty());
        assert!(previous.capacity() >= 12);
    });
    assert_eq!(
        counts, [0; 4],
        "real owning selector restoration includes failure, retry and successful cleanup"
    );
}
