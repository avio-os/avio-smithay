use super::*;
use std::time::Duration;

fn notified_bank(planes: usize, rectangles: usize) -> (PlaneDamageClipBank, Receiver<()>) {
    let (notify, notified) = mpsc::sync_channel(128);
    let callback = Arc::new(move || {
        notify.try_send(()).expect("bounded test observation");
    });
    let notification =
        StateReceiptReturnWakeup::with_retirement(Some(callback), Arc::new(AtomicBool::new(false)), None);
    (
        PlaneDamageClipBank::new(planes, rectangles, false, Some(notification)).unwrap(),
        notified,
    )
}
fn returned(notified: &Receiver<()>) {
    notified
        .recv_timeout(Duration::from_secs(5))
        .expect("exact cold owner-return edge");
}
fn reader(writer: &PlaneDamageWriter) -> PlaneDamageClips {
    // CPU ownership fixture only: no native blob or synthetic DRM receipt.
    PlaneDamageClips {
        owner: Some(DamageOwner::Pooled {
            slot: writer.slot.as_ref().unwrap().clone(),
            wake: writer.wake.clone(),
        }),
    }
}

#[test]
fn damage_clip_slots_follow_actual_native_state_lanes() {
    assert_eq!(PlaneDamageClipBank::owner_slots(3), Some(13));
    assert_eq!(PlaneDamageClipBank::owner_slots(usize::MAX), None);
    assert_eq!(
        PlaneDamageClipBank::bytes_per_rectangle(3),
        Some(13 * std::mem::size_of::<drm_ffi::drm_mode_rect>())
    );
    assert!(PlaneDamageClipBank::fixed_bytes(3).unwrap() >= 13 * std::mem::size_of::<Slot>());
}

#[test]
fn exact_plane_reader_clones_prevent_slot_reuse() {
    let (bank, notified) = notified_bank(1, 8);
    let mut writers: Vec<_> = (0..5).map(|_| bank.try_claim().unwrap()).collect();
    assert!(bank.try_claim().is_err());
    let original = reader(&writers[0]);
    let clone = original.clone();
    for writer in writers.drain(1..) {
        drop(writer);
    }
    for _ in 0..4 {
        returned(&notified);
    }
    assert_eq!(bank.state.available.load(Ordering::Acquire), 4);
    drop(writers.pop());
    drop(original);
    // The remaining cloned plane configuration is the exact reader proof.
    assert_eq!(bank.state.available.load(Ordering::Acquire), 4);
    let held: Vec<_> = (0..4).map(|_| bank.try_claim().unwrap()).collect();
    assert!(bank.try_claim().is_err());
    drop(clone);
    returned(&notified);
    let reused = bank.try_claim().unwrap();
    drop(reused);
    drop(held);
    for _ in 0..5 {
        returned(&notified);
    }
    assert!(bank.is_reclaimable());
}

#[test]
fn closing_bank_keeps_exact_live_reader_payload_rooted_cold() {
    let (bank, notified) = notified_bank(1, 4);
    let writer = bank.try_claim().unwrap();
    let held = reader(&writer);
    let weak = Arc::downgrade(writer.slot.as_ref().unwrap());
    drop(writer);
    drop(bank);
    assert!(
        weak.strong_count() >= 2,
        "cold root plus exact live reader survive closure"
    );
    drop(held);
    returned(&notified);
    // Return observation precedes actor registry removal; no timer or GPU
    // completion is inferred from this CPU-only lifetime fixture.
}

#[test]
fn exact_damage_mapping_refuses_before_any_partial_native_publication() {
    let src = Rectangle::new((10.0, 20.0).into(), (100.0, 50.0).into());
    let dst = Rectangle::from_size((50, 25).into());
    let mut scratch = Vec::with_capacity(2);
    fill_damage(&mut scratch, 2, false, src, dst, |visit| {
        visit(Rectangle::new((1, 2).into(), (3, 4).into()));
        visit(Rectangle::new((5, 6).into(), (7, 8).into()));
    })
    .unwrap();
    assert_eq!(
        (scratch[0].x1, scratch[0].y1, scratch[0].x2, scratch[0].y2),
        (12, 24, 18, 32)
    );
    assert_eq!(
        (scratch[1].x1, scratch[1].y1, scratch[1].x2, scratch[1].y2),
        (20, 32, 34, 48)
    );
    let error = fill_damage(&mut scratch, 2, false, src, dst, |visit| {
        for x in 0..96 {
            visit(Rectangle::new((x, 0).into(), (1, 1).into()));
        }
    })
    .unwrap_err();
    assert_eq!(error.required, 3);
    assert_eq!(scratch.len(), 2);
    // create() invokes this same collector before CREATEPROPBLOB. Refusal is
    // typed and cannot publish these two partial records as an immutable blob.
}

#[cfg(feature = "renderer_vulkan")]
#[test]
fn warmed_damage_claim_fill_clone_and_drop_have_zero_heap_calls() {
    use crate::backend::renderer::vulkan::storage_heap_probe;
    let (bank, notified) = notified_bank(1, 96);
    let src = Rectangle::from_size((128.0, 128.0).into());
    let dst = Rectangle::from_size((128, 128).into());
    for _ in 0..120 {
        let ((), calls) = storage_heap_probe::measure(|| {
            let writer = bank.try_claim().unwrap();
            {
                let mut payload = writer.slot.as_ref().unwrap().payload.try_lock().unwrap();
                fill_damage(&mut payload.rectangles, 96, false, src, dst, |visit| {
                    for x in 0..96 {
                        visit(Rectangle::new((x, 0).into(), (1, 1).into()));
                    }
                })
                .unwrap();
            }
            let original = reader(&writer);
            let clone = original.clone();
            drop(writer);
            drop(original);
            drop(clone);
        });
        assert_eq!(calls, [0; 4]);
        returned(&notified);
    }
    assert!(bank.is_reclaimable());
}

#[test]
fn cold_conservative_overflow_hint_covers_every_transformed_changed_pixel() {
    let src = Rectangle::new((10.25, -7.75).into(), (12.5, 15.5).into());
    let dst = Rectangle::from_size((17, 13).into());
    let raw: Vec<_> = (-4..20)
        .map(|x| Rectangle::new((x, (x + 6) % 13).into(), (5, 4).into()))
        .filter_map(|rect| rect.intersection(dst))
        .collect();
    let before = conservative_damage_clip_unions();
    let mut scratch = Vec::with_capacity(2);
    fill_damage(&mut scratch, 2, true, src, dst, |visit| {
        raw.iter().copied().for_each(visit)
    })
    .unwrap();
    assert_eq!(scratch.len(), 1);
    assert_eq!(conservative_damage_clip_unions(), before + 1);
    let union = scratch[0];
    // Independent endpoint projection checks all changed source pixels. The
    // rectangles supplied above are clipped to the actual physical viewport.
    for rect in &raw {
        let x1 = (10.25 + f64::from(rect.loc.x) * 12.5 / 17.0).floor() as i32;
        let x2 = (10.25 + f64::from(rect.loc.x + rect.size.w) * 12.5 / 17.0).ceil() as i32;
        let y1 = (-7.75 + f64::from(rect.loc.y) * 15.5 / 13.0).floor() as i32;
        let y2 = (-7.75 + f64::from(rect.loc.y + rect.size.h) * 15.5 / 13.0).ceil() as i32;
        for y in y1..y2 {
            for x in x1..x2 {
                assert!(union.x1 <= x && x < union.x2 && union.y1 <= y && y < union.y2);
            }
        }
    }
    let error = fill_damage(&mut scratch, 2, false, src, dst, |visit| {
        raw.iter().copied().for_each(visit)
    })
    .unwrap_err();
    assert_eq!(error.required, 3);
    assert_eq!(
        conservative_damage_clip_unions(),
        before + 1,
        "exact refusal never advertises a union"
    );
    fill_damage(&mut scratch, 2, true, src, dst, |visit| {
        raw.iter().copied().take(2).for_each(visit)
    })
    .unwrap();
    assert_eq!(
        scratch.len(),
        2,
        "conservative policy retains exact hints while they fit"
    );
    assert_eq!(conservative_damage_clip_unions(), before + 1);
}

#[cfg(feature = "renderer_vulkan")]
#[test]
fn closing_bank_and_last_writer_never_final_free_transport_on_warm_caller() {
    use crate::backend::renderer::vulkan::storage_heap_probe;
    for _ in 0..120 {
        let (bank, notified) = notified_bank(1, 96);
        let writer = bank.try_claim().unwrap();
        let held = reader(&writer);
        let ((), calls) = storage_heap_probe::measure(|| {
            drop(bank);
            drop(writer);
            drop(held);
        });
        assert_eq!(calls, [0; 4]);
        returned(&notified);
    }
}
