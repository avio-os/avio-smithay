use super::*;
use crate::backend::renderer::{
    element::solid::{SolidColorBuffer, SolidColorRenderElement},
    test::{DummyFramebuffer, DummyRenderer},
};

fn element() -> SolidColorRenderElement {
    let buffer = SolidColorBuffer::new((20, 20), Color32F::new(1.0, 1.0, 1.0, 1.0));
    SolidColorRenderElement::from_buffer(&buffer, (10, 10), 1.0, 1.0, Kind::Unspecified)
}

#[test]
fn held_damage_receipts_defer_then_reuse_only_the_dropped_slot() {
    let elements = [element()];
    let mut tracker = OutputDamageTracker::new((100, 100), 1.0, Transform::Normal);
    tracker.prepare_frame_storage(1, 32, 2).unwrap();
    let (_, old) = tracker.damage_output(0, &elements).unwrap();
    let (_, latest) = tracker.damage_output(0, &elements).unwrap();
    assert!(matches!(
        tracker.damage_output(0, &elements),
        Err(DamageOutputError::WorkspaceCapacity(_))
    ));
    let old_state = old.element_render_state(elements[0].id().clone()).unwrap();
    drop(latest);
    let (_, replacement) = tracker.damage_output(0, &elements).unwrap();
    assert_eq!(
        old.element_render_state(elements[0].id().clone())
            .unwrap()
            .visible_area,
        old_state.visible_area
    );
    drop(replacement);
}

#[test]
fn capacity_failure_does_not_grow_or_replace_existing_receipts() {
    let elements = [element(), element()];
    let mut tracker = OutputDamageTracker::new((100, 100), 1.0, Transform::Normal);
    tracker.prepare_frame_storage(1, 16, 2).unwrap();
    let indices_capacity = tracker.render_indices.capacity();
    assert!(matches!(
        tracker.damage_output(0, &elements),
        Err(DamageOutputError::WorkspaceCapacity(_))
    ));
    assert_eq!(tracker.render_indices.capacity(), indices_capacity);
    assert!(tracker.last_state.elements.is_empty());
    assert!(tracker.damage_output(0, &elements[..1]).is_ok());
}

#[test]
fn cold_reconfiguration_preserves_old_receipt_ownership() {
    let elements = [element()];
    let mut tracker = OutputDamageTracker::new((100, 100), 1.0, Transform::Normal);
    tracker.prepare_frame_storage(1, 32, 1).unwrap();
    let (_, old) = tracker.damage_output(0, &elements).unwrap();
    tracker.prepare_frame_storage(2, 64, 1).unwrap();
    let (_, next) = tracker.damage_output(0, &elements).unwrap();
    assert!(old.element_was_presented(elements[0].id().clone()));
    drop(old);
    assert!(matches!(
        tracker.damage_output(0, &elements),
        Err(DamageOutputError::WorkspaceCapacity(_))
    ));
    drop(next);
    assert!(tracker.damage_output(0, &elements).is_ok());
}

#[test]
fn bounded_damage_and_history_match_legacy_repaint_and_no_damage() {
    let elements = [element()];
    let mut bounded = OutputDamageTracker::new((100, 100), 1.0, Transform::Normal);
    bounded.prepare_frame_storage(1, 32, 2).unwrap();
    let mut legacy = OutputDamageTracker::new((100, 100), 1.0, Transform::Normal);
    let mut renderer = DummyRenderer::default();
    let mut framebuffer = DummyFramebuffer;
    for age in [0, 1, 2, 0, 3, 1, 0] {
        let result = bounded
            .render_output(&mut renderer, &mut framebuffer, age, &elements, Color32F::BLACK)
            .unwrap();
        let reference = legacy
            .render_output(&mut renderer, &mut framebuffer, age, &elements, Color32F::BLACK)
            .unwrap();
        assert_eq!(result.damage, reference.damage);
        assert_eq!(result.damage_summary, reference.damage_summary);
        let actual = result
            .states
            .element_render_state(elements[0].id().clone())
            .unwrap();
        let expected = reference
            .states
            .element_render_state(elements[0].id().clone())
            .unwrap();
        assert_eq!(actual.visible_area, expected.visible_area);
        assert_eq!(actual.presentation_state, expected.presentation_state);
        assert_eq!(actual.needs_capture, expected.needs_capture);
    }
    assert!(bounded.last_state.old_damage.len() <= MAX_AGE + 1);
}

#[test]
fn conservative_repair_preserves_no_change_and_exact_occluded_visibility() {
    let upper = element();
    let lower = element();
    let elements = [upper, lower];
    let mut tracker = OutputDamageTracker::new((100, 100), 1.0, Transform::Normal);
    tracker
        .prepare_frame_storage_with_policy(2, 2, 2, DamageStoragePolicy::ConservativeFullOutput)
        .unwrap();
    let (damage, states) = tracker.damage_output(0, &elements).unwrap();
    assert_eq!(
        damage.unwrap().as_slice(),
        &[Rectangle::from_size((100, 100).into())]
    );
    assert_eq!(
        states
            .element_render_state(elements[0].id().clone())
            .unwrap()
            .visible_area,
        400
    );
    assert_eq!(
        states
            .element_render_state(elements[1].id().clone())
            .unwrap()
            .visible_area,
        0
    );
    assert_eq!(
        tracker.render_indices,
        [0, 1],
        "occluded source participates in full repair without claiming visiblepixels"
    );
    drop(states);
    assert!(tracker.damage_output(1, &elements).unwrap().0.is_none());
    assert!(
        tracker.damage_output(0, &elements).unwrap().0.is_some(),
        "unknown target age repairs fulltarget"
    );
}

#[test]
fn conservative_repair_recaptures_effect_and_retains_its_occluded_prefix() {
    use super::framebuffer_effect_tests::TestElement;
    let rect = Rectangle::new((10, 10).into(), (20, 20).into());
    let elements = [
        TestElement::draw(rect, 1).opaque(),
        TestElement::effect(rect, rect),
        TestElement::draw(rect, 1),
    ];
    let mut tracker = OutputDamageTracker::new((100, 100), 1.0, Transform::Normal);
    tracker
        .prepare_frame_storage_with_policy(3, 1, 2, DamageStoragePolicy::ConservativeFullOutput)
        .unwrap();
    let (_, states) = tracker.damage_output(0, &elements).unwrap();
    assert!(
        states
            .element_render_state(elements[1].id().clone())
            .unwrap()
            .needs_capture
    );
    assert_eq!(
        states
            .element_render_state(elements[2].id().clone())
            .unwrap()
            .visible_area,
        0
    );
    assert_eq!(tracker.render_indices, [0, 1, 2]);
    drop(states);
    let (_, states) = tracker.damage_output(1, &elements).unwrap();
    assert!(
        !states
            .element_render_state(elements[1].id().clone())
            .unwrap()
            .needs_capture
    );
}

#[test]
fn exact_area_sweep_matches_discrete_union_with_overlaps_and_negative_edges() {
    let rect = Rectangle::new((-3, -2).into(), (13, 11).into());
    let cuts = [
        Rectangle::new((-8, -5).into(), (7, 15).into()),
        Rectangle::new((-1, 1).into(), (7, 4).into()),
        Rectangle::new((4, -1).into(), (20, 8).into()),
    ];
    let mut expected = 0;
    for y in rect.loc.y..rect.loc.y + rect.size.h {
        for x in rect.loc.x..rect.loc.x + rect.size.w {
            if !cuts.iter().any(|cut| cut.contains((x, y))) {
                expected += 1;
            }
        }
    }
    assert_eq!(workspace::visible_area(rect, &cuts), expected);
    assert_eq!(workspace::visible_area(rect, &[rect]), 0);
}

#[test]
fn warmed_120_damage_repairs_allocate_reallocate_and_free_nothing() {
    use super::framebuffer_effect_tests::TestElement;
    let mut elements = [TestElement::draw(
        Rectangle::new((10, 10).into(), (20, 20).into()),
        1,
    )];
    for policy in [
        DamageStoragePolicy::Exact,
        DamageStoragePolicy::ConservativeFullOutput,
    ] {
        let mut tracker = OutputDamageTracker::new((100, 100), 1.0, Transform::Normal);
        tracker
            .prepare_frame_storage_with_policy(1, 64, 2, policy)
            .unwrap();
        drop(tracker.damage_output(0, &elements).unwrap());
        let (_, counts) = crate::backend::renderer::storage_heap_probe::measure(|| {
            for _ in 0..120 {
                elements[0].advance_commit();
                let (damage, states) = tracker.damage_output(1, &elements).unwrap();
                assert!(damage.is_some());
                assert!(states.element_was_presented(elements[0].id().clone()));
                drop(states);
            }
        });
        assert_eq!(
            counts, [0; 4],
            "actual tracker alloc/zero/realloc/free for {policy:?}"
        );
    }
}

#[test]
fn warmed_120_actual_render_repairs_allocate_reallocate_and_free_nothing() {
    let mut buffer = SolidColorBuffer::new((20, 20), Color32F::new(1.0, 1.0, 1.0, 1.0));
    let mut elements = [SolidColorRenderElement::from_buffer(
        &buffer,
        (10, 10),
        1.0,
        1.0,
        Kind::Unspecified,
    )];
    for policy in [
        DamageStoragePolicy::Exact,
        DamageStoragePolicy::ConservativeFullOutput,
    ] {
        let mut tracker = OutputDamageTracker::new((100, 100), 1.0, Transform::Normal);
        tracker
            .prepare_frame_storage_with_policy(1, 64, 2, policy)
            .unwrap();
        let mut renderer = DummyRenderer::default();
        let mut framebuffer = DummyFramebuffer;
        drop(
            tracker
                .render_output(&mut renderer, &mut framebuffer, 0, &elements, Color32F::BLACK)
                .unwrap(),
        );
        let (_, counts) = crate::backend::renderer::storage_heap_probe::measure(|| {
            for frame in 0..120 {
                buffer.set_color(Color32F::new((frame % 2) as f32, 1.0, 1.0, 1.0));
                elements[0] =
                    SolidColorRenderElement::from_buffer(&buffer, (10, 10), 1.0, 1.0, Kind::Unspecified);
                let result = tracker
                    .render_output(&mut renderer, &mut framebuffer, 1, &elements, Color32F::BLACK)
                    .unwrap();
                assert!(result.damage.is_some());
                drop(result);
            }
        });
        assert_eq!(
            counts, [0; 4],
            "actual tracker/render alloc/zero/realloc/free for {policy:?}; CPU DummyRenderer fixture"
        );
    }
}

#[test]
fn subtraction_checks_every_original_after_appending_split_pieces() {
    // Repeated original seeds expose a swap_remove trap: an appended split
    // must not replace the next untested original in the scan.
    for seeds in 1..=3 {
        let seed = Rectangle::from_size((16, 16).into());
        let mut random = 13u32;
        for count in 0..=24 {
            let mut rectangles = Vec::with_capacity(768);
            rectangles.extend(std::iter::repeat_n(seed, seeds));
            let mut expected = [seeds; 256];
            for _ in 0..count {
                random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                let x = (random % 16) as i32;
                random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                let y = (random % 16) as i32;
                let cut = Rectangle::new(
                    (x, y).into(),
                    (
                        1 + (random % (16 - x) as u32) as i32,
                        1 + (random % (16 - y) as u32) as i32,
                    )
                        .into(),
                );
                for py in cut.loc.y..cut.loc.y + cut.size.h {
                    for px in cut.loc.x..cut.loc.x + cut.size.w {
                        expected[(py * 16 + px) as usize] = 0;
                    }
                }
                workspace::subtract(&mut rectangles, [cut], Some(768)).unwrap();
            }
            let mut actual = [0; 256];
            for rectangle in rectangles {
                for y in rectangle.loc.y..rectangle.loc.y + rectangle.size.h {
                    for x in rectangle.loc.x..rectangle.loc.x + rectangle.size.w {
                        actual[(y * 16 + x) as usize] += 1;
                    }
                }
            }
            assert_eq!(actual, expected);
        }
    }
}

struct StreamedRectangles {
    id: Id,
}
impl Element for StreamedRectangles {
    fn id(&self) -> &Id {
        &self.id
    }
    fn current_commit(&self) -> CommitCounter {
        CommitCounter::from(1)
    }
    fn src(&self) -> Rectangle<f64, BufferCoords> {
        Rectangle::from_size((64.0, 64.0).into())
    }
    fn geometry(&self, _: Scale<f64>) -> Rectangle<i32, Physical> {
        Rectangle::new((30, 20).into(), (64, 64).into())
    }
    fn damage_since(
        &self,
        _: Scale<f64>,
        _: Option<CommitCounter>,
    ) -> crate::backend::renderer::utils::DamageSet<i32, Physical> {
        panic!("tracker collected legacy damage")
    }
    fn opaque_regions(&self, _: Scale<f64>) -> crate::backend::renderer::utils::OpaqueRegions<i32, Physical> {
        panic!("tracker collected legacy opaque proof")
    }
    fn visit_damage_since(
        &self,
        _: Scale<f64>,
        commit: Option<CommitCounter>,
        visit: &mut dyn FnMut(Rectangle<i32, Physical>),
    ) {
        if commit == Some(self.current_commit()) {
            return;
        }
        for index in 0..96 {
            visit(Rectangle::new(
                ((index % 12) * 4, (index / 12) * 4).into(),
                (3, 3).into(),
            ));
        }
    }
    fn visit_opaque_regions(&self, _: Scale<f64>, visit: &mut dyn FnMut(Rectangle<i32, Physical>)) {
        for index in 0..96 {
            visit(Rectangle::new(
                ((index % 12) * 4, (index / 12) * 4).into(),
                (3, 3).into(),
            ));
        }
    }
}

#[test]
fn workspace_consumes_large_source_visitors_and_preserves_no_change() {
    let elements = [StreamedRectangles { id: Id::new() }];
    for policy in [
        DamageStoragePolicy::Exact,
        DamageStoragePolicy::ConservativeFullOutput,
    ] {
        let mut tracker = OutputDamageTracker::new((100, 100), 1.0, Transform::Normal);
        tracker
            .prepare_frame_storage_with_policy(1, 1024, 2, policy)
            .unwrap();
        let (damage, state) = tracker.damage_output(0, &elements).unwrap();
        assert!(damage.is_some());
        assert_eq!(
            state
                .element_render_state(elements[0].id().clone())
                .unwrap()
                .visible_area,
            64 * 64
        );
        drop(state);
        assert!(tracker.damage_output(1, &elements).unwrap().0.is_none());
    }
}

#[test]
fn a_streamed_region_overflow_defers_before_installing_partial_source_state() {
    let elements = [StreamedRectangles { id: Id::new() }];
    let mut tracker = OutputDamageTracker::new((100, 100), 1.0, Transform::Normal);
    tracker.prepare_frame_storage(1, 16, 2).unwrap();
    assert!(matches!(
        tracker.damage_output(0, &elements),
        Err(DamageOutputError::WorkspaceCapacity(_))
    ));
    assert!(tracker.last_state.elements.is_empty());
    assert!(tracker.opaque_regions.len() <= 16);
}

#[cfg(feature = "renderer_vulkan")]
#[test]
fn repeated_workspace_source_visits_have_zero_heap_operations() {
    let elements = [StreamedRectangles { id: Id::new() }];
    let mut tracker = OutputDamageTracker::new((100, 100), 1.0, Transform::Normal);
    tracker.prepare_frame_storage(1, 1024, 2).unwrap();
    drop(tracker.damage_output(0, &elements).unwrap());
    let (count, operations) = crate::backend::renderer::vulkan::storage_heap_probe::measure(|| {
        let mut count = 0;
        for _ in 0..120 {
            let (damage, state) = tracker.damage_output(1, &elements).unwrap();
            assert!(damage.is_none());
            count += usize::from(state.element_was_presented(elements[0].id().clone()));
        }
        count
    });
    assert_eq!(count, 120);
    assert_eq!(operations, [0; 4]);
}
