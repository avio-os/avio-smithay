//! Actual descriptor cache and native dispatch; no GPU pixel claim.
use super::super::device_handle::retirement_tests::{device, drain_until_parent, next, Operation};
use super::*;
use ash::vk::Handle;

fn cold_state() -> (
    DescriptorState,
    std::sync::mpsc::Receiver<(Operation, std::thread::ThreadId)>,
) {
    let (device, events) = device();
    let mut samplers = IndexMap::new();
    samplers.insert(TextureSampler::LINEAR, vk::Sampler::null());
    samplers.insert(TextureSampler::NEAREST, vk::Sampler::null());
    let mut last_submissions = HashMap::with_capacity(4);
    let mut free_sets = Vec::with_capacity(4);
    for id in 1..=4 {
        let set = vk::DescriptorSet::from_raw(id);
        free_sets.push(set);
        last_submissions.insert(set, None);
    }
    (
        DescriptorState {
            device,
            texture_layout: vk::DescriptorSetLayout::null(),
            texture_samplers: samplers,
            pools: Vec::new(),
            texture_sets: IndexMap::with_capacity(4),
            retired_sets: VecDeque::with_capacity(4),
            free_sets,
            last_submissions,
            recording_sets: HashSet::with_capacity(4),
            cache_target: 2,
            page_size: 4,
            max_sets: 4,
            cache_stats: Default::default(),
            arena_high_water_sets: 0,
            arena_growth_count: 0,
            arena_deferred_count: 0,
        },
        events,
    )
}

#[test]
fn recycled_native_handle_has_a_new_descriptor_and_keeps_old_submission_custody() {
    let (mut state, events) = cold_state();
    let view = vk::ImageView::from_raw(99);
    let old = state.device.reserve_image_incarnation().unwrap();
    let new = state.device.reserve_image_incarnation().unwrap();
    let old_set = state
        .texture_descriptor_set(view, old, TextureSampler::LINEAR)
        .unwrap();
    assert_eq!(next(&events).0, Operation::DescriptorWrite(old_set.as_raw()));
    state.commit_submission(SubmissionId::for_tests(7));
    let new_set = state
        .texture_descriptor_set(view, new, TextureSampler::LINEAR)
        .unwrap();
    assert_ne!(new_set, old_set);
    assert_eq!(next(&events).0, Operation::DescriptorWrite(new_set.as_raw()));
    assert_eq!(state.retired_sets.len(), 1);
    assert!(!state.is_rewriteable(old_set));
    assert_eq!(state.texture_sets.len(), 1);
    let repeated = state
        .texture_descriptor_set(view, new, TextureSampler::LINEAR)
        .unwrap();
    assert_eq!(repeated, new_set);
    assert!(
        events.try_recv().is_err(),
        "unchanged incarnation is a real warm hit"
    );
    state.device.note_submission_completed(SubmissionId::for_tests(7));
    state.reclaim_rewriteable_sets();
    assert!(state.free_sets.contains(&old_set));
    assert!(
        !state.free_sets.contains(&new_set),
        "current recording still borrows the new set"
    );
    state.abort_recording();
    drop(state);
    drain_until_parent(&events);
}

#[test]
fn incarnation_replacement_preserves_an_unsubmitted_recording() {
    let (mut state, events) = cold_state();
    let view = vk::ImageView::from_raw(100);
    let old = state.device.reserve_image_incarnation().unwrap();
    let new = state.device.reserve_image_incarnation().unwrap();
    let old_set = state
        .texture_descriptor_set(view, old, TextureSampler::LINEAR)
        .unwrap();
    let new_set = state
        .texture_descriptor_set(view, new, TextureSampler::LINEAR)
        .unwrap();
    assert_ne!(old_set, new_set);
    state.reclaim_rewriteable_sets();
    assert_eq!(state.retired_sets.len(), 1);
    state.abort_recording();
    assert!(state.free_sets.contains(&old_set));
    drop(state);
    drain_until_parent(&events);
}

#[test]
fn dormant_context_has_no_per_retirement_backlog_and_warm_metadata_is_bounded() {
    let (mut state, events) = cold_state();
    let view = vk::ImageView::from_raw(101);
    let old = state.device.reserve_image_incarnation().unwrap();
    // Seed the actual cache once; no cold notification subscription exists.
    state
        .texture_descriptor_set(view, old, TextureSampler::LINEAR)
        .unwrap();
    state.abort_recording();
    let (_, heap) = super::super::storage_heap_probe::measure(|| {
        for _ in 0..120_000 {
            let new = state.device.reserve_image_incarnation().unwrap();
            state.retire_previous_incarnations(view, new);
            state.reclaim_rewriteable_sets();
            assert!(state.retired_sets.is_empty());
            let set = state.free_sets.pop().unwrap();
            state.texture_sets.insert(
                TextureDescriptorKey {
                    image_view: view,
                    incarnation: new,
                    sampler: TextureSampler::LINEAR,
                },
                CachedTextureSet { set },
            );
            assert_eq!(state.texture_sets.len(), 1);
            assert_eq!(state.free_sets.len(), 3);
        }
    });
    assert_eq!(
        heap, [0; 4],
        "actual bounded key replacement/return cannot allocate or free"
    );
    drop(state);
    drain_until_parent(&events);
}

#[test]
fn shared_image_incarnations_reject_exhaustion_without_reusing_native_handles() {
    let (device, events) = device();
    let first = device.reserve_image_incarnation().unwrap();
    let second = device.reserve_image_incarnation().unwrap();
    assert_ne!(first, second);
    device
        .offscreen_ids()
        .store((1u64 << 62) - 1, std::sync::atomic::Ordering::Release);
    assert!(device.reserve_image_incarnation().is_err());
    assert!(device.reserve_image_incarnation().is_err());
    assert_eq!(
        device.offscreen_ids().load(std::sync::atomic::Ordering::Acquire),
        (1u64 << 62) - 1
    );
    drop(device);
    drain_until_parent(&events);
}
