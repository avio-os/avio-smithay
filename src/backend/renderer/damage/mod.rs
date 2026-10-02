//! Helper for effective output damage tracking
//!
//! # Why use this implementation
//!
//! The [`OutputDamageTracker`] in combination with the [`RenderElement`] trait
//! can help you to reduce resource consumption by tracking what elements have
//! been damaged and only redraw the damaged parts on an output.
//!
//! It does so by keeping track of the last used [`CommitCounter`] for all provided
//! [`RenderElement`]s and queries the element for new damage on each call to [`render_output`](OutputDamageTracker::render_output) or [`damage_output`](OutputDamageTracker::damage_output).
//!
//! Additionally the damage tracker will automatically generate damage in the following situations:
//! - Current geometry for elements entering the output
//! - Current and last known geometry for moved elements (includes z-index changes)
//! - Last known geometry for elements no longer present
//!
//! Elements fully occluded by opaque regions as defined by elements higher in the stack are skipped.
//! The actual action taken by the damage tracker can be inspected from the returned [`RenderElementStates`].
//!
//! You can initialize it with a static output by using [`OutputDamageTracker::new`] or
//! allow it to track a specific [`Output`] with [`OutputDamageTracker::from_output`].
//!
//! See the [`renderer::element`](crate::backend::renderer::element) module for more information
//! about how to use [`RenderElement`].
//!
//! # How to use it
//!
//! ```no_run
//! # use smithay::{
//! #     backend::renderer::{Color32F, DebugFlags, Frame, ImportMem, Renderer, Texture, TextureFilter, sync::SyncPoint, test::{DummyRenderer, DummyFramebuffer}},
//! #     utils::{Buffer, Physical, Rectangle, Size},
//! # };
//! use smithay::{
//!     backend::{
//!         allocator::Fourcc,
//!         renderer::{
//!             damage::OutputDamageTracker,
//!             element::{
//!                 Kind,
//!                 memory::{MemoryRenderBuffer, MemoryRenderBufferRenderElement},
//!             }
//!         },
//!     },
//!     utils::{Point, Transform},
//! };
//! use std::time::{Duration, Instant};
//!
//! const WIDTH: i32 = 10;
//! const HEIGHT: i32 = 10;
//! # let mut renderer = DummyRenderer::default();
//! # let mut framebuffer = DummyFramebuffer;
//! # let buffer_age = 0;
//!
//! // Initialize a new damage tracker for a static output
//! let mut damage_tracker = OutputDamageTracker::new((800, 600), 1.0, Transform::Normal);
//!
//! // Initialize a buffer to render
//! let mut memory_buffer = MemoryRenderBuffer::new(Fourcc::Argb8888, (WIDTH, HEIGHT), 1, Transform::Normal, None);
//!
//! let mut last_update = Instant::now();
//!
//! loop {
//!     let now = Instant::now();
//!     if now.duration_since(last_update) >= Duration::from_secs(3) {
//!         let mut render_context = memory_buffer.render();
//!
//!         render_context.draw(|_buffer| {
//!             // Update the changed parts of the buffer
//!
//!             // Return the updated parts
//!             Result::<_, ()>::Ok(vec![Rectangle::from_size((WIDTH, HEIGHT).into())])
//!         });
//!
//!         last_update = now;
//!     }
//!
//!     // Create a render element from the buffer
//!     let location = Point::from((100.0, 100.0));
//!     let render_element =
//!         MemoryRenderBufferRenderElement::from_buffer(&mut renderer, location, &memory_buffer, None, None, None, Kind::Unspecified)
//!         .expect("Failed to upload memory to gpu");
//!
//!     // Render the output
//!     damage_tracker
//!         .render_output(
//!             &mut renderer,
//!             &mut framebuffer,
//!             buffer_age,
//!             &[render_element],
//!             [0.8, 0.8, 0.9, 1.0],
//!         )
//!         .expect("failed to render the output");
//! }
//! ```

use std::{
    collections::{HashMap, VecDeque},
    ops::Range,
};
#[cfg(feature = "backend_drm")]
use std::{os::fd::OwnedFd, sync::Arc};

use indexmap::IndexMap;
use tracing::{info_span, instrument, trace};

use crate::{
    backend::renderer::{element::RenderElementPresentationState, Frame},
    output::{Output, OutputModeSource, OutputNoMode},
    utils::{Buffer as BufferCoords, Physical, Rectangle, Scale, Size, Transform},
};

#[cfg(all(feature = "backend_drm", feature = "wayland_frontend"))]
use super::{element::UnderlyingStorage, utils::Buffer as WaylandBuffer};
use super::{
    element::{
        Element, ElementSource, FrameWorkspaceError, FramebufferCapturePolicy, Id, Kind, RenderElement,
        RenderElementState, RenderElementStates, StateMapBank,
    },
    sync::SyncPoint,
    utils::CommitCounter,
    Color32F,
};

use super::{Renderer, Texture};

mod shaper;
pub(crate) mod workspace;

#[cfg(test)]
mod storage_tests;

use shaper::DamageShaper;

const MAX_AGE: usize = 4;

/// Cold-declared damage workspace policy. Exact mode retains ordinary partial
/// repaint; conservative mode bounds repair storage without losing contributors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DamageStoragePolicy {
    /// Preserve the existing occlusion and partial-damage optimizations.
    #[default]
    Exact,
    /// Detect unchanged scenes normally, but repair changed/old targets fully.
    /// Every intersecting contributor draws; reported visibility remains the
    /// exact visible union area rather than claiming occluded pixels visible.
    ConservativeFullOutput,
}

/// Current retained damage inventory for checked cold storage admission.
#[derive(Debug, Clone, Copy, Default)]
pub struct RetainedDamageStorage {
    /// Instances from the last successfully rendered scene, including duplicates.
    pub elements: usize,
    /// Last target's declared opaque rectangle inventory.
    pub opaque_rectangles: usize,
    /// Rectangle seeds considered by the supported target-age restoration.
    pub history_rectangles: usize,
    /// Maximum older history entries considered during restoration.
    pub history_entries: usize,
}

#[derive(Debug, Clone, Copy)]
struct ElementInstanceState {
    last_src: Rectangle<f64, BufferCoords>,
    last_geometry: Rectangle<i32, Physical>,
    last_transform: Transform,
    last_alpha: f32,
    last_z_index: usize,
    last_is_framebuffer_effect: bool,
    last_effect_regions: Option<super::element::FramebufferEffectRegions>,
}

impl ElementInstanceState {
    #[inline]
    fn matches(
        &self,
        src: Rectangle<f64, BufferCoords>,
        geometry: Rectangle<i32, Physical>,
        transform: Transform,
        alpha: f32,
        z_index: usize,
        is_framebuffer_effect: bool,
        effect_regions: Option<super::element::FramebufferEffectRegions>,
    ) -> bool {
        self.last_src == src
            && self.last_geometry == geometry
            && self.last_transform == transform
            && self.last_alpha == alpha
            && self.last_z_index == z_index
            && self.last_is_framebuffer_effect == is_framebuffer_effect
            && self.last_effect_regions == effect_regions
    }
}

#[derive(Debug, Clone)]
struct ElementState {
    last_commit: CommitCounter,
    first_instance: usize,
}

#[derive(Debug, Clone, Copy)]
struct ElementInstanceRecord {
    state: ElementInstanceState,
    next: Option<usize>,
}

impl ElementState {
    fn instances<'a>(
        &self,
        records: &'a [ElementInstanceRecord],
    ) -> impl Iterator<Item = &'a ElementInstanceState> {
        std::iter::successors(Some(self.first_instance), |index| records[*index].next)
            .map(|index| &records[index].state)
    }

    #[inline]
    fn instance_matches(
        &self,
        records: &[ElementInstanceRecord],
        src: Rectangle<f64, BufferCoords>,
        geometry: Rectangle<i32, Physical>,
        transform: Transform,
        alpha: f32,
        z_index: usize,
        is_framebuffer_effect: bool,
        effect_regions: Option<super::element::FramebufferEffectRegions>,
    ) -> bool {
        self.instances(records).any(|instance| {
            instance.matches(
                src,
                geometry,
                transform,
                alpha,
                z_index,
                is_framebuffer_effect,
                effect_regions,
            )
        })
    }
}

#[derive(Debug, Default)]
struct RendererState {
    transform: Option<Transform>,
    size: Option<Size<i32, Physical>>,
    elements: IndexMap<Id, ElementState>,
    instances: Vec<ElementInstanceRecord>,
    old_damage: VecDeque<Vec<Rectangle<i32, Physical>>>,
    opaque_regions: Vec<Rectangle<i32, Physical>>,
    clear_color: Option<Color32F>,
}

/// Damage tracker for a single output
#[derive(Debug)]
pub struct OutputDamageTracker {
    mode: OutputModeSource,
    last_state: RendererState,
    damage_shaper: DamageShaper,
    damage_summary: OutputDamageSummary,
    damage: Vec<Rectangle<i32, Physical>>,
    element_damage: Vec<Rectangle<i32, Physical>>,
    opaque_regions: Vec<Rectangle<i32, Physical>>,
    opaque_regions_index: Vec<Range<usize>>,
    element_damage_index: Vec<usize>,
    element_opaque_regions: Vec<Rectangle<i32, Physical>>,
    element_visible_area_workhouse: Vec<Rectangle<i32, Physical>>,
    visibility_opaque_regions: Vec<Rectangle<i32, Physical>>,
    render_indices: Vec<usize>,
    capture_support_damage: Vec<Rectangle<i32, Physical>>,
    history_spares: Vec<Vec<Rectangle<i32, Physical>>>,
    new_damage: Vec<Rectangle<i32, Physical>>,
    state_bank: Option<StateMapBank>,
    storage_capacity: Option<(usize, usize)>,
    storage_policy: DamageStoragePolicy,
    span: tracing::Span,
}

/// A renderer error that can self-report whether it represents an
/// unrecoverable loss of the underlying graphics device.
///
/// Generic error wrappers (such as [`Error`] and the DRM compositor's
/// `RenderFrameError`) cannot inspect the concrete renderer error they carry.
/// Implementing this trait on a renderer error lets those wrappers forward a
/// typed device-loss query instead of forcing callers to string-match a
/// flattened message. Renderers that have no notion of device loss may use the
/// default implementation, which always returns `false`.
pub trait MaybeDeviceLost {
    /// Returns `true` when this error represents an unrecoverable device loss
    /// (the renderer must be recreated), as opposed to a temporary or
    /// allocation failure.
    fn is_device_lost(&self) -> bool {
        false
    }

    /// Native work may have been submitted without an observable retirement
    /// edge. Its context must stop admission and retain the exact resources;
    /// this does not itself prove that the logical device was lost.
    fn is_completion_unobservable(&self) -> bool {
        false
    }
}

/// Errors thrown by [`OutputDamageTracker::render_output`]
#[derive(thiserror::Error)]
pub enum Error<E: std::error::Error> {
    /// The provided [`Renderer`] returned an error
    #[error(transparent)]
    Rendering(E),
    /// Earlier segments submitted native reads, but this outer frame failed
    /// before producing an edge covering all of its sampled sources.
    #[error("partial-frame completion is unobservable: {0}")]
    PartialFrameCompletionUnobservable(E),
    /// CPU workspace refusal after an earlier segment submitted native reads.
    /// Capacity is retained as the cause, but cannot authorize source release.
    #[error("partial-frame completion is unobservable: {0}")]
    PartialFrameWorkspaceCompletionUnobservable(FrameWorkspaceError),
    /// Cold-prepared CPU workspace or retained receipt slots are exhausted.
    #[error(transparent)]
    WorkspaceCapacity(#[from] FrameWorkspaceError),
    /// Rendering sampled a Wayland buffer but produced no observable completion edge.
    ///
    /// The caller must retain every sampled source until the renderer epoch is
    /// torn down. Waiting here would block the compositor's render path, while
    /// treating the draw as complete could release a client buffer too early.
    #[error("Wayland buffer completion is unobservable; renderer epoch teardown required")]
    WaylandCompletionUnobservable,
    /// The given [`Output`] has no mode set
    #[error(transparent)]
    OutputNoMode(#[from] OutputNoMode),
}

impl<E: std::error::Error> Error<E> {
    fn from_frame_rendering(error: E, previous_submission: bool) -> Self {
        if previous_submission {
            Self::PartialFrameCompletionUnobservable(error)
        } else {
            Self::Rendering(error)
        }
    }
    fn from_frame_workspace(error: FrameWorkspaceError, previous_submission: bool) -> Self {
        if previous_submission {
            Self::PartialFrameWorkspaceCompletionUnobservable(error)
        } else {
            Self::WorkspaceCapacity(error)
        }
    }
}

impl<E: std::error::Error + MaybeDeviceLost> Error<E> {
    /// Returns `true` when this error was caused by an unrecoverable loss of
    /// the rendering device (see [`MaybeDeviceLost`]).
    pub fn is_device_lost(&self) -> bool {
        match self {
            Error::Rendering(err) | Error::PartialFrameCompletionUnobservable(err) => err.is_device_lost(),
            Error::WaylandCompletionUnobservable => false,
            Error::OutputNoMode(_)
            | Error::WorkspaceCapacity(_)
            | Error::PartialFrameWorkspaceCompletionUnobservable(_) => false,
        }
    }

    /// Returns `true` when rendering may have sampled a Wayland buffer but no
    /// exact completion edge can prove when that read finished.
    pub fn is_wayland_completion_unobservable(&self) -> bool {
        matches!(self, Error::WaylandCompletionUnobservable)
    }

    /// Whether any sampled resources lack an exact retirement edge.
    pub fn is_completion_unobservable(&self) -> bool {
        match self {
            Self::Rendering(error) => error.is_completion_unobservable(),
            Self::WaylandCompletionUnobservable
            | Self::PartialFrameCompletionUnobservable(_)
            | Self::PartialFrameWorkspaceCompletionUnobservable(_) => true,
            Self::OutputNoMode(_) | Self::WorkspaceCapacity(_) => false,
        }
    }
}

#[cfg(all(test, feature = "renderer_vulkan"))]
mod partial_frame_tests;

#[cfg(test)]
mod completion_error_tests {
    use super::{Error, MaybeDeviceLost};

    #[derive(Debug, thiserror::Error)]
    #[error("test renderer error")]
    struct TestRendererError;

    impl MaybeDeviceLost for TestRendererError {}

    #[test]
    fn unobservable_wayland_completion_is_typed_and_not_device_loss() {
        let error = Error::<TestRendererError>::WaylandCompletionUnobservable;
        assert!(error.is_wayland_completion_unobservable());
        assert!(error.is_completion_unobservable());
        assert!(!error.is_device_lost());
    }

    #[cfg(feature = "renderer_vulkan")]
    #[test]
    fn native_completion_unknown_survives_damage_wrapping_without_device_loss() {
        use crate::backend::renderer::vulkan::VulkanRendererError;
        let failure = Error::Rendering(VulkanRendererError::CommandCompletionUnavailable);
        assert!(failure.is_completion_unobservable());
        assert!(!failure.is_wayland_completion_unobservable());
        assert!(!failure.is_device_lost());
        let pressure = Error::Rendering(VulkanRendererError::CommandCapacityExhausted { slots: 64 });
        assert!(!pressure.is_completion_unobservable());
        assert!(!pressure.is_device_lost());
    }
}

/// Damage-only failures, including cold workspace admission.
#[derive(Debug, thiserror::Error)]
pub enum DamageOutputError {
    /// Output mode is unavailable.
    #[error(transparent)]
    OutputNoMode(#[from] OutputNoMode),
    /// A prepared workspace or retained receipt bank is exhausted.
    #[error(transparent)]
    WorkspaceCapacity(#[from] FrameWorkspaceError),
}

/// Represents the result from rendering the output
#[derive(Debug)]
pub struct RenderOutputResult<'a> {
    /// Holds the sync point of the rendering operation
    pub sync: SyncPoint,
    /// The render completion exported exactly once for all asynchronous
    /// consumers (KMS and Wayland buffer-release ownership).
    #[cfg(feature = "backend_drm")]
    shared_sync_file: Option<Arc<OwnedFd>>,
    /// Holds the damage from the rendering operation
    pub damage: Option<&'a Vec<Rectangle<i32, Physical>>>,
    /// Explains why the damage tracker decided this output needed repainting.
    pub damage_summary: OutputDamageSummary,
    /// Holds the render element states
    pub states: RenderElementStates,
}

/// Compact attribution for the damage produced by one output damage-tracker pass.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct OutputDamageSummary {
    /// Buffer age supplied by the target swapchain.
    pub buffer_age: usize,
    /// Element-reported damage before buffer-age expansion.
    pub element_damage_rect_count: usize,
    /// Total area of element-reported damage before buffer-age expansion.
    pub element_damage_area: u64,
    /// Number of elements that reported direct buffer/content damage.
    pub element_damage_element_count: usize,
    /// Element index with the largest direct buffer/content damage contribution.
    pub top_element_damage_index: Option<usize>,
    /// Kind of the element with the largest direct buffer/content damage contribution.
    pub top_element_damage_kind: Option<Kind>,
    /// Rectangle count reported by the element with the largest direct buffer/content damage contribution.
    pub top_element_damage_rect_count: usize,
    /// Area reported by the element with the largest direct buffer/content damage contribution.
    pub top_element_damage_area: u64,
    /// Current geometry area of the element with the largest direct buffer/content damage contribution.
    pub top_element_damage_geometry_area: u64,
    /// Damage generated because an element's geometry, source, transform, alpha,
    /// or z-order changed relative to the previous rendered output-layer frame.
    pub element_state_change_count: usize,
    /// Total area damaged by element geometry, source, transform, alpha, or z-order changes.
    pub element_state_change_damage_area: u64,
    /// Element index with the largest state-change damage contribution.
    pub top_element_state_change_index: Option<usize>,
    /// Kind of the element with the largest state-change damage contribution.
    pub top_element_state_change_kind: Option<Kind>,
    /// Area damaged by the largest element state-change contribution.
    pub top_element_state_change_area: u64,
    /// Current geometry area of the element with the largest state-change contribution.
    pub top_element_state_change_geometry_area: u64,
    /// Damage generated because an element from the previous frame disappeared.
    pub element_gone_count: usize,
    /// Total area damaged by elements that disappeared since the previous rendered frame.
    pub element_gone_damage_area: u64,
    /// Damage generated because regions that were opaque last frame are no
    /// longer covered by opaque content this frame.
    pub opaque_uncovered_rect_count: usize,
    /// Total area damaged by previously opaque regions that became uncovered.
    pub opaque_uncovered_area: u64,
    /// True when output size, transform, or clear color forced a full repaint.
    pub output_state_full_damage: bool,
    /// True when the target buffer age was unavailable or too old, forcing a
    /// full repaint of the target buffer.
    pub buffer_age_full_damage: bool,
    /// Final shaped damage submitted to the renderer.
    pub final_damage_rect_count: usize,
    /// Total final shaped damage area submitted to the renderer.
    pub final_damage_area: u64,
}

impl RenderOutputResult<'_> {
    fn skipped(states: RenderElementStates, damage_summary: OutputDamageSummary) -> Self {
        Self {
            sync: SyncPoint::signaled(),
            #[cfg(feature = "backend_drm")]
            shared_sync_file: None,
            damage: None,
            damage_summary,
            states,
        }
    }

    /// Share the one cached sync_file export for this render operation.
    /// Re-exporting some renderer fences is destructive, so all consumers
    /// clone this descriptor instead of independently exporting `sync`.
    #[cfg(feature = "backend_drm")]
    pub(crate) fn shared_sync_file(&self) -> Option<Arc<OwnedFd>> {
        self.shared_sync_file.clone()
    }
}

fn rect_area(rect: Rectangle<i32, Physical>) -> u64 {
    let width = rect.size.w.max(0) as u64;
    let height = rect.size.h.max(0) as u64;
    width.saturating_mul(height)
}

fn rects_area(rects: &[Rectangle<i32, Physical>]) -> u64 {
    rects
        .iter()
        .copied()
        .fold(0u64, |area, rect| area.saturating_add(rect_area(rect)))
}

/// Expand damage through framebuffer-read dependencies in accumulator order.
///
/// Elements arrive front-to-back, so walking them in reverse visits the
/// lowest effect first. Its paint damage is appended before an upper effect
/// is considered, making nested effects transitive without a second scene
/// interpretation. `damage_floor` limits the trigger set to damage introduced
/// by the current phase (new scene damage or buffer-age restoration).
#[allow(clippy::too_many_arguments)]
fn propagate_framebuffer_effect_damage<'e, S: ElementSource + ?Sized>(
    damage: &mut Vec<Rectangle<i32, Physical>>,
    opaque_regions: &mut [Rectangle<i32, Physical>],
    opaque_regions_index: &[Range<usize>],
    element_damage_index: &[usize],
    elements: &'e S,
    render_indices: &[usize],
    capture_support_damage: &mut Vec<Rectangle<i32, Physical>>,
    rectangle_limit: Option<usize>,
    states: &mut RenderElementStates,
    output_scale: Scale<f64>,
    output_geo: Rectangle<i32, Physical>,
    damage_floor: usize,
    force_redraw: bool,
) -> Result<(), FrameWorkspaceError> {
    capture_support_damage.clear();
    for (z_index, element) in render_indices
        .iter()
        .map(|index| elements.element(*index))
        .enumerate()
        .filter(|(_, element)| element.is_framebuffer_effect())
        .rev()
    {
        let Some(regions) = element.framebuffer_effect_regions(output_scale) else {
            continue;
        };
        let read_area = regions.backdrop_read_area.intersection(output_geo);
        let paint_area = regions.paint_area.intersection(output_geo);
        let trigger_start = element_damage_index[z_index].max(damage_floor);
        let backdrop_affected = read_area.is_some_and(|read| {
            damage
                .iter()
                .skip(trigger_start)
                .any(|candidate| candidate.overlaps(read))
        });
        // EveryDraw does not synthesize a frame. Once existing damage will
        // draw the effect, however, its attempt-local capture must rebuild the
        // complete declared read support before the effect is drawn.
        let every_draw_affected = matches!(
            element.framebuffer_capture_policy(),
            FramebufferCapturePolicy::EveryDraw
        ) && paint_area.is_some_and(|paint| {
            damage
                .iter()
                .skip(damage_floor)
                .any(|candidate| candidate.overlaps(paint))
        });
        let affected = force_redraw || backdrop_affected || every_draw_affected;
        if !affected {
            continue;
        }

        let Some(state) = states.states.get_mut(element.id()) else {
            continue;
        };
        state.needs_capture = true;
        if let Some(read) = read_area {
            workspace::extend(capture_support_damage, [read], rectangle_limit)?;

            // An opaque element above the effect normally suppresses lower
            // drawing. The effect captures before either that upper element
            // or the effect itself exists, so reopen every overlapping
            // occluder through the effect's own range for this support region.
            let opaque_above_end = opaque_regions_index[z_index].end;
            for opaque in opaque_regions.iter_mut().take(opaque_above_end) {
                if opaque.overlaps(read) {
                    *opaque = Rectangle::default();
                }
            }
        }
        if let Some(paint) = paint_area {
            workspace::extend(damage, [paint], rectangle_limit)?;
        }
    }
    workspace::extend(damage, capture_support_damage.iter().copied(), rectangle_limit)
}

impl<E: std::error::Error> std::fmt::Debug for Error<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Rendering(err) | Error::PartialFrameCompletionUnobservable(err) => {
                std::fmt::Debug::fmt(err, f)
            }
            Error::WaylandCompletionUnobservable => f.write_str("WaylandCompletionUnobservable"),
            Error::OutputNoMode(err) => std::fmt::Debug::fmt(err, f),
            Error::WorkspaceCapacity(err) | Error::PartialFrameWorkspaceCompletionUnobservable(err) => {
                std::fmt::Debug::fmt(err, f)
            }
        }
    }
}

impl OutputDamageTracker {
    /// Cold-admit CPU workspace and independent returned receipt slots.
    /// Existing receipts keep their previous bank alive across reconfiguration.
    pub fn prepare_frame_storage(
        &mut self,
        elements: usize,
        rectangles: usize,
        receipts: usize,
    ) -> Result<(), FrameWorkspaceError> {
        self.prepare_frame_storage_with_policy(elements, rectangles, receipts, DamageStoragePolicy::Exact)
    }

    /// Cold-admit explicit repair policy; policy changes invalidate target
    /// history before the next native draw, never while a pass is recording.
    pub fn prepare_frame_storage_with_policy(
        &mut self,
        elements: usize,
        rectangles: usize,
        receipts: usize,
        policy: DamageStoragePolicy,
    ) -> Result<(), FrameWorkspaceError> {
        self.prepare_frame_storage_with_return_wakeup(elements, rectangles, receipts, policy, None)
    }

    pub(crate) fn prepare_frame_storage_with_return_wakeup(
        &mut self,
        elements: usize,
        rectangles: usize,
        receipts: usize,
        policy: DamageStoragePolicy,
        returned: Option<crate::backend::renderer::element::StateReceiptReturnWakeup>,
    ) -> Result<(), FrameWorkspaceError> {
        if rectangles == 0 || receipts == 0 {
            return Err(FrameWorkspaceError {
                resource: "cold frame storage",
                required: 1,
                capacity: 0,
            });
        }
        self.state_bank = Some(StateMapBank::new_with_wakeup(elements, receipts, returned));
        self.storage_capacity = Some((elements, rectangles));
        if self.storage_policy != policy {
            self.reset_history();
        }
        self.storage_policy = policy;
        workspace::reserve(&mut self.render_indices, elements);
        workspace::reserve(&mut self.opaque_regions_index, elements);
        workspace::reserve(&mut self.element_damage_index, elements);
        self.last_state
            .elements
            .reserve(elements.saturating_sub(self.last_state.elements.len()));
        workspace::reserve(&mut self.last_state.instances, elements);
        for vec in [
            &mut self.damage,
            &mut self.element_damage,
            &mut self.opaque_regions,
            &mut self.element_opaque_regions,
            &mut self.element_visible_area_workhouse,
            &mut self.visibility_opaque_regions,
            &mut self.capture_support_damage,
            &mut self.new_damage,
            &mut self.last_state.opaque_regions,
        ] {
            workspace::reserve(vec, rectangles);
        }
        self.last_state
            .old_damage
            .reserve((MAX_AGE + 1).saturating_sub(self.last_state.old_damage.len()));
        workspace::reserve(&mut self.history_spares, MAX_AGE + 1);
        for vec in self
            .last_state
            .old_damage
            .iter_mut()
            .chain(self.history_spares.iter_mut())
        {
            workspace::reserve(vec, rectangles);
        }
        while self.last_state.old_damage.len() + self.history_spares.len() < MAX_AGE + 1 {
            self.history_spares.push(Vec::with_capacity(rectangles));
        }
        self.damage_shaper.prepare_storage(rectangles);
        Ok(())
    }

    pub(crate) fn storage_receipts_reclaimable(&self) -> bool {
        self.state_bank.as_ref().is_none_or(StateMapBank::is_reclaimable)
    }

    pub(crate) fn set_mode_source_preserving_storage(&mut self, mode: OutputModeSource) {
        if self.mode != mode {
            self.mode = mode;
            self.reset_history();
        }
    }

    /// The exact policy declared by the cold output owner.
    pub fn storage_policy(&self) -> DamageStoragePolicy {
        self.storage_policy
    }

    /// Exact retained inputs; no guessed maximum from SmallVec inline lengths.
    pub fn retained_frame_storage(&self) -> RetainedDamageStorage {
        RetainedDamageStorage {
            elements: self.last_state.instances.len(),
            opaque_rectangles: self.last_state.opaque_regions.len(),
            history_rectangles: self
                .last_state
                .old_damage
                .iter()
                .take(MAX_AGE - 1)
                .map(Vec::len)
                .sum(),
            history_entries: MAX_AGE - 1,
        }
    }

    /// Exact per-rectangle CPU backing and minimum shaper tile side for cold
    /// output admission, including reusable history and scratch vectors.
    pub fn rectangle_storage_layout() -> (usize, i32) {
        let (tile_bytes, side) = shaper::tile_storage_layout();
        // Nine scratch/state vectors, five reusable history vectors and the
        // shaper output vector. Its tile vector has the same admitted count.
        (
            (9 + MAX_AGE + 1 + 1) * std::mem::size_of::<Rectangle<i32, Physical>>() + tile_bytes,
            side,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn damage_output_conservative<'e, S: ElementSource + ?Sized>(
        &mut self,
        age: usize,
        elements: &'e S,
        output_scale: Scale<f64>,
        output_transform: Transform,
        output_geo: Rectangle<i32, Physical>,
        clear_color: Option<Color32F>,
    ) -> Result<RenderElementStates, FrameWorkspaceError> {
        let mut states = self.claim_states(elements.len())?;
        let limit = self.storage_capacity.map(|(_, rectangles)| rectangles);
        self.render_indices.clear();
        self.opaque_regions.clear();
        self.opaque_regions_index.clear();
        self.element_damage_index.clear();
        self.damage.clear();
        self.new_damage.clear();
        self.damage_summary = OutputDamageSummary {
            buffer_age: age,
            ..Default::default()
        };
        let output_changed = self.last_state.size != Some(output_geo.size)
            || self.last_state.transform != Some(output_transform)
            || self.last_state.clear_color != clear_color;
        let mut scene_changed = output_changed;
        self.damage_summary.output_state_full_damage = output_changed;

        for (index, element) in elements.iter().enumerate() {
            let geometry = element.geometry(output_scale);
            let Some(visible_geometry) = geometry.intersection(output_geo) else {
                states
                    .states
                    .entry(element.id().clone())
                    .or_insert_with(RenderElementState::skipped);
                continue;
            };
            let z_index = self.render_indices.len();
            let previous = self.last_state.elements.get(element.id());
            let same_instance = previous.is_some_and(|previous| {
                previous.instance_matches(
                    &self.last_state.instances,
                    element.src(),
                    geometry,
                    element.transform(),
                    element.alpha(),
                    z_index,
                    element.is_framebuffer_effect(),
                    element.framebuffer_effect_regions(output_scale),
                )
            });
            scene_changed |= !same_instance;
            let mut element_damage_count = 0;
            let mut element_damage_area = 0_u64;
            element.visit_damage_since(
                output_scale,
                previous.map(|state| state.last_commit),
                &mut |rect| {
                    element_damage_count += 1;
                    element_damage_area = element_damage_area.saturating_add(rect_area(rect));
                },
            );
            scene_changed |= element_damage_count != 0;
            self.damage_summary.element_state_change_count += usize::from(!same_instance);
            self.damage_summary.element_damage_rect_count += element_damage_count;
            self.damage_summary.element_damage_area = self
                .damage_summary
                .element_damage_area
                .saturating_add(element_damage_area);

            let area = workspace::visible_area(visible_geometry, &self.opaque_regions);
            states
                .states
                .entry(element.id().clone())
                .and_modify(|state| {
                    if state.presentation_state == RenderElementPresentationState::Skipped {
                        *state = RenderElementState::rendered(area);
                    } else {
                        state.visible_area = state.visible_area.saturating_add(area);
                    }
                })
                .or_insert_with(|| RenderElementState::rendered(area));
            // Every intersecting source participates in full repair, including
            // a currently occluded source required by a later prefix capture.
            self.render_indices.push(index);
            self.opaque_regions_index.push(0..0);
            self.element_damage_index.push(0);
            workspace::try_visit(
                |visit| element.visit_opaque_regions(output_scale, visit),
                |mut rect| {
                    rect.loc += geometry.loc;
                    match rect.intersection(output_geo) {
                        Some(rect) => workspace::extend(&mut self.opaque_regions, [rect], limit),
                        None => Ok(()),
                    }
                },
            )?;
        }
        scene_changed |= self.render_indices.len() != self.last_state.instances.len();
        // Complete membership/order comparison covers removals even when a
        // replacement happens to have the same count and geometry.
        scene_changed |= self.last_state.elements.keys().any(|id| {
            !self
                .render_indices
                .iter()
                .any(|index| elements.element(*index).id() == id)
        });
        let age_valid = age > 0 && age <= MAX_AGE && self.last_state.old_damage.len() >= age;
        let old_target_changed = !age_valid
            || self
                .last_state
                .old_damage
                .iter()
                .take(age.saturating_sub(1))
                .any(|damage| !damage.is_empty());
        self.damage_summary.buffer_age_full_damage = !age_valid;
        if scene_changed {
            workspace::extend(&mut self.new_damage, [output_geo], limit)?;
        }
        if scene_changed || old_target_changed {
            workspace::extend(&mut self.damage, [output_geo], limit)?;
        }
        self.opaque_regions.clear();
        self.damage_summary.final_damage_rect_count = self.damage.len();
        self.damage_summary.final_damage_area = rects_area(&self.damage);
        if self.damage.is_empty() {
            return Ok(states);
        }
        for index in &self.render_indices {
            let element = elements.element(*index);
            if element.is_framebuffer_effect() {
                if let Some(state) = states.states.get_mut(element.id()) {
                    state.needs_capture = true;
                }
            }
        }
        self.last_state.elements.clear();
        self.last_state.instances.clear();
        for (z_index, index) in self.render_indices.iter().enumerate() {
            let element = elements.element(*index);
            let instance_index = self.last_state.instances.len();
            let next = self
                .last_state
                .elements
                .get(element.id())
                .map(|state| state.first_instance);
            self.last_state.instances.push(ElementInstanceRecord {
                state: ElementInstanceState {
                    last_src: element.src(),
                    last_geometry: element.geometry(output_scale),
                    last_transform: element.transform(),
                    last_alpha: element.alpha(),
                    last_z_index: z_index,
                    last_is_framebuffer_effect: element.is_framebuffer_effect(),
                    last_effect_regions: element.framebuffer_effect_regions(output_scale),
                },
                next,
            });
            self.last_state.elements.insert(
                element.id().clone(),
                ElementState {
                    last_commit: element.current_commit(),
                    first_instance: instance_index,
                },
            );
        }
        self.last_state.size = Some(output_geo.size);
        self.last_state.transform = Some(output_transform);
        self.last_state.clear_color = clear_color;
        self.last_state.opaque_regions.clear();
        self.reclaim_history(MAX_AGE);
        let mut new_damage = self.history_spares.pop().unwrap_or_default();
        std::mem::swap(&mut self.new_damage, &mut new_damage);
        self.last_state.old_damage.push_front(new_damage);
        Ok(states)
    }

    fn validate_render_rectangles<'e, S: ElementSource + ?Sized>(
        &mut self,
        elements: &'e S,
        scale: Scale<f64>,
    ) -> Result<(), FrameWorkspaceError> {
        let limit = self.storage_capacity.map(|(_, rectangles)| rectangles);
        self.element_damage.clear();
        workspace::extend(&mut self.element_damage, self.damage.iter().copied(), limit)?;
        workspace::subtract(
            &mut self.element_damage,
            self.opaque_regions.iter().copied(),
            limit,
        )?;
        for (z_index, element) in self
            .render_indices
            .iter()
            .rev()
            .map(|index| elements.element(*index))
            .enumerate()
        {
            let geometry = element.geometry(scale);
            self.element_damage.clear();
            workspace::extend(
                &mut self.element_damage,
                self.damage.iter().filter_map(|d| d.intersection(geometry)),
                limit,
            )?;
            let range = self
                .opaque_regions_index
                .iter()
                .rev()
                .nth(z_index)
                .expect("visible element range");
            workspace::subtract(
                &mut self.element_damage,
                self.opaque_regions[..range.start].iter().copied(),
                limit,
            )?;
            self.element_opaque_regions.clear();
            workspace::extend(
                &mut self.element_opaque_regions,
                self.opaque_regions[range.clone()].iter().copied(),
                limit,
            )?;
        }
        Ok(())
    }

    fn claim_states(&mut self, required: usize) -> Result<RenderElementStates, FrameWorkspaceError> {
        if let Some((capacity, _)) = self.storage_capacity {
            if required > capacity {
                return Err(FrameWorkspaceError {
                    resource: "damage element indices",
                    required,
                    capacity,
                });
            }
        } else {
            workspace::reserve(&mut self.render_indices, required);
            self.last_state
                .elements
                .reserve(required.saturating_sub(self.last_state.elements.len()));
            workspace::reserve(&mut self.last_state.instances, required);
        }
        Ok(RenderElementStates {
            states: match &self.state_bank {
                Some(bank) => bank.acquire(required)?,
                None => HashMap::with_capacity(required).into(),
            },
        })
    }

    fn reclaim_history(&mut self, keep: usize) {
        while self.last_state.old_damage.len() > keep {
            self.history_spares
                .push(self.last_state.old_damage.pop_back().expect("damage history"));
        }
    }

    fn reset_history(&mut self) {
        self.last_state.transform = None;
        self.last_state.size = None;
        self.last_state.clear_color = None;
        self.last_state.elements.clear();
        self.last_state.instances.clear();
        self.last_state.opaque_regions.clear();
        self.reclaim_history(0);
    }

    /// Initialize a static [`OutputDamageTracker`]
    pub fn new(
        size: impl Into<Size<i32, Physical>>,
        scale: impl Into<Scale<f64>>,
        transform: Transform,
    ) -> Self {
        Self {
            mode: OutputModeSource::Static {
                size: size.into(),
                scale: scale.into(),
                transform,
            },
            last_state: Default::default(),
            damage_shaper: Default::default(),
            damage_summary: Default::default(),
            damage: Default::default(),
            element_damage: Default::default(),
            opaque_regions: Default::default(),
            opaque_regions_index: Default::default(),
            element_damage_index: Default::default(),
            element_opaque_regions: Default::default(),
            element_visible_area_workhouse: Default::default(),
            visibility_opaque_regions: Default::default(),
            render_indices: Default::default(),
            capture_support_damage: Default::default(),
            history_spares: Default::default(),
            new_damage: Default::default(),
            state_bank: None,
            storage_capacity: None,
            storage_policy: DamageStoragePolicy::Exact,

            span: info_span!("renderer_damage"),
        }
    }

    /// Initialize a new [`OutputDamageTracker`] from an [`Output`]
    ///
    /// The renderer will keep track of changes to the [`Output`]
    /// and handle size and scaling changes automatically on the
    /// next call to [`render_output`](OutputDamageTracker::render_output)
    pub fn from_output(output: &Output) -> Self {
        Self {
            mode: OutputModeSource::Auto(output.clone()),
            damage_shaper: Default::default(),
            damage_summary: Default::default(),
            damage: Default::default(),
            element_damage: Default::default(),
            opaque_regions: Default::default(),
            opaque_regions_index: Default::default(),
            element_damage_index: Default::default(),
            element_opaque_regions: Default::default(),
            element_visible_area_workhouse: Default::default(),
            visibility_opaque_regions: Default::default(),
            render_indices: Default::default(),
            capture_support_damage: Default::default(),
            history_spares: Default::default(),
            new_damage: Default::default(),
            state_bank: None,
            storage_capacity: None,
            storage_policy: DamageStoragePolicy::Exact,

            last_state: Default::default(),
            span: info_span!("renderer_damage", output = output.name()),
        }
    }

    /// Initialize a new [`OutputDamageTracker`] from an [`OutputModeSource`].
    ///
    /// This should only be used when trying to support both static and automatic output mode
    /// sources. For known modes use [`OutputDamageTracker::new`] or
    /// [`OutputDamageTracker::from_output`] instead.
    pub fn from_mode_source(output_mode_source: impl Into<OutputModeSource>) -> Self {
        Self {
            mode: output_mode_source.into(),
            span: info_span!("render_damage"),
            damage_shaper: Default::default(),
            damage_summary: Default::default(),
            damage: Default::default(),
            element_damage: Default::default(),
            element_opaque_regions: Default::default(),
            opaque_regions: Default::default(),
            opaque_regions_index: Default::default(),
            element_damage_index: Default::default(),
            element_visible_area_workhouse: Default::default(),
            visibility_opaque_regions: Default::default(),
            render_indices: Default::default(),
            capture_support_damage: Default::default(),
            history_spares: Default::default(),
            new_damage: Default::default(),
            state_bank: None,
            storage_capacity: None,
            storage_policy: DamageStoragePolicy::Exact,

            last_state: Default::default(),
        }
    }

    /// Get the [`OutputModeSource`] of the [`OutputDamageTracker`]
    pub fn mode(&self) -> &OutputModeSource {
        &self.mode
    }

    /// Render this output with the provided [`Renderer`]
    ///
    /// - `elements` for this output in front-to-back order
    #[instrument(level = "trace", parent = &self.span, skip(renderer, framebuffer, elements, clear_color))]
    #[profiling::function]
    pub fn render_output<E, R>(
        &mut self,
        renderer: &mut R,
        framebuffer: &mut R::Framebuffer<'_>,
        age: usize,
        elements: &[E],
        clear_color: impl Into<Color32F>,
    ) -> Result<RenderOutputResult<'_>, Error<R::Error>>
    where
        E: RenderElement<R>,
        R: Renderer,
        R::TextureId: Texture,
    {
        self.render_output_from(renderer, framebuffer, age, elements, clear_color)
    }

    /// Render an indexed borrowed source without allocating an element list.
    pub(crate) fn render_output_from<'a, 'e, R, S>(
        &'a mut self,
        renderer: &mut R,
        framebuffer: &mut R::Framebuffer<'_>,
        age: usize,
        elements: &'e S,
        clear_color: impl Into<Color32F>,
    ) -> Result<RenderOutputResult<'a>, Error<R::Error>>
    where
        S: ElementSource + ?Sized,
        S::Element<'e>: RenderElement<R>,
        R: Renderer,
        R::TextureId: Texture,
    {
        let clear_color = clear_color.into();
        let (output_size, output_scale, output_transform) =
            std::convert::TryInto::<(Size<i32, Physical>, Scale<f64>, Transform)>::try_into(&self.mode)?;

        // Output transform is specified in surface-rotation, so inversion gives us the
        // render transform for the output itself.
        let output_transform = output_transform.invert();

        // We have to apply to output transform to the output size so that the intersection
        // tests in damage_output_internal produces the correct results and do not crop
        // damage with the wrong size
        let output_geo = Rectangle::from_size(output_transform.transform_size(output_size));

        // This will hold all the damage we need for this rendering step
        let states = self
            .damage_output_internal(
                age,
                elements,
                output_scale,
                output_transform,
                output_geo,
                Some(clear_color),
            )
            .map_err(Error::WorkspaceCapacity)?;

        if self.damage.is_empty() {
            trace!("no damage, skipping rendering");
            return Ok(RenderOutputResult::skipped(states, self.damage_summary));
        }

        trace!(
            "rendering with damage {:?} and opaque regions {:?}",
            self.damage,
            self.opaque_regions
        );

        #[cfg(all(feature = "backend_drm", feature = "wayland_frontend"))]
        let rendered_wayland_buffers = {
            let mut buffers = Vec::<WaylandBuffer>::new();
            for element in self.render_indices.iter().map(|index| elements.element(*index)) {
                let Some(UnderlyingStorage::Wayland(buffer)) = element.sampled_storage(renderer) else {
                    continue;
                };
                if !buffers.iter().any(|existing| existing.same_instance(buffer)) {
                    buffers.push(buffer.clone());
                }
            }
            buffers
        };

        if let Err(error) = self.validate_render_rectangles(elements, output_scale) {
            // Damage calculation advanced its logical history, but no native
            // frame has begun. Retry must repaint the real unchanged target.
            self.last_state.size = None;
            return Err(Error::WorkspaceCapacity(error));
        }

        let render_res = (|| {
            // we have to take the element damage to be able to move it around
            let rectangle_limit = self.storage_capacity.map(|(_, rectangles)| rectangles);
            let mut frame = renderer
                .render(framebuffer, output_size, output_transform)
                .map_err(Error::Rendering)?;

            self.element_damage.clear();
            workspace::extend(
                &mut self.element_damage,
                self.damage.iter().copied(),
                rectangle_limit,
            )
            .map_err(|error| Error::from_frame_workspace(error, frame.completion_unobservable_on_error()))?;
            workspace::subtract(
                &mut self.element_damage,
                self.opaque_regions.iter().copied(),
                rectangle_limit,
            )
            .map_err(|error| Error::from_frame_workspace(error, frame.completion_unobservable_on_error()))?;

            trace!("clearing damage {:?}", self.element_damage);
            frame.clear(clear_color, &self.element_damage).map_err(|error| {
                Error::from_frame_rendering(error, frame.completion_unobservable_on_error())
            })?;

            for (z_index, element) in self
                .render_indices
                .iter()
                .rev()
                .map(|index| elements.element(*index))
                .enumerate()
            {
                let element_id = element.id();
                let element_geometry = element.geometry(output_scale);

                self.element_damage.clear();
                workspace::extend(
                    &mut self.element_damage,
                    self.damage
                        .iter()
                        .filter_map(|d| d.intersection(element_geometry)),
                    rectangle_limit,
                )
                .map_err(|error| {
                    Error::from_frame_workspace(error, frame.completion_unobservable_on_error())
                })?;

                let element_opaque_regions_range =
                    self.opaque_regions_index.iter().rev().nth(z_index).unwrap();
                workspace::subtract(
                    &mut self.element_damage,
                    self.opaque_regions[..element_opaque_regions_range.start]
                        .iter()
                        .copied(),
                    rectangle_limit,
                )
                .map_err(|error| {
                    Error::from_frame_workspace(error, frame.completion_unobservable_on_error())
                })?;
                self.element_damage.iter_mut().for_each(|d| {
                    d.loc -= element_geometry.loc;
                });

                if self.element_damage.is_empty() {
                    trace!(
                        "skipping rendering element {:?} with geometry {:?}, no damage",
                        element_id,
                        element_geometry
                    );
                    continue;
                }

                self.element_opaque_regions.clear();
                workspace::extend(
                    &mut self.element_opaque_regions,
                    self.opaque_regions[element_opaque_regions_range.start..element_opaque_regions_range.end]
                        .iter()
                        .copied()
                        .map(|mut rect| {
                            rect.loc -= element_geometry.loc;
                            rect
                        }),
                    rectangle_limit,
                )
                .map_err(|error| {
                    Error::from_frame_workspace(error, frame.completion_unobservable_on_error())
                })?;

                trace!(
                    "rendering element {:?} with geometry {:?} and damage {:?}",
                    element_id,
                    element_geometry,
                    self.element_damage,
                );

                if states
                    .element_render_state(element_id.clone())
                    .is_some_and(|state| state.needs_capture)
                {
                    let regions = element
                        .framebuffer_effect_regions(output_scale)
                        .expect("framebuffer effect without read/paint regions");
                    element
                        .capture_framebuffer(&mut frame, regions)
                        .map_err(|error| {
                            Error::from_frame_rendering(error, frame.completion_unobservable_on_error())
                        })?;
                }

                element
                    .draw(
                        &mut frame,
                        element.src(),
                        element_geometry,
                        &self.element_damage,
                        &self.element_opaque_regions,
                    )
                    .map_err(|error| {
                        Error::from_frame_rendering(error, frame.completion_unobservable_on_error())
                    })?;
            }

            let previous_submission = frame.completion_unobservable_on_error();
            frame
                .finish()
                .map_err(|error| Error::from_frame_rendering(error, previous_submission))
        })();

        match render_res {
            Ok(sync) => {
                #[cfg(feature = "backend_drm")]
                let shared_sync_file = sync.export().map(Arc::new);
                #[cfg(all(feature = "backend_drm", feature = "wayland_frontend"))]
                if !rendered_wayland_buffers.is_empty() && !sync.is_reached() && shared_sync_file.is_none() {
                    // An asynchronous Wayland-buffer read must leave one exact,
                    // pollable completion identity for release ownership. The
                    // caller already owns the sampled resources; return typed
                    // unobservable custody so it can quarantine them through
                    // renderer-epoch teardown without blocking this path.
                    self.reset_history();
                    return Err(Error::WaylandCompletionUnobservable);
                }
                #[cfg(all(feature = "backend_drm", feature = "wayland_frontend"))]
                for buffer in rendered_wayland_buffers {
                    if !sync.is_reached() {
                        buffer.record_render_completion(
                            shared_sync_file
                                .as_ref()
                                .expect("pending Wayland render completion passed the typed export gate")
                                .clone(),
                        );
                    }
                }
                Ok(RenderOutputResult {
                    sync,
                    #[cfg(feature = "backend_drm")]
                    shared_sync_file,
                    damage: Some(&self.damage),
                    damage_summary: self.damage_summary,
                    states,
                })
            }
            Err(err) => {
                // if the rendering errors on us, we need to be prepared, that this whole buffer was partially updated and thus now unusable.
                // thus clean our old states before returning
                self.reset_history();
                Err(err)
            }
        }
    }

    /// Damage this output and return the damage without actually rendering the difference
    ///
    /// - `elements` for this output in front-to-back order
    #[instrument(level = "trace", parent = &self.span, skip(elements))]
    #[profiling::function]
    pub fn damage_output<'a, 'e, E>(
        &'a mut self,
        age: usize,
        elements: &'e [E],
    ) -> Result<(Option<&'a Vec<Rectangle<i32, Physical>>>, RenderElementStates), DamageOutputError>
    where
        E: Element,
    {
        self.damage_output_from(age, elements)
    }

    /// Calculate damage from an indexed borrowed source.
    pub(crate) fn damage_output_from<'a, 'e, S: ElementSource + ?Sized>(
        &'a mut self,
        age: usize,
        elements: &'e S,
    ) -> Result<(Option<&'a Vec<Rectangle<i32, Physical>>>, RenderElementStates), DamageOutputError> {
        let (output_size, output_scale, output_transform) =
            std::convert::TryInto::<(Size<i32, Physical>, Scale<f64>, Transform)>::try_into(&self.mode)?;

        // Output transform is specified in surface-rotation, so inversion gives us the
        // render transform for the output itself.
        let output_transform = output_transform.invert();

        // We have to apply to output transform to the output size so that the intersection
        // tests in damage_output_internal produces the correct results and do not crop
        // damage with the wrong size
        let output_geo = Rectangle::from_size(output_transform.transform_size(output_size));

        let states = self.damage_output_internal(
            age,
            elements,
            output_scale,
            output_transform,
            output_geo,
            self.last_state.clear_color,
        )?;

        if self.damage.is_empty() {
            Ok((None, states))
        } else {
            Ok((Some(&self.damage), states))
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[profiling::function]
    fn damage_output_internal<'e, S: ElementSource + ?Sized>(
        &mut self,
        age: usize,
        elements: &'e S,
        output_scale: Scale<f64>,
        output_transform: Transform,
        output_geo: Rectangle<i32, Physical>,
        clear_color: Option<Color32F>,
    ) -> Result<RenderElementStates, FrameWorkspaceError> {
        if self.storage_policy == DamageStoragePolicy::ConservativeFullOutput {
            return self.damage_output_conservative(
                age,
                elements,
                output_scale,
                output_transform,
                output_geo,
                clear_color,
            );
        }
        let mut element_render_states = self.claim_states(elements.len())?;
        let rectangle_limit = self.storage_capacity.map(|(_, rectangles)| rectangles);
        self.render_indices.clear();
        self.damage.clear();
        self.damage_summary = OutputDamageSummary {
            buffer_age: age,
            ..OutputDamageSummary::default()
        };
        self.opaque_regions.clear();
        self.opaque_regions_index.clear();
        self.element_damage_index.clear();

        // Preserve the original opacity traversal with no extra copying until
        // a framebuffer effect actually needs a distinct capture view.
        let mut uses_visibility_opaque_regions = false;
        self.visibility_opaque_regions.clear();
        let mut z_index = 0;
        for (element_index, element) in elements.iter().enumerate() {
            let element_id = element.id();
            let element_loc = element.geometry(output_scale).loc;

            // First test if the element overlaps with the output
            // if not we can skip it
            let element_output_geometry = match element.geometry(output_scale).intersection(output_geo) {
                Some(geo) => geo,
                None => continue,
            };

            // Then test if the element is completely hidden behind opaque regions
            self.element_visible_area_workhouse.clear();
            workspace::extend(
                &mut self.element_visible_area_workhouse,
                [element_output_geometry],
                rectangle_limit,
            )?;
            workspace::subtract(
                &mut self.element_visible_area_workhouse,
                if uses_visibility_opaque_regions {
                    &self.visibility_opaque_regions
                } else {
                    &self.opaque_regions
                }
                .iter()
                .copied(),
                rectangle_limit,
            )?;
            let element_visible_area = self
                .element_visible_area_workhouse
                .iter()
                .fold(0usize, |acc, item| acc + (item.size.w * item.size.h) as usize);

            // No need to draw a completely hidden element
            if element_visible_area == 0 {
                // We allow multiple instance of a single element, so do not
                // override the state if we already have one
                if !element_render_states.states.contains_key(element_id) {
                    element_render_states
                        .states
                        .insert(element_id.clone(), RenderElementState::skipped());
                }
                continue;
            }

            let element_src = element.src();
            let element_geometry = element.geometry(output_scale);
            let element_transform = element.transform();
            let element_alpha = element.alpha();
            let element_last_state = self.last_state.elements.get(element.id());
            let element_is_framebuffer_effect = element.is_framebuffer_effect();
            let element_effect_regions = element.framebuffer_effect_regions(output_scale);

            self.element_damage_index.push(self.damage.len());
            if element_last_state
                .map(|s| {
                    !s.instance_matches(
                        &self.last_state.instances,
                        element_src,
                        element_geometry,
                        element_transform,
                        element_alpha,
                        z_index,
                        element_is_framebuffer_effect,
                        element_effect_regions,
                    )
                })
                .unwrap_or(true)
            {
                if let Some(intersection) = element_geometry.intersection(output_geo) {
                    workspace::extend(&mut self.damage, [intersection], rectangle_limit)?;
                }
                if let Some(state) = element_last_state {
                    workspace::extend(
                        &mut self.damage,
                        state
                            .instances(&self.last_state.instances)
                            .filter_map(|i| i.last_geometry.intersection(output_geo)),
                        rectangle_limit,
                    )?;
                }
                let state_change_damage = &self.damage[self.element_damage_index[z_index]..];
                if !state_change_damage.is_empty() {
                    let state_change_damage_area = rects_area(state_change_damage);
                    self.damage_summary.element_state_change_count =
                        self.damage_summary.element_state_change_count.saturating_add(1);
                    self.damage_summary.element_state_change_damage_area = self
                        .damage_summary
                        .element_state_change_damage_area
                        .saturating_add(state_change_damage_area);
                    if state_change_damage_area > self.damage_summary.top_element_state_change_area {
                        self.damage_summary.top_element_state_change_index = Some(element_index);
                        self.damage_summary.top_element_state_change_kind = Some(element.kind());
                        self.damage_summary.top_element_state_change_area = state_change_damage_area;
                        self.damage_summary.top_element_state_change_geometry_area =
                            rect_area(element_output_geometry);
                    }
                }
            } else {
                let element_output_damage_start = self.damage.len();
                workspace::try_visit(
                    |visit| {
                        element.visit_damage_since(
                            output_scale,
                            self.last_state
                                .elements
                                .get(element_id)
                                .map(|state| state.last_commit),
                            visit,
                        )
                    },
                    |mut rect| {
                        rect.loc += element_loc;
                        match rect.intersection(output_geo) {
                            Some(rect) => workspace::extend(&mut self.damage, [rect], rectangle_limit),
                            None => Ok(()),
                        }
                    },
                )?;
                let element_output_damage = &self.damage[element_output_damage_start..];
                if !element_output_damage.is_empty() {
                    let element_damage_area = rects_area(element_output_damage);
                    self.damage_summary.element_damage_element_count =
                        self.damage_summary.element_damage_element_count.saturating_add(1);
                    self.damage_summary.element_damage_rect_count = self
                        .damage_summary
                        .element_damage_rect_count
                        .saturating_add(element_output_damage.len());
                    self.damage_summary.element_damage_area = self
                        .damage_summary
                        .element_damage_area
                        .saturating_add(element_damage_area);
                    if element_damage_area > self.damage_summary.top_element_damage_area {
                        self.damage_summary.top_element_damage_index = Some(element_index);
                        self.damage_summary.top_element_damage_kind = Some(element.kind());
                        self.damage_summary.top_element_damage_rect_count = element_output_damage.len();
                        self.damage_summary.top_element_damage_area = element_damage_area;
                        self.damage_summary.top_element_damage_geometry_area =
                            rect_area(element_output_geometry);
                    }
                }
            }

            let element_opaque_regions_start_index = self.opaque_regions.len();
            workspace::try_visit(
                |visit| element.visit_opaque_regions(output_scale, visit),
                |mut rect| {
                    rect.loc += element_loc;
                    match rect.intersection(output_geo) {
                        Some(rect) => workspace::extend(&mut self.opaque_regions, [rect], rectangle_limit),
                        None => Ok(()),
                    }
                },
            )?;
            let element_opaque_regions_end_index = self.opaque_regions.len();
            self.opaque_regions_index
                .push(element_opaque_regions_start_index..element_opaque_regions_end_index);

            // Final-output opacity and traversal opacity have different
            // lifetimes around a framebuffer effect. Elements above the
            // effect are absent when its lower prefix is captured, so their
            // opacity cannot permanently cull contributors in the declared
            // read support. Elements encountered after the effect are part of
            // that prefix and may occlude still-lower contributors normally.
            if uses_visibility_opaque_regions {
                workspace::extend(
                    &mut self.visibility_opaque_regions,
                    self.opaque_regions[element_opaque_regions_start_index..element_opaque_regions_end_index]
                        .iter()
                        .copied(),
                    rectangle_limit,
                )?;
            }
            if let Some(read_area) = element_is_framebuffer_effect
                .then_some(element_effect_regions)
                .flatten()
                .and_then(|regions| regions.backdrop_read_area.intersection(output_geo))
            {
                if !uses_visibility_opaque_regions {
                    workspace::extend(
                        &mut self.visibility_opaque_regions,
                        self.opaque_regions.iter().copied(),
                        rectangle_limit,
                    )?;
                    uses_visibility_opaque_regions = true;
                }
                workspace::subtract(&mut self.visibility_opaque_regions, [read_area], rectangle_limit)?;
            }
            self.render_indices.push(element_index);

            if let Some(state) = element_render_states.states.get_mut(element_id) {
                if matches!(state.presentation_state, RenderElementPresentationState::Skipped) {
                    *state = RenderElementState::rendered(element_visible_area);
                } else {
                    state.visible_area += element_visible_area;
                }
                if element_is_framebuffer_effect {
                    // One cache identity cannot safely represent two captures
                    // in the same ordered scene. Force both instances through
                    // capture rather than reusing ambiguous framebuffer state.
                    state.needs_capture = true;
                }
            } else {
                let mut state = RenderElementState::rendered(element_visible_area);
                state.needs_capture = element_is_framebuffer_effect
                    && matches!(
                        element.framebuffer_capture_policy(),
                        FramebufferCapturePolicy::EveryDraw
                    );
                element_render_states.states.insert(element_id.clone(), state);
            }
            z_index += 1;
        }
        let elements_gone = self.last_state.elements.iter().filter(|(id, _)| {
            element_render_states
                .states
                .get(id)
                .map(|state| state.presentation_state == RenderElementPresentationState::Skipped)
                .unwrap_or(true)
        });

        for (_, state) in elements_gone {
            let gone_damage_start = self.damage.len();
            workspace::extend(
                &mut self.damage,
                state
                    .instances(&self.last_state.instances)
                    .filter_map(|i| i.last_geometry.intersection(output_geo)),
                rectangle_limit,
            )?;
            let gone_damage = &self.damage[gone_damage_start..];
            if !gone_damage.is_empty() {
                self.damage_summary.element_gone_count =
                    self.damage_summary.element_gone_count.saturating_add(1);
                self.damage_summary.element_gone_damage_area = self
                    .damage_summary
                    .element_gone_damage_area
                    .saturating_add(rects_area(gone_damage));
            }
        }

        // damage regions no longer covered by opaque regions
        self.element_damage.clear();
        workspace::extend(
            &mut self.element_damage,
            self.last_state.opaque_regions.iter().copied(),
            rectangle_limit,
        )?;
        workspace::subtract(
            &mut self.element_damage,
            self.opaque_regions.iter().copied(),
            rectangle_limit,
        )?;
        if !self.element_damage.is_empty() {
            self.damage_summary.opaque_uncovered_rect_count = self
                .damage_summary
                .opaque_uncovered_rect_count
                .saturating_add(self.element_damage.len());
            self.damage_summary.opaque_uncovered_area = self
                .damage_summary
                .opaque_uncovered_area
                .saturating_add(rects_area(&self.element_damage));
        }
        workspace::extend(
            &mut self.damage,
            self.element_damage.iter().copied(),
            rectangle_limit,
        )?;

        // we no longer need the element damage, return it so that we can
        // re-use its allocation next time

        let force_effect_redraw = self.last_state.size != Some(output_geo.size)
            || self.last_state.transform != Some(output_transform)
            || self.last_state.clear_color != clear_color;
        if force_effect_redraw {
            // The output geometry or transform changed, so just damage everything
            self.damage_summary.output_state_full_damage = true;
            trace!(
                previous_geometry = ?self.last_state.size,
                current_geometry = ?output_geo.size,
                previous_transform = ?self.last_state.transform,
                current_transform = ?output_transform,
                previous_clear_color = ?self.last_state.clear_color,
                current_clear_color = ?clear_color,
                "Output geometry, transform or clear color changed, damaging whole output geometry");
            self.damage.clear();
            workspace::extend(&mut self.damage, [output_geo], rectangle_limit)?;
        }

        propagate_framebuffer_effect_damage(
            &mut self.damage,
            &mut self.opaque_regions,
            &self.opaque_regions_index,
            &self.element_damage_index,
            elements,
            &self.render_indices,
            &mut self.capture_support_damage,
            rectangle_limit,
            &mut element_render_states,
            output_scale,
            output_geo,
            0,
            force_effect_redraw,
        )?;

        // That is all completely new damage, which we need to store for subsequent renders
        self.new_damage.clear();
        workspace::extend(&mut self.new_damage, self.damage.iter().copied(), rectangle_limit)?;

        // We now add old damage states, if we have an age value
        let age_damage_start = self.damage.len();
        let buffer_age_forces_full_damage =
            if age > 0 && age <= MAX_AGE && self.last_state.old_damage.len() >= age {
                trace!("age of {} recent enough, using old damage", age);
                // We do not need even older states anymore
                workspace::extend(
                    &mut self.damage,
                    self.last_state.old_damage.iter().take(age - 1).flatten().copied(),
                    rectangle_limit,
                )?;
                false
            } else {
                self.damage_summary.buffer_age_full_damage = true;
                trace!(
                    "no old damage available, re-render everything. age: {} old_damage len: {}",
                    age,
                    self.last_state.old_damage.len(),
                );
                // we still truncate the old damage to prevent growing
                // indefinitely in case we are continuously called with
                // an age of 0
                // just damage everything, if we have no damage
                self.damage.clear();
                workspace::extend(&mut self.damage, [output_geo], rectangle_limit)?;
                true
            };

        // Buffer-age damage is target-local correctness damage just like new
        // scene damage. If it intersects an effect's read support, the lower
        // prefix is redrawn into this target and the effect must recapture in
        // the same frame. In particular, age 0/full repaint may never draw an
        // effect from an unrelated target's stale scratch image.
        propagate_framebuffer_effect_damage(
            &mut self.damage,
            &mut self.opaque_regions,
            &self.opaque_regions_index,
            &self.element_damage_index,
            elements,
            &self.render_indices,
            &mut self.capture_support_damage,
            rectangle_limit,
            &mut element_render_states,
            output_scale,
            output_geo,
            if buffer_age_forces_full_damage {
                0
            } else {
                age_damage_start
            },
            buffer_age_forces_full_damage,
        )?;

        // Optimize the damage for rendering

        // Clamp all rectangles to the bounds removing the ones without intersection.
        self.damage.retain_mut(|rect| {
            if let Some(intersected) = rect.intersection(output_geo) {
                *rect = intersected;
                true
            } else {
                false
            }
        });

        self.damage_shaper
            .shape_damage_bounded(&mut self.damage, rectangle_limit)?;
        self.damage_summary.final_damage_rect_count = self.damage.len();
        self.damage_summary.final_damage_area = rects_area(&self.damage);

        if self.damage.is_empty() {
            trace!("nothing damaged, exiting early");
            return Ok(element_render_states);
        }

        self.last_state.elements.clear();
        self.last_state.instances.clear();
        for (z_index, elem) in self
            .render_indices
            .iter()
            .map(|index| elements.element(*index))
            .enumerate()
        {
            let id = elem.id();
            let index = self.last_state.instances.len();
            let previous = self.last_state.elements.get(id).map(|state| state.first_instance);
            self.last_state.instances.push(ElementInstanceRecord {
                state: ElementInstanceState {
                    last_src: elem.src(),
                    last_geometry: elem.geometry(output_scale),
                    last_transform: elem.transform(),
                    last_alpha: elem.alpha(),
                    last_z_index: z_index,
                    last_is_framebuffer_effect: elem.is_framebuffer_effect(),
                    last_effect_regions: elem.framebuffer_effect_regions(output_scale),
                },
                next: previous,
            });
            self.last_state.elements.insert(
                id.clone(),
                ElementState {
                    last_commit: elem.current_commit(),
                    first_instance: index,
                },
            );
        }

        self.last_state.size = Some(output_geo.size);
        self.last_state.transform = Some(output_transform);
        self.reclaim_history(MAX_AGE);
        let mut new_damage = self.history_spares.pop().unwrap_or_default();
        std::mem::swap(&mut self.new_damage, &mut new_damage);
        self.last_state.old_damage.push_front(new_damage);
        self.last_state.opaque_regions.clear();
        self.last_state
            .opaque_regions
            .extend(self.opaque_regions.iter().copied());

        self.last_state.clear_color = clear_color;

        Ok(element_render_states)
    }
}

#[cfg(test)]
mod framebuffer_effect_tests {
    use super::*;
    use crate::{
        backend::renderer::{
            element::{FramebufferCapturePolicy, FramebufferEffectRegions},
            utils::{DamageSet, OpaqueRegions},
        },
        utils::Point,
    };

    #[derive(Debug, Clone)]
    pub(super) struct TestElement {
        id: Id,
        commit: CommitCounter,
        geometry: Rectangle<i32, Physical>,
        effect_regions: Option<FramebufferEffectRegions>,
        opaque: bool,
        capture_policy: FramebufferCapturePolicy,
    }

    impl TestElement {
        pub(super) fn advance_commit(&mut self) {
            self.commit.increment();
        }
        pub(super) fn draw(geometry: Rectangle<i32, Physical>, commit: usize) -> Self {
            Self {
                id: Id::new(),
                commit: CommitCounter::from(commit),
                geometry,
                effect_regions: None,
                opaque: false,
                capture_policy: FramebufferCapturePolicy::OnBackdropDamage,
            }
        }

        pub(super) fn effect(
            paint_area: Rectangle<i32, Physical>,
            backdrop_read_area: Rectangle<i32, Physical>,
        ) -> Self {
            Self {
                id: Id::new(),
                commit: CommitCounter::from(1),
                geometry: paint_area,
                effect_regions: Some(FramebufferEffectRegions {
                    backdrop_read_area,
                    paint_area,
                }),
                opaque: false,
                capture_policy: FramebufferCapturePolicy::OnBackdropDamage,
            }
        }

        pub(super) fn opaque(mut self) -> Self {
            self.opaque = true;
            self
        }
    }

    impl Element for TestElement {
        fn id(&self) -> &Id {
            &self.id
        }

        fn current_commit(&self) -> CommitCounter {
            self.commit
        }

        fn src(&self) -> Rectangle<f64, BufferCoords> {
            Rectangle::new(
                Point::from((0.0, 0.0)),
                (self.geometry.size.w as f64, self.geometry.size.h as f64).into(),
            )
        }

        fn geometry(&self, _scale: Scale<f64>) -> Rectangle<i32, Physical> {
            self.geometry
        }

        fn damage_since(
            &self,
            _scale: Scale<f64>,
            commit: Option<CommitCounter>,
        ) -> DamageSet<i32, Physical> {
            if commit == Some(self.commit) {
                DamageSet::default()
            } else {
                DamageSet::from_slice(&[Rectangle::from_size(self.geometry.size)])
            }
        }

        fn opaque_regions(&self, _scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
            if self.opaque {
                OpaqueRegions::from_slice(&[Rectangle::from_size(self.geometry.size)])
            } else {
                OpaqueRegions::default()
            }
        }

        fn is_framebuffer_effect(&self) -> bool {
            self.effect_regions.is_some()
        }

        fn framebuffer_effect_regions(&self, _scale: Scale<f64>) -> Option<FramebufferEffectRegions> {
            self.effect_regions
        }

        fn framebuffer_capture_policy(&self) -> FramebufferCapturePolicy {
            self.capture_policy
        }
    }

    fn rect(x: i32, y: i32, width: i32, height: i32) -> Rectangle<i32, Physical> {
        Rectangle::new((x, y).into(), (width, height).into())
    }

    fn effect_needs_capture(states: &RenderElementStates, effect: &TestElement) -> bool {
        states
            .element_render_state(effect.id.clone())
            .expect("visible effect state")
            .needs_capture
    }

    fn element_was_rendered(states: &RenderElementStates, element: &TestElement) -> bool {
        matches!(
            states
                .element_render_state(element.id.clone())
                .expect("element state")
                .presentation_state,
            RenderElementPresentationState::Rendering { .. }
        )
    }

    #[test]
    fn every_draw_policy_recaptures_for_foreground_only_damage() {
        let upper_opaque_band = TestElement::draw(rect(10, 10, 80, 20), 1).opaque();
        let mut foreground = TestElement::draw(rect(45, 45, 1, 1), 1);
        let mut effect = TestElement::effect(rect(40, 40, 20, 20), rect(10, 10, 80, 80));
        let lower_contributor = TestElement::draw(rect(15, 15, 1, 1), 1);
        let mut tracker = OutputDamageTracker::new((200, 200), 1.0, Transform::Normal);
        tracker
            .damage_output(
                0,
                &[
                    upper_opaque_band.clone(),
                    foreground.clone(),
                    effect.clone(),
                    lower_contributor.clone(),
                ],
            )
            .expect("initial damage");

        foreground.commit = CommitCounter::from(2);
        let (damage, states) = tracker
            .damage_output(
                1,
                &[
                    upper_opaque_band.clone(),
                    foreground.clone(),
                    effect.clone(),
                    lower_contributor.clone(),
                ],
            )
            .expect("foreground-only damage with the default capture policy");
        assert!(!effect_needs_capture(&states, &effect));
        assert!(damage
            .expect("foreground damage")
            .iter()
            .all(|candidate| !candidate.overlaps(lower_contributor.geometry)));

        effect.capture_policy = FramebufferCapturePolicy::EveryDraw;
        foreground.commit = CommitCounter::from(3);
        let (damage, states) = tracker
            .damage_output(
                1,
                &[
                    upper_opaque_band.clone(),
                    foreground.clone(),
                    effect.clone(),
                    lower_contributor.clone(),
                ],
            )
            .expect("foreground-only damage with the every-draw capture policy");
        assert!(effect_needs_capture(&states, &effect));
        assert!(element_was_rendered(&states, &lower_contributor));
        assert!(damage
            .expect("capture support damage")
            .iter()
            .any(|candidate| candidate.overlaps(lower_contributor.geometry)));

        let (damage, states) = tracker
            .damage_output(
                1,
                &[upper_opaque_band, foreground, effect.clone(), lower_contributor],
            )
            .expect("stable every-draw scene");
        assert!(damage.is_none());
        assert!(effect_needs_capture(&states, &effect));
    }

    #[test]
    fn effect_read_support_reopens_only_opacity_above_the_effect() {
        let foreground_band = TestElement::draw(rect(10, 10, 80, 20), 1).opaque();
        let effect = TestElement::effect(rect(40, 40, 20, 20), rect(10, 10, 80, 80));
        let lower_contributor = TestElement::draw(rect(15, 15, 1, 1), 1);
        let mut tracker = OutputDamageTracker::new((200, 200), 1.0, Transform::Normal);
        let (_, states) = tracker
            .damage_output(0, &[foreground_band, effect.clone(), lower_contributor.clone()])
            .expect("effect support behind foreground opacity");

        assert!(effect_needs_capture(&states, &effect));
        assert!(element_was_rendered(&states, &lower_contributor));

        let backdrop_occluder = TestElement::draw(rect(10, 10, 80, 20), 1).opaque();
        let hidden_lower = TestElement::draw(rect(15, 15, 1, 1), 1);
        let mut tracker = OutputDamageTracker::new((200, 200), 1.0, Transform::Normal);
        let (_, states) = tracker
            .damage_output(0, &[effect, backdrop_occluder, hidden_lower.clone()])
            .expect("opacity inside the captured prefix");

        assert!(!element_was_rendered(&states, &hidden_lower));
    }

    #[test]
    fn support_band_damage_recaptures_but_outside_damage_does_not() {
        let effect = TestElement::effect(rect(40, 40, 20, 20), rect(10, 10, 80, 80));
        let mut lower = TestElement::draw(rect(15, 15, 1, 1), 1);
        let mut tracker = OutputDamageTracker::new((200, 200), 1.0, Transform::Normal);
        tracker
            .damage_output(0, &[effect.clone(), lower.clone()])
            .expect("initial damage");

        lower.commit = CommitCounter::from(2);
        let (_, states) = tracker
            .damage_output(1, &[effect.clone(), lower.clone()])
            .expect("support-band damage");
        assert!(effect_needs_capture(&states, &effect));

        lower.geometry = rect(150, 150, 1, 1);
        lower.commit = CommitCounter::from(3);
        tracker
            .damage_output(1, &[effect.clone(), lower.clone()])
            .expect("movement establishes the new lower geometry");
        lower.commit = CommitCounter::from(4);
        let (_, states) = tracker
            .damage_output(1, &[effect.clone(), lower])
            .expect("outside damage");
        assert!(!effect_needs_capture(&states, &effect));
    }

    #[test]
    fn lower_effect_damage_propagates_to_an_overlapping_upper_effect() {
        let upper = TestElement::effect(rect(60, 60, 30, 30), rect(30, 30, 100, 100));
        let lower = TestElement::effect(rect(70, 70, 30, 30), rect(40, 40, 100, 100));
        let mut content = TestElement::draw(rect(80, 80, 1, 1), 1);
        let mut tracker = OutputDamageTracker::new((200, 200), 1.0, Transform::Normal);
        tracker
            .damage_output(0, &[upper.clone(), lower.clone(), content.clone()])
            .expect("initial damage");

        content.commit = CommitCounter::from(2);
        let (_, states) = tracker
            .damage_output(1, &[upper.clone(), lower.clone(), content])
            .expect("nested effect damage");
        assert!(effect_needs_capture(&states, &lower));
        assert!(effect_needs_capture(&states, &upper));
    }

    #[test]
    fn full_buffer_age_repaint_always_recaptures_effects() {
        let effect = TestElement::effect(rect(40, 40, 20, 20), rect(10, 10, 80, 80));
        let lower = TestElement::draw(rect(0, 0, 200, 200), 1);
        let mut tracker = OutputDamageTracker::new((200, 200), 1.0, Transform::Normal);
        tracker
            .damage_output(0, &[effect.clone(), lower.clone()])
            .expect("initial damage");

        let (_, states) = tracker
            .damage_output(0, &[effect.clone(), lower])
            .expect("unknown buffer age");
        assert!(effect_needs_capture(&states, &effect));
    }

    #[test]
    fn moving_effect_recaptures_old_and_new_geometry() {
        let mut effect = TestElement::effect(rect(40, 40, 20, 20), rect(10, 10, 80, 80));
        let lower = TestElement::draw(rect(0, 0, 200, 200), 1);
        let mut tracker = OutputDamageTracker::new((200, 200), 1.0, Transform::Normal);
        tracker
            .damage_output(0, &[effect.clone(), lower.clone()])
            .expect("initial damage");

        effect.geometry = rect(80, 40, 20, 20);
        effect.effect_regions = Some(FramebufferEffectRegions {
            backdrop_read_area: rect(50, 10, 80, 80),
            paint_area: effect.geometry,
        });
        let (damage, states) = tracker
            .damage_output(1, &[effect.clone(), lower])
            .expect("moved effect");
        assert!(effect_needs_capture(&states, &effect));
        let damage = damage.expect("movement damage");
        assert!(damage
            .iter()
            .any(|candidate| candidate.overlaps(rect(40, 40, 20, 20))));
        assert!(damage
            .iter()
            .any(|candidate| candidate.overlaps(rect(80, 40, 20, 20))));
    }

    #[test]
    fn duplicate_effect_identity_forces_capture() {
        let mut first = TestElement::effect(rect(20, 20, 20, 20), rect(0, 0, 60, 60));
        let mut second = TestElement::effect(rect(80, 80, 20, 20), rect(60, 60, 60, 60));
        second.id = first.id.clone();
        let mut tracker = OutputDamageTracker::new((200, 200), 1.0, Transform::Normal);
        tracker
            .damage_output(0, &[first.clone(), second.clone()])
            .expect("initial duplicate identity");

        first.commit = CommitCounter::from(1);
        second.commit = CommitCounter::from(1);
        let (_, states) = tracker
            .damage_output(1, &[first.clone(), second])
            .expect("stable duplicate identity");
        assert!(effect_needs_capture(&states, &first));
    }
}
