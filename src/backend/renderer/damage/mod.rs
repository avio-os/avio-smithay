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
use smallvec::{smallvec, SmallVec};
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
        Element, FramebufferCapturePolicy, Id, Kind, RenderElement, RenderElementState, RenderElementStates,
    },
    sync::SyncPoint,
    utils::CommitCounter,
    Color32F,
};

use super::{Renderer, Texture};

mod shaper;

use shaper::DamageShaper;

const MAX_AGE: usize = 4;

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
    last_instances: SmallVec<[ElementInstanceState; 1]>,
}

impl ElementState {
    #[inline]
    fn instance_matches(
        &self,
        src: Rectangle<f64, BufferCoords>,
        geometry: Rectangle<i32, Physical>,
        transform: Transform,
        alpha: f32,
        z_index: usize,
        is_framebuffer_effect: bool,
        effect_regions: Option<super::element::FramebufferEffectRegions>,
    ) -> bool {
        self.last_instances.iter().any(|instance| {
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
}

/// Errors thrown by [`OutputDamageTracker::render_output`]
#[derive(thiserror::Error)]
pub enum Error<E: std::error::Error> {
    /// The provided [`Renderer`] returned an error
    #[error(transparent)]
    Rendering(E),
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

impl<E: std::error::Error + MaybeDeviceLost> Error<E> {
    /// Returns `true` when this error was caused by an unrecoverable loss of
    /// the rendering device (see [`MaybeDeviceLost`]).
    pub fn is_device_lost(&self) -> bool {
        match self {
            Error::Rendering(err) => err.is_device_lost(),
            Error::WaylandCompletionUnobservable => false,
            Error::OutputNoMode(_) => false,
        }
    }

    /// Returns `true` when rendering may have sampled a Wayland buffer but no
    /// exact completion edge can prove when that read finished.
    pub fn is_wayland_completion_unobservable(&self) -> bool {
        matches!(self, Error::WaylandCompletionUnobservable)
    }
}

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
        assert!(!error.is_device_lost());
    }
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
fn propagate_framebuffer_effect_damage<E: Element>(
    damage: &mut Vec<Rectangle<i32, Physical>>,
    opaque_regions: &mut [Rectangle<i32, Physical>],
    opaque_regions_index: &[Range<usize>],
    element_damage_index: &[usize],
    render_elements: &[&E],
    states: &mut RenderElementStates,
    output_scale: Scale<f64>,
    output_geo: Rectangle<i32, Physical>,
    damage_floor: usize,
    force_redraw: bool,
) {
    let mut capture_support_damage = Vec::new();
    for (z_index, element) in render_elements
        .iter()
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
            capture_support_damage.push(read);

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
            damage.push(paint);
        }
    }
    damage.extend(capture_support_damage);
}

impl<E: std::error::Error> std::fmt::Debug for Error<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Rendering(err) => std::fmt::Debug::fmt(err, f),
            Error::WaylandCompletionUnobservable => f.write_str("WaylandCompletionUnobservable"),
            Error::OutputNoMode(err) => std::fmt::Debug::fmt(err, f),
        }
    }
}

impl OutputDamageTracker {
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
        let mut render_elements: Vec<&E> = Vec::with_capacity(elements.len());
        let states = self.damage_output_internal(
            age,
            elements,
            output_scale,
            output_transform,
            output_geo,
            Some(clear_color),
            &mut render_elements,
        );

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
            for element in &render_elements {
                let Some(UnderlyingStorage::Wayland(buffer)) = element.sampled_storage(renderer) else {
                    continue;
                };
                if !buffers.iter().any(|existing| existing.same_instance(buffer)) {
                    buffers.push(buffer.clone());
                }
            }
            buffers
        };

        let render_res = (|| {
            // we have to take the element damage to be able to move it around
            let mut element_damage = std::mem::take(&mut self.element_damage);
            let mut element_opaque_regions = std::mem::take(&mut self.element_opaque_regions);
            let mut frame = renderer.render(framebuffer, output_size, output_transform)?;

            element_damage.clear();
            element_damage.extend_from_slice(&self.damage);
            element_damage =
                Rectangle::subtract_rects_many_in_place(element_damage, self.opaque_regions.iter().copied());

            trace!("clearing damage {:?}", element_damage);
            frame.clear(clear_color, &element_damage)?;

            for (z_index, element) in render_elements.iter().rev().enumerate() {
                let element_id = element.id();
                let element_geometry = element.geometry(output_scale);

                element_damage.clear();
                element_damage.extend(
                    self.damage
                        .iter()
                        .filter_map(|d| d.intersection(element_geometry)),
                );

                let element_opaque_regions_range =
                    self.opaque_regions_index.iter().rev().nth(z_index).unwrap();
                element_damage = Rectangle::subtract_rects_many_in_place(
                    element_damage,
                    self.opaque_regions[..element_opaque_regions_range.start]
                        .iter()
                        .copied(),
                );
                element_damage.iter_mut().for_each(|d| {
                    d.loc -= element_geometry.loc;
                });

                if element_damage.is_empty() {
                    trace!(
                        "skipping rendering element {:?} with geometry {:?}, no damage",
                        element_id,
                        element_geometry
                    );
                    continue;
                }

                element_opaque_regions.clear();
                element_opaque_regions.extend(
                    self.opaque_regions[element_opaque_regions_range.start..element_opaque_regions_range.end]
                        .iter()
                        .copied()
                        .map(|mut rect| {
                            rect.loc -= element_geometry.loc;
                            rect
                        }),
                );

                trace!(
                    "rendering element {:?} with geometry {:?} and damage {:?}",
                    element_id,
                    element_geometry,
                    element_damage,
                );

                if states
                    .element_render_state(element_id.clone())
                    .is_some_and(|state| state.needs_capture)
                {
                    let regions = element
                        .framebuffer_effect_regions(output_scale)
                        .expect("framebuffer effect without read/paint regions");
                    element.capture_framebuffer(&mut frame, regions)?;
                }

                element.draw(
                    &mut frame,
                    element.src(),
                    element_geometry,
                    &element_damage,
                    &element_opaque_regions,
                )?;
            }

            // return the element damage so that we can re-use the allocation
            std::mem::swap(&mut self.element_damage, &mut element_damage);
            std::mem::swap(&mut self.element_opaque_regions, &mut element_opaque_regions);
            frame.finish()
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
                    self.last_state = Default::default();
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
                self.last_state = Default::default();
                Err(Error::Rendering(err))
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
    ) -> Result<(Option<&'a Vec<Rectangle<i32, Physical>>>, RenderElementStates), OutputNoMode>
    where
        E: Element,
    {
        let (output_size, output_scale, output_transform) = self.mode.clone().try_into()?;

        // Output transform is specified in surface-rotation, so inversion gives us the
        // render transform for the output itself.
        let output_transform = output_transform.invert();

        // We have to apply to output transform to the output size so that the intersection
        // tests in damage_output_internal produces the correct results and do not crop
        // damage with the wrong size
        let output_geo = Rectangle::from_size(output_transform.transform_size(output_size));

        let mut render_elements: Vec<&E> = Vec::with_capacity(elements.len());
        let states = self.damage_output_internal(
            age,
            elements,
            output_scale,
            output_transform,
            output_geo,
            self.last_state.clear_color,
            &mut render_elements,
        );

        if self.damage.is_empty() {
            Ok((None, states))
        } else {
            Ok((Some(&self.damage), states))
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[profiling::function]
    fn damage_output_internal<'a, E>(
        &mut self,
        age: usize,
        elements: &'a [E],
        output_scale: Scale<f64>,
        output_transform: Transform,
        output_geo: Rectangle<i32, Physical>,
        clear_color: Option<Color32F>,
        render_elements: &mut Vec<&'a E>,
    ) -> RenderElementStates
    where
        E: Element,
    {
        self.damage.clear();
        self.damage_summary = OutputDamageSummary {
            buffer_age: age,
            ..OutputDamageSummary::default()
        };
        self.opaque_regions.clear();
        self.opaque_regions_index.clear();
        self.element_damage_index.clear();

        let mut element_render_states = RenderElementStates {
            states: HashMap::with_capacity(elements.len()),
        };

        // we have to take the element damage to be able to move it around
        let mut element_damage = std::mem::take(&mut self.element_damage);

        let mut element_visible_area_workhouse = std::mem::take(&mut self.element_visible_area_workhouse);
        // Preserve the original opacity traversal with no extra copying until
        // a framebuffer effect actually needs a distinct capture view.
        let mut visibility_opaque_regions: Option<Vec<Rectangle<i32, Physical>>> = None;
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
            element_visible_area_workhouse.clear();
            element_visible_area_workhouse.push(element_output_geometry);
            element_visible_area_workhouse = Rectangle::subtract_rects_many_in_place(
                element_visible_area_workhouse,
                visibility_opaque_regions
                    .as_deref()
                    .unwrap_or(&self.opaque_regions)
                    .iter()
                    .copied(),
            );
            let element_visible_area = element_visible_area_workhouse
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
                    self.damage.push(intersection);
                }
                if let Some(state) = element_last_state {
                    self.damage.extend(
                        state
                            .last_instances
                            .iter()
                            .filter_map(|i| i.last_geometry.intersection(output_geo)),
                    );
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
                let element_output_damage = element
                    .damage_since(
                        output_scale,
                        self.last_state.elements.get(element_id).map(|s| s.last_commit),
                    )
                    .into_iter()
                    .map(|mut d| {
                        d.loc += element_loc;
                        d
                    })
                    .filter_map(|geo| geo.intersection(output_geo));
                self.damage.extend(element_output_damage);
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
            let element_opaque_regions = element
                .opaque_regions(output_scale)
                .into_iter()
                .map(|mut region| {
                    region.loc += element_loc;
                    region
                })
                .filter_map(|geo| geo.intersection(output_geo));
            self.opaque_regions.extend(element_opaque_regions);
            let element_opaque_regions_end_index = self.opaque_regions.len();
            self.opaque_regions_index
                .push(element_opaque_regions_start_index..element_opaque_regions_end_index);

            // Final-output opacity and traversal opacity have different
            // lifetimes around a framebuffer effect. Elements above the
            // effect are absent when its lower prefix is captured, so their
            // opacity cannot permanently cull contributors in the declared
            // read support. Elements encountered after the effect are part of
            // that prefix and may occlude still-lower contributors normally.
            if let Some(visibility_opaque_regions) = visibility_opaque_regions.as_mut() {
                visibility_opaque_regions.extend_from_slice(
                    &self.opaque_regions
                        [element_opaque_regions_start_index..element_opaque_regions_end_index],
                );
            }
            if element_is_framebuffer_effect {
                if let Some(read_area) = element_effect_regions
                    .and_then(|regions| regions.backdrop_read_area.intersection(output_geo))
                {
                    let visibility_opaque_regions = visibility_opaque_regions.get_or_insert_with(|| {
                        let mut regions = std::mem::take(&mut self.visibility_opaque_regions);
                        regions.clear();
                        regions.extend_from_slice(&self.opaque_regions);
                        regions
                    });
                    *visibility_opaque_regions = Rectangle::subtract_rects_many_in_place(
                        std::mem::take(visibility_opaque_regions),
                        [read_area],
                    );
                }
            }
            render_elements.push(element);

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
        std::mem::swap(
            &mut self.element_visible_area_workhouse,
            &mut element_visible_area_workhouse,
        );
        if let Some(mut visibility_opaque_regions) = visibility_opaque_regions {
            std::mem::swap(
                &mut self.visibility_opaque_regions,
                &mut visibility_opaque_regions,
            );
        }

        // add the damage for elements gone that are not covered an opaque region
        let elements_gone = self.last_state.elements.iter().filter(|(id, _)| {
            element_render_states
                .states
                .get(id)
                .map(|state| state.presentation_state == RenderElementPresentationState::Skipped)
                .unwrap_or(true)
        });

        for (_, state) in elements_gone {
            let gone_damage_start = self.damage.len();
            self.damage.extend(
                state
                    .last_instances
                    .iter()
                    .filter_map(|i| i.last_geometry.intersection(output_geo)),
            );
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
        element_damage.clear();
        element_damage.extend_from_slice(&self.last_state.opaque_regions);
        element_damage =
            Rectangle::subtract_rects_many_in_place(element_damage, self.opaque_regions.iter().copied());
        if !element_damage.is_empty() {
            self.damage_summary.opaque_uncovered_rect_count = self
                .damage_summary
                .opaque_uncovered_rect_count
                .saturating_add(element_damage.len());
            self.damage_summary.opaque_uncovered_area = self
                .damage_summary
                .opaque_uncovered_area
                .saturating_add(rects_area(&element_damage));
        }
        self.damage.extend_from_slice(&element_damage);

        // we no longer need the element damage, return it so that we can
        // re-use its allocation next time
        std::mem::swap(&mut self.element_damage, &mut element_damage);

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
            self.damage.push(output_geo);
        }

        propagate_framebuffer_effect_damage(
            &mut self.damage,
            &mut self.opaque_regions,
            &self.opaque_regions_index,
            &self.element_damage_index,
            &render_elements,
            &mut element_render_states,
            output_scale,
            output_geo,
            0,
            force_effect_redraw,
        );

        // That is all completely new damage, which we need to store for subsequent renders
        let mut new_damage = self.damage.clone();
        new_damage.shrink_to_fit();

        // We now add old damage states, if we have an age value
        let age_damage_start = self.damage.len();
        let buffer_age_forces_full_damage = if age > 0 && self.last_state.old_damage.len() >= age {
            trace!("age of {} recent enough, using old damage", age);
            // We do not need even older states anymore
            self.last_state.old_damage.truncate(age);
            self.damage
                .extend(self.last_state.old_damage.iter().take(age - 1).flatten().copied());
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
            self.last_state.old_damage.truncate(MAX_AGE);
            // just damage everything, if we have no damage
            self.damage.clear();
            self.damage.push(output_geo);
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
            &render_elements,
            &mut element_render_states,
            output_scale,
            output_geo,
            if buffer_age_forces_full_damage {
                0
            } else {
                age_damage_start
            },
            buffer_age_forces_full_damage,
        );

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

        self.damage_shaper.shape_damage(&mut self.damage);
        self.damage_summary.final_damage_rect_count = self.damage.len();
        self.damage_summary.final_damage_area = rects_area(&self.damage);

        if self.damage.is_empty() {
            trace!("nothing damaged, exiting early");
            return element_render_states;
        }

        let mut new_elements_state = std::mem::take(&mut self.last_state.elements);
        new_elements_state.clear();
        new_elements_state.reserve(render_elements.len());
        let new_elements_state =
            render_elements
                .iter()
                .enumerate()
                .fold(new_elements_state, |mut map, (z_index, elem)| {
                    let id = elem.id();
                    let elem_src = elem.src();
                    let elem_alpha = elem.alpha();
                    let elem_geometry = elem.geometry(output_scale);
                    let elem_transform = elem.transform();
                    let element_is_framebuffer_effect = elem.is_framebuffer_effect();
                    let element_effect_regions = elem.framebuffer_effect_regions(output_scale);

                    if let Some(state) = map.get_mut(id) {
                        state.last_instances.push(ElementInstanceState {
                            last_src: elem_src,
                            last_geometry: elem_geometry,
                            last_transform: elem_transform,
                            last_alpha: elem_alpha,
                            last_z_index: z_index,
                            last_is_framebuffer_effect: element_is_framebuffer_effect,
                            last_effect_regions: element_effect_regions,
                        });
                    } else {
                        let current_commit = elem.current_commit();
                        map.insert(
                            id.clone(),
                            ElementState {
                                last_commit: current_commit,
                                last_instances: smallvec![ElementInstanceState {
                                    last_src: elem_src,
                                    last_geometry: elem_geometry,
                                    last_transform: elem_transform,
                                    last_alpha: elem_alpha,
                                    last_z_index: z_index,
                                    last_is_framebuffer_effect: element_is_framebuffer_effect,
                                    last_effect_regions: element_effect_regions,
                                }],
                            },
                        );
                    }

                    map
                });

        self.last_state.size = Some(output_geo.size);
        self.last_state.transform = Some(output_transform);
        self.last_state.elements = new_elements_state;
        self.last_state.old_damage.push_front(new_damage);
        self.last_state.opaque_regions.clear();
        self.last_state
            .opaque_regions
            .extend(self.opaque_regions.iter().copied());
        self.last_state.opaque_regions.shrink_to_fit();
        self.last_state.clear_color = clear_color;

        element_render_states
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
    struct TestElement {
        id: Id,
        commit: CommitCounter,
        geometry: Rectangle<i32, Physical>,
        effect_regions: Option<FramebufferEffectRegions>,
        opaque: bool,
        capture_policy: FramebufferCapturePolicy,
    }

    impl TestElement {
        fn draw(geometry: Rectangle<i32, Physical>, commit: usize) -> Self {
            Self {
                id: Id::new(),
                commit: CommitCounter::from(commit),
                geometry,
                effect_regions: None,
                opaque: false,
                capture_policy: FramebufferCapturePolicy::OnBackdropDamage,
            }
        }

        fn effect(
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

        fn opaque(mut self) -> Self {
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
