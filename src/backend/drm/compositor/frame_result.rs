use std::{collections::HashSet, os::fd::OwnedFd, sync::Arc};

use crate::{
    backend::{
        allocator::{
            dmabuf::{AsDmabuf, Dmabuf},
            Buffer, Slot,
        },
        drm::Framebuffer,
        renderer::{
            damage::{OutputDamageSummary, OutputDamageTracker},
            element::{Element, Id, RenderElement, RenderElementStates},
            sync::SyncPoint,
            utils::{CommitCounter, DamageSet, DamageSnapshot, OpaqueRegions},
            Bind, Blit, Color32F, Frame, Renderer,
        },
    },
    output::OutputNoMode,
    utils::{Buffer as BufferCoords, Physical, Point, Rectangle, Scale, Size, Transform},
};
use drm::control::{plane, PlaneType};

use super::{DrmScanoutBuffer, ScanoutBuffer};

/// DRM plane selected for a rendered element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneAssignmentInfo {
    /// DRM plane handle.
    pub handle: plane::Handle,
    /// Kernel-reported plane type.
    pub type_: PlaneType,
    /// Optional plane z-position when exposed by the driver.
    pub zpos: Option<i32>,
    /// True when the selected overlay plane is below the primary plane.
    pub is_underlay: bool,
}

/// Result for [`DrmCompositor::render_frame`][super::DrmCompositor::render_frame]
///
/// **Note**: This struct may contain a reference to the composited buffer
/// of the primary display plane. Dropping it will remove said reference and
/// allows the buffer to be reused.
///
/// Keeping the buffer longer may cause the following issues:
/// - **Too much damage** - until the buffer is marked free it is not considered
///   submitted by the swapchain, causing the age value of newly queried buffers
///   to be lower than necessary, potentially resulting in more rendering than necessary.
///   To avoid this make sure the buffer is dropped before starting the next render.
/// - **Exhaustion of swapchain images** - Continuing rendering while holding on
///   to too many buffers may cause the swapchain to run out of images, returning errors
///   on rendering until buffers are freed again. The exact amount of images in a
///   swapchain is an implementation detail, but should generally be expect to be
///   large enough to hold onto at least one `RenderFrameResult`.
pub struct RenderFrameResult<'a, B: Buffer, F: Framebuffer, E> {
    /// If this frame contains any changes and should be submitted
    pub is_empty: bool,
    /// The render element states of this frame
    pub states: RenderElementStates,
    /// Element for the primary plane
    pub primary_element: PrimaryPlaneElement<'a, B, F, E>,
    /// Selected primary scanout plane when `primary_element` is a direct element.
    pub primary_plane_assignment: Option<PlaneAssignmentInfo>,
    /// Overlay elements in front to back order
    pub overlay_elements: Vec<&'a E>,
    /// Selected overlay/underlay planes in the same order as `overlay_elements`.
    pub overlay_plane_assignments: Vec<PlaneAssignmentInfo>,
    /// Selected compositor-owned output layer plane when scene content was
    /// rendered above a direct-scanout primary element.
    pub output_layer_plane_assignment: Option<PlaneAssignmentInfo>,
    /// Scene elements rendered into the compositor-owned output layer.
    pub output_layer_elements: Vec<&'a E>,
    /// Original front-to-back scene index of the primary element within the
    /// output-layer element list. `None` means the primary is behind it.
    pub output_layer_primary_index: Option<usize>,
    /// Number of scene elements rendered into the compositor-owned output layer.
    pub output_layer_element_count: usize,
    /// True when the output layer performed GPU rendering in this frame.
    pub output_layer_rendered_this_frame: bool,
    /// Number of damaged rectangles rendered into the compositor-owned output
    /// layer this frame.
    pub output_layer_damage_rect_count: usize,
    /// Total damaged pixel area rendered into the compositor-owned output layer
    /// this frame.
    pub output_layer_damage_area: u64,
    /// Breakdown from the output-layer damage tracker before Avio sees only the
    /// collapsed aggregate damage.
    pub output_layer_damage_summary: OutputDamageSummary,
    pub(super) output_layer_render_sync: Option<SyncPoint>,
    pub(super) output_layer_exported_sync_file: Option<Arc<OwnedFd>>,
    /// Optional cursor plane element
    ///
    /// If set always above all other elements
    pub cursor_element: Option<&'a E>,
    /// Selected cursor plane when `cursor_element` is present.
    pub cursor_plane_assignment: Option<PlaneAssignmentInfo>,

    pub(super) primary_plane_element_id: Id,
    pub(super) supports_fencing: bool,
    pub(super) replaces_unpresented_render: bool,
}

impl<B: Buffer, F: Framebuffer, E> RenderFrameResult<'_, B, F, E> {
    /// Whether this render replaced a prepared or queued frame that had not
    /// reached a successful presentation. Such replacement resets buffer ages
    /// so damage history cannot cross the missing frame.
    #[inline]
    pub fn replaces_unpresented_render(&self) -> bool {
        self.replaces_unpresented_render
    }

    /// Returns whether scan-out submission may need a host-side wait fallback.
    ///
    /// When this returns `false`, the composited primary plane can be synchronized to KMS
    /// through `IN_FENCE_FD` alone. When it returns `true`,
    /// [`super::DrmCompositor::queue_frame`] and [`super::DrmCompositor::commit_frame`]
    /// will fall back to waiting for the render sync point before submitting the atomic
    /// commit.
    pub fn needs_sync(&self) -> bool {
        if let PrimaryPlaneElement::Swapchain(ref element) = self.primary_element {
            if !self.supports_fencing || element.exported_sync_file.is_none() {
                return true;
            }
        }
        self.output_layer_rendered_this_frame
            && (!self.supports_fencing || self.output_layer_exported_sync_file.is_none())
    }

    /// Clone the compositor-owned output-layer render-completion fence when it
    /// was produced by the current `render_frame` call.
    #[inline]
    pub fn export_current_output_layer_render_sync_file(&self) -> Option<OwnedFd> {
        self.output_layer_rendered_this_frame
            .then(|| {
                self.output_layer_exported_sync_file
                    .as_ref()
                    .and_then(|sync_file| sync_file.try_clone().ok())
            })
            .flatten()
    }

    /// Clone the compositor-owned output-layer render-completion sync point
    /// when it was produced by the current `render_frame` call.
    #[inline]
    pub fn current_output_layer_render_sync(&self) -> Option<SyncPoint> {
        self.output_layer_rendered_this_frame
            .then(|| self.output_layer_render_sync.clone())
            .flatten()
    }
}

struct SwapchainElement<'a, 'b, B: Buffer> {
    id: Id,
    slot: &'a Slot<B>,
    transform: Transform,
    damage: &'b DamageSnapshot<i32, BufferCoords>,
}

impl<B: Buffer> Element for SwapchainElement<'_, '_, B> {
    fn id(&self) -> &Id {
        &self.id
    }

    fn current_commit(&self) -> CommitCounter {
        self.damage.current_commit()
    }

    fn src(&self) -> Rectangle<f64, BufferCoords> {
        Rectangle::from_size(self.slot.size()).to_f64()
    }

    fn geometry(&self, _scale: Scale<f64>) -> Rectangle<i32, Physical> {
        Rectangle::from_size(self.slot.size().to_logical(1, self.transform).to_physical(1))
    }

    fn transform(&self) -> Transform {
        self.transform
    }

    fn damage_since(&self, scale: Scale<f64>, commit: Option<CommitCounter>) -> DamageSet<i32, Physical> {
        self.damage
            .damage_since(commit)
            .map(|d| {
                d.into_iter()
                    .map(|d| d.to_logical(1, self.transform, &self.slot.size()).to_physical(1))
                    .collect()
            })
            .unwrap_or_else(|| DamageSet::from_slice(&[self.geometry(scale)]))
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        OpaqueRegions::from_slice(&[self.geometry(scale)])
    }
}

enum FrameResultDamageElement<'a, 'b, E, B: Buffer> {
    Element(&'a E),
    Swapchain(SwapchainElement<'a, 'b, B>),
}

impl<E, B> Element for FrameResultDamageElement<'_, '_, E, B>
where
    E: Element,
    B: Buffer,
{
    fn id(&self) -> &Id {
        match self {
            FrameResultDamageElement::Element(e) => e.id(),
            FrameResultDamageElement::Swapchain(e) => e.id(),
        }
    }

    fn current_commit(&self) -> CommitCounter {
        match self {
            FrameResultDamageElement::Element(e) => e.current_commit(),
            FrameResultDamageElement::Swapchain(e) => e.current_commit(),
        }
    }

    fn src(&self) -> Rectangle<f64, BufferCoords> {
        match self {
            FrameResultDamageElement::Element(e) => e.src(),
            FrameResultDamageElement::Swapchain(e) => e.src(),
        }
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        match self {
            FrameResultDamageElement::Element(e) => e.geometry(scale),
            FrameResultDamageElement::Swapchain(e) => e.geometry(scale),
        }
    }

    fn location(&self, scale: Scale<f64>) -> Point<i32, Physical> {
        match self {
            FrameResultDamageElement::Element(e) => e.location(scale),
            FrameResultDamageElement::Swapchain(e) => e.location(scale),
        }
    }

    fn transform(&self) -> Transform {
        match self {
            FrameResultDamageElement::Element(e) => e.transform(),
            FrameResultDamageElement::Swapchain(e) => e.transform(),
        }
    }

    fn damage_since(&self, scale: Scale<f64>, commit: Option<CommitCounter>) -> DamageSet<i32, Physical> {
        match self {
            FrameResultDamageElement::Element(e) => e.damage_since(scale, commit),
            FrameResultDamageElement::Swapchain(e) => e.damage_since(scale, commit),
        }
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        match self {
            FrameResultDamageElement::Element(e) => e.opaque_regions(scale),
            FrameResultDamageElement::Swapchain(e) => e.opaque_regions(scale),
        }
    }
}

#[derive(Debug)]
/// Defines the element for the primary plane
pub enum PrimaryPlaneElement<'a, B: Buffer, F: Framebuffer, E> {
    /// A slot from the swapchain was used for rendering
    /// the primary plane
    Swapchain(PrimarySwapchainElement<B, F>),
    /// An element has been assigned for direct scan-out
    Element(&'a E),
}

/// Error for [`RenderFrameResult::blit_frame_result`]
#[derive(Debug, thiserror::Error)]
pub enum BlitFrameResultError<R: std::error::Error, E: std::error::Error> {
    /// A render error occurred
    #[error(transparent)]
    Rendering(R),
    /// A error occurred during exporting the buffer
    #[error(transparent)]
    Export(E),
}

fn render_frame_result_elements<R, E>(
    renderer: &mut R,
    framebuffer: &mut R::Framebuffer<'_>,
    elements: &[&E],
    damage: &[Rectangle<i32, Physical>],
    size: Size<i32, Physical>,
    transform: Transform,
    scale: Scale<f64>,
) -> Result<Option<SyncPoint>, R::Error>
where
    R: Renderer,
    E: Element + RenderElement<R>,
{
    if elements.is_empty() {
        return Ok(None);
    }

    let mut frame = renderer.render(framebuffer, size, transform)?;
    for element in elements.iter().rev() {
        let src = element.src();
        let dst = element.geometry(scale);
        let element_damage = damage
            .iter()
            .filter_map(|damage| {
                damage.intersection(dst).map(|mut damage| {
                    damage.loc -= dst.loc;
                    damage
                })
            })
            .collect::<Vec<_>>();
        if element_damage.is_empty() {
            continue;
        }

        tracing::trace!("drawing frame element with damage: {:#?}", element_damage);
        element.draw(&mut frame, src, dst, &element_damage, &[])?;
    }

    frame.finish().map(Some)
}

fn extend_global_opaque_regions<E: Element>(
    opaque_regions: &mut Vec<Rectangle<i32, Physical>>,
    element: &E,
    scale: Scale<f64>,
) {
    let location = element.geometry(scale).loc;
    opaque_regions.extend(element.opaque_regions(scale).into_iter().map(|mut region| {
        region.loc += location;
        region
    }));
}

impl<B, F, E> RenderFrameResult<'_, B, F, E>
where
    B: Buffer,
    F: Framebuffer,
{
    /// Get the damage of this frame for the specified dtr and age
    pub fn damage_from_age<'d>(
        &self,
        damage_tracker: &'d mut OutputDamageTracker,
        age: usize,
        filter: impl IntoIterator<Item = Id>,
    ) -> Result<(Option<&'d Vec<Rectangle<i32, Physical>>>, RenderElementStates), OutputNoMode>
    where
        E: Element,
    {
        #[allow(clippy::mutable_key_type)]
        let filter_ids: HashSet<Id> = filter.into_iter().collect();

        let mut elements: Vec<FrameResultDamageElement<'_, '_, E, B>> = Vec::with_capacity(
            usize::from(self.cursor_element.is_some())
                + self.output_layer_elements.len()
                + self.overlay_elements.len()
                + 1,
        );
        if let Some(cursor) = self.cursor_element {
            if !filter_ids.contains(cursor.id()) {
                elements.push(FrameResultDamageElement::Element(cursor));
            }
        }

        let primary_render_element = match &self.primary_element {
            PrimaryPlaneElement::Swapchain(PrimarySwapchainElement {
                slot,
                transform,
                damage,
                ..
            }) => FrameResultDamageElement::Swapchain(SwapchainElement {
                id: self.primary_plane_element_id.clone(),
                transform: *transform,
                slot: match &slot.buffer {
                    ScanoutBuffer::Swapchain(slot) => slot,
                    _ => unreachable!(),
                },
                damage,
            }),
            PrimaryPlaneElement::Element(e) => FrameResultDamageElement::Element(*e),
        };

        let mut primary_render_element = Some(primary_render_element);
        for (index, element) in self.output_layer_elements.iter().enumerate() {
            if self.output_layer_primary_index == Some(index) {
                elements.push(primary_render_element.take().unwrap());
            }
            if !filter_ids.contains(element.id()) {
                elements.push(FrameResultDamageElement::Element(*element));
            }
        }
        if self.output_layer_primary_index.is_some() && primary_render_element.is_some() {
            elements.push(primary_render_element.take().unwrap());
        }

        elements.extend(
            self.overlay_elements
                .iter()
                .filter(|e| !filter_ids.contains(e.id()))
                .map(|e| FrameResultDamageElement::Element(*e)),
        );

        if let Some(primary_render_element) = primary_render_element {
            elements.push(primary_render_element);
        }

        damage_tracker.damage_output(age, &elements)
    }
}

impl<'a, B, F, E> RenderFrameResult<'a, B, F, E>
where
    B: Buffer + AsDmabuf,
    <B as AsDmabuf>::Error: std::fmt::Debug,
    F: Framebuffer,
{
    /// Blit the frame result
    #[allow(clippy::too_many_arguments)]
    pub fn blit_frame_result<R>(
        &self,
        size: impl Into<Size<i32, Physical>>,
        transform: Transform,
        scale: impl Into<Scale<f64>>,
        renderer: &mut R,
        framebuffer: &mut R::Framebuffer<'_>,
        damage: impl IntoIterator<Item = Rectangle<i32, Physical>>,
        filter: impl IntoIterator<Item = Id>,
    ) -> Result<SyncPoint, BlitFrameResultError<R::Error, <B as AsDmabuf>::Error>>
    where
        R: Renderer + Bind<Dmabuf> + Blit,
        R::TextureId: 'static,
        E: Element + RenderElement<R>,
    {
        let size = size.into();
        let scale = scale.into();
        #[allow(clippy::mutable_key_type)]
        let filter_ids: HashSet<Id> = filter.into_iter().collect();
        let damage = damage.into_iter().collect::<Vec<_>>();

        // If we have no damage we can exit early
        if damage.is_empty() {
            return Ok(SyncPoint::signaled());
        }

        let mut opaque_regions: Vec<Rectangle<i32, Physical>> = Vec::new();

        let mut elements_to_render: Vec<&'a E> = Vec::with_capacity(
            usize::from(self.cursor_element.is_some())
                + self.output_layer_elements.len()
                + self.overlay_elements.len()
                + 1,
        );

        if let Some(cursor_element) = self.cursor_element.as_ref() {
            if !filter_ids.contains(cursor_element.id()) {
                elements_to_render.push(*cursor_element);
                extend_global_opaque_regions(&mut opaque_regions, *cursor_element, scale);
            }
        }

        for element in &self.output_layer_elements {
            if filter_ids.contains(element.id()) {
                continue;
            }
            elements_to_render.push(element);
            extend_global_opaque_regions(&mut opaque_regions, *element, scale);
        }

        for element in self
            .overlay_elements
            .iter()
            .filter(|e| !filter_ids.contains(e.id()))
        {
            elements_to_render.push(element);
            extend_global_opaque_regions(&mut opaque_regions, *element, scale);
        }

        let primary_dmabuf = match &self.primary_element {
            PrimaryPlaneElement::Swapchain(PrimarySwapchainElement { slot, sync, .. }) => {
                debug_assert!(self.output_layer_primary_index.is_none());
                let dmabuf = match &slot.buffer {
                    ScanoutBuffer::Swapchain(slot) => slot.export().map_err(BlitFrameResultError::Export)?,
                    _ => unreachable!(),
                };
                let size = dmabuf.size();
                let geometry = Rectangle::from_size(size.to_logical(1, Transform::Normal).to_physical(1));
                opaque_regions.push(geometry);
                Some((sync.clone(), dmabuf, geometry))
            }
            PrimaryPlaneElement::Element(e) => {
                extend_global_opaque_regions(&mut opaque_regions, *e, scale);
                if let Some(primary_index) = self.output_layer_primary_index {
                    debug_assert!(self.overlay_elements.is_empty());
                    let cursor_prefix = usize::from(
                        self.cursor_element
                            .as_ref()
                            .is_some_and(|cursor| !filter_ids.contains(cursor.id())),
                    );
                    let visible_output_layer_prefix = self
                        .output_layer_elements
                        .iter()
                        .take(primary_index)
                        .filter(|element| !filter_ids.contains(element.id()))
                        .count();
                    elements_to_render.insert(
                        (cursor_prefix + visible_output_layer_prefix).min(elements_to_render.len()),
                        *e,
                    );
                } else {
                    elements_to_render.push(*e);
                }
                None
            }
        };

        let clear_damage =
            Rectangle::subtract_rects_many_in_place(damage.clone(), opaque_regions.iter().copied());

        let mut sync: Option<SyncPoint> = None;
        if !clear_damage.is_empty() {
            tracing::trace!("clearing frame damage {:#?}", clear_damage);

            let mut frame = renderer
                .render(framebuffer, size, transform)
                .map_err(BlitFrameResultError::Rendering)?;

            frame
                .clear(Color32F::BLACK, &clear_damage)
                .map_err(BlitFrameResultError::Rendering)?;

            sync = Some(frame.finish().map_err(BlitFrameResultError::Rendering)?);
        }

        if let Some((primary_dmabuf_sync, mut dmabuf, geometry)) = primary_dmabuf {
            let blit_damage = damage
                .iter()
                .filter_map(|d| d.intersection(geometry))
                .collect::<Vec<_>>();

            tracing::trace!("blitting frame with damage: {:#?}", blit_damage);

            renderer
                .wait(&primary_dmabuf_sync)
                .map_err(BlitFrameResultError::Rendering)?;
            let fb = renderer
                .bind(&mut dmabuf)
                .map_err(BlitFrameResultError::Rendering)?;
            for rect in blit_damage {
                // TODO: On Vulkan, may need to combine sync points instead of just using latest?
                sync = Some(
                    renderer
                        .blit(
                            &fb,
                            framebuffer,
                            rect,
                            rect,
                            crate::backend::renderer::TextureFilter::Linear,
                        )
                        .map_err(BlitFrameResultError::Rendering)?,
                );
            }
        }

        if let Some(render_sync) = render_frame_result_elements(
            renderer,
            framebuffer,
            &elements_to_render,
            &damage,
            size,
            transform,
            scale,
        )
        .map_err(BlitFrameResultError::Rendering)?
        {
            sync = Some(render_sync);
        }

        Ok(sync.unwrap_or_default())
    }
}

impl<B: Buffer + std::fmt::Debug, F: Framebuffer + std::fmt::Debug, E: std::fmt::Debug> std::fmt::Debug
    for RenderFrameResult<'_, B, F, E>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderFrameResult")
            .field("is_empty", &self.is_empty)
            .field("states", &self.states)
            .field("primary_element", &self.primary_element)
            .field("primary_plane_assignment", &self.primary_plane_assignment)
            .field("overlay_elements", &self.overlay_elements)
            .field("overlay_plane_assignments", &self.overlay_plane_assignments)
            .field(
                "output_layer_plane_assignment",
                &self.output_layer_plane_assignment,
            )
            .field("output_layer_elements", &self.output_layer_elements)
            .field("output_layer_primary_index", &self.output_layer_primary_index)
            .field(
                "output_layer_rendered_this_frame",
                &self.output_layer_rendered_this_frame,
            )
            .field("output_layer_damage_summary", &self.output_layer_damage_summary)
            .field("cursor_element", &self.cursor_element)
            .field("cursor_plane_assignment", &self.cursor_plane_assignment)
            .finish()
    }
}

#[derive(Debug)]
/// Defines the element for the primary plane in cases where a composited buffer was used.
pub struct PrimarySwapchainElement<B: Buffer, F: Framebuffer> {
    /// The slot from the swapchain
    pub(super) slot: DrmScanoutBuffer<B, F>,
    /// Sync point
    pub sync: SyncPoint,
    pub(super) exported_sync_file: Option<Arc<OwnedFd>>,
    /// True when this `render_frame` call rendered into the primary swapchain
    /// buffer and `exported_sync_file` belongs to the current render.
    pub(super) rendered_this_frame: bool,
    /// The transform applied during rendering
    pub transform: Transform,
    /// The damage on the primary plane
    pub damage: DamageSnapshot<i32, BufferCoords>,
}

impl<B: Buffer, F: Framebuffer> PrimarySwapchainElement<B, F> {
    /// Access the underlying swapchain buffer
    #[inline]
    pub fn buffer(&self) -> &B {
        match &self.slot.buffer {
            ScanoutBuffer::Swapchain(slot) => slot,
            _ => unreachable!(),
        }
    }

    /// Clone the underlying swapchain slot keepalive.
    ///
    /// Holding this value keeps the slot acquired and prevents the swapchain
    /// from reusing the same backing buffer while an external consumer still
    /// reads from an exported DMA-BUF view of it.
    #[inline]
    pub fn slot_keepalive(&self) -> Arc<Slot<B>> {
        match &self.slot.buffer {
            ScanoutBuffer::Swapchain(slot) => slot.clone(),
            _ => unreachable!(),
        }
    }

    /// Clone the compositor render-completion fence as a sync_file, if submission
    /// can hand it to both KMS and external consumers without re-exporting.
    #[inline]
    pub fn export_sync_file(&self) -> Option<OwnedFd> {
        self.exported_sync_file
            .as_ref()
            .and_then(|sync_file| sync_file.try_clone().ok())
    }

    /// Clone the render-completion fence only when it was produced by the
    /// current `render_frame` call.
    #[inline]
    pub fn export_current_render_sync_file(&self) -> Option<OwnedFd> {
        self.rendered_this_frame
            .then(|| self.export_sync_file())
            .flatten()
    }

    /// Returns whether this frame actually rendered into the primary
    /// swapchain. If this is false, the element may still carry a cached sync
    /// file from an older primary-plane config.
    #[inline]
    pub fn rendered_this_frame(&self) -> bool {
        self.rendered_this_frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{
        allocator::{Allocator, Fourcc, Modifier, Swapchain},
        drm::Framebuffer,
        renderer::{
            element::{Id, RenderElementStates},
            utils::DamageBag,
        },
    };
    use drm::control::framebuffer;
    use rustix::event::{eventfd, EventfdFlags};
    use std::{collections::HashMap, num::NonZeroU32, os::fd::AsRawFd};

    use super::super::{CachedDrmFramebuffer, DrmFramebuffer};

    #[derive(Debug, Clone)]
    struct DummyBuffer;

    struct DummyElement {
        id: Id,
        geometry: Rectangle<i32, Physical>,
        opaque: Rectangle<i32, Physical>,
    }

    impl Element for DummyElement {
        fn id(&self) -> &Id {
            &self.id
        }

        fn current_commit(&self) -> CommitCounter {
            CommitCounter::default()
        }

        fn src(&self) -> Rectangle<f64, BufferCoords> {
            Rectangle::default()
        }

        fn geometry(&self, _scale: Scale<f64>) -> Rectangle<i32, Physical> {
            self.geometry
        }

        fn opaque_regions(&self, _scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
            OpaqueRegions::from_slice(&[self.opaque])
        }
    }

    impl crate::backend::allocator::Buffer for DummyBuffer {
        fn size(&self) -> crate::utils::Size<i32, crate::utils::Buffer> {
            crate::utils::Size::from((1, 1))
        }

        fn format(&self) -> drm_fourcc::DrmFormat {
            drm_fourcc::DrmFormat {
                code: drm_fourcc::DrmFourcc::Argb8888,
                modifier: drm_fourcc::DrmModifier::Linear,
            }
        }
    }

    #[derive(Debug)]
    struct DummyAllocator;

    impl Allocator for DummyAllocator {
        type Buffer = DummyBuffer;
        type Error = std::convert::Infallible;

        fn create_buffer(
            &mut self,
            _width: u32,
            _height: u32,
            _fourcc: Fourcc,
            _modifiers: &[Modifier],
        ) -> Result<Self::Buffer, Self::Error> {
            Ok(DummyBuffer)
        }
    }

    #[derive(Debug, Clone)]
    struct DummyFramebuffer {
        handle: framebuffer::Handle,
    }

    impl AsRef<framebuffer::Handle> for DummyFramebuffer {
        fn as_ref(&self) -> &framebuffer::Handle {
            &self.handle
        }
    }

    impl Framebuffer for DummyFramebuffer {
        fn format(&self) -> drm_fourcc::DrmFormat {
            drm_fourcc::DrmFormat {
                code: drm_fourcc::DrmFourcc::Argb8888,
                modifier: drm_fourcc::DrmModifier::Linear,
            }
        }
    }

    fn make_primary_swapchain_element(
        exported_sync_file: Option<Arc<OwnedFd>>,
        rendered_this_frame: bool,
    ) -> PrimarySwapchainElement<DummyBuffer, DummyFramebuffer> {
        let mut swapchain = Swapchain::new(
            DummyAllocator,
            1,
            1,
            Fourcc::Argb8888,
            vec![drm_fourcc::DrmModifier::Linear],
        );
        let slot = swapchain
            .acquire()
            .expect("swapchain allocation failed")
            .expect("no swapchain slot available for test");
        let slot = DrmScanoutBuffer {
            buffer: ScanoutBuffer::Swapchain(Arc::new(slot)),
            fb: CachedDrmFramebuffer::new(DrmFramebuffer::Exporter(DummyFramebuffer {
                handle: framebuffer::Handle::from(NonZeroU32::new(1).unwrap()),
            })),
        };

        PrimarySwapchainElement {
            slot,
            sync: SyncPoint::signaled(),
            exported_sync_file,
            rendered_this_frame,
            transform: Transform::Normal,
            damage: DamageBag::<i32, BufferCoords>::new(1).snapshot(),
        }
    }

    fn make_render_frame_result(
        exported_sync_file: Option<Arc<OwnedFd>>,
        supports_fencing: bool,
    ) -> RenderFrameResult<'static, DummyBuffer, DummyFramebuffer, ()> {
        RenderFrameResult {
            is_empty: false,
            states: RenderElementStates {
                states: HashMap::new(),
            },
            primary_element: PrimaryPlaneElement::Swapchain(make_primary_swapchain_element(
                exported_sync_file,
                true,
            )),
            primary_plane_assignment: None,
            overlay_elements: Vec::new(),
            overlay_plane_assignments: Vec::new(),
            output_layer_plane_assignment: None,
            output_layer_elements: Vec::new(),
            output_layer_primary_index: None,
            output_layer_element_count: 0,
            output_layer_rendered_this_frame: false,
            output_layer_damage_rect_count: 0,
            output_layer_damage_area: 0,
            output_layer_damage_summary: OutputDamageSummary::default(),
            output_layer_render_sync: None,
            output_layer_exported_sync_file: None,
            cursor_element: None,
            cursor_plane_assignment: None,
            primary_plane_element_id: Id::new(),
            supports_fencing,
            replaces_unpresented_render: false,
        }
    }

    #[test]
    fn frame_result_opaque_regions_are_translated_to_output_coordinates() {
        let element = DummyElement {
            id: Id::new(),
            geometry: Rectangle::new((300, 200).into(), (800, 600).into()),
            opaque: Rectangle::new((10, 20).into(), (100, 50).into()),
        };
        let mut opaque_regions = Vec::new();

        extend_global_opaque_regions(&mut opaque_regions, &element, Scale::from(1.0));

        assert_eq!(
            opaque_regions,
            vec![Rectangle::new((310, 220).into(), (100, 50).into())]
        );
    }

    #[test]
    fn primary_swapchain_element_export_sync_file_clones_cached_fd() {
        let cached_sync_file = Arc::new(
            eventfd(0, EventfdFlags::CLOEXEC | EventfdFlags::NONBLOCK)
                .expect("failed to allocate eventfd for test"),
        );
        let element = make_primary_swapchain_element(Some(cached_sync_file), true);

        let first = element
            .export_sync_file()
            .expect("expected cached sync_file clone");
        let second = element
            .export_sync_file()
            .expect("expected second cached sync_file clone");

        assert_ne!(first.as_raw_fd(), second.as_raw_fd());
    }

    #[test]
    fn primary_swapchain_element_exports_current_sync_only_for_current_render() {
        let cached_sync_file: Arc<OwnedFd> = Arc::new(
            eventfd(0, EventfdFlags::CLOEXEC | EventfdFlags::NONBLOCK)
                .expect("eventfd should be available")
                .into(),
        );
        let stale = make_primary_swapchain_element(Some(cached_sync_file.clone()), false);
        assert!(stale.export_sync_file().is_some());
        assert!(stale.export_current_render_sync_file().is_none());

        let current = make_primary_swapchain_element(Some(cached_sync_file), true);
        assert!(current.export_current_render_sync_file().is_some());
    }

    #[test]
    fn render_frame_result_needs_sync_until_submit_sync_file_is_cached() {
        let needs_sync = make_render_frame_result(None, true);
        assert!(needs_sync.needs_sync());

        let cached_sync_file = Arc::new(
            eventfd(0, EventfdFlags::CLOEXEC | EventfdFlags::NONBLOCK)
                .expect("failed to allocate eventfd for test"),
        );
        let explicit_submit_sync = make_render_frame_result(Some(cached_sync_file), true);
        assert!(!explicit_submit_sync.needs_sync());
    }
}
