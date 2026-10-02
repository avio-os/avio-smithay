//! Dual-Kawase blur chain for the Vulkan renderer.
//!
//! Replaces the plain bilinear blit pyramid for material blur: each pyramid
//! step is a fullscreen fragment pass sampling 5 (down) or 8 (up) bilinear
//! taps at half-pixel offsets (KWin's formulation), composing a
//! gaussian-like kernel from the same pixels the blit chain already touched
//! — equivalent bandwidth, marginal extra ALU. Passes explicitly select
//! linear-light filtering or CSS-compatible encoded-sRGB filtering; storage
//! stays sRGB-encoded in either case.

use std::sync::Arc;

use ash::vk;
use indexmap::IndexMap;
use tracing::{instrument, trace};

use crate::{
    backend::renderer::{sync::SyncPoint, Texture},
    utils::{Buffer as BufferCoord, Point, Size},
};

use super::{
    blit::transition_tracked_image_layout,
    descriptor::TextureSampler,
    image::{
        acquire_images_from_foreign, commit_foreign_releases, release_images_to_foreign,
        restore_unsubmitted_foreign_acquires, transition_image_layout, VulkanImage,
    },
    pipeline::{push_constants_bytes, KawaseColorTransform, KawasePushConstants},
    VulkanRenderer, VulkanRendererError, VulkanTexture,
};

/// One dual-Kawase pyramid step: renders `source` into `destination` with the
/// down- or upsample kernel.
///
/// The pass records its working space. Linear-light passes choose hardware or
/// shader encoding from the destination format; encoded-sRGB passes use UNORM
/// views so the sampled and stored channel values remain encoded.
#[derive(Debug, Clone)]
pub struct VulkanKawasePass {
    pub(super) source: VulkanTexture,
    pub(super) destination: VulkanTexture,
    pub(super) upsample: bool,
    pub(super) offset: f32,
    pub(super) transform: KawaseColorTransform,
    pub(super) encoding: VulkanKawaseEncoding,
    pub(super) source_origin: Point<i32, BufferCoord>,
    pub(super) source_extent: Size<i32, BufferCoord>,
    pub(super) destination_extent: Size<i32, BufferCoord>,
}

/// Colour-space convention used by a Kawase pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VulkanKawaseEncoding {
    /// Decode sRGB, filter in linear light, then encode for storage.
    #[default]
    LinearLight,
    /// Filter and apply the colour transform directly to gamma-encoded sRGB
    /// values. This intentionally matches CSS backdrop-filter semantics.
    EncodedSrgb,
}

impl VulkanKawaseEncoding {
    fn destination_format(self, storage: vk::Format, render: vk::Format) -> vk::Format {
        match self {
            Self::LinearLight => render,
            Self::EncodedSrgb => storage,
        }
    }
}

impl VulkanKawasePass {
    /// Describes a kawase pass between two renderer textures.
    pub fn new(source: &VulkanTexture, destination: &VulkanTexture, upsample: bool, offset: f32) -> Self {
        Self {
            source: source.clone(),
            destination: destination.clone(),
            upsample,
            offset,
            transform: KawaseColorTransform::IDENTITY,
            encoding: VulkanKawaseEncoding::LinearLight,
            source_origin: (0, 0).into(),
            source_extent: source.size(),
            destination_extent: destination.size(),
        }
    }

    /// Restrict this pass to initialized, origin-aligned regions of reusable
    /// backing images. Sampling clamps at the source region's texel centers;
    /// pixels beyond the destination extent are neither read nor written.
    /// Extents are validated against the image capacities before recording.
    pub fn with_extents(
        mut self,
        source: Size<i32, BufferCoord>,
        destination: Size<i32, BufferCoord>,
    ) -> Self {
        self.source_extent = source;
        self.destination_extent = destination;
        self
    }

    pub(super) fn with_source_origin(mut self, origin: Point<i32, BufferCoord>) -> Self {
        self.source_origin = origin;
        self
    }

    /// Applies a post-blur saturation transform in the destination colour
    /// space. Material graphs should set this on their final pass only.
    ///
    /// Saturation runs last, after [`with_contrast`](Self::with_contrast) and
    /// [`with_brightness`](Self::with_brightness), matching the CSS
    /// `backdrop-filter: blur() contrast() brightness() saturate()` order
    /// regardless of the order the builders are called in.
    pub fn with_saturation(mut self, saturation: f32) -> Self {
        self.transform.saturation = saturation.clamp(0.0, 4.0);
        self
    }

    /// Applies a post-blur CSS `contrast()` transform in the destination
    /// colour space: `(x - 0.5) * contrast + 0.5`, clamped. It runs first
    /// after the blur, before brightness and saturation. `1.0` is the identity.
    pub fn with_contrast(mut self, contrast: f32) -> Self {
        self.transform.contrast = contrast.clamp(0.0, 4.0);
        self
    }

    /// Applies a post-blur CSS `brightness()` multiply in the destination
    /// colour space, clamped. It runs after contrast and before saturation.
    /// `1.0` is the identity.
    pub fn with_brightness(mut self, brightness: f32) -> Self {
        self.transform.brightness = brightness.clamp(0.0, 4.0);
        self
    }

    /// Select CSS-compatible filtering over encoded sRGB channel values.
    pub fn with_encoded_srgb(mut self) -> Self {
        self.encoding = VulkanKawaseEncoding::EncodedSrgb;
        self
    }
}

/// Final encoded-sRGB eight-tap upsample fused into a clipped material draw.
/// The source remains the immutable half-resolution prefix captured before all cards.
#[derive(Debug, Clone, Copy)]
pub struct VulkanKawaseOutput {
    pub(super) extent: Size<i32, BufferCoord>,
    pub(super) upsample: bool,
    pub(super) offset: f32,
    pub(super) transform: KawaseColorTransform,
}
impl VulkanKawaseOutput {
    /// Create a fused final pass over the initialized origin-aligned source region.
    pub fn new(extent: Size<i32, BufferCoord>, offset: f32) -> Self {
        Self {
            extent,
            offset,
            upsample: true,
            transform: KawaseColorTransform::IDENTITY,
        }
    }
    /// Fuse the original five-tap zero-depth blur over one frozen full-size prefix.
    pub fn new_downsample(extent: Size<i32, BufferCoord>, offset: f32) -> Self {
        Self {
            extent,
            offset,
            upsample: false,
            transform: KawaseColorTransform::IDENTITY,
        }
    }
    /// Set the CSS filter-list transform, in contrast/brightness/saturation order.
    pub fn with_filter(mut self, contrast: f32, brightness: f32, saturation: f32) -> Self {
        self.transform = KawaseColorTransform {
            contrast: contrast.clamp(0.0, 4.0),
            brightness: brightness.clamp(0.0, 4.0),
            saturation: saturation.clamp(0.0, 4.0),
        };
        self
    }
}

#[derive(Debug)]
pub(super) struct ResolvedKawasePass {
    pub(super) source: Arc<VulkanImage>,
    pub(super) destination: Arc<VulkanImage>,
    pub(super) destination_view: vk::ImageView,
    pub(super) descriptor_set: vk::DescriptorSet,
    pub(super) pipeline: vk::Pipeline,
    pub(super) layout: vk::PipelineLayout,
    pub(super) render_pass: vk::RenderPass,
    pub(super) constants: KawasePushConstants,
    pub(super) destination_extent: Size<i32, BufferCoord>,
}

pub(super) fn kawase_halfpixel(
    source: Size<i32, BufferCoord>,
    destination: Size<i32, BufferCoord>,
) -> [f32; 2] {
    // Offsets are expressed relative to the SMALLER pyramid level (KWin's
    // convention): for a downsample that is the destination, for an upsample
    // the source.
    let smaller_w = source.w.min(destination.w).max(1) as f32;
    let smaller_h = source.h.min(destination.h).max(1) as f32;
    [0.5 / smaller_w, 0.5 / smaller_h]
}

impl VulkanRenderer {
    /// Prepare exactly the attachment-format banks used by these passes.
    ///
    /// This may create a render pass and native graphics pipelines and belongs
    /// on the renderer owner's cold resource turn. It records no commands,
    /// reads no source pixels, and submits no GPU work. Framebuffer capture
    /// only looks up these banks and fails if preparation was omitted.
    pub fn prepare_kawase_passes(&mut self, passes: &[VulkanKawasePass]) -> Result<(), VulkanRendererError> {
        for pass in passes {
            self.prepare_kawase_destination(&pass.destination, pass.encoding)?;
        }
        Ok(())
    }

    /// Prepare the encoded-sRGB attachment of a direct first downsample.
    ///
    /// Its source is the future active framebuffer, so the cold inventory
    /// names only the immutable destination. No lower scene is captured here.
    pub fn prepare_kawase_capture(&mut self, destination: &VulkanTexture) -> Result<(), VulkanRendererError> {
        self.prepare_kawase_destination(destination, VulkanKawaseEncoding::EncodedSrgb)
    }

    fn prepare_kawase_destination(
        &mut self,
        destination: &VulkanTexture,
        encoding: VulkanKawaseEncoding,
    ) -> Result<(), VulkanRendererError> {
        let image = destination
            .image_resource()
            .ok_or(VulkanRendererError::NotImplemented(
                "Kawase preparation requires an image-backed destination",
            ))?;
        if !image.usage().contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
            return Err(VulkanRendererError::TemporaryFailure(
                "Kawase destination image does not support color-attachment usage",
            ));
        }
        let format = encoding.destination_format(image.vk_format(), image.render_format());
        self.pipelines.pipelines_for_format(format).map(|_| ())
    }

    pub(super) fn resolve_kawase_passes(
        &mut self,
        passes: &[VulkanKawasePass],
    ) -> Result<Vec<ResolvedKawasePass>, VulkanRendererError> {
        let mut resolved = Vec::with_capacity(passes.len());
        self.resolve_kawase_passes_into(passes, &mut resolved)?;
        Ok(resolved)
    }

    pub(super) fn resolve_kawase_passes_into(
        &mut self,
        passes: &[VulkanKawasePass],
        resolved: &mut Vec<ResolvedKawasePass>,
    ) -> Result<(), VulkanRendererError> {
        let requested = resolved.len().saturating_add(passes.len());
        if requested > resolved.capacity() {
            return Err(VulkanRendererError::CommandStorageLimitExceeded {
                resource: "resolved Kawase passes",
                requested,
                limit: resolved.capacity(),
            });
        }
        for pass in passes {
            let Some(source) = pass.source.image_resource().cloned() else {
                return Err(VulkanRendererError::NotImplemented(
                    "kawase chain currently requires image-backed Vulkan source textures",
                ));
            };
            let Some(destination) = pass.destination.image_resource().cloned() else {
                return Err(VulkanRendererError::NotImplemented(
                    "kawase chain currently requires image-backed Vulkan destination textures",
                ));
            };
            for (extent, capacity) in [
                (pass.source_extent, source.size()),
                (pass.destination_extent, destination.size()),
            ] {
                if extent.w <= 0 || extent.h <= 0 || extent.w > capacity.w || extent.h > capacity.h {
                    return Err(VulkanRendererError::TemporaryFailure(
                        "kawase active extent exceeds image capacity",
                    ));
                }
            }
            if pass.source_origin.x < 0
                || pass.source_origin.y < 0
                || pass.source_origin.x > source.size().w - pass.source_extent.w
                || pass.source_origin.y > source.size().h - pass.source_extent.h
            {
                return Err(VulkanRendererError::TemporaryFailure(
                    "kawase source region exceeds image capacity",
                ));
            }
            if source.id() == destination.id() {
                return Err(VulkanRendererError::TemporaryFailure(
                    "kawase source and destination must be different images",
                ));
            }
            if !source.usage().contains(vk::ImageUsageFlags::SAMPLED) {
                return Err(VulkanRendererError::TemporaryFailure(
                    "kawase source image does not support sampled usage",
                ));
            }
            if !destination
                .usage()
                .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            {
                return Err(VulkanRendererError::TemporaryFailure(
                    "kawase destination image does not support color-attachment usage",
                ));
            }

            let encoded_srgb = pass.encoding == VulkanKawaseEncoding::EncodedSrgb;
            let destination_format = pass
                .encoding
                .destination_format(destination.vk_format(), destination.render_format());
            let destination_view = if encoded_srgb {
                destination.view()
            } else {
                destination.render_view()
            };
            let pipelines = self.pipelines.prepared_pipelines_for_format(destination_format)?;
            let descriptor_set = self
                .descriptors
                .texture_descriptor_set(source.view(), TextureSampler::LINEAR)?;
            let constants = KawasePushConstants::new(
                kawase_halfpixel(pass.source_extent, pass.destination_extent),
                pass.offset,
                pass.upsample,
                destination.blends_in_linear_light(),
                pass.transform,
                encoded_srgb,
            )
            .with_source_region(pass.source_origin, pass.source_extent, source.size());
            resolved.push(ResolvedKawasePass {
                source,
                destination,
                destination_view,
                descriptor_set,
                pipeline: pipelines.kawase_pipeline,
                layout: pipelines.kawase_layout,
                render_pass: pipelines.render_pass,
                constants,
                destination_extent: pass.destination_extent,
            });
        }
        Ok(())
    }

    /// Records and submits a dual-Kawase pyramid in one command buffer.
    ///
    /// Mirrors [`VulkanRenderer::blit_texture_chain`]'s contract: textures
    /// must be image-backed, layouts are restored after the chain, and the
    /// returned [`SyncPoint`] signals when the submission retires.
    #[instrument(level = "trace", skip(self, passes))]
    #[profiling::function]
    pub fn kawase_texture_chain(
        &mut self,
        passes: &[VulkanKawasePass],
    ) -> Result<SyncPoint, VulkanRendererError> {
        trace!(pass_count = passes.len(), "recording vulkan kawase chain");
        if passes.is_empty() {
            return Ok(SyncPoint::signaled());
        }

        // The standalone chain has no active output pass. Keep its explicit
        // preparation before recording; in-frame effects never take this path.
        self.prepare_kawase_passes(passes)?;
        let resolved = match self.resolve_kawase_passes(passes) {
            Ok(resolved) => resolved,
            Err(error) => {
                self.descriptors.abort_recording();
                return Err(error);
            }
        };

        let command_buffer = match self.device.acquire_command_buffer() {
            Ok(command_buffer) => command_buffer,
            Err(error) => {
                self.descriptors.abort_recording();
                return Err(error);
            }
        };
        let vk_device = self.device.device_handle();
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: Command buffer belongs to this device command pool and is not currently in-flight.
        if let Err(err) = unsafe { vk_device.begin_command_buffer(command_buffer, &begin_info) } {
            let _ = self.device.discard_command_buffer(command_buffer);
            self.descriptors.abort_recording();
            return Err(err.into());
        }
        self.device
            .insert_debug_label(command_buffer, c"vulkan.kawase_chain", [0.36, 0.65, 0.96, 1.0]);

        let mut foreign_images = acquire_images_from_foreign(
            vk_device,
            command_buffer,
            self.device.queue_family_index(),
            resolved.iter().flat_map(|pass| {
                [
                    (pass.source.clone(), pass.source.current_layout()),
                    (pass.destination.clone(), pass.destination.current_layout()),
                ]
            }),
        );
        let mut layouts = IndexMap::new();
        let mut framebuffers: Vec<vk::Framebuffer> = Vec::with_capacity(resolved.len());
        let mut record = || -> Result<(), VulkanRendererError> {
            for pass in &resolved {
                transition_tracked_image_layout(
                    vk_device,
                    command_buffer,
                    &mut layouts,
                    pass.source.clone(),
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
                transition_tracked_image_layout(
                    vk_device,
                    command_buffer,
                    &mut layouts,
                    pass.destination.clone(),
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                );

                let destination_size = pass.destination_extent;
                let extent = vk::Extent2D {
                    width: destination_size.w.max(1) as u32,
                    height: destination_size.h.max(1) as u32,
                };
                let attachments = [pass.destination_view];
                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(pass.render_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);
                // SAFETY: Device is valid and create info references live handles.
                let framebuffer = unsafe { vk_device.create_framebuffer(&framebuffer_info, None) }?;
                framebuffers.push(framebuffer);

                let render_area = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                };
                let render_pass_begin = vk::RenderPassBeginInfo::default()
                    .render_pass(pass.render_pass)
                    .framebuffer(framebuffer)
                    .render_area(render_area);
                // SAFETY: All handles are valid and recording is active; the
                // pass covers every pixel so LOAD contents are irrelevant.
                unsafe {
                    vk_device.cmd_begin_render_pass(
                        command_buffer,
                        &render_pass_begin,
                        vk::SubpassContents::INLINE,
                    );
                    vk_device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pass.pipeline,
                    );
                    vk_device.cmd_set_viewport(
                        command_buffer,
                        0,
                        &[vk::Viewport {
                            x: 0.0,
                            y: 0.0,
                            width: extent.width as f32,
                            height: extent.height as f32,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        }],
                    );
                    vk_device.cmd_set_scissor(command_buffer, 0, &[render_area]);
                    vk_device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pass.layout,
                        0,
                        &[pass.descriptor_set],
                        &[],
                    );
                    vk_device.cmd_push_constants(
                        command_buffer,
                        pass.layout,
                        vk::ShaderStageFlags::FRAGMENT,
                        0,
                        push_constants_bytes(&pass.constants),
                    );
                    vk_device.cmd_draw(command_buffer, 4, 1, 0, 0);
                    vk_device.cmd_end_render_pass(command_buffer);
                }
            }
            Ok(())
        };

        if let Err(err) = record() {
            let _ = self
                .device
                .discard_recording_resources(command_buffer, &mut framebuffers);
            self.descriptors.abort_recording();
            restore_unsubmitted_foreign_acquires(&foreign_images);
            return Err(err);
        }

        for tracked in layouts.values_mut() {
            transition_image_layout(
                vk_device,
                command_buffer,
                tracked.image.image(),
                tracked.current_layout,
                tracked.restore_layout,
            );
            tracked.current_layout = tracked.restore_layout;
        }
        for (id, (_, layout)) in foreign_images.iter_mut() {
            if let Some(tracked) = layouts.get(id) {
                *layout = tracked.current_layout;
            }
        }
        release_images_to_foreign(
            vk_device,
            command_buffer,
            self.device.queue_family_index(),
            &foreign_images,
        );

        // SAFETY: Command buffer recording is valid and all commands were encoded above.
        if let Err(err) = unsafe { vk_device.end_command_buffer(command_buffer) } {
            let _ = self
                .device
                .discard_recording_resources(command_buffer, &mut framebuffers);
            self.descriptors.abort_recording();
            restore_unsubmitted_foreign_acquires(&foreign_images);
            return Err(err.into());
        }

        let mut retained_images = resolved
            .iter()
            .flat_map(|pass| [pass.source.clone(), pass.destination.clone()])
            .collect::<Vec<_>>();
        let (submission_id, submission_fence) = match self.device.submit_with_resources_and_fence(
            command_buffer,
            &mut framebuffers,
            &mut retained_images,
        ) {
            Ok(submission) => submission,
            Err(err) => {
                self.descriptors.abort_recording();
                restore_unsubmitted_foreign_acquires(&foreign_images);
                return Err(err);
            }
        };
        self.descriptors.commit_submission(submission_id);

        for tracked in layouts.values() {
            tracked.image.set_layout(tracked.restore_layout);
        }
        commit_foreign_releases(&foreign_images);

        Ok(submission_fence)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::{
            allocator::Fourcc,
            renderer::{vulkan::VulkanKawasePass, Bind, Color32F, ExportMem, Frame, Offscreen, Renderer},
        },
        utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform},
    };

    use super::{VulkanKawaseEncoding, VulkanRenderer};

    #[test]
    fn kawase_preparation_uses_actual_encoded_or_render_attachment_format() {
        for (storage, render) in [
            (ash::vk::Format::R8G8B8A8_UNORM, ash::vk::Format::R8G8B8A8_SRGB),
            (ash::vk::Format::B8G8R8A8_UNORM, ash::vk::Format::B8G8R8A8_SRGB),
            (
                ash::vk::Format::R16G16B16A16_SFLOAT,
                ash::vk::Format::R16G16B16A16_SFLOAT,
            ),
        ] {
            assert_eq!(
                VulkanKawaseEncoding::EncodedSrgb.destination_format(storage, render),
                storage
            );
            assert_eq!(
                VulkanKawaseEncoding::LinearLight.destination_format(storage, render),
                render
            );
        }
    }

    /// The reviewed first-glass trigger: the output clear warmed only SRGB,
    /// while the real first downsample and filter destinations use UNORM.
    #[test]
    #[ignore = "requires native Vulkan; run explicitly for cold/first-glass evidence"]
    fn first_glass_requires_exact_cold_bank_and_preserves_active_frame_on_refusal() {
        let physical =
            crate::backend::renderer::vulkan::test_support::physical_device().expect("native Vulkan device");
        let mut renderer = VulkanRenderer::new(&physical).expect("native Vulkan renderer");
        let format = first_working_offscreen_format(&mut renderer).expect("offscreen format");
        let mut output = renderer.create_buffer(format, (16, 16).into()).unwrap();
        let capture = renderer.create_buffer(format, (8, 8).into()).unwrap();
        let filtered = renderer.create_buffer(format, (4, 4).into()).unwrap();
        let destination = capture.image_resource().unwrap();
        assert_ne!(
            destination.vk_format(),
            destination.render_format(),
            "fixture needs the native SRGB sibling"
        );
        let encoded_format = destination.vk_format();
        let passes = [VulkanKawasePass::new(&capture, &filtered, false, 1.5).with_encoded_srgb()];
        let size: Size<i32, Physical> = (16, 16).into();
        let area = Rectangle::from_size(size);
        {
            let mut target = renderer.bind(&mut output).unwrap();
            let mut frame = renderer.render(&mut target, size, Transform::Normal).unwrap();
            frame.clear(Color32F::new(0.2, 0.3, 0.4, 1.0), &[area]).unwrap();
            assert!(frame
                .capture_and_downsample_framebuffer(area, &capture, (8, 8).into(), 1.5, &passes)
                .is_err());
            // Refusal happened before cmdEndRenderPass: the original pass is
            // still usable and its displayed predecessor can be preserved.
            frame
                .draw_solid(area, &[area], Color32F::new(0.4, 0.3, 0.2, 1.0))
                .unwrap();
            frame.finish().unwrap().wait().unwrap();
        }
        assert!(renderer
            .pipelines
            .prepared_pipelines_for_format(encoded_format)
            .is_err());
        renderer.prepare_kawase_capture(&capture).unwrap();
        renderer.prepare_kawase_passes(&passes).unwrap();
        let prepared = renderer
            .pipelines
            .prepared_pipelines_for_format(encoded_format)
            .unwrap();
        for _ in 0..2 {
            let mut target = renderer.bind(&mut output).unwrap();
            let mut frame = renderer.render(&mut target, size, Transform::Normal).unwrap();
            frame.clear(Color32F::new(0.2, 0.3, 0.4, 1.0), &[area]).unwrap();
            frame
                .capture_and_downsample_framebuffer(area, &capture, (8, 8).into(), 1.5, &passes)
                .unwrap();
            frame.finish().unwrap().wait().unwrap();
            drop(target);
            let reused = renderer
                .pipelines
                .prepared_pipelines_for_format(encoded_format)
                .unwrap();
            assert_eq!(reused.render_pass, prepared.render_pass);
            assert_eq!(reused.kawase_pipeline, prepared.kawase_pipeline);
        }
    }

    fn init_renderer() -> Option<VulkanRenderer> {
        let physical_device = crate::backend::renderer::vulkan::test_support::physical_device()?;
        crate::backend::renderer::vulkan::test_support::renderer(&physical_device)
    }

    fn first_working_offscreen_format(renderer: &mut VulkanRenderer) -> Option<Fourcc> {
        super::super::test_support::present(
            [
                Fourcc::Argb8888,
                Fourcc::Abgr8888,
                Fourcc::Xrgb8888,
                Fourcc::Xbgr8888,
            ]
            .into_iter()
            .find(|format| renderer.create_buffer(*format, Size::from((4, 4))).is_ok()),
            "no supported offscreen format",
        )
    }

    /// A kawase down+up round trip of a solid color must reproduce that color
    /// (any resampling kernel is a partition of unity), and must run without
    /// validation/recording errors on a real device.
    #[test]
    fn kawase_round_trip_preserves_solid_color() {
        let Some(mut renderer) = init_renderer() else {
            return;
        };
        let Some(format) = first_working_offscreen_format(&mut renderer) else {
            return;
        };

        let full_size: Size<i32, BufferCoord> = Size::from((16, 16));
        let half_size: Size<i32, BufferCoord> = Size::from((8, 8));
        let physical_size: Size<i32, Physical> = Size::from((16, 16));
        let physical_region: Rectangle<i32, Physical> = Rectangle::from_size(physical_size);

        let mut full = renderer
            .create_buffer(format, full_size)
            .expect("offscreen alloc");
        let half = renderer
            .create_buffer(format, half_size)
            .expect("offscreen alloc");

        {
            let mut target = renderer.bind(&mut full).expect("bind");
            let mut frame = renderer
                .render(&mut target, physical_size, Transform::Normal)
                .expect("render");
            frame
                .clear(Color32F::new(0.5, 0.5, 0.5, 1.0), &[physical_region])
                .expect("clear");
            let sync = frame.finish().expect("finish");
            let _ = sync.wait();
        }

        let passes = [
            VulkanKawasePass::new(&full, &half, false, 1.5),
            VulkanKawasePass::new(&half, &full, true, 1.5),
        ];
        let sync = renderer
            .kawase_texture_chain(&passes)
            .expect("kawase chain should record and submit");
        let _ = sync.wait();

        let target = renderer.bind(&mut full).expect("bind for readback");
        let mapping = renderer
            .copy_framebuffer(&target, Rectangle::from_size(full_size), format)
            .expect("readback");
        let data = renderer.map_texture(&mapping).expect("map");
        // Sample the center pixel; a solid mid-gray must survive the round
        // trip within +-2/255 regardless of linearization.
        let center = ((8 * 16 + 8) * 4) as usize;
        let pixel = &data[center..center + 4];
        for channel in 0..3 {
            let value = pixel[channel] as i32;
            assert!(
                (value - 128).abs() <= 2,
                "solid gray should survive kawase round trip, got {pixel:?}"
            );
        }
    }

    #[test]
    fn final_kawase_pass_applies_saturation_transform() {
        let Some(mut renderer) = init_renderer() else {
            return;
        };
        let Some(format) = first_working_offscreen_format(&mut renderer) else {
            return;
        };

        let full_size: Size<i32, BufferCoord> = Size::from((16, 16));
        let half_size: Size<i32, BufferCoord> = Size::from((8, 8));
        let physical_size: Size<i32, Physical> = Size::from((16, 16));
        let physical_region: Rectangle<i32, Physical> = Rectangle::from_size(physical_size);
        let mut full = renderer
            .create_buffer(format, full_size)
            .expect("offscreen alloc");
        let half = renderer
            .create_buffer(format, half_size)
            .expect("offscreen alloc");

        {
            let mut target = renderer.bind(&mut full).expect("bind");
            let mut frame = renderer
                .render(&mut target, physical_size, Transform::Normal)
                .expect("render");
            frame
                .clear(Color32F::new(0.80, 0.20, 0.10, 1.0), &[physical_region])
                .expect("clear");
            frame.finish().expect("finish").wait().unwrap();
        }

        renderer
            .kawase_texture_chain(&[
                VulkanKawasePass::new(&full, &half, false, 1.5),
                VulkanKawasePass::new(&half, &full, true, 1.5).with_saturation(0.0),
            ])
            .expect("desaturating kawase chain")
            .wait()
            .unwrap();

        let target = renderer.bind(&mut full).expect("bind for readback");
        let mapping = renderer
            .copy_framebuffer(&target, Rectangle::from_size(full_size), format)
            .expect("readback");
        let data = renderer.map_texture(&mapping).expect("map");
        let center = ((8 * 16 + 8) * 4) as usize;
        let pixel = &data[center..center + 4];
        let rgb = &pixel[..3];
        let min = *rgb.iter().min().unwrap() as i32;
        let max = *rgb.iter().max().unwrap() as i32;
        assert!(
            max - min <= 2,
            "zero saturation must produce neutral RGB, got {pixel:?}"
        );
    }

    /// Renders one solid colour through a two-pass chain whose final pass
    /// carries `transform`, in the encoded-sRGB working space, and returns the
    /// centre pixel's stored bytes.
    fn filtered_solid_center_pixel(
        renderer: &mut VulkanRenderer,
        format: Fourcc,
        color: Color32F,
        transform: impl FnOnce(VulkanKawasePass) -> VulkanKawasePass,
    ) -> [u8; 4] {
        let full_size: Size<i32, BufferCoord> = Size::from((16, 16));
        let half_size: Size<i32, BufferCoord> = Size::from((8, 8));
        let physical_size: Size<i32, Physical> = Size::from((16, 16));
        let physical_region: Rectangle<i32, Physical> = Rectangle::from_size(physical_size);
        let mut full = renderer
            .create_buffer(format, full_size)
            .expect("offscreen alloc");
        let half = renderer
            .create_buffer(format, half_size)
            .expect("offscreen alloc");

        {
            let mut target = renderer.bind(&mut full).expect("bind");
            let mut frame = renderer
                .render(&mut target, physical_size, Transform::Normal)
                .expect("render");
            frame.clear(color, &[physical_region]).expect("clear");
            frame.finish().expect("finish").wait().unwrap();
        }

        renderer
            .kawase_texture_chain(&[
                VulkanKawasePass::new(&full, &half, false, 1.5).with_encoded_srgb(),
                transform(VulkanKawasePass::new(&half, &full, true, 1.5).with_encoded_srgb()),
            ])
            .expect("filtering kawase chain")
            .wait()
            .unwrap();

        let target = renderer.bind(&mut full).expect("bind for readback");
        let mapping = renderer
            .copy_framebuffer(&target, Rectangle::from_size(full_size), format)
            .expect("readback");
        let data = renderer.map_texture(&mapping).expect("map");
        let center = ((8 * 16 + 8) * 4) as usize;
        let mut pixel = [0_u8; 4];
        pixel.copy_from_slice(&data[center..center + 4]);
        pixel
    }

    /// The stored bytes of one readback pixel as `[r, g, b]`, undoing the
    /// fourcc's little-endian channel order.
    fn stored_rgb(format: Fourcc, pixel: [u8; 4]) -> [u8; 3] {
        match format {
            Fourcc::Argb8888 | Fourcc::Xrgb8888 => [pixel[2], pixel[1], pixel[0]],
            Fourcc::Abgr8888 | Fourcc::Xbgr8888 => [pixel[0], pixel[1], pixel[2]],
            other => panic!("unexpected readback format {other:?}"),
        }
    }

    fn assert_rgb_within(format: Fourcc, pixel: [u8; 4], expected: [u8; 3], tolerance: i32, what: &str) {
        let rgb = stored_rgb(format, pixel);
        for (channel, want) in expected.into_iter().enumerate() {
            let got = i32::from(rgb[channel]);
            assert!(
                (got - i32::from(want)).abs() <= tolerance,
                "{what}: channel {channel} expected {want} +-{tolerance}, got rgb {rgb:?}"
            );
        }
    }

    /// Contrast runs before brightness, and each primitive clamps before the
    /// next one reads it, exactly like a CSS filter list. Stored bytes of
    /// (0.8, 0.2, 0.4) through `contrast(2) brightness(0.4)`:
    ///
    /// - red: `(0.8 - 0.5) * 2 + 0.5 = 1.1` clamps to 1 before the multiply,
    ///   then `0.4` — a brightness-first order would give `0.14` instead;
    /// - green: `(0.2 - 0.5) * 2 + 0.5 = -0.1` clamps to 0 and stays there;
    /// - blue: `(0.4 - 0.5) * 2 + 0.5 = 0.3`, then `0.12`.
    #[test]
    fn final_kawase_pass_applies_contrast_then_brightness_with_css_clamping() {
        let Some(mut renderer) = init_renderer() else {
            return;
        };
        let Some(format) = first_working_offscreen_format(&mut renderer) else {
            return;
        };

        let pixel =
            filtered_solid_center_pixel(&mut renderer, format, Color32F::new(0.8, 0.2, 0.4, 1.0), |pass| {
                pass.with_brightness(0.4).with_contrast(2.0)
            });

        assert_rgb_within(format, pixel, [102, 0, 31], 2, "contrast(2) brightness(0.4)");
    }

    /// The identity transform leaves a solid colour on its stored bytes, so a
    /// recipe that declares neither contrast nor brightness renders exactly as
    /// it did before the two primitives existed.
    #[test]
    fn identity_contrast_and_brightness_preserve_the_stored_bytes() {
        let Some(mut renderer) = init_renderer() else {
            return;
        };
        let Some(format) = first_working_offscreen_format(&mut renderer) else {
            return;
        };

        let pixel =
            filtered_solid_center_pixel(&mut renderer, format, Color32F::new(0.8, 0.2, 0.4, 1.0), |pass| {
                pass.with_contrast(1.0).with_brightness(1.0).with_saturation(1.0)
            });

        assert_rgb_within(format, pixel, [204, 51, 102], 1, "identity transform");
    }

    /// The Avio dark glass recipe end to end: `contrast(1.35) brightness(0.4)
    /// saturate(2.9)` takes the prototype's pale sky `rgb(150 190 230)` to the
    /// deep blue it documents, `(0.11, 0.36, 0.55)`, with nothing clipped.
    #[test]
    fn dark_glass_transform_lands_the_prototypes_sky_on_its_documented_blue() {
        let Some(mut renderer) = init_renderer() else {
            return;
        };
        let Some(format) = first_working_offscreen_format(&mut renderer) else {
            return;
        };

        let pixel = filtered_solid_center_pixel(
            &mut renderer,
            format,
            Color32F::new(150.0 / 255.0, 190.0 / 255.0, 230.0 / 255.0, 1.0),
            |pass| pass.with_contrast(1.35).with_brightness(0.4).with_saturation(2.9),
        );

        assert_rgb_within(format, pixel, [28, 91, 141], 3, "dark glass over a pale sky");
    }

    fn region_energy(data: &[u8], width: usize, x_start: usize, x_end: usize) -> u64 {
        (8..24)
            .flat_map(|y| (x_start..x_end).map(move |x| (y * width + x) * 4))
            .map(|offset| {
                data[offset..offset + 3]
                    .iter()
                    .map(|channel| u64::from(*channel))
                    .sum::<u64>()
            })
            .sum()
    }

    /// Reusing the same renderer-local images must expose every new producer
    /// generation to the blur. A solid-color test cannot distinguish a fresh
    /// image from a stale one; this alternating pattern catches the exact
    /// persistent-pool failure mode that motivated explicit image provenance.
    #[test]
    fn reused_kawase_images_follow_alternating_pattern_generations() {
        let Some(mut renderer) = init_renderer() else {
            return;
        };
        let Some(format) = first_working_offscreen_format(&mut renderer) else {
            return;
        };

        let full_size: Size<i32, BufferCoord> = Size::from((32, 32));
        let half_size: Size<i32, BufferCoord> = Size::from((16, 16));
        let physical_size: Size<i32, Physical> = Size::from((32, 32));
        let full_region: Rectangle<i32, Physical> = Rectangle::from_size(physical_size);
        let mut full = renderer
            .create_buffer(format, full_size)
            .expect("full offscreen alloc");
        let half = renderer
            .create_buffer(format, half_size)
            .expect("half offscreen alloc");
        let full_image = full.image_resource().expect("full image resource").clone();
        let half_image = half.image_resource().expect("half image resource").clone();
        assert!(full_image.is_renderer_local());
        assert!(half_image.is_renderer_local());

        for generation in 0..32 {
            let bright_left = generation % 2 == 0;
            let bright_region =
                Rectangle::new((if bright_left { 0 } else { 16 }, 0).into(), Size::from((16, 32)));
            {
                let mut target = renderer.bind(&mut full).expect("bind producer target");
                let mut frame = renderer
                    .render(&mut target, physical_size, Transform::Normal)
                    .expect("render producer pattern");
                frame
                    .clear(Color32F::new(0.0, 0.0, 0.0, 1.0), &[full_region])
                    .expect("clear producer pattern");
                frame
                    .draw_solid(bright_region, &[bright_region], Color32F::new(1.0, 0.0, 0.0, 1.0))
                    .expect("draw producer pattern");
                frame.finish().expect("finish producer pattern").wait().unwrap();
            }

            let passes = [
                VulkanKawasePass::new(&full, &half, false, 1.5),
                VulkanKawasePass::new(&half, &full, true, 1.5),
            ];
            renderer
                .kawase_texture_chain(&passes)
                .expect("record reused kawase chain")
                .wait()
                .unwrap();

            let target = renderer.bind(&mut full).expect("bind reused result");
            let mapping = renderer
                .copy_framebuffer(&target, Rectangle::from_size(full_size), format)
                .expect("read reused result");
            let data = renderer.map_texture(&mapping).expect("map reused result");
            let left = region_energy(data, 32, 2, 14);
            let right = region_energy(data, 32, 18, 30);
            assert_eq!(
                left > right,
                bright_left,
                "generation {generation} returned stale or spatially inverted blur: left={left} right={right}",
            );
            assert!(!full_image.is_owned_by_foreign());
            assert!(!half_image.is_owned_by_foreign());
        }
    }
}
