//! Dual-Kawase blur chain for the Vulkan renderer.
//!
//! Replaces the plain bilinear blit pyramid for material blur: each pyramid
//! step is a fullscreen fragment pass sampling 5 (down) or 8 (up) bilinear
//! taps at half-pixel offsets (KWin's formulation), composing a
//! gaussian-like kernel from the same pixels the blit chain already touched
//! — equivalent bandwidth, marginal extra ALU. Passes optionally average in
//! linear light (sRGB decode/encode in-shader; storage stays sRGB-encoded so
//! 8-bit precision is preserved).

use std::sync::Arc;

use ash::vk;
use indexmap::IndexMap;
use tracing::{instrument, trace};

use crate::{
    backend::renderer::sync::SyncPoint,
    utils::{Buffer as BufferCoord, Size},
};

use super::{
    blit::{transition_image_layout, transition_tracked_image_layout},
    descriptor::TextureSampler,
    dmabuf::ImportedDmabufImage,
    pipeline::{push_constants_bytes, KawasePushConstants},
    VulkanRenderer, VulkanRendererError, VulkanTexture,
};

/// One dual-Kawase pyramid step: renders `source` into `destination` with the
/// down- or upsample kernel.
#[derive(Debug, Clone)]
pub struct VulkanKawasePass {
    source: VulkanTexture,
    destination: VulkanTexture,
    upsample: bool,
    offset: f32,
    linearize: bool,
}

impl VulkanKawasePass {
    /// Describes a kawase pass between two renderer textures.
    pub fn new(
        source: &VulkanTexture,
        destination: &VulkanTexture,
        upsample: bool,
        offset: f32,
        linearize: bool,
    ) -> Self {
        Self {
            source: source.clone(),
            destination: destination.clone(),
            upsample,
            offset,
            linearize,
        }
    }
}

struct ResolvedKawasePass {
    source: Arc<ImportedDmabufImage>,
    destination: Arc<ImportedDmabufImage>,
    descriptor_set: vk::DescriptorSet,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    constants: KawasePushConstants,
}

fn kawase_halfpixel(source: Size<i32, BufferCoord>, destination: Size<i32, BufferCoord>) -> [f32; 2] {
    // Offsets are expressed relative to the SMALLER pyramid level (KWin's
    // convention): for a downsample that is the destination, for an upsample
    // the source.
    let smaller_w = source.w.min(destination.w).max(1) as f32;
    let smaller_h = source.h.min(destination.h).max(1) as f32;
    [0.5 / smaller_w, 0.5 / smaller_h]
}

impl VulkanRenderer {
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

        let mut resolved = Vec::with_capacity(passes.len());
        for pass in passes {
            let Some(source) = pass.source.imported_image().cloned() else {
                return Err(VulkanRendererError::NotImplemented(
                    "kawase chain currently requires image-backed Vulkan source textures",
                ));
            };
            let Some(destination) = pass.destination.imported_image().cloned() else {
                return Err(VulkanRendererError::NotImplemented(
                    "kawase chain currently requires image-backed Vulkan destination textures",
                ));
            };
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

            let pipelines = self.pipelines.pipelines_for_format(destination.vk_format())?;
            let descriptor_set = self
                .descriptors
                .texture_descriptor_set(source.view(), TextureSampler::LINEAR)?;
            let constants = KawasePushConstants::new(
                kawase_halfpixel(source.size(), destination.size()),
                pass.offset,
                pass.upsample,
                pass.linearize,
            );
            resolved.push(ResolvedKawasePass {
                source,
                destination,
                descriptor_set,
                pipeline: pipelines.kawase_pipeline,
                layout: pipelines.kawase_layout,
                render_pass: pipelines.render_pass,
                constants,
            });
        }

        let command_buffer = self.device.acquire_command_buffer()?;
        let vk_device = self.device.device_handle();
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: Command buffer belongs to this device command pool and is not currently in-flight.
        if let Err(err) = unsafe { vk_device.begin_command_buffer(command_buffer, &begin_info) } {
            let _ = self.device.discard_command_buffer(command_buffer);
            return Err(err.into());
        }
        self.device
            .insert_debug_label(command_buffer, c"vulkan.kawase_chain", [0.36, 0.65, 0.96, 1.0]);

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

                let destination_size = pass.destination.size();
                let extent = vk::Extent2D {
                    width: destination_size.w.max(1) as u32,
                    height: destination_size.h.max(1) as u32,
                };
                let attachments = [pass.destination.render_view()];
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
            for framebuffer in framebuffers.drain(..) {
                // SAFETY: Framebuffers were created above and the command
                // buffer will never be submitted.
                unsafe { vk_device.destroy_framebuffer(framebuffer, None) };
            }
            let _ = self.device.discard_command_buffer(command_buffer);
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

        // SAFETY: Command buffer recording is valid and all commands were encoded above.
        if let Err(err) = unsafe { vk_device.end_command_buffer(command_buffer) } {
            for framebuffer in framebuffers.drain(..) {
                // SAFETY: Created above; command buffer will not be submitted.
                unsafe { vk_device.destroy_framebuffer(framebuffer, None) };
            }
            let _ = self.device.discard_command_buffer(command_buffer);
            return Err(err.into());
        }

        let retained_images = resolved
            .iter()
            .flat_map(|pass| [pass.source.clone(), pass.destination.clone()])
            .collect::<Vec<_>>();
        let (_, submission_fence) =
            match self
                .device
                .submit_with_resources_and_fence(command_buffer, framebuffers, retained_images)
            {
                Ok(submission) => submission,
                Err(err) => {
                    let _ = self.device.discard_command_buffer(command_buffer);
                    return Err(err);
                }
            };

        for tracked in layouts.values() {
            tracked.image.set_layout(tracked.restore_layout);
        }

        Ok(SyncPoint::from(submission_fence))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::{
            allocator::Fourcc,
            renderer::{vulkan::VulkanKawasePass, Bind, Color32F, ExportMem, Frame, Offscreen, Renderer},
            vulkan::{version::Version, Instance, PhysicalDevice},
        },
        utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform},
    };

    use super::VulkanRenderer;

    fn init_renderer() -> Option<VulkanRenderer> {
        let instance = Instance::new(Version::VERSION_1_3, None).ok()?;
        let physical_device = PhysicalDevice::enumerate(&instance).ok()?.next()?;
        VulkanRenderer::new(&physical_device).ok()
    }

    fn first_working_offscreen_format(renderer: &mut VulkanRenderer) -> Option<Fourcc> {
        [
            Fourcc::Argb8888,
            Fourcc::Abgr8888,
            Fourcc::Xrgb8888,
            Fourcc::Xbgr8888,
        ]
        .into_iter()
        .find(|format| renderer.create_buffer(*format, Size::from((4, 4))).is_ok())
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
            VulkanKawasePass::new(&full, &half, false, 1.5, true),
            VulkanKawasePass::new(&half, &full, true, 1.5, true),
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
}
