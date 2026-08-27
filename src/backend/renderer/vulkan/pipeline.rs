use std::{io::Cursor, sync::Arc};

use ash::{util::read_spv, vk};
use indexmap::IndexMap;

use crate::backend::renderer::{TextureRenderEffect, TextureRenderEffectKind};

use super::{device::DeviceHandle, format::ColorEncoding, VulkanRendererError};

const SOLID_VERTEX_SHADER_SPV: &[u8] = include_bytes!("shaders/solid.vert.spv");
const SOLID_FRAGMENT_SHADER_SPV: &[u8] = include_bytes!("shaders/solid.frag.spv");
const TEXTURE_VERTEX_SHADER_SPV: &[u8] = include_bytes!("shaders/texture.vert.spv");
const TEXTURE_FRAGMENT_SHADER_SPV: &[u8] = include_bytes!("shaders/texture.frag.spv");
const KAWASE_FRAGMENT_SHADER_SPV: &[u8] = include_bytes!("shaders/kawase.frag.spv");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub(crate) enum TextureTransform {
    #[default]
    Normal = 0,
    Rotate90 = 1,
    Rotate180 = 2,
    Rotate270 = 3,
    Flipped = 4,
    Flipped90 = 5,
    Flipped180 = 6,
    Flipped270 = 7,
}

impl From<crate::utils::Transform> for TextureTransform {
    fn from(value: crate::utils::Transform) -> Self {
        match value {
            crate::utils::Transform::Normal => TextureTransform::Normal,
            crate::utils::Transform::_90 => TextureTransform::Rotate90,
            crate::utils::Transform::_180 => TextureTransform::Rotate180,
            crate::utils::Transform::_270 => TextureTransform::Rotate270,
            crate::utils::Transform::Flipped => TextureTransform::Flipped,
            crate::utils::Transform::Flipped90 => TextureTransform::Flipped90,
            crate::utils::Transform::Flipped180 => TextureTransform::Flipped180,
            crate::utils::Transform::Flipped270 => TextureTransform::Flipped270,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SolidPushConstants {
    pub(crate) color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct TexturePushConstants {
    pub(crate) alpha: f32,
    pub(crate) transform: u32,
    pub(crate) y_inverted: u32,
    pub(crate) rounded_clip_flags: u32,
    pub(crate) src_offset: [f32; 2],
    pub(crate) src_scale: [f32; 2],
    pub(crate) clip_rect: [f32; 4],
    pub(crate) clip_params: [f32; 4],
    pub(crate) effect: [f32; 4],
    pub(crate) effect_params: [f32; 4],
    /// How the sampled texels relate to linear light, and therefore which conversion
    /// the shader applies before the blend. See [`ColorEncoding`].
    pub(crate) source_encoding: u32,
    /// Physical pixels per authored logical pixel for parametric clips.
    pub(crate) clip_scale: f32,
}

const BOTTOM_EDGE_CLIP_FLAG: u32 = 1 << 31;
const CLIP_TRANSFORM_SHIFT: u32 = 8;

/// Shader-side spelling of [`ColorEncoding`]. Kept next to the push-constant struct so
/// the two stay in step; `texture.frag` reads the same numbering.
pub(crate) const SOURCE_ENCODING_ELECTRICAL_PREMULTIPLIED: u32 = 0;
pub(crate) const SOURCE_ENCODING_LINEAR_PREMULTIPLIED: u32 = 1;
/// The target blends in gamma space (no `_SRGB` sibling), so no conversion happens.
pub(crate) const SOURCE_ENCODING_PASSTHROUGH: u32 = 2;

impl From<ColorEncoding> for u32 {
    fn from(encoding: ColorEncoding) -> Self {
        match encoding {
            ColorEncoding::ElectricalPremultiplied => SOURCE_ENCODING_ELECTRICAL_PREMULTIPLIED,
            ColorEncoding::LinearPremultiplied => SOURCE_ENCODING_LINEAR_PREMULTIPLIED,
        }
    }
}

impl Default for TexturePushConstants {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            transform: TextureTransform::Normal as u32,
            y_inverted: 0,
            rounded_clip_flags: 0,
            src_offset: [0.0, 0.0],
            src_scale: [1.0, 1.0],
            clip_rect: [0.0, 0.0, 0.0, 0.0],
            clip_params: [0.0, 2.0, 0.5, 0.0],
            effect: [TextureRenderEffectKind::None as u32 as f32, 0.0, 0.5, 0.5],
            effect_params: [0.0, 0.0, 0.0, 0.0],
            source_encoding: SOURCE_ENCODING_PASSTHROUGH,
            clip_scale: 1.0,
        }
    }
}

impl TexturePushConstants {
    pub(crate) fn new(alpha: f32, transform: TextureTransform, y_inverted: bool) -> Self {
        Self {
            alpha,
            transform: transform as u32,
            y_inverted: u32::from(y_inverted),
            rounded_clip_flags: 0,
            src_offset: [0.0, 0.0],
            src_scale: [1.0, 1.0],
            clip_rect: [0.0, 0.0, 0.0, 0.0],
            clip_params: [0.0, 2.0, 0.5, 0.0],
            effect: [TextureRenderEffectKind::None as u32 as f32, 0.0, 0.5, 0.5],
            effect_params: [0.0, 0.0, 0.0, 0.0],
            source_encoding: SOURCE_ENCODING_PASSTHROUGH,
            clip_scale: 1.0,
        }
    }

    /// Declares how the sampled texels relate to linear light.
    ///
    /// `linear_blending` is a property of the bound target: when it is false the pass
    /// blends in gamma space and texels must reach the blend exactly as stored.
    pub(crate) fn with_source_encoding(mut self, encoding: ColorEncoding, linear_blending: bool) -> Self {
        self.source_encoding = if linear_blending {
            encoding.into()
        } else {
            SOURCE_ENCODING_PASSTHROUGH
        };
        self
    }

    pub(crate) fn with_src_rect(mut self, offset: [f32; 2], scale: [f32; 2]) -> Self {
        self.src_offset = offset;
        self.src_scale = scale;
        self
    }

    pub(crate) fn with_rounded_clip(mut self, flags: u32, rect: [f32; 4], params: [f32; 4]) -> Self {
        self.rounded_clip_flags = flags;
        self.clip_rect = rect;
        self.clip_params = params;
        self
    }

    pub(crate) fn with_bottom_edge_clip(
        mut self,
        transform: TextureTransform,
        rect: [f32; 4],
        params: [f32; 4],
        geometry_scale: f32,
    ) -> Self {
        self.rounded_clip_flags = BOTTOM_EDGE_CLIP_FLAG | ((transform as u32) << CLIP_TRANSFORM_SHIFT);
        self.clip_rect = rect;
        self.clip_params = params;
        self.clip_scale = geometry_scale;
        self
    }

    pub(crate) fn with_effect(mut self, effect: TextureRenderEffect) -> Self {
        self.effect = [
            effect.kind as u32 as f32,
            effect.progress.clamp(0.0, 1.0),
            effect.anchor[0],
            effect.anchor[1],
        ];
        self.effect_params = effect.params;
        self
    }
}

/// Push constants for the dual-Kawase blur pass. Layout mirrors the GLSL
/// push-constant block in `shaders/kawase.frag` (std430: vec2 at offset 0,
/// scalars packed after, padded to 32 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct KawasePushConstants {
    pub(crate) halfpixel: [f32; 2],
    pub(crate) offset: f32,
    pub(crate) mode: u32,
    /// 1 when the shader must sRGB-encode its own output because the destination is
    /// not viewed through an `_SRGB` attachment. Taps are always decoded to linear.
    pub(crate) encode_output: u32,
    /// Saturation applied after blur. Intermediate pyramid passes use `1.0`;
    /// material graphs put their colour transform on the final pass only.
    pub(crate) saturation: f32,
    /// 1 when filtering intentionally operates on encoded sRGB channel values.
    pub(crate) encoded_srgb: u32,
    pub(crate) _pad: u32,
}

impl KawasePushConstants {
    pub(crate) fn new(
        halfpixel: [f32; 2],
        offset: f32,
        upsample: bool,
        linear_destination: bool,
        saturation: f32,
        encoded_srgb: bool,
    ) -> Self {
        Self {
            halfpixel,
            offset,
            mode: u32::from(upsample),
            encode_output: u32::from(!linear_destination),
            saturation: saturation.clamp(0.0, 4.0),
            encoded_srgb: u32::from(encoded_srgb),
            _pad: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PipelineHandles {
    pub(crate) render_pass: vk::RenderPass,
    pub(crate) solid_pipeline: vk::Pipeline,
    pub(crate) solid_opaque_pipeline: vk::Pipeline,
    pub(crate) textured_pipeline: vk::Pipeline,
    pub(crate) textured_opaque_pipeline: vk::Pipeline,
    pub(crate) kawase_pipeline: vk::Pipeline,
    pub(crate) solid_layout: vk::PipelineLayout,
    pub(crate) textured_layout: vk::PipelineLayout,
    pub(crate) kawase_layout: vk::PipelineLayout,
}

#[derive(Debug)]
struct FormatPipelineSet {
    render_pass: vk::RenderPass,
    solid_pipeline: vk::Pipeline,
    solid_opaque_pipeline: vk::Pipeline,
    textured_pipeline: vk::Pipeline,
    textured_opaque_pipeline: vk::Pipeline,
    kawase_pipeline: vk::Pipeline,
}

#[derive(Debug)]
pub(crate) struct PipelineState {
    device: Arc<DeviceHandle>,
    pipeline_cache: vk::PipelineCache,
    solid_layout: vk::PipelineLayout,
    textured_layout: vk::PipelineLayout,
    solid_vertex_module: vk::ShaderModule,
    solid_fragment_module: vk::ShaderModule,
    texture_vertex_module: vk::ShaderModule,
    texture_fragment_module: vk::ShaderModule,
    kawase_fragment_module: vk::ShaderModule,
    kawase_layout: vk::PipelineLayout,
    per_format: IndexMap<vk::Format, FormatPipelineSet>,
}

impl PipelineState {
    pub(crate) fn new(
        device: Arc<DeviceHandle>,
        texture_descriptor_layout: vk::DescriptorSetLayout,
    ) -> Result<Self, VulkanRendererError> {
        Self::with_initial_cache(device, texture_descriptor_layout, &[])
    }

    pub(crate) fn with_initial_cache(
        device: Arc<DeviceHandle>,
        texture_descriptor_layout: vk::DescriptorSetLayout,
        initial_cache_data: &[u8],
    ) -> Result<Self, VulkanRendererError> {
        let vk_device = device.handle();

        let cache_info = vk::PipelineCacheCreateInfo::default().initial_data(initial_cache_data);
        // SAFETY: Device is valid and create-info references live data.
        let pipeline_cache = unsafe { vk_device.create_pipeline_cache(&cache_info, None) }?;

        let solid_vertex_module = match create_shader_module(vk_device, SOLID_VERTEX_SHADER_SPV) {
            Ok(module) => module,
            Err(err) => {
                unsafe { vk_device.destroy_pipeline_cache(pipeline_cache, None) };
                return Err(err);
            }
        };

        let solid_fragment_module = match create_shader_module(vk_device, SOLID_FRAGMENT_SHADER_SPV) {
            Ok(module) => module,
            Err(err) => {
                unsafe {
                    vk_device.destroy_shader_module(solid_vertex_module, None);
                    vk_device.destroy_pipeline_cache(pipeline_cache, None);
                }
                return Err(err);
            }
        };

        let texture_vertex_module = match create_shader_module(vk_device, TEXTURE_VERTEX_SHADER_SPV) {
            Ok(module) => module,
            Err(err) => {
                unsafe {
                    vk_device.destroy_shader_module(solid_fragment_module, None);
                    vk_device.destroy_shader_module(solid_vertex_module, None);
                    vk_device.destroy_pipeline_cache(pipeline_cache, None);
                }
                return Err(err);
            }
        };

        let texture_fragment_module = match create_shader_module(vk_device, TEXTURE_FRAGMENT_SHADER_SPV) {
            Ok(module) => module,
            Err(err) => {
                unsafe {
                    vk_device.destroy_shader_module(texture_vertex_module, None);
                    vk_device.destroy_shader_module(solid_fragment_module, None);
                    vk_device.destroy_shader_module(solid_vertex_module, None);
                    vk_device.destroy_pipeline_cache(pipeline_cache, None);
                }
                return Err(err);
            }
        };

        let solid_push_constants = [vk::PushConstantRange::default()
            .offset(0)
            .size(std::mem::size_of::<SolidPushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let solid_layout_info =
            vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&solid_push_constants);

        // SAFETY: Device is valid and create-info references live data.
        let solid_layout = match unsafe { vk_device.create_pipeline_layout(&solid_layout_info, None) } {
            Ok(layout) => layout,
            Err(err) => {
                unsafe {
                    vk_device.destroy_shader_module(texture_fragment_module, None);
                    vk_device.destroy_shader_module(texture_vertex_module, None);
                    vk_device.destroy_shader_module(solid_fragment_module, None);
                    vk_device.destroy_shader_module(solid_vertex_module, None);
                    vk_device.destroy_pipeline_cache(pipeline_cache, None);
                }
                return Err(err.into());
            }
        };

        let textured_push_constants = [vk::PushConstantRange::default()
            .offset(0)
            .size(std::mem::size_of::<TexturePushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let textured_set_layouts = [texture_descriptor_layout];
        let textured_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&textured_set_layouts)
            .push_constant_ranges(&textured_push_constants);

        // SAFETY: Device is valid and create-info references live data.
        let textured_layout = match unsafe { vk_device.create_pipeline_layout(&textured_layout_info, None) } {
            Ok(layout) => layout,
            Err(err) => {
                unsafe {
                    vk_device.destroy_pipeline_layout(solid_layout, None);
                    vk_device.destroy_shader_module(texture_fragment_module, None);
                    vk_device.destroy_shader_module(texture_vertex_module, None);
                    vk_device.destroy_shader_module(solid_fragment_module, None);
                    vk_device.destroy_shader_module(solid_vertex_module, None);
                    vk_device.destroy_pipeline_cache(pipeline_cache, None);
                }
                return Err(err.into());
            }
        };

        let kawase_fragment_module = match create_shader_module(vk_device, KAWASE_FRAGMENT_SHADER_SPV) {
            Ok(module) => module,
            Err(err) => {
                unsafe {
                    vk_device.destroy_pipeline_layout(textured_layout, None);
                    vk_device.destroy_pipeline_layout(solid_layout, None);
                    vk_device.destroy_shader_module(texture_fragment_module, None);
                    vk_device.destroy_shader_module(texture_vertex_module, None);
                    vk_device.destroy_shader_module(solid_fragment_module, None);
                    vk_device.destroy_shader_module(solid_vertex_module, None);
                    vk_device.destroy_pipeline_cache(pipeline_cache, None);
                }
                return Err(err);
            }
        };

        let kawase_push_constants = [vk::PushConstantRange::default()
            .offset(0)
            .size(std::mem::size_of::<KawasePushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let kawase_set_layouts = [texture_descriptor_layout];
        let kawase_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&kawase_set_layouts)
            .push_constant_ranges(&kawase_push_constants);

        // SAFETY: Device is valid and create-info references live data.
        let kawase_layout = match unsafe { vk_device.create_pipeline_layout(&kawase_layout_info, None) } {
            Ok(layout) => layout,
            Err(err) => {
                unsafe {
                    vk_device.destroy_shader_module(kawase_fragment_module, None);
                    vk_device.destroy_pipeline_layout(textured_layout, None);
                    vk_device.destroy_pipeline_layout(solid_layout, None);
                    vk_device.destroy_shader_module(texture_fragment_module, None);
                    vk_device.destroy_shader_module(texture_vertex_module, None);
                    vk_device.destroy_shader_module(solid_fragment_module, None);
                    vk_device.destroy_shader_module(solid_vertex_module, None);
                    vk_device.destroy_pipeline_cache(pipeline_cache, None);
                }
                return Err(err.into());
            }
        };

        Ok(Self {
            device,
            pipeline_cache,
            solid_layout,
            textured_layout,
            solid_vertex_module,
            solid_fragment_module,
            texture_vertex_module,
            texture_fragment_module,
            kawase_fragment_module,
            kawase_layout,
            per_format: IndexMap::new(),
        })
    }

    pub(crate) fn kawase_layout(&self) -> vk::PipelineLayout {
        self.kawase_layout
    }

    pub(crate) fn solid_layout(&self) -> vk::PipelineLayout {
        self.solid_layout
    }

    pub(crate) fn textured_layout(&self) -> vk::PipelineLayout {
        self.textured_layout
    }

    pub(crate) fn pipeline_cache_data(&self) -> Result<Vec<u8>, VulkanRendererError> {
        // SAFETY: Pipeline cache belongs to this device and is valid while `self` is alive.
        Ok(unsafe { self.device.handle().get_pipeline_cache_data(self.pipeline_cache) }?)
    }

    pub(crate) fn merge_pipeline_cache_data(&mut self, cache_data: &[u8]) -> Result<(), VulkanRendererError> {
        if cache_data.is_empty() {
            return Ok(());
        }

        let vk_device = self.device.handle();
        let cache_info = vk::PipelineCacheCreateInfo::default().initial_data(cache_data);

        // SAFETY: Device is valid and create-info references live data.
        let external_cache = unsafe { vk_device.create_pipeline_cache(&cache_info, None) }?;

        // SAFETY: Both pipeline caches belong to this device and are valid handles.
        let merge_result = unsafe { vk_device.merge_pipeline_caches(self.pipeline_cache, &[external_cache]) };

        // SAFETY: Temporary cache is no longer needed after merge attempt.
        unsafe { vk_device.destroy_pipeline_cache(external_cache, None) };

        merge_result?;
        Ok(())
    }

    pub(crate) fn pipelines_for_format(
        &mut self,
        format: vk::Format,
    ) -> Result<PipelineHandles, VulkanRendererError> {
        if !self.per_format.contains_key(&format) {
            let pipelines = self.create_format_pipeline_set(format)?;
            self.per_format.insert(format, pipelines);
        }

        let set = self
            .per_format
            .get(&format)
            .expect("pipelines inserted for requested format");

        Ok(PipelineHandles {
            render_pass: set.render_pass,
            solid_pipeline: set.solid_pipeline,
            solid_opaque_pipeline: set.solid_opaque_pipeline,
            textured_pipeline: set.textured_pipeline,
            textured_opaque_pipeline: set.textured_opaque_pipeline,
            kawase_pipeline: set.kawase_pipeline,
            solid_layout: self.solid_layout,
            textured_layout: self.textured_layout,
            kawase_layout: self.kawase_layout,
        })
    }

    fn create_format_pipeline_set(
        &self,
        format: vk::Format,
    ) -> Result<FormatPipelineSet, VulkanRendererError> {
        let vk_device = self.device.handle();
        let render_pass = create_render_pass(vk_device, format)?;

        let solid_pipeline = match self.create_graphics_pipeline(
            render_pass,
            self.solid_layout,
            self.solid_vertex_module,
            self.solid_fragment_module,
            true,
        ) {
            Ok(pipeline) => pipeline,
            Err(err) => {
                unsafe { vk_device.destroy_render_pass(render_pass, None) };
                return Err(err);
            }
        };

        let solid_opaque_pipeline = match self.create_graphics_pipeline(
            render_pass,
            self.solid_layout,
            self.solid_vertex_module,
            self.solid_fragment_module,
            false,
        ) {
            Ok(pipeline) => pipeline,
            Err(err) => {
                unsafe {
                    vk_device.destroy_pipeline(solid_pipeline, None);
                    vk_device.destroy_render_pass(render_pass, None);
                }
                return Err(err);
            }
        };

        let textured_pipeline = match self.create_graphics_pipeline(
            render_pass,
            self.textured_layout,
            self.texture_vertex_module,
            self.texture_fragment_module,
            true,
        ) {
            Ok(pipeline) => pipeline,
            Err(err) => {
                unsafe {
                    vk_device.destroy_pipeline(solid_opaque_pipeline, None);
                    vk_device.destroy_pipeline(solid_pipeline, None);
                    vk_device.destroy_render_pass(render_pass, None);
                }
                return Err(err);
            }
        };

        let textured_opaque_pipeline = match self.create_graphics_pipeline(
            render_pass,
            self.textured_layout,
            self.texture_vertex_module,
            self.texture_fragment_module,
            false,
        ) {
            Ok(pipeline) => pipeline,
            Err(err) => {
                unsafe {
                    vk_device.destroy_pipeline(textured_pipeline, None);
                    vk_device.destroy_pipeline(solid_opaque_pipeline, None);
                    vk_device.destroy_pipeline(solid_pipeline, None);
                    vk_device.destroy_render_pass(render_pass, None);
                }
                return Err(err);
            }
        };

        // Kawase writes every covered pixel; blending stays disabled so the
        // pass is a pure resample (opaque overwrite).
        let kawase_pipeline = match self.create_graphics_pipeline(
            render_pass,
            self.kawase_layout,
            self.texture_vertex_module,
            self.kawase_fragment_module,
            false,
        ) {
            Ok(pipeline) => pipeline,
            Err(err) => {
                unsafe {
                    vk_device.destroy_pipeline(textured_opaque_pipeline, None);
                    vk_device.destroy_pipeline(textured_pipeline, None);
                    vk_device.destroy_pipeline(solid_opaque_pipeline, None);
                    vk_device.destroy_pipeline(solid_pipeline, None);
                    vk_device.destroy_render_pass(render_pass, None);
                }
                return Err(err);
            }
        };

        Ok(FormatPipelineSet {
            render_pass,
            solid_pipeline,
            solid_opaque_pipeline,
            textured_pipeline,
            textured_opaque_pipeline,
            kawase_pipeline,
        })
    }

    fn create_graphics_pipeline(
        &self,
        render_pass: vk::RenderPass,
        layout: vk::PipelineLayout,
        vertex_shader_module: vk::ShaderModule,
        fragment_shader_module: vk::ShaderModule,
        blend_enabled: bool,
    ) -> Result<vk::Pipeline, VulkanRendererError> {
        let vk_device = self.device.handle();

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader_module)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader_module)
                .name(c"main"),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_STRIP)
            .primitive_restart_enable(false);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        // Shader outputs are premultiplied, matching Wayland/Impeller texture
        // contents and the compositor's solid color path.
        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(blend_enabled)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let create_info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0)];

        // SAFETY: Device and pipeline cache are valid; create-info references live data.
        let pipelines =
            unsafe { vk_device.create_graphics_pipelines(self.pipeline_cache, &create_info, None) }.map_err(
                |(pipelines, err)| {
                    for pipeline in pipelines {
                        unsafe { vk_device.destroy_pipeline(pipeline, None) };
                    }
                    VulkanRendererError::from(err)
                },
            )?;

        pipelines
            .into_iter()
            .next()
            .ok_or(VulkanRendererError::TemporaryFailure(
                "Vulkan did not return a graphics pipeline",
            ))
    }
}

impl Drop for PipelineState {
    fn drop(&mut self) {
        // Skipped on a lost device: destroying these objects on a lost VkDevice faults on NVIDIA.
        // `destroy_with` is the single ownership-encoded teardown gate; a no-op when lost.
        let per_format = std::mem::take(&mut self.per_format);
        self.device.destroy_with(|device| {
            for (_, set) in per_format {
                unsafe {
                    device.destroy_pipeline(set.kawase_pipeline, None);
                    device.destroy_pipeline(set.textured_opaque_pipeline, None);
                    device.destroy_pipeline(set.textured_pipeline, None);
                    device.destroy_pipeline(set.solid_opaque_pipeline, None);
                    device.destroy_pipeline(set.solid_pipeline, None);
                    device.destroy_render_pass(set.render_pass, None);
                }
            }

            unsafe {
                device.destroy_shader_module(self.kawase_fragment_module, None);
                device.destroy_shader_module(self.texture_fragment_module, None);
                device.destroy_shader_module(self.texture_vertex_module, None);
                device.destroy_shader_module(self.solid_fragment_module, None);
                device.destroy_shader_module(self.solid_vertex_module, None);
                device.destroy_pipeline_layout(self.kawase_layout, None);
                device.destroy_pipeline_layout(self.textured_layout, None);
                device.destroy_pipeline_layout(self.solid_layout, None);
                device.destroy_pipeline_cache(self.pipeline_cache, None);
            }
        });
    }
}

pub(crate) fn push_constants_bytes<T>(constants: &T) -> &[u8] {
    // SAFETY: `constants` is a plain old data struct with `#[repr(C)]` and no references.
    unsafe { std::slice::from_raw_parts((constants as *const T).cast::<u8>(), std::mem::size_of::<T>()) }
}

fn create_render_pass(
    device: &ash::Device,
    format: vk::Format,
) -> Result<vk::RenderPass, VulkanRendererError> {
    let attachments = [vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

    let color_attachments = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachments)];

    let dependencies = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::COLOR_ATTACHMENT_READ)];

    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    // SAFETY: Device is valid and create-info references live data.
    Ok(unsafe { device.create_render_pass(&render_pass_info, None) }?)
}

fn create_shader_module(device: &ash::Device, bytes: &[u8]) -> Result<vk::ShaderModule, VulkanRendererError> {
    let code = read_spv(&mut Cursor::new(bytes))?;
    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);

    // SAFETY: Device is valid and shader module create-info references live memory.
    Ok(unsafe { device.create_shader_module(&create_info, None) }?)
}

#[cfg(test)]
mod tests {
    use ash::vk;

    use crate::backend::{
        renderer::{TextureRenderEffect, TextureRenderEffectKind},
        vulkan::{version::Version, Instance, PhysicalDevice},
    };

    use super::{
        super::descriptor::{DescriptorState, TextureSampler},
        super::device::DeviceHandle,
        super::device::DeviceState,
        push_constants_bytes, PipelineState, SolidPushConstants, TexturePushConstants, TextureTransform,
        BOTTOM_EDGE_CLIP_FLAG, CLIP_TRANSFORM_SHIFT, SOURCE_ENCODING_ELECTRICAL_PREMULTIPLIED,
        SOURCE_ENCODING_LINEAR_PREMULTIPLIED, SOURCE_ENCODING_PASSTHROUGH,
    };

    const TEST_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
    const TEST_WIDTH: u32 = 64;
    const TEST_HEIGHT: u32 = 64;

    fn create_mutable_test_image(
        device: &DeviceState,
        view_format: vk::Format,
        usage: vk::ImageUsageFlags,
        extent: vk::Extent3D,
    ) -> Result<TestImage, vk::Result> {
        let vk_device = device.device_handle();
        let view_formats = [vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_SRGB];
        let mut format_list = vk::ImageFormatListCreateInfo::default().view_formats(&view_formats);
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut format_list);

        let image = unsafe { vk_device.create_image(&image_create_info, None) }?;
        let memory_requirements = unsafe { vk_device.get_image_memory_requirements(image) };
        let memory_type_index = pick_image_memory_type(device, memory_requirements.memory_type_bits)
            .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);
        let memory = unsafe { vk_device.allocate_memory(&allocate_info, None) }?;
        unsafe { vk_device.bind_image_memory(image, memory, 0) }?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(view_format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let view = unsafe { vk_device.create_image_view(&view_info, None) }?;

        Ok(TestImage {
            device: device.shared_device(),
            image,
            memory,
            view,
        })
    }

    fn create_upload_buffer(device: &DeviceState, bytes: &[u8]) -> Result<TestBuffer, vk::Result> {
        let vk_device = device.device_handle();
        let size = bytes.len();
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { vk_device.create_buffer(&buffer_create_info, None) }?;
        let memory_requirements = unsafe { vk_device.get_buffer_memory_requirements(buffer) };
        let (memory_type_index, coherent) =
            pick_host_visible_memory_type(device, memory_requirements.memory_type_bits)
                .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);
        let memory = unsafe { vk_device.allocate_memory(&allocate_info, None) }?;
        unsafe { vk_device.bind_buffer_memory(buffer, memory, 0) }?;

        unsafe {
            let ptr = vk_device.map_memory(memory, 0, size as u64, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, size);
            if !coherent {
                let range = [vk::MappedMemoryRange::default()
                    .memory(memory)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)];
                vk_device.flush_mapped_memory_ranges(&range)?;
            }
            vk_device.unmap_memory(memory);
        }

        Ok(TestBuffer {
            device: device.shared_device(),
            buffer,
            memory,
            size,
            coherent,
        })
    }

    /// Composites `src_bytes` over `dst_bytes` at `draw_alpha` coverage through the
    /// real pipeline, and returns the stored bytes.
    ///
    /// `source_encoding` selects the production wiring: `PASSTHROUGH` reproduces the
    /// old gamma-space blend (UNORM everywhere), anything else uses the linear-light
    /// configuration — an encoded UNORM sampled view feeding a shader conversion, and
    /// an `_SRGB` colour attachment view. Both images always start from byte-identical
    /// storage, so the encoding wiring is the only variable.
    fn srgb_blend_probe(
        device: &mut DeviceState,
        descriptors: &mut DescriptorState,
        pipelines: &mut PipelineState,
        keepalive: &mut Vec<TestImage>,
        source_encoding: u32,
        dst_bytes: [u8; 4],
        src_bytes: [u8; 4],
        draw_alpha: f32,
    ) -> [u8; 4] {
        const W: u32 = 8;
        const H: u32 = 8;

        let linear = source_encoding != SOURCE_ENCODING_PASSTHROUGH;
        let attachment_format = if linear {
            vk::Format::R8G8B8A8_SRGB
        } else {
            vk::Format::R8G8B8A8_UNORM
        };
        // Sampled views always declare the encoded storage format; the shader owns the
        // conversion because it has to unpremultiply first.
        let sample_format = vk::Format::R8G8B8A8_UNORM;

        let handles = pipelines
            .pipelines_for_format(attachment_format)
            .expect("pipelines");
        let target = create_mutable_test_image(
            device,
            attachment_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            vk::Extent3D {
                width: W,
                height: H,
                depth: 1,
            },
        )
        .expect("target image");
        let texture = create_mutable_test_image(
            device,
            sample_format,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::Extent3D {
                width: W,
                height: H,
                depth: 1,
            },
        )
        .expect("texture image");

        let dst_pixels: Vec<u8> = dst_bytes
            .iter()
            .copied()
            .cycle()
            .take((W * H * 4) as usize)
            .collect();
        let src_pixels: Vec<u8> = src_bytes
            .iter()
            .copied()
            .cycle()
            .take((W * H * 4) as usize)
            .collect();
        let dst_upload = create_upload_buffer(device, &dst_pixels).expect("dst upload");
        let src_upload = create_upload_buffer(device, &src_pixels).expect("src upload");
        let readback = create_readback_buffer(device, (W * H * 4) as usize).expect("readback");

        let framebuffer = create_framebuffer(
            device,
            handles.render_pass,
            target.view,
            vk::Extent2D { width: W, height: H },
        )
        .expect("framebuffer");
        let descriptor_set = descriptors
            .texture_descriptor_set(texture.view, TextureSampler::LINEAR)
            .expect("descriptor set");
        let command_buffer = device.acquire_command_buffer().expect("command buffer");
        let vk_device = device.device_handle();

        let full_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let copy_region = [vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: W,
                height: H,
                depth: 1,
            })];

        unsafe {
            vk_device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin");

            for (image, upload) in [(target.image, &dst_upload), (texture.image, &src_upload)] {
                transition_image_layout(
                    vk_device,
                    command_buffer,
                    image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                );
                vk_device.cmd_copy_buffer_to_image(
                    command_buffer,
                    upload.buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &copy_region,
                );
            }
            let _ = full_range;

            transition_image_layout(
                vk_device,
                command_buffer,
                target.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            );
            transition_image_layout(
                vk_device,
                command_buffer,
                texture.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
            );

            vk_device.cmd_begin_render_pass(
                command_buffer,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(handles.render_pass)
                    .framebuffer(framebuffer.framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D { width: W, height: H },
                    }),
                vk::SubpassContents::INLINE,
            );
            vk_device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                handles.textured_pipeline,
            );
            vk_device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                handles.textured_layout,
                0,
                &[descriptor_set],
                &[],
            );
            vk_device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: W as f32,
                    height: H as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            vk_device.cmd_set_scissor(
                command_buffer,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: W, height: H },
                }],
            );
            let mut constants = TexturePushConstants::new(draw_alpha, TextureTransform::Normal, false);
            constants.source_encoding = source_encoding;
            vk_device.cmd_push_constants(
                command_buffer,
                handles.textured_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                push_constants_bytes(&constants),
            );
            vk_device.cmd_draw(command_buffer, 4, 1, 0, 0);
            vk_device.cmd_end_render_pass(command_buffer);

            transition_image_layout(
                vk_device,
                command_buffer,
                target.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            );
            vk_device.cmd_copy_image_to_buffer(
                command_buffer,
                target.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback.buffer,
                &copy_region,
            );
            vk_device.end_command_buffer(command_buffer).expect("end");
        }

        device.submit(command_buffer).expect("submit");
        device.wait_for_all_submissions().expect("wait");
        let pixels = readback.read().expect("readback");
        // The descriptor cache is keyed by raw vk::ImageView handle, so dropping
        // these images here would let Vulkan recycle a handle into a stale entry.
        keepalive.push(target);
        keepalive.push(texture);
        [pixels[0], pixels[1], pixels[2], pixels[3]]
    }

    struct LinearBlendFixture {
        device: DeviceState,
        descriptors: DescriptorState,
        pipelines: PipelineState,
        keepalive: Vec<TestImage>,
    }

    impl LinearBlendFixture {
        /// Returns `None` when no usable Vulkan device is present, so these tests are
        /// skipped on hosted runners rather than failing there.
        fn new() -> Option<Self> {
            let instance = Instance::new(Version::VERSION_1_3, None).ok()?;
            let physical_device = PhysicalDevice::enumerate(&instance).ok()?.next()?;
            if !format_supports_test_usage(&physical_device, vk::Format::R8G8B8A8_UNORM)
                || !format_supports_test_usage(&physical_device, vk::Format::R8G8B8A8_SRGB)
            {
                return None;
            }
            let device = DeviceState::new(&physical_device).ok()?;
            let descriptors = DescriptorState::new(device.shared_device()).ok()?;
            let pipelines = PipelineState::new(device.shared_device(), descriptors.texture_layout()).ok()?;
            Some(Self {
                device,
                descriptors,
                pipelines,
                keepalive: Vec::new(),
            })
        }

        fn composite(
            &mut self,
            source_encoding: u32,
            dst: [u8; 4],
            src: [u8; 4],
            draw_alpha: f32,
        ) -> [u8; 4] {
            srgb_blend_probe(
                &mut self.device,
                &mut self.descriptors,
                &mut self.pipelines,
                &mut self.keepalive,
                source_encoding,
                dst,
                src,
                draw_alpha,
            )
        }
    }

    /// Invariant: switching the blend to linear light must not disturb a single byte of
    /// opaque content composited 1:1. If it did, a directly scanned-out buffer and a
    /// composited one would no longer match, and every screenshot would drift.
    #[test]
    fn opaque_one_to_one_composite_is_byte_identical_under_linear_blending() {
        let Some(mut fixture) = LinearBlendFixture::new() else {
            return;
        };

        for value in [0u8, 1, 26, 64, 128, 191, 204, 254, 255] {
            let src = [value, value, value, 255];
            let out = fixture.composite(SOURCE_ENCODING_ELECTRICAL_PREMULTIPLIED, [0, 0, 0, 255], src, 1.0);
            assert_eq!(
                out, src,
                "opaque 1:1 composite of {value} must round-trip exactly through the sRGB attachment"
            );
        }
    }

    /// Offscreen content takes the other conversion (a plain decode, no unpremultiply)
    /// and must round-trip just as exactly for opaque texels.
    #[test]
    fn linear_premultiplied_sources_also_round_trip_opaque_content() {
        let Some(mut fixture) = LinearBlendFixture::new() else {
            return;
        };

        for value in [0u8, 26, 128, 204, 255] {
            let src = [value, value, value, 255];
            let out = fixture.composite(SOURCE_ENCODING_LINEAR_PREMULTIPLIED, [0, 0, 0, 255], src, 1.0);
            assert_eq!(out, src, "opaque offscreen texel {value} must round-trip exactly");
        }
    }

    /// The defect this change repairs: an anti-aliased edge at 50% coverage between a
    /// dark window (26) and a light wallpaper (204) must land near the PERCEPTUAL
    /// midpoint, not the gamma-space average that reads as a harsh dark rim.
    #[test]
    fn coverage_ramp_lands_on_the_perceptual_midpoint() {
        let Some(mut fixture) = LinearBlendFixture::new() else {
            return;
        };

        let gamma = fixture.composite(
            SOURCE_ENCODING_PASSTHROUGH,
            [204, 204, 204, 255],
            [26, 26, 26, 255],
            0.5,
        );
        let linear = fixture.composite(
            SOURCE_ENCODING_ELECTRICAL_PREMULTIPLIED,
            [204, 204, 204, 255],
            [26, 26, 26, 255],
            0.5,
        );

        assert!(
            (110..=120).contains(&gamma[0]),
            "gamma-space blend should reproduce the old dark ramp, got {gamma:?}"
        );
        assert!(
            (148..=154).contains(&linear[0]),
            "linear blend should land near the perceptual midpoint (~151), got {linear:?}"
        );
    }

    /// Translucent client content is premultiplied in ELECTRICAL values. Decoding that
    /// product directly (what a hardware `_SRGB` sampled view would do) collapses white
    /// glass at 50% from ~230 to ~190. The shader's unpremultiply/decode/re-premultiply
    /// keeps it where the shell was tuned, while still blending the destination linearly.
    #[test]
    fn electrical_premultiplied_translucency_is_not_darkened_by_linearization() {
        let Some(mut fixture) = LinearBlendFixture::new() else {
            return;
        };

        // 50%-alpha white, premultiplied in gamma space: 255 * 0.502 = 128.
        let glass = [128u8, 128, 128, 128];
        let backdrop = [204u8, 204, 204, 255];

        let gamma = fixture.composite(SOURCE_ENCODING_PASSTHROUGH, backdrop, glass, 1.0);
        let linear = fixture.composite(SOURCE_ENCODING_ELECTRICAL_PREMULTIPLIED, backdrop, glass, 1.0);
        // Applying the offscreen conversion (a plain decode) to electrical-premultiplied
        // content is exactly the mistake a hardware `_SRGB` sampled view would make.
        let decoded_without_unpremultiply =
            fixture.composite(SOURCE_ENCODING_LINEAR_PREMULTIPLIED, backdrop, glass, 1.0);

        assert!(
            (228..=232).contains(&gamma[0]),
            "gamma-space reference should be ~230, got {gamma:?}"
        );
        assert!(
            (230..=234).contains(&linear[0]),
            "linear-light compositing must keep premultiplied glass where the shell was \
             tuned (~232), got {linear:?}"
        );
        assert!(
            decoded_without_unpremultiply[0] < 200,
            "sanity check: decoding a premultiplied texel without unpremultiplying should \
             visibly darken it (~190); got {decoded_without_unpremultiply:?}, which means \
             this test can no longer tell the two conversions apart"
        );
    }

    /// `MUTABLE_FORMAT` plus a `[UNORM, SRGB]` view-format list has to be accepted for
    /// external DMA-BUF imports under every DRM modifier the driver advertises, or the
    /// attachment view could not be created for real client and shell buffers.
    #[test]
    fn drm_modifier_imports_accept_the_srgb_view_format_list() {
        let Ok(instance) = Instance::new(Version::VERSION_1_3, None) else {
            return;
        };
        let Ok(mut devices) = PhysicalDevice::enumerate(&instance) else {
            return;
        };
        let Some(physical_device) = devices.next() else {
            return;
        };

        for (storage, srgb) in [
            (vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB),
            (vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_SRGB),
        ] {
            let Ok(modifier_properties) = physical_device.get_format_modifier_properties(storage) else {
                continue;
            };

            for properties in modifier_properties {
                for usage in [
                    vk::ImageUsageFlags::SAMPLED,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT,
                ] {
                    let plain = probe_drm_modifier_import(
                        &physical_device,
                        storage,
                        srgb,
                        properties.drm_format_modifier,
                        usage,
                        false,
                    );
                    if !plain {
                        // The driver does not support this modifier at all today, so
                        // MUTABLE_FORMAT cannot regress it.
                        continue;
                    }
                    assert!(
                        probe_drm_modifier_import(
                            &physical_device,
                            storage,
                            srgb,
                            properties.drm_format_modifier,
                            usage,
                            true,
                        ),
                        "modifier {:#x} for {storage:?} with usage {usage:?} is supported today \
                         but rejects MUTABLE_FORMAT + [UNORM, SRGB]; linear-light attachment \
                         views would fail for that buffer",
                        properties.drm_format_modifier
                    );
                }
            }
        }
    }

    fn probe_drm_modifier_import(
        physical_device: &PhysicalDevice,
        storage: vk::Format,
        srgb: vk::Format,
        drm_format_modifier: u64,
        usage: vk::ImageUsageFlags,
        mutable: bool,
    ) -> bool {
        let view_formats = [storage, srgb];
        let mut format_list = vk::ImageFormatListCreateInfo::default().view_formats(&view_formats);
        let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut drm_info = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
            .drm_format_modifier(drm_format_modifier)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let mut info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(storage)
            .ty(vk::ImageType::TYPE_2D)
            .usage(usage)
            .flags(if mutable {
                vk::ImageCreateFlags::MUTABLE_FORMAT
            } else {
                vk::ImageCreateFlags::empty()
            })
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .push_next(&mut external)
            .push_next(&mut drm_info);
        if mutable {
            info = info.push_next(&mut format_list);
        }

        let mut external_properties = vk::ExternalImageFormatProperties::default();
        let mut properties = vk::ImageFormatProperties2::default().push_next(&mut external_properties);

        // SAFETY: the physical device belongs to `instance` and every pointer in the
        // chain refers to storage live for the duration of the call.
        let supported = unsafe {
            physical_device
                .instance()
                .handle()
                .get_physical_device_image_format_properties2(
                    physical_device.handle(),
                    &info,
                    &mut properties,
                )
        }
        .is_ok();

        supported
            && external_properties
                .external_memory_properties
                .external_memory_features
                .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
    }

    #[test]
    fn texture_push_constants_encode_render_effect() {
        let constants = TexturePushConstants::new(1.0, TextureTransform::Normal, false).with_effect(
            TextureRenderEffect {
                kind: TextureRenderEffectKind::Genie,
                progress: 1.5,
                anchor: [0.25, 0.875],
                params: [0.11, 0.0, 0.0, 0.0],
            },
        );

        assert_eq!(
            constants.effect,
            [TextureRenderEffectKind::Genie as u32 as f32, 1.0, 0.25, 0.875]
        );
        assert_eq!(constants.effect_params, [0.11, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn texture_push_constants_encode_bottom_edge_clip() {
        let constants = TexturePushConstants::new(1.0, TextureTransform::Normal, false)
            .with_bottom_edge_clip(
                TextureTransform::Rotate90,
                [12.0, 24.0, 640.0, 78.0],
                [420.0, 0.875, 62.0, 8.0],
                1.25,
            );

        assert_eq!(
            constants.rounded_clip_flags,
            BOTTOM_EDGE_CLIP_FLAG | ((TextureTransform::Rotate90 as u32) << CLIP_TRANSFORM_SHIFT)
        );
        assert_eq!(constants.clip_rect, [12.0, 24.0, 640.0, 78.0]);
        assert_eq!(constants.clip_params, [420.0, 0.875, 62.0, 8.0]);
        assert_eq!(constants.clip_scale, 1.25);
    }

    #[test]
    fn texture_push_constant_layout_matches_the_shader_block() {
        assert_eq!(std::mem::size_of::<TexturePushConstants>(), 104);
        assert_eq!(std::mem::offset_of!(TexturePushConstants, alpha), 0);
        assert_eq!(std::mem::offset_of!(TexturePushConstants, clip_rect), 32);
        assert_eq!(std::mem::offset_of!(TexturePushConstants, source_encoding), 96);
        assert_eq!(std::mem::offset_of!(TexturePushConstants, clip_scale), 100);
    }

    #[derive(Debug)]
    struct TestImage {
        device: std::sync::Arc<DeviceHandle>,
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: vk::ImageView,
    }

    impl Drop for TestImage {
        fn drop(&mut self) {
            self.device.destroy_with(|device| unsafe {
                device.destroy_image_view(self.view, None);
                device.destroy_image(self.image, None);
                device.free_memory(self.memory, None);
            });
        }
    }

    #[derive(Debug)]
    struct TestBuffer {
        device: std::sync::Arc<DeviceHandle>,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        size: usize,
        coherent: bool,
    }

    impl TestBuffer {
        fn read(&self) -> Result<Vec<u8>, vk::Result> {
            let device = self.device.handle();
            let ptr =
                unsafe { device.map_memory(self.memory, 0, self.size as u64, vk::MemoryMapFlags::empty()) }?;

            if !self.coherent {
                let invalidate_ranges = [vk::MappedMemoryRange::default()
                    .memory(self.memory)
                    .offset(0)
                    .size(self.size as u64)];
                unsafe {
                    device.invalidate_mapped_memory_ranges(&invalidate_ranges)?;
                }
            }

            let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), self.size).to_vec() };
            unsafe { device.unmap_memory(self.memory) };
            Ok(bytes)
        }
    }

    impl Drop for TestBuffer {
        fn drop(&mut self) {
            self.device.destroy_with(|device| unsafe {
                device.destroy_buffer(self.buffer, None);
                device.free_memory(self.memory, None);
            });
        }
    }

    #[derive(Debug)]
    struct TestFramebuffer {
        device: std::sync::Arc<DeviceHandle>,
        framebuffer: vk::Framebuffer,
    }

    impl Drop for TestFramebuffer {
        fn drop(&mut self) {
            self.device
                .destroy_with(|device| unsafe { device.destroy_framebuffer(self.framebuffer, None) });
        }
    }

    #[test]
    fn offscreen_clear_solid_and_textured_draw_produce_expected_pixels() {
        let instance = match Instance::new(Version::VERSION_1_3, None) {
            Ok(instance) => instance,
            Err(_) => return,
        };

        let physical_device = match PhysicalDevice::enumerate(&instance) {
            Ok(mut iter) => match iter.next() {
                Some(phd) => phd,
                None => return,
            },
            Err(_) => return,
        };

        if !format_supports_test_usage(&physical_device, TEST_FORMAT) {
            return;
        }

        let mut device = match DeviceState::new(&physical_device) {
            Ok(device) => device,
            Err(_) => return,
        };

        let mut descriptors = match DescriptorState::new(device.shared_device()) {
            Ok(state) => state,
            Err(_) => return,
        };

        let mut pipelines = match PipelineState::new(device.shared_device(), descriptors.texture_layout()) {
            Ok(state) => state,
            Err(_) => return,
        };

        let handles = match pipelines.pipelines_for_format(TEST_FORMAT) {
            Ok(handles) => handles,
            Err(_) => return,
        };

        let target = match create_test_image(
            &device,
            TEST_FORMAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::Extent3D {
                width: TEST_WIDTH,
                height: TEST_HEIGHT,
                depth: 1,
            },
        ) {
            Ok(image) => image,
            Err(_) => return,
        };

        let texture = match create_test_image(
            &device,
            TEST_FORMAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::Extent3D {
                width: 4,
                height: 4,
                depth: 1,
            },
        ) {
            Ok(image) => image,
            Err(_) => return,
        };

        let framebuffer = match create_framebuffer(
            &device,
            handles.render_pass,
            target.view,
            vk::Extent2D {
                width: TEST_WIDTH,
                height: TEST_HEIGHT,
            },
        ) {
            Ok(fb) => fb,
            Err(_) => return,
        };

        let readback = match create_readback_buffer(&device, (TEST_WIDTH * TEST_HEIGHT * 4) as usize) {
            Ok(buffer) => buffer,
            Err(_) => return,
        };

        let descriptor_set = match descriptors.texture_descriptor_set(texture.view, TextureSampler::LINEAR) {
            Ok(set) => set,
            Err(_) => return,
        };

        let command_buffer = match device.acquire_command_buffer() {
            Ok(buffer) => buffer,
            Err(_) => return,
        };

        let vk_device = device.device_handle();

        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 1.0, 1.0],
            },
        }];

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(handles.render_pass)
            .framebuffer(framebuffer.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: TEST_WIDTH,
                    height: TEST_HEIGHT,
                },
            })
            .clear_values(&clear_values);

        let full_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        unsafe {
            if vk_device
                .begin_command_buffer(command_buffer, &begin_info)
                .is_err()
            {
                return;
            }

            transition_image_layout(
                vk_device,
                command_buffer,
                target.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            );

            vk_device.cmd_clear_color_image(
                command_buffer,
                target.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [0.0, 0.0, 1.0, 1.0],
                },
                &[full_range],
            );

            transition_image_layout(
                vk_device,
                command_buffer,
                target.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            );

            transition_image_layout(
                vk_device,
                command_buffer,
                texture.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            );

            vk_device.cmd_clear_color_image(
                command_buffer,
                texture.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [0.5, 0.0, 0.0, 0.5],
                },
                &[full_range],
            );

            transition_image_layout(
                vk_device,
                command_buffer,
                texture.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
            );

            vk_device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );

            vk_device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                handles.solid_pipeline,
            );

            let left_half_viewport = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: (TEST_WIDTH / 2) as f32,
                height: TEST_HEIGHT as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let left_half_scissor = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: TEST_WIDTH / 2,
                    height: TEST_HEIGHT,
                },
            }];

            vk_device.cmd_set_viewport(command_buffer, 0, &left_half_viewport);
            vk_device.cmd_set_scissor(command_buffer, 0, &left_half_scissor);

            let solid_constants = SolidPushConstants {
                color: [0.0, 1.0, 0.0, 1.0],
            };
            vk_device.cmd_push_constants(
                command_buffer,
                handles.solid_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                push_constants_bytes(&solid_constants),
            );
            vk_device.cmd_draw(command_buffer, 4, 1, 0, 0);

            vk_device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                handles.textured_pipeline,
            );
            vk_device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                handles.textured_layout,
                0,
                &[descriptor_set],
                &[],
            );

            let right_half_partial_viewport = [vk::Viewport {
                x: (TEST_WIDTH / 2) as f32,
                y: 0.0,
                width: (TEST_WIDTH / 2) as f32,
                height: (TEST_HEIGHT / 2) as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let right_half_partial_scissor = [vk::Rect2D {
                offset: vk::Offset2D {
                    x: (TEST_WIDTH / 2) as i32,
                    y: 0,
                },
                extent: vk::Extent2D {
                    width: TEST_WIDTH / 2,
                    height: TEST_HEIGHT / 2,
                },
            }];
            vk_device.cmd_set_viewport(command_buffer, 0, &right_half_partial_viewport);
            vk_device.cmd_set_scissor(command_buffer, 0, &right_half_partial_scissor);

            let texture_constants = TexturePushConstants::new(1.0, TextureTransform::Normal, false);
            vk_device.cmd_push_constants(
                command_buffer,
                handles.textured_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                push_constants_bytes(&texture_constants),
            );
            vk_device.cmd_draw(command_buffer, 4, 1, 0, 0);

            vk_device.cmd_end_render_pass(command_buffer);

            transition_image_layout(
                vk_device,
                command_buffer,
                target.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            );

            let copy_region = [vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: TEST_WIDTH,
                    height: TEST_HEIGHT,
                    depth: 1,
                })];

            vk_device.cmd_copy_image_to_buffer(
                command_buffer,
                target.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback.buffer,
                &copy_region,
            );

            if vk_device.end_command_buffer(command_buffer).is_err() {
                return;
            }
        }

        if device.submit(command_buffer).is_err() {
            return;
        }
        if device.wait_for_all_submissions().is_err() {
            return;
        }

        let pixels = match readback.read() {
            Ok(pixels) => pixels,
            Err(_) => return,
        };

        let left_sample = rgba_at(
            &pixels,
            TEST_WIDTH as usize,
            TEST_WIDTH as usize / 4,
            TEST_HEIGHT as usize / 2,
        );
        let right_upper = rgba_at(
            &pixels,
            TEST_WIDTH as usize,
            (TEST_WIDTH as usize * 3) / 4,
            TEST_HEIGHT as usize / 4,
        );
        let right_lower = rgba_at(
            &pixels,
            TEST_WIDTH as usize,
            (TEST_WIDTH as usize * 3) / 4,
            (TEST_HEIGHT as usize * 3) / 4,
        );

        assert_color_near(left_sample, [0, 255, 0, 255], 24);

        let clear_blue = [0, 0, 255, 255];
        let textured_blend = [128, 0, 128, 255];

        let upper_is_blend = color_near(right_upper, textured_blend, 28);
        let upper_is_clear = color_near(right_upper, clear_blue, 24);
        let lower_is_blend = color_near(right_lower, textured_blend, 28);
        let lower_is_clear = color_near(right_lower, clear_blue, 24);

        assert!(
            (upper_is_blend && lower_is_clear) || (upper_is_clear && lower_is_blend),
            "expected one right-half sample to match textured blend and the other to remain clear; upper={right_upper:?}, lower={right_lower:?}"
        );

        let cache_blob = match pipelines.pipeline_cache_data() {
            Ok(data) => data,
            Err(_) => return,
        };
        let _ = pipelines.merge_pipeline_cache_data(&cache_blob);
    }

    fn format_supports_test_usage(physical_device: &PhysicalDevice, format: vk::Format) -> bool {
        let properties = unsafe {
            physical_device
                .instance()
                .handle()
                .get_physical_device_format_properties(physical_device.handle(), format)
        };

        let required_features = vk::FormatFeatureFlags::COLOR_ATTACHMENT
            | vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::TRANSFER_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST;

        properties.optimal_tiling_features.contains(required_features)
    }

    fn create_test_image(
        device: &DeviceState,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        extent: vk::Extent3D,
    ) -> Result<TestImage, vk::Result> {
        let vk_device = device.device_handle();
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { vk_device.create_image(&image_create_info, None) }?;
        let memory_requirements = unsafe { vk_device.get_image_memory_requirements(image) };

        let memory_type_index = pick_image_memory_type(device, memory_requirements.memory_type_bits)
            .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;

        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);
        let memory = match unsafe { vk_device.allocate_memory(&allocate_info, None) } {
            Ok(memory) => memory,
            Err(err) => {
                unsafe { vk_device.destroy_image(image, None) };
                return Err(err);
            }
        };

        if let Err(err) = unsafe { vk_device.bind_image_memory(image, memory, 0) } {
            unsafe {
                vk_device.free_memory(memory, None);
                vk_device.destroy_image(image, None);
            }
            return Err(err);
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let view = match unsafe { vk_device.create_image_view(&view_info, None) } {
            Ok(view) => view,
            Err(err) => {
                unsafe {
                    vk_device.free_memory(memory, None);
                    vk_device.destroy_image(image, None);
                }
                return Err(err);
            }
        };

        Ok(TestImage {
            device: device.shared_device(),
            image,
            memory,
            view,
        })
    }

    fn create_framebuffer(
        device: &DeviceState,
        render_pass: vk::RenderPass,
        view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Result<TestFramebuffer, vk::Result> {
        let attachments = [view];
        let create_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(1);

        let framebuffer = unsafe { device.device_handle().create_framebuffer(&create_info, None) }?;

        Ok(TestFramebuffer {
            device: device.shared_device(),
            framebuffer,
        })
    }

    fn create_readback_buffer(device: &DeviceState, size: usize) -> Result<TestBuffer, vk::Result> {
        let vk_device = device.device_handle();

        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { vk_device.create_buffer(&buffer_create_info, None) }?;
        let memory_requirements = unsafe { vk_device.get_buffer_memory_requirements(buffer) };

        let (memory_type_index, coherent) =
            pick_host_visible_memory_type(device, memory_requirements.memory_type_bits)
                .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;

        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);
        let memory = match unsafe { vk_device.allocate_memory(&allocate_info, None) } {
            Ok(memory) => memory,
            Err(err) => {
                unsafe { vk_device.destroy_buffer(buffer, None) };
                return Err(err);
            }
        };

        if let Err(err) = unsafe { vk_device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                vk_device.free_memory(memory, None);
                vk_device.destroy_buffer(buffer, None);
            }
            return Err(err);
        }

        Ok(TestBuffer {
            device: device.shared_device(),
            buffer,
            memory,
            size,
            coherent,
        })
    }

    fn pick_image_memory_type(device: &DeviceState, memory_type_bits: u32) -> Option<u32> {
        let memory_properties = unsafe {
            device
                .physical_device()
                .instance()
                .handle()
                .get_physical_device_memory_properties(device.physical_device().handle())
        };

        pick_memory_type_index(
            &memory_properties,
            memory_type_bits,
            vk::MemoryPropertyFlags::empty(),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .map(|(index, _)| index)
    }

    fn pick_host_visible_memory_type(device: &DeviceState, memory_type_bits: u32) -> Option<(u32, bool)> {
        let memory_properties = unsafe {
            device
                .physical_device()
                .instance()
                .handle()
                .get_physical_device_memory_properties(device.physical_device().handle())
        };

        pick_memory_type_index(
            &memory_properties,
            memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .map(|(index, flags)| (index, flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)))
    }

    fn pick_memory_type_index(
        properties: &vk::PhysicalDeviceMemoryProperties,
        memory_type_bits: u32,
        required: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
    ) -> Option<(u32, vk::MemoryPropertyFlags)> {
        let mut fallback = None;

        for index in 0..properties.memory_type_count {
            let mask = 1u32 << index;
            if memory_type_bits & mask == 0 {
                continue;
            }

            let flags = properties.memory_types[index as usize].property_flags;
            if !flags.contains(required) {
                continue;
            }

            if flags.contains(preferred) {
                return Some((index, flags));
            }

            if fallback.is_none() {
                fallback = Some((index, flags));
            }
        }

        fallback
    }

    unsafe fn transition_image_layout(
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
    ) {
        let barrier = [vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)];

        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }

    fn rgba_at(pixels: &[u8], width: usize, x: usize, y: usize) -> [u8; 4] {
        let idx = (y * width + x) * 4;
        [pixels[idx], pixels[idx + 1], pixels[idx + 2], pixels[idx + 3]]
    }

    fn assert_color_near(actual: [u8; 4], expected: [u8; 4], tolerance: u8) {
        assert!(
            color_near(actual, expected, tolerance),
            "color mismatch: actual={actual:?}, expected={expected:?}, tolerance={tolerance}"
        );
    }

    fn color_near(actual: [u8; 4], expected: [u8; 4], tolerance: u8) -> bool {
        actual
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| actual.abs_diff(*expected) <= tolerance)
    }
}
