//! Native pipeline construction under the exact cold creation authority.
use super::*;

impl PipelineCreationAuthority {
    pub(super) fn create_format_pipeline_set(
        &self,
        format: vk::Format,
    ) -> Result<FormatPipelineSet, VulkanRendererError> {
        let _cache = self
            .cache_access
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let vk_device = self.device.handle();
        let render_pass = create_render_pass(vk_device, format)?;

        let solid_pipeline = match self.create_graphics_pipeline(
            render_pass,
            self.solid_layout,
            self.solid_vertex_module,
            self.solid_fragment_module,
            true,
            false,
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
            false,
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

        let prefix_mix_pipeline = match self.create_graphics_pipeline(
            render_pass,
            self.textured_layout,
            self.texture_vertex_module,
            self.texture_fragment_module,
            true,
            true,
        ) {
            Ok(pipeline) => pipeline,
            Err(err) => {
                unsafe {
                    vk_device.destroy_pipeline(kawase_pipeline, None);
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
            prefix_mix_pipeline,
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
        prefix_mix: bool,
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
            .src_color_blend_factor(if prefix_mix {
                vk::BlendFactor::CONSTANT_ALPHA
            } else {
                vk::BlendFactor::ONE
            })
            .dst_color_blend_factor(if prefix_mix {
                vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA
            } else {
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA
            })
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(if prefix_mix {
                vk::BlendFactor::CONSTANT_ALPHA
            } else {
                vk::BlendFactor::ONE
            })
            .dst_alpha_blend_factor(if prefix_mix {
                vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA
            } else {
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA
            })
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);
        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::BLEND_CONSTANTS,
        ];
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
