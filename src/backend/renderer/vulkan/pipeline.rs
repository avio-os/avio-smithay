use std::{io::Cursor, sync::Arc};

use ash::{util::read_spv, vk};
use indexmap::IndexMap;

use super::{device::DeviceHandle, VulkanRendererError};

const SOLID_VERTEX_SHADER_SPV: &[u8] = include_bytes!("shaders/solid.vert.spv");
const SOLID_FRAGMENT_SHADER_SPV: &[u8] = include_bytes!("shaders/solid.frag.spv");
const TEXTURE_VERTEX_SHADER_SPV: &[u8] = include_bytes!("shaders/texture.vert.spv");
const TEXTURE_FRAGMENT_SHADER_SPV: &[u8] = include_bytes!("shaders/texture.frag.spv");

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
    pub(crate) _pad0: u32,
    pub(crate) src_offset: [f32; 2],
    pub(crate) src_scale: [f32; 2],
}

impl Default for TexturePushConstants {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            transform: TextureTransform::Normal as u32,
            y_inverted: 0,
            _pad0: 0,
            src_offset: [0.0, 0.0],
            src_scale: [1.0, 1.0],
        }
    }
}

impl TexturePushConstants {
    pub(crate) fn new(alpha: f32, transform: TextureTransform, y_inverted: bool) -> Self {
        Self {
            alpha,
            transform: transform as u32,
            y_inverted: u32::from(y_inverted),
            _pad0: 0,
            src_offset: [0.0, 0.0],
            src_scale: [1.0, 1.0],
        }
    }

    pub(crate) fn with_src_rect(mut self, offset: [f32; 2], scale: [f32; 2]) -> Self {
        self.src_offset = offset;
        self.src_scale = scale;
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PipelineHandles {
    pub(crate) render_pass: vk::RenderPass,
    pub(crate) solid_pipeline: vk::Pipeline,
    pub(crate) solid_opaque_pipeline: vk::Pipeline,
    pub(crate) textured_pipeline: vk::Pipeline,
    pub(crate) textured_opaque_pipeline: vk::Pipeline,
    pub(crate) solid_layout: vk::PipelineLayout,
    pub(crate) textured_layout: vk::PipelineLayout,
}

#[derive(Debug)]
struct FormatPipelineSet {
    render_pass: vk::RenderPass,
    solid_pipeline: vk::Pipeline,
    solid_opaque_pipeline: vk::Pipeline,
    textured_pipeline: vk::Pipeline,
    textured_opaque_pipeline: vk::Pipeline,
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

        Ok(Self {
            device,
            pipeline_cache,
            solid_layout,
            textured_layout,
            solid_vertex_module,
            solid_fragment_module,
            texture_vertex_module,
            texture_fragment_module,
            per_format: IndexMap::new(),
        })
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
            solid_layout: self.solid_layout,
            textured_layout: self.textured_layout,
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

        Ok(FormatPipelineSet {
            render_pass,
            solid_pipeline,
            solid_opaque_pipeline,
            textured_pipeline,
            textured_opaque_pipeline,
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
        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(blend_enabled)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
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
        let device = self.device.handle();

        for (_, set) in self.per_format.drain(..) {
            unsafe {
                device.destroy_pipeline(set.textured_opaque_pipeline, None);
                device.destroy_pipeline(set.textured_pipeline, None);
                device.destroy_pipeline(set.solid_opaque_pipeline, None);
                device.destroy_pipeline(set.solid_pipeline, None);
                device.destroy_render_pass(set.render_pass, None);
            }
        }

        unsafe {
            device.destroy_shader_module(self.texture_fragment_module, None);
            device.destroy_shader_module(self.texture_vertex_module, None);
            device.destroy_shader_module(self.solid_fragment_module, None);
            device.destroy_shader_module(self.solid_vertex_module, None);
            device.destroy_pipeline_layout(self.textured_layout, None);
            device.destroy_pipeline_layout(self.solid_layout, None);
            device.destroy_pipeline_cache(self.pipeline_cache, None);
        }
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

    use crate::backend::vulkan::{version::Version, Instance, PhysicalDevice};

    use super::{
        super::descriptor::DescriptorState, super::device::DeviceHandle, super::device::DeviceState,
        push_constants_bytes, PipelineState, SolidPushConstants, TexturePushConstants, TextureTransform,
    };

    const TEST_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
    const TEST_WIDTH: u32 = 64;
    const TEST_HEIGHT: u32 = 64;

    #[derive(Debug)]
    struct TestImage {
        device: std::sync::Arc<DeviceHandle>,
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: vk::ImageView,
    }

    impl Drop for TestImage {
        fn drop(&mut self) {
            unsafe {
                self.device.handle().destroy_image_view(self.view, None);
                self.device.handle().destroy_image(self.image, None);
                self.device.handle().free_memory(self.memory, None);
            }
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
            unsafe {
                self.device.handle().destroy_buffer(self.buffer, None);
                self.device.handle().free_memory(self.memory, None);
            }
        }
    }

    #[derive(Debug)]
    struct TestFramebuffer {
        device: std::sync::Arc<DeviceHandle>,
        framebuffer: vk::Framebuffer,
    }

    impl Drop for TestFramebuffer {
        fn drop(&mut self) {
            unsafe {
                self.device.handle().destroy_framebuffer(self.framebuffer, None);
            }
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

        let descriptor_set = match descriptors.texture_descriptor_set(texture.view) {
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
                    float32: [1.0, 0.0, 0.0, 1.0],
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

            let texture_constants = TexturePushConstants::new(0.5, TextureTransform::Normal, false);
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
