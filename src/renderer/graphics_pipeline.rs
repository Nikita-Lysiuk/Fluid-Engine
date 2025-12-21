use ash::{vk, Device};
use ash::vk::{AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, ColorComponentFlags, CullModeFlags, DynamicState, Extent2D, FrontFace, GraphicsPipelineCreateInfo, ImageLayout, Offset2D, Pipeline, PipelineBindPoint, PipelineCache, PipelineColorBlendAttachmentState, PipelineDynamicStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PipelineLayout, PipelineMultisampleStateCreateInfo, PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo, PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode, PrimitiveTopology, Rect2D, RenderPass, RenderPassCreateInfo, SampleCountFlags, ShaderModule, ShaderModuleCreateInfo, ShaderStageFlags, StructureType, SubpassDescription, Viewport};
use log::{info};
use crate::errors::application_error::ApplicationError;
use crate::errors::graphics_pipeline_error::GraphicsPipelineError;
use crate::utils::loader::Loader;

pub struct GraphicsPipeline {
    pub render_pass: RenderPass,
    pub pipeline_layout: PipelineLayout,
    pub graphics_pipeline: Pipeline,
}

impl GraphicsPipeline {
    pub fn new (device: &Device, swapchain_image_format: vk::Format) -> Result<GraphicsPipeline, ApplicationError> {
        unsafe {
            let vert_shader_module = Self::create_shader_module(
                device,
                Loader::load_shader_code("shaders/compiled/simple_shader.vert.spv")?)?;
            let fragment_shader_module = Self::create_shader_module(
                device,
                Loader::load_shader_code("shaders/compiled/simple_shader.frag.spv")?)?;
            let shader_stages = Self::create_shader_stages(
                vert_shader_module,
                fragment_shader_module);
            info!("[Graphics Pipeline] Shader modules created.");

            
            let pipeline_layout = Self::create_pipeline_layout(device)?;
            info!("[Graphics Pipeline] Pipeline layout created.");
            let render_pass = Self::create_render_pass(device, swapchain_image_format)?;
            info!("[Graphics Pipeline] Render pass created.");
            let graphics_pipeline = Self::create_pipeline(
                device,
                shader_stages,
                pipeline_layout,
                render_pass)?;
            info!("[Graphics Pipeline] Graphics pipeline successfully created.");

            Self::destroy_shader_modules(
                device, 
                vert_shader_module, 
                fragment_shader_module);

            Ok(GraphicsPipeline {
                render_pass,
                pipeline_layout,
                graphics_pipeline,
            })
        }
    }
    unsafe fn create_shader_module(device: &Device, code: Vec<u32>) -> Result<ShaderModule, GraphicsPipelineError> {
        unsafe {
            let create_info = ShaderModuleCreateInfo {
                s_type: StructureType::SHADER_MODULE_CREATE_INFO,
                code_size: code.len() * 4,
                p_code: code.as_ptr(),
                ..ShaderModuleCreateInfo::default()
            };
            
            device.create_shader_module(&create_info, None)
                .map_err(GraphicsPipelineError::ShaderModuleCreationError)
        }
    }
    unsafe fn create_shader_stages(
        vert_shader_module: ShaderModule,
        fragment_shader_module: ShaderModule
    ) -> [PipelineShaderStageCreateInfo<'static>; 2] {
        let vert_shader_stage_info = PipelineShaderStageCreateInfo {
            s_type: StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: ShaderStageFlags::VERTEX,
            module: vert_shader_module,
            p_name: b"main\0".as_ptr() as *const i8,
            ..PipelineShaderStageCreateInfo::default()
        };

        let frag_shader_stage_info = PipelineShaderStageCreateInfo {
            s_type: StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: ShaderStageFlags::FRAGMENT,
            module: fragment_shader_module,
            p_name: b"main\0".as_ptr() as *const i8,
            ..PipelineShaderStageCreateInfo::default()
        };

        [vert_shader_stage_info, frag_shader_stage_info]
    }
    unsafe fn create_pipeline_layout(device: &Device) -> Result<PipelineLayout, GraphicsPipelineError> {
        unsafe {
            let layout_info = vk::PipelineLayoutCreateInfo {
                s_type: StructureType::PIPELINE_LAYOUT_CREATE_INFO,
                ..vk::PipelineLayoutCreateInfo::default()
            };

            device.create_pipeline_layout(&layout_info, None)
                .map_err(GraphicsPipelineError::PipelineLayoutCreationError)
        }
    }
    unsafe fn create_render_pass(device: &Device, swapchain_image_format: vk::Format) -> Result<RenderPass, GraphicsPipelineError> {
        unsafe {
            let color_attachment = AttachmentDescription {
                format: swapchain_image_format,
                samples: SampleCountFlags::TYPE_1,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                stencil_load_op: AttachmentLoadOp::DONT_CARE,
                stencil_store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::PRESENT_SRC_KHR,
                ..AttachmentDescription::default()
            };

            let color_attachment_ref = AttachmentReference {
                attachment: 0,
                layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                ..AttachmentReference::default()
            };

            let subpass = SubpassDescription {
                pipeline_bind_point: PipelineBindPoint::GRAPHICS,
                color_attachment_count: 1,
                p_color_attachments: &color_attachment_ref,
                ..SubpassDescription::default()
            };

            let render_pass_info = RenderPassCreateInfo {
                s_type: StructureType::RENDER_PASS_CREATE_INFO,
                attachment_count: 1,
                p_attachments: &color_attachment,
                subpass_count: 1,
                p_subpasses: &subpass,
                ..RenderPassCreateInfo::default()
            };

            device.create_render_pass(&render_pass_info, None)
                .map_err(GraphicsPipelineError::RenderPassCreationError)
        }
    }
    unsafe fn create_pipeline(
        device: &Device,
        shader_stages: [PipelineShaderStageCreateInfo; 2],
        pipeline_layout: PipelineLayout,
        render_pass: RenderPass
    ) -> Result<Pipeline, GraphicsPipelineError> {

        let vertex_input_info = Self::create_vertex_input_info();
        let assembly_state_info = Self::create_assembly_state_info();
        let viewport_state_info = Self::create_viewport_state_info();
        let rasterization_state_info = Self::create_rasterization_state_info();
        let multisample_state_info = Self::create_multisample_state_info();

        let color_blend_attachment = PipelineColorBlendAttachmentState {
            color_write_mask: ColorComponentFlags::R | ColorComponentFlags::G | ColorComponentFlags::B | ColorComponentFlags::A,
            blend_enable: vk::FALSE,
            ..Default::default()
        };

        let color_blend_info = vk::PipelineColorBlendStateCreateInfo {
            s_type: StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            attachment_count: 1,
            p_attachments: &color_blend_attachment,
            ..Default::default()
        };

        let dynamic_states = [DynamicState::VIEWPORT, DynamicState::SCISSOR];
        let dynamic_state_info = PipelineDynamicStateCreateInfo {
            s_type: StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let pipeline_info = GraphicsPipelineCreateInfo {
            s_type: StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            stage_count: 2,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &assembly_state_info,
            p_viewport_state: &viewport_state_info,
            p_rasterization_state: &rasterization_state_info,
            p_multisample_state: &multisample_state_info,
            p_color_blend_state: &color_blend_info,
            p_dynamic_state: &dynamic_state_info,
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            ..Default::default()
        };

        unsafe {
            device.create_graphics_pipelines(PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| GraphicsPipelineError::PipelineCreationError(e))
                .map(|pipelines| pipelines[0])
        }
    }
    // TODO: Change to specify quad for ray marching
    fn create_vertex_input_info() -> PipelineVertexInputStateCreateInfo<'static> {
        PipelineVertexInputStateCreateInfo {
            s_type: StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertex_binding_description_count: 0,
            p_vertex_binding_descriptions: std::ptr::null(),
            vertex_attribute_description_count: 0,
            p_vertex_attribute_descriptions: std::ptr::null(),
            ..PipelineVertexInputStateCreateInfo::default()
        }
    }
    fn create_assembly_state_info() -> PipelineInputAssemblyStateCreateInfo<'static> {
        PipelineInputAssemblyStateCreateInfo {
            s_type: StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology: PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            ..PipelineInputAssemblyStateCreateInfo::default()
        }
    }
    fn create_viewport_state_info() -> PipelineViewportStateCreateInfo<'static> {
        PipelineViewportStateCreateInfo {
            s_type: StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewport_count: 1,
            scissor_count: 1,
            ..PipelineViewportStateCreateInfo::default()
        }
    }
    fn create_rasterization_state_info() -> PipelineRasterizationStateCreateInfo<'static> {
        PipelineRasterizationStateCreateInfo {
            s_type: StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: CullModeFlags::BACK,
            front_face: FrontFace::CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            ..PipelineRasterizationStateCreateInfo::default()
        }
    }
    fn create_multisample_state_info() -> PipelineMultisampleStateCreateInfo<'static> {
        PipelineMultisampleStateCreateInfo {
            s_type: StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            rasterization_samples: SampleCountFlags::TYPE_1,
            sample_shading_enable: vk::FALSE,
            ..PipelineMultisampleStateCreateInfo::default()
        }
    }
    unsafe fn destroy_shader_modules(
        device: &Device, 
        vert_shader_module: ShaderModule, 
        fragment_shader_module: ShaderModule
    ) {
        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
            info!("[Graphics Pipeline] Shader modules deleted.");
        }
    }
    pub unsafe fn destroy_pipeline_layout(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            info!("[Graphics Pipeline] Pipeline layout deleted.");
        }
    }
    pub unsafe fn destroy_render_pass(&self, device: &Device) {
        unsafe {
            device.destroy_render_pass(self.render_pass, None);
            info!("[Graphics Pipeline] Render pass deleted");
        }
    }
    pub unsafe fn destroy_graphics_pipeline(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.graphics_pipeline, None);
            info!("[Graphics Pipeline] Graphics pipeline deleted.");
        }
    }
}
