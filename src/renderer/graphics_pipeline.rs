use ash::{vk, Device};
use ash::vk::{ColorComponentFlags, CullModeFlags, DynamicState, Extent2D, FrontFace, Offset2D, PipelineColorBlendAttachmentState, PipelineDynamicStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PipelineLayout, PipelineMultisampleStateCreateInfo, PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo, PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode, PrimitiveTopology, Rect2D, SampleCountFlags, ShaderModule, ShaderModuleCreateInfo, ShaderStageFlags, StructureType, Viewport};
use log::info;
use crate::errors::application_error::ApplicationError;
use crate::errors::graphics_pipeline_error::GraphicsPipelineError;
use crate::utils::loader::Loader;

pub struct GraphicsPipeline {
    pub pipeline_layout: PipelineLayout,
}

impl GraphicsPipeline {
    pub fn new (device: &Device, extent2d: &Extent2D) -> Result<GraphicsPipeline, ApplicationError> {
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
            let dynamic_state_info = Self::create_dynamic_state_info();
            let vertex_input_info = Self::create_vertex_input_info();
            let assembly_state_info = Self::create_assembly_state_info();
            let viewport_state_info = Self::create_viewport_state_info();
            let rasterization_state_info = Self::create_rasterization_state_info();
            let multisample_state_info = Self::create_multisample_state_info();
            let color_blend_state_info = Self::create_color_blend_state_info();
            let pipeline_layout = Self::create_pipeline_layout(device)?;
            
            
            

            info!("[Vulkan] Graphics pipeline successfully created.");

            Self::destroy_shader_modules(
                device, 
                vert_shader_module, 
                fragment_shader_module);

            Ok(GraphicsPipeline {
                pipeline_layout,
            })
        }
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
    pub unsafe fn destroy_pipeline_layout(&self, device: &Device) {
        unsafe {
            info!("[Graphics Pipeline] Deleting pipeline layout...");
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            info!("[Graphics Pipeline] Pipeline layout deleted.");
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
    fn create_dynamic_state_info() -> PipelineDynamicStateCreateInfo<'static> {
        let dynamic_states = vec![
            DynamicState::VIEWPORT,
            DynamicState::SCISSOR,
        ];
        
        PipelineDynamicStateCreateInfo {
            s_type: StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..PipelineDynamicStateCreateInfo::default()
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
    fn create_viewport(extent2d: &Extent2D) -> Viewport {
        Viewport {
            x: 0.0,
            y: 0.0,
            width: extent2d.width as f32,
            height: extent2d.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
    fn create_scissor(extent2d: &Extent2D) -> Rect2D {
        Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: *extent2d,
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
    fn create_color_blend_attachment() -> PipelineColorBlendAttachmentState {
        PipelineColorBlendAttachmentState {
            color_write_mask: ColorComponentFlags::R
                | ColorComponentFlags::G
                | ColorComponentFlags::B
                | ColorComponentFlags::A,
            blend_enable: vk::FALSE,
            ..PipelineColorBlendAttachmentState::default()
        }
    }
    fn create_color_blend_state_info() -> vk::PipelineColorBlendStateCreateInfo<'static> {
        let color_blend_attachment = &Self::create_color_blend_attachment();
        vk::PipelineColorBlendStateCreateInfo {
            s_type: StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logic_op_enable: vk::FALSE,
            attachment_count: 1,
            p_attachments: color_blend_attachment,
            ..vk::PipelineColorBlendStateCreateInfo::default()
        }
    }
    unsafe fn create_shader_stages(
        vert_shader_module: ShaderModule,
        fragment_shader_module: ShaderModule
    ) -> [PipelineShaderStageCreateInfo<'static>; 2] { // Повертаємо масив
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
    unsafe fn destroy_shader_modules(
        device: &Device, 
        vert_shader_module: ShaderModule, 
        fragment_shader_module: ShaderModule
    ) {
        unsafe {
            info!("[Graphics Pipeline] Deleting shader modules...");
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
            info!("[Graphics Pipeline] Shader modules deleted.");
        }
    }
}
