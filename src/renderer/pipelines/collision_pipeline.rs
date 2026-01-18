use std::sync::Arc;
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::pipeline::{DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::input_assembly::PrimitiveTopology::LineList;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{PolygonMode, RasterizationState};
use vulkano::pipeline::graphics::subpass::{PipelineRenderingCreateInfo, PipelineSubpassType};
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::ViewportState;
use crate::utils::shader_loader::load_shader_entry_point;

mod vs {
    use vulkano_shaders::shader;
    
    shader!(
        ty: "vertex",
        path: "shaders/collision.vert" 
    );
}

mod fs {
    use vulkano_shaders::shader;
    
    shader!(
        ty: "fragment",
        path: "shaders/collision.frag"
    );
}

pub struct CollisionPipeline {
    pub inner: Arc<GraphicsPipeline>
}

impl CollisionPipeline {
    pub fn new(
        device: Arc<Device>,
        layout: Arc<PipelineLayout>,
        color_format: Format,
        depth_format: Format
    ) -> Self {
        let vs = load_shader_entry_point(device.clone(), vs::load, "collision vertex");
        let fs = load_shader_entry_point(device.clone(), fs::load, "collision fragment");
        
        let stages = [
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs.clone())
        ];
        
        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                
                vertex_input_state: Some(VertexInputState::new()),
                input_assembly_state: Some(InputAssemblyState {
                    topology: LineList,
                    ..InputAssemblyState::default()
                }),
                dynamic_state: [DynamicState::Viewport, DynamicState::Scissor].into_iter().collect(),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState {
                    polygon_mode: PolygonMode::Line,
                    ..RasterizationState::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState {
                        write_enable: false,
                        compare_op: CompareOp::Always,
                    }),
                    ..DepthStencilState::default()
                }),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    1, ColorBlendAttachmentState::default()
                )),
                subpass: Some(PipelineSubpassType::BeginRendering(PipelineRenderingCreateInfo {
                    color_attachment_formats: vec![Some(color_format)],
                    depth_attachment_format: Some(depth_format),
                    stencil_attachment_format: None,
                    ..PipelineRenderingCreateInfo::default()
                })),
                ..GraphicsPipelineCreateInfo::layout(layout)
            }
        ).map_err(|e| panic!("[Collision Pipeline] Failed to create graphics pipeline:\n{:?}", e)).unwrap();
        
        Self { inner: pipeline }
    }
}