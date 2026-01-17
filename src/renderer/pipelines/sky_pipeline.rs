use std::sync::Arc;
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::pipeline::{DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::subpass::{PipelineRenderingCreateInfo, PipelineSubpassType};
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::ViewportState;
use crate::utils::shader_loader::load_shader_entry_point;

mod vs {
    use vulkano_shaders::shader;

    shader!(
        ty: "vertex",
        path: "shaders/sky.vert"
    );
}

mod fs {
    use vulkano_shaders::shader;

    shader!(
        ty: "fragment",
        path: "shaders/sky.frag"
    );
}

pub struct SkyPipeline {
    pub inner: Arc<GraphicsPipeline>
}

impl SkyPipeline {
    pub fn new(
        device: Arc<Device>,
        layout: Arc<PipelineLayout>,
        color_format: Format,
        depth_format: Format
    ) -> Self {
        let vs = load_shader_entry_point(device.clone(), vs::load, "sky vertex");
        let fs = load_shader_entry_point(device.clone(), fs::load, "sky fragment");

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
                    topology: PrimitiveTopology::TriangleList,
                    ..InputAssemblyState::default()
                }),

                dynamic_state: [DynamicState::Viewport, DynamicState::Scissor].into_iter().collect(),
                viewport_state: Some(ViewportState::default()),

                rasterization_state: Some(RasterizationState {
                    cull_mode: CullMode::Front,
                    ..RasterizationState::default()
                }),

                multisample_state: Some(MultisampleState::default()),

                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState {
                        write_enable: false,
                        compare_op: CompareOp::LessOrEqual,
                        ..DepthState::default()
                    }),
                    ..DepthStencilState::default()
                }),

                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    1, ColorBlendAttachmentState::default()
                )),

                subpass: Some(PipelineSubpassType::BeginRendering(PipelineRenderingCreateInfo {
                    color_attachment_formats: vec![Some(color_format)],
                    depth_attachment_format: Some(depth_format),
                    ..PipelineRenderingCreateInfo::default()
                })),

                ..GraphicsPipelineCreateInfo::layout(layout)
            }
        ).map_err(|e| {
            panic!("[Sky Pipeline] Failed to create graphics pipeline:\n{:?}", e);
        }).unwrap();

        Self { inner: pipeline }
    }
}