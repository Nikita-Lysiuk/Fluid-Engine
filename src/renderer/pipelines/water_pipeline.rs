use vulkano::pipeline::graphics::rasterization::CullMode;
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::pipeline::{DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::subpass::{PipelineRenderingCreateInfo, PipelineSubpassType};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use crate::entities::ModelVertex;
use crate::utils::shader_loader::load_shader_entry_point;

mod vs {
    use vulkano_shaders::shader;

    shader! {
        ty: "vertex",
        path: "shaders\\raymarch.vert"
    }
}

mod fs {
    use vulkano_shaders::shader;

    shader! {
        ty: "fragment",
        path: "shaders\\raymarch.frag"
    }
}

pub struct WaterRenderPipeline {
    pub inner: Arc<GraphicsPipeline>,
}

impl WaterRenderPipeline {
    pub fn new(
        device: Arc<Device>,
        swapchain_format: Format,
        depth_format: Format,
    ) -> Self {
        let vs = load_shader_entry_point(device.clone(), vs::load, "water vertex");
        let fs = load_shader_entry_point(device.clone(), fs::load, "water fragment");

        let  stages = [
            PipelineShaderStageCreateInfo::new(vs.clone()).clone(),
            PipelineShaderStageCreateInfo::new(fs.clone()).clone(),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone()).unwrap()
        ).unwrap();

        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(ModelVertex::per_vertex().definition(&vs).unwrap()),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                dynamic_state: [DynamicState::Viewport, DynamicState::Scissor].into_iter().collect(),
                rasterization_state: Some(RasterizationState {
                    cull_mode: CullMode::None,
                    ..RasterizationState::default()
                }),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState {
                    attachments: vec![ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend::alpha()),
                        ..ColorBlendAttachmentState::default()
                    }],
                    ..Default::default()
                }),
                subpass: Some(PipelineSubpassType::BeginRendering(
                    PipelineRenderingCreateInfo {
                        color_attachment_formats: vec![Some(swapchain_format)],
                        depth_attachment_format: Some(depth_format),
                        ..Default::default()
                    }
                )),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        ).unwrap();

        Self { inner: pipeline }
    }

}