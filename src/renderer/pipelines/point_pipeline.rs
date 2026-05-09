use std::sync::Arc;
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::pipeline::{DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::subpass::{PipelineRenderingCreateInfo, PipelineSubpassType};
use vulkano::pipeline::graphics::vertex_input::{VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, VertexInputState};
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use crate::utils::shader_loader::load_shader_entry_point;

mod vs {
    use vulkano_shaders::shader;

    shader!(
        ty: "vertex",
        path: "shaders\\simple_shader.vert"
    );
}

mod fs {
    use vulkano_shaders::shader;

    shader!(
        ty: "fragment",
        path: "shaders\\simple_shader.frag"
    );
}

pub struct PointPipeline {
    pub inner: Arc<GraphicsPipeline>
}

impl PointPipeline {
    pub fn new(
        device: Arc<Device>,
        color_format: Format,
        depth_format: Format
    ) -> Self {
        let vs = load_shader_entry_point(device.clone(), vs::load, "point vertex");
        let fs = load_shader_entry_point(device.clone(), fs::load, "point fragment");

        let stages = [
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs.clone())
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .expect("[Point Pipeline] Failed to create layout info from shaders")
        ).expect("[Point Pipeline] Failed to create PipelineLayout");

        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),

                vertex_input_state: Some(
                    VertexInputState::new()
                        .binding(0, VertexInputBindingDescription {
                            stride: 16,
                            input_rate: VertexInputRate::Vertex,
                            ..VertexInputBindingDescription::default()
                        })
                        .attribute(0, VertexInputAttributeDescription {
                            binding: 0,
                            format: Format::R32G32B32A32_SFLOAT,
                            offset: 0,
                            ..VertexInputAttributeDescription::default()
                        })

                        .binding(1, VertexInputBindingDescription {
                            stride: 16,
                            input_rate: VertexInputRate::Vertex,
                            ..VertexInputBindingDescription::default()
                        })
                        .attribute(1, VertexInputAttributeDescription {
                            binding: 1,
                            format: Format::R32G32B32A32_SFLOAT,
                            offset: 0,
                            ..VertexInputAttributeDescription::default()
                        })

                        .binding(2, VertexInputBindingDescription {
                            stride: 4,
                            input_rate: VertexInputRate::Vertex,
                            ..VertexInputBindingDescription::default()
                        })
                        .attribute(2, VertexInputAttributeDescription {
                            binding: 2,
                            format: Format::R32_SFLOAT,
                            offset: 0,
                            ..VertexInputAttributeDescription::default()
                        })
                ),
                input_assembly_state: Some(InputAssemblyState {
                    topology: PrimitiveTopology::PointList,
                    ..InputAssemblyState::default()
                }),
                dynamic_state: [DynamicState::Viewport, DynamicState::Scissor].into_iter().collect(),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
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
        ).map_err(|e| panic!("[Point Pipeline] Validation Error:\n{:?}", e)).unwrap();

        Self { inner: pipeline }
    }
}