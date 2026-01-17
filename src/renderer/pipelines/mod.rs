use std::sync::Arc;
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::PipelineLayout;
use vulkano::shader::ShaderStages;
use vulkano_util::context::VulkanoContext;
use crate::renderer::pipelines::point_pipeline::PointPipeline;
use crate::renderer::pipelines::sky_pipeline::SkyPipeline;

pub mod point_pipeline;
pub mod sky_pipeline;


pub struct Pipelines {
    pub sky_layout: Arc<PipelineLayout>,
    pub sky_pipeline: Arc<SkyPipeline>,
    pub common_layout: Arc<PipelineLayout>,
    pub point_pipeline: Arc<PointPipeline>,
}

impl Pipelines {
    pub fn new(context: Arc<VulkanoContext>, swapchain_format: Format, depth_format: Format) -> Self {
        let device = context.device().clone();

        let push_constant_range = PushConstantRange {
            stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            offset: 0,
            size: 16,
        };

        let sky_layout = Self::create_sky_layout(device.clone(), push_constant_range);
        let sky_pipeline = Arc::new(SkyPipeline::new(device.clone(), sky_layout.clone(), swapchain_format, depth_format));

        let common_layout = Self::create_common_layout(device.clone(), push_constant_range);
        let point_pipeline = Arc::new(PointPipeline::new(device.clone(), common_layout.clone(), swapchain_format, depth_format));

        Self {
            sky_layout,
            sky_pipeline,
            common_layout,
            point_pipeline,
        }
    }

    fn create_sky_layout(device: Arc<Device>, pc: PushConstantRange) -> Arc<PipelineLayout> {
        let dsl = DescriptorSetLayout::new(device.clone(), DescriptorSetLayoutCreateInfo {
            bindings: [(0, DescriptorSetLayoutBinding {
                stages: ShaderStages::FRAGMENT,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
            })].into(),
            ..Default::default()
        }).unwrap();

        PipelineLayout::new(device, PipelineLayoutCreateInfo {
            set_layouts: vec![dsl],
            push_constant_ranges: vec![pc],
            ..Default::default()
        }).unwrap()
    }
    fn create_common_layout(device: Arc<Device>, pc: PushConstantRange) -> Arc<PipelineLayout> {
        PipelineLayout::new(device, PipelineLayoutCreateInfo {
            set_layouts: vec![],
            push_constant_ranges: vec![pc],
            ..Default::default()
        }).unwrap()
    }
}