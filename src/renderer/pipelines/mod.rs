use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::pipeline::layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::{ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::shader::{EntryPoint, ShaderStages};
use vulkano_util::context::VulkanoContext;
use crate::entities::particle::{GpuPhysicsData, SimulationParams};
use crate::renderer::pipelines::collision_pipeline::CollisionPipeline;
use crate::renderer::pipelines::density_alpha::DensityAlphaPipeline;
use crate::renderer::pipelines::density_source_term::DensitySourceTermPipeline;
use crate::renderer::pipelines::divergence_integration::DivergenceIntegrationPipeline;
use crate::renderer::pipelines::divergence_source_term::DivergenceSourceTermPipeline;
use crate::renderer::pipelines::neighbor_search::NeighborSearch;
use crate::renderer::pipelines::point_pipeline::PointPipeline;
use crate::renderer::pipelines::pressure_force_pipeline::PressureForcePipeline;
use crate::renderer::pipelines::pressure_integration_pipeline::PressureIntegrationPipeline;
use crate::renderer::pipelines::pressure_update_pipeline::PressureUpdatePipeline;
use crate::renderer::pipelines::sky_pipeline::SkyPipeline;
use crate::renderer::pipelines::viscosity::ViscosityPipeline;

pub mod point_pipeline;
pub mod sky_pipeline;
pub mod collision_pipeline;
mod neighbor_search;
mod sorter;
mod density_alpha;
mod viscosity;
mod density_source_term;
mod pressure_force_pipeline;
mod pressure_update_pipeline;
mod pressure_integration_pipeline;
mod divergence_source_term;
mod divergence_integration;

pub trait ComputeStep: Sized {
    fn load_shader_module(device: Arc<Device>) -> EntryPoint;
    fn from_pipeline(pipeline: Arc<ComputePipeline>) -> Self;
    fn new(device: Arc<Device>) -> Self {
        let entry_point = Self::load_shader_module(device.clone());

        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone()).unwrap()
        ).unwrap();

        let pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout)
        ).unwrap();

        Self::from_pipeline(pipeline)
    }
    fn execute<Cb>(
        &self,
        builder: &mut AutoCommandBufferBuilder<Cb>,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &GpuPhysicsData,
        sim_params: &Subbuffer<SimulationParams>,
    );
}

pub struct ComputePipelines {
    pub neighbor_search: NeighborSearch,
    pub density_alpha: DensityAlphaPipeline,
    pub viscosity: ViscosityPipeline,
    pub density_source_term: DensitySourceTermPipeline,
    pub pressure_force: PressureForcePipeline,
    pub pressure_update: PressureUpdatePipeline,
    pub pressure_integration: PressureIntegrationPipeline,
    pub divergence_source_term: DivergenceSourceTermPipeline,
    pub divergence_integration: DivergenceIntegrationPipeline,
}

impl ComputePipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let neighbor_search = NeighborSearch::new(device.clone());
        let density_alpha = DensityAlphaPipeline::new(device.clone());
        let viscosity = ViscosityPipeline::new(device.clone());
        let density_source_term = DensitySourceTermPipeline::new(device.clone());
        let pressure_force = PressureForcePipeline::new(device.clone());
        let pressure_update = PressureUpdatePipeline::new(device.clone());
        let pressure_integration = PressureIntegrationPipeline::new(device.clone());
        let divergence_source_term = DivergenceSourceTermPipeline::new(device.clone());
        let divergence_integration = DivergenceIntegrationPipeline::new(device.clone());

        Self {
            neighbor_search,
            density_alpha,
            viscosity,
            density_source_term,
            pressure_force,
            pressure_update,
            pressure_integration,
            divergence_source_term,
            divergence_integration
        }
    }
}


pub struct Pipelines {
    pub sky_layout: Arc<PipelineLayout>,
    pub sky_pipeline: Arc<SkyPipeline>,
    pub common_layout: Arc<PipelineLayout>,
    pub point_pipeline: Arc<PointPipeline>,
    pub collision_pipeline: Arc<CollisionPipeline>
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
        let point_pipeline = Arc::new(PointPipeline::new(device.clone(), swapchain_format, depth_format));
        let collision_pipeline = Arc::new(CollisionPipeline::new(device.clone(), common_layout.clone(), swapchain_format, depth_format));

        Self {
            sky_layout,
            sky_pipeline,
            common_layout,
            point_pipeline,
            collision_pipeline
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