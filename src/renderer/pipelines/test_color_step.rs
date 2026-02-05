use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::Device;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use crate::entities::particle::{GpuPhysicsData, SimulationParams};
use crate::renderer::pipelines::ComputeStep;
use crate::utils::shader_loader::load_shader_entry_point;

mod cs {
    use vulkano_shaders::shader;

    shader!(
        ty: "compute",
        path: "shaders\\compute\\test_color_step.comp"
    );
}

pub struct TestColorStep {
    pipeline: Arc<ComputePipeline>,
}

impl TestColorStep {
    pub fn new(device: Arc<Device>) -> Self {
        let shader = load_shader_entry_point(device.clone(), cs::load, "main");

        let stage = PipelineShaderStageCreateInfo::new(shader);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap()
        ).unwrap();

        let pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout)
        ).unwrap();

        Self { pipeline }
    }
}

impl ComputeStep for TestColorStep {
    fn execute<Cb>(
        &self,
        builder: &mut AutoCommandBufferBuilder<Cb>,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &GpuPhysicsData,
        sim_params: &Subbuffer<SimulationParams>,
        dt: f32,
    ) {
        let pipeline_layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        
        let pc = cs::PushConstants {
            dt,
            particle_count: physics_data.count,
        };
        
        let descriptor_set = DescriptorSet::new(
            allocator.clone(),
            pipeline_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_data.positions.clone()),
                WriteDescriptorSet::buffer(1, physics_data.velocities.clone()),
                WriteDescriptorSet::buffer(2, physics_data.colors.clone()),
                WriteDescriptorSet::buffer(3, sim_params.clone()),
            ],
            []
        ).unwrap();
        
        builder
            .bind_pipeline_compute(self.pipeline.clone()).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                descriptor_set
            )
            .unwrap().push_constants(
                self.pipeline.layout().clone(),
                0,
                pc
            )
            .unwrap();

        let group_count = (physics_data.count + 63) / 64;
        unsafe { builder.dispatch([group_count, 1, 1]).unwrap(); }
    }
}