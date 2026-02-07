use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::shader::EntryPoint;
use crate::entities::particle::{GpuPhysicsData, SimulationParams};
use crate::renderer::pipelines::ComputeStep;
use crate::utils::shader_loader::load_shader_entry_point;

mod cs {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\density_and_alpha.comp");
}

pub struct DensityAlphaPipeline {
    pipeline: Arc<ComputePipeline>,
}

impl ComputeStep for DensityAlphaPipeline {
    fn load_shader_module(device: Arc<Device>) -> EntryPoint {
        load_shader_entry_point(device, cs::load, "main")
    }
    fn from_pipeline(pipeline: Arc<ComputePipeline>) -> Self {
        Self { pipeline }
    }
    fn execute<Cb>(
        &self, 
        builder: &mut AutoCommandBufferBuilder<Cb>, 
        allocator: Arc<StandardDescriptorSetAllocator>, 
        physics_data: &GpuPhysicsData, 
        sim_params: &Subbuffer<SimulationParams>,
    ) {
        let num_particles = physics_data.count;
        let group_size = 256;
        let dispatch_count = (num_particles + group_size - 1) / group_size;

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();

        let set = DescriptorSet::new(
            allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_data.grid_entries.clone()),
                WriteDescriptorSet::buffer(1, physics_data.grid_start.clone()),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
                WriteDescriptorSet::buffer(3, physics_data.position_b.clone()),
                WriteDescriptorSet::buffer(4, physics_data.densities.clone()),
                WriteDescriptorSet::buffer(5, physics_data.factors.clone()),
                WriteDescriptorSet::buffer(6, physics_data.color_b.clone()),
            ],
            [],
        ).unwrap();

        builder
            .bind_pipeline_compute(self.pipeline.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.pipeline.layout().clone(), 0, set)
            .unwrap();

        unsafe { builder.dispatch([dispatch_count, 1, 1]).unwrap(); }
    }
}