use std::sync::Arc;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use crate::renderer::pipelines::ComputeStep;

mod cs {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\density_source_term.comp");
}

pub struct DensitySourceTermPipeline {
    pipeline: Arc<ComputePipeline>,
}

impl ComputeStep for DensitySourceTermPipeline {
    fn load_shader_module(device: Arc<vulkano::device::Device>) -> vulkano::shader::EntryPoint {
        crate::utils::shader_loader::load_shader_entry_point(device, cs::load, "main")
    }
    fn from_pipeline(pipeline: Arc<ComputePipeline>) -> Self {
        Self { pipeline }
    }
    fn execute<Cb>(
        &self, 
        builder: &mut vulkano::command_buffer::AutoCommandBufferBuilder<Cb>, 
        allocator: Arc<vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator>, 
        physics_data: &crate::entities::particle::GpuPhysicsData, 
        sim_params: &vulkano::buffer::Subbuffer<crate::entities::particle::SimulationParams>,
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
                WriteDescriptorSet::buffer(5, physics_data.velocity_a.clone()),
                WriteDescriptorSet::buffer(6, physics_data.pressures.clone()),
                WriteDescriptorSet::buffer(7, physics_data.source_terms.clone()),
            ],
            []
        ).unwrap();

        builder
            .bind_pipeline_compute(self.pipeline.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.pipeline.layout().clone(), 0, set)
            .unwrap();

        unsafe {
            builder.dispatch([dispatch_count, 1, 1]).unwrap();
        }
    }
}