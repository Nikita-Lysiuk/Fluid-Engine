use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use crate::entities::particle::{GpuPhysicsData, SimulationParams};
use crate::renderer::pipelines::ComputeStep;

mod cs {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\density_source_term.comp");
}

pub struct DensitySourceTermPipeline {
    pipeline: Arc<ComputePipeline>,
    descriptor_set: Option<Arc<DescriptorSet>>,
    dispatch_count: u32,
}

impl ComputeStep for DensitySourceTermPipeline {
    fn load_shader_module(device: Arc<vulkano::device::Device>) -> vulkano::shader::EntryPoint {
        crate::utils::shader_loader::load_shader_entry_point(device, cs::load, "main")
    }
    fn from_pipeline(pipeline: Arc<ComputePipeline>) -> Self {
        Self { pipeline, descriptor_set: None, dispatch_count: 0 }
    }
    fn prepare(
        &mut self,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &GpuPhysicsData,
        sim_params: &Subbuffer<SimulationParams>,
    ) {
        let num_particles = physics_data.count;
        let group_size = 256;
        self.dispatch_count = (num_particles + group_size - 1) / group_size;

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        self.descriptor_set = Some(DescriptorSet::new(
            allocator,
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
        ).unwrap());
    }
    fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        let set = self.descriptor_set.as_ref().expect("DensitySourceTermPipeline: call prepare() before execute()");
        builder
            .bind_pipeline_compute(self.pipeline.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.pipeline.layout().clone(), 0, set.clone())
            .unwrap();
        unsafe { builder.dispatch([self.dispatch_count, 1, 1]).unwrap(); }
    }
}