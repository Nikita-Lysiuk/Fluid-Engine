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
    shader! { ty: "compute", path: "shaders\\compute\\stats.comp" }
}

pub struct StatsPipeline {
    pub pipeline: Arc<ComputePipeline>,
    descriptor_set: Option<Arc<DescriptorSet>>,
    dispatch_count: u32,
    stats_buffer: Option<Subbuffer<[u32]>>,
}

impl ComputeStep for StatsPipeline {
    fn load_shader_module(device: Arc<Device>) -> EntryPoint {
        load_shader_entry_point(device, cs::load, "main")
    }
    fn from_pipeline(pipeline: Arc<ComputePipeline>) -> Self {
        Self { pipeline, descriptor_set: None, dispatch_count: 0, stats_buffer: None }
    }
    fn prepare(
        &mut self,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &GpuPhysicsData,
        sim_params: &Subbuffer<SimulationParams>,
    ) {
        let num_particles = physics_data.count;
        self.dispatch_count = (num_particles + 255) / 256;
        self.stats_buffer = Some(physics_data.stats_buffer.clone());

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        self.descriptor_set = Some(DescriptorSet::new(
            allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_data.velocity_a.clone()),
                WriteDescriptorSet::buffer(1, physics_data.stats_buffer.clone()),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
                WriteDescriptorSet::buffer(3, physics_data.densities.clone()),
                WriteDescriptorSet::buffer(4, physics_data.source_terms.clone()),
            ],
            []
        ).unwrap());
    }
    fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        let buf = self.stats_buffer.as_ref().expect("StatsPipeline: call prepare() before execute()");
        builder.fill_buffer(buf.clone(), 0).unwrap();

        let set = self.descriptor_set.as_ref().unwrap();
        builder
            .bind_pipeline_compute(self.pipeline.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.pipeline.layout().clone(), 0, set.clone())
            .unwrap();
        unsafe { builder.dispatch([self.dispatch_count, 1, 1]).unwrap(); }
    }
}
