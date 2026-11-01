use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::WriteDescriptorSet;
use crate::compute_pass::ComputePass;
use vulkano::device::Device;
use vulkano::pipeline::ComputePipeline;
use vulkano::shader::EntryPoint;
use crate::entities::particle::{ParticleBuffers, SimulationParams};
use crate::renderer::pipelines::ComputeStep;
use crate::utils::shader_loader::load_shader_entry_point;

mod cs {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\viscosity.comp");
}

pub struct ViscosityPipeline {
    pipeline: Arc<ComputePipeline>,
    pass: Option<ComputePass>,
}

impl ComputeStep for ViscosityPipeline {
    fn load_shader_module(device: Arc<Device>) -> EntryPoint {
        load_shader_entry_point(device, cs::load, "main")
    }
    fn from_pipeline(pipeline: Arc<ComputePipeline>) -> Self {
        Self { pipeline, pass: None }
    }
    fn prepare(
        &mut self,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &ParticleBuffers,
        sim_params: &Subbuffer<SimulationParams>,
    ) {
        self.pass = Some(ComputePass::new(
            self.pipeline.clone(), allocator,
            [
                WriteDescriptorSet::buffer(0, physics_data.neighbor_list.clone()),
                WriteDescriptorSet::buffer(1, physics_data.neighbor_counts.clone()),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
                WriteDescriptorSet::buffer(3, physics_data.velocity_b.clone()),
                WriteDescriptorSet::buffer(4, physics_data.position_b.clone()),
                WriteDescriptorSet::buffer(5, physics_data.densities.clone()),
                WriteDescriptorSet::buffer(6, physics_data.velocity_a.clone()),
            ],
            physics_data.count.div_ceil(256),
        ));
    }

    fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        self.pass.as_ref().expect("prepare the compute pass before dispatch").dispatch(builder);
    }
}
