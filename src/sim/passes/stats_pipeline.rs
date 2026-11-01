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
use crate::utils::shader_loader::load_shader_entry_point;

mod cs {
    use vulkano_shaders::shader;
    shader! { ty: "compute", path: "shaders\\compute\\stats.comp" }
}

pub struct StatsPipeline {
    pub pipeline: Arc<ComputePipeline>,
    pass: Option<ComputePass>,
    stats_buffer: Option<Subbuffer<[u32]>>,
}

impl StatsPipeline {
    pub fn new(device: Arc<vulkano::device::Device>) -> Self {
        let shader = Self::load_shader_module(device.clone());
        Self::from_pipeline(crate::compute_pass::new_pipeline(device, shader))
    }
    fn load_shader_module(device: Arc<Device>) -> EntryPoint {
        load_shader_entry_point(device, cs::load, "main")
    }
    fn from_pipeline(pipeline: Arc<ComputePipeline>) -> Self {
        Self { pipeline, pass: None, stats_buffer: None }
    }
    pub fn prepare(
        &mut self,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &ParticleBuffers,
        _sim_params: &Subbuffer<SimulationParams>,
    ) {
        self.stats_buffer = Some(physics_data.stats_buffer.clone());
        self.pass = Some(ComputePass::new(
            self.pipeline.clone(), allocator,
            [
                WriteDescriptorSet::buffer(0, physics_data.velocity_a.clone()),
                WriteDescriptorSet::buffer(1, physics_data.stats_buffer.clone().slice(0..1)),
            ],
            physics_data.count.div_ceil(256),
        ));
    }

    pub fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        let buf = self.stats_buffer.as_ref().expect("StatsPipeline: call prepare() before execute()");
        builder.fill_buffer(buf.clone().slice(0..1), 0).unwrap();

        self.pass.as_ref().expect("prepare stats before dispatch").dispatch(builder);
    }
}
