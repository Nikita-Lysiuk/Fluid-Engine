use std::sync::Arc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::WriteDescriptorSet;
use crate::compute_pass::ComputePass;
use vulkano::device::Device;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use crate::entities::particle::ParticleBuffers;
use crate::utils::shader_loader::load_shader_entry_point;

mod cs {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders/compute/solver_check.comp");
}

/// Convergence test after each Jacobi iteration (`solver_check.comp`).
pub struct SolverCheckPipeline {
    pipeline: Arc<ComputePipeline>,
    pass: Option<ComputePass>,
    num_particles: u32,
}

impl SolverCheckPipeline {
    pub fn new(device: Arc<Device>) -> Self {
        let stage = PipelineShaderStageCreateInfo::new(load_shader_entry_point(device.clone(), cs::load, "main"));
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let pipeline =
            ComputePipeline::new(device, None, ComputePipelineCreateInfo::stage_layout(stage, layout)).unwrap();
        Self { pipeline, pass: None, num_particles: 0 }
    }

    pub fn prepare(&mut self, allocator: Arc<StandardDescriptorSetAllocator>, physics_data: &ParticleBuffers) {
        self.num_particles = physics_data.count;
        self.pass = Some(ComputePass::new(self.pipeline.clone(), allocator, [
            WriteDescriptorSet::buffer(0, physics_data.solver_state.clone()),
            WriteDescriptorSet::buffer(1, physics_data.solver_dispatch.clone()),
        ], 1));
    }

    /// `max_avg_error` in kg/m^3; a negative value disables the early exit.
    pub fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, max_avg_error: f32, min_iterations: u32) {
        let pc = cs::PushConstants { max_avg_error, min_iterations, num_particles: self.num_particles };
        let pass = self.pass.as_ref().expect("prepare solver check before dispatch");
        pass.bind_with_constants(builder, pc);
        pass.dispatch_bound(builder);
    }
}
