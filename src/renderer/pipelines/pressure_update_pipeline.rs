use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DispatchIndirectCommand};
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
    shader!(
        ty: "compute",
        path: "shaders\\compute\\pressure_update.comp"
    );
}

pub struct PressureUpdatePipeline {
    pub pipeline: Arc<ComputePipeline>,
    pass: Option<ComputePass>,
}

impl PressureUpdatePipeline {
    pub fn new(device: Arc<vulkano::device::Device>) -> Self {
        let shader = Self::load_shader_module(device.clone());
        Self::from_pipeline(crate::compute_pass::new_pipeline(device, shader))
    }
    fn load_shader_module(device: Arc<Device>) -> EntryPoint {
        load_shader_entry_point(device, cs::load, "main")
    }
    fn from_pipeline(pipeline: Arc<ComputePipeline>) -> Self {
        Self { pipeline, pass: None }
    }
    pub fn prepare(
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
                WriteDescriptorSet::buffer(3, physics_data.position_b.clone()),
                WriteDescriptorSet::buffer(4, physics_data.pressure_accelerations.clone()),
                WriteDescriptorSet::buffer(5, physics_data.factors.clone()),
                WriteDescriptorSet::buffer(6, physics_data.source_terms.clone()),
                WriteDescriptorSet::buffer(7, physics_data.densities.clone()),
                WriteDescriptorSet::buffer(8, physics_data.pressures.clone()),
                WriteDescriptorSet::buffer(9, physics_data.solver_state.clone()),
            ],
            physics_data.count.div_ceil(256),
        ));
    }


}

impl PressureUpdatePipeline {
    fn bind<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, is_divergence_solve: bool) {
        let pc = cs::PushConstants { is_divergence_solve: is_divergence_solve as u32 };
        self.pass.as_ref().expect("prepare pressure update before dispatch").bind_with_constants(builder, pc);
    }

    /// Same as `execute`, but the workgroup count is read from `args`, so a
    /// converged solver can skip the dispatch without a CPU round trip.
    pub fn execute_indirect<Cb>(
        &self,
        builder: &mut AutoCommandBufferBuilder<Cb>,
        args: Subbuffer<[DispatchIndirectCommand]>,
        is_divergence_solve: bool,
    ) {
        self.bind(builder, is_divergence_solve);
        self.pass.as_ref().expect("prepare pressure update before dispatch").dispatch_indirect_bound(builder, args);
    }
}
