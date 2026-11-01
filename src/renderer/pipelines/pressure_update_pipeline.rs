use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DispatchIndirectCommand};
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
    shader!(
        ty: "compute",
        path: "shaders\\compute\\pressure_update.comp"
    );
}

pub struct PressureUpdatePipeline {
    pub pipeline: Arc<ComputePipeline>,
    descriptor_set: Option<Arc<DescriptorSet>>,
    dispatch_count: u32,
}

impl ComputeStep for PressureUpdatePipeline {
    fn load_shader_module(device: Arc<Device>) -> EntryPoint {
        load_shader_entry_point(device, cs::load, "main")
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
            []
        ).unwrap());
    }
    fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        self.bind(builder, false);
        unsafe { builder.dispatch([self.dispatch_count, 1, 1]).unwrap(); }
    }
}

impl PressureUpdatePipeline {
    fn bind<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, is_divergence_solve: bool) {
        let set = self.descriptor_set.as_ref().expect("PressureUpdatePipeline: call prepare() before execute()");
        let pc = cs::PushConstants { is_divergence_solve: is_divergence_solve as u32 };
        builder
            .bind_pipeline_compute(self.pipeline.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.pipeline.layout().clone(), 0, set.clone()).unwrap()
            .push_constants(self.pipeline.layout().clone(), 0, pc).unwrap();
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
        // SAFETY: `args` is written only by `solver_check.comp`, which keeps
        // the group count at either the full particle count or zero.
        unsafe { builder.dispatch_indirect(args).unwrap(); }
    }
}
