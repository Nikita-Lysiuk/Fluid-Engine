use std::sync::Arc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use crate::entities::particle::GpuPhysicsData;
use crate::utils::shader_loader::load_shader_entry_point;

mod cs {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders/compute/solver_check.comp");
}

/// Convergence test after each Jacobi iteration (`solver_check.comp`).
pub struct SolverCheckPipeline {
    pipeline: Arc<ComputePipeline>,
    descriptor_set: Option<Arc<DescriptorSet>>,
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
        Self { pipeline, descriptor_set: None, num_particles: 0 }
    }

    pub fn prepare(&mut self, allocator: Arc<StandardDescriptorSetAllocator>, physics_data: &GpuPhysicsData) {
        self.num_particles = physics_data.count;
        let layout = self.pipeline.layout().set_layouts()[0].clone();
        self.descriptor_set = Some(
            DescriptorSet::new(
                allocator,
                layout,
                [
                    WriteDescriptorSet::buffer(0, physics_data.solver_state.clone()),
                    WriteDescriptorSet::buffer(1, physics_data.solver_dispatch.clone()),
                ],
                [],
            )
            .unwrap(),
        );
    }

    /// `max_avg_error` in kg/m^3; a negative value disables the early exit.
    pub fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, max_avg_error: f32, min_iterations: u32) {
        let set = self.descriptor_set.as_ref().expect("SolverCheckPipeline: call prepare() before execute()");
        let pc = cs::PushConstants { max_avg_error, min_iterations, num_particles: self.num_particles };
        builder
            .bind_pipeline_compute(self.pipeline.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.pipeline.layout().clone(), 0, set.clone()).unwrap()
            .push_constants(self.pipeline.layout().clone(), 0, pc).unwrap();
        unsafe { builder.dispatch([1, 1, 1]).unwrap(); }
    }
}
