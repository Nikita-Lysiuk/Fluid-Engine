use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::image::view::ImageView;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::shader::EntryPoint;
use crate::entities::particle::{GpuPhysicsData, SimulationParams};
use crate::renderer::pipelines::ComputeStep;
use crate::utils::shader_loader::load_shader_entry_point;

mod splat_cs {
    use vulkano_shaders::shader;
    shader! {
        ty: "compute",
        path: "shaders\\compute\\splat_density.comp",
        include: ["shaders\\include"],
    }
}

pub struct DensityTexturePipeline {
    pub pipeline: Arc<ComputePipeline>,
    descriptor_set: Option<Arc<DescriptorSet>>,
    dispatch_count: u32,
}

impl DensityTexturePipeline {
    pub fn prepare_with_image(
        &mut self,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &GpuPhysicsData,
        image: Arc<ImageView>,
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
                WriteDescriptorSet::buffer(0, physics_data.position_a.clone()),
                WriteDescriptorSet::image_view(1, image),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
            ],
            []
        ).unwrap());
    }
}

impl ComputeStep for DensityTexturePipeline {
    fn load_shader_module(device: Arc<Device>) -> EntryPoint {
        load_shader_entry_point(device, splat_cs::load, "main")
    }
    fn from_pipeline(pipeline: Arc<ComputePipeline>) -> Self {
        Self { pipeline, descriptor_set: None, dispatch_count: 0 }
    }
    fn prepare(
        &mut self,
        _allocator: Arc<StandardDescriptorSetAllocator>,
        _physics_data: &GpuPhysicsData,
        _sim_params: &Subbuffer<SimulationParams>,
    ) {
        // Use prepare_with_image() instead, as an ImageView is required.
    }
    fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        let set = self.descriptor_set.as_ref().expect("DensityTexturePipeline: call prepare_with_image() before execute()");
        builder
            .bind_pipeline_compute(self.pipeline.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.pipeline.layout().clone(), 0, set.clone())
            .unwrap();
        unsafe { builder.dispatch([self.dispatch_count, 1, 1]).unwrap(); }
    }
}