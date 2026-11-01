use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::WriteDescriptorSet;
use crate::compute_pass::ComputePass;
use vulkano::device::Device;
use vulkano::image::view::ImageView;
use vulkano::pipeline::ComputePipeline;
use vulkano::shader::EntryPoint;
use crate::entities::particle::{ParticleBuffers, SimulationParams};
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
    pass: Option<ComputePass>,
}

impl DensityTexturePipeline {
    pub fn prepare_with_image(
        &mut self,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &ParticleBuffers,
        image: Arc<ImageView>,
        sim_params: &Subbuffer<SimulationParams>,
    ) {
        self.pass = Some(ComputePass::new(self.pipeline.clone(), allocator, [
            WriteDescriptorSet::buffer(0, physics_data.position_a.clone()),
            WriteDescriptorSet::image_view(1, image),
            WriteDescriptorSet::buffer(2, sim_params.clone()),
        ], physics_data.count.div_ceil(256)));
    }
}

impl DensityTexturePipeline {
    pub fn new(device: Arc<vulkano::device::Device>) -> Self {
        let shader = Self::load_shader_module(device.clone());
        Self::from_pipeline(crate::compute_pass::new_pipeline(device, shader))
    }
    fn load_shader_module(device: Arc<Device>) -> EntryPoint {
        load_shader_entry_point(device, splat_cs::load, "main")
    }
    fn from_pipeline(pipeline: Arc<ComputePipeline>) -> Self {
        Self { pipeline, pass: None }
    }
    pub fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        self.pass.as_ref().expect("prepare the density texture before dispatch").dispatch(builder);
    }
}
