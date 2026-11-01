use crate::render::shader_loader::load_shader_entry_point;
use crate::sim::compute_pass::ComputePass;
use crate::sim::params::SimulationParams;
use crate::sim::particle_buffers::ParticleBuffers;
use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::Device;
use vulkano::image::view::ImageView;
use vulkano::pipeline::ComputePipeline;
use vulkano::shader::EntryPoint;

mod splat_cs {
    use vulkano_shaders::shader;
    shader! {
        ty: "compute",
        path: "shaders\\compute\\splat_density.comp",
        include: ["shaders\\include"],
    }
}

/// Compute pipeline that splats particles into a density texture.
pub struct DensityTexturePipeline {
    pub pipeline: Arc<ComputePipeline>,
    pass: Option<ComputePass>,
}

impl DensityTexturePipeline {
    /// Prepares the pass for the density image.
    pub fn prepare_with_image(
        &mut self,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &ParticleBuffers,
        image: Arc<ImageView>,
        sim_params: &Subbuffer<SimulationParams>,
    ) {
        self.pass = Some(ComputePass::new(
            self.pipeline.clone(),
            allocator,
            [
                WriteDescriptorSet::buffer(0, physics_data.position_a.clone()),
                WriteDescriptorSet::image_view(1, image),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
            ],
            physics_data.count.div_ceil(256),
        ));
    }
}

impl DensityTexturePipeline {
    /// Creates the associated resources.
    pub fn new(device: Arc<vulkano::device::Device>) -> Self {
        let shader = Self::load_shader_module(device.clone());
        Self::from_pipeline(crate::sim::compute_pass::new_pipeline(device, shader))
    }
    fn load_shader_module(device: Arc<Device>) -> EntryPoint {
        load_shader_entry_point(device, splat_cs::load, "main")
    }
    fn from_pipeline(pipeline: Arc<ComputePipeline>) -> Self {
        Self {
            pipeline,
            pass: None,
        }
    }
    /// Records the associated GPU commands.
    pub fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        self.pass
            .as_ref()
            .expect("prepare the density texture before dispatch")
            .dispatch(builder);
    }
}
