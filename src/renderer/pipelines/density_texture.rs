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
    pub pipeline: Arc<ComputePipeline>
}

impl DensityTexturePipeline {
    pub fn execute_with_image<Cb>(
        &self,
        builder: &mut AutoCommandBufferBuilder<Cb>,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &GpuPhysicsData,
        image: Arc<ImageView>,
        sim_params: &Subbuffer<SimulationParams>
    ) {
        let num_particles = physics_data.count;
        let group_size = 256;
        let dispatch_count = (num_particles + group_size - 1) / group_size;

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();

        let set = DescriptorSet::new(
            allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_data.position_a.clone()),
                WriteDescriptorSet::image_view(1, image.clone()),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
            ],
            []
        ).unwrap();

        builder
            .bind_pipeline_compute(self.pipeline.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.pipeline.layout().clone(), 0, set)
            .unwrap();

        unsafe {
            builder.dispatch([dispatch_count, 1, 1]).unwrap();
        }

    }
}

impl ComputeStep for DensityTexturePipeline {
    fn load_shader_module(device: Arc<Device>) -> EntryPoint {
        load_shader_entry_point(device, splat_cs::load, "main")
    }
    fn from_pipeline(pipeline: Arc<ComputePipeline>) -> Self {
        Self {
            pipeline,
        }
    }
    fn execute<Cb>(&self, _builder: &mut AutoCommandBufferBuilder<Cb>, _allocator: Arc<StandardDescriptorSetAllocator>, _physics_data: &GpuPhysicsData, _sim_params: &Subbuffer<SimulationParams>) {
        unimplemented!("use execute_with_image instead")
    }
}