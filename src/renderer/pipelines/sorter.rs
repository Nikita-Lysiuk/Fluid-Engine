use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
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
    shader!(ty: "compute", path: "shaders\\compute\\bitonic_sort.comp");
}

pub struct GpuSorter {
    pipeline: Arc<ComputePipeline>,
    descriptor_set: Option<Arc<DescriptorSet>>,
    num_elements: u32,
}

impl ComputeStep for GpuSorter {
    fn load_shader_module(device: Arc<Device>) -> EntryPoint {
        load_shader_entry_point(device, cs::load, "main")
    }
    fn from_pipeline(pipeline: Arc<ComputePipeline>) -> Self {
        Self { pipeline, descriptor_set: None, num_elements: 0 }
    }
    fn prepare(
        &mut self,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &GpuPhysicsData,
        _sim_params: &Subbuffer<SimulationParams>,
    ) {
        let grid_entries = &physics_data.grid_entries;
        self.num_elements = grid_entries.len() as u32;
        assert!(self.num_elements.is_power_of_two(), "GpuSorter: Buffer len must be power of 2");

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        self.descriptor_set = Some(DescriptorSet::new(
            allocator,
            layout.clone(),
            [WriteDescriptorSet::buffer(0, grid_entries.clone())],
            []
        ).unwrap());
    }
    fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        let set = self.descriptor_set.as_ref().expect("GpuSorter: call prepare() before execute()");
        let num_elements = self.num_elements;

        let mut h = 2;
        while h <= num_elements {
            let mut step = h / 2;
            while step > 0 {
                let pc = cs::SortConstants {
                    num_entries: num_elements,
                    block_height: h,
                    block_step: step,
                };

                builder
                    .bind_pipeline_compute(self.pipeline.clone()).unwrap()
                    .bind_descriptor_sets(PipelineBindPoint::Compute, self.pipeline.layout().clone(), 0, set.clone()).unwrap()
                    .push_constants(self.pipeline.layout().clone(), 0, pc).unwrap();

                let threads_needed = num_elements / 2;
                let group_size = 256;
                let dispatch_count = (threads_needed + group_size - 1) / group_size;

                unsafe { builder.dispatch([dispatch_count, 1, 1]).unwrap(); }

                step /= 2;
            }
            h *= 2;
        }
    }
}






















