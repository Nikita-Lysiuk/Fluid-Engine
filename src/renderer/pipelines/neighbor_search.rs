use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::instance::debug::DebugUtilsLabel;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::Device;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::shader::EntryPoint;
use crate::entities::particle::{GpuPhysicsData, SimulationParams};
use crate::renderer::pipelines::{ComputeStep, SortAlgorithm};
use crate::renderer::pipelines::sorter::{GpuSorter, RadixSorter};
use crate::utils::shader_loader::load_shader_entry_point;

mod cs_hash {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\spatial_hash.comp");
}
mod cs_offsets {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\grid_offsets.comp");
}
mod cs_reorder {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\reorder.comp");
}

pub struct NeighborSearch {
    spatial_hash_pipeline: Arc<ComputePipeline>,
    offsets_pipeline: Arc<ComputePipeline>,
    reorder_pipeline: Arc<ComputePipeline>,

    sorter: GpuSorter,
    radix_sorter: RadixSorter,

    hash_set: Option<Arc<DescriptorSet>>,
    offsets_set: Option<Arc<DescriptorSet>>,
    reorder_set: Option<Arc<DescriptorSet>>,

    grid_start: Option<Subbuffer<[u32]>>,

    sort_buffer_len: u32,
    num_particles: u32,

    pub sort_algorithm: SortAlgorithm,
}

impl NeighborSearch {
    pub fn new_with_allocator(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        sort_buffer_size: u32,
    ) -> Self {
        let sorter = GpuSorter::new(device.clone());
        let radix_sorter = RadixSorter::new(device.clone(), memory_allocator, sort_buffer_size);

        let hash_shader = load_shader_entry_point(device.clone(), cs_hash::load, "main");
        let hash_stage = PipelineShaderStageCreateInfo::new(hash_shader);
        let hash_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&hash_stage])
                .into_pipeline_layout_create_info(device.clone()).unwrap()
        ).unwrap();
        let spatial_hash_pipeline = ComputePipeline::new(
            device.clone(), None, ComputePipelineCreateInfo::stage_layout(hash_stage, hash_layout)
        ).unwrap();

        let offsets_shader = load_shader_entry_point(device.clone(), cs_offsets::load, "main");
        let offsets_stage = PipelineShaderStageCreateInfo::new(offsets_shader);
        let offsets_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&offsets_stage])
                .into_pipeline_layout_create_info(device.clone()).unwrap()
        ).unwrap();
        let offsets_pipeline = ComputePipeline::new(
            device.clone(), None, ComputePipelineCreateInfo::stage_layout(offsets_stage, offsets_layout)
        ).unwrap();

        let reorder_shader = load_shader_entry_point(device.clone(), cs_reorder::load, "main");
        let reorder_stage = PipelineShaderStageCreateInfo::new(reorder_shader);
        let reorder_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&reorder_stage])
                .into_pipeline_layout_create_info(device.clone()).unwrap()
        ).unwrap();
        let reorder_pipeline = ComputePipeline::new(
            device.clone(), None, ComputePipelineCreateInfo::stage_layout(reorder_stage, reorder_layout)
        ).unwrap();

        Self {
            spatial_hash_pipeline,
            offsets_pipeline,
            reorder_pipeline,
            sorter,
            radix_sorter,
            sort_algorithm: SortAlgorithm::Bitonic,
            hash_set: None,
            offsets_set: None,
            reorder_set: None,
            grid_start: None,
            sort_buffer_len: 0,
            num_particles: 0,
        }
    }
}

impl ComputeStep for NeighborSearch {
    fn load_shader_module(_device: Arc<Device>) -> EntryPoint {
        unimplemented!("NeighborSearch uses multiple shaders")
    }
    fn from_pipeline(_pipeline: Arc<ComputePipeline>) -> Self {
        unimplemented!("NeighborSearch uses multiple pipelines")
    }
    fn new(_device: Arc<Device>) -> Self {
        unimplemented!("NeighborSearch requires a memory allocator — use new_with_allocator")
    }
    fn prepare(
        &mut self,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &GpuPhysicsData,
        sim_params: &Subbuffer<SimulationParams>,
    ) {
        self.num_particles = physics_data.count;
        self.sort_buffer_len = physics_data.grid_entries.len() as u32;
        self.grid_start = Some(physics_data.grid_start.clone());

        {
            let layout = self.spatial_hash_pipeline.layout().set_layouts().get(0).unwrap();
            self.hash_set = Some(DescriptorSet::new(
                allocator.clone(),
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, physics_data.position_a.clone()),
                    WriteDescriptorSet::buffer(1, physics_data.grid_entries.clone()),
                    WriteDescriptorSet::buffer(2, sim_params.clone()),
                ],
                [],
            ).unwrap());
        }

        {
            let layout = self.offsets_pipeline.layout().set_layouts().get(0).unwrap();
            self.offsets_set = Some(DescriptorSet::new(
                allocator.clone(),
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, physics_data.grid_entries.clone()),
                    WriteDescriptorSet::buffer(1, physics_data.grid_start.clone()),
                ],
                [],
            ).unwrap());
        }

        {
            let layout = self.reorder_pipeline.layout().set_layouts().get(0).unwrap();
            self.reorder_set = Some(DescriptorSet::new(
                allocator.clone(),
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, physics_data.grid_entries.clone()),
                    WriteDescriptorSet::buffer(1, physics_data.position_a.clone()),
                    WriteDescriptorSet::buffer(2, physics_data.velocity_a.clone()),
                    WriteDescriptorSet::buffer(4, physics_data.position_b.clone()),
                    WriteDescriptorSet::buffer(5, physics_data.velocity_b.clone()),
                ],
                [],
            ).unwrap());
        }

        self.sorter.prepare(allocator.clone(), physics_data, sim_params);
        self.radix_sorter.prepare(allocator, &physics_data.grid_entries);
    }
    fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        let sort_buffer_len = self.sort_buffer_len;
        let num_particles = self.num_particles;
        let group_size = 256;

        {
            let set = self.hash_set.as_ref().expect("NeighborSearch: call prepare() before execute()");
            let pc = cs_hash::PushConstants { num_particles, table_size: sort_buffer_len };
            let dispatch_count = (sort_buffer_len + group_size - 1) / group_size;
            builder
                .bind_pipeline_compute(self.spatial_hash_pipeline.clone()).unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, self.spatial_hash_pipeline.layout().clone(), 0, set.clone()).unwrap()
                .push_constants(self.spatial_hash_pipeline.layout().clone(), 0, pc).unwrap();
            unsafe { builder.dispatch([dispatch_count, 1, 1]).unwrap(); }
        }

        match self.sort_algorithm {
            SortAlgorithm::Bitonic => self.sorter.execute(builder),
            SortAlgorithm::Radix   => {
                builder.begin_debug_utils_label(DebugUtilsLabel {
                    label_name: "Radix Sort".into(),
                    color: [1.0, 0.3, 0.0, 1.0],
                    ..Default::default()
                }).unwrap();
                self.radix_sorter.execute(builder);
                unsafe {
                    builder.end_debug_utils_label().unwrap();
                }
            },
        }

        let grid_start = self.grid_start.as_ref().expect("NeighborSearch: grid_start not set");
        builder.fill_buffer(grid_start.clone(), 0xFFFFFFFF).unwrap();

        {
            let set = self.offsets_set.as_ref().unwrap();
            let pc = cs_offsets::PushConstants { num_entries: sort_buffer_len };
            let dispatch_count = (sort_buffer_len + group_size - 1) / group_size;
            builder
                .bind_pipeline_compute(self.offsets_pipeline.clone()).unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, self.offsets_pipeline.layout().clone(), 0, set.clone()).unwrap()
                .push_constants(self.offsets_pipeline.layout().clone(), 0, pc).unwrap();
            unsafe { builder.dispatch([dispatch_count, 1, 1]).unwrap(); }
        }

        {
            let set = self.reorder_set.as_ref().unwrap();
            let pc = cs_reorder::PushConstants { num_particles };
            let dispatch_count = (num_particles + group_size - 1) / group_size;
            builder
                .bind_pipeline_compute(self.reorder_pipeline.clone()).unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, self.reorder_pipeline.layout().clone(), 0, set.clone()).unwrap()
                .push_constants(self.reorder_pipeline.layout().clone(), 0, pc).unwrap();
            unsafe { builder.dispatch([dispatch_count, 1, 1]).unwrap(); }
        }
    }
}
