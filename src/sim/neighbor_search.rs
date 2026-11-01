use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo};
use vulkano::instance::debug::DebugUtilsLabel;
use vulkano::descriptor_set::WriteDescriptorSet;
use crate::compute_pass::ComputePass;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::Device;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use crate::particle::{ParticleBuffers, SimulationParams};
use crate::sim_pipelines::SortAlgorithm;
use crate::sim_pipelines::sorter::{BitonicSorter, RadixSorter};
use crate::shader_loader::load_shader_entry_point;

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

    sorter: BitonicSorter,
    radix_sorter: RadixSorter,

    hash_pass: Option<ComputePass>,
    offsets_pass: Option<ComputePass>,
    reorder_pass: Option<ComputePass>,

    grid_start: Option<Subbuffer<[u32]>>,
    warm_start: Option<(Subbuffer<[[f32; 2]]>, Subbuffer<[[f32; 2]]>)>,

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
        let sorter = BitonicSorter::new(device.clone());
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
            hash_pass: None,
            offsets_pass: None,
            reorder_pass: None,
            grid_start: None,
            warm_start: None,
            sort_buffer_len: 0,
            num_particles: 0,
        }
    }
}

impl NeighborSearch {
    pub fn prepare(
        &mut self,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &ParticleBuffers,
        sim_params: &Subbuffer<SimulationParams>,
    ) {
        self.num_particles = physics_data.count;
        self.sort_buffer_len = physics_data.grid_entries.len() as u32;
        self.grid_start = Some(physics_data.grid_start.clone());
        self.warm_start = Some((physics_data.warm_start_sorted.clone(), physics_data.warm_start.clone()));

        self.hash_pass = Some(ComputePass::new(self.spatial_hash_pipeline.clone(), allocator.clone(), [
            WriteDescriptorSet::buffer(0, physics_data.position_a.clone()),
            WriteDescriptorSet::buffer(1, physics_data.grid_entries.clone()),
            WriteDescriptorSet::buffer(2, sim_params.clone()),
        ], self.sort_buffer_len.div_ceil(256)));
        self.offsets_pass = Some(ComputePass::new(self.offsets_pipeline.clone(), allocator.clone(), [
            WriteDescriptorSet::buffer(0, physics_data.grid_entries.clone()),
            WriteDescriptorSet::buffer(1, physics_data.grid_start.clone()),
        ], self.sort_buffer_len.div_ceil(256)));
        self.reorder_pass = Some(ComputePass::new(self.reorder_pipeline.clone(), allocator.clone(), [
            WriteDescriptorSet::buffer(0, physics_data.grid_entries.clone()),
            WriteDescriptorSet::buffer(1, physics_data.position_a.clone()),
            WriteDescriptorSet::buffer(2, physics_data.velocity_a.clone()),
            WriteDescriptorSet::buffer(4, physics_data.position_b.clone()),
            WriteDescriptorSet::buffer(5, physics_data.velocity_b.clone()),
            WriteDescriptorSet::buffer(6, physics_data.warm_start.clone()),
            WriteDescriptorSet::buffer(7, physics_data.warm_start_sorted.clone()),
        ], self.num_particles.div_ceil(256)));

        self.sorter.prepare(allocator.clone(), physics_data, sim_params);
        self.radix_sorter.prepare(allocator, &physics_data.grid_entries);
    }
    pub fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        let sort_buffer_len = self.sort_buffer_len;
        let num_particles = self.num_particles;

        let pass = self.hash_pass.as_ref().expect("prepare neighbor hash before dispatch");
        pass.bind_with_constants(builder, cs_hash::PushConstants { num_particles, table_size: sort_buffer_len });
        pass.dispatch_bound(builder);

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

        let pass = self.offsets_pass.as_ref().expect("prepare grid offsets before dispatch");
        pass.bind_with_constants(builder, cs_offsets::PushConstants { num_entries: sort_buffer_len });
        pass.dispatch_bound(builder);

        let pass = self.reorder_pass.as_ref().expect("prepare particle reorder before dispatch");
        pass.bind_with_constants(builder, cs_reorder::PushConstants { num_particles });
        pass.dispatch_bound(builder);
        let (sorted, warm_start) = self.warm_start.clone().expect("prepare warm start buffers");
        builder.copy_buffer(CopyBufferInfo::buffers(sorted, warm_start)).unwrap();
    }
}
