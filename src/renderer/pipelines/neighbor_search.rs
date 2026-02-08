use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::Device;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::shader::EntryPoint;
use crate::entities::particle::{GpuPhysicsData, SimulationParams};
use crate::renderer::pipelines::ComputeStep;
use crate::renderer::pipelines::sorter::GpuSorter;
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
}

impl ComputeStep for NeighborSearch {
    fn load_shader_module(_device: Arc<Device>) -> EntryPoint {
        unimplemented!("NeighborSearch uses multiple shaders, so load_shader_module is not implemented")
    }
    fn from_pipeline(_pipeline: Arc<ComputePipeline>) -> Self {
        unimplemented!("NeighborSearch uses multiple pipelines, so from_pipeline is not implemented")
    }
    fn new(device: Arc<Device>) -> Self {
        let sorter = GpuSorter::new(device.clone());

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
        }
    }
    fn execute<Cb>(
        &self, builder: &mut AutoCommandBufferBuilder<Cb>, 
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &GpuPhysicsData, 
        sim_params: &Subbuffer<SimulationParams>,
    ) { 
        let num_particles = physics_data.count;
        let sort_buffer_len = physics_data.grid_entries.len() as u32;

        {
            let layout = self.spatial_hash_pipeline.layout().set_layouts().get(0).unwrap();
            let set = DescriptorSet::new(
                allocator.clone(),
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, physics_data.position_a.clone()),
                    WriteDescriptorSet::buffer(1, physics_data.grid_entries.clone()),
                    WriteDescriptorSet::buffer(2, sim_params.clone())
                ],
                [],
            ).unwrap();

            let pc = cs_hash::PushConstants {
                num_particles,
                table_size: sort_buffer_len,
            };

            builder
                .bind_pipeline_compute(self.spatial_hash_pipeline.clone()).unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, self.spatial_hash_pipeline.layout().clone(), 0, set)
                .unwrap()
                .push_constants(self.spatial_hash_pipeline.layout().clone(), 0, pc).unwrap();
            
            let group_size = 256;
            let dispatch_count = (sort_buffer_len + group_size - 1) / group_size;
            unsafe { builder.dispatch([dispatch_count, 1, 1]).unwrap(); }
        }

       
        self.sorter.execute(builder, allocator.clone(), physics_data, sim_params);
        
        builder.fill_buffer(physics_data.grid_start.clone(), 0xFFFFFFFF).unwrap();

        {
            let layout = self.offsets_pipeline.layout().set_layouts().get(0).unwrap();
            let set = DescriptorSet::new(
                allocator.clone(),
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, physics_data.grid_entries.clone()),
                    WriteDescriptorSet::buffer(1, physics_data.grid_start.clone()),
                ],
                [],
            ).unwrap();

            let pc = cs_offsets::PushConstants {
                num_entries: sort_buffer_len,
            };

            builder
                .bind_pipeline_compute(self.offsets_pipeline.clone()).unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, self.offsets_pipeline.layout().clone(), 0, set)
                .unwrap()
                .push_constants(self.offsets_pipeline.layout().clone(), 0, pc).unwrap();

            let group_size = 256;
            let dispatch_count = (sort_buffer_len + group_size - 1) / group_size;
            unsafe { builder.dispatch([dispatch_count, 1, 1]).unwrap(); }
        }
        
        {
            let layout = self.reorder_pipeline.layout().set_layouts().get(0).unwrap();
            let set = DescriptorSet::new(
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
            ).unwrap();

            let pc = cs_reorder::PushConstants {
                num_particles,
            };

            builder
                .bind_pipeline_compute(self.reorder_pipeline.clone()).unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, self.reorder_pipeline.layout().clone(), 0, set)
                .unwrap()
                .push_constants(self.reorder_pipeline.layout().clone(), 0, pc).unwrap();

            let group_size = 256;
            let dispatch_count = (num_particles + group_size - 1) / group_size;
            unsafe { builder.dispatch([dispatch_count, 1, 1]).unwrap(); }
        }
    }
}