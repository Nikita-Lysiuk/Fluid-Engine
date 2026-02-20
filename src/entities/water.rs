use std::sync::Arc;
use glam::{Mat4, Quat, Vec3};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::DeviceOwned;
use vulkano::image::sampler::{BorderColor, Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout};
use crate::entities::ModelVertex;
use crate::entities::particle::SimulationParams;

#[derive(BufferContents)]
#[repr(C)]
struct WaterPushConstants {
    camera_addr: u64,
    _pad: u64,
    model: [[f32; 4]; 4],
}

pub struct WaterRenderer {
    vertex_buffer: Subbuffer<[ModelVertex]>,
    index_buffer: Subbuffer<[u32]>,
    index_count: u32,
    descriptor_set: Arc<DescriptorSet>,
}

impl WaterRenderer {
    pub fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        pipeline_layout: Arc<PipelineLayout>,
        density_view: Arc<ImageView>,
        skybox_view: Arc<ImageView>,
        sim_params: &Subbuffer<SimulationParams>,
    ) -> Self {
        let (vertices, indices) = Self::generate_cube();

        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        ).unwrap();

        let index_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices.clone()
        ).unwrap();

        let density_sampler = Sampler::new(
            memory_allocator.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToBorder; 3],
                border_color: BorderColor::IntTransparentBlack,
                ..SamplerCreateInfo::default()
            }
        ).unwrap();

        let skybox_sampler = Sampler::new(
            memory_allocator.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                ..SamplerCreateInfo::default()
            }
        ).unwrap();

        let set_layout = pipeline_layout.set_layouts().get(0).unwrap();
        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator,
            set_layout.clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, density_view, density_sampler),
                WriteDescriptorSet::image_view_sampler(1, skybox_view, skybox_sampler),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
            ],
            []
        ).unwrap();

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            descriptor_set,
        }
    }
    fn generate_cube() -> (Vec<ModelVertex>, Vec<u32>) {
        let vertices = vec![

            ModelVertex { position: [-0.5, -0.5,  0.5] },
            ModelVertex { position: [ 0.5, -0.5,  0.5] },
            ModelVertex { position: [ 0.5,  0.5,  0.5] },
            ModelVertex { position: [-0.5,  0.5,  0.5] },
            ModelVertex { position: [-0.5, -0.5, -0.5] },
            ModelVertex { position: [ 0.5, -0.5, -0.5] },
            ModelVertex { position: [ 0.5,  0.5, -0.5] },
            ModelVertex { position: [-0.5,  0.5, -0.5] },
        ];

        let indices = vec![
            0, 1, 2, 2, 3, 0,
            1, 5, 6, 6, 2, 1,
            5, 4, 7, 7, 6, 5,
            4, 0, 3, 3, 7, 4,
            3, 2, 6, 6, 7, 3,
            4, 5, 1, 1, 0, 4,
        ];

        (vertices, indices)
    }
    pub fn bind_to_command_buffer<Cb>(
        &self,
        builder: &mut AutoCommandBufferBuilder<Cb>,
        pipeline: Arc<GraphicsPipeline>,
        camera_addr: u64,
        box_min: [f32; 4],
        box_max: [f32; 4],
    ) {
        let min = Vec3::from_slice(&box_min[0..3]);
        let max = Vec3::from_slice(&box_max[0..3]);
        let size = max - min;
        let center = min + size * 0.5;

        let model_matrix = Mat4::from_scale_rotation_translation(
            size,
            Quat::IDENTITY,
            center
        );

        let push_data = WaterPushConstants {
            camera_addr,
            _pad: 0,
            model: model_matrix.to_cols_array_2d(),
        };

        unsafe {
            builder
                .bind_pipeline_graphics(pipeline.clone()).unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    self.descriptor_set.clone(),
                ).unwrap()
                .bind_vertex_buffers(0, self.vertex_buffer.clone()).unwrap()
                .bind_index_buffer(self.index_buffer.clone()).unwrap()
                .push_constants(pipeline.layout().clone(), 0, push_data).unwrap()
                .draw_indexed(self.index_count, 1, 0, 0, 0).unwrap();
        }
    }
}