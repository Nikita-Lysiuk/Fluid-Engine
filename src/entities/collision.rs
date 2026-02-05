use std::sync::Arc;
use glam::Vec3;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use crate::entities::{Actor, ModelVertex};
use crate::renderer::pipelines::Pipelines;
use crate::utils::constants::MAX_FRAMES_IN_FLIGHT;

pub struct CollisionBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl CollisionBox {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn contains(&self, actor: &impl Actor) -> bool {
        actor.location().x >= self.min.x && actor.location().x <= self.max.x &&
            actor.location().y >= self.min.y && actor.location().y <= self.max.y &&
            actor.location().z >= self.min.z && actor.location().z <= self.max.z
    }
}

pub struct CollisionBoxData {
    vertex_buffer: Vec<Subbuffer<[ModelVertex]>>,
    index_buffer: Vec<Subbuffer<[u16]>>,
}

impl CollisionBoxData {
    pub fn new(memory_allocator: Arc<StandardMemoryAllocator>) -> Self {
        let mut vertex_buffer = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut index_buffer = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        let indices: Vec<u16> = vec![
            0, 1, 1, 2, 2, 3, 3, 0,
            4, 5, 5, 6, 6, 7, 7, 4,
            0, 4, 1, 5, 2, 6, 3, 7,
        ];

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let vb = Buffer::new_slice(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..BufferCreateInfo::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..AllocationCreateInfo::default()
                },
                8
            ).map_err(|e| panic!("[CollisionBox] Failed to create vertex buffer:\n{:?}", e)).unwrap();

            let ib = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER,
                    ..BufferCreateInfo::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..AllocationCreateInfo::default()
                },
                indices.iter().cloned()
            ).map_err(|e| panic!("[CollisionBox] Failed to create index buffer:\n{:?}", e)).unwrap();

            vertex_buffer.push(vb);
            index_buffer.push(ib);
        }

        Self {
            vertex_buffer,
            index_buffer,
        }
    }

    pub fn write_to_buffer(&self, collision_box: &CollisionBox, current_frame_idx: usize) {
        let min = collision_box.min;
        let max = collision_box.max;

        let vertices = [
            ModelVertex { position: [min.x, min.y, min.z] },
            ModelVertex { position: [max.x, min.y, min.z] },
            ModelVertex { position: [max.x, max.y, min.z] },
            ModelVertex { position: [min.x, max.y, min.z] },
            ModelVertex { position: [min.x, min.y, max.z] },
            ModelVertex { position: [max.x, min.y, max.z] },
            ModelVertex { position: [max.x, max.y, max.z] },
            ModelVertex { position: [min.x, max.y, max.z] },
        ];

        self.vertex_buffer[current_frame_idx]
            .write()
            .map_err(|e| panic!("[CollisionBox] Failed to write to vertex buffer:\n{:?}", e))
            .unwrap()
            .copy_from_slice(&vertices);

    }

    pub fn index_len(&self, current_frame_idx: usize) -> u32 {
        self.index_buffer[current_frame_idx].len() as u32
    }

    pub fn vertex_buffer_addr(&self, current_frame_idx: usize) -> u64 {
        self.vertex_buffer[current_frame_idx]
            .device_address()
            .map_err(|e| panic!("[CollisionBox] Failed to get vertex buffer device address:\n{:?}", e))
            .unwrap()
            .get()
    }

    pub fn bind_to_command_buffer<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, pipelines: &Pipelines, camera_addr: u64, current_frame_idx: usize) {
        unsafe {
            builder
                .bind_pipeline_graphics(pipelines.collision_pipeline.inner.clone()).unwrap()
                .push_constants(
                    pipelines.common_layout.clone(),
                    0,
                    [
                        camera_addr,
                        self.vertex_buffer_addr(current_frame_idx),
                    ]
                ).unwrap()
                .bind_index_buffer(self.index_buffer[current_frame_idx].clone()).unwrap()
                .draw_indexed(self.index_len(current_frame_idx), 1, 0, 0, 0).unwrap();
        }
    }
}