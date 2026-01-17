use std::sync::Arc;
use glam::Vec3;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use crate::entities::ModelVertex;
use crate::entities::particle::Particle;
use crate::utils::constants::MAX_FRAMES_IN_FLIGHT;

pub struct CollisionBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl CollisionBox {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn contains(&self, point: Particle) -> bool {
        point.position.x >= self.min.x && point.position.x <= self.max.x &&
        point.position.y >= self.min.y && point.position.y <= self.max.y &&
        point.position.z >= self.min.z && point.position.z <= self.max.z
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

            let ib = Buffer::new_slice(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER,
                    ..BufferCreateInfo::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..AllocationCreateInfo::default()
                },
                24
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

        let vertices = vec![
            ModelVertex { position: [min.x, min.y, min.z] },
            ModelVertex { position: [max.x, min.y, min.z] },
            ModelVertex { position: [max.x, max.y, min.z] },
            ModelVertex { position: [min.x, max.y, min.z] },
            ModelVertex { position: [min.x, min.y, max.z] },
            ModelVertex { position: [max.x, min.y, max.z] },
            ModelVertex { position: [max.x, max.y, max.z] },
            ModelVertex { position: [min.x, max.y, max.z] },
        ];

        let indices: Vec<u16> = vec![
            0, 1, 1, 2, 2, 3, 3, 0,
            4, 5, 5, 6, 6, 7, 7, 4,
            0, 4, 1, 5, 2, 6, 3, 7,
        ];

        self.vertex_buffer[current_frame_idx]
            .write().map_err(|e| panic!("[CollisionBox] Failed to write to vertex buffer:\n{:?}", e))
            .unwrap()
            .copy_from_slice(&vertices);

        self.index_buffer[current_frame_idx]
            .write().map_err(|e| panic!("[CollisionBox] Failed to write to index buffer:\n{:?}", e))
            .unwrap()
            .copy_from_slice(&indices);
    }
    pub fn index_len(&self) -> u32 {
        self.index_buffer.len() as u32
    }
    pub fn vertex_buffer_addr(&self, current_frame_idx: usize) -> u64 {
        self.vertex_buffer[current_frame_idx]
            .device_address()
            .map_err(|e| panic!("[CollisionBox] Failed to get vertex buffer device address:\n{:?}", e))
            .unwrap()
            .get()
    }
    pub fn index_buffer_addr(&self, current_frame_idx: usize) -> u64 {
        self.index_buffer[current_frame_idx]
            .device_address()
            .map_err(|e| panic!("[CollisionBox] Failed to get index buffer device address:\n{:?}", e))
            .unwrap()
            .get()
    }
}