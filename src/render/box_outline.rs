use crate::constants::MAX_FRAMES_IN_FLIGHT;
use crate::render::graphics_pipelines::Pipelines;
use crate::render::model_vertex::ModelVertex;
use crate::sim::collision_box::CollisionBox;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};

/// GPU buffers for the simulation boundary outline.
pub struct CollisionBoxData {
    vertex_buffer: Vec<Subbuffer<[ModelVertex]>>,
    index_buffer: Vec<Subbuffer<[u16]>>,
}

impl CollisionBoxData {
    /// Creates the associated resources.
    pub fn new(memory_allocator: Arc<StandardMemoryAllocator>) -> Self {
        let mut vertex_buffer = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut index_buffer = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        let indices: Vec<u16> = vec![
            0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7,
        ];

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let vb = Buffer::new_slice(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..BufferCreateInfo::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..AllocationCreateInfo::default()
                },
                8,
            )
            .unwrap_or_else(|e| panic!("[CollisionBox] Failed to create vertex buffer:\n{:?}", e));

            let ib = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER,
                    ..BufferCreateInfo::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..AllocationCreateInfo::default()
                },
                indices.iter().cloned(),
            )
            .unwrap_or_else(|e| panic!("[CollisionBox] Failed to create index buffer:\n{:?}", e));

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
            ModelVertex {
                position: [min.x, min.y, min.z],
            },
            ModelVertex {
                position: [max.x, min.y, min.z],
            },
            ModelVertex {
                position: [max.x, max.y, min.z],
            },
            ModelVertex {
                position: [min.x, max.y, min.z],
            },
            ModelVertex {
                position: [min.x, min.y, max.z],
            },
            ModelVertex {
                position: [max.x, min.y, max.z],
            },
            ModelVertex {
                position: [max.x, max.y, max.z],
            },
            ModelVertex {
                position: [min.x, max.y, max.z],
            },
        ];

        self.vertex_buffer[current_frame_idx]
            .write()
            .unwrap_or_else(|e| panic!("[CollisionBox] Failed to write to vertex buffer:\n{:?}", e))
            .copy_from_slice(&vertices);
    }

    pub fn index_len(&self, current_frame_idx: usize) -> u32 {
        self.index_buffer[current_frame_idx].len() as u32
    }

    pub fn vertex_buffer_addr(&self, current_frame_idx: usize) -> u64 {
        self.vertex_buffer[current_frame_idx]
            .device_address()
            .unwrap_or_else(|e| {
                panic!(
                    "[CollisionBox] Failed to get vertex buffer device address:\n{:?}",
                    e
                )
            })
            .get()
    }

    /// Records the graphics commands for this resource.
    pub fn bind_to_command_buffer<Cb>(
        &self,
        builder: &mut AutoCommandBufferBuilder<Cb>,
        pipelines: &Pipelines,
        camera_addr: u64,
        current_frame_idx: usize,
    ) {
        // SAFETY: the pipeline, required descriptors, vertex buffers, and push constants are bound
        //         before drawing.
        unsafe {
            builder
                .bind_pipeline_graphics(pipelines.collision_pipeline.inner.clone())
                .expect("graphics pipeline can be bound")
                .push_constants(
                    pipelines.common_layout.clone(),
                    0,
                    [camera_addr, self.vertex_buffer_addr(current_frame_idx)],
                )
                .expect("push constants match the shader")
                .bind_index_buffer(self.index_buffer[current_frame_idx].clone())
                .expect("index buffer can be bound")
                .draw_indexed(self.index_len(current_frame_idx), 1, 0, 0, 0)
                .expect("indexed draw call is valid");
        }
    }
}
