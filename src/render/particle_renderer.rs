use crate::constants::MAX_FRAMES_IN_FLIGHT;
use crate::render::graphics_pipelines::Pipelines;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::graphics::vertex_input::Vertex;

/// Particle position supplied to the graphics pipeline.
#[repr(C)]
#[derive(BufferContents, Vertex, Copy, Clone, Debug, Default)]
pub struct PositionVertex {
    #[format(R32G32B32A32_SFLOAT)]
    /// Vertex position.
    pub position: [f32; 4],
}

/// Particle color supplied to the graphics pipeline.
#[repr(C)]
#[derive(BufferContents, Vertex, Copy, Clone, Debug, Default)]
pub struct ColorVertex {
    #[format(R32G32B32A32_SFLOAT)]
    /// Particle color.
    pub color: [f32; 4],
}

/// Particle rendering attributes.
#[repr(C)]
#[derive(BufferContents, Vertex, Copy, Clone, Debug, Default)]
pub struct AttributeVertex {
    #[format(R32_SFLOAT)]
    /// Particle radius.
    pub radius: f32,
}

/// GPU buffers used to draw particles.
pub struct GpuRenderData {
    /// Per-frame particle position buffers.
    pub position_buffers: Vec<Subbuffer<[PositionVertex]>>,
    /// Per-frame particle color buffers.
    pub color_buffers: Vec<Subbuffer<[ColorVertex]>>,
    /// Particle attribute buffer.
    pub attribute_buffer: Subbuffer<[AttributeVertex]>,
}

impl GpuRenderData {
    /// Creates the associated resources.
    pub fn new(
        allocator: Arc<StandardMemoryAllocator>,
        initial_positions: &[[f32; 3]],
        radius: f32,
    ) -> Self {
        let particle_count = initial_positions.len();

        let mut position_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut color_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let buffer = Buffer::from_iter(
                allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                initial_positions.iter().map(|p| PositionVertex {
                    position: [p[0], p[1], p[2], 1.0],
                }),
            )
            .expect("Failed to create render position buffer");
            position_buffers.push(buffer);

            let color_buffer = Buffer::from_iter(
                allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                (0..particle_count).map(|_| ColorVertex {
                    color: [0.0, 0.5, 1.0, 1.0],
                }),
            )
            .expect("Failed to create render color buffer");
            color_buffers.push(color_buffer);
        }

        let attribute_buffer = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (0..particle_count).map(|_| AttributeVertex { radius }),
        )
        .expect("Failed to create attribute buffer");

        Self {
            position_buffers,
            attribute_buffer,
            color_buffers,
        }
    }
    /// Records the graphics commands for this resource.
    pub fn bind_to_command_buffer<Cb>(
        &self,
        builder: &mut AutoCommandBufferBuilder<Cb>,
        pipelines: &Pipelines,
        camera_addr: u64,
        frame_idx: usize,
        count: u32,
    ) {
        // SAFETY: the pipeline, required descriptors, vertex buffers, and push constants are bound
        //         before drawing.
        unsafe {
            builder
                .bind_pipeline_graphics(pipelines.point_pipeline.inner.clone())
                .expect("graphics pipeline can be bound")
                .bind_vertex_buffers(
                    0,
                    (
                        self.position_buffers[frame_idx].clone(),
                        self.color_buffers[frame_idx].clone(),
                        self.attribute_buffer.clone(),
                    ),
                )
                .expect("vertex buffers can be bound")
                .push_constants(
                    pipelines.point_pipeline.inner.layout().clone(),
                    0,
                    camera_addr,
                )
                .expect("push constants match the shader")
                .draw(count, 1, 0, 0)
                .expect("Failed to bind particle vertex buffers");
        }
    }
}
