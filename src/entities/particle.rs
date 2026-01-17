use std::cell::Cell;
use std::sync::Arc;
use glam::Vec3;
use vulkano::buffer::{BufferContents, Subbuffer, Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use crate::entities::Actor;
use crate::renderer::pipelines::Pipelines;
use crate::utils::constants::{MAX_FRAMES_IN_FLIGHT, MAX_PARTICLES};

#[derive(BufferContents, Vertex, Debug, Clone, Copy)]
#[repr(C)]
pub struct ParticleVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],

    #[format(R32_SFLOAT)]
    pub radius: f32,

    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
}

pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub acceleration: Vec3,
    pub color: Vec3,
    pub radius: f32,
    pub mass: f32,
    _padding: f32,
}

impl Actor for Particle {
    fn update(&mut self, dt: f32) {
        self.velocity += self.acceleration * dt;
        self.position += self.velocity * dt;

        self.acceleration = Vec3::ZERO;
    }
}

impl Particle {
    pub fn new(position: Vec3, color: Vec3, radius: f32, mass: f32) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            color,
            radius,
            mass,
            _padding: 0.0,
        }
    }

    pub fn add_acceleration(&mut self, acceleration: Vec3) {
        self.acceleration += acceleration / self.mass;
    }
}

pub struct ParticleData {
    pub vertex_buffer: Vec<Subbuffer<[ParticleVertex]>>,
    pub vertices_count: [Cell<u32>; MAX_FRAMES_IN_FLIGHT],
}

impl ParticleData {
    pub fn new(memory_allocator: Arc<StandardMemoryAllocator>) -> Self {
        let mut vertex_buffer = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let vb = Buffer::new_slice(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER
                        | BufferUsage::SHADER_DEVICE_ADDRESS
                        | BufferUsage::STORAGE_BUFFER
                        | BufferUsage::TRANSFER_DST,
                    ..BufferCreateInfo::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..AllocationCreateInfo::default()
                },
                MAX_PARTICLES
            ).map_err(|e| {
                panic!("[Particle Data] Failed to create vertex buffer:\n{:?}", e);
            }).unwrap();

            vertex_buffer.push(vb);
        }

        Self {
            vertex_buffer,
            vertices_count: [const { Cell::new(0) }; MAX_FRAMES_IN_FLIGHT],
        }
    }
    pub fn write_to_buffer(&self, particles: &[Particle], current_frame_idx: usize) {
        let particle_vertices: Vec<ParticleVertex> = particles.iter().map(|p| {
            ParticleVertex {
                position: p.position.to_array(),
                radius: p.radius,
                color: p.color.to_array(),
            }
        }).collect();

        let num_particles = particle_vertices.len();
        if num_particles == 0 { return; }

        let mut write_lock = self.vertex_buffer[current_frame_idx]
            .write()
            .map_err(|e| panic!("[Particle Data] Failed to write to vertex buffer:\n{:?}", e))
            .unwrap();

        write_lock[..num_particles].copy_from_slice(&particle_vertices);

        self.vertices_count[current_frame_idx].set(particle_vertices.len() as u32);
    }
    fn vertex_buffer_addr(&self, current_frame_idx: usize) -> u64 {
        self.vertex_buffer[current_frame_idx]
            .device_address()
            .map_err(|e| panic!("[Particle Data] Failed to get vertex buffer device address:\n{:?}", e))
            .unwrap()
            .get()
    }
    pub fn bind_to_command_buffer<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, pipelines: &Pipelines, camera_addr: u64, current_frame_idx: usize) {
        unsafe {
            builder.bind_pipeline_graphics(pipelines.point_pipeline.inner.clone()).map_err(|e| panic!("[Renderer] Failed to bind point pipeline: {:?}", e)).unwrap()
                .push_constants(
                    pipelines.common_layout.clone(),
                    0,
                    [
                        camera_addr,
                        self.vertex_buffer_addr(current_frame_idx),
                    ]
                ).map_err(|e| panic!("[Renderer] Failed to bind buffers: {:?}", e)).unwrap()
                .draw(self.vertices_count[current_frame_idx].get(), 1, 0, 0).map_err(|e| panic!("[Renderer] Failed to draw particles: {:?}", e)).unwrap();
        }
    }
}