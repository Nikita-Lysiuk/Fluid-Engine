use std::cell::Cell;
use std::sync::Arc;
use glam::Vec3;
use rayon::prelude::*;
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

#[derive(Clone, Copy, Debug)]
#[repr(C)]
#[repr(align(16))]
pub struct Particle {
    position: Vec3,
    velocity: Vec3,
    color: Vec3,
    pub radius: f32,
    pub mass: f32,
    pub density: f32,
    pub alpha: f32,
    _padding: [f32; 3]
}

impl Actor for Particle {
    fn update(&mut self, dt: f32) {
        self.position += self.velocity * dt;
    }
    fn location(&self) -> Vec3 {
        self.position
    }
    fn velocity(&self) -> Vec3 {
        self.velocity
    }
    fn set_velocity(&mut self, _velocity: Vec3) {
        self.velocity = _velocity;
    }
    fn set_position(&mut self, _position: Vec3) {
        self.position = _position;
    }
}

impl Particle {
    pub fn new(position: Vec3, color: Vec3, radius: f32, mass: f32) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            color,
            radius,
            mass,
            density: 0.0,
            alpha: 0.0,
            _padding: [0.0; 3],
        }
    }
    pub fn new_with_count(count: usize, min: Vec3, max: Vec3) -> (Vec<Self>, f32) {
        let size = max - min;

        let volume = (size.x * size.y * size.z).max(0.000001);

        let k = (count as f32 / volume).powf(1.0 / 3.0);

        let n = Vec3::new(
            (size.x * k).ceil(),
            (size.y * k).ceil(),
            (size.z * k).ceil(),
        );

        let spacing = Vec3::new(
            size.x / n.x,
            size.y / n.y,
            size.z / n.z,
        );

        let avg_spacing = (spacing.x + spacing.y + spacing.z) / 3.0;

        let mut particles = Vec::with_capacity(count);

        let offset = spacing / 2.0;

        let mut spawned = 0;

        'outer: for z in 0..n.z as usize {
            for y in 0..n.y as usize {
                for x in 0..n.x as usize {
                    if spawned >= count {
                        break 'outer;
                    }

                    let pos = Vec3::new(
                        min.x + x as f32 * spacing.x + offset.x,
                        min.y + y as f32 * spacing.y + offset.y,
                        min.z + z as f32 * spacing.z + offset.z,
                    );

                    particles.push(Particle::new(
                        pos,
                        Vec3::new(0.4, 0.7, 1.0),
                        0.1,
                        1.0,
                    ));
                    spawned += 1;
                }
            }
        }

        (particles, avg_spacing)
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
                MAX_PARTICLES as u64,
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
        let num_particles = particles.len();
        if num_particles == 0 { return; }

        let mut write_lock = self.vertex_buffer[current_frame_idx]
            .write()
            .map_err(|e| panic!("[Particle Data] Failed to write to vertex buffer:\n{:?}", e))
            .unwrap();

        let dst_slice = &mut write_lock[0..num_particles];
        dst_slice.par_iter_mut().zip(particles.par_iter()).for_each(|(v, p)| {
            *v = ParticleVertex {
                position: p.location().to_array(),
                radius: p.radius,
                color: p.color.to_array(),
            }
        });
        self.vertices_count[current_frame_idx].set(num_particles as u32);
    }
    fn vertex_buffer_addr(&self, current_frame_idx: usize) -> u64 {
        self.vertex_buffer[current_frame_idx]
            .device_address()
            .map_err(|e| panic!("[Particle Data] Failed to get vertex buffer device address:\n{:?}", e))
            .unwrap()
            .get()
    }
    pub fn bind_to_command_buffer<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, pipelines: &Pipelines, camera_addr: u64, current_frame_idx: usize) {
        let _span = tracy_client::span!("Bind Particle Data to Command Buffer");
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