use glam::Vec3;
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use crate::entities::Actor;

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
    pub radius: f32,
    pub color: Vec3,
}

impl Actor for Particle {
    type ShaderDataType = ParticleVertex;
    fn update(&mut self, dt: f32) {
        self.position += self.velocity * dt;
    }

    fn build_shader_data(&self) -> Self::ShaderDataType {
        ParticleVertex {
            position: self.position.into(),
            radius: self.radius,
            color: self.color.into(),
        }
    }
}

impl Particle {
    pub fn new(position: Vec3, radius: f32, color: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            radius,
            color,
        }
    }
}