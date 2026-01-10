use glam::Vec3;
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;

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

impl ParticleVertex {
    pub fn new(position: Vec3, radius: f32, color: Vec3) -> Self {
        Self {
            position: position.to_array(),
            radius,
            color: color.to_array(),
        }
    }
}