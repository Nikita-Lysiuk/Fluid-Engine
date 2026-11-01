use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;

pub mod particle;
pub use crate::camera;
pub mod sky;
pub mod collision;
pub mod water;

#[derive(BufferContents, Vertex, Debug, Clone, Copy)]
#[repr(C)]
pub struct ModelVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
}

