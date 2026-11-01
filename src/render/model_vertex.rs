use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;

/// Position of a mesh vertex.
#[derive(BufferContents, Vertex, Debug, Clone, Copy)]
#[repr(C)]
pub struct ModelVertex {
    #[format(R32G32B32_SFLOAT)]
    /// Vertex position.
    pub position: [f32; 3],
}
