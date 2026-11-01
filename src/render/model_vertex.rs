use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;

#[derive(BufferContents, Vertex, Debug, Clone, Copy)]
#[repr(C)]
pub struct ModelVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
}
