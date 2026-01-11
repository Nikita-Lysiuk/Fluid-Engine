use vulkano::buffer::BufferContents;

pub mod particle;
pub mod camera;

pub trait Actor {
    type ShaderDataType: BufferContents;
    fn update(&mut self, dt: f32);
    fn build_shader_data(&self) -> Self::ShaderDataType;
}