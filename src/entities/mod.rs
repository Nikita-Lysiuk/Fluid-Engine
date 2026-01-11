use glam::Vec3;
use vulkano::buffer::BufferContents;

pub mod particle;
pub mod camera;

pub trait Actor {
    type ShaderDataType: BufferContents;
    fn update(&mut self, dt: f32);
    fn build_shader_data(&self) -> Self::ShaderDataType;
    fn add_input_vector(&mut self, _direction: Vec3, _magnitude: f32) {}
    fn add_rotation(&mut self, _yaw: f32, _pitch: f32) {}
}