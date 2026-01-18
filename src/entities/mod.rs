use glam::Vec3;
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;

pub mod particle;
pub mod camera;
pub mod sky;
pub mod collision;

#[derive(BufferContents, Vertex, Debug, Clone, Copy)]
#[repr(C)]
pub struct ModelVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
}

pub trait Actor {
    fn update(&mut self, dt: f32);
    fn location(&self) -> Vec3;
    fn velocity(&self) -> Vec3 {
        Vec3::ZERO
    }
    fn add_input_vector(&mut self, _direction: Vec3, _magnitude: f32) {}
    fn add_rotation(&mut self, _yaw: f32, _pitch: f32) {}
    fn set_velocity(&mut self, _velocity: Vec3) {}
    fn set_position(&mut self, _position: Vec3) {}
}