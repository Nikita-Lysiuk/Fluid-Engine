use glam::Vec3;
use crate::entities::Actor;
use crate::entities::camera::{Camera, ShaderData};
use crate::entities::particle::{Particle, ParticleVertex};

pub struct Scene {
    pub vertices: Vec<Particle>,
    pub camera: Camera
}

impl Scene {
    pub fn new() -> Self {
        let camera = Camera::new(Vec3::new(0.0, 0.0, -5.0));
        Self {
            vertices: vec![
                Particle::new(
                    Vec3::new(0.0, 0.0, 0.0),
                    0.3,
                    Vec3::new(0.2, 0.5, 1.0),
                ),
            ],
            camera
        }
    }
    pub fn update(&mut self, dt: f32) {
        for vertex in &mut self.vertices {
            vertex.update(dt);
        }
        self.camera.update(dt);
    }
    pub fn get_particle_data(&self) -> Vec<ParticleVertex> {
        self.vertices.iter()
            .map(|p| p.build_shader_data())
            .collect()
    }
    pub fn get_camera_data(&self) -> ShaderData {
        self.camera.build_shader_data()
    }
}