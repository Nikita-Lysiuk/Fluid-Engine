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
        let camera = Camera {
            position: Vec3::new(0.0, 0.0, 5.0),
            velocity: Vec3::ZERO,
            orientation: glam::Quat::IDENTITY,
            fov: 60.0,
            aspect_ratio: 16.0 / 9.0,
            near: 0.1,
            far: 100.0,
        };
        Self {
            vertices: vec![
                Particle::new(
                    Vec3::new(0.0, 0.0, 0.0),
                    0.3,
                    Vec3::new(0.2, 0.5, 1.0),
                ),
                Particle::new(
                    Vec3::new(1.0, 1.0, 0.0),
                    0.3,
                    Vec3::new(1.0, 0.3, 0.3),
                ),
                Particle::new(
                    Vec3::new(0.0, 0.1, -0.5),
                    0.8,
                    Vec3::new(0.3, 1.0, 0.3),
                ),
            ],
            camera
        }
    }
    pub fn update(&mut self, dt: f32) {
        // Update scene logic here (e.g., animate particles)
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