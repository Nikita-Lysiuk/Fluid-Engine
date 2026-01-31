use glam::Vec3;
use crate::entities::camera::Camera;
use crate::entities::collision::CollisionBox;
use crate::entities::particle::{ParticleGenerator, ParticleState, ParticleVertex};
use crate::utils::constants::MAX_PARTICLES;

pub struct Scene {
    pub rendering_data: Vec<ParticleVertex>,
    pub physics_data: Vec<ParticleState>,
    pub camera: Camera,
    pub boundary: CollisionBox,
    pub particle_radius: f32,
}

impl Scene {
    pub fn new() -> Self {
        let particle_radius = 0.05;
        let (rendering_data, physics_data, spacing) = ParticleGenerator::generate(
            MAX_PARTICLES,
            particle_radius * 2.0,
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 5.0, 1.0)
        );

        let mut camera = Camera::new(Vec3::new(0.0, 5.0, -10.0));
        camera.rotate(30.0, 0.0, 0.0);
        let boundary = CollisionBox::new(Vec3::new(-2.0, -1.0, -2.0), Vec3::new(2.0, 5.0, 2.0));

        Self {
            rendering_data,
            physics_data,
            camera,
            boundary,
            particle_radius,
        }
    }
}