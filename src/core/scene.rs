use glam::Vec3;
use crate::entities::camera::Camera;
use crate::entities::collision::CollisionBox;
use crate::entities::particle::{Particle};
use crate::utils::constants::MAX_PARTICLES;

pub struct Scene {
    pub vertices: Vec<Particle>,
    pub camera: Camera,
    pub boundary: CollisionBox,
    pub smoothing_length: f32,
}

impl Scene {
    pub fn new() -> Self {
        let (vertices, avg_spacing) = Particle::new_with_count(
            MAX_PARTICLES,
            Vec3::new(-1.5, -1.0, -1.5),
            Vec3::new(1.5, 1.0, 1.5)
        );

        let smoothing_length = avg_spacing * 1.3;

        let mut camera = Camera::new(Vec3::new(0.0, 5.0, -25.0));
        camera.rotate(30.0, 0.0, 0.0);
        let boundary = CollisionBox::new(
            Vec3::new(-5.0, -4.0, -5.0),
            Vec3::new(5.0, 4.0, 5.0)
        );

        Self {
            vertices,
            camera,
            boundary,
            smoothing_length,
        }
    }
}