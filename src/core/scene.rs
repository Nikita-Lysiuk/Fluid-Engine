use glam::Vec3;
use crate::entities::camera::Camera;
use crate::entities::collision::CollisionBox;
use crate::entities::particle::{Particle};

pub struct Scene {
    pub vertices: Vec<Particle>,
    pub camera: Camera,
    pub boundary: CollisionBox,
    pub smoothing_length: f32,
}

impl Scene {
    pub fn new() -> Self {
        let (vertices, avg_spacing) = Particle::new_with_count(
            1000,
            Vec3::new(-10.0, -5.0, -10.0),
            Vec3::new(10.0, 5.0, 10.0)
        );

        let smoothing_length = avg_spacing * 2.6;

        let mut camera = Camera::new(Vec3::new(0.0, 5.0, -25.0));
        camera.rotate(30.0, 0.0, 0.0);
        let boundary = CollisionBox::new(
            Vec3::new(-10.0, -5.0, -10.0),
            Vec3::new(10.0, 5.0, 10.0)
        );

        Self {
            vertices,
            camera,
            boundary,
            smoothing_length,
        }
    }
}