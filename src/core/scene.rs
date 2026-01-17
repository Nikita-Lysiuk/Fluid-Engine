use glam::Vec3;
use crate::entities::Actor;
use crate::entities::camera::Camera;
use crate::entities::collision::CollisionBox;
use crate::entities::particle::{Particle};

pub struct Scene {
    pub vertices: Vec<Particle>,
    pub camera: Camera,
    pub boundary: CollisionBox
}

impl Scene {
    pub fn new() -> Self {
        let vertices = vec![
            Particle::new(
                Vec3::new(0.0, 150.0, 0.0),
                Vec3::new(0.2, 0.5, 1.0),
                15.0,
                2.0
            ),
        ];
        let camera = Camera::new(Vec3::new(0.0, 0.0, -30.0));
        let boundary = CollisionBox::new(
            Vec3::new(-100.0, -100.0, -100.0),
            Vec3::new(100.0, 100.0, 100.0)
        );

        Self {
            vertices,
            camera,
            boundary,
        }
    }
    pub fn update(&mut self, dt: f32) {
        for vertex in &mut self.vertices {
            vertex.update(dt);
        }
        self.camera.update(dt);
    }
}