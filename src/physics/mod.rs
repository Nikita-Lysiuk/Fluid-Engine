use glam::Vec3;
use crate::core::scene::Scene;

pub struct PhysicsEngine {
    gravity: Vec3
}

impl PhysicsEngine {
    pub fn new() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0)
        }
    }

    pub fn update(&self, scene: &mut Scene) {
        for particle in &mut scene.vertices {
            particle.add_acceleration(self.gravity);
        }
    }
}