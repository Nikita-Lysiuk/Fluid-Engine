use glam::Vec3;
use crate::entities::vertex::ParticleVertex;

pub struct Scene {
    pub vertices: Vec<ParticleVertex>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            vertices: vec![
                ParticleVertex::new(
                    Vec3::new(0.0, 0.0, 0.0),
                    0.5,
                    Vec3::new(0.2, 0.5, 1.0),
                )
            ]
        }
    }
    
    pub fn update(&mut self, dt: f32) {
        // Update scene logic here (e.g., animate particles)
    }
}