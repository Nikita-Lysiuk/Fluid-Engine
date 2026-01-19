mod solver;
mod neighbor_search;
mod kernel;
mod integration;

use glam::Vec3;
use crate::core::scene::Scene;
use crate::entities::Actor;
use crate::entities::particle::Particle;
use rayon::prelude::*;
use crate::physics::kernel::{ SmoothingKernel, WendlandKernel };

pub struct PhysicsEngine {
    gravity: Vec3,
    damping: f32,
    smoothing_kernel: Box<dyn SmoothingKernel + Send + Sync>,
}

impl PhysicsEngine {
    pub fn new(h: f32) -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            smoothing_kernel: Box::new(WendlandKernel::new(h, 3)),
            damping: 0.99,
        }
    }

    pub fn update(&self, scene: &mut Scene, dt: f32) {
        let b = &scene.boundary;


        scene.vertices.par_iter_mut().for_each(|p| {

            p.add_acceleration(self.gravity);

            p.update(dt);

            self.check_collision(p, b.min, b.max);
        });
    }

    pub fn check_collision(&self, p: &mut Particle, box_min: Vec3, box_max: Vec3) {
        if p.location().x - p.radius() < box_min.x {
            p.set_position(Vec3::new(box_min.x + p.radius(), p.location().y, p.location().z));
            p.set_velocity(Vec3::new(-p.velocity().x * self.damping, p.velocity().y, p.velocity().z));
        } else if p.location().x + p.radius() > box_max.x {
            p.set_position(Vec3::new(box_max.x - p.radius(), p.location().y, p.location().z));
            p.set_velocity(Vec3::new(-p.velocity().x * self.damping, p.velocity().y, p.velocity().z));
        }

        if p.location().y - p.radius() < box_min.y {
            p.set_position(Vec3::new(p.location().x, box_min.y + p.radius(), p.location().z));
            p.set_velocity(Vec3::new(p.velocity().x, -p.velocity().y * self.damping, p.velocity().z));
        } else if p.location().y + p.radius() > box_max.y {
            p.set_position(Vec3::new(p.location().x, box_max.y - p.radius(), p.location().z));
            p.set_velocity(Vec3::new(p.velocity().x, -p.velocity().y * self.damping, p.velocity().z));
        }

        if p.location().z - p.radius() < box_min.z {
            p.set_position(Vec3::new(p.location().x, p.location().y, box_min.z + p.radius()));
            p.set_velocity(Vec3::new(p.velocity().x, p.velocity().y, -p.velocity().z * self.damping));
        } else if p.location().z + p.radius() > box_max.z {
            p.set_position(Vec3::new(p.location().x, p.location().y, box_max.z - p.radius()));
            p.set_velocity(Vec3::new(p.velocity().x, p.velocity().y, -p.velocity().z * self.damping));
        }
    }
}