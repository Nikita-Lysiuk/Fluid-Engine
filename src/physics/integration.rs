use glam::Vec3;
use crate::entities::Actor;
use crate::entities::particle::Particle;
use rayon::prelude::*;

pub struct Integrator {
    particle_diameter: f32,
}

impl Integrator {
    pub fn new(particle_diameter: f32) -> Self {
        Self { particle_diameter  }
    }
    pub fn clf_dt(&self, particles: &[Particle], time_left: f32, min_sanity_step: f32) -> f32 {
        let max_v_sq = particles
            .par_iter()
            .map(|p| p.velocity().length_squared())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let max_v = max_v_sq.sqrt();

        if max_v > 1e-6 {
            let cfl_limit = (self.particle_diameter / max_v) * 0.4;
            cfl_limit.min(time_left).max(min_sanity_step)
        } else {
            time_left
        }
    }
    pub fn predict_velocity(&self, particle: &mut Particle, f: Vec3, gravity: Vec3, dt: f32) {
        let new_velocity = particle.velocity() + (f / particle.mass) * dt + gravity * dt;
        particle.set_velocity(new_velocity);
    }
}