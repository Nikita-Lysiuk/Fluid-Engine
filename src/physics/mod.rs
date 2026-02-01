use std::cmp::Ordering;
use glam::Vec3;
use crate::core::scene::Scene;
use rayon::prelude::*;
use crate::physics::kernel::CubicKernel;
use crate::physics::neighbor_search::NeighborSearch;
use crate::physics::solver::DFSPHSolver;

mod solver;
mod neighbor_search;
mod kernel;
pub mod fluid_data;

pub struct PhysicsEngine {
    solver: DFSPHSolver,
    neighbor_search: NeighborSearch,
}



impl PhysicsEngine {

    pub fn new(scene: &Scene) -> Self {
        let h = scene.spacing * 2.5;

        Self {
            solver: DFSPHSolver::new(h, Box::new(CubicKernel::new(h))),
            neighbor_search: NeighborSearch::new(h, scene.boundary.min, scene.boundary.max),
        }
    }
    pub fn update(&mut self, scene: &mut Scene, total_dt: f32) {
        self.neighbor_search.build_grid(&scene.fluid_data.vertices);

        scene.fluid_data.reorder_data(&self.neighbor_search.get_sorted_indices());

        self.solver.update_densities_and_factors(scene, &self.neighbor_search);

        self.solver.apply_gravity(scene, total_dt);
        self.solver.apply_viscosity(scene, &self.neighbor_search);

        let mut t = 0.0;
        let mut step = 0;
        let max_steps = 5;

        while t < total_dt && step < max_steps {
            let dt = self.calculate_adaptive_dt(scene, total_dt - t);
            self.solver.solve_density(scene, &self.neighbor_search, dt);
            self.integrate_particles(scene, dt);
            self.solver.solve_divergence(scene, &self.neighbor_search, dt);
            step += 1;
            t += dt;
        }
    }
    fn calculate_adaptive_dt(&self, scene: &Scene, time_left: f32) -> f32 {
        let max_velocity_sq = scene.fluid_data.velocities
            .par_iter()
            .map(|v| v.length_squared())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(0.0);

        let max_v = max_velocity_sq.sqrt();

        let particle_radius = scene.particle_radius;

        let cfl_dt = if max_v > 1e-6 {
            0.4 * (particle_radius / max_v)
        } else {
            0.01
        };

        let min_dt = 0.01;
        let max_dt = 0.2;

        cfl_dt.clamp(min_dt, max_dt).min(time_left)
    }
    fn integrate_particles(&self, scene: &mut Scene, dt: f32) {
        let vertices = &mut scene.fluid_data.vertices;
        let velocities = &mut scene.fluid_data.velocities;
        let boundary = &scene.boundary;

        vertices.par_iter_mut()
            .zip(velocities.par_iter_mut())
            .for_each(|(vert, vel)| {
                let mut pos = Vec3::from_array(vert.position);
                pos += *vel * dt;

                let r = vert.radius;
                let min = boundary.min + r;
                let max = boundary.max - r;

                if pos.x < min.x { pos.x = min.x; vel.x *= -0.5; }
                if pos.x > max.x { pos.x = max.x; vel.x *= -0.5; }

                if pos.y < min.y { pos.y = min.y; vel.y *= -0.5; }
                if pos.y > max.y { pos.y = max.y; vel.y *= -0.5; }

                if pos.z < min.z { pos.z = min.z; vel.z *= -0.5; }
                if pos.z > max.z { pos.z = max.z; vel.z *= -0.5; }

                vert.position = pos.to_array();
            });
    }
}