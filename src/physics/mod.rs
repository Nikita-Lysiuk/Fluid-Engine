use glam::Vec3;
use crate::core::scene::Scene;
use crate::entities::Actor;
use crate::entities::particle::Particle;
use rayon::prelude::*;
use crate::physics::kernel::WendlandKernel;
use crate::physics::neighbor_search::NeighborSearch;
use crate::physics::solver::DFSPHSolver;
use crate::utils::constants::MAX_PARTICLES;

mod solver;
mod neighbor_search;
mod kernel;
mod integration;

pub struct PhysicsEngine {
    gravity: Vec3,
    damping: f32,
    solver: DFSPHSolver,
    neighbor_search: NeighborSearch,
}

impl PhysicsEngine {
    pub fn new(h: f32) -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.99,
            solver: DFSPHSolver::new(h, Box::new(WendlandKernel::new(h, 3))),
            neighbor_search: NeighborSearch::new(h, MAX_PARTICLES),
        }
    }
    pub fn update(&mut self, scene: &mut Scene, dt: f32) {

        let ns = &self.neighbor_search;
        scene.vertices.par_sort_by_key(|p| {
            let cell = ns.pos_to_cell(p.location());
            ns.hash(cell)
        });

        let positions: Vec<Vec3> = scene.vertices.iter().map(|p| p.location()).collect();
        let masses: Vec<f32> = scene.vertices.iter().map(|p| p.mass).collect();

        self.neighbor_search.build(&positions);

        scene.vertices.par_iter_mut().enumerate().for_each(|(i, p)| {
            let (density, factor) = self.solver.compute_density_and_factor(
                positions[i],
                &self.neighbor_search,
                &positions,
                &masses
            );
            p.density = density;
            p.alpha = factor;
        });

    }
    pub fn check_collision(&self, p: &mut Particle, box_min: Vec3, box_max: Vec3) {
        if p.location().x - p.radius < box_min.x {
            p.set_position(Vec3::new(box_min.x + p.radius, p.location().y, p.location().z));
            p.set_velocity(Vec3::new(-p.velocity().x * self.damping, p.velocity().y, p.velocity().z));
        } else if p.location().x + p.radius > box_max.x {
            p.set_position(Vec3::new(box_max.x - p.radius, p.location().y, p.location().z));
            p.set_velocity(Vec3::new(-p.velocity().x * self.damping, p.velocity().y, p.velocity().z));
        }

        if p.location().y - p.radius < box_min.y {
            p.set_position(Vec3::new(p.location().x, box_min.y + p.radius, p.location().z));
            p.set_velocity(Vec3::new(p.velocity().x, -p.velocity().y * self.damping, p.velocity().z));
        } else if p.location().y + p.radius > box_max.y {
            p.set_position(Vec3::new(p.location().x, box_max.y - p.radius, p.location().z));
            p.set_velocity(Vec3::new(p.velocity().x, -p.velocity().y * self.damping, p.velocity().z));
        }

        if p.location().z - p.radius < box_min.z {
            p.set_position(Vec3::new(p.location().x, p.location().y, box_min.z + p.radius));
            p.set_velocity(Vec3::new(p.velocity().x, p.velocity().y, -p.velocity().z * self.damping));
        } else if p.location().z + p.radius > box_max.z {
            p.set_position(Vec3::new(p.location().x, p.location().y, box_max.z - p.radius));
            p.set_velocity(Vec3::new(p.velocity().x, p.velocity().y, -p.velocity().z * self.damping));
        }
    }
}