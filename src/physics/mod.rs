use glam::Vec3;
use crate::core::scene::Scene;
use rayon::prelude::*;
use crate::physics::kernel::CubicKernel;
use crate::physics::neighbor_search::NeighborSearch;
use crate::physics::solver::DFSPHSolver;

mod solver;
mod neighbor_search;
mod kernel;
mod integration;
pub mod fluid_data;

pub struct PhysicsEngine {
    gravity: Vec3,
    damping: f32,
    solver: DFSPHSolver,
    neighbor_search: NeighborSearch,
}



impl PhysicsEngine {

    pub fn new(scene: &Scene) -> Self {
        let h = scene.spacing * 2.0;

        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.5,
            solver: DFSPHSolver::new(h, Box::new(CubicKernel::new(h))),
            neighbor_search: NeighborSearch::new(h, scene.boundary.min, scene.boundary.max),
        }
    }

    pub fn update(&mut self, scene: &mut Scene, dt: f32) {
        self.neighbor_search.build_grid(&scene.fluid_data.vertices);

        //scene.fluid_data.reorder_data(&self.neighbor_search.get_sorted_indices());

        //self.solver.update_densities_and_factors(scene, &self.neighbor_search);

    }

}