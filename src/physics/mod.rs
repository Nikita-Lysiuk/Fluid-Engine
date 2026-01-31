// use glam::Vec3;
// use crate::core::scene::Scene;
// use crate::entities::Actor;
// use crate::entities::particle::Particle;
// use rayon::prelude::*;
// use crate::physics::integration::Integrator;
// use crate::physics::kernel::{CubicSplineKernel, WendlandKernel};
// use crate::physics::neighbor_search::NeighborSearch;
// use crate::physics::solver::DFSPHSolver;
// use crate::utils::constants::MAX_PARTICLES;
// 
// 
// 
// mod solver;
// mod neighbor_search;
// mod kernel;
// mod integration;
// 
// 
// 
// pub struct PhysicsEngine {
//     gravity: Vec3,
//     damping: f32,
//     solver: DFSPHSolver,
//     neighbor_search: NeighborSearch,
//     integrator: Integrator,
// 
//     buf_positions: Vec<Vec3>,
//     buf_velocities: Vec<Vec3>,
//     buf_densities: Vec<f32>,
//     buf_masses: Vec<f32>,
//     buf_alphas: Vec<f32>,
// 
// 
// 
//     buf_kappas: Vec<f32>,
//     buf_delta_v: Vec<Vec3>,
// }
// 
// 
// 
// impl PhysicsEngine {
// 
//     pub fn new(h: f32, d: f32) -> Self {
// 
//         let cap = MAX_PARTICLES;
// 
//         Self {
//             gravity: Vec3::new(0.0, -9.81, 0.0),
//             damping: 0.5,
//             solver: DFSPHSolver::new(h, Box::new(CubicSplineKernel::new(h * 2.0))),
//             neighbor_search: NeighborSearch::new(h, MAX_PARTICLES),
//             integrator: Integrator::new(d),
//             buf_positions: vec![Vec3::ZERO; cap],
//             buf_velocities: vec![Vec3::ZERO; cap],
//             buf_densities: vec![0.0; cap],
//             buf_masses: vec![0.0; cap],
//             buf_alphas: vec![0.0; cap],
//             buf_kappas: vec![0.0; cap],
//             buf_delta_v: vec![Vec3::ZERO; cap],
//         }
//     }
// 
//     pub fn update(&mut self, scene: &mut Scene, dt: f32) {
// 
//         let particle_count = scene.vertices.len();
//         let ns = &self.neighbor_search;
//         scene.vertices.par_sort_by_key(|p| {
//             let cell = ns.pos_to_cell(p.location());
//             ns.hash(cell)
//         });
// 
//         if self.buf_positions.len() < particle_count { self.buf_positions.resize(particle_count, Vec3::ZERO); }
//         if self.buf_masses.len() < particle_count { self.buf_masses.resize(particle_count, 0.0); }
//         if self.buf_velocities.len() < particle_count { self.buf_velocities.resize(particle_count, Vec3::ZERO); }
//         if self.buf_densities.len() < particle_count { self.buf_densities.resize(particle_count, 0.0); }
//         if self.buf_alphas.len() < particle_count { self.buf_alphas.resize(particle_count, 0.0); }
//         if self.buf_kappas.len() < particle_count { self.buf_kappas.resize(particle_count, 0.0); }
//         if self.buf_delta_v.len() < particle_count { self.buf_delta_v.resize(particle_count, Vec3::ZERO); }
// 
// 
//         (&mut self.buf_positions[0..particle_count], &mut self.buf_masses[0..particle_count], &scene.vertices)
//             .into_par_iter()
//             .for_each(|(pos, mass, p)| {
//                 *pos = p.location();
//                 *mass = p.mass;
//             });
//         self.neighbor_search.build(&self.buf_positions[0..particle_count]);
//         (&mut self.buf_densities[0..particle_count], &mut self.buf_alphas[0..particle_count], &mut scene.vertices)
//             .into_par_iter()
//             .enumerate()
//             .for_each(|(i, (dens, alpha, p))| {
//                 let (d, a) = self.solver.compute_density_and_factor(
//                     self.buf_positions[i],
//                     self.buf_masses[i],
//                     &self.neighbor_search,
//                     &self.buf_positions,
//                     &self.buf_masses
//                 );
// 
//                 *dens = d;
//                 *alpha = a;
//                 p.density = d;
//                 p.alpha = a;
//             });
// 
//         let mut dt_remaining = dt;
//         let mut substeps_done = 0;
//         let max_substeps = 2;
// 
//         while dt_remaining > 1e-6 && substeps_done < max_substeps {
//             let substep = self.integrator.clf_dt(&scene.vertices, dt_remaining, 1e-6);
// 
//             (&mut self.buf_velocities[0..particle_count], &scene.vertices)
//                 .into_par_iter()
//                 .for_each(|(v, p)| *v = p.velocity());
// 
//             scene.vertices
//                 .par_iter_mut()
//                 .enumerate()
//                 .for_each(|(i, p)| {
//                     let f_vis = self.solver.compute_viscosity(
//                         0.05,
//                         self.buf_positions[i],
//                         self.buf_velocities[i],
//                         &self.neighbor_search,
//                         &self.buf_positions,
//                         &self.buf_velocities,
//                         &self.buf_masses,
//                         &self.buf_densities,
//                     );
//                     self.integrator.predict_velocity(p, f_vis, self.gravity, substep);
//                 });
// 
//             self.solver.solve_density_constraints(
//                 &self.neighbor_search,
//                 &mut scene.vertices,
//                 &self.buf_positions,
//                 &self.buf_masses,
//                 &self.buf_densities,
//                 &self.buf_alphas,
//                 &mut self.buf_velocities,
//                 &mut self.buf_kappas,
//                 &mut self.buf_delta_v,
//                 substep,
//             );
// 
//             scene.vertices.par_iter_mut().for_each(|p| {
//                 p.update(substep);
//                 self.check_collision(
//                     p,
//                     Vec3::new(-1.5, 0.0, -1.5),
//                     Vec3::new(1.5, 5.0, 1.5)
//                 );
//             });
// 
//             let num = scene.vertices.len();
//             (&mut self.buf_positions[0..num], &scene.vertices)
//                 .into_par_iter()
//                 .for_each(|(pos, p)| *pos = p.location());
// 
//             self.neighbor_search.build(&self.buf_positions[0..particle_count]);
// 
//             (&mut self.buf_densities[0..particle_count], &mut self.buf_alphas[0..particle_count], &mut scene.vertices)
//                 .into_par_iter()
//                 .enumerate()
//                 .for_each(|(i, (dens, alpha, p))| {
//                     let (d, a) = self.solver.compute_density_and_factor(
//                         self.buf_positions[i],
//                         self.buf_masses[i],
//                         &self.neighbor_search,
//                         &self.buf_positions,
//                         &self.buf_masses
//                     );
//                     *dens = d;
//                     *alpha = a;
//                     p.density = d;
//                     p.alpha = a;
//                 });
// 
//             self.solver.solve_divergence_constraints(
//                 &self.neighbor_search,
//                 &mut scene.vertices,
//                 &self.buf_positions,
//                 &self.buf_masses,
//                 &self.buf_densities,
//                 &self.buf_alphas,
//                 &mut self.buf_velocities,
//                 &mut self.buf_kappas,
//                 &mut self.buf_delta_v,
//                 substep,
//             );
//             dt_remaining -= substep;
//             substeps_done += 1;
//         }
//     }
// 
//     pub fn check_collision(&self, p: &mut Particle, box_min: Vec3, box_max: Vec3) {
//         let mut pos = p.location();
//         let mut vel = p.velocity();
//         let radius = p.radius;
//         let restitution = self.damping;
// 
// 
//         let friction = 0.5;
// 
//         let epsilon = 0.0001;
// 
//         // --- X Axis ---
//         if pos.x < box_min.x + radius {
//             pos.x = box_min.x + radius + epsilon;
//             vel.x *= -restitution;      // Відскок
//             vel.y *= friction;          // Тертя
//             vel.z *= friction;          // Тертя
//         } else if pos.x > box_max.x - radius {
//             pos.x = box_max.x - radius - epsilon;
//             vel.x *= -restitution;
//             vel.y *= friction;
//             vel.z *= friction;
//         }
// 
//         // --- Y Axis ---
//         if pos.y < box_min.y + radius {
//             pos.y = box_min.y + radius + epsilon;
//             vel.y *= -restitution;
//             vel.x *= friction;
//             vel.z *= friction;
//         } else if pos.y > box_max.y - radius {
//             pos.y = box_max.y - radius - epsilon;
//             vel.y *= -restitution;
//             vel.x *= friction;
//             vel.z *= friction;
//         }
// 
//         // --- Z Axis ---
//         if pos.z < box_min.z + radius {
//             pos.z = box_min.z + radius + epsilon;
//             vel.z *= -restitution;
//             vel.x *= friction;
//             vel.y *= friction;
//         } else if pos.z > box_max.z - radius {
//             pos.z = box_max.z - radius - epsilon;
//             vel.z *= -restitution;
//             vel.x *= friction;
//             vel.y *= friction;
//         }
// 
//         p.set_position(pos);
//         p.set_velocity(vel);
//     }
// }