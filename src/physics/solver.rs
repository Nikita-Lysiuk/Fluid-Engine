use glam::Vec3;
use rayon::prelude::*;
use crate::core::scene::Scene;
use crate::physics::kernel::SmoothingKernel;
use crate::physics::neighbor_search::NeighborSearch;
use crate::utils::constants::MAX_PARTICLES;

pub struct DFSPHSolver {
    smoothing_kernel: Box<dyn SmoothingKernel + Send + Sync>,
    h: f32,
    h_sq: f32,

    delta_v: Vec<Vec3>,
}

impl DFSPHSolver {
    pub fn new(h: f32, smoothing_kernel: Box<dyn SmoothingKernel + Send + Sync>) -> Self {
        Self {
            h,
            h_sq: h * h,
            smoothing_kernel,
            delta_v: vec![Vec3::ZERO; MAX_PARTICLES],
        }
    }
    pub fn update_densities_and_factors(
        &mut self,
        scene: &mut Scene,
        search: &NeighborSearch
    ) {
        let mass = scene.particle_mass;
        let w_zero = self.smoothing_kernel.w_zero();
        let h_sq = self.h_sq;

        let vertices = &scene.fluid_data.vertices;
        let densities = &mut scene.fluid_data.densities;
        let alphas = &mut scene.fluid_data.alphas;

        densities.par_iter_mut()
            .zip(alphas.par_iter_mut())
            .enumerate()
            .for_each(|(i, (rho, alpha))| {
                let pos_i_arr = vertices[i].position;
                let pos_i = Vec3::from_array(pos_i_arr);

                let mut density = mass * w_zero;

                let mut grad_w_sum = Vec3::ZERO;
                let mut grad_w_sq_sum = 0.0;

                search.for_each_neighbor(&pos_i, |j| {
                    if i == j { return; }

                    let pos_j = Vec3::from_array(vertices[j].position);
                    let r_vec = pos_i - pos_j;
                    let r_sq = r_vec.length_squared();

                    if r_sq < h_sq && r_sq > 1e-9 {
                        let r = r_sq.sqrt();

                        density += mass * self.smoothing_kernel.evaluate(r);

                        let grad_w = self.smoothing_kernel.gradient(r_vec, r);

                        let m_grad = mass * grad_w;
                        grad_w_sum += m_grad;
                        grad_w_sq_sum += m_grad.length_squared();
                    }
                });

                *rho = density;

                let denominator = grad_w_sum.length_squared() + grad_w_sq_sum;
                *alpha = 1.0 / denominator.max(1e-6);
            })
    }
    pub fn apply_viscosity(&mut self, scene: &mut Scene, search: &NeighborSearch) {
        let viscosity_c = 0.05;
        let mass = scene.particle_mass;
        let h_sq = self.h_sq;

        let vertices = &scene.fluid_data.vertices;
        let velocities = &scene.fluid_data.velocities;
        let densities = &scene.fluid_data.densities;

        let correction_buffer = &mut self.delta_v;

        correction_buffer.par_iter_mut()
            .enumerate()
            .for_each(|(i, delta)| {
                let pos_i = Vec3::from_array(vertices[i].position);
                let vel_i = velocities[i];

                let mut sum_v = Vec3::ZERO;

                search.for_each_neighbor(&pos_i, |j| {
                    if i == j { return; }

                    let pos_j = Vec3::from_array(vertices[j].position);
                    let r_vec = pos_i - pos_j;
                    let r_sq = r_vec.length_squared();

                    if r_sq < h_sq {
                        let r = r_sq.sqrt();

                        let w = self.smoothing_kernel.evaluate(r);
                        let vel_j = velocities[j];
                        let rho_j = densities[j];

                        if rho_j > 1e-6 {
                            let volume_j = mass / rho_j;
                            sum_v += volume_j * (vel_j - vel_i) * w;
                        }
                    }
                });

                *delta = viscosity_c * sum_v;
            });

        let velocities_mut = &mut scene.fluid_data.velocities;
        velocities_mut.par_iter_mut()
            .zip(correction_buffer.par_iter())
            .for_each(|(vel, correction)| {
                *vel += *correction;
            })
    }
    pub fn apply_gravity(&self, scene: &mut Scene, dt: f32) {
        let gravity = Vec3::new(0.0, -9.81, 0.0);

        scene.fluid_data.velocities.par_iter_mut()
            .for_each(|vel| {
                *vel += gravity * dt;
            })
    }
    pub fn solve_density(&mut self, scene: &mut Scene, search: &NeighborSearch, dt: f32) {
        let mass = scene.particle_mass;
        let target_density = scene.target_density;
        let max_iter = 2;
        let dt_sq = dt * dt;
        let h_sq = self.h_sq;

        if dt_sq < 1e-9 { return; }

        let vertices = &scene.fluid_data.vertices;
        let densities = &scene.fluid_data.densities;
        let alphas = &scene.fluid_data.alphas;

        let kappas = &mut scene.fluid_data.kappa;
        let delta_vs = &mut self.delta_v;

        let velocities = &mut scene.fluid_data.velocities;

        for _ in 0..max_iter {
            kappas.par_iter_mut()
                .enumerate()
                .for_each(|(i, kappa)| {
                    let pos_i = Vec3::from_array(vertices[i].position);
                    let vel_i = velocities[i];

                    let mut delta_rho = 0.0;

                    search.for_each_neighbor(&pos_i, |j| {
                        if i == j { return; }
                        let pos_j = Vec3::from_array(vertices[j].position);
                        let r_vec = pos_i - pos_j;

                        if r_vec.length_squared() < self.h_sq {
                            let grad_w = self.smoothing_kernel.gradient(r_vec, r_vec.length());
                            let vel_diff = vel_i - velocities[j];

                            delta_rho += mass * vel_diff.dot(grad_w);
                        }
                    });

                    let rho_star = densities[i] + delta_rho * dt;

                    let density_error = (rho_star - target_density).max(0.0);

                    if density_error > 0.0 {
                        *kappa += (density_error / dt_sq) * alphas[i];
                    }
                });

            delta_vs.par_iter_mut()
                .enumerate()
                .for_each(|(i, dv)| {
                    let pos_i = Vec3::from_array(vertices[i].position);
                    let ki = kappas[i];
                    let rho_i = densities[i];

                    let mut accel_sum = Vec3::ZERO;

                    search.for_each_neighbor(&pos_i, |j| {
                        if i == j { return; }

                        let pos_j = Vec3::from_array(vertices[j].position);
                        let r_vec = pos_i - pos_j;

                        if r_vec.length_squared() < h_sq {
                            let grad_w = self.smoothing_kernel.gradient(r_vec, r_vec.length());
                            let kj = kappas[j];
                            let rho_j = densities[j];

                            let term = (ki / rho_i.max(1e-6)) + (kj / rho_j.max(1e-6));
                            accel_sum += term * grad_w * mass;
                        }
                    });

                    *dv = -dt * accel_sum;
                });

            velocities.par_iter_mut()
                .zip(delta_vs.par_iter())
                .for_each(|(vel, dv)| {
                    *vel += *dv;
                });
        }
    }
    pub fn solve_divergence(&mut self, scene: &mut Scene, search: &NeighborSearch, dt: f32) {
        if dt < 1e-9 { return; }

        let iter_count = 1;
        let mass = scene.particle_mass;
        let h_sq = self.h_sq;

        let vertices = &scene.fluid_data.vertices;
        let densities = &scene.fluid_data.densities;
        let alphas = &scene.fluid_data.alphas;
        let velocities = &mut scene.fluid_data.velocities;

        let kappas = &mut scene.fluid_data.kappa_v;
        let delta_vs = &mut self.delta_v;

        for _ in 0..iter_count {
            kappas.par_iter_mut()
                .enumerate()
                .for_each(|(i, kappa)| {
                    let pos_i = Vec3::from_array(vertices[i].position);
                    let vel_i = velocities[i];
                    let mut density_change = 0.0;

                    search.for_each_neighbor(&pos_i, |j| {
                        if i == j { return; }
                        let pos_j = Vec3::from_array(vertices[j].position);
                        let r_vec = pos_i - pos_j;

                        if r_vec.length_squared() < h_sq {
                            let grad_w = self.smoothing_kernel.gradient(r_vec, r_vec.length());
                            let vel_diff = vel_i - velocities[j];
                            density_change += mass * vel_diff.dot(grad_w);
                        }
                    });

                    *kappa = (density_change / dt) * alphas[i];
                });

            delta_vs.par_iter_mut()
                .enumerate()
                .for_each(|(i, dv)| {
                    let pos_i = Vec3::from_array(vertices[i].position);
                    let ki = kappas[i];
                    let rho_i = densities[i];
                    let mut accel_sum = Vec3::ZERO;

                    search.for_each_neighbor(&pos_i, |j| {
                        if i == j { return; }
                        let pos_j = Vec3::from_array(vertices[j].position);
                        let r_vec = pos_i - pos_j;

                        if r_vec.length_squared() < h_sq {
                            let grad_w = self.smoothing_kernel.gradient(r_vec, r_vec.length());
                            let kj = kappas[j];
                            let rho_j = densities[j];

                            let term = (ki / rho_i.max(1e-6)) + (kj / rho_j.max(1e-6));
                            accel_sum += mass * term * grad_w;
                        }
                    });

                    *dv = -dt * accel_sum;
                });

            velocities.par_iter_mut()
                .zip(delta_vs.par_iter())
                .for_each(|(vel, dv)| {
                    *vel += *dv;
                });
        }
    }
}