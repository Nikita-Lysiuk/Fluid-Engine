use glam::Vec3;
use rayon::prelude::*;

use crate::entities::Actor;
use crate::entities::particle::Particle;
use crate::physics::kernel::SmoothingKernel;
use crate::physics::neighbor_search::NeighborSearch;

pub struct DFSPHSolver {
    smoothing_kernel: Box<dyn SmoothingKernel + Send + Sync>,
    radius: f32,
    radius_sq: f32,
}

impl DFSPHSolver {
    pub fn new(h: f32, smoothing_kernel: Box<dyn SmoothingKernel + Send + Sync>) -> Self {
        Self {
            radius: h * 2.0,
            radius_sq: (h * 2.0) * (h * 2.0),
            smoothing_kernel
        }
    }

    pub fn compute_density_and_factor(&self, pos_i: Vec3, mass_i: f32, search: &NeighborSearch, positions: &[Vec3], masses: &[f32]) -> (f32, f32) {
        let mut density = mass_i * self.smoothing_kernel.evaluate(0.0);

        let mut grad_w_sum = Vec3::ZERO;
        let mut grad_w_sq_sum = 0.0;

        search.for_each_neighbor(pos_i, positions, self.radius_sq, |j_idx| {
            let r_vec = pos_i - positions[j_idx];
            let r_sq = r_vec.length_squared();

            if r_sq > 1e-12 {
                let r = r_sq.sqrt();
                let mass_j = masses[j_idx];

                density += mass_j * self.smoothing_kernel.evaluate(r);

                let grad_w = self.smoothing_kernel.gradient(r_vec, r);
                let m_grad = mass_j * grad_w;
                grad_w_sum += m_grad;
                grad_w_sq_sum += m_grad.length_squared();
            }
        });

        let denominator = grad_w_sum.length_squared() + grad_w_sq_sum;
        (density, 1.0 / denominator.max(1e-6))
    }

    pub fn compute_viscosity(
        &self,
        viscosity_coefficient: f32,
        pos_i: Vec3,
        vel_i: Vec3,
        search: &NeighborSearch,
        positions: &[Vec3],
        velocities: &[Vec3],
        masses: &[f32],
        densities: &[f32],
    ) -> Vec3 {
        let mut viscosity_force = Vec3::ZERO;

        let d = 3.0;
        let factor = 2.0 * (d + 2.0);

        search.for_each_neighbor(pos_i, positions, self.radius_sq, |j_idx| {
            let r_vec = pos_i - positions[j_idx];
            let r_sq = r_vec.length_squared();

            if r_sq > 1e-9 {
                let r = r_sq.sqrt();

                let vel_diff = vel_i - velocities[j_idx];
                let dot = vel_diff.dot(r_vec);

                let grad_w = self.smoothing_kernel.gradient(r_vec, r);
                let h = self.radius * 0.5;
                let denominator = r_sq + h * h * 0.01;
                let term = (masses[j_idx] / densities[j_idx]) * (dot / denominator) * grad_w;

                viscosity_force += term;
            }
        });

        viscosity_force * (viscosity_coefficient * factor)
    }

    pub fn solve_density_constraints(
        &self,
        search: &NeighborSearch,
        particles: &mut [Particle],
        positions: &[Vec3],
        masses: &[f32],
        densities: &[f32],
        alphas: &[f32],
        velocities_buf: &mut [Vec3],
        kappas_buf: &mut [f32],
        delta_v_buf: &mut [Vec3],
        dt: f32,
    ) {
        let density_0 = 1000.0;
        let iter_count = 2;
        let dt_sq = dt * dt;
        let num_particles = particles.len();

        velocities_buf[0..num_particles].par_iter_mut()
            .zip(particles.par_iter())
            .for_each(|(v_buf, p)| *v_buf = p.velocity());

        for _ in 0..iter_count {
            kappas_buf[0..num_particles].par_iter_mut()
                .enumerate()
                .for_each(|(i, kappa_out)| {
                    let mut delta_rho = 0.0;
                    let vel_i = velocities_buf[i]; // Читаємо з буфера

                    search.for_each_neighbor(positions[i], positions, self.radius_sq, |j_idx| {
                        let r_vec = positions[i] - positions[j_idx];
                        let r_sq = r_vec.length_squared();
                        if r_sq > 1e-9 {
                            let r = r_sq.sqrt();
                            let vel_diff = vel_i - velocities_buf[j_idx];
                            let grad_w = self.smoothing_kernel.gradient(r_vec, r);
                            delta_rho += masses[j_idx] * vel_diff.dot(grad_w);
                        }
                    });

                    let rho_star = densities[i] + delta_rho * dt;
                    let deviation = (rho_star - density_0).max(0.0);

                    *kappa_out = if dt_sq > 1e-9 {
                        (deviation / dt_sq) * alphas[i]
                    } else {
                        0.0
                    };
                });

            delta_v_buf[0..num_particles].par_iter_mut()
                .enumerate()
                .for_each(|(i, dv_out)| {
                    let mut accel_sum = Vec3::ZERO;
                    let ki = kappas_buf[i];
                    let rho_i = densities[i];

                    search.for_each_neighbor(positions[i], positions, self.radius_sq, |j_idx| {
                        let kj = kappas_buf[j_idx];
                        let rho_j = densities[j_idx];

                        let r_vec = positions[i] - positions[j_idx];
                        let r_sq = r_vec.length_squared();

                        if r_sq > 1e-9 {
                            let r = r_sq.sqrt();
                            let grad_w = self.smoothing_kernel.gradient(r_vec, r);
                            let term = (ki / rho_i.max(1e-6)) + (kj / rho_j.max(1e-6));
                            accel_sum += masses[j_idx] * term * grad_w;
                        }
                    });

                    *dv_out = -dt * accel_sum;
                });

            particles.par_iter_mut()
                .zip(velocities_buf.par_iter_mut())
                .zip(delta_v_buf.par_iter())
                .for_each(|((p, v_buf), dv)| {
                    *v_buf += *dv;
                    p.set_velocity(*v_buf);
                });
        }
    }

    pub fn solve_divergence_constraints(
        &self,
        search: &NeighborSearch,
        particles: &mut [Particle],
        positions: &[Vec3],
        masses: &[f32],
        densities: &[f32],
        alphas: &[f32],
        velocities_buf: &mut [Vec3],
        kappas_buf: &mut [f32],
        delta_v_buf: &mut [Vec3],
        dt: f32,
    ) {
        let iter_count = 1;
        let num_particles = particles.len();

        velocities_buf[0..num_particles].par_iter_mut()
            .zip(particles.par_iter())
            .for_each(|(v_buf, p)| *v_buf = p.velocity());

        for _ in 0..iter_count {
            kappas_buf[0..num_particles].par_iter_mut()
                .enumerate()
                .for_each(|(i, kappa_out)| {
                    let mut density_change_rate = 0.0;
                    let vel_i = velocities_buf[i];

                    search.for_each_neighbor(positions[i], positions, self.radius_sq, |j_idx| {
                        let r_vec = positions[i] - positions[j_idx];
                        let r_sq = r_vec.length_squared();
                        if r_sq > 1e-9 {
                            let r = r_sq.sqrt();
                            let vel_diff = vel_i - velocities_buf[j_idx];
                            let grad_w = self.smoothing_kernel.gradient(r_vec, r);

                            density_change_rate += masses[j_idx] * vel_diff.dot(grad_w);
                        }
                    });

                    *kappa_out = if dt > 1e-9 {
                        (1.0 / dt) * density_change_rate * alphas[i]
                    } else {
                        0.0
                    };
                });

            delta_v_buf[0..num_particles].par_iter_mut()
                .enumerate()
                .for_each(|(i, dv_out)| {
                    let mut accel_sum = Vec3::ZERO;
                    let ki = kappas_buf[i];
                    let rho_i = densities[i];

                    search.for_each_neighbor(positions[i], positions, self.radius_sq, |j_idx| {
                        let kj = kappas_buf[j_idx];
                        let rho_j = densities[j_idx];

                        let r_vec = positions[i] - positions[j_idx];
                        let r_sq = r_vec.length_squared();
                        if r_sq > 1e-9 {
                            let r = r_sq.sqrt();
                            let grad_w = self.smoothing_kernel.gradient(r_vec, r);

                            // Формула ідентична Density solver
                            let term = (ki / rho_i.max(1e-6)) + (kj / rho_j.max(1e-6));
                            accel_sum += masses[j_idx] * term * grad_w;
                        }
                    });

                    *dv_out = -dt * accel_sum;
                });

            particles.par_iter_mut()
                .zip(velocities_buf.par_iter_mut())
                .zip(delta_v_buf.par_iter())
                .for_each(|((p, v_buf), dv)| {
                    *v_buf += *dv;
                    p.set_velocity(*v_buf);
                });
        }
    }

}