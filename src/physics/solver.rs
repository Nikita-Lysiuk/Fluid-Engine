use glam::Vec3;
use crate::physics::kernel::SmoothingKernel;
use crate::physics::neighbor_search::NeighborSearch;

pub struct DFSPHSolver {
    smoothing_kernel: Box<dyn SmoothingKernel + Send + Sync>,
    radius: f32,
}

impl DFSPHSolver {
    pub fn new(h: f32, smoothing_kernel: Box<dyn SmoothingKernel + Send + Sync>) -> Self {
        Self {
            radius: h * 2.0,
            smoothing_kernel
        }
    }
    pub fn compute_density_and_factor(&self, pos_i: Vec3, search: &NeighborSearch, positions: &[Vec3], masses: &[f32]) -> (f32, f32) {
        let mut density = 0.0;
        let mut grad_w_sum = Vec3::ZERO;
        let mut grad_w_sq_sum = 0.0;

        search.for_each_neighbor(pos_i, positions, self.radius, |j_idx| {
            let mass_j = masses[j_idx];
            let r_vec = pos_i - positions[j_idx];
            let r = r_vec.length();

            if r > 1e-7 {
                density += mass_j * self.smoothing_kernel.evaluate(r);
                let grad_w = self.smoothing_kernel.gradient(r_vec, r);
                let m_grad = mass_j * grad_w;
                grad_w_sum += m_grad;
                grad_w_sq_sum += m_grad.length_squared();
            }
        });

        let denominator = grad_w_sum.length_squared() + grad_w_sq_sum;
        (density, density / denominator.max(1e-6))
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


        search.for_each_neighbor(pos_i, positions, self.radius, |j_idx| {
            let r_vec = pos_i - positions[j_idx];
            let r_sq = r_vec.length_squared();

            if r_sq > 1e-9 {
                let r = r_sq.sqrt();

                let vel_diff = vel_i - velocities[j_idx];
                let dot = vel_diff.dot(r_vec);

                let grad_w = self.smoothing_kernel.gradient(r_vec, r);

                let term = (masses[j_idx] / densities[j_idx]) * (dot / r_sq.max(1e-6)) * grad_w;

                viscosity_force += term;
            }
        });

        viscosity_force * (viscosity_coefficient * factor)
    }

}