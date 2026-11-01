use crate::particle_buffers::stats_layout;
use crate::params::SimulationParams;

/// Solver and speed measurements read from the GPU.
pub struct SimulationStats {
    pub max_speed: f32,
    pub density_iterations: u32,
    pub density_error: f32,
    pub divergence_iterations: u32,
    pub divergence_error: f32,
}

impl SimulationStats {
    /// Decodes the simulation statistics buffer.
    pub fn read(words: &[u32]) -> Self {
        let density = &words[stats_layout::DENSITY_SOLVER.start as usize..stats_layout::DENSITY_SOLVER.end as usize];
        let divergence = &words[stats_layout::DIVERGENCE_SOLVER.start as usize..stats_layout::DIVERGENCE_SOLVER.end as usize];
        Self {
            max_speed: f32::from_bits(words[stats_layout::MAX_SPEED]),
            density_iterations: density[stats_layout::ITERATIONS],
            density_error: f32::from_bits(density[stats_layout::AVG_ERROR]),
            divergence_iterations: divergence[stats_layout::ITERATIONS],
            divergence_error: f32::from_bits(divergence[stats_layout::AVG_ERROR]),
        }
    }
}

/// Computes the bounded CFL time step from the latest measured speed.
pub fn cfl_time_step(params: &SimulationParams, max_speed: f32, factor: f32, min_dt: f32, max_dt: f32) -> f32 {
    let diameter = 2.0 * params.particle_radius;
    let gravity = glam::Vec4::from(params.gravity).length();
    let speed = (max_speed + gravity * params.dt).max(1e-9);
    (factor * 0.4 * diameter / speed).clamp(min_dt, max_dt)
}
