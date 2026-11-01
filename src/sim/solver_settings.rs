#[cfg(test)]
use crate::sim::params::SimulationParams;

/// Which of the two DFSPH pressure solves is being recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PressureSolve {
    Density,
    Divergence,
}

/// Iteration limits and error thresholds of one pressure solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveSettings {
    /// Minimum iterations before convergence is checked.
    pub min_iterations: u32,
    /// Maximum iterations in the solve.
    pub max_iterations: u32,
    /// Average compression error (kg/m^3) at which the solve stops early;
    /// negative to always run `max_iterations`.
    pub max_avg_error: f32,
}

/// Settings for both pressure solves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverSettings {
    /// Settings for the density solve.
    pub density: SolveSettings,
    /// Settings for the divergence solve.
    pub divergence: SolveSettings,
}

impl SolverSettings {
    /// Settings used by the application when its default error thresholds are enabled.
    #[cfg(test)]
    pub fn default_for(params: &SimulationParams) -> Self {
        let error = 0.1 / 100.0 * params.target_density;
        Self {
            density: SolveSettings {
                min_iterations: 2.min(params.density_solver_iterations),
                max_iterations: params.density_solver_iterations,
                max_avg_error: error,
            },
            divergence: SolveSettings {
                min_iterations: 1.min(params.divergence_solver_iterations),
                max_iterations: params.divergence_solver_iterations,
                max_avg_error: error,
            },
        }
    }
    /// Exactly `density` / `divergence` iterations, no early exit.
    pub fn fixed(density: u32, divergence: u32) -> Self {
        let fixed = |n| SolveSettings {
            min_iterations: n,
            max_iterations: n,
            max_avg_error: -1.0,
        };
        Self {
            density: fixed(density),
            divergence: fixed(divergence),
        }
    }
}
