use crate::entities::particle::{stats_layout, ParticleBuffers, SimulationParams};
use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo, DispatchIndirectCommand};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::Device;
use vulkano::memory::allocator::StandardMemoryAllocator;
#[path = "passes/density_alpha.rs"]
mod density_alpha;
use density_alpha::DensityAlphaPipeline;
#[path = "passes/density_source_term.rs"]
mod density_source_term;
use density_source_term::DensitySourceTermPipeline;
#[path = "passes/divergence_integration.rs"]
mod divergence_integration;
use divergence_integration::DivergenceIntegrationPipeline;
#[path = "passes/divergence_source_term.rs"]
mod divergence_source_term;
use divergence_source_term::DivergenceSourceTermPipeline;
#[path = "passes/pressure_force_pipeline.rs"]
mod pressure_force_pipeline;
use pressure_force_pipeline::PressureForcePipeline;
#[path = "passes/pressure_integration_pipeline.rs"]
mod pressure_integration_pipeline;
use pressure_integration_pipeline::PressureIntegrationPipeline;
#[path = "passes/pressure_update_pipeline.rs"]
mod pressure_update_pipeline;
use pressure_update_pipeline::PressureUpdatePipeline;
#[path = "passes/solver_check.rs"]
mod solver_check;
use solver_check::SolverCheckPipeline;
#[path = "passes/stats_pipeline.rs"]
mod stats_pipeline;
use stats_pipeline::StatsPipeline;
#[path = "passes/viscosity.rs"]
mod viscosity;
use viscosity::ViscosityPipeline;
#[path = "neighbor_search.rs"]
mod neighbor_search;
use neighbor_search::NeighborSearch;
#[path = "sort/mod.rs"]
pub(crate) mod sorter;
pub use sorter::SortAlgorithm;
#[cfg(test)]
#[path = "benchmarks/convergence_benchmark.rs"]
mod convergence_benchmark;
#[cfg(test)]
#[path = "benchmarks/substep_benchmark.rs"]
mod substep_benchmark;

pub struct ComputePipelines {
    pub neighbor_search: NeighborSearch,
    pub density_alpha: DensityAlphaPipeline,
    pub viscosity: ViscosityPipeline,
    pub density_source_term: DensitySourceTermPipeline,
    pub pressure_force: PressureForcePipeline,
    pub pressure_update: PressureUpdatePipeline,
    pub pressure_integration: PressureIntegrationPipeline,
    pub divergence_source_term: DivergenceSourceTermPipeline,
    pub divergence_integration: DivergenceIntegrationPipeline,
    pub stats: StatsPipeline,
    pub solver_check: SolverCheckPipeline,

    group_count: u32,
    solver_state: Option<Subbuffer<[u32]>>,
    solver_dispatch: Option<Subbuffer<[DispatchIndirectCommand]>>,
    stats_buffer: Option<Subbuffer<[u32]>>,
}

/// Which of the two DFSPH pressure solves is being recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PressureSolve {
    Density,
    Divergence,
}

/// Iteration limits and error thresholds of one pressure solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveSettings {
    pub min_iterations: u32,
    pub max_iterations: u32,
    /// Average compression error (kg/m^3) at which the solve stops early;
    /// negative to always run `max_iterations`.
    pub max_avg_error: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverSettings {
    pub density: SolveSettings,
    pub divergence: SolveSettings,
}

impl SolverSettings {
    /// Settings used by the application when its default error thresholds are enabled.
    #[cfg(test)]
    pub fn default_for(params: &SimulationParams) -> Self {
        let error = 0.1 / 100.0 * params.target_density;
        Self {
            density: SolveSettings { min_iterations: 2.min(params.density_solver_iterations), max_iterations: params.density_solver_iterations, max_avg_error: error },
            divergence: SolveSettings { min_iterations: 1.min(params.divergence_solver_iterations), max_iterations: params.divergence_solver_iterations, max_avg_error: error },
        }
    }
    /// Exactly `density` / `divergence` iterations, no early exit.
    pub fn fixed(density: u32, divergence: u32) -> Self {
        let fixed = |n| SolveSettings { min_iterations: n, max_iterations: n, max_avg_error: -1.0 };
        Self { density: fixed(density), divergence: fixed(divergence) }
    }
}

impl ComputePipelines {
    pub fn new(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        sort_buffer_size: u32,
    ) -> Self {
        let neighbor_search =
            NeighborSearch::new_with_allocator(device.clone(), memory_allocator, sort_buffer_size);
        let density_alpha = DensityAlphaPipeline::new(device.clone());
        let viscosity = ViscosityPipeline::new(device.clone());
        let density_source_term = DensitySourceTermPipeline::new(device.clone());
        let pressure_force = PressureForcePipeline::new(device.clone());
        let pressure_update = PressureUpdatePipeline::new(device.clone());
        let pressure_integration = PressureIntegrationPipeline::new(device.clone());
        let divergence_source_term = DivergenceSourceTermPipeline::new(device.clone());
        let divergence_integration = DivergenceIntegrationPipeline::new(device.clone());
        let stats = StatsPipeline::new(device.clone());
        let solver_check = SolverCheckPipeline::new(device.clone());

        Self {
            neighbor_search,
            density_alpha,
            viscosity,
            density_source_term,
            pressure_force,
            pressure_update,
            pressure_integration,
            divergence_source_term,
            divergence_integration,
            stats,
            solver_check,
            group_count: 0,
            solver_state: None,
            solver_dispatch: None,
            stats_buffer: None,
        }
    }

    /// Prepares every physics pass for `physics_data`.
    pub fn prepare(
        &mut self,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &ParticleBuffers,
        sim_params: &Subbuffer<SimulationParams>,
    ) {
        self.neighbor_search.prepare(allocator.clone(), physics_data, sim_params);
        self.density_alpha.prepare(allocator.clone(), physics_data, sim_params);
        self.viscosity.prepare(allocator.clone(), physics_data, sim_params);
        self.density_source_term.prepare(allocator.clone(), physics_data, sim_params);
        self.pressure_force.prepare(allocator.clone(), physics_data, sim_params);
        self.pressure_update.prepare(allocator.clone(), physics_data, sim_params);
        self.pressure_integration.prepare(allocator.clone(), physics_data, sim_params);
        self.divergence_source_term.prepare(allocator.clone(), physics_data, sim_params);
        self.divergence_integration.prepare(allocator.clone(), physics_data, sim_params);
        self.stats.prepare(allocator.clone(), physics_data, sim_params);
        self.solver_check.prepare(allocator, physics_data);

        self.group_count = physics_data.count.div_ceil(256);
        self.solver_state = Some(physics_data.solver_state.clone());
        self.solver_dispatch = Some(physics_data.solver_dispatch.clone());
        self.stats_buffer = Some(physics_data.stats_buffer.clone());
    }

    pub fn record_init<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        self.neighbor_search.execute(builder);
        self.density_alpha.execute(builder);
    }

    /// Records one DFSPH substep: non-pressure forces, density solve,
    /// position update, neighbour rebuild, divergence solve.
    pub fn record_substep<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, solver: &SolverSettings) {
        self.viscosity.execute(builder);
        self.density_source_term.execute(builder);
        self.record_pressure_solve(builder, &solver.density, PressureSolve::Density);
        self.pressure_integration.execute(builder);

        self.neighbor_search.execute(builder);
        self.density_alpha.execute(builder);
        self.divergence_source_term.execute(builder);
        self.record_pressure_solve(builder, &solver.divergence, PressureSolve::Divergence);
        self.divergence_integration.execute(builder);
    }

    /// Jacobi iterations on the pressure in `physics_data.pressures`, starting
    /// from the warm-start value written by the source term pass.
    ///
    /// Every iteration updates the pressure and then recomputes the pressure
    /// acceleration from it, so the integration pass that follows applies the
    /// result of the last update. After convergence the remaining iterations
    /// dispatch zero workgroups (see `solver_check.comp`).
    pub(crate) fn record_pressure_solve<Cb>(
        &self,
        builder: &mut AutoCommandBufferBuilder<Cb>,
        settings: &SolveSettings,
        solve: PressureSolve,
    ) {
        let is_divergence_solve = solve == PressureSolve::Divergence;
        let stats_range = match solve {
            PressureSolve::Density => stats_layout::DENSITY_SOLVER,
            PressureSolve::Divergence => stats_layout::DIVERGENCE_SOLVER,
        };
        let state = self.solver_state.clone().expect("ComputePipelines: call prepare() first");
        let dispatch = self.solver_dispatch.clone().expect("ComputePipelines: call prepare() first");
        let stats = self.stats_buffer.clone().expect("ComputePipelines: call prepare() first");

        builder.update_buffer(state.clone(), Box::<[u32]>::from([0u32; 4])).unwrap();
        builder
            .update_buffer(
                dispatch.clone(),
                Box::<[DispatchIndirectCommand]>::from([DispatchIndirectCommand { x: self.group_count, y: 1, z: 1 }]),
            )
            .unwrap();

        self.pressure_force.execute(builder);
        for _ in 0..settings.max_iterations {
            self.pressure_update.execute_indirect(builder, dispatch.clone(), is_divergence_solve);
            self.pressure_force.execute_indirect(builder, dispatch.clone());
            self.solver_check.execute(builder, settings.max_avg_error, settings.min_iterations);
        }

        builder.copy_buffer(CopyBufferInfo::buffers(state, stats.slice(stats_range))).unwrap();
    }
}

