use crate::entities::particle::{stats_layout, ParticleBuffers, SimulationParams};
#[path = "shaders.rs"]
mod shaders;
use std::sync::Arc;
use crate::compute_pass::{ComputePass, new_pipeline};
use crate::utils::shader_loader::load_shader_entry_point;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo, DispatchIndirectCommand};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::Device;
use vulkano::memory::allocator::StandardMemoryAllocator;
#[path = "passes/pressure_force_pipeline.rs"]
mod pressure_force_pipeline;
use pressure_force_pipeline::PressureForcePipeline;
#[path = "passes/pressure_update_pipeline.rs"]
mod pressure_update_pipeline;
use pressure_update_pipeline::PressureUpdatePipeline;
#[path = "passes/solver_check.rs"]
mod solver_check;
use solver_check::SolverCheckPipeline;
#[path = "passes/stats_pipeline.rs"]
mod stats_pipeline;
use stats_pipeline::StatsPipeline;
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
    device: Arc<Device>,
    pub neighbor_search: NeighborSearch,
    pub density_alpha: Option<ComputePass>,
    pub viscosity: Option<ComputePass>,
    pub density_source_term: Option<ComputePass>,
    pub pressure_force: PressureForcePipeline,
    pub pressure_update: PressureUpdatePipeline,
    pub pressure_integration: Option<ComputePass>,
    pub divergence_source_term: Option<ComputePass>,
    pub divergence_integration: Option<ComputePass>,
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
        let density_alpha = None;
        let viscosity = None;
        let density_source_term = None;
        let pressure_force = PressureForcePipeline::new(device.clone());
        let pressure_update = PressureUpdatePipeline::new(device.clone());
        let pressure_integration = None;
        let divergence_source_term = None;
        let divergence_integration = None;
        let stats = StatsPipeline::new(device.clone());
        let solver_check = SolverCheckPipeline::new(device.clone());

        Self {
            device,
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
        self.density_alpha = Some(ComputePass::new(
            new_pipeline(self.device.clone(), load_shader_entry_point(self.device.clone(), shaders::density_alpha_shader::load, "main")),
            allocator.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_data.grid_entries.clone()),
                WriteDescriptorSet::buffer(1, physics_data.grid_start.clone()),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
                WriteDescriptorSet::buffer(3, physics_data.position_b.clone()),
                WriteDescriptorSet::buffer(4, physics_data.densities.clone()),
                WriteDescriptorSet::buffer(5, physics_data.factors.clone()),
                WriteDescriptorSet::buffer(6, physics_data.neighbor_list.clone()),
                WriteDescriptorSet::buffer(7, physics_data.neighbor_counts.clone()),
            ],
            physics_data.count.div_ceil(256),
        ));
        self.viscosity = Some(ComputePass::new(
            new_pipeline(self.device.clone(), load_shader_entry_point(self.device.clone(), shaders::viscosity_shader::load, "main")),
            allocator.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_data.neighbor_list.clone()),
                WriteDescriptorSet::buffer(1, physics_data.neighbor_counts.clone()),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
                WriteDescriptorSet::buffer(3, physics_data.velocity_b.clone()),
                WriteDescriptorSet::buffer(4, physics_data.position_b.clone()),
                WriteDescriptorSet::buffer(5, physics_data.densities.clone()),
                WriteDescriptorSet::buffer(6, physics_data.velocity_a.clone()),
            ],
            physics_data.count.div_ceil(256),
        ));
        self.density_source_term = Some(ComputePass::new(
            new_pipeline(self.device.clone(), load_shader_entry_point(self.device.clone(), shaders::density_source_term_shader::load, "main")),
            allocator.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_data.neighbor_list.clone()),
                WriteDescriptorSet::buffer(1, physics_data.neighbor_counts.clone()),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
                WriteDescriptorSet::buffer(3, physics_data.position_b.clone()),
                WriteDescriptorSet::buffer(4, physics_data.densities.clone()),
                WriteDescriptorSet::buffer(5, physics_data.velocity_a.clone()),
                WriteDescriptorSet::buffer(6, physics_data.pressures.clone()),
                WriteDescriptorSet::buffer(7, physics_data.source_terms.clone()),
                WriteDescriptorSet::buffer(8, physics_data.warm_start.clone()),
            ],
            physics_data.count.div_ceil(256),
        ));
        self.pressure_force.prepare(allocator.clone(), physics_data, sim_params);
        self.pressure_update.prepare(allocator.clone(), physics_data, sim_params);
        self.pressure_integration = Some(ComputePass::new(
            new_pipeline(self.device.clone(), load_shader_entry_point(self.device.clone(), shaders::pressure_integration_shader::load, "main")),
            allocator.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_data.position_b.clone()),
                WriteDescriptorSet::buffer(1, physics_data.pressure_accelerations.clone()),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
                WriteDescriptorSet::buffer(3, physics_data.velocity_a.clone()),
                WriteDescriptorSet::buffer(4, physics_data.position_a.clone()),
                WriteDescriptorSet::buffer(5, physics_data.pressures.clone()),
                WriteDescriptorSet::buffer(6, physics_data.warm_start.clone()),
            ],
            physics_data.count.div_ceil(256),
        ));
        self.divergence_source_term = Some(ComputePass::new(
            new_pipeline(self.device.clone(), load_shader_entry_point(self.device.clone(), shaders::divergence_source_term_shader::load, "main")),
            allocator.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_data.neighbor_list.clone()),
                WriteDescriptorSet::buffer(1, physics_data.neighbor_counts.clone()),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
                WriteDescriptorSet::buffer(3, physics_data.position_b.clone()),
                WriteDescriptorSet::buffer(4, physics_data.velocity_a.clone()),
                WriteDescriptorSet::buffer(5, physics_data.source_terms.clone()),
                WriteDescriptorSet::buffer(6, physics_data.pressures.clone()),
                WriteDescriptorSet::buffer(7, physics_data.warm_start.clone()),
                WriteDescriptorSet::buffer(8, physics_data.densities.clone()),
            ],
            physics_data.count.div_ceil(256),
        ));
        self.divergence_integration = Some(ComputePass::new(
            new_pipeline(self.device.clone(), load_shader_entry_point(self.device.clone(), shaders::divergence_integration_shader::load, "main")),
            allocator.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_data.velocity_a.clone()),
                WriteDescriptorSet::buffer(1, physics_data.pressure_accelerations.clone()),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
                WriteDescriptorSet::buffer(4, physics_data.colors.clone()),
                WriteDescriptorSet::buffer(5, physics_data.pressures.clone()),
                WriteDescriptorSet::buffer(6, physics_data.warm_start.clone()),
            ],
            physics_data.count.div_ceil(256),
        ));
        self.stats.prepare(allocator.clone(), physics_data, sim_params);
        self.solver_check.prepare(allocator, physics_data);

        self.group_count = physics_data.count.div_ceil(256);
        self.solver_state = Some(physics_data.solver_state.clone());
        self.solver_dispatch = Some(physics_data.solver_dispatch.clone());
        self.stats_buffer = Some(physics_data.stats_buffer.clone());
    }

    pub fn record_init<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        self.neighbor_search.execute(builder);
        self.density_alpha.as_ref().expect("prepare density_alpha before dispatch").dispatch(builder);
    }

    /// Records one DFSPH substep: non-pressure forces, density solve,
    /// position update, neighbour rebuild, divergence solve.
    pub fn record_substep<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, solver: &SolverSettings) {
        self.viscosity.as_ref().expect("prepare viscosity before dispatch").dispatch(builder);
        self.density_source_term.as_ref().expect("prepare density_source_term before dispatch").dispatch(builder);
        self.record_pressure_solve(builder, &solver.density, PressureSolve::Density);
        self.pressure_integration.as_ref().expect("prepare pressure_integration before dispatch").dispatch(builder);

        self.neighbor_search.execute(builder);
        self.density_alpha.as_ref().expect("prepare density_alpha before dispatch").dispatch(builder);
        self.divergence_source_term.as_ref().expect("prepare divergence_source_term before dispatch").dispatch(builder);
        self.record_pressure_solve(builder, &solver.divergence, PressureSolve::Divergence);
        self.divergence_integration.as_ref().expect("prepare divergence_integration before dispatch").dispatch(builder);
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

