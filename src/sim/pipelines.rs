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
    num_particles: u32,
    pub neighbor_search: NeighborSearch,
    pub density_alpha: Option<ComputePass>,
    pub viscosity: Option<ComputePass>,
    pub density_source_term: Option<ComputePass>,
    pub pressure_force: Option<ComputePass>,
    pub pressure_update: Option<ComputePass>,
    pub pressure_integration: Option<ComputePass>,
    pub divergence_source_term: Option<ComputePass>,
    pub divergence_integration: Option<ComputePass>,
    pub stats: Option<ComputePass>,
    pub solver_check: Option<ComputePass>,

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
        let pressure_force = None;
        let pressure_update = None;
        let pressure_integration = None;
        let divergence_source_term = None;
        let divergence_integration = None;
        let stats = None;
        let solver_check = None;

        Self {
            device,
            num_particles: 0,
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
        self.pressure_force = Some(ComputePass::new(
            new_pipeline(self.device.clone(), load_shader_entry_point(self.device.clone(), shaders::pressure_force_shader::load, "main")),
            allocator.clone(),             [
                WriteDescriptorSet::buffer(0, physics_data.neighbor_list.clone()),
                WriteDescriptorSet::buffer(1, physics_data.neighbor_counts.clone()),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
                WriteDescriptorSet::buffer(3, physics_data.position_b.clone()),
                WriteDescriptorSet::buffer(4, physics_data.pressures.clone()),
                WriteDescriptorSet::buffer(5, physics_data.densities.clone()),
                WriteDescriptorSet::buffer(6, physics_data.pressure_accelerations.clone()),
            ], physics_data.count.div_ceil(256),
        ));
        self.pressure_update = Some(ComputePass::new(
            new_pipeline(self.device.clone(), load_shader_entry_point(self.device.clone(), shaders::pressure_update_shader::load, "main")),
            allocator.clone(),             [
                WriteDescriptorSet::buffer(0, physics_data.neighbor_list.clone()),
                WriteDescriptorSet::buffer(1, physics_data.neighbor_counts.clone()),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
                WriteDescriptorSet::buffer(3, physics_data.position_b.clone()),
                WriteDescriptorSet::buffer(4, physics_data.pressure_accelerations.clone()),
                WriteDescriptorSet::buffer(5, physics_data.factors.clone()),
                WriteDescriptorSet::buffer(6, physics_data.source_terms.clone()),
                WriteDescriptorSet::buffer(7, physics_data.densities.clone()),
                WriteDescriptorSet::buffer(8, physics_data.pressures.clone()),
                WriteDescriptorSet::buffer(9, physics_data.solver_state.clone()),
            ], physics_data.count.div_ceil(256),
        ));
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
        self.stats = Some(ComputePass::new(
            new_pipeline(self.device.clone(), load_shader_entry_point(self.device.clone(), shaders::stats_shader::load, "main")),
            allocator.clone(),             [
                WriteDescriptorSet::buffer(0, physics_data.velocity_a.clone()),
                WriteDescriptorSet::buffer(1, physics_data.stats_buffer.clone().slice(0..1)),
            ], physics_data.count.div_ceil(256),
        ));
        self.solver_check = Some(ComputePass::new(
            new_pipeline(self.device.clone(), load_shader_entry_point(self.device.clone(), shaders::solver_check_shader::load, "main")),
            allocator, [WriteDescriptorSet::buffer(0, physics_data.solver_state.clone()), WriteDescriptorSet::buffer(1, physics_data.solver_dispatch.clone())], 1,
        ));

        self.num_particles = physics_data.count;
        self.group_count = physics_data.count.div_ceil(256);
        self.solver_state = Some(physics_data.solver_state.clone());
        self.solver_dispatch = Some(physics_data.solver_dispatch.clone());
        self.stats_buffer = Some(physics_data.stats_buffer.clone());
    }

    /// Clears and records the statistics pass.
    pub fn record_stats<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        let stats = self.stats_buffer.as_ref().expect("prepare statistics before dispatch");
        builder.fill_buffer(stats.clone().slice(0..1), 0).expect("statistics can be cleared");
        self.stats.as_ref().expect("prepare statistics before dispatch").dispatch(builder);
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

        let force = self.pressure_force.as_ref().expect("prepare pressure force before dispatch");
        let update = self.pressure_update.as_ref().expect("prepare pressure update before dispatch");
        let check = self.solver_check.as_ref().expect("prepare solver check before dispatch");
        force.dispatch(builder);
        for _ in 0..settings.max_iterations {
            update.bind_with_constants(builder, shaders::pressure_update_shader::PushConstants { is_divergence_solve: is_divergence_solve as u32 });
            update.dispatch_indirect_bound(builder, dispatch.clone());
            force.dispatch_indirect(builder, dispatch.clone());
            check.bind_with_constants(builder, shaders::solver_check_shader::PushConstants { max_avg_error: settings.max_avg_error, min_iterations: settings.min_iterations, num_particles: self.num_particles });
            check.dispatch_bound(builder);
        }

        builder.copy_buffer(CopyBufferInfo::buffers(state, stats.slice(stats_range))).unwrap();
    }
}

