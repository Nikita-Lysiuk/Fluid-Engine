use crate::shader_loader::load_shader_entry_point;
use crate::sim::compute_pass::{ComputePass, new_pipeline};
use crate::sim::neighbor_search::NeighborSearch;
use crate::sim::params::SimulationParams;
use crate::sim::particle_buffers::ParticleBuffers;
use crate::sim::particle_buffers::stats_layout;
use crate::sim::shaders;
use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo, DispatchIndirectCommand};
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::Device;
use vulkano::memory::allocator::StandardMemoryAllocator;

pub use crate::sim::solver_settings::{PressureSolve, SolveSettings, SolverSettings};
pub use crate::sim::sort::SortAlgorithm;

/// Compute passes used by the fluid solver, bound to one set of particle buffers.
/// The passes are visible inside the crate so that benchmarks can time them
/// one by one.
pub struct ComputePipelines {
    num_particles: u32,
    group_count: u32,
    pub neighbor_search: NeighborSearch,
    pub(crate) density_alpha: ComputePass,
    pub(crate) viscosity: ComputePass,
    pub(crate) density_source_term: ComputePass,
    pub(crate) pressure_force: ComputePass,
    pub(crate) pressure_update: ComputePass,
    pub(crate) pressure_integration: ComputePass,
    pub(crate) divergence_source_term: ComputePass,
    pub(crate) divergence_integration: ComputePass,
    pub(crate) stats: ComputePass,
    pub(crate) solver_check: ComputePass,

    solver_state: Subbuffer<[u32]>,
    solver_dispatch: Subbuffer<[DispatchIndirectCommand]>,
    stats_buffer: Subbuffer<[u32]>,
}

impl ComputePipelines {
    /// Creates every physics pass for `physics_data`. The descriptor binding
    /// numbers must match the `layout(binding = N)` of each shader.
    pub fn new(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &ParticleBuffers,
        sim_params: &Subbuffer<SimulationParams>,
    ) -> Self {
        let neighbor_search = NeighborSearch::new(
            device.clone(),
            memory_allocator,
            allocator.clone(),
            physics_data,
            sim_params,
        );
        let pipeline = |load| {
            new_pipeline(
                device.clone(),
                load_shader_entry_point(device.clone(), load, "main"),
            )
        };

        let density_alpha = ComputePass::new(
            pipeline(shaders::density_alpha_shader::load),
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
        );
        let viscosity = ComputePass::new(
            pipeline(shaders::viscosity_shader::load),
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
        );
        let density_source_term = ComputePass::new(
            pipeline(shaders::density_source_term_shader::load),
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
        );
        let pressure_force = ComputePass::new(
            pipeline(shaders::pressure_force_shader::load),
            allocator.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_data.neighbor_list.clone()),
                WriteDescriptorSet::buffer(1, physics_data.neighbor_counts.clone()),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
                WriteDescriptorSet::buffer(3, physics_data.position_b.clone()),
                WriteDescriptorSet::buffer(4, physics_data.pressures.clone()),
                WriteDescriptorSet::buffer(5, physics_data.densities.clone()),
                WriteDescriptorSet::buffer(6, physics_data.pressure_accelerations.clone()),
            ],
            physics_data.count.div_ceil(256),
        );
        let pressure_update = ComputePass::new(
            pipeline(shaders::pressure_update_shader::load),
            allocator.clone(),
            [
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
            ],
            physics_data.count.div_ceil(256),
        );
        let pressure_integration = ComputePass::new(
            pipeline(shaders::pressure_integration_shader::load),
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
        );
        let divergence_source_term = ComputePass::new(
            pipeline(shaders::divergence_source_term_shader::load),
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
        );
        let divergence_integration = ComputePass::new(
            pipeline(shaders::divergence_integration_shader::load),
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
        );
        let stats = ComputePass::new(
            pipeline(shaders::stats_shader::load),
            allocator.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_data.velocity_a.clone()),
                WriteDescriptorSet::buffer(1, physics_data.stats_buffer.clone().slice(0..1)),
            ],
            physics_data.count.div_ceil(256),
        );
        let solver_check = ComputePass::new(
            pipeline(shaders::solver_check_shader::load),
            allocator.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_data.solver_state.clone()),
                WriteDescriptorSet::buffer(1, physics_data.solver_dispatch.clone()),
            ],
            1,
        );

        Self {
            num_particles: physics_data.count,
            group_count: physics_data.count.div_ceil(256),
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
            solver_state: physics_data.solver_state.clone(),
            solver_dispatch: physics_data.solver_dispatch.clone(),
            stats_buffer: physics_data.stats_buffer.clone(),
        }
    }

    /// Clears and records the statistics pass.
    pub fn record_stats<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        builder
            .fill_buffer(self.stats_buffer.clone().slice(0..1), 0)
            .expect("statistics can be cleared");
        self.stats.dispatch(builder);
    }

    /// Records commands needed before the first substep.
    pub fn record_init<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        self.neighbor_search.execute(builder);
        self.density_alpha.dispatch(builder);
    }

    /// Records one DFSPH substep: non-pressure forces, density solve,
    /// position update, neighbour rebuild, divergence solve.
    pub fn record_substep<Cb>(
        &self,
        builder: &mut AutoCommandBufferBuilder<Cb>,
        solver: &SolverSettings,
    ) {
        self.viscosity.dispatch(builder);
        self.density_source_term.dispatch(builder);
        self.record_pressure_solve(builder, &solver.density, PressureSolve::Density);
        self.pressure_integration.dispatch(builder);

        self.neighbor_search.execute(builder);
        self.density_alpha.dispatch(builder);
        self.divergence_source_term.dispatch(builder);
        self.record_pressure_solve(builder, &solver.divergence, PressureSolve::Divergence);
        self.divergence_integration.dispatch(builder);
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
        let state = self.solver_state.clone();
        let dispatch = self.solver_dispatch.clone();
        let stats = self.stats_buffer.clone();

        builder
            .update_buffer(state.clone(), Box::<[u32]>::from([0u32; 4]))
            .expect("solver state can be reset");
        builder
            .update_buffer(
                dispatch.clone(),
                Box::<[DispatchIndirectCommand]>::from([DispatchIndirectCommand {
                    x: self.group_count,
                    y: 1,
                    z: 1,
                }]),
            )
            .expect("solver dispatch arguments can be reset");

        let force = &self.pressure_force;
        let update = &self.pressure_update;
        let check = &self.solver_check;
        force.dispatch(builder);
        for _ in 0..settings.max_iterations {
            update.bind_with_constants(
                builder,
                shaders::pressure_update_shader::PushConstants {
                    is_divergence_solve: is_divergence_solve as u32,
                },
            );
            update.dispatch_indirect_bound(builder, dispatch.clone());
            force.dispatch_indirect(builder, dispatch.clone());
            check.bind_with_constants(
                builder,
                shaders::solver_check_shader::PushConstants {
                    max_avg_error: settings.max_avg_error,
                    min_iterations: settings.min_iterations,
                    num_particles: self.num_particles,
                },
            );
            check.dispatch_bound(builder);
        }

        builder
            .copy_buffer(CopyBufferInfo::buffers(state, stats.slice(stats_range)))
            .expect("solver state can be copied to the stats buffer");
    }
}
