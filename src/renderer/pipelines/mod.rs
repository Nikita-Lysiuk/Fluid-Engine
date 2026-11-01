use crate::entities::particle::{stats_layout, ParticleBuffers, SimulationParams};
use crate::renderer::pipelines::collision_pipeline::CollisionPipeline;
use crate::renderer::pipelines::density_alpha::DensityAlphaPipeline;
use crate::renderer::pipelines::density_source_term::DensitySourceTermPipeline;
use crate::renderer::pipelines::density_texture::DensityTexturePipeline;
use crate::renderer::pipelines::divergence_integration::DivergenceIntegrationPipeline;
use crate::renderer::pipelines::divergence_source_term::DivergenceSourceTermPipeline;
use crate::renderer::pipelines::neighbor_search::NeighborSearch;
use crate::renderer::pipelines::point_pipeline::PointPipeline;
use crate::renderer::pipelines::pressure_force_pipeline::PressureForcePipeline;
use crate::renderer::pipelines::pressure_integration_pipeline::PressureIntegrationPipeline;
use crate::renderer::pipelines::pressure_update_pipeline::PressureUpdatePipeline;
use crate::renderer::pipelines::sky_pipeline::SkyPipeline;
use crate::renderer::pipelines::solver_check::SolverCheckPipeline;
use crate::renderer::pipelines::stats_pipeline::StatsPipeline;
use crate::renderer::pipelines::viscosity::ViscosityPipeline;
use crate::renderer::pipelines::water_pipeline::WaterRenderPipeline;
use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo, DispatchIndirectCommand};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::layout::{
    PipelineLayoutCreateInfo, PushConstantRange,
};
use vulkano::pipeline::PipelineLayout;
use vulkano::shader::ShaderStages;
use vulkano_util::context::VulkanoContext;

pub mod collision_pipeline;
#[path = "../../sim/neighbor_search.rs"]
mod neighbor_search;
pub mod point_pipeline;
pub mod sky_pipeline;
#[path = "../../sim/sort/mod.rs"]
mod sorter;
pub use sorter::SortAlgorithm;
mod density_alpha;
mod density_source_term;
mod density_texture;
mod divergence_integration;
mod divergence_source_term;
mod pressure_force_pipeline;
mod pressure_integration_pipeline;
mod pressure_update_pipeline;
mod solver_check;
mod stats_pipeline;
mod viscosity;
mod water_pipeline;

#[cfg(test)]
#[path = "../../sim/benchmarks/convergence_benchmark.rs"]
mod convergence_benchmark;
#[cfg(test)]
#[path = "../../sim/benchmarks/substep_benchmark.rs"]
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
    pub density_texture: DensityTexturePipeline,
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
        let density_texture = DensityTexturePipeline::new(device.clone());
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
            density_texture,
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

pub struct Pipelines {
    pub sky_layout: Arc<PipelineLayout>,
    pub sky_pipeline: Arc<SkyPipeline>,
    pub common_layout: Arc<PipelineLayout>,
    pub point_pipeline: Arc<PointPipeline>,
    pub collision_pipeline: Arc<CollisionPipeline>,
    pub water_renderer_pipeline: Arc<WaterRenderPipeline>,
}

impl Pipelines {
    pub fn new(
        context: Arc<VulkanoContext>,
        swapchain_format: Format,
        depth_format: Format,
    ) -> Self {
        let device = context.device().clone();

        let push_constant_range = PushConstantRange {
            stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            offset: 0,
            size: 16,
        };

        let sky_layout = Self::create_sky_layout(device.clone(), push_constant_range);
        let sky_pipeline = Arc::new(SkyPipeline::new(
            device.clone(),
            sky_layout.clone(),
            swapchain_format,
            depth_format,
        ));

        let common_layout = Self::create_common_layout(device.clone(), push_constant_range);
        let point_pipeline = Arc::new(PointPipeline::new(
            device.clone(),
            swapchain_format,
            depth_format,
        ));
        let collision_pipeline = Arc::new(CollisionPipeline::new(
            device.clone(),
            common_layout.clone(),
            swapchain_format,
            depth_format,
        ));

        let water_renderer_pipeline = Arc::new(WaterRenderPipeline::new(
            device.clone(),
            swapchain_format,
            depth_format,
        ));

        Self {
            sky_layout,
            sky_pipeline,
            common_layout,
            point_pipeline,
            collision_pipeline,
            water_renderer_pipeline,
        }
    }

    fn create_sky_layout(device: Arc<Device>, pc: PushConstantRange) -> Arc<PipelineLayout> {
        let dsl = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings: [(
                    0,
                    DescriptorSetLayoutBinding {
                        stages: ShaderStages::FRAGMENT,
                        ..DescriptorSetLayoutBinding::descriptor_type(
                            DescriptorType::CombinedImageSampler,
                        )
                    },
                )]
                .into(),
                ..Default::default()
            },
        )
        .unwrap();

        PipelineLayout::new(
            device,
            PipelineLayoutCreateInfo {
                set_layouts: vec![dsl],
                push_constant_ranges: vec![pc],
                ..Default::default()
            },
        )
        .unwrap()
    }
    fn create_common_layout(device: Arc<Device>, pc: PushConstantRange) -> Arc<PipelineLayout> {
        PipelineLayout::new(
            device,
            PipelineLayoutCreateInfo {
                set_layouts: vec![],
                push_constant_ranges: vec![pc],
                ..Default::default()
            },
        )
        .unwrap()
    }
}
