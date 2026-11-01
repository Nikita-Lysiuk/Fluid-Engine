//! Simulation command recording and the cached substep buffer.
//!
//! Each substep reorders position and velocity from A to B, applies viscosity
//! into A, solves density, integrates into position A, rebuilds neighbours,
//! and then solves divergence and updates velocity A and particle colours.

use std::sync::Arc;
use crate::simulation_resources::SimulationResources;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer};
use vulkano::device::Queue;
use vulkano::sync::GpuFuture;
use crate::scene::Scene;
use crate::sim_pipelines::{ComputePipelines, SolverSettings, SortAlgorithm};

/// Owns the simulation pipelines and reusable substep commands.
pub struct Simulation {
    pub pipelines: ComputePipelines,
    pub resources: SimulationResources,
    needs_init: bool,
    substep_cache: Option<((SolverSettings, SortAlgorithm), Arc<PrimaryAutoCommandBuffer>)>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue_family_index: u32,
}

impl Simulation {
    /// Creates the simulation command recorder.
    pub fn new(pipelines: ComputePipelines, resources: SimulationResources, command_buffer_allocator: Arc<StandardCommandBufferAllocator>, queue_family_index: u32) -> Self {
        Self { pipelines, resources, needs_init: true, substep_cache: None, command_buffer_allocator, queue_family_index }
    }

    /// Submits initialization, cached substeps, and readback for one frame.
    pub fn step<F>(&mut self, scene: &mut Scene, frame_dt: f32, max_substeps: u32, solver: SolverSettings, sort_algorithm: SortAlgorithm, frame_index: usize, queue: Arc<Queue>, previous_future: Box<dyn GpuFuture>, copy_render_buffers: F) -> (Box<dyn GpuFuture>, u32)
    where F: FnOnce(&mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        let mut builder = AutoCommandBufferBuilder::primary(self.command_buffer_allocator.clone(), self.queue_family_index, CommandBufferUsage::OneTimeSubmit).expect("initialization command buffer can be allocated");
        builder.update_buffer(self.resources.sim_params_buffer.clone(), Box::new(scene.sim_params)).expect("simulation parameters can be updated");
        self.record_init(&mut builder);
        let mut future = previous_future.then_execute(queue.clone(), builder.build().expect("initialization commands are valid")).expect("initialization can be submitted").boxed();
        let wanted = (frame_dt / scene.sim_params.dt).ceil() as u32;
        let substeps = wanted.max(1).min(max_substeps);
        scene.boundary.update(substeps as f32 * scene.sim_params.dt);
        let substep = self.substep_command_buffer(solver, sort_algorithm);
        for _ in 0..substeps {
            future = future.then_execute(queue.clone(), substep.clone()).expect("substep can be submitted").boxed();
        }
        let mut builder = AutoCommandBufferBuilder::primary(self.command_buffer_allocator.clone(), self.queue_family_index, CommandBufferUsage::OneTimeSubmit).expect("readback command buffer can be allocated");
        self.pipelines.stats.execute(&mut builder);
        builder.copy_buffer(CopyBufferInfo::buffers(self.resources.physics_data.stats_buffer.clone(), self.resources.stats_readback[frame_index].clone())).expect("statistics can be copied");
        copy_render_buffers(&mut builder);
        let future = future.then_execute(queue, builder.build().expect("readback commands are valid")).expect("readback can be submitted").then_signal_semaphore().boxed();
        (future, substeps)
    }

    /// Records initialization before the first substep.
    pub fn record_init<Cb>(&mut self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        if self.needs_init {
            self.pipelines.record_init(builder);
            self.needs_init = false;
        }
    }

    /// Returns the cached command buffer for the selected solver and sorter.
    pub fn substep_command_buffer(&mut self, solver: SolverSettings, sort_algorithm: SortAlgorithm) -> Arc<PrimaryAutoCommandBuffer> {
        let key = (solver, sort_algorithm);
        if let Some((cached_key, command_buffer)) = &self.substep_cache {
            if *cached_key == key { return command_buffer.clone(); }
        }
        let _span = tracy_client::span!("record_substep");
        self.pipelines.neighbor_search.sort_algorithm = sort_algorithm;
        let mut builder = AutoCommandBufferBuilder::primary(self.command_buffer_allocator.clone(), self.queue_family_index, CommandBufferUsage::SimultaneousUse).expect("substep command buffer can be allocated");
        self.pipelines.record_substep(&mut builder, &solver);
        let command_buffer = builder.build().expect("substep commands are valid");
        self.substep_cache = Some((key, command_buffer.clone()));
        command_buffer
    }
}
