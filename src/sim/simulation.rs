//! Simulation command recording and the cached substep buffer.
//!
//! Each substep reorders position and velocity from A to B, applies viscosity
//! into A, solves density, integrates into position A, rebuilds neighbours,
//! and then solves divergence and updates velocity A and particle colours.

use std::sync::Arc;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer};
use crate::sim_pipelines::{ComputePipelines, SolverSettings, SortAlgorithm};

/// Owns the simulation pipelines and reusable substep commands.
pub struct Simulation {
    pub pipelines: ComputePipelines,
    needs_init: bool,
    substep_cache: Option<((SolverSettings, SortAlgorithm), Arc<PrimaryAutoCommandBuffer>)>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue_family_index: u32,
}

impl Simulation {
    /// Creates the simulation command recorder.
    pub fn new(pipelines: ComputePipelines, command_buffer_allocator: Arc<StandardCommandBufferAllocator>, queue_family_index: u32) -> Self {
        Self { pipelines, needs_init: true, substep_cache: None, command_buffer_allocator, queue_family_index }
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
