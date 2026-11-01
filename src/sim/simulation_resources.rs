use crate::constants::MAX_FRAMES_IN_FLIGHT;
use crate::scene::Scene;
use crate::sim::params::SimulationParams;
use crate::sim::particle_buffers::{ParticleBuffers, stats_layout};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};

/// GPU buffers owned by the simulation.
pub struct SimulationResources {
    /// Particle state buffers used by the solver.
    pub physics_data: ParticleBuffers,
    /// GPU buffer holding current simulation parameters.
    pub sim_params_buffer: Subbuffer<SimulationParams>,
    /// Per-frame GPU statistics readback buffers.
    pub stats_readback: Vec<Subbuffer<[u32]>>,
}

impl SimulationResources {
    /// Allocates particle, parameter and statistics buffers.
    pub fn new(allocator: Arc<StandardMemoryAllocator>, scene: &Scene) -> Self {
        let physics_data = ParticleBuffers::new(allocator.clone(), scene.initial_positions.clone());

        let sim_params_buffer = Buffer::from_data(
            allocator.clone(),
            BufferCreateInfo {
                // Updated with `update_buffer` in the command stream, so the
                // CPU never writes it while the GPU may still be reading it.
                usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            scene.sim_params,
        )
        .expect("Failed to create simulation params buffer");

        // One copy of the stats per frame in flight: the CPU reads the oldest
        // one, which the GPU has finished with.
        let stats_readback = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                Buffer::from_iter(
                    allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    [0u32; stats_layout::LEN],
                )
                .expect("Failed to create stats readback buffer")
            })
            .collect();

        Self {
            physics_data,
            sim_params_buffer,
            stats_readback,
        }
    }

    /// Reads the oldest available statistics buffer.
    pub fn oldest_stats(
        &self,
        current_frame_idx: usize,
    ) -> Option<vulkano::buffer::BufferReadGuard<'_, [u32]>> {
        let oldest = (current_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;
        self.stats_readback[oldest].read().ok()
    }
}
