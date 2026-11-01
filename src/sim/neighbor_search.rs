use crate::shader_loader::load_shader_entry_point;
use crate::sim::compute_pass::{ComputePass, new_pipeline};
use crate::sim::params::SimulationParams;
use crate::sim::particle_buffers::ParticleBuffers;
use crate::sim::pipelines::SortAlgorithm;
use crate::sim::shaders::{cs_hash, cs_offsets, cs_reorder};
use crate::sim::sort::{BitonicSorter, RadixSorter};
use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo};
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::Device;
use vulkano::instance::debug::DebugUtilsLabel;
use vulkano::memory::allocator::StandardMemoryAllocator;

/// Builds and sorts the particle neighborhood grid.
pub struct NeighborSearch {
    sorter: BitonicSorter,
    radix_sorter: RadixSorter,

    hash_pass: ComputePass,
    offsets_pass: ComputePass,
    reorder_pass: ComputePass,

    grid_start: Subbuffer<[u32]>,
    warm_start: Subbuffer<[[f32; 2]]>,
    warm_start_sorted: Subbuffer<[[f32; 2]]>,

    sort_buffer_len: u32,
    num_particles: u32,

    /// Particle sorting algorithm used by neighbor search.
    pub sort_algorithm: SortAlgorithm,
}

impl NeighborSearch {
    /// Creates the hash, sort, offsets and reorder passes for `physics_data`.
    pub fn new(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &ParticleBuffers,
        sim_params: &Subbuffer<SimulationParams>,
    ) -> Self {
        let num_particles = physics_data.count;
        let sort_buffer_len = physics_data.grid_entries.len() as u32;

        let mut sorter = BitonicSorter::new(device.clone());
        sorter.prepare(allocator.clone(), physics_data, sim_params);
        let mut radix_sorter = RadixSorter::new(device.clone(), memory_allocator, sort_buffer_len);
        radix_sorter.prepare(allocator.clone(), &physics_data.grid_entries);

        let pipeline = |load| {
            new_pipeline(
                device.clone(),
                load_shader_entry_point(device.clone(), load, "main"),
            )
        };

        let hash_pass = ComputePass::new(
            pipeline(cs_hash::load),
            allocator.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_data.position_a.clone()),
                WriteDescriptorSet::buffer(1, physics_data.grid_entries.clone()),
                WriteDescriptorSet::buffer(2, sim_params.clone()),
            ],
            sort_buffer_len.div_ceil(256),
        );
        let offsets_pass = ComputePass::new(
            pipeline(cs_offsets::load),
            allocator.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_data.grid_entries.clone()),
                WriteDescriptorSet::buffer(1, physics_data.grid_start.clone()),
            ],
            sort_buffer_len.div_ceil(256),
        );
        let reorder_pass = ComputePass::new(
            pipeline(cs_reorder::load),
            allocator,
            [
                WriteDescriptorSet::buffer(0, physics_data.grid_entries.clone()),
                WriteDescriptorSet::buffer(1, physics_data.position_a.clone()),
                WriteDescriptorSet::buffer(2, physics_data.velocity_a.clone()),
                WriteDescriptorSet::buffer(4, physics_data.position_b.clone()),
                WriteDescriptorSet::buffer(5, physics_data.velocity_b.clone()),
                WriteDescriptorSet::buffer(6, physics_data.warm_start.clone()),
                WriteDescriptorSet::buffer(7, physics_data.warm_start_sorted.clone()),
            ],
            num_particles.div_ceil(256),
        );

        Self {
            sorter,
            radix_sorter,
            hash_pass,
            offsets_pass,
            reorder_pass,
            grid_start: physics_data.grid_start.clone(),
            warm_start: physics_data.warm_start.clone(),
            warm_start_sorted: physics_data.warm_start_sorted.clone(),
            sort_buffer_len,
            num_particles,
            sort_algorithm: SortAlgorithm::Bitonic,
        }
    }

    /// Records hashing, sorting, cell offsets and the reorder of particle data
    /// (positions, velocities and warm-start pressures) into sorted order.
    pub fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        let num_particles = self.num_particles;

        self.hash_pass.bind_with_constants(
            builder,
            cs_hash::PushConstants {
                num_particles,
                table_size: self.sort_buffer_len,
            },
        );
        self.hash_pass.dispatch_bound(builder);

        match self.sort_algorithm {
            SortAlgorithm::Bitonic => self.sorter.execute(builder),
            SortAlgorithm::Radix => {
                builder
                    .begin_debug_utils_label(DebugUtilsLabel {
                        label_name: "Radix Sort".into(),
                        color: [1.0, 0.3, 0.0, 1.0],
                        ..Default::default()
                    })
                    .expect("debug label can be opened");
                self.radix_sorter.execute(builder);
                // SAFETY: the radix sort debug label was opened before recording the sort.
                unsafe {
                    builder
                        .end_debug_utils_label()
                        .expect("debug label can be closed");
                }
            }
        }

        builder
            .fill_buffer(self.grid_start.clone(), 0xFFFFFFFF)
            .expect("grid offsets can be cleared");
        self.offsets_pass.bind_with_constants(
            builder,
            cs_offsets::PushConstants {
                num_entries: self.sort_buffer_len,
            },
        );
        self.offsets_pass.dispatch_bound(builder);

        self.reorder_pass
            .bind_with_constants(builder, cs_reorder::PushConstants { num_particles });
        self.reorder_pass.dispatch_bound(builder);
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.warm_start_sorted.clone(),
                self.warm_start.clone(),
            ))
            .expect("sorted warm start can be copied back");
    }
}
