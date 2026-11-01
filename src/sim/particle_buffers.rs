use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::DispatchIndirectCommand;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};

/// Particle index and spatial hash used for sorting.
#[derive(BufferContents, Copy, Clone)]
#[repr(C)]
pub struct Entry {
    /// Spatial hash of the particle.
    pub hash: u32,
    /// Particle index associated with the hash.
    pub index: u32,
}

/// Must match `MAX_NEIGHBORS` in common.glsl.
pub const MAX_NEIGHBORS: u32 = 64;

/// GPU buffers holding simulation state.
pub struct ParticleBuffers {
    /// Number of simulated particles.
    pub count: u32,

    /// Primary particle positions.
    pub position_a: Subbuffer<[[f32; 4]]>,
    /// Reordered particle positions.
    pub position_b: Subbuffer<[[f32; 4]]>,

    /// Primary particle velocities.
    pub velocity_a: Subbuffer<[[f32; 4]]>,
    /// Reordered particle velocities.
    pub velocity_b: Subbuffer<[[f32; 4]]>,

    /// Particle colors for rendering.
    pub colors: Subbuffer<[[f32; 4]]>,

    /// Particle density estimates.
    pub densities: Subbuffer<[f32]>,
    /// Pressure solver coefficients.
    pub factors: Subbuffer<[f32]>,

    /// Pressure solver source terms.
    pub source_terms: Subbuffer<[f32]>,
    /// Particle pressure estimates.
    pub pressures: Subbuffer<[f32]>,
    /// Pressure acceleration per particle.
    pub pressure_accelerations: Subbuffer<[[f32; 4]]>,


    /// Sorted particle hash entries.
    pub grid_entries: Subbuffer<[Entry]>,
    /// First entry for each spatial hash bucket.
    pub grid_start: Subbuffer<[u32]>,

    /// Neighbour indices, `MAX_NEIGHBORS` per particle, entry k of particle i
    /// at `k * count + i`. Built by density_and_alpha.comp.
    pub neighbor_list: Subbuffer<[u32]>,
    /// Number of neighbors per particle.
    pub neighbor_counts: Subbuffer<[u32]>,

    /// Final pressure of the previous density (x) and divergence (y) solve,
    /// scaled to be independent of dt (see `WARM_START_*` in common.glsl).
    pub warm_start: Subbuffer<[[f32; 2]]>,
    /// Target of the neighbour-search reorder, copied back into `warm_start`.
    pub warm_start_sorted: Subbuffer<[[f32; 2]]>,

    /// `SolverState` (solver.glsl) of the pressure solve in progress.
    pub solver_state: Subbuffer<[u32]>,
    /// Workgroup count of the solver passes; zeroed on convergence.
    pub solver_dispatch: Subbuffer<[DispatchIndirectCommand]>,

    /// Host-visible, read by the CPU after the frame. Layout: `stats_layout`.
    pub stats_buffer: Subbuffer<[u32]>,
}

/// Layout of `ParticleBuffers::stats_buffer`.
pub mod stats_layout {
    use std::ops::Range;

    /// Maximum particle speed as `f32` bits.
    pub const MAX_SPEED: usize = 0;
    /// `SolverState` of the last density solve of the frame.
    pub const DENSITY_SOLVER: Range<u64> = 1..5;
    /// `SolverState` of the last divergence solve of the frame.
    pub const DIVERGENCE_SOLVER: Range<u64> = 5..9;
    pub const LEN: usize = 9;

    /// Offsets inside a `SolverState`.
    pub const ITERATIONS: usize = 1;
    pub const AVG_ERROR: usize = 3;
}

impl ParticleBuffers {
    /// Creates the associated resources.
    pub fn new(
        allocator: Arc<StandardMemoryAllocator>,
        initial_positions: Vec<[f32; 3]>,
    ) -> Self {
        let count = initial_positions.len() as u32;

        let positions_vec4: Vec<[f32; 4]> = initial_positions.iter()
            .map(|p| [p[0], p[1], p[2], 1.0])
            .collect();

        let position_a = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            positions_vec4.into_iter(),
        ).expect("Failed to create position buffer");

        let position_b = Self::new_device_buffer(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            allocator.clone(),
            count as u64
        );

        let velocity_a = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (0..count).map(|_| [0.0, 0.0, 0.0, 0.0]),
        ).expect("Failed to create velocity buffer");

        let velocity_b = Self::new_device_buffer(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            allocator.clone(),
            count as u64
        );

        let pressures = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (0..count).map(|_| 0.0f32),
        ).expect("Failed to create pressure buffer");

        let pressure_accelerations = Self::new_device_buffer::<[f32; 4]>(
            BufferUsage::STORAGE_BUFFER,
            allocator.clone(),
            count as u64
        );

        let colors = Self::new_device_buffer(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            allocator.clone(),
            count as u64
        );

        let densities = Self::new_device_buffer::<f32>(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
            allocator.clone(),
            count as u64
        );

        let factors = Self::new_device_buffer::<f32>(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
            allocator.clone(),
            count as u64
        );

        let source_terms = Self::new_device_buffer::<f32>(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
            allocator.clone(),
            count as u64
        );

        let sort_buffer_size = count.next_power_of_two();

        let grid_entries = Self::new_device_buffer::<Entry>(
            BufferUsage::STORAGE_BUFFER,
            allocator.clone(),
            sort_buffer_size as u64
        );

        let grid_start = Self::new_device_buffer::<u32>(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            allocator.clone(),
            sort_buffer_size as u64
        );

        let neighbor_list = Self::new_device_buffer::<u32>(
            BufferUsage::STORAGE_BUFFER,
            allocator.clone(),
            count as u64 * MAX_NEIGHBORS as u64,
        );
        let neighbor_counts = Self::new_device_buffer::<u32>(BufferUsage::STORAGE_BUFFER, allocator.clone(), count as u64);

        let warm_start = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (0..count).map(|_| [0.0f32; 2]),
        ).expect("Failed to create warm start buffer");
        let warm_start_sorted = Self::new_device_buffer::<[f32; 2]>(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
            allocator.clone(),
            count as u64,
        );

        let solver_state = Self::new_device_buffer::<u32>(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            allocator.clone(),
            4,
        );
        let solver_dispatch = Self::new_device_buffer::<DispatchIndirectCommand>(
            BufferUsage::STORAGE_BUFFER | BufferUsage::INDIRECT_BUFFER | BufferUsage::TRANSFER_DST,
            allocator.clone(),
            1,
        );

        let stats_buffer = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            [0u32; stats_layout::LEN].into_iter(),
        ).expect("Failed to create stats buffer");

        Self {
            count,
            position_a,
            position_b,
            velocity_a,
            velocity_b,
            colors,
            densities,
            factors,
            source_terms,
            pressures,
            pressure_accelerations,
            grid_entries,
            grid_start,
            neighbor_list,
            neighbor_counts,
            warm_start,
            warm_start_sorted,
            solver_state,
            solver_dispatch,
            stats_buffer,
        }
    }
    /// Allocates a device-local buffer for simulation data.
    fn new_device_buffer<T>(usage: BufferUsage, allocator: Arc<StandardMemoryAllocator>, count: u64) -> Subbuffer<[T]> where T: BufferContents {
        Buffer::new_slice::<T>(
            allocator.clone(),
            BufferCreateInfo { usage, ..Default::default() },
            AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, ..Default::default() },
            count
        ).unwrap()
    }
}


