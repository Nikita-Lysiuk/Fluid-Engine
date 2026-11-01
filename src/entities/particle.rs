use std::sync::Arc;
use glam::{IVec3, Vec3};
use rand::Rng;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DispatchIndirectCommand};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::Pipeline;
use crate::renderer::pipelines::Pipelines;
use crate::utils::constants::MAX_FRAMES_IN_FLIGHT;

#[repr(C)]
#[derive(BufferContents, Vertex, Copy, Clone, Debug, Default)]
pub struct PositionVertex {
    #[format(R32G32B32A32_SFLOAT)]
    pub position: [f32; 4],
}

#[repr(C)]
#[derive(BufferContents, Vertex, Copy, Clone, Debug, Default)]
pub struct ColorVertex {
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(BufferContents, Vertex, Copy, Clone, Debug, Default)]
pub struct AttributeVertex {
    #[format(R32_SFLOAT)]
    pub radius: f32,
}

pub struct GpuRenderData {
    pub position_buffers: Vec<Subbuffer<[PositionVertex]>>,
    pub color_buffers: Vec<Subbuffer<[ColorVertex]>>,
    pub attribute_buffer: Subbuffer<[AttributeVertex]>,
}

impl GpuRenderData {
    pub fn new(
        allocator: Arc<StandardMemoryAllocator>,
        initial_positions: &[[f32; 3]],
        radius: f32,
    ) -> Self {
        let particle_count = initial_positions.len();

        let mut position_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut color_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let buffer = Buffer::from_iter(
                allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                initial_positions.iter().map(|p| PositionVertex {
                    position: [p[0], p[1], p[2], 1.0]
                }),
            ).expect("Failed to create render position buffer");
            position_buffers.push(buffer);

            let color_buffer = Buffer::from_iter(
                allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                (0..particle_count).map(|_| ColorVertex {
                    color: [0.0, 0.5, 1.0, 1.0]
                }),
            ).expect("Failed to create render color buffer");
            color_buffers.push(color_buffer);
        }

        let attribute_buffer = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (0..particle_count).map(|_| AttributeVertex {
                radius,
            }),
        ).expect("Failed to create attribute buffer");

        Self { position_buffers, attribute_buffer, color_buffers }
    }
    pub fn bind_to_command_buffer<Cb>(
        &self,
        builder: &mut AutoCommandBufferBuilder<Cb>,
        pipelines: &Pipelines,
        camera_addr: u64,
        frame_idx: usize,
        count: u32,
    ) {

        unsafe {
            builder
                .bind_pipeline_graphics(pipelines.point_pipeline.inner.clone()).unwrap()
                .bind_vertex_buffers(0, (
                    self.position_buffers[frame_idx].clone(),
                    self.color_buffers[frame_idx].clone(),
                    self.attribute_buffer.clone()
                ))
                .unwrap()
                .push_constants(
                    pipelines.point_pipeline.inner.layout().clone(),
                    0,
                    camera_addr,
                ).unwrap()
                .draw(count, 1, 0, 0)
                .expect("Failed to bind particle vertex buffers");
        }
    }
}

#[derive(BufferContents, Copy, Clone)]
#[repr(C)]
pub struct Entry {
    pub hash: u32,
    pub index: u32,
}

/// Must match `MAX_NEIGHBORS` in common.glsl.
pub const MAX_NEIGHBORS: u32 = 64;

pub struct GpuPhysicsData {
    pub count: u32,

    pub position_a: Subbuffer<[[f32; 4]]>,
    pub position_b: Subbuffer<[[f32; 4]]>,

    pub velocity_a: Subbuffer<[[f32; 4]]>,
    pub velocity_b: Subbuffer<[[f32; 4]]>,

    pub colors: Subbuffer<[[f32; 4]]>,

    pub densities: Subbuffer<[f32]>,
    pub factors: Subbuffer<[f32]>,

    pub source_terms: Subbuffer<[f32]>,
    pub pressures: Subbuffer<[f32]>,
    pub pressure_accelerations: Subbuffer<[[f32; 4]]>,


    pub grid_entries: Subbuffer<[Entry]>,
    pub grid_start: Subbuffer<[u32]>,

    /// Neighbour indices, `MAX_NEIGHBORS` per particle, entry k of particle i
    /// at `k * count + i`. Built by density_and_alpha.comp.
    pub neighbor_list: Subbuffer<[u32]>,
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

/// Layout of `GpuPhysicsData::stats_buffer`.
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

impl GpuPhysicsData {
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

        let position_b = Self::create_buffer(
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

        let velocity_b = Self::create_buffer(
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

        let pressure_accelerations = Self::create_buffer::<[f32; 4]>(
            BufferUsage::STORAGE_BUFFER,
            allocator.clone(),
            count as u64
        );

        let colors = Self::create_buffer(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            allocator.clone(),
            count as u64
        );

        let densities = Self::create_buffer::<f32>(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
            allocator.clone(),
            count as u64
        );

        let factors = Self::create_buffer::<f32>(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
            allocator.clone(),
            count as u64
        );

        let source_terms = Self::create_buffer::<f32>(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
            allocator.clone(),
            count as u64
        );

        let sort_buffer_size = count.next_power_of_two();

        let grid_entries = Self::create_buffer::<Entry>(
            BufferUsage::STORAGE_BUFFER,
            allocator.clone(),
            sort_buffer_size as u64
        );

        let grid_start = Self::create_buffer::<u32>(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            allocator.clone(),
            sort_buffer_size as u64
        );

        let neighbor_list = Self::create_buffer::<u32>(
            BufferUsage::STORAGE_BUFFER,
            allocator.clone(),
            count as u64 * MAX_NEIGHBORS as u64,
        );
        let neighbor_counts = Self::create_buffer::<u32>(BufferUsage::STORAGE_BUFFER, allocator.clone(), count as u64);

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
        let warm_start_sorted = Self::create_buffer::<[f32; 2]>(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
            allocator.clone(),
            count as u64,
        );

        let solver_state = Self::create_buffer::<u32>(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            allocator.clone(),
            4,
        );
        let solver_dispatch = Self::create_buffer::<DispatchIndirectCommand>(
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
    fn create_buffer<T>(usage: BufferUsage, allocator: Arc<StandardMemoryAllocator>, count: u64) -> Subbuffer<[T]> where T: BufferContents {
        Buffer::new_slice::<T>(
            allocator.clone(),
            BufferCreateInfo { usage, ..Default::default() },
            AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, ..Default::default() },
            count
        ).unwrap()
    }
}

#[repr(C, align(16))]
#[derive(BufferContents, Copy, Clone)]
pub struct SimulationParams {
    pub particle_radius: f32,
    pub particle_mass: f32,
    pub smoothing_radius: f32,
    pub target_density: f32,

    pub viscosity: f32,
    pub relax_factor: f32,
    pub dt: f32,
    pub density_solver_iterations: u32,
    pub divergence_solver_iterations: u32,
    /// Velocity along +x of the moving `box_min.x` wall.
    pub piston_velocity: f32,
    _padding: [f32; 2],

    pub gravity: [f32; 4],
    pub box_min: [f32; 4],
    pub box_max: [f32; 4],

    pub grid_res: [i32; 4],

    /// Mouse attractor: xyz = centre, w = strength in m/s² (0 = off).
    pub attractor: [f32; 4],
    pub attractor_radius: f32,
    _padding_end: [f32; 3],
}

impl SimulationParams {
    pub fn new(
        particle_radius: f32,
        particle_mass: f32,
        smoothing_radius: f32,
        target_density: f32,
        viscosity: f32,
        relax_factor: f32,
        dt: f32,
        density_solver_iterations: u32,
        divergence_solver_iterations: u32,
        gravity: Vec3,
        box_min: Vec3,
        box_max: Vec3,
        grid_res: IVec3,
    ) -> Self {
        Self {
            particle_radius,
            particle_mass,
            smoothing_radius,
            target_density,
            viscosity,
            relax_factor,
            dt,
            density_solver_iterations,
            divergence_solver_iterations,
            piston_velocity: 0.0,
            _padding: [0.0; 2],
            gravity: [gravity.x, gravity.y, gravity.z, 0.0],
            box_min: [box_min.x, box_min.y, box_min.z, 0.0],
            box_max: [box_max.x, box_max.y, box_max.z, 0.0],
            grid_res: [grid_res.x, grid_res.y, grid_res.z, 0],
            attractor: [0.0; 4],
            attractor_radius: 0.0,
            _padding_end: [0.0; 3],
        }
    }
}

pub struct ParticleGenerator;

impl ParticleGenerator {
    pub fn generate_volume(
        origin: Vec3,
        width: f32,
        height: f32,
        depth: f32,
        radius: f32,
        density: f32,
        spacing: f32,
        jitter: f32,
    ) -> (Vec<[f32; 3]>, f32) {
        let count_x = (width / spacing).floor() as usize;
        let count_y = (height / spacing).floor() as usize;
        let count_z = (depth / spacing).floor() as usize;

        let mut positions = Vec::with_capacity(count_x * count_y * count_z);
        
        let offset = origin + Vec3::splat(radius);

        for x in 0..count_x {
            for y in 0..count_y {
                for z in 0..count_z {
                    let jitter = (rand::random::<f32>() - 0.5) * jitter;

                    let px = offset.x + (x as f32 * spacing) + jitter;
                    let py = offset.y + (y as f32 * spacing) + jitter;
                    let pz = offset.z + (z as f32 * spacing) + jitter;

                    positions.push([px, py, pz]); 
                }
            }
        }
        
        let particle_volume = spacing.powi(3);
        let mass = density * particle_volume;

        (positions, mass)
    }
    pub fn generate_cube(
        num_per_axis: usize,
        centre: Vec3,
        size: f32,
        jitter_strength: f32,
        target_density: f32,
    ) -> ( Vec<[f32; 3]>, f32, f32 ) {
        let count = num_per_axis * num_per_axis * num_per_axis;
        let mut positions = Vec::with_capacity(count);
        let mut rng = rand::rng();

        let spacing = size / (num_per_axis as f32).max(1.0);
        let volume_per_particle = spacing.powi(3);
        let mass = target_density * volume_per_particle;

        for x in 0..num_per_axis {
            for y in 0..num_per_axis {
                for z in 0..num_per_axis {
                    let tx = x as f32 / (num_per_axis as f32 - 1.0).max(1.0);
                    let ty = y as f32 / (num_per_axis as f32 - 1.0).max(1.0);
                    let tz = z as f32 / (num_per_axis as f32 - 1.0).max(1.0);

                    let px = (tx - 0.5) * size + centre.x;
                    let py = (ty - 0.5) * size + centre.y;
                    let pz = (tz - 0.5) * size + centre.z;

                    let jitter = Vec3::new(
                        rng.random_range(-1.0..1.0),
                        rng.random_range(-1.0..1.0),
                        rng.random_range(-1.0..1.0),
                    ) * jitter_strength;

                    let pos = Vec3::new(px, py, pz) + jitter;

                    positions.push(pos.to_array());
                }
            }
        }

        ( positions, mass, spacing )
    }
}