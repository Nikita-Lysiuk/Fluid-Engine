#![cfg(test)]
//
// DFSPH solver convergence benchmark.
//
// Measures steady-state density and divergence error as a function of the
// pressure-solver iteration count. Replicates the production substep loop
// exactly (see `Renderer::step`) but in a headless `VulkanoContext`. For each
// iteration count: fresh particle state, `WARMUP_SUBSTEPS` settle, then
// `MEASUREMENT_SUBSTEPS` are submitted one at a time and stats are read after
// each. The averaged errors land in `scripts/convergence.csv`.
//
// Marked `#[ignore]`; run manually with:
//     cargo test --release -p fluid_engine -- --ignored convergence --nocapture

use glam::{IVec3, Vec3};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::device::{DeviceFeatures, Queue};
use vulkano::instance::{InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::GpuFuture;
use vulkano_util::context::{VulkanoConfig, VulkanoContext};

use crate::entities::particle::{GpuPhysicsData, ParticleGenerator, SimulationParams};
use crate::renderer::pipelines::{ComputePipelines, ComputeStep};

// ── Configuration ────────────────────────────────────────────────────────────

const ITER_VALUES: &[u32] = &[1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20];
const WARMUP_SUBSTEPS: u32 = 50;
const MEASUREMENT_SUBSTEPS: u32 = 40;
const CSV_PATH: &str = "scripts/convergence.csv";

// Fixed-point scales — must match the constants in stats.comp / mod.rs.
const DENSITY_SCALE: f32 = 1.0;
const DIVERGENCE_SCALE: f32 = 10.0;

// ── Vulkan setup ─────────────────────────────────────────────────────────────

fn make_context() -> Arc<VulkanoContext> {
    let config = VulkanoConfig {
        instance_create_info: InstanceCreateInfo {
            enabled_extensions: InstanceExtensions {
                ext_debug_utils: true,
                ..InstanceExtensions::default()
            },
            ..Default::default()
        },
        device_features: DeviceFeatures {
            scalar_block_layout: true,
            buffer_device_address: true,
            shader_int64: true,
            ..DeviceFeatures::empty()
        },
        ..VulkanoConfig::default()
    };
    Arc::new(VulkanoContext::new(config))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn prepare_all_pipelines(
    pipelines: &mut ComputePipelines,
    ds_alloc: Arc<StandardDescriptorSetAllocator>,
    physics_data: &GpuPhysicsData,
    sim_params: &Subbuffer<SimulationParams>,
) {
    pipelines.neighbor_search.prepare(ds_alloc.clone(), physics_data, sim_params);
    pipelines.density_alpha.prepare(ds_alloc.clone(), physics_data, sim_params);
    pipelines.viscosity.prepare(ds_alloc.clone(), physics_data, sim_params);
    pipelines.density_source_term.prepare(ds_alloc.clone(), physics_data, sim_params);
    pipelines.pressure_force.prepare(ds_alloc.clone(), physics_data, sim_params);
    pipelines.pressure_update.prepare(ds_alloc.clone(), physics_data, sim_params);
    pipelines.pressure_integration.prepare(ds_alloc.clone(), physics_data, sim_params);
    pipelines.divergence_source_term.prepare(ds_alloc.clone(), physics_data, sim_params);
    pipelines.divergence_integration.prepare(ds_alloc.clone(), physics_data, sim_params);
    pipelines.stats.prepare(ds_alloc, physics_data, sim_params);
}

// Records and submits `n_substeps` substeps in a single command buffer. The
// substep body mirrors `Renderer::step()` exactly: viscosity, density source,
// density solver loop, pressure integration, neighbor search rebuild,
// density_alpha, divergence source, divergence solver loop, divergence
// integration. Stats is recorded once at the end of the batch.
fn submit_substeps(
    cb_alloc: &Arc<StandardCommandBufferAllocator>,
    queue: &Arc<Queue>,
    pipelines: &ComputePipelines,
    iter_count: u32,
    n_substeps: u32,
    is_first_call: bool,
) {
    let mut builder = AutoCommandBufferBuilder::primary(
        cb_alloc.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    // Init pass: populate spatial hash and densities/factors for the initial
    // state. Only required for the very first command buffer of a given
    // physics_data, since subsequent substeps' internal neighbor_search keeps
    // the structure synchronized.
    if is_first_call {
        pipelines.neighbor_search.execute(&mut builder);
        pipelines.density_alpha.execute(&mut builder);
    }

    for _ in 0..n_substeps {
        // Phase 1 — density correction.
        pipelines.viscosity.execute(&mut builder);
        pipelines.density_source_term.execute(&mut builder);
        for _ in 0..iter_count {
            pipelines.pressure_force.execute(&mut builder);
            pipelines.pressure_update.execute(&mut builder);
        }
        pipelines.pressure_integration.execute(&mut builder);

        // Phase 2 — divergence correction (rebuild neighbours first, since
        // pressure_integration just moved every particle).
        pipelines.neighbor_search.execute(&mut builder);
        pipelines.density_alpha.execute(&mut builder);
        pipelines.divergence_source_term.execute(&mut builder);
        for _ in 0..iter_count {
            pipelines.pressure_force.execute(&mut builder);
            pipelines.pressure_update.execute(&mut builder);
        }
        pipelines.divergence_integration.execute(&mut builder);
    }

    pipelines.stats.execute(&mut builder);

    let cb = builder.build().unwrap();
    cb.execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
}

fn read_stats(physics_data: &GpuPhysicsData, n_particles: u32) -> (f32, f32) {
    let stats = physics_data.stats_buffer.read().unwrap();
    let n = n_particles as f32;
    let density_err = stats[1] as f32 / (DENSITY_SCALE * n);
    let div_err = stats[2] as f32 / (DIVERGENCE_SCALE * n);
    (density_err, div_err)
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f32>() / xs.len() as f32
    }
}

fn write_csv(path: &str, rows: &[(u32, f32, f32)]) {
    let p = Path::new(path);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).ok();
    }
    let mut f = fs::File::create(p).unwrap();
    writeln!(f, "iters,density_error,divergence_error").unwrap();
    for &(n, de, ve) in rows {
        writeln!(f, "{},{:.6},{:.6}", n, de, ve).unwrap();
    }
}

// ── Test ─────────────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn solver_convergence_benchmark() {
    let ctx = make_context();
    let device = ctx.device().clone();
    let memory_allocator = ctx.memory_allocator().clone();
    let queue = ctx.graphics_queue().clone();

    let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));
    let ds_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        StandardDescriptorSetAllocatorCreateInfo::default(),
    ));

    // Simulation parameters — match `Scene::new()` defaults but smaller block
    // to keep the benchmark in the seconds-not-minutes range.
    let particle_radius = 0.020f32;
    let target_density = 1000.0f32;
    let smoothing_radius = particle_radius * 4.0;
    let spacing = particle_radius * 2.0;
    let dt = 0.005f32;
    let viscosity_coeff = 0.15f32;
    let relax_factor = 0.5f32;

    let box_min = Vec3::new(-1.5, 0.0, -1.0);
    let box_max = Vec3::new(0.8, 4.0, 1.0);
    let spawn_pos = Vec3::new(-0.8, 0.5, -0.4);

    let (initial_positions, particle_mass) = ParticleGenerator::generate_volume(
        spawn_pos,
        0.8, // water_width
        0.8, // water_height
        0.5, // water_depth
        particle_radius,
        target_density,
        spacing,
        0.01, // jitter
    );

    let n_particles = initial_positions.len() as u32;
    println!(
        "convergence_benchmark: {} particles, {} iter values, warmup={}, measure={}",
        n_particles,
        ITER_VALUES.len(),
        WARMUP_SUBSTEPS,
        MEASUREMENT_SUBSTEPS
    );

    let mut results: Vec<(u32, f32, f32)> = Vec::with_capacity(ITER_VALUES.len());

    for &iter_count in ITER_VALUES {
        let sim_params = SimulationParams::new(
            particle_radius,
            particle_mass,
            smoothing_radius,
            target_density,
            viscosity_coeff,
            relax_factor,
            dt,
            iter_count, // density iters
            iter_count, // divergence iters
            Vec3::new(0.0, -9.81, 0.0),
            box_min,
            box_max,
            IVec3::new(128, 128, 128),
        );

        // Fresh particle state per iteration-count test.
        let physics_data =
            GpuPhysicsData::new(memory_allocator.clone(), initial_positions.clone());
        let sim_params_buffer = Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            sim_params,
        )
        .unwrap();

        let mut pipelines = ComputePipelines::new(
            device.clone(),
            memory_allocator.clone(),
            physics_data.grid_entries.len() as u32,
        );
        prepare_all_pipelines(
            &mut pipelines,
            ds_allocator.clone(),
            &physics_data,
            &sim_params_buffer,
        );

        // Warmup batch — one big command buffer to amortize submission overhead.
        submit_substeps(
            &cb_allocator,
            &queue,
            &pipelines,
            iter_count,
            WARMUP_SUBSTEPS,
            true,
        );

        // Measurement: one substep per submit so we can read stats after each.
        let mut density_errs = Vec::with_capacity(MEASUREMENT_SUBSTEPS as usize);
        let mut div_errs = Vec::with_capacity(MEASUREMENT_SUBSTEPS as usize);

        for _ in 0..MEASUREMENT_SUBSTEPS {
            submit_substeps(&cb_allocator, &queue, &pipelines, iter_count, 1, false);
            let (de, ve) = read_stats(&physics_data, n_particles);
            density_errs.push(de);
            div_errs.push(ve);
        }

        let avg_density = mean(&density_errs);
        let avg_div = mean(&div_errs);

        println!(
            "  iter_count={:>3}  avg_density_error={:>10.4}  avg_divergence_error={:>10.4}",
            iter_count, avg_density, avg_div
        );
        results.push((iter_count, avg_density, avg_div));
    }

    write_csv(CSV_PATH, &results);
    println!("wrote {}", CSV_PATH);
}
