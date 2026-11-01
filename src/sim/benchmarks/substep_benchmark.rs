#![cfg(test)]
//
// Substep cost benchmark on the production scene (`Scene::new`).
//
// For each sort algorithm it reports, per DFSPH substep:
//   * CPU time spent recording commands into an `AutoCommandBufferBuilder`,
//   * GPU time (submit + fence wait wall time of a batch of substeps),
//   * GPU time when one pre-recorded command buffer is resubmitted.
//
// Run manually with:
//     cargo test --release -- --ignored substep_benchmark --nocapture

use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::device::Queue;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType};
use vulkano::sync::{self, GpuFuture, PipelineStage};

use super::convergence_benchmark::make_context;
use crate::scene::Scene;
use crate::sim::particle_buffers::ParticleBuffers;
use crate::sim::pipelines::{ComputePipelines, PressureSolve, SolverSettings, SortAlgorithm};

const SUBSTEPS_PER_BATCH: u32 = 20;
const WARMUP_BATCHES: u32 = 3;
const TIMED_BATCHES: u32 = 7;

fn record_batch(
    cb_alloc: &Arc<StandardCommandBufferAllocator>,
    queue: &Arc<Queue>,
    pipelines: &ComputePipelines,
    usage: CommandBufferUsage,
    solver: &SolverSettings,
) -> Arc<PrimaryAutoCommandBuffer> {
    let mut builder =
        AutoCommandBufferBuilder::primary(cb_alloc.clone(), queue.queue_family_index(), usage)
            .unwrap();
    for _ in 0..SUBSTEPS_PER_BATCH {
        pipelines.record_substep(&mut builder, solver);
    }
    builder.build().unwrap()
}

fn submit_and_wait(queue: &Arc<Queue>, cb: Arc<PrimaryAutoCommandBuffer>) {
    sync::now(queue.device().clone())
        .then_execute(queue.clone(), cb)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
}

/// GPU time of every pass of one substep, measured with timestamp queries.
/// Mirrors `ComputePipelines::record_substep`.
fn profile_substep(
    cb_alloc: &Arc<StandardCommandBufferAllocator>,
    queue: &Arc<Queue>,
    pipelines: &ComputePipelines,
    solver: &SolverSettings,
) -> Vec<(&'static str, f64)> {
    const MAX_QUERIES: u32 = 32;
    let device = queue.device().clone();
    let pool = QueryPool::new(
        device.clone(),
        QueryPoolCreateInfo {
            query_count: MAX_QUERIES,
            ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
        },
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        cb_alloc.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    // SAFETY: the pool is fresh, reset before use, and every query index is
    // written at most once and read only after the fence wait below.
    unsafe {
        builder
            .reset_query_pool(pool.clone(), 0..MAX_QUERIES)
            .unwrap()
    };

    let mut labels: Vec<&'static str> = Vec::new();
    let mut mark = |builder: &mut AutoCommandBufferBuilder<_>, label: &'static str| {
        let idx = labels.len() as u32;
        // SAFETY: each index is reset above and written once before the pool is read.
        unsafe {
            builder
                .write_timestamp(pool.clone(), idx, PipelineStage::AllCommands)
                .unwrap()
        };
        labels.push(label);
    };

    mark(&mut builder, "start");
    pipelines.viscosity.dispatch(&mut builder);
    mark(&mut builder, "viscosity");
    pipelines.density_source_term.dispatch(&mut builder);
    mark(&mut builder, "density_source_term");
    pipelines.record_pressure_solve(&mut builder, &solver.density, PressureSolve::Density);
    mark(&mut builder, "density_solve");
    pipelines.pressure_integration.dispatch(&mut builder);
    mark(&mut builder, "pressure_integration");
    pipelines.neighbor_search.execute(&mut builder);
    mark(&mut builder, "neighbor_search");
    pipelines.density_alpha.dispatch(&mut builder);
    mark(&mut builder, "density_alpha");
    pipelines.divergence_source_term.dispatch(&mut builder);
    mark(&mut builder, "divergence_source_term");
    pipelines.record_pressure_solve(&mut builder, &solver.divergence, PressureSolve::Divergence);
    mark(&mut builder, "divergence_solve");
    pipelines.divergence_integration.dispatch(&mut builder);
    mark(&mut builder, "divergence_integration");

    submit_and_wait(queue, builder.build().unwrap());

    let count = labels.len();
    let mut ticks = vec![0u64; count];
    pool.get_results(0..count as u32, &mut ticks, QueryResultFlags::WAIT)
        .unwrap();
    let ns_per_tick = device.physical_device().properties().timestamp_period as f64;

    // Sum repeated passes (solver iterations) under one label, keep order.
    let mut out: Vec<(&'static str, f64)> = Vec::new();
    for i in 1..count {
        let ms = (ticks[i] - ticks[i - 1]) as f64 * ns_per_tick / 1e6;
        match out.iter_mut().find(|(l, _)| *l == labels[i]) {
            Some(entry) => entry.1 += ms,
            None => out.push((labels[i], ms)),
        }
    }
    out
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

#[test]
#[ignore]
fn substep_benchmark() {
    for particle_count in [25_000, Scene::DEFAULT_PARTICLES] {
        run_substep_benchmark(particle_count);
    }
}

fn run_substep_benchmark(particle_count: u32) {
    let ctx = make_context();
    let device = ctx.device().clone();
    let memory_allocator = ctx.memory_allocator().clone();
    let queue = ctx.graphics_queue().clone();

    let scene = Scene::new(particle_count);
    // Same solver settings as the application defaults.
    let solver = SolverSettings::default_for(&scene.sim_params);

    println!(
        "substep_benchmark: {} particles, dt={} s, {:?}, {} substeps/batch",
        scene.initial_positions.len(),
        scene.sim_params.dt,
        solver,
        SUBSTEPS_PER_BATCH,
    );
    println!(
        "{:>8}  {:>14}  {:>14}  {:>16}",
        "sort", "cpu_record_ms", "gpu_ms", "gpu_reused_cb_ms"
    );

    for sort_algorithm in [SortAlgorithm::Radix, SortAlgorithm::Bitonic] {
        let cb_alloc = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let ds_alloc = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        // Fresh state per algorithm so both start from the same initial block.
        let physics_data =
            ParticleBuffers::new(memory_allocator.clone(), scene.initial_positions.clone());
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
            scene.sim_params,
        )
        .unwrap();

        let mut pipelines = ComputePipelines::new(
            device.clone(),
            memory_allocator.clone(),
            ds_alloc,
            &physics_data,
            &sim_params_buffer,
        );
        pipelines.neighbor_search.sort_algorithm = sort_algorithm;

        let mut init = AutoCommandBufferBuilder::primary(
            cb_alloc.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        pipelines.record_init(&mut init);
        submit_and_wait(&queue, init.build().unwrap());

        let mut cpu_ms = Vec::new();
        let mut gpu_ms = Vec::new();
        for batch in 0..WARMUP_BATCHES + TIMED_BATCHES {
            let t0 = Instant::now();
            let cb = record_batch(
                &cb_alloc,
                &queue,
                &pipelines,
                CommandBufferUsage::OneTimeSubmit,
                &solver,
            );
            let t1 = Instant::now();
            submit_and_wait(&queue, cb);
            let t2 = Instant::now();
            if batch >= WARMUP_BATCHES {
                cpu_ms.push((t1 - t0).as_secs_f64() * 1e3 / SUBSTEPS_PER_BATCH as f64);
                gpu_ms.push((t2 - t1).as_secs_f64() * 1e3 / SUBSTEPS_PER_BATCH as f64);
            }
        }

        // Same work, recorded once and resubmitted: checks that vulkano allows
        // it and shows what the GPU costs without any per-frame recording.
        let reused = record_batch(
            &cb_alloc,
            &queue,
            &pipelines,
            CommandBufferUsage::MultipleSubmit,
            &solver,
        );
        let mut reused_ms = Vec::new();
        for batch in 0..WARMUP_BATCHES + TIMED_BATCHES {
            let t0 = Instant::now();
            submit_and_wait(&queue, reused.clone());
            if batch >= WARMUP_BATCHES {
                reused_ms.push(t0.elapsed().as_secs_f64() * 1e3 / SUBSTEPS_PER_BATCH as f64);
            }
        }

        let mut profile = profile_substep(&cb_alloc, &queue, &pipelines, &solver);
        for _ in 0..4 {
            let next = profile_substep(&cb_alloc, &queue, &pipelines, &solver);
            for (acc, (_, ms)) in profile.iter_mut().zip(next) {
                acc.1 = acc.1.min(ms);
            }
        }
        let total: f64 = profile.iter().map(|(_, ms)| ms).sum();
        println!("  per-pass GPU time, one substep ({sort_algorithm:?}, min of 5):");
        for (label, ms) in &profile {
            println!("    {label:<30} {ms:>8.3} ms  {:>5.1}%", ms / total * 100.0);
        }
        println!("    {:<30} {total:>8.3} ms", "total");
        let stats = super::convergence_benchmark::read_stats(&physics_data, scene.sim_params.dt);
        println!(
            "    iterations: density {} (avg err {:.3} kg/m³), \
             divergence {} (avg err {:.3} kg/m³/s)",
            stats.density_iterations,
            stats.density_error,
            stats.divergence_iterations,
            stats.divergence_error,
        );

        println!(
            "{:>8}  {:>14.3}  {:>14.3}  {:>16.3}",
            format!("{sort_algorithm:?}"),
            median(&mut cpu_ms),
            median(&mut gpu_ms),
            median(&mut reused_ms),
        );
    }
}

/// Solver iterations and errors over 3 s of the production scene (block
/// falls, hits the floor, settles), one sample per 25 substeps.
///
///     cargo test --release -- --ignored solver_iterations_over_time --nocapture
#[test]
#[ignore]
fn solver_iterations_over_time() {
    let ctx = make_context();
    let device = ctx.device().clone();
    let memory_allocator = ctx.memory_allocator().clone();
    let queue = ctx.graphics_queue().clone();

    let particle_count = std::env::var("FLUID_PARTICLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(25_000);
    let scene = Scene::new(particle_count);
    let solver = SolverSettings::default_for(&scene.sim_params);
    let dt = scene.sim_params.dt;

    let cb_alloc = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));
    let ds_alloc = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        StandardDescriptorSetAllocatorCreateInfo::default(),
    ));
    let physics_data =
        ParticleBuffers::new(memory_allocator.clone(), scene.initial_positions.clone());
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
        scene.sim_params,
    )
    .unwrap();
    let pipelines = ComputePipelines::new(
        device.clone(),
        memory_allocator.clone(),
        ds_alloc,
        &physics_data,
        &sim_params_buffer,
    );

    println!(
        "{:>7} {:>6}  {:>9} {:>12}  {:>9} {:>14}",
        "t [s]", "substep", "dens_it", "dens_err", "div_it", "div_err"
    );
    let total_substeps = (3.0 / dt) as u32;
    let mut sum_density_iters = 0u32;
    let mut sum_divergence_iters = 0u32;
    for substep in 0..total_substeps {
        let mut builder = AutoCommandBufferBuilder::primary(
            cb_alloc.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        if substep == 0 {
            pipelines.record_init(&mut builder);
        }
        pipelines.record_substep(&mut builder, &solver);
        submit_and_wait(&queue, builder.build().unwrap());

        let stats = super::convergence_benchmark::read_stats(&physics_data, dt);
        sum_density_iters += stats.density_iterations;
        sum_divergence_iters += stats.divergence_iterations;
        if substep % 25 == 0 {
            println!(
                "{:>7.3} {:>6}  {:>9} {:>12.3}  {:>9} {:>14.1}",
                substep as f32 * dt,
                substep,
                stats.density_iterations,
                stats.density_error,
                stats.divergence_iterations,
                stats.divergence_error,
            );
        }
    }
    println!(
        "mean iterations: density {:.2}, divergence {:.2}",
        sum_density_iters as f32 / total_substeps as f32,
        sum_divergence_iters as f32 / total_substeps as f32,
    );
}

/// Settles 25k particles, then turns the attractor on inside the water and
/// checks that particles gather around it.
///
///     cargo test --release -- --ignored attractor_gathers_water --nocapture
#[test]
#[ignore]
fn attractor_gathers_water() {
    let ctx = make_context();
    let device = ctx.device().clone();
    let memory_allocator = ctx.memory_allocator().clone();
    let queue = ctx.graphics_queue().clone();

    let mut scene = Scene::new(25_000);
    let solver = SolverSettings::default_for(&scene.sim_params);
    let cb_alloc = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));
    let ds_alloc = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        StandardDescriptorSetAllocatorCreateInfo::default(),
    ));
    let physics_data =
        ParticleBuffers::new(memory_allocator.clone(), scene.initial_positions.clone());
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
        scene.sim_params,
    )
    .unwrap();
    let pipelines = ComputePipelines::new(
        device.clone(),
        memory_allocator.clone(),
        ds_alloc,
        &physics_data,
        &sim_params_buffer,
    );

    let run = |substeps: u32, init: bool| {
        let mut builder = AutoCommandBufferBuilder::primary(
            cb_alloc.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        if init {
            pipelines.record_init(&mut builder);
        }
        for _ in 0..substeps {
            pipelines.record_substep(&mut builder, &solver);
        }
        submit_and_wait(&queue, builder.build().unwrap());
    };
    let p = &scene.sim_params;
    let centre = glam::Vec3::new(0.5 * (p.box_min[0] + p.box_max[0]), 0.6, 0.0);
    let radius = 0.6;
    let count_inside = || {
        physics_data
            .position_a
            .read()
            .unwrap()
            .iter()
            .filter(|q| glam::Vec3::new(q[0], q[1], q[2]).distance(centre) < radius * 0.5)
            .count()
    };

    run(400, true);
    let before = count_inside();

    scene.sim_params.attractor = [centre.x, centre.y, centre.z, 40.0];
    scene.sim_params.attractor_radius = radius;
    *sim_params_buffer.write().unwrap() = scene.sim_params;
    run(200, false);
    let after = count_inside();

    println!(
        "particles within {:.2} m of the attractor: {before} -> {after}",
        radius * 0.5
    );
    assert!(
        after > before,
        "attractor did not gather water: {before} -> {after}"
    );
}
