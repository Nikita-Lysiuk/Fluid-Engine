# Refactoring plan: module structure and Rust style

This document is a complete, step-by-step plan for restructuring the Rust code
of Fluid Engine. It is written to be executed by someone (or a model) who has
not seen the code before. Read it to the end before starting.

## 1. Goal and non-goals

**Goal.** Make the code easy to navigate and idiomatic Rust:

- modules organised by *what the code is about* (simulation, rendering,
  input, GPU plumbing) instead of by kind of object (`entities`, `utils`,
  `commands`, `errors`);
- physics code no longer lives inside the renderer;
- no Java-style patterns: no traits whose only purpose is to group methods,
  no unit structs used as namespaces, no Command objects;
- the rules of the Google Rust style guide
  (https://google.github.io/styleguide/rustguide.html) that apply to an
  application crate (listed in section 4).

**Non-goals. Do not do any of these:**

- Do not change behaviour. The simulation, rendering, UI and controls must
  work exactly as before. This is a move-and-rename refactoring.
- Do not edit shaders (`shaders/**`), except for the `path:` strings in
  `shader!` macros if a Rust file moves (the paths are relative to the crate
  root and normally do **not** need to change).
- Do not run `cargo fmt` on the whole project and do not add clippy or
  formatting checks. That is planned as a separate change. New files may be
  written in rustfmt style; moved code keeps its formatting.
- Do not touch `docs/Fluid_Simulation/**` (the thesis), `scripts/**` or
  `assets/**`.
- Do not change dependency versions in `Cargo.toml`.

## 2. Things that must not break (read carefully)

The code has invariants shared between Rust and GLSL that the compiler cannot
check. When moving code, keep these byte-for-byte identical:

| Invariant | Rust side | GLSL side |
|---|---|---|
| Layout of the simulation parameters (std140) | `SimulationParams` (`#[repr(C, align(16))]`, field order, padding fields) | `SimulationParams` block in `shaders/include/common.glsl` |
| Max neighbours per particle | `MAX_NEIGHBORS` | `MAX_NEIGHBORS` in `common.glsl` |
| Stats buffer layout | `stats_layout` module | `stats.comp`, `solver.glsl` (`SolverState`) |
| Descriptor binding numbers | `WriteDescriptorSet::buffer(N, ...)` in every pass | `layout(binding = N)` in the matching shader |
| Push constant structs | generated `cs::PushConstants` | `layout(push_constant)` blocks |
| Ping-pong buffers | which pass reads `position_a/b`, `velocity_a/b` | the shader that pass runs |

Ping-pong data flow per substep (do not change which buffer each pass binds):

```
neighbor_search: position_a, velocity_a, warm_start --sort/reorder--> position_b, velocity_b, warm_start_sorted (copied back to warm_start)
viscosity:            reads velocity_b, position_b      -> writes velocity_a
density_source_term:  reads position_b, velocity_a      -> source_terms, pressures
pressure_force/update (Jacobi, indirect dispatch)        -> pressure_accelerations, pressures
pressure_integration: reads position_b, velocity_a      -> writes velocity_a, position_a, warm_start.x
neighbor_search again, density_alpha (builds neighbor_list / neighbor_counts)
divergence_source_term, pressure_force/update, divergence_integration (velocity_a, colors, warm_start.y)
```

Other things that are easy to break:

- `ComputePipelines::record_substep` is recorded **once** and replayed
  (`Renderer::substep_command_buffer`, `CommandBufferUsage::SimultaneousUse`).
  Anything that changes per frame must stay in the uniform buffer, written
  with `update_buffer` in the per-frame command buffer. Do not move per-frame
  values into push constants of the substep passes.
- `present(gui_future, true)` must stay `true` (see the TODO next to it).
- The `#[global_allocator]` Tracy allocator in `src/bin/main.rs` stays.
- Environment variables used by the benchmark mode keep their names:
  `FLUID_PARTICLES`, `FLUID_BENCH_SECONDS`, `FLUID_BENCH_MAX_SUBSTEPS`,
  `FLUID_BENCH_RENDER`.
- Test names stay the same (`substep_benchmark`, `solver_iterations_over_time`,
  `solver_convergence_benchmark`, `rest_density_diagnostic`,
  `density_after_warmup_diagnostic`, `cfl_comparison`, `sort_benchmark`, and
  the unit tests in `sorter.rs`) because the README and scripts refer to them.
- The ignored tests write CSV files into `scripts/`. If you run them, restore
  those files afterwards with `git checkout -- scripts/`.

## 3. How to check that nothing broke

Do this **before** the first change and keep the output as the baseline. Run
it again after every step. All commands are run from the repository root.

```sh
cargo build --release                      # must succeed; no new warnings
cargo test --release                       # 5 unit tests pass
cargo test --release -- --ignored solver_iterations_over_time --nocapture
cargo test --release -- --ignored substep_benchmark --nocapture
git checkout -- scripts/
```

- `solver_iterations_over_time` prints `mean iterations: density X, divergence Y`.
  The initial particle positions are randomly jittered, so the per-sample
  rows differ between runs, but the means are stable (two runs of the same
  code: `9.85 / 8.73` and `9.85 / 8.76`). After a step the means must stay
  within ±3% of the baseline.
- `substep_benchmark` prints per-pass GPU times and the iterations of the
  last substep for 25k and 100k particles. The total per substep must stay
  within ±10% of the baseline (the laptop GPU heats up; if numbers drift, let
  it cool down and rerun the baseline).
- Run the application once at the end of every step and check by hand:
  `cargo run --release --bin main`, then:
  - water falls and settles, the left wall moves, "Moving wall" stops it;
  - holding the right mouse button pulls water into a ball, the mouse wheel
    moves the ball closer/further;
  - WASD/QE move the camera, the mouse rotates it, Esc releases the cursor;
  - the UI sliders work; switching Raymarching/Particles works;
  - the frame benchmark still works:
    `FLUID_PARTICLES=25000 FLUID_BENCH_SECONDS=5 ./target/release/main.exe`
    prints a `BENCH ...` line and exits.

If any check fails, revert the step (`git checkout -- .` or `git reset --hard`
to the previous step's commit) and redo it in smaller pieces.

Commit after every step with a message `refactor: <step title>`. Every commit
must build and pass the checks.

## 4. Style rules to apply (from the Google Rust style guide)

Apply these to the code you move or touch. They are the rules relevant to
this crate; the guide's Chromium/Android/Fuchsia-only rules do not apply.

1. **Name files and leaf modules after their primary item.** `struct Camera`
   lives in `camera.rs`; `struct NeighborSearch` in `neighbor_search.rs`.
2. **Non-leaf modules contain only module declarations and re-exports.**
   `mod.rs` files must not contain structs, functions or `impl` blocks (a
   short re-export list is fine). Move the code into a named leaf file.
3. **Don't use traits merely to organise code.** Define a trait only when
   code is generic over it or uses `dyn Trait`. This removes `ComputeStep`,
   `Command` and `Actor` (see the steps).
4. **No unit structs as namespaces.** `ParticleGenerator` with associated
   functions becomes free functions.
5. **One module per `use` statement**, no `::` inside braces:
   `use vulkano::buffer::{Buffer, Subbuffer};` is fine,
   `use vulkano::{buffer::Buffer, device::Device};` is not.
6. **No glob imports** (`use foo::*`), except `use super::*` inside
   `#[cfg(test)] mod tests`.
7. **Types are imported and used unqualified** (`Subbuffer`, not
   `vulkano::buffer::Subbuffer`), except where names collide.
8. **Prefer `expect("why this cannot fail")` over `unwrap()`** for invariants
   such as "descriptor set layout 0 exists" or "prepare() was called".
   Do not rewrite every `unwrap()` on Vulkan object creation in one go; do it
   in the files you are already moving.
9. **Every `unsafe` block gets a `// SAFETY:` comment** explaining why it is
   sound (most are `dispatch`/`draw` calls: "all descriptor sets and push
   constants required by the pipeline are bound above").
10. **Avoid abbreviations** in new names (`descriptor_set_allocator`, not
    `ds_alloc`; `command_buffer`, not `cb`). Keep established domain names
    (`dt`, `rho`, `sph`, `cfl`).
11. **Comments say what the code does not.** Delete comments that repeat the
    code. Keep and move the physics comments (they explain the method).
12. **Getters are `foo()`, not `get_foo()`** (e.g. `Controller::get_active_commands`
    disappears anyway; `Camera::get_view_matrix` becomes `view_matrix`).
13. **Prefer named fields over tuples** in new or changed signatures, e.g. a
    returned `(Vec<[f32; 3]>, f32)` becomes a small struct.

## 5. Target layout

```
src/
  bin/main.rs                  unchanged except imports
  lib.rs                       module declarations only
  app.rs                       App: winit ApplicationHandler (was core/engine.rs)
  app_error.rs                 AppError (was errors/application_error.rs)
  input.rs                     Input: keyboard/mouse state -> camera movement (was core/controller.rs + commands/*)
  frame_timer.rs               FrameTimer (was utils/fps_counter.rs)
  frame_benchmark.rs           FrameBenchmark (was utils/frame_benchmark.rs)
  scene.rs                     Scene, Attractor (was core/scene.rs)
  camera.rs                    Camera (+ CameraGpu, CameraBuffers) (was entities/camera.rs)
  sim/
    mod.rs                     declarations + re-exports only
    params.rs                  SimulationParams
    particle_buffers.rs        ParticleBuffers (was GpuPhysicsData), Entry, MAX_NEIGHBORS, stats_layout
    spawn.rs                   fn spawn_block(...) (was ParticleGenerator::generate_volume)
    collision_box.rs           CollisionBox (simulation side of entities/collision.rs)
    compute_pass.rs            ComputePass: one generic compute pass (replaces ComputeStep and 9 near-identical files)
    neighbor_search.rs         NeighborSearch (hash, sort, offsets, reorder)
    sort/
      mod.rs                   declarations + `pub use` of SortAlgorithm, BitonicSorter, RadixSorter
      sort_algorithm.rs        enum SortAlgorithm
      bitonic_sorter.rs        BitonicSorter (was GpuSorter)
      radix_sorter.rs          RadixSorter
      sorter_tests.rs          the unit tests and sort_benchmark from sorter.rs (#[cfg(test)])
    solver_settings.rs         SolveSettings, SolverSettings, PressureSolve
    simulation.rs              Simulation: all passes + record_init/record_substep/record_pressure_solve (was ComputePipelines)
    stats.rs                   SimulationStats: reading the stats readback ring, CFL time step
    benchmarks/
      mod.rs                   #[cfg(test)] declarations only
      common.rs                make_context, read_stats, SolverStats, prepare helpers
      substep_benchmark.rs
      convergence_benchmark.rs
  render/
    mod.rs                     declarations + re-exports only
    renderer.rs                Renderer (frame orchestration only; physics via Simulation)
    gpu_context.rs             Vulkano context/window/allocators setup (from Renderer::new)
    shader_loader.rs           load_shader_entry_point (was utils/shader_loader.rs)
    model_vertex.rs            ModelVertex
    frame_resources.rs         per-frame buffers (was renderer/resources.rs: GpuSceneResources)
    particle_renderer.rs       GpuRenderData + PointPipeline (point sprites)
    sky.rs                     SkyData + SkyPipeline
    fluid_raymarch.rs          WaterRenderer + WaterRenderPipeline + DensityTexturePipeline (splat)
    box_outline.rs             CollisionBoxData + CollisionPipeline (wireframe box)
    ui.rs                      AppUi (was AppUI), RenderMode
  constants.rs                 WINDOW_TITLE, MAX_FRAMES_IN_FLIGHT, PREFERRED_FPS
```

Notes on the layout:

- `sim/` must not depend on `render/`. `render/` depends on `sim/`. `app.rs`
  depends on both. If a `sim` file needs something from `render` (it should
  not), the boundary is wrong; stop and re-check.
- Directory nesting is fine for this crate (the guide's "flat layout" rule is
  Chromium-specific).
- If a target file would be over ~400 lines, split it by responsibility and
  name each part after its main item. The only expected large files are the
  test/benchmark files.

## 6. Mapping from current to target

| Current | Target |
|---|---|
| `src/core/engine.rs` (`Engine`, `load_icon`) | `src/app.rs` (`App`, `load_icon`) |
| `src/core/scene.rs` | `src/scene.rs` |
| `src/core/controller.rs`, `src/commands/mod.rs`, `src/commands/camera_commands.rs` | `src/input.rs` |
| `src/errors/application_error.rs`, `src/errors/mod.rs` | `src/app_error.rs` |
| `src/utils/fps_counter.rs` | `src/frame_timer.rs` |
| `src/utils/frame_benchmark.rs` | `src/frame_benchmark.rs` |
| `src/utils/constants.rs` | `src/constants.rs` |
| `src/utils/shader_loader.rs` | `src/render/shader_loader.rs` |
| `src/entities/mod.rs` (`ModelVertex`) | `src/render/model_vertex.rs` |
| `src/entities/mod.rs` (`Actor`) | removed (see step 6) |
| `src/entities/camera.rs` | `src/camera.rs` |
| `src/entities/collision.rs` (`CollisionBox`) | `src/sim/collision_box.rs` |
| `src/entities/collision.rs` (`CollisionBoxData`) + `renderer/pipelines/collision_pipeline.rs` | `src/render/box_outline.rs` |
| `src/entities/particle.rs` (`SimulationParams`) | `src/sim/params.rs` |
| `src/entities/particle.rs` (`GpuPhysicsData`, `Entry`, `MAX_NEIGHBORS`, `stats_layout`) | `src/sim/particle_buffers.rs` (`ParticleBuffers`) |
| `src/entities/particle.rs` (`ParticleGenerator::generate_volume`) | `src/sim/spawn.rs` (`spawn_block`) |
| `src/entities/particle.rs` (`ParticleGenerator::generate_cube`) | removed (unused, compiler warns) |
| `src/entities/particle.rs` (`PositionVertex`, `ColorVertex`, `AttributeVertex`, `GpuRenderData`) + `renderer/pipelines/point_pipeline.rs` | `src/render/particle_renderer.rs` |
| `src/entities/sky.rs` + `renderer/pipelines/sky_pipeline.rs` | `src/render/sky.rs` |
| `src/entities/water.rs` + `renderer/pipelines/water_pipeline.rs` + `renderer/pipelines/density_texture.rs` | `src/render/fluid_raymarch.rs` |
| `src/renderer/mod.rs` (`Renderer`) | `src/render/renderer.rs` (+ `gpu_context.rs`, `sim/simulation.rs`, `sim/stats.rs`) |
| `src/renderer/resources.rs` | `src/render/frame_resources.rs` |
| `src/renderer/ui.rs` | `src/render/ui.rs` |
| `src/renderer/pipelines/mod.rs` (`ComputeStep`) | removed, replaced by `src/sim/compute_pass.rs` |
| `src/renderer/pipelines/mod.rs` (`ComputePipelines`, `SolverSettings`, ...) | `src/sim/simulation.rs`, `src/sim/solver_settings.rs` |
| `src/renderer/pipelines/mod.rs` (`Pipelines`) | `src/render/renderer.rs` or a `render/graphics_pipelines.rs` leaf |
| `renderer/pipelines/{viscosity, density_alpha, density_source_term, divergence_source_term, pressure_force_pipeline, pressure_update_pipeline, pressure_integration_pipeline, divergence_integration, stats_pipeline}.rs` | `ComputePass` instances created in `src/sim/simulation.rs` |
| `renderer/pipelines/solver_check.rs` | `ComputePass` instance in `simulation.rs` |
| `renderer/pipelines/neighbor_search.rs` | `src/sim/neighbor_search.rs` |
| `renderer/pipelines/sorter.rs` | `src/sim/sort/*` |
| `renderer/pipelines/convergence_benchmark.rs`, `substep_benchmark.rs` | `src/sim/benchmarks/*` |

## 7. Steps

Each step is one commit. Do them in this order; later steps assume earlier
ones. Run section 3 after each.

### Step 0. Baseline

Run the checks of section 3 on the unchanged code and save the output
(`baseline.txt`, not committed). Create a branch `refactor/structure` from
the branch this plan lives on.

### Step 1. Move files without changing code

Pure `git mv` plus `mod`/`use` path fixes. No renames of types yet.

1. Create `src/sim/`, `src/render/` and the target files as empty modules.
2. Move whole files where the mapping is one-to-one:
   `core/scene.rs → scene.rs`, `entities/camera.rs → camera.rs`,
   `utils/constants.rs → constants.rs`, `utils/fps_counter.rs → frame_timer.rs`,
   `utils/frame_benchmark.rs → frame_benchmark.rs`,
   `utils/shader_loader.rs → render/shader_loader.rs`,
   `errors/application_error.rs → app_error.rs`,
   `renderer/ui.rs → render/ui.rs`, `renderer/resources.rs → render/frame_resources.rs`,
   `renderer/pipelines/neighbor_search.rs → sim/neighbor_search.rs`,
   `renderer/pipelines/sorter.rs → sim/sort/mod.rs` (split in step 5),
   `renderer/pipelines/convergence_benchmark.rs → sim/benchmarks/convergence_benchmark.rs`,
   `renderer/pipelines/substep_benchmark.rs → sim/benchmarks/substep_benchmark.rs`.
3. Fix every `use crate::...` path. Use `git grep -n "crate::utils\|crate::entities\|crate::core\|crate::errors\|crate::commands\|crate::renderer"`
   until it returns nothing relevant.
4. Shader paths inside `shader!` macros are relative to `Cargo.toml`, so they
   stay the same. Some use backslashes (`"shaders\\compute\\x.comp"`); leave
   them, or change all of them to forward slashes in one go.
5. `src/bin/main.rs` imports `fluid_engine::core::engine::Engine`; keep it
   compiling (re-export `Engine` from `lib.rs` temporarily if needed).

### Step 2. Split `entities/particle.rs`

- `SimulationParams` → `sim/params.rs`. Keep the struct exactly as it is
  (field order, `_padding`, `_padding_end`, `#[repr(C, align(16))]`). Keep
  `SimulationParams::new` with the same parameters.
- `GpuPhysicsData` → `sim/particle_buffers.rs`, renamed `ParticleBuffers`
  (update all uses). `Entry`, `MAX_NEIGHBORS` and `mod stats_layout` move
  with it. Rename the private `create_buffer` helper to
  `new_device_buffer` and document it.
- `ParticleGenerator::generate_volume` → `sim/spawn.rs` as
  `pub fn spawn_block(origin, size: Vec3, particle_radius, rest_density, spacing, jitter) -> SpawnedBlock`
  where `SpawnedBlock { positions: Vec<[f32; 3]>, particle_mass: f32 }`.
  Keep the loop and the jitter exactly as they are. Update `Scene::new` and the
  tests that call it.
- Delete `ParticleGenerator::generate_cube` (unused).
- Vertex types and `GpuRenderData` → `render/particle_renderer.rs`, together
  with `PointPipeline` from `renderer/pipelines/point_pipeline.rs`.

### Step 3. Replace `ComputeStep` with `ComputePass`

Today every compute pass is a ~60-line file implementing `ComputeStep`
(`load_shader_module`, `from_pipeline`, `new`, `prepare`, `execute`), and
`NeighborSearch` implements it with `unimplemented!()` bodies. Replace this
with one struct:

```rust
/// A compute shader with its descriptor set, dispatched over all particles.
pub struct ComputePass {
    name: &'static str,
    pipeline: Arc<ComputePipeline>,
    descriptor_set: Arc<DescriptorSet>,
    group_count: u32,
}

impl ComputePass {
    /// `bindings` are the descriptor writes of set 0, `group_count` the
    /// number of 256-thread workgroups.
    pub fn new(
        device: Arc<Device>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        name: &'static str,
        shader: EntryPoint,
        bindings: impl IntoIterator<Item = WriteDescriptorSet>,
        group_count: u32,
    ) -> Self;

    /// Binds the pipeline and descriptor set, pushes `push_constants` if the
    /// shader has any.
    pub fn bind<Cb, Pc: BufferContents>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, push_constants: Option<Pc>);

    pub fn dispatch<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>);
    pub fn dispatch_indirect<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, args: Subbuffer<[DispatchIndirectCommand]>);
}
```

Then:

1. Keep one `mod <name>_shader { vulkano_shaders::shader!(...) }` per shader,
   all in one file `sim/shaders.rs` (generated code needs a module each).
2. In `Simulation::new` (step 4) create one `ComputePass` per former pipeline
   file, copying the binding lists **exactly** (binding numbers and buffers)
   from the old `prepare` methods:
   `viscosity`, `density_alpha`, `density_source_term`, `pressure_force`,
   `pressure_update`, `pressure_integration`, `divergence_source_term`,
   `divergence_integration`, `stats`, `solver_check`.
3. Passes with push constants: `pressure_update` (`is_divergence_solve`),
   `solver_check` (`max_avg_error`, `min_iterations`, `num_particles`). Use the
   generated `PushConstants` structs.
4. Special cases to keep:
   - `stats`: before dispatching, `fill_buffer(stats_buffer.slice(0..1), 0)`.
   - `solver_check` dispatches a single workgroup (`group_count = 1`).
   - `pressure_force` and `pressure_update` are dispatched indirectly inside
     the solver loop and directly for the first `pressure_force`.
5. Delete the old pipeline files and the `ComputeStep` trait.
6. `NeighborSearch` keeps its own struct (it has several shaders and the
   sorters) but stops implementing `ComputeStep`: give it inherent
   `new(device, memory_allocator, descriptor_set_allocator, buffers, sim_params)`
   and `record(&self, builder)` methods. Its three passes become
   `ComputePass` fields.
7. `DensityTexturePipeline` (raymarch splatting) is a render concern: move
   it to `render/fluid_raymarch.rs` as a `ComputePass` (it binds an image,
   `WriteDescriptorSet::image_view`).

This step changes the most code. Do it pass by pass and run the checks after
the first two passes to catch mistakes early.

### Step 4. `Simulation`: physics out of the renderer

Create `sim/simulation.rs` with `pub struct Simulation` holding the
`ComputePass`es, `NeighborSearch`, the buffers (`ParticleBuffers`), the
`sim_params` uniform buffer and the substep command-buffer cache. Move into
it, unchanged in behaviour:

- from `ComputePipelines`: `record_init`, `record_substep`,
  `record_pressure_solve` and the solver-state buffers;
- from `Renderer`: `needs_init`, `substep_cache`, `substep_command_buffer`,
  the substep-count logic (`max_substeps_per_frame`, `ceil(frame_dt / dt)`),
  the `update_buffer` of the simulation parameters, and the tail command
  buffer (stats dispatch + copy to the readback ring + copy of positions and
  colours to the render buffers);
- from `Renderer::step`: the stats reading and the CFL time step go to
  `sim/stats.rs` (`SimulationStats::read(...)`, `cfl_time_step(...)`).

The result should be a single entry point the renderer calls, e.g.

```rust
impl Simulation {
    /// Advances the simulation by up to `max_substeps` substeps covering
    /// `frame_dt` seconds and returns the future to chain rendering after.
    pub fn step(&mut self, scene: &mut Scene, settings: &SimulationSettings, frame_dt: f32, after: Box<dyn GpuFuture>) -> Box<dyn GpuFuture>;
}
```

`SimulationSettings` collects what the simulation reads from the UI today
(`use_cfl`, `cfl_factor`, `cfl_min_dt`, `cfl_max_dt`, `max_substeps_per_frame`,
solver thresholds, `sort_algorithm`); the UI owns it and the renderer passes
it through. The `display_*` fields of the UI are filled from the returned
`SimulationStats`.

Keep the order of operations in a frame exactly as now:
1. read oldest stats, update CFL `dt`, compute solver settings;
2. set `piston_velocity` and `attractor` in `scene.sim_params`;
3. write camera/box buffers; record `update_buffer(sim_params)` (+ init on
   the first frame) and submit;
4. compute substep count, animate the box by `substeps * dt`;
5. submit the cached substep buffer `substeps` times;
6. submit the tail buffer, `then_signal_semaphore()`.

`Renderer` then only: owns the window/swapchain, the graphics pipelines, the
GUI, calls `simulation.step(...)`, records rendering, presents.

Move the Vulkano context and window creation from `Renderer::new` into
`render/gpu_context.rs` (`GpuContext::new(window, event_loop) -> GpuContext`)
so that `Renderer::new` reads as a list of the things it builds.

### Step 5. Split the sorters

`sim/sort/` per the target layout. `GpuSorter` is renamed `BitonicSorter`.
`SortAlgorithm` keeps its variants and derives (`PartialEq, Clone, Copy, Debug`).
Move `#[cfg(test)] mod tests` and `mod benchmark` into `sorter_tests.rs`
unchanged (same test function names).

### Step 6. Input without Command objects and without `Actor`

- `Command`, the four `*CameraCommand` structs, `Controller` and
  `KeyboardAction` become one `Input` struct in `src/input.rs`:

  ```rust
  pub struct Input {
      pressed_keys: HashSet<KeyCode>,
      mouse_delta: Vec2,
  }
  impl Input {
      pub fn key_event(&mut self, key: KeyCode, state: ElementState);
      pub fn mouse_motion(&mut self, dx: f32, dy: f32);
      /// Moves and rotates the camera for this frame and clears the mouse delta.
      pub fn apply_to_camera(&mut self, camera: &mut Camera, dt: f32);
  }
  ```

  Keep the key bindings (W/S forward, D/A right, E/Q up), the speed constant
  (`CAMERA_MOVE_SPEED = 5.0`) and the rotation behaviour exactly as the
  current commands implement them (read `camera_commands.rs` and
  `Controller::get_mouse_command` carefully, including how the mouse delta is
  reset).
- `Actor` is implemented only by `Camera`. Replace it with inherent methods
  on `Camera` (`position()`, `move_by(direction, distance)`,
  `rotate_by(yaw_degrees, pitch_degrees)` with the same maths as
  `add_input_vector` / `add_rotation`). Remove `Actor::velocity`,
  `set_velocity`, `set_position`, `update` (no callers).
- `CollisionBox::contains` takes `&impl Actor` and has no callers: delete it.
- Rename getters: `Camera::get_view_matrix` → `view_matrix`,
  `get_projection_matrix` → `projection_matrix`.

### Step 7. `App`

`core/engine.rs` → `app.rs`, `Engine` → `App`. `window_event` is long; split
it into private methods (`on_mouse_button`, `on_key`, `on_redraw`, ...)
without changing what each branch does. `ApplicationError` → `AppError`
(variant names may drop the `Error` suffix: `WindowCreation`,
`EventLoopInit`, `ResourceLoad`, `Other`). Update `src/bin/main.rs`.

### Step 8. Style pass over the moved code

Go through every file once with the list in section 4: imports, `expect`
messages, `// SAFETY:` comments, comments that repeat the code, getter
names, abbreviations. Do not reformat files wholesale.

### Step 9. Documentation

- Every `pub` item in `sim/` and `render/` gets a one-line `///` doc that
  says what it is for (not how).
- Update the "Project structure" part of `README.md` (if present) to the new
  layout.
- Put the substep data flow from section 2 as a `//!` module comment at the
  top of `sim/simulation.rs`.

## 8. Definition of done

- `cargo build --release` and `cargo test --release` pass with no new
  warnings.
- `solver_iterations_over_time` means within ±3% of the baseline,
  `substep_benchmark` totals within ±10%.
- Manual checks of section 3 pass.
- `git grep -n "ComputeStep\|trait Command\|trait Actor\|ParticleGenerator\|GpuPhysicsData\|mod utils\|mod entities"`
  returns nothing.
- No `mod.rs` contains a `struct`, `fn` or `impl`.
- `sim/` does not import anything from `render/` (`git grep -n "crate::render" src/sim` is empty).
- Each step is a separate commit that builds.
