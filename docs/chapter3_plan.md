# Chapter 3 Plan — Implementacja i oryginalne rozwiązania techniczne

**Core style rule:** Every trade-off section ends with a decision table.
Every quantitative comparison gets a `#[test]` (headless Vulkan, no window) → CSV → matplotlib.
Minimal code listings — use diagrams, flowcharts, and pseudocode instead.
Only XSPH viscosity is implemented — no comparison with Laplacian viscosity.

---

## 3.1 Założenia projektowe

**What to write:**
- Hardware requirements: GPU with Vulkan 1.3 support; mandatory device features from
  `VulkanoConfig` block in `renderer/mod.rs`:
  `dynamic_rendering`, `synchronization2`, `scalar_block_layout`,
  `buffer_device_address`, `shader_int64`, `large_points`, `fill_mode_non_solid`
- Software: OS (Windows/Linux), GPU driver version floor
- Build: Rust toolchain (edition 2024), `glslc` / `shaderc` for GLSL
- Runtime-tunable parameters: particle count, smoothing radius, dt, gravity,
  density/divergence solver iterations, viscosity coefficient

**Diagrams/tables:** Requirements table (category / requirement / minimum value).

**Data needed:** None.

---

## 3.2 Architektura systemu

**What to write:**
- CPU side: WinIT event loop → `Engine` → `Renderer` → builds Vulkan command buffers
- GPU side: all compute (physics) + all render (ray march / particles)
- Frame loop: `RedrawRequested` → `update_and_render` → `step` (physics) → render pass
- dt is clamped to 0.1 s in `engine.rs` to prevent spiral-of-death on frame drops
- Key design patterns used throughout:
  - `ComputeStep` trait (all 11 compute pipelines implement it identically)
  - `Actor` trait (Camera)
  - `ApplicationHandler` from WinIT (Engine)
  - Double-buffering for render positions (copied from physics buffer each frame)

**Diagrams/tables:**
- CPU ↔ GPU architecture diagram (boxes + arrows)
- Frame loop flowchart (from `engine.rs` + `renderer/mod.rs`)
- Table: subsystem / CPU or GPU / reason

**Data needed:** None.

---

## 3.3 Warstwa aplikacji

### 3.3.1 Zarządzanie oknem
- WinIT `ApplicationHandler` pattern: `resumed()` creates window and renderer,
  `RedrawRequested` drives the frame loop
- `window_event()` handles keyboard/mouse with focus-lock logic
- `PresentMode::Immediate` (no vsync) for accurate FPS measurement
- Triple buffering: `min_image_count = 3`

**Diagram:** WinIT event → Engine state machine (init / running / focus states).

### 3.3.2 Kamera
- Quaternion-based orientation (`glam::Quat`)
- `Actor` trait: `move_forward`, `rotate` methods
- `CameraGPU` struct (128 bytes): `view`, `proj`, `inv_view_proj` matrices + `camera_pos`
  uploaded as a device-address uniform every frame
- `inv_view_proj` is essential for ray marching (reconstruct world-space rays from NDC)

**Diagram:** Camera transform chain: world → view → clip → NDC → ray.

### 3.3.3 Scena i parametry symulacji
- `SimulationParams` is a single 376-byte struct sent to every shader as a uniform buffer
- Contains: particle geometry (`radius`, `mass`, `smoothing_radius`, `target_density`),
  solver config (`dt`, `density_solver_iterations`, `divergence_solver_iterations`),
  world config (`gravity`, `box_min`, `box_max`, `grid_res`)
- `sync_with_scene()` copies CPU-side params to GPU buffer every frame

**Diagram:** Data flow — `Scene` (CPU) → `sync_with_scene` → `sim_params_buffer` (GPU) →
every compute shader reads it via `layout(set=..., binding=...) uniform SimParams`.

---

## 3.4 Inicjalizacja GPU z Vulkano

### 3.4.1 Kontekst Vulkan
- `VulkanoConfig` → `VulkanoContext`: Instance → PhysicalDevice → LogicalDevice → Queue
- `ext_debug_utils` enabled for validation layer output
- `VulkanoWindowRenderer` wraps swapchain; compute pipelines have no WSI dependency
  (important: means they can run in headless `#[test]`)

**Table:** Vulkan concept / Vulkano type / role in this project.

### 3.4.2 Wzorzec ComputeStep
- **This is the one architectural listing in the chapter** — the `ComputeStep` trait
  (~10 lines): `load_shader_module`, `from_pipeline`, `prepare`, `execute`
- All 11 compute pipelines implement it identically; `ComputePipelines` struct holds them
- `prepare()` binds descriptor sets once at init; `execute()` just records dispatch into the
  command buffer — no allocation in the hot path

**Diagram:** `ComputeStep` implementors list → `ComputePipelines` aggregator →
`step()` calls them in sequence.

### 3.4.3 Deskryptory i synchronizacja
- Descriptor set = how shaders access GPU buffers (positions, velocities, grid, params)
- Pipeline barriers between compute dispatches ensure write-before-read ordering
- SOA layout means each binding is a single flat buffer → maximum coalescing

**Diagram:** Descriptor set layout for one pipeline (bindings numbered, buffer types labeled).

---

## 3.5 Struktury danych cząstek

### 3.5.1 SOA vs AOS
- AOS: `struct Particle { pos, vel, density, pressure }` — all fields for one particle
  are contiguous in memory
- SOA: separate flat buffers for each field — all positions in one buffer, all velocities
  in another, etc.
- **Why AOS fails for DFSPH on GPU:** the solver reads `vel[j]` (neighbour velocity) while
  computing corrections for particle `i`. With AOS, accessing `vel[j]` loads the entire
  `Particle[j]` cache line (including pos, density, pressure) — wasted bandwidth.
  With SOA, the velocity buffer is read sequentially without polluting other fields.
- Additional reason: Vulkan descriptor bindings map directly to flat buffers.
  Splitting fields into separate bindings gives the driver freedom to schedule memory
  loads independently.

**Diagram:** Side-by-side memory layout (AOS vs SOA), with cache-line boundaries marked.
No benchmark needed — argument is structural.

### 3.5.2 GpuPhysicsData
- List of all buffers and their roles
- Dual buffers for velocities (`velocity_a` / `velocity_b`) — solver writes to `b`
  while reading from `a`, then swaps; avoids data races without barriers inside a pass
- `Entry` struct: `hash: u32` + `index: u32` — this is what the sorter works on

---

## 3.6 Wyszukiwanie sąsiadów

### 3.6.1 Problem
- Naïve: check all N particles for each of N particles → O(N²) dispatches
- At N=50k this is 2.5 billion pair evaluations per frame — completely infeasible real-time

**Diagram:** complexity curve O(N²) vs O(N·k) with k≈27 (3³ neighbourhood cells).

### 3.6.2 Spatial Hashing
- Divide space into uniform grid cells of size `h` (smoothing radius)
- For particle at `pos`, compute `cell = floor(pos / h)`
- Hash: `(cell.x * 73856093) XOR (cell.y * 19349663) XOR (cell.z * 83492791) mod table_size`
- Simple choice — no alternative analysis (the function provides uniform distribution
  for typical particle distributions; alternatives were not benchmarked)

**Diagram:** 2D illustration — particles mapped to cells, cells mapped to hash table slots.

### 3.6.3 Pipeline: hash → sort → offsets → reorder
- Stage 1 `spatial_hash`: write `Entry(hash, particle_idx)` for each particle
- Stage 2 sort: sort `Entry[]` by hash (covered in detail in 3.7)
- Stage 3 `grid_offsets`: scan sorted list → `grid_start[hash]` = first index for each cell
- Stage 4 `reorder`: reorder all particle data arrays in the sorted order
  (cache-friendly neighbour traversal afterward)

**Diagram:** Pipeline flowchart showing each stage's input/output buffers.

---

## 3.7 Sortowanie na GPU

### 3.7.1 Bitonic Sort
- Sorting network: fixed sequence of compare-and-swap operations, independent of data
- Works in `log²(N)` passes; each pass is one compute dispatch with `N/2` threads
- Zero branch divergence — every thread executes identical code path
- Local size 256; push constants: `num_entries`, `block_height`, `block_step`
- Cite: `\cite{jain2015bitonic}`

**Diagram:** Sorting network for N=8 showing all compare-and-swap pairs across passes.

### 3.7.2 Radix Sort
- 4 passes, 8 bits per pass (256 buckets), covers 32-bit hash keys
- Each pass: `radix_count` (count per bucket per workgroup) →
  `radix_scan` (prefix sum over counts) → `radix_reorder` (scatter)
- Alternates between two descriptor sets (ping-pong buffers) to avoid a separate copy
- O(kN) where k=4 fixed passes; no comparison overhead
- Cite: `\cite{harada2011introduction}`

**Diagram:** Single-pass pipeline (count → scan → reorder) with buffer annotations.

### 3.7.3 Porównanie

**Decision table:** algorithm / time complexity / branch divergence / memory passes / best N range

**Benchmark:**
- `#[test]` using headless `VulkanoContext` (no `VulkanoWindowRenderer` needed)
- Create `Entry` buffer filled with random values
- Switch `SortAlgorithm` enum (`Bitonic` | `Radix`) — already exists in `sorter.rs`
- Measure wall time with `std::time::Instant` per sort call
- N = 1k, 10k, 50k, 100k, 200k
- Write `sort_benchmark.csv`: `n,algorithm,time_ms`
- matplotlib: line graph (time vs N), two series; log-scale x axis

**Expected result:** Bitonic wins at small N (simpler constant); Radix wins above ~50k.

---

## 3.8 Solver DFSPH

### 3.8.1 Pętla substepów
- Derived from `step()` in `src/renderer/mod.rs`
- Outer loop: `while step < max_dt` accumulates sub-steps of size `dt`
- Each substep: viscosity → density source → density solver (N iters) →
  pressure integrate → *re-run neighbour search* → density alpha →
  divergence source → divergence solver (N iters) → divergence integrate
- Neighbour search runs **twice per substep**: before density and after integration
- Stats (`max_speed`, `avg_density_error`, `avg_divergence_error`) read from GPU at frame start

**Diagram:** Full substep flowchart (one pass through the while loop), all pipeline stages labeled.

### 3.8.2 Obliczanie gęstości
- SPH density estimate: `ρ_i = Σ_j m_j · W(|r_i − r_j|, h)`
- Cubic spline kernel: compact support radius h, C² continuous
- Also computes `α_i` (DFSPH diagonal coefficient) used as preconditioner in solver

**Graph:** `W(r, h)` curve (kernel value vs distance); shows how contribution drops to zero at r=h.

### 3.8.3 Korekcja gęstości (Stage 1)
- Source term: `s_i = (ρ₀ − ρ_i) / Δt`
- Iterative: compute pressure from source term → accumulate pressure forces →
  update pressure → repeat until `avg_density_error < threshold`
- Iteration count controlled by `density_solver_iterations` or error threshold

**Graph:** density error vs iteration number (data from `#[test]`, see below).

### 3.8.4 Korekcja dywergencji (Stage 2)
- After pressure integration, velocity field may still have divergence
- Source term: `d_i = Σ_j (v_i − v_j) · ∇W`
- Same iterative pressure correction loop until `avg_divergence_error < threshold`
- Enforcing ∇·v = 0 prevents artificial compression and particle clumping

**Graph:** divergence error vs iteration number (same test as above, different metric).

**`#[test]` for 3.8.3 + 3.8.4:**
- Headless: create context, fill particle buffer with a uniform block configuration
- Modify solver to write `stats_buffer` after each iteration (currently only written once per frame)
- Log `(iteration, density_error, divergence_error)` to `convergence.csv`
- matplotlib: two subplots (density convergence, divergence convergence)

### 3.8.5 Warunek CFL i dobór Δt

**The CFL condition:**
- Courant–Friedrichs–Lewy: `Δt_CFL = λ · h / v_max` (λ≈0.4 in implementation)
- Intuition diagram: particle must not travel more than one cell per step

**Trap 1 — low speed → large Δt → instability:**
- When `v_max → 0` (fluid at rest), `Δt_CFL → ∞`
- Implementation clamps: `cfl_dt.clamp(0.001, 0.05)` and `smooth_dt = cfl_dt.min(dt * 1.1)`
  (only allows 10% growth per frame to prevent sudden jumps)
- Even with clamping: a large dt combined with accumulated density error causes explosion

**Trap 2 — dynamic iterations explode FPS:**
- With CFL giving large dt, solver needs many iterations to converge → observed 100+
  instead of expected 4–5 → frame time collapses
- Root cause: error threshold mode (`use_solver_error_threshold`) and CFL interact badly;
  large dt means solver never reaches the threshold within budget iterations

**Observed behaviour (to show as table/graph):**
- Static dt=0.005: stable most of the time; occasional crash from error accumulation
- CFL + static iter count: unstable; large dt corrupts simulation
- CFL + error threshold: iteration count explodes → FPS < 5

**Decision table:** approach / FPS / stability / visual quality / notes
**Final choice:** static dt with fixed iteration count — stable, predictable, explainable

**`#[test]` for this section:**
- Headless step loop, run 500 frames
- Log `(frame, dt, max_speed, density_iters_used, divergence_iters_used)` to `cfl_comparison.csv`
- Two runs: `use_cfl=false` (static) vs `use_cfl=true` (adaptive)
- matplotlib: 3 subplots — dt over time, iteration count over time, max_speed over time

---

## 3.9 Siły fizyczne

### 3.9.1 Lepkość XSPH
- Only XSPH is implemented — no comparison with Laplacian viscosity
- Formula: `v_i += ν · Σ_j (m_j / ρ_j) · (v_j − v_i) · W(r, h)`
- Effect: blends each particle's velocity toward its neighbours' average
  → smooth, cohesive flow; prevents particle inter-penetration artifacts
- Note: gravity (`v += g · Δt`) is applied in the same shader pass
- `viscosity` coefficient is user-tunable at runtime via egui slider

**Diagram:** Before/after velocity vectors showing XSPH smoothing effect.
**Formula box** (not a code listing — typeset with amsmath).

### 3.9.2 Kolizje
- Penalty-based boundary: when particle crosses `box_min`/`box_max`, apply restoring force
- Simple, no constraint solving needed; works well for solid box boundaries

**Diagram:** particle path at boundary, velocity reflection vector.

---

## 3.10 Wizualizacja

### 3.10.1 Splatowanie gęstości do tekstury 3D
- `splat_density.comp`: for each particle, add its kernel contribution to nearby 3D voxels
- Result: `density_texture` — a `Format::R32Uint` 3D image
- Trade-off: grid resolution ↔ visual quality + performance
  - Too coarse → blocky surface
  - Too fine → GPU memory bandwidth bottleneck

**Graph:** splat time (ms) vs grid resolution (easy `#[test]`: dispatch splat at different
`grid_res` values, time it, write `splat_benchmark.csv`).

### 3.10.2 Ray Marching
- For each pixel: cast ray from camera using `inv_view_proj`
- March through 3D density texture until `density > threshold` (surface hit)
- Binary search refinement between last empty and first full step → accurate normal
- Normal = density gradient `∇ρ` computed by central differences in 3D texture

**Flowchart:** ray march loop (AABB entry → march → hit/miss → binary refine → shade).

### 3.10.3 Model oświetlenia
- Fresnel blend: reflection vs refraction based on view angle (`schlick` approximation)
- Refraction: Snell's law with water IOR ≈ 1.33
- Sub-surface scattering: approximate — exponential absorption based on thickness
- Foam: rendered where density exceeds a second threshold (surface peaks)
- Sun: directional light with GGX specular term

**Diagram:** light path through water surface (incident → reflected + refracted).

### 3.10.4 Środowisko HDRI
- Loaded from `.exr` file (Citrus Orchard Road PureSky 4K) at startup
- Used for: skybox background render (`sky.frag`) + reflection samples in `raymarch.frag`
- Sphere mesh (radius 500) with inverted normals; camera always at centre

---

## Benchmark test checklist

| Test | File output | Sections using it |
|------|-------------|-------------------|
| Sort benchmark (Bitonic vs Radix, varying N) | `sort_benchmark.csv` | 3.7.3 |
| Solver convergence (error vs iteration) | `convergence.csv` | 3.8.3, 3.8.4 |
| CFL behaviour (static vs adaptive, 500 frames) | `cfl_comparison.csv` | 3.8.5 |
| Splat resolution vs time | `splat_benchmark.csv` | 3.10.1 |

All tests use headless `VulkanoContext` — no `VulkanoWindowRenderer` required.
Data format: CSV with header row. Python scripts use pandas + matplotlib.

---

## Writing order (recommended)

1. 3.2 Architecture (sets up the mental model for everything else)
2. 3.1 Requirements (short, factual)
3. 3.4 GPU init + ComputeStep (foundational — every section refers to it)
4. 3.5 Data structures
5. 3.6 Neighbour search
6. **Collect sort benchmark data → write** 3.7
7. 3.3 Application layer (camera, WinIT, scene)
8. 3.8.1–3.8.4 (flowchart + density/divergence)
9. **Collect CFL + convergence data → write** 3.8.5
10. 3.9 Forces
11. 3.10 Visualization
12. **Collect splat data → write** 3.10.1
