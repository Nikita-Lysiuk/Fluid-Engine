# Fluid Simulation Engine (Diploma Project)

## TL;DR
This project is a **fluid simulation engine in Rust**, targeting both **CPU** and **GPU (Vulkan)**.  
Main goals:
- Implement a fluid solver.
- Build a minimal engine to run and visualize it.
- Compare CPU vs GPU performance.
- Use this as the foundation for my diploma thesis and technical demo.

---

## Conventions

### Workflow
1. Documentation → define problem
2. Analysis → evaluate algorithms and methods
3. Design → plan architecture and modules
4. Implementation → write the code
5. Testing → verify with unit/integration tests
6. Documentation polish → update and summarize results  

### Task Naming
Format: 
- [MODULE] short description

Examples:
- `[SIM] Implement basic CPU solver`
- `[GFX] Vulkan swapchain creation`
- `[DOC] Write thesis introduction`

### Commit Style
- `feat:` — new feature
- `fix:` — bug fix
- `refactor:` — internal changes
- `docs:` — documentation
- `test:` — tests only
- `chore:` — configs, build, formatting

Examples:
- `feat(sim): add 2D grid solver`
- `fix(gfx): swapchain recreation bug`

### Code Style
- Rust 2021 edition
- `snake_case` for functions, variables, files
- `PascalCase` for types and structs
- `SCREAMING_SNAKE_CASE` for constants

---

## Learning Objectives

By building this project I want to:
- Deepen understanding of **fluid dynamics** (Navier–Stokes) in a practical setting.
- Gain experience with **low-level graphics programming** (Vulkan).
- Learn how to structure a simulation/graphics engine from scratch.
- Compare **parallelization paradigms** (CPU vs GPU) with benchmarks.
- Improve **engineering workflow**: documentation, testing, profiling, performance analysis.

---

## References

- [*Vulkan Tutorial*](https://vulkan-tutorial.com)
- [*3D Math Primer for Graphics and Game Development*](https://gamemath.com/book/orient.html#quaternion_slerp)
- *Physics for Game Developers (2nd Edition)*
- [*The Book of Shaders*](https://thebookofshaders.com/)

---