#[path = "app.rs"]
pub mod app;
mod renderer;
pub mod input;
#[path = "camera.rs"]
pub mod camera;
pub mod app_error;
#[path = "scene.rs"]
pub mod scene;
#[path = "constants.rs"]
pub mod constants;
#[path = "frame_timer.rs"]
pub mod frame_timer;
#[path = "frame_benchmark.rs"]
pub mod frame_benchmark;



#[path = "render/particle_renderer.rs"]
mod particle_renderer;
#[path = "sim/particle_buffers.rs"]
mod particle_buffers;
#[path = "sim/params.rs"]
mod params;

#[path = "sim/spawn.rs"]
mod spawn;

#[path = "sim/compute_pass.rs"]
mod compute_pass;

#[path = "render/gpu_context.rs"]
mod gpu_context;

#[path = "sim/stats.rs"]
mod simulation_stats;

#[path = "sim/simulation.rs"]
mod simulation;

#[path = "render/density_texture.rs"]
mod density_texture;

#[path = "sim/pipelines.rs"]
mod sim_pipelines;

#[path = "sim/simulation_resources.rs"]
mod simulation_resources;

mod particle;
#[path = "sim/collision_box.rs"]
mod collision_box;
#[path = "render/box_outline.rs"]
mod box_outline;
#[path = "render/model_vertex.rs"]
mod model_vertex;
#[path = "render/sky.rs"]
mod sky;
#[path = "render/water.rs"]
mod water;
#[path = "render/shader_loader.rs"]
mod shader_loader;
