use glam::{IVec3, Vec3};
use vulkano::buffer::BufferContents;

/// GPU-visible parameters for a simulation substep.
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
    /// Creates the associated resources.
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
