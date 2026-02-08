use glam::Vec3;
use crate::entities::camera::Camera;
use crate::entities::collision::CollisionBox;
use crate::entities::particle::{ParticleGenerator, SimulationParams};

pub struct Scene {
    pub initial_positions: Vec<[f32; 3]>,
    pub sim_params: SimulationParams,
    pub camera: Camera,
    pub boundary: CollisionBox,
}

impl Scene {
    pub fn new() -> Self {
        let particle_radius = 0.05;
        let target_density = 1000.0;

        let box_min = Vec3::new(-2.0, 0.0, -2.0);
        let box_max = Vec3::new(2.0, 8.0, 2.0);
        let boundary = CollisionBox::new(box_min, box_max);

        let water_size = 2.0;
        let spawn_pos = Vec3::new(1.0, 2.0, 0.0);

        let (initial_positions, particle_mass, spacing) = ParticleGenerator::generate_cube(
            20,
            spawn_pos,
            water_size,
            0.01,
            target_density,
        );

        let mut camera = Camera::new(Vec3::new(0.0, 3.0, -8.0));
        camera.rotate(0.0, 0.0, 0.0);

        let smoothing_radius = spacing * 2.5;

        let dt = 1.0 / 90.0;
        let viscosity = 0.05;
        let relax_factor = 0.5;
        let density_iterations = 10;
        let divergence_iterations = 5;

        let sim_params = SimulationParams::new(
            particle_radius,
            particle_mass,
            smoothing_radius,
            target_density,
            viscosity,
            relax_factor,
            dt,
            density_iterations,
            divergence_iterations,
            Vec3::new(0.0, -9.81, 0.0),
            box_min,
            box_max,
        );

        Self {
            initial_positions,
            sim_params,
            camera,
            boundary,
        }
    }
}