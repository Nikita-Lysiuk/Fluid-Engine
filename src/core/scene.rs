use glam::Vec3;
use log::info;
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
        let particle_radius = 0.015;
        let target_density = 1000.0;

        let box_min = Vec3::new(-1.5, 0.0, -1.0);
        let box_max = Vec3::new(1.5, 4.0, 1.0);
        let boundary = CollisionBox::new(box_min, box_max);


        let spawn_pos = Vec3::new(-1.0, 1.0, -0.8);

        let water_width = 1.0;
        let water_height = 2.0;
        let water_depth = 0.8;

        let spacing = particle_radius * 2.0;

        let (initial_positions, particle_mass) = ParticleGenerator::generate_volume(
            spawn_pos,
            water_width,
            water_height,
            water_depth,
            particle_radius,
            target_density,
            spacing,
            0.01
        );

        let mut camera = Camera::new(Vec3::new(0.0, 1.5, -3.5));
        camera.rotate(0.0, 0.0, 0.0);


        let smoothing_radius = particle_radius * 4.0;


        let dt = 0.005;

        let viscosity = 0.15;
        let relax_factor = 0.5;

        let density_iterations = 4;
        let divergence_iterations = 1;

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

        info!("[Scene] Created new scene with {} particles.", initial_positions.len());

        Self {
            initial_positions,
            sim_params,
            camera,
            boundary,
        }
    }
}