use glam::{IVec3, Vec2, Vec3};
use log::info;
use crate::entities::camera::Camera;
use crate::entities::collision::CollisionBox;
use crate::entities::particle::{ParticleGenerator, SimulationParams};

pub struct Scene {
    pub initial_positions: Vec<[f32; 3]>,
    pub sim_params: SimulationParams,
    pub camera: Camera,
    pub boundary: CollisionBox,
    pub attractor: Attractor,
}

/// Pulls fluid towards the point under the mouse cursor while the left
/// mouse button is held.
pub struct Attractor {
    pub active: bool,
    /// The cursor is over the scene (not over the UI); used to show a preview.
    pub hovered: bool,
    /// Distance of the attraction point from the camera along the cursor ray.
    /// While inactive it follows the plane through the box centre facing the
    /// camera; while active it is kept (and changed with the mouse wheel).
    pub distance: f32,
    pub centre: Vec3,
    /// Acceleration towards the point at its centre, in m/s².
    pub strength: f32,
    pub radius: f32,
}

impl Attractor {
    /// Places the attraction point on the ray through the cursor.
    pub fn aim(&mut self, camera: &Camera, cursor: Vec2, window_size: Vec2, boundary: &CollisionBox) {
        let (origin, direction) = camera.ray_through(cursor, window_size);
        if !self.active {
            let box_centre = 0.5 * (boundary.min + boundary.max);
            let forward = camera.forward();
            let facing = direction.dot(forward);
            if facing > 1e-3 {
                self.distance = ((box_centre - origin).dot(forward) / facing).max(0.3);
            }
        }
        self.centre = origin + direction * self.distance;
    }

    /// `[x, y, z, strength]` as expected by `SimulationParams::attractor`;
    /// strength 0 when inactive.
    pub fn to_gpu(&self) -> [f32; 4] {
        if !self.active {
            return [0.0; 4];
        }
        [self.centre.x, self.centre.y, self.centre.z, self.strength]
    }
}

impl Scene {
    /// Particle count of the default scene.
    pub const DEFAULT_PARTICLES: u32 = 100_000;

    /// A block of about `particle_count` particles falling into the box.
    pub fn new(particle_count: u32) -> Self {
        let particle_radius = 0.020;
        let target_density = 1000.0;

        let box_min = Vec3::new(-1.5, 0.0, -1.0);
        let box_max = Vec3::new(0.8, 4.0, 1.0);
        let mut collision_box = CollisionBox::new(box_min, box_max);
        collision_box.wave_amplitude = 0.3;
        collision_box.wave_frequency = 0.5;


        let spacing = particle_radius * 2.0;

        // 2.0 x 2.0 x 1.6 m holds 100k particles at this spacing; other
        // counts scale the block uniformly.
        let scale = (particle_count as f32 / 100_000.0).cbrt();
        let water_width = 2.0 * scale;
        let water_height = 2.0 * scale;
        let water_depth = 1.6 * scale;
        let spawn_pos = Vec3::new(box_min.x + 0.05, 0.5, -0.5 * water_depth);

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

        // Upper bounds; the solvers usually stop earlier (early exit by error).
        let density_iterations = 20;
        let divergence_iterations = 20;

        let grid_res = IVec3::new(128, 128, 128);

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
            grid_res,
        );

        info!("[Scene] Created new scene with {} particles.", initial_positions.len());

        Self {
            initial_positions,
            sim_params,
            camera,
            boundary: collision_box,
            attractor: Attractor {
                active: false,
                hovered: false,
                distance: 3.5,
                centre: Vec3::ZERO,
                strength: 40.0,
                radius: 0.6,
            },
        }
    }
}