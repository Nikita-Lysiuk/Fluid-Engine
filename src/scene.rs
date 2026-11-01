use crate::camera::Camera;
use crate::sim::collision_box::CollisionBox;
use crate::sim::params::SimulationParams;
use crate::sim::spawn::spawn_block;
use glam::{IVec3, Vec2, Vec3};
use log::info;

pub struct Scene {
    pub initial_positions: Vec<[f32; 3]>,
    pub sim_params: SimulationParams,
    pub camera: Camera,
    pub boundary: CollisionBox,
    pub pillar: Pillar,
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
    pub fn aim(
        &mut self,
        camera: &Camera,
        cursor: Vec2,
        window_size: Vec2,
        boundary: &CollisionBox,
    ) {
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

/// A vertical pillar standing on the floor and reaching the top of the box.
#[derive(Clone, Copy, Debug)]
pub struct Pillar {
    /// x and z of the pillar axis.
    pub centre: Vec2,
    pub radius: f32,
    pub enabled: bool,
}

impl Pillar {
    /// `[x, z, radius, 1 or 0]` as expected by `SimulationParams::pillar`.
    pub fn to_gpu(&self) -> [f32; 4] {
        [
            self.centre.x,
            self.centre.y,
            self.radius,
            if self.enabled { 1.0 } else { 0.0 },
        ]
    }

    fn contains(&self, position: [f32; 3], margin: f32) -> bool {
        self.enabled
            && Vec2::new(position[0], position[2]).distance(self.centre) < self.radius + margin
    }
}

/// Everything that differs between scenes; the solver settings are shared.
struct SceneSetup {
    particle_radius: f32,
    box_min: Vec3,
    box_max: Vec3,
    water_origin: Vec3,
    water_size: Vec3,
    pillar: Pillar,
    wave_amplitude: f32,
    wave_frequency: f32,
    camera_position: Vec3,
    camera_target: Vec3,
    dt: f32,
}

impl Scene {
    /// Particle count of the default scene.
    pub const DEFAULT_PARTICLES: u32 = 100_000;

    /// A block of about `particle_count` particles falling into the box,
    /// next to a pillar.
    pub fn new(particle_count: u32) -> Self {
        // 2.0 x 2.0 x 1.6 m holds 100k particles at this spacing; other
        // counts scale the block uniformly.
        let scale = (particle_count as f32 / 100_000.0).cbrt();
        let box_min = Vec3::new(-1.5, 0.0, -1.0);
        let water_size = Vec3::new(2.0, 2.0, 1.6) * scale;
        Self::build(SceneSetup {
            particle_radius: 0.020,
            box_min,
            box_max: Vec3::new(0.8, 4.0, 1.0),
            water_origin: Vec3::new(box_min.x + 0.05, 0.5, -0.5 * water_size.z),
            water_size,
            pillar: Pillar {
                centre: Vec2::new(0.35, 0.0),
                radius: 0.15,
                enabled: true,
            },
            wave_amplitude: 0.3,
            wave_frequency: 0.5,
            camera_position: Vec3::new(0.0, 1.5, -3.5),
            camera_target: Vec3::new(0.0, 1.5, 0.0),
            dt: 0.005,
        })
    }

    /// The scene of the README animation: a long, shallow tank whose left
    /// wall pushes waves against a pillar. Particles are smaller than in the
    /// interactive scene (about 150k), so it is meant for offline rendering.
    pub fn showcase() -> Self {
        let box_min = Vec3::new(-2.0, 0.0, -0.7);
        let box_max = Vec3::new(1.2, 1.2, 0.7);
        Self::build(SceneSetup {
            particle_radius: 0.012,
            box_min,
            box_max,
            water_origin: box_min + Vec3::new(0.2, 0.0, 0.0),
            water_size: Vec3::new(3.0, 0.55, 1.4),
            pillar: Pillar {
                centre: Vec2::new(0.35, 0.0),
                radius: 0.18,
                enabled: true,
            },
            wave_amplitude: 0.3,
            wave_frequency: 0.4,
            camera_position: Vec3::new(-0.6, 1.45, -3.4),
            camera_target: Vec3::new(-0.35, 0.35, 0.1),
            dt: 0.0025,
        })
    }

    fn build(setup: SceneSetup) -> Self {
        let target_density = 1000.0;
        let particle_radius = setup.particle_radius;
        let spacing = particle_radius * 2.0;

        let mut collision_box = CollisionBox::new(setup.box_min, setup.box_max);
        collision_box.wave_amplitude = setup.wave_amplitude;
        collision_box.wave_frequency = setup.wave_frequency;

        let crate::sim::spawn::SpawnedBlock {
            positions,
            particle_mass,
        } = spawn_block(
            setup.water_origin,
            setup.water_size,
            particle_radius,
            target_density,
            spacing,
            0.01,
        );
        // No particles inside the pillar: the solver would push them out
        // violently in the first step.
        let initial_positions: Vec<[f32; 3]> = positions
            .into_iter()
            .filter(|p| !setup.pillar.contains(*p, 0.5 * spacing))
            .collect();

        let camera = Camera::look_at(setup.camera_position, setup.camera_target);

        let smoothing_radius = particle_radius * 4.0;
        let viscosity = 0.15;
        let relax_factor = 0.5;

        // Upper bounds; the solvers usually stop earlier (early exit by error).
        let density_iterations = 20;
        let divergence_iterations = 20;

        let grid_res = IVec3::new(128, 128, 128);

        let mut sim_params = SimulationParams::new(
            particle_radius,
            particle_mass,
            smoothing_radius,
            target_density,
            viscosity,
            relax_factor,
            setup.dt,
            density_iterations,
            divergence_iterations,
            Vec3::new(0.0, -9.81, 0.0),
            setup.box_min,
            setup.box_max,
            grid_res,
        );
        sim_params.pillar = setup.pillar.to_gpu();

        info!(
            "[Scene] Created new scene with {} particles.",
            initial_positions.len()
        );

        Self {
            initial_positions,
            sim_params,
            camera,
            boundary: collision_box,
            pillar: setup.pillar,
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
