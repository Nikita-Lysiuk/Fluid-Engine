use glam::Vec3;
use crate::entities::camera::Camera;
use crate::entities::collision::CollisionBox;
use crate::entities::particle::ParticleGenerator;
use crate::physics::fluid_data::FluidData;
use crate::utils::constants::MAX_PARTICLES;

pub struct Scene {
 
    pub fluid_data: FluidData,
    pub camera: Camera,
    pub boundary: CollisionBox,
    pub spacing: f32,
    pub particle_radius: f32,
    pub particle_mass: f32,
    pub target_density: f32,
}

impl Scene {
    pub fn new() -> Self {
        let particle_radius = 0.05;
        let density = 1000.0;
        let (rendering_data,  spacing, mass) = ParticleGenerator::generate(
            MAX_PARTICLES,
            particle_radius,
            density,
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 5.0, 1.0)
        );

        let mut camera = Camera::new(Vec3::new(0.0, 5.0, -10.0));
        camera.rotate(30.0, 0.0, 0.0);
        let boundary = CollisionBox::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 5.0, 1.0));

        Self {
            fluid_data: FluidData::new(rendering_data),
            camera,
            boundary,
            spacing,
            particle_radius,
            particle_mass: mass,
            target_density: density,
        }
    }
}