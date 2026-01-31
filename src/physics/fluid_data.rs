use glam::Vec3;
use crate::entities::particle::ParticleVertex;
use crate::utils::constants::MAX_PARTICLES;

pub struct FluidData {
    pub vertices: Vec<ParticleVertex>,
    pub velocities: Vec<Vec3>,
    pub densities: Vec<f32>,
    pub alphas: Vec<f32>,

    temp_vertices: Vec<ParticleVertex>,
    temp_velocities: Vec<Vec3>,
}

impl FluidData {
    pub fn new(vertices: Vec<ParticleVertex>) -> Self {
        Self {
            vertices,
            velocities: vec!(Vec3::ZERO; MAX_PARTICLES),
            densities: vec![0.0; MAX_PARTICLES],
            alphas: vec![0.0; MAX_PARTICLES],

            temp_vertices: Vec::with_capacity(MAX_PARTICLES),
            temp_velocities: Vec::with_capacity(MAX_PARTICLES),
        }
    }
    pub fn reorder_data(&mut self, sorted_indices: &[(u32, u32)]) {
        self.temp_vertices.clear();
        self.temp_velocities.clear();

        for &(_, original_idx) in sorted_indices {
            let idx = original_idx as usize;
            unsafe {
                self.temp_vertices.push(*self.vertices.get_unchecked(idx));
                self.temp_velocities.push(*self.velocities.get_unchecked(idx));
            }
        }
        std::mem::swap(&mut self.vertices, &mut self.temp_vertices);
        std::mem::swap(&mut self.velocities, &mut self.temp_velocities);
    }
}