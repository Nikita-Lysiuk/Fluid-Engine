// use glam::Vec3;
// use crate::entities::particle::Particle;
//
// pub struct FluidData {
//     pub vertices: Vec<Particle>,
//     pub velocities: Vec<Vec3>,
//     pub densities: Vec<f32>,
//     pub alphas: Vec<f32>,
//     pub kappa: Vec<f32>,
//     pub kappa_v: Vec<f32>,
//
//     temp_vertices: Vec<Particle>,
//     temp_velocities: Vec<Vec3>,
//     temp_kappa: Vec<f32>,
//     temp_kappa_v: Vec<f32>,
// }
//
// impl FluidData {
//     pub fn new(vertices: Vec<Particle>) -> Self {
//         Self {
//             vertices,
//             velocities: vec!(Vec3::ZERO; MAX_PARTICLES),
//             densities: vec![0.0; MAX_PARTICLES],
//             alphas: vec![0.0; MAX_PARTICLES],
//             kappa: vec![0.0; MAX_PARTICLES],
//             kappa_v: vec![0.0; MAX_PARTICLES],
//
//             temp_vertices: Vec::with_capacity(MAX_PARTICLES),
//             temp_velocities: Vec::with_capacity(MAX_PARTICLES),
//             temp_kappa: Vec::with_capacity(MAX_PARTICLES),
//             temp_kappa_v: Vec::with_capacity(MAX_PARTICLES),
//         }
//     }
//     pub fn reorder_data(&mut self, sorted_indices: &[(u32, u32)]) {
//         self.temp_vertices.clear();
//         self.temp_velocities.clear();
//         self.temp_kappa.clear();
//         self.temp_kappa_v.clear();
//
//         for &(_, original_idx) in sorted_indices {
//             let idx = original_idx as usize;
//             unsafe {
//                 self.temp_vertices.push(*self.vertices.get_unchecked(idx));
//                 self.temp_velocities.push(*self.velocities.get_unchecked(idx));
//                 self.temp_kappa.push(*self.kappa.get_unchecked(idx));
//                 self.temp_kappa_v.push(*self.kappa_v.get_unchecked(idx));
//             }
//         }
//         std::mem::swap(&mut self.vertices, &mut self.temp_vertices);
//         std::mem::swap(&mut self.velocities, &mut self.temp_velocities);
//         std::mem::swap(&mut self.kappa, &mut self.temp_kappa);
//         std::mem::swap(&mut self.kappa_v, &mut self.temp_kappa_v);
//     }
// }