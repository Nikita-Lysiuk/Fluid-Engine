// use glam::{UVec3, Vec3};
// use log::info;
// use rayon::prelude::*;
// use crate::entities::particle::Particle;
//
// pub struct NeighborSearch {
//     cell_size: f32,
//     grid_min: Vec3,
//     grid_size: UVec3,
//     particle_hash: Vec<(u32, u32)>,
//     cell_start: Vec<u32>,
//     cell_end: Vec<u32>,
// }
//
// impl NeighborSearch {
//     pub fn new(h: f32, min: Vec3, max: Vec3) -> Self {
//         let cell_size = h;
//
//         let grid_size = UVec3::new(
//             ((max.x - min.x) / cell_size).ceil() as u32,
//             ((max.y - min.y) / cell_size).ceil() as u32,
//             ((max.z - min.z) / cell_size).ceil() as u32,
//         ) + UVec3::splat(2);
//
//
//         let total_cells = (grid_size.x * grid_size.y * grid_size.z) as usize;
//
//         info!("Grid created: {} cells (Memory: ~{} KB)", total_cells, total_cells * 24 / 1024);
//
//         Self { cell_size, grid_min: min, grid_size, particle_hash: Vec::new(), cell_start: vec![u32::MAX; total_cells], cell_end: vec![u32::MAX; total_cells], }
//     }
//     pub fn build_grid(&mut self, vertices: &Vec<Particle>) {
//         let num_particles = vertices.len();
//
//         if self.particle_hash.len() != num_particles {
//             self.particle_hash.resize(num_particles, (0, 0));
//         }
//
//         let cell_size = self.cell_size;
//         let grid_min = self.grid_min;
//         let grid_size = self.grid_size;
//
//         self.particle_hash.par_iter_mut()
//             .enumerate()
//             .zip(vertices.par_iter())
//             .for_each(|((i, entry), &vert)| {
//                 let pos = Vec3::from_array(vert.position);
//                 let rel_pos = (pos - grid_min).max(Vec3::ZERO);
//
//                 let x = (rel_pos.x / cell_size) as u32;
//                 let y = (rel_pos.y / cell_size) as u32;
//                 let z = (rel_pos.z / cell_size) as u32;
//
//                 let cell_idx = x + y * grid_size.x + z * grid_size.x * grid_size.y;
//                 *entry = (cell_idx, i as u32);
//             });
//
//         self.particle_hash.par_sort_unstable_by_key(|entry| entry.0);
//         self.cell_start.par_iter_mut().for_each(|x| *x = u32::MAX);
//         self.cell_end.par_iter_mut().for_each(|x| *x = u32::MAX);
//
//         let mut prev_cell = u32::MAX;
//
//         for (i, &(cell_idx, _)) in self.particle_hash.iter().enumerate() {
//             if cell_idx != prev_cell {
//                 if (cell_idx as usize) < self.cell_start.len() {
//                     self.cell_start[cell_idx as usize] = i as u32;
//
//                     if prev_cell != u32::MAX && (prev_cell as usize) < self.cell_end.len() {
//                         self.cell_end[prev_cell as usize] = i as u32;
//                     }
//                 }
//                 prev_cell = cell_idx;
//             }
//         }
//
//         if prev_cell != u32::MAX && (prev_cell as usize) < self.cell_end.len() {
//             self.cell_end[prev_cell as usize] = num_particles as u32;
//         }
//     }
//     #[inline(always)]
//     pub fn for_each_neighbor<F>(&self, pos: &Vec3, mut callback: F)
//     where
//         F: FnMut(usize),
//     {
//         let rel_pos = (pos - self.grid_min).max(Vec3::ZERO);
//         let cx = (rel_pos.x / self.cell_size) as i32;
//         let cy = (rel_pos.y / self.cell_size) as i32;
//         let cz = (rel_pos.z / self.cell_size) as i32;
//
//         let dim_x = self.grid_size.x as i32;
//         let dim_y = self.grid_size.y as i32;
//         let dim_z = self.grid_size.z as i32;
//
//         for z in (cz - 1)..=(cz + 1) {
//             if z < 0 || z >= dim_z { continue; }
//             let z_offset = z * dim_y;
//
//             for y in (cy - 1)..=(cy + 1) {
//                 if y < 0 || y >= dim_y { continue; }
//                 let yz_offset = (y + z_offset) * dim_x;
//
//                 for x in (cx - 1)..=(cx + 1) {
//                     if x < 0 || x >= dim_x { continue; }
//
//                     let cell_idx = (x + yz_offset) as usize;
//
//                     let start = self.cell_start[cell_idx];
//
//                     if start != u32::MAX {
//                         let end = self.cell_end[cell_idx];
//
//                         for k in start..end {
//                             callback(k as usize);
//                         }
//                     }
//                 }
//             }
//         }
//     }
//     pub fn get_sorted_indices(&self) -> &Vec<(u32, u32)> {
//         &self.particle_hash
//     }
// }