use glam::{IVec3, Vec3};
use rayon::prelude::*;

pub struct NeighborSearch {
    cell_size: f32,
    inv_cell_size: f32,
    hash_table_size: usize,

    pub particle_indices: Vec<usize>,
    cell_starts: Vec<usize>
}

impl NeighborSearch {
    pub fn new(h: f32, max_particles: usize) -> Self {
        let cell_size = h * 2.0;
        let hash_table_size = max_particles + 1;

        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            hash_table_size,
            particle_indices: vec![0; max_particles],
            cell_starts: vec![0; hash_table_size + 1],
        }
    }
    pub fn hash(&self, cell: IVec3) -> usize {
        let p1 = 73856093i32;
        let p2 = 19349663i32;
        let p3 = 83492791i32;
        let h = (cell.x.wrapping_mul(p1) ^ cell.y.wrapping_mul(p2) ^ cell.z.wrapping_mul(p3)) as usize;
        h % self.hash_table_size
    }
    fn pos_to_cell(&self, pos: Vec3) -> IVec3 {
        IVec3::new(
            (pos.x * self.inv_cell_size).floor() as i32,
            (pos.y * self.inv_cell_size).floor() as i32,
            (pos.z * self.inv_cell_size).floor() as i32,
        )
    }
    pub fn build(&mut self, positions: &[Vec3]) {
        let n = positions.len();

        let mut hashes: Vec<(usize, usize)> = positions
            .par_iter()
            .enumerate()
            .map(|(idx, &pos)| {
                let cell = self.pos_to_cell(pos);
                (self.hash(cell), idx)
            })
            .collect();

        hashes.par_sort_unstable_by_key(|h| h.0);

        self.cell_starts.fill(0);

        self.particle_indices.clear();
        self.particle_indices.extend(hashes.iter().map(|h| h.1));

        let mut current_hash = 999999999;
        for (i, &(h, _)) in hashes.iter().enumerate() {
            while current_hash != h {
                current_hash = if current_hash == 999999999 { h } else { current_hash + 1 };
                if current_hash >= self.hash_table_size { break; }
                self.cell_starts[current_hash] = i;
            }
        }

        self.cell_starts[self.hash_table_size] = n;
    }
    pub fn find_neighbors(&self, pos: Vec3, positions: &[Vec3], radius: f32) -> Vec<usize> {
        let center_cell = self.pos_to_cell(pos);
        let mut neighbors = Vec::with_capacity(64);
        let r_sq = radius * radius;

        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let h = self.hash(center_cell + IVec3::new(x, y, z));
                    let start = self.cell_starts[h];
                    let end = self.cell_starts[h + 1];

                    for i in start..end {
                        let idx = self.particle_indices[i];
                        let dist_sq = (positions[idx] - pos).length_squared();
                        if dist_sq <= r_sq {
                            neighbors.push(idx);
                        }
                    }
                }
            }
        }

        neighbors
    }
}