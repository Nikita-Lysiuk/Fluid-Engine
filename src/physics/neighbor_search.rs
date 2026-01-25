use glam::{IVec3, Vec3};

pub struct NeighborSearch {
    inv_cell_size: f32,
    pub table_size: usize,

    cell_starts: Vec<usize>,
    cell_counts: Vec<u16>,
}

impl NeighborSearch {
    pub fn new(h: f32, max_particles: usize) -> Self {
        let cell_size = h * 2.0;
        let table_size = next_prime(max_particles * 2);

        Self {
            inv_cell_size: 1.0 / cell_size,
            table_size,
            cell_starts: vec![usize::MAX; table_size],
            cell_counts: vec![0; table_size],
        }
    }
    #[inline(always)]
    pub fn hash(&self, cell: IVec3) -> usize {
        let h = (cell.x.wrapping_mul(73856093)
            ^ cell.y.wrapping_mul(19349663)
            ^ cell.z.wrapping_mul(83492791)) as usize;
        h % self.table_size
    }
    #[inline(always)]
    pub fn pos_to_cell(&self, pos: Vec3) -> IVec3 {
        let v = pos * self.inv_cell_size;
        IVec3::new(v.x.floor() as i32, v.y.floor() as i32, v.z.floor() as i32)
    }
    pub fn build(&mut self, positions: &[Vec3]) {
        self.cell_starts.fill(usize::MAX);
        self.cell_counts.fill(0);

        if positions.is_empty() { return; }

        let mut current_hash = self.hash(self.pos_to_cell(positions[0]));
        self.cell_starts[current_hash] = 0;
        let mut count = 0;

        for i in 0..positions.len() {
            let h = self.hash(self.pos_to_cell(positions[i]));

            if h != current_hash {
                self.cell_counts[current_hash] = count;

                current_hash = h;
                self.cell_starts[current_hash] = i;
                count = 0;
            }
            count += 1;
        }
        self.cell_counts[current_hash] = count;
    }
    #[inline(always)]
    pub fn for_each_neighbor<F>(&self, pos_i: Vec3, positions: &[Vec3], r_sq: f32, mut callback: F)
    where F: FnMut(usize)
    {
        let center = self.pos_to_cell(pos_i);

        for z in -1..=1 {
            for y in -1..=1 {
                for x in -1..=1 {
                    let cell_idx = center + IVec3::new(x, y, z);
                    let h = self.hash(cell_idx);

                    let start = self.cell_starts[h];

                    if start == usize::MAX { continue; }

                    let count = self.cell_counts[h] as usize;
                    let end = start + count;

                    for i in start..end {
                        let dist_sq = positions[i].distance_squared(pos_i);
                        if dist_sq <= r_sq {
                            callback(i);
                        }
                    }
                }
            }
        }
    }
}
fn next_prime(mut n: usize) -> usize {
    loop {
        n += 1;
        if is_prime(n) { return n; }
    }
}
fn is_prime(n: usize) -> bool {
    if n <= 1 { return false; }
    for i in 2..=(n as f64).sqrt() as usize {
        if n % i == 0 { return false; }
    }
    true
}