use glam::Vec3;

/// Particle positions and mass generated for a fluid block.
pub struct SpawnedBlock {
    /// Generated particle positions.
    pub positions: Vec<[f32; 3]>,
    /// Mass assigned to each generated particle.
    pub particle_mass: f32,
}

/// Generates particles inside a rectangular block.
pub fn spawn_block(
    origin: Vec3,
    size: Vec3,
    radius: f32,
    density: f32,
    spacing: f32,
    jitter: f32,
) -> SpawnedBlock {
    let count_x = (size.x / spacing).floor() as usize;
    let count_y = (size.y / spacing).floor() as usize;
    let count_z = (size.z / spacing).floor() as usize;

    let mut positions = Vec::with_capacity(count_x * count_y * count_z);

    let offset = origin + Vec3::splat(radius);

    for x in 0..count_x {
        for y in 0..count_y {
            for z in 0..count_z {
                let jitter = (rand::random::<f32>() - 0.5) * jitter;

                let px = offset.x + (x as f32 * spacing) + jitter;
                let py = offset.y + (y as f32 * spacing) + jitter;
                let pz = offset.z + (z as f32 * spacing) + jitter;

                positions.push([px, py, pz]);
            }
        }
    }

    let particle_volume = spacing.powi(3);
    let mass = density * particle_volume;

    SpawnedBlock {
        positions,
        particle_mass: mass,
    }
}
