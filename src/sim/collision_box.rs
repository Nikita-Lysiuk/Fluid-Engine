use std::f32::consts::TAU;
use glam::Vec3;

/// Moving simulation boundary.
pub struct CollisionBox {
    /// Minimum boundary coordinates.
    pub min: Vec3,
    /// Maximum boundary coordinates.
    pub max: Vec3,
    /// Amplitude of the moving wall.
    pub wave_amplitude: f32,
    /// Frequency of the moving wall.
    pub wave_frequency: f32,
    /// Whether the `min.x` wall oscillates; toggled from the UI.
    pub animate: bool,
    time: f32,
    base_min_x: f32,
}

impl CollisionBox {
    /// Creates the associated resources.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self {
            base_min_x: min.x,
            min,
            max,
            wave_amplitude: 0.0,
            wave_frequency: 1.0,
            animate: true,
            time: 0.0,
        }
    }
    /// Advances the boundary animation.
    pub fn update(&mut self, dt: f32) {
        if !self.animate {
            return;
        }
        self.time += dt;
        let offset = self.wave_amplitude * (self.time * self.wave_frequency * TAU).sin();
        self.min.x = self.base_min_x + offset;
    }
    /// Velocity of the `min.x` wall along +x.
    pub fn piston_velocity(&self) -> f32 {
        if !self.animate {
            return 0.0;
        }
        let omega = self.wave_frequency * TAU;
        self.wave_amplitude * omega * (self.time * omega).cos()
    }
}

