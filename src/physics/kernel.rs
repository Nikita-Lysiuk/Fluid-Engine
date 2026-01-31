use std::f32::consts::PI;
use glam::Vec3;

pub trait SmoothingKernel {
    fn evaluate(&self, r: f32) -> f32;
    fn gradient(&self, r_vec: Vec3, r: f32) -> Vec3;
    fn w_zero(&self) -> f32;
}


pub struct CubicKernel {
    radius: f32,
    k: f32,
    l: f32,
    w_zero: f32,
}

impl CubicKernel {
    pub fn new(radius: f32) -> Self {
        let h3 = radius.powi(3);

        let k = 8.0 / (PI * h3);
        let l = 48.0 / (PI * h3);

        let w_zero = k;

        Self {
            radius,
            k,
            l,
            w_zero,
        }
    }
}

impl SmoothingKernel for CubicKernel {
    fn evaluate(&self, r: f32) -> f32 {
        let q = r / self.radius;

        if q > 1.0 {
            return 0.0;
        }

        if q <= 0.5 {
            let q2 = q * q;
            let q3 = q2 * q;
            self.k * (6.0 * q3 - 6.0 * q2 + 1.0)
        } else {
            self.k * (2.0 * (1.0 - q).powi(3))
        }
    }
    fn gradient(&self, r_vec: Vec3, r: f32) -> Vec3 {

        let q = r / self.radius;
        if q > 1.0 || r < 1e-9 {
            Vec3::ZERO
        } else {
            let mut gradq = r_vec / r;
            gradq /= self.radius;
            if q <= 0.5 {
                self.l * q * (3.0 * q - 2.0) * gradq
            } else {
                let factor = 1.0 - q;
                self.l * (-factor * factor) * gradq
            }
        }
    }
    fn w_zero(&self) -> f32 {
        self.w_zero
    }
}

