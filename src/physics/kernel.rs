use glam::Vec3;

pub trait SmoothingKernel {
    fn evaluate(&self, r: f32) -> f32;
    fn gradient(&self, r_vec: Vec3, r: f32) -> Vec3;
}

pub struct WendlandKernel {
    normal_factor: f32,
    h: f32,
}

impl WendlandKernel {
    pub fn new(h: f32, dim: usize) -> Self {
        let normal_factor = match dim {
            1 => 5.0 / (64.0 * h),
            2 => 5.0 / (32.0 * std::f32::consts::PI * h.powi(2)),
            3 => 105.0 / (1024.0 * std::f32::consts::PI * h.powi(3)),
            _ => panic!("Unsupported dimension for Wendland kernel"),
        };

        Self { h, normal_factor }
    }
}

impl SmoothingKernel for WendlandKernel {
    fn evaluate(&self, r: f32) -> f32 {
        let q = r / self.h;
        if q >= 2.0 {
            0.0
        } else {
            let term = 2.0 - q;
            term * term * term * (1.0 + 1.5 * q) * self.normal_factor
        }
    }
    fn gradient(&self, r_vec: Vec3, r: f32) -> Vec3 {
        let q = r / self.h;
        if q >= 2.0 || r < 1e-7 {
            Vec3::ZERO
        } else {
            let term = 2.0 - q;
            let dw_dq = -4.5 * term * term;
            let dw_dr = dw_dq * self.normal_factor / self.h;
            (r_vec / r) * dw_dr
        }
    }
}