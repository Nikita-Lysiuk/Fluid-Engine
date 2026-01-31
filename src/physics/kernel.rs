use std::f32::consts::PI;
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
            2 => 5.0 / (32.0 * PI * h.powi(2)),
            3 => 105.0 / (1024.0 * PI * h.powi(3)),
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
            let inv_r = 1.0 / r;
            let g_q = dw_dr * inv_r;
            r_vec * g_q
        }
    }
}

pub struct CubicSplineKernel {
    radius: f32, // Support Radius
    k: f32,      // Normalization factor for evaluate
    l: f32,      // Normalization factor for gradient
}

impl CubicSplineKernel {
    pub fn new(radius: f32) -> Self {
        let h3 = radius.powi(3);

        // Коефіцієнти для 3D (з файлу SPHKernels.h)
        // m_k = 8.0 / (pi * h^3)
        let k = 8.0 / (PI * h3);

        // m_l = 48.0 / (pi * h^3)
        let l = 48.0 / (PI * h3);

        Self { radius, k, l }
    }
}

impl SmoothingKernel for CubicSplineKernel {
    fn evaluate(&self, r: f32) -> f32 {
        let q = r / self.radius;
        if q > 1.0 {
            0.0
        } else {
            // Формули з SPlisHSPlasH:
            if q <= 0.5 {
                let q2 = q * q;
                let q3 = q2 * q;
                // m_k * (6q^3 - 6q^2 + 1)
                self.k * (6.0 * q3 - 6.0 * q2 + 1.0)
            } else {
                // m_k * (2 * (1 - q)^3)
                let factor = 1.0 - q;
                self.k * 2.0 * factor.powi(3)
            }
        }
    }

    fn gradient(&self, r_vec: Vec3, r: f32) -> Vec3 {
        let q = r / self.radius;
        if r < 1e-6 || q > 1.0 {
            Vec3::ZERO
        } else {
            // Вектор градієнту: gradq = r_vec / (r * radius)
            let grad_q = r_vec / (r * self.radius);

            let factor = if q <= 0.5 {
                // Похідна від (6q^3 - 6q^2 + 1) = (18q^2 - 12q) = 6q(3q - 2)
                // Множимо на k: k * 6 = l
                // Результат: l * q * (3q - 2)
                self.l * q * (3.0 * q - 2.0)
            } else {
                // Похідна від 2(1-q)^3 = -6(1-q)^2
                // Множимо на k: k * (-6) = -l
                // Результат: -l * (1-q)^2
                let f = 1.0 - q;
                -self.l * f * f
            };

            grad_q * factor
        }
    }
}