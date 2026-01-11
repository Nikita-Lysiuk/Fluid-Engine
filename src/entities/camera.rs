use glam::{Mat4, Quat, Vec3};
use vulkano::buffer::BufferContents;
use crate::entities::Actor;

#[derive(BufferContents, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShaderData {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    _padding: f32,
}


pub struct Camera {
    pub position: Vec3,
    pub velocity: Vec3,
    pub orientation: Quat,
    pub fov: f32,
    pub aspect_ratio: f32,
    pub near: f32,
    pub far: f32,
}

impl Actor for Camera {
    type ShaderDataType = ShaderData;
    fn update(&mut self, dt: f32) {
        // Camera update logic can be implemented here
    }
    fn build_shader_data(&self) -> ShaderData {
        let view = self.get_view_matrix();
        let proj = self.get_projection_matrix();

        ShaderData {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            inv_view_proj: (proj * view).inverse().to_cols_array_2d(),
            camera_pos: self.position.into(),
            _padding: 0.0,
        }
    }
}

impl Camera {
    fn get_view_matrix(&self) -> Mat4 {
        let rotation_matrix = Mat4::from_quat(self.orientation);
        let translation_matrix = Mat4::from_translation(-self.position);
        rotation_matrix * translation_matrix
    }

    fn get_projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh_gl(self.fov.to_radians(), self.aspect_ratio, self.near, self.far)
    }
}

