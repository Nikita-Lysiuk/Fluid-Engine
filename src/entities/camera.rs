use std::sync::Arc;
use glam::{EulerRot, Mat4, Quat, Vec3};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use crate::entities::Actor;
use crate::utils::constants::MAX_FRAMES_IN_FLIGHT;

#[derive(BufferContents, Debug, Clone, Copy)]
#[repr(C)]
struct CameraGPU {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    _padding: f32,
}

pub struct Camera {
    position: Vec3,
    orientation: Quat,
    fov: f32,
    aspect_ratio: f32,
    near: f32,
    far: f32,
}

impl Actor for Camera {
    #[warn(unused)]
    fn update(&mut self, _dt: f32) {
        // Can be used for effects on every frame if needed
    }
    fn location(&self) -> Vec3 {
        self.position
    }
    fn add_input_vector(&mut self, direction: Vec3, magnitude: f32) {
        self.position += direction * magnitude;
    }
    fn add_rotation(&mut self, dx: f32, dy: f32) {
        let yaw = Quat::from_rotation_y(dx.to_radians());
        let pitch = Quat::from_rotation_x(dy.to_radians());
        self.orientation = yaw * self.orientation * pitch;
        self.orientation = self.orientation.normalize();
    }
}

impl Camera {
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            orientation: Quat::IDENTITY,
            fov: 45.0,
            aspect_ratio: 16.0 / 9.0,
            near: 0.1,
            far: 1000.0,
        }
    }
    pub fn from_euler(pitch: f32, yaw: f32, roll: f32) -> Self {
        let orientation = Quat::from_euler(EulerRot::YXZ, yaw.to_radians(), pitch.to_radians(), roll.to_radians());

        Self {
            position: Vec3::ZERO,
            orientation,
            fov: 60.0,
            aspect_ratio: 16.0 / 9.0,
            near: 0.1,
            far: 100.0,
        }
    }
    pub fn rotate(&mut self, pitch: f32, yaw: f32, roll: f32) {
        let rotation = Quat::from_euler(EulerRot::YXZ, yaw.to_radians(), pitch.to_radians(), roll.to_radians());
        self.orientation = rotation * self.orientation;
        self.orientation = self.orientation.normalize();
    }
    pub fn rotate_absolute(&mut self, pitch: f32, yaw: f32, roll: f32) {
        self.orientation = Quat::from_euler(EulerRot::YXZ, yaw.to_radians(), pitch.to_radians(), roll.to_radians());
        self.orientation = self.orientation.normalize();
    }
    pub fn rotate_with_quat(&mut self, quat: Quat) {
        self.orientation = quat * self.orientation;
        self.orientation = self.orientation.normalize();
    }
    pub fn fov(&mut self, fov: f32) -> &mut Self {
        self.fov = fov;
        self
    }
    pub fn aspect_ratio(&mut self, aspect_ratio: f32) -> &mut Self {
        self.aspect_ratio = aspect_ratio;
        self
    }
    pub fn near(&mut self, near: f32) -> &mut Self {
        self.near = near;
        self
    }
    pub fn far(&mut self, far: f32) -> &mut Self {
        self.far = far;
        self
    }
    pub fn forward(&self) -> Vec3 {
        (self.orientation * Vec3::Z).normalize()
    }
    pub fn right(&self) -> Vec3 {
        (self.orientation * Vec3::X).normalize()
    }
    pub fn up(&self) -> Vec3 {
        (self.orientation * Vec3::Y).normalize()
    }
    fn get_view_matrix(&self) -> Mat4 {
        let target = self.position + self.forward();
        Mat4::look_at_lh(self.position, target, Vec3::Y)
    }
    fn get_projection_matrix(&self) -> Mat4 {
        Mat4::perspective_lh(self.fov.to_radians(), self.aspect_ratio, self.near, self.far)
    }
}

pub struct CameraData {
    uniform_buffers: Vec<Subbuffer<CameraGPU>>,
}

impl CameraData {
    pub fn new(memory_allocator: Arc<StandardMemoryAllocator>) -> Self {
        let mut uniform_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let ub = Buffer::new_sized(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER
                        | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..BufferCreateInfo::default()
                },AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..AllocationCreateInfo::default()
                }).map_err(|e| {
                panic!("[Camera Data] Failed to create uniform buffer:\n{:?}", e);
            }).unwrap();
            uniform_buffers.push(ub);
        }

        Self {
            uniform_buffers,
        }
    }
    pub fn write_to_buffer(&self, camera: &Camera, current_frame_idx: usize) {
        let view = camera.get_view_matrix();
        let proj = camera.get_projection_matrix();
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();

        let camera_gpu = CameraGPU {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos: camera.position.to_array(),
            _padding: 0.0,
        };

        self.uniform_buffers[current_frame_idx]
            .write().map_err(|e| panic!("[Camera Data] Failed to write to uniform buffer:\n{:?}", e))
            .unwrap()
            .clone_from(&camera_gpu);
    }
    pub fn uniform_buffer_addr(&self, current_frame_idx: usize) -> u64 {
        self.uniform_buffers[current_frame_idx]
            .device_address()
            .map_err(|e| panic!("[Camera Data] Failed to get uniform buffer device address:\n{:?}", e))
            .unwrap()
            .get()
    }
}
