use crate::commands::Command;
use crate::core::scene::Scene;
use crate::entities::Actor;

const CAMERA_MOVE_SPEED: f32 = 5.0;

pub struct MoveForwardCameraCommand {
    is_forward: bool,
}
impl MoveForwardCameraCommand {
    pub fn new(is_forward: bool) -> Self {
        Self { is_forward }
    }
}
impl Command for MoveForwardCameraCommand {
    fn execute(&self, scene: &mut Scene, dt: f32) {
        let forward = scene.camera.forward() * if self.is_forward { 1.0 } else { -1.0 };
        scene.camera.add_input_vector(forward, CAMERA_MOVE_SPEED * dt);
    }
}

pub struct MoveRightCameraCommand {
    is_right: bool,
}
impl MoveRightCameraCommand {
    pub fn new(is_right: bool) -> Self {
        Self { is_right }
    }
}
impl Command for MoveRightCameraCommand {
    fn execute(&self, scene: &mut Scene, dt: f32) {
        let right = scene.camera.right() * if self.is_right { 1.0 } else { -1.0 };
        scene.camera.add_input_vector(right, CAMERA_MOVE_SPEED * dt);
    }
}

pub struct RotateCameraCommand {
    dx: f32,
    dy: f32,
}
impl RotateCameraCommand {
    pub fn new(dx: f32, dy: f32) -> Self {
        Self { dx, dy }
    }
}
impl Command for RotateCameraCommand {
    fn execute(&self, scene: &mut Scene, _dt: f32) {
        scene.camera.add_rotation(self.dx, self.dy);
    }
}