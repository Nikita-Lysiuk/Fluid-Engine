use crate::camera::Camera;
use glam::Vec2;
use std::collections::HashSet;
use winit::event::ElementState;
use winit::keyboard::KeyCode;

const CAMERA_MOVE_SPEED: f32 = 5.0;

/// Keyboard and mouse input used to move the camera.
pub struct Input {
    pressed_keys: HashSet<KeyCode>,
    mouse_delta: Vec2,
}

impl Default for Input {
    fn default() -> Self {
        Self {
            pressed_keys: HashSet::new(),
            mouse_delta: Vec2::ZERO,
        }
    }
}

impl Input {
    /// Clears all held keys, such as when the window loses focus.
    pub fn release_all_keys(&mut self) {
        self.pressed_keys.clear();
    }
    /// Updates the pressed state of a key.
    pub fn key_event(&mut self, key: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.pressed_keys.insert(key);
            }
            ElementState::Released => {
                self.pressed_keys.remove(&key);
            }
        }
    }

    /// Adds mouse movement to the current frame's camera input.
    pub fn mouse_motion(&mut self, dx: f32, dy: f32) {
        self.mouse_delta += Vec2::new(dx, dy);
    }

    /// Applies this frame's input to the camera and clears mouse movement.
    pub fn apply_to_camera(&mut self, camera: &mut Camera, dt: f32) {
        if self.pressed_keys.contains(&KeyCode::KeyW) {
            camera.move_by(camera.forward(), CAMERA_MOVE_SPEED * dt);
        }
        if self.pressed_keys.contains(&KeyCode::KeyS) {
            camera.move_by(-camera.forward(), CAMERA_MOVE_SPEED * dt);
        }
        if self.pressed_keys.contains(&KeyCode::KeyD) {
            camera.move_by(camera.right(), CAMERA_MOVE_SPEED * dt);
        }
        if self.pressed_keys.contains(&KeyCode::KeyA) {
            camera.move_by(-camera.right(), CAMERA_MOVE_SPEED * dt);
        }
        if self.pressed_keys.contains(&KeyCode::KeyE) {
            camera.move_by(camera.up(), CAMERA_MOVE_SPEED * dt);
        }
        if self.pressed_keys.contains(&KeyCode::KeyQ) {
            camera.move_by(-camera.up(), CAMERA_MOVE_SPEED * dt);
        }
        if self.mouse_delta != Vec2::ZERO {
            camera.rotate_by(self.mouse_delta.x, self.mouse_delta.y);
        }
        self.mouse_delta = Vec2::ZERO;
    }
}
