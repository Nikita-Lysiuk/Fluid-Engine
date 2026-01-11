use std::collections::HashSet;
use winit::keyboard::KeyCode;
use crate::commands::camera_commands::{MoveForwardCameraCommand, MoveRightCameraCommand};
use crate::commands::Command;

pub struct Controller {
    pressed_keys: HashSet<KeyCode>,
    mouse_delta: (f32, f32),
}
pub enum KeyboardAction {
    Pressed,
    Released,
}
impl Controller {
    pub fn new() -> Self {
        Self {
            pressed_keys: HashSet::new(),
            mouse_delta: (0.0, 0.0)
        }
    }
    pub fn update_key(&mut self, key: KeyCode, action: KeyboardAction) {
        match action {
            KeyboardAction::Pressed => { self.pressed_keys.insert(key); },
            KeyboardAction::Released => { self.pressed_keys.remove(&key); },
        }
    }
    pub fn update_mouse_delta(&mut self, delta_x: f32, delta_y: f32) {
        self.mouse_delta.0 += delta_x;
        self.mouse_delta.1 += delta_y;
    }
    pub fn get_active_commands(&self) -> Vec<Box<dyn Command>> {
        let mut commands: Vec<Box<dyn Command>> = Vec::new();

        if self.pressed_keys.contains(&KeyCode::KeyW) {
            commands.push(Box::new(MoveForwardCameraCommand::new(true)));
        }
        if self.pressed_keys.contains(&KeyCode::KeyS) {
            commands.push(Box::new(MoveForwardCameraCommand::new(false)));
        }
        if self.pressed_keys.contains(&KeyCode::KeyD) {
            commands.push(Box::new(MoveRightCameraCommand::new(true)));
        }
        if self.pressed_keys.contains(&KeyCode::KeyA) {
            commands.push(Box::new(MoveRightCameraCommand::new(false)));
        }

        commands
    }
}