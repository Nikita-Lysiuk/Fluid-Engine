

use crate::core::scene::Scene;

pub mod camera_commands;

pub trait Command {
    fn execute(&self, scene: &mut Scene, dt: f32);
}

