
use std::time::Duration;
use winit::dpi::{PhysicalSize, Size};

pub const WINDOW_ICON_PATH: &'static [u8] = include_bytes!("../../assets/logo.png");
pub const WINDOW_TITLE: &str = "Fluid Simulation Engine";
pub const MAX_FRAMES_IN_FLIGHT: usize = 3;
pub const IS_PAINT_FPS_COUNTER: bool = true; 
pub const PREFERRED_FPS: u32 = 120;
pub const TARGET_FRAME_DURATION: Duration = Duration::from_nanos(1_000_000_000 / PREFERRED_FPS as u64);

