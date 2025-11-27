use std::os::raw::c_char;
use ash::vk::make_api_version;
use winit::dpi::{PhysicalSize, Size};

pub const APPLICATION_NAME: *const c_char = c"Engine for Fluid Simulation".as_ptr();
pub const APPLICATION_VERSION: u32 = make_api_version(1, 4, 321, 1);
pub const ENGINE_VERSION: u32 = make_api_version(1, 0, 0, 0);

pub const WINDOW_TITLE: &str = "Fluid Simulation Engine";
pub const WINDOW_ICON_PATH: &'static [u8] = include_bytes!("../../assets/logo.png");
pub const WINDOW_SIZE: Size = Size::Physical(PhysicalSize { width: 1280, height: 720 });
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
pub const IS_PAINT_FPS_COUNTER: bool = false;
pub const PREFERRED_FPS: u32 = 120;
