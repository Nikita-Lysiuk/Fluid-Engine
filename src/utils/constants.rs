use std::ffi::CStr;
use std::os::raw::c_char;
use std::time::Duration;
use ash::ext::debug_utils;
use ash::vk::make_api_version;
use winit::dpi::{PhysicalSize, Size};

pub const APPLICATION_NAME: *const c_char = c"Engine for Fluid Simulation".as_ptr();
pub const APPLICATION_VERSION: u32 = make_api_version(0, 1, 3, 0);
pub const ENGINE_VERSION: u32 = make_api_version(1, 0, 0, 0);

pub const WINDOW_TITLE: &str = "Fluid Simulation Engine";
pub const WINDOW_ICON_PATH: &'static [u8] = include_bytes!("../../assets/logo.png");
pub const WINDOW_SIZE: Size = Size::Physical(PhysicalSize { width: 1280, height: 720 });
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
pub const IS_PAINT_FPS_COUNTER: bool = true; 
pub const PREFERRED_FPS: u32 = 120;
pub const TARGET_FRAME_DURATION: Duration = Duration::from_nanos(8_333_333);
pub const VALIDATION_LAYERS: [*const c_char; 1] = [
    c"VK_LAYER_KHRONOS_validation".as_ptr()
];
pub const DEVICE_EXTENSIONS: [*const c_char; 1] = [
ash::khr::swapchain::NAME.as_ptr()
];
pub const ENABLE_VALIDATION_LAYERS: bool = cfg!(debug_assertions);
pub const DEBUG_UTILS_EXTENSION_NAME: &CStr = debug_utils::NAME;