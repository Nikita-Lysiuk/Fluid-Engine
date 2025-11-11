use std::fs::File;
use ash::util;
use crate::platform_window::window_manager::WindowManager;
use simple_logger::SimpleLogger;
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

/// Main engine struct.
///
/// Serves as the entry point and facade for the application.
/// Manages core components such as the window, event loop, and context.
/// All engine control and coordination is handled through this struct.
pub struct Engine {
    window: WindowManager,
    event_loop: Option<EventLoop<()>>,
}

impl Engine {
    /// Initialize the engine
    /// TODO: parameters like window size, choose context, etc
    pub fn new() -> Self {
        // Logger initialization
        SimpleLogger::new().init().unwrap();

        // Activating event loop
        let event_loop = EventLoop::new().unwrap();
        event_loop.set_control_flow(ControlFlow::Poll);

        let window = WindowManager::default();

        let vect_shader_code = Self::read_shader_code("shaders/simple_shader.vert.spv");
        let frag_shader_code = Self::read_shader_code("shaders/simple_shader.frag.spv");
        println!("Vertex shader code length: {}, Fragment shader code length: {}", vect_shader_code.len(), frag_shader_code.len());
        // Getting final engine struct
        Self { window, event_loop: Some(event_loop) }
    }

    fn read_shader_code(path: &str) -> Vec<u32> {
        let mut spv_file = File::open(path).expect("compiled spv filed should be present");
        util::read_spv(&mut spv_file).unwrap()
    }

    pub fn game_loop(&mut self) -> Result<(), EventLoopError> {
        self.event_loop.take().unwrap().run_app(&mut self.window)
    }
}
