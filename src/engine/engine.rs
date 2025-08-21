use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};
use simple_logger::SimpleLogger;
use crate::platform_window::window_app::WindowApp;

/// Main engine struct.
///
/// Serves as the entry point and facade for the application.
/// Manages core components such as the window, event loop, and context.
/// All engine control and coordination is handled through this struct.
pub struct Engine {
    app: WindowApp,
    event_loop: EventLoop<()>,
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

        // Getting final engine struct
        Self {
            app: WindowApp::default(),
            event_loop
        }
    }

    pub fn game_loop(mut self) -> Result<(), EventLoopError> {
        self.event_loop.run_app(&mut self.app)
    }
}