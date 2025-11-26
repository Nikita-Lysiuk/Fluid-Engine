
use winit::raw_window_handle::{DisplayHandle, HasDisplayHandle, HasWindowHandle, WindowHandle};
use winit::window::Window;
use log::warn;

#[derive(Default)]
pub struct WindowManager {
    window: Option<Window>
}

impl WindowManager {
    pub fn is_window_created(&self) -> bool {
        self.window.is_some()
    }
    pub fn create_window(&mut self, window: Window) {
        self.window = Some(window);
    }
    pub fn window_handle(&self) -> Option<WindowHandle> {
        self.window
            .as_ref()
            .and_then(|window| {
                window.window_handle().map_or_else(
                    |err| {
                        warn!("getting window handle: {}", err);
                        None
                    },
                    Some,
                )
            })
    }
    pub fn display_handle(&self) -> Option<DisplayHandle> {
        self.window
            .as_ref()
            .and_then(|window| {
                window.display_handle().map_or_else(
                    |err| {
                        warn!("getting display handle: {}", err);
                        None
                    },
                    Some,
                )
            })
    }
}