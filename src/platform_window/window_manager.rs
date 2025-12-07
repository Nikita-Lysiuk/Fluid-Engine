
use winit::raw_window_handle::{DisplayHandle, HasDisplayHandle, HasWindowHandle, WindowHandle};
use winit::window::Window;
use log::warn;
use crate::utils::constants::WINDOW_TITLE;

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
    pub fn change_window_title(&self, title: &String) {
        let new_title = format!("{} | FPS: {}", WINDOW_TITLE, title);
        
        self.window.as_ref().expect("Cannot change window title. Window is not created!").set_title(new_title.as_str());
    }
    pub fn redraw(&self) {
        self.window.as_ref().expect("Cannot redraw window. Window is not created!").request_redraw();
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