
use winit::raw_window_handle::{DisplayHandle, HasDisplayHandle, HasWindowHandle, WindowHandle};
use winit::window::Window;
use log::warn;
use crate::errors::engine_error::EngineError;
use crate::utils::constants::WINDOW_TITLE;

#[derive(Default)]
pub struct WindowManager {
    pub window: Option<Window>
}

impl WindowManager {
    pub fn is_window_created(&self) -> bool {
        self.window.is_some()
    }
    pub fn create_window(&mut self, window: Window) {
        self.window = Some(window);
    }
    pub fn change_window_title(&self, title: &String) -> Result<(), EngineError> {
        let new_title = format!("{} | FPS: {}", WINDOW_TITLE, title);

        self.window.as_ref()
            .ok_or(EngineError::WindowManagement("Cannot change window title: Window is not created!".to_string()))?
            .set_title(new_title.as_str());

        Ok(())
    }
    pub fn redraw(&self) -> Result<(), EngineError> {
        self.window.as_ref()
            .ok_or(EngineError::WindowManagement("Cannot redraw window: Window is not created!".to_string()))?
            .request_redraw();

        Ok(())
    }
    pub fn window_handle(&self) -> Result<WindowHandle, EngineError> {
        self.window
            .as_ref()
            .ok_or(EngineError::HandleMissing("Window manager is not initialized.".to_string()))?
            .window_handle()
            .map_err(|e| {
                EngineError::HandleMissing(format!("Window handle retrieval failed: {}", e))
            })
    }
    pub fn display_handle(&self) -> Result<DisplayHandle, EngineError> {
        self.window
            .as_ref()
            .ok_or(EngineError::HandleMissing("Window manager is not initialized.".to_string()))?
            .display_handle()
            .map_err(|err| {
                EngineError::HandleMissing(format!("Getting display handle failed: {}", err))
            })
    }
}