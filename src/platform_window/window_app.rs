use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::raw_window_handle::{DisplayHandle, HasDisplayHandle, HasWindowHandle, WindowHandle};
use winit::window::{Window, WindowAttributes, WindowId};
use log::{trace, warn};

#[derive(Default)]
pub struct WindowApp {
    window: Option<Window>
}

impl ApplicationHandler for WindowApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.window = Some(event_loop.create_window(WindowAttributes::default()).unwrap())
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                trace!("Window close requested. Initiating application shutdown.");
                event_loop.exit();
            },
            WindowEvent::RedrawRequested => {
                self.window.as_ref().unwrap().request_redraw();
            },
            _ => (),
        }
    }
}

impl WindowApp {
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