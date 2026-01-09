use log::{info, debug, error};
use simple_logger::SimpleLogger;
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalSize, Size};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::platform::windows::WindowAttributesExtWindows;
use winit::window::{Icon, WindowAttributes, WindowId};

use crate::renderer::Renderer;

use crate::errors::application_error::ApplicationError;
use crate::utils::constants::{IS_PAINT_FPS_COUNTER, PREFERRED_FPS, TARGET_FRAME_DURATION, WINDOW_ICON_PATH, WINDOW_TITLE};
use crate::utils::fps_counter::FpsCounter;

fn load_icon(bytes: &[u8]) -> Result<Icon, ApplicationError> {
    let (icon_rgba, icon_width, icon_height) = {
        let image = image::load_from_memory(bytes).unwrap().into_rgba8();
        let (width, height) = image.dimensions();
        let rgba = image.into_raw();
        (rgba, width, height)
    };
    Icon::from_rgba(icon_rgba, icon_width, icon_height).map_err(|e| {
        ApplicationError::ResourceLoadError(format!("Failed to create icon from RGBA data: {}", e))
    })
}

pub struct Engine {
    renderer: Option<Renderer>,
    event_loop: Option<EventLoop<()>>,
    fps_counter: FpsCounter,
    is_focused: bool,
}

impl Engine {
    pub fn new() -> Result<Self, ApplicationError> {
        SimpleLogger::new().init().unwrap();
        info!("[Engine] Initializing Engine Core...");

        let event_loop = EventLoop::new()
            .map_err(|e| ApplicationError::EventLoopInitializationError(e))?;

        event_loop.set_control_flow(ControlFlow::Poll);

        Ok(Self {
            renderer: None,
            event_loop: Some(event_loop),
            fps_counter: FpsCounter::new(PREFERRED_FPS),
            is_focused: true,
        })
    }

    pub fn run(&mut self) {
        let event_loop = self.event_loop.take()
            .expect("[Engine] Event Loop missing! (already consumed?)");

        debug!("[Engine] Starting Winit Event Loop.");
        let _ = event_loop.run_app(self);
    }
}

impl ApplicationHandler for Engine {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_none() {
            info!("[Engine] System Resumed. Initializing Window and Graphics...");

            let window_attributes = WindowAttributes::default()
                .with_inner_size(Size::Physical(PhysicalSize { width: 1280, height: 720 }))
                .with_window_icon(load_icon(WINDOW_ICON_PATH).ok())
                .with_taskbar_icon(load_icon(WINDOW_ICON_PATH).ok());


            let window = match event_loop.create_window(window_attributes) {
                Ok(w) => w,
                Err(e) => {
                    error!("[Engine] Failed to create window: {}", e);
                    event_loop.exit();
                    return;
                }
            };
            self.renderer = Some(Renderer::new(window));
            info!("[Engine] Systems initialized successfully.");
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                info!("[Engine] Exit requested.");
                event_loop.exit();
            }

            WindowEvent::Focused(focused) => {
                self.is_focused = focused;
                debug!("[Engine] Focus changed: {}", focused);
            }

            WindowEvent::Resized(_new_size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.window_renderer.resize();
                }
            }

            WindowEvent::RedrawRequested => {
                let dt = self.fps_counter.tick();

                if !self.is_focused {
                    event_loop.set_control_flow(ControlFlow::WaitUntil(
                        std::time::Instant::now() + TARGET_FRAME_DURATION
                    ));
                    return;
                }

                if IS_PAINT_FPS_COUNTER {
                    if let Some(renderer) = self.renderer.as_ref() {
                        let title = format!("{} | FPS: {}", WINDOW_TITLE, self.fps_counter.fps());
                        renderer.window_renderer.window().set_title(&title);
                    }
                }
            }

            _ => (),
        }

        if self.is_focused && self.renderer.is_some() {
            if let Some(renderer) = self.renderer.as_ref() {
                renderer.window_renderer.window().request_redraw();
            }
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        info!("[Engine] Graceful shutdown...");
    }
}