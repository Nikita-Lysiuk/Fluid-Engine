use std::time::Instant;
use log::{info, warn, error};
use crate::platform_window::window_manager::WindowManager;
use simple_logger::SimpleLogger;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{WindowAttributes, WindowId};
use crate::errors::application_error::ApplicationError;
use crate::renderer::Renderer;
use crate::utils::constants::{IS_PAINT_FPS_COUNTER, PREFERRED_FPS, TARGET_FRAME_DURATION, WINDOW_ICON_PATH, WINDOW_SIZE, WINDOW_TITLE};
use crate::utils::fps_counter::FpsCounter;
use crate::utils::loader::Loader;

/// Main Engine Core.
///
/// Serves as the primary entry point and facade for the fluid simulation application.
/// Manages the top-level application flow, including
/// 1. Initialization and execution of the Winit event loop.
/// 2. Lifecycle management of the rendering subsystem (Vulkan/Renderer).
/// 3. Coordination of input events and simulation updates.
pub struct Engine {
    window: WindowManager,
    renderer: Option<Renderer>,
    event_loop: Option<EventLoop<()>>,
    fps_counter: FpsCounter,
    is_focused: bool,
}

impl Engine {
    pub fn new() -> Result<Self, ApplicationError> {
        SimpleLogger::new().init().unwrap();
        info!("[Engine] Logger initialized successfully.");

        Ok(Self {
            window: WindowManager::default(),
            renderer: None,
            event_loop: Some(EventLoop::new()?),
            fps_counter: FpsCounter::new(PREFERRED_FPS),
            is_focused: false,
        })
    }

    fn initialize_window_and_renderer(&mut self, event_loop: &ActiveEventLoop) -> Result<(), ApplicationError> {
        if !self.window.is_window_created() {
            info!("[Engine] Creating primary application window.");

            let attr = WindowAttributes::default()
                .with_title(WINDOW_TITLE)
                .with_inner_size(WINDOW_SIZE)
                .with_window_icon(Some(Loader::load_icon(WINDOW_ICON_PATH)?));
            
            let window = event_loop.create_window(attr)?;
            self.window.create_window(window);
        }
        
        let display_handle = self.window.display_handle()?;
        let window_handle = self.window.window_handle()?;
        
        if self.renderer.is_none() {
            self.renderer = Some(Renderer::new(display_handle)?);
            info!("[Engine] Renderer object initialized.");
        }
        
        info!("[Engine] Creating Vulkan Surface for the window.");
        self.renderer.as_mut().unwrap().handle_presentation(
            self.window.window.as_ref().unwrap(),
            display_handle,
            window_handle,
        )?; 

        Ok(())
    }
    /// Consumes the stored EventLoop to start the application's main thread loop.
    /// This is typically called once in `main()`.
    pub fn event_loop(&mut self) -> EventLoop<()> {
        self.event_loop
            .take()
            .expect("[Engine] Engine initialization guarantees EventLoop is present until ownership is transferred.")
    }
}

impl ApplicationHandler for Engine {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Err(e) = self.initialize_window_and_renderer(event_loop) {
            error!("[Engine] FATAL INITIALIZATION ERROR: {}", e);
            event_loop.exit();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Focused(focused) => {
                self.is_focused = focused;
                if focused { self.fps_counter.tick(); }
                info!("[Engine] Window focus: {}", if focused { "Focused" } else { "Unfocused" });

            }
            WindowEvent::Resized(size) => {
                warn!("[Engine] Window resized to {}x{}. Swapchain recreation required.", size.width, size.height);
                if size.width > 0 && size.height > 0 {
                    if let (Some(renderer), Some(window)) = (self.renderer.as_mut(), self.window.window.as_ref()) {
                        renderer.handle_resize(window).ok();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(renderer), Some(window)) = (self.renderer.as_mut(), self.window.window.as_ref()) {
                    if let Err(e) = renderer.update(window) {
                        error!("[Engine] Renderer failed: {}", e);
                    }
                }
                let _ = self.window.redraw();
            }
            _ => (),
        }
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let window = match self.window.window.as_ref() {
            Some(w) => w,
            None => return,
        };

        let size = window.inner_size();
        let is_minimized = size.width == 0 || size.height == 0;

        if is_minimized || !self.is_focused {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        }

        event_loop.set_control_flow(ControlFlow::Poll);

        let now = Instant::now();
        let next_frame_time = self.fps_counter.last_frame_time() + TARGET_FRAME_DURATION;

        if now >= next_frame_time {
            self.fps_counter.tick();

            // ТУТ БУДЕ: оновлення фізики та камери

            window.request_redraw();

            if IS_PAINT_FPS_COUNTER {
                let _ = self.window.change_window_title(&format!("{} | FPS: {}", WINDOW_TITLE, self.fps_counter.fps()));
            }
        }
    }
    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        info!("[Engine] Application suspended. Destroying window-dependent Vulkan resources (Surface, Swapchain).");
        if let Some(renderer) = self.renderer.as_mut() {
            let _ = renderer.delete_presentation();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        info!("[Engine] Application exiting. Final resource cleanup complete.");
    }
}