use log::{info, debug, warn, error};
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
/// Manages the top-level application flow, including:
/// 1. Initialization and execution of the Winit event loop.
/// 2. Lifecycle management of the rendering subsystem (Vulkan/Renderer).
/// 3. Coordination of input events and simulation updates.
pub struct Engine {
    window: WindowManager,
    renderer: Option<Renderer>,
    event_loop: Option<EventLoop<()>>,
    fps_counter: FpsCounter
}

impl Engine {
    pub fn new() -> Result<Self, ApplicationError> {
        SimpleLogger::new().init().unwrap();
        info!("[Init] Logger initialized successfully.");

        let window = WindowManager::default();

        let event_loop = EventLoop::new()?;

        event_loop.set_control_flow(ControlFlow::Wait);
        info!("[Engine] Event Loop initialized. Control Flow set to: Wait");

        Ok(Self {
            window,
            renderer: None,
            event_loop: Some(event_loop),
            fps_counter: FpsCounter::new(PREFERRED_FPS)
        })
    }

    fn initialize_window_and_renderer(&mut self, event_loop: &ActiveEventLoop) -> Result<(), ApplicationError> {
        if !self.window.is_window_created() {
            info!("[Winit] Creating primary application window.");

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
            info!("[Vulkan] Initializing core Renderer.");
            self.renderer = Some(Renderer::new(display_handle)?);
            debug!("[Vulkan] Renderer object initialized.");
        }
        
        info!("[Vulkan] Creating Vulkan Surface for the window.");
        self.renderer.as_mut().unwrap().handle_presentation(
            display_handle,
            window_handle,
        )?; 

        Ok(())
    }

    /// Consumes the stored EventLoop to start the application's main thread loop.
    /// This is typically called once in `main()`.
    pub fn event_loop(&mut self) -> EventLoop<()> {
        debug!("[Engine] Transferring ownership of the Winit Event Loop.");
        self.event_loop
            .take()
            .expect("Engine initialization guarantees EventLoop is present until ownership is transferred.")
    }
    
}

impl ApplicationHandler for Engine {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        match self.initialize_window_and_renderer(event_loop) {
            Ok(_) => {},
            Err(e) => {
                error!("FATAL INITIALIZATION ERROR: {}", e);
                // Тут ми ловимо помилку і ініціюємо вихід, якщо ініціалізація не вдалася
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                info!("[Winit] Close request received. Initiating event loop exit.");
                event_loop.exit();
            },
            WindowEvent::RedrawRequested => {
                let dt = self.fps_counter.tick();

                let frame_end_time = self.fps_counter.last_frame_time() + TARGET_FRAME_DURATION;
                event_loop.set_control_flow(ControlFlow::WaitUntil(frame_end_time));
                
                if let Err(e) = self.window.redraw() {
                    error!("FATAL: Redraw failed: {}", e);
                    event_loop.exit();
                    return;
                }

                if IS_PAINT_FPS_COUNTER {
                    let fps_str = self.fps_counter.fps().to_string();
                    if let Err(e) = self.window.change_window_title(&fps_str) {
                        warn!("Could not update window title: {}", e);
                    }
                }
            },
            WindowEvent::Resized(size) => {
                warn!("[Window] Window resized to {}x{}. Swapchain recreation required.", size.width, size.height);
                // TODO: Handle Swapchain and frame buffer recreation.
            }
            _ => {
                // debug!("[Winit] Unhandled window event: {:?}", event);
            },
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        info!("[Vulkan] Application suspended. Destroying window-dependent Vulkan resources (Surface, Swapchain).");
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.delete_presentation();
            debug!("[Vulkan] Surface successfully destroyed.");
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        info!("[Engine] Application exiting. Final resource cleanup complete.");
    }
}