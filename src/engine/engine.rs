use log::{info, debug, warn, error};
use crate::platform_window::window_manager::WindowManager;
use crate::engine::renderer::Renderer;
use simple_logger::SimpleLogger;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{WindowAttributes, WindowId};
use crate::utils::constants::{IS_PAINT_FPS_COUNTER, WINDOW_ICON_PATH, WINDOW_SIZE, WINDOW_TITLE};
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
}

impl Engine {
    pub fn new() -> Self {
        SimpleLogger::new().init().unwrap();
        info!("[Init] Logger initialized successfully.");

        let window = WindowManager::default();

        let event_loop = EventLoop::new().map_err(|e| {
            error!("[Winit] Failed to create event loop: {}", e);
            panic!("Failed to create event loop: {}", e);
        }).unwrap();

        event_loop.set_control_flow(ControlFlow::Poll);
        info!("[Engine] Event Loop initialized. Control Flow set to: Poll");

        Self {
            window,
            renderer: None,
            event_loop: Some(event_loop),
        }
    }

    /// Consumes the stored EventLoop to start the application's main thread loop.
    /// This is typically called once in `main()`.
    pub fn event_loop(&mut self) -> EventLoop<()> {
        debug!("[Engine] Transferring ownership of the Winit Event Loop.");
        self.event_loop.take().expect("Engine initialization guarantees EventLoop is present until ownership is transferred.")
    }
    
}

impl ApplicationHandler for Engine {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !self.window.is_window_created() {
            info!("[Winit] Creating primary application window.");
   
            let attr = WindowAttributes::default()
                .with_title(WINDOW_TITLE)
                .with_inner_size(WINDOW_SIZE)
                .with_window_icon(Loader::load_icon(WINDOW_ICON_PATH));
            self.window.create_window(
                event_loop.create_window(attr).expect("Winit failed to create the native window handle.")
            );
        }

        let display_handle = self.window.display_handle().expect("Display Handle must be present after window creation.");
        let window_handle = self.window.window_handle().expect("Window Handle must be present after window creation.");

        unsafe {
            if self.renderer.is_none() {
                info!("[Vulkan] Initializing core Renderer.");
                self.renderer = Some(Renderer::new(display_handle).expect("Vulkan Instance creation failed. Check driver/API support."));
                debug!("[Vulkan] Renderer object initialized.");
            }

            info!("[Vulkan] Creating Vulkan Surface for the window.");
            self.renderer.as_mut().unwrap().create_surface(
                display_handle,
                window_handle,
            ).expect("Failed to create Vulkan Surface from Winit handles.");

            // TODO: Add initialization of Logical Device, Swapchain, and render targets here.
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                info!("[Winit] Close request received. Initiating event loop exit.");
                event_loop.exit();
            },
            WindowEvent::RedrawRequested => {
                debug!("[Render] Redraw requested by the OS/Application. Rendering frame...");
                //TODO SPH physics computation and SDF Raymarching rendering.
                
                if IS_PAINT_FPS_COUNTER {
                    // TODO Implement FPS counter.
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
        unsafe {
            if let Some(renderer) = self.renderer.as_mut() {
                // TODO: Add renderer.destroy_swapchain();
                renderer.destroy_surface();
                debug!("[Vulkan] Surface successfully destroyed.");
            }
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        info!("[Engine] Application exiting. Final resource cleanup complete.");
    }
}