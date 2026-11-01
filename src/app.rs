use crate::input::Input;
use crate::render::renderer::Renderer;
use crate::scene::Scene;
use glam::Vec2;
use log::{debug, error, info};
use simple_logger::SimpleLogger;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{
    DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::platform::windows::WindowAttributesExtWindows;
use winit::window::{CursorGrabMode, Icon, WindowAttributes, WindowId};

use crate::app_error::AppError;
use crate::constants::PREFERRED_FPS;
use crate::frame_benchmark::FrameBenchmark;
use crate::frame_timer::FrameTimer;

fn load_icon(path: &str) -> Result<Icon, AppError> {
    let (icon_rgba, icon_width, icon_height) = {
        let image = image::open(path)
            .map_err(|e| AppError::ResourceLoad(format!("Failed to parse icon image: {}", e)))?
            .into_rgba8();
        let (width, height) = image.dimensions();
        let rgba = image.into_raw();
        (rgba, width, height)
    };
    Icon::from_rgba(icon_rgba, icon_width, icon_height)
        .map_err(|e| AppError::ResourceLoad(format!("Failed to create icon from RGBA data: {}", e)))
}

pub struct App {
    renderer: Option<Renderer>,
    event_loop: Option<EventLoop<()>>,
    scene: Scene,
    input: Input,
    fps_counter: FrameTimer,
    /// Right mouse button held: mouse motion rotates the camera.
    rotating_camera: bool,
    /// Cursor position in physical pixels.
    cursor: Vec2,
    pointer_over_ui: bool,
    benchmark: Option<FrameBenchmark>,
}

impl App {
    pub fn new() -> Result<Self, AppError> {
        SimpleLogger::new().init().unwrap();
        info!("[Engine] Initializing Engine Core...");

        let event_loop = EventLoop::new().map_err(|e| AppError::EventLoopInit(e))?;

        event_loop.set_control_flow(ControlFlow::Poll);
        let particle_count = std::env::var("FLUID_PARTICLES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(Scene::DEFAULT_PARTICLES);
        let scene = Scene::new(particle_count);

        Ok(Self {
            renderer: None,
            event_loop: Some(event_loop),
            scene,
            fps_counter: FrameTimer::new(PREFERRED_FPS),
            input: Input::default(),
            rotating_camera: false,
            cursor: Vec2::ZERO,
            pointer_over_ui: true,
            benchmark: FrameBenchmark::from_env(),
        })
    }

    /// Starts or stops camera rotation; the cursor is hidden and kept in the
    /// window while rotating.
    fn set_camera_rotation(&mut self, rotating: bool) {
        self.rotating_camera = rotating;
        if let Some(renderer) = self.renderer.as_ref() {
            let window = renderer.window_renderer.window();
            if rotating {
                window
                    .set_cursor_grab(CursorGrabMode::Locked)
                    .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
                    .unwrap_or_else(|e| error!("Failed to grab cursor: {}", e));
            } else {
                window.set_cursor_grab(CursorGrabMode::None).ok();
            }
            window.set_cursor_visible(!rotating);
        }
    }

    pub fn run(&mut self) {
        let event_loop = self
            .event_loop
            .take()
            .expect("[Engine] Event Loop missing! (already consumed?)");

        debug!("[Engine] Starting Winit Event Loop.");
        let _ = event_loop.run_app(self);
    }
}

impl App {
    fn on_mouse_button(&mut self, button: MouseButton, state: ElementState, over_ui: bool) {
        match (button, state) {
            (MouseButton::Left, ElementState::Pressed) if !over_ui => {
                self.scene.attractor.active = true
            }
            (MouseButton::Left, ElementState::Released) => self.scene.attractor.active = false,
            (MouseButton::Right, ElementState::Pressed) if !over_ui => {
                self.set_camera_rotation(true)
            }
            (MouseButton::Right, ElementState::Released) => self.set_camera_rotation(false),
            _ => {}
        }
    }

    fn on_key(&mut self, key: PhysicalKey, state: ElementState, ui_wants_keyboard: bool) {
        if let PhysicalKey::Code(code) = key {
            match state {
                ElementState::Released => self.input.key_event(code, state),
                ElementState::Pressed if !ui_wants_keyboard => self.input.key_event(code, state),
                ElementState::Pressed => {}
            }
        }
    }

    fn on_redraw(&mut self, event_loop: &ActiveEventLoop) {
        let frame_time = self.fps_counter.tick();
        let dt = frame_time.as_secs_f32();
        let safe_dt = dt.min(0.1);

        self.input.apply_to_camera(&mut self.scene.camera, safe_dt);

        if let Some(renderer) = self.renderer.as_mut() {
            let window_size = Vec2::from(renderer.window_renderer.window_size());
            self.scene.attractor.aim(
                &self.scene.camera,
                self.cursor,
                window_size,
                &self.scene.boundary,
            );
            self.scene.attractor.hovered = !self.pointer_over_ui && !self.rotating_camera;

            renderer.update_and_render(&mut self.scene, safe_dt, self.fps_counter.fps());

            if let Some(benchmark) = self.benchmark.as_mut() {
                if benchmark.record(frame_time, renderer.app_ui.display_substeps) {
                    benchmark.report();
                    event_loop.exit();
                }
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_none() {
            info!("[Engine] System Resumed. Initializing Window and Graphics...");

            let window_attributes = WindowAttributes::default()
                .with_inner_size(PhysicalSize::new(1920, 1080))
                .with_window_icon(load_icon("assets/logo.png").ok())
                .with_taskbar_icon(load_icon("assets/logo.png").ok());

            let window = match event_loop.create_window(window_attributes) {
                Ok(w) => w,
                Err(e) => {
                    error!("[Engine] Failed to create window: {}", e);
                    event_loop.exit();
                    return;
                }
            };
            let mut renderer = Renderer::new(window, &self.scene, event_loop);
            // Lets the benchmark compare substep caps without touching the UI.
            if let Some(cap) = std::env::var("FLUID_BENCH_MAX_SUBSTEPS")
                .ok()
                .and_then(|v| v.parse().ok())
            {
                renderer.app_ui.max_substeps_per_frame = cap;
            }
            if std::env::var("FLUID_BENCH_RENDER").as_deref() == Ok("particles") {
                renderer.app_ui.render_mode = crate::render::ui::RenderMode::Particles;
            }
            self.renderer = Some(renderer);
            info!("[Engine] Systems initialized successfully.");
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let mut egui_wants_pointer = false;
        let mut egui_wants_keyboard = false;

        if let Some(renderer) = self.renderer.as_mut() {
            renderer.gui.update(&event);
            let ctx = renderer.gui.context();
            egui_wants_pointer = ctx.wants_pointer_input() || ctx.is_pointer_over_area();
            egui_wants_keyboard = ctx.wants_keyboard_input();
        }

        match event {
            WindowEvent::CloseRequested => {
                info!("[Engine] Exit requested.");
                event_loop.exit();
            }
            WindowEvent::Focused(false) => {
                self.set_camera_rotation(false);
                self.scene.attractor.active = false;
                self.input.release_all_keys();
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = Vec2::new(position.x as f32, position.y as f32);
                self.pointer_over_ui = egui_wants_pointer;
            }
            WindowEvent::CursorLeft { .. } => {
                self.pointer_over_ui = true;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.on_mouse_button(button, state, egui_wants_pointer)
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if self.scene.attractor.active {
                    let lines = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                    };
                    self.scene.attractor.distance =
                        (self.scene.attractor.distance + 0.2 * lines).clamp(0.3, 20.0);
                }
            }
            WindowEvent::Resized(_new_size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.window_renderer.resize();
                }
            }
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                self.on_key(key_event.physical_key, key_event.state, egui_wants_keyboard);
            }
            WindowEvent::RedrawRequested => self.on_redraw(event_loop),

            _ => (),
        }

        if let Some(renderer) = self.renderer.as_ref() {
            renderer.window_renderer.window().request_redraw();
        }
    }
    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.rotating_camera {
                let sensitivity = 0.1;
                self.input
                    .mouse_motion(delta.0 as f32 * sensitivity, delta.1 as f32 * sensitivity);
            }
        }
    }
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        info!("[Engine] Graceful shutdown...");
    }
}
