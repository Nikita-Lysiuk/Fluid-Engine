use log::{info, debug, error};
use simple_logger::SimpleLogger;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::platform::windows::WindowAttributesExtWindows;
use winit::window::{CursorGrabMode, Icon, WindowAttributes, WindowId};
use crate::core::controller::{Controller, KeyboardAction};
use crate::core::scene::Scene;
use crate::renderer::Renderer;

use crate::errors::application_error::ApplicationError;
use crate::physics::PhysicsEngine;
use crate::utils::constants::{IS_PAINT_FPS_COUNTER, PREFERRED_FPS, WINDOW_TITLE};
use crate::utils::fps_counter::FpsCounter;

fn load_icon(path: &str) -> Result<Icon, ApplicationError> {
    let (icon_rgba, icon_width, icon_height) = {
        let image = image::open(path)
            .map_err(|e| ApplicationError::ResourceLoadError(format!("Failed to parse icon image: {}", e)))?
            .into_rgba8();
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
    scene: Scene,
    physics_engine: PhysicsEngine,
    controller: Controller,
    fps_counter: FpsCounter,
    is_focused: bool,
    accumulator: f32,
}

impl Engine {
    pub fn new() -> Result<Self, ApplicationError> {
        SimpleLogger::new().init().unwrap();
        info!("[Engine] Initializing Engine Core...");

        let event_loop = EventLoop::new()
            .map_err(|e| ApplicationError::EventLoopInitializationError(e))?;

        event_loop.set_control_flow(ControlFlow::Poll);
        let scene = Scene::new();
        let physics_engine = PhysicsEngine::new(scene.smoothing_length);

        Ok(Self {
            renderer: None,
            event_loop: Some(event_loop),
            scene,
            physics_engine,
            fps_counter: FpsCounter::new(PREFERRED_FPS),
            controller: Controller::new(),
            is_focused: true,
            accumulator: 0.0,
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
                .with_inner_size(PhysicalSize::new(1280, 720))
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

                if let Some(renderer) = self.renderer.as_mut() {
                    let window = renderer.window_renderer.window();
                    if focused {
                        window.set_cursor_grab(CursorGrabMode::Locked)
                            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
                            .unwrap_or_else(|e| error!("Failed to grab cursor: {}", e));
                        window.set_cursor_visible(false);
                    } else {
                        window.set_cursor_grab(CursorGrabMode::None).ok();
                        window.set_cursor_visible(true);
                    }
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !self.is_focused {
                    self.is_focused = true;
                    info!("[Engine] Window clicked, restoring focus and cursor grab.");

                    if let Some(renderer) = self.renderer.as_mut() {
                        let window = renderer.window_renderer.window();
                        window.set_cursor_grab(CursorGrabMode::Locked)
                            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
                            .unwrap_or_else(|e| error!("Failed to grab cursor: {}", e));
                        window.set_cursor_visible(false);
                    }
                }
            }
            WindowEvent::Resized(_new_size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.window_renderer.resize();
                }
            }
            WindowEvent::KeyboardInput {
                event: key_event,
                ..
            } => {
                match (key_event.physical_key, key_event.state) {
                    (PhysicalKey::Code(KeyCode::Escape), ElementState::Pressed) => {
                        self.is_focused = false;
                        if let Some(renderer) = self.renderer.as_mut() {
                            renderer.window_renderer.window().set_cursor_grab(CursorGrabMode::None).ok();
                            renderer.window_renderer.window().set_cursor_visible(true);
                        }
                    }
                    (physical_key, ElementState::Pressed) => {
                        if let PhysicalKey::Code(code)  = physical_key {
                            self.controller.update_key(code, KeyboardAction::Pressed);
                        }
                    }
                    (physical_key, ElementState::Released) => {
                        if let PhysicalKey::Code(code)  = physical_key {
                            self.controller.update_key(code, KeyboardAction::Released);
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let dt = self.fps_counter.tick().as_secs_f32();

                let fixed_dt = 0.0166;

                let max_substeps = 5;

                self.accumulator += dt;

                if self.accumulator > 0.1 {
                    self.accumulator = 0.1;
                }

                let mut steps = 0;
                while self.accumulator >= fixed_dt && steps < max_substeps {
                    for command in self.controller.get_active_commands() {
                        command.execute(&mut self.scene, fixed_dt);
                    }
                    if let Some(cmd) = self.controller.get_mouse_command() {
                        cmd.execute(&mut self.scene, fixed_dt);
                    }

                    self.physics_engine.update(&mut self.scene, fixed_dt);

                    self.accumulator -= fixed_dt;
                    steps += 1;
                }

                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.render(&self.scene);
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
    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                if self.is_focused {
                    let sensitivity = 0.1;
                    let dx = delta.0 as f32 * sensitivity;
                    let dy = delta.1 as f32 * sensitivity;

                    self.controller.update_mouse_delta(dx, dy);
                }
            }
            _ => {}
        }
    }
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        info!("[Engine] Graceful shutdown...");
    }
}