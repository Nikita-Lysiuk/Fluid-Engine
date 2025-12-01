use std::error::Error;
use ash::vk::{Handle, SurfaceKHR, SwapchainKHR};
use log::{debug, error, info, warn};
use winit::raw_window_handle::{DisplayHandle, WindowHandle};
use crate::renderer::instance;

pub struct SwapchainHandler {
    surface_loader: ash::khr::surface::Instance,
    surface: Option<SurfaceKHR>,
    swapchain: Option<SwapchainKHR>,
}

impl SwapchainHandler {
    pub fn new(instance_ctx: &instance::VulkanInstanceContext) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let surface_loader = ash::khr::surface::Instance::new(&instance_ctx.entry, &instance_ctx.instance);
            info!("[Vulkan] Surface extension loader initialized.");
            
            Ok(SwapchainHandler {
                surface_loader,
                surface: None,
                swapchain: None,
            })
        }
    }
    pub unsafe fn create_surface(&mut self, instance_ctx: &instance::VulkanInstanceContext, display_handle: DisplayHandle, window_handle: WindowHandle) -> Result<(), &'static str> {
        info!("[Surface] Attempting to create new Vulkan Surface.");
        unsafe {
            let surface = ash_window::create_surface(
                &instance_ctx.entry,
                &instance_ctx.instance,
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            ).map_err(|e| {
                error!("[Surface] FATAL: Surface creation failed: {:?}", e);
                panic!("Failed to create surface: {:?}", e);
            }).unwrap();

            if let Some(old_surface) = self.surface.replace(surface) {
                warn!("[Surface] Logic error: Surface creation detected an existing Surface ({:?}).", old_surface.as_raw());
                self.surface.replace(old_surface);
                return Err("Surface creation called when an existing Surface was already present. Did 'suspended' not run?");
            }

            info!("[Surface] Vulkan Surface created successfully. Handle: {:?}", surface.as_raw());
            Ok(())
        }
    }
    pub unsafe fn destroy_surface(&mut self) {
        if let Some(surface) = self.surface.take() {
            info!("[Surface] Destroying Vulkan Surface. Handle: {:?}", surface.as_raw());
            unsafe { self.surface_loader.destroy_surface(surface, None); }
            info!("[Surface] Surface destroyed successfully.");
        } else {
            debug!("[Surface] Surface destroy called, but Surface was already None.");
        }
    }
    pub unsafe fn create_swapchain() -> Result<SwapchainKHR, Box<dyn Error>> {
        unimplemented!()
    }
}