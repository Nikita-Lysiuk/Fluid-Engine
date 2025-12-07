use log::{info, warn};
use winit::raw_window_handle::{DisplayHandle, WindowHandle};
use crate::errors::application_error::ApplicationError;
use crate::errors::presentation_error::PresentationError;

mod instance;
mod device;
mod swapchain;

/// Core Vulkan renderer component, managing the Vulkan instance and lifecycle-dependent resources.
pub struct Renderer {
    instance_ctx: instance::VulkanInstanceContext,
    device_ctx: device::DeviceContext,
    swapchain_handler: swapchain::SwapchainHandler,
}

impl Renderer {
    /// Initializes the core, window-independent Vulkan components (Entry, Instance, Surface Loader).
    pub fn new(display_handle: DisplayHandle) -> Result<Self, ApplicationError> {
        let instance_ctx = instance::VulkanInstanceContext::new(display_handle)?;
        let device_ctx = device::DeviceContext::new(&instance_ctx.instance)?;
        let swapchain_handler = swapchain::SwapchainHandler::new(&instance_ctx.entry, &instance_ctx.instance);
        
        info!("[Vulkan] Renderer core successfully initialized.");
        Ok (Renderer {
            swapchain_handler,
            device_ctx,
            instance_ctx,
        })
    }
    
    pub fn handle_presentation(
        &mut self,
        display_handle: DisplayHandle, 
        window_handle: WindowHandle) -> Result<(), PresentationError> {
        unsafe {
            self.swapchain_handler.create_surface(&self.instance_ctx, display_handle, window_handle)
        }
    }
    
    pub fn delete_presentation(&mut self) {
        unsafe {
            self.swapchain_handler.destroy_surface()
        }
    }
}
impl Drop for Renderer {
    fn drop(&mut self) {
        info!("[Renderer] Beginning explicit shutdown sequence.");
        unsafe {
            self.swapchain_handler.destroy_surface();
            self.device_ctx.destroy_self();
            self.instance_ctx.destroy_self();
            info!("[Vulkan] Renderer Drop sequence completed.");
        }        
    }
}