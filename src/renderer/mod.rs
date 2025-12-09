use log::{info};
use winit::raw_window_handle::{DisplayHandle, WindowHandle};
use winit::window::Window;
use crate::errors::application_error::ApplicationError;
use crate::errors::device_error::DeviceError;
use crate::errors::presentation_error::PresentationError;

mod instance;
mod device;
mod presentation;
mod swapchain_resources;
mod graphics_pipeline;

/// Core Vulkan renderer component, managing the Vulkan instance and lifecycle-dependent resources.
pub struct Renderer {
    instance_ctx: instance::VulkanInstanceContext,
    device_ctx: Option<device::DeviceContext>,
    swapchain_handler: presentation::PresentationContext,
    swapchain_resources: Option<swapchain_resources::SwapchainResources>,
    graphics_pipeline: Option<graphics_pipeline::GraphicsPipeline>,
}

impl Renderer {
    /// Initializes the core, window-independent Vulkan components (Entry, Instance, Surface Loader).
    pub fn new(display_handle: DisplayHandle) -> Result<Self, ApplicationError> {
        let instance_ctx = instance::VulkanInstanceContext::new(display_handle)?;
        let swapchain_handler = presentation::PresentationContext::new(&instance_ctx.entry, &instance_ctx.instance);
        
        info!("[Vulkan] Renderer core successfully initialized.");
        Ok (Renderer {
            swapchain_handler,
            device_ctx: None,
            instance_ctx,
            swapchain_resources: None,
            graphics_pipeline: None,
        })
    }
    pub fn handle_resize(&mut self, window: &Window) -> Result<(), ApplicationError> {
        unsafe {
            self.swapchain_resources
                .as_mut()
                .map(|res| res.destroy_image_views(
                    &self.device_ctx
                        .as_ref()
                        .unwrap()
                        .device
                ));
            self.create_swapchain(window)?;
            self.create_image_views()?;
            
            Ok(())
        }
    }
    pub fn handle_presentation(
        &mut self,
        window: &Window,
        display_handle: DisplayHandle, 
        window_handle: WindowHandle) -> Result<(), ApplicationError> {
        unsafe {
            self.swapchain_handler.create_surface(&self.instance_ctx, display_handle, window_handle)?;
            self.device_ctx = Some(device::DeviceContext::new(&self.instance_ctx.instance, &self.swapchain_handler)?);
            self.create_swapchain(window)?;
            self.create_image_views()?;
            self.graphics_pipeline = Some(graphics_pipeline::GraphicsPipeline::new(
                &self.device_ctx
                    .as_ref()
                    .ok_or(DeviceError::DeviceContextRetrievalFailure("Logical Device".to_string()))
                    .unwrap()
                    .device,
                &self.swapchain_resources
                    .as_ref()
                    .ok_or(PresentationError::SwapchainResourcesNotInitialized)?
                    .swapchain_extent
            )?);
            
            Ok(())
        }
    }
    pub fn delete_presentation(&mut self) {
        unsafe {
            self.swapchain_resources
                .as_mut()
                .map(|res| res.destroy_image_views(
                    &self.device_ctx
                        .as_ref()
                        .unwrap()
                        .device
                ));
            self.swapchain_handler.destroy_swapchain(
                self.swapchain_resources
                    .take()
                    .map(|res| res.swapchain)
            );
            self.graphics_pipeline
                .as_ref()
                .map(|pipeline| pipeline.destroy_pipeline_layout(
                    &self.device_ctx
                        .as_ref()
                        .unwrap()
                        .device
                ));
        }
    }
    unsafe fn create_swapchain(&mut self, window: &Window) -> Result<(), ApplicationError> {
        unsafe {
            let swapchain_context = self.swapchain_handler.create_swapchain(
                &self.instance_ctx.instance,
                &self.device_ctx
                    .as_ref()
                    .ok_or(DeviceError::DeviceContextRetrievalFailure("Logical Device".to_string()))
                    .unwrap()
                    .device,
                self.swapchain_resources.take().map(|res| res.swapchain),
                self.device_ctx
                    .as_ref()
                    .ok_or(DeviceError::DeviceContextRetrievalFailure("Physical Device".to_string()))
                    .unwrap()
                    .physical_device,
                &self.device_ctx
                    .as_ref()
                    .ok_or(DeviceError::DeviceContextRetrievalFailure("Device Queue Families".to_string()))
                    .unwrap()
                    .indices,
                window,
            )?;
            
            self.swapchain_resources = Some(swapchain_resources::SwapchainResources::new(swapchain_context));

            Ok(())
        }
    }
    unsafe fn create_image_views(&mut self) -> Result<(), ApplicationError> {
        self.swapchain_resources
            .as_mut()
            .ok_or(PresentationError::SwapchainResourcesNotInitialized)?
            .create_image_views(
                &self.device_ctx
                    .as_ref()
                    .ok_or(DeviceError::DeviceContextRetrievalFailure("Logical Device".to_string()))
                    .unwrap()
                    .device
            )?;
        
        Ok(())
    }
}
impl Drop for Renderer {
    fn drop(&mut self) {
        info!("[Renderer] Beginning explicit shutdown sequence.");
        unsafe {
            self.delete_presentation();
            self.device_ctx.take();
            self.swapchain_handler.destroy_surface();
            self.instance_ctx.destroy_self();
            info!("[Vulkan] Renderer Drop sequence completed.");
        }        
    }
}