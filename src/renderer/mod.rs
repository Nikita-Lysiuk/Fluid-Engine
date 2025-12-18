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
        
        info!("[Renderer] Renderer core successfully initialized.");
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
            self.create_swapchain(window)?;

            if let (Some(res), Some(ctx)) = (self.swapchain_resources.as_mut(), self.device_ctx.as_ref()) {
                let device = &ctx.device;

                res.create_image_views(device)?;

                Ok(())
            } else {
                Err(PresentationError::SwapchainResourcesNotInitialized.into())
            }
        }
    }
    pub fn handle_presentation(
        &mut self,
        window: &Window,
        display_handle: DisplayHandle,
        window_handle: WindowHandle
    ) -> Result<(), ApplicationError> {
        unsafe {
            self.swapchain_handler.create_surface(&self.instance_ctx, display_handle, window_handle)?;

            let device_context = device::DeviceContext::new(&self.instance_ctx.instance, &self.swapchain_handler)?;
            self.device_ctx = Some(device_context);
            
            self.create_swapchain(window)?;
            
            if let (Some(res), Some(ctx)) = (self.swapchain_resources.as_mut(), self.device_ctx.as_ref()) {
                let device = &ctx.device;
                
                res.create_image_views(device)?;
                
                self.graphics_pipeline = Some(graphics_pipeline::GraphicsPipeline::new(
                    device,
                    res.swapchain_image_format
                )?);
            } else {
                return Err(PresentationError::SwapchainResourcesNotInitialized.into());
            }

            Ok(())
        }
    }
    pub fn delete_presentation(&mut self) {
        unsafe {
            let device = &self.device_ctx
                .as_ref()
                .ok_or(DeviceError::DeviceContextRetrievalFailure("Logical Device".to_string()))
                .unwrap()
                .device;

            if let Some(pipeline) = self.graphics_pipeline.take() {
                pipeline.destroy_render_pass(device);
                pipeline.destroy_pipeline_layout(device);
                pipeline.destroy_graphics_pipeline(device);
            }

            if let Some(res) = self.swapchain_resources.as_mut() {
                res.destroy_image_views(device);
            }

            let resources = self.swapchain_resources.take();
            self.swapchain_handler.destroy_swapchain(
                resources.map(|res| res.swapchain)
            );
        }
    }
    unsafe fn create_swapchain(&mut self, window: &Window) -> Result<(), ApplicationError> {
        unsafe {
            if let Some(ctx) = self.device_ctx.as_ref() {
                if let Some(mut old_resources) = self.swapchain_resources.take() {
                    old_resources.destroy_image_views(&ctx.device);

                    let swapchain_context = self.swapchain_handler.create_swapchain(
                        &self.instance_ctx.instance,
                        &ctx.device,
                        Some(old_resources.swapchain),
                        ctx.physical_device,
                        &ctx.indices,
                        window,
                    )?;
                    self.swapchain_resources = Some(swapchain_resources::SwapchainResources::new(swapchain_context));
                } else {
                    let swapchain_context = self.swapchain_handler.create_swapchain(
                        &self.instance_ctx.instance,
                        &ctx.device,
                        None,
                        ctx.physical_device,
                        &ctx.indices,
                        window,
                    )?;
                    self.swapchain_resources = Some(swapchain_resources::SwapchainResources::new(swapchain_context));
                }

                Ok(())
            } else {
                Err(DeviceError::DeviceContextRetrievalFailure("Logical Device".to_string()).into())
            }
        }
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
            info!("[Renderer] Renderer Drop sequence completed.");
        }        
    }
}