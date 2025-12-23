use crate::errors::sync_error::SyncError;
use std::time::Duration;
use log::{info};
use winit::raw_window_handle::{DisplayHandle, WindowHandle};
use winit::window::Window;
use crate::errors::application_error::ApplicationError;
use crate::errors::device_error::DeviceError;
use crate::errors::graphics_pipeline_error::GraphicsPipelineError;
use crate::errors::presentation_error::PresentationError;

mod instance;
mod device;
mod presentation;
mod swapchain_resources;
mod graphics_pipeline;
mod command;
mod sync_objects;

/// Core Vulkan renderer component, managing the Vulkan instance and lifecycle-dependent resources.
pub struct Renderer {
    instance_ctx: instance::VulkanInstanceContext,
    device_ctx: Option<device::DeviceContext>,
    presentation_handler: presentation::PresentationContext,
    swapchain_resources: Option<swapchain_resources::SwapchainResources>,
    graphics_pipeline: Option<graphics_pipeline::GraphicsPipeline>,
    command_ctx: command::CommandContext,
    sync_objects: Option<sync_objects::SyncObjects>,
}

impl Renderer {
    pub fn new(display_handle: DisplayHandle) -> Result<Self, ApplicationError> {
        let instance_ctx = instance::VulkanInstanceContext::new(display_handle)?;
        let presentation_handler = presentation::PresentationContext::new(&instance_ctx.entry, &instance_ctx.instance);
        let command_ctx = command::CommandContext::new();

        info!("[Renderer] Renderer core successfully initialized.");
        Ok (Renderer {
            presentation_handler,
            device_ctx: None,
            instance_ctx,
            swapchain_resources: None,
            graphics_pipeline: None,
            command_ctx,
            sync_objects: None,
        })
    }
    pub fn update(&self) -> Result<(), ApplicationError> {
        unsafe {
            if let (
                Some(ctx),
                Some(res),
                Some(gp),
                Some(sync)
            ) = (
                self.device_ctx.as_ref(),
                self.swapchain_resources.as_ref(),
                self.graphics_pipeline.as_ref(),
                self.sync_objects.as_ref(),
            ) {
                let device = &ctx.device;

                device.wait_for_fences(
                    &[sync.in_flight_fence],
                    true,
                    u64::MAX
                ).map_err(|e| SyncError::FailedToWaitForFence(e))?;

                device.reset_fences(&[sync.in_flight_fence])
                    .map_err(|e| SyncError::FailedToResetFence(e))?;

                let (image_index, _is_suboptimal) = self.presentation_handler.acquire_next_image(
                    res.swapchain,
                    u64::MAX,
                    sync
                )?;

                self.command_ctx.reset_command_buffer(
                    device,
                    image_index as usize
                )?;

                self.command_ctx.record_command_buffer(
                    device,
                    image_index as usize
                )?;

                self.command_ctx.recording_render_pass(
                    device,
                    &gp.render_pass,
                    res,
                    image_index as usize
                )?;

                self.command_ctx.record_graphics_commands(
                    device,
                    res,
                    gp,
                    image_index as usize
                )?;

                self.command_ctx.end_recording(
                    device,
                    image_index as usize
                )?;

                self.command_ctx.submit_command_buffer(
                    device,
                    ctx.graphics_queue,
                    sync
                )?;

                self.presentation_handler.present(
                    sync,
                    res.swapchain,
                    ctx.present_queue,
                    image_index
                )?;
            }

            Ok(())
        }
    }
    pub fn handle_resize(&mut self, window: &Window) -> Result<(), ApplicationError> {
        unsafe {
            self.create_swapchain(window)?;


            if let (
                Some(res),
                Some(ctx),
                Some(pipeline)
            ) = (
                self.swapchain_resources.as_mut(),
                self.device_ctx.as_ref(),
                self.graphics_pipeline.as_ref()
            ) {
                let device = &ctx.device;

                res.create_image_views(device)?;
                res.create_framebuffers(
                    device,
                    pipeline.render_pass
                )?;

                let image_count = res.swapchain_images.len();
                self.command_ctx.reallocate_command_buffers(device, image_count)?;

                Ok(())
            } else {
                if self.swapchain_resources.is_none() { Err(PresentationError::SwapchainResourcesNotInitialized.into()) }
                else if self.graphics_pipeline.is_none() { Err(GraphicsPipelineError::GraphicsPipelineNotInitialized.into()) }
                else { Err(DeviceError::DeviceContextRetrievalFailure("Logical Device".to_string()).into()) }
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
            self.presentation_handler.create_surface(&self.instance_ctx, display_handle, window_handle)?;

            let device_context = device::DeviceContext::new(&self.instance_ctx.instance, &self.presentation_handler)?;
            self.device_ctx = Some(device_context);
            
            self.create_swapchain(window)?;
            
            if let (Some(res), Some(ctx)) = (self.swapchain_resources.as_mut(), self.device_ctx.as_mut()) {
                let device = &ctx.device;
                
                res.create_image_views(device)?;
                
                self.graphics_pipeline = Some(graphics_pipeline::GraphicsPipeline::new(
                    device,
                    res.swapchain_image_format
                )?);

                res.create_framebuffers(
                    device,
                    self.graphics_pipeline.as_ref().unwrap().render_pass
                )?;

                self.command_ctx.create_command_pool(
                    device,
                    &ctx.indices
                )?;

                self.command_ctx.create_command_buffer(
                    device
                )?;

                self.sync_objects = Some(sync_objects::SyncObjects::new(
                    device
                )?);
            } else {
                return Err(PresentationError::SwapchainResourcesNotInitialized.into());
            }

            Ok(())
        }
    }
    pub fn delete_presentation(&mut self) {
        unsafe {
            if let Some(ctx) = self.device_ctx.as_ref() {
                let device = &ctx.device;

                device.device_wait_idle()
                    .expect("[Renderer] Failed to wait for device idle during presentation deletion.");

                if let Some(sync) = self.sync_objects.take() {
                    sync.destroy(device);
                }

                self.command_ctx.destroy_command_pool(device);

                if let Some(res) = self.swapchain_resources.as_mut() {
                    res.destroy_framebuffers(device);
                }

                if let Some(pipeline) = self.graphics_pipeline.take() {
                    pipeline.destroy_graphics_pipeline(device);
                    pipeline.destroy_pipeline_layout(device);
                    pipeline.destroy_render_pass(device);
                }

                if let Some(res) = self.swapchain_resources.as_mut() {
                    res.destroy_image_views(device);
                }

                let resources = self.swapchain_resources.take();
                self.presentation_handler.destroy_swapchain(
                    resources.map(|res| res.swapchain)
                );
            } else {
                info!("[Renderer] No Device Context found during presentation deletion. Skipping device-dependent resource cleanup.");
            }
        }
    }
    unsafe fn create_swapchain(&mut self, window: &Window) -> Result<(), ApplicationError> {
        let inner_size = window.inner_size();
        if inner_size.width == 0 || inner_size.height == 0 {
            info!("[Renderer] Window is minimized, skipping swapchain recreation.");
            return Ok(());
        }
        unsafe {
            if let Some(ctx) = self.device_ctx.as_ref() {
                if let Some(mut old_resources) = self.swapchain_resources.take() {
                    old_resources.destroy_image_views(&ctx.device);
                    old_resources.destroy_framebuffers(&ctx.device);

                    let swapchain_context = self.presentation_handler.create_swapchain(
                        &self.instance_ctx.instance,
                        &ctx.device,
                        Some(old_resources.swapchain),
                        ctx.physical_device,
                        &ctx.indices,
                        window,
                    )?;
                    self.swapchain_resources = Some(swapchain_resources::SwapchainResources::new(swapchain_context));
                } else {
                    let swapchain_context = self.presentation_handler.create_swapchain(
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
            self.presentation_handler.destroy_surface();
            self.instance_ctx.destroy_self();
            info!("[Renderer] Renderer Drop sequence completed.");
        }        
    }
}