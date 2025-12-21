use ash::{vk, Device};
use ash::vk::{CommandPoolCreateInfo, FramebufferCreateInfo, ImageAspectFlags, ImageSubresourceRange, ImageViewCreateInfo, ImageViewType, RenderPass, StructureType};
use log::{info};
use crate::errors::device_error::DeviceError;

pub struct SwapchainResources {
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_framebuffers: Vec<vk::Framebuffer>,
}

impl SwapchainResources {
    pub fn new(swapchain_context: (vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D)) -> Self {
        info!("[Swapchain Resources] Swapchain resources initialized.");

        SwapchainResources {
            swapchain_image_views: vec![vk::ImageView::null(); swapchain_context.1.len()],
            swapchain_framebuffers: vec![vk::Framebuffer::null(); swapchain_context.1.len()],
            swapchain: swapchain_context.0,
            swapchain_images: swapchain_context.1,
            swapchain_image_format: swapchain_context.2,
            swapchain_extent: swapchain_context.3,
        }
    }
    pub fn get_viewport(&self) -> vk::Viewport {
        vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.swapchain_extent.width as f32,
            height: self.swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
    pub fn get_scissor(&self) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain_extent,
        }
    }
    pub fn create_image_views(&mut self, device: &Device) -> Result<(), DeviceError> {
        for (i, image_view) in self.swapchain_image_views.iter_mut().enumerate() {
            let create_info = ImageViewCreateInfo {
                s_type: StructureType::IMAGE_VIEW_CREATE_INFO,
                image: self.swapchain_images[i],
                view_type: ImageViewType::TYPE_2D,
                format: self.swapchain_image_format,
                subresource_range: ImageSubresourceRange {
                    aspect_mask: ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..ImageSubresourceRange::default()
                },
                ..ImageViewCreateInfo::default()
            };

            *image_view = unsafe { device.create_image_view(&create_info, None)? };
        }

        info!("[Swapchain Resources] Swapchain image views created.");
        Ok(())
    }
    pub fn create_framebuffers(
        &mut self,
        device: &Device,
        render_pass: RenderPass
    ) -> Result<(), DeviceError> {
        for (i, framebuffer) in self.swapchain_framebuffers.iter_mut().enumerate() {
            let attachments = [self.swapchain_image_views[i]];

            let framebuffer_info = FramebufferCreateInfo {
                s_type: StructureType::FRAMEBUFFER_CREATE_INFO,
                render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                width: self.swapchain_extent.width,
                height: self.swapchain_extent.height,
                layers: 1,
                ..FramebufferCreateInfo::default()
            };

            *framebuffer = unsafe { device.create_framebuffer(&framebuffer_info, None)? };
        }

        info!("[Swapchain Resources] Swapchain framebuffers created.");
        Ok(())
    }
    pub fn destroy_image_views(&mut self, device: &Device) {
        for image_view in self.swapchain_image_views.drain(..) {
            unsafe {
                device.destroy_image_view(image_view, None);
            }
        }
        info!("[Swapchain Resources] Swapchain image views destroyed.");
    }
    pub fn destroy_framebuffers(&mut self, device: &Device) {
        for framebuffer in self.swapchain_framebuffers.drain(..) {
            unsafe {
                device.destroy_framebuffer(framebuffer, None);
            }
        }
        info!("[Swapchain Resources] Swapchain framebuffers destroyed.");
    }
}