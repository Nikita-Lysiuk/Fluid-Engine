use ash::{vk, Device};
use ash::vk::{ImageAspectFlags, ImageSubresourceRange, ImageViewCreateInfo, ImageViewType, StructureType};
use log::info;
use crate::errors::device_error::DeviceError;

pub struct SwapchainResources {
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    
    pub swapchain_image_views: Vec<vk::ImageView>,
}

impl SwapchainResources {
    pub fn new(swapchain_context: (vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D)) -> Self {
        info!("[Vulkan] Swapchain resources initialized.");

        SwapchainResources {
            swapchain_image_views: vec![vk::ImageView::null(); swapchain_context.1.len()],
            swapchain: swapchain_context.0,
            swapchain_images: swapchain_context.1,
            swapchain_image_format: swapchain_context.2,
            swapchain_extent: swapchain_context.3,
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

        info!("[Vulkan] Swapchain image views created.");
        Ok(())
    }
    
    pub fn destroy_image_views(&mut self, device: &Device) {
        for image_view in &self.swapchain_image_views {
            unsafe {
                device.destroy_image_view(*image_view, None);
            }
        }
        info!("[Vulkan] Swapchain image views destroyed.");
    }
}