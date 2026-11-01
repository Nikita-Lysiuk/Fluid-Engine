use crate::constants::WINDOW_TITLE;
use std::sync::Arc;
use vulkano::Version;
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::device::DeviceFeatures;
use vulkano::format::Format;
use vulkano::image::ImageUsage;
use vulkano::instance::{InstanceCreateInfo, InstanceExtensions};
use vulkano::swapchain::PresentMode;
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::renderer::VulkanoWindowRenderer;
use vulkano_util::window::WindowDescriptor;
use winit::window::Window;

/// Window and Vulkan objects shared by rendering and simulation.
pub struct GpuContext {
    /// Shared Vulkan context.
    pub context: Arc<VulkanoContext>,
    /// Window and swapchain renderer.
    pub window_renderer: VulkanoWindowRenderer,
    /// Allocator for command buffers.
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    /// Allocator for descriptor sets.
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    /// Depth attachment format.
    pub depth_format: Format,
}

impl GpuContext {
    /// Creates the window renderer, device context and allocators.
    pub fn new(window: Window) -> Self {
        let context = Arc::new(VulkanoContext::new(vulkano_config()));

        let mut window_renderer = VulkanoWindowRenderer::new(
            &context,
            window,
            &WindowDescriptor {
                title: WINDOW_TITLE.into(),
                width: 1280.,
                height: 720.,
                present_mode: PresentMode::Immediate,
                ..Default::default()
            },
            |create_info| {
                create_info.image_usage = ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST;
                create_info.min_image_count = 3;
            },
        );

        let depth_format = Format::D32_SFLOAT;

        window_renderer.add_additional_image_view(
            1,
            depth_format,
            ImageUsage::DEPTH_STENCIL_ATTACHMENT,
        );

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            context.device().clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        Self {
            context,
            window_renderer,
            command_buffer_allocator,
            descriptor_set_allocator,
            depth_format,
        }
    }
}

/// Instance and device setup shared by the window renderer and the offline
/// renderer.
pub fn vulkano_config() -> VulkanoConfig {
    VulkanoConfig {
        instance_create_info: InstanceCreateInfo {
            enabled_extensions: InstanceExtensions {
                ext_debug_utils: true,
                ..InstanceExtensions::default()
            },
            application_name: Some("Fluid Simulation Engine".into()),
            application_version: Version::V1_3,
            ..Default::default()
        },
        device_features: DeviceFeatures {
            sampler_anisotropy: true,
            dynamic_rendering: true,
            synchronization2: true,
            scalar_block_layout: true,
            buffer_device_address: true,
            shader_int64: true,
            large_points: true,
            fill_mode_non_solid: true,
            ..DeviceFeatures::empty()
        },
        ..VulkanoConfig::default()
    }
}
