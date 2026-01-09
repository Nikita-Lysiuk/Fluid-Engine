mod frame_resources;

use std::sync::Arc;
use log::{debug, error, info, warn};
use vulkano::device::DeviceFeatures;
use vulkano::image::ImageUsage;
use vulkano::instance::debug::{DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessengerCallback, DebugUtilsMessengerCallbackData, DebugUtilsMessengerCreateInfo};
use vulkano::instance::{InstanceCreateInfo, InstanceExtensions};
use vulkano::Version;
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::renderer::VulkanoWindowRenderer;
use vulkano_util::window::WindowDescriptor;
use winit::window::Window;
use crate::utils::constants::WINDOW_TITLE;

pub struct Renderer {
    pub context: Arc<VulkanoContext>,
    pub window_renderer: VulkanoWindowRenderer
}

impl Renderer {
    pub fn new(window: Window) -> Self {
        let callback = unsafe {
            DebugUtilsMessengerCallback::new(|severity: DebugUtilsMessageSeverity, ty: DebugUtilsMessageType, data: DebugUtilsMessengerCallbackData| {
                let type_str = format!("{:?}", ty);
                let description = data.message;

                if severity.intersects(DebugUtilsMessageSeverity::ERROR) {
                    error!("[Vulkan: {}] {}", type_str, description);
                } else if severity.intersects(DebugUtilsMessageSeverity::WARNING) {
                    warn!("[Vulkan: {}] {}", type_str, description);
                } else if severity.intersects(DebugUtilsMessageSeverity::INFO) {
                    info!("[Vulkan: {}] {}", type_str, description);
                } else {
                    debug!("[Vulkan: {}] {}", type_str, description);
                }
            })
        };

        let mut debug_create_info = DebugUtilsMessengerCreateInfo::user_callback(callback);

        debug_create_info.message_severity = DebugUtilsMessageSeverity::ERROR
            | DebugUtilsMessageSeverity::WARNING
            | DebugUtilsMessageSeverity::INFO;
        debug_create_info.message_type = DebugUtilsMessageType::GENERAL
            | DebugUtilsMessageType::VALIDATION
            | DebugUtilsMessageType::PERFORMANCE;

        let mut layers = Vec::new();
        if cfg!(debug_assertions) {
            layers.push("VK_LAYER_KHRONOS_validation".into());
        }

        let config = VulkanoConfig {
            instance_create_info: InstanceCreateInfo {
                enabled_layers: layers,

                enabled_extensions: InstanceExtensions {
                    ext_debug_utils: true,
                    ..InstanceExtensions::default()
                },
                application_name: Some("Fluid Simulation Engine".into()),
                application_version: Version::V1_3,
                ..Default::default()
            },
            debug_create_info: Some(debug_create_info),
            device_features: DeviceFeatures {
                geometry_shader: true,
                sampler_anisotropy: true,
                ..DeviceFeatures::empty()
            },
            ..VulkanoConfig::default()
        };

        let context = VulkanoContext::new(config);

        let window_renderer = VulkanoWindowRenderer::new(
            &context,
            window,
            &WindowDescriptor {
                title: WINDOW_TITLE.into(),
                width: 1280.,
                height: 720.,
                ..Default::default()
            },
            |create_info| {
                create_info.image_usage = ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST;
            }
        );

        Self {
            context: Arc::new(context),
            window_renderer
        }
    }
}