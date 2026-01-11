use std::sync::Arc;
use log::{debug, error, info, warn};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, RenderingAttachmentInfo, RenderingInfo};
use vulkano::device::DeviceFeatures;
use vulkano::format::Format;
use vulkano::image::ImageUsage;
use vulkano::instance::debug::{DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessengerCallback, DebugUtilsMessengerCallbackData, DebugUtilsMessengerCreateInfo};
use vulkano::instance::{InstanceCreateInfo, InstanceExtensions};
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport};
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::PipelineLayout;
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;
use vulkano::swapchain::PresentMode;
use vulkano::sync::GpuFuture;
use vulkano::Version;
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::renderer::VulkanoWindowRenderer;
use vulkano_util::window::WindowDescriptor;
use winit::window::Window;
use crate::core::scene::Scene;
use crate::utils::constants::WINDOW_TITLE;

mod resources;
mod pipelines;

pub struct Renderer {
    pub context: Arc<VulkanoContext>,
    pub window_renderer: VulkanoWindowRenderer,
    pub resources: resources::FrameResources,
    pub point_pipeline: pipelines::point_pipeline::PointPipeline,
    pub common_layout: Arc<PipelineLayout>
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
                sampler_anisotropy: true,
                dynamic_rendering: true,
                synchronization2: true,
                scalar_block_layout: true,
                buffer_device_address: true,
                shader_int64: true,
                large_points: true,
                ..DeviceFeatures::empty()
            },
            ..VulkanoConfig::default()
        };

        let context = VulkanoContext::new(config);

        let mut window_renderer = VulkanoWindowRenderer::new(
            &context,
            window,
            &WindowDescriptor {
                title: WINDOW_TITLE.into(),
                width: 1280.,
                height: 720.,
                present_mode: PresentMode::Fifo,
                ..Default::default()
            },
            |create_info| {
                create_info.image_usage = ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST;
                create_info.min_image_count = 3;
            }
        );

        let depth_format = Format::D32_SFLOAT;

        window_renderer.add_additional_image_view(
            1,
            depth_format,
            ImageUsage::DEPTH_STENCIL_ATTACHMENT
        );

        let resources = resources::FrameResources::new(
            context.device().clone(),
            context.memory_allocator().clone(),
        );

        let push_constant_range = PushConstantRange {
            stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            offset: 0,
            size: 16,
        };

        let common_layout = PipelineLayout::new(
            context.device().clone(),
            PipelineLayoutCreateInfo {
                set_layouts: Vec::new(),
                push_constant_ranges: vec![push_constant_range],
                ..PipelineLayoutCreateInfo::default()
            }
        ).expect("[Renderer] Failed to create common pipeline layout.");

        let point_pipeline = pipelines::point_pipeline::PointPipeline::new(
            context.device().clone(),
            common_layout.clone(),
            window_renderer.swapchain_format(),
            depth_format,
        );

        Self {
            context: Arc::new(context),
            window_renderer,
            resources,
            point_pipeline,
            common_layout,
        }
    }

    pub fn render(&mut self, scene: &Scene) {

        let camera_data = scene.get_camera_data();
        self.resources.current_ub()
            .write()
            .map_err(|e| panic!("[Renderer] Failed to write uniform buffer: {:?}", e))
            .unwrap()
            .clone_from(&camera_data);

        let particle_data = scene.get_particle_data();
        {
            let mut write_lock = self.resources.current_pb()
                .write()
                .map_err(|e| panic!("[Renderer] Failed to write particle buffer: {:?}", e))
                .unwrap();

            let len = particle_data.len();
            write_lock[..len].copy_from_slice(&particle_data);
        }

        let future = self.window_renderer
            .acquire(None, |_img_view|{})
            .map_err(|e| panic!("[Renderer] Failed to acquire next image for rendering: {:?}", e))
            .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.resources.command_buffer_allocator.clone(),
            self.context.graphics_queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit
        ).map_err(|e| panic!("[Renderer] Failed to create command buffer builder: {:?}", e))
        .unwrap();

        let extent = self.window_renderer.window_size();

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [extent[0], extent[1]],
            depth_range: 0.0..=1.0,
        };

        let scissor = Scissor {
            offset: [0, 0],
            extent: [extent[0] as u32, extent[1] as u32],
        };

        unsafe { builder.begin_rendering(RenderingInfo {
            color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                    ..RenderingAttachmentInfo::image_view(self.window_renderer.swapchain_image_view().clone())
                })],
            depth_attachment: Some(RenderingAttachmentInfo {
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
                clear_value: Some(1f32.into()),
                ..RenderingAttachmentInfo::image_view(self.window_renderer.get_additional_image_view(1).clone())
            }),
            ..RenderingInfo::default()
        }).map_err(|e| panic!("[Renderer] Failed to create command buffer builder: {:?}", e)).unwrap()
            .set_viewport(0, [viewport.clone()].into_iter().collect()).map_err(|e| panic!("[Renderer] Failed to set viewport: {:?}", e)).unwrap()
            .set_scissor(0, [scissor.clone()].into_iter().collect()).map_err(|e| panic!("[Renderer] Failed to set scissor: {:?}", e)).unwrap()
            .bind_pipeline_graphics(self.point_pipeline.inner.clone()).map_err(|e| panic!("[Renderer] Failed to bind point pipeline: {:?}", e)).unwrap()
            .push_constants(
                self.common_layout.clone(),
                0,
                [
                    self.resources.current_ub().device_address().map_err(|e| panic!("[Renderer] Failed to get uniform_buffer: {:?}", e)).unwrap().get(),
                    self.resources.current_pb().device_address().map_err(|e| panic!("[Renderer] Failed to get particle_buffer: {:?}", e)).unwrap().get(),
                ]
            ).map_err(|e| panic!("[Renderer] Failed to bind buffers: {:?}", e)).unwrap()
            .draw(scene.vertices.len() as u32, 1, 0, 0).map_err(|e| panic!("[Renderer] Failed to draw particles: {:?}", e)).unwrap()
            .end_rendering().map_err(|e| panic!("[Renderer] Failed to end rendering: {:?}", e)).unwrap();
        }

        let command_buffer = builder.build().map_err(|e| panic!("[Renderer] Failed to build command buffer builder: {:?}", e)).unwrap();

        let joined_future = future
            .then_execute(self.context.graphics_queue().clone(), command_buffer)
            .map_err(|e| panic!("[Renderer] Failed to execute command buffer: {:?}", e))
            .unwrap();

        self.window_renderer.present(joined_future.boxed(), false);
        self.resources.next_frame();
    }
}