use std::sync::Arc;
use log::{debug, error, info, warn};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, RenderingAttachmentInfo, RenderingInfo};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::descriptor_set::allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo};
use vulkano::device::DeviceFeatures;
use vulkano::format::Format;
use vulkano::image::ImageUsage;
use vulkano::instance::debug::{DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessengerCallback, DebugUtilsMessengerCallbackData, DebugUtilsMessengerCreateInfo};
use vulkano::instance::{InstanceCreateInfo, InstanceExtensions};
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::swapchain::PresentMode;
use vulkano::sync::GpuFuture;
use vulkano::Version;
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::renderer::VulkanoWindowRenderer;
use vulkano_util::window::WindowDescriptor;
use winit::window::Window;
use crate::core::scene::Scene;
use crate::entities::sky::SkyData;
use crate::renderer::pipelines::{ComputePipelines, ComputeStep, Pipelines};
use crate::renderer::resources::GpuSceneResources;
use crate::utils::constants::{MAX_FRAMES_IN_FLIGHT, WINDOW_TITLE};

pub mod pipelines;
mod resources;

pub struct Renderer {
    pub window_renderer: VulkanoWindowRenderer,
    context: Arc<VulkanoContext>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pipelines: Pipelines,
    physics_steps: ComputePipelines,
    sky_data: SkyData,

    resources: GpuSceneResources,
 
}

impl Renderer {
    pub fn new(window: Window, scene: &Scene) -> Self {
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
                fill_mode_non_solid: true,
                ..DeviceFeatures::empty()
            },
            ..VulkanoConfig::default()
        };

        let context = Arc::new(VulkanoContext::new(config));

        let mut window_renderer = VulkanoWindowRenderer::new(
            &context,
            window,
            &WindowDescriptor {
                title: WINDOW_TITLE.into(),
                width: 1280.,
                height: 720.,
                present_mode: PresentMode::Mailbox,
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

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            StandardCommandBufferAllocatorCreateInfo::default()
        ));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            context.device().clone(),
            StandardDescriptorSetAllocatorCreateInfo::default()
        ));

        let pipelines = Pipelines::new(
            context.clone(),
            window_renderer.swapchain_format(),
            depth_format
        );

        let resources = GpuSceneResources::new(context.memory_allocator().clone(), scene);

        let sky_data = SkyData::new(
            500.0,
            64,
            64,
            context.memory_allocator().clone(),
            descriptor_set_allocator.clone(),
            command_buffer_allocator.clone(),
            pipelines.sky_layout.clone(),
            context.graphics_queue().clone(),
            "assets/hdri/citrus_orchard_road_puresky_4k.exr"
        );

        let gpu_physics = ComputePipelines::new(context.device().clone());

        Self {
            context,
            window_renderer,
            command_buffer_allocator,
            descriptor_set_allocator,
            pipelines,
            resources,
            sky_data,
            physics_steps: gpu_physics,
        }
    }
    pub fn step(&mut self, scene: &Scene, max_dt: f32) -> Box<dyn GpuFuture> {
        self.resources.sync_with_scene(scene);

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.graphics_queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        self.physics_steps.neighbor_search.execute(
            &mut builder,
            self.descriptor_set_allocator.clone(),
            &self.resources.physics_data,
            &self.resources.sim_params_buffer,
        );

        self.physics_steps.density_alpha.execute(
            &mut builder,
            self.descriptor_set_allocator.clone(),
            &self.resources.physics_data,
            &self.resources.sim_params_buffer,
        );

        let mut step = 0.0;
        while step < max_dt {
            self.physics_steps.viscosity.execute(
                &mut builder,
                self.descriptor_set_allocator.clone(),
                &self.resources.physics_data,
                &self.resources.sim_params_buffer,
            );

            self.physics_steps.density_source_term.execute(
                &mut builder,
                self.descriptor_set_allocator.clone(),
                &self.resources.physics_data,
                &self.resources.sim_params_buffer,
            );

            for _ in 0..scene.sim_params.density_solver_iterations {
                self.physics_steps.pressure_force.execute(
                    &mut builder,
                    self.descriptor_set_allocator.clone(),
                    &self.resources.physics_data,
                    &self.resources.sim_params_buffer,
                );

                self.physics_steps.pressure_update.execute(
                    &mut builder,
                    self.descriptor_set_allocator.clone(),
                    &self.resources.physics_data,
                    &self.resources.sim_params_buffer,
                );
            }

            self.physics_steps.pressure_integration.execute(
                &mut builder,
                self.descriptor_set_allocator.clone(),
                &self.resources.physics_data,
                &self.resources.sim_params_buffer,
            );

            step += scene.sim_params.dt;

            self.physics_steps.density_alpha.execute(
                &mut builder,
                self.descriptor_set_allocator.clone(),
                &self.resources.physics_data,
                &self.resources.sim_params_buffer,
            );

            self.physics_steps.divergence_source_term.execute(
                &mut builder,
                self.descriptor_set_allocator.clone(),
                &self.resources.physics_data,
                &self.resources.sim_params_buffer,
            );

            for _ in 0..scene.sim_params.divergence_solver_iterations {
                self.physics_steps.pressure_force.execute(
                    &mut builder,
                    self.descriptor_set_allocator.clone(),
                    &self.resources.physics_data,
                    &self.resources.sim_params_buffer,
                );

                self.physics_steps.pressure_update.execute(
                    &mut builder,
                    self.descriptor_set_allocator.clone(),
                    &self.resources.physics_data,
                    &self.resources.sim_params_buffer,
                );
            }

            self.physics_steps.divergence_integration.execute(
                &mut builder,
                self.descriptor_set_allocator.clone(),
                &self.resources.physics_data,
                &self.resources.sim_params_buffer,
            );
        }
        
        let next_frame = (self.resources.current_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;


        builder.copy_buffer(CopyBufferInfo::buffers(
            self.resources.physics_data.position_a.clone(),
            self.resources.render_data.position_buffers[next_frame].clone()
        )).unwrap();

        builder.copy_buffer(CopyBufferInfo::buffers(
            self.resources.physics_data.colors.clone(),
            self.resources.render_data.color_buffers[next_frame].clone()
        )).unwrap();

        let command_buffer = builder.build().unwrap();

        vulkano::sync::now(self.context.device().clone())
            .then_execute(self.context.graphics_queue().clone(), command_buffer)
            .unwrap()
            .boxed()
    }
    pub fn render(&mut self) {
        let acquire_future = self.window_renderer
            .acquire(None, |_img_view|{})
            .map_err(|e| panic!("[Renderer] Failed to acquire next image for rendering: {:?}", e))
            .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.graphics_queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit
        ).map_err(|e| panic!("[Renderer] Failed to create command buffer builder: {:?}", e))
            .unwrap();


        let extent = self.window_renderer.window_size();

        let viewport = Viewport {
            offset: [0.0, extent[1]],
            extent: [extent[0], -extent[1]],
            depth_range: 0.0..=1.0,
        };

        let scissor = Scissor {
            offset: [0, 0],
            extent: [extent[0] as u32, extent[1] as u32],
        };

        builder
            .begin_rendering(RenderingInfo {
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
            .set_scissor(0, [scissor.clone()].into_iter().collect()).map_err(|e| panic!("[Renderer] Failed to set scissor: {:?}", e)).unwrap();

        self.sky_data.bind_to_command_buffer(&mut builder, &self.pipelines, self.resources.camera_addr());
        self.resources.bind_to_command_buffer(&mut builder, &self.pipelines);

        builder.end_rendering().map_err(|e| panic!("[Renderer] Failed to end rendering: {:?}", e)).unwrap();
        let render_command_buffer = builder.build().unwrap();

        let combined_future = acquire_future
            .then_execute(self.context.graphics_queue().clone(), render_command_buffer)
            .unwrap();

        self.window_renderer.present(combined_future.boxed(), false);
        self.resources.prepare_next_frame();
    }
}