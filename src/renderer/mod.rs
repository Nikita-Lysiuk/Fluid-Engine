use crate::core::scene::Scene;
use crate::entities::particle::stats_layout;
use crate::entities::sky::SkyData;
use crate::entities::water::WaterRenderer;
use crate::renderer::pipelines::{ComputePipelines, ComputeStep, Pipelines, SolverSettings, SortAlgorithm};
use crate::renderer::resources::GpuSceneResources;
use crate::renderer::ui::AppUI;
pub use crate::renderer::ui::RenderMode;
use crate::utils::constants::{MAX_FRAMES_IN_FLIGHT, WINDOW_TITLE};
use egui_winit_vulkano::{Gui, GuiConfig};
use std::sync::Arc;
use vulkano::Version;
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage, CopyBufferInfo,
    PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::device::DeviceFeatures;
use vulkano::format::{ClearColorValue, Format};
use vulkano::image::ImageUsage;
use vulkano::instance::{InstanceCreateInfo, InstanceExtensions};
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::swapchain::PresentMode;
use vulkano::sync::GpuFuture;
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::renderer::VulkanoWindowRenderer;
use vulkano_util::window::WindowDescriptor;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

pub mod pipelines;
mod resources;
mod ui;

pub struct Renderer {
    pub window_renderer: VulkanoWindowRenderer,
    context: Arc<VulkanoContext>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pipelines: Pipelines,
    physics_steps: ComputePipelines,
    sky_data: SkyData,
    water_renderer: WaterRenderer,

    resources: GpuSceneResources,

    pub gui: Gui,
    pub app_ui: AppUI,

    /// The neighbour grid and densities must be built once before the first substep.
    needs_init: bool,
    substep_cache: Option<((SolverSettings, SortAlgorithm), Arc<PrimaryAutoCommandBuffer>)>,
}

impl Renderer {
    pub fn new(window: Window, scene: &Scene, event_loop: &ActiveEventLoop) -> Self {
        let config = VulkanoConfig {
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
        };

        let context = Arc::new(VulkanoContext::new(config));

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

        let pipelines = Pipelines::new(
            context.clone(),
            window_renderer.swapchain_format(),
            depth_format,
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
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/hdri/citrus_orchard_road_puresky_4k.exr"
            ),
        );

        let sort_buffer_size = resources.physics_data.grid_entries.len() as u32;
        let mut gpu_physics = ComputePipelines::new(
            context.device().clone(),
            context.memory_allocator().clone(),
            sort_buffer_size,
        );

        gpu_physics.prepare(
            descriptor_set_allocator.clone(),
            &resources.physics_data,
            &resources.sim_params_buffer,
        );
        gpu_physics.density_texture.prepare_with_image(
            descriptor_set_allocator.clone(),
            &resources.physics_data,
            resources.density_view.clone(),
            &resources.sim_params_buffer,
        );

        let gui = Gui::new(
            event_loop,
            window_renderer.surface(),
            context.graphics_queue().clone(),
            window_renderer.swapchain_format(),
            GuiConfig {
                is_overlay: true,
                ..Default::default()
            },
        );

        let water_renderer = WaterRenderer::new(
            context.memory_allocator().clone(),
            descriptor_set_allocator.clone(),
            pipelines.water_renderer_pipeline.inner.layout().clone(),
            resources.density_view.clone(),
            sky_data.texture_view.clone(),
            &resources.sim_params_buffer,
        );

        Self {
            context,
            window_renderer,
            command_buffer_allocator,
            pipelines,
            resources,
            sky_data,
            water_renderer,
            physics_steps: gpu_physics,
            gui,
            app_ui: AppUI::new(),
            needs_init: true,
            substep_cache: None,
        }
    }
    pub fn step(
        &mut self,
        scene: &mut Scene,
        max_dt: f32,
        previous_future: Box<dyn GpuFuture>,
    ) -> Box<dyn GpuFuture> {
        if let Some(stats) = self.resources.oldest_stats() {
            let solver_stats = |range: std::ops::Range<u64>| {
                let state = &stats[range.start as usize..range.end as usize];
                (
                    state[stats_layout::ITERATIONS],
                    f32::from_bits(state[stats_layout::AVG_ERROR]),
                )
            };
            let max_speed = f32::from_bits(stats[stats_layout::MAX_SPEED]);
            self.app_ui.display_max_speed = max_speed;
            (
                self.app_ui.display_density_iters_used,
                self.app_ui.display_avg_density_error,
            ) = solver_stats(stats_layout::DENSITY_SOLVER);
            (
                self.app_ui.display_divergence_iters_used,
                self.app_ui.display_avg_divergence_error,
            ) = solver_stats(stats_layout::DIVERGENCE_SOLVER);

            if self.app_ui.use_cfl {
                let p = &scene.sim_params;
                let diameter = 2.0 * p.particle_radius;
                let gravity = glam::Vec4::from(p.gravity).length();
                let speed = (max_speed + gravity * p.dt).max(1e-9);
                let cfl_dt = (self.app_ui.cfl_factor * 0.4 * diameter / speed)
                    .clamp(self.app_ui.cfl_min_dt, self.app_ui.cfl_max_dt);
                scene.sim_params.dt = cfl_dt;
                self.app_ui.display_cfl_dt = cfl_dt;
            }
        }
        let solver = self.app_ui.solver_settings(&scene.sim_params);

        scene.sim_params.piston_velocity = scene.boundary.piston_velocity();
        scene.sim_params.attractor = scene.attractor.to_gpu(&scene.camera);
        scene.sim_params.attractor_radius = scene.attractor.radius;
        self.resources.sync_with_scene(scene);

        let queue = self.context.graphics_queue().clone();

        let mut builder = self.primary_builder(CommandBufferUsage::OneTimeSubmit);
        builder
            .update_buffer(self.resources.sim_params_buffer.clone(), Box::new(scene.sim_params))
            .unwrap();
        if self.needs_init {
            self.physics_steps.record_init(&mut builder);
            self.needs_init = false;
        }
        let mut future = previous_future
            .then_execute(queue.clone(), builder.build().unwrap())
            .unwrap()
            .boxed();

        let wanted = (max_dt / scene.sim_params.dt).ceil() as u32;
        // A cap of 0 pauses the simulation (used to measure rendering alone).
        let substeps = wanted.max(1).min(self.app_ui.max_substeps_per_frame);
        self.app_ui.display_substeps = substeps;
        // Animate the box in simulated time so it stays in step with the fluid.
        scene.boundary.update(substeps as f32 * scene.sim_params.dt);

        let substep = self.substep_command_buffer(solver);
        for _ in 0..substeps {
            future = future.then_execute(queue.clone(), substep.clone()).unwrap().boxed();
        }

        let mut builder = self.primary_builder(CommandBufferUsage::OneTimeSubmit);
        self.physics_steps.stats.execute(&mut builder);
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.resources.physics_data.stats_buffer.clone(),
                self.resources.stats_readback[self.resources.current_frame_idx].clone(),
            ))
            .unwrap();

        let next_frame = (self.resources.current_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.resources.physics_data.position_a.clone(),
                self.resources.render_data.position_buffers[next_frame].clone(),
            ))
            .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.resources.physics_data.colors.clone(),
                self.resources.render_data.color_buffers[next_frame].clone(),
            ))
            .unwrap();

        let command_buffer = builder.build().unwrap();

        future
            .then_execute(queue, command_buffer)
            .unwrap()
            .then_signal_semaphore()
            .boxed()
    }

    fn primary_builder(&self, usage: CommandBufferUsage) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.graphics_queue().queue_family_index(),
            usage,
        )
        .unwrap()
    }

    /// One substep, recorded once and resubmitted every substep of every
    /// frame. Everything that changes per frame (dt, box, attractor) lives in
    /// the uniform buffer; only the solver settings and the sort algorithm are
    /// baked into the commands, so the buffer is re-recorded when they change.
    fn substep_command_buffer(&mut self, solver: SolverSettings) -> Arc<PrimaryAutoCommandBuffer> {
        let key = (solver, self.app_ui.sort_algorithm);
        if let Some((cached_key, command_buffer)) = &self.substep_cache {
            if *cached_key == key {
                return command_buffer.clone();
            }
        }
        let _s = tracy_client::span!("record_substep");
        self.physics_steps.neighbor_search.sort_algorithm = self.app_ui.sort_algorithm;
        // SimultaneousUse: the same buffer is queued several times per frame.
        let mut builder = self.primary_builder(CommandBufferUsage::SimultaneousUse);
        self.physics_steps.record_substep(&mut builder, &solver);
        let command_buffer = builder.build().unwrap();
        self.substep_cache = Some((key, command_buffer.clone()));
        command_buffer
    }

    pub fn update_and_render(&mut self, scene: &mut Scene, safe_dt: f32, fps: u32) {
        self.gui.immediate_ui(|gui| {
            let ctx = gui.context();
            self.app_ui.render(&ctx, scene, fps);
        });

        let acquire_future = self
            .window_renderer
            .acquire(None, |_| {})
            .map_err(|e| panic!("[Renderer] Failed to acquire swapchain image: {:?}", e))
            .unwrap();

        let physics_semaphore = self.step(scene, safe_dt, acquire_future.boxed());

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.graphics_queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| {
            panic!(
                "[Renderer] Failed to create command buffer builder: {:?}",
                e
            )
        })
        .unwrap();

        if self.app_ui.render_mode == RenderMode::Raymarching {
            let mut clear_info = ClearColorImageInfo::image(self.resources.density_texture.clone());
            clear_info.clear_value = ClearColorValue::Uint([0; 4]);
            builder.clear_color_image(clear_info).unwrap();
            self.physics_steps.density_texture.execute(&mut builder);
        }

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
                    ..RenderingAttachmentInfo::image_view(
                        self.window_renderer.swapchain_image_view().clone(),
                    )
                })],
                depth_attachment: Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some(1f32.into()),
                    ..RenderingAttachmentInfo::image_view(
                        self.window_renderer.get_additional_image_view(1).clone(),
                    )
                }),
                ..RenderingInfo::default()
            })
            .map_err(|e| {
                panic!(
                    "[Renderer] Failed to create command buffer builder: {:?}",
                    e
                )
            })
            .unwrap()
            .set_viewport(0, [viewport.clone()].into_iter().collect())
            .map_err(|e| panic!("[Renderer] Failed to set viewport: {:?}", e))
            .unwrap()
            .set_scissor(0, [scissor.clone()].into_iter().collect())
            .map_err(|e| panic!("[Renderer] Failed to set scissor: {:?}", e))
            .unwrap();

        self.sky_data.bind_to_command_buffer(
            &mut builder,
            &self.pipelines,
            self.resources.camera_addr(),
        );
        match self.app_ui.render_mode {
            RenderMode::Raymarching => {
                self.water_renderer.bind_to_command_buffer(
                    &mut builder,
                    self.pipelines.water_renderer_pipeline.inner.clone(),
                    self.resources.camera_addr(),
                    scene.sim_params.box_min,
                    scene.sim_params.box_max,
                );
            }
            RenderMode::Particles => {
                self.resources.render_data.bind_to_command_buffer(
                    &mut builder,
                    &self.pipelines,
                    self.resources.camera_addr(),
                    self.resources.current_frame_idx,
                    self.resources.physics_data.count,
                );
            }
        }
        self.resources
            .bind_to_command_buffer(&mut builder, &self.pipelines);

        builder
            .end_rendering()
            .map_err(|e| panic!("[Renderer] Failed to end rendering: {:?}", e))
            .unwrap();
        let render_command_buffer = builder.build().unwrap();

        let render_future = physics_semaphore
            .then_execute(
                self.window_renderer.graphics_queue().clone(),
                render_command_buffer,
            )
            .unwrap();

        let gui_future = self
            .gui
            .draw_on_image(render_future, self.window_renderer.swapchain_image_view());

        // TODO: `false` would let the CPU prepare the next frame while the GPU
        // still works on this one (~5 ms per frame), but vulkano 0.35 then
        // panics with "Buffer/Image was not locked for GPU write".
        self.window_renderer.present(gui_future, true);
        self.resources.prepare_next_frame();
    }
}
