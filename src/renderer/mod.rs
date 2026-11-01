use crate::scene::Scene;
use crate::sky::SkyData;
use crate::water::WaterRenderer;
use crate::renderer::pipelines::Pipelines;
use crate::sim_pipelines::ComputePipelines;
use crate::simulation::Simulation;
use crate::simulation_resources::SimulationResources;
use crate::density_texture::DensityTexturePipeline;
use crate::renderer::resources::GpuSceneResources;
use crate::renderer::ui::AppUI;
pub use crate::renderer::ui::RenderMode;
use crate::constants::MAX_FRAMES_IN_FLIGHT;
use crate::gpu_context::GpuContext;
use egui_winit_vulkano::{Gui, GuiConfig};
use std::sync::Arc;
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage, CopyBufferInfo,
    RenderingAttachmentInfo, RenderingInfo,
};
use vulkano::format::ClearColorValue;
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::sync::GpuFuture;
use vulkano_util::context::VulkanoContext;
use vulkano_util::renderer::VulkanoWindowRenderer;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

pub mod pipelines;
#[path = "../render/frame_resources.rs"]
mod resources;
#[path = "../render/ui.rs"]
mod ui;

pub struct Renderer {
    pub window_renderer: VulkanoWindowRenderer,
    context: Arc<VulkanoContext>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pipelines: Pipelines,
    simulation: Simulation,
    density_texture: DensityTexturePipeline,
    sky_data: SkyData,
    water_renderer: WaterRenderer,

    resources: GpuSceneResources,

    pub gui: Gui,
    pub app_ui: AppUI,


}

impl Renderer {
    pub fn new(window: Window, scene: &Scene, event_loop: &ActiveEventLoop) -> Self {
        let GpuContext { context, window_renderer, command_buffer_allocator, descriptor_set_allocator, depth_format } = GpuContext::new(window);

        let pipelines = Pipelines::new(
            context.clone(),
            window_renderer.swapchain_format(),
            depth_format,
        );

        let resources = GpuSceneResources::new(context.memory_allocator().clone(), scene);
        let simulation_resources = SimulationResources::new(context.memory_allocator().clone(), scene);

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

        let sort_buffer_size = simulation_resources.physics_data.grid_entries.len() as u32;
        let mut gpu_physics = ComputePipelines::new(
            context.device().clone(),
            context.memory_allocator().clone(),
            sort_buffer_size,
        );

        gpu_physics.prepare(
            descriptor_set_allocator.clone(),
            &simulation_resources.physics_data,
            &simulation_resources.sim_params_buffer,
        );
        let mut density_texture = DensityTexturePipeline::new(context.device().clone());
        density_texture.prepare_with_image(
            descriptor_set_allocator.clone(),
            &simulation_resources.physics_data,
            resources.density_view.clone(),
            &simulation_resources.sim_params_buffer,
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
            &simulation_resources.sim_params_buffer,
        );

        Self {
            context: context.clone(),
            window_renderer,
            command_buffer_allocator: command_buffer_allocator.clone(),
            pipelines,
            resources,
            sky_data,
            water_renderer,
            density_texture,
            simulation: Simulation::new(gpu_physics, simulation_resources, command_buffer_allocator.clone(), context.graphics_queue().queue_family_index()),
            gui,
            app_ui: AppUI::new(),
        }
    }
    pub fn step(
        &mut self,
        scene: &mut Scene,
        max_dt: f32,
        previous_future: Box<dyn GpuFuture>,
    ) -> Box<dyn GpuFuture> {
        let settings = self.app_ui.simulation_settings();
        let next_frame = (self.resources.current_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;
        let position = self.simulation.resources.physics_data.position_a.clone();
        let colors = self.simulation.resources.physics_data.colors.clone();
        let render_positions = self.resources.render_data.position_buffers[next_frame].clone();
        let render_colors = self.resources.render_data.color_buffers[next_frame].clone();
        let (future, frame) = self.simulation.step(
            scene, max_dt, &settings,
            self.resources.current_frame_idx, self.context.graphics_queue().clone(), previous_future,
            |scene| self.resources.sync_with_scene(scene),
            |builder| {
                builder.copy_buffer(CopyBufferInfo::buffers(position, render_positions)).unwrap();
                builder.copy_buffer(CopyBufferInfo::buffers(colors, render_colors)).unwrap();
            },
        );
        if let Some(stats) = frame.stats {
            self.app_ui.display_max_speed = stats.max_speed;
            self.app_ui.display_density_iters_used = stats.density_iterations;
            self.app_ui.display_avg_density_error = stats.density_error;
            self.app_ui.display_divergence_iters_used = stats.divergence_iterations;
            self.app_ui.display_avg_divergence_error = stats.divergence_error;
        }
        if let Some(dt) = frame.cfl_dt { self.app_ui.display_cfl_dt = dt; }
        self.app_ui.display_substeps = frame.substeps;
        future
    }

    pub fn update_and_render(&mut self, scene: &mut Scene, safe_dt: f32, fps: u32) {
        let window_size = glam::Vec2::from(self.window_renderer.window_size());
        self.gui.immediate_ui(|gui| {
            let ctx = gui.context();
            self.app_ui.render(&ctx, scene, fps);
            ui::draw_attractor(&ctx, scene, window_size);
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
            self.density_texture.execute(&mut builder);
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
                    self.simulation.resources.physics_data.count,
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
