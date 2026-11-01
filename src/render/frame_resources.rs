use crate::camera::CameraData;
use crate::constants::MAX_FRAMES_IN_FLIGHT;
use crate::render::box_outline::CollisionBoxData;
use crate::render::graphics_pipelines::Pipelines;
use crate::render::particle_renderer::{GpuRenderData, pillar_surface_points};
use crate::scene::Scene;
use std::sync::Arc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};

/// Colour of the pillar (linear RGB).
const PILLAR_COLOR: [f32; 4] = [0.55, 0.52, 0.48, 1.0];

/// Per-frame scene and render resources.
pub struct GpuSceneResources {
    camera_data: CameraData,
    collision_box_data: CollisionBoxData,
    /// Buffers used to draw particles.
    pub render_data: GpuRenderData,
    /// Static points drawn as the pillar.
    pub pillar_points: GpuRenderData,
    pillar_point_count: u32,

    /// Density volume sampled by the fluid renderer.
    pub density_texture: Arc<Image>,
    /// View of the density volume.
    pub density_view: Arc<ImageView>,

    /// Index of the current frame resource slot.
    pub current_frame_idx: usize,
}

impl GpuSceneResources {
    /// Creates the associated resources.
    pub fn new(allocator: Arc<StandardMemoryAllocator>, scene: &Scene) -> Self {
        let render_data = GpuRenderData::new(
            allocator.clone(),
            &scene.initial_positions,
            scene.sim_params.particle_radius,
        );

        let spacing = scene.sim_params.particle_radius;
        let pillar_height = scene.boundary.max.y - scene.boundary.min.y;
        let points = pillar_surface_points(&scene.pillar, pillar_height, spacing);
        let pillar_points =
            GpuRenderData::with_color(allocator.clone(), &points, 2.0 * spacing, PILLAR_COLOR);

        let grid_res = scene.sim_params.grid_res;

        let density_texture = Image::new(
            allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim3d,
                format: Format::R32_UINT,
                extent: [grid_res[0] as u32, grid_res[1] as u32, grid_res[2] as u32],
                usage: ImageUsage::STORAGE | ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("image can be created");

        let density_view =
            ImageView::new_default(density_texture.clone()).expect("image view can be created");

        Self {
            camera_data: CameraData::new(allocator.clone()),
            collision_box_data: CollisionBoxData::new(allocator.clone()),
            render_data,
            pillar_point_count: pillar_points.position_buffers[0].len() as u32,
            pillar_points,
            current_frame_idx: 0,
            density_texture,
            density_view,
        }
    }
    /// Returns the current camera buffer address.
    pub fn camera_addr(&self) -> u64 {
        self.camera_data.uniform_buffer_addr(self.current_frame_idx)
    }
    /// Updates frame resources from the scene.
    pub fn sync_with_scene(&self, scene: &Scene) {
        self.camera_data
            .write_to_buffer(&scene.camera, self.current_frame_idx);
        self.collision_box_data
            .write_to_buffer(&scene.boundary, self.current_frame_idx);
    }

    /// Records the graphics commands for this resource.
    pub fn bind_to_command_buffer<Cb>(
        &self,
        builder: &mut AutoCommandBufferBuilder<Cb>,
        pipelines: &Pipelines,
    ) {
        self.collision_box_data.bind_to_command_buffer(
            builder,
            pipelines,
            self.camera_addr(),
            self.current_frame_idx,
        );
    }

    /// Advances to the next frame resource slot.
    /// Records the pillar's points; nothing if the scene has no pillar.
    pub fn draw_pillar<Cb>(
        &self,
        builder: &mut AutoCommandBufferBuilder<Cb>,
        pipelines: &Pipelines,
        scene: &Scene,
    ) {
        if scene.pillar.enabled {
            // The points never change, so any frame slot holds them.
            self.pillar_points.bind_to_command_buffer(
                builder,
                pipelines,
                self.camera_addr(),
                0,
                self.pillar_point_count,
            );
        }
    }

    pub fn prepare_next_frame(&mut self) {
        self.current_frame_idx = (self.current_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;
    }
}
