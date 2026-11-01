use crate::camera::CameraData;
use crate::constants::MAX_FRAMES_IN_FLIGHT;
use crate::render::box_outline::CollisionBoxData;
use crate::render::graphics_pipelines::Pipelines;
use crate::render::particle_renderer::GpuRenderData;
use crate::scene::Scene;
use std::sync::Arc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};

/// Per-frame scene and render resources.
pub struct GpuSceneResources {
    camera_data: CameraData,
    collision_box_data: CollisionBoxData,
    /// Buffers used to draw particles.
    pub render_data: GpuRenderData,

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
        .unwrap();

        let density_view = ImageView::new_default(density_texture.clone()).unwrap();

        Self {
            camera_data: CameraData::new(allocator.clone()),
            collision_box_data: CollisionBoxData::new(allocator.clone()),
            render_data,
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
        //self.render_data.bind_to_command_buffer(builder, pipelines, self.camera_addr(), self.current_frame_idx, self.physics_data.count);
    }

    /// Advances to the next frame resource slot.
    pub fn prepare_next_frame(&mut self) {
        self.current_frame_idx = (self.current_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;
    }
}
