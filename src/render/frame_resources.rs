use std::sync::Arc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::format::Format;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use crate::scene::Scene;
use crate::camera::CameraData;
use crate::box_outline::CollisionBoxData;
use crate::particle::GpuRenderData;
use crate::renderer::pipelines::Pipelines;
use crate::constants::MAX_FRAMES_IN_FLIGHT;

pub struct GpuSceneResources {
    camera_data: CameraData,
    collision_box_data: CollisionBoxData,
    pub render_data: GpuRenderData,


    pub density_texture: Arc<Image>,
    pub density_view: Arc<ImageView>,

    pub current_frame_idx: usize,
}

impl GpuSceneResources {
    pub fn new(allocator: Arc<StandardMemoryAllocator>, scene: &Scene) -> Self {
        let render_data = GpuRenderData::new(
            allocator.clone(),
            &scene.initial_positions,
            scene.sim_params.particle_radius
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
            AllocationCreateInfo::default()
        ).unwrap();
        
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
    pub fn camera_addr(&self) -> u64 {
        self.camera_data.uniform_buffer_addr(self.current_frame_idx)
    }
    pub fn sync_with_scene(&self, scene: &Scene) {
        self.camera_data.write_to_buffer(&scene.camera, self.current_frame_idx);
        self.collision_box_data.write_to_buffer(&scene.boundary, self.current_frame_idx);
    }

    pub fn bind_to_command_buffer<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, pipelines: &Pipelines) {
        self.collision_box_data.bind_to_command_buffer(builder, pipelines, self.camera_addr(), self.current_frame_idx);
        //self.render_data.bind_to_command_buffer(builder, pipelines, self.camera_addr(), self.current_frame_idx, self.physics_data.count);
    }

    pub fn prepare_next_frame(&mut self) {
        self.current_frame_idx = (self.current_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;
    }
}