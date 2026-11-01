use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::format::Format;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use crate::core::scene::Scene;
use crate::entities::camera::CameraData;
use crate::entities::collision::CollisionBoxData;
use crate::entities::particle::{stats_layout, GpuPhysicsData, GpuRenderData, SimulationParams};
use crate::renderer::pipelines::Pipelines;
use crate::utils::constants::MAX_FRAMES_IN_FLIGHT;

pub struct GpuSceneResources {
    camera_data: CameraData,
    collision_box_data: CollisionBoxData,
    pub physics_data: GpuPhysicsData,
    pub render_data: GpuRenderData,

    pub sim_params_buffer: Subbuffer<SimulationParams>,
    pub stats_readback: Vec<Subbuffer<[u32]>>,

    pub density_texture: Arc<Image>,
    pub density_view: Arc<ImageView>,

    pub current_frame_idx: usize,
}

impl GpuSceneResources {
    pub fn new(allocator: Arc<StandardMemoryAllocator>, scene: &Scene) -> Self {
        let physics_data = GpuPhysicsData::new(
            allocator.clone(),
            scene.initial_positions.clone()
        );

        let render_data = GpuRenderData::new(
            allocator.clone(),
            &scene.initial_positions,
            scene.sim_params.particle_radius
        );

        let sim_params_buffer = Buffer::from_data(
            allocator.clone(),
            BufferCreateInfo {
                // Updated with `update_buffer` in the command stream, so the
                // CPU never writes it while the GPU may still be reading it.
                usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            scene.sim_params,
        ).expect("Failed to create simulation params buffer");

        // One copy of the stats per frame in flight: the CPU reads the oldest
        // one, which the GPU has finished with.
        let stats_readback = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                Buffer::from_iter(
                    allocator.clone(),
                    BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    [0u32; stats_layout::LEN],
                )
                .expect("Failed to create stats readback buffer")
            })
            .collect();
        
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
            physics_data,
            render_data,
            sim_params_buffer,
            stats_readback,
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

    /// Stats written by the GPU `MAX_FRAMES_IN_FLIGHT - 1` frames ago, if
    /// that frame has finished.
    pub fn oldest_stats(&self) -> Option<vulkano::buffer::BufferReadGuard<'_, [u32]>> {
        let oldest = (self.current_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;
        self.stats_readback[oldest].read().ok()
    }

    pub fn bind_to_command_buffer<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, pipelines: &Pipelines) {
        self.collision_box_data.bind_to_command_buffer(builder, pipelines, self.camera_addr(), self.current_frame_idx);
        //self.render_data.bind_to_command_buffer(builder, pipelines, self.camera_addr(), self.current_frame_idx, self.physics_data.count);
    }

    pub fn prepare_next_frame(&mut self) {
        self.current_frame_idx = (self.current_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;
    }
}