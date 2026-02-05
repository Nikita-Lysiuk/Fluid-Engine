use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use crate::core::scene::Scene;
use crate::entities::camera::CameraData;
use crate::entities::collision::CollisionBoxData;
use crate::entities::particle::{GpuPhysicsData, GpuRenderData, SimulationParams};
use crate::renderer::pipelines::Pipelines;
use crate::utils::constants::MAX_FRAMES_IN_FLIGHT;

pub struct GpuSceneResources {
    camera_data: CameraData,
    collision_box_data: CollisionBoxData,
    pub physics_data: GpuPhysicsData,
    pub render_data: GpuRenderData,

    pub sim_params_buffer: Subbuffer<SimulationParams>,

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
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            scene.sim_params,
        ).expect("Failed to create simulation params buffer");

        Self {
            camera_data: CameraData::new(allocator.clone()),
            collision_box_data: CollisionBoxData::new(allocator.clone()),
            physics_data,
            render_data,
            sim_params_buffer,
            current_frame_idx: 0,
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
        self.render_data.bind_to_command_buffer(builder, pipelines, self.camera_addr(), self.current_frame_idx, self.physics_data.count);
    }

    pub fn prepare_next_frame(&mut self) {
        self.current_frame_idx = (self.current_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;
    }
}