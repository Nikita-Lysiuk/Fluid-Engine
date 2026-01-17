use std::sync::Arc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::memory::allocator::StandardMemoryAllocator;
use crate::core::scene::Scene;
use crate::entities::camera::CameraData;
use crate::entities::collision::CollisionBoxData;
use crate::entities::particle::ParticleData;
use crate::renderer::pipelines::Pipelines;
use crate::utils::constants::MAX_FRAMES_IN_FLIGHT;

pub struct GpuSceneResources {
    camera_data: CameraData,
    particles_data: ParticleData,
    collision_box_data: CollisionBoxData,

    current_frame_idx: usize,
}

impl GpuSceneResources {
    pub fn new(allocator: Arc<StandardMemoryAllocator>) -> Self {
        Self {
            particles_data: ParticleData::new(allocator.clone()),
            camera_data: CameraData::new(allocator.clone()),
            collision_box_data: CollisionBoxData::new(allocator.clone()),
            current_frame_idx: 0,
        }
    }
    pub fn camera_addr(&self) -> u64 {
        self.camera_data.uniform_buffer_addr(self.current_frame_idx)
    }
    pub fn sync_with_scene(&self, scene: &Scene) {
        self.particles_data.write_to_buffer(&scene.vertices, self.current_frame_idx);
        self.camera_data.write_to_buffer(&scene.camera, self.current_frame_idx);
        self.collision_box_data.write_to_buffer(&scene.boundary, self.current_frame_idx);
    }
    pub fn bind_to_command_buffer<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, pipelines: &Pipelines) {
        self.particles_data.bind_to_command_buffer(builder, pipelines, self.camera_addr(), self.current_frame_idx);
    }
    pub fn prepare_next_frame(&mut self) {
        self.current_frame_idx = (self.current_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;
    }
}