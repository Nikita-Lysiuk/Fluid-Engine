use std::sync::Arc;
use log::info;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::descriptor_set::allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo};
use vulkano::device::Device;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use crate::entities::camera::ShaderData;
use crate::entities::particle::ParticleVertex;
use crate::utils::constants::{MAX_FRAMES_IN_FLIGHT, MAX_PARTICLES};


pub struct FrameResources {
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,

    uniform_buffers: Vec<Subbuffer<ShaderData>>,
    particle_buffers: Vec<Subbuffer<[ParticleVertex]>>,

    current_frame_idx: usize
}

impl FrameResources {
    pub fn new(device: Arc<Device>, memory_allocator: Arc<StandardMemoryAllocator>) -> Self {
        let mut uniform_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut particle_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let ub = Buffer::new_sized(
                memory_allocator.clone(),
                BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER
                    | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..BufferCreateInfo::default()
            },AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..AllocationCreateInfo::default()
            }).map_err(|e| {
                panic!("[Frame Resources] Failed to create uniform buffer:\n{:?}", e);
            }).unwrap();
            uniform_buffers.push(ub);

            let pb = Buffer::new_slice(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER
                        | BufferUsage::SHADER_DEVICE_ADDRESS
                        | BufferUsage::STORAGE_BUFFER
                        | BufferUsage::TRANSFER_DST,
                    ..BufferCreateInfo::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..AllocationCreateInfo::default()
                },
                MAX_PARTICLES
            ).map_err(|e| {
                panic!("[Frame Resources] Failed to create particle buffer:\n{:?}", e);
            }).unwrap();
            particle_buffers.push(pb);
        }

        info!("[Frame Resources] Created frame resources with {} uniform buffers and {} particle buffers.", uniform_buffers.len(), particle_buffers.len());

        Self {
            command_buffer_allocator: Arc::new(StandardCommandBufferAllocator::new(
                device.clone(),
                StandardCommandBufferAllocatorCreateInfo::default()
            )),
            descriptor_set_allocator: Arc::new(StandardDescriptorSetAllocator::new(
                device,
                StandardDescriptorSetAllocatorCreateInfo::default(),
            )),
            uniform_buffers,
            particle_buffers,
            current_frame_idx: 0,
        }
    }
    pub fn next_frame(&mut self) {
        self.current_frame_idx = (self.current_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;
    }
    pub fn current_ub(&self) -> &Subbuffer<ShaderData> {
        &self.uniform_buffers[self.current_frame_idx]
    }
    pub fn current_pb(&self) -> &Subbuffer<[ParticleVertex]> {
        &self.particle_buffers[self.current_frame_idx]
    }
}