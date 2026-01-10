use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::descriptor_set::allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo};
use vulkano::device::Device;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use crate::utils::constants::MAX_FRAMES_IN_FLIGHT;

#[derive(BufferContents, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShaderData {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    _padding: f32,
}

pub struct FrameResources {
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,

    pub uniform_buffers: Vec<Subbuffer<ShaderData>>,
    pub current_frame_idx: usize
}

impl FrameResources {
    pub fn new(device: Arc<Device>, memory_allocator: Arc<StandardMemoryAllocator>) -> Self {
        let mut uniform_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let buffer = Buffer::new_sized::<ShaderData>(memory_allocator.clone(), BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..BufferCreateInfo::default()
            },AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..AllocationCreateInfo::default()
            })
                .expect("Failed to create uniform buffer");

            uniform_buffers.push(buffer);
        }

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
            current_frame_idx: 0,
        }
    }
}