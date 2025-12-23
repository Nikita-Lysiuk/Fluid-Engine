use ash::Device;
use ash::vk::{ClearColorValue, ClearValue, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferResetFlags, CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo, Offset2D, PipelineBindPoint, PipelineStageFlags, Queue, Rect2D, RenderPass, RenderPassBeginInfo, StructureType, SubmitInfo};
use log::info;
use crate::errors::command_error::CommandError;
use crate::errors::device_error::DeviceError;
use crate::renderer::device::QueueFamilyIndices;
use crate::renderer::graphics_pipeline::GraphicsPipeline;
use crate::renderer::swapchain_resources::SwapchainResources;
use crate::renderer::sync_objects::SyncObjects;

pub struct CommandContext {
    pub command_pool: Option<CommandPool>,
    pub command_buffer: Vec<CommandBuffer>,
}

impl CommandContext {
    pub fn new() -> Self {

        Self {
            command_pool: None,
            command_buffer: Vec::new(),
        }
    }
    pub fn submit_command_buffer(
        &self,
        device: &Device,
        graphics_queue: Queue,
        sync_objects: &SyncObjects
    ) -> Result<(), CommandError> {
        let submit_info = SubmitInfo {
            s_type: StructureType::SUBMIT_INFO,
            wait_semaphore_count: 1,
            p_wait_semaphores: [sync_objects.image_available_semaphore].as_ptr(),
            p_wait_dst_stage_mask: [PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT].as_ptr(),
            command_buffer_count: self.command_buffer.len() as u32,
            p_command_buffers: self.command_buffer.as_ptr(),
            signal_semaphore_count: 1,
            p_signal_semaphores: [sync_objects.render_finished_semaphore].as_ptr(),
            ..SubmitInfo::default()
        };

        unsafe {
            device.queue_submit(graphics_queue, &[submit_info], sync_objects.in_flight_fence)
                .map_err(|e| CommandError::FailedToSubmitCommandBuffer(e))
        }
    }
    pub fn reset_command_buffer(&self, device: &Device, image_index: usize) -> Result<(), CommandError> {
        unsafe {
            device.reset_command_buffer(
                *self.command_buffer.get(image_index)
                    .ok_or(CommandError::CommandBufferNotAllocated)?,
                CommandBufferResetFlags::empty()
            ).map_err(|e| CommandError::FailedToResetCommandBuffer(e))
        }
    }
    pub fn reallocate_command_buffers(&mut self, device: &Device, image_count: usize) -> Result<(), CommandError> {
        let allocate_info = CommandBufferAllocateInfo {
            s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            command_pool: self.command_pool.ok_or(CommandError::CommandPoolNotCreated)?,
            level: CommandBufferLevel::PRIMARY,
            command_buffer_count: image_count as u32,
            ..CommandBufferAllocateInfo::default()
        };

        self.command_buffer = unsafe {
            device.allocate_command_buffers(&allocate_info)?
        };
        info!("[Command Context] Command buffers re-allocated.");
        Ok(())
    }
    pub fn create_command_pool(&mut self, device: &Device, queue_family_indices: &QueueFamilyIndices) -> Result<(), DeviceError> {
        let pool_info = CommandPoolCreateInfo {
            s_type: StructureType::COMMAND_POOL_CREATE_INFO,
            flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: queue_family_indices.graphics_family
                .ok_or(DeviceError::QueueFamilyNotFound("Graphics".to_string()))?,
            ..CommandPoolCreateInfo::default()
        };

        self.command_pool = unsafe { Some(device.create_command_pool(&pool_info, None)?) };
        info!("[Command Context] Command pool created.");
        Ok(())
    }
    pub fn create_command_buffer(&mut self, device: &Device) -> Result<(), CommandError> {
        let allocate_info = CommandBufferAllocateInfo {
            s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            command_pool: self.command_pool.ok_or(CommandError::CommandPoolNotCreated)?,
            level: CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..CommandBufferAllocateInfo::default()
        };

        self.command_buffer = unsafe {
            device.allocate_command_buffers(&allocate_info)?
        };
        info!("[Command Context] Command buffer allocated.");
        Ok(())
    }
    pub fn record_command_buffer(&self, device: &Device, image_index: usize) -> Result<(), CommandError> {
        let begin_info = CommandBufferBeginInfo {
            s_type: StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_inheritance_info: std::ptr::null(),
            ..CommandBufferBeginInfo::default()
        };

        unsafe {
            device.begin_command_buffer(self.command_buffer[image_index], &begin_info)?
        }
        Ok(())
    }
    pub fn recording_render_pass(
        &self,
        device: &Device,
        render_pass: &RenderPass,
        swapchain_resources: &SwapchainResources,
        image_index: usize
    ) -> Result<(), CommandError> {
        let render_pass_info = RenderPassBeginInfo {
            s_type: StructureType::RENDER_PASS_BEGIN_INFO,
            render_pass: *render_pass,
            framebuffer: *swapchain_resources.swapchain_framebuffers
                .get(image_index).ok_or(CommandError::FramebufferNotFound)?,
            render_area: Rect2D {
                offset: Offset2D { x: 0, y: 0 },
                extent: swapchain_resources.swapchain_extent,
            },
            clear_value_count: 1,
            p_clear_values: &ClearValue {
                color: ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                }
            },
            ..RenderPassBeginInfo::default()
        };

        unsafe {
            device.cmd_begin_render_pass(
                *self.command_buffer
                    .get(image_index).ok_or(CommandError::CommandBufferNotAllocated)?,
                &render_pass_info,
                ash::vk::SubpassContents::INLINE,
            )
        };
        Ok(())
    }
    pub fn record_graphics_commands(&self, device: &Device, swapchain_resources: &SwapchainResources, graphics_pipeline: &GraphicsPipeline, image_index: usize) -> Result<(), CommandError> {
        unsafe {
            let command_buffer = *self.command_buffer.get(image_index).ok_or(CommandError::CommandBufferNotAllocated)?;

            device.cmd_bind_pipeline(
                command_buffer,
                PipelineBindPoint::GRAPHICS,
                graphics_pipeline.graphics_pipeline,
            );

            let viewport = swapchain_resources.get_viewport();
            device.cmd_set_viewport(
                command_buffer,
                0,
                &[viewport],
            );
            let scissor = swapchain_resources.get_scissor();
            device.cmd_set_scissor(
                command_buffer,
                0,
                &[scissor],
            );

            device.cmd_draw(
                command_buffer,
                3,
                1,
                0,
                0,
            );
            Ok(())
        }
    }
    pub fn end_recording(&self, device: &Device, image_index: usize) -> Result<(), CommandError> {
        unsafe {
            let command_buffer = *self.command_buffer.get(image_index).ok_or(CommandError::CommandBufferNotAllocated)?;
            device.cmd_end_render_pass(
                command_buffer,
            );
            device.end_command_buffer(command_buffer)?;
        };
        Ok(())
    }
    pub fn destroy_command_pool(&self, device: &Device) {
        unsafe {
            if let Some(command_pool) = self.command_pool {
                device.destroy_command_pool(command_pool, None);
            }
        }
        info!("[Command Context] Command pool destroyed.");
    }
}