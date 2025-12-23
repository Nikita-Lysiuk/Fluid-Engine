use ash::vk::Result as VkResult;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CommandError {
    #[error("A generic Vulkan error occurred during device operation: {0:?}")]
    Vulkan(#[from] VkResult),
    
    #[error("Command Pool has not been created yet.")]
    CommandPoolNotCreated,
    
    #[error("Command Buffer has not been allocated yet.")]
    CommandBufferNotAllocated,

    #[error("Framebuffer not found for the given index.")]
    FramebufferNotFound,

    #[error("Failed to allocate command buffers: {0:?}")]
    FailedToResetCommandBuffer(VkResult),

    #[error("Failed to submit command buffer: {0:?}")]
    FailedToSubmitCommandBuffer(VkResult),
}