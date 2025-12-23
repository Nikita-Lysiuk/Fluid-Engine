use ash::vk::Result as VkResult;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SyncError {
    
    #[error("A generic Vulkan error occurred during device operation: {0:?}")]
    Vulkan(#[from] VkResult),

    #[error("Failed to create semaphore {0:?}")]
    FailedToCreateSemaphore(VkResult),

    #[error("Failed to create fence {0:?}")]
    FailedToCreateFence(VkResult),

    #[error("Failed to wait for fence {0:?}")]
    FailedToWaitForFence(VkResult),

    #[error("Failed to reset fence {0:?}")]
    FailedToResetFence(VkResult),
}