use ash::vk::Result as VkResult;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PresentationError {
    #[error(transparent)]
    VkError(#[from] VkResult),

    #[error("Winit Event Loop creation failed: {0}")]
    EventLoopCreation(String),

    #[error("Vulkan Surface creation failed: {0:?}")]
    SurfaceCreation(VkResult),

    #[error("Surface creation called when an existing Surface was already present. Did 'suspended' not run?")]
    SurfaceAlreadyExists,

    #[error("No suitable surface format found for the swapchain.")]
    NoSuitableSurfaceFormatFound,
    
    #[error("Swapchain relied resources were not initialized before use.")]
    SwapchainResourcesNotInitialized,
}

