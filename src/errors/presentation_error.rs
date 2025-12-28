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

    #[error("Swapchain dependency (loader) is missing.")]
    SwapchainDependencyMissing,

    #[error("Failed to create swapchain: {0:?}")]
    AcquireNextImage(VkResult),

    #[error("Failed to present swapchain image: {0:?}")]
    QueuePresentFailed(VkResult),

    #[error("Swapchain is out of date and needs to be recreated.")]
    SwapchainOutOfDate
}

