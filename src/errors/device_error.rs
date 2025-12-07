use ash::vk::Result as VkResult;
use thiserror::Error;

#[derive(Debug, Error)] 
pub enum DeviceError {
    #[error("A generic Vulkan error occurred during device operation: {0:?}")]
    Vulkan(#[from] VkResult),

    #[error("Failed to find any Physical Device (GPU) with Vulkan support and required features.")]
    NoSuitableGpuFound,

    #[error("Required command queue family not found on selected GPU: {0}")]
    QueueFamilyNotFound(String),

    #[error("Selected GPU does not support required device extension: {0}")]
    ExtensionNotSupported(String),

    #[error("Failed to create the logical Vulkan device: {0:?}")]
    DeviceCreationFailure(VkResult),

    #[error("An unexpected device-related error occurred: {0}")]
    Other(String),
}