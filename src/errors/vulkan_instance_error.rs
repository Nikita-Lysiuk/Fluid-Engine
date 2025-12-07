use ash::vk::Result as VkResult;
use thiserror::Error;
use std::ffi::NulError;

#[derive(Debug, Error)]
pub enum VulkanInstanceError {

    #[error("Vulkan Instance creation failed (generic error): {0:?}")]
    Vulkan(#[from] VkResult),

    #[error("Failed to enumerate required Vulkan extensions (surface extensions): {0:?}")]
    ExtensionEnumeration(VkResult),

    #[error("One or more required validation layers are not supported: {0}")]
    ValidationLayerNotSupported(String),

    #[error("Internal error: Null character found in a string (CStr conversion failed): {0}")]
    Nul(#[from] NulError),
}