use ash::vk::Result as VkResult;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphicsPipelineError {
    #[error(transparent)]
    VkError(#[from] VkResult),
    
    #[error("Failed to create shader module: {0}")]
    ShaderModuleCreationError(VkResult),

    #[error("Failed to create pipeline layout: {0}")]
    PipelineLayoutCreationError(VkResult),

    #[error("Failed to create render pass: {0}")]
    RenderPassCreationError(VkResult),
}