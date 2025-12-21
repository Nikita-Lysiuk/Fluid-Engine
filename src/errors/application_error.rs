use thiserror::Error;
use winit::error::{EventLoopError, OsError};
use crate::errors::command_error::CommandError;
use crate::errors::device_error::DeviceError;
use crate::errors::engine_error::EngineError;
use crate::errors::graphics_pipeline_error::GraphicsPipelineError;
use crate::errors::presentation_error::PresentationError;
use crate::errors::vulkan_instance_error::VulkanInstanceError;

#[derive(Debug, Error)]
pub enum ApplicationError {
    #[error(transparent)]
    PresentationError(#[from] PresentationError),

    #[error(transparent)]
    WindowCreationError(#[from] OsError),

    #[error(transparent)]
    EventLoopInitializationError(#[from] EventLoopError),

    #[error(transparent)]
    VulkanInstance(#[from] VulkanInstanceError),
    
    #[error(transparent)]
    GraphicsPipeline(#[from] GraphicsPipelineError),

    #[error(transparent)]
    Device(#[from] DeviceError),
    
    #[error(transparent)]
    CommandError(#[from] CommandError),

    #[error(transparent)]
    Engine(#[from] EngineError)
}

