use thiserror::Error;
use std::error::Error as StdError;
use winit::error::{EventLoopError, OsError};
use crate::errors::device_error::DeviceError;
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
    Device(#[from] DeviceError),

    #[error("External (I/O or third-party) error: {0}")]
    External(Box<dyn StdError + Send + Sync>),
}

