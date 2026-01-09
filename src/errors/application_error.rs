use thiserror::Error;
use winit::error::{EventLoopError, OsError};

#[derive(Debug, Error)]
pub enum ApplicationError {
    #[error(transparent)]
    WindowCreationError(#[from] OsError),

    #[error(transparent)]
    EventLoopInitializationError(#[from] EventLoopError),

    #[error("Failed to load resource: {0}")]
    ResourceLoadError(String),
    
    #[error("An unexpected application error occurred: {0}")]
    Other(String),
}

