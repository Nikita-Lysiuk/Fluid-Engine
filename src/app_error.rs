use thiserror::Error;
use winit::error::{EventLoopError, OsError};

#[derive(Debug, Error)]
pub enum AppError {
    #[error(transparent)]
    WindowCreation(#[from] OsError),

    #[error(transparent)]
    EventLoopInit(#[from] EventLoopError),

    #[error("Failed to load resource: {0}")]
    ResourceLoad(String),

    #[error("An unexpected application error occurred: {0}")]
    Other(String),
}
