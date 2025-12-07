use thiserror::Error;
use std::error::Error as StdError; 
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("Resource loading failed: {0}")]
    ResourceLoad(String),

    #[error(transparent)]
    External(#[from] Box<dyn StdError + Send + Sync>),

    #[error("Logical failure: Required Winit handle was missing: {0}")]
    HandleMissing(String),
    
    #[error("Error during window manipulation: {0}")]
    WindowManagement(String)
}