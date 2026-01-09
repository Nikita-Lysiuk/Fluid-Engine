


use thiserror::Error;
use vulkano::LoadingError;

#[derive(Debug, Error)]
pub enum RendererError {

    #[error(transparent)]
    VulkanLibraryLoadError(#[from] LoadingError)
}