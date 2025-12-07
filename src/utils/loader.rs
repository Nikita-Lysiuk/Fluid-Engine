use std::fs::File;
use ash::util;
use winit::window::Icon;
use crate::errors::engine_error::EngineError;

pub struct Loader;

impl Loader {
    pub fn load_icon(bytes: &[u8]) -> Result<Icon, EngineError> {
        let (icon_rgba, icon_width, icon_height) = {
            let image = image::load_from_memory(bytes).unwrap().into_rgba8();
            let (width, height) = image.dimensions();
            let rgba = image.into_raw();
            (rgba, width, height)
        };
        Icon::from_rgba(icon_rgba, icon_width, icon_height).map_err(|e| {
            EngineError::ResourceLoad(format!("Failed to create icon from RGBA data: {}", e))
        })
    }

    pub fn load_shader_code(path: &str) -> Result<Vec<u32>, EngineError> {
        let mut spv_file = File::open(path).map_err(|e| {;
            EngineError::ResourceLoad(format!("Failed to open shader file at path {}: {}", path, e))
        })?;
        util::read_spv(&mut spv_file).map_err(|e| {
            EngineError::ResourceLoad(format!("Failed to read SPIR-V shader code from file {}: {}", path, e))
        })
    }
}