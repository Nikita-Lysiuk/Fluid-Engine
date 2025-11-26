use std::fs::File;
use ash::util;
use log::warn;
use winit::window::Icon;

pub struct Loader;

impl Loader {
    pub fn load_icon(bytes: &[u8]) -> Option<Icon> {
        let (icon_rgba, icon_width, icon_height) = {
            let image = image::load_from_memory(bytes).unwrap().into_rgba8();
            let (width, height) = image.dimensions();
            let rgba = image.into_raw();
            (rgba, width, height)
        };
        Icon::from_rgba(icon_rgba, icon_width, icon_height).map_err(|e| {
            warn!("Failed to create icon from RGBA data: {}", e);
            e
        }).ok()
    }

    pub fn load_shader_code(path: &str) -> Vec<u32> {
        let mut spv_file = File::open(path).expect("compiled spv filed should be present");
        util::read_spv(&mut spv_file).unwrap()
    }
}