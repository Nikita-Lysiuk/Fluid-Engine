use std::sync::Arc;
use vulkano::device::Device;
use vulkano::shader::{EntryPoint, ShaderModule};

pub fn load_shader_entry_point(
    device: Arc<Device>,
    loader: fn(Arc<Device>) -> Result<Arc<ShaderModule>, vulkano::Validated<vulkano::VulkanError>>,
    shader_name: &str,
) -> EntryPoint {
    loader(device)
        .unwrap_or_else(|_| panic!("failed to create {} shader", shader_name))
        .entry_point("main")
        .expect("failed to load entry point")
}