use std::ffi::CStr;
use ash::{vk, Device, Instance};
use ash::vk::{PhysicalDevice, PhysicalDeviceType, QueueFlags};
use log::{debug, warn};
use crate::errors::device_error::DeviceError;

#[derive(Default)]
struct QueueFamilyIndices {
    graphics_family: u32
}
pub struct DeviceContext {
    device: Option<Device>,
}

impl DeviceContext {
    
    pub fn new(instance: &Instance) -> Result<Self, DeviceError> {
        unsafe {
            let physical_device = Self::find_physical_device(&instance)?;
            
            
            Ok(DeviceContext {
                device: None
            })
        }
    }
    unsafe fn find_physical_device(instance: &Instance) -> Result<PhysicalDevice, DeviceError> {
        unsafe {
            let physical_devices = instance.enumerate_physical_devices()?;
            if physical_devices.is_empty() { return Err(DeviceError::NoSuitableGpuFound); }
            
            let best_candidate = physical_devices.iter()
                .copied()
                .map(|device| {
                    let score = Self::rate_device_suitability(device, instance);
                    (score, device) 
                })
                .max_by_key(|(score, _)| *score); 

            let (score, physical_device) = best_candidate.ok_or(DeviceError::NoSuitableGpuFound)?;
            
            let indices = Self::find_queue_families(physical_device, instance)?;

            if score > 0 {
                log::info!("Selected GPU with score: {}", score);
                Ok(physical_device)
            } else {
                Err(DeviceError::NoSuitableGpuFound)
            }
        }
    }
    fn rate_device_suitability(device: PhysicalDevice, instance: &Instance) -> u32 {
        unsafe {
            let properties = instance.get_physical_device_properties(device);
            let features = instance.get_physical_device_features(device);

            let mut score = 0;
            
            if features.geometry_shader == vk::FALSE {
                debug!("Device {} lacks Geometry Shaders.", 
                    CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy());
                return 0;
            }
            
            if features.sampler_anisotropy == vk::FALSE {
                debug!("Device {} lacks Anisotropic Filtering.",
                    CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy());
                return 0;
            }
            
            if properties.limits.max_image_dimension3_d < 2048 {
                warn!("Device {} 3D texture limits are low ({}).", 
                    CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy(),
                    properties.limits.max_image_dimension3_d);
            }
            
            
            if properties.device_type == PhysicalDeviceType::DISCRETE_GPU {
                score += 100000;
            } else if properties.device_type == PhysicalDeviceType::INTEGRATED_GPU {
                score += 5000;
            }
            
            score += properties.limits.max_image_dimension2_d / 100;
            
            score += properties.limits.max_image_dimension3_d / 50;
            score += properties.limits.max_viewport_dimensions[0];
            
            score
        }
    }
    unsafe fn find_queue_families(device: PhysicalDevice, instance: &Instance) -> Result<QueueFamilyIndices, DeviceError> {
        unsafe {
            let mut indices = QueueFamilyIndices::default();

            let graphics_family_option = instance
                .get_physical_device_queue_family_properties(device)
                .iter()
                .enumerate()
                .find(|(_, queue_family_prop)| queue_family_prop.queue_flags.contains(QueueFlags::GRAPHICS))
                .map(|(queue_family_index, _)| queue_family_index as u32);

            if let Some(index) = graphics_family_option {
                indices.graphics_family = index;
            } else {
                return Err(DeviceError::QueueFamilyNotFound("Graphics".to_string()));
            }
            
            Ok(indices)
        }
    }
    unsafe fn create_device() -> Result<Device, DeviceError> {
        unimplemented!()
    }
}