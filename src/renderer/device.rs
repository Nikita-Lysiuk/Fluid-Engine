use std::collections::HashSet;
use std::ffi::CStr;
use ash::{vk, Device, Instance};
use ash::vk::{DeviceCreateInfo, DeviceQueueCreateInfo, PhysicalDevice, PhysicalDeviceFeatures, PhysicalDeviceType, Queue, QueueFlags, StructureType};
use log::{debug, info, warn};
use crate::errors::device_error::DeviceError;
use crate::renderer::swapchain::SwapchainHandler;

#[derive(Default)]
struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>
}
impl QueueFamilyIndices {
    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

pub struct DeviceContext {
    pub device: Device,
    pub graphics_queue: Queue,
    pub present_queue: Queue,
}

impl DeviceContext {
    
    pub fn new(instance: &Instance, surface_ctx: &SwapchainHandler) -> Result<Self, DeviceError> {
        unsafe {
            let (physical_device, queue_family_indices) = Self::find_physical_device(instance, surface_ctx)?;
            info!("[Vulkan] Physical device selected: {:?}", physical_device);
            let logical_device = Self::create_logical_device(&instance, physical_device, &queue_family_indices)?;
            info!("[Vulkan] Logical device created successfully.");

            let graphics_queue = logical_device.get_device_queue(queue_family_indices.graphics_family.unwrap(), 0);
            let present_queue = logical_device.get_device_queue(queue_family_indices.present_family.unwrap(), 0);

            Ok(DeviceContext {
                device: logical_device,
                graphics_queue,
                present_queue,
            })
        }
    }
    unsafe fn find_physical_device(instance: &Instance, surface_ctx: &SwapchainHandler) -> Result<(PhysicalDevice, QueueFamilyIndices), DeviceError> {
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

            let indices = Self::find_queue_families(physical_device, surface_ctx, instance)?;

            if score > 0 {
                info!("Selected GPU with score: {}", score);
                Ok((physical_device, indices))
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
    unsafe fn find_queue_families(
        device: PhysicalDevice,
        surface_ctx: &SwapchainHandler,
        instance: &Instance) -> Result<QueueFamilyIndices, DeviceError> {
        unsafe {
            let mut indices = QueueFamilyIndices::default();

            let queue_family_properties = instance.get_physical_device_queue_family_properties(device);

            for (i, properties) in queue_family_properties.iter().enumerate() {
                let index = i as u32;

                if properties.queue_flags.contains(QueueFlags::GRAPHICS) {
                    indices.graphics_family = Some(index);
                }

                if surface_ctx.get_physical_device_surface_support(device, index)? {
                    indices.present_family = Some(index);
                }
            }

            if !indices.is_complete() {
                Err(DeviceError::QueueFamilyNotFound("Graphics and Present".to_string()))
            } else {
                Ok(indices)
            }
        }
    }
    unsafe fn create_logical_device(
        instance: &Instance,
        physical_device: PhysicalDevice,
        indices: &QueueFamilyIndices) -> Result<Device, DeviceError> {
        let queue_create_info = Self::create_unique_queue_infos(indices);

        let device_features = PhysicalDeviceFeatures {
            geometry_shader: vk::TRUE,
            sampler_anisotropy: vk::TRUE,
            ..PhysicalDeviceFeatures::default()
        };

        let device_create_info = DeviceCreateInfo {
            s_type: StructureType::DEVICE_CREATE_INFO,
            p_queue_create_infos: queue_create_info.as_ptr(),
            queue_create_info_count: queue_create_info.len() as u32,
            p_enabled_features: &device_features,
            ..DeviceCreateInfo::default()
        };

        let device = unsafe {
            instance.create_device(physical_device, &device_create_info, None)?
        };

        Ok(device)
    }

    fn create_unique_queue_infos(indices: &QueueFamilyIndices) -> Vec<DeviceQueueCreateInfo> {
        let mut unique_queue_families = HashSet::<u32>::new();

        unique_queue_families.insert(indices.graphics_family.unwrap());
        unique_queue_families.insert(indices.present_family.unwrap());
        
        let queue_priority = 1.0f32;
        
        unique_queue_families.iter().map(|&queue_family| {
            DeviceQueueCreateInfo {
                s_type: StructureType::DEVICE_QUEUE_CREATE_INFO,
                queue_family_index: queue_family,
                queue_count: 1,
                p_queue_priorities: &queue_priority as *const f32,
                ..DeviceQueueCreateInfo::default()
            }
        }).collect()
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe { self.device.destroy_device(None) }
        info!("[Vulkan] Logical device destroyed.");
    }
}