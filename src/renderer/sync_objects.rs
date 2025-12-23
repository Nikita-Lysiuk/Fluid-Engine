use ash::Device;
use ash::vk::{Fence, FenceCreateFlags, FenceCreateInfo, Semaphore, SemaphoreCreateInfo, StructureType};
use log::info;
use crate::errors::sync_error::SyncError;

pub struct SyncObjects {
    pub image_available_semaphore: Semaphore,
    pub render_finished_semaphore: Semaphore,
    pub in_flight_fence: Fence
}

impl SyncObjects {
    pub fn new(device: &Device) -> Result<Self, SyncError> {
        let image_available_semaphore = Self::create_semaphore(device)?;
        let render_finished_semaphore = Self::create_semaphore(device)?;
        let in_flight_fence = Self::create_fence(device)?;

        info!("[Sync Objects] Synchronization objects created successfully.");
        Ok(SyncObjects {
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence
        })
    }

    fn create_semaphore(device: &Device) -> Result<Semaphore, SyncError> {
        let semaphore_info = SemaphoreCreateInfo {
            s_type: StructureType::SEMAPHORE_CREATE_INFO,
            ..SemaphoreCreateInfo::default()
        };

        unsafe {
            device.create_semaphore(&semaphore_info, None)
                .map_err(|e| SyncError::FailedToCreateSemaphore(e))
        }
    }
    fn create_fence(device: &Device) -> Result<Fence, SyncError> {
        let fence_info = FenceCreateInfo {
            s_type: StructureType::FENCE_CREATE_INFO,
            flags: FenceCreateFlags::SIGNALED,
            ..FenceCreateInfo::default()
        };

        unsafe {
            device.create_fence(&fence_info, None)
                .map_err(|e| SyncError::FailedToCreateFence(e))
        }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.image_available_semaphore, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);
            device.destroy_fence(self.in_flight_fence, None);
            info!("[Sync Objects] All synchronization objects destroyed.");
        }
    }
}