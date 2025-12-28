use ash::Device;
use ash::vk::{Fence, FenceCreateFlags, FenceCreateInfo, Semaphore, SemaphoreCreateInfo, StructureType};
use log::info;
use crate::errors::sync_error::SyncError;
use crate::utils::constants::MAX_FRAMES_IN_FLIGHT;

pub struct SyncObjects {
    pub image_available_semaphores: Vec<Semaphore>,
    pub render_finished_semaphores: Vec<Semaphore>,
    pub in_flight_fences: Vec<Fence>,
    pub images_in_flight: Vec<Fence>
}

impl SyncObjects {
    pub fn new(device: &Device, image_count: usize) -> Result<Self, SyncError> {
        let image_available_semaphores = Self::create_semaphore(device, MAX_FRAMES_IN_FLIGHT as usize)?;
        let render_finished_semaphores = Self::create_semaphore(device, image_count)?;
        let in_flight_fences = Self::create_fence(device)?;

        info!("[Sync Objects] Synchronization objects created successfully.");
        Ok(SyncObjects {
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight: vec![Fence::null(); image_count],
        })
    }
    pub fn resize_images_in_flight(&mut self, image_count: usize) {
        self.images_in_flight = vec![Fence::null(); image_count];
        info!("[Sync Objects] Resized images_in_flight to accommodate {} images.", image_count);
    }
    fn create_semaphore(device: &Device, count: usize) -> Result<Vec<Semaphore>, SyncError> {
        let semaphore_info = SemaphoreCreateInfo {
            s_type: StructureType::SEMAPHORE_CREATE_INFO,
            ..SemaphoreCreateInfo::default()
        };

        let mut semaphores: Vec<Semaphore> = Vec::with_capacity(count);

        for _ in 0..count {
            let semaphore = unsafe {
                device.create_semaphore(&semaphore_info, None)
                    .map_err(|e| SyncError::FailedToCreateSemaphore(e))?
            };

            semaphores.push(semaphore);
        }

        Ok(semaphores)
    }
    fn create_fence(device: &Device) -> Result<Vec<Fence>, SyncError> {
        let fence_info = FenceCreateInfo {
            s_type: StructureType::FENCE_CREATE_INFO,
            flags: FenceCreateFlags::SIGNALED,
            ..FenceCreateInfo::default()
        };

        let mut fences: Vec<Fence> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT as usize);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let fence = unsafe {
                device.create_fence(&fence_info, None)
                    .map_err(|e| SyncError::FailedToCreateFence(e))?
            };

            fences.push(fence);
        }

        Ok(fences)
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            for i in 0..MAX_FRAMES_IN_FLIGHT as usize {
                device.destroy_semaphore(self.image_available_semaphores[i], None);
                device.destroy_semaphore(self.render_finished_semaphores[i], None);
                device.destroy_fence(self.in_flight_fences[i], None);
            }
            info!("[Sync Objects] All synchronization objects destroyed.");
        }
    }
}