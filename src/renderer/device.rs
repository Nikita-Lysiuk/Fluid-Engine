use std::error::Error;
use ash::{Device, Instance};
use crate::errors::device_error::DeviceError;

pub struct DeviceContext {
    device: Option<Device>,
}

impl DeviceContext {
    
    pub fn new(instance: &Instance) -> Result<Self, DeviceError> {
        unsafe {
            let physical_devices = instance.enumerate_physical_devices()?;
            
            if physical_devices.is_empty() {
                return Err(DeviceError::NoSuitableGpuFound);
            }

            Ok ( DeviceContext {
                device: None
            })
        }
    }
    unsafe fn create_device() -> Result<Device, DeviceError> {
        unimplemented!()
    }
}