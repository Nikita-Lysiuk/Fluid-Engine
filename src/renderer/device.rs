use std::error::Error;
use ash::Device;

pub struct DeviceContext {
    device: Option<Device>,
}

impl DeviceContext {
    
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Ok ( DeviceContext {
            device: None
        })
    }
    unsafe fn create_device() -> Result<Device, Box<dyn Error>> {
        unimplemented!()
    }
}