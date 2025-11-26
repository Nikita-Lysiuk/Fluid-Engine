use std::error::Error;
use std::ffi::CString;
use ash::{Entry, Instance, Device, vk};
use ash::vk::{ApplicationInfo, Handle, InstanceCreateInfo, SurfaceKHR, SwapchainKHR};
use ash_window::enumerate_required_extensions;
use winit::raw_window_handle::{DisplayHandle, WindowHandle};
use log::{info, debug, warn, error}; // Додаємо різні рівні логування

/// Core Vulkan renderer component, managing the Vulkan instance and lifecycle-dependent resources.
pub struct Renderer {
    entry: Entry,
    instance: Instance,
    surface_loader: ash::khr::surface::Instance,
    surface: Option<SurfaceKHR>,
    device: Option<Device>,
    swapchain: Option<SwapchainKHR>,
}

impl Renderer {
    /// Initializes the core, window-independent Vulkan components (Entry, Instance, Surface Loader).
    pub fn new(display_handle: DisplayHandle) -> Result<Self, Box<dyn Error>> {
        info!("[Vulkan] Starting Renderer initialization...");
        unsafe {
            let entry = Entry::linked();
            info!("[Vulkan] Entry handle acquired successfully.");

            let instance = Self::create_instance(&entry, display_handle)?;
            info!("[Vulkan] Vulkan Instance (version 1.1+) created.");

            let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
            info!("[Vulkan] Surface extension loader initialized.");

            info!("[Vulkan] Renderer core successfully initialized.");
            Ok(Renderer {
                entry,
                instance,
                surface_loader,
                surface: None,
                device: None,
                swapchain: None,
            })
        }
    }

    unsafe fn create_instance(entry: &Entry, display_handle: DisplayHandle) -> Result<Instance, Box<dyn Error>> {
        let surface_extensions = enumerate_required_extensions(display_handle.as_raw())?;
        debug!("[Vulkan] Required surface extensions: {}", surface_extensions.len());

        let app_name = CString::new("Engine for Fluid Simulation").unwrap();
        let app_info = ApplicationInfo {
            p_application_name:  app_name.as_ptr(),
            application_version: 0,
            api_version: vk::make_api_version(0, 1, 1, 0),
            ..Default::default()
        };

        let instance_info = InstanceCreateInfo {
            p_application_info: &app_info,
            pp_enabled_extension_names: surface_extensions.as_ptr(),
            enabled_extension_count: surface_extensions.len() as u32,
            ..Default::default()
        };
        
        unsafe {
            entry.create_instance(&instance_info, None).map_err(|e| {
                error!("[Vulkan] Failed to create Instance: {:?}", e);
                e.into()
            })
        }
    }
    
    pub unsafe fn create_surface(&mut self, display_handle: DisplayHandle, window_handle: WindowHandle) -> std::result::Result<(), &'static str> {
        info!("[Surface] Attempting to create new Vulkan Surface.");
        unsafe {
            let surface = ash_window::create_surface(
                &self.entry,
                &self.instance,
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            ).map_err(|e| {
                error!("[Surface] FATAL: Surface creation failed: {:?}", e);
                panic!("Failed to create surface: {:?}", e);
            }).unwrap();
            
            if let Some(old_surface) = self.surface.replace(surface) {
                warn!("[Surface] Logic error: Surface creation detected an existing Surface ({:?}).", old_surface.as_raw());
                self.surface.replace(old_surface);
                return Err("Surface creation called when an existing Surface was already present. Did 'suspended' not run?");
            }

            info!("[Surface] Vulkan Surface created successfully. Handle: {:?}", surface.as_raw());
            Ok(())
        }
    }
    
    pub unsafe fn destroy_surface(&mut self) {
        if let Some(surface) = self.surface.take() {
            info!("[Surface] Destroying Vulkan Surface. Handle: {:?}", surface.as_raw());
            unsafe { self.surface_loader.destroy_surface(surface, None); }
            info!("[Surface] Surface destroyed successfully.");
        } else {
            debug!("[Surface] Surface destroy called, but Surface was already None.");
        }
    }
    
    unsafe fn create_device() -> Result<Device, Box<dyn Error>> {
        unimplemented!()
    }

    unsafe fn create_swapchain() -> Result<SwapchainKHR, Box<dyn Error>> {
        unimplemented!()
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        warn!("[Vulkan] Starting Renderer Drop sequence...");
        unsafe {
            self.destroy_surface();

            self.instance.destroy_instance(None);
            info!("[Vulkan] Vulkan Instance destroyed in Drop.");

            warn!("[Vulkan] Renderer Drop sequence completed.");
        }
    }
}