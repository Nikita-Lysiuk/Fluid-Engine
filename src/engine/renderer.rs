use std::error::Error;
use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use ash::{Entry, Instance, Device, vk};
use ash::vk::{ApplicationInfo, DebugUtilsMessengerCreateInfoEXT, Handle, InstanceCreateInfo, StructureType, SurfaceKHR, SwapchainKHR};
use ash_window::enumerate_required_extensions;
use winit::raw_window_handle::{DisplayHandle, WindowHandle};
use log::{info, debug, warn, error};
use crate::utils::constants::{APPLICATION_NAME, APPLICATION_VERSION, DEBUG_UTILS_EXTENSION_NAME, ENABLE_VALIDATION_LAYERS, ENGINE_VERSION, VALIDATION_LAYERS};
use ash::vk::DebugUtilsMessageSeverityFlagsEXT as Severity;
use ash::vk::DebugUtilsMessageTypeFlagsEXT as Type;
use crate::utils::debug_utils::vulkan_debug_callback;
use ash::ext::debug_utils;
// Додаємо різні рівні логування

/// Core Vulkan renderer component, managing the Vulkan instance and lifecycle-dependent resources.
pub struct Renderer {
    entry: Entry,
    instance: Instance,
    surface_loader: ash::khr::surface::Instance,
    surface: Option<SurfaceKHR>,
    device: Option<Device>,
    swapchain: Option<SwapchainKHR>,

    #[cfg(debug_assertions)]
    debug_utils_loader: debug_utils::Instance,
    #[cfg(debug_assertions)]
    debug_messenger: vk::DebugUtilsMessengerEXT,
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
            
            #[cfg(debug_assertions)]
            let (debug_utils_loader, debug_messenger) = Self::setup_debug_messenger(&entry, &instance)?;
            info!("[Vulkan] Debug Messenger initialized.");

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
                #[cfg(debug_assertions)]
                debug_utils_loader,
                #[cfg(debug_assertions)]
                debug_messenger,
            })
        }
    }

    unsafe fn create_instance(entry: &Entry, display_handle: DisplayHandle) -> Result<Instance, Box<dyn Error>> {
        let mut surface_extensions = enumerate_required_extensions(display_handle.as_raw())?.to_vec();
        debug!("[Vulkan] Required surface extensions: {}", surface_extensions.len());
        
        let app_info = ApplicationInfo {
            s_type: StructureType::APPLICATION_INFO,
            p_application_name:  APPLICATION_NAME,
            application_version: APPLICATION_VERSION,
            api_version: APPLICATION_VERSION,
            engine_version: ENGINE_VERSION,
            ..Default::default()
        };

        let debug_messenger_info = Self::create_debug_utils_info();
        
        if ENABLE_VALIDATION_LAYERS {
            // Додаємо розширення Debug Utils
            surface_extensions.push(DEBUG_UTILS_EXTENSION_NAME.as_ptr());
        }

        let mut instance_info = InstanceCreateInfo {
            s_type: StructureType::INSTANCE_CREATE_INFO,
            p_application_info: &app_info,
            p_next: if ENABLE_VALIDATION_LAYERS {
                &debug_messenger_info as *const DebugUtilsMessengerCreateInfoEXT as *const c_void
            } else {
                ptr::null()
            },
            ..Default::default()
        };

        #[cfg(target_os = "macos")]
        {
            info!("[Vulkan] Enabling macOS Portability extensions and flags.");
            instance_info.flags |= InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
            surface_extensions.push(KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
        }
        
        instance_info.enabled_extension_count = surface_extensions.len() as u32;
        instance_info.pp_enabled_extension_names = surface_extensions.as_ptr();
        
        Self::check_and_enable_validation_layers(entry, &mut instance_info, &VALIDATION_LAYERS)?;
        
        unsafe {
            entry.create_instance(&instance_info, None).map_err(|e| {
                error!("[Vulkan] Failed to create Instance: {:?}", e);
                e.into()
            })
        }
    }
    pub unsafe fn create_surface(&mut self, display_handle: DisplayHandle, window_handle: WindowHandle) -> Result<(), &'static str> {
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
    
    unsafe fn setup_debug_messenger(entry: &Entry, instance: &Instance) -> Result<(debug_utils::Instance, vk::DebugUtilsMessengerEXT), Box<dyn Error>> {
        let debug_info = Self::create_debug_utils_info();
        let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
        let debug_messenger = unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_info, None)? };
        Ok((debug_utils_loader, debug_messenger))
    }
    fn check_and_enable_validation_layers(
        entry: &Entry,
        instance_info: &mut InstanceCreateInfo,
        required_layers: &[*const c_char], 
    ) -> Result<(), Box<dyn Error>> {

        if !ENABLE_VALIDATION_LAYERS {
            info!("[Vulkan] Validation layers are disabled in this build.");
            return Ok(());
        }
        
        let available_layers = unsafe { entry.enumerate_instance_layer_properties()? };
        
        let all_supported = required_layers.iter().all(|&required_layer_ptr| {
            let is_supported = available_layers.iter().any(|available_prop| {
                let available_name = unsafe { CStr::from_ptr(available_prop.layer_name.as_ptr()) };
                let required_name = unsafe { CStr::from_ptr(required_layer_ptr) };
                available_name == required_name
            });

            if !is_supported {
                let req_name_str = unsafe { CStr::from_ptr(required_layer_ptr).to_string_lossy() };
                error!("[Vulkan] Required validation layer '{}' is NOT supported.", req_name_str);
            }
            is_supported
        });
        
        if all_supported {
            info!("[Vulkan] All required validation layers are supported and enabled.");
            instance_info.enabled_layer_count = required_layers.len() as u32;
            instance_info.pp_enabled_layer_names = required_layers.as_ptr();
            Ok(())
        } else {
            Err("One or more required validation layers are not supported.".into())
        }
    }
    
    fn create_debug_utils_info() -> DebugUtilsMessengerCreateInfoEXT<'static> {
        DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                Severity::WARNING | Severity::ERROR | Severity::INFO
            )
            .message_type(
                Type::GENERAL | Type::VALIDATION | Type::PERFORMANCE
            )
            .pfn_user_callback(Some(vulkan_debug_callback))
    }
}
impl Drop for Renderer {
    fn drop(&mut self) {
        warn!("[Vulkan] Starting Renderer Drop sequence...");
        unsafe {
            self.destroy_surface();

            #[cfg(debug_assertions)]
            {
                warn!("[Vulkan] Destroying Debug Messenger.");
                self.debug_utils_loader.destroy_debug_utils_messenger(self.debug_messenger, None);
            }

            self.instance.destroy_instance(None);
            info!("[Vulkan] Vulkan Instance destroyed in Drop.");

            warn!("[Vulkan] Renderer Drop sequence completed.");
        }
    }
}