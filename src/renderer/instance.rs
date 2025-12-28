use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use ash::{Entry, Instance};
use ash::vk::{
    ApplicationInfo, 
    DebugUtilsMessageSeverityFlagsEXT as Severity, 
    DebugUtilsMessageTypeFlagsEXT as Type, 
    DebugUtilsMessengerCreateInfoEXT, 
    InstanceCreateInfo, 
    StructureType
};
use crate::utils::debug_utils::vulkan_debug_callback;
use ash_window::enumerate_required_extensions;
use log::{debug, error, info};
use winit::raw_window_handle::DisplayHandle;
use crate::errors::vulkan_instance_error::VulkanInstanceError;
use crate::utils::constants::{
    APPLICATION_NAME, 
    APPLICATION_VERSION, 
    DEBUG_UTILS_EXTENSION_NAME, 
    ENABLE_VALIDATION_LAYERS, 
    ENGINE_VERSION, 
    VALIDATION_LAYERS
};
#[cfg(debug_assertions)]
use ash::vk;
#[cfg(debug_assertions)]
use ash::ext::debug_utils;

pub struct VulkanInstanceContext {
    pub entry: Entry,
    pub instance: Instance,
    #[cfg(debug_assertions)]
    debug_utils_loader: debug_utils::Instance,
    #[cfg(debug_assertions)]
    debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl VulkanInstanceContext {
    pub fn new(display_handle: DisplayHandle) -> Result<Self, VulkanInstanceError> { 
        unsafe { 
            let entry = Entry::linked();
            info!("[Instance Context] Entry handle acquired successfully.");
    
            let instance = Self::create_instance(&entry, display_handle)?;
            info!("[Instance Context] Vulkan Instance (version 1.3) created.");
    
            #[cfg(debug_assertions)]
            let (debug_utils_loader, debug_messenger) = Self::setup_debug_messenger(&entry, &instance)?;
            info!("[Instance Context] Debug Messenger initialized.");
        
            Ok(VulkanInstanceContext {
                entry,
                instance,
                #[cfg(debug_assertions)]
                debug_utils_loader,
                #[cfg(debug_assertions)]
                debug_messenger,
            })
        }
    }
    unsafe fn create_instance(entry: &Entry, display_handle: DisplayHandle) -> Result<Instance, VulkanInstanceError> {
        let mut surface_extensions = enumerate_required_extensions(display_handle.as_raw())
            .map_err(VulkanInstanceError::ExtensionEnumeration)? 
            .to_vec();
        debug!("[Instance Context] Required surface extensions: {}", surface_extensions.len());

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
                error!("[Instance Context] Failed to create Instance: {:?}", e);
                VulkanInstanceError::Vulkan(e)
            })
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
    #[cfg(debug_assertions)]
    unsafe fn setup_debug_messenger(entry: &Entry, instance: &Instance) -> Result<(debug_utils::Instance, vk::DebugUtilsMessengerEXT), VulkanInstanceError> {
        let debug_info = Self::create_debug_utils_info();
        let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
        let debug_messenger = unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_info, None)? };
        Ok((debug_utils_loader, debug_messenger))
    }
    fn check_and_enable_validation_layers(
        entry: &Entry,
        instance_info: &mut InstanceCreateInfo,
        required_layers: &[*const c_char],
    ) -> Result<(), VulkanInstanceError> {

        if !ENABLE_VALIDATION_LAYERS {
            info!("[Instance Context] Validation layers are disabled in this build.");
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
                error!("[Instance Context] Required validation layer '{}' is NOT supported.", req_name_str);
            }
            is_supported
        });

        if all_supported {
            info!("[Instance Context] All required validation layers are supported and enabled.");
            instance_info.enabled_layer_count = required_layers.len() as u32;
            instance_info.pp_enabled_layer_names = required_layers.as_ptr();
            Ok(())
        } else {
            Err(VulkanInstanceError::ValidationLayerNotSupported("One or more required validation layers are not supported.".to_string()))
        }
    }
    pub unsafe fn destroy_self(&mut self) {
        unsafe {
            #[cfg(debug_assertions)]
            {
                info!("[Instance Context] Destroying Debug Messenger.");
                self.debug_utils_loader.destroy_debug_utils_messenger(self.debug_messenger, None);
            }

            self.instance.destroy_instance(None);
            info!("[Instance Context] Vulkan Instance destroyed in Drop.");
        }
    }
}