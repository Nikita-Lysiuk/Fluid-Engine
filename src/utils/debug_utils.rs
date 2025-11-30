use std::ffi::{c_void, CStr};
use ash::vk;
use ash::vk::{DebugUtilsMessageSeverityFlagsEXT as Severity, DebugUtilsMessageTypeFlagsEXT as Type, DebugUtilsMessengerCallbackDataEXT};
use log::{error, warn, info, debug};

pub unsafe extern "system" fn vulkan_debug_callback(
    message_severity: Severity,
    message_type: Type,
    p_callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let callback_data = unsafe { *p_callback_data };

    let message = unsafe { CStr::from_ptr(callback_data.p_message).to_string_lossy() };

    let type_str = match message_type {
        Type::GENERAL => "GENERAL",
        Type::VALIDATION => "VALIDATION",
        Type::PERFORMANCE => "PERFORMANCE",
        _ => "UNKNOWN",
    };

    match message_severity {
        Severity::ERROR => {
            error!("[Vulkan: {}] {}", type_str, message);
        }
        Severity::WARNING => {
            warn!("[Vulkan: {}] {}", type_str, message);
        }
        Severity::INFO => {
            info!("[Vulkan: {}] {}", type_str, message);
        }
        _ => {
            debug!("[Vulkan: {}] {}", type_str, message);
        }
    }

    vk::FALSE
}