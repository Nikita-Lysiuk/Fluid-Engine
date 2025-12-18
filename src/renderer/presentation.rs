use ash::{vk, Device, Entry, Instance};
use ash::vk::{
    CompositeAlphaFlagsKHR,
    Extent2D,
    Handle,
    ImageUsageFlags,
    PhysicalDevice,
    PresentModeKHR,
    SharingMode,
    StructureType,
    SurfaceCapabilitiesKHR,
    SurfaceFormatKHR,
    SurfaceKHR,
    SwapchainCreateInfoKHR,
    SwapchainKHR
};
use log::{debug, info, warn};
use winit::raw_window_handle::{DisplayHandle, WindowHandle};
use winit::window::Window;
use crate::errors::device_error::DeviceError;
use crate::errors::presentation_error::PresentationError;
use crate::renderer::device::QueueFamilyIndices;
use crate::renderer::instance;

#[derive(Default)]
pub struct SwapchainSupportDetails {
    pub capabilities: SurfaceCapabilitiesKHR,
    pub formats: Vec<SurfaceFormatKHR>,
    pub present_modes: Vec<PresentModeKHR>,
}

pub struct PresentationContext {
    surface_loader: ash::khr::surface::Instance,
    surface: Option<SurfaceKHR>,
    swapchain_loader: Option<ash::khr::swapchain::Device>,
}

impl PresentationContext {
    pub fn new(entry: &Entry, instance: &Instance) -> Self {
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        info!("[Presentation Context] Surface extension loader initialized.");

            
        PresentationContext { 
            surface_loader, 
            surface: None,
            swapchain_loader: None,
        }
    }
    pub unsafe fn create_surface(&mut self, instance_ctx: &instance::VulkanInstanceContext, display_handle: DisplayHandle, window_handle: WindowHandle) -> Result<(), PresentationError> {
        info!("[Presentation Context] Attempting to create new Vulkan Surface.");
        unsafe {
            let surface = ash_window::create_surface(
                &instance_ctx.entry,
                &instance_ctx.instance,
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            ).map_err(PresentationError::SurfaceCreation)?;

            if let Some(old_surface) = self.surface.replace(surface) {
                warn!("[Presentation Context] Logic error: Surface creation detected an existing Surface ({:?}).", old_surface.as_raw());
                self.surface.replace(old_surface);
                return Err(PresentationError::SurfaceAlreadyExists);
            }

            info!("[Presentation Context] Vulkan Surface created successfully. Handle: {:?}", surface.as_raw());
            Ok(())
        }
    }
    pub unsafe fn destroy_surface(&mut self) {
        if let Some(surface) = self.surface.take() {
            info!("[Presentation Context] Destroying Vulkan Surface. Handle: {:?}", surface.as_raw());
            unsafe { self.surface_loader.destroy_surface(surface, None); }
            info!("[Presentation Context] Surface destroyed successfully.");
        } else {
            debug!("[Presentation Context] Surface destroy called, but Surface was already None.");
        }
    }
    pub unsafe fn create_swapchain(
        &mut self,
        instance: &Instance,
        device: &Device,
        mut old_swapchain: Option<SwapchainKHR>,
        physical_device: PhysicalDevice,
        indices: &QueueFamilyIndices,
        window: &Window) -> Result<(SwapchainKHR, Vec<vk::Image>, vk::Format, Extent2D), DeviceError> {
        let swapchain_loader = ash::khr::swapchain::Device::new(instance, device);

        let swapchain_support = unsafe { self.query_swapchain_support(physical_device)? };

        let surface_format = self.choose_swap_surface_format(&swapchain_support.formats);
        let present_mode = self.choose_swap_present_mode(&swapchain_support.present_modes);
        let extent = self.choose_swap_extent(&swapchain_support.capabilities, window);

        let mut image_count = swapchain_support.capabilities.min_image_count + 1;
        if swapchain_support.capabilities.max_image_count > 0
            && image_count > swapchain_support.capabilities.max_image_count {
            image_count = swapchain_support.capabilities.max_image_count;
        }

        let mut create_info = SwapchainCreateInfoKHR {
            s_type: StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            surface: self.surface.ok_or(DeviceError::SurfaceDependencyMissing)?,
            min_image_count: image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: ImageUsageFlags::COLOR_ATTACHMENT,
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: SwapchainKHR::null(),
            ..SwapchainCreateInfoKHR::default()
        };

        if indices.graphics_family != indices.present_family {
            let queue_family_indices = [indices.graphics_family.unwrap(), indices.present_family.unwrap()];
            create_info.image_sharing_mode = SharingMode::CONCURRENT;
            create_info.queue_family_index_count = queue_family_indices.len() as u32;
            create_info.p_queue_family_indices = queue_family_indices.as_ptr();
        } else {
            create_info.image_sharing_mode = SharingMode::EXCLUSIVE;
            create_info.queue_family_index_count = 0;
            create_info.p_queue_family_indices = std::ptr::null();
        }

        let mut old_swapchain_handle = None;
        if let Some(swapchain) = old_swapchain.take() {
            warn!("[Presentation Context] Logic error: Swapchain creation detected an existing Swapchain ({:?}). Replacing it.", swapchain.as_raw());
            create_info.old_swapchain = swapchain;
            old_swapchain_handle = Some(swapchain);
        }

        let swapchain = unsafe {
            swapchain_loader.create_swapchain(&create_info, None)?
        };

        info!("[Presentation Context] Swapchain created successfully. Handle: {:?}. With extent: {:?}", swapchain.as_raw(), extent);

        if let Some(swapchain) = old_swapchain_handle {
            unsafe {
                swapchain_loader.destroy_swapchain(swapchain, None);
            }
        }

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

        self.swapchain_loader = Some(swapchain_loader);
        info!("[Presentation Context] Swapchain extension loader initialized.");

        Ok((swapchain, images, surface_format.format, extent))
    }
    pub unsafe fn destroy_swapchain(&mut self, mut swapchain: Option<SwapchainKHR>) {
        if let Some(swapchain) = swapchain.take() {
            info!("[Presentation Context] Destroying Swapchain. Handle: {:?}", swapchain.as_raw());
            unsafe { self.swapchain_loader.as_ref().unwrap().destroy_swapchain(swapchain, None) }
            info!("[Presentation Context] Swapchain destroyed successfully.");
        } else {
            debug!("[Presentation Context] Swapchain destroy called, but Swapchain was already None.");
        }
    }
    pub unsafe fn query_swapchain_support(&self, physical_device: PhysicalDevice) -> Result<SwapchainSupportDetails, DeviceError> {
        unsafe {
            let mut details = SwapchainSupportDetails::default();
            let surface = self.surface.ok_or(DeviceError::SurfaceDependencyMissing)?;

            details.capabilities = self.surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)?;

            details.formats = self.surface_loader
                .get_physical_device_surface_formats(physical_device, surface)?;

            details.present_modes = self.surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface)?;

            Ok(details)
        }
    }
    fn choose_swap_extent(&self, capabilities: &SurfaceCapabilitiesKHR, window: &Window) -> Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            let (width, height) = window.inner_size().into();
            Extent2D {
                width: u32::clamp(width, capabilities.min_image_extent.width, capabilities.max_image_extent.width),
                height: u32::clamp(height, capabilities.min_image_extent.height, capabilities.max_image_extent.height),
            }
        }
    }
    fn choose_swap_surface_format(&self, available_formats: &Vec<SurfaceFormatKHR>) -> SurfaceFormatKHR {
        for available_format in available_formats.iter() {
            if available_format.format == vk::Format::B8G8R8A8_SRGB &&
               available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
                return *available_format;
            }
        }
        available_formats[0]
    }
    fn choose_swap_present_mode(&self, available_present_modes: &Vec<PresentModeKHR>) -> PresentModeKHR {
        for &available_present_mode in available_present_modes.iter() {
            if available_present_mode == PresentModeKHR::MAILBOX {
                return available_present_mode;
            }
        }
        PresentModeKHR::FIFO
    }
    pub unsafe fn get_physical_device_surface_support(
        &self,
        physical_device: PhysicalDevice,
        queue_family_index: u32,
    ) -> Result<bool, DeviceError> {
        unsafe {
            let surface = self.surface.ok_or(DeviceError::SurfaceDependencyMissing)?;
            self.surface_loader.get_physical_device_surface_support(physical_device, queue_family_index, surface)
                .map_err(|e| DeviceError::Vulkan(e))
        }
    }
}