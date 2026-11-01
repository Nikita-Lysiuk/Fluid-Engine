use crate::render::graphics_pipelines::Pipelines;
use crate::render::model_vertex::ModelVertex;
use image::GenericImageView;
use std::f32::consts::PI;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
    PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{DeviceOwned, Queue};
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::{PipelineBindPoint, PipelineLayout};
use vulkano::sync::GpuFuture;

/// Resources used to render the environment.
pub struct SkyData {
    pub vertex_buffer: Subbuffer<[ModelVertex]>,
    pub index_buffer: Subbuffer<[u32]>,
    pub index_count: u32,

    pub descriptor_set: Arc<DescriptorSet>,

    pub texture_view: Arc<ImageView>,
}

impl SkyData {
    /// Creates the associated resources.
    pub fn new(
        radius: f32,
        sectors: u16,
        stacks: u16,
        memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        sky_layout: Arc<PipelineLayout>,
        queue: Arc<Queue>,
        path_to_hdri: &str,
    ) -> Self {
        let (vertices, indices) = Self::generate_uv_sphere(radius, sectors, stacks);

        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..BufferCreateInfo::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..AllocationCreateInfo::default()
            },
            vertices,
        )
        .unwrap_or_else(|e| panic!("[SkySphere] Failed to create vertex buffer:\n{:?}", e));

        let index_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..BufferCreateInfo::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..AllocationCreateInfo::default()
            },
            indices.clone(),
        )
        .unwrap_or_else(|e| panic!("[SkySphere] Failed to create index buffer:\n{:?}", e));

        let (descriptor_set, texture_view) = Self::load_hdri_texture(
            path_to_hdri,
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
            sky_layout.clone(),
            queue.clone(),
        );

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            descriptor_set,
            texture_view,
        }
    }
    fn generate_uv_sphere(radius: f32, sectors: u16, stacks: u16) -> (Vec<ModelVertex>, Vec<u32>) {
        let mut vertices: Vec<ModelVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for i in 0..=stacks {
            let stack_angle = PI / 2.0 - (i as f32) * (PI / stacks as f32);
            let y = radius * stack_angle.sin();
            let xz = radius * stack_angle.cos();

            for j in 0..=sectors {
                let sector_angle = (j as f32) * (2.0 * PI / sectors as f32);
                let x = xz * sector_angle.cos();
                let z = xz * sector_angle.sin();

                vertices.push(ModelVertex {
                    position: [x, y, z],
                });
            }
        }

        for i in 0..stacks {
            let mut k1 = i * (sectors + 1);
            let mut k2 = k1 + sectors + 1;
            for _ in 0..sectors {
                if i != 0 {
                    indices.push(k1 as u32);
                    indices.push(k2 as u32);
                    indices.push((k1 + 1) as u32);
                }
                if i != (stacks - 1) {
                    indices.push((k1 + 1) as u32);
                    indices.push(k2 as u32);
                    indices.push((k2 + 1) as u32);
                }
                k1 += 1;
                k2 += 1;
            }
        }

        (vertices, indices)
    }
    fn load_hdri_texture(
        path: &str,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_allocator: Arc<StandardDescriptorSetAllocator>,
        sky_layout: Arc<PipelineLayout>,
        queue: Arc<Queue>,
    ) -> (Arc<DescriptorSet>, Arc<ImageView>) {
        let img = image::open(path).unwrap_or_else(|e| {
            panic!(
                "[SkySphere] Failed to load HDRI image from path {}:\n{:?}",
                path, e
            )
        });
        let dims = img.dimensions();
        let rgba_data = img.to_rgba32f().into_raw();

        let texture = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R32G32B32A32_SFLOAT,
                extent: [dims.0, dims.1, 1],
                usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
                ..ImageCreateInfo::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap_or_else(|e| panic!("[SkySphere] Failed to create HDRI image:\n{:?}", e));

        let staging_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..BufferCreateInfo::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..AllocationCreateInfo::default()
            },
            rgba_data,
        )
        .unwrap_or_else(|e| {
            panic!(
                "[SkySphere] Failed to create staging buffer for HDRI texture:\n{:?}",
                e
            )
        });

        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap_or_else(|e| {
            panic!(
                "[SkySphere] Failed to create command buffer builder:\n{:?}",
                e
            )
        });

        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                staging_buffer,
                texture.clone(),
            ))
            .unwrap_or_else(|e| {
                panic!(
                    "[SkySphere] Failed to record buffer to image copy command:\n{:?}",
                    e
                )
            });

        let command_buffer = builder
            .build()
            .unwrap_or_else(|e| panic!("[SkySphere] Failed to build command buffer:\n{:?}", e));

        command_buffer
            .execute(queue.clone())
            .unwrap_or_else(|e| panic!("[SkySphere] Failed to execute command buffer:\n{:?}", e))
            .then_signal_fence_and_flush()
            .unwrap_or_else(|e| panic!("[SkySphere] Failed to flush command buffer:\n{:?}", e))
            .wait(None)
            .unwrap_or_else(|e| {
                panic!(
                    "[SkySphere] Failed to wait for command buffer execution:\n{:?}",
                    e
                )
            });

        let texture_view = ImageView::new_default(texture).unwrap_or_else(|e| {
            panic!(
                "[SkySphere] Failed to create image view for HDRI texture:\n{:?}",
                e
            )
        });

        let sampler = Sampler::new(
            memory_allocator.device().clone(),
            SamplerCreateInfo::simple_repeat_linear_no_mipmap(),
        )
        .expect("sampler can be created");

        let set_layout = sky_layout
            .set_layouts()
            .get(0)
            .expect("descriptor set layout 0 exists");
        let descriptor_set = DescriptorSet::new(
            descriptor_allocator,
            set_layout.clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                texture_view.clone(),
                sampler,
            )],
            [],
        )
        .expect("descriptor set bindings match the shader");

        (descriptor_set, texture_view)
    }

    /// Records the graphics commands for this resource.
    pub fn bind_to_command_buffer<Cb>(
        &self,
        builder: &mut AutoCommandBufferBuilder<Cb>,
        pipelines: &Pipelines,
        camera_addr: u64,
    ) {
        // SAFETY: the pipeline, required descriptors, vertex buffers, and push constants are bound
        //         before drawing.
        unsafe {
            builder
                .bind_pipeline_graphics(pipelines.sky_pipeline.inner.clone())
                .unwrap_or_else(|e| panic!("[Renderer] Failed to bind sky pipeline: {:?}", e))
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipelines.sky_layout.clone(),
                    0,
                    self.descriptor_set.clone(),
                )
                .unwrap_or_else(|e| panic!("[Renderer] Failed to bind sky descriptor set: {:?}", e))
                .push_constants(
                    pipelines.sky_layout.clone(),
                    0,
                    [
                        camera_addr,
                        self.vertex_buffer
                            .device_address()
                            .unwrap_or_else(|e| {
                                panic!("[Renderer] Failed to get vertex_buffer: {:?}", e)
                            })
                            .get(),
                    ],
                )
                .unwrap_or_else(|e| panic!("[Renderer] Failed to bind sky buffers: {:?}", e))
                .bind_index_buffer(self.index_buffer.clone())
                .unwrap_or_else(|e| panic!("[Renderer] Failed to bind sky index buffer: {:?}", e))
                .draw_indexed(self.index_count, 1, 0, 0, 0)
                .unwrap_or_else(|e| panic!("[Renderer] Failed to draw sky sphere: {:?}", e));
        }
    }
}
