//! Offline rendering of the README animation.
//!
//! Runs `Scene::showcase` without a window, at a fixed rate of simulated time
//! per frame (not tied to how long a frame takes), draws each frame in
//! particle mode into an offscreen image and saves it as a PNG. A GIF is made
//! from the PNGs with ffmpeg, see `scripts/showcase_gif.sh`.

use crate::render::frame_resources::GpuSceneResources;
use crate::render::gpu_context::vulkano_config;
use crate::render::graphics_pipelines::Pipelines;
use crate::render::particle_renderer::GpuRenderData;
use crate::render::sky::SkyData;
use crate::scene::Scene;
use crate::sim::pipelines::{ComputePipelines, SortAlgorithm};
use crate::sim::simulation::{Simulation, SimulationSettings};
use crate::sim::simulation_resources::SimulationResources;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, CopyImageToBufferInfo,
    RenderingAttachmentInfo, RenderingInfo,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::sync::{self, GpuFuture};
use vulkano_util::context::VulkanoContext;

/// Same layout as the window's swapchain, so colours match the application.
const COLOR_FORMAT: Format = Format::B8G8R8A8_UNORM;
const DEPTH_FORMAT: Format = Format::D32_SFLOAT;
/// Particles are drawn this much larger than their physical radius so that
/// the small particles of the showcase scene read as a continuous surface.
const DRAW_RADIUS_SCALE: f32 = 1.8;

/// What to render and where.
pub struct ShowcaseOptions {
    pub output_dir: PathBuf,
    pub width: u32,
    pub height: u32,
    /// Frames per second of simulated time.
    pub fps: u32,
    /// Simulated time before the first saved frame: the water settles with
    /// the wall at rest, then the wall starts and the waves build up.
    pub settle_seconds: f32,
    pub wave_warmup_seconds: f32,
    /// Length of the saved animation; a whole number of wall periods makes
    /// the GIF loop smoothly.
    pub seconds: f32,
}

/// Simulates and renders the showcase scene, writing `frame_0000.png`, ...
/// into `options.output_dir`.
pub fn render(options: &ShowcaseOptions) {
    std::fs::create_dir_all(&options.output_dir).expect("output directory can be created");

    let context = Arc::new(VulkanoContext::new(vulkano_config()));
    let device = context.device().clone();
    let queue = context.graphics_queue().clone();
    let memory_allocator = context.memory_allocator().clone();
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        StandardDescriptorSetAllocatorCreateInfo::default(),
    ));

    let mut scene = Scene::showcase();
    scene
        .camera
        .aspect_ratio(options.width as f32 / options.height as f32);
    scene.boundary.animate = false;
    println!("showcase: {} particles", scene.initial_positions.len());

    let pipelines = Pipelines::new(context.clone(), COLOR_FORMAT, DEPTH_FORMAT);
    let resources = GpuSceneResources::new(memory_allocator.clone(), &scene);
    let simulation_resources = SimulationResources::new(memory_allocator.clone(), &scene);
    let compute = ComputePipelines::new(
        device.clone(),
        memory_allocator.clone(),
        descriptor_set_allocator.clone(),
        &simulation_resources.physics_data,
        &simulation_resources.sim_params_buffer,
    );
    let mut simulation = Simulation::new(
        compute,
        simulation_resources,
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
    );
    let sky = SkyData::new(
        500.0,
        64,
        64,
        memory_allocator.clone(),
        descriptor_set_allocator,
        command_buffer_allocator.clone(),
        pipelines.sky_layout.clone(),
        queue.clone(),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/hdri/citrus_orchard_road_puresky_4k.exr"
        ),
    );

    let water = GpuRenderData::new(
        memory_allocator.clone(),
        &scene.initial_positions,
        DRAW_RADIUS_SCALE * scene.sim_params.particle_radius,
    );
    let target = Target::new(&context, options.width, options.height);
    let settings = SimulationSettings {
        // Splashes against the pillar are fast; the CFL step keeps them
        // stable. Rendering offline, the extra substeps only cost time.
        use_cfl: true,
        cfl_factor: 0.5,
        cfl_min_dt: 0.0001,
        cfl_max_dt: scene.sim_params.dt,
        // Enough to cover a whole frame of simulated time; nothing waits.
        max_substeps: 64,
        sort_algorithm: SortAlgorithm::Bitonic,
        use_solver_error_threshold: true,
        density_error_pct: 0.1,
        divergence_error_pct: 0.1,
    };

    let frame_dt = 1.0 / options.fps as f32;
    let frames = |seconds: f32| (seconds * options.fps as f32).round() as u32;
    let wave_start = frames(options.settle_seconds);
    let record_start = wave_start + frames(options.wave_warmup_seconds);
    let total = record_start + frames(options.seconds);
    let particle_count = simulation.resources.physics_data.count;
    let started = Instant::now();

    for frame in 0..total {
        if frame == wave_start {
            scene.boundary.animate = true;
        }
        let positions = simulation.resources.physics_data.position_a.clone();
        let colors = simulation.resources.physics_data.colors.clone();
        let render_positions = water.position_buffers[0].clone();
        let render_colors = water.color_buffers[0].clone();
        let (future, _) = simulation.step(
            &mut scene,
            frame_dt,
            &settings,
            frame as usize % crate::constants::MAX_FRAMES_IN_FLIGHT,
            queue.clone(),
            sync::now(device.clone()).boxed(),
            |scene| resources.sync_with_scene(scene),
            |builder| {
                builder
                    .copy_buffer(CopyBufferInfo::buffers(positions, render_positions))
                    .expect("positions can be copied for drawing");
                builder
                    .copy_buffer(CopyBufferInfo::buffers(colors, render_colors))
                    .expect("colours can be copied for drawing");
            },
        );

        // Every frame ends with a command buffer, even an empty one: a fence
        // signalled directly after the simulation's semaphore never fired.
        let recording = frame >= record_start;
        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("command buffer can be allocated");
        if recording {
            target.begin(&mut builder);
            sky.bind_to_command_buffer(&mut builder, &pipelines, resources.camera_addr());
            water.bind_to_command_buffer(
                &mut builder,
                &pipelines,
                resources.camera_addr(),
                0,
                particle_count,
            );
            resources.draw_pillar(&mut builder, &pipelines, &scene);
            resources.bind_to_command_buffer(&mut builder, &pipelines);
            target.end_and_read_back(&mut builder);
        }
        future
            .then_execute(
                queue.clone(),
                builder.build().expect("render commands are valid"),
            )
            .expect("rendering can be submitted")
            .then_signal_fence_and_flush()
            .expect("rendering can be flushed")
            .wait(None)
            .expect("rendering completes");
        if !recording {
            continue;
        }

        let index = frame - record_start;
        target.save_png(&options.output_dir.join(format!("frame_{index:04}.png")));
        if index % 25 == 0 {
            println!(
                "frame {index}/{} ({:.0} s elapsed)",
                total - record_start,
                started.elapsed().as_secs_f32()
            );
        }
    }
    println!("done in {:.0} s", started.elapsed().as_secs_f32());
}

/// Offscreen colour and depth images and the buffer the colour is read into.
struct Target {
    width: u32,
    height: u32,
    color: Arc<Image>,
    color_view: Arc<ImageView>,
    depth_view: Arc<ImageView>,
    readback: Subbuffer<[u8]>,
}

impl Target {
    fn new(context: &VulkanoContext, width: u32, height: u32) -> Self {
        let allocator = context.memory_allocator().clone();
        let image = |format, usage| {
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    format,
                    extent: [width, height, 1],
                    usage,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .expect("offscreen image can be created")
        };
        let color = image(
            COLOR_FORMAT,
            ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
        );
        let depth = image(DEPTH_FORMAT, ImageUsage::DEPTH_STENCIL_ATTACHMENT);
        let readback = Buffer::new_slice::<u8>(
            allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            u64::from(width) * u64::from(height) * 4,
        )
        .expect("readback buffer can be created");
        Self {
            width,
            height,
            color_view: ImageView::new_default(color.clone()).expect("image view can be created"),
            depth_view: ImageView::new_default(depth).expect("image view can be created"),
            color,
            readback,
        }
    }

    /// Starts rendering into the offscreen images with the same flipped
    /// viewport as the window renderer.
    fn begin<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        let (width, height) = (self.width as f32, self.height as f32);
        builder
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                    ..RenderingAttachmentInfo::image_view(self.color_view.clone())
                })],
                depth_attachment: Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::DontCare,
                    clear_value: Some(1f32.into()),
                    ..RenderingAttachmentInfo::image_view(self.depth_view.clone())
                }),
                ..RenderingInfo::default()
            })
            .expect("rendering can begin")
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, height],
                    extent: [width, -height],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .expect("viewport is valid")
            .set_scissor(
                0,
                [Scissor {
                    offset: [0, 0],
                    extent: [self.width, self.height],
                }]
                .into_iter()
                .collect(),
            )
            .expect("scissor is valid");
    }

    fn end_and_read_back<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        builder.end_rendering().expect("rendering can end");
        builder
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                self.color.clone(),
                self.readback.clone(),
            ))
            .expect("frame can be copied to the readback buffer");
    }

    fn save_png(&self, path: &Path) {
        let bgra = self.readback.read().expect("frame has finished rendering");
        let mut rgba = bgra.to_vec();
        for pixel in rgba.chunks_exact_mut(4) {
            pixel.swap(0, 2);
            pixel[3] = 255;
        }
        image::save_buffer(
            path,
            &rgba,
            self.width,
            self.height,
            image::ColorType::Rgba8,
        )
        .expect("frame can be written");
    }
}
