use crate::shader_loader::load_shader_entry_point;
use crate::sim::particle_buffers::Entry;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::instance::debug::DebugUtilsLabel;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};

mod cs_count {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\radix_count.comp");
}
mod cs_scan {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\radix_scan.comp");
}
mod cs_reorder {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\radix_reorder.comp");
}

pub(super) const ELEMENTS_PER_INVOCATION: u32 = 8;

/// GPU radix sorter for particle hashes.
pub struct RadixSorter {
    entries_tmp: Subbuffer<[Entry]>,
    counts: Subbuffer<[u32]>,

    count_pipeline: Arc<ComputePipeline>,
    scan_pipeline: Arc<ComputePipeline>,
    reorder_pipeline: Arc<ComputePipeline>,

    even_count_set: Option<Arc<DescriptorSet>>,
    odd_count_set: Option<Arc<DescriptorSet>>,
    scan_set: Option<Arc<DescriptorSet>>,
    even_reorder_set: Option<Arc<DescriptorSet>>,
    odd_reorder_set: Option<Arc<DescriptorSet>>,

    num_work_groups: u32,
    num_elements: u32,
}

impl RadixSorter {
    /// Creates the associated resources.
    pub fn new(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        sort_buffer_size: u32,
    ) -> Self {
        let num_work_groups = (sort_buffer_size + 256 * ELEMENTS_PER_INVOCATION - 1)
            / (256 * ELEMENTS_PER_INVOCATION);

        let entries_tmp = Buffer::new_slice::<Entry>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            sort_buffer_size as u64,
        )
        .expect("buffer can be created");

        let counts = Buffer::new_slice::<u32>(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            (num_work_groups * 256) as u64,
        )
        .expect("buffer can be created");

        let count_shader = load_shader_entry_point(device.clone(), cs_count::load, "main");
        let count_stage = PipelineShaderStageCreateInfo::new(count_shader);
        let count_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&count_stage])
                .into_pipeline_layout_create_info(device.clone())
                .expect("shader pipeline layout is valid"),
        )
        .expect("pipeline layout can be created");
        let count_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(count_stage, count_layout),
        )
        .expect("compute pipeline can be created");

        let scan_shader = load_shader_entry_point(device.clone(), cs_scan::load, "main");
        let scan_stage = PipelineShaderStageCreateInfo::new(scan_shader);
        let scan_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&scan_stage])
                .into_pipeline_layout_create_info(device.clone())
                .expect("shader pipeline layout is valid"),
        )
        .expect("pipeline layout can be created");
        let scan_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(scan_stage, scan_layout),
        )
        .expect("compute pipeline can be created");

        let reorder_shader = load_shader_entry_point(device.clone(), cs_reorder::load, "main");
        let reorder_stage = PipelineShaderStageCreateInfo::new(reorder_shader);
        let reorder_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&reorder_stage])
                .into_pipeline_layout_create_info(device.clone())
                .expect("shader pipeline layout is valid"),
        )
        .expect("pipeline layout can be created");
        let reorder_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(reorder_stage, reorder_layout),
        )
        .expect("compute pipeline can be created");

        Self {
            entries_tmp,
            counts,
            count_pipeline,
            scan_pipeline,
            reorder_pipeline,
            even_count_set: None,
            odd_count_set: None,
            scan_set: None,
            even_reorder_set: None,
            odd_reorder_set: None,
            num_work_groups,
            num_elements: sort_buffer_size,
        }
    }

    /// Prepares GPU resources for subsequent commands.
    pub fn prepare(
        &mut self,
        allocator: Arc<StandardDescriptorSetAllocator>,
        grid_entries: &Subbuffer<[Entry]>,
    ) {
        let count_layout = self
            .count_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .expect("descriptor set layout 0 exists");
        self.even_count_set = Some(
            DescriptorSet::new(
                allocator.clone(),
                count_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, grid_entries.clone()),
                    WriteDescriptorSet::buffer(1, self.counts.clone()),
                ],
                [],
            )
            .expect("descriptor set bindings match the shader"),
        );
        self.odd_count_set = Some(
            DescriptorSet::new(
                allocator.clone(),
                count_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, self.entries_tmp.clone()),
                    WriteDescriptorSet::buffer(1, self.counts.clone()),
                ],
                [],
            )
            .expect("descriptor set bindings match the shader"),
        );

        let scan_layout = self
            .scan_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .expect("descriptor set layout 0 exists");
        self.scan_set = Some(
            DescriptorSet::new(
                allocator.clone(),
                scan_layout.clone(),
                [WriteDescriptorSet::buffer(0, self.counts.clone())],
                [],
            )
            .expect("descriptor set bindings match the shader"),
        );

        let reorder_layout = self
            .reorder_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .expect("descriptor set layout 0 exists");
        self.even_reorder_set = Some(
            DescriptorSet::new(
                allocator.clone(),
                reorder_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, grid_entries.clone()),
                    WriteDescriptorSet::buffer(1, self.entries_tmp.clone()),
                    WriteDescriptorSet::buffer(2, self.counts.clone()),
                ],
                [],
            )
            .expect("descriptor set bindings match the shader"),
        );
        self.odd_reorder_set = Some(
            DescriptorSet::new(
                allocator,
                reorder_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, self.entries_tmp.clone()),
                    WriteDescriptorSet::buffer(1, grid_entries.clone()),
                    WriteDescriptorSet::buffer(2, self.counts.clone()),
                ],
                [],
            )
            .expect("descriptor set bindings match the shader"),
        );
    }

    /// Records the associated GPU commands.
    pub fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        let num_work_groups = self.num_work_groups;
        let num_elements = self.num_elements;

        for pass in 0u32..4 {
            builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: format!("Radix Sort Pass {pass}").into(),
                    color: [0.0, 1.0, 0.2, 1.0],
                    ..Default::default()
                })
                .expect("debug label can be opened");

            let shift = pass * 8;
            let is_even = pass % 2 == 0;

            // ── Count ────────────────────────────────────────────────────────
            builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: format!("Radix Count (shift={shift})").into(),
                    color: [0.2, 0.4, 1.0, 1.0],
                    ..Default::default()
                })
                .expect("debug label can be opened");
            let count_set = if is_even {
                self.even_count_set
                    .as_ref()
                    .expect("prepare() was called before recording")
            } else {
                self.odd_count_set
                    .as_ref()
                    .expect("prepare() was called before recording")
            };
            builder
                .bind_pipeline_compute(self.count_pipeline.clone())
                .expect("compute pipeline can be bound")
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.count_pipeline.layout().clone(),
                    0,
                    count_set.clone(),
                )
                .expect("descriptor set can be bound")
                .push_constants(
                    self.count_pipeline.layout().clone(),
                    0,
                    cs_count::CountConstant {
                        shift,
                        num_elements,
                        num_workgroups: num_work_groups,
                        elements_per_invocation: ELEMENTS_PER_INVOCATION,
                    },
                )
                .expect("push constants match the shader");
            // SAFETY: the compute pipeline, descriptor set, and push constants are bound above; the
            //         debug label is open.
            unsafe {
                builder
                    .dispatch([num_work_groups, 1, 1])
                    .expect("compute dispatch is valid");
                builder
                    .end_debug_utils_label()
                    .expect("debug label can be closed");
            }

            // ── Scan ─────────────────────────────────────────────────────────
            builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: "Radix Scan".into(),
                    color: [1.0, 0.8, 0.0, 1.0],
                    ..Default::default()
                })
                .expect("debug label can be opened");
            builder
                .bind_pipeline_compute(self.scan_pipeline.clone())
                .expect("compute pipeline can be bound")
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.scan_pipeline.layout().clone(),
                    0,
                    self.scan_set
                        .as_ref()
                        .expect("prepare() was called before recording")
                        .clone(),
                )
                .expect("descriptor set can be bound")
                .push_constants(
                    self.scan_pipeline.layout().clone(),
                    0,
                    cs_scan::ScanConstant {
                        num_workgroups: num_work_groups,
                    },
                )
                .expect("push constants match the shader");
            // SAFETY: the compute pipeline, descriptor set, and push constants are bound above; the
            //         debug label is open.
            unsafe {
                builder
                    .dispatch([1, 1, 1])
                    .expect("compute dispatch is valid");
                builder
                    .end_debug_utils_label()
                    .expect("debug label can be closed");
            }

            // ── Reorder ──────────────────────────────────────────────────────
            builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: format!("Radix Reorder (shift={shift})").into(),
                    color: [1.0, 0.2, 0.2, 1.0],
                    ..Default::default()
                })
                .expect("debug label can be opened");
            let reorder_set = if is_even {
                self.even_reorder_set
                    .as_ref()
                    .expect("prepare() was called before recording")
            } else {
                self.odd_reorder_set
                    .as_ref()
                    .expect("prepare() was called before recording")
            };
            builder
                .bind_pipeline_compute(self.reorder_pipeline.clone())
                .expect("compute pipeline can be bound")
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.reorder_pipeline.layout().clone(),
                    0,
                    reorder_set.clone(),
                )
                .expect("descriptor set can be bound")
                .push_constants(
                    self.reorder_pipeline.layout().clone(),
                    0,
                    cs_reorder::ReorderConstant {
                        shift,
                        num_elements,
                        num_workgroups: num_work_groups,
                        elements_per_invocation: ELEMENTS_PER_INVOCATION,
                    },
                )
                .expect("push constants match the shader");
            // SAFETY: the compute pipeline, descriptor set, and push constants are bound above; the
            //         debug label is open.
            unsafe {
                builder
                    .dispatch([num_work_groups, 1, 1])
                    .expect("compute dispatch is valid");
                builder
                    .end_debug_utils_label()
                    .expect("debug label can be closed");
            }

            // SAFETY: the enclosing radix pass debug label was opened above.
            unsafe {
                builder
                    .end_debug_utils_label()
                    .expect("debug label can be closed");
            }
        }
    }
}
