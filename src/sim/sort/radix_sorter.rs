use crate::entities::particle::Entry;
use crate::utils::shader_loader::load_shader_entry_point;
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
        .unwrap();

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
        .unwrap();

        let count_shader = load_shader_entry_point(device.clone(), cs_count::load, "main");
        let count_stage = PipelineShaderStageCreateInfo::new(count_shader);
        let count_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&count_stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let count_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(count_stage, count_layout),
        )
        .unwrap();

        let scan_shader = load_shader_entry_point(device.clone(), cs_scan::load, "main");
        let scan_stage = PipelineShaderStageCreateInfo::new(scan_shader);
        let scan_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&scan_stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let scan_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(scan_stage, scan_layout),
        )
        .unwrap();

        let reorder_shader = load_shader_entry_point(device.clone(), cs_reorder::load, "main");
        let reorder_stage = PipelineShaderStageCreateInfo::new(reorder_shader);
        let reorder_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&reorder_stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let reorder_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(reorder_stage, reorder_layout),
        )
        .unwrap();

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

    pub fn prepare(
        &mut self,
        allocator: Arc<StandardDescriptorSetAllocator>,
        grid_entries: &Subbuffer<[Entry]>,
    ) {
        let count_layout = self.count_pipeline.layout().set_layouts().get(0).unwrap();
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
            .unwrap(),
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
            .unwrap(),
        );

        let scan_layout = self.scan_pipeline.layout().set_layouts().get(0).unwrap();
        self.scan_set = Some(
            DescriptorSet::new(
                allocator.clone(),
                scan_layout.clone(),
                [WriteDescriptorSet::buffer(0, self.counts.clone())],
                [],
            )
            .unwrap(),
        );

        let reorder_layout = self.reorder_pipeline.layout().set_layouts().get(0).unwrap();
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
            .unwrap(),
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
            .unwrap(),
        );
    }

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
                .unwrap();

            let shift = pass * 8;
            let is_even = pass % 2 == 0;

            // ── Count ────────────────────────────────────────────────────────
            builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: format!("Radix Count (shift={shift})").into(),
                    color: [0.2, 0.4, 1.0, 1.0],
                    ..Default::default()
                })
                .unwrap();
            let count_set = if is_even {
                self.even_count_set.as_ref().unwrap()
            } else {
                self.odd_count_set.as_ref().unwrap()
            };
            builder
                .bind_pipeline_compute(self.count_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.count_pipeline.layout().clone(),
                    0,
                    count_set.clone(),
                )
                .unwrap()
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
                .unwrap();
            unsafe {
                builder.dispatch([num_work_groups, 1, 1]).unwrap();
                builder.end_debug_utils_label().unwrap();
            }

            // ── Scan ─────────────────────────────────────────────────────────
            builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: "Radix Scan".into(),
                    color: [1.0, 0.8, 0.0, 1.0],
                    ..Default::default()
                })
                .unwrap();
            builder
                .bind_pipeline_compute(self.scan_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.scan_pipeline.layout().clone(),
                    0,
                    self.scan_set.as_ref().unwrap().clone(),
                )
                .unwrap()
                .push_constants(
                    self.scan_pipeline.layout().clone(),
                    0,
                    cs_scan::ScanConstant {
                        num_workgroups: num_work_groups,
                    },
                )
                .unwrap();
            unsafe {
                builder.dispatch([1, 1, 1]).unwrap();
                builder.end_debug_utils_label().unwrap();
            }

            // ── Reorder ──────────────────────────────────────────────────────
            builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: format!("Radix Reorder (shift={shift})").into(),
                    color: [1.0, 0.2, 0.2, 1.0],
                    ..Default::default()
                })
                .unwrap();
            let reorder_set = if is_even {
                self.even_reorder_set.as_ref().unwrap()
            } else {
                self.odd_reorder_set.as_ref().unwrap()
            };
            builder
                .bind_pipeline_compute(self.reorder_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.reorder_pipeline.layout().clone(),
                    0,
                    reorder_set.clone(),
                )
                .unwrap()
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
                .unwrap();
            unsafe {
                builder.dispatch([num_work_groups, 1, 1]).unwrap();
                builder.end_debug_utils_label().unwrap();
            }

            unsafe {
                builder.end_debug_utils_label().unwrap();
            }
        }
    }
}

