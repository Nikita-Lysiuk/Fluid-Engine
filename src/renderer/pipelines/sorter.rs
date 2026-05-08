use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::instance::debug::DebugUtilsLabel;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::shader::EntryPoint;
use crate::entities::particle::{Entry, GpuPhysicsData, SimulationParams};
use crate::renderer::pipelines::ComputeStep;
use crate::utils::shader_loader::load_shader_entry_point;

#[derive(PartialEq, Clone, Copy)]
pub enum SortAlgorithm {
    Bitonic,
    Radix,
}

// ── Bitonic sort ─────────────────────────────────────────────────────────────

mod cs {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\bitonic_sort.comp");
}

pub struct GpuSorter {
    pipeline: Arc<ComputePipeline>,
    descriptor_set: Option<Arc<DescriptorSet>>,
    num_elements: u32,
}

impl ComputeStep for GpuSorter {
    fn load_shader_module(device: Arc<Device>) -> EntryPoint {
        load_shader_entry_point(device, cs::load, "main")
    }
    fn from_pipeline(pipeline: Arc<ComputePipeline>) -> Self {
        Self { pipeline, descriptor_set: None, num_elements: 0 }
    }
    fn prepare(
        &mut self,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &GpuPhysicsData,
        _sim_params: &Subbuffer<SimulationParams>,
    ) {
        let grid_entries = &physics_data.grid_entries;
        self.num_elements = grid_entries.len() as u32;
        assert!(self.num_elements.is_power_of_two(), "GpuSorter: Buffer len must be power of 2");

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        self.descriptor_set = Some(DescriptorSet::new(
            allocator,
            layout.clone(),
            [WriteDescriptorSet::buffer(0, grid_entries.clone())],
            []
        ).unwrap());
    }
    fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        let set = self.descriptor_set.as_ref().expect("GpuSorter: call prepare() before execute()");
        let num_elements = self.num_elements;

        let mut h = 2;
        while h <= num_elements {
            let mut step = h / 2;
            while step > 0 {
                let pc = cs::SortConstants {
                    num_entries: num_elements,
                    block_height: h,
                    block_step: step,
                };

                builder
                    .bind_pipeline_compute(self.pipeline.clone()).unwrap()
                    .bind_descriptor_sets(PipelineBindPoint::Compute, self.pipeline.layout().clone(), 0, set.clone()).unwrap()
                    .push_constants(self.pipeline.layout().clone(), 0, pc).unwrap();

                let threads_needed = num_elements / 2;
                let group_size = 256;
                let dispatch_count = (threads_needed + group_size - 1) / group_size;

                unsafe { builder.dispatch([dispatch_count, 1, 1]).unwrap(); }

                step /= 2;
            }
            h *= 2;
        }
    }
}

// ── Radix sort ────────────────────────────────────────────────────────────────

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

const ELEMENTS_PER_INVOCATION: u32 = 8;

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
        let num_work_groups = (sort_buffer_size + 256 * ELEMENTS_PER_INVOCATION - 1) / (256 * ELEMENTS_PER_INVOCATION);

        let entries_tmp = Buffer::new_slice::<Entry>(
            memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, ..Default::default() },
            sort_buffer_size as u64,
        ).unwrap();

        let counts = Buffer::new_slice::<u32>(
            memory_allocator,
            BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, ..Default::default() },
            (num_work_groups * 256) as u64,
        ).unwrap();

        let count_shader = load_shader_entry_point(device.clone(), cs_count::load, "main");
        let count_stage = PipelineShaderStageCreateInfo::new(count_shader);
        let count_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&count_stage])
                .into_pipeline_layout_create_info(device.clone()).unwrap()
        ).unwrap();
        let count_pipeline = ComputePipeline::new(
            device.clone(), None, ComputePipelineCreateInfo::stage_layout(count_stage, count_layout)
        ).unwrap();

        let scan_shader = load_shader_entry_point(device.clone(), cs_scan::load, "main");
        let scan_stage = PipelineShaderStageCreateInfo::new(scan_shader);
        let scan_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&scan_stage])
                .into_pipeline_layout_create_info(device.clone()).unwrap()
        ).unwrap();
        let scan_pipeline = ComputePipeline::new(
            device.clone(), None, ComputePipelineCreateInfo::stage_layout(scan_stage, scan_layout)
        ).unwrap();

        let reorder_shader = load_shader_entry_point(device.clone(), cs_reorder::load, "main");
        let reorder_stage = PipelineShaderStageCreateInfo::new(reorder_shader);
        let reorder_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&reorder_stage])
                .into_pipeline_layout_create_info(device.clone()).unwrap()
        ).unwrap();
        let reorder_pipeline = ComputePipeline::new(
            device.clone(), None, ComputePipelineCreateInfo::stage_layout(reorder_stage, reorder_layout)
        ).unwrap();

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
        self.even_count_set = Some(DescriptorSet::new(
            allocator.clone(), count_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, grid_entries.clone()),
                WriteDescriptorSet::buffer(1, self.counts.clone()),
            ],
            [],
        ).unwrap());
        self.odd_count_set = Some(DescriptorSet::new(
            allocator.clone(), count_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.entries_tmp.clone()),
                WriteDescriptorSet::buffer(1, self.counts.clone()),
            ],
            [],
        ).unwrap());

        let scan_layout = self.scan_pipeline.layout().set_layouts().get(0).unwrap();
        self.scan_set = Some(DescriptorSet::new(
            allocator.clone(), scan_layout.clone(),
            [WriteDescriptorSet::buffer(0, self.counts.clone())],
            [],
        ).unwrap());

        let reorder_layout = self.reorder_pipeline.layout().set_layouts().get(0).unwrap();
        self.even_reorder_set = Some(DescriptorSet::new(
            allocator.clone(), reorder_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, grid_entries.clone()),
                WriteDescriptorSet::buffer(1, self.entries_tmp.clone()),
                WriteDescriptorSet::buffer(2, self.counts.clone()),
            ],
            [],
        ).unwrap());
        self.odd_reorder_set = Some(DescriptorSet::new(
            allocator, reorder_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.entries_tmp.clone()),
                WriteDescriptorSet::buffer(1, grid_entries.clone()),
                WriteDescriptorSet::buffer(2, self.counts.clone()),
            ],
            [],
        ).unwrap());
    }

    pub fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        let num_work_groups = self.num_work_groups;
        let num_elements = self.num_elements;

        for pass in 0u32..4 {
            builder.begin_debug_utils_label(DebugUtilsLabel {
                label_name: format!("Radix Sort Pass {pass}").into(),
                color: [0.0, 1.0, 0.2, 1.0],
                ..Default::default()
            }).unwrap();


            let shift = pass * 8;
            let is_even = pass % 2 == 0;

            // ── Count ────────────────────────────────────────────────────────
            builder.begin_debug_utils_label(DebugUtilsLabel {
                label_name: format!("Radix Count (shift={shift})").into(),
                color: [0.2, 0.4, 1.0, 1.0],
                ..Default::default()
            }).unwrap();
            let count_set = if is_even {
                self.even_count_set.as_ref().unwrap()
            } else {
                self.odd_count_set.as_ref().unwrap()
            };
            builder
                .bind_pipeline_compute(self.count_pipeline.clone()).unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, self.count_pipeline.layout().clone(), 0, count_set.clone()).unwrap()
                .push_constants(self.count_pipeline.layout().clone(), 0, cs_count::CountConstant {
                    shift,
                    num_elements,
                    num_workgroups: num_work_groups,
                    elements_per_invocation: ELEMENTS_PER_INVOCATION,
                }).unwrap();
            unsafe {
                builder.dispatch([num_work_groups, 1, 1]).unwrap();
                builder.end_debug_utils_label().unwrap();
            }


            // ── Scan ─────────────────────────────────────────────────────────
            builder.begin_debug_utils_label(DebugUtilsLabel {
                label_name: "Radix Scan".into(),
                color: [1.0, 0.8, 0.0, 1.0],
                ..Default::default()
            }).unwrap();
            builder
                .bind_pipeline_compute(self.scan_pipeline.clone()).unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, self.scan_pipeline.layout().clone(), 0, self.scan_set.as_ref().unwrap().clone()).unwrap()
                .push_constants(self.scan_pipeline.layout().clone(), 0, cs_scan::ScanConstant {
                    num_workgroups: num_work_groups,
                }).unwrap();
            unsafe { builder.dispatch([1, 1, 1]).unwrap(); builder.end_debug_utils_label().unwrap(); }


            // ── Reorder ──────────────────────────────────────────────────────
            builder.begin_debug_utils_label(DebugUtilsLabel {
                label_name: format!("Radix Reorder (shift={shift})").into(),
                color: [1.0, 0.2, 0.2, 1.0],
                ..Default::default()
            }).unwrap();
            let reorder_set = if is_even {
                self.even_reorder_set.as_ref().unwrap()
            } else {
                self.odd_reorder_set.as_ref().unwrap()
            };
            builder
                .bind_pipeline_compute(self.reorder_pipeline.clone()).unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, self.reorder_pipeline.layout().clone(), 0, reorder_set.clone()).unwrap()
                .push_constants(self.reorder_pipeline.layout().clone(), 0, cs_reorder::ReorderConstant {
                    shift,
                    num_elements,
                    num_workgroups: num_work_groups,
                    elements_per_invocation: ELEMENTS_PER_INVOCATION,
                }).unwrap();
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

#[cfg(test)]
mod tests {
    use crate::entities::particle::Entry;

    const WG_SIZE: usize = 256;
    const EPI: usize = super::ELEMENTS_PER_INVOCATION as usize;

    // CPU mirror of the three shader stages.
    fn cpu_radix_sort(data: &mut Vec<Entry>) {
        let n = data.len();
        let num_wg = (n + WG_SIZE * EPI - 1) / (WG_SIZE * EPI);

        let mut buf_a = data.clone();
        let mut buf_b = vec![Entry { hash: 0, index: 0 }; n];

        for pass in 0u32..4 {
            let shift = pass * 8;
            let is_even = pass % 2 == 0;

            let src = if is_even { buf_a.clone() } else { buf_b.clone() };
            let dst = if is_even { &mut buf_b } else { &mut buf_a };

            // ── Count (mirrors radix_count.comp) ─────────────────────────────
            let mut counts = vec![0u32; 256 * num_wg];
            for wg in 0..num_wg {
                let mut local = [0u32; 256];
                for lid in 0..WG_SIZE {
                    let base = wg * WG_SIZE * EPI + lid * EPI;
                    for i in 0..EPI {
                        let idx = base + i;
                        if idx >= n { break; }
                        let digit = ((src[idx].hash >> shift) & 0xFF) as usize;
                        local[digit] += 1;
                    }
                }
                for digit in 0..256 {
                    counts[digit * num_wg + wg] = local[digit];
                }
            }

            // ── Scan (mirrors radix_scan.comp) ───────────────────────────────
            // Phase 1: sequential exclusive scan within each digit's stripe
            let mut shared_sums = [0u32; 256];
            for lid in 0..256usize {
                let start = lid * num_wg;
                let end = start + num_wg;
                let mut running = 0u32;
                for i in start..end {
                    let val = counts[i];
                    counts[i] = running;
                    running += val;
                }
                shared_sums[lid] = running;
            }

            // Phase 2: Hillis-Steele inclusive prefix scan over shared_sums
            let mut stride = 1usize;
            while stride < 256 {
                let prev = shared_sums;
                for lid in stride..256 {
                    shared_sums[lid] += prev[lid - stride];
                }
                stride *= 2;
            }

            // Phase 3: add block offsets back
            for lid in 0..256usize {
                let block_offset = if lid == 0 { 0 } else { shared_sums[lid - 1] };
                let start = lid * num_wg;
                let end = start + num_wg;
                for i in start..end {
                    counts[i] += block_offset;
                }
            }

            // ── Reorder (mirrors radix_reorder.comp) ─────────────────────────
            for wg in 0..num_wg {
                for lid in 0..WG_SIZE {
                    let base = wg * WG_SIZE * EPI + lid * EPI;
                    for i in 0..EPI {
                        let idx = base + i;
                        if idx >= n { break; }
                        let e = src[idx];
                        let digit = ((e.hash >> shift) & 0xFF) as usize;
                        let dest = counts[digit * num_wg + wg] as usize;
                        counts[digit * num_wg + wg] += 1;
                        dst[dest] = e;
                    }
                }
            }
        }

        // After pass 3 (odd) result lands in buf_a, same as GPU where result lands in grid_entries.
        *data = buf_a;
    }

    fn padded_entries(hashes: &[u32]) -> Vec<Entry> {
        let mut v: Vec<Entry> = hashes.iter().enumerate()
            .map(|(i, &h)| Entry { hash: h, index: i as u32 })
            .collect();
        let padded = v.len().next_power_of_two();
        v.resize(padded, Entry { hash: 0xFFFF_FFFF, index: 0xFFFF_FFFF });
        v
    }

    fn hashes(entries: &[Entry]) -> Vec<u32> {
        entries.iter().map(|e| e.hash).collect()
    }

    #[test]
    fn sort_small_known() {
        let mut data = padded_entries(&[3, 1, 4, 1, 5, 9, 2, 6]);
        cpu_radix_sort(&mut data);
        let h = hashes(&data);
        assert!(h.windows(2).all(|w| w[0] <= w[1]), "not sorted: {:?}", h);
    }

    #[test]
    fn sort_matches_stdlib() {
        // Pseudo-random hashes close to real spatial-hash range
        let mut data: Vec<Entry> = (0u32..1000).map(|i| Entry {
            hash: i.wrapping_mul(2654435761).wrapping_add(i >> 3) % 131072,
            index: i,
        }).collect();
        let padded = data.len().next_power_of_two();
        data.resize(padded, Entry { hash: 0xFFFF_FFFF, index: 0xFFFF_FFFF });

        let mut expected_hashes: Vec<u32> = data.iter().map(|e| e.hash).collect();
        expected_hashes.sort_unstable();

        cpu_radix_sort(&mut data);

        assert_eq!(hashes(&data), expected_hashes);
    }

    #[test]
    fn sort_all_same_hash() {
        let mut data = padded_entries(&vec![42u32; 300]);
        cpu_radix_sort(&mut data);
        assert!(hashes(&data).windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn sort_reverse_order() {
        let input: Vec<u32> = (0u32..512).rev().collect();
        let mut data = padded_entries(&input);
        cpu_radix_sort(&mut data);
        let h = hashes(&data);
        assert!(h.windows(2).all(|w| w[0] <= w[1]), "not sorted after reverse input");
    }

    #[test]
    fn sentinels_land_at_end() {
        let real_n = 100usize;
        let mut data = padded_entries(&(0u32..real_n as u32).collect::<Vec<_>>());
        cpu_radix_sort(&mut data);
        // All real entries must come before all sentinels
        let split = data.partition_point(|e| e.hash != 0xFFFF_FFFF);
        assert_eq!(split, real_n, "sentinels not at end");
    }
}
