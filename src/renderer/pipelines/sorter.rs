use crate::entities::particle::{Entry, GpuPhysicsData, SimulationParams};
use crate::renderer::pipelines::ComputeStep;
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
use vulkano::shader::EntryPoint;

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
        Self {
            pipeline,
            descriptor_set: None,
            num_elements: 0,
        }
    }
    fn prepare(
        &mut self,
        allocator: Arc<StandardDescriptorSetAllocator>,
        physics_data: &GpuPhysicsData,
        _sim_params: &Subbuffer<SimulationParams>,
    ) {
        let grid_entries = &physics_data.grid_entries;
        self.num_elements = grid_entries.len() as u32;
        assert!(
            self.num_elements.is_power_of_two(),
            "GpuSorter: Buffer len must be power of 2"
        );

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        self.descriptor_set = Some(
            DescriptorSet::new(
                allocator,
                layout.clone(),
                [WriteDescriptorSet::buffer(0, grid_entries.clone())],
                [],
            )
            .unwrap(),
        );
    }
    fn execute<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        let set = self
            .descriptor_set
            .as_ref()
            .expect("GpuSorter: call prepare() before execute()");
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
                    .bind_pipeline_compute(self.pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        self.pipeline.layout().clone(),
                        0,
                        set.clone(),
                    )
                    .unwrap()
                    .push_constants(self.pipeline.layout().clone(), 0, pc)
                    .unwrap();

                let threads_needed = num_elements / 2;
                let group_size = 256;
                let dispatch_count = (threads_needed + group_size - 1) / group_size;

                unsafe {
                    builder.dispatch([dispatch_count, 1, 1]).unwrap();
                }

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

            let src = if is_even {
                buf_a.clone()
            } else {
                buf_b.clone()
            };
            let dst = if is_even { &mut buf_b } else { &mut buf_a };

            // ── Count (mirrors radix_count.comp) ─────────────────────────────
            let mut counts = vec![0u32; 256 * num_wg];
            for wg in 0..num_wg {
                let mut local = [0u32; 256];
                for lid in 0..WG_SIZE {
                    let base = wg * WG_SIZE * EPI + lid * EPI;
                    for i in 0..EPI {
                        let idx = base + i;
                        if idx >= n {
                            break;
                        }
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
                        if idx >= n {
                            break;
                        }
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
        let mut v: Vec<Entry> = hashes
            .iter()
            .enumerate()
            .map(|(i, &h)| Entry {
                hash: h,
                index: i as u32,
            })
            .collect();
        let padded = v.len().next_power_of_two();
        v.resize(
            padded,
            Entry {
                hash: 0xFFFF_FFFF,
                index: 0xFFFF_FFFF,
            },
        );
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
        let mut data: Vec<Entry> = (0u32..1000)
            .map(|i| Entry {
                hash: i.wrapping_mul(2654435761).wrapping_add(i >> 3) % 131072,
                index: i,
            })
            .collect();
        let padded = data.len().next_power_of_two();
        data.resize(
            padded,
            Entry {
                hash: 0xFFFF_FFFF,
                index: 0xFFFF_FFFF,
            },
        );

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
        assert!(
            h.windows(2).all(|w| w[0] <= w[1]),
            "not sorted after reverse input"
        );
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

// ── GPU benchmark ─────────────────────────────────────────────────────────────
//
// Wall-clock benchmark comparing bitonic vs radix sort on the GPU across a
// range of N. Writes scripts/sort_benchmark.csv. Marked `#[ignore]` so it is
// not run by default; invoke manually with:
//
//     cargo test --release -p fluid_engine -- --ignored sort_benchmark --nocapture
//
// The test creates a headless `VulkanoContext` (no swapchain) and reuses
// `GpuSorter` and `RadixSorter` exactly as production code does.

#[cfg(test)]
mod benchmark {
    use std::fs;
    use std::io::Write;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Instant;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
    use vulkano::command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    };
    use vulkano::command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract,
    };
    use vulkano::descriptor_set::allocator::{
        StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
    };
    use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
    use vulkano::device::DeviceFeatures;
    use vulkano::instance::{InstanceCreateInfo, InstanceExtensions};
    use vulkano::memory::allocator::{
        AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
    };
    use vulkano::pipeline::Pipeline;
    use vulkano::sync::GpuFuture;
    use vulkano_util::context::{VulkanoConfig, VulkanoContext};

    use super::{GpuSorter, RadixSorter};
    use crate::entities::particle::Entry;
    use crate::renderer::pipelines::ComputeStep;

    // All values are powers of two so bitonic sort can use them directly.
    const N_VALUES: &[u32] = &[
        1_024,     // 1k
        4_096,     // 4k
        16_384,    // 16k
        65_536,    // 64k
        131_072,   // 128k
        262_144,   // 256k
        524_288,   // 512k
        1_048_576, // 1M
        2_097_152, // 2M
        4_194_304, // 4M
    ];
    const WARMUP_RUNS: u32 = 3;
    const TIMED_RUNS: u32 = 15;
    const CSV_PATH: &str = "scripts/sort_benchmark.csv";

    fn make_context() -> Arc<VulkanoContext> {
        // Matches the production VulkanoConfig minus the WSI extensions and
        // rendering-only features. `ext_debug_utils` is required because
        // `RadixSorter::execute` emits `begin_debug_utils_label` for RenderDoc.
        let config = VulkanoConfig {
            instance_create_info: InstanceCreateInfo {
                enabled_extensions: InstanceExtensions {
                    ext_debug_utils: true,
                    ..InstanceExtensions::default()
                },
                ..Default::default()
            },
            device_features: DeviceFeatures {
                scalar_block_layout: true,
                buffer_device_address: true,
                shader_int64: true,
                ..DeviceFeatures::empty()
            },
            ..VulkanoConfig::default()
        };
        Arc::new(VulkanoContext::new(config))
    }

    fn make_entries_buffer(
        memory_allocator: Arc<StandardMemoryAllocator>,
        n: u32,
    ) -> Subbuffer<[Entry]> {
        Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (0..n).map(|i| Entry { hash: 0, index: i }),
        )
        .expect("failed to create entries buffer")
    }

    fn fill_random(entries: &Subbuffer<[Entry]>, seed: u64) {
        let mut data = entries.write().expect("failed to map entries buffer");
        let mut rng = StdRng::seed_from_u64(seed);
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = Entry {
                hash: rng.random::<u32>(),
                index: i as u32,
            };
        }
    }

    fn median(samples: &mut [f64]) -> f64 {
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        samples[samples.len() / 2]
    }

    fn time_runs<F>(mut build_cmd: F, queue: Arc<vulkano::device::Queue>) -> f64
    where
        F: FnMut(u32) -> Arc<PrimaryAutoCommandBuffer>,
    {
        let mut samples: Vec<f64> = Vec::with_capacity(TIMED_RUNS as usize);
        for run in 0..(WARMUP_RUNS + TIMED_RUNS) {
            let cmd = build_cmd(run);
            let start = Instant::now();
            cmd.execute(queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();
            let elapsed = start.elapsed();
            if run >= WARMUP_RUNS {
                samples.push(elapsed.as_secs_f64() * 1000.0);
            }
        }
        median(&mut samples)
    }

    fn bench_bitonic(
        context: &VulkanoContext,
        cmd_allocator: Arc<StandardCommandBufferAllocator>,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
        entries: &Subbuffer<[Entry]>,
        n: u32,
    ) -> f64 {
        assert!(
            n.is_power_of_two(),
            "Bitonic sort requires N to be a power of two"
        );

        let mut sorter = GpuSorter::new(context.device().clone());
        sorter.num_elements = n;
        let layout = sorter.pipeline.layout().set_layouts().get(0).unwrap();
        sorter.descriptor_set = Some(
            DescriptorSet::new(
                desc_allocator,
                layout.clone(),
                [WriteDescriptorSet::buffer(0, entries.clone())],
                [],
            )
            .unwrap(),
        );

        let queue = context.graphics_queue().clone();
        let qfi = queue.queue_family_index();

        time_runs(
            |run| {
                fill_random(entries, run as u64);
                let mut builder = AutoCommandBufferBuilder::primary(
                    cmd_allocator.clone(),
                    qfi,
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();
                sorter.execute(&mut builder);
                builder.build().unwrap()
            },
            queue,
        )
    }

    fn bench_radix(
        context: &VulkanoContext,
        memory_allocator: Arc<StandardMemoryAllocator>,
        cmd_allocator: Arc<StandardCommandBufferAllocator>,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
        entries: &Subbuffer<[Entry]>,
        n: u32,
    ) -> f64 {
        let mut sorter = RadixSorter::new(context.device().clone(), memory_allocator, n);
        sorter.prepare(desc_allocator, entries);

        let queue = context.graphics_queue().clone();
        let qfi = queue.queue_family_index();

        time_runs(
            |run| {
                fill_random(entries, run as u64);
                let mut builder = AutoCommandBufferBuilder::primary(
                    cmd_allocator.clone(),
                    qfi,
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();
                sorter.execute(&mut builder);
                builder.build().unwrap()
            },
            queue,
        )
    }

    #[test]
    #[ignore]
    fn sort_benchmark() {
        let context = make_context();
        let device = context.device().clone();

        let memory_allocator: Arc<StandardMemoryAllocator> =
            Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let cmd_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let desc_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        if let Some(parent) = Path::new(CSV_PATH).parent() {
            fs::create_dir_all(parent).expect("failed to create CSV output dir");
        }
        let mut csv = fs::File::create(CSV_PATH).expect("failed to create CSV file");
        writeln!(csv, "n,algorithm,time_ms").unwrap();

        println!("\nsort benchmark — output: {CSV_PATH}");
        println!("warmup={WARMUP_RUNS}, timed={TIMED_RUNS} (median reported)\n");
        println!("{:>10}  {:>10}  {:>12}", "N", "algorithm", "time_ms");
        println!("{:->10}  {:->10}  {:->12}", "", "", "");

        for &n in N_VALUES {
            let entries = make_entries_buffer(memory_allocator.clone(), n);

            let t_bitonic = bench_bitonic(
                &context,
                cmd_allocator.clone(),
                desc_allocator.clone(),
                &entries,
                n,
            );
            writeln!(csv, "{n},Bitonic,{:.4}", t_bitonic).unwrap();
            println!("{n:>10}  {:>10}  {:>12.4}", "Bitonic", t_bitonic);

            let t_radix = bench_radix(
                &context,
                memory_allocator.clone(),
                cmd_allocator.clone(),
                desc_allocator.clone(),
                &entries,
                n,
            );
            writeln!(csv, "{n},Radix,{:.4}", t_radix).unwrap();
            println!("{n:>10}  {:>10}  {:>12.4}", "Radix", t_radix);
        }

        csv.flush().unwrap();
        println!("\nwrote: {CSV_PATH}");
    }
}
