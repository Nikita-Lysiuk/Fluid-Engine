use super::{BitonicSorter, RadixSorter};
use super::radix_sorter::ELEMENTS_PER_INVOCATION;

#[cfg(test)]
mod tests {
    use crate::particle::Entry;

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
// `BitonicSorter` and `RadixSorter` exactly as production code does.

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

    use super::{BitonicSorter, RadixSorter};
    use crate::particle::Entry;
    
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

        let mut sorter = BitonicSorter::new(context.device().clone());
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
