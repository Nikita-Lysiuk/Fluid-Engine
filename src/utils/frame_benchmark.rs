use std::time::{Duration, Instant};

/// Headless-friendly frame time measurement for the real application.
///
/// Enabled by setting `FLUID_BENCH_SECONDS=<n>`: the first `WARMUP` is
/// discarded, then frame times are collected for `n` seconds, a summary is
/// printed to stdout and the application is asked to exit.
pub struct FrameBenchmark {
    /// Time of the first rendered frame; start-up is not part of the measurement.
    started: Option<Instant>,
    duration: Duration,
    frame_ms: Vec<f32>,
    substeps: Vec<u32>,
}

const WARMUP: Duration = Duration::from_secs(3);

impl FrameBenchmark {
    pub fn from_env() -> Option<Self> {
        let seconds: f32 = std::env::var("FLUID_BENCH_SECONDS").ok()?.parse().ok()?;
        Some(Self {
            started: None,
            duration: Duration::from_secs_f32(seconds),
            frame_ms: Vec::new(),
            substeps: Vec::new(),
        })
    }

    /// Records one frame. Returns `true` once the measurement is finished.
    pub fn record(&mut self, frame_time: Duration, substeps: u32) -> bool {
        let elapsed = self.started.get_or_insert_with(Instant::now).elapsed();
        if elapsed < WARMUP {
            return false;
        }
        self.frame_ms.push(frame_time.as_secs_f32() * 1e3);
        self.substeps.push(substeps);
        elapsed >= WARMUP + self.duration
    }

    pub fn report(&self) {
        let mut sorted = self.frame_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len().max(1);
        let mean = sorted.iter().sum::<f32>() / n as f32;
        let p50 = sorted.get(n / 2).copied().unwrap_or(0.0);
        let p95 = sorted.get(n * 95 / 100).copied().unwrap_or(0.0);
        let max = sorted.last().copied().unwrap_or(0.0);
        let mean_substeps = self.substeps.iter().sum::<u32>() as f32 / n as f32;
        println!(
            "BENCH frames={} mean_ms={mean:.2} p50_ms={p50:.2} p95_ms={p95:.2} max_ms={max:.1} fps={:.1} substeps/frame={mean_substeps:.2}",
            sorted.len(),
            1000.0 / mean,
        );
    }
}
