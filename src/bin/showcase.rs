//! Renders the README animation frames offline.
//!
//!     cargo run --release --bin showcase -- [output_dir] [--preview]
//!
//! then `scripts/showcase_gif.sh [output_dir]` turns them into a GIF.
//! `--preview` renders a handful of frames to check the framing quickly.

use fluid_engine::showcase::{ShowcaseOptions, render};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let preview = args.iter().any(|arg| arg == "--preview");
    let output_dir = args
        .iter()
        .find(|arg| !arg.starts_with("--"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/showcase"));
    render(&ShowcaseOptions {
        output_dir,
        width: 1920,
        height: 1080,
        fps: 20,
        settle_seconds: 1.5,
        wave_warmup_seconds: 4.0,
        // Three periods of the wall (0.4 Hz), so the GIF loops.
        seconds: if preview { 2.0 } else { 7.5 },
    });
}
