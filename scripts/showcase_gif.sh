#!/usr/bin/env bash
# Turns the frames written by `cargo run --release --bin showcase` into the
# README animation. Needs ffmpeg.
#
#   scripts/showcase_gif.sh [frames_dir] [output.gif]
set -euo pipefail

frames="${1:-target/showcase}"
output="${2:-docs/showcase.gif}"
width=640
fps=20

# One palette for the whole clip (stats_mode=diff favours the moving water),
# then dither with it. Downscaling the 1080p frames also smooths the points.
filters="fps=${fps},scale=${width}:-1:flags=lanczos"
ffmpeg -y -loglevel error -framerate "${fps}" -i "${frames}/frame_%04d.png" \
  -vf "${filters},palettegen=max_colors=256:stats_mode=diff" "${frames}/palette.png"
ffmpeg -y -loglevel error -framerate "${fps}" -i "${frames}/frame_%04d.png" -i "${frames}/palette.png" \
  -lavfi "${filters}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
  -loop 0 "${output}"
ls -lh "${output}"
