# fastconv — state at a glance
> Update in-place on changes; git history preserves past snapshots.

## Snapshot
| Item | Value |
| :--- | :--- |
| Product | `fastconv` 0.1.0 · edition 2024 · AGPL-3.0-only · HEAD `c7b3632` + working-tree CPU-HEIC change `fd5e2778` (pre-merge / pre-review branch `feat/cpu-heic-encode`) · 5 crates |
| Verify | `cargo xtask verify` green — **485** default tests (27 cli + 9 cli_help + 64 domain + 287 pipeline + 76 probe + 22 xtask), **341** `--features nvenc` tests (27 cli + 9 cli_help + 305 pipeline); **273** deps license-checked; fmt + clippy `-D warnings` clean (default and nvenc) |
| Release binary | 36.6 MiB default (unstripped); ffmpeg ~38 MB incl. libs |

## CLI
| Command | Form |
| :--- | :--- |
| `probe` | `fastconv probe` — CPU model/topology/SIMD/RAM + GPU adapters & media backends |
| `convert` | `fastconv convert <src> <dst> [--to <fmt>] [--quality 0..=100 or --lossless] [--threads N] [--backend <name>]` |
| `audio` | `fastconv audio <src> <dst> [--to <fmt>] [--threads N] [--backend cpu]` — GPU rejected |
| `remux` | `fastconv remux <src> <dst> [--to mp4, mkv, webm, mov, avi, ts] [--threads N]` — no re-encode |
| `video` | `fastconv video <src> <dst> [--to <fmt>] [--vcodec <codec>] [--acodec <codec>] [--quality 0..=100] [--threads N] [--backend <name>] [--hwdecode auto, cpu, nvdec]` |
| `help` | `fastconv help <probe, convert, audio, remux, video>`; `-h`/`--help`, `-V`/`--version` |

## Formats
- **Image — 11 in × 12 out.** In: `png jpeg webp bmp qoi tiff jxl avif gif ico pnm` (P1–P7); out: same set + `heic`. Still GIF uses a fixed 256-color palette; `--quality`: webp/jpeg/jxl/avif/heic, `--lossless`: webp/jxl + png/bmp/qoi/tiff/ico/pnm; HEIC works on CPU (pure-Rust `heifer` HEVC-Intra, default backend) and on NVENC GPU (`--backend nvenc`).
- **Audio — 9 in × 9 out.** `wav flac opus ogg mp3 aac m4a alac aiff` (Ogg = Vorbis, AAC = ADTS, ALAC in M4A); CPU-only, parallel batches.
- **Video — 6 containers** `mp4 mov mkv webm avi ts`; codecs H.264/HEVC (H.265)/AV1/VP9 + AAC/Opus/MP3 + `copy`. CPU: H.264/AV1/VP9/GIF/WebP; NVENC: H.264/HEVC; `--hwdecode` NVDEC zero-copy.
- **Animation — bidirectional** video ↔ animated GIF/WebP via `fastconv video --to gif,webp --vcodec gif,webp` (CPU + NVENC); GIF/WebP also valid video inputs; `remux` rejects them.

## Key crates (Cargo.toml)
| Area | Versions |
| :--- | :--- |
| Image | `png 0.18`, `zune-jpeg 0.5.16-rc3`, `zenwebp 0.4.6`, `tiff 0.11.3`, `jxl 0.8.0`, `jixel 0.5.13`, `zenavif 0.1.6`, `zenavif-parse 0.6.2`, `gif 0.14.2`, `ico 0.6.0`, `rav1e 0.8.1`, `avif-serialize 0.8.9`, `jpeg-encoder 0.7.1`, `heifer 0.7.3` (CPU HEIC) |
| Audio | `flacenc 0.5.1`, `hound 3.5.1`, `opus-pure 0.2.5`, `vorbis-enc 0.1.1`, `rusty_mp3 1.2.2`, `rusty_aac 2.3.0`, `alac-encoder 0.3.0`, `symphonia 0.6.1`, `rubato 5.1.0`, `ogg 0.9.2` |
| Video | `rusty_h264 0.17.4`, `rusty_h265 0.7.3` (+ `rusty_h265-accel 0.7.3`), `rav1d-safe 0.5.7`, `rusty_vp9 0.3.0`, `containr 0.1.1`, `yuv 0.8.21` |
| GPU / misc | `cudarc 0.16`, `libloading 0.9`, `nvidia-video-codec-sdk 0.4` (nvenc feature only), `serde`/`serde_json 1`; no `wgpu` |

## Workspace / module map
| Crate | Sources |
| :--- | :--- |
| `domain` | `src/{lib,audio,hardware,image,metrics,video}.rs` — pure types/invariants |
| `probe` | `src/{lib,audio,hardware,image,video}.rs` — boundary parsing/validation |
| `pipeline` | `src/{lib,audio,aiff,avif,avi,bmp,cpu,decode,gif,heic,ico,m4a,mpegts,pnm,qoi,remux,video_anim,video_bitstream,video_codecs,video_container,video_transcode}.rs` + `src/gpu/{mod,nvdec,nvenc,heic}.rs` |
| `cli` (`fastconv`) | `src/{main,audio,video}.rs`, `tests/cli_help.rs` — thin orchestration |
| `xtask` | `src/{main,bench,audio_bench,nvenc_bench,remux_bench,fixtures}.rs` — verify / fixtures / bench |

## Invariants
- No FFmpeg in `crates/` (scanner in `cargo xtask verify`); ffmpeg only in `xtask` as bench opponent / fixture generator.
- Boundary validation in `probe`; `domain` holds valid types only; illegal states (audio + GPU, quality on lossless formats, NVENC non-HEIC) are enum-level (`ExecutionPlan`, `ImageExecutionMode`, `ImageCompression`, `GpuMediaKind`); CPU HEIC is a valid CPU image mode.
- Zero silent fallback: failed GPU setup, `--backend vaapi/vulkan`, CPU HEVC video encode, failed NVDEC → hard error; requesting `--backend cpu` for HEIC uses `heifer` directly, not a failed-GPU fallback.
- Lints: `unsafe_code = "forbid"`, `unwrap_used`/`expect_used`/`panic`/`todo`/`unimplemented`/`dbg_macro` = `deny`, `clippy::pedantic = "warn"`; pipeline relaxes unsafe only under `--features nvenc`, confined to narrowly scoped `#[allow(unsafe_code)]` FFI blocks in `gpu/nvenc.rs` and `gpu/nvdec.rs`.
- AGPL-3.0-only; 273 deps permissive/AGPL-compatible; `GPL-2.0-only` rejected (Classpath exception allowed).
- RAM caps (actual constants): JXL/AVIF ≤8 image workers, HEIC ≤8 image workers (RGB/RGBA scratch + HEVC tile state), AVIF inner threads ≤8, MP4 faststart ≤64 MiB in-memory (stream above), ≤8 animation workers, ≤2 CPU video file workers, 1 GiB decoded/track, 512 MiB PCM/file, 1 GiB PCM/batch.
- Turing reality: NVENC = HEVC-intra (HEIC) only on the GPU path; CPU HEIC (`heifer`) works on any host; no HW encoders for webp/jpeg/png/bmp/qoi/tiff/jxl/avif/gif/ico/pnm; AV1 CPU-decode; image Vulkan rejected. GPU probe Linux-only (DRM sysfs + NVIDIA procfs).

## Benchmarks — canonical 2026-11-27 (pre-merge / pre-review branch run)
Host: `Intel Core i7-6700` (4C/8T AVX2/FMA, 32 GiB) + `RTX 2060` (6 GB, driver `620.73.04`); `ffmpeg n9.1`; `rustc`/`cargo 1.100.0`; all runs serial.

- Provenance: HEAD `c7b36320788965d66b6a465bf3c284d29fe0fe83`; code `git diff | sha256sum` = `fd5e27789ab56fc897ce9a41b637d860e86fc093dc2d32c53ec27e112b09eff6` (CPU-HEIC feature diff: `Cargo.toml`/`Cargo.lock`, CLI/probe/domain/pipeline sources, new `crates/pipeline/src/heic.rs`; `STATE.md` updated afterwards).
- Commands: `cargo xtask gen-fixtures`; `timeout -k 10 600s cargo xtask bench`; `cargo xtask bench --audio`; `cargo xtask bench --nvenc`; `cargo xtask bench --remux`; `cargo xtask verify`. CPU HEIC (no `xtask` target): 3 serial runs each of `target/release/fastconv convert <src> <dst> --to heic --backend cpu --quality 80` — single `fixtures/images_4k/img_01.png` and 20-image `fixtures/images_4k/` batch. Wall time + exact peak RSS captured with `wait4` rusage (VmHWM equivalent; `/usr/bin/time` is unavailable on this host); median time and max peak RSS reported.
- Validation: image 20/20 per batch row (21 harness rows + CPU HEIC batch; CPU HEIC single 1/1; all 3840x2160; lossless rows exact RGB, fastconv lossy rows ≥ 33.95 dB); audio 34/34 ffprobe + 18/18 content checks (AAC/M4A SNR 26.4 dB on both); NVENC 4/4 (600 frames / 10 s, clean decode, HEVC 47.62 dB, explicit GPU path); remux 4/4 (codec-copy `aac`+`h264`, 600 frames, no decoder errors). CPU HEIC: single 1/1 + batch 20/20 — `ftypheic` major brand, ffmpeg tile-grid reconstruction at 3840x2160, full clean `rgb24` decode (24,883,200 B/image), PSNR 35.45 dB single / 35.17–35.53 dB batch (mean 35.39) vs source PNG; all ≥ 25 dB.
- Animation: 2026-11-26, `6f7f14a` (PR #21, no `xtask` target; not re-run for this image-HEIC feature).
- Single-image CPU HEIC costs ~6 s, so the 20-image batch used a 600 s guard (3 serial runs ≈ 6.1 min total). No runs were discarded or cherry-picked.

### Image — 20× 4K, Q=80 / same compression class (20/20 per batch row; CPU HEIC single 1/1)
| Workload | Tool | Time (ms) | Throughput | Peak RAM | Input | Output | Quality | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `png_to_webp` | `fastconv (cpu q=80)` | 4204.7 | 72.24 MB/s | 629.4 MiB | 303.75 MB | 5.69 MB | 33.95 dB | **3.00x** |
| `png_to_webp` | `ffmpeg (libwebp q=80)` | 12599.4 | 24.11 MB/s | 671.7 MiB | 303.75 MB | 5.34 MB | 33.61 dB | Baseline |
| `png_to_jpeg` | `fastconv (cpu q=80)` | 657.9 | 461.70 MB/s | 430.8 MiB | 303.75 MB | 13.33 MB | 34.29 dB | **1.30x** |
| `png_to_jpeg` | `ffmpeg (mjpeg qv=3)` | 855.4 | 355.11 MB/s | 865.3 MiB | 303.75 MB | 8.58 MB | 33.98 dB | Baseline |
| `jpeg_to_webp` | `fastconv (cpu q=80)` | 3985.3 | 6.74 MB/s | 425.7 MiB | 26.86 MB | 5.90 MB | 38.12 dB | **2.93x** |
| `jpeg_to_webp` | `ffmpeg (libwebp q=80)` | 11695.8 | 2.30 MB/s | 165.6 MiB | 26.86 MB | 5.44 MB | 38.40 dB | Baseline |
| `webp_to_png` | `fastconv (cpu png)` | 543.2 | 10.47 MB/s | 511.2 MiB | 5.69 MB | 153.64 MB | exact | **3.74x** |
| `webp_to_png` | `ffmpeg (png)` | 2034.3 | 2.80 MB/s | 816.1 MiB | 5.69 MB | 89.97 MB | exact | Baseline |
| `png_to_qoi` | `fastconv (cpu qoi)` | 737.9 | 411.63 MB/s | 642.3 MiB | 303.75 MB | 250.91 MB | exact | **1.54x** |
| `png_to_qoi` | `ffmpeg (qoi)` | 1136.7 | 267.23 MB/s | 972.7 MiB | 303.75 MB | 250.91 MB | exact | Baseline |
| `webp_to_bmp` | `fastconv (cpu bmp)` | 360.4 | 15.78 MB/s | 716.3 MiB | 5.69 MB | 497.67 MB | exact | **1.24x** |
| `webp_to_bmp` | `ffmpeg (bmp)` | 446.6 | 12.73 MB/s | 484.6 MiB | 5.69 MB | 497.67 MB | exact | Baseline |
| `png_to_tiff` | `fastconv (cpu tiff)` | 880.4 | 345.03 MB/s | 399.0 MiB | 303.75 MB | 276.93 MB | exact (both Deflate) | **2.83x** |
| `png_to_tiff` | `ffmpeg (tiff)` | 2490.4 | 121.97 MB/s | 1381.4 MiB | 303.75 MB | 373.65 MB | exact | Baseline |
| `png_to_jxl` | `fastconv (cpu q=80)` | 3853.3 | 78.83 MB/s | 1384.8 MiB | 303.75 MB | 1.25 MB | 34.80 dB | **4.71x** (same distance 1.9) |
| `png_to_jxl` | `ffmpeg (libjxl d=1.9)` | 18133.5 | 16.75 MB/s | 971.3 MiB | 303.75 MB | 6.49 MB | 34.95 dB | Baseline |
| `png_to_avif` | `fastconv (cpu q=80)` | 2710.3 | 112.07 MB/s | 1293.3 MiB | 303.75 MB | 0.63 MB | 35.63 dB | **1.29x** (−0.12 dB, 4.5x smaller) |
| `png_to_avif` | `ffmpeg (libsvtav1 preset 8)` | 3508.2 | 86.58 MB/s | 2825.7 MiB | 303.75 MB | 2.86 MB | 35.75 dB | Baseline |
| `png_to_heic` | `fastconv (gpu:nvenc q=80)` | 947.4 | 320.60 MB/s | 733.3 MiB | 303.75 MB | 27.47 MB | 39.21 dB | **13.30x** vs ff(webp) / **4.44x** vs cpu(webp) |
| `jpeg_to_heic` | `fastconv (gpu:nvenc q=80)` | 844.6 | 31.80 MB/s | 527.4 MiB | 26.86 MB | 19.24 MB | 38.16 dB | **13.85x** vs ff(webp) / **4.72x** vs cpu(webp) |
| `webp_to_heic` | `fastconv (gpu:nvenc q=80)` | 881.6 | 6.45 MB/s | 782.5 MiB | 5.69 MB | 13.21 MB | 38.10 dB | NVENC 20× 4K (no ffmpeg HEIC opponent) |
| `png_to_heic` (single 4K) | `fastconv (cpu heifer q=80, 3 runs)` | 6251.9 | 2.43 MB/s | 135.8 MiB | 15.18 MB | 1.09 MB | 35.45 dB | CPU-only (no ffmpeg HEIC opponent) |
| `png_to_heic` (20× 4K batch) | `fastconv (cpu heifer q=80, 8 workers, 3 runs)` | 122310.5 | 2.48 MB/s | 467.6 MiB | 303.75 MB | 21.92 MB | 35.39 dB mean | CPU-only (no ffmpeg HEIC opponent) |

### Audio — 60 s stereo 48 kHz/16-bit, target-matched bitrate/quality (achieved size/PSNR per row; 34/34 outputs validated: 16 single-file + 18 batch)
| Workload | Tool | Time (ms) | Throughput | Peak RAM | Input | Output | Quality | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wav_to_flac` | `fastconv (flacenc pure rust)` | 133.8 | 86.07 MB/s | 72.0 MiB | 11.52 MB | 4.12 MB | bit-exact | **1.58x** |
| `wav_to_flac` | `ffmpeg (flac -5)` | 211.8 | 54.38 MB/s | 57.3 MiB | 11.52 MB | 4.11 MB | bit-exact | Baseline |
| `wav_to_aiff` | `fastconv (native pcm_s16be)` | 32.4 | 355.67 MB/s | 20.4 MiB | 11.52 MB | 11.52 MB | bit-exact | **3.85x** |
| `wav_to_aiff` | `ffmpeg (pcm_s16be)` | 124.8 | 92.32 MB/s | 49.6 MiB | 11.52 MB | 11.52 MB | bit-exact | Baseline |
| `wav_to_alac` | `fastconv (alac-encoder m4a)` | 149.6 | 77.00 MB/s | 30.2 MiB | 11.52 MB | 4.48 MB | bit-exact | **2.39x** |
| `wav_to_alac` | `ffmpeg (alac)` | 357.8 | 32.20 MB/s | 52.5 MiB | 11.52 MB | 4.69 MB | bit-exact | Baseline |
| `wav_to_opus` | `fastconv (opus-pure 0.2.5)` | 152.5 | 75.52 MB/s | 73.2 MiB | 11.52 MB | 0.98 MB | 37.1 dB | **3.76x** |
| `wav_to_opus` | `ffmpeg (libopus 128k)` | 573.4 | 20.09 MB/s | 55.9 MiB | 11.52 MB | 1.35 MB | — | Baseline |
| `wav_to_ogg` | `fastconv (vorbis-enc q4)` | 844.1 | 13.65 MB/s | 36.2 MiB | 11.52 MB | 0.74 MB | 35.8 dB | **1.09x** |
| `wav_to_ogg` | `ffmpeg (libvorbis q4)` | 916.9 | 12.56 MB/s | 55.9 MiB | 11.52 MB | 0.70 MB | — | Baseline |
| `wav_to_mp3` | `fastconv (rusty_mp3 192k)` | 650.9 | 17.70 MB/s | 38.2 MiB | 11.52 MB | 1.44 MB | 64.3 dB | **1.51x** |
| `wav_to_mp3` | `ffmpeg (libmp3lame 192k)` | 983.1 | 11.72 MB/s | 53.0 MiB | 11.52 MB | 1.44 MB | — | Baseline |
| `wav_to_aac` | `fastconv (rusty_aac 2.3.0, 128k ADTS)` | 150.3 | 76.65 MB/s | 61.9 MiB | 11.52 MB | 0.98 MB | 26.4 dB | **10.60x** |
| `wav_to_aac` | `ffmpeg (aac 128k)` | 1593.6 | 7.23 MB/s | 55.1 MiB | 11.52 MB | 0.98 MB | — | Baseline |
| `wav_to_m4a` | `fastconv (rusty_aac 2.3.0, 128k muxed)` | 154.4 | 74.62 MB/s | 63.6 MiB | 11.52 MB | 0.97 MB | 26.4 dB | **10.45x** |
| `wav_to_m4a` | `ffmpeg (aac 128k)` | 1612.6 | 7.14 MB/s | 54.4 MiB | 11.52 MB | 0.97 MB | — | Baseline |
| `audio_batch_mixed_to_flac` | `fastconv (parallel, 9 inputs)` | 167.5 | 36.09 MB/s | 82.2 MiB | 6.04 MB | 5.65 MB | 9/9 validated | **5.30x** |
| `audio_batch_mixed_to_flac` | `ffmpeg (serial, 9 inputs)` | 887.8 | 6.81 MB/s | 60.5 MiB | 6.04 MB | 6.83 MB | 9/9 validated | Baseline |

### Video — NVENC & Remux, 1080p60, 600 frames, 3 serial runs/side
| Workload | Tool | Time (ms) | FPS | Peak RAM | Output | PSNR | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nvenc_h264_mp4` | `fastconv (nvdec+nvenc)` | 1846.9 | 324.9 | 480.4 MiB | 25.86 MB | 50.72 dB | **1.06x** |
| `nvenc_h264_mp4` | `ffmpeg (h264_nvenc, hwaccel cuda)` | 1959.8 | 306.2 | 631.7 MiB | 25.90 MB | 50.73 dB | Baseline |
| `nvenc_hevc_mp4` | `fastconv (nvdec+nvenc)` | 1996.5 | 300.5 | 595.0 MiB | 20.52 MB | 47.62 dB | **1.03x** |
| `nvenc_hevc_mp4` | `ffmpeg (hevc_nvenc, hwaccel cuda)` | 2047.1 | 293.1 | 580.8 MiB | 20.54 MB | 47.62 dB | Baseline |
| `remux_mp4_to_mkv` | `fastconv (pure rust)` | 15.3 | 39130.4 | 17.2 MiB | 19.16 MB | codec-copy | **5.58x** |
| `remux_mp4_to_mkv` | `ffmpeg (-c copy)` | 85.4 | 7021.9 | 60.4 MiB | 19.14 MB | codec-copy | Baseline |
| `remux_mkv_to_mp4` | `fastconv (pure rust)` | 24.0 | 24962.0 | 37.4 MiB | 19.15 MB | codec-copy | **3.48x** |
| `remux_mkv_to_mp4` | `ffmpeg (-c copy)` | 83.4 | 7198.6 | 57.5 MiB | 19.15 MB | codec-copy | Baseline |

### Animation — 480×270 / 15 fps / 60 frames, 3 serial runs (median time, max VmHWM), 12/12 validated
| Workload | Tool | Time (ms) | FPS | MB/s (input) | Peak RAM | Output | PSNR vs. source | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mp4_to_gif` | `fastconv (cpu gif q80)` | 262.6 | 228.5 | 0.57 | 34.7 MiB | 1.19 MB | 35.39 dB | **2.58x** |
| `mp4_to_gif` | `ffmpeg (palettegen/use)` | 678.3 | 88.5 | 0.22 | 111.9 MiB | 1.02 MB | 36.23 dB | Baseline |
| `mp4_to_webp` | `fastconv (cpu webp q80)` | 163.8 | 366.3 | 0.91 | 46.9 MiB | 0.36 MB | 39.99 dB | **4.13x** |
| `mp4_to_webp` | `ffmpeg (libwebp_anim q80)` | 675.9 | 88.8 | 0.22 | 70.7 MiB | 0.36 MB | 39.04 dB | Baseline |
| `gif_to_mp4` | `fastconv (cpu h264 q80)` | 100.2 | 598.8 | 10.14 | 45.6 MiB | 0.37 MB | 37.73 dB | **1.80x** |
| `gif_to_mp4` | `ffmpeg (libx264 qp17)` | 180.5 | 332.4 | 5.63 | 73.5 MiB | 0.34 MB | 36.93 dB | Baseline |
| `webp_to_mp4` | `fastconv (cpu h264 q80)` | 149.8 | 400.5 | 2.38 | 49.9 MiB | 0.30 MB | 29.79 dB | **1.23x** |
| `webp_to_mp4` | `ffmpeg (libx264 qp17)` | 183.9 | 326.3 | 1.94 | 72.0 MiB | 0.27 MB | 48.94 dB | Baseline |
| `gif_to_mp4_nv` | `fastconv (nvenc h264 q80)` | 361.7 | 165.9 | 2.81 | 228.9 MiB | 0.19 MB | 36.87 dB | **1.14x** |
| `gif_to_mp4_nv` | `ffmpeg (h264_nvenc cq23)` | 411.7 | 145.7 | 2.47 | 289.9 MiB | 0.19 MB | 36.84 dB | Baseline |
| `webp_to_mkv_nv` | `fastconv (nvenc hevc q80)` | 399.2 | 150.3 | 0.89 | 246.0 MiB | 0.18 MB | 29.63 dB | **1.06x** |
| `webp_to_mkv_nv` | `ffmpeg (hevc_nvenc cq23)` | 424.3 | 141.4 | 0.84 | 291.6 MiB | 0.18 MB | 36.56 dB | Baseline |

## Known caveats & gaps
- Audio lossy rows are target-matched, not result-matched: Opus uses 128 kbps on both sides but VBR complexity 9 (fastconv) vs 10 (libopus), so achieved sizes differ (0.98 vs 1.35 MB); per-row PSNR is fastconv-only where ffmpeg shows `—`.
- WebP animation-input PSNR lag: 29.79 vs 48.94 dB (CPU), 29.63 vs 36.56 dB (NVENC) — known `zenwebp` vs `libwebp` color-conversion delta (still-image repro 30.1 dB); not bought with lower compression (0.30 vs 0.27 MB).
- Turing AV1 is CPU-decoded; `--hwdecode nvdec` is rejected with no fallback. CPU HEVC *video* encode is rejected; still-image HEIC is available on CPU (`heifer`) and on NVENC GPU (image output remains HEIC-only).
- GPU probe is Linux-only (DRM sysfs / NVIDIA procfs); other OSes report no adapters/backends. Animation bench not yet in `xtask` (external 3-run scripts); not re-run for this image-HEIC feature.
- Absolute timings on short rows shifted run-to-run on this snapshot for both sides (e.g. `wav_to_flac` 133.8 vs 211.8 ms, `wav_to_aiff` 32.4 vs 124.8 ms, remux fastconv 15.3 vs 24.0 ms). Output sizes, PSNR/SNR and every validation check are unchanged in kind; no runs were discarded. Smallest margin: `wav_to_ogg` 844.1 vs 916.9 ms (1.09x).
- CPU HEIC throughput is ~2.4 MB/s on this 4C/8T host: `heifer`'s Medium HEVC-Intra preset is intentionally kept (no preset cheating), and the 20-image batch caps at 8 image workers to bound RAM (peak 467.6 MiB, far below 3 GiB). The NVENC path encodes the same 20-image batch in 947.4 ms, but that is a different quality class (27.47 MB / 39.21 dB vs 21.92 MB / 35.39 dB CPU), not a fair speedup comparison.
- `ffprobe` lists CPU-HEIC output as 40 × 512×512 HEVC tile items + a 320×180 thumbnail rather than one 3840×2160 stream; full resolution is confirmed via ffmpeg's `Tile Grid: ... 3840x2160` group and the clean 24,883,200-byte RGB24 decode.
- AVIF/JXL matched PSNR and smaller outputs; TIFF both Deflate, exact; AVIF peak RAM 1293.3 vs 2825.7 MiB.
