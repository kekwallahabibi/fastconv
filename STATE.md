# fastconv — state at a glance
> Update in-place on changes; git history preserves past snapshots.

## Snapshot
| Item | Value |
| :--- | :--- |
| Product | `fastconv` 0.1.0 · edition 2024 · AGPL-3.0-only · pre-merge / pre-review branch `chore/shrink-binary-and-codebase` (binary/codebase shrink: fat LTO + strip + shared staging, zero behaviour change) · 5 crates |
| Verify | `cargo xtask verify` green — **707** default tests (33 cli + 3 audio_codecs + 11 cli_help + 101 domain + 407 pipeline + 130 probe + 22 xtask), **474** `--features nvenc` tests (33 cli + 3 audio_codecs + 11 cli_help + 427 pipeline); **273** deps license-checked; fmt + clippy `-D warnings` clean (default and nvenc) |
| Release binary | default **39,724,568 B (37.88 MiB) → 28,503,464 B (27.18 MiB), −28.25%**; `--features nvenc` **40,410,144 B (38.54 MiB) → 29,155,880 B (27.81 MiB), −27.85%** — `strip = true`, `lto = "fat"`, `codegen-units = 1`, `opt-level = 3` |

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
| GPU / misc | `cudarc 0.16`, `libloading 0.8`, `nvidia-video-codec-sdk 0.4` (nvenc feature only), `serde`/`serde_json 1`; no `wgpu` |

## Workspace / module map
| Crate | Sources |
| :--- | :--- |
| `domain` | `src/{lib,audio,hardware,image,metrics,video}.rs` — pure types/invariants |
| `probe` | `src/{lib,audio,hardware,image,video}.rs` — boundary parsing/validation |
| `pipeline` | `src/{lib,audio,aiff,avif,avi,bmp,cpu,decode,gif,heic,ico,m4a,mpegts,pnm,qoi,remux,staging,video_anim,video_bitstream,video_codecs,video_container,video_transcode}.rs` + `src/gpu/{mod,nvdec,nvenc,heic}.rs` |
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

## Benchmarks — canonical 2026-11-27 (pre-merge / pre-review branch run, Feature 13 binary/codebase shrink)
Host: `Intel Core i7-6700` (4C/8T AVX2/FMA, 32023 MiB) + `RTX 2060` (6 GB, driver `620.73.04`); `ffmpeg n9.1`; `rustc`/`cargo 1.100.0`; all runs serial.

- Provenance (this snapshot): branch `chore/shrink-binary-and-codebase`, HEAD `5b718cbf580c5c6521778a60ac7776a5458fc4d2` plus the uncommitted shrink working tree; code `git diff -- Cargo.toml Cargo.lock crates/ xtask/ | sha256sum` = `f2067361f01dc99be4d3de0cf4c43d119d88815bff454639afeb00504c7b8234`; new untracked `crates/pipeline/src/staging.rs` `sha256sum` = `86c8f266525fc425a48c91dc1f92ff631db12e431493905e24eec7ef88e7af42`. `target/release/fastconv probe` reported `Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz` (4C/8T, x86:avx2+fma, 32023 MiB) and `NVIDIA GeForce RTX 2060` (driver `620.73.04`), media backends `nvenc [encode+decode]` and `vaapi [decode-only]`. `STATE.md` was updated after the runs; no code file was modified by the benchmark.
- Commands: `cargo xtask gen-fixtures`; `cargo build --release -p fastconv-cli`; `cargo build --release -p fastconv-cli --features nvenc`; `target/release/fastconv probe`; `timeout -k 10 600s cargo xtask bench`; `timeout -k 10 240s cargo xtask bench --audio` (run twice serially to diagnose the batch row; both retained); `timeout -k 10 240s cargo xtask bench --nvenc`; `timeout -k 10 240s cargo xtask bench --remux`; `cargo xtask verify`. CPU HEIC (no `xtask` target, retained unchanged — `heifer` preset untouched): 3 serial runs each of `target/release/fastconv convert <src> <dst> --to heic --backend cpu --quality 80` — single `fixtures/images_4k/img_01.png` and 20-image `fixtures/images_4k/` batch. Wall time + exact peak RSS captured with `wait4` rusage (VmHWM equivalent; `/usr/bin/time` is unavailable on this host); median time and max peak RSS reported. Animation (no `xtask` target): retained unchanged from the prior snapshot (12/12 validated, external 3-run scripts).
- Validation: image 20/20 per batch row (21 harness rows, all 3840x2160; lossless rows `webp_to_png`/`png_to_qoi`/`webp_to_bmp`/`png_to_tiff` exact RGB vs decoder reference; fastconv lossy min 33.80 dB / mean ≥ 33.95 dB across rows, no output-size inflation); NVENC HEIC 20/20 (3840x2160, explicit GPU path). Audio 34/34 ffprobe + 18/18 content checks (bit-exact PCM for FLAC/AIFF/ALAC single-file and for all 9 batch inputs; lag-aligned SNR Opus 37.1 / Ogg 35.8 / MP3 64.3 / AAC 26.4 / M4A 26.4 dB single-file, 111.9/111.0/104.3/109.7/109.0 dB batch; no output-size inflation or quality regression). CPU HEIC: single 1/1 + batch 20/20 — `ftypheic` major brand, ffmpeg tile-grid reconstruction at 3840x2160, clean `rgb24` decode (24,883,200 B/image), PSNR 35.45 dB single / 35.17–35.53 dB batch (mean 35.39) vs source PNG — retained, not re-run. Video: NVENC 4/4 (600 frames / 10 s, 1920x1080, B-frames 295, `dts_ok=true`, clean decode, H.264 50.72 dB / HEVC 47.62 dB, outputs 25.86 / 20.52 MB, explicit `gpu:nvdec+nvenc on NVIDIA GeForce RTX 2060`, no CPU fallback); remux 4/4 (codec-copy `aac`+`h264`, 1920x1080, 600 frames, no decoder errors).
- Regression verdict: **no throughput or quality regression caused by this shrink change.** Every quality/size/validation constant matches the prior snapshot exactly; image/NVENC/remux timings differ only within run-to-run noise (e.g. NVENC H.264 1884.5 vs 1869.6 ms with ffmpeg 1984.0 vs 1952.8 ms; remux mkv→mp4 24.1 vs 21.9 ms with ffmpeg 85.5 vs 81.1 ms). The one material delta is `audio_batch_mixed_to_flac` 229.4 vs the stale 109.8 ms: it predates this branch and comes from the already-merged post-#34-review worker budget — the AAC ADTS input exposes no `total_samples`, so `resolve_worker_count` charges the full 512 MiB per-file allowance and caps the mixed batch to **1 worker**; omitting only that input restores 8 workers and 108.4 ms, and an isolated build of unmodified HEAD (thin LTO, no working-tree changes) reproduced 1 worker / 217–238 ms. The fresh audio table is authoritative for post-review code; no runs were discarded or cherry-picked (two serial audio runs: 228.4 / 229.4 ms batch, all quality checks 18/18 in both).
- Single-image CPU HEIC costs ~6 s, so the 20-image batch used a 600 s guard (3 serial runs ≈ 6.1 min total). No runs were discarded or cherry-picked.

### Image — 20× 4K, Q=80 / same compression class (20/20 per batch row; CPU HEIC single 1/1 retained from prior snapshot)
| Workload | Tool | Time (ms) | Throughput | Peak RAM | Input | Output | Quality | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `png_to_webp` | `fastconv (cpu q=80)` | 4177.0 | 72.72 MB/s | 659.1 MiB | 303.75 MB | 5.69 MB | 33.95 dB | **3.16x** |
| `png_to_webp` | `ffmpeg (libwebp q=80)` | 13216.1 | 22.98 MB/s | 666.2 MiB | 303.75 MB | 5.34 MB | 33.61 dB | Baseline |
| `png_to_jpeg` | `fastconv (cpu q=80)` | 566.1 | 536.56 MB/s | 431.7 MiB | 303.75 MB | 13.39 MB | 34.30 dB | **1.54x** |
| `png_to_jpeg` | `ffmpeg (mjpeg qv=3)` | 873.9 | 347.59 MB/s | 866.3 MiB | 303.75 MB | 8.58 MB | 33.98 dB | Baseline |
| `jpeg_to_webp` | `fastconv (cpu q=80)` | 3961.5 | 6.78 MB/s | 422.1 MiB | 26.86 MB | 5.90 MB | 38.12 dB | **3.10x** |
| `jpeg_to_webp` | `ffmpeg (libwebp q=80)` | 12262.1 | 2.19 MB/s | 165.6 MiB | 26.86 MB | 5.44 MB | 38.40 dB | Baseline |
| `webp_to_png` | `fastconv (cpu png)` | 582.5 | 9.76 MB/s | 473.2 MiB | 5.69 MB | 153.64 MB | exact | **3.65x** |
| `webp_to_png` | `ffmpeg (png)` | 2124.3 | 2.68 MB/s | 820.4 MiB | 5.69 MB | 89.97 MB | exact | Baseline |
| `png_to_qoi` | `fastconv (cpu qoi)` | 622.7 | 487.77 MB/s | 521.9 MiB | 303.75 MB | 250.91 MB | exact | **1.64x** |
| `png_to_qoi` | `ffmpeg (qoi)` | 1020.6 | 297.60 MB/s | 1015.7 MiB | 303.75 MB | 250.91 MB | exact | Baseline |
| `webp_to_bmp` | `fastconv (cpu bmp)` | 366.4 | 15.52 MB/s | 606.5 MiB | 5.69 MB | 497.67 MB | exact | **1.25x** |
| `webp_to_bmp` | `ffmpeg (bmp)` | 459.4 | 12.38 MB/s | 484.6 MiB | 5.69 MB | 497.67 MB | exact | Baseline |
| `png_to_tiff` | `fastconv (cpu tiff)` | 830.9 | 365.55 MB/s | 600.9 MiB | 303.75 MB | 276.93 MB | exact (both Deflate) | **3.08x** |
| `png_to_tiff` | `ffmpeg (tiff)` | 2558.4 | 118.72 MB/s | 1401.4 MiB | 303.75 MB | 373.65 MB | exact | Baseline |
| `png_to_jxl` | `fastconv (cpu q=80)` | 3794.8 | 80.04 MB/s | 1395.2 MiB | 303.75 MB | 1.25 MB | 34.80 dB | **4.83x** (same distance 1.9) |
| `png_to_jxl` | `ffmpeg (libjxl d=1.9)` | 18315.4 | 16.58 MB/s | 897.1 MiB | 303.75 MB | 6.49 MB | 34.95 dB | Baseline |
| `png_to_avif` | `fastconv (cpu q=80)` | 2290.3 | 132.62 MB/s | 1348.7 MiB | 303.75 MB | 0.64 MB | 35.65 dB | **1.75x** (−0.10 dB, 4.5x smaller) |
| `png_to_avif` | `ffmpeg (libsvtav1 preset 8)` | 4004.6 | 75.85 MB/s | 2377.5 MiB | 303.75 MB | 2.86 MB | 35.75 dB | Baseline |
| `png_to_heic` | `fastconv (gpu:nvenc q=80)` | 974.5 | 311.69 MB/s | 711.8 MiB | 303.75 MB | 27.47 MB | 39.21 dB | **13.56x** vs ff(webp) / **4.29x** vs cpu(webp) |
| `jpeg_to_heic` | `fastconv (gpu:nvenc q=80)` | 901.0 | 29.81 MB/s | 522.4 MiB | 26.86 MB | 19.24 MB | 38.16 dB | **13.61x** vs ff(webp) / **4.40x** vs cpu(webp) |
| `webp_to_heic` | `fastconv (gpu:nvenc q=80)` | 902.9 | 6.30 MB/s | 848.7 MiB | 5.69 MB | 13.21 MB | 38.10 dB | NVENC 20× 4K (no ffmpeg HEIC opponent) |
| `png_to_heic` (single 4K) | `fastconv (cpu heifer q=80, 3 runs)` | 6251.9 | 2.43 MB/s | 135.8 MiB | 15.18 MB | 1.09 MB | 35.45 dB | CPU-only (no ffmpeg HEIC opponent) |
| `png_to_heic` (20× 4K batch) | `fastconv (cpu heifer q=80, 8 workers, 3 runs)` | 122310.5 | 2.48 MB/s | 467.6 MiB | 303.75 MB | 21.92 MB | 35.39 dB mean | CPU-only (no ffmpeg HEIC opponent) |

### Audio — 60 s stereo 48 kHz/16-bit, target-matched bitrate/quality (achieved size/lag-aligned SNR per row; 34/34 outputs validated: 16 single-file + 18 batch; two serial runs, table = second run)
| Workload | Tool | Time (ms) | Throughput | Peak RAM | Input | Output | Quality | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wav_to_flac` | `fastconv (flacenc pure rust)` | 49.3 | 233.44 MB/s | 71.0 MiB | 11.52 MB | 4.12 MB | bit-exact | **3.84x** |
| `wav_to_flac` | `ffmpeg (-c:a flac -compression_level 5)` | 189.3 | 60.86 MB/s | 57.4 MiB | 11.52 MB | 4.11 MB | bit-exact | Baseline |
| `wav_to_aiff` | `fastconv (native pcm_s16be)` | 8.9 | 1299.26 MB/s | 7.6 MiB | 11.52 MB | 11.52 MB | bit-exact | **10.81x** |
| `wav_to_aiff` | `ffmpeg (-c:a pcm_s16be)` | 95.9 | 120.16 MB/s | 52.0 MiB | 11.52 MB | 11.52 MB | bit-exact | Baseline |
| `wav_to_alac` | `fastconv (alac-encoder m4a)` | 96.4 | 119.45 MB/s | 32.7 MiB | 11.52 MB | 4.48 MB | bit-exact | **3.76x** |
| `wav_to_alac` | `ffmpeg (-c:a alac)` | 362.4 | 31.79 MB/s | 52.6 MiB | 11.52 MB | 4.69 MB | bit-exact | Baseline |
| `wav_to_opus` | `fastconv (opus-pure)` | 148.1 | 77.80 MB/s | 72.7 MiB | 11.52 MB | 0.98 MB | 37.1 dB | **3.96x** |
| `wav_to_opus` | `ffmpeg (-c:a libopus -b:a 128k)` | 586.9 | 19.63 MB/s | 55.2 MiB | 11.52 MB | 1.35 MB | — | Baseline |
| `wav_to_ogg` | `fastconv (vorbis-enc q4)` | 861.2 | 13.38 MB/s | 14.1 MiB | 11.52 MB | 0.74 MB | 35.8 dB | **1.08x** |
| `wav_to_ogg` | `ffmpeg (-c:a libvorbis -q:a 4)` | 926.9 | 12.43 MB/s | 55.7 MiB | 11.52 MB | 0.70 MB | — | Baseline |
| `wav_to_mp3` | `fastconv (rusty_mp3 192k)` | 599.0 | 19.23 MB/s | 16.4 MiB | 11.52 MB | 1.44 MB | 64.3 dB | **1.67x** |
| `wav_to_mp3` | `ffmpeg (-c:a libmp3lame -b:a 192k)` | 1000.2 | 11.52 MB/s | 55.2 MiB | 11.52 MB | 1.44 MB | — | Baseline |
| `wav_to_aac` | `fastconv (rusty_aac 128k)` | 143.9 | 80.05 MB/s | 59.3 MiB | 11.52 MB | 0.98 MB | 26.4 dB | **11.23x** |
| `wav_to_aac` | `ffmpeg (-c:a aac -b:a 128k)` | 1616.2 | 7.13 MB/s | 55.3 MiB | 11.52 MB | 0.98 MB | — | Baseline |
| `wav_to_m4a` | `fastconv (rusty_aac 128k muxed)` | 147.7 | 77.98 MB/s | 63.8 MiB | 11.52 MB | 0.97 MB | 26.4 dB | **11.02x** |
| `wav_to_m4a` | `ffmpeg (-c:a aac -b:a 128k)` | 1627.6 | 7.08 MB/s | 55.8 MiB | 11.52 MB | 0.97 MB | — | Baseline |
| `audio_batch_mixed_to_flac` | `fastconv (parallel, 9 inputs, 1 worker budgeted)` | 229.4 | 26.34 MB/s | 55.7 MiB | 6.04 MB | 5.65 MB | 9/9 validated | **3.70x** |
| `audio_batch_mixed_to_flac` | `ffmpeg (sequential, 9 inputs)` | 848.4 | 7.12 MB/s | 60.4 MiB | 6.04 MB | 6.83 MB | 9/9 validated | Baseline |

### Video — NVENC & Remux, 1080p60, 600 frames, 3 serial runs/side
| Workload | Tool | Time (ms) | FPS | Peak RAM | Output | PSNR | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nvenc_h264_mp4` | `fastconv (nvdec+nvenc)` | 1884.5 | 318.4 | 480.0 MiB | 25.86 MB | 50.72 dB | **1.05x** |
| `nvenc_h264_mp4` | `ffmpeg (h264_nvenc, hwaccel cuda)` | 1984.0 | 302.4 | 631.6 MiB | 25.90 MB | 50.73 dB | Baseline |
| `nvenc_hevc_mp4` | `fastconv (nvdec+nvenc)` | 2011.4 | 298.3 | 591.6 MiB | 20.52 MB | 47.62 dB | **1.03x** |
| `nvenc_hevc_mp4` | `ffmpeg (hevc_nvenc, hwaccel cuda)` | 2062.6 | 290.9 | 580.9 MiB | 20.54 MB | 47.62 dB | Baseline |
| `remux_mp4_to_mkv` | `fastconv (pure rust)` | 13.2 | 45462.9 | 16.6 MiB | 19.16 MB | codec-copy | **6.80x** |
| `remux_mp4_to_mkv` | `ffmpeg (-c copy)` | 89.8 | 6683.4 | 60.4 MiB | 19.14 MB | codec-copy | Baseline |
| `remux_mkv_to_mp4` | `fastconv (pure rust)` | 24.1 | 24944.7 | 55.8 MiB | 19.15 MB | codec-copy | **3.55x** |
| `remux_mkv_to_mp4` | `ffmpeg (-c copy)` | 85.5 | 7017.1 | 57.4 MiB | 19.15 MB | codec-copy | Baseline |

### Animation — 480×270 / 15 fps / 60 frames, 3 serial runs (median time, max VmHWM), 12/12 validated (retained unchanged from prior snapshot)
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
- Audio lossy rows are target-matched, not result-matched: Opus uses 128 kbps on both sides but VBR complexity 9 (fastconv) vs 10 (libopus), so achieved sizes differ (0.98 vs 1.35 MB); per-row lag-aligned SNR is fastconv-only where ffmpeg shows `—`.
- WebP animation-input PSNR lag: 29.79 vs 48.94 dB (CPU), 29.63 vs 36.56 dB (NVENC) — known `zenwebp` vs `libwebp` color-conversion delta (still-image repro 30.1 dB); not bought with lower compression (0.30 vs 0.27 MB).
- Turing AV1 is CPU-decoded; `--hwdecode nvdec` is rejected with no fallback. CPU HEVC *video* encode is rejected; still-image HEIC is available on CPU (`heifer`) and on NVENC GPU (image output remains HEIC-only).
- GPU probe is Linux-only (DRM sysfs / NVIDIA procfs); other OSes report no adapters/backends. Animation bench not yet in `xtask` (external 3-run scripts); retained unchanged for this binary/codebase-shrink feature.
- `audio_batch_mixed_to_flac` no longer shows the stale 109.8 ms / 7.53x pre-review number. Merged post-#34 worker budgeting sees no `total_samples` for the AAC ADTS input, reserves the full 512 MiB per-file PCM allowance and caps the 9-file mixed batch at 1 worker (55.7 MiB peak). Removing only that input measures 108.4 ms / 8 workers; unmodified HEAD reproduces 1 worker / 217–238 ms. Quality is unchanged (9/9 bit-exact or SNR-validated tracks); this is a stale-baseline correction, not a Feature 13 regression.
- Audio zero-quality-loss: FLAC/AIFF/ALAC remain bit-exact (5,760,000 samples identical single-file, 9/9 batch tracks decoded; wav/flac/alac/aiff batch tracks bit-identical), lossy single-file SNR (37.1/35.8/64.3/26.4/26.4 dB) and every output size match the prior snapshot exactly. Single-file timings match prior within noise (49.3/8.9/96.4/148.1/861.2/599.0/143.9/147.7 ms). Smallest margin: `wav_to_ogg` 861.2 vs 926.9 ms (1.08x). Both serial audio runs agreed (batch 228.4 / 229.4 ms); no samples were discarded.
- CPU HEIC throughput is ~2.4 MB/s on this 4C/8T host: `heifer`'s Medium HEVC-Intra preset is intentionally kept (no preset cheating), and the 20-image batch caps at 8 image workers to bound RAM (peak 467.6 MiB, far below 3 GiB). The NVENC path encodes the same 20-image batch in 974.5 ms, but that is a different quality class (27.47 MB / 39.21 dB vs 21.92 MB / 35.39 dB CPU), not a fair speedup comparison.
- `ffprobe` lists CPU-HEIC output as 40 × 512×512 HEVC tile items + a 320×180 thumbnail rather than one 3840×2160 stream; full resolution is confirmed via ffmpeg's `Tile Grid: ... 3840x2160` group and the clean 24,883,200-byte RGB24 decode.
- AVIF/JXL matched PSNR and smaller outputs; TIFF both Deflate, exact; AVIF peak RAM 1348.7 vs 2377.5 MiB (fastconv lower; both under 3 GiB), JXL 1395.2 vs 897.1 MiB.
- Absolute timings on short rows shift run-to-run on this snapshot for both sides; the large gains are structural, not sampling noise. Image speedups vs ffmpeg: 3.16x webp, 1.54x jpeg, 3.10x jpeg→webp, 3.65x webp→png, 1.64x qoi, 1.25x bmp, 3.08x tiff, 4.83x JXL, 1.75x AVIF; NVENC HEIC 13.56x/13.61x vs ffmpeg(webp) with 20/20 decoded 4K outputs. FFmpeg opponent timings also shifted (e.g. png_to_webp 13216.1 vs 12612.6 ms), so fastconv's 4177.0 vs 4074.3 ms is run-to-run noise, not a fat-LTO regression. All four lossless image rows remain exact RGB; lossy rows keep PSNR in the same band (e.g. png_to_jpeg 34.30 vs 34.30 dB, png_to_avif 35.65 vs 35.65 dB) with no output-size inflation.
- Video timings are within noise of the prior snapshot: NVENC H.264 1884.5 vs 1869.6 ms (1.05x vs 1.04x), HEVC 2011.4 vs 1988.3 ms (1.03x both); remux mp4→mkv 13.2 vs 13.3 ms (6.80x vs 6.42x) and mkv→mp4 24.1 vs 21.9 ms (3.55x vs 3.70x, ffmpeg also faster then at 81.1 vs 85.5 ms). Output sizes, 50.72/47.62 dB PSNR, 600-frame validation, B-frames 295, `dts_ok=true`, clean decode and the explicit `gpu:nvdec+nvenc on NVIDIA GeForce RTX 2060` proof are unchanged in kind; no runs were discarded or cherry-picked.
