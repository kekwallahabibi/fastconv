# fastconv — state at a glance
> Update in-place on changes; git history preserves past snapshots.

## Snapshot
| Item | Value |
| :--- | :--- |
| Product | `fastconv` 0.1.0 · edition 2024 · AGPL-3.0-only · HEAD `f53e03d` + working-tree image speedup change `fa7c0d7b` (pre-merge / pre-review branch `perf/speedup-image-zero-loss`) · 5 crates |
| Verify | `cargo xtask verify` green — **687** default tests (33 cli + 3 audio_codecs + 11 cli_help + 99 domain + 391 pipeline + 128 probe + 22 xtask), **458** `--features nvenc` tests (33 cli + 3 audio_codecs + 11 cli_help + 411 pipeline); **273** deps license-checked; fmt + clippy `-D warnings` clean (default and nvenc) |
| Release binary | 37.5 MiB default (unstripped); ffmpeg ~38 MB incl. libs |

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

## Benchmarks — canonical 2026-11-27 (pre-merge / pre-review branch runs)
Host: `Intel Core i7-6700` (4C/8T AVX2/FMA, 32 GiB) + `RTX 2060` (6 GB, driver `620.73.04`); `ffmpeg n9.1`; `rustc`/`cargo 1.100.0`; all runs serial.

- Provenance (image — this snapshot): HEAD `f53e03d0bfd936b2422ff376b0fa866270028ff8`; code `git diff | sha256sum` = `fa7c0d7bc5f9bf8a596e23b900980d6b03da4b40f7648fbaae730849c459137c` (zero-quality-loss image speedups: `crates/pipeline/src/{avif,bmp,cpu,decode,heic,qoi}.rs`, `crates/pipeline/src/gpu/heic.rs`; `STATE.md` updated afterwards).
- Provenance (audio rows, retained unchanged): HEAD `fbfdf2203125e8bdcfb5a1f1b3abaaf85936ec8f`; code `git diff | sha256sum` = `d9b40d3be3048e066c2a8efaf8fed5580994c11b932869dca5a36dd8b74254e0` (zero-quality-loss audio speedups: `crates/pipeline/src/{aiff,audio,m4a,ogg_flac}.rs`, `crates/probe/src/audio.rs`). This image-only change does not touch those code paths; their numbers are kept verbatim.
- Provenance (NVENC video / animation / CPU-HEIC rows, retained unchanged): HEAD `c7b36320788965d66b6a465bf3c284d29fe0fe83`; code `git diff | sha256sum` = `fd5e27789ab56fc897ce9a41b637d860e86fc093dc2d32c53ec27e112b09eff6` (CPU-HEIC feature diff). Remux and NVENC video were re-run as serial regression checks on this branch and reproduced the canonical numbers within noise (remux 15.3/83.5/24.0/83.3 ms; NVENC 1877.7/1985.7 ms; 8/8 ffprobe validated, no decoder errors, explicit GPU path); the table rows stay verbatim.
- Commands: `cargo xtask gen-fixtures`; `cargo build --release -p fastconv-cli --features nvenc`; `timeout -k 10 600s cargo xtask bench`; `timeout -k 10 180s cargo xtask bench --remux`; `timeout -k 10 180s cargo xtask bench --nvenc`; `cargo xtask verify`. CPU HEIC (no `xtask` target, not re-run for this image-speedup feature — `heifer` preset unchanged): retained 3 serial runs each of `target/release/fastconv convert <src> <dst> --to heic --backend cpu --quality 80` — single `fixtures/images_4k/img_01.png` and 20-image `fixtures/images_4k/` batch. Wall time + exact peak RSS captured with `wait4` rusage (VmHWM equivalent; `/usr/bin/time` is unavailable on this host); median time and max peak RSS reported. Earlier (retained) runs: `timeout -k 10 220s cargo xtask bench --audio`; animation via external 3-run scripts.
- Validation: image 20/20 per batch row (21 harness rows, all 3840x2160; lossless rows `webp_to_png`/`png_to_qoi`/`webp_to_bmp`/`png_to_tiff` exact RGB vs decoder reference; fastconv lossy rows ≥ 33.80 dB min / ≥ 33.95 dB mean, no output-size inflation); NVENC HEIC 20/20 (3840x2160, explicit GPU path). Audio 34/34 ffprobe + 18/18 content checks (bit-exact PCM for FLAC/AIFF/ALAC single-file and for all 9 batch inputs 9/9; lag-aligned SNR Opus 37.1 / Ogg 35.8 / MP3 64.3 / AAC 26.4 / M4A 26.4 dB single-file, 111.9/111.0/104.3/109.7/109.0 dB batch; no output-size inflation or quality regression) — retained. CPU HEIC: single 1/1 + batch 20/20 — `ftypheic` major brand, ffmpeg tile-grid reconstruction at 3840x2160, full clean `rgb24` decode (24,883,200 B/image), PSNR 35.45 dB single / 35.17–35.53 dB batch (mean 35.39) vs source PNG; all ≥ 25 dB — retained (not re-run). NVENC video 4/4 (600 frames / 10 s, clean decode, HEVC 47.62 dB, explicit GPU path); remux 4/4 (codec-copy `aac`+`h264`, 600 frames, no decoder errors) — re-run as regression checks.
- Animation: 2026-11-26, `6f7f14a` (PR #21, no `xtask` target; not re-run for this image-speedup feature).
- Single-image CPU HEIC costs ~6 s, so the 20-image batch used a 600 s guard (3 serial runs ≈ 6.1 min total). No runs were discarded or cherry-picked.

### Image — 20× 4K, Q=80 / same compression class (20/20 per batch row; CPU HEIC single 1/1 retained from prior snapshot)
| Workload | Tool | Time (ms) | Throughput | Peak RAM | Input | Output | Quality | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `png_to_webp` | `fastconv (cpu q=80)` | 4074.3 | 74.55 MB/s | 635.2 MiB | 303.75 MB | 5.69 MB | 33.95 dB | **3.10x** |
| `png_to_webp` | `ffmpeg (libwebp q=80)` | 12612.6 | 24.08 MB/s | 665.2 MiB | 303.75 MB | 5.34 MB | 33.61 dB | Baseline |
| `png_to_jpeg` | `fastconv (cpu q=80)` | 560.9 | 541.58 MB/s | 432.7 MiB | 303.75 MB | 13.39 MB | 34.30 dB | **1.50x** |
| `png_to_jpeg` | `ffmpeg (mjpeg qv=3)` | 840.9 | 361.23 MB/s | 849.8 MiB | 303.75 MB | 8.58 MB | 33.98 dB | Baseline |
| `jpeg_to_webp` | `fastconv (cpu q=80)` | 3833.9 | 7.01 MB/s | 428.4 MiB | 26.86 MB | 5.90 MB | 38.12 dB | **3.04x** |
| `jpeg_to_webp` | `ffmpeg (libwebp q=80)` | 11637.3 | 2.31 MB/s | 165.8 MiB | 26.86 MB | 5.44 MB | 38.40 dB | Baseline |
| `webp_to_png` | `fastconv (cpu png)` | 531.9 | 10.69 MB/s | 498.4 MiB | 5.69 MB | 153.64 MB | exact | **3.93x** |
| `webp_to_png` | `ffmpeg (png)` | 2092.9 | 2.72 MB/s | 820.5 MiB | 5.69 MB | 89.97 MB | exact | Baseline |
| `png_to_qoi` | `fastconv (cpu qoi)` | 675.0 | 449.98 MB/s | 536.6 MiB | 303.75 MB | 250.91 MB | exact | **1.45x** |
| `png_to_qoi` | `ffmpeg (qoi)` | 978.5 | 310.41 MB/s | 948.8 MiB | 303.75 MB | 250.91 MB | exact | Baseline |
| `webp_to_bmp` | `fastconv (cpu bmp)` | 340.9 | 16.68 MB/s | 606.0 MiB | 5.69 MB | 497.67 MB | exact | **1.28x** |
| `webp_to_bmp` | `ffmpeg (bmp)` | 436.4 | 13.03 MB/s | 484.4 MiB | 5.69 MB | 497.67 MB | exact | Baseline |
| `png_to_tiff` | `fastconv (cpu tiff)` | 844.3 | 359.75 MB/s | 590.6 MiB | 303.75 MB | 276.93 MB | exact (both Deflate) | **3.00x** |
| `png_to_tiff` | `ffmpeg (tiff)` | 2532.2 | 119.96 MB/s | 1436.9 MiB | 303.75 MB | 373.65 MB | exact | Baseline |
| `png_to_jxl` | `fastconv (cpu q=80)` | 3838.9 | 79.12 MB/s | 1387.5 MiB | 303.75 MB | 1.25 MB | 34.80 dB | **4.63x** (same distance 1.9) |
| `png_to_jxl` | `ffmpeg (libjxl d=1.9)` | 17778.1 | 17.09 MB/s | 907.5 MiB | 303.75 MB | 6.49 MB | 34.95 dB | Baseline |
| `png_to_avif` | `fastconv (cpu q=80)` | 2337.8 | 129.93 MB/s | 1328.9 MiB | 303.75 MB | 0.64 MB | 35.65 dB | **1.62x** (−0.10 dB, 4.5x smaller) |
| `png_to_avif` | `ffmpeg (libsvtav1 preset 8)` | 3781.9 | 80.32 MB/s | 2581.8 MiB | 303.75 MB | 2.86 MB | 35.75 dB | Baseline |
| `png_to_heic` | `fastconv (gpu:nvenc q=80)` | 938.6 | 323.63 MB/s | 695.1 MiB | 303.75 MB | 27.47 MB | 39.21 dB | **13.44x** vs ff(webp) / **4.34x** vs cpu(webp) |
| `jpeg_to_heic` | `fastconv (gpu:nvenc q=80)` | 871.2 | 30.83 MB/s | 523.6 MiB | 26.86 MB | 19.24 MB | 38.16 dB | **13.36x** vs ff(webp) / **4.40x** vs cpu(webp) |
| `webp_to_heic` | `fastconv (gpu:nvenc q=80)` | 898.4 | 6.33 MB/s | 805.3 MiB | 5.69 MB | 13.21 MB | 38.10 dB | NVENC 20× 4K (no ffmpeg HEIC opponent) |
| `png_to_heic` (single 4K) | `fastconv (cpu heifer q=80, 3 runs)` | 6251.9 | 2.43 MB/s | 135.8 MiB | 15.18 MB | 1.09 MB | 35.45 dB | CPU-only (no ffmpeg HEIC opponent) |
| `png_to_heic` (20× 4K batch) | `fastconv (cpu heifer q=80, 8 workers, 3 runs)` | 122310.5 | 2.48 MB/s | 467.6 MiB | 303.75 MB | 21.92 MB | 35.39 dB mean | CPU-only (no ffmpeg HEIC opponent) |

### Audio — 60 s stereo 48 kHz/16-bit, target-matched bitrate/quality (achieved size/lag-aligned SNR per row; 34/34 outputs validated: 16 single-file + 18 batch)
| Workload | Tool | Time (ms) | Throughput | Peak RAM | Input | Output | Quality | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wav_to_flac` | `fastconv (flacenc pure rust)` | 49.0 | 235.03 MB/s | 71.3 MiB | 11.52 MB | 4.12 MB | bit-exact | **3.87x** |
| `wav_to_flac` | `ffmpeg (-c:a flac -compression_level 5)` | 189.8 | 60.68 MB/s | 57.4 MiB | 11.52 MB | 4.11 MB | bit-exact | Baseline |
| `wav_to_aiff` | `fastconv (native pcm_s16be)` | 11.1 | 1041.95 MB/s | 10.5 MiB | 11.52 MB | 11.52 MB | bit-exact | **8.47x** |
| `wav_to_aiff` | `ffmpeg (-c:a pcm_s16be)` | 93.7 | 122.97 MB/s | 51.9 MiB | 11.52 MB | 11.52 MB | bit-exact | Baseline |
| `wav_to_alac` | `fastconv (alac-encoder m4a)` | 99.8 | 115.39 MB/s | 35.8 MiB | 11.52 MB | 4.48 MB | bit-exact | **3.53x** |
| `wav_to_alac` | `ffmpeg (-c:a alac)` | 352.3 | 32.70 MB/s | 50.7 MiB | 11.52 MB | 4.69 MB | bit-exact | Baseline |
| `wav_to_opus` | `fastconv (opus-pure)` | 148.1 | 77.80 MB/s | 73.7 MiB | 11.52 MB | 0.98 MB | 37.1 dB | **3.90x** |
| `wav_to_opus` | `ffmpeg (-c:a libopus -b:a 128k)` | 576.7 | 19.97 MB/s | 57.3 MiB | 11.52 MB | 1.35 MB | — | Baseline |
| `wav_to_ogg` | `fastconv (vorbis-enc q4)` | 846.3 | 13.61 MB/s | 15.6 MiB | 11.52 MB | 0.74 MB | 35.8 dB | **1.06x** |
| `wav_to_ogg` | `ffmpeg (-c:a libvorbis -q:a 4)` | 895.8 | 12.86 MB/s | 55.9 MiB | 11.52 MB | 0.70 MB | — | Baseline |
| `wav_to_mp3` | `fastconv (rusty_mp3 192k)` | 648.3 | 17.77 MB/s | 17.2 MiB | 11.52 MB | 1.44 MB | 64.3 dB | **1.52x** |
| `wav_to_mp3` | `ffmpeg (-c:a libmp3lame -b:a 192k)` | 982.5 | 11.73 MB/s | 55.4 MiB | 11.52 MB | 1.44 MB | — | Baseline |
| `wav_to_aac` | `fastconv (rusty_aac 128k)` | 143.5 | 80.25 MB/s | 70.9 MiB | 11.52 MB | 0.98 MB | 26.4 dB | **11.02x** |
| `wav_to_aac` | `ffmpeg (-c:a aac -b:a 128k)` | 1582.2 | 7.28 MB/s | 56.0 MiB | 11.52 MB | 0.98 MB | — | Baseline |
| `wav_to_m4a` | `fastconv (rusty_aac 128k muxed)` | 144.5 | 79.74 MB/s | 70.7 MiB | 11.52 MB | 0.97 MB | 26.4 dB | **10.88x** |
| `wav_to_m4a` | `ffmpeg (-c:a aac -b:a 128k)` | 1571.8 | 7.33 MB/s | 54.6 MiB | 11.52 MB | 0.97 MB | — | Baseline |
| `audio_batch_mixed_to_flac` | `fastconv (parallel, 9 inputs)` | 109.8 | 55.06 MB/s | 152.8 MiB | 6.04 MB | 5.65 MB | 9/9 validated | **7.53x** |
| `audio_batch_mixed_to_flac` | `ffmpeg (sequential, 9 inputs)` | 826.4 | 7.31 MB/s | 60.5 MiB | 6.04 MB | 6.83 MB | 9/9 validated | Baseline |

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
- Audio lossy rows are target-matched, not result-matched: Opus uses 128 kbps on both sides but VBR complexity 9 (fastconv) vs 10 (libopus), so achieved sizes differ (0.98 vs 1.35 MB); per-row lag-aligned SNR is fastconv-only where ffmpeg shows `—`.
- WebP animation-input PSNR lag: 29.79 vs 48.94 dB (CPU), 29.63 vs 36.56 dB (NVENC) — known `zenwebp` vs `libwebp` color-conversion delta (still-image repro 30.1 dB); not bought with lower compression (0.30 vs 0.27 MB).
- Turing AV1 is CPU-decoded; `--hwdecode nvdec` is rejected with no fallback. CPU HEVC *video* encode is rejected; still-image HEIC is available on CPU (`heifer`) and on NVENC GPU (image output remains HEIC-only).
- GPU probe is Linux-only (DRM sysfs / NVIDIA procfs); other OSes report no adapters/backends. Animation bench not yet in `xtask` (external 3-run scripts); not re-run for this image-speedup feature.
- Absolute timings on short rows shift run-to-run on this snapshot for both sides; the large gains here are structural, not sampling noise: `wav_to_flac` 49.0 vs 189.8 ms (was 133.8 vs 211.8 ms), `wav_to_aiff` 11.1 vs 93.7 ms (was 32.4 vs 124.8 ms), `wav_to_alac` 99.8 vs 352.3 ms (was 149.6 vs 357.8 ms), `audio_batch_mixed_to_flac` 109.8 vs 826.4 ms (was 167.5 vs 887.8 ms). Opus/Ogg/MP3/AAC/M4A fastconv times are unchanged within noise (148.1/846.3/648.3/143.5/144.5 ms vs 152.5/844.1/650.9/150.3/154.4 ms prior). Output sizes, PSNR/SNR and every validation check are unchanged in kind; no runs were discarded. Smallest margin: `wav_to_ogg` 846.3 vs 895.8 ms (1.06x), down from 1.09x only because this ffmpeg run was faster (895.8 vs 916.9 ms) — fastconv's Ogg time did not regress.
- Audio zero-quality-loss: FLAC/AIFF/ALAC remain bit-exact (5,760,000 samples identical single-file, 9/9 batch tracks bit-identical); lossy single-file SNR (37.1/35.8/64.3/26.4/26.4 dB) and every output size match the prior snapshot exactly. Batch peak RAM rose to 152.8 MiB (was 82.2 MiB), still far below the 1 GiB batch PCM cap. Remux regression re-run reproduced canonical times (15.4/85.5/23.9/83.3 ms) with 4/4 clean ffprobe checks.
- CPU HEIC throughput is ~2.4 MB/s on this 4C/8T host: `heifer`'s Medium HEVC-Intra preset is intentionally kept (no preset cheating), and the 20-image batch caps at 8 image workers to bound RAM (peak 467.6 MiB, far below 3 GiB). The NVENC path encodes the same 20-image batch in 938.6 ms, but that is a different quality class (27.47 MB / 39.21 dB vs 21.92 MB / 35.39 dB CPU), not a fair speedup comparison.
- `ffprobe` lists CPU-HEIC output as 40 × 512×512 HEVC tile items + a 320×180 thumbnail rather than one 3840×2160 stream; full resolution is confirmed via ffmpeg's `Tile Grid: ... 3840x2160` group and the clean 24,883,200-byte RGB24 decode.
- AVIF/JXL matched PSNR and smaller outputs; TIFF both Deflate, exact; AVIF peak RAM 1328.9 vs 2581.8 MiB (fastconv lower; both under 3 GiB), JXL 1387.5 vs 907.5 MiB.
- Image zero-quality-loss speedups (this snapshot vs retained prior numbers): `png_to_jpeg` 560.9 vs 657.9 ms (1.50x vs 1.30x), `png_to_avif` 2337.8 vs 2710.3 ms (1.62x vs 1.29x), `webp_to_bmp` 340.9 vs 360.4 ms (1.28x vs 1.24x), `png_to_tiff` 844.3 vs 880.4 ms (3.00x vs 2.83x), `png_to_webp` 4074.3 vs 4204.7 ms (3.10x vs 3.00x), `jpeg_to_webp` 3833.9 vs 3985.3 ms (3.04x vs 2.93x), `webp_to_png` 531.9 vs 543.2 ms (3.93x vs 3.74x), `png_to_qoi` 675.0 vs 737.9 ms (1.45x vs 1.54x — fastconv improved; ffmpeg also faster this run, 978.5 vs 1136.7 ms), `png_to_jxl` 3838.9 vs 3853.3 ms (4.63x vs 4.71x — within noise). NVENC HEIC 938.6/871.2/898.4 ms vs 947.4/844.6/881.6 ms (+3.2% / +1.9% on the two smaller batches, within run-to-run noise; the `gpu/heic.rs` change is I/O-only). All four lossless rows remain exact RGB; lossy rows keep PSNR in the same band (e.g. `png_to_jpeg` 34.30 vs 34.29 dB, `png_to_avif` 35.65 vs 35.63 dB) with no output-size inflation. No runs were discarded or cherry-picked.
