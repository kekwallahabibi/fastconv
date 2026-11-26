# fastconv — state at a glance
> Update in-place on changes; git history preserves past snapshots.

## Snapshot
| Item | Value |
| :--- | :--- |
| Product | `fastconv` 0.1.0 · edition 2024 · AGPL-3.0-only · HEAD `f523cef` (PR #22) · 5 crates |
| Verify | `cargo xtask verify` green — **456** default tests (25 cli + 9 cli_help + 54 domain + 273 pipeline + 73 probe + 22 xtask), **325** `--features nvenc` tests (25 cli + 9 cli_help + 291 pipeline); **269** deps license-checked |
| Release binary | 38.9 MiB default (unstripped) / 32.5 MiB stripped; ffmpeg ~38 MB incl. libs |

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
- **Image — 11 in × 12 out.** In: `png jpeg webp bmp qoi tiff jxl avif gif ico pnm` (P1–P7); out: same set + `heic`. Still GIF uses a fixed 256-color palette; `--quality`: webp/jpeg/jxl/avif/heic, `--lossless`: webp/jxl + png/bmp/qoi/tiff/ico/pnm; HEIC requires NVENC.
- **Audio — 9 in × 9 out.** `wav flac opus ogg mp3 aac m4a alac aiff` (Ogg = Vorbis, AAC = ADTS, ALAC in M4A); CPU-only, parallel batches.
- **Video — 6 containers** `mp4 mov mkv webm avi ts`; codecs H.264/HEVC (H.265)/AV1/VP9 + AAC/Opus/MP3 + `copy`. CPU: H.264/AV1/VP9/GIF/WebP; NVENC: H.264/HEVC; `--hwdecode` NVDEC zero-copy.
- **Animation — bidirectional** video ↔ animated GIF/WebP via `fastconv video --to gif,webp --vcodec gif,webp` (CPU + NVENC); GIF/WebP also valid video inputs; `remux` rejects them.

## Key crates (Cargo.toml)
| Area | Versions |
| :--- | :--- |
| Image | `png 0.18`, `zune-jpeg 0.5.16-rc3`, `zenwebp 0.4.6`, `tiff 0.11.3`, `jxl 0.8.0`, `jixel 0.5.13`, `zenavif 0.1.6`, `zenavif-parse 0.6.2`, `gif 0.14.2`, `ico 0.6.0`, `rav1e 0.8.1`, `avif-serialize 0.8.9`, `jpeg-encoder 0.7.1` |
| Audio | `flacenc 0.5.1`, `hound 3.5.1`, `opus-pure 0.2.5`, `vorbis-enc 0.1.1`, `rusty_mp3 1.2.2`, `rusty_aac 2.2.0`, `alac-encoder 0.3.0`, `symphonia 0.6.1`, `rubato 5.1.0`, `ogg 0.9.2` |
| Video | `rusty_h264 0.17.4`, `rusty_h265 0.7.2`, `rav1d-safe 0.5.7`, `rusty_vp9 0.3.0`, `containr 0.1.1`, `yuv 0.8.21` |
| GPU / misc | `cudarc 0.16`, `libloading 0.9`, `nvidia-video-codec-sdk 0.4` (nvenc feature only), `serde`/`serde_json 1`; no `wgpu` |

## Workspace / module map
| Crate | Sources |
| :--- | :--- |
| `domain` | `src/{lib,audio,hardware,image,metrics,video}.rs` — pure types/invariants |
| `probe` | `src/{lib,audio,hardware,image,video}.rs` — boundary parsing/validation |
| `pipeline` | `src/{lib,audio,aiff,avif,avi,bmp,cpu,decode,gif,ico,m4a,mpegts,pnm,qoi,remux,video_anim,video_bitstream,video_codecs,video_container,video_transcode}.rs` + `src/gpu/{mod,nvdec,nvenc,heic}.rs` |
| `cli` (`fastconv`) | `src/{main,audio,video}.rs`, `tests/cli_help.rs` — thin orchestration |
| `xtask` | `src/{main,bench,audio_bench,nvenc_bench,remux_bench,fixtures}.rs` — verify / fixtures / bench |

## Invariants
- No FFmpeg in `crates/` (scanner in `cargo xtask verify`); ffmpeg only in `xtask` as bench opponent / fixture generator.
- Boundary validation in `probe`; `domain` holds valid types only; illegal states (audio + GPU, quality on lossless formats, CPU HEIC, NVENC non-HEIC) are enum-level (`ExecutionPlan`, `ImageExecutionMode`, `ImageCompression`, `GpuMediaKind`).
- Zero silent fallback: failed GPU setup, `--backend vaapi/vulkan`, CPU HEVC encode, failed NVDEC → hard error.
- Lints: `unsafe_code = "forbid"`, `unwrap_used`/`expect_used`/`panic`/`todo`/`unimplemented`/`dbg_macro` = `deny`, `clippy::pedantic = "warn"`; pipeline relaxes unsafe only under `--features nvenc`, confined to narrowly scoped `#[allow(unsafe_code)]` FFI blocks in `gpu/nvenc.rs` and `gpu/nvdec.rs`.
- AGPL-3.0-only; 269 deps permissive/AGPL-compatible; `GPL-2.0-only` rejected (Classpath exception allowed).
- RAM caps (actual constants): JXL/AVIF ≤8 image workers, AVIF inner threads ≤8, MP4 faststart ≤64 MiB in-memory (stream above), ≤8 animation workers, ≤2 CPU video file workers, 1 GiB decoded/track, 512 MiB PCM/file, 1 GiB PCM/batch.
- Turing reality: NVENC = HEVC-intra (HEIC) only; no HW encoders for webp/jpeg/png/bmp/qoi/tiff/jxl/avif/gif/ico/pnm; AV1 CPU-decode; image Vulkan rejected. GPU probe Linux-only (DRM sysfs + NVIDIA procfs).

## Benchmarks — canonical 2026-11-26
Host: `Intel Core i7-6700` (4C/8T AVX2/FMA, 32 GiB) + `RTX 2060` (6 GB, driver `620.73.04`); `ffmpeg n9.1`; `rustc 1.100.0`; all runs serial. Image/Audio/NVENC/Remux: 2026-11-26, `f523cef` (PR #22). Animation: 2026-11-26, `6f7f14a` (PR #21, not re-run in #22).

### Image — 20× 4K, Q=80 / same compression class (20/20 validated)
| Workload | Tool | Time (ms) | Throughput | Peak RAM | Input | Output | Quality | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `png_to_webp` | `fastconv (cpu q=80)` | 4138.0 | 73.41 MB/s | 609.5 MiB | 303.75 MB | 5.69 MB | 33.95 dB | **3.09x** (+0.35 dB) |
| `png_to_webp` | `ffmpeg (libwebp q=80)` | 12775.7 | 23.78 MB/s | 666.1 MiB | 303.75 MB | 5.34 MB | 33.61 dB | Baseline |
| `png_to_jpeg` | `fastconv (cpu q=80)` | 662.7 | 458.35 MB/s | 414.7 MiB | 303.75 MB | 13.33 MB | 34.29 dB | **1.30x** (+0.32 dB) |
| `png_to_jpeg` | `ffmpeg (mjpeg qv=3)` | 862.2 | 352.29 MB/s | 877.8 MiB | 303.75 MB | 8.58 MB | 33.98 dB | Baseline |
| `jpeg_to_webp` | `fastconv (cpu q=80)` | 3877.6 | 6.93 MB/s | 430.2 MiB | 26.86 MB | 5.90 MB | 38.12 dB | **3.05x** (−0.28 dB) |
| `jpeg_to_webp` | `ffmpeg (libwebp q=80)` | 11844.8 | 2.27 MB/s | 165.7 MiB | 26.86 MB | 5.44 MB | 38.40 dB | Baseline |
| `webp_to_png` | `fastconv (cpu png)` | 515.9 | 11.02 MB/s | 476.3 MiB | 5.69 MB | 153.64 MB | exact | **4.04x** |
| `webp_to_png` | `ffmpeg (png)` | 2084.2 | 2.73 MB/s | 816.6 MiB | 5.69 MB | 89.97 MB | exact | Baseline |
| `png_to_qoi` | `fastconv (cpu qoi)` | 672.6 | 451.58 MB/s | 563.7 MiB | 303.75 MB | 250.91 MB | exact | **1.42x** |
| `png_to_qoi` | `ffmpeg (qoi)` | 958.1 | 317.05 MB/s | 970.1 MiB | 303.75 MB | 250.91 MB | exact | Baseline |
| `webp_to_bmp` | `fastconv (cpu bmp)` | 417.1 | 13.63 MB/s | 716.2 MiB | 5.69 MB | 497.67 MB | exact | **1.19x** |
| `webp_to_bmp` | `ffmpeg (bmp)` | 497.3 | 11.43 MB/s | 449.4 MiB | 5.69 MB | 497.67 MB | exact | Baseline |
| `png_to_tiff` | `fastconv (cpu tiff)` | 885.3 | 343.11 MB/s | 398.9 MiB | 303.75 MB | 276.93 MB | exact (both Deflate) | **3.45x** |
| `png_to_tiff` | `ffmpeg (tiff deflate)` | 3049.9 | 99.59 MB/s | 1429.5 MiB | 303.75 MB | 373.65 MB | exact | Baseline |
| `png_to_jxl` | `fastconv (cpu q=80)` | 3869.0 | 78.51 MB/s | 1367.0 MiB | 303.75 MB | 1.25 MB | 34.80 dB | **4.58x** (same distance 1.9) |
| `png_to_jxl` | `ffmpeg (libjxl d=1.9)` | 17701.9 | 17.16 MB/s | 907.2 MiB | 303.75 MB | 6.49 MB | 34.95 dB | Baseline |
| `png_to_avif` | `fastconv (cpu q=80)` | 2356.4 | 128.91 MB/s | 1307.1 MiB | 303.75 MB | 0.63 MB | 35.63 dB | **1.55x** (−0.12 dB, 4.5x smaller) |
| `png_to_avif` | `ffmpeg (libsvtav1 preset 8)` | 3660.1 | 82.99 MB/s | 2452.6 MiB | 303.75 MB | 2.86 MB | 35.75 dB | Baseline |
| `png_to_heic` | `fastconv (gpu:nvenc q=80)` | 931.1 | 326.22 MB/s | 734.6 MiB | 303.75 MB | 27.47 MB | 39.21 dB | **13.72x** vs ff(webp) / **4.44x** vs cpu(webp) |
| `jpeg_to_heic` | `fastconv (gpu:nvenc q=80)` | 877.4 | 30.61 MB/s | 525.4 MiB | 26.86 MB | 19.24 MB | 38.16 dB | **13.50x** vs ff(webp) / **4.42x** vs cpu(webp) |
| `webp_to_heic` | `fastconv (gpu:nvenc q=80)` | 919.7 | 6.18 MB/s | 783.8 MiB | 5.69 MB | 13.21 MB | 38.10 dB | NVENC 20× 4K (no ffmpeg HEIC opponent) |

### Audio — 60 s stereo 48 kHz/16-bit, target-matched bitrate/quality (achieved size/PSNR per row; 34/34 outputs validated: 16 single-file + 18 batch)
| Workload | Tool | Time (ms) | Throughput | Peak RAM | Input | Output | Quality | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wav_to_flac` | `fastconv (flacenc pure rust)` | 50.9 | 226.24 MB/s | 71.4 MiB | 11.52 MB | 4.12 MB | bit-exact | **3.66x** |
| `wav_to_flac` | `ffmpeg (flac -5)` | 186.1 | 61.89 MB/s | 57.6 MiB | 11.52 MB | 4.11 MB | bit-exact | Baseline |
| `wav_to_aiff` | `fastconv (native pcm_s16be)` | 30.1 | 382.38 MB/s | 20.0 MiB | 11.52 MB | 11.52 MB | bit-exact | **3.18x** |
| `wav_to_aiff` | `ffmpeg (pcm_s16be)` | 95.9 | 120.07 MB/s | 50.9 MiB | 11.52 MB | 11.52 MB | bit-exact | Baseline |
| `wav_to_alac` | `fastconv (alac-encoder m4a)` | 116.5 | 98.85 MB/s | 30.0 MiB | 11.52 MB | 4.48 MB | bit-exact | **3.05x** |
| `wav_to_alac` | `ffmpeg (alac)` | 355.6 | 32.40 MB/s | 52.4 MiB | 11.52 MB | 4.69 MB | bit-exact | Baseline |
| `wav_to_opus` | `fastconv (opus-pure 0.2.5)` | 152.2 | 75.68 MB/s | 73.2 MiB | 11.52 MB | 0.98 MB | 37.1 dB | **3.77x** |
| `wav_to_opus` | `ffmpeg (libopus 128k)` | 574.3 | 20.06 MB/s | 56.4 MiB | 11.52 MB | 1.35 MB | — | Baseline |
| `wav_to_ogg` | `fastconv (vorbis-enc q4)` | 864.7 | 13.32 MB/s | 35.8 MiB | 11.52 MB | 0.74 MB | 35.8 dB | **1.03x** |
| `wav_to_ogg` | `ffmpeg (libvorbis q4)` | 887.3 | 12.98 MB/s | 57.5 MiB | 11.52 MB | 0.70 MB | — | Baseline |
| `wav_to_mp3` | `fastconv (rusty_mp3 192k)` | 659.9 | 17.46 MB/s | 37.9 MiB | 11.52 MB | 1.44 MB | 64.3 dB | **1.53x** |
| `wav_to_mp3` | `ffmpeg (libmp3lame 192k)` | 1010.2 | 11.40 MB/s | 54.5 MiB | 11.52 MB | 1.44 MB | — | Baseline |
| `wav_to_aac` | `fastconv (rusty_aac 2.2.0, 128k ADTS)` | 153.9 | 74.85 MB/s | 61.7 MiB | 11.52 MB | 0.98 MB | 26.4 dB | **10.32x** |
| `wav_to_aac` | `ffmpeg (aac 128k)` | 1587.9 | 7.25 MB/s | 54.3 MiB | 11.52 MB | 0.98 MB | — | Baseline |
| `wav_to_m4a` | `fastconv (rusty_aac 2.2.0, 128k muxed)` | 154.2 | 74.69 MB/s | 63.9 MiB | 11.52 MB | 0.97 MB | 26.4 dB | **10.36x** |
| `wav_to_m4a` | `ffmpeg (aac 128k)` | 1597.7 | 7.21 MB/s | 54.5 MiB | 11.52 MB | 0.97 MB | — | Baseline |
| `audio_batch_mixed_to_flac` | `fastconv (parallel, 9 inputs)` | 161.0 | 37.55 MB/s | 83.1 MiB | 6.04 MB | 5.65 MB | 9/9 validated | **5.10x** |
| `audio_batch_mixed_to_flac` | `ffmpeg (serial, 9 inputs)` | 820.2 | 7.37 MB/s | 60.5 MiB | 6.04 MB | 6.83 MB | 9/9 validated | Baseline |

### Video — NVENC & Remux, 1080p60, 600 frames, 3 serial runs/side
| Workload | Tool | Time (ms) | FPS | Peak RAM | Output | PSNR | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nvenc_h264_mp4` | `fastconv (nvdec+nvenc)` | 1894.8 | 316.7 | 480.7 MiB | 25.86 MB | 50.72 dB | **1.04x** |
| `nvenc_h264_mp4` | `ffmpeg (h264_nvenc, hwaccel cuda)` | 1973.0 | 304.1 | 631.5 MiB | 25.90 MB | 50.73 dB | Baseline |
| `nvenc_hevc_mp4` | `fastconv (nvdec+nvenc)` | 2003.9 | 299.4 | 592.0 MiB | 20.52 MB | 47.62 dB | **1.03x** |
| `nvenc_hevc_mp4` | `ffmpeg (hevc_nvenc, hwaccel cuda)` | 2055.9 | 291.8 | 580.9 MiB | 20.54 MB | 47.62 dB | Baseline |
| `remux_mp4_to_mkv` | `fastconv (pure rust)` | 15.5 | 38827.2 | 16.7 MiB | 19.16 MB | codec-copy | **5.53x** |
| `remux_mp4_to_mkv` | `ffmpeg (-c copy)` | 85.7 | 7004.5 | 60.4 MiB | 19.14 MB | codec-copy | Baseline |
| `remux_mkv_to_mp4` | `fastconv (pure rust)` | 24.1 | 24848.5 | 37.2 MiB | 19.15 MB | codec-copy | **3.46x** |
| `remux_mkv_to_mp4` | `ffmpeg (-c copy)` | 83.4 | 7190.8 | 57.5 MiB | 19.15 MB | codec-copy | Baseline |

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
- Turing AV1 is CPU-decoded; `--hwdecode nvdec` is rejected with no fallback. CPU HEVC encode rejected; NVENC image output is HEIC-only.
- GPU probe is Linux-only (DRM sysfs / NVIDIA procfs); other OSes report no adapters/backends. Animation bench not yet in `xtask` (external 3-run scripts); not re-run in PR #22.
- Smallest margin: `wav_to_ogg` 864.7 vs 887.3 ms (1.03x) — within run-to-run variance. AVIF/JXL matched PSNR and smaller outputs; TIFF both Deflate, exact; AVIF peak RAM 1307.1 vs 2452.6 MiB.
