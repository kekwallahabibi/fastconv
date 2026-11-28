# fastconv — state at a glance
> Update in-place on changes; git history preserves past snapshots.

## Snapshot
| Item | Value |
| :--- | :--- |
| Product | `fastconv` 0.1.0 · edition 2024 · AGPL-3.0-only · pre-merge / pre-review branch `feat/extract-frames` (Feature 4/6: `fastconv frames` still-frame extraction from video and animated GIF/WebP) · 5 crates |
| Verify | `cargo xtask verify` green — **816** default tests (40 cli + 3 audio_codecs + 12 cli_help + 4 frames + 6 validation_errors + 7 video_add_audio + 5 video_extract + 120 domain + 451 pipeline + 146 probe + 22 xtask), **548** `--features nvenc` tests (40 cli + 3 audio_codecs + 12 cli_help + 4 frames + 6 validation_errors + 7 video_add_audio + 5 video_extract + 471 pipeline); 273 deps license-checked; fmt + clippy `-D warnings` clean (default and nvenc); no-FFmpeg/ffprobe scanner clean in `crates/` |
| Release binary | default **39,724,568 B (37.88 MiB) → 28,503,464 B (27.18 MiB), −28.25%**; `--features nvenc` **40,410,144 B (38.54 MiB) → 29,155,880 B (27.81 MiB), −27.85%** — `strip = true`, `lto = "fat"`, `codegen-units = 1`, `opt-level = 3` (retained from prior snapshot; not re-measured for this feature) |

## Build size
| Profile | Stripped | Unstripped | Strip reduction |
| :--- | ---: | ---: | ---: |
| `default` | 28,503,464 B (27.18 MiB) | 33,745,336 B (32.18 MiB) | −15.5% |
| `--features nvenc` | 29,155,880 B (27.81 MiB) | 34,472,792 B (32.88 MiB) | −15.4% |

Profile: `lto = "fat"`, `codegen-units = 1`, `opt-level = 3`, `strip = true` (stripped) / `strip = none` (unstripped). Retained from the prior binary-shrink snapshot; not re-measured for Feature 4.

## CLI
| Command | Form |
| :--- | :--- |
| `probe` | `fastconv probe` — CPU model/topology/SIMD/RAM + GPU adapters & media backends |
| `convert` | `fastconv convert <src> <dst> [--to <fmt>] [--quality 0..=100 or --lossless] [--threads N] [--backend <name>]` |
| `audio` | `fastconv audio <src> <dst> [--to <fmt>] [--codec <vorbis\|opus\|flac\|aac\|alac\|pcm\|mp3>] [--bitrate <r> \| --quality 0..=100] [--sample-rate <hz>] [--channels 1..=8] [--bit-depth <8\|16\|24\|32\|f32>] [--track N] [--threads N] [--backend cpu]` — CPU-only; video sources extract an audio track directly |
| `remux` | `fastconv remux <src> <dst> [--to mp4, mkv, webm, mov, avi, ts] [--threads N]` — no re-encode |
| `video` | `fastconv video <src> <dst> [--to <fmt>] [--vcodec <codec>] [--acodec <codec>] [--quality 0..=100 \| --vbitrate <rate>] [--abitrate <rate>] [--asample-rate <hz>] [--achannels N] [--add-audio <path>] [--replace-audio] [--threads N] [--backend <name>] [--hwdecode auto, cpu, nvdec]` — `--add-audio` appends an external audio track (or video-container audio track) to a single video file; `--replace-audio` drops the source tracks first |
| `frames` | `fastconv frames <source> <output-dir> [--to <fmt>] [--quality 0..=100 \| --lossless] [--fps <N> \| --every <N>] [--start <N>] [--end <N>] [--threads <N>] [--backend cpu\|auto]` — CPU-only still-frame extraction; writes `frame_000001.<ext>`, `frame_000002.<ext>`, …; video and animated GIF/WebP inputs; GPU backends rejected explicitly |
| `help` | `fastconv help <probe, convert, audio, remux, video, frames>`; `-h`/`--help`, `-V`/`--version` |

## Formats
- **Image — 11 in × 12 out.** In: `png jpeg webp bmp qoi tiff jxl avif gif ico pnm` (P1–P7); out: same set + `heic`. Still GIF uses a fixed 256-color palette; `--quality`: webp/jpeg/jxl/avif/heic, `--lossless`: webp/jxl + png/bmp/qoi/tiff/ico/pnm; HEIC works on CPU (pure-Rust `heifer` HEVC-Intra, default backend) and on NVENC GPU (`--backend nvenc`).
- **Audio — 9 in × 9 out + video→audio.** `wav flac opus ogg mp3 aac m4a alac aiff` (Ogg = Vorbis, AAC = ADTS, ALAC in M4A); CPU-only, parallel batches. `fastconv audio` also reads the audio track of `mp4 mov mkv webm avi ts` directly (no intermediate file) into any of the 9 outputs; `--track N` selects the track, default 0.
- **Video — 6 containers** `mp4 mov mkv webm avi ts`; codecs H.264/HEVC (H.265)/AV1/VP9 + AAC/Opus/MP3/FLAC/PCM + `copy`. CPU: H.264/AV1/VP9/GIF/WebP; NVENC: H.264/HEVC; `--hwdecode` NVDEC zero-copy. `--add-audio <path>` accepts WAV/FLAC/MP3/OGG/Opus/AAC/M4A/ALAC/AIFF or audio tracks inside video containers, defaults to AAC in MP4/MOV/AVI/TS and Opus in MKV/WebM, and is rejected for directory batches, `--acodec copy/none` and animation targets; `--replace-audio` requires `--add-audio`.
- **Animation — bidirectional** video ↔ animated GIF/WebP via `fastconv video --to gif,webp --vcodec gif,webp` (CPU + NVENC); GIF/WebP also valid video inputs; `remux` rejects them.
- **Frame extraction — CPU-only** from video (`mp4 mov mkv webm avi ts`, H.264/HEVC/AV1/VP9/MJPEG) and animated GIF/WebP into any of the 12 still formats via `fastconv frames`; inclusive `--start`/`--end`, `--every N` or `--fps N`, bounded recycled pixel buffers (≤256 MiB in flight), streaming encode workers; GPU backends are rejected explicitly (no silent CPU fallback).

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
| `domain` | `src/{lib,audio,frames,hardware,image,metrics,video}.rs` — pure types/invariants |
| `probe` | `src/{lib,audio,frames,hardware,image,video}.rs` — boundary parsing/validation |
| `pipeline` | `src/{lib,audio,aiff,avif,avi,bmp,cpu,decode,frames,gif,heic,ico,m4a,mpegts,pnm,qoi,remux,staging,video_anim,video_bitstream,video_codecs,video_container,video_transcode}.rs` + `src/gpu/{mod,nvdec,nvenc,heic}.rs` |
| `cli` (`fastconv`) | `src/{main,audio,frames,video}.rs`, `tests/{audio_codecs,cli_help,frames,validation_errors,video_extract,video_add_audio}.rs` — thin orchestration |
| `xtask` | `src/{main,bench,audio_bench,nvenc_bench,remux_bench,fixtures}.rs` — verify / fixtures / bench |

## Invariants
- No FFmpeg in `crates/` (scanner in `cargo xtask verify`); ffmpeg only in `xtask` as bench opponent / fixture generator.
- Boundary validation in `probe`; `domain` holds valid types only; illegal states (audio + GPU, quality on lossless formats, NVENC non-HEIC) are enum-level (`ExecutionPlan`, `ImageExecutionMode`, `ImageCompression`, `GpuMediaKind`); CPU HEIC is a valid CPU image mode.
- Zero silent fallback: failed GPU setup, `--backend vaapi/vulkan`, CPU HEVC video encode, failed NVDEC → hard error; requesting `--backend cpu` for HEIC uses `heifer` directly, not a failed-GPU fallback.
- Lints: `unsafe_code = "forbid"`, `unwrap_used`/`expect_used`/`panic`/`todo`/`unimplemented`/`dbg_macro` = `deny`, `clippy::pedantic = "warn"`; pipeline relaxes unsafe only under `--features nvenc`, confined to narrowly scoped `#[allow(unsafe_code)]` FFI blocks in `gpu/nvenc.rs` and `gpu/nvdec.rs`.
- AGPL-3.0-only; 273 deps permissive/AGPL-compatible; `GPL-2.0-only` rejected (Classpath exception allowed).
- RAM caps (actual constants): JXL/AVIF ≤8 image workers, HEIC ≤8 image workers (RGB/RGBA scratch + HEVC tile state), AVIF inner threads ≤8, MP4 faststart ≤64 MiB in-memory (stream above), ≤8 animation workers, ≤2 CPU video file workers, 1 GiB decoded/track, 512 MiB PCM/file, 1 GiB PCM/batch.
- Turing reality: NVENC = HEVC-intra (HEIC) only on the GPU path; CPU HEIC (`heifer`) works on any host; no HW encoders for webp/jpeg/png/bmp/qoi/tiff/jxl/avif/gif/ico/pnm; AV1 CPU-decode; image Vulkan rejected. GPU probe Linux-only (DRM sysfs + NVIDIA procfs).
- Video→audio extraction is CPU-only: `fastconv audio <video.mp4> <out> --codec …` demuxes + decodes the selected audio track (default 0) in-process; out-of-range `--track` and GPU backends are rejected explicitly, no silent CPU fallback.
- External audio is a validated boundary input (`probe/video.rs`): decoded once into the target codec's native format, resampled/remixed as needed, and muxed alongside packet-copied video; `--acodec copy/none`, directory batches, animation containers, duplicate/missing paths and target aliases are rejected before any output is created. `--replace-audio` is opt-in and drops all source audio tracks. Matroska targets reject MP4 presentation edits Matroska cannot represent (no silent timestamp corruption).

## Benchmarks — canonical 2026-11-28 (pre-merge / pre-review branch run, Feature 4 `fastconv frames`)
Host: `Intel Core i7-6700` (4C/8T AVX2/FMA, 32023 MiB) + `RTX 2060` (6 GB, driver `620.73.04`); `ffmpeg n9.1`; all runs serial.

- Provenance (this snapshot): branch `feat/extract-frames`, HEAD `ebb7c853c4c73abe59e2929124db00ad74acd73c`. At run time the feature was an uncommitted pre-review working tree (11 modified files + 5 new untracked `frames.rs` files across `domain`/`probe`/`pipeline`/`cli`); `git diff | sha256sum` was `489d55de8e3a4943bcc1d725806adb47643eafbbafc4f7b4e44fc11c986b9926` — code diff only, captured before this `STATE.md` edit (`git diff -- . ':!STATE.md' | sha256sum` still matches), not reproducible from a merged HEAD. `target/release/fastconv probe` reported `Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz` (4C/8T, x86:avx2+fma, 32023 MiB) and `NVIDIA GeForce RTX 2060` (driver `620.73.04`), media backends `nvenc [encode+decode]` and `vaapi [decode-only]`; `ffmpeg -version` = `ffmpeg version n9.1`. Toolchain: rustc/cargo `1.100.0`, gcc `16`. Runs 09:35–10:02 CET. `STATE.md` was updated after the runs; no code file was modified by the benchmark.
- Commands: `cargo xtask gen-fixtures`; `cargo build --release -p fastconv-cli` and `cargo build --release -p fastconv-cli --features nvenc` (pre-built because a cold release build exceeds the 180 s bench guard); `timeout -k 10 180s cargo xtask bench`; `timeout -k 10 180s cargo xtask bench --nvenc`; `timeout -k 10 180s cargo xtask bench --remux`; `timeout -k 10 180s cargo xtask verify`. Audio and add-audio rows are retained verbatim from the prior snapshot (not re-run on this branch). `fastconv frames` workloads (new, no `xtask bench` target), 3 serial runs/side on identical fixture-derived inputs: A `target/release/fastconv frames fixtures/video_1080p60.mp4 <out> --every 10` vs `ffmpeg -nostdin -y -hide_banner -loglevel error -i fixtures/video_1080p60.mp4 -vf "select='not(mod(n,10))'" -fps_mode passthrough -compression_level 1 <out>/frame_%06d.png`; A2 `... --start 0 --end 59` vs `ffmpeg ... -frames:v 60 -compression_level 1 ...`; B `... --every 10 --to jpeg --quality 80` vs `ffmpeg ... -vf "select='not(mod(n,10))'" -fps_mode passthrough -q:v 4 ...`; animations generated once with `ffmpeg -nostdin -y -hide_banner -loglevel error -i fixtures/video_1080p60.mp4 -vf "fps=10,scale=480:270:flags=lanczos" -frames:v 60 -loop 0 <tmp>/anim.gif` and the same with `-c:v libwebp_anim -lossless 1 <tmp>/anim_lossless.webp`; C `fastconv frames <anim> <out> --to png` vs `ffmpeg ... -i <anim> -compression_level 1 <out>/frame_%06d.png`. Temporary outputs lived in `/tmp/opencode/` and are not cited as permanent paths.
- Peak RAM: `/usr/bin/time` is not installed on this host, so peak RSS was sampled from `VmHWM` in `/proc/<pid>/status` every 1 ms during every run (identical source to the xtask harness's `sample_proc_peak_rss_bytes`); the maximum across the 3 serial runs is reported. Median wall time includes process launch, matching the harness methodology. `frames` throughput uses the 19.15 MB MP4 (A/A2/B) or the generated animation (0.47 MB GIF / 1.98 MB WebP) as input and counts all 60 emitted frames for FPS.
- Validation: image 20/20 `ffprobe` + decode per row (all 12 harness rows, exact 3840x2160; lossless rows exact RGB, lossy PSNR ≥ 33.6 dB); remux 4/4 codec-copy (`aac`+`h264`, 1920x1080, 600 frames, no decoder errors); NVENC 4/4 (600 frames / 10 s, 1920x1080, B-frames 295, `dts_ok=true`, clean decode, H.264 50.72 dB / HEVC 47.62 dB, explicit `gpu:nvdec+nvenc on NVIDIA GeForce RTX 2060`, no CPU fallback). Audio/add-audio/animation validation is retained verbatim from the prior snapshot (not re-run here). `frames`: every workload emits 60/60 frames, A/A2/B at 1920x1080 and C at 480x270; A/A2 PNG cross-PSNR 48.20 / 48.16 dB (lossless Deflate on both sides, zlib level 1); B JPEG PSNR 41.99 dB (fastconv) vs 45.00 dB (ffmpeg) against the same lossless PNG reference with sizes 5.64 / 5.29 MB (within 7%); C GIF and lossless-WebP extraction are **bit-exact** (`ffmpeg psnr average:inf` on all four channels).
- Regression verdict: **no throughput or quality regression on any existing image, remux or NVENC row.** Image rows are within run-to-run noise of the prior snapshot (e.g. `png_to_webp` 4012.4 vs 4048.7 ms, `jpeg_to_webp` 3823.4 vs 3842.8 ms, `png_to_avif` 2284.0 vs 2270.9 ms) with identical PSNR/size, confirming the `cpu.rs::encode_frame_pixels` and `StagingTarget::create_for_image` refactors are behaviour-preserving; remux 13.3/21.8 ms vs 13.4/22.1 ms; NVENC H.264 1877.5 vs 1884.1 ms and HEVC 1993.1 vs 1983.1 ms with identical 600-frame/PSNR/size validation. New `frames` result: **CPU video extraction is slower than ffmpeg** (0.21x on `--every 10` PNG/JPEG, 0.54x on a 60-frame segment) because the pure-Rust decoder walks all 600 frames to select every 10th, while **animation extraction is 1.91x (GIF) / 2.30x (WebP) faster with bit-exact output and 3–4x lower peak RAM**. No runs were discarded or cherry-picked.

### Add external audio to video — 1080p60/10 s H.264 video + 60 s stereo WAV, 3 serial runs/side, median time / max VmHWM
| Workload | Tool | Time (ms) | Throughput (total input) | Peak RAM | Input | Output | Quality | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: silent MP4 + WAV → MP4 | `fastconv (video --add-audio --vcodec copy --acodec aac)` | 524.53 | 55.46 MiB/s | 65.50 MiB | 29.09 MiB (18.10 + 10.99) | 19.03 MiB | 1 h264 + 1 AAC, clean decode | **3.04x** |
| A: silent MP4 + WAV → MP4 | `ffmpeg (-map 0:v -map 1:a -c:v copy -c:a aac -b:a 128k)` | 1595.75 | 18.23 MiB/s | 59.45 MiB | 29.09 MiB (18.10 + 10.99) | 19.04 MiB | 1 h264 + 1 AAC, clean decode | Baseline |
| B: silent MP4 + WAV → MKV | `fastconv (video --add-audio --vcodec copy --acodec opus)` | 506.68 | 57.41 MiB/s | 63.18 MiB | 29.09 MiB (18.10 + 10.99) | 19.08 MiB | 1 h264 + 1 Opus, clean decode | **1.18x** |
| B: silent MP4 + WAV → MKV | `ffmpeg (-map 0:v -map 1:a -c:v copy -c:a libopus -b:a 128k)` | 598.46 | 48.61 MiB/s | 63.79 MiB | 29.09 MiB (18.10 + 10.99) | 19.40 MiB | 1 h264 + 1 Opus, clean decode | Baseline |
| C: MP4 with audio + WAV → MP4 (append) | `fastconv (video --add-audio --vcodec copy --acodec aac)` | 522.73 | 55.96 MiB/s | 67.57 MiB | 29.25 MiB (18.26 + 10.99) | 19.19 MiB | 1 h264 + 2 AAC, clean decode | **3.04x** |
| C: MP4 with audio + WAV → MP4 (append) | `ffmpeg (-map 0:v -map 0:a -map 1:a -c:v copy -c:a:0 copy -c:a:1 aac -b:a:1 128k)` | 1586.83 | 18.43 MiB/s | 59.94 MiB | 29.25 MiB (18.26 + 10.99) | 19.19 MiB | 1 h264 + 2 AAC, clean decode | Baseline |

All 3 run times (ms): A fastconv 525.59 / 524.53 / 520.47, ffmpeg 1595.75 / 1607.02 / 1580.83; B fastconv 506.25 / 506.68 / 509.77, ffmpeg 598.46 / 590.80 / 601.64; C fastconv 521.34 / 522.73 / 524.89, ffmpeg 1607.64 / 1576.40 / 1586.83. The command-line bitrates match (fastconv AAC/Opus defaults are both 128000 bps). Output sizes (bytes): A 19,958,320 vs 19,960,863; B 20,011,114 vs 20,343,039; C 20,118,660 vs 20,123,033. `ffprobe` durations: A/C fastconv AAC 60.032 s vs ffmpeg 60.000 s (encoder priming); B fastconv 60.0065 s vs ffmpeg 60.008 s. The source fixture's B-frame presentation edit was removed from the shared silent input with `-movflags +negative_cts_offsets` so both tools receive the same representable stream; the original unmodified fixture was used unchanged for C.

### `fastconv frames` — 60 frames, 3 serial runs/side (median time, max VmHWM)
Input for A/A2/B: `fixtures/video_1080p60.mp4` (1920×1080, 60 fps, 600 frames, 19.15 MB). Input for C: generated 480×270 / 10 fps / 60-frame animations (GIF 0.47 MB, lossless WebP 1.98 MB). Both opponents receive the identical fixture-derived file; exact flags are in the Commands line above.

| Workload | Tool | Time (ms) | FPS | MB/s (input) | Peak RAM | Input | Output | Quality | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: MP4→PNG `--every 10` | `fastconv frames --every 10` | 4310.13 | 13.92 | 4.44 | 107.75 MiB | 19.15 MB | 22.35 MB | 60/60, 1920×1080, cross-PSNR 48.20 dB | 0.21x (ffmpeg faster) |
| A: MP4→PNG `--every 10` | `ffmpeg -vf "select='not(mod(n,10))'" -fps_mode passthrough -compression_level 1` | 896.83 | 66.90 | 21.35 | 209.30 MiB | 19.15 MB | 25.52 MB | 60/60, 1920×1080, cross-PSNR 48.20 dB | Baseline |
| A2: MP4→PNG `--start 0 --end 59` | `fastconv frames --start 0 --end 59` | 578.49 | 103.72 | 33.10 | 123.86 MiB | 19.15 MB | 22.25 MB | 60/60, 1920×1080, cross-PSNR 48.16 dB | 0.54x (ffmpeg faster) |
| A2: MP4→PNG first 60 frames | `ffmpeg -frames:v 60 -compression_level 1` | 313.29 | 191.52 | 61.13 | 263.29 MiB | 19.15 MB | 25.20 MB | 60/60, 1920×1080, cross-PSNR 48.16 dB | Baseline |
| B: MP4→JPEG q80 `--every 10` | `fastconv frames --every 10 --to jpeg --quality 80` | 4373.87 | 13.72 | 4.38 | 121.63 MiB | 19.15 MB | 5.64 MB | 60/60, 1920×1080 yuvj420p, PSNR 41.99 dB | 0.21x (ffmpeg faster) |
| B: MP4→JPEG `--every 10` | `ffmpeg -vf "select='not(mod(n,10))'" -fps_mode passthrough -q:v 4` | 914.32 | 65.62 | 20.95 | 251.96 MiB | 19.15 MB | 5.29 MB | 60/60, 1920×1080 yuvj420p, PSNR 45.00 dB | Baseline |
| C: animated GIF→PNG | `fastconv frames --to png` | 53.69 | 1117.5 | 8.72 | 16.41 MiB | 0.47 MB | 3.98 MB | 60/60, 480×270, bit-exact | **1.91x** |
| C: animated GIF→PNG | `ffmpeg -compression_level 1` | 102.58 | 584.9 | 4.56 | 61.88 MiB | 0.47 MB | 2.34 MB | 60/60, 480×270, bit-exact | Baseline |
| C: lossless animated WebP→PNG | `fastconv frames --to png` | 94.78 | 633.0 | 20.86 | 19.93 MiB | 1.98 MB | 3.22 MB | 60/60, 480×270, bit-exact | **2.30x** |
| C: lossless animated WebP→PNG | `ffmpeg -compression_level 1` | 218.42 | 274.7 | 9.05 | 63.38 MiB | 1.98 MB | 3.89 MB | 60/60, 480×270, bit-exact | Baseline |

All 3 run times (ms): A fastconv 4306.99 / 4327.21 / 4310.13, ffmpeg 895.58 / 902.92 / 896.83; A2 fastconv 578.69 / 575.17 / 578.49, ffmpeg 313.29 / 311.10 / 317.05; B fastconv 4425.64 / 4357.48 / 4373.87, ffmpeg 913.76 / 916.20 / 914.32; C GIF fastconv 53.69 / 53.97 / 52.62, ffmpeg 100.60 / 102.81 / 102.58; C WebP fastconv 95.35 / 94.78 / 94.69, ffmpeg 216.59 / 218.42 / 219.19.

### Image — 20× 4K, Q=80 / same compression class (20/20 per harness row; CPU HEIC single 1/1 retained from prior snapshot)
| Workload | Tool | Time (ms) | Throughput | Peak RAM | Input | Output | Quality | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `png_to_webp` | `fastconv (cpu q=80)` | 4012.4 | 75.70 MB/s | 635.2 MiB | 303.75 MB | 5.69 MB | 33.95 dB | **3.10x** |
| `png_to_webp` | `ffmpeg (libwebp q=80)` | 12457.2 | 24.38 MB/s | 666.0 MiB | 303.75 MB | 5.34 MB | 33.61 dB | Baseline |
| `png_to_jpeg` | `fastconv (cpu q=80)` | 554.0 | 548.29 MB/s | 412.3 MiB | 303.75 MB | 13.39 MB | 34.30 dB | **1.51x** |
| `png_to_jpeg` | `ffmpeg (mjpeg qv=3)` | 836.4 | 363.17 MB/s | 906.5 MiB | 303.75 MB | 8.58 MB | 33.98 dB | Baseline |
| `jpeg_to_webp` | `fastconv (cpu q=80)` | 3823.4 | 7.02 MB/s | 429.1 MiB | 26.86 MB | 5.90 MB | 38.12 dB | **3.02x** |
| `jpeg_to_webp` | `ffmpeg (libwebp q=80)` | 11557.4 | 2.32 MB/s | 165.7 MiB | 26.86 MB | 5.44 MB | 38.40 dB | Baseline |
| `webp_to_png` | `fastconv (cpu png)` | 548.1 | 10.38 MB/s | 514.4 MiB | 5.69 MB | 153.64 MB | exact | **3.78x** |
| `webp_to_png` | `ffmpeg (png)` | 2070.0 | 2.75 MB/s | 816.1 MiB | 5.69 MB | 89.97 MB | exact | Baseline |
| `png_to_qoi` | `fastconv (cpu qoi)` | 646.5 | 469.81 MB/s | 544.1 MiB | 303.75 MB | 250.91 MB | exact | **1.55x** |
| `png_to_qoi` | `ffmpeg (qoi)` | 999.3 | 303.97 MB/s | 983.0 MiB | 303.75 MB | 250.91 MB | exact | Baseline |
| `webp_to_bmp` | `fastconv (cpu bmp)` | 355.8 | 15.98 MB/s | 605.1 MiB | 5.69 MB | 497.67 MB | exact | **1.20x** |
| `webp_to_bmp` | `ffmpeg (bmp)` | 425.7 | 13.36 MB/s | 484.6 MiB | 5.69 MB | 497.67 MB | exact | Baseline |
| `png_to_tiff` | `fastconv (cpu tiff)` | 830.3 | 365.84 MB/s | 595.5 MiB | 303.75 MB | 276.93 MB | exact (both Deflate) | **3.11x** |
| `png_to_tiff` | `ffmpeg (tiff)` | 2586.5 | 117.43 MB/s | 1472.6 MiB | 303.75 MB | 373.65 MB | exact | Baseline |
| `png_to_jxl` | `fastconv (cpu q=80)` | 3799.4 | 79.95 MB/s | 1379.9 MiB | 303.75 MB | 1.25 MB | 34.80 dB | **4.62x** (same distance 1.9) |
| `png_to_jxl` | `ffmpeg (libjxl d=1.9)` | 17551.4 | 17.31 MB/s | 901.1 MiB | 303.75 MB | 6.49 MB | 34.95 dB | Baseline |
| `png_to_avif` | `fastconv (cpu q=80)` | 2284.0 | 132.99 MB/s | 1317.7 MiB | 303.75 MB | 0.64 MB | 35.65 dB | **1.53x** (−0.10 dB, 4.5x smaller) |
| `png_to_avif` | `ffmpeg (libsvtav1 preset 8)` | 3497.8 | 86.84 MB/s | 2952.9 MiB | 303.75 MB | 2.86 MB | 35.75 dB | Baseline |
| `png_to_heic` | `fastconv (gpu:nvenc q=80)` | 983.4 | 308.86 MB/s | 712.1 MiB | 303.75 MB | 27.47 MB | 39.21 dB | **12.67x** vs ff(webp) / **4.08x** vs cpu(webp) |
| `jpeg_to_heic` | `fastconv (gpu:nvenc q=80)` | 855.8 | 31.38 MB/s | 524.4 MiB | 26.86 MB | 19.24 MB | 38.16 dB | **13.50x** vs ff(webp) / **4.47x** vs cpu(webp) |
| `webp_to_heic` | `fastconv (gpu:nvenc q=80)` | 893.1 | 6.37 MB/s | 781.2 MiB | 5.69 MB | 13.21 MB | 38.10 dB | NVENC 20× 4K (no ffmpeg HEIC opponent) |
| `png_to_heic` (single 4K) | `fastconv (cpu heifer q=80, 3 runs)` | 6251.9 | 2.43 MB/s | 135.8 MiB | 15.18 MB | 1.09 MB | 35.45 dB | CPU-only (no ffmpeg HEIC opponent) |
| `png_to_heic` (20× 4K batch) | `fastconv (cpu heifer q=80, 8 workers, 3 runs)` | 122310.5 | 2.48 MB/s | 467.6 MiB | 303.75 MB | 21.92 MB | 35.39 dB mean | CPU-only (no ffmpeg HEIC opponent) |

All harness rows re-run on this branch; the two CPU HEIC rows have no `xtask bench` target and are retained verbatim from the prior snapshot.

### Audio — 60 s stereo 48 kHz/16-bit, target-matched bitrate/quality (achieved size/lag-aligned SNR per row; 34/34 outputs validated: 16 single-file + 18 batch, 18/18 content checks)
| Workload | Tool | Time (ms) | Throughput | Peak RAM | Input | Output | Quality | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wav_to_flac` | `fastconv (flacenc pure rust)` | 49.6 | 232.06 MB/s | 71.1 MiB | 11.52 MB | 4.12 MB | bit-exact | **3.77x** |
| `wav_to_flac` | `ffmpeg (-c:a flac -compression_level 5)` | 187.2 | 61.53 MB/s | 57.3 MiB | 11.52 MB | 4.11 MB | bit-exact | Baseline |
| `wav_to_aiff` | `fastconv (native pcm_s16be)` | 8.9 | 1298.19 MB/s | 7.5 MiB | 11.52 MB | 11.52 MB | bit-exact | **10.55x** |
| `wav_to_aiff` | `ffmpeg (-c:a pcm_s16be)` | 93.9 | 122.64 MB/s | 52.0 MiB | 11.52 MB | 11.52 MB | bit-exact | Baseline |
| `wav_to_alac` | `fastconv (alac-encoder m4a)` | 100.6 | 114.51 MB/s | 32.5 MiB | 11.52 MB | 4.48 MB | bit-exact | **3.50x** |
| `wav_to_alac` | `ffmpeg (-c:a alac)` | 352.2 | 32.71 MB/s | 50.5 MiB | 11.52 MB | 4.69 MB | bit-exact | Baseline |
| `wav_to_opus` | `fastconv (opus-pure)` | 139.5 | 82.61 MB/s | 72.7 MiB | 11.52 MB | 0.98 MB | 37.1 dB | **4.04x** |
| `wav_to_opus` | `ffmpeg (-c:a libopus -b:a 128k)` | 563.1 | 20.46 MB/s | 53.5 MiB | 11.52 MB | 1.35 MB | — | Baseline |
| `wav_to_ogg` | `fastconv (vorbis-enc q4)` | 814.8 | 14.14 MB/s | 14.2 MiB | 11.52 MB | 0.74 MB | 35.8 dB | **1.08x** |
| `wav_to_ogg` | `ffmpeg (-c:a libvorbis -q:a 4)` | 879.8 | 13.09 MB/s | 56.0 MiB | 11.52 MB | 0.70 MB | — | Baseline |
| `wav_to_mp3` | `fastconv (rusty_mp3 192k)` | 592.5 | 19.44 MB/s | 16.3 MiB | 11.52 MB | 1.44 MB | 64.3 dB | **1.65x** |
| `wav_to_mp3` | `ffmpeg (-c:a libmp3lame -b:a 192k)` | 976.0 | 11.80 MB/s | 54.7 MiB | 11.52 MB | 1.44 MB | — | Baseline |
| `wav_to_aac` | `fastconv (rusty_aac 128k)` | 139.3 | 82.72 MB/s | 58.8 MiB | 11.52 MB | 0.98 MB | 26.4 dB | **11.23x** |
| `wav_to_aac` | `ffmpeg (-c:a aac -b:a 128k)` | 1564.7 | 7.36 MB/s | 54.3 MiB | 11.52 MB | 0.98 MB | — | Baseline |
| `wav_to_m4a` | `fastconv (rusty_aac 128k muxed)` | 138.7 | 83.08 MB/s | 64.9 MiB | 11.52 MB | 0.97 MB | 26.4 dB | **11.35x** |
| `wav_to_m4a` | `ffmpeg (-c:a aac -b:a 128k)` | 1574.1 | 7.32 MB/s | 55.6 MiB | 11.52 MB | 0.97 MB | — | Baseline |
| `audio_batch_mixed_to_flac` | `fastconv (parallel, 9 inputs, 1 worker budgeted)` | 224.1 | 26.97 MB/s | 57.7 MiB | 6.04 MB | 5.65 MB | 9/9 validated | **3.65x** |
| `audio_batch_mixed_to_flac` | `ffmpeg (sequential, 9 inputs)` | 817.9 | 7.39 MB/s | 60.5 MiB | 6.04 MB | 6.83 MB | 9/9 validated | Baseline |

### Video — NVENC & Remux, 1080p60, 600 frames, 3 serial runs/side
| Workload | Tool | Time (ms) | FPS | Peak RAM | Output | PSNR | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nvenc_h264_mp4` | `fastconv (nvdec+nvenc)` | 1877.5 | 319.6 | 480.8 MiB | 25.86 MB | 50.72 dB | **1.04x** |
| `nvenc_h264_mp4` | `ffmpeg (h264_nvenc, hwaccel cuda)` | 1957.9 | 306.4 | 631.5 MiB | 25.90 MB | 50.73 dB | Baseline |
| `nvenc_hevc_mp4` | `fastconv (nvdec+nvenc)` | 1993.1 | 301.0 | 591.4 MiB | 20.52 MB | 47.62 dB | **1.03x** |
| `nvenc_hevc_mp4` | `ffmpeg (hevc_nvenc, hwaccel cuda)` | 2057.3 | 291.6 | 580.8 MiB | 20.54 MB | 47.62 dB | Baseline |
| `remux_mp4_to_mkv` | `fastconv (pure rust)` | 13.3 | 45109.9 | 16.2 MiB | 19.16 MB | codec-copy | **6.29x** |
| `remux_mp4_to_mkv` | `ffmpeg (-c copy)` | 83.6 | 7179.4 | 60.4 MiB | 19.14 MB | codec-copy | Baseline |
| `remux_mkv_to_mp4` | `fastconv (pure rust)` | 21.8 | 27491.0 | 55.6 MiB | 19.15 MB | codec-copy | **3.73x** |
| `remux_mkv_to_mp4` | `ffmpeg (-c copy)` | 81.3 | 7377.3 | 57.5 MiB | 19.15 MB | codec-copy | Baseline |

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
- Add-audio duration is encoder-priming-bound: AAC outputs are 60.032 s (fastconv) vs 60.000 s (ffmpeg); Opus-in-MKV 60.0065 vs 60.008 s. Track counts, codecs and full decodes are identical; the copied video stream stays 10 s in every output.
- The shared silent input for A/B was normalised with `-movflags +negative_cts_offsets` because fastconv correctly rejects an MP4 presentation edit that Matroska cannot represent (no silent timestamp corruption); the flag changes container timestamp encoding only, not the H.264 payload, and both opponents receive the identical file. Workload C used the original unmodified fixture (MP4→MP4 edit is representable).
- Add-audio bitrates are matched: fastconv's AAC and Opus defaults are both 128000 bps vs ffmpeg's `-b:a 128k`; output sizes differ by at most 1.7% (B 19.08 vs 19.40 MiB), so the speedup is not bought with inflated output.
- WebP animation-input PSNR lag: 29.79 vs 48.94 dB (CPU), 29.63 vs 36.56 dB (NVENC) — known `zenwebp` vs `libwebp` color-conversion delta (still-image repro 30.1 dB); not bought with lower compression (0.30 vs 0.27 MB).
- Turing AV1 is CPU-decoded; `--hwdecode nvdec` is rejected with no fallback. CPU HEVC *video* encode is rejected; still-image HEIC is available on CPU (`heifer`) and on NVENC GPU (image output remains HEIC-only).
- GPU probe is Linux-only (DRM sysfs / NVIDIA procfs); other OSes report no adapters/backends. Animation bench and CPU HEIC image bench have no `xtask` target (external 3-run scripts); those rows are retained unchanged. All `xtask` image, remux and NVENC benches were re-run on this branch; audio and add-audio rows are retained from the prior snapshot. `fastconv frames` has no `xtask bench` target either and was measured with the serial 3-run helper described above.
- `audio_batch_mixed_to_flac` measures 224.1 ms / 3.65x, not the stale 109.8 ms / 7.53x number. Merged post-#34 worker budgeting sees no `total_samples` for the AAC ADTS input, reserves the full 512 MiB per-file PCM allowance and caps the 9-file mixed batch at 1 worker (57.7 MiB peak); removing only that input restores 8 workers and ~108 ms, and unmodified HEAD reproduces 1 worker / 217–238 ms. Quality is unchanged (9/9 bit-exact or SNR-validated tracks); this is a stale-baseline correction, not a Feature 3 regression.
- Audio zero-quality-loss: FLAC/AIFF/ALAC remain bit-exact (5,760,000 samples identical single-file, 9/9 batch tracks decoded; wav/flac/alac/aiff batch tracks bit-identical), lossy single-file SNR (37.1/35.8/64.3/26.4/26.4 dB) and every output size match the prior snapshot exactly. Single-file timings match prior within noise (49.6/8.9/100.6/139.5/814.8/592.5/139.3/138.7 ms). Smallest margin: `wav_to_ogg` 814.8 vs 879.8 ms (1.08x). No samples were discarded or cherry-picked.
- CPU HEIC throughput is ~2.4 MB/s on this 4C/8T host: `heifer`'s Medium HEVC-Intra preset is intentionally kept (no preset cheating), and the 20-image batch caps at 8 image workers to bound RAM (peak 467.6 MiB, far below 3 GiB). The NVENC path encodes the same 20-image batch in 983.4 ms, but that is a different quality class (27.47 MB / 39.21 dB vs 21.92 MB / 35.39 dB CPU), not a fair speedup comparison.
- `ffprobe` lists CPU-HEIC output as 40 × 512×512 HEVC tile items + a 320×180 thumbnail rather than one 3840×2160 stream; full resolution is confirmed via ffmpeg's `Tile Grid: ... 3840x2160` group and the clean 24,883,200-byte RGB24 decode.
- AVIF/JXL matched PSNR and smaller outputs; TIFF both Deflate, exact; AVIF peak RAM 1317.7 vs 2952.9 MiB (fastconv lower; both under 3 GiB), JXL 1379.9 vs 901.1 MiB.
- Image rows re-run on this branch: 3.10x webp, 1.51x jpeg, 3.02x jpeg→webp, 3.78x webp→png, 1.55x qoi, 1.20x bmp, 3.11x tiff, 4.62x JXL, 1.53x AVIF; NVENC HEIC 12.67x/13.50x vs ffmpeg(webp) with 20/20 decoded 4K outputs. All four lossless image rows remain exact RGB; lossy rows keep PSNR in the same band, but sizes are not uniform: `png_to_jpeg` is 56% larger (13.39 vs 8.58 MB), both WebP rows are within 9%, and JXL/AVIF are smaller than ffmpeg's outputs. Smallest margin: `webp_to_bmp` 355.8 vs 425.7 ms (1.20x).
- Video timings are within noise of the prior snapshot: NVENC H.264 1877.5 vs 1957.9 ms (1.04x vs 1.06x), HEVC 1993.1 vs 2057.3 ms (1.03x vs 1.01x); remux mp4→mkv 13.3 vs 13.4 ms (6.29x vs 6.27x) and mkv→mp4 21.8 vs 22.1 ms (3.73x vs 3.69x, ffmpeg 81.3 vs 81.3 ms). Output sizes, 50.72/47.62 dB PSNR, 600-frame validation, B-frames 295, `dts_ok=true`, clean decode and the explicit `gpu:nvdec+nvenc on NVIDIA GeForce RTX 2060` proof are unchanged in kind; no runs were discarded or cherry-picked.
- `fastconv frames` fairness: A/A2 use lossless Deflate on both sides — `png::Compression::Fastest` (zlib level 1) vs ffmpeg `-compression_level 1`; encoded sizes differ because of filter choices (22.35 vs 25.52 MB), but decoded pixels are compared, not compressed bytes. B size-matches `-q:v 4` (5.29 vs 5.64 MB, within 7%) where ffmpeg is actually 3.0 dB *better* (45.00 vs 41.99 dB); the PSNR-matched `-q:v 7` is 31% smaller (3.88 MB at 42.06 dB), so the slow fastconv result is not bought with inflated output. C uses a lossless VP8L animation so both decoders produce bit-exact RGBA; the lossy animated WebP variant shows the known `zenwebp` vs `libwebp` YUV→RGB delta (29.36 dB) and is not used for the bit-exact claim.
- `fastconv frames` performance reality: CPU video extraction is 4.8x slower than ffmpeg on `--every 10` (4310.13 vs 896.83 ms) because the pure-Rust decoder must decode all 600 frames to select 60, and 1.85x slower even on a 60-frame segment (578.49 vs 313.29 ms). Animation extraction is the strength: GIF 1.91x and lossless WebP 2.30x faster with bit-exact frames and 3–4x lower peak RAM (16–20 MiB vs 62–63 MiB). All 60-frame counts, 1920×1080/480×270 dimensions and PSNR/PSNR-inf checks passed; no runs were discarded.
