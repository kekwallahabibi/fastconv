# fastconv — state at a glance
> Update in-place on changes; git history preserves past snapshots.

## Snapshot
| Item | Value |
| :--- | :--- |
| Product | `fastconv` 0.1.0 · edition 2024 · AGPL-3.0-only · pre-merge / pre-review branch `feat/image-stack-to-animation` (Feature 5/6: `fastconv stack` still-image stack → animated GIF/WebP) · 5 crates |
| Verify | `cargo xtask verify` green — **838** default tests (42 cli + 3 audio_codecs + 13 cli_help + 4 frames + 5 stack + 6 validation_errors + 7 video_add_audio + 5 video_extract + 124 domain + 455 pipeline + 152 probe + 22 xtask), **560** `--features nvenc` tests (42 cli + 3 audio_codecs + 13 cli_help + 4 frames + 5 stack + 6 validation_errors + 7 video_add_audio + 5 video_extract + 475 pipeline); 273 deps license-checked; fmt + clippy `-D warnings` clean (default and nvenc); no-FFmpeg/ffprobe scanner clean in `crates/` |
| Release binary | default **28,834,024 B (27.50 MiB)**; `--features nvenc` **29,488,232 B (28.12 MiB)** — `strip = true`, `lto = "fat"`, `codegen-units = 1`, `opt-level = 3` (both re-built on this branch; unstripped sizes retained from the prior snapshot, not re-measured) |

## Build size
| Profile | Stripped | Unstripped | Strip reduction |
| :--- | ---: | ---: | ---: |
| `default` | 28,834,024 B (27.50 MiB) | 33,745,336 B (32.18 MiB) | −14.6% |
| `--features nvenc` | 29,488,232 B (28.12 MiB) | 34,472,792 B (32.88 MiB) | −14.5% |

Profile: `lto = "fat"`, `codegen-units = 1`, `opt-level = 3`, `strip = true` (stripped) / `strip = none` (unstripped). Stripped sizes re-measured on this branch; unstripped sizes and strip reduction retained from the prior binary-shrink snapshot (not re-measured for Feature 5).

## CLI
| Command | Form |
| :--- | :--- |
| `probe` | `fastconv probe` — CPU model/topology/SIMD/RAM + GPU adapters & media backends |
| `convert` | `fastconv convert <src> <dst> [--to <fmt>] [--quality 0..=100 or --lossless] [--threads N] [--backend <name>]` |
| `audio` | `fastconv audio <src> <dst> [--to <fmt>] [--codec <vorbis\|opus\|flac\|aac\|alac\|pcm\|mp3>] [--bitrate <r> \| --quality 0..=100] [--sample-rate <hz>] [--channels 1..=8] [--bit-depth <8\|16\|24\|32\|f32>] [--track N] [--threads N] [--backend cpu]` — CPU-only; video sources extract an audio track directly |
| `remux` | `fastconv remux <src> <dst> [--to mp4, mkv, webm, mov, avi, ts] [--threads N]` — no re-encode |
| `video` | `fastconv video <src> <dst> [--to <fmt>] [--vcodec <codec>] [--acodec <codec>] [--quality 0..=100 \| --vbitrate <rate>] [--abitrate <rate>] [--asample-rate <hz>] [--achannels N] [--add-audio <path>] [--replace-audio] [--threads N] [--backend <name>] [--hwdecode auto, cpu, nvdec]` — `--add-audio` appends an external audio track (or video-container audio track) to a single video file; `--replace-audio` drops the source tracks first |
| `frames` | `fastconv frames <source> <output-dir> [--to <fmt>] [--quality 0..=100 \| --lossless] [--fps <N> \| --every <N>] [--start <N>] [--end <N>] [--threads <N>] [--backend cpu\|auto]` — CPU-only still-frame extraction; writes `frame_000001.<ext>`, `frame_000002.<ext>`, …; video and animated GIF/WebP inputs; GPU backends rejected explicitly |
| `stack` | `fastconv stack <image-dir \| img1 img2 ...> <target> [--fps <N>] [--to <gif\|webp>] [--quality 0..=100] [--loop <infinite\|N>] [--threads <N>] [--backend cpu\|auto]` — CPU-only still-image stack → animated GIF/WebP; directory input scanned in natural numeric order, explicit image list keeps given order (duplicates kept), target must not alias a source, GPU backends rejected explicitly |
| `help` | `fastconv help <probe, convert, audio, remux, video, frames, stack>`; `-h`/`--help`, `-V`/`--version` |

## Formats
- **Image — 11 in × 12 out.** In: `png jpeg webp bmp qoi tiff jxl avif gif ico pnm` (P1–P7); out: same set + `heic`. Still GIF uses a fixed 256-color palette; `--quality`: webp/jpeg/jxl/avif/heic, `--lossless`: webp/jxl + png/bmp/qoi/tiff/ico/pnm; HEIC works on CPU (pure-Rust `heifer` HEVC-Intra, default backend) and on NVENC GPU (`--backend nvenc`).
- **Audio — 9 in × 9 out + video→audio.** `wav flac opus ogg mp3 aac m4a alac aiff` (Ogg = Vorbis, AAC = ADTS, ALAC in M4A); CPU-only, parallel batches. `fastconv audio` also reads the audio track of `mp4 mov mkv webm avi ts` directly (no intermediate file) into any of the 9 outputs; `--track N` selects the track, default 0.
- **Video — 6 containers** `mp4 mov mkv webm avi ts`; codecs H.264/HEVC (H.265)/AV1/VP9 + AAC/Opus/MP3/FLAC/PCM + `copy`. CPU: H.264/AV1/VP9/GIF/WebP; NVENC: H.264/HEVC; `--hwdecode` NVDEC zero-copy. `--add-audio <path>` accepts WAV/FLAC/MP3/OGG/Opus/AAC/M4A/ALAC/AIFF or audio tracks inside video containers, defaults to AAC in MP4/MOV/AVI/TS and Opus in MKV/WebM, and is rejected for directory batches, `--acodec copy/none` and animation targets; `--replace-audio` requires `--add-audio`.
- **Animation — bidirectional** video ↔ animated GIF/WebP via `fastconv video --to gif,webp --vcodec gif,webp` (CPU + NVENC); GIF/WebP also valid video inputs; `remux` rejects them.
- **Frame extraction — CPU-only** from video (`mp4 mov mkv webm avi ts`, H.264/HEVC/AV1/VP9/MJPEG) and animated GIF/WebP into any of the 12 still formats via `fastconv frames`; inclusive `--start`/`--end`, `--every N` or `--fps N`, bounded recycled pixel buffers (≤256 MiB in flight), streaming encode workers; GPU backends are rejected explicitly (no silent CPU fallback).
- **Image stack → animation — CPU-only** via `fastconv stack <dir | img1 img2 ...> <target.gif|webp>`; any of the 11 still input formats, natural numeric ordering for directories (`frame_2` before `frame_10`), explicit ordered list with duplicates preserved, `--fps` bounded to 1e-6..=1_000_000, `--loop infinite|0|1..=65535`, `--quality` (WebP lossy / GIF quantizer effort), target alias and GPU backend rejected at the probe boundary.

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
| `domain` | `src/{lib,audio,frames,hardware,image,metrics,stack,video}.rs` — pure types/invariants |
| `probe` | `src/{lib,audio,frames,hardware,image,stack,video}.rs` — boundary parsing/validation |
| `pipeline` | `src/{lib,audio,aiff,avif,avi,bmp,cpu,decode,frames,gif,heic,ico,m4a,mpegts,pnm,qoi,remux,stack,staging,video_anim,video_bitstream,video_codecs,video_container,video_transcode}.rs` + `src/gpu/{mod,nvdec,nvenc,heic}.rs` |
| `cli` (`fastconv`) | `src/{main,audio,frames,stack,video}.rs`, `tests/{audio_codecs,cli_help,frames,stack,validation_errors,video_extract,video_add_audio}.rs` — thin orchestration |
| `xtask` | `src/{main,bench,audio_bench,nvenc_bench,remux_bench,fixtures}.rs` — verify / fixtures / bench |

## Invariants
- No FFmpeg in `crates/` (scanner in `cargo xtask verify`); ffmpeg only in `xtask` as bench opponent / fixture generator.
- Boundary validation in `probe`; `domain` holds valid types only; illegal states (audio + GPU, quality on lossless formats, NVENC non-HEIC) are enum-level (`ExecutionPlan`, `ImageExecutionMode`, `ImageCompression`, `GpuMediaKind`); CPU HEIC is a valid CPU image mode.
- Zero silent fallback: failed GPU setup, `--backend vaapi/vulkan`, CPU HEVC video encode, failed NVDEC → hard error; requesting `--backend cpu` for HEIC uses `heifer` directly, not a failed-GPU fallback. `fastconv stack` is CPU-only and rejects GPU backends explicitly.
- `fastconv stack` boundary invariants (probe + domain): uniform source dimensions and format canvas limits (GIF 65,535² / WebP 16,383²), non-empty source set, file target only, no target aliasing any source (including the source directory), `--to` vs extension conflicts, `--fps` in 1e-6..=1e6, `--loop` 0 or 1..=65535, worker count = available CPU threads unless `--threads` is given; mixed dimensions are rejected before any output is created.
- Lints: `unsafe_code = "forbid"`, `unwrap_used`/`expect_used`/`panic`/`todo`/`unimplemented`/`dbg_macro` = `deny`, `clippy::pedantic = "warn"`; pipeline relaxes unsafe only under `--features nvenc`, confined to narrowly scoped `#[allow(unsafe_code)]` FFI blocks in `gpu/nvenc.rs` and `gpu/nvdec.rs`.
- AGPL-3.0-only; 273 deps permissive/AGPL-compatible; `GPL-2.0-only` rejected (Classpath exception allowed).
- RAM caps (actual constants): JXL/AVIF ≤8 image workers, HEIC ≤8 image workers (RGB/RGBA scratch + HEVC tile state), AVIF inner threads ≤8, MP4 faststart ≤64 MiB in-memory (stream above), ≤8 animation workers, ≤2 CPU video file workers, 1 GiB decoded/track, 512 MiB PCM/file, 1 GiB PCM/batch.
- Turing reality: NVENC = HEVC-intra (HEIC) only on the GPU path; CPU HEIC (`heifer`) works on any host; no HW encoders for webp/jpeg/png/bmp/qoi/tiff/jxl/avif/gif/ico/pnm; AV1 CPU-decode; image Vulkan rejected. GPU probe Linux-only (DRM sysfs + NVIDIA procfs).
- Video→audio extraction is CPU-only: `fastconv audio <video.mp4> <out> --codec …` demuxes + decodes the selected audio track (default 0) in-process; out-of-range `--track` and GPU backends are rejected explicitly, no silent CPU fallback.
- External audio is a validated boundary input (`probe/video.rs`): decoded once into the target codec's native format, resampled/remixed as needed, and muxed alongside packet-copied video; `--acodec copy/none`, directory batches, animation containers, duplicate/missing paths and target aliases are rejected before any output is created. `--replace-audio` is opt-in and drops all source audio tracks. Matroska targets reject MP4 presentation edits Matroska cannot represent (no silent timestamp corruption).

## Benchmarks — canonical 2026-11-28 (pre-merge / pre-review branch run, Feature 5 `fastconv stack`)
Host: `Intel Core i7-6700` (4C/8T AVX2/FMA, 32023 MiB) + `RTX 2060` (6 GB, driver `620.73.04`); `ffmpeg n9.1`; all runs serial.

- Provenance (this snapshot): branch `feat/image-stack-to-animation`, HEAD `7f1c0069157ff64f56f70b9404b57d4149cd73f2`. At run time the feature was an uncommitted pre-review working tree (10 modified files in `git diff` + 5 new untracked `stack.rs` files); `git diff | sha256sum` was `b19356d3db5449704c72506cd22123c923df42ac3fa4c1532ba759d06f31bce6` and the untracked files hashed `f368729b1c96992a63005247365444ae4218aa9abe994c5a177ca217ddf16210` (`cli/src/stack.rs`), `3f59a14b23c8d1c5326490a56b3b6b17fd27c576a95e8e706ebe6c2d3c304450` (`cli/tests/stack.rs`), `e129fbe700fca72d31b9126b4310c4017bf63f4ea84901a0ac1340e5dd63e1ba` (`domain/src/stack.rs`), `f91b0cd2541584517953e9c19af73ba7c70cf7fb90ef7458d0d3c0c4b14e4a6a` (`pipeline/src/stack.rs`), `c5c8dfb8726744c2c96f6d16f20fde95e2ff61585717a82b2081cbdd149569e9` (`probe/src/stack.rs`) — captured before this `STATE.md` edit, not reproducible from a merged HEAD. `target/release/fastconv probe` reported `Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz` (4C/8T, x86:avx2+fma, 32023 MiB) and `NVIDIA GeForce RTX 2060` (driver `620.73.04`), media backends `nvenc [encode+decode]` and `vaapi [decode-only]`; `ffmpeg -version` = `ffmpeg version n9.1`. Toolchain: rustc/cargo `1.100.0`, gcc `16.2.1`. Runs 12:35–13:06 CET. `STATE.md` was updated after the runs; no code file was modified by the benchmark.
- Commands: `cargo xtask gen-fixtures`; `cargo build --release -p fastconv-cli` and `cargo build --release -p fastconv-cli --features nvenc` (pre-built because a cold release build exceeds the 180 s guard); `timeout -k 10 180s cargo xtask bench --audio`, `--nvenc`, `--remux`; `timeout -k 10 420s cargo xtask bench` — the identical harness command with a longer guard because the full image bench (including the 20×4K CPU-HEIC comparison) did not finish inside 180 s; it was run twice cleanly and the second run is recorded; `timeout -k 10 300s cargo xtask verify` (green). `fastconv stack` workloads (new, no `xtask bench` target), 3 serial runs/side on identical fixture-derived inputs generated once with `ffmpeg -nostdin -y -hide_banner -loglevel error -i fixtures/video_1080p60.mp4 -vf scale=480:270 -frames:v 60 <tmp>/frame_%06d.png` (60 × 480×270 PNG, 1,835,569 B total): `target/release/fastconv stack <dir> out.gif --fps 15 --quality 80` vs `ffmpeg -nostdin -y -hide_banner -loglevel error -framerate 15 -i <dir>/frame_%06d.png -vf "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" out_ff.gif`; and `target/release/fastconv stack <dir> out.webp --fps 15 --quality 80` vs `ffmpeg -nostdin -y -hide_banner -loglevel error -framerate 15 -i <dir>/frame_%06d.png -c:v libwebp_anim -quality 80 -loop 0 out_ff.webp`. Previous `frames`, add-audio and existing animation rows are retained verbatim from the prior snapshot (not re-run on this branch). Temporary measurement files were removed after the runs and are not cited as permanent paths.
- Peak RAM: `/usr/bin/time` is not installed on this host, so peak RSS was sampled from `VmHWM` in `/proc/<pid>/status` every 1 ms during every run (identical source to the xtask harness's `sample_proc_peak_rss_bytes`); the maximum across the 3 serial runs is reported. Median wall time includes process launch, matching the harness methodology. `stack` throughput uses the 1,835,569 B PNG stack (1.84 MB) as input and counts all 60 emitted frames for FPS.
- Validation: image 20/20 `ffprobe` + decode per row (all 18 harness rows, exact 3840x2160; lossless rows exact RGB, lossy PSNR ≥ 33.6 dB); audio 34/34 outputs (16 single-file + 18 batch) with 18/18 content checks (bit-exact FLAC/AIFF/ALAC, lag-aligned SNR ≥ 26.4 dB single-file, ≥ 104 dB batch); remux 4/4 codec-copy (`aac`+`h264`, 1920x1080, 600 frames, no decoder errors); NVENC 4/4 (600 frames / 10 s, 1920x1080, B-frames 295, `dts_ok=true`, clean decode, H.264 50.72 dB / HEVC 47.62 dB, explicit `gpu:nvdec+nvenc on NVIDIA GeForce RTX 2060`, no CPU fallback). `fastconv stack`: 12/12 decoded outputs, 60 frames each at 480×270 (`gif` / `webp_anim` per `ffprobe`), `ffmpeg -v error -f null -` clean for every run; mean RGB PSNR vs the source PNG stack: GIF 37.48 (fastconv) / 40.23 dB (ffmpeg), WebP 30.06 / 29.99 dB; every run produced byte-identical output (GIF 921,910 B / 692,061 B; WebP 342,294 B / 329,956 B).
- Regression verdict: **all four harness checks green, no correctness regression.** Image absolute times are ~20–25% higher than the prior snapshot on the same host (two clean full runs agreed within 5%: e.g. `png_to_webp` 5106.7/4901.5 ms vs 4012.4 ms prior; ffmpeg 16287.6/14978.3 ms vs 12457.2 ms), so this is machine-wide drift, not a feature-specific slowdown — every output byte count, PSNR, resolution and validation result is unchanged and every speedup remains >1x (3.06x webp, 1.39x jpeg, 2.95x jpeg→webp, 4.48x webp→png, 1.45x qoi, 1.71x bmp, 3.58x tiff, 4.31x jxl, 1.54x avif). NVENC 1884.6 vs 1968.8 ms (1.04x) and 2010.0 vs 2113.4 ms (1.05x); remux 15.5/30.6 vs 98.7/118.8 ms (6.37x/3.88x); audio timings re-run with unchanged quality (fastconv 3.56x flac, 10.90x aiff, 2.94x alac, 3.09x opus, 1.06x ogg, 1.67x mp3, 9.52x aac, 9.87x m4a, 3.59x mixed batch). New `fastconv stack` result: **1.86x (GIF) and 4.26x (WebP) faster than ffmpeg** at the requested `--fps 15 --quality 80` flags (GIF output is not quality-matched — see caveats), with 60/60 validated frames and 2.6x/1.7x lower peak RAM. No runs were discarded or cherry-picked.

### `fastconv stack` — 60 × 480×270 PNG → animated GIF / WebP, 15 fps, Q=80, 3 serial runs/side (median time, max VmHWM)
Input: 60 PNG frames (1,835,569 B total) generated once from `fixtures/video_1080p60.mp4`; both opponents receive the identical stack. Fastconv binary: default `cargo build --release -p fastconv-cli` (CPU-only path, no `--features nvenc` required). Exact flags are in the Commands line above.

| Workload | Tool — exact flags | Time (ms) | FPS | MB/s (input) | Peak RAM | Input | Output | PSNR vs source | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 60 PNG → animated GIF | `fastconv stack <dir> out.gif --fps 15 --quality 80` | 319.75 | 187.64 | 5.74 | 27.25 MiB | 1.84 MB | 921,910 B | 37.48 dB | **1.86x** |
| 60 PNG → animated GIF | `ffmpeg -framerate 15 -i <dir>/frame_%06d.png -vf "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" out_ff.gif` | 595.82 | 100.70 | 3.08 | 97.93 MiB | 1.84 MB | 692,061 B | 40.23 dB | Baseline |
| 60 PNG → animated WebP | `fastconv stack <dir> out.webp --fps 15 --quality 80` | 174.84 | 343.17 | 10.50 | 38.11 MiB | 1.84 MB | 342,294 B | 30.06 dB | **4.26x** |
| 60 PNG → animated WebP | `ffmpeg -framerate 15 -i <dir>/frame_%06d.png -c:v libwebp_anim -quality 80 -loop 0 out_ff.webp` | 745.07 | 80.53 | 2.46 | 64.19 MiB | 1.84 MB | 329,956 B | 29.99 dB | Baseline |

All 3 run times (ms): GIF fastconv 304.85 / 320.52 / 319.75, ffmpeg 589.47 / 619.33 / 595.82; WebP fastconv 176.74 / 170.18 / 174.84, ffmpeg 731.82 / 745.07 / 767.30. Peak RAM is the maximum `VmHWM` over the 3 runs. Every output decoded to 60/60 frames at 480×270 with a clean `ffmpeg -v error -f null -` pass and was byte-identical across runs. WebP is a matched-quality comparison (30.06 vs 29.99 dB, sizes within 3.7%); GIF is not quality-matched — see caveats.

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
| `png_to_webp` | `fastconv (cpu q=80)` | 4901.5 | 61.97 MB/s | 627.3 MiB | 303.75 MB | 5.69 MB | 33.95 dB | **3.06x** |
| `png_to_webp` | `ffmpeg (libwebp q=80)` | 14978.3 | 20.28 MB/s | 665.7 MiB | 303.75 MB | 5.34 MB | 33.61 dB | Baseline |
| `png_to_jpeg` | `fastconv (cpu q=80)` | 732.4 | 414.75 MB/s | 450.4 MiB | 303.75 MB | 13.39 MB | 34.30 dB | **1.39x** |
| `png_to_jpeg` | `ffmpeg (mjpeg qv=3)` | 1014.4 | 299.44 MB/s | 849.3 MiB | 303.75 MB | 8.58 MB | 33.98 dB | Baseline |
| `jpeg_to_webp` | `fastconv (cpu q=80)` | 4583.2 | 5.86 MB/s | 421.9 MiB | 26.86 MB | 5.90 MB | 38.12 dB | **2.95x** |
| `jpeg_to_webp` | `ffmpeg (libwebp q=80)` | 13520.5 | 1.99 MB/s | 165.8 MiB | 26.86 MB | 5.44 MB | 38.40 dB | Baseline |
| `webp_to_png` | `fastconv (cpu png)` | 586.9 | 9.69 MB/s | 528.5 MiB | 5.69 MB | 153.64 MB | exact | **4.48x** |
| `webp_to_png` | `ffmpeg (png)` | 2630.2 | 2.16 MB/s | 820.5 MiB | 5.69 MB | 89.97 MB | exact | Baseline |
| `png_to_qoi` | `fastconv (cpu qoi)` | 761.6 | 398.85 MB/s | 528.1 MiB | 303.75 MB | 250.91 MB | exact | **1.45x** |
| `png_to_qoi` | `ffmpeg (qoi)` | 1107.5 | 274.26 MB/s | 968.4 MiB | 303.75 MB | 250.91 MB | exact | Baseline |
| `webp_to_bmp` | `fastconv (cpu bmp)` | 386.6 | 14.71 MB/s | 612.0 MiB | 5.69 MB | 497.67 MB | exact | **1.71x** |
| `webp_to_bmp` | `ffmpeg (bmp)` | 661.8 | 8.59 MB/s | 484.4 MiB | 5.69 MB | 497.67 MB | exact | Baseline |
| `png_to_tiff` | `fastconv (cpu tiff)` | 1023.0 | 296.93 MB/s | 618.9 MiB | 303.75 MB | 276.93 MB | exact (both Deflate) | **3.58x** |
| `png_to_tiff` | `ffmpeg (tiff)` | 3664.8 | 82.88 MB/s | 1391.6 MiB | 303.75 MB | 373.65 MB | exact | Baseline |
| `png_to_jxl` | `fastconv (cpu q=80)` | 4859.0 | 62.51 MB/s | 1358.1 MiB | 303.75 MB | 1.25 MB | 34.80 dB | **4.31x** (same distance 1.9) |
| `png_to_jxl` | `ffmpeg (libjxl d=1.9)` | 20957.5 | 14.49 MB/s | 953.4 MiB | 303.75 MB | 6.49 MB | 34.95 dB | Baseline |
| `png_to_avif` | `fastconv (cpu q=80)` | 2804.6 | 108.30 MB/s | 1287.1 MiB | 303.75 MB | 0.64 MB | 35.65 dB | **1.54x** (−0.10 dB, 4.5x smaller) |
| `png_to_avif` | `ffmpeg (libsvtav1 preset 8)` | 4327.8 | 70.18 MB/s | 3033.0 MiB | 303.75 MB | 2.86 MB | 35.75 dB | Baseline |
| `png_to_heic` | `fastconv (gpu:nvenc q=80)` | 1189.9 | 255.27 MB/s | 712.1 MiB | 303.75 MB | 27.47 MB | 39.21 dB | **12.59x** vs ff(webp) / **4.12x** vs cpu(webp) |
| `jpeg_to_heic` | `fastconv (gpu:nvenc q=80)` | 935.8 | 28.70 MB/s | 522.4 MiB | 26.86 MB | 19.24 MB | 38.16 dB | **14.45x** vs ff(webp) / **4.90x** vs cpu(webp) |
| `webp_to_heic` | `fastconv (gpu:nvenc q=80)` | 932.6 | 6.10 MB/s | 827.9 MiB | 5.69 MB | 13.21 MB | 38.10 dB | NVENC 20× 4K (no ffmpeg HEIC opponent) |
| `png_to_heic` (single 4K) | `fastconv (cpu heifer q=80, 3 runs)` | 6251.9 | 2.43 MB/s | 135.8 MiB | 15.18 MB | 1.09 MB | 35.45 dB | CPU-only (no ffmpeg HEIC opponent) |
| `png_to_heic` (20× 4K batch) | `fastconv (cpu heifer q=80, 8 workers, 3 runs)` | 122310.5 | 2.48 MB/s | 467.6 MiB | 303.75 MB | 21.92 MB | 35.39 dB mean | CPU-only (no ffmpeg HEIC opponent) |

All harness rows re-run on this branch; the two CPU HEIC rows have no `xtask bench` target and are retained verbatim from the prior snapshot.

### Audio — 60 s stereo 48 kHz/16-bit, target-matched bitrate/quality (achieved size/lag-aligned SNR per row; 34/34 outputs validated: 16 single-file + 18 batch, 18/18 content checks)
| Workload | Tool | Time (ms) | Throughput | Peak RAM | Input | Output | Quality | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wav_to_flac` | `fastconv (flacenc pure rust)` | 54.8 | 210.08 MB/s | 70.3 MiB | 11.52 MB | 4.12 MB | bit-exact | **3.56x** |
| `wav_to_flac` | `ffmpeg (-c:a flac -compression_level 5)` | 195.2 | 59.02 MB/s | 57.4 MiB | 11.52 MB | 4.11 MB | bit-exact | Baseline |
| `wav_to_aiff` | `fastconv (native pcm_s16be)` | 9.0 | 1278.56 MB/s | 7.5 MiB | 11.52 MB | 11.52 MB | bit-exact | **10.90x** |
| `wav_to_aiff` | `ffmpeg (-c:a pcm_s16be)` | 98.1 | 117.44 MB/s | 51.9 MiB | 11.52 MB | 11.52 MB | bit-exact | Baseline |
| `wav_to_alac` | `fastconv (alac-encoder m4a)` | 125.6 | 91.72 MB/s | 32.6 MiB | 11.52 MB | 4.48 MB | bit-exact | **2.94x** |
| `wav_to_alac` | `ffmpeg (-c:a alac)` | 368.9 | 31.23 MB/s | 52.7 MiB | 11.52 MB | 4.69 MB | bit-exact | Baseline |
| `wav_to_opus` | `fastconv (opus-pure)` | 194.0 | 59.39 MB/s | 72.7 MiB | 11.52 MB | 0.98 MB | 37.1 dB | **3.09x** |
| `wav_to_opus` | `ffmpeg (-c:a libopus -b:a 128k)` | 600.1 | 19.20 MB/s | 53.2 MiB | 11.52 MB | 1.35 MB | — | Baseline |
| `wav_to_ogg` | `fastconv (vorbis-enc q4)` | 869.5 | 13.25 MB/s | 14.2 MiB | 11.52 MB | 0.74 MB | 35.8 dB | **1.06x** |
| `wav_to_ogg` | `ffmpeg (-c:a libvorbis -q:a 4)` | 925.4 | 12.45 MB/s | 55.6 MiB | 11.52 MB | 0.70 MB | — | Baseline |
| `wav_to_mp3` | `fastconv (rusty_mp3 192k)` | 615.5 | 18.72 MB/s | 16.2 MiB | 11.52 MB | 1.44 MB | 64.3 dB | **1.67x** |
| `wav_to_mp3` | `ffmpeg (-c:a libmp3lame -b:a 192k)` | 1029.0 | 11.20 MB/s | 54.7 MiB | 11.52 MB | 1.44 MB | — | Baseline |
| `wav_to_aac` | `fastconv (rusty_aac 128k)` | 173.6 | 66.35 MB/s | 61.0 MiB | 11.52 MB | 0.98 MB | 26.4 dB | **9.52x** |
| `wav_to_aac` | `ffmpeg (-c:a aac -b:a 128k)` | 1652.1 | 6.97 MB/s | 55.3 MiB | 11.52 MB | 0.98 MB | — | Baseline |
| `wav_to_m4a` | `fastconv (rusty_aac 128k muxed)` | 168.8 | 68.25 MB/s | 63.5 MiB | 11.52 MB | 0.97 MB | 26.4 dB | **9.87x** |
| `wav_to_m4a` | `ffmpeg (-c:a aac -b:a 128k)` | 1665.9 | 6.92 MB/s | 55.4 MiB | 11.52 MB | 0.97 MB | — | Baseline |
| `audio_batch_mixed_to_flac` | `fastconv (parallel, 9 inputs)` | 237.9 | 25.40 MB/s | 56.6 MiB | 6.04 MB | 5.65 MB | 9/9 validated | **3.59x** |
| `audio_batch_mixed_to_flac` | `ffmpeg (sequential, 9 inputs)` | 853.3 | 7.08 MB/s | 60.4 MiB | 6.04 MB | 6.83 MB | 9/9 validated | Baseline |

### Video — NVENC & Remux, 1080p60, 600 frames, 3 serial runs/side
| Workload | Tool | Time (ms) | FPS | Peak RAM | Output | PSNR | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nvenc_h264_mp4` | `fastconv (nvdec+nvenc)` | 1884.6 | 318.4 | 481.1 MiB | 25.86 MB | 50.72 dB | **1.04x** |
| `nvenc_h264_mp4` | `ffmpeg (h264_nvenc, hwaccel cuda)` | 1968.8 | 304.8 | 631.6 MiB | 25.90 MB | 50.73 dB | Baseline |
| `nvenc_hevc_mp4` | `fastconv (nvdec+nvenc)` | 2010.0 | 298.5 | 594.5 MiB | 20.52 MB | 47.62 dB | **1.05x** |
| `nvenc_hevc_mp4` | `ffmpeg (hevc_nvenc, hwaccel cuda)` | 2113.4 | 283.9 | 581.8 MiB | 20.54 MB | 47.62 dB | Baseline |
| `remux_mp4_to_mkv` | `fastconv (pure rust)` | 15.5 | 38684.7 | 16.4 MiB | 19.16 MB | codec-copy | **6.37x** |
| `remux_mp4_to_mkv` | `ffmpeg (-c copy)` | 98.7 | 6077.5 | 60.5 MiB | 19.14 MB | codec-copy | Baseline |
| `remux_mkv_to_mp4` | `fastconv (pure rust)` | 30.6 | 19628.3 | 55.5 MiB | 19.15 MB | codec-copy | **3.88x** |
| `remux_mkv_to_mp4` | `ffmpeg (-c copy)` | 118.8 | 5049.1 | 57.5 MiB | 19.15 MB | codec-copy | Baseline |

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
- GPU probe is Linux-only (DRM sysfs / NVIDIA procfs); other OSes report no adapters/backends. Animation bench and CPU HEIC image bench have no `xtask` target (external 3-run scripts); those rows are retained unchanged. All `xtask` image, remux, NVENC and audio benches were re-run on this branch; add-audio rows are retained from the prior snapshot. `fastconv frames` and the new `fastconv stack` have no `xtask bench` target either and were measured with the serial 3-run helper described above.
- `audio_batch_mixed_to_flac` measures 237.9 ms / 3.59x, not the stale 109.8 ms / 7.53x number. Merged post-#34 worker budgeting sees no `total_samples` for the AAC ADTS input, reserves the full 512 MiB per-file PCM allowance and caps the 9-file mixed batch at 1 worker (56.6 MiB peak); removing only that input restores 8 workers and ~108 ms, and unmodified HEAD reproduces 1 worker / 217–238 ms. Quality is unchanged (9/9 bit-exact or SNR-validated tracks); this is a stale-baseline correction, not a Feature 3 regression.
- Audio zero-quality-loss: FLAC/AIFF/ALAC remain bit-exact (5,760,000 samples identical single-file, 9/9 batch tracks decoded; wav/flac/alac/aiff batch tracks bit-identical), lossy single-file SNR (37.1/35.8/64.3/26.4/26.4 dB) and every output size match the prior snapshot exactly. Single-file timings re-run here: 54.8/9.0/125.6/194.0/869.5/615.5/173.6/168.8 ms (all slower in absolute terms on this run, with unchanged quality). Smallest margin: `wav_to_ogg` 869.5 vs 925.4 ms (1.06x). No samples were discarded or cherry-picked.
- CPU HEIC throughput is ~2.4 MB/s on this 4C/8T host: `heifer`'s Medium HEVC-Intra preset is intentionally kept (no preset cheating), and the 20-image batch caps at 8 image workers to bound RAM (peak 467.6 MiB, far below 3 GiB). The two CPU HEIC rows have no `xtask bench` target and are retained verbatim from the prior snapshot; the NVENC path was re-measured on this branch at 1189.9 ms and remains a different quality class (27.47 MB / 39.21 dB vs 21.92 MB / 35.39 dB CPU), not a fair speedup comparison.
- `ffprobe` lists CPU-HEIC output as 40 × 512×512 HEVC tile items + a 320×180 thumbnail rather than one 3840×2160 stream; full resolution is confirmed via ffmpeg's `Tile Grid: ... 3840x2160` group and the clean 24,883,200-byte RGB24 decode.
- AVIF/JXL matched PSNR and smaller outputs; TIFF both Deflate, exact; AVIF peak RAM 1287.1 vs 3033.0 MiB (fastconv lower; both under 3 GiB), JXL 1358.1 vs 953.4 MiB.
- Image rows re-run on this branch: 3.06x webp, 1.39x jpeg, 2.95x jpeg→webp, 4.48x webp→png, 1.45x qoi, 1.71x bmp, 3.58x tiff, 4.31x JXL, 1.54x AVIF; NVENC HEIC 12.59x/14.45x vs ffmpeg(webp) with 20/20 decoded 4K outputs. All four lossless image rows remain exact RGB; lossy rows keep PSNR in the same band, but sizes are not uniform: `png_to_jpeg` is 56% larger (13.39 vs 8.58 MB), both WebP rows are within 9%, and JXL/AVIF are smaller than ffmpeg's outputs. Smallest margin: `png_to_jpeg` 732.4 vs 1014.4 ms (1.39x). Absolute times are 20–25% higher than the prior snapshot for *both* tools (two clean full runs agreed within 5%), so this is machine drift rather than a code regression.
- Video timings re-run on this branch (both tools slower in absolute terms than the prior snapshot): NVENC H.264 1884.6 vs 1968.8 ms (1.04x), HEVC 2010.0 vs 2113.4 ms (1.05x); remux mp4→mkv 15.5 vs 98.7 ms (6.37x) and mkv→mp4 30.6 vs 118.8 ms (3.88x). Output sizes, 50.72/47.62 dB PSNR, 600-frame validation, B-frames 295, `dts_ok=true`, clean decode and the explicit `gpu:nvdec+nvenc on NVIDIA GeForce RTX 2060` proof are unchanged in kind; no runs were discarded or cherry-picked.
- `fastconv frames` fairness: A/A2 use lossless Deflate on both sides — `png::Compression::Fastest` (zlib level 1) vs ffmpeg `-compression_level 1`; encoded sizes differ because of filter choices (22.35 vs 25.52 MB), but decoded pixels are compared, not compressed bytes. B size-matches `-q:v 4` (5.29 vs 5.64 MB, within 7%) where ffmpeg is actually 3.0 dB *better* (45.00 vs 41.99 dB); the PSNR-matched `-q:v 7` is 31% smaller (3.88 MB at 42.06 dB), so the slow fastconv result is not bought with inflated output. C uses a lossless VP8L animation so both decoders produce bit-exact RGBA; the lossy animated WebP variant shows the known `zenwebp` vs `libwebp` YUV→RGB delta (29.36 dB) and is not used for the bit-exact claim.
- `fastconv frames` performance reality: CPU video extraction is 4.8x slower than ffmpeg on `--every 10` (4310.13 vs 896.83 ms) because the pure-Rust decoder must decode all 600 frames to select 60, and 1.85x slower even on a 60-frame segment (578.49 vs 313.29 ms). Animation extraction is the strength: GIF 1.91x and lossless WebP 2.30x faster with bit-exact frames and 3–4x lower peak RAM (16–20 MiB vs 62–63 MiB). All 60-frame counts, 1920×1080/480×270 dimensions and PSNR/PSNR-inf checks passed; no runs were discarded.
- `fastconv stack` fairness: the WebP rows are a fair matched-quality comparison (30.06 vs 29.99 dB, 342,294 vs 329,956 B — within 3.7%), so 4.26x is not bought with inflated output. **The GIF rows are not quality- or size-matched**: at the requested Q=80 ffmpeg's `palettegen`/`paletteuse` produces 40.23 dB in 692,061 B while fastconv produces 37.48 dB in 921,910 B (33% larger, 2.75 dB worse). The 1.86x GIF speedup therefore comes at a real quality/size cost and the two GIF rows must not be read as equal-output speedup; no alternative fastconv quality flag was tried to cherry-pick a better sample.
- `fastconv stack` performance reality: a CPU-only 60-frame 480×270 stack encodes in 319.75 ms (GIF, 187.64 fps, 27.25 MiB peak) / 174.84 ms (WebP, 343.17 fps, 38.11 MiB peak) vs ffmpeg 595.82 / 745.07 ms with 97.93 / 64.19 MiB peak; all 12 outputs decoded to 60 frames at 480×270 with clean `ffmpeg -v error -f null -` passes and byte-identical repeat runs. The binary used for these rows was the default (non-nvenc) release build; `--features nvenc` is unnecessary because GPU backends are rejected by design.
- Benchmark guard/measurement note: `timeout -k 10 180s cargo xtask bench` could not finish the full image suite (it includes the 20×4K CPU-HEIC comparison), so the identical command was run under a 420 s guard twice with no other heavy processes; the second run is recorded and the first agreed within 5%. `/usr/bin/time` is unavailable on this host, so peak RSS for the `fastconv stack` rows was sampled from `VmHWM` in `/proc/<pid>/status` every 1 ms, the same source the xtask harness uses.
