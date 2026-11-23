# fastconv — Übergabe-Blatt (Stand: 2026-11-23)

## Aktuelle Fähigkeiten
- **CLI-Befehle**: `fastconv probe`, `fastconv convert <source> <target> [--quality <0..=100> | --lossless] [--to <webp|heic|jpeg|png|bmp|qoi>] [--threads <N>] [--backend <cpu|nvenc|vaapi|qsv|amf|videotoolbox|vulkan>]`, `--help`, `--version`
- **Probe**: CPU (Topologie, Memory, SIMD: AVX2/AVX-512/NEON), `wgpu`-Adapter (Vulkan), GPU-Backends (NVENC, VA-API, Vulkan Compute), strikte Magic-Byte-Erkennung (`PNG`, `JPEG`, `WebP`, `BMP`, `QOI`) + Header-/Chunk-/Masken-/Farbraum-Validierung für Einzeldateien und gemischte 5-Format-Ordner-Batches (ignoriert nicht-Bild-Dateien wie `"BM..."`-Textdateien sicher)
- **N-zu-M Bild-Pipeline (`crates/pipeline`)**: Entkoppeltes Decode (`crates/pipeline/src/decode.rs`) und Encode über wiederverwendbare Worker-Puffer (5 Inputs × 6 Outputs = 30 Kombinationen):
  - **Inputs**: `PNG` (`png`), `JPEG` (`zune-jpeg`), `WebP` (`zenwebp` VP8/VP8L/VP8X Still-Images), `BMP` (nativer safe-Rust-Decoder: 1/4/8-Bit Palette, 16-Bit, 24-Bit BGR, 32-Bit BGRA & `BI_BITFIELDS`/`BI_ALPHABITFIELDS`, Top-Down & Bottom-Up), `QOI` (nativer safe-Rust-Decoder 3/4 Kanäle); direkter `OpaqueRgba8`-Decode-Pfad für `JPEG`/`BMP`/`QOI` Richtung NVENC
  - **Outputs**: `WebP` (`zenwebp` CPU AVX2, Lossy/Lossless), `JPEG` (`jpeg-encoder` CPU-SIMD AVX2, Lossy), `PNG` (`png` Streaming mit `Compression::Fastest` = `fdeflate` + `Filter::Up`, 256 KiB IDAT / 512 KiB `BufWriter`, Lossless), `BMP` (nativer safe-Rust-Encoder mit 2 MiB Chunk-Batching ohne doppelte Pufferkopie, 24-Bit BGR / 32-Bit V4 BGRA / 8-Bit Gray, Lossless), `QOI` (nativer safe-Rust-Encoder mit L1-Cache-Stack-Staging, Lossless), `HEIC` (`--features nvenc` NVIDIA NVENC HEVC-Intra `ABGR` + nativer pure-Rust ISO/IEC 23008-12 Muxer)
- **Video/Audio-Pipeline**: Generisches `execute(&ExecutionPlan)` in `crates/pipeline` liefert noch `PipelineError::BackendNotImplemented`
- **Workload-Benchmarks (`cargo xtask bench`)**: 20× 4K PNG, 20× 4K JPEG & 20× 4K WebP Head-to-Head (`PNG -> WebP`, `PNG -> JPEG`, `JPEG -> WebP`, `WebP -> PNG`, `PNG -> QOI`, `WebP -> BMP`, optional `PNG/JPEG/WebP -> HEIC` auf NVENC) gegen `ffmpeg` (erfordert `ffmpeg` >= 5.1 mit QOI-Decoder/-Encoder sowie `ffprobe`), inkl. Dekodier-, Auflösungs-, Größen-, exakter RGB-Pixel-Gleichheits- (Lossless gegen eigene Decoder-Referenz) bzw. RGB-PSNR-Validierung (`bench-output/head-to-head.json` Schema v4)

## Dateikarte
- `crates/cli/src/main.rs`: CLI-Einstiegspunkt, Argument-Parsing (`--to`, `--quality`, `--lossless`, `--threads`, `--backend`), Dispatch für `probe` & `convert`
- `crates/domain/src/lib.rs`: Kern-Typen (`MediaKind`, `GpuMediaKind`, `GpuBackend`, `HardwareBackend`, `ConversionRequest`, `ExecutionPlan`, `DomainError`)
- `crates/domain/src/hardware.rs`: Validierte Hardware-Domain-Typen (`HostHardware`, `CpuInfo`, `SimdLevel`, `WgpuAdapterInfo`, `DetectedGpuBackend`)
- `crates/domain/src/image.rs`: Bild-Domain-Typen (`ImageInputFormat`, `ImageOutputFormat`, `ImageCompression`, `ImageExecutionMode`, `ImageBatchItem`, `ImageBatchPlan`, `ImageConversionReport`)
- `crates/domain/src/metrics.rs`: Benchmark- und Metrik-Typen (`BenchmarkMetrics`, `BenchmarkSample`, `PeakMemory`, `Throughput`, `WorkloadUnit`)
- `crates/probe/src/lib.rs`: Systemgrenze für OS-File-Checks, CPU-Parallelität & `ProbeError`
- `crates/probe/src/hardware.rs`: OS-Hardware-Erkennung (`/proc/cpuinfo`, `/proc/meminfo`, `wgpu`-Adapter, NVENC/VA-API Dynamic-Lib-Probe)
- `crates/probe/src/image.rs`: Magic-Byte-Sniffing (`PNG`/`JPEG`/`WebP`/`BMP`/`QOI`), feste Header-/Bitmask-Reads (bzw. konstanter Scratch-Space mit RIFF-Chunk-Scan in `probe_webp_vp8x`, dessen I/O linear mit der Chunk-Anzahl wächst) und Aufbau validierter Einzel- sowie gemischter Batch-Pläne (`prepare_image_batch`)
- `crates/pipeline/src/lib.rs`: Dispatcher (`execute`, `execute_image_batch`) und `PipelineError`
- `crates/pipeline/src/decode.rs`: Gemeinsamer 5-Format-Decoder (`decode_image_into`) mit `DecodeTargetHint` und wiederverwendbaren Puffern
- `crates/pipeline/src/bmp.rs`: Pure-Rust BMP-Decoder und 2-MiB-Chunk-gebatchter BMP-Encoder (24-Bit BGR, 32-Bit V4 BGRA, 8-Bit Gray, ungerade Breiten & `BI_BITFIELDS`)
- `crates/pipeline/src/qoi.rs`: Pure-Rust QOI-Decoder und L1-gestagter QOI-Encoder (RGB8/RGBA8/L8/LA8, inkl. direktem `OpaqueRgba8`-Pfad)
- `crates/pipeline/src/cpu.rs`: Paralleler CPU-Worker-Pool mit per-Thread `WorkerScratch`-Puffern für `CpuImageEncoder::{Webp, Jpeg, Png, Bmp, Qoi}` (Video/Audio-Stub in `execute`)
- `crates/pipeline/src/gpu/mod.rs`: GPU-Zero-Copy-Stub (`PipelineError::BackendNotImplemented`) für Video-Pläne
- `crates/pipeline/src/gpu/nvenc.rs`: Überlappendes CPU-Decode (alle 5 Input-Formate) + GPU-NVENC-HEVC-Intra-Encoding (`ABGR` mit gepoolten `RGBA8`-Puffern, BT.601 Limited VUI)
- `crates/pipeline/src/gpu/heic.rs`: Pure-Rust Annex-B-HEVC-Parser (`VPS`/`SPS`/`PPS`/`IDR`) und ISO/IEC 23008-12 (`ftypheic`/`mif1`) Container-Muxer
- `xtask/src/main.rs`: Orchestrierung für `verify` (No-FFmpeg-Scanner, `fmt`, `clippy -D warnings`, Tests inkl. `--features nvenc`), `gen-fixtures` & `bench`
- `xtask/src/fixtures.rs`: Deterministische Testmedien (20× 4K PNG, 20× 4K JPEG, 20× 4K WebP, 60s WAV, 1080p60 MP4)
- `xtask/src/bench.rs`: Head-to-Head Benchmark-Harness (`fastconv` vs. `ffmpeg`) mit Peak-RSS-Messung, Lossless-Exact-RGB- und PSNR-Validierung sowie JSON-Export

## Harte Architektur-Entscheidungen
- **Kein FFmpeg-Wrapper**: `crates/` darf null `ffmpeg`/`ffprobe`-Subprozesse oder -Bindings enthalten; verifiziert per Code-Scanner in `cargo xtask verify`.
- **Zero Silent Fallback**: Schlägt GPU-Setup fehl oder existiert kein passender GPU-Encoder (z. B. `vaapi` decode-only, `vulkan` compute-only, `nvenc` ohne Feature/Hardware oder für WebP/JPEG/PNG/BMP/QOI), bricht die Ausführung hart ab; kein stiller CPU-Fallback.
- **Type-State Discipline**: Illegale Zustände wie Audio + GPU (`GpuMediaKind`), `--quality` auf verlustfreien Formaten (`PNG`/`BMP`/`QOI`), `--lossless` auf rein verlustbehafteten Formaten (`JPEG`/`HEIC`), CPU + HEIC, NVENC + Nicht-HEIC oder Vulkan Compute für Bilder sind über Enums (`ImageExecutionMode`, `ImageCompression`, `ExecutionPlan`) zur Compile-Zeit unmöglich. Ohne explizites Flag wählt `Option<ImageCompression>` automatisch sinnvolle Format-Defaults (`Q=80` für WebP/JPEG/HEIC, verlustfrei für PNG/BMP/QOI).
- **Boundary Discipline**: Dateisystem-Checks, Magic-Byte-Sniffing und Header-/Farbraum-/Bitmask-Prüfungen finden in `crates/probe` statt (`crates/pipeline/src/decode.rs` prüft zusätzlich Defense-in-Depth gegen Dimension-Drift, nicht-opakes Alpha für JPEG/HEIC und ungültige Payloads).
- **Empirische Hardware-Limits RTX 2060 (Turing TU106) & GPU-Realitätscheck**:
  - NVENC beschleunigt HEVC-Intra (`HEIC`, `png_to_heic`, `jpeg_to_heic` & `webp_to_heic`), bietet aber keinen Hardware-Encoder für WebP, JPEG, PNG, BMP, QOI oder AV1.
  - Vulkan-Compute lohnt sich für keines der CPU-Bildformate (`WebP`, `JPEG`, `PNG`, `BMP`, `QOI`): Auf i7-6700 + RTX 2060 verschlingt `wgpu` allein `234.0 ms` Cold-Init + `153.4–160.6 ms` PCIe-Roundtrip (`~389 ms` Overhead vor der ersten Compute-Operation). Auf 8T CPU-SIMD laufen die kompletten 20× 4K-Encode-Phasen dagegen deutlich darunter oder sind inhärent sequenziell pro Bild (`BMP` komplett inkl. 498 MB Disk-I/O in `530.7 ms` / `392.8 ms` tmpfs; `PNG` `fdeflate` + `Filter::Up` komplett in `647.0 ms` mit sequenziellem Deflate; `QOI` komplett in `793.3 ms` als strikt sequenzielle Running-Hash/Run-State-Machine, die über Bilder auf CPU-Kernen perfekt parallelisiert). `--backend vulkan` bleibt daher für alle Bild-Outputs sauber abgelehnt.
- **Lint-Disziplin**: Workspace-weit gelten `unsafe_code = "forbid"`, `unwrap_used = "deny"`, `expect_used = "deny"`, `panic = "deny"`, `clippy::pedantic = "warn"`. Nur `crates/pipeline` setzt lokal `unsafe_code = "deny"` mit `#![cfg_attr(not(feature = "nvenc"), forbid(unsafe_code))]` und genau zwei dokumentierten `#[allow(unsafe_code)]`-Blöcken unter `#[cfg(feature = "nvenc")]` in `crates/pipeline/src/gpu/nvenc.rs` (`NV_ENC_CODEC_CONFIG.hevcConfig` & `BufferLock::write`). `bmp.rs` und `qoi.rs` sind 100 % safe Rust.

## Aktuelle Baseline-Zahlen
- **Host**: `Intel(R) Core(TM) i7-6700` (4C/8T, `x86:avx2+fma`, 32 GiB RAM) + `NVIDIA GeForce RTX 2060` (6 GB, Vulkan/NVENC, Treiber `620.73.04`)
- **Validation**: `20/20` Outputs je Workload vollständig dekodiert, exakt `3840x2160` und je nach Pfad auf exakte RGB-Pixel-Gleichheit (`exact`, ohne Alpha-Prüfung, jeweils gegen die eigene Decoder-Referenz) bzw. Quell-RGB-PSNR (`>= 25.0 dB`) geprüft (`cargo xtask bench`, 20× 4K Fixtures, Q=80 bzw. FFmpeg `mjpeg -q:v 3`):

| Workload | Tool | Zeit (ms) | Durchsatz | Peak RAM | Input | Output | Qualität | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `image_batch_4k_png_to_webp` | `fastconv (cpu q=80)` | 4974.6 | 61.06 MB/s | 470.6 MiB | 303.75 MB | 5.69 MB | 33.95 dB | **3.31x** (+0.35 dB) |
| `image_batch_4k_png_to_webp` | `ffmpeg (libwebp q=80)` | 16445.6 | 18.47 MB/s | 666.2 MiB | 303.75 MB | 5.34 MB | 33.61 dB | Baseline |
| `image_batch_4k_png_to_jpeg` | `fastconv (cpu q=80)` | 834.9 | 363.83 MB/s | 319.0 MiB | 303.75 MB | 13.33 MB | 34.29 dB | **1.30x** (+0.32 dB) |
| `image_batch_4k_png_to_jpeg` | `ffmpeg (mjpeg qv=3)` | 1086.9 | 279.45 MB/s | 841.3 MiB | 303.75 MB | 8.58 MB | 33.98 dB | Baseline |
| `image_batch_4k_jpeg_to_webp` | `fastconv (cpu q=80)` | 4773.4 | 5.63 MB/s | 361.2 MiB | 26.86 MB | 5.90 MB | 38.12 dB | **3.02x** (-0.28 dB) |
| `image_batch_4k_jpeg_to_webp` | `ffmpeg (libwebp q=80)` | 14396.6 | 1.87 MB/s | 165.5 MiB | 26.86 MB | 5.44 MB | 38.40 dB | Baseline |
| `image_batch_4k_webp_to_png` | `fastconv (cpu png)` | 647.0 | 8.79 MB/s | 394.9 MiB | 5.69 MB | 153.64 MB | exact | **4.30x** |
| `image_batch_4k_webp_to_png` | `ffmpeg (png)` | 2779.2 | 2.05 MB/s | 816.1 MiB | 5.69 MB | 89.97 MB | exact | Baseline |
| `image_batch_4k_png_to_qoi` | `fastconv (cpu qoi)` | 793.3 | 382.90 MB/s | 408.8 MiB | 303.75 MB | 250.91 MB | exact | **1.50x** |
| `image_batch_4k_png_to_qoi` | `ffmpeg (qoi)` | 1187.9 | 255.69 MB/s | 984.8 MiB | 303.75 MB | 250.91 MB | exact | Baseline |
| `image_batch_4k_webp_to_bmp` | `fastconv (cpu bmp)` | 530.7 | 10.72 MB/s | 410.8 MiB | 5.69 MB | 497.67 MB | exact | **1.02x** |
| `image_batch_4k_webp_to_bmp` | `ffmpeg (bmp)` | 541.5 | 10.50 MB/s | 484.6 MiB | 5.69 MB | 497.67 MB | exact | Baseline |
| `image_batch_4k_png_to_heic` | `fastconv (gpu:nvenc q=80)` | 1061.9 | 286.04 MB/s | 658.9 MiB | 303.75 MB | 27.47 MB | 39.21 dB | **15.49x** vs ff(webp) / **4.68x** vs cpu(webp) |
| `image_batch_4k_jpeg_to_heic` | `fastconv (gpu:nvenc q=80)` | 1041.1 | 25.80 MB/s | 514.7 MiB | 26.86 MB | 19.24 MB | 38.16 dB | **13.83x** vs ff(webp) / **4.58x** vs cpu(webp) |
| `image_batch_4k_webp_to_heic` | `fastconv (gpu:nvenc q=80)` | 1023.7 | 5.55 MB/s | 722.2 MiB | 5.69 MB | 13.21 MB | 38.10 dB | NVENC 20× 4K (38.10 dB) |
