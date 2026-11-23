# fastconv — Übergabe-Blatt (Stand: 2026-11-23)

## Aktuelle Fähigkeiten
- **CLI-Befehle**: `fastconv probe`, `fastconv convert <source> <target> [--quality <0..=100> | --lossless] [--to <webp|heic|jpeg>] [--threads <N>] [--backend <cpu|nvenc|vaapi|qsv|amf|videotoolbox|vulkan>]`, `--help`, `--version`
- **Probe**: CPU (Topologie, Memory, SIMD: AVX2/AVX-512/NEON), `wgpu`-Adapter (Vulkan), GPU-Backends (NVENC, VA-API, Vulkan Compute), Magic-Byte-Erkennung (`PNG`/`JPEG`) + Header-/Farbraum-Validierung (`Luma`, `YCbCr`, `RGB`, `CMYK`, `YCCK`) für Einzeldateien und gemischte `PNG`+`JPEG`-Ordner-Batches
- **N-zu-M Bild-Pipeline (`crates/pipeline`)**: Entkoppeltes Decode (`crates/pipeline/src/decode.rs`) und Encode über wiederverwendbare Worker-Puffer:
  - **Inputs**: `PNG` (`png`) und `JPEG` (`zune-jpeg`, inkl. direktem `JPEG -> RGBA8`-Pfad für NVENC)
  - **Outputs**: `WebP` (`zenwebp` CPU AVX2, Lossy/Lossless), `JPEG` (`jpeg-encoder` CPU-SIMD AVX2, Lossy), `HEIC` (`--features nvenc` NVIDIA NVENC HEVC-Intra `ABGR` + nativer pure-Rust ISO/IEC 23008-12 Muxer)
- **Video/Audio-Pipeline**: Generisches `execute(&ExecutionPlan)` in `crates/pipeline` liefert noch `PipelineError::BackendNotImplemented`
- **Workload-Benchmarks (`cargo xtask bench`)**: 20× 4K PNG & 20× 4K JPEG Head-to-Head (`PNG -> WebP`, `PNG -> JPEG`, `JPEG -> WebP`, optional `PNG -> HEIC` & `JPEG -> HEIC` auf NVENC) gegen `ffmpeg`, inkl. Dekodier-, Auflösungs-, Größen- und RGB-PSNR-Validierung (`bench-output/head-to-head.json` Schema v3)

## Dateikarte
- `crates/cli/src/main.rs`: CLI-Einstiegspunkt, Argument-Parsing (`--to`, `--quality`, `--lossless`, `--threads`, `--backend`), Dispatch für `probe` & `convert`
- `crates/domain/src/lib.rs`: Kern-Typen (`MediaKind`, `GpuMediaKind`, `GpuBackend`, `HardwareBackend`, `ConversionRequest`, `ExecutionPlan`, `DomainError`)
- `crates/domain/src/hardware.rs`: Validierte Hardware-Domain-Typen (`HostHardware`, `CpuInfo`, `SimdLevel`, `WgpuAdapterInfo`, `DetectedGpuBackend`)
- `crates/domain/src/image.rs`: Bild-Domain-Typen (`ImageInputFormat`, `ImageOutputFormat`, `ImageCompression`, `ImageExecutionMode`, `ImageBatchItem`, `ImageBatchPlan`, `ImageConversionReport`)
- `crates/domain/src/metrics.rs`: Benchmark- und Metrik-Typen (`BenchmarkMetrics`, `BenchmarkSample`, `PeakMemory`, `Throughput`, `WorkloadUnit`)
- `crates/probe/src/lib.rs`: Systemgrenze für OS-File-Checks, CPU-Parallelität & `ProbeError`
- `crates/probe/src/hardware.rs`: OS-Hardware-Erkennung (`/proc/cpuinfo`, `/proc/meminfo`, `wgpu`-Adapter, NVENC/VA-API Dynamic-Lib-Probe)
- `crates/probe/src/image.rs`: Magic-Byte-Sniffing (`PNG`/`JPEG`), JPEG-Farbraum-/Header-Prüfung und Aufbau validierter Einzel- sowie gemischter Batch-Pläne (`prepare_image_batch`)
- `crates/pipeline/src/lib.rs`: Dispatcher (`execute`, `execute_image_batch`) und `PipelineError`
- `crates/pipeline/src/decode.rs`: Gemeinsamer PNG/JPEG-Decoder (`decode_image_into`) mit `DecodeTargetHint` und wiederverwendbaren Puffern
- `crates/pipeline/src/cpu.rs`: Paralleler CPU-Worker-Pool mit per-Thread `WorkerScratch`-Puffern für `CpuImageEncoder::{Webp, Jpeg}` (Video/Audio-Stub in `execute`)
- `crates/pipeline/src/gpu/mod.rs`: GPU-Zero-Copy-Stub (`PipelineError::BackendNotImplemented`) für Video-Pläne
- `crates/pipeline/src/gpu/nvenc.rs`: Überlappendes CPU-Decode + GPU-NVENC-HEVC-Intra-Encoding (`ABGR` mit gepoolten `RGBA8`-Puffern, BT.601 Limited VUI)
- `crates/pipeline/src/gpu/heic.rs`: Pure-Rust Annex-B-HEVC-Parser (`VPS`/`SPS`/`PPS`/`IDR`) und ISO/IEC 23008-12 (`ftypheic`/`mif1`) Container-Muxer
- `xtask/src/main.rs`: Orchestrierung für `verify` (No-FFmpeg-Scanner, `fmt`, `clippy -D warnings`, Tests inkl. `--features nvenc`), `gen-fixtures` & `bench`
- `xtask/src/fixtures.rs`: Deterministische Testmedien (20× 4K PNG, 20× 4K JPEG, 60s WAV, 1080p60 MP4)
- `xtask/src/bench.rs`: Head-to-Head Benchmark-Harness (`fastconv` vs. `ffmpeg`) mit Peak-RSS-Messung, PSNR-/Auflösungs-Checks und JSON-Export

## Harte Architektur-Entscheidungen
- **Kein FFmpeg-Wrapper**: `crates/` darf null `ffmpeg`/`ffprobe`-Subprozesse oder -Bindings enthalten; verifiziert per Code-Scanner in `cargo xtask verify`.
- **Zero Silent Fallback**: Schlägt GPU-Setup fehl oder existiert kein passender GPU-Encoder (z. B. `vaapi` decode-only, `vulkan` compute-only, `nvenc` ohne Feature/Hardware oder für WebP/JPEG), bricht die Ausführung hart ab; kein stiller CPU-Fallback.
- **Type-State Discipline**: Illegale Zustände wie Audio + GPU (`GpuMediaKind`), `--lossless` mit `quality`, CPU + HEIC, NVENC + WebP/JPEG oder Vulkan Compute für Bilder sind über Enums (`ImageExecutionMode`, `ImageCompression`, `ExecutionPlan`) zur Compile-Zeit unmöglich.
- **Boundary Discipline**: Dateisystem-Checks, Magic-Byte-Sniffing, Bild-Header- & JPEG-Farbraum-Prüfungen finden in `crates/probe` statt (`crates/pipeline/src/decode.rs` prüft zusätzlich Defense-in-Depth gegen Dimension-Drift, nicht-opakes Alpha für JPEG/HEIC und `MultiBand`-Farbräume).
- **Empirische Hardware-Limits RTX 2060 (Turing TU106)**:
  - NVENC beschleunigt HEVC-Intra (`HEIC`, `png_to_heic` & `jpeg_to_heic`), bietet aber keinen Hardware-Encoder für WebP, JPEG oder AV1.
  - Vulkan-Compute lohnt sich weder für WebP noch für JPEG: Auf i7-6700 + RTX 2060 braucht `zune-jpeg` (8T CPU-SIMD) `168.2 ms` für 20× 4K RGB-Decode und `jpeg-encoder` (`simd`, `Nearest`) `408.3 ms` für 20× 4K Encode, während `wgpu` bereits für Cold-Init `234.0 ms` plus `153.4–160.6 ms` PCIe-Roundtrip (nur YCbCr↔RGB/RGBA ohne DCT/Huffman) benötigt. WebP & JPEG laufen daher bewusst auf CPU-SIMD; `--backend vulkan` wird explizit abgelehnt.
- **Lint-Disziplin**: Workspace-weit gelten `unsafe_code = "forbid"`, `unwrap_used = "deny"`, `expect_used = "deny"`, `panic = "deny"`, `clippy::pedantic = "warn"`. Nur `crates/pipeline` setzt lokal `unsafe_code = "deny"` mit `#![cfg_attr(not(feature = "nvenc"), forbid(unsafe_code))]` und genau zwei dokumentierten `#[allow(unsafe_code)]`-Blöcken unter `#[cfg(feature = "nvenc")]` in `crates/pipeline/src/gpu/nvenc.rs` (`NV_ENC_CODEC_CONFIG.hevcConfig` & `BufferLock::write`).

## Aktuelle Baseline-Zahlen
- **Host**: `Intel(R) Core(TM) i7-6700` (4C/8T, `x86:avx2+fma`, 32 GiB RAM) + `NVIDIA GeForce RTX 2060` (6 GB, Vulkan/NVENC, Treiber `620.73.04`)
- **Validation**: `20/20` Outputs je Workload vollständig dekodiert, exakt `3840x2160` und gegen Quell-RGB (`PSNR >= 25.0 dB`) geprüft (`cargo xtask bench`, 20× 4K Fixtures, Q=80 bzw. FFmpeg `mjpeg -q:v 3`):

| Workload | Tool | Zeit (ms) | Durchsatz | Peak RAM | Input | Output | Mean PSNR | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `image_batch_4k_png_to_webp` | `fastconv (cpu q=80)` | 5108.5 | 59.46 MB/s | 473.2 MiB | 303.75 MB | 5.69 MB | 33.95 dB | **3.05x** (+0.35 dB) |
| `image_batch_4k_png_to_webp` | `ffmpeg (libwebp q=80)` | 15579.4 | 19.50 MB/s | 671.0 MiB | 303.75 MB | 5.34 MB | 33.61 dB | Baseline |
| `image_batch_4k_png_to_jpeg` | `fastconv (cpu q=80)` | 740.4 | 410.25 MB/s | 318.8 MiB | 303.75 MB | 13.33 MB | 34.29 dB | **1.39x** (+0.32 dB) |
| `image_batch_4k_png_to_jpeg` | `ffmpeg (mjpeg qv=3)` | 1031.8 | 294.38 MB/s | 867.6 MiB | 303.75 MB | 8.58 MB | 33.98 dB | Baseline |
| `image_batch_4k_jpeg_to_webp` | `fastconv (cpu q=80)` | 4597.8 | 5.84 MB/s | 361.7 MiB | 26.86 MB | 5.90 MB | 38.12 dB | **3.18x** (-0.28 dB) |
| `image_batch_4k_jpeg_to_webp` | `ffmpeg (libwebp q=80)` | 14601.9 | 1.84 MB/s | 165.7 MiB | 26.86 MB | 5.44 MB | 38.40 dB | Baseline |
| `image_batch_4k_png_to_heic` | `fastconv (gpu:nvenc q=80)` | 1110.9 | 273.41 MB/s | 658.3 MiB | 303.75 MB | 27.47 MB | 39.21 dB | **14.02x** vs ff(webp) / **4.60x** vs cpu(webp) |
| `image_batch_4k_jpeg_to_heic` | `fastconv (gpu:nvenc q=80)` | 1128.2 | 23.81 MB/s | 615.9 MiB | 26.86 MB | 19.24 MB | 38.16 dB | **12.94x** vs ff(webp) / **4.08x** vs cpu(webp) |
