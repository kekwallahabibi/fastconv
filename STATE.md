# fastconv — Übergabe-Blatt (Stand: 2026-11-23)

## Aktuelle Fähigkeiten
- **CLI-Befehle**: `fastconv probe`, `fastconv convert <source> <target> [--quality <0..=100> | --lossless] [--to <webp|heic|jpeg|png|bmp|qoi|tiff|jxl|avif>] [--threads <N>] [--backend <cpu|nvenc|vaapi|qsv|amf|videotoolbox|vulkan>]`, `--help`, `--version`
- **Probe**: CPU (Topologie, Memory, SIMD: AVX2/AVX-512/NEON), `wgpu`-Adapter (Vulkan), GPU-Backends (NVENC, VA-API, Vulkan Compute), strikte Magic-Byte-Erkennung (`PNG`, `JPEG`, `WebP`, `BMP`, `QOI`, `TIFF` little/big-endian, `JPEG XL` bare codestream `0xFF 0x0A` und ISOBMFF-Container `JXL `) + Header-/Chunk-/Masken-/Farbraum-/Bitdepth-Validierung für Einzeldateien und gemischte Ordner-Batches (ignoriert nicht-Bild-Dateien sicher)
- **N-zu-M Bild-Pipeline (`crates/pipeline`)**: Entkoppeltes Decode (`crates/pipeline/src/decode.rs`) und Encode über wiederverwendbare Worker-Puffer (7 Inputs × 9 Outputs = 63 Kombinationen; AVIF-Input ist bewusst nicht implementiert):
  - **Inputs**: `PNG` (`png`), `JPEG` (`zune-jpeg`), `WebP` (`zenwebp` VP8/VP8L/VP8X Still-Images), `BMP` (nativer safe-Rust-Decoder: 1/4/8-Bit Palette, 16-Bit, 24-Bit BGR, 32-Bit BGRA & `BI_BITFIELDS`/`BI_ALPHABITFIELDS`, Top-Down & Bottom-Up), `QOI` (nativer safe-Rust-Decoder 3/4 Kanäle), `TIFF` (`tiff` 0.11.3: Gray/GrayA/RGB/RGBA, 1..=16 Bit, U8/U16 Downconversion), `JPEG XL` (`jxl` 0.8.0: Grayscale/RGB mit optionalem Alpha); direkter `OpaqueRgba8`-Decode-Pfad für `JPEG`/`BMP`/`QOI`/`TIFF`/`JXL` Richtung NVENC
  - **Outputs**: `WebP` (`zenwebp` 0.2.0 MIT/Apache, CPU AVX2, Lossy/Lossless), `JPEG` (`jpeg-encoder` CPU-SIMD AVX2, Lossy), `PNG` (`png` Streaming mit `Compression::Fastest` = `fdeflate` + `Filter::Up`, 256 KiB IDAT / 512 KiB `BufWriter`, Lossless), `BMP` (nativer safe-Rust-Encoder mit 2 MiB Chunk-Batching ohne doppelte Pufferkopie, 24-Bit BGR / 32-Bit V4 BGRA / 8-Bit Gray, Lossless), `QOI` (nativer safe-Rust-Encoder mit L1-Cache-Stack-Staging, Lossless), `TIFF` (`tiff` 0.11.3, Deflate Fast, Gray8/RGB8/RGBA8; LA8 wird als RGBA8 kodiert), `JPEG XL` (`jixel` 0.5.13, Lossless modular oder lossy VarDCT mit Butteraugli-Distance; Speed Fastest, `--quality 100` wird auf Lossless abgebildet), `AVIF` (`ravif` 0.13.0, rav1e-basiert, lossy AV1 mit Alpha, ASM aktiv, Speed 7, pro Worker seriell mit einem Thread), `HEIC` (`--features nvenc` NVIDIA NVENC HEVC-Intra `ABGR` + nativer pure-Rust ISO/IEC 23008-12 Muxer)
- **Video/Audio-Pipeline**: Generisches `execute(&ExecutionPlan)` in `crates/pipeline` liefert noch `PipelineError::BackendNotImplemented`
- **Workload-Benchmarks (`cargo xtask bench`)**: 20× 4K PNG, 20× 4K JPEG & 20× 4K WebP Head-to-Head (`PNG -> WebP`, `PNG -> JPEG`, `JPEG -> WebP`, `WebP -> PNG`, `PNG -> QOI`, `WebP -> BMP`, `PNG -> TIFF`, `PNG -> JPEG XL`, `PNG -> AVIF`, optional `PNG/JPEG/WebP -> HEIC` auf NVENC) gegen `ffmpeg` (gleiche Compression/Distance für TIFF und JXL, `libsvtav1` Preset 8 für AVIF), inkl. Dekodier-, Auflösungs-, Größen-, exakter RGB-Pixel-Gleichheits- (Lossless gegen eigene Decoder-Referenz) bzw. RGB-PSNR-Validierung (`bench-output/head-to-head.json` Schema v4)

## Dateikarte
- `crates/cli/src/main.rs`: CLI-Einstiegspunkt, Argument-Parsing (`--to`, `--quality`, `--lossless`, `--threads`, `--backend`), Dispatch für `probe` & `convert`
- `crates/domain/src/lib.rs`: Kern-Typen (`MediaKind`, `GpuMediaKind`, `GpuBackend`, `HardwareBackend`, `ConversionRequest`, `ExecutionPlan`, `DomainError`)
- `crates/domain/src/hardware.rs`: Validierte Hardware-Domain-Typen (`HostHardware`, `CpuInfo`, `SimdLevel`, `WgpuAdapterInfo`, `DetectedGpuBackend`)
- `crates/domain/src/image.rs`: Bild-Domain-Typen (`ImageInputFormat`, `ImageOutputFormat`, `ImageCompression`, `ImageExecutionMode`, `ImageBatchItem`, `ImageBatchPlan`, `ImageConversionReport`)
- `crates/domain/src/metrics.rs`: Benchmark- und Metrik-Typen (`BenchmarkMetrics`, `BenchmarkSample`, `PeakMemory`, `Throughput`, `WorkloadUnit`)
- `crates/probe/src/lib.rs`: Systemgrenze für OS-File-Checks, CPU-Parallelität & `ProbeError`
- `crates/probe/src/hardware.rs`: OS-Hardware-Erkennung (`/proc/cpuinfo`, `/proc/meminfo`, `wgpu`-Adapter, NVENC/VA-API Dynamic-Lib-Probe)
- `crates/probe/src/image.rs`: Magic-Byte-Sniffing (`PNG`/`JPEG`/`WebP`/`BMP`/`QOI`/`TIFF`/`JPEG XL`), Header-/Farbraum-/Bitdepth-Prüfungen und Aufbau validierter Einzel- sowie gemischter Batch-Pläne (`prepare_image_batch`)
- `crates/pipeline/src/lib.rs`: Dispatcher (`execute`, `execute_image_batch`) und `PipelineError`
- `crates/pipeline/src/decode.rs`: Gemeinsamer Decoder (`decode_image_into`) für PNG/JPEG/WebP/BMP/QOI/TIFF/JXL mit `DecodeTargetHint` und wiederverwendbaren Puffern
- `crates/pipeline/src/bmp.rs`: Pure-Rust BMP-Decoder und 2-MiB-Chunk-gebatchter BMP-Encoder (24-Bit BGR, 32-Bit V4 BGRA, 8-Bit Gray, ungerade Breiten & `BI_BITFIELDS`)
- `crates/pipeline/src/qoi.rs`: Pure-Rust QOI-Decoder und L1-gestagter QOI-Encoder (RGB8/RGBA8/L8/LA8, inkl. direktem `OpaqueRgba8`-Pfad)
- `crates/pipeline/src/cpu.rs`: Paralleler CPU-Worker-Pool mit per-Thread `WorkerScratch`-Puffern für `CpuImageEncoder::{Webp, Jpeg, Png, Bmp, Qoi, Tiff, Jxl, Avif}` (Video/Audio-Stub in `execute`)
- `crates/pipeline/src/gpu/mod.rs`: GPU-Zero-Copy-Stub (`PipelineError::BackendNotImplemented`) für Video-Pläne
- `crates/pipeline/src/gpu/nvenc.rs`: Überlappendes CPU-Decode (alle 7 Input-Formate) + GPU-NVENC-HEVC-Intra-Encoding (`ABGR` mit gepoolten `RGBA8`-Puffern, BT.601 Limited VUI)
- `crates/pipeline/src/gpu/heic.rs`: Pure-Rust Annex-B-HEVC-Parser (`VPS`/`SPS`/`PPS`/`IDR`) und ISO/IEC 23008-12 (`ftypheic`/`mif1`) Container-Muxer
- `xtask/src/main.rs`: Orchestrierung für `verify` (No-FFmpeg-Scanner, `fmt`, `clippy -D warnings`, Tests inkl. `--features nvenc`), `gen-fixtures` & `bench`
- `xtask/src/fixtures.rs`: Deterministische Testmedien (20× 4K PNG, 20× 4K JPEG, 20× 4K WebP, 60s WAV, 1080p60 MP4)
- `xtask/src/bench.rs`: Head-to-Head Benchmark-Harness (`fastconv` vs. `ffmpeg`) mit Peak-RSS-Messung, Lossless-Exact-RGB- und PSNR-Validierung sowie JSON-Export

## Harte Architektur-Entscheidungen
- **Kein FFmpeg-Wrapper**: `crates/` darf null `ffmpeg`/`ffprobe`-Subprozesse oder -Bindings enthalten; verifiziert per Code-Scanner in `cargo xtask verify`.
- **Zero Silent Fallback**: Schlägt GPU-Setup fehl oder existiert kein passender GPU-Encoder (z. B. `vaapi` decode-only, `vulkan` compute-only, `nvenc` ohne Feature/Hardware oder für WebP/JPEG/PNG/BMP/QOI/TIFF/JXL/AVIF), bricht die Ausführung hart ab; kein stiller CPU-Fallback.
- **Type-State Discipline**: Illegale Zustände wie Audio + GPU (`GpuMediaKind`), `--quality` auf verlustfreien Formaten (`PNG`/`BMP`/`QOI`/`TIFF`), `--lossless` auf rein verlustbehafteten Formaten (`JPEG`/`HEIC`/`AVIF`), CPU + HEIC, NVENC + Nicht-HEIC oder Vulkan Compute für Bilder sind über Enums (`ImageExecutionMode`, `ImageCompression`, `ExecutionPlan`) zur Compile-Zeit unmöglich. Ohne explizites Flag wählt `Option<ImageCompression>` automatisch sinnvolle Format-Defaults (`Q=80` für WebP/JPEG/JXL/AVIF/HEIC, verlustfrei für PNG/BMP/QOI/TIFF).
- **Boundary Discipline**: Dateisystem-Checks, Magic-Byte-Sniffing und Header-/Farbraum-/Bitmask-Prüfungen finden in `crates/probe` statt (`crates/pipeline/src/decode.rs` prüft zusätzlich Defense-in-Depth gegen Dimension-Drift, nicht-opakes Alpha für JPEG/HEIC und ungültige Payloads).
- **GPU-Realitätscheck RTX 2060 (Turing TU106)**: NVENC beschleunigt HEVC-Intra (`HEIC`, `png_to_heic`, `jpeg_to_heic` & `webp_to_heic`), bietet aber keinen Hardware-Encoder für WebP, JPEG, PNG, BMP, QOI, TIFF, JPEG XL oder AVIF. TIFF/JXL/AVIF-Outputs werden deshalb explizit mit klaren Fehlermeldungen auf `--backend cpu` zurückgewiesen, während TIFF/JXL-Inputs weiterhin für NVENC-HEIC decodiert werden. Vulkan-Compute lohnt sich für keines der CPU-Bildformate: Auf i7-6700 + RTX 2060 verschlingt `wgpu` allein `234.0 ms` Cold-Init + `153.4–160.6 ms` PCIe-Roundtrip (`~389 ms` Overhead vor der ersten Compute-Operation). Auf 8T CPU-SIMD laufen die kompletten 20× 4K-Encode-Phasen dagegen deutlich darunter oder sind inhärent sequenziell pro Bild. `--backend vulkan` bleibt daher für alle Bild-Outputs sauber abgelehnt.
- **Lint-Disziplin**: Workspace-weit gelten `unsafe_code = "forbid"`, `unwrap_used = "deny"`, `expect_used = "deny"`, `panic = "deny"`, `clippy::pedantic = "warn"`. Nur `crates/pipeline` setzt lokal `unsafe_code = "deny"` mit `#![cfg_attr(not(feature = "nvenc"), forbid(unsafe_code))]` und genau zwei dokumentierten `#[allow(unsafe_code)]`-Blöcken unter `#[cfg(feature = "nvenc")]` in `crates/pipeline/src/gpu/nvenc.rs` (`NV_ENC_CODEC_CONFIG.hevcConfig` & `BufferLock::write`). `bmp.rs` und `qoi.rs` sind 100 % safe Rust.

## Aktuelle Baseline-Zahlen
- **Host**: `Intel(R) Core(TM) i7-6700` (4C/8T, `x86:avx2+fma`, 32 GiB RAM) + `NVIDIA GeForce RTX 2060` (6 GB, Vulkan/NVENC, Treiber `620.73.04`)
- **Validation**: `20/20` Outputs je Workload vollständig dekodiert, exakt `3840x2160` und je nach Pfad auf exakte RGB-Pixel-Gleichheit (`exact`, ohne Alpha-Prüfung, jeweils gegen die eigene Decoder-Referenz) bzw. Quell-RGB-PSNR (`>= 25.0 dB`) geprüft (`cargo xtask bench`, 20× 4K Fixtures, Q=80 bzw. FFmpeg `mjpeg -q:v 3`):

| Workload | Tool | Zeit (ms) | Durchsatz | Peak RAM | Input | Output | Qualität | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `image_batch_4k_png_to_webp` | `fastconv (cpu q=80)` | 10337.5 | 29.38 MB/s | 413.3 MiB | 303.75 MB | 4.60 MB | 33.63 dB | **1.40x** (+0.02 dB) |
| `image_batch_4k_png_to_webp` | `ffmpeg (libwebp q=80)` | 14471.4 | 20.99 MB/s | 666.2 MiB | 303.75 MB | 5.34 MB | 33.61 dB | Baseline |
| `image_batch_4k_png_to_jpeg` | `fastconv (cpu q=80)` | 813.1 | 373.55 MB/s | 320.9 MiB | 303.75 MB | 13.33 MB | 34.29 dB | **1.50x** (+0.32 dB) |
| `image_batch_4k_png_to_jpeg` | `ffmpeg (mjpeg qv=3)` | 1218.9 | 249.19 MB/s | 881.8 MiB | 303.75 MB | 8.58 MB | 33.98 dB | Baseline |
| `image_batch_4k_jpeg_to_webp` | `fastconv (cpu q=80)` | 9587.8 | 2.80 MB/s | 309.6 MiB | 26.86 MB | 5.20 MB | 37.54 dB | **1.35x** (-0.86 dB) |
| `image_batch_4k_jpeg_to_webp` | `ffmpeg (libwebp q=80)` | 12971.8 | 2.07 MB/s | 165.7 MiB | 26.86 MB | 5.44 MB | 38.40 dB | Baseline |
| `image_batch_4k_webp_to_png` | `fastconv (cpu png)` | 656.5 | 8.66 MB/s | 302.8 MiB | 5.69 MB | 153.64 MB | exact | **4.21x** |
| `image_batch_4k_webp_to_png` | `ffmpeg (png)` | 2765.1 | 2.06 MB/s | 815.8 MiB | 5.69 MB | 89.97 MB | exact | Baseline |
| `image_batch_4k_png_to_qoi` | `fastconv (cpu qoi)` | 698.9 | 434.63 MB/s | 411.8 MiB | 303.75 MB | 250.91 MB | exact | **3.12x** |
| `image_batch_4k_png_to_qoi` | `ffmpeg (qoi)` | 2180.8 | 139.28 MB/s | 1009.4 MiB | 303.75 MB | 250.91 MB | exact | Baseline |
| `image_batch_4k_webp_to_bmp` | `fastconv (cpu bmp)` | 1224.3 | 4.64 MB/s | 318.6 MiB | 5.69 MB | 497.67 MB | exact | **0.44x** |
| `image_batch_4k_webp_to_bmp` | `ffmpeg (bmp)` | 537.8 | 10.57 MB/s | 484.5 MiB | 5.69 MB | 497.67 MB | exact | Baseline |
| `image_batch_4k_png_to_tiff` | `fastconv (cpu tiff)` | 3024.7 | 100.42 MB/s | 445.0 MiB | 303.75 MB | 339.05 MB | exact | **2.18x** vs ffmpeg (beide Deflate) |
| `image_batch_4k_png_to_tiff` | `ffmpeg (tiff deflate)` | 6593.7 | 46.07 MB/s | 1333.8 MiB | 303.75 MB | 373.65 MB | exact | Baseline |
| `image_batch_4k_png_to_jxl` | `fastconv (cpu q=80)` | 4620.0 | 65.75 MB/s | 622.0 MiB | 303.75 MB | 1.25 MB | 34.80 dB | **4.33x** vs ffmpeg (bei gleicher Distance, ~5x kleiner) |
| `image_batch_4k_png_to_jxl` | `ffmpeg (libjxl d=1.9)` | 20004.7 | 15.18 MB/s | 906.1 MiB | 303.75 MB | 6.49 MB | 34.95 dB | Baseline |
| `image_batch_4k_png_to_avif` | `fastconv (cpu q=80)` | 157402.7 | 1.93 MB/s | 1804.2 MiB | 303.75 MB | 3.62 MB | 36.40 dB | **0.03x** vs FFmpeg-Speed-Preset (bei höherem PSNR) |
| `image_batch_4k_png_to_avif` | `ffmpeg (libsvtav1 preset 8)` | 4702.3 | 64.59 MB/s | 2933.0 MiB | 303.75 MB | 2.86 MB | 35.75 dB | Baseline |
| `image_batch_4k_png_to_heic` | `fastconv (gpu:nvenc q=80)` | 1034.9 | 293.51 MB/s | 662.1 MiB | 303.75 MB | 27.47 MB | 39.21 dB | **13.98x** vs ff(webp) / **9.99x** vs cpu(webp) |
| `image_batch_4k_jpeg_to_heic` | `fastconv (gpu:nvenc q=80)` | 985.2 | 27.26 MB/s | 514.9 MiB | 26.86 MB | 19.24 MB | 38.16 dB | **13.17x** vs ff(webp) / **9.73x** vs cpu(webp) |
| `image_batch_4k_webp_to_heic` | `fastconv (gpu:nvenc q=80)` | 1254.4 | 4.53 MB/s | 653.8 MiB | 5.69 MB | 13.21 MB | 38.10 dB | NVENC 20× 4K (38.10 dB) |

## Crate-Wahl und Lizenzen
- **WebP**: `zenwebp = "=0.2.0"` (pure Rust, `MIT OR Apache-2.0`) – neu pinned, damit der Lizenz-Scan keine AGPL-Abhängigkeit im Default-Graph findet.
- **TIFF**: `tiff = "0.11.3"` (pure Rust, MIT) mit `Compressor::Deflate(DeflateLevel::Fast)`; Decoder akzeptiert Gray/GrayA/RGB/RGBA mit 1..=16 Bit und konvertiert auf 8 Bit.
- **JPEG XL Decode**: `jxl = "0.8.0"` (jxl-rs, pure Rust, BSD-3-Clause, SIMD).
- **JPEG XL Encode**: `jixel = "0.5.13"` (reines Rust, Lossless modular + lossy VarDCT, Alpha über `encode_image_with_alpha`; `--quality` wird über `distance_from_quality` auf Butteraugli-Distance gemappt, `Speed::Fastest` (A/B: 5.6 s / 39.6 KB / 34.98 dB auf `Medium` vs. 0.81 s / 61.8 KB / 35.16 dB auf `Fastest` pro 4K-Bild), 1 Thread pro Worker). Lizenz `BSD-3-Clause OR Apache-2.0`; `cargo xtask verify` scannt per `cargo metadata` den kompletten aufgelösten Dependency-Graphen (212 Pakete) und blockt starkes Copyleft ohne Ausnahmeklausel.
- **AVIF Encode**: `ravif = "0.13.0"` (rav1e-basiert, pure Rust, BSD-3-Clause, `encode_rgba`, Alpha-Qualität, Speed 7); das `asm`-Feature ist aktiv (`ubuntu-latest` installiert NASM über den CI-Workflow); pro Worker wird `with_num_threads(Some(1))` gesetzt, um Worker-Thread-Oversubscription zu vermeiden.
- **AVIF Input**: bewusst optional/out of scope; kein reiner Rust-Decoder wurde in die aktuelle Abhängigkeitsmenge aufgenommen. Die Pipeline bietet AVIF-Output für alle 7 Inputs, aber keine AVIF-Eingabe.
- **RAM-Deckel**: Schwere 4K-Encoder werden gedeckelt (JXL max. 4, AVIF max. 3 parallele Bilder; leichte Formate bleiben auf allen Workern). Finaler Peak-RAM 4K-JXL 622.0 MiB (vorher 3303 MiB), 4K-AVIF 1804.2 MiB (vorher 2665 MiB).
- **Lizenz-Scan**: `cargo xtask verify` nutzt `cargo metadata --format-version 1` und prüft `license`/`license_file` aller aufgelösten Pakete; SPDX-Ausdrücke mit `AGPL-*` oder `GPL-*` ohne `WITH`-Ausnahme werden hart abgelehnt (Unit-Test in `xtask/src/main.rs`).
- **GPU-Realität**: RTX 2060 hat keine Hardware-Encoder für TIFF/JXL/AVIF; TIFF/JXL/AVIF-Ausgaben werden auf `--backend nvenc` explizit abgelehnt, TIFF/JXL-Inputs werden aber für NVENC-HEIC decodiert.