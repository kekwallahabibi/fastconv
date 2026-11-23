use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use fastconv_domain::{
    BenchmarkMetrics, BenchmarkSample, GpuBackend, HostHardware, ImageInputFormat,
    ImageOutputFormat, MediaKind, PeakMemory, WorkloadUnit,
};
use serde::Serialize;
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

use crate::fixtures::{self, IMAGE_BATCH_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH, VIDEO_TOTAL_FRAMES};

const MIN_PLAUSIBLE_4K_IMAGE_BYTES: u64 = 1024;
const MIN_LOSSY_PSNR_DB: f64 = 25.0;

#[derive(Clone, Copy)]
enum QualityValidationMode {
    LossyMinPsnrDb(f64),
    LosslessExactPixels,
}

#[derive(Clone, Copy)]
struct ValidationPolicy {
    min_output_bytes: u64,
    require_smaller_than_source: bool,
    mode: QualityValidationMode,
}

const PNG_TO_LOSSY_POLICY: ValidationPolicy = ValidationPolicy {
    min_output_bytes: MIN_PLAUSIBLE_4K_IMAGE_BYTES,
    require_smaller_than_source: true,
    mode: QualityValidationMode::LossyMinPsnrDb(MIN_LOSSY_PSNR_DB),
};

const COMPRESSED_TO_LOSSY_POLICY: ValidationPolicy = ValidationPolicy {
    min_output_bytes: MIN_PLAUSIBLE_4K_IMAGE_BYTES,
    require_smaller_than_source: false,
    mode: QualityValidationMode::LossyMinPsnrDb(MIN_LOSSY_PSNR_DB),
};

const LOSSLESS_EXACT_POLICY: ValidationPolicy = ValidationPolicy {
    min_output_bytes: MIN_PLAUSIBLE_4K_IMAGE_BYTES,
    require_smaller_than_source: false,
    mode: QualityValidationMode::LosslessExactPixels,
};

#[derive(Clone, Copy)]
enum SourceBatchFormat {
    Png,
    Jpeg,
    Webp,
    WebpViaFfmpeg,
}

impl SourceBatchFormat {
    const fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Webp | Self::WebpViaFfmpeg => "webp",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Webp | Self::WebpViaFfmpeg => "WebP",
        }
    }

    fn decode_to_rgb8(self, path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
        match self {
            Self::Png => decode_png_to_rgb8(path),
            Self::Jpeg => decode_jpeg_to_rgb8(path),
            Self::Webp => decode_webp_to_rgb8(path),
            Self::WebpViaFfmpeg => decode_image_to_rgb8_via_ffmpeg(path, "WebP"),
        }
    }
}

#[derive(Serialize)]
struct BaselineReport {
    schema_version: u32,
    ffmpeg_version: String,
    hardware: HostHardware,
    results: Vec<BenchmarkMetrics>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BatchValidationSummary {
    files_validated: usize,
    resolution: String,
    total_output_bytes: NonZeroU64,
    exact_pixel_match: bool,
    min_psnr_db: Option<f64>,
    mean_psnr_db: Option<f64>,
}

#[derive(Serialize)]
struct HeadToHeadEntry {
    metrics: BenchmarkMetrics,
    validation: BatchValidationSummary,
}

#[derive(Serialize)]
struct HeadToHeadComparison {
    workload_name: String,
    fastconv_speedup_vs_ffmpeg: f64,
    /// True when both contenders matched their own decoded reference RGB pixels losslessly
    /// (for lossy inputs like WebP, each tool is validated against its own decoder reference).
    both_match_reference_rgb: bool,
    fastconv_psnr_delta_db: Option<f64>,
}

#[derive(Serialize)]
struct HeadToHeadReport {
    schema_version: u32,
    ffmpeg_version: String,
    hardware: HostHardware,
    png_to_webp_fastconv: HeadToHeadEntry,
    png_to_webp_ffmpeg: HeadToHeadEntry,
    png_to_jpeg_fastconv: HeadToHeadEntry,
    png_to_jpeg_ffmpeg: HeadToHeadEntry,
    jpeg_to_webp_fastconv: HeadToHeadEntry,
    jpeg_to_webp_ffmpeg: HeadToHeadEntry,
    webp_to_png_fastconv: HeadToHeadEntry,
    webp_to_png_ffmpeg: HeadToHeadEntry,
    png_to_qoi_fastconv: HeadToHeadEntry,
    png_to_qoi_ffmpeg: HeadToHeadEntry,
    webp_to_bmp_fastconv: HeadToHeadEntry,
    webp_to_bmp_ffmpeg: HeadToHeadEntry,
    png_to_heic_nvenc: Option<HeadToHeadEntry>,
    jpeg_to_heic_nvenc: Option<HeadToHeadEntry>,
    webp_to_heic_nvenc: Option<HeadToHeadEntry>,
    comparisons: Vec<HeadToHeadComparison>,
}

/// Runs head-to-head benchmarks across 20× 4K PNG, JPEG, and WebP fixtures against `ffmpeg`
/// (including `WebP -> PNG`, `PNG -> QOI`, `WebP -> BMP`, and optional NVENC HEIC), validating
/// every generated image for decodability, exact resolution, size plausibility, and exact RGB
/// equality (lossless vs. each contender's decoder reference) or RGB PSNR (lossy).
///
/// Requires an installed `ffmpeg` build (>= 5.1) with QOI decoder and encoder support
/// alongside `ffprobe` for lossless/HEIC validation.
///
/// # Errors
///
/// Returns an error if fixtures are missing, `ffmpeg` lacks required QOI codec support,
/// `ffprobe` is unavailable, building or executing any contender fails, or any output image
/// fails decode/resolution/size/quality validation.
pub fn run_head_to_head(root: &Path) -> Result<(), String> {
    let hardware = fastconv_probe::detect_host_hardware()
        .map_err(|err| format!("failed to probe host hardware: {err}"))?;
    print_hardware_summary(&hardware);

    let fixtures = verify_head_to_head_fixtures(root)?;
    let ffmpeg_version = read_ffmpeg_version()?;
    println!("ffmpeg        : {ffmpeg_version}");
    verify_ffmpeg_qoi_support()?;
    verify_ffprobe_available()?;

    let has_nvenc =
        hardware.supports_video_encode(GpuBackend::Nvenc) && crate::has_nvenc_link_libraries();
    let fastconv_bin = build_release_fastconv(root, has_nvenc)?;

    let bench_dir = root.join("bench-output");
    let h2h_tmp = bench_dir.join("tmp/head_to_head");
    if h2h_tmp.exists() {
        fs::remove_dir_all(&h2h_tmp)
            .map_err(|err| format!("failed to clean '{}': {err}", h2h_tmp.display()))?;
    }

    let report = run_head_to_head_suite(
        &fastconv_bin,
        &fixtures,
        &h2h_tmp,
        has_nvenc,
        ffmpeg_version,
        hardware,
    )?;

    if h2h_tmp.exists() {
        fs::remove_dir_all(&h2h_tmp).map_err(|err| {
            format!(
                "failed to remove temporary outputs '{}': {err}",
                h2h_tmp.display()
            )
        })?;
    }

    let mut rows: Vec<&HeadToHeadEntry> = Vec::with_capacity(15);
    rows.push(&report.png_to_webp_fastconv);
    rows.push(&report.png_to_webp_ffmpeg);
    rows.push(&report.png_to_jpeg_fastconv);
    rows.push(&report.png_to_jpeg_ffmpeg);
    rows.push(&report.jpeg_to_webp_fastconv);
    rows.push(&report.jpeg_to_webp_ffmpeg);
    rows.push(&report.webp_to_png_fastconv);
    rows.push(&report.webp_to_png_ffmpeg);
    rows.push(&report.png_to_qoi_fastconv);
    rows.push(&report.png_to_qoi_ffmpeg);
    rows.push(&report.webp_to_bmp_fastconv);
    rows.push(&report.webp_to_bmp_ffmpeg);
    if let Some(ref heic) = report.png_to_heic_nvenc {
        rows.push(heic);
    }
    if let Some(ref heic) = report.jpeg_to_heic_nvenc {
        rows.push(heic);
    }
    if let Some(ref heic) = report.webp_to_heic_nvenc {
        rows.push(heic);
    }

    println!();
    print!("{}", format_head_to_head_table(&rows));
    print_head_to_head_summary(&report);

    let json_path = bench_dir.join("head-to-head.json");
    let json = serde_json::to_string_pretty(&report)
        .map_err(|err| format!("failed to serialize head-to-head report: {err}"))?;
    fs::write(&json_path, format!("{json}\n"))
        .map_err(|err| format!("failed to write '{}': {err}", json_path.display()))?;

    println!("head-to-head saved: {}", json_path.display());
    Ok(())
}

struct FastconvBenchJob<'a> {
    workload_name: &'a str,
    tool_label: &'a str,
    source_dir: &'a Path,
    source_format: SourceBatchFormat,
    input_bytes: NonZeroU64,
    out_dir: &'a Path,
    target_format: ImageOutputFormat,
    quality: Option<&'a str>,
    backend: &'a str,
    policy: ValidationPolicy,
}

const fn bench_target_cli_arg(format: ImageOutputFormat) -> &'static str {
    match format {
        ImageOutputFormat::Webp => "webp",
        ImageOutputFormat::Heic => "heic",
        ImageOutputFormat::Jpeg => "jpeg",
        ImageOutputFormat::Png => "png",
        ImageOutputFormat::Bmp => "bmp",
        ImageOutputFormat::Qoi => "qoi",
    }
}

struct LossyRuns {
    png_to_webp_fastconv: HeadToHeadEntry,
    png_to_webp_ffmpeg: HeadToHeadEntry,
    png_to_jpeg_fastconv: HeadToHeadEntry,
    png_to_jpeg_ffmpeg: HeadToHeadEntry,
    jpeg_to_webp_fastconv: HeadToHeadEntry,
    jpeg_to_webp_ffmpeg: HeadToHeadEntry,
}

struct LosslessRuns {
    webp_to_png_fastconv: HeadToHeadEntry,
    webp_to_png_ffmpeg: HeadToHeadEntry,
    png_to_qoi_fastconv: HeadToHeadEntry,
    png_to_qoi_ffmpeg: HeadToHeadEntry,
    webp_to_bmp_fastconv: HeadToHeadEntry,
    webp_to_bmp_ffmpeg: HeadToHeadEntry,
}

struct NvencHeicRuns {
    png: Option<HeadToHeadEntry>,
    jpeg: Option<HeadToHeadEntry>,
    webp: Option<HeadToHeadEntry>,
}

struct DecodedImageRef<'a> {
    rgb: &'a [u8],
    width: u32,
    height: u32,
    label: &'a str,
}

fn create_h2h_dir(base: &Path, name: &str) -> Result<PathBuf, String> {
    let dir = base.join(name);
    fs::create_dir_all(&dir)
        .map_err(|err| format!("failed to create '{}': {err}", dir.display()))?;
    Ok(dir)
}

fn run_head_to_head_suite(
    fastconv_bin: &Path,
    fixtures: &VerifiedHeadToHeadFixtures,
    h2h_tmp: &Path,
    has_nvenc: bool,
    ffmpeg_version: String,
    hardware: HostHardware,
) -> Result<HeadToHeadReport, String> {
    let lossy = run_lossy_contenders(fastconv_bin, fixtures, h2h_tmp)?;
    let lossless = run_lossless_contenders(fastconv_bin, fixtures, h2h_tmp)?;
    let nvenc = run_nvenc_heic_contenders(fastconv_bin, fixtures, h2h_tmp, has_nvenc)?;
    Ok(build_head_to_head_report(
        ffmpeg_version,
        hardware,
        lossy,
        lossless,
        nvenc,
    ))
}

fn run_lossy_contenders(
    fastconv_bin: &Path,
    fixtures: &VerifiedHeadToHeadFixtures,
    h2h_tmp: &Path,
) -> Result<LossyRuns, String> {
    let png_to_webp_fastconv = run_fastconv_image_batch(
        fastconv_bin,
        &FastconvBenchJob {
            workload_name: "image_batch_4k_png_to_webp",
            tool_label: "fastconv (cpu q=80)",
            source_dir: &fixtures.images_dir,
            source_format: SourceBatchFormat::Png,
            input_bytes: fixtures.image_batch_bytes,
            out_dir: &create_h2h_dir(h2h_tmp, "fastconv_png_to_webp")?,
            target_format: ImageOutputFormat::Webp,
            quality: Some("80"),
            backend: "cpu",
            policy: PNG_TO_LOSSY_POLICY,
        },
    )?;
    let png_to_webp_ffmpeg = run_ffmpeg_webp_batch(
        "image_batch_4k_png_to_webp",
        &fixtures.images_dir,
        SourceBatchFormat::Png,
        fixtures.image_batch_bytes,
        &create_h2h_dir(h2h_tmp, "ffmpeg_png_to_webp")?,
        PNG_TO_LOSSY_POLICY,
    )?;

    let png_to_jpeg_fastconv = run_fastconv_image_batch(
        fastconv_bin,
        &FastconvBenchJob {
            workload_name: "image_batch_4k_png_to_jpeg",
            tool_label: "fastconv (cpu q=80)",
            source_dir: &fixtures.images_dir,
            source_format: SourceBatchFormat::Png,
            input_bytes: fixtures.image_batch_bytes,
            out_dir: &create_h2h_dir(h2h_tmp, "fastconv_png_to_jpeg")?,
            target_format: ImageOutputFormat::Jpeg,
            quality: Some("80"),
            backend: "cpu",
            policy: PNG_TO_LOSSY_POLICY,
        },
    )?;
    let png_to_jpeg_ffmpeg = run_ffmpeg_jpeg_batch(
        "image_batch_4k_png_to_jpeg",
        &fixtures.images_dir,
        SourceBatchFormat::Png,
        fixtures.image_batch_bytes,
        &create_h2h_dir(h2h_tmp, "ffmpeg_png_to_jpeg")?,
        PNG_TO_LOSSY_POLICY,
    )?;

    let jpeg_to_webp_fastconv = run_fastconv_image_batch(
        fastconv_bin,
        &FastconvBenchJob {
            workload_name: "image_batch_4k_jpeg_to_webp",
            tool_label: "fastconv (cpu q=80)",
            source_dir: &fixtures.jpeg_images_dir,
            source_format: SourceBatchFormat::Jpeg,
            input_bytes: fixtures.jpeg_batch_bytes,
            out_dir: &create_h2h_dir(h2h_tmp, "fastconv_jpeg_to_webp")?,
            target_format: ImageOutputFormat::Webp,
            quality: Some("80"),
            backend: "cpu",
            policy: COMPRESSED_TO_LOSSY_POLICY,
        },
    )?;
    let jpeg_to_webp_ffmpeg = run_ffmpeg_webp_batch(
        "image_batch_4k_jpeg_to_webp",
        &fixtures.jpeg_images_dir,
        SourceBatchFormat::Jpeg,
        fixtures.jpeg_batch_bytes,
        &create_h2h_dir(h2h_tmp, "ffmpeg_jpeg_to_webp")?,
        COMPRESSED_TO_LOSSY_POLICY,
    )?;

    Ok(LossyRuns {
        png_to_webp_fastconv,
        png_to_webp_ffmpeg,
        png_to_jpeg_fastconv,
        png_to_jpeg_ffmpeg,
        jpeg_to_webp_fastconv,
        jpeg_to_webp_ffmpeg,
    })
}

fn run_lossless_contenders(
    fastconv_bin: &Path,
    fixtures: &VerifiedHeadToHeadFixtures,
    h2h_tmp: &Path,
) -> Result<LosslessRuns, String> {
    let webp_to_png_fastconv = run_fastconv_image_batch(
        fastconv_bin,
        &FastconvBenchJob {
            workload_name: "image_batch_4k_webp_to_png",
            tool_label: "fastconv (cpu png)",
            source_dir: &fixtures.webp_images_dir,
            source_format: SourceBatchFormat::Webp,
            input_bytes: fixtures.webp_batch_bytes,
            out_dir: &create_h2h_dir(h2h_tmp, "fastconv_webp_to_png")?,
            target_format: ImageOutputFormat::Png,
            quality: None,
            backend: "cpu",
            policy: LOSSLESS_EXACT_POLICY,
        },
    )?;
    let webp_to_png_ffmpeg = run_ffmpeg_lossless_batch(
        "image_batch_4k_webp_to_png",
        "ffmpeg (png)",
        &fixtures.webp_images_dir,
        SourceBatchFormat::WebpViaFfmpeg,
        fixtures.webp_batch_bytes,
        &create_h2h_dir(h2h_tmp, "ffmpeg_webp_to_png")?,
        ImageOutputFormat::Png,
        &["-c:v", "png"],
    )?;

    let png_to_qoi_fastconv = run_fastconv_image_batch(
        fastconv_bin,
        &FastconvBenchJob {
            workload_name: "image_batch_4k_png_to_qoi",
            tool_label: "fastconv (cpu qoi)",
            source_dir: &fixtures.images_dir,
            source_format: SourceBatchFormat::Png,
            input_bytes: fixtures.image_batch_bytes,
            out_dir: &create_h2h_dir(h2h_tmp, "fastconv_png_to_qoi")?,
            target_format: ImageOutputFormat::Qoi,
            quality: None,
            backend: "cpu",
            policy: LOSSLESS_EXACT_POLICY,
        },
    )?;
    let png_to_qoi_ffmpeg = run_ffmpeg_lossless_batch(
        "image_batch_4k_png_to_qoi",
        "ffmpeg (qoi)",
        &fixtures.images_dir,
        SourceBatchFormat::Png,
        fixtures.image_batch_bytes,
        &create_h2h_dir(h2h_tmp, "ffmpeg_png_to_qoi")?,
        ImageOutputFormat::Qoi,
        &["-c:v", "qoi"],
    )?;

    let webp_to_bmp_fastconv = run_fastconv_image_batch(
        fastconv_bin,
        &FastconvBenchJob {
            workload_name: "image_batch_4k_webp_to_bmp",
            tool_label: "fastconv (cpu bmp)",
            source_dir: &fixtures.webp_images_dir,
            source_format: SourceBatchFormat::Webp,
            input_bytes: fixtures.webp_batch_bytes,
            out_dir: &create_h2h_dir(h2h_tmp, "fastconv_webp_to_bmp")?,
            target_format: ImageOutputFormat::Bmp,
            quality: None,
            backend: "cpu",
            policy: LOSSLESS_EXACT_POLICY,
        },
    )?;
    let webp_to_bmp_ffmpeg = run_ffmpeg_lossless_batch(
        "image_batch_4k_webp_to_bmp",
        "ffmpeg (bmp)",
        &fixtures.webp_images_dir,
        SourceBatchFormat::WebpViaFfmpeg,
        fixtures.webp_batch_bytes,
        &create_h2h_dir(h2h_tmp, "ffmpeg_webp_to_bmp")?,
        ImageOutputFormat::Bmp,
        &["-c:v", "bmp"],
    )?;

    Ok(LosslessRuns {
        webp_to_png_fastconv,
        webp_to_png_ffmpeg,
        png_to_qoi_fastconv,
        png_to_qoi_ffmpeg,
        webp_to_bmp_fastconv,
        webp_to_bmp_ffmpeg,
    })
}

fn run_nvenc_heic_contenders(
    fastconv_bin: &Path,
    fixtures: &VerifiedHeadToHeadFixtures,
    h2h_tmp: &Path,
    has_nvenc: bool,
) -> Result<NvencHeicRuns, String> {
    if !has_nvenc {
        println!(
            "==> skipping fastconv (gpu:nvenc heic): NVENC hardware or link libraries unavailable"
        );
        return Ok(NvencHeicRuns {
            png: None,
            jpeg: None,
            webp: None,
        });
    }

    let png_heic = run_fastconv_image_batch(
        fastconv_bin,
        &FastconvBenchJob {
            workload_name: "image_batch_4k_png_to_heic",
            tool_label: "fastconv (gpu:nvenc q=80)",
            source_dir: &fixtures.images_dir,
            source_format: SourceBatchFormat::Png,
            input_bytes: fixtures.image_batch_bytes,
            out_dir: &create_h2h_dir(h2h_tmp, "fastconv_png_to_heic")?,
            target_format: ImageOutputFormat::Heic,
            quality: Some("80"),
            backend: "nvenc",
            policy: PNG_TO_LOSSY_POLICY,
        },
    )?;
    let jpg_heic = run_fastconv_image_batch(
        fastconv_bin,
        &FastconvBenchJob {
            workload_name: "image_batch_4k_jpeg_to_heic",
            tool_label: "fastconv (gpu:nvenc q=80)",
            source_dir: &fixtures.jpeg_images_dir,
            source_format: SourceBatchFormat::Jpeg,
            input_bytes: fixtures.jpeg_batch_bytes,
            out_dir: &create_h2h_dir(h2h_tmp, "fastconv_jpeg_to_heic")?,
            target_format: ImageOutputFormat::Heic,
            quality: Some("80"),
            backend: "nvenc",
            policy: COMPRESSED_TO_LOSSY_POLICY,
        },
    )?;
    let webp_heic = run_fastconv_image_batch(
        fastconv_bin,
        &FastconvBenchJob {
            workload_name: "image_batch_4k_webp_to_heic",
            tool_label: "fastconv (gpu:nvenc q=80)",
            source_dir: &fixtures.webp_images_dir,
            source_format: SourceBatchFormat::Webp,
            input_bytes: fixtures.webp_batch_bytes,
            out_dir: &create_h2h_dir(h2h_tmp, "fastconv_webp_to_heic")?,
            target_format: ImageOutputFormat::Heic,
            quality: Some("80"),
            backend: "nvenc",
            policy: COMPRESSED_TO_LOSSY_POLICY,
        },
    )?;
    Ok(NvencHeicRuns {
        png: Some(png_heic),
        jpeg: Some(jpg_heic),
        webp: Some(webp_heic),
    })
}

fn build_head_to_head_report(
    ffmpeg_version: String,
    hardware: HostHardware,
    lossy: LossyRuns,
    lossless: LosslessRuns,
    nvenc: NvencHeicRuns,
) -> HeadToHeadReport {
    let comparisons = vec![
        compare_entries(
            "image_batch_4k_png_to_webp",
            &lossy.png_to_webp_fastconv,
            &lossy.png_to_webp_ffmpeg,
        ),
        compare_entries(
            "image_batch_4k_png_to_jpeg",
            &lossy.png_to_jpeg_fastconv,
            &lossy.png_to_jpeg_ffmpeg,
        ),
        compare_entries(
            "image_batch_4k_jpeg_to_webp",
            &lossy.jpeg_to_webp_fastconv,
            &lossy.jpeg_to_webp_ffmpeg,
        ),
        compare_entries(
            "image_batch_4k_webp_to_png",
            &lossless.webp_to_png_fastconv,
            &lossless.webp_to_png_ffmpeg,
        ),
        compare_entries(
            "image_batch_4k_png_to_qoi",
            &lossless.png_to_qoi_fastconv,
            &lossless.png_to_qoi_ffmpeg,
        ),
        compare_entries(
            "image_batch_4k_webp_to_bmp",
            &lossless.webp_to_bmp_fastconv,
            &lossless.webp_to_bmp_ffmpeg,
        ),
    ];

    HeadToHeadReport {
        schema_version: 4,
        ffmpeg_version,
        hardware,
        png_to_webp_fastconv: lossy.png_to_webp_fastconv,
        png_to_webp_ffmpeg: lossy.png_to_webp_ffmpeg,
        png_to_jpeg_fastconv: lossy.png_to_jpeg_fastconv,
        png_to_jpeg_ffmpeg: lossy.png_to_jpeg_ffmpeg,
        jpeg_to_webp_fastconv: lossy.jpeg_to_webp_fastconv,
        jpeg_to_webp_ffmpeg: lossy.jpeg_to_webp_ffmpeg,
        webp_to_png_fastconv: lossless.webp_to_png_fastconv,
        webp_to_png_ffmpeg: lossless.webp_to_png_ffmpeg,
        png_to_qoi_fastconv: lossless.png_to_qoi_fastconv,
        png_to_qoi_ffmpeg: lossless.png_to_qoi_ffmpeg,
        webp_to_bmp_fastconv: lossless.webp_to_bmp_fastconv,
        webp_to_bmp_ffmpeg: lossless.webp_to_bmp_ffmpeg,
        png_to_heic_nvenc: nvenc.png,
        jpeg_to_heic_nvenc: nvenc.jpeg,
        webp_to_heic_nvenc: nvenc.webp,
        comparisons,
    }
}

fn compare_entries(
    workload_name: &str,
    fastconv: &HeadToHeadEntry,
    ffmpeg: &HeadToHeadEntry,
) -> HeadToHeadComparison {
    let both_match_reference_rgb =
        fastconv.validation.exact_pixel_match && ffmpeg.validation.exact_pixel_match;
    let fastconv_psnr_delta_db = match (
        fastconv.validation.mean_psnr_db,
        ffmpeg.validation.mean_psnr_db,
    ) {
        (Some(fc), Some(ff)) => Some(fc - ff),
        _ => None,
    };
    HeadToHeadComparison {
        workload_name: workload_name.to_owned(),
        fastconv_speedup_vs_ffmpeg: ffmpeg.metrics.duration_ms() / fastconv.metrics.duration_ms(),
        both_match_reference_rgb,
        fastconv_psnr_delta_db,
    }
}

fn format_validation_quality(validation: &BatchValidationSummary) -> String {
    if validation.exact_pixel_match {
        "exact".to_owned()
    } else if let Some(psnr) = validation.mean_psnr_db {
        format!("{psnr:.2} dB")
    } else {
        "n/a".to_owned()
    }
}

fn format_head_to_head_summary(report: &HeadToHeadReport) -> String {
    let mut lines = Vec::new();
    for cmp in &report.comparisons {
        if cmp.both_match_reference_rgb {
            lines.push(format!(
                "{:<28}: {:.2}x vs ffmpeg | both lossless (exact RGB vs reference)",
                cmp.workload_name, cmp.fastconv_speedup_vs_ffmpeg
            ));
        } else if let Some(delta) = cmp.fastconv_psnr_delta_db {
            lines.push(format!(
                "{:<28}: {:.2}x vs ffmpeg | delta mean PSNR = {:+.2} dB",
                cmp.workload_name, cmp.fastconv_speedup_vs_ffmpeg, delta
            ));
        } else {
            lines.push(format!(
                "{:<28}: {:.2}x vs ffmpeg",
                cmp.workload_name, cmp.fastconv_speedup_vs_ffmpeg
            ));
        }
    }
    if let Some(ref heic) = report.png_to_heic_nvenc {
        let vs_ff_webp =
            report.png_to_webp_ffmpeg.metrics.duration_ms() / heic.metrics.duration_ms();
        let vs_cpu_webp =
            report.png_to_webp_fastconv.metrics.duration_ms() / heic.metrics.duration_ms();
        lines.push(format!(
            "png_to_heic gpu:nvenc       : {vs_ff_webp:.2}x vs ffmpeg(webp) | {vs_cpu_webp:.2}x vs cpu(webp) | quality = {} | validated = {}/{} ({})",
            format_validation_quality(&heic.validation),
            heic.validation.files_validated,
            IMAGE_BATCH_COUNT,
            heic.validation.resolution
        ));
    }
    if let Some(ref heic) = report.jpeg_to_heic_nvenc {
        let vs_ff_webp =
            report.jpeg_to_webp_ffmpeg.metrics.duration_ms() / heic.metrics.duration_ms();
        let vs_cpu_webp =
            report.jpeg_to_webp_fastconv.metrics.duration_ms() / heic.metrics.duration_ms();
        lines.push(format!(
            "jpeg_to_heic gpu:nvenc      : {vs_ff_webp:.2}x vs ffmpeg(webp) | {vs_cpu_webp:.2}x vs cpu(webp) | quality = {} | validated = {}/{} ({})",
            format_validation_quality(&heic.validation),
            heic.validation.files_validated,
            IMAGE_BATCH_COUNT,
            heic.validation.resolution
        ));
    }
    if let Some(ref heic) = report.webp_to_heic_nvenc {
        lines.push(format!(
            "webp_to_heic gpu:nvenc      : {:.1} ms ({}) | quality = {} | validated = {}/{} ({})",
            heic.metrics.duration_ms(),
            heic.metrics.throughput(),
            format_validation_quality(&heic.validation),
            heic.validation.files_validated,
            IMAGE_BATCH_COUNT,
            heic.validation.resolution
        ));
    }
    if lines.is_empty() {
        String::new()
    } else {
        format!("{}\n", lines.join("\n"))
    }
}

fn print_head_to_head_summary(report: &HeadToHeadReport) {
    print!("{}", format_head_to_head_summary(report));
}

/// Runs the full `FFmpeg` reference baseline across image, audio, and video workloads.
///
/// # Errors
///
/// Returns an error if fixtures are missing or any `FFmpeg` invocation fails.
pub fn run_baseline(root: &Path) -> Result<(), String> {
    let hardware = fastconv_probe::detect_host_hardware()
        .map_err(|err| format!("failed to probe host hardware: {err}"))?;
    print_hardware_summary(&hardware);

    let fixtures = verify_baseline_fixtures(root)?;
    let ffmpeg_version = read_ffmpeg_version()?;
    println!("ffmpeg        : {ffmpeg_version}");

    let bench_dir = root.join("bench-output");
    let tmp_dir = bench_dir.join("tmp");
    if tmp_dir.exists() {
        fs::remove_dir_all(&tmp_dir)
            .map_err(|err| format!("failed to clean '{}': {err}", tmp_dir.display()))?;
    }
    let tmp_images = tmp_dir.join("images_4k");
    fs::create_dir_all(&tmp_images)
        .map_err(|err| format!("failed to create '{}': {err}", tmp_images.display()))?;

    let mut results = vec![
        bench_image_batch_webp(
            &fixtures.images_dir,
            fixtures.image_batch_bytes,
            &tmp_images,
        )?,
        bench_audio_transcode(
            "audio_60s_wav_to_flac",
            "ffmpeg (flac -5)",
            &fixtures.audio_wav,
            fixtures.audio_bytes,
            &tmp_dir.join("audio_60s_stereo.flac"),
            &["-c:a", "flac", "-compression_level", "5"],
        )?,
        bench_audio_transcode(
            "audio_60s_wav_to_opus",
            "ffmpeg (libopus 128k)",
            &fixtures.audio_wav,
            fixtures.audio_bytes,
            &tmp_dir.join("audio_60s_stereo.opus"),
            &["-c:a", "libopus", "-b:a", "128k"],
        )?,
        bench_video_transcode(
            "video_1080p60_h264_cpu",
            "ffmpeg (libx264 medium)",
            &fixtures.video_mp4,
            fixtures.video_bytes,
            &tmp_dir.join("video_1080p60_x264.mp4"),
            &[
                "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-c:a", "copy",
            ],
        )?,
    ];

    if hardware.supports_video_encode(GpuBackend::Nvenc) {
        results.push(bench_video_transcode(
            "video_1080p60_h264_nvenc",
            "ffmpeg (h264_nvenc p4)",
            &fixtures.video_mp4,
            fixtures.video_bytes,
            &tmp_dir.join("video_1080p60_nvenc.mp4"),
            &[
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-cq",
                "23",
                "-c:a",
                "copy",
            ],
        )?);
    }

    if tmp_dir.exists() {
        fs::remove_dir_all(&tmp_dir).map_err(|err| {
            format!(
                "failed to remove temporary outputs '{}': {err}",
                tmp_dir.display()
            )
        })?;
    }

    println!();
    print!("{}", format_baseline_table(&results));

    let json_path = bench_dir.join("ffmpeg-baseline.json");
    let report = BaselineReport {
        schema_version: 1,
        ffmpeg_version,
        hardware,
        results,
    };
    let json = serde_json::to_string_pretty(&report)
        .map_err(|err| format!("failed to serialize baseline report: {err}"))?;
    fs::write(&json_path, format!("{json}\n"))
        .map_err(|err| format!("failed to write '{}': {err}", json_path.display()))?;

    println!();
    println!("baseline saved: {}", json_path.display());
    Ok(())
}

fn build_release_fastconv(root: &Path, enable_nvenc: bool) -> Result<PathBuf, String> {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    let mut args = vec!["build", "--release", "-p", "fastconv-cli"];
    if enable_nvenc {
        args.extend_from_slice(&["--features", "nvenc"]);
        println!("==> building fastconv --release --features nvenc");
    } else {
        println!("==> building fastconv --release");
    }
    let status = Command::new(cargo)
        .current_dir(root)
        .args(&args)
        .status()
        .map_err(|err| format!("failed to launch cargo build --release: {err}"))?;
    if !status.success() {
        return Err(format!("`cargo {}` failed with {status}", args.join(" ")));
    }

    let bin_name = format!("fastconv{}", env::consts::EXE_SUFFIX);
    let bin_path = root.join("target/release").join(bin_name);
    if !bin_path.is_file() {
        return Err(format!(
            "expected release binary not found at '{}'",
            bin_path.display()
        ));
    }
    Ok(bin_path)
}

fn run_fastconv_image_batch(
    bin_path: &Path,
    job: &FastconvBenchJob<'_>,
) -> Result<HeadToHeadEntry, String> {
    println!("==> running {} [{}]...", job.workload_name, job.tool_label);
    let mut args = vec![
        OsStr::new("convert"),
        job.source_dir.as_os_str(),
        job.out_dir.as_os_str(),
        OsStr::new("--to"),
        OsStr::new(bench_target_cli_arg(job.target_format)),
        OsStr::new("--backend"),
        OsStr::new(job.backend),
    ];
    if let Some(q) = job.quality {
        args.push(OsStr::new("--quality"));
        args.push(OsStr::new(q));
    }
    let (duration, peak_memory) = run_timed_command(bin_path.as_os_str(), &args)?;
    let validation = validate_target_batch(
        job.target_format,
        job.source_dir,
        job.source_format,
        job.out_dir,
        IMAGE_BATCH_COUNT,
        job.policy,
    )?;

    let metrics = BenchmarkMetrics::new(
        job.workload_name,
        MediaKind::Image,
        job.tool_label,
        BenchmarkSample {
            duration,
            input_bytes: job.input_bytes,
            output_bytes: validation.total_output_bytes,
            unit: WorkloadUnit::Bytes,
            peak_memory,
        },
    )
    .map_err(|err| err.to_string())?;

    Ok(HeadToHeadEntry {
        metrics,
        validation,
    })
}

fn validate_target_batch(
    target_format: ImageOutputFormat,
    source_dir: &Path,
    source_format: SourceBatchFormat,
    out_dir: &Path,
    count: usize,
    policy: ValidationPolicy,
) -> Result<BatchValidationSummary, String> {
    match target_format {
        ImageOutputFormat::Webp => {
            validate_webp_batch(source_dir, source_format, out_dir, count, policy)
        }
        ImageOutputFormat::Jpeg => {
            validate_jpeg_batch(source_dir, source_format, out_dir, count, policy)
        }
        ImageOutputFormat::Heic => {
            validate_heic_batch(source_dir, source_format, out_dir, count, policy)
        }
        ImageOutputFormat::Png => {
            validate_png_batch(source_dir, source_format, out_dir, count, policy)
        }
        ImageOutputFormat::Bmp => {
            validate_bmp_batch(source_dir, source_format, out_dir, count, policy)
        }
        ImageOutputFormat::Qoi => {
            validate_qoi_batch(source_dir, source_format, out_dir, count, policy)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_ffmpeg_lossless_batch(
    workload_name: &str,
    tool_label: &str,
    source_dir: &Path,
    source_format: SourceBatchFormat,
    input_bytes: NonZeroU64,
    out_dir: &Path,
    target_format: ImageOutputFormat,
    codec_args: &[&str],
) -> Result<HeadToHeadEntry, String> {
    let input_pattern = source_dir.join(format!("img_%02d.{}", source_format.extension()));
    let output_pattern = out_dir.join(format!("img_%02d.{}", target_format.extension()));
    let in_str = path_to_str(&input_pattern)?;
    let out_str = path_to_str(&output_pattern)?;

    println!("==> running {workload_name} [{tool_label}]...");
    let mut ff_args = vec!["-start_number", "1", "-i", in_str, "-f", "image2"];
    ff_args.extend_from_slice(codec_args);
    ff_args.extend_from_slice(&["-start_number", "1", out_str]);
    let (duration, peak_memory) = run_timed_ffmpeg(&ff_args)?;

    let validation = validate_target_batch(
        target_format,
        source_dir,
        source_format,
        out_dir,
        IMAGE_BATCH_COUNT,
        LOSSLESS_EXACT_POLICY,
    )?;

    let metrics = BenchmarkMetrics::new(
        workload_name,
        MediaKind::Image,
        tool_label,
        BenchmarkSample {
            duration,
            input_bytes,
            output_bytes: validation.total_output_bytes,
            unit: WorkloadUnit::Bytes,
            peak_memory,
        },
    )
    .map_err(|err| err.to_string())?;

    Ok(HeadToHeadEntry {
        metrics,
        validation,
    })
}

fn run_ffmpeg_webp_batch(
    workload_name: &str,
    source_dir: &Path,
    source_format: SourceBatchFormat,
    input_bytes: NonZeroU64,
    out_dir: &Path,
    policy: ValidationPolicy,
) -> Result<HeadToHeadEntry, String> {
    let input_pattern = source_dir.join(format!("img_%02d.{}", source_format.extension()));
    let output_pattern = out_dir.join("img_%02d.webp");
    let in_str = path_to_str(&input_pattern)?;
    let out_str = path_to_str(&output_pattern)?;

    println!("==> running {workload_name} [ffmpeg (libwebp q=80)]...");
    let (duration, peak_memory) = run_timed_ffmpeg(&[
        "-start_number",
        "1",
        "-i",
        in_str,
        "-f",
        "image2",
        "-c:v",
        "libwebp",
        "-quality",
        "80",
        "-start_number",
        "1",
        out_str,
    ])?;

    let validation = validate_webp_batch(
        source_dir,
        source_format,
        out_dir,
        IMAGE_BATCH_COUNT,
        policy,
    )?;

    let metrics = BenchmarkMetrics::new(
        workload_name,
        MediaKind::Image,
        "ffmpeg (libwebp q=80)",
        BenchmarkSample {
            duration,
            input_bytes,
            output_bytes: validation.total_output_bytes,
            unit: WorkloadUnit::Bytes,
            peak_memory,
        },
    )
    .map_err(|err| err.to_string())?;

    Ok(HeadToHeadEntry {
        metrics,
        validation,
    })
}

fn run_ffmpeg_jpeg_batch(
    workload_name: &str,
    source_dir: &Path,
    source_format: SourceBatchFormat,
    input_bytes: NonZeroU64,
    out_dir: &Path,
    policy: ValidationPolicy,
) -> Result<HeadToHeadEntry, String> {
    let input_pattern = source_dir.join(format!("img_%02d.{}", source_format.extension()));
    let output_pattern = out_dir.join("img_%02d.jpg");
    let in_str = path_to_str(&input_pattern)?;
    let out_str = path_to_str(&output_pattern)?;

    println!("==> running {workload_name} [ffmpeg (mjpeg qv=3)]...");
    let (duration, peak_memory) = run_timed_ffmpeg(&[
        "-start_number",
        "1",
        "-i",
        in_str,
        "-f",
        "image2",
        "-c:v",
        "mjpeg",
        "-pix_fmt",
        "yuvj420p",
        "-q:v",
        "3",
        "-start_number",
        "1",
        out_str,
    ])?;

    let validation = validate_jpeg_batch(
        source_dir,
        source_format,
        out_dir,
        IMAGE_BATCH_COUNT,
        policy,
    )?;

    let metrics = BenchmarkMetrics::new(
        workload_name,
        MediaKind::Image,
        "ffmpeg (mjpeg qv=3)",
        BenchmarkSample {
            duration,
            input_bytes,
            output_bytes: validation.total_output_bytes,
            unit: WorkloadUnit::Bytes,
            peak_memory,
        },
    )
    .map_err(|err| err.to_string())?;

    Ok(HeadToHeadEntry {
        metrics,
        validation,
    })
}

struct ValidatedImageResult {
    bytes: u64,
    width: u32,
    height: u32,
    exact_pixel_match: bool,
    psnr_db: Option<f64>,
}

fn ensure_exact_dir_entry_count(dir: &Path, expected: usize) -> Result<(), String> {
    let mut actual = 0_usize;
    for entry_result in fs::read_dir(dir)
        .map_err(|err| format!("failed to read output dir '{}': {err}", dir.display()))?
    {
        entry_result
            .map_err(|err| format!("failed to inspect output dir '{}': {err}", dir.display()))?;
        actual += 1;
    }
    if actual != expected {
        return Err(format!(
            "expected {expected} entries in '{}', found {actual}",
            dir.display()
        ));
    }
    Ok(())
}

fn validate_image_batch<F>(
    source_dir: &Path,
    source_format: SourceBatchFormat,
    out_dir: &Path,
    out_ext: &str,
    label: &str,
    count: usize,
    validate_pair: F,
) -> Result<BatchValidationSummary, String>
where
    F: Fn(&Path, SourceBatchFormat, &Path) -> Result<ValidatedImageResult, String> + Sync,
{
    println!(
        "==> validating {count} generated {label} outputs in {}...",
        out_dir.display()
    );
    if count == 0 {
        return Err(format!("cannot validate empty {label} batch"));
    }
    ensure_exact_dir_entry_count(out_dir, count)?;

    let workers = thread::available_parallelism()
        .map_or(4, std::num::NonZeroUsize::get)
        .min(count);
    let next_index = AtomicUsize::new(1);

    let per_worker: Vec<Result<Vec<ValidatedImageResult>, String>> = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            handles.push(
                scope.spawn(|| -> Result<Vec<ValidatedImageResult>, String> {
                    let mut items = Vec::new();
                    loop {
                        let index = next_index.fetch_add(1, Ordering::Relaxed);
                        if index > count {
                            break;
                        }
                        let src_path = source_dir
                            .join(format!("img_{index:02}.{}", source_format.extension()));
                        let out_path = out_dir.join(format!("img_{index:02}.{out_ext}"));
                        items.push(validate_pair(&src_path, source_format, &out_path)?);
                    }
                    Ok(items)
                }),
            );
        }

        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|_| Err(format!("{label} validation worker thread panicked")))
            })
            .collect()
    });

    summarize_validated_batch(per_worker, count, label)
}

fn validate_webp_batch(
    source_dir: &Path,
    source_format: SourceBatchFormat,
    webp_dir: &Path,
    count: usize,
    policy: ValidationPolicy,
) -> Result<BatchValidationSummary, String> {
    validate_image_batch(
        source_dir,
        source_format,
        webp_dir,
        "webp",
        "WebP",
        count,
        |src, src_fmt, out| validate_single_webp_pair(src, src_fmt, out, policy),
    )
}

fn validate_jpeg_batch(
    source_dir: &Path,
    source_format: SourceBatchFormat,
    jpeg_dir: &Path,
    count: usize,
    policy: ValidationPolicy,
) -> Result<BatchValidationSummary, String> {
    validate_image_batch(
        source_dir,
        source_format,
        jpeg_dir,
        "jpg",
        "JPEG",
        count,
        |src, src_fmt, out| validate_single_jpeg_pair(src, src_fmt, out, policy),
    )
}

fn validate_heic_batch(
    source_dir: &Path,
    source_format: SourceBatchFormat,
    heic_dir: &Path,
    count: usize,
    policy: ValidationPolicy,
) -> Result<BatchValidationSummary, String> {
    validate_image_batch(
        source_dir,
        source_format,
        heic_dir,
        "heic",
        "HEIC",
        count,
        |src, src_fmt, out| validate_single_heic_pair(src, src_fmt, out, policy),
    )
}

fn validate_png_batch(
    source_dir: &Path,
    source_format: SourceBatchFormat,
    png_dir: &Path,
    count: usize,
    policy: ValidationPolicy,
) -> Result<BatchValidationSummary, String> {
    validate_image_batch(
        source_dir,
        source_format,
        png_dir,
        "png",
        "PNG",
        count,
        |src, src_fmt, out| validate_single_png_pair(src, src_fmt, out, policy),
    )
}

fn validate_qoi_batch(
    source_dir: &Path,
    source_format: SourceBatchFormat,
    qoi_dir: &Path,
    count: usize,
    policy: ValidationPolicy,
) -> Result<BatchValidationSummary, String> {
    validate_image_batch(
        source_dir,
        source_format,
        qoi_dir,
        "qoi",
        "QOI",
        count,
        |src, src_fmt, out| validate_single_qoi_pair(src, src_fmt, out, policy),
    )
}

fn validate_bmp_batch(
    source_dir: &Path,
    source_format: SourceBatchFormat,
    bmp_dir: &Path,
    count: usize,
    policy: ValidationPolicy,
) -> Result<BatchValidationSummary, String> {
    validate_image_batch(
        source_dir,
        source_format,
        bmp_dir,
        "bmp",
        "BMP",
        count,
        |src, src_fmt, out| validate_single_bmp_pair(src, src_fmt, out, policy),
    )
}

fn check_output_size(
    src_path: &Path,
    src_label: &str,
    out_path: &Path,
    out_label: &str,
    policy: ValidationPolicy,
) -> Result<u64, String> {
    let src_bytes = file_size_nonzero(src_path)?.get();
    let out_bytes = file_size_nonzero(out_path)?.get();
    if out_bytes < policy.min_output_bytes {
        return Err(format!(
            "implausibly small {out_label} output '{}' ({out_bytes} bytes < {} bytes)",
            out_path.display(),
            policy.min_output_bytes
        ));
    }
    if policy.require_smaller_than_source && out_bytes >= src_bytes {
        return Err(format!(
            "implausibly large {out_label} output '{}' ({out_bytes} bytes >= source {src_label} {src_bytes} bytes)",
            out_path.display()
        ));
    }
    Ok(out_bytes)
}

fn check_dimensions_and_quality(
    src: &DecodedImageRef<'_>,
    out: &DecodedImageRef<'_>,
    out_path: &Path,
    policy: ValidationPolicy,
) -> Result<(bool, Option<f64>), String> {
    if (out.width, out.height) != (src.width, src.height) {
        return Err(format!(
            "resolution mismatch for '{}': source {} is {}x{}, decoded {} is {}x{}",
            out_path.display(),
            src.label,
            src.width,
            src.height,
            out.label,
            out.width,
            out.height
        ));
    }

    match policy.mode {
        QualityValidationMode::LosslessExactPixels => {
            if src.rgb.len() != out.rgb.len() {
                return Err(format!(
                    "pixel buffer length mismatch for '{}': source {} bytes vs output {} bytes",
                    out_path.display(),
                    src.rgb.len(),
                    out.rgb.len()
                ));
            }
            if let Some(pos) = src.rgb.iter().zip(out.rgb.iter()).position(|(a, b)| a != b) {
                return Err(format!(
                    "lossless pixel mismatch in '{}' at byte offset {pos}: expected {}, got {}",
                    out_path.display(),
                    src.rgb[pos],
                    out.rgb[pos]
                ));
            }
            Ok((true, None))
        }
        QualityValidationMode::LossyMinPsnrDb(min_psnr_db) => {
            let psnr = compute_rgb_psnr_db(src.rgb, out.rgb)?;
            if psnr.is_nan() || psnr < min_psnr_db {
                return Err(format!(
                    "implausibly low visual quality for '{}' (PSNR {psnr:.2} dB < {min_psnr_db:.2} dB)",
                    out_path.display()
                ));
            }
            if psnr.is_infinite() {
                Ok((true, None))
            } else {
                Ok((false, Some(psnr)))
            }
        }
    }
}

fn validate_single_webp_pair(
    src_path: &Path,
    src_format: SourceBatchFormat,
    webp_path: &Path,
    policy: ValidationPolicy,
) -> Result<ValidatedImageResult, String> {
    let out_bytes = check_output_size(src_path, src_format.label(), webp_path, "WebP", policy)?;
    let (src_rgb, src_w, src_h) = src_format.decode_to_rgb8(src_path)?;
    let webp_data = fs::read(webp_path)
        .map_err(|err| format!("failed to read WebP '{}': {err}", webp_path.display()))?;
    let (webp_rgb, webp_w, webp_h) = zenwebp::oneshot::decode_rgb(&webp_data)
        .map_err(|err| format!("invalid WebP output '{}': {err}", webp_path.display()))?;

    let (exact_pixel_match, psnr_db) = check_dimensions_and_quality(
        &DecodedImageRef {
            rgb: &src_rgb,
            width: src_w,
            height: src_h,
            label: src_format.label(),
        },
        &DecodedImageRef {
            rgb: &webp_rgb,
            width: webp_w,
            height: webp_h,
            label: "WebP",
        },
        webp_path,
        policy,
    )?;

    Ok(ValidatedImageResult {
        bytes: out_bytes,
        width: webp_w,
        height: webp_h,
        exact_pixel_match,
        psnr_db,
    })
}

fn validate_single_jpeg_pair(
    src_path: &Path,
    src_format: SourceBatchFormat,
    jpeg_path: &Path,
    policy: ValidationPolicy,
) -> Result<ValidatedImageResult, String> {
    let out_bytes = check_output_size(src_path, src_format.label(), jpeg_path, "JPEG", policy)?;
    let jpeg_data = fs::read(jpeg_path)
        .map_err(|err| format!("failed to read JPEG '{}': {err}", jpeg_path.display()))?;
    if !jpeg_data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Err(format!(
            "invalid JPEG magic bytes in '{}'",
            jpeg_path.display()
        ));
    }
    let (src_rgb, src_w, src_h) = src_format.decode_to_rgb8(src_path)?;
    let (jpeg_rgb, jpeg_w, jpeg_h) = decode_jpeg_bytes_to_rgb8(&jpeg_data, jpeg_path)?;

    let (exact_pixel_match, psnr_db) = check_dimensions_and_quality(
        &DecodedImageRef {
            rgb: &src_rgb,
            width: src_w,
            height: src_h,
            label: src_format.label(),
        },
        &DecodedImageRef {
            rgb: &jpeg_rgb,
            width: jpeg_w,
            height: jpeg_h,
            label: "JPEG",
        },
        jpeg_path,
        policy,
    )?;

    Ok(ValidatedImageResult {
        bytes: out_bytes,
        width: jpeg_w,
        height: jpeg_h,
        exact_pixel_match,
        psnr_db,
    })
}

fn validate_single_png_pair(
    src_path: &Path,
    src_format: SourceBatchFormat,
    png_path: &Path,
    policy: ValidationPolicy,
) -> Result<ValidatedImageResult, String> {
    let out_bytes = check_output_size(src_path, src_format.label(), png_path, "PNG", policy)?;
    let probed = fastconv_probe::probe_image(png_path)
        .map_err(|err| format!("invalid PNG output '{}': {err}", png_path.display()))?;
    if probed.format != ImageInputFormat::Png {
        return Err(format!(
            "expected PNG magic bytes in '{}', detected {}",
            png_path.display(),
            probed.format
        ));
    }
    let (src_rgb, src_w, src_h) = src_format.decode_to_rgb8(src_path)?;
    let (png_rgb, png_w, png_h) = decode_png_to_rgb8(png_path)?;

    let (exact_pixel_match, psnr_db) = check_dimensions_and_quality(
        &DecodedImageRef {
            rgb: &src_rgb,
            width: src_w,
            height: src_h,
            label: src_format.label(),
        },
        &DecodedImageRef {
            rgb: &png_rgb,
            width: png_w,
            height: png_h,
            label: "PNG",
        },
        png_path,
        policy,
    )?;

    Ok(ValidatedImageResult {
        bytes: out_bytes,
        width: png_w,
        height: png_h,
        exact_pixel_match,
        psnr_db,
    })
}

fn validate_single_qoi_pair(
    src_path: &Path,
    src_format: SourceBatchFormat,
    qoi_path: &Path,
    policy: ValidationPolicy,
) -> Result<ValidatedImageResult, String> {
    let out_bytes = check_output_size(src_path, src_format.label(), qoi_path, "QOI", policy)?;
    let probed = fastconv_probe::probe_image(qoi_path)
        .map_err(|err| format!("invalid QOI output '{}': {err}", qoi_path.display()))?;
    if probed.format != ImageInputFormat::Qoi {
        return Err(format!(
            "expected QOI magic bytes in '{}', detected {}",
            qoi_path.display(),
            probed.format
        ));
    }
    let (src_rgb, src_w, src_h) = src_format.decode_to_rgb8(src_path)?;
    // Decode QOI independently via FFmpeg so we validate interoperability against a third-party decoder.
    let (qoi_rgb, qoi_w, qoi_h) = decode_image_to_rgb8_via_ffmpeg(qoi_path, "QOI")?;
    if (qoi_w, qoi_h)
        != (
            probed.dimensions.width().get(),
            probed.dimensions.height().get(),
        )
    {
        return Err(format!(
            "QOI header/decoder dimension mismatch for '{}': header {}, decoded {qoi_w}x{qoi_h}",
            qoi_path.display(),
            probed.dimensions
        ));
    }

    let (exact_pixel_match, psnr_db) = check_dimensions_and_quality(
        &DecodedImageRef {
            rgb: &src_rgb,
            width: src_w,
            height: src_h,
            label: src_format.label(),
        },
        &DecodedImageRef {
            rgb: &qoi_rgb,
            width: qoi_w,
            height: qoi_h,
            label: "QOI",
        },
        qoi_path,
        policy,
    )?;

    Ok(ValidatedImageResult {
        bytes: out_bytes,
        width: qoi_w,
        height: qoi_h,
        exact_pixel_match,
        psnr_db,
    })
}

fn validate_single_bmp_pair(
    src_path: &Path,
    src_format: SourceBatchFormat,
    bmp_path: &Path,
    policy: ValidationPolicy,
) -> Result<ValidatedImageResult, String> {
    let out_bytes = check_output_size(src_path, src_format.label(), bmp_path, "BMP", policy)?;
    let probed = fastconv_probe::probe_image(bmp_path)
        .map_err(|err| format!("invalid BMP output '{}': {err}", bmp_path.display()))?;
    if probed.format != ImageInputFormat::Bmp {
        return Err(format!(
            "expected BMP magic bytes in '{}', detected {}",
            bmp_path.display(),
            probed.format
        ));
    }
    let (src_rgb, src_w, src_h) = src_format.decode_to_rgb8(src_path)?;
    let (bmp_rgb, bmp_w, bmp_h) = decode_image_to_rgb8_via_ffmpeg(bmp_path, "BMP")?;
    if (bmp_w, bmp_h)
        != (
            probed.dimensions.width().get(),
            probed.dimensions.height().get(),
        )
    {
        return Err(format!(
            "BMP header/decoder dimension mismatch for '{}': header {}, decoded {bmp_w}x{bmp_h}",
            bmp_path.display(),
            probed.dimensions
        ));
    }

    let (exact_pixel_match, psnr_db) = check_dimensions_and_quality(
        &DecodedImageRef {
            rgb: &src_rgb,
            width: src_w,
            height: src_h,
            label: src_format.label(),
        },
        &DecodedImageRef {
            rgb: &bmp_rgb,
            width: bmp_w,
            height: bmp_h,
            label: "BMP",
        },
        bmp_path,
        policy,
    )?;

    Ok(ValidatedImageResult {
        bytes: out_bytes,
        width: bmp_w,
        height: bmp_h,
        exact_pixel_match,
        psnr_db,
    })
}

fn summarize_validated_batch(
    per_worker: Vec<Result<Vec<ValidatedImageResult>, String>>,
    count: usize,
    label: &str,
) -> Result<BatchValidationSummary, String> {
    let mut total_bytes = 0_u64;
    let mut expected_dims: Option<(u32, u32)> = None;
    let mut all_exact = true;
    let mut min_psnr = f64::INFINITY;
    let mut psnr_sum = 0_f64;
    let mut finite_psnr_count = 0_usize;
    let mut validated_count = 0_usize;

    for worker_result in per_worker {
        for item in worker_result? {
            if let Some((ew, eh)) = expected_dims {
                if (item.width, item.height) != (ew, eh) {
                    return Err(format!(
                        "inconsistent {label} batch resolution: expected {ew}x{eh}, got {}x{}",
                        item.width, item.height
                    ));
                }
            } else {
                expected_dims = Some((item.width, item.height));
            }
            total_bytes = total_bytes
                .checked_add(item.bytes)
                .ok_or_else(|| format!("validated {label} byte size overflow"))?;
            all_exact &= item.exact_pixel_match;
            if let Some(psnr) = item.psnr_db {
                min_psnr = min_psnr.min(psnr);
                psnr_sum += psnr;
                finite_psnr_count += 1;
            }
            validated_count += 1;
        }
    }

    if validated_count != count {
        return Err(format!(
            "expected {count} validated {label} files, verified {validated_count}"
        ));
    }

    let (width, height) = expected_dims.ok_or_else(|| format!("no {label} images validated"))?;
    let total_output_bytes = NonZeroU64::new(total_bytes)
        .ok_or_else(|| format!("validated {label} batch output size is zero"))?;

    let (min_psnr_db, mean_psnr_db) = if finite_psnr_count > 0 {
        let mean = psnr_sum / u64_to_f64(u64::try_from(finite_psnr_count).unwrap_or(1));
        (Some(min_psnr), Some(mean))
    } else {
        (None, None)
    };

    Ok(BatchValidationSummary {
        files_validated: validated_count,
        resolution: format!("{width}x{height}"),
        total_output_bytes,
        exact_pixel_match: all_exact,
        min_psnr_db,
        mean_psnr_db,
    })
}

fn validate_single_heic_pair(
    src_path: &Path,
    src_format: SourceBatchFormat,
    heic_path: &Path,
    policy: ValidationPolicy,
) -> Result<ValidatedImageResult, String> {
    let out_bytes = check_output_size(src_path, src_format.label(), heic_path, "HEIC", policy)?;

    let header = fs::read(heic_path)
        .map_err(|err| format!("failed to read HEIC '{}': {err}", heic_path.display()))?;
    if header.get(4..12) != Some(b"ftypheic".as_slice()) {
        return Err(format!(
            "invalid HEIC container header in '{}' (expected ftypheic)",
            heic_path.display()
        ));
    }

    let (src_rgb, src_w, src_h) = src_format.decode_to_rgb8(src_path)?;
    let (heic_w, heic_h) = probe_dimensions_via_ffprobe(heic_path, "HEIC")?;
    if (heic_w, heic_h) != (src_w, src_h) {
        return Err(format!(
            "resolution mismatch for '{}': source {} is {src_w}x{src_h}, probed HEIC is {heic_w}x{heic_h}",
            heic_path.display(),
            src_format.label()
        ));
    }

    let heic_rgb = decode_raw_rgb24_via_ffmpeg(heic_path, heic_w, heic_h, "HEIC")?;
    let (exact_pixel_match, psnr_db) = check_dimensions_and_quality(
        &DecodedImageRef {
            rgb: &src_rgb,
            width: src_w,
            height: src_h,
            label: src_format.label(),
        },
        &DecodedImageRef {
            rgb: &heic_rgb,
            width: heic_w,
            height: heic_h,
            label: "HEIC",
        },
        heic_path,
        policy,
    )?;

    Ok(ValidatedImageResult {
        bytes: out_bytes,
        width: heic_w,
        height: heic_h,
        exact_pixel_match,
        psnr_db,
    })
}

fn probe_dimensions_via_ffprobe(path: &Path, label: &str) -> Result<(u32, u32), String> {
    let output = Command::new("ffprobe")
        .stdin(Stdio::null())
        .args([
            OsStr::new("-v"),
            OsStr::new("error"),
            OsStr::new("-select_streams"),
            OsStr::new("v:0"),
            OsStr::new("-show_entries"),
            OsStr::new("stream=width,height"),
            OsStr::new("-of"),
            OsStr::new("csv=s=x:p=0"),
            path.as_os_str(),
        ])
        .output()
        .map_err(|err| {
            format!(
                "failed to run ffprobe on {label} '{}': {err}",
                path.display()
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "ffprobe rejected {label} '{}': {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text
        .lines()
        .next()
        .map(str::trim)
        .ok_or_else(|| format!("ffprobe returned empty output for '{}'", path.display()))?;
    let (w_str, h_str) = line.split_once('x').ok_or_else(|| {
        format!(
            "unexpected ffprobe dimensions '{line}' for '{}'",
            path.display()
        )
    })?;
    let w = w_str
        .parse::<u32>()
        .map_err(|err| format!("invalid {label} width '{w_str}': {err}"))?;
    let h = h_str
        .parse::<u32>()
        .map_err(|err| format!("invalid {label} height '{h_str}': {err}"))?;
    Ok((w, h))
}

fn decode_raw_rgb24_via_ffmpeg(
    path: &Path,
    width: u32,
    height: u32,
    label: &str,
) -> Result<Vec<u8>, String> {
    let output = Command::new("ffmpeg")
        .stdin(Stdio::null())
        .args([
            OsStr::new("-nostdin"),
            OsStr::new("-hide_banner"),
            OsStr::new("-loglevel"),
            OsStr::new("error"),
            OsStr::new("-i"),
            path.as_os_str(),
            OsStr::new("-frames:v"),
            OsStr::new("1"),
            OsStr::new("-f"),
            OsStr::new("rawvideo"),
            OsStr::new("-pix_fmt"),
            OsStr::new("rgb24"),
            OsStr::new("-"),
        ])
        .output()
        .map_err(|err| {
            format!(
                "failed to decode {label} '{}' via ffmpeg: {err}",
                path.display()
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg failed to decode {label} '{}': {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let expected_len = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|px| px.checked_mul(3))
        .ok_or_else(|| format!("decoded {label} RGB24 size overflow"))?;
    if output.stdout.len() != expected_len {
        return Err(format!(
            "decoded {label} '{}' produced {} bytes, expected {expected_len} ({width}x{height} rgb24)",
            path.display(),
            output.stdout.len()
        ));
    }
    Ok(output.stdout)
}

fn decode_image_to_rgb8_via_ffmpeg(
    path: &Path,
    label: &str,
) -> Result<(Vec<u8>, u32, u32), String> {
    let (w, h) = probe_dimensions_via_ffprobe(path, label)?;
    let rgb = decode_raw_rgb24_via_ffmpeg(path, w, h, label)?;
    Ok((rgb, w, h))
}

fn decode_webp_to_rgb8(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    let bytes =
        fs::read(path).map_err(|err| format!("failed to read WebP '{}': {err}", path.display()))?;
    let mut decoder = zenwebp::WebPDecoder::build(&bytes)
        .map_err(|err| format!("invalid WebP '{}': {err}", path.display()))?;
    let (w, h) = decoder.dimensions();
    let buf_len = decoder
        .output_buffer_size()
        .ok_or_else(|| format!("WebP '{}' exceeds addressable buffer size", path.display()))?;
    let mut raw = vec![0_u8; buf_len];
    decoder
        .read_image(&mut raw)
        .map_err(|err| format!("failed to decode WebP '{}': {err}", path.display()))?;
    if decoder.has_alpha() {
        let mut rgb = Vec::with_capacity((raw.len() / 4) * 3);
        let (quads, _) = raw.as_chunks::<4>();
        for &[r, g, b, _] in quads {
            rgb.extend_from_slice(&[r, g, b]);
        }
        Ok((rgb, w, h))
    } else {
        Ok((raw, w, h))
    }
}

fn decode_png_to_rgb8(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    let file = File::open(path)
        .map_err(|err| format!("failed to open PNG '{}': {err}", path.display()))?;
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    decoder.set_ignore_text_chunk(true);

    let mut reader = decoder
        .read_info()
        .map_err(|err| format!("failed to read PNG header '{}': {err}", path.display()))?;
    let buf_len = reader
        .output_buffer_size()
        .ok_or_else(|| format!("PNG '{}' exceeds addressable buffer size", path.display()))?;
    let mut raw = vec![0_u8; buf_len];
    let frame = reader
        .next_frame(&mut raw)
        .map_err(|err| format!("failed to decode PNG '{}': {err}", path.display()))?;
    if frame.bit_depth != png::BitDepth::Eight {
        return Err(format!(
            "unsupported decoded PNG bit depth {:?} in '{}'",
            frame.bit_depth,
            path.display()
        ));
    }
    let slice = raw
        .get(..frame.buffer_size())
        .ok_or_else(|| format!("PNG frame exceeded buffer for '{}'", path.display()))?;

    let rgb = match frame.color_type {
        png::ColorType::Rgb => slice.to_vec(),
        png::ColorType::Rgba => {
            let mut out = Vec::with_capacity((slice.len() / 4) * 3);
            let (chunks, remainder) = slice.as_chunks::<4>();
            if !remainder.is_empty() {
                return Err(format!(
                    "unaligned RGBA PNG buffer length {} in '{}'",
                    slice.len(),
                    path.display()
                ));
            }
            for chunk in chunks {
                out.extend_from_slice(&chunk[..3]);
            }
            out
        }
        png::ColorType::Grayscale => {
            let mut out = Vec::with_capacity(slice.len() * 3);
            for &g in slice {
                out.extend_from_slice(&[g, g, g]);
            }
            out
        }
        png::ColorType::GrayscaleAlpha => {
            let mut out = Vec::with_capacity((slice.len() / 2) * 3);
            let (chunks, remainder) = slice.as_chunks::<2>();
            if !remainder.is_empty() {
                return Err(format!(
                    "unaligned GrayscaleAlpha PNG buffer length {} in '{}'",
                    slice.len(),
                    path.display()
                ));
            }
            for chunk in chunks {
                let g = chunk[0];
                out.extend_from_slice(&[g, g, g]);
            }
            out
        }
        other @ png::ColorType::Indexed => {
            return Err(format!(
                "unsupported decoded PNG color type {other:?} in '{}'",
                path.display()
            ));
        }
    };

    Ok((rgb, frame.width, frame.height))
}

fn decode_jpeg_to_rgb8(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    let bytes =
        fs::read(path).map_err(|err| format!("failed to read JPEG '{}': {err}", path.display()))?;
    decode_jpeg_bytes_to_rgb8(&bytes, path)
}

fn decode_jpeg_bytes_to_rgb8(bytes: &[u8], path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    let options = DecoderOptions::default()
        .set_strict_mode(true)
        .jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    decoder
        .decode_headers()
        .map_err(|err| format!("invalid JPEG output '{}': {err}", path.display()))?;
    if !matches!(
        decoder.input_colorspace(),
        Some(
            ColorSpace::Luma
                | ColorSpace::YCbCr
                | ColorSpace::RGB
                | ColorSpace::CMYK
                | ColorSpace::YCCK
        )
    ) {
        return Err(format!(
            "unsupported JPEG color space {:?} in '{}'",
            decoder.input_colorspace(),
            path.display()
        ));
    }
    let rgb = decoder
        .decode()
        .map_err(|err| format!("invalid JPEG output '{}': {err}", path.display()))?;
    let info = decoder
        .info()
        .ok_or_else(|| format!("missing JPEG metadata in '{}'", path.display()))?;
    let width = u32::from(info.width);
    let height = u32::from(info.height);
    let expected_len = usize::from(info.width)
        .checked_mul(usize::from(info.height))
        .and_then(|px| px.checked_mul(3))
        .ok_or_else(|| format!("JPEG RGB buffer size overflow in '{}'", path.display()))?;
    if rgb.len() != expected_len {
        return Err(format!(
            "decoded JPEG '{}' produced {} bytes, expected {expected_len} ({width}x{height} rgb8)",
            path.display(),
            rgb.len()
        ));
    }
    Ok((rgb, width, height))
}

fn compute_rgb_psnr_db(src: &[u8], dst: &[u8]) -> Result<f64, String> {
    if src.is_empty() || src.len() != dst.len() {
        return Err(format!(
            "cannot compute PSNR for buffers of sizes {} and {}",
            src.len(),
            dst.len()
        ));
    }

    let mut sse = 0_u64;
    for (&a, &b) in src.iter().zip(dst.iter()) {
        let diff = u64::from(a.abs_diff(b));
        sse = sse
            .checked_add(diff * diff)
            .ok_or_else(|| "PSNR squared error overflow".to_owned())?;
    }

    if sse == 0 {
        return Ok(f64::INFINITY);
    }

    let len_u64 = u64::try_from(src.len()).map_err(|err| err.to_string())?;
    let mse = u64_to_f64(sse) / u64_to_f64(len_u64);
    Ok(10.0 * ((255.0_f64 * 255.0) / mse).log10())
}

struct VerifiedHeadToHeadFixtures {
    images_dir: PathBuf,
    image_batch_bytes: NonZeroU64,
    jpeg_images_dir: PathBuf,
    jpeg_batch_bytes: NonZeroU64,
    webp_images_dir: PathBuf,
    webp_batch_bytes: NonZeroU64,
}

struct VerifiedBaselineFixtures {
    images_dir: PathBuf,
    image_batch_bytes: NonZeroU64,
    audio_wav: PathBuf,
    audio_bytes: NonZeroU64,
    video_mp4: PathBuf,
    video_bytes: NonZeroU64,
}

fn verify_image_batch_with_dims(
    root: &Path,
    expected_format: ImageInputFormat,
    expected_width: u32,
    expected_height: u32,
) -> Result<NonZeroU64, String> {
    let mut total_bytes = 0_u64;
    let label = match expected_format {
        ImageInputFormat::Png => "PNG",
        ImageInputFormat::Jpeg => "JPEG",
        ImageInputFormat::Webp => "WebP",
        ImageInputFormat::Bmp => "BMP",
        ImageInputFormat::Qoi => "QOI",
    };

    for index in 1..=IMAGE_BATCH_COUNT {
        let path = match expected_format {
            ImageInputFormat::Png => fixtures::image_path(root, index),
            ImageInputFormat::Jpeg => fixtures::jpeg_image_path(root, index),
            ImageInputFormat::Webp => fixtures::webp_image_path(root, index),
            ImageInputFormat::Bmp | ImageInputFormat::Qoi => {
                return Err(format!("no 4K fixture batch configured for {label}"));
            }
        };
        let bytes = file_size_nonzero(&path)
            .map_err(|err| format!("{err}; run `cargo xtask gen-fixtures --force` first"))?;
        let probed = fastconv_probe::probe_image(&path).map_err(|err| {
            format!(
                "invalid {label} fixture '{}': {err}; remove it or run `cargo xtask gen-fixtures --force`",
                path.display()
            )
        })?;
        if probed.format != expected_format {
            return Err(format!(
                "fixture '{}' has format {}, expected {label}; remove it or run `cargo xtask gen-fixtures --force`",
                path.display(),
                probed.format
            ));
        }
        if probed.dimensions.width().get() != expected_width
            || probed.dimensions.height().get() != expected_height
        {
            return Err(format!(
                "fixture '{}' has resolution {}, expected {expected_width}x{expected_height}; run `cargo xtask gen-fixtures --force`",
                path.display(),
                probed.dimensions
            ));
        }
        total_bytes = total_bytes
            .checked_add(bytes.get())
            .ok_or_else(|| format!("{label} batch byte size overflow"))?;
    }

    NonZeroU64::new(total_bytes).ok_or_else(|| format!("{label} batch is empty"))
}

fn verify_4k_image_batch(
    root: &Path,
    expected_format: ImageInputFormat,
) -> Result<NonZeroU64, String> {
    verify_image_batch_with_dims(root, expected_format, IMAGE_WIDTH, IMAGE_HEIGHT)
}

fn verify_head_to_head_fixtures(root: &Path) -> Result<VerifiedHeadToHeadFixtures, String> {
    let image_batch_bytes = verify_4k_image_batch(root, ImageInputFormat::Png)?;
    let jpeg_batch_bytes = verify_4k_image_batch(root, ImageInputFormat::Jpeg)?;
    let webp_batch_bytes = verify_4k_image_batch(root, ImageInputFormat::Webp)?;

    Ok(VerifiedHeadToHeadFixtures {
        images_dir: fixtures::images_dir(root),
        image_batch_bytes,
        jpeg_images_dir: fixtures::jpeg_images_dir(root),
        jpeg_batch_bytes,
        webp_images_dir: fixtures::webp_images_dir(root),
        webp_batch_bytes,
    })
}

fn verify_baseline_fixtures(root: &Path) -> Result<VerifiedBaselineFixtures, String> {
    let image_batch_bytes = verify_4k_image_batch(root, ImageInputFormat::Png)?;

    let audio_wav = fixtures::audio_wav_path(root);
    let audio_bytes = file_size_nonzero(&audio_wav)
        .map_err(|err| format!("{err}; run `cargo xtask gen-fixtures` first"))?;

    let video_mp4 = fixtures::video_mp4_path(root);
    let video_bytes = file_size_nonzero(&video_mp4)
        .map_err(|err| format!("{err}; run `cargo xtask gen-fixtures` first"))?;

    Ok(VerifiedBaselineFixtures {
        images_dir: fixtures::images_dir(root),
        image_batch_bytes,
        audio_wav,
        audio_bytes,
        video_mp4,
        video_bytes,
    })
}

fn file_size_nonzero(path: &Path) -> Result<NonZeroU64, String> {
    let meta = fs::metadata(path)
        .map_err(|err| format!("missing fixture/output '{}': {err}", path.display()))?;
    if !meta.is_file() {
        return Err(format!("expected regular file at '{}'", path.display()));
    }
    NonZeroU64::new(meta.len())
        .ok_or_else(|| format!("file '{}' must not be empty (0 bytes)", path.display()))
}

fn read_ffmpeg_version() -> Result<String, String> {
    let output = Command::new("ffmpeg")
        .stdin(Stdio::null())
        .arg("-version")
        .output()
        .map_err(|err| format!("failed to run `ffmpeg -version`: {err}"))?;
    if !output.status.success() {
        return Err(format!("`ffmpeg -version` failed with {}", output.status));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout
        .lines()
        .next()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .ok_or_else(|| "`ffmpeg -version` produced empty output".to_owned())?;
    Ok(first_line.to_owned())
}

fn parse_ffmpeg_codec_support(codecs_output: &str, codec_name: &str) -> (bool, bool) {
    let mut past_separator = false;
    for line in codecs_output.lines() {
        let trimmed = line.trim();
        if !past_separator {
            if trimmed.starts_with("------") {
                past_separator = true;
            }
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let Some(flags) = parts.next() else {
            continue;
        };
        let Some(name) = parts.next() else {
            continue;
        };
        if name == codec_name {
            let bytes = flags.as_bytes();
            let can_decode = bytes.first() == Some(&b'D');
            let can_encode = bytes.get(1) == Some(&b'E');
            return (can_decode, can_encode);
        }
    }
    (false, false)
}

fn verify_ffmpeg_qoi_support() -> Result<(), String> {
    let output = Command::new("ffmpeg")
        .stdin(Stdio::null())
        .args(["-hide_banner", "-loglevel", "error", "-codecs"])
        .output()
        .map_err(|err| format!("failed to run `ffmpeg -codecs`: {err}"))?;
    if !output.status.success() {
        return Err(format!("`ffmpeg -codecs` failed with {}", output.status));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let (can_decode, can_encode) = parse_ffmpeg_codec_support(&stdout, "qoi");
    if !can_decode || !can_encode {
        return Err(format!(
            "installed ffmpeg lacks required QOI support (decoder={can_decode}, encoder={can_encode}); install FFmpeg >= 5.1 built with QOI codec support before running `cargo xtask bench`"
        ));
    }
    Ok(())
}

fn verify_ffprobe_available() -> Result<(), String> {
    let output = Command::new("ffprobe")
        .stdin(Stdio::null())
        .arg("-version")
        .output()
        .map_err(|err| {
            format!(
                "failed to run `ffprobe -version`: {err}; install `ffprobe` alongside `ffmpeg` before running `cargo xtask bench`"
            )
        })?;
    if !output.status.success() {
        return Err(format!("`ffprobe -version` failed with {}", output.status));
    }
    Ok(())
}

fn print_hardware_summary(hardware: &HostHardware) {
    let cpu = hardware.cpu();
    println!("==> fastconv benchmark oracle");
    println!(
        "cpu           : {} ({}C/{}T, {})",
        cpu.model_name(),
        cpu.physical_cores(),
        cpu.logical_threads(),
        cpu.simd_level()
    );
    if hardware.wgpu_adapters().is_empty() {
        println!("wgpu adapters : none");
    } else {
        for adapter in hardware.wgpu_adapters() {
            println!(
                "wgpu adapter  : {} [{} | {} | driver={}]",
                adapter.name(),
                adapter.graphics_api(),
                adapter.device_type(),
                adapter.driver()
            );
        }
    }
    for backend in hardware.gpu_backends() {
        println!(
            "gpu backend   : {} [{}] ({})",
            backend.backend(),
            backend.capability(),
            backend.evidence()
        );
    }
}

fn bench_image_batch_webp(
    images_dir: &Path,
    input_bytes: NonZeroU64,
    out_dir: &Path,
) -> Result<BenchmarkMetrics, String> {
    let input_pattern = images_dir.join("img_%02d.png");
    let output_pattern = out_dir.join("img_%02d.webp");
    let in_str = path_to_str(&input_pattern)?;
    let out_str = path_to_str(&output_pattern)?;

    println!("==> running image_batch_4k_png_to_webp...");
    let (duration, peak_memory) = run_timed_ffmpeg(&[
        "-start_number",
        "1",
        "-i",
        in_str,
        "-f",
        "image2",
        "-c:v",
        "libwebp",
        "-quality",
        "80",
        "-start_number",
        "1",
        out_str,
    ])?;

    let mut output_total = 0_u64;
    for index in 1..=IMAGE_BATCH_COUNT {
        let out_file = out_dir.join(format!("img_{index:02}.webp"));
        let bytes = file_size_nonzero(&out_file)?;
        output_total = output_total
            .checked_add(bytes.get())
            .ok_or_else(|| "WebP output byte size overflow".to_owned())?;
    }
    let output_bytes = NonZeroU64::new(output_total)
        .ok_or_else(|| "WebP batch produced 0 output bytes".to_owned())?;

    BenchmarkMetrics::new(
        "image_batch_4k_png_to_webp",
        MediaKind::Image,
        "ffmpeg (libwebp q=80)",
        BenchmarkSample {
            duration,
            input_bytes,
            output_bytes,
            unit: WorkloadUnit::Bytes,
            peak_memory,
        },
    )
    .map_err(|err| err.to_string())
}

fn bench_audio_transcode(
    workload_name: &str,
    tool_label: &str,
    input_path: &Path,
    input_bytes: NonZeroU64,
    output_path: &Path,
    codec_args: &[&str],
) -> Result<BenchmarkMetrics, String> {
    let in_str = path_to_str(input_path)?;
    let out_str = path_to_str(output_path)?;

    let mut args = vec!["-i", in_str];
    args.extend_from_slice(codec_args);
    args.push(out_str);

    println!("==> running {workload_name}...");
    let (duration, peak_memory) = run_timed_ffmpeg(&args)?;
    let output_bytes = file_size_nonzero(output_path)?;

    BenchmarkMetrics::new(
        workload_name,
        MediaKind::Audio,
        tool_label,
        BenchmarkSample {
            duration,
            input_bytes,
            output_bytes,
            unit: WorkloadUnit::Bytes,
            peak_memory,
        },
    )
    .map_err(|err| err.to_string())
}

fn bench_video_transcode(
    workload_name: &str,
    tool_label: &str,
    input_path: &Path,
    input_bytes: NonZeroU64,
    output_path: &Path,
    codec_args: &[&str],
) -> Result<BenchmarkMetrics, String> {
    let in_str = path_to_str(input_path)?;
    let out_str = path_to_str(output_path)?;

    let mut args = vec!["-i", in_str];
    args.extend_from_slice(codec_args);
    args.push(out_str);

    println!("==> running {workload_name}...");
    let (duration, peak_memory) = run_timed_ffmpeg(&args)?;
    let output_bytes = file_size_nonzero(output_path)?;
    let frames = NonZeroU64::new(VIDEO_TOTAL_FRAMES)
        .ok_or_else(|| "VIDEO_TOTAL_FRAMES must be non-zero".to_owned())?;

    BenchmarkMetrics::new(
        workload_name,
        MediaKind::Video,
        tool_label,
        BenchmarkSample {
            duration,
            input_bytes,
            output_bytes,
            unit: WorkloadUnit::VideoFrames(frames),
            peak_memory,
        },
    )
    .map_err(|err| err.to_string())
}

fn path_to_str(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| format!("path '{}' is not valid UTF-8", path.display()))
}

fn run_timed_ffmpeg(args: &[&str]) -> Result<(Duration, PeakMemory), String> {
    let mut full_args: Vec<&OsStr> = ["-nostdin", "-y", "-hide_banner", "-loglevel", "error"]
        .into_iter()
        .map(OsStr::new)
        .collect();
    full_args.extend(args.iter().map(OsStr::new));
    run_timed_command(OsStr::new("ffmpeg"), &full_args)
}

fn run_timed_command(program: &OsStr, args: &[&OsStr]) -> Result<(Duration, PeakMemory), String> {
    let start = Instant::now();
    let mut child = Command::new(program)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .args(args)
        .spawn()
        .map_err(|err| format!("failed to launch '{}': {err}", program.to_string_lossy()))?;

    let pid = child.id();
    let stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| "failed to capture child stderr pipe".to_owned())?;

    let (status, max_rss_bytes, stderr_result) = thread::scope(|scope| {
        let stderr_reader = scope.spawn(move || -> Result<String, String> {
            let mut buf = String::new();
            let mut pipe = stderr_pipe;
            pipe.read_to_string(&mut buf)
                .map_err(|err| format!("failed to read child stderr: {err}"))?;
            Ok(buf)
        });

        let mut max_rss_bytes = 0_u64;
        let status = loop {
            match sample_proc_peak_rss_bytes(pid) {
                Ok(Some(bytes)) => max_rss_bytes = max_rss_bytes.max(bytes),
                Ok(None) => {}
                Err(err) => break Err(err),
            }
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) => thread::sleep(Duration::from_millis(2)),
                Err(err) => break Err(format!("failed while waiting for child: {err}")),
            }
        };

        let status = match status {
            Ok(status) => Ok(status),
            Err(loop_err) => Err(terminate_and_reap_child(&mut child, &loop_err)),
        };

        let stderr_result = stderr_reader
            .join()
            .unwrap_or_else(|_| Err("stderr reader thread panicked".to_owned()));
        (status, max_rss_bytes, stderr_result)
    });

    let duration = start.elapsed();
    let status = status?;
    let stderr_text = stderr_result?;
    if !status.success() {
        return Err(format!(
            "'{}' failed with {status}: {}",
            program.to_string_lossy(),
            stderr_text.trim()
        ));
    }

    let peak_memory =
        NonZeroU64::new(max_rss_bytes).map_or(PeakMemory::Unavailable, PeakMemory::MeasuredBytes);
    Ok((duration, peak_memory))
}

fn terminate_and_reap_child(child: &mut std::process::Child, reason: &str) -> String {
    match child.try_wait() {
        Ok(Some(_)) => return reason.to_owned(),
        Ok(None) => {}
        Err(err) => return format!("{reason} (and failed to check child state: {err})"),
    }
    match child.kill() {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::InvalidInput => {}
        Err(err) => return format!("{reason} (and failed to kill child: {err})"),
    }
    match child.wait() {
        Ok(_) => reason.to_owned(),
        Err(err) => format!("{reason} (and failed to wait for child: {err})"),
    }
}

#[cfg(target_os = "linux")]
fn sample_proc_peak_rss_bytes(pid: u32) -> Result<Option<u64>, String> {
    let path = format!("/proc/{pid}/status");
    let status = match fs::read_to_string(&path) {
        Ok(content) => content,
        // Child may exit and reap its /proc entry between spawn and try_wait polls.
        Err(err) if err.kind() == io::ErrorKind::NotFound || err.raw_os_error() == Some(3) => {
            return Ok(None);
        }
        Err(err) => return Err(format!("failed to read '{path}': {err}")),
    };
    Ok(parse_proc_status_vm_hwm_bytes(&status))
}

#[cfg(not(target_os = "linux"))]
const fn sample_proc_peak_rss_bytes(_pid: u32) -> Result<Option<u64>, String> {
    Ok(None)
}

fn parse_proc_status_vm_hwm_bytes(status: &str) -> Option<u64> {
    let mut fallback_rss_bytes = None;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            if let Some(kb) = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<u64>().ok())
            {
                return kb.checked_mul(1024);
            }
        } else if fallback_rss_bytes.is_none()
            && let Some(rest) = line.strip_prefix("VmRSS:")
            && let Some(kb) = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<u64>().ok())
        {
            fallback_rss_bytes = kb.checked_mul(1024);
        }
    }
    fallback_rss_bytes
}

#[must_use]
fn format_head_to_head_table(entries: &[&HeadToHeadEntry]) -> String {
    let mut lines = Vec::with_capacity(entries.len() + 2);
    lines.push(format!(
        "{:<28} | {:<26} | {:>10} | {:>14} | {:>10} | {:>10} | {:>10} | {:>10} | {:<15}",
        "Workload",
        "Tool",
        "Time (ms)",
        "Throughput",
        "Peak RAM",
        "Input",
        "Output",
        "Mean PSNR",
        "Validated"
    ));
    lines.push(format!(
        "{:-<28}-+-{:-<26}-+-{:-<10}-+-{:-<14}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<15}",
        "", "", "", "", "", "", "", "", ""
    ));

    for entry in entries {
        let item = &entry.metrics;
        let peak_str = match item.peak_memory() {
            PeakMemory::MeasuredBytes(bytes) => format_mib(bytes.get()),
            PeakMemory::Unavailable => "n/a".to_owned(),
        };
        let validated_str = format!(
            "{}/{} {}",
            entry.validation.files_validated, IMAGE_BATCH_COUNT, entry.validation.resolution
        );
        let quality_str = format_validation_quality(&entry.validation);
        lines.push(format!(
            "{:<28} | {:<26} | {:>10.1} | {:>14} | {:>10} | {:>10} | {:>10} | {:>10} | {:<15}",
            item.workload_name(),
            item.tool_label(),
            item.duration_ms(),
            item.throughput().to_string(),
            peak_str,
            format_mb(item.input_bytes().get()),
            format_mb(item.output_bytes().get()),
            quality_str,
            validated_str,
        ));
    }

    format!("{}\n", lines.join("\n"))
}

#[must_use]
pub fn format_baseline_table(results: &[BenchmarkMetrics]) -> String {
    let mut lines = Vec::with_capacity(results.len() + 2);
    lines.push(format!(
        "{:<26} | {:<24} | {:>10} | {:>24} | {:>10} | {:>10} | {:>10} | {:>7}",
        "Workload", "Tool", "Time (ms)", "Throughput", "Peak RAM", "Input", "Output", "Ratio"
    ));
    lines.push(format!(
        "{:-<26}-+-{:-<24}-+-{:-<10}-+-{:-<24}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<7}",
        "", "", "", "", "", "", "", ""
    ));

    for item in results {
        let peak_str = match item.peak_memory() {
            PeakMemory::MeasuredBytes(bytes) => format_mib(bytes.get()),
            PeakMemory::Unavailable => "n/a".to_owned(),
        };
        lines.push(format!(
            "{:<26} | {:<24} | {:>10.1} | {:>24} | {:>10} | {:>10} | {:>10} | {:>6.2}x",
            item.workload_name(),
            item.tool_label(),
            item.duration_ms(),
            item.throughput().to_string(),
            peak_str,
            format_mb(item.input_bytes().get()),
            format_mb(item.output_bytes().get()),
            item.size_ratio()
        ));
    }

    format!("{}\n", lines.join("\n"))
}

fn format_mb(bytes: u64) -> String {
    let mb = u64_to_f64(bytes) / 1_000_000.0;
    format!("{mb:.2} MB")
}

fn format_mib(bytes: u64) -> String {
    let mib = u64_to_f64(bytes) / (1024.0 * 1024.0);
    format!("{mib:.1} MiB")
}

fn u64_to_f64(value: u64) -> f64 {
    let hi = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let lo = u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(hi) * 4_294_967_296.0 + f64::from(lo)
}

#[cfg(test)]
mod tests {
    use super::{
        BatchValidationSummary, HeadToHeadComparison, HeadToHeadEntry, HeadToHeadReport,
        QualityValidationMode, SourceBatchFormat, ValidationPolicy, compute_rgb_psnr_db,
        format_baseline_table, format_head_to_head_summary, format_head_to_head_table,
        format_validation_quality, parse_ffmpeg_codec_support, parse_proc_status_vm_hwm_bytes,
        validate_jpeg_batch, validate_single_jpeg_pair, validate_single_png_pair,
        validate_single_webp_pair, validate_webp_batch, verify_4k_image_batch,
    };
    use fastconv_domain::{
        BenchmarkMetrics, BenchmarkSample, CpuInfo, HostHardware, ImageInputFormat, MediaKind,
        PeakMemory, SimdLevel, WorkloadUnit,
    };
    use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder};
    use std::env;
    use std::fs::{self, File};
    use std::io::BufWriter;
    use std::num::NonZeroU64;
    use std::path::Path;
    use std::process;
    use std::time::Duration;
    use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};

    fn write_rgb_png(path: &Path, width: u32, height: u32, pixels: &[u8]) -> Result<(), String> {
        let file = File::create(path).map_err(|err| err.to_string())?;
        let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|err| err.to_string())?;
        writer
            .write_image_data(pixels)
            .map_err(|err| err.to_string())
    }

    fn write_rgb_jpeg(path: &Path, width: u16, height: u16, pixels: &[u8]) -> Result<(), String> {
        let encoder = JpegEncoder::new_file(path, 85).map_err(|err| err.to_string())?;
        encoder
            .encode(pixels, width, height, JpegColorType::Rgb)
            .map_err(|err| err.to_string())
    }

    fn make_pattern_rgb(width: u8, height: u8) -> Vec<u8> {
        let mut rgb = Vec::with_capacity(usize::from(width) * usize::from(height) * 3);
        for y in 0..height {
            for x in 0..width {
                rgb.extend_from_slice(&[x.saturating_mul(8), y.saturating_mul(10), x ^ y]);
            }
        }
        rgb
    }

    #[test]
    fn parses_proc_status_vm_hwm_and_falls_back_to_vm_rss() {
        let with_hwm = "\
Name:\tffmpeg
VmHWM:\t  123456 kB
VmRSS:\t  100000 kB
";
        assert_eq!(
            parse_proc_status_vm_hwm_bytes(with_hwm),
            Some(123_456 * 1024)
        );

        let only_rss = "\
Name:\tffmpeg
VmRSS:\t   42000 kB
";
        assert_eq!(
            parse_proc_status_vm_hwm_bytes(only_rss),
            Some(42_000 * 1024)
        );
        assert_eq!(parse_proc_status_vm_hwm_bytes("Name:\tffmpeg\n"), None);
    }

    #[test]
    fn computes_psnr_and_rejects_mismatched_lengths() -> Result<(), String> {
        let a = vec![100_u8; 30];
        let mut b = vec![101_u8; 30];
        assert!(compute_rgb_psnr_db(&a, &a)?.is_infinite());

        let db = compute_rgb_psnr_db(&a, &b)?;
        assert!((db - 48.1308).abs() < 0.01);

        b.pop();
        assert!(compute_rgb_psnr_db(&a, &b).is_err());
        assert!(compute_rgb_psnr_db(&[], &[]).is_err());
        Ok(())
    }

    #[test]
    fn validates_webp_output_and_rejects_corrupt_or_mismatched_files() -> Result<(), String> {
        let root = env::temp_dir().join(format!("fastconv-xtask-val-{}", process::id()));
        fs::create_dir_all(&root).map_err(|err| err.to_string())?;

        let mut rgb = Vec::with_capacity(32 * 24 * 3);
        for y in 0_u8..24 {
            for x in 0_u8..32 {
                rgb.extend_from_slice(&[x.saturating_mul(8), y.saturating_mul(10), x ^ y]);
            }
        }
        let png_path = root.join("src.png");
        write_rgb_png(&png_path, 32, 24, &rgb)?;

        let valid_webp = root.join("valid.webp");
        let cfg = LossyConfig::new().with_quality(80.0).with_method(4);
        let encoded = EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, 32, 24)
            .encode()
            .map_err(|err| err.to_string())?;
        fs::write(&valid_webp, &encoded).map_err(|err| err.to_string())?;

        let policy = ValidationPolicy {
            min_output_bytes: 32,
            require_smaller_than_source: false,
            mode: QualityValidationMode::LossyMinPsnrDb(25.0),
        };
        let validated =
            validate_single_webp_pair(&png_path, SourceBatchFormat::Png, &valid_webp, policy)?;
        assert!(validated.bytes >= 32);
        assert_eq!((validated.width, validated.height), (32, 24));
        assert!(validated.psnr_db.is_some_and(|p| p > 25.0));

        let corrupt_webp = root.join("corrupt.webp");
        fs::write(&corrupt_webp, vec![0_u8; 128]).map_err(|err| err.to_string())?;
        assert!(
            validate_single_webp_pair(&png_path, SourceBatchFormat::Png, &corrupt_webp, policy)
                .is_err()
        );

        let wrong_dim_webp = root.join("wrong_dim.webp");
        let wrong_encoded =
            EncodeRequest::lossy(&cfg, &rgb[..16 * 12 * 3], PixelLayout::Rgb8, 16, 12)
                .encode()
                .map_err(|err| err.to_string())?;
        fs::write(&wrong_dim_webp, &wrong_encoded).map_err(|err| err.to_string())?;
        assert!(
            validate_single_webp_pair(&png_path, SourceBatchFormat::Png, &wrong_dim_webp, policy)
                .is_err()
        );

        let too_small_policy = ValidationPolicy {
            min_output_bytes: 1_000_000,
            ..policy
        };
        assert!(
            validate_single_webp_pair(
                &png_path,
                SourceBatchFormat::Png,
                &valid_webp,
                too_small_policy
            )
            .is_err()
        );

        let black_rgb = vec![0_u8; 32 * 24 * 3];
        let low_psnr_webp = root.join("black.webp");
        let black_encoded = EncodeRequest::lossy(&cfg, &black_rgb, PixelLayout::Rgb8, 32, 24)
            .encode()
            .map_err(|err| err.to_string())?;
        fs::write(&low_psnr_webp, &black_encoded).map_err(|err| err.to_string())?;
        assert!(
            validate_single_webp_pair(&png_path, SourceBatchFormat::Png, &low_psnr_webp, policy)
                .is_err()
        );

        let flat_png = root.join("flat.png");
        write_rgb_png(&flat_png, 32, 24, &black_rgb)?;
        let smaller_policy = ValidationPolicy {
            min_output_bytes: 32,
            require_smaller_than_source: true,
            mode: QualityValidationMode::LossyMinPsnrDb(25.0),
        };
        assert!(
            validate_single_webp_pair(
                &flat_png,
                SourceBatchFormat::Png,
                &valid_webp,
                smaller_policy
            )
            .is_err()
        );

        fs::remove_dir_all(&root).map_err(|err| err.to_string())?;
        Ok(())
    }

    #[test]
    fn validates_lossless_exact_pixels_and_rejects_single_byte_drift() -> Result<(), String> {
        let root = env::temp_dir().join(format!("fastconv-xtask-exact-{}", process::id()));
        fs::create_dir_all(&root).map_err(|err| err.to_string())?;

        let rgb = make_pattern_rgb(32, 24);
        let src_png = root.join("src.png");
        let exact_png = root.join("exact.png");
        write_rgb_png(&src_png, 32, 24, &rgb)?;
        write_rgb_png(&exact_png, 32, 24, &rgb)?;

        let exact_policy = ValidationPolicy {
            min_output_bytes: 32,
            require_smaller_than_source: false,
            mode: QualityValidationMode::LosslessExactPixels,
        };
        let ok =
            validate_single_png_pair(&src_png, SourceBatchFormat::Png, &exact_png, exact_policy)?;
        assert!(ok.exact_pixel_match);
        assert_eq!(ok.psnr_db, None);

        let mut altered = rgb;
        altered[12] = altered[12].wrapping_add(1);
        let drifted_png = root.join("drifted.png");
        write_rgb_png(&drifted_png, 32, 24, &altered)?;
        assert!(
            validate_single_png_pair(&src_png, SourceBatchFormat::Png, &drifted_png, exact_policy)
                .is_err()
        );

        fs::remove_dir_all(&root).map_err(|err| err.to_string())?;
        Ok(())
    }

    #[test]
    fn validates_webp_batch_and_rejects_unexpected_extra_files() -> Result<(), String> {
        let root = env::temp_dir().join(format!("fastconv-xtask-batch-{}", process::id()));
        let src_dir = root.join("src");
        let out_dir = root.join("out");
        fs::create_dir_all(&src_dir).map_err(|err| err.to_string())?;
        fs::create_dir_all(&out_dir).map_err(|err| err.to_string())?;

        let rgb = make_pattern_rgb(32, 24);
        let cfg = LossyConfig::new().with_quality(80.0).with_method(4);
        let encoded = EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, 32, 24)
            .encode()
            .map_err(|err| err.to_string())?;

        for idx in 1..=2 {
            write_rgb_png(&src_dir.join(format!("img_{idx:02}.png")), 32, 24, &rgb)?;
            fs::write(out_dir.join(format!("img_{idx:02}.webp")), &encoded)
                .map_err(|err| err.to_string())?;
        }

        let policy = ValidationPolicy {
            min_output_bytes: 32,
            require_smaller_than_source: false,
            mode: QualityValidationMode::LossyMinPsnrDb(25.0),
        };
        let summary = validate_webp_batch(&src_dir, SourceBatchFormat::Png, &out_dir, 2, policy)?;
        assert_eq!(summary.files_validated, 2);
        assert_eq!(summary.resolution, "32x24");

        fs::write(out_dir.join("extra.webp"), &encoded).map_err(|err| err.to_string())?;
        assert!(
            validate_webp_batch(&src_dir, SourceBatchFormat::Png, &out_dir, 2, policy).is_err()
        );

        fs::remove_dir_all(&root).map_err(|err| err.to_string())?;
        Ok(())
    }

    #[test]
    fn validates_jpeg_output_and_batch_rejecting_corrupt_or_wrong_res() -> Result<(), String> {
        let root = env::temp_dir().join(format!("fastconv-xtask-jpeg-val-{}", process::id()));
        let src_dir = root.join("src");
        let out_dir = root.join("out");
        fs::create_dir_all(&src_dir).map_err(|err| err.to_string())?;
        fs::create_dir_all(&out_dir).map_err(|err| err.to_string())?;

        let rgb = make_pattern_rgb(32, 24);
        let png_src = src_dir.join("img_01.png");
        let valid_jpg = out_dir.join("img_01.jpg");
        write_rgb_png(&png_src, 32, 24, &rgb)?;
        write_rgb_jpeg(&valid_jpg, 32, 24, &rgb)?;

        let policy = ValidationPolicy {
            min_output_bytes: 32,
            require_smaller_than_source: false,
            mode: QualityValidationMode::LossyMinPsnrDb(25.0),
        };
        let single =
            validate_single_jpeg_pair(&png_src, SourceBatchFormat::Png, &valid_jpg, policy)?;
        assert_eq!((single.width, single.height), (32, 24));
        assert!(single.psnr_db.is_some_and(|p| p > 25.0));

        let batch = validate_jpeg_batch(&src_dir, SourceBatchFormat::Png, &out_dir, 1, policy)?;
        assert_eq!(batch.files_validated, 1);

        let wrong_dim_jpg = root.join("wrong.jpg");
        write_rgb_jpeg(&wrong_dim_jpg, 16, 12, &rgb[..16 * 12 * 3])?;
        assert!(
            validate_single_jpeg_pair(&png_src, SourceBatchFormat::Png, &wrong_dim_jpg, policy)
                .is_err()
        );

        let corrupt_jpg = root.join("corrupt.jpg");
        fs::write(&corrupt_jpg, b"not-jpeg-data-at-all").map_err(|err| err.to_string())?;
        assert!(
            validate_single_jpeg_pair(&png_src, SourceBatchFormat::Png, &corrupt_jpg, policy)
                .is_err()
        );

        let jpg_src = root.join("src_jpg.jpg");
        write_rgb_jpeg(&jpg_src, 32, 24, &rgb)?;
        let cfg = LossyConfig::new().with_quality(80.0).with_method(4);
        let webp_from_jpg = root.join("from_jpg.webp");
        let webp_bytes = EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, 32, 24)
            .encode()
            .map_err(|err| err.to_string())?;
        fs::write(&webp_from_jpg, &webp_bytes).map_err(|err| err.to_string())?;
        let webp_res =
            validate_single_webp_pair(&jpg_src, SourceBatchFormat::Jpeg, &webp_from_jpg, policy)?;
        assert_eq!((webp_res.width, webp_res.height), (32, 24));

        fs::remove_dir_all(&root).map_err(|err| err.to_string())?;
        Ok(())
    }

    #[test]
    fn formats_baseline_and_head_to_head_tables() -> Result<(), String> {
        let input = NonZeroU64::new(300_000_000).ok_or_else(|| "bad const".to_owned())?;
        let output = NonZeroU64::new(5_000_000).ok_or_else(|| "bad const".to_owned())?;
        let mem = NonZeroU64::new(64 * 1024 * 1024).ok_or_else(|| "bad const".to_owned())?;

        let fastconv_metrics = BenchmarkMetrics::new(
            "image_batch_4k_png_to_webp",
            MediaKind::Image,
            "fastconv (zenwebp q=80)",
            BenchmarkSample {
                duration: Duration::from_millis(2_500),
                input_bytes: input,
                output_bytes: output,
                unit: WorkloadUnit::Bytes,
                peak_memory: PeakMemory::MeasuredBytes(mem),
            },
        )
        .map_err(|err| err.to_string())?;
        let ffmpeg_metrics = BenchmarkMetrics::new(
            "image_batch_4k_png_to_webp",
            MediaKind::Image,
            "ffmpeg (libwebp q=80)",
            BenchmarkSample {
                duration: Duration::from_secs(15),
                input_bytes: input,
                output_bytes: output,
                unit: WorkloadUnit::Bytes,
                peak_memory: PeakMemory::MeasuredBytes(mem),
            },
        )
        .map_err(|err| err.to_string())?;

        let baseline = format_baseline_table(std::slice::from_ref(&ffmpeg_metrics));
        assert!(baseline.contains("image_batch_4k_png_to_webp"));
        assert!(baseline.contains("64.0 MiB"));

        let cpu_entry = HeadToHeadEntry {
            metrics: fastconv_metrics,
            validation: BatchValidationSummary {
                files_validated: 20,
                resolution: "3840x2160".to_owned(),
                total_output_bytes: output,
                exact_pixel_match: false,
                min_psnr_db: Some(39.5),
                mean_psnr_db: Some(40.2),
            },
        };
        let ffmpeg_entry = HeadToHeadEntry {
            metrics: ffmpeg_metrics,
            validation: BatchValidationSummary {
                files_validated: 20,
                resolution: "3840x2160".to_owned(),
                total_output_bytes: output,
                exact_pixel_match: false,
                min_psnr_db: Some(39.4),
                mean_psnr_db: Some(40.1),
            },
        };
        let exact_entry = HeadToHeadEntry {
            metrics: BenchmarkMetrics::new(
                "image_batch_4k_png_to_qoi",
                MediaKind::Image,
                "fastconv (cpu qoi)",
                BenchmarkSample {
                    duration: Duration::from_millis(400),
                    input_bytes: input,
                    output_bytes: output,
                    unit: WorkloadUnit::Bytes,
                    peak_memory: PeakMemory::MeasuredBytes(mem),
                },
            )
            .map_err(|err| err.to_string())?,
            validation: BatchValidationSummary {
                files_validated: 20,
                resolution: "3840x2160".to_owned(),
                total_output_bytes: output,
                exact_pixel_match: true,
                min_psnr_db: None,
                mean_psnr_db: None,
            },
        };
        let h2h = format_head_to_head_table(&[&cpu_entry, &ffmpeg_entry, &exact_entry]);
        assert!(h2h.contains("Mean PSNR"));
        assert!(h2h.contains("fastconv (cpu qoi)"));
        assert!(h2h.contains("40.20 dB"));
        assert!(h2h.contains("exact"));
        assert!(h2h.contains("20/20 3840x2160"));
        Ok(())
    }

    #[test]
    fn rejects_truncated_jpeg_scan_and_non_4k_fixture() -> Result<(), String> {
        let root = env::temp_dir().join(format!("fastconv-xtask-strict-jpg-{}", process::id()));
        fs::create_dir_all(&root).map_err(|err| err.to_string())?;

        let mut rgb = Vec::with_capacity(128 * 128 * 3);
        for y in 0_u8..128 {
            for x in 0_u8..128 {
                rgb.extend_from_slice(&[x.wrapping_mul(3), y.wrapping_mul(5), x ^ y]);
            }
        }
        let png_src = root.join("src.png");
        let trunc_jpg = root.join("truncated.jpg");
        write_rgb_png(&png_src, 128, 128, &rgb)?;
        write_rgb_jpeg(&trunc_jpg, 128, 128, &rgb)?;

        let mut jpg_bytes = fs::read(&trunc_jpg).map_err(|err| err.to_string())?;
        if jpg_bytes.len() <= 1088 {
            return Err(format!(
                "test JPEG unexpectedly small: {} bytes",
                jpg_bytes.len()
            ));
        }
        jpg_bytes.truncate(jpg_bytes.len() - 64);
        fs::write(&trunc_jpg, &jpg_bytes).map_err(|err| err.to_string())?;

        let policy = ValidationPolicy {
            min_output_bytes: 1024,
            require_smaller_than_source: false,
            mode: QualityValidationMode::LossyMinPsnrDb(25.0),
        };
        assert!(
            validate_single_jpeg_pair(&png_src, SourceBatchFormat::Png, &trunc_jpg, policy)
                .is_err()
        );

        let fixture_root = root.join("bad_fixtures");
        let bad_jpg_dir = fixture_root.join("fixtures/images_4k_jpeg");
        fs::create_dir_all(&bad_jpg_dir).map_err(|err| err.to_string())?;
        write_rgb_jpeg(
            &bad_jpg_dir.join("img_01.jpg"),
            32,
            24,
            &make_pattern_rgb(32, 24),
        )?;
        let res_err = verify_4k_image_batch(&fixture_root, ImageInputFormat::Jpeg);
        assert!(res_err.is_err());

        fs::remove_dir_all(&root).map_err(|err| err.to_string())?;
        Ok(())
    }

    #[test]
    fn parses_ffmpeg_qoi_codec_support_and_formats_summary_edge_cases() -> Result<(), String> {
        let codecs_ok = "Codecs:\n -------\n DEVI.S qoi                  QOI (Quite OK Image)\n";
        assert_eq!(parse_ffmpeg_codec_support(codecs_ok, "qoi"), (true, true));
        let codecs_dec_only = "Codecs:\n -------\n D.VI.S qoi                  QOI\n";
        assert_eq!(
            parse_ffmpeg_codec_support(codecs_dec_only, "qoi"),
            (true, false)
        );
        let codecs_missing = "Codecs:\n -------\n DEVI.S png                  PNG\n";
        assert_eq!(
            parse_ffmpeg_codec_support(codecs_missing, "qoi"),
            (false, false)
        );

        let nz = |v: u64| NonZeroU64::new(v).ok_or_else(|| "bad const".to_owned());
        let mk_entry = |name: &str,
                        ms: u64,
                        exact: bool,
                        psnr: Option<f64>|
         -> Result<HeadToHeadEntry, String> {
            Ok(HeadToHeadEntry {
                metrics: BenchmarkMetrics::new(
                    name,
                    MediaKind::Image,
                    "tool",
                    BenchmarkSample {
                        duration: Duration::from_millis(ms),
                        input_bytes: nz(1_000_000)?,
                        output_bytes: nz(500_000)?,
                        unit: WorkloadUnit::Bytes,
                        peak_memory: PeakMemory::Unavailable,
                    },
                )
                .map_err(|err| err.to_string())?,
                validation: BatchValidationSummary {
                    files_validated: 20,
                    resolution: "3840x2160".to_owned(),
                    total_output_bytes: nz(500_000)?,
                    exact_pixel_match: exact,
                    min_psnr_db: psnr,
                    mean_psnr_db: psnr,
                },
            })
        };

        let exact_heic = mk_entry("image_batch_4k_png_to_heic", 1000, true, None)?;
        assert_eq!(format_validation_quality(&exact_heic.validation), "exact");

        let hw = HostHardware::new(
            CpuInfo::new(
                "Test CPU",
                "x86_64",
                std::num::NonZeroUsize::MIN,
                std::num::NonZeroUsize::MIN,
                SimdLevel::Scalar,
                None,
            )
            .map_err(|err| err.to_string())?,
            Vec::new(),
            Vec::new(),
        );
        let report = HeadToHeadReport {
            schema_version: 4,
            ffmpeg_version: "ffmpeg test".to_owned(),
            hardware: hw,
            png_to_webp_fastconv: mk_entry("w", 1000, false, Some(34.0))?,
            png_to_webp_ffmpeg: mk_entry("w", 2000, false, Some(33.5))?,
            png_to_jpeg_fastconv: mk_entry("w", 1000, false, Some(34.0))?,
            png_to_jpeg_ffmpeg: mk_entry("w", 2000, false, Some(33.5))?,
            jpeg_to_webp_fastconv: mk_entry("w", 1000, false, Some(38.0))?,
            jpeg_to_webp_ffmpeg: mk_entry("w", 2000, false, Some(38.0))?,
            webp_to_png_fastconv: mk_entry("w", 1000, true, None)?,
            webp_to_png_ffmpeg: mk_entry("w", 2000, true, None)?,
            png_to_qoi_fastconv: mk_entry("w", 1000, true, None)?,
            png_to_qoi_ffmpeg: mk_entry("w", 2000, true, None)?,
            webp_to_bmp_fastconv: mk_entry("w", 1000, true, None)?,
            webp_to_bmp_ffmpeg: mk_entry("w", 2000, true, None)?,
            png_to_heic_nvenc: Some(exact_heic),
            jpeg_to_heic_nvenc: Some(mk_entry("j", 1000, false, None)?),
            webp_to_heic_nvenc: Some(mk_entry("wb", 1000, true, None)?),
            comparisons: vec![
                HeadToHeadComparison {
                    workload_name: "mixed_exact_and_lossy".to_owned(),
                    fastconv_speedup_vs_ffmpeg: 2.5,
                    both_match_reference_rgb: false,
                    fastconv_psnr_delta_db: None,
                },
                HeadToHeadComparison {
                    workload_name: "lossless_pair".to_owned(),
                    fastconv_speedup_vs_ffmpeg: 4.0,
                    both_match_reference_rgb: true,
                    fastconv_psnr_delta_db: None,
                },
            ],
        };

        let summary = format_head_to_head_summary(&report);
        assert!(summary.contains("mixed_exact_and_lossy       : 2.50x vs ffmpeg"));
        assert!(summary.contains(
            "lossless_pair               : 4.00x vs ffmpeg | both lossless (exact RGB vs reference)"
        ));
        assert!(summary.contains("png_to_heic gpu:nvenc") && summary.contains("quality = exact"));
        assert!(summary.contains("jpeg_to_heic gpu:nvenc") && summary.contains("quality = n/a"));
        assert!(!summary.contains("0.00 dB"));
        Ok(())
    }
}
