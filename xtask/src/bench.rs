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
    BenchmarkMetrics, BenchmarkSample, GpuBackend, HostHardware, MediaKind, PeakMemory,
    WorkloadUnit,
};
use serde::Serialize;

use crate::fixtures::{self, IMAGE_BATCH_COUNT, VIDEO_TOTAL_FRAMES};

const MIN_PLAUSIBLE_4K_WEBP_BYTES: u64 = 1024;
const MIN_LOSSY_PSNR_DB: f64 = 25.0;

#[derive(Clone, Copy)]
struct ValidationPolicy {
    min_output_bytes: u64,
    require_smaller_than_source: bool,
    min_psnr_db: f64,
}

const HEAD_TO_HEAD_VALIDATION_POLICY: ValidationPolicy = ValidationPolicy {
    min_output_bytes: MIN_PLAUSIBLE_4K_WEBP_BYTES,
    require_smaller_than_source: true,
    min_psnr_db: MIN_LOSSY_PSNR_DB,
};

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
    min_psnr_db: f64,
    mean_psnr_db: f64,
}

#[derive(Serialize)]
struct HeadToHeadEntry {
    metrics: BenchmarkMetrics,
    validation: BatchValidationSummary,
}

#[derive(Serialize)]
struct HeadToHeadReport {
    schema_version: u32,
    ffmpeg_version: String,
    hardware: HostHardware,
    fastconv_cpu: HeadToHeadEntry,
    ffmpeg_webp: HeadToHeadEntry,
    fastconv_gpu_nvenc_heic: Option<HeadToHeadEntry>,
    cpu_vs_ffmpeg_speedup: f64,
    cpu_vs_ffmpeg_psnr_delta_db: f64,
    nvenc_heic_vs_ffmpeg_webp_speedup: Option<f64>,
}

/// Runs a head-to-head benchmark (`fastconv cpu` WebP vs `ffmpeg` WebP, plus optional GPU NVENC
/// HEIC) on the 4K PNG batch, validating every generated image for decodability, exact resolution,
/// size plausibility, and RGB PSNR.
///
/// # Errors
///
/// Returns an error if fixtures are missing, building or executing any contender fails,
/// or any output image fails decode/resolution/size/PSNR validation.
pub fn run_head_to_head(root: &Path) -> Result<(), String> {
    let hardware = fastconv_probe::detect_host_hardware()
        .map_err(|err| format!("failed to probe host hardware: {err}"))?;
    print_hardware_summary(&hardware);

    let fixtures = verify_fixtures(root)?;
    let ffmpeg_version = read_ffmpeg_version()?;
    println!("ffmpeg        : {ffmpeg_version}");

    let has_nvenc =
        hardware.supports_video_encode(GpuBackend::Nvenc) && crate::has_nvenc_link_libraries();
    let fastconv_bin = build_release_fastconv(root, has_nvenc)?;

    let bench_dir = root.join("bench-output");
    let h2h_tmp = bench_dir.join("tmp/head_to_head");
    if h2h_tmp.exists() {
        fs::remove_dir_all(&h2h_tmp)
            .map_err(|err| format!("failed to clean '{}': {err}", h2h_tmp.display()))?;
    }
    let cpu_out = h2h_tmp.join("fastconv_cpu_webp");
    let ffmpeg_out = h2h_tmp.join("ffmpeg_webp");
    let nvenc_out = h2h_tmp.join("fastconv_nvenc_heic");
    for dir in [&cpu_out, &ffmpeg_out, &nvenc_out] {
        fs::create_dir_all(dir)
            .map_err(|err| format!("failed to create '{}': {err}", dir.display()))?;
    }

    let fastconv_cpu = run_fastconv_cpu_webp_batch(
        &fastconv_bin,
        &fixtures.images_dir,
        fixtures.image_batch_bytes,
        &cpu_out,
    )?;

    let ffmpeg_webp = run_ffmpeg_image_batch_validated(
        &fixtures.images_dir,
        fixtures.image_batch_bytes,
        &ffmpeg_out,
    )?;

    let fastconv_gpu_nvenc_heic = if has_nvenc {
        Some(run_fastconv_nvenc_heic_batch(
            &fastconv_bin,
            &fixtures.images_dir,
            fixtures.image_batch_bytes,
            &nvenc_out,
        )?)
    } else {
        println!(
            "==> skipping fastconv (gpu:nvenc heic): NVENC hardware or link libraries unavailable"
        );
        None
    };

    if h2h_tmp.exists() {
        fs::remove_dir_all(&h2h_tmp).map_err(|err| {
            format!(
                "failed to remove temporary outputs '{}': {err}",
                h2h_tmp.display()
            )
        })?;
    }

    let report = build_head_to_head_report(
        ffmpeg_version,
        hardware,
        fastconv_cpu,
        ffmpeg_webp,
        fastconv_gpu_nvenc_heic,
    );

    let mut rows: Vec<&HeadToHeadEntry> = Vec::with_capacity(3);
    rows.push(&report.fastconv_cpu);
    rows.push(&report.ffmpeg_webp);
    if let Some(ref heic) = report.fastconv_gpu_nvenc_heic {
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

fn build_head_to_head_report(
    ffmpeg_version: String,
    hardware: HostHardware,
    fastconv_cpu: HeadToHeadEntry,
    ffmpeg_webp: HeadToHeadEntry,
    fastconv_gpu_nvenc_heic: Option<HeadToHeadEntry>,
) -> HeadToHeadReport {
    let cpu_vs_ffmpeg_speedup =
        ffmpeg_webp.metrics.duration_ms() / fastconv_cpu.metrics.duration_ms();
    let cpu_vs_ffmpeg_psnr_delta_db =
        fastconv_cpu.validation.mean_psnr_db - ffmpeg_webp.validation.mean_psnr_db;
    let nvenc_heic_vs_ffmpeg_webp_speedup = fastconv_gpu_nvenc_heic
        .as_ref()
        .map(|nv| ffmpeg_webp.metrics.duration_ms() / nv.metrics.duration_ms());

    HeadToHeadReport {
        schema_version: 2,
        ffmpeg_version,
        hardware,
        fastconv_cpu,
        ffmpeg_webp,
        fastconv_gpu_nvenc_heic,
        cpu_vs_ffmpeg_speedup,
        cpu_vs_ffmpeg_psnr_delta_db,
        nvenc_heic_vs_ffmpeg_webp_speedup,
    }
}

fn print_head_to_head_summary(report: &HeadToHeadReport) {
    println!(
        "webp cpu vs ffmpeg  : {:.2}x | delta mean PSNR = {:+.2} dB | validated = {}/{} ({})",
        report.cpu_vs_ffmpeg_speedup,
        report.cpu_vs_ffmpeg_psnr_delta_db,
        report.fastconv_cpu.validation.files_validated,
        IMAGE_BATCH_COUNT,
        report.fastconv_cpu.validation.resolution
    );
    if let (Some(heic), Some(vs_ff)) = (
        report.fastconv_gpu_nvenc_heic.as_ref(),
        report.nvenc_heic_vs_ffmpeg_webp_speedup,
    ) {
        let vs_cpu_webp = report.fastconv_cpu.metrics.duration_ms() / heic.metrics.duration_ms();
        println!(
            "heic gpu:nvenc      : {vs_ff:.2}x vs ffmpeg(webp) | {vs_cpu_webp:.2}x vs cpu(webp) | mean PSNR = {:.2} dB | validated = {}/{} ({})",
            heic.validation.mean_psnr_db,
            heic.validation.files_validated,
            IMAGE_BATCH_COUNT,
            heic.validation.resolution
        );
    }
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

    let fixtures = verify_fixtures(root)?;
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

fn run_fastconv_cpu_webp_batch(
    bin_path: &Path,
    images_dir: &Path,
    input_bytes: NonZeroU64,
    out_dir: &Path,
) -> Result<HeadToHeadEntry, String> {
    let tool_label = "fastconv (cpu q=80)";
    println!("==> running {tool_label}...");
    let args = [
        OsStr::new("convert"),
        images_dir.as_os_str(),
        out_dir.as_os_str(),
        OsStr::new("--quality"),
        OsStr::new("80"),
        OsStr::new("--backend"),
        OsStr::new("cpu"),
    ];
    let (duration, peak_memory) = run_timed_command(bin_path.as_os_str(), &args)?;
    let validation = validate_webp_batch(
        images_dir,
        out_dir,
        IMAGE_BATCH_COUNT,
        HEAD_TO_HEAD_VALIDATION_POLICY,
    )?;

    let metrics = BenchmarkMetrics::new(
        "image_batch_4k_png_to_webp",
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

fn run_fastconv_nvenc_heic_batch(
    bin_path: &Path,
    images_dir: &Path,
    input_bytes: NonZeroU64,
    out_dir: &Path,
) -> Result<HeadToHeadEntry, String> {
    let tool_label = "fastconv (gpu:nvenc q=80)";
    println!("==> running {tool_label} [heic]...");
    let args = [
        OsStr::new("convert"),
        images_dir.as_os_str(),
        out_dir.as_os_str(),
        OsStr::new("--format"),
        OsStr::new("heic"),
        OsStr::new("--quality"),
        OsStr::new("80"),
        OsStr::new("--backend"),
        OsStr::new("nvenc"),
    ];
    let (duration, peak_memory) = run_timed_command(bin_path.as_os_str(), &args)?;
    let validation = validate_heic_batch(
        images_dir,
        out_dir,
        IMAGE_BATCH_COUNT,
        HEAD_TO_HEAD_VALIDATION_POLICY,
    )?;

    let metrics = BenchmarkMetrics::new(
        "image_batch_4k_png_to_heic",
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

fn run_ffmpeg_image_batch_validated(
    images_dir: &Path,
    input_bytes: NonZeroU64,
    out_dir: &Path,
) -> Result<HeadToHeadEntry, String> {
    let input_pattern = images_dir.join("img_%02d.png");
    let output_pattern = out_dir.join("img_%02d.webp");
    let in_str = path_to_str(&input_pattern)?;
    let out_str = path_to_str(&output_pattern)?;

    println!("==> running ffmpeg (libwebp q=80)...");
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
        images_dir,
        out_dir,
        IMAGE_BATCH_COUNT,
        HEAD_TO_HEAD_VALIDATION_POLICY,
    )?;

    let metrics = BenchmarkMetrics::new(
        "image_batch_4k_png_to_webp",
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

struct ValidatedImageResult {
    bytes: u64,
    width: u32,
    height: u32,
    psnr_db: f64,
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

fn validate_webp_batch(
    images_dir: &Path,
    webp_dir: &Path,
    count: usize,
    policy: ValidationPolicy,
) -> Result<BatchValidationSummary, String> {
    println!(
        "==> validating {count} generated WebP outputs in {}...",
        webp_dir.display()
    );
    if count == 0 {
        return Err("cannot validate empty WebP batch".to_owned());
    }
    ensure_exact_dir_entry_count(webp_dir, count)?;

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
                        let png_path = images_dir.join(format!("img_{index:02}.png"));
                        let webp_path = webp_dir.join(format!("img_{index:02}.webp"));
                        items.push(validate_single_webp_pair(&png_path, &webp_path, policy)?);
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
                    .unwrap_or_else(|_| Err("validation worker thread panicked".to_owned()))
            })
            .collect()
    });

    summarize_validated_batch(per_worker, count, "WebP")
}

fn validate_single_webp_pair(
    png_path: &Path,
    webp_path: &Path,
    policy: ValidationPolicy,
) -> Result<ValidatedImageResult, String> {
    let png_bytes = file_size_nonzero(png_path)?.get();
    let out_bytes = file_size_nonzero(webp_path)?.get();
    if out_bytes < policy.min_output_bytes {
        return Err(format!(
            "implausibly small WebP output '{}' ({out_bytes} bytes < {} bytes)",
            webp_path.display(),
            policy.min_output_bytes
        ));
    }
    if policy.require_smaller_than_source && out_bytes >= png_bytes {
        return Err(format!(
            "implausibly large WebP output '{}' ({out_bytes} bytes >= source PNG {png_bytes} bytes)",
            webp_path.display()
        ));
    }

    let (src_rgb, src_w, src_h) = decode_png_to_rgb8(png_path)?;
    let webp_data = fs::read(webp_path)
        .map_err(|err| format!("failed to read WebP '{}': {err}", webp_path.display()))?;
    let (webp_rgb, webp_w, webp_h) = zenwebp::oneshot::decode_rgb(&webp_data)
        .map_err(|err| format!("invalid WebP output '{}': {err}", webp_path.display()))?;

    if (webp_w, webp_h) != (src_w, src_h) {
        return Err(format!(
            "resolution mismatch for '{}': source PNG is {src_w}x{src_h}, decoded WebP is {webp_w}x{webp_h}",
            webp_path.display()
        ));
    }

    let psnr = compute_rgb_psnr_db(&src_rgb, &webp_rgb)?;
    if psnr.is_nan() || psnr < policy.min_psnr_db {
        return Err(format!(
            "implausibly low visual quality for '{}' (PSNR {psnr:.2} dB < {:.2} dB)",
            webp_path.display(),
            policy.min_psnr_db
        ));
    }

    Ok(ValidatedImageResult {
        bytes: out_bytes,
        width: webp_w,
        height: webp_h,
        psnr_db: psnr,
    })
}

fn validate_heic_batch(
    images_dir: &Path,
    heic_dir: &Path,
    count: usize,
    policy: ValidationPolicy,
) -> Result<BatchValidationSummary, String> {
    println!(
        "==> validating {count} generated HEIC outputs in {}...",
        heic_dir.display()
    );
    if count == 0 {
        return Err("cannot validate empty HEIC batch".to_owned());
    }
    ensure_exact_dir_entry_count(heic_dir, count)?;

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
                        let png_path = images_dir.join(format!("img_{index:02}.png"));
                        let heic_path = heic_dir.join(format!("img_{index:02}.heic"));
                        items.push(validate_single_heic_pair(&png_path, &heic_path, policy)?);
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
                    .unwrap_or_else(|_| Err("HEIC validation worker thread panicked".to_owned()))
            })
            .collect()
    });

    summarize_validated_batch(per_worker, count, "HEIC")
}

fn summarize_validated_batch(
    per_worker: Vec<Result<Vec<ValidatedImageResult>, String>>,
    count: usize,
    label: &str,
) -> Result<BatchValidationSummary, String> {
    let mut total_bytes = 0_u64;
    let mut expected_dims: Option<(u32, u32)> = None;
    let mut min_psnr = f64::INFINITY;
    let mut psnr_sum = 0_f64;
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
            min_psnr = min_psnr.min(item.psnr_db);
            psnr_sum += item.psnr_db;
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
    let mean_psnr_db = psnr_sum / u64_to_f64(u64::try_from(validated_count).unwrap_or(1));

    Ok(BatchValidationSummary {
        files_validated: validated_count,
        resolution: format!("{width}x{height}"),
        total_output_bytes,
        min_psnr_db: min_psnr,
        mean_psnr_db,
    })
}

fn validate_single_heic_pair(
    png_path: &Path,
    heic_path: &Path,
    policy: ValidationPolicy,
) -> Result<ValidatedImageResult, String> {
    let png_bytes = file_size_nonzero(png_path)?.get();
    let out_bytes = file_size_nonzero(heic_path)?.get();
    if out_bytes < policy.min_output_bytes {
        return Err(format!(
            "implausibly small HEIC output '{}' ({out_bytes} bytes < {} bytes)",
            heic_path.display(),
            policy.min_output_bytes
        ));
    }
    if policy.require_smaller_than_source && out_bytes >= png_bytes {
        return Err(format!(
            "implausibly large HEIC output '{}' ({out_bytes} bytes >= source PNG {png_bytes} bytes)",
            heic_path.display()
        ));
    }

    let header = fs::read(heic_path)
        .map_err(|err| format!("failed to read HEIC '{}': {err}", heic_path.display()))?;
    if header.get(4..12) != Some(b"ftypheic".as_slice()) {
        return Err(format!(
            "invalid HEIC container header in '{}' (expected ftypheic)",
            heic_path.display()
        ));
    }

    let (src_rgb, src_w, src_h) = decode_png_to_rgb8(png_path)?;
    let (heic_w, heic_h) = probe_heic_dimensions_via_ffprobe(heic_path)?;
    if (heic_w, heic_h) != (src_w, src_h) {
        return Err(format!(
            "resolution mismatch for '{}': source PNG is {src_w}x{src_h}, probed HEIC is {heic_w}x{heic_h}",
            heic_path.display()
        ));
    }

    let heic_rgb = decode_heic_to_rgb8_via_ffmpeg(heic_path, heic_w, heic_h)?;
    let psnr = compute_rgb_psnr_db(&src_rgb, &heic_rgb)?;
    if psnr.is_nan() || psnr < policy.min_psnr_db {
        return Err(format!(
            "implausibly low visual quality for '{}' (PSNR {psnr:.2} dB < {:.2} dB)",
            heic_path.display(),
            policy.min_psnr_db
        ));
    }

    Ok(ValidatedImageResult {
        bytes: out_bytes,
        width: heic_w,
        height: heic_h,
        psnr_db: psnr,
    })
}

fn probe_heic_dimensions_via_ffprobe(path: &Path) -> Result<(u32, u32), String> {
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
        .map_err(|err| format!("failed to run ffprobe on '{}': {err}", path.display()))?;
    if !output.status.success() {
        return Err(format!(
            "ffprobe rejected HEIC '{}': {}",
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
        .map_err(|err| format!("invalid HEIC width '{w_str}': {err}"))?;
    let h = h_str
        .parse::<u32>()
        .map_err(|err| format!("invalid HEIC height '{h_str}': {err}"))?;
    Ok((w, h))
}

fn decode_heic_to_rgb8_via_ffmpeg(path: &Path, width: u32, height: u32) -> Result<Vec<u8>, String> {
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
                "failed to decode HEIC '{}' via ffmpeg: {err}",
                path.display()
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg failed to decode HEIC '{}': {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let expected_len = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|px| px.checked_mul(3))
        .ok_or_else(|| "decoded HEIC RGB24 size overflow".to_owned())?;
    if output.stdout.len() != expected_len {
        return Err(format!(
            "decoded HEIC '{}' produced {} bytes, expected {expected_len} ({width}x{height} rgb24)",
            path.display(),
            output.stdout.len()
        ));
    }
    Ok(output.stdout)
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

struct VerifiedFixtures {
    images_dir: PathBuf,
    image_batch_bytes: NonZeroU64,
    audio_wav: PathBuf,
    audio_bytes: NonZeroU64,
    video_mp4: PathBuf,
    video_bytes: NonZeroU64,
}

fn verify_fixtures(root: &Path) -> Result<VerifiedFixtures, String> {
    let mut image_total = 0_u64;
    for index in 1..=IMAGE_BATCH_COUNT {
        let path = fixtures::image_path(root, index);
        let bytes = file_size_nonzero(&path)
            .map_err(|err| format!("{err}; run `cargo xtask gen-fixtures` first"))?;
        image_total = image_total
            .checked_add(bytes.get())
            .ok_or_else(|| "image batch byte size overflow".to_owned())?;
    }
    let image_batch_bytes =
        NonZeroU64::new(image_total).ok_or_else(|| "4K PNG batch is empty".to_owned())?;

    let audio_wav = fixtures::audio_wav_path(root);
    let audio_bytes = file_size_nonzero(&audio_wav)
        .map_err(|err| format!("{err}; run `cargo xtask gen-fixtures` first"))?;

    let video_mp4 = fixtures::video_mp4_path(root);
    let video_bytes = file_size_nonzero(&video_mp4)
        .map_err(|err| format!("{err}; run `cargo xtask gen-fixtures` first"))?;

    Ok(VerifiedFixtures {
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
        "{:<26} | {:<26} | {:>10} | {:>14} | {:>10} | {:>10} | {:>10} | {:>10} | {:<15}",
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
        "{:-<26}-+-{:-<26}-+-{:-<10}-+-{:-<14}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<15}",
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
        lines.push(format!(
            "{:<26} | {:<26} | {:>10.1} | {:>14} | {:>10} | {:>10} | {:>10} | {:>7.2} dB | {:<15}",
            item.workload_name(),
            item.tool_label(),
            item.duration_ms(),
            item.throughput().to_string(),
            peak_str,
            format_mb(item.input_bytes().get()),
            format_mb(item.output_bytes().get()),
            entry.validation.mean_psnr_db,
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
        BatchValidationSummary, HeadToHeadEntry, ValidationPolicy, compute_rgb_psnr_db,
        format_baseline_table, format_head_to_head_table, parse_proc_status_vm_hwm_bytes,
        validate_single_webp_pair, validate_webp_batch,
    };
    use fastconv_domain::{BenchmarkMetrics, BenchmarkSample, MediaKind, PeakMemory, WorkloadUnit};
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
            min_psnr_db: 25.0,
        };
        let validated = validate_single_webp_pair(&png_path, &valid_webp, policy)?;
        assert!(validated.bytes >= 32);
        assert_eq!((validated.width, validated.height), (32, 24));
        assert!(validated.psnr_db > 25.0);

        let corrupt_webp = root.join("corrupt.webp");
        fs::write(&corrupt_webp, vec![0_u8; 128]).map_err(|err| err.to_string())?;
        assert!(validate_single_webp_pair(&png_path, &corrupt_webp, policy).is_err());

        let wrong_dim_webp = root.join("wrong_dim.webp");
        let wrong_encoded =
            EncodeRequest::lossy(&cfg, &rgb[..16 * 12 * 3], PixelLayout::Rgb8, 16, 12)
                .encode()
                .map_err(|err| err.to_string())?;
        fs::write(&wrong_dim_webp, &wrong_encoded).map_err(|err| err.to_string())?;
        assert!(validate_single_webp_pair(&png_path, &wrong_dim_webp, policy).is_err());

        let too_small_policy = ValidationPolicy {
            min_output_bytes: 1_000_000,
            ..policy
        };
        assert!(validate_single_webp_pair(&png_path, &valid_webp, too_small_policy).is_err());

        let black_rgb = vec![0_u8; 32 * 24 * 3];
        let low_psnr_webp = root.join("black.webp");
        let black_encoded = EncodeRequest::lossy(&cfg, &black_rgb, PixelLayout::Rgb8, 32, 24)
            .encode()
            .map_err(|err| err.to_string())?;
        fs::write(&low_psnr_webp, &black_encoded).map_err(|err| err.to_string())?;
        assert!(validate_single_webp_pair(&png_path, &low_psnr_webp, policy).is_err());

        let flat_png = root.join("flat.png");
        write_rgb_png(&flat_png, 32, 24, &black_rgb)?;
        let smaller_policy = ValidationPolicy {
            min_output_bytes: 32,
            require_smaller_than_source: true,
            min_psnr_db: 25.0,
        };
        assert!(validate_single_webp_pair(&flat_png, &valid_webp, smaller_policy).is_err());

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

        let mut rgb = Vec::with_capacity(32 * 24 * 3);
        for y in 0_u8..24 {
            for x in 0_u8..32 {
                rgb.extend_from_slice(&[x.saturating_mul(8), y.saturating_mul(10), x ^ y]);
            }
        }
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
            min_psnr_db: 25.0,
        };
        let summary = validate_webp_batch(&src_dir, &out_dir, 2, policy)?;
        assert_eq!(summary.files_validated, 2);
        assert_eq!(summary.resolution, "32x24");

        fs::write(out_dir.join("extra.webp"), &encoded).map_err(|err| err.to_string())?;
        assert!(validate_webp_batch(&src_dir, &out_dir, 2, policy).is_err());

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
                min_psnr_db: 39.5,
                mean_psnr_db: 40.2,
            },
        };
        let ffmpeg_entry = HeadToHeadEntry {
            metrics: ffmpeg_metrics,
            validation: BatchValidationSummary {
                files_validated: 20,
                resolution: "3840x2160".to_owned(),
                total_output_bytes: output,
                min_psnr_db: 39.4,
                mean_psnr_db: 40.1,
            },
        };
        let nvenc_entry = HeadToHeadEntry {
            metrics: BenchmarkMetrics::new(
                "image_batch_4k_png_to_heic",
                MediaKind::Image,
                "fastconv (gpu:nvenc q=80)",
                BenchmarkSample {
                    duration: Duration::from_millis(1_100),
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
                min_psnr_db: 38.5,
                mean_psnr_db: 39.2,
            },
        };
        let h2h = format_head_to_head_table(&[&cpu_entry, &ffmpeg_entry, &nvenc_entry]);
        assert!(h2h.contains("Mean PSNR"));
        assert!(h2h.contains("fastconv (gpu:nvenc q=80)"));
        assert!(h2h.contains("40.20 dB"));
        assert!(h2h.contains("20/20 3840x2160"));
        Ok(())
    }
}
