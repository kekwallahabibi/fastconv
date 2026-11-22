use std::fs;
use std::io::{self, Read};
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use fastconv_domain::{
    BenchmarkMetrics, BenchmarkSample, GpuBackend, HostHardware, MediaKind, PeakMemory,
    WorkloadUnit,
};
use serde::Serialize;

use crate::fixtures::{self, IMAGE_BATCH_COUNT, VIDEO_TOTAL_FRAMES};

#[derive(Serialize)]
struct BaselineReport {
    schema_version: u32,
    ffmpeg_version: String,
    hardware: HostHardware,
    results: Vec<BenchmarkMetrics>,
}

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
    println!("==> FFmpeg baseline oracle");
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
    let start = Instant::now();
    let mut child = Command::new("ffmpeg")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .args(["-nostdin", "-y", "-hide_banner", "-loglevel", "error"])
        .args(args)
        .spawn()
        .map_err(|err| format!("failed to launch ffmpeg: {err}"))?;

    let pid = child.id();
    let stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| "failed to capture ffmpeg stderr pipe".to_owned())?;

    let (status, max_rss_bytes, stderr_result) = thread::scope(|scope| {
        let stderr_reader = scope.spawn(move || -> Result<String, String> {
            let mut buf = String::new();
            let mut pipe = stderr_pipe;
            pipe.read_to_string(&mut buf)
                .map_err(|err| format!("failed to read ffmpeg stderr: {err}"))?;
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
                Err(err) => break Err(format!("failed while waiting for ffmpeg: {err}")),
            }
        };

        let stderr_result = stderr_reader
            .join()
            .unwrap_or_else(|_| Err("ffmpeg stderr reader thread panicked".to_owned()));
        (status, max_rss_bytes, stderr_result)
    });

    let duration = start.elapsed();
    let status = status?;
    let stderr_text = stderr_result?;
    if !status.success() {
        return Err(format!(
            "ffmpeg failed with {status}: {}",
            stderr_text.trim()
        ));
    }

    let peak_memory =
        NonZeroU64::new(max_rss_bytes).map_or(PeakMemory::Unavailable, PeakMemory::MeasuredBytes);
    Ok((duration, peak_memory))
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
    use super::{format_baseline_table, parse_proc_status_vm_hwm_bytes};
    use fastconv_domain::{BenchmarkMetrics, BenchmarkSample, MediaKind, PeakMemory, WorkloadUnit};
    use std::num::NonZeroU64;
    use std::time::Duration;

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
    fn formats_baseline_table_rows() -> Result<(), String> {
        let input = NonZeroU64::new(10_000_000).ok_or_else(|| "bad const".to_owned())?;
        let output = NonZeroU64::new(2_500_000).ok_or_else(|| "bad const".to_owned())?;
        let mem = NonZeroU64::new(64 * 1024 * 1024).ok_or_else(|| "bad const".to_owned())?;

        let metrics = BenchmarkMetrics::new(
            "audio_60s_wav_to_flac",
            MediaKind::Audio,
            "ffmpeg (flac -5)",
            BenchmarkSample {
                duration: Duration::from_millis(500),
                input_bytes: input,
                output_bytes: output,
                unit: WorkloadUnit::Bytes,
                peak_memory: PeakMemory::MeasuredBytes(mem),
            },
        )
        .map_err(|err| err.to_string())?;

        let table = format_baseline_table(&[metrics]);
        assert!(table.contains("Workload"));
        assert!(table.contains("audio_60s_wav_to_flac"));
        assert!(table.contains("20.00 MB/s"));
        assert!(table.contains("64.0 MiB"));
        Ok(())
    }
}
