use std::fs;
use std::io::Read;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use fastconv_domain::{BenchmarkMetrics, BenchmarkSample, MediaKind, PeakMemory, WorkloadUnit};
use serde::Deserialize;

use crate::bench::format_baseline_table;
use crate::fixtures::{VIDEO_DURATION_SECS, VIDEO_TOTAL_FRAMES, video_mp4_path};

const REPETITIONS: usize = 3;
const MAX_CAPTURE_BYTES: u64 = 64 * 1024;
const MIN_NVENC_PSNR_DB: f64 = 35.0;

#[derive(Deserialize)]
struct FfprobeOutput {
    streams: Vec<FfprobeStream>,
    format: FfprobeFormat,
}

#[derive(Deserialize)]
struct FfprobeStream {
    codec_name: String,
    codec_type: String,
    width: Option<u32>,
    height: Option<u32>,
    nb_read_packets: Option<String>,
}

#[derive(Deserialize)]
struct FfprobeFormat {
    duration: String,
}

#[derive(Deserialize)]
struct FfprobeFramesOutput {
    frames: Vec<FfprobeFrame>,
}

#[derive(Deserialize)]
struct FfprobeFrame {
    pict_type: Option<String>,
}

/// ffprobe emits packet timestamps as JSON numbers on some builds and as
/// strings on others, so both representations are accepted.
#[derive(Clone, Deserialize)]
#[serde(untagged)]
enum TimestampValue {
    Text(String),
    Number(i64),
}

#[derive(Clone, Deserialize)]
struct FfprobePacket {
    pts: Option<TimestampValue>,
    dts: Option<TimestampValue>,
}

#[derive(Deserialize)]
struct FfprobePacketsOutput {
    packets: Vec<FfprobePacket>,
}

struct NvencValidation {
    workload: String,
    tool: String,
    codec: String,
    resolution: String,
    frames: u64,
    b_frames: usize,
    dts_ok: bool,
    size_mb: f64,
    psnr_db: f64,
    duration_s: f64,
    hardware_device: Option<String>,
}

/// Runs a fair NVENC video head-to-head benchmark against `ffmpeg` on the video fixture.
///
/// Both contenders use identical preset, tuning, rate control, CQ, B-frame count and
/// reference mode, AQ settings, lookahead, and GOP so neither tool can win by lowering
/// quality or output size. Every output is validated with `ffprobe`, a clean full decode,
/// and PSNR against the source.
///
/// # Errors
///
/// Returns a diagnostic when NVENC link libraries are missing, fixtures are absent,
/// a build or transcode fails, or any output fails codec/timestamp/decode/quality checks.
pub fn run(root: &Path) -> Result<(), String> {
    if !crate::has_nvenc_link_libraries() {
        return Err(
            "NVENC benchmark requires libnvidia-encode.so and libnvcuvid.so; install the NVIDIA Video Codec SDK and CUDA runtime"
                .to_owned(),
        );
    }

    let source = video_mp4_path(root);
    if !source.is_file() {
        return Err(format!(
            "missing video fixture '{}'; run `cargo xtask gen-fixtures` first",
            source.display()
        ));
    }

    let fastconv = build_fastconv(root)?;
    let bench_dir = root.join("bench-output/tmp_nvenc");
    if bench_dir.exists() {
        fs::remove_dir_all(&bench_dir)
            .map_err(|err| format!("failed to clean '{}': {err}", bench_dir.display()))?;
    }
    fs::create_dir_all(&bench_dir)
        .map_err(|err| format!("failed to create '{}': {err}", bench_dir.display()))?;

    let input_bytes = NonZeroU64::new(
        fs::metadata(&source)
            .map(|meta| meta.len())
            .map_err(|err| format!("failed to stat '{}': {err}", source.display()))?,
    )
    .ok_or_else(|| "video fixture is empty".to_owned())?;
    let frames = NonZeroU64::new(VIDEO_TOTAL_FRAMES)
        .ok_or_else(|| "video fixture has no frames".to_owned())?;

    println!("==> fair NVENC video benchmark (fastconv vs ffmpeg NVENC)");
    println!("source        : {}", source.display());
    println!("fastconv      : {}", fastconv.display());
    println!(
        "settings      : p4/hq, VBR CQ23, bf3 b_ref_mode=middle, spatial-aq=8, temporal-aq, lookahead=16, g=120, copy audio"
    );
    println!("repetitions   : {REPETITIONS} (median duration, max peak RSS)");

    let workloads: [(&str, &str, &str); 2] = [
        ("nvenc_h264_mp4", "h264", "h264_nvenc"),
        ("nvenc_hevc_mp4", "hevc", "hevc_nvenc"),
    ];

    let context = BenchContext {
        fastconv: fastconv.as_path(),
        source: source.as_path(),
        bench_dir: bench_dir.as_path(),
        input_bytes,
        frames,
    };
    let mut results = Vec::new();
    let mut validations = Vec::new();
    for &(workload, codec, encoder) in &workloads {
        for (metrics, validation) in run_workload(workload, codec, encoder, &context)? {
            results.push(metrics);
            validations.push(validation);
        }
    }

    print_results(&results, &validations);
    Ok(())
}

struct BenchContext<'a> {
    fastconv: &'a Path,
    source: &'a Path,
    bench_dir: &'a Path,
    input_bytes: NonZeroU64,
    frames: NonZeroU64,
}

fn run_workload(
    workload: &str,
    codec: &str,
    encoder: &str,
    context: &BenchContext<'_>,
) -> Result<Vec<(BenchmarkMetrics, NvencValidation)>, String> {
    let output_dir = context.bench_dir.join(workload);
    fs::create_dir_all(&output_dir)
        .map_err(|err| format!("failed to create '{}': {err}", output_dir.display()))?;

    let ffmpeg_label = format!("ffmpeg ({encoder})");
    let tools: Vec<(&str, &Path, bool)> = vec![
        ("fastconv (nvenc)", context.fastconv, true),
        (ffmpeg_label.as_str(), Path::new("ffmpeg"), false),
    ];

    let mut entries = Vec::with_capacity(tools.len());
    for &(tool, program, is_fastconv) in &tools {
        let slug = if is_fastconv { "fastconv" } else { "ffmpeg" };
        let output = output_dir.join(format!("{slug}.mp4"));
        let args = if is_fastconv {
            fastconv_args(codec, context.source, &output)
        } else {
            ffmpeg_args(encoder, context.source, &output)
        };

        println!("==> running {workload} [{tool}]...");
        let mut durations = Vec::with_capacity(REPETITIONS);
        let mut peak = 0_u64;
        let mut last_output_bytes: Option<NonZeroU64> = None;
        let mut last_validation: Option<NvencValidation> = None;
        for repetition in 0..REPETITIONS {
            let (elapsed, rss, stdout) = run_repetition(program, &args)?;
            durations.push(elapsed);
            peak = peak.max(rss);
            // Validate every repetition outside the timed interval before the
            // next iteration overwrites the shared output path.
            let hardware_device = if is_fastconv {
                Some(verify_fastconv_hardware(&stdout)?)
            } else {
                None
            };
            let output_bytes = NonZeroU64::new(
                fs::metadata(&output)
                    .map(|meta| meta.len())
                    .map_err(|err| format!("failed to stat '{}': {err}", output.display()))?,
            )
            .ok_or_else(|| {
                format!(
                    "NVENC output '{}' is empty on repetition {}",
                    output.display(),
                    repetition + 1
                )
            })?;
            let validation = validate_nvenc_output(
                &output,
                context.source,
                workload,
                tool,
                codec,
                output_bytes.get(),
                hardware_device,
            )
            .map_err(|err| format!("repetition {} failed validation: {err}", repetition + 1))?;
            last_output_bytes = Some(output_bytes);
            last_validation = Some(validation);
        }
        durations.sort();
        let duration = durations[durations.len() / 2];
        let output_bytes = last_output_bytes
            .ok_or_else(|| format!("'{workload}' [{tool}] recorded no output bytes"))?;
        let validation = last_validation
            .ok_or_else(|| format!("'{workload}' [{tool}] recorded no validation"))?;

        let metrics = BenchmarkMetrics::new(
            workload,
            MediaKind::Video,
            tool,
            BenchmarkSample {
                duration,
                input_bytes: context.input_bytes,
                output_bytes,
                unit: WorkloadUnit::VideoFrames(context.frames),
                peak_memory: NonZeroU64::new(peak)
                    .map_or(PeakMemory::Unavailable, PeakMemory::MeasuredBytes),
            },
        )
        .map_err(|err| err.to_string())?;
        entries.push((metrics, validation));
    }
    Ok(entries)
}

fn print_results(results: &[BenchmarkMetrics], validations: &[NvencValidation]) {
    println!();
    println!("{}", format_baseline_table(results));
    println!("Validation + quality summary (every output, every tool):");
    println!(
        "{:<18} | {:<22} | {:<5} | {:<9} | {:>6} | {:>8} | {:>5} | {:>9} | {:>8} | {:>9}",
        "Workload",
        "Tool",
        "Codec",
        "Resolution",
        "Frames",
        "B-frames",
        "dts_ok",
        "Size MB",
        "PSNR dB",
        "Dur s"
    );
    println!(
        "{:-<18}-+-{:-<22}-+-{:-<5}-+-{:-<9}-+-{:-<6}-+-{:-<8}-+-{:-<5}-+-{:-<9}-+-{:-<8}-+-{:-<9}",
        "", "", "", "", "", "", "", "", "", ""
    );
    for validation in validations {
        println!(
            "{:<18} | {:<22} | {:<5} | {:<9} | {:>6} | {:>8} | {:>5} | {:>9.3} | {:>8.2} | {:>9.3}",
            validation.workload,
            validation.tool,
            validation.codec,
            validation.resolution,
            validation.frames,
            validation.b_frames,
            validation.dts_ok,
            validation.size_mb,
            validation.psnr_db,
            validation.duration_s
        );
        if let Some(device) = &validation.hardware_device {
            println!("  fastconv hardware line: {device} (explicit GPU proof, no CPU fallback)");
        }
    }
}

#[derive(Deserialize)]
struct CargoBuildMessage {
    reason: String,
    target: Option<CargoTarget>,
    executable: Option<String>,
}

#[derive(Deserialize)]
struct CargoTarget {
    name: String,
}

fn parse_cargo_executable_from_messages(stdout: &str, binary_name: &str) -> Option<PathBuf> {
    let mut executable = None;
    for line in stdout.lines() {
        let Ok(message) = serde_json::from_str::<CargoBuildMessage>(line) else {
            continue;
        };
        if message.reason == "compiler-artifact"
            && message
                .target
                .as_ref()
                .is_some_and(|target| target.name == binary_name)
            && let Some(path) = message.executable
        {
            executable = Some(PathBuf::from(path));
        }
    }
    executable
}

fn build_fastconv(root: &Path) -> Result<PathBuf, String> {
    println!("==> cargo build --release -p fastconv-cli --features nvenc");
    let output = Command::new(crate::cargo_bin())
        .current_dir(root)
        .args([
            "build",
            "--release",
            "-p",
            "fastconv-cli",
            "--features",
            "nvenc",
            "--message-format=json-render-diagnostics",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|err| format!("failed to build fastconv release binary with NVENC: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "fastconv NVENC release build failed with {}",
            output.status
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_cargo_executable_from_messages(&stdout, "fastconv").ok_or_else(|| {
        "cargo build succeeded but emitted no executable artifact for 'fastconv'".to_owned()
    })
}

fn fastconv_args(codec: &str, source: &Path, output: &Path) -> Vec<String> {
    vec![
        "video".to_owned(),
        path_str_lossy(source),
        path_str_lossy(output),
        "--vcodec".to_owned(),
        codec.to_owned(),
        "--acodec".to_owned(),
        "copy".to_owned(),
        "--backend".to_owned(),
        "nvenc".to_owned(),
        "--quality".to_owned(),
        "80".to_owned(),
    ]
}

/// Keeps every quality-relevant NVENC knob identical to fastconv's Stage 2 config
/// (P4/HQ, VBR with zero bitrate and CQ 23, 3 B-frames with middle reference,
/// spatial AQ strength 8, temporal AQ, 16-frame lookahead, 120-frame GOP).
fn ffmpeg_args(encoder: &str, source: &Path, output: &Path) -> Vec<String> {
    vec![
        "-i".to_owned(),
        path_str_lossy(source),
        "-c:v".to_owned(),
        encoder.to_owned(),
        "-preset".to_owned(),
        "p4".to_owned(),
        "-tune".to_owned(),
        "hq".to_owned(),
        "-rc".to_owned(),
        "vbr".to_owned(),
        "-cq".to_owned(),
        "23".to_owned(),
        "-b:v".to_owned(),
        "0".to_owned(),
        "-bf".to_owned(),
        "3".to_owned(),
        "-b_ref_mode".to_owned(),
        "middle".to_owned(),
        "-spatial-aq".to_owned(),
        "1".to_owned(),
        "-aq-strength".to_owned(),
        "8".to_owned(),
        "-temporal-aq".to_owned(),
        "1".to_owned(),
        "-rc-lookahead".to_owned(),
        "16".to_owned(),
        "-g".to_owned(),
        "120".to_owned(),
        "-c:a".to_owned(),
        "copy".to_owned(),
        path_str_lossy(output),
    ]
}

fn path_str_lossy(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn run_repetition(program: &Path, args: &[String]) -> Result<(Duration, u64, String), String> {
    let started = Instant::now();
    let (status, rss, stdout, stderr) = run_timed(program, args)?;
    let elapsed = started.elapsed();
    if !status.success() {
        return Err(format!(
            "'{}' failed during benchmark: {status}: {}",
            program.display(),
            stderr.trim()
        ));
    }
    Ok((elapsed, rss, stdout))
}

fn run_timed(program: &Path, args: &[String]) -> Result<(ExitStatus, u64, String, String), String> {
    let ffmpeg = program == Path::new("ffmpeg");
    let mut command = Command::new(program);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if ffmpeg {
        command.args(["-nostdin", "-y", "-hide_banner", "-loglevel", "error"]);
    }
    let mut child = command
        .args(args)
        .spawn()
        .map_err(|err| format!("failed to launch '{}': {err}", program.display()))?;
    let pid = child.id();
    let stdout_pipe = child
        .stdout
        .take()
        .ok_or_else(|| "failed to capture stdout pipe".to_owned())?;
    let stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| "failed to capture stderr pipe".to_owned())?;

    thread::scope(|scope| {
        let out_reader = scope.spawn(move || read_limited(stdout_pipe));
        let err_reader = scope.spawn(move || read_limited(stderr_pipe));
        let mut rss = 0_u64;
        let status = loop {
            if let Some(bytes) = sample_peak_rss(pid) {
                rss = rss.max(bytes);
            }
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) => thread::sleep(Duration::from_millis(2)),
                Err(err) => break Err(format!("failed while waiting for process: {err}")),
            }
        };
        let stdout = out_reader
            .join()
            .map_err(|_| "stdout reader thread panicked".to_owned())??;
        let stderr = err_reader
            .join()
            .map_err(|_| "stderr reader thread panicked".to_owned())??;
        status.map(|status| (status, rss, stdout, stderr))
    })
}

fn read_limited<R: Read>(reader: R) -> Result<String, String> {
    let mut buf = Vec::new();
    reader
        .take(MAX_CAPTURE_BYTES)
        .read_to_end(&mut buf)
        .map_err(|err| format!("failed to drain process output: {err}"))?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn sample_peak_rss(pid: u32) -> Option<u64> {
    let status = fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()
                .and_then(|kb| kb.checked_mul(1024));
        }
    }
    None
}

/// Proves fastconv really used GPU NVENC by reading its first-run stdout. A
/// silent CPU fallback would print a different backend and no NVIDIA device.
fn verify_fastconv_hardware(stdout: &str) -> Result<String, String> {
    if !stdout.to_ascii_lowercase().contains("nvenc") {
        return Err(format!(
            "fastconv stdout does not report the nvenc backend (possible CPU fallback): {stdout}"
        ));
    }
    let line = stdout
        .lines()
        .find(|line| line.contains("converted ") && line.to_ascii_lowercase().contains("nvenc"))
        .ok_or_else(|| format!("fastconv stdout has no conversion report line: {stdout}"))?;
    let descriptor = line
        .split_once('[')
        .and_then(|(_, rest)| rest.split_once(" | "))
        .map(|(descriptor, _)| descriptor.trim().to_owned())
        .ok_or_else(|| format!("fastconv did not print a GPU device name: {line}"))?;
    if !descriptor.to_ascii_lowercase().contains("nvenc")
        || !descriptor.to_ascii_uppercase().contains("NVIDIA")
    {
        return Err(format!(
            "fastconv reported no NVIDIA NVENC device (possible silent CPU fallback): {line}"
        ));
    }
    Ok(descriptor)
}

struct StreamCheck {
    frames: u64,
    duration_s: f64,
    width: u32,
    height: u32,
}

fn validate_nvenc_output(
    output: &Path,
    source: &Path,
    workload: &str,
    tool: &str,
    codec: &str,
    output_bytes: u64,
    hardware_device: Option<String>,
) -> Result<NvencValidation, String> {
    let checked = check_stream_metadata(output, workload, codec)?;
    let b_frames = count_b_frames(output, workload)?;
    if b_frames == 0 {
        return Err(format!(
            "'{workload}' has no B-frames despite bf3/middle config"
        ));
    }
    check_packet_timestamps(output, workload)?;
    check_clean_decode(output, workload)?;
    let psnr_db = measure_psnr(output, source, workload)?;

    Ok(NvencValidation {
        workload: workload.to_owned(),
        tool: tool.to_owned(),
        codec: codec.to_owned(),
        resolution: format!("{}x{}", checked.width, checked.height),
        frames: checked.frames,
        b_frames,
        dts_ok: true,
        size_mb: u64_to_f64(output_bytes) / 1_000_000.0,
        psnr_db,
        duration_s: checked.duration_s,
        hardware_device,
    })
}

fn check_stream_metadata(
    output: &Path,
    workload: &str,
    codec: &str,
) -> Result<StreamCheck, String> {
    let probe_args = vec![
        "-v".to_owned(),
        "error".to_owned(),
        "-count_packets".to_owned(),
        "-show_entries".to_owned(),
        "stream=codec_name,codec_type,width,height,nb_read_packets:format=duration".to_owned(),
        "-of".to_owned(),
        "json".to_owned(),
        path_str_lossy(output),
    ];
    let probe = Command::new("ffprobe")
        .stdin(Stdio::null())
        .args(&probe_args)
        .output()
        .map_err(|err| format!("failed to launch ffprobe: {err}"))?;
    if !probe.status.success() {
        return Err(format!(
            "ffprobe failed for '{}': {}",
            output.display(),
            String::from_utf8_lossy(&probe.stderr).trim()
        ));
    }
    let parsed: FfprobeOutput = serde_json::from_slice(&probe.stdout)
        .map_err(|err| format!("failed to parse ffprobe JSON: {err}"))?;

    let video = parsed
        .streams
        .iter()
        .find(|stream| stream.codec_type == "video")
        .ok_or_else(|| format!("'{workload}' output has no video stream"))?;
    let audio = parsed
        .streams
        .iter()
        .find(|stream| stream.codec_type == "audio")
        .ok_or_else(|| format!("'{workload}' output has no audio stream"))?;
    if video.codec_name != codec || audio.codec_name != "aac" {
        return Err(format!(
            "'{workload}' codec mismatch: video={}, audio={} (expected {codec}/aac)",
            video.codec_name, audio.codec_name
        ));
    }
    if video.width != Some(1920) || video.height != Some(1080) {
        return Err(format!(
            "'{workload}' resolution mismatch: {:?}x{:?}",
            video.width, video.height
        ));
    }
    let frames = parse_packets(video.nb_read_packets.as_deref())?;
    if frames != VIDEO_TOTAL_FRAMES {
        return Err(format!(
            "'{workload}' frame count mismatch: {frames} (expected {VIDEO_TOTAL_FRAMES})"
        ));
    }
    let duration_s = parsed
        .format
        .duration
        .parse::<f64>()
        .map_err(|err| format!("invalid ffprobe duration: {err}"))?;
    let expected_duration = f64::from(u32::try_from(VIDEO_DURATION_SECS).unwrap_or(u32::MAX));
    if (duration_s - expected_duration).abs() > 0.1 {
        return Err(format!(
            "'{workload}' duration mismatch: {duration_s}s (expected {expected_duration}s)"
        ));
    }
    Ok(StreamCheck {
        frames,
        duration_s,
        width: video.width.unwrap_or(0),
        height: video.height.unwrap_or(0),
    })
}

fn check_packet_timestamps(output: &Path, workload: &str) -> Result<(), String> {
    let args = vec![
        "-v".to_owned(),
        "error".to_owned(),
        "-select_streams".to_owned(),
        "v:0".to_owned(),
        "-show_entries".to_owned(),
        "packet=pts,dts".to_owned(),
        "-of".to_owned(),
        "json".to_owned(),
        path_str_lossy(output),
    ];
    let probe = Command::new("ffprobe")
        .stdin(Stdio::null())
        .args(&args)
        .output()
        .map_err(|err| format!("failed to launch ffprobe packet query: {err}"))?;
    if !probe.status.success() {
        return Err(format!(
            "ffprobe packet query failed for '{}': {}",
            output.display(),
            String::from_utf8_lossy(&probe.stderr).trim()
        ));
    }
    let parsed: FfprobePacketsOutput = serde_json::from_slice(&probe.stdout)
        .map_err(|err| format!("failed to parse ffprobe packet JSON: {err}"))?;
    verify_packet_timestamps(&parsed.packets)
        .map_err(|err| format!("'{workload}' packet timestamps invalid: {err}"))
}

fn check_clean_decode(output: &Path, workload: &str) -> Result<(), String> {
    let args = vec![
        "-v".to_owned(),
        "error".to_owned(),
        "-nostdin".to_owned(),
        "-i".to_owned(),
        path_str_lossy(output),
        "-f".to_owned(),
        "null".to_owned(),
        "-".to_owned(),
    ];
    let decode = Command::new("ffmpeg")
        .stdin(Stdio::null())
        .args(&args)
        .output()
        .map_err(|err| format!("failed to launch ffmpeg decode: {err}"))?;
    if !decode.status.success() || !decode.stderr.is_empty() {
        return Err(format!(
            "'{workload}' decode errors: {}",
            String::from_utf8_lossy(&decode.stderr).trim()
        ));
    }
    Ok(())
}

fn measure_psnr(output: &Path, source: &Path, workload: &str) -> Result<f64, String> {
    let args = vec![
        "-nostdin".to_owned(),
        "-hide_banner".to_owned(),
        "-i".to_owned(),
        path_str_lossy(output),
        "-i".to_owned(),
        path_str_lossy(source),
        "-lavfi".to_owned(),
        "psnr".to_owned(),
        "-f".to_owned(),
        "null".to_owned(),
        "-".to_owned(),
    ];
    let psnr = Command::new("ffmpeg")
        .stdin(Stdio::null())
        .args(&args)
        .output()
        .map_err(|err| format!("failed to launch ffmpeg psnr: {err}"))?;
    if !psnr.status.success() {
        return Err(format!(
            "'{workload}' psnr measurement failed: {}",
            String::from_utf8_lossy(&psnr.stderr).trim()
        ));
    }
    let stderr = String::from_utf8_lossy(&psnr.stderr);
    let psnr_db = parse_psnr_average_db(&stderr)
        .ok_or_else(|| format!("could not parse PSNR average from ffmpeg output: {stderr}"))?;
    if psnr_db < MIN_NVENC_PSNR_DB {
        return Err(format!(
            "'{workload}' PSNR {psnr_db:.2} dB is below the {MIN_NVENC_PSNR_DB:.1} dB sanity floor"
        ));
    }
    Ok(psnr_db)
}

fn parse_packets(value: Option<&str>) -> Result<u64, String> {
    let raw = value.ok_or_else(|| "ffprobe did not report nb_read_packets".to_owned())?;
    raw.parse::<u64>()
        .map_err(|err| format!("invalid nb_read_packets '{raw}': {err}"))
}

fn count_b_frames(output: &Path, workload: &str) -> Result<usize, String> {
    let args = vec![
        "-v".to_owned(),
        "error".to_owned(),
        "-select_streams".to_owned(),
        "v:0".to_owned(),
        "-show_entries".to_owned(),
        "frame=pict_type".to_owned(),
        "-of".to_owned(),
        "json".to_owned(),
        path_str_lossy(output),
    ];
    let probe = Command::new("ffprobe")
        .stdin(Stdio::null())
        .args(&args)
        .output()
        .map_err(|err| format!("failed to launch ffprobe frame query: {err}"))?;
    if !probe.status.success() {
        return Err(format!(
            "ffprobe frame query failed for '{}': {}",
            output.display(),
            String::from_utf8_lossy(&probe.stderr).trim()
        ));
    }
    let parsed: FfprobeFramesOutput = serde_json::from_slice(&probe.stdout)
        .map_err(|err| format!("failed to parse ffprobe frame JSON: {err}"))?;
    let count = parsed
        .frames
        .iter()
        .filter(|frame| {
            frame
                .pict_type
                .as_deref()
                .is_some_and(|pict| pict.starts_with('B') || pict.starts_with('b'))
        })
        .count();
    if parsed.frames.is_empty() {
        return Err(format!("'{workload}' ffprobe reported no video frames"));
    }
    Ok(count)
}

/// Checks decode-order packet timestamps: one packet per source frame, strictly
/// increasing DTS, and `dts <= pts` for every packet (B-frame reorder contract).
fn verify_packet_timestamps(packets: &[FfprobePacket]) -> Result<(), String> {
    if packets.len() != usize::try_from(VIDEO_TOTAL_FRAMES).unwrap_or(usize::MAX) {
        return Err(format!(
            "packet count {} (expected {VIDEO_TOTAL_FRAMES})",
            packets.len()
        ));
    }
    let mut previous_dts: Option<i64> = None;
    for (index, packet) in packets.iter().enumerate() {
        let pts = parse_timestamp(packet.pts.as_ref(), "pts", index)?;
        let dts = parse_timestamp(packet.dts.as_ref(), "dts", index)?;
        if dts > pts {
            return Err(format!("packet {index}: dts {dts} exceeds pts {pts}"));
        }
        if let Some(previous) = previous_dts
            && dts <= previous
        {
            return Err(format!(
                "packet {index}: dts {dts} does not increase past previous dts {previous}"
            ));
        }
        previous_dts = Some(dts);
    }
    Ok(())
}

fn parse_timestamp(
    value: Option<&TimestampValue>,
    label: &str,
    index: usize,
) -> Result<i64, String> {
    let value = value.ok_or_else(|| format!("packet {index}: missing {label}"))?;
    match value {
        TimestampValue::Number(number) => Ok(*number),
        TimestampValue::Text(text) => text
            .parse::<i64>()
            .map_err(|err| format!("packet {index}: invalid {label} '{text}': {err}")),
    }
}

/// Extracts the final `average:<dB>` value from ffmpeg's `psnr` filter output.
/// `inf` (identical frames) is returned as positive infinity; malformed lines
/// are skipped so a later valid summary line still resolves.
fn parse_psnr_average_db(output: &str) -> Option<f64> {
    for line in output.lines() {
        let Some(rest) = line.split_once("average:").map(|(_, rest)| rest) else {
            continue;
        };
        let Some(token) = rest.split_whitespace().next() else {
            continue;
        };
        if token.eq_ignore_ascii_case("inf") || token.eq_ignore_ascii_case("infinity") {
            return Some(f64::INFINITY);
        }
        let Ok(value) = token.parse::<f64>() else {
            continue;
        };
        if !value.is_nan() {
            return Some(value);
        }
    }
    None
}

fn u64_to_f64(value: u64) -> f64 {
    let hi = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let lo = u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(hi) * 4_294_967_296.0 + f64::from(lo)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        FfprobePacket, TimestampValue, parse_cargo_executable_from_messages, parse_psnr_average_db,
        verify_packet_timestamps,
    };
    use crate::fixtures::VIDEO_TOTAL_FRAMES;

    #[test]
    fn resolves_fastconv_executable_from_cargo_json_messages() {
        let messages = "\
{\"reason\":\"compiler-artifact\",\"target\":{\"name\":\"fastconv-pipeline\"},\"executable\":null}
{\"reason\":\"compiler-artifact\",\"target\":{\"name\":\"fastconv\"},\"executable\":\"/custom/target/release/fastconv\"}
{\"reason\":\"build-finished\",\"success\":true}
";
        assert_eq!(
            parse_cargo_executable_from_messages(messages, "fastconv"),
            Some(PathBuf::from("/custom/target/release/fastconv"))
        );
        assert_eq!(
            parse_cargo_executable_from_messages(
                "{\"reason\":\"build-finished\",\"success\":true}\n",
                "fastconv"
            ),
            None
        );
    }

    fn packet(pts: &str, dts: &str) -> FfprobePacket {
        FfprobePacket {
            pts: Some(TimestampValue::Text(pts.to_owned())),
            dts: Some(TimestampValue::Text(dts.to_owned())),
        }
    }

    #[test]
    fn parses_psnr_average_from_real_ffmpeg_lines() {
        let real = "\
frame=600 fps=0.0 q=-0.0 size=N/A time=00:00:10.00 bitrate=N/A speed=20x
[Parsed_psnr_0 @ 0x55d7f0] PSNR y:43.123456 u:45.234567 v:44.345678 average:43.789012 min:38.123456 max:55.654321
";
        assert_eq!(parse_psnr_average_db(real), Some(43.789_012));

        let with_prefix =
            "PSNR y:41.234567 u:43.123456 v:42.987654 average:42.123456 min:33.1 max:55.2\n";
        assert_eq!(parse_psnr_average_db(with_prefix), Some(42.123_456));
    }

    #[test]
    fn parses_psnr_infinity_and_skips_malformed_lines() {
        assert_eq!(
            parse_psnr_average_db("PSNR y:inf u:inf v:inf average:inf\n"),
            Some(f64::INFINITY)
        );
        assert_eq!(
            parse_psnr_average_db("PSNR average:infinity\n"),
            Some(f64::INFINITY)
        );
        assert_eq!(parse_psnr_average_db("no psnr here\n"), None);
        assert_eq!(parse_psnr_average_db("PSNR average:not-a-number\n"), None);
        assert_eq!(parse_psnr_average_db("PSNR average:\n"), None);
        assert_eq!(
            parse_psnr_average_db("PSNR average:oops\nPSNR average:40.5\n"),
            Some(40.5)
        );
    }

    #[test]
    fn verifies_packet_timestamps_accepts_b_frame_reorder() {
        let packets: Vec<FfprobePacket> = (0..VIDEO_TOTAL_FRAMES)
            .map(|index| {
                let pts = i64::try_from(index).unwrap_or(0) * 33_333;
                packet(&pts.to_string(), &(pts - 66_666).to_string())
            })
            .collect();
        assert!(verify_packet_timestamps(&packets).is_ok());

        let numeric: Vec<FfprobePacket> = (0..VIDEO_TOTAL_FRAMES)
            .map(|index| {
                let pts = i64::try_from(index).unwrap_or(0) * 33_333;
                FfprobePacket {
                    pts: Some(TimestampValue::Number(pts)),
                    dts: Some(TimestampValue::Number(pts - 66_666)),
                }
            })
            .collect();
        assert!(verify_packet_timestamps(&numeric).is_ok());
    }

    #[test]
    fn verifies_packet_timestamps_rejects_bad_ordering() {
        let valid: Vec<FfprobePacket> = (0..VIDEO_TOTAL_FRAMES)
            .map(|index| {
                let pts = i64::try_from(index).unwrap_or(0) * 33_333;
                packet(&pts.to_string(), &(pts - 66_666).to_string())
            })
            .collect();

        let mut too_few = valid.clone();
        too_few.pop();
        assert!(verify_packet_timestamps(&too_few).is_err());

        let mut non_monotonic = valid.clone();
        let middle = usize::try_from(VIDEO_TOTAL_FRAMES).unwrap_or(0) / 2;
        non_monotonic[middle] = packet("166650", "166650");
        assert!(verify_packet_timestamps(&non_monotonic).is_err());

        let mut dts_after_pts = valid.clone();
        dts_after_pts[10] = packet("333330", "400000");
        assert!(verify_packet_timestamps(&dts_after_pts).is_err());

        let mut missing_dts = valid.clone();
        missing_dts[10] = FfprobePacket {
            pts: Some(TimestampValue::Text("333330".to_owned())),
            dts: None,
        };
        assert!(verify_packet_timestamps(&missing_dts).is_err());

        let mut not_a_number = valid;
        not_a_number[10] = packet("333330", "N/A");
        assert!(verify_packet_timestamps(&not_a_number).is_err());
    }
}
