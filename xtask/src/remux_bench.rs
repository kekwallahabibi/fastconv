use std::fs;
use std::io::Read;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use fastconv_domain::{BenchmarkMetrics, BenchmarkSample, MediaKind, PeakMemory, WorkloadUnit};
use serde::Deserialize;

use crate::bench::format_baseline_table;
use crate::fixtures::{VIDEO_TOTAL_FRAMES, video_mp4_path};

const REPETITIONS: usize = 3;
const MAX_STDERR_BYTES: u64 = 8 * 1024;

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

#[derive(Debug)]
struct RemuxValidation {
    workload: String,
    tool: String,
    codecs: Vec<String>,
    resolution: Option<(u32, u32)>,
    video_packets: u64,
    audio_packets: u64,
    duration_secs: f64,
}

/// Runs the lossless remuxer head-to-head against `ffmpeg -c copy` on the video fixture.
///
/// # Errors
///
/// Returns a diagnostic when fixtures are missing, tools cannot be launched, a remux
/// fails, or an output fails ffprobe/decode validation.
pub fn run(root: &Path) -> Result<(), String> {
    let fastconv = build_fastconv(root)?;
    let source = video_mp4_path(root);
    if !source.is_file() {
        return Err(format!(
            "missing video fixture '{}'; run `cargo xtask gen-fixtures` first",
            source.display()
        ));
    }

    let bench_dir = root.join("bench-output/tmp_remux");
    if bench_dir.exists() {
        fs::remove_dir_all(&bench_dir)
            .map_err(|err| format!("failed to clean '{}': {err}", bench_dir.display()))?;
    }
    fs::create_dir_all(&bench_dir)
        .map_err(|err| format!("failed to create '{}': {err}", bench_dir.display()))?;

    // Build the MKV input for the reverse direction once, outside the timed section.
    let mkv_source = bench_dir.join("input.mkv");
    run_checked(
        &fastconv,
        &["remux", &path_str(&source)?, &path_str(&mkv_source)?],
    )?;

    let threads = thread::available_parallelism()
        .map_err(|err| format!("failed to determine CPU parallelism: {err}"))?;
    println!("==> remux benchmark (ffmpeg -c copy baseline)");
    println!("source        : {}", source.display());
    println!("fastconv      : {}", fastconv.display());
    println!("CPU threads   : {threads}");

    let input_bytes = NonZeroU64::new(
        fs::metadata(&source)
            .map(|m| m.len())
            .map_err(|err| format!("failed to stat '{}': {err}", source.display()))?,
    )
    .ok_or_else(|| "video fixture is empty".to_owned())?;
    let frames = NonZeroU64::new(VIDEO_TOTAL_FRAMES)
        .ok_or_else(|| "video fixture has no frames".to_owned())?;

    let workloads = [
        ("remux_mp4_to_mkv", source.clone(), "mkv"),
        ("remux_mkv_to_mp4", mkv_source.clone(), "mp4"),
    ];

    let mut results = Vec::new();
    let mut validations = Vec::new();
    for (workload, input, ext) in &workloads {
        for (tool, program) in [
            ("fastconv (pure rust)", fastconv.as_path()),
            ("ffmpeg (-c copy)", Path::new("ffmpeg")),
        ] {
            let output = bench_dir.join(format!("{workload}_{}.{ext}", tool_label_slug(tool)));
            let args = remux_args(tool, input, &output, threads);
            let (duration, peak) = measure(program, &args)?;
            let output_bytes = NonZeroU64::new(
                fs::metadata(&output)
                    .map(|m| m.len())
                    .map_err(|err| format!("failed to stat '{}': {err}", output.display()))?,
            )
            .ok_or_else(|| format!("remux output '{}' is empty", output.display()))?;
            results.push(
                BenchmarkMetrics::new(
                    workload,
                    MediaKind::Video,
                    tool,
                    BenchmarkSample {
                        duration,
                        input_bytes,
                        output_bytes,
                        unit: WorkloadUnit::VideoFrames(frames),
                        peak_memory: NonZeroU64::new(peak)
                            .map_or(PeakMemory::Unavailable, PeakMemory::MeasuredBytes),
                    },
                )
                .map_err(|err| err.to_string())?,
            );

            let validation = validate(&output, workload, tool)?;
            validations.push(validation);
        }
    }

    println!();
    println!("{}", format_baseline_table(&results));
    println!("ffprobe validation (video fixture, no decoder errors):");
    for v in &validations {
        let resolution = v
            .resolution
            .map_or_else(|| "n/a".to_owned(), |(w, h)| format!("{w}x{h}"));
        println!(
            "  {:<20} {:<22} codecs={:<14} {} frames={}/{} dur={:.6}s",
            v.workload,
            v.tool,
            v.codecs.join("+"),
            resolution,
            v.video_packets,
            v.audio_packets,
            v.duration_secs
        );
    }

    Ok(())
}

fn tool_label_slug(tool: &str) -> &'static str {
    if tool.starts_with("ffmpeg") {
        "ffmpeg"
    } else {
        "fastconv"
    }
}

fn remux_args(tool: &str, input: &Path, output: &Path, threads: NonZeroUsize) -> Vec<String> {
    let input = path_str_lossy(input);
    let output = path_str_lossy(output);
    if tool.starts_with("ffmpeg") {
        vec![
            "-i".to_owned(),
            input,
            "-map".to_owned(),
            "0".to_owned(),
            "-c".to_owned(),
            "copy".to_owned(),
            output,
        ]
    } else {
        vec![
            "remux".to_owned(),
            input,
            output,
            "--threads".to_owned(),
            threads.to_string(),
        ]
    }
}

fn path_str(path: &Path) -> Result<String, String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("path '{}' is not valid UTF-8", path.display()))
}

fn path_str_lossy(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn build_fastconv(root: &Path) -> Result<PathBuf, String> {
    println!("==> cargo build --release -p fastconv-cli");
    let status = Command::new(crate::cargo_bin())
        .current_dir(root)
        .args(["build", "--release", "-p", "fastconv-cli"])
        .status()
        .map_err(|err| format!("failed to build fastconv release binary: {err}"))?;
    if !status.success() {
        return Err(format!("fastconv release build failed with {status}"));
    }
    Ok(root.join("target/release/fastconv"))
}

fn run_checked(program: &Path, args: &[&str]) -> Result<(), String> {
    let output = Command::new(program)
        .stdin(Stdio::null())
        .args(args)
        .output()
        .map_err(|err| format!("failed to launch '{}': {err}", program.display()))?;
    if !output.status.success() {
        return Err(format!(
            "'{} {:?}' failed: {}",
            program.display(),
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(())
}

fn measure(program: &Path, args: &[String]) -> Result<(Duration, u64), String> {
    let mut durations = Vec::with_capacity(REPETITIONS);
    let mut peak = 0_u64;
    for _ in 0..REPETITIONS {
        let started = Instant::now();
        let (status, rss, stderr) = run_timed(program, args)?;
        durations.push(started.elapsed());
        peak = peak.max(rss);
        if !status.success() {
            return Err(format!(
                "'{}' failed during benchmark: {status}: {}",
                program.display(),
                stderr.trim()
            ));
        }
    }
    durations.sort();
    Ok((durations[durations.len() / 2], peak))
}

fn run_timed(
    program: &Path,
    args: &[String],
) -> Result<(std::process::ExitStatus, u64, String), String> {
    let mut child = Command::new(program)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .args(if program == Path::new("ffmpeg") {
            &["-nostdin", "-y", "-hide_banner", "-loglevel", "error"][..]
        } else {
            &[][..]
        })
        .args(args)
        .spawn()
        .map_err(|err| format!("failed to launch '{}': {err}", program.display()))?;
    let pid = child.id();
    let stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| "failed to capture stderr pipe".to_owned())?;

    thread::scope(|scope| {
        let reader = scope.spawn(move || -> Result<String, String> {
            let mut buf = Vec::new();
            let mut limited = stderr_pipe.take(MAX_STDERR_BYTES);
            limited
                .read_to_end(&mut buf)
                .map_err(|err| format!("failed to drain stderr: {err}"))?;
            Ok(String::from_utf8_lossy(&buf).into_owned())
        });
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
        let stderr = reader
            .join()
            .map_err(|_| "stderr reader thread panicked".to_owned())??;
        status.map(|status| (status, rss, stderr))
    })
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

fn validate(output: &Path, workload: &str, tool: &str) -> Result<RemuxValidation, String> {
    let probe = Command::new("ffprobe")
        .stdin(Stdio::null())
        .args([
            "-v",
            "error",
            "-count_packets",
            "-show_entries",
            "stream=codec_name,codec_type,width,height,nb_read_packets:format=duration",
            "-of",
            "json",
            &path_str(output)?,
        ])
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
        .find(|s| s.codec_type == "video")
        .ok_or_else(|| format!("'{workload}' output has no video stream"))?;
    let audio = parsed
        .streams
        .iter()
        .find(|s| s.codec_type == "audio")
        .ok_or_else(|| format!("'{workload}' output has no audio stream"))?;
    if video.codec_name != "h264" || audio.codec_name != "aac" {
        return Err(format!(
            "'{workload}' codec mismatch: {} / {}",
            video.codec_name, audio.codec_name
        ));
    }
    if video.width != Some(1920) || video.height != Some(1080) {
        return Err(format!(
            "'{workload}' resolution mismatch: {:?}x{:?}",
            video.width, video.height
        ));
    }
    let video_packets = parse_packets(video.nb_read_packets.as_deref())?;
    let audio_packets = parse_packets(audio.nb_read_packets.as_deref())?;
    if video_packets != VIDEO_TOTAL_FRAMES {
        return Err(format!(
            "'{workload}' frame count mismatch: {video_packets} (expected {VIDEO_TOTAL_FRAMES})"
        ));
    }
    let duration_secs = parsed
        .format
        .duration
        .parse::<f64>()
        .map_err(|err| format!("invalid ffprobe duration: {err}"))?;
    if (duration_secs
        - f64::from(u32::try_from(crate::fixtures::VIDEO_DURATION_SECS).unwrap_or(u32::MAX)))
    .abs()
        > 0.1
    {
        return Err(format!(
            "'{workload}' duration mismatch: {duration_secs}s (expected {})",
            crate::fixtures::VIDEO_DURATION_SECS
        ));
    }

    // Decode the entire output and require a clean run (no decoder errors).
    let decode = Command::new("ffmpeg")
        .stdin(Stdio::null())
        .args([
            "-v",
            "error",
            "-nostdin",
            "-i",
            &path_str(output)?,
            "-f",
            "null",
            "-",
        ])
        .output()
        .map_err(|err| format!("failed to launch ffmpeg decode: {err}"))?;
    if !decode.status.success() || !decode.stderr.is_empty() {
        return Err(format!(
            "'{workload}' decode errors: {}",
            String::from_utf8_lossy(&decode.stderr).trim()
        ));
    }

    let mut codecs = vec![video.codec_name.clone(), audio.codec_name.clone()];
    codecs.sort();
    Ok(RemuxValidation {
        workload: workload.to_owned(),
        tool: tool.to_owned(),
        codecs,
        resolution: video.width.zip(video.height),
        video_packets,
        audio_packets,
        duration_secs,
    })
}

fn parse_packets(value: Option<&str>) -> Result<u64, String> {
    let raw = value.ok_or_else(|| "ffprobe did not report nb_read_packets".to_owned())?;
    raw.parse::<u64>()
        .map_err(|err| format!("invalid nb_read_packets '{raw}': {err}"))
}
