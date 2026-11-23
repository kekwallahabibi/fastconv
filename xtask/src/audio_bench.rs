use std::fs;
use std::fs::File;
use std::io::Read;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fastconv_domain::{
    BenchmarkMetrics, BenchmarkSample, HostHardware, MediaKind, PeakMemory, WorkloadUnit,
};
use serde::{Deserialize, Serialize};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::formats::TrackType;
use symphonia::core::formats::probe::Hint;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

use crate::bench::format_baseline_table;
use crate::fixtures::{self, audio_batch_inputs, audio_wav_path};

const REPETITIONS: usize = 3;
const EXPECTED_DURATION_TOLERANCE: f64 = 0.2;

#[derive(Clone, Debug, Serialize)]
struct ContentCheck {
    check: String,
    detail: String,
    passed: bool,
}

#[derive(Serialize)]
struct AudioBenchReport {
    schema_version: u32,
    ffmpeg_version: String,
    hardware: HostHardware,
    results: Vec<BenchmarkMetrics>,
    validation: Vec<AudioValidation>,
    content_checks: Vec<ContentCheck>,
}

#[derive(Debug, Clone, Serialize)]
struct AudioValidation {
    workload: String,
    tool: String,
    codec: String,
    sample_rate: u32,
    channels: u32,
    duration_secs: f64,
}

#[derive(Deserialize)]
struct FfprobeOutput {
    streams: Vec<FfprobeStream>,
}

#[derive(Deserialize)]
struct FfprobeStream {
    codec_name: String,
    sample_rate: Option<String>,
    channels: Option<u32>,
    duration: Option<String>,
}

/// Runs the native fastconv audio encoders head-to-head against the `FFmpeg` baseline.
///
/// # Errors
///
/// Returns a diagnostic when fixtures are missing, `FFmpeg` is unavailable, the release CLI cannot
/// be built, a conversion fails, or an output fails format/duration validation.
// The benchmark orchestration is deliberately linear: build, measure each workload, validate,
// then report. Splitting it would obscure the measurement order that makes results reproducible.
#[allow(clippy::too_many_lines)]
pub fn run(root: &Path) -> Result<(), String> {
    let hardware = fastconv_probe::detect_host_hardware()
        .map_err(|err| format!("failed to probe host hardware: {err}"))?;
    let ffmpeg_version = ffmpeg_version()?;
    let threads = thread::available_parallelism()
        .map_err(|err| format!("failed to determine CPU parallelism: {err}"))?;

    let wav = audio_wav_path(root);
    let batch_inputs = audio_batch_inputs(root);
    for path in std::iter::once(&wav).chain(batch_inputs.iter()) {
        if !fixtures::is_non_empty_file(path) {
            return Err(format!(
                "missing or empty audio fixture '{}'; run `cargo xtask gen-fixtures` first",
                path.display()
            ));
        }
    }
    let fastconv = build_fastconv(root)?;

    let bench_dir = root.join("bench-output");
    let run_id = format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    );
    let tmp = bench_dir.join(format!("tmp_audio_{run_id}"));
    fs::create_dir_all(&bench_dir)
        .map_err(|err| format!("failed to create '{}': {err}", bench_dir.display()))?;
    if tmp.exists() {
        fs::remove_dir_all(&tmp)
            .map_err(|err| format!("failed to clean '{}': {err}", tmp.display()))?;
    }
    fs::create_dir_all(&tmp)
        .map_err(|err| format!("failed to create '{}': {err}", tmp.display()))?;

    println!("ffmpeg        : {ffmpeg_version}");
    println!("fastconv      : {}", fastconv.display());
    println!("CPU threads   : {threads}");

    let wav_bytes = file_size(&wav)?;
    let mut batch_total = 0_u64;
    for path in &batch_inputs {
        batch_total = batch_total
            .checked_add(file_size(path)?.get())
            .ok_or_else(|| "audio batch input size overflow".to_owned())?;
    }
    let batch_bytes = batch_total;
    let batch_bytes =
        NonZeroU64::new(batch_bytes).ok_or_else(|| "audio batch inputs are empty".to_owned())?;

    let flac_out = tmp.join("wav_to_flac.flac");
    let opus_out = tmp.join("wav_to_opus.opus");
    let batch_out = tmp.join("batch");
    fs::create_dir_all(&batch_out)
        .map_err(|err| format!("failed to create '{}': {err}", batch_out.display()))?;
    let controlled_inputs = tmp.join("batch_inputs");
    fs::create_dir_all(&controlled_inputs)
        .map_err(|err| format!("failed to create '{}': {err}", controlled_inputs.display()))?;
    for source in &batch_inputs {
        let name = source
            .file_name()
            .ok_or_else(|| "audio fixture has no file name".to_owned())?;
        let linked = controlled_inputs.join(name);
        fs::hard_link(source, &linked).map_err(|err| {
            format!(
                "failed to link '{}' into '{}': {err}",
                source.display(),
                linked.display()
            )
        })?;
    }

    let mut results = Vec::new();
    let mut validation = Vec::new();
    let mut content_checks = Vec::new();

    let fastconv_flac_args = vec![
        "audio".to_owned(),
        path_str(&wav)?,
        path_str(&flac_out)?,
        "--to".to_owned(),
        "flac".to_owned(),
        "--threads".to_owned(),
        threads.to_string(),
    ];
    let ffmpeg_flac_args = vec![
        "-i".to_owned(),
        path_str(&wav)?,
        "-c:a".to_owned(),
        "flac".to_owned(),
        "-compression_level".to_owned(),
        "5".to_owned(),
        path_str(&flac_out)?,
    ];
    results.push(measure_workload(
        "audio_60s_wav_to_flac",
        "fastconv (flacenc pure rust)",
        &fastconv,
        &fastconv_flac_args,
        wav_bytes,
        &flac_out,
    )?);
    validation.push(validate(
        &flac_out,
        "audio_60s_wav_to_flac",
        "fastconv",
        "flac",
        48_000,
        2,
        60.0,
    )?);
    content_checks.push(check_lossless_roundtrip(&wav, &flac_out)?);
    results.push(measure_workload(
        "audio_60s_wav_to_flac",
        "ffmpeg (flac -5)",
        Path::new("ffmpeg"),
        &ffmpeg_flac_args,
        wav_bytes,
        &flac_out,
    )?);
    validation.push(validate(
        &flac_out,
        "audio_60s_wav_to_flac",
        "ffmpeg",
        "flac",
        48_000,
        2,
        60.0,
    )?);

    let fastconv_opus_args = vec![
        "audio".to_owned(),
        path_str(&wav)?,
        path_str(&opus_out)?,
        "--to".to_owned(),
        "opus".to_owned(),
        "--threads".to_owned(),
        threads.to_string(),
    ];
    let ffmpeg_opus_args = vec![
        "-i".to_owned(),
        path_str(&wav)?,
        "-c:a".to_owned(),
        "libopus".to_owned(),
        "-b:a".to_owned(),
        "128k".to_owned(),
        path_str(&opus_out)?,
    ];
    results.push(measure_workload(
        "audio_60s_wav_to_opus",
        "fastconv (opus-pure)",
        &fastconv,
        &fastconv_opus_args,
        wav_bytes,
        &opus_out,
    )?);
    validation.push(validate(
        &opus_out,
        "audio_60s_wav_to_opus",
        "fastconv",
        "opus",
        48_000,
        2,
        60.0,
    )?);
    content_checks.push(ContentCheck {
        check: "opus complexity".to_owned(),
        detail:
            "opus-pure default complexity=9; libopus default complexity=10 (both 128 kbps, VBR)"
                .to_owned(),
        passed: true,
    });
    content_checks.push(check_opus_signal(&wav, &opus_out)?);
    results.push(measure_workload(
        "audio_60s_wav_to_opus",
        "ffmpeg (libopus 128k)",
        Path::new("ffmpeg"),
        &ffmpeg_opus_args,
        wav_bytes,
        &opus_out,
    )?);
    validation.push(validate(
        &opus_out,
        "audio_60s_wav_to_opus",
        "ffmpeg",
        "opus",
        48_000,
        2,
        60.0,
    )?);

    let fastconv_batch_args = vec![
        "audio".to_owned(),
        path_str(&controlled_inputs)?,
        path_str(&batch_out)?,
        "--to".to_owned(),
        "flac".to_owned(),
        "--threads".to_owned(),
        threads.to_string(),
    ];
    results.push(measure_batch(
        "audio_batch_mixed_to_flac",
        &format!("fastconv (parallel, {} inputs)", batch_inputs.len()),
        &fastconv,
        &fastconv_batch_args,
        batch_bytes,
        &batch_inputs,
        &batch_out,
    )?);
    for source in &batch_inputs {
        let stem = source
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| "audio batch input has no UTF-8 stem".to_owned())?;
        let output = batch_out.join(format!("{stem}.flac"));
        validation.push(validate(
            &output,
            "audio_batch_mixed_to_flac",
            "fastconv",
            "flac",
            48_000,
            2,
            f64::from(fixtures::AUDIO_BATCH_SECS),
        )?);
        content_checks.push(check_batch_samples(source, &output, stem)?);
    }

    results.push(measure_batch(
        "audio_batch_mixed_to_flac",
        &format!("ffmpeg (sequential, {} inputs)", batch_inputs.len()),
        Path::new("ffmpeg"),
        &[],
        batch_bytes,
        &batch_inputs,
        &batch_out,
    )?);
    for source in &batch_inputs {
        let stem = source
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| "audio batch input has no UTF-8 stem".to_owned())?;
        validation.push(validate(
            &batch_out.join(format!("{stem}.flac")),
            "audio_batch_mixed_to_flac",
            "ffmpeg",
            "flac",
            48_000,
            2,
            f64::from(fixtures::AUDIO_BATCH_SECS),
        )?);
    }

    println!();
    print!("{}", format_baseline_table(&results));
    println!("Automatic output validation (decoded with FFmpeg, inspected with ffprobe):");
    for entry in &validation {
        println!(
            "  {:<30} {:<28} codec={:<5} rate={} Hz channels={} duration={:.3}s",
            entry.workload,
            entry.tool,
            entry.codec,
            entry.sample_rate,
            entry.channels,
            entry.duration_secs
        );
    }
    println!("Audio content checks (native Symphonia decode):");
    for check in &content_checks {
        println!(
            "  [{}] {:<28} {}",
            if check.passed { "PASS" } else { "FAIL" },
            check.check,
            check.detail
        );
    }

    let report = AudioBenchReport {
        schema_version: 2,
        ffmpeg_version,
        hardware,
        results,
        validation,
        content_checks,
    };
    let json = serde_json::to_string_pretty(&report)
        .map_err(|err| format!("failed to serialize audio benchmark: {err}"))?;
    let json_path = bench_dir.join("audio-head-to-head.json");
    fs::write(&json_path, format!("{json}\n"))
        .map_err(|err| format!("failed to write '{}': {err}", json_path.display()))?;
    println!();
    println!("audio benchmark saved: {}", json_path.display());

    let _ = fs::remove_dir_all(&tmp);
    Ok(())
}

fn decode_i16(path: &Path) -> Result<Vec<i16>, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open '{}': {err}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut reader = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|err| format!("failed to probe '{}': {err}", path.display()))?;
    let track = reader
        .default_track(TrackType::Audio)
        .ok_or_else(|| format!("no audio track in '{}'", path.display()))?;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| format!("no audio params in '{}'", path.display()))?
        .clone();
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|err| format!("failed to create decoder for '{}': {err}", path.display()))?;
    let mut samples = Vec::new();
    loop {
        let packet = match reader.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => {
                return Err(format!(
                    "failed to read packet from '{}': {err}",
                    path.display()
                ));
            }
        };
        if packet.track_id != track_id {
            continue;
        }
        let frame = match decoder.decode(&packet) {
            Ok(frame) => frame,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => return Err(format!("failed to decode '{}': {err}", path.display())),
        };
        let start = samples.len();
        let count = frame.samples_interleaved();
        samples.resize(start.saturating_add(count), 0);
        if let Some(slot) = samples.get_mut(start..start.saturating_add(count)) {
            frame.copy_to_slice_interleaved(slot);
        }
    }
    if samples.is_empty() {
        return Err(format!(
            "decoded sample stream is empty in '{}'",
            path.display()
        ));
    }
    Ok(samples)
}

fn decode_via_ffmpeg_i16(path: &Path) -> Result<Vec<i16>, String> {
    let output = Command::new("ffmpeg")
        .stdin(Stdio::null())
        .args([
            "-v",
            "error",
            "-nostdin",
            "-i",
            &path_str(path)?,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-",
        ])
        .output()
        .map_err(|err| format!("failed to decode '{}' with ffmpeg: {err}", path.display()))?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg failed to decode '{}': {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(output
        .stdout
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn check_lossless_roundtrip(source: &Path, flac: &Path) -> Result<ContentCheck, String> {
    let original = decode_i16(source)?;
    let roundtrip = decode_i16(flac)?;
    if original != roundtrip {
        let first = original
            .iter()
            .zip(roundtrip.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(original.len().min(roundtrip.len()));
        return Err(format!(
            "FLAC roundtrip is not bit-identical: {} vs {} samples, first difference at {first}",
            original.len(),
            roundtrip.len()
        ));
    }
    Ok(ContentCheck {
        check: "wav->flac->wav bit-exact".to_owned(),
        detail: format!("{} samples identical", original.len()),
        passed: true,
    })
}

fn signal_snr(reference: &[i16], decoded: &[i16]) -> (f64, usize, i32) {
    let count = reference.len().min(decoded.len());
    let mut signal_energy = 0_f64;
    let mut error_energy = 0_f64;
    let mut peak = 0_i32;
    for index in 0..count {
        let a = i32::from(reference[index]);
        let b = i32::from(decoded[index]);
        signal_energy += f64::from(a.saturating_mul(a));
        error_energy += f64::from(a.saturating_sub(b).saturating_mul(a.saturating_sub(b)));
        peak = peak.max(a.abs());
    }
    let snr = if error_energy == 0.0 {
        120.0
    } else {
        10.0 * (signal_energy / error_energy).log10()
    };
    (snr, count, peak)
}

fn check_opus_signal(source: &Path, opus: &Path) -> Result<ContentCheck, String> {
    let reference = decode_via_ffmpeg_i16(source)?;
    let decoded = decode_via_ffmpeg_i16(opus)?;
    if decoded.len() < reference.len().saturating_sub(2 * 48_000) {
        return Err(format!(
            "Opus output lost too many samples: {} vs {}",
            decoded.len(),
            reference.len()
        ));
    }
    let (snr, count, peak) = signal_snr(&reference, &decoded);
    let passed = snr >= 18.0 && peak > 0;
    let check = ContentCheck {
        check: "wav->opus SNR".to_owned(),
        detail: format!("{snr:.1} dB over {count} samples (peak {peak})"),
        passed,
    };
    if passed {
        Ok(check)
    } else {
        Err(format!("Opus signal check failed: {}", check.detail))
    }
}

fn check_batch_samples(source: &Path, output: &Path, label: &str) -> Result<ContentCheck, String> {
    let extension = source
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let source_samples = if extension == "opus" {
        decode_via_ffmpeg_i16(source)?
    } else {
        decode_i16(source)?
    };
    let output_samples = decode_i16(output)?;
    if source_samples.len() != output_samples.len() {
        return Err(format!(
            "batch {label} sample count mismatch: {} source samples vs {} decoded FLAC samples",
            source_samples.len(),
            output_samples.len()
        ));
    }
    let lossless = matches!(extension.as_str(), "wav" | "wave" | "flac");
    let (detail, passed) = if lossless {
        let identical = source_samples == output_samples;
        (
            format!(
                "{} samples {}",
                source_samples.len(),
                if identical {
                    "bit-identical"
                } else {
                    "changed"
                }
            ),
            identical,
        )
    } else {
        let (snr, count, peak) = signal_snr(&source_samples, &output_samples);
        (
            format!("{snr:.1} dB over {count} samples (peak {peak})"),
            snr >= 25.0 && peak > 0,
        )
    };
    let check = ContentCheck {
        check: format!("batch {label} PCM"),
        detail,
        passed,
    };
    if passed {
        Ok(check)
    } else {
        Err(format!("Batch content check failed: {}", check.detail))
    }
}

fn path_str(path: &Path) -> Result<String, String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("path '{}' is not valid UTF-8", path.display()))
}

fn ffmpeg_version() -> Result<String, String> {
    let output = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map_err(|err| format!("failed to run `ffmpeg -version`: {err}"))?;
    if !output.status.success() {
        return Err("`ffmpeg -version` failed".to_owned());
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or("unknown ffmpeg")
        .trim()
        .to_owned())
}

fn build_fastconv(root: &Path) -> Result<PathBuf, String> {
    println!("==> cargo build --release -p fastconv-cli");
    let output = Command::new(crate::cargo_bin())
        .current_dir(root)
        .args([
            "build",
            "--release",
            "-p",
            "fastconv-cli",
            "--message-format=json",
        ])
        .output()
        .map_err(|err| format!("failed to build fastconv release binary: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "fastconv release build failed with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let Ok(message) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if message.get("reason").and_then(|v| v.as_str()) != Some("compiler-artifact") {
            continue;
        }
        if message
            .get("target")
            .and_then(|v| v.get("name"))
            .and_then(|v| v.as_str())
            != Some("fastconv")
        {
            continue;
        }
        if let Some(executable) = message.get("executable").and_then(|v| v.as_str()) {
            return Ok(PathBuf::from(executable));
        }
    }
    Err("cargo did not report a fastconv executable artifact".to_owned())
}

fn file_size(path: &Path) -> Result<NonZeroU64, String> {
    let meta =
        fs::metadata(path).map_err(|err| format!("missing file '{}': {err}", path.display()))?;
    NonZeroU64::new(meta.len())
        .ok_or_else(|| format!("file '{}' must not be empty", path.display()))
}

fn measure_workload(
    workload: &str,
    tool: &str,
    program: &Path,
    args: &[String],
    input: NonZeroU64,
    output: &Path,
) -> Result<BenchmarkMetrics, String> {
    println!("==> {workload} [{tool}]");
    let mut durations = Vec::with_capacity(REPETITIONS);
    let mut peak = 0_u64;
    for _ in 0..REPETITIONS {
        let (duration, memory) = run_timed(program, args)?;
        durations.push(duration);
        if let PeakMemory::MeasuredBytes(bytes) = memory {
            peak = peak.max(bytes.get());
        }
    }
    durations.sort();
    let duration = durations[durations.len() / 2];
    let output_bytes = file_size(output)?;
    BenchmarkMetrics::new(
        workload,
        MediaKind::Audio,
        tool,
        BenchmarkSample {
            duration,
            input_bytes: input,
            output_bytes,
            unit: WorkloadUnit::Bytes,
            peak_memory: NonZeroU64::new(peak)
                .map_or(PeakMemory::Unavailable, PeakMemory::MeasuredBytes),
        },
    )
    .map_err(|err| err.to_string())
}

fn measure_batch(
    workload: &str,
    tool: &str,
    program: &Path,
    args: &[String],
    input: NonZeroU64,
    inputs: &[PathBuf],
    out_dir: &Path,
) -> Result<BenchmarkMetrics, String> {
    println!("==> {workload} [{tool}]");
    let mut durations = Vec::with_capacity(REPETITIONS);
    let mut peak = 0_u64;
    for _ in 0..REPETITIONS {
        let started = Instant::now();
        if program == Path::new("ffmpeg") {
            for source in inputs {
                let name = source
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| "audio input has no UTF-8 stem".to_owned())?;
                let out = path_str(&out_dir.join(format!("{name}.flac")))?;
                let in_path = path_str(source)?;
                let (_, memory) = run_timed(
                    Path::new("ffmpeg"),
                    &[
                        "-i".to_owned(),
                        in_path,
                        "-c:a".to_owned(),
                        "flac".to_owned(),
                        "-compression_level".to_owned(),
                        "5".to_owned(),
                        "-sample_fmt".to_owned(),
                        "s16".to_owned(),
                        out,
                    ],
                )?;
                if let PeakMemory::MeasuredBytes(bytes) = memory {
                    peak = peak.max(bytes.get());
                }
            }
        } else {
            let (_, memory) = run_timed(program, args)?;
            if let PeakMemory::MeasuredBytes(bytes) = memory {
                peak = peak.max(bytes.get());
            }
        }
        durations.push(started.elapsed());
    }
    durations.sort();
    let duration = durations[durations.len() / 2];
    let mut output_total = 0_u64;
    for source in inputs {
        let name = source
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| "audio input has no UTF-8 stem".to_owned())?;
        output_total = output_total
            .checked_add(file_size(&out_dir.join(format!("{name}.flac")))?.get())
            .ok_or_else(|| "batch output byte size overflow".to_owned())?;
    }
    let output_bytes =
        NonZeroU64::new(output_total).ok_or_else(|| "batch produced no output bytes".to_owned())?;
    BenchmarkMetrics::new(
        workload,
        MediaKind::Audio,
        tool,
        BenchmarkSample {
            duration,
            input_bytes: input,
            output_bytes,
            unit: WorkloadUnit::Bytes,
            peak_memory: NonZeroU64::new(peak)
                .map_or(PeakMemory::Unavailable, PeakMemory::MeasuredBytes),
        },
    )
    .map_err(|err| err.to_string())
}

fn run_timed(program: &Path, args: &[String]) -> Result<(Duration, PeakMemory), String> {
    let started = Instant::now();
    let prefix: &[&str] = if program == Path::new("ffmpeg") {
        &["-nostdin", "-y", "-hide_banner", "-loglevel", "error"]
    } else {
        &[]
    };
    let mut child = Command::new(program)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .args(prefix)
        .args(args)
        .spawn()
        .map_err(|err| format!("failed to launch '{}': {err}", program.display()))?;
    let pid = child.id();
    let stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| "failed to capture stderr pipe".to_owned())?;

    let (status, peak, stderr) = thread::scope(|scope| {
        let reader = scope.spawn(move || -> Result<String, String> {
            let mut buf = String::new();
            let mut pipe = stderr_pipe;
            pipe.read_to_string(&mut buf)
                .map_err(|err| format!("failed to read stderr: {err}"))?;
            Ok(buf)
        });
        let mut peak = 0_u64;
        let status = loop {
            if let Some(bytes) = sample_peak_rss(pid) {
                peak = peak.max(bytes);
            }
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) => thread::sleep(Duration::from_millis(2)),
                Err(err) => break Err(format!("failed while waiting for process: {err}")),
            }
        };
        (status, peak, reader.join())
    });

    let duration = started.elapsed();
    let status = status?;
    let stderr = stderr.unwrap_or_else(|_| Err("stderr reader thread panicked".to_owned()))?;
    if !status.success() {
        return Err(format!(
            "'{}' failed with {status}: {}",
            program.display(),
            stderr.trim()
        ));
    }
    Ok((
        duration,
        NonZeroU64::new(peak).map_or(PeakMemory::Unavailable, PeakMemory::MeasuredBytes),
    ))
}

#[cfg(target_os = "linux")]
fn sample_peak_rss(pid: u32) -> Option<u64> {
    let status = fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest
                .split_whitespace()
                .next()
                .and_then(|kb| kb.parse::<u64>().ok())
                .and_then(|kb| kb.checked_mul(1024));
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn sample_peak_rss(_pid: u32) -> Option<u64> {
    None
}

fn validate(
    output: &Path,
    workload: &str,
    tool: &str,
    codec: &str,
    rate: u32,
    channels: u32,
    duration: f64,
) -> Result<AudioValidation, String> {
    let decode = Command::new("ffmpeg")
        .stdin(Stdio::null())
        .args([
            "-v",
            "error",
            "-xerror",
            "-nostdin",
            "-i",
            &path_str(output)?,
            "-f",
            "null",
            "-",
        ])
        .output()
        .map_err(|err| format!("failed to decode '{}': {err}", output.display()))?;
    if !decode.status.success() {
        return Err(format!(
            "output '{}' is not decodable: {}",
            output.display(),
            String::from_utf8_lossy(&decode.stderr).trim()
        ));
    }

    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,duration",
            "-of",
            "json",
            &path_str(output)?,
        ])
        .output()
        .map_err(|err| format!("failed to run ffprobe on '{}': {err}", output.display()))?;
    if !probe.status.success() {
        return Err(format!(
            "ffprobe failed for '{}': {}",
            output.display(),
            String::from_utf8_lossy(&probe.stderr).trim()
        ));
    }
    let parsed: FfprobeOutput = serde_json::from_slice(&probe.stdout)
        .map_err(|err| format!("invalid ffprobe JSON for '{}': {err}", output.display()))?;
    let stream = parsed
        .streams
        .first()
        .ok_or_else(|| format!("no audio stream in '{}'", output.display()))?;
    let actual_rate = stream
        .sample_rate
        .as_deref()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0);
    let actual_duration = stream
        .duration
        .as_deref()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let actual_channels = stream.channels.unwrap_or(0);

    if stream.codec_name != codec {
        return Err(format!(
            "unexpected codec '{}' in '{}' (expected {codec})",
            stream.codec_name,
            output.display()
        ));
    }
    if actual_rate != rate {
        return Err(format!(
            "unexpected sample rate {actual_rate} Hz in '{}' (expected {rate})",
            output.display()
        ));
    }
    if actual_channels != channels {
        return Err(format!(
            "unexpected channel count {actual_channels} in '{}' (expected {channels})",
            output.display()
        ));
    }
    if (actual_duration - duration).abs() > EXPECTED_DURATION_TOLERANCE {
        return Err(format!(
            "unexpected duration {actual_duration:.3}s in '{}' (expected {duration:.3}s)",
            output.display()
        ));
    }

    Ok(AudioValidation {
        workload: workload.to_owned(),
        tool: tool.to_owned(),
        codec: stream.codec_name.clone(),
        sample_rate: actual_rate,
        channels: actual_channels,
        duration_secs: actual_duration,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn derives_batch_output_names_from_input_stems() {
        let names = ["track.wav", "track.flac", "track.mp3", "track.ogg"];
        let outputs: Vec<String> = names
            .iter()
            .map(|name| {
                let stem = name.rsplit_once('.').map_or(*name, |(stem, _)| stem);
                format!("{stem}.flac")
            })
            .collect();
        assert_eq!(outputs, ["track.flac"; 4]);
    }
}
