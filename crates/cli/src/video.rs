use std::num::NonZeroUsize;
use std::path::PathBuf;

use fastconv_domain::{
    AudioBitrate, AudioChannels, AudioEncodeOptions, AudioSampleRate, HardwareBackend,
    HardwareDecodeBackend, TargetVideoCodec, VideoBitrate, VideoContainerFormat, VideoQuality,
    VideoRateControl,
};
use fastconv_probe::{
    RequestedVideoCodec, VideoTranscodeOptions, parse_requested_audio_codec,
    prepare_video_transcode_batch_with_options,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RemuxArgs {
    source: PathBuf,
    target: PathBuf,
    to: Option<VideoContainerFormat>,
    threads: Option<NonZeroUsize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RemuxCommand {
    Help,
    Run(RemuxArgs),
}

fn parse_target_format(value: &str) -> Result<VideoContainerFormat, String> {
    VideoContainerFormat::from_extension(value).ok_or_else(|| {
        format!("unknown video format '{value}' (expected mp4, mkv, webm, mov, avi, or ts)")
    })
}

pub(crate) const REMUX_HELP: &str = r"fastconv remux - losslessly remux video containers

Usage:
  fastconv remux <source> <target> [--to <mp4|mkv|webm|mov|avi|ts>] [--threads <N>]

Remux across MP4/MOV, MKV, WebM, AVI, and TS without re-encoding.

Arguments:
  <source>              Source video file or directory
  <target>              Target file or directory

Options:
  --to <mp4|mkv|webm|mov|avi|ts>  Output container (required for directory batches)
  --threads <N>         Number of parallel workers
  -h, --help            Print this help

Supported containers: mp4 (m4v), mkv, webm, mov (qt), avi, ts (m2ts, mts)
GIF and WebP cannot be remuxed; use `fastconv video` to encode them.";

pub(crate) fn print_remux_help() {
    println!("{REMUX_HELP}");
}

/// Parses `fastconv remux` arguments without touching the file system.
///
/// # Errors
///
/// Returns a diagnostic for missing positional arguments, duplicate flags, malformed values,
/// or unknown flags.
pub(crate) fn parse_remux_args(args: &[&str]) -> Result<RemuxCommand, String> {
    let mut positional = Vec::new();
    let mut to = None;
    let mut threads = None;
    let mut index = 0_usize;

    while let Some(&arg) = args.get(index) {
        match arg {
            "-h" | "--help" => return Ok(RemuxCommand::Help),
            "--to" => {
                if to.is_some() {
                    return Err("duplicate flag '--to'".to_owned());
                }
                let raw = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --to".to_owned())?;
                let parsed = parse_target_format(raw)?;
                if !parsed.supports_remux() {
                    return Err(format!(
                        "format '{raw}' cannot be remuxed; use `fastconv video` for GIF/WebP"
                    ));
                }
                to = Some(parsed);
                index += 2;
            }
            "--threads" => {
                if threads.is_some() {
                    return Err("duplicate flag '--threads'".to_owned());
                }
                let raw = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --threads".to_owned())?;
                threads = Some(
                    raw.parse::<NonZeroUsize>()
                        .map_err(|err| format!("invalid --threads '{raw}': {err}"))?,
                );
                index += 2;
            }
            flag if flag.starts_with('-') => return Err(format!("unknown flag '{flag}'")),
            value => {
                positional.push(value);
                index += 1;
            }
        }
    }

    let [source, target] = positional.as_slice() else {
        return Err(
            "expected `fastconv remux <source> <target> [--to <mp4|mkv|webm|mov|avi|ts>]`"
                .to_owned(),
        );
    };
    // Do not reject GIF/WebP from the positional target here: an explicit
    // `--to mkv` may name a directory whose path ends in `.gif`. The probe
    // layer rejects animation *file* targets after target resolution.

    Ok(RemuxCommand::Run(RemuxArgs {
        source: PathBuf::from(*source),
        target: PathBuf::from(*target),
        to,
        threads,
    }))
}

/// Orchestrates probe → domain → pipeline for a single file or a video batch.
///
/// # Errors
///
/// Returns a diagnostic if probing, planning, or packet copying fails. Unsupported codecs are
/// rejected by the boundary/domain layers before any file is written.
pub(crate) fn run_remux(args: &RemuxArgs) -> Result<(), String> {
    let plan =
        fastconv_probe::prepare_video_batch(&args.source, &args.target, args.to, args.threads)
            .map_err(|err| err.to_string())?;

    let report = fastconv_pipeline::execute_video_batch(&plan).map_err(|err| err.to_string())?;
    println!(
        "remuxed {} video file(s) [{} -> {} | {} thread(s)]: {:.2} MB -> {:.2} MB in {:.1} ms ({:.2} MB/s)",
        report.files_remuxed(),
        plan.items()
            .first()
            .map_or("?", |i| i.input_format().extension()),
        plan.output_format().extension(),
        report.worker_threads_used(),
        crate::u64_to_f64(report.total_input_bytes().get()) / 1_000_000.0,
        crate::u64_to_f64(report.total_output_bytes().get()) / 1_000_000.0,
        report.duration().as_secs_f64() * 1_000.0,
        report.megabytes_per_sec(),
    );
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VideoArgs {
    source: PathBuf,
    target: PathBuf,
    to: Option<VideoContainerFormat>,
    video_codec: Option<RequestedVideoCodec>,
    /// Raw `--acodec` value. Codec parsing and combination validation belong
    /// to `fastconv-probe`, not this argument collector.
    audio_codec: Option<String>,
    quality: Option<VideoQuality>,
    vbitrate: Option<VideoBitrate>,
    abitrate: Option<AudioBitrate>,
    asample_rate: Option<AudioSampleRate>,
    achannels: Option<AudioChannels>,
    threads: Option<NonZeroUsize>,
    backend: HardwareBackend,
    decode_backend: HardwareDecodeBackend,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VideoCommand {
    Help,
    Run(VideoArgs),
}

pub(crate) const VIDEO_HELP: &str = r"fastconv video - transcode audio/video tracks

Usage:
  fastconv video <source> <target> [--to <format>] [--vcodec <codec>]
                 [--acodec <codec>] [--quality <0..=100> | --vbitrate <rate>]
                 [--abitrate <rate>] [--asample-rate <Hz>] [--achannels <N>]
                 [--threads <N>] [--backend <name>] [--hwdecode <auto|cpu|nvdec>]

Transcode audio/video tracks while converting between MP4/MOV, MKV, WebM, AVI,
TS, GIF, and WebP containers.

Arguments:
  <source>              Source video file or directory
  <target>              Target file or directory

Options:
  --to <format>         Output container (required for directory batches): mp4,
                        mkv, webm, mov, avi, ts, gif, webp
  --vcodec <codec>      Video codec: h264, hevc (h265), av1, vp9, gif, webp, copy
  --acodec <codec>      Audio codec: aac, opus, mp3, vorbis, flac, pcm, none
                        (drop), copy. FLAC/PCM are lossless and reject --abitrate
  --quality <0..=100>   Encoding quality (default 80; mutually exclusive with
                        --vbitrate and --vcodec copy)
  --vbitrate <rate>     Target video bitrate, e.g. 500k, 2M, 2.5M or 2000000
                        (mutually exclusive with --quality and --vcodec copy;
                        rejected for GIF/WebP)
  --abitrate <rate>     Audio bitrate, e.g. 128k or 192000 (requires audio
                        re-encoding: rejected with --acodec copy/none)
  --asample-rate <Hz>   Audio sample rate (requires audio re-encoding; codec-
                        specific limits apply)
  --achannels <N>       Audio channel count (requires audio re-encoding;
                        codec-specific limits apply)
  --threads <N>         Number of parallel workers
  --backend <name>      cpu (default: H.264/AV1/VP9/GIF/WebP and copy) or nvenc
                        (H.264/HEVC only; HEVC has no CPU encoder)
  --hwdecode <name>     auto (default), cpu, or nvdec (NVDEC + NVENC zero-copy).
                        Aliases: software -> cpu, cuda/gpu -> nvdec. A failed
                        NVDEC setup is a hard error, never a CPU fallback.
  -h, --help            Print this help

Backends: cpu encodes H.264/AV1/VP9/GIF/WebP and --vcodec copy; nvenc encodes
          H.264/HEVC only. HEVC requires nvenc; AV1/VP9 require cpu.
Decode: --hwdecode auto uses NVDEC for H.264/HEVC/VP9 with --backend nvenc;
        --hwdecode cpu forces CPU decode; --hwdecode nvdec requires --backend
        nvenc and a hardware-decodable source.
Animated GIF/WebP encoding runs on CPU only and drops audio tracks by default.
Container aliases: m4v -> mp4, qt -> mov, m2ts/mts -> ts";

pub(crate) fn print_video_help() {
    println!("{VIDEO_HELP}");
}

fn flag_value<'a>(args: &'a [&str], index: usize, flag: &str) -> Result<&'a str, String> {
    args.get(index + 1)
        .copied()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse_video_codec(raw: &str) -> Result<RequestedVideoCodec, String> {
    if raw == "copy" {
        return Ok(RequestedVideoCodec::Copy);
    }
    let codec = TargetVideoCodec::from_name(raw).ok_or_else(|| {
        format!(
            "unknown video codec '{raw}' (expected h264, hevc (h265), av1, vp9, gif, webp, or copy)"
        )
    })?;
    Ok(RequestedVideoCodec::Encode(codec))
}

/// Parses `fastconv video` arguments without touching the file system.
///
/// # Errors
///
/// Returns a diagnostic for missing positional arguments, duplicate flags, malformed values,
/// unknown codecs/backends, or combinations with no implementation.
// Each flag is parsed and validated in one explicit branch at the CLI boundary;
// splitting it would hide the mapping from the user-facing error messages.
#[allow(clippy::too_many_lines)]
pub(crate) fn parse_video_args(args: &[&str]) -> Result<VideoCommand, String> {
    let mut positional = Vec::new();
    let mut to = None;
    let mut video_codec = None;
    let mut audio_codec = None;
    let mut quality = None;
    let mut vbitrate = None;
    let mut abitrate = None;
    let mut asample_rate = None;
    let mut achannels = None;
    let mut threads = None;
    let mut backend = None;
    let mut decode_backend = None;
    let mut index = 0_usize;

    while let Some(&arg) = args.get(index) {
        match arg {
            "-h" | "--help" => return Ok(VideoCommand::Help),
            "--to" => {
                if to.is_some() {
                    return Err("duplicate flag '--to'".to_owned());
                }
                to = Some(parse_target_format(flag_value(args, index, "--to")?)?);
                index += 2;
            }
            "--vcodec" => {
                if video_codec.is_some() {
                    return Err("duplicate flag '--vcodec'".to_owned());
                }
                let raw = flag_value(args, index, "--vcodec")?;
                if raw == "copy" && (quality.is_some() || vbitrate.is_some()) {
                    return Err(
                        "cannot combine --quality or --vbitrate with --vcodec copy".to_owned()
                    );
                }
                video_codec = Some(parse_video_codec(raw)?);
                index += 2;
            }
            "--acodec" => {
                if audio_codec.is_some() {
                    return Err("duplicate flag '--acodec'".to_owned());
                }
                // Parsing and codec/option compatibility are validated once at
                // the probe boundary; the CLI only collects the raw value.
                audio_codec = Some(flag_value(args, index, "--acodec")?.to_owned());
                index += 2;
            }
            "--quality" => {
                if quality.is_some() {
                    return Err("duplicate flag '--quality'".to_owned());
                }
                if vbitrate.is_some() {
                    return Err("cannot combine --quality with --vbitrate".to_owned());
                }
                if video_codec == Some(RequestedVideoCodec::Copy) {
                    return Err("cannot combine --quality with --vcodec copy".to_owned());
                }
                let raw = flag_value(args, index, "--quality")?;
                let parsed = raw
                    .parse::<u8>()
                    .map_err(|err| format!("invalid --quality '{raw}': {err}"))?;
                quality = Some(VideoQuality::new(parsed).map_err(|err| err.to_string())?);
                index += 2;
            }
            "--vbitrate" => {
                if vbitrate.is_some() {
                    return Err("duplicate flag '--vbitrate'".to_owned());
                }
                if quality.is_some() {
                    return Err("cannot combine --quality with --vbitrate".to_owned());
                }
                if video_codec == Some(RequestedVideoCodec::Copy) {
                    return Err("cannot combine --vbitrate with --vcodec copy".to_owned());
                }
                let raw = flag_value(args, index, "--vbitrate")?;
                vbitrate = Some(VideoBitrate::parse(raw).map_err(|err| err.to_string())?);
                index += 2;
            }
            "--abitrate" => {
                if abitrate.is_some() {
                    return Err("duplicate flag '--abitrate'".to_owned());
                }
                let raw = flag_value(args, index, "--abitrate")?;
                abitrate = Some(AudioBitrate::parse(raw).map_err(|err| err.to_string())?);
                index += 2;
            }
            "--asample-rate" => {
                if asample_rate.is_some() {
                    return Err("duplicate flag '--asample-rate'".to_owned());
                }
                let raw = flag_value(args, index, "--asample-rate")?;
                let hz = raw
                    .parse::<u32>()
                    .map_err(|err| format!("invalid --asample-rate '{raw}': {err}"))?;
                asample_rate = Some(AudioSampleRate::new(hz).map_err(|err| err.to_string())?);
                index += 2;
            }
            "--achannels" => {
                if achannels.is_some() {
                    return Err("duplicate flag '--achannels'".to_owned());
                }
                let raw = flag_value(args, index, "--achannels")?;
                let count = raw
                    .parse::<u16>()
                    .map_err(|err| format!("invalid --achannels '{raw}': {err}"))?;
                achannels = Some(AudioChannels::new(count).map_err(|err| err.to_string())?);
                index += 2;
            }
            "--threads" => {
                if threads.is_some() {
                    return Err("duplicate flag '--threads'".to_owned());
                }
                let raw = flag_value(args, index, "--threads")?;
                threads = Some(
                    raw.parse::<NonZeroUsize>()
                        .map_err(|err| format!("invalid --threads '{raw}': {err}"))?,
                );
                index += 2;
            }
            "--backend" => {
                if backend.is_some() {
                    return Err("duplicate flag '--backend'".to_owned());
                }
                backend = Some(crate::parse_backend(flag_value(args, index, "--backend")?)?);
                index += 2;
            }
            "--hwdecode" => {
                if decode_backend.is_some() {
                    return Err("duplicate flag '--hwdecode'".to_owned());
                }
                let raw = flag_value(args, index, "--hwdecode")?;
                decode_backend = Some(HardwareDecodeBackend::from_name(raw).ok_or_else(|| {
                    format!("unknown decode backend '{raw}' (expected auto, cpu, or nvdec)")
                })?);
                index += 2;
            }
            flag if flag.starts_with('-') => return Err(format!("unknown flag '{flag}'")),
            value => {
                positional.push(value);
                index += 1;
            }
        }
    }

    let [source, target] = positional.as_slice() else {
        return Err(
            "expected `fastconv video <source> <target> [--to <mp4|mkv|webm|mov|avi|ts|gif|webp>] [options]`"
                .to_owned(),
        );
    };

    Ok(VideoCommand::Run(VideoArgs {
        source: PathBuf::from(*source),
        target: PathBuf::from(*target),
        to,
        video_codec,
        audio_codec,
        quality,
        vbitrate,
        abitrate,
        asample_rate,
        achannels,
        threads,
        backend: backend.unwrap_or(HardwareBackend::Cpu),
        decode_backend: decode_backend.unwrap_or(HardwareDecodeBackend::Auto),
    }))
}

/// Orchestrates probe → domain → pipeline for a single video file or a video batch.
///
/// # Errors
///
/// Returns a diagnostic if probing, planning, decoding, encoding, or muxing fails. Unsupported
/// codec/container pairs and backend combinations are rejected before any file is written.
pub(crate) fn run_video(args: &VideoArgs) -> Result<(), String> {
    let rate_control = match (args.quality, args.vbitrate) {
        (Some(quality), _) => Some(VideoRateControl::Quality(quality)),
        (None, Some(bitrate)) => Some(VideoRateControl::Bitrate(bitrate)),
        (None, None) => None,
    };
    let audio_options = AudioEncodeOptions::new(args.abitrate, args.asample_rate, args.achannels);
    // `--acodec` is parsed once at the probe boundary, together with the
    // codec/option compatibility checks, never in this argument collector.
    let audio_codec = args
        .audio_codec
        .as_deref()
        .map(parse_requested_audio_codec)
        .transpose()
        .map_err(|err| err.to_string())?;
    let plan = prepare_video_transcode_batch_with_options(
        &args.source,
        &args.target,
        args.to,
        args.video_codec,
        audio_codec,
        VideoTranscodeOptions::new(rate_control, audio_options),
        args.backend,
        args.decode_backend,
        args.threads,
    )
    .map_err(|err| err.to_string())?;

    let report =
        fastconv_pipeline::execute_video_transcode_batch(&plan).map_err(|err| err.to_string())?;

    let backend_desc = match report.hardware_device() {
        Some(device) => format!("{} on {device}", report.decode_path()),
        None => report.decode_path().to_owned(),
    };
    let first = plan.items().first();
    let mut input_formats: Vec<&str> = plan
        .items()
        .iter()
        .map(|item| item.input_format().extension())
        .collect();
    input_formats.sort_unstable();
    input_formats.dedup();
    let input_format = match input_formats.as_slice() {
        [] => "?".to_owned(),
        [only] => (*only).to_owned(),
        many => format!("mixed({})", many.join(", ")),
    };
    let video_action = first.map_or_else(|| "?".to_owned(), |item| item.video_action().to_string());
    let audio_action = first.map_or_else(|| "?".to_owned(), |item| item.audio_action().to_string());
    // The sizes below are decimal MB, so throughput must use the same unit;
    // the report's megabytes_per_sec() is MiB/s and would overstate the rate.
    let secs = report.duration().as_secs_f64();
    let mb_per_sec = if secs == 0.0 {
        0.0
    } else {
        crate::u64_to_f64(report.total_input_bytes().get()) / 1_000_000.0 / secs
    };

    println!(
        "converted {} video file(s) [{backend_desc} | {input_format} -> {} | video: {video_action} | audio: {audio_action} | {} frame(s) at {:.1} FPS | {} thread(s)]: {:.2} MB -> {:.2} MB in {:.1} ms ({:.2} MB/s)",
        report.files_converted(),
        plan.output_format().extension(),
        report.frames_processed(),
        report.fps(),
        report.worker_threads_used(),
        crate::u64_to_f64(report.total_input_bytes().get()) / 1_000_000.0,
        crate::u64_to_f64(report.total_output_bytes().get()) / 1_000_000.0,
        report.duration().as_secs_f64() * 1_000.0,
        mb_per_sec,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastconv_domain::GpuBackend;

    #[test]
    fn remux_and_video_help_use_uniform_layout() {
        for (text, flags) in [
            (REMUX_HELP, &["--to", "--threads", "-h, --help"][..]),
            (
                VIDEO_HELP,
                &[
                    "--to",
                    "--vcodec",
                    "--acodec",
                    "--quality",
                    "--vbitrate",
                    "--abitrate",
                    "--asample-rate",
                    "--achannels",
                    "--threads",
                    "--backend",
                    "--hwdecode",
                    "-h, --help",
                ][..],
            ),
        ] {
            assert!(text.contains("Arguments:"));
            assert!(text.contains("Options:"));
            assert!(text.contains("Usage:\n  fastconv "));
            assert!(!text.contains("Commands:"));
            for flag in flags {
                assert!(text.contains(flag), "missing {flag} in help");
            }
        }
        assert!(REMUX_HELP.contains("GIF and WebP cannot be remuxed"));
        assert!(REMUX_HELP.contains("use `fastconv video` to encode them"));
        assert!(VIDEO_HELP.contains("default 80"));
        assert!(VIDEO_HELP.contains("software -> cpu"));
        assert!(VIDEO_HELP.contains("cuda/gpu -> nvdec"));
    }

    #[test]
    fn parses_default_and_flagged_args() -> Result<(), String> {
        assert_eq!(parse_remux_args(&["--help"])?, RemuxCommand::Help);
        assert_eq!(
            parse_remux_args(&["in.mp4", "out.mkv"])?,
            RemuxCommand::Run(RemuxArgs {
                source: PathBuf::from("in.mp4"),
                target: PathBuf::from("out.mkv"),
                to: None,
                threads: None,
            })
        );

        let RemuxCommand::Run(flagged) =
            parse_remux_args(&["in_dir", "out_dir", "--to", "mkv", "--threads", "4"])?
        else {
            return Err("expected remux run args".to_owned());
        };
        assert_eq!(flagged.to, Some(VideoContainerFormat::Mkv));
        assert_eq!(flagged.threads, NonZeroUsize::new(4));
        Ok(())
    }

    #[test]
    fn parses_all_remux_containers_and_aliases() -> Result<(), String> {
        for (raw, expected) in [
            ("mp4", VideoContainerFormat::Mp4),
            ("mkv", VideoContainerFormat::Mkv),
            ("webm", VideoContainerFormat::Webm),
            ("mov", VideoContainerFormat::Mov),
            ("avi", VideoContainerFormat::Avi),
            ("ts", VideoContainerFormat::Ts),
            ("m4v", VideoContainerFormat::Mp4),
            ("qt", VideoContainerFormat::Mov),
            ("m2ts", VideoContainerFormat::Ts),
            ("mts", VideoContainerFormat::Ts),
        ] {
            let RemuxCommand::Run(parsed) = parse_remux_args(&["in.mp4", "out.x", "--to", raw])?
            else {
                return Err("expected remux run args".to_owned());
            };
            assert_eq!(parsed.to, Some(expected), "remux --to {raw}");
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_remux_args() {
        assert!(parse_remux_args(&["only-source"]).is_err());
        assert!(parse_remux_args(&["a", "b", "--to", "flv"]).is_err());
        assert!(parse_remux_args(&["a", "b", "--to", "mp4", "--to", "mkv"]).is_err());
        assert!(parse_remux_args(&["a", "b", "--threads", "0"]).is_err());
        assert!(parse_remux_args(&["a", "b", "--unknown"]).is_err());
    }

    #[test]
    fn rejects_explicit_animation_remux_targets_but_accepts_positional_paths() -> Result<(), String>
    {
        assert!(parse_remux_args(&["clip.mp4", "out.gif", "--to", "gif"]).is_err());
        assert!(parse_remux_args(&["clip.mp4", "out.webp", "--to", "webp"]).is_err());
        let RemuxCommand::Run(gif_target) = parse_remux_args(&["clip.mp4", "out.gif"])? else {
            return Err("expected remux run args".to_owned());
        };
        assert_eq!(gif_target.target, PathBuf::from("out.gif"));
        // A directory target merely named `archive.gif` is valid when `--to`
        // selects a remuxable container; rejection happens after resolution.
        let RemuxCommand::Run(dir_target) =
            parse_remux_args(&["clips", "archive.gif", "--to", "mkv"])?
        else {
            return Err("expected remux run args".to_owned());
        };
        assert_eq!(dir_target.to, Some(VideoContainerFormat::Mkv));
        Ok(())
    }

    #[test]
    fn parses_video_help_and_defaults() -> Result<(), String> {
        assert_eq!(parse_video_args(&["-h"])?, VideoCommand::Help);
        assert_eq!(parse_video_args(&["--help"])?, VideoCommand::Help);
        assert_eq!(
            parse_video_args(&["in.mp4", "out.webm"])?,
            VideoCommand::Run(VideoArgs {
                source: PathBuf::from("in.mp4"),
                target: PathBuf::from("out.webm"),
                to: None,
                video_codec: None,
                audio_codec: None,
                quality: None,
                vbitrate: None,
                abitrate: None,
                asample_rate: None,
                achannels: None,
                threads: None,
                backend: HardwareBackend::Cpu,
                decode_backend: HardwareDecodeBackend::Auto,
            })
        );
        Ok(())
    }

    #[test]
    fn parses_all_video_flags() -> Result<(), String> {
        let VideoCommand::Run(flagged) = parse_video_args(&[
            "in.mp4",
            "out.webm",
            "--to",
            "webm",
            "--vcodec",
            "av1",
            "--acodec",
            "opus",
            "--quality",
            "75",
            "--threads",
            "4",
            "--backend",
            "nvenc",
        ])?
        else {
            return Err("expected video run args".to_owned());
        };
        assert_eq!(flagged.source, PathBuf::from("in.mp4"));
        assert_eq!(flagged.target, PathBuf::from("out.webm"));
        assert_eq!(flagged.to, Some(VideoContainerFormat::Webm));
        assert_eq!(
            flagged.video_codec,
            Some(RequestedVideoCodec::Encode(TargetVideoCodec::Av1))
        );
        assert_eq!(flagged.audio_codec, Some("opus".to_owned()));
        assert_eq!(flagged.quality, VideoQuality::new(75).ok());
        assert_eq!(flagged.threads, NonZeroUsize::new(4));
        assert_eq!(flagged.backend, HardwareBackend::Gpu(GpuBackend::Nvenc));
        assert_eq!(flagged.decode_backend, HardwareDecodeBackend::Auto);
        Ok(())
    }

    #[test]
    fn parses_hwdecode_values_and_rejects_invalid_input() -> Result<(), String> {
        for raw in ["auto", "cpu", "nvdec", "NVDEC", "cuda"] {
            let VideoCommand::Run(flagged) =
                parse_video_args(&["in.mp4", "out.mp4", "--hwdecode", raw])?
            else {
                return Err("expected video run args".to_owned());
            };
            let expected = HardwareDecodeBackend::from_name(raw)
                .ok_or_else(|| format!("test case {raw} must parse"))?;
            assert_eq!(flagged.decode_backend, expected, "--hwdecode {raw}");
        }
        assert!(parse_video_args(&["in.mp4", "out.mp4", "--hwdecode", "magic"]).is_err());
        assert!(
            parse_video_args(&[
                "in.mp4",
                "out.mp4",
                "--hwdecode",
                "cpu",
                "--hwdecode",
                "nvdec"
            ])
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn parses_all_video_formats_and_aliases() -> Result<(), String> {
        for (raw, expected) in [
            ("mp4", VideoContainerFormat::Mp4),
            ("mkv", VideoContainerFormat::Mkv),
            ("webm", VideoContainerFormat::Webm),
            ("mov", VideoContainerFormat::Mov),
            ("avi", VideoContainerFormat::Avi),
            ("ts", VideoContainerFormat::Ts),
            ("m4v", VideoContainerFormat::Mp4),
            ("qt", VideoContainerFormat::Mov),
            ("m2ts", VideoContainerFormat::Ts),
            ("mts", VideoContainerFormat::Ts),
            ("gif", VideoContainerFormat::Gif),
            ("webp", VideoContainerFormat::Webp),
            ("WEBM", VideoContainerFormat::Webm),
            ("GIF", VideoContainerFormat::Gif),
            ("WEBP", VideoContainerFormat::Webp),
        ] {
            let VideoCommand::Run(parsed) = parse_video_args(&["in.mp4", "out.x", "--to", raw])?
            else {
                return Err("expected video run args".to_owned());
            };
            assert_eq!(parsed.to, Some(expected), "video --to {raw}");
        }
        Ok(())
    }

    #[test]
    fn parses_all_video_codec_names_and_copy() -> Result<(), String> {
        for (raw, expected) in [
            ("h264", TargetVideoCodec::H264),
            ("hevc", TargetVideoCodec::Hevc),
            ("h265", TargetVideoCodec::Hevc),
            ("av1", TargetVideoCodec::Av1),
            ("vp9", TargetVideoCodec::Vp9),
            ("gif", TargetVideoCodec::Gif),
            ("webp", TargetVideoCodec::Webp),
        ] {
            let VideoCommand::Run(parsed) =
                parse_video_args(&["in.mp4", "out.x", "--vcodec", raw])?
            else {
                return Err("expected video run args".to_owned());
            };
            assert_eq!(
                parsed.video_codec,
                Some(RequestedVideoCodec::Encode(expected)),
                "video --vcodec {raw}"
            );
        }
        let VideoCommand::Run(copied) = parse_video_args(&["in.mp4", "out.x", "--vcodec", "copy"])?
        else {
            return Err("expected video run args".to_owned());
        };
        assert_eq!(copied.video_codec, Some(RequestedVideoCodec::Copy));

        // The CLI collects raw values; codec meaning is resolved by probe.
        for raw in [
            "aac", "opus", "mp3", "vorbis", "flac", "pcm", "copy", "none",
        ] {
            let VideoCommand::Run(parsed) =
                parse_video_args(&["in.mp4", "out.x", "--acodec", raw])?
            else {
                return Err("expected video run args".to_owned());
            };
            assert_eq!(
                parsed.audio_codec,
                Some(raw.to_owned()),
                "video --acodec {raw}"
            );
        }
        Ok(())
    }

    #[test]
    fn parses_all_video_backends() -> Result<(), String> {
        for (raw, expected) in [
            ("cpu", HardwareBackend::Cpu),
            ("nvenc", HardwareBackend::Gpu(GpuBackend::Nvenc)),
            ("vaapi", HardwareBackend::Gpu(GpuBackend::Vaapi)),
            ("qsv", HardwareBackend::Gpu(GpuBackend::Qsv)),
            ("amf", HardwareBackend::Gpu(GpuBackend::Amf)),
            (
                "videotoolbox",
                HardwareBackend::Gpu(GpuBackend::VideoToolbox),
            ),
            ("vulkan", HardwareBackend::Gpu(GpuBackend::Vulkan)),
        ] {
            let VideoCommand::Run(parsed) =
                parse_video_args(&["in.mp4", "out.x", "--backend", raw])?
            else {
                return Err("expected video run args".to_owned());
            };
            assert_eq!(parsed.backend, expected, "video --backend {raw}");
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_video_args() {
        assert!(parse_video_args(&[]).is_err());
        assert!(parse_video_args(&["only-source"]).is_err());
        assert!(parse_video_args(&["a", "b", "c"]).is_err());
        assert!(parse_video_args(&["a", "b", "--to", "flv"]).is_err());
        assert!(parse_video_args(&["a", "b", "--vcodec", "vp8"]).is_err());
        // Codec-name validity is checked by probe, not by the CLI collector.
        assert!(parse_video_args(&["a", "b", "--acodec", "flac"]).is_ok());
        assert!(parse_video_args(&["a", "b", "--acodec", "pcm"]).is_ok());
        assert!(parse_video_args(&["a", "b", "--backend", "magic"]).is_err());
        assert!(parse_video_args(&["a", "b", "--quality", "101"]).is_err());
        assert!(parse_video_args(&["a", "b", "--quality", "abc"]).is_err());
        assert!(parse_video_args(&["a", "b", "--quality", "-1"]).is_err());
        assert!(parse_video_args(&["a", "b", "--threads", "0"]).is_err());
        assert!(parse_video_args(&["a", "b", "--threads", "many"]).is_err());
        assert!(parse_video_args(&["a", "b", "--unknown"]).is_err());
    }

    #[test]
    fn rejects_duplicate_video_flags() {
        for flag in [
            "--to",
            "--vcodec",
            "--acodec",
            "--quality",
            "--threads",
            "--backend",
        ] {
            let duplicated = match flag {
                "--to" => vec!["a", "b", "--to", "mp4", "--to", "mkv"],
                "--vcodec" => vec!["a", "b", "--vcodec", "h264", "--vcodec", "av1"],
                "--acodec" => vec!["a", "b", "--acodec", "aac", "--acodec", "opus"],
                "--quality" => vec!["a", "b", "--quality", "50", "--quality", "60"],
                "--threads" => vec!["a", "b", "--threads", "2", "--threads", "4"],
                _ => vec!["a", "b", "--backend", "cpu", "--backend", "nvenc"],
            };
            assert!(parse_video_args(&duplicated).is_err(), "duplicate {flag}");
        }
    }

    #[test]
    fn rejects_quality_with_video_copy() {
        assert!(parse_video_args(&["a", "b", "--vcodec", "copy", "--quality", "50"]).is_err());
        assert!(parse_video_args(&["a", "b", "--quality", "50", "--vcodec", "copy"]).is_err());
    }

    #[test]
    fn rejects_missing_video_flag_values() {
        for flag in [
            "--to",
            "--vcodec",
            "--acodec",
            "--quality",
            "--threads",
            "--backend",
        ] {
            assert!(
                parse_video_args(&["a", "b", flag]).is_err(),
                "missing {flag}"
            );
        }
    }

    #[test]
    fn parses_animation_and_dropped_audio_flags() -> Result<(), String> {
        for (to, container, vcodec) in [
            ("gif", VideoContainerFormat::Gif, TargetVideoCodec::Gif),
            ("webp", VideoContainerFormat::Webp, TargetVideoCodec::Webp),
        ] {
            let VideoCommand::Run(parsed) = parse_video_args(&[
                "in.mp4",
                "out.x",
                "--to",
                to,
                "--vcodec",
                to,
                "--acodec",
                "none",
                "--backend",
                "cpu",
            ])?
            else {
                return Err("expected video run args".to_owned());
            };
            assert_eq!(parsed.to, Some(container), "video --to {to}");
            assert_eq!(
                parsed.video_codec,
                Some(RequestedVideoCodec::Encode(vcodec)),
                "video --vcodec {to}"
            );
            assert_eq!(parsed.audio_codec, Some("none".to_owned()));
        }
        Ok(())
    }

    #[test]
    fn accepts_acodec_copy_with_vcodec_copy() -> Result<(), String> {
        let VideoCommand::Run(parsed) =
            parse_video_args(&["in.mp4", "out.mkv", "--vcodec", "copy", "--acodec", "copy"])?
        else {
            return Err("expected video run args".to_owned());
        };
        assert_eq!(parsed.video_codec, Some(RequestedVideoCodec::Copy));
        assert_eq!(parsed.audio_codec, Some("copy".to_owned()));
        assert_eq!(parsed.quality, None);
        Ok(())
    }

    #[test]
    fn parses_rate_control_and_audio_encoding_flags() -> Result<(), String> {
        let VideoCommand::Run(flagged) = parse_video_args(&[
            "in.mp4",
            "out.mp4",
            "--vbitrate",
            "2M",
            "--abitrate",
            "192k",
            "--asample-rate",
            "48000",
            "--achannels",
            "1",
        ])?
        else {
            return Err("expected video run args".to_owned());
        };
        assert_eq!(flagged.vbitrate.map(VideoBitrate::bps), Some(2_000_000));
        assert_eq!(flagged.abitrate.map(AudioBitrate::bps), Some(192_000));
        assert_eq!(flagged.asample_rate.map(AudioSampleRate::hz), Some(48_000));
        assert_eq!(flagged.achannels.map(AudioChannels::count), Some(1));
        assert_eq!(flagged.quality, None);
        Ok(())
    }

    #[test]
    fn rejects_conflicting_rate_control_and_audio_flags() {
        assert!(parse_video_args(&["a", "b", "--quality", "70", "--vbitrate", "1M"]).is_err());
        assert!(parse_video_args(&["a", "b", "--vbitrate", "1M", "--quality", "70"]).is_err());
        assert!(parse_video_args(&["a", "b", "--vcodec", "copy", "--vbitrate", "1M"]).is_err());
        assert!(parse_video_args(&["a", "b", "--vbitrate", "1M", "--vcodec", "copy"]).is_err());

        // Compatibility between --acodec and rate/format flags is decided by
        // the probe boundary, so the CLI must accept the combination here.
        for flag in ["--abitrate", "--asample-rate", "--achannels"] {
            let value = match flag {
                "--abitrate" => "128k",
                "--asample-rate" => "48000",
                _ => "2",
            };
            for codec in ["copy", "none", "flac", "pcm"] {
                assert!(
                    parse_video_args(&["a", "b", "--acodec", codec, flag, value]).is_ok(),
                    "{flag} with --acodec {codec}"
                );
            }
        }
    }

    #[test]
    fn rejects_invalid_rate_control_and_audio_values() {
        for args in [
            vec!["a", "b", "--vbitrate", "0"],
            vec!["a", "b", "--vbitrate", "abc"],
            vec!["a", "b", "--vbitrate", "1G"],
            vec!["a", "b", "--abitrate", "0"],
            vec!["a", "b", "--asample-rate", "0"],
            vec!["a", "b", "--asample-rate", "abc"],
            vec!["a", "b", "--achannels", "0"],
            vec!["a", "b", "--achannels", "abc"],
            vec!["a", "b", "--vbitrate", "1M", "--vbitrate", "2M"],
            vec!["a", "b", "--abitrate", "128k", "--abitrate", "192k"],
            vec![
                "a",
                "b",
                "--asample-rate",
                "48000",
                "--asample-rate",
                "44100",
            ],
            vec!["a", "b", "--achannels", "1", "--achannels", "2"],
        ] {
            assert!(parse_video_args(&args).is_err(), "{args:?}");
        }
    }
}
