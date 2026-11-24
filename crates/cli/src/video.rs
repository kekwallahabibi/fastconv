use std::num::NonZeroUsize;
use std::path::PathBuf;

use fastconv_domain::{
    HardwareBackend, TargetAudioCodec, TargetVideoCodec, VideoContainerFormat, VideoQuality,
};
use fastconv_probe::{RequestedAudioCodec, RequestedVideoCodec};

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

pub(crate) fn print_remux_help() {
    println!(
        "Usage: fastconv remux <source> <target> [--to <mp4|mkv|webm|mov|avi|ts>] [--threads <N>]\n\n\
         Lossless container remuxing across MP4/MOV, MKV, WebM, AVI, and TS without re-encoding.\n\n\
         Arguments:\n  \
           <source>              Source video file or directory\n  \
           <target>              Target file or directory\n\n\
         Options:\n  \
           --to <mp4|mkv|webm|mov|avi|ts>  Output container (required for directory batches)\n  \
           --threads <N>         Number of parallel workers\n  \
           -h, --help            Print this help\n\n\
         Supported containers: mp4 (m4v), mkv, webm, mov (qt), avi, ts (m2ts, mts)"
    );
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
                to = Some(parse_target_format(raw)?);
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
    audio_codec: Option<RequestedAudioCodec>,
    quality: Option<VideoQuality>,
    threads: Option<NonZeroUsize>,
    backend: HardwareBackend,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VideoCommand {
    Help,
    Run(VideoArgs),
}

pub(crate) fn print_video_help() {
    println!(
        "Usage: fastconv video <source> <target> [--to <mp4|mkv|webm|mov|avi|ts>] [--vcodec <h264|hevc|h265|av1|vp9|copy>] [--acodec <aac|opus|mp3|copy>] [--quality <0..=100>] [--threads <N>] [--backend <cpu|nvenc>]\n\n\
         Transcode audio/video tracks while converting between MP4/MOV, MKV, WebM, AVI, and TS containers.\n\n\
         Backends: cpu runs every codec and --vcodec copy; nvenc encodes H.264/HEVC only.\n\n\
         Arguments:\n  \
           <source>              Source video file or directory\n  \
           <target>              Target file or directory\n\n\
         Options:\n  \
           --to <mp4|mkv|webm|mov|avi|ts>  Output container (required for directory batches)\n  \
           --vcodec <codec>      Video codec: h264, hevc (h265), av1, vp9, or copy\n  \
           --acodec <codec>      Audio codec: aac, opus, mp3, or copy\n  \
           --quality <0..=100>   Encoding quality (cannot be combined with --vcodec copy)\n  \
           --threads <N>         Number of parallel workers\n  \
           --backend <name>      cpu (default) or nvenc (H.264/HEVC encoding only)\n  \
           -h, --help            Print this help\n\n\
         Container aliases: m4v -> mp4, qt -> mov, m2ts/mts -> ts"
    );
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
        format!("unknown video codec '{raw}' (expected h264, hevc (h265), av1, vp9, or copy)")
    })?;
    Ok(RequestedVideoCodec::Encode(codec))
}

fn parse_audio_codec(raw: &str) -> Result<RequestedAudioCodec, String> {
    if raw == "copy" {
        return Ok(RequestedAudioCodec::Copy);
    }
    let codec = TargetAudioCodec::from_name(raw)
        .ok_or_else(|| format!("unknown audio codec '{raw}' (expected aac, opus, mp3, or copy)"))?;
    Ok(RequestedAudioCodec::Encode(codec))
}

/// Parses `fastconv video` arguments without touching the file system.
///
/// # Errors
///
/// Returns a diagnostic for missing positional arguments, duplicate flags, malformed values,
/// unknown codecs/backends, or combinations with no implementation.
pub(crate) fn parse_video_args(args: &[&str]) -> Result<VideoCommand, String> {
    let mut positional = Vec::new();
    let mut to = None;
    let mut video_codec = None;
    let mut audio_codec = None;
    let mut quality = None;
    let mut threads = None;
    let mut backend = None;
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
                if raw == "copy" && quality.is_some() {
                    return Err("cannot combine --quality with --vcodec copy".to_owned());
                }
                video_codec = Some(parse_video_codec(raw)?);
                index += 2;
            }
            "--acodec" => {
                if audio_codec.is_some() {
                    return Err("duplicate flag '--acodec'".to_owned());
                }
                audio_codec = Some(parse_audio_codec(flag_value(args, index, "--acodec")?)?);
                index += 2;
            }
            "--quality" => {
                if quality.is_some() {
                    return Err("duplicate flag '--quality'".to_owned());
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
            flag if flag.starts_with('-') => return Err(format!("unknown flag '{flag}'")),
            value => {
                positional.push(value);
                index += 1;
            }
        }
    }

    let [source, target] = positional.as_slice() else {
        return Err(
            "expected `fastconv video <source> <target> [--to <mp4|mkv|webm|mov|avi|ts>] [options]`"
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
        threads,
        backend: backend.unwrap_or(HardwareBackend::Cpu),
    }))
}

/// Orchestrates probe → domain → pipeline for a single video file or a video batch.
///
/// # Errors
///
/// Returns a diagnostic if probing, planning, decoding, encoding, or muxing fails. Unsupported
/// codec/container pairs and backend combinations are rejected before any file is written.
pub(crate) fn run_video(args: &VideoArgs) -> Result<(), String> {
    let plan = fastconv_probe::prepare_video_transcode_batch(
        &args.source,
        &args.target,
        args.to,
        args.video_codec,
        args.audio_codec,
        args.quality.unwrap_or(VideoQuality::DEFAULT),
        args.backend,
        args.threads,
    )
    .map_err(|err| err.to_string())?;

    let report =
        fastconv_pipeline::execute_video_transcode_batch(&plan).map_err(|err| err.to_string())?;

    let backend_desc = match report.hardware_device() {
        Some(device) => format!("{} on {device}", report.backend()),
        None => report.backend().to_string(),
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
    fn rejects_video_to_gif_remux() {
        assert!(parse_remux_args(&["clip.mp4", "out.gif", "--to", "gif"]).is_err());
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
                threads: None,
                backend: HardwareBackend::Cpu,
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
        assert_eq!(
            flagged.audio_codec,
            Some(RequestedAudioCodec::Encode(TargetAudioCodec::Opus))
        );
        assert_eq!(flagged.quality, VideoQuality::new(75).ok());
        assert_eq!(flagged.threads, NonZeroUsize::new(4));
        assert_eq!(flagged.backend, HardwareBackend::Gpu(GpuBackend::Nvenc));
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
            ("WEBM", VideoContainerFormat::Webm),
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

        for (raw, expected) in [
            ("aac", TargetAudioCodec::Aac),
            ("opus", TargetAudioCodec::Opus),
            ("mp3", TargetAudioCodec::Mp3),
        ] {
            let VideoCommand::Run(parsed) =
                parse_video_args(&["in.mp4", "out.x", "--acodec", raw])?
            else {
                return Err("expected video run args".to_owned());
            };
            assert_eq!(
                parsed.audio_codec,
                Some(RequestedAudioCodec::Encode(expected)),
                "video --acodec {raw}"
            );
        }
        let VideoCommand::Run(copied) = parse_video_args(&["in.mp4", "out.x", "--acodec", "copy"])?
        else {
            return Err("expected video run args".to_owned());
        };
        assert_eq!(copied.audio_codec, Some(RequestedAudioCodec::Copy));
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
        assert!(parse_video_args(&["a", "b", "--acodec", "flac"]).is_err());
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
    fn accepts_acodec_copy_with_vcodec_copy() -> Result<(), String> {
        let VideoCommand::Run(parsed) =
            parse_video_args(&["in.mp4", "out.mkv", "--vcodec", "copy", "--acodec", "copy"])?
        else {
            return Err("expected video run args".to_owned());
        };
        assert_eq!(parsed.video_codec, Some(RequestedVideoCodec::Copy));
        assert_eq!(parsed.audio_codec, Some(RequestedAudioCodec::Copy));
        assert_eq!(parsed.quality, None);
        Ok(())
    }
}
