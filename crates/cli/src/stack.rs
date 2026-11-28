//! `fastconv stack` argument parsing and execution.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Instant;

use fastconv_domain::{
    AnimationLoop, GpuBackend, HardwareBackend, StackBackendRequest, StackFps, TargetVideoCodec,
    VideoBitrate, VideoContainerFormat, VideoQuality, VideoRateControl, parse_animation_loop,
    u64_to_f64,
};

use crate::{duplicate_flag, flag_value_with_next, parse_threads};

pub(crate) const STACK_HELP: &str = r"fastconv stack - encode a stack of still images into an animation or video

Usage:
  fastconv stack <image-dir | img1 img2 ...> <target>
                  [--fps <N>] [--to <gif|webp|mp4|mkv|webm|mov|avi|ts>]
                  [--vcodec <h264|hevc|av1|vp9|mjpeg|gif|webp>]
                  [--quality <0..=100> | --vbitrate <rate>]
                  [--loop <infinite|N>] [--threads <N>]
                  [--backend <auto|cpu|nvenc>]

Arguments:
   <image-dir>           Directory of still images, scanned in natural numeric
                         order (frame_2 before frame_10); hidden entries and
                         subdirectories are skipped
   <img1 img2 ...>       Explicit ordered image list; duplicate paths are kept
                         in the order given
   <target>              Output file (must not alias a source image or the
                         source directory)

Options:
   --fps <N>             Frames per second (default 15); finite, strictly
                         positive and bounded to 1e-6..=1_000_000 so frame
                         timings neither round to zero nor overflow; fractional
                         rates are passed to NVENC as num/den
   --to <format>         Output container; inferred from the target extension
                         when omitted, and rejected if it conflicts with that
                         extension. Supports gif, webp, mp4, mkv, webm, mov,
                         avi and ts (with m4v, qt, m2ts and mts aliases)
   --vcodec <codec>      Video codec: h264, hevc (h265), av1, vp9, mjpeg, gif,
                         webp. Defaults per container (mp4/mkv/mov/avi/ts:
                         h264; webm: vp9; gif/webp: their animation codec).
                         'copy' is rejected: a stack has no source bitstream
   --quality <0..=100>   Encoding quality (default 80); mutually exclusive with
                         --vbitrate. MJPEG is quality-driven and accepts it
   --vbitrate <rate>     Target video bitrate (e.g. 500k, 2M); rejected for
                         gif, webp and mjpeg, and mutually exclusive with
                         --quality
   --loop <infinite|N>   infinite (or 0) loops forever; 1..=65535 plays N times
                         (default: infinite). Animation containers only: a
                         loop count on a video container is rejected
   --threads <N>         Number of parallel animation workers
   --backend <name>      auto: NVENC for h264 when an encode-capable GPU is
                         present, CPU otherwise; HEVC always requires NVENC.
                         cpu: native Rust CPU encoders. nvenc: GPU H.264/HEVC
                         (requires --features nvenc and a detected GPU).
                         vaapi, qsv, amf, videotoolbox and vulkan are rejected
                         explicitly - no silent CPU fallback
   -h, --help            Print this help";

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StackArgs {
    pub(crate) sources: Vec<PathBuf>,
    pub(crate) target: PathBuf,
    pub(crate) fps: StackFps,
    pub(crate) to: Option<VideoContainerFormat>,
    pub(crate) vcodec: Option<TargetVideoCodec>,
    pub(crate) rate_control: VideoRateControl,
    pub(crate) loop_count: Option<AnimationLoop>,
    pub(crate) threads: Option<NonZeroUsize>,
    pub(crate) backend: StackBackendRequest,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StackCommand {
    Help,
    Run(StackArgs),
}

#[derive(Default)]
struct StackParse<'a> {
    positional: Vec<&'a str>,
    fps: Option<StackFps>,
    to: Option<VideoContainerFormat>,
    vcodec: Option<TargetVideoCodec>,
    quality: Option<VideoQuality>,
    vbitrate: Option<VideoBitrate>,
    loop_count: Option<AnimationLoop>,
    threads: Option<NonZeroUsize>,
    backend: Option<StackBackendRequest>,
}

fn parse_backend(value: &str) -> Result<StackBackendRequest, String> {
    StackBackendRequest::from_name(value).ok_or_else(|| {
        format!(
            "unknown backend '{value}' (expected auto, cpu, nvenc, vaapi, qsv, amf, videotoolbox, or vulkan)"
        )
    })
}

fn reject_unsupported_gpu(backend: StackBackendRequest) -> Result<StackBackendRequest, String> {
    match backend {
        StackBackendRequest::Explicit(HardwareBackend::Gpu(gpu)) if gpu != GpuBackend::Nvenc => {
            Err(format!(
                "stack encoding does not support backend 'gpu:{gpu}'; use --backend cpu or nvenc (no silent CPU fallback)"
            ))
        }
        other => Ok(other),
    }
}

/// Parses `fastconv stack` arguments without touching the filesystem.
///
/// # Errors
///
/// Returns a diagnostic for fewer than two positionals, unknown/duplicate
/// flags, mutually exclusive options, invalid values, or an unsupported GPU
/// backend.
#[allow(clippy::too_many_lines)] // Flat parser: one explicit match arm per CLI flag.
pub(crate) fn parse_stack_args(args: &[&str]) -> Result<StackCommand, String> {
    let mut state = StackParse::default();
    let mut i = 0_usize;
    while i < args.len() {
        match args[i] {
            "-h" | "--help" => return Ok(StackCommand::Help),
            "--fps" => {
                if state.fps.is_some() {
                    return Err(duplicate_flag("--fps"));
                }
                let (raw, next) = flag_value_with_next(args, i, "--fps")?;
                let fps = raw
                    .parse::<f64>()
                    .map_err(|_| format!("invalid --fps '{raw}' (expected a number)"))?;
                state.fps = Some(StackFps::new(fps).map_err(|err| err.to_string())?);
                i = next;
            }
            "--to" => {
                if state.to.is_some() {
                    return Err(duplicate_flag("--to"));
                }
                let (raw, next) = flag_value_with_next(args, i, "--to")?;
                state.to = Some(
                    VideoContainerFormat::from_extension(raw).ok_or_else(|| {
                        format!("unknown format '{raw}' (expected gif, webp, mp4, mkv, webm, mov, avi, or ts)")
                    })?,
                );
                i = next;
            }
            "--vcodec" => {
                if state.vcodec.is_some() {
                    return Err(duplicate_flag("--vcodec"));
                }
                let (raw, next) = flag_value_with_next(args, i, "--vcodec")?;
                if raw.eq_ignore_ascii_case("copy") {
                    return Err(
                        "--vcodec copy is not supported for image stacks (no source bitstream)"
                            .to_owned(),
                    );
                }
                state.vcodec = Some(TargetVideoCodec::from_name(raw).ok_or_else(|| {
                    format!(
                        "unknown codec '{raw}' (expected h264, hevc, av1, vp9, mjpeg, gif, or webp)"
                    )
                })?);
                i = next;
            }
            "--quality" => {
                if state.quality.is_some() {
                    return Err(duplicate_flag("--quality"));
                }
                if state.vbitrate.is_some() {
                    return Err("cannot combine --quality with --vbitrate".to_owned());
                }
                let (raw, next) = flag_value_with_next(args, i, "--quality")?;
                let quality = raw
                    .parse::<u8>()
                    .map_err(|err| format!("invalid --quality '{raw}': {err}"))?;
                state.quality = Some(VideoQuality::new(quality).map_err(|err| err.to_string())?);
                i = next;
            }
            "--vbitrate" => {
                if state.vbitrate.is_some() {
                    return Err(duplicate_flag("--vbitrate"));
                }
                if state.quality.is_some() {
                    return Err("cannot combine --vbitrate with --quality".to_owned());
                }
                let (raw, next) = flag_value_with_next(args, i, "--vbitrate")?;
                state.vbitrate = Some(VideoBitrate::parse(raw).map_err(|err| err.to_string())?);
                i = next;
            }
            "--loop" => {
                if state.loop_count.is_some() {
                    return Err(duplicate_flag("--loop"));
                }
                let (raw, next) = flag_value_with_next(args, i, "--loop")?;
                state.loop_count = Some(parse_animation_loop(raw).map_err(|err| err.to_string())?);
                i = next;
            }
            "--threads" => {
                let (threads, next) = parse_threads(args, i, state.threads.is_some())?;
                state.threads = Some(threads);
                i = next;
            }
            "--backend" => {
                if state.backend.is_some() {
                    return Err(duplicate_flag("--backend"));
                }
                let (raw, next) = flag_value_with_next(args, i, "--backend")?;
                state.backend = Some(reject_unsupported_gpu(parse_backend(raw)?)?);
                i = next;
            }
            other if other.starts_with('-') => return Err(format!("unknown flag '{other}'")),
            value => {
                state.positional.push(value);
                i += 1;
            }
        }
    }

    if state.positional.len() < 2 {
        return Err(
            "expected `fastconv stack <image-dir | img1 img2 ...> <target> [options]`".to_owned(),
        );
    }
    let target = state
        .positional
        .pop()
        .ok_or_else(|| "missing stack target".to_owned())?;
    let sources = state
        .positional
        .into_iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let rate_control = match (state.quality, state.vbitrate) {
        (Some(quality), _) => VideoRateControl::Quality(quality),
        (None, Some(bitrate)) => VideoRateControl::Bitrate(bitrate),
        (None, None) => VideoRateControl::DEFAULT,
    };

    Ok(StackCommand::Run(StackArgs {
        sources,
        target: PathBuf::from(target),
        fps: state.fps.unwrap_or(StackFps::DEFAULT),
        to: state.to,
        vcodec: state.vcodec,
        rate_control,
        loop_count: state.loop_count,
        threads: state.threads,
        backend: state.backend.unwrap_or(StackBackendRequest::Auto),
    }))
}

fn print_report(report: &fastconv_domain::FrameSequenceReport, elapsed: std::time::Duration) {
    let secs = elapsed.as_secs_f64().max(1e-9);
    let fps = u64_to_f64(report.frames_encoded().get()) / secs;
    let mb = u64_to_f64(report.output_bytes().get()) / 1_000_000.0;
    let path = match (report.backend(), report.hardware_device()) {
        (HardwareBackend::Cpu, _) => "cpu".to_owned(),
        (backend, Some(device)) => format!("{backend} on {device}"),
        (backend, None) => backend.to_string(),
    };
    println!(
        "encoded {} frame(s) [{}x{} | .{} | {} | {} fps | {} | {} thread(s)] into {}: {:.2} MB in {:.1} ms ({:.2} fps)",
        report.frames_encoded(),
        report.dimensions().width(),
        report.dimensions().height(),
        report.target_container().extension(),
        report.video_codec().cli_name(),
        report.fps(),
        path,
        report.worker_threads(),
        report.target_path().display(),
        mb,
        secs * 1_000.0,
        fps
    );
}

pub(crate) fn run_stack(args: &StackArgs) -> Result<(), String> {
    let plan = fastconv_probe::prepare_frame_sequence(
        &args.sources,
        &args.target,
        args.fps,
        args.to,
        args.rate_control,
        args.vcodec,
        args.loop_count,
        args.threads,
        args.backend,
        fastconv_pipeline::nvenc_compiled_in(),
    )
    .map_err(|err| err.to_string())?;

    let started = Instant::now();
    let report = fastconv_pipeline::execute_frame_sequence(&plan).map_err(|err| err.to_string())?;
    print_report(&report, started.elapsed());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{StackCommand, parse_stack_args};
    use fastconv_domain::{
        AnimationLoop, HardwareBackend, StackBackendRequest, StackFps, VideoContainerFormat,
        VideoQuality, VideoRateControl,
    };
    use std::num::{NonZeroU16, NonZeroUsize};
    use std::path::PathBuf;

    #[test]
    fn parses_default_and_flagged_arguments() -> Result<(), String> {
        assert_eq!(parse_stack_args(&["--help"])?, StackCommand::Help);
        let StackCommand::Run(defaults) = parse_stack_args(&["frames", "out.gif"])? else {
            return Err("expected Run".to_owned());
        };
        assert_eq!(defaults.sources, vec![PathBuf::from("frames")]);
        assert_eq!(defaults.target, PathBuf::from("out.gif"));
        assert_eq!(defaults.fps, StackFps::DEFAULT);
        assert_eq!(defaults.to, None);
        assert_eq!(defaults.vcodec, None);
        assert_eq!(defaults.rate_control, VideoRateControl::DEFAULT);
        assert_eq!(defaults.loop_count, None);
        assert_eq!(defaults.backend, StackBackendRequest::Auto);

        let StackCommand::Run(flagged) = parse_stack_args(&[
            "b.png",
            "a.png",
            "b.png",
            "out.mp4",
            "--fps",
            "29.97",
            "--to",
            "mp4",
            "--vcodec",
            "h264",
            "--quality",
            "90",
            "--loop",
            "3",
            "--threads",
            "4",
            "--backend",
            "nvenc",
        ])?
        else {
            return Err("expected Run".to_owned());
        };
        assert_eq!(
            flagged.sources,
            vec![
                PathBuf::from("b.png"),
                PathBuf::from("a.png"),
                PathBuf::from("b.png")
            ]
        );
        assert_eq!(flagged.target, PathBuf::from("out.mp4"));
        assert_eq!(
            flagged.fps,
            StackFps::new(29.97).map_err(|err| err.to_string())?
        );
        assert_eq!(flagged.to, Some(VideoContainerFormat::Mp4));
        assert_eq!(
            flagged.rate_control,
            VideoRateControl::Quality(VideoQuality::new(90).map_err(|err| err.to_string())?)
        );
        assert_eq!(
            flagged.loop_count,
            Some(AnimationLoop::Times(
                NonZeroU16::new(3).unwrap_or(NonZeroU16::MIN)
            ))
        );
        assert_eq!(flagged.threads, NonZeroUsize::new(4));
        assert_eq!(
            flagged.backend,
            StackBackendRequest::Explicit(HardwareBackend::Gpu(fastconv_domain::GpuBackend::Nvenc))
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_stack_arguments() {
        for args in [
            &[][..],
            &["only-source"][..],
            &["a.png", "out.gif", "--fps", "0"][..],
            &["a.png", "out.gif", "--fps", "-1"][..],
            &["a.png", "out.gif", "--fps", "nan"][..],
            &["a.png", "out.gif", "--fps", "inf"][..],
            &["a.png", "out.gif", "--fps", "1e9"][..],
            &["a.png", "out.gif", "--loop", "-1"][..],
            &["a.png", "out.gif", "--loop", "65536"][..],
            &["a.png", "out.gif", "--loop", "many"][..],
            &["a.png", "out.gif", "--to", "bin"][..],
            &["a.png", "out.gif", "--backend", "vaapi"][..],
            &["a.png", "out.gif", "--backend", "vulkan"][..],
            &["a.png", "out.gif", "--threads", "0"][..],
            &["a.png", "out.gif", "--fps", "10", "--fps", "20"][..],
            &["a.png", "out.gif", "--to", "gif", "--to", "webp"][..],
            &["a.png", "out.gif", "--vcodec", "h264", "--vcodec", "av1"][..],
            &["a.png", "out.gif", "--quality", "101"][..],
            &["a.png", "out.gif", "--unknown"][..],
            &["a.png", "out.gif", "--vcodec", "copy"][..],
            &["a.png", "out.mp4", "--quality", "50", "--vbitrate", "1M"][..],
            &["a.png", "out.mp4", "--vbitrate", "1M", "--quality", "50"][..],
            &["a.png", "out.mp4", "--vbitrate", "1M", "--vbitrate", "2M"][..],
        ] {
            assert!(parse_stack_args(args).is_err(), "must reject {args:?}");
        }
    }
}
