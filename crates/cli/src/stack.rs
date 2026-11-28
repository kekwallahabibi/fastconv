//! `fastconv stack` argument parsing and execution.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Instant;

use fastconv_domain::{
    AnimationLoop, HardwareBackend, StackFps, StackTargetFormat, VideoQuality,
    parse_animation_loop, u64_to_f64,
};

use crate::{duplicate_flag, flag_value_with_next, parse_backend, parse_threads};

pub(crate) const STACK_HELP: &str = r"fastconv stack - encode a stack of still images into an animation

Usage:
  fastconv stack <image-dir | img1 img2 ...> <target>
                  [--fps <N>] [--to <gif|webp>] [--quality <0..=100>]
                  [--loop <infinite|N>] [--threads <N>] [--backend <auto|cpu>]

Arguments:
   <image-dir>           Directory of still images, scanned in natural numeric
                         order (frame_2 before frame_10); hidden entries and
                         subdirectories are skipped
   <img1 img2 ...>       Explicit ordered image list; duplicate paths are kept
                         in the order given
   <target>              Animated .gif or .webp output file (must not alias a
                         source image or the source directory)

Options:
   --fps <N>             Frames per second (default 15); finite, strictly
                         positive and bounded to 1e-6..=1_000_000 so frame
                         timings neither round to zero nor overflow
   --to <gif|webp>       Output format; inferred from the target extension when
                         omitted, and rejected if it conflicts with that extension
   --quality <0..=100>   Encoding quality (default 80; lossy quality for WebP,
                         quantizer speed/effort for GIF)
   --loop <infinite|N>   infinite (or 0) loops forever; 1..=65535 plays N times
                         (default: infinite)
   --threads <N>         Number of parallel animation workers
   --backend <auto|cpu>  Only cpu/auto are accepted. GPU backends (nvenc, vaapi,
                         qsv, amf, videotoolbox, vulkan) are rejected explicitly:
                         no silent CPU fallback
   -h, --help            Print this help";

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StackArgs {
    pub(crate) sources: Vec<PathBuf>,
    pub(crate) target: PathBuf,
    pub(crate) fps: StackFps,
    pub(crate) to: Option<StackTargetFormat>,
    pub(crate) quality: VideoQuality,
    pub(crate) loop_count: AnimationLoop,
    pub(crate) threads: Option<NonZeroUsize>,
    pub(crate) backend: HardwareBackend,
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
    to: Option<StackTargetFormat>,
    quality: Option<VideoQuality>,
    loop_count: Option<AnimationLoop>,
    threads: Option<NonZeroUsize>,
    backend: Option<HardwareBackend>,
}

fn parse_stack_backend(value: &str) -> Result<HardwareBackend, String> {
    if value == "auto" {
        return Ok(HardwareBackend::Cpu);
    }
    let backend = parse_backend(value)?;
    match backend {
        HardwareBackend::Cpu => Ok(backend),
        HardwareBackend::Gpu(backend) => Err(format!(
            "stack encoding does not support backend 'gpu:{backend}'; use --backend cpu (no silent CPU fallback)"
        )),
    }
}

/// Parses `fastconv stack` arguments without touching the filesystem.
///
/// # Errors
///
/// Returns a diagnostic for fewer than two positionals, unknown/duplicate
/// flags, invalid values, or any requested GPU backend.
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
                    StackTargetFormat::from_extension(raw)
                        .ok_or_else(|| format!("unknown format '{raw}' (expected gif or webp)"))?,
                );
                i = next;
            }
            "--quality" => {
                if state.quality.is_some() {
                    return Err(duplicate_flag("--quality"));
                }
                let (raw, next) = flag_value_with_next(args, i, "--quality")?;
                let quality = raw
                    .parse::<u8>()
                    .map_err(|err| format!("invalid --quality '{raw}': {err}"))?;
                state.quality = Some(VideoQuality::new(quality).map_err(|err| err.to_string())?);
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
                state.backend = Some(parse_stack_backend(raw)?);
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

    Ok(StackCommand::Run(StackArgs {
        sources,
        target: PathBuf::from(target),
        fps: state.fps.unwrap_or(StackFps::DEFAULT),
        to: state.to,
        quality: state.quality.unwrap_or(VideoQuality::DEFAULT),
        loop_count: state.loop_count.unwrap_or(AnimationLoop::Infinite),
        threads: state.threads,
        backend: state.backend.unwrap_or(HardwareBackend::Cpu),
    }))
}

fn print_report(report: &fastconv_domain::FrameSequenceReport, elapsed: std::time::Duration) {
    let secs = elapsed.as_secs_f64().max(1e-9);
    let fps = u64_to_f64(report.frames_encoded().get()) / secs;
    let mb = u64_to_f64(report.output_bytes().get()) / 1_000_000.0;
    println!(
        "encoded {} frame(s) [{}x{} | .{} | {} fps | {} thread(s)] into {}: {:.2} MB in {:.1} ms ({:.2} fps)",
        report.frames_encoded(),
        report.dimensions().width(),
        report.dimensions().height(),
        report.target_format().extension(),
        report.fps(),
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
        args.quality,
        args.loop_count,
        args.threads,
        args.backend,
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
        AnimationLoop, HardwareBackend, StackFps, StackTargetFormat, VideoQuality,
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
        assert_eq!(defaults.quality, VideoQuality::DEFAULT);
        assert_eq!(defaults.loop_count, AnimationLoop::Infinite);
        assert_eq!(defaults.backend, HardwareBackend::Cpu);

        let StackCommand::Run(flagged) = parse_stack_args(&[
            "b.png",
            "a.png",
            "b.png",
            "out.webp",
            "--fps",
            "24",
            "--to",
            "webp",
            "--quality",
            "90",
            "--loop",
            "3",
            "--threads",
            "4",
            "--backend",
            "cpu",
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
        assert_eq!(flagged.target, PathBuf::from("out.webp"));
        assert_eq!(
            flagged.fps,
            StackFps::new(24.0).map_err(|err| err.to_string())?
        );
        assert_eq!(flagged.to, Some(StackTargetFormat::Webp));
        assert_eq!(
            flagged.quality,
            VideoQuality::new(90).map_err(|err| err.to_string())?
        );
        assert_eq!(
            flagged.loop_count,
            AnimationLoop::Times(NonZeroU16::new(3).unwrap_or(NonZeroU16::MIN))
        );
        assert_eq!(flagged.threads, NonZeroUsize::new(4));
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
            &["a.png", "out.gif", "--to", "mp4"][..],
            &["a.png", "out.gif", "--backend", "nvenc"][..],
            &["a.png", "out.gif", "--backend", "vulkan"][..],
            &["a.png", "out.gif", "--threads", "0"][..],
            &["a.png", "out.gif", "--fps", "10", "--fps", "20"][..],
            &["a.png", "out.gif", "--to", "gif", "--to", "webp"][..],
            &["a.png", "out.gif", "--quality", "101"][..],
            &["a.png", "out.gif", "--unknown"][..],
        ] {
            assert!(parse_stack_args(args).is_err(), "must reject {args:?}");
        }
    }
}
