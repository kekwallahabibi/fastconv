//! `fastconv frames` argument parsing and execution.

use std::num::{NonZeroU32, NonZeroUsize};
use std::path::PathBuf;
use std::time::Instant;

use fastconv_domain::{
    FrameExtractionReport, FrameRange, FrameSelection, HardwareBackend, ImageCompression,
    ImageOutputFormat, u64_to_f64,
};

use crate::{flag_value_with_next, parse_backend, parse_target_format};

pub(crate) const FRAMES_HELP: &str = r"fastconv frames - extract still image frames from video

Usage:
  fastconv frames <source> <output-dir> [--to <format>]
                  [--quality <0..=100> | --lossless] [--fps <N> | --every <N>]
                  [--start <N>] [--end <N>] [--threads <N>] [--backend <auto|cpu>]

Arguments:
  <source>              Video or animated GIF/WebP source file
  <output-dir>          Directory receiving frame_000001.<ext>, frame_000002.<ext>, ...

Options:
  --to <format>         Output image format (default: png): png, jpeg, webp, qoi,
                        bmp, tiff, jxl, avif, gif, ico, pnm, heic
  --quality <0..=100>   Lossy quality (default 80 for lossy formats); mutually
                        exclusive with --lossless
  --lossless            Lossless encoding; mutually exclusive with --quality
  --fps <N>             Extract N frames per second; mutually exclusive with
                        --every. Timestamps are sampled from the first selected
                        frame, skipping gaps without emitting a burst
  --every <N>           Extract every N-th frame (N >= 1); mutually exclusive
                        with --fps
  --start <N>           First zero-based frame index to consider (default 0)
  --end <N>             Last zero-based frame index to consider (inclusive)
  --threads <N>         Number of parallel image encoders
  --backend <name>      Only cpu (default) and auto are accepted. GPU backends
                        (nvenc, vaapi, qsv, amf, videotoolbox, vulkan) are
                        rejected explicitly: no silent CPU fallback
  -h, --help            Print this help";

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FramesArgs {
    source: PathBuf,
    output_dir: PathBuf,
    to: Option<ImageOutputFormat>,
    compression: Option<ImageCompression>,
    selection: FrameSelection,
    start: u64,
    end: Option<u64>,
    threads: Option<NonZeroUsize>,
    backend: HardwareBackend,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum FramesCommand {
    Help,
    Run(FramesArgs),
}

#[derive(Default)]
struct FramesParse<'a> {
    positional: Vec<&'a str>,
    to: Option<ImageOutputFormat>,
    quality: Option<fastconv_domain::ImageQuality>,
    lossless: bool,
    fps: Option<f64>,
    every: Option<NonZeroU32>,
    start: u64,
    start_seen: bool,
    end: Option<u64>,
    threads: Option<NonZeroUsize>,
    backend: Option<HardwareBackend>,
}

fn parse_u64_flag(args: &[&str], index: usize, flag: &str) -> Result<(u64, usize), String> {
    let (raw, next) = flag_value_with_next(args, index, flag)?;
    let value = raw
        .parse::<u64>()
        .map_err(|err| format!("invalid {flag} '{raw}': {err}"))?;
    Ok((value, next))
}

fn parse_frames_backend(value: &str) -> Result<HardwareBackend, String> {
    // `--backend auto` resolves to CPU because extraction has no GPU path.
    if value == "auto" {
        return Ok(HardwareBackend::Cpu);
    }
    let backend = parse_backend(value)?;
    match backend {
        HardwareBackend::Cpu => Ok(backend),
        HardwareBackend::Gpu(backend) => Err(format!(
            "frame extraction does not support backend 'gpu:{backend}'; use --backend cpu (no silent CPU fallback)"
        )),
    }
}

fn apply_selection_flag(
    state: &mut FramesParse<'_>,
    args: &[&str],
    i: usize,
) -> Result<usize, String> {
    let next = match args[i] {
        "--quality" => {
            if state.quality.is_some() {
                return Err(crate::duplicate_flag("--quality"));
            }
            if state.lossless {
                return Err("cannot combine --quality with --lossless".to_owned());
            }
            let (raw, next) = flag_value_with_next(args, i, "--quality")?;
            let value = raw
                .parse::<u8>()
                .map_err(|err| format!("invalid --quality '{raw}': {err}"))?;
            state.quality =
                Some(fastconv_domain::ImageQuality::new(value).map_err(|err| err.to_string())?);
            next
        }
        "--lossless" => {
            if state.lossless {
                return Err(crate::duplicate_flag("--lossless"));
            }
            if state.quality.is_some() {
                return Err("cannot combine --lossless with --quality".to_owned());
            }
            state.lossless = true;
            i + 1
        }
        "--fps" => {
            if state.fps.is_some() {
                return Err(crate::duplicate_flag("--fps"));
            }
            if state.every.is_some() {
                return Err("cannot combine --fps with --every".to_owned());
            }
            let (raw, next) = flag_value_with_next(args, i, "--fps")?;
            let fps = raw
                .parse::<f64>()
                .map_err(|_| format!("invalid --fps '{raw}' (expected a positive number)"))?;
            FrameSelection::target_fps(fps).map_err(|err| err.to_string())?;
            state.fps = Some(fps);
            next
        }
        _ => {
            // `--every`: all other flags are routed to `apply_io_flag`.
            if state.every.is_some() {
                return Err(crate::duplicate_flag("--every"));
            }
            if state.fps.is_some() {
                return Err("cannot combine --every with --fps".to_owned());
            }
            let (raw, next) = flag_value_with_next(args, i, "--every")?;
            let every = raw
                .parse::<NonZeroU32>()
                .map_err(|_| format!("invalid --every '{raw}' (expected a positive integer)"))?;
            state.every = Some(every);
            next
        }
    };
    Ok(next)
}

fn apply_io_flag(state: &mut FramesParse<'_>, args: &[&str], i: usize) -> Result<usize, String> {
    let next = match args[i] {
        "--to" => {
            if state.to.is_some() {
                return Err(crate::duplicate_flag("--to"));
            }
            let (raw, next) = flag_value_with_next(args, i, "--to")?;
            state.to = Some(parse_target_format(raw)?);
            next
        }
        "--start" => {
            if state.start_seen {
                return Err(crate::duplicate_flag("--start"));
            }
            let (value, next) = parse_u64_flag(args, i, "--start")?;
            state.start = value;
            state.start_seen = true;
            next
        }
        "--end" => {
            if state.end.is_some() {
                return Err(crate::duplicate_flag("--end"));
            }
            let (value, next) = parse_u64_flag(args, i, "--end")?;
            state.end = Some(value);
            next
        }
        "--threads" => {
            let (threads, next) = crate::parse_threads(args, i, state.threads.is_some())?;
            state.threads = Some(threads);
            next
        }
        _ => {
            // `--backend`: the only remaining value-taking flag.
            if state.backend.is_some() {
                return Err(crate::duplicate_flag("--backend"));
            }
            let (raw, next) = flag_value_with_next(args, i, "--backend")?;
            state.backend = Some(parse_frames_backend(raw)?);
            next
        }
    };
    Ok(next)
}

/// Parses `fastconv frames` arguments without touching the filesystem.
///
/// # Errors
///
/// Returns a diagnostic for unknown/duplicate flags, invalid values, mutually
/// exclusive options, inverted ranges, and GPU backends.
pub(crate) fn parse_frames_args(args: &[&str]) -> Result<FramesCommand, String> {
    let mut state = FramesParse::default();
    let mut i = 0_usize;
    while i < args.len() {
        match args[i] {
            "-h" | "--help" => return Ok(FramesCommand::Help),
            "--quality" | "--lossless" | "--fps" | "--every" => {
                i = apply_selection_flag(&mut state, args, i)?;
            }
            "--to" | "--start" | "--end" | "--threads" | "--backend" => {
                i = apply_io_flag(&mut state, args, i)?;
            }
            other if other.starts_with('-') => return Err(format!("unknown flag '{other}'")),
            value => {
                state.positional.push(value);
                i += 1;
            }
        }
    }

    let [source, output_dir] = state.positional.as_slice() else {
        return Err("expected `fastconv frames <source> <output-dir> [options]`".to_owned());
    };
    if let Some(end) = state.end
        && state.start > end
    {
        return Err(format!(
            "invalid frame range: --start {} must not exceed --end {end}",
            state.start
        ));
    }
    let selection = match (state.fps, state.every) {
        (Some(fps), None) => FrameSelection::target_fps(fps).map_err(|err| err.to_string())?,
        (None, Some(every)) => FrameSelection::Every(every),
        (None, None) => FrameSelection::All,
        (Some(_), Some(_)) => return Err("cannot combine --fps with --every".to_owned()),
    };
    let compression = if state.lossless {
        Some(ImageCompression::Lossless)
    } else {
        state
            .quality
            .map(|quality| ImageCompression::Lossy { quality })
    };

    Ok(FramesCommand::Run(FramesArgs {
        source: PathBuf::from(source),
        output_dir: PathBuf::from(output_dir),
        to: state.to,
        compression,
        selection,
        start: state.start,
        end: state.end,
        threads: state.threads,
        backend: state.backend.unwrap_or(HardwareBackend::Cpu),
    }))
}

fn print_report(report: &FrameExtractionReport, elapsed: std::time::Duration) {
    let secs = elapsed.as_secs_f64().max(1e-9);
    let fps = u64_to_f64(report.frames_extracted().get()) / secs;
    let mb = u64_to_f64(report.total_output_bytes().get()) / 1_000_000.0;
    println!(
        "extracted {} frame(s) [{}x{} | .{} | {} thread(s)] into {}: {:.2} MB in {:.1} ms ({:.2} fps)",
        report.frames_extracted(),
        report.width(),
        report.height(),
        report.output_format(),
        report.worker_threads(),
        report.output_dir().display(),
        mb,
        secs * 1_000.0,
        fps
    );
}

pub(crate) fn run_frames(args: &FramesArgs) -> Result<(), String> {
    let output_format = args.to.unwrap_or(ImageOutputFormat::Png);
    let range = FrameRange::new(args.start, args.end).map_err(|err| err.to_string())?;
    let plan = fastconv_probe::prepare_frame_extraction(
        &args.source,
        &args.output_dir,
        output_format,
        args.compression,
        args.selection,
        range,
        args.threads,
        args.backend,
    )
    .map_err(|err| err.to_string())?;

    let started = Instant::now();
    let report =
        fastconv_pipeline::execute_frame_extraction(&plan).map_err(|err| err.to_string())?;
    print_report(&report, started.elapsed());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{FramesArgs, FramesCommand, parse_frames_args};
    use fastconv_domain::{
        FrameSelection, HardwareBackend, ImageCompression, ImageOutputFormat, ImageQuality,
        TargetFps,
    };
    use std::num::{NonZeroU32, NonZeroUsize};
    use std::path::PathBuf;

    #[test]
    fn parses_default_frames_arguments() -> Result<(), String> {
        assert_eq!(parse_frames_args(&["--help"])?, FramesCommand::Help);
        let FramesCommand::Run(defaults) = parse_frames_args(&["in.mp4", "frames"])? else {
            return Err("expected Run".to_owned());
        };
        assert_eq!(
            defaults,
            FramesArgs {
                source: PathBuf::from("in.mp4"),
                output_dir: PathBuf::from("frames"),
                to: None,
                compression: None,
                selection: FrameSelection::All,
                start: 0,
                end: None,
                threads: None,
                backend: HardwareBackend::Cpu,
            }
        );
        Ok(())
    }

    #[test]
    fn parses_all_frame_options() -> Result<(), String> {
        let FramesCommand::Run(parsed) = parse_frames_args(&[
            "in.webm",
            "out",
            "--to",
            "jpeg",
            "--quality",
            "90",
            "--fps",
            "12.5",
            "--start",
            "3",
            "--end",
            "9",
            "--threads",
            "4",
            "--backend",
            "auto",
        ])?
        else {
            return Err("expected Run".to_owned());
        };
        assert_eq!(parsed.to, Some(ImageOutputFormat::Jpeg));
        assert_eq!(
            parsed.compression,
            Some(ImageCompression::Lossy {
                quality: ImageQuality::new(90).map_err(|err| err.to_string())?
            })
        );
        assert_eq!(
            parsed.selection,
            FrameSelection::TargetFps(TargetFps::new(12.5).map_err(|err| err.to_string())?)
        );
        assert_eq!((parsed.start, parsed.end), (3, Some(9)));
        assert_eq!(parsed.threads, NonZeroUsize::new(4));
        assert_eq!(parsed.backend, HardwareBackend::Cpu);
        Ok(())
    }

    #[test]
    fn rejects_invalid_and_conflicting_frames_arguments() {
        for args in [
            &["in.mp4"][..],
            &["in.mp4", "out", "--to", "psd"][..],
            &["in.mp4", "out", "--fps", "0"][..],
            &["in.mp4", "out", "--fps", "-1"][..],
            &["in.mp4", "out", "--fps", "nan"][..],
            &["in.mp4", "out", "--fps", "inf"][..],
            &["in.mp4", "out", "--every", "0"][..],
            &["in.mp4", "out", "--start", "10", "--end", "5"][..],
            &["in.mp4", "out", "--quality", "80", "--lossless"][..],
            &["in.mp4", "out", "--lossless", "--quality", "80"][..],
            &["in.mp4", "out", "--fps", "10", "--every", "2"][..],
            &["in.mp4", "out", "--backend", "nvenc"][..],
            &["in.mp4", "out", "--backend", "vulkan"][..],
            &["in.mp4", "out", "--threads", "0"][..],
            &["in.mp4", "out", "--to", "png", "--to", "webp"][..],
            &["in.mp4", "out", "--start", "1", "--start", "2"][..],
            &["in.mp4", "out", "--unknown"][..],
        ] {
            assert!(parse_frames_args(args).is_err(), "must reject {args:?}");
        }
    }

    #[test]
    fn every_selection_is_parsed() -> Result<(), String> {
        let FramesCommand::Run(parsed) =
            parse_frames_args(&["in.mp4", "out", "--every", "5", "--lossless"])?
        else {
            return Err("expected Run".to_owned());
        };
        assert_eq!(
            parsed.selection,
            FrameSelection::Every(NonZeroU32::new(5).ok_or("nz")?)
        );
        assert_eq!(parsed.compression, Some(ImageCompression::Lossless));
        Ok(())
    }
}
