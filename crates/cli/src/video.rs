use std::num::NonZeroUsize;
use std::path::PathBuf;

use fastconv_domain::VideoContainerFormat;

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
    match value.to_ascii_lowercase().as_str() {
        "mp4" | "mov" => Ok(VideoContainerFormat::Mp4),
        "mkv" => Ok(VideoContainerFormat::Mkv),
        other => Err(format!(
            "unknown video format '{other}' (expected mp4 or mkv)"
        )),
    }
}

pub(crate) fn print_remux_help() {
    println!(
        "Usage: fastconv remux <source> <target> [--to <mp4|mkv>] [--threads <N>]\n\n\
         Lossless container remuxing (MP4/MOV <-> MKV) without re-encoding.\n\n\
         Arguments:\n  \
           <source>              Source video file or directory\n  \
           <target>              Target file or directory\n\n\
         Options:\n  \
           --to <mp4|mkv>        Output container (required for directory batches)\n  \
           --threads <N>         Number of parallel workers\n  \
           -h, --help            Print this help"
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
        return Err("expected `fastconv remux <source> <target> [--to <mp4|mkv>]`".to_owned());
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn rejects_invalid_remux_args() {
        assert!(parse_remux_args(&["only-source"]).is_err());
        assert!(parse_remux_args(&["a", "b", "--to", "avi"]).is_err());
        assert!(parse_remux_args(&["a", "b", "--to", "mp4", "--to", "mkv"]).is_err());
        assert!(parse_remux_args(&["a", "b", "--threads", "0"]).is_err());
        assert!(parse_remux_args(&["a", "b", "--unknown"]).is_err());
    }

    #[test]
    fn rejects_video_to_gif_remux() {
        assert!(parse_remux_args(&["clip.mp4", "out.gif", "--to", "gif"]).is_err());
    }
}
