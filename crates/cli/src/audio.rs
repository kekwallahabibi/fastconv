use std::num::NonZeroUsize;
use std::path::PathBuf;

use fastconv_domain::{AudioOutputFormat, HardwareBackend};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AudioArgs {
    source: PathBuf,
    target: PathBuf,
    to: Option<AudioOutputFormat>,
    threads: Option<NonZeroUsize>,
    backend: HardwareBackend,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AudioCommand {
    Help,
    Run(AudioArgs),
}

fn parse_target_format(value: &str) -> Result<AudioOutputFormat, String> {
    match value {
        "flac" => Ok(AudioOutputFormat::Flac),
        "wav" => Ok(AudioOutputFormat::Wav),
        "opus" => Ok(AudioOutputFormat::Opus),
        other => Err(format!(
            "unknown audio format '{other}' (expected flac, wav, or opus)"
        )),
    }
}

/// Parses `fastconv audio` arguments without touching the file system.
///
/// # Errors
///
/// Returns a diagnostic for missing positional arguments, duplicate flags, malformed values,
/// or unknown flags.
pub(crate) fn parse_audio_args(args: &[&str]) -> Result<AudioCommand, String> {
    let mut positional = Vec::new();
    let mut to = None;
    let mut threads = None;
    let mut backend = None;
    let mut index = 0_usize;

    while let Some(&arg) = args.get(index) {
        match arg {
            "-h" | "--help" => return Ok(AudioCommand::Help),
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
            "--backend" => {
                if backend.is_some() {
                    return Err("duplicate flag '--backend'".to_owned());
                }
                let raw = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --backend".to_owned())?;
                backend = Some(crate::parse_backend(raw)?);
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
            "expected `fastconv audio <source> <target> [--to <flac|wav|opus>]`".to_owned(),
        );
    };

    Ok(AudioCommand::Run(AudioArgs {
        source: PathBuf::from(*source),
        target: PathBuf::from(*target),
        to,
        threads,
        backend: backend.unwrap_or(HardwareBackend::Cpu),
    }))
}

/// Orchestrates probe → domain → pipeline for one audio file or a mixed audio batch.
///
/// # Errors
///
/// Returns a diagnostic if probing, planning, decoding, or encoding fails. GPU backends are
/// rejected by the domain layer without a silent CPU fallback.
pub(crate) fn run_audio(args: &AudioArgs) -> Result<(), String> {
    let plan = fastconv_probe::prepare_audio_batch(
        &args.source,
        &args.target,
        args.to,
        args.threads,
        args.backend,
    )
    .map_err(|err| err.to_string())?;

    let report = fastconv_pipeline::execute_audio_batch(&plan).map_err(|err| err.to_string())?;
    println!(
        "converted {} audio file(s) [{} | {} thread(s)]: {:.2} MB -> {:.2} MB in {:.1} ms ({:.2} MB/s)",
        report.files_converted(),
        plan.mode(),
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
    use fastconv_domain::GpuBackend;

    #[test]
    fn parses_default_and_flagged_args() -> Result<(), String> {
        assert_eq!(parse_audio_args(&["--help"])?, AudioCommand::Help);
        let defaults = parse_audio_args(&["in.wav", "out.flac"])?;
        assert_eq!(
            defaults,
            AudioCommand::Run(AudioArgs {
                source: PathBuf::from("in.wav"),
                target: PathBuf::from("out.flac"),
                to: None,
                threads: None,
                backend: HardwareBackend::Cpu,
            })
        );

        let AudioCommand::Run(flagged) = parse_audio_args(&[
            "in_dir",
            "out_dir",
            "--to",
            "opus",
            "--threads",
            "4",
            "--backend",
            "nvenc",
        ])?
        else {
            return Err("expected audio run args".to_owned());
        };
        assert_eq!(flagged.to, Some(AudioOutputFormat::Opus));
        assert_eq!(flagged.threads, NonZeroUsize::new(4));
        assert_eq!(flagged.backend, HardwareBackend::Gpu(GpuBackend::Nvenc));
        Ok(())
    }

    #[test]
    fn rejects_invalid_audio_args() {
        assert!(parse_audio_args(&["only-source"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--to", "mp3"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--to", "flac", "--to", "wav"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--threads", "0"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--backend", "magic"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--unknown"]).is_err());
    }
}
