use std::num::NonZeroUsize;
use std::path::PathBuf;

use fastconv_domain::{AudioOutputFormat, HardwareBackend};

pub(crate) const AUDIO_HELP: &str = r"fastconv audio - convert audio files

Usage:
  fastconv audio <source> <target> [--to <format>] [--threads <N>] [--backend <name>]

Arguments:
  <source>              Source audio file or directory
  <target>              Target file or directory

Options:
  --to <format>         Output format; required for directory batches, inferred
                        from the target extension for single files: flac, wav,
                        opus, ogg, mp3, aac, m4a, alac, aiff
  --threads <N>         Number of parallel workers
  --backend <name>      cpu (default). Audio runs on CPU only; GPU backends
                        nvenc, vaapi, qsv, amf, videotoolbox, and vulkan are
                        rejected explicitly - no silent CPU fallback.
  -h, --help            Print this help

Supported input formats:
  wav, flac, mp3, ogg (Vorbis), opus, aac (ADTS), m4a, alac, aiff
  Output aliases: wave -> wav, oga -> ogg, aif/aifc -> aiff
  Lossless outputs: wav, flac, alac, aiff
  ALAC writes an M4A container; use a .m4a or .alac target extension.";

pub(crate) fn print_audio_help() {
    println!("{AUDIO_HELP}");
}

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
        "wav" | "wave" => Ok(AudioOutputFormat::Wav),
        "opus" => Ok(AudioOutputFormat::Opus),
        "ogg" | "oga" => Ok(AudioOutputFormat::Ogg),
        "mp3" => Ok(AudioOutputFormat::Mp3),
        "aac" => Ok(AudioOutputFormat::Aac),
        "m4a" => Ok(AudioOutputFormat::M4a),
        "alac" => Ok(AudioOutputFormat::Alac),
        "aiff" | "aif" | "aifc" => Ok(AudioOutputFormat::Aiff),
        other => Err(format!(
            "unknown audio format '{other}' (expected flac, wav, opus, ogg, mp3, aac, m4a, alac, or aiff)"
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
            "expected `fastconv audio <source> <target> [--to <flac|wav|opus|ogg|mp3|aac|m4a|alac|aiff>]`".to_owned(),
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
    fn help_is_dedicated_and_documents_only_implemented_flags() {
        assert!(AUDIO_HELP.contains("Usage:\n  fastconv audio <source> <target>"));
        for flag in ["--to", "--threads", "--backend", "-h, --help"] {
            assert!(AUDIO_HELP.contains(flag), "missing {flag} in audio help");
        }
        for format in [
            "flac", "wav", "opus", "ogg", "mp3", "aac", "m4a", "alac", "aiff",
        ] {
            assert!(
                AUDIO_HELP.contains(format),
                "missing {format} in audio help"
            );
        }
        assert!(AUDIO_HELP.contains("no silent CPU fallback"));
        assert!(AUDIO_HELP.contains("ALAC writes an M4A container"));
        assert!(!AUDIO_HELP.contains("--quality"));
        assert!(!AUDIO_HELP.contains("--lossless"));
        assert!(!AUDIO_HELP.contains("Commands:"));
    }

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
    fn parses_every_target_format_and_alias() -> Result<(), String> {
        let expected = [
            ("flac", AudioOutputFormat::Flac),
            ("wav", AudioOutputFormat::Wav),
            ("wave", AudioOutputFormat::Wav),
            ("opus", AudioOutputFormat::Opus),
            ("ogg", AudioOutputFormat::Ogg),
            ("oga", AudioOutputFormat::Ogg),
            ("mp3", AudioOutputFormat::Mp3),
            ("aac", AudioOutputFormat::Aac),
            ("m4a", AudioOutputFormat::M4a),
            ("alac", AudioOutputFormat::Alac),
            ("aiff", AudioOutputFormat::Aiff),
            ("aif", AudioOutputFormat::Aiff),
            ("aifc", AudioOutputFormat::Aiff),
        ];
        for (name, format) in expected {
            let AudioCommand::Run(parsed) = parse_audio_args(&["a", "b", "--to", name])? else {
                return Err("expected audio run args".to_owned());
            };
            assert_eq!(parsed.to, Some(format), "wrong format for '--to {name}'");
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_audio_args() {
        assert!(parse_audio_args(&["only-source"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--to", "wma"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--to", "flac", "--to", "wav"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--threads", "0"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--backend", "magic"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--unknown"]).is_err());
    }
}
