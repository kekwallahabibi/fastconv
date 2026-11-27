use std::num::NonZeroUsize;
use std::path::PathBuf;

use fastconv_domain::{
    AudioBitrate, AudioChannels, AudioOutputFormat, AudioQuality, AudioRateControl,
    AudioSampleRate, HardwareBackend,
};

pub(crate) const AUDIO_HELP: &str = r"fastconv audio - convert audio files

Usage:
  fastconv audio <source> <target> [--to <format>] [--bitrate <rate> | --quality <0..=100>]
                 [--sample-rate <hz>] [--channels <1..=8>] [--threads <N>] [--backend <name>]

Arguments:
  <source>              Source audio file or directory
  <target>              Target file or directory

Options:
  --to <format>         Output format; directory batches default to flac, single
                        files infer it from the target extension: flac, wav,
                        opus, ogg, mp3, aac, m4a, alac, aiff
  --bitrate <rate>      Lossy target bitrate: 128k, 192K, 320kbps, bare kbps
                        below 1000 (e.g. 192), or bare bps (e.g. 192000).
                        Ranges: mp3 32-320k, aac/m4a/ogg 32-512k, opus 6-510k.
                        Mutually exclusive with --quality; rejected by
                        lossless outputs (wav, flac, alac, aiff).
  --quality <0..=100>   Quality-based VBR, supported by mp3, ogg and opus.
                        mutually exclusive with --bitrate; aac/m4a have no
                        VBR quality mode (use --bitrate) and lossless outputs
                        reject it.
  --sample-rate <hz>    Resample output to this rate. Opus accepts only 48000;
                        mp3 and aac/m4a accept only their codec rate tables.
  --channels <1..=8>    Output channel count (mp3/opus 1-2, aac/m4a/ogg 1-6
                        excluding 4, lossless 1-8). Remixing is deterministic.
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
    rate_control: AudioRateControl,
    sample_rate: Option<AudioSampleRate>,
    channels: Option<AudioChannels>,
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

/// State for the four encoding flags, grouped so mutual exclusion is checked once.
#[derive(Default)]
struct AudioOptionState {
    rate_control: Option<AudioRateControl>,
    seen_bitrate: bool,
    seen_quality: bool,
    sample_rate: Option<AudioSampleRate>,
    channels: Option<AudioChannels>,
}

impl AudioOptionState {
    /// Returns the next argument index when `flag` is an encoding flag.
    ///
    /// # Errors
    ///
    /// Returns a diagnostic for duplicate flags, missing or invalid values, or combining
    /// `--bitrate` with `--quality`.
    fn parse(&mut self, flag: &str, args: &[&str], index: usize) -> Result<Option<usize>, String> {
        let value = || {
            args.get(index + 1)
                .copied()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag {
            "--bitrate" => {
                if self.seen_bitrate {
                    return Err("duplicate flag '--bitrate'".to_owned());
                }
                if self.seen_quality {
                    return Err("cannot combine --bitrate with --quality".to_owned());
                }
                let raw = value()?;
                let bitrate = AudioBitrate::parse(raw).map_err(|err| err.to_string())?;
                self.rate_control = Some(AudioRateControl::Bitrate(bitrate));
                self.seen_bitrate = true;
            }
            "--quality" => {
                if self.seen_quality {
                    return Err("duplicate flag '--quality'".to_owned());
                }
                if self.seen_bitrate {
                    return Err("cannot combine --quality with --bitrate".to_owned());
                }
                let raw = value()?;
                let parsed = raw
                    .parse::<u8>()
                    .map_err(|err| format!("invalid --quality '{raw}': {err}"))?;
                let quality = AudioQuality::new(parsed).map_err(|err| err.to_string())?;
                self.rate_control = Some(AudioRateControl::Quality(quality));
                self.seen_quality = true;
            }
            "--sample-rate" => {
                if self.sample_rate.is_some() {
                    return Err("duplicate flag '--sample-rate'".to_owned());
                }
                let raw = value()?;
                let hz = raw
                    .parse::<u32>()
                    .map_err(|err| format!("invalid --sample-rate '{raw}': {err}"))?;
                let rate = AudioSampleRate::new(hz).map_err(|err| err.to_string())?;
                self.sample_rate = Some(rate);
            }
            "--channels" => {
                if self.channels.is_some() {
                    return Err("duplicate flag '--channels'".to_owned());
                }
                let raw = value()?;
                let parsed = raw
                    .parse::<u16>()
                    .map_err(|err| format!("invalid --channels '{raw}': {err}"))?;
                let count = AudioChannels::new(parsed).map_err(|err| err.to_string())?;
                self.channels = Some(count);
            }
            _ => return Ok(None),
        }
        Ok(Some(index + 2))
    }
}

/// Parses `fastconv audio` arguments without touching the file system.
///
/// # Errors
///
/// Returns a diagnostic for missing positional arguments, duplicate flags, malformed values,
/// unknown flags, or combining `--bitrate` with `--quality`.
pub(crate) fn parse_audio_args(args: &[&str]) -> Result<AudioCommand, String> {
    let mut positional = Vec::new();
    let mut to = None;
    let mut options = AudioOptionState::default();
    let mut threads = None;
    let mut backend = None;
    let mut index = 0_usize;

    while let Some(&arg) = args.get(index) {
        if let Some(next) = options.parse(arg, args, index)? {
            index = next;
            continue;
        }
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
            "expected `fastconv audio <source> <target> [--to <flac|wav|opus|ogg|mp3|aac|m4a|alac|aiff>] [--options]`".to_owned(),
        );
    };

    Ok(AudioCommand::Run(AudioArgs {
        source: PathBuf::from(*source),
        target: PathBuf::from(*target),
        to,
        rate_control: options.rate_control.unwrap_or(AudioRateControl::Default),
        sample_rate: options.sample_rate,
        channels: options.channels,
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
        fastconv_probe::AudioEncodingRequest {
            rate_control: args.rate_control,
            sample_rate: args.sample_rate,
            channels: args.channels,
        },
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
    fn help_documents_only_implemented_flags() {
        assert!(AUDIO_HELP.contains("Usage:\n  fastconv audio <source> <target>"));
        for flag in [
            "--to",
            "--bitrate <rate>",
            "--quality <0..=100>",
            "--sample-rate <hz>",
            "--channels <1..=8>",
            "--threads",
            "--backend",
            "-h, --help",
        ] {
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
        assert!(AUDIO_HELP.contains("mutually exclusive with --bitrate"));
        assert!(AUDIO_HELP.contains("use --bitrate"));
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
                rate_control: AudioRateControl::Default,
                sample_rate: None,
                channels: None,
                threads: None,
                backend: HardwareBackend::Cpu,
            })
        );

        let AudioCommand::Run(flagged) = parse_audio_args(&[
            "in_dir",
            "out_dir",
            "--to",
            "opus",
            "--bitrate",
            "192k",
            "--sample-rate",
            "48000",
            "--channels",
            "2",
            "--threads",
            "4",
            "--backend",
            "nvenc",
        ])?
        else {
            return Err("expected audio run args".to_owned());
        };
        assert_eq!(flagged.to, Some(AudioOutputFormat::Opus));
        assert_eq!(
            flagged.rate_control,
            AudioRateControl::Bitrate(AudioBitrate::new(192_000).map_err(|err| err.to_string())?)
        );
        assert_eq!(
            flagged.sample_rate,
            Some(AudioSampleRate::new(48_000).map_err(|err| err.to_string())?)
        );
        assert_eq!(
            flagged.channels,
            Some(AudioChannels::new(2).map_err(|err| err.to_string())?)
        );
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
    fn parses_bitrate_and_quality_ladders() -> Result<(), String> {
        for (raw, bps) in [
            ("128k", 128_000),
            ("320kbps", 320_000),
            ("6", 6_000),
            ("128000", 128_000),
        ] {
            let AudioCommand::Run(parsed) = parse_audio_args(&["a", "b", "--bitrate", raw])? else {
                return Err("expected audio run args".to_owned());
            };
            assert_eq!(
                parsed.rate_control,
                AudioRateControl::Bitrate(AudioBitrate::new(bps).map_err(|err| err.to_string())?),
                "--bitrate {raw}"
            );
        }
        let AudioCommand::Run(quality) = parse_audio_args(&["a", "b", "--quality", "85"])? else {
            return Err("expected audio run args".to_owned());
        };
        assert_eq!(
            quality.rate_control,
            AudioRateControl::Quality(AudioQuality::new(85).map_err(|err| err.to_string())?)
        );
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
        assert!(parse_audio_args(&["a", "b", "--bitrate", "128k", "--quality", "80"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--quality", "80", "--bitrate", "128k"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--bitrate", "128k", "--bitrate", "192k"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--quality", "80", "--quality", "90"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--quality", "101"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--quality", "-1"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--bitrate", "abc"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--bitrate"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--quality"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--sample-rate"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--sample-rate", "0"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--sample-rate", "abc"]).is_err());
        assert!(
            parse_audio_args(&["a", "b", "--sample-rate", "48000", "--sample-rate", "44100"])
                .is_err()
        );
        assert!(parse_audio_args(&["a", "b", "--channels"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--channels", "0"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--channels", "9"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--channels", "abc"]).is_err());
        assert!(parse_audio_args(&["a", "b", "--channels", "2", "--channels", "4"]).is_err());
    }
}
