mod audio;
mod video;

use std::env;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use fastconv_domain::{
    GpuBackend, HardwareBackend, ImageCompression, ImageOutputFormat, ImageQuality,
};

const GLOBAL_HELP: &str = r"Usage:
  fastconv probe
  fastconv convert <source> <target> [--to <format>] [--quality <0..=100> | --lossless] [--threads <N>] [--backend <name>]
  fastconv audio <source> <target> [--to <format>] [--threads <N>] [--backend <name>]
  fastconv remux <source> <target> [--to <mp4|mkv|webm|mov|avi|ts>] [--threads <N>]
  fastconv video <source> <target> [--to <format>] [--vcodec <codec>]
                 [--acodec <codec>] [--quality <0..=100>] [--threads <N>]
                 [--backend <name>] [--hwdecode <auto|cpu|nvdec>]
  fastconv help [command]
  fastconv -V, --version
  fastconv -h, --help

Commands:
  probe    Inspect CPU, GPU adapters, and GPU media backends
  convert  Convert still images (lossy/lossless; CPU or NVENC HEIC)
  audio    Convert audio files (CPU encoders only)
  remux    Losslessly remux video containers (no re-encoding)
  video    Transcode audio/video tracks between containers
  help     Print this help or detailed help for one command

Options:
  -h, --help     Print this help
  -V, --version  Print version

Run `fastconv <command> --help` or `fastconv help <command>` for detailed help.";

const PROBE_HELP: &str = r"fastconv probe - inspect detected host hardware

Usage:
  fastconv probe
  fastconv probe -h | --help

Prints the hardware available to conversion pipelines:

  CPU        model, architecture, physical cores / logical threads, SIMD level,
             and total memory
  GPU        detected gpu adapters (name, vendor, device id, and kernel driver)
  GPU media  detected media backends with capability and device name: nvenc,
             vaapi, qsv, amf, videotoolbox, vulkan

Options:
  -h, --help  Print this help";

const CONVERT_HELP: &str = r"fastconv convert - convert still images

Usage:
  fastconv convert <source> <target> [--to <format>] [--quality <0..=100> | --lossless] [--threads <N>] [--backend <name>]

Arguments:
  <source>              Source image file or directory
  <target>              Target file or directory

Options:
  --to <format>         Output format; directory batches default to webp, single
                        files infer it from the target extension: webp, heic,
                        jpeg, png, bmp, qoi, tiff, jxl, avif, gif, ico, pnm
  --quality <0..=100>   Lossy encoding quality (default 80); mutually exclusive
                        with --lossless. Supported: webp, jpeg, jxl, avif, heic.
                        Rejected: png, bmp, qoi, tiff, ico, pnm (lossless-only)
                        and gif.
  --lossless            Lossless encoding; mutually exclusive with --quality.
                        Supported: webp, jxl, and the lossless-only formats
                        png, bmp, qoi, tiff, ico, pnm (lossless by default).
                        Rejected: jpeg, avif, heic, gif.
  --threads <N>         Number of parallel workers
  --backend <name>      cpu (default) or nvenc (HEIC only; CPU has no native
                        HEIC encoder). vaapi, qsv, amf, videotoolbox, and vulkan
                        are rejected explicitly - no silent CPU fallback.
  -h, --help            Print this help

Supported input formats:
  png, jpeg, webp, bmp, qoi, tiff, jxl, avif, gif, ico, pnm
  Aliases: jpg -> jpeg, tif -> tiff, ppm/pgm/pbm/pam -> pnm
  GIF uses a fixed 256-color palette quantization and accepts neither --quality
  nor --lossless.";

fn print_global_help() {
    println!("fastconv {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("{GLOBAL_HELP}");
}

fn print_probe_help() {
    println!("{PROBE_HELP}");
}

fn print_convert_help() {
    println!("{CONVERT_HELP}");
}

#[must_use]
fn help_text(topic: &str) -> Option<&'static str> {
    match topic {
        "probe" => Some(PROBE_HELP),
        "convert" => Some(CONVERT_HELP),
        "audio" => Some(audio::AUDIO_HELP),
        "remux" => Some(video::REMUX_HELP),
        "video" => Some(video::VIDEO_HELP),
        _ => None,
    }
}

/// Prints the dedicated help for one command and reports whether the topic exists.
fn print_help_topic(topic: &str) -> bool {
    match help_text(topic) {
        Some(text) => {
            println!("{text}");
            true
        }
        None => false,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConvertArgs {
    source: PathBuf,
    target: PathBuf,
    compression: Option<ImageCompression>,
    to: Option<ImageOutputFormat>,
    threads: Option<NonZeroUsize>,
    backend: HardwareBackend,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ConvertCommand {
    Help,
    Run(ConvertArgs),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeCommand {
    Help,
    Run,
}

/// Parses `fastconv probe` arguments without probing hardware.
///
/// # Errors
///
/// Returns a diagnostic for positional arguments or unknown flags.
pub(crate) fn parse_probe_args(args: &[&str]) -> Result<ProbeCommand, String> {
    match args {
        [] => Ok(ProbeCommand::Run),
        ["-h" | "--help"] => Ok(ProbeCommand::Help),
        ["-h" | "--help", ..] => Err("expected `fastconv probe`".to_owned()),
        [flag, ..] if flag.starts_with('-') => Err(format!("unknown flag '{flag}'")),
        _ => Err("expected `fastconv probe`".to_owned()),
    }
}

pub(crate) fn parse_backend(value: &str) -> Result<HardwareBackend, String> {
    match value {
        "cpu" => Ok(HardwareBackend::Cpu),
        "nvenc" => Ok(HardwareBackend::Gpu(GpuBackend::Nvenc)),
        "vaapi" => Ok(HardwareBackend::Gpu(GpuBackend::Vaapi)),
        "qsv" => Ok(HardwareBackend::Gpu(GpuBackend::Qsv)),
        "amf" => Ok(HardwareBackend::Gpu(GpuBackend::Amf)),
        "videotoolbox" => Ok(HardwareBackend::Gpu(GpuBackend::VideoToolbox)),
        "vulkan" => Ok(HardwareBackend::Gpu(GpuBackend::Vulkan)),
        other => Err(format!(
            "unknown backend '{other}' (expected cpu, nvenc, vaapi, qsv, amf, videotoolbox, or vulkan)"
        )),
    }
}

fn parse_target_format(value: &str) -> Result<ImageOutputFormat, String> {
    match value {
        "webp" => Ok(ImageOutputFormat::Webp),
        "heic" => Ok(ImageOutputFormat::Heic),
        "jpeg" | "jpg" => Ok(ImageOutputFormat::Jpeg),
        "png" => Ok(ImageOutputFormat::Png),
        "bmp" => Ok(ImageOutputFormat::Bmp),
        "qoi" => Ok(ImageOutputFormat::Qoi),
        "tiff" | "tif" => Ok(ImageOutputFormat::Tiff),
        "jxl" => Ok(ImageOutputFormat::Jxl),
        "avif" => Ok(ImageOutputFormat::Avif),
        "gif" => Ok(ImageOutputFormat::Gif),
        "ico" => Ok(ImageOutputFormat::Ico),
        "pnm" | "ppm" | "pgm" | "pbm" | "pam" => Ok(ImageOutputFormat::Pnm),
        other => Err(format!(
            "unknown format '{other}' (expected webp, heic, jpeg, png, bmp, qoi, tiff, jxl, avif, gif, ico, or pnm)"
        )),
    }
}

fn parse_convert_args(args: &[&str]) -> Result<ConvertCommand, String> {
    let mut positional: Vec<&str> = Vec::new();
    let mut quality_arg: Option<ImageQuality> = None;
    let mut lossless = false;
    let mut to: Option<ImageOutputFormat> = None;
    let mut threads: Option<NonZeroUsize> = None;
    let mut backend: Option<HardwareBackend> = None;

    let mut i = 0_usize;
    while let Some(&arg) = args.get(i) {
        match arg {
            "-h" | "--help" => return Ok(ConvertCommand::Help),
            "--lossless" => {
                if lossless {
                    return Err("duplicate flag '--lossless'".to_owned());
                }
                if quality_arg.is_some() {
                    return Err("cannot combine --lossless with --quality".to_owned());
                }
                lossless = true;
                i += 1;
            }
            "--quality" => {
                if quality_arg.is_some() {
                    return Err("duplicate flag '--quality'".to_owned());
                }
                if lossless {
                    return Err("cannot combine --quality with --lossless".to_owned());
                }
                let raw = args
                    .get(i + 1)
                    .copied()
                    .ok_or_else(|| "missing value for --quality".to_owned())?;
                let parsed = raw
                    .parse::<u8>()
                    .map_err(|err| format!("invalid --quality '{raw}': {err}"))?;
                quality_arg = Some(ImageQuality::new(parsed).map_err(|err| err.to_string())?);
                i += 2;
            }
            "--to" => {
                if to.is_some() {
                    return Err("duplicate flag '--to'".to_owned());
                }
                let raw = args
                    .get(i + 1)
                    .copied()
                    .ok_or_else(|| "missing value for --to".to_owned())?;
                to = Some(parse_target_format(raw)?);
                i += 2;
            }
            "--threads" => {
                if threads.is_some() {
                    return Err("duplicate flag '--threads'".to_owned());
                }
                let raw = args
                    .get(i + 1)
                    .copied()
                    .ok_or_else(|| "missing value for --threads".to_owned())?;
                let parsed = raw
                    .parse::<NonZeroUsize>()
                    .map_err(|err| format!("invalid --threads '{raw}': {err}"))?;
                threads = Some(parsed);
                i += 2;
            }
            "--backend" => {
                if backend.is_some() {
                    return Err("duplicate flag '--backend'".to_owned());
                }
                let raw = args
                    .get(i + 1)
                    .copied()
                    .ok_or_else(|| "missing value for --backend".to_owned())?;
                backend = Some(parse_backend(raw)?);
                i += 2;
            }
            flag if flag.starts_with('-') => {
                return Err(format!("unknown flag '{flag}'"));
            }
            value => {
                positional.push(value);
                i += 1;
            }
        }
    }

    let [source, target] = positional.as_slice() else {
        return Err("expected `fastconv convert <source> <target> [options]`".to_owned());
    };

    let compression = if lossless {
        Some(ImageCompression::Lossless)
    } else {
        quality_arg.map(|quality| ImageCompression::Lossy { quality })
    };

    Ok(ConvertCommand::Run(ConvertArgs {
        source: PathBuf::from(source),
        target: PathBuf::from(target),
        compression,
        to,
        threads,
        backend: backend.unwrap_or(HardwareBackend::Cpu),
    }))
}

fn run_probe() -> Result<(), String> {
    let hardware = fastconv_probe::detect_host_hardware().map_err(|err| err.to_string())?;
    let cpu = hardware.cpu();

    println!("CPU:");
    println!("  model        : {}", cpu.model_name());
    println!("  arch         : {}", cpu.architecture());
    println!(
        "  cores/threads: {} physical / {} logical",
        cpu.physical_cores(),
        cpu.logical_threads()
    );
    println!("  simd         : {}", cpu.simd_level());
    if let Some(mem_bytes) = cpu.total_memory_bytes() {
        let mib = mem_bytes.get() / (1024 * 1024);
        println!("  memory       : {mib} MiB");
    } else {
        println!("  memory       : unavailable");
    }

    println!("GPU Adapters ({}):", hardware.gpu_adapters().len());
    for adapter in hardware.gpu_adapters() {
        let driver = if adapter.driver_info().is_empty() {
            adapter.driver().to_owned()
        } else {
            format!("{} {}", adapter.driver(), adapter.driver_info())
        };
        println!(
            "  - {} [vendor={} device=0x{:04x} | driver={}]",
            adapter.name(),
            adapter.vendor(),
            adapter.device_id(),
            driver
        );
    }

    println!("GPU Media Backends ({}):", hardware.gpu_backends().len());
    for backend in hardware.gpu_backends() {
        println!(
            "  - {} [{}] on {} ({})",
            backend.backend(),
            backend.capability(),
            backend.device_name(),
            backend.evidence()
        );
    }
    #[cfg(not(target_os = "linux"))]
    println!(
        "  note: GPU adapter/media backend detection uses Linux DRM sysfs and NVIDIA procfs; \
         no adapters or backends are reported on this host"
    );

    Ok(())
}

fn run_convert(args: &ConvertArgs) -> Result<(), String> {
    let plan = fastconv_probe::prepare_image_batch(
        &args.source,
        &args.target,
        args.compression,
        args.to,
        args.threads,
        args.backend,
    )
    .map_err(|err| err.to_string())?;

    let start = Instant::now();
    let report = fastconv_pipeline::execute_image_batch(&plan).map_err(|err| err.to_string())?;
    let elapsed = start.elapsed();

    let secs = elapsed.as_secs_f64().max(1e-9);
    let ms = secs * 1_000.0;
    let in_mb = u64_to_f64(report.total_input_bytes().get()) / 1_000_000.0;
    let out_mb = u64_to_f64(report.total_output_bytes().get()) / 1_000_000.0;
    let throughput_mbps = in_mb / secs;

    let backend_desc = match report.hardware_device() {
        Some(device) => format!("{} on {device}", report.backend()),
        None => report.backend().to_string(),
    };

    println!(
        "converted {} image(s) [{backend_desc} | {} | {} thread(s)]: {:.2} MB -> {:.2} MB in {:.1} ms ({:.2} MB/s)",
        report.files_converted(),
        plan.mode(),
        report.worker_threads_used(),
        in_mb,
        out_mb,
        ms,
        throughput_mbps
    );
    Ok(())
}

pub(crate) fn u64_to_f64(value: u64) -> f64 {
    let hi = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let lo = u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(hi) * 4_294_967_296.0 + f64::from(lo)
}

fn fail(message: &str) -> ExitCode {
    eprintln!("error: {message}");
    ExitCode::FAILURE
}

fn fail_with_hint(message: &str, hint: &str) -> ExitCode {
    eprintln!("error: {message}");
    eprintln!("{hint}");
    ExitCode::FAILURE
}

fn run_with_failure(result: Result<(), String>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => fail(&message),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();

    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] | ["-h" | "--help" | "help"] | ["help", "-h" | "--help"] => {
            print_global_help();
            ExitCode::SUCCESS
        }
        ["help", topic] => {
            if print_help_topic(topic) {
                ExitCode::SUCCESS
            } else {
                fail_with_hint(
                    &format!("unknown help topic '{topic}'"),
                    "Run `fastconv help` to list commands.",
                )
            }
        }
        ["help", ..] => fail_with_hint(
            "expected at most one help topic",
            "Run `fastconv help` to list commands.",
        ),
        ["-V" | "--version"] => {
            println!("fastconv {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        ["probe", rest @ ..] => match parse_probe_args(rest) {
            Ok(ProbeCommand::Help) => {
                print_probe_help();
                ExitCode::SUCCESS
            }
            Ok(ProbeCommand::Run) => run_with_failure(run_probe()),
            Err(message) => fail(&message),
        },
        ["audio", rest @ ..] => match audio::parse_audio_args(rest) {
            Ok(audio::AudioCommand::Help) => {
                audio::print_audio_help();
                ExitCode::SUCCESS
            }
            Ok(audio::AudioCommand::Run(parsed)) => run_with_failure(audio::run_audio(&parsed)),
            Err(message) => fail(&message),
        },
        ["remux", rest @ ..] => match video::parse_remux_args(rest) {
            Ok(video::RemuxCommand::Help) => {
                video::print_remux_help();
                ExitCode::SUCCESS
            }
            Ok(video::RemuxCommand::Run(parsed)) => run_with_failure(video::run_remux(&parsed)),
            Err(message) => fail(&message),
        },
        ["video", rest @ ..] => match video::parse_video_args(rest) {
            Ok(video::VideoCommand::Help) => {
                video::print_video_help();
                ExitCode::SUCCESS
            }
            Ok(video::VideoCommand::Run(parsed)) => run_with_failure(video::run_video(&parsed)),
            Err(message) => fail(&message),
        },
        ["convert", rest @ ..] => match parse_convert_args(rest) {
            Ok(ConvertCommand::Help) => {
                print_convert_help();
                ExitCode::SUCCESS
            }
            Ok(ConvertCommand::Run(parsed)) => run_with_failure(run_convert(&parsed)),
            Err(message) => fail(&message),
        },
        _ => fail_with_hint("invalid arguments", "Run `fastconv --help` for usage."),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ConvertArgs, ConvertCommand, ProbeCommand, help_text, parse_convert_args, parse_probe_args,
    };
    use fastconv_domain::{
        GpuBackend, HardwareBackend, ImageCompression, ImageOutputFormat, ImageQuality,
    };
    use std::num::NonZeroUsize;
    use std::path::PathBuf;

    #[test]
    fn parses_probe_args() -> Result<(), String> {
        assert_eq!(parse_probe_args(&[])?, ProbeCommand::Run);
        assert_eq!(parse_probe_args(&["-h"])?, ProbeCommand::Help);
        assert_eq!(parse_probe_args(&["--help"])?, ProbeCommand::Help);

        assert_eq!(
            parse_probe_args(&["magic"]),
            Err("expected `fastconv probe`".to_owned())
        );
        assert_eq!(
            parse_probe_args(&["--threads", "4"]),
            Err("unknown flag '--threads'".to_owned())
        );
        assert_eq!(
            parse_probe_args(&["--help", "extra"]),
            Err("expected `fastconv probe`".to_owned())
        );
        Ok(())
    }

    #[test]
    fn every_command_has_dedicated_help_text() -> Result<(), String> {
        for topic in ["probe", "convert", "audio", "remux", "video"] {
            let text = help_text(topic).ok_or_else(|| format!("missing help for {topic}"))?;
            assert!(text.contains("Usage:"), "missing usage for {topic}");
            assert!(
                !text.contains("Commands:"),
                "{topic} help fell back to global usage"
            );
        }
        assert_eq!(help_text("magic"), None);
        assert_eq!(help_text(""), None);
        Ok(())
    }

    #[test]
    fn parses_default_flagged_and_help_convert_arguments() -> Result<(), String> {
        assert_eq!(parse_convert_args(&["--help"])?, ConvertCommand::Help);
        assert_eq!(parse_convert_args(&["-h"])?, ConvertCommand::Help);

        let defaults = parse_convert_args(&["in.png", "out.webp"])?;
        assert_eq!(
            defaults,
            ConvertCommand::Run(ConvertArgs {
                source: PathBuf::from("in.png"),
                target: PathBuf::from("out.webp"),
                compression: None,
                to: None,
                threads: None,
                backend: HardwareBackend::Cpu,
            })
        );

        let ConvertCommand::Run(flagged) = parse_convert_args(&[
            "src_dir",
            "out_dir",
            "--quality",
            "85",
            "--to",
            "heic",
            "--threads",
            "4",
            "--backend",
            "nvenc",
        ])?
        else {
            return Err("expected ConvertCommand::Run".to_owned());
        };
        assert_eq!(
            flagged.compression,
            Some(ImageCompression::Lossy {
                quality: ImageQuality::new(85).map_err(|err| err.to_string())?
            })
        );
        assert_eq!(flagged.to, Some(ImageOutputFormat::Heic));
        assert_eq!(flagged.threads, NonZeroUsize::new(4));
        assert_eq!(flagged.backend, HardwareBackend::Gpu(GpuBackend::Nvenc));

        for (flag, expected) in [
            ("jpeg", ImageOutputFormat::Jpeg),
            ("jpg", ImageOutputFormat::Jpeg),
            ("png", ImageOutputFormat::Png),
            ("bmp", ImageOutputFormat::Bmp),
            ("qoi", ImageOutputFormat::Qoi),
            ("tiff", ImageOutputFormat::Tiff),
            ("tif", ImageOutputFormat::Tiff),
            ("jxl", ImageOutputFormat::Jxl),
            ("avif", ImageOutputFormat::Avif),
            ("gif", ImageOutputFormat::Gif),
            ("ico", ImageOutputFormat::Ico),
            ("pnm", ImageOutputFormat::Pnm),
            ("ppm", ImageOutputFormat::Pnm),
            ("pgm", ImageOutputFormat::Pnm),
            ("pbm", ImageOutputFormat::Pnm),
            ("pam", ImageOutputFormat::Pnm),
        ] {
            let ConvertCommand::Run(parsed) =
                parse_convert_args(&["src_dir", "out_dir", "--to", flag])?
            else {
                return Err("expected ConvertCommand::Run".to_owned());
            };
            assert_eq!(parsed.to, Some(expected));
            assert_eq!(parsed.compression, None);
        }

        let ConvertCommand::Run(vulkan_webp) =
            parse_convert_args(&["a.png", "b.webp", "--backend", "vulkan"])?
        else {
            return Err("expected ConvertCommand::Run".to_owned());
        };
        assert_eq!(
            vulkan_webp.backend,
            HardwareBackend::Gpu(GpuBackend::Vulkan)
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_and_duplicate_convert_arguments() {
        assert!(parse_convert_args(&["a.png"]).is_err());
        assert!(parse_convert_args(&["a.png", "b.webp", "--quality", "80", "--lossless"]).is_err());
        assert!(
            parse_convert_args(&["a.png", "b.webp", "--quality", "80", "--quality", "90"]).is_err()
        );
        assert!(parse_convert_args(&["a.png", "b.webp", "--lossless", "--lossless"]).is_err());
        assert!(parse_convert_args(&["a.png", "b.webp", "--to", "webp", "--to", "heic"]).is_err());
        assert!(parse_convert_args(&["a.png", "b.webp", "--to", "psd"]).is_err());
        assert!(parse_convert_args(&["a.png", "b.webp", "--format", "webp"]).is_err());
        assert!(
            parse_convert_args(&["a.png", "b.webp", "--threads", "2", "--threads", "4"]).is_err()
        );
        assert!(
            parse_convert_args(&["a.png", "b.webp", "--backend", "nvenc", "--backend", "cpu"])
                .is_err()
        );
        assert!(parse_convert_args(&["a.png", "b.webp", "--quality", "101"]).is_err());
        assert!(parse_convert_args(&["a.png", "b.webp", "--threads", "0"]).is_err());
        assert!(parse_convert_args(&["a.png", "b.webp", "--backend", "magic"]).is_err());
        assert!(parse_convert_args(&["a.png", "b.webp", "--unknown"]).is_err());
    }
}
