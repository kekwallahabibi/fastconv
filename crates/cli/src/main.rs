use std::env;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use fastconv_domain::{GpuBackend, HardwareBackend, ImageOutputFormat, ImageQuality, WebpEncoding};

fn print_usage() {
    println!("fastconv {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Usage:");
    println!("  fastconv probe");
    println!(
        "  fastconv convert <source> <target> [--quality <0..=100> | --lossless] [--format <webp|heic>] [--threads <N>] [--backend <cpu|nvenc|vaapi|qsv|amf|videotoolbox|vulkan>]"
    );
    println!("  fastconv --version");
    println!("  fastconv --help");
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConvertArgs {
    source: PathBuf,
    target: PathBuf,
    encoding: WebpEncoding,
    format: Option<ImageOutputFormat>,
    threads: Option<NonZeroUsize>,
    backend: HardwareBackend,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ConvertCommand {
    Help,
    Run(ConvertArgs),
}

fn parse_backend(value: &str) -> Result<HardwareBackend, String> {
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

fn parse_format(value: &str) -> Result<ImageOutputFormat, String> {
    match value {
        "webp" => Ok(ImageOutputFormat::Webp),
        "heic" => Ok(ImageOutputFormat::Heic),
        other => Err(format!("unknown format '{other}' (expected webp or heic)")),
    }
}

fn parse_convert_args(args: &[&str]) -> Result<ConvertCommand, String> {
    let mut positional: Vec<&str> = Vec::new();
    let mut quality_arg: Option<ImageQuality> = None;
    let mut lossless = false;
    let mut format: Option<ImageOutputFormat> = None;
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
            "--format" => {
                if format.is_some() {
                    return Err("duplicate flag '--format'".to_owned());
                }
                let raw = args
                    .get(i + 1)
                    .copied()
                    .ok_or_else(|| "missing value for --format".to_owned())?;
                format = Some(parse_format(raw)?);
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

    let encoding = if lossless {
        WebpEncoding::Lossless
    } else {
        WebpEncoding::Lossy {
            quality: quality_arg.unwrap_or(ImageQuality::DEFAULT),
        }
    };

    Ok(ConvertCommand::Run(ConvertArgs {
        source: PathBuf::from(source),
        target: PathBuf::from(target),
        encoding,
        format,
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

    println!("wgpu Adapters ({}):", hardware.wgpu_adapters().len());
    for adapter in hardware.wgpu_adapters() {
        println!(
            "  - {} [{} | {} | {} | vendor={} device=0x{:04x} | driver={} {}]",
            adapter.name(),
            adapter.graphics_api(),
            adapter.device_type(),
            if adapter.is_hardware() {
                "hardware"
            } else {
                "software"
            },
            adapter.vendor(),
            adapter.device_id(),
            adapter.driver(),
            adapter.driver_info()
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

    Ok(())
}

fn run_convert(args: &ConvertArgs) -> Result<(), String> {
    let plan = fastconv_probe::prepare_image_batch(
        &args.source,
        &args.target,
        args.encoding,
        args.format,
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

fn u64_to_f64(value: u64) -> f64 {
    let hi = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let lo = u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(hi) * 4_294_967_296.0 + f64::from(lo)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();

    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] | ["-h" | "--help"] => {
            print_usage();
            ExitCode::SUCCESS
        }
        ["-V" | "--version"] => {
            println!("fastconv {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        ["probe"] => match run_probe() {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("error: {message}");
                ExitCode::FAILURE
            }
        },
        ["convert", rest @ ..] => match parse_convert_args(rest) {
            Ok(ConvertCommand::Help) => {
                print_usage();
                ExitCode::SUCCESS
            }
            Ok(ConvertCommand::Run(parsed)) => match run_convert(&parsed) {
                Ok(()) => ExitCode::SUCCESS,
                Err(message) => {
                    eprintln!("error: {message}");
                    ExitCode::FAILURE
                }
            },
            Err(message) => {
                eprintln!("error: {message}");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("error: invalid arguments");
            eprintln!("Run `fastconv --help` for usage.");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ConvertArgs, ConvertCommand, parse_convert_args};
    use fastconv_domain::{
        GpuBackend, HardwareBackend, ImageOutputFormat, ImageQuality, WebpEncoding,
    };
    use std::num::NonZeroUsize;
    use std::path::PathBuf;

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
                encoding: WebpEncoding::Lossy {
                    quality: ImageQuality::DEFAULT
                },
                format: None,
                threads: None,
                backend: HardwareBackend::Cpu,
            })
        );

        let ConvertCommand::Run(flagged) = parse_convert_args(&[
            "src_dir",
            "out_dir",
            "--quality",
            "85",
            "--format",
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
            flagged.encoding,
            WebpEncoding::Lossy {
                quality: ImageQuality::new(85).map_err(|err| err.to_string())?
            }
        );
        assert_eq!(flagged.format, Some(ImageOutputFormat::Heic));
        assert_eq!(flagged.threads, NonZeroUsize::new(4));
        assert_eq!(flagged.backend, HardwareBackend::Gpu(GpuBackend::Nvenc));

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
        assert!(
            parse_convert_args(&["a.png", "b.webp", "--format", "webp", "--format", "heic"])
                .is_err()
        );
        assert!(parse_convert_args(&["a.png", "b.webp", "--format", "avif"]).is_err());
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
