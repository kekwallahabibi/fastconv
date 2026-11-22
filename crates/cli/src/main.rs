use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use fastconv_domain::{ExecutionPlan, HardwareBackend, MediaKind};

fn print_usage() {
    println!("fastconv {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Usage:");
    println!("  fastconv probe");
    println!("  fastconv convert <source> <target>");
    println!("  fastconv --version");
    println!("  fastconv --help");
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

fn run_convert(source: &str, target: &str) -> Result<(), String> {
    let source_path = PathBuf::from(source);
    let target_path = PathBuf::from(target);

    let request = fastconv_probe::prepare_request(
        &source_path,
        target_path,
        MediaKind::Video,
        HardwareBackend::Cpu,
    )
    .map_err(|err| err.to_string())?;

    let plan = ExecutionPlan::from_request(request).map_err(|err| err.to_string())?;
    fastconv_pipeline::execute(&plan).map_err(|err| err.to_string())
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
        ["convert", source, target] => match run_convert(source, target) {
            Ok(()) => ExitCode::SUCCESS,
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
