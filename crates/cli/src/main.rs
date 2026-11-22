use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use fastconv_domain::{ExecutionPlan, HardwareBackend, MediaKind};

fn print_usage() {
    println!("fastconv {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Usage:");
    println!("  fastconv convert <source> <target>");
    println!("  fastconv --version");
    println!("  fastconv --help");
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
