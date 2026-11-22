use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

struct BenchSpec {
    fixtures_dir: PathBuf,
    fixture_files: Vec<PathBuf>,
}

fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "failed to resolve workspace root from xtask manifest path".to_owned())
}

fn cargo_bin() -> OsString {
    env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

fn run_cargo_step(root: &Path, label: &str, args: &[&str]) -> Result<(), String> {
    println!("==> {label}");
    let status = Command::new(cargo_bin())
        .current_dir(root)
        .args(args)
        .status()
        .map_err(|err| format!("failed to launch '{label}': {err}"))?;

    if !status.success() {
        return Err(format!("step '{label}' failed with status {status}"));
    }

    Ok(())
}

fn cmd_verify(root: &Path) -> Result<(), String> {
    run_cargo_step(
        root,
        "cargo fmt --all -- --check",
        &["fmt", "--all", "--", "--check"],
    )?;
    run_cargo_step(
        root,
        "cargo clippy --workspace --all-targets -- -D warnings",
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
    )?;
    run_cargo_step(root, "cargo test --workspace", &["test", "--workspace"])?;
    println!("==> verify passed");
    Ok(())
}

fn discover_fixtures(root: &Path) -> Result<BenchSpec, String> {
    let fixtures_dir = root.join("fixtures");
    let entries = fs::read_dir(&fixtures_dir).map_err(|err| {
        format!(
            "failed to read fixtures directory '{}': {err}",
            fixtures_dir.display()
        )
    })?;

    let mut fixture_files = Vec::new();
    for entry_result in entries {
        let entry =
            entry_result.map_err(|err| format!("failed to read fixture directory entry: {err}"))?;
        let path = entry.path();
        let is_hidden = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with('.'));

        if !is_hidden && path.is_file() {
            fixture_files.push(path);
        }
    }
    fixture_files.sort();

    Ok(BenchSpec {
        fixtures_dir,
        fixture_files,
    })
}

fn cmd_bench(root: &Path) -> Result<(), String> {
    let spec = discover_fixtures(root)?;
    println!("==> fastconv benchmark harness (placeholder)");
    println!("fixtures dir : {}", spec.fixtures_dir.display());
    println!("fixtures     : {}", spec.fixture_files.len());
    for file in &spec.fixture_files {
        println!("  - {}", file.display());
    }
    println!("planned metrics:");
    println!("  - wall time (ms)");
    println!("  - throughput (MB/s)");
    println!("  - decode/encode rate (FPS)");
    println!("  - verified hardware backend (zero silent CPU fallback)");

    if spec.fixture_files.is_empty() {
        println!(
            "status: add media files to fixtures/ and wire a pipeline backend to run benchmarks."
        );
    } else {
        println!(
            "status: fixtures detected; wire concrete pipeline backends to execute timed runs."
        );
    }

    Ok(())
}

fn print_usage() {
    println!("Usage:");
    println!("  cargo xtask verify   Run fmt --check, clippy -D warnings, and workspace tests");
    println!("  cargo xtask bench    Inspect fixtures/ and report benchmark harness status");
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();

    let result = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["verify"] => workspace_root().and_then(|root| cmd_verify(&root)),
        ["bench"] => workspace_root().and_then(|root| cmd_bench(&root)),
        [] | ["-h" | "--help"] => {
            print_usage();
            Ok(())
        }
        _ => {
            eprintln!("error: unknown xtask subcommand");
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}
