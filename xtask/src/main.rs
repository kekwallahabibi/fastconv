mod bench;
mod fixtures;

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use crate::fixtures::{
    IMAGE_BATCH_COUNT, audio_wav_path, image_path, jpeg_image_path, video_mp4_path,
};

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
    println!("==> check no ffmpeg/ffprobe in crates/");
    check_no_ffmpeg_in_crates(&root.join("crates"))?;

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

    if has_nvenc_link_libraries() {
        run_cargo_step(
            root,
            "cargo clippy -p fastconv-pipeline -p fastconv-cli --all-targets --features nvenc -- -D warnings",
            &[
                "clippy",
                "-p",
                "fastconv-pipeline",
                "-p",
                "fastconv-cli",
                "--all-targets",
                "--features",
                "nvenc",
                "--",
                "-D",
                "warnings",
            ],
        )?;
        run_cargo_step(
            root,
            "cargo test -p fastconv-pipeline -p fastconv-cli --features nvenc",
            &[
                "test",
                "-p",
                "fastconv-pipeline",
                "-p",
                "fastconv-cli",
                "--features",
                "nvenc",
            ],
        )?;
    }

    println!("==> verify passed");
    Ok(())
}

pub(crate) fn has_nvenc_link_libraries() -> bool {
    if !cfg!(unix) {
        return false;
    }
    let mut dirs = vec![
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        PathBuf::from("/usr/local/cuda/lib64"),
        PathBuf::from("/opt/cuda/lib64"),
    ];
    for var in ["NVIDIA_VIDEO_CODEC_SDK_PATH", "CUDA_PATH", "CUDA_ROOT"] {
        if let Some(root) = env::var_os(var) {
            let base = PathBuf::from(root);
            dirs.push(base.join("lib64"));
            dirs.push(base.join("lib"));
            dirs.push(base);
        }
    }

    let has_encode = dirs.iter().any(|d| d.join("libnvidia-encode.so").is_file());
    let has_cuvid = dirs.iter().any(|d| d.join("libnvcuvid.so").is_file());
    has_encode && has_cuvid
}

fn check_no_ffmpeg_in_crates(crates_dir: &Path) -> Result<(), String> {
    let mut stack = vec![crates_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .map_err(|err| format!("failed to read '{}': {err}", dir.display()))?;
        for entry_result in entries {
            let entry = entry_result
                .map_err(|err| format!("failed to inspect entry in '{}': {err}", dir.display()))?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
                let source = fs::read_to_string(&path)
                    .map_err(|err| format!("failed to read '{}': {err}", path.display()))?;
                if let Some((line_no, line)) = find_forbidden_ffmpeg_reference(&source) {
                    return Err(format!(
                        "forbidden ffmpeg/ffprobe reference in {}:{} -> {}",
                        path.display(),
                        line_no,
                        line.trim()
                    ));
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum ScanState {
    Code,
    BlockComment(usize),
    StringLiteral,
    RawString { hashes: usize },
}

fn find_forbidden_ffmpeg_reference(source: &str) -> Option<(usize, String)> {
    let lines: Vec<&str> = source.split('\n').collect();
    let bytes = source.as_bytes();
    let mut i = 0_usize;
    let mut line_no = 1_usize;
    let mut line_code = String::new();
    let mut state = ScanState::Code;

    while let Some(&b) = bytes.get(i) {
        if b == b'\n' {
            if let Some(hit) = check_line_code(line_no, &line_code, &lines) {
                return Some(hit);
            }
            line_code.clear();
            line_no += 1;
            i += 1;
            continue;
        }

        match state {
            ScanState::BlockComment(depth) => {
                if bytes.get(i..i + 2) == Some(b"/*") {
                    state = ScanState::BlockComment(depth + 1);
                    i += 2;
                } else if bytes.get(i..i + 2) == Some(b"*/") {
                    state = if depth <= 1 {
                        ScanState::Code
                    } else {
                        ScanState::BlockComment(depth - 1)
                    };
                    i += 2;
                } else {
                    i += 1;
                }
            }
            ScanState::StringLiteral => {
                if b == b'\\' {
                    if bytes.get(i + 1) == Some(&b'\n') {
                        if let Some(hit) = check_line_code(line_no, &line_code, &lines) {
                            return Some(hit);
                        }
                        line_code.clear();
                        line_no += 1;
                        i += 2;
                    } else if let Some(&next) = bytes.get(i + 1) {
                        line_code.push('\\');
                        line_code.push(char::from(next));
                        i += 2;
                    } else {
                        i += 1;
                    }
                } else if b == b'"' {
                    line_code.push('"');
                    i += 1;
                    state = ScanState::Code;
                } else {
                    line_code.push(char::from(b));
                    i += 1;
                }
            }
            ScanState::RawString { hashes } => {
                if b == b'"' && has_closing_hashes(bytes, i + 1, hashes) {
                    line_code.push('"');
                    i += 1 + hashes;
                    state = ScanState::Code;
                } else {
                    line_code.push(char::from(b));
                    i += 1;
                }
            }
            ScanState::Code => {
                if bytes.get(i..i + 2) == Some(b"//") {
                    i += 2;
                    while bytes.get(i).is_some_and(|&ch| ch != b'\n') {
                        i += 1;
                    }
                } else if bytes.get(i..i + 2) == Some(b"/*") {
                    state = ScanState::BlockComment(1);
                    i += 2;
                } else if let Some((advance, hashes)) = match_raw_string_start(bytes, i) {
                    line_code.push('"');
                    i += advance;
                    state = ScanState::RawString { hashes };
                } else if b == b'"' {
                    line_code.push('"');
                    i += 1;
                    state = ScanState::StringLiteral;
                } else if b == b'\''
                    && let Some(len) = char_literal_len(bytes, i)
                {
                    line_code.push(' ');
                    i += len;
                } else {
                    line_code.push(char::from(b));
                    i += 1;
                }
            }
        }
    }

    check_line_code(line_no, &line_code, &lines)
}

fn check_line_code(line_no: usize, line_code: &str, lines: &[&str]) -> Option<(usize, String)> {
    let lower = line_code.to_ascii_lowercase();
    if lower.contains("ffmpeg") || lower.contains("ffprobe") {
        let raw_line = lines.get(line_no.saturating_sub(1)).copied().unwrap_or("");
        return Some((line_no, raw_line.to_owned()));
    }
    None
}

fn match_raw_string_start(bytes: &[u8], i: usize) -> Option<(usize, usize)> {
    if i > 0
        && bytes
            .get(i - 1)
            .is_some_and(|&prev| prev.is_ascii_alphanumeric() || prev == b'_')
    {
        return None;
    }

    let r_offset = match bytes.get(i) {
        Some(&b'r') => 1,
        Some(&(b'b' | b'c')) if bytes.get(i + 1) == Some(&b'r') => 2,
        _ => return None,
    };

    let mut hashes = 0_usize;
    while bytes.get(i + r_offset + hashes) == Some(&b'#') {
        hashes += 1;
    }
    if bytes.get(i + r_offset + hashes) == Some(&b'"') {
        Some((r_offset + hashes + 1, hashes))
    } else {
        None
    }
}

fn has_closing_hashes(bytes: &[u8], start: usize, hashes: usize) -> bool {
    (0..hashes).all(|offset| bytes.get(start + offset) == Some(&b'#'))
}

fn char_literal_len(bytes: &[u8], i: usize) -> Option<usize> {
    match bytes.get(i + 1) {
        Some(&b'\\') => {
            let mut j = i + 3;
            while let Some(&b) = bytes.get(j) {
                if b == b'\n' || j - i > 16 {
                    return None;
                }
                if b == b'\'' {
                    return Some(j - i + 1);
                }
                j += 1;
            }
            None
        }
        Some(&b) if b != b'\n' && b != b'\'' && bytes.get(i + 2) == Some(&b'\'') => Some(3),
        _ => None,
    }
}

fn cmd_bench_status(root: &Path) {
    let png_count = (1..=IMAGE_BATCH_COUNT)
        .filter(|&idx| fixtures::is_non_empty_file(&image_path(root, idx)))
        .count();
    let jpeg_count = (1..=IMAGE_BATCH_COUNT)
        .filter(|&idx| fixtures::is_non_empty_file(&jpeg_image_path(root, idx)))
        .count();
    let has_wav = fixtures::is_non_empty_file(&audio_wav_path(root));
    let has_mp4 = fixtures::is_non_empty_file(&video_mp4_path(root));

    println!("==> fastconv benchmark harness");
    println!("fixtures dir : {}", root.join("fixtures").display());
    println!("4K PNG batch : {png_count}/{IMAGE_BATCH_COUNT}");
    println!("4K JPEG batch: {jpeg_count}/{IMAGE_BATCH_COUNT}");
    println!(
        "60s WAV      : {}",
        if has_wav { "ready" } else { "missing" }
    );
    println!(
        "1080p60 MP4  : {}",
        if has_mp4 { "ready" } else { "missing" }
    );
    println!("run `cargo xtask gen-fixtures` to generate missing fixtures.");
    println!("run `cargo xtask bench` to run the head-to-head benchmark (fastconv vs FFmpeg).");
    println!("run `cargo xtask bench --baseline` to execute and save the FFmpeg baseline.");
}

fn print_usage() {
    println!("Usage:");
    println!(
        "  cargo xtask verify            Check no-ffmpeg invariant, fmt, clippy -D warnings, and tests"
    );
    println!(
        "  cargo xtask gen-fixtures      Generate reproducible 4K PNG/JPEG, WAV, and 1080p60 MP4 fixtures"
    );
    println!("  cargo xtask gen-fixtures --force");
    println!("                                Regenerate all fixtures from scratch");
    println!(
        "  cargo xtask bench             Run 4K PNG/JPEG head-to-head benchmarks (fastconv vs ffmpeg) + validation"
    );
    println!(
        "  cargo xtask bench --baseline  Run FFmpeg baseline across all workloads and save JSON"
    );
    println!("  cargo xtask bench --status    Report local fixture availability");
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
        ["gen-fixtures"] => workspace_root().and_then(|root| fixtures::generate_all(&root, false)),
        ["gen-fixtures", "--force"] => {
            workspace_root().and_then(|root| fixtures::generate_all(&root, true))
        }
        ["bench"] => workspace_root().and_then(|root| bench::run_head_to_head(&root)),
        ["bench", "--baseline"] => workspace_root().and_then(|root| bench::run_baseline(&root)),
        ["bench", "--status"] => workspace_root().map(|root| cmd_bench_status(&root)),
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

#[cfg(test)]
mod tests {
    use super::{check_no_ffmpeg_in_crates, find_forbidden_ffmpeg_reference, workspace_root};

    #[test]
    fn crates_contain_no_ffmpeg_or_ffprobe_references() -> Result<(), String> {
        let root = workspace_root()?;
        check_no_ffmpeg_in_crates(&root.join("crates"))
    }

    #[test]
    fn detects_ffmpeg_or_ffprobe_in_code_while_allowing_comments() {
        let clean = "\
/// Never invokes ffmpeg or ffprobe.
/* nested /* ffmpeg */ ffprobe */
fn helper<'a>(s: &'a str) -> char { let _ = s; '\"' }
";
        assert_eq!(find_forbidden_ffmpeg_reference(clean), None);

        let bad_ffmpeg = "let cmd = Command::new(\"ffmpeg\");\n";
        assert!(find_forbidden_ffmpeg_reference(bad_ffmpeg).is_some());

        let bad_ffprobe = "let cmd = Command::new(\"ffprobe\");\n";
        assert!(find_forbidden_ffmpeg_reference(bad_ffprobe).is_some());

        let after_char_quote = "let q = '\"';\nlet cmd = \"ffmpeg\";\n";
        assert_eq!(
            find_forbidden_ffmpeg_reference(after_char_quote),
            Some((2, "let cmd = \"ffmpeg\";".to_owned()))
        );

        let raw_with_comment_marker = "let raw = r#\"/* \"#;\nlet cmd = \"ffmpeg\";\n";
        assert_eq!(
            find_forbidden_ffmpeg_reference(raw_with_comment_marker),
            Some((2, "let cmd = \"ffmpeg\";".to_owned()))
        );

        let raw_multiline = "let raw = r#\"\nffprobe\n\"#;\n";
        assert_eq!(
            find_forbidden_ffmpeg_reference(raw_multiline),
            Some((2, "ffprobe".to_owned()))
        );
    }
}
