mod audio_bench;
mod bench;
mod fixtures;
mod remux_bench;

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use crate::fixtures::{
    IMAGE_BATCH_COUNT, audio_wav_path, image_path, jpeg_image_path, video_mp4_path, webp_image_path,
};

fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "failed to resolve workspace root from xtask manifest path".to_owned())
}

pub(crate) fn cargo_bin() -> OsString {
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

    println!("==> check licenses (AGPL-3.0-compatible or permissive)");
    check_licenses(root)?;

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

fn check_licenses(root: &Path) -> Result<(), String> {
    let output = Command::new(cargo_bin())
        .current_dir(root)
        .args(["metadata", "--format-version", "1"])
        .output()
        .map_err(|err| format!("failed to launch cargo metadata: {err}"))?;

    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|err| format!("failed to parse cargo metadata JSON: {err}"))?;

    scan_licenses(&metadata)
}

fn scan_licenses(metadata: &serde_json::Value) -> Result<(), String> {
    let workspace_members = metadata
        .get("workspace_members")
        .and_then(serde_json::Value::as_array)
        .map_or_default(|items| {
            items
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_owned)
                .collect::<std::collections::BTreeSet<String>>()
        });

    let packages = metadata
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "cargo metadata is missing 'packages'".to_owned())?;

    let mut forbidden: Vec<(String, String)> = Vec::new();
    let mut checked = 0_usize;

    for package in packages {
        let id = package
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if workspace_members.contains(id) {
            continue;
        }

        let name = package
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(id)
            .to_owned();

        let license = package
            .get("license")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_owned();

        checked += 1;
        if forbidden_license(&license) {
            let shown = if license.is_empty() {
                "<missing>".to_owned()
            } else {
                license
            };
            forbidden.push((name, shown));
        }
    }

    if forbidden.is_empty() {
        println!(
            "checked {checked} dependencies: all licenses are AGPL-3.0-compatible or permissive"
        );
        Ok(())
    } else {
        let details = forbidden
            .iter()
            .map(|(name, license)| format!("{name} ({license})"))
            .collect::<Vec<_>>()
            .join(", ");
        Err(format!(
            "forbidden or missing license in default dependency tree: {details}"
        ))
    }
}

/// `fastconv` is AGPL-3.0-only, so permissive licenses and AGPL-compatible copyleft are fine.
/// Dual-license expressions are accepted when at least one alternative is compatible; a bare
/// proprietary/non-commercial/unknown or GPL-2.0-only license is rejected.
fn forbidden_license(license: &str) -> bool {
    !license_allowed(license)
}

fn license_allowed(license: &str) -> bool {
    let tokens: Vec<String> = license
        .trim()
        .to_ascii_lowercase()
        .replace('/', " or ")
        .replace('(', " ( ")
        .replace(')', " ) ")
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    if tokens.is_empty() {
        return false;
    }
    let mut parser = LicenseParser {
        tokens: &tokens,
        pos: 0,
        malformed: false,
    };
    let value = parser.parse_or();
    if parser.malformed || parser.pos != tokens.len() {
        return false;
    }
    value
}

struct LicenseParser<'a> {
    tokens: &'a [String],
    pos: usize,
    malformed: bool,
}

impl<'a> LicenseParser<'a> {
    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(String::as_str)
    }

    fn next(&mut self) -> Option<&'a str> {
        let token = self.tokens.get(self.pos).map(String::as_str);
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    fn parse_or(&mut self) -> bool {
        let mut value = self.parse_and();
        while self.peek() == Some("or") {
            self.pos += 1;
            let rhs = self.parse_and();
            value = value || rhs;
        }
        value
    }

    fn parse_and(&mut self) -> bool {
        let mut value = self.parse_primary();
        while self.peek() == Some("and") {
            self.pos += 1;
            let rhs = self.parse_primary();
            value = value && rhs;
        }
        value
    }

    fn parse_primary(&mut self) -> bool {
        if self.peek() == Some("(") {
            self.pos += 1;
            let value = self.parse_or();
            if self.peek() == Some(")") {
                self.pos += 1;
            } else {
                self.malformed = true;
            }
            return value;
        }
        let Some(base) = self.next() else {
            self.malformed = true;
            return false;
        };
        if self.peek() == Some("with") {
            self.pos += 1;
            let Some(exception) = self.next() else {
                self.malformed = true;
                return false;
            };
            return license_term_with_exception(base, exception);
        }
        license_term_allowed(base)
    }
}

fn license_term_with_exception(base: &str, exception: &str) -> bool {
    // Only the explicitly reviewed GPL-2.0 + Classpath combination is acceptable.
    matches!(base, "gpl-2.0" | "gpl-2.0-only" | "gpl-2.0-or-later")
        && exception == "classpath-exception-2.0"
}

fn license_term_allowed(term: &str) -> bool {
    let base = term
        .split(" with ")
        .next()
        .unwrap_or(term)
        .trim()
        .trim_matches('"');
    matches!(
        base,
        "mit"
            | "apache-2.0"
            | "bsd-2-clause"
            | "bsd-3-clause"
            | "0bsd"
            | "zlib"
            | "isc"
            | "unlicense"
            | "cc0"
            | "cc0-1.0"
            | "mpl-2.0"
            | "ijg"
            | "unicode-3.0"
            | "agpl-3.0"
            | "agpl-3.0-only"
            | "agpl-3.0-or-later"
            | "gpl-3.0"
            | "gpl-3.0-only"
            | "gpl-3.0-or-later"
            | "lgpl-3.0"
            | "lgpl-3.0-only"
            | "lgpl-3.0-or-later"
    )
}

fn cmd_bench_status(root: &Path) {
    let png_count = (1..=IMAGE_BATCH_COUNT)
        .filter(|&idx| fixtures::is_non_empty_file(&image_path(root, idx)))
        .count();
    let jpeg_count = (1..=IMAGE_BATCH_COUNT)
        .filter(|&idx| fixtures::is_non_empty_file(&jpeg_image_path(root, idx)))
        .count();
    let webp_count = (1..=IMAGE_BATCH_COUNT)
        .filter(|&idx| fixtures::is_non_empty_file(&webp_image_path(root, idx)))
        .count();
    let has_wav = fixtures::is_non_empty_file(&audio_wav_path(root));
    let audio_batch_count = fixtures::audio_batch_inputs(root)
        .iter()
        .filter(|path| fixtures::is_non_empty_file(path))
        .count();
    let has_mp4 = fixtures::is_non_empty_file(&video_mp4_path(root));

    println!("==> fastconv benchmark harness");
    println!("fixtures dir : {}", root.join("fixtures").display());
    println!("4K PNG batch : {png_count}/{IMAGE_BATCH_COUNT}");
    println!("4K JPEG batch: {jpeg_count}/{IMAGE_BATCH_COUNT}");
    println!("4K WebP batch: {webp_count}/{IMAGE_BATCH_COUNT}");
    println!(
        "60s WAV      : {}",
        if has_wav { "ready" } else { "missing" }
    );
    println!(
        "Mixed audio  : {}/{}",
        audio_batch_count,
        fixtures::audio_batch_inputs(root).len()
    );
    println!(
        "1080p60 MP4  : {}",
        if has_mp4 { "ready" } else { "missing" }
    );
    println!("run `cargo xtask gen-fixtures` to generate missing fixtures.");
    println!("run `cargo xtask bench` to run the head-to-head benchmark (fastconv vs FFmpeg).");
    println!("run `cargo xtask bench --baseline` to execute and save the FFmpeg baseline.");
    println!("run `cargo xtask bench --audio` for the native audio head-to-head benchmark.");
}

fn print_usage() {
    println!("Usage:");
    println!(
        "  cargo xtask verify            Check no-ffmpeg invariant, fmt, clippy -D warnings, and tests"
    );
    println!(
        "  cargo xtask gen-fixtures      Generate reproducible 4K PNG/JPEG/WebP, WAV, and 1080p60 MP4 fixtures"
    );
    println!("  cargo xtask gen-fixtures --force");
    println!("                                Regenerate all fixtures from scratch");
    println!(
        "  cargo xtask bench             Run 4K image head-to-head benchmarks (fastconv vs ffmpeg >= 5.1 with QOI codec + ffprobe) + validation"
    );
    println!(
        "  cargo xtask bench --baseline  Run FFmpeg baseline across all workloads and save JSON"
    );
    println!(
        "  cargo xtask bench --audio     Audio head-to-head (fastconv vs ffmpeg) with output validation"
    );
    println!(
        "  cargo xtask bench --remux     Video remux head-to-head (fastconv vs ffmpeg -c copy) with ffprobe validation"
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
        ["bench", "--audio"] => workspace_root().and_then(|root| audio_bench::run(&root)),
        ["bench", "--status"] => workspace_root().map(|root| cmd_bench_status(&root)),
        ["bench", "--remux"] => workspace_root().and_then(|root| remux_bench::run(&root)),
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
    use super::{
        check_no_ffmpeg_in_crates, find_forbidden_ffmpeg_reference, forbidden_license,
        scan_licenses, workspace_root,
    };

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

    #[test]
    fn license_policy_allows_agpl_compatible_and_permissive() {
        for allowed in [
            "MIT",
            "Apache-2.0",
            "BSD-3-Clause",
            "0BSD OR MIT OR Apache-2.0",
            "(MIT OR Apache-2.0) AND IJG",
            "GPL-3.0-only",
            "LGPL-3.0-only",
            "AGPL-3.0-only OR LicenseRef-Imazen-Commercial",
            "MPL-2.0",
            "Zlib",
            "ISC",
            "Unlicense OR MIT",
            "CC0-1.0",
            "GPL-2.0-only WITH Classpath-exception-2.0",
            "GPL-3.0-only AND (GPL-2.0-only WITH Classpath-exception-2.0)",
        ] {
            assert!(!forbidden_license(allowed), "should be allowed: {allowed}");
        }
    }

    #[test]
    fn license_policy_rejects_incompatible_missing_and_proprietary() {
        for forbidden in [
            "",
            "LicenseRef-Imazen-Commercial",
            "Proprietary",
            "CC-BY-NC-4.0",
            "GPL-2.0-only",
            "MIT AND GPL-2.0-only",
            "(MIT OR Apache-2.0) AND GPL-2.0-only",
            "MIT ) AND GPL-2.0-only",
            "(MIT",
            "GPL-2.0-only WITH Unknown-exception",
        ] {
            assert!(
                forbidden_license(forbidden),
                "should be forbidden: {forbidden}"
            );
        }
    }

    #[test]
    fn scan_licenses_detects_gpl2_and_missing_license() -> Result<(), String> {
        let json = r#"{
            "workspace_members": [],
            "packages": [
                { "id": "legacy 0.1.0", "name": "legacy", "license": "GPL-2.0-only" },
                { "id": "unknown 0.2.0", "name": "unknown" }
            ]
        }"#;
        let metadata: serde_json::Value =
            serde_json::from_str(json).map_err(|err| format!("valid JSON expected: {err}"))?;
        let err = scan_licenses(&metadata)
            .err()
            .ok_or_else(|| "incompatible licenses must be flagged".to_owned())?;
        assert!(err.contains("legacy"), "{err}");
        assert!(err.contains("unknown"), "{err}");
        assert!(err.contains("<missing>"), "{err}");
        Ok(())
    }
}
