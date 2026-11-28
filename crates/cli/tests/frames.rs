use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_fastconv");

/// Minimal valid 1x1 `GIF89a` with one frame.
const TINY_GIF: &[u8] = &[
    0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00, 0x80, 0x00, 0x00, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x21, 0xF9, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2C, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x01, 0x00, 0x00, 0x02, 0x02, 0x44, 0x01, 0x00, 0x3B,
];

fn run(args: &[&str]) -> Result<Output, String> {
    Command::new(BIN)
        .args(args)
        .output()
        .map_err(|err| format!("failed to run fastconv: {err}"))
}

fn stdout(args: &[&str]) -> Result<String, String> {
    let output = run(args)?;
    if !output.status.success() {
        return Err(format!(
            "fastconv {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|err| format!("stdout is not UTF-8: {err}"))
}

fn failure(args: &[&str]) -> Result<String, String> {
    let output = run(args)?;
    if output.status.success() {
        return Err(format!(
            "fastconv {} unexpectedly succeeded",
            args.join(" ")
        ));
    }
    String::from_utf8(output.stderr).map_err(|err| format!("stderr is not UTF-8: {err}"))
}

fn temp_root(label: &str) -> Result<PathBuf, String> {
    let root = std::env::temp_dir().join(format!(
        "fastconv-cli-frames-{label}-{}",
        std::process::id()
    ));
    fs::create_dir_all(&root).map_err(|err| err.to_string())?;
    Ok(root)
}

#[test]
fn frames_help_is_reachable_through_all_entry_points() -> Result<(), String> {
    let baseline = stdout(&["frames", "--help"])?;
    assert_eq!(stdout(&["frames", "-h"])?, baseline);
    assert_eq!(stdout(&["help", "frames"])?, baseline);
    for needle in [
        "fastconv frames <source> <output-dir>",
        "[--fps <N> | --every <N>]",
        "[--start <N>] [--end <N>]",
        "--backend <auto|cpu>",
        "frame_000001",
    ] {
        assert!(baseline.contains(needle), "frames help missing '{needle}'");
    }
    Ok(())
}

#[test]
fn invalid_frames_arguments_fail_before_io() -> Result<(), String> {
    let cases: [(&[&str], &str); 8] = [
        (
            &["frames", "in.mp4", "out", "--quality", "80", "--lossless"],
            "cannot combine",
        ),
        (
            &["frames", "in.mp4", "out", "--fps", "10", "--every", "2"],
            "cannot combine",
        ),
        (&["frames", "in.mp4", "out", "--fps", "0"], "--fps"),
        (&["frames", "in.mp4", "out", "--fps", "nan"], "--fps"),
        (
            &["frames", "in.mp4", "out", "--start", "10", "--end", "5"],
            "must not exceed",
        ),
        (
            &["frames", "in.mp4", "out", "--backend", "nvenc"],
            "no silent CPU fallback",
        ),
        (
            &["frames", "in.mp4", "out", "--backend", "vulkan"],
            "no silent CPU fallback",
        ),
        (
            &["frames", "in.mp4", "out", "--to", "magic"],
            "unknown format",
        ),
    ];
    for (args, expected) in cases {
        let stderr = failure(args)?;
        assert!(
            stderr.contains(expected),
            "expected '{expected}' for {args:?}, got: {stderr}"
        );
    }
    Ok(())
}

#[test]
fn extracts_one_frame_from_animated_gif() -> Result<(), String> {
    let root = temp_root("extract")?;
    let source = root.join("in.gif");
    fs::write(&source, TINY_GIF).map_err(|err| err.to_string())?;
    let out = root.join("frames");
    let out_str = out.to_str().ok_or("non-UTF-8 output path")?;
    let source_str = source.to_str().ok_or("non-UTF-8 source path")?;
    let text = stdout(&["frames", source_str, out_str, "--to", "png"])?;
    for needle in ["extracted 1 frame", "1x1", ".png", "frame"] {
        assert!(text.contains(needle), "summary missing '{needle}': {text}");
    }
    assert!(Path::new(&out).join("frame_000001.png").is_file());
    fs::remove_dir_all(&root).map_err(|err| err.to_string())?;
    Ok(())
}

#[test]
fn backend_auto_and_explicit_cpu_are_equivalent() -> Result<(), String> {
    let root = temp_root("backend")?;
    let source = root.join("in.gif");
    fs::write(&source, TINY_GIF).map_err(|err| err.to_string())?;
    let source_str = source.to_str().ok_or("non-UTF-8 source path")?;
    for backend in ["auto", "cpu"] {
        let out = root.join(backend);
        let out_str = out.to_str().ok_or("non-UTF-8 output path")?;
        stdout(&["frames", source_str, out_str, "--backend", backend])?;
        assert!(Path::new(out_str).join("frame_000001.png").is_file());
    }
    fs::remove_dir_all(&root).map_err(|err| err.to_string())?;
    Ok(())
}
