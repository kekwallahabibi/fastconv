use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_fastconv");

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

#[test]
fn global_help_is_available_through_all_entry_points() -> Result<(), String> {
    let baseline = stdout(&["--help"])?;
    let variants: [&[&str]; 5] = [
        &[],
        &["-h"],
        &["help"],
        &["help", "-h"],
        &["help", "--help"],
    ];
    for args in variants {
        assert_eq!(stdout(args)?, baseline, "help differs for {args:?}");
    }
    for needle in [
        "Commands:",
        "probe ",
        "convert ",
        "audio ",
        "remux ",
        "video ",
        "help ",
        "-h, --help",
        "-V, --version",
        "--hwdecode <auto|cpu|nvdec>",
        "--compression-level <fast|balanced|best>",
        "--tiff-compression <deflate|lzw|packbits|none>",
        "--chroma <420|444>",
        "--bit-depth <8|16|24|32|f32>",
        "fastconv <command> --help",
        "fastconv help <command>",
    ] {
        assert!(baseline.contains(needle), "global help missing '{needle}'");
    }
    Ok(())
}

#[test]
fn version_flag_reports_version() -> Result<(), String> {
    let expected = format!("fastconv {}", env!("CARGO_PKG_VERSION"));
    for args in [&["-V"][..], &["--version"][..]] {
        assert_eq!(stdout(args)?.trim(), expected);
    }
    Ok(())
}

#[test]
fn every_subcommand_has_dedicated_help() -> Result<(), String> {
    let global = stdout(&["--help"])?;
    for command in ["probe", "convert", "audio", "remux", "video"] {
        let by_flag = stdout(&[command, "--help"])?;
        assert_eq!(stdout(&[command, "-h"])?, by_flag, "{command} -h differs");
        assert_eq!(
            stdout(&["help", command])?,
            by_flag,
            "help {command} differs"
        );
        assert!(
            by_flag.contains("Usage:\n  fastconv"),
            "{command} lacks usage"
        );
        assert!(
            !by_flag.contains("Commands:\n"),
            "{command} fell back to global help"
        );
        assert_ne!(by_flag, global, "{command} printed global help");
    }
    Ok(())
}

#[test]
fn probe_help_documents_detected_hardware() -> Result<(), String> {
    let text = stdout(&["probe", "--help"])?;
    for needle in [
        "model",
        "architecture",
        "physical cores",
        "logical threads",
        "SIMD",
        "memory",
        "gpu adapter",
        "nvenc",
        "vaapi",
        "qsv",
        "amf",
        "videotoolbox",
        "vulkan",
        "-h, --help",
    ] {
        assert!(text.contains(needle), "probe help missing '{needle}'");
    }
    Ok(())
}

#[test]
fn convert_help_documents_accepted_flags_and_rules() -> Result<(), String> {
    let text = stdout(&["convert", "--help"])?;
    for needle in [
        "[--compression-level <fast|balanced|best>]",
        "[--tiff-compression <deflate|lzw|packbits|none>]",
        "[--chroma <420|444>]",
        "<source>",
        "<target>",
        "--to <format>",
        "--quality <0..=100>",
        "--lossless",
        "--threads <N>",
        "--backend <name>",
        "-h, --help",
        "webp",
        "heic",
        "jpeg",
        "png",
        "bmp",
        "qoi",
        "tiff",
        "jxl",
        "avif",
        "gif",
        "ico",
        "pnm",
        "default 80",
        "mutually exclusive",
        "jpg -> jpeg",
        "tif -> tiff",
        "ppm/pgm/pbm/pam -> pnm",
        "256-color",
        "HEIC",
        "pure Rust HEVC-Intra encoder",
        "GPU-accelerated HEIC only",
        "no silent CPU fallback",
        "directory batches default to webp",
        "vaapi",
        "qsv",
        "amf",
        "videotoolbox",
        "vulkan",
        "default: fast for png/tiff/jxl/avif",
        "balanced for webp",
        "Supported only for .tiff on the cpu backend",
        "Rejected: bmp, qoi, ico, pnm, jpeg, gif, heic",
        "with lzw/packbits/none",
        "jpeg and avif on the cpu backend",
        "bmp, qoi, tiff, jxl, gif, ico, pnm and heic",
    ] {
        assert!(text.contains(needle), "convert help missing '{needle}'");
    }
    assert!(
        !text.contains("required for directory batches"),
        "convert --to is optional for directory batches (defaults to webp)"
    );
    assert!(
        !text.contains("CPU has no native HEIC encoder"),
        "convert help must not claim the CPU backend lacks a HEIC encoder"
    );
    Ok(())
}

#[test]
fn audio_help_documents_all_encoding_flags_and_rules() -> Result<(), String> {
    let text = stdout(&["audio", "--help"])?;
    for needle in [
        "fastconv audio <source> <target> [--to <format>] [--bitrate <rate> | --quality <0..=100>]",
        "[--sample-rate <hz>] [--channels <1..=8>] [--bit-depth <8|16|24|32|f32>]",
        "[--threads <N>] [--backend <name>]",
        "<source>",
        "<target>",
        "--to <format>",
        "--bitrate <rate>",
        "--quality <0..=100>",
        "--sample-rate <hz>",
        "--channels <1..=8>",
        "--bit-depth <8|16|24|32|f32>",
        "--threads <N>",
        "--backend <name>",
        "-h, --help",
        "flac",
        "wav",
        "opus",
        "ogg",
        "mp3",
        "aac",
        "m4a",
        "alac",
        "aiff",
        "wave -> wav",
        "oga -> ogg",
        "aif/aifc -> aiff",
        "Lossless outputs: wav, flac, alac, aiff",
        "ALAC writes an M4A container",
        ".m4a",
        ".alac",
        "CPU only",
        "no silent CPU fallback",
        "directory batches default to flac",
        "mutually exclusive with --bitrate",
        "use --bitrate",
        "Opus accepts only 48000",
        "excluding 4",
        "nvenc",
        "vaapi",
        "qsv",
        "amf",
        "videotoolbox",
        "vulkan",
    ] {
        assert!(text.contains(needle), "audio help missing '{needle}'");
    }
    assert!(
        !text.contains("--lossless"),
        "audio help must not claim --lossless"
    );
    assert!(
        !text.contains("required for directory batches"),
        "audio --to is optional for directory batches (defaults to flac)"
    );
    Ok(())
}

#[test]
fn remux_and_video_help_cover_their_constraints() -> Result<(), String> {
    let remux = stdout(&["remux", "--help"])?;
    for needle in [
        "--to <mp4|mkv|webm|mov|avi|ts>",
        "--threads <N>",
        "-h, --help",
        "mp4 (m4v)",
        "qt",
        "m2ts",
        "GIF and WebP cannot be remuxed",
        "fastconv video",
    ] {
        assert!(remux.contains(needle), "remux help missing '{needle}'");
    }

    let video = stdout(&["video", "--help"])?;
    for needle in [
        "--hwdecode <auto|cpu|nvdec>",
        "--vcodec",
        "--acodec",
        "--quality <0..=100>",
        "--vbitrate <rate>",
        "--abitrate <rate>",
        "--asample-rate <Hz>",
        "--achannels <N>",
        "default 80",
        "software -> cpu",
        "cuda/gpu -> nvdec",
        "hard error",
        "GIF",
        "webp",
        "HEVC has no CPU encoder",
    ] {
        assert!(video.contains(needle), "video help missing '{needle}'");
    }
    assert!(
        !video.contains("all codecs") && !video.contains("every codec"),
        "video help must not claim the CPU backend supports HEVC"
    );
    Ok(())
}

#[test]
fn invalid_help_topics_and_extra_arguments_fail() -> Result<(), String> {
    let cases: [&[&str]; 3] = [
        &["help", "magic"],
        &["help", "probe", "extra"],
        &["help", "--to"],
    ];
    for args in cases {
        let stderr = failure(args)?;
        assert!(stderr.contains("error:"), "no error for {args:?}");
    }
    Ok(())
}

#[test]
fn invalid_probe_arguments_fail() -> Result<(), String> {
    let cases: [&[&str]; 3] = [
        &["probe", "magic"],
        &["probe", "--threads", "4"],
        &["probe", "--help", "extra"],
    ];
    for args in cases {
        let stderr = failure(args)?;
        assert!(stderr.contains("error:"), "no error for {args:?}");
    }
    assert!(failure(&["probe", "magic"])?.contains("expected `fastconv probe`"));
    assert!(failure(&["probe", "--verbose"])?.contains("unknown flag '--verbose'"));
    Ok(())
}

#[test]
fn invalid_audio_encoding_flags_fail_before_io() -> Result<(), String> {
    let cases: [(&[&str], &str); 11] = [
        (
            &[
                "audio",
                "in.wav",
                "out.flac",
                "--bitrate",
                "128k",
                "--quality",
                "80",
            ],
            "cannot combine --bitrate with --quality",
        ),
        (
            &[
                "audio",
                "in.wav",
                "out.mp3",
                "--bitrate",
                "128k",
                "--bitrate",
                "192k",
            ],
            "duplicate flag",
        ),
        (
            &[
                "audio",
                "in.wav",
                "out.mp3",
                "--quality",
                "40",
                "--quality",
                "60",
            ],
            "duplicate flag",
        ),
        (
            &["audio", "in.wav", "out.mp3", "--bitrate"],
            "missing value",
        ),
        (
            &["audio", "in.wav", "out.mp3", "--bitrate", "abc"],
            "invalid audio bitrate",
        ),
        (
            &["audio", "in.wav", "out.mp3", "--quality", "101"],
            "invalid --quality",
        ),
        (
            &["audio", "in.wav", "out.mp3", "--sample-rate", "0"],
            "invalid --sample-rate",
        ),
        (
            &[
                "audio",
                "in.wav",
                "out.mp3",
                "--sample-rate",
                "44100",
                "--sample-rate",
                "48000",
            ],
            "duplicate flag",
        ),
        (
            &["audio", "in.wav", "out.mp3", "--channels", "0"],
            "invalid --channels",
        ),
        (
            &["audio", "in.wav", "out.mp3", "--channels", "9"],
            "invalid --channels",
        ),
        (
            &[
                "audio",
                "in.wav",
                "out.mp3",
                "--channels",
                "2",
                "--channels",
                "1",
            ],
            "duplicate flag",
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
fn invalid_audio_bit_depth_flags_fail_before_io() -> Result<(), String> {
    let cases: [(&[&str], &str); 3] = [
        (
            &["audio", "in.wav", "out.wav", "--bit-depth"],
            "missing value",
        ),
        (
            &["audio", "in.wav", "out.wav", "--bit-depth", "64"],
            "invalid audio bit depth",
        ),
        (
            &[
                "audio",
                "in.wav",
                "out.wav",
                "--bit-depth",
                "16",
                "--bit-depth",
                "32",
            ],
            "duplicate flag",
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
