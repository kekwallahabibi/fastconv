//! End-to-end `fastconv stack` tests driven through the real binary.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

const BIN: &str = env!("CARGO_BIN_EXE_fastconv");
static COUNTER: AtomicU64 = AtomicU64::new(0);

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
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "fastconv-cli-stack-{label}-{}-{id}",
        std::process::id()
    ));
    fs::create_dir_all(&root).map_err(|err| err.to_string())?;
    Ok(root)
}

fn solid(width: u32, height: u32, level: u8) -> Vec<u8> {
    let value = 32_u8.saturating_add(level.saturating_mul(64));
    vec![value; usize::try_from(width * height * 3).unwrap_or(0)]
}

fn write_png(path: &Path, width: u32, height: u32, pixels: &[u8]) -> Result<(), String> {
    let file = fs::File::create(path).map_err(|err| err.to_string())?;
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|err| err.to_string())?;
    writer
        .write_image_data(pixels)
        .map_err(|err| err.to_string())?;
    Ok(())
}

fn str_path(path: &Path) -> Result<&str, String> {
    path.to_str().ok_or_else(|| "non-UTF-8 path".to_owned())
}

fn extracted_frames(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = fs::read_dir(dir)
        .map_err(|err| err.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("png"))
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

fn png_first_pixel(path: &Path) -> Result<(u32, u32, [u8; 3]), String> {
    let decoder = png::Decoder::new(std::io::BufReader::new(
        fs::File::open(path).map_err(|err| err.to_string())?,
    ));
    let mut reader = decoder.read_info().map_err(|err| err.to_string())?;
    let mut buf = vec![0_u8; reader.output_buffer_size().ok_or("PNG buffer size")?];
    let frame = reader.next_frame(&mut buf).map_err(|err| err.to_string())?;
    if !matches!(frame.color_type, png::ColorType::Rgb | png::ColorType::Rgba) {
        return Err(format!("unexpected PNG color type {:?}", frame.color_type));
    }
    let rgb = [buf[0], buf[1], buf[2]];
    Ok((frame.width, frame.height, rgb))
}

fn assert_close(actual: u8, expected: u8) {
    assert!(
        actual.abs_diff(expected) <= 30,
        "pixel {actual} diverged from {expected}"
    );
}

#[test]
fn directory_stack_roundtrips_through_frames_for_both_formats() -> Result<(), String> {
    let root = temp_root("directory")?;
    let sources = root.join("frames");
    fs::create_dir_all(&sources).map_err(|err| err.to_string())?;
    for (name, level) in [
        ("frame_1.png", 0_u8),
        ("frame_2.png", 1),
        ("frame_10.png", 2),
    ] {
        write_png(&sources.join(name), 24, 16, &solid(24, 16, level))?;
    }
    // Hidden entries and subdirectories must be skipped.
    write_png(&sources.join(".hidden.png"), 24, 16, &solid(24, 16, 3))?;
    let nested = sources.join("nested");
    fs::create_dir_all(&nested).map_err(|err| err.to_string())?;
    write_png(&nested.join("frame_3.png"), 24, 16, &solid(24, 16, 4))?;

    for (format, extension) in [("gif", "gif"), ("webp", "webp")] {
        let target = root.join(format!("out.{extension}"));
        let summary = stdout(&[
            "stack",
            str_path(&sources)?,
            str_path(&target)?,
            "--fps",
            "10",
            "--threads",
            "2",
        ])?;
        assert!(
            summary.contains("encoded 3 frame(s)") && summary.contains("24x16"),
            "unexpected summary: {summary}"
        );
        assert!(fs::metadata(&target).map_err(|err| err.to_string())?.len() > 0);

        let frames_dir = root.join(format!("extracted_{format}"));
        stdout(&[
            "frames",
            str_path(&target)?,
            str_path(&frames_dir)?,
            "--to",
            "png",
        ])?;
        let frames = extracted_frames(&frames_dir)?;
        assert_eq!(frames.len(), 3, "wrong frame count for .{extension}");
        for (index, path) in frames.iter().enumerate() {
            let (width, height, rgb) = png_first_pixel(path)?;
            assert_eq!((width, height), (24, 16));
            let expected =
                32_u8.saturating_add(u8::try_from(index).unwrap_or(0).saturating_mul(64));
            for channel in rgb {
                assert_close(channel, expected);
            }
        }
        let _ = format;
    }
    fs::remove_dir_all(&root).map_err(|err| err.to_string())?;
    Ok(())
}

#[test]
fn explicit_order_and_duplicate_sources_are_preserved() -> Result<(), String> {
    let root = temp_root("order")?;
    let a = root.join("a.png");
    let b = root.join("b.png");
    write_png(&a, 24, 16, &solid(24, 16, 0))?;
    write_png(&b, 24, 16, &solid(24, 16, 2))?;
    let target = root.join("ordered.webp");
    stdout(&[
        "stack",
        str_path(&b)?,
        str_path(&a)?,
        str_path(&b)?,
        str_path(&target)?,
        "--fps",
        "10",
        "--loop",
        "2",
        "--quality",
        "95",
        "--backend",
        "auto",
    ])?;
    let frames_dir = root.join("extracted");
    stdout(&["frames", str_path(&target)?, str_path(&frames_dir)?])?;
    let frames = extracted_frames(&frames_dir)?;
    assert_eq!(frames.len(), 3);
    for (index, expected_level) in [2_u8, 0, 2].iter().enumerate() {
        let (width, height, rgb) = png_first_pixel(&frames[index])?;
        assert_eq!((width, height), (24, 16));
        let expected = 32_u8.saturating_add(expected_level.saturating_mul(64));
        for channel in rgb {
            assert_close(channel, expected);
        }
    }
    fs::remove_dir_all(&root).map_err(|err| err.to_string())?;
    Ok(())
}

#[test]
fn mixed_png_jpeg_and_qoi_directory_stack_encodes() -> Result<(), String> {
    let root = temp_root("mixed")?;
    let sources = root.join("frames");
    fs::create_dir_all(&sources).map_err(|err| err.to_string())?;
    write_png(&sources.join("01.png"), 24, 16, &solid(24, 16, 0))?;
    stdout(&[
        "convert",
        str_path(&sources.join("01.png"))?,
        str_path(&sources.join("02.jpg"))?,
    ])?;
    stdout(&[
        "convert",
        str_path(&sources.join("01.png"))?,
        str_path(&sources.join("03.qoi"))?,
    ])?;
    let target = root.join("mixed.gif");
    let summary = stdout(&[
        "stack",
        str_path(&sources)?,
        str_path(&target)?,
        "--fps",
        "12",
    ])?;
    assert!(summary.contains("encoded 3 frame(s)"), "{summary}");
    fs::remove_dir_all(&root).map_err(|err| err.to_string())?;
    Ok(())
}

#[test]
fn invalid_flags_fail_before_io() -> Result<(), String> {
    for (args, needle) in [
        (&["stack", "a.png"][..], "expected `fastconv stack"),
        (&["stack", "a.png", "out.gif", "--fps", "0"][..], "--fps"),
        (&["stack", "a.png", "out.gif", "--fps", "-1"][..], "--fps"),
        (&["stack", "a.png", "out.gif", "--fps", "nan"][..], "--fps"),
        (&["stack", "a.png", "out.gif", "--fps", "inf"][..], "--fps"),
        (&["stack", "a.png", "out.gif", "--fps", "1e9"][..], "--fps"),
        (&["stack", "a.png", "out.gif", "--loop", "-1"][..], "--loop"),
        (
            &["stack", "a.png", "out.gif", "--loop", "65536"][..],
            "--loop",
        ),
        (
            &["stack", "a.png", "out.gif", "--loop", "many"][..],
            "--loop",
        ),
        (
            &["stack", "a.png", "out.gif", "--to", "bin"][..],
            "unknown format",
        ),
        (
            &["stack", "a.png", "out.gif", "--backend", "vaapi"][..],
            "no silent CPU fallback",
        ),
        (
            &["stack", "a.png", "out.gif", "--backend", "vulkan"][..],
            "no silent CPU fallback",
        ),
        (
            &["stack", "a.png", "out.gif", "--fps", "10", "--fps", "20"][..],
            "duplicate flag",
        ),
        (
            &["stack", "a.png", "out.gif", "--unknown"][..],
            "unknown flag",
        ),
    ] {
        let stderr = failure(args)?;
        assert!(
            stderr.contains(needle),
            "expected '{needle}' for {args:?}: {stderr}"
        );
    }
    Ok(())
}

#[test]
fn rejects_invalid_sources_and_targets() -> Result<(), String> {
    let root = temp_root("reject")?;
    let a = root.join("a.png");
    let b = root.join("b.png");
    write_png(&a, 24, 16, &solid(24, 16, 0))?;
    write_png(&b, 32, 16, &solid(32, 16, 1))?;
    let a_str = str_path(&a)?;
    let b_str = str_path(&b)?;
    let gif_path = root.join("out.gif");
    let gif = str_path(&gif_path)?;

    assert!(failure(&["stack", a_str, b_str, gif])?.contains("is 32x16"));
    assert!(failure(&["stack", a_str, a_str, "--to", "gif"])?.contains("overwrite"));
    assert!(failure(&["stack", a_str, str_path(&root)?, "--to", "gif"])?.contains("directory"));
    assert!(failure(&["stack", a_str, str_path(&root.join("out.bin"))?])?.contains("mp4"));
    assert!(
        failure(&[
            "stack",
            a_str,
            str_path(&root.join("out.webp"))?,
            "--to",
            "gif"
        ])?
        .contains("conflicts")
    );

    let empty = root.join("empty");
    fs::create_dir_all(&empty).map_err(|err| err.to_string())?;
    assert!(failure(&["stack", str_path(&empty)?, gif])?.contains("no supported images"));

    // A previously encoded multi-frame animation is not a valid still source.
    let animated_path = root.join("animated.gif");
    stdout(&[
        "stack",
        a_str,
        a_str,
        str_path(&animated_path)?,
        "--fps",
        "10",
    ])?;
    let again_path = root.join("again.gif");
    assert!(
        failure(&["stack", str_path(&animated_path)?, str_path(&again_path)?])?
            .contains("animated GIF")
    );

    // WebP's 14-bit canvas limit must be rejected from the header alone.
    let wide = root.join("wide.png");
    write_png(&wide, 16_384, 2, &solid(16_384, 2, 0))?;
    assert!(
        failure(&[
            "stack",
            str_path(&wide)?,
            str_path(&root.join("wide.webp"))?
        ])?
        .contains("static limit")
    );

    fs::remove_dir_all(&root).map_err(|err| err.to_string())?;
    Ok(())
}

#[test]
fn video_stacks_roundtrip_through_frames_for_every_container() -> Result<(), String> {
    let root = temp_root("video-roundtrip")?;
    let sources = root.join("frames");
    fs::create_dir_all(&sources).map_err(|err| err.to_string())?;
    for (name, level) in [
        ("frame_1.png", 0_u8),
        ("frame_2.png", 1),
        ("frame_10.png", 2),
    ] {
        write_png(&sources.join(name), 24, 16, &solid(24, 16, level))?;
    }
    for (container, codec) in [
        ("mp4", "h264"),
        ("mkv", "h264"),
        ("webm", "vp9"),
        ("mov", "h264"),
        ("avi", "h264"),
        ("ts", "h264"),
    ] {
        let target = root.join(format!("out.{container}"));
        let summary = stdout(&[
            "stack",
            str_path(&sources)?,
            str_path(&target)?,
            "--fps",
            "10",
            "--threads",
            "2",
            "--backend",
            "cpu",
        ])?;
        assert!(
            summary.contains(&format!("| .{container} |"))
                && summary.contains(codec)
                && summary.contains("cpu"),
            "unexpected summary for .{container}: {summary}"
        );
        let frames_dir = root.join(format!("extracted_{container}"));
        stdout(&[
            "frames",
            str_path(&target)?,
            str_path(&frames_dir)?,
            "--to",
            "png",
        ])?;
        let frames = extracted_frames(&frames_dir)?;
        assert_eq!(frames.len(), 3, "wrong frame count for .{container}");
        for (index, path) in frames.iter().enumerate() {
            let (width, height, rgb) = png_first_pixel(path)?;
            assert_eq!((width, height), (24, 16));
            let expected =
                32_u8.saturating_add(u8::try_from(index).unwrap_or(0).saturating_mul(64));
            for channel in rgb {
                assert_close(channel, expected);
            }
        }
    }
    fs::remove_dir_all(&root).map_err(|err| err.to_string())?;
    Ok(())
}

fn assert_stack_rejections(cases: &[(&[&str], &[&str])]) -> Result<(), String> {
    for &(args, needles) in cases {
        let stderr = failure(args)?;
        for needle in needles {
            assert!(
                stderr.contains(needle),
                "expected '{needle}' for {args:?}: {stderr}"
            );
        }
    }
    Ok(())
}

#[test]
fn rejects_video_stack_option_and_codec_mismatches() -> Result<(), String> {
    let root = temp_root("video-reject")?;
    let even = root.join("even.png");
    let odd = root.join("odd.png");
    write_png(&even, 24, 16, &solid(24, 16, 0))?;
    write_png(&odd, 25, 16, &solid(25, 16, 0))?;
    let even_str = str_path(&even)?;
    let odd_str = str_path(&odd)?;
    let mp4_path = root.join("out.mp4");
    let gif_path = root.join("out.gif");
    let anim_webp_path = root.join("out.webp");
    let mkv_path = root.join("out.mkv");
    let mp4_str = str_path(&mp4_path)?;
    let gif_str = str_path(&gif_path)?;
    let anim_webp = str_path(&anim_webp_path)?;
    let mkv_str = str_path(&mkv_path)?;

    assert_stack_rejections(&[
        (&["stack", odd_str, mp4_str], &["even dimensions"]),
        (&["stack", even_str, mp4_str, "--loop", "2"], &["--loop"]),
        (
            &["stack", even_str, mp4_str, "--vcodec", "copy"],
            &["--vcodec copy"],
        ),
        (
            &[
                "stack",
                even_str,
                mp4_str,
                "--quality",
                "50",
                "--vbitrate",
                "1M",
            ],
            &["cannot combine"],
        ),
        (
            &[
                "stack",
                even_str,
                mp4_str,
                "--vcodec",
                "h264",
                "--vbitrate",
                "1M",
                "--quality",
                "50",
            ],
            &["cannot combine"],
        ),
        (
            &["stack", even_str, gif_str, "--vbitrate", "1M"],
            &["--vbitrate"],
        ),
        (
            &["stack", even_str, anim_webp, "--vbitrate", "1M"],
            &["--vbitrate"],
        ),
        (
            &[
                "stack",
                even_str,
                mkv_str,
                "--vcodec",
                "mjpeg",
                "--vbitrate",
                "1M",
            ],
            &["--vbitrate"],
        ),
        (
            &[
                "stack",
                even_str,
                mp4_str,
                "--vcodec",
                "hevc",
                "--backend",
                "cpu",
            ],
            &["HEVC"],
        ),
    ])?;
    fs::remove_dir_all(&root).map_err(|err| err.to_string())?;
    Ok(())
}

#[test]
fn rejects_video_stack_bidirectional_codec_mismatches() -> Result<(), String> {
    let root = temp_root("video-reject-codec")?;
    let even = root.join("even.png");
    write_png(&even, 24, 16, &solid(24, 16, 0))?;
    let even_str = str_path(&even)?;
    let mp4_path = root.join("out.mp4");
    let gif_path = root.join("out.gif");
    let webm_target = root.join("out.webm");
    let ts_path = root.join("out.ts");
    let mp4_str = str_path(&mp4_path)?;
    let gif_str = str_path(&gif_path)?;
    let webm_out = str_path(&webm_target)?;
    let ts_str = str_path(&ts_path)?;

    assert_stack_rejections(&[
        (
            &["stack", even_str, webm_out, "--vcodec", "h264"],
            &[
                "WebM",
                "supported video codecs:",
                "containers supporting h264:",
            ],
        ),
        (
            &["stack", even_str, ts_str, "--vcodec", "vp9"],
            &[
                "TS",
                "supported video codecs:",
                "containers supporting vp9:",
            ],
        ),
        (
            &["stack", even_str, mp4_str, "--vcodec", "mjpeg"],
            &[
                "MP4",
                "supported video codecs:",
                "containers supporting mjpeg:",
            ],
        ),
        (
            &["stack", even_str, gif_str, "--vcodec", "h264"],
            &[
                "GIF",
                "supported video codecs:",
                "containers supporting h264:",
            ],
        ),
        (
            &["stack", even_str, mp4_str, "--vcodec", "gif"],
            &[
                "MP4",
                "supported video codecs:",
                "containers supporting gif:",
            ],
        ),
    ])?;
    fs::remove_dir_all(&root).map_err(|err| err.to_string())?;
    Ok(())
}

#[test]
fn rejects_video_stack_unsupported_backends() -> Result<(), String> {
    let root = temp_root("video-reject-backend")?;
    let even = root.join("even.png");
    write_png(&even, 24, 16, &solid(24, 16, 0))?;
    let even_str = str_path(&even)?;
    let mp4_path = root.join("out.mp4");
    let gif_path = root.join("out.gif");
    let anim_webp_path = root.join("out.webp");
    let mkv_path = root.join("out.mkv");
    let webm_target = root.join("out.webm");
    let mp4_str = str_path(&mp4_path)?;
    let gif_str = str_path(&gif_path)?;
    let anim_webp = str_path(&anim_webp_path)?;
    let mkv_str = str_path(&mkv_path)?;
    let webm_out = str_path(&webm_target)?;

    assert_stack_rejections(&[
        (
            &["stack", even_str, gif_str, "--backend", "nvenc"],
            &["no silent CPU fallback"],
        ),
        (
            &["stack", even_str, anim_webp, "--backend", "nvenc"],
            &["no silent CPU fallback"],
        ),
        (
            &[
                "stack",
                even_str,
                mkv_str,
                "--vcodec",
                "av1",
                "--backend",
                "nvenc",
            ],
            &["NVENC"],
        ),
        (
            &[
                "stack",
                even_str,
                webm_out,
                "--vcodec",
                "vp9",
                "--backend",
                "nvenc",
            ],
            &["NVENC"],
        ),
        (
            &[
                "stack",
                even_str,
                mkv_str,
                "--vcodec",
                "mjpeg",
                "--backend",
                "nvenc",
            ],
            &["no silent CPU fallback"],
        ),
        (
            &["stack", even_str, mp4_str, "--backend", "vaapi"],
            &["no silent CPU fallback"],
        ),
        (
            &["stack", even_str, mp4_str, "--backend", "vulkan"],
            &["no silent CPU fallback"],
        ),
    ])?;
    fs::remove_dir_all(&root).map_err(|err| err.to_string())?;
    Ok(())
}
