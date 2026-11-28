//! End-to-end checks that container/codec mismatches are rejected with a
//! bidirectional diagnostic before any output file is created. Inputs are
//! synthetic bytes written into a private temp directory: no `fixtures/`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(label: &str) -> Result<Self, String> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| err.to_string())?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "fastconv-cli-validation-{label}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir(&path).map_err(|err| err.to_string())?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if let Err(err) = fs::remove_dir_all(&self.path) {
            eprintln!("failed to clean up temp dir {}: {err}", self.path.display());
        }
    }
}

fn wav_bytes() -> Vec<u8> {
    let samples: Vec<i16> = (0..256)
        .map(|index| i16::try_from(index % 100).unwrap_or(0))
        .collect();
    let mut payload = Vec::new();
    for sample in samples {
        payload.extend_from_slice(&sample.to_le_bytes());
    }
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&1_u16.to_le_bytes()); // PCM
    fmt.extend_from_slice(&1_u16.to_le_bytes()); // mono
    fmt.extend_from_slice(&48_000_u32.to_le_bytes());
    fmt.extend_from_slice(&96_000_u32.to_le_bytes()); // byte rate
    fmt.extend_from_slice(&2_u16.to_le_bytes()); // block align
    fmt.extend_from_slice(&16_u16.to_le_bytes());

    let mut chunks = b"fmt ".to_vec();
    chunks.extend_from_slice(&u32::try_from(fmt.len()).unwrap_or(0).to_le_bytes());
    chunks.extend_from_slice(&fmt);
    chunks.extend_from_slice(b"data");
    chunks.extend_from_slice(&u32::try_from(payload.len()).unwrap_or(0).to_le_bytes());
    chunks.extend_from_slice(&payload);

    let mut bytes = b"RIFF".to_vec();
    bytes.extend_from_slice(
        &u32::try_from(chunks.len().saturating_add(4))
            .unwrap_or(0)
            .to_le_bytes(),
    );
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(&chunks);
    bytes
}

fn riff_chunk(id: [u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8_usize.saturating_add(payload.len()));
    out.extend_from_slice(&id);
    out.extend_from_slice(&u32::try_from(payload.len()).unwrap_or(0).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
    out
}

fn riff_list(kind: [u8; 4], chunks: &[Vec<u8>]) -> Vec<u8> {
    let mut payload = kind.to_vec();
    for chunk in chunks {
        payload.extend_from_slice(chunk);
    }
    riff_chunk(*b"LIST", &payload)
}

fn avi_strh(fcc_type: [u8; 4], handler: [u8; 4], rate: u32, length: u32) -> Vec<u8> {
    let mut body = vec![0_u8; 56];
    body[0..4].copy_from_slice(&fcc_type);
    body[4..8].copy_from_slice(&handler);
    body[20..24].copy_from_slice(&1_u32.to_le_bytes());
    body[24..28].copy_from_slice(&rate.to_le_bytes());
    body[32..36].copy_from_slice(&length.to_le_bytes());
    riff_chunk(*b"strh", &body)
}

fn avi_video_strf(width: u32, height: u32, compression: [u8; 4]) -> Vec<u8> {
    let mut body = vec![0_u8; 40];
    body[4..8].copy_from_slice(&i32::try_from(width).unwrap_or(i32::MAX).to_le_bytes());
    body[8..12].copy_from_slice(&i32::try_from(height).unwrap_or(i32::MAX).to_le_bytes());
    body[16..20].copy_from_slice(&compression);
    riff_chunk(*b"strf", &body)
}

fn avi_audio_strf(format_tag: u16, channels: u16, rate: u32) -> Vec<u8> {
    let mut body = vec![0_u8; 18];
    body[0..2].copy_from_slice(&format_tag.to_le_bytes());
    body[2..4].copy_from_slice(&channels.to_le_bytes());
    body[4..8].copy_from_slice(&rate.to_le_bytes());
    riff_chunk(*b"strf", &body)
}

/// Minimal AVI with an H.264 video track and an AAC audio track, enough for
/// the boundary probe to identify both stream codecs.
fn minimal_avi() -> Vec<u8> {
    let mut avih = vec![0_u8; 56];
    avih[0..4].copy_from_slice(&33_333_u32.to_le_bytes());
    avih[16..20].copy_from_slice(&100_u32.to_le_bytes());
    avih[32..36].copy_from_slice(&640_u32.to_le_bytes());
    avih[36..40].copy_from_slice(&360_u32.to_le_bytes());
    let hdrl = vec![
        riff_chunk(*b"avih", &avih),
        riff_list(
            *b"strl",
            &[
                avi_strh(*b"vids", *b"H264", 30, 100),
                avi_video_strf(640, 360, *b"H264"),
            ],
        ),
        riff_list(
            *b"strl",
            &[
                avi_strh(*b"auds", *b"\0\0\0\0", 48_000, 200),
                avi_audio_strf(0x706D, 2, 48_000),
            ],
        ),
    ];
    let mut body = b"AVI ".to_vec();
    body.extend_from_slice(&riff_list(*b"hdrl", &hdrl));
    let mut bytes = b"RIFF".to_vec();
    bytes.extend_from_slice(&u32::try_from(body.len()).unwrap_or(0).to_le_bytes());
    bytes.extend_from_slice(&body);
    bytes
}

fn run(args: &[&str]) -> Result<(bool, String), String> {
    let output = Command::new(env!("CARGO_BIN_EXE_fastconv"))
        .args(args)
        .output()
        .map_err(|err| err.to_string())?;
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok((output.status.success(), text))
}

fn assert_rejected(
    args: &[&str],
    expected: &[&str],
    absent_output: &Path,
) -> Result<String, String> {
    let (success, text) = run(args)?;
    assert!(!success, "unexpected success: {text}");
    for &needle in expected {
        assert!(text.contains(needle), "missing '{needle}' in: {text}");
    }
    assert!(
        !absent_output.exists(),
        "output was created despite rejection: {}",
        absent_output.display()
    );
    Ok(text)
}

fn write_source(dir: &TempDir, name: &str, bytes: &[u8]) -> Result<PathBuf, String> {
    let path = dir.path().join(name);
    fs::write(&path, bytes).map_err(|err| err.to_string())?;
    Ok(path)
}

#[test]
fn audio_m4a_rejects_mp3_with_both_capabilities() -> Result<(), String> {
    let dir = TempDir::new("m4a-mp3")?;
    let source = write_source(&dir, "in.wav", &wav_bytes())?;
    let target = dir.path().join("out.m4a");
    let source = source.to_str().ok_or("non-UTF-8 source path")?;
    let target = target.to_str().ok_or("non-UTF-8 target path")?;
    assert_rejected(
        &["audio", source, target, "--codec", "mp3"],
        &["aac", "alac", "mp3"],
        Path::new(target),
    )?;
    Ok(())
}

#[test]
fn audio_ogg_rejects_aac_with_both_capabilities() -> Result<(), String> {
    let dir = TempDir::new("ogg-aac")?;
    let source = write_source(&dir, "in.wav", &wav_bytes())?;
    let target = dir.path().join("out.ogg");
    let source = source.to_str().ok_or("non-UTF-8 source path")?;
    let target = target.to_str().ok_or("non-UTF-8 target path")?;
    assert_rejected(
        &["audio", source, target, "--codec", "aac"],
        &["vorbis", "opus", "flac", "aac", "m4a"],
        Path::new(target),
    )?;
    Ok(())
}

#[test]
fn video_webm_rejects_incompatible_audio_and_video_codecs() -> Result<(), String> {
    let dir = TempDir::new("webm-codecs")?;
    let source = write_source(&dir, "in.avi", &minimal_avi())?;
    let source = source.to_str().ok_or("non-UTF-8 source path")?;

    let audio_target = dir.path().join("audio.webm");
    let audio_target = audio_target.to_str().ok_or("non-UTF-8 target path")?;
    assert_rejected(
        &["video", source, audio_target, "--acodec", "aac"],
        &["opus", "vorbis", "mp4", "mov", "mkv", "avi", "ts"],
        Path::new(audio_target),
    )?;

    let video_target = dir.path().join("video.webm");
    let video_target = video_target.to_str().ok_or("non-UTF-8 target path")?;
    assert_rejected(
        &["video", source, video_target, "--vcodec", "h264"],
        &["vp9", "av1", "mp4", "mov", "mkv", "avi", "ts"],
        Path::new(video_target),
    )?;
    Ok(())
}

#[test]
fn video_mp4_rejects_pcm_and_gif_rejects_audio() -> Result<(), String> {
    let dir = TempDir::new("mp4-pcm-gif")?;
    let source = write_source(&dir, "in.avi", &minimal_avi())?;
    let source = source.to_str().ok_or("non-UTF-8 source path")?;

    let mp4 = dir.path().join("out.mp4");
    let mp4 = mp4.to_str().ok_or("non-UTF-8 target path")?;
    assert_rejected(
        &["video", source, mp4, "--acodec", "pcm"],
        &["aac", "opus", "mp3", "flac", "mov", "mkv", "avi"],
        Path::new(mp4),
    )?;

    let gif = dir.path().join("out.gif");
    let gif = gif.to_str().ok_or("non-UTF-8 target path")?;
    assert_rejected(
        &["video", source, gif, "--acodec", "aac"],
        &["GIF supports no audio", "mp4", "mov", "mkv", "avi", "ts"],
        Path::new(gif),
    )?;
    Ok(())
}

#[test]
fn unknown_target_format_distinguishes_video_from_remux() -> Result<(), String> {
    let dir = TempDir::new("unknown-to")?;
    let source = write_source(&dir, "in.avi", &minimal_avi())?;
    let source = source.to_str().ok_or("non-UTF-8 source path")?;
    let target = dir.path().join("out.foo");
    let target = target.to_str().ok_or("non-UTF-8 target path")?;

    let video = assert_rejected(
        &["video", source, target, "--to", "flv"],
        &["mp4", "mkv", "webm", "mov", "avi", "ts", "gif", "webp"],
        Path::new(target),
    )?;
    assert!(video.contains("unknown video format"), "{video}");

    let remux = assert_rejected(
        &["remux", source, target, "--to", "flv"],
        &["mp4", "mkv", "webm", "mov", "avi", "ts"],
        Path::new(target),
    )?;
    assert!(!remux.contains("gif"), "{remux}");
    assert!(!remux.contains("webp"), "{remux}");
    Ok(())
}

#[test]
fn remux_rejects_webm_with_bidirectional_stream_diagnostic() -> Result<(), String> {
    let dir = TempDir::new("remux-webm")?;
    let source = write_source(&dir, "in.avi", &minimal_avi())?;
    let target = dir.path().join("out.webm");
    let source = source.to_str().ok_or("non-UTF-8 source path")?;
    let target = target.to_str().ok_or("non-UTF-8 target path")?;
    assert_rejected(
        &["remux", source, target],
        &["vp9", "av1", "h264", "mp4", "mov", "mkv", "avi", "ts"],
        Path::new(target),
    )?;
    Ok(())
}
