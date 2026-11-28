//! End-to-end boundary checks for `fastconv video --add-audio` /
//! `--replace-audio`. All inputs are synthetic bytes in a private temp dir;
//! no `ffmpeg`/`ffprobe` and no checked-in fixtures are used.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const BIN: &str = env!("CARGO_BIN_EXE_fastconv");

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
            "fastconv-cli-add-audio-{label}-{}-{nanos}",
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
    let samples: Vec<i16> = (0..512)
        .map(|index| i16::try_from(index % 100).unwrap_or(0))
        .collect();
    let mut payload = Vec::new();
    for sample in samples {
        payload.extend_from_slice(&sample.to_le_bytes());
    }
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&1_u16.to_le_bytes()); // PCM
    fmt.extend_from_slice(&1_u16.to_le_bytes()); // mono
    fmt.extend_from_slice(&44_100_u32.to_le_bytes());
    fmt.extend_from_slice(&88_200_u32.to_le_bytes());
    fmt.extend_from_slice(&2_u16.to_le_bytes());
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

fn run(args: &[&str]) -> Result<(bool, String), String> {
    let output = Command::new(BIN)
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

fn fixtures(dir: &TempDir) -> Result<(String, String, String), String> {
    let source = dir.path().join("in.mp4");
    fs::write(
        &source,
        b"not a real mp4, but enough for upfront validation",
    )
    .map_err(|err| err.to_string())?;
    let audio = dir.path().join("song.wav");
    fs::write(&audio, wav_bytes()).map_err(|err| err.to_string())?;
    Ok((
        source.to_str().ok_or("non-UTF-8 source")?.to_owned(),
        audio.to_str().ok_or("non-UTF-8 audio")?.to_owned(),
        dir.path()
            .join("out.mp4")
            .to_str()
            .ok_or("non-UTF-8 target")?
            .to_owned(),
    ))
}

#[test]
fn replace_audio_requires_add_audio() -> Result<(), String> {
    let dir = TempDir::new("replace-without-add")?;
    let (source, _audio, target) = fixtures(&dir)?;
    let text = assert_rejected(
        &[
            "video",
            &source,
            &target,
            "--replace-audio",
            "--vcodec",
            "copy",
        ],
        &["--replace-audio requires --add-audio"],
        Path::new(&target),
    )?;
    assert!(text.contains("error:"));
    Ok(())
}

#[test]
fn add_audio_rejects_copy_and_none_before_output() -> Result<(), String> {
    let dir = TempDir::new("conflicting-acodec")?;
    let (source, audio, target) = fixtures(&dir)?;
    for (codec, needle) in [
        ("copy", "--add-audio cannot be combined with --acodec copy"),
        ("none", "--add-audio cannot be combined with --acodec none"),
    ] {
        assert_rejected(
            &[
                "video",
                &source,
                &target,
                "--add-audio",
                &audio,
                "--acodec",
                codec,
            ],
            &[needle],
            Path::new(&target),
        )?;
    }
    Ok(())
}

#[test]
fn add_audio_rejects_directory_batches_before_creating_target_dir() -> Result<(), String> {
    let dir = TempDir::new("directory-batch")?;
    let source_dir = dir.path().join("clips");
    fs::create_dir(&source_dir).map_err(|err| err.to_string())?;
    let audio = dir.path().join("song.wav");
    fs::write(&audio, wav_bytes()).map_err(|err| err.to_string())?;
    let target_dir = dir.path().join("created-by-failed-request");
    let source = source_dir.to_str().ok_or("non-UTF-8 source")?;
    let audio = audio.to_str().ok_or("non-UTF-8 audio")?;
    let target = target_dir.to_str().ok_or("non-UTF-8 target")?;
    assert_rejected(
        &["video", source, target, "--to", "mp4", "--add-audio", audio],
        &["--add-audio does not support directory batches"],
        &target_dir,
    )?;
    assert!(!target_dir.exists());
    Ok(())
}

#[test]
fn add_audio_rejects_animation_containers_without_a_codec() -> Result<(), String> {
    let dir = TempDir::new("animation")?;
    let (source, audio, _target) = fixtures(&dir)?;
    let target = dir.path().join("out.gif");
    assert_rejected(
        &[
            "video",
            &source,
            target.to_str().ok_or("non-UTF-8 target")?,
            "--to",
            "gif",
            "--add-audio",
            &audio,
        ],
        &["GIF supports no audio", "aac", "mp4", "mkv"],
        &target,
    )?;
    Ok(())
}

#[cfg(unix)]
#[test]
fn add_audio_rejects_target_aliasing_the_audio_source() -> Result<(), String> {
    let dir = TempDir::new("alias-audio")?;
    let (source, audio, target) = fixtures(&dir)?;
    // The target has a video extension but resolves onto the external audio file.
    let alias = dir.path().join("alias.mp4");
    std::os::unix::fs::symlink(&audio, &alias).map_err(|err| err.to_string())?;
    assert_rejected(
        &[
            "video",
            &source,
            alias.to_str().ok_or("non-UTF-8 alias")?,
            "--add-audio",
            &audio,
        ],
        &["source and target path must differ"],
        Path::new(&target),
    )?;
    Ok(())
}

#[test]
fn add_audio_rejects_identical_target_and_external_audio_path() -> Result<(), String> {
    let dir = TempDir::new("same-audio-target")?;
    let (source, _audio, _target) = fixtures(&dir)?;
    let same = dir.path().join("same.mp4");
    fs::write(&same, wav_bytes()).map_err(|err| err.to_string())?;
    let same_str = same.to_str().ok_or("non-UTF-8 path")?;
    let (success, text) = run(&["video", &source, same_str, "--add-audio", same_str])?;
    assert!(!success, "unexpected success: {text}");
    assert!(
        text.contains("source and target path must differ"),
        "unexpected error: {text}"
    );
    Ok(())
}

#[test]
fn duplicate_and_missing_add_audio_flags_fail() -> Result<(), String> {
    let dir = TempDir::new("flag-validation")?;
    let (source, audio, target) = fixtures(&dir)?;
    for args in [
        vec![
            "video",
            &source,
            &target,
            "--add-audio",
            &audio,
            "--add-audio",
            &audio,
        ],
        vec![
            "video",
            &source,
            &target,
            "--replace-audio",
            "--replace-audio",
            "--add-audio",
            &audio,
        ],
        vec!["video", &source, &target, "--add-audio"],
        vec!["video", &source, &target, "--add-audio", ""],
    ] {
        let (success, text) = run(&args)?;
        assert!(!success, "unexpected success for {args:?}: {text}");
        assert!(
            text.contains("duplicate")
                || text.contains("missing value")
                || text.contains("non-empty"),
            "unexpected diagnostic for {args:?}: {text}"
        );
    }
    Ok(())
}
