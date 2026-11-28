//! End-to-end extraction from video containers through the real binary.
//! Fixtures are generated in-process; no `ffmpeg`/`ffprobe` is spawned.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
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
        let path = std::env::temp_dir().join(format!("fastconv-cli-extract-{label}-{nanos}"));
        fs::create_dir(&path).map_err(|err| err.to_string())?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn chunk(id: [u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.extend_from_slice(&(u32::try_from(payload.len()).unwrap_or(0)).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
    out
}

fn list(kind: [u8; 4], chunks: &[Vec<u8>]) -> Vec<u8> {
    let mut payload = kind.to_vec();
    for item in chunks {
        payload.extend_from_slice(item);
    }
    chunk(*b"LIST", &payload)
}

/// Builds a real AVI with one 16-bit PCM audio stream per format tag. Track 0
/// holds a positive tone, track 1 a negative one, so extraction can prove which
/// track was selected without relying on codec-specific probing. An unknown
/// format tag produces a stream the demuxer cannot decode.
fn synthetic_avi(format_tags: &[u16], frames: usize) -> Vec<u8> {
    let mut avih = vec![0_u8; 56];
    avih[0..4].copy_from_slice(&20_000_u32.to_le_bytes());
    avih[16..20].copy_from_slice(&(u32::try_from(frames).unwrap_or(0)).to_le_bytes());
    avih[24..28].copy_from_slice(&(u32::try_from(format_tags.len()).unwrap_or(0)).to_le_bytes());

    let mut strls = Vec::new();
    for &format_tag in format_tags {
        let mut strh = vec![0_u8; 56];
        strh[0..4].copy_from_slice(b"auds");
        strh[20..24].copy_from_slice(&1_u32.to_le_bytes());
        strh[24..28].copy_from_slice(&48_000_u32.to_le_bytes());
        strh[32..36].copy_from_slice(&(u32::try_from(frames).unwrap_or(0)).to_le_bytes());
        strh[44..48].copy_from_slice(&4_u32.to_le_bytes());

        let mut format_chunk = vec![0_u8; 16];
        format_chunk[0..2].copy_from_slice(&format_tag.to_le_bytes());
        format_chunk[2..4].copy_from_slice(&2_u16.to_le_bytes()); // stereo
        format_chunk[4..8].copy_from_slice(&48_000_u32.to_le_bytes());
        format_chunk[8..12].copy_from_slice(&192_000_u32.to_le_bytes());
        format_chunk[12..14].copy_from_slice(&4_u16.to_le_bytes());
        format_chunk[14..16].copy_from_slice(&16_u16.to_le_bytes());

        strls.push(list(
            *b"strl",
            &[chunk(*b"strh", &strh), chunk(*b"strf", &format_chunk)],
        ));
    }

    let mut hdrl_chunks = vec![chunk(*b"avih", &avih)];
    hdrl_chunks.extend(strls);

    let mut movi_chunks = Vec::new();
    for track in 0..format_tags.len() {
        let id = [
            b'0' + u8::try_from(track / 10).unwrap_or(0),
            b'0' + u8::try_from(track % 10).unwrap_or(0),
            b'w',
            b'b',
        ];
        let sign: i16 = if track == 0 { 12_000 } else { -12_000 };
        let mut data = Vec::with_capacity(frames * 4);
        for index in 0..frames {
            let sample = if index % 2 == 0 { sign } else { -sign };
            data.extend_from_slice(&sample.to_le_bytes());
            data.extend_from_slice(&sample.to_le_bytes());
        }
        movi_chunks.push(chunk(id, &data));
    }

    let mut body = b"AVI ".to_vec();
    body.extend_from_slice(&list(*b"hdrl", &hdrl_chunks));
    body.extend_from_slice(&list(*b"movi", &movi_chunks));
    let mut bytes = b"RIFF".to_vec();
    bytes.extend_from_slice(&(u32::try_from(body.len()).unwrap_or(0)).to_le_bytes());
    bytes.extend_from_slice(&body);
    bytes
}

fn run(args: &[&str]) -> Result<Output, String> {
    Command::new(BIN)
        .args(args)
        .output()
        .map_err(|err| format!("failed to run fastconv: {err}"))
}

fn success(args: &[&str]) -> Result<String, String> {
    let output = run(args)?;
    if !output.status.success() {
        return Err(format!(
            "fastconv {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|err| err.to_string())
}

fn failure(args: &[&str]) -> Result<String, String> {
    let output = run(args)?;
    if output.status.success() {
        return Err(format!(
            "fastconv {} unexpectedly succeeded",
            args.join(" ")
        ));
    }
    let mut message = String::from_utf8_lossy(&output.stderr).into_owned();
    message.push_str(&String::from_utf8_lossy(&output.stdout));
    Ok(message)
}

/// Extracts the `data` chunk from a canonical 16-bit PCM WAV file.
fn wav_samples(path: &Path) -> Result<Vec<i16>, String> {
    let bytes = fs::read(path).map_err(|err| err.to_string())?;
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("not a WAV file: {}", path.display()));
    }
    let mut offset = 12_usize;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let len = usize::try_from(u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]))
        .unwrap_or(0);
        let start = offset + 8;
        let end = start.saturating_add(len).min(bytes.len());
        if id == b"data" {
            return Ok(bytes[start..end]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
                .collect());
        }
        offset = end + (len % 2);
    }
    Err(format!("no data chunk in {}", path.display()))
}

#[test]
fn extracts_avi_audio_track_into_wav_and_flac() -> Result<(), String> {
    let dir = TempDir::new("avi-to-audio")?;
    let source = dir.path().join("in.avi");
    fs::write(&source, synthetic_avi(&[1], 4_000)).map_err(|err| err.to_string())?;

    for (name, extension) in [("out.wav", "wav"), ("out.flac", "flac")] {
        let target = dir.path().join(name);
        success(&[
            "audio",
            source.to_str().ok_or("non-UTF-8 path")?,
            target.to_str().ok_or("non-UTF-8 path")?,
        ])?;
        assert!(
            fs::metadata(&target).map_err(|err| err.to_string())?.len() > 0,
            "{extension} output is empty"
        );
    }
    let samples = wav_samples(&dir.path().join("out.wav"))?;
    assert!(!samples.is_empty(), "WAV output has no decoded samples");
    Ok(())
}

#[test]
fn track_flag_selects_the_requested_audio_track() -> Result<(), String> {
    let dir = TempDir::new("avi-track-selection")?;
    let source = dir.path().join("in.avi");
    fs::write(&source, synthetic_avi(&[1, 1], 4_000)).map_err(|err| err.to_string())?;
    let source = source.to_str().ok_or("non-UTF-8 path")?;

    let default = dir.path().join("default.wav");
    success(&["audio", source, default.to_str().ok_or("non-UTF-8 path")?])?;
    let second = dir.path().join("second.wav");
    success(&[
        "audio",
        source,
        second.to_str().ok_or("non-UTF-8 path")?,
        "--track",
        "1",
    ])?;
    let first_samples = wav_samples(&default)?;
    let second_samples = wav_samples(&second)?;
    assert!(!first_samples.is_empty() && !second_samples.is_empty());
    assert!(
        first_samples[0] > 0 && second_samples[0] < 0,
        "track 0 and track 1 must carry different signals"
    );
    Ok(())
}

#[test]
fn out_of_range_track_fails_without_creating_target() -> Result<(), String> {
    let dir = TempDir::new("avi-bad-track")?;
    let source = dir.path().join("in.avi");
    fs::write(&source, synthetic_avi(&[1], 2_000)).map_err(|err| err.to_string())?;
    let target = dir.path().join("never.wav");
    let error = failure(&[
        "audio",
        source.to_str().ok_or("non-UTF-8 path")?,
        target.to_str().ok_or("non-UTF-8 path")?,
        "--track",
        "2",
    ])?;
    assert!(
        error.contains("audio track 2 is out of range"),
        "unexpected error: {error}"
    );
    assert!(
        !target.exists(),
        "failed extraction must not create a target"
    );
    Ok(())
}

#[test]
fn incompatible_codec_on_video_source_fails_immediately() -> Result<(), String> {
    let dir = TempDir::new("avi-bad-codec")?;
    let source = dir.path().join("in.avi");
    fs::write(&source, synthetic_avi(&[1], 2_000)).map_err(|err| err.to_string())?;
    let target = dir.path().join("never.m4a");
    let error = failure(&[
        "audio",
        source.to_str().ok_or("non-UTF-8 path")?,
        target.to_str().ok_or("non-UTF-8 path")?,
        "--codec",
        "mp3",
    ])?;
    assert!(
        error.contains("cannot be stored in M4A") && error.contains("containers supporting mp3"),
        "error must report both directions: {error}"
    );
    assert!(
        !target.exists(),
        "failed extraction must not create a target"
    );
    Ok(())
}

#[test]
fn undecodable_audio_track_does_not_consume_a_track_ordinal() -> Result<(), String> {
    let dir = TempDir::new("avi-unknown-first")?;
    let source = dir.path().join("in.avi");
    // Format tag 0x1234 is unknown to both the probe and the demuxer. It must
    // not consume an ordinal: the PCM track that follows is the only extractable
    // track, so it is `--track 0` on both sides of the boundary.
    fs::write(&source, synthetic_avi(&[0x1234, 1], 2_000)).map_err(|err| err.to_string())?;
    let source = source.to_str().ok_or("non-UTF-8 path")?;

    let target = dir.path().join("pcm.wav");
    success(&["audio", source, target.to_str().ok_or("non-UTF-8 path")?])?;
    assert_ne!(wav_samples(&target)?, [] as [i16; 0]);

    let rejected = dir.path().join("never.wav");
    let error = failure(&[
        "audio",
        source,
        rejected.to_str().ok_or("non-UTF-8 path")?,
        "--track",
        "1",
    ])?;
    assert!(
        error.contains("audio track 1 is out of range"),
        "unexpected error: {error}"
    );
    assert!(!rejected.exists());
    Ok(())
}
