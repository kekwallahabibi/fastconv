use std::fs::{self, File};
use std::io::{self, Read};
use std::num::{NonZeroU16, NonZeroU32, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::thread;

use fastconv_domain::{
    AudioBatchItem, AudioBatchPlan, AudioExecutionMode, AudioInputFormat, AudioOutputFormat,
    AudioStreamInfo, DomainError, HardwareBackend, PathRole,
};
use symphonia::core::formats::FormatOptions;
use symphonia::core::formats::TrackType;
use symphonia::core::formats::probe::Hint;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

use crate::ProbeError;

const MAGIC_PREFIX_BYTES: usize = 4096;

/// Validated input metadata produced at the file-system boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbedAudio {
    pub path: PathBuf,
    pub format: AudioInputFormat,
    pub stream: AudioStreamInfo,
}

fn invalid_audio(path: &Path, reason: impl Into<String>) -> ProbeError {
    ProbeError::InvalidAudio {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

/// Detects the container from leading bytes only; file extensions are never trusted.
#[must_use]
pub fn detect_audio_magic(bytes: &[u8]) -> Option<AudioInputFormat> {
    if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return Some(AudioInputFormat::Wav);
    }
    if bytes.starts_with(b"fLaC") {
        return Some(AudioInputFormat::Flac);
    }
    if bytes.starts_with(b"ID3")
        || (bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] & 0xE0 == 0xE0)
    {
        return Some(AudioInputFormat::Mp3);
    }
    if bytes.starts_with(b"OggS") {
        if contains_marker(bytes, b"OpusHead") {
            return Some(AudioInputFormat::Opus);
        }
        if contains_marker(bytes, b"\x01vorbis") {
            return Some(AudioInputFormat::Vorbis);
        }
    }
    None
}

fn contains_marker(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack.len() >= needle.len()
        && haystack.windows(needle.len()).any(|w| w == needle)
}

fn read_magic_prefix(path: &Path) -> Result<[u8; MAGIC_PREFIX_BYTES], ProbeError> {
    let mut file = File::open(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut prefix = [0_u8; MAGIC_PREFIX_BYTES];
    let mut filled = 0_usize;
    while filled < prefix.len() {
        match file.read(&mut prefix[filled..]) {
            Ok(0) => break,
            Ok(count) => filled += count,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(source) => {
                return Err(ProbeError::Io {
                    path: path.to_path_buf(),
                    source,
                });
            }
        }
    }
    Ok(prefix)
}

fn probe_opus(path: &Path) -> Result<AudioStreamInfo, ProbeError> {
    let file = File::open(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let reader = opus_pure::OggOpusReader::new(file)
        .map_err(|err| invalid_audio(path, format!("invalid Ogg Opus stream: {err}")))?;
    let head = reader.head();
    let channels = NonZeroU16::new(u16::from(head.channel_count))
        .ok_or_else(|| invalid_audio(path, "Opus channel count must not be zero"))?;
    // Ogg Opus always decodes at 48 kHz, regardless of the rate recorded at capture time.
    let sample_rate = NonZeroU32::new(48_000).unwrap_or(NonZeroU32::MIN);
    AudioStreamInfo::new(channels, sample_rate, NonZeroU16::new(16), None)
        .map_err(|err| invalid_audio(path, err.to_string()))
}

fn probe_symphonia(path: &Path, format: AudioInputFormat) -> Result<AudioStreamInfo, ProbeError> {
    let file = File::open(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
    let mut hint = Hint::new();
    hint.with_extension(format.extension());
    let probed = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|err| invalid_audio(path, format!("failed to read audio headers: {err}")))?;
    let track = probed
        .default_track(TrackType::Audio)
        .ok_or_else(|| invalid_audio(path, "no audio track found"))?;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| invalid_audio(path, "audio track has no codec parameters"))?;
    let channels = params
        .channels
        .as_ref()
        .and_then(|c| NonZeroU16::new(u16::try_from(c.count()).unwrap_or(u16::MAX)))
        .ok_or_else(|| invalid_audio(path, "audio channel count must not be zero"))?;
    let sample_rate = params
        .sample_rate
        .and_then(NonZeroU32::new)
        .ok_or_else(|| invalid_audio(path, "audio sample rate must not be zero"))?;
    let bits_per_sample = params
        .bits_per_sample
        .and_then(|bits| NonZeroU16::new(u16::try_from(bits).unwrap_or(u16::MAX)));
    AudioStreamInfo::new(channels, sample_rate, bits_per_sample, track.num_frames)
        .map_err(|err| invalid_audio(path, err.to_string()))
}

/// Probes one file: detects its container by magic bytes and validates stream metadata.
///
/// # Errors
///
/// Returns [`ProbeError`] for unreadable files, unrecognized magic bytes, or invalid metadata.
pub fn probe_audio(path: &Path) -> Result<ProbedAudio, ProbeError> {
    let metadata = fs::metadata(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(ProbeError::NotAFile(path.to_path_buf()));
    }
    if metadata.len() == 0 {
        return Err(invalid_audio(path, "audio file is empty"));
    }

    let prefix = read_magic_prefix(path)?;
    let format = detect_audio_magic(&prefix).ok_or_else(|| {
        invalid_audio(
            path,
            "unrecognized audio magic bytes (expected WAV, FLAC, MP3, Ogg Vorbis, or Ogg Opus)",
        )
    })?;
    let stream = if format == AudioInputFormat::Opus {
        probe_opus(path)?
    } else {
        probe_symphonia(path, format)?
    };

    Ok(ProbedAudio {
        path: path.to_path_buf(),
        format,
        stream,
    })
}

fn resolve_single_target(
    source: &Path,
    target: &Path,
    to: Option<AudioOutputFormat>,
) -> Result<(PathBuf, AudioOutputFormat), ProbeError> {
    let target_is_dir = target.is_dir()
        || target
            .to_str()
            .is_some_and(|s| s.ends_with('/') || s.ends_with('\\'));
    if target_is_dir {
        let format = to.unwrap_or(AudioOutputFormat::Flac);
        let stem = source
            .file_stem()
            .ok_or_else(|| invalid_audio(source, "source path has no file stem"))?;
        let mut name = stem.to_os_string();
        name.push(".");
        name.push(format.extension());
        return Ok((target.join(name), format));
    }

    let ext = target.extension().and_then(|e| e.to_str()).ok_or_else(|| {
        ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: "flac, wav, or opus",
        }
    })?;
    let inferred = AudioOutputFormat::from_extension(ext).ok_or_else(|| {
        ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: "flac, wav, or opus",
        }
    })?;
    if let Some(explicit) = to
        && explicit != inferred
    {
        return Err(ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: explicit.extension(),
        });
    }
    Ok((target.to_path_buf(), inferred))
}

fn prepare_directory_batch(
    source_dir: &Path,
    target_dir: &Path,
    format: AudioOutputFormat,
    threads: NonZeroUsize,
) -> Result<AudioBatchPlan, ProbeError> {
    match fs::metadata(target_dir) {
        Ok(meta) if meta.is_dir() => {}
        Ok(_) => return Err(ProbeError::TargetNotADirectory(target_dir.to_path_buf())),
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(ProbeError::Io {
                path: target_dir.to_path_buf(),
                source,
            });
        }
    }

    if canonical_or_parent(source_dir)? == canonical_or_parent(target_dir)? {
        return Err(DomainError::IdenticalSourceAndTarget(source_dir.to_path_buf()).into());
    }

    let mut candidates = Vec::new();
    for entry in fs::read_dir(source_dir).map_err(|source| ProbeError::Io {
        path: source_dir.to_path_buf(),
        source,
    })? {
        let path = entry
            .map_err(|source| ProbeError::Io {
                path: source_dir.to_path_buf(),
                source,
            })?
            .path();
        let is_hidden = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with('.'));
        let has_audio_extension = path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
            matches!(
                e.to_ascii_lowercase().as_str(),
                "wav" | "wave" | "flac" | "mp3" | "ogg" | "oga" | "opus"
            )
        });
        if path.is_file() && !is_hidden && has_audio_extension {
            candidates.push(path);
        }
    }
    candidates.sort();

    let mut items = Vec::with_capacity(candidates.len());
    for path in candidates {
        let probed = probe_audio(&path)?;
        let stem = path
            .file_stem()
            .ok_or_else(|| invalid_audio(&path, "source path has no file stem"))?;
        let mut name = stem.to_os_string();
        name.push(".");
        name.push(format.extension());
        items.push(AudioBatchItem::new(
            canonical_or_parent(&path)?,
            target_dir.join(name),
            probed.format,
            probed.stream,
        )?);
    }
    if items.is_empty() {
        return Err(ProbeError::EmptyAudioDirectory(source_dir.to_path_buf()));
    }

    let mode = AudioExecutionMode::resolve(format, HardwareBackend::Cpu)?;
    Ok(AudioBatchPlan::new(items, mode, threads)?)
}

/// Resolves a path to an absolute location, using the parent directory for outputs that do not
/// exist yet. This makes relative/absolute aliases comparable at the boundary.
fn canonical_or_parent(path: &Path) -> Result<PathBuf, ProbeError> {
    if let Ok(canonical) = fs::canonicalize(path) {
        return Ok(canonical);
    }
    let parent = path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = fs::canonicalize(parent).map_err(|source| ProbeError::Io {
        path: parent.to_path_buf(),
        source,
    })?;
    let name = path
        .file_name()
        .ok_or_else(|| invalid_audio(path, "path has no file name"))?;
    Ok(parent.join(name))
}

/// Parses external paths and options once, then returns a domain [`AudioBatchPlan`].
///
/// # Errors
///
/// Returns [`ProbeError`] for invalid paths, unsupported formats/backends, unreadable inputs,
/// or conflicting batch targets. GPU backends are rejected explicitly without CPU fallback.
pub fn prepare_audio_batch(
    source: &Path,
    target: &Path,
    to: Option<AudioOutputFormat>,
    worker_threads: Option<NonZeroUsize>,
    backend: HardwareBackend,
) -> Result<AudioBatchPlan, ProbeError> {
    if source.as_os_str().is_empty() {
        return Err(DomainError::EmptyPath(PathRole::Source).into());
    }
    if target.as_os_str().is_empty() {
        return Err(DomainError::EmptyPath(PathRole::Target).into());
    }

    let threads = match worker_threads {
        Some(count) => count,
        None => thread::available_parallelism().map_err(ProbeError::ParallelismUnavailable)?,
    };
    let source_meta = fs::metadata(source).map_err(|err| ProbeError::Io {
        path: source.to_path_buf(),
        source: err,
    })?;

    if source_meta.is_dir() {
        let format = to.unwrap_or(AudioOutputFormat::Flac);
        AudioExecutionMode::resolve(format, backend)?;
        return prepare_directory_batch(source, target, format, threads);
    }
    if !source_meta.is_file() {
        return Err(ProbeError::NotAFileOrDirectory(source.to_path_buf()));
    }

    let (resolved_target, format) = resolve_single_target(source, target, to)?;
    let canonical_source = canonical_or_parent(source)?;
    let canonical_target = canonical_or_parent(&resolved_target)?;
    if canonical_source == canonical_target {
        return Err(DomainError::IdenticalSourceAndTarget(source.to_path_buf()).into());
    }
    let mode = AudioExecutionMode::resolve(format, backend)?;
    let probed = probe_audio(source)?;
    let item = AudioBatchItem::new(
        canonical_source,
        resolved_target,
        probed.format,
        probed.stream,
    )?;
    Ok(AudioBatchPlan::new(vec![item], mode, threads)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_magic_without_extension_help() {
        let mut wav = vec![0_u8; 12];
        wav[..4].copy_from_slice(b"RIFF");
        wav[8..12].copy_from_slice(b"WAVE");
        assert_eq!(detect_audio_magic(&wav), Some(AudioInputFormat::Wav));
        assert_eq!(
            detect_audio_magic(b"fLaC\x00\x00\x00\x22"),
            Some(AudioInputFormat::Flac)
        );
        assert_eq!(
            detect_audio_magic(b"ID3\x04\x00\x00"),
            Some(AudioInputFormat::Mp3)
        );
        assert_eq!(
            detect_audio_magic(&[0xFF, 0xFB, 0x90, 0x64]),
            Some(AudioInputFormat::Mp3)
        );
        assert_eq!(
            detect_audio_magic(b"OggS\x00\x02\x01vorbis"),
            Some(AudioInputFormat::Vorbis)
        );
        assert_eq!(
            detect_audio_magic(b"OggS\x00\x02OpusHead"),
            Some(AudioInputFormat::Opus)
        );
        assert_eq!(detect_audio_magic(b"UNKNOWN"), None);
    }

    #[test]
    fn resolves_single_target_extension_and_directory() {
        let source = Path::new("track.wav");
        let (file, format) = resolve_single_target(source, Path::new("out.flac"), None)
            .unwrap_or_else(|_| (PathBuf::new(), AudioOutputFormat::Wav));
        assert_eq!(file, PathBuf::from("out.flac"));
        assert_eq!(format, AudioOutputFormat::Flac);
        assert!(
            resolve_single_target(source, Path::new("out.opus"), Some(AudioOutputFormat::Flac))
                .is_err()
        );

        let (dir_target, format) =
            resolve_single_target(source, Path::new("dist/"), Some(AudioOutputFormat::Opus))
                .unwrap_or_else(|_| (PathBuf::new(), AudioOutputFormat::Wav));
        assert_eq!(dir_target, PathBuf::from("dist/track.opus"));
        assert_eq!(format, AudioOutputFormat::Opus);
    }

    #[test]
    fn rejects_relative_and_absolute_path_aliases() -> Result<(), String> {
        let cwd = std::env::current_dir().map_err(|err| err.to_string())?;
        let dir = cwd.join(format!("target/audio-alias-{}", std::process::id()));
        fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
        let absolute = dir.join("track.wav");
        fs::write(&absolute, b"RIFF").map_err(|err| err.to_string())?;
        let relative = absolute
            .strip_prefix(&cwd)
            .unwrap_or(&absolute)
            .to_path_buf();
        let result = prepare_audio_batch(&relative, &absolute, None, None, HardwareBackend::Cpu);
        let _ = fs::remove_dir_all(&dir);
        assert!(matches!(
            result,
            Err(ProbeError::InvalidDomain(
                DomainError::IdenticalSourceAndTarget(_)
            ))
        ));
        Ok(())
    }

    #[test]
    fn rejects_gpu_backend_before_building_a_plan() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let result = prepare_audio_batch(
            &source,
            Path::new("/tmp/fastconv-audio-gpu.flac"),
            Some(AudioOutputFormat::Flac),
            None,
            HardwareBackend::Gpu(fastconv_domain::GpuBackend::Nvenc),
        );
        assert!(matches!(
            result,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedAudioPipeline { .. }
            ))
        ));
    }
}
