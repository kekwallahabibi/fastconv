pub mod audio;
pub mod hardware;
pub mod image;
pub mod video;

pub use audio::{
    AudioEncodingRequest, ProbedAudio, parse_audio_bit_depth, parse_audio_bitrate,
    parse_audio_codec, parse_audio_encoding_request, prepare_audio_batch, probe_audio,
};
pub use hardware::detect_host_hardware;
pub use image::{ProbedImage, prepare_image_batch, prepare_image_batch_with_options, probe_image};
pub use video::{
    ProbedVideo, RequestedAudioCodec, RequestedVideoCodec, VideoTranscodeOptions,
    detect_video_magic, parse_requested_audio_codec, prepare_video_batch,
    prepare_video_transcode_batch, prepare_video_transcode_batch_with_decode,
    prepare_video_transcode_batch_with_options, probe_video,
};

use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::thread;

use fastconv_domain::{ConversionRequest, DomainError, HardwareBackend, MediaKind};

/// Errors produced at the system boundary while inspecting files or host capabilities.
#[derive(Debug)]
pub enum ProbeError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    NotAFile(PathBuf),
    NotAFileOrDirectory(PathBuf),
    TargetNotADirectory(PathBuf),
    EmptySourceDirectory(PathBuf),
    EmptyAudioDirectory(PathBuf),
    InvalidImage {
        path: PathBuf,
        reason: String,
    },
    InvalidVideo {
        path: PathBuf,
        reason: String,
    },
    EmptyVideoDirectory(PathBuf),
    InvalidAudio {
        path: PathBuf,
        reason: String,
    },
    UnsupportedTargetFormat {
        path: PathBuf,
        expected: &'static str,
    },
    ParallelismUnavailable(io::Error),
    InvalidDomain(DomainError),
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "failed to inspect '{}': {source}", path.display())
            }
            Self::NotAFile(path) => {
                write!(f, "source path is not a regular file: {}", path.display())
            }
            Self::NotAFileOrDirectory(path) => {
                write!(
                    f,
                    "source path is neither a regular file nor a directory: {}",
                    path.display()
                )
            }
            Self::TargetNotADirectory(path) => {
                write!(
                    f,
                    "target must be a directory when source is a directory: {}",
                    path.display()
                )
            }
            Self::EmptySourceDirectory(path) => {
                write!(
                    f,
                    "no supported images (PNG, JPEG, WebP, BMP, QOI, TIFF, JPEG XL, AVIF, GIF, ICO, or PNM) found in source directory: {}",
                    path.display()
                )
            }
            Self::EmptyAudioDirectory(path) => {
                write!(
                    f,
                    "no supported audio files (WAV, FLAC, MP3, OGG, Opus, AAC, M4A, ALAC, or AIFF) found in source directory: {}",
                    path.display()
                )
            }
            Self::InvalidImage { path, reason } => {
                write!(f, "invalid image '{}': {reason}", path.display())
            }
            Self::InvalidVideo { path, reason } => {
                write!(f, "invalid video '{}': {reason}", path.display())
            }
            Self::EmptyVideoDirectory(path) => {
                write!(
                    f,
                    "no supported video files (MP4, MKV, WebM, MOV, AVI, or TS) found in source directory: {}",
                    path.display()
                )
            }
            Self::InvalidAudio { path, reason } => {
                write!(f, "invalid audio '{}': {reason}", path.display())
            }
            Self::UnsupportedTargetFormat { path, expected } => {
                write!(
                    f,
                    "unsupported target extension for '{}' (expected {expected})",
                    path.display()
                )
            }
            Self::ParallelismUnavailable(source) => {
                write!(f, "failed to determine host CPU parallelism: {source}")
            }
            Self::InvalidDomain(err) => write!(f, "{err}"),
        }
    }
}

impl Error for ProbeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } | Self::ParallelismUnavailable(source) => Some(source),
            Self::InvalidDomain(err) => Some(err),
            Self::NotAFile(_)
            | Self::NotAFileOrDirectory(_)
            | Self::TargetNotADirectory(_)
            | Self::EmptySourceDirectory(_)
            | Self::EmptyAudioDirectory(_)
            | Self::InvalidImage { .. }
            | Self::InvalidVideo { .. }
            | Self::EmptyVideoDirectory(_)
            | Self::InvalidAudio { .. }
            | Self::UnsupportedTargetFormat { .. } => None,
        }
    }
}

impl ProbeError {
    pub(crate) fn requires_to_for_video_directory() -> Self {
        Self::InvalidVideo {
            path: PathBuf::new(),
            reason: "batch video operation requires --to <mp4|mkv|webm|mov|avi|ts> when the target is a directory".to_owned(),
        }
    }
}

impl From<DomainError> for ProbeError {
    fn from(value: DomainError) -> Self {
        Self::InvalidDomain(value)
    }
}

/// Validates at the OS boundary that `path` exists and points to a regular file.
///
/// # Errors
///
/// Returns [`ProbeError::Io`] if metadata cannot be read or [`ProbeError::NotAFile`] if
/// the path exists but is not a regular file.
pub fn validate_source_file(path: &Path) -> Result<PathBuf, ProbeError> {
    let metadata = fs::metadata(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    if !metadata.is_file() {
        return Err(ProbeError::NotAFile(path.to_path_buf()));
    }

    Ok(path.to_path_buf())
}

/// Resolves a path to an absolute location, using the nearest existing ancestor
/// for outputs that do not exist yet. This makes relative/absolute aliases
/// comparable at the boundary, so alias checks can reject in-place overwrites
/// and missing-parent targets still resolve deterministically.
pub(crate) fn canonical_or_parent(path: &Path) -> Result<PathBuf, ProbeError> {
    let mut cursor = path.to_path_buf();
    let mut missing = std::collections::VecDeque::new();
    loop {
        match fs::canonicalize(&cursor) {
            Ok(mut canonical) => {
                for component in missing {
                    canonical = canonical.join(component);
                }
                return Ok(canonical);
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(ProbeError::Io {
                    path: cursor,
                    source,
                });
            }
        }
        match cursor.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => {
                if let Some(name) = cursor.file_name() {
                    missing.push_front(name.to_os_string());
                }
                cursor = parent.to_path_buf();
            }
            _ => {
                let mut canonical = fs::canonicalize(".").map_err(|source| ProbeError::Io {
                    path: PathBuf::from("."),
                    source,
                })?;
                if let Some(name) = cursor.file_name() {
                    missing.push_front(name.to_os_string());
                }
                for component in missing {
                    canonical = canonical.join(component);
                }
                return Ok(canonical);
            }
        }
    }
}

/// Queries the host OS for the available CPU parallelism.
///
/// # Errors
///
/// Returns [`ProbeError::ParallelismUnavailable`] if the operating system does not
/// report available parallelism.
pub fn available_cpu_threads() -> Result<NonZeroUsize, ProbeError> {
    thread::available_parallelism().map_err(ProbeError::ParallelismUnavailable)
}

/// Validates external file inputs once at the boundary and returns a domain request.
///
/// # Errors
///
/// Returns [`ProbeError`] if the source file cannot be validated on disk or if the
/// resulting parameters violate [`ConversionRequest`] domain invariants.
pub fn prepare_request(
    source: &Path,
    target: PathBuf,
    media_kind: MediaKind,
    backend: HardwareBackend,
) -> Result<ConversionRequest, ProbeError> {
    let validated_source = validate_source_file(source)?;
    let request = ConversionRequest::new(validated_source, target, media_kind, backend)?;
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::{ProbeError, available_cpu_threads, validate_source_file};
    use std::path::Path;

    #[test]
    fn accepts_existing_regular_file() -> Result<(), ProbeError> {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let validated = validate_source_file(&manifest)?;
        assert_eq!(validated, manifest);
        Ok(())
    }

    #[test]
    fn rejects_directory_as_source_file() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let result = validate_source_file(dir);
        assert!(matches!(result, Err(ProbeError::NotAFile(_))));
    }

    #[test]
    fn rejects_missing_source_file() {
        let missing = Path::new(env!("CARGO_MANIFEST_DIR")).join("does-not-exist.media");
        let result = validate_source_file(&missing);
        assert!(matches!(result, Err(ProbeError::Io { .. })));
    }

    #[test]
    fn reads_host_cpu_parallelism() -> Result<(), ProbeError> {
        let threads = available_cpu_threads()?;
        assert!(threads.get() >= 1);
        Ok(())
    }
}
