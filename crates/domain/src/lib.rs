pub mod audio;
pub mod hardware;
pub mod image;
pub mod metrics;
pub mod video;

pub use audio::{
    AudioBatchItem, AudioBatchPlan, AudioBitDepth, AudioConversionReport, AudioEncodingOptions,
    AudioExecutionMode, AudioInputFormat, AudioOptionBitrate, AudioOptionChannels,
    AudioOptionSampleRate, AudioOutputFormat, AudioQuality, AudioRateControl, AudioSampleFormat,
    AudioStreamInfo, AudioTargetCodec, PcmWavLayout, aac_frame_capacity_bps, fnv1a64,
};
pub use hardware::{
    CpuInfo, DetectedGpuBackend, GpuAdapterInfo, GpuCapability, GpuVendor, HostHardware, SimdLevel,
};
pub use image::{
    ChromaSubsampling, CompressionLevel, ImageBatchItem, ImageBatchPlan, ImageCompression,
    ImageConversionReport, ImageDimensions, ImageEncodeOptions, ImageExecutionMode,
    ImageInputFormat, ImageOutputFormat, ImageQuality, TiffCompression, TiffCompressionKind,
};
pub use metrics::{BenchmarkMetrics, BenchmarkSample, PeakMemory, Throughput, WorkloadUnit};
pub use video::{
    AnimationLoop, AnimationMeta, AudioBitrate, AudioChannels, AudioEncodeConfig,
    AudioEncodeOptions, AudioSampleRate, AudioTrackAction, HardwareDecodeBackend, TargetAudioCodec,
    TargetVideoCodec, VideoBitrate, VideoCodec, VideoContainerFormat, VideoDecodeMode,
    VideoEncodeMode, VideoQuality, VideoRateControl, VideoRemuxItem, VideoRemuxPlan,
    VideoRemuxReport, VideoStreamInfo, VideoStreamKind, VideoTrackAction, VideoTranscodeItem,
    VideoTranscodePlan, VideoTranscodeReport, resolve_audio_encode_options,
    resolve_video_rate_control, validate_audio_options_for_container,
};

use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

#[cfg(feature = "serde")]
use serde::Serialize;

/// Deterministic `u64` -> `f64` conversion for throughput/quality reporting.
/// Splitting into two `u32` halves keeps every crate's rounded results identical.
#[must_use]
pub fn u64_to_f64(value: u64) -> f64 {
    let hi = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let lo = u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(hi) * 4_294_967_296.0 + f64::from(lo)
}

/// High-level media category supported by `fastconv`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum MediaKind {
    Video,
    Audio,
    Image,
}

/// Subset of media kinds eligible for zero-copy GPU execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum GpuMediaKind {
    Video,
    Image,
}

/// Concrete hardware acceleration backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum GpuBackend {
    Nvenc,
    Vaapi,
    Qsv,
    Amf,
    VideoToolbox,
    Vulkan,
}

/// Execution backend explicitly selected for a conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum HardwareBackend {
    Cpu,
    Gpu(GpuBackend),
}

/// Identifies which path field failed domain validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PathRole {
    Source,
    Target,
}

/// Pure domain validation and planning errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DomainError {
    EmptyPath(PathRole),
    EmptyField(&'static str),
    IdenticalSourceAndTarget(PathBuf),
    InvalidCpuTopology {
        physical_cores: NonZeroUsize,
        logical_threads: NonZeroUsize,
    },
    ZeroDuration,
    InvalidDimensions {
        width: u32,
        height: u32,
    },
    InvalidQuality {
        value: u8,
        max: u8,
    },
    EmptyBatch,
    DuplicateTargetPath(PathBuf),
    TargetOverwritesSource(PathBuf),
    UnsupportedBackend {
        media_kind: MediaKind,
        backend: HardwareBackend,
    },
    UnsupportedImagePipeline {
        format: ImageOutputFormat,
        backend: HardwareBackend,
        reason: String,
    },
    UnsupportedVideoCodec {
        codec: VideoCodec,
        container: VideoContainerFormat,
    },
    UnsupportedVideoPipeline {
        codec: VideoCodec,
        backend: HardwareBackend,
        reason: String,
    },
    InvalidVideoQuality {
        value: u8,
        max: u8,
    },
    InvalidVideoBitrate {
        value: u32,
        min: u32,
        max: u32,
    },
    InvalidAudioBitrate {
        value: u32,
        min: u32,
        max: u32,
    },
    InvalidBitrateSyntax {
        raw: String,
    },
    InvalidAudioSampleRate {
        value: u32,
    },
    InvalidAudioChannels {
        value: u16,
    },
    UnsupportedAudioEncodeOption {
        codec: TargetAudioCodec,
        reason: String,
    },
    ConflictingVideoRateControl {
        reason: String,
    },
    ConflictingAudioEncodeOptions {
        reason: String,
    },
    UnsupportedAudioPipeline {
        format: AudioOutputFormat,
        backend: HardwareBackend,
        reason: String,
    },
    InvalidAudioOptionBitrate {
        raw: String,
        expected: &'static str,
    },
    InvalidAudioQuality {
        value: u8,
        max: u8,
    },
    InvalidAudioOptionSampleRate {
        hz: u32,
        expected: &'static str,
    },
    InvalidAudioOptionChannels {
        channels: u16,
        expected: &'static str,
    },
    IncompatibleAudioBitrateRate {
        bps: u32,
        hz: u32,
        max_bps: u32,
    },
    IncompatibleAacBitrateRate {
        bps: u32,
        hz: u32,
        channels: u16,
        max_bps: u32,
    },
    UnsupportedAudioRateControl {
        format: AudioOutputFormat,
        codec: AudioTargetCodec,
        control: &'static str,
        reason: &'static str,
    },
    InvalidAudioCodec {
        raw: String,
        expected: &'static str,
    },
    UnsupportedAudioCodecForFormat {
        format: AudioOutputFormat,
        codec: AudioTargetCodec,
    },
    InvalidAudioBitDepth {
        raw: String,
        expected: &'static str,
    },
    UnsupportedAudioBitDepth {
        format: AudioOutputFormat,
        codec: AudioTargetCodec,
        bit_depth: AudioBitDepth,
        reason: &'static str,
    },
    InvalidPcmWavLayout {
        reason: &'static str,
    },
}

pub(crate) fn require_non_empty(value: &str, field: &'static str) -> Result<String, DomainError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(DomainError::EmptyField(field));
    }
    Ok(trimmed.to_owned())
}

impl fmt::Display for MediaKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Video => "video",
            Self::Audio => "audio",
            Self::Image => "image",
        };
        f.write_str(label)
    }
}

impl fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Nvenc => "nvenc",
            Self::Vaapi => "vaapi",
            Self::Qsv => "qsv",
            Self::Amf => "amf",
            Self::VideoToolbox => "videotoolbox",
            Self::Vulkan => "vulkan",
        };
        f.write_str(label)
    }
}

impl fmt::Display for HardwareBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Gpu(backend) => write!(f, "gpu:{backend}"),
        }
    }
}

impl fmt::Display for PathRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Source => "source",
            Self::Target => "target",
        };
        f.write_str(label)
    }
}

/// Renders a canonical list of CLI names, using `none` for an empty list so
/// diagnostics never end in an ambiguous dangling separator.
fn cli_name_list<'a>(names: impl IntoIterator<Item = &'a str>) -> String {
    let names: Vec<&str> = names.into_iter().collect();
    if names.is_empty() {
        "none".to_owned()
    } else {
        names.join(", ")
    }
}

/// Reports both directions of a video stream/container mismatch: what the
/// container can carry for this kind of stream and where this codec is legal.
fn fmt_unsupported_video_codec(
    codec: VideoCodec,
    container: VideoContainerFormat,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    let supported = if codec.is_video() {
        format!(
            "supported video codecs: {}",
            cli_name_list(
                container
                    .supported_video_codecs()
                    .iter()
                    .map(|codec| codec.cli_name())
            )
        )
    } else if codec.is_audio() {
        if container.supported_audio_codecs().is_empty() {
            format!("{} supports no audio", container.label())
        } else {
            format!(
                "supported audio codecs: {}",
                cli_name_list(
                    container
                        .supported_audio_codecs()
                        .iter()
                        .map(|codec| codec.cli_name())
                )
            )
        }
    } else if codec == VideoCodec::TimedText {
        if codec.supported_in(container) {
            "supported subtitle codecs: timed-text".to_owned()
        } else {
            format!("{} supports no subtitles", container.label())
        }
    } else {
        // Unknown stream kind: report both codec families so a container's
        // capabilities are always visible.
        format!(
            "supported video codecs: {}; supported audio codecs: {}",
            cli_name_list(
                container
                    .supported_video_codecs()
                    .iter()
                    .map(|codec| codec.cli_name())
            ),
            cli_name_list(
                container
                    .supported_audio_codecs()
                    .iter()
                    .map(|codec| codec.cli_name())
            )
        )
    };
    write!(
        f,
        "codec '{}' cannot be stored in {} ({}; containers supporting {}: {})",
        codec.cli_name(),
        container.label(),
        supported,
        codec.cli_name(),
        cli_name_list(codec.supported_containers().iter().map(|c| c.extension()))
    )
}

impl fmt::Display for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPath(role) => write!(f, "{role} path must not be empty"),
            Self::EmptyField(field) => write!(f, "field '{field}' must not be blank"),
            Self::IdenticalSourceAndTarget(path) => write!(
                f,
                "source and target path must differ to avoid in-place corruption: {}",
                path.display()
            ),
            Self::InvalidCpuTopology {
                physical_cores,
                logical_threads,
            } => write!(
                f,
                "invalid CPU topology: logical threads ({logical_threads}) cannot be fewer than physical cores ({physical_cores})"
            ),
            Self::ZeroDuration => f.write_str("benchmark duration must be greater than zero"),
            Self::InvalidDimensions { width, height } => {
                write!(f, "image dimensions must be non-zero, got {width}x{height}")
            }
            Self::UnsupportedVideoCodec { codec, container } => {
                fmt_unsupported_video_codec(*codec, *container, f)
            }
            Self::UnsupportedVideoPipeline {
                codec,
                backend,
                reason,
            } => write!(
                f,
                "backend '{backend}' cannot encode video codec '{codec}': {reason}"
            ),
            Self::InvalidVideoQuality { value, max } => {
                write!(f, "video quality must be in 0..={max}, got {value}")
            }
            Self::InvalidQuality { value, max } => {
                write!(f, "image quality must be in 0..={max}, got {value}")
            }
            Self::EmptyBatch => {
                f.write_str("image conversion batch must contain at least one file")
            }
            Self::DuplicateTargetPath(path) => write!(
                f,
                "multiple batch items target the same output path: {}",
                path.display()
            ),
            Self::TargetOverwritesSource(path) => write!(
                f,
                "batch output path would overwrite a source file: {}",
                path.display()
            ),
            Self::UnsupportedBackend {
                media_kind,
                backend,
            } => write!(
                f,
                "backend '{backend}' is not supported for {media_kind} conversions"
            ),
            Self::UnsupportedImagePipeline {
                format,
                backend,
                reason,
            } => write!(
                f,
                "backend '{backend}' cannot encode .{format} images: {reason}"
            ),
            Self::UnsupportedAudioPipeline {
                format,
                backend,
                reason,
            } => write!(
                f,
                "backend '{backend}' cannot encode {format} audio: {reason}"
            ),
            Self::InvalidAudioOptionBitrate { raw, expected } => {
                write!(f, "invalid audio bitrate '{raw}' (expected {expected})")
            }
            Self::InvalidAudioQuality { value, max } => {
                write!(f, "audio quality must be in 0..={max}, got {value}")
            }
            Self::UnsupportedAudioRateControl {
                format,
                codec,
                control,
                reason,
            } => write!(
                f,
                "{control} is not supported for {} audio: {reason}",
                format.resolved_label(*codec)
            ),
            _ => fmt_encoding_error(self, f),
        }
    }
}

/// Formats rate-control and audio-option validation errors. The caller's guard
/// is the only path into this helper, so any other variant is a formatter bug
/// and surfaces as `fmt::Error` rather than a misleading message.
fn fmt_encoding_error(err: &DomainError, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match err {
        DomainError::InvalidVideoBitrate { value, min, max } => {
            write!(f, "video bitrate must be in {min}..={max} bps, got {value}")
        }
        DomainError::InvalidAudioBitrate { value, min, max } => {
            write!(f, "audio bitrate must be in {min}..={max} bps, got {value}")
        }
        DomainError::InvalidBitrateSyntax { raw } => write!(
            f,
            "invalid bitrate '{raw}' (expected e.g. 500k, 2.5M or 5000000)"
        ),
        DomainError::InvalidAudioSampleRate { value } => {
            write!(f, "audio sample rate must be non-zero, got {value}")
        }
        DomainError::InvalidAudioChannels { value } => {
            write!(f, "audio channel count must be non-zero, got {value}")
        }
        DomainError::UnsupportedAudioEncodeOption { codec, reason } => {
            write!(
                f,
                "{codec} encoding rejects requested audio options: {reason}"
            )
        }
        DomainError::ConflictingVideoRateControl { reason } => {
            write!(f, "conflicting video rate-control options: {reason}")
        }
        DomainError::ConflictingAudioEncodeOptions { reason } => {
            write!(f, "conflicting audio encoding options: {reason}")
        }
        DomainError::IncompatibleAacBitrateRate {
            bps,
            hz,
            channels,
            max_bps,
        } => write!(
            f,
            "audio bitrate {bps} bps at {hz} Hz with {channels} channels exceeds the AAC frame capacity of {max_bps} bps"
        ),
        DomainError::InvalidAudioOptionSampleRate { hz, expected } => {
            write!(f, "invalid audio sample rate {hz} Hz (expected {expected})")
        }
        DomainError::InvalidAudioOptionChannels { channels, expected } => {
            write!(
                f,
                "invalid audio channel count {channels} (expected {expected})"
            )
        }
        DomainError::IncompatibleAudioBitrateRate { bps, hz, max_bps } => write!(
            f,
            "audio bitrate {bps} bps at {hz} Hz is incompatible with MP3 (MPEG-2/2.5 Layer III caps at {max_bps} bps)"
        ),
        DomainError::InvalidAudioCodec { raw, expected } => {
            write!(f, "invalid audio codec '{raw}' (expected {expected})")
        }
        DomainError::UnsupportedAudioCodecForFormat { format, codec } => write!(
            f,
            "codec '{}' cannot be stored in {} (supported codecs: {}; containers supporting {}: {})",
            codec.cli_name(),
            format.container_label(),
            cli_name_list(
                format
                    .supported_codecs()
                    .iter()
                    .map(|codec| codec.cli_name())
            ),
            codec.cli_name(),
            cli_name_list(
                codec
                    .supported_formats()
                    .iter()
                    .map(|format| format.cli_name())
            )
        ),
        DomainError::InvalidAudioBitDepth { raw, expected } => {
            write!(f, "invalid audio bit depth '{raw}' (expected {expected})")
        }
        DomainError::UnsupportedAudioBitDepth {
            format,
            codec,
            bit_depth,
            reason,
        } => write!(
            f,
            "--bit-depth {bit_depth} is not supported for {} audio: {reason}",
            format.resolved_label(*codec)
        ),
        DomainError::InvalidPcmWavLayout { reason } => {
            write!(f, "invalid canonical PCM WAV layout: {reason}")
        }
        _ => Err(fmt::Error),
    }
}

impl Error for DomainError {}

/// Validated media conversion request with non-empty, distinct source and target paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionRequest {
    source: PathBuf,
    target: PathBuf,
    media_kind: MediaKind,
    backend: HardwareBackend,
}

impl ConversionRequest {
    /// Creates a validated conversion request without performing any I/O.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyPath`] if `source` or `target` is empty, or
    /// [`DomainError::IdenticalSourceAndTarget`] if both paths are identical.
    pub fn new(
        source: PathBuf,
        target: PathBuf,
        media_kind: MediaKind,
        backend: HardwareBackend,
    ) -> Result<Self, DomainError> {
        if source.as_os_str().is_empty() {
            return Err(DomainError::EmptyPath(PathRole::Source));
        }
        if target.as_os_str().is_empty() {
            return Err(DomainError::EmptyPath(PathRole::Target));
        }
        if source == target {
            return Err(DomainError::IdenticalSourceAndTarget(source));
        }

        Ok(Self {
            source,
            target,
            media_kind,
            backend,
        })
    }

    #[must_use]
    pub fn source(&self) -> &Path {
        &self.source
    }

    #[must_use]
    pub fn target(&self) -> &Path {
        &self.target
    }

    #[must_use]
    pub const fn media_kind(&self) -> MediaKind {
        self.media_kind
    }

    #[must_use]
    pub const fn backend(&self) -> HardwareBackend {
        self.backend
    }
}

/// Type-safe execution plan that makes invalid backend/media combinations unrepresentable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionPlan {
    Cpu {
        source: PathBuf,
        target: PathBuf,
        media_kind: MediaKind,
    },
    GpuZeroCopy {
        source: PathBuf,
        target: PathBuf,
        media_kind: GpuMediaKind,
        backend: GpuBackend,
    },
}

impl ExecutionPlan {
    /// Compiles a validated [`ConversionRequest`] into a concrete [`ExecutionPlan`].
    ///
    /// Never performs silent CPU fallbacks: requesting a GPU backend for an
    /// unsupported [`MediaKind`] fails explicitly.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedBackend`] when a GPU backend is requested for
    /// [`MediaKind::Audio`].
    pub fn from_request(request: ConversionRequest) -> Result<Self, DomainError> {
        match request.backend {
            HardwareBackend::Cpu => Ok(Self::Cpu {
                source: request.source,
                target: request.target,
                media_kind: request.media_kind,
            }),
            HardwareBackend::Gpu(backend) => {
                let gpu_media_kind = match request.media_kind {
                    MediaKind::Video => GpuMediaKind::Video,
                    MediaKind::Image => GpuMediaKind::Image,
                    MediaKind::Audio => {
                        return Err(DomainError::UnsupportedBackend {
                            media_kind: MediaKind::Audio,
                            backend: HardwareBackend::Gpu(backend),
                        });
                    }
                };

                Ok(Self::GpuZeroCopy {
                    source: request.source,
                    target: request.target,
                    media_kind: gpu_media_kind,
                    backend,
                })
            }
        }
    }

    #[must_use]
    pub fn source(&self) -> &Path {
        match self {
            Self::Cpu { source, .. } | Self::GpuZeroCopy { source, .. } => source,
        }
    }

    #[must_use]
    pub fn target(&self) -> &Path {
        match self {
            Self::Cpu { target, .. } | Self::GpuZeroCopy { target, .. } => target,
        }
    }

    #[must_use]
    pub const fn backend(&self) -> HardwareBackend {
        match self {
            Self::Cpu { .. } => HardwareBackend::Cpu,
            Self::GpuZeroCopy { backend, .. } => HardwareBackend::Gpu(*backend),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ConversionRequest, DomainError, ExecutionPlan, GpuBackend, GpuMediaKind, HardwareBackend,
        MediaKind, PathRole,
    };
    use std::path::PathBuf;

    #[test]
    fn builds_cpu_plan_from_valid_request() -> Result<(), DomainError> {
        let request = ConversionRequest::new(
            PathBuf::from("fixtures/input.mp4"),
            PathBuf::from("output.mkv"),
            MediaKind::Video,
            HardwareBackend::Cpu,
        )?;

        let plan = ExecutionPlan::from_request(request)?;
        assert_eq!(plan.backend(), HardwareBackend::Cpu);
        assert_eq!(plan.source(), PathBuf::from("fixtures/input.mp4"));
        assert_eq!(plan.target(), PathBuf::from("output.mkv"));
        Ok(())
    }

    #[test]
    fn builds_gpu_zero_copy_plan_for_video() -> Result<(), DomainError> {
        let request = ConversionRequest::new(
            PathBuf::from("fixtures/clip.mov"),
            PathBuf::from("clip.mp4"),
            MediaKind::Video,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
        )?;

        let plan = ExecutionPlan::from_request(request)?;
        assert_eq!(
            plan,
            ExecutionPlan::GpuZeroCopy {
                source: PathBuf::from("fixtures/clip.mov"),
                target: PathBuf::from("clip.mp4"),
                media_kind: GpuMediaKind::Video,
                backend: GpuBackend::Nvenc,
            }
        );
        Ok(())
    }

    #[test]
    fn rejects_gpu_backend_for_audio_without_silent_fallback() -> Result<(), DomainError> {
        let request = ConversionRequest::new(
            PathBuf::from("fixtures/track.wav"),
            PathBuf::from("track.flac"),
            MediaKind::Audio,
            HardwareBackend::Gpu(GpuBackend::Vaapi),
        )?;

        let result = ExecutionPlan::from_request(request);
        assert_eq!(
            result,
            Err(DomainError::UnsupportedBackend {
                media_kind: MediaKind::Audio,
                backend: HardwareBackend::Gpu(GpuBackend::Vaapi),
            })
        );
        Ok(())
    }

    #[test]
    fn rejects_identical_source_and_target() {
        let result = ConversionRequest::new(
            PathBuf::from("fixtures/same.mp4"),
            PathBuf::from("fixtures/same.mp4"),
            MediaKind::Video,
            HardwareBackend::Cpu,
        );

        assert_eq!(
            result,
            Err(DomainError::IdenticalSourceAndTarget(PathBuf::from(
                "fixtures/same.mp4"
            )))
        );
    }

    #[test]
    fn rejects_empty_paths() {
        let empty_source = ConversionRequest::new(
            PathBuf::new(),
            PathBuf::from("out.mp4"),
            MediaKind::Video,
            HardwareBackend::Cpu,
        );
        assert_eq!(empty_source, Err(DomainError::EmptyPath(PathRole::Source)));

        let empty_target = ConversionRequest::new(
            PathBuf::from("in.mp4"),
            PathBuf::new(),
            MediaKind::Image,
            HardwareBackend::Cpu,
        );
        assert_eq!(empty_target, Err(DomainError::EmptyPath(PathRole::Target)));
    }
}
