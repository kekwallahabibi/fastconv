use std::fmt;
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;

use crate::{DomainError, GpuBackend, HardwareBackend, PathRole};

/// Video container formats supported for remuxing and transcoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum VideoContainerFormat {
    Mp4,
    Mkv,
    Webm,
    Mov,
    Avi,
    Ts,
    Gif,
    Webp,
}

impl VideoContainerFormat {
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Mkv => "mkv",
            Self::Webm => "webm",
            Self::Mov => "mov",
            Self::Avi => "avi",
            Self::Ts => "ts",
            Self::Gif => "gif",
            Self::Webp => "webp",
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Mp4 => "MP4",
            Self::Mkv => "MKV",
            Self::Webm => "WebM",
            Self::Mov => "MOV",
            Self::Avi => "AVI",
            Self::Ts => "TS",
            Self::Gif => "GIF",
            Self::Webp => "WebP",
        }
    }

    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "mp4" | "m4v" => Some(Self::Mp4),
            "mov" | "qt" => Some(Self::Mov),
            "mkv" => Some(Self::Mkv),
            "webm" => Some(Self::Webm),
            "avi" => Some(Self::Avi),
            "ts" | "m2ts" | "mts" => Some(Self::Ts),
            "gif" => Some(Self::Gif),
            "webp" => Some(Self::Webp),
            _ => None,
        }
    }

    /// Whether this container can carry a stream copy (no re-encoding).
    #[must_use]
    pub const fn supports_remux(self) -> bool {
        !matches!(self, Self::Gif | Self::Webp)
    }

    /// Whether this container can carry an audio track.
    #[must_use]
    pub const fn supports_audio(self) -> bool {
        !matches!(self, Self::Gif | Self::Webp)
    }

    #[must_use]
    pub const fn default_video_codec(self) -> TargetVideoCodec {
        match self {
            Self::Webm => TargetVideoCodec::Vp9,
            Self::Gif => TargetVideoCodec::Gif,
            Self::Webp => TargetVideoCodec::Webp,
            Self::Mp4 | Self::Mkv | Self::Mov | Self::Avi | Self::Ts => TargetVideoCodec::H264,
        }
    }

    /// `None` for animation containers, which have no audio track.
    #[must_use]
    pub const fn default_audio_codec(self) -> Option<TargetAudioCodec> {
        match self {
            Self::Gif | Self::Webp => None,
            Self::Webm => Some(TargetAudioCodec::Opus),
            Self::Mp4 | Self::Mkv | Self::Mov | Self::Avi | Self::Ts => Some(TargetAudioCodec::Aac),
        }
    }
}

impl fmt::Display for VideoContainerFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Video/audio codecs that fastconv recognises in containers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum VideoCodec {
    H264,
    Hevc,
    Av1,
    Vp9,
    Aac,
    Opus,
    Mp3,
    Vorbis,
    TimedText,
    Gif,
    Webp,
    Unknown,
}

impl VideoCodec {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::Hevc => "HEVC",
            Self::Av1 => "AV1",
            Self::Vp9 => "VP9",
            Self::Aac => "AAC",
            Self::Opus => "Opus",
            Self::Mp3 => "MP3",
            Self::Vorbis => "Vorbis",
            Self::TimedText => "Timed text",
            Self::Gif => "GIF",
            Self::Webp => "WebP",
            Self::Unknown => "unknown",
        }
    }

    /// Whether this codec can be stream-copied into `container`.
    #[must_use]
    pub const fn supported_in(self, container: VideoContainerFormat) -> bool {
        match container {
            VideoContainerFormat::Mp4 | VideoContainerFormat::Mov => matches!(
                self,
                Self::H264
                    | Self::Hevc
                    | Self::Av1
                    | Self::Vp9
                    | Self::Aac
                    | Self::Opus
                    | Self::Mp3
                    | Self::TimedText
            ),
            VideoContainerFormat::Mkv => matches!(
                self,
                Self::H264
                    | Self::Hevc
                    | Self::Av1
                    | Self::Vp9
                    | Self::Aac
                    | Self::Opus
                    | Self::Mp3
                    | Self::Vorbis
                    | Self::TimedText
            ),
            VideoContainerFormat::Webm => {
                matches!(self, Self::Vp9 | Self::Av1 | Self::Opus | Self::Vorbis)
            }
            VideoContainerFormat::Avi => matches!(
                self,
                Self::H264
                    | Self::Hevc
                    | Self::Av1
                    | Self::Vp9
                    | Self::Aac
                    | Self::Opus
                    | Self::Mp3
            ),
            // Standard MPEG-TS carries no VP9 bitstream mapping.
            VideoContainerFormat::Ts => matches!(
                self,
                Self::H264 | Self::Hevc | Self::Av1 | Self::Aac | Self::Opus | Self::Mp3
            ),
            VideoContainerFormat::Gif => matches!(self, Self::Gif),
            VideoContainerFormat::Webp => matches!(self, Self::Webp),
        }
    }

    #[must_use]
    pub const fn is_video(self) -> bool {
        matches!(
            self,
            Self::H264 | Self::Hevc | Self::Av1 | Self::Vp9 | Self::Gif | Self::Webp
        )
    }

    #[must_use]
    pub const fn is_audio(self) -> bool {
        matches!(self, Self::Aac | Self::Opus | Self::Mp3 | Self::Vorbis)
    }

    /// Whether NVDEC can decode this bitstream on a Turing-or-newer GPU.
    ///
    /// AV1 is deliberately excluded: this host's RTX 2060 (Turing) has no
    /// AV1 decode block, and GIF/WebP are image sequences, not NVDEC inputs.
    #[must_use]
    pub const fn nvdec_decodable(self) -> bool {
        matches!(self, Self::H264 | Self::Hevc | Self::Vp9)
    }
}

impl fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Video codec selected for re-encoding, distinct from probed bitstream codecs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum TargetVideoCodec {
    H264,
    Hevc,
    Av1,
    Vp9,
    Gif,
    Webp,
}

impl TargetVideoCodec {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::Hevc => "HEVC",
            Self::Av1 => "AV1",
            Self::Vp9 => "VP9",
            Self::Gif => "GIF",
            Self::Webp => "WebP",
        }
    }

    #[must_use]
    pub const fn as_video_codec(self) -> VideoCodec {
        match self {
            Self::H264 => VideoCodec::H264,
            Self::Hevc => VideoCodec::Hevc,
            Self::Av1 => VideoCodec::Av1,
            Self::Vp9 => VideoCodec::Vp9,
            Self::Gif => VideoCodec::Gif,
            Self::Webp => VideoCodec::Webp,
        }
    }

    #[must_use]
    pub const fn supported_in(self, container: VideoContainerFormat) -> bool {
        self.as_video_codec().supported_in(container)
    }

    #[must_use]
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "h264" | "avc" | "x264" => Some(Self::H264),
            "hevc" | "h265" | "x265" => Some(Self::Hevc),
            "av1" => Some(Self::Av1),
            "vp9" => Some(Self::Vp9),
            "gif" => Some(Self::Gif),
            "webp" => Some(Self::Webp),
            _ => None,
        }
    }
}

impl fmt::Display for TargetVideoCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Audio codec selected for re-encoding inside a video container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum TargetAudioCodec {
    Aac,
    Opus,
    Mp3,
}

impl TargetAudioCodec {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Aac => "AAC",
            Self::Opus => "Opus",
            Self::Mp3 => "MP3",
        }
    }

    #[must_use]
    pub const fn as_video_codec(self) -> VideoCodec {
        match self {
            Self::Aac => VideoCodec::Aac,
            Self::Opus => VideoCodec::Opus,
            Self::Mp3 => VideoCodec::Mp3,
        }
    }

    #[must_use]
    pub const fn supported_in(self, container: VideoContainerFormat) -> bool {
        self.as_video_codec().supported_in(container)
    }

    #[must_use]
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "aac" => Some(Self::Aac),
            "opus" => Some(Self::Opus),
            "mp3" => Some(Self::Mp3),
            _ => None,
        }
    }
}

impl fmt::Display for TargetAudioCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Validated lossy video quality factor in `0..=100`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct VideoQuality(u8);

impl VideoQuality {
    pub const MAX: u8 = 100;
    pub const DEFAULT: Self = Self(80);

    /// Creates a validated [`VideoQuality`] in `0..=100`.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidVideoQuality`] when `value > 100`.
    pub const fn new(value: u8) -> Result<Self, DomainError> {
        if value > Self::MAX {
            return Err(DomainError::InvalidVideoQuality {
                value,
                max: Self::MAX,
            });
        }
        Ok(Self(value))
    }

    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

impl fmt::Display for VideoQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Decode backend requested at the CLI/system boundary.
///
/// `Auto` is resolved exactly once in `fastconv-probe` from the probed source
/// codec and the selected encoder backend. Forced `Nvdec` never silently
/// degrades to CPU decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum HardwareDecodeBackend {
    Auto,
    Cpu,
    Nvdec,
}

impl HardwareDecodeBackend {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Nvdec => "nvdec",
        }
    }

    #[must_use]
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "cpu" | "software" => Some(Self::Cpu),
            "nvdec" | "cuda" | "gpu" => Some(Self::Nvdec),
            _ => None,
        }
    }

    /// Resolves the user request into the concrete decode path that the
    /// pipeline must execute.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedVideoPipeline`] when `Nvdec` was
    /// forced but the encoder backend is not NVENC, or the source codec has
    /// no NVDEC implementation. `Auto` picks NVDEC for hardware-decodable
    /// codecs on NVENC and explicitly picks CPU decode otherwise.
    pub fn resolve(
        self,
        source_codec: VideoCodec,
        encode_backend: HardwareBackend,
    ) -> Result<VideoDecodeMode, DomainError> {
        let nvenc = matches!(encode_backend, HardwareBackend::Gpu(GpuBackend::Nvenc));
        match (self, nvenc, source_codec.nvdec_decodable()) {
            (Self::Nvdec | Self::Auto, true, true) => Ok(VideoDecodeMode::Nvdec),
            (Self::Nvdec, false, _) => Err(DomainError::UnsupportedVideoPipeline {
                codec: source_codec,
                backend: encode_backend,
                reason: "--hwdecode nvdec requires --backend nvenc; no silent CPU decode fallback"
                    .to_owned(),
            }),
            (Self::Nvdec, true, false) => Err(DomainError::UnsupportedVideoPipeline {
                codec: source_codec,
                backend: encode_backend,
                reason: format!(
                    "NVDEC cannot decode {source_codec}; use --hwdecode cpu or --hwdecode auto"
                ),
            }),
            (Self::Cpu | Self::Auto, _, _) => Ok(VideoDecodeMode::Cpu),
        }
    }
}

impl fmt::Display for HardwareDecodeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Concrete, validated decode path carried from probe into the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum VideoDecodeMode {
    Cpu,
    Nvdec,
}

impl VideoDecodeMode {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu decode",
            Self::Nvdec => "nvdec",
        }
    }

    #[must_use]
    pub const fn is_nvdec(self) -> bool {
        matches!(self, Self::Nvdec)
    }

    /// Exact executed hardware path, surfaced in reports and the CLI summary.
    #[must_use]
    pub const fn path(self, encode_backend: HardwareBackend) -> &'static str {
        match (self, encode_backend) {
            (Self::Nvdec, HardwareBackend::Gpu(GpuBackend::Nvenc)) => "gpu:nvdec+nvenc",
            (Self::Cpu, HardwareBackend::Gpu(GpuBackend::Nvenc)) => "gpu:nvenc (cpu decode)",
            (Self::Cpu, HardwareBackend::Cpu) => "cpu",
            // Forced NVDEC with a CPU encoder is rejected at the boundary; these
            // arms only exist to keep the match exhaustive without a panic.
            (Self::Nvdec, _) => "invalid:nvdec+cpu",
            (Self::Cpu, _) => "cpu (unsupported encoder)",
        }
    }
}

impl fmt::Display for VideoDecodeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Compile-time-legal video encoder choice: a codec/backend pair and its quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum VideoEncodeMode {
    CpuH264 { quality: VideoQuality },
    CpuAv1 { quality: VideoQuality },
    CpuVp9 { quality: VideoQuality },
    CpuGif { quality: VideoQuality },
    CpuWebp { quality: VideoQuality },
    NvencH264 { quality: VideoQuality },
    NvencHevc { quality: VideoQuality },
}

impl VideoEncodeMode {
    /// Resolves an encode mode without ever falling back to CPU silently.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedVideoPipeline`] when the requested codec
    /// has no encoder on the selected backend.
    pub fn resolve(
        codec: TargetVideoCodec,
        quality: VideoQuality,
        backend: HardwareBackend,
    ) -> Result<Self, DomainError> {
        match (codec, backend) {
            (TargetVideoCodec::H264, HardwareBackend::Cpu) => Ok(Self::CpuH264 { quality }),
            (TargetVideoCodec::Av1, HardwareBackend::Cpu) => Ok(Self::CpuAv1 { quality }),
            (TargetVideoCodec::Vp9, HardwareBackend::Cpu) => Ok(Self::CpuVp9 { quality }),
            (TargetVideoCodec::Gif, HardwareBackend::Cpu) => Ok(Self::CpuGif { quality }),
            (TargetVideoCodec::Webp, HardwareBackend::Cpu) => Ok(Self::CpuWebp { quality }),
            (TargetVideoCodec::Gif | TargetVideoCodec::Webp, HardwareBackend::Gpu(_)) => {
                Err(DomainError::UnsupportedVideoPipeline {
                    codec: codec.as_video_codec(),
                    backend,
                    reason:
                        "animation encoding (GIF/WebP) runs on CPU only; no silent CPU fallback"
                            .to_owned(),
                })
            }
            (TargetVideoCodec::Hevc, HardwareBackend::Cpu) => {
                Err(DomainError::UnsupportedVideoPipeline {
                    codec: VideoCodec::Hevc,
                    backend,
                    reason:
                        "HEVC (H.265) encoding requires hardware acceleration via --backend nvenc"
                            .to_owned(),
                })
            }
            (TargetVideoCodec::H264, HardwareBackend::Gpu(GpuBackend::Nvenc)) => {
                Ok(Self::NvencH264 { quality })
            }
            (TargetVideoCodec::Hevc, HardwareBackend::Gpu(GpuBackend::Nvenc)) => {
                Ok(Self::NvencHevc { quality })
            }
            (
                TargetVideoCodec::Av1 | TargetVideoCodec::Vp9,
                HardwareBackend::Gpu(GpuBackend::Nvenc),
            ) => Err(DomainError::UnsupportedVideoPipeline {
                codec: codec.as_video_codec(),
                backend,
                reason: "NVENC supports H.264 and HEVC encoding; use --backend cpu for AV1/VP9"
                    .to_owned(),
            }),
            (_, HardwareBackend::Gpu(other)) => Err(DomainError::UnsupportedVideoPipeline {
                codec: codec.as_video_codec(),
                backend: HardwareBackend::Gpu(other),
                reason: format!(
                    "GPU backend '{other}' video encoding is not implemented; no silent CPU fallback"
                ),
            }),
        }
    }

    #[must_use]
    pub const fn codec(self) -> TargetVideoCodec {
        match self {
            Self::CpuH264 { .. } | Self::NvencH264 { .. } => TargetVideoCodec::H264,
            Self::CpuAv1 { .. } => TargetVideoCodec::Av1,
            Self::CpuVp9 { .. } => TargetVideoCodec::Vp9,
            Self::CpuGif { .. } => TargetVideoCodec::Gif,
            Self::CpuWebp { .. } => TargetVideoCodec::Webp,
            Self::NvencHevc { .. } => TargetVideoCodec::Hevc,
        }
    }

    #[must_use]
    pub const fn quality(self) -> VideoQuality {
        match self {
            Self::CpuH264 { quality }
            | Self::CpuAv1 { quality }
            | Self::CpuVp9 { quality }
            | Self::CpuGif { quality }
            | Self::CpuWebp { quality }
            | Self::NvencH264 { quality }
            | Self::NvencHevc { quality } => quality,
        }
    }

    #[must_use]
    pub const fn backend(self) -> HardwareBackend {
        match self {
            Self::CpuH264 { .. }
            | Self::CpuAv1 { .. }
            | Self::CpuVp9 { .. }
            | Self::CpuGif { .. }
            | Self::CpuWebp { .. } => HardwareBackend::Cpu,
            Self::NvencH264 { .. } | Self::NvencHevc { .. } => {
                HardwareBackend::Gpu(GpuBackend::Nvenc)
            }
        }
    }
}

impl fmt::Display for VideoEncodeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} (quality {})",
            self.backend(),
            self.codec(),
            self.quality()
        )
    }
}

/// Per-track decision for a video stream during transcoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum VideoTrackAction {
    Copy,
    Encode(VideoEncodeMode),
}

impl VideoTrackAction {
    #[must_use]
    pub const fn backend(self) -> HardwareBackend {
        match self {
            Self::Copy => HardwareBackend::Cpu,
            Self::Encode(mode) => mode.backend(),
        }
    }
}

impl fmt::Display for VideoTrackAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Copy => f.write_str("copy"),
            Self::Encode(mode) => write!(f, "encode {mode}"),
        }
    }
}

/// Per-track decision for an audio stream during video transcoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum AudioTrackAction {
    Copy,
    Encode(TargetAudioCodec),
    Drop,
}

impl fmt::Display for AudioTrackAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Copy => f.write_str("copy"),
            Self::Encode(codec) => write!(f, "encode {codec}"),
            Self::Drop => f.write_str("drop"),
        }
    }
}

/// Kind of a stream inside a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum VideoStreamKind {
    Video,
    Audio,
    Subtitle,
    Data,
    Other,
}

impl fmt::Display for VideoStreamKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Video => "video",
            Self::Audio => "audio",
            Self::Subtitle => "subtitle",
            Self::Data => "data",
            Self::Other => "other",
        };
        f.write_str(label)
    }
}

/// Playback loop count signalled by an animation source. `Times(1)` plays once;
/// there is deliberately no `Times(0)` (that bit pattern means "forever").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub enum AnimationLoop {
    #[default]
    Infinite,
    Times(NonZeroU16),
}

/// Animation facts validated once at the probe boundary and carried unchanged
/// into execution, so the pipeline never re-parses container metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct AnimationMeta {
    loop_count: AnimationLoop,
    has_alpha: bool,
}

impl AnimationMeta {
    #[must_use]
    pub const fn new(loop_count: AnimationLoop, has_alpha: bool) -> Self {
        Self {
            loop_count,
            has_alpha,
        }
    }

    #[must_use]
    pub const fn loop_count(self) -> AnimationLoop {
        self.loop_count
    }

    #[must_use]
    pub const fn has_alpha(self) -> bool {
        self.has_alpha
    }
}

/// Probed stream metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VideoStreamInfo {
    codec: VideoCodec,
    kind: VideoStreamKind,
    width: Option<NonZeroU32>,
    height: Option<NonZeroU32>,
    channels: Option<NonZeroU16>,
    sample_rate: Option<NonZeroU32>,
    timebase_num: u32,
    timebase_den: u32,
    duration: Option<u64>,
    language: Option<String>,
    animation: Option<AnimationMeta>,
}

impl VideoStreamInfo {
    // One argument per container fact keeps the boundary mapping explicit.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        codec: VideoCodec,
        kind: VideoStreamKind,
        width: Option<u32>,
        height: Option<u32>,
        channels: Option<u16>,
        sample_rate: Option<u32>,
        timebase_num: u32,
        timebase_den: u32,
        duration: Option<u64>,
        language: Option<String>,
    ) -> Self {
        Self {
            codec,
            kind,
            width: width.and_then(NonZeroU32::new),
            height: height.and_then(NonZeroU32::new),
            channels: channels.and_then(NonZeroU16::new),
            sample_rate: sample_rate.and_then(NonZeroU32::new),
            timebase_num,
            timebase_den,
            duration,
            language,
            animation: None,
        }
    }

    /// Attaches animation metadata validated by the probe boundary.
    #[must_use]
    pub const fn with_animation(mut self, animation: AnimationMeta) -> Self {
        self.animation = Some(animation);
        self
    }

    #[must_use]
    pub const fn animation(&self) -> Option<AnimationMeta> {
        self.animation
    }

    #[must_use]
    pub const fn codec(&self) -> VideoCodec {
        self.codec
    }

    #[must_use]
    pub const fn kind(&self) -> VideoStreamKind {
        self.kind
    }

    #[must_use]
    pub const fn width(&self) -> Option<NonZeroU32> {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> Option<NonZeroU32> {
        self.height
    }

    #[must_use]
    pub const fn channels(&self) -> Option<NonZeroU16> {
        self.channels
    }

    #[must_use]
    pub const fn sample_rate(&self) -> Option<NonZeroU32> {
        self.sample_rate
    }

    #[must_use]
    pub const fn timebase(&self) -> (u32, u32) {
        (self.timebase_num, self.timebase_den)
    }

    #[must_use]
    pub const fn duration(&self) -> Option<u64> {
        self.duration
    }

    #[must_use]
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }
}

impl fmt::Display for VideoStreamInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.codec)?;
        if let (Some(width), Some(height)) = (self.width, self.height) {
            write!(f, " {width}x{height}")?;
        }
        if let Some(sample_rate) = self.sample_rate {
            write!(f, " {sample_rate}Hz")?;
        }
        if let Some(channels) = self.channels {
            write!(f, " {channels}ch")?;
        }
        Ok(())
    }
}

/// A single remux item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoRemuxItem {
    source: PathBuf,
    target: PathBuf,
    input_format: VideoContainerFormat,
    output_format: VideoContainerFormat,
    streams: Vec<VideoStreamInfo>,
}

impl VideoRemuxItem {
    #[must_use]
    pub fn new(
        source: PathBuf,
        target: PathBuf,
        input_format: VideoContainerFormat,
        output_format: VideoContainerFormat,
        streams: Vec<VideoStreamInfo>,
    ) -> Self {
        Self {
            source,
            target,
            input_format,
            output_format,
            streams,
        }
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
    pub const fn input_format(&self) -> VideoContainerFormat {
        self.input_format
    }

    #[must_use]
    pub const fn output_format(&self) -> VideoContainerFormat {
        self.output_format
    }

    #[must_use]
    pub fn streams(&self) -> &[VideoStreamInfo] {
        &self.streams
    }
}

/// Validated batch plan for remuxing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoRemuxPlan {
    items: Vec<VideoRemuxItem>,
    worker_threads: NonZeroUsize,
}

impl VideoRemuxPlan {
    /// Creates a validated plan.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyBatch`] if `items` is empty, or
    /// [`DomainError::DuplicateTargetPath`] / [`DomainError::TargetOverwritesSource`].
    pub fn new(
        items: Vec<VideoRemuxItem>,
        worker_threads: NonZeroUsize,
    ) -> Result<Self, DomainError> {
        if items.is_empty() {
            return Err(DomainError::EmptyBatch);
        }
        let mut targets = std::collections::HashSet::new();
        let sources: std::collections::HashSet<PathBuf> =
            items.iter().map(|i| i.source().to_path_buf()).collect();
        for item in &items {
            if !item.input_format().supports_remux() || !item.output_format().supports_remux() {
                return Err(DomainError::UnsupportedVideoPipeline {
                    codec: VideoCodec::Unknown,
                    backend: HardwareBackend::Cpu,
                    reason: "GIF and WebP are animation containers and cannot be stream-copied; use `fastconv video`"
                        .to_owned(),
                });
            }
            if !targets.insert(item.target().to_path_buf()) {
                return Err(DomainError::DuplicateTargetPath(
                    item.target().to_path_buf(),
                ));
            }
            if sources.contains(item.target()) {
                return Err(DomainError::TargetOverwritesSource(
                    item.target().to_path_buf(),
                ));
            }
        }
        Ok(Self {
            items,
            worker_threads,
        })
    }

    #[must_use]
    pub fn items(&self) -> &[VideoRemuxItem] {
        &self.items
    }

    #[must_use]
    pub const fn worker_threads(&self) -> NonZeroUsize {
        self.worker_threads
    }

    #[must_use]
    pub fn effective_workers(&self) -> NonZeroUsize {
        let n = self.worker_threads.get().min(self.items.len()).max(1);
        NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN)
    }

    #[must_use]
    pub fn output_format(&self) -> VideoContainerFormat {
        self.items[0].output_format()
    }
}

/// Report after a remux batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VideoRemuxReport {
    files_remuxed: NonZeroUsize,
    total_input_bytes: NonZeroU64,
    total_output_bytes: NonZeroU64,
    duration: Duration,
    worker_threads_used: NonZeroUsize,
    backend: HardwareBackend,
}

impl VideoRemuxReport {
    #[must_use]
    pub fn new(
        files_remuxed: NonZeroUsize,
        total_input_bytes: NonZeroU64,
        total_output_bytes: NonZeroU64,
        duration: Duration,
        worker_threads_used: NonZeroUsize,
        backend: HardwareBackend,
    ) -> Self {
        Self {
            files_remuxed,
            total_input_bytes,
            total_output_bytes,
            duration,
            worker_threads_used,
            backend,
        }
    }

    #[must_use]
    pub const fn files_remuxed(&self) -> NonZeroUsize {
        self.files_remuxed
    }

    #[must_use]
    pub const fn total_input_bytes(&self) -> NonZeroU64 {
        self.total_input_bytes
    }

    #[must_use]
    pub const fn total_output_bytes(&self) -> NonZeroU64 {
        self.total_output_bytes
    }

    #[must_use]
    pub const fn duration(&self) -> Duration {
        self.duration
    }

    #[must_use]
    pub const fn worker_threads_used(&self) -> NonZeroUsize {
        self.worker_threads_used
    }

    #[must_use]
    pub const fn backend(&self) -> HardwareBackend {
        self.backend
    }

    #[must_use]
    pub fn megabytes_per_sec(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        (u64_to_f64(self.total_input_bytes.get()) / (1024.0 * 1024.0)) / secs
    }
}

/// A single transcode item with validated per-track actions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoTranscodeItem {
    source: PathBuf,
    target: PathBuf,
    input_format: VideoContainerFormat,
    output_format: VideoContainerFormat,
    streams: Vec<VideoStreamInfo>,
    video_action: VideoTrackAction,
    audio_action: AudioTrackAction,
    decode_mode: VideoDecodeMode,
}

impl VideoTranscodeItem {
    /// Creates a transcode item after rejecting codec/container mismatches.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyPath`] or [`DomainError::IdenticalSourceAndTarget`]
    /// for invalid paths, [`DomainError::UnsupportedVideoCodec`] when a copied or
    /// encoded track cannot be carried by `output_format`, and
    /// [`DomainError::UnsupportedVideoPipeline`] when `decode_mode` is NVDEC but
    /// the encode backend or source codec cannot support it.
    // Every validated input is passed explicitly at the probe boundary; the
    // decode-mode invariant belongs with the other track actions.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn new(
        source: PathBuf,
        target: PathBuf,
        input_format: VideoContainerFormat,
        output_format: VideoContainerFormat,
        streams: Vec<VideoStreamInfo>,
        video_action: VideoTrackAction,
        audio_action: AudioTrackAction,
        decode_mode: VideoDecodeMode,
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

        if decode_mode.is_nvdec() {
            if video_action.backend() != HardwareBackend::Gpu(GpuBackend::Nvenc) {
                return Err(DomainError::UnsupportedVideoPipeline {
                    codec: VideoCodec::Unknown,
                    backend: video_action.backend(),
                    reason: "NVDEC decode requires --backend nvenc; no silent CPU fallback"
                        .to_owned(),
                });
            }
            let video_streams: Vec<&VideoStreamInfo> = streams
                .iter()
                .filter(|s| s.kind() == VideoStreamKind::Video)
                .collect();
            let unsupported = video_streams
                .iter()
                .find(|s| !s.codec().nvdec_decodable())
                .map(|s| s.codec());
            if video_streams.is_empty() || unsupported.is_some() {
                return Err(DomainError::UnsupportedVideoPipeline {
                    codec: unsupported.unwrap_or(VideoCodec::Unknown),
                    backend: video_action.backend(),
                    reason: "NVDEC supports H.264, HEVC and VP9 sources only; use --hwdecode cpu"
                        .to_owned(),
                });
            }
        }

        match video_action {
            VideoTrackAction::Copy => {
                if !input_format.supports_remux() || !output_format.supports_remux() {
                    return Err(DomainError::UnsupportedVideoPipeline {
                        codec: VideoCodec::Unknown,
                        backend: HardwareBackend::Cpu,
                        reason: "stream copy requires remuxable containers; animation formats must be re-encoded"
                            .to_owned(),
                    });
                }
                for stream in streams
                    .iter()
                    .filter(|s| s.kind() == VideoStreamKind::Video)
                {
                    if !stream.codec().supported_in(output_format) {
                        return Err(DomainError::UnsupportedVideoCodec {
                            codec: stream.codec(),
                            container: output_format,
                        });
                    }
                }
            }
            VideoTrackAction::Encode(mode) => {
                if !mode.codec().supported_in(output_format) {
                    return Err(DomainError::UnsupportedVideoCodec {
                        codec: mode.codec().as_video_codec(),
                        container: output_format,
                    });
                }
            }
        }

        match audio_action {
            AudioTrackAction::Drop => {}
            AudioTrackAction::Copy => {
                if !output_format.supports_audio() {
                    return Err(DomainError::UnsupportedVideoPipeline {
                        codec: VideoCodec::Unknown,
                        backend: HardwareBackend::Cpu,
                        reason: "GIF and WebP containers have no audio track; drop audio streams for animation output"
                            .to_owned(),
                    });
                }
                for stream in streams
                    .iter()
                    .filter(|s| s.kind() == VideoStreamKind::Audio)
                {
                    if !stream.codec().supported_in(output_format) {
                        return Err(DomainError::UnsupportedVideoCodec {
                            codec: stream.codec(),
                            container: output_format,
                        });
                    }
                }
            }
            AudioTrackAction::Encode(codec) => {
                if !codec.supported_in(output_format) {
                    return Err(DomainError::UnsupportedVideoCodec {
                        codec: codec.as_video_codec(),
                        container: output_format,
                    });
                }
            }
        }

        if !output_format.supports_audio()
            && !streams.is_empty()
            && !streams.iter().any(|s| s.kind() == VideoStreamKind::Video)
        {
            return Err(DomainError::UnsupportedVideoPipeline {
                codec: VideoCodec::Unknown,
                backend: HardwareBackend::Cpu,
                reason: "GIF and WebP containers require at least one video stream".to_owned(),
            });
        }

        Ok(Self {
            source,
            target,
            input_format,
            output_format,
            streams,
            video_action,
            audio_action,
            decode_mode,
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
    pub const fn input_format(&self) -> VideoContainerFormat {
        self.input_format
    }

    #[must_use]
    pub const fn output_format(&self) -> VideoContainerFormat {
        self.output_format
    }

    #[must_use]
    pub fn streams(&self) -> &[VideoStreamInfo] {
        &self.streams
    }

    #[must_use]
    pub const fn video_action(&self) -> VideoTrackAction {
        self.video_action
    }

    #[must_use]
    pub const fn audio_action(&self) -> AudioTrackAction {
        self.audio_action
    }

    /// Concrete decode path that the pipeline must execute for this item.
    #[must_use]
    pub const fn decode_mode(&self) -> VideoDecodeMode {
        self.decode_mode
    }
}

/// Validated batch plan for video transcoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoTranscodePlan {
    items: Vec<VideoTranscodeItem>,
    worker_threads: NonZeroUsize,
}

impl VideoTranscodePlan {
    /// Creates a validated transcode batch plan.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyBatch`] if `items` is empty, or
    /// [`DomainError::DuplicateTargetPath`] / [`DomainError::TargetOverwritesSource`].
    pub fn new(
        items: Vec<VideoTranscodeItem>,
        worker_threads: NonZeroUsize,
    ) -> Result<Self, DomainError> {
        if items.is_empty() {
            return Err(DomainError::EmptyBatch);
        }
        let mut targets = std::collections::HashSet::new();
        let sources: std::collections::HashSet<PathBuf> =
            items.iter().map(|i| i.source().to_path_buf()).collect();
        for item in &items {
            if !targets.insert(item.target().to_path_buf()) {
                return Err(DomainError::DuplicateTargetPath(
                    item.target().to_path_buf(),
                ));
            }
            if sources.contains(item.target()) {
                return Err(DomainError::TargetOverwritesSource(
                    item.target().to_path_buf(),
                ));
            }
        }
        Ok(Self {
            items,
            worker_threads,
        })
    }

    #[must_use]
    pub fn items(&self) -> &[VideoTranscodeItem] {
        &self.items
    }

    #[must_use]
    pub const fn worker_threads(&self) -> NonZeroUsize {
        self.worker_threads
    }

    #[must_use]
    pub fn effective_workers(&self) -> NonZeroUsize {
        let n = self.worker_threads.get().min(self.items.len()).max(1);
        NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN)
    }

    #[must_use]
    pub fn output_format(&self) -> VideoContainerFormat {
        self.items[0].output_format()
    }

    #[must_use]
    pub fn backend(&self) -> HardwareBackend {
        self.items[0].video_action().backend()
    }

    /// Decode path of the first item. All items in a batch share the requested
    /// decode backend; mixed source codecs may resolve differently under `auto`,
    /// which the pipeline handles per item.
    #[must_use]
    pub fn decode_mode(&self) -> VideoDecodeMode {
        self.items[0].decode_mode()
    }
}

/// Report after a transcode batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VideoTranscodeReport {
    files_converted: NonZeroUsize,
    frames_processed: u64,
    total_input_bytes: NonZeroU64,
    total_output_bytes: NonZeroU64,
    duration: Duration,
    worker_threads_used: NonZeroUsize,
    backend: HardwareBackend,
    hardware_device: Option<String>,
    decode_mode: VideoDecodeMode,
}

impl VideoTranscodeReport {
    // One argument per reported metric keeps the report construction explicit.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        files_converted: NonZeroUsize,
        frames_processed: u64,
        total_input_bytes: NonZeroU64,
        total_output_bytes: NonZeroU64,
        duration: Duration,
        worker_threads_used: NonZeroUsize,
        backend: HardwareBackend,
        hardware_device: Option<String>,
        decode_mode: VideoDecodeMode,
    ) -> Self {
        Self {
            files_converted,
            frames_processed,
            total_input_bytes,
            total_output_bytes,
            duration,
            worker_threads_used,
            backend,
            hardware_device,
            decode_mode,
        }
    }

    #[must_use]
    pub const fn files_converted(&self) -> NonZeroUsize {
        self.files_converted
    }

    #[must_use]
    pub const fn frames_processed(&self) -> u64 {
        self.frames_processed
    }

    #[must_use]
    pub const fn total_input_bytes(&self) -> NonZeroU64 {
        self.total_input_bytes
    }

    #[must_use]
    pub const fn total_output_bytes(&self) -> NonZeroU64 {
        self.total_output_bytes
    }

    #[must_use]
    pub const fn duration(&self) -> Duration {
        self.duration
    }

    #[must_use]
    pub const fn worker_threads_used(&self) -> NonZeroUsize {
        self.worker_threads_used
    }

    #[must_use]
    pub const fn backend(&self) -> HardwareBackend {
        self.backend
    }

    #[must_use]
    pub fn hardware_device(&self) -> Option<&str> {
        self.hardware_device.as_deref()
    }

    /// Concrete decode path executed for the batch.
    #[must_use]
    pub const fn decode_mode(&self) -> VideoDecodeMode {
        self.decode_mode
    }

    /// Exact executed hardware path for the CLI summary,
    /// e.g. `gpu:nvdec+nvenc` or `gpu:nvenc (cpu decode)`.
    #[must_use]
    pub const fn decode_path(&self) -> &'static str {
        self.decode_mode.path(self.backend)
    }

    #[must_use]
    pub fn megabytes_per_sec(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        (u64_to_f64(self.total_input_bytes.get()) / (1024.0 * 1024.0)) / secs
    }

    #[must_use]
    pub fn fps(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        u64_to_f64(self.frames_processed) / secs
    }
}

fn u64_to_f64(value: u64) -> f64 {
    let hi = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let lo = u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(hi) * 4_294_967_296.0 + f64::from(lo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    fn stream(codec: VideoCodec, kind: VideoStreamKind) -> VideoStreamInfo {
        VideoStreamInfo::new(codec, kind, None, None, None, None, 1, 1000, None, None)
    }

    fn item(
        source: &str,
        target: &str,
        input: VideoContainerFormat,
        output: VideoContainerFormat,
        streams: Vec<VideoStreamInfo>,
        video: VideoTrackAction,
        audio: AudioTrackAction,
    ) -> Result<VideoTranscodeItem, DomainError> {
        VideoTranscodeItem::new(
            PathBuf::from(source),
            PathBuf::from(target),
            input,
            output,
            streams,
            video,
            audio,
            VideoDecodeMode::Cpu,
        )
    }

    fn cpu_encode(codec: TargetVideoCodec) -> VideoTrackAction {
        let quality = VideoQuality::DEFAULT;
        VideoTrackAction::Encode(match codec {
            TargetVideoCodec::H264 => VideoEncodeMode::CpuH264 { quality },
            TargetVideoCodec::Av1 => VideoEncodeMode::CpuAv1 { quality },
            TargetVideoCodec::Vp9 => VideoEncodeMode::CpuVp9 { quality },
            TargetVideoCodec::Hevc => VideoEncodeMode::NvencHevc { quality },
            TargetVideoCodec::Gif => VideoEncodeMode::CpuGif { quality },
            TargetVideoCodec::Webp => VideoEncodeMode::CpuWebp { quality },
        })
    }

    #[test]
    fn codec_support_matrix() {
        let all = [
            VideoCodec::H264,
            VideoCodec::Hevc,
            VideoCodec::Av1,
            VideoCodec::Vp9,
            VideoCodec::Aac,
            VideoCodec::Opus,
            VideoCodec::Mp3,
            VideoCodec::Vorbis,
            VideoCodec::TimedText,
        ];
        let webm = [
            VideoCodec::Vp9,
            VideoCodec::Av1,
            VideoCodec::Opus,
            VideoCodec::Vorbis,
        ];
        let ts = [
            VideoCodec::H264,
            VideoCodec::Hevc,
            VideoCodec::Av1,
            VideoCodec::Aac,
            VideoCodec::Opus,
            VideoCodec::Mp3,
        ];
        let rich_audio = [
            VideoCodec::H264,
            VideoCodec::Hevc,
            VideoCodec::Av1,
            VideoCodec::Vp9,
            VideoCodec::Aac,
            VideoCodec::Opus,
            VideoCodec::Mp3,
        ];
        let mp4 = [
            VideoCodec::H264,
            VideoCodec::Hevc,
            VideoCodec::Av1,
            VideoCodec::Vp9,
            VideoCodec::Aac,
            VideoCodec::Opus,
            VideoCodec::Mp3,
            VideoCodec::TimedText,
        ];
        let mkv = [
            VideoCodec::H264,
            VideoCodec::Hevc,
            VideoCodec::Av1,
            VideoCodec::Vp9,
            VideoCodec::Aac,
            VideoCodec::Opus,
            VideoCodec::Mp3,
            VideoCodec::Vorbis,
            VideoCodec::TimedText,
        ];

        for codec in all {
            assert_eq!(
                codec.supported_in(VideoContainerFormat::Webm),
                webm.contains(&codec)
            );
            assert_eq!(
                codec.supported_in(VideoContainerFormat::Ts),
                ts.contains(&codec)
            );
            assert_eq!(
                codec.supported_in(VideoContainerFormat::Avi),
                rich_audio.contains(&codec)
            );
            assert_eq!(
                codec.supported_in(VideoContainerFormat::Mp4),
                mp4.contains(&codec)
            );
            assert_eq!(
                codec.supported_in(VideoContainerFormat::Mov),
                mp4.contains(&codec)
            );
            assert_eq!(
                codec.supported_in(VideoContainerFormat::Mkv),
                mkv.contains(&codec)
            );
        }
    }

    #[test]
    fn codec_support_specific_rules() {
        // WebM is restricted to VP9/AV1 video and Opus/Vorbis audio.
        for codec in [
            VideoCodec::H264,
            VideoCodec::Hevc,
            VideoCodec::Aac,
            VideoCodec::Mp3,
        ] {
            assert!(!codec.supported_in(VideoContainerFormat::Webm));
        }
        // MPEG-TS has no VP9 mapping.
        assert!(!VideoCodec::Vp9.supported_in(VideoContainerFormat::Ts));
        for container in [
            VideoContainerFormat::Avi,
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mov,
            VideoContainerFormat::Mkv,
        ] {
            for codec in [
                VideoCodec::H264,
                VideoCodec::Hevc,
                VideoCodec::Av1,
                VideoCodec::Vp9,
                VideoCodec::Aac,
                VideoCodec::Opus,
                VideoCodec::Mp3,
            ] {
                assert!(codec.supported_in(container));
            }
        }
        assert!(!VideoCodec::Vorbis.supported_in(VideoContainerFormat::Mp4));
        assert!(!VideoCodec::TimedText.supported_in(VideoContainerFormat::Avi));
        assert!(!VideoCodec::Unknown.supported_in(VideoContainerFormat::Mkv));
        assert!(VideoCodec::H264.is_video());
        assert!(VideoCodec::Hevc.is_video());
        assert!(VideoCodec::Av1.is_video());
        assert!(VideoCodec::Vp9.is_video());
        assert!(VideoCodec::Gif.is_video());
        assert!(VideoCodec::Webp.is_video());
        assert!(VideoCodec::Aac.is_audio());
        assert!(VideoCodec::Opus.is_audio());
        assert!(VideoCodec::Mp3.is_audio());
        assert!(VideoCodec::Vorbis.is_audio());
        assert!(!VideoCodec::H264.is_audio());
        assert!(!VideoCodec::Mp3.is_video());
    }

    #[test]
    fn animation_codecs_are_confined_to_their_own_container() {
        assert!(VideoCodec::Gif.supported_in(VideoContainerFormat::Gif));
        assert!(VideoCodec::Webp.supported_in(VideoContainerFormat::Webp));
        for container in [
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Webm,
            VideoContainerFormat::Mov,
            VideoContainerFormat::Avi,
            VideoContainerFormat::Ts,
        ] {
            assert!(!VideoCodec::Gif.supported_in(container), "{container}");
            assert!(!VideoCodec::Webp.supported_in(container), "{container}");
        }
        assert!(!VideoCodec::Gif.supported_in(VideoContainerFormat::Webp));
        assert!(!VideoCodec::Webp.supported_in(VideoContainerFormat::Gif));
        assert!(!VideoCodec::H264.supported_in(VideoContainerFormat::Gif));
        assert!(!VideoCodec::Aac.supported_in(VideoContainerFormat::Webp));
    }

    #[test]
    fn animation_containers_are_video_only_and_cpu_encoded() {
        let gif = VideoContainerFormat::Gif;
        let webp = VideoContainerFormat::Webp;
        assert_eq!(gif.extension(), "gif");
        assert_eq!(webp.extension(), "webp");
        assert_eq!(gif.label(), "GIF");
        assert_eq!(webp.label(), "WebP");
        assert!(!gif.supports_remux());
        assert!(!webp.supports_remux());
        assert!(!gif.supports_audio());
        assert!(!webp.supports_audio());
        for container in [
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Webm,
            VideoContainerFormat::Mov,
            VideoContainerFormat::Avi,
            VideoContainerFormat::Ts,
        ] {
            assert!(container.supports_remux(), "{container}");
            assert!(container.supports_audio(), "{container}");
        }
        assert_eq!(gif.default_video_codec(), TargetVideoCodec::Gif);
        assert_eq!(webp.default_video_codec(), TargetVideoCodec::Webp);
        assert_eq!(gif.default_audio_codec(), None);
        assert_eq!(webp.default_audio_codec(), None);
        assert_eq!(
            VideoContainerFormat::Mp4.default_audio_codec(),
            Some(TargetAudioCodec::Aac)
        );
        assert_eq!(
            VideoContainerFormat::Webm.default_audio_codec(),
            Some(TargetAudioCodec::Opus)
        );
        assert_eq!(AudioTrackAction::Drop.to_string(), "drop");
    }

    #[test]
    fn animation_meta_carries_validated_playback_facts() {
        let times = NonZeroU16::new(3).unwrap_or(NonZeroU16::MIN);
        let meta = AnimationMeta::new(AnimationLoop::Times(times), true);
        assert_eq!(meta.loop_count(), AnimationLoop::Times(times));
        assert!(meta.has_alpha());
        assert_eq!(
            AnimationMeta::default().loop_count(),
            AnimationLoop::Infinite
        );
        assert!(!AnimationMeta::default().has_alpha());
        let with_meta = stream(VideoCodec::Webp, VideoStreamKind::Video).with_animation(meta);
        assert_eq!(with_meta.animation(), Some(meta));
        assert_eq!(
            stream(VideoCodec::Webp, VideoStreamKind::Video).animation(),
            None
        );
    }

    #[test]
    fn from_extension_maps_all_aliases() {
        let cases = [
            ("mp4", VideoContainerFormat::Mp4),
            ("m4v", VideoContainerFormat::Mp4),
            ("mov", VideoContainerFormat::Mov),
            ("qt", VideoContainerFormat::Mov),
            ("mkv", VideoContainerFormat::Mkv),
            ("webm", VideoContainerFormat::Webm),
            ("avi", VideoContainerFormat::Avi),
            ("ts", VideoContainerFormat::Ts),
            ("m2ts", VideoContainerFormat::Ts),
            ("mts", VideoContainerFormat::Ts),
            ("gif", VideoContainerFormat::Gif),
            ("webp", VideoContainerFormat::Webp),
            ("MP4", VideoContainerFormat::Mp4),
            ("WEBM", VideoContainerFormat::Webm),
            ("GIF", VideoContainerFormat::Gif),
            ("WEBP", VideoContainerFormat::Webp),
        ];
        for (ext, expected) in cases {
            assert_eq!(
                VideoContainerFormat::from_extension(ext),
                Some(expected),
                "extension {ext}"
            );
        }
        assert_eq!(VideoContainerFormat::from_extension("flv"), None);
    }

    #[test]
    fn container_defaults_and_codec_names() {
        assert_eq!(VideoContainerFormat::Mp4.extension(), "mp4");
        assert_eq!(VideoContainerFormat::Mkv.extension(), "mkv");
        assert_eq!(VideoContainerFormat::Webm.extension(), "webm");
        assert_eq!(VideoContainerFormat::Mov.extension(), "mov");
        assert_eq!(VideoContainerFormat::Avi.extension(), "avi");
        assert_eq!(VideoContainerFormat::Ts.extension(), "ts");
        assert_eq!(VideoContainerFormat::Webm.label(), "WebM");
        assert_eq!(VideoContainerFormat::Ts.label(), "TS");
        assert_eq!(
            VideoContainerFormat::Webm.default_video_codec(),
            TargetVideoCodec::Vp9
        );
        assert_eq!(
            VideoContainerFormat::Mp4.default_video_codec(),
            TargetVideoCodec::H264
        );
        assert_eq!(
            VideoContainerFormat::Webm.default_audio_codec(),
            Some(TargetAudioCodec::Opus)
        );
        assert_eq!(
            VideoContainerFormat::Mp4.default_audio_codec(),
            Some(TargetAudioCodec::Aac)
        );
        assert_eq!(
            TargetVideoCodec::from_name("x264"),
            Some(TargetVideoCodec::H264)
        );
        assert_eq!(
            TargetVideoCodec::from_name("HEVC"),
            Some(TargetVideoCodec::Hevc)
        );
        assert_eq!(
            TargetVideoCodec::from_name("h265"),
            Some(TargetVideoCodec::Hevc)
        );
        assert_eq!(
            TargetVideoCodec::from_name("av1"),
            Some(TargetVideoCodec::Av1)
        );
        assert_eq!(
            TargetVideoCodec::from_name("vp9"),
            Some(TargetVideoCodec::Vp9)
        );
        assert_eq!(
            TargetVideoCodec::from_name("gif"),
            Some(TargetVideoCodec::Gif)
        );
        assert_eq!(
            TargetVideoCodec::from_name("WebP"),
            Some(TargetVideoCodec::Webp)
        );
        assert_eq!(TargetVideoCodec::from_name("vp8"), None);
        assert_eq!(
            TargetAudioCodec::from_name("aac"),
            Some(TargetAudioCodec::Aac)
        );
        assert_eq!(
            TargetAudioCodec::from_name("opus"),
            Some(TargetAudioCodec::Opus)
        );
        assert_eq!(
            TargetAudioCodec::from_name("mp3"),
            Some(TargetAudioCodec::Mp3)
        );
        assert_eq!(TargetAudioCodec::from_name("flac"), None);
        assert!(TargetVideoCodec::Vp9.supported_in(VideoContainerFormat::Webm));
        assert!(!TargetVideoCodec::H264.supported_in(VideoContainerFormat::Webm));
        assert!(!TargetAudioCodec::Mp3.supported_in(VideoContainerFormat::Webm));
    }

    #[test]
    fn encode_mode_resolve_accepts_supported_pairs() -> Result<(), DomainError> {
        let q = VideoQuality::DEFAULT;
        assert_eq!(
            VideoEncodeMode::resolve(TargetVideoCodec::H264, q, HardwareBackend::Cpu)?,
            VideoEncodeMode::CpuH264 { quality: q }
        );
        assert_eq!(
            VideoEncodeMode::resolve(TargetVideoCodec::Av1, q, HardwareBackend::Cpu)?,
            VideoEncodeMode::CpuAv1 { quality: q }
        );
        assert_eq!(
            VideoEncodeMode::resolve(TargetVideoCodec::Vp9, q, HardwareBackend::Cpu)?,
            VideoEncodeMode::CpuVp9 { quality: q }
        );
        assert_eq!(
            VideoEncodeMode::resolve(TargetVideoCodec::Gif, q, HardwareBackend::Cpu)?,
            VideoEncodeMode::CpuGif { quality: q }
        );
        assert_eq!(
            VideoEncodeMode::resolve(TargetVideoCodec::Webp, q, HardwareBackend::Cpu)?,
            VideoEncodeMode::CpuWebp { quality: q }
        );
        let nvenc = HardwareBackend::Gpu(GpuBackend::Nvenc);
        assert_eq!(
            VideoEncodeMode::resolve(TargetVideoCodec::H264, q, nvenc)?,
            VideoEncodeMode::NvencH264 { quality: q }
        );
        assert_eq!(
            VideoEncodeMode::resolve(TargetVideoCodec::Hevc, q, nvenc)?,
            VideoEncodeMode::NvencHevc { quality: q }
        );
        assert_eq!(
            VideoEncodeMode::CpuH264 { quality: q }.codec(),
            TargetVideoCodec::H264
        );
        assert_eq!(
            VideoEncodeMode::NvencHevc { quality: q }.codec(),
            TargetVideoCodec::Hevc
        );
        assert_eq!(VideoEncodeMode::CpuAv1 { quality: q }.quality(), q);
        assert_eq!(
            VideoEncodeMode::CpuGif { quality: q }.codec(),
            TargetVideoCodec::Gif
        );
        assert_eq!(
            VideoEncodeMode::CpuWebp { quality: q }.codec(),
            TargetVideoCodec::Webp
        );
        assert_eq!(VideoEncodeMode::CpuGif { quality: q }.quality(), q);
        assert_eq!(VideoEncodeMode::CpuWebp { quality: q }.quality(), q);
        assert_eq!(
            VideoEncodeMode::CpuVp9 { quality: q }.backend(),
            HardwareBackend::Cpu
        );
        assert_eq!(
            VideoEncodeMode::CpuWebp { quality: q }.backend(),
            HardwareBackend::Cpu
        );
        assert_eq!(VideoEncodeMode::NvencH264 { quality: q }.backend(), nvenc);
        assert_eq!(VideoTrackAction::Copy.backend(), HardwareBackend::Cpu);
        Ok(())
    }

    #[test]
    fn encode_mode_resolve_rejects_unsupported_pairs_without_fallback() {
        let q = VideoQuality::DEFAULT;
        assert_eq!(
            VideoEncodeMode::resolve(TargetVideoCodec::Hevc, q, HardwareBackend::Cpu),
            Err(DomainError::UnsupportedVideoPipeline {
                codec: VideoCodec::Hevc,
                backend: HardwareBackend::Cpu,
                reason: "HEVC (H.265) encoding requires hardware acceleration via --backend nvenc"
                    .to_owned(),
            })
        );
        for codec in [TargetVideoCodec::Av1, TargetVideoCodec::Vp9] {
            assert_eq!(
                VideoEncodeMode::resolve(codec, q, HardwareBackend::Gpu(GpuBackend::Nvenc)),
                Err(DomainError::UnsupportedVideoPipeline {
                    codec: codec.as_video_codec(),
                    backend: HardwareBackend::Gpu(GpuBackend::Nvenc),
                    reason: "NVENC supports H.264 and HEVC encoding; use --backend cpu for AV1/VP9"
                        .to_owned(),
                })
            );
        }
        for codec in [TargetVideoCodec::Gif, TargetVideoCodec::Webp] {
            for backend in [
                HardwareBackend::Gpu(GpuBackend::Nvenc),
                HardwareBackend::Gpu(GpuBackend::Vaapi),
            ] {
                assert_eq!(
                    VideoEncodeMode::resolve(codec, q, backend),
                    Err(DomainError::UnsupportedVideoPipeline {
                        codec: codec.as_video_codec(),
                        backend,
                        reason:
                            "animation encoding (GIF/WebP) runs on CPU only; no silent CPU fallback"
                                .to_owned(),
                    })
                );
            }
        }
        for gpu in [
            GpuBackend::Vaapi,
            GpuBackend::Vulkan,
            GpuBackend::Qsv,
            GpuBackend::Amf,
            GpuBackend::VideoToolbox,
        ] {
            let backend = HardwareBackend::Gpu(gpu);
            assert_eq!(
                VideoEncodeMode::resolve(TargetVideoCodec::H264, q, backend),
                Err(DomainError::UnsupportedVideoPipeline {
                    codec: VideoCodec::H264,
                    backend,
                    reason: format!(
                        "GPU backend '{gpu}' video encoding is not implemented; no silent CPU fallback"
                    ),
                })
            );
        }
    }

    #[test]
    fn video_quality_rejects_out_of_range() {
        assert_eq!(VideoQuality::MAX, 100);
        assert_eq!(VideoQuality::DEFAULT.get(), 80);
        assert_eq!(
            VideoQuality::new(101),
            Err(DomainError::InvalidVideoQuality {
                value: 101,
                max: 100
            })
        );
        assert_eq!(VideoQuality::new(100).map(VideoQuality::get), Ok(100));
    }

    #[test]
    fn transcode_item_rejects_webm_codec_mismatch() -> Result<(), DomainError> {
        let webm = VideoContainerFormat::Webm;
        let mp4 = VideoContainerFormat::Mp4;
        assert_eq!(
            item(
                "in.mp4",
                "out.webm",
                mp4,
                webm,
                Vec::new(),
                cpu_encode(TargetVideoCodec::H264),
                AudioTrackAction::Encode(TargetAudioCodec::Opus)
            ),
            Err(DomainError::UnsupportedVideoCodec {
                codec: VideoCodec::H264,
                container: webm
            })
        );
        assert_eq!(
            item(
                "in.mp4",
                "out.webm",
                mp4,
                webm,
                vec![stream(VideoCodec::H264, VideoStreamKind::Video)],
                VideoTrackAction::Copy,
                AudioTrackAction::Copy
            ),
            Err(DomainError::UnsupportedVideoCodec {
                codec: VideoCodec::H264,
                container: webm
            })
        );
        assert_eq!(
            item(
                "in.mp4",
                "out.webm",
                mp4,
                webm,
                Vec::new(),
                cpu_encode(TargetVideoCodec::Vp9),
                AudioTrackAction::Encode(TargetAudioCodec::Aac)
            ),
            Err(DomainError::UnsupportedVideoCodec {
                codec: VideoCodec::Aac,
                container: webm
            })
        );
        assert_eq!(
            item(
                "in.mkv",
                "out.webm",
                VideoContainerFormat::Mkv,
                webm,
                vec![stream(VideoCodec::Aac, VideoStreamKind::Audio)],
                VideoTrackAction::Copy,
                AudioTrackAction::Copy
            ),
            Err(DomainError::UnsupportedVideoCodec {
                codec: VideoCodec::Aac,
                container: webm
            })
        );

        let valid = item(
            "in.mkv",
            "out.webm",
            VideoContainerFormat::Mkv,
            webm,
            Vec::new(),
            cpu_encode(TargetVideoCodec::Vp9),
            AudioTrackAction::Encode(TargetAudioCodec::Opus),
        )?;
        assert_eq!(valid.output_format(), webm);
        let copied = item(
            "in.webm",
            "out2.webm",
            webm,
            webm,
            vec![
                stream(VideoCodec::Vp9, VideoStreamKind::Video),
                stream(VideoCodec::Opus, VideoStreamKind::Audio),
            ],
            VideoTrackAction::Copy,
            AudioTrackAction::Copy,
        )?;
        assert_eq!(copied.streams().len(), 2);
        Ok(())
    }

    #[test]
    fn transcode_item_rejects_ts_vp9() -> Result<(), DomainError> {
        let ts = VideoContainerFormat::Ts;
        assert_eq!(
            item(
                "in.webm",
                "out.ts",
                VideoContainerFormat::Webm,
                ts,
                Vec::new(),
                cpu_encode(TargetVideoCodec::Vp9),
                AudioTrackAction::Encode(TargetAudioCodec::Opus)
            ),
            Err(DomainError::UnsupportedVideoCodec {
                codec: VideoCodec::Vp9,
                container: ts
            })
        );
        assert_eq!(
            item(
                "in.webm",
                "out.ts",
                VideoContainerFormat::Webm,
                ts,
                vec![stream(VideoCodec::Vp9, VideoStreamKind::Video)],
                VideoTrackAction::Copy,
                AudioTrackAction::Copy
            ),
            Err(DomainError::UnsupportedVideoCodec {
                codec: VideoCodec::Vp9,
                container: ts
            })
        );
        let valid = item(
            "in.mp4",
            "out.ts",
            VideoContainerFormat::Mp4,
            ts,
            Vec::new(),
            cpu_encode(TargetVideoCodec::H264),
            AudioTrackAction::Encode(TargetAudioCodec::Mp3),
        )?;
        assert_eq!(valid.output_format(), ts);
        Ok(())
    }

    #[test]
    fn transcode_item_rejects_invalid_paths() {
        let target = vec![
            stream(VideoCodec::Vp9, VideoStreamKind::Video),
            stream(VideoCodec::Opus, VideoStreamKind::Audio),
        ];
        assert_eq!(
            item(
                "",
                "out.webm",
                VideoContainerFormat::Webm,
                VideoContainerFormat::Webm,
                target.clone(),
                VideoTrackAction::Copy,
                AudioTrackAction::Copy
            ),
            Err(DomainError::EmptyPath(PathRole::Source))
        );
        assert_eq!(
            item(
                "in.webm",
                "",
                VideoContainerFormat::Webm,
                VideoContainerFormat::Webm,
                target.clone(),
                VideoTrackAction::Copy,
                AudioTrackAction::Copy
            ),
            Err(DomainError::EmptyPath(PathRole::Target))
        );
        assert_eq!(
            item(
                "in.webm",
                "in.webm",
                VideoContainerFormat::Webm,
                VideoContainerFormat::Webm,
                target,
                VideoTrackAction::Copy,
                AudioTrackAction::Copy
            ),
            Err(DomainError::IdenticalSourceAndTarget(PathBuf::from(
                "in.webm"
            )))
        );
    }

    #[test]
    fn transcode_plan_rejects_empty_duplicates_and_overwrites() -> Result<(), DomainError> {
        let w = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        assert!(matches!(
            VideoTranscodePlan::new(Vec::new(), w),
            Err(DomainError::EmptyBatch)
        ));

        let make = |source: &str, target: &str| {
            item(
                source,
                target,
                VideoContainerFormat::Mp4,
                VideoContainerFormat::Webm,
                Vec::new(),
                cpu_encode(TargetVideoCodec::Vp9),
                AudioTrackAction::Encode(TargetAudioCodec::Opus),
            )
        };
        assert!(matches!(
            VideoTranscodePlan::new(
                vec![make("a.mp4", "out.webm")?, make("b.mp4", "out.webm")?],
                w
            ),
            Err(DomainError::DuplicateTargetPath(_))
        ));
        assert!(matches!(
            VideoTranscodePlan::new(vec![make("a.mp4", "b.mp4")?, make("b.mp4", "out.webm")?], w),
            Err(DomainError::TargetOverwritesSource(_))
        ));

        let plan =
            VideoTranscodePlan::new(vec![make("a.mp4", "a.webm")?, make("b.mp4", "b.webm")?], w)?;
        assert_eq!(plan.items().len(), 2);
        assert_eq!(plan.worker_threads(), w);
        assert_eq!(plan.effective_workers().get(), 2);
        assert_eq!(plan.output_format(), VideoContainerFormat::Webm);
        assert_eq!(plan.backend(), HardwareBackend::Cpu);
        Ok(())
    }

    #[test]
    fn transcode_report_metrics() {
        let w = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        let input = NonZeroU64::new(2 * 1024 * 1024).unwrap_or(NonZeroU64::MIN);
        let output = NonZeroU64::new(1024 * 1024).unwrap_or(NonZeroU64::MIN);
        let files = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        let report = VideoTranscodeReport::new(
            files,
            600,
            input,
            output,
            Duration::from_secs(2),
            w,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            Some("cuda:0".to_owned()),
            VideoDecodeMode::Nvdec,
        );
        assert_eq!(report.files_converted(), files);
        assert_eq!(report.frames_processed(), 600);
        assert_eq!(report.total_input_bytes(), input);
        assert_eq!(report.total_output_bytes(), output);
        assert_eq!(report.duration(), Duration::from_secs(2));
        assert_eq!(report.worker_threads_used(), w);
        assert_eq!(report.backend(), HardwareBackend::Gpu(GpuBackend::Nvenc));
        assert_eq!(report.hardware_device(), Some("cuda:0"));
        assert_eq!(report.megabytes_per_sec(), 1.0);
        assert_eq!(report.fps(), 300.0);
    }

    #[test]
    fn remux_plan_rejects_empty_and_duplicates() {
        let w = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        assert!(matches!(
            VideoRemuxPlan::new(Vec::new(), w),
            Err(DomainError::EmptyBatch)
        ));
    }

    #[test]
    fn remux_plan_rejects_animation_formats() -> Result<(), String> {
        let w = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        let gif_source = VideoRemuxItem::new(
            PathBuf::from("in.gif"),
            PathBuf::from("out.mkv"),
            VideoContainerFormat::Gif,
            VideoContainerFormat::Mkv,
            vec![stream(VideoCodec::Gif, VideoStreamKind::Video)],
        );
        assert!(matches!(
            VideoRemuxPlan::new(vec![gif_source], w),
            Err(DomainError::UnsupportedVideoPipeline { .. })
        ));
        let webp_target = VideoRemuxItem::new(
            PathBuf::from("in.mp4"),
            PathBuf::from("out.webp"),
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Webp,
            vec![stream(VideoCodec::H264, VideoStreamKind::Video)],
        );
        assert!(matches!(
            VideoRemuxPlan::new(vec![webp_target], w),
            Err(DomainError::UnsupportedVideoPipeline { .. })
        ));
        let valid = VideoRemuxItem::new(
            PathBuf::from("in.mp4"),
            PathBuf::from("out.mkv"),
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mkv,
            vec![stream(VideoCodec::H264, VideoStreamKind::Video)],
        );
        assert_eq!(
            VideoRemuxPlan::new(vec![valid], w)
                .map_err(|err| err.to_string())?
                .items()
                .len(),
            1
        );
        Ok(())
    }

    #[test]
    fn transcode_item_accepts_animation_with_dropped_audio()
    -> Result<(), Box<dyn std::error::Error>> {
        let video = stream(VideoCodec::Gif, VideoStreamKind::Video);
        let into_gif = item(
            "in.mp4",
            "out.gif",
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Gif,
            vec![video.clone()],
            cpu_encode(TargetVideoCodec::Gif),
            AudioTrackAction::Drop,
        )?;
        assert_eq!(into_gif.output_format(), VideoContainerFormat::Gif);
        assert_eq!(into_gif.audio_action(), AudioTrackAction::Drop);
        let into_webp = item(
            "in.gif",
            "out.webp",
            VideoContainerFormat::Gif,
            VideoContainerFormat::Webp,
            vec![video],
            cpu_encode(TargetVideoCodec::Webp),
            AudioTrackAction::Drop,
        )?;
        assert_eq!(into_webp.video_action(), cpu_encode(TargetVideoCodec::Webp));
        // An animation container with no streams has nothing to reject.
        let empty = item(
            "in.mp4",
            "empty.gif",
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Gif,
            Vec::new(),
            cpu_encode(TargetVideoCodec::Gif),
            AudioTrackAction::Drop,
        )?;
        assert_eq!(empty.streams(), []);
        Ok(())
    }

    #[test]
    fn transcode_item_rejects_illegal_animation_audio_and_copy() {
        let video = stream(VideoCodec::Gif, VideoStreamKind::Video);
        let audio = stream(VideoCodec::Aac, VideoStreamKind::Audio);
        let result = item(
            "in.gif",
            "out.gif",
            VideoContainerFormat::Gif,
            VideoContainerFormat::Gif,
            vec![video.clone()],
            VideoTrackAction::Copy,
            AudioTrackAction::Drop,
        );
        assert!(
            matches!(result, Err(DomainError::UnsupportedVideoPipeline { .. })),
            "animation streams cannot be copied"
        );
        let result = item(
            "in.mp4",
            "out.gif",
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Gif,
            vec![video.clone(), audio.clone()],
            cpu_encode(TargetVideoCodec::Gif),
            AudioTrackAction::Copy,
        );
        assert!(
            matches!(result, Err(DomainError::UnsupportedVideoPipeline { .. })),
            "animation containers cannot carry copied audio"
        );
        let result = item(
            "in.mp4",
            "out.gif",
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Gif,
            vec![audio],
            cpu_encode(TargetVideoCodec::Gif),
            AudioTrackAction::Drop,
        );
        assert!(
            matches!(result, Err(DomainError::UnsupportedVideoPipeline { .. })),
            "animation containers require at least one video stream"
        );
    }

    #[test]
    fn hwdecode_auto_picks_nvdec_only_for_supported_codecs() {
        let nvenc = HardwareBackend::Gpu(GpuBackend::Nvenc);
        for codec in [VideoCodec::H264, VideoCodec::Hevc, VideoCodec::Vp9] {
            assert_eq!(
                HardwareDecodeBackend::Auto.resolve(codec, nvenc),
                Ok(VideoDecodeMode::Nvdec),
                "auto must use NVDEC for {codec}"
            );
        }
        for codec in [VideoCodec::Av1, VideoCodec::Gif, VideoCodec::Webp] {
            assert_eq!(
                HardwareDecodeBackend::Auto.resolve(codec, nvenc),
                Ok(VideoDecodeMode::Cpu),
                "auto must use CPU decode for {codec}"
            );
        }
        assert_eq!(
            HardwareDecodeBackend::Auto.resolve(VideoCodec::H264, HardwareBackend::Cpu),
            Ok(VideoDecodeMode::Cpu)
        );
        assert_eq!(
            HardwareDecodeBackend::Cpu.resolve(VideoCodec::Hevc, nvenc),
            Ok(VideoDecodeMode::Cpu)
        );
    }

    #[test]
    fn hwdecode_nvdec_rejects_unsupported_pipelines_without_fallback() {
        let nvenc = HardwareBackend::Gpu(GpuBackend::Nvenc);
        assert!(matches!(
            HardwareDecodeBackend::Nvdec.resolve(VideoCodec::H264, HardwareBackend::Cpu),
            Err(DomainError::UnsupportedVideoPipeline { .. })
        ));
        for codec in [VideoCodec::Av1, VideoCodec::Gif, VideoCodec::Webp] {
            assert!(
                matches!(
                    HardwareDecodeBackend::Nvdec.resolve(codec, nvenc),
                    Err(DomainError::UnsupportedVideoPipeline { .. })
                ),
                "forced NVDEC must reject {codec}"
            );
        }
        assert_eq!(
            HardwareDecodeBackend::Nvdec.resolve(VideoCodec::Hevc, nvenc),
            Ok(VideoDecodeMode::Nvdec)
        );
    }

    #[test]
    fn video_item_rejects_nvdec_with_cpu_or_unsupported_stream() {
        let quality = VideoQuality::DEFAULT;
        let h264 = stream(VideoCodec::H264, VideoStreamKind::Video);
        let av1 = stream(VideoCodec::Av1, VideoStreamKind::Video);
        let nvenc_action = VideoTrackAction::Encode(VideoEncodeMode::NvencH264 { quality });
        assert!(
            item(
                "in.mp4",
                "out.mp4",
                VideoContainerFormat::Mp4,
                VideoContainerFormat::Mp4,
                vec![h264.clone()],
                nvenc_action,
                AudioTrackAction::Drop,
            )
            .is_ok()
        );

        let result = VideoTranscodeItem::new(
            PathBuf::from("in.mp4"),
            PathBuf::from("out.mp4"),
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mp4,
            vec![h264],
            cpu_encode(TargetVideoCodec::H264),
            AudioTrackAction::Drop,
            VideoDecodeMode::Nvdec,
        );
        assert!(
            matches!(result, Err(DomainError::UnsupportedVideoPipeline { .. })),
            "NVDEC decode with CPU encoding must be rejected"
        );

        let result = VideoTranscodeItem::new(
            PathBuf::from("in.mp4"),
            PathBuf::from("out.mp4"),
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mp4,
            vec![av1],
            nvenc_action,
            AudioTrackAction::Drop,
            VideoDecodeMode::Nvdec,
        );
        assert!(
            matches!(result, Err(DomainError::UnsupportedVideoPipeline { .. })),
            "NVDEC decode of AV1 must be rejected"
        );
    }

    #[test]
    fn report_surfaces_exact_hardware_path() {
        let one = NonZeroUsize::MIN;
        let bytes = NonZeroU64::MIN;
        let report = VideoTranscodeReport::new(
            one,
            1,
            bytes,
            bytes,
            Duration::from_secs(1),
            one,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            Some("RTX 2060".to_owned()),
            VideoDecodeMode::Nvdec,
        );
        assert_eq!(report.decode_path(), "gpu:nvdec+nvenc");
        let cpu_decode = VideoTranscodeReport::new(
            one,
            1,
            bytes,
            bytes,
            Duration::from_secs(1),
            one,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            None,
            VideoDecodeMode::Cpu,
        );
        assert_eq!(cpu_decode.decode_path(), "gpu:nvenc (cpu decode)");
    }
}
