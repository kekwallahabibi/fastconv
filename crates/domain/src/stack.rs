//! Pure data models for encoding an ordered sequence of still images into an
//! animation or a video container. No I/O happens here: every external fact is
//! validated once in `fastconv-probe` before a plan is built.
//!
//! [`StackExecutionMode`] has exactly two legal variants: animation (GIF/WebP)
//! and video (a validated container/encoder pair). There is no backend field
//! that could disagree with the encoder, so a GPU request can only be rejected
//! at the boundary, never silently downgraded.

use std::fmt;
use std::num::{NonZeroU16, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};

#[cfg(feature = "serde")]
use serde::Serialize;

use crate::{
    AnimationLoop, DomainError, GpuBackend, HardwareBackend, HostHardware, ImageDimensions,
    ImageInputFormat, PathRole, TargetVideoCodec, VideoContainerFormat, VideoEncodeMode,
    VideoQuality, VideoRateControl,
};

/// Animation-only output formats. Video containers are intentionally absent:
/// Feature 6 extends [`StackExecutionMode`] instead of overloading this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum StackTargetFormat {
    Gif,
    Webp,
}

impl StackTargetFormat {
    pub const ALL: [Self; 2] = [Self::Gif, Self::Webp];

    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Gif => "gif",
            Self::Webp => "webp",
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Gif => "GIF",
            Self::Webp => "WebP",
        }
    }

    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.trim().to_ascii_lowercase().as_str() {
            "gif" => Some(Self::Gif),
            "webp" => Some(Self::Webp),
            _ => None,
        }
    }

    /// The video container used by the shared animation encoder. This single
    /// mapping is the only bridge the stack pipeline needs to the codec layer.
    #[must_use]
    pub const fn as_video_container(self) -> VideoContainerFormat {
        match self {
            Self::Gif => VideoContainerFormat::Gif,
            Self::Webp => VideoContainerFormat::Webp,
        }
    }

    /// Static per-axis limits of the animation container.
    #[must_use]
    pub const fn max_dimensions(self) -> (u32, u32) {
        match self {
            // GIF's logical screen descriptor stores each axis as a u16.
            Self::Gif => (65_535, 65_535),
            // WebP's VP8X canvas stores each axis minus one in 14 bits.
            Self::Webp => (16_383, 16_383),
        }
    }
}

impl fmt::Display for StackTargetFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, ".{}", self.extension())
    }
}

/// Backend requested at the CLI boundary for a frame stack. `Auto` is resolved
/// exactly once in `fastconv-probe`; an explicit GPU backend is never replaced
/// by a CPU encoder when hardware is unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum StackBackendRequest {
    Auto,
    Explicit(HardwareBackend),
}

impl StackBackendRequest {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Explicit(HardwareBackend::Cpu) => "cpu",
            Self::Explicit(HardwareBackend::Gpu(GpuBackend::Nvenc)) => "nvenc",
            Self::Explicit(HardwareBackend::Gpu(GpuBackend::Vaapi)) => "vaapi",
            Self::Explicit(HardwareBackend::Gpu(GpuBackend::Qsv)) => "qsv",
            Self::Explicit(HardwareBackend::Gpu(GpuBackend::Amf)) => "amf",
            Self::Explicit(HardwareBackend::Gpu(GpuBackend::VideoToolbox)) => "videotoolbox",
            Self::Explicit(HardwareBackend::Gpu(GpuBackend::Vulkan)) => "vulkan",
        }
    }

    #[must_use]
    pub fn from_name(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "cpu" | "software" => Some(Self::Explicit(HardwareBackend::Cpu)),
            "nvenc" => Some(Self::Explicit(HardwareBackend::Gpu(GpuBackend::Nvenc))),
            "vaapi" => Some(Self::Explicit(HardwareBackend::Gpu(GpuBackend::Vaapi))),
            "qsv" => Some(Self::Explicit(HardwareBackend::Gpu(GpuBackend::Qsv))),
            "amf" => Some(Self::Explicit(HardwareBackend::Gpu(GpuBackend::Amf))),
            "videotoolbox" => Some(Self::Explicit(HardwareBackend::Gpu(
                GpuBackend::VideoToolbox,
            ))),
            "vulkan" => Some(Self::Explicit(HardwareBackend::Gpu(GpuBackend::Vulkan))),
            _ => None,
        }
    }
}

impl fmt::Display for StackBackendRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Finite, strictly positive frame rate. The bounds keep one frame
/// duration at least 1 µs and below `u64::MAX` µs even after multiplication.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct StackFps(f64);

impl StackFps {
    pub const MIN: f64 = 1e-6;
    pub const MAX: f64 = 1_000_000.0;
    pub const DEFAULT: Self = Self(15.0);

    /// Creates a validated [`StackFps`].
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidStackFps`] when `fps` is not finite, not
    /// strictly positive, or outside `1e-6..=1_000_000`.
    pub fn new(fps: f64) -> Result<Self, DomainError> {
        if !fps.is_finite() || fps < Self::MIN || fps > Self::MAX {
            return Err(DomainError::InvalidStackFps {
                reason: format!(
                    "--fps must be a finite number in {}..={}, got {fps}",
                    Self::MIN,
                    Self::MAX
                ),
            });
        }
        Ok(Self(fps))
    }

    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// Absolute presentation timestamp of frame `index` in microseconds.
    /// Deriving every timestamp from `index` (rather than accumulating a
    /// rounded duration) gives zero drift at 24, 30 and 29.97 fps.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "`StackFps` validation bounds the result into u64; Rust float casts saturate"
    )]
    pub fn frame_timestamp_us(self, index: u64) -> u64 {
        let secs = crate::u64_to_f64(index);
        let micros = secs * 1_000_000.0 / self.0;
        micros.round() as u64
    }

    /// Best rational approximation of the frame rate as `(numerator, denominator)`.
    ///
    /// NVENC receives these as `frameRateNum`/`frameRateDen`, so `29.97` stays
    /// `2997/100` and `0.5` stays `1/2` rather than collapsing to `1/1`. The
    /// continued-fraction search is bounded to `u32` components; the exact
    /// integer frame rates always terminate after the first term.
    #[must_use]
    pub fn rational(self) -> (u32, u32) {
        let (num, den) = if self.0 >= 1.0 {
            approximate_rational(self.0)
        } else {
            let (num, den) = approximate_rational(1.0 / self.0);
            (den, num)
        };
        (
            u32::try_from(num.clamp(1, u64::from(u32::MAX))).unwrap_or(u32::MAX),
            u32::try_from(den.clamp(1, u64::from(u32::MAX))).unwrap_or(u32::MAX),
        )
    }
}

/// Continued-fraction convergents, stopping at the first exact term or when
/// components exceed NVENC-safe bounds (`num <= 1_000_000`, `den <= 10_000`).
/// `h2/k2` and `h1/k1` are the two preceding convergents, which keeps every
/// intermediate fraction fully reduced and avoids driver-hostile 10^8+
/// denominators for high-precision user floats like `29.970000001`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the term is floored and bounded to u64 before conversion"
)]
fn approximate_rational(value: f64) -> (u64, u64) {
    const MAX_NUM: u64 = 1_000_000;
    const MAX_DEN: u64 = 10_000;
    let (mut h2, mut k2, mut h1, mut k1) = (0_u64, 1_u64, 1_u64, 0_u64);
    let mut best = (1_u64, 1_u64);
    let mut x = value;
    for _ in 0..64 {
        let term = x.floor();
        if !term.is_finite() || term < 1.0 {
            break;
        }
        let a = term as u64;
        let h = a.saturating_mul(h1).saturating_add(h2);
        let k = a.saturating_mul(k1).saturating_add(k2);
        if k == 0 || k > MAX_DEN || h == 0 || h > MAX_NUM {
            break;
        }
        best = (h, k);
        if (x - term).abs() < 1e-12 {
            break;
        }
        h2 = h1;
        k2 = k1;
        h1 = h;
        k1 = k;
        x = 1.0 / (x - term);
    }
    best
}

impl fmt::Display for StackFps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One validated source frame of a stack.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct StackFrame {
    source: PathBuf,
    format: ImageInputFormat,
    has_alpha: bool,
}

impl StackFrame {
    /// Creates a validated [`StackFrame`].
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyPath`] when `source` is empty.
    pub fn new(
        source: PathBuf,
        format: ImageInputFormat,
        has_alpha: bool,
    ) -> Result<Self, DomainError> {
        if source.as_os_str().is_empty() {
            return Err(DomainError::EmptyPath(PathRole::Source));
        }
        Ok(Self {
            source,
            format,
            has_alpha,
        })
    }

    #[must_use]
    pub fn source(&self) -> &Path {
        &self.source
    }

    #[must_use]
    pub const fn format(&self) -> ImageInputFormat {
        self.format
    }

    #[must_use]
    pub const fn has_alpha(&self) -> bool {
        self.has_alpha
    }
}

/// Compile-time-legal execution mode for a frame sequence. The encoder is part
/// of the variant, so a backend can never disagree with it and a video plan can
/// never carry an animation loop count (or vice versa).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum StackExecutionMode {
    Animation {
        format: StackTargetFormat,
        quality: VideoQuality,
        loop_count: AnimationLoop,
    },
    Video {
        container: VideoContainerFormat,
        encode_mode: VideoEncodeMode,
    },
}

impl StackExecutionMode {
    #[must_use]
    pub const fn animation(
        format: StackTargetFormat,
        quality: VideoQuality,
        loop_count: AnimationLoop,
    ) -> Self {
        Self::Animation {
            format,
            quality,
            loop_count,
        }
    }

    /// Backwards-compatible constructor for animation plans that use the
    /// default infinite loop count.
    #[must_use]
    pub const fn cpu_animation(format: StackTargetFormat, quality: VideoQuality) -> Self {
        Self::animation(format, quality, AnimationLoop::Infinite)
    }

    /// Builds a video mode, rejecting animation containers and codecs that the
    /// container cannot carry. The bidirectional diagnostic comes from
    /// `DomainError::UnsupportedVideoCodec`.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedVideoCodec`] for an animation container
    /// or a container/codec mismatch.
    pub fn video(
        container: VideoContainerFormat,
        encode_mode: VideoEncodeMode,
    ) -> Result<Self, DomainError> {
        let codec = encode_mode.codec();
        if matches!(
            container,
            VideoContainerFormat::Gif | VideoContainerFormat::Webp
        ) || !codec.supported_in(container)
        {
            return Err(DomainError::UnsupportedVideoCodec {
                codec: codec.as_video_codec(),
                container,
            });
        }
        Ok(Self::Video {
            container,
            encode_mode,
        })
    }

    /// Resolves the CLI request exactly once, without any silent CPU fallback.
    ///
    /// # Errors
    ///
    /// Returns a domain error for an unsupported codec/container pair, rate
    /// control on a codec that cannot use it, a loop count on a video container,
    /// an unsupported GPU backend, or an NVENC request without compiled support
    /// and a detected encode-capable NVIDIA adapter.
    #[allow(
        clippy::too_many_arguments,
        reason = "Every argument is an independently validated boundary input; bundling them would add an unvalidated intermediate type"
    )]
    pub fn resolve_with_hardware(
        container: VideoContainerFormat,
        requested_codec: Option<TargetVideoCodec>,
        rate_control: VideoRateControl,
        explicit_loop: Option<AnimationLoop>,
        backend: StackBackendRequest,
        nvenc_compiled: bool,
        hardware: Option<&HostHardware>,
    ) -> Result<Self, DomainError> {
        let codec = match requested_codec {
            Some(codec) if !codec.supported_in(container) => {
                return Err(DomainError::UnsupportedVideoCodec {
                    codec: codec.as_video_codec(),
                    container,
                });
            }
            Some(codec) => codec,
            None => container.default_video_codec(),
        };
        let nvenc_available = nvenc_compiled
            && hardware.is_some_and(|host| host.supports_video_encode(GpuBackend::Nvenc));
        let require_nvenc = || -> Result<HardwareBackend, DomainError> {
            if nvenc_available {
                Ok(HardwareBackend::Gpu(GpuBackend::Nvenc))
            } else {
                let reason = if nvenc_compiled {
                    "NVENC stack encoding requires an encode-capable NVIDIA GPU detected on this host (no silent CPU fallback)"
                } else {
                    "NVENC stack encoding requires building with --features nvenc (no silent CPU fallback)"
                };
                Err(DomainError::UnsupportedVideoPipeline {
                    codec: codec.as_video_codec(),
                    backend: HardwareBackend::Gpu(GpuBackend::Nvenc),
                    reason: reason.to_owned(),
                })
            }
        };

        if matches!(
            container,
            VideoContainerFormat::Gif | VideoContainerFormat::Webp
        ) {
            let resolved = match backend {
                StackBackendRequest::Auto => HardwareBackend::Cpu,
                StackBackendRequest::Explicit(backend) => backend,
            };
            let encode_mode = VideoEncodeMode::resolve(codec, rate_control, resolved)?;
            let format = StackTargetFormat::from_extension(container.extension()).ok_or(
                DomainError::UnsupportedVideoCodec {
                    codec: codec.as_video_codec(),
                    container,
                },
            )?;
            return Ok(Self::Animation {
                format,
                quality: encode_mode.quality(),
                loop_count: explicit_loop.unwrap_or(AnimationLoop::Infinite),
            });
        }

        if explicit_loop.is_some() {
            return Err(DomainError::UnsupportedStackLoop { container });
        }
        let resolved = match (backend, codec) {
            (StackBackendRequest::Explicit(HardwareBackend::Gpu(GpuBackend::Nvenc)), _) => {
                VideoEncodeMode::resolve(
                    codec,
                    rate_control,
                    HardwareBackend::Gpu(GpuBackend::Nvenc),
                )?;
                require_nvenc()?
            }
            (StackBackendRequest::Explicit(other), _) => {
                VideoEncodeMode::resolve(codec, rate_control, other)?;
                other
            }
            (
                StackBackendRequest::Auto,
                TargetVideoCodec::Av1 | TargetVideoCodec::Vp9 | TargetVideoCodec::Mjpeg,
            ) => HardwareBackend::Cpu,
            (StackBackendRequest::Auto, TargetVideoCodec::Hevc) => require_nvenc()?,
            (StackBackendRequest::Auto, _) if nvenc_available => {
                HardwareBackend::Gpu(GpuBackend::Nvenc)
            }
            (StackBackendRequest::Auto, _) => HardwareBackend::Cpu,
        };
        let encode_mode = VideoEncodeMode::resolve(codec, rate_control, resolved)?;
        Self::video(container, encode_mode)
    }

    #[must_use]
    pub const fn animation_format(self) -> Option<StackTargetFormat> {
        match self {
            Self::Animation { format, .. } => Some(format),
            Self::Video { .. } => None,
        }
    }

    #[must_use]
    pub const fn target_container(self) -> VideoContainerFormat {
        match self {
            Self::Animation { format, .. } => format.as_video_container(),
            Self::Video { container, .. } => container,
        }
    }

    #[must_use]
    pub const fn video_codec(self) -> TargetVideoCodec {
        match self {
            Self::Animation { format, .. } => format.as_video_container().default_video_codec(),
            Self::Video { encode_mode, .. } => encode_mode.codec(),
        }
    }

    #[must_use]
    pub const fn encode_mode(self) -> Option<VideoEncodeMode> {
        match self {
            Self::Animation { .. } => None,
            Self::Video { encode_mode, .. } => Some(encode_mode),
        }
    }

    #[must_use]
    pub const fn quality(self) -> VideoQuality {
        match self {
            Self::Animation { quality, .. } => quality,
            Self::Video { encode_mode, .. } => encode_mode.quality(),
        }
    }

    #[must_use]
    pub const fn loop_count(self) -> Option<AnimationLoop> {
        match self {
            Self::Animation { loop_count, .. } => Some(loop_count),
            Self::Video { .. } => None,
        }
    }

    #[must_use]
    pub const fn backend(self) -> HardwareBackend {
        match self {
            Self::Animation { .. } => HardwareBackend::Cpu,
            Self::Video { encode_mode, .. } => encode_mode.backend(),
        }
    }
}

impl fmt::Display for StackExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Animation {
                format, quality, ..
            } => write!(f, "cpu animation .{} q={}", format.extension(), quality),
            Self::Video {
                container,
                encode_mode,
            } => write!(
                f,
                "{} video .{} {}",
                encode_mode.backend(),
                container.extension(),
                encode_mode.codec().cli_name()
            ),
        }
    }
}

/// Validated, non-empty, uniformly sized frame sequence ready for execution.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct FrameSequencePlan {
    frames: Vec<StackFrame>,
    dimensions: ImageDimensions,
    fps: StackFps,
    target_path: PathBuf,
    mode: StackExecutionMode,
    worker_threads: NonZeroUsize,
}

impl FrameSequencePlan {
    /// Builds a validated plan without performing any I/O.
    ///
    /// # Errors
    ///
    /// Rejects empty frame lists and target paths, a target that aliases any
    /// source path, dimensions beyond the target's static limit, odd video
    /// dimensions, and video dimensions beyond the container track's u16 range.
    pub fn new(
        frames: Vec<StackFrame>,
        dimensions: ImageDimensions,
        fps: StackFps,
        target_path: PathBuf,
        mode: StackExecutionMode,
        worker_threads: NonZeroUsize,
    ) -> Result<Self, DomainError> {
        if frames.is_empty() {
            return Err(DomainError::EmptyFrameSequence);
        }
        if target_path.as_os_str().is_empty() {
            return Err(DomainError::EmptyPath(PathRole::Target));
        }
        let width = dimensions.width().get();
        let height = dimensions.height().get();
        match mode {
            StackExecutionMode::Animation { format, .. } => {
                let (max_width, max_height) = format.max_dimensions();
                if width > max_width || height > max_height {
                    return Err(DomainError::StackDimensionLimit {
                        format,
                        width,
                        height,
                        max_width,
                        max_height,
                    });
                }
            }
            StackExecutionMode::Video { container, .. } => {
                let max = u32::from(u16::MAX);
                if width > max || height > max {
                    return Err(DomainError::StackVideoDimensionLimit {
                        container,
                        width,
                        height,
                        max_width: max,
                        max_height: max,
                    });
                }
                if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
                    return Err(DomainError::StackOddVideoDimensions {
                        container,
                        width,
                        height,
                    });
                }
            }
        }
        if frames
            .iter()
            .any(|frame| frame.source() == target_path.as_path())
        {
            return Err(DomainError::StackTargetOverlapsSource(target_path));
        }
        Ok(Self {
            frames,
            dimensions,
            fps,
            target_path,
            mode,
            worker_threads,
        })
    }

    /// Number of frames in the sequence, as a non-zero count.
    #[must_use]
    pub fn frame_count(&self) -> NonZeroU64 {
        let count = self.frames.len();
        NonZeroU64::new(u64::try_from(count).unwrap_or(u64::MAX)).unwrap_or(NonZeroU64::MIN)
    }

    #[must_use]
    pub fn frames(&self) -> &[StackFrame] {
        &self.frames
    }

    #[must_use]
    pub const fn dimensions(&self) -> ImageDimensions {
        self.dimensions
    }

    #[must_use]
    pub const fn fps(&self) -> StackFps {
        self.fps
    }

    #[must_use]
    pub fn target_path(&self) -> &Path {
        &self.target_path
    }

    #[must_use]
    pub const fn animation_format(&self) -> Option<StackTargetFormat> {
        self.mode.animation_format()
    }

    #[must_use]
    pub const fn target_container(&self) -> VideoContainerFormat {
        self.mode.target_container()
    }

    #[must_use]
    pub const fn video_codec(&self) -> TargetVideoCodec {
        self.mode.video_codec()
    }

    #[must_use]
    pub const fn quality(&self) -> VideoQuality {
        self.mode.quality()
    }

    #[must_use]
    pub const fn mode(&self) -> StackExecutionMode {
        self.mode
    }

    #[must_use]
    pub const fn backend(&self) -> HardwareBackend {
        self.mode.backend()
    }

    #[must_use]
    pub const fn loop_count(&self) -> Option<AnimationLoop> {
        self.mode.loop_count()
    }

    #[must_use]
    pub const fn worker_threads(&self) -> NonZeroUsize {
        self.worker_threads
    }
}

/// Execution summary returned after a frame sequence is encoded.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct FrameSequenceReport {
    frames_encoded: NonZeroU64,
    dimensions: ImageDimensions,
    fps: f64,
    target_container: VideoContainerFormat,
    video_codec: TargetVideoCodec,
    backend: HardwareBackend,
    hardware_device: Option<String>,
    target_path: PathBuf,
    output_bytes: NonZeroU64,
    worker_threads: NonZeroUsize,
}

impl FrameSequenceReport {
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        frames_encoded: NonZeroU64,
        dimensions: ImageDimensions,
        fps: f64,
        mode: StackExecutionMode,
        target_path: PathBuf,
        output_bytes: NonZeroU64,
        worker_threads: NonZeroUsize,
        hardware_device: Option<String>,
    ) -> Self {
        Self {
            frames_encoded,
            dimensions,
            fps,
            target_container: mode.target_container(),
            video_codec: mode.video_codec(),
            backend: mode.backend(),
            hardware_device,
            target_path,
            output_bytes,
            worker_threads,
        }
    }

    #[must_use]
    pub const fn frames_encoded(&self) -> NonZeroU64 {
        self.frames_encoded
    }

    #[must_use]
    pub const fn dimensions(&self) -> ImageDimensions {
        self.dimensions
    }

    #[must_use]
    pub const fn fps(&self) -> f64 {
        self.fps
    }

    #[must_use]
    pub const fn target_container(&self) -> VideoContainerFormat {
        self.target_container
    }

    #[must_use]
    pub const fn video_codec(&self) -> TargetVideoCodec {
        self.video_codec
    }

    #[must_use]
    pub const fn backend(&self) -> HardwareBackend {
        self.backend
    }

    #[must_use]
    pub fn hardware_device(&self) -> Option<&str> {
        self.hardware_device.as_deref()
    }

    #[must_use]
    pub fn target_path(&self) -> &Path {
        &self.target_path
    }

    #[must_use]
    pub const fn output_bytes(&self) -> NonZeroU64 {
        self.output_bytes
    }

    #[must_use]
    pub const fn worker_threads(&self) -> NonZeroUsize {
        self.worker_threads
    }
}

/// Parses a CLI `infinite|0|1..=65535` loop value into a validated loop count.
///
/// # Errors
///
/// Returns [`DomainError::InvalidAnimationLoop`] for anything outside that set.
pub fn parse_animation_loop(raw: &str) -> Result<AnimationLoop, DomainError> {
    let value = raw.trim();
    if matches!(value, "infinite" | "0") {
        return Ok(AnimationLoop::Infinite);
    }
    let times = value
        .parse::<NonZeroU16>()
        .map_err(|_| DomainError::InvalidAnimationLoop {
            raw: raw.to_owned(),
        })?;
    Ok(AnimationLoop::Times(times))
}

#[cfg(test)]
mod tests {
    use super::{
        FrameSequencePlan, FrameSequenceReport, StackBackendRequest, StackExecutionMode, StackFps,
        StackFrame, StackTargetFormat, parse_animation_loop,
    };
    use crate::{
        AnimationLoop, CpuInfo, DetectedGpuBackend, DomainError, GpuBackend, GpuCapability,
        HardwareBackend, HostHardware, ImageDimensions, ImageInputFormat, SimdLevel,
        TargetVideoCodec, VideoContainerFormat, VideoEncodeMode, VideoQuality, VideoRateControl,
    };
    use std::num::{NonZeroU16, NonZeroU64, NonZeroUsize};
    use std::path::PathBuf;

    fn plan() -> Result<FrameSequencePlan, DomainError> {
        let frames = vec![StackFrame::new(
            PathBuf::from("frame.png"),
            ImageInputFormat::Png,
            false,
        )?];
        FrameSequencePlan::new(
            frames,
            ImageDimensions::new(32, 24)?,
            StackFps::DEFAULT,
            PathBuf::from("out.gif"),
            StackExecutionMode::cpu_animation(StackTargetFormat::Gif, VideoQuality::DEFAULT),
            NonZeroUsize::MIN,
        )
    }

    fn nvenc_host() -> Result<HostHardware, DomainError> {
        Ok(HostHardware::new(
            CpuInfo::new(
                "test",
                "x86_64",
                NonZeroUsize::MIN,
                NonZeroUsize::MIN,
                SimdLevel::Scalar,
                None,
            )?,
            Vec::new(),
            vec![DetectedGpuBackend::new(
                GpuBackend::Nvenc,
                GpuCapability::VideoEncodeOnly,
                "Test RTX",
                "unit test",
            )?],
        ))
    }

    #[test]
    fn target_formats_expose_static_limits_and_container_mapping() {
        assert_eq!(
            StackTargetFormat::from_extension("GIF"),
            Some(StackTargetFormat::Gif)
        );
        assert_eq!(StackTargetFormat::from_extension(".webp"), None);
        assert_eq!(StackTargetFormat::Gif.max_dimensions(), (65_535, 65_535));
        assert_eq!(StackTargetFormat::Webp.max_dimensions(), (16_383, 16_383));
        assert_eq!(
            StackTargetFormat::Webp.as_video_container().extension(),
            "webp"
        );
        assert_eq!(StackTargetFormat::Gif.to_string(), ".gif");
    }

    #[test]
    fn fps_is_bounded_drift_free_and_rational() -> Result<(), DomainError> {
        for bad in [f64::NAN, f64::INFINITY, 0.0, -1.0, 1e9, 1e-7] {
            assert!(StackFps::new(bad).is_err(), "must reject {bad}");
        }
        let fps24 = StackFps::new(24.0)?;
        assert_eq!(fps24.frame_timestamp_us(1), 41_667);
        assert_eq!(fps24.frame_timestamp_us(3), 125_000);
        let fps30 = StackFps::new(30.0)?;
        assert_eq!(fps30.frame_timestamp_us(3), 100_000);
        let fps2997 = StackFps::new(29.97)?;
        assert_eq!(fps2997.frame_timestamp_us(1001), 33_400_067);
        assert_eq!(fps2997.rational(), (2997, 100));
        assert_eq!(StackFps::new(24_000.0 / 1001.0)?.rational(), (24_000, 1001));
        assert_eq!(StackFps::new(30_000.0 / 1001.0)?.rational(), (30_000, 1001));
        assert_eq!(StackFps::new(29.970_000_001)?.rational(), (2997, 100));
        assert_eq!(StackFps::new(0.5)?.rational(), (1, 2));
        assert_eq!(StackFps::new(15.0)?.rational(), (15, 1));
        assert_eq!(StackFps::new(StackFps::MIN)?.rational(), (1, 1_000_000));
        assert_eq!(StackFps::new(StackFps::MAX)?.rational(), (1_000_000, 1));
        let max = StackFps::new(StackFps::MAX)?;
        assert_eq!(max.frame_timestamp_us(u64::MAX), u64::MAX);
        Ok(())
    }

    #[test]
    fn validates_plan_invariants() -> Result<(), DomainError> {
        let built = plan()?;
        assert_eq!(built.frame_count().get(), 1);
        assert_eq!(built.animation_format(), Some(StackTargetFormat::Gif));
        assert_eq!(built.loop_count(), Some(AnimationLoop::Infinite));
        assert_eq!(built.backend(), HardwareBackend::Cpu);

        let oversized = FrameSequencePlan::new(
            vec![StackFrame::new(
                PathBuf::from("a.png"),
                ImageInputFormat::Png,
                false,
            )?],
            ImageDimensions::new(16_384, 10)?,
            StackFps::DEFAULT,
            PathBuf::from("out.webp"),
            StackExecutionMode::cpu_animation(StackTargetFormat::Webp, VideoQuality::DEFAULT),
            NonZeroUsize::MIN,
        );
        assert!(matches!(
            oversized,
            Err(DomainError::StackDimensionLimit { .. })
        ));

        let aliasing = FrameSequencePlan::new(
            vec![StackFrame::new(
                PathBuf::from("a.png"),
                ImageInputFormat::Png,
                false,
            )?],
            ImageDimensions::new(16, 16)?,
            StackFps::DEFAULT,
            PathBuf::from("a.png"),
            StackExecutionMode::cpu_animation(StackTargetFormat::Gif, VideoQuality::DEFAULT),
            NonZeroUsize::MIN,
        );
        assert!(matches!(
            aliasing,
            Err(DomainError::StackTargetOverlapsSource(_))
        ));
        Ok(())
    }

    #[test]
    fn video_plans_reject_odd_and_oversized_dimensions() -> Result<(), DomainError> {
        let mode = StackExecutionMode::video(
            VideoContainerFormat::Mp4,
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
        )?;
        let make = |width, height| {
            FrameSequencePlan::new(
                vec![StackFrame::new(
                    PathBuf::from("a.png"),
                    ImageInputFormat::Png,
                    false,
                )?],
                ImageDimensions::new(width, height)?,
                StackFps::DEFAULT,
                PathBuf::from("out.mp4"),
                mode,
                NonZeroUsize::MIN,
            )
        };
        assert!(matches!(
            make(21, 20),
            Err(DomainError::StackOddVideoDimensions { .. })
        ));
        assert!(matches!(
            make(65_536, 20),
            Err(DomainError::StackVideoDimensionLimit { .. })
        ));
        let good = make(20, 20)?;
        assert_eq!(good.loop_count(), None);
        assert_eq!(good.target_container(), VideoContainerFormat::Mp4);
        assert_eq!(good.video_codec(), TargetVideoCodec::H264);
        Ok(())
    }

    #[test]
    fn video_constructor_rejects_animation_containers_and_mismatches() {
        assert!(matches!(
            StackExecutionMode::video(
                VideoContainerFormat::Gif,
                VideoEncodeMode::CpuH264 {
                    rate_control: VideoRateControl::DEFAULT,
                }
            ),
            Err(DomainError::UnsupportedVideoCodec { .. })
        ));
        assert!(matches!(
            StackExecutionMode::video(
                VideoContainerFormat::Webm,
                VideoEncodeMode::CpuMjpeg {
                    quality: VideoQuality::DEFAULT,
                }
            ),
            Err(DomainError::UnsupportedVideoCodec { .. })
        ));
    }

    #[test]
    fn resolver_picks_zero_fallback_backends() -> Result<(), DomainError> {
        let host = nvenc_host()?;
        let rc = VideoRateControl::DEFAULT;
        let resolve = |container, codec, backend, host: Option<&HostHardware>| {
            StackExecutionMode::resolve_with_hardware(
                container, codec, rc, None, backend, true, host,
            )
        };
        assert_eq!(
            resolve(
                VideoContainerFormat::Mp4,
                Some(TargetVideoCodec::H264),
                StackBackendRequest::Auto,
                Some(&host)
            )?
            .backend(),
            HardwareBackend::Gpu(GpuBackend::Nvenc)
        );
        assert_eq!(
            resolve(
                VideoContainerFormat::Mp4,
                Some(TargetVideoCodec::H264),
                StackBackendRequest::Auto,
                None
            )?
            .backend(),
            HardwareBackend::Cpu
        );
        assert_eq!(
            resolve(
                VideoContainerFormat::Webm,
                Some(TargetVideoCodec::Vp9),
                StackBackendRequest::Auto,
                None
            )?
            .backend(),
            HardwareBackend::Cpu
        );
        assert_eq!(
            resolve(
                VideoContainerFormat::Gif,
                None,
                StackBackendRequest::Auto,
                None
            )?,
            StackExecutionMode::cpu_animation(StackTargetFormat::Gif, VideoQuality::DEFAULT)
        );
        Ok(())
    }

    #[test]
    fn resolver_rejects_all_silent_fallback_requests() -> Result<(), DomainError> {
        let host = nvenc_host()?;
        let rc = VideoRateControl::DEFAULT;
        let resolve = |container, codec, backend, explicit_loop, compiled, host| {
            StackExecutionMode::resolve_with_hardware(
                container,
                codec,
                rc,
                explicit_loop,
                backend,
                compiled,
                host,
            )
        };
        assert!(matches!(
            resolve(
                VideoContainerFormat::Mp4,
                None,
                StackBackendRequest::Auto,
                Some(AnimationLoop::Infinite),
                true,
                Some(&host)
            ),
            Err(DomainError::UnsupportedStackLoop { .. })
        ));
        assert!(matches!(
            resolve(
                VideoContainerFormat::Mkv,
                Some(TargetVideoCodec::Hevc),
                StackBackendRequest::Auto,
                None,
                true,
                None
            ),
            Err(DomainError::UnsupportedVideoPipeline { .. })
        ));
        assert!(matches!(
            resolve(
                VideoContainerFormat::Mp4,
                Some(TargetVideoCodec::H264),
                StackBackendRequest::Explicit(HardwareBackend::Gpu(GpuBackend::Nvenc)),
                None,
                false,
                Some(&host)
            ),
            Err(DomainError::UnsupportedVideoPipeline { .. })
        ));
        assert!(matches!(
            resolve(
                VideoContainerFormat::Mp4,
                Some(TargetVideoCodec::H264),
                StackBackendRequest::Explicit(HardwareBackend::Gpu(GpuBackend::Vulkan)),
                None,
                true,
                Some(&host)
            ),
            Err(DomainError::UnsupportedVideoPipeline { .. })
        ));
        assert!(matches!(
            resolve(
                VideoContainerFormat::Mp4,
                Some(TargetVideoCodec::Hevc),
                StackBackendRequest::Explicit(HardwareBackend::Cpu),
                None,
                true,
                Some(&host)
            ),
            Err(DomainError::UnsupportedVideoPipeline { .. })
        ));
        assert!(matches!(
            resolve(
                VideoContainerFormat::Webm,
                Some(TargetVideoCodec::H264),
                StackBackendRequest::Auto,
                None,
                true,
                Some(&host)
            ),
            Err(DomainError::UnsupportedVideoCodec { .. })
        ));
        Ok(())
    }

    #[test]
    fn report_exposes_execution_path() -> Result<(), DomainError> {
        let report = FrameSequenceReport::new(
            NonZeroU64::MIN,
            ImageDimensions::new(16, 16)?,
            10.0,
            StackExecutionMode::cpu_animation(StackTargetFormat::Gif, VideoQuality::DEFAULT),
            PathBuf::from("out.gif"),
            NonZeroU64::MIN,
            NonZeroUsize::MIN,
            None,
        );
        assert_eq!(report.target_container(), VideoContainerFormat::Gif);
        assert_eq!(report.video_codec(), TargetVideoCodec::Gif);
        assert_eq!(report.backend(), HardwareBackend::Cpu);
        assert_eq!(report.hardware_device(), None);
        Ok(())
    }

    #[test]
    fn parses_animation_loop_values() {
        assert_eq!(
            parse_animation_loop("infinite"),
            Ok(AnimationLoop::Infinite)
        );
        assert_eq!(parse_animation_loop("0"), Ok(AnimationLoop::Infinite));
        assert_eq!(
            parse_animation_loop("1"),
            Ok(AnimationLoop::Times(
                NonZeroU16::new(1).unwrap_or(NonZeroU16::MIN)
            ))
        );
        assert!(parse_animation_loop("65536").is_err());
        assert!(parse_animation_loop("-1").is_err());
        assert!(parse_animation_loop("many").is_err());
    }
}
