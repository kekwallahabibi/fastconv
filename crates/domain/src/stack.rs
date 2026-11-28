//! Pure data models for encoding an ordered sequence of still images into an
//! animation. No I/O happens here: every external fact is validated once in
//! `fastconv-probe` before a plan is built.
//!
//! [`StackExecutionMode`] has exactly one legal variant today. That is the
//! deliberate extension seam: adding video containers later means adding
//! variants to this enum and matching them in the pipeline, not loosening the
//! plan with flags, optional codec fields or a silent CPU fallback.

use std::fmt;
use std::num::{NonZeroU16, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};

#[cfg(feature = "serde")]
use serde::Serialize;

use crate::{
    AnimationLoop, DomainError, HardwareBackend, ImageDimensions, ImageInputFormat, PathRole,
    VideoContainerFormat, VideoQuality,
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

/// Finite, strictly positive animation frame rate. The bounds keep one frame
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

/// Compile-time-legal execution mode for a frame sequence. Every variant is
/// CPU-only by construction; there is no backend field that could disagree
/// with it, so a GPU request can only be rejected at the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum StackExecutionMode {
    CpuAnimation {
        format: StackTargetFormat,
        quality: VideoQuality,
    },
}

impl StackExecutionMode {
    #[must_use]
    pub const fn cpu_animation(format: StackTargetFormat, quality: VideoQuality) -> Self {
        Self::CpuAnimation { format, quality }
    }

    #[must_use]
    pub const fn target_format(self) -> StackTargetFormat {
        match self {
            Self::CpuAnimation { format, .. } => format,
        }
    }

    #[must_use]
    pub const fn quality(self) -> VideoQuality {
        match self {
            Self::CpuAnimation { quality, .. } => quality,
        }
    }

    #[must_use]
    pub const fn backend(self) -> HardwareBackend {
        HardwareBackend::Cpu
    }
}

impl fmt::Display for StackExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CpuAnimation { format, quality } => {
                write!(f, "cpu animation .{} q={}", format.extension(), quality)
            }
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
    loop_count: AnimationLoop,
    worker_threads: NonZeroUsize,
}

impl FrameSequencePlan {
    /// Builds a validated plan without performing any I/O.
    ///
    /// # Errors
    ///
    /// Rejects empty frame lists and target paths, a target that aliases any
    /// source path, dimensions beyond the target format's static limit, and
    /// any execution mode that is not CPU-only.
    pub fn new(
        frames: Vec<StackFrame>,
        dimensions: ImageDimensions,
        fps: StackFps,
        target_path: PathBuf,
        mode: StackExecutionMode,
        loop_count: AnimationLoop,
        worker_threads: NonZeroUsize,
    ) -> Result<Self, DomainError> {
        if frames.is_empty() {
            return Err(DomainError::EmptyFrameSequence);
        }
        if target_path.as_os_str().is_empty() {
            return Err(DomainError::EmptyPath(PathRole::Target));
        }
        if !matches!(mode, StackExecutionMode::CpuAnimation { .. }) {
            return Err(DomainError::UnsupportedStackBackend {
                backend: mode.backend(),
            });
        }
        let (max_width, max_height) = mode.target_format().max_dimensions();
        let width = dimensions.width().get();
        let height = dimensions.height().get();
        if width > max_width || height > max_height {
            return Err(DomainError::StackDimensionLimit {
                format: mode.target_format(),
                width,
                height,
                max_width,
                max_height,
            });
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
            loop_count,
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
    pub const fn target_format(&self) -> StackTargetFormat {
        self.mode.target_format()
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
    pub const fn loop_count(&self) -> AnimationLoop {
        self.loop_count
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
    target_format: StackTargetFormat,
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
        target_format: StackTargetFormat,
        target_path: PathBuf,
        output_bytes: NonZeroU64,
        worker_threads: NonZeroUsize,
    ) -> Self {
        Self {
            frames_encoded,
            dimensions,
            fps,
            target_format,
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
    pub const fn target_format(&self) -> StackTargetFormat {
        self.target_format
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
        FrameSequencePlan, StackExecutionMode, StackFps, StackFrame, StackTargetFormat,
        parse_animation_loop,
    };
    use crate::{AnimationLoop, DomainError, ImageDimensions, ImageInputFormat, VideoQuality};
    use std::num::{NonZeroU16, NonZeroUsize};
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
            AnimationLoop::Infinite,
            NonZeroUsize::MIN,
        )
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
    fn fps_is_bounded_and_drift_free() -> Result<(), DomainError> {
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
        let max = StackFps::new(StackFps::MAX)?;
        assert_eq!(max.frame_timestamp_us(u64::MAX), u64::MAX);
        Ok(())
    }

    #[test]
    fn validates_plan_invariants() -> Result<(), DomainError> {
        let built = plan()?;
        assert_eq!(built.frame_count().get(), 1);
        assert_eq!(built.target_format(), StackTargetFormat::Gif);
        assert_eq!(built.backend(), crate::HardwareBackend::Cpu);

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
            AnimationLoop::Infinite,
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
            AnimationLoop::Infinite,
            NonZeroUsize::MIN,
        );
        assert!(matches!(
            aliasing,
            Err(DomainError::StackTargetOverlapsSource(_))
        ));
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
