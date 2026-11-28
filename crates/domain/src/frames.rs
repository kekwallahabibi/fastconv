//! Pure data models for still-frame extraction from video and animation sources.
//!
//! No I/O happens here: every external fact (filesystem, media headers, host
//! parallelism) is validated once in `fastconv-probe` before a plan is built.

use std::fmt;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};

#[cfg(feature = "serde")]
use serde::Serialize;

use crate::{
    DomainError, HardwareBackend, ImageDimensions, ImageExecutionMode, ImageOutputFormat, PathRole,
    VideoContainerFormat,
};

/// Which decoded frames are kept. Illegal states (a non-positive, non-finite
/// FPS, or a zero sampling interval) can only be built through the validated
/// constructor.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum FrameSelection {
    /// Keep every frame in the requested range.
    All,
    /// Keep every N-th frame in the requested range, starting with its first.
    Every(NonZeroU32),
    /// Keep frames at whole `1 / fps` second intervals from the range's first frame.
    TargetFps(f64),
}

impl FrameSelection {
    /// Creates a validated target-FPS selection.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidFrameSelection`] when `fps` is NaN,
    /// infinite, or not strictly positive.
    pub fn target_fps(fps: f64) -> Result<Self, DomainError> {
        if !fps.is_finite() || fps <= 0.0 {
            return Err(DomainError::InvalidFrameSelection {
                reason: format!("--fps must be a finite number greater than zero, got {fps}"),
            });
        }
        Ok(Self::TargetFps(fps))
    }

    /// Human-readable description for progress and diagnostics.
    #[must_use]
    pub fn describe(self) -> String {
        match self {
            Self::All => "all frames".to_owned(),
            Self::Every(n) => format!("every {n} frames"),
            Self::TargetFps(fps) => format!("{fps} fps"),
        }
    }
}

impl fmt::Display for FrameSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

/// Inclusive, zero-based frame range `[start..=end]`; `None` means "to the end".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct FrameRange {
    start: u64,
    end: Option<u64>,
}

impl FrameRange {
    /// Creates a validated range.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidFrameRange`] when `start > end`.
    pub const fn new(start: u64, end: Option<u64>) -> Result<Self, DomainError> {
        if let Some(end) = end
            && start > end
        {
            return Err(DomainError::InvalidFrameRange { start, end });
        }
        Ok(Self { start, end })
    }

    #[must_use]
    pub const fn start(self) -> u64 {
        self.start
    }

    #[must_use]
    pub const fn end(self) -> Option<u64> {
        self.end
    }

    /// Whether `index` falls inside this inclusive range.
    #[must_use]
    pub const fn contains(self, index: u64) -> bool {
        if index < self.start {
            return false;
        }
        match self.end {
            Some(end) => index <= end,
            None => true,
        }
    }

    /// Whether decoding can stop because `index` is past the last wanted frame.
    #[must_use]
    pub const fn is_past_end(self, index: u64) -> bool {
        match self.end {
            Some(end) => index > end,
            None => false,
        }
    }
}

impl fmt::Display for FrameRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.end {
            Some(end) => write!(f, "{}..={}", self.start, end),
            None => write!(f, "{}..", self.start),
        }
    }
}

/// Icon encoders cannot represent dimensions above 256 on either axis.
const ICO_MAX_AXIS: u32 = 256;
/// GIF's logical screen descriptor stores each axis as a `u16`.
const GIF_MAX_AXIS: u32 = 65_535;
/// JPEG's SOF stores each axis as a `u16`.
const JPEG_MAX_AXIS: u32 = 65_535;
/// The WebP RIFF container stores canvas dimensions in 14 bits.
const WEBP_MAX_AXIS: u32 = 16_383;
/// HEVC-Intra encoders bound each axis and the total luma sample count.
const HEIC_MAX_AXIS: u32 = 256 * 512;
const HEIC_MAX_PIXELS: u64 = 4_096 * 4_096;

fn validate_dimensions(
    format: ImageOutputFormat,
    dimensions: ImageDimensions,
) -> Result<(), DomainError> {
    let width = dimensions.width().get();
    let height = dimensions.height().get();
    let (max_width, max_height) = match format {
        ImageOutputFormat::Ico => (ICO_MAX_AXIS, ICO_MAX_AXIS),
        ImageOutputFormat::Gif => (GIF_MAX_AXIS, GIF_MAX_AXIS),
        ImageOutputFormat::Jpeg => (JPEG_MAX_AXIS, JPEG_MAX_AXIS),
        ImageOutputFormat::Webp => (WEBP_MAX_AXIS, WEBP_MAX_AXIS),
        ImageOutputFormat::Heic => (HEIC_MAX_AXIS, HEIC_MAX_AXIS),
        ImageOutputFormat::Png
        | ImageOutputFormat::Bmp
        | ImageOutputFormat::Qoi
        | ImageOutputFormat::Tiff
        | ImageOutputFormat::Jxl
        | ImageOutputFormat::Avif
        | ImageOutputFormat::Pnm => return Ok(()),
    };
    if width > max_width || height > max_height {
        return Err(DomainError::FrameDimensionLimit {
            format,
            width,
            height,
            max_width,
            max_height,
        });
    }
    if format == ImageOutputFormat::Heic {
        let pixels = u64::from(width).saturating_mul(u64::from(height));
        if pixels > HEIC_MAX_PIXELS {
            return Err(DomainError::FrameDimensionLimit {
                format,
                width,
                height,
                max_width: 4_096,
                max_height: 4_096,
            });
        }
    }
    Ok(())
}

/// A validated request to extract still frames from one video/animation source.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct FrameExtractionPlan {
    source: PathBuf,
    source_container: VideoContainerFormat,
    source_dimensions: Option<ImageDimensions>,
    output_dir: PathBuf,
    output_format: ImageOutputFormat,
    mode: ImageExecutionMode,
    selection: FrameSelection,
    range: FrameRange,
    worker_threads: NonZeroUsize,
}

impl FrameExtractionPlan {
    /// Builds a plan without performing any I/O.
    ///
    /// # Errors
    ///
    /// Rejects empty/identical paths, GPU backends (no silent fallback), a mode
    /// whose format disagrees with `output_format`, an invalid frame range, and
    /// static output-format dimension limits when dimensions are known.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source: PathBuf,
        source_container: VideoContainerFormat,
        source_dimensions: Option<ImageDimensions>,
        output_dir: PathBuf,
        output_format: ImageOutputFormat,
        mode: ImageExecutionMode,
        selection: FrameSelection,
        range: FrameRange,
        worker_threads: NonZeroUsize,
    ) -> Result<Self, DomainError> {
        if source.as_os_str().is_empty() {
            return Err(DomainError::EmptyPath(PathRole::Source));
        }
        if output_dir.as_os_str().is_empty() {
            return Err(DomainError::EmptyPath(PathRole::Target));
        }
        if source == output_dir {
            return Err(DomainError::IdenticalSourceAndTarget(source));
        }
        if let HardwareBackend::Gpu(backend) = mode.backend() {
            return Err(DomainError::FrameSourceRejected {
                reason: format!(
                    "frame extraction runs on CPU only; backend 'gpu:{backend}' is rejected with no silent CPU fallback"
                ),
            });
        }
        if mode.output_format() != output_format {
            return Err(DomainError::FrameSourceRejected {
                reason: format!(
                    "encoder mode '{}' does not produce .{output_format}",
                    mode.output_format()
                ),
            });
        }
        if let Some(dimensions) = source_dimensions {
            validate_dimensions(output_format, dimensions)?;
        }

        Ok(Self {
            source,
            source_container,
            source_dimensions,
            output_dir,
            output_format,
            mode,
            selection,
            range,
            worker_threads,
        })
    }

    #[must_use]
    pub fn source(&self) -> &Path {
        &self.source
    }

    #[must_use]
    pub const fn source_container(&self) -> VideoContainerFormat {
        self.source_container
    }

    #[must_use]
    pub const fn source_dimensions(&self) -> Option<ImageDimensions> {
        self.source_dimensions
    }

    #[must_use]
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    #[must_use]
    pub const fn output_format(&self) -> ImageOutputFormat {
        self.output_format
    }

    #[must_use]
    pub const fn mode(&self) -> ImageExecutionMode {
        self.mode
    }

    #[must_use]
    pub const fn backend(&self) -> HardwareBackend {
        self.mode.backend()
    }

    #[must_use]
    pub const fn selection(&self) -> FrameSelection {
        self.selection
    }

    #[must_use]
    pub const fn range(&self) -> FrameRange {
        self.range
    }

    #[must_use]
    pub const fn worker_threads(&self) -> NonZeroUsize {
        self.worker_threads
    }
}

/// Execution summary returned after a frame extraction completes.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct FrameExtractionReport {
    frames_extracted: NonZeroU64,
    width: NonZeroU32,
    height: NonZeroU32,
    output_format: ImageOutputFormat,
    output_dir: PathBuf,
    total_output_bytes: NonZeroU64,
    worker_threads: NonZeroUsize,
}

impl FrameExtractionReport {
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        frames_extracted: NonZeroU64,
        width: NonZeroU32,
        height: NonZeroU32,
        output_format: ImageOutputFormat,
        output_dir: PathBuf,
        total_output_bytes: NonZeroU64,
        worker_threads: NonZeroUsize,
    ) -> Self {
        Self {
            frames_extracted,
            width,
            height,
            output_format,
            output_dir,
            total_output_bytes,
            worker_threads,
        }
    }

    #[must_use]
    pub const fn frames_extracted(&self) -> NonZeroU64 {
        self.frames_extracted
    }

    #[must_use]
    pub const fn width(&self) -> NonZeroU32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> NonZeroU32 {
        self.height
    }

    #[must_use]
    pub const fn output_format(&self) -> ImageOutputFormat {
        self.output_format
    }

    #[must_use]
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    #[must_use]
    pub const fn total_output_bytes(&self) -> NonZeroU64 {
        self.total_output_bytes
    }

    #[must_use]
    pub const fn worker_threads(&self) -> NonZeroUsize {
        self.worker_threads
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameExtractionPlan, FrameRange, FrameSelection};
    use crate::{
        DomainError, HardwareBackend, ImageDimensions, ImageExecutionMode, ImageOutputFormat,
        PathRole, VideoContainerFormat,
    };
    use std::num::NonZeroUsize;
    use std::path::PathBuf;

    fn plan(format: ImageOutputFormat) -> Result<FrameExtractionPlan, DomainError> {
        let mode = ImageExecutionMode::resolve_cpu(format, None)?;
        FrameExtractionPlan::new(
            PathBuf::from("in.mp4"),
            VideoContainerFormat::Mp4,
            Some(ImageDimensions::new(320, 240)?),
            PathBuf::from("frames"),
            format,
            mode,
            FrameSelection::All,
            FrameRange::new(0, None)?,
            NonZeroUsize::new(2).ok_or(DomainError::EmptyBatch)?,
        )
    }

    #[test]
    fn validates_fps_selection() -> Result<(), DomainError> {
        assert_eq!(
            FrameSelection::target_fps(29.97)?,
            FrameSelection::TargetFps(29.97)
        );
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            assert!(matches!(
                FrameSelection::target_fps(bad),
                Err(DomainError::InvalidFrameSelection { .. })
            ));
        }
        Ok(())
    }

    #[test]
    fn frame_range_is_inclusive_and_rejects_inverted_bounds() -> Result<(), DomainError> {
        let range = FrameRange::new(3, Some(5))?;
        assert!(!range.contains(2));
        assert!(range.contains(3));
        assert!(range.contains(5));
        assert!(!range.contains(6));
        assert!(range.is_past_end(6));
        assert!(!range.is_past_end(5));
        assert_eq!(
            FrameRange::new(10, Some(5)),
            Err(DomainError::InvalidFrameRange { start: 10, end: 5 })
        );
        let open = FrameRange::new(2, None)?;
        assert!(open.contains(u64::MAX));
        assert!(!open.is_past_end(u64::MAX));
        Ok(())
    }

    #[test]
    fn builds_valid_plan() -> Result<(), DomainError> {
        let built = plan(ImageOutputFormat::Png)?;
        assert_eq!(built.source(), PathBuf::from("in.mp4"));
        assert_eq!(built.output_dir(), PathBuf::from("frames"));
        assert_eq!(built.output_format(), ImageOutputFormat::Png);
        assert_eq!(built.backend(), HardwareBackend::Cpu);
        Ok(())
    }

    #[test]
    fn rejects_gpu_backend_without_fallback() -> Result<(), DomainError> {
        // A GPU mode is built through the public enum to prove the guard does
        // not depend on how the mode was obtained.
        let result = FrameExtractionPlan::new(
            PathBuf::from("in.mp4"),
            VideoContainerFormat::Mp4,
            None,
            PathBuf::from("frames"),
            ImageOutputFormat::Heic,
            ImageExecutionMode::NvencHeic {
                quality: crate::ImageQuality::DEFAULT,
            },
            FrameSelection::All,
            FrameRange::new(0, None)?,
            NonZeroUsize::MIN,
        );
        assert!(matches!(
            result,
            Err(DomainError::FrameSourceRejected { ref reason }) if reason.contains("no silent CPU fallback")
        ));
        Ok(())
    }

    #[test]
    fn rejects_static_dimension_limits() -> Result<(), DomainError> {
        let mode = ImageExecutionMode::resolve_cpu(ImageOutputFormat::Ico, None)?;
        let oversized = FrameExtractionPlan::new(
            PathBuf::from("in.mp4"),
            VideoContainerFormat::Mp4,
            Some(ImageDimensions::new(257, 100)?),
            PathBuf::from("frames"),
            ImageOutputFormat::Ico,
            mode,
            FrameSelection::All,
            FrameRange::new(0, None)?,
            NonZeroUsize::MIN,
        );
        assert!(matches!(
            oversized,
            Err(DomainError::FrameDimensionLimit {
                format: ImageOutputFormat::Ico,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn rejects_empty_and_identical_paths() -> Result<(), DomainError> {
        let mode = ImageExecutionMode::resolve_cpu(ImageOutputFormat::Png, None)?;
        assert!(matches!(
            FrameExtractionPlan::new(
                PathBuf::new(),
                VideoContainerFormat::Mp4,
                None,
                PathBuf::from("frames"),
                ImageOutputFormat::Png,
                mode,
                FrameSelection::All,
                FrameRange::new(0, None)?,
                NonZeroUsize::MIN,
            ),
            Err(DomainError::EmptyPath(PathRole::Source))
        ));
        assert!(matches!(
            FrameExtractionPlan::new(
                PathBuf::from("same"),
                VideoContainerFormat::Mp4,
                None,
                PathBuf::from("same"),
                ImageOutputFormat::Png,
                mode,
                FrameSelection::All,
                FrameRange::new(0, None)?,
                NonZeroUsize::MIN,
            ),
            Err(DomainError::IdenticalSourceAndTarget(_))
        ));
        Ok(())
    }
}
