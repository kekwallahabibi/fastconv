use std::collections::HashSet;
use std::fmt;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::{DomainError, PathRole};

/// Non-zero pixel dimensions of a decoded or probed image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct ImageDimensions {
    width: NonZeroU32,
    height: NonZeroU32,
}

impl ImageDimensions {
    /// Creates validated non-zero [`ImageDimensions`].
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidDimensions`] if `width` or `height` is zero.
    pub const fn new(width: u32, height: u32) -> Result<Self, DomainError> {
        let Some(w) = NonZeroU32::new(width) else {
            return Err(DomainError::InvalidDimensions { width, height });
        };
        let Some(h) = NonZeroU32::new(height) else {
            return Err(DomainError::InvalidDimensions { width, height });
        };
        Ok(Self {
            width: w,
            height: h,
        })
    }

    #[must_use]
    pub const fn width(self) -> NonZeroU32 {
        self.width
    }

    #[must_use]
    pub const fn height(self) -> NonZeroU32 {
        self.height
    }
}

impl fmt::Display for ImageDimensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// Validated lossy WebP quality factor in `0..=100`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct WebpQuality(u8);

impl WebpQuality {
    pub const DEFAULT: Self = Self(80);
    pub const MAX: u8 = 100;

    /// Creates a validated [`WebpQuality`] in `0..=100`.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidQuality`] when `value > 100`.
    pub const fn new(value: u8) -> Result<Self, DomainError> {
        if value > Self::MAX {
            return Err(DomainError::InvalidQuality {
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

    #[must_use]
    pub fn as_f32(self) -> f32 {
        f32::from(self.0)
    }
}

/// WebP compression mode, making lossless + quality contradictions unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum WebpEncoding {
    Lossy { quality: WebpQuality },
    Lossless,
}

impl fmt::Display for WebpEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lossy { quality } => write!(f, "lossy q={}", quality.get()),
            Self::Lossless => f.write_str("lossless"),
        }
    }
}

/// Single validated PNG-to-WebP conversion item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PngToWebpItem {
    source: PathBuf,
    target: PathBuf,
    dimensions: ImageDimensions,
}

impl PngToWebpItem {
    /// Creates a validated [`PngToWebpItem`].
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyPath`] if `source` or `target` is empty, or
    /// [`DomainError::IdenticalSourceAndTarget`] if both paths are identical.
    pub fn new(
        source: PathBuf,
        target: PathBuf,
        dimensions: ImageDimensions,
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
            dimensions,
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
    pub const fn dimensions(&self) -> ImageDimensions {
        self.dimensions
    }
}

/// Validated non-empty batch of PNG-to-WebP conversions with conflict-free target paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageBatchPlan {
    items: Vec<PngToWebpItem>,
    encoding: WebpEncoding,
    worker_threads: NonZeroUsize,
}

impl ImageBatchPlan {
    /// Creates a validated [`ImageBatchPlan`].
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyBatch`] if `items` is empty,
    /// [`DomainError::DuplicateTargetPath`] if two items target the same output path, or
    /// [`DomainError::TargetOverwritesSource`] if any target path equals a source path in the batch.
    pub fn new(
        items: Vec<PngToWebpItem>,
        encoding: WebpEncoding,
        worker_threads: NonZeroUsize,
    ) -> Result<Self, DomainError> {
        if items.is_empty() {
            return Err(DomainError::EmptyBatch);
        }

        let source_paths: HashSet<&Path> = items.iter().map(PngToWebpItem::source).collect();
        let mut seen_targets = HashSet::with_capacity(items.len());

        for item in &items {
            if !seen_targets.insert(item.target()) {
                return Err(DomainError::DuplicateTargetPath(
                    item.target().to_path_buf(),
                ));
            }
            if source_paths.contains(item.target()) {
                return Err(DomainError::TargetOverwritesSource(
                    item.target().to_path_buf(),
                ));
            }
        }

        Ok(Self {
            items,
            encoding,
            worker_threads,
        })
    }

    #[must_use]
    pub fn items(&self) -> &[PngToWebpItem] {
        &self.items
    }

    #[must_use]
    pub const fn encoding(&self) -> WebpEncoding {
        self.encoding
    }

    #[must_use]
    pub const fn worker_threads(&self) -> NonZeroUsize {
        self.worker_threads
    }

    #[must_use]
    pub fn effective_workers(&self) -> NonZeroUsize {
        let count = self.worker_threads.get().min(self.items.len()).max(1);
        NonZeroUsize::new(count).unwrap_or(NonZeroUsize::MIN)
    }
}

/// Execution summary returned after completing an [`ImageBatchPlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ImageConversionReport {
    files_converted: NonZeroUsize,
    total_input_bytes: NonZeroU64,
    total_output_bytes: NonZeroU64,
    worker_threads_used: NonZeroUsize,
}

impl ImageConversionReport {
    #[must_use]
    pub const fn new(
        files_converted: NonZeroUsize,
        total_input_bytes: NonZeroU64,
        total_output_bytes: NonZeroU64,
        worker_threads_used: NonZeroUsize,
    ) -> Self {
        Self {
            files_converted,
            total_input_bytes,
            total_output_bytes,
            worker_threads_used,
        }
    }

    #[must_use]
    pub const fn files_converted(self) -> NonZeroUsize {
        self.files_converted
    }

    #[must_use]
    pub const fn total_input_bytes(self) -> NonZeroU64 {
        self.total_input_bytes
    }

    #[must_use]
    pub const fn total_output_bytes(self) -> NonZeroU64 {
        self.total_output_bytes
    }

    #[must_use]
    pub const fn worker_threads_used(self) -> NonZeroUsize {
        self.worker_threads_used
    }
}

#[cfg(test)]
mod tests {
    use super::{ImageBatchPlan, ImageDimensions, PngToWebpItem, WebpEncoding, WebpQuality};
    use crate::{DomainError, PathRole};
    use std::num::NonZeroUsize;
    use std::path::PathBuf;

    #[test]
    fn constructs_valid_batch_and_caps_effective_workers() -> Result<(), DomainError> {
        let dims = ImageDimensions::new(3840, 2160)?;
        assert_eq!(dims.to_string(), "3840x2160");

        let items = vec![
            PngToWebpItem::new(PathBuf::from("a.png"), PathBuf::from("a.webp"), dims)?,
            PngToWebpItem::new(PathBuf::from("b.png"), PathBuf::from("b.webp"), dims)?,
        ];
        let requested_workers = NonZeroUsize::new(8).ok_or(DomainError::EmptyBatch)?;
        let plan = ImageBatchPlan::new(
            items,
            WebpEncoding::Lossy {
                quality: WebpQuality::DEFAULT,
            },
            requested_workers,
        )?;

        assert_eq!(plan.items().len(), 2);
        assert_eq!(plan.worker_threads().get(), 8);
        assert_eq!(plan.effective_workers().get(), 2);
        Ok(())
    }

    #[test]
    fn rejects_invalid_dimensions_and_quality() {
        assert_eq!(
            ImageDimensions::new(0, 1080),
            Err(DomainError::InvalidDimensions {
                width: 0,
                height: 1080
            })
        );
        assert_eq!(
            ImageDimensions::new(1920, 0),
            Err(DomainError::InvalidDimensions {
                width: 1920,
                height: 0
            })
        );
        assert_eq!(
            WebpQuality::new(101),
            Err(DomainError::InvalidQuality {
                value: 101,
                max: 100
            })
        );
        assert!(WebpQuality::new(100).is_ok());
    }

    #[test]
    fn rejects_empty_batch_duplicate_targets_and_source_overwrite() -> Result<(), DomainError> {
        let workers = NonZeroUsize::MIN;
        assert_eq!(
            ImageBatchPlan::new(Vec::new(), WebpEncoding::Lossless, workers),
            Err(DomainError::EmptyBatch)
        );

        let dims = ImageDimensions::new(64, 64)?;
        let dup_items = vec![
            PngToWebpItem::new(PathBuf::from("a.png"), PathBuf::from("same.webp"), dims)?,
            PngToWebpItem::new(PathBuf::from("b.png"), PathBuf::from("same.webp"), dims)?,
        ];
        assert_eq!(
            ImageBatchPlan::new(dup_items, WebpEncoding::Lossless, workers),
            Err(DomainError::DuplicateTargetPath(PathBuf::from("same.webp")))
        );

        let overwrite_items = vec![
            PngToWebpItem::new(PathBuf::from("a.png"), PathBuf::from("b.png"), dims)?,
            PngToWebpItem::new(PathBuf::from("b.png"), PathBuf::from("c.webp"), dims)?,
        ];
        assert_eq!(
            ImageBatchPlan::new(overwrite_items, WebpEncoding::Lossless, workers),
            Err(DomainError::TargetOverwritesSource(PathBuf::from("b.png")))
        );

        assert_eq!(
            PngToWebpItem::new(PathBuf::new(), PathBuf::from("out.webp"), dims),
            Err(DomainError::EmptyPath(PathRole::Source))
        );
        Ok(())
    }
}
