use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

/// High-level media category supported by `fastconv`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    Video,
    Audio,
    Image,
}

/// Subset of media kinds eligible for zero-copy GPU execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuMediaKind {
    Video,
    Image,
}

/// Concrete hardware acceleration backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    Nvenc,
    Vaapi,
    VideoToolbox,
    Vulkan,
}

/// Execution backend explicitly selected for a conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    IdenticalSourceAndTarget(PathBuf),
    UnsupportedBackend {
        media_kind: MediaKind,
        backend: HardwareBackend,
    },
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

impl fmt::Display for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPath(role) => write!(f, "{role} path must not be empty"),
            Self::IdenticalSourceAndTarget(path) => {
                write!(
                    f,
                    "source and target path must differ to avoid in-place corruption: {}",
                    path.display()
                )
            }
            Self::UnsupportedBackend {
                media_kind,
                backend,
            } => write!(
                f,
                "backend '{backend}' is not supported for {media_kind} conversions"
            ),
        }
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
