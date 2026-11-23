use std::collections::HashSet;
use std::fmt;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::{DomainError, GpuBackend, GpuCapability, HardwareBackend, HostHardware, PathRole};

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

/// Supported still-image input container formats validated at the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum ImageInputFormat {
    Png,
    Jpeg,
}

impl ImageInputFormat {
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
        }
    }
}

impl fmt::Display for ImageInputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.extension())
    }
}

/// Supported still-image output container formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum ImageOutputFormat {
    Webp,
    Heic,
    Jpeg,
}

impl ImageOutputFormat {
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Webp => "webp",
            Self::Heic => "heic",
            Self::Jpeg => "jpg",
        }
    }
}

impl fmt::Display for ImageOutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.extension())
    }
}

/// Validated lossy image quality factor in `0..=100`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct ImageQuality(u8);

impl ImageQuality {
    pub const DEFAULT: Self = Self(80);
    pub const MAX: u8 = 100;

    /// Creates a validated [`ImageQuality`] in `0..=100`.
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

/// Compression mode requested for an image conversion, making lossless + quality
/// contradictions unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum ImageCompression {
    Lossy { quality: ImageQuality },
    Lossless,
}

impl fmt::Display for ImageCompression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lossy { quality } => write!(f, "lossy q={}", quality.get()),
            Self::Lossless => f.write_str("lossless"),
        }
    }
}

/// Compile-time-legal combination of output image format, codec mode, and hardware backend.
///
/// Illegal pairings (e.g. NVENC + WebP/JPEG, Vulkan + WebP/HEIC/JPEG, CPU + HEIC,
/// lossless + JPEG/HEIC, or hardware-absent video encoder backends) cannot be constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum ImageExecutionMode {
    CpuWebp(ImageCompression),
    CpuJpeg { quality: ImageQuality },
    NvencHeic { quality: ImageQuality },
}

impl ImageExecutionMode {
    /// Resolves a CPU-only [`ImageExecutionMode`] for `format` without probing GPU hardware.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedImagePipeline`] if `format` has no native CPU encoder
    /// or the requested compression mode is unsupported for `format`.
    pub fn resolve_cpu(
        format: ImageOutputFormat,
        compression: ImageCompression,
    ) -> Result<Self, DomainError> {
        match format {
            ImageOutputFormat::Webp => Ok(Self::CpuWebp(compression)),
            ImageOutputFormat::Jpeg => match compression {
                ImageCompression::Lossy { quality } => Ok(Self::CpuJpeg { quality }),
                ImageCompression::Lossless => Err(DomainError::UnsupportedImagePipeline {
                    format,
                    backend: HardwareBackend::Cpu,
                    reason:
                        "JPEG is a lossy format and does not support --lossless; use --quality 0..=100"
                            .to_owned(),
                }),
            },
            ImageOutputFormat::Heic => Err(DomainError::UnsupportedImagePipeline {
                format,
                backend: HardwareBackend::Cpu,
                reason: "no native CPU HEVC encoder is implemented for .heic; use --backend nvenc on NVIDIA hardware".to_owned(),
            }),
        }
    }

    /// Resolves a validated [`ImageExecutionMode`] against probed [`HostHardware`].
    ///
    /// Never performs silent CPU fallbacks: rejects backends that are physically absent,
    /// decode-only, or do not support the selected image codec with a concrete reason.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedImagePipeline`] if the requested combination of
    /// format, compression mode, and backend is not supported on `hardware`.
    pub fn resolve(
        format: ImageOutputFormat,
        compression: ImageCompression,
        backend: HardwareBackend,
        hardware: &HostHardware,
    ) -> Result<Self, DomainError> {
        match (format, backend) {
            (format, HardwareBackend::Cpu) => Self::resolve_cpu(format, compression),
            (ImageOutputFormat::Webp, HardwareBackend::Gpu(GpuBackend::Nvenc)) => Err(
                nvenc_unsupported_image_error(format, "VP8/WebP", ".webp", hardware),
            ),
            (ImageOutputFormat::Jpeg, HardwareBackend::Gpu(GpuBackend::Nvenc)) => Err(
                nvenc_unsupported_image_error(format, "JPEG", ".jpg", hardware),
            ),
            (ImageOutputFormat::Heic, HardwareBackend::Gpu(GpuBackend::Nvenc)) => {
                if !hardware.supports_video_encode(GpuBackend::Nvenc) {
                    return Err(DomainError::UnsupportedImagePipeline {
                        format,
                        backend,
                        reason: "no NVENC hardware video encoder detected on this host".to_owned(),
                    });
                }
                match compression {
                    ImageCompression::Lossy { quality } => Ok(Self::NvencHeic { quality }),
                    ImageCompression::Lossless => Err(DomainError::UnsupportedImagePipeline {
                        format,
                        backend,
                        reason: "NVENC HEIC currently supports lossy HEVC-Intra (--quality 0..=100), not --lossless".to_owned(),
                    }),
                }
            }
            (format, HardwareBackend::Gpu(gpu_backend)) => {
                let reason = explain_unimplemented_or_absent_gpu(format, gpu_backend, hardware);
                Err(DomainError::UnsupportedImagePipeline {
                    format,
                    backend: HardwareBackend::Gpu(gpu_backend),
                    reason,
                })
            }
        }
    }

    #[must_use]
    pub const fn output_format(self) -> ImageOutputFormat {
        match self {
            Self::CpuWebp(_) => ImageOutputFormat::Webp,
            Self::CpuJpeg { .. } => ImageOutputFormat::Jpeg,
            Self::NvencHeic { .. } => ImageOutputFormat::Heic,
        }
    }

    #[must_use]
    pub const fn backend(self) -> HardwareBackend {
        match self {
            Self::CpuWebp(_) | Self::CpuJpeg { .. } => HardwareBackend::Cpu,
            Self::NvencHeic { .. } => HardwareBackend::Gpu(GpuBackend::Nvenc),
        }
    }
}

impl fmt::Display for ImageExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CpuWebp(compression) => write!(f, "webp {compression}"),
            Self::CpuJpeg { quality } => write!(f, "jpeg q={}", quality.get()),
            Self::NvencHeic { quality } => {
                write!(f, "heic hevc-intra q={}", quality.get())
            }
        }
    }
}

fn nvenc_unsupported_image_error(
    format: ImageOutputFormat,
    codec_label: &str,
    cpu_ext: &str,
    hardware: &HostHardware,
) -> DomainError {
    let reason = if hardware.supports_video_encode(GpuBackend::Nvenc) {
        format!(
            "NVENC only provides hardware H.264/HEVC/AV1 encoders, not {codec_label}; use --backend cpu for {cpu_ext}, or target .heic (--to heic) for NVENC HEVC-Intra"
        )
    } else {
        format!("NVENC is not available on this host and does not provide a {codec_label} encoder")
    };
    DomainError::UnsupportedImagePipeline {
        format,
        backend: HardwareBackend::Gpu(GpuBackend::Nvenc),
        reason,
    }
}

fn explain_unimplemented_or_absent_gpu(
    format: ImageOutputFormat,
    gpu_backend: GpuBackend,
    hardware: &HostHardware,
) -> String {
    match hardware.find_gpu_backend(gpu_backend) {
        None => format!("backend 'gpu:{gpu_backend}' is not physically available on this host"),
        Some(entry) => match entry.capability() {
            GpuCapability::VideoDecodeOnly => format!(
                "{gpu_backend} on {} is decode-only ({}); hardware encoding is unavailable",
                entry.device_name(),
                entry.evidence()
            ),
            GpuCapability::ComputeOnly => format!(
                "{gpu_backend} on {} only exposes compute, not a native {format} hardware encoder",
                entry.device_name()
            ),
            GpuCapability::VideoEncodeOnly | GpuCapability::VideoEncodeAndDecode => format!(
                "{gpu_backend} hardware encoding for .{format} on {} is not implemented yet",
                entry.device_name()
            ),
        },
    }
}

/// Single validated still-image conversion item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageBatchItem {
    source: PathBuf,
    target: PathBuf,
    input_format: ImageInputFormat,
    dimensions: ImageDimensions,
}

impl ImageBatchItem {
    /// Creates a validated [`ImageBatchItem`].
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyPath`] if `source` or `target` is empty, or
    /// [`DomainError::IdenticalSourceAndTarget`] if both paths are identical.
    pub fn new(
        source: PathBuf,
        target: PathBuf,
        input_format: ImageInputFormat,
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
            input_format,
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
    pub const fn input_format(&self) -> ImageInputFormat {
        self.input_format
    }

    #[must_use]
    pub const fn dimensions(&self) -> ImageDimensions {
        self.dimensions
    }
}

/// Validated non-empty batch of image conversions with conflict-free target paths and a
/// compile-time-legal [`ImageExecutionMode`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageBatchPlan {
    items: Vec<ImageBatchItem>,
    mode: ImageExecutionMode,
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
        items: Vec<ImageBatchItem>,
        mode: ImageExecutionMode,
        worker_threads: NonZeroUsize,
    ) -> Result<Self, DomainError> {
        if items.is_empty() {
            return Err(DomainError::EmptyBatch);
        }

        let source_paths: HashSet<&Path> = items.iter().map(ImageBatchItem::source).collect();
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
            mode,
            worker_threads,
        })
    }

    #[must_use]
    pub fn items(&self) -> &[ImageBatchItem] {
        &self.items
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
    pub const fn output_format(&self) -> ImageOutputFormat {
        self.mode.output_format()
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ImageConversionReport {
    files_converted: NonZeroUsize,
    total_input_bytes: NonZeroU64,
    total_output_bytes: NonZeroU64,
    worker_threads_used: NonZeroUsize,
    backend: HardwareBackend,
    hardware_device: Option<String>,
}

impl ImageConversionReport {
    #[must_use]
    pub const fn new(
        files_converted: NonZeroUsize,
        total_input_bytes: NonZeroU64,
        total_output_bytes: NonZeroU64,
        worker_threads_used: NonZeroUsize,
        backend: HardwareBackend,
        hardware_device: Option<String>,
    ) -> Self {
        Self {
            files_converted,
            total_input_bytes,
            total_output_bytes,
            worker_threads_used,
            backend,
            hardware_device,
        }
    }

    #[must_use]
    pub const fn files_converted(&self) -> NonZeroUsize {
        self.files_converted
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
}

#[cfg(test)]
mod tests {
    use super::{
        ImageBatchItem, ImageBatchPlan, ImageCompression, ImageDimensions, ImageExecutionMode,
        ImageInputFormat, ImageOutputFormat, ImageQuality,
    };
    use crate::{
        CpuInfo, DetectedGpuBackend, DomainError, GpuBackend, GpuCapability, GpuDeviceType,
        GpuVendor, GraphicsApi, HardwareBackend, HostHardware, PathRole, SimdLevel,
        WgpuAdapterInfo,
    };
    use std::num::{NonZeroU64, NonZeroUsize};
    use std::path::PathBuf;

    fn sample_cpu() -> Result<CpuInfo, DomainError> {
        let cores = NonZeroUsize::new(4).ok_or(DomainError::EmptyBatch)?;
        let threads = NonZeroUsize::new(8).ok_or(DomainError::EmptyBatch)?;
        CpuInfo::new(
            "Test CPU",
            "x86_64",
            cores,
            threads,
            SimdLevel::X86Avx2Fma,
            NonZeroU64::new(16 * 1024 * 1024 * 1024),
        )
    }

    fn rtx_2060_hardware() -> Result<HostHardware, DomainError> {
        let adapter = WgpuAdapterInfo::new(
            "NVIDIA GeForce RTX 2060",
            GpuVendor::Nvidia,
            0x1e89,
            GpuDeviceType::Discrete,
            GraphicsApi::Vulkan,
            "NVIDIA",
            "620.73.04",
        )?;
        let backends = vec![
            DetectedGpuBackend::new(
                GpuBackend::Nvenc,
                GpuCapability::VideoEncodeAndDecode,
                "NVIDIA GeForce RTX 2060",
                "/usr/lib/libnvidia-encode.so.1",
            )?,
            DetectedGpuBackend::new(
                GpuBackend::Vulkan,
                GpuCapability::ComputeOnly,
                "NVIDIA GeForce RTX 2060",
                "wgpu vulkan",
            )?,
            DetectedGpuBackend::new(
                GpuBackend::Vaapi,
                GpuCapability::VideoDecodeOnly,
                "NVIDIA GeForce RTX 2060",
                "/usr/lib/libva.so.2 + /usr/lib/dri/nvidia_drv_video.so",
            )?,
        ];
        Ok(HostHardware::new(sample_cpu()?, vec![adapter], backends))
    }

    fn cpu_only_hardware() -> Result<HostHardware, DomainError> {
        Ok(HostHardware::new(sample_cpu()?, Vec::new(), Vec::new()))
    }

    #[test]
    fn resolves_valid_cpu_webp_cpu_jpeg_and_nvenc_heic_modes() -> Result<(), DomainError> {
        let rtx = rtx_2060_hardware()?;
        let cpu_only = cpu_only_hardware()?;
        let lossy = ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        };

        assert_eq!(
            ImageExecutionMode::resolve(
                ImageOutputFormat::Webp,
                lossy,
                HardwareBackend::Cpu,
                &cpu_only
            )?,
            ImageExecutionMode::CpuWebp(lossy)
        );
        assert_eq!(
            ImageExecutionMode::resolve(
                ImageOutputFormat::Jpeg,
                lossy,
                HardwareBackend::Cpu,
                &cpu_only
            )?,
            ImageExecutionMode::CpuJpeg {
                quality: ImageQuality::DEFAULT
            }
        );
        assert_eq!(
            ImageExecutionMode::resolve(
                ImageOutputFormat::Heic,
                lossy,
                HardwareBackend::Gpu(GpuBackend::Nvenc),
                &rtx
            )?,
            ImageExecutionMode::NvencHeic {
                quality: ImageQuality::DEFAULT
            }
        );
        Ok(())
    }

    #[test]
    fn rejects_absent_decode_only_and_codec_mismatched_backends() -> Result<(), DomainError> {
        let rtx = rtx_2060_hardware()?;
        let cpu_only = cpu_only_hardware()?;
        let lossy = ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        };

        let vulkan_webp_rtx = ImageExecutionMode::resolve(
            ImageOutputFormat::Webp,
            lossy,
            HardwareBackend::Gpu(GpuBackend::Vulkan),
            &rtx,
        );
        assert!(matches!(
            vulkan_webp_rtx,
            Err(DomainError::UnsupportedImagePipeline {
                format: ImageOutputFormat::Webp,
                backend: HardwareBackend::Gpu(GpuBackend::Vulkan),
                ref reason,
            }) if reason.contains("only exposes compute")
        ));

        let vulkan_no_gpu = ImageExecutionMode::resolve(
            ImageOutputFormat::Webp,
            lossy,
            HardwareBackend::Gpu(GpuBackend::Vulkan),
            &cpu_only,
        );
        assert!(matches!(
            vulkan_no_gpu,
            Err(DomainError::UnsupportedImagePipeline {
                format: ImageOutputFormat::Webp,
                backend: HardwareBackend::Gpu(GpuBackend::Vulkan),
                ..
            })
        ));

        let nvenc_webp = ImageExecutionMode::resolve(
            ImageOutputFormat::Webp,
            lossy,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            &rtx,
        );
        assert!(matches!(
            nvenc_webp,
            Err(DomainError::UnsupportedImagePipeline {
                format: ImageOutputFormat::Webp,
                backend: HardwareBackend::Gpu(GpuBackend::Nvenc),
                ref reason,
            }) if reason.contains("HEVC") && reason.contains(".heic")
        ));

        let vaapi_decode_only = ImageExecutionMode::resolve(
            ImageOutputFormat::Webp,
            lossy,
            HardwareBackend::Gpu(GpuBackend::Vaapi),
            &rtx,
        );
        assert!(matches!(
            vaapi_decode_only,
            Err(DomainError::UnsupportedImagePipeline {
                backend: HardwareBackend::Gpu(GpuBackend::Vaapi),
                ref reason,
                ..
            }) if reason.contains("decode-only")
        ));

        let qsv_missing = ImageExecutionMode::resolve(
            ImageOutputFormat::Webp,
            lossy,
            HardwareBackend::Gpu(GpuBackend::Qsv),
            &rtx,
        );
        assert!(matches!(
            qsv_missing,
            Err(DomainError::UnsupportedImagePipeline {
                backend: HardwareBackend::Gpu(GpuBackend::Qsv),
                ref reason,
                ..
            }) if reason.contains("not physically available")
        ));

        let nvenc_no_gpu = ImageExecutionMode::resolve(
            ImageOutputFormat::Heic,
            lossy,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            &cpu_only,
        );
        assert!(matches!(
            nvenc_no_gpu,
            Err(DomainError::UnsupportedImagePipeline {
                format: ImageOutputFormat::Heic,
                backend: HardwareBackend::Gpu(GpuBackend::Nvenc),
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn rejects_unsupported_jpeg_modes_and_backends() -> Result<(), DomainError> {
        let rtx = rtx_2060_hardware()?;
        let cpu_only = cpu_only_hardware()?;
        let lossy = ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        };

        let jpeg_lossless = ImageExecutionMode::resolve(
            ImageOutputFormat::Jpeg,
            ImageCompression::Lossless,
            HardwareBackend::Cpu,
            &cpu_only,
        );
        assert!(matches!(
            jpeg_lossless,
            Err(DomainError::UnsupportedImagePipeline {
                format: ImageOutputFormat::Jpeg,
                backend: HardwareBackend::Cpu,
                ref reason,
            }) if reason.contains("lossless")
        ));

        let vulkan_jpeg_rtx = ImageExecutionMode::resolve(
            ImageOutputFormat::Jpeg,
            lossy,
            HardwareBackend::Gpu(GpuBackend::Vulkan),
            &rtx,
        );
        assert!(matches!(
            vulkan_jpeg_rtx,
            Err(DomainError::UnsupportedImagePipeline {
                format: ImageOutputFormat::Jpeg,
                backend: HardwareBackend::Gpu(GpuBackend::Vulkan),
                ref reason,
            }) if reason.contains("only exposes compute")
        ));

        let nvenc_jpeg = ImageExecutionMode::resolve(
            ImageOutputFormat::Jpeg,
            lossy,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            &rtx,
        );
        assert!(matches!(
            nvenc_jpeg,
            Err(DomainError::UnsupportedImagePipeline {
                format: ImageOutputFormat::Jpeg,
                backend: HardwareBackend::Gpu(GpuBackend::Nvenc),
                ref reason,
            }) if reason.contains("JPEG") && reason.contains("--to heic")
        ));
        Ok(())
    }

    #[test]
    fn constructs_valid_mixed_batch_and_caps_effective_workers() -> Result<(), DomainError> {
        let dims = ImageDimensions::new(3840, 2160)?;
        assert_eq!(dims.to_string(), "3840x2160");

        let items = vec![
            ImageBatchItem::new(
                PathBuf::from("a.png"),
                PathBuf::from("a.webp"),
                ImageInputFormat::Png,
                dims,
            )?,
            ImageBatchItem::new(
                PathBuf::from("b.jpg"),
                PathBuf::from("b.webp"),
                ImageInputFormat::Jpeg,
                dims,
            )?,
        ];
        let requested_workers = NonZeroUsize::new(8).ok_or(DomainError::EmptyBatch)?;
        let plan = ImageBatchPlan::new(
            items,
            ImageExecutionMode::CpuWebp(ImageCompression::Lossy {
                quality: ImageQuality::DEFAULT,
            }),
            requested_workers,
        )?;

        assert_eq!(plan.items().len(), 2);
        assert_eq!(plan.items()[0].input_format(), ImageInputFormat::Png);
        assert_eq!(plan.items()[1].input_format(), ImageInputFormat::Jpeg);
        assert_eq!(plan.worker_threads().get(), 8);
        assert_eq!(plan.effective_workers().get(), 2);
        assert_eq!(plan.output_format(), ImageOutputFormat::Webp);
        assert_eq!(plan.backend(), HardwareBackend::Cpu);
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
            ImageQuality::new(101),
            Err(DomainError::InvalidQuality {
                value: 101,
                max: 100
            })
        );
        assert!(ImageQuality::new(100).is_ok());
    }

    #[test]
    fn rejects_empty_batch_duplicate_targets_and_source_overwrite() -> Result<(), DomainError> {
        let workers = NonZeroUsize::MIN;
        let lossless = ImageExecutionMode::CpuWebp(ImageCompression::Lossless);
        assert_eq!(
            ImageBatchPlan::new(Vec::new(), lossless, workers),
            Err(DomainError::EmptyBatch)
        );

        let dims = ImageDimensions::new(64, 64)?;
        let dup_items = vec![
            ImageBatchItem::new(
                PathBuf::from("photo.png"),
                PathBuf::from("photo.webp"),
                ImageInputFormat::Png,
                dims,
            )?,
            ImageBatchItem::new(
                PathBuf::from("photo.jpg"),
                PathBuf::from("photo.webp"),
                ImageInputFormat::Jpeg,
                dims,
            )?,
        ];
        assert_eq!(
            ImageBatchPlan::new(dup_items, lossless, workers),
            Err(DomainError::DuplicateTargetPath(PathBuf::from(
                "photo.webp"
            )))
        );

        let overwrite_items = vec![
            ImageBatchItem::new(
                PathBuf::from("a.png"),
                PathBuf::from("b.jpg"),
                ImageInputFormat::Png,
                dims,
            )?,
            ImageBatchItem::new(
                PathBuf::from("b.jpg"),
                PathBuf::from("c.webp"),
                ImageInputFormat::Jpeg,
                dims,
            )?,
        ];
        assert_eq!(
            ImageBatchPlan::new(overwrite_items, lossless, workers),
            Err(DomainError::TargetOverwritesSource(PathBuf::from("b.jpg")))
        );

        assert_eq!(
            ImageBatchItem::new(
                PathBuf::new(),
                PathBuf::from("out.webp"),
                ImageInputFormat::Png,
                dims,
            ),
            Err(DomainError::EmptyPath(PathRole::Source))
        );
        Ok(())
    }
}
