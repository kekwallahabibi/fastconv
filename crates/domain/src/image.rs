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
    Webp,
    Bmp,
    Qoi,
    Tiff,
    Jxl,
    Avif,
    Heic,
    Gif,
    Ico,
    Pnm,
}

impl ImageInputFormat {
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Webp => "webp",
            Self::Bmp => "bmp",
            Self::Qoi => "qoi",
            Self::Tiff => "tiff",
            Self::Jxl => "jxl",
            Self::Avif => "avif",
            Self::Heic => "heic",
            Self::Gif => "gif",
            Self::Ico => "ico",
            Self::Pnm => "pnm",
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
    Png,
    Bmp,
    Qoi,
    Tiff,
    Jxl,
    Avif,
    Gif,
    Ico,
    Pnm,
}

impl ImageOutputFormat {
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Webp => "webp",
            Self::Heic => "heic",
            Self::Jpeg => "jpg",
            Self::Png => "png",
            Self::Bmp => "bmp",
            Self::Qoi => "qoi",
            Self::Tiff => "tiff",
            Self::Jxl => "jxl",
            Self::Avif => "avif",
            Self::Gif => "gif",
            Self::Ico => "ico",
            Self::Pnm => "pnm",
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

/// Effort/speed tradeoff for image encoders that expose multiple compression levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum CompressionLevel {
    Fast,
    Balanced,
    Best,
}

impl fmt::Display for CompressionLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::Best => "best",
        };
        f.write_str(label)
    }
}

/// User-selected TIFF compression family, before an effort level is resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum TiffCompressionKind {
    Deflate,
    Lzw,
    Packbits,
    None,
}

impl fmt::Display for TiffCompressionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Deflate => "deflate",
            Self::Lzw => "lzw",
            Self::Packbits => "packbits",
            Self::None => "none",
        };
        f.write_str(label)
    }
}

/// Validated TIFF compression, making `CompressionLevel` + non-Deflate combinations
/// compile-time unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum TiffCompression {
    Deflate { level: CompressionLevel },
    Lzw,
    Packbits,
    None,
}

impl TiffCompression {
    pub const DEFAULT: Self = Self::Deflate {
        level: CompressionLevel::Fast,
    };
}

impl fmt::Display for TiffCompression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Deflate { level } => write!(f, "deflate level={level}"),
            Self::Lzw => f.write_str("lzw"),
            Self::Packbits => f.write_str("packbits"),
            Self::None => f.write_str("none"),
        }
    }
}

/// Chroma subsampling for formats that expose a 4:2:0 / 4:4:4 choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum ChromaSubsampling {
    Yuv420,
    Yuv444,
}

impl fmt::Display for ChromaSubsampling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Yuv420 => "4:2:0",
            Self::Yuv444 => "4:4:4",
        };
        f.write_str(label)
    }
}

/// Optional encoder controls validated once at the probe boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize)]
pub struct ImageEncodeOptions {
    pub compression_level: Option<CompressionLevel>,
    pub tiff_compression: Option<TiffCompressionKind>,
    pub chroma: Option<ChromaSubsampling>,
}

/// Compile-time-legal combination of output image format, codec mode, and hardware backend.
///
/// Illegal pairings (e.g. NVENC + WebP/JPEG/PNG/BMP/QOI/TIFF/JXL/AVIF/GIF/ICO/PNM,
/// Vulkan + images, lossless + JPEG/HEIC/AVIF/GIF, lossy quality +
/// PNG/BMP/QOI/TIFF/ICO/PNM, or hardware-absent video encoder backends) cannot be constructed.
/// HEIC is encodable either as pure-Rust CPU HEVC-Intra ([`Self::CpuHeic`]) or via
/// NVIDIA NVENC ([`Self::NvencHeic`]); the latter never silently falls back to the CPU path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum ImageExecutionMode {
    CpuWebp {
        compression: ImageCompression,
        level: CompressionLevel,
    },
    CpuJpeg {
        quality: ImageQuality,
        chroma: ChromaSubsampling,
    },
    CpuPng {
        level: CompressionLevel,
    },
    CpuBmp,
    CpuQoi,
    CpuTiff {
        compression: TiffCompression,
    },
    CpuJxl {
        compression: ImageCompression,
        level: CompressionLevel,
    },
    CpuAvif {
        quality: ImageQuality,
        level: CompressionLevel,
        chroma: ChromaSubsampling,
    },
    CpuGif,
    CpuIco,
    CpuPnm,
    CpuHeic {
        quality: ImageQuality,
    },
    NvencHeic {
        quality: ImageQuality,
    },
}

impl ImageExecutionMode {
    /// Resolves a CPU-only [`ImageExecutionMode`] for `format` without probing GPU hardware.
    ///
    /// When `compression` is `None`, selects the natural default for `format` (lossy quality 80
    /// for WebP/JPEG/JXL/AVIF/HEIC, lossless for PNG/BMP/QOI/TIFF/ICO/PNM, and indexed 256-color
    /// quantization for GIF). Omitted encoder options keep the pre-existing encoder defaults.
    /// Lossless HEIC is rejected because the CPU encoder only produces lossy HEVC-Intra.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedImagePipeline`] if `format` has no native CPU encoder,
    /// the compression mode is incompatible with `format`, or an encoder option is not supported.
    pub fn resolve_cpu(
        format: ImageOutputFormat,
        compression: Option<ImageCompression>,
    ) -> Result<Self, DomainError> {
        Self::resolve_cpu_with_options(format, compression, ImageEncodeOptions::default())
    }

    /// Resolves a CPU-only [`ImageExecutionMode`] with explicit encoder options.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedImagePipeline`] for incompatible compression modes or
    /// encoder options; see [`validate_encode_options`].
    pub fn resolve_cpu_with_options(
        format: ImageOutputFormat,
        compression: Option<ImageCompression>,
        options: ImageEncodeOptions,
    ) -> Result<Self, DomainError> {
        validate_encode_options(format, HardwareBackend::Cpu, options)?;
        Self::resolve_cpu_validated(format, compression, options)
    }

    fn resolve_cpu_validated(
        format: ImageOutputFormat,
        compression: Option<ImageCompression>,
        options: ImageEncodeOptions,
    ) -> Result<Self, DomainError> {
        match format {
            ImageOutputFormat::Webp => Ok(Self::webp_mode(compression, options)),
            ImageOutputFormat::Jxl => Ok(Self::jxl_mode(compression, options)),
            ImageOutputFormat::Jpeg => Self::jpeg_mode(compression, options),
            ImageOutputFormat::Png => {
                resolve_lossless_only_cpu(format, "PNG", compression, Self::png_mode(options))
            }
            ImageOutputFormat::Bmp => {
                resolve_lossless_only_cpu(format, "BMP", compression, Self::CpuBmp)
            }
            ImageOutputFormat::Qoi => {
                resolve_lossless_only_cpu(format, "QOI", compression, Self::CpuQoi)
            }
            ImageOutputFormat::Tiff => {
                resolve_lossless_only_cpu(format, "TIFF", compression, Self::tiff_mode(options))
            }
            ImageOutputFormat::Avif => Self::avif_mode(compression, options),
            ImageOutputFormat::Ico => {
                resolve_lossless_only_cpu(format, "ICO", compression, Self::CpuIco)
            }
            ImageOutputFormat::Pnm => {
                resolve_lossless_only_cpu(format, "PNM", compression, Self::CpuPnm)
            }
            ImageOutputFormat::Gif => Self::gif_mode(compression),
            ImageOutputFormat::Heic => Self::heic_cpu_mode(compression),
        }
    }

    fn heic_cpu_mode(compression: Option<ImageCompression>) -> Result<Self, DomainError> {
        let quality = match compression {
            None => ImageQuality::DEFAULT,
            Some(ImageCompression::Lossy { quality }) => quality,
            Some(ImageCompression::Lossless) => {
                return Err(DomainError::UnsupportedImagePipeline {
                    format: ImageOutputFormat::Heic,
                    backend: HardwareBackend::Cpu,
                    reason: "CPU HEIC currently supports lossy HEVC-Intra (--quality 0..=100), not --lossless".to_owned(),
                });
            }
        };
        Ok(Self::CpuHeic { quality })
    }

    fn webp_mode(compression: Option<ImageCompression>, options: ImageEncodeOptions) -> Self {
        Self::CpuWebp {
            compression: compression.unwrap_or(default_lossy()),
            level: options
                .compression_level
                .unwrap_or(CompressionLevel::Balanced),
        }
    }

    fn jxl_mode(compression: Option<ImageCompression>, options: ImageEncodeOptions) -> Self {
        Self::CpuJxl {
            compression: compression.unwrap_or(default_lossy()),
            level: options.compression_level.unwrap_or(CompressionLevel::Fast),
        }
    }

    fn jpeg_mode(
        compression: Option<ImageCompression>,
        options: ImageEncodeOptions,
    ) -> Result<Self, DomainError> {
        let quality = match compression {
            None => ImageQuality::DEFAULT,
            Some(ImageCompression::Lossy { quality }) => quality,
            Some(ImageCompression::Lossless) => {
                return Err(DomainError::UnsupportedImagePipeline {
                    format: ImageOutputFormat::Jpeg,
                    backend: HardwareBackend::Cpu,
                    reason: "JPEG is a lossy format and does not support --lossless; use --quality 0..=100".to_owned(),
                });
            }
        };
        Ok(Self::CpuJpeg {
            chroma: options
                .chroma
                .unwrap_or_else(|| default_jpeg_chroma(quality)),
            quality,
        })
    }

    fn png_mode(options: ImageEncodeOptions) -> Self {
        Self::CpuPng {
            level: options.compression_level.unwrap_or(CompressionLevel::Fast),
        }
    }

    fn tiff_mode(options: ImageEncodeOptions) -> Self {
        Self::CpuTiff {
            compression: resolve_tiff_compression(options),
        }
    }

    fn avif_mode(
        compression: Option<ImageCompression>,
        options: ImageEncodeOptions,
    ) -> Result<Self, DomainError> {
        let quality = match compression {
            None => ImageQuality::DEFAULT,
            Some(ImageCompression::Lossy { quality }) => quality,
            Some(ImageCompression::Lossless) => {
                return Err(DomainError::UnsupportedImagePipeline {
                    format: ImageOutputFormat::Avif,
                    backend: HardwareBackend::Cpu,
                    reason:
                        "AVIF is lossy-only and does not support --lossless; use --quality 0..=100"
                            .to_owned(),
                });
            }
        };
        Ok(Self::CpuAvif {
            quality,
            level: options.compression_level.unwrap_or(CompressionLevel::Fast),
            chroma: options.chroma.unwrap_or(ChromaSubsampling::Yuv420),
        })
    }

    fn gif_mode(compression: Option<ImageCompression>) -> Result<Self, DomainError> {
        match compression {
            None => Ok(Self::CpuGif),
            Some(ImageCompression::Lossy { .. }) => Err(DomainError::UnsupportedImagePipeline {
                format: ImageOutputFormat::Gif,
                backend: HardwareBackend::Cpu,
                reason: "GIF uses fixed 256-color palette quantization and does not accept --quality; omit --quality".to_owned(),
            }),
            Some(ImageCompression::Lossless) => Err(DomainError::UnsupportedImagePipeline {
                format: ImageOutputFormat::Gif,
                backend: HardwareBackend::Cpu,
                reason: "GIF is limited to a 256-color palette and cannot guarantee lossless encoding for truecolor images; omit --lossless".to_owned(),
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
    /// format, compression mode, options, and backend is not supported on `hardware`.
    pub fn resolve(
        format: ImageOutputFormat,
        compression: Option<ImageCompression>,
        backend: HardwareBackend,
        hardware: &HostHardware,
    ) -> Result<Self, DomainError> {
        Self::resolve_with_options(
            format,
            compression,
            ImageEncodeOptions::default(),
            backend,
            hardware,
        )
    }

    /// Resolves a validated [`ImageExecutionMode`] with explicit encoder options.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedImagePipeline`] for unsupported options or backends.
    pub fn resolve_with_options(
        format: ImageOutputFormat,
        compression: Option<ImageCompression>,
        options: ImageEncodeOptions,
        backend: HardwareBackend,
        hardware: &HostHardware,
    ) -> Result<Self, DomainError> {
        validate_encode_options(format, backend, options)?;
        match (format, backend) {
            (format, HardwareBackend::Cpu) => {
                Self::resolve_cpu_validated(format, compression, options)
            }
            (ImageOutputFormat::Webp, HardwareBackend::Gpu(GpuBackend::Nvenc)) => Err(
                nvenc_unsupported_image_error(format, "VP8/WebP", ".webp", hardware),
            ),
            (ImageOutputFormat::Jpeg, HardwareBackend::Gpu(GpuBackend::Nvenc)) => Err(
                nvenc_unsupported_image_error(format, "JPEG", ".jpg", hardware),
            ),
            (ImageOutputFormat::Png, HardwareBackend::Gpu(GpuBackend::Nvenc)) => Err(
                nvenc_unsupported_image_error(format, "PNG", ".png", hardware),
            ),
            (ImageOutputFormat::Bmp, HardwareBackend::Gpu(GpuBackend::Nvenc)) => Err(
                nvenc_unsupported_image_error(format, "BMP", ".bmp", hardware),
            ),
            (ImageOutputFormat::Qoi, HardwareBackend::Gpu(GpuBackend::Nvenc)) => Err(
                nvenc_unsupported_image_error(format, "QOI", ".qoi", hardware),
            ),
            (ImageOutputFormat::Tiff, HardwareBackend::Gpu(GpuBackend::Nvenc)) => Err(
                nvenc_unsupported_image_error(format, "TIFF", ".tiff", hardware),
            ),
            (ImageOutputFormat::Jxl, HardwareBackend::Gpu(GpuBackend::Nvenc)) => Err(
                nvenc_unsupported_image_error(format, "JPEG XL", ".jxl", hardware),
            ),
            (ImageOutputFormat::Avif, HardwareBackend::Gpu(GpuBackend::Nvenc)) => Err(
                nvenc_unsupported_image_error(format, "AV1/AVIF still images", ".avif", hardware),
            ),
            (ImageOutputFormat::Gif, HardwareBackend::Gpu(GpuBackend::Nvenc)) => Err(
                nvenc_unsupported_image_error(format, "GIF", ".gif", hardware),
            ),
            (ImageOutputFormat::Ico, HardwareBackend::Gpu(GpuBackend::Nvenc)) => Err(
                nvenc_unsupported_image_error(format, "ICO", ".ico", hardware),
            ),
            (ImageOutputFormat::Pnm, HardwareBackend::Gpu(GpuBackend::Nvenc)) => Err(
                nvenc_unsupported_image_error(format, "PNM", ".pnm", hardware),
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
                    None => Ok(Self::NvencHeic {
                        quality: ImageQuality::DEFAULT,
                    }),
                    Some(ImageCompression::Lossy { quality }) => Ok(Self::NvencHeic { quality }),
                    Some(ImageCompression::Lossless) => Err(DomainError::UnsupportedImagePipeline {
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
            Self::CpuWebp { .. } => ImageOutputFormat::Webp,
            Self::CpuJpeg { .. } => ImageOutputFormat::Jpeg,
            Self::CpuPng { .. } => ImageOutputFormat::Png,
            Self::CpuBmp => ImageOutputFormat::Bmp,
            Self::CpuQoi => ImageOutputFormat::Qoi,
            Self::CpuTiff { .. } => ImageOutputFormat::Tiff,
            Self::CpuJxl { .. } => ImageOutputFormat::Jxl,
            Self::CpuAvif { .. } => ImageOutputFormat::Avif,
            Self::CpuGif => ImageOutputFormat::Gif,
            Self::CpuIco => ImageOutputFormat::Ico,
            Self::CpuPnm => ImageOutputFormat::Pnm,
            Self::CpuHeic { .. } | Self::NvencHeic { .. } => ImageOutputFormat::Heic,
        }
    }

    #[must_use]
    pub const fn backend(self) -> HardwareBackend {
        match self {
            Self::CpuWebp { .. }
            | Self::CpuJpeg { .. }
            | Self::CpuPng { .. }
            | Self::CpuBmp
            | Self::CpuQoi
            | Self::CpuTiff { .. }
            | Self::CpuJxl { .. }
            | Self::CpuAvif { .. }
            | Self::CpuGif
            | Self::CpuIco
            | Self::CpuPnm
            | Self::CpuHeic { .. } => HardwareBackend::Cpu,
            Self::NvencHeic { .. } => HardwareBackend::Gpu(GpuBackend::Nvenc),
        }
    }
}

impl fmt::Display for ImageExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CpuWebp { compression, level } => write!(f, "webp {compression} level={level}"),
            Self::CpuJpeg { quality, chroma } => {
                write!(f, "jpeg q={} {chroma}", quality.get())
            }
            Self::CpuPng { level } => write!(f, "png lossless level={level}"),
            Self::CpuBmp => f.write_str("bmp lossless"),
            Self::CpuQoi => f.write_str("qoi lossless"),
            Self::CpuTiff { compression } => write!(f, "tiff lossless {compression}"),
            Self::CpuJxl { compression, level } => write!(f, "jxl {compression} level={level}"),
            Self::CpuAvif {
                quality,
                level,
                chroma,
            } => write!(f, "avif av1 q={} level={level} {chroma}", quality.get()),
            Self::CpuGif => f.write_str("gif indexed"),
            Self::CpuIco => f.write_str("ico lossless"),
            Self::CpuPnm => f.write_str("pnm lossless"),
            Self::CpuHeic { quality } | Self::NvencHeic { quality } => {
                write!(f, "heic hevc-intra q={}", quality.get())
            }
        }
    }
}

const fn default_lossy() -> ImageCompression {
    ImageCompression::Lossy {
        quality: ImageQuality::DEFAULT,
    }
}

/// `jpeg-encoder 0.7.1` downsamples chroma below quality 90 and uses 4:4:4 above it.
const fn default_jpeg_chroma(quality: ImageQuality) -> ChromaSubsampling {
    if quality.get() < 90 {
        ChromaSubsampling::Yuv420
    } else {
        ChromaSubsampling::Yuv444
    }
}

fn resolve_tiff_compression(options: ImageEncodeOptions) -> TiffCompression {
    match options
        .tiff_compression
        .unwrap_or(TiffCompressionKind::Deflate)
    {
        TiffCompressionKind::Deflate => TiffCompression::Deflate {
            level: options.compression_level.unwrap_or(CompressionLevel::Fast),
        },
        TiffCompressionKind::Lzw => TiffCompression::Lzw,
        TiffCompressionKind::Packbits => TiffCompression::Packbits,
        TiffCompressionKind::None => TiffCompression::None,
    }
}

/// Validates optional encoder controls exactly once, before any mode is constructed.
///
/// GPU execution never exposes these CPU encoder knobs; TIFF compression is exclusive to
/// Deflate-capable CPU TIFF, effort only applies to encoders with multiple speed presets, and
/// chroma subsampling only applies to JPEG and AVIF.
fn validate_encode_options(
    format: ImageOutputFormat,
    backend: HardwareBackend,
    options: ImageEncodeOptions,
) -> Result<(), DomainError> {
    let cpu = HardwareBackend::Cpu;
    if options.tiff_compression.is_some() && (backend != cpu || format != ImageOutputFormat::Tiff) {
        return Err(DomainError::UnsupportedImagePipeline {
            format,
            backend,
            reason: "--tiff-compression is only supported for .tiff on the cpu backend".to_owned(),
        });
    }
    if options.compression_level.is_some() {
        if backend != cpu {
            return Err(DomainError::UnsupportedImagePipeline {
                format,
                backend,
                reason: "--compression-level is a cpu encoder option and is not supported on gpu backends".to_owned(),
            });
        }
        if format == ImageOutputFormat::Tiff
            && matches!(
                options.tiff_compression,
                Some(
                    TiffCompressionKind::Lzw
                        | TiffCompressionKind::Packbits
                        | TiffCompressionKind::None
                )
            )
        {
            return Err(DomainError::UnsupportedImagePipeline {
                format,
                backend,
                reason: "TIFF compression does not accept --compression-level; use --tiff-compression deflate or omit --compression-level".to_owned(),
            });
        }
        if !matches!(
            format,
            ImageOutputFormat::Png
                | ImageOutputFormat::Webp
                | ImageOutputFormat::Jxl
                | ImageOutputFormat::Avif
                | ImageOutputFormat::Tiff
        ) {
            return Err(DomainError::UnsupportedImagePipeline {
                format,
                backend,
                reason: "--compression-level (fast|balanced|best) is only supported for png, webp, jxl, avif, and deflate-compressed tiff".to_owned(),
            });
        }
    }
    if options.chroma.is_some()
        && (backend != cpu || !matches!(format, ImageOutputFormat::Jpeg | ImageOutputFormat::Avif))
    {
        return Err(DomainError::UnsupportedImagePipeline {
            format,
            backend,
            reason: "--chroma (420|444) is only supported for jpeg and avif on the cpu backend"
                .to_owned(),
        });
    }
    Ok(())
}

fn resolve_lossless_only_cpu(
    format: ImageOutputFormat,
    label: &str,
    compression: Option<ImageCompression>,
    mode: ImageExecutionMode,
) -> Result<ImageExecutionMode, DomainError> {
    match compression {
        None | Some(ImageCompression::Lossless) => Ok(mode),
        Some(ImageCompression::Lossy { .. }) => Err(DomainError::UnsupportedImagePipeline {
            format,
            backend: HardwareBackend::Cpu,
            reason: format!(
                "{label} is a lossless format and does not accept --quality; omit --quality or pass --lossless"
            ),
        }),
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
            "NVENC provides hardware video encoders (H.264/HEVC/AV1 on supported GPUs), not {codec_label}; use --backend cpu for {cpu_ext}, or target .heic (--to heic) for NVENC HEVC-Intra"
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
        ChromaSubsampling, CompressionLevel, ImageBatchItem, ImageBatchPlan, ImageCompression,
        ImageDimensions, ImageEncodeOptions, ImageExecutionMode, ImageInputFormat,
        ImageOutputFormat, ImageQuality, TiffCompression, TiffCompressionKind,
    };
    use crate::{
        CpuInfo, DetectedGpuBackend, DomainError, GpuAdapterInfo, GpuBackend, GpuCapability,
        GpuVendor, HardwareBackend, HostHardware, PathRole, SimdLevel,
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
        let adapter = GpuAdapterInfo::new(
            "NVIDIA GeForce RTX 2060",
            GpuVendor::Nvidia,
            0x1e89,
            "nvidia",
            "590.48.01",
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
                "vulkan (nvidia)",
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
    fn resolves_valid_cpu_modes_including_defaults() -> Result<(), DomainError> {
        let cpu_only = cpu_only_hardware()?;
        let lossy = Some(ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        });

        assert_eq!(
            ImageExecutionMode::resolve(
                ImageOutputFormat::Webp,
                lossy,
                HardwareBackend::Cpu,
                &cpu_only
            )?,
            ImageExecutionMode::CpuWebp {
                compression: ImageCompression::Lossy {
                    quality: ImageQuality::DEFAULT,
                },
                level: CompressionLevel::Balanced,
            }
        );
        assert_eq!(
            ImageExecutionMode::resolve(
                ImageOutputFormat::Webp,
                None,
                HardwareBackend::Cpu,
                &cpu_only
            )?,
            ImageExecutionMode::CpuWebp {
                compression: ImageCompression::Lossy {
                    quality: ImageQuality::DEFAULT,
                },
                level: CompressionLevel::Balanced,
            }
        );
        assert_eq!(
            ImageExecutionMode::resolve(
                ImageOutputFormat::Jpeg,
                None,
                HardwareBackend::Cpu,
                &cpu_only
            )?,
            ImageExecutionMode::CpuJpeg {
                quality: ImageQuality::DEFAULT,
                chroma: ChromaSubsampling::Yuv420,
            }
        );
        assert_eq!(
            ImageExecutionMode::resolve(
                ImageOutputFormat::Png,
                None,
                HardwareBackend::Cpu,
                &cpu_only
            )?,
            ImageExecutionMode::CpuPng {
                level: CompressionLevel::Fast,
            }
        );
        assert_eq!(
            ImageExecutionMode::resolve(
                ImageOutputFormat::Png,
                Some(ImageCompression::Lossless),
                HardwareBackend::Cpu,
                &cpu_only
            )?,
            ImageExecutionMode::CpuPng {
                level: CompressionLevel::Fast,
            }
        );
        assert_eq!(
            ImageExecutionMode::resolve(
                ImageOutputFormat::Bmp,
                None,
                HardwareBackend::Cpu,
                &cpu_only
            )?,
            ImageExecutionMode::CpuBmp
        );
        assert_eq!(
            ImageExecutionMode::resolve(
                ImageOutputFormat::Qoi,
                None,
                HardwareBackend::Cpu,
                &cpu_only
            )?,
            ImageExecutionMode::CpuQoi
        );
        Ok(())
    }

    #[test]
    fn resolves_nvenc_heic_defaults() -> Result<(), DomainError> {
        let rtx = rtx_2060_hardware()?;
        let lossy = Some(ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        });
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
        assert_eq!(
            ImageExecutionMode::resolve(
                ImageOutputFormat::Heic,
                None,
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
    fn resolves_cpu_heic_lossy_and_rejects_lossless() -> Result<(), DomainError> {
        let cpu_only = cpu_only_hardware()?;
        let quality = ImageQuality::new(90)?;

        assert_eq!(
            ImageExecutionMode::resolve(
                ImageOutputFormat::Heic,
                None,
                HardwareBackend::Cpu,
                &cpu_only
            )?,
            ImageExecutionMode::CpuHeic {
                quality: ImageQuality::DEFAULT,
            }
        );
        assert_eq!(
            ImageExecutionMode::resolve(
                ImageOutputFormat::Heic,
                Some(ImageCompression::Lossy { quality }),
                HardwareBackend::Cpu,
                &cpu_only
            )?,
            ImageExecutionMode::CpuHeic { quality }
        );
        assert_eq!(
            ImageExecutionMode::resolve_cpu(ImageOutputFormat::Heic, None)?,
            ImageExecutionMode::CpuHeic {
                quality: ImageQuality::DEFAULT,
            }
        );

        let mode = ImageExecutionMode::CpuHeic { quality };
        assert_eq!(mode.output_format(), ImageOutputFormat::Heic);
        assert_eq!(mode.backend(), HardwareBackend::Cpu);
        assert_eq!(
            mode.to_string(),
            format!("heic hevc-intra q={}", quality.get())
        );

        let lossless = ImageExecutionMode::resolve(
            ImageOutputFormat::Heic,
            Some(ImageCompression::Lossless),
            HardwareBackend::Cpu,
            &cpu_only,
        );
        assert!(matches!(
            lossless,
            Err(DomainError::UnsupportedImagePipeline {
                format: ImageOutputFormat::Heic,
                backend: HardwareBackend::Cpu,
                ref reason,
            }) if reason.contains("lossy HEVC-Intra") && reason.contains("--lossless")
        ));
        Ok(())
    }

    #[test]
    fn nvenc_heic_never_silently_falls_back_to_cpu() -> Result<(), DomainError> {
        let rtx = rtx_2060_hardware()?;
        let cpu_only = cpu_only_hardware()?;
        let lossy = Some(ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        });

        assert_eq!(
            ImageExecutionMode::resolve(
                ImageOutputFormat::Heic,
                lossy,
                HardwareBackend::Gpu(GpuBackend::Nvenc),
                &rtx,
            )?,
            ImageExecutionMode::NvencHeic {
                quality: ImageQuality::DEFAULT
            }
        );

        let absent = ImageExecutionMode::resolve(
            ImageOutputFormat::Heic,
            lossy,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            &cpu_only,
        );
        assert!(matches!(
            absent,
            Err(DomainError::UnsupportedImagePipeline {
                format: ImageOutputFormat::Heic,
                backend: HardwareBackend::Gpu(GpuBackend::Nvenc),
                ref reason,
            }) if reason.contains("no NVENC hardware video encoder")
        ));
        Ok(())
    }

    #[test]
    fn rejects_absent_decode_only_and_codec_mismatched_backends() -> Result<(), DomainError> {
        let rtx = rtx_2060_hardware()?;
        let cpu_only = cpu_only_hardware()?;
        let lossy = Some(ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        });

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
    fn rejects_unsupported_jpeg_and_lossless_format_modes_and_backends() -> Result<(), DomainError>
    {
        let rtx = rtx_2060_hardware()?;
        let cpu_only = cpu_only_hardware()?;
        let lossy = Some(ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        });

        let jpeg_lossless = ImageExecutionMode::resolve(
            ImageOutputFormat::Jpeg,
            Some(ImageCompression::Lossless),
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

        for fmt in [
            ImageOutputFormat::Png,
            ImageOutputFormat::Bmp,
            ImageOutputFormat::Qoi,
            ImageOutputFormat::Tiff,
        ] {
            let quality_rejected =
                ImageExecutionMode::resolve(fmt, lossy, HardwareBackend::Cpu, &cpu_only);
            assert!(matches!(
                quality_rejected,
                Err(DomainError::UnsupportedImagePipeline {
                    format,
                    backend: HardwareBackend::Cpu,
                    ref reason,
                }) if format == fmt && reason.contains("--quality")
            ));

            let vulkan_rejected = ImageExecutionMode::resolve(
                fmt,
                None,
                HardwareBackend::Gpu(GpuBackend::Vulkan),
                &rtx,
            );
            assert!(matches!(
                vulkan_rejected,
                Err(DomainError::UnsupportedImagePipeline {
                    format,
                    backend: HardwareBackend::Gpu(GpuBackend::Vulkan),
                    ref reason,
                }) if format == fmt && reason.contains("only exposes compute")
            ));

            let nvenc_rejected = ImageExecutionMode::resolve(
                fmt,
                None,
                HardwareBackend::Gpu(GpuBackend::Nvenc),
                &rtx,
            );
            assert!(matches!(
                nvenc_rejected,
                Err(DomainError::UnsupportedImagePipeline {
                    format,
                    backend: HardwareBackend::Gpu(GpuBackend::Nvenc),
                    ref reason,
                }) if format == fmt && reason.contains("--to heic")
            ));
        }
        Ok(())
    }

    #[test]
    fn resolves_jxl_avif_and_tiff_constraints() -> Result<(), DomainError> {
        let rtx = rtx_2060_hardware()?;
        let cpu_only = cpu_only_hardware()?;
        let lossy = Some(ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        });

        assert_eq!(
            ImageExecutionMode::resolve(
                ImageOutputFormat::Jxl,
                lossy,
                HardwareBackend::Cpu,
                &cpu_only
            )?,
            ImageExecutionMode::CpuJxl {
                compression: ImageCompression::Lossy {
                    quality: ImageQuality::DEFAULT,
                },
                level: CompressionLevel::Fast,
            }
        );
        assert_eq!(
            ImageExecutionMode::CpuAvif {
                quality: ImageQuality::DEFAULT,
                level: CompressionLevel::Fast,
                chroma: ChromaSubsampling::Yuv420,
            }
            .output_format(),
            ImageOutputFormat::Avif
        );
        assert_eq!(
            ImageExecutionMode::CpuJxl {
                compression: ImageCompression::Lossless,
                level: CompressionLevel::Fast,
            }
            .output_format(),
            ImageOutputFormat::Jxl
        );
        assert_eq!(
            ImageExecutionMode::CpuTiff {
                compression: TiffCompression::DEFAULT,
            }
            .output_format(),
            ImageOutputFormat::Tiff
        );

        let avif_lossless = ImageExecutionMode::resolve(
            ImageOutputFormat::Avif,
            Some(ImageCompression::Lossless),
            HardwareBackend::Cpu,
            &cpu_only,
        );
        assert!(matches!(
            avif_lossless,
            Err(DomainError::UnsupportedImagePipeline {
                format: ImageOutputFormat::Avif,
                backend: HardwareBackend::Cpu,
                ref reason,
            }) if reason.contains("lossy-only")
        ));

        for (fmt, needle) in [
            (ImageOutputFormat::Tiff, "TIFF"),
            (ImageOutputFormat::Jxl, "JPEG XL"),
            (ImageOutputFormat::Avif, "AV1/AVIF"),
        ] {
            let nvenc_rejected = ImageExecutionMode::resolve(
                fmt,
                Some(ImageCompression::Lossy {
                    quality: ImageQuality::DEFAULT,
                }),
                HardwareBackend::Gpu(GpuBackend::Nvenc),
                &rtx,
            );
            assert!(matches!(
                nvenc_rejected,
                Err(DomainError::UnsupportedImagePipeline {
                    format,
                    backend: HardwareBackend::Gpu(GpuBackend::Nvenc),
                    ref reason,
                }) if format == fmt && reason.contains(needle) && reason.contains("cpu")
            ));
        }

        Ok(())
    }

    #[test]
    fn constructs_valid_mixed_batch_and_caps_effective_workers() -> Result<(), DomainError> {
        let dims = ImageDimensions::new(3840, 2160)?;
        assert_eq!(dims.to_string(), "3840x2160");

        assert!(matches!(
            ImageBatchItem::new(
                PathBuf::from("c.webp"),
                PathBuf::from("c.webp"),
                ImageInputFormat::Webp,
                dims,
            ),
            Err(DomainError::IdenticalSourceAndTarget(_))
        ));

        let valid_items = vec![
            ImageBatchItem::new(
                PathBuf::from("a.png"),
                PathBuf::from("out/a.webp"),
                ImageInputFormat::Png,
                dims,
            )?,
            ImageBatchItem::new(
                PathBuf::from("b.jpg"),
                PathBuf::from("out/b.webp"),
                ImageInputFormat::Jpeg,
                dims,
            )?,
            ImageBatchItem::new(
                PathBuf::from("c.webp"),
                PathBuf::from("out/c.webp"),
                ImageInputFormat::Webp,
                dims,
            )?,
            ImageBatchItem::new(
                PathBuf::from("d.bmp"),
                PathBuf::from("out/d.webp"),
                ImageInputFormat::Bmp,
                dims,
            )?,
            ImageBatchItem::new(
                PathBuf::from("e.qoi"),
                PathBuf::from("out/e.webp"),
                ImageInputFormat::Qoi,
                dims,
            )?,
        ];
        let requested_workers = NonZeroUsize::new(8).ok_or(DomainError::EmptyBatch)?;
        let plan = ImageBatchPlan::new(
            valid_items,
            ImageExecutionMode::CpuWebp {
                compression: ImageCompression::Lossy {
                    quality: ImageQuality::DEFAULT,
                },
                level: CompressionLevel::Balanced,
            },
            requested_workers,
        )?;

        assert_eq!(plan.items().len(), 5);
        assert_eq!(plan.items()[0].input_format(), ImageInputFormat::Png);
        assert_eq!(plan.items()[1].input_format(), ImageInputFormat::Jpeg);
        assert_eq!(plan.items()[2].input_format(), ImageInputFormat::Webp);
        assert_eq!(plan.items()[3].input_format(), ImageInputFormat::Bmp);
        assert_eq!(plan.items()[4].input_format(), ImageInputFormat::Qoi);
        assert_eq!(plan.worker_threads().get(), 8);
        assert_eq!(plan.effective_workers().get(), 5);
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
        let lossless = ImageExecutionMode::CpuWebp {
            compression: ImageCompression::Lossless,
            level: CompressionLevel::Balanced,
        };
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

    #[test]
    fn new_image_formats_report_extensions_and_display() {
        for (format, ext) in [
            (ImageInputFormat::Avif, "avif"),
            (ImageInputFormat::Heic, "heic"),
            (ImageInputFormat::Gif, "gif"),
            (ImageInputFormat::Ico, "ico"),
            (ImageInputFormat::Pnm, "pnm"),
        ] {
            assert_eq!(format.extension(), ext);
            assert_eq!(format.to_string(), ext);
        }

        for (format, ext) in [
            (ImageOutputFormat::Gif, "gif"),
            (ImageOutputFormat::Ico, "ico"),
            (ImageOutputFormat::Pnm, "pnm"),
        ] {
            assert_eq!(format.extension(), ext);
            assert_eq!(format.to_string(), ext);
        }
    }

    #[test]
    fn heic_input_format_uses_heic_extension_and_display() {
        assert_eq!(ImageInputFormat::Heic.extension(), "heic");
        assert_eq!(ImageInputFormat::Heic.to_string(), "heic");
        // HEIF files are detected by their ISOBMFF brand, so both file extensions
        // resolve to the same validated input format.
        assert_ne!(ImageInputFormat::Heic, ImageInputFormat::Avif);
    }

    #[test]
    fn resolves_gif_indexed_mode_and_rejects_incompatible_requests() -> Result<(), DomainError> {
        let rtx = rtx_2060_hardware()?;
        let cpu_only = cpu_only_hardware()?;
        let lossy = Some(ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        });

        assert_eq!(
            ImageExecutionMode::resolve(
                ImageOutputFormat::Gif,
                None,
                HardwareBackend::Cpu,
                &cpu_only
            )?,
            ImageExecutionMode::CpuGif
        );
        assert_eq!(
            ImageExecutionMode::CpuGif.output_format(),
            ImageOutputFormat::Gif
        );
        assert_eq!(ImageExecutionMode::CpuGif.backend(), HardwareBackend::Cpu);
        assert_eq!(ImageExecutionMode::CpuGif.to_string(), "gif indexed");

        for (compression, needle) in [
            (lossy, "--quality"),
            (Some(ImageCompression::Lossless), "--lossless"),
        ] {
            let rejected = ImageExecutionMode::resolve(
                ImageOutputFormat::Gif,
                compression,
                HardwareBackend::Cpu,
                &cpu_only,
            );
            assert!(matches!(
                rejected,
                Err(DomainError::UnsupportedImagePipeline {
                    format: ImageOutputFormat::Gif,
                    backend: HardwareBackend::Cpu,
                    ref reason,
                }) if reason.contains(needle)
            ));
        }

        let nvenc = ImageExecutionMode::resolve(
            ImageOutputFormat::Gif,
            None,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            &rtx,
        );
        assert!(matches!(
            nvenc,
            Err(DomainError::UnsupportedImagePipeline {
                format: ImageOutputFormat::Gif,
                backend: HardwareBackend::Gpu(GpuBackend::Nvenc),
                ref reason,
            }) if reason.contains("GIF") && reason.contains(".gif")
        ));

        let vulkan = ImageExecutionMode::resolve(
            ImageOutputFormat::Gif,
            None,
            HardwareBackend::Gpu(GpuBackend::Vulkan),
            &rtx,
        );
        assert!(matches!(
            vulkan,
            Err(DomainError::UnsupportedImagePipeline {
                format: ImageOutputFormat::Gif,
                backend: HardwareBackend::Gpu(GpuBackend::Vulkan),
                ref reason,
            }) if reason.contains("only exposes compute")
        ));
        Ok(())
    }

    #[test]
    fn resolves_ico_and_pnm_lossless_modes_and_rejects_incompatible_requests()
    -> Result<(), DomainError> {
        let rtx = rtx_2060_hardware()?;
        let cpu_only = cpu_only_hardware()?;
        let lossy = Some(ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        });

        for (format, mode, label, ext, display) in [
            (
                ImageOutputFormat::Ico,
                ImageExecutionMode::CpuIco,
                "ICO",
                ".ico",
                "ico lossless",
            ),
            (
                ImageOutputFormat::Pnm,
                ImageExecutionMode::CpuPnm,
                "PNM",
                ".pnm",
                "pnm lossless",
            ),
        ] {
            assert_eq!(
                ImageExecutionMode::resolve(format, None, HardwareBackend::Cpu, &cpu_only)?,
                mode
            );
            assert_eq!(
                ImageExecutionMode::resolve(
                    format,
                    Some(ImageCompression::Lossless),
                    HardwareBackend::Cpu,
                    &cpu_only
                )?,
                mode
            );
            assert_eq!(mode.output_format(), format);
            assert_eq!(mode.backend(), HardwareBackend::Cpu);
            assert_eq!(mode.to_string(), display);

            let quality_rejected =
                ImageExecutionMode::resolve(format, lossy, HardwareBackend::Cpu, &cpu_only);
            assert!(matches!(
                quality_rejected,
                Err(DomainError::UnsupportedImagePipeline {
                    format: rejected_format,
                    backend: HardwareBackend::Cpu,
                    ref reason,
                }) if rejected_format == format && reason.contains("--quality")
            ));

            let nvenc = ImageExecutionMode::resolve(
                format,
                None,
                HardwareBackend::Gpu(GpuBackend::Nvenc),
                &rtx,
            );
            assert!(matches!(
                nvenc,
                Err(DomainError::UnsupportedImagePipeline {
                    format: rejected_format,
                    backend: HardwareBackend::Gpu(GpuBackend::Nvenc),
                    ref reason,
                }) if rejected_format == format && reason.contains(label) && reason.contains(ext)
            ));

            let vulkan = ImageExecutionMode::resolve(
                format,
                None,
                HardwareBackend::Gpu(GpuBackend::Vulkan),
                &rtx,
            );
            assert!(matches!(
                vulkan,
                Err(DomainError::UnsupportedImagePipeline {
                    format: rejected_format,
                    backend: HardwareBackend::Gpu(GpuBackend::Vulkan),
                    ref reason,
                }) if rejected_format == format && reason.contains("only exposes compute")
            ));
        }
        Ok(())
    }

    #[test]
    fn constructs_batch_with_new_image_input_formats() -> Result<(), DomainError> {
        let dims = ImageDimensions::new(640, 480)?;
        let valid_items = vec![
            ImageBatchItem::new(
                PathBuf::from("a.avif"),
                PathBuf::from("out/a.gif"),
                ImageInputFormat::Avif,
                dims,
            )?,
            ImageBatchItem::new(
                PathBuf::from("b.gif"),
                PathBuf::from("out/b.gif"),
                ImageInputFormat::Gif,
                dims,
            )?,
            ImageBatchItem::new(
                PathBuf::from("c.ico"),
                PathBuf::from("out/c.gif"),
                ImageInputFormat::Ico,
                dims,
            )?,
            ImageBatchItem::new(
                PathBuf::from("d.pnm"),
                PathBuf::from("out/d.gif"),
                ImageInputFormat::Pnm,
                dims,
            )?,
        ];
        let plan = ImageBatchPlan::new(
            valid_items,
            ImageExecutionMode::CpuGif,
            NonZeroUsize::new(8).ok_or(DomainError::EmptyBatch)?,
        )?;

        assert_eq!(plan.items().len(), 4);
        assert_eq!(plan.items()[0].input_format(), ImageInputFormat::Avif);
        assert_eq!(plan.items()[1].input_format(), ImageInputFormat::Gif);
        assert_eq!(plan.items()[2].input_format(), ImageInputFormat::Ico);
        assert_eq!(plan.items()[3].input_format(), ImageInputFormat::Pnm);
        assert_eq!(plan.output_format(), ImageOutputFormat::Gif);
        assert_eq!(plan.backend(), HardwareBackend::Cpu);
        Ok(())
    }

    #[test]
    fn new_encoder_types_display_exact_labels() {
        assert_eq!(CompressionLevel::Fast.to_string(), "fast");
        assert_eq!(CompressionLevel::Balanced.to_string(), "balanced");
        assert_eq!(CompressionLevel::Best.to_string(), "best");

        assert_eq!(TiffCompressionKind::Deflate.to_string(), "deflate");
        assert_eq!(TiffCompressionKind::Lzw.to_string(), "lzw");
        assert_eq!(TiffCompressionKind::Packbits.to_string(), "packbits");
        assert_eq!(TiffCompressionKind::None.to_string(), "none");

        assert_eq!(
            TiffCompression::DEFAULT,
            TiffCompression::Deflate {
                level: CompressionLevel::Fast,
            }
        );
        assert_eq!(TiffCompression::DEFAULT.to_string(), "deflate level=fast");
        assert_eq!(
            TiffCompression::Deflate {
                level: CompressionLevel::Best,
            }
            .to_string(),
            "deflate level=best"
        );
        assert_eq!(TiffCompression::Lzw.to_string(), "lzw");
        assert_eq!(TiffCompression::Packbits.to_string(), "packbits");
        assert_eq!(TiffCompression::None.to_string(), "none");

        assert_eq!(ChromaSubsampling::Yuv420.to_string(), "4:2:0");
        assert_eq!(ChromaSubsampling::Yuv444.to_string(), "4:4:4");
        assert_eq!(
            ImageEncodeOptions::default(),
            ImageEncodeOptions {
                compression_level: None,
                tiff_compression: None,
                chroma: None,
            }
        );
    }

    #[test]
    fn every_format_resolves_the_documented_encoder_defaults() -> Result<(), DomainError> {
        for (format, expected) in [
            (
                ImageOutputFormat::Webp,
                ImageExecutionMode::CpuWebp {
                    compression: ImageCompression::Lossy {
                        quality: ImageQuality::DEFAULT,
                    },
                    level: CompressionLevel::Balanced,
                },
            ),
            (
                ImageOutputFormat::Jpeg,
                ImageExecutionMode::CpuJpeg {
                    quality: ImageQuality::DEFAULT,
                    chroma: ChromaSubsampling::Yuv420,
                },
            ),
            (
                ImageOutputFormat::Png,
                ImageExecutionMode::CpuPng {
                    level: CompressionLevel::Fast,
                },
            ),
            (ImageOutputFormat::Bmp, ImageExecutionMode::CpuBmp),
            (ImageOutputFormat::Qoi, ImageExecutionMode::CpuQoi),
            (
                ImageOutputFormat::Tiff,
                ImageExecutionMode::CpuTiff {
                    compression: TiffCompression::DEFAULT,
                },
            ),
            (
                ImageOutputFormat::Jxl,
                ImageExecutionMode::CpuJxl {
                    compression: ImageCompression::Lossy {
                        quality: ImageQuality::DEFAULT,
                    },
                    level: CompressionLevel::Fast,
                },
            ),
            (
                ImageOutputFormat::Avif,
                ImageExecutionMode::CpuAvif {
                    quality: ImageQuality::DEFAULT,
                    level: CompressionLevel::Fast,
                    chroma: ChromaSubsampling::Yuv420,
                },
            ),
            (ImageOutputFormat::Gif, ImageExecutionMode::CpuGif),
            (ImageOutputFormat::Ico, ImageExecutionMode::CpuIco),
            (ImageOutputFormat::Pnm, ImageExecutionMode::CpuPnm),
            (
                ImageOutputFormat::Heic,
                ImageExecutionMode::CpuHeic {
                    quality: ImageQuality::DEFAULT,
                },
            ),
        ] {
            assert_eq!(
                ImageExecutionMode::resolve_cpu(format, None)?,
                expected,
                "default mode for .{format}"
            );
        }

        // jpeg-encoder 0.7.1 switches from 4:2:0 to 4:4:4 at quality 90.
        assert_eq!(
            ImageExecutionMode::resolve_cpu(
                ImageOutputFormat::Jpeg,
                Some(ImageCompression::Lossy {
                    quality: ImageQuality::new(89)?,
                }),
            )?,
            ImageExecutionMode::CpuJpeg {
                quality: ImageQuality::new(89)?,
                chroma: ChromaSubsampling::Yuv420,
            }
        );
        assert_eq!(
            ImageExecutionMode::resolve_cpu(
                ImageOutputFormat::Jpeg,
                Some(ImageCompression::Lossy {
                    quality: ImageQuality::new(90)?,
                }),
            )?,
            ImageExecutionMode::CpuJpeg {
                quality: ImageQuality::new(90)?,
                chroma: ChromaSubsampling::Yuv444,
            }
        );
        Ok(())
    }

    #[test]
    fn resolves_all_valid_compression_level_combinations() -> Result<(), DomainError> {
        let lossy = Some(ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        });
        let best = |kind| ImageEncodeOptions {
            compression_level: Some(kind),
            ..ImageEncodeOptions::default()
        };

        assert_eq!(
            ImageExecutionMode::resolve_cpu_with_options(
                ImageOutputFormat::Png,
                None,
                best(CompressionLevel::Best),
            )?,
            ImageExecutionMode::CpuPng {
                level: CompressionLevel::Best,
            }
        );
        assert_eq!(
            ImageExecutionMode::resolve_cpu_with_options(
                ImageOutputFormat::Webp,
                lossy,
                best(CompressionLevel::Fast),
            )?,
            ImageExecutionMode::CpuWebp {
                compression: ImageCompression::Lossy {
                    quality: ImageQuality::DEFAULT,
                },
                level: CompressionLevel::Fast,
            }
        );
        assert_eq!(
            ImageExecutionMode::resolve_cpu_with_options(
                ImageOutputFormat::Webp,
                Some(ImageCompression::Lossless),
                best(CompressionLevel::Best),
            )?,
            ImageExecutionMode::CpuWebp {
                compression: ImageCompression::Lossless,
                level: CompressionLevel::Best,
            }
        );
        assert_eq!(
            ImageExecutionMode::resolve_cpu_with_options(
                ImageOutputFormat::Jxl,
                Some(ImageCompression::Lossless),
                best(CompressionLevel::Balanced),
            )?,
            ImageExecutionMode::CpuJxl {
                compression: ImageCompression::Lossless,
                level: CompressionLevel::Balanced,
            }
        );
        assert_eq!(
            ImageExecutionMode::resolve_cpu_with_options(
                ImageOutputFormat::Jxl,
                lossy,
                best(CompressionLevel::Best),
            )?,
            ImageExecutionMode::CpuJxl {
                compression: ImageCompression::Lossy {
                    quality: ImageQuality::DEFAULT,
                },
                level: CompressionLevel::Best,
            }
        );
        assert_eq!(
            ImageExecutionMode::resolve_cpu_with_options(
                ImageOutputFormat::Avif,
                lossy,
                ImageEncodeOptions {
                    compression_level: Some(CompressionLevel::Best),
                    chroma: Some(ChromaSubsampling::Yuv444),
                    ..ImageEncodeOptions::default()
                },
            )?,
            ImageExecutionMode::CpuAvif {
                quality: ImageQuality::DEFAULT,
                level: CompressionLevel::Best,
                chroma: ChromaSubsampling::Yuv444,
            }
        );
        assert_eq!(
            ImageExecutionMode::resolve_cpu_with_options(
                ImageOutputFormat::Jpeg,
                lossy,
                ImageEncodeOptions {
                    chroma: Some(ChromaSubsampling::Yuv444),
                    ..ImageEncodeOptions::default()
                },
            )?,
            ImageExecutionMode::CpuJpeg {
                quality: ImageQuality::DEFAULT,
                chroma: ChromaSubsampling::Yuv444,
            }
        );
        Ok(())
    }

    #[test]
    fn resolves_all_valid_tiff_compression_combinations() -> Result<(), DomainError> {
        for (kind, expected) in [
            (TiffCompressionKind::Deflate, TiffCompression::DEFAULT),
            (TiffCompressionKind::Lzw, TiffCompression::Lzw),
            (TiffCompressionKind::Packbits, TiffCompression::Packbits),
            (TiffCompressionKind::None, TiffCompression::None),
        ] {
            assert_eq!(
                ImageExecutionMode::resolve_cpu_with_options(
                    ImageOutputFormat::Tiff,
                    None,
                    ImageEncodeOptions {
                        tiff_compression: Some(kind),
                        ..ImageEncodeOptions::default()
                    },
                )?,
                ImageExecutionMode::CpuTiff {
                    compression: expected,
                },
                "TIFF compression {kind}"
            );
        }
        assert_eq!(
            ImageExecutionMode::resolve_cpu_with_options(
                ImageOutputFormat::Tiff,
                None,
                ImageEncodeOptions {
                    compression_level: Some(CompressionLevel::Best),
                    tiff_compression: Some(TiffCompressionKind::Deflate),
                    ..ImageEncodeOptions::default()
                },
            )?,
            ImageExecutionMode::CpuTiff {
                compression: TiffCompression::Deflate {
                    level: CompressionLevel::Best,
                },
            }
        );
        Ok(())
    }

    #[test]
    fn rejects_compression_level_on_formats_and_tiff_modes_without_it() -> Result<(), DomainError> {
        let level = ImageEncodeOptions {
            compression_level: Some(CompressionLevel::Balanced),
            ..ImageEncodeOptions::default()
        };
        for format in [
            ImageOutputFormat::Bmp,
            ImageOutputFormat::Qoi,
            ImageOutputFormat::Ico,
            ImageOutputFormat::Pnm,
            ImageOutputFormat::Jpeg,
            ImageOutputFormat::Gif,
            ImageOutputFormat::Heic,
        ] {
            let rejected = ImageExecutionMode::resolve_cpu_with_options(format, None, level);
            assert!(
                matches!(
                    rejected,
                    Err(DomainError::UnsupportedImagePipeline {
                        format: rejected_format,
                        backend: HardwareBackend::Cpu,
                        ref reason,
                    }) if rejected_format == format && reason.contains("--compression-level")
                ),
                "--compression-level must be rejected for .{format}"
            );
        }

        for kind in [
            TiffCompressionKind::Lzw,
            TiffCompressionKind::Packbits,
            TiffCompressionKind::None,
        ] {
            let rejected = ImageExecutionMode::resolve_cpu_with_options(
                ImageOutputFormat::Tiff,
                None,
                ImageEncodeOptions {
                    compression_level: Some(CompressionLevel::Best),
                    tiff_compression: Some(kind),
                    ..ImageEncodeOptions::default()
                },
            );
            assert!(
                matches!(
                    rejected,
                    Err(DomainError::UnsupportedImagePipeline {
                        format: ImageOutputFormat::Tiff,
                        ref reason,
                        ..
                    }) if reason.contains("deflate")
                ),
                "--compression-level must be rejected for TIFF {kind}"
            );
        }

        let nvenc_heic = ImageExecutionMode::resolve_with_options(
            ImageOutputFormat::Heic,
            Some(ImageCompression::Lossy {
                quality: ImageQuality::DEFAULT,
            }),
            level,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            &rtx_2060_hardware()?,
        );
        assert!(matches!(
            nvenc_heic,
            Err(DomainError::UnsupportedImagePipeline {
                format: ImageOutputFormat::Heic,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn rejects_chroma_on_every_format_except_jpeg_and_avif() {
        let chroma = ImageEncodeOptions {
            chroma: Some(ChromaSubsampling::Yuv444),
            ..ImageEncodeOptions::default()
        };
        for format in [
            ImageOutputFormat::Webp,
            ImageOutputFormat::Png,
            ImageOutputFormat::Bmp,
            ImageOutputFormat::Qoi,
            ImageOutputFormat::Tiff,
            ImageOutputFormat::Jxl,
            ImageOutputFormat::Gif,
            ImageOutputFormat::Ico,
            ImageOutputFormat::Pnm,
            ImageOutputFormat::Heic,
        ] {
            let rejected = ImageExecutionMode::resolve_cpu_with_options(format, None, chroma);
            assert!(
                matches!(
                    rejected,
                    Err(DomainError::UnsupportedImagePipeline {
                        format: rejected_format,
                        backend: HardwareBackend::Cpu,
                        ref reason,
                    }) if rejected_format == format && reason.contains("--chroma")
                ),
                "--chroma must be rejected for .{format}"
            );
        }
    }

    #[test]
    fn rejects_tiff_compression_on_other_formats_and_gpu_backends() -> Result<(), DomainError> {
        let rtx = rtx_2060_hardware()?;
        let options = ImageEncodeOptions {
            tiff_compression: Some(TiffCompressionKind::Lzw),
            ..ImageEncodeOptions::default()
        };
        for format in [
            ImageOutputFormat::Webp,
            ImageOutputFormat::Heic,
            ImageOutputFormat::Jpeg,
            ImageOutputFormat::Png,
            ImageOutputFormat::Bmp,
            ImageOutputFormat::Qoi,
            ImageOutputFormat::Jxl,
            ImageOutputFormat::Avif,
            ImageOutputFormat::Gif,
            ImageOutputFormat::Ico,
            ImageOutputFormat::Pnm,
        ] {
            assert!(matches!(
                ImageExecutionMode::resolve_cpu_with_options(format, None, options),
                Err(DomainError::UnsupportedImagePipeline {
                    format: rejected_format,
                    backend: HardwareBackend::Cpu,
                    ref reason,
                }) if rejected_format == format && reason.contains("--tiff-compression")
            ));
        }
        for backend in [
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            HardwareBackend::Gpu(GpuBackend::Vulkan),
        ] {
            assert!(matches!(
                ImageExecutionMode::resolve_with_options(
                    ImageOutputFormat::Tiff,
                    None,
                    options,
                    backend,
                    &rtx,
                ),
                Err(DomainError::UnsupportedImagePipeline { .. })
            ));
        }
        Ok(())
    }
}
