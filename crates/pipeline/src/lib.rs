#![cfg_attr(not(feature = "nvenc"), forbid(unsafe_code))]

pub mod audio;
mod avif;
mod bmp;
pub mod cpu;
mod decode;
pub mod gpu;
mod qoi;
pub mod remux;

pub use audio::execute_audio_batch;

use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;

use cpu::CpuImageEncoder;

use fastconv_domain::{
    ExecutionPlan, GpuBackend, HardwareBackend, ImageBatchPlan, ImageConversionReport,
    ImageExecutionMode,
};
pub use remux::execute_video_batch;

/// Errors raised while executing a concrete media conversion pipeline.
#[derive(Debug)]
pub enum PipelineError {
    BackendNotImplemented(HardwareBackend),
    GpuBackendFailed { backend: GpuBackend, reason: String },
    Io { path: PathBuf, source: io::Error },
    DecodeFailed { path: PathBuf, reason: String },
    EncodeFailed { path: PathBuf, reason: String },
    RemuxFailed { path: PathBuf, reason: String },
    UnsupportedPixelFormat { path: PathBuf, format: String },
    UnsupportedSampleDepth { path: PathBuf, bits: usize },
    EmptyOutput(PathBuf),
    WorkerPanicked,
    WorkerStartFailed(io::Error),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendNotImplemented(backend) => {
                write!(f, "conversion backend '{backend}' is not implemented yet")
            }
            Self::GpuBackendFailed { backend, reason } => {
                write!(f, "GPU backend 'gpu:{backend}' failed: {reason}")
            }
            Self::Io { path, source } => {
                write!(f, "I/O error at '{}': {source}", path.display())
            }
            Self::DecodeFailed { path, reason } => {
                write!(f, "failed to decode image '{}': {reason}", path.display())
            }
            Self::EncodeFailed { path, reason } => {
                write!(f, "failed to encode image '{}': {reason}", path.display())
            }
            Self::RemuxFailed { path, reason } => {
                write!(f, "remux failed for '{}': {reason}", path.display())
            }
            Self::UnsupportedPixelFormat { path, format } => {
                write!(
                    f,
                    "unsupported decoded pixel format '{format}' in '{}'",
                    path.display()
                )
            }
            Self::UnsupportedSampleDepth { path, bits } => write!(
                f,
                "unsupported audio sample depth {bits} in '{}'; supported depths are 8, 16 and 24 bits",
                path.display()
            ),
            Self::EmptyOutput(path) => {
                write!(f, "encoder produced empty output for '{}'", path.display())
            }
            Self::WorkerPanicked => f.write_str("conversion worker thread panicked"),
            Self::WorkerStartFailed(source) => {
                write!(f, "failed to start conversion worker: {source}")
            }
        }
    }
}

impl Error for PipelineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } | Self::WorkerStartFailed(source) => Some(source),
            Self::BackendNotImplemented(_)
            | Self::GpuBackendFailed { .. }
            | Self::DecodeFailed { .. }
            | Self::EncodeFailed { .. }
            | Self::RemuxFailed { .. }
            | Self::UnsupportedPixelFormat { .. }
            | Self::UnsupportedSampleDepth { .. }
            | Self::EmptyOutput(_)
            | Self::WorkerPanicked => None,
        }
    }
}

/// Dispatches a validated [`ExecutionPlan`] to its concrete CPU or GPU backend.
///
/// # Errors
///
/// Returns [`PipelineError::BackendNotImplemented`] until the selected concrete backend
/// is implemented. Never falls back silently from GPU to CPU.
pub fn execute(plan: &ExecutionPlan) -> Result<(), PipelineError> {
    match plan {
        ExecutionPlan::Cpu { .. } => cpu::execute(plan),
        ExecutionPlan::GpuZeroCopy { backend, .. } => gpu::execute(*backend, plan),
    }
}

/// Executes a validated [`ImageBatchPlan`] on its exact selected CPU or GPU backend.
///
/// Never performs silent CPU fallbacks when a GPU mode is selected.
///
/// # Errors
///
/// Returns [`PipelineError`] if reading, decoding, GPU compute/encode, or writing any image fails.
pub fn execute_image_batch(plan: &ImageBatchPlan) -> Result<ImageConversionReport, PipelineError> {
    match plan.mode() {
        ImageExecutionMode::CpuWebp(compression) => {
            cpu::execute_cpu_image_batch(plan, CpuImageEncoder::Webp(compression))
        }
        ImageExecutionMode::CpuJpeg { quality } => {
            cpu::execute_cpu_image_batch(plan, CpuImageEncoder::Jpeg { quality })
        }
        ImageExecutionMode::CpuPng => cpu::execute_cpu_image_batch(plan, CpuImageEncoder::Png),
        ImageExecutionMode::CpuBmp => cpu::execute_cpu_image_batch(plan, CpuImageEncoder::Bmp),
        ImageExecutionMode::CpuQoi => cpu::execute_cpu_image_batch(plan, CpuImageEncoder::Qoi),
        ImageExecutionMode::CpuTiff => cpu::execute_cpu_image_batch(plan, CpuImageEncoder::Tiff),
        ImageExecutionMode::CpuJxl(compression) => {
            cpu::execute_cpu_image_batch(plan, CpuImageEncoder::Jxl(compression))
        }
        ImageExecutionMode::CpuAvif { quality } => cpu::execute_cpu_image_batch(
            plan,
            CpuImageEncoder::Avif {
                quality,
                threads: 1,
            },
        ),
        ImageExecutionMode::NvencHeic { quality } => gpu::nvenc::execute_heic_batch(plan, quality),
    }
}

#[cfg(test)]
mod tests {
    use super::{PipelineError, execute, execute_image_batch};
    use fastconv_domain::{
        ConversionRequest, DomainError, ExecutionPlan, GpuBackend, HardwareBackend, ImageBatchItem,
        ImageBatchPlan, ImageCompression, ImageDimensions, ImageExecutionMode, ImageInputFormat,
        ImageQuality, MediaKind,
    };
    use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder};
    use std::env;
    use std::fs::{self, File};
    use std::io::BufWriter;
    use std::num::NonZeroUsize;
    use std::path::{Path, PathBuf};
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};
    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::bytestream::ZCursor;

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn create_temp_dir(label: &str) -> Result<PathBuf, PipelineError> {
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("fastconv-pipeline-{label}-{}-{id}", process::id()));
        fs::create_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(dir)
    }

    fn write_test_png(
        path: &Path,
        width: u32,
        height: u32,
        color: png::ColorType,
        pixels: &[u8],
    ) -> Result<(), PipelineError> {
        let file = File::create(path).map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
        encoder.set_color(color);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|err| PipelineError::EncodeFailed {
                path: path.to_path_buf(),
                reason: err.to_string(),
            })?;
        writer
            .write_image_data(pixels)
            .map_err(|err| PipelineError::EncodeFailed {
                path: path.to_path_buf(),
                reason: err.to_string(),
            })?;
        Ok(())
    }

    fn write_test_jpeg(
        path: &Path,
        width: u16,
        height: u16,
        color: JpegColorType,
        pixels: &[u8],
    ) -> Result<(), PipelineError> {
        let encoder =
            JpegEncoder::new_file(path, 90).map_err(|err| PipelineError::EncodeFailed {
                path: path.to_path_buf(),
                reason: err.to_string(),
            })?;
        encoder
            .encode(pixels, width, height, color)
            .map_err(|err| PipelineError::EncodeFailed {
                path: path.to_path_buf(),
                reason: err.to_string(),
            })
    }

    fn make_rgb_gradient(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                let r = u8::try_from((x * 5) % 256).unwrap_or(0);
                let g = u8::try_from((y * 7) % 256).unwrap_or(0);
                let b = u8::try_from(((x ^ y) * 3) % 256).unwrap_or(0);
                pixels.extend_from_slice(&[r, g, b]);
            }
        }
        pixels
    }

    #[test]
    fn dispatches_cpu_plan_explicitly() -> Result<(), DomainError> {
        let request = ConversionRequest::new(
            PathBuf::from("in.mp4"),
            PathBuf::from("out.mkv"),
            MediaKind::Video,
            HardwareBackend::Cpu,
        )?;
        let plan = ExecutionPlan::from_request(request)?;

        assert!(matches!(
            execute(&plan),
            Err(PipelineError::BackendNotImplemented(HardwareBackend::Cpu))
        ));
        Ok(())
    }

    #[test]
    fn dispatches_gpu_plan_without_silent_cpu_fallback() -> Result<(), DomainError> {
        let request = ConversionRequest::new(
            PathBuf::from("in.mp4"),
            PathBuf::from("out.mp4"),
            MediaKind::Video,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
        )?;
        let plan = ExecutionPlan::from_request(request)?;

        assert!(matches!(
            execute(&plan),
            Err(PipelineError::BackendNotImplemented(HardwareBackend::Gpu(
                GpuBackend::Nvenc
            )))
        ));
        Ok(())
    }

    #[test]
    fn converts_rgb_and_rgba_pngs_to_valid_lossy_and_lossless_webp()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = create_temp_dir("webp-cpu")?;
        let rgb_path = root.join("rgb.png");
        let alpha_path = root.join("alpha.png");
        let rgb_out = root.join("nested/rgb.webp");
        let alpha_out = root.join("nested/alpha.webp");

        let rgb_pixels = make_rgb_gradient(64, 48);
        let mut alpha_pixels = Vec::with_capacity(32 * 24 * 4);
        for y in 0_u32..24 {
            for x in 0_u32..32 {
                let r = u8::try_from((x * 8) % 256).unwrap_or(0);
                let g = u8::try_from((y * 10) % 256).unwrap_or(0);
                let b = u8::try_from((x + y) % 256).unwrap_or(0);
                let a = if x < 16 { 255 } else { 180 };
                alpha_pixels.extend_from_slice(&[r, g, b, a]);
            }
        }

        write_test_png(&rgb_path, 64, 48, png::ColorType::Rgb, &rgb_pixels)?;
        write_test_png(&alpha_path, 32, 24, png::ColorType::Rgba, &alpha_pixels)?;

        let workers = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        let lossy_plan = ImageBatchPlan::new(
            vec![
                ImageBatchItem::new(
                    rgb_path.clone(),
                    rgb_out.clone(),
                    ImageInputFormat::Png,
                    ImageDimensions::new(64, 48)?,
                )?,
                ImageBatchItem::new(
                    alpha_path.clone(),
                    alpha_out.clone(),
                    ImageInputFormat::Png,
                    ImageDimensions::new(32, 24)?,
                )?,
            ],
            ImageExecutionMode::CpuWebp(ImageCompression::Lossy {
                quality: ImageQuality::DEFAULT,
            }),
            workers,
        )?;

        let report = execute_image_batch(&lossy_plan)?;
        assert_eq!(report.files_converted().get(), 2);
        assert_eq!(report.backend(), HardwareBackend::Cpu);
        assert_eq!(report.hardware_device(), None);

        let rgb_webp = fs::read(&rgb_out)?;
        let (dec_rgb, w, h) = zenwebp::decode_rgb(&rgb_webp)?;
        assert_eq!((w, h), (64, 48));
        assert_eq!(dec_rgb.len(), 64 * 48 * 3);

        let lossless_out = root.join("alpha_lossless.webp");
        let lossless_plan = ImageBatchPlan::new(
            vec![ImageBatchItem::new(
                alpha_path,
                lossless_out.clone(),
                ImageInputFormat::Png,
                ImageDimensions::new(32, 24)?,
            )?],
            ImageExecutionMode::CpuWebp(ImageCompression::Lossless),
            NonZeroUsize::MIN,
        )?;
        let lossless_report = execute_image_batch(&lossless_plan)?;
        assert_eq!(lossless_report.files_converted().get(), 1);
        let lossless_bytes = fs::read(&lossless_out)?;
        let (dec_alpha, lw, lh) = zenwebp::decode_rgba(&lossless_bytes)?;
        assert_eq!((lw, lh), (32, 24));
        assert_eq!(dec_alpha, alpha_pixels);

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    fn u64_to_f64(value: u64) -> f64 {
        let hi = u32::try_from(value >> 32).unwrap_or(u32::MAX);
        let lo = u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX);
        f64::from(hi) * 4_294_967_296.0 + f64::from(lo)
    }

    fn mean_abs_channel_diff(a: &[u8], b: &[u8]) -> Result<f64, Box<dyn std::error::Error>> {
        if a.is_empty() || a.len() != b.len() {
            return Err(format!("buffer length mismatch: {} vs {}", a.len(), b.len()).into());
        }
        let sum: u64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| u64::from(x.abs_diff(y)))
            .sum();
        let len = u64::try_from(a.len())?;
        Ok(u64_to_f64(sum) / u64_to_f64(len))
    }

    fn jpeg_item(
        source: PathBuf,
        target: PathBuf,
        input_format: ImageInputFormat,
        w: u32,
        h: u32,
    ) -> Result<ImageBatchItem, DomainError> {
        ImageBatchItem::new(source, target, input_format, ImageDimensions::new(w, h)?)
    }

    #[test]
    fn converts_png_and_jpeg_inputs_to_jpeg() -> Result<(), Box<dyn std::error::Error>> {
        let root = create_temp_dir("nm-cpu-jpeg")?;
        let (png_rgb, png_luma, png_rgba, jpg_color, png_la) = (
            root.join("a.png"),
            root.join("b.png"),
            root.join("c.png"),
            root.join("d.jpg"),
            root.join("e.png"),
        );

        let grad_rgb = make_rgb_gradient(64, 48);
        let grad_luma: Vec<u8> = (0_u32..48 * 32)
            .map(|i| u8::try_from((i * 7) % 256).unwrap_or(0))
            .collect();
        let grad_luma_rgb: Vec<u8> = grad_luma.iter().flat_map(|&v| [v, v, v]).collect();
        let opaque_rgba: Vec<u8> = grad_rgb
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&[r, g, b]| [r, g, b, 255])
            .collect();
        let opaque_la: Vec<u8> = grad_luma.iter().flat_map(|&v| [v, 255]).collect();

        write_test_png(&png_rgb, 64, 48, png::ColorType::Rgb, &grad_rgb)?;
        write_test_png(&png_luma, 32, 48, png::ColorType::Grayscale, &grad_luma)?;
        write_test_png(&png_rgba, 64, 48, png::ColorType::Rgba, &opaque_rgba)?;
        write_test_jpeg(&jpg_color, 64, 48, JpegColorType::Rgb, &grad_rgb)?;
        write_test_png(&png_la, 32, 48, png::ColorType::GrayscaleAlpha, &opaque_la)?;

        let workers = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        let plan = ImageBatchPlan::new(
            vec![
                jpeg_item(
                    png_rgb,
                    root.join("out/a.jpg"),
                    ImageInputFormat::Png,
                    64,
                    48,
                )?,
                jpeg_item(
                    png_luma,
                    root.join("out/b.jpg"),
                    ImageInputFormat::Png,
                    32,
                    48,
                )?,
                jpeg_item(
                    png_rgba,
                    root.join("out/c.jpg"),
                    ImageInputFormat::Png,
                    64,
                    48,
                )?,
                jpeg_item(
                    jpg_color,
                    root.join("out/d.jpg"),
                    ImageInputFormat::Jpeg,
                    64,
                    48,
                )?,
                jpeg_item(
                    png_la,
                    root.join("out/e.jpg"),
                    ImageInputFormat::Png,
                    32,
                    48,
                )?,
            ],
            ImageExecutionMode::CpuJpeg {
                quality: ImageQuality::DEFAULT,
            },
            workers,
        )?;
        assert_eq!(execute_image_batch(&plan)?.files_converted().get(), 5);

        for (out_name, expected_dims, expected_rgb) in [
            ("out/a.jpg", (64, 48), grad_rgb.as_slice()),
            ("out/b.jpg", (32, 48), grad_luma_rgb.as_slice()),
            ("out/c.jpg", (64, 48), grad_rgb.as_slice()),
            ("out/d.jpg", (64, 48), grad_rgb.as_slice()),
            ("out/e.jpg", (32, 48), grad_luma_rgb.as_slice()),
        ] {
            let bytes = fs::read(root.join(out_name))?;
            let mut dec = JpegDecoder::new(ZCursor::new(bytes.as_slice()));
            let pixels = dec.decode()?;
            let info = dec.info().ok_or("missing JPEG info")?;
            assert_eq!(
                (usize::from(info.width), usize::from(info.height)),
                expected_dims
            );
            assert_eq!(pixels.len(), expected_dims.0 * expected_dims.1 * 3);
            assert!(
                mean_abs_channel_diff(&pixels, expected_rgb)? < 12.0,
                "{out_name} diverged too far from expected RGB"
            );
        }

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn converts_mixed_png_and_jpeg_inputs_to_webp() -> Result<(), Box<dyn std::error::Error>> {
        let root = create_temp_dir("nm-cpu-webp")?;
        let png_rgb = root.join("a.png");
        let jpg_color = root.join("d.jpg");
        let jpg_luma = root.join("e.jpg");

        let grad_rgb = make_rgb_gradient(64, 48);
        let grad_luma: Vec<u8> = (0_u32..48 * 32)
            .map(|i| u8::try_from((i * 7) % 256).unwrap_or(0))
            .collect();
        let grad_luma_rgb: Vec<u8> = grad_luma.iter().flat_map(|&v| [v, v, v]).collect();
        write_test_png(&png_rgb, 64, 48, png::ColorType::Rgb, &grad_rgb)?;
        write_test_jpeg(&jpg_color, 64, 48, JpegColorType::Rgb, &grad_rgb)?;
        write_test_jpeg(&jpg_luma, 32, 48, JpegColorType::Luma, &grad_luma)?;

        let workers = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        let mixed_webp_plan = ImageBatchPlan::new(
            vec![
                ImageBatchItem::new(
                    png_rgb,
                    root.join("out_webp/a.webp"),
                    ImageInputFormat::Png,
                    ImageDimensions::new(64, 48)?,
                )?,
                ImageBatchItem::new(
                    jpg_color,
                    root.join("out_webp/d.webp"),
                    ImageInputFormat::Jpeg,
                    ImageDimensions::new(64, 48)?,
                )?,
                ImageBatchItem::new(
                    jpg_luma,
                    root.join("out_webp/e.webp"),
                    ImageInputFormat::Jpeg,
                    ImageDimensions::new(32, 48)?,
                )?,
            ],
            ImageExecutionMode::CpuWebp(ImageCompression::Lossless),
            workers,
        )?;
        let webp_report = execute_image_batch(&mixed_webp_plan)?;
        assert_eq!(webp_report.files_converted().get(), 3);

        let (dec_a, aw, ah) = zenwebp::decode_rgb(&fs::read(root.join("out_webp/a.webp"))?)?;
        assert_eq!((aw, ah), (64, 48));
        assert_eq!(dec_a, grad_rgb);

        let (dec_d, dw, dh) = zenwebp::decode_rgb(&fs::read(root.join("out_webp/d.webp"))?)?;
        assert_eq!((dw, dh), (64, 48));
        assert!(mean_abs_channel_diff(&dec_d, &grad_rgb)? < 12.0);

        let (dec_e, ew, eh) = zenwebp::decode_rgb(&fs::read(root.join("out_webp/e.webp"))?)?;
        assert_eq!((ew, eh), (32, 48));
        assert!(mean_abs_channel_diff(&dec_e, &grad_luma_rgb)? < 12.0);

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn jpeg_encode_rejects_alpha_and_corrupt_jpeg_scan_fails_decode()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = create_temp_dir("jpeg-errors")?;
        let alpha_png = root.join("alpha.png");
        write_test_png(
            &alpha_png,
            8,
            8,
            png::ColorType::Rgba,
            &vec![128_u8; 8 * 8 * 4],
        )?;

        let alpha_plan = ImageBatchPlan::new(
            vec![ImageBatchItem::new(
                alpha_png,
                root.join("alpha.jpg"),
                ImageInputFormat::Png,
                ImageDimensions::new(8, 8)?,
            )?],
            ImageExecutionMode::CpuJpeg {
                quality: ImageQuality::DEFAULT,
            },
            NonZeroUsize::MIN,
        )?;
        assert!(matches!(
            execute_image_batch(&alpha_plan),
            Err(PipelineError::UnsupportedPixelFormat { .. })
        ));

        let valid_jpg = root.join("valid.jpg");
        write_test_jpeg(
            &valid_jpg,
            32,
            32,
            JpegColorType::Rgb,
            &make_rgb_gradient(32, 32),
        )?;
        let mut truncated = fs::read(&valid_jpg)?;
        truncated.truncate(truncated.len() / 2);
        let broken_jpg = root.join("truncated.jpg");
        fs::write(&broken_jpg, &truncated)?;

        let broken_plan = ImageBatchPlan::new(
            vec![ImageBatchItem::new(
                broken_jpg,
                root.join("broken.webp"),
                ImageInputFormat::Jpeg,
                ImageDimensions::new(32, 32)?,
            )?],
            ImageExecutionMode::CpuWebp(ImageCompression::Lossy {
                quality: ImageQuality::DEFAULT,
            }),
            NonZeroUsize::MIN,
        )?;
        assert!(matches!(
            execute_image_batch(&broken_plan),
            Err(PipelineError::DecodeFailed { .. })
        ));

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn jpeg_encode_supports_quality_zero_and_rejects_transparent_la8()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = create_temp_dir("jpeg-edge")?;
        let alpha_la_png = root.join("alpha_la.png");
        write_test_png(
            &alpha_la_png,
            8,
            8,
            png::ColorType::GrayscaleAlpha,
            &[128_u8; 8 * 8 * 2],
        )?;
        let alpha_la_plan = ImageBatchPlan::new(
            vec![ImageBatchItem::new(
                alpha_la_png,
                root.join("alpha_la.jpg"),
                ImageInputFormat::Png,
                ImageDimensions::new(8, 8)?,
            )?],
            ImageExecutionMode::CpuJpeg {
                quality: ImageQuality::DEFAULT,
            },
            NonZeroUsize::MIN,
        )?;
        assert!(matches!(
            execute_image_batch(&alpha_la_plan),
            Err(PipelineError::UnsupportedPixelFormat { .. })
        ));

        let q0_src = root.join("q0_in.png");
        let q0_out = root.join("q0_out.jpg");
        write_test_png(
            &q0_src,
            16,
            16,
            png::ColorType::Rgb,
            &make_rgb_gradient(16, 16),
        )?;
        let q0_plan = ImageBatchPlan::new(
            vec![ImageBatchItem::new(
                q0_src,
                q0_out.clone(),
                ImageInputFormat::Png,
                ImageDimensions::new(16, 16)?,
            )?],
            ImageExecutionMode::CpuJpeg {
                quality: ImageQuality::new(0)?,
            },
            NonZeroUsize::MIN,
        )?;
        execute_image_batch(&q0_plan)?;
        let q0_bytes = fs::read(&q0_out)?;
        let q0_pixels = JpegDecoder::new(ZCursor::new(q0_bytes.as_slice())).decode()?;
        assert_eq!(q0_pixels.len(), 16 * 16 * 3);

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[cfg(not(feature = "nvenc"))]
    #[test]
    fn nvenc_heic_fails_explicitly_without_nvenc_feature() -> Result<(), Box<dyn std::error::Error>>
    {
        let root = create_temp_dir("nvenc-disabled")?;
        let src = root.join("in.png");
        write_test_png(
            &src,
            256,
            256,
            png::ColorType::Rgb,
            &make_rgb_gradient(256, 256),
        )?;
        let plan = ImageBatchPlan::new(
            vec![ImageBatchItem::new(
                src,
                root.join("out.heic"),
                ImageInputFormat::Png,
                ImageDimensions::new(256, 256)?,
            )?],
            ImageExecutionMode::NvencHeic {
                quality: ImageQuality::DEFAULT,
            },
            NonZeroUsize::MIN,
        )?;
        assert!(matches!(
            execute_image_batch(&plan),
            Err(PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                ..
            })
        ));
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    struct DecodedTestPng {
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        color_type: png::ColorType,
    }

    fn decode_test_png(path: &Path) -> Result<DecodedTestPng, Box<dyn std::error::Error>> {
        let dec = png::Decoder::new(std::io::BufReader::new(File::open(path)?));
        let mut reader = dec.read_info()?;
        let mut buf = vec![0_u8; reader.output_buffer_size().ok_or("png buf")?];
        let frame = reader.next_frame(&mut buf)?;
        buf.truncate(frame.buffer_size());
        Ok(DecodedTestPng {
            pixels: buf,
            width: frame.width,
            height: frame.height,
            color_type: frame.color_type,
        })
    }

    #[test]
    fn roundtrips_png_qoi_webp_png_losslessly() -> Result<(), Box<dyn std::error::Error>> {
        let root = create_temp_dir("nm-lossless-chain")?;
        let grad_rgb = make_rgb_gradient(53, 37);
        let alpha_rgba: Vec<u8> = grad_rgb
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
            .flat_map(|(i, &[r, g, b])| [r, g, b, if i % 4 == 0 { 160 } else { 255 }])
            .collect();

        let src_png = root.join("src.png");
        let src_alpha_png = root.join("src_alpha.png");
        write_test_png(&src_png, 53, 37, png::ColorType::Rgb, &grad_rgb)?;
        write_test_png(&src_alpha_png, 53, 37, png::ColorType::Rgba, &alpha_rgba)?;
        let workers = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);

        let qoi_rgb = root.join("stage1/rgb.qoi");
        let qoi_rgba = root.join("stage1/rgba.qoi");
        execute_image_batch(&ImageBatchPlan::new(
            vec![
                jpeg_item(src_png, qoi_rgb.clone(), ImageInputFormat::Png, 53, 37)?,
                jpeg_item(
                    src_alpha_png,
                    qoi_rgba.clone(),
                    ImageInputFormat::Png,
                    53,
                    37,
                )?,
            ],
            ImageExecutionMode::CpuQoi,
            workers,
        )?)?;

        let webp_rgb = root.join("stage2/rgb.webp");
        let webp_rgba = root.join("stage2/rgba.webp");
        execute_image_batch(&ImageBatchPlan::new(
            vec![
                jpeg_item(qoi_rgb, webp_rgb.clone(), ImageInputFormat::Qoi, 53, 37)?,
                jpeg_item(qoi_rgba, webp_rgba.clone(), ImageInputFormat::Qoi, 53, 37)?,
            ],
            ImageExecutionMode::CpuWebp(ImageCompression::Lossless),
            workers,
        )?)?;

        let png_rgb = root.join("stage3/rgb.png");
        let png_rgba = root.join("stage3/rgba.png");
        execute_image_batch(&ImageBatchPlan::new(
            vec![
                jpeg_item(webp_rgb, png_rgb.clone(), ImageInputFormat::Webp, 53, 37)?,
                jpeg_item(webp_rgba, png_rgba.clone(), ImageInputFormat::Webp, 53, 37)?,
            ],
            ImageExecutionMode::CpuPng,
            workers,
        )?)?;

        let dec_rgb = decode_test_png(&png_rgb)?;
        assert_eq!(
            (dec_rgb.width, dec_rgb.height, dec_rgb.color_type),
            (53, 37, png::ColorType::Rgb)
        );
        assert_eq!(dec_rgb.pixels, grad_rgb);

        let dec_alpha = decode_test_png(&png_rgba)?;
        assert_eq!(
            (dec_alpha.width, dec_alpha.height, dec_alpha.color_type),
            (53, 37, png::ColorType::Rgba)
        );
        assert_eq!(dec_alpha.pixels, alpha_rgba);

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn converts_bmp_to_jpeg_and_rejects_transparent_bmp_for_jpeg()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = create_temp_dir("nm-bmp-jpeg")?;
        let grad_rgb = make_rgb_gradient(53, 37);
        let alpha_rgba: Vec<u8> = grad_rgb
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
            .flat_map(|(i, &[r, g, b])| [r, g, b, if i % 4 == 0 { 160 } else { 255 }])
            .collect();

        let src_png = root.join("src.png");
        let src_alpha_png = root.join("src_alpha.png");
        write_test_png(&src_png, 53, 37, png::ColorType::Rgb, &grad_rgb)?;
        write_test_png(&src_alpha_png, 53, 37, png::ColorType::Rgba, &alpha_rgba)?;
        let workers = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);

        let bmp_rgb = root.join("rgb.bmp");
        let bmp_rgba = root.join("rgba.bmp");
        execute_image_batch(&ImageBatchPlan::new(
            vec![
                jpeg_item(src_png, bmp_rgb.clone(), ImageInputFormat::Png, 53, 37)?,
                jpeg_item(
                    src_alpha_png,
                    bmp_rgba.clone(),
                    ImageInputFormat::Png,
                    53,
                    37,
                )?,
            ],
            ImageExecutionMode::CpuBmp,
            workers,
        )?)?;

        let jpg_from_bmp = root.join("rgb.jpg");
        execute_image_batch(&ImageBatchPlan::new(
            vec![jpeg_item(
                bmp_rgb,
                jpg_from_bmp.clone(),
                ImageInputFormat::Bmp,
                53,
                37,
            )?],
            ImageExecutionMode::CpuJpeg {
                quality: ImageQuality::DEFAULT,
            },
            NonZeroUsize::MIN,
        )?)?;
        let jpg_bytes = fs::read(&jpg_from_bmp)?;
        let jpg_pixels = JpegDecoder::new(ZCursor::new(jpg_bytes.as_slice())).decode()?;
        assert!(mean_abs_channel_diff(&jpg_pixels, &grad_rgb)? < 12.0);

        let alpha_bmp_to_jpg = ImageBatchPlan::new(
            vec![jpeg_item(
                bmp_rgba,
                root.join("alpha.jpg"),
                ImageInputFormat::Bmp,
                53,
                37,
            )?],
            ImageExecutionMode::CpuJpeg {
                quality: ImageQuality::DEFAULT,
            },
            NonZeroUsize::MIN,
        )?;
        assert!(matches!(
            execute_image_batch(&alpha_bmp_to_jpg),
            Err(PipelineError::UnsupportedPixelFormat { .. })
        ));

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn converts_mixed_five_format_batch_to_webp() -> Result<(), Box<dyn std::error::Error>> {
        let root = create_temp_dir("nm-mixed-five")?;
        let grad_rgb = make_rgb_gradient(53, 37);
        let src_png = root.join("src.png");
        let src_jpg = root.join("src.jpg");
        let src_webp = root.join("src.webp");
        let src_bmp = root.join("src.bmp");
        let src_qoi = root.join("src.qoi");
        write_test_png(&src_png, 53, 37, png::ColorType::Rgb, &grad_rgb)?;
        write_test_jpeg(&src_jpg, 53, 37, JpegColorType::Rgb, &grad_rgb)?;
        execute_image_batch(&ImageBatchPlan::new(
            vec![jpeg_item(
                src_png.clone(),
                src_webp.clone(),
                ImageInputFormat::Png,
                53,
                37,
            )?],
            ImageExecutionMode::CpuWebp(ImageCompression::Lossless),
            NonZeroUsize::MIN,
        )?)?;
        execute_image_batch(&ImageBatchPlan::new(
            vec![jpeg_item(
                src_png.clone(),
                src_bmp.clone(),
                ImageInputFormat::Png,
                53,
                37,
            )?],
            ImageExecutionMode::CpuBmp,
            NonZeroUsize::MIN,
        )?)?;
        execute_image_batch(&ImageBatchPlan::new(
            vec![jpeg_item(
                src_png.clone(),
                src_qoi.clone(),
                ImageInputFormat::Png,
                53,
                37,
            )?],
            ImageExecutionMode::CpuQoi,
            NonZeroUsize::MIN,
        )?)?;

        let workers = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        let mixed_plan = ImageBatchPlan::new(
            vec![
                jpeg_item(
                    src_png,
                    root.join("mixed/01.webp"),
                    ImageInputFormat::Png,
                    53,
                    37,
                )?,
                jpeg_item(
                    src_jpg,
                    root.join("mixed/02.webp"),
                    ImageInputFormat::Jpeg,
                    53,
                    37,
                )?,
                jpeg_item(
                    src_webp,
                    root.join("mixed/03.webp"),
                    ImageInputFormat::Webp,
                    53,
                    37,
                )?,
                jpeg_item(
                    src_bmp,
                    root.join("mixed/04.webp"),
                    ImageInputFormat::Bmp,
                    53,
                    37,
                )?,
                jpeg_item(
                    src_qoi,
                    root.join("mixed/05.webp"),
                    ImageInputFormat::Qoi,
                    53,
                    37,
                )?,
            ],
            ImageExecutionMode::CpuWebp(ImageCompression::Lossy {
                quality: ImageQuality::DEFAULT,
            }),
            workers,
        )?;
        assert_eq!(execute_image_batch(&mixed_plan)?.files_converted().get(), 5);

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[cfg(feature = "nvenc")]
    #[test]
    fn converts_mixed_all_five_input_formats_to_valid_heic_via_nvenc_hardware()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = create_temp_dir("nvenc-heic")?;
        let src_png = root.join("a.png");
        let src_jpg = root.join("b.jpg");
        let src_webp = root.join("c.webp");
        let src_bmp = root.join("d.bmp");
        let src_qoi = root.join("e.qoi");
        let out_a = root.join("out/a.heic");
        let out_b = root.join("out/b.heic");
        let out_c = root.join("out/c.heic");
        let out_d = root.join("out/d.heic");
        let out_e = root.join("out/e.heic");

        let grad = make_rgb_gradient(256, 256);
        write_test_png(&src_png, 256, 256, png::ColorType::Rgb, &grad)?;
        write_test_jpeg(&src_jpg, 256, 256, JpegColorType::Rgb, &grad)?;
        execute_image_batch(&ImageBatchPlan::new(
            vec![jpeg_item(
                src_png.clone(),
                src_webp.clone(),
                ImageInputFormat::Png,
                256,
                256,
            )?],
            ImageExecutionMode::CpuWebp(ImageCompression::Lossy {
                quality: ImageQuality::DEFAULT,
            }),
            NonZeroUsize::MIN,
        )?)?;
        execute_image_batch(&ImageBatchPlan::new(
            vec![jpeg_item(
                src_png.clone(),
                src_bmp.clone(),
                ImageInputFormat::Png,
                256,
                256,
            )?],
            ImageExecutionMode::CpuBmp,
            NonZeroUsize::MIN,
        )?)?;
        execute_image_batch(&ImageBatchPlan::new(
            vec![jpeg_item(
                src_png.clone(),
                src_qoi.clone(),
                ImageInputFormat::Png,
                256,
                256,
            )?],
            ImageExecutionMode::CpuQoi,
            NonZeroUsize::MIN,
        )?)?;

        let plan = ImageBatchPlan::new(
            vec![
                jpeg_item(src_png, out_a.clone(), ImageInputFormat::Png, 256, 256)?,
                jpeg_item(src_jpg, out_b.clone(), ImageInputFormat::Jpeg, 256, 256)?,
                jpeg_item(src_webp, out_c.clone(), ImageInputFormat::Webp, 256, 256)?,
                jpeg_item(src_bmp, out_d.clone(), ImageInputFormat::Bmp, 256, 256)?,
                jpeg_item(src_qoi, out_e.clone(), ImageInputFormat::Qoi, 256, 256)?,
            ],
            ImageExecutionMode::NvencHeic {
                quality: ImageQuality::DEFAULT,
            },
            NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN),
        )?;

        let report = execute_image_batch(&plan)?;
        assert_eq!(report.files_converted().get(), 5);
        assert_eq!(report.backend(), HardwareBackend::Gpu(GpuBackend::Nvenc));
        assert!(report.hardware_device().is_some());

        for path in [&out_a, &out_b, &out_c, &out_d, &out_e] {
            let bytes = fs::read(path)?;
            assert_eq!(bytes.get(4..12), Some(b"ftypheic".as_slice()));
        }

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[cfg(feature = "nvenc")]
    #[test]
    fn nvenc_heic_rejects_transparent_png_without_silent_fallback()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = create_temp_dir("nvenc-heic-alpha")?;
        let src = root.join("alpha.png");
        let out = root.join("alpha.heic");
        let mut alpha_pixels = Vec::with_capacity(256 * 256 * 4);
        for _ in 0..(256 * 256) {
            alpha_pixels.extend_from_slice(&[120, 180, 220, 128]);
        }
        write_test_png(&src, 256, 256, png::ColorType::Rgba, &alpha_pixels)?;

        let plan = ImageBatchPlan::new(
            vec![ImageBatchItem::new(
                src,
                out,
                ImageInputFormat::Png,
                ImageDimensions::new(256, 256)?,
            )?],
            ImageExecutionMode::NvencHeic {
                quality: ImageQuality::DEFAULT,
            },
            NonZeroUsize::MIN,
        )?;

        assert!(matches!(
            execute_image_batch(&plan),
            Err(PipelineError::UnsupportedPixelFormat { .. })
        ));

        fs::remove_dir_all(&root)?;
        Ok(())
    }
}
