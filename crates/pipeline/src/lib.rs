#![cfg_attr(not(feature = "nvenc"), forbid(unsafe_code))]

pub mod cpu;
pub mod gpu;
mod png_decode;

use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;

use fastconv_domain::{
    ExecutionPlan, GpuBackend, HardwareBackend, ImageBatchPlan, ImageConversionReport,
    ImageExecutionMode,
};

/// Errors raised while executing a concrete media conversion pipeline.
#[derive(Debug)]
pub enum PipelineError {
    BackendNotImplemented(HardwareBackend),
    GpuBackendFailed { backend: GpuBackend, reason: String },
    Io { path: PathBuf, source: io::Error },
    DecodeFailed { path: PathBuf, reason: String },
    EncodeFailed { path: PathBuf, reason: String },
    UnsupportedPixelFormat { path: PathBuf, format: String },
    EmptyOutput(PathBuf),
    WorkerPanicked,
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
                write!(f, "failed to decode PNG '{}': {reason}", path.display())
            }
            Self::EncodeFailed { path, reason } => {
                write!(f, "failed to encode image '{}': {reason}", path.display())
            }
            Self::UnsupportedPixelFormat { path, format } => {
                write!(
                    f,
                    "unsupported decoded PNG pixel format '{format}' in '{}'",
                    path.display()
                )
            }
            Self::EmptyOutput(path) => {
                write!(f, "encoder produced empty output for '{}'", path.display())
            }
            Self::WorkerPanicked => f.write_str("conversion worker thread panicked"),
        }
    }
}

impl Error for PipelineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::BackendNotImplemented(_)
            | Self::GpuBackendFailed { .. }
            | Self::DecodeFailed { .. }
            | Self::EncodeFailed { .. }
            | Self::UnsupportedPixelFormat { .. }
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
        ImageExecutionMode::CpuWebp(encoding) => cpu::execute_webp_batch(plan, encoding),
        ImageExecutionMode::NvencHeic { quality } => gpu::nvenc::execute_heic_batch(plan, quality),
    }
}

#[cfg(test)]
mod tests {
    use super::{PipelineError, execute, execute_image_batch};
    use fastconv_domain::{
        ConversionRequest, DomainError, ExecutionPlan, GpuBackend, HardwareBackend, ImageBatchItem,
        ImageBatchPlan, ImageDimensions, ImageExecutionMode, ImageQuality, MediaKind, WebpEncoding,
    };
    use std::env;
    use std::fs::{self, File};
    use std::io::BufWriter;
    use std::num::NonZeroUsize;
    use std::path::{Path, PathBuf};
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};

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
                    ImageDimensions::new(64, 48)?,
                )?,
                ImageBatchItem::new(
                    alpha_path.clone(),
                    alpha_out.clone(),
                    ImageDimensions::new(32, 24)?,
                )?,
            ],
            ImageExecutionMode::CpuWebp(WebpEncoding::Lossy {
                quality: ImageQuality::DEFAULT,
            }),
            workers,
        )?;

        let report = execute_image_batch(&lossy_plan)?;
        assert_eq!(report.files_converted().get(), 2);
        assert_eq!(report.backend(), HardwareBackend::Cpu);
        assert_eq!(report.hardware_device(), None);

        let rgb_webp = fs::read(&rgb_out)?;
        let (dec_rgb, w, h) = zenwebp::oneshot::decode_rgb(&rgb_webp)?;
        assert_eq!((w, h), (64, 48));
        assert_eq!(dec_rgb.len(), 64 * 48 * 3);

        let lossless_out = root.join("alpha_lossless.webp");
        let lossless_plan = ImageBatchPlan::new(
            vec![ImageBatchItem::new(
                alpha_path,
                lossless_out.clone(),
                ImageDimensions::new(32, 24)?,
            )?],
            ImageExecutionMode::CpuWebp(WebpEncoding::Lossless),
            NonZeroUsize::MIN,
        )?;
        let lossless_report = execute_image_batch(&lossless_plan)?;
        assert_eq!(lossless_report.files_converted().get(), 1);
        let lossless_bytes = fs::read(&lossless_out)?;
        let (dec_alpha, lw, lh) = zenwebp::oneshot::decode_rgba(&lossless_bytes)?;
        assert_eq!((lw, lh), (32, 24));
        assert_eq!(dec_alpha, alpha_pixels);

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

    #[cfg(feature = "nvenc")]
    #[test]
    fn converts_png_batch_to_valid_heic_via_nvenc_hardware()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = create_temp_dir("nvenc-heic")?;
        let src_a = root.join("a.png");
        let src_b = root.join("b.png");
        let out_a = root.join("out/a.heic");
        let out_b = root.join("out/b.heic");

        write_test_png(
            &src_a,
            256,
            256,
            png::ColorType::Rgb,
            &make_rgb_gradient(256, 256),
        )?;
        write_test_png(
            &src_b,
            256,
            256,
            png::ColorType::Rgb,
            &make_rgb_gradient(256, 256),
        )?;

        let plan = ImageBatchPlan::new(
            vec![
                ImageBatchItem::new(src_a, out_a.clone(), ImageDimensions::new(256, 256)?)?,
                ImageBatchItem::new(src_b, out_b.clone(), ImageDimensions::new(256, 256)?)?,
            ],
            ImageExecutionMode::NvencHeic {
                quality: ImageQuality::DEFAULT,
            },
            NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN),
        )?;

        let report = execute_image_batch(&plan)?;
        assert_eq!(report.files_converted().get(), 2);
        assert_eq!(report.backend(), HardwareBackend::Gpu(GpuBackend::Nvenc));
        assert!(report.hardware_device().is_some());

        let bytes_a = fs::read(&out_a)?;
        assert_eq!(bytes_a.get(4..12), Some(b"ftypheic".as_slice()));
        let bytes_b = fs::read(&out_b)?;
        assert_eq!(bytes_b.get(4..12), Some(b"ftypheic".as_slice()));

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
