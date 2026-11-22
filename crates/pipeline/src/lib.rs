pub mod cpu;
pub mod gpu;

use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;

use fastconv_domain::{ExecutionPlan, HardwareBackend, ImageBatchPlan, ImageConversionReport};

/// Errors raised while executing a concrete media conversion pipeline.
#[derive(Debug)]
pub enum PipelineError {
    BackendNotImplemented(HardwareBackend),
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
            Self::Io { path, source } => {
                write!(f, "I/O error at '{}': {source}", path.display())
            }
            Self::DecodeFailed { path, reason } => {
                write!(f, "failed to decode PNG '{}': {reason}", path.display())
            }
            Self::EncodeFailed { path, reason } => {
                write!(f, "failed to encode WebP '{}': {reason}", path.display())
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

/// Executes a validated PNG-to-WebP batch plan on the native multi-threaded CPU pipeline.
///
/// # Errors
///
/// Returns [`PipelineError`] if reading, decoding, encoding, or writing any image fails.
pub fn execute_image_batch(plan: &ImageBatchPlan) -> Result<ImageConversionReport, PipelineError> {
    cpu::execute_image_batch(plan)
}

#[cfg(test)]
mod tests {
    use super::{PipelineError, execute, execute_image_batch};
    use fastconv_domain::{
        ConversionRequest, DomainError, ExecutionPlan, GpuBackend, HardwareBackend, ImageBatchPlan,
        ImageDimensions, MediaKind, PngToWebpItem, WebpEncoding, WebpQuality,
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

    fn domain_err<T>(res: Result<T, DomainError>) -> Result<T, PipelineError> {
        res.map_err(|err| PipelineError::EncodeFailed {
            path: PathBuf::new(),
            reason: err.to_string(),
        })
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

    fn sample_rgb_16x12() -> Vec<u8> {
        let mut rgb = Vec::with_capacity(16 * 12 * 3);
        for y in 0_u8..12 {
            for x in 0_u8..16 {
                let r = x.saturating_mul(16);
                let g = y.saturating_mul(20);
                rgb.extend_from_slice(&[r, g, r ^ g]);
            }
        }
        rgb
    }

    #[test]
    fn converts_rgb_rgba_gray_lossy_and_lossless() -> Result<(), PipelineError> {
        let root = create_temp_dir("formats")?;
        let dims = domain_err(ImageDimensions::new(16, 12))?;
        let rgb = sample_rgb_16x12();

        let mut alpha_pixels = Vec::with_capacity(16 * 12 * 4);
        let mut gray = Vec::with_capacity(16 * 12);
        for y in 0_u8..12 {
            for x in 0_u8..16 {
                let r = x.saturating_mul(16);
                let g = y.saturating_mul(20);
                alpha_pixels.extend_from_slice(&[r, g, r ^ g, 200]);
                gray.push(r.wrapping_add(g));
            }
        }

        let rgb_png = root.join("rgb.png");
        let alpha_png = root.join("rgba.png");
        let gray_png = root.join("gray.png");
        write_test_png(&rgb_png, 16, 12, png::ColorType::Rgb, &rgb)?;
        write_test_png(&alpha_png, 16, 12, png::ColorType::Rgba, &alpha_pixels)?;
        write_test_png(&gray_png, 16, 12, png::ColorType::Grayscale, &gray)?;

        let workers_4 = NonZeroUsize::new(4).unwrap_or(NonZeroUsize::MIN);
        let lossy_plan = domain_err(ImageBatchPlan::new(
            vec![
                domain_err(PngToWebpItem::new(
                    rgb_png.clone(),
                    root.join("out/rgb.webp"),
                    dims,
                ))?,
                domain_err(PngToWebpItem::new(
                    alpha_png,
                    root.join("out/rgba.webp"),
                    dims,
                ))?,
                domain_err(PngToWebpItem::new(
                    gray_png,
                    root.join("out/gray.webp"),
                    dims,
                ))?,
            ],
            WebpEncoding::Lossy {
                quality: WebpQuality::DEFAULT,
            },
            workers_4,
        ))?;
        let lossy_report = execute_image_batch(&lossy_plan)?;
        assert_eq!(lossy_report.files_converted().get(), 3);
        assert_eq!(lossy_report.worker_threads_used().get(), 3);

        for name in ["out/rgb.webp", "out/rgba.webp", "out/gray.webp"] {
            let webp_path = root.join(name);
            let bytes = fs::read(&webp_path).map_err(|source| PipelineError::Io {
                path: webp_path.clone(),
                source,
            })?;
            let (_, w, h) = zenwebp::oneshot::decode_rgb(&bytes).map_err(|err| {
                PipelineError::DecodeFailed {
                    path: webp_path,
                    reason: err.to_string(),
                }
            })?;
            assert_eq!((w, h), (16, 12));
        }

        let lossless_webp = root.join("lossless/rgb.webp");
        let lossless_plan = domain_err(ImageBatchPlan::new(
            vec![domain_err(PngToWebpItem::new(
                rgb_png,
                lossless_webp.clone(),
                dims,
            ))?],
            WebpEncoding::Lossless,
            NonZeroUsize::MIN,
        ))?;
        execute_image_batch(&lossless_plan)?;
        let lossless_bytes = fs::read(&lossless_webp).map_err(|source| PipelineError::Io {
            path: lossless_webp.clone(),
            source,
        })?;
        let (decoded_pixels, lw, lh) =
            zenwebp::oneshot::decode_rgb(&lossless_bytes).map_err(|err| {
                PipelineError::DecodeFailed {
                    path: lossless_webp,
                    reason: err.to_string(),
                }
            })?;
        assert_eq!((lw, lh), (16, 12));
        assert_eq!(decoded_pixels, rgb);

        fs::remove_dir_all(&root).map_err(|source| PipelineError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn converts_multi_image_batch_in_parallel() -> Result<(), PipelineError> {
        let root = create_temp_dir("parallel")?;
        let dims = domain_err(ImageDimensions::new(16, 12))?;
        let rgb = sample_rgb_16x12();
        let batch_dir = root.join("batch");
        fs::create_dir_all(&batch_dir).map_err(|source| PipelineError::Io {
            path: batch_dir.clone(),
            source,
        })?;

        let mut batch_items = Vec::with_capacity(6);
        for i in 0..6 {
            let src = batch_dir.join(format!("in_{i}.png"));
            write_test_png(&src, 16, 12, png::ColorType::Rgb, &rgb)?;
            batch_items.push(domain_err(PngToWebpItem::new(
                src,
                root.join(format!("batch_out/out_{i}.webp")),
                dims,
            ))?);
        }

        let workers_4 = NonZeroUsize::new(4).unwrap_or(NonZeroUsize::MIN);
        let batch_plan = domain_err(ImageBatchPlan::new(
            batch_items,
            WebpEncoding::Lossy {
                quality: WebpQuality::DEFAULT,
            },
            workers_4,
        ))?;
        let batch_report = execute_image_batch(&batch_plan)?;
        assert_eq!(batch_report.files_converted().get(), 6);
        assert_eq!(batch_report.worker_threads_used().get(), 4);

        for i in 0..6 {
            let out = root.join(format!("batch_out/out_{i}.webp"));
            let bytes = fs::read(&out).map_err(|source| PipelineError::Io {
                path: out.clone(),
                source,
            })?;
            let (_, w, h) = zenwebp::oneshot::decode_rgb(&bytes).map_err(|err| {
                PipelineError::DecodeFailed {
                    path: out,
                    reason: err.to_string(),
                }
            })?;
            assert_eq!((w, h), (16, 12));
        }

        fs::remove_dir_all(&root).map_err(|source| PipelineError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn reuses_worker_scratch_across_shrinking_dimensions() -> Result<(), PipelineError> {
        let root = create_temp_dir("scratch")?;
        let rgba_32x24 = vec![120_u8; 32 * 24 * 4];
        let gray_8x6 = vec![200_u8; 8 * 6];
        let rgb_16x12 = sample_rgb_16x12();

        let src_large = root.join("large.png");
        let src_small = root.join("small.png");
        let src_medium = root.join("medium.png");
        write_test_png(&src_large, 32, 24, png::ColorType::Rgba, &rgba_32x24)?;
        write_test_png(&src_small, 8, 6, png::ColorType::Grayscale, &gray_8x6)?;
        write_test_png(&src_medium, 16, 12, png::ColorType::Rgb, &rgb_16x12)?;

        let out_specs = [
            (root.join("out/large.webp"), (32_u32, 24_u32)),
            (root.join("out/small.webp"), (8_u32, 6_u32)),
            (root.join("out/medium.webp"), (16_u32, 12_u32)),
        ];
        let plan = domain_err(ImageBatchPlan::new(
            vec![
                domain_err(PngToWebpItem::new(
                    src_large,
                    out_specs[0].0.clone(),
                    domain_err(ImageDimensions::new(32, 24))?,
                ))?,
                domain_err(PngToWebpItem::new(
                    src_small,
                    out_specs[1].0.clone(),
                    domain_err(ImageDimensions::new(8, 6))?,
                ))?,
                domain_err(PngToWebpItem::new(
                    src_medium,
                    out_specs[2].0.clone(),
                    domain_err(ImageDimensions::new(16, 12))?,
                ))?,
            ],
            WebpEncoding::Lossy {
                quality: WebpQuality::DEFAULT,
            },
            NonZeroUsize::MIN,
        ))?;

        let report = execute_image_batch(&plan)?;
        assert_eq!(report.files_converted().get(), 3);
        assert_eq!(report.worker_threads_used().get(), 1);

        for (webp_path, expected_dims) in &out_specs {
            let bytes = fs::read(webp_path).map_err(|source| PipelineError::Io {
                path: webp_path.clone(),
                source,
            })?;
            let (_, w, h) = zenwebp::oneshot::decode_rgb(&bytes).map_err(|err| {
                PipelineError::DecodeFailed {
                    path: webp_path.clone(),
                    reason: err.to_string(),
                }
            })?;
            assert_eq!((w, h), *expected_dims);
        }

        fs::remove_dir_all(&root).map_err(|source| PipelineError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn rejects_corrupt_png_in_parallel_batch_and_dimension_mismatch() -> Result<(), PipelineError> {
        let root = create_temp_dir("errors")?;
        let dims = domain_err(ImageDimensions::new(16, 12))?;
        let rgb = sample_rgb_16x12();

        let mut items = Vec::with_capacity(4);
        for i in 0..4 {
            let src = root.join(format!("img_{i}.png"));
            write_test_png(&src, 16, 12, png::ColorType::Rgb, &rgb)?;
            items.push(domain_err(PngToWebpItem::new(
                src,
                root.join(format!("out/img_{i}.webp")),
                dims,
            ))?);
        }

        let corrupt_src = root.join("img_2.png");
        fs::write(&corrupt_src, b"corrupted").map_err(|source| PipelineError::Io {
            path: corrupt_src,
            source,
        })?;

        let workers_2 = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        let parallel_plan = domain_err(ImageBatchPlan::new(
            items,
            WebpEncoding::Lossy {
                quality: WebpQuality::DEFAULT,
            },
            workers_2,
        ))?;
        assert!(matches!(
            execute_image_batch(&parallel_plan),
            Err(PipelineError::DecodeFailed { .. })
        ));

        let mismatch_src = root.join("img_0.png");
        let wrong_dims = domain_err(ImageDimensions::new(32, 24))?;
        let mismatch_plan = domain_err(ImageBatchPlan::new(
            vec![domain_err(PngToWebpItem::new(
                mismatch_src,
                root.join("out/mismatch.webp"),
                wrong_dims,
            ))?],
            WebpEncoding::Lossless,
            NonZeroUsize::MIN,
        ))?;
        assert!(matches!(
            execute_image_batch(&mismatch_plan),
            Err(PipelineError::DecodeFailed { .. })
        ));

        fs::remove_dir_all(&root).map_err(|source| PipelineError::Io { path: root, source })?;
        Ok(())
    }
}
