use std::fs;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;

use fastconv_domain::{
    ExecutionPlan, HardwareBackend, ImageBatchItem, ImageBatchPlan, ImageConversionReport,
    WebpEncoding,
};
use zenwebp::{EncodeRequest, LosslessConfig, LossyConfig, PixelLayout};

use crate::PipelineError;
use crate::png_decode::{DecodedPixelFormat, decode_png_into, precreate_target_directories};

/// Concrete CPU execution backend entry point.
///
/// # Errors
///
/// Returns [`PipelineError::BackendNotImplemented`] until video/audio CPU transcoding
/// is integrated.
pub const fn execute(_plan: &ExecutionPlan) -> Result<(), PipelineError> {
    Err(PipelineError::BackendNotImplemented(HardwareBackend::Cpu))
}

struct WorkerScratch {
    file_bytes: Vec<u8>,
    pixel_buf: Vec<u8>,
    webp_buf: Vec<u8>,
}

impl WorkerScratch {
    const fn new() -> Self {
        Self {
            file_bytes: Vec::new(),
            pixel_buf: Vec::new(),
            webp_buf: Vec::new(),
        }
    }
}

/// Converts every validated PNG item in `plan` into WebP in parallel across CPU workers.
///
/// # Errors
///
/// Returns [`PipelineError`] if creating output directories, decoding PNGs, encoding WebP,
/// or writing outputs fails.
pub fn execute_webp_batch(
    plan: &ImageBatchPlan,
    encoding: WebpEncoding,
) -> Result<ImageConversionReport, PipelineError> {
    let items = plan.items();
    let Some(first_item) = items.first() else {
        return Err(PipelineError::EmptyOutput(PathBuf::new()));
    };
    precreate_target_directories(items)?;

    let worker_threads = plan.effective_workers();
    let files_converted = NonZeroUsize::new(items.len())
        .ok_or_else(|| PipelineError::EmptyOutput(first_item.target().to_path_buf()))?;

    let (total_input, total_output) = if worker_threads.get() == 1 || items.len() == 1 {
        execute_sequential(items, encoding)?
    } else {
        execute_parallel(items, encoding, worker_threads.get())?
    };

    let input_nz = NonZeroU64::new(total_input)
        .ok_or_else(|| PipelineError::EmptyOutput(first_item.source().to_path_buf()))?;
    let output_nz = NonZeroU64::new(total_output)
        .ok_or_else(|| PipelineError::EmptyOutput(first_item.target().to_path_buf()))?;

    Ok(ImageConversionReport::new(
        files_converted,
        input_nz,
        output_nz,
        worker_threads,
        HardwareBackend::Cpu,
        None,
    ))
}

fn execute_sequential(
    items: &[ImageBatchItem],
    encoding: WebpEncoding,
) -> Result<(u64, u64), PipelineError> {
    let mut scratch = WorkerScratch::new();
    let mut total_in = 0_u64;
    let mut total_out = 0_u64;

    for item in items {
        let (in_bytes, out_bytes) = convert_single_item(item, encoding, &mut scratch)?;
        total_in = total_in.saturating_add(in_bytes);
        total_out = total_out.saturating_add(out_bytes);
    }

    Ok((total_in, total_out))
}

fn execute_parallel(
    items: &[ImageBatchItem],
    encoding: WebpEncoding,
    worker_count: usize,
) -> Result<(u64, u64), PipelineError> {
    let next_index = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let total_in = AtomicU64::new(0);
    let total_out = AtomicU64::new(0);

    let results: Vec<Result<(), PipelineError>> = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            handles.push(scope.spawn(|| -> Result<(), PipelineError> {
                let mut scratch = WorkerScratch::new();
                while !failed.load(Ordering::Relaxed) {
                    let index = next_index.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(index) else {
                        break;
                    };
                    match convert_single_item(item, encoding, &mut scratch) {
                        Ok((in_bytes, out_bytes)) => {
                            total_in.fetch_add(in_bytes, Ordering::Relaxed);
                            total_out.fetch_add(out_bytes, Ordering::Relaxed);
                        }
                        Err(err) => {
                            failed.store(true, Ordering::Relaxed);
                            return Err(err);
                        }
                    }
                }
                Ok(())
            }));
        }

        handles
            .into_iter()
            .map(|handle| handle.join().unwrap_or(Err(PipelineError::WorkerPanicked)))
            .collect()
    });

    for result in results {
        result?;
    }

    Ok((
        total_in.load(Ordering::Relaxed),
        total_out.load(Ordering::Relaxed),
    ))
}

fn convert_single_item(
    item: &ImageBatchItem,
    encoding: WebpEncoding,
    scratch: &mut WorkerScratch,
) -> Result<(u64, u64), PipelineError> {
    let decoded = decode_png_into(item, &mut scratch.file_bytes, &mut scratch.pixel_buf)?;
    let pixels =
        scratch
            .pixel_buf
            .get(..decoded.byte_len)
            .ok_or_else(|| PipelineError::DecodeFailed {
                path: item.source().to_path_buf(),
                reason: "decoded PNG frame exceeded allocated pixel buffer".to_owned(),
            })?;
    let layout = match decoded.format {
        DecodedPixelFormat::L8 => PixelLayout::L8,
        DecodedPixelFormat::La8 => PixelLayout::La8,
        DecodedPixelFormat::Rgb8 => PixelLayout::Rgb8,
        DecodedPixelFormat::Rgba8 => PixelLayout::Rgba8,
    };

    scratch.webp_buf.clear();
    match encoding {
        WebpEncoding::Lossy { quality } => {
            let config = LossyConfig::new()
                .with_quality(quality.as_f32())
                .with_method(4);
            EncodeRequest::lossy(&config, pixels, layout, decoded.width, decoded.height)
                .encode_into(&mut scratch.webp_buf)
        }
        WebpEncoding::Lossless => {
            let config = LosslessConfig::new();
            EncodeRequest::lossless(&config, pixels, layout, decoded.width, decoded.height)
                .encode_into(&mut scratch.webp_buf)
        }
    }
    .map_err(|err| PipelineError::EncodeFailed {
        path: item.target().to_path_buf(),
        reason: err.to_string(),
    })?;

    if scratch.webp_buf.is_empty() {
        return Err(PipelineError::EmptyOutput(item.target().to_path_buf()));
    }

    fs::write(item.target(), &scratch.webp_buf).map_err(|source| PipelineError::Io {
        path: item.target().to_path_buf(),
        source,
    })?;

    let output_bytes = u64::try_from(scratch.webp_buf.len()).unwrap_or(u64::MAX);
    Ok((decoded.input_file_bytes, output_bytes))
}
