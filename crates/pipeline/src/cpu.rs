use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, Read};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;

use fastconv_domain::{
    ExecutionPlan, HardwareBackend, ImageBatchPlan, ImageConversionReport, PngToWebpItem,
    WebpEncoding,
};
use zenwebp::{EncodeRequest, LosslessConfig, LossyConfig, PixelLayout};

use crate::PipelineError;

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
pub fn execute_image_batch(plan: &ImageBatchPlan) -> Result<ImageConversionReport, PipelineError> {
    let items = plan.items();
    let Some(first_item) = items.first() else {
        return Err(PipelineError::EmptyOutput(PathBuf::new()));
    };
    precreate_target_directories(items)?;

    let worker_threads = plan.effective_workers();
    let files_converted = NonZeroUsize::new(items.len())
        .ok_or_else(|| PipelineError::EmptyOutput(first_item.target().to_path_buf()))?;

    let (total_input, total_output) = if worker_threads.get() == 1 || items.len() == 1 {
        execute_sequential(items, plan.encoding())?
    } else {
        execute_parallel(items, plan.encoding(), worker_threads.get())?
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
    ))
}

fn precreate_target_directories(items: &[PngToWebpItem]) -> Result<(), PipelineError> {
    let mut parents = BTreeSet::new();
    for item in items {
        if let Some(parent) = item.target().parent()
            && !parent.as_os_str().is_empty()
        {
            parents.insert(parent);
        }
    }
    for dir in parents {
        fs::create_dir_all(dir).map_err(|source| PipelineError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}

fn execute_sequential(
    items: &[PngToWebpItem],
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
    items: &[PngToWebpItem],
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
    item: &PngToWebpItem,
    encoding: WebpEncoding,
    scratch: &mut WorkerScratch,
) -> Result<(u64, u64), PipelineError> {
    scratch.file_bytes.clear();
    let mut file = File::open(item.source()).map_err(|source| PipelineError::Io {
        path: item.source().to_path_buf(),
        source,
    })?;
    file.read_to_end(&mut scratch.file_bytes)
        .map_err(|source| PipelineError::Io {
            path: item.source().to_path_buf(),
            source,
        })?;
    let input_bytes = u64::try_from(scratch.file_bytes.len()).unwrap_or(u64::MAX);

    let mut decoder = png::Decoder::new(io::Cursor::new(scratch.file_bytes.as_slice()));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    decoder.set_ignore_text_chunk(true);

    let mut reader = decoder
        .read_info()
        .map_err(|err| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: err.to_string(),
        })?;

    let required_len = reader
        .output_buffer_size()
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: "decoded PNG buffer size exceeds addressable memory".to_owned(),
        })?;
    if scratch.pixel_buf.len() < required_len {
        scratch.pixel_buf.resize(required_len, 0);
    }

    let frame =
        reader
            .next_frame(&mut scratch.pixel_buf)
            .map_err(|err| PipelineError::DecodeFailed {
                path: item.source().to_path_buf(),
                reason: err.to_string(),
            })?;

    let expected_dims = item.dimensions();
    if frame.width != expected_dims.width().get() || frame.height != expected_dims.height().get() {
        return Err(PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: format!(
                "decoded PNG dimensions {}x{} do not match probed {}",
                frame.width, frame.height, expected_dims
            ),
        });
    }

    let layout = match (frame.color_type, frame.bit_depth) {
        (png::ColorType::Grayscale, png::BitDepth::Eight) => PixelLayout::L8,
        (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight) => PixelLayout::La8,
        (png::ColorType::Rgb, png::BitDepth::Eight) => PixelLayout::Rgb8,
        (png::ColorType::Rgba, png::BitDepth::Eight) => PixelLayout::Rgba8,
        (color, depth) => {
            return Err(PipelineError::UnsupportedPixelFormat {
                path: item.source().to_path_buf(),
                format: format!("{color:?} {depth:?}"),
            });
        }
    };

    let pixels = scratch
        .pixel_buf
        .get(..frame.buffer_size())
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: "decoded PNG frame exceeded allocated pixel buffer".to_owned(),
        })?;

    scratch.webp_buf.clear();
    match encoding {
        WebpEncoding::Lossy { quality } => {
            let config = LossyConfig::new()
                .with_quality(quality.as_f32())
                .with_method(4);
            EncodeRequest::lossy(&config, pixels, layout, frame.width, frame.height)
                .encode_into(&mut scratch.webp_buf)
        }
        WebpEncoding::Lossless => {
            let config = LosslessConfig::new();
            EncodeRequest::lossless(&config, pixels, layout, frame.width, frame.height)
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
    Ok((input_bytes, output_bytes))
}
