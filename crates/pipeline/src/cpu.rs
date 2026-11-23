use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;

use fastconv_domain::{
    ExecutionPlan, HardwareBackend, ImageBatchItem, ImageBatchPlan, ImageCompression,
    ImageConversionReport, ImageQuality,
};
use jpeg_encoder::{ChromaSubsamplingMethod, ColorType as JpegColorType, Encoder as JpegEncoder};
use zenwebp::{EncodeRequest, LosslessConfig, LossyConfig, PixelLayout};

use crate::PipelineError;
use crate::bmp::encode_bmp_to_writer;
use crate::decode::{
    DecodeTargetHint, DecodedFrameMeta, DecodedPixelFormat, decode_image_into,
    precreate_target_directories,
};
use crate::qoi::encode_qoi_into;

const STREAM_IO_BUF_BYTES: usize = 512 * 1024;
const PNG_IDAT_CHUNK_BYTES: usize = 256 * 1024;

/// Concrete CPU execution backend entry point.
///
/// # Errors
///
/// Returns [`PipelineError::BackendNotImplemented`] until video/audio CPU transcoding
/// is integrated.
pub const fn execute(_plan: &ExecutionPlan) -> Result<(), PipelineError> {
    Err(PipelineError::BackendNotImplemented(HardwareBackend::Cpu))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CpuImageEncoder {
    Webp(ImageCompression),
    Jpeg { quality: ImageQuality },
    Png,
    Bmp,
    Qoi,
}

struct CountingWriter<W> {
    inner: W,
    bytes_written: u64,
}

impl<W> CountingWriter<W> {
    const fn new(inner: W) -> Self {
        Self {
            inner,
            bytes_written: 0,
        }
    }
}

impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(buf)?;
        self.bytes_written = self
            .bytes_written
            .saturating_add(u64::try_from(written).unwrap_or(u64::MAX));
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn cleanup_partial_output(path: &Path) {
    if fs::remove_file(path).is_err() {
        // Best-effort cleanup when a streaming encode already failed.
    }
}

struct WorkerScratch {
    file_bytes: Vec<u8>,
    pixel_buf: Vec<u8>,
    encoded_buf: Vec<u8>,
    aux_buf: Vec<u8>,
}

impl WorkerScratch {
    const fn new() -> Self {
        Self {
            file_bytes: Vec::new(),
            pixel_buf: Vec::new(),
            encoded_buf: Vec::new(),
            aux_buf: Vec::new(),
        }
    }
}

/// Converts every validated PNG/JPEG item in `plan` into the selected CPU output format in
/// parallel across CPU workers using reusable per-thread scratch buffers.
///
/// # Errors
///
/// Returns [`PipelineError`] if creating output directories, decoding inputs, encoding outputs,
/// or writing files fails.
pub(crate) fn execute_cpu_image_batch(
    plan: &ImageBatchPlan,
    encoder: CpuImageEncoder,
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
        execute_sequential(items, encoder)?
    } else {
        execute_parallel(items, encoder, worker_threads.get())?
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
    encoder: CpuImageEncoder,
) -> Result<(u64, u64), PipelineError> {
    let mut scratch = WorkerScratch::new();
    let mut total_in = 0_u64;
    let mut total_out = 0_u64;

    for item in items {
        let (in_bytes, out_bytes) = convert_single_item(item, encoder, &mut scratch)?;
        total_in = total_in.saturating_add(in_bytes);
        total_out = total_out.saturating_add(out_bytes);
    }

    Ok((total_in, total_out))
}

fn execute_parallel(
    items: &[ImageBatchItem],
    encoder: CpuImageEncoder,
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
                    match convert_single_item(item, encoder, &mut scratch) {
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
    encoder: CpuImageEncoder,
    scratch: &mut WorkerScratch,
) -> Result<(u64, u64), PipelineError> {
    let decoded = decode_image_into(
        item,
        DecodeTargetHint::Native,
        &mut scratch.file_bytes,
        &mut scratch.pixel_buf,
    )?;
    let pixels =
        scratch
            .pixel_buf
            .get(..decoded.byte_len)
            .ok_or_else(|| PipelineError::DecodeFailed {
                path: item.source().to_path_buf(),
                reason: "decoded image frame exceeded allocated pixel buffer".to_owned(),
            })?;

    let output_bytes = match encoder {
        CpuImageEncoder::Webp(compression) => {
            scratch.encoded_buf.clear();
            encode_webp_into(
                item.target(),
                compression,
                pixels,
                decoded,
                &mut scratch.encoded_buf,
            )?;
            write_encoded_buffer(item.target(), &scratch.encoded_buf)?
        }
        CpuImageEncoder::Jpeg { quality } => {
            scratch.encoded_buf.clear();
            encode_jpeg_into(
                item,
                quality,
                pixels,
                decoded,
                &mut scratch.aux_buf,
                &mut scratch.encoded_buf,
            )?;
            write_encoded_buffer(item.target(), &scratch.encoded_buf)?
        }
        CpuImageEncoder::Qoi => {
            scratch.encoded_buf.clear();
            encode_qoi_into(item.target(), pixels, decoded, &mut scratch.encoded_buf)?;
            write_encoded_buffer(item.target(), &scratch.encoded_buf)?
        }
        CpuImageEncoder::Png => encode_png_streaming(item.target(), pixels, decoded)?,
        CpuImageEncoder::Bmp => {
            encode_bmp_streaming(item.target(), pixels, decoded, &mut scratch.aux_buf)?
        }
    };

    Ok((decoded.input_file_bytes, output_bytes))
}

fn write_encoded_buffer(target: &Path, encoded_buf: &[u8]) -> Result<u64, PipelineError> {
    if encoded_buf.is_empty() {
        return Err(PipelineError::EmptyOutput(target.to_path_buf()));
    }
    fs::write(target, encoded_buf).map_err(|source| PipelineError::Io {
        path: target.to_path_buf(),
        source,
    })?;
    Ok(u64::try_from(encoded_buf.len()).unwrap_or(u64::MAX))
}

fn encode_png_streaming(
    target: &Path,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
) -> Result<u64, PipelineError> {
    let file = File::create(target).map_err(|source| PipelineError::Io {
        path: target.to_path_buf(),
        source,
    })?;
    let mut writer = CountingWriter::new(BufWriter::with_capacity(STREAM_IO_BUF_BYTES, file));

    let write_res = (|| -> Result<(), PipelineError> {
        let mut encoder = png::Encoder::new(&mut writer, decoded.width, decoded.height);
        encoder.set_color(match decoded.format {
            DecodedPixelFormat::L8 => png::ColorType::Grayscale,
            DecodedPixelFormat::La8 => png::ColorType::GrayscaleAlpha,
            DecodedPixelFormat::Rgb8 => png::ColorType::Rgb,
            DecodedPixelFormat::Rgba8 => png::ColorType::Rgba,
        });
        encoder.set_depth(png::BitDepth::Eight);
        // `Compression::Fastest` selects `fdeflate` + `Filter::Up` for high CPU throughput
        // while keeping PNG file sizes well compressed and avoiding full-frame zlib buffering.
        encoder.set_compression(png::Compression::Fastest);

        let mut png_writer = encoder
            .write_header()
            .map_err(|err| PipelineError::EncodeFailed {
                path: target.to_path_buf(),
                reason: err.to_string(),
            })?;
        {
            let mut stream = png_writer
                .stream_writer_with_size(PNG_IDAT_CHUNK_BYTES)
                .map_err(|err| PipelineError::EncodeFailed {
                    path: target.to_path_buf(),
                    reason: err.to_string(),
                })?;
            stream
                .write_all(pixels)
                .map_err(|source| PipelineError::Io {
                    path: target.to_path_buf(),
                    source,
                })?;
            stream.finish().map_err(|err| PipelineError::EncodeFailed {
                path: target.to_path_buf(),
                reason: err.to_string(),
            })?;
        }
        png_writer
            .finish()
            .map_err(|err| PipelineError::EncodeFailed {
                path: target.to_path_buf(),
                reason: err.to_string(),
            })?;
        writer.flush().map_err(|source| PipelineError::Io {
            path: target.to_path_buf(),
            source,
        })
    })();

    if let Err(err) = write_res {
        drop(writer);
        cleanup_partial_output(target);
        return Err(err);
    }

    if writer.bytes_written == 0 {
        return Err(PipelineError::EmptyOutput(target.to_path_buf()));
    }
    Ok(writer.bytes_written)
}

fn encode_bmp_streaming(
    target: &Path,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    batch_scratch: &mut Vec<u8>,
) -> Result<u64, PipelineError> {
    let mut file = File::create(target).map_err(|source| PipelineError::Io {
        path: target.to_path_buf(),
        source,
    })?;
    let write_res = encode_bmp_to_writer(target, pixels, decoded, batch_scratch, &mut file);

    match write_res {
        Ok(0) => Err(PipelineError::EmptyOutput(target.to_path_buf())),
        Ok(bytes) => Ok(bytes),
        Err(err) => {
            drop(file);
            cleanup_partial_output(target);
            Err(err)
        }
    }
}

fn encode_webp_into(
    target: &Path,
    compression: ImageCompression,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    encoded_buf: &mut Vec<u8>,
) -> Result<(), PipelineError> {
    let layout = match decoded.format {
        DecodedPixelFormat::L8 => PixelLayout::L8,
        DecodedPixelFormat::La8 => PixelLayout::La8,
        DecodedPixelFormat::Rgb8 => PixelLayout::Rgb8,
        DecodedPixelFormat::Rgba8 => PixelLayout::Rgba8,
    };

    match compression {
        ImageCompression::Lossy { quality } => {
            let config = LossyConfig::new()
                .with_quality(quality.as_f32())
                .with_method(4);
            EncodeRequest::lossy(&config, pixels, layout, decoded.width, decoded.height)
                .encode_into(encoded_buf)
        }
        ImageCompression::Lossless => {
            let config = LosslessConfig::new();
            EncodeRequest::lossless(&config, pixels, layout, decoded.width, decoded.height)
                .encode_into(encoded_buf)
        }
    }
    .map_err(|err| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: err.to_string(),
    })
}

fn encode_jpeg_into(
    item: &ImageBatchItem,
    quality: ImageQuality,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    aux_buf: &mut Vec<u8>,
    encoded_buf: &mut Vec<u8>,
) -> Result<(), PipelineError> {
    let target = item.target();
    let width_u16 = u16::try_from(decoded.width).map_err(|_| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: format!(
            "JPEG dimensions exceed 65535 limit: {}x{}",
            decoded.width, decoded.height
        ),
    })?;
    let height_u16 = u16::try_from(decoded.height).map_err(|_| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: format!(
            "JPEG dimensions exceed 65535 limit: {}x{}",
            decoded.width, decoded.height
        ),
    })?;

    let (input_slice, color_type) = match decoded.format {
        DecodedPixelFormat::L8 => (pixels, JpegColorType::Luma),
        DecodedPixelFormat::Rgb8 => (pixels, JpegColorType::Rgb),
        DecodedPixelFormat::Rgba8 => {
            let (quads, remainder) = pixels.as_chunks::<4>();
            if !remainder.is_empty() {
                return Err(PipelineError::DecodeFailed {
                    path: item.source().to_path_buf(),
                    reason: format!("unaligned RGBA8 pixel buffer length {}", pixels.len()),
                });
            }
            if quads.iter().any(|&[_, _, _, a]| a < 255) {
                return Err(PipelineError::UnsupportedPixelFormat {
                    path: item.source().to_path_buf(),
                    format: "non-opaque alpha channel (JPEG does not support transparency)"
                        .to_owned(),
                });
            }
            (pixels, JpegColorType::Rgba)
        }
        DecodedPixelFormat::La8 => {
            let (pairs, remainder) = pixels.as_chunks::<2>();
            if !remainder.is_empty() {
                return Err(PipelineError::DecodeFailed {
                    path: item.source().to_path_buf(),
                    reason: format!("unaligned LA8 pixel buffer length {}", pixels.len()),
                });
            }
            aux_buf.clear();
            aux_buf.reserve(pairs.len());
            for &[v, a] in pairs {
                if a < 255 {
                    return Err(PipelineError::UnsupportedPixelFormat {
                        path: item.source().to_path_buf(),
                        format: "non-opaque alpha channel (JPEG does not support transparency)"
                            .to_owned(),
                    });
                }
                aux_buf.push(v);
            }
            (aux_buf.as_slice(), JpegColorType::Luma)
        }
    };

    let mut encoder = JpegEncoder::new(encoded_buf, quality.get().max(1));
    encoder.set_chroma_subsampling_method(ChromaSubsamplingMethod::Nearest);
    encoder
        .encode(input_slice, width_u16, height_u16, color_type)
        .map_err(|err| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: err.to_string(),
        })
}
