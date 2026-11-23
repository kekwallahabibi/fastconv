use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::num::{NonZeroU64, NonZeroUsize};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;

use fastconv_domain::{
    ExecutionPlan, HardwareBackend, ImageBatchItem, ImageBatchPlan, ImageCompression,
    ImageConversionReport, ImageQuality,
};
use jixel::{
    EncodeConfig, EncodeError as JixelEncodeError, Speed, distance_from_quality, encode_image,
    encode_image_gray, encode_image_gray_alpha, encode_image_with_alpha,
};
use jpeg_encoder::{ChromaSubsamplingMethod, ColorType as JpegColorType, Encoder as JpegEncoder};
use tiff::encoder::{Compression as TiffCompression, DeflateLevel, TiffEncoder, colortype};
use zenwebp::{Encoder as WebpEncoder, EncoderConfig as WebpEncoderConfig};

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
    Jpeg {
        quality: ImageQuality,
    },
    Png,
    Bmp,
    Qoi,
    Tiff,
    Jxl(ImageCompression),
    Avif {
        quality: ImageQuality,
        threads: usize,
    },
}

static STAGING_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

struct CountingWriter<W> {
    inner: W,
    bytes_written: u64,
    first_error: Option<io::Error>,
}

impl<W> CountingWriter<W> {
    const fn new(inner: W) -> Self {
        Self {
            inner,
            bytes_written: 0,
            first_error: None,
        }
    }

    fn record_error(&mut self, err: &io::Error) {
        if self.first_error.is_none() {
            self.first_error = Some(io::Error::new(err.kind(), err.to_string()));
        }
    }

    fn take_error(&mut self) -> Option<io::Error> {
        self.first_error.take()
    }
}

impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if let Some(ref err) = self.first_error {
            return Err(io::Error::new(err.kind(), err.to_string()));
        }
        match self.inner.write(buf) {
            Ok(0) if !buf.is_empty() => {
                let err = io::Error::new(io::ErrorKind::WriteZero, "failed to write whole buffer");
                self.record_error(&err);
                Err(err)
            }
            Ok(written) => {
                self.bytes_written = self
                    .bytes_written
                    .saturating_add(u64::try_from(written).unwrap_or(u64::MAX));
                Ok(written)
            }
            Err(err) if err.kind() == io::ErrorKind::Interrupted => Err(err),
            Err(err) => {
                self.record_error(&err);
                Err(err)
            }
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(ref err) = self.first_error {
            return Err(io::Error::new(err.kind(), err.to_string()));
        }
        match self.inner.flush() {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == io::ErrorKind::Interrupted => Err(err),
            Err(err) => {
                self.record_error(&err);
                Err(err)
            }
        }
    }
}

struct StagingFile {
    path: PathBuf,
    committed: bool,
}

impl StagingFile {
    fn create_for(target: &Path) -> Result<(Self, File), PipelineError> {
        let parent = target
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let pid = process::id();
        for _ in 0..64 {
            let seq = STAGING_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
            let tmp_path = parent.join(format!(".fastconv-{pid}-{seq}.tmp"));
            let mut open_opts = OpenOptions::new();
            open_opts.write(true).create_new(true);
            // Create staging files owner-only regardless of process umask so image data is never
            // exposed during staging or after replacing a restrictive destination file.
            #[cfg(unix)]
            open_opts.mode(0o600);
            match open_opts.open(&tmp_path) {
                Ok(file) => {
                    return Ok((
                        Self {
                            path: tmp_path,
                            committed: false,
                        },
                        file,
                    ));
                }
                Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {}
                Err(source) => {
                    return Err(PipelineError::Io {
                        path: tmp_path,
                        source,
                    });
                }
            }
        }
        Err(PipelineError::Io {
            path: target.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                "failed to create unique temporary staging file",
            ),
        })
    }

    fn commit(mut self, target: &Path, file: File) -> Result<(), PipelineError> {
        preserve_existing_permissions(&file, target)?;
        drop(file);
        fs::rename(&self.path, target).map_err(|source| PipelineError::Io {
            path: target.to_path_buf(),
            source,
        })?;
        self.committed = true;
        Ok(())
    }
}

fn preserve_existing_permissions(file: &File, target: &Path) -> Result<(), PipelineError> {
    match fs::symlink_metadata(target) {
        Ok(meta) if meta.file_type().is_file() => {
            file.set_permissions(meta.permissions())
                .map_err(|source| PipelineError::Io {
                    path: target.to_path_buf(),
                    source,
                })
        }
        Ok(_) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(PipelineError::Io {
            path: target.to_path_buf(),
            source,
        }),
    }
}

impl Drop for StagingFile {
    fn drop(&mut self) {
        if !self.committed && fs::remove_file(&self.path).is_err() {
            // Best-effort cleanup of the staging file when encoding or rename failed.
        }
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

    // Heavy 4K encoders can otherwise allocate several gigabytes of working
    // state across all worker threads. Keep lightweight formats on the full
    // worker set, but cap JXL/AVIF concurrency to four images.
    let requested_workers = plan.effective_workers().get();
    let worker_count = match encoder {
        CpuImageEncoder::Jxl(_) => requested_workers.clamp(1, 4),
        CpuImageEncoder::Avif { .. } => requested_workers.clamp(1, 3),
        _ => requested_workers,
    };
    let worker_threads = NonZeroUsize::new(worker_count).unwrap_or(NonZeroUsize::MIN);
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
        CpuImageEncoder::Tiff => {
            scratch.encoded_buf.clear();
            encode_tiff_into(item.target(), pixels, decoded, &mut scratch.encoded_buf)?;
            write_encoded_buffer(item.target(), &scratch.encoded_buf)?
        }
        CpuImageEncoder::Jxl(compression) => {
            scratch.encoded_buf.clear();
            encode_jxl_into(
                item.target(),
                compression,
                pixels,
                decoded,
                &mut scratch.encoded_buf,
            )?;
            write_encoded_buffer(item.target(), &scratch.encoded_buf)?
        }
        CpuImageEncoder::Avif { quality, threads } => {
            scratch.encoded_buf.clear();
            encode_avif_into(
                item.target(),
                quality,
                threads,
                pixels,
                decoded,
                &mut scratch.encoded_buf,
            )?;
            write_encoded_buffer(item.target(), &scratch.encoded_buf)?
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

fn encode_png_to_writer<W: Write>(
    target: &Path,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    sink: W,
) -> Result<u64, PipelineError> {
    let mut writer = CountingWriter::new(sink);

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
            // In `png` 0.18 `Fastest`, `StreamWriter::finish()` drops the final `ChunkWriter`
            // whose `Drop` ignores flush errors; `CountingWriter` latches any sink error so it
            // cannot be lost across finalization.
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

    if let Some(source) = writer.take_error() {
        return Err(PipelineError::Io {
            path: target.to_path_buf(),
            source,
        });
    }
    write_res?;

    if writer.bytes_written == 0 {
        return Err(PipelineError::EmptyOutput(target.to_path_buf()));
    }
    Ok(writer.bytes_written)
}

fn encode_png_streaming(
    target: &Path,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
) -> Result<u64, PipelineError> {
    let (staging, file) = StagingFile::create_for(target)?;
    let mut buf_writer = BufWriter::with_capacity(STREAM_IO_BUF_BYTES, file);
    let bytes = encode_png_to_writer(target, pixels, decoded, &mut buf_writer)?;
    let file = buf_writer.into_inner().map_err(|err| PipelineError::Io {
        path: target.to_path_buf(),
        source: err.into_error(),
    })?;
    staging.commit(target, file)?;
    Ok(bytes)
}

fn encode_bmp_streaming(
    target: &Path,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    batch_scratch: &mut Vec<u8>,
) -> Result<u64, PipelineError> {
    let (staging, mut file) = StagingFile::create_for(target)?;
    let bytes = encode_bmp_to_writer(target, pixels, decoded, batch_scratch, &mut file)?;
    if bytes == 0 {
        drop(file);
        return Err(PipelineError::EmptyOutput(target.to_path_buf()));
    }
    file.flush().map_err(|source| PipelineError::Io {
        path: target.to_path_buf(),
        source,
    })?;
    staging.commit(target, file)?;
    Ok(bytes)
}

fn encode_webp_into(
    target: &Path,
    compression: ImageCompression,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    encoded_buf: &mut Vec<u8>,
) -> Result<(), PipelineError> {
    let config = match compression {
        ImageCompression::Lossy { quality } => {
            WebpEncoderConfig::new().quality(quality.as_f32()).method(4)
        }
        ImageCompression::Lossless => WebpEncoderConfig::new_lossless(),
    };

    let encode_res = match decoded.format {
        DecodedPixelFormat::L8 => WebpEncoder::new_l8(pixels, decoded.width, decoded.height)
            .config(config)
            .encode_into(encoded_buf),
        DecodedPixelFormat::La8 => WebpEncoder::new_la8(pixels, decoded.width, decoded.height)
            .config(config)
            .encode_into(encoded_buf),
        DecodedPixelFormat::Rgb8 => WebpEncoder::new_rgb(pixels, decoded.width, decoded.height)
            .config(config)
            .encode_into(encoded_buf),
        DecodedPixelFormat::Rgba8 => WebpEncoder::new_rgba(pixels, decoded.width, decoded.height)
            .config(config)
            .encode_into(encoded_buf),
    };

    encode_res.map_err(|err| PipelineError::EncodeFailed {
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

fn encode_tiff_into(
    target: &Path,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    encoded_buf: &mut Vec<u8>,
) -> Result<(), PipelineError> {
    let width = decoded.width;
    let height = decoded.height;

    let mut writer = io::Cursor::new(encoded_buf);
    let mut encoder = TiffEncoder::new(&mut writer).map_err(|err| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: format!("failed to create TIFF encoder: {err}"),
    })?;
    encoder = encoder.with_compression(TiffCompression::Deflate(DeflateLevel::Fast));

    match decoded.format {
        DecodedPixelFormat::L8 => encoder
            .write_image::<colortype::Gray8>(width, height, pixels)
            .map_err(|err| PipelineError::EncodeFailed {
                path: target.to_path_buf(),
                reason: format!("failed to encode TIFF grayscale image: {err}"),
            })?,
        DecodedPixelFormat::La8 => {
            let (pairs, remainder) = pixels.as_chunks::<2>();
            if !remainder.is_empty() {
                return Err(PipelineError::DecodeFailed {
                    path: target.to_path_buf(),
                    reason: "unaligned LA8 pixel buffer".to_owned(),
                });
            }
            let mut rgba = Vec::with_capacity(pairs.len() * 4);
            for pair in pairs {
                let value = pair[0];
                rgba.extend_from_slice(&[value, value, value, pair[1]]);
            }
            encoder
                .write_image::<colortype::RGBA8>(width, height, &rgba)
                .map_err(|err| PipelineError::EncodeFailed {
                    path: target.to_path_buf(),
                    reason: format!("failed to encode TIFF gray+alpha image: {err}"),
                })?;
        }
        DecodedPixelFormat::Rgb8 => encoder
            .write_image::<colortype::RGB8>(width, height, pixels)
            .map_err(|err| PipelineError::EncodeFailed {
                path: target.to_path_buf(),
                reason: format!("failed to encode TIFF RGB image: {err}"),
            })?,
        DecodedPixelFormat::Rgba8 => encoder
            .write_image::<colortype::RGBA8>(width, height, pixels)
            .map_err(|err| PipelineError::EncodeFailed {
                path: target.to_path_buf(),
                reason: format!("failed to encode TIFF RGBA image: {err}"),
            })?,
    }

    Ok(())
}

fn encode_jxl_into(
    target: &Path,
    compression: ImageCompression,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    encoded_buf: &mut Vec<u8>,
) -> Result<(), PipelineError> {
    let width = decoded.width;
    let height = decoded.height;

    let mut config = EncodeConfig::default();
    config.speed = Speed::Fastest;
    config.num_threads = 1;

    match compression {
        ImageCompression::Lossless => {
            config.lossless = true;
        }
        ImageCompression::Lossy { quality } => {
            if quality.get() >= 100 {
                config.lossless = true;
            } else {
                config.distance = distance_from_quality(quality.as_f32());
            }
        }
    }

    let encoded = match decoded.format {
        DecodedPixelFormat::L8 => {
            encode_image_gray(pixels, width as usize, height as usize, &config)
        }
        DecodedPixelFormat::La8 => {
            encode_image_gray_alpha(pixels, width as usize, height as usize, &config)
        }
        DecodedPixelFormat::Rgb8 => encode_image(pixels, width as usize, height as usize, &config),
        DecodedPixelFormat::Rgba8 => {
            encode_image_with_alpha(pixels, width as usize, height as usize, &config)
        }
    }
    .map_err(|err: JixelEncodeError| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: format!("failed to encode JPEG XL: {err}"),
    })?;

    encoded_buf.extend_from_slice(&encoded);
    Ok(())
}

fn decoded_pixels_to_rgba(
    target: &Path,
    pixels: &[u8],
    format: DecodedPixelFormat,
    pixel_count: usize,
) -> Result<Vec<ravif::RGBA8>, PipelineError> {
    match format {
        DecodedPixelFormat::L8 => {
            if pixels.len() != pixel_count {
                return Err(PipelineError::DecodeFailed {
                    path: target.to_path_buf(),
                    reason: "decoded L8 buffer has unexpected length".to_owned(),
                });
            }
            Ok(pixels
                .iter()
                .map(|&value| ravif::RGBA8 {
                    r: value,
                    g: value,
                    b: value,
                    a: 255,
                })
                .collect())
        }
        DecodedPixelFormat::La8 => {
            if pixels.len() != pixel_count * 2 {
                return Err(PipelineError::DecodeFailed {
                    path: target.to_path_buf(),
                    reason: "decoded LA8 buffer has unexpected length".to_owned(),
                });
            }
            let (pairs, _) = pixels.as_chunks::<2>();
            Ok(pairs
                .iter()
                .map(|pair| ravif::RGBA8 {
                    r: pair[0],
                    g: pair[0],
                    b: pair[0],
                    a: pair[1],
                })
                .collect())
        }
        DecodedPixelFormat::Rgb8 => {
            if pixels.len() != pixel_count * 3 {
                return Err(PipelineError::DecodeFailed {
                    path: target.to_path_buf(),
                    reason: "decoded RGB8 buffer has unexpected length".to_owned(),
                });
            }
            let (triples, _) = pixels.as_chunks::<3>();
            Ok(triples
                .iter()
                .map(|triple| ravif::RGBA8 {
                    r: triple[0],
                    g: triple[1],
                    b: triple[2],
                    a: 255,
                })
                .collect())
        }
        DecodedPixelFormat::Rgba8 => {
            if pixels.len() != pixel_count * 4 {
                return Err(PipelineError::DecodeFailed {
                    path: target.to_path_buf(),
                    reason: "decoded RGBA8 buffer has unexpected length".to_owned(),
                });
            }
            let (quads, _) = pixels.as_chunks::<4>();
            Ok(quads
                .iter()
                .map(|quad| ravif::RGBA8 {
                    r: quad[0],
                    g: quad[1],
                    b: quad[2],
                    a: quad[3],
                })
                .collect())
        }
    }
}

fn encode_avif_into(
    target: &Path,
    quality: ImageQuality,
    threads: usize,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    encoded_buf: &mut Vec<u8>,
) -> Result<(), PipelineError> {
    let width = usize::try_from(decoded.width).map_err(|_| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: format!("AVIF width {} is too large", decoded.width),
    })?;
    let height = usize::try_from(decoded.height).map_err(|_| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: format!("AVIF height {} is too large", decoded.height),
    })?;

    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: "AVIF dimensions overflow pixel count".to_owned(),
        })?;

    let rgba_pixels = decoded_pixels_to_rgba(target, pixels, decoded.format, pixel_count)?;

    let quality = f32::from(quality.get().max(1));
    let encoded = ravif::Encoder::new()
        .with_quality(quality)
        .with_alpha_quality(quality)
        .with_speed(7)
        .with_num_threads(Some(threads.max(1)))
        .encode_rgba(ravif::Img::new(&rgba_pixels, width, height))
        .map_err(|err| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: format!("failed to encode AVIF: {err}"),
        })?;

    if encoded.avif_file.is_empty() {
        return Err(PipelineError::EmptyOutput(target.to_path_buf()));
    }
    encoded_buf.extend_from_slice(&encoded.avif_file);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{encode_bmp_streaming, encode_png_streaming, encode_png_to_writer};
    use crate::PipelineError;
    use crate::decode::{DecodedFrameMeta, DecodedPixelFormat};
    use std::env;
    use std::fs;
    use std::io::{self, Write};
    use std::path::Path;
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};

    static CPU_TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct FailOnceAfterIdatHeader {
        saw_idat_header: bool,
        failed_once: bool,
        written: Vec<u8>,
    }

    impl FailOnceAfterIdatHeader {
        const fn new() -> Self {
            Self {
                saw_idat_header: false,
                failed_once: false,
                written: Vec::new(),
            }
        }
    }

    impl Write for FailOnceAfterIdatHeader {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.saw_idat_header && !self.failed_once {
                self.failed_once = true;
                return Err(io::Error::other(
                    "injected transient final IDAT write error",
                ));
            }
            if buf == b"IDAT" {
                self.saw_idat_header = true;
            }
            self.written.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    struct ZeroWriteOnceAfterIdat {
        saw_idat: bool,
        returned_zero: bool,
    }

    impl Write for ZeroWriteOnceAfterIdat {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.saw_idat && !self.returned_zero && !buf.is_empty() {
                self.returned_zero = true;
                return Ok(0);
            }
            if buf == b"IDAT" {
                self.saw_idat = true;
            }
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn png_encode_propagates_transient_final_idat_write_error() {
        let pixels = vec![42_u8; 16 * 16 * 3];
        let meta = DecodedFrameMeta {
            width: 16,
            height: 16,
            format: DecodedPixelFormat::Rgb8,
            byte_len: pixels.len(),
            input_file_bytes: 1,
        };
        let mut sink = FailOnceAfterIdatHeader::new();
        let res = encode_png_to_writer(Path::new("fault.png"), &pixels, meta, &mut sink);
        assert!(sink.failed_once, "fault injector should have failed IDAT");
        assert!(matches!(res, Err(PipelineError::Io { .. })));

        // Also verify Ok(0) on a non-empty buffer during final IDAT is latched as WriteZero.
        let mut zero_sink = ZeroWriteOnceAfterIdat {
            saw_idat: false,
            returned_zero: false,
        };
        let zero_res = encode_png_to_writer(Path::new("zero.png"), &pixels, meta, &mut zero_sink);
        assert!(zero_sink.returned_zero);
        assert!(matches!(
            zero_res,
            Err(PipelineError::Io { ref source, .. }) if source.kind() == io::ErrorKind::WriteZero
        ));
    }

    #[cfg(unix)]
    #[test]
    fn staged_png_and_bmp_outputs_do_not_widen_private_permissions()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::PermissionsExt;

        let id = CPU_TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("fastconv-cpu-perms-{}-{id}", process::id()));
        fs::create_dir_all(&dir)?;

        let pixels = vec![99_u8; 8 * 8 * 3];
        let meta = DecodedFrameMeta {
            width: 8,
            height: 8,
            format: DecodedPixelFormat::Rgb8,
            byte_len: pixels.len(),
            input_file_bytes: 1,
        };

        let png_target = dir.join("private.png");
        let bmp_target = dir.join("private.bmp");
        fs::write(&png_target, b"old")?;
        fs::write(&bmp_target, b"old")?;
        fs::set_permissions(&png_target, fs::Permissions::from_mode(0o600))?;
        fs::set_permissions(&bmp_target, fs::Permissions::from_mode(0o600))?;

        encode_png_streaming(&png_target, &pixels, meta)?;
        let mut scratch = Vec::new();
        encode_bmp_streaming(&bmp_target, &pixels, meta, &mut scratch)?;

        assert_eq!(
            fs::metadata(&png_target)?.permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            fs::metadata(&bmp_target)?.permissions().mode() & 0o777,
            0o600
        );

        fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn failed_png_and_bmp_streaming_preserves_existing_destination()
    -> Result<(), Box<dyn std::error::Error>> {
        let id = CPU_TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("fastconv-cpu-atomic-{}-{id}", process::id()));
        fs::create_dir_all(&dir)?;

        let sentinel = b"existing-destination-must-survive";
        let png_target = dir.join("out.png");
        let bmp_target = dir.join("out.bmp");
        fs::write(&png_target, sentinel)?;
        fs::write(&bmp_target, sentinel)?;

        // Truncated pixel slice triggers encode error during streaming.
        let bad_pixels = vec![0_u8; 3];
        let meta = DecodedFrameMeta {
            width: 16,
            height: 16,
            format: DecodedPixelFormat::Rgb8,
            byte_len: bad_pixels.len(),
            input_file_bytes: 1,
        };
        assert!(encode_png_streaming(&png_target, &bad_pixels, meta).is_err());
        let mut scratch = Vec::new();
        assert!(encode_bmp_streaming(&bmp_target, &bad_pixels, meta, &mut scratch).is_err());

        assert_eq!(fs::read(&png_target)?, sentinel);
        assert_eq!(fs::read(&bmp_target)?, sentinel);

        let leftover_tmps: Vec<_> = fs::read_dir(&dir)?
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(
            leftover_tmps.is_empty(),
            "expected no leftover staging files: {leftover_tmps:?}"
        );

        fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
