use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::num::{NonZeroU64, NonZeroUsize};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Mutex;
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
use tiff::tags::{ExtraSamples, Predictor, Tag};
use zenwebp::{EncodeRequest, LosslessConfig, LossyConfig, PixelLayout};

use crate::PipelineError;
use crate::avif::encode_avif_into;
use crate::bmp::{BmpFrameScratch, encode_bmp_into_frame};
use crate::decode::{
    DecodeTargetHint, DecodedFrameMeta, DecodedPixelFormat, decode_image_into,
    precreate_target_directories,
};
use crate::gif::encode_gif_into;
use crate::ico::encode_ico_into;
use crate::pnm::encode_pnm_to_writer;
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
    Gif,
    Ico,
    Pnm,
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
    bmp_frame: BmpFrameScratch,
}

impl WorkerScratch {
    const fn new() -> Self {
        Self {
            file_bytes: Vec::new(),
            pixel_buf: Vec::new(),
            encoded_buf: Vec::new(),
            aux_buf: Vec::new(),
            bmp_frame: BmpFrameScratch::new(),
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
    // worker set, and cap JXL/AVIF image concurrency (8/6) while assigning
    // remaining logical cores to per-encoder internal threading.
    let effective_workers = plan.effective_workers().get();
    // Inner threading draws from the configured budget, not the item-capped
    // worker count, so a small batch of heavy images still uses idle cores.
    let configured_workers = plan.worker_threads().get();
    let (worker_count, inner_threads) = match encoder {
        CpuImageEncoder::Jxl(_) => {
            let count = configured_workers.min(items.len()).clamp(1, 8);
            (count, (configured_workers / count).max(1))
        }
        CpuImageEncoder::Avif { .. } => {
            // Prefer image-level parallelism up to 8 workers so rav1e avoids
            // per-image Rayon pool creation and tile-sync overhead on batches,
            // while small batches assign remaining cores to inner tile threads.
            let count = configured_workers.min(items.len()).clamp(1, 8);
            let inner = (configured_workers / count).clamp(1, 8);
            (count, inner)
        }
        _ => (effective_workers, 1),
    };
    let worker_threads = NonZeroUsize::new(worker_count).unwrap_or(NonZeroUsize::MIN);
    let files_converted = NonZeroUsize::new(items.len())
        .ok_or_else(|| PipelineError::EmptyOutput(first_item.target().to_path_buf()))?;

    // BMP frames are prepared in parallel but committed one at a time: concurrent
    // 25 MiB writes thrash the filesystem and erase the gains from parallel decoding.
    let bmp_write_lock = Mutex::new(());
    let (total_input, total_output) = if worker_threads.get() == 1 || items.len() == 1 {
        execute_sequential(items, encoder, inner_threads, &bmp_write_lock)?
    } else {
        execute_parallel(
            items,
            encoder,
            worker_threads.get(),
            inner_threads,
            &bmp_write_lock,
        )?
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
    jxl_inner_threads: usize,
    bmp_write_lock: &Mutex<()>,
) -> Result<(u64, u64), PipelineError> {
    let mut scratch = WorkerScratch::new();
    let mut total_in = 0_u64;
    let mut total_out = 0_u64;

    for item in items {
        let (in_bytes, out_bytes) = convert_single_item(
            item,
            encoder,
            &mut scratch,
            jxl_inner_threads,
            bmp_write_lock,
        )?;
        total_in = total_in.saturating_add(in_bytes);
        total_out = total_out.saturating_add(out_bytes);
    }

    Ok((total_in, total_out))
}

fn execute_parallel(
    items: &[ImageBatchItem],
    encoder: CpuImageEncoder,
    worker_count: usize,
    inner_threads: usize,
    bmp_write_lock: &Mutex<()>,
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
                    match convert_single_item(
                        item,
                        encoder,
                        &mut scratch,
                        inner_threads,
                        bmp_write_lock,
                    ) {
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
    encode_threads: usize,
    bmp_write_lock: &Mutex<()>,
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
        CpuImageEncoder::Bmp => encode_bmp_streaming(
            item.target(),
            pixels,
            decoded,
            &mut scratch.bmp_frame,
            bmp_write_lock,
        )?,
        CpuImageEncoder::Tiff => {
            encode_tiff_streaming(item.target(), pixels, decoded, &mut scratch.aux_buf)?
        }
        CpuImageEncoder::Jxl(compression) => {
            let jxl_output =
                encode_jxl_into(item.target(), compression, pixels, decoded, encode_threads)?;
            write_encoded_buffer(item.target(), &jxl_output)?
        }
        CpuImageEncoder::Avif { quality, threads } => {
            scratch.encoded_buf.clear();
            encode_avif_into(
                item.target(),
                quality,
                threads.max(encode_threads),
                pixels,
                decoded,
                &mut scratch.encoded_buf,
            )?;
            write_encoded_buffer(item.target(), &scratch.encoded_buf)?
        }
        CpuImageEncoder::Gif => {
            scratch.encoded_buf.clear();
            encode_gif_into(
                item.target(),
                pixels,
                decoded,
                &mut scratch.aux_buf,
                &mut scratch.encoded_buf,
            )?;
            write_encoded_buffer(item.target(), &scratch.encoded_buf)?
        }
        CpuImageEncoder::Ico => {
            scratch.encoded_buf.clear();
            encode_ico_into(item.target(), pixels, decoded, &mut scratch.encoded_buf)?;
            write_encoded_buffer(item.target(), &scratch.encoded_buf)?
        }
        CpuImageEncoder::Pnm => {
            encode_pnm_streaming(item.target(), pixels, decoded, &mut scratch.aux_buf)?
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

fn encode_pnm_streaming(
    target: &Path,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    aux_buf: &mut Vec<u8>,
) -> Result<u64, PipelineError> {
    let (staging, file) = StagingFile::create_for(target)?;
    let mut buf_writer = BufWriter::with_capacity(STREAM_IO_BUF_BYTES, file);
    let bytes = encode_pnm_to_writer(target, pixels, decoded, aux_buf, &mut buf_writer)?;
    let file = buf_writer.into_inner().map_err(|err| PipelineError::Io {
        path: target.to_path_buf(),
        source: err.into_error(),
    })?;
    if bytes == 0 {
        return Err(PipelineError::EmptyOutput(target.to_path_buf()));
    }
    staging.commit(target, file)?;
    Ok(bytes)
}

/// Prepares the whole BMP frame in parallel, then flushes it to disk under a
/// shared lock. Only the contiguous 25 MiB write is serialized; decode, swizzle
/// and header setup still overlap across workers.
fn encode_bmp_streaming(
    target: &Path,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    frame: &mut BmpFrameScratch,
    write_lock: &Mutex<()>,
) -> Result<u64, PipelineError> {
    let bytes = encode_bmp_into_frame(target, pixels, decoded, frame)?;
    let len =
        usize::try_from(bytes).map_err(|_| PipelineError::EmptyOutput(target.to_path_buf()))?;
    if len == 0 {
        return Err(PipelineError::EmptyOutput(target.to_path_buf()));
    }

    let guard = write_lock
        .lock()
        .map_err(|_| PipelineError::WorkerPanicked)?;
    let (staging, file) = StagingFile::create_for(target)?;
    file.set_len(bytes).map_err(|source| PipelineError::Io {
        path: target.to_path_buf(),
        source,
    })?;
    let mut file = file;
    file.write_all(&frame.bytes()[..len])
        .map_err(|source| PipelineError::Io {
            path: target.to_path_buf(),
            source,
        })?;
    file.flush().map_err(|source| PipelineError::Io {
        path: target.to_path_buf(),
        source,
    })?;
    staging.commit(target, file)?;
    drop(guard);
    Ok(bytes)
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

    // `LossyConfig::new()` defaults to method 4 — the prior 0.2.0 speed/quality balance.
    let encode_res = match compression {
        ImageCompression::Lossy { quality } => {
            let config = LossyConfig::new().with_quality(quality.as_f32());
            EncodeRequest::lossy(&config, pixels, layout, decoded.width, decoded.height)
                .encode_into(encoded_buf)
        }
        ImageCompression::Lossless => {
            let config = LosslessConfig::new();
            EncodeRequest::lossless(&config, pixels, layout, decoded.width, decoded.height)
                .encode_into(encoded_buf)
        }
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

fn expand_la8_to_rgba(
    target: &Path,
    pixels: &[u8],
    aux_buf: &mut Vec<u8>,
) -> Result<usize, PipelineError> {
    let (pairs, remainder) = pixels.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(PipelineError::DecodeFailed {
            path: target.to_path_buf(),
            reason: "invalid grayscale-alpha pixel buffer length".to_owned(),
        });
    }
    let needed = pairs
        .len()
        .checked_mul(4)
        .ok_or_else(|| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: "TIFF RGBA scratch buffer size overflow".to_owned(),
        })?;
    if aux_buf.len() < needed {
        aux_buf.resize(needed, 0);
    }
    let (quads, _) = aux_buf[..needed].as_chunks_mut::<4>();
    for (pair, out) in pairs.iter().zip(quads.iter_mut()) {
        let value = pair[0];
        out.copy_from_slice(&[value, value, value, pair[1]]);
    }
    Ok(needed)
}

fn encode_tiff_streaming(
    target: &Path,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    aux_buf: &mut Vec<u8>,
) -> Result<u64, PipelineError> {
    use std::io::Write;

    use crate::PipelineError;

    let (staging, file) = StagingFile::create_for(target)?;
    let mut writer = BufWriter::with_capacity(STREAM_IO_BUF_BYTES, file);
    let mut encoder = TiffEncoder::new(&mut writer).map_err(|err| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: format!("failed to create TIFF encoder: {err}"),
    })?;
    encoder = encoder
        .with_compression(TiffCompression::Deflate(DeflateLevel::Fast))
        .with_predictor(Predictor::Horizontal);

    match decoded.format {
        DecodedPixelFormat::L8 => encoder
            .write_image::<colortype::Gray8>(decoded.width, decoded.height, pixels)
            .map_err(|err| PipelineError::EncodeFailed {
                path: target.to_path_buf(),
                reason: format!("failed to encode TIFF grayscale image: {err}"),
            })?,
        DecodedPixelFormat::Rgb8 => encoder
            .write_image::<colortype::RGB8>(decoded.width, decoded.height, pixels)
            .map_err(|err| PipelineError::EncodeFailed {
                path: target.to_path_buf(),
                reason: format!("failed to encode TIFF RGB image: {err}"),
            })?,
        DecodedPixelFormat::La8 | DecodedPixelFormat::Rgba8 => {
            let rgba = if decoded.format == DecodedPixelFormat::La8 {
                let needed = expand_la8_to_rgba(target, pixels, aux_buf)?;
                &aux_buf[..needed]
            } else {
                pixels
            };
            // Use RGBA8 (not RGB8 + extra_samples) so tiff's horizontal predictor
            // steps by 4 samples instead of 3, while still tagging unassociated alpha.
            let mut image = encoder
                .new_image::<colortype::RGBA8>(decoded.width, decoded.height)
                .map_err(|err| PipelineError::EncodeFailed {
                    path: target.to_path_buf(),
                    reason: format!("failed to create TIFF RGBA image: {err}"),
                })?;
            image
                .encoder()
                .write_tag(
                    Tag::ExtraSamples,
                    &[ExtraSamples::UnassociatedAlpha.to_u16()][..],
                )
                .map_err(|err| PipelineError::EncodeFailed {
                    path: target.to_path_buf(),
                    reason: format!("failed to declare TIFF alpha sample: {err}"),
                })?;
            image
                .write_data(rgba)
                .map_err(|err| PipelineError::EncodeFailed {
                    path: target.to_path_buf(),
                    reason: format!("failed to encode TIFF RGBA image: {err}"),
                })?;
        }
    }
    let mut file = writer
        .into_inner()
        .map_err(|err| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: format!("failed to flush TIFF writer: {err}"),
        })?;
    file.flush().map_err(|source| PipelineError::Io {
        path: target.to_path_buf(),
        source,
    })?;
    let bytes = file
        .metadata()
        .map_err(|source| PipelineError::Io {
            path: target.to_path_buf(),
            source,
        })?
        .len();
    if bytes == 0 {
        return Err(PipelineError::EmptyOutput(target.to_path_buf()));
    }
    staging.commit(target, file)?;
    Ok(bytes)
}

fn encode_jxl_into(
    target: &Path,
    compression: ImageCompression,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    inner_threads: usize,
) -> Result<Vec<u8>, PipelineError> {
    let width = decoded.width as usize;
    let height = decoded.height as usize;
    let mut config = EncodeConfig::default();
    config.speed = Speed::Fastest;
    config.num_threads = inner_threads.max(1);

    match compression {
        ImageCompression::Lossless => config.lossless = true,
        ImageCompression::Lossy { quality } => {
            if quality.get() >= 100 {
                config.lossless = true;
            } else {
                config.distance = distance_from_quality(quality.as_f32());
            }
        }
    }

    let encoded = match decoded.format {
        DecodedPixelFormat::L8 => encode_image_gray(pixels, width, height, &config),
        DecodedPixelFormat::La8 => encode_image_gray_alpha(pixels, width, height, &config),
        DecodedPixelFormat::Rgb8 => encode_image(pixels, width, height, &config),
        DecodedPixelFormat::Rgba8 => encode_image_with_alpha(pixels, width, height, &config),
    }
    .map_err(|err: JixelEncodeError| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: format!("failed to encode JPEG XL: {err}"),
    })?;
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use super::{encode_bmp_streaming, encode_png_streaming, encode_png_to_writer};
    use crate::PipelineError;
    use crate::bmp::BmpFrameScratch;
    use crate::decode::{DecodedFrameMeta, DecodedPixelFormat};
    use std::env;
    use std::fs;
    use std::io::{self, Write};
    use std::path::Path;
    use std::process;
    use std::sync::Mutex;
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
            has_non_opaque_native_alpha: false,
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
            has_non_opaque_native_alpha: false,
        };

        let png_target = dir.join("private.png");
        let bmp_target = dir.join("private.bmp");
        fs::write(&png_target, b"old")?;
        fs::write(&bmp_target, b"old")?;
        fs::set_permissions(&png_target, fs::Permissions::from_mode(0o600))?;
        fs::set_permissions(&bmp_target, fs::Permissions::from_mode(0o600))?;

        encode_png_streaming(&png_target, &pixels, meta)?;
        let mut scratch = BmpFrameScratch::new();
        encode_bmp_streaming(&bmp_target, &pixels, meta, &mut scratch, &Mutex::new(()))?;

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
            has_non_opaque_native_alpha: false,
        };
        assert!(encode_png_streaming(&png_target, &bad_pixels, meta).is_err());
        let mut scratch = BmpFrameScratch::new();
        assert!(
            encode_bmp_streaming(
                &bmp_target,
                &bad_pixels,
                meta,
                &mut scratch,
                &Mutex::new(())
            )
            .is_err()
        );

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

    #[test]
    fn tiff_streaming_roundtrips_through_zero_copy_decoder()
    -> Result<(), Box<dyn std::error::Error>> {
        use super::encode_tiff_streaming;
        use crate::decode::{DecodeTargetHint, decode_image_into};
        use fastconv_domain::{ImageBatchItem, ImageDimensions, ImageInputFormat};

        let id = CPU_TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("fastconv-cpu-tiff-{}-{id}", process::id()));
        fs::create_dir_all(&dir)?;

        let mut file_bytes = Vec::new();
        let mut decoded_buf = Vec::new();
        let mut aux_buf = Vec::new();

        for (name, format, channels) in [
            ("l8.tif", DecodedPixelFormat::L8, 1_usize),
            ("rgb8.tif", DecodedPixelFormat::Rgb8, 3_usize),
            ("rgba8.tif", DecodedPixelFormat::Rgba8, 4_usize),
        ] {
            let pixels: Vec<u8> = (0..17 * 13 * channels)
                .map(|i| u8::try_from((i * 29 + 7) % 256).unwrap_or(0))
                .collect();
            let meta = DecodedFrameMeta {
                width: 17,
                height: 13,
                format,
                byte_len: pixels.len(),
                input_file_bytes: 1,
                has_non_opaque_native_alpha: false,
            };
            let target = dir.join(name);
            encode_tiff_streaming(&target, &pixels, meta, &mut aux_buf)?;

            let item = ImageBatchItem::new(
                target,
                dir.join(format!("{name}.out.png")),
                ImageInputFormat::Tiff,
                ImageDimensions::new(17, 13)?,
            )?;
            let roundtrip = decode_image_into(
                &item,
                DecodeTargetHint::Native,
                &mut file_bytes,
                &mut decoded_buf,
            )?;
            assert_eq!(roundtrip.format, format);
            assert_eq!(roundtrip.byte_len, pixels.len());
            assert_eq!(&decoded_buf[..roundtrip.byte_len], pixels.as_slice());
        }

        let la_pixels: Vec<u8> = (0..15 * 11 * 2)
            .map(|i| u8::try_from((i * 19 + 3) % 256).unwrap_or(0))
            .collect();
        let la_meta = DecodedFrameMeta {
            width: 15,
            height: 11,
            format: DecodedPixelFormat::La8,
            byte_len: la_pixels.len(),
            input_file_bytes: 1,
            has_non_opaque_native_alpha: false,
        };
        let la_target = dir.join("la8.tif");
        encode_tiff_streaming(&la_target, &la_pixels, la_meta, &mut aux_buf)?;
        let la_item = ImageBatchItem::new(
            la_target,
            dir.join("la8.out.png"),
            ImageInputFormat::Tiff,
            ImageDimensions::new(15, 11)?,
        )?;
        let la_roundtrip = decode_image_into(
            &la_item,
            DecodeTargetHint::Native,
            &mut file_bytes,
            &mut decoded_buf,
        )?;
        assert_eq!(la_roundtrip.format, DecodedPixelFormat::Rgba8);
        let expected_rgba: Vec<u8> = la_pixels
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|pair| [pair[0], pair[0], pair[0], pair[1]])
            .collect();
        assert_eq!(
            &decoded_buf[..la_roundtrip.byte_len],
            expected_rgba.as_slice()
        );

        let odd_la_target = dir.join("odd_la8.tif");
        assert!(matches!(
            encode_tiff_streaming(&odd_la_target, &[1_u8, 2, 3], la_meta, &mut aux_buf),
            Err(PipelineError::DecodeFailed { .. })
        ));

        fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn jxl_encodes_lossless_and_lossy_roundtrips() -> Result<(), Box<dyn std::error::Error>> {
        use super::encode_jxl_into;
        use crate::decode::{DecodeTargetHint, decode_image_into};
        use fastconv_domain::{
            ImageBatchItem, ImageCompression, ImageDimensions, ImageInputFormat, ImageQuality,
        };

        let id = CPU_TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("fastconv-cpu-jxl-{}-{id}", process::id()));
        fs::create_dir_all(&dir)?;

        let mut file_bytes = Vec::new();
        let mut decoded_buf = Vec::new();

        for (name, format, channels) in [
            ("l8.jxl", DecodedPixelFormat::L8, 1_usize),
            ("la8.jxl", DecodedPixelFormat::La8, 2_usize),
            ("rgb8.jxl", DecodedPixelFormat::Rgb8, 3_usize),
            ("rgba8.jxl", DecodedPixelFormat::Rgba8, 4_usize),
        ] {
            let pixels: Vec<u8> = (0..16 * 16 * channels)
                .map(|i| u8::try_from((i * 31 + 11) % 256).unwrap_or(0))
                .collect();
            let meta = DecodedFrameMeta {
                width: 16,
                height: 16,
                format,
                byte_len: pixels.len(),
                input_file_bytes: 1,
                has_non_opaque_native_alpha: false,
            };
            let target = dir.join(name);
            let encoded = encode_jxl_into(&target, ImageCompression::Lossless, &pixels, meta, 1)?;
            fs::write(&target, &encoded)?;

            let item = ImageBatchItem::new(
                target,
                dir.join(format!("{name}.out.png")),
                ImageInputFormat::Jxl,
                ImageDimensions::new(16, 16)?,
            )?;
            let roundtrip = decode_image_into(
                &item,
                DecodeTargetHint::Native,
                &mut file_bytes,
                &mut decoded_buf,
            )?;
            assert_eq!(roundtrip.width, 16);
            assert_eq!(roundtrip.height, 16);
            assert_eq!(roundtrip.format, format);
            assert_eq!(roundtrip.byte_len, pixels.len());
            assert_eq!(&decoded_buf[..roundtrip.byte_len], pixels.as_slice());
        }

        let lossy_pixels: Vec<u8> = (0..16 * 16 * 3)
            .map(|i| u8::try_from((i * 17) % 256).unwrap_or(0))
            .collect();
        let lossy_meta = DecodedFrameMeta {
            width: 16,
            height: 16,
            format: DecodedPixelFormat::Rgb8,
            byte_len: lossy_pixels.len(),
            input_file_bytes: 1,
            has_non_opaque_native_alpha: false,
        };
        let lossy_target = dir.join("lossy.jxl");
        let lossy_encoded = encode_jxl_into(
            &lossy_target,
            ImageCompression::Lossy {
                quality: ImageQuality::new(80)?,
            },
            &lossy_pixels,
            lossy_meta,
            1,
        )?;
        fs::write(&lossy_target, &lossy_encoded)?;
        let lossy_item = ImageBatchItem::new(
            lossy_target,
            dir.join("lossy.out.png"),
            ImageInputFormat::Jxl,
            ImageDimensions::new(16, 16)?,
        )?;
        let lossy_roundtrip = decode_image_into(
            &lossy_item,
            DecodeTargetHint::Native,
            &mut file_bytes,
            &mut decoded_buf,
        )?;
        assert_eq!(lossy_roundtrip.width, 16);
        assert_eq!(lossy_roundtrip.height, 16);
        assert_eq!(lossy_roundtrip.format, DecodedPixelFormat::Rgb8);

        fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
