use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::process;
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
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp_path)
            {
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

    fn commit(mut self, target: &Path) -> Result<(), PipelineError> {
        fs::rename(&self.path, target).map_err(|source| PipelineError::Io {
            path: target.to_path_buf(),
            source,
        })?;
        self.committed = true;
        Ok(())
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
    let bytes = encode_png_to_writer(
        target,
        pixels,
        decoded,
        BufWriter::with_capacity(STREAM_IO_BUF_BYTES, file),
    )?;
    staging.commit(target)?;
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
    drop(file);
    staging.commit(target)?;
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
