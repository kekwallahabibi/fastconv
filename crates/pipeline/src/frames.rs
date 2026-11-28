//! Streaming still-frame extraction with bounded, recycled pixel buffers.
//!
//! Frames are never accumulated in memory: the decode thread fills one
//! recycled RGB(A) buffer at a time and hands it to a bounded pool of encoder
//! workers through a channel whose capacity is derived from a hard byte budget.
//! Workers return their buffer to the decoder instead of dropping it, so the
//! in-flight raw pixel footprint stays far below 256 MiB even for 4K sources.

use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

use crossbeam_channel as channel;

use fastconv_domain::{
    FrameExtractionPlan, FrameExtractionReport, FrameRange, FrameSelection, ImageExecutionMode,
    VideoContainerFormat,
};

use crate::PipelineError;
use crate::bmp::BmpFrameScratch;
use crate::cpu::{CpuImageEncoder, EncodeBuffers, encode_frame_pixels};
use crate::decode::DecodedPixelFormat;
use crate::video_anim::{
    decode_gif_frames_streaming_pixels_ctrl, decode_webp_frames_streaming_pixels_ctrl,
    nv12_to_rgb8_into,
};
use crate::video_codecs::{decode_video_track_streaming_ctrl, nv12_buffer_len};
use crate::video_container::demux_video;

/// Upper bound on raw pixel bytes held by in-flight extraction jobs. Buffers
/// returned by workers are reused, so the allocation count tracks this budget
/// rather than the total number of frames in the source.
const MAX_IN_FLIGHT_BYTES: usize = 256 * 1024 * 1024;
/// Memory-hungry 4K still encoders use at most this many image workers.
const MAX_HEAVY_WORKERS: usize = 8;
/// Fallback per-pixel size when the source has no probed dimensions.
const FALLBACK_FRAME_BYTES: usize = 3840 * 2160 * 4;

/// One recycled decoded frame travelling from decoder to encoder worker.
struct FrameJob {
    sequence: u64,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    format: DecodedPixelFormat,
}

/// Chooses frames deterministically from an inclusive zero-based range.
struct Selector {
    mode: FrameSelection,
    start: u64,
    interval_us: u128,
    base_pts_us: Option<i64>,
    next_target_us: u128,
}

impl Selector {
    fn new(selection: FrameSelection, start: u64) -> Self {
        let interval_us = match selection {
            FrameSelection::TargetFps(fps) => fps_interval_us(fps),
            FrameSelection::All | FrameSelection::Every(_) => 1,
        };
        Self {
            mode: selection,
            start,
            interval_us,
            base_pts_us: None,
            next_target_us: 0,
        }
    }

    /// Decides whether the frame at `index` / `pts_us` should be written.
    fn should_emit(&mut self, index: u64, pts_us: i64) -> bool {
        match self.mode {
            FrameSelection::All => true,
            FrameSelection::Every(n) => (index - self.start).is_multiple_of(u64::from(n.get())),
            FrameSelection::TargetFps(_) => {
                let base = *self.base_pts_us.get_or_insert(pts_us);
                // Non-monotonic timestamps never rewind the selection clock.
                if pts_us < base {
                    return false;
                }
                let delta = u128::try_from(pts_us.saturating_sub(base)).unwrap_or(u128::MAX);
                if delta < self.next_target_us {
                    return false;
                }
                // Jump to the next whole interval so a timestamp gap emits one
                // frame instead of a burst of skipped intervals.
                let steps = delta / self.interval_us;
                self.next_target_us = steps
                    .checked_add(1)
                    .and_then(|n| n.checked_mul(self.interval_us))
                    .unwrap_or(u128::MAX);
                true
            }
        }
    }
}

/// Converts a validated FPS into a microsecond interval. `fps` is finite and
/// positive by construction; clamping before the lossy narrowing keeps the
/// result total and avoids overflow for pathological tiny rates.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn fps_interval_us(fps: f64) -> u128 {
    let seconds = (1_000_000.0 / fps).clamp(1.0, u64::MAX as f64);
    u128::from(seconds as u64)
}

fn encoder_for_mode(mode: ImageExecutionMode) -> Result<CpuImageEncoder, PipelineError> {
    Ok(match mode {
        ImageExecutionMode::CpuWebp { compression, level } => {
            CpuImageEncoder::Webp { compression, level }
        }
        ImageExecutionMode::CpuJpeg { quality, chroma } => {
            CpuImageEncoder::Jpeg { quality, chroma }
        }
        ImageExecutionMode::CpuPng { level } => CpuImageEncoder::Png { level },
        ImageExecutionMode::CpuBmp => CpuImageEncoder::Bmp,
        ImageExecutionMode::CpuQoi => CpuImageEncoder::Qoi,
        ImageExecutionMode::CpuTiff { compression } => CpuImageEncoder::Tiff { compression },
        ImageExecutionMode::CpuJxl { compression, level } => {
            CpuImageEncoder::Jxl { compression, level }
        }
        ImageExecutionMode::CpuAvif {
            quality,
            level,
            chroma,
        } => CpuImageEncoder::Avif {
            quality,
            level,
            chroma,
            threads: 1,
        },
        ImageExecutionMode::CpuGif => CpuImageEncoder::Gif,
        ImageExecutionMode::CpuIco => CpuImageEncoder::Ico,
        ImageExecutionMode::CpuPnm => CpuImageEncoder::Pnm,
        ImageExecutionMode::CpuHeic { quality } => CpuImageEncoder::Heic { quality },
        ImageExecutionMode::NvencHeic { .. } => {
            return Err(PipelineError::GpuBackendFailed {
                backend: fastconv_domain::GpuBackend::Nvenc,
                reason: "GPU frame extraction is rejected by the validated plan".to_owned(),
            });
        }
    })
}

fn frame_byte_len(width: u32, height: u32, channels: usize) -> Result<usize, PipelineError> {
    let pixels = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: PathBuf::new(),
            reason: "frame dimensions overflow usize".to_owned(),
        })?;
    pixels
        .checked_mul(channels)
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: PathBuf::new(),
            reason: "frame pixel buffer overflows usize".to_owned(),
        })
}

/// Reuses a returned buffer when possible, otherwise allocates one within the
/// hard in-flight budget. `vec!` is only reached after the cap check.
fn take_buffer(pool: &mut Vec<Vec<u8>>, needed: usize) -> Result<Vec<u8>, PipelineError> {
    if needed > MAX_IN_FLIGHT_BYTES {
        return Err(PipelineError::DecodeFailed {
            path: PathBuf::new(),
            reason: format!(
                "single frame needs {needed} bytes, above the {MAX_IN_FLIGHT_BYTES}-byte frame budget"
            ),
        });
    }
    if let Some(index) = pool.iter().position(|buffer| buffer.capacity() >= needed) {
        let mut buffer = pool.remove(index);
        buffer.clear();
        buffer.resize(needed, 0);
        return Ok(buffer);
    }
    // Discard oversized stale buffers so an anomalously large frame cannot pin
    // memory for the rest of the extraction.
    pool.clear();
    Ok(vec![0_u8; needed])
}

fn channel_capacity(frame_bytes: usize, workers: usize) -> usize {
    let slots = MAX_IN_FLIGHT_BYTES / frame_bytes.max(1);
    slots.clamp(1, workers.saturating_mul(2)).max(1)
}

/// Decode-side state: selection, buffer pool, and bounded job production.
struct FrameEmitter<'a> {
    selector: Selector,
    range: FrameRange,
    pool: Vec<Vec<u8>>,
    job_tx: channel::Sender<FrameJob>,
    buffer_rx: &'a channel::Receiver<Vec<u8>>,
    error_rx: &'a channel::Receiver<PipelineError>,
    failed: &'a AtomicBool,
    sequence: u64,
    last_dims: Option<(u32, u32)>,
}

impl FrameEmitter<'_> {
    /// Fills a recycled buffer for one selected frame and queues it.
    #[allow(clippy::too_many_arguments)]
    fn emit(
        &mut self,
        source: &Path,
        index: u64,
        pts_us: i64,
        width: u32,
        height: u32,
        format: DecodedPixelFormat,
        channels: usize,
        fill: &mut dyn FnMut(&mut [u8]) -> Result<(), PipelineError>,
    ) -> Result<(), PipelineError> {
        if !self.selector.should_emit(index, pts_us) {
            return Ok(());
        }
        if self.failed.load(Ordering::Relaxed) {
            return Err(PipelineError::WorkerPanicked);
        }
        let needed =
            frame_byte_len(width, height, channels).map_err(|_| PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: "frame dimensions overflow the pixel buffer".to_owned(),
            })?;
        while let Ok(returned) = self.buffer_rx.try_recv() {
            if returned.capacity() <= MAX_IN_FLIGHT_BYTES && self.pool.len() < 32 {
                self.pool.push(returned);
            }
        }
        let mut pixels = take_buffer(&mut self.pool, needed)?;
        fill(&mut pixels)?;
        self.sequence = self.sequence.saturating_add(1);
        self.job_tx
            .send(FrameJob {
                sequence: self.sequence,
                pixels,
                width,
                height,
                format,
            })
            .map_err(|_| {
                self.error_rx
                    .try_recv()
                    .unwrap_or(PipelineError::WorkerPanicked)
            })?;
        self.last_dims = Some((width, height));
        Ok(())
    }
}

fn decode_animation(
    source: &Path,
    container: VideoContainerFormat,
    emitter: &mut FrameEmitter<'_>,
) -> Result<ControlFlow<()>, PipelineError> {
    let mut frame_index = 0_u64;
    let mut on_frame = |info: crate::video_anim::AnimFrameInfo, pixels: &[u8]| {
        let index = frame_index;
        frame_index = frame_index.saturating_add(1);
        if emitter.range.is_past_end(index) {
            return Ok(ControlFlow::Break(()));
        }
        if !emitter.range.contains(index) {
            return Ok(ControlFlow::Continue(()));
        }
        emitter.emit(
            source,
            index,
            info.pts_us,
            info.width,
            info.height,
            DecodedPixelFormat::Rgba8,
            4,
            &mut |dst| {
                let Some(slice) = pixels.get(..dst.len()) else {
                    return Err(PipelineError::DecodeFailed {
                        path: source.to_path_buf(),
                        reason: format!(
                            "animation decoder produced {} bytes, expected at least {}",
                            pixels.len(),
                            dst.len()
                        ),
                    });
                };
                dst.copy_from_slice(slice);
                Ok(())
            },
        )?;
        Ok(ControlFlow::Continue(()))
    };
    match container {
        VideoContainerFormat::Gif => {
            decode_gif_frames_streaming_pixels_ctrl(source, |info, pixels| on_frame(info, pixels))
        }
        VideoContainerFormat::Webp => {
            decode_webp_frames_streaming_pixels_ctrl(source, |info, pixels| on_frame(info, pixels))
        }
        _ => Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "not an animation container".to_owned(),
        }),
    }
}

fn decode_video(
    source: &Path,
    container: VideoContainerFormat,
    emitter: &mut FrameEmitter<'_>,
) -> Result<ControlFlow<()>, PipelineError> {
    let media = demux_video(source, container)?;
    let track_index = media
        .tracks
        .iter()
        .position(|track| track.codec.is_video())
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "source has no video track".to_owned(),
        })?;
    let track = &media.tracks[track_index];
    let packets: Vec<_> = media
        .packets
        .iter()
        .filter(|packet| packet.track == track_index)
        .cloned()
        .collect();
    let mut frame_index = 0_u64;
    let mut nv12 = Vec::new();
    decode_video_track_streaming_ctrl(source, track, &packets, |info, pack| {
        let index = frame_index;
        frame_index = frame_index.saturating_add(1);
        if emitter.range.is_past_end(index) {
            return Ok(ControlFlow::Break(()));
        }
        if !emitter.range.contains(index) {
            return Ok(ControlFlow::Continue(()));
        }
        emitter.emit(
            source,
            index,
            info.pts_us,
            info.width,
            info.height,
            DecodedPixelFormat::Rgb8,
            3,
            &mut |dst| {
                let pitch =
                    usize::try_from(info.width).map_err(|_| PipelineError::DecodeFailed {
                        path: source.to_path_buf(),
                        reason: "NV12 pitch overflows usize".to_owned(),
                    })?;
                let needed = nv12_buffer_len(info.width, info.height, pitch)?;
                if nv12.len() < needed {
                    if needed > MAX_IN_FLIGHT_BYTES {
                        return Err(PipelineError::DecodeFailed {
                            path: source.to_path_buf(),
                            reason: "NV12 scratch exceeds the frame budget".to_owned(),
                        });
                    }
                    nv12.clear();
                    nv12.resize(needed, 0);
                }
                pack(nv12.as_mut_slice(), pitch)?;
                nv12_to_rgb8_into(&nv12, dst, pitch, info.width, info.height, info.color)
            },
        )?;
        Ok(ControlFlow::Continue(()))
    })
}

fn run_decode(
    plan: &FrameExtractionPlan,
    emitter: &mut FrameEmitter<'_>,
) -> Result<(), PipelineError> {
    let flow = match plan.source_container() {
        VideoContainerFormat::Gif | VideoContainerFormat::Webp => {
            decode_animation(plan.source(), plan.source_container(), emitter)?
        }
        container => decode_video(plan.source(), container, emitter)?,
    };
    // Breaking early (past `--end`) is an expected optimisation.
    let _ = flow;
    Ok(())
}

struct WorkerContext<'a> {
    job_rx: &'a channel::Receiver<FrameJob>,
    buffer_tx: &'a channel::Sender<Vec<u8>>,
    error_tx: &'a channel::Sender<PipelineError>,
    failed: &'a AtomicBool,
    total_bytes: &'a AtomicU64,
    bmp_write_lock: &'a Mutex<()>,
    encoder: &'a CpuImageEncoder,
    inner_threads: usize,
    source: &'a Path,
    output_dir: &'a Path,
    extension: &'a str,
}

fn run_worker(ctx: &WorkerContext<'_>) {
    let job_rx = ctx.job_rx;
    let buffer_tx = ctx.buffer_tx;
    let error_tx = ctx.error_tx;
    let failed = ctx.failed;
    let total_bytes = ctx.total_bytes;
    let bmp_write_lock = ctx.bmp_write_lock;
    let encoder = ctx.encoder;
    let inner_threads = ctx.inner_threads;
    let source = ctx.source;
    let output_dir = ctx.output_dir;
    let extension = ctx.extension;
    let mut encode_buf = Vec::new();
    let mut aux = Vec::new();
    let mut bmp_frame = BmpFrameScratch::new();
    while let Ok(mut job) = job_rx.recv() {
        if failed.load(Ordering::Relaxed) {
            // Keep draining jobs and returning buffers so the decoder can
            // never block on a failed worker.
            if buffer_tx.send(job.pixels).is_err() {
                break;
            }
            continue;
        }
        let target = output_dir.join(format!("frame_{:06}.{extension}", job.sequence));
        let decoded = crate::decode::DecodedFrameMeta {
            width: job.width,
            height: job.height,
            format: job.format,
            byte_len: job.pixels.len(),
            input_file_bytes: 0,
            has_non_opaque_native_alpha: false,
        };
        let mut buffers = EncodeBuffers {
            encoded: &mut encode_buf,
            aux: &mut aux,
            bmp_frame: &mut bmp_frame,
        };
        let result = encode_frame_pixels(
            &target,
            source,
            *encoder,
            &mut job.pixels,
            decoded,
            &mut buffers,
            inner_threads,
            bmp_write_lock,
        );
        match result {
            Ok(bytes) => {
                total_bytes.fetch_add(bytes, Ordering::Relaxed);
            }
            Err(err) => {
                failed.store(true, Ordering::Relaxed);
                if error_tx.send(err).is_err() {
                    // The decoder has already stopped; nothing can block on
                    // this worker's remaining jobs.
                    break;
                }
            }
        }
        if buffer_tx.send(job.pixels).is_err() {
            break;
        }
    }
}

/// Executes a validated [`FrameExtractionPlan`] on its CPU encoder path.
///
/// # Errors
///
/// Returns [`PipelineError`] on demux/decode/encode/write failure, or
/// [`PipelineError::NoFramesSelected`] (with no files or directory created)
/// when the selection and range match no frame.
pub fn execute_frame_extraction(
    plan: &FrameExtractionPlan,
) -> Result<FrameExtractionReport, PipelineError> {
    let encoder = encoder_for_mode(plan.mode())?;
    let configured = plan.worker_threads().get();
    let (workers, inner_threads) = match encoder {
        CpuImageEncoder::Jxl { .. } | CpuImageEncoder::Heic { .. } => {
            let count = configured.clamp(1, MAX_HEAVY_WORKERS);
            (count, (configured / count).max(1))
        }
        CpuImageEncoder::Avif { .. } => {
            let count = configured.clamp(1, MAX_HEAVY_WORKERS);
            (count, (configured / count).clamp(1, 8))
        }
        _ => (configured.max(1), 1),
    };
    let known_bytes = plan
        .source_dimensions()
        .and_then(|dims| frame_byte_len(dims.width().get(), dims.height().get(), 4).ok())
        .unwrap_or(FALLBACK_FRAME_BYTES);
    let (job_tx, job_rx) = channel::bounded::<FrameJob>(channel_capacity(known_bytes, workers));
    let (buffer_tx, buffer_rx) = channel::unbounded::<Vec<u8>>();
    let (error_tx, error_rx) = channel::unbounded::<PipelineError>();
    let failed = AtomicBool::new(false);
    let total_bytes = AtomicU64::new(0);
    let bmp_write_lock = Mutex::new(());
    let extension = plan.output_format().extension();

    let decode_result = thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                let ctx = WorkerContext {
                    job_rx: &job_rx,
                    buffer_tx: &buffer_tx,
                    error_tx: &error_tx,
                    failed: &failed,
                    total_bytes: &total_bytes,
                    bmp_write_lock: &bmp_write_lock,
                    encoder: &encoder,
                    inner_threads,
                    source: plan.source(),
                    output_dir: plan.output_dir(),
                    extension,
                };
                run_worker(&ctx);
            });
        }
        let mut emitter = FrameEmitter {
            selector: Selector::new(plan.selection(), plan.range().start()),
            range: plan.range(),
            pool: Vec::new(),
            job_tx,
            buffer_rx: &buffer_rx,
            error_rx: &error_rx,
            failed: &failed,
            sequence: 0,
            last_dims: None,
        };
        run_decode(plan, &mut emitter)?;
        Ok((emitter.sequence, emitter.last_dims))
    });

    if let Ok(err) = error_rx.try_recv() {
        return Err(err);
    }
    let (frame_count, dims) = decode_result?;
    if failed.load(Ordering::Relaxed) {
        return Err(error_rx.try_recv().unwrap_or(PipelineError::WorkerPanicked));
    }
    let output_dir = plan.output_dir();
    // A zero-frame selection is reported before checking output bytes so the
    // caller sees the actionable "nothing matched" error, not an empty-output
    // error from an encoder that never ran.
    let frames = NonZeroU64::new(frame_count)
        .ok_or_else(|| PipelineError::NoFramesSelected(plan.source().to_path_buf()))?;
    let (width, height) =
        dims.ok_or_else(|| PipelineError::NoFramesSelected(plan.source().to_path_buf()))?;
    let bytes = NonZeroU64::new(total_bytes.load(Ordering::Relaxed))
        .ok_or_else(|| PipelineError::EmptyOutput(output_dir.to_path_buf()))?;
    let width = NonZeroU32::new(width)
        .ok_or_else(|| PipelineError::EmptyOutput(output_dir.to_path_buf()))?;
    let height = NonZeroU32::new(height)
        .ok_or_else(|| PipelineError::EmptyOutput(output_dir.to_path_buf()))?;

    Ok(FrameExtractionReport::new(
        frames,
        width,
        height,
        plan.output_format(),
        output_dir.to_path_buf(),
        bytes,
        NonZeroUsize::new(workers).unwrap_or(NonZeroUsize::MIN),
    ))
}

#[cfg(test)]
mod tests {
    use super::{FrameExtractionPlan, PipelineError, execute_frame_extraction};
    use crate::video_anim::{AnimFrameInfo, encode_animation_stream};
    use crate::video_codecs::{Yuv420Frame, YuvColor, encode_video_cpu};
    use crate::video_container::{ContainerMedia, mux_video};
    use fastconv_domain::{
        AnimationLoop, FrameRange, FrameSelection, ImageDimensions, ImageExecutionMode,
        ImageOutputFormat, VideoContainerFormat, VideoEncodeMode, VideoQuality, VideoRateControl,
    };
    use std::env;
    use std::error::Error;
    use std::fs::{self, File};
    use std::io::BufReader;
    use std::num::{NonZeroU32, NonZeroUsize};
    use std::path::{Path, PathBuf};
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_root(label: &str) -> Result<PathBuf, Box<dyn Error>> {
        let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!("fastconv-frames-{label}-{}-{id}", process::id()));
        fs::create_dir_all(&path)?;
        Ok(path)
    }

    fn solid_frames(count: u32, width: u32, height: u32) -> Vec<Yuv420Frame> {
        let uv_len = usize::try_from((width / 2) * (height / 2)).unwrap_or(0);
        let uv = vec![128_u8; uv_len];
        (0..count)
            .map(|index| Yuv420Frame {
                width,
                height,
                y: vec![
                    u8::try_from(70_u32 + index.saturating_mul(20)).unwrap_or(128);
                    usize::try_from(width * height).unwrap_or(0)
                ],
                u: uv.clone(),
                v: uv.clone(),
                pts_us: i64::from(index) * 100_000,
                duration_us: 100_000,
                color: YuvColor::default(),
            })
            .collect()
    }

    fn write_video(
        path: &Path,
        format: VideoContainerFormat,
        frames: &[Yuv420Frame],
    ) -> Result<(), Box<dyn Error>> {
        let mode = match format {
            VideoContainerFormat::Mp4 | VideoContainerFormat::Mov | VideoContainerFormat::Ts => {
                VideoEncodeMode::CpuH264 {
                    rate_control: VideoRateControl::Quality(VideoQuality::DEFAULT),
                }
            }
            VideoContainerFormat::Mkv | VideoContainerFormat::Avi => VideoEncodeMode::CpuMjpeg {
                quality: VideoQuality::DEFAULT,
            },
            VideoContainerFormat::Webm => VideoEncodeMode::CpuVp9 {
                rate_control: VideoRateControl::Quality(VideoQuality::DEFAULT),
            },
            VideoContainerFormat::Gif | VideoContainerFormat::Webp => {
                return Err("animation fixtures use encode_animation_stream".into());
            }
        };
        let stream = encode_video_cpu(path, frames, mode, 0)?;
        let media = ContainerMedia {
            tracks: vec![stream.track],
            packets: stream.packets,
        };
        let file = File::create(path)?;
        mux_video(&file, path, format, &media)?;
        Ok(())
    }

    fn plan(
        source: &Path,
        out: &Path,
        format: VideoContainerFormat,
        output_format: ImageOutputFormat,
        selection: FrameSelection,
        range: FrameRange,
    ) -> Result<FrameExtractionPlan, Box<dyn Error>> {
        let dimensions = ImageDimensions::new(32, 32)?;
        Ok(FrameExtractionPlan::new(
            source.to_path_buf(),
            format,
            Some(dimensions),
            out.to_path_buf(),
            output_format,
            ImageExecutionMode::resolve_cpu(output_format, None)?,
            selection,
            range,
            NonZeroUsize::new(2).ok_or("nonzero")?,
        )?)
    }

    fn decode_png(path: &Path) -> Result<(u32, u32, usize), Box<dyn Error>> {
        let decoder = png::Decoder::new(BufReader::new(File::open(path)?));
        let mut reader = decoder.read_info()?;
        let mut buf = vec![0_u8; reader.output_buffer_size().ok_or("png buffer")?];
        let frame = reader.next_frame(&mut buf)?;
        Ok((frame.width, frame.height, frame.buffer_size()))
    }

    fn all_formats() -> [ImageOutputFormat; 12] {
        [
            ImageOutputFormat::Png,
            ImageOutputFormat::Jpeg,
            ImageOutputFormat::Webp,
            ImageOutputFormat::Qoi,
            ImageOutputFormat::Bmp,
            ImageOutputFormat::Tiff,
            ImageOutputFormat::Jxl,
            ImageOutputFormat::Avif,
            ImageOutputFormat::Gif,
            ImageOutputFormat::Ico,
            ImageOutputFormat::Pnm,
            ImageOutputFormat::Heic,
        ]
    }

    #[test]
    fn extracts_mp4_frames_with_deterministic_names() -> Result<(), Box<dyn Error>> {
        let root = temp_root("mp4")?;
        let source = root.join("in.mp4");
        write_video(&source, VideoContainerFormat::Mp4, &solid_frames(4, 32, 32))?;
        let out = root.join("png");
        let report = execute_frame_extraction(&plan(
            &source,
            &out,
            VideoContainerFormat::Mp4,
            ImageOutputFormat::Png,
            FrameSelection::All,
            FrameRange::new(0, None)?,
        )?)?;
        assert_eq!(report.frames_extracted().get(), 4);
        for index in 1_u64..=4 {
            let file = out.join(format!("frame_{index:06}.png"));
            let (w, h, bytes) = decode_png(&file)?;
            assert_eq!((w, h), (32, 32));
            assert_eq!(bytes, 32 * 32 * 3);
        }
        assert!(!out.join("frame_000005.png").exists());
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn extracts_all_twelve_image_formats_from_mkv() -> Result<(), Box<dyn Error>> {
        let root = temp_root("formats")?;
        let source = root.join("in.mkv");
        write_video(&source, VideoContainerFormat::Mkv, &solid_frames(2, 32, 32))?;
        for format in all_formats() {
            let out = root.join(format!("out-{}", format.extension()));
            let report = execute_frame_extraction(&plan(
                &source,
                &out,
                VideoContainerFormat::Mkv,
                format,
                FrameSelection::All,
                FrameRange::new(0, None)?,
            )?)?;
            assert_eq!(report.frames_extracted().get(), 2, ".{format} frame count");
            for index in 1_u64..=2 {
                let file = out.join(format!("frame_{index:06}.{}", format.extension()));
                assert!(fs::metadata(&file)?.len() > 0, ".{format} output is empty");
            }
            if format == ImageOutputFormat::Png {
                assert_eq!(decode_png(&out.join("frame_000001.png"))?.0, 32);
            }
        }
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn extracts_every_fps_and_range_selections() -> Result<(), Box<dyn Error>> {
        let root = temp_root("selection")?;
        let source = root.join("in.mp4");
        write_video(&source, VideoContainerFormat::Mp4, &solid_frames(8, 32, 32))?;

        let out_every = root.join("every");
        let every = execute_frame_extraction(&plan(
            &source,
            &out_every,
            VideoContainerFormat::Mp4,
            ImageOutputFormat::Png,
            FrameSelection::Every(NonZeroU32::new(2).ok_or("nz")?),
            FrameRange::new(0, None)?,
        )?)?;
        assert_eq!(every.frames_extracted().get(), 4);
        assert!(out_every.join("frame_000004.png").exists());

        let out_range = root.join("range");
        let range = execute_frame_extraction(&plan(
            &source,
            &out_range,
            VideoContainerFormat::Mp4,
            ImageOutputFormat::Png,
            FrameSelection::All,
            FrameRange::new(2, Some(4))?,
        )?)?;
        assert_eq!(range.frames_extracted().get(), 3);
        assert!(out_range.join("frame_000001.png").exists());
        assert!(!out_range.join("frame_000004.png").exists());

        let out_fps = root.join("fps");
        let fps = execute_frame_extraction(&plan(
            &source,
            &out_fps,
            VideoContainerFormat::Mp4,
            ImageOutputFormat::Png,
            FrameSelection::target_fps(5.0)?,
            FrameRange::new(0, None)?,
        )?)?;
        assert_eq!(
            fps.frames_extracted().get(),
            4,
            "5 fps from a 10 fps source"
        );

        let out_none = root.join("none");
        let missing = execute_frame_extraction(&plan(
            &source,
            &out_none,
            VideoContainerFormat::Mp4,
            ImageOutputFormat::Png,
            FrameSelection::All,
            FrameRange::new(100, None)?,
        )?);
        assert!(matches!(missing, Err(PipelineError::NoFramesSelected(_))));
        assert!(!out_none.exists(), "no directory is created without frames");
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    fn write_animated_webp(path: &Path, count: u32) -> Result<(), Box<dyn Error>> {
        let file = File::create(path)?;
        encode_animation_stream(
            VideoContainerFormat::Webp,
            VideoQuality::new(90)?,
            1,
            AnimationLoop::Infinite,
            &file,
            |sink| {
                for index in 0..count {
                    let mut rgba = Vec::with_capacity(8 * 8 * 4);
                    for y in 0_u32..8 {
                        for x in 0_u32..8 {
                            let alpha = if x < 4 { 255 } else { 0 };
                            rgba.extend_from_slice(&[
                                u8::try_from(index * 40).unwrap_or(0),
                                u8::try_from(y * 20).unwrap_or(0),
                                u8::try_from(x * 20).unwrap_or(0),
                                alpha,
                            ]);
                        }
                    }
                    sink(
                        AnimFrameInfo {
                            width: 8,
                            height: 8,
                            pts_us: i64::from(index) * 100_000,
                            duration_us: 100_000,
                            channels: 4,
                        },
                        &rgba,
                    )?;
                }
                Ok(())
            },
        )?;
        Ok(())
    }

    #[test]
    fn extracts_animated_webp_with_alpha_into_png() -> Result<(), Box<dyn Error>> {
        let root = temp_root("webp-alpha")?;
        let source = root.join("anim.webp");
        write_animated_webp(&source, 3)?;
        let out = root.join("png");
        let report = execute_frame_extraction(&plan(
            &source,
            &out,
            VideoContainerFormat::Webp,
            ImageOutputFormat::Png,
            FrameSelection::All,
            FrameRange::new(0, None)?,
        )?)?;
        assert_eq!(report.frames_extracted().get(), 3);
        let decoder = png::Decoder::new(BufReader::new(File::open(out.join("frame_000002.png"))?));
        let mut reader = decoder.read_info()?;
        let mut buf = vec![0_u8; reader.output_buffer_size().ok_or("png buffer")?];
        let frame = reader.next_frame(&mut buf)?;
        assert_eq!(frame.color_type, png::ColorType::Rgba);
        assert_eq!(buf[3], 255, "left half stays opaque");
        assert_eq!(buf[(4 * 7 + 3) * 4 + 3], 0, "right half stays transparent");
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn extracts_animated_gif_with_disposal_and_compositing() -> Result<(), Box<dyn Error>> {
        let root = temp_root("gif")?;
        let source = root.join("anim.gif");
        {
            let file = File::create(&source)?;
            let mut encoder = gif::Encoder::new(file, 4, 4, &[])?;
            encoder.set_repeat(gif::Repeat::Infinite)?;
            let mut put = |mut rgba: [u8; 4], left: u16, top: u16, dispose: gif::DisposalMethod| {
                let mut frame = gif::Frame::from_rgba_speed(1, 1, &mut rgba, 10);
                frame.left = left;
                frame.top = top;
                frame.delay = 10;
                frame.dispose = dispose;
                encoder.write_frame(&frame)
            };
            put([255, 0, 0, 255], 0, 0, gif::DisposalMethod::Keep)?;
            put([0, 255, 0, 255], 1, 1, gif::DisposalMethod::Keep)?;
            put([0, 0, 0, 0], 1, 1, gif::DisposalMethod::Background)?;
            put([0, 0, 255, 255], 2, 2, gif::DisposalMethod::Keep)?;
        }
        let out = root.join("png");
        let report = execute_frame_extraction(&plan(
            &source,
            &out,
            VideoContainerFormat::Gif,
            ImageOutputFormat::Png,
            FrameSelection::All,
            FrameRange::new(0, None)?,
        )?)?;
        assert_eq!(report.frames_extracted().get(), 4);
        let pixel = |file: &str, x: usize, y: usize| -> Result<[u8; 4], Box<dyn Error>> {
            let decoder = png::Decoder::new(BufReader::new(File::open(out.join(file))?));
            let mut reader = decoder.read_info()?;
            let mut buf = vec![0_u8; reader.output_buffer_size().ok_or("png buffer")?];
            reader.next_frame(&mut buf)?;
            let start = (y * 4 + x) * 4;
            Ok([buf[start], buf[start + 1], buf[start + 2], buf[start + 3]])
        };
        assert_eq!(pixel("frame_000002.png", 1, 1)?, [0, 255, 0, 255]);
        // A disposal method applies before the following frame is composited.
        assert_eq!(pixel("frame_000003.png", 1, 1)?, [0, 255, 0, 255]);
        assert_eq!(pixel("frame_000004.png", 1, 1)?, [0, 0, 0, 0]);
        assert_eq!(pixel("frame_000004.png", 2, 2)?, [0, 0, 255, 255]);
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn rejects_non_opaque_alpha_for_jpeg_without_writing_files() -> Result<(), Box<dyn Error>> {
        let root = temp_root("alpha-jpeg")?;
        let source = root.join("anim.webp");
        write_animated_webp(&source, 2)?;
        let out = root.join("jpeg");
        let result = execute_frame_extraction(&plan(
            &source,
            &out,
            VideoContainerFormat::Webp,
            ImageOutputFormat::Jpeg,
            FrameSelection::All,
            FrameRange::new(0, None)?,
        )?);
        assert!(matches!(
            result,
            Err(PipelineError::UnsupportedPixelFormat { .. })
        ));
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn extracts_frames_from_all_six_video_containers() -> Result<(), Box<dyn Error>> {
        let root = temp_root("all-containers")?;
        for container in [
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mov,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Webm,
            VideoContainerFormat::Avi,
            VideoContainerFormat::Ts,
        ] {
            let source = root.join(format!("clip.{}", container.extension()));
            write_video(&source, container, &solid_frames(3, 32, 32))?;
            let out = root.join(format!("out-{}", container.extension()));
            let report = execute_frame_extraction(&plan(
                &source,
                &out,
                container,
                ImageOutputFormat::Png,
                FrameSelection::All,
                FrameRange::new(0, None)?,
            )?)?;
            assert_eq!(
                report.frames_extracted().get(),
                3,
                "{container} frame count"
            );
            let (w, h, bytes) = decode_png(&out.join("frame_000001.png"))?;
            assert_eq!(
                (w, h, bytes),
                (32, 32, 32 * 32 * 3),
                "{container} png decode"
            );
            assert!(out.join("frame_000003.png").is_file());
            assert!(!out.join("frame_000004.png").exists());
        }
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn nv12_to_rgb8_into_matches_allocating_variant() -> Result<(), Box<dyn Error>> {
        use crate::video_anim::{nv12_to_rgb8, nv12_to_rgb8_into};
        let (width, height, pitch) = (16_u32, 16_u32, 16_usize);
        let nv12 = vec![96_u8; pitch * (usize::try_from(height)? + usize::try_from(height)? / 2)];
        let expected = nv12_to_rgb8(&nv12, pitch, width, height, YuvColor::default())?;
        let mut dst = vec![0_u8; expected.len()];
        nv12_to_rgb8_into(&nv12, &mut dst, pitch, width, height, YuvColor::default())?;
        assert_eq!(dst, expected);
        Ok(())
    }
}
