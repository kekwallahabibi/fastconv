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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
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

/// Hard upper bound on the *aggregate* raw decoded-pixel bytes held at once by
/// one extraction: decoder-held, queued, worker-held and buffers kept for reuse
/// all count. Every live buffer stays charged to this budget until dropped.
const MAX_IN_FLIGHT_BYTES: usize = 256 * 1024 * 1024;
/// Memory-hungry 4K still encoders use at most this many image workers.
const MAX_HEAVY_WORKERS: usize = 8;

/// A decoded frame travelling from decoder to encoder worker. `charged` is the
/// byte amount this buffer holds against the pool budget until it is returned.
struct FrameJob {
    sequence: u64,
    pixels: Vec<u8>,
    charged: usize,
    width: u32,
    height: u32,
    format: DecodedPixelFormat,
}

fn frame_error(reason: impl Into<String>) -> PipelineError {
    PipelineError::DecodeFailed {
        path: PathBuf::new(),
        reason: reason.into(),
    }
}

/// Copies the valid columns/rows of a padded decoded frame into a cropped
/// destination. The padded stride is required by NV12 conversion but must not
/// leak into the emitted image dimensions.
fn copy_cropped_rows(
    src: &[u8],
    src_width: u32,
    dst: &mut [u8],
    dst_width: u32,
    height: u32,
    channels: usize,
) -> Result<(), PipelineError> {
    let overflow = || frame_error("frame row offset overflows usize");
    let src_w = usize::try_from(src_width).map_err(|_| overflow())?;
    let dst_w = usize::try_from(dst_width).map_err(|_| overflow())?;
    let rows = usize::try_from(height).map_err(|_| overflow())?;
    let row_bytes = dst_w.checked_mul(channels).ok_or_else(overflow)?;
    for row in 0..rows {
        let start = row
            .checked_mul(src_w)
            .and_then(|v| v.checked_mul(channels))
            .ok_or_else(overflow)?;
        let end = start.checked_add(row_bytes).ok_or_else(overflow)?;
        let dst_start = row.checked_mul(row_bytes).ok_or_else(overflow)?;
        let dst_end = dst_start.checked_add(row_bytes).ok_or_else(overflow)?;
        let src_row = src
            .get(start..end)
            .ok_or_else(|| frame_error("decoded frame row is out of bounds"))?;
        let dst_row = dst
            .get_mut(dst_start..dst_end)
            .ok_or_else(|| frame_error("cropped frame row is out of bounds"))?;
        dst_row.copy_from_slice(src_row);
    }
    Ok(())
}

fn row_bytes(width: u32, channels: usize) -> Result<usize, PipelineError> {
    usize::try_from(width)
        .ok()
        .and_then(|width| width.checked_mul(channels))
        .ok_or_else(|| frame_error("frame row stride overflows usize"))
}

/// Compacts a padded RGB buffer down to its cropped dimensions in place, so no
/// second raw-pixel allocation is needed for odd-sized decoded frames.
fn compact_cropped_rgb(
    rgb: &mut Vec<u8>,
    padded_width: u32,
    out_width: u32,
    out_height: u32,
) -> Result<(), PipelineError> {
    let overflow = || frame_error("cropped row offset overflows usize");
    let src_stride = row_bytes(padded_width, 3)?;
    let out_stride = row_bytes(out_width, 3)?;
    let rows = usize::try_from(out_height).map_err(|_| overflow())?;
    for row in 0..rows {
        let start = row.checked_mul(src_stride).ok_or_else(overflow)?;
        let end = start.checked_add(out_stride).ok_or_else(overflow)?;
        let target = row.checked_mul(out_stride).ok_or_else(overflow)?;
        rgb.copy_within(start..end, target);
    }
    let cropped = rows.checked_mul(out_stride).ok_or_else(overflow)?;
    rgb.truncate(cropped);
    Ok(())
}

/// Byte-bounded pool of raw decoded-pixel buffers. Admission is based on the
/// actual frame length at `take` time; pooled buffers remain charged so the
/// decoder, the job queue and all workers together can never exceed `limit`.
struct PixelPool {
    state: Mutex<PoolState>,
    changed: Condvar,
    limit: usize,
}

struct PoolState {
    /// Returned buffers with the byte amount charged for each.
    free: Vec<(Vec<u8>, usize)>,
    /// Sum of charges for every live buffer, including worker-held ones.
    live_bytes: usize,
}

impl PixelPool {
    fn new(limit: usize) -> Self {
        Self {
            state: Mutex::new(PoolState {
                free: Vec::new(),
                live_bytes: 0,
            }),
            changed: Condvar::new(),
            limit,
        }
    }

    /// Reserves `needed` bytes and returns a cleared buffer with at least that
    /// capacity, plus the amount charged to the budget for this buffer.
    fn take(&self, needed: usize) -> Result<(Vec<u8>, usize), PipelineError> {
        if needed > self.limit {
            return Err(frame_error(format!(
                "a frame needs {needed} pixel bytes, above the {}-byte budget",
                self.limit
            )));
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if let Some(pos) = state
                .free
                .iter()
                .position(|(buffer, _)| buffer.capacity() >= needed)
            {
                let (mut buffer, charged) = state.free.remove(pos);
                buffer.clear();
                buffer.resize(needed, 0);
                return Ok((buffer, charged));
            }
            let additional = state.live_bytes.saturating_add(needed);
            if additional <= self.limit {
                state.live_bytes = additional;
                return Ok((vec![0_u8; needed], needed));
            }
            // An idle buffer too small for this frame can be released; its
            // charge stays counted until this exact point.
            if let Some(pos) = state
                .free
                .iter()
                .position(|(buffer, _)| buffer.capacity() < needed)
            {
                let (buffer, charged) = state.free.remove(pos);
                drop(buffer);
                state.live_bytes = state.live_bytes.saturating_sub(charged);
                continue;
            }
            // Every remaining byte is in a queued or worker-owned buffer; a
            // returning worker notifies this wait.
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Returns a buffer to the pool; its bytes stay charged to the budget.
    fn release(&self, buffer: Vec<u8>, charged: usize) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.free.push((buffer, charged));
        drop(state);
        self.changed.notify_one();
    }
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
            FrameSelection::TargetFps(fps) => fps_interval_us(fps.get()),
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

type FillFn<'a> = &'a mut (dyn FnMut(&mut Vec<u8>, u32, u32) -> Result<(), PipelineError> + 'a);

/// Decode-side state: selection, byte budget, and job production.
struct FrameEmitter<'a> {
    selector: Selector,
    range: FrameRange,
    expected: Option<(u32, u32)>,
    pool: &'a PixelPool,
    job_tx: channel::Sender<FrameJob>,
    error_rx: &'a channel::Receiver<PipelineError>,
    failed: &'a AtomicBool,
    sequence: u64,
    last_dims: Option<(u32, u32)>,
}

impl FrameEmitter<'_> {
    /// Fills a budgeted buffer for one selected frame and queues it. `fill`
    /// receives the cropped output dimensions so padded decoders can trim the
    /// extra columns/rows before emission.
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
        reserve: Option<usize>,
        fill: FillFn<'_>,
    ) -> Result<(), PipelineError> {
        if !self.selector.should_emit(index, pts_us) {
            return Ok(());
        }
        if self.failed.load(Ordering::Relaxed) {
            return Err(PipelineError::WorkerPanicked);
        }
        // Padded decoded dimensions (e.g. odd-sized MJPEG) are trimmed to the
        // probed source size when that is smaller; never emit padding.
        let (out_width, out_height) = match self.expected {
            Some((w, h)) if w <= width && h <= height => (w, h),
            _ => (width, height),
        };
        let needed = frame_byte_len(out_width, out_height, channels).map_err(|_| {
            PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: "frame dimensions overflow the pixel buffer".to_owned(),
            }
        })?;
        // A padded decoder reserves its larger frame size but compacts to the
        // cropped length inside `fill`; both sizes stay charged to the same pool.
        let reserved = reserve.unwrap_or(needed).max(needed);
        let (mut pixels, charged) = self.pool.take(reserved)?;
        if let Err(err) = fill(&mut pixels, out_width, out_height) {
            self.pool.release(pixels, charged);
            return Err(err);
        }
        if pixels.len() != needed {
            let actual = pixels.len();
            self.pool.release(pixels, charged);
            return Err(frame_error(format!(
                "frame buffer is {actual} bytes, expected {needed} after cropping"
            )));
        }
        self.sequence = self.sequence.saturating_add(1);
        self.job_tx
            .send(FrameJob {
                sequence: self.sequence,
                pixels,
                charged,
                width: out_width,
                height: out_height,
                format,
            })
            .map_err(|_| {
                self.error_rx
                    .try_recv()
                    .unwrap_or(PipelineError::WorkerPanicked)
            })?;
        self.last_dims = Some((out_width, out_height));
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
        let at_end = emitter.range.end() == Some(index);
        if emitter.range.is_past_end(index) {
            return Ok(ControlFlow::Break(()));
        }
        if emitter.range.contains(index) {
            emitter.emit(
                source,
                index,
                info.pts_us,
                info.width,
                info.height,
                DecodedPixelFormat::Rgba8,
                4,
                None,
                &mut |dst, out_width, out_height| {
                    copy_cropped_rows(pixels, info.width, &mut dst[..], out_width, out_height, 4)
                },
            )?;
        }
        // Stop on the inclusive end frame itself: decoding another frame just
        // to observe `index > end` would fail on a corrupt trailing frame that
        // was never requested.
        if at_end {
            return Ok(ControlFlow::Break(()));
        }
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
        let at_end = emitter.range.end() == Some(index);
        if emitter.range.is_past_end(index) {
            return Ok(ControlFlow::Break(()));
        }
        if emitter.range.contains(index) {
            // MJPEG rounds odd dimensions up to even for NV12. Reserve the
            // padded RGB bytes from the same pool as the emitted frame, then
            // compact in place: no second raw-pixel allocation escapes the cap.
            let padded = frame_byte_len(info.width, info.height, 3)?;
            emitter.emit(
                source,
                index,
                info.pts_us,
                info.width,
                info.height,
                DecodedPixelFormat::Rgb8,
                3,
                Some(padded),
                &mut |dst, out_width, out_height| {
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
                    nv12_to_rgb8_into(
                        &nv12,
                        &mut dst[..padded],
                        pitch,
                        info.width,
                        info.height,
                        info.color,
                    )?;
                    if (info.width, info.height) != (out_width, out_height) {
                        compact_cropped_rgb(dst, info.width, out_width, out_height)?;
                    }
                    Ok(())
                },
            )?;
        }
        if at_end {
            return Ok(ControlFlow::Break(()));
        }
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
    // Breaking early (past `--end`) is an expected optimisation. This discards
    // only a `#[must_use]` flow marker, never an error or a fallback.
    let _ = flow;
    Ok(())
}

struct WorkerContext<'a> {
    job_rx: &'a channel::Receiver<FrameJob>,
    pool: &'a PixelPool,
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
    let pool = ctx.pool;
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
            pool.release(job.pixels, job.charged);
            continue;
        }
        let target = output_dir.join(format!("frame_{:06}.{extension}", job.sequence));
        if target == source {
            // The probe boundary rejects this layout first; keep a lexical
            // backstop so no caller can publish a frame over its own source.
            let err = PipelineError::EncodeFailed {
                path: target,
                reason: "frame target would overwrite the extraction source".to_owned(),
            };
            failed.store(true, Ordering::Relaxed);
            if error_tx.send(err).is_err() {
                break;
            }
            pool.release(job.pixels, job.charged);
            continue;
        }
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
        pool.release(job.pixels, job.charged);
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
    // The aggregate byte budget, not a slot count, is the memory limiter: the
    // pool charges every live buffer regardless of where it currently sits.
    let pool = PixelPool::new(MAX_IN_FLIGHT_BYTES);
    let (job_tx, job_rx) = channel::unbounded::<FrameJob>();
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
                    pool: &pool,
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
        let expected = plan
            .source_dimensions()
            .map(|dims| (dims.width().get(), dims.height().get()));
        let mut emitter = FrameEmitter {
            selector: Selector::new(plan.selection(), plan.range().start()),
            range: plan.range(),
            expected,
            pool: &pool,
            job_tx,
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
    use super::{
        FrameExtractionPlan, PipelineError, compact_cropped_rgb, copy_cropped_rows,
        execute_frame_extraction,
    };
    use crate::video_anim::{AnimFrameInfo, encode_animation_stream};
    use crate::video_codecs::{
        Yuv420Frame, YuvColor, decode_video_track_streaming_ctrl, encode_video_cpu,
    };
    use crate::video_container::{ContainerMedia, demux_video, mux_video};
    use fastconv_domain::{
        AnimationLoop, FrameRange, FrameSelection, ImageDimensions, ImageExecutionMode,
        ImageOutputFormat, VideoContainerFormat, VideoEncodeMode, VideoQuality, VideoRateControl,
    };
    use std::env;
    use std::error::Error;
    use std::fs::{self, File};
    use std::io::BufReader;
    use std::num::{NonZeroU32, NonZeroUsize};
    use std::ops::ControlFlow;
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

    #[test]
    fn copy_cropped_rows_trims_padded_columns() -> Result<(), Box<dyn Error>> {
        let padded = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let mut cropped = [0_u8; 6];
        copy_cropped_rows(&padded, 4, &mut cropped, 3, 2, 1)?;
        assert_eq!(cropped, [1, 2, 3, 5, 6, 7]);

        // RGBA rows with a padding column and row are trimmed on both axes.
        let rgba = [1_u8, 2, 3, 4, 9, 9, 9, 9, 5, 6, 7, 8, 9, 9, 9, 9];
        let mut out = [0_u8; 8];
        copy_cropped_rows(&rgba, 2, &mut out, 1, 2, 4)?;
        assert_eq!(out, [1, 2, 3, 4, 5, 6, 7, 8]);
        Ok(())
    }

    #[test]
    fn compact_cropped_rgb_trims_padded_rows_in_place() -> Result<(), Box<dyn Error>> {
        // Padded 4x2 RGB buffer (24 bytes) cropped to 3x2 (18 bytes).
        let mut rgb: Vec<u8> = (1_u8..=24).collect();
        compact_cropped_rgb(&mut rgb, 4, 3, 2)?;
        assert_eq!(
            rgb,
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21
            ]
        );
        Ok(())
    }

    #[test]
    fn ctrl_decode_stops_after_callback_break() -> Result<(), Box<dyn Error>> {
        let root = temp_root("ctrl-stop")?;
        let source = root.join("in.mp4");
        write_video(&source, VideoContainerFormat::Mp4, &solid_frames(3, 32, 32))?;
        let media = demux_video(&source, VideoContainerFormat::Mp4)?;
        let track_index = media
            .tracks
            .iter()
            .position(|track| track.codec.is_video())
            .ok_or("no video track")?;
        let packets: Vec<_> = media
            .packets
            .iter()
            .filter(|packet| packet.track == track_index)
            .cloned()
            .collect();
        let mut seen = 0_u64;
        let flow = decode_video_track_streaming_ctrl(
            &source,
            &media.tracks[track_index],
            &packets,
            |_info, _pack| {
                seen = seen.saturating_add(1);
                Ok(ControlFlow::Break(()))
            },
        )?;
        assert_eq!(seen, 1, "decode must stop at the callback break");
        assert!(matches!(flow, ControlFlow::Break(())));
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn end_zero_extracts_only_the_first_frame() -> Result<(), Box<dyn Error>> {
        let root = temp_root("end-zero")?;
        let source = root.join("in.mp4");
        write_video(&source, VideoContainerFormat::Mp4, &solid_frames(3, 32, 32))?;
        let out = root.join("png");
        let report = execute_frame_extraction(&plan(
            &source,
            &out,
            VideoContainerFormat::Mp4,
            ImageOutputFormat::Png,
            FrameSelection::All,
            FrameRange::new(0, Some(0))?,
        )?)?;
        assert_eq!(report.frames_extracted().get(), 1);
        assert!(out.join("frame_000001.png").is_file());
        assert!(!out.join("frame_000002.png").exists());
        fs::remove_dir_all(&root)?;
        Ok(())
    }
}
