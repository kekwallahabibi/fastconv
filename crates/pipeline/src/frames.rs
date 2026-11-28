//! Streaming still-frame extraction with bounded, recycled pixel buffers.
//!
//! Frames are never accumulated in memory: the decode thread reserves all
//! extraction-pipeline raw pixel buffers (NV12 conversion scratch, GIF
//! sub-frame decode scratch and canvas holds, WebP frame holds, and emitted
//! RGB/RGBA frame buffers) from one byte-bounded [`PixelPool`] and hands each
//! job to a pool of encoder workers. Workers return their buffer to the pool on
//! drop, so aggregate extraction-pipeline pixel bytes never exceed
//! [`MAX_IN_FLIGHT_BYTES`] (256 MiB). Codec-internal decoder state (such as
//! H.264/HEVC/AV1/VP9 reference DPBs and MJPEG's bounded RGB scratch) is capped
//! separately by each codec decoder.

use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
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

/// Hard upper bound on the *aggregate* raw pixel bytes held by the extraction
/// pipeline: NV12 conversion scratch, GIF sub-frame scratch + canvas holds,
/// WebP frame holds, queued frames, worker-held frames, and pooled buffers kept
/// for reuse all count against this single 256 MiB budget.
const MAX_IN_FLIGHT_BYTES: usize = 256 * 1024 * 1024;
/// Memory-hungry 4K still encoders use at most this many image workers.
const MAX_HEAVY_WORKERS: usize = 8;

/// A decoded frame travelling from decoder to encoder worker. Dropping
/// `pixels` automatically returns its buffer to the pool.
struct FrameJob<'pool> {
    sequence: u64,
    pixels: PooledBuffer<'pool>,
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

fn attach_path(err: PipelineError, source: &Path) -> PipelineError {
    match err {
        PipelineError::DecodeFailed { path, reason } if path.as_os_str().is_empty() => {
            PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason,
            }
        }
        other => other,
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

/// Trims only even-alignment padding (at most one extra column/row) from a
/// decoded frame. A frame larger than the probed size is real content, not
/// padding, so it is emitted at its decoded size instead of being truncated.
fn output_dimensions(decoded: (u32, u32), probed: Option<(u32, u32)>) -> (u32, u32) {
    match probed {
        Some((width, height))
            if decoded.0 >= width
                && decoded.1 >= height
                && decoded.0 - width <= 1
                && decoded.1 - height <= 1 =>
        {
            (width, height)
        }
        _ => decoded,
    }
}

/// Compacts a padded RGB buffer down to its cropped dimensions in place,
/// returning the cropped byte length so no second raw-pixel allocation is
/// needed for odd-sized decoded frames.
fn compact_cropped_rgb(
    rgb: &mut [u8],
    padded_width: u32,
    out_width: u32,
    out_height: u32,
) -> Result<usize, PipelineError> {
    let overflow = || frame_error("cropped row offset overflows usize");
    if out_width > padded_width {
        return Err(frame_error("cropped width exceeds padded frame width"));
    }
    let src_stride = row_bytes(padded_width, 3)?;
    let out_stride = row_bytes(out_width, 3)?;
    let rows = usize::try_from(out_height).map_err(|_| overflow())?;
    let required = rows.checked_mul(src_stride).ok_or_else(overflow)?;
    if rgb.len() < required {
        return Err(frame_error("padded RGB buffer is shorter than expected"));
    }
    for row in 0..rows {
        let start = row.checked_mul(src_stride).ok_or_else(overflow)?;
        let end = start.checked_add(out_stride).ok_or_else(overflow)?;
        let target = row.checked_mul(out_stride).ok_or_else(overflow)?;
        rgb.copy_within(start..end, target);
    }
    rows.checked_mul(out_stride).ok_or_else(overflow)
}

/// Byte-bounded pool of raw decoded-pixel buffers.
///
/// Every live raw pixel buffer (decoder NV12 scratch, animation canvas holds,
/// queued frame jobs, worker-held buffers, and idle recycled buffers) counts
/// against `limit`. Multi-buffer admission (`take_pair`) reserves both buffers
/// atomically under the mutex, so no caller ever holds one reservation while
/// blocking for a second and deadlock is impossible.
struct PixelPool {
    state: Mutex<PoolState>,
    changed: Condvar,
    limit: usize,
}

struct PoolState {
    /// Idle buffers available for reuse. Each buffer's byte charge is its
    /// exact `capacity()`.
    free: Vec<Vec<u8>>,
    /// Bytes currently checked out via active `PooledBuffer` leases.
    leased_bytes: usize,
    /// Bytes held by live decoder-owned buffers (e.g. animation canvas) that
    /// workers cannot release.
    external_bytes: usize,
    /// Sum of `Vec::capacity()` for all buffers currently sitting in `free`.
    free_bytes: usize,
    /// Set on worker error/exit so a waiting producer wakes immediately.
    closed: bool,
    /// Peak observed aggregate `leased_bytes + external_bytes + free_bytes`.
    peak_live_bytes: usize,
    /// Producers currently blocked in the admission wait. Lets tests confirm
    /// a waiter is truly parked before exercising the wake-up path.
    waiters: usize,
}

impl PoolState {
    fn live_bytes(&self) -> usize {
        self.leased_bytes
            .saturating_add(self.external_bytes)
            .saturating_add(self.free_bytes)
    }

    fn record_peak(&mut self) {
        let live = self.live_bytes();
        if live > self.peak_live_bytes {
            self.peak_live_bytes = live;
        }
    }

    /// Moves idle buffers into `evicted` (smallest first) until
    /// `self.live_bytes() + additional <= limit`.
    fn evict_free_for(&mut self, additional: usize, limit: usize, evicted: &mut Vec<Vec<u8>>) {
        while self.live_bytes().saturating_add(additional) > limit {
            let Some((pos, _)) = self
                .free
                .iter()
                .enumerate()
                .min_by_key(|(_, buf)| buf.capacity())
            else {
                break;
            };
            let buf = self.free.swap_remove(pos);
            self.free_bytes = self.free_bytes.saturating_sub(buf.capacity());
            evicted.push(buf);
        }
    }
}

/// Best-fit index in `free`: smallest buffer with `capacity() >= needed`,
/// excluding `exclude`. Zero-byte requests never claim a non-empty buffer.
fn best_fit(free: &[Vec<u8>], needed: usize, exclude: Option<usize>) -> Option<usize> {
    if needed == 0 {
        return None;
    }
    free.iter()
        .enumerate()
        .filter(|(idx, buf)| Some(*idx) != exclude && buf.capacity() >= needed)
        .min_by_key(|(_, buf)| buf.capacity())
        .map(|(idx, _)| idx)
}

fn alloc_zeroed(len: usize) -> Result<Vec<u8>, PipelineError> {
    let mut buf = Vec::new();
    buf.try_reserve_exact(len)
        .map_err(|_| frame_error(format!("failed to allocate {len}-byte pixel buffer")))?;
    buf.resize(len, 0);
    Ok(buf)
}

fn reset_len(buf: &mut Vec<u8>, len: usize) {
    buf.clear();
    buf.resize(len, 0);
}

/// RAII lease for one pooled raw-pixel buffer. Callers can read, mutate in
/// place, or truncate, but cannot grow the backing `Vec` past its charged
/// capacity. Dropping returns the buffer to the pool automatically.
struct PooledBuffer<'pool> {
    pool: &'pool PixelPool,
    pixels: Vec<u8>,
    charged: usize,
}

impl PooledBuffer<'_> {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.pixels
    }

    fn len(&self) -> usize {
        self.pixels.len()
    }

    fn truncate(&mut self, len: usize) {
        self.pixels.truncate(len);
    }
}

impl Drop for PooledBuffer<'_> {
    fn drop(&mut self) {
        if self.charged > 0 {
            let pixels = std::mem::take(&mut self.pixels);
            let charged = self.charged;
            self.charged = 0;
            self.pool.release(pixels, charged);
        }
    }
}

/// RAII accounting hold for a live external decoder pixel buffer (such as an
/// animation canvas) so it shares the exact same aggregate byte budget.
struct ExternalHold<'pool> {
    pool: &'pool PixelPool,
    bytes: usize,
}

impl ExternalHold<'_> {
    fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for ExternalHold<'_> {
    fn drop(&mut self) {
        if self.bytes > 0 {
            let bytes = self.bytes;
            self.bytes = 0;
            self.pool.release_external(bytes);
        }
    }
}

/// Rolls back pre-reserved `leased_bytes` if an allocation outside the lock
/// fails (e.g. OOM) before constructing a `PooledBuffer`.
struct LeaseReservation<'pool> {
    pool: &'pool PixelPool,
    bytes: usize,
}

impl LeaseReservation<'_> {
    fn disarm(&mut self) {
        self.bytes = 0;
    }
}

impl Drop for LeaseReservation<'_> {
    fn drop(&mut self) {
        if self.bytes > 0 {
            let mut state = self.pool.lock_state();
            state.leased_bytes = state.leased_bytes.saturating_sub(self.bytes);
            drop(state);
            self.pool.changed.notify_all();
        }
    }
}

enum SinglePlan {
    Reuse(Vec<u8>, usize),
    Alloc(usize),
}

enum PairPlan {
    ReuseBoth(Vec<u8>, usize, Vec<u8>, usize),
    ReuseFirstAllocSecond {
        first: Vec<u8>,
        first_cap: usize,
        second_len: usize,
    },
    AllocFirstReuseSecond {
        first_len: usize,
        second: Vec<u8>,
        second_cap: usize,
    },
    AllocBoth {
        first_len: usize,
        second_len: usize,
    },
}

impl PixelPool {
    fn new(limit: usize) -> Self {
        Self {
            state: Mutex::new(PoolState {
                free: Vec::new(),
                leased_bytes: 0,
                external_bytes: 0,
                free_bytes: 0,
                closed: false,
                peak_live_bytes: 0,
                waiters: 0,
            }),
            changed: Condvar::new(),
            limit,
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, PoolState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Blocks until a budget change, advertising the parked producer so tests
    /// can synchronize on real admission instead of a timing guess.
    fn wait_for_change<'a>(
        &self,
        mut state: MutexGuard<'a, PoolState>,
    ) -> MutexGuard<'a, PoolState> {
        state.waiters = state.waiters.saturating_add(1);
        state = self
            .changed
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.waiters = state.waiters.saturating_sub(1);
        state
    }

    /// Marks the pool closed and wakes any blocked admission waiter so worker
    /// failures or panics can never leave the decoder hanging.
    fn close(&self) {
        let mut state = self.lock_state();
        state.closed = true;
        drop(state);
        self.changed.notify_all();
    }

    #[cfg(test)]
    fn peak_live_bytes(&self) -> usize {
        self.lock_state().peak_live_bytes
    }

    #[cfg(test)]
    fn live_bytes(&self) -> usize {
        self.lock_state().live_bytes()
    }

    #[cfg(test)]
    fn leased_bytes(&self) -> usize {
        self.lock_state().leased_bytes
    }

    #[cfg(test)]
    fn waiters(&self) -> usize {
        self.lock_state().waiters
    }

    /// Reserves `bytes` of external decoder memory (e.g. a live animation
    /// canvas) against the same aggregate budget without allocating a buffer.
    fn reserve_external(&self, bytes: usize) -> Result<ExternalHold<'_>, PipelineError> {
        if bytes == 0 {
            return Ok(ExternalHold {
                pool: self,
                bytes: 0,
            });
        }
        if bytes > self.limit {
            return Err(frame_error(format!(
                "decoder pixel canvas needs {bytes} bytes, above the {}-byte budget",
                self.limit
            )));
        }
        let mut evicted = Vec::new();
        let mut state = self.lock_state();
        loop {
            if state.closed {
                return Err(PipelineError::WorkerPanicked);
            }
            if state
                .external_bytes
                .checked_add(bytes)
                .is_none_or(|sum| sum > self.limit)
            {
                return Err(frame_error(format!(
                    "decoder pixel canvas needs {bytes} bytes alongside {} live decoder bytes, above the {}-byte budget",
                    state.external_bytes, self.limit
                )));
            }
            let min_live = state
                .leased_bytes
                .checked_add(state.external_bytes)
                .and_then(|v| v.checked_add(bytes));
            if min_live.is_some_and(|v| v <= self.limit) {
                state.evict_free_for(bytes, self.limit, &mut evicted);
                state.external_bytes = state.external_bytes.saturating_add(bytes);
                state.record_peak();
                break;
            }
            state = self.wait_for_change(state);
        }
        drop(state);
        drop(evicted);
        Ok(ExternalHold { pool: self, bytes })
    }

    fn release_external(&self, bytes: usize) {
        let mut state = self.lock_state();
        state.external_bytes = state.external_bytes.saturating_sub(bytes);
        drop(state);
        self.changed.notify_all();
    }

    /// Reserves one buffer of `needed` bytes from the aggregate budget.
    fn take(&self, needed: usize) -> Result<PooledBuffer<'_>, PipelineError> {
        if needed == 0 {
            return Ok(PooledBuffer {
                pool: self,
                pixels: Vec::new(),
                charged: 0,
            });
        }
        if needed > self.limit {
            return Err(frame_error(format!(
                "a frame needs {needed} pixel bytes, above the {}-byte budget",
                self.limit
            )));
        }
        let mut evicted = Vec::new();
        let mut state = self.lock_state();
        let plan = loop {
            if state.closed {
                return Err(PipelineError::WorkerPanicked);
            }
            if state
                .external_bytes
                .checked_add(needed)
                .is_none_or(|sum| sum > self.limit)
            {
                return Err(frame_error(format!(
                    "a frame needs {needed} pixel bytes alongside {} decoder bytes, above the {}-byte budget",
                    state.external_bytes, self.limit
                )));
            }
            if let Some(pos) = best_fit(&state.free, needed, None) {
                let buf = state.free.swap_remove(pos);
                let cap = buf.capacity();
                state.free_bytes = state.free_bytes.saturating_sub(cap);
                state.leased_bytes = state.leased_bytes.saturating_add(cap);
                state.record_peak();
                break SinglePlan::Reuse(buf, cap);
            }
            let min_live = state
                .leased_bytes
                .checked_add(state.external_bytes)
                .and_then(|v| v.checked_add(needed));
            if min_live.is_some_and(|v| v <= self.limit) {
                state.evict_free_for(needed, self.limit, &mut evicted);
                state.leased_bytes = state.leased_bytes.saturating_add(needed);
                state.record_peak();
                break SinglePlan::Alloc(needed);
            }
            state = self.wait_for_change(state);
        };
        drop(state);
        // Free evicted buffers before allocating so physical heap never spikes
        // above `self.limit`.
        drop(evicted);

        match plan {
            SinglePlan::Reuse(mut pixels, charged) => {
                reset_len(&mut pixels, needed);
                Ok(PooledBuffer {
                    pool: self,
                    pixels,
                    charged,
                })
            }
            SinglePlan::Alloc(len) => {
                let mut reservation = LeaseReservation {
                    pool: self,
                    bytes: len,
                };
                let pixels = alloc_zeroed(len)?;
                reservation.disarm();
                Ok(PooledBuffer {
                    pool: self,
                    pixels,
                    charged: len,
                })
            }
        }
    }

    /// Atomically reserves two buffers (`first_len` and `second_len`) under
    /// the same aggregate budget. Rejects requests whose combined footprint
    /// cannot fit in `limit` without waiting, and never holds one buffer while
    /// blocking for the second, making deadlock impossible.
    fn take_pair(
        &self,
        first_len: usize,
        second_len: usize,
    ) -> Result<(PooledBuffer<'_>, PooledBuffer<'_>), PipelineError> {
        let total = first_len
            .checked_add(second_len)
            .ok_or_else(|| frame_error("combined frame pixel buffers overflow usize"))?;
        if total > self.limit {
            return Err(frame_error(format!(
                "a frame needs {total} pixel bytes ({first_len} + {second_len}), above the {}-byte budget",
                self.limit
            )));
        }
        let mut evicted = Vec::new();
        let mut state = self.lock_state();
        let plan = loop {
            if state.closed {
                return Err(PipelineError::WorkerPanicked);
            }
            if state
                .external_bytes
                .checked_add(total)
                .is_none_or(|sum| sum > self.limit)
            {
                return Err(frame_error(format!(
                    "a frame needs {total} pixel bytes alongside {} decoder bytes, above the {}-byte budget",
                    state.external_bytes, self.limit
                )));
            }
            if let Some(plan) =
                self.try_plan_pair(&mut state, first_len, second_len, total, &mut evicted)
            {
                break plan;
            }
            state = self.wait_for_change(state);
        };
        drop(state);
        drop(evicted);
        self.materialize_pair(plan, first_len, second_len, total)
    }

    fn try_plan_pair(
        &self,
        state: &mut PoolState,
        first_len: usize,
        second_len: usize,
        total: usize,
        evicted: &mut Vec<Vec<u8>>,
    ) -> Option<PairPlan> {
        // Match the larger request first so a smaller scratch slot (NV12)
        // never steals a larger frame buffer (RGB) from `free`.
        let (first_idx, second_idx) = if first_len >= second_len {
            let fi = best_fit(&state.free, first_len, None);
            let si = best_fit(&state.free, second_len, fi);
            (fi, si)
        } else {
            let si = best_fit(&state.free, second_len, None);
            let fi = best_fit(&state.free, first_len, si);
            (fi, si)
        };
        match (first_idx, second_idx) {
            (Some(fi), Some(si)) => {
                let (first, second) = if fi > si {
                    let f = state.free.swap_remove(fi);
                    let s = state.free.swap_remove(si);
                    (f, s)
                } else {
                    let s = state.free.swap_remove(si);
                    let f = state.free.swap_remove(fi);
                    (f, s)
                };
                let fc = first.capacity();
                let sc = second.capacity();
                let sum = fc.saturating_add(sc);
                state.free_bytes = state.free_bytes.saturating_sub(sum);
                state.leased_bytes = state.leased_bytes.saturating_add(sum);
                state.record_peak();
                return Some(PairPlan::ReuseBoth(first, fc, second, sc));
            }
            (Some(fi), None) => {
                let fc = state.free[fi].capacity();
                let min_live = state
                    .leased_bytes
                    .checked_add(state.external_bytes)
                    .and_then(|v| v.checked_add(fc))
                    .and_then(|v| v.checked_add(second_len));
                if min_live.is_some_and(|v| v <= self.limit) {
                    let first = state.free.swap_remove(fi);
                    state.free_bytes = state.free_bytes.saturating_sub(fc);
                    state.leased_bytes = state.leased_bytes.saturating_add(fc);
                    state.evict_free_for(second_len, self.limit, evicted);
                    state.leased_bytes = state.leased_bytes.saturating_add(second_len);
                    state.record_peak();
                    return Some(PairPlan::ReuseFirstAllocSecond {
                        first,
                        first_cap: fc,
                        second_len,
                    });
                }
            }
            (None, Some(si)) => {
                let sc = state.free[si].capacity();
                let min_live = state
                    .leased_bytes
                    .checked_add(state.external_bytes)
                    .and_then(|v| v.checked_add(sc))
                    .and_then(|v| v.checked_add(first_len));
                if min_live.is_some_and(|v| v <= self.limit) {
                    let second = state.free.swap_remove(si);
                    state.free_bytes = state.free_bytes.saturating_sub(sc);
                    state.leased_bytes = state.leased_bytes.saturating_add(sc);
                    state.evict_free_for(first_len, self.limit, evicted);
                    state.leased_bytes = state.leased_bytes.saturating_add(first_len);
                    state.record_peak();
                    return Some(PairPlan::AllocFirstReuseSecond {
                        first_len,
                        second,
                        second_cap: sc,
                    });
                }
            }
            (None, None) => {}
        }
        // Fallback: either no free buffer matched, or an oversized free buffer
        // would push `live_bytes` past `limit`. Evicting free buffers and
        // allocating exact sizes guarantees admission whenever
        // `leased_bytes + external_bytes + total <= limit`.
        let min_live = state
            .leased_bytes
            .checked_add(state.external_bytes)
            .and_then(|v| v.checked_add(total));
        if min_live.is_some_and(|v| v <= self.limit) {
            state.evict_free_for(total, self.limit, evicted);
            state.leased_bytes = state.leased_bytes.saturating_add(total);
            state.record_peak();
            return Some(PairPlan::AllocBoth {
                first_len,
                second_len,
            });
        }
        None
    }

    fn materialize_pair(
        &self,
        plan: PairPlan,
        first_len: usize,
        second_len: usize,
        total: usize,
    ) -> Result<(PooledBuffer<'_>, PooledBuffer<'_>), PipelineError> {
        match plan {
            PairPlan::ReuseBoth(mut first, fc, mut second, sc) => {
                reset_len(&mut first, first_len);
                reset_len(&mut second, second_len);
                Ok((
                    PooledBuffer {
                        pool: self,
                        pixels: first,
                        charged: fc,
                    },
                    PooledBuffer {
                        pool: self,
                        pixels: second,
                        charged: sc,
                    },
                ))
            }
            PairPlan::ReuseFirstAllocSecond {
                first,
                first_cap,
                second_len,
            } => {
                let mut first_buf = PooledBuffer {
                    pool: self,
                    pixels: first,
                    charged: first_cap,
                };
                let mut reservation = LeaseReservation {
                    pool: self,
                    bytes: second_len,
                };
                reset_len(&mut first_buf.pixels, first_len);
                let second = alloc_zeroed(second_len)?;
                reservation.disarm();
                Ok((
                    first_buf,
                    PooledBuffer {
                        pool: self,
                        pixels: second,
                        charged: second_len,
                    },
                ))
            }
            PairPlan::AllocFirstReuseSecond {
                first_len,
                second,
                second_cap,
            } => {
                let mut second_buf = PooledBuffer {
                    pool: self,
                    pixels: second,
                    charged: second_cap,
                };
                let mut reservation = LeaseReservation {
                    pool: self,
                    bytes: first_len,
                };
                reset_len(&mut second_buf.pixels, second_len);
                let first = alloc_zeroed(first_len)?;
                reservation.disarm();
                Ok((
                    PooledBuffer {
                        pool: self,
                        pixels: first,
                        charged: first_len,
                    },
                    second_buf,
                ))
            }
            PairPlan::AllocBoth {
                first_len,
                second_len,
            } => {
                let mut reservation = LeaseReservation {
                    pool: self,
                    bytes: total,
                };
                let first = alloc_zeroed(first_len)?;
                let second = alloc_zeroed(second_len)?;
                reservation.disarm();
                Ok((
                    PooledBuffer {
                        pool: self,
                        pixels: first,
                        charged: first_len,
                    },
                    PooledBuffer {
                        pool: self,
                        pixels: second,
                        charged: second_len,
                    },
                ))
            }
        }
    }

    /// Returns a leased buffer to the pool (or drops it if the pool is closed
    /// or its backing capacity drifted from `charged`).
    fn release(&self, buffer: Vec<u8>, charged: usize) {
        let mut discard = None;
        let mut state = self.lock_state();
        state.leased_bytes = state.leased_bytes.saturating_sub(charged);
        if state.closed || buffer.capacity() != charged {
            discard = Some(buffer);
        } else {
            state.free_bytes = state.free_bytes.saturating_add(charged);
            state.free.push(buffer);
        }
        drop(state);
        drop(discard);
        self.changed.notify_all();
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

type FillFn<'a> =
    &'a mut (dyn FnMut(&mut PooledBuffer<'_>, u32, u32) -> Result<(), PipelineError> + 'a);
type ScratchFillFn<'a> = &'a mut (
            dyn FnMut(&mut PooledBuffer<'_>, &mut [u8], u32, u32) -> Result<(), PipelineError> + 'a
        );

/// Decode-side state: selection, byte budget, and job production.
struct FrameEmitter<'ctx, 'pool> {
    selector: Selector,
    range: FrameRange,
    expected: Option<(u32, u32)>,
    pool: &'pool PixelPool,
    job_tx: channel::Sender<FrameJob<'pool>>,
    error_rx: &'ctx channel::Receiver<PipelineError>,
    failed: &'ctx AtomicBool,
    sequence: u64,
    last_dims: Option<(u32, u32)>,
}

impl<'pool> FrameEmitter<'_, 'pool> {
    fn check_worker_health(&self) -> Result<(), PipelineError> {
        if self.failed.load(Ordering::Acquire) {
            return Err(self
                .error_rx
                .try_recv()
                .unwrap_or(PipelineError::WorkerPanicked));
        }
        Ok(())
    }

    fn enqueue(
        &mut self,
        pixels: PooledBuffer<'pool>,
        expected_len: usize,
        width: u32,
        height: u32,
        format: DecodedPixelFormat,
    ) -> Result<(), PipelineError> {
        if pixels.len() != expected_len {
            let actual = pixels.len();
            return Err(frame_error(format!(
                "frame buffer is {actual} bytes, expected {expected_len} after cropping"
            )));
        }
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

    /// Fills a budgeted buffer for one selected frame and queues it. `fill`
    /// receives the cropped output dimensions so decoders can trim alignment
    /// padding before emission.
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
        fill: FillFn<'_>,
    ) -> Result<(), PipelineError> {
        // A worker failure must be reported even when every remaining frame is
        // skipped (e.g. `--every` with a failing first frame): skipped frames
        // make no pool reservation, so only this check can surface the error.
        self.check_worker_health()?;
        if !self.selector.should_emit(index, pts_us) {
            return Ok(());
        }
        let (out_width, out_height) = output_dimensions((width, height), self.expected);
        let needed = frame_byte_len(out_width, out_height, channels)
            .map_err(|err| attach_path(err, source))?;
        let mut pixels = self
            .pool
            .take(needed)
            .map_err(|err| attach_path(err, source))?;
        fill(&mut pixels, out_width, out_height)?;
        self.enqueue(pixels, needed, out_width, out_height, format)
    }

    /// Atomically reserves both an output pixel buffer (`reserve` bytes) and a
    /// decode scratch buffer (`scratch_len` bytes) from the same `PixelPool`,
    /// converts the frame, immediately returns the scratch buffer to the pool,
    /// and queues the compacted output buffer.
    #[allow(clippy::too_many_arguments)]
    fn emit_with_scratch(
        &mut self,
        source: &Path,
        index: u64,
        pts_us: i64,
        width: u32,
        height: u32,
        format: DecodedPixelFormat,
        channels: usize,
        reserve: usize,
        scratch_len: usize,
        fill: ScratchFillFn<'_>,
    ) -> Result<(), PipelineError> {
        // A worker failure must be reported even when every remaining frame is
        // skipped (e.g. `--every` with a failing first frame): skipped frames
        // make no pool reservation, so only this check can surface the error.
        self.check_worker_health()?;
        if !self.selector.should_emit(index, pts_us) {
            return Ok(());
        }
        let (out_width, out_height) = output_dimensions((width, height), self.expected);
        let needed = frame_byte_len(out_width, out_height, channels)
            .map_err(|err| attach_path(err, source))?;
        let reserved = reserve.max(needed);
        let (mut pixels, mut scratch) = self
            .pool
            .take_pair(reserved, scratch_len)
            .map_err(|err| attach_path(err, source))?;
        fill(&mut pixels, scratch.as_mut_slice(), out_width, out_height)?;
        // Return decode scratch immediately so its bytes can be reused or
        // evicted while encoder workers process queued frames.
        drop(scratch);
        self.enqueue(pixels, needed, out_width, out_height, format)
    }
}

fn ensure_canvas_hold<'pool>(
    pool: &'pool PixelPool,
    source: &Path,
    slot: &mut Option<ExternalHold<'pool>>,
    needed_hold: usize,
) -> Result<(), PipelineError> {
    match slot {
        Some(hold) if hold.bytes() == needed_hold => Ok(()),
        slot => {
            // Release any prior hold before acquiring a resized one so we
            // never double-charge across a size change.
            *slot = None;
            *slot = Some(
                pool.reserve_external(needed_hold)
                    .map_err(|err| attach_path(err, source))?,
            );
            Ok(())
        }
    }
}

fn decode_animation(
    source: &Path,
    container: VideoContainerFormat,
    emitter: &mut FrameEmitter<'_, '_>,
) -> Result<ControlFlow<()>, PipelineError> {
    let pool = emitter.pool;
    // Reject canvases that cannot coexist with an emitted frame before
    // allocating them: GIF retains up to 2x canvas (active + `Previous`
    // backup) alongside a sub-frame/emitted frame (<= 2/3 of `pool.limit`),
    // while WebP holds 1x frame buffer alongside an emitted frame (<= 1/2).
    let max_gif_canvas_bytes = pool.limit.saturating_sub(pool.limit / 3);
    let max_webp_frame_bytes = pool.limit / 2;
    let mut frame_index = 0_u64;
    let mut webp_canvas_hold: Option<ExternalHold<'_>> = None;
    let mut emit_anim_frame = |info: crate::video_anim::AnimFrameInfo,
                               pixels: &[u8],
                               charge_hold: bool|
     -> Result<ControlFlow<()>, PipelineError> {
        let index = frame_index;
        frame_index = frame_index.saturating_add(1);
        let at_end = emitter.range.end() == Some(index);
        if emitter.range.is_past_end(index) {
            return Ok(ControlFlow::Break(()));
        }
        if emitter.range.contains(index) {
            let (format, channels) = match info.channels {
                3 => (DecodedPixelFormat::Rgb8, 3_usize),
                4 => (DecodedPixelFormat::Rgba8, 4_usize),
                other => {
                    return Err(PipelineError::DecodeFailed {
                        path: source.to_path_buf(),
                        reason: format!("unsupported animation pixel layout: {other} channels"),
                    });
                }
            };
            if charge_hold {
                ensure_canvas_hold(pool, source, &mut webp_canvas_hold, pixels.len())?;
            }
            emitter.emit(
                source,
                index,
                info.pts_us,
                info.width,
                info.height,
                format,
                channels,
                &mut |dst, out_width, out_height| {
                    copy_cropped_rows(
                        pixels,
                        info.width,
                        dst.as_mut_slice(),
                        out_width,
                        out_height,
                        channels,
                    )
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
            // GIF disposal `Previous` retains a backup canvas alongside the
            // active canvas; charge that second canvas only for those frames
            // (1x otherwise, released on the next provider call after the
            // previous frame was disposed). Sub-frame decode buffers come from
            // the same pool, so canvas + backup + sub-frame + queued/worker
            // frames all stay under the exact same aggregate budget.
            let mut gif_canvas_hold: Option<ExternalHold<'_>> = None;
            decode_gif_frames_streaming_pixels_ctrl(
                source,
                max_gif_canvas_bytes,
                |canvas_len, frame_buf_len, needs_backup, fill| {
                    let needed_hold = if needs_backup {
                        canvas_len
                            .checked_mul(2)
                            .ok_or_else(|| PipelineError::DecodeFailed {
                                path: source.to_path_buf(),
                                reason: "GIF canvas byte hold overflows usize".to_owned(),
                            })?
                    } else {
                        canvas_len
                    };
                    ensure_canvas_hold(pool, source, &mut gif_canvas_hold, needed_hold)?;
                    let mut frame_buf = pool
                        .take(frame_buf_len)
                        .map_err(|err| attach_path(err, source))?;
                    fill(frame_buf.as_mut_slice())
                },
                |info, pixels| emit_anim_frame(info, pixels, false),
            )
        }
        VideoContainerFormat::Webp => decode_webp_frames_streaming_pixels_ctrl(
            source,
            max_webp_frame_bytes,
            |info, pixels| emit_anim_frame(info, pixels, true),
        ),
        _ => Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "not an animation container".to_owned(),
        }),
    }
}

fn decode_video(
    source: &Path,
    container: VideoContainerFormat,
    emitter: &mut FrameEmitter<'_, '_>,
) -> Result<ControlFlow<()>, PipelineError> {
    let mut media = demux_video(source, container)?;
    let track_index = media
        .tracks
        .iter()
        .position(|track| track.codec.is_video())
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "source has no video track".to_owned(),
        })?;
    let track = media.tracks.swap_remove(track_index);
    let packets: Vec<_> = media
        .packets
        .into_iter()
        .filter(|packet| packet.track == track_index)
        .collect();
    let mut frame_index = 0_u64;
    decode_video_track_streaming_ctrl(source, &track, &packets, |info, pack| {
        let index = frame_index;
        frame_index = frame_index.saturating_add(1);
        let at_end = emitter.range.end() == Some(index);
        if emitter.range.is_past_end(index) {
            return Ok(ControlFlow::Break(()));
        }
        if emitter.range.contains(index) {
            // MJPEG rounds odd dimensions up to even for NV12. Reserve BOTH
            // the NV12 scratch and the padded RGB output atomically from the
            // same pool, convert, compact cropped rows in place, and release
            // NV12 immediately: no raw-pixel allocation escapes the budget.
            let padded_rgb = frame_byte_len(info.width, info.height, 3)
                .map_err(|err| attach_path(err, source))?;
            let pitch = usize::try_from(info.width).map_err(|_| PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: "NV12 pitch overflows usize".to_owned(),
            })?;
            let nv12_len = nv12_buffer_len(info.width, info.height, pitch)?;
            emitter.emit_with_scratch(
                source,
                index,
                info.pts_us,
                info.width,
                info.height,
                DecodedPixelFormat::Rgb8,
                3,
                padded_rgb,
                nv12_len,
                &mut |dst, nv12, out_width, out_height| {
                    pack(&mut nv12[..nv12_len], pitch)?;
                    nv12_to_rgb8_into(
                        &nv12[..nv12_len],
                        &mut dst.as_mut_slice()[..padded_rgb],
                        pitch,
                        info.width,
                        info.height,
                        info.color,
                    )?;
                    if (info.width, info.height) != (out_width, out_height) {
                        let cropped = compact_cropped_rgb(
                            dst.as_mut_slice(),
                            info.width,
                            out_width,
                            out_height,
                        )?;
                        dst.truncate(cropped);
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
    emitter: &mut FrameEmitter<'_, '_>,
) -> Result<(), PipelineError> {
    let flow = match plan.source_container() {
        VideoContainerFormat::Gif | VideoContainerFormat::Webp => {
            decode_animation(plan.source(), plan.source_container(), emitter)?
        }
        container => decode_video(plan.source(), container, emitter)?,
    };
    match flow {
        ControlFlow::Continue(()) | ControlFlow::Break(()) => Ok(()),
    }
}

/// Ensures that if a worker unwinds (or the last worker exits), `PixelPool` is
/// closed so the decode thread can never remain blocked in `take`/`take_pair`.
struct WorkerExitGuard<'pool> {
    active_workers: &'pool AtomicUsize,
    failed: &'pool AtomicBool,
    pool: &'pool PixelPool,
}

impl Drop for WorkerExitGuard<'_> {
    fn drop(&mut self) {
        let panicking = thread::panicking();
        let remaining = self.active_workers.fetch_sub(1, Ordering::AcqRel);
        if panicking {
            self.failed.store(true, Ordering::Release);
            self.pool.close();
        } else if remaining == 1 {
            self.pool.close();
        }
    }
}

struct WorkerContext<'ctx, 'pool> {
    job_rx: &'ctx channel::Receiver<FrameJob<'pool>>,
    pool: &'pool PixelPool,
    error_tx: &'ctx channel::Sender<PipelineError>,
    failed: &'ctx AtomicBool,
    total_bytes: &'ctx AtomicU64,
    bmp_write_lock: &'ctx Mutex<()>,
    encoder: &'ctx CpuImageEncoder,
    inner_threads: usize,
    source: &'ctx Path,
    output_dir: &'ctx Path,
    extension: &'ctx str,
}

fn run_worker(ctx: &WorkerContext<'_, '_>) {
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
        if failed.load(Ordering::Acquire) {
            // Dropping `job` returns its `PooledBuffer` immediately.
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
            let sent = error_tx.send(err).is_ok();
            failed.store(true, Ordering::Release);
            pool.close();
            if !sent {
                break;
            }
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
            job.pixels.as_mut_slice(),
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
                // Send the real error BEFORE closing the pool so the main
                // thread always observes it rather than `WorkerPanicked`.
                let sent = error_tx.send(err).is_ok();
                failed.store(true, Ordering::Release);
                pool.close();
                if !sent {
                    break;
                }
            }
        }
    }
}

/// Resolves `path` to an absolute location, using the nearest existing
/// ancestor for outputs that do not exist yet. Equivalent spellings of the
/// same directory (`frames`, `frames/../frames`) then compare equal, which is
/// required to reject a generated frame that would overwrite its own source.
fn canonical_or_parent(path: &Path) -> Result<PathBuf, PipelineError> {
    use std::path::Component;

    let mut cursor = path.to_path_buf();
    let mut missing = Vec::new();
    loop {
        match std::fs::canonicalize(&cursor) {
            Ok(mut canonical) => {
                for component in missing.into_iter().rev() {
                    canonical.push(component);
                }
                return Ok(canonical);
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                // A `..` in the unresolved suffix needs its preceding directory
                // to exist; dropping it would silently change the target.
                if matches!(cursor.components().next_back(), Some(Component::ParentDir)) {
                    return Err(PipelineError::Io {
                        path: cursor,
                        source: err,
                    });
                }
            }
            Err(source) => {
                return Err(PipelineError::Io {
                    path: cursor,
                    source,
                });
            }
        }
        match cursor.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => {
                if let Some(name) = cursor.file_name() {
                    missing.push(name.to_os_string());
                }
                cursor = parent.to_path_buf();
            }
            _ => {
                return Err(PipelineError::DecodeFailed {
                    path: cursor,
                    reason: "path has no existing ancestor to resolve".to_owned(),
                });
            }
        }
    }
}

/// True when the source itself is at a path this extraction would generate
/// (`frame_<digits>.<extension>` inside an equivalent output directory). This
/// is checked once at execution time so callers that build a plan directly
/// cannot publish a frame over their own source through an alias spelling.
fn source_is_generated_target(plan: &FrameExtractionPlan) -> Result<bool, PipelineError> {
    let source = plan.source();
    let Some(name) = source.file_name().and_then(|name| name.to_str()) else {
        return Ok(false);
    };
    let Some(parent) = source.parent() else {
        return Ok(false);
    };
    let Some((digits, ext)) = name
        .strip_prefix("frame_")
        .and_then(|rest| rest.rsplit_once('.'))
    else {
        return Ok(false);
    };
    if ext != plan.output_format().extension()
        || digits.is_empty()
        || !digits.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Ok(false);
    }
    Ok(canonical_or_parent(parent)? == canonical_or_parent(plan.output_dir())?)
}

/// Executes a validated [`FrameExtractionPlan`] on its CPU encoder path.
///
/// # Errors
///
/// Returns [`PipelineError`] on demux/decode/encode/write failure, or
/// [`PipelineError::NoFramesSelected`] (with no files or directory created)
/// when the selection and range match no frame.
// Keep the worker lifecycle in one scope: decode result storage, sender drop,
// and a join of every worker must all run on the same path so no panic can
// escape `thread::scope` and no sender can keep a join blocked.
#[allow(clippy::too_many_lines)]
pub fn execute_frame_extraction(
    plan: &FrameExtractionPlan,
) -> Result<FrameExtractionReport, PipelineError> {
    if source_is_generated_target(plan)? {
        return Err(PipelineError::EncodeFailed {
            path: plan.source().to_path_buf(),
            reason: "extraction would overwrite its own source via a generated frame path"
                .to_owned(),
        });
    }
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
    // pool charges every live raw pixel buffer (decoder scratch, animation
    // canvas, queued jobs, worker-held frames and recycled buffers).
    let pool = PixelPool::new(MAX_IN_FLIGHT_BYTES);
    let (error_tx, error_rx) = channel::unbounded::<PipelineError>();
    let failed = AtomicBool::new(false);
    let active_workers = AtomicUsize::new(workers);
    let total_bytes = AtomicU64::new(0);
    let bmp_write_lock = Mutex::new(());
    let extension = plan.output_format().extension();
    let (job_tx, job_rx) = channel::unbounded::<FrameJob<'_>>();

    let decode_result = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let spawned = thread::Builder::new().spawn_scoped(scope, || {
                let _guard = WorkerExitGuard {
                    active_workers: &active_workers,
                    failed: &failed,
                    pool: &pool,
                };
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
            match spawned {
                Ok(handle) => handles.push(handle),
                Err(err) => {
                    // Returning drops `job_tx` (captured below), disconnecting
                    // the channel so already-started workers exit before the
                    // scope joins them.
                    return Err(PipelineError::WorkerStartFailed(err));
                }
            }
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
        // Store the outcome instead of `?`: cleanup and joins must run on both
        // success and failure, and a panicked worker's join must consume its
        // panic here rather than let `thread::scope` re-raise it on unwind.
        let decoded = run_decode(plan, &mut emitter);
        let outcome = (emitter.sequence, emitter.last_dims);
        drop(emitter); // disconnect the job channel so workers can exit
        let mut all_joined = true;
        for handle in handles {
            if handle.join().is_err() {
                all_joined = false;
            }
        }
        decoded?;
        if !all_joined {
            return Err(PipelineError::WorkerPanicked);
        }
        Ok(outcome)
    });

    if let Ok(err) = error_rx.try_recv() {
        return Err(err);
    }
    let (frame_count, dims) = decode_result?;
    if failed.load(Ordering::Acquire) {
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
        FrameEmitter, FrameExtractionPlan, FrameJob, LeaseReservation, MAX_IN_FLIGHT_BYTES,
        PipelineError, PixelPool, Selector, WorkerExitGuard, compact_cropped_rgb,
        copy_cropped_rows, decode_animation, decode_video, execute_frame_extraction,
        frame_byte_len, frame_error, output_dimensions,
    };
    use crate::decode::DecodedPixelFormat;
    use crate::video_anim::{AnimFrameInfo, encode_animation_stream};
    use crate::video_codecs::{
        Yuv420Frame, YuvColor, decode_video_track_streaming_ctrl, encode_video_cpu, nv12_buffer_len,
    };
    use crate::video_container::{ContainerMedia, demux_video, mux_video};
    use crossbeam_channel as channel;
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
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

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
    fn output_dimensions_trims_only_alignment_padding() {
        let probed = Some((321_u32, 241_u32));
        // MJPEG's even-alignment padding is trimmed.
        assert_eq!(output_dimensions((322, 242), probed), (321, 241));
        assert_eq!(output_dimensions((321, 241), probed), (321, 241));
        // Real content beyond one pixel of padding is never truncated.
        assert_eq!(output_dimensions((400, 300), probed), (400, 300));
        assert_eq!(output_dimensions((322, 300), probed), (322, 300));
        // A smaller decoded frame is emitted at its own size.
        assert_eq!(output_dimensions((320, 240), probed), (320, 240));
        assert_eq!(output_dimensions((322, 242), None), (322, 242));
    }

    #[test]
    fn compact_cropped_rgb_trims_padded_rows_in_place() -> Result<(), Box<dyn Error>> {
        // Padded 4x2 RGB buffer (24 bytes) cropped to 3x2 (18 bytes).
        let mut rgb: Vec<u8> = (1_u8..=24).collect();
        let cropped = compact_cropped_rgb(&mut rgb, 4, 3, 2)?;
        rgb.truncate(cropped);
        assert_eq!(
            rgb,
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21
            ]
        );
        Ok(())
    }

    #[test]
    fn pixel_pool_enforces_aggregate_cap_with_concurrent_workers() -> Result<(), Box<dyn Error>> {
        // Simulates 2:1 RGB:NV12 frames (192 + 96 = 288 bytes) under a 1024-byte
        // aggregate budget with 4 concurrent workers.
        let limit = 1024_usize;
        let rgb_len = 192_usize;
        let nv12_len = 96_usize;
        let pool = PixelPool::new(limit);
        let (tx, rx) = channel::unbounded::<FrameJob<'_>>();
        let processed = AtomicUsize::new(0);

        thread::scope(|scope| -> Result<(), PipelineError> {
            for _ in 0..4 {
                scope.spawn(|| {
                    while let Ok(job) = rx.recv() {
                        assert_eq!(job.pixels.len(), rgb_len);
                        assert!(pool.live_bytes() <= limit);
                        thread::sleep(Duration::from_millis(2));
                        processed.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
            for seq in 1_u64..=40 {
                let (mut rgb, mut nv12) = pool.take_pair(rgb_len, nv12_len)?;
                assert!(
                    pool.live_bytes() <= limit,
                    "live bytes {} exceeded limit {limit} while holding RGB + NV12",
                    pool.live_bytes()
                );
                nv12.as_mut_slice()
                    .fill(u8::try_from(seq % 251).unwrap_or(0));
                rgb.as_mut_slice()
                    .fill(u8::try_from(seq % 251).unwrap_or(0));
                drop(nv12);
                tx.send(FrameJob {
                    sequence: seq,
                    pixels: rgb,
                    width: 8,
                    height: 8,
                    format: DecodedPixelFormat::Rgb8,
                })
                .map_err(|_| PipelineError::WorkerPanicked)?;
            }
            drop(tx);
            Ok(())
        })?;

        assert_eq!(processed.load(Ordering::Relaxed), 40);
        assert!(
            pool.peak_live_bytes() <= limit,
            "peak live bytes {} exceeded aggregate cap {limit}",
            pool.peak_live_bytes()
        );
        assert!(
            pool.peak_live_bytes() > rgb_len + nv12_len,
            "expected concurrent pipelining beyond one frame (peak {})",
            pool.peak_live_bytes()
        );
        assert_eq!(pool.leased_bytes(), 0, "all buffers must be returned");
        Ok(())
    }

    #[test]
    fn pixel_pool_rejects_oversized_nv12_plus_rgb_without_deadlocking() -> Result<(), Box<dyn Error>>
    {
        // 8192x8192 needs 192 MiB RGB + 96 MiB NV12 = 288 MiB > 256 MiB cap.
        let rgb = frame_byte_len(8192, 8192, 3)?;
        let nv12 = nv12_buffer_len(8192, 8192, 8192)?;
        assert_eq!(rgb, 192 * 1024 * 1024);
        assert_eq!(nv12, 96 * 1024 * 1024);
        let pool = PixelPool::new(MAX_IN_FLIGHT_BYTES);
        let result = pool.take_pair(rgb, nv12);
        assert!(matches!(result, Err(PipelineError::DecodeFailed { .. })));
        assert_eq!(
            pool.peak_live_bytes(),
            0,
            "oversized pair must fail before allocating"
        );
        assert!(matches!(
            pool.take_pair(usize::MAX, 1),
            Err(PipelineError::DecodeFailed { .. })
        ));
        Ok(())
    }

    #[test]
    fn pixel_pool_handles_borderline_frame_and_resizing_without_deadlock()
    -> Result<(), Box<dyn Error>> {
        // Part A: single frame consumes 100% of the budget (192 + 96 == 288).
        let exact_limit = 288_usize;
        let pool = PixelPool::new(exact_limit);
        let (tx, rx) = channel::unbounded::<FrameJob<'_>>();
        let processed = AtomicUsize::new(0);
        thread::scope(|scope| -> Result<(), PipelineError> {
            for _ in 0..2 {
                scope.spawn(|| {
                    while let Ok(_job) = rx.recv() {
                        thread::sleep(Duration::from_millis(1));
                        processed.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
            for seq in 1_u64..=12 {
                let (rgb, nv12) = pool.take_pair(192, 96)?;
                drop(nv12);
                tx.send(FrameJob {
                    sequence: seq,
                    pixels: rgb,
                    width: 8,
                    height: 8,
                    format: DecodedPixelFormat::Rgb8,
                })
                .map_err(|_| PipelineError::WorkerPanicked)?;
            }
            drop(tx);
            Ok(())
        })?;
        assert_eq!(processed.load(Ordering::Relaxed), 12);
        assert_eq!(pool.peak_live_bytes(), exact_limit);

        // Part B: variable-resolution resize where keeping an oversized idle
        // buffer (100 bytes) for an 80-byte slot alongside a 160-byte slot
        // would exceed the 256-byte limit; evicting idle buffers must succeed.
        let resize_pool = PixelPool::new(256);
        {
            let (_a, _b) = resize_pool.take_pair(100, 50)?;
        }
        let (big_rgb, big_nv12) = resize_pool.take_pair(160, 80)?;
        assert_eq!(big_rgb.len(), 160);
        assert_eq!(big_nv12.len(), 80);
        assert!(resize_pool.peak_live_bytes() <= 256);
        drop(big_rgb);
        drop(big_nv12);
        assert_eq!(resize_pool.leased_bytes(), 0);
        Ok(())
    }

    #[test]
    fn pixel_pool_external_canvas_hold_shares_budget_and_rejects_overflow()
    -> Result<(), Box<dyn Error>> {
        let pool = PixelPool::new(256);
        let hold = pool.reserve_external(128)?;
        assert_eq!(pool.live_bytes(), 128);
        {
            let frame = pool.take(128)?;
            assert_eq!(frame.len(), 128);
            assert_eq!(pool.live_bytes(), 256);
        }
        assert!(matches!(
            pool.take(129),
            Err(PipelineError::DecodeFailed { .. })
        ));
        drop(hold);
        let full = pool.take(256)?;
        assert_eq!(full.len(), 256);
        assert_eq!(pool.peak_live_bytes(), 256);
        Ok(())
    }

    #[test]
    fn pixel_pool_close_wakes_blocked_waiter_without_deadlock() -> Result<(), Box<dyn Error>> {
        let pool = PixelPool::new(256);
        let held = pool.take(256)?; // entire budget is leased
        let (done_tx, done_rx) = channel::bounded(1);
        let result = thread::scope(|scope| {
            let waiter = scope.spawn(|| {
                let outcome = pool.take_pair(128, 64).map(|_| ());
                drop(done_tx.send(outcome));
            });
            // Synchronize on actual admission: the waiter must be parked in
            // the condvar wait before `close()` is called.
            let start = Instant::now();
            while pool.waiters() == 0 {
                assert!(
                    start.elapsed() < Duration::from_secs(5),
                    "waiter was never admitted to the pool wait"
                );
                thread::yield_now();
            }
            pool.close();
            // Require a bounded wake-up from `close()` alone while `held` is
            // still alive; releasing the lease must not be what wakes it.
            let outcome = done_rx.recv_timeout(Duration::from_secs(5)).ok();
            drop(waiter.join());
            drop(held); // cleanup happens only after the result is collected
            outcome
        });
        assert!(matches!(result, Some(Err(PipelineError::WorkerPanicked))));
        Ok(())
    }

    #[test]
    fn worker_exit_guard_and_reservation_rollback_prevent_deadlock() -> Result<(), Box<dyn Error>> {
        // WorkerExitGuard closes the pool when the last worker exits while the
        // decoder is blocked in `take_pair`.
        let pool = PixelPool::new(256);
        let active = AtomicUsize::new(1);
        let failed = AtomicBool::new(false);
        let held = pool.take(256)?;
        let (done_tx, done_rx) = channel::bounded(1);
        let (exit_tx, exit_rx) = channel::bounded::<()>(1);
        let result = thread::scope(|scope| {
            let waiter = scope.spawn(|| {
                let outcome = pool.take_pair(128, 64).map(|_| ());
                drop(done_tx.send(outcome));
            });
            let worker = scope.spawn(|| {
                // Only exit once the decoder is confirmed parked, so this
                // exercises a genuinely blocked waiter rather than a race.
                match exit_rx.recv() {
                    Ok(()) => {}
                    Err(_) => return,
                }
                let _guard = WorkerExitGuard {
                    active_workers: &active,
                    failed: &failed,
                    pool: &pool,
                };
            });
            // Wait until the decoder is genuinely parked before the last
            // worker exits and closes the pool.
            let start = Instant::now();
            while pool.waiters() == 0 {
                assert!(
                    start.elapsed() < Duration::from_secs(5),
                    "waiter was never admitted to the pool wait"
                );
                thread::yield_now();
            }
            let signalled = exit_tx.send(());
            assert!(signalled.is_ok(), "worker exited before its start signal");
            drop(worker.join());
            let outcome = done_rx.recv_timeout(Duration::from_secs(5)).ok();
            drop(waiter.join());
            drop(held); // keep the lease until after the waiter result is known
            outcome
        });
        assert!(matches!(result, Some(Err(PipelineError::WorkerPanicked))));
        assert_eq!(active.load(Ordering::Acquire), 0);

        // LeaseReservation rolls back pre-reserved bytes on drop (e.g. OOM)
        // and wakes any thread waiting for budget.
        let rollback_pool = PixelPool::new(256);
        {
            let mut state = rollback_pool.lock_state();
            state.leased_bytes = 256;
            drop(state);
            let reservation = LeaseReservation {
                pool: &rollback_pool,
                bytes: 256,
            };
            let acquired = thread::scope(|scope| {
                let waiter = scope.spawn(|| rollback_pool.take(256).map(|buf| buf.len()));
                thread::sleep(Duration::from_millis(10));
                drop(reservation);
                waiter.join().ok()
            });
            assert!(matches!(acquired, Some(Ok(256))));
        }
        assert_eq!(rollback_pool.leased_bytes(), 0);
        Ok(())
    }

    #[test]
    fn decode_animation_enforces_gif_budget_end_to_end() -> Result<(), Box<dyn Error>> {
        let root = temp_root("gif-budget")?;
        // Animated GIF with DisposalMethod::Previous (4x4 RGBA = 64 bytes/canvas).
        // Hold = 2 * 64 = 128 bytes; sub-frame / output frame = 64 bytes;
        // exact single-frame budget = 192 bytes.
        let gif_path = root.join("prev.gif");
        {
            let file = File::create(&gif_path)?;
            let mut encoder = gif::Encoder::new(file, 4, 4, &[])?;
            encoder.set_repeat(gif::Repeat::Infinite)?;
            for step in 0_u8..4 {
                let mut rgba = vec![step.saturating_mul(50); 4 * 4 * 4];
                for px in rgba.as_chunks_mut::<4>().0 {
                    px[3] = 255;
                }
                let mut frame = gif::Frame::from_rgba_speed(4, 4, &mut rgba, 10);
                frame.delay = 10;
                frame.dispose = if step % 2 == 0 {
                    gif::DisposalMethod::Previous
                } else {
                    gif::DisposalMethod::Keep
                };
                encoder.write_frame(&frame)?;
            }
        }
        let gif_canvas = 4 * 4 * 4;
        let gif_budget = gif_canvas * 3;
        let gif_pool = PixelPool::new(gif_budget);
        let (job_tx, job_rx) = channel::unbounded::<FrameJob<'_>>();
        let (_error_tx, error_rx) = channel::unbounded::<PipelineError>();
        let failed = AtomicBool::new(false);
        let gif_seen = AtomicUsize::new(0);
        let gif_emitted = thread::scope(|scope| -> Result<u64, PipelineError> {
            scope.spawn(|| {
                while let Ok(job) = job_rx.recv() {
                    assert_eq!(job.pixels.len(), gif_canvas);
                    thread::sleep(Duration::from_millis(1));
                    gif_seen.fetch_add(1, Ordering::Relaxed);
                }
            });
            let mut emitter = FrameEmitter {
                selector: Selector::new(FrameSelection::All, 0),
                range: FrameRange::new(0, None).map_err(|err| frame_error(err.to_string()))?,
                expected: Some((4, 4)),
                pool: &gif_pool,
                job_tx,
                error_rx: &error_rx,
                failed: &failed,
                sequence: 0,
                last_dims: None,
            };
            let _flow = decode_animation(&gif_path, VideoContainerFormat::Gif, &mut emitter)?;
            Ok(emitter.sequence)
        })?;
        assert_eq!(gif_emitted, 4);
        assert_eq!(gif_seen.load(Ordering::Relaxed), 4);
        assert!(gif_pool.peak_live_bytes() <= gif_budget);
        assert_eq!(gif_pool.leased_bytes(), 0);

        // One byte below GIF canvas hold + frame rejects cleanly without hanging.
        let tiny_gif_pool = PixelPool::new(gif_budget - 1);
        let (job_tx, _job_rx) = channel::unbounded::<FrameJob<'_>>();
        let (_error_tx, error_rx) = channel::unbounded::<PipelineError>();
        let failed = AtomicBool::new(false);
        let mut tiny_gif_emitter = FrameEmitter {
            selector: Selector::new(FrameSelection::All, 0),
            range: FrameRange::new(0, None)?,
            expected: Some((4, 4)),
            pool: &tiny_gif_pool,
            job_tx,
            error_rx: &error_rx,
            failed: &failed,
            sequence: 0,
            last_dims: None,
        };
        assert!(matches!(
            decode_animation(&gif_path, VideoContainerFormat::Gif, &mut tiny_gif_emitter),
            Err(PipelineError::DecodeFailed { .. })
        ));
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn decode_animation_gif_keep_frames_charge_one_canvas_only() -> Result<(), Box<dyn Error>> {
        let root = temp_root("gif-keep-budget")?;
        let gif_path = root.join("keep.gif");
        {
            let file = File::create(&gif_path)?;
            let mut encoder = gif::Encoder::new(file, 4, 4, &[])?;
            encoder.set_repeat(gif::Repeat::Infinite)?;
            for step in 0_u8..3 {
                let mut rgba = vec![step.saturating_mul(70); 4 * 4 * 4];
                for px in rgba.as_chunks_mut::<4>().0 {
                    px[3] = 255;
                }
                let mut frame = gif::Frame::from_rgba_speed(4, 4, &mut rgba, 10);
                frame.delay = 10;
                frame.dispose = gif::DisposalMethod::Keep;
                encoder.write_frame(&frame)?;
            }
        }
        // Exactly 2x canvas: enough for the 1x canvas hold + 1x sub-frame.
        // A 2x backup hold would reject; `Keep` frames must never be charged
        // for a backup they do not retain.
        let canvas = 4 * 4 * 4;
        let pool = PixelPool::new(canvas * 2);
        let (job_tx, job_rx) = channel::unbounded::<FrameJob<'_>>();
        let (_error_tx, error_rx) = channel::unbounded::<PipelineError>();
        let failed = AtomicBool::new(false);
        let emitted = thread::scope(|scope| -> Result<u64, PipelineError> {
            scope.spawn(|| {
                while let Ok(job) = job_rx.recv() {
                    assert_eq!(job.pixels.len(), canvas);
                }
            });
            let mut emitter = FrameEmitter {
                selector: Selector::new(FrameSelection::All, 0),
                range: FrameRange::new(0, None).map_err(|err| frame_error(err.to_string()))?,
                expected: Some((4, 4)),
                pool: &pool,
                job_tx,
                error_rx: &error_rx,
                failed: &failed,
                sequence: 0,
                last_dims: None,
            };
            let _flow = decode_animation(&gif_path, VideoContainerFormat::Gif, &mut emitter)?;
            Ok(emitter.sequence)
        })?;
        assert_eq!(emitted, 3);
        assert!(pool.peak_live_bytes() <= canvas * 2);
        assert_eq!(pool.leased_bytes(), 0);
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn execute_rejects_generated_frame_path_alias() -> Result<(), Box<dyn Error>> {
        let root = temp_root("frame-path-alias")?;
        let source = root.join("frame_000001.gif");
        // The execution-time guard runs before decoding, so file content is
        // irrelevant here.
        File::create(&source)?;
        let decoy = root.join("decoy");
        fs::create_dir_all(&decoy)?;
        // `decoy/..` names the source directory through an equivalent spelling.
        let alias_out = decoy.join("..");
        let plan = plan(
            &source,
            &alias_out,
            VideoContainerFormat::Gif,
            ImageOutputFormat::Gif,
            FrameSelection::All,
            FrameRange::new(0, None)?,
        )?;
        match execute_frame_extraction(&plan) {
            Err(PipelineError::EncodeFailed { .. }) => {}
            other => {
                return Err(format!(
                    "expected EncodeFailed for an aliased generated frame path, got {other:?}"
                )
                .into());
            }
        }
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn decode_animation_enforces_webp_budget_end_to_end() -> Result<(), Box<dyn Error>> {
        let root = temp_root("webp-budget")?;
        // Animated WebP with alpha (8x8 RGBA = 256 bytes/frame).
        // Hold = 256 bytes; output = 256 bytes; exact budget = 512 bytes.
        let webp_path = root.join("anim.webp");
        write_animated_webp(&webp_path, 3)?;
        let webp_frame = 8 * 8 * 4;
        let webp_budget = webp_frame * 2;
        let webp_pool = PixelPool::new(webp_budget);
        let (job_tx, job_rx) = channel::unbounded::<FrameJob<'_>>();
        let (_error_tx, error_rx) = channel::unbounded::<PipelineError>();
        let failed = AtomicBool::new(false);
        let webp_seen = AtomicUsize::new(0);
        let webp_emitted = thread::scope(|scope| -> Result<u64, PipelineError> {
            scope.spawn(|| {
                while let Ok(job) = job_rx.recv() {
                    assert_eq!(job.pixels.len(), webp_frame);
                    thread::sleep(Duration::from_millis(1));
                    webp_seen.fetch_add(1, Ordering::Relaxed);
                }
            });
            let mut emitter = FrameEmitter {
                selector: Selector::new(FrameSelection::All, 0),
                range: FrameRange::new(0, None).map_err(|err| frame_error(err.to_string()))?,
                expected: Some((8, 8)),
                pool: &webp_pool,
                job_tx,
                error_rx: &error_rx,
                failed: &failed,
                sequence: 0,
                last_dims: None,
            };
            let _flow = decode_animation(&webp_path, VideoContainerFormat::Webp, &mut emitter)?;
            Ok(emitter.sequence)
        })?;
        assert_eq!(webp_emitted, 3);
        assert_eq!(webp_seen.load(Ordering::Relaxed), 3);
        assert!(webp_pool.peak_live_bytes() <= webp_budget);
        assert_eq!(webp_pool.leased_bytes(), 0);

        let tiny_webp_pool = PixelPool::new(webp_budget - 1);
        let (job_tx, _job_rx) = channel::unbounded::<FrameJob<'_>>();
        let (_error_tx, error_rx) = channel::unbounded::<PipelineError>();
        let failed = AtomicBool::new(false);
        let mut tiny_webp_emitter = FrameEmitter {
            selector: Selector::new(FrameSelection::All, 0),
            range: FrameRange::new(0, None)?,
            expected: Some((8, 8)),
            pool: &tiny_webp_pool,
            job_tx,
            error_rx: &error_rx,
            failed: &failed,
            sequence: 0,
            last_dims: None,
        };
        assert!(matches!(
            decode_animation(
                &webp_path,
                VideoContainerFormat::Webp,
                &mut tiny_webp_emitter
            ),
            Err(PipelineError::DecodeFailed { .. })
        ));
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn decode_video_enforces_pooled_nv12_and_rgb_budget_end_to_end() -> Result<(), Box<dyn Error>> {
        let root = temp_root("video-budget")?;
        let source = root.join("in.mp4");
        write_video(&source, VideoContainerFormat::Mp4, &solid_frames(6, 32, 32))?;
        let rgb_bytes = frame_byte_len(32, 32, 3)?;
        let nv12_bytes = nv12_buffer_len(32, 32, 32)?;
        let exact_frame_budget = rgb_bytes + nv12_bytes;

        // Exact combined RGB + NV12 budget succeeds across 6 frames with
        // concurrent consumers and never exceeds the cap.
        let pool = PixelPool::new(exact_frame_budget);
        let (job_tx, job_rx) = channel::unbounded::<FrameJob<'_>>();
        let (_error_tx, error_rx) = channel::unbounded::<PipelineError>();
        let failed = AtomicBool::new(false);
        let seen = AtomicUsize::new(0);
        let frame_count = thread::scope(|scope| -> Result<u64, PipelineError> {
            for _ in 0..2 {
                scope.spawn(|| {
                    while let Ok(job) = job_rx.recv() {
                        assert_eq!(job.pixels.len(), rgb_bytes);
                        thread::sleep(Duration::from_millis(1));
                        seen.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
            let mut emitter = FrameEmitter {
                selector: Selector::new(FrameSelection::All, 0),
                range: FrameRange::new(0, None).map_err(|err| frame_error(err.to_string()))?,
                expected: Some((32, 32)),
                pool: &pool,
                job_tx,
                error_rx: &error_rx,
                failed: &failed,
                sequence: 0,
                last_dims: None,
            };
            let _flow = decode_video(&source, VideoContainerFormat::Mp4, &mut emitter)?;
            Ok(emitter.sequence)
        })?;
        assert_eq!(frame_count, 6);
        assert_eq!(seen.load(Ordering::Relaxed), 6);
        assert_eq!(pool.peak_live_bytes(), exact_frame_budget);
        assert_eq!(pool.leased_bytes(), 0);

        // One byte below RGB + NV12 rejects cleanly without hanging.
        let tiny_pool = PixelPool::new(exact_frame_budget - 1);
        let (job_tx, _job_rx) = channel::unbounded::<FrameJob<'_>>();
        let (_error_tx, error_rx) = channel::unbounded::<PipelineError>();
        let failed = AtomicBool::new(false);
        let mut emitter = FrameEmitter {
            selector: Selector::new(FrameSelection::All, 0),
            range: FrameRange::new(0, None)?,
            expected: Some((32, 32)),
            pool: &tiny_pool,
            job_tx,
            error_rx: &error_rx,
            failed: &failed,
            sequence: 0,
            last_dims: None,
        };
        let rejected = decode_video(&source, VideoContainerFormat::Mp4, &mut emitter);
        assert!(matches!(rejected, Err(PipelineError::DecodeFailed { .. })));
        assert_eq!(tiny_pool.peak_live_bytes(), 0);

        fs::remove_dir_all(&root)?;
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
