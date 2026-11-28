//! Concrete execution of a validated [`FrameSequencePlan`].
//!
//! Animation stacks stream one image at a time into the shared GIF/WebP
//! encoder. Video stacks either decode into a budgeted `Yuv420Frame` list on the
//! CPU or stream NV12 directly into the pinned NVENC ring. Every legal state is
//! described by the domain plan, so there is no backend switch and no CPU
//! fallback when a GPU encoder was selected.

use std::collections::VecDeque;
use std::fs;
use std::io;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Component, Path, PathBuf};

use fastconv_domain::{
    AnimationLoop, FrameSequencePlan, FrameSequenceReport, ImageDimensions, StackExecutionMode,
    StackFrame, VideoContainerFormat, VideoEncodeMode,
};

use crate::PipelineError;
use crate::decode::{DecodeTargetHint, DecodedPixelFormat, decode_source_image_into};
use crate::gpu::nvenc::encode_stack_nvenc_streaming;
use crate::staging::StagingTarget;
use crate::video_anim::{AnimFrameInfo, encode_animation_stream_bounded, pixels_to_yuv420};
use crate::video_container::{ContainerMedia, mux_video};
use crate::video_transcode::enforce_decoded_budget;

/// Raw-pixel ceiling for in-flight animation frames (256 MiB). This caps
/// normal frames; a single frame whose raw size exceeds the budget is still
/// processed through one worker, with memory bounded to that one frame.
const RAW_PIXEL_BUDGET: u64 = 256 * 1024 * 1024;

/// Resolves an output path against the nearest existing ancestor so a
/// not-yet-created target can be compared with already-existing sources.
fn canonical_or_parent(path: &Path) -> Result<PathBuf, PipelineError> {
    let mut cursor = path.to_path_buf();
    let mut missing = VecDeque::new();
    loop {
        match fs::canonicalize(&cursor) {
            Ok(mut canonical) => {
                for component in missing {
                    canonical = canonical.join(component);
                }
                return Ok(canonical);
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                if matches!(cursor.components().next_back(), Some(Component::ParentDir)) {
                    return Err(PipelineError::Io {
                        path: path.to_path_buf(),
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
                    missing.push_front(name.to_os_string());
                }
                cursor = parent.to_path_buf();
            }
            _ => {
                let mut canonical = fs::canonicalize(".").map_err(|source| PipelineError::Io {
                    path: PathBuf::from("."),
                    source,
                })?;
                if let Some(name) = cursor.file_name() {
                    missing.push_front(name.to_os_string());
                }
                for component in missing {
                    canonical = canonical.join(component);
                }
                return Ok(canonical);
            }
        }
    }
}

/// Re-checks at execution time that the target does not alias a source. The
/// probe performed the same check on the user's paths; this catches a target
/// that was replaced by a symlink between planning and commit.
fn verify_no_alias(plan: &FrameSequencePlan) -> Result<(), PipelineError> {
    let target = canonical_or_parent(plan.target_path())?;
    for frame in plan.frames() {
        if canonical_or_parent(frame.source())? == target {
            return Err(PipelineError::EncodeFailed {
                path: plan.target_path().to_path_buf(),
                reason: format!(
                    "stack target aliases source image '{}'",
                    frame.source().display()
                ),
            });
        }
    }
    Ok(())
}

const fn channels_of(format: DecodedPixelFormat) -> u8 {
    match format {
        DecodedPixelFormat::L8 => 1,
        DecodedPixelFormat::La8 => 2,
        DecodedPixelFormat::Rgb8 => 3,
        DecodedPixelFormat::Rgba8 => 4,
    }
}

fn raw_len(dimensions: ImageDimensions, channels: u8) -> Result<usize, PipelineError> {
    let width =
        usize::try_from(dimensions.width().get()).map_err(|_| PipelineError::EncodeFailed {
            path: PathBuf::from("frame-stack"),
            reason: "frame width exceeds addressable memory".to_owned(),
        })?;
    let height =
        usize::try_from(dimensions.height().get()).map_err(|_| PipelineError::EncodeFailed {
            path: PathBuf::from("frame-stack"),
            reason: "frame height exceeds addressable memory".to_owned(),
        })?;
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(usize::from(channels)))
        .ok_or_else(|| PipelineError::EncodeFailed {
            path: PathBuf::from("frame-stack"),
            reason: "frame pixel buffer size overflow".to_owned(),
        })
}

fn ensure_scratch(scratch: &mut Vec<u8>, len: usize) -> Result<(), PipelineError> {
    if scratch.capacity() < len {
        scratch
            .try_reserve_exact(len)
            .map_err(|_| PipelineError::EncodeFailed {
                path: PathBuf::from("frame-stack"),
                reason: "cannot allocate normalized frame scratch buffer".to_owned(),
            })?;
    }
    scratch.resize(len, 0);
    Ok(())
}

/// Normalizes a decoded frame into the stack's single channel layout, using
/// `scratch` only when a conversion is actually needed (zero-copy otherwise).
fn normalize_frame<'a>(
    source: &Path,
    pixels: &'a [u8],
    format: DecodedPixelFormat,
    target_channels: u8,
    scratch: &'a mut Vec<u8>,
) -> Result<&'a [u8], PipelineError> {
    let channels = channels_of(format);
    if channels == target_channels {
        return Ok(pixels);
    }
    let pixel_count = pixels.len() / usize::from(channels);
    let needed = pixel_count
        .checked_mul(usize::from(target_channels))
        .ok_or_else(|| PipelineError::EncodeFailed {
            path: source.to_path_buf(),
            reason: "normalized frame size overflow".to_owned(),
        })?;
    ensure_scratch(scratch, needed)?;
    let reject_alpha = || PipelineError::UnsupportedPixelFormat {
        path: source.to_path_buf(),
        format: "non-opaque alpha in an opaque image stack".to_owned(),
    };

    match (format, target_channels) {
        (DecodedPixelFormat::L8, 3) => {
            for (dst, &lum) in scratch.as_chunks_mut::<3>().0.iter_mut().zip(pixels.iter()) {
                dst.copy_from_slice(&[lum, lum, lum]);
            }
        }
        (DecodedPixelFormat::L8, 4) => {
            for (dst, &lum) in scratch.as_chunks_mut::<4>().0.iter_mut().zip(pixels.iter()) {
                dst.copy_from_slice(&[lum, lum, lum, u8::MAX]);
            }
        }
        (DecodedPixelFormat::La8, 3) => {
            for (dst, px) in scratch
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(pixels.as_chunks::<2>().0.iter())
            {
                if px[1] != u8::MAX {
                    return Err(reject_alpha());
                }
                dst.copy_from_slice(&[px[0], px[0], px[0]]);
            }
        }
        (DecodedPixelFormat::La8, 4) => {
            for (dst, px) in scratch
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(pixels.as_chunks::<2>().0.iter())
            {
                dst.copy_from_slice(&[px[0], px[0], px[0], px[1]]);
            }
        }
        (DecodedPixelFormat::Rgb8, 4) => {
            for (dst, px) in scratch
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(pixels.as_chunks::<3>().0.iter())
            {
                dst[..3].copy_from_slice(px);
                dst[3] = u8::MAX;
            }
        }
        (DecodedPixelFormat::Rgba8, 3) => {
            for (dst, px) in scratch
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(pixels.as_chunks::<4>().0.iter())
            {
                if px[3] != u8::MAX {
                    return Err(reject_alpha());
                }
                dst.copy_from_slice(&px[..3]);
            }
        }
        _ => {
            return Err(PipelineError::UnsupportedPixelFormat {
                path: source.to_path_buf(),
                format: format!(
                    "cannot normalize {} channels into {target_channels}",
                    u64::from(channels)
                ),
            });
        }
    }
    Ok(scratch.as_slice())
}

/// Decodes one still image into the caller-owned scratch buffer. The returned
/// slice borrows only `pixel_buf`, so the same allocation is reused per frame.
pub(crate) fn decode_stack_frame_into<'a>(
    frame: &StackFrame,
    dimensions: ImageDimensions,
    file_bytes: &mut Vec<u8>,
    pixel_buf: &'a mut Vec<u8>,
) -> Result<(DecodedPixelFormat, &'a [u8]), PipelineError> {
    let meta = decode_source_image_into(
        frame.source(),
        frame.format(),
        dimensions,
        DecodeTargetHint::Native,
        file_bytes,
        pixel_buf,
    )?;
    let decoded = pixel_buf
        .get(..meta.byte_len)
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: frame.source().to_path_buf(),
            reason: "decoded frame buffer is shorter than reported".to_owned(),
        })?;
    Ok((meta.format, decoded))
}

/// Returns an opaque pixel layout suitable for a video encoder. RGB8 and fully
/// opaque RGBA8 keep their native layout (and the SIMD conversion); grayscale
/// and translucent sources are flattened onto opaque black into `flat`.
pub(crate) fn opaque_video_pixels<'a>(
    source: &Path,
    format: DecodedPixelFormat,
    decoded: &'a [u8],
    flat: &'a mut Vec<u8>,
) -> Result<(&'a [u8], u8), PipelineError> {
    match format {
        DecodedPixelFormat::Rgb8 => Ok((decoded, 3)),
        DecodedPixelFormat::Rgba8
            if decoded
                .as_chunks::<4>()
                .0
                .iter()
                .all(|pixel| pixel[3] == u8::MAX) =>
        {
            Ok((decoded, 4))
        }
        DecodedPixelFormat::L8 | DecodedPixelFormat::La8 | DecodedPixelFormat::Rgba8 => {
            let channels = channels_of(format);
            let pixels = decoded.len() / usize::from(channels);
            let needed =
                pixels
                    .checked_mul(3)
                    .ok_or_else(|| PipelineError::UnsupportedPixelFormat {
                        path: source.to_path_buf(),
                        format: "RGB flatten size overflow".to_owned(),
                    })?;
            ensure_scratch(flat, needed)?;
            match (format, decoded.len() % usize::from(channels)) {
                (DecodedPixelFormat::L8, 0) => {
                    for (dst, &lum) in flat.as_chunks_mut::<3>().0.iter_mut().zip(decoded.iter()) {
                        dst.copy_from_slice(&[lum, lum, lum]);
                    }
                }
                (DecodedPixelFormat::La8, 0) => {
                    for (dst, pixel) in flat
                        .as_chunks_mut::<3>()
                        .0
                        .iter_mut()
                        .zip(decoded.as_chunks::<2>().0.iter())
                    {
                        let alpha = u32::from(pixel[1]);
                        let lum = u32::from(pixel[0]);
                        for component in dst.iter_mut() {
                            *component = u8::try_from((lum * alpha + 127) / 255).unwrap_or(u8::MAX);
                        }
                    }
                }
                (DecodedPixelFormat::Rgba8, 0) => {
                    for (dst, pixel) in flat
                        .as_chunks_mut::<3>()
                        .0
                        .iter_mut()
                        .zip(decoded.as_chunks::<4>().0.iter())
                    {
                        let alpha = u32::from(pixel[3]);
                        for (component, &source) in dst.iter_mut().zip(pixel.iter().take(3)) {
                            *component = u8::try_from((u32::from(source) * alpha + 127) / 255)
                                .unwrap_or(u8::MAX);
                        }
                    }
                }
                _ => {
                    return Err(PipelineError::UnsupportedPixelFormat {
                        path: source.to_path_buf(),
                        format: "decoded frame length is not a whole number of pixels".to_owned(),
                    });
                }
            }
            Ok((flat.as_slice(), 3))
        }
    }
}

fn frame_timing(fps: fastconv_domain::StackFps, index: u64) -> (i64, u64) {
    let pts = fps.frame_timestamp_us(index);
    let next = fps.frame_timestamp_us(index.saturating_add(1));
    (
        i64::try_from(pts).unwrap_or(i64::MAX),
        next.saturating_sub(pts).max(1),
    )
}

fn finish_execution(
    plan: &FrameSequencePlan,
    staging: StagingTarget,
    workers: Option<NonZeroUsize>,
    hardware_device: Option<String>,
) -> Result<FrameSequenceReport, PipelineError> {
    let output_bytes = staging
        .file()?
        .metadata()
        .map_err(|source| PipelineError::Io {
            path: plan.target_path().to_path_buf(),
            source,
        })?;
    let output_bytes = NonZeroU64::new(output_bytes.len())
        .ok_or_else(|| PipelineError::EmptyOutput(plan.target_path().to_path_buf()))?;
    // Re-check immediately before atomic rename so a symlink planted during
    // encoding cannot overwrite a source image.
    verify_no_alias(plan)?;
    staging.commit()?;
    Ok(FrameSequenceReport::new(
        plan.frame_count(),
        plan.dimensions(),
        plan.fps().get(),
        plan.mode(),
        plan.target_path().to_path_buf(),
        output_bytes,
        workers,
        hardware_device,
    ))
}

/// CPU video stacks: budget every decoded frame up front, then decode and
/// convert one image at a time into a bounded `Yuv420Frame` list.
fn execute_cpu_video_sequence(
    plan: &FrameSequencePlan,
    container: VideoContainerFormat,
    mode: VideoEncodeMode,
) -> Result<FrameSequenceReport, PipelineError> {
    let dimensions = plan.dimensions();
    let width = u64::from(dimensions.width().get());
    let height = u64::from(dimensions.height().get());
    let plane_bytes = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .and_then(|bytes| bytes.checked_div(2))
        .ok_or_else(|| PipelineError::EncodeFailed {
            path: plan.target_path().to_path_buf(),
            reason: "video frame size overflow".to_owned(),
        })?;
    let total_bytes = plane_bytes
        .checked_mul(plan.frame_count().get())
        .ok_or_else(|| PipelineError::EncodeFailed {
            path: plan.target_path().to_path_buf(),
            reason: "decoded stack size overflow".to_owned(),
        })?;
    enforce_decoded_budget(plan.target_path(), total_bytes)?;

    let capacity =
        usize::try_from(plan.frame_count().get()).map_err(|_| PipelineError::EncodeFailed {
            path: plan.target_path().to_path_buf(),
            reason: "stack frame count exceeds addressable memory".to_owned(),
        })?;
    let mut frames = Vec::new();
    frames
        .try_reserve_exact(capacity)
        .map_err(|_| PipelineError::EncodeFailed {
            path: plan.target_path().to_path_buf(),
            reason: "cannot allocate decoded stack frame list".to_owned(),
        })?;
    let mut file_bytes = Vec::new();
    let mut pixel_buf = Vec::new();
    let mut flat = Vec::new();
    for (index, frame) in plan.frames().iter().enumerate() {
        let (format, decoded) =
            decode_stack_frame_into(frame, dimensions, &mut file_bytes, &mut pixel_buf)?;
        let (pixels, channels) = opaque_video_pixels(frame.source(), format, decoded, &mut flat)?;
        let (pts_us, duration_us) = frame_timing(plan.fps(), u64::try_from(index).unwrap_or(0));
        frames.push(pixels_to_yuv420(
            AnimFrameInfo {
                width: dimensions.width().get(),
                height: dimensions.height().get(),
                pts_us,
                duration_us,
                channels,
            },
            pixels,
            usize::from(channels),
        )?);
    }

    let stream = crate::video_codecs::encode_video_cpu(plan.target_path(), &frames, mode, 0)?;
    let staging = StagingTarget::create_with_handle(plan.target_path(), 0)?;
    mux_video(
        staging.file()?,
        plan.target_path(),
        container,
        &ContainerMedia {
            tracks: vec![stream.track],
            packets: stream.packets,
        },
    )?;
    // The CPU encoder owns private GOP workers and does not expose a count;
    // do not misreport the requested `--threads` as execution threads.
    finish_execution(plan, staging, None, None)
}

/// NVENC video stacks: one image at a time goes straight into the pinned NV12
/// ring. A missing `nvenc` feature or GPU failure is a hard error from the
/// backend; there is no CPU path here.
fn execute_nvenc_video_sequence(
    plan: &FrameSequencePlan,
    mode: VideoEncodeMode,
) -> Result<FrameSequenceReport, PipelineError> {
    let (stream, frames_encoded, device) =
        encode_stack_nvenc_streaming(plan.frames(), plan.dimensions(), plan.fps(), mode)?;
    let expected = plan.frame_count().get();
    if frames_encoded != expected {
        return Err(PipelineError::EncodeFailed {
            path: plan.target_path().to_path_buf(),
            reason: format!("NVENC wrote {frames_encoded} frames for {expected} sources"),
        });
    }
    let container = match plan.mode() {
        StackExecutionMode::Video(mode) => mode.container(),
        StackExecutionMode::Animation { .. } => {
            return Err(PipelineError::EncodeFailed {
                path: plan.target_path().to_path_buf(),
                reason: "animation stacks cannot use the NVENC video path".to_owned(),
            });
        }
    };
    let staging = StagingTarget::create_with_handle(plan.target_path(), 1)?;
    mux_video(
        staging.file()?,
        plan.target_path(),
        container,
        &ContainerMedia {
            tracks: vec![stream.track],
            packets: stream.packets,
        },
    )?;
    // One image decode worker plus encoding on the calling thread.
    finish_execution(plan, staging, Some(NonZeroUsize::MIN), Some(device))
}

/// Executes a validated frame sequence into its target.
///
/// # Errors
///
/// Returns [`PipelineError`] on an execution-time alias, decode failure,
/// incompatible pixels, encoder failure, GPU failure, or empty output.
pub fn execute_frame_sequence(
    plan: &FrameSequencePlan,
) -> Result<FrameSequenceReport, PipelineError> {
    match plan.mode() {
        StackExecutionMode::Animation { .. } => execute_animation_sequence(plan),
        StackExecutionMode::Video(mode) => {
            verify_no_alias(plan)?;
            let container = mode.container();
            let encode_mode = mode.encode_mode();
            match encode_mode {
                VideoEncodeMode::CpuH264 { .. }
                | VideoEncodeMode::CpuAv1 { .. }
                | VideoEncodeMode::CpuVp9 { .. }
                | VideoEncodeMode::CpuMjpeg { .. } => {
                    execute_cpu_video_sequence(plan, container, encode_mode)
                }
                VideoEncodeMode::NvencH264 { .. } | VideoEncodeMode::NvencHevc { .. } => {
                    execute_nvenc_video_sequence(plan, encode_mode)
                }
                VideoEncodeMode::CpuGif { .. } | VideoEncodeMode::CpuWebp { .. } => {
                    Err(PipelineError::EncodeFailed {
                        path: plan.target_path().to_path_buf(),
                        reason: "animated GIF/WebP stacks must use the animation path".to_owned(),
                    })
                }
            }
        }
    }
}

/// Executes a validated frame sequence into its animated GIF/WebP target.
///
/// # Errors
///
/// Returns [`PipelineError`] on an execution-time alias, decode failure,
/// transparent pixels in an opaque stack, encoder failure, or empty output.
fn execute_animation_sequence(
    plan: &FrameSequencePlan,
) -> Result<FrameSequenceReport, PipelineError> {
    verify_no_alias(plan)?;

    let dimensions = plan.dimensions();
    let target_channels = if plan.frames().iter().any(StackFrame::has_alpha) {
        4_u8
    } else {
        3_u8
    };
    let frame_bytes = u64::try_from(raw_len(dimensions, target_channels)?).unwrap_or(u64::MAX);
    // Each in-flight frame occupies two slots (queued + prepared/pending), so
    // the byte budget is charged twice per worker.
    let by_memory = RAW_PIXEL_BUDGET / frame_bytes.saturating_mul(2).max(1);
    let memory_workers = usize::try_from(by_memory).unwrap_or(usize::MAX).max(1);
    let workers = plan
        .worker_threads()
        .get()
        .min(memory_workers)
        .clamp(1, crate::video_anim::MAX_ANIM_WORKERS);

    let staging = StagingTarget::create_with_handle(plan.target_path(), 0)?;
    let mut file_bytes = Vec::new();
    let mut pixel_buf = Vec::new();
    let mut normalized = Vec::new();

    let written = {
        let file = staging.file()?;
        let fps = plan.fps();
        encode_animation_stream_bounded(
            plan.target_container(),
            plan.quality(),
            workers,
            plan.loop_count().unwrap_or(AnimationLoop::Infinite),
            file,
            RAW_PIXEL_BUDGET,
            frame_bytes,
            |push| {
                for (index, frame) in plan.frames().iter().enumerate() {
                    let index_u64 = u64::try_from(index).unwrap_or(u64::MAX);
                    let meta = decode_source_image_into(
                        frame.source(),
                        frame.format(),
                        dimensions,
                        DecodeTargetHint::Native,
                        &mut file_bytes,
                        &mut pixel_buf,
                    )?;
                    let decoded = pixel_buf.get(..meta.byte_len).ok_or_else(|| {
                        PipelineError::DecodeFailed {
                            path: frame.source().to_path_buf(),
                            reason: "decoded frame buffer is shorter than reported".to_owned(),
                        }
                    })?;
                    let pixels = normalize_frame(
                        frame.source(),
                        decoded,
                        meta.format,
                        target_channels,
                        &mut normalized,
                    )?;
                    let pts_us = fps.frame_timestamp_us(index_u64);
                    let next_pts_us = fps.frame_timestamp_us(index_u64.saturating_add(1));
                    let info = AnimFrameInfo {
                        width: dimensions.width().get(),
                        height: dimensions.height().get(),
                        pts_us: i64::try_from(pts_us).unwrap_or(i64::MAX),
                        duration_us: next_pts_us.saturating_sub(pts_us).max(1),
                        channels: target_channels,
                    };
                    push(info, pixels)?;
                }
                Ok(())
            },
        )?
    };

    let expected = u64::try_from(plan.frames().len()).unwrap_or(u64::MAX);
    if written != expected {
        return Err(PipelineError::EncodeFailed {
            path: plan.target_path().to_path_buf(),
            reason: format!("encoder wrote {written} frames for {expected} sources"),
        });
    }
    finish_execution(plan, staging, NonZeroUsize::new(workers), None)
}

#[cfg(test)]
mod tests {
    use super::execute_frame_sequence;
    use crate::PipelineError;
    use crate::video_anim::{
        decode_gif_frames_streaming_pixels, decode_webp_frames_streaming_pixels,
    };
    #[cfg(feature = "nvenc")]
    use fastconv_domain::TargetVideoCodec;
    use fastconv_domain::{
        AnimationLoop, FrameSequencePlan, GpuBackend, HardwareBackend, ImageDimensions,
        ImageInputFormat, StackExecutionMode, StackFps, StackFrame, StackTargetFormat,
        VideoContainerFormat, VideoEncodeMode, VideoQuality, VideoRateControl,
    };
    use std::fs;
    use std::num::{NonZeroU16, NonZeroUsize};
    use std::path::{Path, PathBuf};
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(label: &str) -> Result<PathBuf, PipelineError> {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "fastconv-stack-pipeline-{label}-{}-{id}",
            process::id()
        ));
        fs::create_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(dir)
    }

    fn write_png(
        path: &Path,
        width: u32,
        height: u32,
        color: png::ColorType,
        pixels: &[u8],
    ) -> Result<(), PipelineError> {
        let file = fs::File::create(path).map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
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
            })
    }

    fn rgb_gradient(width: u32, height: u32, frame_index: u8) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(usize::try_from(width * height * 3).unwrap_or(0));
        for y in 0..height {
            for x in 0..width {
                let coarse_x = x / 4;
                let coarse_y = y / 4;
                pixels.extend_from_slice(&[
                    u8::try_from(((coarse_x + u32::from(frame_index)) % 4) * 64).unwrap_or(0),
                    u8::try_from(((coarse_y + u32::from(frame_index)) % 4) * 64).unwrap_or(0),
                    u8::try_from(((coarse_x ^ coarse_y ^ u32::from(frame_index)) % 4) * 64)
                        .unwrap_or(0),
                ]);
            }
        }
        pixels
    }

    fn solid_rgb(width: u32, height: u32, frame_index: u8) -> Vec<u8> {
        vec![
            32_u8.saturating_add(frame_index.saturating_mul(64));
            usize::try_from(width * height * 3).unwrap_or(0)
        ]
    }

    fn frame(path: PathBuf, has_alpha: bool) -> Result<StackFrame, PipelineError> {
        StackFrame::new(path, ImageInputFormat::Png, has_alpha).map_err(|err| {
            PipelineError::EncodeFailed {
                path: PathBuf::from("frame-stack-test"),
                reason: err.to_string(),
            }
        })
    }

    fn assert_close(rgba: &[u8], expected_rgb: &[u8], tolerance: u8) {
        for (pixel, rgb) in rgba
            .as_chunks::<4>()
            .0
            .iter()
            .zip(expected_rgb.as_chunks::<3>().0)
        {
            for channel in 0..3 {
                assert!(
                    pixel[channel].abs_diff(rgb[channel]) <= tolerance,
                    "decoded {pixel:?} diverged from expected {rgb:?}"
                );
            }
        }
    }

    fn plan(
        frames: Vec<StackFrame>,
        target: PathBuf,
        format: StackTargetFormat,
    ) -> Result<FrameSequencePlan, PipelineError> {
        let fps = StackFps::new(10.0).map_err(|err| PipelineError::EncodeFailed {
            path: target.clone(),
            reason: err.to_string(),
        })?;
        let quality = VideoQuality::new(95).map_err(|err| PipelineError::EncodeFailed {
            path: target.clone(),
            reason: err.to_string(),
        })?;
        FrameSequencePlan::new(
            frames,
            ImageDimensions::new(24, 16).map_err(|err| PipelineError::EncodeFailed {
                path: target.clone(),
                reason: err.to_string(),
            })?,
            fps,
            target.clone(),
            StackExecutionMode::animation(
                format,
                quality,
                AnimationLoop::Times(NonZeroU16::new(2).unwrap_or(NonZeroU16::MIN)),
            ),
            NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN),
        )
        .map_err(|err| PipelineError::EncodeFailed {
            path: target,
            reason: err.to_string(),
        })
    }

    #[test]
    fn encodes_ordered_three_frame_gif_and_webp() -> Result<(), PipelineError> {
        let root = temp_dir("ordered")?;
        let mut frames = Vec::new();
        for index in 0_u8..3 {
            let path = root.join(format!("frame_{index}.png"));
            write_png(
                &path,
                24,
                16,
                png::ColorType::Rgb,
                &solid_rgb(24, 16, index),
            )?;
            frames.push(frame(path, false)?);
        }
        let expected: Vec<Vec<u8>> = (0_u8..3).map(|i| solid_rgb(24, 16, i)).collect();

        let gif_target = root.join("out.gif");
        let gif_plan = plan(frames.clone(), gif_target.clone(), StackTargetFormat::Gif)?;
        let report = execute_frame_sequence(&gif_plan)?;
        assert_eq!(report.frames_encoded().get(), 3);
        assert!(report.output_bytes().get() > 0);
        let mut decoded = Vec::new();
        decode_gif_frames_streaming_pixels(&gif_target, |info, pixels| {
            assert_eq!((info.width, info.height, info.channels), (24, 16, 4));
            assert_eq!(info.duration_us, 100_000);
            decoded.push(pixels.to_vec());
            Ok(())
        })?;
        assert_eq!(decoded.len(), 3);
        for (frame, expected_rgb) in decoded.iter().zip(expected.iter()) {
            assert_close(frame, expected_rgb, 20);
            assert!(
                frame
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|pixel| pixel[3] == u8::MAX)
            );
        }

        let webp_target = root.join("out.webp");
        let webp_plan = plan(frames, webp_target.clone(), StackTargetFormat::Webp)?;
        let report = execute_frame_sequence(&webp_plan)?;
        assert_eq!(report.frames_encoded().get(), 3);
        let mut decoded = Vec::new();
        decode_webp_frames_streaming_pixels(&webp_target, |info, pixels| {
            assert_eq!((info.width, info.height, info.channels), (24, 16, 4));
            assert_eq!(info.duration_us, 100_000);
            decoded.push(pixels.to_vec());
            Ok(())
        })?;
        assert_eq!(decoded.len(), 3);
        // Lossy VP8 ringing around sharp block edges needs a generous tolerance;
        // frame count, geometry and ordering are exact.
        for (frame, expected_rgb) in decoded.iter().zip(expected.iter()) {
            assert_close(frame, expected_rgb, 20);
        }
        fs::remove_dir_all(&root).map_err(|source| PipelineError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn mixed_rgb_and_alpha_frames_preserve_transparency() -> Result<(), PipelineError> {
        let root = temp_dir("alpha")?;
        let rgb = root.join("rgb.png");
        let rgba = root.join("rgba.png");
        let opaque_pixels = rgb_gradient(24, 16, 0);
        let mut transparent_pixels = Vec::new();
        for (index, pixel) in rgb_gradient(24, 16, 1)
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
        {
            let alpha = if index % 2 == 0 { 0 } else { 255 };
            transparent_pixels.extend_from_slice(&[pixel[0], pixel[1], pixel[2], alpha]);
        }
        write_png(&rgb, 24, 16, png::ColorType::Rgb, &opaque_pixels)?;
        write_png(&rgba, 24, 16, png::ColorType::Rgba, &transparent_pixels)?;
        let frames = vec![frame(rgb, false)?, frame(rgba, true)?];

        for format in [StackTargetFormat::Gif, StackTargetFormat::Webp] {
            let extension = format.extension();
            let target = root.join(format!("mixed.{extension}"));
            execute_frame_sequence(&plan(frames.clone(), target.clone(), format)?)?;
            let mut alphas = Vec::new();
            let inspect = |info: crate::video_anim::AnimFrameInfo, pixels: &[u8]| {
                assert_eq!((info.width, info.height), (24, 16));
                alphas.push(pixels.as_chunks::<4>().0[0][3]);
                Ok(())
            };
            if format == StackTargetFormat::Gif {
                decode_gif_frames_streaming_pixels(&target, inspect)?;
            } else {
                decode_webp_frames_streaming_pixels(&target, inspect)?;
            }
            assert_eq!(alphas.len(), 2);
            assert_eq!(alphas[0], u8::MAX);
            assert_eq!(alphas[1], 0);
        }
        fs::remove_dir_all(&root).map_err(|source| PipelineError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn rejects_execution_time_alias() -> Result<(), PipelineError> {
        let root = temp_dir("alias")?;
        let source = root.join("source.png");
        write_png(
            &source,
            24,
            16,
            png::ColorType::Rgb,
            &rgb_gradient(24, 16, 0),
        )?;
        let frames = vec![frame(source, false)?];
        let nested = root.join("nested");
        fs::create_dir_all(&nested).map_err(|source| PipelineError::Io {
            path: nested.clone(),
            source,
        })?;
        // Textually different but canonicalizes to the source file.
        let alias_target = nested.join("..").join("source.png");
        let alias_plan = plan(frames, alias_target, StackTargetFormat::Gif)?;
        assert!(matches!(
            execute_frame_sequence(&alias_plan),
            Err(PipelineError::EncodeFailed { .. })
        ));
        fs::remove_dir_all(&root).map_err(|source| PipelineError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn reports_effective_worker_count_capped_at_encoder_limit() -> Result<(), PipelineError> {
        let root = temp_dir("worker-cap")?;
        let source = root.join("source.png");
        write_png(&source, 24, 16, png::ColorType::Rgb, &solid_rgb(24, 16, 0))?;
        let frames = vec![frame(source, false)?];

        let target = root.join("capped.gif");
        let fps = StackFps::new(10.0).map_err(|err| PipelineError::EncodeFailed {
            path: target.clone(),
            reason: err.to_string(),
        })?;
        let quality = VideoQuality::new(95).map_err(|err| PipelineError::EncodeFailed {
            path: target.clone(),
            reason: err.to_string(),
        })?;
        let capped_plan = FrameSequencePlan::new(
            frames.clone(),
            ImageDimensions::new(24, 16).map_err(|err| PipelineError::EncodeFailed {
                path: target.clone(),
                reason: err.to_string(),
            })?,
            fps,
            target,
            StackExecutionMode::animation(
                StackTargetFormat::Gif,
                quality,
                AnimationLoop::Times(NonZeroU16::new(2).unwrap_or(NonZeroU16::MIN)),
            ),
            NonZeroUsize::new(16).unwrap_or(NonZeroUsize::MIN),
        )
        .map_err(|err| PipelineError::EncodeFailed {
            path: PathBuf::from("frame-stack-test"),
            reason: err.to_string(),
        })?;
        let report = execute_frame_sequence(&capped_plan)?;
        assert_eq!(
            report.worker_threads().map(NonZeroUsize::get),
            Some(crate::video_anim::MAX_ANIM_WORKERS)
        );

        // A request below the cap must be reported unchanged.
        let uncapped_target = root.join("uncapped.gif");
        let uncapped_plan = plan(frames, uncapped_target, StackTargetFormat::Gif)?;
        let report = execute_frame_sequence(&uncapped_plan)?;
        assert_eq!(report.worker_threads().map(NonZeroUsize::get), Some(2));

        fs::remove_dir_all(&root).map_err(|source| PipelineError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }

    fn video_plan(
        frames: Vec<StackFrame>,
        target: PathBuf,
        container: VideoContainerFormat,
        mode: VideoEncodeMode,
    ) -> Result<FrameSequencePlan, PipelineError> {
        video_plan_dims(frames, target, container, mode, 24, 16)
    }

    fn video_plan_dims(
        frames: Vec<StackFrame>,
        target: PathBuf,
        container: VideoContainerFormat,
        mode: VideoEncodeMode,
        width: u32,
        height: u32,
    ) -> Result<FrameSequencePlan, PipelineError> {
        let plan_mode = StackExecutionMode::video(container, mode).map_err(|err| {
            PipelineError::EncodeFailed {
                path: target.clone(),
                reason: err.to_string(),
            }
        })?;
        FrameSequencePlan::new(
            frames,
            ImageDimensions::new(width, height).map_err(|err| PipelineError::EncodeFailed {
                path: target.clone(),
                reason: err.to_string(),
            })?,
            StackFps::new(10.0).map_err(|err| PipelineError::EncodeFailed {
                path: target.clone(),
                reason: err.to_string(),
            })?,
            target,
            plan_mode,
            NonZeroUsize::MIN,
        )
        .map_err(|err| PipelineError::EncodeFailed {
            path: PathBuf::from("frame-stack-test"),
            reason: err.to_string(),
        })
    }

    #[test]
    fn encodes_cpu_video_stacks_into_every_container_and_roundtrips() -> Result<(), PipelineError> {
        let root = temp_dir("cpu-video")?;
        let mut frames = Vec::new();
        for index in 0_u8..3 {
            let source = root.join(format!("frame_{index}.png"));
            write_png(
                &source,
                24,
                16,
                png::ColorType::Rgb,
                &solid_rgb(24, 16, index),
            )?;
            frames.push(frame(source, false)?);
        }
        let modes = [
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            VideoEncodeMode::CpuAv1 {
                rate_control: VideoRateControl::DEFAULT,
            },
            VideoEncodeMode::CpuVp9 {
                rate_control: VideoRateControl::DEFAULT,
            },
            VideoEncodeMode::CpuMjpeg {
                quality: VideoQuality::DEFAULT,
            },
        ];
        let containers = [
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Webm,
            VideoContainerFormat::Mov,
            VideoContainerFormat::Avi,
            VideoContainerFormat::Ts,
        ];
        for container in containers {
            for mode in modes {
                if !mode.codec().supported_in(container) {
                    continue;
                }
                let target = root.join(format!(
                    "{}-{}.{}",
                    container.extension(),
                    mode.codec().cli_name(),
                    container.extension()
                ));
                let report = execute_frame_sequence(&video_plan(
                    frames.clone(),
                    target.clone(),
                    container,
                    mode,
                )?)?;
                assert_eq!(report.frames_encoded().get(), 3);
                assert_eq!(report.target_container(), container);
                assert_eq!(report.video_codec(), mode.codec());
                assert_eq!(report.backend(), HardwareBackend::Cpu);
                let media = crate::video_container::demux_video(&target, container)?;
                assert!(
                    media.packets.windows(2).all(|w| w[0].dts < w[1].dts),
                    "non-monotonic DTS in {} {}",
                    container,
                    mode.codec()
                );
                let decoded = crate::video_codecs::decode_video_track(
                    &target,
                    &media.tracks[0],
                    &media.packets,
                )?;
                assert_eq!(decoded.len(), 3, "{} {}", container, mode.codec());
                assert_eq!((decoded[0].width, decoded[0].height), (24, 16));
                assert!(
                    decoded.windows(2).all(|w| w[0].pts_us < w[1].pts_us),
                    "non-monotonic PTS in {} {}",
                    container,
                    mode.codec()
                );
            }
        }
        fs::remove_dir_all(&root).map_err(|source| PipelineError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn video_stacks_flatten_translucent_and_grayscale_sources() -> Result<(), PipelineError> {
        let root = temp_dir("video-alpha")?;
        let rgba_path = root.join("rgba.png");
        let l8_path = root.join("l8.png");
        let la8_path = root.join("la8.png");
        // Fully transparent white must flatten to black (0, 0, 0), not leak 255.
        let transparent_white = [255_u8, 255, 255, 0].repeat((24 * 16) as usize);
        write_png(&rgba_path, 24, 16, png::ColorType::Rgba, &transparent_white)?;
        let gray = vec![160_u8; (24 * 16) as usize];
        write_png(&l8_path, 24, 16, png::ColorType::Grayscale, &gray)?;
        let gray_alpha = [200_u8, 0].repeat((24 * 16) as usize);
        write_png(
            &la8_path,
            24,
            16,
            png::ColorType::GrayscaleAlpha,
            &gray_alpha,
        )?;

        let frames = vec![
            frame(rgba_path, true)?,
            frame(l8_path, false)?,
            frame(la8_path, true)?,
        ];
        let target = root.join("mixed_formats.mp4");
        let report = execute_frame_sequence(&video_plan(
            frames,
            target.clone(),
            VideoContainerFormat::Mp4,
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
        )?)?;
        assert_eq!(report.frames_encoded().get(), 3);
        let media = crate::video_container::demux_video(&target, VideoContainerFormat::Mp4)?;
        let decoded =
            crate::video_codecs::decode_video_track(&target, &media.tracks[0], &media.packets)?;
        assert_eq!(decoded.len(), 3);
        // Frame 0 (transparent white flattened onto black) and Frame 2 (transparent LA8)
        // have near-black limited-range luma (~16); Frame 1 (L8=160) is visibly brighter.
        assert!(
            decoded[0].y[0] <= 24,
            "RGBA alpha leaked: {}",
            decoded[0].y[0]
        );
        assert!(
            decoded[1].y[0] >= 120,
            "L8 luma too dark: {}",
            decoded[1].y[0]
        );
        assert!(
            decoded[2].y[0] <= 24,
            "LA8 alpha leaked: {}",
            decoded[2].y[0]
        );

        // Odd-length corrupt buffer for a 2-channel format must fail explicitly.
        let mut flat = Vec::new();
        assert!(matches!(
            super::opaque_video_pixels(
                Path::new("corrupt.png"),
                crate::decode::DecodedPixelFormat::La8,
                &[128, 255, 64],
                &mut flat,
            ),
            Err(PipelineError::UnsupportedPixelFormat { .. })
        ));

        fs::remove_dir_all(&root).map_err(|source| PipelineError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn cpu_video_stack_rejects_decoded_budget_before_allocation() -> Result<(), PipelineError> {
        let root = temp_dir("budget")?;
        let frames = (0..2)
            .map(|index| frame(root.join(format!("missing_{index}.png")), false))
            .collect::<Result<Vec<_>, _>>()?;
        // 40k x 40k x 1.5 bytes per frame already exceeds the 1 GiB cap; the
        // budget check must fire before any frame is decoded or retained.
        let plan = FrameSequencePlan::new(
            frames,
            ImageDimensions::new(40_000, 40_000).map_err(|err| PipelineError::EncodeFailed {
                path: PathBuf::from("budget"),
                reason: err.to_string(),
            })?,
            StackFps::DEFAULT,
            root.join("budget.mp4"),
            StackExecutionMode::video(
                VideoContainerFormat::Mp4,
                VideoEncodeMode::CpuH264 {
                    rate_control: VideoRateControl::DEFAULT,
                },
            )
            .map_err(|err| PipelineError::EncodeFailed {
                path: PathBuf::from("budget"),
                reason: err.to_string(),
            })?,
            NonZeroUsize::MIN,
        )
        .map_err(|err| PipelineError::EncodeFailed {
            path: PathBuf::from("budget"),
            reason: err.to_string(),
        })?;
        assert!(
            matches!(
                execute_frame_sequence(&plan),
                Err(PipelineError::DecodeFailed { ref reason, .. })
                    if reason.contains("per-track budget")
            ),
            "expected the per-track decoded-budget guard, not an unrelated decode failure"
        );
        fs::remove_dir_all(&root).map_err(|source| PipelineError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }

    #[cfg(not(feature = "nvenc"))]
    #[test]
    fn nvenc_stack_fails_explicitly_without_the_feature() -> Result<(), PipelineError> {
        let root = temp_dir("nvenc-disabled")?;
        let source = root.join("frame.png");
        write_png(&source, 24, 16, png::ColorType::Rgb, &solid_rgb(24, 16, 0))?;
        let target = root.join("out.mp4");
        let plan = video_plan(
            vec![frame(source, false)?],
            target,
            VideoContainerFormat::Mp4,
            VideoEncodeMode::NvencH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
        )?;
        assert!(matches!(
            execute_frame_sequence(&plan),
            Err(PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                ..
            })
        ));
        fs::remove_dir_all(&root).map_err(|source| PipelineError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }

    #[cfg(feature = "nvenc")]
    #[test]
    fn nvenc_stack_encoding_reports_gpu_path_and_roundtrips() -> Result<(), PipelineError> {
        let root = temp_dir("nvenc-stack")?;
        let mut frames = Vec::new();
        for index in 0_u8..3 {
            let source = root.join(format!("frame_{index}.png"));
            write_png(
                &source,
                320,
                240,
                png::ColorType::Rgb,
                &solid_rgb(320, 240, index),
            )?;
            frames.push(frame(source, false)?);
        }
        for (codec, container) in [
            (TargetVideoCodec::H264, VideoContainerFormat::Mp4),
            (TargetVideoCodec::Hevc, VideoContainerFormat::Mkv),
        ] {
            let mode = match codec {
                TargetVideoCodec::H264 => VideoEncodeMode::NvencH264 {
                    rate_control: VideoRateControl::DEFAULT,
                },
                TargetVideoCodec::Hevc => VideoEncodeMode::NvencHevc {
                    rate_control: VideoRateControl::DEFAULT,
                },
                _ => {
                    return Err(PipelineError::EncodeFailed {
                        path: PathBuf::from("nvenc-stack-test"),
                        reason: "unexpected codec".to_owned(),
                    });
                }
            };
            // CUDA context creation alone does not establish NVENC support
            // (e.g. A100 has CUDA but no NVENC). Skip only codecs this adapter
            // cannot encode with NV12 input; real encode/mux/decode failures
            // below remain test failures.
            if !crate::gpu::nvenc::codec_supports_encode_enabled(codec.as_video_codec())
                .unwrap_or(false)
            {
                continue;
            }
            let target = root.join(format!(
                "out-{}.{}",
                codec.cli_name(),
                container.extension()
            ));
            let report = execute_frame_sequence(&video_plan_dims(
                frames.clone(),
                target.clone(),
                container,
                mode,
                320,
                240,
            )?)?;
            assert_eq!(report.frames_encoded().get(), 3);
            assert_eq!(report.backend(), HardwareBackend::Gpu(GpuBackend::Nvenc));
            assert!(
                report
                    .hardware_device()
                    .is_some_and(|name| !name.is_empty())
            );
            let media = crate::video_container::demux_video(&target, container)?;
            let decoded =
                crate::video_codecs::decode_video_track(&target, &media.tracks[0], &media.packets)?;
            assert_eq!(decoded.len(), 3);
        }
        fs::remove_dir_all(&root).map_err(|source| PipelineError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }
}
