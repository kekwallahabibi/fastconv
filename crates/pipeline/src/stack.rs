//! Concrete CPU execution of a validated [`FrameSequencePlan`].
//!
//! Sources are decoded one at a time into reusable scratch buffers and
//! streamed straight into the shared animation encoder; no frame list is ever
//! retained in memory. Every legal state is described by the domain plan, so
//! there is no backend switch and no CPU fallback here.

use std::collections::VecDeque;
use std::fs;
use std::io;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Component, Path, PathBuf};

use fastconv_domain::{FrameSequencePlan, FrameSequenceReport, ImageDimensions, StackFrame};

use crate::PipelineError;
use crate::decode::{DecodeTargetHint, DecodedPixelFormat, decode_source_image_into};
use crate::staging::StagingTarget;
use crate::video_anim::{AnimFrameInfo, encode_animation_stream_bounded};

/// Raw-pixel ceiling for in-flight animation frames (256 MiB).
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

/// Executes a validated frame sequence into its animated GIF/WebP target.
///
/// # Errors
///
/// Returns [`PipelineError`] on an execution-time alias, decode failure,
/// transparent pixels in an opaque stack, encoder failure, or empty output.
pub fn execute_frame_sequence(
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
    let workers = plan.worker_threads().get().min(memory_workers).max(1);

    let staging = StagingTarget::create_with_handle(plan.target_path(), 0)?;
    let mut file_bytes = Vec::new();
    let mut pixel_buf = Vec::new();
    let mut normalized = Vec::new();

    let written = {
        let file = staging.file()?;
        let fps = plan.fps();
        encode_animation_stream_bounded(
            plan.target_format().as_video_container(),
            plan.quality(),
            workers,
            plan.loop_count(),
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
    let output_bytes = staging
        .file()?
        .metadata()
        .map_err(|source| PipelineError::Io {
            path: plan.target_path().to_path_buf(),
            source,
        })?;
    let output_bytes = NonZeroU64::new(output_bytes.len())
        .ok_or_else(|| PipelineError::EmptyOutput(plan.target_path().to_path_buf()))?;
    let worker_threads = NonZeroUsize::new(workers).unwrap_or_else(|| plan.worker_threads());
    // Re-check immediately before atomic rename so a symlink planted during
    // encoding cannot overwrite a source image.
    verify_no_alias(plan)?;
    staging.commit()?;

    Ok(FrameSequenceReport::new(
        plan.frame_count(),
        dimensions,
        plan.fps().get(),
        plan.target_format(),
        plan.target_path().to_path_buf(),
        output_bytes,
        worker_threads,
    ))
}

#[cfg(test)]
mod tests {
    use super::execute_frame_sequence;
    use crate::PipelineError;
    use crate::video_anim::{
        decode_gif_frames_streaming_pixels, decode_webp_frames_streaming_pixels,
    };
    use fastconv_domain::{
        AnimationLoop, FrameSequencePlan, ImageDimensions, ImageInputFormat, StackExecutionMode,
        StackFps, StackFrame, StackTargetFormat, VideoQuality,
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
            StackExecutionMode::cpu_animation(format, quality),
            AnimationLoop::Times(NonZeroU16::new(2).unwrap_or(NonZeroU16::MIN)),
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
}
