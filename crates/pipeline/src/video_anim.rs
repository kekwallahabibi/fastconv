//! Animated GIF and animated WebP decoding/encoding with BT.601 colour math.
//!
//! Everything in this module is safe Rust. Animation frames are composited in
//! RGB space (never through 4:2:0) on the decode-to-encode path, and encoded
//! with bounded parallel worker pools that preserve strict display order.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::num::NonZeroU64;
use std::ops::ControlFlow;
use std::path::Path;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use fastconv_domain::{AnimationLoop, VideoContainerFormat, VideoQuality};
use yuv::{
    BufferStoreMut, YuvBiPlanarImage, YuvBiPlanarImageMut, YuvConversionMode, YuvPlanarImageMut,
    YuvRange as SimdYuvRange, YuvStandardMatrix,
};

use crate::PipelineError;
use crate::video_codecs::{Yuv420Frame, YuvColor, YuvMatrix, YuvRange};
use crate::video_container::{ContainerMedia, ContainerPacket, ContainerTrack, US_TIMESCALE};

/// Decoder memory ceiling for both animation codecs (512 MiB).
const ANIM_MEMORY_LIMIT: u64 = 512 * 1024 * 1024;
/// Total decoded bytes one animation track may retain during a full decode.
pub(crate) const ANIM_DECODED_BUDGET: u64 = 1_073_741_824;
/// Upper bound on parallel per-frame quantisation/encoding workers.
const MAX_ANIM_WORKERS: usize = 8;
/// Private sentinel used to translate a `ControlFlow::Break` from a streaming
/// callback through the older `Result`-based decoder plumbing. It never leaves
/// the extraction pipeline as a user-visible error.
pub(crate) const FRAMES_EARLY_STOP: &str = "\u{0}fastconv-frames-early-stop";

fn decode_error(path: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::DecodeFailed {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

fn encode_error(path: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

fn io_error(path: &Path, source: std::io::Error) -> PipelineError {
    PipelineError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn clamp_u8(value: i32) -> u8 {
    u8::try_from(value.clamp(0, 255)).unwrap_or(0)
}

/// Rounds a dimension up to the even value required by 4:2:0 chroma.
fn even_dim(value: u32) -> Option<u32> {
    value.checked_add(value & 1)
}

fn byte_len(width: u32, height: u32, bpp: usize) -> Result<usize, PipelineError> {
    let path = Path::new("animation-frame");
    let pixels = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .ok_or_else(|| decode_error(path, "animation frame dimensions overflow usize"))?;
    pixels
        .checked_mul(bpp)
        .ok_or_else(|| decode_error(path, "animation frame buffer overflows usize"))
}

/// Geometry, exact timing and pixel layout of one decoded animation frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AnimFrameInfo {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pts_us: i64,
    pub(crate) duration_us: u64,
    /// 3 for flattened RGB, 4 when transparency is preserved.
    pub(crate) channels: u8,
}

// ---------------------------------------------------------------------------
// BT.601 limited-range integer colour conversion
// ---------------------------------------------------------------------------

fn rgb_to_yuv_block(red: i32, green: i32, blue: i32) -> (u8, u8, u8) {
    let luma = (((66 * red + 129 * green + 25 * blue + 128) >> 8) + 16).clamp(16, 235);
    let chroma_u = (((-38 * red - 74 * green + 112 * blue + 128) >> 8) + 128).clamp(16, 240);
    let chroma_v = (((112 * red - 94 * green - 18 * blue + 128) >> 8) + 128).clamp(16, 240);
    (
        u8::try_from(luma).unwrap_or(235),
        u8::try_from(chroma_u).unwrap_or(128),
        u8::try_from(chroma_v).unwrap_or(128),
    )
}

/// SIMD conversions use BT.601 limited range — the same colour description the
/// scalar path and the downstream H.264/NV12 encoders assume.
const SIMD_RANGE: SimdYuvRange = SimdYuvRange::Limited;
const SIMD_MATRIX: YuvStandardMatrix = YuvStandardMatrix::Bt601;
const SIMD_MODE: YuvConversionMode = YuvConversionMode::Balanced;

/// True when every pixel of an RGBA frame is fully opaque. Such frames can skip
/// the premultiplication/flattening copy and feed `rgba_to_yuv*` directly.
fn rgba_is_opaque(rgba: &[u8]) -> bool {
    rgba.as_chunks::<4>().0.iter().all(|px| px[3] == 255)
}

fn packed_stride(width: usize, channels: usize, path: &Path) -> Result<u32, PipelineError> {
    let stride = width
        .checked_mul(channels)
        .ok_or_else(|| decode_error(path, "animation row stride overflow"))?;
    u32::try_from(stride).map_err(|_| decode_error(path, "animation row stride exceeds u32"))
}

fn even_geometry(
    path: &Path,
    width: u32,
    height: u32,
) -> Result<(u32, u32, usize, usize), PipelineError> {
    let even_width = even_dim(width).ok_or_else(|| decode_error(path, "width overflow"))?;
    let even_height = even_dim(height).ok_or_else(|| decode_error(path, "height overflow"))?;
    if even_width == 0 || even_height == 0 {
        return Err(decode_error(path, "zero-sized animation frame"));
    }
    let padded_w =
        usize::try_from(even_width).map_err(|_| decode_error(path, "padded width overflow"))?;
    let padded_h =
        usize::try_from(even_height).map_err(|_| decode_error(path, "padded height overflow"))?;
    Ok((even_width, even_height, padded_w, padded_h))
}

/// SIMD conversion of an even-dimension, tightly packed frame. `channels` is
/// 3 for RGB8 and 4 for RGBA8.
fn yuv420_simd_frame(
    info: AnimFrameInfo,
    pixels: &[u8],
    channels: usize,
    padded_w: usize,
    padded_h: usize,
) -> Result<Yuv420Frame, PipelineError> {
    let path = Path::new("animation-yuv420-simd");
    let chroma_w = padded_w / 2;
    let chroma_h = padded_h / 2;
    let mut y_plane = vec![0_u8; padded_w * padded_h];
    let mut u_plane = vec![0_u8; chroma_w * chroma_h];
    let mut v_plane = vec![0_u8; chroma_w * chroma_h];
    let stride = packed_stride(padded_w, channels, path)?;
    let mut image = YuvPlanarImageMut {
        y_plane: BufferStoreMut::Borrowed(&mut y_plane),
        y_stride: u32::try_from(padded_w)
            .map_err(|_| decode_error(path, "padded width exceeds u32"))?,
        u_plane: BufferStoreMut::Borrowed(&mut u_plane),
        u_stride: u32::try_from(chroma_w)
            .map_err(|_| decode_error(path, "chroma width exceeds u32"))?,
        v_plane: BufferStoreMut::Borrowed(&mut v_plane),
        v_stride: u32::try_from(chroma_w)
            .map_err(|_| decode_error(path, "chroma width exceeds u32"))?,
        width: info.width,
        height: info.height,
    };
    let converted = if channels == 4 {
        yuv::rgba_to_yuv420(
            &mut image,
            pixels,
            stride,
            SIMD_RANGE,
            SIMD_MATRIX,
            SIMD_MODE,
        )
    } else {
        yuv::rgb_to_yuv420(
            &mut image,
            pixels,
            stride,
            SIMD_RANGE,
            SIMD_MATRIX,
            SIMD_MODE,
        )
    };
    converted.map_err(|err| decode_error(path, format!("SIMD YUV420 conversion failed: {err}")))?;
    Ok(Yuv420Frame {
        width: info.width,
        height: info.height,
        y: y_plane,
        u: u_plane,
        v: v_plane,
        pts_us: info.pts_us,
        duration_us: info.duration_us,
        color: YuvColor::default(),
    })
}

/// Converts tightly packed RGB8 into planar YUV420, padding odd dimensions by
/// replicating the rightmost column and bottommost row. Even dimensions use
/// the SIMD converter; odd dimensions keep the exact scalar fallback.
pub(crate) fn rgb_to_yuv420(info: AnimFrameInfo, rgb: &[u8]) -> Result<Yuv420Frame, PipelineError> {
    let path = Path::new("animation-rgb-to-yuv420");
    if rgb.len() != byte_len(info.width, info.height, 3)? {
        return Err(decode_error(
            path,
            format!(
                "RGB frame buffer is {} bytes, expected {}",
                rgb.len(),
                byte_len(info.width, info.height, 3)?
            ),
        ));
    }
    let (even_width, even_height, padded_w, padded_h) =
        even_geometry(path, info.width, info.height)?;
    if info.width == even_width && info.height == even_height {
        return yuv420_simd_frame(info, rgb, 3, padded_w, padded_h);
    }

    let source_w = usize::try_from(info.width).map_err(|_| decode_error(path, "width overflow"))?;
    let source_h =
        usize::try_from(info.height).map_err(|_| decode_error(path, "height overflow"))?;
    let right = source_w.saturating_sub(1);
    let bottom = source_h.saturating_sub(1);
    let pixel = |sample_x: usize, sample_y: usize| -> (i32, i32, i32) {
        let idx = (sample_y.min(bottom) * source_w + sample_x.min(right)) * 3;
        (
            i32::from(rgb[idx]),
            i32::from(rgb[idx + 1]),
            i32::from(rgb[idx + 2]),
        )
    };
    let chroma_w = padded_w / 2;
    let chroma_h = padded_h / 2;
    let mut y_plane = vec![0_u8; padded_w * padded_h];
    for (row_y, out) in y_plane.chunks_exact_mut(padded_w).enumerate() {
        for (sample_x, dst) in out.iter_mut().enumerate() {
            let (red, green, blue) = pixel(sample_x, row_y);
            *dst = rgb_to_yuv_block(red, green, blue).0;
        }
    }
    let mut u_plane = vec![0_u8; chroma_w * chroma_h];
    let mut v_plane = vec![0_u8; chroma_w * chroma_h];
    for chroma_y in 0..chroma_h {
        for chroma_x in 0..chroma_w {
            let mut sum_red = 0_i32;
            let mut sum_green = 0_i32;
            let mut sum_blue = 0_i32;
            for block_y in 0..2 {
                for block_x in 0..2 {
                    let (red, green, blue) = pixel(chroma_x * 2 + block_x, chroma_y * 2 + block_y);
                    sum_red = sum_red.saturating_add(red);
                    sum_green = sum_green.saturating_add(green);
                    sum_blue = sum_blue.saturating_add(blue);
                }
            }
            let (_, chroma_u, chroma_v) =
                rgb_to_yuv_block(sum_red / 4, sum_green / 4, sum_blue / 4);
            u_plane[chroma_y * chroma_w + chroma_x] = chroma_u;
            v_plane[chroma_y * chroma_w + chroma_x] = chroma_v;
        }
    }
    Ok(Yuv420Frame {
        width: even_width,
        height: even_height,
        y: y_plane,
        u: u_plane,
        v: v_plane,
        pts_us: info.pts_us,
        duration_us: info.duration_us,
        color: YuvColor::default(),
    })
}

/// RGBA-aware entry point used by the animation decoders. Fully opaque RGBA
/// frames go straight to the SIMD RGBA converter (no flattening copy); frames
/// with transparency are premultiplied onto black exactly as before.
fn pixels_to_yuv420(
    info: AnimFrameInfo,
    pixels: &[u8],
    channels: usize,
) -> Result<Yuv420Frame, PipelineError> {
    let path = Path::new("animation-pixels-to-yuv420");
    match channels {
        3 => rgb_to_yuv420(info, pixels),
        4 => {
            if pixels.len() != byte_len(info.width, info.height, 4)? {
                return Err(decode_error(path, "RGBA frame buffer has the wrong length"));
            }
            let (even_width, even_height, padded_w, padded_h) =
                even_geometry(path, info.width, info.height)?;
            if info.width == even_width && info.height == even_height && rgba_is_opaque(pixels) {
                return yuv420_simd_frame(info, pixels, 4, padded_w, padded_h);
            }
            let flat = rgba_to_rgb(pixels, 4, info.width, info.height)?;
            rgb_to_yuv420(info, &flat)
        }
        other => Err(decode_error(
            path,
            format!("unsupported animation pixel layout: {other} channels"),
        )),
    }
}

fn nv12_validate_dst(
    path: &Path,
    width: u32,
    height: u32,
    dst: &[u8],
    pitch: usize,
) -> Result<(u32, u32, usize, usize), PipelineError> {
    let (even_width, even_height, padded_w, padded_h) = even_geometry(path, width, height)?;
    if pitch < padded_w {
        return Err(decode_error(
            path,
            format!("NV12 pitch {pitch} is smaller than padded width {padded_w}"),
        ));
    }
    let rows = padded_h
        .checked_add(padded_h / 2)
        .ok_or_else(|| decode_error(path, "NV12 row count overflow"))?;
    let needed = pitch
        .checked_mul(rows)
        .ok_or_else(|| decode_error(path, "NV12 buffer length overflow"))?;
    if dst.len() < needed {
        return Err(decode_error(
            path,
            format!("NV12 buffer is {} bytes, needs {needed}", dst.len()),
        ));
    }
    Ok((even_width, even_height, padded_w, padded_h))
}

/// SIMD conversion of an even-dimension, tightly packed frame into a pitched
/// NV12 destination. `channels` is 3 for RGB8 and 4 for RGBA8.
#[allow(clippy::too_many_arguments)]
fn nv12_simd_write(
    path: &Path,
    pixels: &[u8],
    width: u32,
    height: u32,
    dst: &mut [u8],
    pitch: usize,
    channels: usize,
    padded_w: usize,
    padded_h: usize,
) -> Result<(), PipelineError> {
    let uv_base = padded_h
        .checked_mul(pitch)
        .ok_or_else(|| decode_error(path, "NV12 UV offset overflow"))?;
    let y_len = pitch
        .checked_mul(padded_h)
        .ok_or_else(|| decode_error(path, "NV12 luma region overflow"))?;
    let uv_len = pitch
        .checked_mul(padded_h / 2)
        .ok_or_else(|| decode_error(path, "NV12 chroma region overflow"))?;
    let (y_region, uv_region) = dst.split_at_mut(uv_base);
    let mut image = YuvBiPlanarImageMut {
        y_plane: BufferStoreMut::Borrowed(&mut y_region[..y_len]),
        y_stride: u32::try_from(pitch).map_err(|_| decode_error(path, "NV12 pitch exceeds u32"))?,
        uv_plane: BufferStoreMut::Borrowed(&mut uv_region[..uv_len]),
        uv_stride: u32::try_from(pitch)
            .map_err(|_| decode_error(path, "NV12 pitch exceeds u32"))?,
        width,
        height,
    };
    let stride = packed_stride(padded_w, channels, path)?;
    let converted = if channels == 4 {
        yuv::rgba_to_yuv_nv12(
            &mut image,
            pixels,
            stride,
            SIMD_RANGE,
            SIMD_MATRIX,
            SIMD_MODE,
        )
    } else {
        yuv::rgb_to_yuv_nv12(
            &mut image,
            pixels,
            stride,
            SIMD_RANGE,
            SIMD_MATRIX,
            SIMD_MODE,
        )
    };
    converted.map_err(|err| decode_error(path, format!("SIMD NV12 conversion failed: {err}")))
}

/// Writes tightly packed RGB8 as pitched NV12, padding odd dimensions by
/// replicating edge pixels. Rows beyond `width` are left untouched.
pub(crate) fn write_nv12(
    rgb: &[u8],
    width: u32,
    height: u32,
    dst: &mut [u8],
    pitch: usize,
) -> Result<(), PipelineError> {
    let path = Path::new("animation-write-nv12");
    if rgb.len() != byte_len(width, height, 3)? {
        return Err(decode_error(path, "RGB frame buffer has the wrong length"));
    }
    let (even_width, even_height, padded_w, padded_h) =
        nv12_validate_dst(path, width, height, dst, pitch)?;
    if width == even_width && height == even_height {
        return nv12_simd_write(path, rgb, width, height, dst, pitch, 3, padded_w, padded_h);
    }

    let source_w = usize::try_from(width).map_err(|_| decode_error(path, "width overflow"))?;
    let source_h = usize::try_from(height).map_err(|_| decode_error(path, "height overflow"))?;
    let right = source_w.saturating_sub(1);
    let bottom = source_h.saturating_sub(1);
    let pixel = |sample_x: usize, sample_y: usize| -> (i32, i32, i32) {
        let idx = (sample_y.min(bottom) * source_w + sample_x.min(right)) * 3;
        (
            i32::from(rgb[idx]),
            i32::from(rgb[idx + 1]),
            i32::from(rgb[idx + 2]),
        )
    };
    for row_y in 0..padded_h {
        let base = row_y * pitch;
        for (sample_x, dst_byte) in dst[base..base + padded_w].iter_mut().enumerate() {
            let (red, green, blue) = pixel(sample_x, row_y);
            *dst_byte = rgb_to_yuv_block(red, green, blue).0;
        }
    }
    let uv_base = padded_h * pitch;
    for chroma_y in 0..padded_h / 2 {
        for chroma_x in 0..padded_w / 2 {
            let mut sum_red = 0_i32;
            let mut sum_green = 0_i32;
            let mut sum_blue = 0_i32;
            for block_y in 0..2 {
                for block_x in 0..2 {
                    let (red, green, blue) = pixel(chroma_x * 2 + block_x, chroma_y * 2 + block_y);
                    sum_red = sum_red.saturating_add(red);
                    sum_green = sum_green.saturating_add(green);
                    sum_blue = sum_blue.saturating_add(blue);
                }
            }
            let (_, chroma_u, chroma_v) =
                rgb_to_yuv_block(sum_red / 4, sum_green / 4, sum_blue / 4);
            dst[uv_base + chroma_y * pitch + chroma_x * 2] = chroma_u;
            dst[uv_base + chroma_y * pitch + chroma_x * 2 + 1] = chroma_v;
        }
    }
    Ok(())
}

/// RGBA-aware NV12 entry point used by the animation decoders. Opaque RGBA
/// frames avoid the flattening copy entirely.
fn write_pixels_nv12(
    pixels: &[u8],
    width: u32,
    height: u32,
    dst: &mut [u8],
    pitch: usize,
    channels: usize,
) -> Result<(), PipelineError> {
    let path = Path::new("animation-write-nv12");
    match channels {
        3 => write_nv12(pixels, width, height, dst, pitch),
        4 => {
            if pixels.len() != byte_len(width, height, 4)? {
                return Err(decode_error(path, "RGBA frame buffer has the wrong length"));
            }
            let (even_width, even_height, padded_w, padded_h) =
                nv12_validate_dst(path, width, height, dst, pitch)?;
            if width == even_width && height == even_height && rgba_is_opaque(pixels) {
                return nv12_simd_write(
                    path, pixels, width, height, dst, pitch, 4, padded_w, padded_h,
                );
            }
            let flat = rgba_to_rgb(pixels, 4, width, height)?;
            write_nv12(&flat, width, height, dst, pitch)
        }
        other => Err(decode_error(
            path,
            format!("unsupported animation pixel layout: {other} channels"),
        )),
    }
}

/// Maps the signalled source colour description onto the SIMD converter.
/// Unsupported matrices stay a hard error at the RGB boundary: they are legal
/// on video-to-video paths that never invert the matrix, but cannot be
/// converted to RGB without guessing coefficients.
fn simd_color(color: YuvColor) -> Result<(SimdYuvRange, YuvStandardMatrix), PipelineError> {
    let range = match color.range {
        YuvRange::Limited => SimdYuvRange::Limited,
        YuvRange::Full => SimdYuvRange::Full,
    };
    let matrix = match color.matrix {
        YuvMatrix::Bt601 => YuvStandardMatrix::Bt601,
        YuvMatrix::Bt709 => YuvStandardMatrix::Bt709,
        YuvMatrix::Bt2020 => YuvStandardMatrix::Bt2020,
        YuvMatrix::Unsupported(code) => {
            return Err(decode_error(
                Path::new("animation-nv12-to-rgb8"),
                format!("YUV→RGB conversion for matrix coefficient {code} is not supported"),
            ));
        }
    };
    Ok((range, matrix))
}

/// Inverse conversion of pitched NV12 into packed RGB8 using the source's
/// signalled colour matrix and range into a caller-owned buffer. The buffer is
/// reused across frames so extraction never allocates a fresh RGB image per
/// decoded frame.
pub(crate) fn nv12_to_rgb8_into(
    nv12: &[u8],
    dst: &mut [u8],
    pitch: usize,
    width: u32,
    height: u32,
    color: YuvColor,
) -> Result<(), PipelineError> {
    let path = Path::new("animation-nv12-to-rgb8");
    if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(decode_error(
            path,
            format!("NV12 requires even, non-zero dimensions; got {width}x{height}"),
        ));
    }
    let frame_w = usize::try_from(width).map_err(|_| decode_error(path, "width overflow"))?;
    let frame_h = usize::try_from(height).map_err(|_| decode_error(path, "height overflow"))?;
    if pitch < frame_w {
        return Err(decode_error(
            path,
            format!("NV12 pitch {pitch} is smaller than width {frame_w}"),
        ));
    }
    let uv_base = frame_h
        .checked_mul(pitch)
        .ok_or_else(|| decode_error(path, "NV12 offset overflow"))?;
    let needed = uv_base
        .checked_add((frame_h / 2) * pitch)
        .ok_or_else(|| decode_error(path, "NV12 buffer length overflow"))?;
    if nv12.len() < needed {
        return Err(decode_error(
            path,
            format!("NV12 buffer is {} bytes, needs {needed}", nv12.len()),
        ));
    }
    let rgb_len = byte_len(width, height, 3)?;
    if dst.len() < rgb_len {
        return Err(decode_error(
            path,
            format!("RGB buffer is {} bytes, needs {rgb_len}", dst.len()),
        ));
    }
    let (range, matrix) = simd_color(color)?;
    let stride = u32::try_from(pitch).map_err(|_| decode_error(path, "NV12 pitch exceeds u32"))?;
    let rgb_stride = u32::try_from(frame_w.saturating_mul(3))
        .map_err(|_| decode_error(path, "RGB row stride exceeds u32"))?;
    let image = YuvBiPlanarImage {
        y_plane: &nv12[..uv_base],
        y_stride: stride,
        uv_plane: &nv12[uv_base..],
        uv_stride: stride,
        width,
        height,
    };
    yuv::yuv_nv12_to_rgb(
        &image,
        &mut dst[..rgb_len],
        rgb_stride,
        range,
        matrix,
        SIMD_MODE,
    )
    .map_err(|err| decode_error(path, format!("SIMD NV12 → RGB conversion failed: {err}")))
}

/// Inverse conversion of pitched NV12 into a freshly allocated packed RGB8.
/// Prefer [`nv12_to_rgb8_into`] on hot paths that can recycle a buffer.
pub(crate) fn nv12_to_rgb8(
    nv12: &[u8],
    pitch: usize,
    width: u32,
    height: u32,
    color: YuvColor,
) -> Result<Vec<u8>, PipelineError> {
    let mut rgb = vec![0_u8; byte_len(width, height, 3)?];
    nv12_to_rgb8_into(nv12, &mut rgb, pitch, width, height, color)?;
    Ok(rgb)
}

// ---------------------------------------------------------------------------
// Metadata demuxers (lightweight packets, no pixel data)
// ---------------------------------------------------------------------------

/// Fast metadata scan of an animated GIF: one track and one packet per frame.
pub(crate) fn demux_gif_metadata(path: &Path) -> Result<ContainerMedia, PipelineError> {
    let mut options = gif::DecodeOptions::new();
    options.skip_frame_decoding(true);
    options.set_memory_limit(gif::MemoryLimit::Bytes(
        NonZeroU64::new(ANIM_MEMORY_LIMIT).unwrap_or(NonZeroU64::MIN),
    ));
    let file = File::open(path).map_err(|source| io_error(path, source))?;
    let mut decoder = options.read_info(BufReader::new(file)).map_err(|err| {
        decode_error(
            path,
            format!("GIF header or frame metadata is invalid: {err}"),
        )
    })?;
    let width = u32::from(decoder.width());
    let height = u32::from(decoder.height());
    if width == 0 || height == 0 {
        return Err(decode_error(path, "GIF canvas has a zero width or height"));
    }
    let even_w = u16::try_from(even_dim(width).unwrap_or(u32::from(u16::MAX))).unwrap_or(u16::MAX);
    let even_h = u16::try_from(even_dim(height).unwrap_or(u32::from(u16::MAX))).unwrap_or(u16::MAX);

    let mut pts_us = 0_i64;
    let mut packets = Vec::new();
    while let Some(info) = decoder
        .next_frame_info()
        .map_err(|err| decode_error(path, format!("GIF frame metadata is invalid: {err}")))?
    {
        let duration = if info.delay == 0 {
            100_000_u64
        } else {
            u64::from(info.delay) * 10_000
        };
        packets.push(ContainerPacket {
            track: 0,
            pts: pts_us,
            dts: pts_us,
            duration,
            keyframe: true,
            data: Vec::new(),
        });
        pts_us = pts_us.saturating_add(i64::try_from(duration).unwrap_or(i64::MAX));
    }
    if packets.is_empty() {
        return Err(decode_error(path, "GIF contains no frames"));
    }
    Ok(ContainerMedia {
        tracks: vec![ContainerTrack {
            codec: fastconv_domain::VideoCodec::Gif,
            width: Some(even_w),
            height: Some(even_h),
            channels: None,
            sample_rate: None,
            config: Vec::new(),
            timescale: US_TIMESCALE,
            presentation_edit: None,
            pcm_format: None,
        }],
        packets,
    })
}

fn le_u32_at(bytes: &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?))
}

fn le_u24_at(bytes: &[u8]) -> Option<u32> {
    let raw = bytes.get(..3)?;
    Some(u32::from(raw[0]) | (u32::from(raw[1]) << 8) | (u32::from(raw[2]) << 16))
}

fn read_at(path: &Path, file: &mut File, offset: u64, buf: &mut [u8]) -> Result<(), PipelineError> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| io_error(path, source))?;
    file.read_exact(buf)
        .map_err(|source| io_error(path, source))?;
    Ok(())
}

#[derive(Default)]
struct WebpScan {
    still: Option<(u32, u32)>,
    extended: Option<(bool, u32, u32)>,
    animated: bool,
    durations: Vec<u64>,
}

/// Visits top-level RIFF chunks with only fixed-size reads and seeks.
fn scan_webp_chunks(
    path: &Path,
    file: &mut File,
    riff_end: u64,
) -> Result<WebpScan, PipelineError> {
    let mut scan = WebpScan::default();
    let mut pos = 12_u64;
    while pos.saturating_add(8) <= riff_end {
        let mut header = [0_u8; 8];
        read_at(path, file, pos, &mut header)?;
        let id = [header[0], header[1], header[2], header[3]];
        let size = u64::from(
            le_u32_at(&header[4..])
                .ok_or_else(|| decode_error(path, "WebP chunk size is truncated"))?,
        );
        let payload_start = pos.saturating_add(8);
        let payload_end = payload_start
            .checked_add(size)
            .ok_or_else(|| decode_error(path, "WebP chunk size overflows"))?;
        if payload_end > riff_end {
            return Err(decode_error(path, "WebP RIFF chunk is truncated"));
        }
        match &id {
            b"VP8 " if scan.still.is_none() => {
                let mut prefix = [0_u8; 10];
                if size < 10 {
                    return Err(decode_error(path, "WebP VP8 chunk is truncated"));
                }
                read_at(path, file, payload_start, &mut prefix)?;
                scan.still = Some(parse_vp8_dimensions(&prefix).ok_or_else(|| {
                    decode_error(path, "WebP VP8 bitstream has invalid dimensions")
                })?);
            }
            b"VP8L" if scan.still.is_none() => {
                let mut prefix = [0_u8; 5];
                if size < 5 {
                    return Err(decode_error(path, "WebP VP8L chunk is truncated"));
                }
                read_at(path, file, payload_start, &mut prefix)?;
                scan.still = Some(parse_vp8l_dimensions(&prefix).ok_or_else(|| {
                    decode_error(path, "WebP VP8L bitstream has invalid dimensions")
                })?);
            }
            b"VP8X" if scan.extended.is_none() => {
                let mut prefix = [0_u8; 10];
                if size < 10 {
                    return Err(decode_error(path, "WebP VP8X chunk is truncated"));
                }
                read_at(path, file, payload_start, &mut prefix)?;
                let width = le_u24_at(&prefix[4..7])
                    .ok_or_else(|| decode_error(path, "WebP VP8X width is truncated"))?
                    .saturating_add(1);
                let height = le_u24_at(&prefix[7..10])
                    .ok_or_else(|| decode_error(path, "WebP VP8X height is truncated"))?
                    .saturating_add(1);
                scan.extended = Some((prefix[0] & 0x02 != 0, width, height));
            }
            b"ANIM" => {
                if size < 6 {
                    return Err(decode_error(
                        path,
                        "WebP ANIM chunk is shorter than six bytes",
                    ));
                }
                scan.animated = true;
            }
            b"ANMF" => {
                let mut duration = [0_u8; 3];
                if size < 16 {
                    return Err(decode_error(path, "WebP ANMF chunk is too short"));
                }
                read_at(path, file, payload_start.saturating_add(12), &mut duration)?;
                let ms = le_u24_at(&duration)
                    .ok_or_else(|| decode_error(path, "WebP ANMF duration is truncated"))?;
                if !anmf_has_bitstream(path, file, payload_start, payload_end)? {
                    return Err(decode_error(
                        path,
                        "WebP ANMF has no nested VP8 or VP8L bitstream",
                    ));
                }
                // A 24-bit millisecond value overflows u32 when scaled to microseconds.
                let duration_us = if ms == 0 {
                    100_000
                } else {
                    u64::from(ms).saturating_mul(1_000)
                };
                scan.durations.push(duration_us);
            }
            _ => {}
        }
        let padded_end = payload_end.saturating_add(size & 1);
        if padded_end > riff_end {
            return Err(decode_error(path, "WebP RIFF chunk padding is truncated"));
        }
        pos = padded_end;
    }
    Ok(scan)
}

/// Mirrors the probe's RFC 9649 check so directly decoded files cannot smuggle
/// an ANMF without a nested image bitstream past the boundary validation.
fn anmf_has_bitstream(
    path: &Path,
    file: &mut File,
    payload_start: u64,
    payload_end: u64,
) -> Result<bool, PipelineError> {
    let mut pos = payload_start.saturating_add(16);
    while pos.saturating_add(8) <= payload_end {
        let mut header = [0_u8; 8];
        read_at(path, file, pos, &mut header)?;
        let id = [header[0], header[1], header[2], header[3]];
        let size = u64::from(
            le_u32_at(&header[4..])
                .ok_or_else(|| decode_error(path, "WebP ANMF subchunk size truncated"))?,
        );
        let start = pos.saturating_add(8);
        let end = start
            .checked_add(size)
            .ok_or_else(|| decode_error(path, "WebP ANMF subchunk size overflows"))?;
        if end > payload_end {
            return Err(decode_error(path, "WebP ANMF subchunk is truncated"));
        }
        let padded = end
            .checked_add(size & 1)
            .ok_or_else(|| decode_error(path, "WebP ANMF subchunk padding overflows"))?;
        if padded > payload_end {
            return Err(decode_error(
                path,
                "WebP ANMF subchunk padding is truncated",
            ));
        }
        if (id == *b"VP8 " || id == *b"VP8L") && size > 0 {
            return Ok(true);
        }
        pos = padded;
    }
    Ok(false)
}

/// Seek-based RIFF/WEBP walk: only fixed-size chunk headers are read and ANMF
/// payloads are skipped, so RAM use is O(1) for arbitrarily large animations.
pub(crate) fn demux_webp_metadata(path: &Path) -> Result<ContainerMedia, PipelineError> {
    let mut file = File::open(path).map_err(|source| io_error(path, source))?;
    let file_len = file
        .metadata()
        .map_err(|source| io_error(path, source))?
        .len();
    let mut signature = [0_u8; 12];
    read_at(path, &mut file, 0, &mut signature)?;
    if &signature[..4] != b"RIFF" || &signature[8..12] != b"WEBP" {
        return Err(decode_error(path, "truncated RIFF/WEBP header"));
    }
    let declared = u64::from(
        le_u32_at(&signature[4..8])
            .ok_or_else(|| decode_error(path, "WebP RIFF size is missing"))?,
    );
    let riff_end = 8_u64
        .checked_add(declared)
        .ok_or_else(|| decode_error(path, "WebP RIFF size overflows"))?;
    if declared < 4 || riff_end > file_len {
        return Err(decode_error(
            path,
            "WebP RIFF size extends past the end of the file",
        ));
    }

    let scan = scan_webp_chunks(path, &mut file, riff_end)?;
    let (width, height) = match scan.extended {
        Some((true, width, height)) => {
            if !scan.animated || scan.durations.is_empty() {
                return Err(decode_error(
                    path,
                    "animated WebP is missing ANIM or ANMF chunks",
                ));
            }
            (width, height)
        }
        Some((false, width, height)) => {
            if scan.still.is_none() {
                return Err(decode_error(
                    path,
                    "extended WebP has no VP8 or VP8L image chunk",
                ));
            }
            (width, height)
        }
        None => scan
            .still
            .ok_or_else(|| decode_error(path, "WebP has no VP8, VP8L, or VP8X chunk"))?,
    };
    if width == 0 || height == 0 {
        return Err(decode_error(path, "WebP canvas has a zero width or height"));
    }
    let even_w = u16::try_from(even_dim(width).unwrap_or(0xFFFF_FFFF)).unwrap_or(u16::MAX);
    let even_h = u16::try_from(even_dim(height).unwrap_or(0xFFFF_FFFF)).unwrap_or(u16::MAX);

    let mut durations = scan.durations;
    if durations.is_empty() {
        durations.push(100_000);
    }
    let mut packets = Vec::with_capacity(durations.len());
    let mut pts_us = 0_i64;
    for duration in durations {
        packets.push(ContainerPacket {
            track: 0,
            pts: pts_us,
            dts: pts_us,
            duration,
            keyframe: true,
            data: Vec::new(),
        });
        pts_us = pts_us.saturating_add(i64::try_from(duration).unwrap_or(i64::MAX));
    }

    Ok(ContainerMedia {
        tracks: vec![ContainerTrack {
            codec: fastconv_domain::VideoCodec::Webp,
            width: Some(even_w),
            height: Some(even_h),
            channels: None,
            sample_rate: None,
            config: Vec::new(),
            timescale: US_TIMESCALE,
            presentation_edit: None,
            pcm_format: None,
        }],
        packets,
    })
}

fn parse_vp8_dimensions(payload: &[u8]) -> Option<(u32, u32)> {
    if payload.get(3..6)? != [0x9D, 0x01, 0x2A] {
        return None;
    }
    let width = u32::from(le_u16_at(payload.get(6..8)?)?) & 0x3FFF;
    let height = u32::from(le_u16_at(payload.get(8..10)?)?) & 0x3FFF;
    (width > 0 && height > 0).then_some((width, height))
}

fn parse_vp8l_dimensions(payload: &[u8]) -> Option<(u32, u32)> {
    if *payload.first()? != 0x2F {
        return None;
    }
    let packed = le_u32_at(payload.get(1..5)?)?;
    let width = (packed & 0x3FFF).saturating_add(1);
    let height = ((packed >> 14) & 0x3FFF).saturating_add(1);
    Some((width, height))
}

fn le_u16_at(bytes: &[u8]) -> Option<u16> {
    Some(u16::from_le_bytes(bytes.get(..2)?.try_into().ok()?))
}

// ---------------------------------------------------------------------------
// Animated GIF decoding
// ---------------------------------------------------------------------------

fn rgba_to_rgb(src: &[u8], bpp: usize, width: u32, height: u32) -> Result<Vec<u8>, PipelineError> {
    if src.len() != byte_len(width, height, bpp)? {
        return Err(decode_error(
            Path::new("animation-rgba-to-rgb"),
            "decoded frame buffer has the wrong length",
        ));
    }
    let mut rgb = vec![0_u8; byte_len(width, height, 3)?];
    if bpp == 3 {
        rgb.copy_from_slice(src);
        return Ok(rgb);
    }
    for (dst, px) in rgb
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<4>().0.iter())
    {
        let alpha = u32::from(px[3]);
        dst[0] = clamp_u8(i32::try_from((u32::from(px[0]) * alpha + 127) / 255).unwrap_or(255));
        dst[1] = clamp_u8(i32::try_from((u32::from(px[1]) * alpha + 127) / 255).unwrap_or(255));
        dst[2] = clamp_u8(i32::try_from((u32::from(px[2]) * alpha + 127) / 255).unwrap_or(255));
    }
    Ok(rgb)
}

/// Expands flattened RGB into RGBA with fully opaque alpha, so an animation
/// source without transparency still has a well-defined pixel layout.
fn rgb_to_rgba(src: &[u8], width: u32, height: u32) -> Result<Vec<u8>, PipelineError> {
    if src.len() != byte_len(width, height, 3)? {
        return Err(decode_error(
            Path::new("animation-rgb-to-rgba"),
            "decoded frame buffer has the wrong length",
        ));
    }
    let mut rgba = vec![0_u8; byte_len(width, height, 4)?];
    for (dst, px) in rgba
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<3>().0.iter())
    {
        dst[..3].copy_from_slice(px);
        dst[3] = 255;
    }
    Ok(rgba)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AnimPixelMode {
    FlattenedRgb,
    AlwaysRgba,
    /// Preserves the decoder's native channel count (RGB8 when opaque, RGBA8
    /// when alpha is present) without allocating a temporary expansion buffer.
    Native,
}

fn save_gif_previous_canvas(
    path: &Path,
    canvas: &[u8],
    canvas_memory_limit: u64,
    saved: &mut Option<Vec<u8>>,
) -> Result<(), PipelineError> {
    let two_canvases = u64::try_from(canvas.len())
        .unwrap_or(u64::MAX)
        .saturating_mul(2);
    if two_canvases > canvas_memory_limit {
        return Err(decode_error(
            path,
            format!(
                "GIF previous-frame canvas backup ({two_canvases} bytes) exceeds the {canvas_memory_limit}-byte memory limit"
            ),
        ));
    }
    match saved {
        Some(buf) if buf.len() == canvas.len() => buf.copy_from_slice(canvas),
        slot => *slot = Some(canvas.to_vec()),
    }
    Ok(())
}

fn composite_gif_frame(
    canvas: &mut [u8],
    canvas_dims: (usize, usize),
    frame_buf: &[u8],
    rect: (usize, usize, usize, usize),
) {
    let (canvas_w, canvas_h) = canvas_dims;
    let (left, top, fw, fh) = rect;
    let x0 = left.min(canvas_w);
    let y0 = top.min(canvas_h);
    let x1 = left.saturating_add(fw).min(canvas_w);
    let y1 = top.saturating_add(fh).min(canvas_h);
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    for y in y0..y1 {
        let sy = y - top;
        for x in x0..x1 {
            let sx = x - left;
            let src_idx = (sy * fw + sx) * 4;
            let Some(src) = frame_buf.get(src_idx..src_idx + 4) else {
                continue;
            };
            if src[3] == 0 {
                continue;
            }
            let dst_idx = (y * canvas_w + x) * 4;
            canvas[dst_idx..dst_idx + 4].copy_from_slice(src);
        }
    }
}

type GifFrameFill<'a> = dyn FnMut(&mut [u8]) -> Result<(), PipelineError> + 'a;

/// Composites every GIF frame onto an RGBA canvas and streams either packed
/// RGB8 (flattened) or RGBA8 (transparency preserved).
///
/// Decodes each sub-frame directly into a buffer supplied by `with_frame_buf`
/// via `next_frame_info` + `read_into_buffer`, so `gif::Decoder` never retains
/// an unbudgeted internal `frame.buffer` allocation.
fn gif_streaming_pixels_impl<W, F>(
    path: &Path,
    mode: AnimPixelMode,
    canvas_memory_limit: u64,
    mut with_frame_buf: W,
    mut on_frame: F,
) -> Result<(), PipelineError>
where
    W: FnMut(usize, usize, bool, &mut GifFrameFill<'_>) -> Result<(), PipelineError>,
    F: FnMut(AnimFrameInfo, &[u8]) -> Result<(), PipelineError>,
{
    let preserve_alpha = mode != AnimPixelMode::FlattenedRgb;
    let decoder_limit = ANIM_MEMORY_LIMIT.min(canvas_memory_limit.max(1));
    let mut options = gif::DecodeOptions::new();
    options.set_color_output(gif::ColorOutput::RGBA);
    options.set_memory_limit(gif::MemoryLimit::Bytes(
        NonZeroU64::new(decoder_limit).unwrap_or(NonZeroU64::MIN),
    ));
    let file = File::open(path).map_err(|source| io_error(path, source))?;
    let mut decoder = options
        .read_info(BufReader::new(file))
        .map_err(|err| decode_error(path, format!("GIF header is invalid: {err}")))?;
    let width = usize::from(decoder.width());
    let height = usize::from(decoder.height());
    let anim_w = u32::from(decoder.width());
    let anim_h = u32::from(decoder.height());
    if width == 0 || height == 0 {
        return Err(decode_error(path, "GIF canvas has a zero width or height"));
    }
    let canvas_len = byte_len(anim_w, anim_h, 4)?;
    // A hostile logical screen (e.g. 65535x65535) must fail before the RGBA
    // canvas is requested from the allocator.
    if u64::try_from(canvas_len).unwrap_or(u64::MAX) > canvas_memory_limit {
        return Err(decode_error(
            path,
            format!(
                "GIF canvas ({canvas_len} bytes) exceeds the {canvas_memory_limit}-byte memory limit"
            ),
        ));
    }
    let mut canvas = vec![0_u8; canvas_len];
    let mut pts_us = 0_i64;
    let mut saved: Option<Vec<u8>> = None;

    while let Some(frame_info) = decoder
        .next_frame_info()
        .map_err(|err| decode_error(path, format!("GIF frame is invalid: {err}")))?
    {
        let disposal = frame_info.dispose;
        let rect = (
            usize::from(frame_info.left),
            usize::from(frame_info.top),
            usize::from(frame_info.width),
            usize::from(frame_info.height),
        );
        let delay = frame_info.delay;
        let frame_buf_len = byte_len(u32::from(frame_info.width), u32::from(frame_info.height), 4)?;

        if disposal == gif::DisposalMethod::Previous {
            save_gif_previous_canvas(path, &canvas, canvas_memory_limit, &mut saved)?;
        }
        // `needs_backup` tells the caller to charge a second canvas only for
        // `Previous` disposal, since only those frames retain a backup.
        with_frame_buf(
            canvas_len,
            frame_buf_len,
            disposal == gif::DisposalMethod::Previous,
            &mut |frame_buf| {
                decoder
                    .read_into_buffer(frame_buf)
                    .map_err(|err| decode_error(path, format!("GIF frame is invalid: {err}")))?;
                composite_gif_frame(&mut canvas, (width, height), frame_buf, rect);
                Ok(())
            },
        )?;

        let duration = if delay == 0 {
            100_000_u64
        } else {
            u64::from(delay) * 10_000
        };
        if preserve_alpha {
            let info = AnimFrameInfo {
                width: anim_w,
                height: anim_h,
                pts_us,
                duration_us: duration,
                channels: 4,
            };
            on_frame(info, &canvas)?;
        } else {
            let rgb = rgba_to_rgb(&canvas, 4, anim_w, anim_h)?;
            let info = AnimFrameInfo {
                width: anim_w,
                height: anim_h,
                pts_us,
                duration_us: duration,
                channels: 3,
            };
            on_frame(info, &rgb)?;
        }
        pts_us = pts_us.saturating_add(i64::try_from(duration).unwrap_or(i64::MAX));

        apply_gif_disposal(&mut canvas, width, rect, disposal, &mut saved);
    }
    Ok(())
}

fn apply_gif_disposal(
    canvas: &mut Vec<u8>,
    width: usize,
    rect: (usize, usize, usize, usize),
    disposal: gif::DisposalMethod,
    saved: &mut Option<Vec<u8>>,
) {
    let (left, top, fw, fh) = rect;
    let height = canvas.len() / 4 / width;
    let x0 = left.min(width);
    let y0 = top.min(height);
    let x1 = left.saturating_add(fw).min(width);
    let y1 = top.saturating_add(fh).min(height);
    match disposal {
        gif::DisposalMethod::Background => {
            for y in y0..y1 {
                let start = (y * width + x0) * 4;
                let end = (y * width + x1) * 4;
                canvas[start..end].fill(0);
            }
        }
        gif::DisposalMethod::Previous => {
            if let Some(previous) = saved.take() {
                *canvas = previous;
            }
        }
        gif::DisposalMethod::Keep | gif::DisposalMethod::Any => {}
    }
}

fn default_gif_frame_buf(
    path: &Path,
) -> impl FnMut(usize, usize, bool, &mut GifFrameFill<'_>) -> Result<(), PipelineError> + '_ {
    let mut scratch = Vec::new();
    move |_canvas_len, frame_buf_len, _needs_backup, fill| {
        if u64::try_from(frame_buf_len).unwrap_or(u64::MAX) > ANIM_MEMORY_LIMIT {
            return Err(decode_error(
                path,
                format!(
                    "GIF frame buffer ({frame_buf_len} bytes) exceeds the {ANIM_MEMORY_LIMIT}-byte memory limit"
                ),
            ));
        }
        scratch.clear();
        if scratch.capacity() < frame_buf_len {
            scratch
                .try_reserve_exact(frame_buf_len)
                .map_err(|_| decode_error(path, "GIF frame buffer allocation failed"))?;
        }
        scratch.resize(frame_buf_len, 0);
        fill(scratch.as_mut_slice())
    }
}

/// Composites GIF frames and streams flattened RGB8 for non-animation outputs.
#[cfg(test)]
pub(crate) fn decode_gif_frames_streaming_rgb<F>(
    path: &Path,
    on_frame: F,
) -> Result<(), PipelineError>
where
    F: FnMut(AnimFrameInfo, &[u8]) -> Result<(), PipelineError>,
{
    gif_streaming_pixels_impl(
        path,
        AnimPixelMode::FlattenedRgb,
        ANIM_MEMORY_LIMIT,
        default_gif_frame_buf(path),
        on_frame,
    )
}

/// Composites GIF frames and streams RGBA8, preserving the one-bit GIF mask.
pub(crate) fn decode_gif_frames_streaming_pixels<F>(
    path: &Path,
    on_frame: F,
) -> Result<(), PipelineError>
where
    F: FnMut(AnimFrameInfo, &[u8]) -> Result<(), PipelineError>,
{
    gif_streaming_pixels_impl(
        path,
        AnimPixelMode::AlwaysRgba,
        ANIM_MEMORY_LIMIT,
        default_gif_frame_buf(path),
        on_frame,
    )
}

/// `ControlFlow`-returning GIF variant: `Break` stops compositing immediately.
/// `max_canvas_bytes` caps canvas allocations upfront, and `with_frame_buf`
/// supplies the temporary sub-frame decode buffer from the caller's pool so
/// every raw pixel allocation stays under one aggregate budget.
pub(crate) fn decode_gif_frames_streaming_pixels_ctrl<W, F>(
    path: &Path,
    max_canvas_bytes: usize,
    with_frame_buf: W,
    mut on_frame: F,
) -> Result<ControlFlow<()>, PipelineError>
where
    W: FnMut(usize, usize, bool, &mut GifFrameFill<'_>) -> Result<(), PipelineError>,
    F: FnMut(AnimFrameInfo, &[u8]) -> Result<ControlFlow<()>, PipelineError>,
{
    let mut stopped = false;
    let wrapped = |info: AnimFrameInfo, pixels: &[u8]| -> Result<(), PipelineError> {
        match on_frame(info, pixels)? {
            ControlFlow::Continue(()) => Ok(()),
            ControlFlow::Break(()) => {
                stopped = true;
                Err(decode_error(path, FRAMES_EARLY_STOP))
            }
        }
    };
    let limit = u64::try_from(max_canvas_bytes)
        .unwrap_or(u64::MAX)
        .min(ANIM_MEMORY_LIMIT);
    match gif_streaming_pixels_impl(
        path,
        AnimPixelMode::AlwaysRgba,
        limit,
        with_frame_buf,
        wrapped,
    ) {
        Ok(()) => Ok(ControlFlow::Continue(())),
        Err(PipelineError::DecodeFailed { reason, .. })
            if stopped && reason == FRAMES_EARLY_STOP =>
        {
            Ok(ControlFlow::Break(()))
        }
        Err(err) => Err(err),
    }
}

/// Streams GIF frames directly into a caller-supplied pitched NV12 buffer.
pub(crate) fn decode_gif_frames_streaming_nv12<F>(
    path: &Path,
    mut on_frame: F,
) -> Result<(), PipelineError>
where
    F: FnMut(
        AnimFrameInfo,
        &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
    ) -> Result<(), PipelineError>,
{
    decode_gif_frames_streaming_pixels(path, |info, pixels| {
        let padded_width =
            even_dim(info.width).ok_or_else(|| decode_error(path, "GIF width overflow"))?;
        let padded_height =
            even_dim(info.height).ok_or_else(|| decode_error(path, "GIF height overflow"))?;
        let nv12_info = AnimFrameInfo {
            width: padded_width,
            height: padded_height,
            ..info
        };
        let mut pack = |dst: &mut [u8], pitch: usize| {
            write_pixels_nv12(
                pixels,
                info.width,
                info.height,
                dst,
                pitch,
                usize::from(info.channels),
            )
        };
        on_frame(nv12_info, &mut pack)
    })
}

/// Decodes all GIF frames into display-ordered planar YUV420 frames.
pub(crate) fn decode_gif_frames_yuv420(
    path: &Path,
    max_decoded_bytes: u64,
) -> Result<Vec<Yuv420Frame>, PipelineError> {
    let mut frames = Vec::new();
    let mut used = 0_u64;
    decode_gif_frames_streaming_pixels(path, |info, pixels| {
        let channels = usize::from(info.channels);
        let frame = pixels_to_yuv420(info, pixels, channels)?;
        used = used.saturating_add(
            u64::try_from(
                frame
                    .y
                    .len()
                    .saturating_add(frame.u.len())
                    .saturating_add(frame.v.len()),
            )
            .unwrap_or(u64::MAX),
        );
        if used > max_decoded_bytes {
            return Err(decode_error(
                path,
                format!("decoded GIF exceeds the {max_decoded_bytes}-byte budget"),
            ));
        }
        frames.push(frame);
        Ok(())
    })?;
    Ok(frames)
}

// ---------------------------------------------------------------------------
// Animated WebP decoding
// ---------------------------------------------------------------------------

/// Reads a WebP from one open handle, capped at `limit + 1` bytes. Reading one
/// extra byte closes the metadata/read TOCTOU window: the decoder receives
/// exactly the bytes that were checked against the memory limit.
fn read_bounded_animation_file(path: &Path, limit: u64) -> Result<Vec<u8>, PipelineError> {
    let mut file = File::open(path).map_err(|source| io_error(path, source))?;
    let capacity = limit.saturating_add(1);
    let mut bytes = Vec::new();
    (&mut file)
        .take(capacity)
        .read_to_end(&mut bytes)
        .map_err(|source| io_error(path, source))?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > limit {
        return Err(decode_error(
            path,
            format!("WebP input exceeds the {limit}-byte memory limit"),
        ));
    }
    Ok(bytes)
}

fn emit_webp_frame<F>(
    mode: AnimPixelMode,
    bpp: usize,
    info: AnimFrameInfo,
    buf: &[u8],
    on_frame: &mut F,
) -> Result<(), PipelineError>
where
    F: FnMut(AnimFrameInfo, &[u8]) -> Result<(), PipelineError>,
{
    match mode {
        AnimPixelMode::Native => on_frame(info, buf),
        AnimPixelMode::AlwaysRgba => {
            if bpp == 4 {
                on_frame(
                    AnimFrameInfo {
                        channels: 4,
                        ..info
                    },
                    buf,
                )
            } else {
                let rgba = rgb_to_rgba(buf, info.width, info.height)?;
                on_frame(
                    AnimFrameInfo {
                        channels: 4,
                        ..info
                    },
                    &rgba,
                )
            }
        }
        AnimPixelMode::FlattenedRgb => {
            let rgb = rgba_to_rgb(buf, bpp, info.width, info.height)?;
            on_frame(info, &rgb)
        }
    }
}

/// Decodes a still or animated WebP file and streams packed RGB8 or RGBA8.
fn webp_streaming_pixels_impl<F>(
    path: &Path,
    mode: AnimPixelMode,
    frame_memory_limit: u64,
    mut on_frame: F,
) -> Result<(), PipelineError>
where
    F: FnMut(AnimFrameInfo, &[u8]) -> Result<(), PipelineError>,
{
    let bytes = read_bounded_animation_file(path, ANIM_MEMORY_LIMIT)?;
    let mut decoder = zenwebp::WebPDecoder::new(&bytes)
        .map_err(|err| decode_error(path, format!("WebP header is invalid: {err}")))?;
    decoder.set_memory_limit(usize::try_from(ANIM_MEMORY_LIMIT).unwrap_or(usize::MAX));
    let (width, height) = decoder.dimensions();
    if width == 0 || height == 0 {
        return Err(decode_error(path, "WebP canvas has a zero width or height"));
    }
    let bpp = if decoder.has_alpha() {
        4_usize
    } else {
        3_usize
    };
    let buf_len = decoder
        .output_buffer_size()
        .ok_or_else(|| decode_error(path, "WebP frame buffer size overflows"))?;
    if buf_len != byte_len(width, height, bpp)? {
        return Err(decode_error(
            path,
            "WebP decoder reported an inconsistent buffer size",
        ));
    }
    if u64::try_from(buf_len).unwrap_or(u64::MAX) > frame_memory_limit {
        return Err(decode_error(
            path,
            format!(
                "WebP frame buffer ({buf_len} bytes) exceeds the {frame_memory_limit}-byte memory limit"
            ),
        ));
    }
    let mut buf = vec![0_u8; buf_len];
    let mut pts_us = 0_i64;
    let out_channels: u8 = match mode {
        AnimPixelMode::FlattenedRgb => 3,
        AnimPixelMode::AlwaysRgba => 4,
        AnimPixelMode::Native => u8::try_from(bpp).unwrap_or(4),
    };

    if decoder.is_animated() {
        loop {
            let ms = match decoder.read_frame(&mut buf) {
                Ok(ms) => ms,
                Err(err) if matches!(err.error(), zenwebp::DecodeError::NoMoreFrames) => break,
                Err(err) => {
                    return Err(decode_error(
                        path,
                        format!("WebP frame decode failed: {err}"),
                    ));
                }
            };
            let duration = if ms == 0 {
                100_000_u64
            } else {
                u64::from(ms) * 1_000
            };
            let info = AnimFrameInfo {
                width,
                height,
                pts_us,
                duration_us: duration,
                channels: out_channels,
            };
            emit_webp_frame(mode, bpp, info, &buf, &mut on_frame)?;
            pts_us = pts_us.saturating_add(i64::try_from(duration).unwrap_or(i64::MAX));
        }
    } else {
        decoder
            .read_image(&mut buf)
            .map_err(|err| decode_error(path, format!("WebP decode failed: {err}")))?;
        let info = AnimFrameInfo {
            width,
            height,
            pts_us,
            duration_us: 100_000,
            channels: out_channels,
        };
        emit_webp_frame(mode, bpp, info, &buf, &mut on_frame)?;
    }
    Ok(())
}

/// Streams flattened RGB8 frames from a still or animated WebP file.
#[cfg(test)]
pub(crate) fn decode_webp_frames_streaming_rgb<F>(
    path: &Path,
    on_frame: F,
) -> Result<(), PipelineError>
where
    F: FnMut(AnimFrameInfo, &[u8]) -> Result<(), PipelineError>,
{
    webp_streaming_pixels_impl(
        path,
        AnimPixelMode::FlattenedRgb,
        ANIM_MEMORY_LIMIT,
        on_frame,
    )
}

/// Streams WebP frames as RGBA8 so transparency survives animation re-encoding.
pub(crate) fn decode_webp_frames_streaming_pixels<F>(
    path: &Path,
    on_frame: F,
) -> Result<(), PipelineError>
where
    F: FnMut(AnimFrameInfo, &[u8]) -> Result<(), PipelineError>,
{
    webp_streaming_pixels_impl(path, AnimPixelMode::AlwaysRgba, ANIM_MEMORY_LIMIT, on_frame)
}

/// `ControlFlow`-returning WebP variant: `Break` stops frame decoding immediately.
/// Streams the decoder's native pixel format (RGB8 or RGBA8) with no extra
/// per-frame expansion copy and enforces `max_frame_bytes` before allocation.
pub(crate) fn decode_webp_frames_streaming_pixels_ctrl<F>(
    path: &Path,
    max_frame_bytes: usize,
    mut on_frame: F,
) -> Result<ControlFlow<()>, PipelineError>
where
    F: FnMut(AnimFrameInfo, &[u8]) -> Result<ControlFlow<()>, PipelineError>,
{
    let mut stopped = false;
    let wrapped = |info: AnimFrameInfo, pixels: &[u8]| -> Result<(), PipelineError> {
        match on_frame(info, pixels)? {
            ControlFlow::Continue(()) => Ok(()),
            ControlFlow::Break(()) => {
                stopped = true;
                Err(decode_error(path, FRAMES_EARLY_STOP))
            }
        }
    };
    let limit = u64::try_from(max_frame_bytes)
        .unwrap_or(u64::MAX)
        .min(ANIM_MEMORY_LIMIT);
    match webp_streaming_pixels_impl(path, AnimPixelMode::Native, limit, wrapped) {
        Ok(()) => Ok(ControlFlow::Continue(())),
        Err(PipelineError::DecodeFailed { reason, .. })
            if stopped && reason == FRAMES_EARLY_STOP =>
        {
            Ok(ControlFlow::Break(()))
        }
        Err(err) => Err(err),
    }
}

/// Streams WebP frames directly into a caller-supplied pitched NV12 buffer.
pub(crate) fn decode_webp_frames_streaming_nv12<F>(
    path: &Path,
    mut on_frame: F,
) -> Result<(), PipelineError>
where
    F: FnMut(
        AnimFrameInfo,
        &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
    ) -> Result<(), PipelineError>,
{
    decode_webp_frames_streaming_pixels(path, |info, pixels| {
        let padded_width =
            even_dim(info.width).ok_or_else(|| decode_error(path, "WebP width overflow"))?;
        let padded_height =
            even_dim(info.height).ok_or_else(|| decode_error(path, "WebP height overflow"))?;
        let nv12_info = AnimFrameInfo {
            width: padded_width,
            height: padded_height,
            ..info
        };
        let mut pack = |dst: &mut [u8], pitch: usize| {
            write_pixels_nv12(
                pixels,
                info.width,
                info.height,
                dst,
                pitch,
                usize::from(info.channels),
            )
        };
        on_frame(nv12_info, &mut pack)
    })
}

/// Decodes all WebP frames into display-ordered planar YUV420 frames.
pub(crate) fn decode_webp_frames_yuv420(
    path: &Path,
    max_decoded_bytes: u64,
) -> Result<Vec<Yuv420Frame>, PipelineError> {
    let mut frames = Vec::new();
    let mut used = 0_u64;
    decode_webp_frames_streaming_pixels(path, |info, pixels| {
        let channels = usize::from(info.channels);
        let frame = pixels_to_yuv420(info, pixels, channels)?;
        used = used.saturating_add(
            u64::try_from(
                frame
                    .y
                    .len()
                    .saturating_add(frame.u.len())
                    .saturating_add(frame.v.len()),
            )
            .unwrap_or(u64::MAX),
        );
        if used > max_decoded_bytes {
            return Err(decode_error(
                path,
                format!("decoded WebP exceeds the {max_decoded_bytes}-byte budget"),
            ));
        }
        frames.push(frame);
        Ok(())
    })?;
    Ok(frames)
}

// ---------------------------------------------------------------------------
// Streaming parallel animation encoding
// ---------------------------------------------------------------------------

/// Maps `VideoQuality` onto the GIF quantiser speed (1 = best, 30 = fastest).
fn gif_speed(quality: VideoQuality) -> i32 {
    let q = u32::from(quality.get());
    if q >= 80 {
        1 + i32::try_from(((100 - q) * 9 + 10) / 20).unwrap_or(0)
    } else {
        10 + i32::try_from(((80 - q) * 20 + 40) / 80).unwrap_or(20)
    }
}

/// Extracts the nested image chunks (including even-padding bytes) for an
/// RFC 9649 `ANMF` frame and reports whether the parsed chunks carry alpha
/// (via an `ALPH` chunk or the `VP8L` bitstream alpha bit).
fn extract_inner_vp8_chunk(bytes: &[u8]) -> Result<(Vec<u8>, bool), PipelineError> {
    let path = Path::new("webp-encoder");
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
        return Err(encode_error(
            path,
            "encoder produced a malformed WebP image",
        ));
    }
    let mut pos = 12_usize;
    let mut inner = Vec::new();
    let mut has_alpha = false;
    while pos.saturating_add(8) <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = usize::try_from(le_u32_at(&bytes[pos + 4..pos + 8]).unwrap_or(0))
            .map_err(|_| encode_error(path, "WebP chunk size overflows usize"))?;
        let payload_start = pos.saturating_add(8);
        let payload_end = payload_start.saturating_add(size);
        if payload_end > bytes.len() {
            return Err(encode_error(path, "WebP chunk is truncated"));
        }
        let padded = size.saturating_add(size & 1);
        let end = payload_start.saturating_add(padded);
        if end > bytes.len() {
            return Err(encode_error(path, "WebP chunk padding is truncated"));
        }
        if id == b"ALPH" {
            has_alpha = true;
            inner.extend_from_slice(&bytes[pos..end]);
        } else if id == b"VP8 " || id == b"VP8L" {
            if id == b"VP8L"
                && size >= 5
                && bytes[payload_start] == 0x2F
                && le_u32_at(&bytes[payload_start + 1..payload_start + 5])
                    .is_some_and(|packed| (packed >> 28) & 1 != 0)
            {
                has_alpha = true;
            }
            inner.extend_from_slice(&bytes[pos..end]);
            return Ok((inner, has_alpha));
        }
        pos = payload_end.saturating_add(size & 1);
    }
    Err(encode_error(
        path,
        "encoder produced a WebP without a VP8 or VP8L chunk",
    ))
}

struct AnimJob {
    index: usize,
    info: AnimFrameInfo,
    rgb: Vec<u8>,
    delay: u32,
}

enum PreparedFrame {
    Gif(gif::Frame<'static>),
    Webp {
        inner: Vec<u8>,
        delay_ms: u32,
        has_alpha: bool,
    },
    Failed(String),
}

fn prepare_frame(
    format: VideoContainerFormat,
    quality: VideoQuality,
    job: AnimJob,
) -> Result<PreparedFrame, PipelineError> {
    let w = u16::try_from(job.info.width).unwrap_or(u16::MAX);
    let h = u16::try_from(job.info.height).unwrap_or(u16::MAX);
    if format == VideoContainerFormat::Gif {
        let mut frame = if job.info.channels == 4 {
            // GIF stores a one-bit transparency mask; `from_rgba_speed`
            // selects a transparent palette index for zero-alpha pixels.
            let mut rgba = job.rgb;
            gif::Frame::from_rgba_speed(w, h, &mut rgba, gif_speed(quality))
        } else {
            gif::Frame::from_rgb_speed(w, h, &job.rgb, gif_speed(quality))
        };
        frame.delay = u16::try_from(job.delay).unwrap_or(u16::MAX);
        // Each RGBA frame is an independent full-canvas snapshot; clearing to
        // background prevents prior opaque pixels from showing through later
        // transparent pixels.
        frame.dispose = if job.info.channels == 4 {
            gif::DisposalMethod::Background
        } else {
            gif::DisposalMethod::Keep
        };
        Ok(PreparedFrame::Gif(frame))
    } else {
        let path = Path::new("webp-encoder");
        let layout = if job.info.channels == 4 {
            zenwebp::PixelLayout::Rgba8
        } else {
            zenwebp::PixelLayout::Rgb8
        };
        let config = zenwebp::LossyConfig::new()
            .with_quality(f32::from(quality.get()))
            .with_method(4);
        let webp_bytes = zenwebp::EncodeRequest::lossy(
            &config,
            &job.rgb,
            layout,
            job.info.width,
            job.info.height,
        )
        .encode()
        .map_err(|err| encode_error(path, format!("WebP encoding failed: {err}")))?;
        let (inner, has_alpha) = extract_inner_vp8_chunk(&webp_bytes)?;
        Ok(PreparedFrame::Webp {
            inner,
            delay_ms: job.delay,
            has_alpha,
        })
    }
}

fn write_webp_header<W: Write>(
    out: &mut W,
    width: u16,
    height: u16,
    loop_count: AnimationLoop,
    has_alpha: bool,
) -> Result<(), PipelineError> {
    let path = Path::new("webp-encoder");
    out.write_all(b"RIFF")
        .and_then(|()| out.write_all(&[0_u8; 4]))
        .and_then(|()| out.write_all(b"WEBP"))
        .map_err(|source| io_error(path, source))?;
    // VP8X stores canvas dimensions minus one as 24-bit little-endian values.
    let w = u32::from(width).saturating_sub(1).to_le_bytes();
    let h = u32::from(height).saturating_sub(1).to_le_bytes();
    // VP8X flags: animation (0x02) and, when present, the alpha hint (0x10)
    // that tells decoders to plan for an RGBA output buffer.
    let flags = 0x02 | if has_alpha { 0x10 } else { 0 };
    let vp8x = [
        b'V', b'P', b'8', b'X', 10, 0, 0, 0, flags, 0, 0, 0, w[0], w[1], w[2], h[0], h[1], h[2],
    ];
    out.write_all(&vp8x)
        .map_err(|source| io_error(path, source))?;
    // ANIM payload: four-byte background colour then the little-endian loop
    // count. Zero is RFC 9649's "loop forever".
    let iterations = match loop_count {
        AnimationLoop::Infinite => 0_u16,
        AnimationLoop::Times(times) => times.get(),
    };
    let [lo, hi] = iterations.to_le_bytes();
    out.write_all(&[b'A', b'N', b'I', b'M', 6, 0, 0, 0, 0, 0, 0, 0, lo, hi])
        .map_err(|source| io_error(path, source))?;
    Ok(())
}

fn spawn_animation_workers<'scope>(
    scope: &'scope thread::Scope<'scope, '_>,
    workers: usize,
    format: VideoContainerFormat,
    quality: VideoQuality,
    job_rx: Arc<Mutex<mpsc::Receiver<AnimJob>>>,
    ready_tx: mpsc::SyncSender<(usize, AnimFrameInfo, PreparedFrame)>,
) -> Result<Vec<thread::ScopedJoinHandle<'scope, Result<(), PipelineError>>>, PipelineError> {
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let job_rx = Arc::clone(&job_rx);
        let ready_tx = ready_tx.clone();
        handles.push(
            thread::Builder::new()
                .name("fastconv-anim-worker".to_owned())
                .spawn_scoped(scope, move || -> Result<(), PipelineError> {
                    loop {
                        let received = {
                            let guard = job_rx.lock().map_err(|_| PipelineError::WorkerPanicked)?;
                            guard.recv()
                        };
                        let Ok(job) = received else {
                            return Ok(());
                        };
                        let index = job.index;
                        let info = job.info;
                        let prepared =
                            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                prepare_frame(format, quality, job)
                            })) {
                                Ok(Ok(frame)) => frame,
                                Ok(Err(err)) => PreparedFrame::Failed(err.to_string()),
                                Err(_) => PreparedFrame::Failed(
                                    "animation frame preparation panicked".to_owned(),
                                ),
                            };
                        if ready_tx.send((index, info, prepared)).is_err() {
                            // The writer is gone; intentionally keep draining jobs so a producer
                            // blocked on a full job channel can reach its next admission token
                            // and observe the failure instead of hanging.
                        }
                    }
                })
                .map_err(PipelineError::WorkerStartFailed)?,
        );
    }
    drop(job_rx);
    drop(ready_tx);
    Ok(handles)
}

/// Drives a bounded, ordered animation encode from a frame producer.
///
/// The producer pushes raw RGB frames; worker threads quantise/encode them in
/// parallel and a dedicated writer emits them in strict display order.
pub(crate) fn encode_animation_stream<P>(
    format: VideoContainerFormat,
    quality: VideoQuality,
    worker_threads: usize,
    loop_count: AnimationLoop,
    file: &File,
    produce: P,
) -> Result<u64, PipelineError>
where
    P: FnOnce(
        &mut dyn FnMut(AnimFrameInfo, &[u8]) -> Result<(), PipelineError>,
    ) -> Result<(), PipelineError>,
{
    encode_animation_stream_bounded(
        format,
        quality,
        worker_threads,
        loop_count,
        file,
        u64::MAX,
        1,
        produce,
    )
}

/// Bounded variant used by frame stacks. `max_inflight_bytes` caps the total
/// raw pixels held by the admission queue, prepared jobs and pending writes;
/// `bytes_per_frame` scales that budget to the plan's actual frame size so an
/// 8K stack cannot multiply a small worker count into gigabytes of RAM.
#[allow(clippy::too_many_arguments)] // Flat budget knobs keep the default wrapper zero-cost.
pub(crate) fn encode_animation_stream_bounded<P>(
    format: VideoContainerFormat,
    quality: VideoQuality,
    worker_threads: usize,
    loop_count: AnimationLoop,
    file: &File,
    max_inflight_bytes: u64,
    bytes_per_frame: u64,
    produce: P,
) -> Result<u64, PipelineError>
where
    P: FnOnce(
        &mut dyn FnMut(AnimFrameInfo, &[u8]) -> Result<(), PipelineError>,
    ) -> Result<(), PipelineError>,
{
    let workers = worker_threads.clamp(1, MAX_ANIM_WORKERS);
    // Two admission slots per worker (one queued, one being prepared) plus one
    // reserved for the ordered writer. Every slot costs one raw frame.
    let mut capacity = workers.saturating_mul(2).max(1);
    if max_inflight_bytes != u64::MAX {
        let frame_bytes = bytes_per_frame.max(1);
        let slot_budget = max_inflight_bytes / frame_bytes.saturating_mul(2);
        let slot_budget = usize::try_from(slot_budget).unwrap_or(usize::MAX).max(1);
        capacity = capacity.min(slot_budget);
    }
    let workers = workers.min(capacity).max(1);
    thread::scope(|scope| -> Result<u64, PipelineError> {
        let (job_tx, job_rx) = mpsc::sync_channel::<AnimJob>(capacity);
        let (ready_tx, ready_rx) =
            mpsc::sync_channel::<(usize, AnimFrameInfo, PreparedFrame)>(capacity);
        // One admission token per outstanding frame. Tokens are consumed only
        // after an ordered write, so out-of-order arrivals cannot accumulate
        // an unbounded pending map. Dropping the writer closes this channel
        // and immediately wakes a blocked producer.
        let (slot_tx, slot_rx) = mpsc::sync_channel::<()>(capacity);
        let job_rx = Arc::new(Mutex::new(job_rx));

        let workers_handles =
            spawn_animation_workers(scope, workers, format, quality, job_rx, ready_tx)?;

        let writer = thread::Builder::new()
            .name("fastconv-anim-writer".to_owned())
            .spawn_scoped(scope, move || -> Result<u64, PipelineError> {
                write_animation_frames(format, loop_count, file, ready_rx, slot_rx)
            })
            .map_err(PipelineError::WorkerStartFailed)?;

        let mut cumulative_us = 0_u64;
        let mut emitted_delay = 0_u64;
        let mut expected_dims: Option<(u32, u32, u8)> = None;
        let mut index = 0_usize;
        let produced = {
            let mut push = |info: AnimFrameInfo, rgb: &[u8]| -> Result<(), PipelineError> {
                push_animation_frame(
                    format,
                    info,
                    rgb,
                    &mut cumulative_us,
                    &mut emitted_delay,
                    &mut expected_dims,
                    index,
                    &job_tx,
                    &slot_tx,
                )?;
                index = index.saturating_add(1);
                Ok(())
            };
            produce(&mut push)
        };
        drop(job_tx);

        let mut worker_error = None;
        for handle in workers_handles {
            match handle.join() {
                Ok(Err(err)) => {
                    if worker_error.is_none() {
                        worker_error = Some(err);
                    }
                }
                Ok(Ok(())) => {}
                Err(_) => {
                    if worker_error.is_none() {
                        worker_error = Some(PipelineError::WorkerPanicked);
                    }
                }
            }
        }
        let written = match writer.join() {
            Ok(result) => result,
            Err(_) => Err(PipelineError::WorkerPanicked),
        };
        if let Some(err) = worker_error {
            return Err(err);
        }
        match produced {
            Ok(()) => written,
            Err(PipelineError::WorkerPanicked) => match written {
                Err(err) => Err(err),
                Ok(_) => Err(PipelineError::WorkerPanicked),
            },
            Err(err) => Err(err),
        }
    })
}

// `push_animation_frame` keeps the mutable timing state in one place so the
// producer closure itself stays tiny.
#[allow(clippy::too_many_arguments)]
fn push_animation_frame(
    format: VideoContainerFormat,
    info: AnimFrameInfo,
    rgb: &[u8],
    cumulative_us: &mut u64,
    emitted_delay: &mut u64,
    expected_dims: &mut Option<(u32, u32, u8)>,
    index: usize,
    job_tx: &mpsc::SyncSender<AnimJob>,
    slot_tx: &mpsc::SyncSender<()>,
) -> Result<(), PipelineError> {
    let path = Path::new("animation-encoder");
    let dims = (info.width, info.height, info.channels);
    match expected_dims {
        Some(expected) if *expected != dims => {
            return Err(encode_error(
                path,
                format!(
                    "animation frames must share dimensions and pixel layout; expected {}x{} ({} ch), got {}x{} ({} ch)",
                    expected.0, expected.1, expected.2, dims.0, dims.1, dims.2
                ),
            ));
        }
        Some(_) => {}
        None => {
            let (max_w, max_h) = match format {
                VideoContainerFormat::Gif => (u32::from(u16::MAX), u32::from(u16::MAX)),
                VideoContainerFormat::Webp => (16_383, 16_383),
                _ => return Err(encode_error(path, "unsupported animation output")),
            };
            if info.width == 0 || info.height == 0 || info.width > max_w || info.height > max_h {
                return Err(encode_error(
                    path,
                    format!("invalid animation dimensions {}x{}", dims.0, dims.1),
                ));
            }
            *expected_dims = Some((info.width, info.height, info.channels));
        }
    }
    let channels = match info.channels {
        3 | 4 => usize::from(info.channels),
        other => {
            return Err(encode_error(
                path,
                format!("unsupported animation pixel layout: {other} channels"),
            ));
        }
    };
    if rgb.len() != byte_len(info.width, info.height, channels)? {
        return Err(encode_error(
            path,
            "animation frame has the wrong pixel buffer length",
        ));
    }
    // Wait for a free admission slot before queueing; a failed writer closes
    // the channel so this send fails instead of deadlocking.
    slot_tx
        .send(())
        .map_err(|_| PipelineError::WorkerPanicked)?;
    *cumulative_us = cumulative_us.saturating_add(info.duration_us);
    let target = if format == VideoContainerFormat::Gif {
        cumulative_us.saturating_add(5_000) / 10_000
    } else {
        cumulative_us.saturating_add(500) / 1_000
    };
    let max_delay = if format == VideoContainerFormat::Gif {
        u64::from(u16::MAX)
    } else {
        0x00FF_FFFF_u64
    };
    let raw_delay = target.saturating_sub(*emitted_delay);
    let delay = raw_delay.clamp(1, max_delay);
    // Even when the requested delay exceeds the container maximum, the next
    // frame must start after the delay actually written, not the ideal target.
    *emitted_delay = emitted_delay.saturating_add(delay);
    job_tx
        .send(AnimJob {
            index,
            info,
            rgb: rgb.to_vec(),
            delay: u32::try_from(delay).unwrap_or(u32::MAX),
        })
        .map_err(|_| PipelineError::WorkerPanicked)
}

/// Emits one RFC 9649 `ANMF` chunk with a nested VP8/VP8L frame and returns its
/// total on-disk size (including the eight-byte chunk header).
fn write_webp_frame(
    out: &mut BufWriter<&File>,
    info: AnimFrameInfo,
    inner: &[u8],
    delay_ms: u32,
) -> Result<u64, PipelineError> {
    let path = Path::new("webp-encoder");
    let size = 16_u64
        .checked_add(u64::try_from(inner.len()).unwrap_or(u64::MAX))
        .ok_or_else(|| encode_error(path, "ANMF size overflow"))?;
    let width_minus_1 = info.width.saturating_sub(1).to_le_bytes();
    let height_minus_1 = info.height.saturating_sub(1).to_le_bytes();
    let delay = delay_ms.to_le_bytes();
    let mut header = [0_u8; 24];
    header[0..4].copy_from_slice(b"ANMF");
    header[4..8].copy_from_slice(&u32::try_from(size).unwrap_or(u32::MAX).to_le_bytes());
    // x=0, y=0 as two 24-bit little-endian fields (bytes 8..14 remain zero).
    header[14..17].copy_from_slice(&width_minus_1[..3]);
    header[17..20].copy_from_slice(&height_minus_1[..3]);
    header[20..23].copy_from_slice(&delay[..3]);
    header[23] = 0x02;
    out.write_all(&header)
        .map_err(|source| io_error(path, source))?;
    out.write_all(inner)
        .map_err(|source| io_error(path, source))?;
    size.checked_add(8)
        .ok_or_else(|| encode_error(path, "ANMF chunk size overflow"))
}

struct AnimationWriters<'a> {
    gif: Option<gif::Encoder<BufWriter<&'a File>>>,
    webp: Option<BufWriter<&'a File>>,
    riff_len: u64,
    webp_has_alpha: bool,
}

impl AnimationWriters<'_> {
    fn new() -> Self {
        Self {
            gif: None,
            webp: None,
            riff_len: 0,
            webp_has_alpha: false,
        }
    }

    fn is_empty(&self) -> bool {
        self.gif.is_none() && self.webp.is_none()
    }
}

fn init_animation_writer<'a>(
    format: VideoContainerFormat,
    loop_count: AnimationLoop,
    file: &'a File,
    info: AnimFrameInfo,
    prepared: &PreparedFrame,
) -> Result<AnimationWriters<'a>, PipelineError> {
    let path = Path::new("animation-encoder");
    let width = u16::try_from(info.width).unwrap_or(u16::MAX);
    let height = u16::try_from(info.height).unwrap_or(u16::MAX);
    match (format, prepared) {
        (VideoContainerFormat::Gif, PreparedFrame::Gif(_)) => {
            let mut encoder = gif::Encoder::new(BufWriter::new(file), width, height, &[])
                .map_err(|err| encode_error(path, format!("GIF encoder failed: {err}")))?;
            // `Times` counts total plays; the GIF repeat extension counts
            // replays after the first one (`Finite(0)` means play once).
            let repeat = match loop_count {
                AnimationLoop::Infinite => gif::Repeat::Infinite,
                AnimationLoop::Times(times) => gif::Repeat::Finite(times.get().saturating_sub(1)),
            };
            encoder
                .set_repeat(repeat)
                .map_err(|err| encode_error(path, format!("GIF encoder failed: {err}")))?;
            Ok(AnimationWriters {
                gif: Some(encoder),
                webp: None,
                riff_len: 0,
                webp_has_alpha: false,
            })
        }
        (VideoContainerFormat::Webp, PreparedFrame::Webp { has_alpha, .. }) => {
            let mut out = BufWriter::new(file);
            write_webp_header(&mut out, width, height, loop_count, *has_alpha)?;
            // 12-byte RIFF/WEBP prefix + 18-byte VP8X + 14-byte ANIM.
            Ok(AnimationWriters {
                gif: None,
                webp: Some(out),
                riff_len: 44_u64,
                webp_has_alpha: *has_alpha,
            })
        }
        _ => Err(encode_error(path, "animation encoder/format mismatch")),
    }
}

fn finish_animation_writer(
    writers: AnimationWriters<'_>,
    mut file: &File,
) -> Result<(), PipelineError> {
    let path = Path::new("animation-encoder");
    if let Some(encoder) = writers.gif {
        let mut buffered = encoder
            .into_inner()
            .map_err(|err| encode_error(path, format!("GIF encoder failed: {err}")))?;
        buffered.flush().map_err(|source| io_error(path, source))?;
    }
    if let Some(mut out) = writers.webp {
        out.flush().map_err(|source| io_error(path, source))?;
        if writers.riff_len > u64::from(u32::MAX).saturating_add(8) {
            return Err(encode_error(
                path,
                "animated WebP exceeds the 4 GiB RIFF limit",
            ));
        }
        let size = u32::try_from(writers.riff_len.saturating_sub(8))
            .unwrap_or(u32::MAX)
            .to_le_bytes();
        file.seek(SeekFrom::Start(4))
            .map_err(|source| io_error(path, source))?;
        file.write_all(&size)
            .map_err(|source| io_error(path, source))?;
        // RFC 9649: VP8X flags live at byte 20 (12-byte RIFF/WEBP + 8-byte VP8X header).
        // Patch the alpha bit from all written frames so later alpha frames are signalled.
        let vp8x_flags = [0x02 | if writers.webp_has_alpha { 0x10 } else { 0 }];
        file.seek(SeekFrom::Start(20))
            .map_err(|source| io_error(path, source))?;
        file.write_all(&vp8x_flags)
            .map_err(|source| io_error(path, source))?;
        file.flush().map_err(|source| io_error(path, source))?;
    }
    Ok(())
}

fn write_animation_frames(
    format: VideoContainerFormat,
    loop_count: AnimationLoop,
    file: &File,
    ready_rx: mpsc::Receiver<(usize, AnimFrameInfo, PreparedFrame)>,
    slot_rx: mpsc::Receiver<()>,
) -> Result<u64, PipelineError> {
    let path = Path::new("animation-encoder");
    let mut pending: HashMap<usize, (AnimFrameInfo, PreparedFrame)> = HashMap::new();
    let mut next = 0_usize;
    let mut written = 0_u64;
    let mut writers = AnimationWriters::new();

    for (index, info, frame) in ready_rx {
        pending.insert(index, (info, frame));
        while let Some((frame_info, prepared)) = pending.remove(&next) {
            if let PreparedFrame::Failed(reason) = &prepared {
                return Err(encode_error(path, reason.clone()));
            }
            if writers.is_empty() {
                writers = init_animation_writer(format, loop_count, file, frame_info, &prepared)?;
            }
            let mut webp_bytes = 0_u64;
            match (&mut writers.gif, &mut writers.webp, prepared) {
                (Some(encoder), _, PreparedFrame::Gif(frame)) => {
                    encoder
                        .write_frame(&frame)
                        .map_err(|err| encode_error(path, format!("GIF encode failed: {err}")))?;
                }
                (
                    _,
                    Some(out),
                    PreparedFrame::Webp {
                        inner,
                        delay_ms,
                        has_alpha,
                    },
                ) => {
                    writers.webp_has_alpha |= has_alpha;
                    webp_bytes = write_webp_frame(out, frame_info, &inner, delay_ms)?;
                }
                (_, _, PreparedFrame::Failed(reason)) => {
                    return Err(encode_error(path, reason));
                }
                _ => return Err(encode_error(path, "animation encoder/format mismatch")),
            }
            writers.riff_len = writers.riff_len.saturating_add(webp_bytes);
            next = next.saturating_add(1);
            written = written.saturating_add(1);
            // Release the slot only after the frame has been written in order;
            // receiving alone would let the pending map grow without bound.
            slot_rx.recv().map_err(|_| PipelineError::WorkerPanicked)?;
        }
    }
    finish_animation_writer(writers, file)?;
    if !pending.is_empty() {
        return Err(encode_error(
            path,
            "animation frames were dropped out of order",
        ));
    }
    if written == 0 {
        return Err(encode_error(path, "animation source produced no frames"));
    }
    drop(slot_rx);
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn info(w: u32, h: u32) -> AnimFrameInfo {
        AnimFrameInfo {
            width: w,
            height: h,
            pts_us: 0,
            duration_us: 100_000,
            channels: 3,
        }
    }

    #[test]
    fn bt601_forward_and_inverse_roundtrip() {
        let colors: [(u8, u8, u8); 5] = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 255),
            (16, 32, 48),
        ];
        for (red, green, blue) in colors {
            let (luma, chroma_u, chroma_v) =
                rgb_to_yuv_block(i32::from(red), i32::from(green), i32::from(blue));
            let luma_delta = i32::from(luma) - 16;
            let u_delta = i32::from(chroma_u) - 128;
            let v_delta = i32::from(chroma_v) - 128;
            let back = (
                clamp_u8((298 * luma_delta + 409 * v_delta + 128) >> 8),
                clamp_u8((298 * luma_delta - 100 * u_delta - 208 * v_delta + 128) >> 8),
                clamp_u8((298 * luma_delta + 516 * u_delta + 128) >> 8),
            );
            assert!(
                i32::from(back.0).abs_diff(i32::from(red)) <= 3,
                "red mismatch {red} -> {back:?}"
            );
            assert!(
                i32::from(back.1).abs_diff(i32::from(green)) <= 3,
                "green mismatch {green} -> {back:?}"
            );
            assert!(
                i32::from(back.2).abs_diff(i32::from(blue)) <= 3,
                "blue mismatch {blue} -> {back:?}"
            );
        }
        assert_eq!(rgb_to_yuv_block(255, 0, 0), (82, 90, 240));
    }

    #[test]
    fn rgb_to_yuv420_pads_odd_dimensions() -> Result<(), PipelineError> {
        let mut rgb = vec![0_u8; 3 * 3 * 3];
        rgb[0..3].copy_from_slice(&[255, 0, 0]);
        rgb[3..6].copy_from_slice(&[0, 255, 0]);
        rgb[6..9].copy_from_slice(&[0, 0, 255]);
        rgb[18..21].copy_from_slice(&[255, 255, 255]);
        let frame = rgb_to_yuv420(info(3, 3), &rgb)?;
        assert_eq!((frame.width, frame.height), (4, 4));
        assert_eq!(frame.y.len(), 16);
        assert_eq!(frame.u.len(), 4);
        assert_eq!(frame.v.len(), 4);
        assert_eq!(frame.y[3], frame.y[2], "rightmost column replicated");
        assert_eq!(frame.y[12], frame.y[8], "bottom row replicated");
        assert_eq!(frame.y[15], frame.y[14], "bottom-right corner replicated");
        Ok(())
    }

    #[test]
    fn simd_rgb_to_yuv420_even_dimensions() -> Result<(), PipelineError> {
        let colors: [[u8; 3]; 8] = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255],
            [16, 32, 48],
            [180, 96, 40],
            [10, 200, 90],
            [90, 10, 200],
        ];
        let rgb: Vec<u8> = colors.iter().flatten().copied().collect();
        let frame = rgb_to_yuv420(info(4, 2), &rgb)?;
        assert_eq!((frame.width, frame.height), (4, 2));
        assert_eq!((frame.y.len(), frame.u.len(), frame.v.len()), (8, 2, 2));
        for (idx, color) in colors.iter().enumerate() {
            let want = rgb_to_yuv_block(
                i32::from(color[0]),
                i32::from(color[1]),
                i32::from(color[2]),
            )
            .0;
            assert!(
                frame.y[idx].abs_diff(want) <= 2,
                "SIMD luma for {color:?}: {} vs scalar {want}",
                frame.y[idx]
            );
        }
        Ok(())
    }

    #[test]
    fn simd_opaque_rgba_matches_rgb_without_flattening() -> Result<(), PipelineError> {
        let colors: [[u8; 3]; 8] = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255],
            [16, 32, 48],
            [180, 96, 40],
            [10, 200, 90],
            [90, 10, 200],
        ];
        let rgb: Vec<u8> = colors.iter().flatten().copied().collect();
        let rgba: Vec<u8> = colors
            .iter()
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect();
        let from_rgb = rgb_to_yuv420(info(4, 2), &rgb)?;
        let from_rgba = pixels_to_yuv420(info(4, 2), &rgba, 4)?;
        assert_eq!(from_rgb.y, from_rgba.y, "opaque alpha must be ignored");
        assert_eq!(from_rgb.u, from_rgba.u);
        assert_eq!(from_rgb.v, from_rgba.v);
        Ok(())
    }

    #[test]
    fn transparent_rgba_composites_onto_black() -> Result<(), PipelineError> {
        let rgba = vec![
            255, 0, 0, 0, // fully transparent -> limited-range black
            255, 0, 0, 255, // fully opaque red
            0, 255, 0, 128, // half-transparent green
            0, 0, 255, 255, // fully opaque blue
        ];
        let frame = pixels_to_yuv420(info(2, 2), &rgba, 4)?;
        assert_eq!(frame.y[0], 16, "transparent pixel must become black luma");
        assert!(
            frame.y[1].abs_diff(rgb_to_yuv_block(255, 0, 0).0) <= 2,
            "opaque red luma drift: {}",
            frame.y[1]
        );
        assert!(frame.y[2] > 16, "half-transparent green must be visible");
        assert!(
            frame.y[3].abs_diff(rgb_to_yuv_block(0, 0, 255).0) <= 2,
            "opaque blue luma drift: {}",
            frame.y[3]
        );
        Ok(())
    }

    #[test]
    fn odd_dimension_rgba_yuv420_matches_scalar_flatten() -> Result<(), PipelineError> {
        let mut rgba = Vec::new();
        for index in 0..15_u32 {
            let alpha = if index == 5 { 0 } else { 255 };
            rgba.extend_from_slice(&[
                u8::try_from(index * 13).unwrap_or(255),
                u8::try_from(255 - index * 7).unwrap_or(0),
                u8::try_from(index * 11).unwrap_or(255),
                alpha,
            ]);
        }
        let flat = rgba_to_rgb(&rgba, 4, 5, 3)?;
        let expected = rgb_to_yuv420(info(5, 3), &flat)?;
        let actual = pixels_to_yuv420(info(5, 3), &rgba, 4)?;
        assert_eq!(actual, expected, "odd transparent RGBA must be bit-exact");
        assert_eq!((actual.width, actual.height), (6, 4));
        assert_eq!(actual.y[5], actual.y[4], "rightmost column replicated");
        assert_eq!(actual.y[18], actual.y[12], "bottom row replicated");
        Ok(())
    }

    #[test]
    fn simd_nv12_opaque_rgba_matches_rgb() -> Result<(), PipelineError> {
        let colors: [[u8; 3]; 8] = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255],
            [16, 32, 48],
            [180, 96, 40],
            [10, 200, 90],
            [90, 10, 200],
        ];
        let rgb: Vec<u8> = colors.iter().flatten().copied().collect();
        let rgba: Vec<u8> = colors
            .iter()
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect();
        let mut from_rgb = vec![0xAA_u8; 4 * 6];
        let mut from_rgba = vec![0xAA_u8; 4 * 6];
        write_nv12(&rgb, 4, 2, &mut from_rgb, 4)?;
        write_pixels_nv12(&rgba, 4, 2, &mut from_rgba, 4, 4)?;
        assert_eq!(from_rgb, from_rgba, "NV12 RGBA fast path must match RGB");
        for (row, color_row) in colors.as_chunks::<4>().0.iter().enumerate() {
            for (col, color) in color_row.iter().enumerate() {
                let got = from_rgba[row * 4 + col];
                let want = rgb_to_yuv_block(
                    i32::from(color[0]),
                    i32::from(color[1]),
                    i32::from(color[2]),
                )
                .0;
                assert!(
                    got.abs_diff(want) <= 2,
                    "NV12 luma at ({row},{col}): {got} vs scalar {want}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn odd_dimension_nv12_rgba_matches_scalar_flatten() -> Result<(), PipelineError> {
        let rgba: Vec<u8> = (0..9_u32)
            .flat_map(|i| {
                [
                    u8::try_from(i * 20).unwrap_or(255),
                    u8::try_from(255 - i * 10).unwrap_or(0),
                    64_u8,
                    255,
                ]
            })
            .collect();
        let flat = rgba_to_rgb(&rgba, 4, 3, 3)?;
        let mut from_rgb = vec![0xAA_u8; 4 * 6];
        let mut from_rgba = vec![0xAA_u8; 4 * 6];
        write_nv12(&flat, 3, 3, &mut from_rgb, 4)?;
        write_pixels_nv12(&rgba, 3, 3, &mut from_rgba, 4, 4)?;
        assert_eq!(
            from_rgb, from_rgba,
            "odd NV12 RGBA must match scalar flatten"
        );
        Ok(())
    }

    #[test]
    fn nv12_write_and_inverse_preserves_pitch() -> Result<(), PipelineError> {
        // A uniform colour keeps 4:2:0 chroma averaging exact, so only the
        // integer BT.601 rounding is under test here.
        let rgb: Vec<u8> = [180_u8, 96, 40].repeat(4);
        let mut buf = vec![0xAA_u8; 2 * 6 + 4];
        write_nv12(&rgb, 2, 2, &mut buf, 4)?;
        assert_eq!(buf[2], 0xAA, "row padding untouched");
        let decoded = nv12_to_rgb8(&buf, 4, 2, 2, YuvColor::default())?;
        assert_eq!(decoded.len(), 12);
        for (got, want) in decoded.iter().zip(&rgb) {
            assert!(got.abs_diff(*want) <= 4, "NV12 roundtrip drift");
        }
        Ok(())
    }

    #[test]
    fn gif_subrect_transparency_and_disposal() -> Result<(), PipelineError> {
        let dir = std::env::temp_dir().join(format!("fastconv-anim-gif-{}", std::process::id()));
        fs::create_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        let path = dir.join("sub.gif");
        {
            let mut file = File::create(&path).map_err(|source| io_error(&path, source))?;
            let mut encoder = gif::Encoder::new(&mut file, 5, 5, &[])
                .map_err(|err| encode_error(&path, format!("{err}")))?;
            let mut put =
                |rgba: [u8; 4], left: u16, top: u16, delay: u16, dispose: gif::DisposalMethod| {
                    let mut pixels = rgba.to_vec();
                    let mut frame = gif::Frame::from_rgba_speed(1, 1, &mut pixels, 1);
                    frame.left = left;
                    frame.top = top;
                    frame.delay = delay;
                    frame.dispose = dispose;
                    encoder
                        .write_frame(&frame)
                        .map_err(|err| encode_error(&path, format!("{err}")))
                };
            // Opaque red sub-rect, transparent frame with background disposal,
            // then an opaque green sub-rect that must survive a `Previous` frame.
            put([255, 0, 0, 255], 1, 1, 0, gif::DisposalMethod::Keep)?;
            put([0, 0, 0, 0], 1, 1, 2, gif::DisposalMethod::Background)?;
            put([0, 255, 0, 255], 3, 3, 3, gif::DisposalMethod::Keep)?;
            put([0, 0, 0, 0], 3, 3, 1, gif::DisposalMethod::Previous)?;
            put([0, 0, 0, 0], 0, 0, 1, gif::DisposalMethod::Keep)?;
        }
        let mut frames = Vec::new();
        decode_gif_frames_streaming_rgb(&path, |frame, rgb| {
            frames.push((frame, rgb.to_vec()));
            Ok(())
        })?;
        fs::remove_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        assert_eq!(frames.len(), 5);
        let durations: Vec<u64> = frames.iter().map(|f| f.0.duration_us).collect();
        assert_eq!(durations, [100_000, 20_000, 30_000, 10_000, 10_000]);
        assert_eq!(frames[0].1[18], 255, "opaque red at (1,1)");
        assert_eq!(frames[0].1[0], 0, "canvas starts black");
        assert_eq!(frames[1].1[18], 255, "transparent pixel does not overwrite");
        assert_eq!(frames[2].1[18], 0, "background disposal clears (1,1)");
        assert_eq!(frames[2].1[55], 255, "opaque green at (3,3)");
        assert_eq!(frames[4].1[55], 255, "previous disposal restored (3,3)");
        Ok(())
    }

    #[test]
    fn animation_webp_rfc9649_roundtrip() -> Result<(), PipelineError> {
        let dir = std::env::temp_dir().join(format!("fastconv-anim-webp-{}", std::process::id()));
        fs::create_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        let path = dir.join("anim.webp");
        let quality = VideoQuality::new(80).map_err(|_| encode_error(&path, "quality"))?;
        let file = File::create(&path).map_err(|source| io_error(&path, source))?;
        let written = encode_animation_stream(
            VideoContainerFormat::Webp,
            quality,
            2,
            AnimationLoop::Infinite,
            &file,
            |sink| {
                for i in 0..3_u32 {
                    let rgb = vec![u8::try_from(i * 40).unwrap_or(0); 3_usize * 5 * 5];
                    sink(info(5, 5), &rgb)?;
                }
                Ok(())
            },
        )?;
        assert_eq!(written, 3);
        drop(file);

        let bytes = fs::read(&path).map_err(|source| io_error(&path, source))?;
        let mut decoder = zenwebp::WebPDecoder::new(&bytes)
            .map_err(|err| decode_error(&path, format!("{err}")))?;
        assert!(decoder.is_animated());
        assert_eq!(decoder.loop_count(), zenwebp::decoder::LoopCount::Forever);
        assert_eq!(decoder.dimensions(), (5, 5));
        let mut count = 0_u32;
        let mut buf = vec![0_u8; decoder.output_buffer_size().unwrap_or(0)];
        loop {
            match decoder.read_frame(&mut buf) {
                Ok(ms) => {
                    assert_eq!(ms, 100, "100 ms per frame");
                    count = count.saturating_add(1);
                }
                Err(err) if matches!(err.error(), zenwebp::DecodeError::NoMoreFrames) => break,
                Err(err) => return Err(decode_error(&path, format!("{err}"))),
            }
        }
        assert_eq!(count, 3);
        fs::remove_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        Ok(())
    }

    #[test]
    fn webp_metadata_handles_large_anmf_duration() -> Result<(), PipelineError> {
        let dir =
            std::env::temp_dir().join(format!("fastconv-anim-duration-{}", std::process::id()));
        fs::create_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        let path = dir.join("long.webp");
        let ms = 5_000_000_u32.to_le_bytes();
        let w = 4_u32.to_le_bytes();
        let h = 4_u32.to_le_bytes();
        let mut frame = vec![0_u8; 12];
        frame.extend_from_slice(&ms[..3]);
        frame.push(0x02);
        let mut bitstream = vec![0x2F];
        bitstream.extend_from_slice(&3_u32.to_le_bytes());
        frame.extend_from_slice(b"VP8L");
        frame.extend_from_slice(&5_u32.to_le_bytes());
        frame.extend_from_slice(&bitstream);
        frame.push(0); // RIFF pad for the odd-sized nested chunk
        let mut body = b"WEBP".to_vec();
        body.extend_from_slice(b"VP8X");
        body.extend_from_slice(&10_u32.to_le_bytes());
        body.extend_from_slice(&[0x02, 0, 0, 0, w[0], w[1], w[2], h[0], h[1], h[2]]);
        body.extend_from_slice(b"ANIM");
        body.extend_from_slice(&6_u32.to_le_bytes());
        body.extend_from_slice(&[0; 6]);
        body.extend_from_slice(b"ANMF");
        body.extend_from_slice(&u32::try_from(frame.len()).unwrap_or(0).to_le_bytes());
        body.extend_from_slice(&frame);
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&u32::try_from(body.len()).unwrap_or(0).to_le_bytes());
        bytes.extend_from_slice(&body);
        fs::write(&path, &bytes).map_err(|source| io_error(&path, source))?;

        let media = demux_webp_metadata(&path)?;
        assert_eq!(media.packets.len(), 1);
        assert_eq!(media.packets[0].duration, 5_000_000_000);
        fs::remove_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        Ok(())
    }

    #[test]
    fn webp_encoder_handles_odd_chunk_sizes() -> Result<(), PipelineError> {
        let dir = std::env::temp_dir().join(format!("fastconv-anim-odd-{}", std::process::id()));
        fs::create_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        let path = dir.join("odd.webp");
        let quality = VideoQuality::new(80).map_err(|_| encode_error(&path, "quality"))?;
        let file = File::create(&path).map_err(|source| io_error(&path, source))?;
        encode_animation_stream(
            VideoContainerFormat::Webp,
            quality,
            1,
            AnimationLoop::Infinite,
            &file,
            |sink| {
                let rgb = vec![128_u8; 3_usize * 7 * 3];
                sink(info(7, 3), &rgb)?;
                Ok(())
            },
        )?;
        drop(file);
        let mut decoded = Vec::new();
        decode_webp_frames_streaming_rgb(&path, |frame, rgb| {
            decoded.push((frame, rgb.to_vec()));
            Ok(())
        })?;
        assert_eq!(decoded.len(), 1);
        assert_eq!((decoded[0].0.width, decoded[0].0.height), (7, 3));
        assert_eq!(decoded[0].1.len(), 7 * 3 * 3);
        fs::remove_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        Ok(())
    }

    #[test]
    fn gif_speed_maps_quality_range() {
        let q = |v| VideoQuality::new(v).unwrap_or(VideoQuality::DEFAULT);
        assert_eq!(gif_speed(q(100)), 1);
        assert_eq!(gif_speed(q(80)), 10);
        assert_eq!(gif_speed(q(0)), 30);
        assert!((1..=30).contains(&gif_speed(q(50))));
    }

    #[test]
    fn nv12_uses_signalled_source_color_matrix() -> Result<(), PipelineError> {
        // A mid-tone colour encoded with BT.709 limited-range coefficients.
        let nv12 = [117_u8, 117, 117, 117, 96, 178, 96, 178];
        let bt709 = nv12_to_rgb8(
            &nv12,
            2,
            2,
            2,
            YuvColor::specified(YuvMatrix::Bt709, YuvRange::Limited),
        )?;
        let bt601 = nv12_to_rgb8(&nv12, 2, 2, 2, YuvColor::default())?;
        assert!(bt709[0].abs_diff(208) <= 3, "BT.709 red: {}", bt709[0]);
        assert!(
            i32::from(bt709[0]) - i32::from(bt601[0]) >= 8,
            "source matrix must change the decoded colour: {bt601:?} vs {bt709:?}"
        );
        let full = nv12_to_rgb8(
            &nv12,
            2,
            2,
            2,
            YuvColor::specified(YuvMatrix::Bt709, YuvRange::Full),
        )?;
        assert_ne!(full, bt709, "range must be honoured as well as matrix");
        Ok(())
    }

    #[test]
    fn nv12_rejects_unsupported_rgb_matrix_at_the_conversion_boundary() {
        let nv12 = [117_u8, 117, 117, 117, 96, 178, 96, 178];
        for code in [4_u8, 7, 10, 14] {
            let result = nv12_to_rgb8(
                &nv12,
                2,
                2,
                2,
                YuvColor::specified(YuvMatrix::Unsupported(code), YuvRange::Limited),
            );
            assert!(
                matches!(result, Err(PipelineError::DecodeFailed { .. })),
                "matrix coefficient {code} must be rejected only where RGB is produced"
            );
        }
    }

    fn temp_anim_dir(label: &str) -> Result<std::path::PathBuf, PipelineError> {
        let dir =
            std::env::temp_dir().join(format!("fastconv-anim-{label}-{}", std::process::id()));
        fs::create_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        Ok(dir)
    }

    #[test]
    fn animation_alpha_survives_gif_and_webp_roundtrip() -> Result<(), PipelineError> {
        let dir = temp_anim_dir("alpha")?;
        let quality = VideoQuality::new(90).map_err(|_| encode_error(&dir, "quality"))?;
        let mut rgba = vec![255_u8; 4 * 2 * 2];
        rgba[0..4].copy_from_slice(&[0, 0, 0, 0]); // transparent top-left
        rgba[4..8].copy_from_slice(&[220, 24, 24, 255]);
        rgba[8..12].copy_from_slice(&[24, 220, 24, 255]);
        rgba[12..16].copy_from_slice(&[24, 24, 220, 255]);
        let mut info = info(2, 2);
        info.channels = 4;

        for format in [VideoContainerFormat::Gif, VideoContainerFormat::Webp] {
            let path = dir.join(format!("alpha.{}", format.extension()));
            let file = File::create(&path).map_err(|source| io_error(&path, source))?;
            encode_animation_stream(format, quality, 1, AnimationLoop::Infinite, &file, |sink| {
                sink(info, &rgba)?;
                Ok(())
            })?;
            drop(file);
            let mut decoded = Vec::new();
            let decode = |frame: AnimFrameInfo, pixels: &[u8]| {
                decoded.push((frame.channels, pixels.to_vec()));
                Ok(())
            };
            if format == VideoContainerFormat::Gif {
                decode_gif_frames_streaming_pixels(&path, decode)?;
            } else {
                decode_webp_frames_streaming_pixels(&path, decode)?;
            }
            assert_eq!(decoded.len(), 1);
            let (channels, pixels) = &decoded[0];
            assert_eq!(*channels, 4, "alpha must not be flattened");
            assert_eq!(pixels.len(), 4 * 2 * 2);
            assert_eq!(pixels[3], 0, "{format} transparent pixel");
            assert_eq!(pixels[15], 255, "{format} opaque pixel");
        }
        fs::remove_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        Ok(())
    }

    #[test]
    fn gif_canvas_is_bounded_before_allocation() -> Result<(), PipelineError> {
        let dir = temp_anim_dir("huge-gif")?;
        let path = dir.join("huge.gif");
        let mut bytes = b"GIF89a".to_vec();
        bytes.extend_from_slice(&u16::MAX.to_le_bytes());
        bytes.extend_from_slice(&u16::MAX.to_le_bytes());
        bytes.extend_from_slice(&[0x80, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF]);
        bytes.extend_from_slice(&[0x2C, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 1, 0, 0]);
        bytes.push(0x3B);
        fs::write(&path, &bytes).map_err(|source| io_error(&path, source))?;
        assert!(matches!(
            decode_gif_frames_streaming_rgb(&path, |_, _| Ok(())),
            Err(PipelineError::DecodeFailed { .. })
        ));
        fs::remove_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        Ok(())
    }

    #[test]
    fn webp_read_is_bounded_by_one_handle() -> Result<(), PipelineError> {
        let dir = temp_anim_dir("bounded-read")?;
        let path = dir.join("small.webp");
        let quality = VideoQuality::new(80).map_err(|_| encode_error(&path, "quality"))?;
        let file = File::create(&path).map_err(|source| io_error(&path, source))?;
        encode_animation_stream(
            VideoContainerFormat::Webp,
            quality,
            1,
            AnimationLoop::Infinite,
            &file,
            |sink| {
                sink(info(2, 2), &[128_u8; 12])?;
                Ok(())
            },
        )?;
        drop(file);
        assert!(read_bounded_animation_file(&path, 4).is_err());
        assert_ne!(read_bounded_animation_file(&path, 1_000_000)?, Vec::new());
        fs::remove_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        Ok(())
    }

    #[test]
    fn finite_loop_count_reaches_both_animation_headers() -> Result<(), PipelineError> {
        let dir = temp_anim_dir("loop")?;
        let quality = VideoQuality::new(80).map_err(|_| encode_error(&dir, "quality"))?;
        let times = std::num::NonZeroU16::new(3).unwrap_or(std::num::NonZeroU16::MIN);
        let webp = dir.join("loop.webp");
        let file = File::create(&webp).map_err(|source| io_error(&webp, source))?;
        encode_animation_stream(
            VideoContainerFormat::Webp,
            quality,
            1,
            AnimationLoop::Times(times),
            &file,
            |sink| {
                sink(info(2, 2), &[90_u8; 12])?;
                Ok(())
            },
        )?;
        drop(file);
        let bytes = fs::read(&webp).map_err(|source| io_error(&webp, source))?;
        let decoder = zenwebp::WebPDecoder::new(&bytes)
            .map_err(|err| decode_error(&webp, format!("{err}")))?;
        assert_eq!(
            decoder.loop_count(),
            zenwebp::decoder::LoopCount::Times(times)
        );

        let gif = dir.join("loop.gif");
        let file = File::create(&gif).map_err(|source| io_error(&gif, source))?;
        encode_animation_stream(
            VideoContainerFormat::Gif,
            quality,
            1,
            AnimationLoop::Times(times),
            &file,
            |sink| {
                sink(info(2, 2), &[90_u8; 12])?;
                Ok(())
            },
        )?;
        drop(file);
        let bytes = fs::read(&gif).map_err(|source| io_error(&gif, source))?;
        assert!(
            bytes.windows(11).any(|w| w == b"NETSCAPE2.0"),
            "GIF must carry a Netscape loop extension"
        );
        assert!(
            bytes.windows(2).any(|w| w == [2, 0]),
            "GIF repeat count must be total plays minus one"
        );
        fs::remove_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        Ok(())
    }

    #[test]
    fn capped_gif_delay_advances_only_by_written_delay() -> Result<(), PipelineError> {
        let (job_tx, job_rx) = mpsc::sync_channel::<AnimJob>(4);
        let (slot_tx, _slot_rx) = mpsc::sync_channel::<()>(4);
        let mut cumulative = 0_u64;
        let mut emitted = 0_u64;
        let mut dims = None;
        push_animation_frame(
            VideoContainerFormat::Gif,
            AnimFrameInfo { ..info(2, 2) },
            &[128_u8; 12],
            &mut cumulative,
            &mut emitted,
            &mut dims,
            0,
            &job_tx,
            &slot_tx,
        )?;
        let mut long = info(2, 2);
        long.duration_us = 700_000_000; // 700 s exceeds GIF's 65535 cs limit
        push_animation_frame(
            VideoContainerFormat::Gif,
            long,
            &[128_u8; 12],
            &mut cumulative,
            &mut emitted,
            &mut dims,
            1,
            &job_tx,
            &slot_tx,
        )?;
        let first = job_rx.recv().map_err(|_| PipelineError::WorkerPanicked)?;
        assert_eq!(first.delay, 10);
        let second = job_rx.recv().map_err(|_| PipelineError::WorkerPanicked)?;
        assert_eq!(second.delay, u32::from(u16::MAX));
        push_animation_frame(
            VideoContainerFormat::Gif,
            info(2, 2),
            &[128_u8; 12],
            &mut cumulative,
            &mut emitted,
            &mut dims,
            2,
            &job_tx,
            &slot_tx,
        )?;
        let third = job_rx.recv().map_err(|_| PipelineError::WorkerPanicked)?;
        assert!(
            third.delay > 1_000,
            "the next frame must wait for the real 65535 cs hold, got {}",
            third.delay
        );
        Ok(())
    }

    #[test]
    fn admission_slots_release_after_ordered_writes() -> Result<(), PipelineError> {
        let dir = temp_anim_dir("slot-release")?;
        let path = dir.join("slots.gif");
        let file = File::create(&path).map_err(|source| io_error(&path, source))?;
        let (ready_tx, ready_rx) = mpsc::sync_channel(8);
        let (slot_tx, slot_rx) = mpsc::sync_channel::<()>(2);
        // Two tokens for four frames: receiving frames must not free slots.
        slot_tx
            .send(())
            .map_err(|_| PipelineError::WorkerPanicked)?;
        slot_tx
            .send(())
            .map_err(|_| PipelineError::WorkerPanicked)?;
        let (done_tx, done_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            let result = write_animation_frames(
                VideoContainerFormat::Gif,
                AnimationLoop::Infinite,
                &file,
                ready_rx,
                slot_rx,
            );
            done_tx.send(result).ok();
        });
        let frame = || PreparedFrame::Gif(gif::Frame::from_rgb_speed(2, 2, &[0_u8; 12], 1));
        for index in [2_usize, 3, 0, 1] {
            ready_tx
                .send((index, info(2, 2), frame()))
                .map_err(|_| PipelineError::WorkerPanicked)?;
        }
        drop(ready_tx);
        assert!(
            done_rx
                .recv_timeout(std::time::Duration::from_millis(100))
                .is_err(),
            "writer must block once the admission window is exhausted"
        );
        slot_tx
            .send(())
            .map_err(|_| PipelineError::WorkerPanicked)?;
        slot_tx
            .send(())
            .map_err(|_| PipelineError::WorkerPanicked)?;
        let written = done_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|_| encode_error(&path, "writer stayed blocked after slot release"))?
            .map_err(|err| encode_error(&path, format!("writer failed: {err}")))?;
        assert_eq!(written, 4);
        handle.join().map_err(|_| PipelineError::WorkerPanicked)?;
        drop(slot_tx);
        fs::remove_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        Ok(())
    }

    #[test]
    fn writer_failure_closes_admission_window() -> Result<(), PipelineError> {
        let dir = temp_anim_dir("writer-failure")?;
        let path = dir.join("readonly.webp");
        fs::write(&path, b"seed").map_err(|source| io_error(&path, source))?;
        let quality = VideoQuality::new(80).map_err(|_| encode_error(&path, "quality"))?;
        // A read-only handle cannot receive frames; the writer exits and must
        // wake the producer instead of leaving it blocked on a full window.
        let file = File::open(&path).map_err(|source| io_error(&path, source))?;
        let (result_tx, result_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            let result = encode_animation_stream(
                VideoContainerFormat::Webp,
                quality,
                1,
                AnimationLoop::Infinite,
                &file,
                |sink| {
                    for index in 0..64_u32 {
                        let mut frame = info(2, 2);
                        frame.pts_us = i64::from(index) * 100_000;
                        sink(frame, &[128_u8; 12])?;
                    }
                    Ok(())
                },
            );
            result_tx.send(result.is_err()).ok();
        });
        let failed = result_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .map_err(|_| encode_error(&path, "producer was not woken by writer failure"))?;
        handle.join().map_err(|_| PipelineError::WorkerPanicked)?;
        assert!(failed, "a read-only output must fail the encode");
        fs::remove_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        Ok(())
    }

    #[test]
    fn gif_rgba_clears_canvas_between_independent_frames() -> Result<(), PipelineError> {
        let dir = temp_anim_dir("gif-dispose")?;
        let path = dir.join("opaque-to-transparent.gif");
        let quality = VideoQuality::new(90).map_err(|_| encode_error(&path, "quality"))?;
        let mut frame0 = vec![0_u8; 4 * 2 * 2];
        for px in frame0.as_chunks_mut::<4>().0 {
            px.copy_from_slice(&[220, 20, 20, 255]);
        }
        let mut frame1 = vec![0_u8; 4 * 2 * 2];
        frame1[0..4].copy_from_slice(&[0, 0, 0, 0]);
        frame1[4..8].copy_from_slice(&[20, 220, 20, 255]);
        frame1[8..12].copy_from_slice(&[20, 20, 220, 255]);
        frame1[12..16].copy_from_slice(&[220, 220, 20, 255]);
        let mut frame_info = info(2, 2);
        frame_info.channels = 4;

        let file = File::create(&path).map_err(|source| io_error(&path, source))?;
        encode_animation_stream(
            VideoContainerFormat::Gif,
            quality,
            1,
            AnimationLoop::Infinite,
            &file,
            |sink| {
                sink(frame_info, &frame0)?;
                let mut second = frame_info;
                second.pts_us = 100_000;
                sink(second, &frame1)?;
                Ok(())
            },
        )?;
        drop(file);

        let mut decoded = Vec::new();
        decode_gif_frames_streaming_pixels(&path, |_, pixels| {
            decoded.push(pixels.to_vec());
            Ok(())
        })?;
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0][3], 255, "frame 0 top-left must be opaque");
        assert_eq!(
            decoded[1][3], 0,
            "frame 1 top-left must be cleared to transparent, not retain frame 0 red"
        );
        fs::remove_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        Ok(())
    }

    #[test]
    fn webp_vp8x_alpha_tracks_all_frames_by_chunk_boundaries() -> Result<(), PipelineError> {
        let dir = temp_anim_dir("webp-late-alpha")?;
        let quality = VideoQuality::new(90).map_err(|_| encode_error(&dir, "quality"))?;

        // Frame 0 is fully opaque; frame 1 introduces transparency. VP8X alpha must be set.
        let mut frame0 = vec![0_u8; 4 * 4 * 4];
        for px in frame0.as_chunks_mut::<4>().0 {
            px.copy_from_slice(&[200, 80, 40, 255]);
        }
        let mut frame1 = frame0.clone();
        frame1[0..4].copy_from_slice(&[0, 0, 0, 0]);
        let mut frame_info = info(4, 4);
        frame_info.channels = 4;

        let late_alpha = dir.join("late-alpha.webp");
        let file = File::create(&late_alpha).map_err(|source| io_error(&late_alpha, source))?;
        encode_animation_stream(
            VideoContainerFormat::Webp,
            quality,
            1,
            AnimationLoop::Infinite,
            &file,
            |sink| {
                sink(frame_info, &frame0)?;
                let mut second = frame_info;
                second.pts_us = 100_000;
                sink(second, &frame1)?;
                Ok(())
            },
        )?;
        drop(file);
        let bytes = fs::read(&late_alpha).map_err(|source| io_error(&late_alpha, source))?;
        assert_ne!(
            bytes[20] & 0x10,
            0,
            "VP8X alpha bit must be set when a later frame has alpha"
        );
        let mut decoded = Vec::new();
        decode_webp_frames_streaming_pixels(&late_alpha, |_, pixels| {
            decoded.push(pixels.to_vec());
            Ok(())
        })?;
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[1][3], 0, "frame 1 transparent pixel must survive");

        // Opaque RGB animation must leave the VP8X alpha bit clear.
        let opaque = dir.join("opaque.webp");
        let file = File::create(&opaque).map_err(|source| io_error(&opaque, source))?;
        encode_animation_stream(
            VideoContainerFormat::Webp,
            quality,
            1,
            AnimationLoop::Infinite,
            &file,
            |sink| {
                sink(info(4, 4), &[120_u8; 3 * 4 * 4])?;
                let mut second = info(4, 4);
                second.pts_us = 100_000;
                sink(second, &[140_u8; 3 * 4 * 4])?;
                Ok(())
            },
        )?;
        drop(file);
        let bytes = fs::read(&opaque).map_err(|source| io_error(&opaque, source))?;
        assert_eq!(
            bytes[20] & 0x10,
            0,
            "opaque WebP must leave VP8X alpha bit clear"
        );

        // Literal "ALPH" bytes inside a VP8 payload must not trigger alpha detection.
        let mut fake_vp8 = b"RIFF".to_vec();
        let vp8_payload = b"xxALPHxx";
        let riff_size = u32::try_from(4 + 8 + vp8_payload.len()).unwrap_or(0);
        fake_vp8.extend_from_slice(&riff_size.to_le_bytes());
        fake_vp8.extend_from_slice(b"WEBP");
        fake_vp8.extend_from_slice(b"VP8 ");
        fake_vp8.extend_from_slice(&u32::try_from(vp8_payload.len()).unwrap_or(0).to_le_bytes());
        fake_vp8.extend_from_slice(vp8_payload);
        let (_, has_alpha) = extract_inner_vp8_chunk(&fake_vp8)?;
        assert!(
            !has_alpha,
            "ALPH bytes inside VP8 payload must not be treated as an ALPH chunk"
        );

        // VP8L bit 28 signals lossless alpha without a separate ALPH chunk.
        let mut vp8l_alpha = b"RIFF".to_vec();
        vp8l_alpha.extend_from_slice(&(4_u32 + 8 + 6).to_le_bytes());
        vp8l_alpha.extend_from_slice(b"WEBP");
        vp8l_alpha.extend_from_slice(b"VP8L");
        vp8l_alpha.extend_from_slice(&5_u32.to_le_bytes());
        vp8l_alpha.push(0x2F);
        vp8l_alpha.extend_from_slice(&(1_u32 << 28).to_le_bytes());
        vp8l_alpha.push(0); // odd-chunk pad
        let (_, has_alpha) = extract_inner_vp8_chunk(&vp8l_alpha)?;
        assert!(has_alpha, "VP8L alpha_is_used bit must be detected");

        fs::remove_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        Ok(())
    }

    #[test]
    fn anmf_demux_rejects_unpadded_odd_subchunk() -> Result<(), PipelineError> {
        let dir = temp_anim_dir("anmf-unpadded")?;
        let path = dir.join("unpadded.webp");
        let mut frame = vec![0_u8; 16];
        frame[6..9].copy_from_slice(&[1, 0, 0]);
        frame[9..12].copy_from_slice(&[1, 0, 0]);
        frame[12..15].copy_from_slice(&[20, 0, 0]);
        // Odd 5-byte VP8L subchunk with NO pad byte inside ANMF payload.
        frame.extend_from_slice(b"VP8L");
        frame.extend_from_slice(&5_u32.to_le_bytes());
        frame.extend_from_slice(&[0x2F, 0, 0, 0, 0]);
        let mut body = b"WEBP".to_vec();
        body.extend_from_slice(b"VP8X");
        body.extend_from_slice(&10_u32.to_le_bytes());
        body.extend_from_slice(&[0x02, 0, 0, 0, 1, 0, 0, 1, 0, 0]);
        body.extend_from_slice(b"ANIM");
        body.extend_from_slice(&6_u32.to_le_bytes());
        body.extend_from_slice(&[0; 6]);
        body.extend_from_slice(b"ANMF");
        body.extend_from_slice(&u32::try_from(frame.len()).unwrap_or(0).to_le_bytes());
        body.extend_from_slice(&frame);
        body.push(0); // outer ANMF pad byte (ANMF payload length is odd: 16 + 8 + 5 = 29)
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&u32::try_from(body.len()).unwrap_or(0).to_le_bytes());
        bytes.extend_from_slice(&body);
        fs::write(&path, &bytes).map_err(|source| io_error(&path, source))?;
        match demux_webp_metadata(&path) {
            Err(PipelineError::DecodeFailed { reason, .. }) => {
                assert!(reason.contains("padding"), "unexpected reason: {reason}");
            }
            other => {
                return Err(decode_error(
                    &path,
                    format!("expected padding error, got {other:?}"),
                ));
            }
        }
        fs::remove_dir_all(&dir).map_err(|source| io_error(&dir, source))?;
        Ok(())
    }
}
