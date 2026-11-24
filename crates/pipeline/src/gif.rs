use std::io::Cursor;
use std::num::NonZeroU64;
use std::path::Path;

use crate::PipelineError;
use crate::decode::{DecodedFrameMeta, DecodedPixelFormat};

const MAX_GIF_DECODE_BYTES: u64 = 512 * 1024 * 1024;

fn decode_error(source: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::DecodeFailed {
        path: source.to_path_buf(),
        reason: reason.into(),
    }
}

fn store_rgba_canvas(
    source: &Path,
    pixel_buf: &mut Vec<u8>,
    canvas: &[u8],
    width: usize,
    height: usize,
) -> Result<(DecodedPixelFormat, usize), PipelineError> {
    let expected = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| decode_error(source, "GIF canvas size overflow"))?;
    if canvas.len() != expected {
        return Err(decode_error(
            source,
            "GIF frame raster has unexpected length",
        ));
    }
    let all_opaque = canvas
        .as_chunks::<4>()
        .0
        .iter()
        .all(|pixel| pixel[3] == 255);
    let (format, bpp) = if all_opaque {
        (DecodedPixelFormat::Rgb8, 3_usize)
    } else {
        (DecodedPixelFormat::Rgba8, 4_usize)
    };
    let byte_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(bpp))
        .ok_or_else(|| decode_error(source, "GIF output buffer size overflow"))?;
    if pixel_buf.len() < byte_len {
        pixel_buf.resize(byte_len, 0);
    }
    let dst = &mut pixel_buf[..byte_len];
    if all_opaque {
        for (dst_pixel, src_pixel) in dst
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(canvas.as_chunks::<4>().0.iter())
        {
            dst_pixel.copy_from_slice(&src_pixel[..3]);
        }
    } else {
        dst.copy_from_slice(canvas);
    }
    Ok((format, byte_len))
}

#[derive(Clone, Copy)]
struct FrameRect {
    left: u32,
    top: u32,
    width: u32,
    height: u32,
}

fn compose_sub_rect(
    source: &Path,
    canvas: &mut [u8],
    rgba: &[u8],
    canvas_w: u32,
    canvas_h: u32,
    frame: FrameRect,
) -> Result<(), PipelineError> {
    let FrameRect {
        left,
        top,
        width: frame_w,
        height: frame_h,
    } = frame;
    let canvas_w_usize = usize::try_from(canvas_w)
        .map_err(|_| decode_error(source, "GIF canvas width exceeds addressable memory"))?;
    let frame_w_usize = usize::try_from(frame_w)
        .map_err(|_| decode_error(source, "GIF frame width exceeds addressable memory"))?;
    let left_usize = usize::try_from(left)
        .map_err(|_| decode_error(source, "GIF frame offset exceeds addressable memory"))?;
    let copy_w = frame_w.min(canvas_w.saturating_sub(left));
    if copy_w == 0 {
        return Ok(());
    }
    let byte_count = usize::try_from(copy_w)
        .ok()
        .and_then(|w| w.checked_mul(4))
        .ok_or_else(|| decode_error(source, "GIF blit size overflow"))?;
    for y in 0..frame_h {
        let canvas_y = top.saturating_add(y);
        if canvas_y >= canvas_h {
            break;
        }
        let src_start = usize::try_from(y)
            .ok()
            .and_then(|row| row.checked_mul(frame_w_usize))
            .and_then(|row| row.checked_mul(4))
            .ok_or_else(|| decode_error(source, "GIF frame offset overflow"))?;
        let dst_start = usize::try_from(canvas_y)
            .ok()
            .and_then(|row| row.checked_mul(canvas_w_usize))
            .and_then(|row| row.checked_add(left_usize))
            .and_then(|pixel| pixel.checked_mul(4))
            .ok_or_else(|| decode_error(source, "GIF canvas offset overflow"))?;
        let src_end = src_start
            .checked_add(byte_count)
            .ok_or_else(|| decode_error(source, "GIF frame blit exceeds decoded raster"))?;
        let dst_end = dst_start
            .checked_add(byte_count)
            .ok_or_else(|| decode_error(source, "GIF canvas blit exceeds allocated raster"))?;
        let src = rgba
            .get(src_start..src_end)
            .ok_or_else(|| decode_error(source, "GIF frame blit exceeds decoded raster"))?;
        let dst = canvas
            .get_mut(dst_start..dst_end)
            .ok_or_else(|| decode_error(source, "GIF canvas blit exceeds allocated raster"))?;
        dst.copy_from_slice(src);
    }
    Ok(())
}

/// Decodes the first (and only) frame of a still GIF into `pixel_buf`.
pub(crate) fn decode_gif_slice_into(
    source: &Path,
    file_bytes: &[u8],
    pixel_buf: &mut Vec<u8>,
) -> Result<(u32, u32, DecodedPixelFormat, usize), PipelineError> {
    let mut options = gif::DecodeOptions::new();
    options.set_color_output(gif::ColorOutput::RGBA);
    options.set_memory_limit(gif::MemoryLimit::Bytes(
        NonZeroU64::new(MAX_GIF_DECODE_BYTES).unwrap_or(NonZeroU64::MIN),
    ));
    let mut decoder = options
        .read_info(Cursor::new(file_bytes))
        .map_err(|err| decode_error(source, format!("invalid GIF container: {err}")))?;

    let canvas_w = u32::from(decoder.width());
    let canvas_h = u32::from(decoder.height());
    if canvas_w == 0 || canvas_h == 0 {
        return Err(decode_error(source, "GIF has a zero-sized canvas"));
    }
    let canvas_pixels = u64::from(canvas_w)
        .checked_mul(u64::from(canvas_h))
        .ok_or_else(|| decode_error(source, "GIF canvas pixel count overflow"))?;
    if canvas_pixels > MAX_GIF_DECODE_BYTES / 4 {
        return Err(decode_error(
            source,
            "GIF canvas exceeds the decoded memory limit",
        ));
    }

    let (left, top, frame_w, frame_h, rgba) = {
        let frame = decoder
            .read_next_frame()
            .map_err(|err| decode_error(source, format!("failed to decode GIF frame: {err}")))?
            .ok_or_else(|| decode_error(source, "GIF contains no image frames"))?;
        (
            u32::from(frame.left),
            u32::from(frame.top),
            u32::from(frame.width),
            u32::from(frame.height),
            frame.buffer.to_vec(),
        )
    };
    let frame_bytes = usize::try_from(frame_w)
        .ok()
        .and_then(|w| {
            usize::try_from(frame_h)
                .ok()
                .and_then(|h| w.checked_mul(h).and_then(|p| p.checked_mul(4)))
        })
        .ok_or_else(|| decode_error(source, "GIF frame size overflow"))?;
    if rgba.len() != frame_bytes {
        return Err(decode_error(
            source,
            "decoded GIF frame has unexpected length",
        ));
    }

    if decoder
        .next_frame_info()
        .map_err(|err| decode_error(source, format!("invalid GIF frame: {err}")))?
        .is_some()
    {
        return Err(decode_error(
            source,
            "animated GIF is not supported by the still-image pipeline",
        ));
    }

    let canvas_len = usize::try_from(canvas_w)
        .ok()
        .and_then(|w| {
            usize::try_from(canvas_h)
                .ok()
                .and_then(|h| w.checked_mul(h).and_then(|p| p.checked_mul(4)))
        })
        .ok_or_else(|| decode_error(source, "GIF canvas size overflow"))?;
    let canvas = if left == 0 && top == 0 && frame_w == canvas_w && frame_h == canvas_h {
        rgba
    } else {
        let mut canvas = vec![0_u8; canvas_len];
        compose_sub_rect(
            source,
            &mut canvas,
            &rgba,
            canvas_w,
            canvas_h,
            FrameRect {
                left,
                top,
                width: frame_w,
                height: frame_h,
            },
        )?;
        canvas
    };

    let (format, byte_len) = store_rgba_canvas(
        source,
        pixel_buf,
        &canvas,
        usize::from(decoder.width()),
        usize::from(decoder.height()),
    )?;
    Ok((canvas_w, canvas_h, format, byte_len))
}

/// Encodes one still GIF frame from already validated decoded pixels.
pub(crate) fn encode_gif_into(
    target: &Path,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    aux_buf: &mut Vec<u8>,
    encoded_buf: &mut Vec<u8>,
) -> Result<(), PipelineError> {
    let reason = |why: String| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: why,
    };
    let w = u16::try_from(decoded.width).map_err(|_| {
        reason(format!(
            "GIF dimensions exceed 65535 limit: {}x{}",
            decoded.width, decoded.height
        ))
    })?;
    let h = u16::try_from(decoded.height).map_err(|_| {
        reason(format!(
            "GIF dimensions exceed 65535 limit: {}x{}",
            decoded.width, decoded.height
        ))
    })?;
    if w == 0 || h == 0 {
        return Err(reason("GIF dimensions must be non-zero".to_owned()));
    }

    let bpp = match decoded.format {
        DecodedPixelFormat::L8 => 1_usize,
        DecodedPixelFormat::La8 => 2_usize,
        DecodedPixelFormat::Rgb8 => 3_usize,
        DecodedPixelFormat::Rgba8 => 4_usize,
    };
    let expected = usize::from(w)
        .checked_mul(usize::from(h))
        .and_then(|pixels| pixels.checked_mul(bpp))
        .ok_or_else(|| reason("GIF pixel buffer size overflow".to_owned()))?;
    if pixels.len() != expected {
        return Err(reason(format!(
            "GIF decoded pixel buffer has {} bytes, expected {expected}",
            pixels.len()
        )));
    }

    let frame = match decoded.format {
        DecodedPixelFormat::L8 => {
            let palette: Vec<u8> = (0_u8..=255).flat_map(|v| [v, v, v]).collect();
            gif::Frame::from_palette_pixels(w, h, pixels.to_vec(), palette, None)
        }
        DecodedPixelFormat::La8 => gif::Frame::from_grayscale_with_alpha(w, h, pixels),
        DecodedPixelFormat::Rgb8 => gif::Frame::from_rgb_speed(w, h, pixels, 10),
        DecodedPixelFormat::Rgba8 => {
            aux_buf.clear();
            aux_buf.extend_from_slice(pixels);
            gif::Frame::from_rgba_speed(w, h, aux_buf.as_mut_slice(), 10)
        }
    };

    encoded_buf.clear();
    let mut encoder = gif::Encoder::new(&mut *encoded_buf, w, h, &[])
        .map_err(|err| reason(format!("failed to create GIF encoder: {err}")))?;
    encoder
        .write_frame(&frame)
        .map_err(|err| reason(format!("failed to encode GIF frame: {err}")))?;
    drop(encoder);
    if encoded_buf.is_empty() {
        return Err(PipelineError::EmptyOutput(target.to_path_buf()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(format: DecodedPixelFormat, width: u32, height: u32) -> DecodedFrameMeta {
        DecodedFrameMeta {
            width,
            height,
            format,
            byte_len: 0,
            input_file_bytes: 0,
            has_non_opaque_native_alpha: false,
        }
    }

    fn encode_pixels(
        format: DecodedPixelFormat,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Result<Vec<u8>, PipelineError> {
        let mut encoded = Vec::new();
        let mut aux = Vec::new();
        encode_gif_into(
            Path::new("test.gif"),
            pixels,
            meta(format, width, height),
            &mut aux,
            &mut encoded,
        )?;
        assert_eq!(&encoded[..3], b"GIF");
        Ok(encoded)
    }

    fn decode_pixels(
        encoded: &[u8],
    ) -> Result<(u32, u32, DecodedPixelFormat, Vec<u8>), PipelineError> {
        let mut pixels = Vec::new();
        let (w, h, format, len) =
            decode_gif_slice_into(Path::new("test.gif"), encoded, &mut pixels)?;
        pixels.truncate(len);
        Ok((w, h, format, pixels))
    }

    fn rgba_from(opaque: &[u8], format: DecodedPixelFormat) -> Vec<u8> {
        match format {
            DecodedPixelFormat::Rgb8 => opaque
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|&[r, g, b]| [r, g, b, 255])
                .collect(),
            DecodedPixelFormat::Rgba8 => opaque.to_vec(),
            DecodedPixelFormat::L8 => opaque.iter().flat_map(|&v| [v, v, v, 255]).collect(),
            DecodedPixelFormat::La8 => opaque
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|&[v, a]| [v, v, v, a])
                .collect(),
        }
    }

    #[test]
    fn l8_lossless_roundtrip() -> Result<(), PipelineError> {
        let pixels: Vec<u8> = (0..16_u32 * 16)
            .map(|i| u8::try_from(i % 256).unwrap_or(0))
            .collect();
        let encoded = encode_pixels(DecodedPixelFormat::L8, 16, 16, &pixels)?;
        let (w, h, format, decoded) = decode_pixels(&encoded)?;
        assert_eq!((w, h), (16, 16));
        assert!(matches!(
            format,
            DecodedPixelFormat::L8 | DecodedPixelFormat::Rgb8
        ));
        let rgba = rgba_from(&decoded, format);
        for (got, expected) in rgba.as_chunks::<4>().0.iter().zip(pixels.iter()) {
            assert_eq!(got[0], *expected);
            assert_eq!(got[1], *expected);
            assert_eq!(got[2], *expected);
            assert_eq!(got[3], 255);
        }
        Ok(())
    }

    #[test]
    fn la8_transparent_roundtrip() -> Result<(), PipelineError> {
        let mut pixels = Vec::new();
        for i in 0..16_u32 * 16 {
            pixels.extend_from_slice(&[
                u8::try_from(i % 256).unwrap_or(0),
                if i % 3 == 0 { 0 } else { 255 },
            ]);
        }
        let encoded = encode_pixels(DecodedPixelFormat::La8, 16, 16, &pixels)?;
        let (w, h, format, decoded) = decode_pixels(&encoded)?;
        assert_eq!((w, h), (16, 16));
        let rgba = rgba_from(&decoded, format);
        for (i, got) in rgba.as_chunks::<4>().0.iter().enumerate() {
            let expected_alpha = if i % 3 == 0 { 0 } else { 255 };
            assert_eq!(got[3], expected_alpha);
            if expected_alpha == 255 {
                assert_eq!(got[0], pixels[i * 2]);
            }
        }
        Ok(())
    }

    #[test]
    fn rgb8_roundtrip() -> Result<(), PipelineError> {
        let mut pixels = Vec::new();
        for i in 0..16_u32 * 16 {
            let color = (i % 16) as u8 * 16;
            pixels.extend_from_slice(&[color, 255 - color, color ^ 255]);
        }
        let encoded = encode_pixels(DecodedPixelFormat::Rgb8, 16, 16, &pixels)?;
        let (w, h, format, decoded) = decode_pixels(&encoded)?;
        assert_eq!((w, h), (16, 16));
        assert_eq!(format, DecodedPixelFormat::Rgb8);
        assert_eq!(decoded, pixels);
        Ok(())
    }

    #[test]
    fn rgba8_transparent_roundtrip() -> Result<(), PipelineError> {
        let mut pixels = Vec::new();
        for i in 0..16_u32 * 16 {
            let color = (i % 16) as u8 * 16;
            let alpha = if i % 4 == 0 { 0 } else { 255 };
            pixels.extend_from_slice(&[color, 255 - color, color, alpha]);
        }
        let encoded = encode_pixels(DecodedPixelFormat::Rgba8, 16, 16, &pixels)?;
        let (w, h, format, decoded) = decode_pixels(&encoded)?;
        assert_eq!((w, h), (16, 16));
        assert_eq!(format, DecodedPixelFormat::Rgba8);
        for (i, got) in decoded.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(got[3], pixels[i * 4 + 3]);
            if got[3] == 255 {
                assert_eq!(got[..3], pixels[i * 4..i * 4 + 3]);
            }
        }
        Ok(())
    }

    #[test]
    fn sub_rectangle_canvas_decode() -> Result<(), PipelineError> {
        let mut encoded = Vec::new();
        let palette: Vec<u8> = vec![0, 0, 0, 255, 0, 0];
        let frame = gif::Frame {
            left: 1,
            top: 1,
            width: 2,
            height: 2,
            buffer: std::borrow::Cow::Borrowed(&[0, 1, 1, 0]),
            palette: Some(palette),
            ..gif::Frame::default()
        };
        let mut gif_encoder = gif::Encoder::new(&mut encoded, 4, 4, &[])
            .map_err(|err| decode_error(Path::new("test.gif"), err.to_string()))?;
        gif_encoder
            .write_frame(&frame)
            .map_err(|err| decode_error(Path::new("test.gif"), err.to_string()))?;
        drop(gif_encoder);

        let (w, h, format, decoded) = decode_pixels(&encoded)?;
        assert_eq!((w, h), (4, 4));
        assert_eq!(format, DecodedPixelFormat::Rgba8);
        let rgba = rgba_from(&decoded, format);
        let at = |x: usize, y: usize| -> [u8; 4] {
            let offset = (y * 4 + x) * 4;
            [
                rgba[offset],
                rgba[offset + 1],
                rgba[offset + 2],
                rgba[offset + 3],
            ]
        };
        assert_eq!(at(0, 0), [0, 0, 0, 0]);
        assert_eq!(at(1, 1), [0, 0, 0, 255]);
        assert_eq!(at(2, 2), [0, 0, 0, 255]);
        assert_eq!(at(1, 2), [255, 0, 0, 255]);
        assert_eq!(at(0, 3), [0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn animated_gif_is_rejected() -> Result<(), PipelineError> {
        let mut encoded = Vec::new();
        {
            let mut gif_encoder = gif::Encoder::new(&mut encoded, 4, 4, &[])
                .map_err(|err| decode_error(Path::new("test.gif"), err.to_string()))?;
            let frame = gif::Frame::from_rgb_speed(4, 4, &[64_u8; 4 * 4 * 3], 10);
            gif_encoder
                .write_frame(&frame)
                .map_err(|err| decode_error(Path::new("test.gif"), err.to_string()))?;
            gif_encoder
                .write_frame(&frame)
                .map_err(|err| decode_error(Path::new("test.gif"), err.to_string()))?;
        }
        let mut pixels = Vec::new();
        let result = decode_gif_slice_into(Path::new("test.gif"), &encoded, &mut pixels);
        assert!(matches!(
            result,
            Err(PipelineError::DecodeFailed { ref reason, .. })
                if reason.contains("animated GIF")
        ));
        Ok(())
    }
}
