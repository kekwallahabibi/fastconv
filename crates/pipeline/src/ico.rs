use std::io::Cursor;
use std::path::Path;

use crate::PipelineError;
use crate::decode::{DecodedFrameMeta, DecodedPixelFormat};

const MAX_ICO_DIMENSION: u32 = 256;

fn decode_error(source: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::DecodeFailed {
        path: source.to_path_buf(),
        reason: reason.into(),
    }
}

fn store_pixels(
    source: &Path,
    pixel_buf: &mut Vec<u8>,
    rgba: &[u8],
    width: usize,
    height: usize,
) -> Result<(DecodedPixelFormat, usize), PipelineError> {
    let expected = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| decode_error(source, "ICO canvas size overflow"))?;
    if rgba.len() != expected {
        return Err(decode_error(
            source,
            "decoded ICO raster has unexpected length",
        ));
    }
    let all_opaque = rgba.as_chunks::<4>().0.iter().all(|pixel| pixel[3] == 255);
    let (format, bpp) = if all_opaque {
        (DecodedPixelFormat::Rgb8, 3_usize)
    } else {
        (DecodedPixelFormat::Rgba8, 4_usize)
    };
    let byte_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(bpp))
        .ok_or_else(|| decode_error(source, "ICO output buffer size overflow"))?;
    if pixel_buf.len() < byte_len {
        pixel_buf.resize(byte_len, 0);
    }
    let dst = &mut pixel_buf[..byte_len];
    if all_opaque {
        for (dst_pixel, src_pixel) in dst
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(rgba.as_chunks::<4>().0.iter())
        {
            dst_pixel.copy_from_slice(&src_pixel[..3]);
        }
    } else {
        dst.copy_from_slice(rgba);
    }
    Ok((format, byte_len))
}

/// Decodes the largest still icon from an ICO directory into `pixel_buf`.
pub(crate) fn decode_ico_slice_into(
    source: &Path,
    file_bytes: &[u8],
    pixel_buf: &mut Vec<u8>,
) -> Result<(u32, u32, DecodedPixelFormat, usize), PipelineError> {
    let icon_dir = ico::IconDir::read(Cursor::new(file_bytes))
        .map_err(|err| decode_error(source, format!("invalid ICO container: {err}")))?;
    if icon_dir.resource_type() != ico::ResourceType::Icon {
        return Err(decode_error(
            source,
            "CUR cursor files are not supported (expected ICO icon resource)",
        ));
    }

    let mut best: Option<&ico::IconDirEntry> = None;
    for entry in icon_dir.entries() {
        let score = (
            u64::from(entry.width()).saturating_mul(u64::from(entry.height())),
            entry.bits_per_pixel(),
        );
        let select = match best {
            None => true,
            Some(current) => {
                let current_score = (
                    u64::from(current.width()).saturating_mul(u64::from(current.height())),
                    current.bits_per_pixel(),
                );
                score > current_score
            }
        };
        if select {
            best = Some(entry);
        }
    }
    let entry =
        best.ok_or_else(|| decode_error(source, "ICO container contains no icon entries"))?;
    let image = entry
        .decode()
        .map_err(|err| decode_error(source, format!("failed to decode ICO icon: {err}")))?;
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        return Err(decode_error(source, "ICO icon has a zero-sized canvas"));
    }
    let width_usize = usize::try_from(width)
        .map_err(|_| decode_error(source, "ICO width exceeds addressable memory"))?;
    let height_usize = usize::try_from(height)
        .map_err(|_| decode_error(source, "ICO height exceeds addressable memory"))?;

    let (format, byte_len) = store_pixels(
        source,
        pixel_buf,
        image.rgba_data(),
        width_usize,
        height_usize,
    )?;
    Ok((width, height, format, byte_len))
}

/// Encodes one still ICO icon from already validated decoded pixels.
pub(crate) fn encode_ico_into(
    target: &Path,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    encoded_buf: &mut Vec<u8>,
) -> Result<(), PipelineError> {
    let reason = |why: String| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: why,
    };
    for dimension in [decoded.width, decoded.height] {
        if dimension == 0 || dimension > MAX_ICO_DIMENSION {
            return Err(reason(format!(
                "ICO format supports maximum dimensions of 256x256, got {}x{}",
                decoded.width, decoded.height
            )));
        }
    }

    let width = usize::try_from(decoded.width)
        .map_err(|_| reason("ICO width exceeds addressable memory".to_owned()))?;
    let height = usize::try_from(decoded.height)
        .map_err(|_| reason("ICO height exceeds addressable memory".to_owned()))?;
    let bpp = match decoded.format {
        DecodedPixelFormat::L8 => 1_usize,
        DecodedPixelFormat::La8 => 2_usize,
        DecodedPixelFormat::Rgb8 => 3_usize,
        DecodedPixelFormat::Rgba8 => 4_usize,
    };
    let expected = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(bpp))
        .ok_or_else(|| reason("ICO pixel buffer size overflow".to_owned()))?;
    if pixels.len() != expected {
        return Err(reason(format!(
            "ICO decoded pixel buffer has {} bytes, expected {expected}",
            pixels.len()
        )));
    }

    let rgba: Vec<u8> = match decoded.format {
        DecodedPixelFormat::Rgba8 => pixels.to_vec(),
        DecodedPixelFormat::L8 => pixels.iter().flat_map(|&v| [v, v, v, 255]).collect(),
        DecodedPixelFormat::La8 => pixels
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|&[v, a]| [v, v, v, a])
            .collect(),
        DecodedPixelFormat::Rgb8 => pixels
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&[r, g, b]| [r, g, b, 255])
            .collect(),
    };
    let icon = ico::IconImage::from_rgba_data(decoded.width, decoded.height, rgba);
    let entry = ico::IconDirEntry::encode(&icon)
        .map_err(|err| reason(format!("failed to encode ICO entry: {err}")))?;
    let mut dir = ico::IconDir::new(ico::ResourceType::Icon);
    dir.add_entry(entry);

    encoded_buf.clear();
    dir.write(&mut *encoded_buf)
        .map_err(|err| reason(format!("failed to serialize ICO directory: {err}")))?;
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
        encode_ico_into(
            Path::new("test.ico"),
            pixels,
            meta(format, width, height),
            &mut encoded,
        )?;
        Ok(encoded)
    }

    fn decode_pixels(
        encoded: &[u8],
    ) -> Result<(u32, u32, DecodedPixelFormat, Vec<u8>), PipelineError> {
        let mut pixels = Vec::new();
        let (w, h, format, len) =
            decode_ico_slice_into(Path::new("test.ico"), encoded, &mut pixels)?;
        pixels.truncate(len);
        Ok((w, h, format, pixels))
    }

    fn rgba_from(decoded: &[u8], format: DecodedPixelFormat) -> Vec<u8> {
        match format {
            DecodedPixelFormat::Rgb8 => decoded
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|&[r, g, b]| [r, g, b, 255])
                .collect(),
            DecodedPixelFormat::Rgba8 => decoded.to_vec(),
            DecodedPixelFormat::L8 => decoded.iter().flat_map(|&v| [v, v, v, 255]).collect(),
            DecodedPixelFormat::La8 => decoded
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|&[v, a]| [v, v, v, a])
                .collect(),
        }
    }

    fn sample_rgb(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::new();
        for i in 0..width * height {
            let color = u8::try_from((i % 8) * 31).unwrap_or(0);
            pixels.extend_from_slice(&[color, 255 - color, color ^ 255]);
        }
        pixels
    }

    fn sample_rgba(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::new();
        for i in 0..width * height {
            let color = u8::try_from((i % 8) * 31).unwrap_or(0);
            let alpha = if i % 5 == 0 { 0 } else { 255 };
            pixels.extend_from_slice(&[color, 255 - color, color, alpha]);
        }
        pixels
    }

    #[test]
    fn rgb_and_rgba_roundtrip_for_all_sizes() -> Result<(), PipelineError> {
        for size in [16_u32, 32, 128, 256] {
            for (format, pixels) in [
                (DecodedPixelFormat::Rgb8, sample_rgb(size, size)),
                (DecodedPixelFormat::Rgba8, sample_rgba(size, size)),
            ] {
                let encoded = encode_pixels(format, size, size, &pixels)?;
                let (w, h, decoded_format, decoded) = decode_pixels(&encoded)?;
                assert_eq!((w, h), (size, size));
                assert_eq!(decoded_format, format);
                assert_eq!(decoded, pixels);
            }
        }
        Ok(())
    }

    #[test]
    fn grayscale_and_grayscale_alpha_roundtrip() -> Result<(), PipelineError> {
        let size = 32_u32;
        let l8: Vec<u8> = (0..size * size)
            .map(|i| u8::try_from(i % 256).unwrap_or(0))
            .collect();
        let encoded = encode_pixels(DecodedPixelFormat::L8, size, size, &l8)?;
        let (w, h, format, decoded) = decode_pixels(&encoded)?;
        assert_eq!((w, h), (size, size));
        let rgba = rgba_from(&decoded, format);
        for (got, value) in rgba.as_chunks::<4>().0.iter().zip(l8.iter()) {
            assert_eq!(&got[..3], &[*value, *value, *value]);
            assert_eq!(got[3], 255);
        }

        let mut la = Vec::new();
        for i in 0..size * size {
            la.extend_from_slice(&[
                u8::try_from(i % 256).unwrap_or(0),
                if i % 4 == 0 { 0 } else { 255 },
            ]);
        }
        let encoded = encode_pixels(DecodedPixelFormat::La8, size, size, &la)?;
        let (w, h, format, decoded) = decode_pixels(&encoded)?;
        assert_eq!((w, h), (size, size));
        assert_eq!(format, DecodedPixelFormat::Rgba8);
        let rgba = rgba_from(&decoded, format);
        for (i, got) in rgba.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(got[3], if i % 4 == 0 { 0 } else { 255 });
            if got[3] == 255 {
                assert_eq!(got[0], la[i * 2]);
            }
        }
        Ok(())
    }

    #[test]
    fn selects_largest_icon_entry() -> Result<(), PipelineError> {
        let small = sample_rgb(16, 16);
        let large = sample_rgb(32, 32);
        let small_encoded = encode_pixels(DecodedPixelFormat::Rgb8, 16, 16, &small)?;
        let large_encoded = encode_pixels(DecodedPixelFormat::Rgb8, 32, 32, &large)?;

        let mut dir = ico::IconDir::new(ico::ResourceType::Icon);
        let small_entry = ico::IconDir::read(Cursor::new(&small_encoded))
            .map_err(|err| decode_error(Path::new("test.ico"), err.to_string()))?
            .entries()[0]
            .clone();
        let large_entry = ico::IconDir::read(Cursor::new(&large_encoded))
            .map_err(|err| decode_error(Path::new("test.ico"), err.to_string()))?
            .entries()[0]
            .clone();
        dir.add_entry(small_entry);
        dir.add_entry(large_entry);
        let mut combined = Vec::new();
        dir.write(&mut combined)
            .map_err(|err| decode_error(Path::new("test.ico"), err.to_string()))?;

        let (w, h, format, decoded) = decode_pixels(&combined)?;
        assert_eq!((w, h), (32, 32));
        assert_eq!(format, DecodedPixelFormat::Rgb8);
        assert_eq!(decoded, large);
        Ok(())
    }

    #[test]
    fn rejects_oversized_dimension_and_truncated_pixels() {
        let bad = encode_ico_into(
            Path::new("big.ico"),
            &vec![0_u8; 257 * 257 * 3],
            meta(DecodedPixelFormat::Rgb8, 257, 257),
            &mut Vec::new(),
        );
        assert!(matches!(
            bad,
            Err(PipelineError::EncodeFailed { ref reason, .. }) if reason.contains("256x256")
        ));

        let truncated = encode_ico_into(
            Path::new("short.ico"),
            &[0_u8; 3],
            meta(DecodedPixelFormat::Rgb8, 16, 16),
            &mut Vec::new(),
        );
        assert!(matches!(truncated, Err(PipelineError::EncodeFailed { .. })));
    }

    #[test]
    fn rejects_cursor_resource_type() -> Result<(), PipelineError> {
        let mut icon = ico::IconImage::from_rgba_data(16, 16, vec![255_u8; 16 * 16 * 4]);
        icon.set_cursor_hotspot(Some((0, 0)));
        let entry = ico::IconDirEntry::encode(&icon)
            .map_err(|err| decode_error(Path::new("test.ico"), err.to_string()))?;
        let mut dir = ico::IconDir::new(ico::ResourceType::Cursor);
        dir.add_entry(entry);
        let mut bytes = Vec::new();
        dir.write(&mut bytes)
            .map_err(|err| decode_error(Path::new("test.cur"), err.to_string()))?;

        let mut pixels = Vec::new();
        let result = decode_ico_slice_into(Path::new("test.cur"), &bytes, &mut pixels);
        assert!(matches!(
            result,
            Err(PipelineError::DecodeFailed { ref reason, .. }) if reason.contains("CUR")
        ));
        Ok(())
    }
}
