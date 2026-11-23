use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, Read};

use fastconv_domain::{ImageBatchItem, ImageInputFormat};
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

use crate::PipelineError;
use crate::bmp::decode_bmp_slice_into;
use crate::qoi::decode_qoi_slice_into;

/// Decoded 8-bit pixel format supported across CPU and GPU image pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecodedPixelFormat {
    L8,
    La8,
    Rgb8,
    Rgba8,
}

/// Preferred pixel layout hint requested by an encoder backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecodeTargetHint {
    /// Keep native luma/RGB/alpha channels without unnecessary expansion.
    Native,
    /// Prefer direct 4-channel RGBA8 output when the decoder natively supports it (e.g. JPEG -> NVENC).
    #[cfg(feature = "nvenc")]
    OpaqueRgba8,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DecodedFrameMeta {
    pub width: u32,
    pub height: u32,
    pub format: DecodedPixelFormat,
    pub byte_len: usize,
    pub input_file_bytes: u64,
}

pub(crate) fn precreate_target_directories(items: &[ImageBatchItem]) -> Result<(), PipelineError> {
    let mut parents = BTreeSet::new();
    for item in items {
        if let Some(parent) = item.target().parent()
            && !parent.as_os_str().is_empty()
        {
            parents.insert(parent);
        }
    }
    for dir in parents {
        fs::create_dir_all(dir).map_err(|source| PipelineError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}

/// Reads and decodes a validated still-image item into `pixel_buf`, reusing `file_bytes`.
pub(crate) fn decode_image_into(
    item: &ImageBatchItem,
    hint: DecodeTargetHint,
    file_bytes: &mut Vec<u8>,
    pixel_buf: &mut Vec<u8>,
) -> Result<DecodedFrameMeta, PipelineError> {
    file_bytes.clear();
    let mut file = File::open(item.source()).map_err(|source| PipelineError::Io {
        path: item.source().to_path_buf(),
        source,
    })?;
    file.read_to_end(file_bytes)
        .map_err(|source| PipelineError::Io {
            path: item.source().to_path_buf(),
            source,
        })?;
    let input_file_bytes = u64::try_from(file_bytes.len()).unwrap_or(u64::MAX);

    match item.input_format() {
        ImageInputFormat::Png => {
            decode_png_slice_into(item, file_bytes.as_slice(), input_file_bytes, pixel_buf)
        }
        ImageInputFormat::Jpeg => decode_jpeg_slice_into(
            item,
            hint,
            file_bytes.as_slice(),
            input_file_bytes,
            pixel_buf,
        ),
        ImageInputFormat::Webp => {
            decode_webp_slice_into(item, file_bytes.as_slice(), input_file_bytes, pixel_buf)
        }
        ImageInputFormat::Bmp => {
            let (width, height, format, byte_len) =
                decode_bmp_slice_into(item.source(), file_bytes.as_slice(), hint, pixel_buf)?;
            verify_decoded_dimensions(item, width, height)?;
            Ok(DecodedFrameMeta {
                width,
                height,
                format,
                byte_len,
                input_file_bytes,
            })
        }
        ImageInputFormat::Qoi => {
            let (width, height, format, byte_len) =
                decode_qoi_slice_into(item.source(), file_bytes.as_slice(), hint, pixel_buf)?;
            verify_decoded_dimensions(item, width, height)?;
            Ok(DecodedFrameMeta {
                width,
                height,
                format,
                byte_len,
                input_file_bytes,
            })
        }
    }
}

fn decode_webp_slice_into(
    item: &ImageBatchItem,
    file_bytes: &[u8],
    input_file_bytes: u64,
    pixel_buf: &mut Vec<u8>,
) -> Result<DecodedFrameMeta, PipelineError> {
    let mut decoder =
        zenwebp::WebPDecoder::build(file_bytes).map_err(|err| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: err.to_string(),
        })?;
    if decoder.is_animated() {
        return Err(PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: "animated WebP is not supported by the still-image pipeline".to_owned(),
        });
    }

    let (width, height) = decoder.dimensions();
    verify_decoded_dimensions(item, width, height)?;

    let required_len = decoder
        .output_buffer_size()
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: "decoded WebP buffer size exceeds addressable memory".to_owned(),
        })?;
    if pixel_buf.len() < required_len {
        pixel_buf.resize(required_len, 0);
    }

    let dst = pixel_buf
        .get_mut(..required_len)
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: "decoded WebP buffer slice out of bounds".to_owned(),
        })?;
    decoder
        .read_image(dst)
        .map_err(|err| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: err.to_string(),
        })?;

    let format = if decoder.has_alpha() {
        DecodedPixelFormat::Rgba8
    } else {
        DecodedPixelFormat::Rgb8
    };

    Ok(DecodedFrameMeta {
        width,
        height,
        format,
        byte_len: required_len,
        input_file_bytes,
    })
}

fn verify_decoded_dimensions(
    item: &ImageBatchItem,
    width: u32,
    height: u32,
) -> Result<(), PipelineError> {
    let expected = item.dimensions();
    if width != expected.width().get() || height != expected.height().get() {
        return Err(PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: format!("decoded dimensions {width}x{height} do not match probed {expected}"),
        });
    }
    Ok(())
}

fn decode_png_slice_into(
    item: &ImageBatchItem,
    file_bytes: &[u8],
    input_file_bytes: u64,
    pixel_buf: &mut Vec<u8>,
) -> Result<DecodedFrameMeta, PipelineError> {
    let mut decoder = png::Decoder::new(io::Cursor::new(file_bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    decoder.set_ignore_text_chunk(true);

    let mut reader = decoder
        .read_info()
        .map_err(|err| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: err.to_string(),
        })?;

    let required_len = reader
        .output_buffer_size()
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: "decoded PNG buffer size exceeds addressable memory".to_owned(),
        })?;
    if pixel_buf.len() < required_len {
        pixel_buf.resize(required_len, 0);
    }

    let frame = reader
        .next_frame(pixel_buf)
        .map_err(|err| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: err.to_string(),
        })?;

    verify_decoded_dimensions(item, frame.width, frame.height)?;

    let format = match (frame.color_type, frame.bit_depth) {
        (png::ColorType::Grayscale, png::BitDepth::Eight) => DecodedPixelFormat::L8,
        (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight) => DecodedPixelFormat::La8,
        (png::ColorType::Rgb, png::BitDepth::Eight) => DecodedPixelFormat::Rgb8,
        (png::ColorType::Rgba, png::BitDepth::Eight) => DecodedPixelFormat::Rgba8,
        (color, depth) => {
            return Err(PipelineError::UnsupportedPixelFormat {
                path: item.source().to_path_buf(),
                format: format!("{color:?} {depth:?}"),
            });
        }
    };

    let byte_len = frame.buffer_size();
    if byte_len > pixel_buf.len() {
        return Err(PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: "decoded PNG frame exceeded allocated pixel buffer".to_owned(),
        });
    }

    Ok(DecodedFrameMeta {
        width: frame.width,
        height: frame.height,
        format,
        byte_len,
        input_file_bytes,
    })
}

fn decode_jpeg_slice_into(
    item: &ImageBatchItem,
    hint: DecodeTargetHint,
    file_bytes: &[u8],
    input_file_bytes: u64,
    pixel_buf: &mut Vec<u8>,
) -> Result<DecodedFrameMeta, PipelineError> {
    let initial_cs = match hint {
        DecodeTargetHint::Native => ColorSpace::RGB,
        #[cfg(feature = "nvenc")]
        DecodeTargetHint::OpaqueRgba8 => ColorSpace::RGBA,
    };

    let mut decoder = JpegDecoder::new_with_options(
        ZCursor::new(file_bytes),
        DecoderOptions::default().jpeg_set_out_colorspace(initial_cs),
    );
    decoder
        .decode_headers()
        .map_err(|err| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: err.to_string(),
        })?;

    let info = decoder.info().ok_or_else(|| PipelineError::DecodeFailed {
        path: item.source().to_path_buf(),
        reason: "missing JPEG SOF header metadata".to_owned(),
    })?;
    let width = u32::from(info.width);
    let height = u32::from(info.height);
    verify_decoded_dimensions(item, width, height)?;

    let (final_cs, format) = match (hint, decoder.input_colorspace()) {
        (DecodeTargetHint::Native, Some(ColorSpace::Luma)) => {
            (ColorSpace::Luma, DecodedPixelFormat::L8)
        }
        (
            DecodeTargetHint::Native,
            Some(ColorSpace::YCbCr | ColorSpace::RGB | ColorSpace::CMYK | ColorSpace::YCCK),
        ) => (ColorSpace::RGB, DecodedPixelFormat::Rgb8),
        // `zune-jpeg` leaves RGB->RGBA unimplemented and packs 3-byte pixels/strides into
        // CMYK/YCCK->RGBA outputs; route all three via RGB so the caller expands to RGBA8 cleanly.
        #[cfg(feature = "nvenc")]
        (
            DecodeTargetHint::OpaqueRgba8,
            Some(ColorSpace::RGB | ColorSpace::CMYK | ColorSpace::YCCK),
        ) => (ColorSpace::RGB, DecodedPixelFormat::Rgb8),
        #[cfg(feature = "nvenc")]
        (DecodeTargetHint::OpaqueRgba8, Some(ColorSpace::Luma | ColorSpace::YCbCr)) => {
            (ColorSpace::RGBA, DecodedPixelFormat::Rgba8)
        }
        (_, other) => {
            return Err(PipelineError::UnsupportedPixelFormat {
                path: item.source().to_path_buf(),
                format: format!("JPEG color space {other:?}"),
            });
        }
    };

    if final_cs != initial_cs {
        // Re-create so `color_convert_16` and `output_buffer_size` match `final_cs`.
        decoder = JpegDecoder::new_with_options(
            ZCursor::new(file_bytes),
            DecoderOptions::default().jpeg_set_out_colorspace(final_cs),
        );
        decoder
            .decode_headers()
            .map_err(|err| PipelineError::DecodeFailed {
                path: item.source().to_path_buf(),
                reason: err.to_string(),
            })?;
    }

    let required_len = decoder
        .output_buffer_size()
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: "decoded JPEG buffer size exceeds addressable memory".to_owned(),
        })?;
    if pixel_buf.len() < required_len {
        pixel_buf.resize(required_len, 0);
    }

    let dst = pixel_buf
        .get_mut(..required_len)
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: "decoded JPEG buffer slice out of bounds".to_owned(),
        })?;
    decoder
        .decode_into(dst)
        .map_err(|err| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: err.to_string(),
        })?;

    Ok(DecodedFrameMeta {
        width,
        height,
        format,
        byte_len: required_len,
        input_file_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::{DecodeTargetHint, DecodedPixelFormat, decode_image_into};
    use crate::PipelineError;
    use fastconv_domain::{ImageBatchItem, ImageDimensions, ImageInputFormat};
    use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder, SamplingFactor};
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};

    static DECODE_TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn create_temp_dir(label: &str) -> Result<PathBuf, PipelineError> {
        let id = DECODE_TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("fastconv-decode-{label}-{}-{id}", process::id()));
        fs::create_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(dir)
    }

    #[test]
    fn decodes_optimized_huffman_solid_green_jpeg_with_accurate_color()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = create_temp_dir("huffman-green")?;
        let src = root.join("solid_green.jpg");
        let dst = root.join("unused.webp");

        let mut pixels = Vec::with_capacity(16 * 16 * 3);
        for _ in 0..(16 * 16) {
            pixels.extend_from_slice(&[0, 255, 0]);
        }

        let mut encoder = JpegEncoder::new_file(&src, 80)?;
        encoder.set_sampling_factor(SamplingFactor::F_2_2);
        encoder.set_optimized_huffman_tables(true);
        encoder.encode(&pixels, 16, 16, JpegColorType::Rgb)?;

        let item = ImageBatchItem::new(
            src,
            dst,
            ImageInputFormat::Jpeg,
            ImageDimensions::new(16, 16)?,
        )?;
        let mut file_bytes = Vec::new();
        let mut pixel_buf = Vec::new();
        let meta = decode_image_into(
            &item,
            DecodeTargetHint::Native,
            &mut file_bytes,
            &mut pixel_buf,
        )?;

        assert_eq!(meta.format, DecodedPixelFormat::Rgb8);
        assert_eq!(meta.byte_len, 16 * 16 * 3);
        let center_idx = (8 * 16 + 8) * 3;
        let center = pixel_buf
            .get(center_idx..center_idx + 3)
            .ok_or("missing center pixel")?;
        assert!(
            center[0] <= 5 && center[1] >= 245 && center[2] <= 5,
            "unexpected decoded RGB color for solid green JPEG: {center:?}"
        );

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[cfg(feature = "nvenc")]
    #[test]
    fn decodes_cmyk_and_ycck_jpegs_via_rgb_for_opaque_rgba_hint()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = create_temp_dir("cmyk-ycck")?;
        let mut cmyk_pixels = Vec::with_capacity(16 * 16 * 4);
        for y in 0_u8..16 {
            for x in 0_u8..16 {
                cmyk_pixels.extend_from_slice(&[
                    x.saturating_mul(16),
                    y.saturating_mul(16),
                    (x ^ y).saturating_mul(16),
                    (x + y).saturating_mul(4),
                ]);
            }
        }

        for (name, color_type) in [
            ("cmyk.jpg", JpegColorType::Cmyk),
            ("ycck.jpg", JpegColorType::CmykAsYcck),
        ] {
            let src = root.join(name);
            let encoder = JpegEncoder::new_file(&src, 90)?;
            encoder.encode(&cmyk_pixels, 16, 16, color_type)?;

            let item = ImageBatchItem::new(
                src,
                root.join(format!("{name}.heic")),
                ImageInputFormat::Jpeg,
                ImageDimensions::new(16, 16)?,
            )?;
            let mut file_bytes = Vec::new();
            let mut native_buf = Vec::new();
            let native_meta = decode_image_into(
                &item,
                DecodeTargetHint::Native,
                &mut file_bytes,
                &mut native_buf,
            )?;
            let mut hint_buf = Vec::new();
            let hint_meta = decode_image_into(
                &item,
                DecodeTargetHint::OpaqueRgba8,
                &mut file_bytes,
                &mut hint_buf,
            )?;

            assert_eq!(native_meta.format, DecodedPixelFormat::Rgb8);
            assert_eq!(hint_meta.format, DecodedPixelFormat::Rgb8);
            assert_eq!(hint_meta.byte_len, 16 * 16 * 3);
            assert_eq!(
                hint_buf.get(..hint_meta.byte_len),
                native_buf.get(..native_meta.byte_len)
            );
        }

        fs::remove_dir_all(&root)?;
        Ok(())
    }
}
