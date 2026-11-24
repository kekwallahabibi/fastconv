use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, Read};

use fastconv_domain::{ImageBatchItem, ImageInputFormat};
use jxl::api::{
    Event, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
    JxlPixelFormat,
};
use tiff::decoder::Decoder as TiffDecoder;
use tiff::tags::Tag;
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

use crate::PipelineError;
use crate::avif::decode_avif_slice_into;
use crate::bmp::decode_bmp_slice_into;
use crate::gif::decode_gif_slice_into;
use crate::ico::decode_ico_slice_into;
use crate::pnm::decode_pnm_slice_into;
use crate::qoi::decode_qoi_slice_into;

// 512 MiB decoded RGBA8, matching the probe budget for hostile headers.
const MAX_TIFF_DECODED_BYTES: usize = 512 * 1024 * 1024;

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

/// Only NVENC inspects native alpha before 8-bit scaling.
const fn needs_native_alpha_scan(hint: DecodeTargetHint) -> bool {
    #[cfg(feature = "nvenc")]
    {
        matches!(hint, DecodeTargetHint::OpaqueRgba8)
    }
    #[cfg(not(feature = "nvenc"))]
    {
        let _ = hint;
        false
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DecodedFrameMeta {
    pub width: u32,
    pub height: u32,
    pub format: DecodedPixelFormat,
    pub byte_len: usize,
    pub input_file_bytes: u64,
    // Only the NVENC path needs to inspect native alpha before 8-bit scaling.
    #[cfg_attr(not(feature = "nvenc"), allow(dead_code))]
    pub has_non_opaque_native_alpha: bool,
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
#[allow(clippy::too_many_lines)] // One explicit decoding arm per validated input container.
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
                has_non_opaque_native_alpha: false,
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
                has_non_opaque_native_alpha: false,
            })
        }
        ImageInputFormat::Tiff => decode_tiff_slice_into(
            item,
            file_bytes.as_slice(),
            input_file_bytes,
            pixel_buf,
            hint,
        ),
        ImageInputFormat::Jxl => {
            decode_jxl_slice_into(item, file_bytes.as_slice(), input_file_bytes, pixel_buf)
        }
        ImageInputFormat::Avif => {
            let (width, height, format, byte_len) =
                decode_avif_slice_into(item.source(), file_bytes.as_slice(), pixel_buf)?;
            verify_decoded_dimensions(item, width, height)?;
            Ok(DecodedFrameMeta {
                width,
                height,
                format,
                byte_len,
                input_file_bytes,
                has_non_opaque_native_alpha: false,
            })
        }
        ImageInputFormat::Gif => {
            let (width, height, format, byte_len) =
                decode_gif_slice_into(item.source(), file_bytes.as_slice(), pixel_buf)?;
            verify_decoded_dimensions(item, width, height)?;
            Ok(DecodedFrameMeta {
                width,
                height,
                format,
                byte_len,
                input_file_bytes,
                has_non_opaque_native_alpha: false,
            })
        }
        ImageInputFormat::Ico => {
            let (width, height, format, byte_len) =
                decode_ico_slice_into(item.source(), file_bytes.as_slice(), pixel_buf)?;
            verify_decoded_dimensions(item, width, height)?;
            Ok(DecodedFrameMeta {
                width,
                height,
                format,
                byte_len,
                input_file_bytes,
                has_non_opaque_native_alpha: false,
            })
        }
        ImageInputFormat::Pnm => {
            let (width, height, format, byte_len) =
                decode_pnm_slice_into(item.source(), file_bytes.as_slice(), pixel_buf)?;
            verify_decoded_dimensions(item, width, height)?;
            Ok(DecodedFrameMeta {
                width,
                height,
                format,
                byte_len,
                input_file_bytes,
                has_non_opaque_native_alpha: false,
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
        zenwebp::WebPDecoder::new(file_bytes).map_err(|err| PipelineError::DecodeFailed {
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
        has_non_opaque_native_alpha: false,
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
        has_non_opaque_native_alpha: false,
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
        has_non_opaque_native_alpha: false,
    })
}

fn unpremultiply_native_u16(samples: &mut [u16], color_samples: usize, max_sample: u32) {
    let stride = color_samples + 1;
    for pixel in samples.chunks_exact_mut(stride) {
        let alpha = u32::from(pixel[color_samples]);
        if alpha == 0 {
            for sample in &mut pixel[..color_samples] {
                *sample = 0;
            }
        } else {
            for sample in &mut pixel[..color_samples] {
                *sample = u32::from(*sample)
                    .checked_mul(max_sample)
                    .and_then(|value| value.checked_div(alpha))
                    .map_or(0, |value| {
                        u16::try_from(value.min(max_sample)).unwrap_or(u16::MAX)
                    });
            }
        }
    }
}

fn unpremultiply_native_u8(samples: &mut [u8], color_samples: usize) {
    let stride = color_samples + 1;
    for pixel in samples.chunks_exact_mut(stride) {
        let alpha = u32::from(pixel[color_samples]);
        if alpha == 0 {
            for sample in &mut pixel[..color_samples] {
                *sample = 0;
            }
        } else {
            for sample in &mut pixel[..color_samples] {
                *sample = u32::from(*sample)
                    .checked_mul(255)
                    .and_then(|value| value.checked_div(alpha))
                    .map_or(0, |value| u8::try_from(value.min(255)).unwrap_or(u8::MAX));
            }
        }
    }
}

#[allow(clippy::too_many_lines)] // TIFF decoding validates headers and converts all supported sample widths in one cohesive path.
fn decode_tiff_slice_into(
    item: &ImageBatchItem,
    file_bytes: &[u8],
    input_file_bytes: u64,
    pixel_buf: &mut Vec<u8>,
    hint: DecodeTargetHint,
) -> Result<DecodedFrameMeta, PipelineError> {
    let mut decoder = TiffDecoder::new(io::Cursor::new(file_bytes)).map_err(|err| {
        PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: format!("invalid TIFF stream: {err}"),
        }
    })?;

    let (width, height) = decoder
        .dimensions()
        .map_err(|err| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: format!("invalid TIFF dimensions: {err}"),
        })?;
    verify_decoded_dimensions(item, width, height)?;

    let color = decoder
        .colortype()
        .map_err(|err| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: format!("invalid TIFF color type: {err}"),
        })?;

    let (channels, format, bits) = match color {
        tiff::ColorType::Gray(bits) => (1_usize, DecodedPixelFormat::L8, bits),
        tiff::ColorType::GrayA(bits) => (2_usize, DecodedPixelFormat::La8, bits),
        tiff::ColorType::RGB(bits) => (3_usize, DecodedPixelFormat::Rgb8, bits),
        tiff::ColorType::RGBA(bits) => (4_usize, DecodedPixelFormat::Rgba8, bits),
        other => {
            return Err(PipelineError::UnsupportedPixelFormat {
                path: item.source().to_path_buf(),
                format: format!("TIFF color type {other:?}"),
            });
        }
    };
    if bits == 0 || bits > 16 {
        return Err(PipelineError::UnsupportedPixelFormat {
            path: item.source().to_path_buf(),
            format: format!("TIFF sample depth {bits}"),
        });
    }

    let expected_len = usize::try_from(width)
        .ok()
        .and_then(|w| {
            usize::try_from(height)
                .ok()
                .and_then(|h| w.checked_mul(h).and_then(|p| p.checked_mul(channels)))
        })
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: "TIFF dimensions overflow decoded buffer size".to_owned(),
        })?;
    if expected_len > MAX_TIFF_DECODED_BYTES {
        return Err(PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: format!(
                "TIFF decoded output ({expected_len} bytes) exceeds the {MAX_TIFF_DECODED_BYTES}-byte budget"
            ),
        });
    }
    let layout = decoder
        .image_buffer_layout()
        .map_err(|err| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: format!("invalid TIFF buffer layout: {err}"),
        })?;
    if layout.complete_len > MAX_TIFF_DECODED_BYTES {
        return Err(PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: format!(
                "TIFF decoder buffer ({} bytes) exceeds the {MAX_TIFF_DECODED_BYTES}-byte budget",
                layout.complete_len
            ),
        });
    }
    if pixel_buf.len() < expected_len {
        pixel_buf.resize(expected_len, 0);
    }

    let alpha_associated = matches!(format, DecodedPixelFormat::La8 | DecodedPixelFormat::Rgba8)
        && decoder
            .find_tag_unsigned_vec::<u16>(Tag::ExtraSamples)
            .map_err(|err| PipelineError::DecodeFailed {
                path: item.source().to_path_buf(),
                reason: format!("invalid TIFF ExtraSamples tag: {err}"),
            })?
            .and_then(|values| values.first().copied())
            == Some(1);

    let width_usize = usize::try_from(width).map_err(|_| PipelineError::DecodeFailed {
        path: item.source().to_path_buf(),
        reason: "TIFF width exceeds addressable memory".to_owned(),
    })?;
    let height_usize = usize::try_from(height).map_err(|_| PipelineError::DecodeFailed {
        path: item.source().to_path_buf(),
        reason: "TIFF height exceeds addressable memory".to_owned(),
    })?;
    let fast_u8 = bits == 8
        && layout.planes == 1
        && layout.sample_type == Some(tiff::decoder::DecodingSampleType::U8)
        && layout.complete_len == expected_len
        && layout
            .row_stride
            .is_some_and(|stride| stride.get() == width_usize.saturating_mul(channels));

    if fast_u8 {
        decoder
            .read_image_bytes(&mut pixel_buf[..expected_len])
            .map_err(|err| PipelineError::DecodeFailed {
                path: item.source().to_path_buf(),
                reason: format!("failed to decode TIFF image: {err}"),
            })?;
        if alpha_associated && matches!(format, DecodedPixelFormat::La8 | DecodedPixelFormat::Rgba8)
        {
            let color_samples = if format == DecodedPixelFormat::La8 {
                1
            } else {
                3
            };
            unpremultiply_native_u8(&mut pixel_buf[..expected_len], color_samples);
        }
        let has_non_opaque_native_alpha = needs_native_alpha_scan(hint)
            && matches!(format, DecodedPixelFormat::La8 | DecodedPixelFormat::Rgba8)
            && {
                let alpha_stride = if format == DecodedPixelFormat::La8 {
                    2
                } else {
                    4
                };
                pixel_buf[..expected_len]
                    .iter()
                    .skip(alpha_stride - 1)
                    .step_by(alpha_stride)
                    .any(|&alpha| alpha < u8::MAX)
            };
        return Ok(DecodedFrameMeta {
            width,
            height,
            format,
            byte_len: expected_len,
            input_file_bytes,
            has_non_opaque_native_alpha,
        });
    }

    let result = decoder
        .read_image()
        .map_err(|err| PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: format!("failed to decode TIFF image: {err}"),
        })?;

    let max_sample = (1_u32 << bits) - 1;
    let scale_u8 = |value: u32| -> u8 {
        u8::try_from((value * 255 + max_sample / 2) / max_sample).unwrap_or(u8::MAX)
    };

    let mut has_non_opaque_native_alpha = false;

    match result {
        tiff::decoder::DecodingResult::U8(mut samples) => {
            if bits < 8 && format == DecodedPixelFormat::L8 {
                let row_bytes = width_usize
                    .checked_mul(usize::from(bits))
                    .map(|bits| bits.div_ceil(8))
                    .ok_or_else(|| PipelineError::DecodeFailed {
                        path: item.source().to_path_buf(),
                        reason: "TIFF row bit count overflow".to_owned(),
                    })?;
                let needed = row_bytes.checked_mul(height_usize).ok_or_else(|| {
                    PipelineError::DecodeFailed {
                        path: item.source().to_path_buf(),
                        reason: "TIFF packed buffer size overflow".to_owned(),
                    }
                })?;
                if samples.len() < needed {
                    return Err(PipelineError::DecodeFailed {
                        path: item.source().to_path_buf(),
                        reason: format!(
                            "decoded TIFF buffer is too short: {} < {needed}",
                            samples.len()
                        ),
                    });
                }
                for y in 0..height_usize {
                    for x in 0..width_usize {
                        let bit_offset = x * usize::from(bits);
                        let byte_index = bit_offset / 8;
                        let intra_byte = bit_offset % 8;
                        let row_start = y * row_bytes;
                        let window = (u16::from(samples[row_start + byte_index]) << 8)
                            | u16::from(
                                samples
                                    .get(row_start + byte_index + 1)
                                    .copied()
                                    .unwrap_or(0),
                            );
                        let shift = 16_usize - usize::from(bits) - intra_byte;
                        let mask = (1_u16 << bits) - 1;
                        let value = u8::try_from((window >> shift) & mask).unwrap_or(0);
                        pixel_buf[y * width_usize + x] = scale_u8(u32::from(value));
                    }
                }
            } else {
                if samples.len() < expected_len {
                    return Err(PipelineError::DecodeFailed {
                        path: item.source().to_path_buf(),
                        reason: format!(
                            "decoded TIFF buffer is too short: {} < {expected_len}",
                            samples.len()
                        ),
                    });
                }
                if bits == 8 {
                    if alpha_associated
                        && matches!(format, DecodedPixelFormat::La8 | DecodedPixelFormat::Rgba8)
                    {
                        let color_samples = if format == DecodedPixelFormat::La8 {
                            1
                        } else {
                            3
                        };
                        unpremultiply_native_u8(&mut samples[..expected_len], color_samples);
                    }
                    pixel_buf[..expected_len].copy_from_slice(&samples[..expected_len]);
                    if needs_native_alpha_scan(hint)
                        && matches!(format, DecodedPixelFormat::La8 | DecodedPixelFormat::Rgba8)
                    {
                        let alpha_stride = if format == DecodedPixelFormat::La8 {
                            2
                        } else {
                            4
                        };
                        has_non_opaque_native_alpha = samples[..expected_len]
                            .iter()
                            .skip(alpha_stride - 1)
                            .step_by(alpha_stride)
                            .any(|&alpha| u32::from(alpha) < max_sample);
                    }
                } else {
                    for (dst, src) in pixel_buf[..expected_len]
                        .iter_mut()
                        .zip(samples[..expected_len].iter().copied())
                    {
                        *dst = scale_u8(u32::from(src));
                    }
                }
            }
        }
        tiff::decoder::DecodingResult::U16(mut samples) => {
            if samples.len() < expected_len {
                return Err(PipelineError::DecodeFailed {
                    path: item.source().to_path_buf(),
                    reason: format!(
                        "decoded TIFF buffer is too short: {} < {expected_len}",
                        samples.len()
                    ),
                });
            }
            if needs_native_alpha_scan(hint)
                && matches!(format, DecodedPixelFormat::La8 | DecodedPixelFormat::Rgba8)
            {
                let alpha_stride = if format == DecodedPixelFormat::La8 {
                    2
                } else {
                    4
                };
                has_non_opaque_native_alpha = samples[..expected_len]
                    .iter()
                    .skip(alpha_stride - 1)
                    .step_by(alpha_stride)
                    .any(|&alpha| u32::from(alpha) < max_sample);
            }
            if alpha_associated
                && matches!(format, DecodedPixelFormat::La8 | DecodedPixelFormat::Rgba8)
            {
                let color_samples = if format == DecodedPixelFormat::La8 {
                    1
                } else {
                    3
                };
                unpremultiply_native_u16(&mut samples[..expected_len], color_samples, max_sample);
            }
            for (dst, src) in pixel_buf[..expected_len]
                .iter_mut()
                .zip(samples[..expected_len].iter().copied())
            {
                *dst = scale_u8(u32::from(src));
            }
        }
        other => {
            return Err(PipelineError::UnsupportedPixelFormat {
                path: item.source().to_path_buf(),
                format: format!("TIFF decoded sample type {other:?}"),
            });
        }
    }

    Ok(DecodedFrameMeta {
        width,
        height,
        format,
        byte_len: expected_len,
        input_file_bytes,
        has_non_opaque_native_alpha,
    })
}

#[allow(clippy::too_many_lines)] // JPEG XL decoding drives the full libjxl event loop and output-buffer setup.
fn decode_jxl_slice_into(
    item: &ImageBatchItem,
    file_bytes: &[u8],
    input_file_bytes: u64,
    pixel_buf: &mut Vec<u8>,
) -> Result<DecodedFrameMeta, PipelineError> {
    let mut options = JxlDecoderOptions::default();
    options.sample_limit = Some(512 * 1024 * 1024);
    options.alpha_mode = jxl::api::AlphaMode::Unpremultiplied;
    let mut decoder = JxlDecoder::new(options);

    let mut input = file_bytes;
    let mut buffers: Option<[JxlOutputBuffer<'_>; 1]> = None;
    let mut width = 0_u32;
    let mut height = 0_u32;
    let mut channels = 0_usize;
    let mut format = DecodedPixelFormat::Rgba8;

    loop {
        let out = buffers.as_mut().map(|bufs| &mut bufs[..]);
        let event =
            decoder
                .process(&mut input, out, None)
                .map_err(|err| PipelineError::DecodeFailed {
                    path: item.source().to_path_buf(),
                    reason: format!("invalid JPEG XL stream: {err}"),
                })?;

        match event {
            Event::BasicInfo => {
                let info = decoder
                    .basic_info()
                    .ok_or_else(|| PipelineError::DecodeFailed {
                        path: item.source().to_path_buf(),
                        reason: "missing JPEG XL basic image information".to_owned(),
                    })?;
                let (w, h) = info.size;
                width = u32::try_from(w).map_err(|_| PipelineError::DecodeFailed {
                    path: item.source().to_path_buf(),
                    reason: format!("JPEG XL width {w} is too large"),
                })?;
                height = u32::try_from(h).map_err(|_| PipelineError::DecodeFailed {
                    path: item.source().to_path_buf(),
                    reason: format!("JPEG XL height {h} is too large"),
                })?;
                verify_decoded_dimensions(item, width, height)?;
                if info.animation.is_some() {
                    return Err(PipelineError::DecodeFailed {
                        path: item.source().to_path_buf(),
                        reason: "animated JPEG XL is not supported by the still-image pipeline"
                            .to_owned(),
                    });
                }

                let grayscale = decoder
                    .embedded_color_profile()
                    .is_some_and(|profile| profile.channels() == 1);
                let has_alpha = info
                    .extra_channels
                    .iter()
                    .any(|channel| channel.ec_type == jxl::api::ExtraChannel::Alpha)
                    || decoder
                        .current_pixel_format()
                        .is_some_and(|format| format.color_type.has_alpha());

                let color_type = match (grayscale, has_alpha) {
                    (true, true) => JxlColorType::GrayscaleAlpha,
                    (true, false) => JxlColorType::Grayscale,
                    (false, true) => JxlColorType::Rgba,
                    (false, false) => JxlColorType::Rgb,
                };
                channels = color_type.samples_per_pixel();
                format = if color_type == JxlColorType::Grayscale {
                    DecodedPixelFormat::L8
                } else if color_type == JxlColorType::GrayscaleAlpha {
                    DecodedPixelFormat::La8
                } else if color_type == JxlColorType::Rgb {
                    DecodedPixelFormat::Rgb8
                } else {
                    DecodedPixelFormat::Rgba8
                };

                let required = channels
                    .checked_mul(width as usize)
                    .and_then(|row| row.checked_mul(height as usize))
                    .ok_or_else(|| PipelineError::DecodeFailed {
                        path: item.source().to_path_buf(),
                        reason: "JPEG XL dimensions overflow decoded buffer size".to_owned(),
                    })?;
                if pixel_buf.len() < required {
                    pixel_buf.resize(required, 0);
                }

                let pixel_format = JxlPixelFormat {
                    color_type,
                    color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
                    extra_channel_format: vec![None; info.extra_channels.len()],
                };
                decoder.set_pixel_format(pixel_format).map_err(|err| {
                    PipelineError::DecodeFailed {
                        path: item.source().to_path_buf(),
                        reason: format!("failed to configure JPEG XL output format: {err}"),
                    }
                })?;

                let buffer = JxlOutputBuffer::new(
                    pixel_buf.as_mut_slice(),
                    height as usize,
                    width as usize * channels,
                );
                buffers = Some([buffer]);
            }
            Event::FrameHeader => {}
            Event::FrameComplete { has_more_frames } => {
                if has_more_frames {
                    return Err(PipelineError::DecodeFailed {
                        path: item.source().to_path_buf(),
                        reason: "animated JPEG XL is not supported by the still-image pipeline"
                            .to_owned(),
                    });
                }
            }
            Event::Complete => break,
            Event::NeedMoreInput { .. } => {
                if input.is_empty() {
                    return Err(PipelineError::DecodeFailed {
                        path: item.source().to_path_buf(),
                        reason: "truncated JPEG XL stream".to_owned(),
                    });
                }
            }
        }
    }

    if channels == 0 {
        return Err(PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: "JPEG XL decoder produced no image data".to_owned(),
        });
    }

    let byte_len = channels * width as usize * height as usize;
    Ok(DecodedFrameMeta {
        width,
        height,
        format,
        byte_len,
        input_file_bytes,
        has_non_opaque_native_alpha: false,
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
