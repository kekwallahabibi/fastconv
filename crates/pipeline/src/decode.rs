use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, Read};

use fastconv_domain::{ImageBatchItem, ImageInputFormat};
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

use crate::PipelineError;

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

/// Reads and decodes a validated PNG or JPEG item into `pixel_buf`, reusing `file_bytes`.
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
    }
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
        // `zune-jpeg` leaves RGB->RGBA unimplemented; fall back to RGB so the caller appends alpha.
        #[cfg(feature = "nvenc")]
        (DecodeTargetHint::OpaqueRgba8, Some(ColorSpace::RGB)) => {
            (ColorSpace::RGB, DecodedPixelFormat::Rgb8)
        }
        #[cfg(feature = "nvenc")]
        (
            DecodeTargetHint::OpaqueRgba8,
            Some(ColorSpace::Luma | ColorSpace::YCbCr | ColorSpace::CMYK | ColorSpace::YCCK),
        ) => (ColorSpace::RGBA, DecodedPixelFormat::Rgba8),
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
