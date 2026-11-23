use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, Read};

use fastconv_domain::ImageBatchItem;

use crate::PipelineError;

/// Decoded 8-bit PNG pixel format supported across CPU and GPU image pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecodedPixelFormat {
    L8,
    La8,
    Rgb8,
    Rgba8,
}

pub(crate) struct DecodedPng {
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

/// Reads and decodes a PNG file into `pixel_buf`, reusing `file_bytes` as scratch space.
pub(crate) fn decode_png_into(
    item: &ImageBatchItem,
    file_bytes: &mut Vec<u8>,
    pixel_buf: &mut Vec<u8>,
) -> Result<DecodedPng, PipelineError> {
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

    let mut decoder = png::Decoder::new(io::Cursor::new(file_bytes.as_slice()));
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

    let expected_dims = item.dimensions();
    if frame.width != expected_dims.width().get() || frame.height != expected_dims.height().get() {
        return Err(PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: format!(
                "decoded PNG dimensions {}x{} do not match probed {}",
                frame.width, frame.height, expected_dims
            ),
        });
    }

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

    Ok(DecodedPng {
        width: frame.width,
        height: frame.height,
        format,
        byte_len,
        input_file_bytes,
    })
}
