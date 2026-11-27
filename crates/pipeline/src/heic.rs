//! Pure-Rust CPU HEIC still-image encoding and decoding via `heifer`.
//!
//! Validated decoded pixels are handed straight to `heifer`'s HEVC-Intra encoder:
//! RGB8/RGBA8 are passed through untouched, while L8/La8 are expanded once into
//! the reusable worker scratch buffer. The encoder keeps its default `Medium`
//! speed/quality trade-off (no preset cheating); multi-core scaling comes from
//! the batch workers plus `heifer`'s own 512-pixel tile threading.
//!
//! Decoding is bounded to one HEVC decoder thread (the batch already parallelizes
//! across images) and to a fixed pixel budget. Opaque NVENC targets receive the
//! decoder's RGBA buffer directly, avoiding a lossy RGB -> RGBA expansion.

use std::path::Path;

use fastconv_domain::ImageQuality;
use heifer::EncodeOptions;
use heifer::heifer_isobmff::boxes::FileType;

use crate::PipelineError;
use crate::decode::{DecodeTargetHint, DecodedFrameMeta, DecodedPixelFormat};

// Half of heifer's default 2^28-pixel limit, matching the probe budget.
const MAX_HEIC_DECODE_PIXELS: u64 = 128 * 1024 * 1024;

const fn pixel_channels(format: DecodedPixelFormat) -> u64 {
    match format {
        DecodedPixelFormat::L8 => 1,
        DecodedPixelFormat::La8 => 2,
        DecodedPixelFormat::Rgb8 => 3,
        DecodedPixelFormat::Rgba8 => 4,
    }
}

fn encode_error(target: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: reason.into(),
    }
}

/// Validates decoder output before handing pixels to `heifer`. Geometry admission
/// (pixel and per-worker expansion budgets) is checked once in `probe` on the media
/// header; here we only verify the decoder produced a buffer matching its metadata.
fn validate_input(
    target: &Path,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
) -> Result<(), PipelineError> {
    if decoded.width == 0 || decoded.height == 0 {
        return Err(encode_error(
            target,
            "HEIC encoding requires non-zero image dimensions",
        ));
    }

    let expected = u64::from(decoded.width)
        .checked_mul(u64::from(decoded.height))
        .and_then(|count| count.checked_mul(pixel_channels(decoded.format)))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| encode_error(target, "HEIC pixel buffer size overflow"))?;
    if pixels.len() != expected {
        return Err(PipelineError::DecodeFailed {
            path: target.to_path_buf(),
            reason: format!(
                "HEIC decoded pixel buffer length {} does not match expected {expected}",
                pixels.len()
            ),
        });
    }
    Ok(())
}

/// Expands grayscale into RGB in `aux_buf`, reusing the allocation across images.
fn expand_l8_to_rgb(
    target: &Path,
    pixels: &[u8],
    aux_buf: &mut Vec<u8>,
) -> Result<usize, PipelineError> {
    let needed = pixels
        .len()
        .checked_mul(3)
        .ok_or_else(|| encode_error(target, "HEIC grayscale expansion size overflow"))?;
    aux_buf.clear();
    aux_buf
        .try_reserve_exact(needed)
        .map_err(|_| encode_error(target, "cannot allocate HEIC grayscale expansion buffer"))?;
    aux_buf.resize(needed, 0);
    let (triples, _) = aux_buf[..needed].as_chunks_mut::<3>();
    for (&value, out) in pixels.iter().zip(triples.iter_mut()) {
        out.copy_from_slice(&[value, value, value]);
    }
    Ok(needed)
}

/// Expands grayscale-alpha into RGBA in `aux_buf`, reusing the allocation across images.
fn expand_la8_to_rgba(
    target: &Path,
    pixels: &[u8],
    aux_buf: &mut Vec<u8>,
) -> Result<usize, PipelineError> {
    let (pairs, remainder) = pixels.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(PipelineError::DecodeFailed {
            path: target.to_path_buf(),
            reason: "invalid grayscale-alpha pixel buffer length for HEIC".to_owned(),
        });
    }
    let needed = pairs
        .len()
        .checked_mul(4)
        .ok_or_else(|| encode_error(target, "HEIC grayscale-alpha expansion size overflow"))?;
    aux_buf.clear();
    aux_buf
        .try_reserve_exact(needed)
        .map_err(|_| encode_error(target, "cannot allocate HEIC grayscale-alpha buffer"))?;
    aux_buf.resize(needed, 0);
    let (quads, _) = aux_buf[..needed].as_chunks_mut::<4>();
    for (&[value, alpha], out) in pairs.iter().zip(quads.iter_mut()) {
        out.copy_from_slice(&[value, value, value, alpha]);
    }
    Ok(needed)
}

/// Encodes one decoded image into a HEIC file. The returned buffer is non-empty
/// and starts with the ISOBMFF `ftypheic` box written by `heifer`.
///
/// # Errors
///
/// Returns [`PipelineError`] for invalid dimensions/overflow, a pixel buffer
/// length mismatch, an allocation failure, or an `heifer` encoder failure.
pub(crate) fn encode_heic_into(
    target: &Path,
    quality: ImageQuality,
    threads: usize,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    aux_buf: &mut Vec<u8>,
) -> Result<Vec<u8>, PipelineError> {
    validate_input(target, pixels, decoded)?;

    // `0` means "all cores" to heifer and would oversubscribe the batch workers;
    // the batch scheduler always supplies a bounded, non-zero inner thread count.
    let options = EncodeOptions::new()
        .with_quality(quality.get())
        .with_max_threads(threads.max(1));
    // Intentionally keep `EncodeSpeed::Medium`: a faster preset would trade away
    // compression/quality fairness. Parallelism comes from workers + HEVC tiles.

    let encoded = match decoded.format {
        DecodedPixelFormat::Rgb8 => {
            heifer::encode_rgb8(decoded.width, decoded.height, pixels, &options)
        }
        // `encode_rgba8` omits the auxiliary alpha image when every pixel is opaque
        // and writes the standard `auxl` monochrome alpha item when transparency exists.
        DecodedPixelFormat::Rgba8 => {
            heifer::encode_rgba8(decoded.width, decoded.height, pixels, &options)
        }
        DecodedPixelFormat::L8 => {
            let len = expand_l8_to_rgb(target, pixels, aux_buf)?;
            heifer::encode_rgb8(decoded.width, decoded.height, &aux_buf[..len], &options)
        }
        DecodedPixelFormat::La8 => {
            let len = expand_la8_to_rgba(target, pixels, aux_buf)?;
            heifer::encode_rgba8(decoded.width, decoded.height, &aux_buf[..len], &options)
        }
    }
    .map_err(|err| encode_error(target, format!("HEVC encoding failed: {err}")))?;

    if encoded.is_empty() {
        return Err(PipelineError::EmptyOutput(target.to_path_buf()));
    }
    Ok(encoded)
}

fn decode_error(path: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::DecodeFailed {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

/// Sequence brands are a cheap, explicit rejection before any HEVC work. A file
/// without them can still carry a picture/video track, so the track table is
/// inspected as well.
fn reject_sequence(source: &Path, file_bytes: &[u8]) -> Result<(), PipelineError> {
    let header = heifer::heifer_isobmff::read_box_header(file_bytes)
        .map_err(|err| decode_error(source, format!("invalid HEIC box header: {err}")))?;
    if header.box_type.0 == *b"ftyp" {
        let start = usize::try_from(header.header_size)
            .map_err(|_| decode_error(source, "HEIC ftyp header size overflow"))?;
        let content = match header.content_size() {
            Some(len) => {
                let len = usize::try_from(len)
                    .map_err(|_| decode_error(source, "HEIC ftyp size overflow"))?;
                let end = start
                    .checked_add(len)
                    .ok_or_else(|| decode_error(source, "HEIC ftyp size overflow"))?;
                file_bytes
                    .get(start..end)
                    .ok_or_else(|| decode_error(source, "truncated HEIC ftyp box"))?
            }
            None => file_bytes
                .get(start..)
                .ok_or_else(|| decode_error(source, "truncated HEIC ftyp box"))?,
        };
        let file_type = FileType::parse(content)
            .map_err(|err| decode_error(source, format!("invalid HEIC ftyp box: {err}")))?;
        if [*b"msf1", *b"hevc", *b"hevx"]
            .iter()
            .any(|brand| file_type.has_brand(brand))
        {
            return Err(decode_error(
                source,
                "animated/sequence HEIC is not supported by the still-image pipeline",
            ));
        }
    }
    let sequence = heifer::read_sequence(file_bytes)
        .map_err(|err| decode_error(source, format!("invalid HEIC track table: {err}")))?;
    if sequence.is_some_and(|info| info.frames.len() > 1) {
        return Err(decode_error(
            source,
            "animated/sequence HEIC is not supported by the still-image pipeline",
        ));
    }
    Ok(())
}

/// Resizes `pixel_buf` with a checked reservation so a hostile header cannot
/// force an unchecked multi-hundred-megabyte allocation.
fn reserve_pixel_buf(
    source: &Path,
    pixel_buf: &mut Vec<u8>,
    len: usize,
) -> Result<(), PipelineError> {
    pixel_buf.clear();
    pixel_buf
        .try_reserve_exact(len)
        .map_err(|_| decode_error(source, "cannot allocate HEIC decoded pixel buffer"))?;
    pixel_buf.resize(len, 0);
    Ok(())
}

/// Decodes a validated HEIC still image into `pixel_buf` with `heifer`'s pure-Rust
/// decoder. Returns the exact decoded dimensions, native pixel format and byte
/// length written into the buffer.
pub(crate) fn decode_heic_slice_into(
    source: &Path,
    file_bytes: &[u8],
    hint: DecodeTargetHint,
    pixel_buf: &mut Vec<u8>,
) -> Result<(u32, u32, DecodedPixelFormat, usize), PipelineError> {
    reject_sequence(source, file_bytes)?;
    let options = heifer::Options::new()
        .with_max_threads(1)
        .with_max_pixels(MAX_HEIC_DECODE_PIXELS);
    let image = heifer::decode_rgba8(file_bytes, &options)
        .map_err(|err| decode_error(source, format!("HEIC decode failed: {err}")))?;
    let (width, height) = (image.width, image.height);
    if width == 0 || height == 0 {
        return Err(decode_error(
            source,
            format!("HEIC decoded zero dimension {width}x{height}"),
        ));
    }
    let count = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| decode_error(source, "HEIC decoded dimensions overflow pixel count"))?;
    if count > MAX_HEIC_DECODE_PIXELS {
        return Err(decode_error(
            source,
            format!(
                "HEIC decoded pixel count ({count}) exceeds {MAX_HEIC_DECODE_PIXELS}-pixel limit"
            ),
        ));
    }
    let expected = usize::try_from(count)
        .ok()
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| decode_error(source, "HEIC decoded buffer size overflow"))?;
    if image.data.len() != expected {
        return Err(decode_error(
            source,
            format!(
                "HEIC decoded buffer length {} does not match expected {expected}",
                image.data.len()
            ),
        ));
    }
    let pixels = usize::try_from(count)
        .map_err(|_| decode_error(source, "HEIC decoded pixel count overflow"))?;

    match hint {
        DecodeTargetHint::Native => {
            if image.has_alpha {
                reserve_pixel_buf(source, pixel_buf, expected)?;
                pixel_buf.copy_from_slice(&image.data);
                Ok((width, height, DecodedPixelFormat::Rgba8, expected))
            } else {
                let rgb_len = pixels
                    .checked_mul(3)
                    .ok_or_else(|| decode_error(source, "HEIC RGB buffer size overflow"))?;
                reserve_pixel_buf(source, pixel_buf, rgb_len)?;
                for (out, pixel) in pixel_buf
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(image.data.as_chunks::<4>().0)
                {
                    out.copy_from_slice(&pixel[..3]);
                }
                Ok((width, height, DecodedPixelFormat::Rgb8, rgb_len))
            }
        }
        #[cfg(feature = "nvenc")]
        DecodeTargetHint::OpaqueRgba8 => {
            if image.has_alpha
                && image
                    .data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|pixel| pixel[3] < 255)
            {
                return Err(PipelineError::UnsupportedPixelFormat {
                    path: source.to_path_buf(),
                    format: "non-opaque alpha channel (NVENC HEVC-Intra HEIC currently supports opaque images)".to_owned(),
                });
            }
            reserve_pixel_buf(source, pixel_buf, expected)?;
            pixel_buf.copy_from_slice(&image.data);
            Ok((width, height, DecodedPixelFormat::Rgba8, expected))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_heic_slice_into, encode_heic_into};
    use crate::decode::{
        DecodeTargetHint, DecodedFrameMeta, DecodedPixelFormat, decode_image_into,
    };
    use crate::{PipelineError, execute_image_batch};
    use fastconv_domain::{
        ImageBatchItem, ImageBatchPlan, ImageDimensions, ImageExecutionMode, ImageInputFormat,
        ImageOutputFormat, ImageQuality,
    };
    use std::env;
    use std::fs;
    use std::num::NonZeroUsize;
    use std::path::{Path, PathBuf};
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};

    const TARGET: &str = "test.heic";

    fn meta(width: u32, height: u32, format: DecodedPixelFormat, len: usize) -> DecodedFrameMeta {
        DecodedFrameMeta {
            width,
            height,
            format,
            byte_len: len,
            input_file_bytes: 0,
            has_non_opaque_native_alpha: false,
        }
    }

    fn gradient_rgb(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(usize::try_from(width * height * 3).unwrap_or(0));
        for y in 0..height {
            for x in 0..width {
                // Smooth chroma: 4:2:0 HEVC-Intra is meant for photo-like content,
                // not per-pixel chroma noise.
                pixels.extend_from_slice(&[
                    u8::try_from((x * 4) % 256).unwrap_or(0),
                    u8::try_from((y * 4) % 256).unwrap_or(0),
                    u8::try_from(((x + y) * 2) % 256).unwrap_or(0),
                ]);
            }
        }
        pixels
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        if a.is_empty() {
            return f64::INFINITY;
        }
        let mut squared_error = 0.0_f64;
        for (&x, &y) in a.iter().zip(b) {
            let delta = f64::from(x.abs_diff(y));
            squared_error += delta * delta;
        }
        if squared_error == 0.0 {
            return f64::INFINITY;
        }
        let samples = f64::from(u32::try_from(a.len()).unwrap_or(u32::MAX));
        10.0 * (255.0_f64 * 255.0 / (squared_error / samples)).log10()
    }

    fn encode(
        pixels: &[u8],
        decoded: DecodedFrameMeta,
        quality: u8,
        aux: &mut Vec<u8>,
    ) -> Result<Vec<u8>, PipelineError> {
        encode_heic_into(
            Path::new(TARGET),
            ImageQuality::new(quality).map_err(|_| PipelineError::WorkerPanicked)?,
            2,
            pixels,
            decoded,
            aux,
        )
    }

    fn expect_heic_boxes(bytes: &[u8]) {
        assert_eq!(
            bytes.get(4..12),
            Some(b"ftypheic".as_slice()),
            "HEIC major brand"
        );
        assert!(
            bytes.windows(4).any(|window| window == b"mif1"),
            "HEIC compatible brand mif1 missing"
        );
    }

    #[test]
    fn rgb_and_opaque_rgba_roundtrip_without_expansion() -> Result<(), PipelineError> {
        let (width, height) = (64_u32, 48_u32);
        let rgb = gradient_rgb(width, height);
        let rgba: Vec<u8> = rgb
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&[r, g, b]| [r, g, b, 255])
            .collect();
        let mut aux = Vec::new();

        for (format, pixels) in [
            (DecodedPixelFormat::Rgb8, rgb.as_slice()),
            (DecodedPixelFormat::Rgba8, rgba.as_slice()),
        ] {
            let bytes = encode(
                pixels,
                meta(width, height, format, pixels.len()),
                90,
                &mut aux,
            )?;
            expect_heic_boxes(&bytes);

            let image = heifer::decode(&bytes).map_err(|err| PipelineError::EncodeFailed {
                path: TARGET.into(),
                reason: err.to_string(),
            })?;
            assert_eq!(
                (image.width, image.height, image.bit_depth),
                (width, height, 8)
            );
            assert!(!image.has_alpha, "opaque input must not gain an alpha item");
            assert!(
                psnr(&image.to_rgb8(), &rgb) > 30.0,
                "RGB fidelity for {format:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn l8_and_la8_expand_and_preserve_alpha() -> Result<(), PipelineError> {
        let (width, height) = (53_u32, 37_u32);
        let luma: Vec<u8> = (0..width * height)
            .map(|index| u8::try_from((index * 11) % 256).unwrap_or(0))
            .collect();
        let mut aux = Vec::new();

        let bytes = encode(
            &luma,
            meta(width, height, DecodedPixelFormat::L8, luma.len()),
            90,
            &mut aux,
        )?;
        expect_heic_boxes(&bytes);
        let image = heifer::decode(&bytes).map_err(|err| PipelineError::EncodeFailed {
            path: TARGET.into(),
            reason: err.to_string(),
        })?;
        assert_eq!((image.width, image.height), (width, height));
        let decoded_rgb = image.to_rgb8();
        let expected_rgb: Vec<u8> = luma.iter().flat_map(|&v| [v, v, v]).collect();
        assert!(
            psnr(&decoded_rgb, &expected_rgb) > 35.0,
            "grayscale expansion diverged"
        );

        // An odd width keeps the LA8 -> RGBA8 expansion from accidentally relying on
        // an even number of source pixels per row.
        let alpha_width = width;
        let la: Vec<u8> = luma
            .iter()
            .enumerate()
            .flat_map(|(index, &value)| {
                let alpha = if index % 5 == 0 { 64 } else { 255 };
                [value, alpha]
            })
            .collect();
        let bytes = encode(
            &la,
            meta(alpha_width, height, DecodedPixelFormat::La8, la.len()),
            90,
            &mut aux,
        )?;
        expect_heic_boxes(&bytes);
        let image = heifer::decode(&bytes).map_err(|err| PipelineError::EncodeFailed {
            path: TARGET.into(),
            reason: err.to_string(),
        })?;
        assert_eq!((image.width, image.height), (alpha_width, height));
        assert!(
            image.has_alpha,
            "transparent LA8 input must write an alpha item"
        );
        let decoded = image.to_rgba8();
        for (index, &[value, alpha]) in la.as_chunks::<2>().0.iter().enumerate() {
            let pixel = &decoded[index * 4..index * 4 + 4];
            assert!(
                pixel[3].abs_diff(alpha) <= 12,
                "alpha at pixel {index}: {} vs {alpha}",
                pixel[3]
            );
            assert!(pixel[0].abs_diff(value) <= 24, "luma at pixel {index}");
        }
        Ok(())
    }

    #[test]
    fn odd_and_tiled_dimensions_with_monotonic_quality() -> Result<(), PipelineError> {
        let mut aux = Vec::new();
        for (width, height) in [(77_u32, 51_u32), (600, 300)] {
            let rgb = gradient_rgb(width, height);
            let mut fidelity = Vec::new();
            for quality in [50, 90] {
                let bytes = encode(
                    &rgb,
                    meta(width, height, DecodedPixelFormat::Rgb8, rgb.len()),
                    quality,
                    &mut aux,
                )?;
                expect_heic_boxes(&bytes);
                let image = heifer::decode(&bytes).map_err(|err| PipelineError::EncodeFailed {
                    path: TARGET.into(),
                    reason: err.to_string(),
                })?;
                assert_eq!((image.width, image.height), (width, height));
                fidelity.push(psnr(&image.to_rgb8(), &rgb));
            }
            assert!(
                fidelity[1] > fidelity[0],
                "Q90 must be more faithful than Q50 at {width}x{height}: {fidelity:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_zero_dimensions_and_mismatched_buffer() {
        let mut aux = Vec::new();
        assert!(matches!(
            encode(&[], meta(0, 8, DecodedPixelFormat::Rgb8, 0), 80, &mut aux,),
            Err(PipelineError::EncodeFailed { .. })
        ));
        assert!(matches!(
            encode(
                &[0_u8; 3],
                meta(8, 8, DecodedPixelFormat::Rgb8, 192),
                80,
                &mut aux,
            ),
            Err(PipelineError::DecodeFailed { .. })
        ));
    }

    static E2E_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn e2e_temp_dir() -> Result<PathBuf, PipelineError> {
        let id = E2E_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("fastconv-heic-e2e-{}-{id}", process::id()));
        fs::create_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(dir)
    }

    fn one_worker() -> Result<NonZeroUsize, PipelineError> {
        NonZeroUsize::new(1).ok_or(PipelineError::WorkerPanicked)
    }

    fn write_heic_source(
        path: &Path,
        width: u32,
        height: u32,
        pixels: &[u8],
        format: DecodedPixelFormat,
    ) -> Result<(), PipelineError> {
        let mut aux = Vec::new();
        let bytes = encode(
            pixels,
            meta(width, height, format, pixels.len()),
            90,
            &mut aux,
        )?;
        fs::write(path, bytes).map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })
    }

    fn convert_heic(
        source: &Path,
        target: &Path,
        format: ImageOutputFormat,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let item = ImageBatchItem::new(
            source.to_path_buf(),
            target.to_path_buf(),
            ImageInputFormat::Heic,
            ImageDimensions::new(width, height)?,
        )?;
        let plan = ImageBatchPlan::new(
            vec![item],
            ImageExecutionMode::resolve_cpu(format, None)?,
            one_worker()?,
        )?;
        let report = execute_image_batch(&plan)?;
        if report.files_converted().get() != 1 {
            return Err("HEIC batch did not convert exactly one file".into());
        }
        Ok(())
    }

    fn decode_output(
        path: &Path,
        format: ImageInputFormat,
        width: u32,
        height: u32,
    ) -> Result<(DecodedFrameMeta, Vec<u8>), Box<dyn std::error::Error>> {
        let item = ImageBatchItem::new(
            path.to_path_buf(),
            path.with_extension("decoded"),
            format,
            ImageDimensions::new(width, height)?,
        )?;
        let mut file_bytes = Vec::new();
        let mut pixels = Vec::new();
        let decoded = decode_image_into(
            &item,
            DecodeTargetHint::Native,
            &mut file_bytes,
            &mut pixels,
        )?;
        Ok((decoded, pixels))
    }

    fn extract_rgb(
        pixels: &[u8],
        format: DecodedPixelFormat,
        count: usize,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        match format {
            DecodedPixelFormat::Rgb8 => Ok(pixels
                .get(..count.saturating_mul(3))
                .ok_or("short RGB8 decode")?
                .to_vec()),
            DecodedPixelFormat::Rgba8 => {
                let data = pixels
                    .get(..count.saturating_mul(4))
                    .ok_or("short RGBA8 decode")?;
                Ok(data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
                    .collect())
            }
            DecodedPixelFormat::L8 | DecodedPixelFormat::La8 => {
                Err("unexpected grayscale output from a colour roundtrip".into())
            }
        }
    }

    fn mean_alpha_diff(
        pixels: &[u8],
        format: DecodedPixelFormat,
        expected: &[u8],
        count: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if format != DecodedPixelFormat::Rgba8 {
            return Ok(0.0);
        }
        let data = pixels
            .get(..count.saturating_mul(4))
            .ok_or("short RGBA8 decode")?;
        let total: u64 = data
            .as_chunks::<4>()
            .0
            .iter()
            .zip(expected.as_chunks::<4>().0)
            .map(|(pixel, expected)| u64::from(pixel[3].abs_diff(expected[3])))
            .sum();
        let total = f64::from(u32::try_from(total).unwrap_or(u32::MAX));
        let samples = f64::from(u32::try_from(count).unwrap_or(u32::MAX));
        Ok(total / samples)
    }

    #[test]
    fn heic_roundtrips_to_every_cpu_still_format() -> Result<(), Box<dyn std::error::Error>> {
        let root = e2e_temp_dir()?;
        let (width, height) = (64_u32, 48_u32);
        let rgb = gradient_rgb(width, height);
        let source = root.join("rgb.heic");
        write_heic_source(&source, width, height, &rgb, DecodedPixelFormat::Rgb8)?;

        // Start from the HEIC decoder itself, then feed it back through the
        // full batch pipeline for every CPU output container.
        let mut file_bytes = Vec::new();
        let mut pixels = Vec::new();
        let item = ImageBatchItem::new(
            source.clone(),
            root.join("unused.out"),
            ImageInputFormat::Heic,
            ImageDimensions::new(width, height)?,
        )?;
        let decoded = decode_image_into(
            &item,
            DecodeTargetHint::Native,
            &mut file_bytes,
            &mut pixels,
        )?;
        assert_eq!((decoded.width, decoded.height), (width, height));
        assert_eq!(decoded.format, DecodedPixelFormat::Rgb8);
        assert!(psnr(&pixels[..decoded.byte_len], &rgb) > 30.0);

        for (index, (ext, output, input)) in [
            ("png", ImageOutputFormat::Png, ImageInputFormat::Png),
            ("webp", ImageOutputFormat::Webp, ImageInputFormat::Webp),
            ("jpg", ImageOutputFormat::Jpeg, ImageInputFormat::Jpeg),
            ("bmp", ImageOutputFormat::Bmp, ImageInputFormat::Bmp),
            ("qoi", ImageOutputFormat::Qoi, ImageInputFormat::Qoi),
            ("tif", ImageOutputFormat::Tiff, ImageInputFormat::Tiff),
            ("heic", ImageOutputFormat::Heic, ImageInputFormat::Heic),
        ]
        .into_iter()
        .enumerate()
        {
            let target = root.join(format!("rgb_{index}.{ext}"));
            convert_heic(&source, &target, output, width, height)?;
            let (meta, decoded_pixels) = decode_output(&target, input, width, height)?;
            assert_eq!((meta.width, meta.height), (width, height), "{ext} size");
            let decoded_rgb = extract_rgb(&decoded_pixels, meta.format, (width * height) as usize)?;
            let fidelity = psnr(&decoded_rgb, &rgb);
            assert!(fidelity > 24.0, "{ext} fidelity was {fidelity} dB");
        }

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn heic_alpha_roundtrips_with_odd_clap_dimensions_and_jpeg_rejects_alpha()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = e2e_temp_dir()?;
        let (width, height) = (37_u32, 29_u32);
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                rgba.extend_from_slice(&[
                    u8::try_from((x * 6) % 256).unwrap_or(0),
                    u8::try_from((y * 8) % 256).unwrap_or(0),
                    u8::try_from(((x + y) * 3) % 256).unwrap_or(0),
                    if (x + y) % 5 == 0 { 96 } else { 255 },
                ]);
            }
        }
        let source = root.join("alpha.heic");
        write_heic_source(&source, width, height, &rgba, DecodedPixelFormat::Rgba8)?;

        for (index, (ext, output, input)) in [
            ("png", ImageOutputFormat::Png, ImageInputFormat::Png),
            ("webp", ImageOutputFormat::Webp, ImageInputFormat::Webp),
            ("qoi", ImageOutputFormat::Qoi, ImageInputFormat::Qoi),
            ("tif", ImageOutputFormat::Tiff, ImageInputFormat::Tiff),
            ("bmp", ImageOutputFormat::Bmp, ImageInputFormat::Bmp),
            ("heic", ImageOutputFormat::Heic, ImageInputFormat::Heic),
        ]
        .into_iter()
        .enumerate()
        {
            let target = root.join(format!("alpha_{index}.{ext}"));
            convert_heic(&source, &target, output, width, height)?;
            let (meta, decoded_pixels) = decode_output(&target, input, width, height)?;
            assert_eq!((meta.width, meta.height), (width, height), "{ext} size");
            assert_eq!(meta.format, DecodedPixelFormat::Rgba8, "{ext} alpha");
            let decoded_rgb = extract_rgb(&decoded_pixels, meta.format, (width * height) as usize)?;
            let expected_rgb = rgba
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
                .collect::<Vec<_>>();
            assert!(psnr(&decoded_rgb, &expected_rgb) > 24.0, "{ext} RGB");
            let alpha_diff = mean_alpha_diff(
                &decoded_pixels,
                meta.format,
                &rgba,
                (width * height) as usize,
            )?;
            assert!(alpha_diff < 20.0, "{ext} alpha drift was {alpha_diff}");
        }

        let jpeg_target = root.join("alpha.jpg");
        assert!(matches!(
            convert_heic(
                &source,
                &jpeg_target,
                ImageOutputFormat::Jpeg,
                width,
                height
            ),
            Err(err) if err.to_string().contains("alpha")
        ));

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn heic_decode_rejects_animated_and_sequence_brands() -> Result<(), PipelineError> {
        let mut aux = Vec::new();
        let rgb = gradient_rgb(32, 24);
        let still = encode(
            &rgb,
            meta(32, 24, DecodedPixelFormat::Rgb8, rgb.len()),
            80,
            &mut aux,
        )?;
        let mut pixels = Vec::new();
        let (width, height, format, len) = decode_heic_slice_into(
            Path::new(TARGET),
            &still,
            DecodeTargetHint::Native,
            &mut pixels,
        )?;
        assert_eq!(
            (width, height, format, len),
            (32, 24, DecodedPixelFormat::Rgb8, 32 * 24 * 3)
        );

        let mut animated = still.clone();
        animated[8..12].copy_from_slice(b"msf1");
        assert!(matches!(
            decode_heic_slice_into(
                Path::new(TARGET),
                &animated,
                DecodeTargetHint::Native,
                &mut pixels,
            ),
            Err(PipelineError::DecodeFailed { .. })
        ));
        Ok(())
    }
}
