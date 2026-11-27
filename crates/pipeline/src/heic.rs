//! Pure-Rust CPU HEIC still-image encoding via `heifer`.
//!
//! Validated decoded pixels are handed straight to `heifer`'s HEVC-Intra encoder:
//! RGB8/RGBA8 are passed through untouched, while L8/La8 are expanded once into
//! the reusable worker scratch buffer. The encoder keeps its default `Medium`
//! speed/quality trade-off (no preset cheating); multi-core scaling comes from
//! the batch workers plus `heifer`'s own 512-pixel tile threading.

use std::path::Path;

use fastconv_domain::ImageQuality;
use heifer::EncodeOptions;

use crate::PipelineError;
use crate::decode::{DecodedFrameMeta, DecodedPixelFormat};

/// Half of `heifer`'s own 2^28 pixel limit: bounds tile scratch memory on 4K batches.
const MAX_HEIC_PIXELS: u64 = 128 * 1024 * 1024;

/// Per-worker cap on the expanded RGB/RGBA scratch buffer. RGB8/RGBA8 are passed
/// to `heifer` zero-copy, but L8/La8 are expanded 3x/4x here; with at most 8 image
/// workers this keeps that scratch under 1 GiB for adversarial near-cap sources.
const MAX_HEIC_EXPANDED_BYTES: u64 = 128 * 1024 * 1024;

const fn pixel_channels(format: DecodedPixelFormat) -> u64 {
    match format {
        DecodedPixelFormat::L8 => 1,
        DecodedPixelFormat::La8 => 2,
        DecodedPixelFormat::Rgb8 => 3,
        DecodedPixelFormat::Rgba8 => 4,
    }
}

/// Bytes of reusable `aux_buf` scratch one pixel costs for a given layout.
const fn aux_expansion_channels(format: DecodedPixelFormat) -> u64 {
    match format {
        DecodedPixelFormat::L8 => 3,
        DecodedPixelFormat::La8 => 4,
        DecodedPixelFormat::Rgb8 | DecodedPixelFormat::Rgba8 => 1,
    }
}

fn encode_error(target: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: reason.into(),
    }
}

/// Checks dimensions and exact buffer length before handing pixels to `heifer`.
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

    let pixel_count = u64::from(decoded.width)
        .checked_mul(u64::from(decoded.height))
        .filter(|&count| count <= MAX_HEIC_PIXELS)
        .ok_or_else(|| {
            encode_error(
                target,
                format!(
                    "HEIC image has too many pixels: {}x{} (limit {MAX_HEIC_PIXELS})",
                    decoded.width, decoded.height
                ),
            )
        })?;

    pixel_count
        .checked_mul(aux_expansion_channels(decoded.format))
        .filter(|&bytes| bytes <= MAX_HEIC_EXPANDED_BYTES)
        .ok_or_else(|| {
            encode_error(
                target,
                format!(
                    "HEIC expansion scratch needs too many bytes (limit {MAX_HEIC_EXPANDED_BYTES})"
                ),
            )
        })?;

    let expected = pixel_count
        .checked_mul(pixel_channels(decoded.format))
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

#[cfg(test)]
mod tests {
    use super::{MAX_HEIC_PIXELS, encode_heic_into};
    use crate::PipelineError;
    use crate::decode::{DecodedFrameMeta, DecodedPixelFormat};
    use fastconv_domain::ImageQuality;
    use std::path::Path;

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
    fn rejects_zero_oversized_and_mismatched_inputs() {
        let mut aux = Vec::new();
        assert!(matches!(
            encode(&[], meta(0, 8, DecodedPixelFormat::Rgb8, 0), 80, &mut aux,),
            Err(PipelineError::EncodeFailed { .. })
        ));
        // 16384^2 = 2^28 pixels exceeds the 2^27 pixel cap.
        assert!(matches!(
            encode(
                &[],
                meta(16_384, 16_384, DecodedPixelFormat::Rgb8, 0),
                80,
                &mut aux,
            ),
            Err(PipelineError::EncodeFailed { .. })
        ));
        assert!(u64::from(16_384_u32) * u64::from(16_384_u32) > MAX_HEIC_PIXELS);
        // 100M L8 pixels pass the pixel cap but need ~300 MiB of expansion scratch,
        // above the per-worker cap that keeps an 8-worker batch under ~1 GiB.
        assert!(matches!(
            encode(
                &[],
                meta(100_000_000, 1, DecodedPixelFormat::L8, 0),
                80,
                &mut aux,
            ),
            Err(PipelineError::EncodeFailed { ref reason, .. })
                if reason.contains("expansion scratch")
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
}
