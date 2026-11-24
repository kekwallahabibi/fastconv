//! Direct 8-bit AVIF encoding via rav1e + avif-serialize.
//!
//! Decoded pixels are converted once into full-range BT.601 YUV 4:2:0 and
//! written straight into a still AV1 frame, avoiding the `ravif` RGBA8 -> u8
//! -> RGBA8 round trip. An auxiliary monochrome frame is encoded only when the
//! source really has transparency.

use std::path::Path;

use avif_serialize::Aviffy;
use fastconv_domain::ImageQuality;
use rav1e::color::{
    ColorDescription, ColorPrimaries, MatrixCoefficients, PixelRange, TransferCharacteristics,
};
use rav1e::prelude::{
    ChromaSampling, Config, EncoderConfig, Frame, SceneDetectionSpeed, SpeedSettings,
};

use crate::PipelineError;
use crate::decode::{DecodedFrameMeta, DecodedPixelFormat};

/// 1..=100 quality to AV1 quantizer. Matches ravif 0.13's transfer curve so
/// high-quality settings keep the same encoder quality as the previous backend.
fn quality_to_quantizer(quality: u8) -> usize {
    let q = f64::from(quality.clamp(1, 100)) / 100.0;
    let x = if q >= 0.82 {
        (1.0 - q) * 2.6
    } else if q > 0.25 {
        q.mul_add(-0.5, 0.875)
    } else {
        1.0 - q
    };
    let clamped = (x * 255.0).round().clamp(0.0, 255.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let result = clamped as usize;
    result
}

#[allow(clippy::too_many_arguments)] // All fields configure one still AV1 frame.
fn encode_av1_frame(
    target: &Path,
    width: usize,
    height: usize,
    chroma: ChromaSampling,
    monochrome: bool,
    quantizer: usize,
    threads: usize,
    fill_planes: impl FnOnce(&mut Frame<u8>),
) -> Result<Vec<u8>, PipelineError> {
    let mut enc = EncoderConfig::with_speed_preset(8);
    enc.width = width;
    enc.height = height;
    enc.bit_depth = 8;
    enc.chroma_sampling = chroma;
    enc.pixel_range = PixelRange::Full;
    enc.color_description = if monochrome {
        None
    } else {
        Some(ColorDescription {
            color_primaries: ColorPrimaries::BT709,
            transfer_characteristics: TransferCharacteristics::SRGB,
            matrix_coefficients: MatrixCoefficients::BT601,
        })
    };
    enc.still_picture = true;
    enc.low_latency = true;
    enc.bitrate = 0;
    enc.min_key_frame_interval = 0;
    enc.max_key_frame_interval = 1;
    enc.enable_timing_info = false;
    enc.error_resilient = false;
    enc.quantizer = quantizer;
    enc.min_quantizer = u8::try_from(quantizer).unwrap_or(u8::MAX);

    enc.speed_settings = SpeedSettings::from_preset(8);
    enc.speed_settings.multiref = false;
    enc.speed_settings.rdo_lookahead_frames = 1;
    enc.speed_settings.scene_detection_mode = SceneDetectionSpeed::None;
    enc.speed_settings.lrf = false;
    enc.speed_settings.cdef = true;
    enc.speed_settings.transform.rdo_tx_decision = false;
    enc.speed_settings.transform.reduced_tx_set = true;
    enc.tiles = if threads > 1 && width >= 256 && height >= 256 {
        threads.next_power_of_two().min(16)
    } else {
        0
    };

    let cfg = Config::new()
        .with_encoder_config(enc)
        .with_threads(threads.max(1));
    let mut ctx = cfg
        .new_context::<u8>()
        .map_err(|err| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: format!("failed to create AV1 encoder context: {err}"),
        })?;

    let mut frame = ctx.new_frame();
    fill_planes(&mut frame);
    ctx.send_frame(frame)
        .map_err(|err| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: format!("failed to send AV1 frame: {err}"),
        })?;
    ctx.flush();

    let mut packet_data = Vec::new();
    loop {
        match ctx.receive_packet() {
            Ok(packet) => {
                if packet_data.is_empty() {
                    packet_data = packet.data;
                }
            }
            Err(rav1e::EncoderStatus::Encoded) => {}
            Err(rav1e::EncoderStatus::NeedMoreData | rav1e::EncoderStatus::LimitReached) => break,
            Err(err) => {
                return Err(PipelineError::EncodeFailed {
                    path: target.to_path_buf(),
                    reason: format!("failed to receive AV1 packet: {err}"),
                });
            }
        }
    }
    if packet_data.is_empty() {
        return Err(PipelineError::EmptyOutput(target.to_path_buf()));
    }
    Ok(packet_data)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Explicitly clamped to 0..=255.
const fn clamp_u8(value: i32) -> u8 {
    if value < 0 {
        0
    } else if value > 255 {
        255
    } else {
        value as u8
    }
}

#[allow(clippy::many_single_char_names)] // r/g/b are canonical color-component names.
const fn luma_bt601(r: i32, g: i32, b: i32) -> u8 {
    clamp_u8((19_595 * r + 38_470 * g + 7_471 * b + 32_768) >> 16)
}

#[allow(clippy::many_single_char_names)] // r/g/b are canonical color-component names.
const fn chroma_bt601(r: i32, g: i32, b: i32) -> (u8, u8) {
    let cb = 128 + ((-11_058 * r - 21_710 * g + 32_768 * b + 32_768) >> 16);
    let cr = 128 + ((32_768 * r - 27_439 * g - 5_329 * b + 32_768) >> 16);
    (clamp_u8(cb), clamp_u8(cr))
}

#[allow(clippy::many_single_char_names)] // x/y and r/g/b are canonical image coordinates/components.
fn fill_yuv420(
    frame: &mut Frame<u8>,
    pixels: &[u8],
    width: usize,
    height: usize,
    format: DecodedPixelFormat,
) {
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let [y_plane_ref, u_plane_ref, v_plane_ref] = &mut frame.planes;

    match format {
        DecodedPixelFormat::L8 | DecodedPixelFormat::La8 => {
            let bpp = if format == DecodedPixelFormat::L8 {
                1
            } else {
                2
            };
            for (y, row) in y_plane_ref.rows_iter_mut().take(height).enumerate() {
                let row_offset = y * width * bpp;
                for (x, dst) in row.iter_mut().take(width).enumerate() {
                    *dst = pixels[row_offset + x * bpp];
                }
            }
            for row in u_plane_ref.rows_iter_mut().take(ch) {
                row[..cw].fill(128);
            }
            for row in v_plane_ref.rows_iter_mut().take(ch) {
                row[..cw].fill(128);
            }
        }
        DecodedPixelFormat::Rgb8 | DecodedPixelFormat::Rgba8 => {
            let bpp = if format == DecodedPixelFormat::Rgb8 {
                3
            } else {
                4
            };
            let has_alpha = format == DecodedPixelFormat::Rgba8;
            for (y, row) in y_plane_ref.rows_iter_mut().take(height).enumerate() {
                let row_offset = y * width * bpp;
                for (x, dst) in row.iter_mut().take(width).enumerate() {
                    let idx = row_offset + x * bpp;
                    *dst = luma_bt601(
                        i32::from(pixels[idx]),
                        i32::from(pixels[idx + 1]),
                        i32::from(pixels[idx + 2]),
                    );
                }
            }
            for (cy, (u_row, v_row)) in u_plane_ref
                .rows_iter_mut()
                .zip(v_plane_ref.rows_iter_mut())
                .take(ch)
                .enumerate()
            {
                let y0 = cy * 2;
                let y1 = (y0 + 1).min(height - 1);
                let row0 = y0 * width * bpp;
                let row1 = y1 * width * bpp;
                for cx in 0..cw {
                    let x0 = cx * 2;
                    let x1 = (x0 + 1).min(width - 1);
                    let mut r_sum = 0_i32;
                    let mut g_sum = 0_i32;
                    let mut b_sum = 0_i32;
                    let mut a_sum = 0_i32;
                    for idx in [
                        row0 + x0 * bpp,
                        row0 + x1 * bpp,
                        row1 + x0 * bpp,
                        row1 + x1 * bpp,
                    ] {
                        // Fully transparent pixels carry invisible RGB that must not
                        // shift the chroma shared with opaque neighbours in this block.
                        let alpha = if has_alpha {
                            i32::from(pixels[idx + 3])
                        } else {
                            255
                        };
                        r_sum += i32::from(pixels[idx]) * alpha;
                        g_sum += i32::from(pixels[idx + 1]) * alpha;
                        b_sum += i32::from(pixels[idx + 2]) * alpha;
                        a_sum += alpha;
                    }
                    let (cb, cr) = if a_sum == 0 {
                        (128, 128)
                    } else {
                        let half = a_sum / 2;
                        chroma_bt601(
                            (r_sum + half) / a_sum,
                            (g_sum + half) / a_sum,
                            (b_sum + half) / a_sum,
                        )
                    };
                    u_row[cx] = cb;
                    v_row[cx] = cr;
                }
            }
        }
    }
}

fn fill_alpha(
    frame: &mut Frame<u8>,
    pixels: &[u8],
    width: usize,
    height: usize,
    alpha_stride: usize,
    alpha_offset: usize,
) {
    for (y, row) in frame.planes[0].rows_iter_mut().take(height).enumerate() {
        let row_offset = y * width * alpha_stride + alpha_offset;
        for (x, dst) in row.iter_mut().take(width).enumerate() {
            *dst = pixels[row_offset + x * alpha_stride];
        }
    }
}

pub(crate) fn encode_avif_into(
    target: &Path,
    quality: ImageQuality,
    threads: usize,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    encoded_buf: &mut Vec<u8>,
) -> Result<(), PipelineError> {
    let width = usize::try_from(decoded.width).map_err(|_| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: format!("AVIF width {} is too large", decoded.width),
    })?;
    let height = usize::try_from(decoded.height).map_err(|_| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: format!("AVIF height {} is too large", decoded.height),
    })?;
    if width == 0 || height == 0 {
        return Err(PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: "AVIF dimensions must be non-zero".to_owned(),
        });
    }
    let bpp = match decoded.format {
        DecodedPixelFormat::L8 => 1,
        DecodedPixelFormat::La8 => 2,
        DecodedPixelFormat::Rgb8 => 3,
        DecodedPixelFormat::Rgba8 => 4,
    };
    let expected = width
        .checked_mul(height)
        .and_then(|p| p.checked_mul(bpp))
        .ok_or_else(|| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: "AVIF pixel buffer size overflow".to_owned(),
        })?;
    if pixels.len() != expected {
        return Err(PipelineError::DecodeFailed {
            path: target.to_path_buf(),
            reason: format!(
                "AVIF decoded pixel buffer has {} bytes, expected {expected}",
                pixels.len()
            ),
        });
    }

    let quantizer = quality_to_quantizer(quality.get());
    // L8/La8 use Cs420 with neutral chroma because avif-serialize 0.8.9
    // hardcodes a 3-channel PixiBox even when monochrome is signalled.
    let color = encode_av1_frame(
        target,
        width,
        height,
        ChromaSampling::Cs420,
        false,
        quantizer,
        threads,
        |frame| fill_yuv420(frame, pixels, width, height, decoded.format),
    )?;

    let alpha = match decoded.format {
        DecodedPixelFormat::La8 | DecodedPixelFormat::Rgba8 => {
            let alpha_offset = bpp - 1;
            let opaque = pixels
                .iter()
                .skip(alpha_offset)
                .step_by(bpp)
                .all(|&a| a == u8::MAX);
            if opaque {
                None
            } else {
                Some(encode_av1_frame(
                    target,
                    width,
                    height,
                    ChromaSampling::Cs400,
                    true,
                    quantizer,
                    threads,
                    |frame| fill_alpha(frame, pixels, width, height, bpp, alpha_offset),
                )?)
            }
        }
        _ => None,
    };

    encoded_buf.clear();
    let mut aviffy = Aviffy::new();
    aviffy
        .set_seq_profile(0)
        .set_chroma_subsampling((true, true))
        .set_monochrome(false)
        .set_full_color_range(true)
        .set_color_primaries(avif_serialize::constants::ColorPrimaries::Bt709)
        .set_transfer_characteristics(avif_serialize::constants::TransferCharacteristics::Srgb)
        .set_matrix_coefficients(avif_serialize::constants::MatrixCoefficients::Bt601);
    aviffy
        .write(
            &mut *encoded_buf,
            &color,
            alpha.as_deref(),
            decoded.width,
            decoded.height,
            8,
        )
        .map_err(|err| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: format!("failed to serialize AVIF boxes: {err}"),
        })?;
    if encoded_buf.is_empty() {
        return Err(PipelineError::EmptyOutput(target.to_path_buf()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastconv_domain::ImageQuality;

    fn first_chroma(frame: &Frame<u8>, plane: usize) -> Option<u8> {
        frame
            .planes
            .get(plane)
            .and_then(|p| p.rows_iter().next())
            .and_then(|row| row.first().copied())
    }

    fn missing_chroma() -> PipelineError {
        PipelineError::EncodeFailed {
            path: Path::new("test.avif").to_path_buf(),
            reason: "missing chroma sample in test frame".to_owned(),
        }
    }

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

    fn encode_test_avif(
        format: DecodedPixelFormat,
        width: u32,
        height: u32,
        transparent: bool,
    ) -> Result<Vec<u8>, PipelineError> {
        let bpp = match format {
            DecodedPixelFormat::L8 => 1,
            DecodedPixelFormat::La8 => 2,
            DecodedPixelFormat::Rgb8 => 3,
            DecodedPixelFormat::Rgba8 => 4,
        };
        let mut pixels: Vec<u8> = (0..width as usize * height as usize * bpp)
            .map(|i| u8::try_from((i * 37) % 256).unwrap_or(0))
            .collect();
        if matches!(format, DecodedPixelFormat::La8 | DecodedPixelFormat::Rgba8) {
            let alpha_value = if transparent { 128 } else { u8::MAX };
            for pixel in pixels.chunks_exact_mut(bpp) {
                if let Some(alpha) = pixel.last_mut() {
                    *alpha = alpha_value;
                }
            }
        }
        let quality = ImageQuality::new(80).unwrap_or(ImageQuality::DEFAULT);
        let mut out = Vec::new();
        encode_avif_into(
            Path::new("test.avif"),
            quality,
            1,
            &pixels,
            meta(format, width, height),
            &mut out,
        )?;
        assert!(out.len() > 8);
        assert_eq!(&out[4..8], b"ftyp", "AVIF must start with an ftyp box");
        Ok(out)
    }

    fn has_alpha_aux_box(avif: &[u8]) -> bool {
        avif.windows(4).any(|window| window == b"auxC")
    }

    #[test]
    fn encodes_rgb8_and_opaque_rgba8_without_alpha_plane() -> Result<(), PipelineError> {
        let rgb = encode_test_avif(DecodedPixelFormat::Rgb8, 53, 37, false)?;
        let opaque_rgba = encode_test_avif(DecodedPixelFormat::Rgba8, 53, 37, false)?;
        assert!(!has_alpha_aux_box(&rgb));
        assert!(!has_alpha_aux_box(&opaque_rgba));
        Ok(())
    }

    #[test]
    fn encodes_transparent_and_grayscale_variants() -> Result<(), PipelineError> {
        let transparent_rgba = encode_test_avif(DecodedPixelFormat::Rgba8, 16, 16, true)?;
        let gray = encode_test_avif(DecodedPixelFormat::L8, 16, 16, false)?;
        let opaque_la = encode_test_avif(DecodedPixelFormat::La8, 17, 15, false)?;
        let transparent_la = encode_test_avif(DecodedPixelFormat::La8, 17, 15, true)?;
        assert!(has_alpha_aux_box(&transparent_rgba));
        assert!(!has_alpha_aux_box(&gray));
        assert!(!has_alpha_aux_box(&opaque_la));
        assert!(has_alpha_aux_box(&transparent_la));
        Ok(())
    }

    #[test]
    fn encodes_1x1_and_odd_dimensions() -> Result<(), PipelineError> {
        encode_test_avif(DecodedPixelFormat::Rgb8, 1, 1, false)?;
        encode_test_avif(DecodedPixelFormat::L8, 1, 1, false)?;
        encode_test_avif(DecodedPixelFormat::La8, 1, 1, true)?;
        encode_test_avif(DecodedPixelFormat::Rgba8, 53, 37, true)?;
        Ok(())
    }

    #[test]
    fn transparent_pixels_do_not_shift_shared_chroma() -> Result<(), PipelineError> {
        // rav1e requires at least 16x16; the pattern lives in the first 2x2 block.
        const SIDE: usize = 16;
        let cfg = EncoderConfig {
            width: SIDE,
            height: SIDE,
            bit_depth: 8,
            chroma_sampling: ChromaSampling::Cs420,
            ..Default::default()
        };
        let ctx = Config::new()
            .with_encoder_config(cfg)
            .with_threads(1)
            .new_context::<u8>()
            .map_err(|err| PipelineError::EncodeFailed {
                path: Path::new("test.avif").to_path_buf(),
                reason: format!("failed to create AV1 encoder context: {err}"),
            })?;
        let mut frame = ctx.new_frame();

        // One opaque red pixel and three transparent blue pixels: hidden RGB
        // must not pull the 4:2:0 chroma away from the visible red pixel.
        let mut pixels = vec![0_u8; SIDE * SIDE * 4];
        pixels[..16].copy_from_slice(&[255, 0, 0, 255, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0]);
        fill_yuv420(&mut frame, &pixels, SIDE, SIDE, DecodedPixelFormat::Rgba8);
        let cb = first_chroma(&frame, 1).ok_or_else(missing_chroma)?;
        let cr = first_chroma(&frame, 2).ok_or_else(missing_chroma)?;
        assert_eq!((cb, cr), chroma_bt601(255, 0, 0));

        // An all-transparent block has no visible colour: neutral chroma.
        let transparent = vec![0_u8; SIDE * SIDE * 4];
        fill_yuv420(
            &mut frame,
            &transparent,
            SIDE,
            SIDE,
            DecodedPixelFormat::Rgba8,
        );
        let cb = first_chroma(&frame, 1).ok_or_else(missing_chroma)?;
        let cr = first_chroma(&frame, 2).ok_or_else(missing_chroma)?;
        assert_eq!((cb, cr), (128, 128));
        Ok(())
    }

    #[test]
    fn quality_curve_is_monotonic() {
        assert_eq!(quality_to_quantizer(100), 0);
        assert_eq!(quality_to_quantizer(99), 7);
        assert_eq!(quality_to_quantizer(90), 66);
        assert!(quality_to_quantizer(90) < quality_to_quantizer(80));
        assert!(quality_to_quantizer(80) < quality_to_quantizer(50));
        assert!(quality_to_quantizer(50) < quality_to_quantizer(10));
    }
}
