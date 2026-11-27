//! Direct 8-bit AVIF encoding via rav1e + avif-serialize.
//!
//! Decoded pixels are converted once into full-range BT.601 YUV 4:2:0 and
//! written straight into a still AV1 frame, avoiding the `ravif` RGBA8 -> u8
//! -> RGBA8 round trip. An auxiliary monochrome frame is encoded only when the
//! source really has transparency.

use std::path::Path;

use avif_serialize::Aviffy;
use fastconv_domain::{ChromaSubsampling, CompressionLevel, ImageQuality};
use rav1e::color::{
    ColorDescription, ColorPrimaries, MatrixCoefficients, PixelRange, TransferCharacteristics,
};
use rav1e::prelude::{
    BlockSize, ChromaSampling, Config, EncoderConfig, Frame, PartitionRange, SceneDetectionSpeed,
    SegmentationLevel, SpeedSettings, Tune,
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
    level: CompressionLevel,
    threads: usize,
    fill_planes: impl FnOnce(&mut Frame<u8>),
) -> Result<Vec<u8>, PipelineError> {
    // Presets 10/8/6 keep the still-frame invariants below while trading encode
    // effort for compression. Preset 10 additionally disables tools that only
    // matter for the fastest path and pins 64x64 partitions on large images.
    let preset = match level {
        CompressionLevel::Fast => 10,
        CompressionLevel::Balanced => 8,
        CompressionLevel::Best => 6,
    };
    let mut enc = EncoderConfig::with_speed_preset(preset);
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
    // Tune::Psnr enables TX-domain distortion (ignored under Psychovisual) and
    // avoids pixel-domain psy distortion passes on every intra candidate.
    enc.tune = Tune::Psnr;

    // Multi-reference, lookahead and scene detection cannot help a still frame.
    enc.speed_settings = SpeedSettings::from_preset(preset);
    enc.speed_settings.multiref = false;
    enc.speed_settings.rdo_lookahead_frames = 1;
    enc.speed_settings.scene_detection_mode = SceneDetectionSpeed::None;
    if level == CompressionLevel::Fast {
        // Preset 10 fixes partitions to 32x32 and skips CDEF/fine directional
        // intra, eliminating the 8x8/16x16 partition search bottleneck on 4K.
        enc.speed_settings.lrf = false;
        enc.speed_settings.cdef = false;
        enc.speed_settings.segmentation = SegmentationLevel::Disabled;
        enc.speed_settings.transform.rdo_tx_decision = false;
        enc.speed_settings.transform.reduced_tx_set = true;
        enc.speed_settings.transform.tx_domain_distortion = true;
        enc.speed_settings.transform.tx_domain_rate = true;
        enc.speed_settings.prediction.fine_directional_intra = false;
        if width >= 1024 && height >= 1024 {
            enc.speed_settings.partition.partition_range =
                PartitionRange::new(BlockSize::BLOCK_64X64, BlockSize::BLOCK_64X64);
        }
    }
    enc.tiles = if threads > 1 && width >= 256 && height >= 256 {
        threads.next_power_of_two().clamp(4, 16)
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

#[allow(clippy::many_single_char_names)] // r/g/b are canonical color-component names.
#[inline]
fn chroma_from_opaque_block(r_sum: i32, g_sum: i32, b_sum: i32) -> (u8, u8) {
    // Four opaque pixels: (sum * 255 + 510) / 1020 == (sum + 2) / 4.
    chroma_bt601((r_sum + 2) / 4, (g_sum + 2) / 4, (b_sum + 2) / 4)
}

/// Fused two-row luma + chroma pass for opaque RGB8: no alpha weighting or
/// division, and independent row slices let LLVM auto-vectorize the luma loop.
#[allow(clippy::many_single_char_names)]
fn fill_yuv420_rgb(frame: &mut Frame<u8>, pixels: &[u8], width: usize, height: usize) {
    let cw = width.div_ceil(2);
    let [y_plane, u_plane, v_plane] = &mut frame.planes;
    let mut y_rows = y_plane.rows_iter_mut();
    let mut u_rows = u_plane.rows_iter_mut();
    let mut v_rows = v_plane.rows_iter_mut();
    let row_bytes = width * 3;

    for y0 in (0..height).step_by(2) {
        let Some(y_dst0) = y_rows.next() else {
            break;
        };
        let y1 = (y0 + 1).min(height.saturating_sub(1));
        let y_dst1 = if y1 == y0 { None } else { y_rows.next() };
        let row0 = &pixels[y0 * row_bytes..(y0 + 1) * row_bytes];
        let row1 = &pixels[y1 * row_bytes..(y1 + 1) * row_bytes];

        for (x, dst) in y_dst0.iter_mut().take(width).enumerate() {
            let i = x * 3;
            *dst = luma_bt601(
                i32::from(row0[i]),
                i32::from(row0[i + 1]),
                i32::from(row0[i + 2]),
            );
        }
        if let Some(dst1) = y_dst1 {
            for (x, dst) in dst1.iter_mut().take(width).enumerate() {
                let i = x * 3;
                *dst = luma_bt601(
                    i32::from(row1[i]),
                    i32::from(row1[i + 1]),
                    i32::from(row1[i + 2]),
                );
            }
        }

        let (Some(u_dst), Some(v_dst)) = (u_rows.next(), v_rows.next()) else {
            break;
        };
        for cx in 0..cw {
            let x0 = cx * 2;
            let x1 = (x0 + 1).min(width.saturating_sub(1));
            let i0 = x0 * 3;
            let i1 = x1 * 3;
            let (cb, cr) = chroma_from_opaque_block(
                i32::from(row0[i0])
                    + i32::from(row0[i1])
                    + i32::from(row1[i0])
                    + i32::from(row1[i1]),
                i32::from(row0[i0 + 1])
                    + i32::from(row0[i1 + 1])
                    + i32::from(row1[i0 + 1])
                    + i32::from(row1[i1 + 1]),
                i32::from(row0[i0 + 2])
                    + i32::from(row0[i1 + 2])
                    + i32::from(row1[i0 + 2])
                    + i32::from(row1[i1 + 2]),
            );
            u_dst[cx] = cb;
            v_dst[cx] = cr;
        }
    }
}

/// Same fused pass for RGBA8; a fully opaque 2x2 block takes the division-free
/// path, while partially transparent blocks keep alpha-weighted chroma.
#[allow(clippy::many_single_char_names)]
fn fill_yuv420_rgba(frame: &mut Frame<u8>, pixels: &[u8], width: usize, height: usize) {
    let cw = width.div_ceil(2);
    let [y_plane, u_plane, v_plane] = &mut frame.planes;
    let mut y_rows = y_plane.rows_iter_mut();
    let mut u_rows = u_plane.rows_iter_mut();
    let mut v_rows = v_plane.rows_iter_mut();
    let row_bytes = width * 4;

    for y0 in (0..height).step_by(2) {
        let Some(y_dst0) = y_rows.next() else {
            break;
        };
        let y1 = (y0 + 1).min(height.saturating_sub(1));
        let y_dst1 = if y1 == y0 { None } else { y_rows.next() };
        let row0 = &pixels[y0 * row_bytes..(y0 + 1) * row_bytes];
        let row1 = &pixels[y1 * row_bytes..(y1 + 1) * row_bytes];

        for (x, dst) in y_dst0.iter_mut().take(width).enumerate() {
            let i = x * 4;
            *dst = luma_bt601(
                i32::from(row0[i]),
                i32::from(row0[i + 1]),
                i32::from(row0[i + 2]),
            );
        }
        if let Some(dst1) = y_dst1 {
            for (x, dst) in dst1.iter_mut().take(width).enumerate() {
                let i = x * 4;
                *dst = luma_bt601(
                    i32::from(row1[i]),
                    i32::from(row1[i + 1]),
                    i32::from(row1[i + 2]),
                );
            }
        }

        let (Some(u_dst), Some(v_dst)) = (u_rows.next(), v_rows.next()) else {
            break;
        };
        for cx in 0..cw {
            let x0 = cx * 2;
            let x1 = (x0 + 1).min(width.saturating_sub(1));
            let i0 = x0 * 4;
            let i1 = x1 * 4;
            let [r0, g0, b0, a0] = [row0[i0], row0[i0 + 1], row0[i0 + 2], row0[i0 + 3]];
            let [r1, g1, b1, a1] = [row0[i1], row0[i1 + 1], row0[i1 + 2], row0[i1 + 3]];
            let [r2, g2, b2, a2] = [row1[i0], row1[i0 + 1], row1[i0 + 2], row1[i0 + 3]];
            let [r3, g3, b3, a3] = [row1[i1], row1[i1 + 1], row1[i1 + 2], row1[i1 + 3]];

            let (cb, cr) = if a0 & a1 & a2 & a3 == u8::MAX {
                chroma_from_opaque_block(
                    i32::from(r0) + i32::from(r1) + i32::from(r2) + i32::from(r3),
                    i32::from(g0) + i32::from(g1) + i32::from(g2) + i32::from(g3),
                    i32::from(b0) + i32::from(b1) + i32::from(b2) + i32::from(b3),
                )
            } else {
                // Fully transparent pixels carry invisible RGB that must not shift
                // the chroma shared with opaque neighbours in this block.
                let a0 = i32::from(a0);
                let a1 = i32::from(a1);
                let a2 = i32::from(a2);
                let a3 = i32::from(a3);
                let r_sum = i32::from(r0) * a0
                    + i32::from(r1) * a1
                    + i32::from(r2) * a2
                    + i32::from(r3) * a3;
                let g_sum = i32::from(g0) * a0
                    + i32::from(g1) * a1
                    + i32::from(g2) * a2
                    + i32::from(g3) * a3;
                let b_sum = i32::from(b0) * a0
                    + i32::from(b1) * a1
                    + i32::from(b2) * a2
                    + i32::from(b3) * a3;
                let a_sum = a0 + a1 + a2 + a3;
                if a_sum == 0 {
                    (128, 128)
                } else {
                    let half = a_sum / 2;
                    chroma_bt601(
                        (r_sum + half) / a_sum,
                        (g_sum + half) / a_sum,
                        (b_sum + half) / a_sum,
                    )
                }
            };
            u_dst[cx] = cb;
            v_dst[cx] = cr;
        }
    }
}

#[allow(clippy::many_single_char_names)] // x/y and r/g/b are canonical image coordinates/components.
fn fill_yuv420(
    frame: &mut Frame<u8>,
    pixels: &[u8],
    width: usize,
    height: usize,
    format: DecodedPixelFormat,
) {
    match format {
        DecodedPixelFormat::L8 | DecodedPixelFormat::La8 => {
            let cw = width.div_ceil(2);
            let ch = height.div_ceil(2);
            let [y_plane, u_plane, v_plane] = &mut frame.planes;
            let bpp = if format == DecodedPixelFormat::L8 {
                1
            } else {
                2
            };
            for (y, row) in y_plane.rows_iter_mut().take(height).enumerate() {
                let row_offset = y * width * bpp;
                for (x, dst) in row.iter_mut().take(width).enumerate() {
                    *dst = pixels[row_offset + x * bpp];
                }
            }
            for row in u_plane.rows_iter_mut().take(ch) {
                row[..cw].fill(128);
            }
            for row in v_plane.rows_iter_mut().take(ch) {
                row[..cw].fill(128);
            }
        }
        DecodedPixelFormat::Rgb8 => fill_yuv420_rgb(frame, pixels, width, height),
        DecodedPixelFormat::Rgba8 => fill_yuv420_rgba(frame, pixels, width, height),
    }
}

/// Full-resolution luma/chroma planes for 4:4:4. Transparent RGB samples keep
/// their luma but contribute neutral chroma so hidden colours cannot bleed into
/// neighbouring visible pixels.
#[allow(clippy::many_single_char_names)] // r/g/b/a are canonical colour components.
fn fill_yuv444(
    frame: &mut Frame<u8>,
    pixels: &[u8],
    width: usize,
    height: usize,
    format: DecodedPixelFormat,
) {
    let [y_plane, u_plane, v_plane] = &mut frame.planes;
    let mut y_rows = y_plane.rows_iter_mut();
    let mut u_rows = u_plane.rows_iter_mut();
    let mut v_rows = v_plane.rows_iter_mut();
    let bpp = match format {
        DecodedPixelFormat::L8 => 1_usize,
        DecodedPixelFormat::La8 => 2_usize,
        DecodedPixelFormat::Rgb8 => 3_usize,
        DecodedPixelFormat::Rgba8 => 4_usize,
    };

    for row_index in 0..height {
        let (Some(y_dst), Some(u_dst), Some(v_dst)) = (y_rows.next(), u_rows.next(), v_rows.next())
        else {
            break;
        };
        let row_start = row_index * width * bpp;
        for x in 0..width {
            let i = row_start + x * bpp;
            let (r, g, b, alpha) = match format {
                DecodedPixelFormat::L8 | DecodedPixelFormat::La8 => {
                    let value = i32::from(pixels[i]);
                    (value, value, value, None)
                }
                DecodedPixelFormat::Rgb8 => (
                    i32::from(pixels[i]),
                    i32::from(pixels[i + 1]),
                    i32::from(pixels[i + 2]),
                    None,
                ),
                DecodedPixelFormat::Rgba8 => (
                    i32::from(pixels[i]),
                    i32::from(pixels[i + 1]),
                    i32::from(pixels[i + 2]),
                    Some(pixels[i + 3]),
                ),
            };
            y_dst[x] = luma_bt601(r, g, b);
            if alpha == Some(0) {
                (u_dst[x], v_dst[x]) = (128, 128);
            } else {
                let (cb, cr) = chroma_bt601(r, g, b);
                (u_dst[x], v_dst[x]) = (cb, cr);
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

type FillYuv = fn(&mut Frame<u8>, &[u8], usize, usize, DecodedPixelFormat);

struct AvifColorPlan {
    sampling: ChromaSampling,
    fill: FillYuv,
    seq_profile: u8,
    subsampled_xy: (bool, bool),
}

impl AvifColorPlan {
    fn new(chroma: ChromaSubsampling) -> Self {
        // Color frames always carry three planes: avif-serialize 0.8.9 hardcodes a
        // 3-channel PixiBox even when monochrome is signalled, so grayscale sources
        // use neutral chroma rather than a single-plane AV1 frame.
        let (sampling, fill, seq_profile, subsampled_xy) = match chroma {
            ChromaSubsampling::Yuv420 => (
                ChromaSampling::Cs420,
                fill_yuv420 as FillYuv,
                0_u8,
                (true, true),
            ),
            ChromaSubsampling::Yuv444 => (
                ChromaSampling::Cs444,
                fill_yuv444 as FillYuv,
                1_u8,
                (false, false),
            ),
        };
        Self {
            sampling,
            fill,
            seq_profile,
            subsampled_xy,
        }
    }
}

struct AvifFrameSettings {
    quantizer: usize,
    level: CompressionLevel,
    threads: usize,
}

pub(crate) struct AvifEncodeRequest<'a> {
    pub(crate) target: &'a Path,
    pub(crate) quality: ImageQuality,
    pub(crate) level: CompressionLevel,
    pub(crate) chroma: ChromaSubsampling,
    pub(crate) threads: usize,
    pub(crate) pixels: &'a [u8],
    pub(crate) decoded: DecodedFrameMeta,
    pub(crate) encoded_buf: &'a mut Vec<u8>,
}

fn validated_avif_layout(
    target: &Path,
    decoded: DecodedFrameMeta,
    pixels: &[u8],
) -> Result<(usize, usize, usize), PipelineError> {
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
    Ok((width, height, bpp))
}

fn encode_avif_alpha(
    target: &Path,
    pixels: &[u8],
    width: usize,
    height: usize,
    bpp: usize,
    settings: &AvifFrameSettings,
) -> Result<Option<Vec<u8>>, PipelineError> {
    let alpha_offset = bpp - 1;
    let opaque = pixels
        .iter()
        .skip(alpha_offset)
        .step_by(bpp)
        .all(|&a| a == u8::MAX);
    if opaque {
        return Ok(None);
    }
    encode_av1_frame(
        target,
        width,
        height,
        ChromaSampling::Cs400,
        true,
        settings.quantizer,
        settings.level,
        settings.threads,
        |frame| fill_alpha(frame, pixels, width, height, bpp, alpha_offset),
    )
    .map(Some)
}

fn serialize_avif(
    target: &Path,
    encoded_buf: &mut Vec<u8>,
    color: &[u8],
    alpha: Option<&[u8]>,
    decoded: DecodedFrameMeta,
    plan: &AvifColorPlan,
) -> Result<(), PipelineError> {
    encoded_buf.clear();
    let mut aviffy = Aviffy::new();
    aviffy
        .set_seq_profile(plan.seq_profile)
        .set_chroma_subsampling(plan.subsampled_xy)
        .set_monochrome(false)
        .set_full_color_range(true)
        .set_color_primaries(avif_serialize::constants::ColorPrimaries::Bt709)
        .set_transfer_characteristics(avif_serialize::constants::TransferCharacteristics::Srgb)
        .set_matrix_coefficients(avif_serialize::constants::MatrixCoefficients::Bt601);
    aviffy
        .write(
            &mut *encoded_buf,
            color,
            alpha,
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

pub(crate) fn encode_avif_into(request: AvifEncodeRequest<'_>) -> Result<(), PipelineError> {
    let AvifEncodeRequest {
        target,
        quality,
        level,
        chroma,
        threads,
        pixels,
        decoded,
        encoded_buf,
    } = request;
    let (width, height, bpp) = validated_avif_layout(target, decoded, pixels)?;
    let settings = AvifFrameSettings {
        quantizer: quality_to_quantizer(quality.get()),
        level,
        threads,
    };
    let plan = AvifColorPlan::new(chroma);
    let color = encode_av1_frame(
        target,
        width,
        height,
        plan.sampling,
        false,
        settings.quantizer,
        settings.level,
        settings.threads,
        |frame| (plan.fill)(frame, pixels, width, height, decoded.format),
    )?;
    let alpha = match decoded.format {
        DecodedPixelFormat::La8 | DecodedPixelFormat::Rgba8 => {
            encode_avif_alpha(target, pixels, width, height, bpp, &settings)?
        }
        _ => None,
    };
    serialize_avif(
        target,
        encoded_buf,
        &color,
        alpha.as_deref(),
        decoded,
        &plan,
    )
}

/// Decodes a still AVIF into `pixel_buf`, reusing its allocation when large enough.
pub(crate) fn decode_avif_slice_into(
    source: &Path,
    file_bytes: &[u8],
    pixel_buf: &mut Vec<u8>,
) -> Result<(u32, u32, DecodedPixelFormat, usize), PipelineError> {
    let parse_config = zenavif_parse::DecodeConfig::default()
        .lenient(true)
        .with_peak_memory_limit(512 * 1024 * 1024)
        .with_total_megapixels_limit(128);
    let parser = zenavif_parse::AvifParser::from_bytes_with_config(
        file_bytes,
        &parse_config,
        &zenavif_parse::Unstoppable,
    )
    .map_err(|err| PipelineError::DecodeFailed {
        path: source.to_path_buf(),
        reason: format!("invalid AVIF container: {err}"),
    })?;
    if parser.major_brand() == b"avis" || parser.animation_info().is_some() {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "animated AVIF is not supported by the still-image pipeline".to_owned(),
        });
    }

    let config = zenavif::DecoderConfig::new()
        .threads(1)
        .prefer_8bit(true)
        .frame_size_limit(128 * 1024 * 1024);
    let decoded =
        zenavif::decode_with(file_bytes, &config, &zenavif::Unstoppable).map_err(|err| {
            PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: format!("failed to decode AVIF image: {err}"),
            }
        })?;

    let width = decoded.width();
    let height = decoded.height();
    let (format, bpp) = match (decoded.is_grayscale(), decoded.has_alpha()) {
        (true, false) => (DecodedPixelFormat::L8, 1_usize),
        (true, true) => (DecodedPixelFormat::La8, 2_usize),
        (false, false) => (DecodedPixelFormat::Rgb8, 3_usize),
        (false, true) => (DecodedPixelFormat::Rgba8, 4_usize),
    };
    let expected_len = usize::try_from(width)
        .ok()
        .and_then(|w| {
            usize::try_from(height)
                .ok()
                .and_then(|h| w.checked_mul(h).and_then(|pixels| pixels.checked_mul(bpp)))
        })
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "AVIF dimensions overflow decoded buffer size".to_owned(),
        })?;

    let owned_bytes;
    let bytes: &[u8] = if let Some(contiguous) = decoded.as_contiguous_bytes() {
        contiguous
    } else {
        owned_bytes = decoded.copy_to_contiguous_bytes();
        &owned_bytes
    };
    if bytes.len() != expected_len {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: format!(
                "decoded AVIF buffer has {} bytes, expected {expected_len}",
                bytes.len()
            ),
        });
    }
    if pixel_buf.len() < expected_len {
        pixel_buf.resize(expected_len, 0);
    }
    pixel_buf[..expected_len].copy_from_slice(bytes);
    Ok((width, height, format, expected_len))
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

    fn encode_test_avif_with(
        format: DecodedPixelFormat,
        width: u32,
        height: u32,
        transparent: bool,
        level: CompressionLevel,
        chroma: ChromaSubsampling,
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
        encode_avif_into(AvifEncodeRequest {
            target: Path::new("test.avif"),
            quality,
            level,
            chroma,
            threads: 1,
            pixels: &pixels,
            decoded: meta(format, width, height),
            encoded_buf: &mut out,
        })?;
        assert!(out.len() > 8);
        assert_eq!(&out[4..8], b"ftyp", "AVIF must start with an ftyp box");
        Ok(out)
    }

    fn encode_test_avif(
        format: DecodedPixelFormat,
        width: u32,
        height: u32,
        transparent: bool,
    ) -> Result<Vec<u8>, PipelineError> {
        encode_test_avif_with(
            format,
            width,
            height,
            transparent,
            CompressionLevel::Fast,
            ChromaSubsampling::Yuv420,
        )
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

    fn assert_decode_roundtrip(
        format: DecodedPixelFormat,
        width: u32,
        height: u32,
        transparent: bool,
    ) -> Result<(), PipelineError> {
        let encoded = encode_test_avif(format, width, height, transparent)?;
        let mut pixels = Vec::new();
        let (decoded_w, decoded_h, decoded_format, byte_len) =
            decode_avif_slice_into(Path::new("test.avif"), encoded.as_slice(), &mut pixels)?;
        assert_eq!((decoded_w, decoded_h), (width, height));
        let bpp = match decoded_format {
            DecodedPixelFormat::L8 => 1_usize,
            DecodedPixelFormat::La8 => 2_usize,
            DecodedPixelFormat::Rgb8 => 3_usize,
            DecodedPixelFormat::Rgba8 => 4_usize,
        };
        assert_eq!(
            byte_len,
            usize::try_from(width).unwrap_or(0) * usize::try_from(height).unwrap_or(0) * bpp
        );
        assert!(pixels.len() >= byte_len);
        if transparent {
            assert!(
                matches!(
                    decoded_format,
                    DecodedPixelFormat::La8 | DecodedPixelFormat::Rgba8
                ),
                "transparent AVIF must decode with alpha, got {decoded_format:?}"
            );
        }
        if matches!(format, DecodedPixelFormat::L8 | DecodedPixelFormat::Rgb8) {
            assert!(
                matches!(
                    decoded_format,
                    DecodedPixelFormat::L8 | DecodedPixelFormat::Rgb8
                ),
                "opaque AVIF must decode without alpha, got {decoded_format:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn decodes_rgb8_and_opaque_rgba8_roundtrip() -> Result<(), PipelineError> {
        assert_decode_roundtrip(DecodedPixelFormat::Rgb8, 53, 37, false)?;
        assert_decode_roundtrip(DecodedPixelFormat::Rgba8, 53, 37, false)?;
        Ok(())
    }

    #[test]
    fn decodes_transparent_rgba8_roundtrip() -> Result<(), PipelineError> {
        assert_decode_roundtrip(DecodedPixelFormat::Rgba8, 16, 16, true)
    }

    #[test]
    fn decodes_grayscale_and_grayscale_alpha_roundtrip() -> Result<(), PipelineError> {
        assert_decode_roundtrip(DecodedPixelFormat::L8, 16, 16, false)?;
        assert_decode_roundtrip(DecodedPixelFormat::La8, 17, 15, false)?;
        assert_decode_roundtrip(DecodedPixelFormat::La8, 17, 15, true)?;
        Ok(())
    }

    #[test]
    fn decodes_1x1_and_odd_dimensions() -> Result<(), PipelineError> {
        assert_decode_roundtrip(DecodedPixelFormat::Rgb8, 1, 1, false)?;
        assert_decode_roundtrip(DecodedPixelFormat::L8, 1, 1, false)?;
        assert_decode_roundtrip(DecodedPixelFormat::Rgba8, 53, 37, true)?;
        Ok(())
    }

    fn assert_level_chroma_roundtrip(
        level: CompressionLevel,
        chroma: ChromaSubsampling,
        format: DecodedPixelFormat,
        transparent: bool,
        width: u32,
        height: u32,
    ) -> Result<(), PipelineError> {
        let encoded = encode_test_avif_with(format, width, height, transparent, level, chroma)?;
        assert!(encoded.len() > 8, "{level:?} {chroma:?} produced no AVIF");
        let mut pixels = Vec::new();
        let (decoded_w, decoded_h, decoded_format, byte_len) =
            decode_avif_slice_into(Path::new("test.avif"), &encoded, &mut pixels)?;
        assert_eq!((decoded_w, decoded_h), (width, height));
        let bpp = match decoded_format {
            DecodedPixelFormat::L8 => 1_usize,
            DecodedPixelFormat::La8 => 2_usize,
            DecodedPixelFormat::Rgb8 => 3_usize,
            DecodedPixelFormat::Rgba8 => 4_usize,
        };
        let usize_w = usize::try_from(width).unwrap_or(0);
        let usize_h = usize::try_from(height).unwrap_or(0);
        assert_eq!(byte_len, usize_w * usize_h * bpp);
        assert!(pixels.len() >= byte_len);
        if transparent {
            assert!(
                matches!(
                    decoded_format,
                    DecodedPixelFormat::La8 | DecodedPixelFormat::Rgba8
                ),
                "transparent AVIF must keep alpha"
            );
        }
        Ok(())
    }

    #[test]
    fn encodes_and_decodes_every_level_and_chroma_mode() -> Result<(), PipelineError> {
        for level in [
            CompressionLevel::Fast,
            CompressionLevel::Balanced,
            CompressionLevel::Best,
        ] {
            for chroma in [ChromaSubsampling::Yuv420, ChromaSubsampling::Yuv444] {
                for format in [
                    DecodedPixelFormat::Rgb8,
                    DecodedPixelFormat::Rgba8,
                    DecodedPixelFormat::L8,
                    DecodedPixelFormat::La8,
                ] {
                    assert_level_chroma_roundtrip(level, chroma, format, false, 37, 27)?;
                }
                assert_level_chroma_roundtrip(
                    level,
                    chroma,
                    DecodedPixelFormat::Rgba8,
                    true,
                    37,
                    27,
                )?;
                assert_level_chroma_roundtrip(
                    level,
                    chroma,
                    DecodedPixelFormat::La8,
                    true,
                    19,
                    17,
                )?;
            }
        }
        Ok(())
    }

    #[test]
    fn yuv444_keeps_full_resolution_chroma_and_neutralizes_transparent_samples()
    -> Result<(), PipelineError> {
        const SIDE: usize = 16;
        let cfg = EncoderConfig {
            width: SIDE,
            height: SIDE,
            bit_depth: 8,
            chroma_sampling: ChromaSampling::Cs444,
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
        let mut pixels = vec![0_u8; SIDE * SIDE * 4];
        pixels[..4].copy_from_slice(&[255, 0, 0, 255]);
        pixels[4..8].copy_from_slice(&[0, 255, 0, 0]);
        fill_yuv444(&mut frame, &pixels, SIDE, SIDE, DecodedPixelFormat::Rgba8);

        assert_eq!(first_luma(&frame, 0), luma_bt601(255, 0, 0));
        assert_eq!(
            (
                first_chroma(&frame, 1).ok_or_else(missing_chroma)?,
                first_chroma(&frame, 2).ok_or_else(missing_chroma)?
            ),
            chroma_bt601(255, 0, 0)
        );
        let u_plane = frame
            .planes
            .get(1)
            .and_then(|plane| plane.rows_iter().next())
            .ok_or_else(missing_chroma)?;
        let v_plane = frame
            .planes
            .get(2)
            .and_then(|plane| plane.rows_iter().next())
            .ok_or_else(missing_chroma)?;
        assert_eq!((u_plane[1], v_plane[1]), (128, 128));
        assert!(u_plane.len() >= SIDE, "4:4:4 chroma must be full width");
        Ok(())
    }

    fn first_luma(frame: &Frame<u8>, plane: usize) -> u8 {
        frame
            .planes
            .get(plane)
            .and_then(|p| p.rows_iter().next())
            .and_then(|row| row.first().copied())
            .unwrap_or_default()
    }
}
