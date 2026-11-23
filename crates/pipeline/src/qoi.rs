use std::path::Path;

use crate::PipelineError;
use crate::decode::{DecodeTargetHint, DecodedFrameMeta, DecodedPixelFormat};

const QOI_MAGIC: [u8; 4] = *b"qoif";
const QOI_END_MARKER: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 1];
const QOI_HEADER_LEN: usize = 14;
const QOI_MAX_RUN: usize = 62;
// Shared pixel budget across QOI probe, decode, and encode (128 Mi pixels = 512 MiB RGBA8)
// so fastconv always accepts its own QOI outputs and malformed headers cannot exhaust RAM.
const MAX_QOI_PIXELS: usize = 128 * 1024 * 1024;
const QOI_OP_RUN: u8 = 0xC0;
const QOI_OP_DIFF: u8 = 0x40;
const QOI_OP_LUMA: u8 = 0x80;
const QOI_OP_RGB: u8 = 0xFE;
const QOI_OP_RGBA: u8 = 0xFF;

#[inline]
const fn qoi_hash_rgba(r: u8, g: u8, b: u8, a: u8) -> usize {
    (r.wrapping_mul(3)
        .wrapping_add(g.wrapping_mul(5))
        .wrapping_add(b.wrapping_mul(7))
        .wrapping_add(a.wrapping_mul(11))
        & 63) as usize
}

#[inline]
const fn qoi_hash_rgb_opaque(r: u8, g: u8, b: u8) -> usize {
    // 255 * 11 == 2805 ≡ 53 (mod 64).
    (r.wrapping_mul(3)
        .wrapping_add(g.wrapping_mul(5))
        .wrapping_add(b.wrapping_mul(7))
        .wrapping_add(53)
        & 63) as usize
}

#[inline]
const fn pack_rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
    u32::from_ne_bytes([r, g, b, a])
}

#[inline]
fn write_qoi_header(buf: &mut Vec<u8>, width: u32, height: u32, channels: u8) {
    buf.extend_from_slice(&QOI_MAGIC);
    buf.extend_from_slice(&width.to_be_bytes());
    buf.extend_from_slice(&height.to_be_bytes());
    buf.push(channels);
    buf.push(0); // sRGB with linear alpha
}

/// Encodes decoded 8-bit pixels into a spec-compliant QOI bitstream in `encoded_buf`.
pub(crate) fn encode_qoi_into(
    target: &Path,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    encoded_buf: &mut Vec<u8>,
) -> Result<(), PipelineError> {
    let pixel_count = usize::try_from(decoded.width)
        .ok()
        .and_then(|w| {
            usize::try_from(decoded.height)
                .ok()
                .and_then(|h| w.checked_mul(h))
        })
        .ok_or_else(|| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: "QOI image pixel count overflow".to_owned(),
        })?;
    if pixel_count > MAX_QOI_PIXELS {
        return Err(PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: format!("QOI pixel count ({pixel_count}) exceeds {MAX_QOI_PIXELS}-pixel limit"),
        });
    }

    let channels = match decoded.format {
        DecodedPixelFormat::L8 | DecodedPixelFormat::Rgb8 => 3_u8,
        DecodedPixelFormat::La8 | DecodedPixelFormat::Rgba8 => 4_u8,
    };

    // Reserve a realistic initial size without over-committing worst-case 5x pixel RAM.
    let desired_cap = QOI_HEADER_LEN
        .saturating_add(pixels.len() / 2)
        .saturating_add(64);
    if encoded_buf.capacity() < desired_cap {
        encoded_buf.reserve(desired_cap - encoded_buf.len());
    }

    write_qoi_header(encoded_buf, decoded.width, decoded.height, channels);

    match decoded.format {
        DecodedPixelFormat::Rgb8 => encode_rgb8_body(target, pixels, pixel_count, encoded_buf)?,
        DecodedPixelFormat::Rgba8 => encode_rgba8_body(target, pixels, pixel_count, encoded_buf)?,
        DecodedPixelFormat::L8 => encode_l8_body(target, pixels, pixel_count, encoded_buf)?,
        DecodedPixelFormat::La8 => encode_la8_body(target, pixels, pixel_count, encoded_buf)?,
    }

    encoded_buf.extend_from_slice(&QOI_END_MARKER);
    Ok(())
}

const QOI_CHUNK_PIXELS: usize = 512;
// Worst case per pixel is 1 byte (flushed pending RUN) + 5 bytes (QOI_OP_RGBA) = 6 bytes;
// 512 * 5 + 1 = 2561 <= 4096 bytes, keeping the staging buffer L1-cache-resident.
const QOI_STAGE_BYTES: usize = 4096;

fn encode_rgb8_body(
    target: &Path,
    pixels: &[u8],
    pixel_count: usize,
    out: &mut Vec<u8>,
) -> Result<(), PipelineError> {
    let (triplets, remainder) = pixels.as_chunks::<3>();
    if !remainder.is_empty() || triplets.len() != pixel_count {
        return Err(PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: format!(
                "RGB8 buffer length {} does not match {pixel_count} pixels",
                pixels.len()
            ),
        });
    }

    let mut index = [0_u32; 64];
    let mut pr = 0_u8;
    let mut pg = 0_u8;
    let mut pb = 0_u8;
    let mut prev_packed = pack_rgba(0, 0, 0, 255);
    let mut run = 0_u8;
    let mut stage = [0_u8; QOI_STAGE_BYTES];

    for chunk in triplets.chunks(QOI_CHUNK_PIXELS) {
        let mut cursor = 0_usize;
        for &[r, g, b] in chunk {
            let packed = pack_rgba(r, g, b, 255);
            if packed == prev_packed {
                run += 1;
                if run == 62 {
                    stage[cursor] = QOI_OP_RUN | (run - 1);
                    cursor += 1;
                    run = 0;
                }
                continue;
            }

            if run > 0 {
                stage[cursor] = QOI_OP_RUN | (run - 1);
                cursor += 1;
                run = 0;
            }

            let hash = qoi_hash_rgb_opaque(r, g, b);
            if index[hash] == packed {
                #[allow(clippy::cast_possible_truncation)]
                {
                    stage[cursor] = hash as u8;
                }
                cursor += 1;
            } else {
                index[hash] = packed;
                let vr = r.wrapping_sub(pr);
                let vg = g.wrapping_sub(pg);
                let vb = b.wrapping_sub(pb);

                let dr = vr.wrapping_add(2);
                let dg = vg.wrapping_add(2);
                let db = vb.wrapping_add(2);

                if (dr | dg | db) < 4 {
                    stage[cursor] = QOI_OP_DIFF | (dr << 4) | (dg << 2) | db;
                    cursor += 1;
                } else {
                    let dg_luma = vg.wrapping_add(32);
                    let red_green = vr.wrapping_sub(vg).wrapping_add(8);
                    let blue_green = vb.wrapping_sub(vg).wrapping_add(8);
                    if dg_luma < 64 && (red_green | blue_green) < 16 {
                        stage[cursor] = QOI_OP_LUMA | dg_luma;
                        stage[cursor + 1] = (red_green << 4) | blue_green;
                        cursor += 2;
                    } else {
                        stage[cursor] = QOI_OP_RGB;
                        stage[cursor + 1] = r;
                        stage[cursor + 2] = g;
                        stage[cursor + 3] = b;
                        cursor += 4;
                    }
                }
            }

            pr = r;
            pg = g;
            pb = b;
            prev_packed = packed;
        }
        out.extend_from_slice(&stage[..cursor]);
    }

    if run > 0 {
        out.push(QOI_OP_RUN | (run - 1));
    }
    Ok(())
}

fn encode_rgba8_body(
    target: &Path,
    pixels: &[u8],
    pixel_count: usize,
    out: &mut Vec<u8>,
) -> Result<(), PipelineError> {
    let (quads, remainder) = pixels.as_chunks::<4>();
    if !remainder.is_empty() || quads.len() != pixel_count {
        return Err(PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: format!(
                "RGBA8 buffer length {} does not match {pixel_count} pixels",
                pixels.len()
            ),
        });
    }

    let mut index = [0_u32; 64];
    let mut pr = 0_u8;
    let mut pg = 0_u8;
    let mut pb = 0_u8;
    let mut pa = 255_u8;
    let mut prev_packed = pack_rgba(0, 0, 0, 255);
    let mut run = 0_u8;
    let mut stage = [0_u8; QOI_STAGE_BYTES];

    for chunk in quads.chunks(QOI_CHUNK_PIXELS) {
        let mut cursor = 0_usize;
        for &[r, g, b, a] in chunk {
            let packed = pack_rgba(r, g, b, a);
            if packed == prev_packed {
                run += 1;
                if run == 62 {
                    stage[cursor] = QOI_OP_RUN | (run - 1);
                    cursor += 1;
                    run = 0;
                }
                continue;
            }

            if run > 0 {
                stage[cursor] = QOI_OP_RUN | (run - 1);
                cursor += 1;
                run = 0;
            }

            let hash = qoi_hash_rgba(r, g, b, a);
            if index[hash] == packed {
                #[allow(clippy::cast_possible_truncation)]
                {
                    stage[cursor] = hash as u8;
                }
                cursor += 1;
            } else {
                index[hash] = packed;
                if a == pa {
                    let vr = r.wrapping_sub(pr);
                    let vg = g.wrapping_sub(pg);
                    let vb = b.wrapping_sub(pb);

                    let dr = vr.wrapping_add(2);
                    let dg = vg.wrapping_add(2);
                    let db = vb.wrapping_add(2);

                    if (dr | dg | db) < 4 {
                        stage[cursor] = QOI_OP_DIFF | (dr << 4) | (dg << 2) | db;
                        cursor += 1;
                    } else {
                        let dg_luma = vg.wrapping_add(32);
                        let red_green = vr.wrapping_sub(vg).wrapping_add(8);
                        let blue_green = vb.wrapping_sub(vg).wrapping_add(8);
                        if dg_luma < 64 && (red_green | blue_green) < 16 {
                            stage[cursor] = QOI_OP_LUMA | dg_luma;
                            stage[cursor + 1] = (red_green << 4) | blue_green;
                            cursor += 2;
                        } else {
                            stage[cursor] = QOI_OP_RGB;
                            stage[cursor + 1] = r;
                            stage[cursor + 2] = g;
                            stage[cursor + 3] = b;
                            cursor += 4;
                        }
                    }
                } else {
                    stage[cursor] = QOI_OP_RGBA;
                    stage[cursor + 1] = r;
                    stage[cursor + 2] = g;
                    stage[cursor + 3] = b;
                    stage[cursor + 4] = a;
                    cursor += 5;
                }
            }

            pr = r;
            pg = g;
            pb = b;
            pa = a;
            prev_packed = packed;
        }
        out.extend_from_slice(&stage[..cursor]);
    }

    if run > 0 {
        out.push(QOI_OP_RUN | (run - 1));
    }
    Ok(())
}

fn encode_l8_body(
    target: &Path,
    pixels: &[u8],
    pixel_count: usize,
    out: &mut Vec<u8>,
) -> Result<(), PipelineError> {
    if pixels.len() != pixel_count {
        return Err(PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: format!(
                "L8 buffer length {} does not match {pixel_count} pixels",
                pixels.len()
            ),
        });
    }

    let mut index = [0_u32; 64];
    let mut prev_v = 0_u8;
    let mut prev_packed = pack_rgba(0, 0, 0, 255);
    let mut run = 0_u8;

    for &v in pixels {
        let packed = pack_rgba(v, v, v, 255);
        if packed == prev_packed {
            run += 1;
            if run == 62 {
                out.push(QOI_OP_RUN | (run - 1));
                run = 0;
            }
            continue;
        }
        if run > 0 {
            out.push(QOI_OP_RUN | (run - 1));
            run = 0;
        }

        let hash = qoi_hash_rgb_opaque(v, v, v);
        if index[hash] == packed {
            #[allow(clippy::cast_possible_truncation)]
            out.push(hash as u8);
        } else {
            index[hash] = packed;
            let dv = v.wrapping_sub(prev_v);
            let d = dv.wrapping_add(2);
            if d < 4 {
                out.push(QOI_OP_DIFF | (d << 4) | (d << 2) | d);
            } else {
                let dg_luma = dv.wrapping_add(32);
                if dg_luma < 64 {
                    out.extend_from_slice(&[QOI_OP_LUMA | dg_luma, (8 << 4) | 8]);
                } else {
                    out.extend_from_slice(&[QOI_OP_RGB, v, v, v]);
                }
            }
        }

        prev_v = v;
        prev_packed = packed;
    }

    if run > 0 {
        out.push(QOI_OP_RUN | (run - 1));
    }
    Ok(())
}

fn encode_la8_body(
    target: &Path,
    pixels: &[u8],
    pixel_count: usize,
    out: &mut Vec<u8>,
) -> Result<(), PipelineError> {
    let (pairs, remainder) = pixels.as_chunks::<2>();
    if !remainder.is_empty() || pairs.len() != pixel_count {
        return Err(PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: format!(
                "LA8 buffer length {} does not match {pixel_count} pixels",
                pixels.len()
            ),
        });
    }

    let mut index = [0_u32; 64];
    let mut prev_v = 0_u8;
    let mut prev_a = 255_u8;
    let mut prev_packed = pack_rgba(0, 0, 0, 255);
    let mut run = 0_u8;

    for &[v, a] in pairs {
        let packed = pack_rgba(v, v, v, a);
        if packed == prev_packed {
            run += 1;
            if run == 62 {
                out.push(QOI_OP_RUN | (run - 1));
                run = 0;
            }
            continue;
        }
        if run > 0 {
            out.push(QOI_OP_RUN | (run - 1));
            run = 0;
        }

        let hash = qoi_hash_rgba(v, v, v, a);
        if index[hash] == packed {
            #[allow(clippy::cast_possible_truncation)]
            out.push(hash as u8);
        } else {
            index[hash] = packed;
            if a == prev_a {
                let dv = v.wrapping_sub(prev_v);
                let d = dv.wrapping_add(2);
                if d < 4 {
                    out.push(QOI_OP_DIFF | (d << 4) | (d << 2) | d);
                } else {
                    let dg_luma = dv.wrapping_add(32);
                    if dg_luma < 64 {
                        out.extend_from_slice(&[QOI_OP_LUMA | dg_luma, (8 << 4) | 8]);
                    } else {
                        out.extend_from_slice(&[QOI_OP_RGB, v, v, v]);
                    }
                }
            } else {
                out.extend_from_slice(&[QOI_OP_RGBA, v, v, v, a]);
            }
        }

        prev_v = v;
        prev_a = a;
        prev_packed = packed;
    }

    if run > 0 {
        out.push(QOI_OP_RUN | (run - 1));
    }
    Ok(())
}

struct QoiHeader {
    width: u32,
    height: u32,
    channels: u8,
    pixel_count: usize,
}

fn parse_qoi_header(source: &Path, file_bytes: &[u8]) -> Result<QoiHeader, PipelineError> {
    if file_bytes.len() < QOI_HEADER_LEN + QOI_END_MARKER.len() {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "truncated QOI image".to_owned(),
        });
    }
    if &file_bytes[0..4] != QOI_MAGIC.as_slice() {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "invalid QOI magic bytes".to_owned(),
        });
    }
    if !file_bytes.ends_with(&QOI_END_MARKER) {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "invalid or missing QOI end marker".to_owned(),
        });
    }

    let width = u32::from_be_bytes([file_bytes[4], file_bytes[5], file_bytes[6], file_bytes[7]]);
    let height = u32::from_be_bytes([file_bytes[8], file_bytes[9], file_bytes[10], file_bytes[11]]);
    let channels = file_bytes[12];
    let colorspace = file_bytes[13];

    if width == 0 || height == 0 {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: format!("invalid QOI dimensions {width}x{height}"),
        });
    }
    if !matches!(channels, 3 | 4) {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: format!("invalid QOI channel count {channels}"),
        });
    }
    if !matches!(colorspace, 0 | 1) {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: format!("invalid QOI colorspace {colorspace}"),
        });
    }

    let pixel_count = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "QOI pixel count overflow".to_owned(),
        })?;

    Ok(QoiHeader {
        width,
        height,
        channels,
        pixel_count,
    })
}

fn prepare_qoi_output<'a>(
    source: &Path,
    pixel_count: usize,
    bpp: usize,
    pixel_buf: &'a mut Vec<u8>,
) -> Result<&'a mut [u8], PipelineError> {
    if pixel_count > MAX_QOI_PIXELS {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: format!("QOI pixel count ({pixel_count}) exceeds {MAX_QOI_PIXELS}-pixel limit"),
        });
    }
    let out_len = pixel_count
        .checked_mul(bpp)
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "QOI output buffer overflow".to_owned(),
        })?;
    if pixel_buf.len() < out_len {
        let additional = out_len - pixel_buf.len();
        pixel_buf
            .try_reserve_exact(additional)
            .map_err(|_| PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: format!("failed to reserve {out_len} bytes for decoded QOI frame"),
            })?;
        pixel_buf.resize(out_len, 0);
    }
    pixel_buf
        .get_mut(..out_len)
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "QOI output slice out of bounds".to_owned(),
        })
}

/// Decodes a QOI image slice directly into `pixel_buf`, honoring `hint`.
pub(crate) fn decode_qoi_slice_into(
    source: &Path,
    file_bytes: &[u8],
    hint: DecodeTargetHint,
    pixel_buf: &mut Vec<u8>,
) -> Result<(u32, u32, DecodedPixelFormat, usize), PipelineError> {
    let hdr = parse_qoi_header(source, file_bytes)?;
    if hdr.pixel_count > MAX_QOI_PIXELS {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: format!(
                "QOI pixel count ({}) exceeds {MAX_QOI_PIXELS}-pixel limit",
                hdr.pixel_count
            ),
        });
    }
    let stream = &file_bytes[QOI_HEADER_LEN..file_bytes.len() - QOI_END_MARKER.len()];
    // Each QOI byte encodes at most 62 pixels (QOI_OP_RUN); reject impossible streams before allocating.
    let min_stream_bytes = hdr.pixel_count.div_ceil(QOI_MAX_RUN);
    if stream.len() < min_stream_bytes {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: format!(
                "QOI bitstream ({} bytes) is too short to represent {} pixels (minimum {min_stream_bytes} bytes)",
                stream.len(),
                hdr.pixel_count
            ),
        });
    }

    match (hint, hdr.channels) {
        (DecodeTargetHint::Native, 3) => {
            let dst = prepare_qoi_output(source, hdr.pixel_count, 3, pixel_buf)?;
            let out_len = dst.len();
            decode_qoi_rgb_stream(source, stream, dst)?;
            Ok((hdr.width, hdr.height, DecodedPixelFormat::Rgb8, out_len))
        }
        (DecodeTargetHint::Native, 4) => {
            let dst = prepare_qoi_output(source, hdr.pixel_count, 4, pixel_buf)?;
            let out_len = dst.len();
            decode_qoi_rgba_stream(source, stream, dst, false)?;
            Ok((hdr.width, hdr.height, DecodedPixelFormat::Rgba8, out_len))
        }
        // QOI's header channel field is purely informative, so opaque-only callers must validate
        // decoded alpha even when `channels == 3`.
        #[cfg(feature = "nvenc")]
        (DecodeTargetHint::OpaqueRgba8, 3 | 4) => {
            let dst = prepare_qoi_output(source, hdr.pixel_count, 4, pixel_buf)?;
            let out_len = dst.len();
            decode_qoi_rgba_stream(source, stream, dst, true)?;
            Ok((hdr.width, hdr.height, DecodedPixelFormat::Rgba8, out_len))
        }
        (_, other) => Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: format!("invalid QOI channel count {other}"),
        }),
    }
}

fn decode_qoi_rgb_stream(
    source: &Path,
    stream: &[u8],
    dst: &mut [u8],
) -> Result<(), PipelineError> {
    let (dst_pixels, _) = dst.as_chunks_mut::<3>();
    let mut index = [[0_u8; 4]; 64];
    let mut px = [0_u8, 0, 0, 255];
    let mut pos = 0_usize;
    let mut out_idx = 0_usize;
    let total = dst_pixels.len();

    while out_idx < total {
        let &b1 = stream.get(pos).ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "unexpected end of QOI bitstream".to_owned(),
        })?;
        pos += 1;

        if b1 == QOI_OP_RGB {
            let rgb = stream
                .get(pos..pos + 3)
                .ok_or_else(|| PipelineError::DecodeFailed {
                    path: source.to_path_buf(),
                    reason: "truncated QOI_OP_RGB chunk".to_owned(),
                })?;
            pos += 3;
            px[0] = rgb[0];
            px[1] = rgb[1];
            px[2] = rgb[2];
            index[qoi_hash_rgba(px[0], px[1], px[2], px[3])] = px;
            dst_pixels[out_idx] = [px[0], px[1], px[2]];
            out_idx += 1;
        } else if b1 == QOI_OP_RGBA {
            let rgba = stream
                .get(pos..pos + 4)
                .ok_or_else(|| PipelineError::DecodeFailed {
                    path: source.to_path_buf(),
                    reason: "truncated QOI_OP_RGBA chunk".to_owned(),
                })?;
            pos += 4;
            px = [rgba[0], rgba[1], rgba[2], rgba[3]];
            index[qoi_hash_rgba(px[0], px[1], px[2], px[3])] = px;
            dst_pixels[out_idx] = [px[0], px[1], px[2]];
            out_idx += 1;
        } else {
            match b1 & 0xC0 {
                0x00 => {
                    px = index[usize::from(b1 & 0x3F)];
                    dst_pixels[out_idx] = [px[0], px[1], px[2]];
                    out_idx += 1;
                }
                0x40 => {
                    px[0] = px[0].wrapping_add(((b1 >> 4) & 0x03).wrapping_sub(2));
                    px[1] = px[1].wrapping_add(((b1 >> 2) & 0x03).wrapping_sub(2));
                    px[2] = px[2].wrapping_add((b1 & 0x03).wrapping_sub(2));
                    index[qoi_hash_rgba(px[0], px[1], px[2], px[3])] = px;
                    dst_pixels[out_idx] = [px[0], px[1], px[2]];
                    out_idx += 1;
                }
                0x80 => {
                    let &b2 = stream.get(pos).ok_or_else(|| PipelineError::DecodeFailed {
                        path: source.to_path_buf(),
                        reason: "truncated QOI_OP_LUMA chunk".to_owned(),
                    })?;
                    pos += 1;
                    let vg = (b1 & 0x3F).wrapping_sub(32);
                    px[0] = px[0]
                        .wrapping_add(vg)
                        .wrapping_add(((b2 >> 4) & 0x0F).wrapping_sub(8));
                    px[1] = px[1].wrapping_add(vg);
                    px[2] = px[2]
                        .wrapping_add(vg)
                        .wrapping_add((b2 & 0x0F).wrapping_sub(8));
                    index[qoi_hash_rgba(px[0], px[1], px[2], px[3])] = px;
                    dst_pixels[out_idx] = [px[0], px[1], px[2]];
                    out_idx += 1;
                }
                _ => {
                    let run = usize::from(b1 & 0x3F) + 1;
                    let end = out_idx
                        .checked_add(run)
                        .filter(|&e| e <= total)
                        .ok_or_else(|| PipelineError::DecodeFailed {
                            path: source.to_path_buf(),
                            reason: "QOI_OP_RUN exceeds image pixel count".to_owned(),
                        })?;
                    // QOI reference decoder updates the running index on QOI_OP_RUN as well.
                    index[qoi_hash_rgba(px[0], px[1], px[2], px[3])] = px;
                    dst_pixels[out_idx..end].fill([px[0], px[1], px[2]]);
                    out_idx = end;
                }
            }
        }
    }

    if pos != stream.len() {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: format!(
                "QOI bitstream has {} unexpected trailing byte(s)",
                stream.len().saturating_sub(pos)
            ),
        });
    }
    Ok(())
}

#[inline]
fn resolve_rgba_out_pixel(
    source: &Path,
    px: [u8; 4],
    reject_transparent: bool,
) -> Result<[u8; 4], PipelineError> {
    if reject_transparent && px[3] < 255 {
        Err(PipelineError::UnsupportedPixelFormat {
            path: source.to_path_buf(),
            format:
                "non-opaque alpha channel (NVENC HEVC-Intra HEIC currently supports opaque images)"
                    .to_owned(),
        })
    } else {
        Ok(px)
    }
}

fn decode_qoi_rgba_stream(
    source: &Path,
    stream: &[u8],
    dst: &mut [u8],
    reject_transparent: bool,
) -> Result<(), PipelineError> {
    let (dst_pixels, _) = dst.as_chunks_mut::<4>();
    let mut index = [[0_u8; 4]; 64];
    let mut px = [0_u8, 0, 0, 255];
    let mut pos = 0_usize;
    let mut out_idx = 0_usize;
    let total = dst_pixels.len();

    while out_idx < total {
        let &b1 = stream.get(pos).ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "unexpected end of QOI bitstream".to_owned(),
        })?;
        pos += 1;

        if b1 == QOI_OP_RGB {
            let rgb = stream
                .get(pos..pos + 3)
                .ok_or_else(|| PipelineError::DecodeFailed {
                    path: source.to_path_buf(),
                    reason: "truncated QOI_OP_RGB chunk".to_owned(),
                })?;
            pos += 3;
            px[0] = rgb[0];
            px[1] = rgb[1];
            px[2] = rgb[2];
            index[qoi_hash_rgba(px[0], px[1], px[2], px[3])] = px;
        } else if b1 == QOI_OP_RGBA {
            let rgba = stream
                .get(pos..pos + 4)
                .ok_or_else(|| PipelineError::DecodeFailed {
                    path: source.to_path_buf(),
                    reason: "truncated QOI_OP_RGBA chunk".to_owned(),
                })?;
            pos += 4;
            px = [rgba[0], rgba[1], rgba[2], rgba[3]];
            index[qoi_hash_rgba(px[0], px[1], px[2], px[3])] = px;
        } else {
            match b1 & 0xC0 {
                0x00 => {
                    px = index[usize::from(b1 & 0x3F)];
                }
                0x40 => {
                    px[0] = px[0].wrapping_add(((b1 >> 4) & 0x03).wrapping_sub(2));
                    px[1] = px[1].wrapping_add(((b1 >> 2) & 0x03).wrapping_sub(2));
                    px[2] = px[2].wrapping_add((b1 & 0x03).wrapping_sub(2));
                    index[qoi_hash_rgba(px[0], px[1], px[2], px[3])] = px;
                }
                0x80 => {
                    let &b2 = stream.get(pos).ok_or_else(|| PipelineError::DecodeFailed {
                        path: source.to_path_buf(),
                        reason: "truncated QOI_OP_LUMA chunk".to_owned(),
                    })?;
                    pos += 1;
                    let vg = (b1 & 0x3F).wrapping_sub(32);
                    px[0] = px[0]
                        .wrapping_add(vg)
                        .wrapping_add(((b2 >> 4) & 0x0F).wrapping_sub(8));
                    px[1] = px[1].wrapping_add(vg);
                    px[2] = px[2]
                        .wrapping_add(vg)
                        .wrapping_add((b2 & 0x0F).wrapping_sub(8));
                    index[qoi_hash_rgba(px[0], px[1], px[2], px[3])] = px;
                }
                _ => {
                    let run = usize::from(b1 & 0x3F) + 1;
                    let end = out_idx
                        .checked_add(run)
                        .filter(|&e| e <= total)
                        .ok_or_else(|| PipelineError::DecodeFailed {
                            path: source.to_path_buf(),
                            reason: "QOI_OP_RUN exceeds image pixel count".to_owned(),
                        })?;
                    index[qoi_hash_rgba(px[0], px[1], px[2], px[3])] = px;
                    let out_px = resolve_rgba_out_pixel(source, px, reject_transparent)?;
                    dst_pixels[out_idx..end].fill(out_px);
                    out_idx = end;
                    continue;
                }
            }
        }

        dst_pixels[out_idx] = resolve_rgba_out_pixel(source, px, reject_transparent)?;
        out_idx += 1;
    }

    if pos != stream.len() {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: format!(
                "QOI bitstream has {} unexpected trailing byte(s)",
                stream.len().saturating_sub(pos)
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{decode_qoi_slice_into, encode_qoi_into};
    use crate::decode::{DecodeTargetHint, DecodedFrameMeta, DecodedPixelFormat};
    use std::path::Path;

    #[test]
    fn roundtrips_rgb_rgba_luma_and_la_losslessly() -> Result<(), Box<dyn std::error::Error>> {
        let p = Path::new("test.qoi");

        // Exercise RUN, INDEX, DIFF, LUMA, RGB, and RGBA opcodes plus leading black pixel.
        let mut rgba = Vec::with_capacity(64 * 16 * 4);
        for y in 0_u32..16 {
            for x in 0_u32..64 {
                let px = if y == 0 && x < 10 {
                    [0, 0, 0, 255]
                } else if y == 1 && x < 5 {
                    [0, 0, 0, 0]
                } else if y == 2 {
                    [
                        u8::try_from((x * 2) % 256).unwrap_or(0),
                        u8::try_from((x * 2) % 256).unwrap_or(0),
                        u8::try_from((x * 2) % 256).unwrap_or(0),
                        255,
                    ]
                } else {
                    [
                        u8::try_from((x * 13 + y * 5) % 256).unwrap_or(0),
                        u8::try_from((y * 19 + x * 3) % 256).unwrap_or(0),
                        u8::try_from(((x ^ y) * 7) % 256).unwrap_or(0),
                        if x % 5 == 0 { 180 } else { 255 },
                    ]
                };
                rgba.extend_from_slice(&px);
            }
        }

        let mut encoded = Vec::new();
        encode_qoi_into(
            p,
            &rgba,
            DecodedFrameMeta {
                width: 64,
                height: 16,
                format: DecodedPixelFormat::Rgba8,
                byte_len: rgba.len(),
                input_file_bytes: 1,
            },
            &mut encoded,
        )?;

        let mut decoded = Vec::new();
        let (w, h, fmt, len) =
            decode_qoi_slice_into(p, &encoded, DecodeTargetHint::Native, &mut decoded)?;
        assert_eq!(
            (w, h, fmt, len),
            (64, 16, DecodedPixelFormat::Rgba8, rgba.len())
        );
        assert_eq!(&decoded[..len], rgba.as_slice());

        let rgb: Vec<u8> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|&[r, g, b, _]| [r, g, b])
            .collect();
        encoded.clear();
        encode_qoi_into(
            p,
            &rgb,
            DecodedFrameMeta {
                width: 64,
                height: 16,
                format: DecodedPixelFormat::Rgb8,
                byte_len: rgb.len(),
                input_file_bytes: 1,
            },
            &mut encoded,
        )?;
        let (w, h, fmt, len) =
            decode_qoi_slice_into(p, &encoded, DecodeTargetHint::Native, &mut decoded)?;
        assert_eq!(
            (w, h, fmt, len),
            (64, 16, DecodedPixelFormat::Rgb8, rgb.len())
        );
        assert_eq!(&decoded[..len], rgb.as_slice());

        let luma: Vec<u8> = (0_u8..=255).collect();
        encoded.clear();
        encode_qoi_into(
            p,
            &luma,
            DecodedFrameMeta {
                width: 16,
                height: 16,
                format: DecodedPixelFormat::L8,
                byte_len: luma.len(),
                input_file_bytes: 1,
            },
            &mut encoded,
        )?;
        let (w, h, fmt, len) =
            decode_qoi_slice_into(p, &encoded, DecodeTargetHint::Native, &mut decoded)?;
        assert_eq!((w, h, fmt), (16, 16, DecodedPixelFormat::Rgb8));
        let expected_luma_rgb: Vec<u8> = luma.iter().flat_map(|&v| [v, v, v]).collect();
        assert_eq!(&decoded[..len], expected_luma_rgb.as_slice());

        Ok(())
    }

    #[test]
    fn rejects_truncated_or_overflowing_qoi_streams() {
        let p = Path::new("bad.qoi");
        let mut buf = Vec::new();
        assert!(decode_qoi_slice_into(p, b"qoif", DecodeTargetHint::Native, &mut buf).is_err());

        // Header declares 2x2 RGB (4 pixels), single RUN of 10 pixels -> must error.
        let mut overflow_run = Vec::new();
        overflow_run.extend_from_slice(b"qoif");
        overflow_run.extend_from_slice(&2_u32.to_be_bytes());
        overflow_run.extend_from_slice(&2_u32.to_be_bytes());
        overflow_run.extend_from_slice(&[3, 0, 0xC0 | 9, 0, 0, 0, 0, 0, 0, 0, 1]);
        assert!(
            decode_qoi_slice_into(p, &overflow_run, DecodeTargetHint::Native, &mut buf).is_err()
        );

        // Short QOI file declaring 1_000_000 x 1_000_000 must fail before allocation.
        let mut huge_short = Vec::new();
        huge_short.extend_from_slice(b"qoif");
        huge_short.extend_from_slice(&1_000_000_u32.to_be_bytes());
        huge_short.extend_from_slice(&1_000_000_u32.to_be_bytes());
        huge_short.extend_from_slice(&[4, 0, 0xC0, 0, 0, 0, 0, 0, 0, 0, 1]);
        assert!(matches!(
            decode_qoi_slice_into(p, &huge_short, DecodeTargetHint::Native, &mut buf),
            Err(crate::PipelineError::DecodeFailed { .. })
        ));

        // Encoder enforces the same pixel limit as probe/decoder (12000x12000 = 144M > 128 Mi).
        let mut encoded = Vec::new();
        assert!(matches!(
            encode_qoi_into(
                p,
                &[],
                DecodedFrameMeta {
                    width: 12_000,
                    height: 12_000,
                    format: DecodedPixelFormat::Rgb8,
                    byte_len: 0,
                    input_file_bytes: 1,
                },
                &mut encoded,
            ),
            Err(crate::PipelineError::EncodeFailed { .. })
        ));
    }

    #[test]
    fn updates_color_index_on_initial_run_and_validates_alpha_regardless_of_header_channels()
    -> Result<(), Box<dyn std::error::Error>> {
        let p = Path::new("hand_authored.qoi");
        // Hand-authored 3x1 QOI:
        //   px0 = QOI_OP_RUN(1) -> [0, 0, 0, 255] (hashes to slot 53, must update index[53])
        //   px1 = QOI_OP_RGB(255, 0, 0) -> [255, 0, 0, 255]
        //   px2 = QOI_OP_INDEX(53) -> must recall [0, 0, 0, 255], NOT transparent [0, 0, 0, 0]
        let mut qoi_rgba = Vec::new();
        qoi_rgba.extend_from_slice(b"qoif");
        qoi_rgba.extend_from_slice(&3_u32.to_be_bytes());
        qoi_rgba.extend_from_slice(&1_u32.to_be_bytes());
        qoi_rgba.extend_from_slice(&[4, 0]);
        qoi_rgba.extend_from_slice(&[0xC0, 0xFE, 255, 0, 0, 53]);
        qoi_rgba.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]);

        let mut buf = Vec::new();
        let (w, h, fmt, len) =
            decode_qoi_slice_into(p, &qoi_rgba, DecodeTargetHint::Native, &mut buf)?;
        assert_eq!((w, h, fmt, len), (3, 1, DecodedPixelFormat::Rgba8, 12));
        assert_eq!(&buf[..len], &[0, 0, 0, 255, 255, 0, 0, 255, 0, 0, 0, 255]);

        // Same hand-authored stream with channels=3.
        qoi_rgba[12] = 3;
        let (w, h, fmt, len) =
            decode_qoi_slice_into(p, &qoi_rgba, DecodeTargetHint::Native, &mut buf)?;
        assert_eq!((w, h, fmt, len), (3, 1, DecodedPixelFormat::Rgb8, 9));
        assert_eq!(&buf[..len], &[0, 0, 0, 255, 0, 0, 0, 0, 0]);

        #[cfg(feature = "nvenc")]
        {
            // Opaque 3-channel stream succeeds under OpaqueRgba8.
            let (w, h, fmt, len) =
                decode_qoi_slice_into(p, &qoi_rgba, DecodeTargetHint::OpaqueRgba8, &mut buf)?;
            assert_eq!((w, h, fmt, len), (3, 1, DecodedPixelFormat::Rgba8, 12));
            assert_eq!(&buf[..len], &[0, 0, 0, 255, 255, 0, 0, 255, 0, 0, 0, 255]);

            // 3-channel QOI header with a QOI_OP_RGBA chunk supplying alpha=128 must be rejected.
            let mut sneak_alpha = Vec::new();
            sneak_alpha.extend_from_slice(b"qoif");
            sneak_alpha.extend_from_slice(&1_u32.to_be_bytes());
            sneak_alpha.extend_from_slice(&1_u32.to_be_bytes());
            sneak_alpha.extend_from_slice(&[3, 0, 0xFF, 10, 20, 30, 128, 0, 0, 0, 0, 0, 0, 0, 1]);
            assert!(matches!(
                decode_qoi_slice_into(p, &sneak_alpha, DecodeTargetHint::OpaqueRgba8, &mut buf),
                Err(crate::PipelineError::UnsupportedPixelFormat { .. })
            ));
        }

        Ok(())
    }
}
