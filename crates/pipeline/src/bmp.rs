use std::io::Write;
use std::path::Path;

use crate::PipelineError;
use crate::decode::{DecodeTargetHint, DecodedFrameMeta, DecodedPixelFormat};

const BMP_FILE_HEADER_LEN: usize = 14;
const BMP_INFO_HEADER_LEN: usize = 40;
const BMP_V4_HEADER_LEN: usize = 108;
// Flush BMP scanlines in 2 MiB contiguous chunks directly to the OS file handle so
// parallel workers avoid both BufWriter double-copies and filesystem extent contention.
const BMP_BATCH_BUFFER_BYTES: usize = 2 * 1024 * 1024;

#[inline]
const fn is_contiguous_mask(mask: u32) -> bool {
    if mask == 0 {
        return true;
    }
    let shifted = mask >> mask.trailing_zeros();
    (shifted & shifted.wrapping_add(1)) == 0
}

#[inline]
fn swizzle_rgb24_to_bgr24(src_row: &[u8], dst_row: &mut [u8]) {
    let (src_blocks, src_tail) = src_row.as_chunks::<12>();
    let (dst_blocks, dst_tail) = dst_row.as_chunks_mut::<12>();
    for (s, d) in src_blocks.iter().zip(dst_blocks.iter_mut()) {
        *d = [
            s[2], s[1], s[0], s[5], s[4], s[3], s[8], s[7], s[6], s[11], s[10], s[9],
        ];
    }
    let (s3, _) = src_tail.as_chunks::<3>();
    let (d3, _) = dst_tail.as_chunks_mut::<3>();
    for (&[r, g, b], dst) in s3.iter().zip(d3.iter_mut()) {
        *dst = [b, g, r];
    }
}

#[inline]
fn flush_slice<W: Write>(target: &Path, bytes: &[u8], writer: &mut W) -> Result<(), PipelineError> {
    if !bytes.is_empty() {
        writer
            .write_all(bytes)
            .map_err(|source| PipelineError::Io {
                path: target.to_path_buf(),
                source,
            })?;
    }
    Ok(())
}

/// Streams decoded 8-bit pixels as a spec-compliant BMP directly into `writer` using a
/// single reusable scanline buffer (`row_scratch`) to minimize memory overhead.
pub(crate) fn encode_bmp_to_writer<W: Write>(
    target: &Path,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    row_scratch: &mut Vec<u8>,
    writer: &mut W,
) -> Result<u64, PipelineError> {
    let width_i32 = i32::try_from(decoded.width).map_err(|_| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: format!("BMP width exceeds i32::MAX: {}", decoded.width),
    })?;
    let height_i32 = i32::try_from(decoded.height).map_err(|_| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: format!("BMP height exceeds i32::MAX: {}", decoded.height),
    })?;
    let width = usize::try_from(decoded.width).map_err(|_| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: "BMP width exceeds addressable memory".to_owned(),
    })?;
    let height = usize::try_from(decoded.height).map_err(|_| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: "BMP height exceeds addressable memory".to_owned(),
    })?;

    match decoded.format {
        DecodedPixelFormat::Rgb8 => encode_bmp_rgb24(
            target,
            pixels,
            width,
            height,
            width_i32,
            height_i32,
            row_scratch,
            writer,
        ),
        DecodedPixelFormat::Rgba8 => encode_bmp_bgra32(
            target,
            pixels,
            width,
            height,
            width_i32,
            height_i32,
            false,
            row_scratch,
            writer,
        ),
        DecodedPixelFormat::L8 => encode_bmp_gray8(
            target,
            pixels,
            width,
            height,
            width_i32,
            height_i32,
            row_scratch,
            writer,
        ),
        DecodedPixelFormat::La8 => encode_bmp_bgra32(
            target,
            pixels,
            width,
            height,
            width_i32,
            height_i32,
            true,
            row_scratch,
            writer,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_bmp_rgb24<W: Write>(
    target: &Path,
    pixels: &[u8],
    width: usize,
    height: usize,
    width_i32: i32,
    height_i32: i32,
    row_scratch: &mut Vec<u8>,
    writer: &mut W,
) -> Result<u64, PipelineError> {
    let src_row_bytes = width
        .checked_mul(3)
        .ok_or_else(|| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: "BMP row size overflow".to_owned(),
        })?;
    let expected_len =
        src_row_bytes
            .checked_mul(height)
            .ok_or_else(|| PipelineError::EncodeFailed {
                path: target.to_path_buf(),
                reason: "BMP pixel size overflow".to_owned(),
            })?;
    if pixels.len() != expected_len {
        return Err(PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: format!(
                "RGB8 pixel buffer length {} does not match expected {expected_len}",
                pixels.len()
            ),
        });
    }

    let row_stride = src_row_bytes.div_ceil(4) * 4;
    let pixel_data_bytes =
        row_stride
            .checked_mul(height)
            .ok_or_else(|| PipelineError::EncodeFailed {
                path: target.to_path_buf(),
                reason: "BMP pixel payload overflow".to_owned(),
            })?;
    let header_len = BMP_FILE_HEADER_LEN + BMP_INFO_HEADER_LEN;
    let total_bytes =
        header_len
            .checked_add(pixel_data_bytes)
            .ok_or_else(|| PipelineError::EncodeFailed {
                path: target.to_path_buf(),
                reason: "BMP file size overflow".to_owned(),
            })?;
    let file_size_u32 = u32::try_from(total_bytes).map_err(|_| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: format!("BMP file size {total_bytes} exceeds 4 GiB BMP format limit"),
    })?;
    let image_size_u32 =
        u32::try_from(pixel_data_bytes).map_err(|_| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: "BMP image payload exceeds u32::MAX".to_owned(),
        })?;

    let mut header = [0_u8; BMP_FILE_HEADER_LEN + BMP_INFO_HEADER_LEN];
    header[0..2].copy_from_slice(b"BM");
    header[2..6].copy_from_slice(&file_size_u32.to_le_bytes());
    header[10..14].copy_from_slice(&54_u32.to_le_bytes());
    header[14..18].copy_from_slice(&40_u32.to_le_bytes());
    header[18..22].copy_from_slice(&width_i32.to_le_bytes());
    header[22..26].copy_from_slice(&height_i32.to_le_bytes());
    header[26..28].copy_from_slice(&1_u16.to_le_bytes());
    header[28..30].copy_from_slice(&24_u16.to_le_bytes());
    header[34..38].copy_from_slice(&image_size_u32.to_le_bytes());
    let batch_cap = total_bytes.min(BMP_BATCH_BUFFER_BYTES.max(header_len + row_stride));
    if row_scratch.len() < batch_cap {
        row_scratch.resize(batch_cap, 0);
    }
    let buf = &mut row_scratch[..batch_cap];
    buf[..header_len].copy_from_slice(&header);
    let mut cursor = header_len;

    for row in pixels.chunks_exact(src_row_bytes).rev() {
        if cursor + row_stride > batch_cap {
            flush_slice(target, &buf[..cursor], writer)?;
            cursor = 0;
        }
        swizzle_rgb24_to_bgr24(row, &mut buf[cursor..cursor + src_row_bytes]);
        if row_stride > src_row_bytes {
            buf[cursor + src_row_bytes..cursor + row_stride].fill(0);
        }
        cursor += row_stride;
    }
    flush_slice(target, &buf[..cursor], writer)?;

    Ok(u64::from(file_size_u32))
}

#[allow(clippy::too_many_arguments)]
fn encode_bmp_bgra32<W: Write>(
    target: &Path,
    pixels: &[u8],
    width: usize,
    height: usize,
    width_i32: i32,
    height_i32: i32,
    from_la8: bool,
    row_scratch: &mut Vec<u8>,
    writer: &mut W,
) -> Result<u64, PipelineError> {
    let src_bpp = if from_la8 { 2_usize } else { 4_usize };
    let src_row_bytes = width
        .checked_mul(src_bpp)
        .ok_or_else(|| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: "BMP row size overflow".to_owned(),
        })?;
    let expected_len =
        src_row_bytes
            .checked_mul(height)
            .ok_or_else(|| PipelineError::EncodeFailed {
                path: target.to_path_buf(),
                reason: "BMP pixel size overflow".to_owned(),
            })?;
    if pixels.len() != expected_len {
        return Err(PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: format!(
                "pixel buffer length {} does not match expected {expected_len}",
                pixels.len()
            ),
        });
    }

    let row_stride = width
        .checked_mul(4)
        .ok_or_else(|| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: "BMP BGRA row size overflow".to_owned(),
        })?;
    let pixel_data_bytes =
        row_stride
            .checked_mul(height)
            .ok_or_else(|| PipelineError::EncodeFailed {
                path: target.to_path_buf(),
                reason: "BMP pixel payload overflow".to_owned(),
            })?;
    let header_len = BMP_FILE_HEADER_LEN + BMP_V4_HEADER_LEN;
    let total_bytes =
        header_len
            .checked_add(pixel_data_bytes)
            .ok_or_else(|| PipelineError::EncodeFailed {
                path: target.to_path_buf(),
                reason: "BMP file size overflow".to_owned(),
            })?;
    let file_size_u32 = u32::try_from(total_bytes).map_err(|_| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: format!("BMP file size {total_bytes} exceeds 4 GiB BMP format limit"),
    })?;
    let image_size_u32 =
        u32::try_from(pixel_data_bytes).map_err(|_| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: "BMP image payload exceeds u32::MAX".to_owned(),
        })?;

    // BITMAPV4HEADER (108 bytes) with BI_BITFIELDS explicitly declaring BGRA channel masks + sRGB.
    let mut header = [0_u8; BMP_FILE_HEADER_LEN + BMP_V4_HEADER_LEN];
    header[0..2].copy_from_slice(b"BM");
    header[2..6].copy_from_slice(&file_size_u32.to_le_bytes());
    header[10..14].copy_from_slice(&122_u32.to_le_bytes());
    header[14..18].copy_from_slice(&108_u32.to_le_bytes());
    header[18..22].copy_from_slice(&width_i32.to_le_bytes());
    header[22..26].copy_from_slice(&height_i32.to_le_bytes());
    header[26..28].copy_from_slice(&1_u16.to_le_bytes());
    header[28..30].copy_from_slice(&32_u16.to_le_bytes());
    header[30..34].copy_from_slice(&3_u32.to_le_bytes()); // BI_BITFIELDS
    header[34..38].copy_from_slice(&image_size_u32.to_le_bytes());
    header[54..58].copy_from_slice(&0x00FF_0000_u32.to_le_bytes()); // R
    header[58..62].copy_from_slice(&0x0000_FF00_u32.to_le_bytes()); // G
    header[62..66].copy_from_slice(&0x0000_00FF_u32.to_le_bytes()); // B
    header[66..70].copy_from_slice(&0xFF00_0000_u32.to_le_bytes()); // A
    header[70..74].copy_from_slice(b"BGRs"); // LCS_sRGB
    let batch_cap = total_bytes.min(BMP_BATCH_BUFFER_BYTES.max(header_len + row_stride));
    if row_scratch.len() < batch_cap {
        row_scratch.resize(batch_cap, 0);
    }
    let buf = &mut row_scratch[..batch_cap];
    buf[..header_len].copy_from_slice(&header);
    let mut cursor = header_len;

    for row in pixels.chunks_exact(src_row_bytes).rev() {
        if cursor + row_stride > batch_cap {
            flush_slice(target, &buf[..cursor], writer)?;
            cursor = 0;
        }
        let (dst_quads, _) = buf[cursor..cursor + row_stride].as_chunks_mut::<4>();
        if from_la8 {
            let (src_pairs, _) = row.as_chunks::<2>();
            for (&[v, a], dst) in src_pairs.iter().zip(dst_quads.iter_mut()) {
                *dst = [v, v, v, a];
            }
        } else {
            let (src_quads, _) = row.as_chunks::<4>();
            for (&[r, g, b, a], dst) in src_quads.iter().zip(dst_quads.iter_mut()) {
                *dst = [b, g, r, a];
            }
        }
        cursor += row_stride;
    }
    flush_slice(target, &buf[..cursor], writer)?;

    Ok(u64::from(file_size_u32))
}

#[allow(clippy::too_many_arguments)]
fn encode_bmp_gray8<W: Write>(
    target: &Path,
    pixels: &[u8],
    width: usize,
    height: usize,
    width_i32: i32,
    height_i32: i32,
    row_scratch: &mut Vec<u8>,
    writer: &mut W,
) -> Result<u64, PipelineError> {
    let expected_len = width
        .checked_mul(height)
        .ok_or_else(|| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: "BMP grayscale pixel size overflow".to_owned(),
        })?;
    if pixels.len() != expected_len {
        return Err(PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: format!(
                "L8 pixel buffer length {} does not match expected {expected_len}",
                pixels.len()
            ),
        });
    }

    let palette_bytes = 256 * 4;
    let pixel_offset = BMP_FILE_HEADER_LEN + BMP_INFO_HEADER_LEN + palette_bytes;
    let row_stride = width.div_ceil(4) * 4;
    let pixel_data_bytes =
        row_stride
            .checked_mul(height)
            .ok_or_else(|| PipelineError::EncodeFailed {
                path: target.to_path_buf(),
                reason: "BMP pixel payload overflow".to_owned(),
            })?;
    let total_bytes =
        pixel_offset
            .checked_add(pixel_data_bytes)
            .ok_or_else(|| PipelineError::EncodeFailed {
                path: target.to_path_buf(),
                reason: "BMP file size overflow".to_owned(),
            })?;
    let file_size_u32 = u32::try_from(total_bytes).map_err(|_| PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: format!("BMP file size {total_bytes} exceeds 4 GiB BMP format limit"),
    })?;
    let image_size_u32 =
        u32::try_from(pixel_data_bytes).map_err(|_| PipelineError::EncodeFailed {
            path: target.to_path_buf(),
            reason: "BMP image payload exceeds u32::MAX".to_owned(),
        })?;
    let offset_u32 = u32::try_from(pixel_offset).unwrap_or(1078);

    let mut header = [0_u8; BMP_FILE_HEADER_LEN + BMP_INFO_HEADER_LEN];
    header[0..2].copy_from_slice(b"BM");
    header[2..6].copy_from_slice(&file_size_u32.to_le_bytes());
    header[10..14].copy_from_slice(&offset_u32.to_le_bytes());
    header[14..18].copy_from_slice(&40_u32.to_le_bytes());
    header[18..22].copy_from_slice(&width_i32.to_le_bytes());
    header[22..26].copy_from_slice(&height_i32.to_le_bytes());
    header[26..28].copy_from_slice(&1_u16.to_le_bytes());
    header[28..30].copy_from_slice(&8_u16.to_le_bytes());
    header[34..38].copy_from_slice(&image_size_u32.to_le_bytes());
    header[46..50].copy_from_slice(&256_u32.to_le_bytes());
    let batch_cap = total_bytes.min(BMP_BATCH_BUFFER_BYTES.max(pixel_offset + row_stride));
    if row_scratch.len() < batch_cap {
        row_scratch.resize(batch_cap, 0);
    }
    let buf = &mut row_scratch[..batch_cap];
    let header_len = BMP_FILE_HEADER_LEN + BMP_INFO_HEADER_LEN;
    buf[..header_len].copy_from_slice(&header);
    for i in 0_usize..256 {
        #[allow(clippy::cast_possible_truncation)]
        let v = i as u8;
        let base = header_len + i * 4;
        buf[base] = v;
        buf[base + 1] = v;
        buf[base + 2] = v;
        buf[base + 3] = 0;
    }
    let mut cursor = pixel_offset;

    for row in pixels.chunks_exact(width).rev() {
        if cursor + row_stride > batch_cap {
            flush_slice(target, &buf[..cursor], writer)?;
            cursor = 0;
        }
        buf[cursor..cursor + width].copy_from_slice(row);
        if row_stride > width {
            buf[cursor + width..cursor + row_stride].fill(0);
        }
        cursor += row_stride;
    }
    flush_slice(target, &buf[..cursor], writer)?;

    Ok(u64::from(file_size_u32))
}

#[derive(Clone, Copy)]
struct BitMask {
    mask: u32,
    shift: u32,
    bits: u32,
}

impl BitMask {
    const fn from_mask(mask: u32) -> Self {
        if mask == 0 {
            return Self {
                mask: 0,
                shift: 0,
                bits: 0,
            };
        }
        let shift = mask.trailing_zeros();
        let bits = (mask >> shift).trailing_ones();
        Self { mask, shift, bits }
    }

    #[inline]
    const fn is_opaque(self, value: u32) -> bool {
        self.mask == 0 || (value & self.mask) == self.mask
    }

    #[inline]
    fn extract_u8(self, value: u32) -> u8 {
        if self.mask == 0 || self.bits == 0 {
            return 0;
        }
        let raw = (value & self.mask) >> self.shift;
        if self.bits == 8 {
            #[allow(clippy::cast_possible_truncation)]
            return raw as u8;
        }
        let max = (1_u64 << self.bits) - 1;
        let scaled = (u64::from(raw) * 255 + max / 2) / max;
        u8::try_from(scaled).unwrap_or(255)
    }
}

struct ParsedBmp<'a> {
    width: u32,
    height: u32,
    top_down: bool,
    bit_count: u16,
    compression: u32,
    dib_size: u32,
    colors_used: u32,
    row_stride: usize,
    pixel_data: &'a [u8],
    palette_data: &'a [u8],
    masks: [BitMask; 4],
}

struct BmpDibInfo {
    width: u32,
    height: u32,
    top_down: bool,
    bit_count: u16,
    compression: u32,
    colors_used: u32,
}

fn parse_bmp_dib_info(
    source: &Path,
    data: &[u8],
    dib_size: u32,
) -> Result<BmpDibInfo, PipelineError> {
    if dib_size == 12 {
        let w = u32::from(u16::from_le_bytes([data[18], data[19]]));
        let h = u32::from(u16::from_le_bytes([data[20], data[21]]));
        let planes = u16::from_le_bytes([data[22], data[23]]);
        let bpp = u16::from_le_bytes([data[24], data[25]]);
        if planes != 1 || w == 0 || h == 0 {
            return Err(PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: "invalid BITMAPCOREHEADER dimensions or planes".to_owned(),
            });
        }
        return Ok(BmpDibInfo {
            width: w,
            height: h,
            top_down: false,
            bit_count: bpp,
            compression: 0,
            colors_used: 0,
        });
    }

    if data.len() < 54 {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "truncated BITMAPINFOHEADER".to_owned(),
        });
    }
    let raw_w = i32::from_le_bytes([data[18], data[19], data[20], data[21]]);
    let raw_h = i32::from_le_bytes([data[22], data[23], data[24], data[25]]);
    let planes = u16::from_le_bytes([data[26], data[27]]);
    let bpp = u16::from_le_bytes([data[28], data[29]]);
    let comp = u32::from_le_bytes([data[30], data[31], data[32], data[33]]);
    let colors_used = u32::from_le_bytes([data[46], data[47], data[48], data[49]]);
    if raw_w <= 0 || raw_h == 0 || raw_h == i32::MIN || planes != 1 {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: format!("invalid BMP dimensions {raw_w}x{raw_h} or planes {planes}"),
        });
    }
    let width = u32::try_from(raw_w).map_err(|_| PipelineError::DecodeFailed {
        path: source.to_path_buf(),
        reason: format!("invalid BMP width {raw_w}"),
    })?;
    Ok(BmpDibInfo {
        width,
        height: raw_h.unsigned_abs(),
        top_down: raw_h < 0,
        bit_count: bpp,
        compression: comp,
        colors_used,
    })
}

fn parse_bmp<'a>(source: &Path, data: &'a [u8]) -> Result<ParsedBmp<'a>, PipelineError> {
    if data.len() < 26 || &data[0..2] != b"BM" {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "invalid or truncated BMP header".to_owned(),
        });
    }
    let pixel_offset =
        usize::try_from(u32::from_le_bytes([data[10], data[11], data[12], data[13]])).map_err(
            |_| PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: "BMP pixel offset overflow".to_owned(),
            },
        )?;
    let dib_size = u32::from_le_bytes([data[14], data[15], data[16], data[17]]);
    let header_end = BMP_FILE_HEADER_LEN
        .checked_add(usize::try_from(dib_size).unwrap_or(usize::MAX))
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "BMP DIB header size overflow".to_owned(),
        })?;
    if !matches!(dib_size, 12 | 40 | 52 | 56 | 108 | 124)
        || pixel_offset < header_end
        || pixel_offset > data.len()
    {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: format!("invalid BMP DIB size ({dib_size}) or pixel offset ({pixel_offset})"),
        });
    }

    let dib = parse_bmp_dib_info(source, data, dib_size)?;

    let row_bits = u64::from(dib.width)
        .checked_mul(u64::from(dib.bit_count))
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "BMP row bit count overflow".to_owned(),
        })?;
    let row_stride =
        usize::try_from(row_bits.div_ceil(32) * 4).map_err(|_| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "BMP row stride overflow".to_owned(),
        })?;
    let required_pixels = row_stride
        .checked_mul(usize::try_from(dib.height).unwrap_or(usize::MAX))
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "BMP pixel data size overflow".to_owned(),
        })?;
    let pixel_end = pixel_offset
        .checked_add(required_pixels)
        .filter(|&end| end <= data.len())
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "truncated BMP pixel payload".to_owned(),
        })?;

    let masks = parse_bmp_masks(source, data, dib_size, pixel_offset, dib.compression)?;

    Ok(ParsedBmp {
        width: dib.width,
        height: dib.height,
        top_down: dib.top_down,
        bit_count: dib.bit_count,
        compression: dib.compression,
        dib_size,
        colors_used: dib.colors_used,
        row_stride,
        pixel_data: &data[pixel_offset..pixel_end],
        palette_data: &data[header_end..pixel_offset],
        masks,
    })
}

fn parse_bmp_masks(
    source: &Path,
    data: &[u8],
    dib_size: u32,
    pixel_offset: usize,
    compression: u32,
) -> Result<[BitMask; 4], PipelineError> {
    if !matches!(compression, 3 | 6) {
        return Ok([BitMask::from_mask(0); 4]);
    }
    let has_alpha_mask = compression == 6 || dib_size >= 56;
    let needed = if has_alpha_mask { 16_usize } else { 12_usize };
    if 54 + needed > pixel_offset || 54 + needed > data.len() {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "truncated BMP BI_BITFIELDS masks".to_owned(),
        });
    }
    let r = u32::from_le_bytes([data[54], data[55], data[56], data[57]]);
    let g = u32::from_le_bytes([data[58], data[59], data[60], data[61]]);
    let b = u32::from_le_bytes([data[62], data[63], data[64], data[65]]);
    let a = if has_alpha_mask {
        u32::from_le_bytes([data[66], data[67], data[68], data[69]])
    } else {
        0
    };
    if r == 0
        || g == 0
        || b == 0
        || !is_contiguous_mask(r)
        || !is_contiguous_mask(g)
        || !is_contiguous_mask(b)
        || !is_contiguous_mask(a)
    {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "invalid BMP BI_BITFIELDS channel mask".to_owned(),
        });
    }
    Ok([
        BitMask::from_mask(r),
        BitMask::from_mask(g),
        BitMask::from_mask(b),
        BitMask::from_mask(a),
    ])
}

/// Decodes a BMP image slice directly into `pixel_buf`, honoring `hint`.
pub(crate) fn decode_bmp_slice_into(
    source: &Path,
    file_bytes: &[u8],
    hint: DecodeTargetHint,
    pixel_buf: &mut Vec<u8>,
) -> Result<(u32, u32, DecodedPixelFormat, usize), PipelineError> {
    let bmp = parse_bmp(source, file_bytes)?;
    let width = usize::try_from(bmp.width).map_err(|_| PipelineError::DecodeFailed {
        path: source.to_path_buf(),
        reason: "BMP width overflow".to_owned(),
    })?;
    let height = usize::try_from(bmp.height).map_err(|_| PipelineError::DecodeFailed {
        path: source.to_path_buf(),
        reason: "BMP height overflow".to_owned(),
    })?;
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "BMP pixel count overflow".to_owned(),
        })?;

    match (bmp.compression, bmp.bit_count) {
        (0, 24) => decode_bmp_24(source, &bmp, width, height, pixel_count, hint, pixel_buf),
        // Per BMP BITMAPINFOHEADER spec, the high byte of 32-bit BI_RGB pixels is unused padding.
        (0, 32) => decode_bmp_32_bgrx(source, &bmp, width, height, pixel_count, hint, pixel_buf),
        (3 | 6, 32) => {
            decode_bmp_32_bitfields(source, &bmp, width, height, pixel_count, hint, pixel_buf)
        }
        (0 | 3 | 6, 16) => decode_bmp_16(source, &bmp, width, height, pixel_count, hint, pixel_buf),
        (0, 1 | 4 | 8) => decode_bmp_indexed(source, &bmp, width, height, pixel_count, pixel_buf),
        (comp, bpp) => Err(PipelineError::UnsupportedPixelFormat {
            path: source.to_path_buf(),
            format: format!("BMP compression={comp} bpp={bpp}"),
        }),
    }
}

fn decode_bmp_24(
    source: &Path,
    bmp: &ParsedBmp<'_>,
    width: usize,
    height: usize,
    pixel_count: usize,
    hint: DecodeTargetHint,
    pixel_buf: &mut Vec<u8>,
) -> Result<(u32, u32, DecodedPixelFormat, usize), PipelineError> {
    #[cfg(feature = "nvenc")]
    if hint == DecodeTargetHint::OpaqueRgba8 {
        let out_len = pixel_count
            .checked_mul(4)
            .ok_or_else(|| PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: "BMP RGBA8 output overflow".to_owned(),
            })?;
        if pixel_buf.len() < out_len {
            pixel_buf.resize(out_len, 0);
        }
        let dst_row_bytes = width * 4;
        for dst_y in 0..height {
            let src_y = if bmp.top_down {
                dst_y
            } else {
                height - 1 - dst_y
            };
            let src_row = &bmp.pixel_data[src_y * bmp.row_stride..][..width * 3];
            let dst_row = &mut pixel_buf[dst_y * dst_row_bytes..(dst_y + 1) * dst_row_bytes];
            let (src_triplets, _) = src_row.as_chunks::<3>();
            let (dst_quads, _) = dst_row.as_chunks_mut::<4>();
            for (&[b, g, r], dst) in src_triplets.iter().zip(dst_quads.iter_mut()) {
                *dst = [r, g, b, 255];
            }
        }
        return Ok((bmp.width, bmp.height, DecodedPixelFormat::Rgba8, out_len));
    }

    let _ = hint;
    let out_len = pixel_count
        .checked_mul(3)
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "BMP RGB8 output overflow".to_owned(),
        })?;
    if pixel_buf.len() < out_len {
        pixel_buf.resize(out_len, 0);
    }
    let dst_row_bytes = width * 3;
    for dst_y in 0..height {
        let src_y = if bmp.top_down {
            dst_y
        } else {
            height - 1 - dst_y
        };
        let src_row = &bmp.pixel_data[src_y * bmp.row_stride..][..dst_row_bytes];
        let dst_row = &mut pixel_buf[dst_y * dst_row_bytes..(dst_y + 1) * dst_row_bytes];
        swizzle_rgb24_to_bgr24(src_row, dst_row);
    }
    Ok((bmp.width, bmp.height, DecodedPixelFormat::Rgb8, out_len))
}

fn decode_bmp_32_bgrx(
    source: &Path,
    bmp: &ParsedBmp<'_>,
    width: usize,
    height: usize,
    pixel_count: usize,
    hint: DecodeTargetHint,
    pixel_buf: &mut Vec<u8>,
) -> Result<(u32, u32, DecodedPixelFormat, usize), PipelineError> {
    #[cfg(feature = "nvenc")]
    if hint == DecodeTargetHint::OpaqueRgba8 {
        let out_len = pixel_count
            .checked_mul(4)
            .ok_or_else(|| PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: "BMP RGBA8 output overflow".to_owned(),
            })?;
        if pixel_buf.len() < out_len {
            pixel_buf.resize(out_len, 0);
        }
        let dst_row_bytes = width * 4;
        for dst_y in 0..height {
            let src_y = if bmp.top_down {
                dst_y
            } else {
                height - 1 - dst_y
            };
            let src_row = &bmp.pixel_data[src_y * bmp.row_stride..][..dst_row_bytes];
            let dst_row = &mut pixel_buf[dst_y * dst_row_bytes..(dst_y + 1) * dst_row_bytes];
            let (src_quads, _) = src_row.as_chunks::<4>();
            let (dst_quads, _) = dst_row.as_chunks_mut::<4>();
            for (&[b, g, r, _], dst) in src_quads.iter().zip(dst_quads.iter_mut()) {
                *dst = [r, g, b, 255];
            }
        }
        return Ok((bmp.width, bmp.height, DecodedPixelFormat::Rgba8, out_len));
    }

    let _ = hint;
    let out_len = pixel_count
        .checked_mul(3)
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "BMP RGB8 output overflow".to_owned(),
        })?;
    if pixel_buf.len() < out_len {
        pixel_buf.resize(out_len, 0);
    }
    let src_row_bytes = width * 4;
    let dst_row_bytes = width * 3;
    for dst_y in 0..height {
        let src_y = if bmp.top_down {
            dst_y
        } else {
            height - 1 - dst_y
        };
        let src_row = &bmp.pixel_data[src_y * bmp.row_stride..][..src_row_bytes];
        let dst_row = &mut pixel_buf[dst_y * dst_row_bytes..(dst_y + 1) * dst_row_bytes];
        let (src_quads, _) = src_row.as_chunks::<4>();
        let (dst_triplets, _) = dst_row.as_chunks_mut::<3>();
        for (&[b, g, r, _], dst) in src_quads.iter().zip(dst_triplets.iter_mut()) {
            *dst = [r, g, b];
        }
    }
    Ok((bmp.width, bmp.height, DecodedPixelFormat::Rgb8, out_len))
}

fn decode_bmp_32_bgra(
    source: &Path,
    bmp: &ParsedBmp<'_>,
    width: usize,
    height: usize,
    pixel_count: usize,
    hint: DecodeTargetHint,
    pixel_buf: &mut Vec<u8>,
) -> Result<(u32, u32, DecodedPixelFormat, usize), PipelineError> {
    let out_len = pixel_count
        .checked_mul(4)
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "BMP RGBA8 output overflow".to_owned(),
        })?;
    if pixel_buf.len() < out_len {
        pixel_buf.resize(out_len, 0);
    }
    #[cfg(feature = "nvenc")]
    let reject_transparent = hint == DecodeTargetHint::OpaqueRgba8;
    #[cfg(not(feature = "nvenc"))]
    let reject_transparent = {
        let _ = hint;
        false
    };

    let dst_row_bytes = width * 4;
    for dst_y in 0..height {
        let src_y = if bmp.top_down {
            dst_y
        } else {
            height - 1 - dst_y
        };
        let src_row = &bmp.pixel_data[src_y * bmp.row_stride..][..dst_row_bytes];
        let dst_row = &mut pixel_buf[dst_y * dst_row_bytes..(dst_y + 1) * dst_row_bytes];
        let (src_quads, _) = src_row.as_chunks::<4>();
        let (dst_quads, _) = dst_row.as_chunks_mut::<4>();
        for (&[b, g, r, a], dst) in src_quads.iter().zip(dst_quads.iter_mut()) {
            if reject_transparent && a < 255 {
                return Err(PipelineError::UnsupportedPixelFormat {
                    path: source.to_path_buf(),
                    format: "non-opaque alpha channel (NVENC HEVC-Intra HEIC currently supports opaque images)".to_owned(),
                });
            }
            *dst = [r, g, b, a];
        }
    }
    Ok((bmp.width, bmp.height, DecodedPixelFormat::Rgba8, out_len))
}

fn decode_bmp_32_bitfields(
    source: &Path,
    bmp: &ParsedBmp<'_>,
    width: usize,
    height: usize,
    pixel_count: usize,
    hint: DecodeTargetHint,
    pixel_buf: &mut Vec<u8>,
) -> Result<(u32, u32, DecodedPixelFormat, usize), PipelineError> {
    let [rm, gm, bm, am] = bmp.masks;
    let has_alpha = am.mask != 0;

    // Fast path for standard 8-bit BGRA masks (e.g. our own BITMAPV4HEADER output).
    if rm.mask == 0x00FF_0000
        && gm.mask == 0x0000_FF00
        && bm.mask == 0x0000_00FF
        && am.mask == 0xFF00_0000
    {
        return decode_bmp_32_bgra(source, bmp, width, height, pixel_count, hint, pixel_buf);
    }

    if has_alpha {
        let out_len = pixel_count
            .checked_mul(4)
            .ok_or_else(|| PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: "BMP RGBA8 output overflow".to_owned(),
            })?;
        if pixel_buf.len() < out_len {
            pixel_buf.resize(out_len, 0);
        }
        #[cfg(feature = "nvenc")]
        let reject_transparent = hint == DecodeTargetHint::OpaqueRgba8;
        #[cfg(not(feature = "nvenc"))]
        let reject_transparent = {
            let _ = hint;
            false
        };
        let dst_row_bytes = width * 4;
        for dst_y in 0..height {
            let src_y = if bmp.top_down {
                dst_y
            } else {
                height - 1 - dst_y
            };
            let src_row = &bmp.pixel_data[src_y * bmp.row_stride..][..dst_row_bytes];
            let dst_row = &mut pixel_buf[dst_y * dst_row_bytes..(dst_y + 1) * dst_row_bytes];
            let (src_quads, _) = src_row.as_chunks::<4>();
            let (dst_quads, _) = dst_row.as_chunks_mut::<4>();
            for (&bytes, dst) in src_quads.iter().zip(dst_quads.iter_mut()) {
                let v = u32::from_le_bytes(bytes);
                if reject_transparent && !am.is_opaque(v) {
                    return Err(PipelineError::UnsupportedPixelFormat {
                        path: source.to_path_buf(),
                        format: "non-opaque alpha channel (NVENC HEVC-Intra HEIC currently supports opaque images)".to_owned(),
                    });
                }
                *dst = [
                    rm.extract_u8(v),
                    gm.extract_u8(v),
                    bm.extract_u8(v),
                    am.extract_u8(v),
                ];
            }
        }
        Ok((bmp.width, bmp.height, DecodedPixelFormat::Rgba8, out_len))
    } else {
        let out_len = pixel_count
            .checked_mul(3)
            .ok_or_else(|| PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: "BMP RGB8 output overflow".to_owned(),
            })?;
        if pixel_buf.len() < out_len {
            pixel_buf.resize(out_len, 0);
        }
        let dst_row_bytes = width * 3;
        for dst_y in 0..height {
            let src_y = if bmp.top_down {
                dst_y
            } else {
                height - 1 - dst_y
            };
            let src_row = &bmp.pixel_data[src_y * bmp.row_stride..][..width * 4];
            let dst_row = &mut pixel_buf[dst_y * dst_row_bytes..(dst_y + 1) * dst_row_bytes];
            let (src_quads, _) = src_row.as_chunks::<4>();
            let (dst_triplets, _) = dst_row.as_chunks_mut::<3>();
            for (&bytes, dst) in src_quads.iter().zip(dst_triplets.iter_mut()) {
                let v = u32::from_le_bytes(bytes);
                *dst = [rm.extract_u8(v), gm.extract_u8(v), bm.extract_u8(v)];
            }
        }
        Ok((bmp.width, bmp.height, DecodedPixelFormat::Rgb8, out_len))
    }
}

fn decode_bmp_16(
    source: &Path,
    bmp: &ParsedBmp<'_>,
    width: usize,
    height: usize,
    pixel_count: usize,
    hint: DecodeTargetHint,
    pixel_buf: &mut Vec<u8>,
) -> Result<(u32, u32, DecodedPixelFormat, usize), PipelineError> {
    let [rm, gm, bm, am] = if bmp.compression == 0 {
        [
            BitMask::from_mask(0x7C00),
            BitMask::from_mask(0x03E0),
            BitMask::from_mask(0x001F),
            BitMask::from_mask(0),
        ]
    } else {
        bmp.masks
    };
    let has_alpha = am.mask != 0;
    #[cfg(feature = "nvenc")]
    let reject_transparent = hint == DecodeTargetHint::OpaqueRgba8;
    #[cfg(not(feature = "nvenc"))]
    let reject_transparent = {
        let _ = hint;
        false
    };
    let out_bpp = if has_alpha { 4_usize } else { 3_usize };
    let out_len = pixel_count
        .checked_mul(out_bpp)
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "BMP 16-bit output buffer overflow".to_owned(),
        })?;
    if pixel_buf.len() < out_len {
        pixel_buf.resize(out_len, 0);
    }

    let dst_row_bytes = width * out_bpp;
    for dst_y in 0..height {
        let src_y = if bmp.top_down {
            dst_y
        } else {
            height - 1 - dst_y
        };
        let src_row = &bmp.pixel_data[src_y * bmp.row_stride..][..width * 2];
        let dst_row = &mut pixel_buf[dst_y * dst_row_bytes..(dst_y + 1) * dst_row_bytes];
        let (src_pairs, _) = src_row.as_chunks::<2>();
        if has_alpha {
            let (dst_quads, _) = dst_row.as_chunks_mut::<4>();
            for (&bytes, dst) in src_pairs.iter().zip(dst_quads.iter_mut()) {
                let v = u32::from(u16::from_le_bytes(bytes));
                if reject_transparent && !am.is_opaque(v) {
                    return Err(PipelineError::UnsupportedPixelFormat {
                        path: source.to_path_buf(),
                        format: "non-opaque alpha channel (NVENC HEVC-Intra HEIC currently supports opaque images)".to_owned(),
                    });
                }
                *dst = [
                    rm.extract_u8(v),
                    gm.extract_u8(v),
                    bm.extract_u8(v),
                    am.extract_u8(v),
                ];
            }
        } else {
            let (dst_triplets, _) = dst_row.as_chunks_mut::<3>();
            for (&bytes, dst) in src_pairs.iter().zip(dst_triplets.iter_mut()) {
                let v = u32::from(u16::from_le_bytes(bytes));
                *dst = [rm.extract_u8(v), gm.extract_u8(v), bm.extract_u8(v)];
            }
        }
    }

    let format = if has_alpha {
        DecodedPixelFormat::Rgba8
    } else {
        DecodedPixelFormat::Rgb8
    };
    Ok((bmp.width, bmp.height, format, out_len))
}

fn parse_indexed_bmp_palette(
    source: &Path,
    bmp: &ParsedBmp<'_>,
) -> Result<([[u8; 3]; 256], usize), PipelineError> {
    let max_colors = 1_usize << bmp.bit_count;
    let entry_size = if bmp.dib_size == 12 { 3_usize } else { 4_usize };
    let entry_count = if bmp.dib_size == 12 {
        let count = (bmp.palette_data.len() / entry_size).min(max_colors);
        if bmp.bit_count == 1 && count < 2 {
            return Err(PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: "1-bit BITMAPCOREHEADER BMP requires 2 color palette entries".to_owned(),
            });
        }
        count
    } else {
        let declared = if bmp.colors_used == 0 {
            max_colors
        } else {
            usize::try_from(bmp.colors_used).map_err(|_| PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: format!(
                    "BMP biClrUsed {} exceeds addressable memory",
                    bmp.colors_used
                ),
            })?
        };
        if declared > max_colors {
            return Err(PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: format!(
                    "BMP biClrUsed ({declared}) exceeds maximum palette size ({max_colors}) for {}-bit image",
                    bmp.bit_count
                ),
            });
        }
        let required_bytes =
            declared
                .checked_mul(entry_size)
                .ok_or_else(|| PipelineError::DecodeFailed {
                    path: source.to_path_buf(),
                    reason: "BMP palette byte size overflow".to_owned(),
                })?;
        if bmp.palette_data.len() < required_bytes {
            return Err(PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: format!(
                    "indexed BMP palette truncated: declared {declared} entries ({required_bytes} bytes), only {} bytes before pixel data",
                    bmp.palette_data.len()
                ),
            });
        }
        declared
    };
    if entry_count == 0 {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "indexed BMP has an empty color palette".to_owned(),
        });
    }
    let mut palette = [[0_u8; 3]; 256];
    for (i, slot) in palette.iter_mut().enumerate().take(entry_count) {
        let base = i * entry_size;
        *slot = [
            bmp.palette_data[base + 2],
            bmp.palette_data[base + 1],
            bmp.palette_data[base],
        ];
    }
    Ok((palette, entry_count))
}

fn decode_bmp_indexed(
    source: &Path,
    bmp: &ParsedBmp<'_>,
    width: usize,
    height: usize,
    pixel_count: usize,
    pixel_buf: &mut Vec<u8>,
) -> Result<(u32, u32, DecodedPixelFormat, usize), PipelineError> {
    let (palette, entry_count) = parse_indexed_bmp_palette(source, bmp)?;

    let out_len = pixel_count
        .checked_mul(3)
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: "BMP indexed RGB8 output overflow".to_owned(),
        })?;
    if pixel_buf.len() < out_len {
        pixel_buf.resize(out_len, 0);
    }

    let dst_row_bytes = width * 3;
    for dst_y in 0..height {
        let src_y = if bmp.top_down {
            dst_y
        } else {
            height - 1 - dst_y
        };
        let src_row = &bmp.pixel_data[src_y * bmp.row_stride..(src_y + 1) * bmp.row_stride];
        let dst_row = &mut pixel_buf[dst_y * dst_row_bytes..(dst_y + 1) * dst_row_bytes];
        let (dst_triplets, _) = dst_row.as_chunks_mut::<3>();
        match bmp.bit_count {
            8 => {
                for (&idx, dst) in src_row.iter().zip(dst_triplets.iter_mut()) {
                    let i = usize::from(idx);
                    if i >= entry_count {
                        return Err(PipelineError::DecodeFailed {
                            path: source.to_path_buf(),
                            reason: format!("BMP palette index {i} out of bounds ({entry_count})"),
                        });
                    }
                    *dst = palette[i];
                }
            }
            4 => {
                for (x, dst) in dst_triplets.iter_mut().enumerate() {
                    let byte = src_row[x / 2];
                    let idx = usize::from(if x % 2 == 0 { byte >> 4 } else { byte & 0x0F });
                    if idx >= entry_count {
                        return Err(PipelineError::DecodeFailed {
                            path: source.to_path_buf(),
                            reason: format!(
                                "BMP palette index {idx} out of bounds ({entry_count})"
                            ),
                        });
                    }
                    *dst = palette[idx];
                }
            }
            1 => {
                for (x, dst) in dst_triplets.iter_mut().enumerate() {
                    let byte = src_row[x / 8];
                    let idx = usize::from((byte >> (7 - (x % 8))) & 1);
                    if idx >= entry_count {
                        return Err(PipelineError::DecodeFailed {
                            path: source.to_path_buf(),
                            reason: format!(
                                "BMP palette index {idx} out of bounds ({entry_count})"
                            ),
                        });
                    }
                    *dst = palette[idx];
                }
            }
            other => {
                return Err(PipelineError::UnsupportedPixelFormat {
                    path: source.to_path_buf(),
                    format: format!("BMP indexed bit depth {other}"),
                });
            }
        }
    }

    Ok((bmp.width, bmp.height, DecodedPixelFormat::Rgb8, out_len))
}

#[cfg(test)]
mod tests {
    use super::{decode_bmp_slice_into, encode_bmp_to_writer};
    use crate::decode::{DecodeTargetHint, DecodedFrameMeta, DecodedPixelFormat};
    use std::path::Path;

    #[test]
    fn roundtrips_odd_width_rgb24_rgba32_and_gray8_bmp() -> Result<(), Box<dyn std::error::Error>> {
        let p = Path::new("test.bmp");
        // Width 13 forces non-zero 4-byte scanline padding for both 24-bit and 8-bit BMPs.
        let (w, h) = (13_u32, 9_u32);
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                rgb.extend_from_slice(&[
                    u8::try_from((x * 19) % 256).unwrap_or(0),
                    u8::try_from((y * 23) % 256).unwrap_or(0),
                    u8::try_from(((x + y) * 11) % 256).unwrap_or(0),
                ]);
            }
        }

        let mut row_scratch = Vec::new();
        let mut encoded = Vec::new();
        let written = encode_bmp_to_writer(
            p,
            &rgb,
            DecodedFrameMeta {
                width: w,
                height: h,
                format: DecodedPixelFormat::Rgb8,
                byte_len: rgb.len(),
                input_file_bytes: 1,
            },
            &mut row_scratch,
            &mut encoded,
        )?;
        assert_eq!(written, u64::try_from(encoded.len())?);

        let mut decoded = Vec::new();
        let (dw, dh, fmt, len) =
            decode_bmp_slice_into(p, &encoded, DecodeTargetHint::Native, &mut decoded)?;
        assert_eq!(
            (dw, dh, fmt, len),
            (w, h, DecodedPixelFormat::Rgb8, rgb.len())
        );
        assert_eq!(&decoded[..len], rgb.as_slice());

        // RGBA8 -> BITMAPV4HEADER 32-bit BGRA roundtrip preserving transparency.
        let rgba: Vec<u8> = rgb
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
            .flat_map(|(i, &[r, g, b])| [r, g, b, if i % 3 == 0 { 128 } else { 255 }])
            .collect();
        encoded.clear();
        encode_bmp_to_writer(
            p,
            &rgba,
            DecodedFrameMeta {
                width: w,
                height: h,
                format: DecodedPixelFormat::Rgba8,
                byte_len: rgba.len(),
                input_file_bytes: 1,
            },
            &mut row_scratch,
            &mut encoded,
        )?;
        let (dw, dh, fmt, len) =
            decode_bmp_slice_into(p, &encoded, DecodeTargetHint::Native, &mut decoded)?;
        assert_eq!(
            (dw, dh, fmt, len),
            (w, h, DecodedPixelFormat::Rgba8, rgba.len())
        );
        assert_eq!(&decoded[..len], rgba.as_slice());

        // L8 -> 8-bit paletted grayscale BMP roundtrip.
        let luma: Vec<u8> = (0..(w * h))
            .map(|i| u8::try_from((i * 7) % 256).unwrap_or(0))
            .collect();
        encoded.clear();
        encode_bmp_to_writer(
            p,
            &luma,
            DecodedFrameMeta {
                width: w,
                height: h,
                format: DecodedPixelFormat::L8,
                byte_len: luma.len(),
                input_file_bytes: 1,
            },
            &mut row_scratch,
            &mut encoded,
        )?;
        let (dw, dh, fmt, len) =
            decode_bmp_slice_into(p, &encoded, DecodeTargetHint::Native, &mut decoded)?;
        assert_eq!((dw, dh, fmt), (w, h, DecodedPixelFormat::Rgb8));
        let expected_rgb: Vec<u8> = luma.iter().flat_map(|&v| [v, v, v]).collect();
        assert_eq!(&decoded[..len], expected_rgb.as_slice());

        Ok(())
    }

    #[test]
    fn decodes_32bit_bi_rgb_as_opaque_bgrx_ignoring_zero_high_byte()
    -> Result<(), Box<dyn std::error::Error>> {
        let p = Path::new("bgrx32.bmp");
        // 2x1 32-bit BI_RGB BITMAPINFOHEADER where the unused high byte is 0x00.
        let mut bmp = Vec::new();
        bmp.extend_from_slice(b"BM");
        bmp.extend_from_slice(&(54_u32 + 8).to_le_bytes());
        bmp.extend_from_slice(&0_u32.to_le_bytes());
        bmp.extend_from_slice(&54_u32.to_le_bytes());
        bmp.extend_from_slice(&40_u32.to_le_bytes());
        bmp.extend_from_slice(&2_i32.to_le_bytes());
        bmp.extend_from_slice(&1_i32.to_le_bytes());
        bmp.extend_from_slice(&1_u16.to_le_bytes());
        bmp.extend_from_slice(&32_u16.to_le_bytes());
        bmp.extend_from_slice(&0_u32.to_le_bytes()); // BI_RGB
        bmp.extend_from_slice(&8_u32.to_le_bytes());
        bmp.extend_from_slice(&[0_u8; 16]);
        // Pixel 0: B=10, G=20, R=30, X=0; Pixel 1: B=40, G=50, R=60, X=0.
        bmp.extend_from_slice(&[10, 20, 30, 0, 40, 50, 60, 0]);

        let mut decoded = Vec::new();
        let (w, h, fmt, len) =
            decode_bmp_slice_into(p, &bmp, DecodeTargetHint::Native, &mut decoded)?;
        assert_eq!((w, h, fmt, len), (2, 1, DecodedPixelFormat::Rgb8, 6));
        assert_eq!(&decoded[..len], &[30, 20, 10, 60, 50, 40]);

        #[cfg(feature = "nvenc")]
        {
            let (w, h, fmt, len) =
                decode_bmp_slice_into(p, &bmp, DecodeTargetHint::OpaqueRgba8, &mut decoded)?;
            assert_eq!((w, h, fmt, len), (2, 1, DecodedPixelFormat::Rgba8, 8));
            assert_eq!(&decoded[..len], &[30, 20, 10, 255, 60, 50, 40, 255]);
        }
        Ok(())
    }

    #[test]
    fn enforces_declared_palette_size_for_indexed_bmp() {
        let p = Path::new("indexed.bmp");
        // 8-bit BITMAPINFOHEADER declaring biClrUsed = 2, with 4 gap bytes before pixels.
        let mut bmp = Vec::new();
        bmp.extend_from_slice(b"BM");
        bmp.extend_from_slice(&(54_u32 + 12 + 4).to_le_bytes());
        bmp.extend_from_slice(&0_u32.to_le_bytes());
        bmp.extend_from_slice(&(54_u32 + 12).to_le_bytes()); // 8 bytes palette + 4 gap bytes
        bmp.extend_from_slice(&40_u32.to_le_bytes());
        bmp.extend_from_slice(&1_i32.to_le_bytes());
        bmp.extend_from_slice(&1_i32.to_le_bytes());
        bmp.extend_from_slice(&1_u16.to_le_bytes());
        bmp.extend_from_slice(&8_u16.to_le_bytes());
        bmp.extend_from_slice(&0_u32.to_le_bytes()); // BI_RGB
        bmp.extend_from_slice(&4_u32.to_le_bytes());
        bmp.extend_from_slice(&[0_u8; 8]); // ppm
        bmp.extend_from_slice(&2_u32.to_le_bytes()); // biClrUsed = 2
        bmp.extend_from_slice(&0_u32.to_le_bytes());
        // 2 palette entries + 4 gap bytes pretending to be a 3rd entry.
        bmp.extend_from_slice(&[0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 0]);
        // Pixel index 2 must be rejected because biClrUsed == 2.
        bmp.extend_from_slice(&[2, 0, 0, 0]);

        let mut decoded = Vec::new();
        assert!(decode_bmp_slice_into(p, &bmp, DecodeTargetHint::Native, &mut decoded).is_err());

        // Pixel index 1 is within biClrUsed == 2 and decodes cleanly.
        bmp[54 + 12] = 1;
        let res = decode_bmp_slice_into(p, &bmp, DecodeTargetHint::Native, &mut decoded);
        assert!(res.is_ok());
    }

    #[test]
    fn enforces_16bit_and_high_bitdepth_alpha_opacity_for_nvenc()
    -> Result<(), Box<dyn std::error::Error>> {
        // 16-bit BI_ALPHABITFIELDS RGB555 + A1 (1x1 transparent pixel: alpha bit 0).
        let mut bmp16 = Vec::new();
        bmp16.extend_from_slice(b"BM");
        bmp16.extend_from_slice(&(14_u32 + 56 + 4).to_le_bytes());
        bmp16.extend_from_slice(&0_u32.to_le_bytes());
        bmp16.extend_from_slice(&(14_u32 + 56).to_le_bytes());
        bmp16.extend_from_slice(&56_u32.to_le_bytes());
        bmp16.extend_from_slice(&1_i32.to_le_bytes());
        bmp16.extend_from_slice(&1_i32.to_le_bytes());
        bmp16.extend_from_slice(&1_u16.to_le_bytes());
        bmp16.extend_from_slice(&16_u16.to_le_bytes());
        bmp16.extend_from_slice(&6_u32.to_le_bytes()); // BI_ALPHABITFIELDS
        bmp16.extend_from_slice(&4_u32.to_le_bytes());
        bmp16.extend_from_slice(&[0_u8; 16]);
        bmp16.extend_from_slice(&0x7C00_u32.to_le_bytes()); // R
        bmp16.extend_from_slice(&0x03E0_u32.to_le_bytes()); // G
        bmp16.extend_from_slice(&0x001F_u32.to_le_bytes()); // B
        bmp16.extend_from_slice(&0x8000_u32.to_le_bytes()); // A (1 bit)
        bmp16.extend_from_slice(&0x7FFF_u16.to_le_bytes()); // alpha bit = 0
        bmp16.extend_from_slice(&[0, 0]);

        let p16 = Path::new("argb1555.bmp");
        let mut decoded = Vec::new();
        let (_, _, fmt, len) =
            decode_bmp_slice_into(p16, &bmp16, DecodeTargetHint::Native, &mut decoded)?;
        assert_eq!((fmt, len), (DecodedPixelFormat::Rgba8, 4));
        assert_eq!(decoded[3], 0);

        #[cfg(feature = "nvenc")]
        {
            assert!(matches!(
                decode_bmp_slice_into(p16, &bmp16, DecodeTargetHint::OpaqueRgba8, &mut decoded),
                Err(crate::PipelineError::UnsupportedPixelFormat { .. })
            ));
        }

        // 32-bit BI_BITFIELDS with 10-bit alpha mask (0xFFC0_0000): sample 1022/1023 rounds
        // to 255 in 8-bit native output but is not fully opaque and must fail NVENC check.
        let mut bmp32 = Vec::new();
        bmp32.extend_from_slice(b"BM");
        bmp32.extend_from_slice(&(14_u32 + 56 + 4).to_le_bytes());
        bmp32.extend_from_slice(&0_u32.to_le_bytes());
        bmp32.extend_from_slice(&(14_u32 + 56).to_le_bytes());
        bmp32.extend_from_slice(&56_u32.to_le_bytes());
        bmp32.extend_from_slice(&1_i32.to_le_bytes());
        bmp32.extend_from_slice(&1_i32.to_le_bytes());
        bmp32.extend_from_slice(&1_u16.to_le_bytes());
        bmp32.extend_from_slice(&32_u16.to_le_bytes());
        bmp32.extend_from_slice(&3_u32.to_le_bytes()); // BI_BITFIELDS
        bmp32.extend_from_slice(&4_u32.to_le_bytes());
        bmp32.extend_from_slice(&[0_u8; 16]);
        bmp32.extend_from_slice(&0x003F_0000_u32.to_le_bytes()); // R (6 bits)
        bmp32.extend_from_slice(&0x0000_FF00_u32.to_le_bytes()); // G (8 bits)
        bmp32.extend_from_slice(&0x0000_00FF_u32.to_le_bytes()); // B (8 bits)
        bmp32.extend_from_slice(&0xFFC0_0000_u32.to_le_bytes()); // A (10 bits)
        let pixel_a1022: u32 = (1022_u32 << 22) | 0x0015_8040;
        bmp32.extend_from_slice(&pixel_a1022.to_le_bytes());

        let p32 = Path::new("alpha10.bmp");
        let (_, _, fmt, len) =
            decode_bmp_slice_into(p32, &bmp32, DecodeTargetHint::Native, &mut decoded)?;
        assert_eq!((fmt, len), (DecodedPixelFormat::Rgba8, 4));
        assert_eq!(decoded[3], 255);

        #[cfg(feature = "nvenc")]
        {
            assert!(matches!(
                decode_bmp_slice_into(p32, &bmp32, DecodeTargetHint::OpaqueRgba8, &mut decoded),
                Err(crate::PipelineError::UnsupportedPixelFormat { .. })
            ));
        }

        Ok(())
    }
}
