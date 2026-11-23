use std::fs::{self, File};
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use fastconv_domain::{
    DomainError, HardwareBackend, HostHardware, ImageBatchItem, ImageBatchPlan, ImageCompression,
    ImageDimensions, ImageExecutionMode, ImageInputFormat, ImageOutputFormat, PathRole,
};
use jxl::api::JxlBitstreamInput;
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::colorspace::ColorSpace;

use crate::{ProbeError, available_cpu_threads, detect_host_hardware};

const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
const JPEG_MAGIC: [u8; 3] = [0xFF, 0xD8, 0xFF];
const RIFF_MAGIC: [u8; 4] = *b"RIFF";
const WEBP_MAGIC: [u8; 4] = *b"WEBP";
const QOI_MAGIC: [u8; 4] = *b"qoif";
const QOI_END_MARKER: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 1];
const QOI_MAX_RUN: u64 = 62;
// Shared pixel budget matching `fastconv-pipeline` QOI encode/decode (128 Mi pixels = 512 MiB RGBA8).
const MAX_QOI_PIXELS: u64 = 128 * 1024 * 1024;
const BMP_MAGIC: [u8; 2] = *b"BM";
const TIFF_LE_MAGIC: [u8; 4] = *b"II*\0";
const TIFF_BE_MAGIC: [u8; 4] = *b"MM\0*";
const JXL_CODESTREAM_MAGIC: [u8; 2] = [0xFF, 0x0A];
const JXL_CONTAINER_MAGIC: [u8; 4] = *b"JXL ";
// 512 Mi samples = 512 MiB RGBA8; keeps a hostile JXL header from forcing huge allocations.
const MAX_JXL_SAMPLES: usize = 512 * 1024 * 1024;
const KNOWN_IMAGE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "webp", "bmp", "qoi", "tif", "tiff", "jxl",
];
const UNRECOGNIZED_MAGIC_REASON: &str =
    "unrecognized image magic bytes (expected PNG, JPEG, WebP, BMP, QOI, TIFF, or JPEG XL)";

/// Detected input container format and validated pixel dimensions from image headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbedImage {
    pub format: ImageInputFormat,
    pub dimensions: ImageDimensions,
}

fn is_known_bmp_dib_size(size: u32) -> bool {
    matches!(size, 12 | 40 | 52 | 56 | 108 | 124)
}

fn sniff_magic_format(header: &[u8], strict_bmp_dib: bool) -> Option<ImageInputFormat> {
    if header.starts_with(&PNG_MAGIC) {
        Some(ImageInputFormat::Png)
    } else if header.starts_with(&JPEG_MAGIC) {
        Some(ImageInputFormat::Jpeg)
    } else if header.get(0..4) == Some(RIFF_MAGIC.as_slice())
        && header.get(8..12) == Some(WEBP_MAGIC.as_slice())
    {
        Some(ImageInputFormat::Webp)
    } else if header.starts_with(&QOI_MAGIC) {
        Some(ImageInputFormat::Qoi)
    } else if header.starts_with(&BMP_MAGIC) {
        if !strict_bmp_dib {
            return Some(ImageInputFormat::Bmp);
        }
        let dib_bytes = header.get(14..18)?;
        let dib_size = u32::from_le_bytes([dib_bytes[0], dib_bytes[1], dib_bytes[2], dib_bytes[3]]);
        is_known_bmp_dib_size(dib_size).then_some(ImageInputFormat::Bmp)
    } else if header.starts_with(&TIFF_LE_MAGIC) || header.starts_with(&TIFF_BE_MAGIC) {
        Some(ImageInputFormat::Tiff)
    } else if header.starts_with(&JXL_CODESTREAM_MAGIC)
        || (header.get(4..8) == Some(JXL_CONTAINER_MAGIC.as_slice())
            && header.get(0..4).is_some_and(|size| size != [0, 0, 0, 0]))
    {
        Some(ImageInputFormat::Jxl)
    } else {
        None
    }
}

fn read_magic_prefix(file: &mut File, path: &Path) -> Result<([u8; 18], usize), ProbeError> {
    let mut buf = [0_u8; 18];
    let mut filled = 0_usize;
    while filled < buf.len() {
        let slot = buf
            .get_mut(filled..)
            .ok_or_else(|| ProbeError::InvalidImage {
                path: path.to_path_buf(),
                reason: "internal magic-byte buffer offset overflow".to_owned(),
            })?;
        match file.read(slot) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(source) => {
                return Err(ProbeError::Io {
                    path: path.to_path_buf(),
                    source,
                });
            }
        }
    }
    Ok((buf, filled))
}

fn read_exact_or_invalid(
    file: &mut File,
    path: &Path,
    buf: &mut [u8],
    eof_reason: &str,
) -> Result<(), ProbeError> {
    file.read_exact(buf).map_err(|err| {
        if err.kind() == io::ErrorKind::UnexpectedEof {
            ProbeError::InvalidImage {
                path: path.to_path_buf(),
                reason: eof_reason.to_owned(),
            }
        } else {
            ProbeError::Io {
                path: path.to_path_buf(),
                source: err,
            }
        }
    })
}

fn invalid_image(path: &Path, reason: impl Into<String>) -> ProbeError {
    ProbeError::InvalidImage {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

fn is_supported_jpeg_colorspace(cs: Option<ColorSpace>) -> bool {
    matches!(
        cs,
        Some(
            ColorSpace::Luma
                | ColorSpace::YCbCr
                | ColorSpace::RGB
                | ColorSpace::CMYK
                | ColorSpace::YCCK
        )
    )
}

fn probe_webp(file: &mut File, path: &Path) -> Result<ImageDimensions, ProbeError> {
    let file_len = file
        .metadata()
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if file_len < 20 {
        return Err(invalid_image(path, "truncated WebP RIFF container"));
    }

    let mut header = [0_u8; 20];
    read_exact_or_invalid(file, path, &mut header, "truncated WebP header")?;
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WEBP" {
        return Err(invalid_image(path, "invalid RIFF/WEBP header signature"));
    }
    let riff_size = u64::from(u32::from_le_bytes([
        header[4], header[5], header[6], header[7],
    ]));
    if riff_size < 12 || riff_size.saturating_add(8) > file_len {
        return Err(invalid_image(
            path,
            format!(
                "invalid or truncated WebP RIFF size ({riff_size} bytes, file {file_len} bytes)"
            ),
        ));
    }
    let riff_end = riff_size + 8;

    let first_fourcc = [header[12], header[13], header[14], header[15]];
    let chunk_size = u64::from(u32::from_le_bytes([
        header[16], header[17], header[18], header[19],
    ]));
    let chunk_end = 20_u64
        .checked_add(chunk_size)
        .ok_or_else(|| invalid_image(path, "WebP chunk size overflow"))?;
    if chunk_end > riff_end {
        return Err(invalid_image(path, "truncated WebP primary chunk payload"));
    }

    match &first_fourcc {
        b"VP8 " => probe_webp_vp8(file, path, chunk_size),
        b"VP8L" => probe_webp_vp8l(file, path, chunk_size),
        b"VP8X" => probe_webp_vp8x(file, path, chunk_size, riff_end),
        other => Err(invalid_image(
            path,
            format!("unsupported WebP primary chunk {other:?}"),
        )),
    }
}

fn probe_webp_vp8(
    file: &mut File,
    path: &Path,
    chunk_size: u64,
) -> Result<ImageDimensions, ProbeError> {
    if chunk_size < 10 {
        return Err(invalid_image(path, "truncated VP8 lossy frame header"));
    }
    let mut vp8 = [0_u8; 10];
    read_exact_or_invalid(file, path, &mut vp8, "truncated VP8 lossy frame header")?;
    if (vp8[0] & 1) != 0 {
        return Err(invalid_image(path, "VP8 frame is not a keyframe"));
    }
    if vp8[3..6] != [0x9D, 0x01, 0x2A] {
        return Err(invalid_image(path, "invalid VP8 keyframe start code"));
    }
    let width = u32::from(u16::from_le_bytes([vp8[6], vp8[7]]) & 0x3FFF);
    let height = u32::from(u16::from_le_bytes([vp8[8], vp8[9]]) & 0x3FFF);
    Ok(ImageDimensions::new(width, height)?)
}

fn probe_webp_vp8l(
    file: &mut File,
    path: &Path,
    chunk_size: u64,
) -> Result<ImageDimensions, ProbeError> {
    if chunk_size < 5 {
        return Err(invalid_image(path, "truncated VP8L lossless header"));
    }
    let mut vp8l = [0_u8; 5];
    read_exact_or_invalid(file, path, &mut vp8l, "truncated VP8L lossless header")?;
    if vp8l[0] != 0x2F {
        return Err(invalid_image(path, "invalid VP8L lossless signature byte"));
    }
    let bits = u32::from_le_bytes([vp8l[1], vp8l[2], vp8l[3], vp8l[4]]);
    if (bits >> 29) != 0 {
        return Err(invalid_image(
            path,
            format!("unsupported VP8L version {}", bits >> 29),
        ));
    }
    let width = (bits & 0x3FFF) + 1;
    let height = ((bits >> 14) & 0x3FFF) + 1;
    Ok(ImageDimensions::new(width, height)?)
}

fn probe_webp_vp8x(
    file: &mut File,
    path: &Path,
    chunk_size: u64,
    riff_end: u64,
) -> Result<ImageDimensions, ProbeError> {
    if chunk_size < 10 {
        return Err(invalid_image(path, "truncated VP8X extended header"));
    }
    let mut vp8x = [0_u8; 10];
    read_exact_or_invalid(file, path, &mut vp8x, "truncated VP8X extended header")?;
    if (vp8x[0] & 0x02) != 0 {
        return Err(invalid_image(
            path,
            "animated WebP is not supported by the still-image pipeline",
        ));
    }
    let width = (u32::from(vp8x[4]) | (u32::from(vp8x[5]) << 8) | (u32::from(vp8x[6]) << 16)) + 1;
    let height = (u32::from(vp8x[7]) | (u32::from(vp8x[8]) << 8) | (u32::from(vp8x[9]) << 16)) + 1;
    let canvas_dims = ImageDimensions::new(width, height)?;

    let padded_vp8x = chunk_size + (chunk_size & 1);
    let mut pos = 20_u64
        .checked_add(padded_vp8x)
        .ok_or_else(|| invalid_image(path, "WebP VP8X chunk offset overflow"))?;
    let mut has_image_chunk = false;

    while pos.saturating_add(8) <= riff_end {
        file.seek(SeekFrom::Start(pos))
            .map_err(|source| ProbeError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        let mut hdr = [0_u8; 8];
        read_exact_or_invalid(file, path, &mut hdr, "truncated WebP chunk header")?;
        let fourcc = [hdr[0], hdr[1], hdr[2], hdr[3]];
        let len = u64::from(u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]));
        let payload_end = pos
            .checked_add(8)
            .and_then(|p| p.checked_add(len))
            .ok_or_else(|| invalid_image(path, "WebP chunk length overflow"))?;
        if payload_end > riff_end {
            return Err(invalid_image(
                path,
                "truncated chunk inside VP8X WebP container",
            ));
        }

        match &fourcc {
            b"ANIM" | b"ANMF" => {
                return Err(invalid_image(
                    path,
                    "animated WebP is not supported by the still-image pipeline",
                ));
            }
            b"VP8 " | b"VP8L" => {
                if has_image_chunk {
                    return Err(invalid_image(
                        path,
                        "VP8X WebP container must contain exactly one VP8 or VP8L image chunk",
                    ));
                }
                let frame_dims = if &fourcc == b"VP8 " {
                    probe_webp_vp8(file, path, len)?
                } else {
                    probe_webp_vp8l(file, path, len)?
                };
                if frame_dims != canvas_dims {
                    return Err(invalid_image(
                        path,
                        format!(
                            "VP8X canvas dimensions {canvas_dims} do not match embedded frame {frame_dims}"
                        ),
                    ));
                }
                has_image_chunk = true;
            }
            _ => {}
        }

        let padded_len = len + (len & 1);
        pos = pos
            .checked_add(8)
            .and_then(|p| p.checked_add(padded_len))
            .ok_or_else(|| invalid_image(path, "WebP chunk offset overflow"))?;
    }

    if !has_image_chunk {
        return Err(invalid_image(
            path,
            "VP8X WebP container must contain exactly one VP8 or VP8L image chunk",
        ));
    }

    Ok(canvas_dims)
}

fn read_bmp_dib_header(
    file: &mut File,
    path: &Path,
    dib_size: u32,
) -> Result<(u32, u32, u16, u32, u32), ProbeError> {
    if dib_size == 12 {
        let mut core = [0_u8; 8];
        read_exact_or_invalid(file, path, &mut core, "truncated BITMAPCOREHEADER")?;
        let w = u32::from(u16::from_le_bytes([core[0], core[1]]));
        let h = u32::from(u16::from_le_bytes([core[2], core[3]]));
        let planes = u16::from_le_bytes([core[4], core[5]]);
        let bpp = u16::from_le_bytes([core[6], core[7]]);
        if planes != 1 {
            return Err(invalid_image(
                path,
                format!("unsupported BMP plane count {planes}"),
            ));
        }
        return Ok((w, h, bpp, 0_u32, 0_u32));
    }

    let mut info = [0_u8; 36];
    read_exact_or_invalid(file, path, &mut info, "truncated BITMAPINFOHEADER")?;
    let raw_w = i32::from_le_bytes([info[0], info[1], info[2], info[3]]);
    let raw_h = i32::from_le_bytes([info[4], info[5], info[6], info[7]]);
    let planes = u16::from_le_bytes([info[8], info[9]]);
    let bpp = u16::from_le_bytes([info[10], info[11]]);
    let comp = u32::from_le_bytes([info[12], info[13], info[14], info[15]]);
    let colors_used = u32::from_le_bytes([info[28], info[29], info[30], info[31]]);
    if raw_w <= 0 || raw_h == 0 || raw_h == i32::MIN {
        return Err(invalid_image(
            path,
            format!("invalid BMP dimensions {raw_w}x{raw_h}"),
        ));
    }
    if planes != 1 {
        return Err(invalid_image(
            path,
            format!("unsupported BMP plane count {planes}"),
        ));
    }
    let w = u32::try_from(raw_w)
        .map_err(|_| invalid_image(path, format!("invalid BMP width {raw_w}")))?;
    Ok((w, raw_h.unsigned_abs(), bpp, comp, colors_used))
}

fn probe_bmp(file: &mut File, path: &Path) -> Result<ImageDimensions, ProbeError> {
    let file_len = file
        .metadata()
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if file_len < 26 {
        return Err(invalid_image(path, "truncated BMP header"));
    }

    let mut file_hdr = [0_u8; 18];
    read_exact_or_invalid(file, path, &mut file_hdr, "truncated BMP file header")?;
    if &file_hdr[0..2] != b"BM" {
        return Err(invalid_image(path, "invalid BMP signature"));
    }
    let pixel_offset = u64::from(u32::from_le_bytes([
        file_hdr[10],
        file_hdr[11],
        file_hdr[12],
        file_hdr[13],
    ]));
    let dib_size = u32::from_le_bytes([file_hdr[14], file_hdr[15], file_hdr[16], file_hdr[17]]);
    if !is_known_bmp_dib_size(dib_size) {
        return Err(invalid_image(
            path,
            format!("unsupported BMP DIB header size {dib_size}"),
        ));
    }
    let header_end = 14_u64 + u64::from(dib_size);
    if pixel_offset < header_end || pixel_offset > file_len {
        return Err(invalid_image(
            path,
            format!(
                "invalid BMP pixel offset {pixel_offset} (header ends at {header_end}, file is {file_len} bytes)"
            ),
        ));
    }

    let (width, height, bit_count, compression, colors_used) =
        read_bmp_dib_header(file, path, dib_size)?;

    validate_bmp_encoding(
        file,
        path,
        dib_size,
        header_end,
        pixel_offset,
        bit_count,
        compression,
        colors_used,
    )?;

    let dims = ImageDimensions::new(width, height)?;
    let row_bits = u64::from(width)
        .checked_mul(u64::from(bit_count))
        .ok_or_else(|| invalid_image(path, "BMP row bit count overflow"))?;
    let row_stride = row_bits
        .checked_add(31)
        .ok_or_else(|| invalid_image(path, "BMP row stride overflow"))?
        / 32
        * 4;
    let pixel_bytes = row_stride
        .checked_mul(u64::from(height))
        .ok_or_else(|| invalid_image(path, "BMP pixel data size overflow"))?;
    let required_len = pixel_offset
        .checked_add(pixel_bytes)
        .ok_or_else(|| invalid_image(path, "BMP total size overflow"))?;
    if required_len > file_len {
        return Err(invalid_image(
            path,
            format!("truncated BMP pixel payload (need {required_len} bytes, file has {file_len})"),
        ));
    }

    Ok(dims)
}

fn validate_bmp_palette_bounds(
    path: &Path,
    dib_size: u32,
    header_end: u64,
    pixel_offset: u64,
    bit_count: u16,
    colors_used: u32,
) -> Result<(), ProbeError> {
    let avail_palette_bytes = pixel_offset.saturating_sub(header_end);
    if dib_size == 12 {
        let min_bytes = if bit_count == 1 { 6_u64 } else { 3_u64 };
        if avail_palette_bytes < min_bytes {
            return Err(invalid_image(
                path,
                "indexed BMP is missing color palette entries",
            ));
        }
        return Ok(());
    }
    let max_colors = 1_u32 << bit_count;
    let declared = if colors_used == 0 {
        max_colors
    } else {
        colors_used
    };
    if declared > max_colors {
        return Err(invalid_image(
            path,
            format!(
                "BMP biClrUsed ({declared}) exceeds maximum palette size ({max_colors}) for {bit_count}-bit image"
            ),
        ));
    }
    let required_bytes = u64::from(declared) * 4;
    if avail_palette_bytes < required_bytes {
        return Err(invalid_image(
            path,
            format!(
                "indexed BMP palette truncated: declared {declared} entries ({required_bytes} bytes), only {avail_palette_bytes} bytes before pixel data"
            ),
        ));
    }
    Ok(())
}

fn validate_bmp_bitfields_masks(
    file: &mut File,
    path: &Path,
    dib_size: u32,
    pixel_offset: u64,
    bit_count: u16,
    compression: u32,
) -> Result<(), ProbeError> {
    if !matches!(bit_count, 16 | 32) {
        return Err(invalid_image(
            path,
            format!("BMP BI_BITFIELDS requires 16-bit or 32-bit pixels, got {bit_count}"),
        ));
    }
    let need_alpha_mask = compression == 6 || dib_size >= 56;
    let mask_bytes_needed = if need_alpha_mask { 16_u64 } else { 12_u64 };
    let mask_offset = 54_u64;
    if mask_offset + mask_bytes_needed > pixel_offset {
        return Err(invalid_image(
            path,
            "truncated BMP BI_BITFIELDS channel masks",
        ));
    }
    file.seek(SeekFrom::Start(mask_offset))
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let mut raw = [0_u8; 16];
    let read_len = usize::try_from(mask_bytes_needed).unwrap_or(12);
    read_exact_or_invalid(
        file,
        path,
        &mut raw[..read_len],
        "truncated BMP BI_BITFIELDS channel masks",
    )?;
    let r = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let g = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    let b = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]);
    let a = if need_alpha_mask {
        u32::from_le_bytes([raw[12], raw[13], raw[14], raw[15]])
    } else {
        0
    };
    if r == 0 || g == 0 || b == 0 {
        return Err(invalid_image(
            path,
            "invalid BMP BI_BITFIELDS with zero RGB channel mask",
        ));
    }
    let is_contiguous = |mask: u32| -> bool {
        if mask == 0 {
            return true;
        }
        let shifted = mask >> mask.trailing_zeros();
        (shifted & shifted.wrapping_add(1)) == 0
    };
    if !is_contiguous(r) || !is_contiguous(g) || !is_contiguous(b) || !is_contiguous(a) {
        return Err(invalid_image(
            path,
            "invalid BMP BI_BITFIELDS with non-contiguous channel mask",
        ));
    }
    if (r & g) != 0 || (r & b) != 0 || (g & b) != 0 || ((r | g | b) & a) != 0 {
        return Err(invalid_image(
            path,
            "invalid BMP BI_BITFIELDS with overlapping channel masks",
        ));
    }
    if bit_count == 16 && ((r | g | b | a) >> 16) != 0 {
        return Err(invalid_image(
            path,
            "invalid 16-bit BMP BI_BITFIELDS mask exceeding 16 bits",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_bmp_encoding(
    file: &mut File,
    path: &Path,
    dib_size: u32,
    header_end: u64,
    pixel_offset: u64,
    bit_count: u16,
    compression: u32,
    colors_used: u32,
) -> Result<(), ProbeError> {
    match compression {
        0 => {
            if !matches!(bit_count, 1 | 4 | 8 | 16 | 24 | 32) {
                return Err(invalid_image(
                    path,
                    format!("unsupported uncompressed BMP bit depth {bit_count}"),
                ));
            }
            if dib_size == 12 && !matches!(bit_count, 1 | 4 | 8 | 24) {
                return Err(invalid_image(
                    path,
                    format!("unsupported BITMAPCOREHEADER bit depth {bit_count}"),
                ));
            }
            if matches!(bit_count, 1 | 4 | 8) {
                validate_bmp_palette_bounds(
                    path,
                    dib_size,
                    header_end,
                    pixel_offset,
                    bit_count,
                    colors_used,
                )?;
            }
            Ok(())
        }
        3 | 6 => {
            validate_bmp_bitfields_masks(file, path, dib_size, pixel_offset, bit_count, compression)
        }
        other => Err(invalid_image(
            path,
            format!("unsupported BMP compression mode {other}"),
        )),
    }
}

fn probe_qoi(file: &mut File, path: &Path) -> Result<ImageDimensions, ProbeError> {
    let file_len = file
        .metadata()
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if file_len < 22 {
        return Err(invalid_image(path, "truncated QOI image"));
    }

    let mut hdr = [0_u8; 14];
    read_exact_or_invalid(file, path, &mut hdr, "truncated QOI header")?;
    if hdr[0..4] != QOI_MAGIC {
        return Err(invalid_image(path, "invalid QOI magic bytes"));
    }
    let width = u32::from_be_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
    let height = u32::from_be_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]);
    let channels = hdr[12];
    let colorspace = hdr[13];

    if !matches!(channels, 3 | 4) {
        return Err(invalid_image(
            path,
            format!("invalid QOI channel count {channels} (expected 3 or 4)"),
        ));
    }
    if !matches!(colorspace, 0 | 1) {
        return Err(invalid_image(
            path,
            format!("invalid QOI colorspace {colorspace} (expected 0 or 1)"),
        ));
    }

    let dims = ImageDimensions::new(width, height)?;
    let pixel_count = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| invalid_image(path, "QOI dimensions overflow pixel count"))?;
    if pixel_count > MAX_QOI_PIXELS {
        return Err(invalid_image(
            path,
            format!("QOI pixel count ({pixel_count}) exceeds {MAX_QOI_PIXELS}-pixel limit"),
        ));
    }
    let total_rgba = pixel_count
        .checked_mul(4)
        .filter(|&bytes| usize::try_from(bytes).is_ok());
    if total_rgba.is_none() {
        return Err(invalid_image(
            path,
            "QOI dimensions exceed addressable memory",
        ));
    }
    let stream_len = file_len - 22;
    let min_stream_bytes = pixel_count.div_ceil(QOI_MAX_RUN);
    if stream_len < min_stream_bytes {
        return Err(invalid_image(
            path,
            format!(
                "QOI bitstream ({stream_len} bytes) is too short to represent {pixel_count} pixels (minimum {min_stream_bytes} bytes)"
            ),
        ));
    }

    file.seek(SeekFrom::Start(file_len - 8))
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let mut tail = [0_u8; 8];
    read_exact_or_invalid(file, path, &mut tail, "truncated QOI end marker")?;
    if tail != QOI_END_MARKER {
        return Err(invalid_image(path, "invalid or missing QOI end marker"));
    }

    Ok(dims)
}

fn probe_opened_image(
    mut file: File,
    path: &Path,
    format: ImageInputFormat,
) -> Result<ProbedImage, ProbeError> {
    file.seek(SeekFrom::Start(0))
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;

    let dimensions = match format {
        ImageInputFormat::Png => {
            let mut decoder = png::Decoder::new(BufReader::new(file));
            decoder.set_ignore_text_chunk(true);
            let reader = decoder
                .read_info()
                .map_err(|err| invalid_image(path, err.to_string()))?;
            let info = reader.info();
            ImageDimensions::new(info.width, info.height)?
        }
        ImageInputFormat::Jpeg => {
            let mut decoder = JpegDecoder::new(BufReader::new(file));
            decoder
                .decode_headers()
                .map_err(|err| invalid_image(path, err.to_string()))?;
            let info = decoder
                .info()
                .ok_or_else(|| invalid_image(path, "missing JPEG SOF header metadata"))?;
            let input_cs = decoder.input_colorspace();
            if !is_supported_jpeg_colorspace(input_cs) {
                return Err(invalid_image(
                    path,
                    format!("unsupported JPEG color space: {input_cs:?}"),
                ));
            }
            ImageDimensions::new(u32::from(info.width), u32::from(info.height))?
        }
        ImageInputFormat::Webp => probe_webp(&mut file, path)?,
        ImageInputFormat::Bmp => probe_bmp(&mut file, path)?,
        ImageInputFormat::Qoi => probe_qoi(&mut file, path)?,
        ImageInputFormat::Tiff => probe_tiff(file, path)?,
        ImageInputFormat::Jxl => probe_jxl(file, path)?,
    };

    Ok(ProbedImage { format, dimensions })
}

fn probe_tiff(file: File, path: &Path) -> Result<ImageDimensions, ProbeError> {
    let mut decoder = tiff::decoder::Decoder::new(BufReader::new(file))
        .map_err(|err| invalid_image(path, format!("invalid TIFF header: {err}")))?;
    let (width, height) = decoder
        .dimensions()
        .map_err(|err| invalid_image(path, format!("invalid TIFF dimensions: {err}")))?;
    let color = decoder
        .colortype()
        .map_err(|err| invalid_image(path, format!("invalid TIFF color type: {err}")))?;
    let bits = match color {
        tiff::ColorType::Gray(bits)
        | tiff::ColorType::GrayA(bits)
        | tiff::ColorType::RGB(bits)
        | tiff::ColorType::RGBA(bits) => bits,
        other => {
            return Err(invalid_image(
                path,
                format!(
                    "unsupported TIFF color type {other:?}; supported are grayscale, gray+alpha, RGB, and RGBA"
                ),
            ));
        }
    };
    if !matches!(bits, 1..=16) {
        return Err(invalid_image(
            path,
            format!("unsupported TIFF sample depth {bits} (expected 1..=16 bits)"),
        ));
    }
    if width == 0 || height == 0 {
        return Err(invalid_image(
            path,
            format!("invalid TIFF dimensions {width}x{height}"),
        ));
    }
    Ok(ImageDimensions::new(width, height)?)
}

fn probe_jxl(file: File, path: &Path) -> Result<ImageDimensions, ProbeError> {
    let mut reader = BufReader::new(file);
    let mut options = jxl::api::JxlDecoderOptions::default();
    options.sample_limit = Some(MAX_JXL_SAMPLES);
    let mut decoder = jxl::api::JxlDecoder::new(options);

    loop {
        match decoder.process(&mut reader, None, None) {
            Ok(jxl::api::Event::BasicInfo) => {
                let info = decoder.basic_info().ok_or_else(|| {
                    invalid_image(path, "missing JPEG XL basic image information")
                })?;
                if info.animation.is_some() {
                    return Err(invalid_image(
                        path,
                        "animated JPEG XL is not supported by the still-image pipeline",
                    ));
                }
                let (width, height) = info.size;
                let width = u32::try_from(width).map_err(|_| {
                    invalid_image(path, format!("JPEG XL width {width} is too large"))
                })?;
                let height = u32::try_from(height).map_err(|_| {
                    invalid_image(path, format!("JPEG XL height {height} is too large"))
                })?;
                return Ok(ImageDimensions::new(width, height)?);
            }
            Ok(jxl::api::Event::NeedMoreInput { .. }) => {
                let remaining = reader.available_bytes().map_err(|source| ProbeError::Io {
                    path: path.to_path_buf(),
                    source,
                })?;
                if remaining == 0 {
                    return Err(invalid_image(path, "truncated JPEG XL stream"));
                }
            }
            Ok(jxl::api::Event::Complete) => {
                return Err(invalid_image(
                    path,
                    "JPEG XL codestream contains no visible frame",
                ));
            }
            Ok(jxl::api::Event::FrameHeader | jxl::api::Event::FrameComplete { .. }) => {}
            Err(err) => {
                return Err(invalid_image(
                    path,
                    format!("invalid JPEG XL stream: {err}"),
                ));
            }
        }
    }
}

/// Probes image format via magic bytes and reads header dimensions without decoding pixels.
///
/// # Errors
///
/// Returns [`ProbeError::Io`] if `path` cannot be opened/seeked, [`ProbeError::InvalidImage`]
/// if magic bytes, headers, or pixel/color encodings are not valid/supported, or
/// [`ProbeError::InvalidDomain`] if dimensions are zero.
pub fn probe_image(path: &Path) -> Result<ProbedImage, ProbeError> {
    let mut file = File::open(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let (prefix, len) = read_magic_prefix(&mut file, path)?;
    let header_slice = prefix.get(..len).unwrap_or(&prefix);
    let format = sniff_magic_format(header_slice, false)
        .ok_or_else(|| invalid_image(path, UNRECOGNIZED_MAGIC_REASON))?;

    probe_opened_image(file, path, format)
}

/// Validates a single image file or mixed image folder at the boundary against live host
/// hardware and returns a compile-time-legal [`ImageBatchPlan`].
///
/// Avoids initializing GPU probing (`wgpu`/Vulkan) when [`HardwareBackend::Cpu`] is explicitly
/// requested so CPU invocations incur zero loader overhead. Never performs silent CPU fallbacks.
///
/// # Errors
///
/// Returns [`ProbeError`] on missing/corrupt inputs, incompatible target paths or extensions,
/// empty source directories, or backends unavailable/unsupported for the chosen output format.
pub fn prepare_image_batch(
    source: &Path,
    target: &Path,
    compression: Option<ImageCompression>,
    output_format: Option<ImageOutputFormat>,
    worker_threads: Option<NonZeroUsize>,
    backend: HardwareBackend,
) -> Result<ImageBatchPlan, ProbeError> {
    prepare_image_batch_inner(
        source,
        target,
        compression,
        output_format,
        worker_threads,
        backend,
        None,
    )
}

fn prepare_image_batch_inner(
    source: &Path,
    target: &Path,
    compression: Option<ImageCompression>,
    output_format: Option<ImageOutputFormat>,
    worker_threads: Option<NonZeroUsize>,
    backend: HardwareBackend,
    hardware: Option<&HostHardware>,
) -> Result<ImageBatchPlan, ProbeError> {
    if source.as_os_str().is_empty() {
        return Err(ProbeError::InvalidDomain(DomainError::EmptyPath(
            PathRole::Source,
        )));
    }
    if target.as_os_str().is_empty() {
        return Err(ProbeError::InvalidDomain(DomainError::EmptyPath(
            PathRole::Target,
        )));
    }

    let threads = match worker_threads {
        Some(count) => count,
        None => available_cpu_threads()?,
    };

    let source_meta = fs::metadata(source).map_err(|err| ProbeError::Io {
        path: source.to_path_buf(),
        source: err,
    })?;

    if source_meta.is_file() {
        let (resolved_target, resolved_format) =
            resolve_single_target(source, target, output_format)?;
        let mode = resolve_execution_mode(resolved_format, compression, backend, hardware)?;
        let probed = probe_image(source)?;
        reject_canonical_same_path(source, &resolved_target)?;
        let item = ImageBatchItem::new(
            source.to_path_buf(),
            resolved_target,
            probed.format,
            probed.dimensions,
        )?;
        return Ok(ImageBatchPlan::new(vec![item], mode, threads)?);
    }

    if source_meta.is_dir() {
        let resolved_format = output_format.unwrap_or(ImageOutputFormat::Webp);
        let mode = resolve_execution_mode(resolved_format, compression, backend, hardware)?;
        return prepare_directory_batch(source, target, mode, threads);
    }

    Err(ProbeError::NotAFileOrDirectory(source.to_path_buf()))
}

fn resolve_execution_mode(
    format: ImageOutputFormat,
    compression: Option<ImageCompression>,
    backend: HardwareBackend,
    hardware: Option<&HostHardware>,
) -> Result<ImageExecutionMode, ProbeError> {
    match (backend, hardware) {
        (HardwareBackend::Cpu, _) => Ok(ImageExecutionMode::resolve_cpu(format, compression)?),
        (HardwareBackend::Gpu(_), Some(hw)) => Ok(ImageExecutionMode::resolve(
            format,
            compression,
            backend,
            hw,
        )?),
        (HardwareBackend::Gpu(_), None) => {
            let detected = detect_host_hardware()?;
            Ok(ImageExecutionMode::resolve(
                format,
                compression,
                backend,
                &detected,
            )?)
        }
    }
}

fn prepare_directory_batch(
    source_dir: &Path,
    target_dir: &Path,
    mode: ImageExecutionMode,
    threads: NonZeroUsize,
) -> Result<ImageBatchPlan, ProbeError> {
    match fs::metadata(target_dir) {
        Ok(meta) if meta.is_dir() => {}
        Ok(_) => {
            return Err(ProbeError::TargetNotADirectory(target_dir.to_path_buf()));
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => {
            return Err(ProbeError::Io {
                path: target_dir.to_path_buf(),
                source: err,
            });
        }
    }

    let entries = fs::read_dir(source_dir).map_err(|err| ProbeError::Io {
        path: source_dir.to_path_buf(),
        source: err,
    })?;

    let mut probed_entries = Vec::new();
    for entry_result in entries {
        let entry = entry_result.map_err(|err| ProbeError::Io {
            path: source_dir.to_path_buf(),
            source: err,
        })?;
        let path = entry.path();
        if let Some(probed) = probe_directory_entry(&path)? {
            probed_entries.push((path, probed));
        }
    }
    probed_entries.sort_by(|(a, _), (b, _)| a.cmp(b));

    if probed_entries.is_empty() {
        return Err(ProbeError::EmptySourceDirectory(source_dir.to_path_buf()));
    }

    let format = mode.output_format();
    let mut items = Vec::with_capacity(probed_entries.len());
    for (src_path, probed) in probed_entries {
        let out_path = derive_target_in_dir(&src_path, target_dir, format)?;
        reject_canonical_same_path(&src_path, &out_path)?;
        items.push(ImageBatchItem::new(
            src_path,
            out_path,
            probed.format,
            probed.dimensions,
        )?);
    }

    Ok(ImageBatchPlan::new(items, mode, threads)?)
}

fn probe_directory_entry(path: &Path) -> Result<Option<ProbedImage>, ProbeError> {
    let is_hidden = path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'));
    if is_hidden {
        return Ok(None);
    }

    let meta = fs::metadata(path).map_err(|err| ProbeError::Io {
        path: path.to_path_buf(),
        source: err,
    })?;
    if !meta.is_file() {
        return Ok(None);
    }

    let mut file = File::open(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let (prefix, len) = read_magic_prefix(&mut file, path)?;
    let header_slice = prefix.get(..len).unwrap_or(&prefix);
    let has_bmp_ext = has_extension(path, "bmp");
    if let Some(format) = sniff_magic_format(header_slice, !has_bmp_ext) {
        return probe_opened_image(file, path, format).map(Some);
    }

    if has_any_extension(path, KNOWN_IMAGE_EXTENSIONS) {
        return Err(invalid_image(path, UNRECOGNIZED_MAGIC_REASON));
    }

    Ok(None)
}

fn resolve_single_target(
    source: &Path,
    target: &Path,
    explicit_format: Option<ImageOutputFormat>,
) -> Result<(PathBuf, ImageOutputFormat), ProbeError> {
    match fs::metadata(target) {
        Ok(meta) if meta.is_dir() => {
            let format = explicit_format.unwrap_or(ImageOutputFormat::Webp);
            let derived = derive_target_in_dir(source, target, format)?;
            return Ok((derived, format));
        }
        Ok(meta) if !meta.is_file() => {
            return Err(ProbeError::NotAFileOrDirectory(target.to_path_buf()));
        }
        Ok(_) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => {
            return Err(ProbeError::Io {
                path: target.to_path_buf(),
                source: err,
            });
        }
    }

    let inferred = infer_format_from_extension(target);
    match (explicit_format, inferred) {
        (Some(expected), Some(actual)) if expected == actual => {
            Ok((target.to_path_buf(), expected))
        }
        (Some(expected), _) => Err(ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: match expected {
                ImageOutputFormat::Webp => ".webp",
                ImageOutputFormat::Heic => ".heic",
                ImageOutputFormat::Jpeg => ".jpg or .jpeg",
                ImageOutputFormat::Png => ".png",
                ImageOutputFormat::Bmp => ".bmp",
                ImageOutputFormat::Qoi => ".qoi",
                ImageOutputFormat::Tiff => ".tif or .tiff",
                ImageOutputFormat::Jxl => ".jxl",
                ImageOutputFormat::Avif => ".avif",
            },
        }),
        (None, Some(actual)) => Ok((target.to_path_buf(), actual)),
        (None, None) => Err(ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: ".webp, .heic, .jpg, .jpeg, .png, .bmp, .qoi, .tif, .tiff, .jxl, or .avif",
        }),
    }
}

fn infer_format_from_extension(path: &Path) -> Option<ImageOutputFormat> {
    if has_extension(path, "webp") {
        Some(ImageOutputFormat::Webp)
    } else if has_extension(path, "heic") {
        Some(ImageOutputFormat::Heic)
    } else if has_any_extension(path, &["jpg", "jpeg"]) {
        Some(ImageOutputFormat::Jpeg)
    } else if has_extension(path, "png") {
        Some(ImageOutputFormat::Png)
    } else if has_extension(path, "bmp") {
        Some(ImageOutputFormat::Bmp)
    } else if has_extension(path, "qoi") {
        Some(ImageOutputFormat::Qoi)
    } else if has_any_extension(path, &["tif", "tiff"]) {
        Some(ImageOutputFormat::Tiff)
    } else if has_extension(path, "jxl") {
        Some(ImageOutputFormat::Jxl)
    } else if has_extension(path, "avif") {
        Some(ImageOutputFormat::Avif)
    } else {
        None
    }
}

fn derive_target_in_dir(
    source: &Path,
    target_dir: &Path,
    format: ImageOutputFormat,
) -> Result<PathBuf, ProbeError> {
    let stem = source.file_stem().ok_or_else(|| ProbeError::InvalidImage {
        path: source.to_path_buf(),
        reason: "source path has no file stem".to_owned(),
    })?;
    let mut name = stem.to_os_string();
    name.push(".");
    name.push(format.extension());
    Ok(target_dir.join(name))
}

fn has_extension(path: &Path, expected: &str) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case(expected))
}

fn has_any_extension(path: &Path, candidates: &[&str]) -> bool {
    candidates.iter().any(|ext| has_extension(path, ext))
}

fn reject_canonical_same_path(source: &Path, target: &Path) -> Result<(), ProbeError> {
    let canon_target = match fs::canonicalize(target) {
        Ok(path) => path,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => {
            return Err(ProbeError::Io {
                path: target.to_path_buf(),
                source: err,
            });
        }
    };
    let canon_source = fs::canonicalize(source).map_err(|err| ProbeError::Io {
        path: source.to_path_buf(),
        source: err,
    })?;
    if canon_source == canon_target {
        return Err(ProbeError::InvalidDomain(
            DomainError::IdenticalSourceAndTarget(source.to_path_buf()),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ProbedImage, prepare_image_batch, prepare_image_batch_inner, probe_image};
    use crate::ProbeError;
    use fastconv_domain::{
        CpuInfo, DetectedGpuBackend, DomainError, GpuBackend, GpuCapability, GpuDeviceType,
        GpuVendor, GraphicsApi, HardwareBackend, HostHardware, ImageCompression, ImageDimensions,
        ImageExecutionMode, ImageInputFormat, ImageOutputFormat, ImageQuality, PathRole, SimdLevel,
        WgpuAdapterInfo,
    };
    use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder};
    use std::env;
    use std::fs::{self, File};
    use std::io::BufWriter;
    use std::num::{NonZeroU64, NonZeroUsize};
    use std::path::{Path, PathBuf};
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn create_temp_dir(label: &str) -> Result<PathBuf, ProbeError> {
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("fastconv-probe-{label}-{}-{id}", process::id()));
        fs::create_dir_all(&dir).map_err(|source| ProbeError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(dir)
    }

    fn write_tiny_png(path: &Path, width: u32, height: u32) -> Result<(), ProbeError> {
        let file = File::create(path).map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|err| ProbeError::InvalidImage {
                path: path.to_path_buf(),
                reason: err.to_string(),
            })?;
        let data_len = usize::try_from(width * height * 3).unwrap_or(0);
        writer
            .write_image_data(&vec![128_u8; data_len])
            .map_err(|err| ProbeError::InvalidImage {
                path: path.to_path_buf(),
                reason: err.to_string(),
            })?;
        Ok(())
    }

    fn write_tiny_jpeg(path: &Path, width: u16, height: u16) -> Result<(), ProbeError> {
        let data_len = usize::from(width) * usize::from(height) * 3;
        let encoder = JpegEncoder::new_file(path, 85).map_err(|err| ProbeError::InvalidImage {
            path: path.to_path_buf(),
            reason: err.to_string(),
        })?;
        encoder
            .encode(&vec![140_u8; data_len], width, height, JpegColorType::Rgb)
            .map_err(|err| ProbeError::InvalidImage {
                path: path.to_path_buf(),
                reason: err.to_string(),
            })?;
        Ok(())
    }

    fn write_tiny_webp(
        path: &Path,
        width: u32,
        height: u32,
        lossless: bool,
        with_alpha: bool,
    ) -> Result<(), ProbeError> {
        let bpp = if with_alpha { 4_usize } else { 3_usize };
        let len = usize::try_from(width * height).unwrap_or(0) * bpp;
        let pixels = vec![180_u8; len];
        let cfg = if lossless {
            zenwebp::EncoderConfig::new_lossless()
        } else {
            zenwebp::EncoderConfig::new().quality(80.0)
        };
        let encoded = if with_alpha {
            zenwebp::Encoder::new_rgba(&pixels, width, height)
                .config(cfg)
                .encode()
        } else {
            zenwebp::Encoder::new_rgb(&pixels, width, height)
                .config(cfg)
                .encode()
        }
        .map_err(|err| ProbeError::InvalidImage {
            path: path.to_path_buf(),
            reason: err.to_string(),
        })?;
        fs::write(path, encoded).map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })
    }

    fn write_tiny_bmp(path: &Path, width: u32, height: i32) -> Result<(), ProbeError> {
        let abs_h = height.unsigned_abs();
        let row_stride = (width * 3).div_ceil(4) * 4;
        let pixel_bytes = row_stride * abs_h;
        let file_size = 54 + pixel_bytes;
        let mut bytes = Vec::with_capacity(usize::try_from(file_size).unwrap_or(54));
        bytes.extend_from_slice(b"BM");
        bytes.extend_from_slice(&file_size.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&54_u32.to_le_bytes());
        bytes.extend_from_slice(&40_u32.to_le_bytes());
        bytes.extend_from_slice(&i32::try_from(width).unwrap_or(1).to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&24_u16.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&pixel_bytes.to_le_bytes());
        bytes.extend_from_slice(&[0_u8; 16]);
        bytes.resize(usize::try_from(file_size).unwrap_or(54), 90);
        fs::write(path, bytes).map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })
    }

    fn write_tiny_qoi(
        path: &Path,
        width: u32,
        height: u32,
        channels: u8,
    ) -> Result<(), ProbeError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"qoif");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.push(channels);
        bytes.push(0);
        let mut remaining = usize::try_from(width * height).unwrap_or(0);
        while remaining > 0 {
            let run = remaining.min(62);
            bytes.push(0xC0 | u8::try_from(run - 1).unwrap_or(0));
            remaining -= run;
        }
        bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]);
        fs::write(path, bytes).map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })
    }

    fn synthetic_rtx_hardware() -> Result<HostHardware, ProbeError> {
        let cores =
            NonZeroUsize::new(4).ok_or(ProbeError::InvalidDomain(DomainError::EmptyBatch))?;
        let threads =
            NonZeroUsize::new(8).ok_or(ProbeError::InvalidDomain(DomainError::EmptyBatch))?;
        let cpu = CpuInfo::new(
            "Test CPU",
            "x86_64",
            cores,
            threads,
            SimdLevel::X86Avx2Fma,
            NonZeroU64::new(16 * 1024 * 1024 * 1024),
        )?;
        let adapter = WgpuAdapterInfo::new(
            "NVIDIA GeForce RTX 2060",
            GpuVendor::Nvidia,
            0x1e89,
            GpuDeviceType::Discrete,
            GraphicsApi::Vulkan,
            "NVIDIA",
            "620.73.04",
        )?;
        let backends = vec![
            DetectedGpuBackend::new(
                GpuBackend::Nvenc,
                GpuCapability::VideoEncodeAndDecode,
                "NVIDIA GeForce RTX 2060",
                "/usr/lib/libnvidia-encode.so.1",
            )?,
            DetectedGpuBackend::new(
                GpuBackend::Vulkan,
                GpuCapability::ComputeOnly,
                "NVIDIA GeForce RTX 2060",
                "wgpu vulkan",
            )?,
            DetectedGpuBackend::new(
                GpuBackend::Vaapi,
                GpuCapability::VideoDecodeOnly,
                "NVIDIA GeForce RTX 2060",
                "/usr/lib/libva.so.2 + /usr/lib/dri/nvidia_drv_video.so",
            )?,
        ];
        Ok(HostHardware::new(cpu, vec![adapter], backends))
    }

    #[test]
    fn prepares_single_image_targets() -> Result<(), ProbeError> {
        let root = create_temp_dir("single")?;
        let png_path = root.join("alpha.png");
        let jpg_path = root.join("beta.JPG");
        write_tiny_png(&png_path, 16, 12)?;
        write_tiny_jpeg(&jpg_path, 32, 24)?;

        assert_eq!(
            probe_image(&png_path)?,
            ProbedImage {
                format: ImageInputFormat::Png,
                dimensions: ImageDimensions::new(16, 12)?,
            }
        );
        assert_eq!(
            probe_image(&jpg_path)?,
            ProbedImage {
                format: ImageInputFormat::Jpeg,
                dimensions: ImageDimensions::new(32, 24)?,
            }
        );

        let rtx = synthetic_rtx_hardware()?;
        let lossy = Some(ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        });
        let single_jpeg = prepare_image_batch(
            &png_path,
            &root.join("single.jpeg"),
            lossy,
            Some(ImageOutputFormat::Jpeg),
            None,
            HardwareBackend::Cpu,
        )?;
        assert_eq!(
            single_jpeg.mode(),
            ImageExecutionMode::CpuJpeg {
                quality: ImageQuality::DEFAULT
            }
        );

        let single_png = prepare_image_batch(
            &jpg_path,
            &root.join("single.png"),
            None,
            None,
            None,
            HardwareBackend::Cpu,
        )?;
        assert_eq!(single_png.mode(), ImageExecutionMode::CpuPng);

        let single_bmp = prepare_image_batch(
            &png_path,
            &root.join("single.bmp"),
            None,
            Some(ImageOutputFormat::Bmp),
            None,
            HardwareBackend::Cpu,
        )?;
        assert_eq!(single_bmp.mode(), ImageExecutionMode::CpuBmp);

        let single_qoi = prepare_image_batch(
            &png_path,
            &root.join("single.qoi"),
            None,
            Some(ImageOutputFormat::Qoi),
            None,
            HardwareBackend::Cpu,
        )?;
        assert_eq!(single_qoi.mode(), ImageExecutionMode::CpuQoi);

        let heic_single = prepare_image_batch_inner(
            &jpg_path,
            &root.join("nvenc.heic"),
            lossy,
            None,
            None,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            Some(&rtx),
        )?;
        assert_eq!(
            heic_single.items()[0].input_format(),
            ImageInputFormat::Jpeg
        );
        assert!(matches!(
            heic_single.mode(),
            ImageExecutionMode::NvencHeic { .. }
        ));

        let out_dir = root.join("out");
        fs::create_dir_all(&out_dir).map_err(|source| ProbeError::Io {
            path: out_dir.clone(),
            source,
        })?;
        let into_dir =
            prepare_image_batch(&jpg_path, &out_dir, lossy, None, None, HardwareBackend::Cpu)?;
        assert_eq!(into_dir.items()[0].target(), out_dir.join("beta.webp"));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn prepares_mixed_directory_batches() -> Result<(), ProbeError> {
        let root = create_temp_dir("batch")?;
        let src_dir = root.join("src");
        let out_dir = root.join("out");
        fs::create_dir_all(&src_dir).map_err(|source| ProbeError::Io {
            path: src_dir.clone(),
            source,
        })?;

        let img_a = src_dir.join("01_alpha.png");
        let img_b = src_dir.join("02_beta.JPG");
        let img_c = src_dir.join("03_gamma.webp");
        let img_d = src_dir.join("04_delta.bmp");
        let img_e = src_dir.join("05_epsilon.qoi");
        let hidden = src_dir.join(".hidden.jpg");
        let non_img = src_dir.join("notes.txt");
        let bm_text = src_dir.join("bmx.txt");
        write_tiny_png(&img_a, 16, 12)?;
        write_tiny_jpeg(&img_b, 32, 24)?;
        write_tiny_webp(&img_c, 20, 14, false, false)?;
        write_tiny_bmp(&img_d, 15, -10)?;
        write_tiny_qoi(&img_e, 18, 12, 4)?;
        write_tiny_jpeg(&hidden, 8, 8)?;
        fs::write(&non_img, b"ignore").map_err(|source| ProbeError::Io {
            path: non_img.clone(),
            source,
        })?;
        fs::write(&bm_text, b"BMX riding notes are not a bitmap").map_err(|source| {
            ProbeError::Io {
                path: bm_text.clone(),
                source,
            }
        })?;

        let rtx = synthetic_rtx_hardware()?;
        let lossy = Some(ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        });
        let heic_batch = prepare_image_batch_inner(
            &src_dir,
            &out_dir,
            lossy,
            Some(ImageOutputFormat::Heic),
            None,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            Some(&rtx),
        )?;
        assert_eq!(heic_batch.items().len(), 5);
        assert_eq!(heic_batch.items()[0].input_format(), ImageInputFormat::Png);
        assert_eq!(heic_batch.items()[1].input_format(), ImageInputFormat::Jpeg);
        assert_eq!(heic_batch.items()[2].input_format(), ImageInputFormat::Webp);
        assert_eq!(heic_batch.items()[3].input_format(), ImageInputFormat::Bmp);
        assert_eq!(heic_batch.items()[4].input_format(), ImageInputFormat::Qoi);
        assert_eq!(
            heic_batch.items()[3].dimensions(),
            ImageDimensions::new(15, 10)?
        );
        assert_eq!(
            heic_batch.items()[4].target(),
            out_dir.join("05_epsilon.heic")
        );

        let png_batch = prepare_image_batch(
            &src_dir,
            &out_dir,
            None,
            Some(ImageOutputFormat::Png),
            None,
            HardwareBackend::Cpu,
        )?;
        assert_eq!(png_batch.mode(), ImageExecutionMode::CpuPng);
        assert_eq!(png_batch.items().len(), 5);
        assert_eq!(png_batch.items()[2].target(), out_dir.join("03_gamma.png"));

        let qoi_batch = prepare_image_batch(
            &src_dir,
            &out_dir,
            None,
            Some(ImageOutputFormat::Qoi),
            None,
            HardwareBackend::Cpu,
        )?;
        assert_eq!(qoi_batch.mode(), ImageExecutionMode::CpuQoi);
        assert_eq!(qoi_batch.items()[0].target(), out_dir.join("01_alpha.qoi"));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn probes_webp_bmp_qoi_via_magic_bytes_and_rejects_corrupt_or_animated()
    -> Result<(), ProbeError> {
        let root = create_temp_dir("magic-new")?;

        let webp_lossy = root.join("misnamed_lossy.png");
        write_tiny_webp(&webp_lossy, 24, 16, false, false)?;
        assert_eq!(
            probe_image(&webp_lossy)?,
            ProbedImage {
                format: ImageInputFormat::Webp,
                dimensions: ImageDimensions::new(24, 16)?,
            }
        );

        let webp_lossless = root.join("misnamed_lossless.jpg");
        write_tiny_webp(&webp_lossless, 20, 14, true, false)?;
        assert_eq!(
            probe_image(&webp_lossless)?,
            ProbedImage {
                format: ImageInputFormat::Webp,
                dimensions: ImageDimensions::new(20, 14)?,
            }
        );

        let webp_vp8x = root.join("extended_alpha.bin");
        write_tiny_webp(&webp_vp8x, 18, 12, false, true)?;
        assert_eq!(
            probe_image(&webp_vp8x)?,
            ProbedImage {
                format: ImageInputFormat::Webp,
                dimensions: ImageDimensions::new(18, 12)?,
            }
        );

        // Synthetic animated VP8X header (animation bit 0x02 set) must be rejected cleanly.
        let animated_webp = root.join("anim.webp");
        let mut anim_bytes = Vec::new();
        anim_bytes.extend_from_slice(b"RIFF");
        anim_bytes.extend_from_slice(&22_u32.to_le_bytes());
        anim_bytes.extend_from_slice(b"WEBP");
        anim_bytes.extend_from_slice(b"VP8X");
        anim_bytes.extend_from_slice(&10_u32.to_le_bytes());
        anim_bytes.extend_from_slice(&[0x02, 0, 0, 0, 15, 0, 0, 15, 0, 0]);
        fs::write(&animated_webp, anim_bytes).map_err(|source| ProbeError::Io {
            path: animated_webp.clone(),
            source,
        })?;
        assert!(matches!(
            probe_image(&animated_webp),
            Err(ProbeError::InvalidImage { ref reason, .. }) if reason.contains("animated WebP")
        ));

        let misnamed_bmp = root.join("actually_bmp.png");
        write_tiny_bmp(&misnamed_bmp, 17, 9)?;
        assert_eq!(
            probe_image(&misnamed_bmp)?,
            ProbedImage {
                format: ImageInputFormat::Bmp,
                dimensions: ImageDimensions::new(17, 9)?,
            }
        );

        let misnamed_qoi = root.join("actually_qoi.bmp");
        write_tiny_qoi(&misnamed_qoi, 19, 11, 3)?;
        assert_eq!(
            probe_image(&misnamed_qoi)?,
            ProbedImage {
                format: ImageInputFormat::Qoi,
                dimensions: ImageDimensions::new(19, 11)?,
            }
        );

        let bad_qoi_tail = root.join("bad_tail.qoi");
        let mut qoi_bytes = fs::read(&misnamed_qoi).map_err(|source| ProbeError::Io {
            path: misnamed_qoi.clone(),
            source,
        })?;
        if let Some(last) = qoi_bytes.last_mut() {
            *last = 0;
        }
        fs::write(&bad_qoi_tail, qoi_bytes).map_err(|source| ProbeError::Io {
            path: bad_qoi_tail.clone(),
            source,
        })?;
        assert!(matches!(
            probe_image(&bad_qoi_tail),
            Err(ProbeError::InvalidImage { .. })
        ));

        let trunc_bmp = root.join("trunc.bmp");
        let mut bmp_bytes = fs::read(&misnamed_bmp).map_err(|source| ProbeError::Io {
            path: misnamed_bmp.clone(),
            source,
        })?;
        bmp_bytes.truncate(60);
        fs::write(&trunc_bmp, bmp_bytes).map_err(|source| ProbeError::Io {
            path: trunc_bmp.clone(),
            source,
        })?;
        assert!(matches!(
            probe_image(&trunc_bmp),
            Err(ProbeError::InvalidImage { .. })
        ));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn probes_jpeg_via_magic_bytes_even_when_misnamed_and_rejects_corrupt_files()
    -> Result<(), ProbeError> {
        let root = create_temp_dir("magic")?;
        let misnamed = root.join("actually_jpeg.png");
        write_tiny_jpeg(&misnamed, 24, 16)?;

        let probed = probe_image(&misnamed)?;
        assert_eq!(probed.format, ImageInputFormat::Jpeg);
        assert_eq!(probed.dimensions, ImageDimensions::new(24, 16)?);

        let corrupt_jpg = root.join("corrupt.jpg");
        fs::write(&corrupt_jpg, b"\xff\xd8\xffnot-a-valid-jpeg").map_err(|source| {
            ProbeError::Io {
                path: corrupt_jpg.clone(),
                source,
            }
        })?;
        assert!(matches!(
            probe_image(&corrupt_jpg),
            Err(ProbeError::InvalidImage { .. })
        ));

        let empty_jpg = root.join("empty.jpg");
        fs::write(&empty_jpg, b"").map_err(|source| ProbeError::Io {
            path: empty_jpg.clone(),
            source,
        })?;
        assert!(matches!(
            probe_image(&empty_jpg),
            Err(ProbeError::InvalidImage { .. })
        ));

        let dir = root.join("dir_with_corrupt");
        fs::create_dir_all(&dir).map_err(|source| ProbeError::Io {
            path: dir.clone(),
            source,
        })?;
        write_tiny_png(&dir.join("good.png"), 8, 8)?;
        fs::write(dir.join("bad.jpeg"), b"not jpeg magic").map_err(|source| ProbeError::Io {
            path: dir.join("bad.jpeg"),
            source,
        })?;
        let batch_err = prepare_image_batch(
            &dir,
            &root.join("out"),
            Some(ImageCompression::Lossy {
                quality: ImageQuality::DEFAULT,
            }),
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(batch_err, Err(ProbeError::InvalidImage { .. })));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn rejects_incompatible_gpu_backends_and_format_mismatches() -> Result<(), ProbeError> {
        let root = create_temp_dir("gpu-errors")?;
        let valid = root.join("good.png");
        write_tiny_png(&valid, 8, 8)?;

        let rtx = synthetic_rtx_hardware()?;
        let lossy = Some(ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        });

        let nvenc_webp = prepare_image_batch_inner(
            &valid,
            &root.join("out.webp"),
            lossy,
            None,
            None,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            Some(&rtx),
        );
        assert!(matches!(
            nvenc_webp,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedImagePipeline {
                    format: ImageOutputFormat::Webp,
                    backend: HardwareBackend::Gpu(GpuBackend::Nvenc),
                    ..
                }
            ))
        ));

        let vulkan_jpeg = prepare_image_batch_inner(
            &valid,
            &root.join("out.jpg"),
            lossy,
            None,
            None,
            HardwareBackend::Gpu(GpuBackend::Vulkan),
            Some(&rtx),
        );
        assert!(matches!(
            vulkan_jpeg,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedImagePipeline {
                    format: ImageOutputFormat::Jpeg,
                    backend: HardwareBackend::Gpu(GpuBackend::Vulkan),
                    ..
                }
            ))
        ));

        let vaapi_err = prepare_image_batch_inner(
            &valid,
            &root.join("out.webp"),
            lossy,
            None,
            None,
            HardwareBackend::Gpu(GpuBackend::Vaapi),
            Some(&rtx),
        );
        assert!(matches!(
            vaapi_err,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedImagePipeline {
                    backend: HardwareBackend::Gpu(GpuBackend::Vaapi),
                    ..
                }
            ))
        ));

        let format_mismatch = prepare_image_batch_inner(
            &valid,
            &root.join("out.webp"),
            lossy,
            Some(ImageOutputFormat::Jpeg),
            None,
            HardwareBackend::Cpu,
            Some(&rtx),
        );
        assert!(matches!(
            format_mismatch,
            Err(ProbeError::UnsupportedTargetFormat {
                expected: ".jpg or .jpeg",
                ..
            })
        ));

        let png_quality_err = prepare_image_batch_inner(
            &valid,
            &root.join("out.png"),
            lossy,
            None,
            None,
            HardwareBackend::Cpu,
            Some(&rtx),
        );
        assert!(matches!(
            png_quality_err,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedImagePipeline {
                    format: ImageOutputFormat::Png,
                    ..
                }
            ))
        ));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn rejects_corrupt_png_bad_extension_and_empty_dir() -> Result<(), ProbeError> {
        let root = create_temp_dir("errors")?;
        let valid = root.join("good.png");
        let corrupt = root.join("corrupt.png");
        let empty_dir = root.join("empty");
        fs::create_dir_all(&empty_dir).map_err(|source| ProbeError::Io {
            path: empty_dir.clone(),
            source,
        })?;
        write_tiny_png(&valid, 8, 8)?;
        fs::write(&corrupt, b"not a png").map_err(|source| ProbeError::Io {
            path: corrupt.clone(),
            source,
        })?;

        let corrupt_err = prepare_image_batch(
            &corrupt,
            &root.join("out.webp"),
            Some(ImageCompression::Lossless),
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(corrupt_err, Err(ProbeError::InvalidImage { .. })));

        let ext_err = prepare_image_batch(
            &valid,
            &root.join("out.gif"),
            Some(ImageCompression::Lossless),
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            ext_err,
            Err(ProbeError::UnsupportedTargetFormat { .. })
        ));

        let jpeg_lossless_err = prepare_image_batch(
            &valid,
            &root.join("out.jpg"),
            Some(ImageCompression::Lossless),
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            jpeg_lossless_err,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedImagePipeline {
                    format: ImageOutputFormat::Jpeg,
                    ..
                }
            ))
        ));

        let empty_err = prepare_image_batch(
            &empty_dir,
            &root.join("out"),
            Some(ImageCompression::Lossless),
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            empty_err,
            Err(ProbeError::EmptySourceDirectory(_))
        ));

        let dir_to_file = prepare_image_batch(
            &root,
            &valid,
            Some(ImageCompression::Lossless),
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            dir_to_file,
            Err(ProbeError::TargetNotADirectory(_))
        ));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn rejects_empty_paths_duplicate_targets_and_same_file_overwrite() -> Result<(), ProbeError> {
        let root = create_temp_dir("collisions")?;
        let compression = Some(ImageCompression::Lossless);

        assert!(matches!(
            prepare_image_batch(
                Path::new(""),
                &root.join("out.webp"),
                compression,
                None,
                None,
                HardwareBackend::Cpu,
            ),
            Err(ProbeError::InvalidDomain(DomainError::EmptyPath(
                PathRole::Source
            )))
        ));
        assert!(matches!(
            prepare_image_batch(
                &root.join("in.png"),
                Path::new(""),
                compression,
                None,
                None,
                HardwareBackend::Cpu,
            ),
            Err(ProbeError::InvalidDomain(DomainError::EmptyPath(
                PathRole::Target
            )))
        ));

        let dup_dir = root.join("dup");
        fs::create_dir_all(&dup_dir).map_err(|source| ProbeError::Io {
            path: dup_dir.clone(),
            source,
        })?;
        write_tiny_png(&dup_dir.join("photo.png"), 8, 8)?;
        write_tiny_jpeg(&dup_dir.join("photo.jpg"), 8, 8)?;
        let dup_err = prepare_image_batch(
            &dup_dir,
            &root.join("out"),
            compression,
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            dup_err,
            Err(ProbeError::InvalidDomain(DomainError::DuplicateTargetPath(
                _
            )))
        ));

        let same_webp = root.join("same.webp");
        write_tiny_png(&same_webp, 8, 8)?;
        let same_err = prepare_image_batch(
            &same_webp,
            &root.join("./same.webp"),
            compression,
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            same_err,
            Err(ProbeError::InvalidDomain(
                DomainError::IdenticalSourceAndTarget(_)
            ))
        ));

        let in_place_dir = root.join("in_place_jpeg");
        fs::create_dir_all(&in_place_dir).map_err(|source| ProbeError::Io {
            path: in_place_dir.clone(),
            source,
        })?;
        write_tiny_jpeg(&in_place_dir.join("img.jpg"), 8, 8)?;
        let in_place_err = prepare_image_batch(
            &in_place_dir,
            &in_place_dir,
            Some(ImageCompression::Lossy {
                quality: ImageQuality::DEFAULT,
            }),
            Some(ImageOutputFormat::Jpeg),
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            in_place_err,
            Err(ProbeError::InvalidDomain(
                DomainError::IdenticalSourceAndTarget(_)
            ))
        ));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn rejects_invalid_and_duplicate_vp8x_frame_chunks() -> Result<(), ProbeError> {
        let root = create_temp_dir("strict-vp8x")?;

        // 1. VP8X with a zero-length VP8L chunk must be rejected.
        let vp8x_empty_vp8l = root.join("vp8x_empty_vp8l.webp");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(4_u32 + 18 + 8).to_le_bytes());
        bytes.extend_from_slice(b"WEBP");
        bytes.extend_from_slice(b"VP8X");
        bytes.extend_from_slice(&10_u32.to_le_bytes());
        bytes.extend_from_slice(&[0, 0, 0, 0, 15, 0, 0, 15, 0, 0]); // 16x16 canvas
        bytes.extend_from_slice(b"VP8L");
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        fs::write(&vp8x_empty_vp8l, &bytes).map_err(|source| ProbeError::Io {
            path: vp8x_empty_vp8l.clone(),
            source,
        })?;
        assert!(matches!(
            probe_image(&vp8x_empty_vp8l),
            Err(ProbeError::InvalidImage { .. })
        ));

        let vp8l_16x16: [u8; 5] = {
            let bits: u32 = 15 | (15 << 14);
            let le = bits.to_le_bytes();
            [0x2F, le[0], le[1], le[2], le[3]]
        };
        let vp8l_8x8: [u8; 5] = {
            let bits: u32 = 7 | (7 << 14);
            let le = bits.to_le_bytes();
            [0x2F, le[0], le[1], le[2], le[3]]
        };

        // 2. VP8X canvas 16x16 with embedded 8x8 VP8L frame must be rejected.
        let vp8x_mismatch = root.join("vp8x_mismatch.webp");
        let mut mismatch = Vec::new();
        mismatch.extend_from_slice(b"RIFF");
        mismatch.extend_from_slice(&(4_u32 + 18 + 14).to_le_bytes());
        mismatch.extend_from_slice(b"WEBP");
        mismatch.extend_from_slice(b"VP8X");
        mismatch.extend_from_slice(&10_u32.to_le_bytes());
        mismatch.extend_from_slice(&[0, 0, 0, 0, 15, 0, 0, 15, 0, 0]);
        mismatch.extend_from_slice(b"VP8L");
        mismatch.extend_from_slice(&5_u32.to_le_bytes());
        mismatch.extend_from_slice(&vp8l_8x8);
        mismatch.push(0);
        fs::write(&vp8x_mismatch, &mismatch).map_err(|source| ProbeError::Io {
            path: vp8x_mismatch.clone(),
            source,
        })?;
        assert!(matches!(
            probe_image(&vp8x_mismatch),
            Err(ProbeError::InvalidImage { .. })
        ));

        // 3. VP8X with two VP8L chunks of the same type must be rejected.
        let vp8x_dup = root.join("vp8x_dup.webp");
        let mut dup = Vec::new();
        dup.extend_from_slice(b"RIFF");
        dup.extend_from_slice(&(4_u32 + 18 + 14 + 14).to_le_bytes());
        dup.extend_from_slice(b"WEBP");
        dup.extend_from_slice(b"VP8X");
        dup.extend_from_slice(&10_u32.to_le_bytes());
        dup.extend_from_slice(&[0, 0, 0, 0, 15, 0, 0, 15, 0, 0]);
        for _ in 0..2 {
            dup.extend_from_slice(b"VP8L");
            dup.extend_from_slice(&5_u32.to_le_bytes());
            dup.extend_from_slice(&vp8l_16x16);
            dup.push(0);
        }
        fs::write(&vp8x_dup, &dup).map_err(|source| ProbeError::Io {
            path: vp8x_dup.clone(),
            source,
        })?;
        assert!(matches!(
            probe_image(&vp8x_dup),
            Err(ProbeError::InvalidImage { .. })
        ));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn rejects_oversized_qoi_and_short_bmp_palettes() -> Result<(), ProbeError> {
        let root = create_temp_dir("strict-qoi-bmp")?;

        // Short QOI declaring 1_000_000 x 1_000_000 must be rejected at probe boundary.
        let bad_qoi = root.join("bomb.qoi");
        let mut qoi = Vec::new();
        qoi.extend_from_slice(b"qoif");
        qoi.extend_from_slice(&1_000_000_u32.to_be_bytes());
        qoi.extend_from_slice(&1_000_000_u32.to_be_bytes());
        qoi.extend_from_slice(&[4, 0, 0xC0, 0, 0, 0, 0, 0, 0, 0, 1]);
        fs::write(&bad_qoi, &qoi).map_err(|source| ProbeError::Io {
            path: bad_qoi.clone(),
            source,
        })?;
        assert!(matches!(
            probe_image(&bad_qoi),
            Err(ProbeError::InvalidImage { .. })
        ));

        // Indexed 8-bit BMP with biClrUsed = 16 but only 8 palette bytes must be rejected.
        let bad_bmp = root.join("short_palette.bmp");
        let mut bmp = Vec::new();
        bmp.extend_from_slice(b"BM");
        bmp.extend_from_slice(&(54_u32 + 8 + 4).to_le_bytes());
        bmp.extend_from_slice(&0_u32.to_le_bytes());
        bmp.extend_from_slice(&(54_u32 + 8).to_le_bytes());
        bmp.extend_from_slice(&40_u32.to_le_bytes());
        bmp.extend_from_slice(&1_i32.to_le_bytes());
        bmp.extend_from_slice(&1_i32.to_le_bytes());
        bmp.extend_from_slice(&1_u16.to_le_bytes());
        bmp.extend_from_slice(&8_u16.to_le_bytes());
        bmp.extend_from_slice(&0_u32.to_le_bytes());
        bmp.extend_from_slice(&4_u32.to_le_bytes());
        bmp.extend_from_slice(&[0_u8; 8]);
        bmp.extend_from_slice(&16_u32.to_le_bytes());
        bmp.extend_from_slice(&0_u32.to_le_bytes());
        bmp.extend_from_slice(&[0_u8; 12]);
        fs::write(&bad_bmp, &bmp).map_err(|source| ProbeError::Io {
            path: bad_bmp.clone(),
            source,
        })?;
        assert!(matches!(
            probe_image(&bad_bmp),
            Err(ProbeError::InvalidImage { .. })
        ));

        // 1-bit BITMAPCOREHEADER with only 1 palette entry (3 bytes) must be rejected (needs 6).
        let bad_core_1bit = root.join("core_1bit_short_palette.bmp");
        let mut core = Vec::new();
        core.extend_from_slice(b"BM");
        core.extend_from_slice(&(14_u32 + 12 + 3 + 4).to_le_bytes());
        core.extend_from_slice(&0_u32.to_le_bytes());
        core.extend_from_slice(&(14_u32 + 12 + 3).to_le_bytes()); // only 3 palette bytes
        core.extend_from_slice(&12_u32.to_le_bytes());
        core.extend_from_slice(&1_u16.to_le_bytes());
        core.extend_from_slice(&1_u16.to_le_bytes());
        core.extend_from_slice(&1_u16.to_le_bytes());
        core.extend_from_slice(&1_u16.to_le_bytes());
        core.extend_from_slice(&[0, 0, 0]); // 1 RGBTRIPLE
        core.extend_from_slice(&[0, 0, 0, 0]); // 1 padded row
        fs::write(&bad_core_1bit, &core).map_err(|source| ProbeError::Io {
            path: bad_core_1bit.clone(),
            source,
        })?;
        assert!(matches!(
            probe_image(&bad_core_1bit),
            Err(ProbeError::InvalidImage { .. })
        ));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }
}
