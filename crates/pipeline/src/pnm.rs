//! Dependency-free Netpbm (PBM/PGM/PPM/PAM) still-image codec.
//!
//! All parsing happens on borrowed bytes with explicit bounds checks so hostile
//! headers can never allocate more than the decoded-pixel budget.

use std::io::{self, Write};
use std::path::Path;

use crate::PipelineError;
use crate::decode::{DecodedFrameMeta, DecodedPixelFormat};

const MAX_PNM_PIXELS: u64 = 128 * 1024 * 1024;

type ParseResult<T> = Result<T, String>;

struct PnmReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> PnmReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_space_and_comments(&mut self) {
        loop {
            match self.peek() {
                Some(b'#') => {
                    while let Some(byte) = self.peek() {
                        self.pos += 1;
                        if byte == b'\n' {
                            break;
                        }
                    }
                }
                Some(byte) if byte.is_ascii_whitespace() => self.pos += 1,
                _ => break,
            }
        }
    }

    fn read_bytes(&mut self, count: usize) -> ParseResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(count)
            .ok_or_else(|| "PNM read offset overflow".to_owned())?;
        let bytes = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| "truncated PNM raster".to_owned())?;
        self.pos = end;
        Ok(bytes)
    }

    fn read_token(&mut self) -> ParseResult<&'a [u8]> {
        self.skip_space_and_comments();
        let start = self.pos;
        while let Some(byte) = self.peek() {
            if byte.is_ascii_whitespace() || byte == b'#' {
                break;
            }
            self.pos += 1;
        }
        if start == self.pos {
            return Err("missing PNM header token".to_owned());
        }
        self.bytes
            .get(start..self.pos)
            .ok_or_else(|| "invalid PNM header token".to_owned())
    }

    fn read_number(&mut self, label: &str, max: u32) -> ParseResult<u32> {
        let token = self.read_token()?;
        let text = std::str::from_utf8(token).map_err(|_| format!("invalid PNM {label}"))?;
        let value: u64 = text.parse().map_err(|_| format!("invalid PNM {label}"))?;
        if value == 0 || value > u64::from(max) {
            return Err(format!("PNM {label} {value} is outside 1..={max}"));
        }
        u32::try_from(value).map_err(|_| format!("PNM {label} is too large"))
    }

    /// Binary raster headers end their last token with exactly one whitespace
    /// byte or one `#...\n` comment; the raster begins on the very next byte.
    fn consume_binary_separator(&mut self) -> ParseResult<()> {
        match self.peek() {
            Some(b'#') => {
                let mut saw_newline = false;
                while let Some(byte) = self.peek() {
                    self.pos += 1;
                    if byte == b'\n' {
                        saw_newline = true;
                        break;
                    }
                }
                if saw_newline {
                    Ok(())
                } else {
                    Err("unterminated PNM header comment".to_owned())
                }
            }
            Some(byte) if byte.is_ascii_whitespace() => {
                self.pos += 1;
                Ok(())
            }
            _ => Err("binary PNM header must end with one whitespace byte".to_owned()),
        }
    }

    /// Accepts either a single `0`/`1` token or a packed run such as `1011`.
    fn read_ascii_bits(&mut self) -> ParseResult<Vec<u8>> {
        let token = self.read_token()?;
        if token.is_empty() || !token.iter().all(|&byte| byte == b'0' || byte == b'1') {
            return Err("invalid PBM bit".to_owned());
        }
        Ok(token.iter().map(|&byte| byte - b'0').collect())
    }
}

struct Header {
    kind: u8,
    width: u32,
    height: u32,
    maxval: u32,
    depth: u8,
}

fn check_dimensions(width: u32, height: u32) -> ParseResult<()> {
    if u64::from(width) * u64::from(height) > MAX_PNM_PIXELS {
        return Err(format!(
            "PNM pixel count {} exceeds the {MAX_PNM_PIXELS}-pixel limit",
            u64::from(width) * u64::from(height)
        ));
    }
    Ok(())
}

fn parse_pam_number(raw: &[u8], label: &str, max: u32) -> ParseResult<u32> {
    let trimmed = raw.trim_ascii();
    let text = std::str::from_utf8(trimmed).map_err(|_| format!("invalid PAM {label}"))?;
    let value: u64 = text.parse().map_err(|_| format!("invalid PAM {label}"))?;
    if value == 0 || value > u64::from(max) {
        return Err(format!("PAM {label} {value} is outside 1..={max}"));
    }
    u32::try_from(value).map_err(|_| format!("PAM {label} is too large"))
}

fn parse_pam_header(reader: &mut PnmReader<'_>) -> ParseResult<Header> {
    let bytes = reader.bytes;
    if !bytes.starts_with(b"P7\n") && !bytes.starts_with(b"P7\r\n") {
        return Err("invalid PAM header signature (expected P7\\n)".to_owned());
    }
    let mut pos = if bytes.starts_with(b"P7\r\n") { 4 } else { 3 };
    let mut width = None;
    let mut height = None;
    let mut depth = None;
    let mut maxval = None;
    let mut endhdr = false;

    while pos < bytes.len() {
        let Some(eol_offset) = bytes[pos..].iter().position(|&byte| byte == b'\n') else {
            break;
        };
        let eol = pos + eol_offset;
        let mut line = &bytes[pos..eol];
        pos = eol + 1;
        if line.last() == Some(&b'\r') {
            line = &line[..line.len() - 1];
        }
        if line.is_empty() || line.first() == Some(&b'#') {
            continue;
        }
        let split = line
            .iter()
            .position(|byte| matches!(byte, b' ' | b'\t' | 0x0B | 0x0C))
            .unwrap_or(line.len());
        let key = &line[..split];
        let rest = &line[split..];
        match key {
            b"WIDTH" => width = Some(parse_pam_number(rest, "width", u32::MAX)?),
            b"HEIGHT" => height = Some(parse_pam_number(rest, "height", u32::MAX)?),
            b"DEPTH" => {
                let parsed = parse_pam_number(rest, "depth", 4)?;
                depth = Some(u8::try_from(parsed).map_err(|_| "invalid PAM depth".to_owned())?);
            }
            b"MAXVAL" => maxval = Some(parse_pam_number(rest, "maxval", 65535)?),
            b"TUPLTYPE" => {}
            b"ENDHDR" => {
                if !rest.trim_ascii().is_empty() {
                    return Err("unexpected trailing tokens on PAM ENDHDR line".to_owned());
                }
                endhdr = true;
                break;
            }
            _ => {
                return Err(format!(
                    "unknown PAM header key: {}",
                    String::from_utf8_lossy(key)
                ));
            }
        }
    }
    if !endhdr {
        return Err("PAM header is missing ENDHDR".to_owned());
    }
    reader.pos = pos;
    let width = width.ok_or_else(|| "PAM header is missing WIDTH".to_owned())?;
    let height = height.ok_or_else(|| "PAM header is missing HEIGHT".to_owned())?;
    let depth = depth.ok_or_else(|| "PAM header is missing DEPTH".to_owned())?;
    let maxval = maxval.ok_or_else(|| "PAM header is missing MAXVAL".to_owned())?;
    check_dimensions(width, height)?;
    Ok(Header {
        kind: b'7',
        width,
        height,
        maxval,
        depth,
    })
}

fn parse_header(reader: &mut PnmReader<'_>) -> ParseResult<Header> {
    if reader.bytes.starts_with(b"P7") {
        return parse_pam_header(reader);
    }
    let magic = reader.read_bytes(2)?;
    if magic[0] != b'P' || !matches!(magic[1], b'1'..=b'6') {
        return Err("invalid PNM magic number".to_owned());
    }
    let kind = magic[1];

    let width = reader.read_number("width", u32::MAX)?;
    let height = reader.read_number("height", u32::MAX)?;
    check_dimensions(width, height)?;
    let maxval = if matches!(kind, b'1' | b'4') {
        1
    } else {
        reader.read_number("maxval", 65535)?
    };
    if matches!(kind, b'4' | b'5' | b'6') {
        reader.consume_binary_separator()?;
    }
    let depth = match kind {
        b'1' | b'4' | b'2' | b'5' => 1,
        b'3' | b'6' => 3,
        _ => 0,
    };
    Ok(Header {
        kind,
        width,
        height,
        maxval,
        depth,
    })
}

#[inline]
fn scale_sample(value: u32, maxval: u32) -> u8 {
    let scaled = (u64::from(value) * 255 + u64::from(maxval) / 2) / u64::from(maxval);
    u8::try_from(scaled).unwrap_or(u8::MAX)
}

fn read_binary_samples(
    reader: &mut PnmReader<'_>,
    count: usize,
    maxval: u32,
) -> ParseResult<Vec<u8>> {
    let bytes_per_sample = if maxval < 256 { 1_usize } else { 2 };
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let bytes = reader.read_bytes(bytes_per_sample)?;
        let value = if bytes_per_sample == 1 {
            u32::from(bytes[0])
        } else {
            (u32::from(bytes[0]) << 8) | u32::from(bytes[1])
        };
        if value > maxval {
            return Err(format!("PNM sample {value} exceeds maxval {maxval}"));
        }
        out.push(scale_sample(value, maxval));
    }
    Ok(out)
}

fn format_for_kind(kind: u8, depth: u8) -> DecodedPixelFormat {
    match (kind, depth) {
        (b'1' | b'2' | b'4' | b'5', _) | (b'7', 1) => DecodedPixelFormat::L8,
        (b'7', 2) => DecodedPixelFormat::La8,
        (b'3' | b'6', _) | (b'7', 3) => DecodedPixelFormat::Rgb8,
        _ => DecodedPixelFormat::Rgba8,
    }
}

fn store_raster(
    source: &Path,
    pixel_buf: &mut Vec<u8>,
    raster: &[u8],
    expected_len: usize,
) -> Result<(), PipelineError> {
    if raster.len() != expected_len {
        return Err(PipelineError::DecodeFailed {
            path: source.to_path_buf(),
            reason: format!(
                "decoded PNM raster has {} bytes, expected {expected_len}",
                raster.len()
            ),
        });
    }
    if pixel_buf.len() < expected_len {
        pixel_buf.resize(expected_len, 0);
    }
    pixel_buf[..expected_len].copy_from_slice(raster);
    Ok(())
}

fn decode_error(source: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::DecodeFailed {
        path: source.to_path_buf(),
        reason: reason.into(),
    }
}

/// Decodes any Netpbm still image (P1..P7) into `pixel_buf`.
pub(crate) fn decode_pnm_slice_into(
    source: &Path,
    file_bytes: &[u8],
    pixel_buf: &mut Vec<u8>,
) -> Result<(u32, u32, DecodedPixelFormat, usize), PipelineError> {
    let mut reader = PnmReader::new(file_bytes);
    let header = parse_header(&mut reader).map_err(|err| decode_error(source, err))?;
    let width_usize = usize::try_from(header.width)
        .map_err(|_| decode_error(source, "PNM width exceeds addressable memory"))?;
    let height_usize = usize::try_from(header.height)
        .map_err(|_| decode_error(source, "PNM height exceeds addressable memory"))?;
    let format = format_for_kind(header.kind, header.depth);
    let bpp = match format {
        DecodedPixelFormat::L8 => 1_usize,
        DecodedPixelFormat::La8 => 2_usize,
        DecodedPixelFormat::Rgb8 => 3_usize,
        DecodedPixelFormat::Rgba8 => 4_usize,
    };
    let pixel_count = width_usize
        .checked_mul(height_usize)
        .ok_or_else(|| decode_error(source, "PNM pixel count overflow"))?;
    let expected_len = pixel_count
        .checked_mul(bpp)
        .ok_or_else(|| decode_error(source, "PNM decoded buffer size overflow"))?;

    match header.kind {
        b'1' => {
            let mut raster = Vec::with_capacity(expected_len);
            while raster.len() < pixel_count {
                let bits = reader
                    .read_ascii_bits()
                    .map_err(|err| decode_error(source, err))?;
                for bit in bits {
                    if raster.len() == pixel_count {
                        break;
                    }
                    raster.push(if bit == 1 { 0 } else { 255 });
                }
            }
            store_raster(source, pixel_buf, &raster, expected_len)?;
        }
        b'4' => {
            let row_bytes = width_usize.div_ceil(8);
            let total = row_bytes
                .checked_mul(height_usize)
                .ok_or_else(|| decode_error(source, "PBM raster size overflow"))?;
            let packed = reader
                .read_bytes(total)
                .map_err(|err| decode_error(source, err))?;
            let mut raster = Vec::with_capacity(expected_len);
            for row in packed.chunks_exact(row_bytes) {
                for x in 0..width_usize {
                    let byte = row[x / 8];
                    let bit = (byte >> (7 - (x % 8))) & 1;
                    raster.push(if bit == 1 { 0 } else { 255 });
                }
            }
            store_raster(source, pixel_buf, &raster, expected_len)?;
        }
        b'2' | b'3' => {
            let samples = pixel_count * usize::from(header.depth);
            let mut raster = Vec::with_capacity(expected_len);
            for _ in 0..samples {
                let token = reader
                    .read_token()
                    .map_err(|err| decode_error(source, err))?;
                let text = std::str::from_utf8(token)
                    .map_err(|err| decode_error(source, err.to_string()))?;
                let value: u64 = text
                    .parse()
                    .map_err(|_| decode_error(source, "invalid ASCII PNM sample"))?;
                if value > u64::from(header.maxval) {
                    return Err(decode_error(
                        source,
                        format!("PNM sample {value} exceeds maxval {}", header.maxval),
                    ));
                }
                raster.push(scale_sample(
                    u32::try_from(value).unwrap_or(u32::MAX),
                    header.maxval,
                ));
            }
            store_raster(source, pixel_buf, &raster, expected_len)?;
        }
        b'5' | b'6' | b'7' => {
            let samples = pixel_count * usize::from(header.depth);
            if header.maxval == 255 {
                let raster = reader
                    .read_bytes(samples)
                    .map_err(|err| decode_error(source, err))?;
                store_raster(source, pixel_buf, raster, expected_len)?;
            } else {
                let raster = read_binary_samples(&mut reader, samples, header.maxval)
                    .map_err(|err| decode_error(source, err))?;
                store_raster(source, pixel_buf, &raster, expected_len)?;
            }
        }
        other => {
            return Err(decode_error(
                source,
                format!("unsupported PNM magic number P{other}"),
            ));
        }
    }
    Ok((header.width, header.height, format, expected_len))
}

struct CountingWriter<'a, W: Write> {
    inner: &'a mut W,
    total: u64,
}

impl<'a, W: Write> CountingWriter<'a, W> {
    const fn new(inner: &'a mut W) -> Self {
        Self { inner, total: 0 }
    }
}

impl<W: Write> Write for CountingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(buf)?;
        self.total = self
            .total
            .saturating_add(u64::try_from(written).unwrap_or(u64::MAX));
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn io_error(target: &Path, source: io::Error) -> PipelineError {
    PipelineError::Io {
        path: target.to_path_buf(),
        source,
    }
}

fn encode_error(target: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::EncodeFailed {
        path: target.to_path_buf(),
        reason: reason.into(),
    }
}

fn unsupported(target: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::UnsupportedPixelFormat {
        path: target.to_path_buf(),
        format: reason.into(),
    }
}

fn expand_l8_to_rgb<'a>(
    aux_buf: &'a mut Vec<u8>,
    pixels: &[u8],
    count: usize,
) -> Result<&'a [u8], String> {
    let needed = count
        .checked_mul(3)
        .ok_or_else(|| "PNM RGB expand overflow".to_owned())?;
    if aux_buf.len() < needed {
        aux_buf.resize(needed, 0);
    }
    for (dst, &value) in aux_buf[..needed]
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(pixels.iter())
    {
        dst.copy_from_slice(&[value, value, value]);
    }
    Ok(&aux_buf[..needed])
}

fn strip_opaque_alpha(
    target: &Path,
    pixels: &[u8],
    bpp: usize,
    color_samples: usize,
) -> Result<Vec<u8>, PipelineError> {
    let count = pixels.len() / bpp;
    let mut out = Vec::with_capacity(count * color_samples);
    for pixel in pixels.chunks_exact(bpp) {
        if pixel[color_samples] != 255 {
            return Err(unsupported(
                target,
                "non-opaque alpha channel (binary PNM has no alpha)",
            ));
        }
        out.extend_from_slice(&pixel[..color_samples]);
    }
    Ok(out)
}

fn pack_bilevel(
    target: &Path,
    pixels: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, PipelineError> {
    let row_bytes = width.div_ceil(8);
    let mut packed = vec![0_u8; row_bytes * height];
    for (y, row) in packed.chunks_exact_mut(row_bytes).enumerate() {
        for (x, &value) in pixels[y * width..(y + 1) * width].iter().enumerate() {
            match value {
                0 => row[x / 8] |= 1 << (7 - (x % 8)),
                255 => {}
                other => {
                    return Err(unsupported(
                        target,
                        format!("PBM requires bilevel L8 pixels (0 or 255), got {other}"),
                    ));
                }
            }
        }
    }
    Ok(packed)
}

/// Encodes one still image into the Netpbm variant selected by `target`'s extension.
#[allow(clippy::too_many_lines)] // The per-extension branches share one counted writer and pixel validation.
pub(crate) fn encode_pnm_to_writer<W: Write>(
    target: &Path,
    pixels: &[u8],
    decoded: DecodedFrameMeta,
    aux_buf: &mut Vec<u8>,
    writer: &mut W,
) -> Result<u64, PipelineError> {
    let width = usize::try_from(decoded.width)
        .map_err(|_| encode_error(target, "PNM width exceeds addressable memory"))?;
    let height = usize::try_from(decoded.height)
        .map_err(|_| encode_error(target, "PNM height exceeds addressable memory"))?;
    let bpp = match decoded.format {
        DecodedPixelFormat::L8 => 1_usize,
        DecodedPixelFormat::La8 => 2_usize,
        DecodedPixelFormat::Rgb8 => 3_usize,
        DecodedPixelFormat::Rgba8 => 4_usize,
    };
    let expected = width
        .checked_mul(height)
        .and_then(|count| count.checked_mul(bpp))
        .ok_or_else(|| encode_error(target, "PNM pixel buffer size overflow"))?;
    if pixels.len() != expected {
        return Err(encode_error(
            target,
            format!(
                "PNM decoded pixel buffer has {} bytes, expected {expected}",
                pixels.len()
            ),
        ));
    }

    let extension = target
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mut counted = CountingWriter::new(writer);
    let write_all =
        |counted: &mut CountingWriter<'_, W>, bytes: &[u8]| -> Result<(), PipelineError> {
            counted
                .write_all(bytes)
                .map_err(|source| io_error(target, source))
        };
    let write_header = |counted: &mut CountingWriter<'_, W>,
                        header: String|
     -> Result<(), PipelineError> { write_all(counted, header.as_bytes()) };

    match extension.as_str() {
        "pam" => {
            let (depth, tuple_type) = match decoded.format {
                DecodedPixelFormat::L8 => (1_u8, "GRAYSCALE"),
                DecodedPixelFormat::La8 => (2, "GRAYSCALE_ALPHA"),
                DecodedPixelFormat::Rgb8 => (3, "RGB"),
                DecodedPixelFormat::Rgba8 => (4, "RGB_ALPHA"),
            };
            write_header(
                &mut counted,
                format!(
                    "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH {depth}\nMAXVAL 255\nTUPLTYPE {tuple_type}\nENDHDR\n"
                ),
            )?;
            write_all(&mut counted, pixels)?;
        }
        "ppm" => {
            let data = match decoded.format {
                DecodedPixelFormat::Rgb8 => pixels,
                DecodedPixelFormat::L8 => expand_l8_to_rgb(aux_buf, pixels, width * height)
                    .map_err(|reason| encode_error(target, reason))?,
                DecodedPixelFormat::La8 => {
                    let gray = strip_opaque_alpha(target, pixels, bpp, 1)?;
                    expand_l8_to_rgb(aux_buf, &gray, width * height)
                        .map_err(|reason| encode_error(target, reason))?
                }
                DecodedPixelFormat::Rgba8 => {
                    let rgb = strip_opaque_alpha(target, pixels, bpp, 3)?;
                    aux_buf.clear();
                    aux_buf.extend_from_slice(&rgb);
                    aux_buf.as_slice()
                }
            };
            write_header(&mut counted, format!("P6\n{width} {height}\n255\n"))?;
            write_all(&mut counted, data)?;
        }
        "pgm" => {
            let data = match decoded.format {
                DecodedPixelFormat::L8 => pixels,
                DecodedPixelFormat::La8 => {
                    let converted = strip_opaque_alpha(target, pixels, bpp, 1)?;
                    aux_buf.clear();
                    aux_buf.extend_from_slice(&converted);
                    aux_buf.as_slice()
                }
                DecodedPixelFormat::Rgb8 => {
                    return Err(unsupported(target, "RGB pixels cannot be written as PGM"));
                }
                DecodedPixelFormat::Rgba8 => {
                    return Err(unsupported(target, "RGBA pixels cannot be written as PGM"));
                }
            };
            write_header(&mut counted, format!("P5\n{width} {height}\n255\n"))?;
            write_all(&mut counted, data)?;
        }
        "pbm" => {
            if decoded.format != DecodedPixelFormat::L8 {
                return Err(unsupported(target, "PBM requires bilevel L8 pixels"));
            }
            let packed = pack_bilevel(target, pixels, width, height)?;
            write_header(&mut counted, format!("P4\n{width} {height}\n"))?;
            write_all(&mut counted, &packed)?;
        }
        _ => match decoded.format {
            DecodedPixelFormat::L8 => {
                write_header(&mut counted, format!("P5\n{width} {height}\n255\n"))?;
                write_all(&mut counted, pixels)?;
            }
            DecodedPixelFormat::Rgb8 => {
                write_header(&mut counted, format!("P6\n{width} {height}\n255\n"))?;
                write_all(&mut counted, pixels)?;
            }
            DecodedPixelFormat::La8 | DecodedPixelFormat::Rgba8 => {
                let (depth, tuple_type) = if decoded.format == DecodedPixelFormat::La8 {
                    (2_u8, "GRAYSCALE_ALPHA")
                } else {
                    (4, "RGB_ALPHA")
                };
                write_header(
                    &mut counted,
                    format!(
                        "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH {depth}\nMAXVAL 255\nTUPLTYPE {tuple_type}\nENDHDR\n"
                    ),
                )?;
                write_all(&mut counted, pixels)?;
            }
        },
    }
    counted.flush().map_err(|source| io_error(target, source))?;
    Ok(counted.total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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

    fn decode_bytes(
        bytes: &[u8],
    ) -> Result<(u32, u32, DecodedPixelFormat, Vec<u8>), PipelineError> {
        let mut pixels = Vec::new();
        let (w, h, format, len) = decode_pnm_slice_into(Path::new("test.pnm"), bytes, &mut pixels)?;
        pixels.truncate(len);
        Ok((w, h, format, pixels))
    }

    fn encode_bytes(
        name: &str,
        pixels: &[u8],
        format: DecodedPixelFormat,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, PipelineError> {
        let mut encoded = Vec::new();
        let mut aux = Vec::new();
        let written = encode_pnm_to_writer(
            Path::new(name),
            pixels,
            meta(format, width, height),
            &mut aux,
            &mut encoded,
        )?;
        assert_eq!(written, u64::try_from(encoded.len()).unwrap_or(u64::MAX));
        Ok(encoded)
    }

    fn roundtrip(
        name: &str,
        pixels: &[u8],
        format: DecodedPixelFormat,
        width: u32,
        height: u32,
    ) -> Result<(), PipelineError> {
        let encoded = encode_bytes(name, pixels, format, width, height)?;
        let (w, h, decoded_format, decoded) = decode_bytes(&encoded)?;
        assert_eq!((w, h), (width, height), "{name} dimensions");
        assert_eq!(decoded_format, format, "{name} format");
        assert_eq!(
            decoded,
            pixels,
            "{name} pixels (header: {:?})",
            &encoded[..24.min(encoded.len())]
        );
        Ok(())
    }

    #[test]
    fn decodes_ascii_pbm_with_and_without_whitespace() -> Result<(), PipelineError> {
        let spaced = b"P1\n3 2\n1 0 1\n0 1 1\n";
        let (w, h, format, pixels) = decode_bytes(spaced)?;
        assert_eq!((w, h, format), (3, 2, DecodedPixelFormat::L8));
        assert_eq!(pixels, [0, 255, 0, 255, 0, 0]);

        let unspaced = b"P1\n3 2\n101011\n";
        let (_, _, _, pixels) = decode_bytes(unspaced)?;
        assert_eq!(pixels, [0, 255, 0, 255, 0, 0]);

        let commented = b"P1\n# top\n3 2\n1#mid\n0 1\n0 1 1\n";
        let (_, _, _, pixels) = decode_bytes(commented)?;
        assert_eq!(pixels, [0, 255, 0, 255, 0, 0]);
        Ok(())
    }

    #[test]
    fn decodes_ascii_pgm_and_ppm() -> Result<(), PipelineError> {
        let (w, h, format, pixels) = decode_bytes(b"P2\n2 2\n15\n0 15 7 3\n")?;
        assert_eq!((w, h, format), (2, 2, DecodedPixelFormat::L8));
        assert_eq!(pixels, [0, 255, 119, 51]);

        let (w, h, format, pixels) = decode_bytes(b"P3\n2 1\n255\n0 128 255 10 20 30\n")?;
        assert_eq!((w, h, format), (2, 1, DecodedPixelFormat::Rgb8));
        assert_eq!(pixels, [0, 128, 255, 10, 20, 30]);
        Ok(())
    }

    #[test]
    fn decodes_binary_pbm_odd_widths() -> Result<(), PipelineError> {
        // 5x3: MSB-first bits `10100 11001 01110`, padded to one byte per row.
        let mut bytes = b"P4\n5 3\n".to_vec();
        bytes.extend_from_slice(&[0b1010_0000, 0b1100_1000, 0b0111_0000]);
        let (w, h, format, pixels) = decode_bytes(&bytes)?;
        assert_eq!((w, h, format), (5, 3, DecodedPixelFormat::L8));
        let expected: Vec<u8> = [
            [0_u8, 255, 0, 255, 255],
            [0, 0, 255, 255, 0],
            [255, 0, 0, 0, 255],
        ]
        .concat();
        assert_eq!(pixels, expected);

        // 13x7: 2 bytes per row, last padding bits must be ignored.
        let mut bytes = b"P4\n13 7\n".to_vec();
        let pattern = [0xA5_u8, 0x3C];
        for _ in 0..7 {
            bytes.extend_from_slice(&pattern);
        }
        let (w, h, format, pixels) = decode_bytes(&bytes)?;
        assert_eq!((w, h, format), (13, 7, DecodedPixelFormat::L8));
        let mut expected = Vec::new();
        for _ in 0..7 {
            for bit in 0..13 {
                let value = (pattern[bit / 8] >> (7 - bit % 8)) & 1;
                expected.push(if value == 1 { 0 } else { 255 });
            }
        }
        assert_eq!(pixels, expected);
        Ok(())
    }

    #[test]
    fn decodes_binary_pgm_8_and_16_bit() -> Result<(), PipelineError> {
        let mut bytes = b"P5\n3 2\n255\n".to_vec();
        bytes.extend_from_slice(&[0, 128, 255, 10, 20, 30]);
        let (w, h, format, pixels) = decode_bytes(&bytes)?;
        assert_eq!((w, h, format), (3, 2, DecodedPixelFormat::L8));
        assert_eq!(pixels, [0, 128, 255, 10, 20, 30]);

        let mut bytes = b"P5\n2 1\n65535\n".to_vec();
        bytes.extend_from_slice(&[0x00, 0x00, 0xFF, 0xFF]);
        let (_, _, format, pixels) = decode_bytes(&bytes)?;
        assert_eq!(format, DecodedPixelFormat::L8);
        assert_eq!(pixels, [0, 255]);

        let mut bytes = b"P5\n2 1\n1023\n".to_vec();
        bytes.extend_from_slice(&[0x00, 0x00, 0x03, 0xFF]);
        let (_, _, _, pixels) = decode_bytes(&bytes)?;
        assert_eq!(pixels, [0, 255]);
        Ok(())
    }

    #[test]
    fn decodes_binary_ppm_8_and_16_bit() -> Result<(), PipelineError> {
        let mut bytes = b"P6\n2 1\n255\n".to_vec();
        bytes.extend_from_slice(&[0, 128, 255, 10, 20, 30]);
        let (w, h, format, pixels) = decode_bytes(&bytes)?;
        assert_eq!((w, h, format), (2, 1, DecodedPixelFormat::Rgb8));
        assert_eq!(pixels, [0, 128, 255, 10, 20, 30]);

        let mut bytes = b"P6\n1 2\n65535\n".to_vec();
        bytes.extend_from_slice(&[
            0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0xFF, 0xFF,
        ]);
        let (_, _, format, pixels) = decode_bytes(&bytes)?;
        assert_eq!(format, DecodedPixelFormat::Rgb8);
        assert_eq!(pixels, [255, 0, 0, 0, 128, 255]);
        Ok(())
    }

    #[test]
    fn decodes_pam_all_depths() -> Result<(), PipelineError> {
        for (depth, tuple_type, format, samples) in [
            (
                1_u8,
                "GRAYSCALE",
                DecodedPixelFormat::L8,
                [7_u8, 200].as_slice(),
            ),
            (
                2,
                "GRAYSCALE_ALPHA",
                DecodedPixelFormat::La8,
                [7, 255, 200, 0].as_slice(),
            ),
            (
                3,
                "RGB",
                DecodedPixelFormat::Rgb8,
                [1, 2, 3, 250, 251, 252].as_slice(),
            ),
            (
                4,
                "RGB_ALPHA",
                DecodedPixelFormat::Rgba8,
                [1, 2, 3, 255, 250, 251, 252, 0].as_slice(),
            ),
        ] {
            let mut bytes = format!(
                "P7\nWIDTH 2\nHEIGHT 1\nDEPTH {depth}\nMAXVAL 255\nTUPLTYPE {tuple_type}\nENDHDR\n"
            )
            .into_bytes();
            bytes.extend_from_slice(samples);
            let (w, h, decoded_format, pixels) = decode_bytes(&bytes)?;
            assert_eq!((w, h, decoded_format), (2, 1, format));
            assert_eq!(pixels, samples);
        }
        Ok(())
    }

    #[test]
    fn skips_comments_and_accepts_single_binary_separator() -> Result<(), PipelineError> {
        let commented =
            b"P2\n# comment before width\n2 # trailing\n1\n# before maxval\n255 # after\n7 8\n";
        let (_, _, _, pixels) = decode_bytes(commented)?;
        assert_eq!(pixels, [7, 8]);

        let hash_separator = b"P5\n2 1\n255#binary starts next\n\x01\x02";
        let (_, _, _, pixels) = decode_bytes(hash_separator)?;
        assert_eq!(pixels, [1, 2]);

        let missing_separator = b"P5\n2 1\n255\x01\x02";
        assert!(decode_bytes(missing_separator).is_err());
        Ok(())
    }

    #[test]
    fn roundtrips_pnm_ppm_pgm_pbm_and_pam_losslessly() -> Result<(), PipelineError> {
        let l8: Vec<u8> = (0..8_u32 * 5)
            .map(|i| u8::try_from((i * 31) % 256).unwrap_or(0))
            .collect();
        let bilevel: Vec<u8> = (0..13_u32 * 7)
            .map(|i| if i % 3 == 0 { 0 } else { 255 })
            .collect();
        let la8: Vec<u8> = l8.iter().flat_map(|&v| [v, 255]).collect();
        let rgb8: Vec<u8> = (0..8_u32 * 5 * 3)
            .map(|i| u8::try_from((i * 17) % 256).unwrap_or(0))
            .collect();
        let alpha_rgba: Vec<u8> = rgb8
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
            .flat_map(|(i, &[r, g, b])| [r, g, b, if i % 4 == 0 { 0 } else { 255 }])
            .collect();

        roundtrip("out.pnm", &l8, DecodedPixelFormat::L8, 8, 5)?;
        roundtrip("out.pnm", &rgb8, DecodedPixelFormat::Rgb8, 8, 5)?;
        roundtrip("out.pnm", &la8, DecodedPixelFormat::La8, 8, 5)?;
        roundtrip("out.pnm", &alpha_rgba, DecodedPixelFormat::Rgba8, 8, 5)?;
        roundtrip("out.ppm", &rgb8, DecodedPixelFormat::Rgb8, 8, 5)?;
        let ppm_l8 = encode_bytes("out.ppm", &l8, DecodedPixelFormat::L8, 8, 5)?;
        let (_, _, ppm_format, ppm_pixels) = decode_bytes(&ppm_l8)?;
        assert_eq!(ppm_format, DecodedPixelFormat::Rgb8);
        assert_eq!(
            ppm_pixels,
            l8.iter().flat_map(|&v| [v, v, v]).collect::<Vec<u8>>()
        );
        roundtrip("out.pgm", &l8, DecodedPixelFormat::L8, 8, 5)?;
        roundtrip("out.pbm", &bilevel, DecodedPixelFormat::L8, 13, 7)?;
        roundtrip("out.pam", &l8, DecodedPixelFormat::L8, 8, 5)?;
        roundtrip("out.pam", &la8, DecodedPixelFormat::La8, 8, 5)?;
        roundtrip("out.pam", &rgb8, DecodedPixelFormat::Rgb8, 8, 5)?;
        roundtrip("out.pam", &alpha_rgba, DecodedPixelFormat::Rgba8, 8, 5)?;
        Ok(())
    }

    #[test]
    fn converts_opaque_alpha_and_rejects_incompatible_targets() -> Result<(), PipelineError> {
        let opaque_rgba: Vec<u8> = (0..4_u32 * 4)
            .flat_map(|i| [u8::try_from(i).unwrap_or(0), 0, 255, 255])
            .collect();
        let encoded = encode_bytes("out.ppm", &opaque_rgba, DecodedPixelFormat::Rgba8, 4, 4)?;
        let (_, _, format, pixels) = decode_bytes(&encoded)?;
        assert_eq!(format, DecodedPixelFormat::Rgb8);
        assert_eq!(pixels.len(), 4 * 4 * 3);

        // Opaque LA8 -> PPM must expand each luma sample to 3-byte RGB for P6.
        let opaque_la: Vec<u8> = (0..4_u32 * 4)
            .flat_map(|i| [u8::try_from(i * 15).unwrap_or(0), 255])
            .collect();
        let encoded_la_ppm = encode_bytes("out.ppm", &opaque_la, DecodedPixelFormat::La8, 4, 4)?;
        let (lw, lh, lformat, lpixels) = decode_bytes(&encoded_la_ppm)?;
        assert_eq!((lw, lh, lformat), (4, 4, DecodedPixelFormat::Rgb8));
        assert_eq!(lpixels.len(), 4 * 4 * 3);
        let expected_la_rgb: Vec<u8> = (0..4_u32 * 4)
            .flat_map(|i| {
                let v = u8::try_from(i * 15).unwrap_or(0);
                [v, v, v]
            })
            .collect();
        assert_eq!(lpixels, expected_la_rgb);

        // Multi-word PAM TUPLTYPE must parse cleanly; trailing numeric garbage must fail.
        let multi_word_pam = b"P7\nWIDTH 1\nHEIGHT 1\nDEPTH 3\nMAXVAL 255\nTUPLTYPE RGB CUSTOM\nENDHDR\n\x0a\x14\x1e";
        let (pw, ph, pfmt, ppix) = decode_bytes(multi_word_pam)?;
        assert_eq!(
            (pw, ph, pfmt, ppix),
            (1, 1, DecodedPixelFormat::Rgb8, vec![10, 20, 30])
        );
        assert!(
            decode_bytes(b"P7\nWIDTH 1foo\nHEIGHT 1\nDEPTH 1\nMAXVAL 255\nENDHDR\n\x00").is_err()
        );

        let transparent_rgba: Vec<u8> = (0..4_u32 * 4)
            .flat_map(|i| [u8::try_from(i).unwrap_or(0), 0, 255, 0])
            .collect();
        let ppm = encode_bytes(
            "out.ppm",
            &transparent_rgba,
            DecodedPixelFormat::Rgba8,
            4,
            4,
        );
        assert!(matches!(
            ppm,
            Err(PipelineError::UnsupportedPixelFormat { .. })
        ));
        let pgm = encode_bytes(
            "out.pgm",
            &transparent_rgba,
            DecodedPixelFormat::Rgba8,
            4,
            4,
        );
        assert!(matches!(
            pgm,
            Err(PipelineError::UnsupportedPixelFormat { .. })
        ));
        let rgb_as_pgm = encode_bytes(
            "out.pgm",
            &[0_u8; 4 * 4 * 3],
            DecodedPixelFormat::Rgb8,
            4,
            4,
        );
        assert!(matches!(
            rgb_as_pgm,
            Err(PipelineError::UnsupportedPixelFormat { .. })
        ));
        let bad_pbm_format = encode_bytes(
            "out.pbm",
            &[0_u8; 4 * 4 * 3],
            DecodedPixelFormat::Rgb8,
            4,
            4,
        );
        assert!(matches!(
            bad_pbm_format,
            Err(PipelineError::UnsupportedPixelFormat { .. })
        ));
        let non_bilevel = encode_bytes("out.pbm", &[1_u8; 4 * 4], DecodedPixelFormat::L8, 4, 4);
        assert!(matches!(
            non_bilevel,
            Err(PipelineError::UnsupportedPixelFormat { .. })
        ));
        Ok(())
    }

    #[test]
    fn rejects_malformed_headers_and_truncated_rasters() {
        for bytes in [
            b"P9\n1 1\n".as_slice(),
            b"P5\n0 1\n255\n".as_slice(),
            b"P5\n1 1\n0\n\x00".as_slice(),
            b"P5\n1 1\n65536\n".as_slice(),
            b"P7\nWIDTH 1\nHEIGHT 1\nDEPTH 5\nMAXVAL 255\nENDHDR\n\x00".as_slice(),
            b"P7\nWIDTH 1\nHEIGHT 1\nDEPTH 1\nMAXVAL 0\nENDHDR\n\x00".as_slice(),
            b"P5\n2 1\n255\n\x01".as_slice(),
            b"P4\n5 1\n".as_slice(),
            b"P6\n1 1\n255\n\x01\x02".as_slice(),
        ] {
            assert!(
                decode_bytes(bytes).is_err(),
                "expected failure for {:?}",
                String::from_utf8_lossy(bytes)
            );
        }

        let target = PathBuf::from("out.pam");
        let mut aux = Vec::new();
        let mut out = Vec::new();
        let bad = encode_pnm_to_writer(
            &target,
            &[0_u8; 3],
            meta(DecodedPixelFormat::Rgb8, 4, 4),
            &mut aux,
            &mut out,
        );
        assert!(matches!(bad, Err(PipelineError::EncodeFailed { .. })));
    }
}
