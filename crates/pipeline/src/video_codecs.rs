//! Native video codec backends: pure-Rust decoding of all four supported
//! video codecs into a uniform 8-bit planar 4:2:0 frame, and CPU encoding of
//! H.264, AV1 and VP9 into canonical container packets.
//!
//! Every illegal state is rejected explicitly; there is no silent fallback
//! between codecs or bit depths.

use std::borrow::Cow;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

use fastconv_domain::{VideoCodec, VideoEncodeMode, VideoQuality, VideoRateControl};
use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder, SamplingFactor};
use rav1d_safe::src::managed::{Decoder as Av1Decoder, PixelLayout, Planes as Av1Planes};
use rav1e::color::ChromaSampling;
use rav1e::prelude::{Config, EncoderConfig, FrameType, Rational, SpeedSettings};
use rusty_h264::{
    Decoder as H264Decoder, Encoder as H264Encoder, EncoderConfig as H264EncoderConfig,
    Preset as H264Preset, YuvFrame as H264YuvFrame, YuvPlanes as H264YuvPlanes,
};
use rusty_h265::Decoder as HevcDecoder;
use rusty_vp9::{
    DecodedFrame as Vp9DecodedFrame, Error as Vp9Error, Vp9Decoder, Vp9Encoder, Vp9EncoderConfig,
};
use yuv::{YuvPlanarImage, YuvRange as SimdYuvRange, YuvStandardMatrix};
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

use crate::PipelineError;
use crate::video_bitstream::{
    annex_b_to_avcc, annex_b_to_hvcc, avcc_to_annex_b, extract_av1c_from_obu,
    extract_avcc_from_annex_b, extract_hvcc_from_annex_b, hvcc_to_annex_b,
    is_h264_keyframe_annex_b,
};
use crate::video_container::{ContainerPacket, ContainerTrack, US_TIMESCALE};

/// Uniform decoded video frame: 8-bit planar YUV 4:2:0 with exact timestamps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Yuv420Frame {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) y: Vec<u8>,
    pub(crate) u: Vec<u8>,
    pub(crate) v: Vec<u8>,
    pub(crate) pts_us: i64,
    pub(crate) duration_us: u64,
    /// Signalled colour of the planes; unsignalled means BT.601 limited.
    pub(crate) color: YuvColor,
}

/// YUV→RGB matrix signalled by the source bitstream. BT.601 is the explicit
/// default only for sources that signal no usable matrix. `Bt2020` covers
/// non-constant luminance (H.273 coefficient 9); signalled matrices without an
/// RGB converter are carried as `Unsupported` and rejected only where RGB is
/// actually produced, so NV12 video-to-video paths stay unaffected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum YuvMatrix {
    #[default]
    Bt601,
    Bt709,
    Bt2020,
    Unsupported(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum YuvRange {
    #[default]
    Limited,
    Full,
}

/// Signalled source colour description carried with each decoded frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct YuvColor {
    pub(crate) matrix: YuvMatrix,
    pub(crate) range: YuvRange,
    /// False only when the source bitstream does not signal colour at all.
    pub(crate) signalled: bool,
}

impl YuvColor {
    pub(crate) const fn specified(matrix: YuvMatrix, range: YuvRange) -> Self {
        Self {
            matrix,
            range,
            signalled: true,
        }
    }
}

/// Geometry, timing and colour metadata for one decoded 8-bit 4:2:0 frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecodedFrameInfo {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pts_us: i64,
    pub(crate) duration_us: u64,
    pub(crate) color: YuvColor,
}

/// One encoded video track with canonical container packets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EncodedVideoStream {
    pub(crate) track: ContainerTrack,
    pub(crate) packets: Vec<ContainerPacket>,
}

fn decode_error(path: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::DecodeFailed {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

fn encode_error(path: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

/// Decode order: increasing DTS, then PTS as a tie-breaker.
fn decode_order(packets: &[ContainerPacket]) -> Vec<&ContainerPacket> {
    let mut ordered: Vec<&ContainerPacket> = packets.iter().collect();
    ordered.sort_by_key(|packet| (packet.dts, packet.pts));
    ordered
}

fn median_duration(packets: &[ContainerPacket]) -> u64 {
    let mut durations: Vec<u64> = packets
        .iter()
        .filter(|packet| packet.duration > 0)
        .map(|packet| packet.duration)
        .collect();
    durations.sort_unstable();
    if durations.is_empty() {
        return 1_000_000 / 30;
    }
    let mid = durations.len() / 2;
    if durations.len().is_multiple_of(2) {
        let lo = durations[mid.saturating_sub(1)];
        let hi = durations[mid];
        lo.saturating_add(hi) / 2
    } else {
        durations[mid]
    }
}

fn require_even(path: &Path, width: u32, height: u32) -> Result<(), PipelineError> {
    if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(encode_error(
            path,
            format!("4:2:0 video requires even, non-zero dimensions; got {width}x{height}"),
        ));
    }
    Ok(())
}

/// Copies a strided plane into a tightly packed 8-bit plane, narrowing
/// higher bit depths with a plain MSB shift as required by 4:2:0 8-bit output.
// Every plane's geometry is passed explicitly so extraction needs no wrapper type.
#[allow(clippy::too_many_arguments)]
fn copy_plane_u16(
    src: &[u16],
    stride: usize,
    crop_x: usize,
    crop_y: usize,
    width: usize,
    height: usize,
    bit_depth: u8,
    out: &mut [u8],
    path: &Path,
) -> Result<(), PipelineError> {
    let shift = bit_depth.saturating_sub(8);
    for row in 0..height {
        let src_row = crop_y
            .saturating_add(row)
            .checked_mul(stride)
            .and_then(|start| start.checked_add(crop_x))
            .ok_or_else(|| decode_error(path, "plane row offset overflow"))?;
        let src = src
            .get(src_row..src_row.saturating_add(width))
            .ok_or_else(|| decode_error(path, "decoded plane row out of bounds"))?;
        let dst = out
            .get_mut(row.saturating_mul(width)..row.saturating_mul(width).saturating_add(width))
            .ok_or_else(|| decode_error(path, "output plane row out of bounds"))?;
        for (slot, sample) in dst.iter_mut().zip(src.iter().copied()) {
            let narrowed = sample >> shift;
            *slot = u8::try_from(narrowed).unwrap_or(u8::MAX);
        }
    }
    Ok(())
}

/// Copies a strided byte plane into a tightly packed 8-bit plane.
// Every plane's geometry is passed explicitly so extraction needs no wrapper type.
#[allow(clippy::too_many_arguments)]
fn copy_plane_u8(
    src: &[u8],
    stride: usize,
    crop_x: usize,
    crop_y: usize,
    width: usize,
    height: usize,
    out: &mut [u8],
    path: &Path,
) -> Result<(), PipelineError> {
    for row in 0..height {
        let src_row = crop_y
            .saturating_add(row)
            .checked_mul(stride)
            .and_then(|start| start.checked_add(crop_x))
            .ok_or_else(|| decode_error(path, "plane row offset overflow"))?;
        let src = src
            .get(src_row..src_row.saturating_add(width))
            .ok_or_else(|| decode_error(path, "decoded plane row out of bounds"))?;
        let dst = out
            .get_mut(row.saturating_mul(width)..row.saturating_mul(width).saturating_add(width))
            .ok_or_else(|| decode_error(path, "output plane row out of bounds"))?;
        dst.copy_from_slice(src);
    }
    Ok(())
}

// MJPEG is a sequence of standalone base-line JPEG images. These caps bound
// work done before any pixel buffer is allocated, so a hostile packet cannot
// turn a small AVI chunk into an unbounded allocation.
const MJPEG_PACKET_CAP: usize = 16 * 1024 * 1024;
const MJPEG_MAX_DIMENSION: u32 = 8192;
const MJPEG_MAX_PIXELS: u64 = 16_777_216;

// ISO/IEC 10918-1 Annex K default Huffman tables. AVI MJPEG often omits DHT
// segments entirely, relying on the decoder to supply these four tables.
const DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_LUMA_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const AC_LUMA_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D];
const AC_LUMA_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];
const DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_CHROMA_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const AC_CHROMA_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const AC_CHROMA_VALUES: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];

fn dht_segment(class: u8, id: u8, bits: &[u8; 16], values: &[u8]) -> Vec<u8> {
    let body_len = u16::try_from(19_usize.saturating_add(values.len())).unwrap_or(u16::MAX);
    let mut segment = Vec::with_capacity(21_usize.saturating_add(values.len()));
    segment.extend_from_slice(&[0xFF, 0xC4]);
    segment.extend_from_slice(&body_len.to_be_bytes());
    segment.push((class << 4) | (id & 0x0F));
    segment.extend_from_slice(bits);
    segment.extend_from_slice(values);
    segment
}

/// Standard JPEG Huffman tables are immutable and identical for every MJPEG
/// packet that lacks its own DHT; build them once instead of per packet.
static JPEG_HUFFMAN_TABLES: OnceLock<Vec<u8>> = OnceLock::new();

fn standard_jpeg_huffman_tables() -> &'static [u8] {
    JPEG_HUFFMAN_TABLES.get_or_init(|| {
        let mut tables = Vec::with_capacity(1024);
        tables.extend_from_slice(&dht_segment(0, 0, &DC_LUMA_BITS, &DC_LUMA_VALUES));
        tables.extend_from_slice(&dht_segment(1, 0, &AC_LUMA_BITS, &AC_LUMA_VALUES));
        tables.extend_from_slice(&dht_segment(0, 1, &DC_CHROMA_BITS, &DC_CHROMA_VALUES));
        tables.extend_from_slice(&dht_segment(1, 1, &AC_CHROMA_BITS, &AC_CHROMA_VALUES));
        tables
    })
}

/// Returns the packet with standard Huffman tables injected immediately before
/// SOS when it carries no DHT segment at all. Marker lengths are validated so a
/// truncated or malformed header is rejected rather than decoded partially.
fn mjpeg_with_huffman_tables<'a>(
    path: &Path,
    packet: &'a [u8],
) -> Result<Cow<'a, [u8]>, PipelineError> {
    if packet.len() < 4 || packet[..2] != [0xFF, 0xD8] {
        return Err(decode_error(
            path,
            "MJPEG packet does not start with a JPEG SOI marker",
        ));
    }
    let mut idx = 2_usize;
    let mut has_dht = false;
    let mut sos_start = None;
    while idx < packet.len() {
        if packet[idx] != 0xFF {
            return Err(decode_error(
                path,
                "malformed MJPEG marker segment: missing 0xFF marker prefix",
            ));
        }
        let marker_prefix = idx;
        while packet.get(idx) == Some(&0xFF) {
            idx = idx.saturating_add(1);
        }
        let marker = *packet
            .get(idx)
            .ok_or_else(|| decode_error(path, "truncated MJPEG marker"))?;
        idx = idx.saturating_add(1);
        if marker == 0xDA {
            // Inject before the whole marker, including any 0xFF fill bytes.
            sos_start = Some(marker_prefix);
            break;
        }
        if marker == 0xD9 {
            return Err(decode_error(
                path,
                "MJPEG packet has no start-of-scan marker",
            ));
        }
        if marker == 0x00 {
            return Err(decode_error(
                path,
                "unexpected 0xFF 0x00 byte-stuffing before JPEG start-of-scan",
            ));
        }
        // Standalone markers carry no length.
        if marker == 0x01 || (0xD0..=0xD7).contains(&marker) || marker == 0xD8 {
            continue;
        }
        let len_bytes = packet
            .get(idx..idx.saturating_add(2))
            .ok_or_else(|| decode_error(path, "truncated MJPEG marker length"))?;
        let len = usize::from(u16::from_be_bytes([len_bytes[0], len_bytes[1]]));
        if len < 2 {
            return Err(decode_error(
                path,
                "MJPEG marker segment declares an invalid length",
            ));
        }
        let end = idx.checked_add(len).ok_or_else(|| {
            decode_error(path, "MJPEG marker segment length overflows the packet")
        })?;
        if end > packet.len() {
            return Err(decode_error(
                path,
                "MJPEG marker segment extends past the end of the packet",
            ));
        }
        if marker == 0xC4 {
            has_dht = true;
        }
        idx = end;
    }
    let sos_start =
        sos_start.ok_or_else(|| decode_error(path, "MJPEG packet has no start-of-scan marker"))?;
    if has_dht {
        return Ok(Cow::Borrowed(packet));
    }
    let tables = standard_jpeg_huffman_tables();
    let mut complete = Vec::with_capacity(packet.len().saturating_add(tables.len()));
    complete.extend_from_slice(&packet[..sos_start]);
    complete.extend_from_slice(tables);
    complete.extend_from_slice(&packet[sos_start..]);
    Ok(Cow::Owned(complete))
}

/// Decodes one MJPEG packet into a reusable tightly packed RGB8 scratch buffer.
/// Returns the JPEG's natural (possibly odd) dimensions.
fn mjpeg_decode_rgb(
    path: &Path,
    packet: &[u8],
    rgb: &mut Vec<u8>,
) -> Result<(u16, u16), PipelineError> {
    if packet.len() > MJPEG_PACKET_CAP {
        return Err(decode_error(
            path,
            format!(
                "MJPEG packet of {} bytes exceeds the {MJPEG_PACKET_CAP}-byte cap",
                packet.len()
            ),
        ));
    }
    // `zune-jpeg` treats premature EOF as recoverable and may still return
    // pixels, so require a complete image with an EOI marker before decoding.
    let mut end = packet.len();
    while end > 0 && packet[end - 1] == 0 {
        end -= 1;
    }
    if end < 2 || packet[end - 2..end] != [0xFF, 0xD9] {
        return Err(decode_error(
            path,
            "MJPEG packet is missing its JPEG EOI marker",
        ));
    }
    let prepared = mjpeg_with_huffman_tables(path, &packet[..end])?;
    let max_dim = usize::try_from(MJPEG_MAX_DIMENSION).unwrap_or(usize::MAX);
    let options = DecoderOptions::default()
        .jpeg_set_out_colorspace(ColorSpace::RGB)
        .set_max_width(max_dim)
        .set_max_height(max_dim);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(prepared.as_ref()), options);
    decoder
        .decode_headers()
        .map_err(|err| decode_error(path, format!("MJPEG header parse failed: {err}")))?;
    let info = decoder
        .info()
        .ok_or_else(|| decode_error(path, "MJPEG packet has no JPEG SOF metadata"))?;
    let width = u32::from(info.width);
    let height = u32::from(info.height);
    if width == 0 || height == 0 || width > MJPEG_MAX_DIMENSION || height > MJPEG_MAX_DIMENSION {
        return Err(decode_error(
            path,
            format!("MJPEG dimensions {width}x{height} are out of range"),
        ));
    }
    let pixels = u64::from(width).saturating_mul(u64::from(height));
    if pixels > MJPEG_MAX_PIXELS {
        return Err(decode_error(
            path,
            format!("MJPEG frame of {pixels} pixels exceeds the {MJPEG_MAX_PIXELS}-pixel cap"),
        ));
    }
    let expected_rgb = usize::from(info.width)
        .checked_mul(usize::from(info.height))
        .and_then(|count| count.checked_mul(3))
        .ok_or_else(|| decode_error(path, "MJPEG decoded buffer size overflows usize"))?;
    let required = decoder
        .output_buffer_size()
        .ok_or_else(|| decode_error(path, "MJPEG decoded buffer size overflows usize"))?;
    if required != expected_rgb {
        return Err(decode_error(
            path,
            format!(
                "MJPEG decoded RGB buffer size {required} does not match expected {expected_rgb}"
            ),
        ));
    }
    rgb.resize(required, 0);
    decoder
        .decode_into(rgb.as_mut_slice())
        .map_err(|err| decode_error(path, format!("MJPEG decode failed: {err}")))?;
    Ok((info.width, info.height))
}

fn h264_frame(frame: H264YuvFrame, pts_us: i64, duration_us: u64) -> Yuv420Frame {
    Yuv420Frame {
        width: u32::try_from(frame.width).unwrap_or(u32::MAX),
        height: u32::try_from(frame.height).unwrap_or(u32::MAX),
        y: frame.y,
        u: frame.u,
        v: frame.v,
        pts_us,
        duration_us,
        color: YuvColor::default(),
    }
}

fn hevc_frame(
    frame: &rusty_h265::Frame,
    pts_us: i64,
    duration_us: u64,
    path: &Path,
) -> Result<Yuv420Frame, PipelineError> {
    let width = frame.width;
    let height = frame.height;
    if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(decode_error(
            path,
            format!("HEVC decoded odd or zero dimensions: {width}x{height}"),
        ));
    }
    let (cw, ch) = (width / 2, height / 2);
    let picture = &frame.picture;
    let (crop_x, crop_y) = (picture.crop.0, picture.crop.1);
    if picture.chroma_format_idc != 1 {
        return Err(decode_error(
            path,
            format!(
                "HEVC chroma format {} is not 4:2:0",
                picture.chroma_format_idc
            ),
        ));
    }
    let mut y = vec![0_u8; width.saturating_mul(height)];
    let mut u = vec![0_u8; cw.saturating_mul(ch)];
    let mut v = vec![0_u8; cw.saturating_mul(ch)];
    copy_plane_u16(
        &picture.planes[0].data,
        picture.planes[0].stride,
        crop_x,
        crop_y,
        width,
        height,
        picture.bit_depth_luma,
        &mut y,
        path,
    )?;
    copy_plane_u16(
        &picture.planes[1].data,
        picture.planes[1].stride,
        crop_x / 2,
        crop_y / 2,
        cw,
        ch,
        picture.bit_depth_chroma,
        &mut u,
        path,
    )?;
    copy_plane_u16(
        &picture.planes[2].data,
        picture.planes[2].stride,
        crop_x / 2,
        crop_y / 2,
        cw,
        ch,
        picture.bit_depth_chroma,
        &mut v,
        path,
    )?;
    Ok(Yuv420Frame {
        width: u32::try_from(width).unwrap_or(u32::MAX),
        height: u32::try_from(height).unwrap_or(u32::MAX),
        y,
        u,
        v,
        pts_us,
        duration_us,
        color: YuvColor::default(),
    })
}

/// Maps a picture's presentation timestamp to the source packet that produced
/// it. A reordered picture can surface while a different access unit is being
/// pushed, so the current packet is not necessarily its source.
fn hevc_pts_to_packet(packets: &[ContainerPacket]) -> HashMap<i64, usize> {
    let mut mapping = HashMap::with_capacity(packets.len());
    for (index, packet) in packets.iter().enumerate() {
        mapping.entry(packet.pts).or_insert(index);
    }
    mapping
}

/// Uses the decoded picture's own timestamp when the decoder supplies one,
/// falling back to the access unit being pushed (or the next expected
/// presentation time while flushing) only when the picture has none.
fn hevc_picture_pts(frame: &rusty_h265::Frame, fallback: i64) -> i64 {
    frame.pts.unwrap_or(fallback)
}

fn hevc_picture_duration(
    pts: i64,
    packets: &[ContainerPacket],
    mapping: &HashMap<i64, usize>,
    fallback: u64,
) -> u64 {
    mapping
        .get(&pts)
        .and_then(|&index| packets.get(index))
        .map_or(fallback, |packet| packet.duration)
}

fn av1_frame(
    frame: &rav1d_safe::src::managed::Frame,
    pts_us: i64,
    duration_us: u64,
    path: &Path,
) -> Result<Yuv420Frame, PipelineError> {
    let width = usize::try_from(frame.width()).map_err(|_| decode_error(path, "AV1 width"))?;
    let height = usize::try_from(frame.height()).map_err(|_| decode_error(path, "AV1 height"))?;
    if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(decode_error(
            path,
            format!("AV1 decoded odd or zero dimensions: {width}x{height}"),
        ));
    }
    let layout = frame.pixel_layout();
    if layout != PixelLayout::I420 {
        return Err(decode_error(
            path,
            format!("AV1 chroma layout {layout:?} is not 4:2:0"),
        ));
    }
    let (cw, ch) = (width / 2, height / 2);
    let mut y = vec![0_u8; width.saturating_mul(height)];
    let mut u = vec![0_u8; cw.saturating_mul(ch)];
    let mut v = vec![0_u8; cw.saturating_mul(ch)];
    match frame.planes() {
        Av1Planes::Depth8(planes) => {
            let y_plane = planes.y();
            let u_plane = planes
                .u()
                .ok_or_else(|| decode_error(path, "AV1 frame has no U plane"))?;
            let v_plane = planes
                .v()
                .ok_or_else(|| decode_error(path, "AV1 frame has no V plane"))?;
            for row in 0..height.min(y_plane.height()) {
                let src = y_plane
                    .row(row)
                    .get(..width.min(y_plane.width()))
                    .ok_or_else(|| decode_error(path, "AV1 Y row is too short"))?;
                let dest = y
                    .get_mut(
                        row.saturating_mul(width)
                            ..row.saturating_mul(width).saturating_add(src.len()),
                    )
                    .ok_or_else(|| decode_error(path, "AV1 Y output overflow"))?;
                dest.copy_from_slice(src);
            }
            for (plane, out, pw, ph) in [(&u_plane, &mut u, cw, ch), (&v_plane, &mut v, cw, ch)] {
                for row in 0..ph.min(plane.height()) {
                    let src = plane
                        .row(row)
                        .get(..pw.min(plane.width()))
                        .ok_or_else(|| decode_error(path, "AV1 chroma row is too short"))?;
                    let dest = out
                        .get_mut(
                            row.saturating_mul(pw)
                                ..row.saturating_mul(pw).saturating_add(src.len()),
                        )
                        .ok_or_else(|| decode_error(path, "AV1 chroma output overflow"))?;
                    dest.copy_from_slice(src);
                }
            }
        }
        Av1Planes::Depth16(planes) => {
            let shift = frame.bit_depth().saturating_sub(8);
            let y_plane = planes.y();
            let u_plane = planes
                .u()
                .ok_or_else(|| decode_error(path, "AV1 frame has no U plane"))?;
            let v_plane = planes
                .v()
                .ok_or_else(|| decode_error(path, "AV1 frame has no V plane"))?;
            for (rows, out, pw, ph) in [
                (y_plane.rows().collect::<Vec<_>>(), &mut y, width, height),
                (u_plane.rows().collect::<Vec<_>>(), &mut u, cw, ch),
                (v_plane.rows().collect::<Vec<_>>(), &mut v, cw, ch),
            ] {
                for (row, src) in rows.into_iter().take(ph).enumerate() {
                    for (x, &sample) in src.iter().take(pw).enumerate() {
                        let value = sample >> shift;
                        if let Some(slot) = out.get_mut(row.saturating_mul(pw).saturating_add(x)) {
                            *slot = u8::try_from(value).unwrap_or(u8::MAX);
                        }
                    }
                }
            }
        }
    }
    Ok(Yuv420Frame {
        width: u32::try_from(width).unwrap_or(u32::MAX),
        height: u32::try_from(height).unwrap_or(u32::MAX),
        y,
        u,
        v,
        pts_us,
        duration_us,
        color: YuvColor::default(),
    })
}

fn vp9_plane(
    src: &[u8],
    stride: usize,
    width: usize,
    height: usize,
    bit_depth: u32,
    path: &Path,
) -> Result<Vec<u8>, PipelineError> {
    let mut out = vec![0_u8; width.saturating_mul(height)];
    if bit_depth <= 8 {
        copy_plane_u8(src, stride, 0, 0, width, height, &mut out, path)?;
    } else {
        let shift = bit_depth.saturating_sub(8);
        for row in 0..height {
            for x in 0..width {
                let idx = row
                    .saturating_mul(stride)
                    .saturating_add(x.saturating_mul(2));
                let bytes = src
                    .get(idx..idx.saturating_add(2))
                    .ok_or_else(|| decode_error(path, "VP9 10-bit plane row is too short"))?;
                let value = u16::from_le_bytes([bytes[0], bytes[1]]) >> shift;
                if let Some(slot) = out.get_mut(row.saturating_mul(width).saturating_add(x)) {
                    *slot = u8::try_from(value).unwrap_or(u8::MAX);
                }
            }
        }
    }
    Ok(out)
}

fn vp9_frame(
    frame: &Vp9DecodedFrame,
    duration_us: u64,
    path: &Path,
) -> Result<Yuv420Frame, PipelineError> {
    let width = usize::try_from(frame.width).map_err(|_| decode_error(path, "VP9 width"))?;
    let height = usize::try_from(frame.height).map_err(|_| decode_error(path, "VP9 height"))?;
    if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(decode_error(
            path,
            format!("VP9 decoded odd or zero dimensions: {width}x{height}"),
        ));
    }
    let (cw, ch) = (width / 2, height / 2);
    let planes = &frame.planes;
    let strides = &frame.strides;
    let plane = |index: usize| -> Result<&[u8], PipelineError> {
        planes
            .get(index)
            .map(Vec::as_slice)
            .ok_or_else(|| decode_error(path, "VP9 frame is missing a plane"))
    };
    let stride = |index: usize| -> Result<usize, PipelineError> {
        strides
            .get(index)
            .copied()
            .ok_or_else(|| decode_error(path, "VP9 frame is missing a stride"))
    };
    Ok(Yuv420Frame {
        width: u32::try_from(width).unwrap_or(u32::MAX),
        height: u32::try_from(height).unwrap_or(u32::MAX),
        y: vp9_plane(plane(0)?, stride(0)?, width, height, frame.bit_depth, path)?,
        u: vp9_plane(plane(1)?, stride(1)?, cw, ch, frame.bit_depth, path)?,
        v: vp9_plane(plane(2)?, stride(2)?, cw, ch, frame.bit_depth, path)?,
        pts_us: frame.pts.unwrap_or(0),
        duration_us,
        color: YuvColor::default(),
    })
}

/// Minimum presentation time at or after each decode-order index.
///
/// Used by the H.264 reorder buffer so a buffered frame is emitted only once no
/// later decode-order packet (including open-GOP leading pictures after a
/// sync-marked frame) can precede it in presentation order.
fn suffix_min_pts(ordered: &[&ContainerPacket]) -> Vec<i64> {
    let mut suffix = vec![0_i64; ordered.len()];
    let mut running_min = i64::MAX;
    for (index, packet) in ordered.iter().enumerate().rev() {
        running_min = running_min.min(packet.pts);
        suffix[index] = running_min;
    }
    suffix
}

/// Drains buffered H.264 frames whose presentation time cannot be preceded by
/// any remaining decode-order packet. At the 16-frame DPB cap an unproven
/// order is rejected explicitly instead of emitting a frame a later picture
/// could still precede.
fn drain_ready_h264_reorder_vec(
    path: &Path,
    reorder: &mut Vec<Yuv420Frame>,
    future_min_pts: Option<i64>,
    frames: &mut Vec<Yuv420Frame>,
) -> Result<(), PipelineError> {
    while !reorder.is_empty() {
        let earliest = reorder
            .iter()
            .enumerate()
            .min_by_key(|(_, frame)| frame.pts_us)
            .map_or(0, |(index, _)| index);
        let safe_to_emit =
            future_min_pts.is_none_or(|future_min| reorder[earliest].pts_us <= future_min);
        if safe_to_emit {
            frames.push(reorder.remove(earliest));
        } else if reorder.len() > 16 {
            return Err(decode_error(
                path,
                "H.264 reorder buffer exceeded 16 frames before presentation order was provable",
            ));
        } else {
            break;
        }
    }
    Ok(())
}

/// Emits remaining buffered H.264 frames in presentation order at EOF.
fn flush_h264_reorder_vec(reorder: &mut Vec<Yuv420Frame>, frames: &mut Vec<Yuv420Frame>) {
    reorder.sort_by_key(|frame| frame.pts_us);
    frames.append(reorder);
}

fn has_b_frame_reordering(ordered: &[&ContainerPacket]) -> bool {
    ordered.windows(2).any(|pair| pair[1].pts < pair[0].pts)
}

/// Decodes one canonical video track into display-ordered 8-bit YUV 4:2:0
/// frames. Every failure is explicit; unsupported codecs are rejected.
// One branch per codec keeps decode-order and timestamp handling explicit.
#[allow(clippy::too_many_lines)]
pub(crate) fn decode_video_track(
    path: &Path,
    track: &ContainerTrack,
    packets: &[ContainerPacket],
) -> Result<Vec<Yuv420Frame>, PipelineError> {
    let ordered = decode_order(packets);
    let track_color = source_color(track, &ordered);
    let decoded = match track.codec {
        VideoCodec::H264 => {
            let mut decoder = H264Decoder::new();
            let mut frames = Vec::with_capacity(ordered.len());
            let needs_reorder = has_b_frame_reordering(&ordered);
            let future_mins = if needs_reorder {
                suffix_min_pts(&ordered)
            } else {
                Vec::new()
            };
            let mut reorder: Vec<Yuv420Frame> =
                Vec::with_capacity(if needs_reorder { 16 } else { 0 });
            for (index, packet) in ordered.iter().enumerate() {
                let annex_b = avcc_to_annex_b(&track.config, &packet.data, true)
                    .map_err(|reason| decode_error(path, reason))?;
                let frame = decoder
                    .decode(&annex_b)
                    .map_err(|err| decode_error(path, format!("H.264 decode failed: {err}")))?;
                if let Some(frame) = frame {
                    let yuv = h264_frame(frame, packet.pts, packet.duration);
                    if needs_reorder {
                        reorder.push(yuv);
                        let future_min = future_mins.get(index + 1).copied();
                        drain_ready_h264_reorder_vec(path, &mut reorder, future_min, &mut frames)?;
                    } else {
                        frames.push(yuv);
                    }
                }
            }
            if needs_reorder {
                flush_h264_reorder_vec(&mut reorder, &mut frames);
            }
            Ok(frames)
        }
        VideoCodec::Hevc => {
            let mut decoder = HevcDecoder::new();
            let mut frames = Vec::with_capacity(ordered.len());
            let mapping = hevc_pts_to_packet(packets);
            let fallback_dur = median_duration(packets);
            for packet in ordered {
                let annex_b = hvcc_to_annex_b(&track.config, &packet.data, true)
                    .map_err(|reason| decode_error(path, reason))?;
                decoder
                    .push_annexb(&annex_b, Some(packet.pts))
                    .map_err(|err| decode_error(path, format!("HEVC decode failed: {err}")))?;
                loop {
                    match decoder.next_frame() {
                        Ok(frame) => {
                            let pts = hevc_picture_pts(&frame, packet.pts);
                            let duration =
                                hevc_picture_duration(pts, packets, &mapping, fallback_dur);
                            frames.push(hevc_frame(&frame, pts, duration, path)?);
                        }
                        Err(rusty_h265::Error::Again | rusty_h265::Error::Eof) => break,
                        Err(err) => {
                            return Err(decode_error(path, format!("HEVC drain failed: {err}")));
                        }
                    }
                }
            }
            decoder.flush();
            loop {
                match decoder.next_frame() {
                    Ok(frame) => {
                        let fallback_pts = frames.last().map_or(0, |last| {
                            last.pts_us
                                .saturating_add(i64::try_from(last.duration_us).unwrap_or(1))
                        });
                        let pts = hevc_picture_pts(&frame, fallback_pts);
                        let duration = hevc_picture_duration(pts, packets, &mapping, fallback_dur);
                        frames.push(hevc_frame(&frame, pts, duration, path)?);
                    }
                    Err(rusty_h265::Error::Again | rusty_h265::Error::Eof) => break,
                    Err(err) => {
                        return Err(decode_error(path, format!("HEVC flush failed: {err}")));
                    }
                }
            }
            Ok(frames)
        }
        VideoCodec::Av1 => {
            let mut decoder =
                Av1Decoder::new().map_err(|err| decode_error(path, format!("{err}")))?;
            let mut frames = Vec::with_capacity(ordered.len());
            for packet in ordered {
                if let Some(frame) = decoder
                    .decode(&packet.data)
                    .map_err(|err| decode_error(path, format!("AV1 decode failed: {err}")))?
                {
                    frames.push(av1_frame(&frame, packet.pts, packet.duration, path)?);
                }
                while let Some(frame) = decoder
                    .get_frame()
                    .map_err(|err| decode_error(path, format!("AV1 drain failed: {err}")))?
                {
                    frames.push(av1_frame(&frame, packet.pts, packet.duration, path)?);
                }
            }
            for frame in decoder
                .flush()
                .map_err(|err| decode_error(path, format!("AV1 flush failed: {err}")))?
            {
                let pts = u64::try_from(frames.len())
                    .unwrap_or(u64::MAX)
                    .saturating_mul(median_duration(packets));
                frames.push(av1_frame(
                    &frame,
                    i64::try_from(pts).unwrap_or(i64::MAX),
                    median_duration(packets),
                    path,
                )?);
            }
            Ok(frames)
        }
        VideoCodec::Vp9 => {
            let mut decoder = Vp9Decoder::new();
            let mut frames = Vec::with_capacity(ordered.len());
            for packet in ordered {
                decoder
                    .push(&packet.data, Some(packet.pts))
                    .map_err(|err| decode_error(path, format!("VP9 push failed: {err}")))?;
                loop {
                    match decoder.next_frame() {
                        Ok(frame) => frames.push(vp9_frame(&frame, packet.duration, path)?),
                        Err(Vp9Error::Again | Vp9Error::Eof) => break,
                        Err(err) => {
                            return Err(decode_error(path, format!("VP9 drain failed: {err}")));
                        }
                    }
                }
            }
            decoder.flush();
            loop {
                match decoder.next_frame() {
                    Ok(frame) => frames.push(vp9_frame(&frame, median_duration(packets), path)?),
                    Err(Vp9Error::Again | Vp9Error::Eof) => break,
                    Err(err) => {
                        return Err(decode_error(path, format!("VP9 flush failed: {err}")));
                    }
                }
            }
            Ok(frames)
        }
        VideoCodec::Mjpeg => {
            let mut frames = Vec::with_capacity(ordered.len());
            let mut rgb = Vec::new();
            let mut retained = 0_u64;
            for packet in ordered {
                let (width, height) = mjpeg_decode_rgb(path, &packet.data, &mut rgb)?;
                let frame = crate::video_anim::rgb_to_yuv420(
                    crate::video_anim::AnimFrameInfo {
                        width: u32::from(width),
                        height: u32::from(height),
                        pts_us: packet.pts,
                        duration_us: packet.duration,
                        channels: 3,
                    },
                    rgb.as_slice(),
                )?;
                // JPEG frames carry their own dimensions, which may be larger
                // than the container geometry the caller used for its estimate.
                // Count each frame as it is retained, before the next decode.
                retained = retained
                    .saturating_add(u64::try_from(frame.y.len()).unwrap_or(u64::MAX))
                    .saturating_add(u64::try_from(frame.u.len()).unwrap_or(u64::MAX))
                    .saturating_add(u64::try_from(frame.v.len()).unwrap_or(u64::MAX));
                crate::video_transcode::enforce_decoded_budget(path, retained)?;
                frames.push(frame);
            }
            Ok(frames)
        }
        VideoCodec::Gif => crate::video_anim::decode_gif_frames_yuv420(
            path,
            crate::video_anim::ANIM_DECODED_BUDGET,
        ),
        VideoCodec::Webp => crate::video_anim::decode_webp_frames_yuv420(
            path,
            crate::video_anim::ANIM_DECODED_BUDGET,
        ),
        other => Err(decode_error(
            path,
            format!("codec {other} is not a supported video decoder"),
        )),
    };
    let mut frames = decoded?;
    // The streaming path applies the signalled track colour to every frame;
    // do the same for the buffered path so RGB-producing encoders agree with
    // it. JPEG-derived frames are already BT.601 limited and carry no signal,
    // so they keep the explicit default.
    for frame in &mut frames {
        if !frame.color.signalled {
            frame.color = track_color;
        }
    }
    Ok(frames)
}

/// Streaming NV12 decode path shared by the CPU animation pipeline and the
/// NVENC backend.
mod streaming {
    use std::ops::ControlFlow;

    use super::{
        Av1Decoder, Av1Planes, ContainerPacket, ContainerTrack, DecodedFrameInfo, H264Decoder,
        H264YuvFrame, HevcDecoder, Path, PipelineError, PixelLayout, VideoCodec, Vp9DecodedFrame,
        Vp9Decoder, Vp9Error, YuvColor, YuvMatrix, YuvRange, avcc_to_annex_b, decode_error,
        decode_order, has_b_frame_reordering, hevc_picture_duration, hevc_picture_pts,
        hevc_pts_to_packet, hvcc_to_annex_b, median_duration, mjpeg_decode_rgb, source_color,
    };

    type PackFn<'a> = dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError> + 'a;

    /// Estimates the frame rate implied by the median packet duration, rounded
    /// to the nearest integer so 16,666 µs and 16,667 µs both resolve to 60 fps.
    ///
    /// Kept as a pure helper so the streaming GPU path and CPU path share one
    /// definition of "fps" derived from container timing. Only the optional
    /// NVENC backend consumes it in a non-test build.
    #[cfg_attr(not(any(feature = "nvenc", test)), allow(dead_code))]
    pub(crate) fn estimate_video_fps_from_packets(packets: &[ContainerPacket]) -> u32 {
        let duration = median_duration(packets).max(1);
        let fps = 1_000_000_u64.saturating_add(duration / 2) / duration;
        u32::try_from(fps.clamp(1, 240)).unwrap_or(30)
    }

    /// Byte length of one pitched NV12 frame (Y plane then interleaved UV plane).
    ///
    /// Validates even, non-zero dimensions and that `pitch` can hold a full row.
    pub(crate) fn nv12_buffer_len(
        width: u32,
        height: u32,
        pitch: usize,
    ) -> Result<usize, PipelineError> {
        let path = Path::new("nv12-buffer");
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(decode_error(
                path,
                format!("NV12 requires even, non-zero dimensions; got {width}x{height}"),
            ));
        }
        let w = usize::try_from(width).map_err(|_| decode_error(path, "NV12 width overflow"))?;
        let h = usize::try_from(height).map_err(|_| decode_error(path, "NV12 height overflow"))?;
        if pitch < w {
            return Err(decode_error(
                path,
                format!("NV12 pitch {pitch} is smaller than width {w}"),
            ));
        }
        let rows = h
            .checked_add(h / 2)
            .ok_or_else(|| decode_error(path, "NV12 row count overflow"))?;
        pitch
            .checked_mul(rows)
            .ok_or_else(|| decode_error(path, "NV12 buffer length overflow"))
    }

    /// Writes one decoded frame into a caller-owned NV12 buffer.
    ///
    /// Rows are written sequentially into `dst` and padding bytes past `width`
    /// are never touched; the caller zeroes the buffer once at allocation. Every
    /// row writer receives a slice covering exactly `width` bytes.
    fn write_nv12<FY, FUV>(
        dst: &mut [u8],
        pitch: usize,
        width: usize,
        height: usize,
        path: &Path,
        write_y: &mut FY,
        write_uv: &mut FUV,
    ) -> Result<(), PipelineError>
    where
        FY: FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        FUV: FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
    {
        if pitch < width {
            return Err(decode_error(
                path,
                format!("NV12 pitch {pitch} is smaller than width {width}"),
            ));
        }
        let uv_rows = height / 2;
        let plane_rows = height
            .checked_add(uv_rows)
            .ok_or_else(|| decode_error(path, "NV12 row count overflow"))?;
        let required = pitch
            .checked_mul(plane_rows)
            .ok_or_else(|| decode_error(path, "NV12 buffer size overflow"))?;
        if dst.len() < required {
            return Err(decode_error(
                path,
                format!(
                    "NV12 buffer of {} bytes is smaller than required {required}",
                    dst.len()
                ),
            ));
        }
        for row in 0..height {
            let start = row * pitch;
            write_y(&mut dst[start..start + width], row)?;
        }
        let uv_base = height * pitch;
        for row in 0..uv_rows {
            let start = uv_base + row * pitch;
            write_uv(&mut dst[start..start + width], row)?;
        }
        Ok(())
    }

    /// Validates a decoded plane is large enough for the requested crop window.
    fn require_plane_dims(
        path: &Path,
        name: &str,
        (need_w, need_h): (usize, usize),
        (got_w, got_h): (usize, usize),
    ) -> Result<(), PipelineError> {
        if got_w < need_w || got_h < need_h {
            return Err(decode_error(
                path,
                format!("AV1 {name} plane {got_w}x{got_h} is smaller than {need_w}x{need_h}"),
            ));
        }
        Ok(())
    }

    /// Packs one decoded frame through the caller's callback. Geometry is checked
    /// once here so every codec's row writer can rely on even, non-zero dimensions.
    fn emit_nv12_frame<FY, FUV>(
        path: &Path,
        on_frame: &mut impl FnMut(
            DecodedFrameInfo,
            &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        ) -> Result<(), PipelineError>,
        info: DecodedFrameInfo,
        mut write_y: FY,
        mut write_uv: FUV,
    ) -> Result<(), PipelineError>
    where
        FY: FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        FUV: FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
    {
        let w =
            usize::try_from(info.width).map_err(|_| decode_error(path, "frame width overflow"))?;
        let h = usize::try_from(info.height)
            .map_err(|_| decode_error(path, "frame height overflow"))?;
        if w == 0 || h == 0 || !w.is_multiple_of(2) || !h.is_multiple_of(2) {
            return Err(decode_error(
                path,
                format!(
                    "decoded 4:2:0 frame has odd or zero dimensions: {}x{}",
                    info.width, info.height
                ),
            ));
        }
        let mut pack = |dst: &mut [u8], pitch: usize| -> Result<(), PipelineError> {
            write_nv12(dst, pitch, w, h, path, &mut write_y, &mut write_uv)
        };
        on_frame(info, &mut pack)
    }

    fn emit_av1_depth8<F>(
        path: &Path,
        planes: &rav1d_safe::src::managed::Planes8<'_>,
        info: DecodedFrameInfo,
        width: usize,
        height: usize,
        on_frame: &mut F,
    ) -> Result<(), PipelineError>
    where
        F: FnMut(
            DecodedFrameInfo,
            &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        ) -> Result<(), PipelineError>,
    {
        let y_plane = planes.y();
        let u_plane = planes
            .u()
            .ok_or_else(|| decode_error(path, "AV1 frame has no U plane"))?;
        let v_plane = planes
            .v()
            .ok_or_else(|| decode_error(path, "AV1 frame has no V plane"))?;
        require_plane_dims(
            path,
            "Y",
            (width, height),
            (y_plane.width(), y_plane.height()),
        )?;
        let cw = width / 2;
        require_plane_dims(
            path,
            "U",
            (cw, height / 2),
            (u_plane.width(), u_plane.height()),
        )?;
        require_plane_dims(
            path,
            "V",
            (cw, height / 2),
            (v_plane.width(), v_plane.height()),
        )?;
        let write_y = |dst_row: &mut [u8], row: usize| {
            let src = y_plane
                .row(row)
                .get(..width)
                .ok_or_else(|| decode_error(path, "AV1 Y row is too short"))?;
            dst_row.copy_from_slice(src);
            Ok(())
        };
        let write_uv = |dst_row: &mut [u8], row: usize| {
            let u_src = u_plane
                .row(row)
                .get(..cw)
                .ok_or_else(|| decode_error(path, "AV1 U row is too short"))?;
            let v_src = v_plane
                .row(row)
                .get(..cw)
                .ok_or_else(|| decode_error(path, "AV1 V row is too short"))?;
            for (pair, (&u, &v)) in dst_row
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(u_src.iter().zip(v_src.iter()))
            {
                pair[0] = u;
                pair[1] = v;
            }
            Ok(())
        };
        emit_nv12_frame(path, on_frame, info, write_y, write_uv)
    }

    fn emit_av1_depth16<F>(
        path: &Path,
        planes: &rav1d_safe::src::managed::Planes16<'_>,
        shift: u8,
        info: DecodedFrameInfo,
        width: usize,
        height: usize,
        on_frame: &mut F,
    ) -> Result<(), PipelineError>
    where
        F: FnMut(
            DecodedFrameInfo,
            &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        ) -> Result<(), PipelineError>,
    {
        let y_plane = planes.y();
        let u_plane = planes
            .u()
            .ok_or_else(|| decode_error(path, "AV1 frame has no U plane"))?;
        let v_plane = planes
            .v()
            .ok_or_else(|| decode_error(path, "AV1 frame has no V plane"))?;
        require_plane_dims(
            path,
            "Y",
            (width, height),
            (y_plane.width(), y_plane.height()),
        )?;
        let cw = width / 2;
        require_plane_dims(
            path,
            "U",
            (cw, height / 2),
            (u_plane.width(), u_plane.height()),
        )?;
        require_plane_dims(
            path,
            "V",
            (cw, height / 2),
            (v_plane.width(), v_plane.height()),
        )?;
        let write_y = |dst_row: &mut [u8], row: usize| {
            let src = y_plane
                .row(row)
                .get(..width)
                .ok_or_else(|| decode_error(path, "AV1 Y row is too short"))?;
            for (slot, &sample) in dst_row.iter_mut().zip(src.iter()) {
                *slot = u8::try_from(sample >> shift).unwrap_or(u8::MAX);
            }
            Ok(())
        };
        let write_uv = |dst_row: &mut [u8], row: usize| {
            let u_src = u_plane
                .row(row)
                .get(..cw)
                .ok_or_else(|| decode_error(path, "AV1 U row is too short"))?;
            let v_src = v_plane
                .row(row)
                .get(..cw)
                .ok_or_else(|| decode_error(path, "AV1 V row is too short"))?;
            for (pair, (&u, &v)) in dst_row
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(u_src.iter().zip(v_src.iter()))
            {
                pair[0] = u8::try_from(u >> shift).unwrap_or(u8::MAX);
                pair[1] = u8::try_from(v >> shift).unwrap_or(u8::MAX);
            }
            Ok(())
        };
        emit_nv12_frame(path, on_frame, info, write_y, write_uv)
    }

    /// Maps the AV1 signal description onto our RGB matrix/range model. The
    /// range is independent of the matrix: an AV1 frame can be full range with
    /// no matrix description at all.
    pub(super) fn av1_color(
        matrix: rav1d_safe::src::managed::MatrixCoefficients,
        range: YuvRange,
    ) -> YuvColor {
        use rav1d_safe::src::managed::MatrixCoefficients as Mc;
        match matrix {
            Mc::BT709 => YuvColor::specified(YuvMatrix::Bt709, range),
            Mc::BT2020NCL => YuvColor::specified(YuvMatrix::Bt2020, range),
            Mc::BT470BG | Mc::BT601 | Mc::Unspecified | Mc::Reserved => {
                YuvColor::specified(YuvMatrix::Bt601, range)
            }
            other => YuvColor::specified(YuvMatrix::Unsupported(other as u8), range),
        }
    }

    fn emit_av1_frame<F>(
        path: &Path,
        frame: &rav1d_safe::src::managed::Frame,
        pts_us: i64,
        duration_us: u64,
        on_frame: &mut F,
    ) -> Result<(), PipelineError>
    where
        F: FnMut(
            DecodedFrameInfo,
            &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        ) -> Result<(), PipelineError>,
    {
        let width = usize::try_from(frame.width()).map_err(|_| decode_error(path, "AV1 width"))?;
        let height =
            usize::try_from(frame.height()).map_err(|_| decode_error(path, "AV1 height"))?;
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(decode_error(
                path,
                format!("AV1 decoded odd or zero dimensions: {width}x{height}"),
            ));
        }
        let layout = frame.pixel_layout();
        if layout != PixelLayout::I420 {
            return Err(decode_error(
                path,
                format!("AV1 chroma layout {layout:?} is not 4:2:0"),
            ));
        }
        let infos = frame.color_info();
        let range = match infos.color_range {
            rav1d_safe::src::managed::ColorRange::Full => YuvRange::Full,
            rav1d_safe::src::managed::ColorRange::Limited => YuvRange::Limited,
        };
        let color = av1_color(infos.matrix_coefficients, range);
        let info = DecodedFrameInfo {
            width: frame.width(),
            height: frame.height(),
            pts_us,
            duration_us,
            color,
        };
        match frame.planes() {
            Av1Planes::Depth8(planes) => {
                emit_av1_depth8(path, &planes, info, width, height, on_frame)
            }
            Av1Planes::Depth16(planes) => emit_av1_depth16(
                path,
                &planes,
                frame.bit_depth().saturating_sub(8),
                info,
                width,
                height,
                on_frame,
            ),
        }
    }

    fn emit_h264_frame<F>(
        path: &Path,
        frame: &H264YuvFrame,
        pts_us: i64,
        duration_us: u64,
        on_frame: &mut F,
    ) -> Result<(), PipelineError>
    where
        F: FnMut(
            DecodedFrameInfo,
            &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        ) -> Result<(), PipelineError>,
    {
        let width = frame.width;
        let height = frame.height;
        let cw = width / 2;
        let info = DecodedFrameInfo {
            width: u32::try_from(width).unwrap_or(u32::MAX),
            height: u32::try_from(height).unwrap_or(u32::MAX),
            pts_us,
            duration_us,
            color: YuvColor::default(),
        };
        let write_y = |dst_row: &mut [u8], row: usize| {
            let start = row
                .checked_mul(width)
                .ok_or_else(|| decode_error(path, "H.264 Y row offset overflow"))?;
            let end = start
                .checked_add(width)
                .ok_or_else(|| decode_error(path, "H.264 Y row offset overflow"))?;
            let src = frame
                .y
                .get(start..end)
                .ok_or_else(|| decode_error(path, "H.264 Y row out of bounds"))?;
            dst_row.copy_from_slice(src);
            Ok(())
        };
        let write_uv = |dst_row: &mut [u8], row: usize| {
            let start = row
                .checked_mul(cw)
                .ok_or_else(|| decode_error(path, "H.264 chroma row offset overflow"))?;
            let end = start
                .checked_add(cw)
                .ok_or_else(|| decode_error(path, "H.264 chroma row offset overflow"))?;
            let u_src = frame
                .u
                .get(start..end)
                .ok_or_else(|| decode_error(path, "H.264 U row out of bounds"))?;
            let v_src = frame
                .v
                .get(start..end)
                .ok_or_else(|| decode_error(path, "H.264 V row out of bounds"))?;
            for (pair, (&u, &v)) in dst_row
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(u_src.iter().zip(v_src.iter()))
            {
                pair[0] = u;
                pair[1] = v;
            }
            Ok(())
        };
        emit_nv12_frame(path, on_frame, info, write_y, write_uv)
    }

    fn emit_hevc_frame<F>(
        path: &Path,
        frame: &rusty_h265::Frame,
        pts_us: i64,
        duration_us: u64,
        on_frame: &mut F,
    ) -> Result<(), PipelineError>
    where
        F: FnMut(
            DecodedFrameInfo,
            &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        ) -> Result<(), PipelineError>,
    {
        let width = frame.width;
        let height = frame.height;
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(decode_error(
                path,
                format!("HEVC decoded odd or zero dimensions: {width}x{height}"),
            ));
        }
        let picture = &frame.picture;
        let (crop_x, crop_y) = (picture.crop.0, picture.crop.1);
        if picture.chroma_format_idc != 1 {
            return Err(decode_error(
                path,
                format!(
                    "HEVC chroma format {} is not 4:2:0",
                    picture.chroma_format_idc
                ),
            ));
        }
        let cw = width / 2;
        let shift_y = picture.bit_depth_luma.saturating_sub(8);
        let shift_c = picture.bit_depth_chroma.saturating_sub(8);
        let info = DecodedFrameInfo {
            width: u32::try_from(width).unwrap_or(u32::MAX),
            height: u32::try_from(height).unwrap_or(u32::MAX),
            pts_us,
            duration_us,
            color: YuvColor::default(),
        };
        let write_y = |dst_row: &mut [u8], row: usize| {
            let plane = &picture.planes[0];
            let start = row
                .checked_add(crop_y)
                .and_then(|r| r.checked_mul(plane.stride))
                .and_then(|s| s.checked_add(crop_x))
                .ok_or_else(|| decode_error(path, "HEVC Y row offset overflow"))?;
            let end = start
                .checked_add(width)
                .ok_or_else(|| decode_error(path, "HEVC Y row offset overflow"))?;
            let src = plane
                .data
                .get(start..end)
                .ok_or_else(|| decode_error(path, "HEVC Y row out of bounds"))?;
            for (slot, &sample) in dst_row.iter_mut().zip(src.iter()) {
                *slot = u8::try_from(sample >> shift_y).unwrap_or(u8::MAX);
            }
            Ok(())
        };
        let write_uv = |dst_row: &mut [u8], row: usize| {
            let u_plane = &picture.planes[1];
            let v_plane = &picture.planes[2];
            let (cx, cy) = (crop_x / 2, crop_y / 2);
            let u_start = row
                .checked_add(cy)
                .and_then(|r| r.checked_mul(u_plane.stride))
                .and_then(|s| s.checked_add(cx))
                .ok_or_else(|| decode_error(path, "HEVC U row offset overflow"))?;
            let v_start = row
                .checked_add(cy)
                .and_then(|r| r.checked_mul(v_plane.stride))
                .and_then(|s| s.checked_add(cx))
                .ok_or_else(|| decode_error(path, "HEVC V row offset overflow"))?;
            let u_src = u_plane
                .data
                .get(u_start..u_start.saturating_add(cw))
                .ok_or_else(|| decode_error(path, "HEVC U row out of bounds"))?;
            let v_src = v_plane
                .data
                .get(v_start..v_start.saturating_add(cw))
                .ok_or_else(|| decode_error(path, "HEVC V row out of bounds"))?;
            for (pair, (&u, &v)) in dst_row
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(u_src.iter().zip(v_src.iter()))
            {
                pair[0] = u8::try_from(u >> shift_c).unwrap_or(u8::MAX);
                pair[1] = u8::try_from(v >> shift_c).unwrap_or(u8::MAX);
            }
            Ok(())
        };
        emit_nv12_frame(path, on_frame, info, write_y, write_uv)
    }

    fn vp9_sample(
        plane: &[u8],
        stride: usize,
        row: usize,
        x: usize,
        bit_depth: u32,
        path: &Path,
    ) -> Result<u8, PipelineError> {
        if bit_depth <= 8 {
            let index = row
                .checked_mul(stride)
                .and_then(|v| v.checked_add(x))
                .ok_or_else(|| decode_error(path, "VP9 sample offset overflow"))?;
            plane
                .get(index)
                .copied()
                .ok_or_else(|| decode_error(path, "VP9 8-bit sample out of bounds"))
        } else {
            let shift = bit_depth.saturating_sub(8);
            let x2 = x
                .checked_mul(2)
                .ok_or_else(|| decode_error(path, "VP9 sample offset overflow"))?;
            let index = row
                .checked_mul(stride)
                .and_then(|v| v.checked_add(x2))
                .ok_or_else(|| decode_error(path, "VP9 sample offset overflow"))?;
            let bytes = plane
                .get(index..index.saturating_add(2))
                .ok_or_else(|| decode_error(path, "VP9 10-bit sample out of bounds"))?;
            Ok(u8::try_from(u16::from_le_bytes([bytes[0], bytes[1]]) >> shift).unwrap_or(u8::MAX))
        }
    }

    fn emit_vp9_frame<F>(
        path: &Path,
        frame: &Vp9DecodedFrame,
        duration_us: u64,
        on_frame: &mut F,
    ) -> Result<(), PipelineError>
    where
        F: FnMut(
            DecodedFrameInfo,
            &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        ) -> Result<(), PipelineError>,
    {
        let width = usize::try_from(frame.width).map_err(|_| decode_error(path, "VP9 width"))?;
        let height = usize::try_from(frame.height).map_err(|_| decode_error(path, "VP9 height"))?;
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(decode_error(
                path,
                format!("VP9 decoded odd or zero dimensions: {width}x{height}"),
            ));
        }
        let planes = &frame.planes;
        let strides = &frame.strides;
        let plane = |index: usize| {
            planes
                .get(index)
                .map(Vec::as_slice)
                .ok_or_else(|| decode_error(path, "VP9 frame is missing a plane"))
        };
        let stride = |index: usize| {
            strides
                .get(index)
                .copied()
                .ok_or_else(|| decode_error(path, "VP9 frame is missing a stride"))
        };
        let y_plane = plane(0)?;
        let y_stride = stride(0)?;
        let u_plane = plane(1)?;
        let u_stride = stride(1)?;
        let v_plane = plane(2)?;
        let v_stride = stride(2)?;
        let bit_depth = frame.bit_depth;
        let write_y = |dst_row: &mut [u8], row: usize| {
            for (x, slot) in dst_row.iter_mut().enumerate() {
                *slot = vp9_sample(y_plane, y_stride, row, x, bit_depth, path)?;
            }
            Ok(())
        };
        let write_uv = |dst_row: &mut [u8], row: usize| {
            for (x, pair) in dst_row.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                pair[0] = vp9_sample(u_plane, u_stride, row, x, bit_depth, path)?;
                pair[1] = vp9_sample(v_plane, v_stride, row, x, bit_depth, path)?;
            }
            Ok(())
        };
        let info = DecodedFrameInfo {
            width: u32::try_from(width).unwrap_or(u32::MAX),
            height: u32::try_from(height).unwrap_or(u32::MAX),
            pts_us: frame.pts.unwrap_or(0),
            duration_us,
            color: YuvColor::default(),
        };
        emit_nv12_frame(path, on_frame, info, write_y, write_uv)
    }

    fn drain_ready_stream_h264_reorder<F>(
        path: &Path,
        reorder: &mut Vec<(H264YuvFrame, i64, u64)>,
        future_min_pts: Option<i64>,
        on_frame: &mut F,
    ) -> Result<(), PipelineError>
    where
        F: FnMut(
            DecodedFrameInfo,
            &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        ) -> Result<(), PipelineError>,
    {
        while !reorder.is_empty() {
            let earliest = reorder
                .iter()
                .enumerate()
                .min_by_key(|(_, (_, pts, _))| *pts)
                .map_or(0, |(index, _)| index);
            let safe_to_emit =
                future_min_pts.is_none_or(|future_min| reorder[earliest].1 <= future_min);
            if safe_to_emit {
                let (frame, pts, duration) = reorder.remove(earliest);
                emit_h264_frame(path, &frame, pts, duration, on_frame)?;
            } else if reorder.len() > 16 {
                return Err(decode_error(
                    path,
                    "H.264 reorder buffer exceeded 16 frames before presentation order was provable",
                ));
            } else {
                break;
            }
        }
        Ok(())
    }

    fn flush_stream_h264_reorder<F>(
        path: &Path,
        reorder: &mut Vec<(H264YuvFrame, i64, u64)>,
        on_frame: &mut F,
    ) -> Result<(), PipelineError>
    where
        F: FnMut(
            DecodedFrameInfo,
            &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        ) -> Result<(), PipelineError>,
    {
        reorder.sort_by_key(|(_, pts, _)| *pts);
        for (frame, pts, duration) in reorder.drain(..) {
            emit_h264_frame(path, &frame, pts, duration, on_frame)?;
        }
        Ok(())
    }

    fn stream_h264(
        path: &Path,
        track: &ContainerTrack,
        ordered: &[&ContainerPacket],
        on_frame: &mut impl FnMut(
            DecodedFrameInfo,
            &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        ) -> Result<(), PipelineError>,
    ) -> Result<(), PipelineError> {
        let mut decoder = H264Decoder::new();
        let needs_reorder = has_b_frame_reordering(ordered);
        let future_mins = if needs_reorder {
            super::suffix_min_pts(ordered)
        } else {
            Vec::new()
        };
        let mut reorder: Vec<(H264YuvFrame, i64, u64)> =
            Vec::with_capacity(if needs_reorder { 16 } else { 0 });
        for (index, packet) in ordered.iter().enumerate() {
            let annex_b = avcc_to_annex_b(&track.config, &packet.data, true)
                .map_err(|reason| decode_error(path, reason))?;
            let frame = decoder
                .decode(&annex_b)
                .map_err(|err| decode_error(path, format!("H.264 decode failed: {err}")))?;
            if let Some(frame) = frame {
                if needs_reorder {
                    reorder.push((frame, packet.pts, packet.duration));
                    let future_min = future_mins.get(index + 1).copied();
                    drain_ready_stream_h264_reorder(path, &mut reorder, future_min, on_frame)?;
                } else {
                    emit_h264_frame(path, &frame, packet.pts, packet.duration, on_frame)?;
                }
            }
        }
        if needs_reorder {
            flush_stream_h264_reorder(path, &mut reorder, on_frame)?;
        }
        Ok(())
    }

    fn stream_hevc(
        path: &Path,
        track: &ContainerTrack,
        ordered: &[&ContainerPacket],
        packets: &[ContainerPacket],
        on_frame: &mut impl FnMut(
            DecodedFrameInfo,
            &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        ) -> Result<(), PipelineError>,
    ) -> Result<(), PipelineError> {
        let mut decoder = HevcDecoder::new();
        let mapping = hevc_pts_to_packet(packets);
        let fallback_dur = median_duration(packets);
        let mut last_pts_dur: Option<(i64, u64)> = None;
        for packet in ordered {
            let annex_b = hvcc_to_annex_b(&track.config, &packet.data, true)
                .map_err(|reason| decode_error(path, reason))?;
            decoder
                .push_annexb(&annex_b, Some(packet.pts))
                .map_err(|err| decode_error(path, format!("HEVC decode failed: {err}")))?;
            loop {
                match decoder.next_frame() {
                    Ok(frame) => {
                        let pts = hevc_picture_pts(&frame, packet.pts);
                        let duration = hevc_picture_duration(pts, packets, &mapping, fallback_dur);
                        emit_hevc_frame(path, &frame, pts, duration, on_frame)?;
                        last_pts_dur = Some((pts, duration));
                    }
                    Err(rusty_h265::Error::Again | rusty_h265::Error::Eof) => break,
                    Err(err) => {
                        return Err(decode_error(path, format!("HEVC drain failed: {err}")));
                    }
                }
            }
        }
        decoder.flush();
        loop {
            match decoder.next_frame() {
                Ok(frame) => {
                    let fallback_pts = last_pts_dur.map_or(0, |(pts, duration)| {
                        pts.saturating_add(i64::try_from(duration).unwrap_or(1))
                    });
                    let pts = hevc_picture_pts(&frame, fallback_pts);
                    let duration = hevc_picture_duration(pts, packets, &mapping, fallback_dur);
                    emit_hevc_frame(path, &frame, pts, duration, on_frame)?;
                    last_pts_dur = Some((pts, duration));
                }
                Err(rusty_h265::Error::Again | rusty_h265::Error::Eof) => break,
                Err(err) => return Err(decode_error(path, format!("HEVC flush failed: {err}"))),
            }
        }
        Ok(())
    }

    fn stream_av1(
        path: &Path,
        ordered: &[&ContainerPacket],
        packets: &[ContainerPacket],
        on_frame: &mut impl FnMut(
            DecodedFrameInfo,
            &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        ) -> Result<(), PipelineError>,
    ) -> Result<(), PipelineError> {
        let mut decoder = Av1Decoder::new().map_err(|err| decode_error(path, format!("{err}")))?;
        let mut emitted = 0_usize;
        for packet in ordered {
            if let Some(frame) = decoder
                .decode(&packet.data)
                .map_err(|err| decode_error(path, format!("AV1 decode failed: {err}")))?
            {
                emit_av1_frame(path, &frame, packet.pts, packet.duration, on_frame)?;
                emitted = emitted.saturating_add(1);
            }
            while let Some(frame) = decoder
                .get_frame()
                .map_err(|err| decode_error(path, format!("AV1 drain failed: {err}")))?
            {
                emit_av1_frame(path, &frame, packet.pts, packet.duration, on_frame)?;
                emitted = emitted.saturating_add(1);
            }
        }
        let median = median_duration(packets);
        for frame in decoder
            .flush()
            .map_err(|err| decode_error(path, format!("AV1 flush failed: {err}")))?
        {
            let pts = u64::try_from(emitted)
                .unwrap_or(u64::MAX)
                .saturating_mul(median);
            emit_av1_frame(
                path,
                &frame,
                i64::try_from(pts).unwrap_or(i64::MAX),
                median,
                on_frame,
            )?;
            emitted = emitted.saturating_add(1);
        }
        Ok(())
    }

    fn stream_vp9(
        path: &Path,
        ordered: &[&ContainerPacket],
        packets: &[ContainerPacket],
        on_frame: &mut impl FnMut(
            DecodedFrameInfo,
            &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        ) -> Result<(), PipelineError>,
    ) -> Result<(), PipelineError> {
        let mut decoder = Vp9Decoder::new();
        for packet in ordered {
            decoder
                .push(&packet.data, Some(packet.pts))
                .map_err(|err| decode_error(path, format!("VP9 push failed: {err}")))?;
            loop {
                match decoder.next_frame() {
                    Ok(frame) => emit_vp9_frame(path, &frame, packet.duration, on_frame)?,
                    Err(Vp9Error::Again | Vp9Error::Eof) => break,
                    Err(err) => return Err(decode_error(path, format!("VP9 drain failed: {err}"))),
                }
            }
        }
        decoder.flush();
        loop {
            match decoder.next_frame() {
                Ok(frame) => emit_vp9_frame(path, &frame, median_duration(packets), on_frame)?,
                Err(Vp9Error::Again | Vp9Error::Eof) => break,
                Err(err) => return Err(decode_error(path, format!("VP9 flush failed: {err}"))),
            }
        }
        Ok(())
    }

    /// MJPEG has no reorder state: decode each JPEG into RGB, then write it
    /// straight into the caller's NV12 buffer. Odd dimensions are padded by
    /// edge replication inside `write_nv12`, so the reported frame geometry is
    /// rounded up to even exactly like every other 4:2:0 decoder here.
    fn stream_mjpeg(
        path: &Path,
        ordered: &[&ContainerPacket],
        on_frame: &mut impl FnMut(
            DecodedFrameInfo,
            &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        ) -> Result<(), PipelineError>,
    ) -> Result<(), PipelineError> {
        let mut rgb = Vec::new();
        for packet in ordered {
            let (width, height) = mjpeg_decode_rgb(path, &packet.data, &mut rgb)?;
            let width = u32::from(width);
            let height = u32::from(height);
            let even_width = width
                .checked_add(width & 1)
                .ok_or_else(|| decode_error(path, "MJPEG width overflows when padded"))?;
            let even_height = height
                .checked_add(height & 1)
                .ok_or_else(|| decode_error(path, "MJPEG height overflows when padded"))?;
            let info = DecodedFrameInfo {
                width: even_width,
                height: even_height,
                pts_us: packet.pts,
                duration_us: packet.duration,
                // JPEG is decoded to RGB first; the fixed BT.601 conversion in
                // `write_nv12` is the same one used by the animation path.
                color: YuvColor::default(),
            };
            let mut pack = |dst: &mut [u8], pitch: usize| {
                crate::video_anim::write_nv12(&rgb, width, height, dst, pitch)
            };
            on_frame(info, &mut pack)?;
        }
        Ok(())
    }

    /// Decodes one canonical video track and streams each frame straight into a
    /// caller-provided NV12 buffer via `on_frame`. No `Yuv420Frame` is ever built.
    ///
    /// The callback receives frame metadata plus a `pack_nv12` closure with the
    /// signature `FnMut(&mut [u8], usize)`. It supplies the destination buffer and
    /// pitch; the closure writes one NV12 frame directly into it. Returning `Err`
    /// from the callback aborts decoding immediately.
    // One branch per codec keeps decode-order and timestamp handling explicit.
    pub(crate) fn decode_video_track_streaming<F>(
        path: &Path,
        track: &ContainerTrack,
        packets: &[ContainerPacket],
        mut on_frame: F,
    ) -> Result<(), PipelineError>
    where
        F: FnMut(
            DecodedFrameInfo,
            &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        ) -> Result<(), PipelineError>,
    {
        let ordered = decode_order(packets);
        // Derive the source colour description once per track. Frames whose
        // decoder signals its own colour (AV1) keep that; everything else gets
        // the signalled track value, or the explicit BT.601 default only when
        // the bitstream carries no colour description at all.
        let track_color = source_color(track, &ordered);
        let mut deliver =
            |mut info: DecodedFrameInfo, pack: &mut PackFn<'_>| -> Result<(), PipelineError> {
                if !info.color.signalled {
                    info.color = track_color;
                }
                on_frame(info, pack)
            };
        match track.codec {
            VideoCodec::H264 => stream_h264(path, track, &ordered, &mut deliver),
            VideoCodec::Hevc => stream_hevc(path, track, &ordered, packets, &mut deliver),
            VideoCodec::Av1 => stream_av1(path, &ordered, packets, &mut deliver),
            VideoCodec::Vp9 => stream_vp9(path, &ordered, packets, &mut deliver),
            VideoCodec::Mjpeg => stream_mjpeg(path, &ordered, &mut deliver),
            VideoCodec::Gif => {
                crate::video_anim::decode_gif_frames_streaming_nv12(path, |info, pack| {
                    deliver(
                        DecodedFrameInfo {
                            width: info.width,
                            height: info.height,
                            pts_us: info.pts_us,
                            duration_us: info.duration_us,
                            color: YuvColor::default(),
                        },
                        pack,
                    )
                })
            }
            VideoCodec::Webp => {
                crate::video_anim::decode_webp_frames_streaming_nv12(path, |info, pack| {
                    deliver(
                        DecodedFrameInfo {
                            width: info.width,
                            height: info.height,
                            pts_us: info.pts_us,
                            duration_us: info.duration_us,
                            color: YuvColor::default(),
                        },
                        pack,
                    )
                })
            }
            other => Err(decode_error(
                path,
                format!("codec {other} is not a supported video decoder"),
            )),
        }
    }

    /// `ControlFlow`-returning variant used by frame extraction. A `Break`
    /// from the callback stops decoding immediately: the existing `Result`-
    /// based plumbing is translated through a private sentinel that can never
    /// escape into a real pipeline error.
    pub(crate) fn decode_video_track_streaming_ctrl<F>(
        path: &Path,
        track: &ContainerTrack,
        packets: &[ContainerPacket],
        mut on_frame: F,
    ) -> Result<ControlFlow<()>, PipelineError>
    where
        F: FnMut(
            DecodedFrameInfo,
            &mut dyn FnMut(&mut [u8], usize) -> Result<(), PipelineError>,
        ) -> Result<ControlFlow<()>, PipelineError>,
    {
        let mut stopped = false;
        let wrapped =
            |info: DecodedFrameInfo, pack: &mut PackFn<'_>| -> Result<(), PipelineError> {
                match on_frame(info, pack)? {
                    ControlFlow::Continue(()) => Ok(()),
                    ControlFlow::Break(()) => {
                        stopped = true;
                        Err(decode_error(path, crate::video_anim::FRAMES_EARLY_STOP))
                    }
                }
            };
        match decode_video_track_streaming(path, track, packets, wrapped) {
            Ok(()) => Ok(ControlFlow::Continue(())),
            Err(PipelineError::DecodeFailed { reason, .. })
                if stopped && reason == crate::video_anim::FRAMES_EARLY_STOP =>
            {
                Ok(ControlFlow::Break(()))
            }
            Err(err) => Err(err),
        }
    }
} // mod streaming

/// Reads the source colour description from canonical codec configuration.
/// Unparseable or absent VUI data means "not signalled", never a silent guess.
/// Preserves `full_range` independently of matrix availability. Signalled
/// matrices without an RGB converter (constant luminance 10, SMPTE 240M, ...)
/// are carried as `Unsupported` so only the RGB conversion rejects them.
fn source_color(track: &ContainerTrack, packets: &[&ContainerPacket]) -> YuvColor {
    let resolve = |matrix: Option<u8>, full_range: bool| -> YuvColor {
        let range = if full_range {
            YuvRange::Full
        } else {
            YuvRange::Limited
        };
        let matrix = match matrix {
            Some(1) => YuvMatrix::Bt709,
            Some(9) => YuvMatrix::Bt2020,
            Some(code @ (10 | 4 | 7 | 8 | 11..=14)) => YuvMatrix::Unsupported(code),
            _ => YuvMatrix::Bt601,
        };
        YuvColor::specified(matrix, range)
    };
    match track.codec {
        VideoCodec::H264 => {
            let annex_b = avcc_to_annex_b(&track.config, &[], true).unwrap_or_default();
            match crate::video_bitstream::h264_color_from_annex_b(&annex_b) {
                Some(color) => resolve(color.matrix, color.full_range),
                None => YuvColor::default(),
            }
        }
        VideoCodec::Hevc => {
            let annex_b = hvcc_to_annex_b(&track.config, &[], true).unwrap_or_default();
            match crate::video_bitstream::hevc_color_from_annex_b(&annex_b) {
                Some(color) => resolve(color.matrix, color.full_range),
                None => YuvColor::default(),
            }
        }
        VideoCodec::Vp9 => match packets
            .iter()
            .find_map(|packet| crate::video_bitstream::vp9_color_from_keyframe(&packet.data))
        {
            Some(color) => resolve(Some(color.matrix), color.full_range),
            None => YuvColor::default(),
        },
        // AV1 frames signal their own colour; leave it unsignalled here so the
        // per-frame decoder metadata wins.
        _ => YuvColor::default(),
    }
}

#[cfg(any(feature = "nvenc", test))]
pub(crate) use streaming::estimate_video_fps_from_packets;
pub(crate) use streaming::{
    decode_video_track_streaming, decode_video_track_streaming_ctrl, nv12_buffer_len,
};

/// Splits an Annex-B byte stream into one packet per coded access unit,
/// grouping parameter sets and SEI with the VCL NAL that follows them.
fn split_annex_b_access_units(data: &[u8], hevc: bool) -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut idx = 0_usize;
    while idx.saturating_add(2) < data.len() {
        if data[idx] == 0 && data[idx + 1] == 0 && data[idx + 2] == 1 {
            starts.push(idx);
            idx = idx.saturating_add(3);
        } else {
            idx = idx.saturating_add(1);
        }
    }
    let mut units: Vec<Vec<u8>> = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut has_vcl = false;
    for (position, &start) in starts.iter().enumerate() {
        let end = starts
            .get(position.saturating_add(1))
            .copied()
            .unwrap_or(data.len());
        let payload_start = start
            .saturating_add(3)
            .saturating_add(usize::from(data.get(start.saturating_add(2)) == Some(&0)));
        let Some(header) = data.get(payload_start).copied() else {
            continue;
        };
        let nal_type = if hevc {
            (header >> 1) & 0x3F
        } else {
            header & 0x1F
        };
        let is_aud = if hevc { nal_type == 35 } else { nal_type == 9 };
        let is_vcl = if hevc {
            matches!(nal_type, 0..=21)
        } else {
            matches!(nal_type, 1 | 5)
        };
        if !current.is_empty() && (is_aud || (is_vcl && has_vcl)) {
            units.push(std::mem::take(&mut current));
            has_vcl = false;
        }
        let mut end_trimmed = end;
        while end_trimmed > payload_start && data[end_trimmed.saturating_sub(1)] == 0 {
            end_trimmed -= 1;
        }
        if end_trimmed > payload_start {
            current.extend_from_slice(&data[start..end_trimmed]);
        }
        has_vcl |= is_vcl;
    }
    if !current.is_empty() {
        units.push(current);
    }
    units
}

fn frame_pts(frame: &Yuv420Frame, index: usize, fps_num: u32, fps_den: u32) -> i64 {
    if frame.pts_us > i64::from(i32::MIN) {
        return frame.pts_us;
    }
    let ticks = u64::try_from(index)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::from(fps_den))
        .saturating_mul(u64::from(US_TIMESCALE))
        / u64::from(fps_num.max(1));
    i64::try_from(ticks).unwrap_or(i64::MAX)
}

// All packet metadata is passed explicitly to avoid an extra builder type.
#[allow(clippy::too_many_arguments)]
fn make_stream(
    path: &Path,
    track_index: usize,
    codec: VideoCodec,
    width: u32,
    height: u32,
    config: Vec<u8>,
    units: Vec<(Vec<u8>, bool)>,
    frames: &[Yuv420Frame],
    fps_num: u32,
    fps_den: u32,
) -> Result<EncodedVideoStream, PipelineError> {
    let mut track = ContainerTrack {
        codec,
        width: u16::try_from(width).ok(),
        height: u16::try_from(height).ok(),
        channels: None,
        sample_rate: None,
        config,
        timescale: US_TIMESCALE,
        presentation_edit: None,
        pcm_format: None,
    };
    let mut packets = Vec::with_capacity(units.len());
    let base_pts = frames
        .first()
        .filter(|frame| frame.pts_us > i64::from(i32::MIN))
        .map_or(0, |frame| frame.pts_us);
    let mut prev_pts: Option<i64> = None;
    for (index, (data, keyframe)) in units.into_iter().enumerate() {
        let frame = frames.get(index).or_else(|| frames.last());
        let (frame_pts_us, duration_us) = frame.map_or(
            (
                0,
                u64::from(fps_den).saturating_mul(1_000_000) / u64::from(fps_num.max(1)),
            ),
            |frame| {
                (
                    frame_pts(frame, index, fps_num, fps_den),
                    frame.duration_us.max(1),
                )
            },
        );
        let raw_pts = frame_pts_us.saturating_sub(base_pts).max(0);
        // Packets are sorted by DTS before muxing; strictly increasing times
        // keep decoded TS/HEVC frames from being reordered out of encode order.
        let min_step = i64::try_from(duration_us).unwrap_or(1).max(1);
        let pts_us = match prev_pts {
            None => 0,
            Some(prev) => {
                if raw_pts > prev {
                    raw_pts
                } else {
                    prev.saturating_add(min_step)
                }
            }
        };
        prev_pts = Some(pts_us);
        if track.config.is_empty()
            && keyframe
            && let Some(config) = match codec {
                VideoCodec::H264 => extract_avcc_from_annex_b(&data),
                VideoCodec::Hevc => extract_hvcc_from_annex_b(&data),
                VideoCodec::Av1 => extract_av1c_from_obu(&data),
                _ => None,
            }
        {
            track.config = config;
        }
        let payload = match codec {
            VideoCodec::H264 => annex_b_to_avcc(&data),
            VideoCodec::Hevc => annex_b_to_hvcc(&data),
            _ => data,
        };
        if payload.is_empty() {
            return Err(encode_error(path, "encoder produced an empty access unit"));
        }
        packets.push(ContainerPacket {
            track: track_index,
            pts: pts_us,
            dts: pts_us,
            duration: duration_us.max(1),
            keyframe,
            data: payload,
        });
    }
    if packets.is_empty() {
        return Err(encode_error(path, "encoder produced no video packets"));
    }
    Ok(EncodedVideoStream { track, packets })
}

fn quality_to_h264_qp(quality: VideoQuality) -> u8 {
    let inv = u32::from(100_u8.saturating_sub(quality.get()));
    // Higher quality must mean a lower QP, matching the NVENC curve.
    let qp = 8_u32.saturating_add(inv.saturating_mul(43).saturating_add(50) / 100);
    u8::try_from(qp.clamp(1, 51)).unwrap_or(51)
}

fn quality_to_vp9_qindex(quality: VideoQuality) -> u32 {
    let inv = u32::from(100_u8.saturating_sub(quality.get()));
    8_u32
        .saturating_add(inv.saturating_mul(200) / 100)
        .clamp(8, 208)
}

/// One H.264 access unit paired with its keyframe flag.
type H264Units = Vec<(Vec<u8>, bool)>;

fn h264_access_units(annex_b: &[u8]) -> H264Units {
    split_annex_b_access_units(annex_b, false)
        .into_iter()
        .map(|data| {
            let keyframe = is_h264_keyframe_annex_b(&data);
            (data, keyframe)
        })
        .collect()
}

/// Encodes independent GOPs concurrently. Every worker owns its encoder but
/// only borrows the source frames, so no plane data is cloned. GOPs are
/// collected back into display order before muxing. The scoped worker body,
/// error collection and ordered reassembly belong together: splitting them
/// would obscure the zero-copy borrow and the join-before-drain lifecycle.
#[allow(clippy::too_many_lines)]
fn cpu_h264_parallel_gops(
    path: &Path,
    frames: &[Yuv420Frame],
    w: usize,
    h: usize,
    gop_size: usize,
    config: &H264EncoderConfig,
) -> Result<Vec<(Vec<u8>, bool)>, PipelineError> {
    let gop_count = frames.len().div_ceil(gop_size);
    let workers = thread::available_parallelism()
        .unwrap_or(NonZeroUsize::MIN)
        .get()
        .min(gop_count)
        .min(8);
    let (tx, rx) = mpsc::channel::<(usize, H264Units)>();
    let next_gop = AtomicUsize::new(0);

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let tx = tx.clone();
            let next_gop = &next_gop;
            let worker_config = config.clone();
            handles.push(
                thread::Builder::new()
                    .name("fastconv-h264-gop".to_owned())
                    .spawn_scoped(scope, move || -> Result<(), PipelineError> {
                        loop {
                            let gop = next_gop.fetch_add(1, Ordering::Relaxed);
                            if gop >= gop_count {
                                return Ok(());
                            }
                            let start = gop.saturating_mul(gop_size);
                            let end = start.saturating_add(gop_size).min(frames.len());
                            let slice = &frames[start..end];
                            let mut encoder = H264Encoder::new(worker_config.clone())
                                .map_err(|err| encode_error(path, format!("{err}")))?;
                            let mut annex_b = Vec::new();
                            for frame in slice {
                                let planes =
                                    H264YuvPlanes::tight(w, h, &frame.y, &frame.u, &frame.v)
                                        .ok_or_else(|| {
                                            encode_error(
                                                path,
                                                "invalid H.264 source frame geometry",
                                            )
                                        })?;
                                annex_b.extend_from_slice(
                                    &encoder.encode_planes(&planes).map_err(|err| {
                                        encode_error(path, format!("H.264 encode failed: {err}"))
                                    })?,
                                );
                            }
                            annex_b.extend_from_slice(&encoder.flush());
                            let units = h264_access_units(&annex_b);
                            if units.len() != slice.len() {
                                return Err(encode_error(
                                    path,
                                    format!(
                                        "H.264 GOP {gop} produced {} access units for {} frames",
                                        units.len(),
                                        slice.len()
                                    ),
                                ));
                            }
                            if !units.first().is_some_and(|(_, keyframe)| *keyframe) {
                                return Err(encode_error(
                                    path,
                                    format!("H.264 GOP {gop} did not start with an IDR keyframe"),
                                ));
                            }
                            tx.send((gop, units))
                                .map_err(|_| PipelineError::WorkerPanicked)?;
                        }
                    })
                    .map_err(PipelineError::WorkerStartFailed)?,
            );
        }
        drop(tx);

        let mut worker_error = None;
        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    if worker_error.is_none() {
                        worker_error = Some(err);
                    }
                }
                Err(_) => {
                    if worker_error.is_none() {
                        worker_error = Some(PipelineError::WorkerPanicked);
                    }
                }
            }
        }
        if let Some(err) = worker_error {
            return Err(err);
        }

        let mut ordered: Vec<Option<H264Units>> = (0..gop_count).map(|_| None).collect();
        for (gop, units) in rx {
            if gop < gop_count {
                ordered[gop] = Some(units);
            }
        }
        let mut result = Vec::with_capacity(frames.len());
        for (gop, units) in ordered.into_iter().enumerate() {
            let units = units
                .ok_or_else(|| encode_error(path, format!("H.264 GOP {gop} produced no output")))?;
            result.extend(units);
        }
        if result.len() != frames.len() {
            return Err(encode_error(
                path,
                format!(
                    "H.264 produced {} access units for {} frames",
                    result.len(),
                    frames.len()
                ),
            ));
        }
        Ok(result)
    })
}

fn cpu_h264(
    path: &Path,
    frames: &[Yuv420Frame],
    rate_control: VideoRateControl,
    track_index: usize,
    fps_num: u32,
    fps_den: u32,
) -> Result<EncodedVideoStream, PipelineError> {
    let (width, height) = (frames[0].width, frames[0].height);
    require_even(path, width, height)?;
    let w = usize::try_from(width).map_err(|_| encode_error(path, "H.264 width overflow"))?;
    let h = usize::try_from(height).map_err(|_| encode_error(path, "H.264 height overflow"))?;
    // In bitrate mode rusty_h264 drives QP from `bitrate`/`framerate` and uses
    // `qp` only as the controller's base/fallback. Rate-control state spans the
    // whole clip, so parallel GOP encoders would each restart the controller.
    let (qp, bitrate, parallel_gops) = match rate_control {
        VideoRateControl::Quality(quality) => (quality_to_h264_qp(quality), 0, true),
        VideoRateControl::Bitrate(bitrate) => (
            quality_to_h264_qp(VideoQuality::DEFAULT),
            bitrate.bps(),
            false,
        ),
    };
    let mut config = H264EncoderConfig::new(w, h);
    config.qp = qp;
    config.preset = H264Preset::Fast;
    // One-second keyframe interval, bounded so low/high frame rates stay sane.
    let fps_rounded = fps_num.div_ceil(fps_den.max(1));
    let gop_size = fps_rounded.clamp(15, 60);
    config.gop_size = gop_size;
    // Frame rates from packet durations are small; f32 is the encoder's field type.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let framerate = (f64::from(fps_num) / f64::from(fps_den.max(1))) as f32;
    config.framerate = framerate;
    config.bitrate = bitrate;
    config.lookahead = 0;
    config.scenecut = 0;
    config.mbtree = false;

    let gop_len = usize::try_from(gop_size).unwrap_or(usize::MAX);
    let units = if parallel_gops && frames.len() > gop_len {
        cpu_h264_parallel_gops(path, frames, w, h, gop_len, &config)?
    } else {
        let mut encoder =
            H264Encoder::new(config).map_err(|err| encode_error(path, format!("{err}")))?;
        let mut annex_b = Vec::new();
        for frame in frames {
            let planes = H264YuvPlanes::tight(w, h, &frame.y, &frame.u, &frame.v)
                .ok_or_else(|| encode_error(path, "invalid H.264 source frame geometry"))?;
            annex_b.extend_from_slice(
                &encoder
                    .encode_planes(&planes)
                    .map_err(|err| encode_error(path, format!("H.264 encode failed: {err}")))?,
            );
        }
        annex_b.extend_from_slice(&encoder.flush());
        h264_access_units(&annex_b)
    };
    make_stream(
        path,
        track_index,
        VideoCodec::H264,
        width,
        height,
        Vec::new(),
        units,
        frames,
        fps_num,
        fps_den,
    )
}

fn quality_to_av1_quantizer(quality: VideoQuality) -> usize {
    usize::from(
        (100_u8.saturating_sub(quality.get()))
            .saturating_mul(2)
            .clamp(20, 200),
    )
}

fn cpu_av1(
    path: &Path,
    frames: &[Yuv420Frame],
    rate_control: VideoRateControl,
    track_index: usize,
    fps_num: u32,
    fps_den: u32,
) -> Result<EncodedVideoStream, PipelineError> {
    let (width, height) = (frames[0].width, frames[0].height);
    require_even(path, width, height)?;
    let w = usize::try_from(width).map_err(|_| encode_error(path, "AV1 width overflow"))?;
    let h = usize::try_from(height).map_err(|_| encode_error(path, "AV1 height overflow"))?;
    // rav1e's `select_qi` ignores `quantizer` whenever `bitrate > 0`, and
    // `min_quantizer` defaults to 0, so rate control can reach any quantizer
    // needed to hit the target. Keep a sane base quantizer for either path.
    let (quantizer, bitrate) = match rate_control {
        VideoRateControl::Quality(quality) => (quality_to_av1_quantizer(quality), 0),
        VideoRateControl::Bitrate(bitrate) => (
            quality_to_av1_quantizer(VideoQuality::DEFAULT),
            i32::try_from(bitrate.bps())
                .map_err(|_| encode_error(path, "AV1 bitrate exceeds i32"))?,
        ),
    };
    let mut config = EncoderConfig::with_speed_preset(8);
    config.width = w;
    config.height = h;
    config.bit_depth = 8;
    config.chroma_sampling = ChromaSampling::Cs420;
    config.time_base = Rational::new(u64::from(fps_den), u64::from(fps_num));
    config.quantizer = quantizer;
    config.bitrate = bitrate;
    config.min_key_frame_interval = 12;
    config.max_key_frame_interval = 120;
    config.speed_settings = SpeedSettings::from_preset(8);
    config.low_latency = true;
    let threads = std::thread::available_parallelism()
        .unwrap_or(NonZeroUsize::MIN)
        .get()
        .min(4);
    let ctx = Config::new()
        .with_encoder_config(config)
        .with_threads(threads)
        .new_context::<u8>()
        .map_err(|err| encode_error(path, format!("AV1 encoder init failed: {err}")))?;
    let mut ctx = ctx;
    let mut units = Vec::with_capacity(frames.len());
    for frame in frames {
        let mut av1_frame = ctx.new_frame();
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let [y_plane, u_plane, v_plane] = &mut av1_frame.planes;
        for (plane, source, pw, ph) in [
            (y_plane, frame.y.as_slice(), w, h),
            (u_plane, frame.u.as_slice(), cw, ch),
            (v_plane, frame.v.as_slice(), cw, ch),
        ] {
            for (row, out) in plane.rows_iter_mut().take(ph).enumerate() {
                let start = row.saturating_mul(pw);
                let Some(src) = source.get(start..start.saturating_add(pw)) else {
                    return Err(encode_error(path, "AV1 source plane is too small"));
                };
                let dest = out
                    .get_mut(..pw)
                    .ok_or_else(|| encode_error(path, "AV1 encoder plane is too small"))?;
                dest.copy_from_slice(src);
            }
        }
        ctx.send_frame(av1_frame)
            .map_err(|err| encode_error(path, format!("AV1 send failed: {err:?}")))?;
        loop {
            match ctx.receive_packet() {
                Ok(packet) => {
                    let keyframe = packet.frame_type == FrameType::KEY;
                    units.push((packet.data, keyframe));
                }
                Err(rav1e::EncoderStatus::Encoded) => {}
                Err(rav1e::EncoderStatus::NeedMoreData | rav1e::EncoderStatus::LimitReached) => {
                    break;
                }
                Err(err) => return Err(encode_error(path, format!("AV1 receive failed: {err:?}"))),
            }
        }
    }
    ctx.flush();
    loop {
        match ctx.receive_packet() {
            Ok(packet) => {
                let keyframe = packet.frame_type == FrameType::KEY;
                units.push((packet.data, keyframe));
            }
            Err(rav1e::EncoderStatus::Encoded) => {}
            Err(rav1e::EncoderStatus::NeedMoreData | rav1e::EncoderStatus::LimitReached) => break,
            Err(err) => return Err(encode_error(path, format!("AV1 drain failed: {err:?}"))),
        }
    }
    make_stream(
        path,
        track_index,
        VideoCodec::Av1,
        width,
        height,
        Vec::new(),
        units,
        frames,
        fps_num,
        fps_den,
    )
}

fn cpu_vp9(
    path: &Path,
    frames: &[Yuv420Frame],
    rate_control: VideoRateControl,
    track_index: usize,
    fps_num: u32,
    fps_den: u32,
) -> Result<EncodedVideoStream, PipelineError> {
    let (width, height) = (frames[0].width, frames[0].height);
    require_even(path, width, height)?;
    let mut encoder = Vp9Encoder::default();
    let fps = f64::from(fps_num) / f64::from(fps_den.max(1));
    // rusty_vp9 overrides `qindex` with its per-frame rate controller once
    // `bitrate_bps` is set; the configured qindex remains its starting point.
    let (qindex, bitrate_bps) = match rate_control {
        VideoRateControl::Quality(quality) => (quality_to_vp9_qindex(quality), None),
        VideoRateControl::Bitrate(bitrate) => (
            quality_to_vp9_qindex(VideoQuality::DEFAULT),
            Some(f64::from(bitrate.bps())),
        ),
    };
    let config = Vp9EncoderConfig {
        qindex: Some(qindex),
        bitrate_bps,
        fps: Some(fps),
        speed: Some(6),
        keyframe_interval: Some(60),
        lag: Some(0),
        chain: Some(true),
        ..Vp9EncoderConfig::default()
    };
    encoder
        .configure(&config)
        .map_err(|err| encode_error(path, format!("VP9 encoder config failed: {err}")))?;
    let mut units = Vec::with_capacity(frames.len());
    for frame in frames {
        encoder
            .push_frame(
                [frame.y.as_slice(), frame.u.as_slice(), frame.v.as_slice()],
                [
                    usize::try_from(width).unwrap_or(0),
                    usize::try_from(width.div_ceil(2)).unwrap_or(0),
                    usize::try_from(width.div_ceil(2)).unwrap_or(0),
                ],
                width,
                height,
            )
            .map_err(|err| encode_error(path, format!("VP9 push failed: {err}")))?;
        loop {
            match encoder.next_packet() {
                Ok(packet) => units.push((packet.data, packet.keyframe)),
                Err(Vp9Error::Again | Vp9Error::Eof) => break,
                Err(err) => {
                    return Err(encode_error(
                        path,
                        format!("VP9 packet drain failed: {err}"),
                    ));
                }
            }
        }
    }
    encoder.flush();
    loop {
        match encoder.next_packet() {
            Ok(packet) => units.push((packet.data, packet.keyframe)),
            Err(Vp9Error::Again | Vp9Error::Eof) => break,
            Err(err) => return Err(encode_error(path, format!("VP9 flush failed: {err}"))),
        }
    }
    make_stream(
        path,
        track_index,
        VideoCodec::Vp9,
        width,
        height,
        Vec::new(),
        units,
        frames,
        fps_num,
        fps_den,
    )
}

/// Maps the signalled source colour to the `yuv` converter inputs. An
/// unsupported matrix is rejected where RGB is produced, mirroring the
/// streaming path instead of silently using a different matrix.
fn mjpeg_color(
    path: &Path,
    color: YuvColor,
) -> Result<(SimdYuvRange, YuvStandardMatrix), PipelineError> {
    let matrix = match color.matrix {
        YuvMatrix::Bt601 => YuvStandardMatrix::Bt601,
        YuvMatrix::Bt709 => YuvStandardMatrix::Bt709,
        YuvMatrix::Bt2020 => YuvStandardMatrix::Bt2020,
        YuvMatrix::Unsupported(code) => {
            return Err(encode_error(
                path,
                format!("YUV→RGB conversion for matrix coefficient {code} is not supported"),
            ));
        }
    };
    let range = match color.range {
        YuvRange::Limited => SimdYuvRange::Limited,
        YuvRange::Full => SimdYuvRange::Full,
    };
    Ok((range, matrix))
}

/// Encodes one independent baseline JPEG per frame. `rgb` is a reusable
/// scratch buffer, so a clip of equally sized frames allocates it only once.
fn cpu_mjpeg(
    path: &Path,
    frames: &[Yuv420Frame],
    quality: VideoQuality,
    track_index: usize,
    fps_num: u32,
    fps_den: u32,
) -> Result<EncodedVideoStream, PipelineError> {
    let (width, height) = (frames[0].width, frames[0].height);
    let w = usize::try_from(width).map_err(|_| encode_error(path, "MJPEG width overflow"))?;
    let h = usize::try_from(height).map_err(|_| encode_error(path, "MJPEG height overflow"))?;
    let w16 = u16::try_from(width).map_err(|_| encode_error(path, "MJPEG width exceeds 65535"))?;
    let h16 =
        u16::try_from(height).map_err(|_| encode_error(path, "MJPEG height exceeds 65535"))?;
    if width > MJPEG_MAX_DIMENSION || height > MJPEG_MAX_DIMENSION {
        return Err(encode_error(
            path,
            format!(
                "MJPEG dimensions {width}x{height} exceed the {MJPEG_MAX_DIMENSION}-pixel per-axis cap"
            ),
        ));
    }
    let pixels = u64::from(width).saturating_mul(u64::from(height));
    if pixels > MJPEG_MAX_PIXELS {
        return Err(encode_error(
            path,
            format!("MJPEG frame of {pixels} pixels exceeds the {MJPEG_MAX_PIXELS}-pixel cap"),
        ));
    }
    let cw = w / 2;
    let ch = h / 2;
    let rgb_len = w
        .checked_mul(h)
        .and_then(|count| count.checked_mul(3))
        .ok_or_else(|| encode_error(path, "MJPEG RGB buffer length overflow"))?;
    // Validate every plane before calling the SIMD converter, which assumes
    // correctly sized slices.
    for frame in frames {
        if frame.y.len() < w.saturating_mul(h)
            || frame.u.len() < cw.saturating_mul(ch)
            || frame.v.len() < cw.saturating_mul(ch)
        {
            return Err(encode_error(path, "MJPEG source plane is too small"));
        }
    }
    let mut rgb = Vec::with_capacity(rgb_len);
    let mut units = Vec::with_capacity(frames.len());
    for (index, frame) in frames.iter().enumerate() {
        let image = YuvPlanarImage {
            y_plane: frame.y.as_slice(),
            y_stride: u32::try_from(w)
                .map_err(|_| encode_error(path, "MJPEG luma stride exceeds u32"))?,
            u_plane: frame.u.as_slice(),
            u_stride: u32::try_from(cw)
                .map_err(|_| encode_error(path, "MJPEG chroma stride exceeds u32"))?,
            v_plane: frame.v.as_slice(),
            v_stride: u32::try_from(cw)
                .map_err(|_| encode_error(path, "MJPEG chroma stride exceeds u32"))?,
            width,
            height,
        };
        rgb.resize(rgb_len, 0);
        // Use the signalled source colour instead of a fixed matrix: H.264/HEVC
        // sources routinely signal BT.709, and decoding those planes as BT.601
        // would shift the colours of every MJPEG frame.
        let (range, matrix) = mjpeg_color(path, frame.color)?;
        yuv::yuv420_to_rgb(
            &image,
            &mut rgb,
            u32::try_from(w.saturating_mul(3))
                .map_err(|_| encode_error(path, "MJPEG RGB stride exceeds u32"))?,
            range,
            matrix,
        )
        .map_err(|err| encode_error(path, format!("MJPEG YUV→RGB conversion failed: {err}")))?;
        let mut jpeg = Vec::new();
        let mut encoder = JpegEncoder::new(&mut jpeg, quality.get().max(1));
        encoder.set_sampling_factor(SamplingFactor::F_2_2);
        encoder
            .encode(&rgb, w16, h16, JpegColorType::Rgb)
            .map_err(|err| encode_error(path, format!("MJPEG encode failed: {err}")))?;
        // The decode side caps packets at MJPEG_PACKET_CAP; never emit a frame
        // that fastconv itself would reject.
        if jpeg.len() > MJPEG_PACKET_CAP {
            return Err(encode_error(
                path,
                format!(
                    "MJPEG frame {index} encoded to {} bytes, above the {MJPEG_PACKET_CAP}-byte cap",
                    jpeg.len()
                ),
            ));
        }
        // Every JPEG frame is independently decodable: keyframe = true, and
        // make_stream sets dts = pts for all packets.
        units.push((jpeg, true));
    }
    make_stream(
        path,
        track_index,
        VideoCodec::Mjpeg,
        width,
        height,
        Vec::new(),
        units,
        frames,
        fps_num,
        fps_den,
    )
}

/// Encodes decoded frames on the CPU into canonical container packets.
pub(crate) fn encode_video_cpu(
    path: &Path,
    frames: &[Yuv420Frame],
    mode: VideoEncodeMode,
    track_index: usize,
) -> Result<EncodedVideoStream, PipelineError> {
    let first = frames
        .first()
        .ok_or_else(|| encode_error(path, "no video frames to encode"))?;
    require_even(path, first.width, first.height)?;
    for frame in frames {
        if frame.width != first.width || frame.height != first.height {
            return Err(encode_error(
                path,
                "video frame dimensions change mid-stream",
            ));
        }
    }
    let duration = median_duration(
        &frames
            .iter()
            .map(|frame| ContainerPacket {
                track: track_index,
                pts: frame.pts_us,
                dts: frame.pts_us,
                duration: frame.duration_us,
                keyframe: false,
                data: Vec::new(),
            })
            .collect::<Vec<_>>(),
    );
    let fps_num = US_TIMESCALE;
    let fps_den = u32::try_from(duration.max(1)).unwrap_or(US_TIMESCALE / 30);
    match mode {
        VideoEncodeMode::CpuH264 { rate_control } => {
            cpu_h264(path, frames, rate_control, track_index, fps_num, fps_den)
        }
        VideoEncodeMode::CpuAv1 { rate_control } => {
            cpu_av1(path, frames, rate_control, track_index, fps_num, fps_den)
        }
        VideoEncodeMode::CpuVp9 { rate_control } => {
            cpu_vp9(path, frames, rate_control, track_index, fps_num, fps_den)
        }
        VideoEncodeMode::CpuMjpeg { quality } => {
            cpu_mjpeg(path, frames, quality, track_index, fps_num, fps_den)
        }
        VideoEncodeMode::CpuGif { .. } | VideoEncodeMode::CpuWebp { .. } => Err(encode_error(
            path,
            "animated GIF/WebP frames must be encoded by the animation path, not the frame encoder",
        )),
        VideoEncodeMode::NvencH264 { .. } | VideoEncodeMode::NvencHevc { .. } => Err(encode_error(
            path,
            "NVENC modes must be encoded on the NVENC backend",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn synthetic_frames(width: u32, height: u32, count: usize) -> Vec<Yuv420Frame> {
        (0..count)
            .map(|index| {
                let y = (0..width.saturating_mul(height) as usize)
                    .map(|i| u8::try_from((i.saturating_add(index * 7)) % 251).unwrap_or(0))
                    .collect();
                let chroma = usize::try_from(width.div_ceil(2))
                    .unwrap_or(0)
                    .saturating_mul(usize::try_from(height.div_ceil(2)).unwrap_or(0));
                Yuv420Frame {
                    width,
                    height,
                    y,
                    u: vec![128_u8; chroma],
                    v: vec![120_u8; chroma],
                    pts_us: i64::try_from(index).unwrap_or(0) * 40_000,
                    duration_us: 40_000,
                    color: YuvColor::default(),
                }
            })
            .collect()
    }

    // Non-empty plane assertions document decoded-frame validity directly.
    #[allow(clippy::assert_is_empty)]
    fn assert_roundtrip(
        mode: VideoEncodeMode,
        width: u32,
        height: u32,
    ) -> Result<(), PipelineError> {
        let path = Path::new("video-codec-test");
        let frames = synthetic_frames(width, height, 4);
        let stream = encode_video_cpu(path, &frames, mode, 0)?;
        let decoded = decode_video_track(path, &stream.track, &stream.packets)?;
        assert_eq!(decoded.len(), 4);
        for frame in &decoded {
            assert_eq!((frame.width, frame.height), (width, height));
            assert!(!frame.y.is_empty());
            assert_eq!(
                frame.u.len(),
                usize::try_from(width / 2).unwrap_or(0) * usize::try_from(height / 2).unwrap_or(0)
            );
        }
        Ok(())
    }

    #[test]
    fn roundtrips_h264() -> Result<(), PipelineError> {
        assert_roundtrip(
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            64,
            48,
        )
    }

    #[test]
    fn h264_parallel_gops_preserve_count_cadence_and_sps_pps() -> Result<(), PipelineError> {
        let path = Path::new("h264-parallel-gop");
        let frames = synthetic_frames(32, 24, 70);
        // 15 fps -> a one-second, 15-frame GOP, so 70 frames span five GOPs.
        let stream = cpu_h264(path, &frames, VideoRateControl::DEFAULT, 0, 15, 1)?;
        assert_eq!(
            stream.packets.len(),
            frames.len(),
            "one packet per source frame"
        );
        assert!(
            !stream.track.config.is_empty(),
            "the first parallel GOP must contribute an avcC record"
        );
        for (index, packet) in stream.packets.iter().enumerate() {
            assert_eq!(
                packet.keyframe,
                index % 15 == 0,
                "keyframe cadence broke at frame {index}"
            );
            assert!(!packet.data.is_empty(), "empty access unit at {index}");
        }
        let decoded = decode_video_track(path, &stream.track, &stream.packets)?;
        assert_eq!(
            decoded.len(),
            frames.len(),
            "every independently encoded GOP must decode"
        );
        Ok(())
    }

    #[test]
    fn h264_parallel_gops_with_higher_fps() -> Result<(), PipelineError> {
        let path = Path::new("h264-parallel-gop-60");
        let frames = synthetic_frames(32, 24, 130);
        let stream = cpu_h264(path, &frames, VideoRateControl::DEFAULT, 0, 60, 1)?;
        assert_eq!(stream.packets.len(), frames.len());
        for (index, packet) in stream.packets.iter().enumerate() {
            assert_eq!(
                packet.keyframe,
                index % 60 == 0,
                "keyframe cadence broke at frame {index}"
            );
        }
        Ok(())
    }

    #[test]
    fn roundtrips_vp9() -> Result<(), PipelineError> {
        assert_roundtrip(
            VideoEncodeMode::CpuVp9 {
                rate_control: VideoRateControl::DEFAULT,
            },
            64,
            48,
        )
    }

    #[test]
    fn roundtrips_av1() -> Result<(), PipelineError> {
        assert_roundtrip(
            VideoEncodeMode::CpuAv1 {
                rate_control: VideoRateControl::DEFAULT,
            },
            64,
            48,
        )
    }

    #[test]
    fn roundtrips_mjpeg() -> Result<(), PipelineError> {
        let path = Path::new("mjpeg-roundtrip");
        let frames = synthetic_frames(64, 48, 4);
        let stream = encode_video_cpu(
            path,
            &frames,
            VideoEncodeMode::CpuMjpeg {
                quality: VideoQuality::DEFAULT,
            },
            0,
        )?;
        assert_eq!(stream.track.codec, VideoCodec::Mjpeg);
        assert_eq!(stream.packets.len(), frames.len());
        for packet in &stream.packets {
            assert!(packet.keyframe, "every MJPEG frame must be a keyframe");
            assert_eq!(packet.dts, packet.pts, "MJPEG has no reorder delay");
            assert!(packet.data.starts_with(&[0xFF, 0xD8]));
            assert!(packet.data.ends_with(&[0xFF, 0xD9]));
        }
        let decoded = decode_video_track(path, &stream.track, &stream.packets)?;
        assert_eq!(decoded.len(), frames.len());
        for frame in &decoded {
            assert_eq!((frame.width, frame.height), (64, 48));
        }
        Ok(())
    }

    /// Removes every DHT marker segment, keeping the rest of the JPEG intact.
    fn strip_dht_segments(jpeg: &[u8]) -> Vec<u8> {
        let mut out = jpeg.get(..2).unwrap_or(&[]).to_vec(); // SOI
        let mut idx = 2_usize;
        while idx.saturating_add(4) <= jpeg.len() {
            if jpeg[idx] != 0xFF {
                break;
            }
            let marker = jpeg[idx.saturating_add(1)];
            let len = usize::from(u16::from_be_bytes([
                jpeg[idx.saturating_add(2)],
                jpeg[idx.saturating_add(3)],
            ]));
            let end = idx.saturating_add(2).saturating_add(len).min(jpeg.len());
            if marker == 0xC4 {
                idx = end;
                continue;
            }
            if marker == 0xDA {
                out.extend_from_slice(&jpeg[idx..]);
                return out;
            }
            out.extend_from_slice(&jpeg[idx..end]);
            idx = end;
        }
        out.extend_from_slice(jpeg.get(idx..).unwrap_or(&[]));
        out
    }

    #[test]
    fn dht_less_mjpeg_frames_decode_cleanly() -> Result<(), PipelineError> {
        let path = Path::new("mjpeg-dht-less");
        let frames = synthetic_frames(64, 48, 3);
        let stream = cpu_mjpeg(path, &frames, VideoQuality::DEFAULT, 0, 25, 1)?;
        let stripped = strip_dht_segments(&stream.packets[0].data);
        assert_ne!(stripped.len(), stream.packets[0].data.len());
        assert!(!stripped.windows(2).any(|pair| pair == [0xFF, 0xC4]));

        let mut rgb = Vec::new();
        let (width, height) = mjpeg_decode_rgb(path, &stripped, &mut rgb)?;
        assert_eq!((width, height), (64, 48));
        assert_eq!(rgb.len(), 64 * 48 * 3);

        let mut packets = stream.packets.clone();
        for packet in &mut packets {
            packet.data = strip_dht_segments(&packet.data);
        }
        let decoded = decode_video_track(path, &stream.track, &packets)?;
        assert_eq!(decoded.len(), frames.len());
        Ok(())
    }

    #[test]
    fn mjpeg_streaming_decode_packs_nv12() -> Result<(), PipelineError> {
        let path = Path::new("mjpeg-streaming");
        let frames = synthetic_frames(64, 48, 3);
        let stream = cpu_mjpeg(path, &frames, VideoQuality::DEFAULT, 0, 25, 1)?;
        let mut seen = 0_usize;
        decode_video_track_streaming(path, &stream.track, &stream.packets, |info, pack| {
            assert_eq!((info.width, info.height), (64, 48));
            let pitch = usize::try_from(info.width)
                .map_err(|_| decode_error(path, "frame width overflows usize"))?;
            let length = nv12_buffer_len(info.width, info.height, pitch)?;
            let mut nv12 = vec![0_u8; length];
            pack(&mut nv12, pitch)?;
            assert!(nv12.iter().any(|&byte| byte != 0));
            seen = seen.saturating_add(1);
            Ok(())
        })?;
        assert_eq!(seen, frames.len());
        Ok(())
    }

    #[test]
    fn malformed_and_oversized_mjpeg_packets_are_rejected() -> Result<(), PipelineError> {
        let path = Path::new("mjpeg-malformed");
        let mut rgb = Vec::new();
        // Wrong magic, truncated SOI, and truncated marker length.
        for packet in [
            &b"not a jpeg"[..],
            &[0xFF, 0xD8][..],
            &[0xFF, 0xD8, 0xFF, 0xC0, 0x00][..],
            &[0xFF, 0xD8, 0xFF, 0xC0, 0xFF, 0xFF, 0x01][..],
        ] {
            assert!(
                mjpeg_decode_rgb(path, packet, &mut rgb).is_err(),
                "malformed packet must be rejected"
            );
        }

        // A valid frame truncated mid-scan must not be decoded partially.
        let stream = cpu_mjpeg(
            path,
            &synthetic_frames(64, 48, 1),
            VideoQuality::DEFAULT,
            0,
            25,
            1,
        )?;
        let jpeg = &stream.packets[0].data;
        let half = jpeg.len() / 2;
        assert!(mjpeg_decode_rgb(path, &jpeg[..half], &mut rgb).is_err());

        // Oversized packets are rejected before any header parsing.
        let oversized = vec![0_u8; MJPEG_PACKET_CAP.saturating_add(1)];
        assert!(mjpeg_decode_rgb(path, &oversized, &mut rgb).is_err());
        Ok(())
    }

    #[test]
    fn mjpeg_encode_rejects_oversized_axis() {
        let path = Path::new("mjpeg-oversized-axis");
        // Well under the pixel cap, but the decoder would refuse this width.
        let frames = vec![Yuv420Frame {
            width: 10_000,
            height: 2,
            y: vec![0; 20_000],
            u: vec![128; 5_000],
            v: vec![128; 5_000],
            pts_us: 0,
            duration_us: 40_000,
            color: YuvColor::default(),
        }];
        assert!(matches!(
            cpu_mjpeg(path, &frames, VideoQuality::DEFAULT, 0, 25, 1),
            Err(PipelineError::EncodeFailed { .. })
        ));
    }

    #[test]
    fn mjpeg_encode_uses_signalled_source_colour() -> Result<(), PipelineError> {
        let path = Path::new("mjpeg-colour");
        let frames = synthetic_frames(64, 48, 1);
        let default = cpu_mjpeg(path, &frames, VideoQuality::DEFAULT, 0, 25, 1)?;
        let mut bt709 = frames.clone();
        bt709[0].color = YuvColor::specified(YuvMatrix::Bt709, YuvRange::Limited);
        let signalled = cpu_mjpeg(path, &bt709, VideoQuality::DEFAULT, 0, 25, 1)?;
        assert_ne!(
            default.packets[0].data, signalled.packets[0].data,
            "BT.709 planes must not be encoded as BT.601"
        );
        let mut unsupported = frames.clone();
        unsupported[0].color = YuvColor::specified(YuvMatrix::Unsupported(10), YuvRange::Limited);
        assert!(matches!(
            cpu_mjpeg(path, &unsupported, VideoQuality::DEFAULT, 0, 25, 1),
            Err(PipelineError::EncodeFailed { .. })
        ));
        Ok(())
    }

    #[test]
    fn mjpeg_encode_rejects_oversized_jpeg_packet() -> Result<(), PipelineError> {
        let path = Path::new("mjpeg-oversized-jpeg");
        // 4096x4096 is the largest frame the pixel cap allows; high-entropy
        // content at quality 100 encodes above the 16 MiB packet cap.
        let mut state = 1_u32;
        let mut noise = |len: usize| -> Vec<u8> {
            (0..len)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    u8::try_from(state >> 24).unwrap_or(0)
                })
                .collect()
        };
        let side = usize::try_from(MJPEG_MAX_DIMENSION).unwrap_or(0) / 2;
        let frame = Yuv420Frame {
            width: u32::try_from(side).map_err(|_| encode_error(path, "side width"))?,
            height: u32::try_from(side).map_err(|_| encode_error(path, "side height"))?,
            y: noise(side.saturating_mul(side)),
            u: noise(side.saturating_mul(side) / 4),
            v: noise(side.saturating_mul(side) / 4),
            pts_us: 0,
            duration_us: 40_000,
            color: YuvColor::default(),
        };
        let quality = VideoQuality::new(VideoQuality::MAX)
            .map_err(|err| encode_error(path, format!("quality: {err}")))?;
        assert!(matches!(
            cpu_mjpeg(path, &[frame], quality, 0, 25, 1),
            Err(PipelineError::EncodeFailed { .. })
        ));
        Ok(())
    }

    #[test]
    fn rejects_odd_dimensions() {
        let frames = synthetic_frames(65, 48, 2);
        let result = encode_video_cpu(
            Path::new("odd.mp4"),
            &frames,
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
        );
        assert!(matches!(result, Err(PipelineError::EncodeFailed { .. })));
    }

    #[test]
    fn hevc_reordered_b_frame_uses_its_own_timestamp_and_duration() {
        let packet = |pts: i64, duration: u64| ContainerPacket {
            track: 0,
            pts,
            dts: pts,
            duration,
            keyframe: true,
            data: Vec::new(),
        };
        // Decode order I (0 µs), P (80 ms), B (40 ms): the B picture is
        // emitted while the P access unit is pushed, so packet.pts is wrong.
        let packets = [
            packet(0, 40_000),
            packet(80_000, 40_000),
            packet(40_000, 33_333),
        ];
        let mapping = hevc_pts_to_packet(&packets);
        let mut frame = rusty_h265::Frame {
            picture: std::sync::Arc::new(rusty_h265::Picture::new(16, 16, 1, 8, 8)),
            poc: 2,
            pts: Some(40_000),
            width: 16,
            height: 16,
        };
        let pts = hevc_picture_pts(&frame, 80_000);
        assert_eq!(pts, 40_000);
        assert_eq!(
            hevc_picture_duration(pts, &packets, &mapping, 40_000),
            33_333
        );
        // Only a picture without its own PTS falls back, and the same mapping
        // supplies the duration during flush.
        frame.pts = None;
        let flushed = hevc_picture_pts(&frame, 120_000);
        assert_eq!(flushed, 120_000);
        assert_eq!(
            hevc_picture_duration(flushed, &packets, &mapping, 40_000),
            40_000
        );
    }

    #[cfg(not(feature = "nvenc"))]
    #[test]
    fn nvenc_video_fails_explicitly_without_feature() {
        use fastconv_domain::GpuBackend;
        let track = ContainerTrack {
            codec: VideoCodec::H264,
            width: Some(256),
            height: Some(144),
            channels: None,
            sample_rate: None,
            config: Vec::new(),
            timescale: US_TIMESCALE,
            presentation_edit: None,
            pcm_format: None,
        };
        for mode in [
            VideoEncodeMode::NvencH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            VideoEncodeMode::NvencHevc {
                rate_control: VideoRateControl::DEFAULT,
            },
        ] {
            let result = crate::gpu::nvenc::encode_video_nvenc_streaming(
                Path::new("nvenc.mp4"),
                &track,
                &[],
                mode,
                0,
                || Ok(()),
            );
            assert!(matches!(
                result,
                Err(PipelineError::GpuBackendFailed {
                    backend: GpuBackend::Nvenc,
                    ..
                })
            ));
        }
    }

    #[cfg(feature = "nvenc")]
    fn assert_nvenc_quality_scales_bitrate(
        path: &Path,
        source: &EncodedVideoStream,
        codec: VideoCodec,
    ) -> Result<(), PipelineError> {
        let low = VideoQuality::new(40).unwrap_or(VideoQuality::DEFAULT);
        let high = VideoQuality::new(90).unwrap_or(VideoQuality::DEFAULT);
        let make_mode = |quality| match codec {
            VideoCodec::H264 => Ok(VideoEncodeMode::NvencH264 {
                rate_control: VideoRateControl::Quality(quality),
            }),
            VideoCodec::Hevc => Ok(VideoEncodeMode::NvencHevc {
                rate_control: VideoRateControl::Quality(quality),
            }),
            _ => Err(PipelineError::WorkerPanicked),
        };
        let encoded_bytes = |mode| -> Result<u64, PipelineError> {
            let (stream, _, _) = crate::gpu::nvenc::encode_video_nvenc_streaming(
                path,
                &source.track,
                &source.packets,
                mode,
                0,
                || Ok(()),
            )?;
            Ok(stream
                .packets
                .iter()
                .map(|packet| u64::try_from(packet.data.len()).unwrap_or(u64::MAX))
                .sum())
        };
        let low_bytes = encoded_bytes(make_mode(low)?)?;
        let high_bytes = encoded_bytes(make_mode(high)?)?;
        assert!(
            high_bytes > low_bytes,
            "{codec}: quality 90 ({high_bytes} bytes) must exceed quality 40 ({low_bytes} bytes)"
        );
        Ok(())
    }

    #[cfg(feature = "nvenc")]
    fn assert_nvenc_bitrate_roundtrip(
        path: &Path,
        source: &EncodedVideoStream,
        codec: VideoCodec,
        frame_count: usize,
    ) -> Result<(), PipelineError> {
        let low = fastconv_domain::VideoBitrate::new(200_000)
            .map_err(|err| encode_error(path, err.to_string()))?;
        let high = fastconv_domain::VideoBitrate::new(4_000_000)
            .map_err(|err| encode_error(path, err.to_string()))?;
        let make_mode = |bitrate| match codec {
            VideoCodec::H264 => Ok(VideoEncodeMode::NvencH264 {
                rate_control: VideoRateControl::Bitrate(bitrate),
            }),
            VideoCodec::Hevc => Ok(VideoEncodeMode::NvencHevc {
                rate_control: VideoRateControl::Bitrate(bitrate),
            }),
            _ => Err(PipelineError::WorkerPanicked),
        };
        let encoded = |bitrate| -> Result<(u64, usize), PipelineError> {
            let (stream, frames, _) = crate::gpu::nvenc::encode_video_nvenc_streaming(
                path,
                &source.track,
                &source.packets,
                make_mode(bitrate)?,
                0,
                || Ok(()),
            )?;
            let decoded = decode_video_track(path, &stream.track, &stream.packets)?;
            assert_eq!(decoded.len(), frame_count, "{codec} decoded frame count");
            assert_eq!(
                usize::try_from(frames).unwrap_or(0),
                frame_count,
                "{codec} encoded frame count"
            );
            let bytes = stream
                .packets
                .iter()
                .map(|packet| u64::try_from(packet.data.len()).unwrap_or(u64::MAX))
                .sum();
            Ok((bytes, stream.packets.len()))
        };
        let (low_bytes, low_packets) = encoded(low)?;
        let (high_bytes, high_packets) = encoded(high)?;
        assert!(
            low_packets > 0 && high_packets > 0,
            "{codec}: bitrate modes produced no packets"
        );
        assert!(
            high_bytes > low_bytes,
            "{codec}: 4M bitrate ({high_bytes} bytes) must exceed 200k ({low_bytes} bytes)"
        );
        Ok(())
    }

    #[cfg(feature = "nvenc")]
    #[test]
    fn nvenc_bitrate_roundtrips_h264_and_hevc() -> Result<(), PipelineError> {
        if cudarc::driver::CudaContext::new(0).is_err() {
            return Ok(());
        }
        let path = Path::new("nvenc-bitrate.mp4");
        let frames = synthetic_frames(256, 144, 24);
        let frame_count = frames.len();
        let source = encode_video_cpu(
            path,
            &frames,
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
        )?;
        for codec in [VideoCodec::H264, VideoCodec::Hevc] {
            assert_nvenc_bitrate_roundtrip(path, &source, codec, frame_count)?;
        }
        Ok(())
    }

    #[cfg(feature = "nvenc")]
    fn assert_forced_nvenc_b_frames_roundtrip(
        path: &Path,
        source: &EncodedVideoStream,
        mode: VideoEncodeMode,
        codec: VideoCodec,
        frame_count: usize,
    ) -> Result<(), PipelineError> {
        if !crate::gpu::nvenc::codec_supports_b_frames(codec)? {
            return Ok(());
        }
        // With `disableBadapt = 1`, NVENC guarantees B-frame placement on
        // B-frame-capable GPUs, so decode-order PTS inversions and display-order
        // decode reordering are deterministically verified.
        let (forced_stream, forced_count, _) =
            crate::gpu::nvenc::encode_video_nvenc_streaming_forced_b_frames(
                path,
                &source.track,
                &source.packets,
                mode,
                0,
                || Ok(()),
            )?;
        assert_eq!(forced_count, u64::try_from(frame_count).unwrap_or(0));
        assert_eq!(forced_stream.packets.len(), frame_count);
        let mut forced_max_pts: Option<i64> = None;
        let mut forced_inversion = false;
        for (index, packet) in forced_stream.packets.iter().enumerate() {
            assert!(packet.dts <= packet.pts);
            if index > 0 {
                assert!(packet.dts > forced_stream.packets[index - 1].dts);
            }
            if let Some(prev_max) = forced_max_pts
                && packet.pts < prev_max
            {
                forced_inversion = true;
            }
            forced_max_pts = Some(forced_max_pts.map_or(packet.pts, |m| m.max(packet.pts)));
        }
        assert!(
            forced_inversion,
            "{codec}: forced B-frame mode (`disableBadapt = 1`) must emit PTS inversions in decode order"
        );
        let forced_decoded =
            decode_video_track(path, &forced_stream.track, &forced_stream.packets)?;
        assert_eq!(forced_decoded.len(), frame_count);
        for (index, frame) in forced_decoded.iter().enumerate() {
            if index > 0 {
                assert!(
                    frame.pts_us > forced_decoded[index - 1].pts_us,
                    "{codec}: forced B-frame decoded frames must be in display order"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "nvenc")]
    #[test]
    fn roundtrips_nvenc_h264_and_hevc() -> Result<(), PipelineError> {
        // The encode itself probes CUDA device 0. Skip only that expected
        // no-device condition; a real device must still surface encode errors.
        if cudarc::driver::CudaContext::new(0).is_err() {
            return Ok(());
        }
        let path = Path::new("nvenc-video.mp4");
        // 24 frames give NVENC enough lookahead and GOP depth to emit real
        // B-frame reordering rather than a single IPPP anchor.
        let frames = synthetic_frames(256, 144, 24);
        let source = encode_video_cpu(
            path,
            &frames,
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
        )?;
        for (mode, codec) in [
            (
                VideoEncodeMode::NvencH264 {
                    rate_control: VideoRateControl::DEFAULT,
                },
                VideoCodec::H264,
            ),
            (
                VideoEncodeMode::NvencHevc {
                    rate_control: VideoRateControl::DEFAULT,
                },
                VideoCodec::Hevc,
            ),
        ] {
            let (stream, frames_encoded, _device) =
                crate::gpu::nvenc::encode_video_nvenc_streaming(
                    path,
                    &source.track,
                    &source.packets,
                    mode,
                    0,
                    || Ok(()),
                )?;
            assert_eq!(stream.track.codec, codec);
            assert_eq!(frames_encoded, u64::try_from(frames.len()).unwrap_or(0));
            assert_eq!(
                stream.packets.len(),
                frames.len(),
                "{codec}: every input frame must produce exactly one packet"
            );
            let mut max_pts_seen: Option<i64> = None;
            let mut has_pts_inversion = false;
            for (index, packet) in stream.packets.iter().enumerate() {
                assert!(
                    packet.dts <= packet.pts,
                    "{codec}: packet {index} has dts {} > pts {}",
                    packet.dts,
                    packet.pts
                );
                if index > 0 {
                    assert!(
                        packet.dts > stream.packets[index - 1].dts,
                        "{codec}: DTS must increase at packet {index}"
                    );
                }
                if let Some(prev_max) = max_pts_seen
                    && packet.pts < prev_max
                {
                    has_pts_inversion = true;
                }
                max_pts_seen = Some(max_pts_seen.map_or(packet.pts, |m| m.max(packet.pts)));
            }
            // Production NVENC leaves adaptive B-frame placement enabled under
            // lookahead, so an inversion is optional here, but DTS may lag PTS
            // only when an actual presentation-order inversion occurred.
            let any_dts_lag = stream.packets.iter().any(|packet| packet.dts < packet.pts);
            assert_eq!(
                any_dts_lag, has_pts_inversion,
                "{codec}: DTS lag must match actual PTS inversions"
            );
            let decoded = decode_video_track(path, &stream.track, &stream.packets)?;
            assert_eq!(decoded.len(), frames.len());
            for (index, frame) in decoded.iter().enumerate() {
                assert_eq!((frame.width, frame.height), (256, 144));
                if index > 0 {
                    assert!(
                        frame.pts_us > decoded[index - 1].pts_us,
                        "{codec}: decoded frames must be in display order"
                    );
                }
            }
            assert_forced_nvenc_b_frames_roundtrip(path, &source, mode, codec, frames.len())?;
            assert_nvenc_quality_scales_bitrate(path, &source, codec)?;
        }
        Ok(())
    }

    #[test]
    fn h264_reorder_preserves_presentation_order_for_open_gop_leading_pictures()
    -> Result<(), PipelineError> {
        // Open-GOP sequence where a sync-marked picture (PTS 200) is followed in
        // decode order by a leading picture (PTS 110) earlier than a previously
        // decoded frame (PTS 120). A keyframe-triggered flush would emit 120
        // before 110; suffix-min-PTS holds 120 until 110 arrives.
        let pkt = |pts: i64, dts: i64, keyframe: bool| ContainerPacket {
            track: 0,
            pts,
            dts,
            duration: 40_000,
            keyframe,
            data: Vec::new(),
        };
        let packets = vec![
            pkt(0, 0, true),
            pkt(120, 40, false),
            pkt(40, 80, false),
            pkt(80, 120, false),
            pkt(200, 160, true),
            pkt(110, 200, false),
            pkt(160, 240, false),
        ];
        let ordered = decode_order(&packets);
        assert!(has_b_frame_reordering(&ordered));
        let future_mins = suffix_min_pts(&ordered);
        let mut reorder = Vec::new();
        let mut emitted = Vec::new();
        for (index, packet) in ordered.iter().enumerate() {
            reorder.push(Yuv420Frame {
                width: 2,
                height: 2,
                y: vec![0; 4],
                u: vec![128; 1],
                v: vec![128; 1],
                pts_us: packet.pts,
                duration_us: packet.duration,
                color: YuvColor::default(),
            });
            let future_min = future_mins.get(index + 1).copied();
            drain_ready_h264_reorder_vec(
                Path::new("open-gop.mkv"),
                &mut reorder,
                future_min,
                &mut emitted,
            )?;
        }
        flush_h264_reorder_vec(&mut reorder, &mut emitted);
        let pts_out: Vec<i64> = emitted.iter().map(|frame| frame.pts_us).collect();
        assert_eq!(pts_out, vec![0, 40, 80, 110, 120, 160, 200]);
        Ok(())
    }

    #[test]
    fn h264_reorder_rejects_unprovable_order_at_dpb_cap() {
        // 17 buffered frames with a later decode-order PTS of 0: the earliest
        // buffered frame (10) is not yet provably in presentation order, so the
        // DPB cap must produce an explicit decode error instead of emitting it.
        let mut reorder: Vec<Yuv420Frame> = (1_i64..=17)
            .map(|index| Yuv420Frame {
                width: 2,
                height: 2,
                y: vec![0; 4],
                u: vec![128; 1],
                v: vec![128; 1],
                pts_us: index * 10,
                duration_us: 10,
                color: YuvColor::default(),
            })
            .collect();
        let mut emitted = Vec::new();
        let result = drain_ready_h264_reorder_vec(
            Path::new("dpb-cap.mkv"),
            &mut reorder,
            Some(0),
            &mut emitted,
        );
        assert!(matches!(result, Err(PipelineError::DecodeFailed { .. })));
        assert_eq!(emitted.len(), 0);
    }

    /// Packs a streamed NV12 frame back into `Yuv420Frame` so the two decode
    /// paths can be compared byte-for-byte, including row padding.
    fn unpack_streamed_frame(
        path: &Path,
        info: DecodedFrameInfo,
        nv12: &[u8],
        pitch: usize,
    ) -> Result<Yuv420Frame, PipelineError> {
        let frame_width = usize::try_from(info.width).unwrap_or(0);
        let frame_height = usize::try_from(info.height).unwrap_or(0);
        let chroma_width = frame_width / 2;
        let chroma_height = frame_height / 2;
        let mut y_plane = vec![0_u8; frame_width.saturating_mul(frame_height)];
        let mut u_plane = vec![0_u8; chroma_width.saturating_mul(chroma_height)];
        let mut v_plane = vec![0_u8; chroma_width.saturating_mul(chroma_height)];
        for row in 0..frame_height {
            let start = row * pitch;
            let src = nv12
                .get(start..start.saturating_add(frame_width))
                .ok_or_else(|| decode_error(path, "streamed Y row out of bounds"))?;
            let y_row = y_plane
                .get_mut(row * frame_width..row * frame_width + frame_width)
                .ok_or_else(|| decode_error(path, "streamed Y destination out of bounds"))?;
            y_row.copy_from_slice(src);
            let padding = nv12
                .get(start.saturating_add(frame_width)..start.saturating_add(pitch))
                .ok_or_else(|| decode_error(path, "streamed Y padding out of bounds"))?;
            assert!(
                padding.iter().all(|&byte| byte == 0),
                "streaming decoder touched row padding"
            );
        }
        for row in 0..chroma_height {
            let start = frame_height * pitch + row * pitch;
            let src = nv12
                .get(start..start.saturating_add(frame_width))
                .ok_or_else(|| decode_error(path, "streamed UV row out of bounds"))?;
            for (sample_index, pair) in src.as_chunks::<2>().0.iter().enumerate() {
                u_plane[row * chroma_width + sample_index] = pair[0];
                v_plane[row * chroma_width + sample_index] = pair[1];
            }
            let padding = nv12
                .get(start.saturating_add(frame_width)..start.saturating_add(pitch))
                .ok_or_else(|| decode_error(path, "streamed UV padding out of bounds"))?;
            assert!(
                padding.iter().all(|&byte| byte == 0),
                "streaming decoder touched chroma padding"
            );
        }
        Ok(Yuv420Frame {
            width: info.width,
            height: info.height,
            y: y_plane,
            u: u_plane,
            v: v_plane,
            pts_us: info.pts_us,
            duration_us: info.duration_us,
            color: info.color,
        })
    }

    #[test]
    fn streaming_nv12_matches_buffered_decode_for_av1_h264_vp9() -> Result<(), PipelineError> {
        let path = Path::new("video-codec-streaming-test");
        let (width, height) = (64_u32, 48_u32);
        let pitch = usize::try_from(width).unwrap_or(0) + 64;
        let frames = synthetic_frames(width, height, 4);
        for mode in [
            VideoEncodeMode::CpuAv1 {
                rate_control: VideoRateControl::DEFAULT,
            },
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            VideoEncodeMode::CpuVp9 {
                rate_control: VideoRateControl::DEFAULT,
            },
        ] {
            let stream = encode_video_cpu(path, &frames, mode, 0)?;
            let buffered = decode_video_track(path, &stream.track, &stream.packets)?;
            let mut streamed = Vec::new();
            decode_video_track_streaming(path, &stream.track, &stream.packets, |info, pack| {
                let length = nv12_buffer_len(info.width, info.height, pitch)?;
                let mut dst = vec![0_u8; length];
                pack(&mut dst, pitch)?;
                streamed.push(unpack_streamed_frame(path, info, &dst, pitch)?);
                Ok(())
            })?;
            // This test compares decoded pixels, not colour metadata: AV1
            // signals colour per frame on the streaming path while the buffered
            // frame carries the track default.
            let without_color = |frames: &[Yuv420Frame]| -> Vec<Yuv420Frame> {
                frames
                    .iter()
                    .cloned()
                    .map(|mut frame| {
                        frame.color = YuvColor::default();
                        frame
                    })
                    .collect()
            };
            assert_eq!(without_color(&streamed), without_color(&buffered));
        }
        Ok(())
    }

    #[test]
    fn streaming_decode_stops_immediately_on_callback_error() -> Result<(), PipelineError> {
        let path = Path::new("video-codec-streaming-error");
        let frames = synthetic_frames(64, 48, 4);
        let stream = encode_video_cpu(
            path,
            &frames,
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
        )?;
        let mut calls = 0_usize;
        let result = decode_video_track_streaming(path, &stream.track, &stream.packets, |_, _| {
            calls = calls.saturating_add(1);
            Err(PipelineError::EncodeFailed {
                path: path.to_path_buf(),
                reason: "stop decoding".to_string(),
            })
        });
        assert!(matches!(result, Err(PipelineError::EncodeFailed { .. })));
        assert_eq!(calls, 1);
        Ok(())
    }

    #[test]
    fn streaming_nv12_rejects_insufficient_pitch_or_buffer() -> Result<(), PipelineError> {
        let path = Path::new("video-codec-streaming-bounds");
        let frames = synthetic_frames(64, 48, 2);
        let stream = encode_video_cpu(
            path,
            &frames,
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
        )?;
        let mut checked = 0_usize;
        decode_video_track_streaming(path, &stream.track, &stream.packets, |info, pack| {
            let width = usize::try_from(info.width).unwrap_or(0);
            let valid_pitch = width + 8;
            let valid_len = nv12_buffer_len(info.width, info.height, valid_pitch)?;

            let mut short_pitch = vec![0_u8; valid_len];
            let invalid = pack(&mut short_pitch, width.saturating_sub(2));
            assert!(matches!(invalid, Err(PipelineError::DecodeFailed { .. })));

            let mut short_buffer = vec![0_u8; valid_len.saturating_sub(1)];
            let invalid = pack(&mut short_buffer, valid_pitch);
            assert!(matches!(invalid, Err(PipelineError::DecodeFailed { .. })));

            let mut valid = vec![0_u8; valid_len];
            assert!(pack(&mut valid, valid_pitch).is_ok());
            checked = checked.saturating_add(1);
            Ok(())
        })?;
        assert!(checked > 0);
        Ok(())
    }

    #[test]
    fn estimates_fps_from_packet_durations() {
        let packet = |duration: u64| ContainerPacket {
            track: 0,
            pts: 0,
            dts: 0,
            duration,
            keyframe: true,
            data: Vec::new(),
        };
        assert_eq!(estimate_video_fps_from_packets(&[packet(16_666)]), 60);
        assert_eq!(estimate_video_fps_from_packets(&[packet(16_667)]), 60);
        assert_eq!(estimate_video_fps_from_packets(&[packet(33_333)]), 30);
        assert_eq!(estimate_video_fps_from_packets(&[packet(33_334)]), 30);
        assert_eq!(estimate_video_fps_from_packets(&[packet(40_000)]), 25);
        assert_eq!(estimate_video_fps_from_packets(&[packet(41_667)]), 24);
        assert_eq!(estimate_video_fps_from_packets(&[]), 30);
        assert_eq!(estimate_video_fps_from_packets(&[packet(1)]), 240);
    }

    fn h264_avcc_with_vui(full_range: bool, matrix: Option<u8>) -> Vec<u8> {
        // Minimal baseline 16x16 SPS + VUI colour fields packed MSB-first.
        let mut bits: Vec<u8> = Vec::new();
        let mut push_bits = |value: u32, count: usize| {
            for shift in (0..count).rev() {
                bits.push(u8::try_from((value >> shift) & 1).unwrap_or(0));
            }
        };
        let push_ue = |bits: &mut Vec<u8>, value: u32| {
            let code = value.saturating_add(1);
            let bit_len = usize::try_from(32 - code.leading_zeros()).unwrap_or(1);
            for _ in 0..bit_len.saturating_sub(1) {
                bits.push(0);
            }
            for shift in (0..bit_len).rev() {
                bits.push(u8::try_from((code >> shift) & 1).unwrap_or(0));
            }
        };
        push_bits(0x42, 8); // profile_idc
        push_bits(0, 8); // constraints
        push_bits(30, 8); // level_idc
        push_ue(&mut bits, 0); // sps_id
        push_ue(&mut bits, 4); // log2_max_frame_num_minus4
        push_ue(&mut bits, 0); // pic_order_cnt_type
        push_ue(&mut bits, 4); // log2_max_pic_order_cnt_lsb_minus4
        push_ue(&mut bits, 1); // max_num_ref_frames
        bits.push(0); // gaps_in_frame_num_value_allowed_flag
        push_ue(&mut bits, 0); // width_mbs_minus1
        push_ue(&mut bits, 0); // height_map_units_minus1
        bits.push(1); // frame_mbs_only_flag
        bits.push(1); // direct_8x8_inference_flag
        bits.push(0); // frame_cropping_flag
        bits.push(1); // vui_parameters_present_flag
        bits.push(0); // aspect_ratio_info_present_flag
        bits.push(0); // overscan_info_present_flag
        bits.push(1); // video_signal_type_present_flag
        for bit in [1_u8, 0, 1] {
            bits.push(bit); // video_format = 5 (unspecified)
        }
        bits.push(u8::from(full_range));
        if let Some(matrix_coeffs) = matrix {
            bits.push(1); // colour_description_present_flag
            for byte in [2_u8, 2, matrix_coeffs] {
                for shift in (0..8).rev() {
                    bits.push((byte >> shift) & 1);
                }
            }
        } else {
            bits.push(0); // colour_description_present_flag
        }
        bits.push(1); // rbsp_stop_one_bit
        let mut sps_rbsp = Vec::new();
        for chunk in bits.chunks(8) {
            let mut byte = 0_u8;
            for (i, &bit) in chunk.iter().enumerate() {
                byte |= bit << (7 - i);
            }
            sps_rbsp.push(byte);
        }
        let mut annex_b = vec![0, 0, 0, 1, 0x67];
        annex_b.extend_from_slice(&sps_rbsp);
        annex_b.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE]);
        extract_avcc_from_annex_b(&annex_b).unwrap_or_default()
    }

    #[test]
    fn source_color_preserves_range_and_flags_unsupported_rgb_matrices() {
        let make_track = |config| ContainerTrack {
            codec: VideoCodec::H264,
            width: Some(16),
            height: Some(16),
            channels: None,
            sample_rate: None,
            config,
            timescale: US_TIMESCALE,
            presentation_edit: None,
            pcm_format: None,
        };

        // Full range without colour_description keeps BT.601 default matrix + Full range.
        let full_only = make_track(h264_avcc_with_vui(true, None));
        assert_eq!(
            source_color(&full_only, &[]),
            YuvColor::specified(YuvMatrix::Bt601, YuvRange::Full)
        );

        // Matrix coefficient 9 stays on BT.2020 non-constant luminance.
        let bt2020_ncl = make_track(h264_avcc_with_vui(false, Some(9)));
        assert_eq!(
            source_color(&bt2020_ncl, &[]),
            YuvColor::specified(YuvMatrix::Bt2020, YuvRange::Limited)
        );

        // Unsupported matrices are carried (with their range) past NV12 delivery;
        // only the RGB conversion rejects them.
        let constant_luminance = make_track(h264_avcc_with_vui(false, Some(10)));
        assert_eq!(
            source_color(&constant_luminance, &[]),
            YuvColor::specified(YuvMatrix::Unsupported(10), YuvRange::Limited)
        );
        let smpte_240m = make_track(h264_avcc_with_vui(true, Some(7)));
        assert_eq!(
            source_color(&smpte_240m, &[]),
            YuvColor::specified(YuvMatrix::Unsupported(7), YuvRange::Full)
        );
    }

    #[test]
    fn av1_color_preserves_range_when_matrix_is_unspecified() {
        use rav1d_safe::src::managed::MatrixCoefficients as Mc;
        for matrix in [Mc::Unspecified, Mc::Reserved] {
            assert_eq!(
                streaming::av1_color(matrix, YuvRange::Full),
                YuvColor::specified(YuvMatrix::Bt601, YuvRange::Full)
            );
            assert_eq!(
                streaming::av1_color(matrix, YuvRange::Limited),
                YuvColor::specified(YuvMatrix::Bt601, YuvRange::Limited)
            );
        }
        assert_eq!(
            streaming::av1_color(Mc::BT709, YuvRange::Full),
            YuvColor::specified(YuvMatrix::Bt709, YuvRange::Full)
        );
    }
}
