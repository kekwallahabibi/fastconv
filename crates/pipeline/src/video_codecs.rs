//! Native video codec backends: pure-Rust decoding of all four supported
//! video codecs into a uniform 8-bit planar 4:2:0 frame, and CPU encoding of
//! H.264, AV1 and VP9 into canonical container packets.
//!
//! Every illegal state is rejected explicitly; there is no silent fallback
//! between codecs or bit depths.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::Path;

use fastconv_domain::{VideoCodec, VideoEncodeMode, VideoQuality};
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

fn h264_frame(frame: H264YuvFrame, pts_us: i64, duration_us: u64) -> Yuv420Frame {
    Yuv420Frame {
        width: u32::try_from(frame.width).unwrap_or(u32::MAX),
        height: u32::try_from(frame.height).unwrap_or(u32::MAX),
        y: frame.y,
        u: frame.u,
        v: frame.v,
        pts_us,
        duration_us,
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
    })
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
    match track.codec {
        VideoCodec::H264 => {
            let mut decoder = H264Decoder::new();
            let mut frames = Vec::with_capacity(ordered.len());
            for packet in ordered {
                let annex_b = avcc_to_annex_b(&track.config, &packet.data, true)
                    .map_err(|reason| decode_error(path, reason))?;
                let frame = decoder
                    .decode(&annex_b)
                    .map_err(|err| decode_error(path, format!("H.264 decode failed: {err}")))?;
                if let Some(frame) = frame {
                    frames.push(h264_frame(frame, packet.pts, packet.duration));
                }
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
        other => Err(decode_error(
            path,
            format!("codec {other} is not a supported video decoder"),
        )),
    }
}

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

fn cpu_h264(
    path: &Path,
    frames: &[Yuv420Frame],
    quality: VideoQuality,
    track_index: usize,
    fps_num: u32,
    fps_den: u32,
) -> Result<EncodedVideoStream, PipelineError> {
    let (width, height) = (frames[0].width, frames[0].height);
    require_even(path, width, height)?;
    let w = usize::try_from(width).map_err(|_| encode_error(path, "H.264 width overflow"))?;
    let h = usize::try_from(height).map_err(|_| encode_error(path, "H.264 height overflow"))?;
    let mut config = H264EncoderConfig::new(w, h);
    config.qp = quality_to_h264_qp(quality);
    config.preset = H264Preset::Fast;
    config.gop_size = 60;
    // Frame rates from packet durations are small; f32 is the encoder's field type.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let framerate = (f64::from(fps_num) / f64::from(fps_den.max(1))) as f32;
    config.framerate = framerate;
    config.bitrate = 0;
    config.lookahead = 0;
    config.scenecut = 0;
    config.mbtree = false;
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
    let units = split_annex_b_access_units(&annex_b, false)
        .into_iter()
        .map(|data| {
            let keyframe = is_h264_keyframe_annex_b(&data);
            (data, keyframe)
        })
        .collect();
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

fn cpu_av1(
    path: &Path,
    frames: &[Yuv420Frame],
    quality: VideoQuality,
    track_index: usize,
    fps_num: u32,
    fps_den: u32,
) -> Result<EncodedVideoStream, PipelineError> {
    let (width, height) = (frames[0].width, frames[0].height);
    require_even(path, width, height)?;
    let w = usize::try_from(width).map_err(|_| encode_error(path, "AV1 width overflow"))?;
    let h = usize::try_from(height).map_err(|_| encode_error(path, "AV1 height overflow"))?;
    let quantizer = usize::from(
        (100_u8.saturating_sub(quality.get()))
            .saturating_mul(2)
            .clamp(20, 200),
    );
    let mut config = EncoderConfig::with_speed_preset(8);
    config.width = w;
    config.height = h;
    config.bit_depth = 8;
    config.chroma_sampling = ChromaSampling::Cs420;
    config.time_base = Rational::new(u64::from(fps_den), u64::from(fps_num));
    config.quantizer = quantizer;
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
    quality: VideoQuality,
    track_index: usize,
    fps_num: u32,
    fps_den: u32,
) -> Result<EncodedVideoStream, PipelineError> {
    let (width, height) = (frames[0].width, frames[0].height);
    require_even(path, width, height)?;
    let mut encoder = Vp9Encoder::default();
    let fps = f64::from(fps_num) / f64::from(fps_den.max(1));
    let config = Vp9EncoderConfig {
        qindex: Some(quality_to_vp9_qindex(quality)),
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
        VideoEncodeMode::CpuH264 { quality } => {
            cpu_h264(path, frames, quality, track_index, fps_num, fps_den)
        }
        VideoEncodeMode::CpuAv1 { quality } => {
            cpu_av1(path, frames, quality, track_index, fps_num, fps_den)
        }
        VideoEncodeMode::CpuVp9 { quality } => {
            cpu_vp9(path, frames, quality, track_index, fps_num, fps_den)
        }
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
                quality: VideoQuality::DEFAULT,
            },
            64,
            48,
        )
    }

    #[test]
    fn roundtrips_vp9() -> Result<(), PipelineError> {
        assert_roundtrip(
            VideoEncodeMode::CpuVp9 {
                quality: VideoQuality::DEFAULT,
            },
            64,
            48,
        )
    }

    #[test]
    fn roundtrips_av1() -> Result<(), PipelineError> {
        assert_roundtrip(
            VideoEncodeMode::CpuAv1 {
                quality: VideoQuality::DEFAULT,
            },
            64,
            48,
        )
    }

    #[test]
    fn rejects_odd_dimensions() {
        let frames = synthetic_frames(65, 48, 2);
        let result = encode_video_cpu(
            Path::new("odd.mp4"),
            &frames,
            VideoEncodeMode::CpuH264 {
                quality: VideoQuality::DEFAULT,
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
        let frames = synthetic_frames(256, 144, 2);
        for mode in [
            VideoEncodeMode::NvencH264 {
                quality: VideoQuality::DEFAULT,
            },
            VideoEncodeMode::NvencHevc {
                quality: VideoQuality::DEFAULT,
            },
        ] {
            let result =
                crate::gpu::nvenc::encode_video_nvenc(Path::new("nvenc.mp4"), &frames, mode, 0);
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
    #[test]
    fn roundtrips_nvenc_h264_and_hevc() -> Result<(), PipelineError> {
        // The encode itself probes CUDA device 0. Skip only that expected
        // no-device condition; a real device must still surface encode errors.
        if cudarc::driver::CudaContext::new(0).is_err() {
            return Ok(());
        }
        let path = Path::new("nvenc-video.mp4");
        let frames = synthetic_frames(256, 144, 4);
        for (mode, codec) in [
            (
                VideoEncodeMode::NvencH264 {
                    quality: VideoQuality::DEFAULT,
                },
                VideoCodec::H264,
            ),
            (
                VideoEncodeMode::NvencHevc {
                    quality: VideoQuality::DEFAULT,
                },
                VideoCodec::Hevc,
            ),
        ] {
            let (stream, _device) = crate::gpu::nvenc::encode_video_nvenc(path, &frames, mode, 0)?;
            assert_eq!(stream.track.codec, codec);
            let decoded = decode_video_track(path, &stream.track, &stream.packets)?;
            assert_eq!(decoded.len(), 4);
            for frame in &decoded {
                assert_eq!((frame.width, frame.height), (256, 144));
            }
        }
        Ok(())
    }
}
