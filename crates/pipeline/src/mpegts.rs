//! Pure-Rust 188-byte MPEG-TS demuxer and muxer with MPEG-2 CRC-32.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use fastconv_domain::VideoCodec;

use crate::PipelineError;
use crate::video_bitstream::{
    annex_b_to_avcc, annex_b_to_hvcc, avcc_to_annex_b, default_opus_head,
    ensure_av1_temporal_delimiter, extract_av1c_from_obu, extract_avcc_from_annex_b,
    extract_hvcc_from_annex_b, h264_dimensions_from_annex_b, hevc_dimensions_from_annex_b,
    hvcc_to_annex_b, is_av1_keyframe, is_h264_keyframe_annex_b, is_hevc_keyframe_annex_b,
    parse_audio_specific_config, unwrap_adts, unwrap_ts_opus, wrap_adts, wrap_ts_opus,
};
use crate::video_container::{
    ContainerMedia, ContainerPacket, ContainerTrack, DemuxedMedia, US_TIMESCALE, rescale_ticks,
};

const TS_SIZE: usize = 188;
const TS_SCAN_CAP: u64 = 2 * 1024 * 1024 * 1024;
const PMT_PID: u16 = 0x1000;
/// Initial PCR/PTS offset: 1.4 seconds expressed in canonical microseconds.
const PCR_OFFSET_US: i64 = 1_400_000;
/// ISO/IEC 13818-1 requires a PCR at least every 100 ms (9000 ticks at 90 kHz).
const PCR_INTERVAL_90K: u64 = 9_000;

type PendingPes = (u16, Vec<u8>, Option<u64>, Option<u64>);

struct PesPayload {
    data: Vec<u8>,
    pts: Option<u64>,
    dts: Option<u64>,
    /// `Some(n)` means n payload bytes still belong to this PES; `None` means
    /// an unbounded PES running until the next PUSI or end of file.
    remaining: Option<usize>,
}

const ADTS_RATES: [u32; 16] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350, 0, 0, 0,
];

fn ts_error(path: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::RemuxFailed {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

/// ISO/IEC 13818-1 CRC-32, polynomial `0x04C1_1DB7`, initial `0xFFFF_FFFF`.
pub(crate) fn crc32_mpeg2(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &byte in bytes {
        crc ^= u32::from(byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04C1_1DB7
            } else {
                crc << 1
            };
        }
    }
    crc
}

fn be_u16(bytes: &[u8]) -> Option<u16> {
    Some(u16::from_be_bytes(bytes.get(..2)?.try_into().ok()?))
}

fn ts_stride(bytes: &[u8]) -> Option<(usize, usize)> {
    if bytes.len() >= 377 && bytes[0] == 0x47 && bytes[188] == 0x47 && bytes[376] == 0x47 {
        return Some((188, 0));
    }
    if bytes.len() >= 389 && bytes[4] == 0x47 && bytes[196] == 0x47 && bytes[388] == 0x47 {
        return Some((192, 4));
    }
    None
}

#[derive(Clone, Copy)]
struct TsPacket<'a> {
    pid: u16,
    pusi: bool,
    payload: &'a [u8],
}

fn parse_ts_packet(packet: &[u8]) -> Option<TsPacket<'_>> {
    if packet.len() < TS_SIZE || packet[0] != 0x47 {
        return None;
    }
    let pusi = packet[1] & 0x40 != 0;
    let pid = (u16::from(packet[1] & 0x1F) << 8) | u16::from(packet[2]);
    let adaptation = (packet[3] >> 4) & 0x03;
    if adaptation & 0x01 == 0 {
        return None;
    }
    let mut idx = 4_usize;
    if adaptation & 0x02 != 0 {
        idx = idx.checked_add(1)?.checked_add(usize::from(packet[4]))?;
        if idx > TS_SIZE {
            return None;
        }
    }
    Some(TsPacket {
        pid,
        pusi,
        payload: &packet[idx..TS_SIZE],
    })
}

fn section_start<'a>(packet: &TsPacket<'a>) -> Option<&'a [u8]> {
    if !packet.pusi {
        return None;
    }
    let pointer = usize::from(*packet.payload.first()?);
    packet.payload.get(1_usize.checked_add(pointer)?..)
}

fn section_length(section: &[u8]) -> Option<usize> {
    let high = usize::from(*section.get(1)? & 0x0F);
    let low = usize::from(*section.get(2)?);
    high.checked_mul(256)?.checked_add(low)
}

fn parse_pat(section: &[u8]) -> Option<u16> {
    if section.first() != Some(&0) || section.get(1)? & 0x80 == 0 {
        return None;
    }
    let table_length = section_length(section)?;
    if table_length < 9 {
        return None;
    }
    let end = 3_usize.checked_add(table_length)?;
    if end > section.len() {
        return None;
    }
    let crc_end = end - 4;
    let mut idx = 8_usize;
    while idx + 4 <= crc_end {
        let program = be_u16(section.get(idx..idx + 2)?)?;
        if program != 0 {
            return be_u16(section.get(idx + 2..idx + 4)?).map(|pid| pid & 0x1FFF);
        }
        idx += 4;
    }
    None
}

fn descriptor(bytes: &[u8], tag: u8) -> Option<&[u8]> {
    let mut idx = 0_usize;
    while idx + 2 <= bytes.len() {
        let found = bytes[idx];
        let len = usize::from(bytes[idx + 1]);
        let start = idx + 2;
        let end = start.checked_add(len)?;
        if end > bytes.len() {
            return None;
        }
        if found == tag {
            return Some(&bytes[start..end]);
        }
        idx = end;
    }
    None
}

#[derive(Clone, Copy)]
struct EsInfo {
    codec: VideoCodec,
    channels: u16,
    sample_rate: u32,
}

fn es_info(stream_type: u8, desc: &[u8]) -> EsInfo {
    match stream_type {
        0x1B => EsInfo {
            codec: VideoCodec::H264,
            channels: 0,
            sample_rate: 0,
        },
        0x24 => EsInfo {
            codec: VideoCodec::Hevc,
            channels: 0,
            sample_rate: 0,
        },
        0x0F => EsInfo {
            codec: VideoCodec::Aac,
            channels: 0,
            sample_rate: 0,
        },
        0x03 | 0x04 => EsInfo {
            codec: VideoCodec::Mp3,
            channels: 0,
            sample_rate: 0,
        },
        0x06 => {
            let opus_channels = descriptor(desc, 0x7F)
                .filter(|ext| ext.len() >= 2 && ext[0] == 0x80)
                .map(|ext| u16::from(ext[1] & 0x0F).max(1));
            if let Some(reg) = descriptor(desc, 0x05)
                && reg.len() >= 4
            {
                if &reg[..4] == b"Opus" {
                    return EsInfo {
                        codec: VideoCodec::Opus,
                        channels: opus_channels.unwrap_or(2),
                        sample_rate: 48_000,
                    };
                }
                if &reg[..4] == b"AV01" {
                    return EsInfo {
                        codec: VideoCodec::Av1,
                        channels: 0,
                        sample_rate: 0,
                    };
                }
            }
            if let Some(channels) = opus_channels {
                return EsInfo {
                    codec: VideoCodec::Opus,
                    channels,
                    sample_rate: 48_000,
                };
            }
            EsInfo {
                codec: VideoCodec::Unknown,
                channels: 0,
                sample_rate: 0,
            }
        }
        _ => EsInfo {
            codec: VideoCodec::Unknown,
            channels: 0,
            sample_rate: 0,
        },
    }
}

fn parse_pmt(section: &[u8]) -> Option<Vec<(u16, EsInfo)>> {
    if section.first() != Some(&2) || section.get(1)? & 0x80 == 0 {
        return None;
    }
    let table_length = section_length(section)?;
    if table_length < 13 {
        return None;
    }
    let end = 3_usize.checked_add(table_length)?;
    if end > section.len() {
        return None;
    }
    let crc_end = end - 4;
    let program_info =
        (usize::from(*section.get(10)? & 0x0F) << 8) | usize::from(*section.get(11)?);
    let mut idx = 12_usize.checked_add(program_info)?;
    if idx > crc_end {
        return None;
    }
    let mut streams = Vec::new();
    while idx + 5 <= crc_end {
        let es_len = (usize::from(section[idx + 3] & 0x0F) << 8) | usize::from(section[idx + 4]);
        let desc_end = idx.checked_add(5)?.checked_add(es_len)?;
        if desc_end > crc_end {
            return None;
        }
        let pid = (u16::from(section[idx + 1] & 0x1F) << 8) | u16::from(section[idx + 2]);
        streams.push((pid, es_info(section[idx], &section[idx + 5..desc_end])));
        idx = desc_end;
    }
    (!streams.is_empty()).then_some(streams)
}

fn decode_timestamp(bytes: &[u8], offset: usize, prefix: u8) -> Option<u64> {
    let b = bytes.get(offset..offset + 5)?;
    if b[0] >> 4 != prefix || b[0] & 1 == 0 || b[2] & 1 == 0 || b[4] & 1 == 0 {
        return None;
    }
    let value = (u64::from((b[0] >> 1) & 0x07) << 30)
        | (u64::from(b[1]) << 22)
        | (u64::from((b[2] >> 1) & 0x7F) << 15)
        | (u64::from(b[3]) << 7)
        | u64::from((b[4] >> 1) & 0x7F);
    Some(value)
}

const PTS_MODULUS: u64 = 1_u64 << 33;
const PTS_HALF_MODULUS: u64 = 1_u64 << 32;

/// Shortest signed distance from a 33-bit reference value to `raw`.
fn timestamp_delta(raw: u64, reference: u64) -> i64 {
    let forward = raw.wrapping_sub(reference % PTS_MODULUS) % PTS_MODULUS;
    if forward >= PTS_HALF_MODULUS {
        i64::try_from(forward).unwrap_or(i64::MAX) - i64::try_from(PTS_MODULUS).unwrap_or(i64::MAX)
    } else {
        i64::try_from(forward).unwrap_or(i64::MAX)
    }
}

fn advance_timestamp(base: u64, delta: i64) -> u64 {
    let value = i128::from(base).saturating_add(i128::from(delta));
    u64::try_from(value.max(0)).unwrap_or(u64::MAX)
}

/// Unwraps a PES timestamp pair onto a running 64-bit 90 kHz clock, keeping
/// the presentation/decode offset intact when the 33-bit counter wraps.
fn unwrap_stamps(
    presentation: Option<u64>,
    decode: Option<u64>,
    last: Option<u64>,
) -> ((Option<u64>, Option<u64>), Option<u64>) {
    match (presentation, decode) {
        (Some(pts), Some(dts)) => {
            let dts = unwrap_timestamp(dts, last);
            let pts = advance_timestamp(dts, timestamp_delta(pts, dts));
            ((Some(pts), Some(dts)), Some(dts))
        }
        (Some(pts), None) => {
            let pts = unwrap_timestamp(pts, last);
            ((Some(pts), None), Some(pts))
        }
        (None, Some(dts)) => {
            let dts = unwrap_timestamp(dts, last);
            ((None, Some(dts)), Some(dts))
        }
        // An unstamped PES must never create or reset the PID clock.
        (None, None) => ((None, None), last),
    }
}

fn unwrap_timestamp(raw: u64, last: Option<u64>) -> u64 {
    match last {
        Some(previous) => advance_timestamp(previous, timestamp_delta(raw, previous)),
        None => raw,
    }
}

/// Parses the PES header on a payload-unit-start packet and reports how many
/// payload bytes are still missing from a length-bounded PES.
fn pes_payload(packet: &TsPacket<'_>) -> Option<PesPayload> {
    let start = packet.payload;
    if start.get(..3)? != [0, 0, 1] {
        return None;
    }
    let flags = *start.get(7)?;
    let header_len = usize::from(*start.get(8)?);
    let header = start.get(9..9 + header_len)?;
    let mut pts = None;
    let mut dts = None;
    if flags & 0x80 != 0 && header.len() >= 5 {
        pts = decode_timestamp(header, 0, if flags & 0x40 != 0 { 3 } else { 2 });
        if flags & 0x40 != 0 && header.len() >= 10 {
            dts = decode_timestamp(header, 5, 1);
        }
    }
    let payload_start = 9_usize.checked_add(header_len)?;
    let declared = usize::from(be_u16(start.get(4..6)?)?);
    if declared == 0 {
        return Some(PesPayload {
            data: start.get(payload_start..)?.to_vec(),
            pts,
            dts,
            remaining: None,
        });
    }
    let total_payload = declared.saturating_add(6);
    let payload_end = total_payload.min(start.len()).max(payload_start);
    let data = start.get(payload_start..payload_end)?.to_vec();
    let remaining = total_payload
        .saturating_sub(payload_start)
        .saturating_sub(data.len());
    Some(PesPayload {
        data,
        pts,
        dts,
        remaining: Some(remaining),
    })
}

fn mp3_frames(data: &[u8]) -> Vec<(Vec<u8>, u32)> {
    let mut out = Vec::new();
    let mut idx = 0_usize;
    while idx + 4 <= data.len() {
        if data[idx] != 0xFF || data[idx + 1] & 0xE0 != 0xE0 {
            idx += 1;
            continue;
        }
        let version = (data[idx + 1] >> 3) & 0x03;
        let layer = (data[idx + 1] >> 1) & 0x03;
        let bitrate_idx = usize::from((data[idx + 2] >> 4) & 0x0F);
        let rate_idx = usize::from((data[idx + 2] >> 2) & 0x03);
        let padding = usize::from((data[idx + 2] >> 1) & 0x01);
        if version == 1 || layer != 1 || bitrate_idx == 0 || bitrate_idx == 15 || rate_idx == 3 {
            idx += 1;
            continue;
        }
        let rates_mpeg1 = [44_100_u32, 48_000, 32_000];
        let rates_mpeg2 = [22_050_u32, 24_000, 16_000];
        let rates_mpeg25 = [11_025_u32, 12_000, 8_000];
        let rates = match version {
            3 => &rates_mpeg1,
            2 => &rates_mpeg2,
            _ => &rates_mpeg25,
        };
        let bitrates_mpeg1 = [
            0_u32, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
        ];
        let bitrates_mpeg2 = [
            0_u32, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160,
        ];
        let bitrates = if version == 3 {
            &bitrates_mpeg1
        } else {
            &bitrates_mpeg2
        };
        let rate = rates[rate_idx];
        let bitrate = bitrates[bitrate_idx];
        let samples = if version == 3 { 1_152_u32 } else { 576 };
        let frame_len = if version == 3 {
            (144_u32 * bitrate * 1000 / rate).saturating_add(u32::try_from(padding).unwrap_or(0))
        } else {
            (72_u32 * bitrate * 1000 / rate).saturating_add(u32::try_from(padding).unwrap_or(0))
        };
        if frame_len == 0 || idx + frame_len as usize > data.len() {
            idx += 1;
            continue;
        }
        out.push((data[idx..idx + frame_len as usize].to_vec(), samples));
        idx += frame_len as usize;
    }
    out
}

/// Demuxes MPEG-TS (188-byte) and M2TS (192-byte) into canonical media.
// The PES reassembly and per-codec conversion stay together: extracting the
// flush closure would require threading six borrowed accumulators through it.
#[allow(clippy::too_many_lines)]
pub(crate) fn demux_ts(path: &Path) -> Result<DemuxedMedia, PipelineError> {
    let metadata = std::fs::metadata(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.len() > TS_SCAN_CAP {
        return Err(ts_error(path, "MPEG-TS file is too large to demux"));
    }
    let bytes = std::fs::read(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let Some((stride, offset)) = ts_stride(&bytes) else {
        return Err(ts_error(path, "no MPEG-TS packet sync cadence"));
    };
    let mut pmt_pid = None;
    let mut pmt_streams = Vec::new();
    let packets: Vec<&[u8]> = bytes
        .chunks(stride)
        .filter(|chunk| chunk.len() == stride)
        .map(|chunk| &chunk[offset..offset + TS_SIZE])
        .collect();
    let mut pmt_section: Vec<u8> = Vec::new();
    let mut pmt_needed: Option<usize> = None;
    for packet in &packets {
        let Some(parsed) = parse_ts_packet(packet) else {
            continue;
        };
        if pmt_pid.is_none()
            && parsed.pid == 0
            && let Some(section) = section_start(&parsed)
            && let Some(pid) = parse_pat(section)
        {
            pmt_pid = Some(pid);
        }
        if let Some(pid) = pmt_pid
            && parsed.pid == pid
        {
            if parsed.pusi {
                pmt_section.clear();
                pmt_needed = None;
                if let Some(section) = section_start(&parsed) {
                    if section.len() < 3 {
                        // The header is split; continuation packets complete it.
                        pmt_section.extend_from_slice(section);
                    } else if let Some(len) = section_length(section)
                        && let Some(needed) = 3_usize.checked_add(len)
                    {
                        // Never strip trailing 0xFF: it may be the low byte of
                        // a legitimate section length rather than TS stuffing.
                        let take = needed.min(section.len());
                        pmt_section.extend_from_slice(&section[..take]);
                        pmt_needed = Some(needed);
                    }
                }
            } else if !pmt_section.is_empty() {
                let mut consumed = 0_usize;
                if pmt_needed.is_none() {
                    let take = 3_usize
                        .saturating_sub(pmt_section.len())
                        .min(parsed.payload.len());
                    pmt_section.extend_from_slice(&parsed.payload[..take]);
                    consumed = take;
                    pmt_needed =
                        section_length(&pmt_section).and_then(|len| 3_usize.checked_add(len));
                }
                if let Some(needed) = pmt_needed {
                    let left = needed.saturating_sub(pmt_section.len());
                    let take = left.min(parsed.payload.len().saturating_sub(consumed));
                    if take > 0 {
                        pmt_section.extend_from_slice(&parsed.payload[consumed..consumed + take]);
                    }
                }
            }
            if let Some(needed) = pmt_needed
                && pmt_section.len() >= needed
            {
                if let Some(found) = parse_pmt(&pmt_section[..needed]) {
                    pmt_streams = found;
                    break;
                }
                pmt_needed = None;
                pmt_section.clear();
            }
        }
    }
    let Some(_pmt_pid) = pmt_pid else {
        return Err(ts_error(path, "MPEG-TS has no PAT"));
    };
    if pmt_streams.is_empty() {
        return Err(ts_error(path, "MPEG-TS has no elementary streams"));
    }
    let mut media = ContainerMedia::default();
    let mut pid_to_track = HashMap::new();
    for (pid, info) in &pmt_streams {
        if info.codec == VideoCodec::Unknown {
            continue;
        }
        let config = match info.codec {
            VideoCodec::Opus => default_opus_head(info.channels.max(1), 0),
            _ => Vec::new(),
        };
        pid_to_track.insert(*pid, media.tracks.len());
        media.tracks.push(ContainerTrack {
            codec: info.codec,
            width: None,
            height: None,
            channels: (info.channels > 0).then_some(info.channels),
            sample_rate: (info.sample_rate > 0).then_some(info.sample_rate),
            config,
            timescale: US_TIMESCALE,
            presentation_edit: None,
        });
    }
    if media.tracks.is_empty() {
        return Err(ts_error(
            path,
            "MPEG-TS has no supported elementary streams",
        ));
    }

    let mut buffers: HashMap<u16, Vec<u8>> = HashMap::new();
    let mut stamps: HashMap<u16, (Option<u64>, Option<u64>)> = HashMap::new();
    // `Some(n)` means a bounded PES still needs n payload bytes; `None` means
    // an unbounded PES that continues until the next PUSI.
    let mut remaining: HashMap<u16, Option<usize>> = HashMap::new();
    let mut clocks: HashMap<u16, u64> = HashMap::new();
    let flush = |pid: u16,
                 data: Vec<u8>,
                 pts: Option<u64>,
                 dts: Option<u64>,
                 media: &mut ContainerMedia,
                 tracks: &HashMap<u16, usize>,
                 path: &Path| {
        let Some(&track) = tracks.get(&pid) else {
            return;
        };
        let codec = media.tracks[track].codec;
        let pts_us = from_90k(pts.unwrap_or(0));
        // Only a PES with both PTS and DTS (flags == 0x3) carries a distinct
        // decode timestamp; otherwise DTS equals PTS.
        let dts_us = dts.map_or(pts_us, from_90k);
        match codec {
            VideoCodec::H264 | VideoCodec::Hevc => {
                let keyframe = if codec == VideoCodec::H264 {
                    is_h264_keyframe_annex_b(&data)
                } else {
                    is_hevc_keyframe_annex_b(&data)
                };
                let dimensions = if codec == VideoCodec::H264 {
                    if media.tracks[track].config.is_empty()
                        && keyframe
                        && let Some(config) = extract_avcc_from_annex_b(&data)
                    {
                        media.tracks[track].config = config;
                    }
                    h264_dimensions_from_annex_b(&data)
                } else {
                    if media.tracks[track].config.is_empty()
                        && keyframe
                        && let Some(config) = extract_hvcc_from_annex_b(&data)
                    {
                        media.tracks[track].config = config;
                    }
                    hevc_dimensions_from_annex_b(&data)
                };
                if let Some((width, height)) = dimensions {
                    if media.tracks[track].width.is_none() {
                        media.tracks[track].width = Some(width);
                    }
                    if media.tracks[track].height.is_none() {
                        media.tracks[track].height = Some(height);
                    }
                }
                let canonical = if codec == VideoCodec::H264 {
                    annex_b_to_avcc(&data)
                } else {
                    annex_b_to_hvcc(&data)
                };
                push(media, track, pts_us, dts_us, canonical, keyframe);
            }
            VideoCodec::Av1 => {
                if media.tracks[track].config.is_empty()
                    && let Some(config) = extract_av1c_from_obu(&data)
                {
                    media.tracks[track].config = config;
                }
                let keyframe = is_av1_keyframe(&data);
                push(media, track, pts_us, dts_us, data, keyframe);
            }
            VideoCodec::Aac => {
                let frames = unwrap_adts(&data);
                if !frames.is_empty()
                    && media.tracks[track].config.is_empty()
                    && let Some((_, index, channel_config)) =
                        parse_audio_specific_config(&frames[0].1)
                    && let Some(&rate) = ADTS_RATES.get(usize::from(index))
                    && rate > 0
                {
                    media.tracks[track].sample_rate = Some(rate);
                    let channels = match channel_config {
                        1..=6 => Some(channel_config),
                        7 => Some(8),
                        _ => None,
                    };
                    if let Some(channels) = channels {
                        media.tracks[track].channels = Some(u16::from(channels));
                    }
                }
                let rate = media.tracks[track].sample_rate.unwrap_or(48_000);
                let step = rescale_ticks(1024, 1, rate);
                let mut frame_pts = pts_us;
                for (frame, asc) in frames {
                    if media.tracks[track].config.is_empty() && !asc.is_empty() {
                        media.tracks[track].config = asc;
                    }
                    push(media, track, frame_pts, frame_pts, frame, true);
                    frame_pts = frame_pts.saturating_add(step);
                }
            }
            VideoCodec::Opus => {
                let mut frame_pts = pts_us;
                for frame in unwrap_ts_opus(&data) {
                    // Opus packets carry 2.5-120 ms of audio; the TOC gives the
                    // coded duration instead of assuming a fixed 20 ms frame.
                    let duration = crate::video_bitstream::opus_packet_samples_48k(&frame)
                        .and_then(|samples| {
                            u64::try_from(rescale_ticks(i64::from(samples), 1, 48_000)).ok()
                        })
                        .filter(|&value| value > 0)
                        .unwrap_or(20_000);
                    push_with_duration(media, track, frame_pts, frame_pts, duration, frame, true);
                    frame_pts = frame_pts.saturating_add(i64::try_from(duration).unwrap_or(20_000));
                }
            }
            VideoCodec::Mp3 => {
                let rate = media.tracks[track].sample_rate.unwrap_or(44_100);
                let mut frame_pts = pts_us;
                for (frame, samples) in mp3_frames(&data) {
                    push(media, track, frame_pts, frame_pts, frame, true);
                    let step = rescale_ticks(i64::from(samples), 1, rate);
                    frame_pts = frame_pts.saturating_add(step);
                }
            }
            _ => {
                let _ = path;
            }
        }
    };
    for packet in &packets {
        let Some(parsed) = parse_ts_packet(packet) else {
            continue;
        };
        if !pid_to_track.contains_key(&parsed.pid) {
            continue;
        }
        if !parsed.pusi {
            // A payload unit start begins a new PES; every other packet of a
            // known PID is a continuation and must be appended, never dropped.
            let left = remaining
                .get(&parsed.pid)
                .copied()
                .flatten()
                .unwrap_or(parsed.payload.len())
                .min(parsed.payload.len());
            if let Some(buffer) = buffers.get_mut(&parsed.pid) {
                buffer.extend_from_slice(&parsed.payload[..left]);
            }
            if let Some(Some(open)) = remaining.get_mut(&parsed.pid) {
                *open = open.saturating_sub(left);
            }
            continue;
        }
        let incomplete = remaining
            .remove(&parsed.pid)
            .flatten()
            .is_some_and(|left| left > 0);
        let (pts, dts) = stamps.remove(&parsed.pid).unwrap_or((None, None));
        if let Some(previous) = buffers.remove(&parsed.pid)
            && !incomplete
        {
            flush(
                parsed.pid,
                previous,
                pts,
                dts,
                &mut media,
                &pid_to_track,
                path,
            );
        }
        let Some(PesPayload {
            data,
            pts,
            dts,
            remaining: left,
        }) = pes_payload(&parsed)
        else {
            continue;
        };
        let ((pts, dts), clock) = unwrap_stamps(pts, dts, clocks.get(&parsed.pid).copied());
        if let Some(clock) = clock {
            clocks.insert(parsed.pid, clock);
        }
        buffers.insert(parsed.pid, data);
        stamps.insert(parsed.pid, (pts, dts));
        remaining.insert(parsed.pid, left);
    }
    let pending: Vec<PendingPes> = buffers
        .into_iter()
        .filter_map(|(pid, data)| {
            // A truncated, length-bounded PES must never reach a codec parser.
            if remaining
                .get(&pid)
                .copied()
                .flatten()
                .is_some_and(|left| left > 0)
            {
                return None;
            }
            let (pts, dts) = stamps.remove(&pid).unwrap_or((None, None));
            Some((pid, data, pts, dts))
        })
        .collect();
    let mut ordered: Vec<PendingPes> = pending;
    ordered.sort_by_key(|(pid, _, _, _)| *pid);
    for (pid, data, pts, dts) in ordered {
        flush(pid, data, pts, dts, &mut media, &pid_to_track, path);
    }
    fix_durations(&mut media);
    Ok(media)
}

fn push(
    media: &mut ContainerMedia,
    track: usize,
    pts: i64,
    dts: i64,
    data: Vec<u8>,
    keyframe: bool,
) {
    push_with_duration(media, track, pts, dts, 1, data, keyframe);
}

fn push_with_duration(
    media: &mut ContainerMedia,
    track: usize,
    pts: i64,
    dts: i64,
    duration: u64,
    data: Vec<u8>,
    keyframe: bool,
) {
    media.packets.push(ContainerPacket {
        track,
        pts,
        dts,
        duration,
        keyframe,
        data,
    });
}

fn fix_durations(media: &mut ContainerMedia) {
    for track in 0..media.tracks.len() {
        let mut indexes: Vec<usize> = media
            .packets
            .iter()
            .enumerate()
            .filter(|(_, p)| p.track == track)
            .map(|(i, _)| i)
            .collect();
        // Durations are presentation-order deltas but the packet vector stays
        // in decode order so B-frame composition offsets are not destroyed.
        indexes.sort_by_key(|&index| media.packets[index].pts);
        for pair in indexes.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let delta = media.packets[b]
                .pts
                .saturating_sub(media.packets[a].pts)
                .max(1);
            media.packets[a].duration = u64::try_from(delta).unwrap_or(1);
        }
        if let Some(&last) = indexes.last() {
            let duration = if indexes.len() == 1 {
                // A lone packet has no presentation-order neighbour; keep the
                // duration its demuxer already derived from the codec TOC.
                i64::try_from(media.packets[last].duration)
                    .unwrap_or(1)
                    .max(1)
            } else {
                indexes.iter().rev().nth(1).map_or(1, |&prev| {
                    media.packets[last]
                        .pts
                        .saturating_sub(media.packets[prev].pts)
                        .max(1)
                })
            };
            media.packets[last].duration = u64::try_from(duration).unwrap_or(1);
        }
    }
}

fn encode_timestamp(prefix: u8, value: u64) -> [u8; 5] {
    [
        (prefix << 4) | (u8::try_from((value >> 30) & 0x07).unwrap_or(0) << 1) | 1,
        u8::try_from((value >> 22) & 0xFF).unwrap_or(0),
        (u8::try_from((value >> 15) & 0x7F).unwrap_or(0) << 1) | 1,
        u8::try_from((value >> 7) & 0xFF).unwrap_or(0),
        (u8::try_from(value & 0x7F).unwrap_or(0) << 1) | 1,
    ]
}

/// Converts a 90 kHz timestamp into rounded canonical microseconds.
fn from_90k(ticks_90k: u64) -> i64 {
    let micros = i128::from(ticks_90k)
        .saturating_mul(1_000_000)
        .saturating_add(45_000)
        / 90_000;
    i64::try_from(micros).unwrap_or(i64::MAX)
}

/// Converts canonical microseconds into rounded 90 kHz ticks.
fn to_90k(us: i64) -> u64 {
    if us <= 0 {
        return 0;
    }
    let ticks = i128::from(us)
        .saturating_mul(90_000)
        .saturating_add(500_000)
        / 1_000_000;
    u64::try_from(ticks).unwrap_or(u64::MAX)
}

/// Splits a PSI section across TS packets without ever truncating it.
fn psi_packets(pid: u16, table: &[u8], cc: &mut u8) -> Result<Vec<[u8; TS_SIZE]>, String> {
    if table.len() > 1021 {
        return Err(format!("PSI section of {} bytes exceeds 1021", table.len()));
    }
    let mut out = Vec::new();
    let mut pos = 0_usize;
    let mut first = true;
    while pos < table.len() {
        let mut packet = [0xFF_u8; TS_SIZE];
        packet[0] = 0x47;
        packet[1] = (if first { 0x40 } else { 0 }) | (u8::try_from(pid >> 8).unwrap_or(0) & 0x1F);
        packet[2] = u8::try_from(pid & 0xFF).unwrap_or(0);
        packet[3] = 0x10 | (*cc & 0x0F);
        *cc = cc.wrapping_add(1) & 0x0F;
        let idx = if first {
            // A PUSI PSI packet starts with a one-byte pointer_field.
            packet[4] = 0;
            5
        } else {
            4
        };
        let take = (table.len() - pos).min(TS_SIZE - idx);
        packet[idx..idx + take].copy_from_slice(&table[pos..pos + take]);
        pos += take;
        first = false;
        out.push(packet);
    }
    Ok(out)
}

fn build_pat() -> Vec<u8> {
    let mut section = vec![
        0x00,
        0x80,
        0x0D,
        0x00,
        0x01,
        0xC1,
        0x00,
        0x00,
        0x00,
        0x01,
        0xE0 | (u8::try_from(PMT_PID >> 8).unwrap_or(0) & 0x1F),
        u8::try_from(PMT_PID & 0xFF).unwrap_or(0),
    ];
    let crc = crc32_mpeg2(&section);
    section.extend_from_slice(&crc.to_be_bytes());
    section
}

fn build_pmt(media: &ContainerMedia, pids: &[u16], pcr_pid: u16) -> Vec<u8> {
    let mut es = Vec::new();
    for (track, &pid) in media.tracks.iter().zip(pids) {
        let (stream_type, mut descriptors) = match track.codec {
            VideoCodec::H264 => (0x1B_u8, Vec::new()),
            VideoCodec::Hevc => (0x24, Vec::new()),
            VideoCodec::Av1 => {
                let mut desc = vec![0x05, 0x04];
                desc.extend_from_slice(b"AV01");
                (0x06, desc)
            }
            VideoCodec::Aac => (0x0F, Vec::new()),
            VideoCodec::Mp3 => (0x03, Vec::new()),
            VideoCodec::Opus => {
                let mut desc = vec![0x05, 0x04];
                desc.extend_from_slice(b"Opus");
                desc.extend_from_slice(&[
                    0x7F,
                    0x02,
                    0x80,
                    u8::try_from(track.channels.unwrap_or(2).clamp(1, 15)).unwrap_or(2),
                ]);
                (0x06, desc)
            }
            _ => (0x06, Vec::new()),
        };
        let desc_len = u16::try_from(descriptors.len()).unwrap_or(u16::MAX);
        let mut entry = vec![
            stream_type,
            0xE0 | (u8::try_from(pid >> 8).unwrap_or(0) & 0x1F),
            u8::try_from(pid & 0xFF).unwrap_or(0),
            0xF0 | u8::try_from((desc_len >> 8) & 0x0F).unwrap_or(0),
            u8::try_from(desc_len & 0xFF).unwrap_or(0),
        ];
        entry.append(&mut descriptors);
        es.extend_from_slice(&entry);
    }
    let section_len = es.len() + 9 + 4;
    let mut section = vec![
        0x02,
        0x80 | u8::try_from((section_len >> 8) & 0x0F).unwrap_or(0),
        u8::try_from(section_len & 0xFF).unwrap_or(0),
        0x00,
        0x01,
        0xC1,
        0x00,
        0x00,
        0xE0 | (u8::try_from(pcr_pid >> 8).unwrap_or(0) & 0x1F),
        u8::try_from(pcr_pid & 0xFF).unwrap_or(0),
        0xF0,
        0x00,
    ];
    section.extend_from_slice(&es);
    let crc = crc32_mpeg2(&section);
    section.extend_from_slice(&crc.to_be_bytes());
    section
}

#[derive(Clone, Copy)]
struct Adaptation {
    flags: u8,
    pcr_base: Option<u64>,
    stuffing: usize,
}

fn ts_packet(
    out: &mut Vec<u8>,
    pid: u16,
    pusi: bool,
    payload: &[u8],
    adaptation: Option<Adaptation>,
    cc: &mut u8,
) {
    let mut packet = [0xFF_u8; TS_SIZE];
    packet[0] = 0x47;
    packet[1] = (if pusi { 0x40 } else { 0x00 }) | (u8::try_from(pid >> 8).unwrap_or(0) & 0x1F);
    packet[2] = u8::try_from(pid & 0xFF).unwrap_or(0);
    if payload.is_empty() {
        // Adaptation-field-only packets do not carry a payload: re-use the
        // counter of the preceding payload packet so the following one can
        // continue with the same value it would have had without the PCR.
        packet[3] = cc.wrapping_sub(1) & 0x0F;
    } else {
        packet[3] = *cc & 0x0F;
        *cc = cc.wrapping_add(1) & 0x0F;
    }
    // A short payload always needs an adaptation field to fill the 188-byte
    // packet; build it before writing the payload.
    let adaptation = adaptation.or_else(|| {
        (payload.len() < TS_SIZE - 4).then(|| Adaptation {
            flags: 0,
            pcr_base: None,
            stuffing: TS_SIZE - 5 - payload.len(),
        })
    });
    let mut idx = 4_usize;
    if let Some(adapt) = adaptation {
        packet[3] |= 0x20;
        let mut field = vec![adapt.flags];
        if let Some(base) = adapt.pcr_base {
            field.extend_from_slice(&[
                u8::try_from((base >> 25) & 0xFF).unwrap_or(0),
                u8::try_from((base >> 17) & 0xFF).unwrap_or(0),
                u8::try_from((base >> 9) & 0xFF).unwrap_or(0),
                u8::try_from((base >> 1) & 0xFF).unwrap_or(0),
                (u8::try_from((base & 1) << 7).unwrap_or(0)) | 0x7E,
                0x00,
            ]);
        }
        field.resize(field.len() + adapt.stuffing, 0xFF);
        let body = field.len().min(TS_SIZE - idx - 1 - payload.len());
        packet[idx] = u8::try_from(body).unwrap_or(0);
        idx += 1;
        packet[idx..idx + body].copy_from_slice(&field[..body]);
        idx += body;
    }
    if !payload.is_empty() {
        packet[3] |= 0x10;
    }
    let available = TS_SIZE.saturating_sub(idx);
    let take = payload.len().min(available);
    packet[idx..idx + take].copy_from_slice(&payload[..take]);
    out.extend_from_slice(&packet);
}

/// Writes an adaptation-field-only packet carrying a PCR when the PCR PID has
/// no payload packet available at the scheduled reference time.
fn pcr_packet(out: &mut Vec<u8>, pcr_pid: u16, pcr: u64, cc: &mut u8) {
    ts_packet(
        out,
        pcr_pid,
        false,
        &[],
        Some(Adaptation {
            flags: 0x10,
            pcr_base: Some(pcr),
            // 188 bytes minus 4-byte header, length, flags and six PCR bytes.
            stuffing: TS_SIZE - 12,
        }),
        cc,
    );
}

fn build_pes(stream_id: u8, pts: u64, dts: Option<u64>, payload: &[u8]) -> Vec<u8> {
    let with_dts = dts.is_some();
    let header_len = if with_dts { 10 } else { 5 };
    let mut header = Vec::with_capacity(header_len + payload.len() + 9);
    header.extend_from_slice(&[0, 0, 1, stream_id]);
    let total = 3 + header_len + payload.len();
    let length = u16::try_from(total).unwrap_or(0);
    header.extend_from_slice(&length.to_be_bytes());
    header.push(0x80);
    header.push(if with_dts { 0xC0 } else { 0x80 });
    header.push(u8::try_from(header_len).unwrap_or(0));
    header.extend_from_slice(&encode_timestamp(if with_dts { 3 } else { 2 }, pts));
    if let Some(dts) = dts {
        header.extend_from_slice(&encode_timestamp(1, dts));
    }
    header.extend_from_slice(payload);
    header
}

/// Writes a compliant MPEG-TS stream with PAT/PMT, PCR and continuity counters.
// Table generation, payload framing and PES segmentation share one output
// buffer and continuity-counter state; splitting them would add indirection.
#[allow(clippy::too_many_lines)]
pub(crate) fn mux_ts(
    target: &File,
    target_path: &Path,
    media: &ContainerMedia,
) -> Result<(), PipelineError> {
    let pids: Vec<u16> = (0..media.tracks.len())
        .map(|i| u16::try_from(0x0100_u32 + u32::try_from(i).unwrap_or(0)).unwrap_or(0x0100))
        .collect();
    let pcr_track = media
        .tracks
        .iter()
        .position(|track| track.codec.is_video())
        .or_else(|| (!media.tracks.is_empty()).then_some(0));
    let pcr_pid = pcr_track
        .and_then(|index| pids.get(index).copied())
        .unwrap_or(PMT_PID);
    let pat = build_pat();
    let pmt = build_pmt(media, &pids, pcr_pid);
    let mut cc = HashMap::new();
    let mut out: Vec<u8> = Vec::new();
    let mut association_cc = 0_u8;
    let mut program_cc = 0_u8;
    out.extend(
        psi_packets(0, &pat, &mut association_cc)
            .map_err(|reason| ts_error(target_path, reason))?
            .into_iter()
            .flatten(),
    );
    out.extend(
        psi_packets(PMT_PID, &pmt, &mut program_cc)
            .map_err(|reason| ts_error(target_path, reason))?
            .into_iter()
            .flatten(),
    );

    let mut min_dts = i64::MAX;
    for packet in &media.packets {
        min_dts = min_dts.min(packet.dts);
    }
    if min_dts == i64::MAX {
        min_dts = 0;
    }
    let shift = PCR_OFFSET_US.saturating_sub(min_dts);
    let start_90k = to_90k(min_dts.saturating_add(shift));
    let mut next_pcr_90k = start_90k;

    // Callers may hand us packets grouped by track. Merge the per-track decode
    // queues so PCRs are scheduled in DTS order and always precede the PES
    // packets that reference them.
    let mut queues: Vec<Vec<usize>> = vec![Vec::new(); media.tracks.len()];
    for (index, packet) in media.packets.iter().enumerate() {
        if packet.track < queues.len() {
            queues[packet.track].push(index);
        }
    }
    let mut cursors = vec![0_usize; queues.len()];
    let mut order = Vec::with_capacity(media.packets.len());
    loop {
        let mut best: Option<(i64, usize, usize)> = None;
        for (track, queue) in queues.iter().enumerate() {
            if let Some(&index) = queue.get(cursors[track]) {
                let dts = media.packets[index].dts;
                if best
                    .is_none_or(|(best_dts, best_track, _)| (dts, track) < (best_dts, best_track))
                {
                    best = Some((dts, track, index));
                }
            }
        }
        match best {
            Some((_, track, index)) => {
                cursors[track] += 1;
                order.push(index);
            }
            None => break,
        }
    }

    for &index in &order {
        let packet = &media.packets[index];
        let track = &media.tracks[packet.track];
        let is_video = track.codec.is_video();
        let pid = pids[packet.track];
        let dts_90k = to_90k(packet.dts.saturating_add(shift));
        let pts_90k = to_90k(packet.pts.saturating_add(shift)).max(dts_90k);
        let mut due = Vec::new();
        while next_pcr_90k <= dts_90k {
            due.push(next_pcr_90k);
            let following = next_pcr_90k.saturating_add(PCR_INTERVAL_90K);
            if following == next_pcr_90k {
                break;
            }
            next_pcr_90k = following;
        }
        // Attach the most recent due PCR to this payload packet when possible;
        // earlier references go out as adaptation-only packets on the PCR PID.
        let attached = if pid == pcr_pid {
            due.len().checked_sub(1)
        } else {
            None
        };
        for (index, &at) in due.iter().enumerate() {
            if attached != Some(index) {
                let counter = cc.entry(pcr_pid).or_insert(0_u8);
                pcr_packet(&mut out, pcr_pid, at, counter);
            }
        }
        let with_pcr = attached.is_some();
        let pcr_base = attached.map_or(dts_90k, |index| due[index]);
        let payload = match track.codec {
            VideoCodec::H264 => {
                let mut data = vec![0, 0, 0, 1, 0x09, 0xF0];
                data.extend_from_slice(
                    &avcc_to_annex_b(&track.config, &packet.data, packet.keyframe)
                        .map_err(|reason| ts_error(target_path, reason))?,
                );
                data
            }
            VideoCodec::Hevc => {
                let mut data = vec![0, 0, 0, 1, 0x46, 0x01, 0x50];
                data.extend_from_slice(
                    &hvcc_to_annex_b(&track.config, &packet.data, packet.keyframe)
                        .map_err(|reason| ts_error(target_path, reason))?,
                );
                data
            }
            VideoCodec::Av1 => ensure_av1_temporal_delimiter(&packet.data),
            VideoCodec::Aac => {
                let asc = track
                    .config
                    .get(..2)
                    .ok_or_else(|| ts_error(target_path, "missing AAC AudioSpecificConfig"))?;
                wrap_adts(asc, &packet.data).map_err(|reason| ts_error(target_path, reason))?
            }
            VideoCodec::Opus => wrap_ts_opus(&packet.data),
            VideoCodec::Mp3 => packet.data.clone(),
            _ => {
                return Err(ts_error(
                    target_path,
                    format!("codec {} cannot be stored in TS", track.codec),
                ));
            }
        };
        if is_video && packet.keyframe {
            out.extend(
                psi_packets(0, &pat, &mut association_cc)
                    .map_err(|reason| ts_error(target_path, reason))?
                    .into_iter()
                    .flatten(),
            );
            out.extend(
                psi_packets(PMT_PID, &pmt, &mut program_cc)
                    .map_err(|reason| ts_error(target_path, reason))?
                    .into_iter()
                    .flatten(),
            );
        }
        let stream_id = if is_video {
            0xE0_u8
        } else if track.codec == VideoCodec::Opus {
            0xBD
        } else {
            0xC0
        };
        let pes = build_pes(stream_id, pts_90k, is_video.then_some(dts_90k), &payload);
        let counter = cc.entry(pid).or_insert(0_u8);
        let mut first = true;
        let mut pos = 0_usize;
        while pos < pes.len() {
            // The packet carrying PCR has 8 fewer payload bytes to leave room
            // for the adaptation-field length, flags and six PCR bytes.
            let capacity = if first && with_pcr {
                TS_SIZE - 4 - 8
            } else {
                TS_SIZE - 4
            };
            let chunk = &pes[pos..(pos + capacity).min(pes.len())];
            let last = pos + chunk.len() == pes.len();
            let adapt = if first && with_pcr {
                Some(Adaptation {
                    flags: if is_video && packet.keyframe {
                        0x50
                    } else {
                        0x10
                    },
                    pcr_base: Some(pcr_base),
                    stuffing: if last {
                        capacity.saturating_sub(chunk.len())
                    } else {
                        0
                    },
                })
            } else if last && chunk.len() < TS_SIZE - 4 {
                Some(Adaptation {
                    flags: 0,
                    pcr_base: None,
                    stuffing: TS_SIZE - 4 - chunk.len() - 1,
                })
            } else {
                None
            };
            ts_packet(&mut out, pid, first, chunk, adapt, counter);
            first = false;
            pos += chunk.len();
        }
    }
    let end_90k = media
        .packets
        .iter()
        .map(|packet| {
            to_90k(
                packet
                    .pts
                    .saturating_add(i64::try_from(packet.duration).unwrap_or(i64::MAX))
                    .saturating_add(shift),
            )
        })
        .max()
        .unwrap_or(start_90k);
    while next_pcr_90k <= end_90k {
        let counter = cc.entry(pcr_pid).or_insert(0_u8);
        pcr_packet(&mut out, pcr_pid, next_pcr_90k, counter);
        let following = next_pcr_90k.saturating_add(PCR_INTERVAL_90K);
        if following == next_pcr_90k {
            break;
        }
        next_pcr_90k = following;
    }
    let mut writer = BufWriter::new(target);
    writer.write_all(&out).map_err(|source| PipelineError::Io {
        path: target_path.to_path_buf(),
        source,
    })?;
    writer.flush().map_err(|source| PipelineError::Io {
        path: target_path.to_path_buf(),
        source,
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(label: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!(
            "fastconv-ts-{label}-{nanos}-{:?}",
            std::process::id()
        ))
    }

    fn h264_config() -> Vec<u8> {
        crate::video_bitstream::extract_avcc_from_annex_b(&[
            0, 0, 1, 0x67, 0x42, 0x00, 0x1E, 0, 0, 1, 0x68, 0xCE,
        ])
        .unwrap_or_default()
    }

    fn hevc_config() -> Vec<u8> {
        crate::video_bitstream::extract_hvcc_from_annex_b(&[
            0, 0, 1, 0x40, 0x01, 0x0C, 0x01, 0, 0, 1, 0x42, 0x01, 0x01, 0x01, 0, 0, 1, 0x44, 0x01,
            0xC0,
        ])
        .unwrap_or_default()
    }

    /// Complete baseline SPS describing a 1280x720 frame.
    fn h264_full_config() -> Vec<u8> {
        crate::video_bitstream::extract_avcc_from_annex_b(&[
            0, 0, 1, 0x67, 0x42, 0x00, 0x1E, 0xF4, 0x02, 0x80, 0x2D, 0x80, 0, 0, 1, 0x68, 0xCE,
        ])
        .unwrap_or_default()
    }

    #[test]
    fn psi_sections_are_bounded() {
        assert_eq!(
            parse_pat(&[0x00, 0x80, 0x08, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            None
        );
        assert_eq!(parse_pat(&[0x00, 0x80, 0x20, 0, 0, 0]), None);
        assert!(parse_pmt(&[0x02, 0x80, 0x0C, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).is_none());
        assert!(parse_pmt(&[0x02, 0x80, 0x30, 0, 0, 0]).is_none());
        let mut bad_info = vec![0x02, 0x80, 0x0D, 0, 1, 0xC1, 0, 0, 0xE0, 0x00, 0xF0, 0xFF];
        bad_info.extend_from_slice(&[0; 5]);
        assert!(parse_pmt(&bad_info).is_none());
    }

    #[test]
    fn opus_stream_type_reads_extension_channels() {
        let mut descriptors = vec![0x05, 0x04];
        descriptors.extend_from_slice(b"Opus");
        descriptors.extend_from_slice(&[0x7F, 0x02, 0x80, 0x06]);
        let info = es_info(0x06, &descriptors);
        assert_eq!(info.codec, VideoCodec::Opus);
        assert_eq!(info.channels, 6);
        let fallback = es_info(0x06, &descriptors[..6]);
        assert_eq!(fallback.channels, 2);
    }

    #[test]
    fn timestamps_unwrap_across_the_33_bit_boundary() {
        assert_eq!(timestamp_delta(PTS_MODULUS - 10, 10), -20);
        assert_eq!(timestamp_delta(5, PTS_MODULUS - 5), 10);
        let last = PTS_MODULUS - 5;
        let (stamps, clock) = unwrap_stamps(Some(3), Some(2), Some(last));
        assert_eq!(stamps.1, Some(PTS_MODULUS + 2));
        assert_eq!(stamps.0.unwrap_or(0) - stamps.1.unwrap_or(0), 1);
        assert_eq!(clock, stamps.1);
        let (stamps, _) = unwrap_stamps(Some(0), None, Some(PTS_MODULUS - 1));
        assert_eq!(stamps.0, Some(PTS_MODULUS));
    }

    #[test]
    fn unstamped_pes_does_not_create_or_reset_the_pid_clock() {
        let ((pts, dts), clock) = unwrap_stamps(None, None, None);
        assert_eq!((pts, dts), (None, None));
        assert_eq!(clock, None);
        // A first valid stamp just before rollover must stay near 2^33, not be
        // unwrapped as a negative offset from a synthetic zero clock.
        let near_rollover = PTS_MODULUS - 5;
        let ((pts, dts), clock) = unwrap_stamps(Some(near_rollover), None, clock);
        assert_eq!(pts, Some(near_rollover));
        assert_eq!(dts, None);
        assert_eq!(clock, Some(near_rollover));
        let ((pts, dts), clock) = unwrap_stamps(None, None, clock);
        assert_eq!((pts, dts), (None, None));
        assert_eq!(clock, Some(near_rollover));
    }

    #[test]
    fn durations_follow_presentation_order_without_reordering_packets() {
        let mut media = ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::H264,
                width: Some(16),
                height: Some(16),
                channels: None,
                sample_rate: None,
                config: Vec::new(),
                timescale: US_TIMESCALE,
                presentation_edit: None,
            }],
            packets: [30_000_i64, 0, 10_000]
                .into_iter()
                .enumerate()
                .map(|(index, pts)| ContainerPacket {
                    track: 0,
                    pts,
                    dts: 0,
                    duration: 1,
                    keyframe: index == 1,
                    data: vec![u8::try_from(index).unwrap_or(0)],
                })
                .collect(),
        };
        fix_durations(&mut media);
        assert_eq!(
            media.packets.iter().map(|p| p.data[0]).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(media.packets[0].duration, 20_000);
        assert_eq!(media.packets[1].duration, 10_000);
        assert_eq!(media.packets[2].duration, 20_000);
    }

    #[test]
    fn crc32_mpeg2_matches_reference() {
        assert_eq!(crc32_mpeg2(&[]), 0xFFFF_FFFF);
        let expected = {
            let mut value = 0xFFFF_FFFF_u32;
            for byte in [0x00_u8, 0x01, 0xC1, 0x00, 0x00] {
                value ^= u32::from(byte) << 24;
                for _ in 0..8 {
                    value = if value & 0x8000_0000 != 0 {
                        (value << 1) ^ 0x04C1_1DB7
                    } else {
                        value << 1
                    };
                }
            }
            value
        };
        assert_eq!(crc32_mpeg2(&[0x00, 0x01, 0xC1, 0x00, 0x00]), expected);
    }

    #[test]
    fn timestamps_encode_and_decode() {
        let encoded = encode_timestamp(3, 126_000);
        assert_eq!(decode_timestamp(&encoded, 0, 3), Some(126_000));
        assert_eq!(decode_timestamp(&encoded, 0, 2), None);
    }

    #[test]
    fn microseconds_and_90k_ticks_round_trip_within_one_tick() {
        for us in [0_i64, 1, 15, 21_333, 33_333, 1_400_000, 999_990] {
            let back = from_90k(to_90k(us));
            assert!((back - us).abs() <= 12, "{us} µs -> 90 kHz -> {back} µs");
        }
    }

    fn large_pes_media(config: &[u8], video_frame: Vec<u8>) -> ContainerMedia {
        ContainerMedia {
            tracks: vec![
                ContainerTrack {
                    codec: VideoCodec::H264,
                    width: Some(320),
                    height: Some(240),
                    channels: None,
                    sample_rate: None,
                    config: config.to_vec(),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                },
                ContainerTrack {
                    codec: VideoCodec::Aac,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: vec![0x12, 0x10],
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                },
            ],
            packets: vec![
                ContainerPacket {
                    track: 0,
                    pts: 0,
                    dts: 0,
                    duration: 33_333,
                    keyframe: true,
                    data: video_frame,
                },
                ContainerPacket {
                    track: 1,
                    pts: 0,
                    dts: 0,
                    duration: 21_333,
                    keyframe: true,
                    data: vec![0xAA; 8],
                },
                ContainerPacket {
                    track: 1,
                    pts: 21_333,
                    dts: 21_333,
                    duration: 21_333,
                    keyframe: true,
                    data: vec![0xBB; 8],
                },
                ContainerPacket {
                    track: 1,
                    pts: 42_666,
                    dts: 42_666,
                    duration: 21_333,
                    keyframe: true,
                    data: vec![0xCC; 8],
                },
            ],
        }
    }

    #[test]
    fn multi_packet_pes_reassembly_is_byte_exact_and_accurate()
    -> Result<(), Box<dyn std::error::Error>> {
        let config = h264_config();
        // One 4 KiB AVCC access unit: a single IDR NALU of 4092 bytes.
        let mut video_frame = Vec::with_capacity(4096);
        video_frame.extend_from_slice(&4092_u32.to_be_bytes());
        video_frame.push(0x65);
        video_frame.resize(4096, 0x5A);

        // The TS muxer prepends an access-unit delimiter and the parameter
        // sets; demuxing converts the reassembled Annex-B stream back to AVCC.
        let mut annex_b = vec![0_u8, 0, 0, 1, 0x09, 0xF0];
        annex_b.extend_from_slice(&avcc_to_annex_b(&config, &video_frame, true)?);
        let expected_video = annex_b_to_avcc(&annex_b);
        let media = large_pes_media(&config, video_frame);

        let path = temp_path("large-pes");
        let file = File::create(&path)?;
        mux_ts(&file, &path, &media)?;
        drop(file);
        let bytes = std::fs::read(&path)?;
        let packet_count = bytes.len() / TS_SIZE;
        assert!(
            packet_count > 22,
            "4 KiB PES should span more than 22 TS packets, got {packet_count}"
        );
        let out = demux_ts(&path)?;
        std::fs::remove_file(&path)?;

        let video: Vec<_> = out.packets.iter().filter(|p| p.track == 0).collect();
        assert_eq!(video.len(), 1);
        assert_eq!(video[0].data, expected_video);
        assert!(
            (video[0].pts - PCR_OFFSET_US).abs() <= 15,
            "video pts {} µs",
            video[0].pts
        );
        assert_eq!(video[0].dts, video[0].pts);

        let audio: Vec<_> = out.packets.iter().filter(|p| p.track == 1).collect();
        assert_eq!(audio.len(), 3);
        for (packet, (expected_data, expected_pts)) in audio.iter().zip([
            (vec![0xAA; 8], 0_i64),
            (vec![0xBB; 8], 21_333),
            (vec![0xCC; 8], 42_666),
        ]) {
            assert_eq!(packet.data, expected_data);
            assert!(
                (packet.pts - PCR_OFFSET_US - expected_pts).abs() <= 15,
                "audio pts {} µs, expected {}",
                packet.pts,
                PCR_OFFSET_US + expected_pts
            );
            assert_eq!(packet.dts, packet.pts);
        }
        Ok(())
    }

    #[test]
    fn pes_with_distinct_dts_preserves_the_composition_offset()
    -> Result<(), Box<dyn std::error::Error>> {
        let media = ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::H264,
                width: Some(320),
                height: Some(240),
                channels: None,
                sample_rate: None,
                config: h264_config(),
                timescale: US_TIMESCALE,
                presentation_edit: None,
            }],
            packets: vec![ContainerPacket {
                track: 0,
                pts: 20_000,
                dts: 10_000,
                duration: 33_333,
                keyframe: true,
                data: vec![0, 0, 0, 2, 0x65, 0x88],
            }],
        };
        let out = round_trip("distinct-dts", &media)?;
        assert_eq!(out.packets.len(), 1);
        let offset = out.packets[0].pts - out.packets[0].dts;
        assert!(
            (offset - 10_000).abs() <= 15,
            "pts - dts = {offset} µs, expected 10000"
        );
        Ok(())
    }

    fn round_trip(
        label: &str,
        media: &ContainerMedia,
    ) -> Result<ContainerMedia, Box<dyn std::error::Error>> {
        let path = temp_path(label);
        let file = File::create(&path)?;
        mux_ts(&file, &path, media)?;
        drop(file);
        let bytes = std::fs::read(&path)?;
        assert_eq!(bytes.len() % 188, 0);
        assert!(bytes.chunks(188).all(|p| p[0] == 0x47));
        let out = demux_ts(&path)?;
        std::fs::remove_file(&path)?;
        Ok(out)
    }

    #[test]
    fn ts_h264_aac_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        let media = ContainerMedia {
            tracks: vec![
                ContainerTrack {
                    codec: VideoCodec::H264,
                    width: Some(320),
                    height: Some(240),
                    channels: None,
                    sample_rate: None,
                    config: h264_config(),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                },
                ContainerTrack {
                    codec: VideoCodec::Aac,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: vec![0x12, 0x10],
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                },
            ],
            packets: vec![
                ContainerPacket {
                    track: 0,
                    pts: 0,
                    dts: 0,
                    duration: 33_333,
                    keyframe: true,
                    data: vec![0, 0, 0, 2, 0x65, 0x88],
                },
                ContainerPacket {
                    track: 1,
                    pts: 0,
                    dts: 0,
                    duration: 21_000,
                    keyframe: true,
                    data: vec![0xAA; 8],
                },
                ContainerPacket {
                    track: 0,
                    pts: 33_333,
                    dts: 33_333,
                    duration: 33_333,
                    keyframe: false,
                    data: vec![0, 0, 0, 2, 0x41, 0x22],
                },
            ],
        };
        let out = round_trip("h264-aac", &media)?;
        assert_eq!(out.tracks.len(), 2);
        assert_eq!(out.tracks[0].codec, VideoCodec::H264);
        assert_eq!(out.tracks[1].codec, VideoCodec::Aac);
        assert_eq!(out.tracks[1].sample_rate, Some(44_100));
        let video: Vec<_> = out.packets.iter().filter(|p| p.track == 0).collect();
        let audio: Vec<_> = out.packets.iter().filter(|p| p.track == 1).collect();
        assert_eq!(video.len(), 2);
        assert_eq!(audio.len(), 1);
        assert!(video[0].data.ends_with(&media.packets[0].data));
        assert!(video[0].keyframe);
        assert_eq!(audio[0].data, vec![0xAA; 8]);
        assert!(!video[1].keyframe);
        Ok(())
    }

    #[test]
    fn ts_hevc_mp3_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        let media = ContainerMedia {
            tracks: vec![
                ContainerTrack {
                    codec: VideoCodec::Hevc,
                    width: Some(640),
                    height: Some(360),
                    channels: None,
                    sample_rate: None,
                    config: hevc_config(),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                },
                ContainerTrack {
                    codec: VideoCodec::Mp3,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(44_100),
                    config: Vec::new(),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                },
            ],
            packets: vec![
                ContainerPacket {
                    track: 0,
                    pts: 0,
                    dts: 0,
                    duration: 33_333,
                    keyframe: true,
                    data: vec![0, 0, 0, 3, 0x26, 0x01, 0x02],
                },
                ContainerPacket {
                    track: 1,
                    pts: 0,
                    dts: 0,
                    duration: 26_000,
                    keyframe: true,
                    data: {
                        let mut frame = vec![0xFF, 0xFB, 0x10, 0x00];
                        frame.resize(104, 0);
                        frame
                    },
                },
            ],
        };
        let out = round_trip("hevc-mp3", &media)?;
        assert_eq!(out.tracks[0].codec, VideoCodec::Hevc);
        assert_eq!(out.tracks[1].codec, VideoCodec::Mp3);
        let video: Vec<_> = out.packets.iter().filter(|p| p.track == 0).collect();
        let audio: Vec<_> = out.packets.iter().filter(|p| p.track == 1).collect();
        assert!(video[0].data.ends_with(&media.packets[0].data));
        assert_eq!(audio[0].data.len(), 104);
        Ok(())
    }

    #[test]
    fn ts_av1_opus_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        let media = ContainerMedia {
            tracks: vec![
                ContainerTrack {
                    codec: VideoCodec::Av1,
                    width: Some(1920),
                    height: Some(1080),
                    channels: None,
                    sample_rate: None,
                    config: Vec::new(),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                },
                ContainerTrack {
                    codec: VideoCodec::Opus,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: default_opus_head(2, 0),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                },
            ],
            packets: vec![
                ContainerPacket {
                    track: 0,
                    pts: 0,
                    dts: 0,
                    duration: 33_333,
                    keyframe: true,
                    data: vec![0x0A, 0x02, 0x00, 0x0A],
                },
                ContainerPacket {
                    track: 1,
                    pts: 0,
                    dts: 0,
                    duration: 20_000,
                    keyframe: true,
                    data: vec![0xFC, 0x01, 0x02],
                },
            ],
        };
        let out = round_trip("av1-opus", &media)?;
        assert_eq!(out.tracks[0].codec, VideoCodec::Av1);
        assert_ne!(out.tracks[0].config.len(), 0);
        assert_eq!(out.tracks[1].codec, VideoCodec::Opus);
        let video: Vec<_> = out.packets.iter().filter(|p| p.track == 0).collect();
        let audio: Vec<_> = out.packets.iter().filter(|p| p.track == 1).collect();
        assert!(video[0].data.ends_with(&[0x0A, 0x02, 0x00, 0x0A]));
        assert_eq!(audio[0].data, vec![0xFC, 0x01, 0x02]);
        Ok(())
    }

    fn many_h264_tracks(count: usize) -> ContainerMedia {
        ContainerMedia {
            tracks: (0..count)
                .map(|_| ContainerTrack {
                    codec: VideoCodec::H264,
                    width: Some(16),
                    height: Some(16),
                    channels: None,
                    sample_rate: None,
                    config: h264_config(),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                })
                .collect(),
            packets: Vec::new(),
        }
    }

    #[test]
    fn multi_packet_pmt_is_emitted_and_reassembled() -> Result<(), Box<dyn std::error::Error>> {
        let media = many_h264_tracks(40);
        let pids: Vec<u16> = (0..media.tracks.len())
            .map(|index| u16::try_from(0x0100 + index).unwrap_or(0x0100))
            .collect();
        let pmt = build_pmt(&media, &pids, pids[0]);
        assert!(pmt.len() > TS_SIZE - 5);
        let mut cc = 0_u8;
        let packets = psi_packets(PMT_PID, &pmt, &mut cc)?;
        assert!(packets.len() > 1);
        for (index, packet) in packets.iter().enumerate() {
            assert_eq!(packet[0], 0x47);
            assert_eq!(packet[1] & 0x40 != 0, index == 0);
            assert_eq!(packet[3] & 0x0F, u8::try_from(index).unwrap_or(0));
        }
        assert_eq!(packets[0][4], 0);
        assert!(psi_packets(0, &[0_u8; 1022], &mut cc).is_err());

        let mut bytes = Vec::new();
        let mut pat_cc = 0_u8;
        for packet in psi_packets(0, &build_pat(), &mut pat_cc)? {
            bytes.extend_from_slice(&packet);
        }
        for packet in packets {
            bytes.extend_from_slice(&packet);
        }
        let path = temp_path("split-pmt");
        std::fs::write(&path, &bytes)?;
        let out = demux_ts(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(out.tracks.len(), 40);
        Ok(())
    }

    #[test]
    fn pmt_section_start_byte_alone_is_reassembled() -> Result<(), Box<dyn std::error::Error>> {
        let media = ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::Aac,
                width: None,
                height: None,
                channels: Some(2),
                sample_rate: Some(48_000),
                config: vec![0x12, 0x10],
                timescale: US_TIMESCALE,
                presentation_edit: None,
            }],
            packets: Vec::new(),
        };
        let pcr_pid = 0x0100_u16;
        let pmt = build_pmt(&media, &[pcr_pid], pcr_pid);
        let mut cc = 0_u8;
        let mut bytes = Vec::new();
        for packet in psi_packets(0, &build_pat(), &mut cc)? {
            bytes.extend_from_slice(&packet);
        }
        // The PUSI packet carries only the pointer and the PMT table id; the
        // 3-byte header cannot be read until the continuation arrives.
        ts_packet(&mut bytes, PMT_PID, true, &[0, pmt[0]], None, &mut cc);
        ts_packet(&mut bytes, PMT_PID, false, &pmt[1..], None, &mut cc);
        let path = temp_path("pmt-first-byte");
        std::fs::write(&path, &bytes)?;
        let out = demux_ts(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(out.tracks.len(), 1);
        assert_eq!(out.tracks[0].codec, VideoCodec::Aac);
        Ok(())
    }

    fn pcr_values(bytes: &[u8]) -> Vec<(u16, u64)> {
        bytes
            .chunks(TS_SIZE)
            .filter_map(|packet| {
                if packet.len() != TS_SIZE || packet[0] != 0x47 || (packet[3] >> 4) & 0x02 == 0 {
                    return None;
                }
                let pid = (u16::from(packet[1] & 0x1F) << 8) | u16::from(packet[2]);
                if packet[4] < 7 || packet[5] & 0x10 == 0 || packet[10] & 0x7E != 0x7E {
                    return None;
                }
                let value = (u64::from(packet[6]) << 25)
                    | (u64::from(packet[7]) << 17)
                    | (u64::from(packet[8]) << 9)
                    | (u64::from(packet[9]) << 1)
                    | u64::from(packet[10] >> 7);
                Some((pid, value))
            })
            .collect()
    }

    #[test]
    fn audio_only_ts_carries_periodic_pcr_on_the_audio_pid()
    -> Result<(), Box<dyn std::error::Error>> {
        let media = ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::Aac,
                width: None,
                height: None,
                channels: Some(2),
                sample_rate: Some(44_100),
                config: vec![0x12, 0x10],
                timescale: US_TIMESCALE,
                presentation_edit: None,
            }],
            packets: (0..100)
                .map(|index| ContainerPacket {
                    track: 0,
                    pts: i64::from(index) * 20_000,
                    dts: i64::from(index) * 20_000,
                    duration: 20_000,
                    keyframe: true,
                    data: vec![0xAA; 8],
                })
                .collect(),
        };
        let path = temp_path("audio-only-pcr");
        let file = File::create(&path)?;
        mux_ts(&file, &path, &media)?;
        drop(file);
        let bytes = std::fs::read(&path)?;
        std::fs::remove_file(&path)?;
        let mut pmt_pcr = 0_u16;
        for packet in bytes.chunks(TS_SIZE) {
            let pid = (u16::from(packet[1] & 0x1F) << 8) | u16::from(packet[2]);
            if pid == PMT_PID && packet[1] & 0x40 != 0 {
                let start = 5_usize + usize::from(packet[4]);
                let section = &packet[start..];
                pmt_pcr = (u16::from(section[8] & 0x1F) << 8) | u16::from(section[9]);
            }
        }
        assert_eq!(pmt_pcr, 0x0100);
        let values = pcr_values(&bytes);
        assert!(values.len() >= 20, "only {} PCRs in 2 s", values.len());
        assert!(values.iter().all(|(pid, _)| *pid == 0x0100));
        for pair in values.windows(2) {
            let gap = pair[1].1.saturating_sub(pair[0].1);
            assert!(gap > 0 && gap <= PCR_INTERVAL_90K, "PCR gap {gap} ticks");
        }
        let first = values.first().map_or(0, |value| value.1);
        let last = values.last().map_or(0, |value| value.1);
        assert!(last.saturating_sub(first) >= 18 * PCR_INTERVAL_90K);
        Ok(())
    }

    #[test]
    fn adts_channel_configuration_seven_means_eight_channels()
    -> Result<(), Box<dyn std::error::Error>> {
        let media = ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::Aac,
                width: None,
                height: None,
                channels: Some(8),
                sample_rate: Some(44_100),
                config: vec![0x12, 0x38],
                timescale: US_TIMESCALE,
                presentation_edit: None,
            }],
            packets: vec![ContainerPacket {
                track: 0,
                pts: 0,
                dts: 0,
                duration: 21_333,
                keyframe: true,
                data: vec![0xAA; 8],
            }],
        };
        let out = round_trip("adts-8ch", &media)?;
        assert_eq!(out.tracks[0].sample_rate, Some(44_100));
        assert_eq!(out.tracks[0].channels, Some(8));
        Ok(())
    }

    #[test]
    fn demux_reads_coded_dimensions_from_sps() -> Result<(), Box<dyn std::error::Error>> {
        let media = ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::H264,
                width: None,
                height: None,
                channels: None,
                sample_rate: None,
                config: h264_full_config(),
                timescale: US_TIMESCALE,
                presentation_edit: None,
            }],
            packets: vec![ContainerPacket {
                track: 0,
                pts: 0,
                dts: 0,
                duration: 33_333,
                keyframe: true,
                data: vec![0, 0, 0, 2, 0x65, 0x88],
            }],
        };
        let out = round_trip("sps-dimensions", &media)?;
        assert_eq!(out.tracks[0].width, Some(1280));
        assert_eq!(out.tracks[0].height, Some(720));
        Ok(())
    }

    #[test]
    fn truncated_bounded_pes_is_discarded() -> Result<(), Box<dyn std::error::Error>> {
        let media = ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::H264,
                width: Some(16),
                height: Some(16),
                channels: None,
                sample_rate: None,
                config: h264_config(),
                timescale: US_TIMESCALE,
                presentation_edit: None,
            }],
            packets: (0..2)
                .map(|index| ContainerPacket {
                    track: 0,
                    pts: i64::from(index) * 33_333,
                    dts: i64::from(index) * 33_333,
                    duration: 33_333,
                    keyframe: index == 0,
                    data: vec![0, 0, 0, 2, 0x65, u8::try_from(index).unwrap_or(0)],
                })
                .collect(),
        };
        let path = temp_path("truncated-pes");
        let file = File::create(&path)?;
        mux_ts(&file, &path, &media)?;
        drop(file);
        let mut bytes = std::fs::read(&path)?;
        let start = bytes
            .windows(4)
            .position(|window| window == [0_u8, 0, 1, 0xE0])
            .ok_or("first video PES")?;
        let length_pos = start + 4;
        bytes[length_pos..length_pos + 2].copy_from_slice(&0xFFFE_u16.to_be_bytes());
        std::fs::write(&path, &bytes)?;
        let out = demux_ts(&path)?;
        std::fs::remove_file(&path)?;
        let video: Vec<_> = out.packets.iter().filter(|p| p.track == 0).collect();
        assert_eq!(video.len(), 1);
        Ok(())
    }

    #[test]
    fn adaptation_only_packets_reuse_the_preceding_continuity_counter() {
        let mut out = Vec::new();
        let mut cc = 0_u8;
        ts_packet(&mut out, 0x0100, false, &[0xAA], None, &mut cc);
        ts_packet(
            &mut out,
            0x0100,
            false,
            &[],
            Some(Adaptation {
                flags: 0x10,
                pcr_base: Some(42),
                stuffing: TS_SIZE - 12,
            }),
            &mut cc,
        );
        ts_packet(&mut out, 0x0100, false, &[0xBB], None, &mut cc);
        let counters: Vec<u8> = out.chunks(TS_SIZE).map(|packet| packet[3] & 0x0F).collect();
        assert_eq!(counters, vec![0, 0, 1]);

        // The same rule applies when the payload counter wraps at 15.
        let mut out = Vec::new();
        let mut cc = 15_u8;
        ts_packet(&mut out, 0x0100, false, &[0xAA], None, &mut cc);
        ts_packet(&mut out, 0x0100, false, &[], None, &mut cc);
        ts_packet(&mut out, 0x0100, false, &[0xBB], None, &mut cc);
        let counters: Vec<u8> = out.chunks(TS_SIZE).map(|packet| packet[3] & 0x0F).collect();
        assert_eq!(counters, vec![15, 15, 0]);
    }

    /// Returns `(pid, PES stamp, most recent PCR)` for every PES start in the
    /// byte stream, so tests can verify PCR ordering without demuxing.
    fn pes_stamps_and_pcrs(bytes: &[u8]) -> Vec<(u16, u64, u64)> {
        let mut out = Vec::new();
        let mut latest_pcr = None;
        for packet in bytes.chunks(TS_SIZE) {
            let pid = (u16::from(packet[1] & 0x1F) << 8) | u16::from(packet[2]);
            let afc = (packet[3] >> 4) & 0x03;
            let mut idx = 4_usize;
            if afc & 0x02 != 0 {
                let len = usize::from(packet[4]);
                if idx.saturating_add(1).saturating_add(len) > packet.len() {
                    continue;
                }
                if len >= 7 && packet[5] & 0x10 != 0 && packet[10] & 0x7E == 0x7E {
                    let value = (u64::from(packet[6]) << 25)
                        | (u64::from(packet[7]) << 17)
                        | (u64::from(packet[8]) << 9)
                        | (u64::from(packet[9]) << 1)
                        | u64::from(packet[10] >> 7);
                    latest_pcr = Some(value);
                }
                idx += 1 + len;
            }
            if afc & 0x01 == 0 || idx >= packet.len() {
                continue;
            }
            let payload = &packet[idx..];
            if pid != 0x0100 && pid != 0x0101 {
                continue;
            }
            if packet[1] & 0x40 == 0 || payload.get(..3) != Some([0_u8, 0, 1].as_slice()) {
                continue;
            }
            let stamp = if payload.get(7).is_some_and(|&flags| flags & 0xC0 == 0xC0) {
                decode_timestamp(payload, 14, 1)
            } else {
                decode_timestamp(payload, 9, 2)
            };
            if let (Some(stamp), Some(pcr)) = (stamp, latest_pcr) {
                out.push((pid, stamp, pcr));
            }
        }
        out
    }

    #[test]
    fn pcrs_precede_pes_packets_from_every_track_in_decode_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let media = ContainerMedia {
            tracks: vec![
                ContainerTrack {
                    codec: VideoCodec::H264,
                    width: Some(320),
                    height: Some(240),
                    channels: None,
                    sample_rate: None,
                    config: h264_config(),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                },
                ContainerTrack {
                    codec: VideoCodec::Aac,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: vec![0x12, 0x10],
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                },
            ],
            // All video packets precede all audio packets in the input.
            packets: (0..50)
                .map(|index| ContainerPacket {
                    track: 0,
                    pts: i64::from(index) * 20_000,
                    dts: i64::from(index) * 20_000,
                    duration: 20_000,
                    keyframe: true,
                    data: vec![0, 0, 0, 2, 0x65, 0x88],
                })
                .chain((0..50).map(|index| ContainerPacket {
                    track: 1,
                    pts: i64::from(index) * 20_000,
                    dts: i64::from(index) * 20_000,
                    duration: 20_000,
                    keyframe: true,
                    data: vec![0xAA; 8],
                }))
                .collect(),
        };
        let path = temp_path("interleaved-pcr");
        let file = File::create(&path)?;
        mux_ts(&file, &path, &media)?;
        drop(file);
        let bytes = std::fs::read(&path)?;
        std::fs::remove_file(&path)?;

        let pcrs = pcr_values(&bytes);
        assert!(pcrs.len() > 2, "only {} PCRs emitted", pcrs.len());
        for pair in pcrs.windows(2) {
            let gap = pair[1].1.saturating_sub(pair[0].1);
            assert!(gap > 0 && gap <= PCR_INTERVAL_90K, "PCR gap {gap} ticks");
        }

        let stamps = pes_stamps_and_pcrs(&bytes);
        assert_eq!(stamps.len(), 100);
        assert_eq!(
            stamps.iter().filter(|(pid, _, _)| *pid == 0x0100).count(),
            50
        );
        assert_eq!(
            stamps.iter().filter(|(pid, _, _)| *pid == 0x0101).count(),
            50
        );
        for (pid, stamp, pcr) in stamps {
            assert!(
                stamp >= pcr,
                "PID {pid:#06x} stamp {stamp} precedes PCR {pcr}"
            );
        }
        Ok(())
    }

    #[test]
    fn ts_pmt_header_ending_in_ff_is_not_trimmed() -> Result<(), Box<dyn std::error::Error>> {
        let mut body = vec![0x00, 0x01, 0xC1, 0x00, 0x00, 0xF0, 0x00, 0xF0, 0x00];
        for index in 0..45_u16 {
            let pid = 0x0200 + index;
            body.extend_from_slice(&[
                0x1B,
                u8::try_from(pid >> 8).unwrap_or(0) | 0xE0,
                u8::try_from(pid & 0xFF).unwrap_or(0),
                0xF0,
                0x00,
            ]);
        }
        body.extend_from_slice(&[0x1B, 0xE2, 0xFF, 0xF0, 0x01, 0xAA]);
        body.extend_from_slice(&[
            0x06, 0xE1, 0x12, 0xF0, 0x06, 0x05, 0x04, b'O', b'p', b'u', b's',
        ]);
        assert_eq!(body.len(), 251);
        let mut pmt = vec![0x02, 0x80, 0xFF];
        pmt.extend_from_slice(&body);
        pmt.extend_from_slice(&[0, 0, 0, 0]);
        assert_eq!(pmt.len(), 258);

        let mut bytes = Vec::new();
        let mut cc = 0_u8;
        for packet in psi_packets(0, &build_pat(), &mut cc)? {
            bytes.extend_from_slice(&packet);
        }
        // The PUSI carries only the pointer and the 3-byte header; the final
        // header byte is 0xFF and must not be mistaken for stuffing.
        ts_packet(
            &mut bytes,
            PMT_PID,
            true,
            &[0, 0x02, 0x80, 0xFF],
            None,
            &mut cc,
        );
        for part in pmt[3..].chunks(184) {
            ts_packet(&mut bytes, PMT_PID, false, part, None, &mut cc);
        }
        let path = temp_path("pmt-header-ff");
        std::fs::write(&path, &bytes)?;
        let out = demux_ts(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(out.tracks.len(), 47);
        assert_eq!(out.tracks[46].codec, VideoCodec::Opus);
        Ok(())
    }

    #[test]
    fn ts_single_opus_packet_keeps_the_coded_duration() -> Result<(), Box<dyn std::error::Error>> {
        let media = ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::Opus,
                width: None,
                height: None,
                channels: Some(2),
                sample_rate: Some(48_000),
                config: default_opus_head(2, 0),
                timescale: US_TIMESCALE,
                presentation_edit: None,
            }],
            packets: vec![ContainerPacket {
                track: 0,
                pts: 0,
                dts: 0,
                duration: 1,
                keyframe: true,
                data: vec![0xFC; 20],
            }],
        };
        let out = round_trip("opus-single-duration", &media)?;
        assert_eq!(out.packets.len(), 1);
        assert_eq!(out.packets[0].duration, 20_000);
        Ok(())
    }
}
