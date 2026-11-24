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
    extract_hvcc_from_annex_b, hvcc_to_annex_b, is_av1_keyframe, is_h264_keyframe_annex_b,
    is_hevc_keyframe_annex_b, parse_audio_specific_config, unwrap_adts, unwrap_ts_opus, wrap_adts,
    wrap_ts_opus,
};
use crate::video_container::{
    ContainerMedia, ContainerPacket, ContainerTrack, DemuxedMedia, US_TIMESCALE, rescale_ticks,
};

const TS_SIZE: usize = 188;
const TS_SCAN_CAP: u64 = 2 * 1024 * 1024 * 1024;
const PMT_PID: u16 = 0x1000;
/// Initial PCR/PTS offset: 1.4 seconds expressed in canonical microseconds.
const PCR_OFFSET_US: i64 = 1_400_000;

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
    let end = 3_usize.checked_add(section_length(section)?)?;
    if end > section.len() {
        return None;
    }
    let mut idx = 8_usize;
    while idx + 4 <= end - 4 {
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
            if let Some(reg) = descriptor(desc, 0x05)
                && reg.len() >= 4
            {
                if &reg[..4] == b"Opus" {
                    return EsInfo {
                        codec: VideoCodec::Opus,
                        channels: 2,
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
            if let Some(ext) = descriptor(desc, 0x7F)
                && ext.len() >= 2
                && ext[0] == 0x80
            {
                return EsInfo {
                    codec: VideoCodec::Opus,
                    channels: u16::from(ext[1] & 0x0F).max(1),
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
    let end = 3_usize.checked_add(section_length(section)?)?;
    if end > section.len() {
        return None;
    }
    let program_info = (usize::from(section.get(10)? & 0x0F) << 8) | usize::from(section[11]);
    let mut idx = 12_usize.checked_add(program_info)?;
    let mut streams = Vec::new();
    while idx + 5 <= end - 4 {
        let es_len = (usize::from(section[idx + 3] & 0x0F) << 8) | usize::from(section[idx + 4]);
        let desc_end = idx.checked_add(5)?.checked_add(es_len)?;
        if desc_end > end - 4 {
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
            && let Some(section) = section_start(&parsed)
            && let Some(streams) = parse_pmt(section)
        {
            pmt_streams = streams;
            break;
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
                let canonical = if codec == VideoCodec::H264 {
                    if media.tracks[track].config.is_empty()
                        && keyframe
                        && let Some(config) = extract_avcc_from_annex_b(&data)
                    {
                        media.tracks[track].config = config;
                    }
                    annex_b_to_avcc(&data)
                } else {
                    if media.tracks[track].config.is_empty()
                        && keyframe
                        && let Some(config) = extract_hvcc_from_annex_b(&data)
                    {
                        media.tracks[track].config = config;
                    }
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
                    && let Some((_, index, _)) = parse_audio_specific_config(&frames[0].1)
                    && let Some(&rate) = ADTS_RATES.get(usize::from(index))
                    && rate > 0
                {
                    media.tracks[track].sample_rate = Some(rate);
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
                let step = 20_000;
                let mut frame_pts = pts_us;
                for frame in unwrap_ts_opus(&data) {
                    push(media, track, frame_pts, frame_pts, frame, true);
                    frame_pts = frame_pts.saturating_add(step);
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
        if let Some(previous) = buffers.remove(&parsed.pid) {
            let (pts, dts) = stamps.remove(&parsed.pid).unwrap_or((None, None));
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
        remaining.remove(&parsed.pid);
        let Some(PesPayload {
            data,
            pts,
            dts,
            remaining: left,
        }) = pes_payload(&parsed)
        else {
            continue;
        };
        buffers.insert(parsed.pid, data);
        stamps.insert(parsed.pid, (pts, dts));
        remaining.insert(parsed.pid, left);
    }
    let pending: Vec<PendingPes> = buffers
        .into_iter()
        .map(|(pid, data)| {
            let (pts, dts) = stamps.remove(&pid).unwrap_or((None, None));
            (pid, data, pts, dts)
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
    media.packets.push(ContainerPacket {
        track,
        pts,
        dts,
        duration: 1,
        keyframe,
        data,
    });
}

fn fix_durations(media: &mut ContainerMedia) {
    for track in 0..media.tracks.len() {
        let indexes: Vec<usize> = media
            .packets
            .iter()
            .enumerate()
            .filter(|(_, p)| p.track == track)
            .map(|(i, _)| i)
            .collect();
        for pair in indexes.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let delta = media.packets[b]
                .pts
                .saturating_sub(media.packets[a].pts)
                .max(1);
            media.packets[a].duration = u64::try_from(delta).unwrap_or(1);
        }
        if let Some(&last) = indexes.last() {
            let duration = indexes.iter().rev().nth(1).map_or(1, |&prev| {
                media.packets[last]
                    .pts
                    .saturating_sub(media.packets[prev].pts)
                    .max(1)
            });
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

fn psi_packet(pid: u16, table: &[u8], cc: &mut u8) -> [u8; TS_SIZE] {
    let mut packet = [0xFF_u8; TS_SIZE];
    packet[0] = 0x47;
    packet[1] = 0x40 | (u8::try_from(pid >> 8).unwrap_or(0) & 0x1F);
    packet[2] = u8::try_from(pid & 0xFF).unwrap_or(0);
    packet[3] = 0x10;
    packet[4] = 0;
    let len = table.len().min(TS_SIZE - 5);
    packet[5..5 + len].copy_from_slice(&table[..len]);
    packet[3] |= *cc & 0x0F;
    *cc = cc.wrapping_add(1) & 0x0F;
    packet
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

fn build_pmt(media: &ContainerMedia, pids: &[u16]) -> Vec<u8> {
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
        0xE0 | (u8::try_from(pids.first().copied().unwrap_or(0x0100) >> 8).unwrap_or(0) & 0x1F),
        u8::try_from(pids.first().copied().unwrap_or(0x0100) & 0xFF).unwrap_or(0),
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
    packet[3] = *cc & 0x0F;
    *cc = cc.wrapping_add(1) & 0x0F;
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
    let pat = build_pat();
    let pmt = build_pmt(media, &pids);
    let mut cc = HashMap::new();
    let mut out = Vec::new();
    let mut association_cc = 0_u8;
    let mut program_cc = 0_u8;
    out.extend_from_slice(&psi_packet(0, &pat, &mut association_cc));
    out.extend_from_slice(&psi_packet(PMT_PID, &pmt, &mut program_cc));

    let mut min_dts = i64::MAX;
    for packet in &media.packets {
        min_dts = min_dts.min(packet.dts);
    }
    if min_dts == i64::MAX {
        min_dts = 0;
    }
    let shift = PCR_OFFSET_US.saturating_sub(min_dts);

    for packet in &media.packets {
        let track = &media.tracks[packet.track];
        let is_video = track.codec.is_video();
        let dts_90k = to_90k(packet.dts.saturating_add(shift));
        let pts_90k = to_90k(packet.pts.saturating_add(shift)).max(dts_90k);
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
            out.extend_from_slice(&psi_packet(0, &pat, &mut association_cc));
            out.extend_from_slice(&psi_packet(PMT_PID, &pmt, &mut program_cc));
        }
        let stream_id = if is_video {
            0xE0_u8
        } else if track.codec == VideoCodec::Opus {
            0xBD
        } else {
            0xC0
        };
        let pes = build_pes(stream_id, pts_90k, is_video.then_some(dts_90k), &payload);
        let pid = pids[packet.track];
        let counter = cc.entry(pid).or_insert(0_u8);
        let mut first = true;
        let mut pos = 0_usize;
        while pos < pes.len() {
            // A first video packet carries a PCR field (one length byte, one
            // flags byte and six PCR bytes), so its payload must be 8 bytes
            // shorter than the usual 184 to leave room for that field.
            let capacity = if first && is_video {
                TS_SIZE - 4 - 8
            } else {
                TS_SIZE - 4
            };
            let chunk = &pes[pos..(pos + capacity).min(pes.len())];
            let last = pos + chunk.len() == pes.len();
            let adapt = if first && is_video {
                Some(Adaptation {
                    flags: if packet.keyframe { 0x50 } else { 0x10 },
                    pcr_base: Some(dts_90k),
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
                },
                ContainerTrack {
                    codec: VideoCodec::Aac,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: vec![0x12, 0x10],
                    timescale: US_TIMESCALE,
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
                },
                ContainerTrack {
                    codec: VideoCodec::Aac,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: vec![0x12, 0x10],
                    timescale: US_TIMESCALE,
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
                },
                ContainerTrack {
                    codec: VideoCodec::Mp3,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(44_100),
                    config: Vec::new(),
                    timescale: US_TIMESCALE,
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
                },
                ContainerTrack {
                    codec: VideoCodec::Opus,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: default_opus_head(2, 0),
                    timescale: US_TIMESCALE,
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
}
