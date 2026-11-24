//! Pure-Rust RIFF/AVI 1.0 demuxer and muxer for lossless video remuxing.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use fastconv_domain::VideoCodec;

use crate::PipelineError;
use crate::video_bitstream::{
    default_opus_head, extract_av1c_from_obu, is_av1_keyframe, unwrap_adts, wrap_adts,
};
use crate::video_container::{
    ContainerMedia, ContainerPacket, ContainerTrack, DemuxedMedia, US_TIMESCALE,
};

const AVIIF_KEYFRAME: u32 = 0x10;
const AVI_HEADER_CAP: u64 = 512 * 1024 * 1024;

fn avi_error(path: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::RemuxFailed {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

fn le_u16(bytes: &[u8]) -> Option<u16> {
    Some(u16::from_le_bytes(bytes.get(..2)?.try_into().ok()?))
}

fn le_u32(bytes: &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?))
}

fn riff_chunk(id: [u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + payload.len() + (payload.len() & 1));
    out.extend_from_slice(&id);
    out.extend_from_slice(
        &u32::try_from(payload.len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    out.extend_from_slice(payload);
    if payload.len() & 1 == 1 {
        out.push(0);
    }
    out
}

fn riff_list(kind: [u8; 4], chunks: &[Vec<u8>]) -> Vec<u8> {
    let mut payload = kind.to_vec();
    for chunk in chunks {
        payload.extend_from_slice(chunk);
    }
    riff_chunk(*b"LIST", &payload)
}

/// Visits RIFF chunks with their payload and the byte offset of the `FourCC`.
fn riff_chunks<'a>(bytes: &'a [u8], mut visit: impl FnMut([u8; 4], &'a [u8], usize)) {
    let mut idx = 0_usize;
    while idx + 8 <= bytes.len() {
        let Ok(id) = <[u8; 4]>::try_from(&bytes[idx..idx + 4]) else {
            return;
        };
        let Some(size) = bytes
            .get(idx + 4..idx + 8)
            .and_then(le_u32)
            .and_then(|v| usize::try_from(v).ok())
        else {
            return;
        };
        let start = idx + 8;
        let Some(end) = start.checked_add(size) else {
            return;
        };
        if end > bytes.len() {
            return;
        }
        visit(id, &bytes[start..end], idx);
        let Some(padded) = size.checked_add(size & 1) else {
            return;
        };
        let Some(next) = start.checked_add(padded) else {
            return;
        };
        if next <= idx {
            return;
        }
        idx = next;
    }
}

fn upper_fourcc(bytes: &[u8]) -> Option<[u8; 4]> {
    let raw = <[u8; 4]>::try_from(bytes.get(..4)?).ok()?;
    Some(raw.map(|byte| byte.to_ascii_uppercase()))
}

fn video_codec(compression: [u8; 4], handler: Option<[u8; 4]>) -> Option<VideoCodec> {
    for fourcc in handler.into_iter().chain(std::iter::once(compression)) {
        match &fourcc {
            b"H264" | b"X264" | b"AVC1" | b"DAVC" => return Some(VideoCodec::H264),
            b"HEVC" | b"H265" | b"HVC1" | b"HEV1" | b"X265" => return Some(VideoCodec::Hevc),
            b"AV01" => return Some(VideoCodec::Av1),
            b"VP90" | b"VP09" => return Some(VideoCodec::Vp9),
            _ => {}
        }
    }
    None
}

fn audio_codec(format_tag: u16, handler: Option<[u8; 4]>) -> Option<VideoCodec> {
    if handler == Some(*b"OPUS") {
        return Some(VideoCodec::Opus);
    }
    match format_tag {
        0x00FF | 0x1600 | 0x1601 | 0x1602 | 0x706D => Some(VideoCodec::Aac),
        0x704F | 0x7361 => Some(VideoCodec::Opus),
        0x0055 => Some(VideoCodec::Mp3),
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct AvStream {
    codec: VideoCodec,
    width: Option<u16>,
    height: Option<u16>,
    channels: Option<u16>,
    sample_rate: Option<u32>,
    config: Vec<u8>,
    duration_us: u64,
    next_pts: u64,
    packets: usize,
}

// The linear RIFF walk is kept in one function: splitting it would require
// passing five mutable accumulators through helpers without adding clarity.
#[allow(clippy::too_many_lines)]
fn parse_header(path: &Path, bytes: &[u8]) -> Result<Vec<AvStream>, PipelineError> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"AVI " {
        return Err(avi_error(path, "truncated RIFF/AVI header"));
    }
    let mut headers = Vec::new();
    let mut strls = Vec::new();
    riff_chunks(&bytes[12..], |id, payload, _| {
        if id != *b"LIST" || payload.len() < 4 {
            return;
        }
        if &payload[..4] == b"hdrl" {
            riff_chunks(&payload[4..], |kind, body, _| {
                if kind == *b"avih" {
                    headers.push(body.to_vec());
                } else if kind == *b"LIST" && body.len() >= 4 && &body[..4] == b"strl" {
                    strls.push(body[4..].to_vec());
                }
            });
        }
    });
    if strls.is_empty() {
        return Err(avi_error(path, "AVI header has no streams"));
    }
    let avih = headers.first().map_or(&[][..], Vec::as_slice);
    let micro_sec_per_frame = le_u32(avih.get(..4).unwrap_or(&[])).unwrap_or(0);
    let avih_width = le_u32(avih.get(32..36).unwrap_or(&[])).unwrap_or(0);
    let avih_height = le_u32(avih.get(36..40).unwrap_or(&[])).unwrap_or(0);
    let mut streams = Vec::with_capacity(strls.len());
    for strl in &strls {
        let mut fcc_type = [0_u8; 4];
        let mut handler = None;
        let mut scale = 1_u32;
        let mut rate = 1_u32;
        let mut strf = None;
        riff_chunks(strl, |kind, body, _| {
            if kind == *b"strh" && body.len() >= 36 {
                fcc_type = <[u8; 4]>::try_from(&body[..4]).unwrap_or([0; 4]);
                handler = upper_fourcc(body.get(4..8).unwrap_or(&[]));
                scale = le_u32(body.get(20..24).unwrap_or(&[])).unwrap_or(1);
                rate = le_u32(body.get(24..28).unwrap_or(&[])).unwrap_or(1);
            } else if kind == *b"strf" {
                strf = Some(body.to_vec());
            }
        });
        let is_video = fcc_type == *b"vids";
        let is_audio = fcc_type == *b"auds";
        if !is_video && !is_audio {
            continue;
        }
        let format = strf.unwrap_or_default();
        if is_video {
            let compression =
                <[u8; 4]>::try_from(format.get(16..20).unwrap_or(&[])).unwrap_or([0; 4]);
            let Some(codec) = video_codec(compression, handler) else {
                continue;
            };
            let width = le_u32(format.get(4..8).unwrap_or(&[]))
                .filter(|&v| v > 0)
                .or((avih_width > 0).then_some(avih_width))
                .and_then(|v| u16::try_from(v).ok());
            let height = le_u32(format.get(8..12).unwrap_or(&[]))
                .filter(|&v| v > 0)
                .or((avih_height > 0).then_some(avih_height))
                .and_then(|v| u16::try_from(v).ok());
            let config = format.get(40..).unwrap_or(&[]).to_vec();
            let duration_us = if micro_sec_per_frame > 0 {
                u64::from(micro_sec_per_frame)
            } else if scale > 0 && rate > 0 {
                u64::from(scale).saturating_mul(1_000_000) / u64::from(rate)
            } else {
                33_333
            };
            streams.push(AvStream {
                codec,
                width,
                height,
                channels: None,
                sample_rate: None,
                config,
                duration_us: duration_us.max(1),
                next_pts: 0,
                packets: 0,
            });
        } else {
            let tag = le_u16(format.get(..2).unwrap_or(&[])).unwrap_or(0);
            let Some(codec) = audio_codec(tag, handler) else {
                continue;
            };
            let channels = le_u16(format.get(2..4).unwrap_or(&[])).filter(|&v| v > 0);
            let sample_rate = le_u32(format.get(4..8).unwrap_or(&[])).filter(|&v| v > 0);
            let extra = format.get(18..).unwrap_or(&[]);
            let config = match codec {
                VideoCodec::Opus if extra.len() >= 19 && &extra[..8] == b"OpusHead" => {
                    extra.to_vec()
                }
                VideoCodec::Opus => default_opus_head(channels.unwrap_or(2), 0),
                _ => extra.to_vec(),
            };
            let samples_per_packet: u64 = match codec {
                VideoCodec::Mp3 => 1152,
                VideoCodec::Opus => 960,
                _ => 1024,
            };
            let duration_us = if scale > 0 && rate > 0 {
                samples_per_packet
                    .saturating_mul(u64::from(scale))
                    .saturating_mul(1_000_000)
                    / u64::from(rate)
            } else {
                1
            };
            streams.push(AvStream {
                codec,
                width: None,
                height: None,
                channels,
                sample_rate,
                config,
                duration_us: duration_us.max(1),
                next_pts: 0,
                packets: 0,
            });
        }
    }
    if streams.is_empty() {
        return Err(avi_error(path, "AVI header has no usable streams"));
    }
    Ok(streams)
}

fn stream_number(id: [u8; 4]) -> Option<usize> {
    let high = (id[0] as char).to_digit(16)?;
    let low = (id[1] as char).to_digit(16)?;
    usize::try_from(high * 16 + low).ok()
}

fn is_annex_b(data: &[u8]) -> bool {
    data.starts_with(&[0, 0, 1]) || data.starts_with(&[0, 0, 0, 1])
}

/// Demuxes a RIFF/AVI file into the canonical representation.
// The movi walk handles seven codec-specific payload conversions inline so
// no per-codec state leaks across helper boundaries.
#[allow(clippy::too_many_lines)]
pub(crate) fn demux_avi(path: &Path) -> Result<DemuxedMedia, PipelineError> {
    let metadata = std::fs::metadata(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.len() > AVI_HEADER_CAP {
        return Err(avi_error(path, "AVI file is too large to demux"));
    }
    let bytes = std::fs::read(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut streams = parse_header(path, &bytes)?;
    let mut movi: &[u8] = &[];
    let mut keyframes = HashMap::new();
    riff_chunks(&bytes[12..], |id, payload, _| {
        if id == *b"LIST" && payload.len() >= 4 && &payload[..4] == b"movi" {
            movi = &payload[4..];
        } else if id == *b"idx1" {
            for entry in payload.as_chunks::<16>().0 {
                let flags = le_u32(entry.get(4..8).unwrap_or(&[])).unwrap_or(0);
                let offset = u64::from(le_u32(entry.get(8..12).unwrap_or(&[])).unwrap_or(0));
                keyframes.insert(offset, flags & AVIIF_KEYFRAME != 0);
            }
        }
    });
    if movi.is_empty() {
        return Err(avi_error(path, "AVI has no movi list"));
    }
    let mut media = ContainerMedia {
        tracks: streams
            .iter()
            .map(|s| ContainerTrack {
                codec: s.codec,
                width: s.width,
                height: s.height,
                channels: s.channels,
                sample_rate: s.sample_rate,
                config: s.config.clone(),
                timescale: US_TIMESCALE,
            })
            .collect(),
        ..ContainerMedia::default()
    };
    let mut chunk_list = Vec::new();
    riff_chunks(movi, |id, payload, offset| {
        chunk_list.push((id, payload, offset));
    });
    for (id, payload, offset) in chunk_list {
        let Some(number) = stream_number(id) else {
            continue;
        };
        if number >= streams.len() {
            continue;
        }
        let keyframe = keyframes
            .get(&u64::try_from(offset).unwrap_or(0))
            .copied()
            .unwrap_or(streams[number].packets == 0);
        match streams[number].codec {
            VideoCodec::H264 | VideoCodec::Hevc => {
                let data = payload;
                let (canonical, sync) = if is_annex_b(data) {
                    let mut proxy = ContainerTrack {
                        codec: streams[number].codec,
                        width: streams[number].width,
                        height: streams[number].height,
                        channels: None,
                        sample_rate: None,
                        config: streams[number].config.clone(),
                        timescale: US_TIMESCALE,
                    };
                    let (canonical, sync) = crate::video_container::video_packet_from_annex_b(
                        &mut proxy, data, keyframe,
                    );
                    streams[number].config = proxy.config;
                    (canonical, sync)
                } else {
                    (data.to_vec(), keyframe)
                };
                push_packet(&mut media, &mut streams, number, canonical, sync);
            }
            VideoCodec::Av1 => {
                if streams[number].config.is_empty()
                    && keyframe
                    && let Some(config) = extract_av1c_from_obu(payload)
                {
                    streams[number].config = config;
                }
                let sync = is_av1_keyframe(payload) || keyframe;
                push_packet(&mut media, &mut streams, number, payload.to_vec(), sync);
            }
            VideoCodec::Vp9 | VideoCodec::Opus | VideoCodec::Mp3 => {
                push_packet(&mut media, &mut streams, number, payload.to_vec(), keyframe);
            }
            VideoCodec::Aac => {
                let frames = if payload.first() == Some(&0xFF)
                    && payload.get(1).is_some_and(|b| b & 0xF0 == 0xF0)
                {
                    unwrap_adts(payload)
                } else {
                    vec![(payload.to_vec(), Vec::new())]
                };
                for (frame, asc) in frames {
                    if streams[number].config.is_empty() && !asc.is_empty() {
                        streams[number].config = asc;
                    }
                    push_packet(&mut media, &mut streams, number, frame, keyframe);
                }
            }
            _ => {}
        }
    }
    for (index, track) in streams.iter().enumerate() {
        media.tracks[index].config.clone_from(&track.config);
    }
    Ok(media)
}

fn push_packet(
    media: &mut ContainerMedia,
    streams: &mut [AvStream],
    number: usize,
    data: Vec<u8>,
    sync: bool,
) {
    let stream = &mut streams[number];
    let pts = i64::try_from(stream.next_pts).unwrap_or(i64::MAX);
    let duration = stream.duration_us.min(i64::MAX as u64);
    stream.next_pts = stream.next_pts.saturating_add(duration);
    stream.packets += 1;
    media.packets.push(ContainerPacket {
        track: number,
        pts,
        dts: pts,
        duration,
        keyframe: sync,
        data,
    });
}

fn video_fourcc(codec: VideoCodec) -> [u8; 4] {
    match codec {
        VideoCodec::H264 => *b"H264",
        VideoCodec::Hevc => *b"HEVC",
        VideoCodec::Av1 => *b"AV01",
        VideoCodec::Vp9 => *b"VP90",
        _ => *b"\0\0\0\0",
    }
}

fn build_waveformatex(
    codec: VideoCodec,
    track: &ContainerTrack,
    path: &Path,
) -> Result<Vec<u8>, PipelineError> {
    let (tag, extra) = match codec {
        VideoCodec::Aac => {
            let asc = track
                .config
                .get(..2)
                .ok_or_else(|| avi_error(path, "missing AAC AudioSpecificConfig"))?;
            (0x00FF_u16, asc.to_vec())
        }
        VideoCodec::Opus => {
            if track.config.len() < 19 || &track.config[..8] != b"OpusHead" {
                return Err(avi_error(path, "invalid OpusHead"));
            }
            (0x704F_u16, track.config.clone())
        }
        VideoCodec::Mp3 => {
            let mut extra = Vec::with_capacity(12);
            extra.extend_from_slice(&1_u16.to_le_bytes());
            extra.extend_from_slice(&2_u32.to_le_bytes());
            extra.extend_from_slice(&576_u16.to_le_bytes());
            extra.extend_from_slice(&1_u16.to_le_bytes());
            extra.extend_from_slice(&0_u16.to_le_bytes());
            (0x0055_u16, extra)
        }
        _ => (0_u16, Vec::new()),
    };
    let rate = track.sample_rate.unwrap_or(48_000);
    let channels = track.channels.unwrap_or(2);
    let block_align = match codec {
        VideoCodec::Mp3 => u16::try_from((rate / 8).max(1)).unwrap_or(1),
        _ => channels.saturating_mul(2),
    };
    let mut out = Vec::with_capacity(18 + extra.len());
    out.extend_from_slice(&tag.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&rate.saturating_mul(u32::from(block_align)).to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16_u16.to_le_bytes());
    out.extend_from_slice(&u16::try_from(extra.len()).unwrap_or(u16::MAX).to_le_bytes());
    out.extend_from_slice(&extra);
    Ok(out)
}

/// Writes a compliant RIFF/AVI 1.0 file with an `idx1` index.
// Header, chunk conversion and index construction form one linear write pass;
// splitting them would only move mutable buffers between helpers.
#[allow(clippy::too_many_lines)]
pub(crate) fn mux_avi(
    target: &File,
    target_path: &Path,
    media: &ContainerMedia,
) -> Result<(), PipelineError> {
    let mut avih = vec![0_u8; 56];
    let video_frames = u32::try_from(
        media
            .packets
            .iter()
            .filter(|p| media.tracks[p.track].codec.is_video())
            .count(),
    )
    .unwrap_or(u32::MAX);
    let first_video = media.tracks.iter().find(|t| t.codec.is_video());
    let video_duration_us = first_video.map_or(33_333, |t| track_duration_us(media, t));
    avih[0..4].copy_from_slice(&video_duration_us.to_le_bytes());
    avih[16..20].copy_from_slice(&video_frames.to_le_bytes());
    avih[32..36]
        .copy_from_slice(&u32::from(first_video.and_then(|t| t.width).unwrap_or(0)).to_le_bytes());
    avih[36..40]
        .copy_from_slice(&u32::from(first_video.and_then(|t| t.height).unwrap_or(0)).to_le_bytes());
    avih[24..28].copy_from_slice(&u32::try_from(media.tracks.len()).unwrap_or(0).to_le_bytes());

    let mut strl_chunks = Vec::new();
    for (index, track) in media.tracks.iter().enumerate() {
        let length = u32::try_from(media.packets.iter().filter(|p| p.track == index).count())
            .unwrap_or(u32::MAX);
        let (scale, rate, handler, strf) = if track.codec.is_video() {
            let scale = track_duration_us(media, track).max(1);
            // `movi` carries Annex-B for H.264/HEVC and raw frames for VP9/AV1,
            // so `strf` stays a clean 40-byte header: extradata here would make
            // decoders interpret those packets as length-prefixed samples.
            let mut bitmap = vec![0_u8; 40];
            bitmap[0..4].copy_from_slice(&40_u32.to_le_bytes());
            bitmap[4..8].copy_from_slice(&i32::from(track.width.unwrap_or(0)).to_le_bytes());
            bitmap[8..12].copy_from_slice(&i32::from(track.height.unwrap_or(0)).to_le_bytes());
            bitmap[12..14].copy_from_slice(&1_u16.to_le_bytes());
            bitmap[14..16].copy_from_slice(&24_u16.to_le_bytes());
            bitmap[16..20].copy_from_slice(&video_fourcc(track.codec));
            (scale, 1_000_000_u32, video_fourcc(track.codec), bitmap)
        } else {
            let handler = if track.codec == VideoCodec::Opus {
                *b"OPUS"
            } else {
                [0; 4]
            };
            let sr = track.sample_rate.unwrap_or(48_000);
            // One scale unit is one sample, so the demuxer's
            // `samples_per_packet * dwScale / dwRate` yields exact frame
            // duration in microseconds.
            (
                1,
                sr,
                handler,
                build_waveformatex(track.codec, track, target_path)?,
            )
        };
        let mut strh_body = vec![0_u8; 56];
        strh_body[0..4].copy_from_slice(if track.codec.is_video() {
            b"vids"
        } else {
            b"auds"
        });
        strh_body[4..8].copy_from_slice(&handler);
        strh_body[20..24].copy_from_slice(&scale.to_le_bytes());
        strh_body[24..28].copy_from_slice(&rate.to_le_bytes());
        strh_body[32..36].copy_from_slice(&length.to_le_bytes());
        strl_chunks.push(riff_list(
            *b"strl",
            &[
                riff_chunk(*b"strh", &strh_body),
                riff_chunk(*b"strf", &strf),
            ],
        ));
    }
    let hdrl = riff_list(
        *b"hdrl",
        &std::iter::once(riff_chunk(*b"avih", &avih))
            .chain(strl_chunks)
            .collect::<Vec<_>>(),
    );

    let mut movi = b"movi".to_vec();
    let mut index = Vec::new();
    for packet in &media.packets {
        let track = &media.tracks[packet.track];
        let number = u8::try_from(packet.track).unwrap_or(u8::MAX);
        let hex = b"0123456789ABCDEF";
        let mut id = [0_u8; 4];
        id[0] = hex[usize::from(number >> 4)];
        id[1] = hex[usize::from(number & 0x0F)];
        id[2..4].copy_from_slice(if track.codec.is_video() { b"dc" } else { b"wb" });
        let payload = if track.codec.is_video() {
            crate::video_container::video_payload_for_annex_b(track, packet)
                .map_err(|reason| avi_error(target_path, reason))?
        } else if track.codec == VideoCodec::Aac {
            let asc = track
                .config
                .get(..2)
                .ok_or_else(|| avi_error(target_path, "missing AAC AudioSpecificConfig"))?;
            wrap_adts(asc, &packet.data).map_err(|reason| avi_error(target_path, reason))?
        } else {
            packet.data.clone()
        };
        let offset = u32::try_from(movi.len().saturating_sub(4)).unwrap_or(u32::MAX);
        let size = u32::try_from(payload.len()).unwrap_or(u32::MAX);
        movi.extend_from_slice(&riff_chunk(id, &payload));
        index.extend_from_slice(&id);
        index.extend_from_slice(&if packet.keyframe { AVIIF_KEYFRAME } else { 0 }.to_le_bytes());
        index.extend_from_slice(&offset.to_le_bytes());
        index.extend_from_slice(&size.to_le_bytes());
    }
    let movi_list = riff_chunk(*b"LIST", &movi);
    let idx1 = riff_chunk(*b"idx1", &index);
    let mut body = b"AVI ".to_vec();
    body.extend_from_slice(&hdrl);
    body.extend_from_slice(&movi_list);
    body.extend_from_slice(&idx1);
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&u32::try_from(body.len()).unwrap_or(u32::MAX).to_le_bytes());
    out.extend_from_slice(&body);

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

fn track_duration_us(media: &ContainerMedia, track: &ContainerTrack) -> u32 {
    let index = media
        .tracks
        .iter()
        .position(|candidate| std::ptr::eq(candidate, track))
        .unwrap_or(0);
    let duration = media
        .packets
        .iter()
        .filter(|p| p.track == index)
        .map(|p| p.duration)
        .max()
        .unwrap_or(33_333);
    u32::try_from(duration).unwrap_or(33_333).max(1)
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
            "fastconv-avi-{label}-{nanos}-{:?}",
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

    fn round_trip(
        label: &str,
        media: &ContainerMedia,
    ) -> Result<ContainerMedia, Box<dyn std::error::Error>> {
        let path = temp_path(label);
        let file = File::create(&path)?;
        mux_avi(&file, &path, media)?;
        drop(file);
        let out = demux_avi(&path)?;
        std::fs::remove_file(&path)?;
        Ok(out)
    }

    #[test]
    fn avi_h264_aac_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        let asc = [0x12, 0x10];
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
                    config: asc.to_vec(),
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
                ContainerPacket {
                    track: 1,
                    pts: 21_000,
                    dts: 21_000,
                    duration: 21_000,
                    keyframe: true,
                    data: vec![0xBB; 8],
                },
            ],
        };
        let out = round_trip("h264-aac", &media)?;
        assert_eq!(out.tracks.len(), 2);
        assert_eq!(out.tracks[0].codec, VideoCodec::H264);
        assert_eq!(out.tracks[0].width, Some(320));
        assert_eq!(out.tracks[1].codec, VideoCodec::Aac);
        assert_eq!(out.tracks[1].sample_rate, Some(48_000));
        assert_eq!(out.packets.len(), 4);
        assert!(out.packets[0].data.ends_with(&media.packets[0].data));
        assert!(out.packets[0].keyframe);
        assert_eq!(out.packets[1].data, vec![0xAA; 8]);
        assert_eq!(out.packets[2].data, media.packets[2].data);
        assert!(!out.packets[2].keyframe);
        Ok(())
    }

    #[test]
    fn avi_hevc_mp3_round_trip() -> Result<(), Box<dyn std::error::Error>> {
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
                    data: vec![0xFF, 0xFB, 0x90, 0x00],
                },
            ],
        };
        let out = round_trip("hevc-mp3", &media)?;
        assert_eq!(out.tracks[0].codec, VideoCodec::Hevc);
        assert_eq!(out.tracks[0].height, Some(360));
        assert_eq!(out.tracks[1].codec, VideoCodec::Mp3);
        assert!(out.packets[0].data.ends_with(&media.packets[0].data));
        assert_eq!(out.packets[1].data, media.packets[1].data);
        Ok(())
    }

    #[test]
    fn avi_vp9_opus_and_av1_aac_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        let media = ContainerMedia {
            tracks: vec![
                ContainerTrack {
                    codec: VideoCodec::Vp9,
                    width: Some(1280),
                    height: Some(720),
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
                    data: vec![0x82, 0x01],
                },
                ContainerPacket {
                    track: 1,
                    pts: 0,
                    dts: 0,
                    duration: 20_000,
                    keyframe: true,
                    data: vec![0xFC, 0x01],
                },
            ],
        };
        let out = round_trip("vp9-opus", &media)?;
        assert_eq!(out.tracks[0].codec, VideoCodec::Vp9);
        assert_eq!(out.tracks[1].codec, VideoCodec::Opus);
        assert_eq!(out.packets[0].data, vec![0x82, 0x01]);
        assert_eq!(out.packets[1].data, vec![0xFC, 0x01]);

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
                    codec: VideoCodec::Aac,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(44_100),
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
                    data: vec![0x0A, 0x02, 0x00, 0x0A],
                },
                ContainerPacket {
                    track: 1,
                    pts: 0,
                    dts: 0,
                    duration: 23_000_000,
                    keyframe: true,
                    data: vec![0x11; 4],
                },
            ],
        };
        let out = round_trip("av1-aac", &media)?;
        assert_eq!(out.tracks[0].codec, VideoCodec::Av1);
        assert_ne!(out.tracks[0].config.len(), 0);
        assert_eq!(out.packets[0].data, vec![0x0A, 0x02, 0x00, 0x0A]);
        Ok(())
    }
}
