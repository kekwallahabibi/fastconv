//! Pure-Rust RIFF/AVI 1.0 demuxer and muxer for lossless video remuxing.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use fastconv_domain::VideoCodec;

use crate::PipelineError;
use crate::video_bitstream::{
    PcmFormat, default_opus_head, extract_av1c_from_obu, is_av1_keyframe, unwrap_adts, wrap_adts,
};
use crate::video_container::{
    ContainerMedia, ContainerPacket, ContainerTrack, DemuxedMedia, US_TIMESCALE,
};

const AVIIF_KEYFRAME: u32 = 0x10;
const AVI_HEADER_CAP: u64 = 512 * 1024 * 1024;
/// Matroska block timestamps are rounded to the track tick (1 ms by default),
/// so PCM boundaries can be off by less than this while still being contiguous.
const PCM_PTS_TOLERANCE_US: i64 = 1_000;

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
            b"MJPG" => return Some(VideoCodec::Mjpeg),
            _ => {}
        }
    }
    None
}

fn audio_codec(
    format_tag: u16,
    bits: Option<u16>,
    handler: Option<[u8; 4]>,
) -> Result<Option<VideoCodec>, String> {
    if handler == Some(*b"OPUS") {
        return Ok(Some(VideoCodec::Opus));
    }
    match format_tag {
        0x00FF | 0x1600 | 0x1601 | 0x1602 | 0x706D => Ok(Some(VideoCodec::Aac)),
        0x704F | 0x7361 => Ok(Some(VideoCodec::Opus)),
        0x0055 => Ok(Some(VideoCodec::Mp3)),
        0x0001 => match bits {
            Some(8 | 16 | 24 | 32) => Ok(Some(VideoCodec::Pcm)),
            Some(bits) => Err(format!("unsupported AVI PCM bit depth {bits}")),
            None => Err("AVI PCM stream is missing wBitsPerSample".to_owned()),
        },
        0x0003 => match bits {
            Some(32 | 64) => Ok(Some(VideoCodec::Pcm)),
            Some(bits) => Err(format!("unsupported AVI IEEE-float bit depth {bits}")),
            None => Err("AVI IEEE-float stream is missing wBitsPerSample".to_owned()),
        },
        _ => Ok(None),
    }
}

#[derive(Clone, Copy)]
struct AvIndexEntry {
    id: [u8; 4],
    offset: u64,
    size: u64,
    keyframe: bool,
}

#[derive(Debug, Clone)]
struct AvStream {
    codec: VideoCodec,
    width: Option<u16>,
    height: Option<u16>,
    channels: Option<u16>,
    sample_rate: Option<u32>,
    config: Vec<u8>,
    /// `dwScale`/`dwRate` are kept as a rational: truncating to whole
    /// microseconds per sample would drift over long streams.
    scale: u64,
    rate: u64,
    /// `dwSampleSize`: zero means one sample per chunk, otherwise the fixed
    /// byte size of a sample and a chunk may carry several of them.
    sample_size: u32,
    /// Exact PCM representation for `VideoCodec::Pcm`, `None` otherwise.
    pcm_format: Option<PcmFormat>,
    samples_emitted: u64,
    packets: usize,
}

impl AvStream {
    /// Cumulative timestamp at AVI sample `n`, rounded half-up from the
    /// `dwScale`/`dwRate` time base.
    fn timestamp_us(&self, samples: u64) -> u64 {
        let rate = u128::from(self.rate).max(1);
        let numerator = u128::from(samples)
            .saturating_mul(u128::from(self.scale))
            .saturating_mul(1_000_000)
            .saturating_add(rate / 2);
        u64::try_from(numerator / rate).unwrap_or(u64::MAX)
    }

    fn chunk_samples(&self, chunk_bytes: usize) -> u64 {
        if self.sample_size == 0 {
            1
        } else {
            u64::try_from(chunk_bytes)
                .ok()
                .and_then(|bytes| bytes.checked_div(u64::from(self.sample_size)))
                .unwrap_or(0)
                .max(1)
        }
    }

    /// Advances this stream by one chunk and returns its `(pts, duration)`.
    fn advance(&mut self, chunk_bytes: usize) -> (u64, u64) {
        let chunk_samples = self.chunk_samples(chunk_bytes);
        let start_us = self.timestamp_us(self.samples_emitted);
        let end_us = self.timestamp_us(self.samples_emitted.saturating_add(chunk_samples));
        self.samples_emitted = self.samples_emitted.saturating_add(chunk_samples);
        self.packets += 1;
        (start_us, end_us.saturating_sub(start_us).max(1))
    }
}

// The linear RIFF walk is kept in one function: splitting it would require
// passing five mutable accumulators through helpers without adding clarity.
#[allow(clippy::too_many_lines)]
fn parse_header(
    path: &Path,
    bytes: &[u8],
) -> Result<(Vec<AvStream>, Vec<Option<usize>>), PipelineError> {
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
    let mut ordinal_map = vec![None; strls.len()];
    for (original_index, strl) in strls.iter().enumerate() {
        let mut fcc_type = [0_u8; 4];
        let mut handler = None;
        let mut scale = 1_u32;
        let mut rate = 1_u32;
        let mut sample_size = 0_u32;
        let mut strf = None;
        riff_chunks(strl, |kind, body, _| {
            if kind == *b"strh" && body.len() >= 36 {
                fcc_type = <[u8; 4]>::try_from(&body[..4]).unwrap_or([0; 4]);
                handler = upper_fourcc(body.get(4..8).unwrap_or(&[]));
                scale = le_u32(body.get(20..24).unwrap_or(&[])).unwrap_or(1);
                rate = le_u32(body.get(24..28).unwrap_or(&[])).unwrap_or(1);
                sample_size = le_u32(body.get(44..48).unwrap_or(&[])).unwrap_or(0);
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
            let compression = upper_fourcc(format.get(16..20).unwrap_or(&[])).unwrap_or([0; 4]);
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
            let (scale, rate) = if micro_sec_per_frame > 0 {
                (u64::from(micro_sec_per_frame), 1_000_000_u64)
            } else if scale > 0 && rate > 0 {
                (u64::from(scale), u64::from(rate))
            } else {
                (33_333_u64, 1_000_000_u64)
            };
            ordinal_map[original_index] = Some(streams.len());
            streams.push(AvStream {
                codec,
                width,
                height,
                channels: None,
                sample_rate: None,
                config,
                scale,
                rate,
                sample_size: 0,
                pcm_format: None,
                samples_emitted: 0,
                packets: 0,
            });
        } else {
            let tag = le_u16(format.get(..2).unwrap_or(&[])).unwrap_or(0);
            let bits = le_u16(format.get(14..16).unwrap_or(&[]));
            let codec =
                audio_codec(tag, bits, handler).map_err(|reason| avi_error(path, reason))?;
            let Some(codec) = codec else {
                continue;
            };
            let channels = le_u16(format.get(2..4).unwrap_or(&[])).filter(|&v| v > 0);
            let sample_rate = le_u32(format.get(4..8).unwrap_or(&[])).filter(|&v| v > 0);
            let pcm_format = if codec == VideoCodec::Pcm {
                if channels.is_none() || sample_rate.is_none() {
                    return Err(avi_error(path, "AVI PCM stream has zero channels or rate"));
                }
                Some(match (tag, bits) {
                    (0x0001, Some(8)) => PcmFormat::U8,
                    (0x0001, Some(16)) => PcmFormat::S16Le,
                    (0x0001, Some(24)) => PcmFormat::S24Le,
                    (0x0001, Some(32)) => PcmFormat::S32Le,
                    (0x0003, Some(32)) => PcmFormat::F32Le,
                    (0x0003, Some(64)) => PcmFormat::F64Le,
                    _ => return Err(avi_error(path, "unsupported AVI PCM format")),
                })
            } else {
                None
            };
            let extra = format.get(18..).unwrap_or(&[]);
            let config = match codec {
                VideoCodec::Opus if extra.len() >= 19 && &extra[..8] == b"OpusHead" => {
                    extra.to_vec()
                }
                VideoCodec::Opus => default_opus_head(channels.unwrap_or(2), 0),
                _ => extra.to_vec(),
            };
            // `dwScale`/`dwRate` describe one AVI sample, not one codec frame.
            // For fixed-size samples the chunk byte count decides how many of
            // them a chunk holds; for VBR there is exactly one per chunk.
            let (scale, rate) = if scale > 0 && rate > 0 {
                (u64::from(scale), u64::from(rate))
            } else {
                (1_u64, 1_000_000_u64)
            };
            ordinal_map[original_index] = Some(streams.len());
            streams.push(AvStream {
                codec,
                width: None,
                height: None,
                channels,
                sample_rate,
                config,
                scale,
                rate,
                sample_size,
                pcm_format,
                samples_emitted: 0,
                packets: 0,
            });
        }
    }
    if streams.is_empty() {
        return Err(avi_error(path, "AVI header has no usable streams"));
    }
    Ok((streams, ordinal_map))
}

fn stream_number(id: [u8; 4]) -> Option<usize> {
    let high = id[0].checked_sub(b'0')?;
    let low = id[1].checked_sub(b'0')?;
    if high > 9 || low > 9 {
        return None;
    }
    Some(usize::from(high) * 10 + usize::from(low))
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
    let (mut streams, ordinal_map) = parse_header(path, &bytes)?;
    let mut movi: &[u8] = &[];
    let mut movi_abs_start = 0_usize;
    let mut index_entries = Vec::new();
    riff_chunks(&bytes[12..], |id, payload, offset| {
        if id == *b"LIST" && payload.len() >= 4 && &payload[..4] == b"movi" {
            movi = &payload[4..];
            movi_abs_start = 12_usize.saturating_add(offset).saturating_add(12);
        } else if id == *b"idx1" {
            for entry in payload.as_chunks::<16>().0 {
                let id = <[u8; 4]>::try_from(entry.get(..4).unwrap_or(&[])).unwrap_or([0; 4]);
                let flags = le_u32(entry.get(4..8).unwrap_or(&[])).unwrap_or(0);
                let offset = u64::from(le_u32(entry.get(8..12).unwrap_or(&[])).unwrap_or(0));
                let size = u64::from(le_u32(entry.get(12..16).unwrap_or(&[])).unwrap_or(0));
                index_entries.push(AvIndexEntry {
                    id,
                    offset,
                    size,
                    keyframe: flags & AVIIF_KEYFRAME != 0,
                });
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
                presentation_edit: None,
                pcm_format: s.pcm_format,
            })
            .collect(),
        ..ContainerMedia::default()
    };
    let mut chunk_list = Vec::new();
    riff_chunks(movi, |id, payload, offset| {
        chunk_list.push((id, payload, offset));
    });
    // Index writers disagree on the base: file-relative, movi-payload-relative
    // or including the four-byte `movi` FourCC. Pick one convention for the
    // whole index, validated against each chunk's ID and size, so a numeric
    // collision never assigns a keyframe flag to the wrong chunk.
    let stored_offset = |kind: usize, offset: usize| -> u64 {
        let relative = u64::try_from(offset).unwrap_or(0);
        let absolute = u64::try_from(movi_abs_start.saturating_add(offset)).unwrap_or(u64::MAX);
        match kind {
            0 => relative,
            1 => relative.saturating_add(4),
            2 => relative.saturating_sub(4),
            3 => absolute,
            4 => absolute.saturating_add(4),
            _ => absolute.saturating_sub(4),
        }
    };
    let mut best_keyframes: Vec<(u64, bool)> = Vec::new();
    for kind in 0..6 {
        let mut by_key: HashMap<(u64, [u8; 4], u64), u64> = HashMap::new();
        for (id, payload, offset) in &chunk_list {
            by_key
                .entry((
                    stored_offset(kind, *offset),
                    *id,
                    u64::try_from(payload.len()).unwrap_or(0),
                ))
                .or_insert(u64::try_from(*offset).unwrap_or(0));
        }
        let resolved: Vec<(u64, bool)> = index_entries
            .iter()
            .filter_map(|entry| {
                by_key
                    .get(&(entry.offset, entry.id, entry.size))
                    .map(|&canonical| (canonical, entry.keyframe))
            })
            .collect();
        if resolved.len() > best_keyframes.len() {
            best_keyframes = resolved;
        }
    }
    let keyframes: HashMap<u64, bool> = best_keyframes.into_iter().collect();
    for (id, payload, offset) in chunk_list {
        let Some(number) =
            stream_number(id).and_then(|original| ordinal_map.get(original).copied().flatten())
        else {
            continue;
        };
        // Every MJPEG chunk is a standalone JPEG image, so no idx1 flag is
        // needed to mark it as a random-access point.
        let keyframe = streams[number].codec == VideoCodec::Mjpeg
            || keyframes
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
                        presentation_edit: None,
                        pcm_format: None,
                    };
                    let (canonical, sync) = crate::video_container::video_packet_from_annex_b(
                        &mut proxy, data, keyframe,
                    );
                    streams[number].config = proxy.config;
                    (canonical, sync)
                } else {
                    (data.to_vec(), keyframe)
                };
                push_packet(
                    &mut media,
                    &mut streams,
                    number,
                    payload.len(),
                    canonical,
                    sync,
                );
            }
            VideoCodec::Av1 => {
                if streams[number].config.is_empty()
                    && keyframe
                    && let Some(config) = extract_av1c_from_obu(payload)
                {
                    streams[number].config = config;
                }
                let sync = is_av1_keyframe(payload) || keyframe;
                push_packet(
                    &mut media,
                    &mut streams,
                    number,
                    payload.len(),
                    payload.to_vec(),
                    sync,
                );
            }
            VideoCodec::Vp9
            | VideoCodec::Mjpeg
            | VideoCodec::Opus
            | VideoCodec::Mp3
            | VideoCodec::Pcm => {
                push_packet(
                    &mut media,
                    &mut streams,
                    number,
                    payload.len(),
                    payload.to_vec(),
                    keyframe,
                );
            }
            VideoCodec::Aac => {
                let frames = if payload.first() == Some(&0xFF)
                    && payload.get(1).is_some_and(|b| b & 0xF0 == 0xF0)
                {
                    unwrap_adts(payload)
                } else {
                    vec![(payload.to_vec(), Vec::new())]
                };
                if frames.is_empty() {
                    continue;
                }
                // The AVI clock counts the original chunk bytes, not the ADTS
                // headers stripped below. Advance it once for the whole chunk.
                let (chunk_pts, total_duration) = streams[number].advance(payload.len());
                let count = frames.len();
                for (index, (frame, asc)) in frames.into_iter().enumerate() {
                    if streams[number].config.is_empty() && !asc.is_empty() {
                        streams[number].config = asc;
                    }
                    let (pts, duration) = if count == 1 {
                        (chunk_pts, total_duration)
                    } else {
                        let n = u128::try_from(count).unwrap_or(1);
                        let seg_start = u128::from(total_duration)
                            .saturating_mul(u128::try_from(index).unwrap_or(0))
                            / n;
                        let seg_end = u128::from(total_duration)
                            .saturating_mul(u128::try_from(index.saturating_add(1)).unwrap_or(0))
                            / n;
                        let seg_start = u64::try_from(seg_start).unwrap_or(u64::MAX);
                        let seg_end = u64::try_from(seg_end).unwrap_or(u64::MAX);
                        (
                            chunk_pts.saturating_add(seg_start),
                            seg_end.saturating_sub(seg_start).max(1),
                        )
                    };
                    push_timed_packet(&mut media, number, pts, duration, frame, keyframe);
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

fn push_timed_packet(
    media: &mut ContainerMedia,
    number: usize,
    pts: u64,
    duration: u64,
    data: Vec<u8>,
    sync: bool,
) {
    let pts = i64::try_from(pts).unwrap_or(i64::MAX);
    let duration = duration.min(i64::MAX as u64);
    media.packets.push(ContainerPacket {
        track: number,
        pts,
        dts: pts,
        duration,
        keyframe: sync,
        data,
    });
}

fn push_packet(
    media: &mut ContainerMedia,
    streams: &mut [AvStream],
    number: usize,
    chunk_bytes: usize,
    data: Vec<u8>,
    sync: bool,
) {
    let (pts, duration) = streams[number].advance(chunk_bytes);
    push_timed_packet(media, number, pts, duration, data, sync);
}

fn video_fourcc(codec: VideoCodec) -> [u8; 4] {
    match codec {
        VideoCodec::H264 => *b"H264",
        VideoCodec::Hevc => *b"HEVC",
        VideoCodec::Av1 => *b"AV01",
        VideoCodec::Vp9 => *b"VP90",
        VideoCodec::Mjpeg => *b"MJPG",
        _ => *b"\0\0\0\0",
    }
}

fn build_waveformatex(
    codec: VideoCodec,
    track: &ContainerTrack,
    path: &Path,
) -> Result<Vec<u8>, PipelineError> {
    let (tag, extra, pcm): (u16, Vec<u8>, Option<PcmFormat>) = match codec {
        VideoCodec::Aac => {
            let asc = track
                .config
                .get(..2)
                .ok_or_else(|| avi_error(path, "missing AAC AudioSpecificConfig"))?;
            (0x00FF_u16, asc.to_vec(), None)
        }
        VideoCodec::Opus => {
            if track.config.len() < 19 || &track.config[..8] != b"OpusHead" {
                return Err(avi_error(path, "invalid OpusHead"));
            }
            (0x704F_u16, track.config.clone(), None)
        }
        VideoCodec::Mp3 => {
            let mut extra = Vec::with_capacity(12);
            extra.extend_from_slice(&1_u16.to_le_bytes());
            extra.extend_from_slice(&2_u32.to_le_bytes());
            extra.extend_from_slice(&576_u16.to_le_bytes());
            extra.extend_from_slice(&1_u16.to_le_bytes());
            extra.extend_from_slice(&0_u16.to_le_bytes());
            (0x0055_u16, extra, None)
        }
        VideoCodec::Pcm => {
            let format = track
                .pcm_format
                .ok_or_else(|| avi_error(path, "PCM track has no sample format"))?;
            if format.is_big_endian() {
                return Err(avi_error(path, "big-endian PCM cannot be stored in AVI"));
            }
            let tag = if format.is_float() {
                0x0003_u16
            } else {
                0x0001
            };
            (tag, Vec::new(), Some(format))
        }
        _ => (0_u16, Vec::new(), None),
    };
    let rate = track.sample_rate.unwrap_or(48_000);
    let channels = track.channels.unwrap_or(2);
    let block_align = match codec {
        VideoCodec::Mp3 => u16::try_from((rate / 8).max(1)).unwrap_or(1),
        _ => match pcm {
            Some(format) => u16::try_from(
                usize::from(channels).saturating_mul(usize::from(format.bytes_per_sample())),
            )
            .unwrap_or(u16::MAX),
            None => channels.saturating_mul(2),
        },
    };
    let bits = pcm.map_or(16, PcmFormat::bit_depth);
    let mut out = Vec::with_capacity(18 + extra.len());
    out.extend_from_slice(&tag.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&rate.saturating_mul(u32::from(block_align)).to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(&u16::try_from(extra.len()).unwrap_or(u16::MAX).to_le_bytes());
    out.extend_from_slice(&extra);
    Ok(out)
}

/// Position of AVI sample `n` in microseconds, using the same half-up rational
/// rounding as the demuxer's `AvStream::timestamp_us` with `dwScale == 1`.
fn avi_pcm_pts(samples: u64, rate: u32) -> i64 {
    let rate = u128::from(rate.max(1));
    let numerator = u128::from(samples)
        .saturating_mul(1_000_000)
        .saturating_add(rate / 2);
    i64::try_from(numerator / rate).unwrap_or(i64::MAX)
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
        let mut length = u32::try_from(media.packets.iter().filter(|p| p.track == index).count())
            .unwrap_or(u32::MAX);
        let mut sample_size = 0_u32;
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
            if track.codec == VideoCodec::Pcm {
                // Uncompressed PCM uses dwScale=1 / dwRate=sample_rate with a
                // fixed dwSampleSize, so `movi` chunks may hold many frames.
                let format = track
                    .pcm_format
                    .ok_or_else(|| avi_error(target_path, "PCM track has no sample format"))?;
                if format.is_big_endian() {
                    return Err(avi_error(
                        target_path,
                        "big-endian PCM cannot be stored in AVI",
                    ));
                }
                let block_align = usize::from(track.channels.unwrap_or(2).max(1))
                    .saturating_mul(usize::from(format.bytes_per_sample()));
                if block_align == 0 {
                    return Err(avi_error(target_path, "PCM block alignment is zero"));
                }
                sample_size = u32::try_from(block_align).unwrap_or(u32::MAX);
                let bytes: usize = media
                    .packets
                    .iter()
                    .filter(|packet| packet.track == index)
                    .map(|packet| packet.data.len())
                    .sum();
                // dwLength counts whole samples; a partial frame here would
                // make the header disagree with the bytes written to `movi`.
                if !bytes.is_multiple_of(block_align) {
                    return Err(avi_error(
                        target_path,
                        "AVI PCM data is not a whole number of frames",
                    ));
                }
                length = u32::try_from(bytes / block_align).unwrap_or(u32::MAX);
                // AVI readers derive PCM timestamps from the cumulative sample
                // count, so a source gap would silently disappear. Reject it
                // before writing the header or any payload instead. Matroska
                // block timestamps are rounded to the track's tick (1 ms by
                // default), so allow that rounding window before calling a
                // boundary a real gap.
                let within_tolerance = |value: i64, expected: i64| {
                    value.saturating_sub(expected).abs() <= PCM_PTS_TOLERANCE_US
                        && expected.saturating_sub(value).abs() <= PCM_PTS_TOLERANCE_US
                };
                let mut samples = 0_u64;
                for pcm in media.packets.iter().filter(|packet| packet.track == index) {
                    let expected = avi_pcm_pts(samples, sr);
                    if !within_tolerance(pcm.pts, expected) || !within_tolerance(pcm.dts, expected)
                    {
                        return Err(avi_error(
                            target_path,
                            "AVI PCM packet timestamps are not contiguous sample positions",
                        ));
                    }
                    samples = samples.saturating_add(
                        u64::try_from(pcm.data.len() / block_align).unwrap_or(u64::MAX),
                    );
                }
                (
                    1_u32,
                    sr,
                    handler,
                    build_waveformatex(track.codec, track, target_path)?,
                )
            } else {
                // One coded frame per chunk: dwScale counts the samples in that
                // frame and dwSampleSize stays zero, the standard AVI VBR layout.
                let samples_per_packet: u32 = match track.codec {
                    VideoCodec::Opus => opus_samples_per_packet(media, index, target_path)?,
                    VideoCodec::Mp3 => {
                        if sr >= 32_000 {
                            1152
                        } else {
                            576
                        }
                    }
                    _ => 1024,
                };
                (
                    samples_per_packet,
                    sr,
                    handler,
                    build_waveformatex(track.codec, track, target_path)?,
                )
            }
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
        // dwSampleSize = 0 marks one coded frame per chunk; PCM is the sole
        // fixed-size layout and stores its frame byte count here.
        strh_body[44..48].copy_from_slice(&sample_size.to_le_bytes());
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
        if packet.track >= 100 {
            return Err(avi_error(target_path, "AVI supports at most 100 streams"));
        }
        let number = u8::try_from(packet.track).unwrap_or(99);
        let mut id = [0_u8; 4];
        id[0] = b'0' + number / 10;
        id[1] = b'0' + number % 10;
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
        } else if track.codec == VideoCodec::Pcm {
            let format = track
                .pcm_format
                .ok_or_else(|| avi_error(target_path, "PCM track has no sample format"))?;
            let block_align = usize::from(track.channels.unwrap_or(2).max(1))
                .saturating_mul(usize::from(format.bytes_per_sample()));
            if block_align == 0 || !packet.data.len().is_multiple_of(block_align) {
                return Err(avi_error(
                    target_path,
                    "AVI PCM chunk is not a whole number of frames",
                ));
            }
            packet.data.clone()
        } else {
            packet.data.clone()
        };
        let offset = u32::try_from(movi.len().saturating_sub(4)).unwrap_or(u32::MAX);
        let size = u32::try_from(payload.len()).unwrap_or(u32::MAX);
        movi.extend_from_slice(&riff_chunk(id, &payload));
        index.extend_from_slice(&id);
        index.extend_from_slice(
            &if packet.keyframe || track.codec == VideoCodec::Mjpeg {
                AVIIF_KEYFRAME
            } else {
                0
            }
            .to_le_bytes(),
        );
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

/// AVI has one constant `dwScale` per stream. Derive it from the actual Opus
/// packet durations so the written header and the demux timeline agree.
fn opus_samples_per_packet(
    media: &ContainerMedia,
    track: usize,
    path: &Path,
) -> Result<u32, PipelineError> {
    let rate = media.tracks[track]
        .sample_rate
        .filter(|&value| value > 0)
        .unwrap_or(48_000);
    let mut count: Option<u32> = None;
    for packet in media.packets.iter().filter(|packet| packet.track == track) {
        let scaled = u128::from(packet.duration)
            .saturating_mul(u128::from(rate))
            .saturating_add(500_000);
        let mut samples = u32::try_from(scaled / 1_000_000).unwrap_or(0);
        let mut from_toc = false;
        // Demuxers that cannot read an Opus TOC leave a 1 µs placeholder. Use
        // the coded duration from the packet itself rather than rejecting it.
        if samples == 0
            && let Some(s48) = crate::video_bitstream::opus_packet_samples_48k(&packet.data)
        {
            let fallback = u128::from(s48)
                .saturating_mul(u128::from(rate))
                .saturating_add(24_000)
                / 48_000;
            samples = u32::try_from(fallback).unwrap_or(0);
            from_toc = samples > 0;
        }
        if samples == 0 {
            return Err(avi_error(path, "Opus packet duration has no AVI sample"));
        }
        if !from_toc {
            let nominal = u128::from(samples) * 1_000_000 / u128::from(rate);
            if u128::from(packet.duration).abs_diff(nominal) > 1 {
                return Err(avi_error(
                    path,
                    "Opus packet duration is not representable by the AVI scale",
                ));
            }
        }
        if count.is_some_and(|expected| expected != samples) {
            return Err(avi_error(path, "Opus packet durations are not constant"));
        }
        count = Some(samples);
    }
    Ok(count.unwrap_or(960))
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
    fn avi_rejects_pcm_chunks_that_are_not_whole_frames() -> Result<(), Box<dyn std::error::Error>>
    {
        let media = ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::Pcm,
                width: None,
                height: None,
                channels: Some(2),
                sample_rate: Some(48_000),
                config: Vec::new(),
                timescale: US_TIMESCALE,
                presentation_edit: None,
                pcm_format: Some(PcmFormat::S16Le),
            }],
            packets: (0..2)
                .map(|index| ContainerPacket {
                    track: 0,
                    pts: i64::from(index) * 1_042,
                    dts: i64::from(index) * 1_042,
                    duration: 1_042,
                    keyframe: true,
                    // 6 bytes is 1.5 stereo S16 frames; the 12-byte total
                    // alone would satisfy the header's whole-frame check.
                    data: vec![0_u8; 6],
                })
                .collect(),
        };
        let path = temp_path("pcm-partial");
        let file = File::create(&path)?;
        let result = mux_avi(&file, &path, &media);
        drop(file);
        std::fs::remove_file(&path)?;
        assert!(
            result.is_err(),
            "each PCM chunk must contain a whole number of frames"
        );
        Ok(())
    }

    #[test]
    fn avi_rejects_pcm_timestamp_gaps() -> Result<(), Box<dyn std::error::Error>> {
        // 480 stereo S16 frames are 10 ms at 48 kHz; the AVI timeline comes
        // from the cumulative sample count, not from the source PTS.
        let media = |second_pts: i64| ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::Pcm,
                width: None,
                height: None,
                channels: Some(2),
                sample_rate: Some(48_000),
                config: Vec::new(),
                timescale: US_TIMESCALE,
                presentation_edit: None,
                pcm_format: Some(PcmFormat::S16Le),
            }],
            packets: [0_i64, second_pts]
                .into_iter()
                .enumerate()
                .map(|(index, pts)| ContainerPacket {
                    track: 0,
                    pts,
                    dts: pts,
                    duration: 10_000,
                    keyframe: true,
                    data: vec![u8::try_from(index).unwrap_or(1); 480 * 4],
                })
                .collect(),
        };
        let path = temp_path("pcm-pts-gap");
        let file = File::create(&path)?;
        let result = mux_avi(&file, &path, &media(1_000_000));
        drop(file);
        std::fs::remove_file(&path)?;
        assert!(result.is_err(), "a 990 ms PCM gap must not be dropped");

        let out = round_trip("pcm-pts-contiguous", &media(10_000))?;
        let pcm: Vec<_> = out
            .packets
            .iter()
            .filter(|packet| out.tracks[packet.track].codec == VideoCodec::Pcm)
            .collect();
        assert_eq!(pcm.len(), 2);
        assert_eq!(pcm[0].pts, 0);
        assert_eq!(pcm[1].pts, 10_000, "contiguous PCM must be unchanged");
        Ok(())
    }

    #[test]
    fn avi_accepts_pcm_timestamps_rounded_to_source_tick() -> Result<(), Box<dyn std::error::Error>>
    {
        // Matroska stores block timestamps in 1 ms ticks; a 1024-frame packet
        // at 48 kHz lands on 21,000 µs rather than the exact 21,333 µs.
        let media = ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::Pcm,
                width: None,
                height: None,
                channels: Some(2),
                sample_rate: Some(48_000),
                config: Vec::new(),
                timescale: US_TIMESCALE,
                presentation_edit: None,
                pcm_format: Some(PcmFormat::S16Le),
            }],
            packets: [0_i64, 21_000, 43_000, 64_000]
                .into_iter()
                .map(|pts| ContainerPacket {
                    track: 0,
                    pts,
                    dts: pts,
                    duration: 21_333,
                    keyframe: true,
                    data: vec![0_u8; 4 * 1_024],
                })
                .collect(),
        };
        let path = temp_path("pcm-rounded-pts");
        let file = File::create(&path)?;
        let result = mux_avi(&file, &path, &media);
        drop(file);
        std::fs::remove_file(&path)?;
        assert!(
            result.is_ok(),
            "source-tick rounding is not a timing gap: {result:?}"
        );
        Ok(())
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
                    presentation_edit: None,
                    pcm_format: None,
                },
                ContainerTrack {
                    codec: VideoCodec::Aac,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: asc.to_vec(),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                    pcm_format: None,
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
                    presentation_edit: None,
                    pcm_format: None,
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
                    pcm_format: None,
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
                    presentation_edit: None,
                    pcm_format: None,
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
                    pcm_format: None,
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
                    presentation_edit: None,
                    pcm_format: None,
                },
                ContainerTrack {
                    codec: VideoCodec::Aac,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(44_100),
                    config: vec![0x12, 0x10],
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                    pcm_format: None,
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

    #[test]
    fn avi_file_relative_idx1_offsets_preserve_later_vp9_keyframes()
    -> Result<(), Box<dyn std::error::Error>> {
        let media = ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::Vp9,
                width: Some(320),
                height: Some(240),
                channels: None,
                sample_rate: None,
                config: Vec::new(),
                timescale: US_TIMESCALE,
                presentation_edit: None,
                pcm_format: None,
            }],
            packets: [true, false, true]
                .into_iter()
                .enumerate()
                .map(|(index, keyframe)| ContainerPacket {
                    track: 0,
                    pts: i64::try_from(index).unwrap_or(0) * 33_333,
                    dts: i64::try_from(index).unwrap_or(0) * 33_333,
                    duration: 33_333,
                    keyframe,
                    data: vec![0x82, u8::try_from(index).unwrap_or(0)],
                })
                .collect(),
        };
        let path = temp_path("vp9-file-idx1");
        let file = File::create(&path)?;
        mux_avi(&file, &path, &media)?;
        drop(file);
        let mut bytes = std::fs::read(&path)?;
        let movi_pos = bytes
            .windows(4)
            .position(|window| window == b"movi".as_slice())
            .ok_or("movi list")?;
        let base = movi_pos.saturating_add(4);
        let idx_pos = bytes
            .windows(4)
            .position(|window| window == b"idx1".as_slice())
            .ok_or("idx1 chunk")?;
        let size = usize::try_from(
            le_u32(bytes.get(idx_pos + 4..idx_pos + 8).ok_or("idx1 size")?).ok_or("idx1 size")?,
        )?;
        let start = idx_pos + 8;
        for entry in bytes
            .get_mut(start..start + size)
            .ok_or("idx1 payload")?
            .as_chunks_mut::<16>()
            .0
        {
            let offset = le_u32(entry.get(8..12).ok_or("offset")?).ok_or("offset")?;
            entry[8..12].copy_from_slice(&offset.wrapping_add(u32::try_from(base)?).to_le_bytes());
        }
        std::fs::write(&path, &bytes)?;
        let out = demux_avi(&path)?;
        std::fs::remove_file(&path)?;
        let flags: Vec<bool> = out.packets.iter().map(|packet| packet.keyframe).collect();
        assert_eq!(flags, vec![true, false, true]);
        Ok(())
    }

    #[test]
    fn avi_unsupported_strl_does_not_shift_stream_numbers() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut unsupported_header = vec![0_u8; 56];
        unsupported_header[0..4].copy_from_slice(b"txts");
        let unsupported = riff_list(
            *b"strl",
            &[
                riff_chunk(*b"strh", &unsupported_header),
                riff_chunk(*b"strf", &[0; 4]),
            ],
        );
        let mut video_header = vec![0_u8; 56];
        video_header[0..4].copy_from_slice(b"vids");
        video_header[4..8].copy_from_slice(b"H264");
        video_header[20..24].copy_from_slice(&33_333_u32.to_le_bytes());
        video_header[24..28].copy_from_slice(&1_000_000_u32.to_le_bytes());
        let mut format = vec![0_u8; 40];
        format[4..8].copy_from_slice(&320_u32.to_le_bytes());
        format[8..12].copy_from_slice(&240_u32.to_le_bytes());
        format[16..20].copy_from_slice(b"H264");
        let video = riff_list(
            *b"strl",
            &[
                riff_chunk(*b"strh", &video_header),
                riff_chunk(*b"strf", &format),
            ],
        );
        let mut avih = vec![0_u8; 56];
        avih[0..4].copy_from_slice(&33_333_u32.to_le_bytes());
        avih[16..20].copy_from_slice(&1_u32.to_le_bytes());
        avih[24..28].copy_from_slice(&1_u32.to_le_bytes());
        avih[32..36].copy_from_slice(&320_u32.to_le_bytes());
        avih[36..40].copy_from_slice(&240_u32.to_le_bytes());
        let hdrl = riff_list(*b"hdrl", &[riff_chunk(*b"avih", &avih), unsupported, video]);
        let mut movi = b"movi".to_vec();
        movi.extend_from_slice(&riff_chunk(*b"01dc", &[0, 0, 0, 2, 0x65, 0x88]));
        let body = [
            b"AVI ".as_slice(),
            hdrl.as_slice(),
            riff_chunk(*b"LIST", &movi).as_slice(),
        ]
        .concat();
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&u32::try_from(body.len())?.to_le_bytes());
        bytes.extend_from_slice(&body);
        let path = temp_path("strl-ordinal");
        std::fs::write(&path, &bytes)?;
        let out = demux_avi(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(out.tracks.len(), 1);
        assert_eq!(out.packets.len(), 1);
        assert_eq!(out.packets[0].track, 0);
        Ok(())
    }

    #[test]
    fn avi_audio_frame_duration_survives_round_trip() -> Result<(), Box<dyn std::error::Error>> {
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
                pcm_format: None,
            }],
            packets: (0..47)
                .map(|index| ContainerPacket {
                    track: 0,
                    pts: i64::from(index) * 21_333,
                    dts: i64::from(index) * 21_333,
                    duration: 21_333,
                    keyframe: true,
                    data: vec![0xAA; 8],
                })
                .collect(),
        };
        let out = round_trip("audio-duration", &media)?;
        let total: u64 = out.packets.iter().map(|packet| packet.duration).sum();
        assert!(
            total.abs_diff(1_000_000) <= 5_000,
            "total audio duration {total} µs"
        );

        let path = temp_path("audio-header");
        let file = File::create(&path)?;
        mux_avi(&file, &path, &media)?;
        drop(file);
        let bytes = std::fs::read(&path)?;
        std::fs::remove_file(&path)?;
        let strh = bytes
            .windows(4)
            .position(|window| window == b"strh".as_slice())
            .ok_or("strh")?;
        let body = strh + 8;
        assert_eq!(
            le_u32(bytes.get(body + 20..body + 24).ok_or("scale")?),
            Some(1024)
        );
        assert_eq!(
            le_u32(bytes.get(body + 24..body + 28).ok_or("rate")?),
            Some(48_000)
        );
        assert_eq!(
            le_u32(bytes.get(body + 44..body + 48).ok_or("sample size")?),
            Some(0)
        );
        Ok(())
    }

    #[test]
    fn avi_fixed_size_mp3_chunks_use_chunk_bytes_for_duration()
    -> Result<(), Box<dyn std::error::Error>> {
        let media = ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::Mp3,
                width: None,
                height: None,
                channels: Some(2),
                sample_rate: Some(44_100),
                config: Vec::new(),
                timescale: US_TIMESCALE,
                presentation_edit: None,
                pcm_format: None,
            }],
            packets: (0..2)
                .map(|index| ContainerPacket {
                    track: 0,
                    pts: i64::from(index) * 26_122,
                    dts: i64::from(index) * 26_122,
                    duration: 26_122,
                    keyframe: true,
                    data: {
                        let mut frame = vec![0xFF, 0xFB, 0x90, 0x00];
                        frame.resize(417, 0);
                        frame
                    },
                })
                .collect(),
        };
        let path = temp_path("fixed-mp3");
        let file = File::create(&path)?;
        mux_avi(&file, &path, &media)?;
        drop(file);
        let mut bytes = std::fs::read(&path)?;
        // One fixed-size 417-byte AVI sample per chunk: dwScale/dwRate must
        // apply once per chunk, not once per 1152 codec samples.
        let strh = bytes
            .windows(4)
            .position(|window| window == b"strh".as_slice())
            .ok_or("strh")?
            + 8;
        bytes[strh + 44..strh + 48].copy_from_slice(&417_u32.to_le_bytes());
        std::fs::write(&path, &bytes)?;
        let out = demux_avi(&path)?;
        std::fs::remove_file(&path)?;
        let expected = 1_152_u64 * 1_000_000 / 44_100;
        assert_eq!(out.packets.len(), 2);
        for packet in &out.packets {
            assert!(
                packet.duration.abs_diff(expected) <= 1,
                "duration {} µs, expected {} µs",
                packet.duration,
                expected
            );
        }
        assert!(out.packets[1].pts - out.packets[0].pts < 100_000);
        Ok(())
    }

    #[test]
    fn avi_idx1_uses_one_offset_base_when_numbers_overlap() -> Result<(), Box<dyn std::error::Error>>
    {
        let make = |sizes: [usize; 3]| ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::Vp9,
                width: Some(320),
                height: Some(240),
                channels: None,
                sample_rate: None,
                config: Vec::new(),
                timescale: US_TIMESCALE,
                presentation_edit: None,
                pcm_format: None,
            }],
            packets: sizes
                .into_iter()
                .enumerate()
                .map(|(index, size)| {
                    let mut data = vec![0x82, u8::try_from(index).unwrap_or(0)];
                    data.resize(size, 0);
                    ContainerPacket {
                        track: 0,
                        pts: i64::try_from(index).unwrap_or(0) * 33_333,
                        dts: i64::try_from(index).unwrap_or(0) * 33_333,
                        duration: 33_333,
                        keyframe: index != 1,
                        data,
                    }
                })
                .collect(),
        };
        let probe = temp_path("idx1-base");
        let file = File::create(&probe)?;
        mux_avi(&file, &probe, &make([16, 16, 16]))?;
        drop(file);
        let bytes = std::fs::read(&probe)?;
        std::fs::remove_file(&probe)?;
        let movi_pos = bytes
            .windows(4)
            .position(|window| window == b"movi".as_slice())
            .ok_or("movi")?;
        let base = movi_pos.saturating_add(4);

        // Make the second chunk's movi-relative offset equal the first chunk's
        // file-relative offset so both index conventions collide numerically.
        let mut first_len = base.saturating_sub(8);
        if first_len % 2 == 1 {
            first_len = first_len.saturating_sub(1);
        }
        let path = temp_path("idx1-overlap");
        let file = File::create(&path)?;
        mux_avi(&file, &path, &make([first_len, 16, 16]))?;
        drop(file);
        let mut bytes = std::fs::read(&path)?;
        let idx_pos = bytes
            .windows(4)
            .position(|window| window == b"idx1".as_slice())
            .ok_or("idx1")?;
        let size = usize::try_from(
            le_u32(bytes.get(idx_pos + 4..idx_pos + 8).ok_or("idx1 size")?).ok_or("idx1 size")?,
        )?;
        for entry in bytes
            .get_mut(idx_pos + 8..idx_pos + 8 + size)
            .ok_or("idx1 payload")?
            .as_chunks_mut::<16>()
            .0
        {
            let offset = le_u32(entry.get(8..12).ok_or("offset")?).ok_or("offset")?;
            entry[8..12].copy_from_slice(&offset.wrapping_add(u32::try_from(base)?).to_le_bytes());
        }
        std::fs::write(&path, &bytes)?;
        let out = demux_avi(&path)?;
        std::fs::remove_file(&path)?;
        let flags: Vec<bool> = out.packets.iter().map(|packet| packet.keyframe).collect();
        assert_eq!(flags, vec![true, false, true]);
        Ok(())
    }

    #[test]
    fn avi_ten_ms_opus_packets_derive_scale_from_durations()
    -> Result<(), Box<dyn std::error::Error>> {
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
                pcm_format: None,
            }],
            packets: (0..3)
                .map(|index| ContainerPacket {
                    track: 0,
                    pts: i64::from(index) * 10_000,
                    dts: i64::from(index) * 10_000,
                    duration: 10_000,
                    keyframe: true,
                    // 0xF0 is a 10 ms CELT TOC; 0xFC would declare 20 ms.
                    data: vec![0xF0, u8::try_from(index).unwrap_or(0)],
                })
                .collect(),
        };
        let path = temp_path("opus-10ms");
        let file = File::create(&path)?;
        mux_avi(&file, &path, &media)?;
        drop(file);
        let bytes = std::fs::read(&path)?;
        let strh = bytes
            .windows(4)
            .position(|window| window == b"strh".as_slice())
            .ok_or("strh")?
            + 8;
        assert_eq!(
            le_u32(bytes.get(strh + 20..strh + 24).ok_or("scale")?),
            Some(480)
        );
        assert_eq!(
            le_u32(bytes.get(strh + 24..strh + 28).ok_or("rate")?),
            Some(48_000)
        );
        let out = demux_avi(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(out.packets.len(), 3);
        assert!(out.packets.iter().all(|packet| packet.duration == 10_000));
        assert_eq!(out.packets[2].pts, 20_000);

        let mut variable = media;
        variable.packets[1].duration = 20_000;
        let path = temp_path("opus-variable");
        let file = File::create(&path)?;
        assert!(mux_avi(&file, &path, &variable).is_err());
        drop(file);
        std::fs::remove_file(&path)?;
        Ok(())
    }

    /// Builds a minimal AVI with one audio stream and caller-supplied chunks so
    /// tests can exercise `dwSampleSize` layouts the muxer itself never writes.
    fn avi_with_raw_audio_chunks(
        label: &str,
        scale: u32,
        rate: u32,
        sample_size: u32,
        chunks: &[Vec<u8>],
    ) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
        let mut avih = vec![0_u8; 56];
        avih[24..28].copy_from_slice(&1_u32.to_le_bytes());
        let mut strh = vec![0_u8; 56];
        strh[..4].copy_from_slice(b"auds");
        strh[20..24].copy_from_slice(&scale.to_le_bytes());
        strh[24..28].copy_from_slice(&rate.to_le_bytes());
        strh[32..36].copy_from_slice(&u32::try_from(chunks.len())?.to_le_bytes());
        strh[44..48].copy_from_slice(&sample_size.to_le_bytes());
        let mut wave_format = vec![0_u8; 40];
        wave_format[..2].copy_from_slice(&0x00FF_u16.to_le_bytes());
        wave_format[2..4].copy_from_slice(&2_u16.to_le_bytes());
        wave_format[4..8].copy_from_slice(&48_000_u32.to_le_bytes());
        let hdrl = riff_list(
            *b"hdrl",
            &[
                riff_chunk(*b"avih", &avih),
                riff_list(
                    *b"strl",
                    &[
                        riff_chunk(*b"strh", &strh),
                        riff_chunk(*b"strf", &wave_format),
                    ],
                ),
            ],
        );
        let mut movi = b"movi".to_vec();
        for chunk in chunks {
            movi.extend_from_slice(&riff_chunk(*b"00wb", chunk));
        }
        let mut body = b"AVI ".to_vec();
        body.extend_from_slice(&hdrl);
        body.extend_from_slice(&riff_chunk(*b"LIST", &movi));
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&u32::try_from(body.len())?.to_le_bytes());
        bytes.extend_from_slice(&body);
        let path = temp_path(label);
        std::fs::write(&path, &bytes)?;
        Ok(path)
    }

    #[test]
    fn avi_rational_scale_avoids_cumulative_truncation() -> Result<(), Box<dyn std::error::Error>> {
        // 160 chunks of 100 one-byte samples at dwScale=1/dwRate=16000 is
        // exactly one second (62.5 µs per sample).
        let chunks: Vec<Vec<u8>> = std::iter::repeat_n(vec![0xAA; 100], 160).collect();
        let path = avi_with_raw_audio_chunks("rational-scale", 1, 16_000, 1, &chunks)?;
        let out = demux_avi(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(out.packets.len(), 160);
        let total: u64 = out.packets.iter().map(|packet| packet.duration).sum();
        assert_eq!(total, 1_000_000, "truncated timeline {total} µs");
        assert_eq!(
            out.packets[159].pts + i64::try_from(out.packets[159].duration)?,
            1_000_000
        );

        // With dwSampleSize=0 every chunk is a single AVI sample: 62.5 µs
        // rounds half-up to 63 µs, and the next chunk starts there.
        let path = avi_with_raw_audio_chunks("single-sample", 1, 16_000, 0, &chunks[..2])?;
        let out = demux_avi(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(out.packets.len(), 2);
        assert_eq!(out.packets[0].duration, 63);
        assert_eq!(out.packets[1].pts, 63);
        Ok(())
    }

    #[test]
    fn avi_aac_chunk_timing_uses_original_adts_bytes() -> Result<(), Box<dyn std::error::Error>> {
        let asc = [0x12, 0x10];
        let frame =
            crate::video_bitstream::wrap_adts(&asc, &[0xAA; 100]).map_err(std::io::Error::other)?;
        assert_eq!(frame.len(), 107);
        let mut first_chunk = frame.clone();
        first_chunk.extend_from_slice(&frame);
        assert_eq!(first_chunk.len(), 214);
        let path =
            avi_with_raw_audio_chunks("aac-original-chunk", 1, 16_000, 1, &[first_chunk, frame])?;
        let out = demux_avi(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(out.packets.len(), 3);
        assert_eq!(out.packets[0].pts, 0);
        assert_eq!(out.packets[1].pts, 6_687);
        assert_eq!(out.packets[0].duration + out.packets[1].duration, 13_375);
        // The second AVI chunk starts after the full 214-byte chunk, not the
        // 200 header-stripped bytes of its two ADTS frames.
        assert_eq!(out.packets[2].pts, 13_375);
        Ok(())
    }

    #[test]
    fn avi_single_opus_packet_with_placeholder_duration_uses_toc()
    -> Result<(), Box<dyn std::error::Error>> {
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
                pcm_format: None,
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
        let path = temp_path("opus-toc-fallback");
        let file = File::create(&path)?;
        mux_avi(&file, &path, &media)?;
        drop(file);
        let bytes = std::fs::read(&path)?;
        let strh = bytes
            .windows(4)
            .position(|window| window == b"strh".as_slice())
            .ok_or("strh")?
            + 8;
        assert_eq!(
            le_u32(bytes.get(strh + 20..strh + 24).ok_or("scale")?),
            Some(960)
        );
        assert_eq!(
            le_u32(bytes.get(strh + 24..strh + 28).ok_or("rate")?),
            Some(48_000)
        );
        let out = demux_avi(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(out.packets.len(), 1);
        assert_eq!(out.packets[0].duration, 20_000);

        let mut variable = media;
        variable.packets.push(ContainerPacket {
            track: 0,
            pts: 20_000,
            dts: 20_000,
            duration: 10_000,
            keyframe: true,
            data: vec![0xF0; 20],
        });
        let file = File::create(&path)?;
        assert!(mux_avi(&file, &path, &variable).is_err());
        drop(file);
        std::fs::remove_file(&path)?;
        Ok(())
    }
}
