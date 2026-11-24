//! Unified pure-Rust container demux/mux for lossless video remuxing.
//!
//! The internal representation uses exact microsecond timestamps, MP4-style
//! codec configuration records and length-prefixed H.264/HEVC samples. AVI and
//! MPEG-TS translate to and from that canonical form; MP4/MOV/MKV/WebM speak
//! it natively through `containr`.

use std::fs::File;
use std::io::{self, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;

use containr::matroska::{MatroskaAudio, MatroskaVideo};
use containr::mkv::{DocType, MkvOptions, TrackSpec as MkvTrackSpec};
use containr::mp4::{
    AudioCodec, Brand, MuxOptions, SampleEntrySpec, TrackSpec as Mp4TrackSpec, VisualCodec,
};
use containr::{CodecConfig, EditEntry, EditList, Timebase, Track, TrackKind};
use fastconv_domain::{VideoCodec, VideoContainerFormat};

use crate::PipelineError;
use crate::video_bitstream::{
    annex_b_to_avcc, annex_b_to_hvcc, avcc_to_annex_b, build_aac_esds, build_mp3_mp4a_sample_entry,
    build_vp09_sample_entry, dops_to_opus_head, extract_av1c_from_obu, extract_avcc_from_annex_b,
    extract_hvcc_from_annex_b, hvcc_to_annex_b, is_av1_keyframe, is_h264_keyframe_annex_b,
    is_hevc_keyframe_annex_b, opus_head_to_dops,
};

/// Canonical timestamp unit: exact microseconds, i.e. one tick is 1 µs and
/// there are [`US_TIMESCALE`] ticks per second.
pub(crate) const US_TIMESCALE: u32 = 1_000_000;
/// One microsecond expressed in nanoseconds — the Matroska `TimestampScale`
/// and the unit of Matroska `DefaultDuration`.
pub(crate) const MKV_TICK_NS: u64 = 1_000;

/// Above this estimated output size MP4 faststart is done in place on the
/// target instead of staging the whole movie in RAM.
const IN_MEMORY_MP4_FASTSTART_LIMIT: u64 = 64 * 1024 * 1024;

/// One decoded track in the canonical representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ContainerTrack {
    pub(crate) codec: VideoCodec,
    pub(crate) width: Option<u16>,
    pub(crate) height: Option<u16>,
    pub(crate) channels: Option<u16>,
    pub(crate) sample_rate: Option<u32>,
    /// `avcC`/`hvcC`/`av1C`/`vpcC` for video, `AudioSpecificConfig` for AAC,
    /// `OpusHead` for Opus, empty for MP3.
    pub(crate) config: Vec<u8>,
    /// Always [`US_TIMESCALE`]; kept explicit so AVI/TS can rescale.
    pub(crate) timescale: u32,
}

/// One coded frame with microsecond timestamps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ContainerPacket {
    pub(crate) track: usize,
    pub(crate) pts: i64,
    pub(crate) dts: i64,
    pub(crate) duration: u64,
    pub(crate) keyframe: bool,
    pub(crate) data: Vec<u8>,
}

/// A fully demuxed, canonical media graph.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct ContainerMedia {
    pub(crate) tracks: Vec<ContainerTrack>,
    pub(crate) packets: Vec<ContainerPacket>,
}

/// Alias used by the AVI and MPEG-TS demuxers.
pub(crate) type DemuxedMedia = ContainerMedia;

fn container_error(path: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::RemuxFailed {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

fn containr_error(path: &Path, err: &containr::Error) -> PipelineError {
    container_error(path, format!("container error: {err}"))
}

/// Rescales `value` ticks of a `num / den` seconds timebase into rounded
/// microseconds, the canonical [`ContainerMedia`] timestamp unit.
pub(crate) fn rescale_ticks(value: i64, num: u32, den: u32) -> i64 {
    let numerator = i128::from(value)
        .saturating_mul(i128::from(num))
        .saturating_mul(i128::from(US_TIMESCALE));
    let denominator = i128::from(den.max(1));
    let rounded = if numerator >= 0 {
        numerator.saturating_add(denominator / 2) / denominator
    } else {
        numerator.saturating_sub(denominator / 2) / denominator
    };
    i64::try_from(rounded).unwrap_or(if numerator >= 0 { i64::MAX } else { i64::MIN })
}

/// A windowed reader that keeps containr's absolute seeks cheap.
struct WindowedFileReader {
    file: File,
    file_len: u64,
    pos: u64,
    buf: Vec<u8>,
    buf_start: u64,
    buf_valid: usize,
}

impl WindowedFileReader {
    fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let capacity = usize::try_from(file_len.min(1024 * 1024)).unwrap_or(1024 * 1024);
        Ok(Self {
            file,
            file_len,
            pos: 0,
            buf: vec![0_u8; capacity],
            buf_start: 0,
            buf_valid: 0,
        })
    }
}

impl Seek for WindowedFileReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match pos {
            SeekFrom::Start(offset) => {
                self.pos = offset;
                Ok(self.pos)
            }
            SeekFrom::Current(delta) => {
                self.pos = self.pos.checked_add_signed(delta).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "seek before start of file")
                })?;
                Ok(self.pos)
            }
            SeekFrom::End(delta) => {
                self.pos = self.file_len.checked_add_signed(delta).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "seek before start of file")
                })?;
                Ok(self.pos)
            }
        }
    }
}

impl Read for WindowedFileReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() || self.pos >= self.file_len {
            return Ok(0);
        }
        if let Some(rel) = self.pos.checked_sub(self.buf_start)
            && let Ok(rel_usize) = usize::try_from(rel)
            && rel_usize < self.buf_valid
        {
            let count = out.len().min(self.buf_valid - rel_usize);
            out[..count].copy_from_slice(&self.buf[rel_usize..rel_usize + count]);
            self.pos = self
                .pos
                .saturating_add(u64::try_from(count).unwrap_or(u64::MAX));
            return Ok(count);
        }
        if out.len() >= self.buf.len() {
            self.file.seek(SeekFrom::Start(self.pos))?;
            let count = self.file.read(out)?;
            self.pos = self
                .pos
                .saturating_add(u64::try_from(count).unwrap_or(u64::MAX));
            return Ok(count);
        }
        self.file.seek(SeekFrom::Start(self.pos))?;
        // A partial read failure must never expose bytes from the old window.
        self.buf_valid = 0;
        let mut filled = 0_usize;
        while filled < self.buf.len() {
            match self.file.read(&mut self.buf[filled..]) {
                Ok(0) => break,
                Ok(count) => filled += count,
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) => return Err(err),
            }
        }
        self.buf_start = self.pos;
        self.buf_valid = filled;
        if filled == 0 {
            return Ok(0);
        }
        let count = out.len().min(filled);
        out[..count].copy_from_slice(&self.buf[..count]);
        self.pos = self
            .pos
            .saturating_add(u64::try_from(count).unwrap_or(u64::MAX));
        Ok(count)
    }
}

fn esds_object_type(esds: &[u8]) -> Option<u8> {
    let body = esds.get(4..)?;
    let es = mpeg4_descriptor(body, 0x03)?;
    let flags = *es.get(2)?;
    let mut idx = 3_usize;
    if flags & 0x80 != 0 {
        idx = idx.checked_add(2)?;
    }
    if flags & 0x40 != 0 {
        let url_len = usize::from(*es.get(idx)?);
        idx = idx.checked_add(1)?.checked_add(url_len)?;
    }
    if flags & 0x20 != 0 {
        idx = idx.checked_add(2)?;
    }
    mpeg4_descriptor(es.get(idx..)?, 0x04)?.first().copied()
}

fn mpeg4_descriptor(bytes: &[u8], tag: u8) -> Option<&[u8]> {
    let mut idx = 0_usize;
    while idx < bytes.len() {
        let found = bytes[idx];
        idx += 1;
        // MPEG-4 descriptor sizes are big-endian base-128: each continuation
        // byte precedes the final byte, so accumulate with a left shift.
        let mut parts = Vec::with_capacity(4);
        loop {
            let byte = *bytes.get(idx)?;
            idx += 1;
            parts.push(byte & 0x7F);
            if byte & 0x80 == 0 {
                break;
            }
            if parts.len() >= 4 {
                return None;
            }
        }
        let mut size = 0_usize;
        for part in parts {
            size = size.checked_shl(7)?.checked_add(usize::from(part))?;
        }
        let end = idx.checked_add(size)?;
        if end > bytes.len() {
            return None;
        }
        if found == tag {
            return Some(&bytes[idx..end]);
        }
        idx = end;
    }
    None
}

fn esds_asc(esds: &[u8]) -> Option<Vec<u8>> {
    let body = esds.get(4..)?;
    let es = mpeg4_descriptor(body, 0x03)?;
    let dcd = {
        let flags = *es.get(2)?;
        let mut idx = 3_usize;
        if flags & 0x80 != 0 {
            idx = idx.checked_add(2)?;
        }
        if flags & 0x40 != 0 {
            let url_len = usize::from(*es.get(idx)?);
            idx = idx.checked_add(1)?.checked_add(url_len)?;
        }
        if flags & 0x20 != 0 {
            idx = idx.checked_add(2)?;
        }
        mpeg4_descriptor(es.get(idx..)?, 0x04)?
    };
    mpeg4_descriptor(dcd.get(13..)?, 0x05).map(<[u8]>::to_vec)
}

fn mp4_codec(track: &Track) -> Option<VideoCodec> {
    if let Some(fourcc) = track.codec() {
        return match &fourcc.0 {
            b"avc1" | b"avc3" => Some(VideoCodec::H264),
            b"hvc1" | b"hev1" => Some(VideoCodec::Hevc),
            b"av01" => Some(VideoCodec::Av1),
            b"vp09" => Some(VideoCodec::Vp9),
            b"Opus" => Some(VideoCodec::Opus),
            b".mp3" => Some(VideoCodec::Mp3),
            b"mp4a" => {
                let object = track
                    .sample_entry()
                    .and_then(|entry| entry.codec_config.as_ref())
                    .and_then(|config| match config {
                        CodecConfig::Esds(esds) => esds_object_type(esds),
                        _ => None,
                    });
                match object {
                    Some(0x69 | 0x6B) => Some(VideoCodec::Mp3),
                    _ => Some(VideoCodec::Aac),
                }
            }
            _ => None,
        };
    }
    match track.codec_id()? {
        id if id.contains("AVC") => Some(VideoCodec::H264),
        id if id.contains("HEVC") => Some(VideoCodec::Hevc),
        "V_AV1" => Some(VideoCodec::Av1),
        "V_VP9" => Some(VideoCodec::Vp9),
        "A_AAC" => Some(VideoCodec::Aac),
        "A_OPUS" => Some(VideoCodec::Opus),
        "A_MPEG/L3" => Some(VideoCodec::Mp3),
        _ => None,
    }
}

// Matroska stores its sampling frequency as `f64`. Canonical rates are
// integral; the value is range-validated before the conversion.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn matroska_sample_rate(rate: f64) -> Option<u32> {
    if rate.is_finite() && rate >= 1.0 && rate <= f64::from(u32::MAX) {
        Some(rate as u32)
    } else {
        None
    }
}

fn mkv_codec(codec_id: &str) -> Option<VideoCodec> {
    match codec_id {
        "V_MPEG4/ISO/AVC" => Some(VideoCodec::H264),
        "V_MPEGH/ISO/HEVC" => Some(VideoCodec::Hevc),
        "V_AV1" => Some(VideoCodec::Av1),
        "V_VP9" => Some(VideoCodec::Vp9),
        "A_AAC" => Some(VideoCodec::Aac),
        "A_OPUS" => Some(VideoCodec::Opus),
        "A_MPEG/L3" => Some(VideoCodec::Mp3),
        _ => None,
    }
}

/// VP9 codec metadata in a format-neutral form. Matroska carries it as
/// `(id, length, data)` codec features, MP4 as a `vpcC` record.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Vp9Features {
    profile: Option<u8>,
    level: Option<u8>,
    bit_depth: Option<u8>,
    chroma_subsampling: Option<u8>,
    full_range: Option<bool>,
    colour_primaries: Option<u8>,
    transfer: Option<u8>,
    matrix: Option<u8>,
    init_data: Vec<u8>,
}

fn parse_vp9_codec_private(data: &[u8]) -> Option<Vp9Features> {
    let mut features = Vp9Features::default();
    let mut idx = 0_usize;
    while idx < data.len() {
        let id = *data.get(idx)?;
        let len = usize::from(*data.get(idx.saturating_add(1))?);
        idx = idx.checked_add(2)?;
        let end = idx.checked_add(len)?;
        let value = data.get(idx..end)?;
        idx = end;
        if (1..=8).contains(&id) {
            if len != 1 {
                return None;
            }
            let &byte = value.first()?;
            match id {
                1 => features.profile = Some(byte),
                2 => features.level = Some(byte),
                3 => features.bit_depth = Some(byte),
                4 => features.chroma_subsampling = Some(byte),
                5 => features.full_range = Some(byte != 0),
                6 => features.colour_primaries = Some(byte),
                7 => features.transfer = Some(byte),
                8 => features.matrix = Some(byte),
                _ => {}
            }
        } else if id == 9 {
            features.init_data = value.to_vec();
        }
        // Unknown features have no vpcC representation and are dropped.
    }
    Some(features)
}

fn push_vp9_feature(out: &mut Vec<u8>, id: u8, value: &[u8]) {
    if let Ok(len) = u8::try_from(value.len()) {
        out.push(id);
        out.push(len);
        out.extend_from_slice(value);
    }
}

fn vp9_features_to_codec_private(features: &Vp9Features) -> Vec<u8> {
    let mut out = Vec::new();
    if let Some(value) = features.profile {
        push_vp9_feature(&mut out, 1, &[value]);
    }
    if let Some(value) = features.level {
        push_vp9_feature(&mut out, 2, &[value]);
    }
    if let Some(value) = features.bit_depth {
        push_vp9_feature(&mut out, 3, &[value]);
    }
    if let Some(value) = features.chroma_subsampling {
        push_vp9_feature(&mut out, 4, &[value]);
    }
    if let Some(value) = features.full_range {
        push_vp9_feature(&mut out, 5, &[u8::from(value)]);
    }
    if let Some(value) = features.colour_primaries {
        push_vp9_feature(&mut out, 6, &[value]);
    }
    if let Some(value) = features.transfer {
        push_vp9_feature(&mut out, 7, &[value]);
    }
    if let Some(value) = features.matrix {
        push_vp9_feature(&mut out, 8, &[value]);
    }
    if !features.init_data.is_empty() {
        push_vp9_feature(&mut out, 9, &features.init_data);
    }
    out
}

/// Serializes derived features into a vpcC body (version, flags, then record).
/// Missing optional metadata falls back to the specification defaults.
fn vp9_features_to_vpcc(features: &Vp9Features) -> Option<Vec<u8>> {
    let bit_depth = features.bit_depth.unwrap_or(8);
    let chroma = features.chroma_subsampling.unwrap_or(0);
    let matrix = features.matrix.unwrap_or(1);
    if bit_depth > 15
        || chroma > 3
        || (matrix == 0 && chroma != 3)
        || features.init_data.len() > usize::from(u16::MAX)
    {
        return None;
    }
    let packed = (bit_depth << 4) | (chroma << 1) | u8::from(features.full_range.unwrap_or(false));
    let size = u16::try_from(features.init_data.len()).ok()?;
    let mut out = Vec::with_capacity(12_usize.saturating_add(features.init_data.len()));
    out.extend_from_slice(&[1, 0, 0, 0]);
    out.push(features.profile.unwrap_or(0));
    out.push(features.level.unwrap_or(0));
    out.push(packed);
    out.push(features.colour_primaries.unwrap_or(1));
    out.push(features.transfer.unwrap_or(1));
    out.push(matrix);
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(&features.init_data);
    Some(out)
}

fn parse_vp9_vpcc(body: &[u8]) -> Option<Vp9Features> {
    if *body.first()? != 1 {
        return None;
    }
    let packed = *body.get(6)?;
    let chroma = (packed >> 1) & 0x07;
    if chroma > 3 {
        return None;
    }
    let size = u16::from_be_bytes([*body.get(10)?, *body.get(11)?]);
    let end = 12_usize.checked_add(usize::from(size))?;
    let init_data = body.get(12..end)?.to_vec();
    Some(Vp9Features {
        profile: Some(*body.get(4)?),
        level: Some(*body.get(5)?),
        bit_depth: Some(packed >> 4),
        chroma_subsampling: Some(chroma),
        full_range: Some(packed & 1 != 0),
        colour_primaries: Some(*body.get(7)?),
        transfer: Some(*body.get(8)?),
        matrix: Some(*body.get(9)?),
        init_data,
    })
}

fn vp9_vpcc_to_codec_private(body: &[u8]) -> Vec<u8> {
    parse_vp9_vpcc(body).map_or_else(Vec::new, |features| {
        vp9_features_to_codec_private(&features)
    })
}

fn vp9_codec_private_to_vpcc(data: &[u8]) -> Option<Vec<u8>> {
    if data.is_empty() {
        return None;
    }
    vp9_features_to_vpcc(&parse_vp9_codec_private(data)?)
}

fn vp9_validated_codec_private(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    parse_vp9_codec_private(data).map_or_else(Vec::new, |features| {
        vp9_features_to_codec_private(&features)
    })
}

fn canonical_config(
    track: &Track,
    codec: VideoCodec,
    path: &Path,
) -> Result<Vec<u8>, PipelineError> {
    let raw = track.codec_config().map_or(&[][..], CodecConfig::bytes);
    match codec {
        VideoCodec::H264 | VideoCodec::Hevc | VideoCodec::Av1 => Ok(raw.to_vec()),
        // Source vpcC is converted into canonical Matroska codec features.
        VideoCodec::Vp9 => Ok(vp9_vpcc_to_codec_private(raw)),
        VideoCodec::Aac => {
            esds_asc(raw).ok_or_else(|| container_error(path, "missing AAC AudioSpecificConfig"))
        }
        VideoCodec::Opus => {
            if raw.len() >= 19 && &raw[..8] == b"OpusHead" {
                Ok(raw.to_vec())
            } else {
                dops_to_opus_head(raw).map_err(|reason| container_error(path, reason))
            }
        }
        VideoCodec::Mp3 => Ok(Vec::new()),
        _ => Err(container_error(path, format!("unsupported codec {codec}"))),
    }
}

fn demux_mp4(path: &Path) -> Result<ContainerMedia, PipelineError> {
    let reader = WindowedFileReader::open(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut input =
        containr::mp4::Demuxer::open(reader).map_err(|err| containr_error(path, &err))?;
    let mut media = ContainerMedia::default();
    let mut id_to_index = std::collections::HashMap::new();
    for track in input.tracks().iter().filter(|t| !t.chapter_track) {
        let Some(codec) = mp4_codec(track) else {
            continue;
        };
        let config = canonical_config(track, codec, path)?;
        let (width, height) = track
            .video()
            .map_or((None, None), |v| (Some(v.width), Some(v.height)));
        let (channels, sample_rate) = track.audio().map_or((None, None), |a| {
            (
                u16::try_from(a.channels).ok(),
                (a.sample_rate > 0).then_some(a.sample_rate),
            )
        });
        id_to_index.insert(track.id, media.tracks.len());
        media.tracks.push(ContainerTrack {
            codec,
            width,
            height,
            channels,
            sample_rate,
            config,
            timescale: US_TIMESCALE,
        });
    }
    if media.tracks.is_empty() {
        return Err(container_error(path, "MP4 source has no supported tracks"));
    }
    let mut buf = Vec::new();
    for packet in input.packets() {
        let packet = packet.map_err(|err| containr_error(path, &err))?;
        let Some(&track) = id_to_index.get(&packet.track_id) else {
            continue;
        };
        input
            .read_packet_into(&packet, &mut buf)
            .map_err(|err| containr_error(path, &err))?;
        let dts = packet.dts.unwrap_or(packet.pts);
        media.packets.push(ContainerPacket {
            track,
            pts: rescale_ticks(
                packet.pts,
                1,
                input
                    .tracks()
                    .iter()
                    .find(|t| t.id == packet.track_id)
                    .map_or(US_TIMESCALE, |t| t.timebase.den),
            ),
            dts: rescale_ticks(
                dts,
                1,
                input
                    .tracks()
                    .iter()
                    .find(|t| t.id == packet.track_id)
                    .map_or(US_TIMESCALE, |t| t.timebase.den),
            ),
            duration: packet.duration.map_or(1, |d| {
                u64::try_from(rescale_ticks(
                    i64::try_from(d).unwrap_or(i64::MAX),
                    1,
                    input
                        .tracks()
                        .iter()
                        .find(|t| t.id == packet.track_id)
                        .map_or(US_TIMESCALE, |t| t.timebase.den),
                ))
                .unwrap_or(1)
            }),
            keyframe: packet.sync,
            data: buf.clone(),
        });
    }
    Ok(media)
}

// Track facts and packet timestamps come from one demuxer walk; extracting
// them would require threading the decoded track index through every packet.
#[allow(clippy::too_many_lines)]
fn demux_mkv(path: &Path) -> Result<ContainerMedia, PipelineError> {
    let reader = WindowedFileReader::open(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut input =
        containr::mkv::Demuxer::open(reader).map_err(|err| containr_error(path, &err))?;
    if input.truncated() {
        return Err(container_error(path, "truncated Matroska input"));
    }
    let mut media = ContainerMedia::default();
    let mut id_to_index = std::collections::HashMap::new();
    for track in input.tracks() {
        let Some(codec_id) = track.codec_id() else {
            continue;
        };
        let Some(codec) = mkv_codec(codec_id) else {
            continue;
        };
        let config = track
            .matroska
            .as_ref()
            .and_then(|mkv| mkv.codec_private.clone())
            .unwrap_or_default();
        let config = if codec == VideoCodec::Vp9 {
            vp9_validated_codec_private(&config)
        } else {
            config
        };
        let (width, height) = track
            .matroska
            .as_ref()
            .and_then(|mkv| mkv.video.as_ref())
            .map_or((None, None), |v| {
                (
                    u16::try_from(v.pixel_width).ok(),
                    u16::try_from(v.pixel_height).ok(),
                )
            });
        let (channels, sample_rate) = track
            .matroska
            .as_ref()
            .and_then(|mkv| mkv.audio.as_ref())
            .map_or((None, None), |a| {
                (
                    u16::try_from(a.channels).ok(),
                    matroska_sample_rate(a.sampling_frequency),
                )
            });
        id_to_index.insert(track.id, media.tracks.len());
        media.tracks.push(ContainerTrack {
            codec,
            width,
            height,
            channels,
            sample_rate,
            config,
            timescale: US_TIMESCALE,
        });
    }
    if media.tracks.is_empty() {
        return Err(container_error(
            path,
            "Matroska source has no supported tracks",
        ));
    }
    let scales: Vec<(i64, i64)> = input
        .tracks()
        .iter()
        .map(|t| {
            (
                i64::from(t.timebase.num).max(1),
                i64::from(t.timebase.den).max(1),
            )
        })
        .collect();
    let mut buf = Vec::new();
    for packet in input.packets() {
        let packet = packet.map_err(|err| containr_error(path, &err))?;
        let Some(&track) = id_to_index.get(&packet.track_id) else {
            continue;
        };
        input
            .read_packet_into(&packet, &mut buf)
            .map_err(|err| containr_error(path, &err))?;
        let (num, den) = scales[track];
        let pts = rescale_ticks(
            packet.pts,
            u32::try_from(num).unwrap_or(1),
            u32::try_from(den).unwrap_or(1),
        );
        let duration = packet.duration.map_or(1, |d| {
            u64::try_from(rescale_ticks(
                i64::try_from(d).unwrap_or(0),
                u32::try_from(num).unwrap_or(1),
                u32::try_from(den).unwrap_or(1),
            ))
            .unwrap_or(1)
        });
        media.packets.push(ContainerPacket {
            track,
            pts,
            dts: pts,
            duration,
            keyframe: packet.sync,
            data: buf.clone(),
        });
    }
    Ok(media)
}

/// Demuxes any supported container into the canonical representation.
pub(crate) fn demux_video(
    path: &Path,
    format: VideoContainerFormat,
) -> Result<ContainerMedia, PipelineError> {
    match format {
        VideoContainerFormat::Mp4 | VideoContainerFormat::Mov => demux_mp4(path),
        VideoContainerFormat::Mkv | VideoContainerFormat::Webm => demux_mkv(path),
        VideoContainerFormat::Avi => crate::avi::demux_avi(path),
        VideoContainerFormat::Ts => crate::mpegts::demux_ts(path),
    }
}

fn mux_mp4(
    target: &File,
    target_path: &Path,
    media: &ContainerMedia,
    brand: Brand,
) -> Result<(), PipelineError> {
    let estimated: u64 = media
        .packets
        .iter()
        .map(|p| u64::try_from(p.data.len()).unwrap_or(0))
        .sum();
    let options = MuxOptions {
        brand,
        faststart: true,
        // Track-start edits are expressed in microseconds, matching the
        // canonical timeline, so no offset is lost to rescaling.
        movie_timescale: US_TIMESCALE,
        ..MuxOptions::default()
    };
    if estimated <= IN_MEMORY_MP4_FASTSTART_LIMIT {
        let mut mem = Cursor::new(Vec::with_capacity(
            usize::try_from(estimated)
                .unwrap_or(0)
                .saturating_add(256 * 1024),
        ));
        mux_mp4_into(&mut mem, target_path, media, options)?;
        let mut file: &File = target;
        file.write_all(mem.get_ref())
            .map_err(|source| PipelineError::Io {
                path: target_path.to_path_buf(),
                source,
            })?;
        file.flush().map_err(|source| PipelineError::Io {
            path: target_path.to_path_buf(),
            source,
        })?;
    } else {
        mux_mp4_into(target, target_path, media, options)?;
    }
    Ok(())
}

// Sample-entry selection and timestamp shifting are a single ordered pass
// over tracks then packets, as the muxer requires every track added first.
#[allow(clippy::too_many_lines)]
fn mux_mp4_into<W: Read + Write + Seek>(
    writer: W,
    target_path: &Path,
    media: &ContainerMedia,
    options: MuxOptions,
) -> Result<(), PipelineError> {
    let mut output = containr::mp4::Muxer::new(writer, options)
        .map_err(|err| containr_error(target_path, &err))?;
    let mut ids = Vec::with_capacity(media.tracks.len());
    for track in &media.tracks {
        let sample_entry = match track.codec {
            VideoCodec::H264 => SampleEntrySpec::Visual {
                codec: VisualCodec::Avc,
                config: track.config.clone(),
                width: track.width.unwrap_or(1920),
                height: track.height.unwrap_or(1080),
            },
            VideoCodec::Hevc => SampleEntrySpec::Visual {
                codec: VisualCodec::Hevc,
                config: track.config.clone(),
                width: track.width.unwrap_or(1920),
                height: track.height.unwrap_or(1080),
            },
            VideoCodec::Av1 => SampleEntrySpec::Visual {
                codec: VisualCodec::Av1,
                config: track.config.clone(),
                width: track.width.unwrap_or(1920),
                height: track.height.unwrap_or(1080),
            },
            VideoCodec::Vp9 => {
                // Canonical config is Matroska codec features; MP4 needs a
                // parsed vpcC record, never a byte-for-byte copy.
                let vpcc = vp9_codec_private_to_vpcc(&track.config);
                SampleEntrySpec::Raw(build_vp09_sample_entry(
                    track.width.unwrap_or(0),
                    track.height.unwrap_or(0),
                    vpcc.as_deref(),
                ))
            }
            VideoCodec::Aac => SampleEntrySpec::Audio {
                codec: AudioCodec::Aac,
                config: build_aac_esds(&track.config),
                channels: track.channels.unwrap_or(2),
                sample_rate: track.sample_rate.unwrap_or(48_000),
            },
            VideoCodec::Opus => SampleEntrySpec::Audio {
                codec: AudioCodec::Opus,
                config: opus_head_to_dops(&track.config)
                    .map_err(|reason| container_error(target_path, reason))?,
                channels: track.channels.unwrap_or(2),
                sample_rate: 48_000,
            },
            VideoCodec::Mp3 => SampleEntrySpec::Raw(build_mp3_mp4a_sample_entry(
                track.channels.unwrap_or(2),
                track.sample_rate.unwrap_or(44_100),
            )),
            _ => {
                return Err(container_error(
                    target_path,
                    format!("codec {} cannot be stored in MP4", track.codec),
                ));
            }
        };
        let kind = if track.codec.is_video() {
            TrackKind::Video
        } else {
            TrackKind::Audio
        };
        let mut spec = Mp4TrackSpec::new(kind, US_TIMESCALE, sample_entry);
        spec.default_duration =
            Some(u32::try_from(track_duration(media, track_index(media, track))).unwrap_or(1));
        if kind == TrackKind::Video {
            spec.display_width = u32::from(track.width.unwrap_or(0)) << 16;
            spec.display_height = u32::from(track.height.unwrap_or(0)) << 16;
        }
        ids.push(
            output
                .add_track(spec)
                .map_err(|err| containr_error(target_path, &err))?,
        );
    }
    // containr's MP4 muxer requires each track's first decode timestamp to be
    // zero and every following one to be strictly greater. Rebase each track
    // on its own first DTS (MPEG-TS starts at the PCR offset) while keeping the
    // presentation/decode composition offset intact for B-frames. A track that
    // starts later on the common timeline keeps that delay through an edit
    // list: zeroing every track's first DTS would silently align them.
    let mut first_dts = vec![0_i64; media.tracks.len()];
    let mut seen = vec![false; media.tracks.len()];
    for packet in &media.packets {
        if !seen[packet.track] || packet.dts < first_dts[packet.track] {
            first_dts[packet.track] = packet.dts;
            seen[packet.track] = true;
        }
    }
    let timeline_origin = first_dts
        .iter()
        .enumerate()
        .filter(|(index, _)| seen[*index])
        .map(|(_, dts)| *dts)
        .min()
        .unwrap_or(0);
    let mut last_packet_for_track = vec![None; media.tracks.len()];
    for (index, packet) in media.packets.iter().enumerate() {
        last_packet_for_track[packet.track] = Some(index);
    }
    let mut prev_dts: Vec<Option<i64>> = vec![None; media.tracks.len()];
    let mut media_end: Vec<u64> = vec![0; media.tracks.len()];
    for (packet_index, packet) in media.packets.iter().enumerate() {
        let shifted_dts = packet.dts.saturating_sub(first_dts[packet.track]);
        let dts = match prev_dts[packet.track] {
            None => 0,
            Some(prev) => shifted_dts.max(prev.saturating_add(1)),
        };
        let cto = packet.pts.saturating_sub(packet.dts);
        let pts = dts.saturating_add(cto);
        prev_dts[packet.track] = Some(dts);
        let end = pts
            .saturating_add(i64::try_from(packet.duration).unwrap_or(i64::MAX))
            .max(0);
        media_end[packet.track] =
            media_end[packet.track].max(u64::try_from(end).unwrap_or(u64::MAX));
        let input = containr::mp4::PacketIn {
            data: &packet.data,
            pts,
            dts,
            // Only the final packet needs an explicit duration; intermediate
            // stts deltas come from the next DTS without 1 µs rounding.
            duration: (last_packet_for_track[packet.track] == Some(packet_index))
                .then(|| packet.duration.max(1)),
            sync: packet.keyframe,
        };
        output
            .write_packet(ids[packet.track], &input)
            .map_err(|err| containr_error(target_path, &err))?;
    }
    for (index, id) in ids.iter().enumerate() {
        if !seen[index] {
            continue;
        }
        let delay = first_dts[index].saturating_sub(timeline_origin);
        let Ok(delay) = u64::try_from(delay) else {
            return Err(container_error(
                target_path,
                format!("track {index} starts at an unrepresentable negative offset"),
            ));
        };
        if delay == 0 {
            continue;
        }
        // An empty edit delays the whole track; the following edit plays the
        // media from its start. Asking the muxer to accept the list first makes
        // an unsupported edit an explicit failure, not silent wrong timing.
        let list = EditList {
            movie_timescale: US_TIMESCALE,
            entries: vec![
                EditEntry {
                    segment_duration: delay,
                    media_time: EditEntry::EMPTY_MEDIA_TIME,
                    media_rate: EditEntry::NORMAL_RATE,
                },
                EditEntry {
                    segment_duration: media_end[index],
                    media_time: 0,
                    media_rate: EditEntry::NORMAL_RATE,
                },
            ],
        };
        output
            .set_edit_list(*id, &list)
            .map_err(|err| containr_error(target_path, &err))?;
    }
    output
        .finish()
        .map_err(|err| containr_error(target_path, &err))?;
    Ok(())
}

fn track_index(media: &ContainerMedia, track: &ContainerTrack) -> usize {
    media
        .tracks
        .iter()
        .position(|candidate| std::ptr::eq(candidate, track))
        .unwrap_or(0)
}

fn track_duration(media: &ContainerMedia, index: usize) -> u64 {
    media
        .packets
        .iter()
        .filter(|p| p.track == index)
        .map(|p| p.duration)
        .max()
        .unwrap_or(1)
        .max(1)
}

fn mux_mkv(
    target: &File,
    target_path: &Path,
    media: &ContainerMedia,
    doc_type: DocType,
) -> Result<(), PipelineError> {
    let mut writer = BufWriter::with_capacity(512 * 1024, target);
    let options = MkvOptions {
        doc_type,
        timestamp_scale: MKV_TICK_NS,
        ..MkvOptions::default()
    };
    let mut output = containr::mkv::Muxer::new(&mut writer, options)
        .map_err(|err| containr_error(target_path, &err))?;
    let timebase = Timebase::from_nanoseconds(u32::try_from(MKV_TICK_NS).unwrap_or(1_000));
    let mut ids = Vec::with_capacity(media.tracks.len());
    for (index, track) in media.tracks.iter().enumerate() {
        let codec_id = match track.codec {
            VideoCodec::H264 => "V_MPEG4/ISO/AVC",
            VideoCodec::Hevc => "V_MPEGH/ISO/HEVC",
            VideoCodec::Av1 => "V_AV1",
            VideoCodec::Vp9 => "V_VP9",
            VideoCodec::Aac => "A_AAC",
            VideoCodec::Opus => "A_OPUS",
            VideoCodec::Mp3 => "A_MPEG/L3",
            _ => {
                return Err(container_error(
                    target_path,
                    format!("codec {} cannot be stored in Matroska", track.codec),
                ));
            }
        };
        let kind = if track.codec.is_video() {
            TrackKind::Video
        } else {
            TrackKind::Audio
        };
        let mut spec = MkvTrackSpec::new(kind, codec_id, timebase);
        spec.matroska.codec_private = match track.codec {
            VideoCodec::Vp9 => {
                let private = vp9_validated_codec_private(&track.config);
                (!private.is_empty()).then_some(private)
            }
            _ => (!track.config.is_empty()).then(|| track.config.clone()),
        };
        if track.codec == VideoCodec::Opus {
            spec.matroska.seek_pre_roll = 80_000_000;
            if track.config.len() >= 19 && &track.config[..8] == b"OpusHead" {
                let pre_skip = u16::from_le_bytes([track.config[10], track.config[11]]);
                spec.matroska.codec_delay =
                    u64::from(pre_skip).saturating_mul(1_000_000_000) / 48_000;
            }
        }
        if kind == TrackKind::Video {
            spec.matroska.video = Some(video_element(track));
        } else {
            spec.matroska.audio = Some(audio_element(track));
        }
        // Matroska's DefaultDuration is in nanoseconds, not in ticks.
        spec.matroska.default_duration =
            Some(track_duration(media, index).saturating_mul(MKV_TICK_NS));
        ids.push(
            output
                .add_track(spec)
                .map_err(|err| containr_error(target_path, &err))?,
        );
    }
    let base = media.packets.iter().map(|p| p.pts).min().unwrap_or(0);
    for packet in &media.packets {
        let pts = packet.pts.saturating_sub(base).max(0);
        let input = containr::mkv::PacketIn {
            data: &packet.data,
            pts,
            duration: Some(packet.duration.max(1)),
            sync: packet.keyframe,
            discardable: false,
            discard_padding: None,
        };
        output
            .write_packet(ids[packet.track], &input)
            .map_err(|err| containr_error(target_path, &err))?;
    }
    output
        .finish()
        .map_err(|err| containr_error(target_path, &err))?;
    writer.flush().map_err(|source| PipelineError::Io {
        path: target_path.to_path_buf(),
        source,
    })?;
    Ok(())
}

fn video_element(track: &ContainerTrack) -> MatroskaVideo {
    MatroskaVideo {
        // MPEG-TS carries no display geometry; default to a 16:9 frame so
        // Matroska's mandatory PixelWidth/PixelHeight stay valid.
        pixel_width: u64::from(track.width.unwrap_or(1920)),
        pixel_height: u64::from(track.height.unwrap_or(1080)),
        display_width: None,
        display_height: None,
        display_unit: containr::matroska::DisplayUnit::Pixels,
        crop: [0; 4],
        interlaced: 0,
        projection: None,
        colour: None,
        extra: Vec::new(),
    }
}

fn audio_element(track: &ContainerTrack) -> MatroskaAudio {
    MatroskaAudio {
        sampling_frequency: f64::from(track.sample_rate.unwrap_or(48_000)),
        output_sampling_frequency: None,
        channels: u64::from(track.channels.unwrap_or(2)),
        bit_depth: None,
        extra: Vec::new(),
    }
}

/// Muxes the canonical representation into any supported output container.
pub(crate) fn mux_video(
    target: &File,
    target_path: &Path,
    format: VideoContainerFormat,
    media: &ContainerMedia,
) -> Result<(), PipelineError> {
    for track in &media.tracks {
        if !track.codec.supported_in(format) {
            return Err(container_error(
                target_path,
                format!(
                    "codec {} is not supported in {}",
                    track.codec,
                    format.label()
                ),
            ));
        }
    }
    if media.tracks.is_empty() {
        return Err(container_error(target_path, "no tracks to mux"));
    }
    match format {
        VideoContainerFormat::Mp4 => mux_mp4(target, target_path, media, Brand::Isom),
        VideoContainerFormat::Mov => mux_mp4(target, target_path, media, Brand::QuickTime),
        VideoContainerFormat::Mkv => mux_mkv(target, target_path, media, DocType::Matroska),
        VideoContainerFormat::Webm => mux_mkv(target, target_path, media, DocType::WebM),
        VideoContainerFormat::Avi => crate::avi::mux_avi(target, target_path, media),
        VideoContainerFormat::Ts => crate::mpegts::mux_ts(target, target_path, media),
    }
}

/// Converts canonical AVCC/HVCC samples into Annex-B with parameter sets when
/// the source container needs them for random access.
pub(crate) fn video_payload_for_annex_b(
    track: &ContainerTrack,
    packet: &ContainerPacket,
) -> Result<Vec<u8>, String> {
    match track.codec {
        VideoCodec::H264 => avcc_to_annex_b(&track.config, &packet.data, packet.keyframe),
        VideoCodec::Hevc => hvcc_to_annex_b(&track.config, &packet.data, packet.keyframe),
        _ => Ok(packet.data.clone()),
    }
}

/// Converts an Annex-B sample into canonical AVCC/HVCC and extracts its config
/// when the track has none yet.
pub(crate) fn video_packet_from_annex_b(
    track: &mut ContainerTrack,
    data: &[u8],
    keyframe: bool,
) -> (Vec<u8>, bool) {
    match track.codec {
        VideoCodec::H264 => {
            if track.config.is_empty()
                && keyframe
                && let Some(config) = extract_avcc_from_annex_b(data)
            {
                track.config = config;
            }
            (
                annex_b_to_avcc(data),
                is_h264_keyframe_annex_b(data) || keyframe,
            )
        }
        VideoCodec::Hevc => {
            if track.config.is_empty()
                && keyframe
                && let Some(config) = extract_hvcc_from_annex_b(data)
            {
                track.config = config;
            }
            (
                annex_b_to_hvcc(data),
                is_hevc_keyframe_annex_b(data) || keyframe,
            )
        }
        VideoCodec::Av1 => {
            if track.config.is_empty()
                && keyframe
                && let Some(config) = extract_av1c_from_obu(data)
            {
                track.config = config;
            }
            (data.to_vec(), is_av1_keyframe(data) || keyframe)
        }
        _ => (data.to_vec(), keyframe),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vp9_codec_private_roundtrips_through_vpcc() -> Result<(), String> {
        // Matroska codec features: profile 2, level 30, 10-bit 4:2:0 full-range
        // BT.709 plus two bytes of codec initialization data.
        let private = [
            1, 1, 2, // profile
            2, 1, 30, // level
            3, 1, 10, // bit depth
            4, 1, 0, // 4:2:0
            5, 1, 1, // full range
            6, 1, 1, // primaries
            7, 1, 1, // transfer
            8, 1, 1, // matrix
            9, 2, 0xAA, 0xBB, // init data
        ];
        let vpcc = vp9_codec_private_to_vpcc(&private).ok_or("valid VP9 features")?;
        assert_eq!(vpcc.len(), 14);
        assert_eq!(&vpcc[..4], &[1, 0, 0, 0]);
        assert_eq!(vpcc[4], 2);
        assert_eq!(vpcc[5], 30);
        assert_eq!(vpcc[6], 0xA1);
        assert_eq!(&vpcc[7..10], &[1, 1, 1]);
        assert_eq!(&vpcc[10..12], &[0, 2]);
        assert_eq!(&vpcc[12..], &[0xAA, 0xBB]);
        assert_eq!(vp9_vpcc_to_codec_private(&vpcc), private.to_vec());
        Ok(())
    }

    #[test]
    fn vp9_omits_underivable_metadata() {
        assert_eq!(vp9_codec_private_to_vpcc(&[]), None);
        // An unknown feature alone cannot be expressed as vpcC.
        assert_eq!(vp9_validated_codec_private(&[0x7F, 0x01]), Vec::<u8>::new());
        assert_eq!(vp9_vpcc_to_codec_private(&[2, 0, 0, 0]), Vec::<u8>::new());
        // A truncated feature must not be carried over byte-for-byte.
        assert_eq!(vp9_validated_codec_private(&[9, 4, 1]), Vec::<u8>::new());
    }
}
