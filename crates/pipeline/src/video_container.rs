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
    is_hevc_keyframe_annex_b, opus_head_to_dops, vp9_color_from_keyframe,
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

/// Explicit visible presentation interval from a container edit list (such as
/// MP4 `elst`), expressed in canonical microseconds on the same timeline as
/// the track's packets.
///
/// Packets with `pts < start_us` are retained for decoding as preroll/priming
/// rather than discarded at demux time, but excluded from visible playback and
/// movie-origin calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TrackPresentation {
    pub(crate) start_us: i64,
    /// Optional explicit presentation duration (`segment_duration > 0` in
    /// `elst`). `None` plays through the end of the track.
    pub(crate) duration_us: Option<u64>,
}

impl TrackPresentation {
    pub(crate) fn shifted(self, offset_us: i64) -> Self {
        Self {
            start_us: self.start_us.saturating_add(offset_us),
            duration_us: self.duration_us,
        }
    }
}

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
    /// Optional explicit presentation window from a source edit list.
    pub(crate) presentation_edit: Option<TrackPresentation>,
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

/// Rescales a canonical microsecond timestamp into rounded ticks of `timescale`
/// ticks per second (identity when `timescale == US_TIMESCALE`).
fn us_to_ticks(value_us: i64, timescale: u32) -> i64 {
    if timescale == US_TIMESCALE {
        return value_us;
    }
    let numerator = i128::from(value_us).saturating_mul(i128::from(timescale.max(1)));
    let denominator = i128::from(US_TIMESCALE);
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

/// Parses a supported single-segment normal-rate MP4 `elst` box into:
/// - the timestamp shift (`delay_us - media_start_us`) added to raw packet
///   PTS/DTS so visible samples sit at their movie presentation times, and
/// - the track's visible [`TrackPresentation`] interval starting at `delay_us`,
///   keeping earlier raw packets as decode preroll or audio priming.
fn mp4_track_edit(track: &Track) -> Option<(i64, TrackPresentation)> {
    let EditList {
        movie_timescale,
        entries,
    } = &track.edit_list;
    let (empty, media) = match entries.as_slice() {
        [media] if !media.is_empty() => (None, media),
        [empty, media] if empty.is_empty() && !media.is_empty() => (Some(empty), media),
        _ => return None,
    };
    if media.media_rate != EditEntry::NORMAL_RATE || *movie_timescale == 0 {
        return None;
    }
    let delay_us = empty.map_or(0_i64, |entry| {
        rescale_ticks(
            i64::try_from(entry.segment_duration).unwrap_or(i64::MAX),
            1,
            *movie_timescale,
        )
    });
    let media_start_us = if media.media_time > 0 {
        rescale_ticks(media.media_time, 1, track.timebase.den.max(1))
    } else {
        0
    };
    let duration_us = (media.segment_duration > 0).then(|| {
        u64::try_from(rescale_ticks(
            i64::try_from(media.segment_duration).unwrap_or(i64::MAX),
            1,
            *movie_timescale,
        ))
        .unwrap_or(u64::MAX)
    });
    Some((
        delay_us.saturating_sub(media_start_us),
        TrackPresentation {
            start_us: delay_us,
            duration_us,
        },
    ))
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
    let mut track_meta: Vec<(u32, i64)> = Vec::new();
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
        let (edit_shift, presentation_edit) = match mp4_track_edit(track) {
            Some((shift, edit)) => (shift, Some(edit)),
            None => (0, None),
        };
        id_to_index.insert(track.id, media.tracks.len());
        track_meta.push((track.timebase.den.max(1), edit_shift));
        media.tracks.push(ContainerTrack {
            codec,
            width,
            height,
            channels,
            sample_rate,
            config,
            timescale: US_TIMESCALE,
            presentation_edit,
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
        let (timescale, edit_shift) = track_meta[track];
        let dts = packet.dts.unwrap_or(packet.pts);
        media.packets.push(ContainerPacket {
            track,
            pts: rescale_ticks(packet.pts, 1, timescale).saturating_add(edit_shift),
            dts: rescale_ticks(dts, 1, timescale).saturating_add(edit_shift),
            duration: packet.duration.map_or(1, |d| {
                u64::try_from(rescale_ticks(
                    i64::try_from(d).unwrap_or(i64::MAX),
                    1,
                    timescale,
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
            presentation_edit: None,
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
    synthesize_mkv_decode_timestamps(path, &mut media.packets, media.tracks.len())?;
    Ok(media)
}

/// Matroska blocks sit in coding (decode) order and carry only presentation
/// timestamps. When a track contains B-frames (PTS inversions in block order),
/// assign strictly increasing `dts <= pts` in block order so downstream
/// `decode_order` sorts never move B-frames ahead of their reference frames.
fn synthesize_mkv_decode_timestamps(
    path: &Path,
    packets: &mut [ContainerPacket],
    track_count: usize,
) -> Result<(), PipelineError> {
    let mut by_track: Vec<Vec<usize>> = vec![Vec::new(); track_count];
    for (index, packet) in packets.iter().enumerate() {
        if let Some(indices) = by_track.get_mut(packet.track) {
            indices.push(index);
        }
    }
    for indices in by_track {
        if indices.len() < 2 {
            continue;
        }
        let has_inversion = indices
            .windows(2)
            .any(|pair| packets[pair[1]].pts < packets[pair[0]].pts);
        if !has_inversion {
            continue;
        }
        let mut sorted_pts: Vec<i64> = indices.iter().map(|&idx| packets[idx].pts).collect();
        sorted_pts.sort_unstable();
        let reorder_shift = sorted_pts
            .iter()
            .zip(&indices)
            .fold(0_i128, |max, (&display_pts, &idx)| {
                max.max(i128::from(display_pts) - i128::from(packets[idx].pts))
            })
            .max(0);
        let mut dts = Vec::with_capacity(sorted_pts.len());
        for &display_pts in &sorted_pts {
            let shifted = i64::try_from(i128::from(display_pts) - reorder_shift).map_err(|_| {
                container_error(
                    path,
                    "Matroska synthesized DTS underflowed signed 64-bit range",
                )
            })?;
            dts.push(shifted);
        }
        // Single O(n) backward pass: bounding each provisional DTS by `next - 1`
        // keeps block order and PTS intact while ensuring strict monotonicity.
        for k in (0..dts.len().saturating_sub(1)).rev() {
            let max_allowed = dts[k + 1].checked_sub(1).ok_or_else(|| {
                container_error(
                    path,
                    "Matroska DTS tie repair exhausted signed 64-bit range",
                )
            })?;
            if dts[k] > max_allowed {
                dts[k] = max_allowed;
            }
        }
        for (&idx, packet_dts) in indices.iter().zip(dts) {
            packets[idx].dts = packet_dts;
        }
    }
    Ok(())
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

/// First decode timestamp per track in decode order. The smallest DTS can
/// belong to a reordered B-frame, so it is not a valid track origin.
fn mp4_decode_origins(media: &ContainerMedia) -> Vec<Option<i64>> {
    let mut origins = vec![None; media.tracks.len()];
    for packet in &media.packets {
        if origins[packet.track].is_none() {
            origins[packet.track] = Some(packet.dts);
        }
    }
    origins
}

/// Earliest presentation time per track, clamped to that track's first DTS.
/// A reordered B-frame may present before the first packet in decode order;
/// using the smaller of the two as the presentation base keeps every rebased
/// PTS non-negative.
fn mp4_earliest_presentation_times(media: &ContainerMedia) -> Vec<Option<i64>> {
    let mut earliest_pts: Vec<Option<i64>> = vec![None; media.tracks.len()];
    for packet in &media.packets {
        earliest_pts[packet.track] =
            Some(earliest_pts[packet.track].map_or(packet.pts, |min| min.min(packet.pts)));
    }
    earliest_pts
}

fn mp4_presentation_bases(media: &ContainerMedia, origins: &[Option<i64>]) -> Vec<Option<i64>> {
    let earliest_pts = mp4_earliest_presentation_times(media);
    origins
        .iter()
        .zip(earliest_pts)
        .map(|(origin, earliest)| {
            origin.and_then(|first_dts| earliest.map(|earliest| first_dts.min(earliest)))
        })
        .collect()
}

/// Visible movie-timeline start per track. When a track carries an explicit
/// container edit interval, decode preroll / audio priming packets with
/// `pts < edit.start_us` stay in the track for decoding without pulling the
/// track's visible start or the movie origin backward.
fn mp4_visible_starts(
    media: &ContainerMedia,
    pts_bases: &[Option<i64>],
    earliest_pts: &[Option<i64>],
) -> Vec<Option<i64>> {
    media
        .tracks
        .iter()
        .enumerate()
        .map(|(index, track)| {
            let pts_base = pts_bases[index]?;
            let earliest = earliest_pts[index]?;
            Some(match track.presentation_edit {
                Some(edit) => edit.start_us.max(pts_base),
                None => earliest,
            })
        })
        .collect()
}

/// `(decode, presentation)` timestamps in decode order. DTS is made strictly
/// increasing from the first decode timestamp while PTS is rebased from the
/// presentation base, so forcing a later DTS cannot push a frame's PTS below
/// zero or move its presentation time relative to the other tracks.
fn mp4_rebased_timestamps(
    media: &ContainerMedia,
    origins: &[Option<i64>],
    pts_bases: &[Option<i64>],
) -> Vec<(i64, i64)> {
    let mut previous: Vec<Option<i64>> = vec![None; media.tracks.len()];
    media
        .packets
        .iter()
        .map(|packet| {
            let origin = origins[packet.track].unwrap_or(0);
            let dts = match previous[packet.track] {
                None => 0,
                Some(prev) => packet
                    .dts
                    .saturating_sub(origin)
                    .max(prev.saturating_add(1)),
            };
            previous[packet.track] = Some(dts);
            let pts_base = pts_bases[packet.track].unwrap_or(origin);
            (dts, packet.pts.saturating_sub(pts_base))
        })
        .collect()
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
    let mut track_timescales = Vec::with_capacity(media.tracks.len());
    for (index, track) in media.tracks.iter().enumerate() {
        let (sample_entry, track_timescale) = match track.codec {
            VideoCodec::H264 => (
                SampleEntrySpec::Visual {
                    codec: VisualCodec::Avc,
                    config: track.config.clone(),
                    width: track.width.unwrap_or(1920),
                    height: track.height.unwrap_or(1080),
                },
                US_TIMESCALE,
            ),
            VideoCodec::Hevc => (
                SampleEntrySpec::Visual {
                    codec: VisualCodec::Hevc,
                    config: track.config.clone(),
                    width: track.width.unwrap_or(1920),
                    height: track.height.unwrap_or(1080),
                },
                US_TIMESCALE,
            ),
            VideoCodec::Av1 => (
                SampleEntrySpec::Visual {
                    codec: VisualCodec::Av1,
                    config: track.config.clone(),
                    width: track.width.unwrap_or(1920),
                    height: track.height.unwrap_or(1080),
                },
                US_TIMESCALE,
            ),
            VideoCodec::Vp9 => {
                // Canonical config is Matroska codec features; MP4 needs a
                // parsed vpcC record, never a byte-for-byte copy. Without
                // CodecPrivate, profile and bit depth come from a keyframe
                // header; an underivable stream fails before add_track.
                let vpcc = if let Some(vpcc) = vp9_codec_private_to_vpcc(&track.config) {
                    vpcc
                } else {
                    let color = media
                        .packets
                        .iter()
                        .filter(|packet| packet.track == index && packet.keyframe)
                        .find_map(|packet| vp9_color_from_keyframe(&packet.data))
                        .ok_or_else(|| {
                            container_error(
                                target_path,
                                "VP9 color config cannot be derived from CodecPrivate or keyframe",
                            )
                        })?;
                    let features = Vp9Features {
                        profile: Some(color.profile),
                        bit_depth: Some(color.bit_depth),
                        chroma_subsampling: Some(color.chroma_subsampling),
                        full_range: Some(color.full_range),
                        colour_primaries: Some(color.colour_primaries),
                        transfer: Some(color.transfer),
                        matrix: Some(color.matrix),
                        ..Vp9Features::default()
                    };
                    vp9_features_to_vpcc(&features).ok_or_else(|| {
                        container_error(target_path, "derived VP9 features exceed vpcC limits")
                    })?
                };
                (
                    SampleEntrySpec::Raw(build_vp09_sample_entry(
                        track.width.unwrap_or(0),
                        track.height.unwrap_or(0),
                        &vpcc,
                    )),
                    US_TIMESCALE,
                )
            }
            VideoCodec::Aac => {
                let sample_rate = track.sample_rate.unwrap_or(48_000).max(1);
                // Audio `mdhd` timescale must match the sample rate so MP4
                // priming `elst.media_time` maps to exact sample counts.
                (
                    SampleEntrySpec::Audio {
                        codec: AudioCodec::Aac,
                        config: build_aac_esds(&track.config),
                        channels: track.channels.unwrap_or(2),
                        sample_rate,
                    },
                    sample_rate,
                )
            }
            VideoCodec::Opus => (
                SampleEntrySpec::Audio {
                    codec: AudioCodec::Opus,
                    config: opus_head_to_dops(&track.config)
                        .map_err(|reason| container_error(target_path, reason))?,
                    channels: track.channels.unwrap_or(2),
                    sample_rate: 48_000,
                },
                48_000,
            ),
            VideoCodec::Mp3 => {
                let sample_rate = track.sample_rate.unwrap_or(44_100).max(1);
                (
                    SampleEntrySpec::Raw(build_mp3_mp4a_sample_entry(
                        track.channels.unwrap_or(2),
                        sample_rate,
                    )),
                    sample_rate,
                )
            }
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
        let mut spec = Mp4TrackSpec::new(kind, track_timescale, sample_entry);
        let default_dur_ticks = us_to_ticks(
            i64::try_from(track_duration(media, index)).unwrap_or(i64::MAX),
            track_timescale,
        )
        .max(1);
        spec.default_duration = Some(u32::try_from(default_dur_ticks).unwrap_or(1));
        if kind == TrackKind::Video {
            spec.display_width = u32::from(track.width.unwrap_or(0)) << 16;
            spec.display_height = u32::from(track.height.unwrap_or(0)) << 16;
        }
        track_timescales.push(track_timescale);
        ids.push(
            output
                .add_track(spec)
                .map_err(|err| containr_error(target_path, &err))?,
        );
    }
    // containr's MP4 muxer requires each track's first decode timestamp to be
    // zero and every following one to be strictly greater. Rebase each track's
    // DTS on its first packet in decode order (MPEG-TS starts at the PCR
    // offset) and its PTS on the earliest presentation time, so a reordered
    // B-frame never produces a negative presentation time. The edit list
    // restores each track's delay relative to the common timeline.
    let dts_origins = mp4_decode_origins(media);
    let pts_bases = mp4_presentation_bases(media, &dts_origins);
    let earliest_pts = mp4_earliest_presentation_times(media);
    let visible_starts = mp4_visible_starts(media, &pts_bases, &earliest_pts);
    // The common timeline starts at the earliest *visible* presentation time.
    // Decode preroll and audio priming packets sit before `visible_starts` and
    // must not pull the movie origin backward or shift sibling tracks.
    let timeline_origin = visible_starts.iter().flatten().copied().min().unwrap_or(0);
    let timestamps = mp4_rebased_timestamps(media, &dts_origins, &pts_bases);
    let mut last_packet_for_track = vec![None; media.tracks.len()];
    for (index, packet) in media.packets.iter().enumerate() {
        last_packet_for_track[packet.track] = Some(index);
    }
    let mut media_end: Vec<u64> = vec![0; media.tracks.len()];
    let mut prev_dts_ticks: Vec<Option<i64>> = vec![None; media.tracks.len()];
    for (packet_index, (packet, &(dts, pts))) in media.packets.iter().zip(&timestamps).enumerate() {
        let track_timescale = track_timescales[packet.track];
        // pts already includes the composition offset derived from the
        // original PTS, independent of any monotonic DTS adjustment.
        let end = pts
            .saturating_add(i64::try_from(packet.duration).unwrap_or(i64::MAX))
            .max(0);
        media_end[packet.track] =
            media_end[packet.track].max(u64::try_from(end).unwrap_or(u64::MAX));
        let raw_dts_ticks = us_to_ticks(dts, track_timescale);
        let dts_ticks = match prev_dts_ticks[packet.track] {
            None => 0,
            Some(prev) => raw_dts_ticks.max(prev.saturating_add(1)),
        };
        prev_dts_ticks[packet.track] = Some(dts_ticks);
        let pts_ticks = us_to_ticks(pts, track_timescale);
        let final_duration_ticks = (last_packet_for_track[packet.track] == Some(packet_index))
            .then(|| {
                u64::try_from(
                    us_to_ticks(
                        i64::try_from(packet.duration).unwrap_or(i64::MAX),
                        track_timescale,
                    )
                    .max(1),
                )
                .unwrap_or(1)
            });
        let input = containr::mp4::PacketIn {
            data: &packet.data,
            pts: pts_ticks,
            dts: dts_ticks,
            // Only the final packet needs an explicit duration; intermediate
            // stts deltas come from the next DTS without 1 µs rounding.
            duration: final_duration_ticks,
            sync: packet.keyframe,
        };
        output
            .write_packet(ids[packet.track], &input)
            .map_err(|err| containr_error(target_path, &err))?;
    }
    for (index, id) in ids.iter().enumerate() {
        let (Some(track_visible_start), Some(pts_base)) = (visible_starts[index], pts_bases[index])
        else {
            continue;
        };
        let delay = track_visible_start.saturating_sub(timeline_origin).max(0);
        let media_start = track_visible_start.saturating_sub(pts_base).max(0);
        let media_start_us = u64::try_from(media_start).unwrap_or(u64::MAX);
        let available_duration = media_end[index].saturating_sub(media_start_us);
        let explicit_duration = media.tracks[index]
            .presentation_edit
            .and_then(|edit| edit.duration_us);
        let trimmed_end = explicit_duration.is_some_and(|dur| dur < available_duration);
        let segment_duration =
            explicit_duration.map_or(available_duration, |dur| dur.min(available_duration));
        if delay == 0 && media_start == 0 && !trimmed_end {
            continue;
        }
        let delay_us = u64::try_from(delay).unwrap_or(u64::MAX);
        // `EditEntry.media_time` is expressed in the track's `mdhd` timescale
        // (sample rate for audio, microseconds for video), while
        // `segment_duration` is in `movie_timescale` (`US_TIMESCALE`).
        let media_time_ticks = us_to_ticks(media_start, track_timescales[index]);
        // An empty edit delays a track whose first visible presentation time is
        // later than the movie origin. `media_time_ticks` skips B-frame DTS
        // preroll, edit-list decode preroll, or audio priming samples so only
        // the visible presentation interval is played.
        let mut entries = Vec::with_capacity(2);
        if delay > 0 {
            entries.push(EditEntry {
                segment_duration: delay_us,
                media_time: EditEntry::EMPTY_MEDIA_TIME,
                media_rate: EditEntry::NORMAL_RATE,
            });
        }
        entries.push(EditEntry {
            segment_duration,
            media_time: media_time_ticks,
            media_rate: EditEntry::NORMAL_RATE,
        });
        let list = EditList {
            movie_timescale: US_TIMESCALE,
            entries,
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

/// Matroska has no edit list: a source edit whose visible window excludes
/// packets would require hiding those packets or collapsing their timestamps
/// onto a clamped base. Reject it explicitly instead of silently corrupting
/// the timeline.
fn validate_mkv_presentation_edits(
    media: &ContainerMedia,
    target_path: &Path,
) -> Result<(), PipelineError> {
    for (index, track) in media.tracks.iter().enumerate() {
        let Some(edit) = track.presentation_edit else {
            continue;
        };
        let visible_end = edit.duration_us.map(|duration| {
            edit.start_us
                .saturating_add(i64::try_from(duration).unwrap_or(i64::MAX))
        });
        let has_excluded_packet = media.packets.iter().any(|packet| {
            packet.track == index
                && (packet.pts < edit.start_us || visible_end.is_some_and(|end| packet.pts >= end))
        });
        if has_excluded_packet {
            return Err(container_error(
                target_path,
                format!(
                    "track {index} has an MP4 presentation edit with preroll or a trimmed duration that Matroska cannot represent; use MP4/MOV output"
                ),
            ));
        }
    }
    Ok(())
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
    validate_mkv_presentation_edits(media, target_path)?;
    let earliest_pts = mp4_earliest_presentation_times(media);
    let base = media
        .tracks
        .iter()
        .enumerate()
        .filter_map(|(index, track)| {
            let earliest = earliest_pts[index]?;
            Some(
                track
                    .presentation_edit
                    .map_or(earliest, |edit| edit.start_us),
            )
        })
        .min()
        .unwrap_or(0);
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

    fn vp9_media(config: Vec<u8>, keyframe: Vec<u8>) -> ContainerMedia {
        ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::Vp9,
                width: Some(640),
                height: Some(360),
                channels: None,
                sample_rate: None,
                config,
                timescale: US_TIMESCALE,
                presentation_edit: None,
            }],
            packets: vec![ContainerPacket {
                track: 0,
                pts: 0,
                dts: 0,
                duration: 33_333,
                keyframe: true,
                data: keyframe,
            }],
        }
    }

    fn vp9_muxed_vpcc(keyframe: &[u8]) -> Result<Vec<u8>, String> {
        let media = vp9_media(Vec::new(), keyframe.to_vec());
        let mut output = Cursor::new(Vec::new());
        let options = MuxOptions {
            brand: Brand::Isom,
            faststart: true,
            ..MuxOptions::default()
        };
        mux_mp4_into(&mut output, Path::new("vp9.mp4"), &media, options)
            .map_err(|err| err.to_string())?;
        let bytes = output.into_inner();
        let vpcc_offset = bytes
            .windows(4)
            .position(|window| window == b"vpcC")
            .ok_or("missing vpcC box")?;
        // Four bytes of version/flags precede the record in the box body.
        Ok(bytes[vpcc_offset + 4..].to_vec())
    }

    #[test]
    fn vp9_without_codec_private_derives_color_from_keyframe() -> Result<(), String> {
        let vpcc = vp9_muxed_vpcc(&[0x92, 0x49, 0x83, 0x42, 0x00])?;
        assert_eq!(vpcc[4], 2, "profile from keyframe");
        assert_eq!(vpcc[6], 0xA0, "10-bit 4:2:0 limited range");
        assert_eq!(&vpcc[7..10], &[2, 2, 2], "unspecified color primaries");
        Ok(())
    }

    #[test]
    fn vp9_keyframe_rgb_and_422_color_reach_vpcc() -> Result<(), String> {
        let rgb = vp9_muxed_vpcc(&[0xA2, 0x49, 0x83, 0x42, 0xE0])?;
        assert_eq!(rgb[4], 1, "profile 1");
        assert_eq!(rgb[6], 0x87, "8-bit 4:4:4 full range");
        assert_eq!(&rgb[7..10], &[1, 13, 0], "sRGB primaries, identity matrix");

        let four_two_two = vp9_muxed_vpcc(&[0xA2, 0x49, 0x83, 0x42, 0x48])?;
        assert_eq!(four_two_two[6], 0x84, "8-bit 4:2:2 limited range");
        assert_eq!(&four_two_two[7..10], &[1, 1, 1], "BT.709 matrix");
        Ok(())
    }

    #[test]
    fn vp9_without_derivable_color_fails_before_sample_entry() {
        for keyframe in [
            vec![0x82, 0x31, 0x01],
            vec![0x82, 0x49, 0x83, 0x42, 0xE0], // CS_RGB in profile 0
            vec![0xA2, 0x49, 0x83, 0x42, 0x44], // (0, 1) subsampling
            vec![0xA2, 0x49, 0x83, 0x42],       // truncated color_config
        ] {
            let media = vp9_media(Vec::new(), keyframe);
            let mut output = Cursor::new(Vec::new());
            let options = MuxOptions {
                brand: Brand::Isom,
                faststart: true,
                ..MuxOptions::default()
            };
            let result = mux_mp4_into(&mut output, Path::new("vp9.mp4"), &media, options);
            assert!(matches!(result, Err(PipelineError::RemuxFailed { .. })));
            assert!(
                !output.get_ref().windows(4).any(|w| w == b"vp09"),
                "no VP9 sample entry was written"
            );
        }
    }

    #[test]
    fn mp4_timestamps_keep_presentation_times_with_reordered_decode_order() {
        let packet = |pts: i64| ContainerPacket {
            track: 0,
            pts,
            dts: pts,
            duration: 33_333,
            keyframe: pts == 100_000,
            data: vec![0, 0, 0, 2, 0x65, 0x88],
        };
        let media = ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::H264,
                width: Some(320),
                height: Some(240),
                channels: None,
                sample_rate: None,
                config: Vec::new(),
                timescale: US_TIMESCALE,
                presentation_edit: None,
            }],
            packets: vec![packet(100_000), packet(200_000), packet(50_000)],
        };
        let origins = mp4_decode_origins(&media);
        assert_eq!(origins, vec![Some(100_000)]);
        let pts_bases = mp4_presentation_bases(&media, &origins);
        assert_eq!(pts_bases, vec![Some(50_000)]);
        let timestamps = mp4_rebased_timestamps(&media, &origins, &pts_bases);
        assert_eq!(
            timestamps,
            vec![(0, 50_000), (100_000, 150_000), (100_001, 0)]
        );
        // Presentation deltas stay exact and no frame gets a negative PTS.
        assert_eq!(timestamps[2].1 - timestamps[1].1, -150_000);
        assert!(timestamps.iter().all(|&(_, pts)| pts >= 0));
    }

    #[test]
    fn mp4_delayed_track_edit_list_preserves_early_b_frames() -> Result<(), String> {
        let video = |pts: i64| ContainerPacket {
            track: 1,
            pts,
            dts: pts,
            duration: 33_333,
            keyframe: pts == 100_000,
            data: vec![0, 0, 0, 2, 0x65, 0x88],
        };
        let media = ContainerMedia {
            tracks: vec![
                ContainerTrack {
                    codec: VideoCodec::Aac,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: vec![0x11, 0x90],
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                },
                ContainerTrack {
                    codec: VideoCodec::H264,
                    width: Some(320),
                    height: Some(240),
                    channels: None,
                    sample_rate: None,
                    config: extract_avcc_from_annex_b(&[
                        0, 0, 1, 0x67, 0x42, 0x00, 0x1E, 0, 0, 1, 0x68, 0xCE,
                    ])
                    .unwrap_or_default(),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                },
            ],
            packets: vec![
                ContainerPacket {
                    track: 0,
                    pts: 0,
                    dts: 0,
                    duration: 21_000,
                    keyframe: true,
                    data: vec![0xAA; 8],
                },
                video(100_000),
                video(200_000),
                video(50_000),
            ],
        };
        let origins = mp4_decode_origins(&media);
        assert_eq!(origins, vec![Some(0), Some(100_000)]);
        let pts_bases = mp4_presentation_bases(&media, &origins);
        assert_eq!(pts_bases, vec![Some(0), Some(50_000)]);
        let timeline_origin = pts_bases.iter().flatten().copied().min().unwrap_or(0);
        assert_eq!(timeline_origin, 0);
        assert_eq!(
            pts_bases[1].unwrap_or(0).saturating_sub(timeline_origin),
            50_000
        );
        let timestamps = mp4_rebased_timestamps(&media, &origins, &pts_bases);
        assert_eq!(timestamps[0], (0, 0));
        assert_eq!(timestamps[1], (0, 50_000));
        assert_eq!(timestamps[2], (100_000, 150_000));
        assert_eq!(timestamps[3], (100_001, 0));
        assert!(timestamps.iter().all(|&(_, pts)| pts >= 0));

        let mut output = Cursor::new(Vec::new());
        let options = MuxOptions {
            brand: Brand::Isom,
            faststart: true,
            movie_timescale: US_TIMESCALE,
            ..MuxOptions::default()
        };
        mux_mp4_into(&mut output, Path::new("delayed.mp4"), &media, options)
            .map_err(|err| err.to_string())?;
        let bytes = output.into_inner();
        let elst_offset = bytes
            .windows(4)
            .position(|window| window == b"elst")
            .ok_or("missing elst box")?;
        // Full-box version and flags, then entry_count, then the first entry.
        let version = bytes[elst_offset + 4];
        let entries = u32::from_be_bytes([
            bytes[elst_offset + 8],
            bytes[elst_offset + 9],
            bytes[elst_offset + 10],
            bytes[elst_offset + 11],
        ]);
        assert_eq!(version, 0);
        assert_eq!(entries, 2);
        let empty_duration = u32::from_be_bytes([
            bytes[elst_offset + 12],
            bytes[elst_offset + 13],
            bytes[elst_offset + 14],
            bytes[elst_offset + 15],
        ]);
        assert_eq!(
            empty_duration, 50_000,
            "video track's early B-frame plays at movie time 50 ms"
        );
        Ok(())
    }

    fn b_frame_preroll_media() -> ContainerMedia {
        const D: u64 = 33_333;
        const D_I64: i64 = 33_333;
        let video = |pts: i64, dts: i64, keyframe: bool| ContainerPacket {
            track: 1,
            pts,
            dts,
            duration: D,
            keyframe,
            data: vec![0, 0, 0, 2, 0x65, 0x88],
        };
        ContainerMedia {
            tracks: vec![
                ContainerTrack {
                    codec: VideoCodec::Aac,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: vec![0x11, 0x90],
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                },
                ContainerTrack {
                    codec: VideoCodec::H264,
                    width: Some(320),
                    height: Some(240),
                    channels: None,
                    sample_rate: None,
                    config: extract_avcc_from_annex_b(&[
                        0, 0, 1, 0x67, 0x42, 0x00, 0x1E, 0, 0, 1, 0x68, 0xCE,
                    ])
                    .unwrap_or_default(),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                },
            ],
            packets: vec![
                ContainerPacket {
                    track: 0,
                    pts: 0,
                    dts: 0,
                    duration: 21_000,
                    keyframe: true,
                    data: vec![0xAA; 8],
                },
                video(0, -2 * D_I64, true),
                video(2 * D_I64, -D_I64, false),
                video(D_I64, 0, false),
            ],
        }
    }

    #[test]
    fn mp4_b_frame_preroll_timestamps_stay_non_negative() {
        const D_I64: i64 = 33_333;
        let media = b_frame_preroll_media();
        let origins = mp4_decode_origins(&media);
        assert_eq!(origins, vec![Some(0), Some(-2 * D_I64)]);
        let pts_bases = mp4_presentation_bases(&media, &origins);
        assert_eq!(pts_bases, vec![Some(0), Some(-2 * D_I64)]);
        let earliest_pts = mp4_earliest_presentation_times(&media);
        assert_eq!(earliest_pts, vec![Some(0), Some(0)]);
        let timeline_origin = earliest_pts.iter().flatten().copied().min().unwrap_or(0);
        assert_eq!(timeline_origin, 0);
        let timestamps = mp4_rebased_timestamps(&media, &origins, &pts_bases);
        assert!(timestamps.iter().all(|&(_, pts)| pts >= 0));
        assert_eq!(timestamps[1], (0, 2 * D_I64));
        assert_eq!(timestamps[2], (D_I64, 4 * D_I64));
        assert_eq!(timestamps[3], (2 * D_I64, 3 * D_I64));
    }

    #[test]
    fn mp4_b_frame_preroll_edit_list_skips_negative_dts_without_delaying_audio()
    -> Result<(), String> {
        const D_U32: u32 = 33_333;
        let media = b_frame_preroll_media();
        let mut output = Cursor::new(Vec::new());
        let options = MuxOptions {
            brand: Brand::Isom,
            faststart: true,
            movie_timescale: US_TIMESCALE,
            ..MuxOptions::default()
        };
        mux_mp4_into(&mut output, Path::new("bframes.mp4"), &media, options)
            .map_err(|err| err.to_string())?;
        let bytes = output.into_inner();
        let elst_offsets: Vec<usize> = bytes
            .windows(4)
            .enumerate()
            .filter_map(|(offset, window)| (window == b"elst").then_some(offset))
            .collect();
        assert_eq!(
            elst_offsets.len(),
            1,
            "only the B-frame video track gets an elst"
        );
        let elst = elst_offsets[0];
        let version = bytes[elst + 4];
        let entries = u32::from_be_bytes([
            bytes[elst + 8],
            bytes[elst + 9],
            bytes[elst + 10],
            bytes[elst + 11],
        ]);
        assert_eq!(version, 0);
        assert_eq!(entries, 1, "no empty edit: audio and video start together");
        let segment_duration = u32::from_be_bytes([
            bytes[elst + 12],
            bytes[elst + 13],
            bytes[elst + 14],
            bytes[elst + 15],
        ]);
        let media_time = u32::from_be_bytes([
            bytes[elst + 16],
            bytes[elst + 17],
            bytes[elst + 18],
            bytes[elst + 19],
        ]);
        assert_eq!(media_time, 2 * D_U32, "elst skips the 2-frame DTS preroll");
        assert_eq!(
            segment_duration,
            3 * D_U32,
            "exact video presentation duration"
        );
        Ok(())
    }

    #[test]
    fn mp4_b_frame_preroll_roundtrips_through_demux_mp4_at_zero_origin() -> Result<(), String> {
        const D_I64: i64 = 33_333;
        let media = b_frame_preroll_media();
        let dir = std::env::temp_dir().join(format!("fastconv-mp4-elst-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
        let path = dir.join("bframes.mp4");
        let file = File::create(&path).map_err(|err| err.to_string())?;
        mux_video(&file, &path, VideoContainerFormat::Mp4, &media)
            .map_err(|err| err.to_string())?;
        drop(file);
        let reopened = demux_mp4(&path).map_err(|err| err.to_string())?;
        std::fs::remove_dir_all(&dir).map_err(|err| err.to_string())?;
        let audio_pts: Vec<i64> = reopened
            .packets
            .iter()
            .filter(|packet| packet.track == 0)
            .map(|packet| packet.pts)
            .collect();
        let video_stamps: Vec<(i64, i64)> = reopened
            .packets
            .iter()
            .filter(|packet| packet.track == 1)
            .map(|packet| (packet.pts, packet.dts))
            .collect();
        assert_eq!(audio_pts, vec![0]);
        assert_eq!(
            video_stamps,
            vec![(0, -2 * D_I64), (2 * D_I64, -D_I64), (D_I64, 0)]
        );
        Ok(())
    }

    #[test]
    fn mkv_b_frame_roundtrip_preserves_decode_order() -> Result<(), String> {
        const D: u64 = 25_000;
        let video = |pts: i64, dts: i64, keyframe: bool, tag: u8| ContainerPacket {
            track: 0,
            pts,
            dts,
            duration: D,
            keyframe,
            data: vec![0, 0, 0, 2, if keyframe { 0x65 } else { 0x41 }, tag],
        };
        let media = ContainerMedia {
            tracks: vec![ContainerTrack {
                codec: VideoCodec::H264,
                width: Some(320),
                height: Some(240),
                channels: None,
                sample_rate: None,
                config: extract_avcc_from_annex_b(&[
                    0, 0, 1, 0x67, 0x42, 0x00, 0x1E, 0, 0, 1, 0x68, 0xCE,
                ])
                .unwrap_or_default(),
                timescale: US_TIMESCALE,
                presentation_edit: None,
            }],
            // Decode order I0, P3, B1, B2.
            packets: vec![
                video(0, -25_000, true, 0),
                video(75_000, 0, false, 3),
                video(25_000, 25_000, false, 1),
                video(50_000, 50_000, false, 2),
            ],
        };
        let dir =
            std::env::temp_dir().join(format!("fastconv-mkv-bframe-order-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
        let path = dir.join("bframes.mkv");
        let file = File::create(&path).map_err(|err| err.to_string())?;
        mux_video(&file, &path, VideoContainerFormat::Mkv, &media)
            .map_err(|err| err.to_string())?;
        drop(file);
        let reopened = demux_mkv(&path).map_err(|err| err.to_string())?;
        std::fs::remove_dir_all(&dir).map_err(|err| err.to_string())?;
        let mut sorted: Vec<&ContainerPacket> = reopened.packets.iter().collect();
        sorted.sort_by_key(|packet| (packet.dts, packet.pts));
        let tags: Vec<u8> = sorted.iter().map(|packet| packet.data[5]).collect();
        assert_eq!(
            tags,
            vec![0, 3, 1, 2],
            "decode order I0, P3, B1, B2 must survive MKV reopen"
        );
        for (index, packet) in sorted.iter().enumerate() {
            assert!(packet.dts <= packet.pts);
            if index > 0 {
                assert!(packet.dts > sorted[index - 1].dts);
            }
        }
        Ok(())
    }

    fn write_edited_mp4(
        path: &Path,
        audio_packets: &[(i64, u64)],
        audio_edit: Option<&EditList>,
        video_packets: &[(i64, i64, u64, bool)],
        video_edit: Option<&EditList>,
    ) -> Result<(), String> {
        let avcc =
            extract_avcc_from_annex_b(&[0, 0, 1, 0x67, 0x42, 0x00, 0x1E, 0, 0, 1, 0x68, 0xCE])
                .unwrap_or_default();
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|err| err.to_string())?;
        let options = MuxOptions {
            brand: Brand::Isom,
            faststart: true,
            movie_timescale: US_TIMESCALE,
            ..MuxOptions::default()
        };
        let mut muxer = containr::mp4::Muxer::new(file, options).map_err(|err| err.to_string())?;
        let audio_id = muxer
            .add_track(Mp4TrackSpec::new(
                TrackKind::Audio,
                US_TIMESCALE,
                SampleEntrySpec::Audio {
                    codec: AudioCodec::Aac,
                    config: build_aac_esds(&[0x11, 0x90]),
                    channels: 2,
                    sample_rate: 48_000,
                },
            ))
            .map_err(|err| err.to_string())?;
        let video_id = muxer
            .add_track(Mp4TrackSpec::new(
                TrackKind::Video,
                US_TIMESCALE,
                SampleEntrySpec::Visual {
                    codec: VisualCodec::Avc,
                    config: avcc,
                    width: 320,
                    height: 240,
                },
            ))
            .map_err(|err| err.to_string())?;
        for &(pts, duration) in audio_packets {
            muxer
                .write_packet(
                    audio_id,
                    &containr::mp4::PacketIn {
                        data: &[0xAA; 8],
                        pts,
                        dts: pts,
                        duration: Some(duration),
                        sync: true,
                    },
                )
                .map_err(|err| err.to_string())?;
        }
        for &(dts, pts, duration, sync) in video_packets {
            muxer
                .write_packet(
                    video_id,
                    &containr::mp4::PacketIn {
                        data: &[0, 0, 0, 2, if sync { 0x65 } else { 0x41 }, 0x88],
                        pts,
                        dts,
                        duration: Some(duration),
                        sync,
                    },
                )
                .map_err(|err| err.to_string())?;
        }
        if let Some(list) = audio_edit {
            muxer
                .set_edit_list(audio_id, list)
                .map_err(|err| err.to_string())?;
        }
        if let Some(list) = video_edit {
            muxer
                .set_edit_list(video_id, list)
                .map_err(|err| err.to_string())?;
        }
        muxer.finish().map_err(|err| err.to_string())?;
        Ok(())
    }

    #[test]
    fn mp4_empty_edit_with_media_start_keeps_preroll_off_timeline_origin() -> Result<(), String> {
        let dir = std::env::temp_dir().join(format!("fastconv-mp4-preroll-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
        let source = dir.join("in.mp4");
        let remuxed = dir.join("out.mp4");

        // Audio plays from 0; video has an 80 ms empty edit followed by a media
        // edit starting at 120 ms, while raw video packets start at media time 0.
        let audio: Vec<(i64, u64)> = (0_i64..8).map(|idx| (idx * 21_333, 21_333)).collect();
        let video: Vec<(i64, i64, u64, bool)> = (0_i64..5)
            .map(|idx| (idx * 40_000, idx * 40_000, 40_000, idx == 0))
            .collect();
        let video_edit = EditList {
            movie_timescale: US_TIMESCALE,
            entries: vec![
                EditEntry {
                    segment_duration: 80_000,
                    media_time: EditEntry::EMPTY_MEDIA_TIME,
                    media_rate: EditEntry::NORMAL_RATE,
                },
                EditEntry {
                    segment_duration: 80_000,
                    media_time: 120_000,
                    media_rate: EditEntry::NORMAL_RATE,
                },
            ],
        };
        write_edited_mp4(&source, &audio, None, &video, Some(&video_edit))?;

        let demuxed = demux_mp4(&source).map_err(|err| err.to_string())?;
        assert_eq!(demuxed.tracks[0].presentation_edit, None);
        assert_eq!(
            demuxed.tracks[1].presentation_edit,
            Some(TrackPresentation {
                start_us: 80_000,
                duration_us: Some(80_000),
            })
        );
        let video_pts: Vec<i64> = demuxed
            .packets
            .iter()
            .filter(|packet| packet.track == 1)
            .map(|packet| packet.pts)
            .collect();
        assert_eq!(
            video_pts,
            vec![-40_000, 0, 40_000, 80_000, 120_000],
            "preroll packets are retained for decode with negative/pre-edit PTS"
        );

        let out_file = File::create(&remuxed).map_err(|err| err.to_string())?;
        mux_video(&out_file, &remuxed, VideoContainerFormat::Mp4, &demuxed)
            .map_err(|err| err.to_string())?;
        drop(out_file);

        let reader = WindowedFileReader::open(&remuxed).map_err(|err| err.to_string())?;
        let inspected = containr::mp4::Demuxer::open(reader).map_err(|err| err.to_string())?;
        let audio_track = inspected
            .tracks()
            .iter()
            .find(|t| t.kind == TrackKind::Audio)
            .ok_or("missing audio track")?;
        let video_track = inspected
            .tracks()
            .iter()
            .find(|t| t.kind == TrackKind::Video)
            .ok_or("missing video track")?;
        assert!(
            audio_track.edit_list.entries.is_empty(),
            "audio must remain at movie time 0 without a spurious delay edit"
        );
        assert_eq!(video_track.edit_list.entries.len(), 2);
        assert!(video_track.edit_list.entries[0].is_empty());
        assert_eq!(video_track.edit_list.entries[0].segment_duration, 80_000);
        assert_eq!(video_track.edit_list.entries[1].media_time, 120_000);
        assert_eq!(video_track.edit_list.entries[1].segment_duration, 80_000);

        std::fs::remove_dir_all(&dir).map_err(|err| err.to_string())?;
        Ok(())
    }

    #[test]
    fn mp4_audio_priming_edit_roundtrips_through_demux_and_mux() -> Result<(), String> {
        let dir =
            std::env::temp_dir().join(format!("fastconv-mp4-audio-priming-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
        let source = dir.join("in.mp4");
        let remuxed = dir.join("out.mp4");

        let audio: Vec<(i64, u64)> = (0_i64..4).map(|idx| (idx * 21_333, 21_333)).collect();
        let audio_edit = EditList {
            movie_timescale: US_TIMESCALE,
            entries: vec![EditEntry {
                segment_duration: 63_999,
                media_time: 21_333,
                media_rate: EditEntry::NORMAL_RATE,
            }],
        };
        let video = [
            (0_i64, 66_666_i64, 33_333_u64, true),
            (33_333, 133_332, 33_333, false),
            (66_666, 99_999, 33_333, false),
        ];
        let video_edit = EditList {
            movie_timescale: US_TIMESCALE,
            entries: vec![EditEntry {
                segment_duration: 99_999,
                media_time: 66_666,
                media_rate: EditEntry::NORMAL_RATE,
            }],
        };
        write_edited_mp4(
            &source,
            &audio,
            Some(&audio_edit),
            &video,
            Some(&video_edit),
        )?;

        let demuxed = demux_mp4(&source).map_err(|err| err.to_string())?;
        let out_file = File::create(&remuxed).map_err(|err| err.to_string())?;
        mux_video(&out_file, &remuxed, VideoContainerFormat::Mp4, &demuxed)
            .map_err(|err| err.to_string())?;
        drop(out_file);

        let reader = WindowedFileReader::open(&remuxed).map_err(|err| err.to_string())?;
        let inspected = containr::mp4::Demuxer::open(reader).map_err(|err| err.to_string())?;
        let audio_track = inspected
            .tracks()
            .iter()
            .find(|t| t.kind == TrackKind::Audio)
            .ok_or("missing audio track")?;
        let video_track = inspected
            .tracks()
            .iter()
            .find(|t| t.kind == TrackKind::Video)
            .ok_or("missing video track")?;
        assert_eq!(audio_track.timebase.den, 48_000);
        assert_eq!(audio_track.edit_list.entries.len(), 1);
        assert_eq!(
            audio_track.edit_list.entries[0].media_time, 1024,
            "audio priming edit must use sample-rate mdhd ticks (1024 samples at 48 kHz)"
        );
        assert_eq!(audio_track.edit_list.entries[0].segment_duration, 63_999);
        assert_eq!(video_track.edit_list.entries.len(), 1);
        assert_eq!(video_track.edit_list.entries[0].media_time, 66_666);
        assert_eq!(video_track.edit_list.entries[0].segment_duration, 99_999);

        std::fs::remove_dir_all(&dir).map_err(|err| err.to_string())?;
        Ok(())
    }

    #[test]
    fn mkv_dts_tie_repair_is_linear_and_rejects_signed_range_exhaustion() -> Result<(), String> {
        let count = 20_000_usize;
        let mut packets: Vec<ContainerPacket> = (0..count)
            .map(|idx| {
                let pts = if idx == 0 {
                    20
                } else if idx == 1 {
                    0
                } else {
                    i64::try_from(idx / 2).unwrap_or(0).saturating_mul(10)
                };
                ContainerPacket {
                    track: 0,
                    pts,
                    dts: pts,
                    duration: 10,
                    keyframe: idx == 0,
                    data: Vec::new(),
                }
            })
            .collect();
        let original_pts: Vec<i64> = packets.iter().map(|p| p.pts).collect();
        synthesize_mkv_decode_timestamps(Path::new("ties.mkv"), &mut packets, 1)
            .map_err(|err| err.to_string())?;
        for (index, packet) in packets.iter().enumerate() {
            assert_eq!(packet.pts, original_pts[index]);
            assert!(packet.dts <= packet.pts);
            if index > 0 {
                assert!(packet.dts > packets[index - 1].dts);
            }
        }

        let mut exhausted = vec![
            ContainerPacket {
                track: 0,
                pts: i64::MIN + 1,
                dts: i64::MIN + 1,
                duration: 1,
                keyframe: true,
                data: Vec::new(),
            },
            ContainerPacket {
                track: 0,
                pts: i64::MIN,
                dts: i64::MIN,
                duration: 1,
                keyframe: false,
                data: Vec::new(),
            },
            ContainerPacket {
                track: 0,
                pts: i64::MIN,
                dts: i64::MIN,
                duration: 1,
                keyframe: false,
                data: Vec::new(),
            },
        ];
        let result = synthesize_mkv_decode_timestamps(Path::new("min.mkv"), &mut exhausted, 1);
        assert!(matches!(result, Err(PipelineError::RemuxFailed { .. })));
        Ok(())
    }

    #[test]
    fn mux_mkv_rejects_unrepresentable_mp4_edit() -> Result<(), String> {
        let dir =
            std::env::temp_dir().join(format!("fastconv-mp4-edit-mkv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
        let source = dir.join("in.mp4");
        let target = dir.join("out.mkv");

        let audio: Vec<(i64, u64)> = (0_i64..8).map(|idx| (idx * 21_333, 21_333)).collect();
        let video: Vec<(i64, i64, u64, bool)> = (0_i64..5)
            .map(|idx| (idx * 40_000, idx * 40_000, 40_000, idx == 0))
            .collect();
        let video_edit = EditList {
            movie_timescale: US_TIMESCALE,
            entries: vec![
                EditEntry {
                    segment_duration: 80_000,
                    media_time: EditEntry::EMPTY_MEDIA_TIME,
                    media_rate: EditEntry::NORMAL_RATE,
                },
                EditEntry {
                    segment_duration: 80_000,
                    media_time: 120_000,
                    media_rate: EditEntry::NORMAL_RATE,
                },
            ],
        };
        write_edited_mp4(&source, &audio, None, &video, Some(&video_edit))?;
        let demuxed = demux_mp4(&source).map_err(|err| err.to_string())?;
        let out_file = File::create(&target).map_err(|err| err.to_string())?;
        let result = mux_video(&out_file, &target, VideoContainerFormat::Mkv, &demuxed);
        drop(out_file);
        assert!(
            matches!(result, Err(PipelineError::RemuxFailed { .. })),
            "an edit with preroll must be rejected instead of clamping timestamps"
        );

        std::fs::remove_dir_all(&dir).map_err(|err| err.to_string())?;
        Ok(())
    }
}
