use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use containr::CodecConfig;
use containr::Track;
use containr::TrackKind;
use fastconv_domain::{
    AudioTrackAction, DomainError, HardwareBackend, PathRole, TargetAudioCodec, TargetVideoCodec,
    VideoCodec, VideoContainerFormat, VideoEncodeMode, VideoQuality, VideoRemuxItem,
    VideoRemuxPlan, VideoStreamInfo, VideoStreamKind, VideoTrackAction, VideoTranscodeItem,
    VideoTranscodePlan,
};

use crate::{ProbeError, available_cpu_threads};

/// Three 188-byte TS or 192-byte M2TS packets, plus EBML/`ftyp`/RIFF headers.
const MAGIC_PREFIX_BYTES: usize = 576;

/// Window size used by [`WindowedFileReader`].
const WINDOW_BYTES: usize = 1024 * 1024;

/// Upper bound for the AVI header walk; protects against hostile chunk sizes.
const AVI_HEADER_CAP: usize = 4 * 1024 * 1024;

/// Upper bound for the MPEG-TS table/PES scan.
const TS_SCAN_CAP: usize = 2 * 1024 * 1024;

/// Top-level ISO base media boxes (other than `ftyp`) that identify `QuickTime` MOV.
const MOV_TOP_LEVEL_BOXES: [&[u8; 4]; 9] = [
    b"moov", b"mdat", b"free", b"skip", b"wide", b"pnot", b"udta", b"cmov", b"rmra",
];

/// ISOBMFF major brands for still images; never media containers for us.
const IMAGE_MAJOR_BRANDS: [&[u8; 4]; 9] = [
    b"avif", b"avis", b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"jxl ",
];

/// Validated video metadata produced at the file-system boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbedVideo {
    pub path: PathBuf,
    pub format: VideoContainerFormat,
    pub streams: Vec<VideoStreamInfo>,
}

fn invalid_video(path: &Path, reason: impl Into<String>) -> ProbeError {
    ProbeError::InvalidVideo {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

/// Decodes an EBML VINT used as an element size: `(value, encoded_length)`.
fn ebml_vint_size(bytes: &[u8]) -> Option<(usize, usize)> {
    let first = *bytes.first()?;
    if first == 0 {
        return None;
    }
    let len = usize::try_from(first.leading_zeros())
        .ok()?
        .checked_add(1)?;
    if len > 8 || len > bytes.len() {
        return None;
    }
    let marker = 1_u8 << (8 - len);
    let mut value = u64::from(first & !marker);
    for &byte in &bytes[1..len] {
        value = (value << 8) | u64::from(byte);
    }
    usize::try_from(value).ok().map(|size| (size, len))
}

/// Finds the `DocType` payload inside the EBML header, bounded by the prefix.
fn ebml_doc_type(bytes: &[u8]) -> Option<&[u8]> {
    let (header_size, size_len) = ebml_vint_size(bytes.get(4..)?)?;
    let mut idx = 4_usize.checked_add(size_len)?;
    let header_end = idx.checked_add(header_size)?.min(bytes.len());
    while idx + 2 <= header_end {
        let id_len = usize::try_from(bytes[idx].leading_zeros())
            .ok()?
            .checked_add(1)?;
        if id_len > 4 {
            return None;
        }
        let id_end = idx.checked_add(id_len)?;
        if id_end > header_end {
            return None;
        }
        let id = &bytes[idx..id_end];
        idx = id_end;
        let (payload_size, vint_len) = ebml_vint_size(bytes.get(idx..header_end)?)?;
        idx = idx.checked_add(vint_len)?;
        let payload_end = idx.checked_add(payload_size)?;
        if payload_end > header_end {
            return None;
        }
        if id == [0x42, 0x82] {
            return Some(&bytes[idx..payload_end]);
        }
        idx = payload_end;
    }
    None
}

fn trim_trailing_nuls(mut bytes: &[u8]) -> &[u8] {
    while bytes.last() == Some(&0) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}

/// Detects the container from leading bytes only; file extensions are never trusted.
#[must_use]
pub fn detect_video_magic(bytes: &[u8]) -> Option<VideoContainerFormat> {
    // Matroska / WebM start with the EBML header ID; DocType separates them.
    if bytes.len() >= 4 && bytes[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
        let doc_type = ebml_doc_type(bytes).map_or(&[][..], trim_trailing_nuls);
        return Some(if doc_type == b"webm" {
            VideoContainerFormat::Webm
        } else {
            VideoContainerFormat::Mkv
        });
    }

    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"AVI " {
        return Some(VideoContainerFormat::Avi);
    }

    if bytes.len() >= 8 && &bytes[4..8] == b"ftyp" {
        if bytes.len() >= 12 {
            let brand = &bytes[8..12];
            if IMAGE_MAJOR_BRANDS.iter().any(|known| *known == brand) {
                return None;
            }
            return Some(if brand == b"qt  " {
                VideoContainerFormat::Mov
            } else {
                VideoContainerFormat::Mp4
            });
        }
        return Some(VideoContainerFormat::Mp4);
    }

    if bytes.len() >= 8
        && MOV_TOP_LEVEL_BOXES
            .iter()
            .any(|box_type| *box_type == &bytes[4..8])
    {
        return Some(VideoContainerFormat::Mov);
    }

    if bytes.len() >= 377 && bytes[0] == 0x47 && bytes[188] == 0x47 && bytes[376] == 0x47 {
        return Some(VideoContainerFormat::Ts);
    }
    if bytes.len() >= 389 && bytes[4] == 0x47 && bytes[196] == 0x47 && bytes[388] == 0x47 {
        return Some(VideoContainerFormat::Ts);
    }
    None
}

/// A `Read + Seek` adapter that keeps a 1 MiB window of the file in memory.
///
/// `containr` seeks to an absolute offset before every EBML/box header and
/// packet. `BufReader::seek(Start(..))` discards its buffer on every such call,
/// which turns a clip into thousands of tiny reads; this reader only refills
/// the window when the requested position falls outside it.
struct WindowedFileReader {
    file: File,
    file_len: u64,
    pos: u64,
    buf: Vec<u8>,
    buf_start: u64,
    buf_valid: usize,
}

impl WindowedFileReader {
    #[cfg(test)]
    fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        Ok(Self::from_file(file, file_len))
    }

    fn from_file(file: File, file_len: u64) -> Self {
        let window_len = u64::try_from(WINDOW_BYTES).unwrap_or(u64::MAX);
        let capacity = usize::try_from(file_len.min(window_len)).unwrap_or(WINDOW_BYTES);
        Self {
            file,
            file_len,
            pos: 0,
            buf: vec![0_u8; capacity],
            buf_start: 0,
            buf_valid: 0,
        }
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
                // `containr::*::Demuxer::open` probes the input length with
                // `seek(SeekFrom::End(0))`; resolve it from metadata with no I/O.
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

fn read_magic_prefix(
    file: &mut File,
    path: &Path,
) -> Result<([u8; MAGIC_PREFIX_BYTES], usize), ProbeError> {
    let mut prefix = [0_u8; MAGIC_PREFIX_BYTES];
    let mut filled = 0_usize;
    while filled < prefix.len() {
        match file.read(&mut prefix[filled..]) {
            Ok(0) => break,
            Ok(count) => filled += count,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(source) => {
                return Err(ProbeError::Io {
                    path: path.to_path_buf(),
                    source,
                });
            }
        }
    }
    Ok((prefix, filled))
}

/// Reads at most `cap` bytes from the start of `file` for bounded header parsing.
fn read_bounded_prefix(
    file: &mut File,
    path: &Path,
    file_len: u64,
    cap: usize,
) -> Result<Vec<u8>, ProbeError> {
    file.seek(SeekFrom::Start(0))
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let capacity = usize::try_from(file_len).unwrap_or(usize::MAX).min(cap);
    let mut bytes = Vec::with_capacity(capacity);
    let limit = u64::try_from(cap).unwrap_or(u64::MAX);
    file.by_ref()
        .take(limit)
        .read_to_end(&mut bytes)
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(bytes)
}

fn stream_kind(track: &Track) -> VideoStreamKind {
    match track.kind {
        TrackKind::Video => VideoStreamKind::Video,
        TrackKind::Audio => VideoStreamKind::Audio,
        TrackKind::Subtitle => VideoStreamKind::Subtitle,
        TrackKind::Data => VideoStreamKind::Data,
        TrackKind::Other => VideoStreamKind::Other,
    }
}

/// Finds the payload of the first MPEG-4 descriptor with `tag` in `bytes`.
fn mpeg4_descriptor(bytes: &[u8], tag: u8) -> Option<&[u8]> {
    let mut idx = 0_usize;
    while idx < bytes.len() {
        let found = bytes[idx];
        idx += 1;
        // MPEG-4 descriptor sizes are big-endian base-128: every byte but the
        // last has the continuation bit set.
        let mut size = 0_usize;
        let mut parts = 0_u8;
        loop {
            let byte = *bytes.get(idx)?;
            idx += 1;
            size = size.checked_shl(7)?.checked_add(usize::from(byte & 0x7F))?;
            if byte & 0x80 == 0 {
                break;
            }
            parts += 1;
            if parts >= 4 {
                return None;
            }
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

/// Reads `objectTypeIndication` from an `esds` box body.
fn esds_object_type(esds: &[u8]) -> Option<u8> {
    let body = esds.get(4..)?; // version and flags
    let es = mpeg4_descriptor(body, 0x03)?;
    let flags = *es.get(2)?; // ES_ID (2 bytes) then flags
    let mut idx = 3_usize;
    if flags & 0x80 != 0 {
        idx = idx.checked_add(2)?; // streamDependenceFlag
    }
    if flags & 0x40 != 0 {
        let url_len = usize::from(*es.get(idx)?);
        idx = idx.checked_add(1)?.checked_add(url_len)?;
    }
    if flags & 0x20 != 0 {
        idx = idx.checked_add(2)?; // OCRstreamFlag
    }
    mpeg4_descriptor(es.get(idx..)?, 0x04)?.first().copied()
}

fn stream_codec(track: &Track) -> VideoCodec {
    if let Some(codec_id) = track.codec_id() {
        // Matroska CodecID.
        if codec_id.contains("AVC") || codec_id == "V_MPEG4/ISO/AVC" {
            return VideoCodec::H264;
        }
        if codec_id.contains("HEVC") {
            return VideoCodec::Hevc;
        }
        if codec_id == "V_AV1" {
            return VideoCodec::Av1;
        }
        if codec_id == "V_VP9" {
            return VideoCodec::Vp9;
        }
        if codec_id == "A_MPEG/L3" {
            return VideoCodec::Mp3;
        }
        if codec_id == "A_AAC" {
            return VideoCodec::Aac;
        }
        if codec_id == "A_OPUS" {
            return VideoCodec::Opus;
        }
        if codec_id == "A_VORBIS" {
            return VideoCodec::Vorbis;
        }
        if codec_id == "S_TEXT/UTF8" {
            return VideoCodec::TimedText;
        }
        return VideoCodec::Unknown;
    }

    // MP4/MOV sample entry fourcc.
    match track.codec().map(|fourcc| fourcc.0) {
        Some([b'a', b'v', b'c', b'1' | b'3']) => VideoCodec::H264,
        Some([b'h', b'v', b'c', b'1'] | [b'h', b'e', b'v', b'1']) => VideoCodec::Hevc,
        Some([b'a', b'v', b'0', b'1']) => VideoCodec::Av1,
        Some([b'v', b'p', b'0', b'9']) => VideoCodec::Vp9,
        Some([b'.', b'm', b'p', b'3']) => VideoCodec::Mp3,
        Some([b'm', b'p', b'4', b'a']) => {
            // `mp4a` carries either AAC or MPEG-4/MP3 in its ES descriptor.
            match track
                .sample_entries
                .first()
                .and_then(|e| e.codec_config.as_ref())
            {
                Some(CodecConfig::Esds(esds)) => match esds_object_type(esds) {
                    Some(0x6B | 0x69) => VideoCodec::Mp3,
                    Some(_) | None => VideoCodec::Aac,
                },
                _ => VideoCodec::Aac,
            }
        }
        Some([b'O', b'p', b'u', b's']) => VideoCodec::Opus,
        Some([b't', b'x', b'3', b'g']) => VideoCodec::TimedText,
        _ => VideoCodec::Unknown,
    }
}

fn dimensions(track: &Track) -> (Option<u32>, Option<u32>) {
    if let Some(video) = track.video() {
        return (Some(u32::from(video.width)), Some(u32::from(video.height)));
    }
    if let Some(mkv) = &track.matroska
        && let Some(video) = &mkv.video
    {
        return (
            u32::try_from(video.pixel_width).ok(),
            u32::try_from(video.pixel_height).ok(),
        );
    }
    (None, None)
}

// Matroska stores sampling frequency as f64; the range is validated before conversion.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn audio_facts(track: &Track) -> (Option<u16>, Option<u32>) {
    if let Some(audio) = track.audio() {
        return (u16::try_from(audio.channels).ok(), Some(audio.sample_rate));
    }
    if let Some(mkv) = &track.matroska
        && let Some(audio) = &mkv.audio
    {
        // Matroska records sample rate as f64; canonical rates are integral.
        let rate = audio.sampling_frequency;
        let rate = if rate.is_finite() && rate > 0.0 && rate <= f64::from(u32::MAX) {
            Some(rate as u32)
        } else {
            None
        };
        return (u16::try_from(audio.channels).ok(), rate);
    }
    (None, None)
}

fn stream_info(track: &Track) -> VideoStreamInfo {
    let (width, height) = dimensions(track);
    let (channels, sample_rate) = audio_facts(track);
    let language = track.language.iso639_mapped().map(str::to_owned);
    VideoStreamInfo::new(
        stream_codec(track),
        stream_kind(track),
        width,
        height,
        channels,
        sample_rate,
        track.timebase.num,
        track.timebase.den,
        track.duration,
        language,
    )
}

fn le_u16(bytes: &[u8]) -> Option<u16> {
    Some(u16::from_le_bytes(bytes.get(..2)?.try_into().ok()?))
}

fn le_u32(bytes: &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?))
}

fn be_u16(bytes: &[u8]) -> Option<u16> {
    Some(u16::from_be_bytes(bytes.get(..2)?.try_into().ok()?))
}

/// Visits RIFF chunks as `(id, payload)`, stopping at the first truncated chunk.
fn walk_riff_chunks<'a>(bytes: &'a [u8], mut visit: impl FnMut([u8; 4], &'a [u8])) {
    let mut idx = 0_usize;
    while idx + 8 <= bytes.len() {
        let Ok(id) = <[u8; 4]>::try_from(&bytes[idx..idx + 4]) else {
            return;
        };
        let Some(size) = le_u32(&bytes[idx + 4..idx + 8]).and_then(|v| usize::try_from(v).ok())
        else {
            return;
        };
        let payload_start = idx + 8;
        let Some(payload_end) = payload_start.checked_add(size) else {
            return;
        };
        if payload_end > bytes.len() {
            return;
        }
        visit(id, &bytes[payload_start..payload_end]);
        let Some(padded) = size.checked_add(size & 1) else {
            return;
        };
        let Some(next) = payload_start.checked_add(padded) else {
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

fn fourcc_in(fourcc: [u8; 4], known: &[&[u8; 4]]) -> bool {
    known.iter().any(|candidate| **candidate == fourcc)
}

fn avi_video_codec(compression: [u8; 4], handler: Option<[u8; 4]>) -> VideoCodec {
    for fourcc in handler.into_iter().chain(std::iter::once(compression)) {
        if fourcc_in(fourcc, &[b"H264", b"X264", b"AVC1", b"DAVC", b"VSSH"]) {
            return VideoCodec::H264;
        }
        if fourcc_in(fourcc, &[b"HEVC", b"H265", b"HVC1", b"HEV1", b"X265"]) {
            return VideoCodec::Hevc;
        }
        if fourcc_in(fourcc, &[b"AV01"]) {
            return VideoCodec::Av1;
        }
        if fourcc_in(fourcc, &[b"VP90", b"VP09"]) {
            return VideoCodec::Vp9;
        }
    }
    VideoCodec::Unknown
}

fn avi_audio_codec(format_tag: u16, handler: Option<[u8; 4]>) -> VideoCodec {
    if handler == Some(*b"OPUS") {
        return VideoCodec::Opus;
    }
    match format_tag {
        0x00FF | 0x1600 | 0x1601 | 0x1602 | 0x706D => VideoCodec::Aac,
        0x704F | 0x7361 => VideoCodec::Opus,
        0x0055 => VideoCodec::Mp3,
        _ => VideoCodec::Unknown,
    }
}

#[derive(Default)]
struct AvihInfo {
    micro_sec_per_frame: u32,
    total_frames: u32,
    width: u32,
    height: u32,
}

/// Parses a RIFF/AVI `hdrl` header with a pure-Rust, bounded chunk walk.
fn probe_avi(
    path: &Path,
    file: &mut File,
    file_len: u64,
) -> Result<Vec<VideoStreamInfo>, ProbeError> {
    let bytes = read_bounded_prefix(file, path, file_len, AVI_HEADER_CAP)?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"AVI " {
        return Err(invalid_video(path, "truncated RIFF/AVI header"));
    }

    let mut avih = AvihInfo::default();
    let mut strl_chunks: Vec<&[u8]> = Vec::new();
    walk_riff_chunks(&bytes[12..], |id, payload| {
        if id != *b"LIST" || payload.len() < 4 || &payload[..4] != b"hdrl" {
            return;
        }
        walk_riff_chunks(&payload[4..], |kind, body| {
            if kind == *b"avih" && body.len() >= 56 {
                avih = AvihInfo {
                    micro_sec_per_frame: le_u32(&body[..4]).unwrap_or(0),
                    total_frames: le_u32(&body[16..20]).unwrap_or(0),
                    width: le_u32(&body[32..36]).unwrap_or(0),
                    height: le_u32(&body[36..40]).unwrap_or(0),
                };
            } else if kind == *b"LIST" && body.len() >= 4 && &body[..4] == b"strl" {
                strl_chunks.push(&body[4..]);
            }
        });
    });
    if strl_chunks.is_empty() {
        return Err(invalid_video(path, "AVI header has no streams"));
    }

    let mut streams = Vec::with_capacity(strl_chunks.len());
    for chunk in strl_chunks {
        if let Some(info) = avi_stream_info(chunk, &avih) {
            streams.push(info);
        }
    }
    if streams.is_empty() {
        return Err(invalid_video(path, "AVI header has no usable streams"));
    }
    Ok(streams)
}

/// Converts one AVI `strl` list into a validated stream, ignoring unknown stream types.
fn avi_stream_info(chunk: &[u8], avih: &AvihInfo) -> Option<VideoStreamInfo> {
    let mut fcc_type = [0_u8; 4];
    let mut handler = None;
    let mut scale = 1_u32;
    let mut rate = 1_u32;
    let mut length = 0_u32;
    let mut strf = None;
    walk_riff_chunks(chunk, |kind, body| {
        if kind == *b"strh" && body.len() >= 36 {
            fcc_type = <[u8; 4]>::try_from(&body[..4]).unwrap_or([0; 4]);
            handler = upper_fourcc(&body[4..8]);
            scale = le_u32(&body[20..24]).unwrap_or(1);
            rate = le_u32(&body[24..28]).unwrap_or(1);
            length = le_u32(&body[32..36]).unwrap_or(0);
        } else if kind == *b"strf" {
            strf = Some(body);
        }
    });

    let is_video = fcc_type == *b"vids";
    let is_audio = fcc_type == *b"auds";
    if !is_video && !is_audio {
        return None;
    }
    let (codec, width, height, channels, sample_rate) = if is_video {
        let mut width = None;
        let mut height = None;
        let mut codec = avi_video_codec([0; 4], handler);
        if let Some(body) = strf
            && body.len() >= 40
        {
            width = le_i32_unsigned(&body[4..8]);
            height = le_i32_unsigned(&body[8..12]);
            codec = avi_video_codec(
                <[u8; 4]>::try_from(&body[16..20]).unwrap_or([0; 4]),
                handler,
            );
        }
        let width = width
            .filter(|&w| w > 0)
            .or((avih.width > 0).then_some(avih.width));
        let height = height
            .filter(|&h| h > 0)
            .or((avih.height > 0).then_some(avih.height));
        (codec, width, height, None, None)
    } else {
        let mut codec = avi_audio_codec(0, handler);
        let mut channels = None;
        let mut sample_rate = None;
        if let Some(body) = strf
            && body.len() >= 16
        {
            codec = avi_audio_codec(le_u16(&body[..2]).unwrap_or(0), handler);
            channels = le_u16(&body[2..4]);
            sample_rate = le_u32(&body[4..8]);
        }
        (codec, None, None, channels, sample_rate)
    };

    let duration = if length > 0 {
        Some(u64::from(length))
    } else if is_video && avih.total_frames > 0 {
        Some(u64::from(avih.total_frames))
    } else {
        None
    };
    let (timebase_num, timebase_den) = if scale > 0 && rate > 0 {
        (u64::from(scale), u64::from(rate))
    } else if avih.micro_sec_per_frame > 0 {
        // Exact rational instead of truncating 1_000_000 / us_per_frame.
        (u64::from(avih.micro_sec_per_frame), 1_000_000)
    } else {
        (1, 1)
    };
    Some(VideoStreamInfo::new(
        codec,
        if is_video {
            VideoStreamKind::Video
        } else {
            VideoStreamKind::Audio
        },
        width,
        height,
        channels,
        sample_rate,
        u32::try_from(timebase_num).unwrap_or(u32::MAX),
        u32::try_from(timebase_den).unwrap_or(u32::MAX),
        duration,
        None,
    ))
}

fn le_i32_unsigned(bytes: &[u8]) -> Option<u32> {
    let value = i32::from_le_bytes(bytes.get(..4)?.try_into().ok()?);
    Some(value.unsigned_abs())
}

const ADTS_SAMPLE_RATES: [u32; 16] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350, 0, 0, 0,
];

fn ts_stride(bytes: &[u8]) -> Option<usize> {
    if bytes.len() >= 377 && bytes[0] == 0x47 && bytes[188] == 0x47 && bytes[376] == 0x47 {
        return Some(188);
    }
    if bytes.len() >= 389 && bytes[4] == 0x47 && bytes[196] == 0x47 && bytes[388] == 0x47 {
        return Some(192);
    }
    None
}

struct TsPacket<'a> {
    pid: u16,
    pusi: bool,
    payload: &'a [u8],
}

fn ts_packet(packet: &[u8]) -> Option<TsPacket<'_>> {
    if packet.len() < 188 || packet[0] != 0x47 {
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
        if idx > 188 {
            return None;
        }
    }
    Some(TsPacket {
        pid,
        pusi,
        payload: &packet[idx..],
    })
}

/// Skips the `pointer_field` that precedes a table or PES start.
fn ts_section_start<'a>(packet: &TsPacket<'a>) -> Option<&'a [u8]> {
    if !packet.pusi {
        return None;
    }
    let pointer = usize::from(*packet.payload.first()?);
    packet.payload.get(1_usize.checked_add(pointer)?..)
}

fn ts_section_length(section: &[u8]) -> Option<usize> {
    let high = usize::from(*section.get(1)? & 0x0F);
    let low = usize::from(*section.get(2)?);
    high.checked_mul(256)?.checked_add(low)
}

fn parse_pat(section: &[u8]) -> Option<u16> {
    if *section.first()? != 0x00 || *section.get(1)? & 0x80 == 0 {
        return None;
    }
    let end = 3_usize.checked_add(ts_section_length(section)?)?;
    if end > section.len() {
        return None;
    }
    let crc_end = end.checked_sub(4)?;
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

#[derive(Clone, Copy)]
struct TsStream {
    pid: u16,
    codec: VideoCodec,
    kind: VideoStreamKind,
    sample_rate: Option<u32>,
    channels: Option<u16>,
}

fn ts_descriptor(bytes: &[u8], tag: u8) -> Option<&[u8]> {
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

fn map_ts_stream(stream_type: u8, descriptors: &[u8]) -> TsStreamInfo {
    match stream_type {
        0x1B => TsStreamInfo::new(VideoCodec::H264, VideoStreamKind::Video),
        0x24 => TsStreamInfo::new(VideoCodec::Hevc, VideoStreamKind::Video),
        0x0F => TsStreamInfo::new(VideoCodec::Aac, VideoStreamKind::Audio),
        0x03 | 0x04 => TsStreamInfo::new(VideoCodec::Mp3, VideoStreamKind::Audio),
        0x06 => {
            let opus_channels = ts_descriptor(descriptors, 0x7F)
                .filter(|ext| ext.len() >= 2 && ext[0] == 0x80)
                .map(|ext| u16::from(ext[1] & 0x0F).max(1));
            if let Some(reg) = ts_descriptor(descriptors, 0x05)
                && reg.len() >= 4
            {
                if &reg[..4] == b"Opus" {
                    return TsStreamInfo::audio(
                        VideoCodec::Opus,
                        Some(48_000),
                        Some(opus_channels.unwrap_or(2)),
                    );
                }
                if &reg[..4] == b"AV01" {
                    return TsStreamInfo::new(VideoCodec::Av1, VideoStreamKind::Video);
                }
            }
            if let Some(channels) = opus_channels {
                return TsStreamInfo::audio(VideoCodec::Opus, Some(48_000), Some(channels));
            }
            // The AV1 descriptor is unambiguous inside a private-data stream.
            if ts_descriptor(descriptors, 0x80).is_some() {
                return TsStreamInfo::new(VideoCodec::Av1, VideoStreamKind::Video);
            }
            TsStreamInfo::new(VideoCodec::Unknown, VideoStreamKind::Other)
        }
        _ => TsStreamInfo::new(VideoCodec::Unknown, VideoStreamKind::Other),
    }
}

struct TsStreamInfo {
    codec: VideoCodec,
    kind: VideoStreamKind,
    sample_rate: Option<u32>,
    channels: Option<u16>,
}

impl TsStreamInfo {
    fn new(codec: VideoCodec, kind: VideoStreamKind) -> Self {
        Self {
            codec,
            kind,
            sample_rate: None,
            channels: None,
        }
    }

    fn audio(codec: VideoCodec, sample_rate: Option<u32>, channels: Option<u16>) -> Self {
        Self {
            codec,
            kind: VideoStreamKind::Audio,
            sample_rate,
            channels,
        }
    }
}

fn parse_pmt(section: &[u8]) -> Option<Vec<TsStream>> {
    if *section.first()? != 0x02 || *section.get(1)? & 0x80 == 0 {
        return None;
    }
    let end = 3_usize.checked_add(ts_section_length(section)?)?;
    if end > section.len() {
        return None;
    }
    let crc_end = end.checked_sub(4)?;
    let program_info_length =
        (usize::from(*section.get(10)? & 0x0F) << 8) | usize::from(*section.get(11)?);
    let mut idx = 12_usize.checked_add(program_info_length)?;
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
        let info = map_ts_stream(section[idx], &section[idx + 5..desc_end]);
        let pid = (u16::from(section[idx + 1] & 0x1F) << 8) | u16::from(section[idx + 2]);
        streams.push(TsStream {
            pid,
            codec: info.codec,
            kind: info.kind,
            sample_rate: info.sample_rate,
            channels: info.channels,
        });
        idx = desc_end;
    }
    (!streams.is_empty()).then_some(streams)
}

fn pes_payload<'a>(packet: &TsPacket<'a>) -> Option<&'a [u8]> {
    if !packet.pusi {
        return None;
    }
    // PES packets start directly at the TS payload with 0x000001 (no PSI pointer_field).
    let start = packet.payload;
    if start.get(..3)? != [0_u8, 0, 1] {
        return None;
    }
    let payload_start = 9_usize.checked_add(usize::from(*start.get(8)?))?;
    start.get(payload_start..)
}

fn adts_facts(bytes: &[u8]) -> Option<(u32, u16)> {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] & 0xF0 != 0xF0 {
        return None;
    }
    let rate = ADTS_SAMPLE_RATES[usize::from((bytes[2] >> 2) & 0x0F)];
    let config = ((bytes[2] & 0x01) << 2) | (bytes[3] >> 6);
    let channels = match config {
        1..=6 => Some(u16::from(config)),
        7 => Some(8),
        _ => None,
    }?;
    (rate > 0).then_some((rate, channels))
}

fn mp3_facts(bytes: &[u8]) -> Option<(u32, u16)> {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] & 0xE0 != 0xE0 {
        return None;
    }
    let version = (bytes[1] >> 3) & 0x03;
    let layer = (bytes[1] >> 1) & 0x03;
    let bitrate = (bytes[2] >> 4) & 0x0F;
    let rate_idx = usize::from((bytes[2] >> 2) & 0x03);
    if version == 1 || layer == 0 || bitrate == 0 || bitrate == 15 || rate_idx == 3 {
        return None;
    }
    let rates: [u32; 3] = match version {
        3 => [44_100, 48_000, 32_000],
        2 => [22_050, 24_000, 16_000],
        _ => [11_025, 12_000, 8_000],
    };
    let channels = if bytes[3] >> 6 == 3 { 1 } else { 2 };
    Some((rates[rate_idx], channels))
}

/// Parses MPEG-TS PAT/PMT tables and the first PES headers within a 2 MiB scan.
// A single linear scan keeps the boundary validation and its five mutable
// accumulators in one auditable pass; splitting would only thread state around.
#[allow(clippy::too_many_lines)]
fn probe_ts(
    path: &Path,
    file: &mut File,
    file_len: u64,
) -> Result<Vec<VideoStreamInfo>, ProbeError> {
    let bytes = read_bounded_prefix(file, path, file_len, TS_SCAN_CAP)?;
    let stride =
        ts_stride(&bytes).ok_or_else(|| invalid_video(path, "no MPEG-TS packet sync cadence"))?;
    let offset = stride - 188;

    let mut pmt_pid = None;
    let mut pmt_section: Vec<u8> = Vec::new();
    let mut pmt_needed: Option<usize> = None;
    let mut streams: Vec<TsStream> = Vec::new();
    for chunk in bytes.chunks(stride) {
        if chunk.len() < stride {
            continue;
        }
        let Some(packet) = ts_packet(&chunk[offset..offset + 188]) else {
            continue;
        };
        if pmt_pid.is_none()
            && packet.pid == 0
            && packet.pusi
            && let Some(section) = ts_section_start(&packet)
        {
            pmt_pid = parse_pat(section);
        }
        if let Some(pid) = pmt_pid
            && packet.pid == pid
        {
            if packet.pusi {
                pmt_section.clear();
                let section = ts_section_start(&packet);
                pmt_needed = section.and_then(|section| {
                    ts_section_length(section).and_then(|len| 3_usize.checked_add(len))
                });
                // A TS payload is padded with 0xFF; only the declared section bytes count.
                if let (Some(section), Some(needed)) = (section, pmt_needed) {
                    let take = needed.min(section.len());
                    pmt_section.extend_from_slice(&section[..take]);
                }
            } else if let Some(needed) = pmt_needed {
                let left = needed.saturating_sub(pmt_section.len());
                let take = left.min(packet.payload.len());
                pmt_section.extend_from_slice(&packet.payload[..take]);
            }
            if let Some(needed) = pmt_needed
                && pmt_section.len() >= needed
            {
                if let Some(found) = parse_pmt(&pmt_section[..needed]) {
                    streams = found;
                    break;
                }
                pmt_needed = None;
            }
        }
    }
    pmt_pid.ok_or_else(|| invalid_video(path, "MPEG-TS has no PAT"))?;
    if streams.is_empty() {
        return Err(invalid_video(path, "MPEG-TS has no elementary streams"));
    }

    let audio_indexes: HashMap<u16, usize> = streams
        .iter()
        .enumerate()
        .filter(|(_, stream)| stream.kind == VideoStreamKind::Audio)
        .map(|(idx, stream)| (stream.pid, idx))
        .collect();
    let mut facts: Vec<(Option<u32>, Option<u16>)> = streams
        .iter()
        .map(|stream| (stream.sample_rate, stream.channels))
        .collect();
    for chunk in bytes.chunks(stride) {
        if chunk.len() < stride {
            continue;
        }
        let Some(packet) = ts_packet(&chunk[offset..offset + 188]) else {
            continue;
        };
        let Some(&idx) = audio_indexes.get(&packet.pid) else {
            continue;
        };
        let Some(payload) = pes_payload(&packet) else {
            continue;
        };
        let parsed = match streams[idx].codec {
            VideoCodec::Aac => adts_facts(payload),
            VideoCodec::Mp3 => mp3_facts(payload),
            VideoCodec::Opus => Some((48_000, streams[idx].channels.unwrap_or(2))),
            _ => None,
        };
        if let Some((rate, channels)) = parsed {
            facts[idx] = (Some(rate), Some(channels));
        }
    }

    let mut infos = Vec::with_capacity(streams.len());
    for (idx, stream) in streams.iter().enumerate() {
        let (sample_rate, channels) = facts[idx];
        infos.push(VideoStreamInfo::new(
            stream.codec,
            stream.kind,
            None,
            None,
            channels,
            sample_rate,
            1,
            90_000,
            None,
            None,
        ));
    }
    Ok(infos)
}

fn probe_containr(
    path: &Path,
    format: VideoContainerFormat,
    file: File,
    file_len: u64,
) -> Result<Vec<VideoStreamInfo>, ProbeError> {
    let mut file = file;
    let streams = match format {
        VideoContainerFormat::Mp4 | VideoContainerFormat::Mov => {
            let reader = WindowedFileReader::from_file(file, file_len);
            let demuxer = containr::mp4::Demuxer::open(reader)
                .map_err(|err| invalid_video(path, format!("container parse failed: {err}")))?;
            demuxer.tracks().iter().map(stream_info).collect::<Vec<_>>()
        }
        VideoContainerFormat::Mkv | VideoContainerFormat::Webm => {
            let reader = WindowedFileReader::from_file(file, file_len);
            let demuxer = containr::mkv::Demuxer::open(reader)
                .map_err(|err| invalid_video(path, format!("container parse failed: {err}")))?;
            if demuxer.truncated() {
                return Err(invalid_video(path, "truncated MKV input"));
            }
            demuxer.tracks().iter().map(stream_info).collect::<Vec<_>>()
        }
        VideoContainerFormat::Avi => probe_avi(path, &mut file, file_len)?,
        VideoContainerFormat::Ts => probe_ts(path, &mut file, file_len)?,
    };
    Ok(streams)
}

/// Probes a video file with magic-byte detection and extracts stream metadata.
///
/// # Errors
///
/// Returns [`ProbeError`] for I/O failures, unrecognised magic bytes, or malformed containers.
pub fn probe_video(path: &Path) -> Result<ProbedVideo, ProbeError> {
    let metadata = fs::metadata(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(ProbeError::NotAFile(path.to_path_buf()));
    }
    if metadata.len() == 0 {
        return Err(invalid_video(path, "video file is empty"));
    }

    let mut file = File::open(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let (prefix, filled) = read_magic_prefix(&mut file, path)?;
    file.seek(SeekFrom::Start(0))
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let format = detect_video_magic(&prefix[..filled]).ok_or_else(|| {
        invalid_video(
            path,
            "unrecognized container magic bytes (expected MP4, MKV, WebM, MOV, AVI, or TS)",
        )
    })?;
    let streams = probe_containr(path, format, file, metadata.len())?;
    if streams.is_empty() {
        return Err(invalid_video(path, "container has no tracks"));
    }

    Ok(ProbedVideo {
        path: path.to_path_buf(),
        format,
        streams,
    })
}

fn canonical_source(path: &Path) -> Result<PathBuf, ProbeError> {
    fs::canonicalize(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Resolves a target that may not exist yet through its canonical parent.
fn canonical_target(target: &Path) -> Result<PathBuf, ProbeError> {
    if target.exists() {
        Ok(fs::canonicalize(target).map_err(|err| ProbeError::Io {
            path: target.to_path_buf(),
            source: err,
        })?)
    } else {
        let parent = target
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let canonical_parent = fs::canonicalize(parent).map_err(|err| ProbeError::Io {
            path: parent.to_path_buf(),
            source: err,
        })?;
        Ok(match target.file_name() {
            Some(name) => canonical_parent.join(name),
            None => canonical_parent,
        })
    }
}

/// Rejects targets that name the same file as the source through a different path spelling.
fn reject_alias_collision(source: &Path, target: &Path) -> Result<(), ProbeError> {
    if canonical_source(source)? == canonical_target(target)? {
        return Err(DomainError::IdenticalSourceAndTarget(source.to_path_buf()).into());
    }
    Ok(())
}

fn canonical_sources(paths: &[PathBuf]) -> Result<HashSet<PathBuf>, ProbeError> {
    paths.iter().map(|path| canonical_source(path)).collect()
}

/// Rejects a batch target that resolves onto any source in the same batch.
fn reject_batch_target_collision(
    sources: &HashSet<PathBuf>,
    target: &Path,
) -> Result<(), ProbeError> {
    if sources.contains(&canonical_target(target)?) {
        return Err(ProbeError::InvalidVideo {
            path: target.to_path_buf(),
            reason: "target aliases another source in this batch".to_owned(),
        });
    }
    Ok(())
}

fn resolve_single_target(
    source: &Path,
    target: &Path,
    to: Option<VideoContainerFormat>,
) -> Result<(PathBuf, VideoContainerFormat), ProbeError> {
    let target_is_dir = target.is_dir()
        || target
            .to_str()
            .is_some_and(|s| s.ends_with('/') || s.ends_with('\\'));
    if target_is_dir {
        let format = to.ok_or_else(ProbeError::requires_to_for_video_directory)?;
        let stem = source
            .file_stem()
            .ok_or_else(|| invalid_video(source, "source path has no file stem"))?;
        let mut name = stem.to_os_string();
        name.push(".");
        name.push(format.extension());
        return Ok((target.join(name), format));
    }

    let ext = target.extension().and_then(|e| e.to_str()).ok_or_else(|| {
        ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: "mp4, mkv, webm, mov, avi, or ts",
        }
    })?;
    let inferred = VideoContainerFormat::from_extension(ext).ok_or_else(|| {
        ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: "mp4, mkv, webm, mov, avi, or ts",
        }
    })?;
    if let Some(explicit) = to
        && explicit != inferred
    {
        return Err(ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: explicit.extension(),
        });
    }
    Ok((target.to_path_buf(), inferred))
}

fn prepare_directory_batch(
    source_dir: &Path,
    target_dir: &Path,
    format: VideoContainerFormat,
    threads: NonZeroUsize,
) -> Result<VideoRemuxPlan, ProbeError> {
    if !target_dir.is_dir() {
        let exists = fs::metadata(target_dir).is_ok();
        if exists {
            return Err(ProbeError::TargetNotADirectory(target_dir.to_path_buf()));
        }
        fs::create_dir_all(target_dir).map_err(|source| ProbeError::Io {
            path: target_dir.to_path_buf(),
            source,
        })?;
    }

    let mut sources: Vec<PathBuf> = fs::read_dir(source_dir)
        .map_err(|source| ProbeError::Io {
            path: source_dir.to_path_buf(),
            source,
        })?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter(|path| detect_video_magic_file(path).is_some())
        .collect();
    sources.sort();

    if sources.is_empty() {
        return Err(ProbeError::EmptyVideoDirectory(source_dir.to_path_buf()));
    }
    let canonical_sources = canonical_sources(&sources)?;

    let mut items = Vec::with_capacity(sources.len());
    for source in sources {
        let probed = probe_video(&source)?;
        let stem = source
            .file_stem()
            .ok_or_else(|| invalid_video(&source, "source path has no file stem"))?;
        let mut name = stem.to_os_string();
        name.push(".");
        name.push(format.extension());
        let target = target_dir.join(name);
        // Reject an unsupported codec at the boundary before building the plan.
        for stream in &probed.streams {
            if !stream.codec().supported_in(format) {
                return Err(DomainError::UnsupportedVideoCodec {
                    codec: stream.codec(),
                    container: format,
                }
                .into());
            }
        }
        reject_alias_collision(&source, &target)?;
        reject_batch_target_collision(&canonical_sources, &target)?;
        items.push(VideoRemuxItem::new(
            source,
            target,
            probed.format,
            format,
            probed.streams,
        ));
    }

    Ok(VideoRemuxPlan::new(items, threads)?)
}

fn detect_video_magic_file(path: &Path) -> Option<VideoContainerFormat> {
    let mut file = File::open(path).ok()?;
    read_magic_prefix(&mut file, path)
        .ok()
        .and_then(|(prefix, filled)| detect_video_magic(&prefix[..filled]))
}

/// Builds a validated [`VideoRemuxPlan`] from CLI inputs.
///
/// # Errors
///
/// Returns [`ProbeError`] for invalid paths, unsupported target formats, or unsupported codecs.
pub fn prepare_video_batch(
    source: &Path,
    target: &Path,
    to: Option<VideoContainerFormat>,
    worker_threads: Option<NonZeroUsize>,
) -> Result<VideoRemuxPlan, ProbeError> {
    if source.as_os_str().is_empty() {
        return Err(DomainError::EmptyPath(PathRole::Source).into());
    }
    if target.as_os_str().is_empty() {
        return Err(DomainError::EmptyPath(PathRole::Target).into());
    }

    let threads = match worker_threads {
        Some(count) => count,
        None => available_cpu_threads()?,
    };
    let source_meta = fs::metadata(source).map_err(|err| ProbeError::Io {
        path: source.to_path_buf(),
        source: err,
    })?;

    if source_meta.is_dir() {
        let format = to.ok_or_else(ProbeError::requires_to_for_video_directory)?;
        return prepare_directory_batch(source, target, format, threads);
    }
    if !source_meta.is_file() {
        return Err(ProbeError::NotAFileOrDirectory(source.to_path_buf()));
    }

    let (resolved_target, format) = resolve_single_target(source, target, to)?;
    reject_alias_collision(source, &resolved_target)?;
    let probed = probe_video(source)?;
    for stream in &probed.streams {
        if !stream.codec().supported_in(format) {
            return Err(DomainError::UnsupportedVideoCodec {
                codec: stream.codec(),
                container: format,
            }
            .into());
        }
    }
    let item = VideoRemuxItem::new(
        source.to_path_buf(),
        resolved_target,
        probed.format,
        format,
        probed.streams,
    );
    Ok(VideoRemuxPlan::new(vec![item], threads)?)
}

/// Codec selection for the video track of a transcode batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestedVideoCodec {
    Copy,
    Encode(TargetVideoCodec),
}

/// Codec selection for audio tracks of a transcode batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestedAudioCodec {
    Copy,
    Encode(TargetAudioCodec),
}

/// Builds a validated [`VideoTranscodePlan`] from CLI inputs.
///
/// # Errors
///
/// Returns [`ProbeError`] for invalid paths, unsupported codec/container pairs, or
/// encoder/backend combinations that have no implementation.
// One argument per CLI input keeps the boundary mapping explicit and auditable;
// the directory walk likewise stays in one validation pass before planning.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn prepare_video_transcode_batch(
    source: &Path,
    target: &Path,
    to: Option<VideoContainerFormat>,
    video_codec: Option<RequestedVideoCodec>,
    audio_codec: Option<RequestedAudioCodec>,
    quality: VideoQuality,
    backend: HardwareBackend,
    worker_threads: Option<NonZeroUsize>,
) -> Result<VideoTranscodePlan, ProbeError> {
    if source.as_os_str().is_empty() {
        return Err(DomainError::EmptyPath(PathRole::Source).into());
    }
    if target.as_os_str().is_empty() {
        return Err(DomainError::EmptyPath(PathRole::Target).into());
    }
    let threads = match worker_threads {
        Some(count) => count,
        None => available_cpu_threads()?,
    };

    let source_meta = fs::metadata(source).map_err(|source_err| ProbeError::Io {
        path: source.to_path_buf(),
        source: source_err,
    })?;
    let mut specs: Vec<(PathBuf, PathBuf)> = Vec::new();
    let output_format = if source_meta.is_dir() {
        let format = to.ok_or_else(ProbeError::requires_to_for_video_directory)?;
        if !target.is_dir() {
            if fs::metadata(target).is_ok() {
                return Err(ProbeError::TargetNotADirectory(target.to_path_buf()));
            }
            fs::create_dir_all(target).map_err(|source_err| ProbeError::Io {
                path: target.to_path_buf(),
                source: source_err,
            })?;
        }
        let mut sources: Vec<PathBuf> = fs::read_dir(source)
            .map_err(|source_err| ProbeError::Io {
                path: source.to_path_buf(),
                source: source_err,
            })?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .filter(|path| detect_video_magic_file(path).is_some())
            .collect();
        sources.sort();
        if sources.is_empty() {
            return Err(ProbeError::EmptyVideoDirectory(source.to_path_buf()));
        }
        let canonical_sources = canonical_sources(&sources)?;
        for file in sources {
            let stem = file
                .file_stem()
                .ok_or_else(|| invalid_video(&file, "source path has no file stem"))?;
            let mut name = stem.to_os_string();
            name.push(".");
            name.push(format.extension());
            let output = target.join(name);
            reject_batch_target_collision(&canonical_sources, &output)?;
            specs.push((file, output));
        }
        format
    } else if source_meta.is_file() {
        let (resolved, format) = resolve_single_target(source, target, to)?;
        specs.push((source.to_path_buf(), resolved));
        format
    } else {
        return Err(ProbeError::NotAFileOrDirectory(source.to_path_buf()));
    };

    let video_action = match video_codec {
        Some(RequestedVideoCodec::Copy) => {
            if backend != HardwareBackend::Cpu {
                return Err(DomainError::UnsupportedVideoPipeline {
                    codec: VideoCodec::Unknown,
                    backend,
                    reason: "video stream copy runs on CPU; omit --backend or pass --backend cpu"
                        .to_owned(),
                }
                .into());
            }
            VideoTrackAction::Copy
        }
        Some(RequestedVideoCodec::Encode(codec)) => {
            VideoTrackAction::Encode(VideoEncodeMode::resolve(codec, quality, backend)?)
        }
        None => VideoTrackAction::Encode(VideoEncodeMode::resolve(
            output_format.default_video_codec(),
            quality,
            backend,
        )?),
    };
    let audio_action = match audio_codec {
        Some(RequestedAudioCodec::Copy) => AudioTrackAction::Copy,
        Some(RequestedAudioCodec::Encode(codec)) => AudioTrackAction::Encode(codec),
        None => AudioTrackAction::Encode(output_format.default_audio_codec()),
    };

    let mut items = Vec::with_capacity(specs.len());
    for (input, output) in specs {
        reject_alias_collision(&input, &output)?;
        let probed = probe_video(&input)?;
        items.push(VideoTranscodeItem::new(
            input,
            output,
            probed.format,
            output_format,
            probed.streams,
            video_action,
            audio_action,
        )?);
    }
    Ok(VideoTranscodePlan::new(items, threads)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

    fn temp_path(label: &str) -> PathBuf {
        let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!("fastconv-probe-video-{label}-{nanos}-{seq}"))
    }

    fn temp_file(label: &str, bytes: &[u8]) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let path = temp_path(label);
        fs::write(&path, bytes)?;
        Ok(path)
    }

    fn ebml_el(id: u32, body: &[u8]) -> Vec<u8> {
        let raw = id.to_be_bytes();
        let skip = raw.iter().take_while(|&&b| b == 0).count();
        let mut out = raw[skip..].to_vec();
        let size = u8::try_from(body.len()).unwrap_or(u8::MAX);
        out.push(0x80_u8 | size);
        out.extend_from_slice(body);
        out
    }

    fn ebml_uint(id: u32, value: u64) -> Vec<u8> {
        let bytes = value.to_be_bytes();
        let skip = bytes.iter().take_while(|&&b| b == 0).count().min(7);
        ebml_el(id, &bytes[skip..])
    }

    /// An EBML/MKV file whose declared Segment size extends past the real data.
    fn truncated_mkv() -> Vec<u8> {
        let mut body = Vec::new();
        body.extend(ebml_uint(0x4286, 1));
        body.extend(ebml_uint(0x42F2, 4));
        body.extend(ebml_uint(0x42F3, 8));
        body.extend(ebml_el(0x4282, b"matroska"));
        body.extend(ebml_uint(0x4287, 4));
        body.extend(ebml_uint(0x4285, 2));
        let mut bytes = ebml_el(0x1A45_DFA3, &body);
        // Segment id followed by a one-byte size declaring 100 bytes of data.
        bytes.extend_from_slice(&[0x18, 0x53, 0x80, 0x67, 0xE4]);
        bytes.extend(ebml_el(0x1549_A966, &ebml_uint(0x002A_D7B1, 1_000_000)));
        let mut entry = Vec::new();
        entry.extend(ebml_uint(0xD7, 1));
        entry.extend(ebml_uint(0x73C5, 1));
        entry.extend(ebml_uint(0x83, 1));
        entry.extend(ebml_el(0x86, b"V_MPEG4/ISO/AVC"));
        bytes.extend(ebml_el(0x1654_AE6B, &ebml_el(0xAE, &entry)));
        bytes
    }

    #[test]
    fn detects_magic_without_extension_help() {
        let mut mkv = vec![0_u8; 8];
        mkv[..4].copy_from_slice(&[0x1A, 0x45, 0xDF, 0xA3]);
        assert_eq!(detect_video_magic(&mkv), Some(VideoContainerFormat::Mkv));

        let mut mp4 = vec![0_u8; 8];
        mp4[4..8].copy_from_slice(b"ftyp");
        assert_eq!(detect_video_magic(&mp4), Some(VideoContainerFormat::Mp4));

        let mut mov = vec![0_u8; 8];
        mov[4..8].copy_from_slice(b"moov");
        assert_eq!(detect_video_magic(&mov), Some(VideoContainerFormat::Mov));

        let mut avi = vec![0_u8; 12];
        avi[..4].copy_from_slice(b"RIFF");
        avi[8..12].copy_from_slice(b"AVI ");
        assert_eq!(detect_video_magic(&avi), Some(VideoContainerFormat::Avi));

        let ts_188 = ts_stream(3, 188);
        assert_eq!(detect_video_magic(&ts_188), Some(VideoContainerFormat::Ts));
        let ts_192 = ts_stream(3, 192);
        assert_eq!(detect_video_magic(&ts_192), Some(VideoContainerFormat::Ts));

        assert_eq!(detect_video_magic(b"unknown"), None);
    }

    fn ts_stream(count: usize, stride: usize) -> Vec<u8> {
        let offset = stride - 188;
        let mut bytes = vec![0xFF_u8; offset + count * stride];
        for idx in 0..count {
            bytes[offset + idx * stride] = 0x47;
        }
        bytes
    }

    #[test]
    fn ebml_doc_type_splits_webm_and_mkv() {
        let webm = ebml_el(0x1A45_DFA3, &ebml_el(0x4282, b"webm"));
        assert_eq!(detect_video_magic(&webm), Some(VideoContainerFormat::Webm));

        let padded = ebml_el(0x1A45_DFA3, &ebml_el(0x4282, b"webm\0\0"));
        assert_eq!(
            detect_video_magic(&padded),
            Some(VideoContainerFormat::Webm)
        );

        let mkv = ebml_el(0x1A45_DFA3, &ebml_el(0x4282, b"matroska"));
        assert_eq!(detect_video_magic(&mkv), Some(VideoContainerFormat::Mkv));

        // No DocType inside the prefix defaults to Matroska.
        let mut tiny = vec![0_u8; 8];
        tiny[..4].copy_from_slice(&[0x1A, 0x45, 0xDF, 0xA3]);
        assert_eq!(detect_video_magic(&tiny), Some(VideoContainerFormat::Mkv));
    }

    #[test]
    fn isobmff_brands_split_mp4_mov_and_reject_images() {
        fn ftyp(brand: [u8; 4]) -> Vec<u8> {
            let mut bytes = 16_u32.to_be_bytes().to_vec();
            bytes.extend_from_slice(b"ftyp");
            bytes.extend_from_slice(&brand);
            bytes.extend_from_slice(&[0, 0, 0, 0]);
            bytes.extend_from_slice(b"isom");
            bytes
        }
        assert_eq!(
            detect_video_magic(&ftyp(*b"isom")),
            Some(VideoContainerFormat::Mp4)
        );
        assert_eq!(
            detect_video_magic(&ftyp(*b"qt  ")),
            Some(VideoContainerFormat::Mov)
        );
        for brand in [
            b"avif", b"avis", b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"jxl ",
        ] {
            assert_eq!(detect_video_magic(&ftyp(*brand)), None, "brand {brand:?}");
        }
        let mut bare = 8_u32.to_be_bytes().to_vec();
        bare.extend_from_slice(b"ftyp");
        assert_eq!(detect_video_magic(&bare), Some(VideoContainerFormat::Mp4));
        for box_type in [b"mdat", b"free", b"udta", b"rmra"] {
            let mut bytes = vec![0_u8; 8];
            bytes[4..8].copy_from_slice(box_type);
            assert_eq!(detect_video_magic(&bytes), Some(VideoContainerFormat::Mov));
        }
    }

    fn riff_chunk(id: [u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + payload.len());
        out.extend_from_slice(&id);
        out.extend_from_slice(&u32::try_from(payload.len()).unwrap_or(0).to_le_bytes());
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

    fn avi_strh(fcc_type: [u8; 4], handler: [u8; 4], rate: u32, length: u32) -> Vec<u8> {
        let mut body = vec![0_u8; 56];
        body[0..4].copy_from_slice(&fcc_type);
        body[4..8].copy_from_slice(&handler);
        body[20..24].copy_from_slice(&1_u32.to_le_bytes());
        body[24..28].copy_from_slice(&rate.to_le_bytes());
        body[32..36].copy_from_slice(&length.to_le_bytes());
        riff_chunk(*b"strh", &body)
    }

    fn avi_video_strf(width: u32, height: u32, compression: [u8; 4]) -> Vec<u8> {
        let mut body = vec![0_u8; 40];
        body[4..8].copy_from_slice(&i32::try_from(width).unwrap_or(i32::MAX).to_le_bytes());
        body[8..12].copy_from_slice(&i32::try_from(height).unwrap_or(i32::MAX).to_le_bytes());
        body[16..20].copy_from_slice(&compression);
        riff_chunk(*b"strf", &body)
    }

    fn avi_audio_strf(format_tag: u16, channels: u16, rate: u32) -> Vec<u8> {
        let mut body = vec![0_u8; 18];
        body[0..2].copy_from_slice(&format_tag.to_le_bytes());
        body[2..4].copy_from_slice(&channels.to_le_bytes());
        body[4..8].copy_from_slice(&rate.to_le_bytes());
        riff_chunk(*b"strf", &body)
    }

    fn minimal_avi() -> Vec<u8> {
        let mut avih = vec![0_u8; 56];
        avih[0..4].copy_from_slice(&33_333_u32.to_le_bytes());
        avih[16..20].copy_from_slice(&100_u32.to_le_bytes());
        avih[32..36].copy_from_slice(&1280_u32.to_le_bytes());
        avih[36..40].copy_from_slice(&720_u32.to_le_bytes());
        let hdrl = vec![
            riff_chunk(*b"avih", &avih),
            riff_list(
                *b"strl",
                &[
                    avi_strh(*b"vids", *b"H264", 30, 100),
                    avi_video_strf(640, 360, *b"H264"),
                ],
            ),
            riff_list(
                *b"strl",
                &[
                    avi_strh(*b"auds", *b"\0\0\0\0", 44_100, 200),
                    avi_audio_strf(0x0055, 2, 44_100),
                ],
            ),
            riff_list(
                *b"strl",
                &[
                    avi_strh(*b"auds", *b"\0\0\0\0", 48_000, 200),
                    avi_audio_strf(0x706D, 2, 48_000),
                ],
            ),
            riff_list(
                *b"strl",
                &[
                    avi_strh(*b"auds", *b"OPUS", 48_000, 200),
                    avi_audio_strf(0, 6, 48_000),
                ],
            ),
        ];
        let mut body = b"AVI ".to_vec();
        body.extend_from_slice(&riff_list(*b"hdrl", &hdrl));
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&u32::try_from(body.len()).unwrap_or(0).to_le_bytes());
        bytes.extend_from_slice(&body);
        bytes
    }

    #[test]
    fn synthetic_avi_probe_extracts_streams() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_file("synthetic-avi", &minimal_avi())?;
        let probed = probe_video(&path)?;
        assert_eq!(probed.format, VideoContainerFormat::Avi);
        assert_eq!(probed.streams.len(), 4);

        let video = &probed.streams[0];
        assert_eq!(video.codec(), VideoCodec::H264);
        assert_eq!(video.kind(), VideoStreamKind::Video);
        assert_eq!(video.width().map(std::num::NonZero::get), Some(640));
        assert_eq!(video.height().map(std::num::NonZero::get), Some(360));

        let mp3 = &probed.streams[1];
        assert_eq!(mp3.codec(), VideoCodec::Mp3);
        assert_eq!(mp3.sample_rate().map(std::num::NonZero::get), Some(44_100));
        assert_eq!(mp3.channels().map(std::num::NonZero::get), Some(2));

        let aac = &probed.streams[2];
        assert_eq!(aac.codec(), VideoCodec::Aac);
        assert_eq!(aac.sample_rate().map(std::num::NonZero::get), Some(48_000));

        let opus = &probed.streams[3];
        assert_eq!(opus.codec(), VideoCodec::Opus);
        assert_eq!(opus.channels().map(std::num::NonZero::get), Some(6));
        Ok(())
    }

    #[test]
    fn truncated_avi_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_file("truncated-avi", b"RIFF\x04\0\0\0AVI ")?;
        assert!(matches!(
            probe_video(&path),
            Err(ProbeError::InvalidVideo { .. })
        ));
        Ok(())
    }

    fn mpeg4_desc(tag: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![tag, u8::try_from(payload.len()).unwrap_or(u8::MAX)];
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn mpeg4_descriptor_accepts_expandable_big_endian_lengths() {
        assert_eq!(
            mpeg4_descriptor(&mpeg4_desc(0x05, b"abc"), 0x05),
            Some(b"abc".as_slice())
        );
        let mut padded = vec![0x05, 0x80, 0x80, 0x80, 3];
        padded.extend_from_slice(b"abc");
        assert_eq!(mpeg4_descriptor(&padded, 0x05), Some(b"abc".as_slice()));
        let mut too_long = vec![0x05, 0x80, 0x80, 0x80, 0x80, 3];
        too_long.extend_from_slice(b"abc");
        assert_eq!(mpeg4_descriptor(&too_long, 0x05), None);
    }

    #[test]
    fn opus_registration_honours_extension_channel_count() {
        let mut descriptors = ts_desc(0x05, b"Opus");
        descriptors.extend_from_slice(&ts_desc(0x7F, &[0x80, 6]));
        let info = map_ts_stream(0x06, &descriptors);
        assert_eq!(info.codec, VideoCodec::Opus);
        assert_eq!(info.channels, Some(6));
        let fallback = map_ts_stream(0x06, &ts_desc(0x05, b"Opus"));
        assert_eq!(fallback.channels, Some(2));
    }

    #[test]
    fn adts_channel_config_follows_mpeg4_table() {
        let header = |config: u8| {
            [
                0xFF,
                0xF1,
                0x40 | (4 << 2) | (config >> 2),
                (config & 0x03) << 6,
                0x00,
            ]
        };
        assert_eq!(adts_facts(&header(2)), Some((44_100, 2)));
        assert_eq!(adts_facts(&header(7)), Some((44_100, 8)));
        assert_eq!(adts_facts(&header(0)), None);
    }

    #[test]
    fn synthetic_ts_probe_assembles_multi_packet_pmt() -> Result<(), Box<dyn std::error::Error>> {
        let mut association_body = vec![0x00, 0x01, 0xC1, 0x00, 0x00];
        association_body.extend_from_slice(&[0x00, 0x01, 0xF0, 0x00]);
        let mut program_map_body = vec![0x00, 0x01, 0xC1, 0x00, 0x00, 0xF0, 0x00, 0xF0, 0x00];
        // Enough elementary streams to force the PMT section past one TS packet.
        for index in 0..40_u16 {
            program_map_body.extend_from_slice(&ts_es(0x1B, 0x0200 + index, &[]));
        }
        program_map_body.extend_from_slice(&ts_es(0x06, 0x0112, &ts_desc(0x05, b"Opus")));
        let mut pat = vec![0x00];
        pat.extend_from_slice(&ts_section(0x00, &association_body));
        let pmt = ts_section(0x02, &program_map_body);
        let split = 183;
        let mut first = vec![0x00];
        first.extend_from_slice(&pmt[..split]);
        let mut bytes = ts_packet(0, true, &pat);
        bytes.extend_from_slice(&ts_packet(0x1000, true, &first));
        bytes.extend_from_slice(&ts_packet(0x1000, false, &pmt[split..]));
        let path = temp_file("split-pmt-ts", &bytes)?;
        let probed = probe_video(&path)?;
        assert_eq!(probed.streams.len(), 41);
        assert_eq!(probed.streams[0].codec(), VideoCodec::H264);
        assert_eq!(probed.streams[40].codec(), VideoCodec::Opus);
        assert_eq!(
            probed.streams[40].channels().map(std::num::NonZero::get),
            Some(2)
        );
        Ok(())
    }

    #[test]
    fn esds_object_type_distinguishes_mp3_from_aac() {
        let build = |object_type: u8| {
            let decoder_config = mpeg4_desc(0x04, &[object_type, 0x15, 0, 0, 0, 0, 0, 0, 0, 0]);
            let mut es_body = vec![0x00, 0x01, 0x00];
            es_body.extend_from_slice(&decoder_config);
            let mut esds = vec![0, 0, 0, 0];
            esds.extend_from_slice(&mpeg4_desc(0x03, &es_body));
            esds
        };
        assert_eq!(esds_object_type(&build(0x6B)), Some(0x6B));
        assert_eq!(esds_object_type(&build(0x69)), Some(0x69));
        assert_eq!(esds_object_type(&build(0x40)), Some(0x40));
        assert_eq!(esds_object_type(&[0, 0, 0, 0]), None);
    }

    fn ts_section(table_id: u8, body: &[u8]) -> Vec<u8> {
        let length = body.len() + 4; // CRC32 placeholder
        let mut out = vec![
            table_id,
            0x80 | u8::try_from((length >> 8) & 0x0F).unwrap_or(0),
            u8::try_from(length & 0xFF).unwrap_or(0),
        ];
        out.extend_from_slice(body);
        out.extend_from_slice(&[0, 0, 0, 0]);
        out
    }

    fn ts_packet(pid: u16, pusi: bool, payload: &[u8]) -> Vec<u8> {
        let high = u8::try_from(pid >> 8).unwrap_or(0) & 0x1F;
        let low = u8::try_from(pid & 0x00FF).unwrap_or(0);
        let mut packet = vec![0x47, high | if pusi { 0x40 } else { 0 }, low, 0x10];
        packet.extend_from_slice(payload);
        packet.resize(188, 0xFF);
        packet
    }

    fn ts_desc(tag: u8, data: &[u8]) -> Vec<u8> {
        let mut out = vec![tag, u8::try_from(data.len()).unwrap_or(u8::MAX)];
        out.extend_from_slice(data);
        out
    }

    fn ts_es(stream_type: u8, pid: u16, descriptors: &[u8]) -> Vec<u8> {
        let len = u16::try_from(descriptors.len()).unwrap_or(u16::MAX);
        let mut out = vec![
            stream_type,
            0xE0 | (u8::try_from(pid >> 8).unwrap_or(0) & 0x1F),
            u8::try_from(pid & 0x00FF).unwrap_or(0),
            0xF0 | u8::try_from((len >> 8) & 0x0F).unwrap_or(0),
            u8::try_from(len & 0xFF).unwrap_or(0),
        ];
        out.extend_from_slice(descriptors);
        out
    }

    fn ts_pes(pid: u16, stream_id: u8, header_data: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut pes = vec![0x00, 0x00, 0x01, stream_id];
        let length = 3_usize + header_data.len() + payload.len();
        pes.extend_from_slice(&u16::try_from(length).unwrap_or(u16::MAX).to_be_bytes());
        pes.extend_from_slice(&[
            0x80,
            0x00,
            u8::try_from(header_data.len()).unwrap_or(u8::MAX),
        ]);
        pes.extend_from_slice(header_data);
        pes.extend_from_slice(payload);
        ts_packet(pid, true, &pes)
    }

    fn minimal_ts() -> Vec<u8> {
        let mut association_body = vec![0x00, 0x01, 0xC1, 0x00, 0x00];
        association_body.extend_from_slice(&[0x00, 0x01, 0xF0, 0x00]); // program 1 -> PMT PID 0x1000
        let mut program_map_body = vec![0x00, 0x01, 0xC1, 0x00, 0x00, 0xF0, 0x00, 0xF0, 0x00];
        program_map_body.extend_from_slice(&ts_es(0x1B, 0x0100, &[]));
        program_map_body.extend_from_slice(&ts_es(0x24, 0x0101, &[]));
        program_map_body.extend_from_slice(&ts_es(0x06, 0x0102, &ts_desc(0x05, b"AV01")));
        program_map_body.extend_from_slice(&ts_es(0x0F, 0x0110, &[]));
        program_map_body.extend_from_slice(&ts_es(0x03, 0x0111, &[]));
        program_map_body.extend_from_slice(&ts_es(0x06, 0x0112, &ts_desc(0x7F, &[0x80, 0x06])));

        let mut pat = vec![0x00];
        pat.extend_from_slice(&ts_section(0x00, &association_body));
        let mut pmt = vec![0x00];
        pmt.extend_from_slice(&ts_section(0x02, &program_map_body));
        let mut bytes = ts_packet(0, true, &pat);
        bytes.extend_from_slice(&ts_packet(0x1000, true, &pmt));
        bytes.extend_from_slice(&ts_pes(0x0110, 0xC0, &[], &[0xFF, 0xF1, 0x10, 0x80, 0x00]));
        bytes.extend_from_slice(&ts_pes(0x0111, 0xC0, &[], &[0xFF, 0xFB, 0x90, 0x00, 0x00]));
        bytes.extend_from_slice(&ts_pes(0x0112, 0xC0, &[], &[]));
        bytes
    }

    #[test]
    fn synthetic_ts_probe_extracts_tables_and_audio_facts() -> Result<(), Box<dyn std::error::Error>>
    {
        let path = temp_file("synthetic-ts", &minimal_ts())?;
        let probed = probe_video(&path)?;
        assert_eq!(probed.format, VideoContainerFormat::Ts);
        assert_eq!(probed.streams.len(), 6);
        let expected = [
            VideoCodec::H264,
            VideoCodec::Hevc,
            VideoCodec::Av1,
            VideoCodec::Aac,
            VideoCodec::Mp3,
            VideoCodec::Opus,
        ];
        for (stream, codec) in probed.streams.iter().zip(expected) {
            assert_eq!(stream.codec(), codec);
        }
        assert_eq!(
            probed.streams[3].sample_rate().map(std::num::NonZero::get),
            Some(44_100)
        );
        assert_eq!(
            probed.streams[3].channels().map(std::num::NonZero::get),
            Some(2)
        );
        assert_eq!(
            probed.streams[4].sample_rate().map(std::num::NonZero::get),
            Some(44_100)
        );
        assert_eq!(
            probed.streams[5].sample_rate().map(std::num::NonZero::get),
            Some(48_000)
        );
        assert_eq!(
            probed.streams[5].channels().map(std::num::NonZero::get),
            Some(6)
        );
        Ok(())
    }

    #[test]
    fn ts_without_pat_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_file("ts-no-pat", &ts_stream(4, 188))?;
        assert!(matches!(
            probe_video(&path),
            Err(ProbeError::InvalidVideo { .. })
        ));
        Ok(())
    }

    #[test]
    fn windowed_reader_end_seek_and_current_seek() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_file("end-seek", b"0123456789")?;
        let mut reader = WindowedFileReader::open(&path)?;
        assert_eq!(reader.seek(SeekFrom::End(0))?, 10);
        assert_eq!(reader.seek(SeekFrom::End(-4))?, 6);
        let mut out = [0_u8; 4];
        assert_eq!(reader.read(&mut out)?, 4);
        assert_eq!(&out, b"6789");
        assert_eq!(reader.seek(SeekFrom::Current(-2))?, 8);
        assert_eq!(reader.read(&mut out)?, 2);
        assert_eq!(&out[..2], b"89");
        Ok(())
    }

    #[test]
    fn windowed_reader_rejects_seek_before_start() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_file("negative-seek", b"0123456789")?;
        let mut reader = WindowedFileReader::open(&path)?;
        for pos in [SeekFrom::Current(-1), SeekFrom::End(-11)] {
            match reader.seek(pos) {
                Ok(result) => return Err(format!("unexpected seek result: {result}").into()),
                Err(err) => assert_eq!(err.kind(), io::ErrorKind::InvalidInput),
            }
        }
        Ok(())
    }

    #[test]
    fn windowed_reader_past_eof_read_returns_zero() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_file("past-eof", b"0123456789")?;
        let mut reader = WindowedFileReader::open(&path)?;
        reader.seek(SeekFrom::Start(100))?;
        assert_eq!(reader.read(&mut [0_u8; 4])?, 0);
        Ok(())
    }

    #[test]
    fn windowed_reader_backward_seek_hits_buffer() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_file("backward", b"0123456789")?;
        let mut reader = WindowedFileReader::open(&path)?;
        let mut out = [0_u8; 4];
        assert_eq!(reader.read(&mut out)?, 4);
        assert_eq!(&out, b"0123");
        assert!(reader.buf_valid >= 4);
        assert_eq!(reader.buf_start, 0);
        reader.seek(SeekFrom::Start(2))?;
        assert_eq!(reader.read(&mut out)?, 4);
        assert_eq!(&out, b"2345");
        Ok(())
    }

    #[test]
    fn windowed_reader_crosses_window_boundary() -> Result<(), Box<dyn std::error::Error>> {
        let len = WINDOW_BYTES + 4096;
        let data: Vec<u8> = (0..len)
            .map(|i| u8::try_from(i % 251).unwrap_or(0))
            .collect();
        let path = temp_file("cross-window", &data)?;
        let mut reader = WindowedFileReader::open(&path)?;
        let mut out = vec![0_u8; 4096];
        let mut total = 0_usize;
        loop {
            let count = reader.read(&mut out)?;
            if count == 0 {
                break;
            }
            assert_eq!(&out[..count], &data[total..total + count]);
            total += count;
        }
        assert_eq!(total, len);
        Ok(())
    }

    #[test]
    fn windowed_reader_large_read_exceeds_window() -> Result<(), Box<dyn std::error::Error>> {
        let len = WINDOW_BYTES + 4096;
        let data: Vec<u8> = (0..len)
            .map(|i| u8::try_from(i % 251).unwrap_or(0))
            .collect();
        let path = temp_file("large-read", &data)?;
        let mut reader = WindowedFileReader::open(&path)?;
        let mut out = vec![0_u8; len];
        let mut total = 0_usize;
        loop {
            let count = reader.read(&mut out[total..])?;
            if count == 0 {
                break;
            }
            total += count;
        }
        assert_eq!(total, len);
        assert_eq!(out, data);
        Ok(())
    }

    #[test]
    fn truncated_mkv_is_still_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let bytes = truncated_mkv();
        let path = temp_file("truncated-mkv", &bytes)?;
        let reason = match probe_video(&path) {
            Ok(_) => return Err("a truncated MKV must be rejected".into()),
            Err(ProbeError::InvalidVideo { reason, .. }) => reason,
            Err(err) => return Err(format!("unexpected probe error: {err}").into()),
        };
        assert!(reason.contains("truncated MKV input"), "reason: {reason}");
        Ok(())
    }

    fn plan_threads() -> NonZeroUsize {
        NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN)
    }

    #[test]
    fn transcode_batch_defaults_depend_on_output_container()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = temp_file("transcode-defaults", &minimal_avi())?;
        let quality = VideoQuality::DEFAULT;

        let webm_target = temp_path("transcode-defaults-webm").with_extension("webm");
        let webm = prepare_video_transcode_batch(
            &source,
            &webm_target,
            Some(VideoContainerFormat::Webm),
            None,
            None,
            quality,
            HardwareBackend::Cpu,
            Some(plan_threads()),
        )?;
        assert_eq!(webm.output_format(), VideoContainerFormat::Webm);
        assert_eq!(
            webm.items()[0].video_action(),
            VideoTrackAction::Encode(VideoEncodeMode::CpuVp9 { quality })
        );
        assert_eq!(
            webm.items()[0].audio_action(),
            AudioTrackAction::Encode(TargetAudioCodec::Opus)
        );

        let mp4_target = temp_path("transcode-defaults-mp4").with_extension("mp4");
        let mp4 = prepare_video_transcode_batch(
            &source,
            &mp4_target,
            Some(VideoContainerFormat::Mp4),
            None,
            None,
            quality,
            HardwareBackend::Cpu,
            Some(plan_threads()),
        )?;
        assert_eq!(mp4.output_format(), VideoContainerFormat::Mp4);
        assert_eq!(
            mp4.items()[0].video_action(),
            VideoTrackAction::Encode(VideoEncodeMode::CpuH264 { quality })
        );
        assert_eq!(
            mp4.items()[0].audio_action(),
            AudioTrackAction::Encode(TargetAudioCodec::Aac)
        );
        Ok(())
    }

    #[test]
    fn transcode_batch_rejects_cpu_hevc_and_nvenc_copy() -> Result<(), Box<dyn std::error::Error>> {
        let source = temp_file("transcode-backend", &minimal_avi())?;
        let target = temp_path("transcode-backend").with_extension("mp4");
        let quality = VideoQuality::DEFAULT;

        let result = prepare_video_transcode_batch(
            &source,
            &target,
            Some(VideoContainerFormat::Mp4),
            Some(RequestedVideoCodec::Encode(TargetVideoCodec::Hevc)),
            None,
            quality,
            HardwareBackend::Cpu,
            Some(plan_threads()),
        );
        match result {
            Err(ProbeError::InvalidDomain(DomainError::UnsupportedVideoPipeline {
                codec,
                backend,
                reason,
            })) => {
                assert_eq!(codec, VideoCodec::Hevc);
                assert_eq!(backend, HardwareBackend::Cpu);
                assert!(reason.contains("--backend nvenc"));
            }
            other => return Err(format!("unexpected result: {other:?}").into()),
        }

        let nvenc = HardwareBackend::Gpu(fastconv_domain::GpuBackend::Nvenc);
        let result = prepare_video_transcode_batch(
            &source,
            &target,
            Some(VideoContainerFormat::Mp4),
            Some(RequestedVideoCodec::Copy),
            None,
            quality,
            nvenc,
            Some(plan_threads()),
        );
        match result {
            Err(ProbeError::InvalidDomain(DomainError::UnsupportedVideoPipeline {
                reason,
                ..
            })) => assert!(reason.contains("copy runs on CPU")),
            other => return Err(format!("unexpected result: {other:?}").into()),
        }
        Ok(())
    }

    #[test]
    fn transcode_batch_rejects_container_codec_mismatches() -> Result<(), Box<dyn std::error::Error>>
    {
        let source = temp_file("transcode-mismatch", &minimal_avi())?;
        let quality = VideoQuality::DEFAULT;

        let webm_target = temp_path("transcode-mismatch-webm").with_extension("webm");
        let result = prepare_video_transcode_batch(
            &source,
            &webm_target,
            Some(VideoContainerFormat::Webm),
            Some(RequestedVideoCodec::Copy),
            Some(RequestedAudioCodec::Copy),
            quality,
            HardwareBackend::Cpu,
            Some(plan_threads()),
        );
        match result {
            Err(ProbeError::InvalidDomain(DomainError::UnsupportedVideoCodec {
                codec,
                container,
            })) => {
                assert_eq!(codec, VideoCodec::H264);
                assert_eq!(container, VideoContainerFormat::Webm);
            }
            other => return Err(format!("unexpected result: {other:?}").into()),
        }

        let ts_target = temp_path("transcode-mismatch-ts").with_extension("ts");
        let result = prepare_video_transcode_batch(
            &source,
            &ts_target,
            Some(VideoContainerFormat::Ts),
            Some(RequestedVideoCodec::Encode(TargetVideoCodec::Vp9)),
            None,
            quality,
            HardwareBackend::Cpu,
            Some(plan_threads()),
        );
        match result {
            Err(ProbeError::InvalidDomain(DomainError::UnsupportedVideoCodec {
                codec,
                container,
            })) => {
                assert_eq!(codec, VideoCodec::Vp9);
                assert_eq!(container, VideoContainerFormat::Ts);
            }
            other => return Err(format!("unexpected result: {other:?}").into()),
        }
        Ok(())
    }

    #[test]
    fn transcode_batch_directory_uses_magic_instead_of_extensions()
    -> Result<(), Box<dyn std::error::Error>> {
        let source_dir = temp_path("transcode-dir-source");
        fs::create_dir_all(&source_dir)?;
        fs::write(source_dir.join("a.avi"), minimal_avi())?;
        fs::write(source_dir.join("b.ts"), minimal_ts())?;
        fs::write(source_dir.join("ignored.txt"), b"not a video")?;
        let target_dir = temp_path("transcode-dir-target");

        let plan = prepare_video_transcode_batch(
            &source_dir,
            &target_dir,
            Some(VideoContainerFormat::Webm),
            None,
            None,
            VideoQuality::DEFAULT,
            HardwareBackend::Cpu,
            Some(plan_threads()),
        )?;
        assert_eq!(plan.items().len(), 2);
        assert_eq!(plan.items()[0].target(), &target_dir.join("a.webm"));
        assert_eq!(plan.items()[1].target(), &target_dir.join("b.webm"));
        assert!(target_dir.is_dir());
        Ok(())
    }
}
