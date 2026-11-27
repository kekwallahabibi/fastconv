use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::num::{NonZeroU16, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};

use containr::CodecConfig;
use containr::Track;
use containr::TrackKind;
use fastconv_domain::{
    AnimationLoop, AnimationMeta, AudioEncodeConfig, AudioEncodeOptions, AudioTrackAction,
    DomainError, HardwareBackend, HardwareDecodeBackend, PathRole, TargetAudioCodec,
    TargetVideoCodec, VideoCodec, VideoContainerFormat, VideoEncodeMode, VideoQuality,
    VideoRateControl, VideoRemuxItem, VideoRemuxPlan, VideoStreamInfo, VideoStreamKind,
    VideoTrackAction, VideoTranscodeItem, VideoTranscodePlan, resolve_audio_encode_options,
    resolve_video_rate_control, validate_audio_options_for_container,
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

/// Per-frame memory ceiling for GIF metadata probing (512 MiB).
const GIF_FRAME_MEMORY_LIMIT: u64 = 512 * 1024 * 1024;

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

    if bytes.len() >= 6 && (&bytes[..6] == b"GIF87a" || &bytes[..6] == b"GIF89a") {
        return Some(VideoContainerFormat::Gif);
    }

    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"AVI " {
        return Some(VideoContainerFormat::Avi);
    }

    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some(VideoContainerFormat::Webp);
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
        if codec_id == "A_FLAC" {
            return VideoCodec::Flac;
        }
        if matches!(
            codec_id,
            "A_PCM/INT/LIT" | "A_PCM/INT/BIG" | "A_PCM/FLOAT/IEEE"
        ) {
            return VideoCodec::Pcm;
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
        Some([b'f', b'L', b'a', b'C']) => VideoCodec::Flac,
        Some(
            [b's', b'o', b'w', b't']
            | [b't', b'w', b'o', b's']
            | [b'l', b'p', b'c', b'm']
            | [b'f', b'l', b'3', b'2'],
        ) => VideoCodec::Pcm,
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

/// Narrows a validated Matroska `f64` sample rate to `u32`.
// The range check makes the narrowing lossless for canonical integral rates.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn matroska_rate(rate: f64) -> Option<u32> {
    if rate.is_finite() && rate > 0.0 && rate <= f64::from(u32::MAX) {
        Some(rate as u32)
    } else {
        None
    }
}

fn stream_info(track: &Track, ident_facts: Option<(u16, u32)>) -> VideoStreamInfo {
    let (width, height) = dimensions(track);
    let (mut channels, mut sample_rate) = audio_facts(track);
    // containr collapses missing Matroska Audio children to spec defaults
    // (1 channel, 8000 Hz) without exposing child presence. The validated
    // Vorbis ident header is authoritative for the encoded stream, so those
    // defaults must not mask it; explicit non-default values still win.
    if let Some((ident_channels, ident_rate)) = ident_facts {
        if channels.is_none_or(|value| value == 1) {
            channels = Some(ident_channels);
        }
        if sample_rate.is_none_or(|value| value == 8000) {
            sample_rate = Some(ident_rate);
        }
    }
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

/// Reads one Matroska Xiph-laced length. A lacing sequence that ends on `0xFF`
/// is truncated and rejected instead of being silently accepted.
fn read_xiph_length(data: &[u8], idx: &mut usize) -> Result<usize, String> {
    let mut len = 0_usize;
    loop {
        let byte = *data
            .get(*idx)
            .ok_or("truncated Xiph lacing in Vorbis CodecPrivate")?;
        *idx = idx.checked_add(1).ok_or("Xiph lacing overflow")?;
        if byte == 0xFF {
            len = len.checked_add(255).ok_or("Xiph length overflow")?;
            continue;
        }
        return len
            .checked_add(usize::from(byte))
            .ok_or_else(|| "Xiph length overflow".to_owned());
    }
}

/// Validates a Matroska `CodecPrivate` blob carrying three Xiph-laced Vorbis
/// headers (RFC 3533 / Vorbis I). Returns `(channels, sample_rate)` from the
/// validated ident packet.
fn validate_vorbis_codec_private(data: &[u8]) -> Result<(u16, u32), String> {
    const MAX_LEN: usize = 1_048_576;
    if data.len() > MAX_LEN {
        return Err(format!("Vorbis CodecPrivate exceeds {MAX_LEN} bytes"));
    }
    if data.first() != Some(&0x02) {
        return Err("Vorbis CodecPrivate must contain three Xiph-laced packets".to_owned());
    }
    let mut idx = 1_usize;
    let ident_len = read_xiph_length(data, &mut idx)?;
    let comment_len = read_xiph_length(data, &mut idx)?;
    let ident_end = idx
        .checked_add(ident_len)
        .ok_or("Vorbis ident length overflow")?;
    let comment_end = ident_end
        .checked_add(comment_len)
        .ok_or("Vorbis comment length overflow")?;
    if comment_end >= data.len() {
        return Err("Vorbis setup packet is empty".to_owned());
    }
    let ident = data
        .get(idx..ident_end)
        .ok_or("truncated Vorbis identification header")?;
    let comment = data
        .get(ident_end..comment_end)
        .ok_or("truncated Vorbis comment header")?;
    let setup = data
        .get(comment_end..)
        .ok_or("truncated Vorbis setup header")?;
    if ident.len() < 30 || ident[0] != 0x01 || &ident[1..7] != b"vorbis" {
        return Err("invalid Vorbis identification header".to_owned());
    }
    if comment.len() < 7 || comment[0] != 0x03 || &comment[1..7] != b"vorbis" {
        return Err("invalid Vorbis comment header".to_owned());
    }
    if setup.len() < 7 || setup[0] != 0x05 || &setup[1..7] != b"vorbis" {
        return Err("invalid Vorbis setup header".to_owned());
    }
    let version = le_u32(ident.get(7..11).ok_or("truncated Vorbis version")?)
        .ok_or("truncated Vorbis version")?;
    if version != 0 {
        return Err(format!("unsupported Vorbis version {version}"));
    }
    let channels = u16::from(ident[11]);
    if channels == 0 {
        return Err("Vorbis identification header declares zero channels".to_owned());
    }
    let rate =
        le_u32(ident.get(12..16).ok_or("truncated Vorbis rate")?).ok_or("truncated Vorbis rate")?;
    if rate == 0 {
        return Err("Vorbis identification header declares zero sample rate".to_owned());
    }
    let blocksize = ident[28];
    let bs0 = blocksize & 0x0F;
    let bs1 = blocksize >> 4;
    if !(6..=13).contains(&bs0) || !(6..=13).contains(&bs1) || bs0 > bs1 {
        return Err("invalid Vorbis block sizes in identification header".to_owned());
    }
    if ident[29] & 1 == 0 {
        return Err("Vorbis identification header has no framing bit".to_owned());
    }
    Ok((channels, rate))
}

/// Upper bound on a FLAC configuration blob held in memory while probing.
const FLAC_CONFIG_CAP: usize = 1_048_576;

/// Reads the authoritative channel/rate facts from a 34-byte FLAC STREAMINFO.
fn flac_streaminfo_facts(streaminfo: &[u8]) -> Result<(u16, u32), String> {
    let si: [u8; 34] = streaminfo
        .try_into()
        .map_err(|_| "FLAC STREAMINFO must be exactly 34 bytes".to_owned())?;
    let min_block = u16::from_be_bytes([si[0], si[1]]);
    let max_block = u16::from_be_bytes([si[2], si[3]]);
    if min_block < 16 {
        return Err("FLAC STREAMINFO declares min_block_size below 16".to_owned());
    }
    if max_block < min_block {
        return Err("FLAC STREAMINFO max_block_size is below min_block_size".to_owned());
    }
    let packed = u64::from_be_bytes([
        si[10], si[11], si[12], si[13], si[14], si[15], si[16], si[17],
    ]);
    let rate = u32::try_from((packed >> 44) & 0xF_FFFF)
        .map_err(|_| "FLAC sample rate exceeds 32 bits".to_owned())?;
    let channels = u16::try_from(((packed >> 41) & 0x7).saturating_add(1))
        .map_err(|_| "FLAC channel count overflow".to_owned())?;
    let bits = ((packed >> 36) & 0x1F).saturating_add(1);
    if !(1..=655_350).contains(&rate) {
        return Err(format!("FLAC STREAMINFO rate {rate} is out of range"));
    }
    if !(1..=8).contains(&channels) {
        return Err(format!(
            "FLAC STREAMINFO channel count {channels} is out of range"
        ));
    }
    if !(4..=32).contains(&bits) {
        return Err(format!("FLAC STREAMINFO bit depth {bits} is out of range"));
    }
    Ok((channels, rate))
}

/// Walks a bounded FLAC metadata block chain and returns its STREAMINFO body.
/// `data` starts at the first metadata block header (after `fLaC` or `dfLa`'s
/// version/flags).
fn flac_streaminfo_from_metadata(data: &[u8]) -> Result<[u8; 34], String> {
    if data.len() > FLAC_CONFIG_CAP {
        return Err(format!("FLAC metadata exceeds {FLAC_CONFIG_CAP} bytes"));
    }
    let mut idx = 0_usize;
    let mut streaminfo = None;
    loop {
        let header = *data
            .get(idx)
            .ok_or("truncated FLAC metadata block header")?;
        idx = idx.checked_add(1).ok_or("FLAC metadata offset overflow")?;
        let last = header & 0x80 != 0;
        let kind = header & 0x7F;
        let len_bytes = data
            .get(idx..idx.saturating_add(3))
            .ok_or("truncated FLAC metadata block length")?;
        let len = usize::try_from(u32::from_be_bytes([
            0,
            len_bytes[0],
            len_bytes[1],
            len_bytes[2],
        ]))
        .map_err(|_| "FLAC metadata block length overflow")?;
        idx = idx.checked_add(3).ok_or("FLAC metadata offset overflow")?;
        let end = idx.checked_add(len).ok_or("FLAC metadata block overflow")?;
        let body = data.get(idx..end).ok_or("truncated FLAC metadata block")?;
        idx = end;
        if streaminfo.is_none() {
            if kind != 0 || len != 34 {
                return Err("first FLAC metadata block must be a 34-byte STREAMINFO".to_owned());
            }
            let si: [u8; 34] = body
                .try_into()
                .map_err(|_| "invalid FLAC STREAMINFO length".to_owned())?;
            flac_streaminfo_facts(&si)?;
            streaminfo = Some(si);
        }
        if last {
            break;
        }
    }
    streaminfo.ok_or_else(|| "FLAC metadata has no STREAMINFO".to_owned())
}

/// Validates a Matroska `CodecPrivate` carrying FLAC headers.
fn validate_flac_codec_private(data: &[u8]) -> Result<(u16, u32), String> {
    if !data.starts_with(b"fLaC") {
        return Err("A_FLAC CodecPrivate must start with 'fLaC'".to_owned());
    }
    flac_streaminfo_facts(&flac_streaminfo_from_metadata(&data[4..])?)
}

/// Finds a child ISOBMFF box body inside a raw sample entry.
fn child_box(raw: &[u8], start: usize, fourcc: [u8; 4]) -> Option<&[u8]> {
    let mut idx = start;
    while idx.checked_add(8)? <= raw.len() {
        let size = u32::from_be_bytes(raw.get(idx..idx.checked_add(4)?)?.try_into().ok()?);
        let kind = raw.get(idx + 4..idx + 8)?;
        let header = if size == 1 { 16 } else { 8 };
        let body_start = idx.checked_add(header)?;
        if body_start > raw.len() {
            return None;
        }
        let body_len = if size == 0 {
            raw.len().saturating_sub(body_start)
        } else {
            usize::try_from(size).ok()?.checked_sub(header)?
        };
        let body_end = body_start.checked_add(body_len)?;
        if body_end > raw.len() {
            return None;
        }
        if kind == fourcc {
            return Some(&raw[body_start..body_end]);
        }
        if size == 0 {
            return None;
        }
        idx = body_end;
    }
    None
}

/// Validates an MP4/MOV `dfLa` box body (full box + FLAC metadata chain).
fn validate_dfla(data: &[u8]) -> Result<(u16, u32), String> {
    if data.len() < 4 || data[..4] != [0, 0, 0, 0] {
        return Err("dfLa must be version 0 with zero flags".to_owned());
    }
    flac_streaminfo_facts(&flac_streaminfo_from_metadata(&data[4..])?)
}

/// Finds an audio configuration box in an MP4/MOV sample entry, including
/// inside a `QuickTime` `wave` wrapper.
fn mp4_audio_config_box(entry: &containr::SampleEntry, fourcc: [u8; 4]) -> Option<&[u8]> {
    let raw = &entry.raw;
    let version = be_u16(raw.get(16..18)?)?;
    let children_at = 8 + match version {
        0 => 28,
        1 => 44,
        2 => 64,
        _ => return None,
    };
    if let Some(body) = child_box(raw, children_at, fourcc) {
        return Some(body);
    }
    child_box(child_box(raw, children_at, *b"wave")?, 0, fourcc)
}

/// Validates the fixed fields of an MP4/MOV PCM sample entry.
fn validate_mp4_pcm_entry(entry: &containr::SampleEntry) -> Result<(), String> {
    let audio = entry
        .audio
        .ok_or("PCM sample entry has no audio fields".to_owned())?;
    if audio.channels == 0 {
        return Err("PCM sample entry declares zero channels".to_owned());
    }
    if audio.sample_rate == 0 {
        return Err("PCM sample entry declares zero sample rate".to_owned());
    }
    let raw = &entry.raw;
    let version = be_u16(raw.get(16..18).ok_or("truncated PCM sample entry")?)
        .ok_or("truncated PCM sample entry version")?;
    let bits = match version {
        0 | 1 => u32::from(
            be_u16(raw.get(26..28).ok_or("truncated PCM sample entry")?)
                .ok_or("truncated PCM sample size")?,
        ),
        2 => u32::from_be_bytes(
            raw.get(56..60)
                .ok_or("truncated PCM v2 sample entry")?
                .try_into()
                .map_err(|_| "invalid PCM v2 bits per channel".to_owned())?,
        ),
        other => return Err(format!("unsupported PCM sample entry version {other}")),
    };
    let is_float = entry.fourcc.0 == *b"fl32"
        || (version == 2
            && u32::from_be_bytes(
                raw.get(60..64)
                    .ok_or("truncated PCM v2 flags")?
                    .try_into()
                    .map_err(|_| "invalid PCM v2 flags".to_owned())?,
            ) & 0x01
                != 0);
    let allowed: &[u32] = if is_float {
        &[32, 64]
    } else {
        &[8, 16, 24, 32]
    };
    if !allowed.contains(&bits) {
        return Err(format!("unsupported PCM bit depth {bits}"));
    }
    Ok(())
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

fn avi_audio_codec(
    format_tag: u16,
    bits: Option<u16>,
    handler: Option<[u8; 4]>,
) -> Result<VideoCodec, String> {
    if handler == Some(*b"OPUS") {
        return Ok(VideoCodec::Opus);
    }
    match format_tag {
        0x00FF | 0x1600 | 0x1601 | 0x1602 | 0x706D => Ok(VideoCodec::Aac),
        0x704F | 0x7361 => Ok(VideoCodec::Opus),
        0x0055 => Ok(VideoCodec::Mp3),
        0x0001 => match bits {
            Some(8 | 16 | 24 | 32) => Ok(VideoCodec::Pcm),
            Some(bits) => Err(format!("unsupported AVI PCM bit depth {bits}")),
            None => Err("AVI PCM stream is missing wBitsPerSample".to_owned()),
        },
        0x0003 => match bits {
            Some(32 | 64) => Ok(VideoCodec::Pcm),
            Some(bits) => Err(format!("unsupported AVI IEEE-float bit depth {bits}")),
            None => Err("AVI IEEE-float stream is missing wBitsPerSample".to_owned()),
        },
        _ => Ok(VideoCodec::Unknown),
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
        match avi_stream_info(chunk, &avih) {
            Ok(Some(info)) => streams.push(info),
            Ok(None) => {}
            Err(reason) => return Err(invalid_video(path, reason)),
        }
    }
    if streams.is_empty() {
        return Err(invalid_video(path, "AVI header has no usable streams"));
    }
    Ok(streams)
}

/// Converts one AVI `strl` list into a validated stream, ignoring unknown stream types.
fn avi_stream_info(chunk: &[u8], avih: &AvihInfo) -> Result<Option<VideoStreamInfo>, String> {
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
        return Ok(None);
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
        let bits = strf.and_then(|body| le_u16(body.get(14..16).unwrap_or(&[])));
        let mut codec = avi_audio_codec(0, bits, handler)?;
        let mut channels = None;
        let mut sample_rate = None;
        if let Some(body) = strf
            && body.len() >= 16
        {
            codec = avi_audio_codec(le_u16(&body[..2]).unwrap_or(0), bits, handler)?;
            channels = le_u16(&body[2..4]);
            sample_rate = le_u32(&body[4..8]);
        }
        if codec == VideoCodec::Pcm
            && (channels.is_none_or(|v| v == 0) || sample_rate.is_none_or(|v| v == 0))
        {
            return Err("AVI PCM stream declares zero channels or sample rate".to_owned());
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
    Ok(Some(VideoStreamInfo::new(
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
    )))
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
                pmt_needed = None;
                if let Some(section) = ts_section_start(&packet) {
                    if section.len() < 3 {
                        // The header is split; continuation packets complete it.
                        pmt_section.extend_from_slice(section);
                    } else if let Some(len) = ts_section_length(section)
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
                        .min(packet.payload.len());
                    pmt_section.extend_from_slice(&packet.payload[..take]);
                    consumed = take;
                    pmt_needed =
                        ts_section_length(&pmt_section).and_then(|len| 3_usize.checked_add(len));
                }
                if let Some(needed) = pmt_needed {
                    let left = needed.saturating_sub(pmt_section.len());
                    let take = left.min(packet.payload.len().saturating_sub(consumed));
                    if take > 0 {
                        pmt_section.extend_from_slice(&packet.payload[consumed..consumed + take]);
                    }
                }
            }
            if let Some(needed) = pmt_needed
                && pmt_section.len() >= needed
            {
                if let Some(found) = parse_pmt(&pmt_section[..needed]) {
                    streams = found;
                    break;
                }
                pmt_needed = None;
                pmt_section.clear();
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

/// Counts GIF frames and their centisecond delays without decoding any pixels.
///
/// A zero delay is normalised to 10 cs, which is the conventional minimum
/// presentation time used by animated GIF players.
fn probe_gif_video(path: &Path, file: File) -> Result<Vec<VideoStreamInfo>, ProbeError> {
    let mut options = gif::DecodeOptions::new();
    options.skip_frame_decoding(true);
    options.set_memory_limit(gif::MemoryLimit::Bytes(
        NonZeroU64::new(GIF_FRAME_MEMORY_LIMIT).unwrap_or(NonZeroU64::MIN),
    ));
    let mut decoder = options.read_info(BufReader::new(file)).map_err(|err| {
        invalid_video(
            path,
            format!("GIF header or frame metadata is invalid: {err}"),
        )
    })?;
    let width = u32::from(decoder.width());
    let height = u32::from(decoder.height());
    if width == 0 || height == 0 {
        return Err(invalid_video(path, "GIF canvas has a zero width or height"));
    }

    let mut frames = 0_u64;
    let mut total_delay_cs = 0_u64;
    // `skip_frame_decoding(true)` is set above, so metadata alone is available
    // without allocating any frame pixel buffers.
    while let Some(frame) = decoder
        .next_frame_info()
        .map_err(|err| invalid_video(path, format!("GIF frame data is invalid: {err}")))?
    {
        frames = frames.saturating_add(1);
        let delay = if frame.delay == 0 { 10 } else { frame.delay };
        total_delay_cs = total_delay_cs.saturating_add(u64::from(delay));
    }
    if frames == 0 {
        return Err(invalid_video(path, "GIF contains no frames"));
    }

    // GIF has a one-bit transparency mask per frame. A NETSCAPE2.0 repeat
    // count is the number of *replays* after the first play, while
    // `AnimationLoop::Times` counts total plays. No extension means one play.
    let loop_count = match decoder.repeat() {
        gif::Repeat::Infinite => AnimationLoop::Infinite,
        gif::Repeat::Finite(count) => {
            let plays = count.checked_add(1).ok_or_else(|| {
                invalid_video(path, "GIF loop count exceeds the maximum play count")
            })?;
            AnimationLoop::Times(
                NonZeroU16::new(plays)
                    .ok_or_else(|| invalid_video(path, "GIF loop count is zero"))?,
            )
        }
    };
    let animation = AnimationMeta::new(loop_count, true);
    Ok(vec![
        VideoStreamInfo::new(
            VideoCodec::Gif,
            VideoStreamKind::Video,
            Some(width),
            Some(height),
            None,
            None,
            1,
            100,
            Some(total_delay_cs),
            None,
        )
        .with_animation(animation),
    ])
}

/// Three-byte little-endian field used by the WebP RIFF headers.
fn le_u24(bytes: &[u8]) -> Option<u32> {
    let raw = bytes.get(..3)?;
    Some(u32::from(raw[0]) | (u32::from(raw[1]) << 8) | (u32::from(raw[2]) << 16))
}

/// Reads a fixed-size header at an absolute offset without buffering the file.
fn read_chunk_at(
    path: &Path,
    file: &mut File,
    offset: u64,
    buf: &mut [u8],
) -> Result<(), ProbeError> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    file.read_exact(buf).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

/// Dimensions of a raw VP8 keyframe (the payload of a `VP8 ` chunk).
fn vp8_dimensions(payload: &[u8]) -> Option<(u32, u32)> {
    if payload.get(3..6)? != [0x9D, 0x01, 0x2A] {
        return None;
    }
    let width = u32::from(le_u16(payload.get(6..8)?)?) & 0x3FFF;
    let height = u32::from(le_u16(payload.get(8..10)?)?) & 0x3FFF;
    (width > 0 && height > 0).then_some((width, height))
}

/// Dimensions of a lossless VP8L bitstream (the payload of a `VP8L` chunk).
fn vp8l_dimensions(payload: &[u8]) -> Option<(u32, u32)> {
    if *payload.first()? != 0x2F {
        return None;
    }
    let packed = le_u32(payload.get(1..5)?)?;
    let width = (packed & 0x3FFF).saturating_add(1);
    let height = ((packed >> 14) & 0x3FFF).saturating_add(1);
    Some((width, height))
}

fn non_zero_dimensions(width: u32, height: u32) -> bool {
    width > 0 && height > 0
}

#[derive(Default)]
struct WebpProbeScan {
    still: Option<(u32, u32)>,
    extended: Option<(bool, u32, u32)>,
    animated: bool,
    frames: u64,
    total_ms: u64,
    /// Raw ANIM loop count; zero means infinite playback.
    loop_count: u16,
    alpha: bool,
}

/// An `ANMF` payload must contain at least one nested `VP8 ` or `VP8L`
/// bitstream subchunk (RFC 9649). Only fixed-size headers are read.
fn anmf_has_bitstream(
    path: &Path,
    file: &mut File,
    payload_start: u64,
    payload_end: u64,
) -> Result<bool, ProbeError> {
    let mut pos = payload_start.saturating_add(16);
    let mut found_image = false;
    // RFC 9649 defines the ANMF payload as a sequence of complete padded
    // subchunks; an overrun after the image chunk still invalidates the frame.
    while pos < payload_end {
        if payload_end.saturating_sub(pos) < 8 {
            return Err(invalid_video(
                path,
                "WebP ANMF subchunk header is truncated",
            ));
        }
        let mut header = [0_u8; 8];
        read_chunk_at(path, file, pos, &mut header)?;
        let id = [header[0], header[1], header[2], header[3]];
        let size = u64::from(
            le_u32(&header[4..])
                .ok_or_else(|| invalid_video(path, "WebP ANMF subchunk size truncated"))?,
        );
        let start = pos.saturating_add(8);
        let end = start
            .checked_add(size)
            .ok_or_else(|| invalid_video(path, "WebP ANMF subchunk size overflows"))?;
        if end > payload_end {
            return Err(invalid_video(path, "WebP ANMF subchunk is truncated"));
        }
        let padded = end
            .checked_add(size & 1)
            .ok_or_else(|| invalid_video(path, "WebP ANMF subchunk padding overflows"))?;
        if padded > payload_end {
            return Err(invalid_video(
                path,
                "WebP ANMF subchunk padding is truncated",
            ));
        }
        if (id == *b"VP8 " || id == *b"VP8L") && size > 0 {
            found_image = true;
        }
        pos = padded;
    }
    Ok(found_image)
}

/// Visits RIFF chunks with fixed-size reads, skipping `ANMF` payloads.
fn scan_webp_chunks(
    path: &Path,
    file: &mut File,
    riff_end: u64,
) -> Result<WebpProbeScan, ProbeError> {
    let mut scan = WebpProbeScan::default();
    let mut pos = 12_u64;
    while pos.saturating_add(8) <= riff_end {
        let mut header = [0_u8; 8];
        read_chunk_at(path, file, pos, &mut header)?;
        let id = [header[0], header[1], header[2], header[3]];
        let size = u64::from(
            le_u32(&header[4..]).ok_or_else(|| invalid_video(path, "WebP chunk size truncated"))?,
        );
        let payload_start = pos.saturating_add(8);
        let payload_end = payload_start
            .checked_add(size)
            .ok_or_else(|| invalid_video(path, "WebP chunk size overflows"))?;
        if payload_end > riff_end {
            return Err(invalid_video(path, "WebP RIFF chunk is truncated"));
        }
        match &id {
            b"VP8 " if scan.still.is_none() => {
                let mut prefix = [0_u8; 10];
                if size < 10 {
                    return Err(invalid_video(path, "WebP VP8 chunk is truncated"));
                }
                read_chunk_at(path, file, payload_start, &mut prefix)?;
                scan.still = Some(vp8_dimensions(&prefix).ok_or_else(|| {
                    invalid_video(path, "WebP VP8 bitstream has invalid dimensions")
                })?);
            }
            b"VP8L" if scan.still.is_none() => {
                let mut prefix = [0_u8; 5];
                if size < 5 {
                    return Err(invalid_video(path, "WebP VP8L chunk is truncated"));
                }
                read_chunk_at(path, file, payload_start, &mut prefix)?;
                scan.still = Some(vp8l_dimensions(&prefix).ok_or_else(|| {
                    invalid_video(path, "WebP VP8L bitstream has invalid dimensions")
                })?);
            }
            b"VP8X" if scan.extended.is_none() => {
                let mut prefix = [0_u8; 10];
                if size < 10 {
                    return Err(invalid_video(path, "WebP VP8X chunk is truncated"));
                }
                read_chunk_at(path, file, payload_start, &mut prefix)?;
                let width = le_u24(&prefix[4..7])
                    .ok_or_else(|| invalid_video(path, "WebP VP8X width is truncated"))?
                    .saturating_add(1);
                let height = le_u24(&prefix[7..10])
                    .ok_or_else(|| invalid_video(path, "WebP VP8X height is truncated"))?
                    .saturating_add(1);
                scan.extended = Some((prefix[0] & 0x02 != 0, width, height));
                // RFC 9649 VP8X flags: 0x20 ICC, 0x10 alpha, 0x08 EXIF, 0x04 XMP, 0x02 animation.
                scan.alpha = prefix[0] & 0x10 != 0;
            }
            b"ANIM" => {
                if size < 6 {
                    return Err(invalid_video(
                        path,
                        "WebP ANIM chunk is shorter than six bytes",
                    ));
                }
                scan.animated = true;
                // RFC 9649 ANIM payload: 4-byte background color followed by 2-byte LE loop count.
                let mut loop_bytes = [0_u8; 2];
                read_chunk_at(path, file, payload_start.saturating_add(4), &mut loop_bytes)?;
                scan.loop_count = u16::from_le_bytes(loop_bytes);
            }
            b"ANMF" => {
                let mut duration = [0_u8; 3];
                if size < 16 {
                    return Err(invalid_video(path, "WebP ANMF chunk is too short"));
                }
                read_chunk_at(path, file, payload_start.saturating_add(12), &mut duration)?;
                let ms = le_u24(&duration)
                    .ok_or_else(|| invalid_video(path, "WebP ANMF duration is truncated"))?;
                if !anmf_has_bitstream(path, file, payload_start, payload_end)? {
                    return Err(invalid_video(
                        path,
                        "WebP ANMF has no nested VP8 or VP8L bitstream",
                    ));
                }
                scan.frames = scan.frames.saturating_add(1);
                scan.total_ms =
                    scan.total_ms
                        .saturating_add(u64::from(if ms == 0 { 100 } else { ms }));
            }
            _ => {}
        }
        let padded_end = payload_end.saturating_add(size & 1);
        if padded_end > riff_end {
            return Err(invalid_video(path, "WebP RIFF chunk padding is truncated"));
        }
        pos = padded_end;
    }
    Ok(scan)
}

/// Probes animated and still WebP files with a seek-based RIFF chunk walk.
///
/// Only fixed-size chunk headers are read and `ANMF` payloads are skipped, so
/// the probe uses O(1) RAM even for animations larger than 16 MiB. Bytes after
/// the declared RIFF end are ignored as trailing padding.
fn probe_webp_video(
    path: &Path,
    file: &mut File,
    file_len: u64,
) -> Result<Vec<VideoStreamInfo>, ProbeError> {
    let mut signature = [0_u8; 12];
    read_chunk_at(path, file, 0, &mut signature)?;
    if &signature[..4] != b"RIFF" || &signature[8..12] != b"WEBP" {
        return Err(invalid_video(path, "truncated RIFF/WEBP header"));
    }
    let declared = u64::from(
        le_u32(&signature[4..8]).ok_or_else(|| invalid_video(path, "WebP RIFF size is missing"))?,
    );
    let riff_end = declared
        .checked_add(8)
        .ok_or_else(|| invalid_video(path, "WebP RIFF size overflows"))?;
    if declared < 4 || riff_end > file_len {
        return Err(invalid_video(
            path,
            "WebP RIFF size extends past the end of the file",
        ));
    }

    let scan = scan_webp_chunks(path, file, riff_end)?;
    let (width, height, total_ms) = match scan.extended {
        Some((true, width, height)) => {
            if !scan.animated || scan.frames == 0 {
                return Err(invalid_video(
                    path,
                    "animated WebP is missing ANIM or ANMF chunks",
                ));
            }
            (width, height, scan.total_ms)
        }
        Some((false, width, height)) => {
            if scan.still.is_none() {
                return Err(invalid_video(
                    path,
                    "extended WebP has no VP8 or VP8L image chunk",
                ));
            }
            (width, height, 100)
        }
        None => {
            let (width, height) = scan
                .still
                .ok_or_else(|| invalid_video(path, "WebP has no VP8, VP8L, or VP8X image chunk"))?;
            (width, height, 100)
        }
    };
    if !non_zero_dimensions(width, height) {
        return Err(invalid_video(
            path,
            "WebP canvas has a zero width or height",
        ));
    }

    let loop_count = if scan.loop_count == 0 {
        AnimationLoop::Infinite
    } else {
        AnimationLoop::Times(
            std::num::NonZeroU16::new(scan.loop_count).unwrap_or(std::num::NonZeroU16::MIN),
        )
    };
    let animation = AnimationMeta::new(loop_count, scan.alpha);
    Ok(vec![
        VideoStreamInfo::new(
            VideoCodec::Webp,
            VideoStreamKind::Video,
            Some(width),
            Some(height),
            None,
            None,
            1,
            1000,
            Some(total_ms),
            None,
        )
        .with_animation(animation),
    ])
}

// Each container family validates its codec-specific headers in one pass.
#[allow(clippy::too_many_lines)]
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
            let mut streams = Vec::new();
            for track in demuxer.tracks() {
                // `containr` has no `dfLa`/PCM codec-config variant, so validate
                // the raw sample entry here at the single system boundary.
                let facts = match stream_codec(track) {
                    VideoCodec::Flac => {
                        let entry = track
                            .sample_entry()
                            .ok_or_else(|| invalid_video(path, "FLAC track has no sample entry"))?;
                        let dfla = mp4_audio_config_box(entry, *b"dfLa").ok_or_else(|| {
                            invalid_video(path, "FLAC sample entry is missing dfLa")
                        })?;
                        Some(validate_dfla(dfla).map_err(|reason| invalid_video(path, reason))?)
                    }
                    VideoCodec::Pcm => {
                        let entry = track
                            .sample_entry()
                            .ok_or_else(|| invalid_video(path, "PCM track has no sample entry"))?;
                        validate_mp4_pcm_entry(entry)
                            .map_err(|reason| invalid_video(path, reason))?;
                        None
                    }
                    _ => None,
                };
                streams.push(stream_info(track, facts));
            }
            streams
        }
        VideoContainerFormat::Mkv | VideoContainerFormat::Webm => {
            let reader = WindowedFileReader::from_file(file, file_len);
            let demuxer = containr::mkv::Demuxer::open(reader)
                .map_err(|err| invalid_video(path, format!("container parse failed: {err}")))?;
            if demuxer.truncated() {
                return Err(invalid_video(path, "truncated MKV input"));
            }
            let mut streams = Vec::new();
            for track in demuxer.tracks() {
                let codec_id = track.codec_id();
                let ident_facts = match codec_id {
                    Some("A_VORBIS") => {
                        let private = track
                            .matroska
                            .as_ref()
                            .and_then(|mkv| mkv.codec_private.as_deref())
                            .ok_or_else(|| {
                                invalid_video(path, "A_VORBIS track is missing CodecPrivate")
                            })?;
                        Some(
                            validate_vorbis_codec_private(private)
                                .map_err(|reason| invalid_video(path, reason))?,
                        )
                    }
                    Some("A_FLAC") => {
                        let private = track
                            .matroska
                            .as_ref()
                            .and_then(|mkv| mkv.codec_private.as_deref())
                            .ok_or_else(|| {
                                invalid_video(path, "A_FLAC track is missing CodecPrivate")
                            })?;
                        Some(
                            validate_flac_codec_private(private)
                                .map_err(|reason| invalid_video(path, reason))?,
                        )
                    }
                    Some(id) if id.starts_with("A_PCM/") => {
                        let audio = track
                            .matroska
                            .as_ref()
                            .and_then(|mkv| mkv.audio.as_ref())
                            .ok_or_else(|| invalid_video(path, "A_PCM track is missing Audio"))?;
                        let bits = audio.bit_depth.unwrap_or(0);
                        let float = id == "A_PCM/FLOAT/IEEE";
                        let allowed: &[u64] = if float { &[32, 64] } else { &[8, 16, 24, 32] };
                        if !allowed.contains(&bits) {
                            return Err(invalid_video(
                                path,
                                format!("unsupported A_PCM bit depth {bits}"),
                            ));
                        }
                        if audio.channels == 0 {
                            return Err(invalid_video(path, "A_PCM track declares zero channels"));
                        }
                        let rate = matroska_rate(audio.sampling_frequency).ok_or_else(|| {
                            invalid_video(path, "A_PCM track has invalid sample rate")
                        })?;
                        let channels = u16::try_from(audio.channels).map_err(|_| {
                            invalid_video(path, "A_PCM channel count exceeds 65535")
                        })?;
                        Some((channels, rate))
                    }
                    _ => None,
                };
                streams.push(stream_info(track, ident_facts));
            }
            streams
        }
        VideoContainerFormat::Avi => probe_avi(path, &mut file, file_len)?,
        VideoContainerFormat::Ts => probe_ts(path, &mut file, file_len)?,
        VideoContainerFormat::Gif => probe_gif_video(path, file)?,
        VideoContainerFormat::Webp => probe_webp_video(path, &mut file, file_len)?,
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
            "unrecognized container magic bytes (expected MP4, MKV, WebM, MOV, AVI, TS, GIF, or WebP)",
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
    if !format.supports_remux() {
        return Err(ProbeError::UnsupportedTargetFormat {
            path: target_dir.to_path_buf(),
            expected: "mp4, mkv, webm, mov, avi, or ts",
        });
    }
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
        .filter(|path| {
            detect_video_magic_file(path).is_some_and(VideoContainerFormat::supports_remux)
        })
        .collect();
    sources.sort();

    if sources.is_empty() {
        return Err(ProbeError::EmptyVideoDirectory(source_dir.to_path_buf()));
    }
    let canonical_sources = canonical_sources(&sources)?;

    let mut items = Vec::with_capacity(sources.len());
    for source in sources {
        let probed = probe_video(&source)?;
        if !probed.format.supports_remux() {
            return Err(invalid_video(
                &source,
                "GIF and WebP sources cannot be stream-copied; use `fastconv video`",
            ));
        }
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
        if !format.supports_remux() {
            return Err(ProbeError::UnsupportedTargetFormat {
                path: target.to_path_buf(),
                expected: "mp4, mkv, webm, mov, avi, or ts",
            });
        }
        return prepare_directory_batch(source, target, format, threads);
    }
    if !source_meta.is_file() {
        return Err(ProbeError::NotAFileOrDirectory(source.to_path_buf()));
    }

    let (resolved_target, format) = resolve_single_target(source, target, to)?;
    if !format.supports_remux() {
        return Err(ProbeError::UnsupportedTargetFormat {
            path: resolved_target,
            expected: "mp4, mkv, webm, mov, avi, or ts",
        });
    }
    reject_alias_collision(source, &resolved_target)?;
    let probed = probe_video(source)?;
    if !probed.format.supports_remux() {
        return Err(invalid_video(
            source,
            "GIF and WebP sources cannot be stream-copied; use `fastconv video`",
        ));
    }
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
    Drop,
}

/// Parses the raw `--acodec` value at the system boundary. Codec names and the
/// copy/drop aliases are recognised here so no other layer re-parses CLI text.
///
/// # Errors
///
/// Returns [`ProbeError::InvalidVideo`] with an empty path when `raw` names no
/// known audio codec.
pub fn parse_requested_audio_codec(raw: &str) -> Result<RequestedAudioCodec, ProbeError> {
    match raw.to_ascii_lowercase().as_str() {
        "copy" => Ok(RequestedAudioCodec::Copy),
        "none" => Ok(RequestedAudioCodec::Drop),
        _ => match TargetAudioCodec::from_name(raw) {
            Some(codec) => Ok(RequestedAudioCodec::Encode(codec)),
            None => Err(ProbeError::InvalidVideo {
                path: PathBuf::new(),
                reason: format!(
                    "unknown audio codec '{raw}' (expected aac, opus, mp3, vorbis, flac, pcm, none, or copy)"
                ),
            }),
        },
    }
}

/// Rate-control and audio overrides for a transcode batch. `None` rate control
/// means the user did not pass `--quality` or `--vbitrate`, which keeps stream
/// copy legal and leaves the encoder on its `VideoQuality::DEFAULT` behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VideoTranscodeOptions {
    rate_control: Option<VideoRateControl>,
    audio: AudioEncodeOptions,
}

impl VideoTranscodeOptions {
    #[must_use]
    pub const fn new(rate_control: Option<VideoRateControl>, audio: AudioEncodeOptions) -> Self {
        Self {
            rate_control,
            audio,
        }
    }

    /// Legacy quality-only callers: the default quality is not an explicit
    /// flag, so it must not turn `--vcodec copy` into a conflict.
    #[must_use]
    pub const fn with_quality(quality: VideoQuality) -> Self {
        let rate_control = if quality.get() == VideoQuality::DEFAULT.get() {
            None
        } else {
            Some(VideoRateControl::Quality(quality))
        };
        Self {
            rate_control,
            audio: AudioEncodeOptions::new(None, None, None),
        }
    }

    #[must_use]
    pub const fn rate_control(self) -> Option<VideoRateControl> {
        self.rate_control
    }

    #[must_use]
    pub const fn audio_options(self) -> AudioEncodeOptions {
        self.audio
    }
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
    prepare_video_transcode_batch_with_decode(
        source,
        target,
        to,
        video_codec,
        audio_codec,
        quality,
        backend,
        HardwareDecodeBackend::Auto,
        worker_threads,
    )
}

/// Same as [`prepare_video_transcode_batch`] but with an explicit decode
/// backend request. `Nvdec` is rejected here unless the source codec and the
/// encoder backend can actually execute it — no silent CPU fallback.
///
/// # Errors
///
/// Returns [`ProbeError`] for invalid paths, unsupported combinations, or a
/// forced NVDEC request that cannot be honoured.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn prepare_video_transcode_batch_with_decode(
    source: &Path,
    target: &Path,
    to: Option<VideoContainerFormat>,
    video_codec: Option<RequestedVideoCodec>,
    audio_codec: Option<RequestedAudioCodec>,
    quality: VideoQuality,
    backend: HardwareBackend,
    decode_backend: HardwareDecodeBackend,
    worker_threads: Option<NonZeroUsize>,
) -> Result<VideoTranscodePlan, ProbeError> {
    prepare_video_transcode_batch_with_options(
        source,
        target,
        to,
        video_codec,
        audio_codec,
        VideoTranscodeOptions::with_quality(quality),
        backend,
        decode_backend,
        worker_threads,
    )
}

/// Full boundary planning entry point: carries explicit video rate control and
/// audio overrides into the validated domain plan.
///
/// # Errors
///
/// Returns [`ProbeError`] for invalid paths, conflicting options, unsupported
/// codec/container pairs, or a forced NVDEC request that cannot be honoured.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn prepare_video_transcode_batch_with_options(
    source: &Path,
    target: &Path,
    to: Option<VideoContainerFormat>,
    video_codec: Option<RequestedVideoCodec>,
    audio_codec: Option<RequestedAudioCodec>,
    options: VideoTranscodeOptions,
    backend: HardwareBackend,
    decode_backend: HardwareDecodeBackend,
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

    let copy_video = matches!(video_codec, Some(RequestedVideoCodec::Copy));
    let rate_control = resolve_video_rate_control(copy_video, options.rate_control())?;
    let encode_codec = match video_codec {
        Some(RequestedVideoCodec::Encode(codec)) => codec,
        _ => output_format.default_video_codec(),
    };
    // GIF/WebP have no bitrate-aware encoder. Check the resolved codec *and*
    // the container so an implicit animation default is rejected as well.
    if matches!(rate_control, VideoRateControl::Bitrate(_))
        && (matches!(encode_codec, TargetVideoCodec::Gif | TargetVideoCodec::Webp)
            || !output_format.supports_audio())
    {
        return Err(DomainError::UnsupportedVideoPipeline {
            codec: encode_codec.as_video_codec(),
            backend,
            reason: format!(
                "{encode_codec} animation encoding supports only --quality, not --vbitrate"
            ),
        }
        .into());
    }

    let animation_target = !output_format.supports_audio();
    validate_audio_options_for_container(options.audio_options(), output_format)?;
    let audio_copy_or_drop = matches!(
        audio_codec,
        Some(RequestedAudioCodec::Copy | RequestedAudioCodec::Drop)
    ) || (audio_codec.is_none() && animation_target);
    let audio_options = resolve_audio_encode_options(audio_copy_or_drop, options.audio_options())?;

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
            VideoTrackAction::Encode(VideoEncodeMode::resolve(codec, rate_control, backend)?)
        }
        None => VideoTrackAction::Encode(VideoEncodeMode::resolve(
            encode_codec,
            rate_control,
            backend,
        )?),
    };
    let audio_action = match audio_codec {
        Some(RequestedAudioCodec::Copy) => AudioTrackAction::Copy,
        Some(RequestedAudioCodec::Drop) => AudioTrackAction::Drop,
        Some(RequestedAudioCodec::Encode(codec)) => AudioTrackAction::Encode(
            AudioEncodeConfig::new(codec, audio_options).map_err(ProbeError::InvalidDomain)?,
        ),
        None => match output_format.default_audio_codec() {
            Some(codec) => AudioTrackAction::Encode(
                AudioEncodeConfig::new(codec, audio_options).map_err(ProbeError::InvalidDomain)?,
            ),
            None => AudioTrackAction::Drop,
        },
    };

    let mut items = Vec::with_capacity(specs.len());
    for (input, output) in specs {
        reject_alias_collision(&input, &output)?;
        let probed = probe_video(&input)?;
        let video_streams: Vec<&VideoStreamInfo> = probed
            .streams
            .iter()
            .filter(|s| s.kind() == VideoStreamKind::Video)
            .collect();
        let source_codec = video_streams
            .first()
            .map_or(VideoCodec::Unknown, |s| s.codec());
        // One decode mode is stored per item, so `auto` may only pick NVDEC when
        // every video track is decodable; a single unsupported track forces CPU
        // decode for the whole item. Forced NVDEC must stay a hard error.
        let resolved_backend = if decode_backend == HardwareDecodeBackend::Auto
            && !video_streams.iter().all(|s| s.codec().nvdec_decodable())
        {
            HardwareDecodeBackend::Cpu
        } else {
            decode_backend
        };
        let decode_mode = resolved_backend.resolve(source_codec, video_action.backend())?;
        items.push(VideoTranscodeItem::new(
            input,
            output,
            probed.format,
            output_format,
            probed.streams,
            video_action,
            audio_action,
            decode_mode,
        )?);
    }
    Ok(VideoTranscodePlan::new(items, threads)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastconv_domain::{AudioBitrate, AudioChannels, AudioSampleRate, VideoBitrate};
    use std::io::{Read, Seek, SeekFrom};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

    fn err_contains<T>(result: &Result<T, ProbeError>, needle: &str) -> bool {
        result
            .as_ref()
            .err()
            .is_some_and(|err| err.to_string().contains(needle))
    }

    fn audio_encode(codec: TargetAudioCodec) -> AudioTrackAction {
        AudioTrackAction::Encode(AudioEncodeConfig::default_for(codec))
    }

    fn quality_mode(codec: TargetVideoCodec, quality: VideoQuality) -> VideoTrackAction {
        let rate_control = VideoRateControl::Quality(quality);
        VideoTrackAction::Encode(match codec {
            TargetVideoCodec::H264 => VideoEncodeMode::CpuH264 { rate_control },
            TargetVideoCodec::Av1 => VideoEncodeMode::CpuAv1 { rate_control },
            TargetVideoCodec::Vp9 => VideoEncodeMode::CpuVp9 { rate_control },
            TargetVideoCodec::Hevc => VideoEncodeMode::NvencHevc { rate_control },
            TargetVideoCodec::Gif => VideoEncodeMode::CpuGif { quality },
            TargetVideoCodec::Webp => VideoEncodeMode::CpuWebp { quality },
        })
    }

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

    /// EBML element with a variable-width size integer, for payloads above
    /// the 126-byte limit of `ebml_el`.
    fn ebml_el_vint(id: u32, body: &[u8]) -> Vec<u8> {
        let raw = id.to_be_bytes();
        let skip = raw.iter().take_while(|&&b| b == 0).count();
        let mut out = raw[skip..].to_vec();
        let len = body.len();
        let mut width = 1_usize;
        while len >= (1_usize << (7 * width)) - 1 {
            width += 1;
        }
        let marker = 0x100_u16 >> width;
        for shift in (0..width).rev() {
            // EBML VINT_DATA is a big-endian integer of 8-bit octets, not
            // 7-bit groups; only the first octet's marker bit is reserved.
            let byte = u8::try_from((len >> (8 * shift)) & 0xFF).unwrap_or(0);
            out.push(if shift == width - 1 {
                byte | u8::try_from(marker).unwrap_or(0)
            } else {
                byte
            });
        }
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn ebml_el_vint_encodes_exact_body_boundaries() -> Result<(), Box<dyn std::error::Error>> {
        for len in [126, 127, 128, 255] {
            let body = vec![0x5A_u8; len];
            let encoded = ebml_el_vint(0xAE, &body);
            let size_bytes = encoded.get(1..).ok_or("missing size VINT")?;
            let (size, vint_len) = ebml_vint_size(size_bytes).ok_or("size VINT did not decode")?;
            assert_eq!(size, len, "wrong size for {len}-byte body");
            assert_eq!(
                encoded.get(1 + vint_len..).ok_or("truncated body")?,
                body.as_slice(),
                "wrong body boundary for {len}-byte body"
            );
        }
        Ok(())
    }

    fn vorbis_ident_bytes() -> Vec<u8> {
        let mut ident = vec![0_u8; 30];
        ident[0] = 0x01;
        ident[1..7].copy_from_slice(b"vorbis");
        ident[7..11].copy_from_slice(&0_u32.to_le_bytes());
        ident[11] = 2;
        ident[12..16].copy_from_slice(&44_100_u32.to_le_bytes());
        ident[28] = 0xB8; // bs0=8, bs1=11
        ident[29] = 1;
        ident
    }

    fn vorbis_codec_private() -> Vec<u8> {
        let ident = vorbis_ident_bytes();
        let comment = b"\x03vorbis";
        let setup = b"\x05vorbis";
        let mut private = vec![0x02, u8::try_from(ident.len()).unwrap_or(0)];
        private.push(u8::try_from(comment.len()).unwrap_or(0));
        private.extend_from_slice(&ident);
        private.extend_from_slice(comment);
        private.extend_from_slice(setup);
        private
    }

    /// Selectable Matroska `Audio` children; `None` omits the child entirely.
    #[derive(Default)]
    struct AudioFacts {
        channels: Option<u64>,
        sampling_frequency: Option<f64>,
        bit_depth: Option<u64>,
    }

    fn mkv_with_vorbis(private: &[u8], audio: Option<AudioFacts>) -> Vec<u8> {
        let mut header = Vec::new();
        header.extend(ebml_uint(0x4286, 1));
        header.extend(ebml_uint(0x42F2, 4));
        header.extend(ebml_uint(0x42F3, 8));
        header.extend(ebml_el(0x4282, b"matroska"));
        header.extend(ebml_uint(0x4287, 4));
        header.extend(ebml_uint(0x4285, 2));
        let mut segment = Vec::new();
        segment.extend(ebml_el_vint(
            0x1549_A966,
            &ebml_uint(0x002A_D7B1, 1_000_000),
        ));
        let mut entry = Vec::new();
        entry.extend(ebml_uint(0xD7, 1));
        entry.extend(ebml_uint(0x73C5, 1));
        entry.extend(ebml_uint(0x83, 2)); // audio track
        entry.extend(ebml_el(0x86, b"A_VORBIS"));
        entry.extend(ebml_el_vint(0x63A2, private));
        if let Some(audio) = audio {
            let mut audio_body = Vec::new();
            if let Some(channels) = audio.channels {
                audio_body.extend(ebml_uint(0x9F, channels));
            }
            if let Some(rate) = audio.sampling_frequency {
                audio_body.extend(ebml_el(0xB5, &rate.to_be_bytes()));
            }
            entry.extend(ebml_el_vint(0xE1, &audio_body));
        }
        segment.extend(ebml_el_vint(0x1654_AE6B, &ebml_el_vint(0xAE, &entry)));
        let mut bytes = ebml_el_vint(0x1A45_DFA3, &header);
        bytes.extend(ebml_el_vint(0x1853_8067, &segment));
        bytes
    }

    #[test]
    fn probe_accepts_vorbis_and_falls_back_to_ident_facts() -> Result<(), Box<dyn std::error::Error>>
    {
        let path = temp_file("vorbis-ok", &mkv_with_vorbis(&vorbis_codec_private(), None))?;
        let probed = probe_video(&path)?;
        let audio = probed
            .streams
            .iter()
            .find(|stream| stream.codec() == VideoCodec::Vorbis)
            .ok_or("no Vorbis stream was probed")?;
        assert!(audio.channels().is_some_and(|value| value.get() == 2));
        assert!(
            audio
                .sample_rate()
                .is_some_and(|value| value.get() == 44_100)
        );
        fs::remove_file(&path)?;
        Ok(())
    }

    #[test]
    fn vorbis_ident_facts_override_matroska_audio_defaults()
    -> Result<(), Box<dyn std::error::Error>> {
        let cases = [
            ("empty Audio", AudioFacts::default(), 2, 44_100),
            (
                "Channels only",
                AudioFacts {
                    channels: Some(2),
                    sampling_frequency: None,
                    bit_depth: None,
                },
                2,
                44_100,
            ),
            (
                "SamplingFrequency only",
                AudioFacts {
                    channels: None,
                    sampling_frequency: Some(44_100.0),
                    bit_depth: None,
                },
                2,
                44_100,
            ),
            (
                "explicit non-defaults",
                AudioFacts {
                    channels: Some(6),
                    sampling_frequency: Some(48_000.0),
                    bit_depth: None,
                },
                6,
                48_000,
            ),
        ];
        for (label, audio, expected_channels, expected_rate) in cases {
            let path = temp_file(
                "vorbis-audio",
                &mkv_with_vorbis(&vorbis_codec_private(), Some(audio)),
            )?;
            let probed = probe_video(&path)?;
            let stream = probed
                .streams
                .iter()
                .find(|stream| stream.codec() == VideoCodec::Vorbis)
                .ok_or("no Vorbis stream was probed")?;
            assert_eq!(
                stream.channels().map(std::num::NonZero::get),
                Some(expected_channels),
                "{label}: wrong channels"
            );
            assert_eq!(
                stream.sample_rate().map(std::num::NonZero::get),
                Some(expected_rate),
                "{label}: wrong sample rate"
            );
            fs::remove_file(&path)?;
        }
        Ok(())
    }

    #[test]
    fn probe_rejects_malformed_vorbis_codec_private() -> Result<(), Box<dyn std::error::Error>> {
        let good = vorbis_codec_private();
        let mut variants = vec![Vec::new(), vec![0x01, 30], vec![0x02, 0xFF]];
        let mut missing_setup = good.clone();
        missing_setup.truncate(missing_setup.len() - 8);
        variants.push(missing_setup);
        for index in [0, 1, 2, good.len() - 7] {
            let mut bad = good.clone();
            if let Some(byte) = bad.get_mut(index) {
                *byte ^= 0x01;
                variants.push(bad);
            }
        }
        for private in variants {
            let path = temp_file("vorbis-bad", &mkv_with_vorbis(&private, None))?;
            let result = probe_video(&path);
            assert!(
                matches!(result, Err(ProbeError::InvalidVideo { .. })),
                "malformed CodecPrivate was accepted: {result:?}"
            );
            fs::remove_file(&path)?;
        }
        Ok(())
    }

    #[test]
    fn probe_rejects_missing_vorbis_codec_private() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_file("vorbis-missing", &mkv_with_vorbis(&[], None))?;
        let result = probe_video(&path);
        assert!(
            matches!(result, Err(ProbeError::InvalidVideo { .. })),
            "missing CodecPrivate was accepted: {result:?}"
        );
        fs::remove_file(&path)?;
        Ok(())
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
        let short = payload.len() < 184;
        let mut packet = vec![
            0x47,
            high | if pusi { 0x40 } else { 0 },
            low,
            if short { 0x30 } else { 0x10 },
        ];
        if short {
            // Fill the remainder with a real adaptation field so no 0xFF
            // stuffing is mistaken for part of a PSI section.
            let af_len = 183_usize.saturating_sub(payload.len());
            packet.push(u8::try_from(af_len).unwrap_or(0));
            packet.push(0x00);
            packet.resize(188_usize.saturating_sub(payload.len()), 0xFF);
        }
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
    fn ts_pmt_first_byte_alone_is_reassembled() -> Result<(), Box<dyn std::error::Error>> {
        let mut association_body = vec![0x00, 0x01, 0xC1, 0x00, 0x00];
        association_body.extend_from_slice(&[0x00, 0x01, 0xF0, 0x00]);
        let mut program_body = vec![0x00, 0x01, 0xC1, 0x00, 0x00, 0xF0, 0x00, 0xF0, 0x00];
        program_body.extend_from_slice(&ts_es(0x0F, 0x0110, &[]));
        let pat = ts_section(0x00, &association_body);
        let pmt = ts_section(0x02, &program_body);
        let mut bytes = ts_packet(0, true, &[&[0_u8][..], pat.as_slice()].concat());
        // Only the table id arrives with the PUSI; the section header is
        // completed by the continuation packet.
        bytes.extend_from_slice(&ts_packet(0x1000, true, &[0, pmt[0]]));
        bytes.extend_from_slice(&ts_packet(0x1000, false, &pmt[1..]));
        let path = temp_file("ts-pmt-first-byte", &bytes)?;
        let probed = probe_video(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(probed.streams.len(), 1);
        assert_eq!(probed.streams[0].codec(), VideoCodec::Aac);
        Ok(())
    }

    #[test]
    fn ts_pmt_header_ending_in_ff_is_not_trimmed() -> Result<(), Box<dyn std::error::Error>> {
        let mut association_body = vec![0x00, 0x01, 0xC1, 0x00, 0x00];
        association_body.extend_from_slice(&[0x00, 0x01, 0xF0, 0x00]);
        let mut program_body = vec![0x00, 0x01, 0xC1, 0x00, 0x00, 0xF0, 0x00, 0xF0, 0x00];
        for index in 0..45_u16 {
            program_body.extend_from_slice(&ts_es(0x1B, 0x0200 + index, &[]));
        }
        program_body.extend_from_slice(&ts_es(0x1B, 0x02FF, &[0xAA]));
        program_body.extend_from_slice(&ts_es(0x06, 0x0112, &ts_desc(0x05, b"Opus")));
        assert_eq!(program_body.len(), 251);
        let pmt = ts_section(0x02, &program_body);
        assert_eq!(&pmt[..3], &[0x02, 0x80, 0xFF]);
        assert_eq!(pmt.len(), 258);

        let mut pat = vec![0x00];
        pat.extend_from_slice(&ts_section(0x00, &association_body));
        let mut bytes = ts_packet(0, true, &pat);
        // The PUSI carries exactly the 3-byte header, whose last byte is the
        // low byte of section_length (255) and must survive reassembly.
        let mut first = vec![0x00];
        first.extend_from_slice(&pmt[..3]);
        bytes.extend_from_slice(&ts_packet(0x1000, true, &first));
        for part in pmt[3..].chunks(184) {
            bytes.extend_from_slice(&ts_packet(0x1000, false, part));
        }
        let path = temp_file("ts-pmt-header-ff", &bytes)?;
        let probed = probe_video(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(probed.streams.len(), 47);
        assert_eq!(probed.streams[46].codec(), VideoCodec::Opus);
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
            quality_mode(TargetVideoCodec::Vp9, quality)
        );
        assert_eq!(
            webm.items()[0].audio_action(),
            audio_encode(TargetAudioCodec::Opus)
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
            quality_mode(TargetVideoCodec::H264, quality)
        );
        assert_eq!(
            mp4.items()[0].audio_action(),
            audio_encode(TargetAudioCodec::Aac)
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

    /// Minimal TS with two video tracks; each flag marks an AV1 track (else H.264).
    fn two_video_track_ts(tracks: [bool; 2]) -> Vec<u8> {
        let mut association_body = vec![0x00, 0x01, 0xC1, 0x00, 0x00];
        association_body.extend_from_slice(&[0x00, 0x01, 0xF0, 0x00]);
        let mut program_map_body = vec![0x00, 0x01, 0xC1, 0x00, 0x00, 0xF0, 0x00, 0xF0, 0x00];
        for (index, av1) in tracks.into_iter().enumerate() {
            let pid = 0x0100_u16.saturating_add(u16::try_from(index).unwrap_or(0));
            if av1 {
                program_map_body.extend_from_slice(&ts_es(0x06, pid, &ts_desc(0x05, b"AV01")));
            } else {
                program_map_body.extend_from_slice(&ts_es(0x1B, pid, &[]));
            }
        }
        let pat = ts_section(0x00, &association_body);
        let pmt = ts_section(0x02, &program_map_body);
        let mut bytes = ts_packet(0, true, &[&[0_u8][..], pat.as_slice()].concat());
        bytes.extend_from_slice(&ts_packet(
            0x1000,
            true,
            &[&[0_u8][..], pmt.as_slice()].concat(),
        ));
        // MPEG-TS detection needs three sync bytes.
        bytes.extend_from_slice(&ts_packet(0x1FFF, false, &[]));
        bytes
    }

    #[test]
    fn auto_decode_resolves_against_all_video_tracks() -> Result<(), Box<dyn std::error::Error>> {
        let quality = VideoQuality::DEFAULT;
        let nvenc = HardwareBackend::Gpu(fastconv_domain::GpuBackend::Nvenc);

        for tracks in [
            [false, true], // H.264 then AV1
            [true, false], // AV1 then H.264
        ] {
            let source = temp_file("auto-decode-mixed", &two_video_track_ts(tracks))?;
            let probed = probe_video(&source)?;
            assert_eq!(probed.streams.len(), 2);
            for (stream, av1) in probed.streams.iter().zip(tracks) {
                assert_eq!(
                    stream.codec(),
                    if av1 {
                        VideoCodec::Av1
                    } else {
                        VideoCodec::H264
                    }
                );
            }

            let target = temp_path("auto-decode-mixed-target").with_extension("mp4");
            let plan = prepare_video_transcode_batch_with_decode(
                &source,
                &target,
                Some(VideoContainerFormat::Mp4),
                None,
                None,
                quality,
                nvenc,
                HardwareDecodeBackend::Auto,
                Some(plan_threads()),
            )?;
            assert_eq!(plan.items().len(), 1);
            assert_eq!(
                plan.items()[0].decode_mode(),
                fastconv_domain::VideoDecodeMode::Cpu
            );

            let forced_target = temp_path("forced-nvdec-mixed-target").with_extension("mp4");
            let forced = prepare_video_transcode_batch_with_decode(
                &source,
                &forced_target,
                Some(VideoContainerFormat::Mp4),
                None,
                None,
                quality,
                nvenc,
                HardwareDecodeBackend::Nvdec,
                Some(plan_threads()),
            );
            match forced {
                Err(ProbeError::InvalidDomain(DomainError::UnsupportedVideoPipeline {
                    reason,
                    ..
                })) => assert!(reason.contains("NVDEC"), "reason: {reason}"),
                other => return Err(format!("forced NVDEC must be rejected: {other:?}").into()),
            }
            fs::remove_file(&source)?;
        }

        // Every track qualifies: `auto` must keep the NVDEC path.
        let source = temp_file("auto-decode-all-nvdec", &two_video_track_ts([false, false]))?;
        let target = temp_path("auto-decode-all-nvdec-target").with_extension("mp4");
        let plan = prepare_video_transcode_batch_with_decode(
            &source,
            &target,
            Some(VideoContainerFormat::Mp4),
            None,
            None,
            quality,
            nvenc,
            HardwareDecodeBackend::Auto,
            Some(plan_threads()),
        )?;
        assert_eq!(
            plan.items()[0].decode_mode(),
            fastconv_domain::VideoDecodeMode::Nvdec
        );
        fs::remove_file(&source)?;
        Ok(())
    }

    fn le24(value: u32) -> [u8; 3] {
        let bytes = value.to_le_bytes();
        [bytes[0], bytes[1], bytes[2]]
    }

    /// Minimal animated GIF with an optional `NETSCAPE2.0` application extension.
    fn synthetic_gif_with_loop(
        width: u16,
        height: u16,
        frames: &[(u16, u16, u16)],
        netscape_loop: Option<u16>,
    ) -> Vec<u8> {
        let mut bytes = b"GIF89a".to_vec();
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.push(0x80); // global colour table with two entries
        bytes.push(0);
        bytes.push(0);
        bytes.extend_from_slice(&[0, 0, 0, 0xFF, 0xFF, 0xFF]);
        if let Some(count) = netscape_loop {
            bytes.extend_from_slice(&[0x21, 0xFF, 11]);
            bytes.extend_from_slice(b"NETSCAPE2.0");
            bytes.extend_from_slice(&[3, 1]);
            bytes.extend_from_slice(&count.to_le_bytes());
            bytes.push(0);
        }
        for &(delay, frame_width, frame_height) in frames {
            bytes.extend_from_slice(&[0x21, 0xF9, 0x04, 0x00]);
            bytes.extend_from_slice(&delay.to_le_bytes());
            bytes.extend_from_slice(&[0x00, 0x00]);
            bytes.push(0x2C);
            bytes.extend_from_slice(&[0, 0, 0, 0]);
            bytes.extend_from_slice(&frame_width.to_le_bytes());
            bytes.extend_from_slice(&frame_height.to_le_bytes());
            bytes.push(0);
            bytes.push(2); // LZW minimum code size
            bytes.extend_from_slice(&[1, 0x00, 0x00]);
        }
        bytes.push(0x3B);
        bytes
    }

    /// Minimal animated GIF: one palette, then one graphic-control + image per frame.
    fn synthetic_gif(width: u16, height: u16, frames: &[(u16, u16, u16)]) -> Vec<u8> {
        synthetic_gif_with_loop(width, height, frames, None)
    }

    fn synthetic_webp_header(width: u32, height: u32, animated: bool) -> Vec<u8> {
        let mut payload = vec![u8::from(animated) << 1, 0, 0, 0];
        payload.extend_from_slice(&le24(width.saturating_sub(1)));
        payload.extend_from_slice(&le24(height.saturating_sub(1)));
        riff_chunk(*b"VP8X", &payload)
    }

    fn synthetic_webp_frame(width: u32, height: u32, delay_ms: u32) -> Vec<u8> {
        let mut payload = vec![0_u8; 16];
        payload[6..9].copy_from_slice(&le24(width.saturating_sub(1)));
        payload[9..12].copy_from_slice(&le24(height.saturating_sub(1)));
        payload[12..15].copy_from_slice(&le24(delay_ms));
        // RFC 9649: every ANMF frame carries at least one nested image chunk.
        let packed =
            ((width.saturating_sub(1)) & 0x3FFF) | (((height.saturating_sub(1)) & 0x3FFF) << 14);
        let mut bitstream = vec![0x2F];
        bitstream.extend_from_slice(&packed.to_le_bytes());
        payload.extend_from_slice(&riff_chunk(*b"VP8L", &bitstream));
        riff_chunk(*b"ANMF", &payload)
    }

    fn synthetic_webp_with_loop(delays: &[u32], loop_count: u16, alpha: bool) -> Vec<u8> {
        let mut body = b"WEBP".to_vec();
        // RFC 9649 VP8X flags: alpha is bit 4 (0x10), animation is bit 1 (0x02).
        let mut flags = vec![(u8::from(alpha) << 4) | 0x02];
        flags.extend_from_slice(&[0, 0, 0]);
        flags.extend_from_slice(&le24(7));
        flags.extend_from_slice(&le24(7));
        body.extend_from_slice(&riff_chunk(*b"VP8X", &flags));
        let mut anim = [0_u8; 6];
        anim[4..6].copy_from_slice(&loop_count.to_le_bytes());
        body.extend_from_slice(&riff_chunk(*b"ANIM", &anim));
        for &delay in delays {
            body.extend_from_slice(&synthetic_webp_frame(8, 8, delay));
        }
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(u32::try_from(body.len()).unwrap_or(0)).to_le_bytes());
        bytes.extend_from_slice(&body);
        bytes
    }

    fn synthetic_animated_webp(width: u32, height: u32, delays: &[u32]) -> Vec<u8> {
        let mut body = b"WEBP".to_vec();
        body.extend_from_slice(&synthetic_webp_header(width, height, true));
        body.extend_from_slice(&riff_chunk(*b"ANIM", &[0_u8; 6]));
        for &delay in delays {
            body.extend_from_slice(&synthetic_webp_frame(width, height, delay));
        }
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(u32::try_from(body.len()).unwrap_or(0)).to_le_bytes());
        bytes.extend_from_slice(&body);
        bytes
    }

    fn synthetic_still_webp(width: u32, height: u32, lossless: bool, extended: bool) -> Vec<u8> {
        let (id, payload) = if lossless {
            let packed = ((width.saturating_sub(1)) & 0x3FFF)
                | (((height.saturating_sub(1)) & 0x3FFF) << 14);
            let mut payload = vec![0x2F];
            payload.extend_from_slice(&packed.to_le_bytes());
            (*b"VP8L", payload)
        } else {
            let mut payload = vec![0_u8; 10];
            payload[3..6].copy_from_slice(&[0x9D, 0x01, 0x2A]);
            let w = (width & 0x3FFF).to_le_bytes();
            let h = (height & 0x3FFF).to_le_bytes();
            payload[6..8].copy_from_slice(&w[..2]);
            payload[8..10].copy_from_slice(&h[..2]);
            (*b"VP8 ", payload)
        };
        let mut body = b"WEBP".to_vec();
        if extended {
            body.extend_from_slice(&synthetic_webp_header(width, height, false));
        }
        body.extend_from_slice(&riff_chunk(id, &payload));
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(u32::try_from(body.len()).unwrap_or(0)).to_le_bytes());
        bytes.extend_from_slice(&body);
        bytes
    }

    #[test]
    fn detects_gif_and_webp_magic() {
        assert_eq!(
            detect_video_magic(b"GIF87a\x01\x00\x01\x00"),
            Some(VideoContainerFormat::Gif)
        );
        assert_eq!(
            detect_video_magic(b"GIF89a\x01\x00\x01\x00"),
            Some(VideoContainerFormat::Gif)
        );
        assert_eq!(
            detect_video_magic(b"RIFF\x04\x00\x00\x00WEBP"),
            Some(VideoContainerFormat::Webp)
        );
        assert_eq!(
            detect_video_magic(b"RIFF\x04\x00\x00\x00AVI "),
            Some(VideoContainerFormat::Avi)
        );
    }

    #[test]
    fn probes_animated_and_still_gif() -> Result<(), Box<dyn std::error::Error>> {
        let animated = temp_file(
            "animated-gif",
            &synthetic_gif(32, 24, &[(0, 32, 24), (7, 32, 24), (3, 32, 24)]),
        )?;
        let probed = probe_video(&animated)?;
        assert_eq!(probed.format, VideoContainerFormat::Gif);
        assert_eq!(probed.streams.len(), 1);
        let stream = &probed.streams[0];
        assert_eq!(stream.codec(), VideoCodec::Gif);
        assert_eq!(stream.kind(), VideoStreamKind::Video);
        assert_eq!(stream.width().map(std::num::NonZero::get), Some(32));
        assert_eq!(stream.height().map(std::num::NonZero::get), Some(24));
        assert_eq!(stream.timebase(), (1, 100));
        assert_eq!(stream.duration(), Some(20)); // 0 -> 10 cs, then 7 + 3

        let still = temp_file("still-gif", &synthetic_gif(8, 6, &[(5, 8, 6)]))?;
        let probed = probe_video(&still)?;
        assert_eq!(probed.streams[0].duration(), Some(5));
        let once = probed.streams[0]
            .animation()
            .ok_or("GIF must carry animation metadata")?;
        assert_eq!(once.loop_count(), AnimationLoop::Times(NonZeroU16::MIN));

        let finite = temp_file(
            "gif-finite-loop",
            &synthetic_gif_with_loop(8, 6, &[(5, 8, 6)], Some(3)),
        )?;
        let probed = probe_video(&finite)?;
        let meta = probed.streams[0]
            .animation()
            .ok_or("GIF must carry animation metadata")?;
        // NETSCAPE count 3 means three replays after the first play: four total plays.
        let times = NonZeroU16::new(4).unwrap_or(NonZeroU16::MIN);
        assert_eq!(meta.loop_count(), AnimationLoop::Times(times));

        // 65,535 replays would be 65,536 total plays and cannot be represented.
        let overflow = temp_file(
            "gif-loop-overflow",
            &synthetic_gif_with_loop(8, 6, &[(5, 8, 6)], Some(u16::MAX)),
        )?;
        assert!(matches!(
            probe_video(&overflow),
            Err(ProbeError::InvalidVideo { .. })
        ));

        let infinite = temp_file(
            "gif-infinite-loop",
            &synthetic_gif_with_loop(8, 6, &[(5, 8, 6)], Some(0)),
        )?;
        let probed = probe_video(&infinite)?;
        let meta = probed.streams[0]
            .animation()
            .ok_or("GIF must carry animation metadata")?;
        assert_eq!(meta.loop_count(), AnimationLoop::Infinite);
        Ok(())
    }

    #[test]
    fn rejects_gif_without_frames_or_canvas() -> Result<(), Box<dyn std::error::Error>> {
        let empty = temp_file("empty-gif", &synthetic_gif(16, 16, &[]))?;
        assert!(matches!(
            probe_video(&empty),
            Err(ProbeError::InvalidVideo { .. })
        ));
        let zero_canvas = temp_file("zero-gif", &synthetic_gif(0, 16, &[(5, 0, 16)]))?;
        assert!(matches!(
            probe_video(&zero_canvas),
            Err(ProbeError::InvalidVideo { .. })
        ));
        let corrupt = temp_file("corrupt-gif", b"GIF89a\x00\x00\x00\x00\x80\x00\x00")?;
        assert!(matches!(
            probe_video(&corrupt),
            Err(ProbeError::InvalidVideo { .. })
        ));
        Ok(())
    }

    #[test]
    fn probes_animated_and_still_webp() -> Result<(), Box<dyn std::error::Error>> {
        let animated = temp_file(
            "animated-webp",
            &synthetic_animated_webp(64, 48, &[0, 33, 33]),
        )?;
        let probed = probe_video(&animated)?;
        assert_eq!(probed.format, VideoContainerFormat::Webp);
        assert_eq!(probed.streams.len(), 1);
        let stream = &probed.streams[0];
        assert_eq!(stream.codec(), VideoCodec::Webp);
        assert_eq!(stream.kind(), VideoStreamKind::Video);
        assert_eq!(stream.width().map(std::num::NonZero::get), Some(64));
        assert_eq!(stream.height().map(std::num::NonZero::get), Some(48));
        assert_eq!(stream.timebase(), (1, 1000));
        assert_eq!(stream.duration(), Some(166)); // 0 -> 100 ms

        for lossless in [false, true] {
            for extended in [false, true] {
                let still = temp_file(
                    "still-webp",
                    &synthetic_still_webp(20, 10, lossless, extended),
                )?;
                let probed = probe_video(&still)?;
                assert_eq!(probed.streams[0].codec(), VideoCodec::Webp);
                assert_eq!(
                    probed.streams[0].width().map(std::num::NonZero::get),
                    Some(20)
                );
                assert_eq!(
                    probed.streams[0].height().map(std::num::NonZero::get),
                    Some(10)
                );
                assert_eq!(probed.streams[0].duration(), Some(100));
            }
        }
        Ok(())
    }

    #[test]
    fn rejects_corrupt_webp() -> Result<(), Box<dyn std::error::Error>> {
        let truncated = temp_file("truncated-webp", b"RIFF\x10\x00\x00\x00WEBP")?;
        assert!(matches!(
            probe_video(&truncated),
            Err(ProbeError::InvalidVideo { .. })
        ));
        let mut no_anim = b"WEBP".to_vec();
        no_anim.extend_from_slice(&synthetic_webp_header(16, 16, true));
        no_anim.extend_from_slice(&synthetic_webp_frame(16, 16, 20));
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(u32::try_from(no_anim.len()).unwrap_or(0)).to_le_bytes());
        bytes.extend_from_slice(&no_anim);
        let path = temp_file("webp-no-anim", &bytes)?;
        assert!(matches!(
            probe_video(&path),
            Err(ProbeError::InvalidVideo { .. })
        ));

        let mut no_frames = b"WEBP".to_vec();
        no_frames.extend_from_slice(&synthetic_webp_header(16, 16, true));
        no_frames.extend_from_slice(&riff_chunk(*b"ANIM", &[0_u8; 6]));
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(u32::try_from(no_frames.len()).unwrap_or(0)).to_le_bytes());
        bytes.extend_from_slice(&no_frames);
        let path = temp_file("webp-no-frames", &bytes)?;
        assert!(matches!(
            probe_video(&path),
            Err(ProbeError::InvalidVideo { .. })
        ));

        let path = temp_file(
            "webp-zero-dimensions",
            &synthetic_still_webp(0, 0, false, false),
        )?;
        assert!(matches!(
            probe_video(&path),
            Err(ProbeError::InvalidVideo { .. })
        ));
        Ok(())
    }

    #[test]
    fn animated_webp_carries_loop_count_and_alpha() -> Result<(), Box<dyn std::error::Error>> {
        let finite = temp_file(
            "webp-finite-loop",
            &synthetic_webp_with_loop(&[10], 3, true),
        )?;
        let probed = probe_video(&finite)?;
        let meta = probed.streams[0]
            .animation()
            .ok_or("animated WebP must carry animation metadata")?;
        let times = std::num::NonZeroU16::new(3).unwrap_or(std::num::NonZeroU16::MIN);
        assert_eq!(meta.loop_count(), AnimationLoop::Times(times));
        assert!(meta.has_alpha());

        let forever = temp_file(
            "webp-forever-loop",
            &synthetic_webp_with_loop(&[10], 0, false),
        )?;
        let probed = probe_video(&forever)?;
        let meta = probed.streams[0]
            .animation()
            .ok_or("animated WebP must carry animation metadata")?;
        assert_eq!(meta.loop_count(), AnimationLoop::Infinite);
        assert!(!meta.has_alpha());

        // Setting the XMP flag (0x04) without the alpha flag (0x10) must not report alpha.
        let mut xmp_only = synthetic_webp_with_loop(&[10], 0, false);
        xmp_only[20] |= 0x04;
        let path = temp_file("webp-xmp-no-alpha", &xmp_only)?;
        let probed = probe_video(&path)?;
        let meta = probed.streams[0]
            .animation()
            .ok_or("animated WebP must carry animation metadata")?;
        assert!(!meta.has_alpha());
        Ok(())
    }

    #[test]
    fn webp_anmf_requires_bitstream_and_padding() -> Result<(), Box<dyn std::error::Error>> {
        let mut body = b"WEBP".to_vec();
        body.extend_from_slice(&synthetic_webp_header(16, 16, true));
        body.extend_from_slice(&riff_chunk(*b"ANIM", &[0_u8; 6]));
        // An ANMF with only the 16-byte header has no nested image bitstream.
        body.extend_from_slice(&riff_chunk(*b"ANMF", &[0_u8; 16]));
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(u32::try_from(body.len()).unwrap_or(0)).to_le_bytes());
        bytes.extend_from_slice(&body);
        let path = temp_file("webp-anmf-no-bitstream", &bytes)?;
        assert!(matches!(
            probe_video(&path),
            Err(ProbeError::InvalidVideo { .. })
        ));

        // An ANMF with an odd-sized nested VP8L subchunk missing its pad byte must fail.
        let mut anmf_payload = vec![0_u8; 16];
        anmf_payload[6..9].copy_from_slice(&le24(15));
        anmf_payload[9..12].copy_from_slice(&le24(15));
        anmf_payload[12..15].copy_from_slice(&le24(20));
        anmf_payload.extend_from_slice(b"VP8L");
        anmf_payload.extend_from_slice(&5_u32.to_le_bytes());
        anmf_payload.extend_from_slice(&[0x2F, 0, 0, 0, 0]);
        let mut body = b"WEBP".to_vec();
        body.extend_from_slice(&synthetic_webp_header(16, 16, true));
        body.extend_from_slice(&riff_chunk(*b"ANIM", &[0_u8; 6]));
        body.extend_from_slice(&riff_chunk(*b"ANMF", &anmf_payload));
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(u32::try_from(body.len()).unwrap_or(0)).to_le_bytes());
        bytes.extend_from_slice(&body);
        let path = temp_file("webp-anmf-missing-subchunk-pad", &bytes)?;
        match probe_video(&path) {
            Err(ProbeError::InvalidVideo { reason, .. }) => {
                assert!(reason.contains("padding"), "unexpected reason: {reason}");
            }
            other => return Err(format!("expected padding error, got {other:?}").into()),
        }

        // A valid image followed by a truncated subchunk header must still be rejected.
        let mut anmf_payload = vec![0_u8; 16];
        anmf_payload[6..9].copy_from_slice(&le24(15));
        anmf_payload[9..12].copy_from_slice(&le24(15));
        anmf_payload[12..15].copy_from_slice(&le24(20));
        anmf_payload.extend_from_slice(b"VP8L");
        anmf_payload.extend_from_slice(&5_u32.to_le_bytes());
        anmf_payload.extend_from_slice(&[0x2F, 0, 0, 0, 0]);
        anmf_payload.extend_from_slice(&[0xAB, 0xCD, 0xEF]);
        let mut body = b"WEBP".to_vec();
        body.extend_from_slice(&synthetic_webp_header(16, 16, true));
        body.extend_from_slice(&riff_chunk(*b"ANIM", &[0_u8; 6]));
        body.extend_from_slice(&riff_chunk(*b"ANMF", &anmf_payload));
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&u32::try_from(body.len()).unwrap_or(0).to_le_bytes());
        bytes.extend_from_slice(&body);
        let path = temp_file("webp-anmf-trailing-header", &bytes)?;
        match probe_video(&path) {
            Err(ProbeError::InvalidVideo { reason, .. }) => {
                assert!(reason.contains("header"), "unexpected reason: {reason}");
            }
            other => return Err(format!("expected truncated-header error, got {other:?}").into()),
        }

        // An odd-sized RIFF chunk must have its pad byte inside the container.
        let mut body = b"WEBP".to_vec();
        body.extend_from_slice(&synthetic_webp_header(16, 16, false));
        let mut odd = b"VP8L".to_vec();
        odd.extend_from_slice(&5_u32.to_le_bytes());
        odd.extend_from_slice(&[0x2F, 0, 0, 0, 0]);
        body.extend_from_slice(&odd);
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&u32::try_from(body.len()).unwrap_or(0).to_le_bytes());
        bytes.extend_from_slice(&body);
        bytes.push(0);
        let path = temp_file("webp-missing-pad", &bytes)?;
        match probe_video(&path) {
            Err(ProbeError::InvalidVideo { reason, .. }) => {
                assert!(reason.contains("padding"), "unexpected reason: {reason}");
            }
            other => return Err(format!("expected padding error, got {other:?}").into()),
        }

        // Trailing bytes after a complete RIFF container are still accepted.
        let valid = synthetic_still_webp(16, 16, true, false);
        let mut with_tail = valid.clone();
        with_tail.extend_from_slice(b"trailing");
        let path = temp_file("webp-trailing", &with_tail)?;
        assert_eq!(probe_video(&path)?.format, VideoContainerFormat::Webp);
        Ok(())
    }

    #[test]
    fn remux_rejects_animation_sources_and_targets() -> Result<(), Box<dyn std::error::Error>> {
        let gif = temp_file("remux-gif", &synthetic_gif(8, 8, &[(5, 8, 8)]))?;
        let target = temp_path("remux-gif-target").with_extension("mkv");
        assert!(prepare_video_batch(&gif, &target, Some(VideoContainerFormat::Mkv), None).is_err());

        let avi = temp_file("remux-avi", &minimal_avi())?;
        let gif_target = temp_path("remux-to-gif").with_extension("gif");
        assert!(prepare_video_batch(&avi, &gif_target, None, Some(plan_threads())).is_err());
        let webp_target = temp_path("remux-to-webp").with_extension("webp");
        assert!(
            prepare_video_batch(
                &avi,
                &webp_target,
                Some(VideoContainerFormat::Webp),
                Some(plan_threads())
            )
            .is_err()
        );

        let source_dir = temp_path("remux-animation-dir");
        fs::create_dir_all(&source_dir)?;
        fs::write(source_dir.join("a.gif"), synthetic_gif(8, 8, &[(5, 8, 8)]))?;
        let target_dir = temp_path("remux-animation-out");
        let result = prepare_video_batch(
            &source_dir,
            &target_dir,
            Some(VideoContainerFormat::Mkv),
            Some(plan_threads()),
        );
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn transcode_plans_animation_sources_and_targets() -> Result<(), Box<dyn std::error::Error>> {
        let gif = temp_file("transcode-gif", &synthetic_gif(8, 8, &[(5, 8, 8)]))?;
        let webp = temp_file("transcode-webp", &synthetic_animated_webp(8, 8, &[10]))?;
        let quality = VideoQuality::DEFAULT;

        let gif_target = temp_path("transcode-into-gif").with_extension("gif");
        let plan = prepare_video_transcode_batch(
            &gif,
            &gif_target,
            Some(VideoContainerFormat::Gif),
            None,
            None,
            quality,
            HardwareBackend::Cpu,
            Some(plan_threads()),
        )?;
        assert_eq!(plan.output_format(), VideoContainerFormat::Gif);
        assert_eq!(
            plan.items()[0].video_action(),
            VideoTrackAction::Encode(VideoEncodeMode::CpuGif { quality })
        );
        assert_eq!(plan.items()[0].audio_action(), AudioTrackAction::Drop);

        let webp_target = temp_path("transcode-into-webp").with_extension("webp");
        let plan = prepare_video_transcode_batch(
            &webp,
            &webp_target,
            Some(VideoContainerFormat::Webp),
            None,
            Some(RequestedAudioCodec::Drop),
            quality,
            HardwareBackend::Cpu,
            Some(plan_threads()),
        )?;
        assert_eq!(
            plan.items()[0].video_action(),
            VideoTrackAction::Encode(VideoEncodeMode::CpuWebp { quality })
        );
        assert_eq!(plan.items()[0].audio_action(), AudioTrackAction::Drop);

        let mp4_target = temp_path("transcode-gif-nvenc").with_extension("mp4");
        let plan = prepare_video_transcode_batch(
            &gif,
            &mp4_target,
            Some(VideoContainerFormat::Mp4),
            Some(RequestedVideoCodec::Encode(TargetVideoCodec::H264)),
            None,
            quality,
            HardwareBackend::Gpu(fastconv_domain::GpuBackend::Nvenc),
            Some(plan_threads()),
        )?;
        assert_eq!(
            plan.items()[0].video_action(),
            VideoTrackAction::Encode(VideoEncodeMode::NvencH264 {
                rate_control: VideoRateControl::Quality(quality)
            })
        );
        assert_eq!(
            plan.items()[0].audio_action(),
            audio_encode(TargetAudioCodec::Aac)
        );

        let cpu_target = temp_path("transcode-webp-cpu").with_extension("mp4");
        let plan = prepare_video_transcode_batch(
            &webp,
            &cpu_target,
            Some(VideoContainerFormat::Mp4),
            None,
            None,
            quality,
            HardwareBackend::Cpu,
            Some(plan_threads()),
        )?;
        assert_eq!(
            plan.items()[0].video_action(),
            quality_mode(TargetVideoCodec::H264, quality)
        );
        Ok(())
    }

    #[test]
    fn transcode_options_carry_video_bitrate() -> Result<(), Box<dyn std::error::Error>> {
        let source = temp_file("rate-bitrate", &minimal_avi())?;
        let target = temp_path("rate-bitrate").with_extension("mp4");
        let bitrate = VideoBitrate::parse("1M")?;
        let options = VideoTranscodeOptions::new(
            Some(VideoRateControl::Bitrate(bitrate)),
            AudioEncodeOptions::default(),
        );
        let plan = prepare_video_transcode_batch_with_options(
            &source,
            &target,
            Some(VideoContainerFormat::Mp4),
            None,
            None,
            options,
            HardwareBackend::Cpu,
            HardwareDecodeBackend::Auto,
            Some(plan_threads()),
        )?;
        assert_eq!(
            plan.items()[0].video_action(),
            VideoTrackAction::Encode(VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::Bitrate(bitrate)
            })
        );
        Ok(())
    }

    #[test]
    fn transcode_options_reject_vbitrate_with_copy_and_animation()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = temp_file("rate-conflict", &minimal_avi())?;
        let target = temp_path("rate-conflict").with_extension("mp4");
        let options = VideoTranscodeOptions::new(
            Some(VideoRateControl::Bitrate(VideoBitrate::parse("1M")?)),
            AudioEncodeOptions::default(),
        );
        let copy = prepare_video_transcode_batch_with_options(
            &source,
            &target,
            Some(VideoContainerFormat::Mp4),
            Some(RequestedVideoCodec::Copy),
            None,
            options,
            HardwareBackend::Cpu,
            HardwareDecodeBackend::Auto,
            Some(plan_threads()),
        );
        assert!(
            err_contains(&copy, "stream copy cannot re-encode"),
            "unexpected copy error: {copy:?}"
        );

        for to in [VideoContainerFormat::Gif, VideoContainerFormat::Webp] {
            let animation_target = temp_path("rate-animation").with_extension(to.extension());
            let result = prepare_video_transcode_batch_with_options(
                &source,
                &animation_target,
                Some(to),
                None,
                None,
                options,
                HardwareBackend::Cpu,
                HardwareDecodeBackend::Auto,
                Some(plan_threads()),
            );
            assert!(
                err_contains(&result, "animation encoding supports only --quality"),
                "unexpected {to} error: {result:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn transcode_options_validate_audio_overrides() -> Result<(), Box<dyn std::error::Error>> {
        let source = temp_file("audio-options", &minimal_avi())?;
        let target = temp_path("audio-options").with_extension("mp4");
        let options = VideoTranscodeOptions::new(
            None,
            AudioEncodeOptions::new(
                Some(AudioBitrate::parse("192k")?),
                Some(AudioSampleRate::new(48_000)?),
                Some(AudioChannels::new(1)?),
            ),
        );
        let plan = prepare_video_transcode_batch_with_options(
            &source,
            &target,
            Some(VideoContainerFormat::Mp4),
            None,
            Some(RequestedAudioCodec::Encode(TargetAudioCodec::Aac)),
            options,
            HardwareBackend::Cpu,
            HardwareDecodeBackend::Auto,
            Some(plan_threads()),
        )?;
        let AudioTrackAction::Encode(config) = plan.items()[0].audio_action() else {
            return Err("expected an audio encode action".into());
        };
        assert_eq!(config.bitrate(), Some(AudioBitrate::parse("192k")?));
        assert_eq!(config.sample_rate(), Some(AudioSampleRate::new(48_000)?));
        assert_eq!(config.channels(), Some(AudioChannels::new(1)?));

        for requested in [
            Some(RequestedAudioCodec::Copy),
            Some(RequestedAudioCodec::Drop),
        ] {
            let result = prepare_video_transcode_batch_with_options(
                &source,
                &target,
                Some(VideoContainerFormat::Mp4),
                None,
                requested,
                options,
                HardwareBackend::Cpu,
                HardwareDecodeBackend::Auto,
                Some(plan_threads()),
            );
            assert!(
                err_contains(&result, "require audio re-encoding"),
                "unexpected {requested:?} error: {result:?}"
            );
        }

        let animation_target = temp_path("audio-animation").with_extension("gif");
        let animation = prepare_video_transcode_batch_with_options(
            &source,
            &animation_target,
            Some(VideoContainerFormat::Gif),
            None,
            None,
            options,
            HardwareBackend::Cpu,
            HardwareDecodeBackend::Auto,
            Some(plan_threads()),
        );
        assert!(
            err_contains(&animation, "have no audio track"),
            "unexpected animation error: {animation:?}"
        );

        let illegal = VideoTranscodeOptions::new(
            None,
            AudioEncodeOptions::new(None, Some(AudioSampleRate::new(44_100)?), None),
        );
        let opus_target = temp_path("audio-opus-limit").with_extension("webm");
        let opus = prepare_video_transcode_batch_with_options(
            &source,
            &opus_target,
            Some(VideoContainerFormat::Webm),
            None,
            Some(RequestedAudioCodec::Encode(TargetAudioCodec::Opus)),
            illegal,
            HardwareBackend::Cpu,
            HardwareDecodeBackend::Auto,
            Some(plan_threads()),
        );
        assert!(
            err_contains(&opus, "not supported by Opus"),
            "unexpected codec-limit error: {opus:?}"
        );
        Ok(())
    }

    #[test]
    fn parse_requested_audio_codec_covers_every_alias() {
        assert!(matches!(
            parse_requested_audio_codec("copy"),
            Ok(RequestedAudioCodec::Copy)
        ));
        assert!(matches!(
            parse_requested_audio_codec("NONE"),
            Ok(RequestedAudioCodec::Drop)
        ));
        assert!(matches!(
            parse_requested_audio_codec("Flac"),
            Ok(RequestedAudioCodec::Encode(TargetAudioCodec::Flac))
        ));
        assert!(matches!(
            parse_requested_audio_codec("pcm"),
            Ok(RequestedAudioCodec::Encode(TargetAudioCodec::Pcm))
        ));
        assert!(parse_requested_audio_codec("vp8").is_err());
    }

    fn flac_streaminfo(rate: u32, channels: u16, bits: u8) -> [u8; 34] {
        let mut si = [0_u8; 34];
        si[0..2].copy_from_slice(&4096_u16.to_be_bytes());
        si[2..4].copy_from_slice(&4096_u16.to_be_bytes());
        let packed =
            (u64::from(rate) << 44) | (u64::from(channels - 1) << 41) | (u64::from(bits - 1) << 36);
        si[10..18].copy_from_slice(&packed.to_be_bytes());
        si
    }

    fn flac_codec_private(si: &[u8; 34]) -> Vec<u8> {
        let mut private = b"fLaC".to_vec();
        private.extend_from_slice(&[0x80, 0x00, 0x00, 0x22]);
        private.extend_from_slice(si);
        private
    }

    fn mkv_with_audio_codec(codec_id: &[u8], private: &[u8], audio: Option<AudioFacts>) -> Vec<u8> {
        let mut header = Vec::new();
        header.extend(ebml_uint(0x4286, 1));
        header.extend(ebml_uint(0x42F2, 4));
        header.extend(ebml_uint(0x42F3, 8));
        header.extend(ebml_el(0x4282, b"matroska"));
        header.extend(ebml_uint(0x4287, 4));
        header.extend(ebml_uint(0x4285, 2));
        let mut segment = Vec::new();
        segment.extend(ebml_el_vint(
            0x1549_A966,
            &ebml_uint(0x002A_D7B1, 1_000_000),
        ));
        let mut entry = Vec::new();
        entry.extend(ebml_uint(0xD7, 1));
        entry.extend(ebml_uint(0x73C5, 1));
        entry.extend(ebml_uint(0x83, 2));
        entry.extend(ebml_el(0x86, codec_id));
        if !private.is_empty() {
            entry.extend(ebml_el_vint(0x63A2, private));
        }
        if let Some(audio) = audio {
            let mut audio_body = Vec::new();
            if let Some(channels) = audio.channels {
                audio_body.extend(ebml_uint(0x9F, channels));
            }
            if let Some(rate) = audio.sampling_frequency {
                audio_body.extend(ebml_el(0xB5, &rate.to_be_bytes()));
            }
            if let Some(depth) = audio.bit_depth {
                audio_body.extend(ebml_uint(0x6264, depth));
            }
            entry.extend(ebml_el_vint(0xE1, &audio_body));
        }
        segment.extend(ebml_el_vint(0x1654_AE6B, &ebml_el_vint(0xAE, &entry)));
        let mut bytes = ebml_el_vint(0x1A45_DFA3, &header);
        bytes.extend(ebml_el_vint(0x1853_8067, &segment));
        bytes
    }

    #[test]
    fn probe_accepts_flac_and_uses_streaminfo_facts() -> Result<(), Box<dyn std::error::Error>> {
        let si = flac_streaminfo(44_100, 2, 16);
        let bytes = mkv_with_audio_codec(b"A_FLAC", &flac_codec_private(&si), None);
        let path = temp_file("flac-ok", &bytes)?;
        let probed = probe_video(&path)?;
        let audio = probed
            .streams
            .iter()
            .find(|stream| stream.codec() == VideoCodec::Flac)
            .ok_or("no FLAC stream was probed")?;
        assert_eq!(audio.channels().map(NonZeroU16::get), Some(2));
        assert_eq!(
            audio.sample_rate().map(std::num::NonZeroU32::get),
            Some(44_100)
        );
        fs::remove_file(&path)?;
        Ok(())
    }

    #[test]
    fn probe_rejects_malformed_flac_codec_private() -> Result<(), Box<dyn std::error::Error>> {
        let si = flac_streaminfo(0, 2, 16);
        let mut bad_rate = flac_codec_private(&si);
        let path = temp_file(
            "flac-bad-rate",
            &mkv_with_audio_codec(b"A_FLAC", &bad_rate, None),
        )?;
        assert!(err_contains(&probe_video(&path), "rate"));
        fs::remove_file(&path)?;

        bad_rate[4] = 0x01; // not a STREAMINFO block
        let path = temp_file(
            "flac-bad-block",
            &mkv_with_audio_codec(b"A_FLAC", &bad_rate, None),
        )?;
        assert!(err_contains(&probe_video(&path), "STREAMINFO"));
        fs::remove_file(&path)?;

        let path = temp_file("flac-missing", &mkv_with_audio_codec(b"A_FLAC", &[], None))?;
        assert!(err_contains(&probe_video(&path), "CodecPrivate"));
        fs::remove_file(&path)?;
        Ok(())
    }

    #[test]
    fn mp4_dfla_and_pcm_entry_helpers_validate_headers() -> Result<(), Box<dyn std::error::Error>> {
        let si = flac_streaminfo(48_000, 2, 16);
        let mut dfla = vec![0_u8; 4];
        dfla.extend_from_slice(&[0x80, 0x00, 0x00, 0x22]);
        dfla.extend_from_slice(&si);
        assert_eq!(validate_dfla(&dfla).ok(), Some((2_u16, 48_000_u32)));
        let mut bad = dfla.clone();
        bad[0] = 1;
        assert!(validate_dfla(&bad).is_err());
        bad[0] = 0;
        bad[4] = 0x01;
        assert!(validate_dfla(&bad).is_err());

        // A version 0 audio sample entry with a dfLa child box.
        let mut child = u32::try_from(dfla.len() + 8)
            .map_err(|_| "test box too large")?
            .to_be_bytes()
            .to_vec();
        child.extend_from_slice(b"dfLa");
        child.extend_from_slice(&dfla);
        let mut body = vec![0_u8; 28];
        body[6..8].copy_from_slice(&1_u16.to_be_bytes());
        body[16..18].copy_from_slice(&2_u16.to_be_bytes());
        body[18..20].copy_from_slice(&16_u16.to_be_bytes());
        body[24..28].copy_from_slice(&(48_000_u32 << 16).to_be_bytes());
        let mut raw = u32::try_from(8 + body.len() + child.len())
            .map_err(|_| "test entry too large")?
            .to_be_bytes()
            .to_vec();
        raw.extend_from_slice(b"fLaC");
        raw.extend_from_slice(&body);
        raw.extend_from_slice(&child);
        let entry = containr::SampleEntry {
            fourcc: containr::FourCc(*b"fLaC"),
            raw,
            data_reference_index: 1,
            codec_config: None,
            video: None,
            audio: Some(containr::AudioInfo {
                channels: 2,
                sample_size: 16,
                sample_rate: 48_000,
            }),
        };
        let found = mp4_audio_config_box(&entry, *b"dfLa").ok_or("missing dfLa");
        assert!(found.is_ok(), "dfLa must be found in the sample entry");
        assert_eq!(validate_dfla(found?).ok(), Some((2_u16, 48_000_u32)));
        assert!(validate_mp4_pcm_entry(&entry).is_ok());
        Ok(())
    }

    #[test]
    fn avi_probe_accepts_only_supported_pcm_bit_depths() {
        let strl = |tag: u16, channels: u16, rate: u32, bits: u16| -> Vec<u8> {
            let mut out = b"strh".to_vec();
            out.extend_from_slice(&36_u32.to_le_bytes());
            out.extend_from_slice(b"auds");
            out.extend_from_slice(&[0; 32]);
            out.extend_from_slice(b"strf");
            out.extend_from_slice(&16_u32.to_le_bytes());
            out.extend_from_slice(&tag.to_le_bytes());
            out.extend_from_slice(&channels.to_le_bytes());
            out.extend_from_slice(&rate.to_le_bytes());
            out.extend_from_slice(&[0; 6]);
            out.extend_from_slice(&bits.to_le_bytes());
            out
        };
        for (tag, bits, accepted) in [
            (0x0001, 8, true),
            (0x0001, 16, true),
            (0x0001, 24, true),
            (0x0001, 32, true),
            (0x0001, 20, false),
            (0x0003, 32, true),
            (0x0003, 64, true),
            (0x0003, 16, false),
        ] {
            let result = avi_stream_info(&strl(tag, 2, 48_000, bits), &AvihInfo::default());
            assert_eq!(
                result.as_ref().is_ok_and(|info| info
                    .as_ref()
                    .is_some_and(|info| info.codec() == VideoCodec::Pcm)),
                accepted,
                "AVI tag {tag:#06x} at {bits} bits"
            );
        }
        assert!(avi_stream_info(&strl(0x0001, 0, 48_000, 16), &AvihInfo::default()).is_err());
        assert!(avi_stream_info(&strl(0x0001, 2, 0, 16), &AvihInfo::default()).is_err());
    }

    #[test]
    fn probe_accepts_and_rejects_matroska_pcm_bit_depths() -> Result<(), Box<dyn std::error::Error>>
    {
        for (codec_id, bits, accepted) in [
            ("A_PCM/INT/LIT", 8, true),
            ("A_PCM/INT/LIT", 16, true),
            ("A_PCM/INT/LIT", 24, true),
            ("A_PCM/INT/LIT", 32, true),
            ("A_PCM/INT/LIT", 20, false),
            ("A_PCM/FLOAT/IEEE", 32, true),
            ("A_PCM/FLOAT/IEEE", 64, true),
            ("A_PCM/FLOAT/IEEE", 16, false),
        ] {
            let audio = AudioFacts {
                channels: Some(2),
                sampling_frequency: Some(48_000.0),
                bit_depth: Some(bits),
            };
            let bytes = mkv_with_audio_codec(codec_id.as_bytes(), &[], Some(audio));
            let path = temp_file("mkv-pcm", &bytes)?;
            let result = probe_video(&path);
            assert_eq!(
                result.is_ok(),
                accepted,
                "{codec_id} at {bits} bits: {result:?}"
            );
            if let Ok(probed) = result {
                assert!(probed.streams.iter().any(|s| s.codec() == VideoCodec::Pcm));
            }
            fs::remove_file(&path)?;
        }
        Ok(())
    }
}
