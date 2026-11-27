use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::num::{NonZeroU16, NonZeroU32, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::thread;

use fastconv_domain::{
    AudioBatchItem, AudioBatchPlan, AudioBitDepth, AudioEncodingOptions, AudioExecutionMode,
    AudioInputFormat, AudioOptionBitrate, AudioOptionChannels, AudioOptionSampleRate,
    AudioOutputFormat, AudioQuality, AudioRateControl, AudioSampleFormat, AudioStreamInfo,
    AudioTargetCodec, DomainError, HardwareBackend, PathRole, PcmWavLayout, fnv1a64,
};
use symphonia::core::audio::sample::SampleFormat as SymphoniaSampleFormat;
use symphonia::core::codecs::audio::{
    AudioDecoderOptions,
    well_known::{
        CODEC_ID_ALAC, CODEC_ID_PCM_F32BE, CODEC_ID_PCM_F32LE, CODEC_ID_PCM_F64BE,
        CODEC_ID_PCM_F64LE,
    },
};
use symphonia::core::formats::FormatOptions;
use symphonia::core::formats::TrackType;
use symphonia::core::formats::probe::Hint;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

use crate::ProbeError;

const MAGIC_PREFIX_BYTES: usize = 4096;
const SUPPORTED_AUDIO_TARGETS: &str = "flac, wav, opus, ogg, mp3, aac, m4a, alac, or aiff";
const CODEC_PARSE_EXPECTED: &str = "vorbis, opus, flac, aac, alac, pcm, or mp3";

/// Raw encoder knobs collected at the CLI boundary before the output format is known.
///
/// Field-level validity is established by the domain constructors; compatibility with the
/// resolved output format and codec is checked once in [`prepare_audio_batch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AudioEncodingRequest {
    pub codec: Option<AudioTargetCodec>,
    pub rate_control: AudioRateControl,
    pub sample_rate: Option<AudioOptionSampleRate>,
    pub channels: Option<AudioOptionChannels>,
    pub bit_depth: Option<AudioBitDepth>,
}

const BITRATE_PARSE_EXPECTED: &str =
    "a positive rate such as 128k, 192k, 320k, 128kbps, 128, or 128000";
const BIT_DEPTH_PARSE_EXPECTED: &str = "8, 16, 24, 32, or f32";

/// Parses the raw `--bitrate` value at the system boundary into a validated domain rate.
///
/// A `k`/`kbps` suffix (case-insensitive) means kbps, a bare value below `1_000` is kbps,
/// and a bare value of at least `1_000` is bps.
///
/// # Errors
///
/// Returns [`DomainError::InvalidAudioOptionBitrate`] for empty, zero, non-digit, overflowing,
/// or otherwise malformed input.
pub fn parse_audio_bitrate(raw: &str) -> Result<AudioOptionBitrate, DomainError> {
    let invalid = || DomainError::InvalidAudioOptionBitrate {
        raw: raw.to_owned(),
        expected: BITRATE_PARSE_EXPECTED,
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(invalid());
    }
    let lowered = trimmed.to_ascii_lowercase();
    let (digits, suffix_kbps) = if let Some(rest) = lowered.strip_suffix("kbps") {
        (rest, true)
    } else if let Some(rest) = lowered.strip_suffix('k') {
        (rest, true)
    } else {
        (trimmed, false)
    };
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid());
    }
    let value = digits.parse::<u32>().map_err(|_| invalid())?;
    if value == 0 {
        return Err(invalid());
    }
    let bps = if suffix_kbps || value < 1_000 {
        value.checked_mul(1_000).ok_or_else(invalid)?
    } else {
        value
    };
    AudioOptionBitrate::new(bps)
}

/// Parses the raw `--bit-depth` value at the system boundary.
///
/// Signed integer aliases (`s8`/`s16`/`s24`/`s32`) and the spelling variants
/// `float`/`float32` are accepted case-insensitively; 8-bit WAV input is unsigned,
/// so `u8` is accepted as an alias for the 8-bit integer depth as well.
///
/// # Errors
///
/// Returns [`DomainError::InvalidAudioBitDepth`] for any unrecognized value.
pub fn parse_audio_bit_depth(raw: &str) -> Result<AudioBitDepth, DomainError> {
    let trimmed = raw.trim();
    match trimmed.to_ascii_lowercase().as_str() {
        "8" | "s8" | "u8" => Ok(AudioBitDepth::Int8),
        "16" | "s16" => Ok(AudioBitDepth::Int16),
        "24" | "s24" => Ok(AudioBitDepth::Int24),
        "32" | "s32" => Ok(AudioBitDepth::Int32),
        "f32" | "float" | "float32" => Ok(AudioBitDepth::Float32),
        _ => Err(DomainError::InvalidAudioBitDepth {
            raw: raw.to_owned(),
            expected: BIT_DEPTH_PARSE_EXPECTED,
        }),
    }
}

/// Parses the raw `--codec` value at the system boundary.
///
/// Values are case-insensitive and surrounding whitespace is ignored.
///
/// # Errors
///
/// Returns [`DomainError::InvalidAudioCodec`] for any unrecognized value.
pub fn parse_audio_codec(raw: &str) -> Result<AudioTargetCodec, DomainError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "vorbis" => Ok(AudioTargetCodec::Vorbis),
        "opus" => Ok(AudioTargetCodec::Opus),
        "flac" => Ok(AudioTargetCodec::Flac),
        "aac" => Ok(AudioTargetCodec::Aac),
        "alac" => Ok(AudioTargetCodec::Alac),
        "pcm" => Ok(AudioTargetCodec::Pcm),
        "mp3" => Ok(AudioTargetCodec::Mp3),
        _ => Err(DomainError::InvalidAudioCodec {
            raw: raw.to_owned(),
            expected: CODEC_PARSE_EXPECTED,
        }),
    }
}

/// Parses and validates the raw encoding flag values into an [`AudioEncodingRequest`].
///
/// This is the single boundary where the CLI's raw `--bitrate`, `--quality`,
/// `--sample-rate`, `--channels`, and `--bit-depth` strings become domain types;
/// the CLI only collects raw values and flag presence.
///
/// # Errors
///
/// Returns a diagnostic for a bitrate/quality conflict or any malformed flag value.
pub fn parse_audio_encoding_request(
    codec: Option<&str>,
    bitrate: Option<&str>,
    quality: Option<&str>,
    sample_rate: Option<&str>,
    channels: Option<&str>,
    bit_depth: Option<&str>,
) -> Result<AudioEncodingRequest, String> {
    let codec = codec
        .map(parse_audio_codec)
        .transpose()
        .map_err(|err| err.to_string())?;
    if bitrate.is_some() && quality.is_some() {
        return Err("cannot combine --bitrate with --quality".to_owned());
    }
    let rate_control = if let Some(raw) = bitrate {
        AudioRateControl::Bitrate(parse_audio_bitrate(raw).map_err(|err| err.to_string())?)
    } else if let Some(raw) = quality {
        let value = raw
            .parse::<u8>()
            .map_err(|err| format!("invalid --quality '{raw}': {err}"))?;
        AudioRateControl::Quality(
            AudioQuality::new(value).map_err(|err| format!("invalid --quality '{raw}': {err}"))?,
        )
    } else {
        AudioRateControl::Default
    };
    let sample_rate = match sample_rate {
        Some(raw) => {
            let hz = raw
                .parse::<u32>()
                .map_err(|err| format!("invalid --sample-rate '{raw}': {err}"))?;
            Some(
                AudioOptionSampleRate::new(hz)
                    .map_err(|err| format!("invalid --sample-rate '{raw}': {err}"))?,
            )
        }
        None => None,
    };
    let channels = match channels {
        Some(raw) => {
            let count = raw
                .parse::<u16>()
                .map_err(|err| format!("invalid --channels '{raw}': {err}"))?;
            Some(
                AudioOptionChannels::new(count)
                    .map_err(|err| format!("invalid --channels '{raw}': {err}"))?,
            )
        }
        None => None,
    };
    let bit_depth = bit_depth
        .map(parse_audio_bit_depth)
        .transpose()
        .map_err(|err| err.to_string())?;
    Ok(AudioEncodingRequest {
        codec,
        rate_control,
        sample_rate,
        channels,
        bit_depth,
    })
}

/// Validated input metadata produced at the file-system boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbedAudio {
    pub path: PathBuf,
    pub format: AudioInputFormat,
    pub stream: AudioStreamInfo,
}

fn invalid_audio(path: &Path, reason: impl Into<String>) -> ProbeError {
    ProbeError::InvalidAudio {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

/// Returns the byte length of a leading `ID3v2` tag (header + payload + optional footer), if valid.
fn id3v2_tag_len(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 10 || !bytes.starts_with(b"ID3") {
        return None;
    }
    if bytes[3] == 0xFF || bytes[4] == 0xFF {
        return None;
    }
    let size_bytes = &bytes[6..10];
    if size_bytes.iter().any(|&b| b & 0x80 != 0) {
        return None;
    }
    let payload = (usize::from(size_bytes[0]) << 21)
        | (usize::from(size_bytes[1]) << 14)
        | (usize::from(size_bytes[2]) << 7)
        | usize::from(size_bytes[3]);
    let footer = if bytes[5] & 0x10 != 0 { 10 } else { 0 };
    10_usize.checked_add(payload)?.checked_add(footer)
}

/// True when `bytes` begins with a valid ADTS AAC frame header (syncword 0xFFF, layer 00).
fn is_adts_sync(bytes: &[u8]) -> bool {
    if bytes.len() < 7 || bytes[0] != 0xFF || (bytes[1] & 0xF6) != 0xF0 {
        return false;
    }
    let sf_idx = (bytes[2] >> 2) & 0x0F;
    if sf_idx >= 13 {
        return false;
    }
    let frame_len = ((usize::from(bytes[3] & 0x03)) << 11)
        | (usize::from(bytes[4]) << 3)
        | (usize::from(bytes[5] >> 5));
    frame_len >= 7
}

/// True when `bytes` begins with a valid MPEG Layer I/II/III frame sync (layer != 00).
fn is_mp3_sync(bytes: &[u8]) -> bool {
    bytes.len() >= 2
        && bytes[0] == 0xFF
        && (bytes[1] & 0xE0) == 0xE0
        && (bytes[1] & 0x18) != 0x08
        && (bytes[1] & 0x06) != 0x00
}

/// Detects the container from leading bytes only; file extensions are never trusted.
#[must_use]
pub fn detect_audio_magic(bytes: &[u8]) -> Option<AudioInputFormat> {
    if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return Some(AudioInputFormat::Wav);
    }
    if bytes.len() >= 12
        && &bytes[..4] == b"FORM"
        && (&bytes[8..12] == b"AIFF" || &bytes[8..12] == b"AIFC")
    {
        return Some(AudioInputFormat::Aiff);
    }
    if bytes.starts_with(b"fLaC") {
        return Some(AudioInputFormat::Flac);
    }
    if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
        let box_size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if box_size == 0 || box_size >= 12 {
            return Some(AudioInputFormat::M4a);
        }
    }
    if bytes.starts_with(b"OggS")
        && let Some((identification, after_skeleton)) = ogg_identification_packet(bytes)
    {
        if identification.starts_with(b"OpusHead") {
            // `opus-pure` reads from offset zero and requires OpusHead as the
            // first packet; claiming a Skeleton-prefixed file would create a
            // plan that can never be opened.
            return (!after_skeleton).then_some(AudioInputFormat::Opus);
        }
        if identification.starts_with(b"\x01vorbis") {
            return Some(AudioInputFormat::Vorbis);
        }
        if identification.starts_with(b"\x7fFLAC") {
            return Some(AudioInputFormat::OggFlac);
        }
    }
    if let Some(tag_len) = id3v2_tag_len(bytes) {
        if let Some(after_tag) = bytes.get(tag_len..)
            && is_adts_sync(after_tag)
        {
            return Some(AudioInputFormat::Aac);
        }
        return Some(AudioInputFormat::Mp3);
    }
    if is_adts_sync(bytes) {
        return Some(AudioInputFormat::Aac);
    }
    if is_mp3_sync(bytes) {
        return Some(AudioInputFormat::Mp3);
    }
    None
}

/// Extracts the first packet of the first non-Skeleton Ogg page within the leading prefix.
///
/// Ogg Skeleton streams prefix the audio stream with `fishead`/`fisbone` pages, so reading
/// only the first page would miss the codec identification packet. The codec must come from
/// a framed identification packet; scanning arbitrary bytes can mistake Vorbis comments for
/// Opus. The BOS flag is deliberately ignored so a lone identification page still works.
/// Returns the first audio identification packet and whether a Skeleton page
/// preceded it. The flag matters for Opus: `opus-pure` requires `OpusHead` to be
/// the first packet of the file, so a Skeleton-prefixed Opus stream must not be
/// detected as openable.
fn ogg_identification_packet(bytes: &[u8]) -> Option<(&[u8], bool)> {
    const TABLE_START: usize = 27;
    let limit = bytes.len().min(MAGIC_PREFIX_BYTES);
    let mut offset = 0_usize;
    let mut after_skeleton = false;
    while offset < limit {
        let mut page = bytes.get(offset..limit)?;
        if !page.starts_with(b"OggS") || page.len() < TABLE_START {
            return None;
        }
        let segments = usize::from(page[26]);
        let table_end = TABLE_START.checked_add(segments)?;
        let table = page.get(TABLE_START..table_end)?;
        let mut body_len = 0_usize;
        let mut first_packet_len = None;
        for &lacing in table {
            body_len = body_len.checked_add(usize::from(lacing))?;
            if first_packet_len.is_none() && lacing < 255 {
                first_packet_len = Some(body_len);
            }
        }
        // The page body is the sum of *all* lacing values, not just the first packet's.
        let page_len = table_end.checked_add(body_len)?;
        page = page.get(..page_len)?;
        if let Some(packet_len) = first_packet_len {
            let packet = page.get(table_end..table_end.checked_add(packet_len)?)?;
            if !packet.starts_with(b"fishead") && !packet.starts_with(b"fisbone") {
                return Some((packet, after_skeleton));
            }
            after_skeleton = true;
        }
        offset = offset.checked_add(page_len)?;
    }
    None
}

fn read_magic_prefix(file: &mut File, path: &Path) -> Result<[u8; MAGIC_PREFIX_BYTES], ProbeError> {
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
    Ok(prefix)
}

/// At most this many trailing bytes are inspected when recovering an Ogg
/// granule position. A longer tail only adds work and never improves certainty.
const OGG_TAIL_SCAN_BYTES: u64 = 65_536;
const OGG_PAGE_HEADER_FIXED: usize = 27;
/// Bounded prefix scanned for the OpusHead/OpusTags pages and the first audio page.
const OGG_HEADER_SCAN_BYTES: u64 = 256 * 1024;
const MAX_OGG_HEADER_PAGES: usize = 8;
/// Ogg's non-reflected CRC-32 (RFC 3533 §6, polynomial `0x04c11db7`, zero seed,
/// no final inversion). A page is checksummed with its own CRC field zeroed.
const OGG_CRC_POLY: u32 = 0x04c1_1db7;
// `i` is bounded to 0..=255 by the loop below, so narrowing the table index to
// `u32` is lossless; `TryFrom` is not usable in a const context on this toolchain.
#[allow(clippy::cast_possible_truncation)]
const OGG_CRC_TABLE: [u32; 256] = {
    let mut table = [0_u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut value = (i as u32) << 24;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 0x8000_0000 != 0 {
                (value << 1) ^ OGG_CRC_POLY
            } else {
                value << 1
            };
            bit += 1;
        }
        table[i] = value;
        i += 1;
    }
    table
};

fn ogg_crc32(data: &[u8]) -> u32 {
    let mut crc = 0_u32;
    for &byte in data {
        let index = ((crc >> 24) as u8 ^ byte) as usize;
        crc = (crc << 8) ^ OGG_CRC_TABLE[index];
    }
    crc
}

/// Recovers the decoded sample count of an Ogg Opus stream from its final page.
///
/// The granule position of the last page includes the pre-skip samples and any
/// positive initial granule offset allowed by RFC 7845 §4.5, so the decoded
/// length is `granule - initial_offset - pre_skip`. A candidate is accepted only
/// when it is a version-0 page whose lacing table and body end exactly at EOF,
/// whose serial matches the one locked by `OggOpusReader`, and whose Ogg CRC-32
/// verifies, and whose granule equals the one reached by `OggOpusReader`.
/// Anything else yields `None` so a crafted suffix page cannot inject a granule
/// into the batch memory reservation.
fn opus_total_samples(
    file: &mut File,
    pre_skip: u64,
    initial_offset: u64,
    expected_serial: u32,
    expected_granule: u64,
) -> Option<u64> {
    let len = file.metadata().ok()?.len();
    let start = len.saturating_sub(OGG_TAIL_SCAN_BYTES);
    file.seek(SeekFrom::Start(start)).ok()?;
    let mut tail = Vec::new();
    file.take(OGG_TAIL_SCAN_BYTES).read_to_end(&mut tail).ok()?;
    if tail.len() < OGG_PAGE_HEADER_FIXED + 1 {
        return None;
    }
    for at in (0..=tail.len() - OGG_PAGE_HEADER_FIXED).rev() {
        if &tail[at..at + 4] != b"OggS" || tail.get(at + 4) != Some(&0) {
            continue;
        }
        let Some(&seg_byte) = tail.get(at + 26) else {
            continue;
        };
        let segments = usize::from(seg_byte);
        if segments == 0 {
            continue;
        }
        let Some(table_end) = at
            .checked_add(OGG_PAGE_HEADER_FIXED)
            .and_then(|end| end.checked_add(segments))
        else {
            continue;
        };
        let Some(table) = tail.get(at + OGG_PAGE_HEADER_FIXED..table_end) else {
            continue;
        };
        let Some(body_len) = table.iter().try_fold(0_usize, |total, &lacing| {
            total.checked_add(usize::from(lacing))
        }) else {
            continue;
        };
        if table_end.checked_add(body_len) != Some(tail.len()) {
            continue;
        }
        let Some(granule_bytes) = tail
            .get(at + 6..at + 14)
            .and_then(|slice| <[u8; 8]>::try_from(slice).ok())
        else {
            continue;
        };
        let granule = i64::from_le_bytes(granule_bytes);
        if granule < 0 {
            continue;
        }
        let Ok(granule) = u64::try_from(granule) else {
            continue;
        };
        let serial_bytes = <[u8; 4]>::try_from(tail.get(at + 14..at + 18)?).ok()?;
        if u32::from_le_bytes(serial_bytes) != expected_serial {
            continue;
        }
        // The CRC covers the whole page with the CRC field itself zeroed. The
        // final candidate extends to EOF, so it is contiguous in `tail`.
        let crc_bytes = <[u8; 4]>::try_from(tail.get(at + 22..at + 26)?).ok()?;
        let mut checksummed = Vec::with_capacity(tail.len() - at);
        checksummed.extend_from_slice(&tail[at..at + 22]);
        checksummed.extend_from_slice(&[0_u8; 4]);
        checksummed.extend_from_slice(&tail[at + 26..]);
        if ogg_crc32(&checksummed) != u32::from_le_bytes(crc_bytes) {
            continue;
        }
        // A CRC-valid page with the right serial can still be appended after
        // the logical EOS. The reader stops at that EOS, so only its terminal
        // granule is trustworthy; a different candidate is rejected.
        if granule != expected_granule {
            continue;
        }
        return granule.checked_sub(initial_offset)?.checked_sub(pre_skip);
    }
    None
}

/// One CRC-validated Ogg page, with the packets that COMPLETE on it (a lacing
/// value below 255 terminates a packet).
struct OggPageInfo<'a> {
    granule: i64,
    serial: u32,
    header_type: u8,
    end: usize,
    completed: Vec<&'a [u8]>,
}

fn parse_ogg_page(bytes: &[u8], at: usize) -> Option<OggPageInfo<'_>> {
    if bytes.get(at..at + 4)? != b"OggS" || bytes.get(at + 4) != Some(&0) {
        return None;
    }
    let granule = i64::from_le_bytes(<[u8; 8]>::try_from(bytes.get(at + 6..at + 14)?).ok()?);
    let serial = u32::from_le_bytes(<[u8; 4]>::try_from(bytes.get(at + 14..at + 18)?).ok()?);
    let header_type = *bytes.get(at + 5)?;
    let segments = usize::from(*bytes.get(at + 26)?);
    if segments == 0 {
        return None;
    }
    let table_end = at
        .checked_add(OGG_PAGE_HEADER_FIXED)?
        .checked_add(segments)?;
    let table = bytes.get(at + OGG_PAGE_HEADER_FIXED..table_end)?;
    let body_len = table
        .iter()
        .try_fold(0_usize, |total, &lace| total.checked_add(usize::from(lace)))?;
    let end = table_end.checked_add(body_len)?;
    let body = bytes.get(table_end..end)?;
    let stored_crc = u32::from_le_bytes(<[u8; 4]>::try_from(bytes.get(at + 22..at + 26)?).ok()?);
    let mut checksummed = Vec::with_capacity(end - at);
    checksummed.extend_from_slice(&bytes[at..at + 22]);
    checksummed.extend_from_slice(&[0_u8; 4]);
    checksummed.extend_from_slice(&bytes[at + 26..end]);
    if ogg_crc32(&checksummed) != stored_crc {
        return None;
    }
    let mut completed = Vec::new();
    let mut start = 0_usize;
    for &lace in table {
        let next = start.checked_add(usize::from(lace))?;
        if next > body.len() {
            return None;
        }
        if lace < 255 {
            completed.push(&body[start..next]);
            start = next;
        }
    }
    if start != body.len() {
        // A packet still spanning into the next page is allowed in general,
        // but not on a header page where this parser is used.
        return None;
    }
    Some(OggPageInfo {
        granule,
        serial,
        header_type,
        end,
        completed,
    })
}

/// Derives the timeline offset from the FIRST audio data page by walking Ogg
/// pages from the start of the file. RFC 7845 §4.5 permits a positive initial
/// granule position; RFC 7845 §4.4 permits a later page to carry the same
/// granule, so page membership must come from the page layout, not from the
/// granule value reported on a packet.
fn opus_initial_offset(path: &Path) -> Option<u64> {
    let file = File::open(path).ok()?;
    let mut bytes = Vec::new();
    file.take(OGG_HEADER_SCAN_BYTES)
        .read_to_end(&mut bytes)
        .ok()?;
    let mut at = 0_usize;
    let mut pages = 0_usize;
    let mut serial = None;
    while at < bytes.len() {
        if pages >= MAX_OGG_HEADER_PAGES {
            return None;
        }
        let page = parse_ogg_page(&bytes, at)?;
        if serial.is_some_and(|expected| expected != page.serial) {
            return None;
        }
        serial = Some(page.serial);
        match pages {
            0 => {
                // Identification header: begins the stream with granule zero.
                if page.header_type & 0x02 == 0
                    || page.granule != 0
                    || page.completed.len() != 1
                    || !page.completed[0].starts_with(b"OpusHead")
                {
                    return None;
                }
            }
            1 => {
                // Comment header must finish its own page.
                if page.header_type & 0x02 != 0
                    || page.granule != 0
                    || page.completed.len() != 1
                    || !page.completed[0].starts_with(b"OpusTags")
                {
                    return None;
                }
            }
            _ => {
                // First audio page: sum only the packets completing here.
                if page.header_type & 0x01 != 0 || page.granule < 0 {
                    return None;
                }
                let decoded = |data: &[u8]| -> Option<u64> {
                    u64::try_from(opus_pure::packet::samples(data, 48_000).ok()?).ok()
                };
                let sum = page
                    .completed
                    .iter()
                    .try_fold(0_u64, |total, packet| total.checked_add(decoded(packet)?))?;
                return u64::try_from(page.granule).ok()?.checked_sub(sum);
            }
        }
        at = page.end;
        pages += 1;
    }
    None
}

/// Granule of the terminal packet actually reached by `OggOpusReader`. It stops
/// at the logical EOS and ignores any CRC-valid page appended after it.
fn opus_terminal_granule(reader: &mut opus_pure::OggOpusReader<File>) -> Option<u64> {
    let mut last = None;
    for packet in reader.packets() {
        last = Some(packet.ok()?.page_granule);
    }
    let granule = last?;
    if granule < 0 {
        return None;
    }
    u64::try_from(granule).ok()
}

fn probe_opus(path: &Path, file: File) -> Result<AudioStreamInfo, ProbeError> {
    let mut probe_file = file.try_clone().map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = opus_pure::OggOpusReader::new(file)
        .map_err(|err| invalid_audio(path, format!("invalid Ogg Opus stream: {err}")))?;
    let head = reader.head();
    let channels = NonZeroU16::new(u16::from(head.channel_count))
        .ok_or_else(|| invalid_audio(path, "Opus channel count must not be zero"))?;
    let pre_skip = u64::from(head.pre_skip);
    let serial = reader.serial();
    let terminal_granule = opus_terminal_granule(&mut reader);
    // A nonzero initial granule shifts the converter's Trim timeline, so the
    // probed decoded length cannot be trusted; withhold it and keep the
    // conservative unknown-length reservation.
    let initial_offset = opus_initial_offset(path);
    let total_samples = match (terminal_granule, initial_offset) {
        (Some(final_granule), Some(offset)) if offset == 0 => {
            opus_total_samples(&mut probe_file, pre_skip, offset, serial, final_granule)
        }
        _ => None,
    };
    // Ogg Opus always decodes at 48 kHz, regardless of the rate recorded at capture time.
    let sample_rate = NonZeroU32::new(48_000).unwrap_or(NonZeroU32::MIN);
    AudioStreamInfo::new(channels, sample_rate, NonZeroU16::new(16), total_samples)
        .map_err(|err| invalid_audio(path, err.to_string()))
}

fn probe_adts_aac(path: &Path, mut file: File) -> Result<AudioStreamInfo, ProbeError> {
    let prefix = read_magic_prefix(&mut file, path)?;
    let offset = if prefix.starts_with(b"ID3") {
        let tag_len = id3v2_tag_len(&prefix)
            .ok_or_else(|| invalid_audio(path, "invalid ID3v2 tag before AAC frame"))?;
        u64::try_from(tag_len).map_err(|_| invalid_audio(path, "ID3v2 tag length out of range"))?
    } else {
        0
    };
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let mut header_buf = [0_u8; 7];
    file.read_exact(&mut header_buf)
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let header = rusty_aac::parse_adts(&header_buf)
        .map_err(|err| invalid_audio(path, format!("invalid ADTS AAC header: {err}")))?;
    if header.channels == 0 {
        return Err(invalid_audio(path, "AAC channel count must not be zero"));
    }
    if header.sample_rate == 0 {
        return Err(invalid_audio(path, "AAC sample rate must not be zero"));
    }
    if header.frame_length < header.header_len {
        return Err(invalid_audio(
            path,
            "AAC frame length is shorter than its header",
        ));
    }
    let channels = NonZeroU16::new(header.channels)
        .ok_or_else(|| invalid_audio(path, "AAC channel count must not be zero"))?;
    let sample_rate = NonZeroU32::new(header.sample_rate)
        .ok_or_else(|| invalid_audio(path, "AAC sample rate must not be zero"))?;
    // An ADTS header cannot signal SBR. HE-AAC dual-rate decoding produces twice
    // the PCM of the core-rate frame count this header describes, and the raw
    // blocks do not reveal whether the stream is SBR-capable. Do not report a
    // count we cannot prove: the batch keeps its conservative unknown-length
    // reservation instead.
    let total_samples = None;
    AudioStreamInfo::new(channels, sample_rate, None, total_samples)
        .map_err(|err| invalid_audio(path, err.to_string()))
}

/// Upper bound on the WAV header bytes hashed into the layout fingerprint.
/// A larger header (many junk chunks) withholds the fast path rather than
/// allocating an unbounded buffer.
const MAX_WAV_HEADER_FINGERPRINT_BYTES: u64 = 1024 * 1024;

/// PCM `WAVEFORMATEXTENSIBLE` sub-format GUID for signed integer samples.
const WAVE_FORMAT_PCM_GUID: [u8; 16] = [
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

fn le_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn le_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Parses and validates a canonical signed-integer PCM RIFF/WAVE header exactly
/// once, at the system boundary. `None` means the file needs the generic decoder
/// (unsigned 8-bit, float, RF64, non-canonical extensible PCM, or a malformed
/// chunk). Every size is checked against the RIFF-declared end and the actual
/// file length, so no partial header can reach a downstream fast path.
fn probe_pcm_wav_layout(file: &mut File) -> Option<PcmWavLayout> {
    let file_len = file.metadata().ok()?.len();
    let mut head = [0_u8; 12];
    file.read_exact(&mut head).ok()?;
    if &head[..4] != b"RIFF" || &head[8..12] != b"WAVE" {
        return None;
    }
    if le_u32(&head, 4) == u32::MAX {
        return None;
    }
    let riff_end = u64::from(le_u32(&head, 4)).saturating_add(8).min(file_len);

    let mut pos = 12_u64;
    let mut fmt: Option<(usize, u32, usize, usize, u32)> = None;
    let mut data: Option<(u64, u64)> = None;
    let mut header = [0_u8; 8];
    let mut payload = vec![0_u8; 4096];

    while pos.saturating_add(8) <= riff_end {
        file.seek(SeekFrom::Start(pos)).ok()?;
        file.read_exact(&mut header).ok()?;
        let size = u64::from(le_u32(&header, 4));
        if size == u64::from(u32::MAX) || pos.saturating_add(8).saturating_add(size) > riff_end {
            return None;
        }
        let padded = size.saturating_add(size & 1);
        match &header[..4] {
            b"fmt " => {
                if !(16..=4096).contains(&size) {
                    return None;
                }
                let len = usize::try_from(size).ok()?;
                payload.resize(len, 0);
                file.read_exact(&mut payload[..len]).ok()?;
                let audio_format = le_u16(&payload, 0);
                let channels = usize::from(le_u16(&payload, 2));
                let rate = le_u32(&payload, 4);
                let block_align = usize::from(le_u16(&payload, 12));
                let bits = usize::from(le_u16(&payload, 14));
                let canonical = bits == 16 || bits == 24 || bits == 32;
                let extensible = audio_format == 0xFFFE
                    && size >= 40
                    && le_u16(&payload, 18) == u16::try_from(bits).unwrap_or(0)
                    && payload
                        .get(24..40)
                        .is_some_and(|guid| guid == WAVE_FORMAT_PCM_GUID.as_slice());
                if !(canonical && (audio_format == 1 || extensible))
                    || !(1..=8).contains(&channels)
                    || rate == 0
                    || block_align != channels.saturating_mul(bits / 8)
                {
                    return None;
                }
                let channel_mask = if audio_format == 0xFFFE {
                    <[u8; 4]>::try_from(payload.get(20..24)?).ok()?
                } else {
                    [0; 4]
                };
                fmt = Some((
                    channels,
                    rate,
                    bits,
                    block_align,
                    u32::from_le_bytes(channel_mask),
                ));
            }
            b"data" => {
                data = Some((pos.saturating_add(8), size));
                if fmt.is_some() {
                    break;
                }
            }
            _ => {}
        }
        pos = pos.saturating_add(8).saturating_add(padded);
    }

    let (Some((channels, rate, bits, block_align, channel_mask)), Some((data_offset, data_bytes))) =
        (fmt, data)
    else {
        return None;
    };
    // Hash the exact prefix the pipeline will re-read before seeking to the
    // payload. This binds the saved offset to the header that produced it.
    if data_offset > MAX_WAV_HEADER_FINGERPRINT_BYTES {
        return None;
    }
    let header_len = usize::try_from(data_offset).ok()?;
    file.seek(SeekFrom::Start(0)).ok()?;
    let mut header_bytes = vec![0_u8; header_len];
    file.read_exact(&mut header_bytes).ok()?;
    let fingerprint = fnv1a64(&header_bytes);
    PcmWavLayout::new(
        NonZeroU16::new(u16::try_from(channels).ok()?)?,
        NonZeroU16::new(u16::try_from(bits).ok()?)?,
        NonZeroU32::new(rate)?,
        NonZeroU16::new(u16::try_from(block_align).ok()?)?,
        channel_mask,
        data_offset,
        data_bytes,
        fingerprint,
    )
    .ok()
}

fn probe_symphonia(
    path: &Path,
    format: AudioInputFormat,
    file: File,
) -> Result<(AudioInputFormat, AudioStreamInfo), ProbeError> {
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
    let mut hint = Hint::new();
    hint.with_extension(format.extension());
    let probed = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|err| invalid_audio(path, format!("failed to read audio headers: {err}")))?;
    let track = probed
        .default_track(TrackType::Audio)
        .ok_or_else(|| invalid_audio(path, "no audio track found"))?;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| invalid_audio(path, "audio track has no codec parameters"))?;
    symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(|err| invalid_audio(path, format!("unsupported audio codec: {err}")))?;
    let detected_format = if format == AudioInputFormat::M4a && params.codec == CODEC_ID_ALAC {
        AudioInputFormat::Alac
    } else {
        format
    };
    let channels = params
        .channels
        .as_ref()
        .and_then(|c| NonZeroU16::new(u16::try_from(c.count()).unwrap_or(u16::MAX)))
        .ok_or_else(|| invalid_audio(path, "audio channel count must not be zero"))?;
    let sample_rate = params
        .sample_rate
        .and_then(NonZeroU32::new)
        .ok_or_else(|| invalid_audio(path, "audio sample rate must not be zero"))?;
    // Symphonia 0.6.1's RIFF demuxer leaves both `bits_per_sample` and
    // `sample_format` unset for the plain IEEE-float WAV tag (0x0003) and for
    // AIFF-C `fl32`; the codec id is the only reliable float signal there.
    let float_codec = matches!(
        params.codec,
        CODEC_ID_PCM_F32LE | CODEC_ID_PCM_F32BE | CODEC_ID_PCM_F64LE | CODEC_ID_PCM_F64BE
    );
    let float_samples = matches!(
        params.sample_format,
        Some(SymphoniaSampleFormat::F32 | SymphoniaSampleFormat::F64)
    );
    let sample_format = if float_codec || float_samples {
        AudioSampleFormat::Float
    } else {
        AudioSampleFormat::Int
    };
    let inferred_bits = if matches!(params.codec, CODEC_ID_PCM_F32LE | CODEC_ID_PCM_F32BE) {
        Some(32)
    } else if matches!(params.codec, CODEC_ID_PCM_F64LE | CODEC_ID_PCM_F64BE) {
        Some(64)
    } else {
        match params.sample_format {
            Some(SymphoniaSampleFormat::F32) => Some(32),
            Some(SymphoniaSampleFormat::F64) => Some(64),
            _ => None,
        }
    };
    let bits_per_sample = params
        .bits_per_sample
        .or(inferred_bits)
        .and_then(|bits| NonZeroU16::new(u16::try_from(bits).unwrap_or(u16::MAX)));
    let stream = AudioStreamInfo::with_sample_format(
        channels,
        sample_rate,
        bits_per_sample,
        sample_format,
        track.num_frames,
    )
    .map_err(|err| invalid_audio(path, err.to_string()))?;
    Ok((detected_format, stream))
}

/// Probes one file: detects its container by magic bytes and validates stream metadata.
///
/// # Errors
///
/// Returns [`ProbeError`] for unreadable files, unrecognized magic bytes, or invalid metadata.
pub fn probe_audio(path: &Path) -> Result<ProbedAudio, ProbeError> {
    let metadata = fs::metadata(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(ProbeError::NotAFile(path.to_path_buf()));
    }
    if metadata.len() == 0 {
        return Err(invalid_audio(path, "audio file is empty"));
    }

    let mut file = File::open(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let prefix = read_magic_prefix(&mut file, path)?;
    file.seek(SeekFrom::Start(0))
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let mut format = detect_audio_magic(&prefix).ok_or_else(|| {
        invalid_audio(
            path,
            "unrecognized audio magic bytes (expected WAV, FLAC, MP3, Ogg Vorbis, Ogg Opus, Ogg FLAC, AAC, M4A, ALAC, or AIFF)",
        )
    })?;

    // Large ID3v2 tags (e.g. cover art) push the ADTS frame past the prefix, or leave
    // fewer than 7 header bytes in it so `detect_audio_magic` cannot see the syncword.
    if prefix.starts_with(b"ID3")
        && let Some(tag_len) = id3v2_tag_len(&prefix)
        && tag_len.saturating_add(7) > prefix.len()
    {
        let offset = u64::try_from(tag_len)
            .map_err(|_| invalid_audio(path, "ID3v2 tag length out of range"))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|source| ProbeError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        let mut after_tag = [0_u8; 16];
        let mut filled = 0_usize;
        while filled < after_tag.len() {
            match file.read(&mut after_tag[filled..]) {
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
        if is_adts_sync(&after_tag[..filled]) {
            format = AudioInputFormat::Aac;
        }
        file.seek(SeekFrom::Start(0))
            .map_err(|source| ProbeError::Io {
                path: path.to_path_buf(),
                source,
            })?;
    }

    let (format, stream) = match format {
        AudioInputFormat::Opus => (format, probe_opus(path, file)?),
        AudioInputFormat::Aac => (format, probe_adts_aac(path, file)?),
        AudioInputFormat::Wav => {
            // WAV headers are parsed exactly once. A canonical signed PCM file
            // yields both the validated layout and its stream metadata from the
            // same parse; anything else falls back to symphonia on the same
            // handle, which stays open and is rewound first.
            let mut wav_file = file;
            if let Some(layout) = probe_pcm_wav_layout(&mut wav_file) {
                let stream = AudioStreamInfo::new(
                    layout.channels(),
                    layout.sample_rate(),
                    Some(layout.bits_per_sample()),
                    Some(layout.total_samples()),
                )
                .map_err(|err| invalid_audio(path, err.to_string()))?;
                (format, stream.with_pcm_wav_layout(layout))
            } else {
                wav_file
                    .seek(SeekFrom::Start(0))
                    .map_err(|source| ProbeError::Io {
                        path: path.to_path_buf(),
                        source,
                    })?;
                probe_symphonia(path, format, wav_file)?
            }
        }
        _ => probe_symphonia(path, format, file)?,
    };

    Ok(ProbedAudio {
        path: path.to_path_buf(),
        format,
        stream,
    })
}

fn resolve_single_target(
    source: &Path,
    target: &Path,
    to: Option<AudioOutputFormat>,
) -> Result<(PathBuf, AudioOutputFormat), ProbeError> {
    let target_is_dir = target.is_dir()
        || target
            .to_str()
            .is_some_and(|s| s.ends_with('/') || s.ends_with('\\'));
    if target_is_dir {
        let format = to.unwrap_or(AudioOutputFormat::Flac);
        let stem = source
            .file_stem()
            .ok_or_else(|| invalid_audio(source, "source path has no file stem"))?;
        let mut name = stem.to_os_string();
        name.push(".");
        name.push(format.extension());
        return Ok((target.join(name), format));
    }

    let ext = target.extension().and_then(|e| e.to_str()).ok_or_else(|| {
        ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: SUPPORTED_AUDIO_TARGETS,
        }
    })?;
    let inferred = AudioOutputFormat::from_extension(ext).ok_or_else(|| {
        ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: SUPPORTED_AUDIO_TARGETS,
        }
    })?;
    if let Some(explicit) = to {
        if !explicit.accepts_extension(ext) {
            return Err(ProbeError::UnsupportedTargetFormat {
                path: target.to_path_buf(),
                expected: explicit.extension(),
            });
        }
        return Ok((target.to_path_buf(), explicit));
    }
    Ok((target.to_path_buf(), inferred))
}

fn prepare_directory_batch(
    source_dir: &Path,
    target_dir: &Path,
    mode: AudioExecutionMode,
    threads: NonZeroUsize,
    options: AudioEncodingOptions,
) -> Result<AudioBatchPlan, ProbeError> {
    let format = mode.output_format();
    match fs::metadata(target_dir) {
        Ok(meta) if meta.is_dir() => {}
        Ok(_) => return Err(ProbeError::TargetNotADirectory(target_dir.to_path_buf())),
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(ProbeError::Io {
                path: target_dir.to_path_buf(),
                source,
            });
        }
    }

    if canonical_or_parent(source_dir)? == canonical_or_parent(target_dir)? {
        return Err(DomainError::IdenticalSourceAndTarget(source_dir.to_path_buf()).into());
    }

    let mut candidates = Vec::new();
    for entry in fs::read_dir(source_dir).map_err(|source| ProbeError::Io {
        path: source_dir.to_path_buf(),
        source,
    })? {
        let path = entry
            .map_err(|source| ProbeError::Io {
                path: source_dir.to_path_buf(),
                source,
            })?
            .path();
        let is_hidden = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with('.'));
        let has_audio_extension = path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
            matches!(
                e.to_ascii_lowercase().as_str(),
                "wav"
                    | "wave"
                    | "flac"
                    | "mp3"
                    | "ogg"
                    | "oga"
                    | "opus"
                    | "aac"
                    | "m4a"
                    | "alac"
                    | "aiff"
                    | "aif"
                    | "aifc"
            )
        });
        if path.is_file() && !is_hidden && has_audio_extension {
            candidates.push(path);
        }
    }
    candidates.sort();

    let mut items = Vec::with_capacity(candidates.len());
    for path in candidates {
        let probed = probe_audio(&path)?;
        let stem = path
            .file_stem()
            .ok_or_else(|| invalid_audio(&path, "source path has no file stem"))?;
        let mut name = stem.to_os_string();
        name.push(".");
        name.push(format.extension());
        items.push(AudioBatchItem::new(
            canonical_or_parent(&path)?,
            target_dir.join(name),
            probed.format,
            probed.stream,
        )?);
    }
    if items.is_empty() {
        return Err(ProbeError::EmptyAudioDirectory(source_dir.to_path_buf()));
    }

    Ok(AudioBatchPlan::with_options(items, mode, threads, options)?)
}

/// Resolves a path to an absolute location, using the parent directory for outputs that do not
/// exist yet. This makes relative/absolute aliases comparable at the boundary.
fn canonical_or_parent(path: &Path) -> Result<PathBuf, ProbeError> {
    let mut cursor = path.to_path_buf();
    let mut missing = std::collections::VecDeque::new();
    loop {
        match fs::canonicalize(&cursor) {
            Ok(mut canonical) => {
                for component in missing {
                    canonical = canonical.join(component);
                }
                return Ok(canonical);
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(ProbeError::Io {
                    path: cursor,
                    source,
                });
            }
        }
        match cursor.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => {
                if let Some(name) = cursor.file_name() {
                    missing.push_front(name.to_os_string());
                }
                cursor = parent.to_path_buf();
            }
            _ => {
                let mut canonical = fs::canonicalize(".").map_err(|source| ProbeError::Io {
                    path: PathBuf::from("."),
                    source,
                })?;
                if let Some(name) = cursor.file_name() {
                    missing.push_front(name.to_os_string());
                }
                for component in missing {
                    canonical = canonical.join(component);
                }
                return Ok(canonical);
            }
        }
    }
}

/// Parses external paths and options once, then returns a domain [`AudioBatchPlan`].
///
/// # Errors
///
/// Returns [`ProbeError`] for invalid paths, unsupported formats/backends, unreadable inputs,
/// or conflicting batch targets. GPU backends are rejected explicitly without CPU fallback.
pub fn prepare_audio_batch(
    source: &Path,
    target: &Path,
    to: Option<AudioOutputFormat>,
    worker_threads: Option<NonZeroUsize>,
    backend: HardwareBackend,
    request: AudioEncodingRequest,
) -> Result<AudioBatchPlan, ProbeError> {
    if source.as_os_str().is_empty() {
        return Err(DomainError::EmptyPath(PathRole::Source).into());
    }
    if target.as_os_str().is_empty() {
        return Err(DomainError::EmptyPath(PathRole::Target).into());
    }

    let threads = match worker_threads {
        Some(count) => count,
        None => thread::available_parallelism().map_err(ProbeError::ParallelismUnavailable)?,
    };
    let source_meta = fs::metadata(source).map_err(|err| ProbeError::Io {
        path: source.to_path_buf(),
        source: err,
    })?;

    if source_meta.is_dir() {
        let format = to.unwrap_or(AudioOutputFormat::Flac);
        let mode = AudioExecutionMode::resolve_with_codec(format, request.codec, backend)?;
        // Validate encoding options before touching any input file in the batch.
        let options = AudioEncodingOptions::for_codec(
            format,
            mode.codec(),
            request.rate_control,
            request.sample_rate,
            request.channels,
            request.bit_depth,
        )?;
        return prepare_directory_batch(source, target, mode, threads, options);
    }
    if !source_meta.is_file() {
        return Err(ProbeError::NotAFileOrDirectory(source.to_path_buf()));
    }

    let (resolved_target, format) = resolve_single_target(source, target, to)?;
    let mode = AudioExecutionMode::resolve_with_codec(format, request.codec, backend)?;
    let options = AudioEncodingOptions::for_codec(
        format,
        mode.codec(),
        request.rate_control,
        request.sample_rate,
        request.channels,
        request.bit_depth,
    )?;
    let canonical_source = canonical_or_parent(source)?;
    let canonical_target = canonical_or_parent(&resolved_target)?;
    if canonical_source == canonical_target {
        return Err(DomainError::IdenticalSourceAndTarget(source.to_path_buf()).into());
    }
    let probed = probe_audio(source)?;
    let item = AudioBatchItem::new(
        canonical_source,
        resolved_target,
        probed.format,
        probed.stream,
    )?;
    Ok(AudioBatchPlan::with_options(
        vec![item],
        mode,
        threads,
        options,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastconv_domain::AudioOptionBitrate;

    #[test]
    fn detects_magic_without_extension_help() {
        let mut wav = vec![0_u8; 12];
        wav[..4].copy_from_slice(b"RIFF");
        wav[8..12].copy_from_slice(b"WAVE");
        assert_eq!(detect_audio_magic(&wav), Some(AudioInputFormat::Wav));
        assert_eq!(
            detect_audio_magic(b"fLaC\x00\x00\x00\x22"),
            Some(AudioInputFormat::Flac)
        );
        assert_eq!(
            detect_audio_magic(b"ID3\x04\x00\x00\x00\x00\x00\x00"),
            Some(AudioInputFormat::Mp3)
        );
        assert_eq!(
            detect_audio_magic(&[0xFF, 0xFB, 0x90, 0x64]),
            Some(AudioInputFormat::Mp3)
        );
        let mut vorbis_page = b"OggS".to_vec();
        vorbis_page.extend_from_slice(&[0_u8; 22]);
        vorbis_page.extend_from_slice(&[1, 7]);
        vorbis_page.extend_from_slice(b"\x01vorbis");
        assert_eq!(
            detect_audio_magic(&vorbis_page),
            Some(AudioInputFormat::Vorbis)
        );
        let mut opus_page = b"OggS".to_vec();
        opus_page.extend_from_slice(&[0_u8; 22]);
        opus_page.extend_from_slice(&[1, 8]);
        opus_page.extend_from_slice(b"OpusHead");
        assert_eq!(detect_audio_magic(&opus_page), Some(AudioInputFormat::Opus));
        assert_eq!(detect_audio_magic(b"UNKNOWN"), None);
    }

    #[test]
    fn identifies_ogg_codec_from_identification_packet() {
        let mut vorbis = b"OggS".to_vec();
        vorbis.extend_from_slice(&[0_u8; 22]);
        vorbis.extend_from_slice(&[2, 7, 20]);
        vorbis.extend_from_slice(b"\x01vorbis");
        let mut comment = b"TITLE=OpusHead".to_vec();
        comment.resize(20, 0);
        vorbis.extend_from_slice(&comment);
        assert_eq!(detect_audio_magic(&vorbis), Some(AudioInputFormat::Vorbis));

        let mut opus = b"OggS".to_vec();
        opus.extend_from_slice(&[0_u8; 22]);
        opus.extend_from_slice(&[1, 8]);
        opus.extend_from_slice(b"OpusHead");
        assert_eq!(detect_audio_magic(&opus), Some(AudioInputFormat::Opus));
    }

    #[test]
    fn resolves_single_target_extension_and_directory() {
        let source = Path::new("track.wav");
        let (file, format) = resolve_single_target(source, Path::new("out.flac"), None)
            .unwrap_or_else(|_| (PathBuf::new(), AudioOutputFormat::Wav));
        assert_eq!(file, PathBuf::from("out.flac"));
        assert_eq!(format, AudioOutputFormat::Flac);
        assert!(
            resolve_single_target(source, Path::new("out.opus"), Some(AudioOutputFormat::Flac))
                .is_err()
        );

        let (dir_target, format) =
            resolve_single_target(source, Path::new("dist/"), Some(AudioOutputFormat::Opus))
                .unwrap_or_else(|_| (PathBuf::new(), AudioOutputFormat::Wav));
        assert_eq!(dir_target, PathBuf::from("dist/track.opus"));
        assert_eq!(format, AudioOutputFormat::Opus);
    }

    #[test]
    fn canonicalizes_paths_with_missing_parent_directories() -> Result<(), String> {
        let cwd = std::env::current_dir().map_err(|err| err.to_string())?;
        let nested = cwd.join("target/nested/does/not/exist/output.flac");
        let resolved = canonical_or_parent(&nested).map_err(|err| err.to_string())?;
        assert_eq!(resolved, nested);
        Ok(())
    }

    #[test]
    fn rejects_relative_and_absolute_path_aliases() -> Result<(), String> {
        let cwd = std::env::current_dir().map_err(|err| err.to_string())?;
        let dir = cwd.join(format!("target/audio-alias-{}", std::process::id()));
        fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
        let absolute = dir.join("track.wav");
        fs::write(&absolute, b"RIFF").map_err(|err| err.to_string())?;
        let relative = absolute
            .strip_prefix(&cwd)
            .unwrap_or(&absolute)
            .to_path_buf();
        let result = prepare_audio_batch(
            &relative,
            &absolute,
            None,
            None,
            HardwareBackend::Cpu,
            AudioEncodingRequest::default(),
        );
        let _ = fs::remove_dir_all(&dir);
        assert!(matches!(
            result,
            Err(ProbeError::InvalidDomain(
                DomainError::IdenticalSourceAndTarget(_)
            ))
        ));
        Ok(())
    }

    #[test]
    fn rejects_gpu_backend_before_building_a_plan() -> Result<(), String> {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let source = manifest.join("Cargo.toml");
        let probe_dir = manifest
            .join("target")
            .join(format!("fastconv-audio-gpu-test-{}", std::process::id()));
        fs::create_dir_all(&probe_dir).map_err(|err| err.to_string())?;
        let target = probe_dir.join("out.flac");
        let result = prepare_audio_batch(
            &source,
            &target,
            Some(AudioOutputFormat::Flac),
            Some(NonZeroUsize::MIN),
            HardwareBackend::Gpu(fastconv_domain::GpuBackend::Nvenc),
            AudioEncodingRequest::default(),
        );
        assert!(matches!(
            result,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedAudioPipeline { .. }
            ))
        ));
        Ok(())
    }

    fn id3_prefixed(payload: &[u8]) -> Vec<u8> {
        // 10-byte ID3v2.4 header with a 10-byte synchsafe size, then 10 bytes of tag payload.
        let mut bytes = b"ID3\x04\x00\x00\x00\x00\x00\x0A".to_vec();
        bytes.extend_from_slice(&[0_u8; 10]);
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn detects_new_audio_magic_formats() {
        assert_eq!(
            detect_audio_magic(b"FORM\x00\x00\x00\x00AIFF"),
            Some(AudioInputFormat::Aiff)
        );
        assert_eq!(
            detect_audio_magic(b"FORM\x00\x00\x00\x00AIFC"),
            Some(AudioInputFormat::Aiff)
        );
        assert_eq!(
            detect_audio_magic(b"\x00\x00\x00\x18ftypM4A "),
            Some(AudioInputFormat::M4a)
        );
        let adts = [0xFF, 0xF1, 0x4C, 0x80, 0x02, 0x00, 0xFC];
        let mp3 = [0xFF, 0xFB, 0x90, 0x64];
        assert_eq!(detect_audio_magic(&adts), Some(AudioInputFormat::Aac));
        assert_eq!(
            detect_audio_magic(&id3_prefixed(&adts)),
            Some(AudioInputFormat::Aac)
        );
        assert_eq!(
            detect_audio_magic(&id3_prefixed(&mp3)),
            Some(AudioInputFormat::Mp3)
        );
        assert_eq!(detect_audio_magic(&mp3), Some(AudioInputFormat::Mp3));
    }

    #[test]
    fn resolves_new_target_formats_and_explicit_alac() {
        let source = Path::new("track.wav");
        for (name, expected) in [
            ("out.ogg", AudioOutputFormat::Ogg),
            ("out.mp3", AudioOutputFormat::Mp3),
            ("out.aac", AudioOutputFormat::Aac),
            ("out.m4a", AudioOutputFormat::M4a),
            ("out.aiff", AudioOutputFormat::Aiff),
            ("out.alac", AudioOutputFormat::Alac),
        ] {
            let (file, format) = resolve_single_target(source, Path::new(name), None)
                .unwrap_or_else(|_| (PathBuf::new(), AudioOutputFormat::Wav));
            assert_eq!(file, PathBuf::from(name), "{name}");
            assert_eq!(format, expected, "{name}");
        }

        let (file, format) =
            resolve_single_target(source, Path::new("out.m4a"), Some(AudioOutputFormat::Alac))
                .unwrap_or_else(|_| (PathBuf::new(), AudioOutputFormat::Wav));
        assert_eq!(file, PathBuf::from("out.m4a"));
        assert_eq!(format, AudioOutputFormat::Alac);

        assert!(
            resolve_single_target(source, Path::new("out.flac"), Some(AudioOutputFormat::Alac))
                .is_err()
        );

        let (dir_target, format) =
            resolve_single_target(source, Path::new("dist/"), Some(AudioOutputFormat::Alac))
                .unwrap_or_else(|_| (PathBuf::new(), AudioOutputFormat::Wav));
        assert_eq!(dir_target, PathBuf::from("dist/track.m4a"));
        assert_eq!(format, AudioOutputFormat::Alac);
    }

    #[test]
    fn probes_minimal_adts_aac_frame() -> Result<(), String> {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let dir = manifest
            .join("target")
            .join(format!("fastconv-audio-adts-test-{}", std::process::id()));
        fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
        let path = dir.join("tone.aac");
        let header = rusty_aac::AdtsHeader {
            object_type: 2,
            sample_rate: 44_100,
            channels: 2,
            frame_length: 7,
            header_len: 7,
        };
        let frame = rusty_aac::write_adts_header(&header);
        fs::write(&path, &frame).map_err(|err| err.to_string())?;
        let probed = probe_audio(&path).map_err(|err| err.to_string())?;
        assert_eq!(probed.format, AudioInputFormat::Aac);
        assert_eq!(probed.stream.channels().get(), 2);
        assert_eq!(probed.stream.sample_rate().get(), 44_100);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    fn id3_tag_of_len(total_len: usize) -> Vec<u8> {
        // 10-byte ID3v2.4 header whose payload size is a synchsafe integer.
        let payload = total_len.saturating_sub(10);
        let mut bytes = b"ID3\x04\x00\x00".to_vec();
        for shift in [21, 14, 7, 0] {
            let byte = (payload >> shift) & 0x7F;
            bytes.push(u8::try_from(byte).unwrap_or(0));
        }
        bytes.resize(total_len, 0);
        bytes
    }

    #[test]
    fn probes_adts_after_id3_tag_at_prefix_boundary() -> Result<(), String> {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let dir = manifest.join("target").join(format!(
            "fastconv-audio-adts-boundary-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
        let header = rusty_aac::AdtsHeader {
            object_type: 2,
            sample_rate: 44_100,
            channels: 2,
            frame_length: 7,
            header_len: 7,
        };
        let frame = rusty_aac::write_adts_header(&header);
        // A 4090-byte tag leaves only 6 ADTS header bytes in the 4096-byte prefix;
        // a 4096-byte tag leaves none. Both need the follow-up read after the tag.
        for tag_len in [4090, 4096] {
            let path = dir.join(format!("boundary-{tag_len}.aac"));
            let mut bytes = id3_tag_of_len(tag_len);
            bytes.extend_from_slice(&frame);
            fs::write(&path, &bytes).map_err(|err| err.to_string())?;
            let probed = probe_audio(&path).map_err(|err| err.to_string())?;
            assert_eq!(probed.format, AudioInputFormat::Aac, "tag_len={tag_len}");
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn parses_bitrate_syntax_at_the_boundary() -> Result<(), DomainError> {
        for (raw, bps) in [
            ("128k", 128_000),
            ("192K", 192_000),
            ("320kbps", 320_000),
            ("128KBPS", 128_000),
            (" 6 ", 6_000),
            ("510", 510_000),
            ("999", 999_000),
            ("128000", 128_000),
            ("6000", 6_000),
        ] {
            assert_eq!(parse_audio_bitrate(raw)?.bps(), bps, "raw={raw}");
        }
        for raw in [
            "",
            " ",
            "0",
            "0k",
            "k",
            "kbps",
            "-128k",
            "128kb",
            "12x8k",
            "4294967296k",
            "4294967296",
        ] {
            assert!(
                matches!(
                    parse_audio_bitrate(raw),
                    Err(DomainError::InvalidAudioOptionBitrate { .. })
                ),
                "'{raw}' must be rejected"
            );
        }
        Ok(())
    }

    #[test]
    fn parses_raw_encoding_request_and_rejects_conflicts() -> Result<(), String> {
        let expected = AudioEncodingRequest {
            codec: None,
            rate_control: AudioRateControl::Bitrate(
                AudioOptionBitrate::new(192_000).map_err(|err| err.to_string())?,
            ),
            sample_rate: Some(AudioOptionSampleRate::new(48_000).map_err(|err| err.to_string())?),
            channels: Some(AudioOptionChannels::new(2).map_err(|err| err.to_string())?),
            bit_depth: None,
        };
        assert_eq!(
            parse_audio_encoding_request(None, Some("192k"), None, Some("48000"), Some("2"), None)?,
            expected
        );
        assert_eq!(
            parse_audio_encoding_request(None, None, None, None, None, None)?,
            AudioEncodingRequest::default()
        );
        assert_eq!(
            parse_audio_encoding_request(None, None, Some("85"), None, None, None)?.rate_control,
            AudioRateControl::Quality(AudioQuality::new(85).map_err(|err| err.to_string())?)
        );

        for (bitrate, quality) in [(Some("128k"), Some("80")), (Some("128k"), Some("101"))] {
            let Err(err) = parse_audio_encoding_request(None, bitrate, quality, None, None, None)
            else {
                return Err("bitrate/quality must be rejected".to_owned());
            };
            assert!(
                err.contains("cannot combine --bitrate with --quality"),
                "unexpected diagnostic: {err}"
            );
        }
        for (args, needle) in [
            ((None, Some("101"), None, None), "invalid --quality"),
            ((None, Some("abc"), None, None), "invalid --quality"),
            ((None, None, Some("0"), None), "invalid --sample-rate"),
            ((None, None, Some("abc"), None), "invalid --sample-rate"),
            ((None, None, None, Some("0")), "invalid --channels"),
            ((None, None, None, Some("9")), "invalid --channels"),
            ((Some("abc"), None, None, None), "invalid audio bitrate"),
        ] {
            let Err(err) = parse_audio_encoding_request(None, args.0, args.1, args.2, args.3, None)
            else {
                return Err(format!("malformed value must be rejected: {args:?}"));
            };
            assert!(err.contains(needle), "expected '{needle}' in '{err}'");
        }
        Ok(())
    }

    fn bitrate_request(bps: u32) -> AudioEncodingRequest {
        AudioEncodingRequest {
            rate_control: AudioRateControl::Bitrate(
                AudioOptionBitrate::new(bps).unwrap_or_else(|_| unreachable!()),
            ),
            ..AudioEncodingRequest::default()
        }
    }

    #[test]
    fn rejects_illegal_options_before_probing_single_file() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let target = Path::new("target/rejected.flac");
        let result = prepare_audio_batch(
            &source,
            target,
            Some(AudioOutputFormat::Flac),
            Some(NonZeroUsize::MIN),
            HardwareBackend::Cpu,
            bitrate_request(128_000),
        );
        assert!(matches!(
            result,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedAudioRateControl { .. }
            ))
        ));

        let opus_target = Path::new("target/rejected.opus");
        let request = AudioEncodingRequest {
            sample_rate: Some(
                AudioOptionSampleRate::new(44_100).unwrap_or_else(|_| unreachable!()),
            ),
            ..AudioEncodingRequest::default()
        };
        let result = prepare_audio_batch(
            &source,
            opus_target,
            Some(AudioOutputFormat::Opus),
            Some(NonZeroUsize::MIN),
            HardwareBackend::Cpu,
            request,
        );
        assert!(matches!(
            result,
            Err(ProbeError::InvalidDomain(
                DomainError::InvalidAudioOptionSampleRate { .. }
            ))
        ));
    }

    #[test]
    fn rejects_illegal_options_before_probing_directory_batch() -> Result<(), String> {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let source_dir = manifest
            .join("target")
            .join(format!("fastconv-audio-options-src-{}", std::process::id()));
        let target_dir = manifest
            .join("target")
            .join(format!("fastconv-audio-options-dst-{}", std::process::id()));
        fs::create_dir_all(&source_dir).map_err(|err| err.to_string())?;
        fs::create_dir_all(&target_dir).map_err(|err| err.to_string())?;
        fs::write(source_dir.join("dummy.wav"), b"not decoded").map_err(|err| err.to_string())?;

        let result = prepare_audio_batch(
            &source_dir,
            &target_dir,
            Some(AudioOutputFormat::Flac),
            Some(NonZeroUsize::MIN),
            HardwareBackend::Cpu,
            bitrate_request(192_000),
        );
        assert!(
            matches!(
                result,
                Err(ProbeError::InvalidDomain(
                    DomainError::UnsupportedAudioRateControl { .. }
                ))
            ),
            "directory batches must reject illegal options before probing files"
        );

        let result = prepare_audio_batch(
            &source_dir,
            &target_dir,
            Some(AudioOutputFormat::Mp3),
            Some(NonZeroUsize::MIN),
            HardwareBackend::Cpu,
            AudioEncodingRequest {
                channels: Some(AudioOptionChannels::new(4).unwrap_or_else(|_| unreachable!())),
                ..AudioEncodingRequest::default()
            },
        );
        assert!(matches!(
            result,
            Err(ProbeError::InvalidDomain(
                DomainError::InvalidAudioOptionChannels { .. }
            ))
        ));
        let _ = fs::remove_dir_all(&source_dir);
        let _ = fs::remove_dir_all(&target_dir);
        Ok(())
    }

    #[test]
    fn parses_bit_depth_tokens_at_the_boundary() {
        for (raw, expected) in [
            ("8", AudioBitDepth::Int8),
            ("s8", AudioBitDepth::Int8),
            ("u8", AudioBitDepth::Int8),
            ("16", AudioBitDepth::Int16),
            ("S16", AudioBitDepth::Int16),
            ("24", AudioBitDepth::Int24),
            (" s24 ", AudioBitDepth::Int24),
            ("32", AudioBitDepth::Int32),
            ("S32", AudioBitDepth::Int32),
            ("f32", AudioBitDepth::Float32),
            ("FLOAT", AudioBitDepth::Float32),
            ("float32", AudioBitDepth::Float32),
        ] {
            assert_eq!(parse_audio_bit_depth(raw), Ok(expected), "raw={raw}");
        }
        for raw in [
            "", " ", "0", "7", "12", "s64", "f64", "double", "abc", "-16",
        ] {
            assert!(
                matches!(
                    parse_audio_bit_depth(raw),
                    Err(DomainError::InvalidAudioBitDepth { .. })
                ),
                "'{raw}' must be rejected"
            );
        }
    }

    #[test]
    fn parses_bit_depth_in_the_raw_encoding_request() -> Result<(), String> {
        let request = parse_audio_encoding_request(None, None, None, None, None, Some(" f32 "))?;
        assert_eq!(request.bit_depth, Some(AudioBitDepth::Float32));
        assert_eq!(
            parse_audio_encoding_request(None, None, None, None, None, None)?.bit_depth,
            None
        );
        let error = parse_audio_encoding_request(None, None, None, None, None, Some("64"))
            .err()
            .ok_or_else(|| "invalid depth must be rejected".to_owned())?;
        assert!(error.contains("invalid audio bit depth"), "{error}");
        Ok(())
    }

    fn probe_pcm_bytes(
        tag: u16,
        channels: u16,
        bits: u16,
        data: &[u8],
        extensible: bool,
    ) -> Vec<u8> {
        let block_align = channels * (bits / 8);
        let rate = 48_000_u32;
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&tag.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&rate.saturating_mul(u32::from(block_align)).to_le_bytes());
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        if extensible {
            fmt.extend_from_slice(&22_u16.to_le_bytes());
            fmt.extend_from_slice(&bits.to_le_bytes());
            fmt.extend_from_slice(&3_u32.to_le_bytes());
            fmt.extend_from_slice(&[
                0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38,
                0x9B, 0x71,
            ]);
        }
        let mut chunks = Vec::new();
        chunks.extend_from_slice(b"fmt ");
        chunks.extend_from_slice(&u32::try_from(fmt.len()).unwrap_or(0).to_le_bytes());
        chunks.extend_from_slice(&fmt);
        chunks.extend_from_slice(b"data");
        chunks.extend_from_slice(&u32::try_from(data.len()).unwrap_or(0).to_le_bytes());
        chunks.extend_from_slice(data);
        if data.len() % 2 == 1 {
            chunks.push(0);
        }
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(
            &u32::try_from(chunks.len().saturating_add(4))
                .unwrap_or(0)
                .to_le_bytes(),
        );
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(&chunks);
        bytes
    }

    #[test]
    fn probes_canonical_pcm_wav_layout_and_rejects_noncanonical_headers() -> Result<(), String> {
        let dir = probe_temp_dir("wav-layout");
        let data: Vec<u8> = (0..2_000_u32)
            .map(|index| u8::try_from(index % 251).unwrap_or(0))
            .collect();

        let valid = dir.join("valid.wav");
        fs::write(&valid, probe_pcm_bytes(1, 1, 16, &data, false))
            .map_err(|err| err.to_string())?;
        let probed = probe_audio(&valid).map_err(|err| err.to_string())?;
        let layout = probed
            .stream
            .pcm_wav_layout()
            .ok_or_else(|| "plain PCM must expose a validated layout".to_owned())?;
        assert_eq!(layout.channels().get(), 1);
        assert_eq!(layout.bits_per_sample().get(), 16);
        assert_eq!(layout.sample_rate().get(), 48_000);
        assert_eq!(layout.data_offset(), 44);
        assert_eq!(layout.data_bytes(), 2_000);
        assert_eq!(layout.total_samples(), 1_000);
        assert_eq!(layout.channel_mask(), 0);

        // WAVEFORMATEXTENSIBLE carries a speaker mask; the probe must keep it so
        // a consumer can reject layouts it cannot map correctly.
        let mut extensible = probe_pcm_bytes(0xFFFE, 2, 16, &data, true);
        extensible[40..44].copy_from_slice(&0xB_u32.to_le_bytes());
        extensible[44] = 0x01; // signed-integer PCM sub-format GUID
        let ext_path = dir.join("extensible.wav");
        fs::write(&ext_path, &extensible).map_err(|err| err.to_string())?;
        let probed = probe_audio(&ext_path).map_err(|err| err.to_string())?;
        let layout = probed
            .stream
            .pcm_wav_layout()
            .ok_or_else(|| "canonical extensible PCM must expose a layout".to_owned())?;
        assert_eq!(layout.channel_mask(), 0xB);
        assert_eq!(layout.channels().get(), 2);

        // Unsigned 8-bit, IEEE-float and a non-PCM extensible GUID have no
        // canonical signed-PCM layout, so the generic decoder must handle them.
        for bytes in [
            probe_pcm_bytes(1, 1, 8, &data, false),
            probe_pcm_bytes(3, 1, 32, &data, false),
            probe_pcm_bytes(0xFFFE, 1, 32, &data, true),
        ] {
            let path = dir.join("noncanonical.wav");
            fs::write(&path, &bytes).map_err(|err| err.to_string())?;
            assert_eq!(
                probe_audio(&path)
                    .map_err(|err| err.to_string())?
                    .stream
                    .pcm_wav_layout(),
                None
            );
        }

        // Truncating the payload while leaving the declared sizes intact must
        // not hand the pipeline a layout pointing past EOF.
        let mut truncated = probe_pcm_bytes(1, 1, 16, &data, false);
        truncated.truncate(truncated.len() - 2);
        let path = dir.join("truncated.wav");
        fs::write(&path, &truncated).map_err(|err| err.to_string())?;
        assert_eq!(
            probe_audio(&path)
                .map_err(|err| err.to_string())?
                .stream
                .pcm_wav_layout(),
            None
        );

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    fn add_chunk(bytes: &mut Vec<u8>, id: [u8; 4], body: &[u8]) {
        bytes.extend_from_slice(&id);
        bytes.extend_from_slice(&u32::try_from(body.len()).unwrap_or(0).to_be_bytes());
        bytes.extend_from_slice(body);
        if body.len() % 2 == 1 {
            bytes.push(0);
        }
    }

    fn aiff_pcm_bytes(
        compression: Option<&[u8; 4]>,
        data: &[u8],
        bits: u16,
        frames: u32,
    ) -> Vec<u8> {
        let form_type = if compression.is_some() {
            b"AIFC"
        } else {
            b"AIFF"
        };
        let mut chunks = Vec::new();
        if compression.is_some() {
            add_chunk(&mut chunks, *b"FVER", &0xA280_5140_u32.to_be_bytes());
        }
        let mut comm = Vec::new();
        comm.extend_from_slice(&1_u16.to_be_bytes());
        comm.extend_from_slice(&frames.to_be_bytes());
        comm.extend_from_slice(&bits.to_be_bytes());
        comm.extend_from_slice(&[0x40, 0x0E, 0xBB, 0x80, 0, 0, 0, 0, 0, 0]);
        if let Some(kind) = compression {
            comm.extend_from_slice(kind);
            comm.extend_from_slice(b"\x0f32-bit floating");
        }
        add_chunk(&mut chunks, *b"COMM", &comm);
        let mut ssnd = Vec::new();
        ssnd.extend_from_slice(&0_u32.to_be_bytes());
        ssnd.extend_from_slice(&0_u32.to_be_bytes());
        ssnd.extend_from_slice(data);
        add_chunk(&mut chunks, *b"SSND", &ssnd);
        let mut bytes = b"FORM".to_vec();
        let payload = chunks.len().saturating_add(4);
        bytes.extend_from_slice(&u32::try_from(payload).unwrap_or(0).to_be_bytes());
        bytes.extend_from_slice(form_type);
        bytes.extend_from_slice(&chunks);
        bytes
    }

    #[test]
    fn probes_32bit_integer_and_float_wav_and_aiff() -> Result<(), String> {
        let dir = std::env::temp_dir().join(format!("fastconv-probe-32bit-{}", std::process::id()));
        fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
        let i32_data: Vec<u8> = [1_i32, -2, i32::MAX - 7]
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        let f32_data: Vec<u8> = [0.25_f32, -0.5, 0.75]
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        // AIFF/AIFC sample payloads are big-endian.
        let mut float_payload = Vec::new();
        for sample in [0.25_f32, -0.5, 0.75] {
            float_payload.extend_from_slice(&sample.to_be_bytes());
        }
        let int_payload: Vec<u8> = [1_i32, -2, i32::MAX - 7]
            .iter()
            .flat_map(|sample| sample.to_be_bytes())
            .collect();
        let cases: Vec<(&str, Vec<u8>, u16, AudioSampleFormat)> = vec![
            (
                "int32.wav",
                probe_pcm_bytes(1, 1, 32, &i32_data, false),
                32,
                AudioSampleFormat::Int,
            ),
            (
                "float32.wav",
                probe_pcm_bytes(3, 1, 32, &f32_data, false),
                32,
                AudioSampleFormat::Float,
            ),
            (
                "extfloat32.wav",
                probe_pcm_bytes(0xFFFE, 1, 32, &f32_data, true),
                32,
                AudioSampleFormat::Float,
            ),
            (
                "float32.aiff",
                aiff_pcm_bytes(Some(b"fl32"), &float_payload, 32, 3),
                32,
                AudioSampleFormat::Float,
            ),
            (
                "int32.aiff",
                aiff_pcm_bytes(None, &int_payload, 32, 3),
                32,
                AudioSampleFormat::Int,
            ),
        ];
        for (name, bytes, bits, expected_format) in cases {
            let path = dir.join(name);
            fs::write(&path, &bytes).map_err(|err| err.to_string())?;
            let probed = probe_audio(&path).map_err(|err| err.to_string())?;
            assert_eq!(probed.stream.sample_format(), expected_format, "{name}");
            assert_eq!(
                probed.stream.bits_per_sample().map(NonZeroU16::get),
                Some(bits),
                "{name}"
            );
            assert_eq!(probed.stream.channels().get(), 1, "{name}");
            assert_eq!(probed.stream.sample_rate().get(), 48_000, "{name}");
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn rejects_bit_depth_for_lossy_targets_before_probing() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        for format in [
            AudioOutputFormat::Mp3,
            AudioOutputFormat::Aac,
            AudioOutputFormat::M4a,
            AudioOutputFormat::Opus,
            AudioOutputFormat::Ogg,
        ] {
            let request = AudioEncodingRequest {
                bit_depth: Some(AudioBitDepth::Int16),
                ..AudioEncodingRequest::default()
            };
            let result = prepare_audio_batch(
                &source,
                Path::new(&format!("target/rejected.{}", format.extension())),
                Some(format),
                Some(NonZeroUsize::MIN),
                HardwareBackend::Cpu,
                request,
            );
            assert!(
                matches!(
                    result,
                    Err(ProbeError::InvalidDomain(
                        DomainError::UnsupportedAudioBitDepth { .. }
                    ))
                ),
                "{format} must reject --bit-depth"
            );
        }
    }

    #[test]
    fn parses_codec_tokens_at_the_boundary() -> Result<(), DomainError> {
        for (raw, expected) in [
            ("vorbis", AudioTargetCodec::Vorbis),
            (" Opus ", AudioTargetCodec::Opus),
            ("FLAC", AudioTargetCodec::Flac),
            ("aac", AudioTargetCodec::Aac),
            ("ALAC", AudioTargetCodec::Alac),
            ("pcm", AudioTargetCodec::Pcm),
            ("Mp3", AudioTargetCodec::Mp3),
        ] {
            assert_eq!(parse_audio_codec(raw), Ok(expected), "raw={raw}");
        }
        for raw in ["", " ", "wma", "flac1", "vorbis opus"] {
            assert!(
                matches!(
                    parse_audio_codec(raw),
                    Err(DomainError::InvalidAudioCodec { .. })
                ),
                "'{raw}' must be rejected"
            );
        }
        let request =
            parse_audio_encoding_request(Some(" flac "), None, None, None, None, Some("24"))
                .map_err(|_| DomainError::EmptyField("audio.codec"))?;
        assert_eq!(request.codec, Some(AudioTargetCodec::Flac));
        assert_eq!(request.bit_depth, Some(AudioBitDepth::Int24));
        assert_eq!(
            parse_audio_encoding_request(None, None, None, None, None, None)
                .map_err(|_| DomainError::EmptyField("audio.request"))?
                .codec,
            None
        );
        Ok(())
    }

    #[test]
    fn detects_ogg_flac_from_identification_packet() {
        let mut page = b"OggS".to_vec();
        page.extend_from_slice(&[0_u8; 22]);
        page.extend_from_slice(&[1, 51]);
        let mut identification = b"\x7fFLAC\x01\x00\x00\x01fLaC".to_vec();
        identification.resize(51, 0);
        page.extend_from_slice(&identification);
        assert_eq!(detect_audio_magic(&page), Some(AudioInputFormat::OggFlac));
        // A Vorbis comment mentioning FLAC must not be mistaken for Ogg FLAC.
        let mut vorbis = b"OggS".to_vec();
        vorbis.extend_from_slice(&[0_u8; 22]);
        vorbis.extend_from_slice(&[1, 7]);
        vorbis.extend_from_slice(b"\x01vorbis");
        assert_eq!(detect_audio_magic(&vorbis), Some(AudioInputFormat::Vorbis));
    }

    // Builds a single-packet Ogg page with correct lacing (packets <= 255 bytes here).
    fn ogg_page(header_type: u8, packet: &[u8]) -> Vec<u8> {
        let mut page = b"OggS".to_vec();
        page.push(0); // stream version
        page.push(header_type);
        page.extend_from_slice(&[0_u8; 16]); // granule position, serial number, sequence
        page.extend_from_slice(&[0_u8; 4]); // CRC (the detector does not verify it)
        page.push(1);
        page.push(u8::try_from(packet.len()).unwrap_or(u8::MAX));
        page.extend_from_slice(packet);
        page
    }

    /// Builds one Ogg page with correct lacing and CRC, used to forge suffixes.
    fn ogg_page_with(granule: u64, serial: u32, header_type: u8, packet: &[u8]) -> Vec<u8> {
        let mut lacing = Vec::new();
        let mut remaining = packet.len();
        while remaining >= 255 {
            lacing.push(255);
            remaining -= 255;
        }
        lacing.push(u8::try_from(remaining).unwrap_or(0));
        let mut page = b"OggS".to_vec();
        page.push(0);
        page.push(header_type);
        page.extend_from_slice(&granule.to_le_bytes());
        page.extend_from_slice(&serial.to_le_bytes());
        page.extend_from_slice(&0_u32.to_le_bytes()); // sequence
        page.extend_from_slice(&[0_u8; 4]); // CRC placeholder
        page.push(u8::try_from(lacing.len()).unwrap_or(u8::MAX));
        page.extend_from_slice(&lacing);
        page.extend_from_slice(packet);
        let crc = ogg_crc32(&page);
        page[22..26].copy_from_slice(&crc.to_le_bytes());
        page
    }

    /// Encodes three 20 ms packets into a valid Opus stream. A one-byte page
    /// target puts each packet on its own page, which lets a test shift the
    /// granule timeline without changing how many samples complete per page.
    fn opus_stream(serial: u32, page_target: usize) -> Result<Vec<u8>, String> {
        let mut encoder = opus_pure::OpusEncoder::new(48_000, 2, opus_pure::Application::Audio)
            .map_err(|err| err.to_string())?;
        let head = opus_pure::OpusHead::for_encoder(&encoder, 48_000);
        let mut writer = opus_pure::OggOpusWriter::with_serial(
            Vec::new(),
            head,
            opus_pure::OpusTags::new(),
            serial,
        )
        .map_err(|err| err.to_string())?;
        writer.set_page_target(page_target);
        let mut packet = vec![0_u8; opus_pure::MAX_PACKET_BYTES];
        let silence = vec![0.0_f32; 960 * 2];
        for _ in 0..3 {
            let written = encoder
                .encode(&silence, 960, &mut packet)
                .map_err(|err| err.to_string())?;
            writer
                .write_packet(&packet[..written])
                .map_err(|err| err.to_string())?;
        }
        writer.finish().map_err(|err| err.to_string())
    }

    /// Adds `delta` to every audio page's granule, recomputing each CRC. The
    /// two leading header pages keep granule zero as RFC 7845 requires.
    fn shift_audio_granules(bytes: &[u8], delta: u64) -> Option<Vec<u8>> {
        let mut out = bytes.to_vec();
        let mut at = 0_usize;
        let mut page_index = 0_usize;
        while at < out.len() {
            let rel = out[at..].windows(4).position(|window| window == b"OggS")?;
            let start = at + rel;
            let segments = usize::from(*out.get(start + 26)?);
            let table_end = start.checked_add(27)?.checked_add(segments)?;
            let body_len = out
                .get(start + 27..table_end)?
                .iter()
                .try_fold(0_usize, |total, &lace| total.checked_add(usize::from(lace)))?;
            let end = table_end.checked_add(body_len)?;
            if page_index >= 2 {
                let granule_bytes = <[u8; 8]>::try_from(out.get(start + 6..start + 14)?).ok()?;
                let granule = u64::from_le_bytes(granule_bytes).checked_add(delta)?;
                out[start + 6..start + 14].copy_from_slice(&granule.to_le_bytes());
                let mut crc_buf = out.get(start..end)?.to_vec();
                crc_buf[22..26].copy_from_slice(&[0_u8; 4]);
                let crc = ogg_crc32(&crc_buf);
                out[start + 22..start + 26].copy_from_slice(&crc.to_le_bytes());
            }
            page_index += 1;
            at = end;
        }
        Some(out)
    }

    /// Splits a valid Ogg stream into complete page byte ranges.
    fn split_ogg_pages(bytes: &[u8]) -> Vec<Vec<u8>> {
        let mut pages = Vec::new();
        let mut at = 0_usize;
        while at < bytes.len() {
            let Some(rel) = bytes[at..].windows(4).position(|window| window == b"OggS") else {
                break;
            };
            let start = at + rel;
            let Some(&segments) = bytes.get(start + 26) else {
                break;
            };
            let segments = usize::from(segments);
            let Some(table_end) = start
                .checked_add(27)
                .and_then(|end| end.checked_add(segments))
            else {
                break;
            };
            let Some(table) = bytes.get(start + 27..table_end) else {
                break;
            };
            let body_len = table
                .iter()
                .try_fold(0_usize, |total, &lace| total.checked_add(usize::from(lace)));
            let Some(body_len) = body_len else {
                break;
            };
            let Some(end) = table_end.checked_add(body_len) else {
                break;
            };
            let Some(page) = bytes.get(start..end) else {
                break;
            };
            pages.push(page.to_vec());
            at = end;
        }
        pages
    }

    #[test]
    fn detects_ogg_audio_after_skeleton_pages() {
        let mut fishead = vec![0_u8; 44];
        fishead[..8].copy_from_slice(b"fishead\0");
        let mut fisbone = vec![0_u8; 30];
        fisbone[..8].copy_from_slice(b"fisbone\0");

        // Skeleton alone is not an audio identification packet.
        assert_eq!(detect_audio_magic(&ogg_page(0x02, &fishead)), None);

        let mut bytes = ogg_page(0x02, &fishead);
        bytes.extend_from_slice(&ogg_page(0x00, &fisbone));
        bytes.extend_from_slice(&ogg_page(0x02, b"\x7fFLAC\x01\x00\x00\x01fLaC"));
        assert_eq!(detect_audio_magic(&bytes), Some(AudioInputFormat::OggFlac));

        let mut bytes = ogg_page(0x02, &fishead);
        bytes.extend_from_slice(&ogg_page(0x02, b"\x01vorbis"));
        assert_eq!(detect_audio_magic(&bytes), Some(AudioInputFormat::Vorbis));

        // `opus-pure` needs OpusHead as the very first packet, so a Skeleton
        // prefix must not be reported as an openable Opus stream.
        let mut opus = ogg_page(0x02, &fishead);
        opus.extend_from_slice(&ogg_page(0x02, b"OpusHead"));
        assert_eq!(detect_audio_magic(&opus), None);
        assert_eq!(
            detect_audio_magic(&ogg_page(0x02, b"OpusHead")),
            Some(AudioInputFormat::Opus)
        );
    }

    #[test]
    fn resolves_codec_specific_modes_for_batches_and_directory_targets() -> Result<(), String> {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = manifest
            .join("target")
            .join(format!("fastconv-audio-codec-batch-{}", std::process::id()));
        let source_dir = root.join("src");
        let target_dir = root.join("dst");
        fs::create_dir_all(&source_dir).map_err(|err| err.to_string())?;
        fs::create_dir_all(&target_dir).map_err(|err| err.to_string())?;
        let data: Vec<u8> = (0..2_048)
            .map(|index| u8::try_from(index % 251).unwrap_or(0))
            .collect();
        for name in ["first.wav", "second.wav"] {
            fs::write(
                source_dir.join(name),
                probe_pcm_bytes(1, 2, 16, &data, false),
            )
            .map_err(|err| err.to_string())?;
        }
        let threads = Some(NonZeroUsize::MIN);

        let flac_request = AudioEncodingRequest {
            codec: Some(AudioTargetCodec::Flac),
            bit_depth: Some(AudioBitDepth::Int24),
            ..AudioEncodingRequest::default()
        };
        let plan = prepare_audio_batch(
            &source_dir,
            &target_dir,
            Some(AudioOutputFormat::Ogg),
            threads,
            HardwareBackend::Cpu,
            flac_request,
        )
        .map_err(|err| err.to_string())?;
        assert_eq!(plan.mode(), AudioExecutionMode::CpuOggFlac);
        assert_eq!(plan.target_codec(), AudioTargetCodec::Flac);
        assert_eq!(plan.options().bit_depth(), Some(AudioBitDepth::Int24));
        assert_eq!(plan.items().len(), 2);
        assert!(
            plan.items()
                .iter()
                .all(|item| item.target().extension().is_some_and(|ext| ext == "ogg"))
        );

        let alac_request = AudioEncodingRequest {
            codec: Some(AudioTargetCodec::Alac),
            bit_depth: Some(AudioBitDepth::Int16),
            ..AudioEncodingRequest::default()
        };
        let plan = prepare_audio_batch(
            &source_dir,
            &target_dir,
            Some(AudioOutputFormat::M4a),
            threads,
            HardwareBackend::Cpu,
            alac_request,
        )
        .map_err(|err| err.to_string())?;
        assert_eq!(plan.mode(), AudioExecutionMode::CpuM4aAlac);
        assert_eq!(plan.target_codec(), AudioTargetCodec::Alac);
        assert_eq!(plan.options().bit_depth(), Some(AudioBitDepth::Int16));
        assert!(
            plan.items()
                .iter()
                .all(|item| item.target().extension().is_some_and(|ext| ext == "m4a"))
        );

        // An invalid codec/container pair is rejected before any item is probed.
        let invalid = AudioEncodingRequest {
            codec: Some(AudioTargetCodec::Aac),
            ..AudioEncodingRequest::default()
        };
        let result = prepare_audio_batch(
            &source_dir,
            &target_dir,
            Some(AudioOutputFormat::Ogg),
            threads,
            HardwareBackend::Cpu,
            invalid,
        );
        assert!(matches!(
            result,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedAudioCodecForFormat { .. }
            ))
        ));

        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn resolves_codec_for_single_file_directory_targets_and_transforms() -> Result<(), String> {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = manifest.join("target").join(format!(
            "fastconv-audio-codec-single-{}",
            std::process::id()
        ));
        let source_dir = root.join("src");
        let target_dir = root.join("dst");
        fs::create_dir_all(&source_dir).map_err(|err| err.to_string())?;
        fs::create_dir_all(&target_dir).map_err(|err| err.to_string())?;
        let data: Vec<u8> = (0..2_048)
            .map(|index| u8::try_from(index % 251).unwrap_or(0))
            .collect();
        let single = source_dir.join("first.wav");
        fs::write(&single, probe_pcm_bytes(1, 2, 16, &data, false))
            .map_err(|err| err.to_string())?;
        let threads = Some(NonZeroUsize::MIN);

        // `.oga` is an alias for the Ogg container, including with --codec.
        let (oga_target, oga_format) =
            resolve_single_target(&single, Path::new("out.oga"), Some(AudioOutputFormat::Ogg))
                .unwrap_or_else(|_| (PathBuf::new(), AudioOutputFormat::Wav));
        assert_eq!(oga_target, PathBuf::from("out.oga"));
        assert_eq!(oga_format, AudioOutputFormat::Ogg);

        // Single file targeting a directory: the directory suffix selects the
        // container and --codec decides which encoder the plan carries.
        let nested = target_dir.join("nested/");
        let plan = prepare_audio_batch(
            &single,
            &nested,
            Some(AudioOutputFormat::Ogg),
            threads,
            HardwareBackend::Cpu,
            AudioEncodingRequest {
                codec: Some(AudioTargetCodec::Opus),
                sample_rate: Some(
                    AudioOptionSampleRate::new(48_000).unwrap_or_else(|_| unreachable!()),
                ),
                ..AudioEncodingRequest::default()
            },
        )
        .map_err(|err| err.to_string())?;
        assert_eq!(plan.mode(), AudioExecutionMode::CpuOggOpus);
        assert_eq!(plan.target_codec(), AudioTargetCodec::Opus);
        assert_eq!(plan.items()[0].target(), nested.join("first.ogg"));

        // Resampling plus remixing stays valid for multi-codec containers.
        let plan = prepare_audio_batch(
            &single,
            &target_dir.join("transformed.ogg"),
            Some(AudioOutputFormat::Ogg),
            threads,
            HardwareBackend::Cpu,
            AudioEncodingRequest {
                codec: Some(AudioTargetCodec::Flac),
                sample_rate: Some(
                    AudioOptionSampleRate::new(44_100).unwrap_or_else(|_| unreachable!()),
                ),
                channels: Some(AudioOptionChannels::new(1).unwrap_or_else(|_| unreachable!())),
                bit_depth: Some(AudioBitDepth::Int24),
                ..AudioEncodingRequest::default()
            },
        )
        .map_err(|err| err.to_string())?;
        assert_eq!(plan.mode(), AudioExecutionMode::CpuOggFlac);

        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    fn probe_temp_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir = std::env::temp_dir().join(format!("fastconv-probe-{label}-{nanos}"));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn probes_adts_sample_counts_and_withholds_them_for_malformed_files() -> Result<(), String> {
        let dir = probe_temp_dir("adts-count");
        let path = dir.join("frames.aac");
        let frame = |blocks: u8| -> Vec<u8> {
            let mut header = rusty_aac::write_adts_header(&rusty_aac::AdtsHeader {
                object_type: 2,
                sample_rate: 48_000,
                channels: 2,
                frame_length: 9,
                header_len: 7,
            });
            header[6] |= blocks;
            let mut bytes = header;
            bytes.extend_from_slice(&[0_u8; 2]);
            bytes
        };
        let mut bytes = frame(0);
        bytes.extend_from_slice(&frame(1));
        bytes.extend_from_slice(&frame(0));
        fs::write(&path, &bytes).map_err(|err| err.to_string())?;
        // The ADTS core rate cannot establish the decoded length of an
        // SBR-capable stream, so no count is published even for clean frames.
        assert_eq!(
            probe_audio(&path)
                .map_err(|err| err.to_string())?
                .stream
                .total_samples(),
            None
        );

        // A truncated final frame leaves the walk unable to prove the stream end.
        let truncated = dir.join("truncated.aac");
        fs::write(&truncated, &bytes[..bytes.len() - 1]).map_err(|err| err.to_string())?;
        assert_eq!(
            probe_audio(&truncated)
                .map_err(|err| err.to_string())?
                .stream
                .total_samples(),
            None
        );

        // A broken second syncword is malformed, so no count is reported.
        let corrupt = dir.join("corrupt.aac");
        let mut corrupt_bytes = bytes.clone();
        corrupt_bytes[9] = 0;
        fs::write(&corrupt, &corrupt_bytes).map_err(|err| err.to_string())?;
        assert_eq!(
            probe_audio(&corrupt)
                .map_err(|err| err.to_string())?
                .stream
                .total_samples(),
            None
        );

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn probes_opus_sample_count_and_withholds_it_for_trailing_garbage() -> Result<(), String> {
        let dir = probe_temp_dir("opus-count");
        let path = dir.join("tone.opus");
        let mut encoder = opus_pure::OpusEncoder::new(48_000, 2, opus_pure::Application::Audio)
            .map_err(|err| err.to_string())?;
        let head = opus_pure::OpusHead::for_encoder(&encoder, 48_000);
        let mut writer =
            opus_pure::OggOpusWriter::new(Vec::new(), head).map_err(|err| err.to_string())?;
        let mut packet = vec![0_u8; opus_pure::MAX_PACKET_BYTES];
        let silence = vec![0.0_f32; 960 * 2];
        for _ in 0..3 {
            let written = encoder
                .encode(&silence, 960, &mut packet)
                .map_err(|err| err.to_string())?;
            writer
                .write_packet(&packet[..written])
                .map_err(|err| err.to_string())?;
        }
        let bytes = writer.finish().map_err(|err| err.to_string())?;
        fs::write(&path, &bytes).map_err(|err| err.to_string())?;
        // Three 20 ms packets decode to 2880 samples, of which the first
        // `pre_skip` (312) are discarded by the Ogg Opus pre-roll.
        assert_eq!(
            probe_audio(&path)
                .map_err(|err| err.to_string())?
                .stream
                .total_samples(),
            Some(2_568)
        );

        // Bytes after the final page make that page's end ambiguous.
        let mut junk = bytes;
        junk.extend_from_slice(b"trailing-garbage");
        fs::write(&path, &junk).map_err(|err| err.to_string())?;
        assert_eq!(
            probe_audio(&path)
                .map_err(|err| err.to_string())?
                .stream
                .total_samples(),
            None
        );

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn opus_withholds_count_for_forged_suffix_pages() -> Result<(), String> {
        let dir = probe_temp_dir("opus-forged-suffix");
        let path = dir.join("tone.opus");
        let serial = 0x1122_3344_u32;
        let base = opus_stream(serial, 1)?;
        fs::write(&path, &base).map_err(|err| err.to_string())?;
        assert_eq!(
            probe_audio(&path)
                .map_err(|err| err.to_string())?
                .stream
                .total_samples(),
            Some(2_568)
        );

        // A page with an arbitrary granule but the wrong logical stream serial
        // must not be mistaken for the final page of this stream.
        let mut wrong_serial = base.clone();
        wrong_serial.extend_from_slice(&ogg_page_with(100_000, serial ^ 1, 0x04, &[0_u8; 1]));
        fs::write(&path, &wrong_serial).map_err(|err| err.to_string())?;
        assert_eq!(
            probe_audio(&path)
                .map_err(|err| err.to_string())?
                .stream
                .total_samples(),
            None
        );

        // The right serial does not help a page whose body was changed without
        // updating its CRC.
        let mut bad_crc_page = ogg_page_with(100_000, serial, 0x04, &[0_u8; 1]);
        let last = bad_crc_page.len().saturating_sub(1);
        bad_crc_page[last] ^= 0x01;
        let mut bad_crc = base.clone();
        bad_crc.extend_from_slice(&bad_crc_page);
        fs::write(&path, &bad_crc).map_err(|err| err.to_string())?;
        assert_eq!(
            probe_audio(&path)
                .map_err(|err| err.to_string())?
                .stream
                .total_samples(),
            None
        );

        // Corrupting the genuine final page also withholds the count.
        let mut corrupt = base.clone();
        let last = corrupt.len().saturating_sub(1);
        corrupt[last] ^= 0x01;
        fs::write(&path, &corrupt).map_err(|err| err.to_string())?;
        assert_eq!(
            probe_audio(&path)
                .map_err(|err| err.to_string())?
                .stream
                .total_samples(),
            None
        );

        // A CRC-valid page with the right serial appended after the logical EOS
        // is not the page the reader reached; its small granule must be ignored.
        let mut after_eos = base;
        after_eos.extend_from_slice(&ogg_page_with(1_000, serial, 0x04, &[0_u8; 1]));
        fs::write(&path, &after_eos).map_err(|err| err.to_string())?;
        assert_eq!(
            probe_audio(&path)
                .map_err(|err| err.to_string())?
                .stream
                .total_samples(),
            None
        );

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn opus_withholds_count_for_nonzero_initial_granule() -> Result<(), String> {
        let dir = probe_temp_dir("opus-initial-granule");
        let path = dir.join("tone.opus");
        let base = opus_stream(0x1122_3344, 1)?;
        // A positive initial granule shifts the converter's Trim timeline, so
        // no decoded length can be proven from the granule position.
        let shifted = shift_audio_granules(&base, 1_000)
            .ok_or_else(|| "failed to shift Opus granules".to_owned())?;
        fs::write(&path, &shifted).map_err(|err| err.to_string())?;
        assert_eq!(
            probe_audio(&path)
                .map_err(|err| err.to_string())?
                .stream
                .total_samples(),
            None
        );
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn opus_initial_offset_counts_only_the_first_audio_page() -> Result<(), String> {
        let dir = probe_temp_dir("opus-first-page-boundary");
        let path = dir.join("tone.opus");
        let serial = 0x1122_3344_u32;
        let pages = split_ogg_pages(&opus_stream(serial, 1)?);
        assert!(
            pages.len() >= 5,
            "expected two header and three audio pages"
        );
        // Keep the real first audio page (granule 960) but insert a later page
        // that repeats that granule. It must not be counted into the offset.
        let mut stream = Vec::new();
        stream.extend(pages[0].iter().copied());
        stream.extend(pages[1].iter().copied());
        stream.extend(pages[2].iter().copied());
        stream.extend_from_slice(&ogg_page_with(960, serial, 0, &[0_u8; 1]));
        for page in pages.iter().skip(3) {
            stream.extend(page.iter().copied());
        }
        fs::write(&path, &stream).map_err(|err| err.to_string())?;
        assert_eq!(
            probe_audio(&path)
                .map_err(|err| err.to_string())?
                .stream
                .total_samples(),
            Some(2_568)
        );
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }
}
