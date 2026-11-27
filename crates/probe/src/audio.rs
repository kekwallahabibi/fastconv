use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::num::{NonZeroU16, NonZeroU32, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::thread;

use fastconv_domain::{
    AudioBatchItem, AudioBatchPlan, AudioBitDepth, AudioEncodingOptions, AudioExecutionMode,
    AudioInputFormat, AudioOptionBitrate, AudioOptionChannels, AudioOptionSampleRate,
    AudioOutputFormat, AudioQuality, AudioRateControl, AudioSampleFormat, AudioStreamInfo,
    AudioTargetCodec, DomainError, HardwareBackend, PathRole,
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
        && let Some(identification) = ogg_identification_packet(bytes)
    {
        if identification.starts_with(b"OpusHead") {
            return Some(AudioInputFormat::Opus);
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
fn ogg_identification_packet(bytes: &[u8]) -> Option<&[u8]> {
    const TABLE_START: usize = 27;
    let limit = bytes.len().min(MAGIC_PREFIX_BYTES);
    let mut offset = 0_usize;
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
                return Some(packet);
            }
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

fn probe_opus(path: &Path, file: File) -> Result<AudioStreamInfo, ProbeError> {
    let reader = opus_pure::OggOpusReader::new(file)
        .map_err(|err| invalid_audio(path, format!("invalid Ogg Opus stream: {err}")))?;
    let head = reader.head();
    let channels = NonZeroU16::new(u16::from(head.channel_count))
        .ok_or_else(|| invalid_audio(path, "Opus channel count must not be zero"))?;
    // Ogg Opus always decodes at 48 kHz, regardless of the rate recorded at capture time.
    let sample_rate = NonZeroU32::new(48_000).unwrap_or(NonZeroU32::MIN);
    AudioStreamInfo::new(channels, sample_rate, NonZeroU16::new(16), None)
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
    AudioStreamInfo::new(channels, sample_rate, None, None)
        .map_err(|err| invalid_audio(path, err.to_string()))
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
}
