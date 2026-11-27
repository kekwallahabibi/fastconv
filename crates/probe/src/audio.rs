use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::num::{NonZeroU16, NonZeroU32, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::thread;

use fastconv_domain::{
    AudioBatchItem, AudioBatchPlan, AudioBitrate, AudioChannels, AudioEncodingOptions,
    AudioExecutionMode, AudioInputFormat, AudioOutputFormat, AudioQuality, AudioRateControl,
    AudioSampleRate, AudioStreamInfo, DomainError, HardwareBackend, PathRole,
};
use symphonia::core::codecs::audio::{AudioDecoderOptions, well_known::CODEC_ID_ALAC};
use symphonia::core::formats::FormatOptions;
use symphonia::core::formats::TrackType;
use symphonia::core::formats::probe::Hint;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

use crate::ProbeError;

const MAGIC_PREFIX_BYTES: usize = 4096;
const SUPPORTED_AUDIO_TARGETS: &str = "flac, wav, opus, ogg, mp3, aac, m4a, alac, or aiff";

/// Raw encoder knobs collected at the CLI boundary before the output format is known.
///
/// Field-level validity is established by the domain constructors; compatibility with the
/// resolved output format is checked once in [`prepare_audio_batch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AudioEncodingRequest {
    pub rate_control: AudioRateControl,
    pub sample_rate: Option<AudioSampleRate>,
    pub channels: Option<AudioChannels>,
}

const BITRATE_PARSE_EXPECTED: &str =
    "a positive rate such as 128k, 192k, 320k, 128kbps, 128, or 128000";

/// Parses the raw `--bitrate` value at the system boundary into a validated domain rate.
///
/// A `k`/`kbps` suffix (case-insensitive) means kbps, a bare value below `1_000` is kbps,
/// and a bare value of at least `1_000` is bps.
///
/// # Errors
///
/// Returns [`DomainError::InvalidAudioBitrate`] for empty, zero, non-digit, overflowing,
/// or otherwise malformed input.
pub fn parse_audio_bitrate(raw: &str) -> Result<AudioBitrate, DomainError> {
    let invalid = || DomainError::InvalidAudioBitrate {
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
    AudioBitrate::new(bps)
}

/// Parses and validates the four raw encoding flag values into an [`AudioEncodingRequest`].
///
/// This is the single boundary where the CLI's raw `--bitrate`, `--quality`,
/// `--sample-rate`, and `--channels` strings become domain types; the CLI only collects
/// raw values and flag presence.
///
/// # Errors
///
/// Returns a diagnostic for a bitrate/quality conflict or any malformed flag value.
pub fn parse_audio_encoding_request(
    bitrate: Option<&str>,
    quality: Option<&str>,
    sample_rate: Option<&str>,
    channels: Option<&str>,
) -> Result<AudioEncodingRequest, String> {
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
                AudioSampleRate::new(hz)
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
                AudioChannels::new(count)
                    .map_err(|err| format!("invalid --channels '{raw}': {err}"))?,
            )
        }
        None => None,
    };
    Ok(AudioEncodingRequest {
        rate_control,
        sample_rate,
        channels,
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

/// Extracts the first packet from the leading Ogg page. Codec must be read from this framed
/// identification packet; scanning arbitrary bytes can mistake Vorbis comments for Opus.
fn ogg_identification_packet(bytes: &[u8]) -> Option<&[u8]> {
    if !bytes.starts_with(b"OggS") || bytes.len() < 28 {
        return None;
    }
    let segments = usize::from(*bytes.get(26)?);
    let table_start = 27;
    let table = bytes.get(table_start..table_start + segments)?;
    let mut packet_len = 0_usize;
    for &lacing in table {
        packet_len += usize::from(lacing);
        if lacing < 255 {
            break;
        }
    }
    let payload_start = table_start + segments;
    bytes.get(payload_start..payload_start + packet_len)
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
    let bits_per_sample = params
        .bits_per_sample
        .and_then(|bits| NonZeroU16::new(u16::try_from(bits).unwrap_or(u16::MAX)));
    let stream = AudioStreamInfo::new(channels, sample_rate, bits_per_sample, track.num_frames)
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
            "unrecognized audio magic bytes (expected WAV, FLAC, MP3, Ogg Vorbis, Ogg Opus, AAC, M4A, ALAC, or AIFF)",
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
    format: AudioOutputFormat,
    threads: NonZeroUsize,
    options: AudioEncodingOptions,
) -> Result<AudioBatchPlan, ProbeError> {
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

    let mode = AudioExecutionMode::resolve(format, HardwareBackend::Cpu)?;
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
        AudioExecutionMode::resolve(format, backend)?;
        // Validate encoding options before touching any input file in the batch.
        let options = AudioEncodingOptions::new(
            format,
            request.rate_control,
            request.sample_rate,
            request.channels,
        )?;
        return prepare_directory_batch(source, target, format, threads, options);
    }
    if !source_meta.is_file() {
        return Err(ProbeError::NotAFileOrDirectory(source.to_path_buf()));
    }

    let (resolved_target, format) = resolve_single_target(source, target, to)?;
    let options = AudioEncodingOptions::new(
        format,
        request.rate_control,
        request.sample_rate,
        request.channels,
    )?;
    let canonical_source = canonical_or_parent(source)?;
    let canonical_target = canonical_or_parent(&resolved_target)?;
    if canonical_source == canonical_target {
        return Err(DomainError::IdenticalSourceAndTarget(source.to_path_buf()).into());
    }
    let mode = AudioExecutionMode::resolve(format, backend)?;
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
    use fastconv_domain::AudioBitrate;

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
                    Err(DomainError::InvalidAudioBitrate { .. })
                ),
                "'{raw}' must be rejected"
            );
        }
        Ok(())
    }

    #[test]
    fn parses_raw_encoding_request_and_rejects_conflicts() -> Result<(), String> {
        let expected = AudioEncodingRequest {
            rate_control: AudioRateControl::Bitrate(
                AudioBitrate::new(192_000).map_err(|err| err.to_string())?,
            ),
            sample_rate: Some(AudioSampleRate::new(48_000).map_err(|err| err.to_string())?),
            channels: Some(AudioChannels::new(2).map_err(|err| err.to_string())?),
        };
        assert_eq!(
            parse_audio_encoding_request(Some("192k"), None, Some("48000"), Some("2"))?,
            expected
        );
        assert_eq!(
            parse_audio_encoding_request(None, None, None, None)?,
            AudioEncodingRequest::default()
        );
        assert_eq!(
            parse_audio_encoding_request(None, Some("85"), None, None)?.rate_control,
            AudioRateControl::Quality(AudioQuality::new(85).map_err(|err| err.to_string())?)
        );

        for (bitrate, quality) in [(Some("128k"), Some("80")), (Some("128k"), Some("101"))] {
            let Err(err) = parse_audio_encoding_request(bitrate, quality, None, None) else {
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
            let Err(err) = parse_audio_encoding_request(args.0, args.1, args.2, args.3) else {
                return Err(format!("malformed value must be rejected: {args:?}"));
            };
            assert!(err.contains(needle), "expected '{needle}' in '{err}'");
        }
        Ok(())
    }

    fn bitrate_request(bps: u32) -> AudioEncodingRequest {
        AudioEncodingRequest {
            rate_control: AudioRateControl::Bitrate(
                AudioBitrate::new(bps).unwrap_or_else(|_| unreachable!()),
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
            sample_rate: Some(AudioSampleRate::new(44_100).unwrap_or_else(|_| unreachable!())),
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
                channels: Some(AudioChannels::new(4).unwrap_or_else(|_| unreachable!())),
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
}
