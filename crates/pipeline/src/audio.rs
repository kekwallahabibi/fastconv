//! Native audio backends. FLAC uses `flacenc` and WAV uses `hound`; the lossy encoder is
//! `opus-pure`, a dependency-free pure-Rust RFC 6716/Ogg Opus implementation. C-backed crates
//! such as `libopus`/`vorbis_rs` are deliberately avoided so CI needs no system audio libraries.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::num::{NonZeroU8, NonZeroU32, NonZeroU64, NonZeroUsize};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use fastconv_domain::{
    AudioBatchItem, AudioBatchPlan, AudioConversionReport, AudioInputFormat, AudioOutputFormat,
    HardwareBackend,
};
use flacenc::bitsink::ByteSink;
use flacenc::component::BitRepr;
use flacenc::error::{SourceError, Verify};
use flacenc::source::{Fill, Source};
use hound::{SampleFormat, WavSpec, WavWriter};
use opus_pure::{
    Application, DEFAULT_WARMUP_MS, MAX_PACKET_BYTES, MAX_PACKET_SAMPLES, OggOpusReader,
    OggOpusWriter, OpusEncoder, OpusHead, ParallelConfig, Trim, encode_parallel,
};
use rubato::{FftFixedInOut, Resampler};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::formats::TrackType;
use symphonia::core::formats::probe::Hint;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

use crate::PipelineError;

const FLAC_BLOCK_SIZE: usize = 4096;
const OPUS_RATE: u32 = 48_000;
const OPUS_FRAME_MS: u32 = 20;
const OPUS_BITRATE: i32 = 128_000;
const RESAMPLE_CHUNK: usize = 1024;
const WAV_READ_BUF_BYTES: usize = 256 * 1024;
const WAV_WRITE_BUF_BYTES: usize = 256 * 1024;
const OPUS_WRITE_BUF_BYTES: usize = 64 * 1024;
const MAX_PCM_BYTES_PER_FILE: u64 = 512 * 1024 * 1024;
const MAX_PCM_BYTES_PER_BATCH: u64 = 1024 * 1024 * 1024;

/// Per-worker scratch space. All buffers are retained across files so a batch does not
/// reallocate its decode/encode working set for every item.
#[derive(Debug)]
pub struct AudioWorkerBuffer {
    samples_f32: Vec<f32>,
    samples_i16: Vec<i16>,
    samples_i32: Vec<i32>,
    resampled_f32: Vec<f32>,
    packet: Vec<u8>,
}

impl Default for AudioWorkerBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioWorkerBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            samples_f32: Vec::with_capacity(OPUS_RATE as usize * 2),
            samples_i16: Vec::with_capacity(OPUS_RATE as usize * 2),
            samples_i32: Vec::with_capacity(OPUS_RATE as usize * 2),
            resampled_f32: Vec::new(),
            packet: vec![0_u8; MAX_PACKET_BYTES],
        }
    }

    fn reset(&mut self) {
        self.samples_f32.clear();
        self.samples_i16.clear();
        self.samples_i32.clear();
    }

    #[cfg(test)]
    fn decoded_len(&self) -> usize {
        self.samples_f32.len()
    }

    #[cfg(test)]
    fn capacity_of_samples(&self) -> usize {
        self.samples_i16.capacity()
    }
}

fn usize_from(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

fn to_i32_saturating(value: f32) -> i32 {
    // The caller clamps to the target format's range before rounding; the allow covers only the
    // final float-to-int narrowing, which is well-defined (saturating) in Rust since 1.45.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let narrowed = value.round() as i32;
    narrowed
}

fn quantize_i8(sample: f32) -> i8 {
    let value = to_i32_saturating(sample.clamp(-1.0, 1.0) * 128.0).clamp(-128, 127);
    i8::try_from(value).unwrap_or(if value < 0 { i8::MIN } else { i8::MAX })
}

fn quantize_i16(sample: f32) -> i16 {
    let value = to_i32_saturating(sample.clamp(-1.0, 1.0) * 32_768.0).clamp(-32_768, 32_767);
    i16::try_from(value).unwrap_or(if value < 0 { i16::MIN } else { i16::MAX })
}

fn quantize_i24(sample: f32) -> i32 {
    to_i32_saturating(sample.clamp(-1.0, 1.0) * 8_388_608.0).clamp(-8_388_608, 8_388_607)
}

fn decode_opus(path: &Path, buf: &mut AudioWorkerBuffer) -> Result<(u32, usize), PipelineError> {
    let file = File::open(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = OggOpusReader::new(file).map_err(|err| PipelineError::DecodeFailed {
        path: path.to_path_buf(),
        reason: format!("invalid Ogg Opus stream: {err}"),
    })?;
    let head = reader.head().clone();
    let channels = usize::from(head.channel_count);
    if channels == 0 {
        return Err(PipelineError::DecodeFailed {
            path: path.to_path_buf(),
            reason: "Opus stream has zero channels".to_owned(),
        });
    }
    let rate = i32::try_from(OPUS_RATE).unwrap_or(i32::MAX);
    let mut decoder = head
        .decoder(rate)
        .map_err(|err| PipelineError::DecodeFailed {
            path: path.to_path_buf(),
            reason: format!("failed to create Opus decoder: {err}"),
        })?;
    let mut trim = Trim::new(&head, rate, channels).map_err(|err| PipelineError::DecodeFailed {
        path: path.to_path_buf(),
        reason: format!("failed to create Opus trim state: {err}"),
    })?;
    let mut block = vec![0_f32; MAX_PACKET_SAMPLES * channels];
    for packet in reader.packets() {
        let packet = packet.map_err(|err| PipelineError::DecodeFailed {
            path: path.to_path_buf(),
            reason: format!("failed to read Opus packet: {err}"),
        })?;
        let produced = decoder
            .decode(&packet.data, MAX_PACKET_SAMPLES, &mut block)
            .map_err(|err| PipelineError::DecodeFailed {
                path: path.to_path_buf(),
                reason: format!("failed to decode Opus frame: {err}"),
            })?;
        let end = produced.saturating_mul(channels).min(block.len());
        let kept = trim.keep(&packet, &block[..end]);
        buf.samples_f32.extend_from_slice(kept);
    }
    Ok((OPUS_RATE, channels))
}

/// Returns the byte offset of the first ADTS frame, skipping a leading `ID3v2` tag.
fn skip_id3v2(path: &Path, data: &[u8]) -> Result<usize, PipelineError> {
    if !data.starts_with(b"ID3") {
        return Ok(0);
    }
    let fail = |reason: String| PipelineError::DecodeFailed {
        path: path.to_path_buf(),
        reason,
    };
    if data.len() < 10 || data[3] == 0xFF || data[4] == 0xFF {
        return Err(fail("invalid ID3v2 tag before AAC frame".to_owned()));
    }
    if data[6..10].iter().any(|&byte| byte & 0x80 != 0) {
        return Err(fail(
            "invalid ID3v2 synchsafe size before AAC frame".to_owned(),
        ));
    }
    let payload = (usize::from(data[6]) << 21)
        | (usize::from(data[7]) << 14)
        | (usize::from(data[8]) << 7)
        | usize::from(data[9]);
    let footer = if data[5] & 0x10 != 0 { 10 } else { 0 };
    let pos = 10_usize.saturating_add(payload).saturating_add(footer);
    if pos > data.len() {
        return Err(fail("ID3v2 tag extends past end of AAC file".to_owned()));
    }
    Ok(pos)
}

fn decode_adts_aac(
    path: &Path,
    buf: &mut AudioWorkerBuffer,
) -> Result<(u32, usize), PipelineError> {
    // Reject oversized input before `fs::read` so a malicious/oversized file cannot spike RAM
    // ahead of the incremental PCM limit enforced in the decode loop below.
    let file_len = fs::metadata(path)
        .map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if file_len > MAX_PCM_BYTES_PER_FILE {
        return Err(PipelineError::DecodeFailed {
            path: path.to_path_buf(),
            reason: format!(
                "AAC input of {file_len} bytes exceeds the {MAX_PCM_BYTES_PER_FILE}-byte per-file limit"
            ),
        });
    }
    let data = fs::read(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut pos = skip_id3v2(path, &data)?;
    let mut decoder = rusty_aac::AacDecoder::new();
    let mut header_rate = 0_u32;
    let mut header_channels = 0_usize;
    let mut decoded_rate = 0_u32;
    let mut decoded_channels = 0_usize;
    while pos < data.len() {
        let header =
            rusty_aac::parse_adts(&data[pos..]).map_err(|err| PipelineError::DecodeFailed {
                path: path.to_path_buf(),
                reason: format!("invalid ADTS frame at byte {pos}: {err}"),
            })?;
        if header.frame_length < header.header_len {
            return Err(PipelineError::DecodeFailed {
                path: path.to_path_buf(),
                reason: format!("ADTS frame at byte {pos} has an invalid length"),
            });
        }
        let end = pos.saturating_add(header.frame_length);
        if end > data.len() {
            return Err(PipelineError::DecodeFailed {
                path: path.to_path_buf(),
                reason: format!("ADTS frame at byte {pos} extends past end of file"),
            });
        }
        if header_rate == 0 {
            header_rate = header.sample_rate;
            header_channels = usize::from(header.channels);
        }
        let frame =
            decoder
                .decode(&data[pos..end], None)
                .map_err(|err| PipelineError::DecodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("failed to decode ADTS frame at byte {pos}: {err}"),
                })?;
        if decoded_rate == 0 && frame.sample_rate != 0 {
            decoded_rate = frame.sample_rate;
            decoded_channels = usize::from(frame.channels);
        }
        let pcm_bytes = u64::try_from(buf.samples_f32.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(4)
            .saturating_add(
                u64::try_from(frame.samples.len())
                    .unwrap_or(u64::MAX)
                    .saturating_mul(4),
            );
        if pcm_bytes > MAX_PCM_BYTES_PER_FILE {
            return Err(PipelineError::DecodeFailed {
                path: path.to_path_buf(),
                reason: format!(
                    "decoded PCM exceeds the {MAX_PCM_BYTES_PER_FILE}-byte per-file limit"
                ),
            });
        }
        buf.samples_f32.extend_from_slice(&frame.samples);
        pos = end;
    }
    let rate = if decoded_rate != 0 {
        decoded_rate
    } else {
        header_rate
    };
    let channels = if decoded_channels != 0 {
        decoded_channels
    } else {
        header_channels
    };
    if rate == 0 || channels == 0 {
        return Err(PipelineError::DecodeFailed {
            path: path.to_path_buf(),
            reason: "ADTS stream has no decodable audio frames".to_owned(),
        });
    }
    Ok((rate, channels))
}

// The decode loop and buffer reservation must stay together to keep the direct-i16
// decision and its borrow of `buf` in one place.
#[allow(clippy::too_many_lines)]
fn decode_symphonia(
    path: &Path,
    format: AudioInputFormat,
    target: AudioOutputFormat,
    buf: &mut AudioWorkerBuffer,
) -> Result<(u32, usize, usize), PipelineError> {
    let file = File::open(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
    let mut hint = Hint::new();
    hint.with_extension(format.extension());
    let mut reader = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|err| PipelineError::DecodeFailed {
            path: path.to_path_buf(),
            reason: format!("failed to read audio container: {err}"),
        })?;
    let track =
        reader
            .default_track(TrackType::Audio)
            .ok_or_else(|| PipelineError::DecodeFailed {
                path: path.to_path_buf(),
                reason: "no audio track found".to_owned(),
            })?;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| PipelineError::DecodeFailed {
            path: path.to_path_buf(),
            reason: "audio track has no codec parameters".to_owned(),
        })?
        .clone();
    let sample_rate = params.sample_rate.unwrap_or(0);
    let channels = params
        .channels
        .as_ref()
        .map_or(0, symphonia::core::audio::Channels::count);
    let bits = usize::try_from(params.bits_per_sample.unwrap_or(0)).unwrap_or(0);
    if sample_rate == 0 || channels == 0 {
        return Err(PipelineError::DecodeFailed {
            path: path.to_path_buf(),
            reason: format!("invalid decoded stream format: {sample_rate} Hz, {channels} channels"),
        });
    }
    let track_id = track.id;
    let total_samples = track.num_frames;
    // Symphonia left-aligns 16-bit integer samples when converting to i32 (i16 << 16),
    // so a direct i16 copy is both faster and bit-exact for lossless targets.
    let integer_pcm = !matches!(
        params.sample_format,
        Some(
            symphonia::core::audio::sample::SampleFormat::F32
                | symphonia::core::audio::sample::SampleFormat::F64,
        )
    );
    let direct_i16 = matches!(
        target,
        AudioOutputFormat::Wav
            | AudioOutputFormat::Flac
            | AudioOutputFormat::Aiff
            | AudioOutputFormat::Alac
    ) && bits == 16
        && integer_pcm;
    let sample_bytes = if direct_i16 { 2_u64 } else { 4_u64 };
    let max_samples = usize::try_from(MAX_PCM_BYTES_PER_FILE / sample_bytes).unwrap_or(usize::MAX);
    let reserve = total_samples
        .map_or(0, |frames| usize::try_from(frames).unwrap_or(usize::MAX))
        .saturating_mul(channels)
        .min(max_samples);
    if direct_i16 {
        if buf.samples_i16.capacity() < reserve {
            buf.samples_i16
                .reserve(reserve.saturating_sub(buf.samples_i16.len()));
        }
    } else if buf.samples_f32.capacity() < reserve {
        buf.samples_f32
            .reserve(reserve.saturating_sub(buf.samples_f32.len()));
    }
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|err| PipelineError::DecodeFailed {
            path: path.to_path_buf(),
            reason: format!("failed to create audio decoder: {err}"),
        })?;

    loop {
        let packet = match reader.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(err) => {
                return Err(PipelineError::DecodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("failed to read audio packet: {err}"),
                });
            }
        };
        if packet.track_id != track_id {
            continue;
        }
        let audio_frame = match decoder.decode(&packet) {
            Ok(frame) => frame,
            Err(err) => {
                return Err(PipelineError::DecodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("failed to decode audio packet: {err}"),
                });
            }
        };
        let count = audio_frame.samples_interleaved();
        let start = if direct_i16 {
            buf.samples_i16.len()
        } else {
            buf.samples_f32.len()
        };
        let end = start.saturating_add(count);
        if end > max_samples {
            return Err(PipelineError::DecodeFailed {
                path: path.to_path_buf(),
                reason: format!(
                    "decoded PCM exceeds the {MAX_PCM_BYTES_PER_FILE}-byte per-file limit"
                ),
            });
        }
        if direct_i16 {
            buf.samples_i16.resize(end, 0);
            if let Some(slot) = buf.samples_i16.get_mut(start..end) {
                audio_frame.copy_to_slice_interleaved::<i16, _>(slot);
            }
        } else {
            buf.samples_f32.resize(end, 0.0);
            if let Some(slot) = buf.samples_f32.get_mut(start..end) {
                audio_frame.copy_to_slice_interleaved(slot);
            }
        }
    }
    Ok((sample_rate, channels, bits))
}

fn decode_audio(
    item: &AudioBatchItem,
    target: AudioOutputFormat,
    buf: &mut AudioWorkerBuffer,
) -> Result<(u32, usize, usize), PipelineError> {
    buf.reset();
    let (rate, channels, source_bits) = match item.input_format() {
        AudioInputFormat::Opus => {
            let (rate, channels) = decode_opus(item.source(), buf)?;
            (rate, channels, 16)
        }
        AudioInputFormat::Aac => {
            let (rate, channels) = decode_adts_aac(item.source(), buf)?;
            (rate, channels, 16)
        }
        _ => decode_symphonia(item.source(), item.input_format(), target, buf)?,
    };

    if channels == 0 {
        return Err(PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: "decoded audio has zero channels".to_owned(),
        });
    }
    if buf.samples_f32.is_empty() && buf.samples_i16.is_empty() {
        return Err(PipelineError::EmptyOutput(item.source().to_path_buf()));
    }

    Ok((rate, channels, source_bits))
}

fn quantize_for(target: AudioOutputFormat, bits: usize, buf: &mut AudioWorkerBuffer) {
    buf.samples_i16.clear();
    buf.samples_i32.clear();
    match (target, bits) {
        (AudioOutputFormat::Wav | AudioOutputFormat::Aiff, 16 | 0)
        | (AudioOutputFormat::Alac, 8 | 16 | 0) => {
            buf.samples_i16
                .extend(buf.samples_f32.iter().copied().map(quantize_i16));
        }
        (AudioOutputFormat::Wav | AudioOutputFormat::Aiff, 8) => {
            buf.samples_i16.extend(
                buf.samples_f32
                    .iter()
                    .copied()
                    .map(|s| i16::from(quantize_i8(s))),
            );
        }
        (AudioOutputFormat::Wav | AudioOutputFormat::Aiff | AudioOutputFormat::Flac, 24) => {
            buf.samples_i32
                .extend(buf.samples_f32.iter().copied().map(quantize_i24));
        }
        (AudioOutputFormat::Flac, 8) => {
            buf.samples_i32.extend(
                buf.samples_f32
                    .iter()
                    .copied()
                    .map(|s| i32::from(quantize_i8(s))),
            );
        }
        (AudioOutputFormat::Flac, _) => {
            buf.samples_i32.extend(
                buf.samples_f32
                    .iter()
                    .copied()
                    .map(|s| i32::from(quantize_i16(s))),
            );
        }
        // Lossy targets keep their f32 samples; unsupported depths are rejected before this call.
        _ => {}
    }
    // Only one PCM representation is needed after quantization; release the float copy so peak
    // memory stays bounded to a single full-file sample array per active worker.
    buf.samples_f32.clear();
}

const WAVE_FORMAT_PCM_GUID: [u8; 16] = [
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

fn le_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn le_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// A validated signed-integer PCM `data` chunk in a RIFF/WAVE file.
struct PcmWavStream {
    reader: BufReader<File>,
    channels: usize,
    bits: usize,
    rate: u32,
    frame_bytes: usize,
    total_samples: usize,
    remaining_bytes: u64,
}

/// Opens a plain signed 16/24-bit PCM WAV for direct FLAC encoding.
///
/// Returns `Ok(None)` for anything that needs the full decoder: unsigned 8-bit
/// WAV, float WAV, RF64, non-canonical extensible PCM, or malformed chunks.
fn try_open_pcm_wav(path: &Path) -> Result<Option<PcmWavStream>, PipelineError> {
    let io_err = |source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    };
    let mut file = File::open(path).map_err(io_err)?;
    let file_len = file.metadata().map_err(io_err)?.len();
    let mut head = [0_u8; 12];
    if file.read_exact(&mut head).is_err() {
        return Ok(None);
    }
    if &head[..4] != b"RIFF" || &head[8..12] != b"WAVE" {
        return Ok(None);
    }
    if le_u32(&head, 4) == u32::MAX {
        return Ok(None);
    }
    let riff_end = u64::from(le_u32(&head, 4)).saturating_add(8).min(file_len);

    let mut pos = 12_u64;
    let mut fmt: Option<(usize, u32, usize, usize)> = None;
    let mut data: Option<(u64, u64)> = None;
    let mut header = [0_u8; 8];
    let mut payload = vec![0_u8; 4096];

    while pos.saturating_add(8) <= riff_end {
        if file.seek(SeekFrom::Start(pos)).is_err() || file.read_exact(&mut header).is_err() {
            return Ok(None);
        }
        let size = u64::from(le_u32(&header, 4));
        if size == u64::from(u32::MAX) {
            return Ok(None);
        }
        if pos.saturating_add(8).saturating_add(size) > riff_end {
            return Ok(None);
        }
        let padded = size.saturating_add(size & 1);
        match &header[..4] {
            b"fmt " => {
                if !(16..=4096).contains(&size) {
                    return Ok(None);
                }
                let len = usize::try_from(size).unwrap_or(0);
                payload.resize(len, 0);
                if file.read_exact(&mut payload[..len]).is_err() {
                    return Ok(None);
                }
                let audio_format = le_u16(&payload, 0);
                let channels = usize::from(le_u16(&payload, 2));
                let rate = le_u32(&payload, 4);
                let block_align = usize::from(le_u16(&payload, 12));
                let bits = usize::from(le_u16(&payload, 14));
                let canonical = bits == 16 || bits == 24;
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
                    return Ok(None);
                }
                fmt = Some((channels, rate, bits, block_align));
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

    let (Some((channels, rate, bits, block_align)), Some((data_offset, data_bytes))) = (fmt, data)
    else {
        return Ok(None);
    };
    let align = u64::try_from(block_align).unwrap_or(u64::MAX);
    if data_bytes == 0
        || data_offset.saturating_add(data_bytes) > riff_end
        || data_bytes % align != 0
    {
        return Ok(None);
    }
    file.seek(SeekFrom::Start(data_offset)).map_err(io_err)?;
    Ok(Some(PcmWavStream {
        reader: BufReader::with_capacity(WAV_READ_BUF_BYTES, file),
        channels,
        bits,
        rate,
        frame_bytes: block_align,
        total_samples: usize::try_from(data_bytes / align).unwrap_or(usize::MAX),
        remaining_bytes: data_bytes,
    }))
}

/// Streams little-endian PCM bytes straight into flacenc's `Fill` target.
struct LeBytesFlacSource {
    reader: BufReader<File>,
    read_buf: Vec<u8>,
    channels: usize,
    bits: usize,
    rate: usize,
    frame_bytes: usize,
    total_samples: usize,
    remaining_bytes: u64,
}

impl Source for LeBytesFlacSource {
    fn channels(&self) -> usize {
        self.channels
    }

    fn bits_per_sample(&self) -> usize {
        self.bits
    }

    fn sample_rate(&self) -> usize {
        self.rate
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.total_samples)
    }

    fn read_samples<F: Fill>(
        &mut self,
        block_size: usize,
        dest: &mut F,
    ) -> Result<usize, SourceError> {
        let available =
            usize::try_from(self.remaining_bytes).unwrap_or(usize::MAX) / self.frame_bytes.max(1);
        let frames = block_size.min(available);
        if frames == 0 {
            return Ok(0);
        }
        let wanted = frames.saturating_mul(self.frame_bytes);
        if self.read_buf.len() < wanted {
            self.read_buf.resize(wanted, 0);
        }
        let mut filled = 0_usize;
        while filled < wanted {
            match self.reader.read(&mut self.read_buf[filled..wanted]) {
                Ok(0) => {
                    return Err(SourceError::from_io_error(io::Error::from(
                        io::ErrorKind::UnexpectedEof,
                    )));
                }
                Ok(count) => filled += count,
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) => return Err(SourceError::from_io_error(err)),
            }
        }
        let frames = filled / self.frame_bytes;
        let used = frames.saturating_mul(self.frame_bytes);
        self.remaining_bytes = self
            .remaining_bytes
            .saturating_sub(u64::try_from(used).unwrap_or(u64::MAX));
        dest.fill_le_bytes(&self.read_buf[..used], self.bits / 8)?;
        Ok(frames)
    }
}

/// Borrows decoded `i32` samples instead of cloning them via `MemSource`.
struct BorrowedI32FlacSource<'a> {
    samples: &'a [i32],
    read_head: usize,
    channels: usize,
    bits: usize,
    rate: usize,
}

impl Source for BorrowedI32FlacSource<'_> {
    fn channels(&self) -> usize {
        self.channels
    }

    fn bits_per_sample(&self) -> usize {
        self.bits
    }

    fn sample_rate(&self) -> usize {
        self.rate
    }

    // flacenc calls `len_hint` both before and after consuming the source, so
    // the declared length stays fixed while only `read_head` advances.
    fn len_hint(&self) -> Option<usize> {
        Some(self.samples.len() / self.channels.max(1))
    }

    fn read_samples<F: Fill>(
        &mut self,
        block_size: usize,
        dest: &mut F,
    ) -> Result<usize, SourceError> {
        let remaining = self.samples.len().saturating_sub(self.read_head);
        let take = block_size.saturating_mul(self.channels).min(remaining);
        let frames = take / self.channels.max(1);
        let used = frames.saturating_mul(self.channels);
        dest.fill_interleaved(&self.samples[self.read_head..self.read_head + used])?;
        self.read_head = self.read_head.saturating_add(used);
        Ok(frames)
    }
}

/// Keeps flacenc's parallel feeder running after a source read failure so its
/// workers are always stopped and joined; the failure is reported afterwards.
struct EofTolerantFlacSource<S> {
    inner: S,
    failed: Arc<AtomicBool>,
    remaining: usize,
}

impl<S: Source> EofTolerantFlacSource<S> {
    fn new(inner: S, failed: Arc<AtomicBool>) -> Self {
        let remaining = inner.len_hint().unwrap_or(0);
        Self {
            inner,
            failed,
            remaining,
        }
    }
}

impl<S: Source> Source for EofTolerantFlacSource<S> {
    fn channels(&self) -> usize {
        self.inner.channels()
    }

    fn bits_per_sample(&self) -> usize {
        self.inner.bits_per_sample()
    }

    fn sample_rate(&self) -> usize {
        self.inner.sample_rate()
    }

    fn len_hint(&self) -> Option<usize> {
        self.inner.len_hint()
    }

    fn read_samples<F: Fill>(
        &mut self,
        block_size: usize,
        dest: &mut F,
    ) -> Result<usize, SourceError> {
        if let Ok(frames) = self.inner.read_samples(block_size, dest) {
            self.remaining = self.remaining.saturating_sub(frames);
            Ok(frames)
        } else {
            // Never propagate the error into flacenc's feeder: it would skip
            // request_stop() and leave all encoder workers blocked forever.
            self.failed.store(true, Ordering::Relaxed);
            let frames = self.remaining.min(block_size);
            self.remaining -= frames;
            if frames > 0 {
                let zeros = vec![
                    0_u8;
                    frames
                        .saturating_mul(self.channels())
                        .saturating_mul(self.bits_per_sample() / 8)
                ];
                dest.fill_le_bytes(&zeros, self.bits_per_sample() / 8)?;
            }
            Ok(frames)
        }
    }
}

fn encode_flac_from_source<S: Source>(
    path: &Path,
    source: S,
    batch_workers: usize,
    source_failed: Option<&AtomicBool>,
) -> Result<(), PipelineError> {
    let pcm_bytes = source
        .len_hint()
        .unwrap_or(0)
        .saturating_mul(source.channels())
        .saturating_mul(source.bits_per_sample() / 8);
    let mut config = flacenc::config::Encoder::default();
    config.block_size = FLAC_BLOCK_SIZE;
    // The quantized LPC search dominates runtime for only a small compression gain.
    config.subframe_coding.use_lpc = false;
    let available = std::thread::available_parallelism()
        .map_or(1, NonZeroUsize::get)
        .min(8);
    // A batch already saturates the host; only spend idle capacity on inner workers.
    let inner_threads = (available / batch_workers.max(1)).max(1);
    config.multithread = inner_threads > 1;
    config.workers = if config.multithread {
        NonZeroUsize::new(inner_threads)
    } else {
        None
    };
    let config = config
        .into_verified()
        .map_err(|(_, err)| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("FLAC config invalid: {err}"),
        })?;
    let mut stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("FLAC encoding failed: {err}"),
        })?;
    stream
        .stream_info_mut()
        .set_block_sizes(config.block_size, config.block_size)
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("FLAC stream-info update failed: {err}"),
        })?;
    let max_sink_bytes = usize::try_from(MAX_PCM_BYTES_PER_FILE).unwrap_or(usize::MAX);
    let estimated_bytes =
        ((pcm_bytes / 10).saturating_mul(6).saturating_add(65_536)).min(max_sink_bytes);
    let mut sink = ByteSink::with_capacity(estimated_bytes.saturating_mul(8).max(4096));
    stream
        .write(&mut sink)
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("FLAC bitstream write failed: {err}"),
        })?;
    if source_failed.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        return Err(PipelineError::DecodeFailed {
            path: path.to_path_buf(),
            reason: "WAV source ended before all declared PCM samples were read".to_owned(),
        });
    }
    fs::write(path, sink.as_slice()).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn encode_flac(
    path: &Path,
    rate: u32,
    channels: usize,
    bits: usize,
    buf: &AudioWorkerBuffer,
    batch_workers: usize,
) -> Result<(), PipelineError> {
    let source = BorrowedI32FlacSource {
        samples: &buf.samples_i32,
        read_head: 0,
        channels,
        bits,
        rate: usize_from(rate),
    };
    encode_flac_from_source(path, source, batch_workers, None)
}

fn encode_wav(
    path: &Path,
    rate: u32,
    channels: usize,
    bits: usize,
    buf: &AudioWorkerBuffer,
) -> Result<(), PipelineError> {
    let channel_count = u16::try_from(channels).map_err(|_| PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason: format!("WAV supports at most 65535 channels, got {channels}"),
    })?;
    let spec = WavSpec {
        channels: channel_count,
        sample_rate: rate,
        bits_per_sample: u16::try_from(bits).unwrap_or(16),
        sample_format: SampleFormat::Int,
    };
    let file = File::create(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = WavWriter::new(BufWriter::with_capacity(WAV_WRITE_BUF_BYTES, file), spec)
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("failed to create WAV writer: {err}"),
        })?;
    if bits == 24 {
        for &sample in &buf.samples_i32 {
            writer
                .write_sample(sample)
                .map_err(|err| PipelineError::EncodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("WAV sample write failed: {err}"),
                })?;
        }
    } else if bits == 8 {
        for &sample in &buf.samples_i16 {
            let value = i8::try_from(sample).unwrap_or(if sample < 0 { i8::MIN } else { i8::MAX });
            writer
                .write_sample(value)
                .map_err(|err| PipelineError::EncodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("WAV sample write failed: {err}"),
                })?;
        }
    } else {
        const CHUNK: usize = 65_536;
        let samples = &buf.samples_i16;
        for chunk in samples.chunks(CHUNK) {
            let mut fast = writer.get_i16_writer(chunk.len().try_into().unwrap_or(u32::MAX));
            for &sample in chunk {
                fast.write_sample(sample);
            }
            fast.flush().map_err(|err| PipelineError::EncodeFailed {
                path: path.to_path_buf(),
                reason: format!("WAV sample write failed: {err}"),
            })?;
        }
    }
    writer
        .finalize()
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("failed to finalize WAV: {err}"),
        })
}

// The resampler state, chunk buffer and output accumulation form one linear pipeline.
#[allow(clippy::too_many_lines)]
fn resample_f32(
    from_rate: u32,
    to_rate: u32,
    channels: usize,
    buf: &mut AudioWorkerBuffer,
    label: &str,
) -> Result<(), PipelineError> {
    if from_rate == to_rate {
        // The encoder already consumes f32 interleaved at this rate.
        return Ok(());
    }
    let input = std::mem::take(&mut buf.samples_f32);
    let fail = |reason: String| PipelineError::EncodeFailed {
        path: Path::new(label).to_path_buf(),
        reason,
    };

    let mut planar_in = Vec::with_capacity(channels);
    let mut planar_out = Vec::with_capacity(channels);
    for channel in 0..channels {
        let values: Vec<f32> = input
            .iter()
            .skip(channel)
            .step_by(channels)
            .copied()
            .collect();
        planar_in.push(values);
        planar_out.push(Vec::new());
    }

    let mut resampler = FftFixedInOut::<f32>::new(
        usize_from(from_rate),
        usize_from(to_rate),
        RESAMPLE_CHUNK,
        channels,
    )
    .map_err(|err| {
        fail(format!(
            "failed to create resampler from {from_rate} Hz to {to_rate} Hz: {err}"
        ))
    })?;
    let chunk_in = resampler.input_frames_next();
    let chunk_out = resampler.output_frames_next().max(1);
    let mut out_chunk = resampler.output_buffer_allocate(true);
    let input_frames = input.len() / channels;
    let expected_u64 = (u64::try_from(input_frames).unwrap_or(u64::MAX))
        .saturating_mul(u64::from(to_rate))
        / u64::from(from_rate);
    let expected = usize::try_from(expected_u64).unwrap_or(usize::MAX);
    let delay = resampler.output_delay();
    let needed = expected.saturating_add(delay);
    let max_calls = input_frames
        .div_ceil(chunk_in)
        .saturating_add(needed.div_ceil(chunk_out))
        .saturating_add(4);
    let max_f32_samples = usize::try_from(MAX_PCM_BYTES_PER_FILE / 4).unwrap_or(usize::MAX);
    let plane_reserve = needed.saturating_add(4).min(max_f32_samples);
    let interleaved_reserve = plane_reserve.saturating_mul(channels).min(max_f32_samples);
    for plane in &mut planar_out {
        plane.reserve(plane_reserve);
    }
    if buf.resampled_f32.capacity() < interleaved_reserve {
        buf.resampled_f32
            .reserve(interleaved_reserve.saturating_sub(buf.resampled_f32.len()));
    }

    let mut calls = 0_usize;
    let mut chunk = vec![vec![0_f32; chunk_in]; channels];
    loop {
        let produced_frames = planar_out.first().map_or(0, Vec::len);
        if produced_frames >= needed || calls >= max_calls {
            break;
        }
        let offset = calls.saturating_mul(chunk_in);
        let take = chunk_in.min(input_frames.saturating_sub(offset));
        for slot in &mut out_chunk {
            slot.fill(0.0);
        }
        for channel in 0..channels {
            if let Some(dst) = chunk.get_mut(channel) {
                let copied = if let Some(src) = planar_in.get(channel) {
                    let available = src.len().saturating_sub(offset).min(take).min(dst.len());
                    if available > 0 {
                        dst[..available].copy_from_slice(&src[offset..offset + available]);
                    }
                    available
                } else {
                    0
                };
                if copied < dst.len() {
                    dst[copied..].fill(0.0);
                }
            }
        }
        let (_, produced) = resampler
            .process_into_buffer(&chunk, &mut out_chunk, None)
            .map_err(|err| fail(format!("resampling failed: {err}")))?;
        for channel in 0..channels {
            if let (Some(dst), Some(src)) = (planar_out.get_mut(channel), out_chunk.get(channel)) {
                dst.extend_from_slice(&src[..produced.min(src.len())]);
            }
        }
        calls += 1;
    }

    let available = planar_out.first().map_or(0, Vec::len);
    let skip = delay.min(available);
    let total_frames = expected.min(available.saturating_sub(skip));
    buf.resampled_f32.clear();
    for frame in skip..skip.saturating_add(total_frames) {
        for channel in 0..channels {
            if let Some(value) = planar_out.get(channel).and_then(|v| v.get(frame)) {
                buf.resampled_f32.push(*value);
            }
        }
    }
    std::mem::swap(&mut buf.samples_f32, &mut buf.resampled_f32);
    buf.resampled_f32.clear();
    Ok(())
}

// Frame sizing, parallel/serial dispatch and Ogg writing share one encoder lifetime.
#[allow(clippy::too_many_lines)]
fn encode_opus(
    path: &Path,
    rate: u32,
    channels: usize,
    buf: &mut AudioWorkerBuffer,
    batch_workers: usize,
) -> Result<(), PipelineError> {
    if channels > 2 {
        return Err(PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("Opus output supports mono/stereo only, got {channels} channels"),
        });
    }
    resample_f32(rate, OPUS_RATE, channels, buf, "opus-resampler")?;
    let total_frames = buf.samples_f32.len() / channels;
    if total_frames == 0 {
        return Err(PipelineError::EmptyOutput(path.to_path_buf()));
    }

    let mut encoder = OpusEncoder::new(
        i32::try_from(OPUS_RATE).unwrap_or(i32::MAX),
        channels,
        Application::Audio,
    )
    .map_err(|err| PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason: format!("failed to create Opus encoder: {err}"),
    })?;
    encoder.bitrate_bps = OPUS_BITRATE;
    encoder.complexity = 9;
    let head = OpusHead::for_encoder(&encoder, OPUS_RATE);
    let pre_skip = usize::from(head.pre_skip);
    let frame = usize_from(OPUS_RATE / (1000 / OPUS_FRAME_MS));
    let per_frame = frame.saturating_mul(channels);
    let packet_count = total_frames.saturating_add(pre_skip).div_ceil(frame);
    let final_granule = u64::from(u16::try_from(pre_skip).unwrap_or(u16::MAX))
        + u64::try_from(total_frames).unwrap_or(u64::MAX);
    buf.samples_f32
        .resize(packet_count.saturating_mul(per_frame), 0.0_f32);

    let file = File::create(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = OggOpusWriter::new(BufWriter::with_capacity(OPUS_WRITE_BUF_BYTES, file), head)
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("failed to create Ogg Opus writer: {err}"),
        })?;

    let available = std::thread::available_parallelism()
        .map_or(1, NonZeroUsize::get)
        .min(8);
    let inner_threads = (available / batch_workers.max(1)).max(1);
    let mut pcfg = ParallelConfig::new(
        i32::try_from(OPUS_RATE).unwrap_or(i32::MAX),
        channels,
        Application::Audio,
    );
    pcfg.bitrate_bps = OPUS_BITRATE;
    pcfg.complexity = 9;
    pcfg.warmup_ms = DEFAULT_WARMUP_MS;
    pcfg.threads = inner_threads;
    let parallel = inner_threads > 1 && pcfg.plan(packet_count, frame).workers > 1;

    let write_packet = |writer: &mut OggOpusWriter<BufWriter<File>>,
                        index: usize,
                        packet: &[u8]|
     -> Result<(), PipelineError> {
        if index + 1 == packet_count {
            let duration =
                final_granule.saturating_sub(u64::try_from(writer.granule()).unwrap_or(0));
            let duration = u32::try_from(duration).unwrap_or(u32::MAX);
            writer
                .write_packet_with_duration(packet, duration)
                .map_err(|err| PipelineError::EncodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("failed to write final Opus packet: {err}"),
                })?;
        } else {
            writer
                .write_packet(packet)
                .map_err(|err| PipelineError::EncodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("failed to write Opus packet: {err}"),
                })?;
        }
        Ok(())
    };

    if parallel {
        let packets = encode_parallel(
            &pcfg,
            &buf.samples_f32[..packet_count.saturating_mul(per_frame)],
            frame,
        )
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("Opus parallel encoding failed: {err}"),
        })?;
        if packets.len() != packet_count {
            return Err(PipelineError::EncodeFailed {
                path: path.to_path_buf(),
                reason: format!(
                    "Opus parallel encoding produced {} packets, expected {packet_count}",
                    packets.len()
                ),
            });
        }
        for (index, packet) in packets.iter().enumerate() {
            write_packet(&mut writer, index, packet)?;
        }
    } else {
        for index in 0..packet_count {
            let start = index.saturating_mul(per_frame);
            let Some(block) = buf.samples_f32.get(start..start + per_frame) else {
                return Err(PipelineError::EncodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("Opus PCM buffer shorter than expected packet {index}"),
                });
            };
            let written = encoder
                .encode(block, frame, &mut buf.packet)
                .map_err(|err| PipelineError::EncodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("Opus encoding failed: {err}"),
                })?;
            write_packet(&mut writer, index, &buf.packet[..written])?;
        }
    }

    let mut buf_writer = writer.finish().map_err(|err| PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason: format!("failed to finalize Ogg Opus stream: {err}"),
    })?;
    buf_writer.flush().map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })
}

const AAC_BITRATE_BPS: u32 = 128_000;
const MP3_BITRATE_KBPS: u32 = 192;
const MP3_RATES: [u32; 9] = [
    8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000,
];
const OGG_MAX_RATE: u32 = 192_000;
const OGG_CHUNK_FRAMES: usize = 4096;

fn aac_rate(rate: u32) -> u32 {
    if rusty_aac::sf_index_for_rate(rate).is_some() {
        rate
    } else {
        OPUS_RATE
    }
}

fn check_channels(
    path: &Path,
    channels: usize,
    min: usize,
    max: usize,
    codec: &str,
) -> Result<(), PipelineError> {
    if (min..=max).contains(&channels) {
        Ok(())
    } else {
        Err(PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("{codec} output supports {min}-{max} channels, got {channels}"),
        })
    }
}

fn encode_aac(
    path: &Path,
    rate: u32,
    channels: usize,
    buf: &mut AudioWorkerBuffer,
) -> Result<(), PipelineError> {
    check_channels(path, channels, 1, 6, "AAC")?;
    let target_rate = aac_rate(rate);
    resample_f32(rate, target_rate, channels, buf, "aac-resampler")?;
    if buf.samples_f32.is_empty() {
        return Err(PipelineError::EmptyOutput(path.to_path_buf()));
    }
    let ch_u16 = u16::try_from(channels).unwrap_or(u16::MAX);
    let mut encoder = rusty_aac::AacEncoder::new(rusty_aac::AacEncoderConfig {
        bitrate_bps: AAC_BITRATE_BPS,
        ..rusty_aac::AacEncoderConfig::default()
    });
    encoder
        .push_pcm(&buf.samples_f32, ch_u16, target_rate)
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("AAC encoding failed: {err}"),
        })?;
    encoder.finish();
    let file = File::create(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = BufWriter::with_capacity(OPUS_WRITE_BUF_BYTES, file);
    loop {
        match encoder.next_packet() {
            Ok(packet) => {
                let header = rusty_aac::write_adts_header(&rusty_aac::AdtsHeader {
                    object_type: 2,
                    sample_rate: target_rate,
                    channels: ch_u16,
                    frame_length: 7_usize.saturating_add(packet.data.len()),
                    header_len: 7,
                });
                writer
                    .write_all(&header)
                    .map_err(|source| PipelineError::Io {
                        path: path.to_path_buf(),
                        source,
                    })?;
                writer
                    .write_all(&packet.data)
                    .map_err(|source| PipelineError::Io {
                        path: path.to_path_buf(),
                        source,
                    })?;
            }
            Err(rusty_aac::Error::Eof | rusty_aac::Error::Again) => break,
            Err(err) => {
                return Err(PipelineError::EncodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("failed to drain AAC packets: {err}"),
                });
            }
        }
    }
    writer.flush().map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn encode_m4a(
    path: &Path,
    rate: u32,
    channels: usize,
    buf: &mut AudioWorkerBuffer,
) -> Result<(), PipelineError> {
    check_channels(path, channels, 1, 6, "M4A/AAC")?;
    let target_rate = aac_rate(rate);
    resample_f32(rate, target_rate, channels, buf, "m4a-resampler")?;
    if buf.samples_f32.is_empty() {
        return Err(PipelineError::EmptyOutput(path.to_path_buf()));
    }
    let ch_u16 = u16::try_from(channels).unwrap_or(u16::MAX);
    let total_input_frames = u64::try_from(buf.samples_f32.len() / channels).unwrap_or(u64::MAX);
    let mut encoder = rusty_aac::AacEncoder::new(rusty_aac::AacEncoderConfig {
        bitrate_bps: AAC_BITRATE_BPS,
        ..rusty_aac::AacEncoderConfig::default()
    });
    encoder
        .push_pcm(&buf.samples_f32, ch_u16, target_rate)
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("AAC encoding failed: {err}"),
        })?;
    encoder.finish();
    let mut packets: Vec<(Vec<u8>, u32)> = Vec::new();
    loop {
        match encoder.next_packet() {
            Ok(packet) => packets.push((packet.data, packet.duration)),
            Err(rusty_aac::Error::Eof | rusty_aac::Error::Again) => break,
            Err(err) => {
                return Err(PipelineError::EncodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("failed to drain AAC packets: {err}"),
                });
            }
        }
    }
    let priming = encoder.priming_samples();
    crate::m4a::write_m4a_aac(
        path,
        target_rate,
        ch_u16,
        AAC_BITRATE_BPS,
        total_input_frames,
        priming,
        &packets,
    )
}

fn encode_mp3(
    path: &Path,
    rate: u32,
    channels: usize,
    buf: &mut AudioWorkerBuffer,
) -> Result<(), PipelineError> {
    check_channels(path, channels, 1, 2, "MP3")?;
    let target_rate = if MP3_RATES.contains(&rate) {
        rate
    } else {
        OPUS_RATE
    };
    resample_f32(rate, target_rate, channels, buf, "mp3-resampler")?;
    if buf.samples_f32.is_empty() {
        return Err(PipelineError::EmptyOutput(path.to_path_buf()));
    }
    let ch_u16 = u16::try_from(channels).unwrap_or(u16::MAX);
    let mut encoder = rusty_mp3::Mp3Encoder::new(rusty_mp3::Mp3EncoderConfig {
        bitrate_kbps: MP3_BITRATE_KBPS,
        vbr_quality: None,
    });
    encoder
        .push_pcm_f32(&buf.samples_f32, ch_u16, target_rate)
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("MP3 encoding failed: {err}"),
        })?;
    encoder.finish();
    let file = File::create(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = BufWriter::with_capacity(OPUS_WRITE_BUF_BYTES, file);
    loop {
        match encoder.next_packet() {
            Ok(packet) => writer
                .write_all(&packet)
                .map_err(|source| PipelineError::Io {
                    path: path.to_path_buf(),
                    source,
                })?,
            Err(rusty_mp3::Error::Eof | rusty_mp3::Error::Again) => break,
            Err(err) => {
                return Err(PipelineError::EncodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("failed to drain MP3 packets: {err}"),
                });
            }
        }
    }
    writer.flush().map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn encode_ogg(
    path: &Path,
    rate: u32,
    channels: usize,
    buf: &mut AudioWorkerBuffer,
) -> Result<(), PipelineError> {
    check_channels(path, channels, 1, usize::from(u8::MAX), "Ogg Vorbis")?;
    if rate == 0 {
        return Err(PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: "Ogg Vorbis requires a positive sample rate".to_owned(),
        });
    }
    let target_rate = if rate <= OGG_MAX_RATE {
        rate
    } else {
        OPUS_RATE
    };
    resample_f32(rate, target_rate, channels, buf, "ogg-resampler")?;
    if buf.samples_f32.is_empty() {
        return Err(PipelineError::EmptyOutput(path.to_path_buf()));
    }
    let rate_nz = NonZeroU32::new(target_rate).ok_or_else(|| PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason: "Ogg Vorbis sample rate must be non-zero".to_owned(),
    })?;
    let ch_nz = NonZeroU8::new(u8::try_from(channels).unwrap_or(u8::MAX)).ok_or_else(|| {
        PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: "Ogg Vorbis channel count must be non-zero".to_owned(),
        }
    })?;
    let file = File::create(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let writer = BufWriter::with_capacity(64 * 1024, file);
    let mut encoder = vorbis_enc::EncoderBuilder::new(rate_nz, ch_nz, writer)
        .bitrate_management_strategy(vorbis_enc::BitrateManagementStrategy::QualityVbr {
            target_quality: 0.4,
        })
        .build()
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("failed to create Ogg Vorbis encoder: {err}"),
        })?;
    let total_frames = buf.samples_f32.len() / channels;
    let mut planes: Vec<Vec<f32>> = vec![Vec::with_capacity(OGG_CHUNK_FRAMES); channels];
    for start in (0..total_frames).step_by(OGG_CHUNK_FRAMES) {
        let frames = OGG_CHUNK_FRAMES.min(total_frames.saturating_sub(start));
        for plane in &mut planes {
            plane.clear();
        }
        for frame in 0..frames {
            let base = (start.saturating_add(frame)).saturating_mul(channels);
            for (channel, plane) in planes.iter_mut().enumerate() {
                if let Some(&value) = buf.samples_f32.get(base.saturating_add(channel)) {
                    plane.push(value);
                }
            }
        }
        encoder
            .encode_audio_block(planes.iter())
            .map_err(|err| PipelineError::EncodeFailed {
                path: path.to_path_buf(),
                reason: format!("Ogg Vorbis encoding failed: {err}"),
            })?;
    }
    let mut writer = encoder
        .finish()
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("failed to finalize Ogg Vorbis stream: {err}"),
        })?;
    writer.flush().map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn encode_alac(
    path: &Path,
    rate: u32,
    channels: usize,
    buf: &AudioWorkerBuffer,
) -> Result<(), PipelineError> {
    crate::m4a::encode_alac_m4a(path, rate, channels, &buf.samples_i16)
}

fn encode_aiff(
    path: &Path,
    rate: u32,
    channels: usize,
    bits: usize,
    buf: &AudioWorkerBuffer,
) -> Result<(), PipelineError> {
    crate::aiff::encode_aiff(
        path,
        rate,
        channels,
        bits,
        &buf.samples_i16,
        &buf.samples_i32,
    )
}

static STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Reserves a unique, exclusively created staging file in the target directory so concurrent
/// processes and source/target aliases cannot collide with an existing file.
fn create_staging_file(target: &Path) -> Result<PathBuf, PipelineError> {
    let parent = target
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    for _ in 0..64 {
        let sequence = STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let staging = parent.join(format!(
            ".fastconv-audio-{}-{sequence}.tmp",
            std::process::id()
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&staging) {
            Ok(_) => return Ok(staging),
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {}
            Err(source) => {
                return Err(PipelineError::Io {
                    path: staging,
                    source,
                });
            }
        }
    }
    Err(PipelineError::Io {
        path: target.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::AlreadyExists,
            "failed to reserve a unique staging file",
        ),
    })
}

fn encode_staged(
    item: &AudioBatchItem,
    encode: impl FnOnce(&Path) -> Result<(), PipelineError>,
) -> Result<u64, PipelineError> {
    let target = item.target();
    if let Some(parent) = target.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let temp = create_staging_file(target)?;
    let result = (|| -> Result<u64, PipelineError> {
        encode(&temp)?;
        let output_bytes = fs::metadata(&temp).map_or(0, |meta| meta.len());
        if output_bytes == 0 {
            return Err(PipelineError::EmptyOutput(target.to_path_buf()));
        }
        fs::rename(&temp, target).map_err(|source| PipelineError::Io {
            path: target.to_path_buf(),
            source,
        })?;
        Ok(output_bytes)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn encode_audio(
    item: &AudioBatchItem,
    mode: fastconv_domain::AudioExecutionMode,
    rate: u32,
    channels: usize,
    bits: usize,
    buf: &mut AudioWorkerBuffer,
    batch_workers: usize,
) -> Result<u64, PipelineError> {
    let output = mode.output_format();
    if output == AudioOutputFormat::Alac && bits == 24 {
        return Err(PipelineError::EncodeFailed {
            path: item.source().to_path_buf(),
            reason: "ALAC encoder currently supports 8/16-bit PCM only (24-bit would require lossy truncation)".to_owned(),
        });
    }
    let bits = if matches!(
        output,
        AudioOutputFormat::Opus
            | AudioOutputFormat::Ogg
            | AudioOutputFormat::Mp3
            | AudioOutputFormat::Aac
            | AudioOutputFormat::M4a
    ) {
        16
    } else {
        match bits {
            0 => 16,
            8 | 16 | 24 => bits,
            unsupported => {
                return Err(PipelineError::UnsupportedSampleDepth {
                    path: item.source().to_path_buf(),
                    bits: unsupported,
                });
            }
        }
    };
    // A direct i16 decode leaves samples_i16 populated; promote to i32 for FLAC
    // or pass the i16 buffer through to the 16-bit WAV/AIFF/ALAC writers.
    let direct_i16 = !buf.samples_i16.is_empty() && buf.samples_f32.is_empty();
    if direct_i16 {
        if output == AudioOutputFormat::Flac {
            buf.samples_i32.clear();
            buf.samples_i32
                .extend(buf.samples_i16.iter().copied().map(i32::from));
        }
    } else if matches!(
        output,
        AudioOutputFormat::Wav
            | AudioOutputFormat::Flac
            | AudioOutputFormat::Alac
            | AudioOutputFormat::Aiff
    ) {
        quantize_for(output, bits, buf);
    }
    encode_staged(item, |temp| match output {
        AudioOutputFormat::Wav => encode_wav(temp, rate, channels, bits, buf),
        AudioOutputFormat::Flac => encode_flac(temp, rate, channels, bits, buf, batch_workers),
        AudioOutputFormat::Opus => encode_opus(temp, rate, channels, buf, batch_workers),
        AudioOutputFormat::Ogg => encode_ogg(temp, rate, channels, buf),
        AudioOutputFormat::Mp3 => encode_mp3(temp, rate, channels, buf),
        AudioOutputFormat::Aac => encode_aac(temp, rate, channels, buf),
        AudioOutputFormat::M4a => encode_m4a(temp, rate, channels, buf),
        AudioOutputFormat::Alac => encode_alac(temp, rate, channels, buf),
        AudioOutputFormat::Aiff => encode_aiff(temp, rate, channels, bits, buf),
    })
}

fn convert_pcm_wav_to_flac(
    item: &AudioBatchItem,
    stream: PcmWavStream,
    buf: &mut AudioWorkerBuffer,
    batch_workers: usize,
) -> Result<u64, PipelineError> {
    buf.reset();
    encode_staged(item, |temp| {
        let PcmWavStream {
            reader,
            channels,
            bits,
            rate,
            frame_bytes,
            total_samples,
            remaining_bytes,
        } = stream;
        let source = LeBytesFlacSource {
            reader,
            read_buf: Vec::new(),
            channels,
            bits,
            rate: usize_from(rate),
            frame_bytes,
            total_samples,
            remaining_bytes,
        };
        let failed = Arc::new(AtomicBool::new(false));
        let source = EofTolerantFlacSource::new(source, Arc::clone(&failed));
        encode_flac_from_source(temp, source, batch_workers, Some(&failed))
    })
}

fn convert_item(
    item: &AudioBatchItem,
    mode: fastconv_domain::AudioExecutionMode,
    buf: &mut AudioWorkerBuffer,
    batch_workers: usize,
) -> Result<(u64, u64), PipelineError> {
    let input_bytes = fs::metadata(item.source())
        .map_err(|source| PipelineError::Io {
            path: item.source().to_path_buf(),
            source,
        })?
        .len();
    if item.input_format() == AudioInputFormat::Wav
        && mode.output_format() == AudioOutputFormat::Flac
        && let Some(stream) = try_open_pcm_wav(item.source())?
    {
        let output_bytes = convert_pcm_wav_to_flac(item, stream, buf, batch_workers)?;
        return Ok((input_bytes, output_bytes));
    }
    let (rate, channels, bits) = decode_audio(item, mode.output_format(), buf)?;
    let output_bytes = encode_audio(item, mode, rate, channels, bits, buf, batch_workers)?;
    Ok((input_bytes, output_bytes))
}

fn estimate_pcm_bytes(item: &AudioBatchItem) -> u64 {
    let stream = item.stream();
    let channels = u64::from(stream.channels().get()).max(1);
    let bits = stream
        .bits_per_sample()
        .map_or(16, |value| u64::from(value.get()))
        .max(8);
    let frames = stream.total_samples().unwrap_or_else(|| {
        let file_bytes = fs::metadata(item.source()).map_or(0, |meta| meta.len());
        file_bytes / channels / (bits / 8).max(1)
    });
    frames.saturating_mul(channels).saturating_mul(4)
}

fn resolve_worker_count(
    items: &[AudioBatchItem],
    requested: usize,
) -> Result<usize, PipelineError> {
    let largest_pcm = items.iter().map(estimate_pcm_bytes).max().unwrap_or(0);
    if largest_pcm > MAX_PCM_BYTES_PER_FILE {
        return Err(PipelineError::DecodeFailed {
            path: items[0].source().to_path_buf(),
            reason: format!(
                "estimated PCM working set {largest_pcm} bytes exceeds the {MAX_PCM_BYTES_PER_FILE} byte per-file limit"
            ),
        });
    }
    let budget_workers = u64::try_from(requested)
        .unwrap_or(u64::MAX)
        .min(MAX_PCM_BYTES_PER_BATCH / largest_pcm.max(1));
    Ok(usize::try_from(budget_workers.max(1))
        .unwrap_or(usize::MAX)
        .min(requested))
}

/// Executes a validated [`AudioBatchPlan`] on reusable per-worker sample buffers.
///
/// # Errors
///
/// Returns [`PipelineError`] if decoding, encoding, staging, or writing any item fails.
pub fn execute_audio_batch(plan: &AudioBatchPlan) -> Result<AudioConversionReport, PipelineError> {
    let started = Instant::now();
    let items = plan.items();
    let next = AtomicUsize::new(0);
    let abort = AtomicBool::new(false);
    let first_error: Mutex<Option<PipelineError>> = Mutex::new(None);
    let input_total = AtomicU64::new(0);
    let output_total = AtomicU64::new(0);
    let worker_count = resolve_worker_count(items, plan.effective_workers().get())?;

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        let mut startup_error = None;
        for worker_id in 0..worker_count {
            let next = &next;
            let abort = &abort;
            let first_error = &first_error;
            let input_total = &input_total;
            let output_total = &output_total;
            let worker = thread::Builder::new()
                .name(format!("fastconv-audio-{worker_id}"))
                .spawn_scoped(scope, move || {
                    let mut buffer = AudioWorkerBuffer::new();
                    loop {
                        if abort.load(Ordering::Relaxed)
                            || first_error.lock().is_ok_and(|guard| guard.is_some())
                        {
                            break;
                        }
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(index) else { break };
                        match convert_item(item, plan.mode(), &mut buffer, worker_count) {
                            Ok((in_bytes, out_bytes)) => {
                                input_total.fetch_add(in_bytes, Ordering::Relaxed);
                                output_total.fetch_add(out_bytes, Ordering::Relaxed);
                            }
                            Err(err) => {
                                if let Ok(mut guard) = first_error.lock()
                                    && guard.is_none()
                                {
                                    *guard = Some(err);
                                }
                                break;
                            }
                        }
                    }
                });
            match worker {
                Ok(handle) => handles.push(handle),
                Err(err) => {
                    abort.store(true, Ordering::Relaxed);
                    startup_error = Some(err);
                    break;
                }
            }
        }
        for handle in handles {
            if handle.join().is_err()
                && let Ok(mut guard) = first_error.lock()
                && guard.is_none()
            {
                *guard = Some(PipelineError::WorkerPanicked);
            }
        }
        if let Some(err) = startup_error
            && let Ok(mut guard) = first_error.lock()
            && guard.is_none()
        {
            *guard = Some(PipelineError::WorkerStartFailed(err));
        }
    });

    if let Ok(mut guard) = first_error.lock()
        && let Some(err) = guard.take()
    {
        return Err(err);
    }

    let input_bytes = input_total.load(Ordering::Relaxed);
    let output_bytes = output_total.load(Ordering::Relaxed);
    let input = NonZeroU64::new(input_bytes)
        .ok_or_else(|| PipelineError::EmptyOutput(items[0].source().to_path_buf()))?;
    let output = NonZeroU64::new(output_bytes)
        .ok_or_else(|| PipelineError::EmptyOutput(items[0].target().to_path_buf()))?;
    let files = NonZeroUsize::new(items.len())
        .ok_or_else(|| PipelineError::EmptyOutput(items[0].target().to_path_buf()))?;
    AudioConversionReport::new(
        files,
        input,
        output,
        started.elapsed(),
        NonZeroUsize::new(worker_count).unwrap_or(plan.effective_workers()),
        HardwareBackend::Cpu,
    )
    .map_err(|err| PipelineError::EncodeFailed {
        path: items[0].target().to_path_buf(),
        reason: err.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::{NonZeroU16, NonZeroU32};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = std::env::temp_dir().join(format!("fastconv-audio-{label}-{nanos}"));
        let _ = fs::create_dir_all(&path);
        path
    }

    fn write_wav(path: &Path, samples: &[i16], rate: u32, channels: u16) {
        let spec = WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let Ok(mut writer) = WavWriter::create(path, spec) else {
            return;
        };
        for &sample in samples {
            let _ = writer.write_sample(sample);
        }
        let _ = writer.finalize();
    }

    fn read_wav(path: &Path) -> Vec<i16> {
        let Ok(mut reader) = hound::WavReader::open(path) else {
            return Vec::new();
        };
        reader.samples::<i16>().filter_map(Result::ok).collect()
    }

    fn item(
        source: PathBuf,
        target: PathBuf,
        format: AudioInputFormat,
        channels: u16,
        rate: u32,
    ) -> AudioBatchItem {
        AudioBatchItem::new(
            source,
            target,
            format,
            fastconv_domain::AudioStreamInfo::new(
                NonZeroU16::new(channels).unwrap_or(NonZeroU16::MIN),
                NonZeroU32::new(rate).unwrap_or(NonZeroU32::MIN),
                NonZeroU16::new(16),
                None,
            )
            .unwrap_or_else(|_| unreachable!()),
        )
        .unwrap_or_else(|_| unreachable!())
    }

    fn plan(
        items: Vec<AudioBatchItem>,
        mode: fastconv_domain::AudioExecutionMode,
    ) -> AudioBatchPlan {
        AudioBatchPlan::new(items, mode, NonZeroUsize::MIN).unwrap_or_else(|_| unreachable!())
    }

    fn tone(frames: usize, channels: usize) -> Vec<i16> {
        let mut samples = Vec::with_capacity(frames * channels);
        for frame in 0..frames {
            for channel in 0..channels {
                let phase = frame.saturating_mul(137 + channel * 29) % 4096;
                let phase_i = i64::try_from(phase).unwrap_or(0);
                let value = phase_i.saturating_mul(24_000) / 4096 - 12_000;
                samples.push(i16::try_from(value).unwrap_or(0));
            }
        }
        samples
    }

    #[test]
    fn reuses_worker_buffer_capacity() {
        let mut buffer = AudioWorkerBuffer::new();
        buffer.samples_f32.extend_from_slice(&[0.1, 0.2]);
        buffer.samples_i16.extend_from_slice(&[1, 2]);
        let capacity = buffer.capacity_of_samples();
        buffer.reset();
        assert_eq!(buffer.decoded_len(), 0);
        assert_eq!(buffer.samples_i16.len(), 0);
        assert!(buffer.capacity_of_samples() >= capacity);
    }

    #[test]
    fn roundtrips_wav_flac_wav_bit_identically() -> Result<(), PipelineError> {
        let dir = temp_dir("wav-flac-wav");
        let wav1 = dir.join("in.wav");
        let flac = dir.join("middle.flac");
        let wav2 = dir.join("out.wav");
        let samples = tone(48_000, 2);
        write_wav(&wav1, &samples, 48_000, 2);

        execute_audio_batch(&plan(
            vec![item(wav1, flac.clone(), AudioInputFormat::Wav, 2, 48_000)],
            fastconv_domain::AudioExecutionMode::CpuFlac,
        ))?;
        execute_audio_batch(&plan(
            vec![item(flac, wav2.clone(), AudioInputFormat::Flac, 2, 48_000)],
            fastconv_domain::AudioExecutionMode::CpuWav,
        ))?;

        let decoded = read_wav(&wav2);
        assert_eq!(decoded.len(), samples.len());
        assert_eq!(decoded, samples);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn converts_mixed_format_batch_in_parallel() -> Result<(), PipelineError> {
        let dir = temp_dir("mixed-batch");
        let wav = dir.join("a.wav");
        let flac = dir.join("b.flac");
        let opus = dir.join("c.opus");
        let out = dir.join("out");
        let samples = tone(8_000, 2);
        write_wav(&wav, &samples, 48_000, 2);

        execute_audio_batch(&plan(
            vec![item(
                wav.clone(),
                flac.clone(),
                AudioInputFormat::Wav,
                2,
                48_000,
            )],
            fastconv_domain::AudioExecutionMode::CpuFlac,
        ))?;
        execute_audio_batch(&plan(
            vec![item(
                flac.clone(),
                opus.clone(),
                AudioInputFormat::Flac,
                2,
                48_000,
            )],
            fastconv_domain::AudioExecutionMode::CpuOpus,
        ))?;

        let mixed = AudioBatchPlan::new(
            vec![
                item(
                    wav.clone(),
                    out.join("a.wav"),
                    AudioInputFormat::Wav,
                    2,
                    48_000,
                ),
                item(flac, out.join("b.wav"), AudioInputFormat::Flac, 2, 48_000),
                item(opus, out.join("c.wav"), AudioInputFormat::Opus, 2, 48_000),
            ],
            fastconv_domain::AudioExecutionMode::CpuWav,
            NonZeroUsize::new(3).unwrap_or(NonZeroUsize::MIN),
        )
        .unwrap_or_else(|_| unreachable!());
        let report = execute_audio_batch(&mixed)?;
        assert_eq!(report.files_converted().get(), 3);
        assert_eq!(report.worker_threads_used().get(), 3);
        assert_ne!(read_wav(&out.join("a.wav")).len(), 0);
        assert_ne!(read_wav(&out.join("b.wav")).len(), 0);
        assert_ne!(read_wav(&out.join("c.wav")).len(), 0);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn encodes_opus_and_decodes_it_back() -> Result<(), PipelineError> {
        let dir = temp_dir("opus");
        let wav = dir.join("in.wav");
        let opus = dir.join("out.opus");
        let back = dir.join("back.wav");
        write_wav(&wav, &tone(48_000, 2), 48_000, 2);

        execute_audio_batch(&plan(
            vec![item(wav, opus.clone(), AudioInputFormat::Wav, 2, 48_000)],
            fastconv_domain::AudioExecutionMode::CpuOpus,
        ))?;
        execute_audio_batch(&plan(
            vec![item(opus, back.clone(), AudioInputFormat::Opus, 2, 48_000)],
            fastconv_domain::AudioExecutionMode::CpuWav,
        ))?;

        let decoded = read_wav(&back);
        assert_ne!(decoded.len(), 0);
        let spec = hound::WavReader::open(&back)
            .map_err(|err| PipelineError::DecodeFailed {
                path: back.clone(),
                reason: format!("failed to reopen decoded WAV: {err}"),
            })?
            .spec();
        assert_eq!(spec.sample_rate, 48_000);
        assert_eq!(spec.channels, 2);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    fn fmt_pcm(channels: u16, rate: u32, bits: u16) -> Vec<u8> {
        let block_align = channels.saturating_mul(bits / 8);
        let byte_rate = rate.saturating_mul(u32::from(block_align));
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&1_u16.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&byte_rate.to_le_bytes());
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        fmt
    }

    fn wav_bytes(fmt: &[u8], data: &[u8], extra: &[([u8; 4], &[u8])]) -> Vec<u8> {
        let mut chunks = Vec::new();
        chunks.extend_from_slice(b"fmt ");
        chunks.extend_from_slice(&u32::try_from(fmt.len()).unwrap_or(0).to_le_bytes());
        chunks.extend_from_slice(fmt);
        if fmt.len() % 2 == 1 {
            chunks.push(0);
        }
        for (id, body) in extra {
            chunks.extend_from_slice(id);
            chunks.extend_from_slice(&u32::try_from(body.len()).unwrap_or(0).to_le_bytes());
            chunks.extend_from_slice(body);
            if body.len() % 2 == 1 {
                chunks.push(0);
            }
        }
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

    fn write_raw_wav(path: &Path, bytes: &[u8]) -> Result<(), PipelineError> {
        fs::write(path, bytes).map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })
    }

    fn read_wav_i32(path: &Path) -> Vec<i32> {
        let Ok(mut reader) = hound::WavReader::open(path) else {
            return Vec::new();
        };
        reader.samples::<i32>().filter_map(Result::ok).collect()
    }

    fn item_bits(
        source: PathBuf,
        target: PathBuf,
        format: AudioInputFormat,
        channels: u16,
        rate: u32,
        bits: u16,
    ) -> AudioBatchItem {
        AudioBatchItem::new(
            source,
            target,
            format,
            fastconv_domain::AudioStreamInfo::new(
                NonZeroU16::new(channels).unwrap_or(NonZeroU16::MIN),
                NonZeroU32::new(rate).unwrap_or(NonZeroU32::MIN),
                NonZeroU16::new(bits),
                None,
            )
            .unwrap_or_else(|_| unreachable!()),
        )
        .unwrap_or_else(|_| unreachable!())
    }

    #[test]
    fn direct_16bit_wav_to_flac_with_odd_junk_chunk_roundtrips_bit_exactly()
    -> Result<(), PipelineError> {
        let dir = temp_dir("direct-flac-16-junk");
        let wav = dir.join("in.wav");
        let flac = dir.join("out.flac");
        let back = dir.join("back.wav");
        let samples = tone(12_000, 2);
        let mut data = Vec::with_capacity(samples.len() * 2);
        for sample in &samples {
            data.extend_from_slice(&sample.to_le_bytes());
        }
        // A JUNK chunk with an odd body length exercises RIFF even-byte padding.
        let junk = b"JNK";
        write_raw_wav(
            &wav,
            &wav_bytes(&fmt_pcm(2, 48_000, 16), &data, &[(*b"JUNK", junk)]),
        )?;
        assert!(
            try_open_pcm_wav(&wav)?.is_some(),
            "16-bit PCM must use the direct source"
        );

        execute_audio_batch(&plan(
            vec![item_bits(
                wav,
                flac.clone(),
                AudioInputFormat::Wav,
                2,
                48_000,
                16,
            )],
            fastconv_domain::AudioExecutionMode::CpuFlac,
        ))?;
        execute_audio_batch(&plan(
            vec![item_bits(
                flac,
                back.clone(),
                AudioInputFormat::Flac,
                2,
                48_000,
                16,
            )],
            fastconv_domain::AudioExecutionMode::CpuWav,
        ))?;
        assert_eq!(read_wav(&back), samples);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn direct_24bit_wav_to_flac_roundtrips_bit_exactly() -> Result<(), PipelineError> {
        let dir = temp_dir("direct-flac-24");
        let wav = dir.join("in.wav");
        let flac = dir.join("out.flac");
        let back = dir.join("back.wav");
        let samples: Vec<i32> = (0..8_000_i32)
            .map(|i| (i.saturating_mul(7919) % 4_000_000) - 2_000_000)
            .collect();
        let mut data = Vec::with_capacity(samples.len() * 3);
        for sample in &samples {
            let bytes = sample.to_le_bytes();
            data.extend_from_slice(&bytes[..3]);
        }
        write_raw_wav(&wav, &wav_bytes(&fmt_pcm(1, 44_100, 24), &data, &[]))?;
        assert!(
            try_open_pcm_wav(&wav)?.is_some(),
            "24-bit PCM must use the direct source"
        );

        execute_audio_batch(&plan(
            vec![item_bits(
                wav,
                flac.clone(),
                AudioInputFormat::Wav,
                1,
                44_100,
                24,
            )],
            fastconv_domain::AudioExecutionMode::CpuFlac,
        ))?;
        execute_audio_batch(&plan(
            vec![item_bits(
                flac,
                back.clone(),
                AudioInputFormat::Flac,
                1,
                44_100,
                24,
            )],
            fastconv_domain::AudioExecutionMode::CpuWav,
        ))?;
        assert_eq!(read_wav_i32(&back), samples);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn unsigned_8bit_wav_is_not_misread_as_signed() -> Result<(), PipelineError> {
        let dir = temp_dir("wav-8bit-unsigned");
        let wav = dir.join("in.wav");
        let flac = dir.join("out.flac");
        let back = dir.join("back.wav");
        // WAV 8-bit is unsigned (0..=255); FLAC 8-bit is signed. The direct
        // source must reject this file so Symphonia performs the conversion.
        let unsigned: [u8; 8] = [0, 1, 64, 127, 128, 192, 254, 255];
        write_raw_wav(&wav, &wav_bytes(&fmt_pcm(1, 8_000, 8), &unsigned, &[]))?;
        assert!(
            try_open_pcm_wav(&wav)?.is_none(),
            "8-bit WAV must fall back to Symphonia"
        );

        execute_audio_batch(&plan(
            vec![item_bits(
                wav,
                flac.clone(),
                AudioInputFormat::Wav,
                1,
                8_000,
                8,
            )],
            fastconv_domain::AudioExecutionMode::CpuFlac,
        ))?;
        execute_audio_batch(&plan(
            vec![item_bits(
                flac,
                back.clone(),
                AudioInputFormat::Flac,
                1,
                8_000,
                8,
            )],
            fastconv_domain::AudioExecutionMode::CpuWav,
        ))?;
        let mut reader =
            hound::WavReader::open(&back).map_err(|err| PipelineError::DecodeFailed {
                path: back.clone(),
                reason: format!("failed to reopen 8-bit WAV: {err}"),
            })?;
        let decoded: Vec<u8> = reader
            .samples::<i8>()
            .filter_map(Result::ok)
            .map(|sample| sample.wrapping_add(i8::MIN).cast_unsigned())
            .collect();
        assert_eq!(decoded, unsigned);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn float_wav_falls_back_to_symphonia() -> Result<(), PipelineError> {
        let dir = temp_dir("wav-float");
        let wav = dir.join("in.wav");
        // audio_format 3 == IEEE float.
        let mut fmt = fmt_pcm(1, 48_000, 32);
        fmt[..2].copy_from_slice(&3_u16.to_le_bytes());
        write_raw_wav(&wav, &wav_bytes(&fmt, &[0_u8; 32], &[]))?;
        assert!(
            try_open_pcm_wav(&wav)?.is_none(),
            "float WAV must fall back to Symphonia"
        );
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn ten_second_stereo_wav_parallel_opus_roundtrip() -> Result<(), PipelineError> {
        let dir = temp_dir("opus-parallel");
        let wav = dir.join("in.wav");
        let opus = dir.join("out.opus");
        let back = dir.join("back.wav");
        // 10 s at 48 kHz stereo: long enough for encode_parallel to split frames.
        let samples = tone(480_000, 2);
        write_wav(&wav, &samples, 48_000, 2);

        execute_audio_batch(&plan(
            vec![item(wav, opus.clone(), AudioInputFormat::Wav, 2, 48_000)],
            fastconv_domain::AudioExecutionMode::CpuOpus,
        ))?;
        execute_audio_batch(&plan(
            vec![item(opus, back.clone(), AudioInputFormat::Opus, 2, 48_000)],
            fastconv_domain::AudioExecutionMode::CpuWav,
        ))?;
        let decoded = read_wav(&back);
        assert_eq!(decoded.len(), samples.len());
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    fn deep_tone(samples: usize, channels: usize) -> Vec<i32> {
        (0..samples.saturating_mul(channels))
            .map(|i| {
                let value = i64::try_from(i).unwrap_or(0).saturating_mul(7919) % 4_000_000;
                i32::try_from(value.saturating_sub(2_000_000)).unwrap_or(0)
            })
            .collect()
    }

    fn write_wav24(
        path: &Path,
        samples: &[i32],
        rate: u32,
        channels: u16,
    ) -> Result<(), PipelineError> {
        let mut data = Vec::with_capacity(samples.len().saturating_mul(3));
        for &sample in samples {
            let bytes = sample.to_le_bytes();
            data.extend_from_slice(&bytes[..3]);
        }
        write_raw_wav(path, &wav_bytes(&fmt_pcm(channels, rate, 24), &data, &[]))
    }

    #[test]
    fn roundtrips_wav_aiff_wav_16bit_and_24bit_bit_identically() -> Result<(), PipelineError> {
        let dir = temp_dir("wav-aiff-wav");
        let samples16 = tone(12_000, 2);
        let wav16 = dir.join("in16.wav");
        let aiff16 = dir.join("mid16.aiff");
        let back16 = dir.join("out16.wav");
        write_wav(&wav16, &samples16, 48_000, 2);
        execute_audio_batch(&plan(
            vec![item(
                wav16,
                aiff16.clone(),
                AudioInputFormat::Wav,
                2,
                48_000,
            )],
            fastconv_domain::AudioExecutionMode::CpuAiff,
        ))?;
        execute_audio_batch(&plan(
            vec![item(
                aiff16,
                back16.clone(),
                AudioInputFormat::Aiff,
                2,
                48_000,
            )],
            fastconv_domain::AudioExecutionMode::CpuWav,
        ))?;
        assert_eq!(read_wav(&back16), samples16);

        let samples24 = deep_tone(12_000, 2);
        let wav24 = dir.join("in24.wav");
        let aiff24 = dir.join("mid24.aiff");
        let back24 = dir.join("out24.wav");
        write_wav24(&wav24, &samples24, 48_000, 2)?;
        execute_audio_batch(&plan(
            vec![item_bits(
                wav24,
                aiff24.clone(),
                AudioInputFormat::Wav,
                2,
                48_000,
                24,
            )],
            fastconv_domain::AudioExecutionMode::CpuAiff,
        ))?;
        execute_audio_batch(&plan(
            vec![item_bits(
                aiff24,
                back24.clone(),
                AudioInputFormat::Aiff,
                2,
                48_000,
                24,
            )],
            fastconv_domain::AudioExecutionMode::CpuWav,
        ))?;
        assert_eq!(read_wav_i32(&back24), samples24);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn roundtrips_wav_alac_wav_bit_identically() -> Result<(), PipelineError> {
        let dir = temp_dir("wav-alac-wav");
        let samples = tone(12_000, 2);
        let wav = dir.join("in.wav");
        let m4a = dir.join("mid.m4a");
        let back = dir.join("out.wav");
        write_wav(&wav, &samples, 48_000, 2);
        execute_audio_batch(&plan(
            vec![item(
                wav.clone(),
                m4a.clone(),
                AudioInputFormat::Wav,
                2,
                48_000,
            )],
            fastconv_domain::AudioExecutionMode::CpuAlac,
        ))?;
        execute_audio_batch(&plan(
            vec![item(m4a, back.clone(), AudioInputFormat::Alac, 2, 48_000)],
            fastconv_domain::AudioExecutionMode::CpuWav,
        ))?;
        assert_eq!(read_wav(&back), samples);

        let wav24 = dir.join("in24.wav");
        let m4a24 = dir.join("reject.m4a");
        write_wav24(&wav24, &deep_tone(4_096, 1), 48_000, 1)?;
        let result = execute_audio_batch(&plan(
            vec![item_bits(
                wav24,
                m4a24,
                AudioInputFormat::Wav,
                1,
                48_000,
                24,
            )],
            fastconv_domain::AudioExecutionMode::CpuAlac,
        ));
        assert!(
            matches!(result, Err(PipelineError::EncodeFailed { .. })),
            "24-bit ALAC must fail instead of truncating silently"
        );
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    fn assert_lossy_roundtrip(
        label: &str,
        extension: &str,
        mode: fastconv_domain::AudioExecutionMode,
        decoded_format: AudioInputFormat,
    ) -> Result<(), PipelineError> {
        let dir = temp_dir(label);
        let wav = dir.join("in.wav");
        let encoded = dir.join(format!("out.{extension}"));
        let back = dir.join("back.wav");
        write_wav(&wav, &tone(16_000, 2), 48_000, 2);
        execute_audio_batch(&plan(
            vec![item(wav, encoded.clone(), AudioInputFormat::Wav, 2, 48_000)],
            mode,
        ))?;
        execute_audio_batch(&plan(
            vec![item(encoded, back.clone(), decoded_format, 2, 48_000)],
            fastconv_domain::AudioExecutionMode::CpuWav,
        ))?;
        assert!(!read_wav(&back).is_empty(), "decoded WAV must not be empty");
        let spec = hound::WavReader::open(&back)
            .map_err(|err| PipelineError::DecodeFailed {
                path: back.clone(),
                reason: format!("failed to reopen decoded WAV: {err}"),
            })?
            .spec();
        assert_eq!(spec.sample_rate, 48_000);
        assert_eq!(spec.channels, 2);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn roundtrips_wav_aac_wav() -> Result<(), PipelineError> {
        assert_lossy_roundtrip(
            "wav-aac-wav",
            "aac",
            fastconv_domain::AudioExecutionMode::CpuAac,
            AudioInputFormat::Aac,
        )
    }

    #[test]
    fn roundtrips_wav_m4a_wav() -> Result<(), PipelineError> {
        assert_lossy_roundtrip(
            "wav-m4a-wav",
            "m4a",
            fastconv_domain::AudioExecutionMode::CpuM4a,
            AudioInputFormat::M4a,
        )
    }

    #[test]
    fn roundtrips_wav_mp3_wav() -> Result<(), PipelineError> {
        assert_lossy_roundtrip(
            "wav-mp3-wav",
            "mp3",
            fastconv_domain::AudioExecutionMode::CpuMp3,
            AudioInputFormat::Mp3,
        )
    }

    #[test]
    fn roundtrips_wav_ogg_wav() -> Result<(), PipelineError> {
        assert_lossy_roundtrip(
            "wav-ogg-wav",
            "ogg",
            fastconv_domain::AudioExecutionMode::CpuOgg,
            AudioInputFormat::Vorbis,
        )
    }
}
