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
    AudioBatchItem, AudioBatchPlan, AudioBitDepth, AudioConversionReport, AudioEncodingOptions,
    AudioInputFormat, AudioOptionChannels, AudioOptionSampleRate, AudioOutputFormat,
    AudioRateControl, AudioSampleFormat, HardwareBackend, aac_frame_capacity_bps,
};
use flacenc::bitsink::ByteSink;
use flacenc::component::BitRepr;
use flacenc::error::{SourceError, Verify};
use flacenc::source::{Fill, Source};
use hound::{SampleFormat, WavSpec, WavWriter};
use opus_pure::{
    Application, DEFAULT_WARMUP_MS, MAX_PACKET_BYTES, MAX_PACKET_SAMPLES, OggOpusReader,
    OggOpusWriter, OpusEncoder, OpusHead, ParallelConfig, RateControl, Trim, encode_parallel,
};
use rubato::audioadapter_buffers::owned::InterleavedOwned;
use rubato::{Fft, FixedSync, Indexing, Resampler};
use symphonia::core::audio::sample::SampleFormat as SymphoniaSampleFormat;
use symphonia::core::codecs::audio::{
    AudioDecoderOptions,
    well_known::{CODEC_ID_PCM_F32BE, CODEC_ID_PCM_F32LE, CODEC_ID_PCM_F64BE, CODEC_ID_PCM_F64LE},
};
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
/// `MAX_PCM_BYTES_PER_FILE / 4`, expressed as `usize` so caps are checked before allocation.
const MAX_F32_SAMPLES_PER_FILE: usize = 128 * 1024 * 1024;

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
    if sample.is_nan() {
        return 0;
    }
    let value = to_i32_saturating(sample.clamp(-1.0, 1.0) * 128.0).clamp(-128, 127);
    i8::try_from(value).unwrap_or(if value < 0 { i8::MIN } else { i8::MAX })
}

fn quantize_i16(sample: f32) -> i16 {
    if sample.is_nan() {
        return 0;
    }
    let value = to_i32_saturating(sample.clamp(-1.0, 1.0) * 32_768.0).clamp(-32_768, 32_767);
    i16::try_from(value).unwrap_or(if value < 0 { i16::MIN } else { i16::MAX })
}

fn quantize_i24(sample: f32) -> i32 {
    if sample.is_nan() {
        return 0;
    }
    to_i32_saturating(sample.clamp(-1.0, 1.0) * 8_388_608.0).clamp(-8_388_608, 8_388_607)
}

/// Quantizes a normalized sample to full-scale 32-bit integer PCM.
///
/// The `f64` intermediate matters: doing `* 2_147_483_648.0` in `f32` rounds
/// `1.0` past `i32::MAX` and loses the low eight bits of large values.
fn quantize_i32(sample: f32) -> i32 {
    if sample.is_nan() {
        return 0;
    }
    let scaled = f64::from(sample.clamp(-1.0, 1.0)) * 2_147_483_648.0_f64;
    // The clamp happens in f64, so the final narrowing is saturating and
    // well-defined; the allow covers only that narrowing cast.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let narrowed = scaled
        .round()
        .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32;
    narrowed
}

/// Checks the post-transform sample count *before* any buffer is grown. Both resampling
/// and channel remixing can increase the sample count, so neither may allocate first.
fn checked_f32_capacity(frames: u64, channels: usize, path: &Path) -> Result<usize, PipelineError> {
    let samples = frames.saturating_mul(u64::try_from(channels.max(1)).unwrap_or(u64::MAX));
    if samples > u64::try_from(MAX_F32_SAMPLES_PER_FILE).unwrap_or(u64::MAX) {
        return Err(PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!(
                "transformed PCM would exceed the {MAX_PCM_BYTES_PER_FILE}-byte per-file limit"
            ),
        });
    }
    Ok(usize::try_from(samples).unwrap_or(MAX_F32_SAMPLES_PER_FILE))
}

const ITU_CENTER: f32 = std::f32::consts::FRAC_1_SQRT_2;
const ITU_SURROUND: f32 = 0.5;

const SPEAKER_FL: u8 = 0;
const SPEAKER_FR: u8 = 1;
const SPEAKER_FC: u8 = 2;
const SPEAKER_LFE: u8 = 3;
const SPEAKER_BL: u8 = 4;
const SPEAKER_BR: u8 = 5;
const SPEAKER_SL: u8 = 6;
const SPEAKER_SR: u8 = 7;

/// Canonical speaker positions for the layouts the native encoders understand.
/// Four- and seven-channel layouts are ambiguous, so they are deliberately absent.
fn speaker_layout(channels: usize) -> Option<&'static [u8]> {
    match channels {
        1 => Some(&[SPEAKER_FC]),
        2 => Some(&[SPEAKER_FL, SPEAKER_FR]),
        3 => Some(&[SPEAKER_FL, SPEAKER_FR, SPEAKER_FC]),
        5 => Some(&[SPEAKER_FL, SPEAKER_FR, SPEAKER_FC, SPEAKER_BL, SPEAKER_BR]),
        6 => Some(&[
            SPEAKER_FL,
            SPEAKER_FR,
            SPEAKER_FC,
            SPEAKER_LFE,
            SPEAKER_BL,
            SPEAKER_BR,
        ]),
        8 => Some(&[
            SPEAKER_FL,
            SPEAKER_FR,
            SPEAKER_FC,
            SPEAKER_LFE,
            SPEAKER_BL,
            SPEAKER_BR,
            SPEAKER_SL,
            SPEAKER_SR,
        ]),
        _ => None,
    }
}

/// Per-input-channel contribution to [left, right] for an ITU-style stereo downmix.
/// `0` is FL, `1` FR, `2` FC, `3` LFE (omitted for 6/8-channel layouts), then
/// BL/BR and optional surround channels. Positive weights only means the output
/// can be normalized by its own weight sum so full-scale inputs cannot clip.
fn downmix_pair(channels: usize, index: usize) -> [f32; 2] {
    match (channels, index) {
        (_, 0) => [1.0, 0.0],
        (_, 1) => [0.0, 1.0],
        (4, 2) | (5, 3) | (6..=8, 4) => [ITU_CENTER, 0.0],
        (_, 2) => [ITU_CENTER, ITU_CENTER],
        (4, 3) | (5, 4) | (6..=8, 5) => [0.0, ITU_CENTER],
        (7, 6) => [ITU_SURROUND, ITU_SURROUND],
        (8, 6) => [ITU_SURROUND, 0.0],
        (8, 7) => [0.0, ITU_SURROUND],
        // Index 3 is LFE for 6/7/8-channel layouts and is deliberately omitted.
        _ => [0.0, 0.0],
    }
}

/// Remixes interleaved `f32` samples in place. Downmixes (fewer output channels)
/// happen before resampling; upmixes happen after it, as orchestrated by `convert_item`.
fn remix_channels_f32(
    from: usize,
    to: usize,
    buf: &mut AudioWorkerBuffer,
    path: &Path,
) -> Result<(), PipelineError> {
    if from == 0 || to == 0 {
        return Err(PipelineError::DecodeFailed {
            path: path.to_path_buf(),
            reason: format!("cannot remix {from} channels to {to}"),
        });
    }
    if from == to {
        return Ok(());
    }
    let source_layout = speaker_layout(from);
    let target_layout = speaker_layout(to);
    if from > 2 && to > 1 && (source_layout.is_none() || target_layout.is_none()) {
        // Copying the first N channels would silently move LFE samples into a
        // surround slot (or drop a surround channel) for 5/6/7/8-source layouts.
        return Err(PipelineError::DecodeFailed {
            path: path.to_path_buf(),
            reason: format!(
                "ambiguous channel remix from {from} to {to}: speaker layout is not defined"
            ),
        });
    }
    let frames = buf.samples_f32.len() / from;
    let capacity = checked_f32_capacity(u64::try_from(frames).unwrap_or(u64::MAX), to, path)?;
    let input = std::mem::take(&mut buf.samples_f32);
    let mut output = Vec::with_capacity(capacity);

    if to == 1 {
        // Arithmetic mean of every input channel: bounded by the input range,
        // so it cannot clip regardless of source layout.
        let scale = 1.0 / f32::from(u16::try_from(from).unwrap_or(u16::MAX));
        for frame in input.chunks_exact(from) {
            let sum = frame.iter().copied().sum::<f32>();
            output.push(sum * scale);
        }
    } else if to == 2 && from > 2 {
        let mut weights = [[0.0_f32; 2]; 8];
        let mut sums = [0.0_f32; 2];
        for (index, slot) in weights.iter_mut().enumerate().take(from.min(8)) {
            *slot = downmix_pair(from, index);
            sums[0] += slot[0];
            sums[1] += slot[1];
        }
        let divisors = [
            sums[0].max(f32::MIN_POSITIVE),
            sums[1].max(f32::MIN_POSITIVE),
        ];
        for frame in input.chunks_exact(from) {
            let mut mixed = [0.0_f32; 2];
            for (index, &value) in frame.iter().enumerate().take(8) {
                let weight = weights[index];
                mixed[0] += value * weight[0];
                mixed[1] += value * weight[1];
            }
            output.push(mixed[0] / divisors[0]);
            output.push(mixed[1] / divisors[1]);
        }
    } else {
        for frame in input.chunks_exact(from) {
            for index in 0..to {
                // Mono is duplicated to FL/FR. Otherwise map by speaker identity so
                // a dropped LFE never becomes a surround channel (and vice versa).
                let value = if from == 1 {
                    if index < 2 { frame[0] } else { 0.0 }
                } else if let (Some(src), Some(dst)) = (source_layout, target_layout) {
                    dst.get(index)
                        .and_then(|speaker| src.iter().position(|s| s == speaker))
                        .map_or(0.0, |source_index| frame[source_index])
                } else if index < from {
                    frame[index]
                } else {
                    0.0
                };
                output.push(value);
            }
        }
    }
    buf.samples_f32 = output;
    Ok(())
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

// The bounded read, per-frame format checks and PCM limit must stay in one pass.
#[allow(clippy::too_many_lines)]
fn decode_adts_aac(
    path: &Path,
    buf: &mut AudioWorkerBuffer,
) -> Result<(u32, usize), PipelineError> {
    // Read at most one byte past the PCM limit so an oversized file cannot spike RAM
    // before the incremental PCM check in the decode loop below.
    let limit = MAX_PCM_BYTES_PER_FILE.saturating_add(1);
    let mut limited = File::open(path)
        .map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .take(limit);
    let mut data = Vec::new();
    limited
        .read_to_end(&mut data)
        .map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    if u64::try_from(data.len()).unwrap_or(u64::MAX) > MAX_PCM_BYTES_PER_FILE {
        return Err(PipelineError::DecodeFailed {
            path: path.to_path_buf(),
            reason: format!("AAC input exceeds the {MAX_PCM_BYTES_PER_FILE}-byte per-file limit"),
        });
    }
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
        } else if frame.sample_rate != 0
            && (frame.sample_rate != decoded_rate
                || usize::from(frame.channels) != decoded_channels)
        {
            return Err(PipelineError::DecodeFailed {
                path: path.to_path_buf(),
                reason: format!(
                    "ADTS frame at byte {pos} changes format from {decoded_rate} Hz/{decoded_channels} channels to {} Hz/{} channels",
                    frame.sample_rate,
                    usize::from(frame.channels)
                ),
            });
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
// and direct-i32 decisions and their borrows of `buf` in one place.
#[allow(clippy::too_many_lines)]
fn decode_symphonia(
    path: &Path,
    format: AudioInputFormat,
    target: AudioOutputFormat,
    options: &AudioEncodingOptions,
    buf: &mut AudioWorkerBuffer,
) -> Result<(u32, usize, usize, AudioSampleFormat), PipelineError> {
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
    if sample_rate == 0 || channels == 0 {
        return Err(PipelineError::DecodeFailed {
            path: path.to_path_buf(),
            reason: format!("invalid decoded stream format: {sample_rate} Hz, {channels} channels"),
        });
    }
    // Symphonia 0.6.1 leaves `bits_per_sample` and `sample_format` unset for
    // plain IEEE-float WAV (0x0003) and AIFC `fl32`; only the codec id is set.
    // Extensible IEEE float does set both, so consult every available signal.
    let f32_codec = matches!(params.codec, CODEC_ID_PCM_F32LE | CODEC_ID_PCM_F32BE);
    let f64_codec = matches!(params.codec, CODEC_ID_PCM_F64LE | CODEC_ID_PCM_F64BE);
    let f32_samples = matches!(params.sample_format, Some(SymphoniaSampleFormat::F32));
    let f64_samples = matches!(params.sample_format, Some(SymphoniaSampleFormat::F64));
    let is_float = f32_codec || f64_codec || f32_samples || f64_samples;
    let sample_format = if is_float {
        AudioSampleFormat::Float
    } else {
        AudioSampleFormat::Int
    };
    let inferred_bits = if f32_codec || f32_samples {
        32
    } else if f64_codec || f64_samples {
        64
    } else {
        0
    };
    let bits = params
        .bits_per_sample
        .and_then(|bits| usize::try_from(bits).ok())
        .unwrap_or(inferred_bits);
    let track_id = track.id;
    let total_samples = track.num_frames;
    // A resample or channel remix needs the normalized f32 path: the direct integer
    // copies are bit-exact only when samples pass through untouched.
    let rate_channel_change = options
        .sample_rate()
        .is_some_and(|rate| rate.hz() != sample_rate)
        || options
            .channels()
            .is_some_and(|channels_opt| usize::from(channels_opt.get()) != channels);
    // Symphonia left-aligns 16-bit integer samples when converting to i32 (i16 << 16),
    // so a direct i16 copy is both faster and bit-exact for lossless targets.
    let direct_i16 = matches!(
        target,
        AudioOutputFormat::Wav
            | AudioOutputFormat::Flac
            | AudioOutputFormat::Aiff
            | AudioOutputFormat::Alac
    ) && bits == 16
        && !is_float
        && !rate_channel_change
        && options
            .bit_depth()
            .is_none_or(|depth| depth == AudioBitDepth::Int16);
    // Going from i32 through f32 truncates the low eight bits, so 32-bit integer
    // WAV/AIFF conversions copy the decoded i32 samples straight through.
    let direct_i32 = matches!(target, AudioOutputFormat::Wav | AudioOutputFormat::Aiff)
        && bits == 32
        && !is_float
        && !rate_channel_change
        && options
            .bit_depth()
            .is_none_or(|depth| depth == AudioBitDepth::Int32);
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
    } else if direct_i32 {
        if buf.samples_i32.capacity() < reserve {
            buf.samples_i32
                .reserve(reserve.saturating_sub(buf.samples_i32.len()));
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
        } else if direct_i32 {
            buf.samples_i32.len()
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
        } else if direct_i32 {
            buf.samples_i32.resize(end, 0);
            if let Some(slot) = buf.samples_i32.get_mut(start..end) {
                audio_frame.copy_to_slice_interleaved::<i32, _>(slot);
            }
        } else {
            buf.samples_f32.resize(end, 0.0);
            if let Some(slot) = buf.samples_f32.get_mut(start..end) {
                audio_frame.copy_to_slice_interleaved(slot);
            }
        }
    }
    Ok((sample_rate, channels, bits, sample_format))
}

fn decode_audio(
    item: &AudioBatchItem,
    target: AudioOutputFormat,
    options: &AudioEncodingOptions,
    buf: &mut AudioWorkerBuffer,
) -> Result<(u32, usize, usize, AudioSampleFormat), PipelineError> {
    buf.reset();
    let (rate, channels, source_bits, sample_format) = match item.input_format() {
        AudioInputFormat::Opus => {
            let (rate, channels) = decode_opus(item.source(), buf)?;
            (rate, channels, 16, AudioSampleFormat::Int)
        }
        AudioInputFormat::Aac => {
            let (rate, channels) = decode_adts_aac(item.source(), buf)?;
            (rate, channels, 16, AudioSampleFormat::Int)
        }
        _ => decode_symphonia(item.source(), item.input_format(), target, options, buf)?,
    };

    if channels == 0 {
        return Err(PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: "decoded audio has zero channels".to_owned(),
        });
    }
    if buf.samples_f32.is_empty() && buf.samples_i16.is_empty() && buf.samples_i32.is_empty() {
        return Err(PipelineError::EmptyOutput(item.source().to_path_buf()));
    }

    Ok((rate, channels, source_bits, sample_format))
}

/// Resolves the lossless encoder's PCM depth from the source stream and the
/// explicit `--bit-depth` option. An explicit depth was already validated by the
/// domain and always wins, so 24/32/f32 input can be quantized down to 16-bit
/// ALAC or 24-bit FLAC on request.
fn resolve_target_depth(
    path: &Path,
    output: AudioOutputFormat,
    source_bits: usize,
    source_format: AudioSampleFormat,
    options: &AudioEncodingOptions,
) -> Result<AudioBitDepth, PipelineError> {
    if let Some(depth) = options.bit_depth() {
        return Ok(depth);
    }
    if matches!(
        output,
        AudioOutputFormat::Opus
            | AudioOutputFormat::Ogg
            | AudioOutputFormat::Mp3
            | AudioOutputFormat::Aac
            | AudioOutputFormat::M4a
    ) {
        return Ok(AudioBitDepth::Float32);
    }
    let unsupported = |bits| PipelineError::UnsupportedSampleDepth {
        path: path.to_path_buf(),
        bits,
    };
    let encode_err = |reason: String| PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason,
    };
    match (output, source_format, source_bits) {
        (AudioOutputFormat::Wav | AudioOutputFormat::Aiff, AudioSampleFormat::Float, _) => {
            Ok(AudioBitDepth::Float32)
        }
        (AudioOutputFormat::Wav | AudioOutputFormat::Aiff, AudioSampleFormat::Int, 8) => {
            Ok(AudioBitDepth::Int8)
        }
        (
            AudioOutputFormat::Wav | AudioOutputFormat::Aiff | AudioOutputFormat::Flac,
            AudioSampleFormat::Int,
            0 | 16,
        )
        | (AudioOutputFormat::Alac, AudioSampleFormat::Int, 0 | 8 | 16) => Ok(AudioBitDepth::Int16),
        (AudioOutputFormat::Wav | AudioOutputFormat::Aiff, AudioSampleFormat::Int, 24) => {
            Ok(AudioBitDepth::Int24)
        }
        (AudioOutputFormat::Wav | AudioOutputFormat::Aiff, AudioSampleFormat::Int, 32) => {
            Ok(AudioBitDepth::Int32)
        }
        // f32 has a 24-bit mantissa, so Int24 keeps every meaningful bit without
        // pretending the source carried 32 bits of integer precision.
        (AudioOutputFormat::Flac, AudioSampleFormat::Float, _)
        | (AudioOutputFormat::Flac, AudioSampleFormat::Int, 24) => Ok(AudioBitDepth::Int24),
        (AudioOutputFormat::Flac, AudioSampleFormat::Int, 8) => Ok(AudioBitDepth::Int8),
        (AudioOutputFormat::Flac, AudioSampleFormat::Int, 32) => Err(encode_err(
            "FLAC stores at most 24-bit integer PCM; use --bit-depth 24".to_owned(),
        )),
        (AudioOutputFormat::Alac, _, _) => Err(encode_err(
            "ALAC stores 16-bit integer PCM only; use --bit-depth 16".to_owned(),
        )),
        (_, _, bits) => Err(unsupported(bits)),
    }
}

/// Quantizes the decoded `f32` samples into the integer representation the target
/// encoder consumes. A `Float32` target keeps the original float buffer untouched.
fn quantize_for(output: AudioOutputFormat, depth: AudioBitDepth, buf: &mut AudioWorkerBuffer) {
    buf.samples_i16.clear();
    buf.samples_i32.clear();
    match (output, depth) {
        (_, AudioBitDepth::Float32) => return,
        (AudioOutputFormat::Flac, AudioBitDepth::Int8) => buf.samples_i32.extend(
            buf.samples_f32
                .iter()
                .copied()
                .map(|sample| i32::from(quantize_i8(sample))),
        ),
        (AudioOutputFormat::Flac, AudioBitDepth::Int16) => buf.samples_i32.extend(
            buf.samples_f32
                .iter()
                .copied()
                .map(|sample| i32::from(quantize_i16(sample))),
        ),
        (
            AudioOutputFormat::Flac | AudioOutputFormat::Wav | AudioOutputFormat::Aiff,
            AudioBitDepth::Int24,
        ) => buf
            .samples_i32
            .extend(buf.samples_f32.iter().copied().map(quantize_i24)),
        (
            AudioOutputFormat::Wav | AudioOutputFormat::Aiff | AudioOutputFormat::Alac,
            AudioBitDepth::Int8,
        ) => buf.samples_i16.extend(
            buf.samples_f32
                .iter()
                .copied()
                .map(|sample| i16::from(quantize_i8(sample))),
        ),
        (
            AudioOutputFormat::Wav | AudioOutputFormat::Aiff | AudioOutputFormat::Alac,
            AudioBitDepth::Int16,
        ) => buf
            .samples_i16
            .extend(buf.samples_f32.iter().copied().map(quantize_i16)),
        (AudioOutputFormat::Wav | AudioOutputFormat::Aiff, AudioBitDepth::Int32) => buf
            .samples_i32
            .extend(buf.samples_f32.iter().copied().map(quantize_i32)),
        // Unsupported combinations are rejected before this point.
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
    depth: AudioBitDepth,
    buf: &AudioWorkerBuffer,
) -> Result<(), PipelineError> {
    let channel_count = u16::try_from(channels).map_err(|_| PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason: format!("WAV supports at most 65535 channels, got {channels}"),
    })?;
    let spec = WavSpec {
        channels: channel_count,
        sample_rate: rate,
        bits_per_sample: depth.bits(),
        sample_format: if depth.is_float() {
            SampleFormat::Float
        } else {
            SampleFormat::Int
        },
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
    let write_err = |err: hound::Error| PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason: format!("WAV sample write failed: {err}"),
    };
    match depth {
        AudioBitDepth::Int8 => {
            for &sample in &buf.samples_i16 {
                let value =
                    i8::try_from(sample).unwrap_or(if sample < 0 { i8::MIN } else { i8::MAX });
                writer.write_sample(value).map_err(write_err)?;
            }
        }
        AudioBitDepth::Int16 => {
            // The bulk writer avoids a virtual call and bounds check per sample.
            const CHUNK: usize = 65_536;
            for chunk in buf.samples_i16.chunks(CHUNK) {
                let mut fast = writer.get_i16_writer(chunk.len().try_into().unwrap_or(u32::MAX));
                for &sample in chunk {
                    fast.write_sample(sample);
                }
                fast.flush().map_err(write_err)?;
            }
        }
        AudioBitDepth::Int24 | AudioBitDepth::Int32 => {
            for &sample in &buf.samples_i32 {
                writer.write_sample(sample).map_err(write_err)?;
            }
        }
        AudioBitDepth::Float32 => {
            for &sample in &buf.samples_f32 {
                writer.write_sample(sample).map_err(write_err)?;
            }
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

    let mut resampler = Fft::<f32>::new(
        usize_from(from_rate),
        usize_from(to_rate),
        RESAMPLE_CHUNK,
        channels,
        FixedSync::Both,
    )
    .map_err(|err| {
        fail(format!(
            "failed to create resampler from {from_rate} Hz to {to_rate} Hz: {err}"
        ))
    })?;
    let chunk_in = resampler.input_frames_next();
    let chunk_out = resampler.output_frames_next().max(1);
    let input_frames = input.len() / channels;
    let expected_u64 = (u64::try_from(input_frames).unwrap_or(u64::MAX))
        .saturating_mul(u64::from(to_rate))
        / u64::from(from_rate);
    // Check the grown sample count before reserving or extending any buffer.
    let expected_cap = checked_f32_capacity(expected_u64, channels, Path::new(label))?;
    let expected = usize::try_from(expected_u64)
        .unwrap_or(usize::MAX)
        .min(expected_cap);
    let delay = resampler.output_delay();
    let needed = expected.saturating_add(delay);
    let max_calls = input_frames
        .div_ceil(chunk_in.max(1))
        .saturating_add(needed.div_ceil(chunk_out))
        .saturating_add(4);
    let interleaved_reserve = needed
        .saturating_add(chunk_out)
        .saturating_mul(channels)
        .min(MAX_F32_SAMPLES_PER_FILE);
    buf.resampled_f32.clear();
    if buf.resampled_f32.capacity() < interleaved_reserve {
        buf.resampled_f32
            .reserve(interleaved_reserve.saturating_sub(buf.resampled_f32.len()));
    }

    let mut in_storage = vec![0.0_f32; chunk_in.saturating_mul(channels)];
    let mut out_storage = vec![0.0_f32; chunk_out.saturating_mul(channels)];
    let mut consumed = 0_usize;
    let mut produced = 0_usize;
    for _ in 0..max_calls {
        if produced >= needed {
            break;
        }
        let take = chunk_in.min(input_frames.saturating_sub(consumed));
        in_storage.fill(0.0);
        if take > 0 {
            let start = consumed.saturating_mul(channels);
            let end = start.saturating_add(take.saturating_mul(channels));
            if let Some(source) = input.get(start..end) {
                in_storage[..source.len()].copy_from_slice(source);
            }
        }
        let in_view =
            InterleavedOwned::new_from(std::mem::take(&mut in_storage), channels, chunk_in)
                .map_err(|err| fail(format!("invalid resampler input: {err}")))?;
        let mut out_view =
            InterleavedOwned::new_from(std::mem::take(&mut out_storage), channels, chunk_out)
                .map_err(|err| fail(format!("invalid resampler output: {err}")))?;
        let indexing = Indexing::new().partial_len(take);
        let (_, written) = resampler
            .process_into_buffer(&in_view, &mut out_view, Some(&indexing))
            .map_err(|err| fail(format!("resampling failed: {err}")))?;
        in_storage = in_view.take_data();
        out_storage = out_view.take_data();
        let usable = written.saturating_mul(channels).min(out_storage.len());
        if buf.resampled_f32.len().saturating_add(usable) > MAX_F32_SAMPLES_PER_FILE {
            return Err(fail(format!(
                "resampled PCM exceeds the {MAX_PCM_BYTES_PER_FILE}-byte per-file limit"
            )));
        }
        buf.resampled_f32.extend_from_slice(&out_storage[..usable]);
        consumed = consumed.saturating_add(take);
        produced = produced.saturating_add(written);
    }

    drop(input);
    let start = delay.saturating_mul(channels);
    let end = start.saturating_add(expected.saturating_mul(channels));
    if buf.resampled_f32.len() < end {
        return Err(fail(format!(
            "resampler produced {} samples, expected at least {end}",
            buf.resampled_f32.len()
        )));
    }
    buf.resampled_f32.truncate(end);
    buf.resampled_f32.drain(..start);
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
    options: &AudioEncodingOptions,
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
    let (bitrate_bps, rate_control) = opus_settings(options.rate_control());
    encoder.bitrate_bps = bitrate_bps;
    encoder.rate_control = rate_control;
    encoder.complexity = 9;
    let head = OpusHead::for_encoder(&encoder, OPUS_RATE);
    let pre_skip = usize::from(head.pre_skip);
    let frame = usize_from(OPUS_RATE / (1000 / OPUS_FRAME_MS));
    let per_frame = frame.saturating_mul(channels);
    let packet_count = total_frames.saturating_add(pre_skip).div_ceil(frame);
    let final_granule = u64::from(u16::try_from(pre_skip).unwrap_or(u16::MAX))
        + u64::try_from(total_frames).unwrap_or(u64::MAX);
    let padded_samples = u64::try_from(packet_count)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::try_from(frame).unwrap_or(u64::MAX));
    // Pre-skip plus tail rounding pads by less than two Opus frames. Allow that
    // small, bounded slack above the decoded-PCM cap instead of rejecting a file
    // that fit before packet alignment; larger padding still fails up front.
    let padding_allowance = per_frame.saturating_mul(3);
    let padded_limit = MAX_F32_SAMPLES_PER_FILE.saturating_add(padding_allowance);
    let padded_samples = usize::try_from(padded_samples)
        .ok()
        .and_then(|frames| frames.checked_mul(channels))
        .filter(|&samples| samples <= padded_limit)
        .ok_or_else(|| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!(
                "Opus packet padding would exceed the {MAX_PCM_BYTES_PER_FILE}-byte per-file limit"
            ),
        })?;
    buf.samples_f32.resize(padded_samples, 0.0_f32);

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
    pcfg.bitrate_bps = bitrate_bps;
    pcfg.rate_control = rate_control;
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

fn mp3_config(rate_control: AudioRateControl) -> rusty_mp3::Mp3EncoderConfig {
    match rate_control {
        AudioRateControl::Default => rusty_mp3::Mp3EncoderConfig {
            bitrate_kbps: MP3_BITRATE_KBPS,
            vbr_quality: None,
        },
        AudioRateControl::Bitrate(bitrate) => rusty_mp3::Mp3EncoderConfig {
            bitrate_kbps: bitrate.kbps_rounded(),
            vbr_quality: None,
        },
        AudioRateControl::Quality(quality) => {
            let lame_q = 9.0 * (100.0 - f32::from(quality.value())) / 100.0;
            rusty_mp3::Mp3EncoderConfig {
                bitrate_kbps: 0,
                vbr_quality: Some(rusty_mp3::vbr_quality_index(lame_q)),
            }
        }
    }
}

fn aac_bitrate_bps(rate_control: AudioRateControl) -> u32 {
    match rate_control {
        AudioRateControl::Default | AudioRateControl::Quality(_) => AAC_BITRATE_BPS,
        AudioRateControl::Bitrate(bitrate) => bitrate.bps(),
    }
}

/// Rejects a bitrate the AAC frame cannot carry, using the effective rate and
/// channel count when the caller did not pin them explicitly.
fn check_aac_bitrate(
    path: &Path,
    rate: u32,
    channels: usize,
    options: &AudioEncodingOptions,
) -> Result<(), PipelineError> {
    if let AudioRateControl::Bitrate(bitrate) = options.rate_control() {
        let max_bps = aac_frame_capacity_bps(rate, u16::try_from(channels).unwrap_or(u16::MAX));
        if bitrate.bps() > max_bps {
            return Err(PipelineError::EncodeFailed {
                path: path.to_path_buf(),
                reason: format!(
                    "AAC bitrate {} bps at {rate} Hz with {channels} channels exceeds the frame capacity of {max_bps} bps",
                    bitrate.bps()
                ),
            });
        }
    }
    Ok(())
}

fn opus_settings(rate_control: AudioRateControl) -> (i32, RateControl) {
    match rate_control {
        AudioRateControl::Default => (OPUS_BITRATE, RateControl::ConstrainedVbr),
        AudioRateControl::Bitrate(bitrate) => (
            i32::try_from(bitrate.bps()).unwrap_or(i32::MAX),
            RateControl::ConstrainedVbr,
        ),
        AudioRateControl::Quality(quality) => {
            let bps = 16_000_i32.saturating_add(i32::from(quality.value()).saturating_mul(2_400));
            (bps, RateControl::Vbr)
        }
    }
}

fn vorbis_strategy(
    rate_control: AudioRateControl,
) -> Result<vorbis_enc::BitrateManagementStrategy, PipelineError> {
    let strategy = match rate_control {
        AudioRateControl::Default => vorbis_enc::BitrateManagementStrategy::QualityVbr {
            target_quality: 0.4,
        },
        AudioRateControl::Bitrate(bitrate) => {
            let average =
                NonZeroU32::new(bitrate.bps()).ok_or_else(|| PipelineError::EncodeFailed {
                    path: Path::new("ogg-bitrate").to_path_buf(),
                    reason: "Ogg average bitrate must be non-zero".to_owned(),
                })?;
            vorbis_enc::BitrateManagementStrategy::Abr {
                average_bitrate: average,
            }
        }
        AudioRateControl::Quality(quality) => {
            let target = (-0.1_f32 + f32::from(quality.value()) * 0.011_f32).clamp(-0.1, 1.0);
            vorbis_enc::BitrateManagementStrategy::QualityVbr {
                target_quality: target,
            }
        }
    };
    Ok(strategy)
}

fn aac_rate(rate: u32) -> u32 {
    if rusty_aac::sf_index_for_rate(rate).is_some() {
        rate
    } else {
        OPUS_RATE
    }
}

/// Single source of truth for the rate fed to the encoder. Explicit options win;
/// otherwise every format keeps the codec default it used before these options existed.
fn effective_sample_rate(
    output: AudioOutputFormat,
    rate: u32,
    options: &AudioEncodingOptions,
) -> u32 {
    let codec_default = match output {
        AudioOutputFormat::Opus => OPUS_RATE,
        AudioOutputFormat::Aac | AudioOutputFormat::M4a => aac_rate(rate),
        AudioOutputFormat::Mp3 => {
            if MP3_RATES.contains(&rate) {
                rate
            } else {
                OPUS_RATE
            }
        }
        AudioOutputFormat::Ogg => {
            if rate <= OGG_MAX_RATE {
                rate
            } else {
                OPUS_RATE
            }
        }
        AudioOutputFormat::Wav
        | AudioOutputFormat::Flac
        | AudioOutputFormat::Alac
        | AudioOutputFormat::Aiff => rate,
    };
    options
        .sample_rate()
        .map_or(codec_default, AudioOptionSampleRate::hz)
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

/// MPEG `channel_configuration` 4 is C/L/R/rear-C, which cannot carry WAV quad
/// (FL/FR/BL/BR): `rusty_aac` maps input 2 to the centre element and input 3 to
/// the rear-centre element, so those samples would be rendered by wrong speakers.
/// The 3/5/6-channel configurations do preserve WAV speaker order.
fn check_aac_channels(path: &Path, channels: usize, codec: &str) -> Result<(), PipelineError> {
    check_channels(path, channels, 1, 6, codec)?;
    if channels == 4 {
        return Err(PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!(
                "{codec} cannot represent 4-channel WAV quad (FL, FR, BL, BR): MPEG channel_configuration 4 is C, L, R, rear-C and would put the back channels on the centre and rear-centre speakers"
            ),
        });
    }
    Ok(())
}

fn encode_aac(
    path: &Path,
    rate: u32,
    channels: usize,
    buf: &mut AudioWorkerBuffer,
    options: &AudioEncodingOptions,
) -> Result<(), PipelineError> {
    check_aac_channels(path, channels, "AAC")?;
    let target_rate = effective_sample_rate(AudioOutputFormat::Aac, rate, options);
    check_aac_bitrate(path, target_rate, channels, options)?;
    resample_f32(rate, target_rate, channels, buf, "aac-resampler")?;
    if buf.samples_f32.is_empty() {
        return Err(PipelineError::EmptyOutput(path.to_path_buf()));
    }
    let ch_u16 = u16::try_from(channels).unwrap_or(u16::MAX);
    let mut encoder = rusty_aac::AacEncoder::new(rusty_aac::AacEncoderConfig {
        bitrate_bps: aac_bitrate_bps(options.rate_control()),
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
    options: &AudioEncodingOptions,
) -> Result<(), PipelineError> {
    check_aac_channels(path, channels, "M4A/AAC")?;
    let target_rate = effective_sample_rate(AudioOutputFormat::M4a, rate, options);
    check_aac_bitrate(path, target_rate, channels, options)?;
    resample_f32(rate, target_rate, channels, buf, "m4a-resampler")?;
    if buf.samples_f32.is_empty() {
        return Err(PipelineError::EmptyOutput(path.to_path_buf()));
    }
    let ch_u16 = u16::try_from(channels).unwrap_or(u16::MAX);
    let total_input_frames = u64::try_from(buf.samples_f32.len() / channels).unwrap_or(u64::MAX);
    let aac_bitrate = aac_bitrate_bps(options.rate_control());
    let mut encoder = rusty_aac::AacEncoder::new(rusty_aac::AacEncoderConfig {
        bitrate_bps: aac_bitrate,
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
        aac_bitrate,
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
    options: &AudioEncodingOptions,
) -> Result<(), PipelineError> {
    check_channels(path, channels, 1, 2, "MP3")?;
    let target_rate = effective_sample_rate(AudioOutputFormat::Mp3, rate, options);
    if target_rate < 32_000
        && matches!(
            options.rate_control(),
            AudioRateControl::Bitrate(bitrate) if bitrate.bps() > 160_000
        )
    {
        // Domain validation can only pair the bitrate with an explicitly requested
        // rate. With the codec-derived rate, reject here instead of letting the
        // encoder silently snap 320 kbps down to the MPEG-2/2.5 maximum of 160.
        return Err(PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!(
                "MP3 at {target_rate} Hz supports at most 160000 bps (MPEG-2/2.5 Layer III limit)"
            ),
        });
    }
    resample_f32(rate, target_rate, channels, buf, "mp3-resampler")?;
    if buf.samples_f32.is_empty() {
        return Err(PipelineError::EmptyOutput(path.to_path_buf()));
    }
    let ch_u16 = u16::try_from(channels).unwrap_or(u16::MAX);
    let mut encoder = rusty_mp3::Mp3Encoder::new(mp3_config(options.rate_control()));
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

/// Returns the source index feeding each Vorbis I plane (FL, FC, FR, BL, BR, LFE).
/// WAV/PCM 5.1 carries FL, FR, FC, LFE, BL, BR; the plane order differs from
/// three channels onward. Four channels are ambiguous (WAV quad FL/FR/BL/BR vs.
/// FL/FR/FC/LFE), so they are rejected instead of being silently mislabeled.
/// Expressed source-to-plane, the 6-channel permutation is `[0, 2, 1, 5, 3, 4]`.
fn vorbis_plane_sources(channels: usize) -> Option<&'static [usize]> {
    match channels {
        1 => Some(&[0]),
        2 => Some(&[0, 1]),
        3 => Some(&[0, 2, 1]),
        // 4 is deliberately absent: quad (FL/FR/BL/BR) and FL/FR/FC/LFE are
        // indistinguishable from the channel count alone.
        5 => Some(&[0, 2, 1, 3, 4]),
        6 => Some(&[0, 2, 1, 4, 5, 3]),
        _ => None,
    }
}

fn encode_ogg(
    path: &Path,
    rate: u32,
    channels: usize,
    buf: &mut AudioWorkerBuffer,
    options: &AudioEncodingOptions,
) -> Result<(), PipelineError> {
    check_channels(path, channels, 1, usize::from(u8::MAX), "Ogg Vorbis")?;
    let channel_order =
        vorbis_plane_sources(channels).ok_or_else(|| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!(
                "Ogg Vorbis cannot safely map {channels} channels: 4-channel layouts are ambiguous (quad FL/FR/BL/BR vs. FL/FR/FC/LFE) and layouts above 6 channels are unsupported"
            ),
        })?;
    if rate == 0 {
        return Err(PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: "Ogg Vorbis requires a positive sample rate".to_owned(),
        });
    }
    let target_rate = effective_sample_rate(AudioOutputFormat::Ogg, rate, options);
    resample_f32(rate, target_rate, channels, buf, "ogg-resampler")?;
    if buf.samples_f32.is_empty() {
        return Err(PipelineError::EmptyOutput(path.to_path_buf()));
    }
    let total_frames = buf.samples_f32.len() / channels;
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
        .bitrate_management_strategy(vorbis_strategy(options.rate_control())?)
        .build()
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("failed to create Ogg Vorbis encoder: {err}"),
        })?;
    let mut planes: Vec<Vec<f32>> = vec![Vec::with_capacity(OGG_CHUNK_FRAMES); channels];
    for start in (0..total_frames).step_by(OGG_CHUNK_FRAMES) {
        let frames = OGG_CHUNK_FRAMES.min(total_frames.saturating_sub(start));
        for plane in &mut planes {
            plane.clear();
        }
        for frame_samples in buf
            .samples_f32
            .chunks_exact(channels)
            .skip(start)
            .take(frames)
        {
            for (plane, &source) in planes.iter_mut().zip(channel_order) {
                if let Some(&value) = frame_samples.get(source) {
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
    batch_workers: usize,
) -> Result<(), PipelineError> {
    let available = std::thread::available_parallelism()
        .map_or(1, NonZeroUsize::get)
        .min(8);
    let inner_threads = (available / batch_workers.max(1)).max(1);
    crate::m4a::encode_alac_m4a(path, rate, channels, &buf.samples_i16, inner_threads)
}

fn encode_aiff(
    path: &Path,
    rate: u32,
    channels: usize,
    depth: AudioBitDepth,
    buf: &AudioWorkerBuffer,
) -> Result<(), PipelineError> {
    crate::aiff::encode_aiff(
        path,
        rate,
        channels,
        depth,
        &buf.samples_i16,
        &buf.samples_i32,
        &buf.samples_f32,
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

/// Bundles the batch-level encoder inputs so `encode_audio` stays within the
/// project's argument-count limit without hiding either value behind a global.
struct AudioEncodeSettings<'a> {
    options: &'a AudioEncodingOptions,
    batch_workers: usize,
}

/// Decoded source PCM metadata needed to pick the target depth.
#[derive(Debug, Clone, Copy)]
struct AudioSourcePcm {
    bits: usize,
    sample_format: AudioSampleFormat,
}

fn encode_audio(
    item: &AudioBatchItem,
    mode: fastconv_domain::AudioExecutionMode,
    rate: u32,
    channels: usize,
    source: AudioSourcePcm,
    buf: &mut AudioWorkerBuffer,
    settings: &AudioEncodeSettings<'_>,
) -> Result<u64, PipelineError> {
    let options = settings.options;
    let batch_workers = settings.batch_workers;
    let output = mode.output_format();
    let depth = resolve_target_depth(
        item.source(),
        output,
        source.bits,
        source.sample_format,
        options,
    )?;
    // A direct integer decode leaves exactly one integer buffer populated. Passing
    // it through avoids a lossy f32 round trip that would truncate low bits.
    let direct_i16 =
        !buf.samples_i16.is_empty() && buf.samples_f32.is_empty() && buf.samples_i32.is_empty();
    let direct_i32 = !buf.samples_i32.is_empty() && buf.samples_f32.is_empty();
    if !direct_i16 && !direct_i32 && output.is_lossless() {
        quantize_for(output, depth, buf);
    }
    if direct_i16 && output == AudioOutputFormat::Flac {
        buf.samples_i32.clear();
        buf.samples_i32
            .extend(buf.samples_i16.iter().copied().map(i32::from));
    }
    let flac_bits = usize::from(depth.bits());
    encode_staged(item, |temp| match output {
        AudioOutputFormat::Wav => encode_wav(temp, rate, channels, depth, buf),
        AudioOutputFormat::Flac => encode_flac(temp, rate, channels, flac_bits, buf, batch_workers),
        AudioOutputFormat::Opus => encode_opus(temp, rate, channels, buf, batch_workers, options),
        AudioOutputFormat::Ogg => encode_ogg(temp, rate, channels, buf, options),
        AudioOutputFormat::Mp3 => encode_mp3(temp, rate, channels, buf, options),
        AudioOutputFormat::Aac => encode_aac(temp, rate, channels, buf, options),
        AudioOutputFormat::M4a => encode_m4a(temp, rate, channels, buf, options),
        AudioOutputFormat::Alac => encode_alac(temp, rate, channels, buf, batch_workers),
        AudioOutputFormat::Aiff => encode_aiff(temp, rate, channels, depth, buf),
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
    options: &AudioEncodingOptions,
) -> Result<(u64, u64), PipelineError> {
    let input_bytes = fs::metadata(item.source())
        .map_err(|source| PipelineError::Io {
            path: item.source().to_path_buf(),
            source,
        })?
        .len();
    let output = mode.output_format();
    if item.input_format() == AudioInputFormat::Wav
        && output == AudioOutputFormat::Flac
        && let Some(stream) = try_open_pcm_wav(item.source())?
    {
        // The streaming fast path preserves the source rate/channels exactly; any
        // transform needs the full decode/quantize pipeline instead.
        let transforming = options
            .sample_rate()
            .is_some_and(|rate| rate.hz() != stream.rate)
            || options
                .channels()
                .is_some_and(|channels| usize::from(channels.get()) != stream.channels)
            || options
                .bit_depth()
                .is_some_and(|depth| usize::from(depth.bits()) != stream.bits);
        if !transforming {
            let output_bytes = convert_pcm_wav_to_flac(item, stream, buf, batch_workers)?;
            return Ok((input_bytes, output_bytes));
        }
    }
    let (rate, channels, bits, sample_format) = decode_audio(item, output, options, buf)?;
    let wanted_channels = options
        .channels()
        .map_or(channels, |channels| usize::from(channels.get()));

    // Downmix before resampling: the FFT resampler then runs with the smaller plane
    // count, which keeps its scratch buffers proportional to the encoded output.
    let mut current_channels = channels;
    if wanted_channels < current_channels {
        remix_channels_f32(current_channels, wanted_channels, buf, item.source())?;
        current_channels = wanted_channels;
    }
    let target_rate = effective_sample_rate(output, rate, options);
    if target_rate != rate {
        let label = match output {
            AudioOutputFormat::Wav => "wav-resampler",
            AudioOutputFormat::Flac => "flac-resampler",
            AudioOutputFormat::Alac => "alac-resampler",
            AudioOutputFormat::Aiff => "aiff-resampler",
            AudioOutputFormat::Opus => "opus-resampler",
            AudioOutputFormat::Ogg => "ogg-resampler",
            AudioOutputFormat::Mp3 => "mp3-resampler",
            AudioOutputFormat::Aac => "aac-resampler",
            AudioOutputFormat::M4a => "m4a-resampler",
        };
        resample_f32(rate, target_rate, current_channels, buf, label)?;
        if buf.samples_f32.is_empty() {
            // A very short input can resample to zero frames; writing a header-only
            // file would report a successful conversion with no audio in it.
            return Err(PipelineError::EmptyOutput(item.target().to_path_buf()));
        }
    }
    // Upmix last: extra planes are generated at the final sample rate.
    if wanted_channels > current_channels {
        remix_channels_f32(current_channels, wanted_channels, buf, item.source())?;
    }
    let output_bytes = encode_audio(
        item,
        mode,
        target_rate,
        wanted_channels,
        AudioSourcePcm {
            bits,
            sample_format,
        },
        buf,
        &AudioEncodeSettings {
            options,
            batch_workers,
        },
    )?;
    Ok((input_bytes, output_bytes))
}

/// Peak live PCM for a batch item. `largest_buffer_bytes` is the biggest single
/// allocation and is checked against the per-file cap; `peak_bytes` is the sum of
/// buffers that coexist during a transform and is used for the batch budget.
struct PcmEstimate {
    peak_bytes: u64,
    largest_buffer_bytes: u64,
}

fn estimate_pcm_bytes(
    item: &AudioBatchItem,
    output: AudioOutputFormat,
    options: &AudioEncodingOptions,
) -> PcmEstimate {
    let stream = item.stream();
    let unknown = PcmEstimate {
        // Without a frame count we cannot prove a rate change is absent, so reserve
        // the decoded input and a simultaneous resample output.
        peak_bytes: MAX_PCM_BYTES_PER_FILE.saturating_mul(2),
        largest_buffer_bytes: MAX_PCM_BYTES_PER_FILE,
    };
    let Some(frames) = stream.total_samples() else {
        // A compressed file size is not a PCM frame count: probed AAC streams, for
        // example, expose no decoded length. Reserve the full per-file allowance so
        // unknown-length inputs cannot over-subscribe the batch memory budget.
        return unknown;
    };
    let source_rate = stream.sample_rate().get();
    let target_rate = effective_sample_rate(output, source_rate, options);
    let source_channels = u64::from(stream.channels().get()).max(1);
    let target_channels = u64::from(
        options
            .channels()
            .map_or(stream.channels().get(), AudioOptionChannels::get),
    )
    .max(1);
    let bytes =
        |frame_count: u64, channels: u64| frame_count.saturating_mul(channels).saturating_mul(4);
    let scaled_frames = frames
        .saturating_mul(u64::from(target_rate))
        .div_ceil(u64::from(source_rate).max(1));
    let source_bytes = bytes(frames, source_channels);
    let final_bytes = bytes(scaled_frames, target_channels);
    let mut largest_buffer_bytes = source_bytes.max(final_bytes);
    let mut peak_bytes = largest_buffer_bytes;

    if target_rate != source_rate {
        // The decoded input remains live while the resampler grows its output.
        // Downmixes run before resampling and upmixes after it, so the plane count
        // at that point is the smaller of the source and target counts.
        let resample_channels = source_channels.min(target_channels);
        let resampled_bytes = bytes(scaled_frames, resample_channels);
        largest_buffer_bytes = largest_buffer_bytes.max(resampled_bytes);
        let resample_input_bytes = if target_channels < source_channels {
            bytes(frames, target_channels)
        } else {
            source_bytes
        };
        peak_bytes = peak_bytes
            .max(resample_input_bytes.saturating_add(resampled_bytes))
            .max(resampled_bytes.saturating_add(final_bytes));
    }
    if source_channels != target_channels {
        // A downmix runs before resampling and sizes its output from the input
        // frame count; an upmix runs after resampling and uses the final count.
        // A remix holds its input and output at the same time.
        let remix_output_bytes = if target_channels < source_channels {
            bytes(frames, target_channels)
        } else {
            final_bytes
        };
        peak_bytes = peak_bytes.max(source_bytes.saturating_add(remix_output_bytes));
    }
    PcmEstimate {
        peak_bytes,
        largest_buffer_bytes,
    }
}

fn resolve_worker_count(
    items: &[AudioBatchItem],
    requested: usize,
    output: AudioOutputFormat,
    options: &AudioEncodingOptions,
) -> Result<usize, PipelineError> {
    let estimates: Vec<PcmEstimate> = items
        .iter()
        .map(|item| estimate_pcm_bytes(item, output, options))
        .collect();
    let largest_buffer = estimates
        .iter()
        .map(|estimate| estimate.largest_buffer_bytes)
        .max()
        .unwrap_or(0);
    if largest_buffer > MAX_PCM_BYTES_PER_FILE {
        return Err(PipelineError::DecodeFailed {
            path: items[0].source().to_path_buf(),
            reason: format!(
                "estimated PCM buffer {largest_buffer} bytes exceeds the {MAX_PCM_BYTES_PER_FILE} byte per-file limit"
            ),
        });
    }
    let largest_peak = estimates
        .iter()
        .map(|estimate| estimate.peak_bytes)
        .max()
        .unwrap_or(0);
    let budget_workers = u64::try_from(requested)
        .unwrap_or(u64::MAX)
        .min(MAX_PCM_BYTES_PER_BATCH / largest_peak.max(1));
    Ok(usize::try_from(budget_workers.max(1))
        .unwrap_or(usize::MAX)
        .min(requested))
}

/// Shared queue, cancellation flag and totals owned by the batch coordinator.
struct AudioWorkerShared<'a> {
    next: &'a AtomicUsize,
    abort: &'a AtomicBool,
    first_error: &'a Mutex<Option<PipelineError>>,
    input_total: &'a AtomicU64,
    output_total: &'a AtomicU64,
}

/// Drains the shared work queue for one worker until it is empty or another worker fails.
fn run_audio_worker(
    items: &[AudioBatchItem],
    mode: fastconv_domain::AudioExecutionMode,
    options: &AudioEncodingOptions,
    batch_workers: usize,
    shared: &AudioWorkerShared<'_>,
) {
    let mut buffer = AudioWorkerBuffer::new();
    loop {
        if shared.abort.load(Ordering::Relaxed)
            || shared.first_error.lock().is_ok_and(|guard| guard.is_some())
        {
            break;
        }
        let index = shared.next.fetch_add(1, Ordering::Relaxed);
        let Some(item) = items.get(index) else { break };
        match convert_item(item, mode, &mut buffer, batch_workers, options) {
            Ok((in_bytes, out_bytes)) => {
                shared.input_total.fetch_add(in_bytes, Ordering::Relaxed);
                shared.output_total.fetch_add(out_bytes, Ordering::Relaxed);
            }
            Err(err) => {
                if let Ok(mut guard) = shared.first_error.lock()
                    && guard.is_none()
                {
                    *guard = Some(err);
                }
                break;
            }
        }
    }
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
    let options = plan.options();
    let shared = AudioWorkerShared {
        next: &next,
        abort: &abort,
        first_error: &first_error,
        input_total: &input_total,
        output_total: &output_total,
    };
    let worker_count = resolve_worker_count(
        items,
        plan.effective_workers().get(),
        plan.mode().output_format(),
        &options,
    )?;
    // Idle workers must not shrink the inner-encoder budget of the active files.

    thread::scope(|scope| {
        let shared = &shared;
        let mut handles = Vec::with_capacity(worker_count);
        let mut startup_error = None;
        for worker_id in 0..worker_count {
            let worker = thread::Builder::new()
                .name(format!("fastconv-audio-{worker_id}"))
                .spawn_scoped(scope, move || {
                    run_audio_worker(
                        items,
                        plan.mode(),
                        &options,
                        worker_count.min(items.len()),
                        shared,
                    );
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

    let input = NonZeroU64::new(input_total.load(Ordering::Relaxed))
        .ok_or_else(|| PipelineError::EmptyOutput(items[0].source().to_path_buf()))?;
    let output = NonZeroU64::new(output_total.load(Ordering::Relaxed))
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
    use fastconv_domain::{AudioOptionBitrate, AudioQuality};
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

    fn item_frames(
        source: PathBuf,
        target: PathBuf,
        format: AudioInputFormat,
        channels: u16,
        rate: u32,
        total_samples: u64,
    ) -> AudioBatchItem {
        AudioBatchItem::new(
            source,
            target,
            format,
            fastconv_domain::AudioStreamInfo::new(
                NonZeroU16::new(channels).unwrap_or(NonZeroU16::MIN),
                NonZeroU32::new(rate).unwrap_or(NonZeroU32::MIN),
                NonZeroU16::new(16),
                Some(total_samples),
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
    fn resample_f32_same_rate_is_a_noop() -> Result<(), PipelineError> {
        let mut buffer = AudioWorkerBuffer::new();
        buffer.samples_f32.extend([0.1_f32, -0.2, 0.3, -0.4]);
        resample_f32(48_000, 48_000, 2, &mut buffer, "resample-test")?;
        assert_eq!(buffer.samples_f32, [0.1_f32, -0.2, 0.3, -0.4]);
        Ok(())
    }

    #[test]
    fn resample_f32_produces_exact_frame_count_for_edge_inputs() -> Result<(), PipelineError> {
        for channels in [1_usize, 2, 3, 6] {
            for frames in [0_usize, 1, 137, 2_049] {
                let mut buffer = AudioWorkerBuffer::new();
                buffer
                    .samples_f32
                    .extend((0..frames.saturating_mul(channels)).map(|index| {
                        let step = i16::try_from(index % 251).unwrap_or(0);
                        f32::from(step) / 251.0_f32 - 0.5_f32
                    }));
                resample_f32(44_100, 48_000, channels, &mut buffer, "resample-test")?;
                let expected = usize::try_from(
                    u64::try_from(frames)
                        .unwrap_or(u64::MAX)
                        .saturating_mul(48_000)
                        / 44_100,
                )
                .unwrap_or(usize::MAX);
                assert_eq!(
                    buffer.samples_f32.len(),
                    expected.saturating_mul(channels),
                    "channels={channels}, frames={frames}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn resample_f32_keeps_signal_after_trimming_the_delay() -> Result<(), PipelineError> {
        let mut buffer = AudioWorkerBuffer::new();
        buffer.samples_f32.extend((0..8_000_usize).map(|index| {
            let step = i16::try_from(index % 251).unwrap_or(0);
            f32::from(step) / 251.0_f32 - 0.5_f32
        }));
        resample_f32(44_100, 48_000, 1, &mut buffer, "resample-test")?;
        assert!(
            buffer.samples_f32.iter().any(|&sample| sample.abs() > 0.1),
            "trimming the FFT startup delay must not remove the whole signal"
        );
        Ok(())
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

        // Known frame counts keep the estimated PCM well below the budget, so all
        // three requested workers are available for this parallelism check.
        let mixed = AudioBatchPlan::new(
            vec![
                item_frames(
                    wav.clone(),
                    out.join("a.wav"),
                    AudioInputFormat::Wav,
                    2,
                    48_000,
                    8_000,
                ),
                item_frames(
                    flac,
                    out.join("b.wav"),
                    AudioInputFormat::Flac,
                    2,
                    48_000,
                    8_000,
                ),
                item_frames(
                    opus,
                    out.join("c.wav"),
                    AudioInputFormat::Opus,
                    2,
                    48_000,
                    8_000,
                ),
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
    fn unknown_stream_length_caps_batch_workers() {
        let items: Vec<_> = (0..8)
            .map(|index| {
                item(
                    PathBuf::from(format!("/nonexistent/in-{index}.aac")),
                    PathBuf::from(format!("/nonexistent/out-{index}.wav")),
                    AudioInputFormat::Aac,
                    2,
                    48_000,
                )
            })
            .collect();
        let defaults = AudioEncodingOptions::default();
        let estimate = estimate_pcm_bytes(&items[0], AudioOutputFormat::Wav, &defaults);
        assert_eq!(estimate.largest_buffer_bytes, MAX_PCM_BYTES_PER_FILE);
        assert_eq!(
            estimate.peak_bytes,
            MAX_PCM_BYTES_PER_FILE.saturating_mul(2)
        );
        let workers =
            resolve_worker_count(&items, 8, AudioOutputFormat::Wav, &defaults).unwrap_or(0);
        assert_eq!(
            workers, 1,
            "unknown decoded lengths must reserve two coexisting buffers"
        );
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

    fn assert_four_channel_rejected(
        label: &str,
        extension: &str,
        mode: fastconv_domain::AudioExecutionMode,
        expected_reason: &str,
    ) {
        let dir = temp_dir(label);
        let wav = dir.join("in.wav");
        let out = dir.join(format!("out.{extension}"));
        write_wav(&wav, &tone(1_024, 4), 48_000, 4);
        let result = execute_audio_batch(&plan(
            vec![item(wav, out, AudioInputFormat::Wav, 4, 48_000)],
            mode,
        ));
        assert!(
            matches!(result, Err(PipelineError::EncodeFailed { .. })),
            "4-channel {extension} must fail with EncodeFailed, got {result:?}"
        );
        let Err(PipelineError::EncodeFailed { reason, .. }) = result else {
            return;
        };
        assert!(
            reason.contains(expected_reason),
            "4-channel {extension} must be rejected with a clear reason, got: {reason}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_four_channel_aac_and_m4a() {
        assert_four_channel_rejected(
            "quad-aac",
            "aac",
            fastconv_domain::AudioExecutionMode::CpuAac,
            "quad",
        );
        assert_four_channel_rejected(
            "quad-m4a",
            "m4a",
            fastconv_domain::AudioExecutionMode::CpuM4a,
            "quad",
        );
    }

    #[test]
    fn rejects_four_channel_ogg() {
        assert_four_channel_rejected(
            "quad-ogg",
            "ogg",
            fastconv_domain::AudioExecutionMode::CpuOgg,
            "ambiguous",
        );
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

    fn ogg_packets(path: &Path) -> Result<Vec<ogg::Packet>, PipelineError> {
        let file = File::open(path).map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut reader = ogg::reading::PacketReader::new(file);
        let mut packets = Vec::new();
        loop {
            match reader.read_packet() {
                Ok(Some(packet)) => packets.push(packet),
                Ok(None) => break,
                Err(err) => {
                    return Err(PipelineError::DecodeFailed {
                        path: path.to_path_buf(),
                        reason: format!("invalid Ogg page: {err}"),
                    });
                }
            }
        }
        Ok(packets)
    }

    fn best_snr_db(reference: &[f32], decoded: &[f32], max_lag: isize, channels: usize) -> f64 {
        // Interleaved samples carry channel identity in index parity, so only
        // whole-frame shifts may be searched. Odd shifts would line a channel up
        // with its neighbour and let swapped stereo pass the fidelity check.
        let channels = isize::try_from(channels.max(1)).unwrap_or(1);
        let max_frames = max_lag / channels;
        let count = 4096.min(reference.len()).min(decoded.len());
        let mut best = f64::NEG_INFINITY;
        for frame_lag in -max_frames..=max_frames {
            let lag = frame_lag.saturating_mul(channels);
            let mut signal = 0.0_f64;
            let mut error = 0.0_f64;
            for (index, &decoded_value) in decoded.iter().enumerate().take(count) {
                let decoded_index = isize::try_from(index).unwrap_or(isize::MAX) + max_lag;
                let reference_index = decoded_index + lag;
                if reference_index < 0 {
                    continue;
                }
                let Ok(reference_index) = usize::try_from(reference_index) else {
                    continue;
                };
                let Some(&original) = reference.get(reference_index) else {
                    break;
                };
                let value = f64::from(decoded_value);
                signal += f64::from(original) * f64::from(original);
                let delta = value - f64::from(original);
                error += delta * delta;
            }
            if error > 0.0 && signal > 0.0 {
                best = best.max(10.0 * (signal / error).log10());
            }
        }
        best
    }

    #[test]
    fn ogg_stereo_roundtrip_has_valid_ogg_paging() -> Result<(), PipelineError> {
        const FRAMES: usize = 16_384;
        let dir = temp_dir("ogg-stereo-paging");
        let ogg = dir.join("out.ogg");
        // Smooth parabolic sweeps, as used by the benchmark fixture, keep the
        // fidelity assertion meaningful for a q4-class Vorbis encoder.
        let parabolic = |phase: u16| -> i64 {
            let signed = i64::from(phase) - 32_768;
            let abs = signed.abs();
            (signed * (32_768 - abs)) >> 14
        };
        let mut phase_l = 0_u16;
        let mut phase_r = 8_192_u16;
        let samples: Vec<i16> = (0..FRAMES)
            .flat_map(|_| {
                phase_l = phase_l.wrapping_add(300);
                phase_r = phase_r.wrapping_add(450);
                [
                    i16::try_from(parabolic(phase_l)).unwrap_or(0),
                    i16::try_from(parabolic(phase_r)).unwrap_or(0),
                ]
            })
            .collect();
        let mut encoder_buf = AudioWorkerBuffer::new();
        encoder_buf
            .samples_f32
            .extend(samples.iter().map(|&value| f32::from(value) / 32_768.0));
        encode_ogg(
            &ogg,
            48_000,
            2,
            &mut encoder_buf,
            &AudioEncodingOptions::default(),
        )?;

        let packets = ogg_packets(&ogg)?;
        assert!(packets.len() > 4, "expected ident/comment/setup and audio");
        assert!(packets[0].first_in_stream() && packets[0].last_in_page());
        assert_eq!(packets[0].absgp_page(), 0);
        assert_ne!(packets[0].stream_serial(), 0, "Ogg serial must be non-zero");
        assert_eq!(packets[0].data.first().copied(), Some(1), "ident header");
        assert_eq!(packets[1].data.first().copied(), Some(3), "comment header");
        assert!(!packets[1].last_in_page(), "comment shares the setup page");
        assert!(packets[2].last_in_page(), "setup ends its page");
        assert_eq!(packets[2].absgp_page(), 0);
        let mut previous_granule = 0_u64;
        for packet in packets.iter().skip(3) {
            assert_eq!(packet.stream_serial(), packets[0].stream_serial());
            assert!(
                packet.absgp_page() >= previous_granule,
                "Ogg granules must not decrease"
            );
            previous_granule = packet.absgp_page();
        }
        assert!(packets.last().is_some_and(ogg::Packet::last_in_stream));
        assert_eq!(previous_granule, u64::try_from(FRAMES).unwrap_or(u64::MAX));

        let mut decoded_buf = AudioWorkerBuffer::new();
        let (rate, channels, _, _) = decode_symphonia(
            &ogg,
            AudioInputFormat::Vorbis,
            AudioOutputFormat::Wav,
            &AudioEncodingOptions::default(),
            &mut decoded_buf,
        )?;
        assert_eq!(rate, 48_000);
        assert_eq!(channels, 2);
        assert_eq!(decoded_buf.samples_f32.len() / channels, FRAMES);
        assert!(
            decoded_buf
                .samples_f32
                .iter()
                .all(|value| value.is_finite() && value.abs() <= 1.05),
            "decoded Vorbis samples must be finite and unclipped"
        );
        let snr = best_snr_db(
            &encoder_buf.samples_f32,
            &decoded_buf.samples_f32,
            2048,
            channels,
        );
        assert!(snr > 25.0, "Vorbis q4 SNR too low: {snr:.1} dB");

        // Negative control: swapping left/right must not pass, because a whole-frame
        // lag keeps each decoded sample on its original channel.
        let swapped: Vec<f32> = encoder_buf
            .samples_f32
            .chunks_exact(channels)
            .flat_map(|frame| [frame[1], frame[0]])
            .collect();
        let swapped_snr = best_snr_db(&swapped, &decoded_buf.samples_f32, 2048, channels);
        assert!(
            swapped_snr < 25.0,
            "swapped stereo placement must be rejected: {swapped_snr:.1} dB"
        );
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn ogg_mono_and_short_audio_use_the_plain_vorbis_path() -> Result<(), PipelineError> {
        let dir = temp_dir("ogg-fallback");
        let mono = dir.join("mono.ogg");
        let mut mono_buf = AudioWorkerBuffer::new();
        mono_buf
            .samples_f32
            .extend(tone(8_192, 1).iter().map(|&v| f32::from(v) / 32_768.0));
        encode_ogg(
            &mono,
            48_000,
            1,
            &mut mono_buf,
            &AudioEncodingOptions::default(),
        )?;
        let mut decoded = AudioWorkerBuffer::new();
        let (rate, channels, _, _) = decode_symphonia(
            &mono,
            AudioInputFormat::Vorbis,
            AudioOutputFormat::Wav,
            &AudioEncodingOptions::default(),
            &mut decoded,
        )?;
        assert_eq!((rate, channels), (48_000, 1));
        assert_ne!(decoded.samples_f32.len(), 0);

        let short = dir.join("short.ogg");
        let mut short_buf = AudioWorkerBuffer::new();
        short_buf
            .samples_f32
            .extend(tone(2_048, 2).iter().map(|&v| f32::from(v) / 32_768.0));
        encode_ogg(
            &short,
            48_000,
            2,
            &mut short_buf,
            &AudioEncodingOptions::default(),
        )?;
        let mut decoded_short = AudioWorkerBuffer::new();
        let (rate, channels, _, _) = decode_symphonia(
            &short,
            AudioInputFormat::Vorbis,
            AudioOutputFormat::Wav,
            &AudioEncodingOptions::default(),
            &mut decoded_short,
        )?;
        assert_eq!((rate, channels), (48_000, 2));
        assert_ne!(decoded_short.samples_f32.len(), 0);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn vorbis_plane_sources_map_pcm_layouts() {
        let identity: &[&[usize]] = &[&[0], &[0, 1]];
        for (channels, expected) in [1_usize, 2].iter().zip(identity) {
            assert_eq!(vorbis_plane_sources(*channels), Some(*expected));
        }
        assert_eq!(vorbis_plane_sources(3), Some(&[0, 2, 1][..]));
        assert_eq!(vorbis_plane_sources(4), None, "quad layout is ambiguous");
        assert_eq!(vorbis_plane_sources(5), Some(&[0, 2, 1, 3, 4][..]));
        assert_eq!(vorbis_plane_sources(6), Some(&[0, 2, 1, 4, 5, 3][..]));
        assert_eq!(vorbis_plane_sources(0), None);
        assert_eq!(vorbis_plane_sources(7), None);

        // Inverting the plane-to-source mapping yields the documented
        // source-to-plane permutation, including the non-involutive 5.1 case.
        let mut source_to_plane = vec![0_usize; 6];
        for (plane, &source) in vorbis_plane_sources(6).unwrap_or(&[]).iter().enumerate() {
            if let Some(slot) = source_to_plane.get_mut(source) {
                *slot = plane;
            }
        }
        assert_eq!(source_to_plane, [0, 2, 1, 5, 3, 4]);
    }

    #[test]
    fn ogg_rejects_channel_counts_without_a_known_layout() {
        let mut buf = AudioWorkerBuffer::new();
        buf.samples_f32 = vec![0.0; 4];
        let result = encode_ogg(
            Path::new("reject4.ogg"),
            48_000,
            4,
            &mut buf,
            &AudioEncodingOptions::default(),
        );
        assert!(
            matches!(result, Err(PipelineError::EncodeFailed { .. })),
            "ambiguous 4-channel layout must be rejected instead of mislabeled"
        );
        buf.samples_f32 = vec![0.0; 7];
        let result = encode_ogg(
            Path::new("reject.ogg"),
            48_000,
            7,
            &mut buf,
            &AudioEncodingOptions::default(),
        );
        assert!(
            matches!(result, Err(PipelineError::EncodeFailed { .. })),
            "more than 6 channels must be rejected instead of mislabeled"
        );
    }

    #[test]
    fn roundtrips_wav_6ch_ogg_with_vorbis_channel_order() -> Result<(), PipelineError> {
        const CHANNELS: usize = 6;
        let dir = temp_dir("ogg-6ch-order");
        let wav = dir.join("in.wav");
        let ogg = dir.join("out.ogg");
        let back = dir.join("back.wav");
        let frames = 8_192_usize;
        // Integer-cycle sines stay orthogonal over the block, so a dot product still
        // identifies each source channel after lossy Vorbis coding (DC levels bleed).
        let bins = [40_u16, 80, 120, 160, 200, 240];
        let frames_f32 = f32::from(u16::try_from(frames).unwrap_or(u16::MAX));
        let two_pi = 2.0_f32 * std::f32::consts::PI;
        let steps: Vec<f32> = bins
            .iter()
            .map(|&bin| two_pi * f32::from(bin) / frames_f32)
            .collect();
        let mut references: Vec<Vec<f32>> =
            (0..CHANNELS).map(|_| Vec::with_capacity(frames)).collect();
        let mut phases = [0.0_f32; CHANNELS];
        let mut samples = Vec::with_capacity(frames.saturating_mul(CHANNELS));
        for _ in 0..frames {
            for ((reference, phase), &step) in references.iter_mut().zip(&mut phases).zip(&steps) {
                let sine = phase.sin();
                reference.push(sine);
                samples.push(i16::try_from(to_i32_saturating(sine * 12_000.0)).unwrap_or(0));
                let next = *phase + step;
                *phase = if next >= two_pi { next - two_pi } else { next };
            }
        }
        let channels_u16 = u16::try_from(CHANNELS).unwrap_or(u16::MAX);
        write_wav(&wav, &samples, 48_000, channels_u16);

        execute_audio_batch(&plan(
            vec![item(
                wav,
                ogg.clone(),
                AudioInputFormat::Wav,
                channels_u16,
                48_000,
            )],
            fastconv_domain::AudioExecutionMode::CpuOgg,
        ))?;
        execute_audio_batch(&plan(
            vec![item(
                ogg,
                back.clone(),
                AudioInputFormat::Vorbis,
                channels_u16,
                48_000,
            )],
            fastconv_domain::AudioExecutionMode::CpuWav,
        ))?;

        let decoded = read_wav(&back);
        assert_eq!(decoded.len(), samples.len());
        // WAV 5.1 is FL, FR, FC, LFE, BL, BR; Vorbis writes FL, FC, FR, BL, BR, LFE.
        // Symphonia's Vorbis decoder maps those planes back to WAV order, so a correct
        // encoder permutation makes this round trip preserve the original placement.
        for (plane, expected) in (0..CHANNELS).enumerate() {
            let mut scores: Vec<(f64, usize)> = references
                .iter()
                .enumerate()
                .map(|(source, reference)| {
                    let score: f64 = decoded
                        .iter()
                        .skip(plane)
                        .step_by(CHANNELS)
                        .zip(reference)
                        .map(|(&value, &sine)| f64::from(value) * f64::from(sine))
                        .sum();
                    (score.abs(), source)
                })
                .collect();
            scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let (best, detected) = scores.first().copied().unwrap_or((0.0, 0));
            let runner_up = scores.get(1).map_or(0.0, |&(score, _)| score);
            assert_eq!(
                detected, expected,
                "Vorbis plane {plane} should carry source channel {expected}; scores {scores:?}"
            );
            assert!(
                best > runner_up * 1.2,
                "plane {plane} tone must dominate its competitors: {scores:?}"
            );
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    fn br(bps: u32) -> AudioOptionBitrate {
        AudioOptionBitrate::new(bps).unwrap_or_else(|_| unreachable!())
    }

    fn ql(value: u8) -> AudioQuality {
        AudioQuality::new(value).unwrap_or_else(|_| unreachable!())
    }

    #[test]
    fn checked_f32_capacity_rejects_oversized_transform_before_allocation() {
        let path = Path::new("cap.wav");
        assert_eq!(checked_f32_capacity(1_000, 2, path).unwrap_or(0), 2_000);
        assert_eq!(checked_f32_capacity(u64::MAX, 8, path).unwrap_or(0), 0);
        assert!(checked_f32_capacity(u64::MAX, 8, path).is_err());
        assert!(
            checked_f32_capacity(
                u64::try_from(MAX_F32_SAMPLES_PER_FILE).unwrap_or(u64::MAX) + 1,
                1,
                path
            )
            .is_err()
        );
    }

    #[test]
    fn remix_channels_handles_edge_layouts_without_clipping() -> Result<(), PipelineError> {
        let path = Path::new("remix.wav");
        let mut buf = AudioWorkerBuffer::new();

        buf.samples_f32 = vec![1.0, -1.0, 0.5, 0.5];
        remix_channels_f32(2, 1, &mut buf, path)?;
        assert_eq!(buf.samples_f32, [0.0, 0.5]);

        buf.samples_f32 = vec![0.25, -0.5];
        remix_channels_f32(1, 2, &mut buf, path)?;
        assert_eq!(buf.samples_f32, [0.25, 0.25, -0.5, -0.5]);

        // Full-scale 5.1 must stay inside [-1, 1] after the normalized downmix.
        buf.samples_f32 = vec![1.0; 6 * 3];
        remix_channels_f32(6, 2, &mut buf, path)?;
        assert_eq!(buf.samples_f32.len(), 6);
        assert!(
            buf.samples_f32
                .iter()
                .all(|&value| value.is_finite() && value.abs() <= 1.0 + 1e-6),
            "normalized 5.1 downmix must not clip: {:?}",
            buf.samples_f32
        );
        assert!(
            buf.samples_f32
                .iter()
                .all(|&value| (value - 1.0).abs() < 1e-6),
            "all channels at full scale must stay at full scale: {:?}",
            buf.samples_f32
        );

        // LFE is omitted for 6/8-channel layouts: a lone LFE signal is silence.
        buf.samples_f32 = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        remix_channels_f32(6, 2, &mut buf, path)?;
        assert_eq!(buf.samples_f32, [0.0, 0.0]);

        // Mono upmix duplicates into FL/FR; general upmix copies and zero-fills.
        buf.samples_f32 = vec![0.75, -0.25];
        remix_channels_f32(1, 5, &mut buf, path)?;
        assert_eq!(
            buf.samples_f32,
            [0.75, 0.75, 0.0, 0.0, 0.0, -0.25, -0.25, 0.0, 0.0, 0.0]
        );
        buf.samples_f32 = vec![0.1, 0.2, 0.3];
        remix_channels_f32(3, 5, &mut buf, path)?;
        assert_eq!(buf.samples_f32, [0.1, 0.2, 0.3, 0.0, 0.0]);

        // An odd trailing partial frame cannot panic or produce extra samples.
        buf.samples_f32 = vec![0.9, 0.1, 0.7];
        remix_channels_f32(2, 1, &mut buf, path)?;
        assert_eq!(buf.samples_f32, [0.5]);
        Ok(())
    }

    #[test]
    fn rate_control_helpers_match_the_codec_apis() -> Result<(), PipelineError> {
        let mp3_default = mp3_config(AudioRateControl::Default);
        assert_eq!(mp3_default.bitrate_kbps, MP3_BITRATE_KBPS);
        assert_eq!(mp3_default.vbr_quality, None);
        let mp3_bitrate = mp3_config(AudioRateControl::Bitrate(br(191_500)));
        assert_eq!(mp3_bitrate.bitrate_kbps, 192);
        assert_eq!(mp3_bitrate.vbr_quality, None);
        let mp3_best = mp3_config(AudioRateControl::Quality(ql(100)));
        assert_eq!(mp3_best.bitrate_kbps, 0);
        assert_eq!(
            mp3_best.vbr_quality,
            Some(rusty_mp3::vbr_quality_index(0.0))
        );
        let mp3_small = mp3_config(AudioRateControl::Quality(ql(0)));
        assert_eq!(
            mp3_small.vbr_quality,
            Some(rusty_mp3::vbr_quality_index(9.0))
        );

        assert_eq!(aac_bitrate_bps(AudioRateControl::Default), 128_000);
        assert_eq!(
            aac_bitrate_bps(AudioRateControl::Bitrate(br(256_000))),
            256_000
        );

        assert_eq!(
            opus_settings(AudioRateControl::Default),
            (128_000, RateControl::ConstrainedVbr)
        );
        assert_eq!(
            opus_settings(AudioRateControl::Bitrate(br(6_000))),
            (6_000, RateControl::ConstrainedVbr)
        );
        assert_eq!(
            opus_settings(AudioRateControl::Quality(ql(100))),
            (256_000, RateControl::Vbr)
        );
        assert_eq!(
            opus_settings(AudioRateControl::Quality(ql(0))),
            (16_000, RateControl::Vbr)
        );

        assert!(matches!(
            vorbis_strategy(AudioRateControl::Default)?,
            vorbis_enc::BitrateManagementStrategy::QualityVbr { target_quality }
                if (target_quality - 0.4).abs() < 1e-6
        ));
        assert!(matches!(
            vorbis_strategy(AudioRateControl::Bitrate(br(128_000)))?,
            vorbis_enc::BitrateManagementStrategy::Abr { average_bitrate }
                if average_bitrate.get() == 128_000
        ));
        assert!(matches!(
            vorbis_strategy(AudioRateControl::Quality(ql(100)))?,
            vorbis_enc::BitrateManagementStrategy::QualityVbr { target_quality }
                if (target_quality - 1.0).abs() < 1e-6
        ));
        Ok(())
    }

    #[test]
    fn effective_sample_rate_and_worker_estimate_respect_options() {
        let explicit = options_with(
            AudioOutputFormat::Wav,
            Some(44_100),
            None,
            AudioRateControl::Default,
        );
        assert_eq!(
            effective_sample_rate(AudioOutputFormat::Wav, 48_000, &explicit),
            44_100
        );
        assert_eq!(
            effective_sample_rate(
                AudioOutputFormat::Opus,
                44_100,
                &AudioEncodingOptions::default()
            ),
            48_000
        );
        assert_eq!(
            effective_sample_rate(
                AudioOutputFormat::Mp3,
                44_100,
                &AudioEncodingOptions::default()
            ),
            44_100
        );

        let item = item_frames(
            PathBuf::from("in.wav"),
            PathBuf::from("out.wav"),
            AudioInputFormat::Wav,
            2,
            44_100,
            1_000,
        );
        let defaults = AudioEncodingOptions::default();
        let plain = estimate_pcm_bytes(&item, AudioOutputFormat::Wav, &defaults);
        assert_eq!(
            (plain.peak_bytes, plain.largest_buffer_bytes),
            (8_000, 8_000)
        );
        let upsampled = options_with(
            AudioOutputFormat::Wav,
            Some(88_200),
            Some(6),
            AudioRateControl::Default,
        );
        let grown = estimate_pcm_bytes(&item, AudioOutputFormat::Wav, &upsampled);
        assert_eq!(grown.largest_buffer_bytes, 2_000 * 6 * 4);
        // Decoded input and the growing resample output coexist with the upmix output.
        assert_eq!(grown.peak_bytes, 2_000 * 2 * 4 + 2_000 * 6 * 4);
        let huge = item_frames(
            PathBuf::from("in.wav"),
            PathBuf::from("out.wav"),
            AudioInputFormat::Wav,
            2,
            8_000,
            MAX_PCM_BYTES_PER_FILE / 8,
        );
        let huge_estimate = estimate_pcm_bytes(&huge, AudioOutputFormat::Wav, &upsampled);
        assert!(huge_estimate.largest_buffer_bytes > MAX_PCM_BYTES_PER_FILE);
        assert!(resolve_worker_count(&[huge], 4, AudioOutputFormat::Wav, &upsampled).is_err());
    }

    fn plan_with_options(
        items: Vec<AudioBatchItem>,
        mode: fastconv_domain::AudioExecutionMode,
        options: AudioEncodingOptions,
    ) -> AudioBatchPlan {
        AudioBatchPlan::with_options(items, mode, NonZeroUsize::MIN, options)
            .unwrap_or_else(|_| unreachable!())
    }

    fn options_with(
        format: AudioOutputFormat,
        rate: Option<u32>,
        channels: Option<u16>,
        control: AudioRateControl,
    ) -> AudioEncodingOptions {
        let rate = rate.map(|hz| AudioOptionSampleRate::new(hz).unwrap_or_else(|_| unreachable!()));
        let channels = channels
            .map(|count| AudioOptionChannels::new(count).unwrap_or_else(|_| unreachable!()));
        AudioEncodingOptions::new(format, control, rate, channels, None)
            .unwrap_or_else(|_| unreachable!())
    }

    fn wav_spec(path: &Path) -> Result<hound::WavSpec, PipelineError> {
        hound::WavReader::open(path)
            .map(|reader| reader.spec())
            .map_err(|err| PipelineError::DecodeFailed {
                path: path.to_path_buf(),
                reason: format!("failed to reopen WAV: {err}"),
            })
    }

    #[test]
    fn resamples_and_remixes_lossless_outputs() -> Result<(), PipelineError> {
        let dir = temp_dir("transform-lossless");
        let wav = dir.join("in.wav");
        let mono = dir.join("mono.wav");
        let stereo = dir.join("stereo.wav");
        let deep = dir.join("deep.wav");
        let deep_out = dir.join("deep-out.wav");
        write_wav(&wav, &tone(16_384, 2), 44_100, 2);

        let downmix = options_with(
            AudioOutputFormat::Wav,
            Some(24_000),
            Some(1),
            AudioRateControl::Default,
        );
        execute_audio_batch(&plan_with_options(
            vec![item(
                wav.clone(),
                mono.clone(),
                AudioInputFormat::Wav,
                2,
                44_100,
            )],
            fastconv_domain::AudioExecutionMode::CpuWav,
            downmix,
        ))?;
        let spec = wav_spec(&mono)?;
        assert_eq!((spec.sample_rate, spec.channels), (24_000, 1));
        assert_ne!(read_wav(&mono).len(), 0);

        let upmix = options_with(
            AudioOutputFormat::Wav,
            Some(96_000),
            Some(2),
            AudioRateControl::Default,
        );
        write_wav(&wav, &tone(8_192, 1), 48_000, 1);
        execute_audio_batch(&plan_with_options(
            vec![item(
                wav.clone(),
                stereo.clone(),
                AudioInputFormat::Wav,
                1,
                48_000,
            )],
            fastconv_domain::AudioExecutionMode::CpuWav,
            upmix,
        ))?;
        let spec = wav_spec(&stereo)?;
        assert_eq!((spec.sample_rate, spec.channels), (96_000, 2));
        let upmixed = read_wav(&stereo);
        assert_eq!(upmixed.len(), 16_384 * 2);
        assert!(
            upmixed
                .as_chunks::<2>()
                .0
                .iter()
                .all(|pair| pair[0] == pair[1])
        );

        write_wav24(&deep, &deep_tone(8_192, 2), 44_100, 2)?;
        let resample_24 = options_with(
            AudioOutputFormat::Wav,
            Some(24_000),
            None,
            AudioRateControl::Default,
        );
        execute_audio_batch(&plan_with_options(
            vec![item_bits(
                deep,
                deep_out.clone(),
                AudioInputFormat::Wav,
                2,
                44_100,
                24,
            )],
            fastconv_domain::AudioExecutionMode::CpuWav,
            resample_24,
        ))?;
        let spec = wav_spec(&deep_out)?;
        assert_eq!(
            (spec.sample_rate, spec.channels, spec.bits_per_sample),
            (24_000, 2, 24)
        );

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn wav_to_flac_resamples_instead_of_taking_the_streaming_fast_path() -> Result<(), PipelineError>
    {
        let dir = temp_dir("transform-flac");
        let wav = dir.join("in.wav");
        let flac = dir.join("out.flac");
        let back = dir.join("back.wav");
        write_wav(&wav, &tone(16_384, 2), 44_100, 2);

        let options = options_with(
            AudioOutputFormat::Flac,
            Some(24_000),
            Some(1),
            AudioRateControl::Default,
        );
        execute_audio_batch(&plan_with_options(
            vec![item_bits(
                wav,
                flac.clone(),
                AudioInputFormat::Wav,
                2,
                44_100,
                16,
            )],
            fastconv_domain::AudioExecutionMode::CpuFlac,
            options,
        ))?;
        execute_audio_batch(&plan(
            vec![item(flac, back.clone(), AudioInputFormat::Flac, 1, 24_000)],
            fastconv_domain::AudioExecutionMode::CpuWav,
        ))?;
        let spec = wav_spec(&back)?;
        assert_eq!((spec.sample_rate, spec.channels), (24_000, 1));
        assert_ne!(read_wav(&back).len(), 0);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn lossy_codecs_honor_bitrate_and_quality_options() -> Result<(), PipelineError> {
        let dir = temp_dir("lossy-options");
        let wav = dir.join("in.wav");
        write_wav(&wav, &tone(16_384, 2), 48_000, 2);

        let cases = [
            (
                "mp3-bitrate",
                "mp3",
                AudioOutputFormat::Mp3,
                fastconv_domain::AudioExecutionMode::CpuMp3,
                AudioInputFormat::Mp3,
                AudioRateControl::Bitrate(br(320_000)),
            ),
            (
                "mp3-quality",
                "mp3q",
                AudioOutputFormat::Mp3,
                fastconv_domain::AudioExecutionMode::CpuMp3,
                AudioInputFormat::Mp3,
                AudioRateControl::Quality(ql(80)),
            ),
            (
                "aac-bitrate",
                "aac",
                AudioOutputFormat::Aac,
                fastconv_domain::AudioExecutionMode::CpuAac,
                AudioInputFormat::Aac,
                AudioRateControl::Bitrate(br(256_000)),
            ),
            (
                "m4a-bitrate",
                "m4a",
                AudioOutputFormat::M4a,
                fastconv_domain::AudioExecutionMode::CpuM4a,
                AudioInputFormat::M4a,
                AudioRateControl::Bitrate(br(160_000)),
            ),
            (
                "ogg-bitrate",
                "ogg",
                AudioOutputFormat::Ogg,
                fastconv_domain::AudioExecutionMode::CpuOgg,
                AudioInputFormat::Vorbis,
                AudioRateControl::Bitrate(br(128_000)),
            ),
            (
                "ogg-quality",
                "oggq",
                AudioOutputFormat::Ogg,
                fastconv_domain::AudioExecutionMode::CpuOgg,
                AudioInputFormat::Vorbis,
                AudioRateControl::Quality(ql(80)),
            ),
            (
                "opus-bitrate",
                "opus",
                AudioOutputFormat::Opus,
                fastconv_domain::AudioExecutionMode::CpuOpus,
                AudioInputFormat::Opus,
                AudioRateControl::Bitrate(br(96_000)),
            ),
            (
                "opus-quality",
                "opusq",
                AudioOutputFormat::Opus,
                fastconv_domain::AudioExecutionMode::CpuOpus,
                AudioInputFormat::Opus,
                AudioRateControl::Quality(ql(90)),
            ),
        ];
        for (label, ext, format, mode, decoded_format, control) in cases {
            let encoded = dir.join(format!("{label}.{ext}"));
            let back = dir.join(format!("{label}.wav"));
            let options = options_with(format, None, None, control);
            execute_audio_batch(&plan_with_options(
                vec![item(
                    wav.clone(),
                    encoded.clone(),
                    AudioInputFormat::Wav,
                    2,
                    48_000,
                )],
                mode,
                options,
            ))?;
            execute_audio_batch(&plan(
                vec![item(encoded, back.clone(), decoded_format, 2, 48_000)],
                fastconv_domain::AudioExecutionMode::CpuWav,
            ))?;
            assert_ne!(read_wav(&back).len(), 0, "{label}: decoded output is empty");
            let spec = wav_spec(&back)?;
            assert_eq!((spec.sample_rate, spec.channels), (48_000, 2), "{label}");
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    fn mp3_first_frame_bitrate_kbps(path: &Path) -> Result<u32, PipelineError> {
        const MPEG1_KBPS: [u32; 16] = [
            0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
        ];
        const MPEG2_KBPS: [u32; 16] = [
            0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
        ];
        let bytes = fs::read(path).map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        for (index, pair) in bytes.windows(2).enumerate() {
            if pair[0] != 0xFF || pair[1] & 0xE0 != 0xE0 {
                continue;
            }
            let version = (pair[1] >> 3) & 0b11;
            if version == 1 {
                continue;
            }
            let Some(header) = bytes.get(index + 2) else {
                break;
            };
            let bitrate_index = usize::from((header >> 4) & 0x0F);
            let table = if version == 3 { MPEG1_KBPS } else { MPEG2_KBPS };
            return Ok(table[bitrate_index]);
        }
        Err(PipelineError::DecodeFailed {
            path: path.to_path_buf(),
            reason: "no MP3 frame sync found".to_owned(),
        })
    }

    #[test]
    fn mp3_bitrate_control_reaches_the_encoded_frame_header() -> Result<(), PipelineError> {
        let dir = temp_dir("mp3-rate-control");
        let wav = dir.join("in.wav");
        write_wav(&wav, &tone(16_384, 2), 48_000, 2);
        let high = dir.join("high.mp3");
        let default = dir.join("default.mp3");
        for (path, control) in [
            (&high, AudioRateControl::Bitrate(br(320_000))),
            (&default, AudioRateControl::Default),
        ] {
            let options = options_with(AudioOutputFormat::Mp3, None, None, control);
            execute_audio_batch(&plan_with_options(
                vec![item(
                    wav.clone(),
                    path.clone(),
                    AudioInputFormat::Wav,
                    2,
                    48_000,
                )],
                fastconv_domain::AudioExecutionMode::CpuMp3,
                options,
            ))?;
        }
        assert_eq!(mp3_first_frame_bitrate_kbps(&high)?, 320);
        assert_eq!(mp3_first_frame_bitrate_kbps(&default)?, MP3_BITRATE_KBPS);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn aac_rejects_bitrate_above_frame_capacity_without_explicit_options() {
        let dir = temp_dir("aac-frame-capacity");
        let aac = dir.join("out.aac");
        let mut buf = AudioWorkerBuffer::new();
        buf.samples_f32 = vec![0.0_f32; 2_048];
        let options = options_with(
            AudioOutputFormat::Aac,
            None,
            None,
            AudioRateControl::Bitrate(br(512_000)),
        );
        let result = encode_aac(&aac, 8_000, 1, &mut buf, &options);
        assert!(
            matches!(result, Err(PipelineError::EncodeFailed { .. })),
            "512 kbps at 8 kHz mono must be rejected instead of silently reduced"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mp3_rejects_bitrate_above_mpeg2_limit_without_an_explicit_rate() {
        let dir = temp_dir("mp3-low-rate");
        let wav = dir.join("in.wav");
        write_wav(&wav, &tone(4_096, 2), 24_000, 2);
        let mp3 = dir.join("out.mp3");
        let options = options_with(
            AudioOutputFormat::Mp3,
            None,
            None,
            AudioRateControl::Bitrate(br(320_000)),
        );
        let result = execute_audio_batch(&plan_with_options(
            vec![item(wav, mp3, AudioInputFormat::Wav, 2, 24_000)],
            fastconv_domain::AudioExecutionMode::CpuMp3,
            options,
        ));
        assert!(
            matches!(result, Err(PipelineError::EncodeFailed { .. })),
            "320 kbps at 24 kHz must be rejected instead of silently snapped to 160"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn remix_maps_speaker_positions_when_dropping_channels() -> Result<(), PipelineError> {
        let path = Path::new("remix.wav");
        let mut buf = AudioWorkerBuffer::new();
        buf.samples_f32 = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        remix_channels_f32(6, 5, &mut buf, path)?;
        assert_eq!(
            buf.samples_f32,
            vec![0.0; 5],
            "LFE must be dropped, not moved to BL"
        );

        buf.samples_f32 = vec![0.0, 0.0, 0.0, 1.0, 0.0];
        remix_channels_f32(5, 6, &mut buf, path)?;
        assert_eq!(
            buf.samples_f32,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "BL must stay BL when an LFE plane is inserted before it"
        );

        buf.samples_f32 = vec![0.0; 4];
        assert!(matches!(
            remix_channels_f32(4, 2, &mut buf, path),
            Err(PipelineError::DecodeFailed { .. })
        ));
        Ok(())
    }

    #[test]
    fn downsampling_a_short_input_rejects_empty_output() {
        let dir = temp_dir("empty-resample");
        let wav = dir.join("in.wav");
        write_wav(&wav, &tone(2, 2), 48_000, 2);
        let out = dir.join("out.wav");
        let options = options_with(
            AudioOutputFormat::Wav,
            Some(8_000),
            None,
            AudioRateControl::Default,
        );
        let result = execute_audio_batch(&plan_with_options(
            vec![item(wav, out, AudioInputFormat::Wav, 2, 48_000)],
            fastconv_domain::AudioExecutionMode::CpuWav,
            options,
        ));
        assert!(
            matches!(result, Err(PipelineError::EmptyOutput(_))),
            "a zero-frame resample must not publish a header-only WAV"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    fn write_wav_f32(path: &Path, samples: &[f32], rate: u32, channels: u16) {
        let spec = WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let Ok(mut writer) = WavWriter::create(path, spec) else {
            return;
        };
        for &sample in samples {
            let _ = writer.write_sample(sample);
        }
        let _ = writer.finalize();
    }

    fn write_wav_i32(path: &Path, samples: &[i32], rate: u32, channels: u16) {
        let spec = WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 32,
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

    fn read_wav_f32(path: &Path) -> Vec<f32> {
        let Ok(mut reader) = hound::WavReader::open(path) else {
            return Vec::new();
        };
        reader.samples::<f32>().filter_map(Result::ok).collect()
    }

    fn options_depth(format: AudioOutputFormat, depth: AudioBitDepth) -> AudioEncodingOptions {
        AudioEncodingOptions::new(format, AudioRateControl::Default, None, None, Some(depth))
            .unwrap_or_else(|_| unreachable!())
    }

    fn float_samples(frames: usize, channels: usize) -> Vec<f32> {
        (0..frames.saturating_mul(channels))
            .map(|index| {
                let step = index % 2_047;
                (f32::from(u16::try_from(step).unwrap_or(0)) / 2_046.0) * 1.8 - 0.9
            })
            .collect()
    }

    fn deep_samples(count: usize) -> Vec<i32> {
        (0..count)
            .map(|index| {
                let base = i32::try_from(index % 65_536)
                    .unwrap_or(0)
                    .saturating_mul(32_767);
                base ^ 0x5A5A_A5A5
            })
            .collect()
    }

    #[test]
    fn float32_wav_to_float32_wav_is_bit_exact() -> Result<(), PipelineError> {
        let dir = temp_dir("f32-wav-wav");
        let src = dir.join("in.wav");
        let out = dir.join("out.wav");
        let samples = float_samples(4_096, 2);
        write_wav_f32(&src, &samples, 48_000, 2);
        execute_audio_batch(&plan_with_options(
            vec![item_bits(
                src,
                out.clone(),
                AudioInputFormat::Wav,
                2,
                48_000,
                32,
            )],
            fastconv_domain::AudioExecutionMode::CpuWav,
            options_depth(AudioOutputFormat::Wav, AudioBitDepth::Float32),
        ))?;
        let spec = wav_spec(&out)?;
        assert_eq!(spec.sample_format, SampleFormat::Float);
        assert_eq!(spec.bits_per_sample, 32);
        assert_eq!(read_wav_f32(&out), samples);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn float32_wav_aifc_wav_roundtrip_is_bit_exact() -> Result<(), PipelineError> {
        let dir = temp_dir("f32-wav-aifc-wav");
        let src = dir.join("in.wav");
        let aifc = dir.join("middle.aiff");
        let out = dir.join("out.wav");
        let samples = float_samples(4_096, 2);
        write_wav_f32(&src, &samples, 48_000, 2);
        execute_audio_batch(&plan_with_options(
            vec![item_bits(
                src,
                aifc.clone(),
                AudioInputFormat::Wav,
                2,
                48_000,
                32,
            )],
            fastconv_domain::AudioExecutionMode::CpuAiff,
            options_depth(AudioOutputFormat::Aiff, AudioBitDepth::Float32),
        ))?;
        let bytes = fs::read(&aifc).map_err(|source| PipelineError::Io {
            path: aifc.clone(),
            source,
        })?;
        assert_eq!(&bytes[..4], b"FORM");
        assert_eq!(&bytes[8..12], b"AIFC");
        assert!(bytes.windows(4).any(|window| window == b"fl32"));
        execute_audio_batch(&plan_with_options(
            vec![item_bits(
                aifc,
                out.clone(),
                AudioInputFormat::Aiff,
                2,
                48_000,
                32,
            )],
            fastconv_domain::AudioExecutionMode::CpuWav,
            options_depth(AudioOutputFormat::Wav, AudioBitDepth::Float32),
        ))?;
        assert_eq!(wav_spec(&out)?.sample_format, SampleFormat::Float);
        assert_eq!(read_wav_f32(&out), samples);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn float32_wav_converts_to_24bit_wav_flac_mp3_and_opus() -> Result<(), PipelineError> {
        let dir = temp_dir("f32-targets");
        let src = dir.join("in.wav");
        let samples = float_samples(8_192, 2);
        write_wav_f32(&src, &samples, 48_000, 2);

        let wav24 = dir.join("out24.wav");
        execute_audio_batch(&plan_with_options(
            vec![item_bits(
                src.clone(),
                wav24.clone(),
                AudioInputFormat::Wav,
                2,
                48_000,
                32,
            )],
            fastconv_domain::AudioExecutionMode::CpuWav,
            options_depth(AudioOutputFormat::Wav, AudioBitDepth::Int24),
        ))?;
        assert_eq!(wav_spec(&wav24)?.bits_per_sample, 24);
        let quantized = read_wav_i32(&wav24);
        assert_eq!(quantized.len(), samples.len());
        for (index, (&got, &want)) in quantized.iter().zip(&samples).enumerate() {
            let expected = quantize_i24(want);
            assert_eq!(got, expected, "24-bit sample {index}");
        }

        let flac = dir.join("out.flac");
        execute_audio_batch(&plan(
            vec![item_bits(
                src.clone(),
                flac.clone(),
                AudioInputFormat::Wav,
                2,
                48_000,
                32,
            )],
            fastconv_domain::AudioExecutionMode::CpuFlac,
        ))?;
        let back = dir.join("from-flac.wav");
        execute_audio_batch(&plan(
            vec![item_bits(
                flac,
                back.clone(),
                AudioInputFormat::Flac,
                2,
                48_000,
                24,
            )],
            fastconv_domain::AudioExecutionMode::CpuWav,
        ))?;
        assert_eq!(wav_spec(&back)?.bits_per_sample, 24);
        assert_ne!(read_wav_i32(&back), [] as [i32; 0]);

        for (label, ext, mode, decoded) in [
            (
                "mp3",
                "mp3",
                fastconv_domain::AudioExecutionMode::CpuMp3,
                AudioInputFormat::Mp3,
            ),
            (
                "opus",
                "opus",
                fastconv_domain::AudioExecutionMode::CpuOpus,
                AudioInputFormat::Opus,
            ),
        ] {
            let encoded = dir.join(format!("lossy-{label}.{ext}"));
            let decoded_path = dir.join(format!("lossy-{label}.wav"));
            execute_audio_batch(&plan(
                vec![item_bits(
                    src.clone(),
                    encoded.clone(),
                    AudioInputFormat::Wav,
                    2,
                    48_000,
                    32,
                )],
                mode,
            ))?;
            execute_audio_batch(&plan(
                vec![item_bits(
                    encoded,
                    decoded_path.clone(),
                    decoded,
                    2,
                    48_000,
                    16,
                )],
                fastconv_domain::AudioExecutionMode::CpuWav,
            ))?;
            assert!(
                !read_wav(&decoded_path).is_empty(),
                "{label} decoded output"
            );
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn int32_wav_aiff_wav_roundtrip_preserves_low_bits() -> Result<(), PipelineError> {
        let dir = temp_dir("i32-wav-aiff-wav");
        let src = dir.join("in.wav");
        let aiff = dir.join("middle.aiff");
        let out = dir.join("out.wav");
        let samples = deep_samples(4_096);
        assert!(samples.iter().any(|sample| sample & 0xFF != 0));
        write_wav_i32(&src, &samples, 48_000, 2);
        execute_audio_batch(&plan_with_options(
            vec![item_bits(
                src,
                aiff.clone(),
                AudioInputFormat::Wav,
                2,
                48_000,
                32,
            )],
            fastconv_domain::AudioExecutionMode::CpuAiff,
            options_depth(AudioOutputFormat::Aiff, AudioBitDepth::Int32),
        ))?;
        execute_audio_batch(&plan_with_options(
            vec![item_bits(
                aiff,
                out.clone(),
                AudioInputFormat::Aiff,
                2,
                48_000,
                32,
            )],
            fastconv_domain::AudioExecutionMode::CpuWav,
            options_depth(AudioOutputFormat::Wav, AudioBitDepth::Int32),
        ))?;
        assert_eq!(wav_spec(&out)?.bits_per_sample, 32);
        assert_eq!(read_wav_i32(&out), samples);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    fn assert_explicit_bit_depth(
        dir: &Path,
        src: &Path,
        label: &str,
        format: AudioOutputFormat,
        depth: AudioBitDepth,
        mode: fastconv_domain::AudioExecutionMode,
    ) -> Result<(), PipelineError> {
        let out = dir.join(format!("{label}.{}", format.extension()));
        execute_audio_batch(&plan_with_options(
            vec![item_bits(
                src.to_path_buf(),
                out.clone(),
                AudioInputFormat::Wav,
                2,
                48_000,
                16,
            )],
            mode,
            options_depth(format, depth),
        ))?;
        if format == AudioOutputFormat::Flac {
            let decoded = dir.join("flac24.wav");
            execute_audio_batch(&plan(
                vec![item_bits(
                    out,
                    decoded.clone(),
                    AudioInputFormat::Flac,
                    2,
                    48_000,
                    24,
                )],
                fastconv_domain::AudioExecutionMode::CpuWav,
            ))?;
            assert_eq!(wav_spec(&decoded)?.bits_per_sample, 24);
        } else if format == AudioOutputFormat::Wav {
            let spec = wav_spec(&out)?;
            assert_eq!(spec.bits_per_sample, depth.bits(), "{label}");
            assert_eq!(spec.sample_format == SampleFormat::Float, depth.is_float());
        } else {
            let bytes = fs::read(&out).map_err(|source| PipelineError::Io {
                path: out.clone(),
                source,
            })?;
            let expected_form = if depth.is_float() { b"AIFC" } else { b"AIFF" };
            assert_eq!(&bytes[8..12], expected_form, "{label}");
            let comm_bits_at = if depth.is_float() { 38 } else { 26 };
            let Some(bits) = bytes.get(comm_bits_at..comm_bits_at + 2) else {
                return Err(PipelineError::DecodeFailed {
                    path: out,
                    reason: "AIFF COMM header is truncated".to_owned(),
                });
            };
            assert_eq!(bits, depth.bits().to_be_bytes(), "{label}");
        }
        Ok(())
    }

    #[test]
    fn sixteen_bit_wav_honors_explicit_integer_depths() -> Result<(), PipelineError> {
        let dir = temp_dir("i16-explicit-int");
        let src = dir.join("in.wav");
        write_wav(&src, &tone(4_096, 2), 48_000, 2);
        let wav = fastconv_domain::AudioExecutionMode::CpuWav;
        for (label, format, depth, mode) in [
            ("wav24", AudioOutputFormat::Wav, AudioBitDepth::Int24, wav),
            (
                "flac24",
                AudioOutputFormat::Flac,
                AudioBitDepth::Int24,
                fastconv_domain::AudioExecutionMode::CpuFlac,
            ),
            (
                "aiff24",
                AudioOutputFormat::Aiff,
                AudioBitDepth::Int24,
                fastconv_domain::AudioExecutionMode::CpuAiff,
            ),
            ("wav32", AudioOutputFormat::Wav, AudioBitDepth::Int32, wav),
            (
                "aiff32",
                AudioOutputFormat::Aiff,
                AudioBitDepth::Int32,
                fastconv_domain::AudioExecutionMode::CpuAiff,
            ),
        ] {
            assert_explicit_bit_depth(&dir, &src, label, format, depth, mode)?;
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn sixteen_bit_wav_honors_explicit_float_depths() -> Result<(), PipelineError> {
        let dir = temp_dir("i16-explicit-f32");
        let src = dir.join("in.wav");
        write_wav(&src, &tone(4_096, 2), 48_000, 2);
        assert_explicit_bit_depth(
            &dir,
            &src,
            "wavf32",
            AudioOutputFormat::Wav,
            AudioBitDepth::Float32,
            fastconv_domain::AudioExecutionMode::CpuWav,
        )?;
        assert_explicit_bit_depth(
            &dir,
            &src,
            "aifff32",
            AudioOutputFormat::Aiff,
            AudioBitDepth::Float32,
            fastconv_domain::AudioExecutionMode::CpuAiff,
        )?;
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }
}
