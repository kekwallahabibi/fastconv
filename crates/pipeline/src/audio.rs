//! Native audio backends. FLAC uses `flacenc` and WAV uses `hound`; the lossy encoder is
//! `opus-pure`, a dependency-free pure-Rust RFC 6716/Ogg Opus implementation. C-backed crates
//! such as `libopus`/`vorbis_rs` are deliberately avoided so CI needs no system audio libraries.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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
    opus_block: Vec<i16>,
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
            opus_block: vec![
                0_i16;
                usize::try_from(OPUS_RATE / (1000 / OPUS_FRAME_MS)).unwrap_or(960) * 2
            ],
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
    // Symphonia left-aligns 16-bit integer samples when converting to f32 (i16 << 16),
    // so a direct i16 copy is both faster and bit-exact for lossless targets.
    let integer_pcm = !matches!(
        params.sample_format,
        Some(symphonia::core::audio::sample::SampleFormat::F32)
            | Some(symphonia::core::audio::sample::SampleFormat::F64)
    );
    let direct_i16 = matches!(target, AudioOutputFormat::Wav | AudioOutputFormat::Flac)
        && bits == 16
        && integer_pcm;
    let reserve = total_samples
        .map_or(0, |frames| usize::try_from(frames).unwrap_or(usize::MAX))
        .saturating_mul(channels);
    if direct_i16 {
        if buf.samples_i16.capacity() < reserve {
            buf.samples_i16.reserve(reserve.saturating_sub(buf.samples_i16.capacity()));
        }
    } else if buf.samples_f32.capacity() < reserve {
        buf.samples_f32.reserve(reserve.saturating_sub(buf.samples_f32.capacity()));
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
        if direct_i16 {
            let start = buf.samples_i16.len();
            buf.samples_i16.resize(start.saturating_add(count), 0);
            if let Some(slot) = buf.samples_i16.get_mut(start..start.saturating_add(count)) {
                audio_frame.copy_to_slice_interleaved::<i16, _>(slot);
            }
        } else {
            let start = buf.samples_f32.len();
            buf.samples_f32.resize(start.saturating_add(count), 0.0);
            if let Some(slot) = buf.samples_f32.get_mut(start..start.saturating_add(count)) {
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
    let (rate, channels, source_bits) = if item.input_format() == AudioInputFormat::Opus {
        let (rate, channels) = decode_opus(item.source(), buf)?;
        (rate, channels, 16)
    } else {
        decode_symphonia(item.source(), item.input_format(), target, buf)?
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
        (AudioOutputFormat::Opus, _) | (AudioOutputFormat::Wav, 16 | 0) => {
            buf.samples_i16
                .extend(buf.samples_f32.iter().copied().map(quantize_i16));
        }
        (AudioOutputFormat::Wav, 8) => {
            buf.samples_i16.extend(
                buf.samples_f32
                    .iter()
                    .copied()
                    .map(|s| i16::from(quantize_i8(s))),
            );
        }
        (AudioOutputFormat::Wav | AudioOutputFormat::Flac, 24) => {
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
        // Unsupported depths are rejected before quantization; keep the match exhaustive.
        _ => {}
    }
    // Only one PCM representation is needed after quantization; release the float copy so peak
    // memory stays bounded to a single full-file sample array per active worker.
    buf.samples_f32.clear();
}


const WAVE_FORMAT_PCM_GUID: [u8; 16] = [
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B,
    0x71,
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

    let mut pos = 12_u64;
    let mut fmt: Option<(usize, u32, usize, usize)> = None;
    let mut data: Option<(u64, u64)> = None;
    let mut header = [0_u8; 8];
    let mut payload = vec![0_u8; 4096];

    while pos.saturating_add(8) <= file_len {
        if file.seek(SeekFrom::Start(pos)).is_err() || file.read_exact(&mut header).is_err() {
            return Ok(None);
        }
        let size = u64::from(le_u32(&header, 4));
        if size == u64::from(u32::MAX) {
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
        || data_offset.saturating_add(data_bytes) > file_len
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
        let available = usize::try_from(self.remaining_bytes).unwrap_or(usize::MAX)
            / self.frame_bytes.max(1);
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
                Ok(0) => break,
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

fn encode_flac_from_source<S: Source>(
    path: &Path,
    source: S,
    batch_workers: usize,
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
    let mut sink = ByteSink::with_capacity(pcm_bytes.saturating_mul(8).max(4096));
    stream
        .write(&mut sink)
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("FLAC bitstream write failed: {err}"),
        })?;
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
    encode_flac_from_source(path, source, batch_workers)
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

fn resample_to_opus_rate(
    rate: u32,
    channels: usize,
    buf: &mut AudioWorkerBuffer,
) -> Result<(), PipelineError> {
    if rate == OPUS_RATE {
        // Opus consumes f32 interleaved at 48 kHz; keep the decoded buffer in place.
        return Ok(());
    }
    let input = std::mem::take(&mut buf.samples_f32);
    if channels > 2 {
        return Err(PipelineError::EncodeFailed {
            path: Path::new("opus-resampler").to_path_buf(),
            reason: format!("Opus supports at most 2 channels, got {channels}"),
        });
    }

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
        usize_from(rate),
        usize_from(OPUS_RATE),
        RESAMPLE_CHUNK,
        channels,
    )
    .map_err(|err| PipelineError::EncodeFailed {
        path: Path::new("opus-resampler").to_path_buf(),
        reason: format!("failed to create resampler from {rate} Hz to 48000 Hz: {err}"),
    })?;
    let chunk_in = resampler.input_frames_next();
    let chunk_out = resampler.output_frames_next().max(1);
    let mut out_chunk = resampler.output_buffer_allocate(true);
    let input_frames = input.len() / channels;
    let expected_u64 = (u64::try_from(input_frames).unwrap_or(u64::MAX))
        .saturating_mul(u64::from(OPUS_RATE))
        / u64::from(rate);
    let expected = usize::try_from(expected_u64).unwrap_or(usize::MAX);
    let delay = resampler.output_delay();
    let needed = expected.saturating_add(delay);
    let max_calls = input_frames
        .div_ceil(chunk_in)
        .saturating_add(needed.div_ceil(chunk_out))
        .saturating_add(4);
    let reserve = needed.saturating_add(4).saturating_mul(channels);
    for plane in &mut planar_out {
        plane.reserve(reserve);
    }
    if buf.resampled_f32.capacity() < reserve {
        buf.resampled_f32.reserve(reserve.saturating_sub(buf.resampled_f32.capacity()));
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
                dst.fill(0.0);
                if let Some(src) = planar_in.get(channel) {
                    for frame in 0..take {
                        if let (Some(value), Some(slot)) = (src.get(offset + frame), dst.get_mut(frame)) {
                            *slot = *value;
                        }
                    }
                }
            }
        }
        let (_, produced) = resampler
            .process_into_buffer(&chunk, &mut out_chunk, None)
            .map_err(|err| PipelineError::EncodeFailed {
                path: Path::new("opus-resampler").to_path_buf(),
                reason: format!("resampling failed: {err}"),
            })?;
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
    resample_to_opus_rate(rate, channels, buf)?;
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
    let mut writer = OggOpusWriter::new(
        BufWriter::with_capacity(OPUS_WRITE_BUF_BYTES, file),
        head,
    )
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
            writer.write_packet(packet).map_err(|err| PipelineError::EncodeFailed {
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
            let end = start.saturating_add(per_frame).min(buf.samples_f32.len());
            buf.opus_block.fill(0_i16);
            // Serial fallback uses the existing i16 encoder path on quantized samples.
            let mut block = vec![0_f32; per_frame];
            if let Some(slot) = block.get_mut(..end.saturating_sub(start)) {
                slot.copy_from_slice(&buf.samples_f32[start..end]);
            }
            let written = encoder
                .encode(&block, frame, &mut buf.packet)
                .map_err(|err| PipelineError::EncodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("Opus encoding failed: {err}"),
                })?;
            write_packet(&mut writer, index, &buf.packet[..written])?;
        }
    }

    let mut buf_writer = writer
        .finish()
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("failed to finalize Ogg Opus stream: {err}"),
        })?;
    buf_writer.flush().map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })
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
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging)
        {
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
    let bits = if mode.output_format() == AudioOutputFormat::Opus {
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
    // or pass the i16 buffer through to the bulk WAV writer.
    let direct_i16 = !buf.samples_i16.is_empty() && buf.samples_f32.is_empty();
    if direct_i16 {
        if mode.output_format() == AudioOutputFormat::Flac {
            buf.samples_i32.clear();
            buf.samples_i32
                .extend(buf.samples_i16.iter().copied().map(i32::from));
        }
    } else if mode.output_format() != AudioOutputFormat::Opus {
        quantize_for(mode.output_format(), bits, buf);
    }
    encode_staged(item, |temp| match mode.output_format() {
        AudioOutputFormat::Wav => encode_wav(temp, rate, channels, bits, buf),
        AudioOutputFormat::Flac => encode_flac(temp, rate, channels, bits, buf, batch_workers),
        AudioOutputFormat::Opus => encode_opus(temp, rate, channels, buf, batch_workers),
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
        encode_flac_from_source(temp, source, batch_workers)
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
        bytes.extend_from_slice(&u32::try_from(chunks.len().saturating_add(4)).unwrap_or(0).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(&chunks);
        bytes
    }

    fn write_raw_wav(path: &Path, bytes: &[u8]) {
        if let Err(err) = fs::write(path, bytes) {
            assert!(false, "failed to write test WAV: {err}");
        }
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
        write_raw_wav(&wav, &wav_bytes(&fmt_pcm(2, 48_000, 16), &data, &[(*b"JUNK", junk)]));
        assert!(try_open_pcm_wav(&wav)?.is_some(), "16-bit PCM must use the direct source");

        execute_audio_batch(&plan(
            vec![item_bits(wav, flac.clone(), AudioInputFormat::Wav, 2, 48_000, 16)],
            fastconv_domain::AudioExecutionMode::CpuFlac,
        ))?;
        execute_audio_batch(&plan(
            vec![item_bits(flac, back.clone(), AudioInputFormat::Flac, 2, 48_000, 16)],
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
            .map(|i| {
                let magnitude = (i.saturating_mul(7919) % 4_000_000) - 2_000_000;
                magnitude
            })
            .collect();
        let mut data = Vec::with_capacity(samples.len() * 3);
        for sample in &samples {
            let bytes = sample.to_le_bytes();
            data.extend_from_slice(&bytes[..3]);
        }
        write_raw_wav(&wav, &wav_bytes(&fmt_pcm(1, 44_100, 24), &data, &[]));
        assert!(try_open_pcm_wav(&wav)?.is_some(), "24-bit PCM must use the direct source");

        execute_audio_batch(&plan(
            vec![item_bits(wav, flac.clone(), AudioInputFormat::Wav, 1, 44_100, 24)],
            fastconv_domain::AudioExecutionMode::CpuFlac,
        ))?;
        execute_audio_batch(&plan(
            vec![item_bits(flac, back.clone(), AudioInputFormat::Flac, 1, 44_100, 24)],
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
        write_raw_wav(&wav, &wav_bytes(&fmt_pcm(1, 8_000, 8), &unsigned, &[]));
        assert!(try_open_pcm_wav(&wav)?.is_none(), "8-bit WAV must fall back to Symphonia");

        execute_audio_batch(&plan(
            vec![item_bits(wav, flac.clone(), AudioInputFormat::Wav, 1, 8_000, 8)],
            fastconv_domain::AudioExecutionMode::CpuFlac,
        ))?;
        execute_audio_batch(&plan(
            vec![item_bits(flac, back.clone(), AudioInputFormat::Flac, 1, 8_000, 8)],
            fastconv_domain::AudioExecutionMode::CpuWav,
        ))?;
        let Ok(mut reader) = hound::WavReader::open(&back) else {
            assert!(false, "failed to reopen 8-bit WAV");
            return Ok(());
        };
        let decoded: Vec<u8> = reader
            .samples::<i8>()
            .filter_map(Result::ok)
            .map(|sample| sample.wrapping_add(i8::MIN) as u8)
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
        write_raw_wav(&wav, &wav_bytes(&fmt, &[0_u8; 32], &[]));
        assert!(try_open_pcm_wav(&wav)?.is_none(), "float WAV must fall back to Symphonia");
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

}
