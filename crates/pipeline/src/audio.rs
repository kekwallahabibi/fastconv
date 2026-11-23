//! Native audio backends. FLAC uses `flacenc` and WAV uses `hound`; the lossy encoder is
//! `opus-pure`, a dependency-free pure-Rust RFC 6716/Ogg Opus implementation. C-backed crates
//! such as `libopus`/`vorbis_rs` are deliberately avoided so CI needs no system audio libraries.

use std::fs::{self, File};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

use fastconv_domain::{
    AudioBatchItem, AudioBatchPlan, AudioConversionReport, AudioInputFormat, AudioOutputFormat,
    HardwareBackend,
};
use flacenc::bitsink::ByteSink;
use flacenc::component::BitRepr;
use flacenc::error::Verify;
use hound::{SampleFormat, WavSpec, WavWriter};
use opus_pure::{
    Application, MAX_PACKET_BYTES, MAX_PACKET_SAMPLES, OggOpusReader, OggOpusWriter, OpusEncoder,
    OpusHead, Trim,
};
use rubato::{FftFixedInOut, Resampler};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
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
    buf: &mut AudioWorkerBuffer,
) -> Result<(), PipelineError> {
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
    let track_id = track.id;
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
            Err(SymphoniaError::DecodeError(_)) => continue,
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
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => {
                return Err(PipelineError::DecodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("failed to decode audio packet: {err}"),
                });
            }
        };
        let start = buf.samples_f32.len();
        let count = audio_frame.samples_interleaved();
        buf.samples_f32.resize(start.saturating_add(count), 0.0);
        if let Some(slot) = buf.samples_f32.get_mut(start..start.saturating_add(count)) {
            audio_frame.copy_to_slice_interleaved(slot);
        }
    }
    Ok(())
}

fn decode_audio(
    item: &AudioBatchItem,
    buf: &mut AudioWorkerBuffer,
) -> Result<(u32, usize, usize), PipelineError> {
    buf.reset();
    let (rate, channels) = if item.input_format() == AudioInputFormat::Opus {
        decode_opus(item.source(), buf)?
    } else {
        decode_symphonia(item.source(), item.input_format(), buf)?;
        let stream = item.stream();
        (
            stream.sample_rate().get(),
            usize::from(stream.channels().get()),
        )
    };

    if channels == 0 {
        return Err(PipelineError::DecodeFailed {
            path: item.source().to_path_buf(),
            reason: "decoded audio has zero channels".to_owned(),
        });
    }
    if buf.samples_f32.is_empty() {
        return Err(PipelineError::EmptyOutput(item.source().to_path_buf()));
    }

    let source_bits = item
        .stream()
        .bits_per_sample()
        .map_or(0, |bits| usize::from(bits.get()));
    Ok((rate, channels, source_bits))
}

fn quantize_for(target: AudioOutputFormat, bits: usize, buf: &mut AudioWorkerBuffer) {
    buf.samples_i16.clear();
    buf.samples_i32.clear();
    match target {
        AudioOutputFormat::Opus => {
            buf.samples_i16
                .extend(buf.samples_f32.iter().copied().map(quantize_i16));
        }
        AudioOutputFormat::Wav | AudioOutputFormat::Flac if bits == 24 => {
            buf.samples_i32
                .extend(buf.samples_f32.iter().copied().map(quantize_i24));
        }
        AudioOutputFormat::Wav | AudioOutputFormat::Flac => {
            buf.samples_i16
                .extend(buf.samples_f32.iter().copied().map(quantize_i16));
            buf.samples_i32
                .extend(buf.samples_i16.iter().copied().map(i32::from));
        }
    }
}

fn encode_flac(
    path: &Path,
    rate: u32,
    channels: usize,
    bits: usize,
    buf: &AudioWorkerBuffer,
) -> Result<(), PipelineError> {
    let mut config = flacenc::config::Encoder::default();
    config.block_size = FLAC_BLOCK_SIZE;
    // Prefer encode speed over compression ratio: fixed LPC (order <= 4) is cheap, while the
    // quantized LPC search dominates runtime and only improves compression slightly.
    config.subframe_coding.use_lpc = false;
    config.multithread = true;
    config.workers = NonZeroUsize::new(
        std::thread::available_parallelism()
            .map_or(1, NonZeroUsize::get)
            .min(8),
    );
    let config = config
        .into_verified()
        .map_err(|(_, err)| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("FLAC config invalid: {err}"),
        })?;
    let source = flacenc::source::MemSource::from_samples(
        &buf.samples_i32,
        channels,
        bits,
        usize_from(rate),
    );
    let mut stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("FLAC encoding failed: {err}"),
        })?;
    // `add_frame` shrinks STREAMINFO's minimum to the short final frame. Symphonia treats such a
    // stream as variable-blocksize and then rejects the fixed-block frame headers, so advertise
    // the fixed block size we actually used.
    stream
        .stream_info_mut()
        .set_block_sizes(config.block_size, config.block_size)
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("FLAC stream-info update failed: {err}"),
        })?;
    let mut sink = ByteSink::new();
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
    let mut writer = WavWriter::create(path, spec).map_err(|err| PipelineError::EncodeFailed {
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
    } else {
        for &sample in &buf.samples_i16 {
            writer
                .write_sample(sample)
                .map_err(|err| PipelineError::EncodeFailed {
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
    let input = std::mem::take(&mut buf.samples_f32);
    if rate == OPUS_RATE {
        buf.resampled_f32.clear();
        buf.resampled_f32.extend_from_slice(&input);
        return Ok(());
    }
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
    let mut out_chunk = resampler.output_buffer_allocate(true);
    let input_frames = input.len() / channels;
    let mut offset = 0_usize;
    while offset < input_frames {
        let take = chunk_in.min(input_frames.saturating_sub(offset));
        for slot in &mut out_chunk {
            slot.fill(0.0);
        }
        let mut chunk = vec![vec![0_f32; chunk_in]; channels];
        for channel in 0..channels {
            if let (Some(src), Some(dst)) = (planar_in.get(channel), chunk.get_mut(channel)) {
                for frame in 0..take {
                    if let (Some(value), Some(slot)) = (src.get(offset + frame), dst.get_mut(frame))
                    {
                        *slot = *value;
                    }
                }
            }
        }
        let (consumed, produced) = resampler
            .process_into_buffer(&chunk, &mut out_chunk, None)
            .map_err(|err| PipelineError::EncodeFailed {
                path: Path::new("opus-resampler").to_path_buf(),
                reason: format!("resampling failed: {err}"),
            })?;
        if consumed == 0 {
            break;
        }
        for channel in 0..channels {
            if let (Some(dst), Some(src)) = (planar_out.get_mut(channel), out_chunk.get(channel)) {
                dst.extend_from_slice(&src[..produced.min(src.len())]);
            }
        }
        offset = offset.saturating_add(consumed);
    }

    let expected_u64 = (u64::try_from(input_frames).unwrap_or(u64::MAX))
        .saturating_mul(u64::from(OPUS_RATE))
        / u64::from(rate);
    let expected = usize::try_from(expected_u64).unwrap_or(usize::MAX);
    buf.resampled_f32.clear();
    let total_frames = planar_out.first().map_or(0, Vec::len).min(expected);
    for frame in 0..total_frames {
        for channel in 0..channels {
            if let Some(value) = planar_out.get(channel).and_then(|v| v.get(frame)) {
                buf.resampled_f32.push(*value);
            }
        }
    }
    Ok(())
}

fn encode_opus(
    path: &Path,
    rate: u32,
    channels: usize,
    buf: &mut AudioWorkerBuffer,
) -> Result<(), PipelineError> {
    if channels > 2 {
        return Err(PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("Opus output supports mono/stereo only, got {channels} channels"),
        });
    }
    resample_to_opus_rate(rate, channels, buf)?;
    let total_frames = buf.resampled_f32.len() / channels;
    std::mem::swap(&mut buf.samples_f32, &mut buf.resampled_f32);
    buf.resampled_f32.clear();
    quantize_for(AudioOutputFormat::Opus, 16, buf);
    let pcm_i16 = std::mem::take(&mut buf.samples_i16);
    buf.samples_i16.clear();

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
    let head = OpusHead::for_encoder(&encoder, OPUS_RATE);
    let pre_skip = usize::from(head.pre_skip);
    let file = File::create(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = OggOpusWriter::new(file, head).map_err(|err| PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason: format!("failed to create Ogg Opus writer: {err}"),
    })?;

    let frame = usize_from(OPUS_RATE / (1000 / OPUS_FRAME_MS));
    let per_frame = frame.saturating_mul(channels);
    let packet_count = total_frames.saturating_add(pre_skip).div_ceil(frame);
    let final_granule = u64::from(u16::try_from(pre_skip).unwrap_or(u16::MAX))
        + u64::try_from(total_frames).unwrap_or(u64::MAX);

    for index in 0..packet_count {
        let start = index.saturating_mul(per_frame);
        let end = start.saturating_add(per_frame).min(pcm_i16.len());
        buf.opus_block.fill(0);
        if end > start
            && let Some(slot) = buf.opus_block.get_mut(..end - start)
        {
            slot.copy_from_slice(&pcm_i16[start..end]);
        }
        let written = encoder
            .encode_s16(&buf.opus_block, frame, &mut buf.packet)
            .map_err(|err| PipelineError::EncodeFailed {
                path: path.to_path_buf(),
                reason: format!("Opus encoding failed: {err}"),
            })?;
        if index + 1 == packet_count {
            let duration =
                final_granule.saturating_sub(u64::try_from(writer.granule()).unwrap_or(0));
            let duration = u32::try_from(duration).unwrap_or(u32::MAX);
            writer
                .write_packet_with_duration(&buf.packet[..written], duration)
                .map_err(|err| PipelineError::EncodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("failed to write final Opus packet: {err}"),
                })?;
        } else {
            writer.write_packet(&buf.packet[..written]).map_err(|err| {
                PipelineError::EncodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("failed to write Opus packet: {err}"),
                }
            })?;
        }
    }
    writer
        .finish()
        .map(|_| ())
        .map_err(|err| PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("failed to finalize Ogg Opus stream: {err}"),
        })
}

fn encode_audio(
    item: &AudioBatchItem,
    mode: fastconv_domain::AudioExecutionMode,
    rate: u32,
    channels: usize,
    bits: usize,
    buf: &mut AudioWorkerBuffer,
    worker_id: usize,
) -> Result<u64, PipelineError> {
    let target = item.target();
    if let Some(parent) = target.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let temp = target.with_extension(format!(
        "fastconv-{worker_id}-{}",
        mode.output_format().extension()
    ));
    let bits = if bits == 8 || bits == 16 || bits == 24 {
        bits
    } else {
        16
    };
    quantize_for(mode.output_format(), bits, buf);

    let result = match mode.output_format() {
        AudioOutputFormat::Wav => encode_wav(&temp, rate, channels, bits, buf),
        AudioOutputFormat::Flac => encode_flac(&temp, rate, channels, bits, buf),
        AudioOutputFormat::Opus => encode_opus(&temp, rate, channels, buf),
    };
    let output_bytes = fs::metadata(&temp).map_or(0, |meta| meta.len());
    if let Err(err) = result {
        let _ = fs::remove_file(&temp);
        return Err(err);
    }
    if output_bytes == 0 {
        let _ = fs::remove_file(&temp);
        return Err(PipelineError::EmptyOutput(target.to_path_buf()));
    }
    fs::rename(&temp, target).map_err(|source| PipelineError::Io {
        path: target.to_path_buf(),
        source,
    })?;
    Ok(output_bytes)
}

fn convert_item(
    item: &AudioBatchItem,
    mode: fastconv_domain::AudioExecutionMode,
    buf: &mut AudioWorkerBuffer,
    worker_id: usize,
) -> Result<(u64, u64), PipelineError> {
    let input_bytes = fs::metadata(item.source()).map_or(0, |meta| meta.len());
    let (rate, channels, bits) = decode_audio(item, buf)?;
    let output_bytes = encode_audio(item, mode, rate, channels, bits, buf, worker_id)?;
    Ok((input_bytes, output_bytes))
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
    let first_error: Mutex<Option<PipelineError>> = Mutex::new(None);
    let worker_count = plan.effective_workers().get();

    thread::scope(|scope| {
        for worker_id in 0..worker_count {
            let next = &next;
            let first_error = &first_error;
            scope.spawn(move || {
                let mut buffer = AudioWorkerBuffer::new();
                loop {
                    if first_error.lock().is_ok_and(|guard| guard.is_some()) {
                        break;
                    }
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(index) else { break };
                    if let Err(err) = convert_item(item, plan.mode(), &mut buffer, worker_id) {
                        if let Ok(mut guard) = first_error.lock()
                            && guard.is_none()
                        {
                            *guard = Some(err);
                        }
                        break;
                    }
                }
            });
        }
    });

    if let Ok(mut guard) = first_error.lock()
        && let Some(err) = guard.take()
    {
        return Err(err);
    }

    let mut input_bytes = 0_u64;
    let mut output_bytes = 0_u64;
    for item in items {
        input_bytes =
            input_bytes.saturating_add(fs::metadata(item.source()).map_or(0, |m| m.len()));
        output_bytes =
            output_bytes.saturating_add(fs::metadata(item.target()).map_or(0, |m| m.len()));
    }
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
        plan.effective_workers(),
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
}
