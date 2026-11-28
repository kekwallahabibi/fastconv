//! Native video + audio transcoding across all supported containers.
//!
//! Source packets are demuxed once into the canonical representation, each
//! track is either copied or decoded/re-encoded by a pure-Rust codec backend,
//! then the tracks are interleaved and muxed through a crash-safe staging
//! file. NVENC modes are routed to the GPU backend with no silent CPU fallback.

use std::borrow::Cow;
use std::fs;
#[cfg(test)]
use std::fs::{File, OpenOptions};
use std::io::{self, Cursor, Write};
use std::num::{NonZeroU8, NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use flacenc::bitsink::ByteSink;
use flacenc::component::{BitRepr, StreamInfo as FlacStreamInfo};
use flacenc::error::Verify;
use flacenc::source::{Fill, FrameBuf};

use fastconv_domain::video::VideoDecodePaths;
use fastconv_domain::{
    AnimationMeta, AudioBitrate, AudioChannels, AudioEncodeConfig, AudioSampleRate,
    AudioTrackAction, DomainError, GpuBackend, HardwareBackend, TargetAudioCodec, VideoCodec,
    VideoContainerFormat, VideoDecodeMode, VideoEncodeMode, VideoTrackAction, VideoTranscodeItem,
    VideoTranscodePlan, VideoTranscodeReport,
};
use opus_pure::{Application, MAX_PACKET_BYTES, OpusEncoder};
use rubato::audioadapter_buffers::owned::InterleavedOwned;
use rubato::{Fft, FixedSync, Indexing, Resampler};
use symphonia::core::codecs::audio::well_known::{CODEC_ID_FLAC, CODEC_ID_VORBIS};
use symphonia::core::codecs::audio::{AudioCodecParameters, AudioDecoderOptions};
use symphonia::core::formats::FormatOptions;
use symphonia::core::formats::TrackType;
use symphonia::core::formats::probe::Hint;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::packet::PacketRef;
use symphonia::core::units::{Duration, Timestamp};

use crate::PipelineError;
use crate::staging::StagingTarget;
use crate::video_anim::{self, AnimFrameInfo};
use crate::video_bitstream::{
    PcmFormat, default_opus_head, extract_flac_streaminfo_from_mkv_codec_private,
    pack_xiph_laced_vorbis_headers, parse_flac_streaminfo, unpack_xiph_laced_vorbis_headers,
    vorbis_ident_info, wrap_adts,
};
use crate::video_codecs::{
    DecodedFrameInfo, Yuv420Frame, decode_video_track, decode_video_track_streaming,
    encode_video_cpu, nv12_buffer_len,
};
use crate::video_container::{
    ContainerPacket, ContainerTrack, TrackPresentation, US_TIMESCALE, demux_video, mux_video,
    rescale_ticks,
};

const AAC_BITRATE_BPS: u32 = 128_000;
const MP3_BITRATE_KBPS: u32 = 192;
const OPUS_BITRATE: i32 = 128_000;
const OPUS_RATE: u32 = 48_000;
const OPUS_FRAME: usize = 960;
const AAC_FRAME: usize = 1024;
const MP3_RATES: [u32; 9] = [
    8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000,
];
/// CPU video encoding is memory hungry (especially AV1/VP9), so at most two
/// files are transcoded in parallel regardless of the requested worker count.
const MAX_CPU_FILE_WORKERS: usize = 2;
/// Both CPU and NVENC encoders take the whole display-ordered frame slice:
/// their reference state spans the clip, so a bounded streaming handoff would
/// require new encoder APIs. Until then, cap retained decoded bytes per track
/// so a hostile input fails explicitly instead of exhausting RAM.
const MAX_DECODED_BYTES_PER_TRACK: u64 = 1_073_741_824;
/// ~32 KB per stereo `f32` chunk: the decoded PCM working set stays bounded
/// regardless of the source duration.
const AUDIO_PCM_CHUNK_FRAMES: usize = 4_096;
const AUDIO_PACKET_BATCH_SIZE: usize = 32;
const AUDIO_QUEUE_CAPACITY: usize = 16;
/// Cap on a single decoded Vorbis frame.
/// A hostile packet can never make us resize beyond this bound.
const VORBIS_FRAME_SAMPLE_CAP: usize = 8_192 * 32;
/// Number of PCM frames fed to `vorbis-enc` per `encode_audio_block` call.
const VORBIS_CHUNK_FRAMES: usize = 4_096;
/// Upper bound on a Vorbis `CodecPrivate` blob held in memory.
const VORBIS_CONFIG_CAP: usize = 1_048_576;
/// Cap on a single decoded FLAC frame (65 535 samples, up to 8 channels).
const FLAC_FRAME_SAMPLE_CAP: usize = 65_535 * 8;
/// Number of FLAC/PCM frames encoded per block before flushing a packet.
const FLAC_CHUNK_FRAMES: usize = 4_096;
const PCM_CHUNK_FRAMES: usize = 1_024;

/// Absolute presentation time of one codec frame, computed from its exact
/// rational sample position so a long stream never accumulates rounding drift.
fn sample_pts_us(sample_offset: u64, sample_rate: u32) -> i64 {
    rescale_ticks(
        i64::try_from(sample_offset).unwrap_or(i64::MAX),
        1,
        sample_rate,
    )
}

fn decode_error(path: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::DecodeFailed {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

fn encode_error(path: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

fn decode_order(packets: &[ContainerPacket]) -> Vec<&ContainerPacket> {
    let mut ordered: Vec<&ContainerPacket> = packets.iter().collect();
    ordered.sort_by_key(|packet| (packet.dts, packet.pts));
    ordered
}

/// Linear resampling of interleaved `f32` PCM. Retained as the reference
/// whole-track implementation for the codec unit tests; production transcodes
/// use [`StreamingResampler`].
#[cfg(test)]
fn resample_interleaved(
    samples: &[f32],
    channels: usize,
    from_rate: u32,
    to_rate: u32,
) -> Result<Vec<f32>, PipelineError> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }
    if channels == 0 || from_rate == 0 || to_rate == 0 {
        return Err(encode_error(
            Path::new("resampler"),
            "resampling requires non-zero rates and channels",
        ));
    }
    let mut resampler = Fft::<f32>::new(
        usize::try_from(from_rate).unwrap_or(usize::MAX),
        usize::try_from(to_rate).unwrap_or(usize::MAX),
        1024,
        channels,
        FixedSync::Both,
    )
    .map_err(|err| {
        encode_error(
            Path::new("resampler"),
            format!("failed to create resampler from {from_rate} Hz to {to_rate} Hz: {err}"),
        )
    })?;
    let chunk_in = resampler.input_frames_next();
    let chunk_out = resampler.output_frames_next().max(1);
    let delay = resampler.output_delay();
    let input_frames = samples.len() / channels;
    let expected = usize::try_from(
        u64::try_from(input_frames)
            .unwrap_or(u64::MAX)
            .saturating_mul(u64::from(to_rate))
            / u64::from(from_rate),
    )
    .unwrap_or(usize::MAX);
    // A chunk always produces `chunk_out` frames. Feed zero chunks after the
    // input ends until `expected` valid frames remain once the startup delay is
    // dropped; a short clip (even an empty one) still needs those zero chunks.
    let needed = expected.saturating_add(delay);
    let max_calls = input_frames
        .div_ceil(chunk_in.max(1))
        .saturating_add(needed.div_ceil(chunk_out))
        .saturating_add(4);
    let mut raw = Vec::new();
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
            if let Some(source) = samples.get(start..end) {
                in_storage[..source.len()].copy_from_slice(source);
            }
        }
        let input = InterleavedOwned::new_from(std::mem::take(&mut in_storage), channels, chunk_in)
            .map_err(|err| {
                encode_error(
                    Path::new("resampler"),
                    format!("invalid resampler input: {err}"),
                )
            })?;
        let mut output =
            InterleavedOwned::new_from(std::mem::take(&mut out_storage), channels, chunk_out)
                .map_err(|err| {
                    encode_error(
                        Path::new("resampler"),
                        format!("invalid resampler output: {err}"),
                    )
                })?;
        let indexing = Indexing::new().partial_len(take);
        let (_, written) = resampler
            .process_into_buffer(&input, &mut output, Some(&indexing))
            .map_err(|err| {
                encode_error(Path::new("resampler"), format!("resampling failed: {err}"))
            })?;
        in_storage = input.take_data();
        out_storage = output.take_data();
        let usable = written.saturating_mul(channels).min(out_storage.len());
        raw.extend_from_slice(&out_storage[..usable]);
        consumed = consumed.saturating_add(take);
        produced = produced.saturating_add(written);
    }
    let start = delay.saturating_mul(channels);
    let end = start.saturating_add(expected.saturating_mul(channels));
    Ok(raw.get(start..end).unwrap_or(&[]).to_vec())
}

/// Incremental resampler. Only a rate change instantiates the FFT engine;
/// otherwise chunks pass straight through. Full input blocks are processed as
/// they arrive and the final partial block plus the FFT overlap tail are
/// flushed in `finish`, capped to the exact `in * to / from` frame count.
struct StreamingResampler {
    resampler: Option<Fft<f32>>,
    channels: usize,
    from_rate: u32,
    to_rate: u32,
    /// Interleaved samples not yet forming a complete input chunk.
    staging: Vec<f32>,
    chunk_in: usize,
    chunk_out: usize,
    /// Startup delay still to drop from the very front of the output stream.
    skip_remaining: usize,
    in_storage: Vec<f32>,
    out_storage: Vec<f32>,
    total_in_frames: u64,
    emitted_out_frames: u64,
}

impl StreamingResampler {
    fn new(channels: usize, from_rate: u32, to_rate: u32) -> Result<Self, PipelineError> {
        if channels == 0 || from_rate == 0 || to_rate == 0 {
            return Err(encode_error(
                Path::new("resampler"),
                "resampling requires non-zero rates and channels",
            ));
        }
        if from_rate == to_rate {
            return Ok(Self {
                resampler: None,
                channels,
                from_rate,
                to_rate,
                staging: Vec::new(),
                chunk_in: 0,
                chunk_out: 0,
                skip_remaining: 0,
                in_storage: Vec::new(),
                out_storage: Vec::new(),
                total_in_frames: 0,
                emitted_out_frames: 0,
            });
        }
        let resampler = Fft::<f32>::new(
            usize::try_from(from_rate).unwrap_or(usize::MAX),
            usize::try_from(to_rate).unwrap_or(usize::MAX),
            1024,
            channels,
            FixedSync::Both,
        )
        .map_err(|err| {
            encode_error(
                Path::new("resampler"),
                format!("failed to create resampler from {from_rate} Hz to {to_rate} Hz: {err}"),
            )
        })?;
        let chunk_in = resampler.input_frames_next();
        let chunk_out = resampler.output_frames_next().max(1);
        let delay = resampler.output_delay();
        Ok(Self {
            resampler: Some(resampler),
            channels,
            from_rate,
            to_rate,
            staging: Vec::new(),
            chunk_in,
            chunk_out,
            skip_remaining: delay,
            in_storage: vec![0.0_f32; chunk_in.saturating_mul(channels)],
            out_storage: vec![0.0_f32; chunk_out.saturating_mul(channels)],
            total_in_frames: 0,
            emitted_out_frames: 0,
        })
    }

    /// Appends full output frames to `out`, dropping the leading startup delay
    /// and never exceeding `budget` total emitted frames.
    fn emit(&mut self, out: &mut Vec<f32>, raw: &[f32], budget: u64) {
        let channels = self.channels.max(1);
        for frame in raw.chunks(channels) {
            if self.skip_remaining > 0 {
                self.skip_remaining -= 1;
                continue;
            }
            if self.emitted_out_frames >= budget {
                break;
            }
            out.extend_from_slice(frame);
            self.emitted_out_frames = self.emitted_out_frames.saturating_add(1);
        }
    }

    /// Processes one zero-padded input chunk with `valid_frames` real frames at
    /// the front and appends its output, trimmed to the frame budget.
    fn process_block(
        &mut self,
        valid_frames: usize,
        budget: u64,
        out: &mut Vec<f32>,
    ) -> Result<(), PipelineError> {
        let input = InterleavedOwned::new_from(
            std::mem::take(&mut self.in_storage),
            self.channels,
            self.chunk_in,
        )
        .map_err(|err| {
            encode_error(
                Path::new("resampler"),
                format!("invalid resampler input: {err}"),
            )
        })?;
        let mut output = InterleavedOwned::new_from(
            std::mem::take(&mut self.out_storage),
            self.channels,
            self.chunk_out,
        )
        .map_err(|err| {
            encode_error(
                Path::new("resampler"),
                format!("invalid resampler output: {err}"),
            )
        })?;
        let indexing = Indexing::new().partial_len(valid_frames);
        let (_, written) = self
            .resampler
            .as_mut()
            .ok_or_else(|| encode_error(Path::new("resampler"), "resampler is not configured"))?
            .process_into_buffer(&input, &mut output, Some(&indexing))
            .map_err(|err| {
                encode_error(Path::new("resampler"), format!("resampling failed: {err}"))
            })?;
        self.in_storage = input.take_data();
        self.out_storage = output.take_data();
        let usable = written
            .saturating_mul(self.channels)
            .min(self.out_storage.len());
        let raw = self.out_storage[..usable].to_vec();
        self.emit(out, &raw, budget);
        Ok(())
    }

    fn push(&mut self, samples: &[f32]) -> Result<Vec<f32>, PipelineError> {
        self.total_in_frames = self.total_in_frames.saturating_add(
            u64::try_from(samples.len() / self.channels.max(1)).unwrap_or(u64::MAX),
        );
        if self.resampler.is_none() {
            return Ok(samples.to_vec());
        }
        if self.chunk_in == 0 {
            return Err(encode_error(
                Path::new("resampler"),
                "resampler requires a non-zero input block size",
            ));
        }
        self.staging.extend_from_slice(samples);
        let block_samples = self.chunk_in.saturating_mul(self.channels);
        let mut out = Vec::new();
        while self.staging.len() >= block_samples {
            let rest = self.staging.split_off(block_samples);
            let block = std::mem::replace(&mut self.staging, rest);
            let copied = block_samples.min(self.in_storage.len());
            self.in_storage[..copied].copy_from_slice(&block[..copied]);
            self.process_block(self.chunk_in, u64::MAX, &mut out)?;
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<f32>, PipelineError> {
        if self.resampler.is_none() {
            return Ok(Vec::new());
        }
        // Cap the whole tail (padded final chunk + zero drain) at the exact
        // whole-track frame count so the timeline never stretches.
        let expected = self.total_in_frames.saturating_mul(u64::from(self.to_rate))
            / u64::from(self.from_rate.max(1));
        let budget = expected;
        let mut out = Vec::new();
        let staged_frames = self.staging.len() / self.channels.max(1);
        if staged_frames > 0 {
            self.in_storage.fill(0.0);
            let copied = staged_frames
                .saturating_mul(self.channels)
                .min(self.in_storage.len())
                .min(self.staging.len());
            self.in_storage[..copied].copy_from_slice(&self.staging[..copied]);
            self.process_block(staged_frames, budget, &mut out)?;
            self.staging.clear();
        }
        let chunk_out_u64 = u64::try_from(self.chunk_out.max(1)).unwrap_or(u64::MAX);
        let max_flush = usize::try_from(budget.div_ceil(chunk_out_u64))
            .unwrap_or(usize::MAX)
            .saturating_add(8);
        for _ in 0..max_flush {
            if self.emitted_out_frames >= budget {
                break;
            }
            self.in_storage.fill(0.0);
            self.process_block(0, budget, &mut out)?;
        }
        let allowed = usize::try_from(
            budget.saturating_mul(u64::try_from(self.channels).unwrap_or(u64::MAX)),
        )
        .unwrap_or(usize::MAX);
        out.truncate(allowed);
        Ok(out)
    }
}

/// Converts interleaved container PCM bytes to bounded interleaved `f32`.
/// The caller validates that `data` is a whole number of frames.
// One arm per exact representation keeps endianness and depth explicit.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn pcm_bytes_to_f32(data: &[u8], format: PcmFormat) -> Vec<f32> {
    let bytes = usize::from(format.bytes_per_sample());
    let mut out = Vec::with_capacity(data.len() / bytes.max(1));
    for sample in data.chunks_exact(bytes.max(1)) {
        let value = match format {
            PcmFormat::U8 => (f32::from(sample[0]) - 128.0_f32) / 128.0_f32,
            PcmFormat::S16Le | PcmFormat::S16Be => {
                let raw = if format == PcmFormat::S16Le {
                    i16::from_le_bytes([sample[0], sample[1]])
                } else {
                    i16::from_be_bytes([sample[0], sample[1]])
                };
                f32::from(raw) / 32_768.0_f32
            }
            PcmFormat::S24Le => {
                let raw = (i32::from(sample[2]) << 24
                    | i32::from(sample[1]) << 16
                    | i32::from(sample[0]) << 8)
                    >> 8;
                raw as f32 / 8_388_608.0_f32
            }
            PcmFormat::S24Be => {
                let raw = (i32::from(sample[0]) << 24
                    | i32::from(sample[1]) << 16
                    | i32::from(sample[2]) << 8)
                    >> 8;
                raw as f32 / 8_388_608.0_f32
            }
            PcmFormat::S32Le => {
                let raw = i32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]);
                raw as f32 / 2_147_483_648.0_f32
            }
            PcmFormat::S32Be => {
                let raw = i32::from_be_bytes([sample[0], sample[1], sample[2], sample[3]]);
                raw as f32 / 2_147_483_648.0_f32
            }
            PcmFormat::F32Le => f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]),
            PcmFormat::F32Be => f32::from_be_bytes([sample[0], sample[1], sample[2], sample[3]]),
            PcmFormat::F64Le => match <[u8; 8]>::try_from(sample) {
                Ok(bits) => f64::from_le_bytes(bits) as f32,
                Err(_) => continue,
            },
        };
        out.push(value);
    }
    out
}

/// Decodes a track in bounded PCM chunks. `on_chunk` receives interleaved
/// `f32` samples at the decoder's native rate; cancellation is observed on
/// every packet so a failing sibling worker can stop this one immediately.
// One branch per codec keeps decode-order and rate/channel handling explicit.
#[allow(clippy::too_many_lines)]
fn decode_audio_track_chunks<F>(
    path: &Path,
    track: &ContainerTrack,
    packets: &[ContainerPacket],
    cancelled: &AtomicBool,
    mut on_chunk: F,
) -> Result<(), PipelineError>
where
    F: FnMut(&[f32], u32, u16) -> Result<(), PipelineError>,
{
    let ordered = decode_order(packets);
    let channels = track.channels.unwrap_or(2).max(1);
    let mut chunk = Vec::new();
    // `cursor` advances past emitted blocks so a large chunk is not shifted
    // from the front once per output block; the consumed prefix is compacted
    // once per call instead.
    let mut cursor = 0_usize;
    let mut last_rate = 0_u32;
    let mut last_channels = 0_u16;
    let emit_ready = |chunk: &mut Vec<f32>,
                      cursor: &mut usize,
                      rate: u32,
                      decoded: u16,
                      on_chunk: &mut F|
     -> Result<(), PipelineError> {
        let ready = AUDIO_PCM_CHUNK_FRAMES.saturating_mul(usize::from(decoded.max(1)));
        loop {
            let available = chunk.len().saturating_sub(*cursor);
            if available < ready {
                break;
            }
            let end = cursor.saturating_add(ready);
            on_chunk(&chunk[*cursor..end], rate, decoded)?;
            *cursor = end;
        }
        if *cursor > 0 {
            chunk.drain(..*cursor);
            *cursor = 0;
        }
        Ok(())
    };

    match track.codec {
        VideoCodec::Opus => {
            let pre_skip = if track.config.len() >= 12 && &track.config[..8] == b"OpusHead" {
                usize::from(u16::from_le_bytes([track.config[10], track.config[11]]))
            } else {
                0
            };
            let mut decoder = opus_pure::OpusDecoder::new(
                i32::try_from(OPUS_RATE).unwrap_or(i32::MAX),
                usize::from(channels),
            )
            .map_err(|err| decode_error(path, format!("failed to create Opus decoder: {err}")))?;
            let mut skip = pre_skip;
            let mut block = Vec::new();
            for packet in ordered {
                if cancelled.load(Ordering::Acquire) {
                    return Ok(());
                }
                // Opus packets may hold 2.5-120 ms of audio, not just the
                // 20 ms frame we choose when encoding. Size the output from
                // this packet's TOC instead of over- or under-shadowing it.
                let frames = opus_pure::packet::samples(
                    &packet.data,
                    i32::try_from(OPUS_RATE).unwrap_or(i32::MAX),
                )
                .map_err(|err| {
                    decode_error(path, format!("invalid Opus packet duration: {err}"))
                })?;
                let needed = frames.saturating_mul(usize::from(channels));
                if block.len() < needed {
                    block.resize(needed, 0.0_f32);
                }
                let count = decoder
                    .decode(&packet.data, frames, &mut block[..needed])
                    .map_err(|err| decode_error(path, format!("Opus decode failed: {err}")))?;
                let start = skip.min(count);
                skip = skip.saturating_sub(start);
                if count > start {
                    let from = start.saturating_mul(usize::from(channels));
                    let to = count.saturating_mul(usize::from(channels)).min(needed);
                    if let Some(samples) = block.get(from..to) {
                        chunk.extend_from_slice(samples);
                    }
                }
                emit_ready(&mut chunk, &mut cursor, OPUS_RATE, channels, &mut on_chunk)?;
            }
            last_rate = OPUS_RATE;
            last_channels = channels;
        }
        VideoCodec::Aac => {
            let mut decoder = rusty_aac::AacDecoder::new();
            for packet in ordered {
                if cancelled.load(Ordering::Acquire) {
                    return Ok(());
                }
                let adts = wrap_adts(&track.config, &packet.data)
                    .map_err(|reason| decode_error(path, reason))?;
                let frame = decoder
                    .decode(&adts, None)
                    .map_err(|err| decode_error(path, format!("AAC decode failed: {err}")))?;
                last_rate = frame.sample_rate;
                last_channels = frame.channels;
                chunk.extend_from_slice(&frame.samples);
                emit_ready(
                    &mut chunk,
                    &mut cursor,
                    last_rate,
                    last_channels,
                    &mut on_chunk,
                )?;
            }
        }
        VideoCodec::Mp3 => {
            let mut bytes = Vec::new();
            for packet in ordered {
                bytes.extend_from_slice(&packet.data);
            }
            let mss = MediaSourceStream::new(
                Box::new(Cursor::new(bytes)),
                MediaSourceStreamOptions::default(),
            );
            let mut hint = Hint::new();
            hint.with_extension("mp3");
            let mut reader = symphonia::default::get_probe()
                .probe(
                    &hint,
                    mss,
                    FormatOptions::default(),
                    MetadataOptions::default(),
                )
                .map_err(|err| decode_error(path, format!("failed to probe MP3 audio: {err}")))?;
            let audio_track = reader
                .default_track(TrackType::Audio)
                .ok_or_else(|| decode_error(path, "MP3 track has no audio stream"))?;
            let params = audio_track
                .codec_params
                .as_ref()
                .and_then(|p| p.audio())
                .ok_or_else(|| decode_error(path, "MP3 track has no codec parameters"))?
                .clone();
            last_rate = params.sample_rate.unwrap_or(0);
            last_channels = u16::try_from(
                params
                    .channels
                    .as_ref()
                    .map_or(0, symphonia::core::audio::Channels::count),
            )
            .unwrap_or(0);
            let track_id = audio_track.id;
            let mut decoder = symphonia::default::get_codecs()
                .make_audio_decoder(&params, &AudioDecoderOptions::default())
                .map_err(|err| {
                    decode_error(path, format!("failed to create MP3 decoder: {err}"))
                })?;
            let mut frame = Vec::new();
            loop {
                if cancelled.load(Ordering::Acquire) {
                    return Ok(());
                }
                let packet = match reader.next_packet() {
                    Ok(Some(packet)) => packet,
                    Ok(None) => break,
                    Err(err) => {
                        return Err(decode_error(path, format!("MP3 packet read failed: {err}")));
                    }
                };
                if packet.track_id != track_id {
                    continue;
                }
                let audio_frame = decoder
                    .decode(&packet)
                    .map_err(|err| decode_error(path, format!("MP3 decode failed: {err}")))?;
                let count = audio_frame.samples_interleaved();
                if frame.len() < count {
                    frame.resize(count, 0.0_f32);
                }
                if let Some(slot) = frame.get_mut(..count) {
                    audio_frame.copy_to_slice_interleaved(&mut *slot);
                    chunk.extend_from_slice(&slot[..count]);
                }
                emit_ready(
                    &mut chunk,
                    &mut cursor,
                    last_rate,
                    last_channels,
                    &mut on_chunk,
                )?;
            }
        }
        VideoCodec::Vorbis => {
            let (ident, _, _) = unpack_xiph_laced_vorbis_headers(&track.config)
                .map_err(|reason| decode_error(path, format!("invalid Vorbis header: {reason}")))?;
            let (ident_channels, ident_rate, _, _) = vorbis_ident_info(ident)
                .map_err(|reason| decode_error(path, format!("invalid Vorbis header: {reason}")))?;
            let mut params = AudioCodecParameters::new();
            params
                .for_codec(CODEC_ID_VORBIS)
                .with_sample_rate(ident_rate)
                .with_extra_data(track.config.clone().into_boxed_slice());
            // `gapless: true` (symphonia's default) silences the first packet,
            // which is the Vorbis priming block.
            let mut decoder = symphonia::default::get_codecs()
                .make_audio_decoder(&params, &AudioDecoderOptions::default())
                .map_err(|err| {
                    decode_error(path, format!("failed to create Vorbis decoder: {err}"))
                })?;
            let mut frame: Vec<f32> = Vec::new();
            for packet in ordered {
                if cancelled.load(Ordering::Acquire) {
                    return Ok(());
                }
                let audio_buf = decoder
                    .decode_ref(&PacketRef::new(
                        0,
                        Timestamp::ZERO,
                        Duration::ZERO,
                        &packet.data,
                    ))
                    .map_err(|err| decode_error(path, format!("Vorbis decode failed: {err}")))?;
                let count = audio_buf.samples_interleaved();
                if count == 0 {
                    // The gapless priming packet decodes to an empty buffer.
                    continue;
                }
                if count > VORBIS_FRAME_SAMPLE_CAP {
                    return Err(decode_error(
                        path,
                        format!("Vorbis frame of {count} samples exceeds the allocation cap"),
                    ));
                }
                if frame.len() < count {
                    frame.resize(count, 0.0_f32);
                }
                if let Some(slot) = frame.get_mut(..count) {
                    audio_buf.copy_to_slice_interleaved(&mut *slot);
                    chunk.extend_from_slice(&slot[..count]);
                }
                last_rate = ident_rate;
                last_channels = ident_channels;
                emit_ready(
                    &mut chunk,
                    &mut cursor,
                    last_rate,
                    last_channels,
                    &mut on_chunk,
                )?;
            }
        }
        VideoCodec::Flac => {
            // A remuxed track may carry the full `fLaC` metadata chain (e.g. a
            // VORBIS_COMMENT block). Walk it to the validated STREAMINFO body
            // instead of demanding a bare 34-byte config; `track.config` itself
            // stays untouched so a later mux preserves every native block.
            let streaminfo: [u8; 34] = if track.config.starts_with(b"fLaC") {
                extract_flac_streaminfo_from_mkv_codec_private(&track.config).map_err(|reason| {
                    decode_error(path, format!("invalid FLAC metadata: {reason}"))
                })?
            } else {
                track
                    .config
                    .as_slice()
                    .try_into()
                    .map_err(|_| decode_error(path, "FLAC track has no 34-byte STREAMINFO"))?
            };
            let (flac_channels, flac_rate, _) =
                parse_flac_streaminfo(&streaminfo).map_err(|reason| decode_error(path, reason))?;
            let mut params = AudioCodecParameters::new();
            params.for_codec(CODEC_ID_FLAC).with_sample_rate(flac_rate);
            // The decoder amends channels/bit depth from STREAMINFO itself.
            // `symphonia-bundle-flac` reads the raw 34-byte STREAMINFO directly,
            // so pass only the validated body, never the whole metadata chain.
            params.with_extra_data(streaminfo.to_vec().into_boxed_slice());
            let mut decoder = symphonia::default::get_codecs()
                .make_audio_decoder(&params, &AudioDecoderOptions::default())
                .map_err(|err| {
                    decode_error(path, format!("failed to create FLAC decoder: {err}"))
                })?;
            let mut frame: Vec<f32> = Vec::new();
            for packet in ordered {
                if cancelled.load(Ordering::Acquire) {
                    return Ok(());
                }
                let audio_buf = decoder
                    .decode_ref(&PacketRef::new(
                        0,
                        Timestamp::ZERO,
                        Duration::ZERO,
                        &packet.data,
                    ))
                    .map_err(|err| decode_error(path, format!("FLAC decode failed: {err}")))?;
                let count = audio_buf.samples_interleaved();
                if count > FLAC_FRAME_SAMPLE_CAP {
                    return Err(decode_error(
                        path,
                        format!("FLAC frame of {count} samples exceeds the allocation cap"),
                    ));
                }
                if frame.len() < count {
                    frame.resize(count, 0.0_f32);
                }
                if let Some(slot) = frame.get_mut(..count) {
                    audio_buf.copy_to_slice_interleaved(&mut *slot);
                    chunk.extend_from_slice(&slot[..count]);
                }
                last_rate = flac_rate;
                last_channels = flac_channels;
                emit_ready(
                    &mut chunk,
                    &mut cursor,
                    last_rate,
                    last_channels,
                    &mut on_chunk,
                )?;
            }
        }
        VideoCodec::Pcm => {
            let format = track
                .pcm_format
                .ok_or_else(|| decode_error(path, "PCM track has no sample format"))?;
            let frame_bytes =
                usize::from(format.bytes_per_sample()).saturating_mul(usize::from(channels));
            if frame_bytes == 0 {
                return Err(decode_error(path, "PCM frame size is zero"));
            }
            last_rate = track.sample_rate.unwrap_or(0);
            last_channels = channels;
            // Convert at most one output block of whole frames at a time so a
            // large packet is never duplicated wholesale as f32 and `chunk`
            // stays at one block plus one slice.
            let slice_bytes = AUDIO_PCM_CHUNK_FRAMES
                .max(1)
                .saturating_mul(frame_bytes)
                .max(1);
            for packet in ordered {
                if cancelled.load(Ordering::Acquire) {
                    return Ok(());
                }
                if packet.data.len() % frame_bytes != 0 {
                    return Err(decode_error(
                        path,
                        "PCM packet is not a whole number of frames",
                    ));
                }
                for slice in packet.data.chunks(slice_bytes) {
                    if slice.len() % frame_bytes != 0 {
                        return Err(decode_error(
                            path,
                            "PCM packet is not a whole number of frames",
                        ));
                    }
                    let samples = u64::try_from(slice.len() / frame_bytes)
                        .unwrap_or(u64::MAX)
                        .saturating_mul(u64::from(channels));
                    // `emit_ready` hands off and compacts after every slice, so
                    // budget the live f32 working set, not a cumulative counter
                    // that would eventually reject a long, healthy stream.
                    enforce_live_pcm_budget(path, &chunk, samples)?;
                    chunk.extend_from_slice(&pcm_bytes_to_f32(slice, format));
                    emit_ready(
                        &mut chunk,
                        &mut cursor,
                        last_rate,
                        last_channels,
                        &mut on_chunk,
                    )?;
                }
            }
        }
        other => {
            return Err(decode_error(
                path,
                format!("audio codec {other} is not supported for transcoding"),
            ));
        }
    }

    if chunk.is_empty() {
        return Ok(());
    }
    if last_rate == 0 || last_channels == 0 {
        return Err(decode_error(
            path,
            "audio track has no valid rate or channels",
        ));
    }
    let tail = chunk.clone();
    on_chunk(&tail, last_rate, last_channels)
}

/// Decodes one audio track to interleaved `f32` PCM at its native rate.
#[cfg(test)]
pub(crate) fn decode_audio_track_pcm(
    path: &Path,
    track: &ContainerTrack,
    packets: &[ContainerPacket],
) -> Result<(Vec<f32>, u32, u16), PipelineError> {
    let mut pcm = Vec::new();
    let mut rate = 0_u32;
    let mut channels = 0_u16;
    decode_audio_track_chunks(
        path,
        track,
        packets,
        &AtomicBool::new(false),
        |chunk, sample_rate, decoded_channels| {
            if rate == 0 {
                rate = sample_rate;
                channels = decoded_channels;
            }
            pcm.extend_from_slice(chunk);
            Ok(())
        },
    )?;
    if pcm.is_empty() || rate == 0 || channels == 0 {
        return Err(decode_error(
            path,
            format!("{} track decoded to no audio", track.codec),
        ));
    }
    Ok((pcm, rate, channels))
}

fn aac_config(rate: u32, channels: u16) -> Result<[u8; 2], PipelineError> {
    let index = rusty_aac::sf_index_for_rate(rate).ok_or_else(|| {
        encode_error(
            Path::new("aac"),
            format!("sample rate {rate} Hz is not supported by AAC"),
        )
    })?;
    Ok([
        (2_u8 << 3) | (index >> 1),
        ((index & 1) << 7) | u8::try_from(channels << 3).unwrap_or(u8::MAX),
    ])
}

#[cfg(test)]
fn encode_aac_track(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    config: AudioEncodeConfig,
) -> Result<(ContainerTrack, Vec<ContainerPacket>), PipelineError> {
    let rate = configured_audio_rate(TargetAudioCodec::Aac, sample_rate, config);
    let target_channels = configured_audio_channels(channels, config);
    if !(1..=6).contains(&target_channels) || target_channels == 4 {
        return Err(encode_error(
            path,
            format!("AAC output supports 1-3 and 5-6 channels, got {target_channels}"),
        ));
    }
    let resampled = resample_interleaved(samples, usize::from(channels), sample_rate, rate)?;
    let pcm = mix_to_channels(
        &resampled,
        usize::from(channels),
        usize::from(target_channels),
    )?;
    if pcm.is_empty() {
        return Err(encode_error(path, "no samples to encode as AAC"));
    }
    let channels = target_channels;
    let asc = aac_config(rate, channels)?;
    let mut encoder = rusty_aac::AacEncoder::new(rusty_aac::AacEncoderConfig {
        bitrate_bps: configured_aac_bitrate(config),
        ..rusty_aac::AacEncoderConfig::default()
    });
    encoder
        .push_pcm(&pcm, channels, rate)
        .map_err(|err| encode_error(path, format!("AAC encoding failed: {err}")))?;
    encoder.finish();
    let mut raw_frames = Vec::new();
    loop {
        match encoder.next_packet() {
            Ok(packet) => {
                // `rusty_aac::AacEncoder` emits raw AAC access units directly.
                if !packet.data.is_empty() {
                    raw_frames.push(packet.data);
                }
            }
            Err(rusty_aac::Error::Eof | rusty_aac::Error::Again) => break,
            Err(err) => return Err(encode_error(path, format!("AAC drain failed: {err}"))),
        }
    }
    if raw_frames.is_empty() {
        return Err(encode_error(path, "AAC encoder produced no frames"));
    }
    let config = asc.to_vec();
    let frame_samples = u64::try_from(AAC_FRAME).unwrap_or(u64::MAX);
    let mut packets = Vec::with_capacity(raw_frames.len());
    for (index, data) in raw_frames.into_iter().enumerate() {
        let sample_offset = u64::try_from(index)
            .unwrap_or(u64::MAX)
            .saturating_mul(frame_samples);
        let pts = sample_pts_us(sample_offset, rate);
        let next_pts = sample_pts_us(sample_offset.saturating_add(frame_samples), rate);
        let duration = u64::try_from(next_pts.saturating_sub(pts))
            .unwrap_or(1)
            .max(1);
        packets.push(ContainerPacket {
            track: 0,
            pts,
            dts: pts,
            duration,
            keyframe: true,
            data,
        });
    }
    Ok((
        ContainerTrack {
            codec: VideoCodec::Aac,
            width: None,
            height: None,
            channels: Some(channels),
            sample_rate: Some(rate),
            config,
            timescale: US_TIMESCALE,
            presentation_edit: None,
            pcm_format: None,
        },
        packets,
    ))
}

#[cfg(test)]
fn encode_opus_track(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    config: AudioEncodeConfig,
) -> Result<(ContainerTrack, Vec<ContainerPacket>), PipelineError> {
    let target_channels = configured_audio_channels(channels, config);
    if target_channels == 0 || target_channels > 2 {
        return Err(encode_error(
            path,
            format!("Opus output supports mono/stereo only, got {target_channels}"),
        ));
    }
    let rate = configured_audio_rate(TargetAudioCodec::Opus, sample_rate, config);
    let resampled = resample_interleaved(samples, usize::from(channels), sample_rate, rate)?;
    let pcm = mix_to_channels(
        &resampled,
        usize::from(channels),
        usize::from(target_channels),
    )?;
    if pcm.is_empty() {
        return Err(encode_error(path, "no samples to encode as Opus"));
    }
    let channels = target_channels;
    let mut encoder = OpusEncoder::new(
        i32::try_from(rate).unwrap_or(i32::MAX),
        usize::from(channels),
        Application::Audio,
    )
    .map_err(|err| encode_error(path, format!("failed to create Opus encoder: {err}")))?;
    encoder.bitrate_bps = configured_opus_bitrate(config);
    encoder.complexity = 9;
    let head = opus_pure::OpusHead::for_encoder(&encoder, rate);
    let pre_skip = head.pre_skip;
    let frame_samples = OPUS_FRAME.saturating_mul(usize::from(channels));
    let input_frames = pcm.len() / usize::from(channels);
    // The encoder's lookahead is reported as OpusHead pre-skip, so those
    // samples must be fed through the encoder as extra padded frames. Without
    // them playback would end pre-skip samples too early.
    let total_frames = input_frames.saturating_add(usize::from(pre_skip));
    let frames = total_frames.div_ceil(OPUS_FRAME).max(1);
    let final_frames = total_frames
        .saturating_sub(frames.saturating_sub(1).saturating_mul(OPUS_FRAME))
        .clamp(1, OPUS_FRAME);
    let final_duration = u64::try_from(final_frames)
        .unwrap_or(u64::MAX)
        .saturating_mul(1_000_000)
        / u64::from(OPUS_RATE);
    let mut packets = Vec::with_capacity(frames);
    let mut packet = vec![0_u8; MAX_PACKET_BYTES];
    for index in 0..frames {
        let start = index.saturating_mul(frame_samples);
        let mut block = vec![0.0_f32; frame_samples];
        if let Some(source) = pcm.get(start..start.saturating_add(frame_samples).min(pcm.len())) {
            block[..source.len()].copy_from_slice(source);
        }
        let count = encoder
            .encode(&block, OPUS_FRAME, &mut packet)
            .map_err(|err| encode_error(path, format!("Opus encoding failed: {err}")))?;
        if count == 0 {
            return Err(encode_error(path, "Opus encoder produced an empty packet"));
        }
        let pts = i64::try_from(index)
            .unwrap_or(i64::MAX)
            .saturating_mul(20_000);
        packets.push(ContainerPacket {
            track: 0,
            pts,
            dts: pts,
            // Tell the muxer the exact unplayed tail in the last packet: the
            // total decoded duration then equals the source's input duration.
            duration: if index + 1 == frames {
                final_duration
            } else {
                20_000
            },
            keyframe: true,
            data: packet[..count].to_vec(),
        });
    }
    Ok((
        ContainerTrack {
            codec: VideoCodec::Opus,
            width: None,
            height: None,
            channels: Some(channels),
            sample_rate: Some(rate),
            config: default_opus_head(channels, pre_skip),
            timescale: US_TIMESCALE,
            presentation_edit: None,
            pcm_format: None,
        },
        packets,
    ))
}

#[cfg(test)]
fn encode_mp3_track(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    config: AudioEncodeConfig,
) -> Result<(ContainerTrack, Vec<ContainerPacket>), PipelineError> {
    let target_channels = configured_audio_channels(channels, config);
    if target_channels == 0 || target_channels > 2 {
        return Err(encode_error(
            path,
            format!("MP3 output supports mono/stereo only, got {target_channels}"),
        ));
    }
    let rate = configured_audio_rate(TargetAudioCodec::Mp3, sample_rate, config);
    let resampled = resample_interleaved(samples, usize::from(channels), sample_rate, rate)?;
    let pcm = mix_to_channels(
        &resampled,
        usize::from(channels),
        usize::from(target_channels),
    )?;
    if pcm.is_empty() {
        return Err(encode_error(path, "no samples to encode as MP3"));
    }
    let channels = target_channels;
    let mut encoder = rusty_mp3::Mp3Encoder::new(rusty_mp3::Mp3EncoderConfig {
        bitrate_kbps: checked_mp3_bitrate_kbps(config, rate, path)?,
        vbr_quality: None,
    });
    encoder
        .push_pcm_f32(&pcm, channels, rate)
        .map_err(|err| encode_error(path, format!("MP3 encoding failed: {err}")))?;
    encoder.finish();
    let mut frames = Vec::new();
    loop {
        match encoder.next_packet() {
            Ok(frame) => frames.push(frame),
            Err(rusty_mp3::Error::Eof | rusty_mp3::Error::Again) => break,
            Err(err) => return Err(encode_error(path, format!("MP3 drain failed: {err}"))),
        }
    }
    if frames.is_empty() {
        return Err(encode_error(path, "MP3 encoder produced no frames"));
    }
    let frame_samples = if rate >= 32_000 { 1_152_u64 } else { 576_u64 };
    let mut packets = Vec::with_capacity(frames.len());
    for (index, data) in frames.into_iter().enumerate() {
        let sample_offset = u64::try_from(index)
            .unwrap_or(u64::MAX)
            .saturating_mul(frame_samples);
        let pts = sample_pts_us(sample_offset, rate);
        let next_pts = sample_pts_us(sample_offset.saturating_add(frame_samples), rate);
        let duration = u64::try_from(next_pts.saturating_sub(pts))
            .unwrap_or(1)
            .max(1);
        packets.push(ContainerPacket {
            track: 0,
            pts,
            dts: pts,
            duration,
            keyframe: true,
            data,
        });
    }
    Ok((
        ContainerTrack {
            codec: VideoCodec::Mp3,
            width: None,
            height: None,
            channels: Some(channels),
            sample_rate: Some(rate),
            config: Vec::new(),
            timescale: US_TIMESCALE,
            presentation_edit: None,
            pcm_format: None,
        },
        packets,
    ))
}

/// Quantizes one `f32` sample to the signed 16-bit range flacenc expects.
// Rounding/clamping is deliberate: the f32 decoder output is normalized.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn to_s16_i32(sample: f32) -> i32 {
    // Decoding divides signed samples by 32768, so -1.0 must map back to
    // -32768 (not -32767). Multiplying by 32768 keeps the negative endpoint
    // exact; clamping still keeps +1.0 inside i16's asymmetric positive max.
    (sample * 32_768.0).round().clamp(-32_768.0, 32_767.0) as i32
}

/// Interleaved `f32` to signed 16-bit little-endian PCM bytes.
fn pcm_to_s16le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len().saturating_mul(2));
    for &sample in samples {
        let value = i16::try_from(to_s16_i32(sample)).unwrap_or(0);
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// Validated, single-threaded FLAC encoder settings for in-container audio.
fn flac_encoder_config() -> Result<flacenc::error::Verified<flacenc::config::Encoder>, PipelineError>
{
    let mut config = flacenc::config::Encoder::default();
    config.block_size = FLAC_CHUNK_FRAMES;
    config.multithread = false;
    config.subframe_coding.use_lpc = false;
    config.into_verified().map_err(|(_, err)| {
        encode_error(
            Path::new("flac"),
            format!("invalid FLAC encoder configuration: {err}"),
        )
    })
}

/// Builds the FLAC encoder's `StreamInfo`; `total_samples == 0` means unknown.
fn flac_streaminfo(
    rate: u32,
    channels: u16,
    total_samples: usize,
) -> Result<FlacStreamInfo, PipelineError> {
    let rate = usize::try_from(rate)
        .map_err(|_| encode_error(Path::new("flac"), "FLAC rate exceeds usize"))?;
    let mut info = FlacStreamInfo::new(rate, usize::from(channels), 16).map_err(|err| {
        encode_error(Path::new("flac"), format!("invalid FLAC StreamInfo: {err}"))
    })?;
    // Fixed-block frames must declare equal min/max bounds: when they differ,
    // readers treat frame numbers as absolute sample positions and a second
    // 4096-sample block lands at sample 1 instead of 4096, breaking seeks.
    // The final shorter block is excluded from the min-block-size rule.
    info.set_block_sizes(FLAC_CHUNK_FRAMES, FLAC_CHUNK_FRAMES)
        .map_err(|err| {
            encode_error(
                Path::new("flac"),
                format!("invalid FLAC block sizes: {err}"),
            )
        })?;
    info.set_total_samples(total_samples);
    Ok(info)
}

/// Writes exactly the 34-byte raw FLAC STREAMINFO used as container config.
fn flac_streaminfo_bytes(info: &FlacStreamInfo) -> Result<[u8; 34], PipelineError> {
    let mut sink = ByteSink::new();
    BitRepr::write(info, &mut sink).map_err(|err| {
        encode_error(Path::new("flac"), format!("STREAMINFO write failed: {err}"))
    })?;
    let bytes = sink.into_inner();
    <[u8; 34]>::try_from(bytes).map_err(|bytes| {
        encode_error(
            Path::new("flac"),
            format!("STREAMINFO is {} bytes, expected 34", bytes.len()),
        )
    })
}

/// Minimum FLAC block size accepted by RFC 9639 / `flacenc`.
const FLAC_MIN_BLOCK_FRAMES: usize = 16;

/// Encodes complete FLAC frames from interleaved `f32` PCM into packets.
///
/// `start_frame_number` is the 0-based FLAC frame index (`0, 1, 2, ...`);
/// `start_sample_offset` is the 0-based sample offset used for timestamps.
/// Returns `(packets, frames_emitted, samples_emitted)`.
fn encode_flac_frames(
    pcm: &[f32],
    channels: u16,
    rate: u32,
    start_frame_number: usize,
    start_sample_offset: u64,
    config: &flacenc::error::Verified<flacenc::config::Encoder>,
    info: &FlacStreamInfo,
) -> Result<(Vec<ContainerPacket>, usize, u64), PipelineError> {
    let ch = usize::from(channels);
    if ch == 0 || !pcm.len().is_multiple_of(ch) {
        return Err(encode_error(
            Path::new("flac"),
            "FLAC input is not whole interleaved frames",
        ));
    }
    let mut full_framebuf = FrameBuf::with_size(ch, FLAC_CHUNK_FRAMES).map_err(|err| {
        encode_error(
            Path::new("flac"),
            format!("failed to allocate FLAC frame buffer: {err}"),
        )
    })?;
    let mut packets = Vec::new();
    let mut frame_index = start_frame_number;
    let mut sample_offset = start_sample_offset;
    for block in pcm.chunks(FLAC_CHUNK_FRAMES.saturating_mul(ch)) {
        let frames = block.len() / ch;
        if frames == 0 {
            continue;
        }
        let encoded_frames = frames.max(FLAC_MIN_BLOCK_FRAMES);
        let mut samples = Vec::with_capacity(encoded_frames.saturating_mul(ch));
        samples.extend(block.iter().copied().map(to_s16_i32));
        if encoded_frames > frames {
            samples.resize(encoded_frames.saturating_mul(ch), 0);
        }
        let frame = if encoded_frames == FLAC_CHUNK_FRAMES {
            full_framebuf.fill_interleaved(&samples).map_err(|err| {
                encode_error(Path::new("flac"), format!("FLAC fill failed: {err}"))
            })?;
            flacenc::encode_fixed_size_frame(config, &full_framebuf, frame_index, info).map_err(
                |err| encode_error(Path::new("flac"), format!("FLAC encoding failed: {err}")),
            )?
        } else {
            let mut tail_buf = FrameBuf::with_size(ch, encoded_frames).map_err(|err| {
                encode_error(
                    Path::new("flac"),
                    format!("failed to allocate final FLAC frame buffer: {err}"),
                )
            })?;
            tail_buf.fill_interleaved(&samples).map_err(|err| {
                encode_error(Path::new("flac"), format!("FLAC fill failed: {err}"))
            })?;
            flacenc::encode_fixed_size_frame(config, &tail_buf, frame_index, info).map_err(
                |err| encode_error(Path::new("flac"), format!("FLAC encoding failed: {err}")),
            )?
        };
        let mut sink = ByteSink::new();
        BitRepr::write(&frame, &mut sink).map_err(|err| {
            encode_error(Path::new("flac"), format!("FLAC frame write failed: {err}"))
        })?;
        let advance = u64::try_from(frames).unwrap_or(u64::MAX);
        let pts = sample_pts_us(sample_offset, rate);
        let next = sample_pts_us(sample_offset.saturating_add(advance), rate);
        let duration = u64::try_from(next.saturating_sub(pts)).unwrap_or(1).max(1);
        packets.push(ContainerPacket {
            track: 0,
            pts,
            dts: pts,
            duration,
            keyframe: true,
            data: sink.into_inner(),
        });
        frame_index = frame_index.saturating_add(1);
        sample_offset = sample_offset.saturating_add(advance);
    }
    Ok((
        packets,
        frame_index.saturating_sub(start_frame_number),
        sample_offset.saturating_sub(start_sample_offset),
    ))
}

/// Whole-track FLAC encoder retained for codec round-trip tests. Production
/// transcodes use [`StreamingAudioEncoder`] instead.
#[cfg(test)]
fn encode_flac_track(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    config: AudioEncodeConfig,
) -> Result<(ContainerTrack, Vec<ContainerPacket>), PipelineError> {
    let rate = configured_audio_rate(TargetAudioCodec::Flac, sample_rate, config);
    let target_channels = configured_audio_channels(channels, config);
    let resampled = resample_interleaved(samples, usize::from(channels), sample_rate, rate)?;
    let pcm = mix_to_channels(
        &resampled,
        usize::from(channels),
        usize::from(target_channels),
    )?;
    if pcm.is_empty() {
        return Err(encode_error(path, "no samples to encode as FLAC"));
    }
    let total = pcm.len() / usize::from(target_channels.max(1));
    let encoder_config = flac_encoder_config()?;
    let info = flac_streaminfo(rate, target_channels, total)?;
    let streaminfo = flac_streaminfo_bytes(&info)?;
    parse_flac_streaminfo(&streaminfo)
        .map_err(|reason| encode_error(path, format!("invalid FLAC STREAMINFO: {reason}")))?;
    let (mut packets, _, _) =
        encode_flac_frames(&pcm, target_channels, rate, 0, 0, &encoder_config, &info)?;
    for packet in &mut packets {
        packet.track = 0;
    }
    if packets.is_empty() {
        return Err(encode_error(path, "FLAC encoder produced no frames"));
    }
    Ok((
        ContainerTrack {
            codec: VideoCodec::Flac,
            width: None,
            height: None,
            channels: Some(target_channels),
            sample_rate: Some(rate),
            config: streaminfo.to_vec(),
            timescale: US_TIMESCALE,
            presentation_edit: None,
            pcm_format: None,
        },
        packets,
    ))
}

/// Whole-track PCM encoder retained for codec round-trip tests.
#[cfg(test)]
fn encode_pcm_track(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    config: AudioEncodeConfig,
) -> Result<(ContainerTrack, Vec<ContainerPacket>), PipelineError> {
    let rate = configured_audio_rate(TargetAudioCodec::Pcm, sample_rate, config);
    let target_channels = configured_audio_channels(channels, config);
    let resampled = resample_interleaved(samples, usize::from(channels), sample_rate, rate)?;
    let pcm = mix_to_channels(
        &resampled,
        usize::from(channels),
        usize::from(target_channels),
    )?;
    if pcm.is_empty() {
        return Err(encode_error(path, "no samples to encode as PCM"));
    }
    let bytes = pcm_to_s16le(&pcm);
    let frame_bytes = usize::from(target_channels).saturating_mul(2);
    let mut packets = Vec::new();
    for (index, chunk) in bytes
        .chunks(frame_bytes.saturating_mul(PCM_CHUNK_FRAMES))
        .enumerate()
    {
        let first = index.saturating_mul(PCM_CHUNK_FRAMES);
        let frames = chunk.len() / frame_bytes;
        let pts = sample_pts_us(u64::try_from(first).unwrap_or(u64::MAX), rate);
        let next = sample_pts_us(
            u64::try_from(first.saturating_add(frames)).unwrap_or(u64::MAX),
            rate,
        );
        packets.push(ContainerPacket {
            track: 0,
            pts,
            dts: pts,
            duration: u64::try_from(next.saturating_sub(pts)).unwrap_or(1).max(1),
            keyframe: true,
            data: chunk.to_vec(),
        });
    }
    Ok((
        ContainerTrack {
            codec: VideoCodec::Pcm,
            width: None,
            height: None,
            channels: Some(target_channels),
            sample_rate: Some(rate),
            config: Vec::new(),
            timescale: US_TIMESCALE,
            presentation_edit: None,
            pcm_format: Some(PcmFormat::S16Le),
        },
        packets,
    ))
}

/// Whole-track Vorbis encoder retained for codec round-trip tests. Production
/// transcodes use [`StreamingAudioEncoder`] instead.
#[cfg(test)]
fn encode_vorbis_track(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    config: AudioEncodeConfig,
) -> Result<(ContainerTrack, Vec<ContainerPacket>), PipelineError> {
    let target_channels = configured_audio_channels(channels, config);
    let ch = usize::from(target_channels);
    let channel_order = crate::audio::vorbis_plane_sources(ch).ok_or_else(|| {
        encode_error(
            path,
            format!("Vorbis cannot safely map {target_channels} channels"),
        )
    })?;
    let rate = configured_audio_rate(TargetAudioCodec::Vorbis, sample_rate, config);
    let resampled = resample_interleaved(samples, usize::from(channels), sample_rate, rate)?;
    let pcm = mix_to_channels(&resampled, usize::from(channels), ch)?;
    if pcm.is_empty() {
        return Err(encode_error(path, "no samples to encode as Vorbis"));
    }
    let ch_nz = u8::try_from(ch)
        .ok()
        .and_then(NonZeroU8::new)
        .ok_or_else(|| encode_error(path, "Vorbis channel count must be in 1..=255"))?;
    let parts = start_vorbis_encoder(rate, ch_nz.get(), config)?;
    let mut encoder = parts.encoder;
    let mut planes: Vec<Vec<f32>> = vec![Vec::with_capacity(VORBIS_CHUNK_FRAMES); ch];
    let total_frames = pcm.len() / ch;
    for start in (0..total_frames).step_by(VORBIS_CHUNK_FRAMES) {
        let frames = VORBIS_CHUNK_FRAMES.min(total_frames.saturating_sub(start));
        for plane in &mut planes {
            plane.clear();
        }
        for frame_samples in pcm.chunks_exact(ch).skip(start).take(frames) {
            for (plane, &source) in planes.iter_mut().zip(channel_order) {
                if let Some(&value) = frame_samples.get(source) {
                    plane.push(value);
                }
            }
        }
        encoder
            .encode_audio_block(planes.iter())
            .map_err(|err| encode_error(path, format!("Vorbis encoding failed: {err}")))?;
    }
    let finished = encoder
        .finish()
        .map_err(|err| encode_error(path, format!("failed to finalize Vorbis stream: {err}")))?;
    drop(finished);
    let mut timestamps = VorbisTimestamps::new(rate, parts.bs0, parts.bs1);
    let mut packets = Vec::new();
    for packet in drain_ogg_sink(&parts.sink)? {
        packets.push(timestamps.next_packet(packet)?);
    }
    if packets.is_empty() {
        return Err(encode_error(path, "Vorbis encoder produced no packets"));
    }
    Ok((
        ContainerTrack {
            codec: VideoCodec::Vorbis,
            width: None,
            height: None,
            channels: Some(target_channels),
            sample_rate: Some(rate),
            config: parts.private,
            timescale: US_TIMESCALE,
            presentation_edit: None,
            pcm_format: None,
        },
        packets,
    ))
}

/// Encodes decoded PCM into the requested in-container audio codec.
#[cfg(test)]
pub(crate) fn encode_audio_track(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    target: TargetAudioCodec,
    config: AudioEncodeConfig,
) -> Result<(ContainerTrack, Vec<ContainerPacket>), PipelineError> {
    match target {
        TargetAudioCodec::Aac => encode_aac_track(path, samples, sample_rate, channels, config),
        TargetAudioCodec::Opus => encode_opus_track(path, samples, sample_rate, channels, config),
        TargetAudioCodec::Mp3 => encode_mp3_track(path, samples, sample_rate, channels, config),
        TargetAudioCodec::Vorbis => {
            encode_vorbis_track(path, samples, sample_rate, channels, config)
        }
        TargetAudioCodec::Flac => encode_flac_track(path, samples, sample_rate, channels, config),
        TargetAudioCodec::Pcm => encode_pcm_track(path, samples, sample_rate, channels, config),
    }
}

/// Bounded handoff from the audio worker to the muxing thread. A header always
/// precedes that track's packet batches so output track order stays fixed.
enum AudioWorkerEvent {
    TrackHeader {
        new_index: usize,
        track: ContainerTrack,
    },
    PacketBatch(Vec<ContainerPacket>),
}

fn target_audio_rate(target: TargetAudioCodec, source_rate: u32) -> u32 {
    match target {
        TargetAudioCodec::Aac => {
            if rusty_aac::sf_index_for_rate(source_rate).is_some() {
                source_rate
            } else {
                OPUS_RATE
            }
        }
        TargetAudioCodec::Opus => OPUS_RATE,
        TargetAudioCodec::Mp3 => {
            if MP3_RATES.contains(&source_rate) {
                source_rate
            } else {
                OPUS_RATE
            }
        }
        TargetAudioCodec::Vorbis => {
            if (8_000..=192_000).contains(&source_rate) {
                source_rate
            } else {
                OPUS_RATE
            }
        }
        TargetAudioCodec::Flac => {
            if (1..=655_350).contains(&source_rate) {
                source_rate
            } else {
                44_100
            }
        }
        TargetAudioCodec::Pcm => {
            if (1..=384_000).contains(&source_rate) {
                source_rate
            } else {
                48_000
            }
        }
    }
}

fn configured_audio_rate(
    target: TargetAudioCodec,
    source_rate: u32,
    config: AudioEncodeConfig,
) -> u32 {
    config.sample_rate().map_or_else(
        || target_audio_rate(target, source_rate),
        AudioSampleRate::hz,
    )
}

fn configured_audio_channels(source_channels: u16, config: AudioEncodeConfig) -> u16 {
    config
        .channels()
        .map_or(source_channels, AudioChannels::count)
}

fn configured_aac_bitrate(config: AudioEncodeConfig) -> u32 {
    config.bitrate().map_or(AAC_BITRATE_BPS, AudioBitrate::bps)
}

fn configured_opus_bitrate(config: AudioEncodeConfig) -> i32 {
    config.bitrate().map_or(OPUS_BITRATE, |bitrate| {
        i32::try_from(bitrate.bps()).unwrap_or(OPUS_BITRATE)
    })
}

fn configured_mp3_kbps(config: AudioEncodeConfig) -> u32 {
    config.bitrate().map_or(MP3_BITRATE_KBPS, |bitrate| {
        bitrate.bps().saturating_add(500) / 1_000
    })
}

/// Resolves and validates the concrete MP3 pair before encoder creation.
/// Either field can come from a worker default rather than an explicit option,
/// so the band check must run on what the encoder actually receives.
fn checked_mp3_bitrate_kbps(
    config: AudioEncodeConfig,
    rate: u32,
    path: &Path,
) -> Result<u32, PipelineError> {
    let kbps = configured_mp3_kbps(config);
    config
        .validate_effective_mp3(kbps.saturating_mul(1_000), rate)
        .map_err(|err| encode_error(path, err.to_string()))?;
    Ok(kbps)
}

/// Interleaved channel remix. `from == to` borrows the input unchanged; any
/// other ratio is expressed per frame with bounded lookups so a truncated PCM
/// chunk can never panic. Mono/stereo are the sources for multichannel AAC
/// upmixes. Every reduction to fewer channels folds the full source average
/// into each retained channel, so center, LFE and surround content survives
/// without exceeding the peak input sample.
fn mix_to_channels(pcm: &[f32], from: usize, to: usize) -> Result<Cow<'_, [f32]>, PipelineError> {
    let invalid = || {
        encode_error(
            Path::new("audio-mix"),
            format!("invalid channel remix from {from} to {to}"),
        )
    };
    if from == 0 || to == 0 {
        return Err(invalid());
    }
    if from == to {
        return Ok(Cow::Borrowed(pcm));
    }
    let frames = pcm.len() / from;
    let out_len = frames.checked_mul(to).ok_or_else(invalid)?;
    let mut out = Vec::with_capacity(out_len);
    let divisor = f32::from(u16::try_from(from).unwrap_or(u16::MAX));
    for frame in pcm.chunks_exact(from) {
        let left = frame.first().copied().unwrap_or(0.0);
        let right = frame.get(1).copied().unwrap_or(left);
        if to == 1 {
            let sum = frame.iter().fold(0.0_f32, |acc, sample| acc + *sample);
            out.push(sum / divisor);
            continue;
        }
        if from > 2 && to == 2 {
            let average = frame.iter().fold(0.0_f32, |sum, sample| sum + *sample) / divisor;
            let left = frame.first().copied().unwrap_or(0.0);
            let right = frame.get(1).copied().unwrap_or(left);
            out.push(f32::midpoint(left, average));
            out.push(f32::midpoint(right, average));
            continue;
        }
        if from > to {
            // Mono and stereo reductions are handled above. For any other
            // target layout, blend the full-source average into every retained
            // channel so discarded channels still reach the output.
            let average = frame.iter().fold(0.0_f32, |sum, sample| sum + *sample) / divisor;
            for channel in 0..to {
                let direct = frame.get(channel).copied().unwrap_or(0.0);
                out.push(f32::midpoint(direct, average));
            }
            continue;
        }
        for channel in 0..to {
            let mixed = match channel {
                0 => left,
                1 => right,
                2 => f32::midpoint(left, right),
                even if even % 2 == 0 => left * 0.5,
                _ => right * 0.5,
            };
            out.push(frame.get(channel).copied().unwrap_or(mixed));
        }
    }
    Ok(Cow::Owned(out))
}

// ---- Vorbis: streaming Ogg page -> Matroska packet bridge ------------------

/// One complete Vorbis packet extracted from Ogg pages.
#[derive(Debug)]
struct OggPacket {
    data: Vec<u8>,
    granulepos: i64,
    /// Only the last completed packet on an EOS page carries the end position.
    eos_last: bool,
}

/// Incremental RFC 3533 page parser. It keeps only the bytes of one incomplete
/// page and the bytes of one incomplete packet, never the whole Ogg stream.
#[derive(Default)]
struct OggPacketCollector {
    partial: Vec<u8>,
    continuing: Vec<u8>,
    packets: Vec<OggPacket>,
}

impl OggPacketCollector {
    /// One Ogg page is at most 255 segments of 255 bytes plus its 27-byte
    /// header and segment table (65,307 bytes); 128 KiB gives safe headroom
    /// for a partial page plus the next incoming write.
    const MAX_PARTIAL: usize = 128 * 1024;
    const MAX_PACKET: usize = 1_048_576;

    fn invalid(reason: &'static str) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, reason)
    }

    fn drain(&mut self) -> Vec<OggPacket> {
        std::mem::take(&mut self.packets)
    }

    fn push_bytes(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.partial.len().saturating_add(bytes.len()) > Self::MAX_PARTIAL {
            return Err(Self::invalid("Ogg page buffer overflow"));
        }
        self.partial.extend_from_slice(bytes);
        self.parse_pages()?;
        Ok(bytes.len())
    }

    fn parse_pages(&mut self) -> io::Result<()> {
        while self.partial.len() >= 27 {
            if &self.partial[..4] != b"OggS" {
                return Err(Self::invalid("invalid Ogg capture pattern"));
            }
            if self.partial[4] != 0 {
                return Err(Self::invalid("unsupported Ogg stream structure version"));
            }
            let nseg = usize::from(self.partial[26]);
            let table_end = 27_usize
                .checked_add(nseg)
                .ok_or_else(|| Self::invalid("Ogg segment table overflow"))?;
            if self.partial.len() < table_end {
                break;
            }
            let mut body_len = 0_usize;
            for &lacing in &self.partial[27..table_end] {
                body_len = usize::from(lacing)
                    .checked_add(body_len)
                    .ok_or_else(|| Self::invalid("Ogg page body overflow"))?;
            }
            let page_len = table_end
                .checked_add(body_len)
                .ok_or_else(|| Self::invalid("Ogg page length overflow"))?;
            if self.partial.len() < page_len {
                break;
            }
            let header_type = self.partial[5];
            let continued = header_type & 0x01 != 0;
            if continued && self.continuing.is_empty() {
                return Err(Self::invalid("unexpected Ogg continuation page"));
            }
            if !continued && !self.continuing.is_empty() {
                return Err(Self::invalid(
                    "incomplete Ogg packet before non-continuation page",
                ));
            }
            let granule_bytes: [u8; 8] = self.partial[6..14]
                .try_into()
                .map_err(|_| Self::invalid("truncated Ogg granule position"))?;
            let granulepos = i64::from_le_bytes(granule_bytes);
            let completed = self.partial[27..table_end]
                .iter()
                .filter(|&&lacing| lacing < 255)
                .count();
            let eos = header_type & 0x04 != 0;
            let mut offset = table_end;
            let mut seen = 0_usize;
            for &lacing in &self.partial[27..table_end] {
                let end = offset
                    .checked_add(usize::from(lacing))
                    .ok_or_else(|| Self::invalid("Ogg segment overflow"))?;
                let segment = self
                    .partial
                    .get(offset..end)
                    .ok_or_else(|| Self::invalid("truncated Ogg segment"))?;
                if self.continuing.len().saturating_add(segment.len()) > Self::MAX_PACKET {
                    return Err(Self::invalid("Ogg packet exceeds the allocation cap"));
                }
                self.continuing.extend_from_slice(segment);
                offset = end;
                // A lacing value below 255 terminates this packet. A value of
                // exactly 255 continues it, possibly onto the next page.
                if lacing < 255 {
                    seen = seen.saturating_add(1);
                    self.packets.push(OggPacket {
                        data: std::mem::take(&mut self.continuing),
                        granulepos,
                        eos_last: eos && seen == completed,
                    });
                }
            }
            self.partial.drain(..page_len);
        }
        Ok(())
    }
}

/// `io::Write` sink handed to `vorbis-enc`; parses pages as they are written.
struct OggPacketSink(Arc<Mutex<OggPacketCollector>>);

impl Write for OggPacketSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut collector = self
            .0
            .lock()
            .map_err(|_| io::Error::other("Ogg collector lock poisoned"))?;
        collector.push_bytes(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn drain_ogg_sink(sink: &Arc<Mutex<OggPacketCollector>>) -> Result<Vec<OggPacket>, PipelineError> {
    let mut collector = sink
        .lock()
        .map_err(|_| encode_error(Path::new("vorbis"), "Ogg collector lock poisoned"))?;
    Ok(collector.drain())
}

/// State produced once the Vorbis headers and encoder are ready.
struct VorbisEncoderParts {
    encoder: vorbis_enc::Encoder<OggPacketSink>,
    sink: Arc<Mutex<OggPacketCollector>>,
    private: Vec<u8>,
    bs0: u64,
    bs1: u64,
}

/// Builds a `vorbis-enc` encoder over a bounded Ogg sink and extracts the
/// three Xiph-laced header packets into a Matroska `CodecPrivate` blob.
fn start_vorbis_encoder(
    rate: u32,
    channels: u8,
    config: AudioEncodeConfig,
) -> Result<VorbisEncoderParts, PipelineError> {
    let rate_nz = NonZeroU32::new(rate)
        .ok_or_else(|| encode_error(Path::new("vorbis"), "Vorbis rate must be non-zero"))?;
    let ch_nz = NonZeroU8::new(channels)
        .ok_or_else(|| encode_error(Path::new("vorbis"), "Vorbis channels must be non-zero"))?;
    let strategy = match config.bitrate() {
        None => vorbis_enc::BitrateManagementStrategy::QualityVbr {
            target_quality: 0.4,
        },
        Some(bitrate) => {
            let average = NonZeroU32::new(bitrate.bps()).ok_or_else(|| {
                encode_error(
                    Path::new("vorbis"),
                    "Vorbis average bitrate must be non-zero",
                )
            })?;
            vorbis_enc::BitrateManagementStrategy::Abr {
                average_bitrate: average,
            }
        }
    };
    let sink = Arc::new(Mutex::new(OggPacketCollector::default()));
    let encoder = vorbis_enc::EncoderBuilder::new(rate_nz, ch_nz, OggPacketSink(Arc::clone(&sink)))
        .bitrate_management_strategy(strategy)
        .build()
        .map_err(|err| {
            encode_error(
                Path::new("vorbis"),
                format!("failed to create Vorbis encoder: {err}"),
            )
        })?;
    let headers = drain_ogg_sink(&sink)?;
    if headers.len() < 3 {
        return Err(encode_error(
            Path::new("vorbis"),
            format!(
                "Vorbis encoder emitted {} header packets, expected 3",
                headers.len()
            ),
        ));
    }
    let private =
        pack_xiph_laced_vorbis_headers(&headers[0].data, &headers[1].data, &headers[2].data)
            .map_err(|reason| {
                encode_error(
                    Path::new("vorbis"),
                    format!("invalid Vorbis header: {reason}"),
                )
            })?;
    if private.len() > VORBIS_CONFIG_CAP {
        return Err(encode_error(
            Path::new("vorbis"),
            "Vorbis CodecPrivate exceeds the allocation cap",
        ));
    }
    let (ident_channels, ident_rate, bs0, bs1) = vorbis_ident_info(&headers[0].data)
        .map_err(|reason| encode_error(Path::new("vorbis"), reason))?;
    if ident_channels != u16::from(channels) || ident_rate != rate {
        return Err(encode_error(
            Path::new("vorbis"),
            format!(
                "Vorbis ident header disagrees with the encoder: {ident_channels}ch/{ident_rate} Hz vs {channels}ch/{rate} Hz"
            ),
        ));
    }
    Ok(VorbisEncoderParts {
        encoder,
        sink,
        private,
        bs0,
        bs1,
    })
}

/// Converts extracted Ogg packets into microsecond-timestamped Matroska
/// packets. `bs0`/`bs1` are Vorbis blocksize exponents (Vorbis I §4.2.2), so
/// the sample advance of a block pair is `(2^prev + 2^cur) / 4`.
struct VorbisTimestamps {
    rate: u32,
    bs0: u64,
    bs1: u64,
    sample_offset: u64,
    prev_bs: Option<u64>,
    last_pts: i64,
}

impl VorbisTimestamps {
    fn new(rate: u32, bs0: u64, bs1: u64) -> Self {
        Self {
            rate,
            bs0,
            bs1,
            sample_offset: 0,
            prev_bs: None,
            last_pts: 0,
        }
    }

    fn next_packet(&mut self, packet: OggPacket) -> Result<ContainerPacket, PipelineError> {
        let path = Path::new("vorbis");
        if packet.data.is_empty() || packet.data[0] & 1 != 0 {
            return Err(encode_error(path, "invalid Vorbis audio packet header"));
        }
        let cur_bs = if self.bs0 == self.bs1 || (packet.data[0] & 0x02) == 0 {
            self.bs0
        } else {
            self.bs1
        };
        let (pts, duration) = match self.prev_bs {
            None => {
                // The first audio block is the priming packet; it plays no
                // samples and sits at time zero.
                (0_i64, 1_u64)
            }
            Some(prev) => {
                let prev_size = 1_u64
                    .checked_shl(u32::try_from(prev).unwrap_or(32))
                    .unwrap_or(u64::MAX);
                let cur_size = 1_u64
                    .checked_shl(u32::try_from(cur_bs).unwrap_or(32))
                    .unwrap_or(u64::MAX);
                let raw_samples = prev_size.saturating_add(cur_size) / 4;
                let effective = if packet.eos_last && packet.granulepos >= 0 {
                    let granule = u64::try_from(packet.granulepos).unwrap_or(u64::MAX);
                    granule.saturating_sub(self.sample_offset).min(raw_samples)
                } else {
                    raw_samples
                };
                let pts = sample_pts_us(self.sample_offset, self.rate);
                let next_pts =
                    sample_pts_us(self.sample_offset.saturating_add(effective), self.rate);
                let duration = u64::try_from(next_pts.saturating_sub(pts))
                    .unwrap_or(1)
                    .max(1);
                self.sample_offset = self.sample_offset.saturating_add(effective);
                (pts, duration)
            }
        };
        if pts < self.last_pts {
            return Err(encode_error(
                path,
                format!(
                    "Vorbis timestamps went backwards ({pts} < {})",
                    self.last_pts
                ),
            ));
        }
        self.last_pts = pts;
        self.prev_bs = Some(cur_bs);
        Ok(ContainerPacket {
            track: 0,
            pts,
            dts: pts,
            duration,
            keyframe: true,
            data: packet.data,
        })
    }
}

/// Incremental encoder for the four in-container audio codecs. It is created
/// on the first decoded chunk, so its rate and channel count come from real
/// PCM rather than guessed container metadata.
enum StreamingAudioEncoder {
    Aac {
        rate: u32,
        channels: u16,
        encoder: rusty_aac::AacEncoder,
        sample_offset: u64,
    },
    Opus {
        channels: u16,
        encoder: Box<OpusEncoder>,
        pre_skip: u16,
        frame_samples: usize,
        leftover: Vec<f32>,
        held: Option<ContainerPacket>,
        frames_emitted: usize,
        input_frames: usize,
    },
    Mp3 {
        rate: u32,
        channels: u16,
        encoder: Box<rusty_mp3::Mp3Encoder>,
        sample_offset: u64,
        frame_samples: u64,
        // The first packet is the LAME Info tag; its counts are only final
        // after `finish()`, so hold it back until the tag is completed.
        held_tag: Option<ContainerPacket>,
    },
    Vorbis {
        channels: u8,
        encoder: Option<Box<vorbis_enc::Encoder<OggPacketSink>>>,
        sink: Arc<Mutex<OggPacketCollector>>,
        planes: Vec<Vec<f32>>,
        channel_order: &'static [usize],
        timestamps: VorbisTimestamps,
    },
    Flac {
        channels: u16,
        rate: u32,
        config: flacenc::error::Verified<flacenc::config::Encoder>,
        info: FlacStreamInfo,
        frame_number: usize,
        sample_offset: u64,
        pending: Vec<f32>,
    },
    Pcm {
        channels: u16,
        rate: u32,
        sample_offset: u64,
        pending: Vec<u8>,
    },
}

impl StreamingAudioEncoder {
    // One branch per target codec keeps validation and header construction explicit.
    #[allow(clippy::too_many_lines)]
    fn start(
        target: TargetAudioCodec,
        channels: u16,
        rate: u32,
        config: AudioEncodeConfig,
    ) -> Result<(Self, ContainerTrack), PipelineError> {
        match target {
            TargetAudioCodec::Aac => {
                if !(1..=6).contains(&channels) || channels == 4 {
                    return Err(encode_error(
                        Path::new("aac"),
                        format!("AAC output supports 1-3 and 5-6 channels, got {channels}"),
                    ));
                }
                let asc = aac_config(rate, channels)?.to_vec();
                let mut encoder = rusty_aac::AacEncoder::new(rusty_aac::AacEncoderConfig {
                    bitrate_bps: configured_aac_bitrate(config),
                    ..rusty_aac::AacEncoderConfig::default()
                });
                // Without this the encoder buffers the whole track and emits no
                // packet until `finish()`, defeating the bounded audio queue.
                // Must be set before the first `push_pcm`.
                encoder.set_streaming(true);
                let track = ContainerTrack {
                    codec: VideoCodec::Aac,
                    width: None,
                    height: None,
                    channels: Some(channels),
                    sample_rate: Some(rate),
                    config: asc,
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                    pcm_format: None,
                };
                Ok((
                    Self::Aac {
                        rate,
                        channels,
                        encoder,
                        sample_offset: 0,
                    },
                    track,
                ))
            }
            TargetAudioCodec::Opus => {
                if channels == 0 || channels > 2 {
                    return Err(encode_error(
                        Path::new("opus"),
                        format!("Opus output supports mono/stereo only, got {channels}"),
                    ));
                }
                let mut opus_encoder = OpusEncoder::new(
                    i32::try_from(OPUS_RATE).unwrap_or(i32::MAX),
                    usize::from(channels),
                    Application::Audio,
                )
                .map_err(|err| {
                    encode_error(
                        Path::new("opus"),
                        format!("failed to create Opus encoder: {err}"),
                    )
                })?;
                opus_encoder.bitrate_bps = configured_opus_bitrate(config);
                opus_encoder.complexity = 9;
                let head = opus_pure::OpusHead::for_encoder(&opus_encoder, OPUS_RATE);
                let pre_skip = head.pre_skip;
                let track = ContainerTrack {
                    codec: VideoCodec::Opus,
                    width: None,
                    height: None,
                    channels: Some(channels),
                    sample_rate: Some(OPUS_RATE),
                    config: default_opus_head(channels, pre_skip),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                    pcm_format: None,
                };
                Ok((
                    Self::Opus {
                        channels,
                        encoder: Box::new(opus_encoder),
                        pre_skip,
                        frame_samples: OPUS_FRAME.saturating_mul(usize::from(channels)),
                        leftover: Vec::new(),
                        held: None,
                        frames_emitted: 0,
                        input_frames: 0,
                    },
                    track,
                ))
            }
            TargetAudioCodec::Mp3 => {
                if channels == 0 || channels > 2 {
                    return Err(encode_error(
                        Path::new("mp3"),
                        format!("MP3 output supports mono/stereo only, got {channels}"),
                    ));
                }
                let mp3_encoder =
                    Box::new(rusty_mp3::Mp3Encoder::new(rusty_mp3::Mp3EncoderConfig {
                        bitrate_kbps: checked_mp3_bitrate_kbps(config, rate, Path::new("mp3"))?,
                        vbr_quality: None,
                    }));
                let track = ContainerTrack {
                    codec: VideoCodec::Mp3,
                    width: None,
                    height: None,
                    channels: Some(channels),
                    sample_rate: Some(rate),
                    config: Vec::new(),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                    pcm_format: None,
                };
                Ok((
                    Self::Mp3 {
                        rate,
                        channels,
                        encoder: mp3_encoder,
                        sample_offset: 0,
                        frame_samples: if rate >= 32_000 { 1_152 } else { 576 },
                        held_tag: None,
                    },
                    track,
                ))
            }
            TargetAudioCodec::Vorbis => {
                let channel_order = crate::audio::vorbis_plane_sources(usize::from(channels))
                    .ok_or_else(|| {
                        encode_error(
                            Path::new("vorbis"),
                            format!("Vorbis cannot safely map {channels} channels"),
                        )
                    })?;
                let ch = u8::try_from(channels).map_err(|_| {
                    encode_error(
                        Path::new("vorbis"),
                        format!("Vorbis channel count {channels} exceeds 255"),
                    )
                })?;
                let parts = start_vorbis_encoder(rate, ch, config)?;
                let track = ContainerTrack {
                    codec: VideoCodec::Vorbis,
                    width: None,
                    height: None,
                    channels: Some(channels),
                    sample_rate: Some(rate),
                    config: parts.private,
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                    pcm_format: None,
                };
                Ok((
                    Self::Vorbis {
                        channels: ch,
                        encoder: Some(Box::new(parts.encoder)),
                        sink: parts.sink,
                        planes: vec![Vec::with_capacity(VORBIS_CHUNK_FRAMES); usize::from(ch)],
                        channel_order,
                        timestamps: VorbisTimestamps::new(rate, parts.bs0, parts.bs1),
                    },
                    track,
                ))
            }
            TargetAudioCodec::Flac => {
                let encoder_config = flac_encoder_config()?;
                let info = flac_streaminfo(rate, channels, 0)?;
                let streaminfo = flac_streaminfo_bytes(&info)?;
                parse_flac_streaminfo(&streaminfo).map_err(|reason| {
                    encode_error(Path::new("flac"), format!("invalid STREAMINFO: {reason}"))
                })?;
                let track = ContainerTrack {
                    codec: VideoCodec::Flac,
                    width: None,
                    height: None,
                    channels: Some(channels),
                    sample_rate: Some(rate),
                    config: streaminfo.to_vec(),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                    pcm_format: None,
                };
                Ok((
                    Self::Flac {
                        channels,
                        rate,
                        config: encoder_config,
                        info,
                        frame_number: 0,
                        sample_offset: 0,
                        pending: Vec::with_capacity(
                            FLAC_CHUNK_FRAMES.saturating_mul(usize::from(channels)),
                        ),
                    },
                    track,
                ))
            }
            TargetAudioCodec::Pcm => {
                let track = ContainerTrack {
                    codec: VideoCodec::Pcm,
                    width: None,
                    height: None,
                    channels: Some(channels),
                    sample_rate: Some(rate),
                    config: Vec::new(),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                    pcm_format: Some(PcmFormat::S16Le),
                };
                Ok((
                    Self::Pcm {
                        channels,
                        rate,
                        sample_offset: 0,
                        pending: Vec::new(),
                    },
                    track,
                ))
            }
        }
    }

    /// Encodes one resampled chunk and drains every packet that is ready now.
    // One branch per codec keeps timestamp and drain handling explicit.
    #[allow(clippy::too_many_lines)]
    fn push(&mut self, pcm: &[f32]) -> Result<Vec<ContainerPacket>, PipelineError> {
        let mut out = Vec::new();
        match self {
            Self::Aac {
                rate,
                channels,
                encoder,
                sample_offset,
            } => {
                encoder.push_pcm(pcm, *channels, *rate).map_err(|err| {
                    encode_error(Path::new("aac"), format!("AAC encoding failed: {err}"))
                })?;
                let frame_samples = u64::try_from(AAC_FRAME).unwrap_or(u64::MAX);
                loop {
                    match encoder.next_packet() {
                        Ok(packet) => {
                            if packet.data.is_empty() {
                                continue;
                            }
                            let pts = sample_pts_us(*sample_offset, *rate);
                            let next_pts =
                                sample_pts_us(sample_offset.saturating_add(frame_samples), *rate);
                            let duration = u64::try_from(next_pts.saturating_sub(pts))
                                .unwrap_or(1)
                                .max(1);
                            *sample_offset = sample_offset.saturating_add(frame_samples);
                            out.push(ContainerPacket {
                                track: 0,
                                pts,
                                dts: pts,
                                duration,
                                keyframe: true,
                                data: packet.data,
                            });
                        }
                        Err(rusty_aac::Error::Eof | rusty_aac::Error::Again) => break,
                        Err(err) => {
                            return Err(encode_error(
                                Path::new("aac"),
                                format!("AAC drain failed: {err}"),
                            ));
                        }
                    }
                }
            }
            Self::Mp3 {
                rate,
                channels,
                encoder,
                sample_offset,
                frame_samples,
                held_tag,
            } => {
                encoder.push_pcm_f32(pcm, *channels, *rate).map_err(|err| {
                    encode_error(Path::new("mp3"), format!("MP3 encoding failed: {err}"))
                })?;
                loop {
                    match encoder.next_packet() {
                        Ok(data) => {
                            let pts = sample_pts_us(*sample_offset, *rate);
                            let next_pts =
                                sample_pts_us(sample_offset.saturating_add(*frame_samples), *rate);
                            let duration = u64::try_from(next_pts.saturating_sub(pts))
                                .unwrap_or(1)
                                .max(1);
                            let first = *sample_offset == 0 && held_tag.is_none();
                            *sample_offset = sample_offset.saturating_add(*frame_samples);
                            let packet = ContainerPacket {
                                track: 0,
                                pts,
                                dts: pts,
                                duration,
                                keyframe: true,
                                data,
                            };
                            if first {
                                *held_tag = Some(packet);
                            } else {
                                out.push(packet);
                            }
                        }
                        Err(rusty_mp3::Error::Eof | rusty_mp3::Error::Again) => break,
                        Err(err) => {
                            return Err(encode_error(
                                Path::new("mp3"),
                                format!("MP3 drain failed: {err}"),
                            ));
                        }
                    }
                }
            }
            Self::Opus {
                channels,
                encoder,
                frame_samples,
                leftover,
                held,
                frames_emitted,
                input_frames,
                ..
            } => {
                *input_frames = input_frames.saturating_add(pcm.len() / usize::from(*channels));
                leftover.extend_from_slice(pcm);
                while leftover.len() >= *frame_samples {
                    let tail = leftover.split_off(*frame_samples);
                    let block = std::mem::replace(leftover, tail);
                    let mut packet = vec![0_u8; MAX_PACKET_BYTES];
                    let count = encoder
                        .encode(&block, OPUS_FRAME, &mut packet)
                        .map_err(|err| {
                            encode_error(Path::new("opus"), format!("Opus encoding failed: {err}"))
                        })?;
                    if count == 0 {
                        return Err(encode_error(
                            Path::new("opus"),
                            "Opus encoder produced an empty packet",
                        ));
                    }
                    let sample_offset = u64::try_from(*frames_emitted)
                        .unwrap_or(u64::MAX)
                        .saturating_mul(u64::try_from(OPUS_FRAME).unwrap_or(u64::MAX));
                    let pts = sample_pts_us(sample_offset, OPUS_RATE);
                    if let Some(mut previous) = held.replace(ContainerPacket {
                        track: 0,
                        pts,
                        dts: pts,
                        duration: 20_000,
                        keyframe: true,
                        data: packet[..count].to_vec(),
                    }) {
                        previous.duration = 20_000;
                        out.push(previous);
                    }
                    *frames_emitted = frames_emitted.saturating_add(1);
                }
            }
            Self::Vorbis {
                channels,
                encoder,
                sink,
                planes,
                channel_order,
                timestamps,
                ..
            } => {
                let ch = usize::from(*channels);
                if pcm.len() / ch == 0 {
                    return Ok(out);
                }
                let encoder = encoder.as_mut().ok_or_else(|| {
                    encode_error(Path::new("vorbis"), "Vorbis encoder was already finished")
                })?;
                // A resampled+remixed push can exceed one encoded block, so feed
                // bounded blocks instead of rejecting the whole chunk: this keeps
                // the reusable planes within their reserved capacity.
                let block_samples = VORBIS_CHUNK_FRAMES.saturating_mul(ch);
                for block in pcm.chunks(block_samples) {
                    if block.len() < ch {
                        continue;
                    }
                    for plane in planes.iter_mut() {
                        plane.clear();
                    }
                    for frame_samples in block.chunks_exact(ch) {
                        for (plane, &source) in planes.iter_mut().zip(*channel_order) {
                            if let Some(&value) = frame_samples.get(source) {
                                plane.push(value);
                            }
                        }
                    }
                    encoder.encode_audio_block(planes.iter()).map_err(|err| {
                        encode_error(
                            Path::new("vorbis"),
                            format!("Vorbis encoding failed: {err}"),
                        )
                    })?;
                }
                for packet in drain_ogg_sink(sink)? {
                    out.push(timestamps.next_packet(packet)?);
                }
            }
            Self::Flac {
                channels,
                rate,
                config,
                info,
                frame_number,
                sample_offset,
                pending,
            } => {
                let ch = usize::from(*channels);
                if ch == 0 || !pcm.len().is_multiple_of(ch) {
                    return Err(encode_error(
                        Path::new("flac"),
                        "FLAC input is not whole interleaved frames",
                    ));
                }
                pending.extend_from_slice(pcm);
                let block_samples = FLAC_CHUNK_FRAMES.saturating_mul(ch);
                while pending.len() >= block_samples {
                    let block: Vec<f32> = pending.drain(..block_samples).collect();
                    let (mut packets, emitted_frames, emitted_samples) = encode_flac_frames(
                        &block,
                        *channels,
                        *rate,
                        *frame_number,
                        *sample_offset,
                        config,
                        info,
                    )?;
                    *frame_number = frame_number.saturating_add(emitted_frames);
                    *sample_offset = sample_offset.saturating_add(emitted_samples);
                    out.append(&mut packets);
                }
            }
            Self::Pcm {
                channels,
                rate,
                sample_offset,
                pending,
            } => {
                let frame_bytes = usize::from(*channels).saturating_mul(2);
                pending.extend_from_slice(&pcm_to_s16le(pcm));
                let block_bytes = frame_bytes.saturating_mul(PCM_CHUNK_FRAMES);
                while pending.len() >= block_bytes {
                    let chunk: Vec<u8> = pending.drain(..block_bytes).collect();
                    let frames = u64::try_from(PCM_CHUNK_FRAMES).unwrap_or(u64::MAX);
                    let pts = sample_pts_us(*sample_offset, *rate);
                    let next = sample_pts_us(sample_offset.saturating_add(frames), *rate);
                    *sample_offset = sample_offset.saturating_add(frames);
                    out.push(ContainerPacket {
                        track: 0,
                        pts,
                        dts: pts,
                        duration: u64::try_from(next.saturating_sub(pts)).unwrap_or(1).max(1),
                        keyframe: true,
                        data: chunk,
                    });
                }
            }
        }
        Ok(out)
    }

    /// Flushes the tail and emits the final Opus packet with the exact
    /// remaining unplayed duration, mirroring `encode_opus_track`.
    // One branch per codec keeps the flush and tail-duration math explicit.
    #[allow(clippy::too_many_lines)]
    fn finish(&mut self) -> Result<Vec<ContainerPacket>, PipelineError> {
        match self {
            Self::Aac {
                rate,
                encoder,
                sample_offset,
                ..
            } => {
                encoder.finish();
                let frame_samples = u64::try_from(AAC_FRAME).unwrap_or(u64::MAX);
                let mut out = Vec::new();
                loop {
                    match encoder.next_packet() {
                        Ok(packet) => {
                            if packet.data.is_empty() {
                                continue;
                            }
                            let pts = sample_pts_us(*sample_offset, *rate);
                            let next_pts =
                                sample_pts_us(sample_offset.saturating_add(frame_samples), *rate);
                            let duration = u64::try_from(next_pts.saturating_sub(pts))
                                .unwrap_or(1)
                                .max(1);
                            *sample_offset = sample_offset.saturating_add(frame_samples);
                            out.push(ContainerPacket {
                                track: 0,
                                pts,
                                dts: pts,
                                duration,
                                keyframe: true,
                                data: packet.data,
                            });
                        }
                        Err(rusty_aac::Error::Eof | rusty_aac::Error::Again) => break,
                        Err(err) => {
                            return Err(encode_error(
                                Path::new("aac"),
                                format!("AAC drain failed: {err}"),
                            ));
                        }
                    }
                }
                Ok(out)
            }
            Self::Mp3 {
                rate,
                encoder,
                sample_offset,
                frame_samples,
                held_tag,
                ..
            } => {
                encoder.finish();
                let mut out = Vec::new();
                // `finish()` finalizes the Info tag's frame and byte counts.
                // Replace the held placeholder before the muxer sees it.
                if let (Some(tag), Some(mut held)) = (encoder.lame_tag_frame(), held_tag.take()) {
                    held.data = tag.to_vec();
                    out.push(held);
                }
                loop {
                    match encoder.next_packet() {
                        Ok(data) => {
                            let pts = sample_pts_us(*sample_offset, *rate);
                            let next_pts =
                                sample_pts_us(sample_offset.saturating_add(*frame_samples), *rate);
                            let duration = u64::try_from(next_pts.saturating_sub(pts))
                                .unwrap_or(1)
                                .max(1);
                            *sample_offset = sample_offset.saturating_add(*frame_samples);
                            out.push(ContainerPacket {
                                track: 0,
                                pts,
                                dts: pts,
                                duration,
                                keyframe: true,
                                data,
                            });
                        }
                        Err(rusty_mp3::Error::Eof | rusty_mp3::Error::Again) => break,
                        Err(err) => {
                            return Err(encode_error(
                                Path::new("mp3"),
                                format!("MP3 drain failed: {err}"),
                            ));
                        }
                    }
                }
                Ok(out)
            }
            Self::Opus {
                encoder,
                pre_skip,
                frame_samples,
                leftover,
                held,
                frames_emitted,
                input_frames,
                ..
            } => {
                let mut out = Vec::new();
                let total_frames = input_frames.saturating_add(usize::from(*pre_skip));
                let required = total_frames.div_ceil(OPUS_FRAME).max(1);
                while *frames_emitted < required {
                    let mut block = vec![0.0_f32; *frame_samples];
                    if !leftover.is_empty() {
                        let count = leftover.len().min(*frame_samples);
                        block[..count].copy_from_slice(&leftover[..count]);
                        leftover.drain(..count);
                    }
                    let mut packet = vec![0_u8; MAX_PACKET_BYTES];
                    let count = encoder
                        .encode(&block, OPUS_FRAME, &mut packet)
                        .map_err(|err| {
                            encode_error(Path::new("opus"), format!("Opus encoding failed: {err}"))
                        })?;
                    if count == 0 {
                        return Err(encode_error(
                            Path::new("opus"),
                            "Opus encoder produced an empty packet",
                        ));
                    }
                    let sample_offset = u64::try_from(*frames_emitted)
                        .unwrap_or(u64::MAX)
                        .saturating_mul(u64::try_from(OPUS_FRAME).unwrap_or(u64::MAX));
                    let pts = sample_pts_us(sample_offset, OPUS_RATE);
                    if let Some(mut previous) = held.replace(ContainerPacket {
                        track: 0,
                        pts,
                        dts: pts,
                        duration: 20_000,
                        keyframe: true,
                        data: packet[..count].to_vec(),
                    }) {
                        previous.duration = 20_000;
                        out.push(previous);
                    }
                    *frames_emitted = frames_emitted.saturating_add(1);
                }
                if let Some(mut last) = held.take() {
                    let final_frames = total_frames
                        .saturating_sub(frames_emitted.saturating_sub(1).saturating_mul(OPUS_FRAME))
                        .clamp(1, OPUS_FRAME);
                    last.duration = u64::try_from(final_frames)
                        .unwrap_or(u64::MAX)
                        .saturating_mul(1_000_000)
                        / u64::from(OPUS_RATE);
                    out.push(last);
                }
                Ok(out)
            }
            Self::Vorbis {
                encoder,
                sink,
                timestamps,
                ..
            } => {
                let encoder = encoder.take().ok_or_else(|| {
                    encode_error(Path::new("vorbis"), "Vorbis encoder was already finished")
                })?;
                let finished = (*encoder).finish().map_err(|err| {
                    encode_error(
                        Path::new("vorbis"),
                        format!("failed to finalize Vorbis stream: {err}"),
                    )
                })?;
                // `finish()` flushes the final EOS page into the sink.
                drop(finished);
                let mut out = Vec::new();
                for packet in drain_ogg_sink(sink)? {
                    out.push(timestamps.next_packet(packet)?);
                }
                Ok(out)
            }
            Self::Flac {
                channels,
                rate,
                config,
                info,
                frame_number,
                sample_offset,
                pending,
            } => {
                if pending.is_empty() {
                    return Ok(Vec::new());
                }
                let (mut out, emitted_frames, emitted_samples) = encode_flac_frames(
                    pending,
                    *channels,
                    *rate,
                    *frame_number,
                    *sample_offset,
                    config,
                    info,
                )?;
                pending.clear();
                *frame_number = frame_number.saturating_add(emitted_frames);
                *sample_offset = sample_offset.saturating_add(emitted_samples);
                for packet in &mut out {
                    packet.track = 0;
                }
                Ok(out)
            }
            Self::Pcm {
                channels,
                rate,
                sample_offset,
                pending,
            } => {
                if pending.is_empty() {
                    return Ok(Vec::new());
                }
                let frame_bytes = usize::from(*channels).saturating_mul(2);
                if frame_bytes == 0 || pending.len() % frame_bytes != 0 {
                    return Err(encode_error(
                        Path::new("pcm"),
                        "PCM tail is not a whole number of frames",
                    ));
                }
                let frames = u64::try_from(pending.len() / frame_bytes).unwrap_or(u64::MAX);
                let pts = sample_pts_us(*sample_offset, *rate);
                let next = sample_pts_us(sample_offset.saturating_add(frames), *rate);
                *sample_offset = sample_offset.saturating_add(frames);
                let data = std::mem::take(pending);
                Ok(vec![ContainerPacket {
                    track: 0,
                    pts,
                    dts: pts,
                    duration: u64::try_from(next.saturating_sub(pts)).unwrap_or(1).max(1),
                    keyframe: true,
                    data,
                }])
            }
        }
    }
}

/// Per-track handoff state shared by the worker and the muxing thread.
struct AudioStreamContext<'a> {
    new_index: usize,
    rebase: Option<(i64, i64)>,
    cancelled: &'a AtomicBool,
    audio_tx: &'a SyncSender<AudioWorkerEvent>,
}

/// Full streaming audio path: decode in PCM chunks, resample incrementally,
/// encode into batches and hand them to the muxing thread with backpressure.
// Decoder, resampler, channel mixer and encoder state are deliberately owned
// by one callback: splitting them would hide the bounded queue lifecycle and
// the exact point where a cancelled consumer stops the producer.
#[allow(clippy::too_many_lines)]
fn transcode_audio_track_streaming(
    path: &Path,
    source_track: &ContainerTrack,
    track_packets: &[ContainerPacket],
    target: TargetAudioCodec,
    config: AudioEncodeConfig,
    ctx: &AudioStreamContext<'_>,
) -> Result<(), PipelineError> {
    let mut encoder: Option<StreamingAudioEncoder> = None;
    let mut resampler: Option<StreamingResampler> = None;
    let mut target_channels: Option<u16> = None;
    let mut source_channels: Option<usize> = None;
    let mut batch = Vec::with_capacity(AUDIO_PACKET_BATCH_SIZE);
    let cancelled = ctx.cancelled;

    let queue_batch = |batch: &mut Vec<ContainerPacket>| {
        if batch.is_empty() || cancelled.load(Ordering::Acquire) {
            return;
        }
        let mut ready = std::mem::replace(batch, Vec::with_capacity(AUDIO_PACKET_BATCH_SIZE));
        for packet in &mut ready {
            packet.track = ctx.new_index;
        }
        if let Some((track_origin, origin)) = ctx.rebase {
            rebase_encoded_packets(&mut ready, track_origin, origin);
        }
        if ctx
            .audio_tx
            .send(AudioWorkerEvent::PacketBatch(ready))
            .is_err()
        {
            // The consumer stopped; unblock the decoder loop cleanly.
            cancelled.store(true, Ordering::Release);
        }
    };

    decode_audio_track_chunks(
        path,
        source_track,
        track_packets,
        cancelled,
        |pcm, rate, channels| {
            if cancelled.load(Ordering::Acquire) {
                return Ok(());
            }
            if encoder.is_none() {
                if channels == 0 || rate == 0 {
                    return Err(encode_error(
                        path,
                        "audio track has no valid rate or channels",
                    ));
                }
                let to_rate = configured_audio_rate(target, rate, config);
                let to_channels = configured_audio_channels(channels, config);
                target_channels = Some(to_channels);
                source_channels = Some(usize::from(channels));
                let (built, mut track) =
                    StreamingAudioEncoder::start(target, to_channels, to_rate, config)?;
                track.presentation_edit = source_track.presentation_edit.map(|edit| {
                    if let Some((_, origin)) = ctx.rebase {
                        edit.shifted(origin.saturating_neg())
                    } else {
                        edit
                    }
                });
                if ctx
                    .audio_tx
                    .send(AudioWorkerEvent::TrackHeader {
                        new_index: ctx.new_index,
                        track,
                    })
                    .is_err()
                {
                    cancelled.store(true, Ordering::Release);
                    return Ok(());
                }
                resampler = Some(StreamingResampler::new(
                    usize::from(channels),
                    rate,
                    to_rate,
                )?);
                encoder = Some(built);
            } else if let Some(expected) = source_channels
                && usize::from(channels) != expected
            {
                return Err(encode_error(
                    path,
                    format!("audio channel count changed mid-stream from {expected} to {channels}"),
                ));
            }
            let resampler = resampler
                .as_mut()
                .ok_or_else(|| encode_error(path, "resampler was not initialized"))?;
            let encoded_chunk = resampler.push(pcm)?;
            if encoded_chunk.is_empty() {
                return Ok(());
            }
            let encoder = encoder
                .as_mut()
                .ok_or_else(|| encode_error(path, "audio encoder was not initialized"))?;
            let to_channels = target_channels
                .ok_or_else(|| encode_error(path, "audio channel layout was not initialized"))?;
            let remixed = mix_to_channels(
                &encoded_chunk,
                usize::from(channels),
                usize::from(to_channels),
            )?;
            batch.extend(encoder.push(&remixed)?);
            if batch.len() >= AUDIO_PACKET_BATCH_SIZE {
                queue_batch(&mut batch);
            }
            Ok(())
        },
    )?;

    if cancelled.load(Ordering::Acquire) {
        return Ok(());
    }
    let encoder = encoder
        .as_mut()
        .ok_or_else(|| encode_error(path, "audio track decoded to no audio"))?;
    if let Some(resampler) = resampler.as_mut() {
        let tail = resampler.finish()?;
        if !tail.is_empty() {
            let to_channels = target_channels
                .ok_or_else(|| encode_error(path, "audio channel layout was not initialized"))?;
            let from_channels = source_channels
                .ok_or_else(|| encode_error(path, "audio input layout was not initialized"))?;
            let remixed = mix_to_channels(&tail, from_channels, usize::from(to_channels))?;
            batch.extend(encoder.push(&remixed)?);
        }
    }
    batch.extend(encoder.finish()?);
    queue_batch(&mut batch);
    Ok(())
}

fn input_size(path: &Path) -> u64 {
    fs::metadata(path).map_or(0, |meta| meta.len())
}

pub(crate) fn enforce_decoded_budget(path: &Path, bytes: u64) -> Result<(), PipelineError> {
    if bytes > MAX_DECODED_BYTES_PER_TRACK {
        return Err(decode_error(
            path,
            format!(
                "decoded track needs {bytes} bytes, exceeding the {MAX_DECODED_BYTES_PER_TRACK}-byte per-track budget"
            ),
        ));
    }
    Ok(())
}

/// Checks the live PCM working set before a decoded slice is appended: the
/// `f32` samples still retained in `chunk` plus the incoming slice's samples.
/// Samples already handed to the caller were drained from `chunk`, so source
/// duration never contributes to the cap.
fn enforce_live_pcm_budget(
    path: &Path,
    chunk: &[f32],
    incoming_samples: u64,
) -> Result<(), PipelineError> {
    let retained = u64::try_from(chunk.len()).unwrap_or(u64::MAX);
    enforce_decoded_budget(
        path,
        retained.saturating_add(incoming_samples).saturating_mul(4),
    )
}

fn estimated_decoded_bytes(track: &ContainerTrack, packet_count: usize) -> u64 {
    let width = u64::from(track.width.unwrap_or(0));
    let height = u64::from(track.height.unwrap_or(0));
    let bytes_per_frame = width.saturating_mul(height).saturating_mul(3) / 2_u64;
    bytes_per_frame.saturating_mul(u64::try_from(packet_count).unwrap_or(u64::MAX))
}

fn decoded_bytes(frames: &[Yuv420Frame]) -> u64 {
    frames.iter().fold(0_u64, |total, frame| {
        total
            .saturating_add(u64::try_from(frame.y.len()).unwrap_or(u64::MAX))
            .saturating_add(u64::try_from(frame.u.len()).unwrap_or(u64::MAX))
            .saturating_add(u64::try_from(frame.v.len()).unwrap_or(u64::MAX))
    })
}

/// Moves packets that already carry source timestamps onto the one file-level
/// origin shared by copied, encoded and NVENC tracks.
fn rebase_source_packets(packets: &mut [ContainerPacket], origin: i64) {
    let offset = origin.saturating_neg();
    for packet in packets {
        packet.pts = packet.pts.saturating_add(offset);
        packet.dts = packet.dts.saturating_add(offset);
    }
}

/// Moves packets whose timeline starts at zero relative to their source track
/// (CPU video and decoded audio) onto that same file-level origin.
fn rebase_encoded_packets(packets: &mut [ContainerPacket], track_origin: i64, origin: i64) {
    let offset = track_origin.saturating_sub(origin);
    for packet in packets {
        packet.pts = packet.pts.saturating_add(offset);
        packet.dts = packet.dts.saturating_add(offset);
    }
}

/// For Opus tracks decoded from MP4 with an `elst` priming edit, the decoder
/// already strips `OpusHead.pre_skip` samples from the PCM output, so the
/// decoded stream begins after `pre_skip` rather than at the raw priming packet.
fn audio_decoded_origin(source_track: &ContainerTrack, track_origin: i64) -> i64 {
    if source_track.codec == VideoCodec::Opus
        && source_track.presentation_edit.is_some()
        && source_track.config.len() >= 12
        && &source_track.config[..8] == b"OpusHead"
    {
        let pre_skip = u16::from_le_bytes([source_track.config[10], source_track.config[11]]);
        if pre_skip > 0 {
            let pre_skip_us = rescale_ticks(i64::from(pre_skip), 1, OPUS_RATE);
            return track_origin.saturating_add(pre_skip_us);
        }
    }
    track_origin
}

/// The packet-copy path shared by video, audio and timed-text tracks.
fn copy_track(
    source_track: &ContainerTrack,
    track_packets: Vec<ContainerPacket>,
    new_index: usize,
    rebase: Option<i64>,
    target: &Path,
    tracks: &mut [Option<ContainerTrack>],
    packets: &mut Vec<ContainerPacket>,
) -> Result<(), PipelineError> {
    // Compatibility was already checked in `transcode_file` before any worker
    // started; do not validate the same codec/format pair twice here.
    let slot = tracks
        .get_mut(new_index)
        .ok_or_else(|| PipelineError::RemuxFailed {
            path: target.to_path_buf(),
            reason: format!("track index {new_index} is out of range"),
        })?;
    let mut copied_track = source_track.clone();
    let mut copied = track_packets;
    for packet in &mut copied {
        packet.track = new_index;
    }
    if let Some(origin) = rebase {
        rebase_source_packets(&mut copied, origin);
        copied_track.presentation_edit = copied_track
            .presentation_edit
            .map(|edit| edit.shifted(origin.saturating_neg()));
    }
    *slot = Some(copied_track);
    packets.extend(copied);
    Ok(())
}

/// Internal short-circuit used when the audio worker has already failed: the
/// video side must stop instead of encoding the rest of a long clip. The real
/// worker error is recovered when the thread is joined.
const AUDIO_ABORT_REASON: &str = "audio worker aborted";

fn audio_abort_error() -> PipelineError {
    PipelineError::RemuxFailed {
        path: PathBuf::new(),
        reason: AUDIO_ABORT_REASON.to_owned(),
    }
}

struct AudioEncodeJob {
    source_track: ContainerTrack,
    track_packets: Vec<ContainerPacket>,
    codec: TargetAudioCodec,
    config: AudioEncodeConfig,
    new_index: usize,
    track_origin: i64,
}

struct VideoEncodeJob {
    source_track: ContainerTrack,
    track_packets: Vec<ContainerPacket>,
    mode: VideoEncodeMode,
    decode_mode: VideoDecodeMode,
    new_index: usize,
    track_origin: i64,
}

fn apply_audio_event(
    event: AudioWorkerEvent,
    tracks: &mut [Option<ContainerTrack>],
    packets: &mut Vec<ContainerPacket>,
) {
    match event {
        AudioWorkerEvent::TrackHeader { new_index, track } => {
            if let Some(slot) = tracks.get_mut(new_index) {
                *slot = Some(track);
            }
        }
        AudioWorkerEvent::PacketBatch(batch) => packets.extend(batch),
    }
}

/// Drains everything the audio worker has queued so far without ever blocking
/// the video encoder. A failed worker short-circuits video immediately.
fn drain_audio_nonblocking(
    audio_rx: Option<&mpsc::Receiver<AudioWorkerEvent>>,
    audio_failed: &AtomicBool,
    tracks: &mut [Option<ContainerTrack>],
    packets: &mut Vec<ContainerPacket>,
) -> Result<(), PipelineError> {
    let Some(rx) = audio_rx else {
        return Ok(());
    };
    while let Ok(event) = rx.try_recv() {
        apply_audio_event(event, tracks, packets);
    }
    if audio_failed.load(Ordering::Acquire) {
        return Err(audio_abort_error());
    }
    Ok(())
}

fn presentation_edit_end(edit: TrackPresentation) -> Option<i64> {
    edit.duration_us.map(|duration| {
        edit.start_us
            .saturating_add(i64::try_from(duration).unwrap_or(i64::MAX))
    })
}

/// Selects the decode-ordered packet span needed to decode every visible frame
/// in `edit`, including preceding reference frames back to the governing
/// keyframe so inter-frame decoders (H.264/HEVC/AV1/VP9) never start mid-GOP.
fn select_packets_for_presentation_edit(
    edit: Option<TrackPresentation>,
    packets: &[ContainerPacket],
) -> &[ContainerPacket] {
    let Some(edit) = edit else {
        return packets;
    };
    let end_us = presentation_edit_end(edit);
    let is_visible = |packet: &ContainerPacket| {
        packet.pts >= edit.start_us && end_us.is_none_or(|end| packet.pts < end)
    };
    let Some(first_visible) = packets.iter().position(is_visible) else {
        return &[];
    };
    let last_visible = packets
        .iter()
        .rposition(is_visible)
        .unwrap_or(first_visible);
    let mut decode_start = packets[..=first_visible]
        .iter()
        .rposition(|packet| packet.keyframe)
        .unwrap_or(0);
    // Open GOP: if a later B-frame in the visible window displays earlier than
    // the chosen keyframe, back up to the preceding keyframe too.
    if let Some(min_visible_pts) = packets[first_visible..=last_visible]
        .iter()
        .filter(|packet| is_visible(packet))
        .map(|packet| packet.pts)
        .min()
    {
        while decode_start > 0 && packets[decode_start].pts > min_visible_pts {
            decode_start = packets[..decode_start]
                .iter()
                .rposition(|packet| packet.keyframe)
                .unwrap_or(0);
        }
    }
    &packets[decode_start..=last_visible]
}

/// Filters one decoded frame against `edit`, clips its duration to a bounded
/// edit end, and rebases PTS to the edit interval start.
fn apply_frame_presentation_edit(
    edit: Option<TrackPresentation>,
    info: DecodedFrameInfo,
) -> Option<DecodedFrameInfo> {
    let Some(edit) = edit else {
        return Some(info);
    };
    let end_us = presentation_edit_end(edit);
    if info.pts_us < edit.start_us || end_us.is_some_and(|end| info.pts_us >= end) {
        return None;
    }
    let mut duration_us = info.duration_us;
    if let Some(end) = end_us {
        let frame_end = info
            .pts_us
            .saturating_add(i64::try_from(duration_us).unwrap_or(i64::MAX));
        if frame_end > end {
            duration_us = u64::try_from(end.saturating_sub(info.pts_us)).unwrap_or(0);
        }
    }
    Some(DecodedFrameInfo {
        pts_us: info.pts_us.saturating_sub(edit.start_us),
        duration_us,
        ..info
    })
}

/// Encodes an animation container directly from decoded frames: audio is
/// dropped and the video frames bypass 4:2:0 entirely for animation sources.
fn transcode_to_animation(
    item: &VideoTranscodeItem,
    tag: usize,
    frame_workers: usize,
    decoded_video: &mut Vec<VideoDecodeMode>,
) -> Result<(u64, u64, u64, Option<String>), PipelineError> {
    let source = item.source();
    let target = item.target();
    let format = item.output_format();
    let VideoTrackAction::Encode(mode) = item.video_action() else {
        return Err(encode_error(
            source,
            "animation output requires video re-encoding",
        ));
    };
    if !matches!(
        mode,
        VideoEncodeMode::CpuGif { .. } | VideoEncodeMode::CpuWebp { .. }
    ) {
        return Err(encode_error(
            source,
            "animation output only supports CPU GIF/WebP encoding",
        ));
    }
    let quality = mode.quality();
    let media = demux_video(source, item.input_format())?;
    let video_index = media
        .tracks
        .iter()
        .position(|track| track.codec.is_video())
        .ok_or_else(|| decode_error(source, "source has no video track to encode"))?;
    let track = &media.tracks[video_index];
    let mut packets: Vec<ContainerPacket> = media
        .packets
        .iter()
        .filter(|packet| packet.track == video_index)
        .cloned()
        .collect();
    packets.sort_by_key(|packet| (packet.dts, packet.pts));
    let decode_packets = select_packets_for_presentation_edit(track.presentation_edit, &packets);

    // Loop count and alpha support were validated at the probe boundary;
    // plain video sources keep infinite playback and opaque pixels.
    let animation: AnimationMeta = item
        .streams()
        .iter()
        .find_map(fastconv_domain::VideoStreamInfo::animation)
        .unwrap_or_default();
    let loop_count = animation.loop_count();

    let staging = StagingTarget::create_with_handle(target, tag)?;
    let written = video_anim::encode_animation_stream(
        format,
        quality,
        frame_workers,
        loop_count,
        staging.file()?,
        |sink| match item.input_format() {
            VideoContainerFormat::Gif => {
                video_anim::decode_gif_frames_streaming_pixels(source, |info, rgb| sink(info, rgb))
            }
            VideoContainerFormat::Webp => {
                video_anim::decode_webp_frames_streaming_pixels(source, |info, rgb| sink(info, rgb))
            }
            _ => decode_video_track_streaming(source, track, decode_packets, |info, pack| {
                let Some(info) = apply_frame_presentation_edit(track.presentation_edit, info)
                else {
                    return Ok(());
                };
                let pitch = usize::try_from(info.width)
                    .map_err(|_| decode_error(source, "video frame width overflows usize"))?;
                let length = nv12_buffer_len(info.width, info.height, pitch)?;
                let mut nv12 = vec![0_u8; length];
                pack(&mut nv12, pitch)?;
                let rgb =
                    video_anim::nv12_to_rgb8(&nv12, pitch, info.width, info.height, info.color)?;
                sink(
                    AnimFrameInfo {
                        width: info.width,
                        height: info.height,
                        pts_us: info.pts_us,
                        duration_us: info.duration_us,
                        channels: 3,
                    },
                    &rgb,
                )
            }),
        },
    )?;
    staging.commit()?;
    // Animation output decodes frames even though it has no video encode job.
    decoded_video.push(item.decode_mode());
    Ok((written, input_size(source), input_size(target), None))
}

// Track-by-track dispatch keeps copy/encode decisions explicit per codec.
#[allow(clippy::too_many_lines)]
fn transcode_file(
    item: &VideoTranscodeItem,
    tag: usize,
    frame_workers: usize,
    decoded_video: &mut Vec<VideoDecodeMode>,
) -> Result<(u64, u64, u64, Option<String>), PipelineError> {
    let source = item.source();
    let target = item.target();
    let format = item.output_format();
    if matches!(
        format,
        VideoContainerFormat::Gif | VideoContainerFormat::Webp
    ) {
        return transcode_to_animation(item, tag, frame_workers, decoded_video);
    }
    let media = demux_video(source, item.input_format())?;
    // One file-level origin for every track: copied/NVENC packets carry source
    // timestamps, while CPU-encoded video and decoded audio start at zero
    // relative to their own source track. Rebasing both onto this origin keeps
    // the input's inter-track spacing instead of zeroing each track separately.
    let needs_rebase = media.tracks.iter().any(|track| {
        (track.codec.is_video() && matches!(item.video_action(), VideoTrackAction::Encode(_)))
            || (track.codec.is_audio()
                && matches!(item.audio_action(), AudioTrackAction::Encode(_)))
    });
    let origin = media
        .packets
        .iter()
        .map(|packet| packet.dts)
        .min()
        .unwrap_or(0);
    let rebase = needs_rebase.then_some(origin);
    let unsupported = |codec: VideoCodec| PipelineError::RemuxFailed {
        path: target.to_path_buf(),
        reason: DomainError::UnsupportedVideoCodec {
            codec,
            container: format,
        }
        .to_string(),
    };

    // Partition once, without cloning packets once per track.
    let mut per_track: Vec<Vec<ContainerPacket>> = vec![Vec::new(); media.tracks.len()];
    for packet in media.packets {
        if let Some(slot) = per_track.get_mut(packet.track) {
            slot.push(packet);
        }
    }
    let track_origins = per_track
        .iter()
        .map(|packets| packets.iter().map(|packet| packet.pts).min().unwrap_or(0))
        .collect::<Vec<_>>();
    // Copied tracks emit one packet per source packet; reserve only the slots
    // that survive into the output. Dropped PCM tracks can hold millions of
    // small packets and encoded tracks emit far fewer, so counting them here
    // would reserve hundreds of MiB for packets never written.
    let source_packet_count: usize = media
        .tracks
        .iter()
        .enumerate()
        .map(|(index, track)| {
            let copied = if track.codec.is_video() {
                matches!(item.video_action(), VideoTrackAction::Copy)
            } else if track.codec.is_audio() {
                matches!(item.audio_action(), AudioTrackAction::Copy)
            } else {
                true // timed text is copy-only
            };
            if copied { per_track[index].len() } else { 0 }
        })
        .sum();

    // Fail before any worker starts if a target codec cannot be carried.
    for track in &media.tracks {
        let codec = if track.codec.is_video() {
            match item.video_action() {
                VideoTrackAction::Copy => track.codec,
                VideoTrackAction::Encode(mode) => mode.codec().as_video_codec(),
            }
        } else if track.codec.is_audio() {
            match item.audio_action() {
                AudioTrackAction::Copy => track.codec,
                AudioTrackAction::Encode(config) => config.codec().as_video_codec(),
                AudioTrackAction::Drop => continue,
            }
        } else {
            track.codec
        };
        if !codec.supported_in(format) {
            return Err(unsupported(codec));
        }
    }

    // Pre-sized slots preserve source track order even when the two encoder
    // workers finish in a different order.
    let mut tracks: Vec<Option<ContainerTrack>> = vec![None; media.tracks.len()];
    let mut packets = Vec::with_capacity(source_packet_count);
    let mut audio_jobs = Vec::new();
    let mut video_jobs = Vec::new();

    for (index, source_track) in media.tracks.iter().enumerate() {
        let track_packets = std::mem::take(&mut per_track[index]);
        let new_index = index;
        let track_origin = track_origins[index];
        match (source_track.codec.is_video(), source_track.codec.is_audio()) {
            (true, _) => match item.video_action() {
                VideoTrackAction::Copy => copy_track(
                    source_track,
                    track_packets,
                    new_index,
                    rebase,
                    target,
                    &mut tracks,
                    &mut packets,
                )?,
                VideoTrackAction::Encode(mode) => {
                    video_jobs.push(VideoEncodeJob {
                        source_track: source_track.clone(),
                        track_packets,
                        mode,
                        decode_mode: item.decode_mode(),
                        new_index,
                        track_origin,
                    });
                }
            },
            (false, true) => match item.audio_action() {
                AudioTrackAction::Copy => copy_track(
                    source_track,
                    track_packets,
                    new_index,
                    rebase,
                    target,
                    &mut tracks,
                    &mut packets,
                )?,
                AudioTrackAction::Encode(config) => {
                    audio_jobs.push(AudioEncodeJob {
                        source_track: source_track.clone(),
                        track_packets,
                        codec: config.codec(),
                        config,
                        new_index,
                        track_origin,
                    });
                }
                AudioTrackAction::Drop => {}
            },
            // Timed text is copy-only. The domain already allows it in MOV/MP4
            // and skips it in codec validation, so execution must not abort.
            (false, false) => copy_track(
                source_track,
                track_packets,
                new_index,
                rebase,
                target,
                &mut tracks,
                &mut packets,
            )?,
        }
    }

    let cancelled = AtomicBool::new(false);
    let audio_failed = AtomicBool::new(false);
    thread::scope(
        |scope| -> Result<(u64, u64, u64, Option<String>), PipelineError> {
            let (audio_tx, audio_rx) = if audio_jobs.is_empty() {
                (None, None)
            } else {
                let (tx, rx) = mpsc::sync_channel::<AudioWorkerEvent>(AUDIO_QUEUE_CAPACITY);
                (Some(tx), Some(rx))
            };
            let audio_handle = if let Some(audio_tx) = &audio_tx {
                let audio_tx = audio_tx.clone();
                let jobs = &audio_jobs;
                let cancelled = &cancelled;
                let audio_failed = &audio_failed;
                Some(
                    thread::Builder::new()
                        .name("fastconv-audio-worker".to_owned())
                        .spawn_scoped(scope, move || -> Result<(), PipelineError> {
                            for job in jobs {
                                if cancelled.load(Ordering::Acquire) {
                                    break;
                                }
                                let ctx = AudioStreamContext {
                                    new_index: job.new_index,
                                    rebase: needs_rebase.then_some((
                                        audio_decoded_origin(&job.source_track, job.track_origin),
                                        origin,
                                    )),
                                    cancelled,
                                    audio_tx: &audio_tx,
                                };
                                if let Err(err) = transcode_audio_track_streaming(
                                    source,
                                    &job.source_track,
                                    &job.track_packets,
                                    job.codec,
                                    job.config,
                                    &ctx,
                                ) {
                                    audio_failed.store(true, Ordering::Release);
                                    return Err(err);
                                }
                            }
                            Ok(())
                        })
                        .map_err(PipelineError::WorkerStartFailed)?,
                )
            } else {
                None
            };
            // Only the worker's clone stays alive, so a blocking drain ends
            // exactly when the worker finishes.
            drop(audio_tx);

            let mut frames_processed = 0_u64;
            let mut hardware_device = None;
            let mut video_error = None;
            'video: for job in &video_jobs {
                match job.mode {
                    VideoEncodeMode::NvencH264 { .. } | VideoEncodeMode::NvencHevc { .. } => {
                        let (mut stream, frames, device) = match match job.decode_mode {
                            VideoDecodeMode::Nvdec => {
                                crate::gpu::nvenc::encode_video_nvdec_nvenc_streaming(
                                    &job.source_track,
                                    &job.track_packets,
                                    job.source_track.codec,
                                    job.mode,
                                    job.new_index,
                                    || {
                                        drain_audio_nonblocking(
                                            audio_rx.as_ref(),
                                            &audio_failed,
                                            &mut tracks,
                                            &mut packets,
                                        )
                                    },
                                )
                            }
                            VideoDecodeMode::Cpu => {
                                crate::gpu::nvenc::encode_video_nvenc_streaming(
                                    source,
                                    &job.source_track,
                                    &job.track_packets,
                                    job.mode,
                                    job.new_index,
                                    || {
                                        drain_audio_nonblocking(
                                            audio_rx.as_ref(),
                                            &audio_failed,
                                            &mut tracks,
                                            &mut packets,
                                        )
                                    },
                                )
                            }
                        } {
                            Ok(result) => {
                                decoded_video.push(job.decode_mode);
                                result
                            }
                            Err(err) => {
                                video_error = Some(err);
                                break 'video;
                            }
                        };
                        stream.track.presentation_edit = job.source_track.presentation_edit;
                        if needs_rebase {
                            rebase_source_packets(&mut stream.packets, origin);
                            stream.track.presentation_edit = stream
                                .track
                                .presentation_edit
                                .map(|edit| edit.shifted(origin.saturating_neg()));
                        }
                        for packet in &mut stream.packets {
                            packet.track = job.new_index;
                        }
                        if let Some(slot) = tracks.get_mut(job.new_index) {
                            *slot = Some(stream.track);
                        }
                        packets.extend(stream.packets);
                        frames_processed = frames_processed.saturating_add(frames);
                        if hardware_device.is_none() {
                            hardware_device = Some(device);
                        }
                    }
                    VideoEncodeMode::CpuGif { .. } | VideoEncodeMode::CpuWebp { .. } => {
                        video_error = Some(encode_error(
                            source,
                            "animation output must be encoded before container muxing",
                        ));
                        break 'video;
                    }
                    VideoEncodeMode::CpuH264 { .. }
                    | VideoEncodeMode::CpuAv1 { .. }
                    | VideoEncodeMode::CpuVp9 { .. }
                    | VideoEncodeMode::CpuMjpeg { .. } => {
                        if let Err(err) = drain_audio_nonblocking(
                            audio_rx.as_ref(),
                            &audio_failed,
                            &mut tracks,
                            &mut packets,
                        ) {
                            video_error = Some(err);
                            break 'video;
                        }
                        let decoded =
                            (|| -> Result<Vec<crate::video_codecs::Yuv420Frame>, PipelineError> {
                                enforce_decoded_budget(
                                    source,
                                    estimated_decoded_bytes(
                                        &job.source_track,
                                        job.track_packets.len(),
                                    ),
                                )?;
                                let frames = decode_video_track(
                                    source,
                                    &job.source_track,
                                    &job.track_packets,
                                )?;
                                enforce_decoded_budget(source, decoded_bytes(&frames))?;
                                Ok(frames)
                            })();
                        let frames = match decoded {
                            Ok(frames) => frames,
                            Err(err) => {
                                video_error = Some(err);
                                break 'video;
                            }
                        };
                        decoded_video.push(job.decode_mode);
                        // An audio worker failure during decode must stop the job
                        // before the (much longer) CPU encode starts.
                        if let Err(err) = drain_audio_nonblocking(
                            audio_rx.as_ref(),
                            &audio_failed,
                            &mut tracks,
                            &mut packets,
                        ) {
                            video_error = Some(err);
                            break 'video;
                        }
                        let count = u64::try_from(frames.len()).unwrap_or(u64::MAX);
                        let mut stream =
                            match encode_video_cpu(source, &frames, job.mode, job.new_index) {
                                Ok(stream) => stream,
                                Err(err) => {
                                    video_error = Some(err);
                                    break 'video;
                                }
                            };
                        frames_processed = frames_processed.saturating_add(count);
                        stream.track.presentation_edit = job.source_track.presentation_edit;
                        if needs_rebase {
                            rebase_encoded_packets(&mut stream.packets, job.track_origin, origin);
                            stream.track.presentation_edit = stream
                                .track
                                .presentation_edit
                                .map(|edit| edit.shifted(origin.saturating_neg()));
                        }
                        for packet in &mut stream.packets {
                            packet.track = job.new_index;
                        }
                        if let Some(slot) = tracks.get_mut(job.new_index) {
                            *slot = Some(stream.track);
                        }
                        packets.extend(stream.packets);
                        if let Err(err) = drain_audio_nonblocking(
                            audio_rx.as_ref(),
                            &audio_failed,
                            &mut tracks,
                            &mut packets,
                        ) {
                            video_error = Some(err);
                            break 'video;
                        }
                    }
                }
            }

            let mut audio_result = Ok(());
            if let Some(handle) = audio_handle {
                if video_error.is_none() {
                    // The worker owns the only remaining sender.
                    if let Some(rx) = &audio_rx {
                        while let Ok(event) = rx.recv() {
                            apply_audio_event(event, &mut tracks, &mut packets);
                        }
                    }
                } else {
                    cancelled.store(true, Ordering::Release);
                    // Drop the receiver before joining: a worker blocked in send
                    // must be released immediately, not after it fills the queue.
                    drop(audio_rx);
                }
                audio_result = match handle.join() {
                    Ok(result) => result,
                    Err(_) => Err(PipelineError::WorkerPanicked),
                };
            }
            audio_result?;
            if let Some(err) = video_error {
                if matches!(&err, PipelineError::RemuxFailed { reason, .. } if reason == AUDIO_ABORT_REASON)
                {
                    return Err(audio_abort_error());
                }
                return Err(err);
            }

            // A dropped audio track leaves its pre-sized slot empty. Compact the
            // surviving tracks and remap every packet onto the dense indices so
            // the muxer never sees a hole or a packet for a removed track.
            let mut index_map = vec![None; media.tracks.len()];
            let mut staged_tracks = Vec::with_capacity(tracks.len());
            for (old_index, slot) in tracks.into_iter().enumerate() {
                if let Some(track) = slot {
                    index_map[old_index] = Some(staged_tracks.len());
                    staged_tracks.push(track);
                }
            }
            if staged_tracks.is_empty() || packets.is_empty() {
                return Err(PipelineError::RemuxFailed {
                    path: target.to_path_buf(),
                    reason: "source has no supported tracks to transcode".to_owned(),
                });
            }
            packets.retain(|packet| index_map.get(packet.track).is_some_and(Option::is_some));
            for packet in &mut packets {
                if let Some(new_index) = index_map.get(packet.track).copied().flatten() {
                    packet.track = new_index;
                }
            }
            packets.sort_by_key(|packet| (packet.dts, packet.track));
            let out_media = crate::video_container::ContainerMedia {
                tracks: staged_tracks,
                packets,
            };
            let staging = StagingTarget::create_with_handle(target, tag)?;
            mux_video(staging.file()?, target, format, &out_media)?;
            staging.commit()?;
            Ok((
                frames_processed,
                input_size(source),
                input_size(target),
                hardware_device,
            ))
        },
    )
}

/// Worker admission under one lock: checking cancellation and claiming the
/// next item are indivisible, so no worker can claim work after cancellation.
#[derive(Default)]
struct BatchAdmission {
    next: usize,
    cancelled: bool,
    failure: Option<PipelineError>,
}

impl BatchAdmission {
    fn claim(&mut self) -> Option<usize> {
        if self.cancelled {
            return None;
        }
        let index = self.next;
        self.next = self.next.saturating_add(1);
        Some(index)
    }

    fn stop(&mut self) {
        self.cancelled = true;
    }

    fn fail(&mut self, err: PipelineError) {
        if self.failure.is_none() {
            self.failure = Some(err);
        }
        self.stop();
    }
}

/// Executes a validated [`VideoTranscodePlan`] with bounded file-level
/// parallelism. NVENC items run sequentially so GPU sessions do not contend.
///
/// # Errors
///
/// Returns [`PipelineError`] if demuxing, decoding, encoding, muxing or staging
/// any item fails. No silent CPU fallback is ever performed.
// Worker setup, sequential/parallel dispatch and report construction stay together.
#[allow(clippy::too_many_lines)]
pub fn execute_video_transcode_batch(
    plan: &VideoTranscodePlan,
) -> Result<VideoTranscodeReport, PipelineError> {
    let started = Instant::now();
    let items = plan.items();
    let total_input: u64 = items.iter().map(|item| input_size(item.source())).sum();
    let uses_nvenc = items.iter().any(|item| {
        matches!(
            item.video_action(),
            VideoTrackAction::Encode(
                VideoEncodeMode::NvencH264 { .. } | VideoEncodeMode::NvencHevc { .. }
            )
        )
    });
    let configured_threads = plan.worker_threads().get();
    let workers = if uses_nvenc {
        1
    } else {
        plan.effective_workers().get().min(MAX_CPU_FILE_WORKERS)
    };
    let frame_workers = if items.len() == 1 {
        configured_threads
    } else {
        configured_threads.div_ceil(workers.max(1)).max(1)
    };
    let admission = Mutex::new(BatchAdmission::default());
    let frames = AtomicU64::new(0);
    let device: Mutex<Option<String>> = Mutex::new(None);
    let tag = AtomicUsize::new(0);
    let decoded_video: Mutex<Vec<VideoDecodeMode>> = Mutex::new(Vec::new());

    if items.len() == 1 {
        let mut modes = Vec::new();
        let (processed, _in, _out, hw) = transcode_file(&items[0], 0, frame_workers, &mut modes)?;
        frames.store(processed, Ordering::Relaxed);
        if let Some(hw) = hw
            && let Ok(mut guard) = device.lock()
        {
            *guard = Some(hw);
        }
        if let Ok(mut guard) = decoded_video.lock() {
            guard.extend(modes);
        }
    } else {
        thread::scope(|scope| -> Result<(), PipelineError> {
            let mut handles = Vec::with_capacity(workers);
            let mut startup_error = None;
            for worker in 0..workers {
                let (admission, frames, device, tag, decoded_video) =
                    (&admission, &frames, &device, &tag, &decoded_video);
                let worker = thread::Builder::new()
                    .name(format!("fastconv-video-{worker}"))
                    .spawn_scoped(scope, move || {
                        while let Some(index) =
                            admission.lock().ok().and_then(|mut guard| guard.claim())
                        {
                            let Some(item) = items.get(index) else { break };
                            let my_tag = tag.fetch_add(1, Ordering::Relaxed);
                            let mut modes = Vec::new();
                            match transcode_file(item, my_tag, frame_workers, &mut modes) {
                                Ok((processed, _in, _out, hw)) => {
                                    frames.fetch_add(processed, Ordering::Relaxed);
                                    if let Some(hw) = hw
                                        && let Ok(mut guard) = device.lock()
                                        && guard.is_none()
                                    {
                                        *guard = Some(hw);
                                    }
                                    if let Ok(mut guard) = decoded_video.lock() {
                                        guard.extend(modes);
                                    }
                                }
                                Err(err) => {
                                    if let Ok(mut guard) = admission.lock() {
                                        guard.fail(err);
                                    }
                                    break;
                                }
                            }
                        }
                    });
                match worker {
                    Ok(handle) => handles.push(handle),
                    Err(err) => {
                        // Stop the workers already running from taking more
                        // items, then join all of them before propagating.
                        if let Ok(mut guard) = admission.lock() {
                            guard.stop();
                        }
                        startup_error = Some(err);
                        break;
                    }
                }
            }
            for handle in handles {
                handle.join().map_err(|_| PipelineError::WorkerPanicked)?;
            }
            if let Some(err) = startup_error {
                return Err(PipelineError::WorkerStartFailed(err));
            }
            Ok(())
        })?;
    }
    let admission = admission
        .into_inner()
        .map_err(|_| PipelineError::WorkerPanicked)?;
    if let Some(err) = admission.failure {
        return Err(err);
    }

    let total_output: u64 = items.iter().map(|item| input_size(item.target())).sum();
    let files =
        NonZeroUsize::new(items.len()).ok_or_else(|| PipelineError::EmptyOutput(PathBuf::new()))?;
    let input_bytes = NonZeroU64::new(total_input.max(1)).unwrap_or(NonZeroU64::MIN);
    let output_bytes = NonZeroU64::new(total_output.max(1)).unwrap_or(NonZeroU64::MIN);
    let worker_threads = NonZeroUsize::new(workers).unwrap_or(NonZeroUsize::MIN);
    let backend = if uses_nvenc {
        HardwareBackend::Gpu(GpuBackend::Nvenc)
    } else {
        HardwareBackend::Cpu
    };
    let hardware_device = device
        .into_inner()
        .map_err(|_| PipelineError::WorkerPanicked)?;
    // Only files that actually dispatched a video decode contribute a path;
    // an audio-only copy has no video job and must not be reported as CPU decode.
    let executed = decoded_video
        .into_inner()
        .map_err(|_| PipelineError::WorkerPanicked)?;
    let decode_paths = VideoDecodePaths::from_modes(executed).unwrap_or(VideoDecodePaths::NoVideo);
    Ok(VideoTranscodeReport::new(
        files,
        frames.load(Ordering::Relaxed),
        input_bytes,
        output_bytes,
        started.elapsed(),
        worker_threads,
        backend,
        hardware_device,
        decode_paths,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_codecs::{Yuv420Frame, YuvColor};
    use crate::video_container::{ContainerMedia, mux_video};
    use fastconv_domain::{
        AnimationLoop, AudioBitrate, AudioChannels, AudioEncodeOptions, AudioSampleRate,
        DomainError, TargetVideoCodec, VideoBitrate, VideoContainerFormat, VideoQuality,
        VideoRateControl, VideoStreamKind,
    };
    use std::num::NonZeroUsize;
    use std::process::{Command, ExitStatus, Stdio};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(label: &str) -> Result<PathBuf, PipelineError> {
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "fastconv-transcode-{label}-{}-{id}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(dir)
    }

    fn ogg_page(header_type: u8, granulepos: i64, lacing: &[u8], body: &[u8]) -> Vec<u8> {
        let mut page = Vec::new();
        page.extend_from_slice(b"OggS");
        page.push(0); // stream version
        page.push(header_type);
        page.extend_from_slice(&granulepos.to_le_bytes());
        page.extend_from_slice(&[0; 4]); // serial number
        page.extend_from_slice(&[0; 4]); // page sequence
        page.extend_from_slice(&[0; 4]); // CRC (not checked by the collector)
        page.push(u8::try_from(lacing.len()).unwrap_or(0));
        page.extend_from_slice(lacing);
        page.extend_from_slice(body);
        page
    }

    #[test]
    fn ogg_collector_extracts_multi_segment_and_exact_255_packets()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut collector = OggPacketCollector::default();
        let body = vec![0x5A_u8; 255];
        // A packet whose length is an exact multiple of 255 ends with a zero
        // lacing byte, not with an implicit continuation.
        let page = ogg_page(0, 255, &[255, 0], &body);
        collector.push_bytes(&page)?;
        let packets = collector.drain();
        assert_eq!(packets.len(), 1);
        assert_eq!(packets[0].data, body);
        assert_eq!(packets[0].granulepos, 255);
        assert!(!packets[0].eos_last);

        let body = vec![0x33_u8; 510];
        let page = ogg_page(0x04, 512, &[255, 255, 0], &body);
        collector.push_bytes(&page)?;
        let packets = collector.drain();
        assert_eq!(packets.len(), 1);
        assert_eq!(packets[0].data, body);
        assert!(packets[0].eos_last);
        assert_eq!(packets[0].granulepos, 512);
        Ok(())
    }

    #[test]
    fn ogg_collector_reassembles_packets_across_pages() -> Result<(), Box<dyn std::error::Error>> {
        let mut collector = OggPacketCollector::default();
        let first = vec![0x11_u8; 255];
        collector.push_bytes(&ogg_page(0, -1, &[255], &first))?;
        assert!(collector.drain().is_empty());

        let tail = vec![0x22_u8; 4];
        let next = vec![0x33_u8; 255];
        let mut body = tail.clone();
        body.extend_from_slice(&next);
        collector.push_bytes(&ogg_page(0x01, 259, &[4, 255], &body))?;
        let packets = collector.drain();
        assert_eq!(packets.len(), 1);
        let mut expected = first.clone();
        expected.extend_from_slice(&tail);
        assert_eq!(packets[0].data, expected);
        assert_eq!(packets[0].granulepos, 259);

        let final_body = [0x44_u8; 1];
        collector.push_bytes(&ogg_page(0x01 | 0x04, 260, &[1], &final_body))?;
        let packets = collector.drain();
        assert_eq!(packets.len(), 1);
        let mut expected = next;
        expected.push(0x44);
        assert_eq!(packets[0].data, expected);
        assert!(packets[0].eos_last);
        assert_eq!(packets[0].granulepos, 260);
        Ok(())
    }

    #[test]
    fn ogg_collector_marks_only_last_packet_on_eos_page() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut collector = OggPacketCollector::default();
        let body = [1_u8, 2, 3, 4];
        collector.push_bytes(&ogg_page(0x04, 1234, &[2, 2], &body))?;
        let packets = collector.drain();
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].data, vec![1, 2]);
        assert!(!packets[0].eos_last);
        assert_eq!(packets[1].data, vec![3, 4]);
        assert!(packets[1].eos_last);
        assert_eq!(packets[1].granulepos, 1234);
        Ok(())
    }

    #[test]
    fn ogg_collector_rejects_invalid_capture_pattern() {
        let mut collector = OggPacketCollector::default();
        let mut page = ogg_page(0, 0, &[1], &[7]);
        page[0] = b'X';
        assert!(collector.push_bytes(&page).is_err());
    }

    fn frames(count: usize) -> Vec<Yuv420Frame> {
        (0..count)
            .map(|index| {
                let (w, h) = (64_usize, 48_usize);
                Yuv420Frame {
                    width: 64,
                    height: 48,
                    y: (0..w * h)
                        .map(|i| u8::try_from((i.saturating_add(index * 13)) % 251).unwrap_or(0))
                        .collect(),
                    u: vec![128_u8; (w / 2) * (h / 2)],
                    v: vec![118_u8; (w / 2) * (h / 2)],
                    pts_us: i64::try_from(index).unwrap_or(i64::MAX) * 20_000,
                    duration_us: 20_000,
                    color: YuvColor::default(),
                }
            })
            .collect()
    }

    // Synthetic test tone only: the f64-to-f32 narrowing is intentional here.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn sine(frames: usize, rate: u32, channels: u16) -> Vec<f32> {
        (0..frames * usize::from(channels))
            .map(|i| {
                let n = f64::from(u32::try_from(i).unwrap_or(0));
                let period = f64::from(rate) / 440.0;
                (n * std::f64::consts::TAU / period).sin() as f32 * 0.35
            })
            .collect()
    }

    fn default_config(codec: TargetAudioCodec) -> AudioEncodeConfig {
        AudioEncodeConfig::default_for(codec)
    }

    fn make_source(
        path: &Path,
        format: VideoContainerFormat,
        video: VideoEncodeMode,
        audio: TargetAudioCodec,
    ) -> Result<(), PipelineError> {
        let video_frames = frames(4);
        let encoded = encode_video_cpu(path, &video_frames, video, 0)?;
        let samples = sine(1_920, 48_000, 2);
        let (audio_track, mut audio_packets) =
            encode_audio_track(path, &samples, 48_000, 2, audio, default_config(audio))?;
        for packet in &mut audio_packets {
            packet.track = 1;
        }
        let media = ContainerMedia {
            tracks: vec![encoded.track, audio_track],
            packets: encoded.packets.into_iter().chain(audio_packets).collect(),
        };
        let file = File::create(path).map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        mux_video(&file, path, format, &media)?;
        drop(file);
        Ok(())
    }

    fn streams(
        path: &Path,
        format: VideoContainerFormat,
    ) -> Result<Vec<fastconv_domain::VideoStreamInfo>, PipelineError> {
        let media = demux_video(path, format)?;
        Ok(media
            .tracks
            .iter()
            .map(|track| {
                fastconv_domain::VideoStreamInfo::new(
                    track.codec,
                    if track.codec.is_video() {
                        VideoStreamKind::Video
                    } else {
                        VideoStreamKind::Audio
                    },
                    track.width.map(u32::from),
                    track.height.map(u32::from),
                    track.channels,
                    track.sample_rate,
                    1,
                    US_TIMESCALE,
                    None,
                    None,
                )
            })
            .collect())
    }

    fn transcode(
        source: &Path,
        target: &Path,
        input: VideoContainerFormat,
        output: VideoContainerFormat,
        video: VideoTrackAction,
        audio: AudioTrackAction,
    ) -> Result<VideoTranscodeReport, PipelineError> {
        let plan = VideoTranscodePlan::new(
            vec![
                VideoTranscodeItem::new(
                    source.to_path_buf(),
                    target.to_path_buf(),
                    input,
                    output,
                    streams(source, input)?,
                    video,
                    audio,
                    fastconv_domain::VideoDecodeMode::Cpu,
                )
                .map_err(|err| PipelineError::RemuxFailed {
                    path: target.to_path_buf(),
                    reason: err.to_string(),
                })?,
            ],
            NonZeroUsize::MIN,
        )
        .map_err(|err| PipelineError::RemuxFailed {
            path: target.to_path_buf(),
            reason: err.to_string(),
        })?;
        execute_video_transcode_batch(&plan)
    }

    #[test]
    fn mjpeg_transcodes_from_h264_with_audio() -> Result<(), PipelineError> {
        let dir = temp_dir("mjpeg-transcode-from")?;
        let mjpeg_mode = VideoEncodeMode::CpuMjpeg {
            quality: VideoQuality::DEFAULT,
        };

        // H.264/AAC -> MJPEG in each supported container, with audio copied.
        let h264_source = dir.join("h264-source.mkv");
        make_source(
            &h264_source,
            VideoContainerFormat::Mkv,
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            TargetAudioCodec::Aac,
        )?;
        for (ext, format) in [
            ("avi", VideoContainerFormat::Avi),
            ("mov", VideoContainerFormat::Mov),
            ("mkv", VideoContainerFormat::Mkv),
        ] {
            let target = dir.join(format!("mjpeg-out.{ext}"));
            transcode(
                &h264_source,
                &target,
                VideoContainerFormat::Mkv,
                format,
                VideoTrackAction::Encode(mjpeg_mode),
                AudioTrackAction::Copy,
            )?;
            assert_output(&target, format, VideoCodec::Mjpeg, VideoCodec::Aac)?;
            let media = demux_video(&target, format)?;
            let packets: Vec<ContainerPacket> = media
                .packets
                .iter()
                .filter(|packet| media.tracks[packet.track].codec == VideoCodec::Mjpeg)
                .cloned()
                .collect();
            assert!(
                packets
                    .iter()
                    .all(|packet| packet.keyframe && packet.dts == packet.pts)
            );
            let decoded = decode_video_track(&target, &media.tracks[0], &packets)?;
            assert_eq!(decoded.len(), 4, "MJPEG frames must decode in {format}");
        }
        Ok(())
    }

    #[test]
    fn mjpeg_transcodes_to_other_codecs() -> Result<(), PipelineError> {
        let dir = temp_dir("mjpeg-transcode-to")?;
        let mjpeg_mode = VideoEncodeMode::CpuMjpeg {
            quality: VideoQuality::DEFAULT,
        };
        // MJPEG -> H.264/VP9 (audio copied) and -> GIF/WebP (audio dropped).
        let mjpeg_source = dir.join("mjpeg-source.avi");
        make_source(
            &mjpeg_source,
            VideoContainerFormat::Avi,
            mjpeg_mode,
            TargetAudioCodec::Aac,
        )?;
        for (label, ext, format, mode, expected, copy_audio) in [
            (
                "h264",
                "mkv",
                VideoContainerFormat::Mkv,
                VideoEncodeMode::CpuH264 {
                    rate_control: VideoRateControl::DEFAULT,
                },
                VideoCodec::H264,
                true,
            ),
            (
                "vp9",
                "webm",
                VideoContainerFormat::Webm,
                VideoEncodeMode::CpuVp9 {
                    rate_control: VideoRateControl::DEFAULT,
                },
                VideoCodec::Vp9,
                true,
            ),
            (
                "gif",
                "gif",
                VideoContainerFormat::Gif,
                VideoEncodeMode::CpuGif {
                    quality: VideoQuality::DEFAULT,
                },
                VideoCodec::Gif,
                false,
            ),
            (
                "webp",
                "webp",
                VideoContainerFormat::Webp,
                VideoEncodeMode::CpuWebp {
                    quality: VideoQuality::DEFAULT,
                },
                VideoCodec::Webp,
                false,
            ),
        ] {
            let target = dir.join(format!("{label}-from-mjpeg.{ext}"));
            // WebM cannot carry AAC, so that hop transcodes audio to Opus.
            let audio = if format == VideoContainerFormat::Webm {
                AudioTrackAction::Encode(default_config(TargetAudioCodec::Opus))
            } else if copy_audio {
                AudioTrackAction::Copy
            } else {
                AudioTrackAction::Drop
            };
            transcode(
                &mjpeg_source,
                &target,
                VideoContainerFormat::Avi,
                format,
                VideoTrackAction::Encode(mode),
                audio,
            )?;
            let media = demux_video(&target, format)?;
            assert_eq!(media.tracks[0].codec, expected, "{label} output codec");
            if copy_audio || format == VideoContainerFormat::Webm {
                let audio_codec = if format == VideoContainerFormat::Webm {
                    VideoCodec::Opus
                } else {
                    VideoCodec::Aac
                };
                assert_output(&target, format, expected, audio_codec)?;
            }
        }
        Ok(())
    }

    fn assert_output(
        path: &Path,
        format: VideoContainerFormat,
        video: VideoCodec,
        audio: VideoCodec,
    ) -> Result<(), PipelineError> {
        let media = demux_video(path, format)?;
        assert_eq!(media.tracks.len(), 2, "track count for {}", path.display());
        assert_eq!(media.tracks[0].codec, video, "video codec");
        assert_eq!(media.tracks[1].codec, audio, "audio codec");
        let video_packets = media.packets.iter().filter(|p| p.track == 0).count();
        let audio_packets = media.packets.iter().filter(|p| p.track == 1).count();
        assert!(
            video_packets >= 4,
            "expected video frames, got {video_packets}"
        );
        assert!(audio_packets > 0, "expected audio packets");
        Ok(())
    }

    // Non-empty PCM/packet assertions document successful codec roundtrips.
    #[allow(clippy::assert_is_empty)]
    #[test]
    fn encodes_all_three_audio_codecs() -> Result<(), PipelineError> {
        let dir = temp_dir("audio-codecs")?;
        let samples = sine(4_800, 48_000, 2);
        for codec in [
            TargetAudioCodec::Aac,
            TargetAudioCodec::Opus,
            TargetAudioCodec::Mp3,
        ] {
            let (track, packets) = encode_audio_track(
                &dir.join("x"),
                &samples,
                48_000,
                2,
                codec,
                default_config(codec),
            )?;
            assert_eq!(track.codec, codec.as_video_codec());
            assert!(!packets.is_empty());
            let (decoded, rate, channels) =
                decode_audio_track_pcm(&dir.join("x"), &track, &packets)?;
            assert!(!decoded.is_empty());
            assert!(rate > 0);
            assert_eq!(channels, 2);
        }
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn mp4_h264_aac_to_webm_vp9_opus() -> Result<(), PipelineError> {
        let dir = temp_dir("mp4-webm")?;
        let mp4 = dir.join("in.mp4");
        let webm = dir.join("out.webm");
        make_source(
            &mp4,
            VideoContainerFormat::Mp4,
            VideoEncodeMode::CpuH264 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            TargetAudioCodec::Aac,
        )?;
        let report = transcode(
            &mp4,
            &webm,
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Webm,
            VideoTrackAction::Encode(VideoEncodeMode::CpuVp9 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            }),
            AudioTrackAction::Encode(default_config(TargetAudioCodec::Opus)),
        )?;
        assert_eq!(report.files_converted().get(), 1);
        assert_eq!(report.frames_processed(), 4);
        assert_eq!(report.decode_paths(), VideoDecodePaths::Cpu);
        assert_eq!(report.decode_path(), "cpu");
        assert_output(
            &webm,
            VideoContainerFormat::Webm,
            VideoCodec::Vp9,
            VideoCodec::Opus,
        )?;
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn aac_and_opus_sources_transcode_to_vorbis_mkv_and_webm() -> Result<(), PipelineError> {
        let dir = temp_dir("to-vorbis")?;
        let mp4 = dir.join("aac.mp4");
        let webm = dir.join("opus.webm");
        let out_mkv = dir.join("vorbis.mkv");
        let out_webm = dir.join("vorbis.webm");
        make_source(
            &mp4,
            VideoContainerFormat::Mp4,
            VideoEncodeMode::CpuH264 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            TargetAudioCodec::Aac,
        )?;
        make_source(
            &webm,
            VideoContainerFormat::Webm,
            VideoEncodeMode::CpuVp9 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            TargetAudioCodec::Opus,
        )?;

        transcode(
            &mp4,
            &out_mkv,
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mkv,
            VideoTrackAction::Encode(VideoEncodeMode::CpuH264 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            }),
            AudioTrackAction::Encode(default_config(TargetAudioCodec::Vorbis)),
        )?;
        assert_output(
            &out_mkv,
            VideoContainerFormat::Mkv,
            VideoCodec::H264,
            VideoCodec::Vorbis,
        )?;

        transcode(
            &webm,
            &out_webm,
            VideoContainerFormat::Webm,
            VideoContainerFormat::Webm,
            VideoTrackAction::Copy,
            AudioTrackAction::Encode(default_config(TargetAudioCodec::Vorbis)),
        )?;
        assert_output(
            &out_webm,
            VideoContainerFormat::Webm,
            VideoCodec::Vp9,
            VideoCodec::Vorbis,
        )?;
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn vorbis_source_transcodes_to_aac_opus_and_vorbis() -> Result<(), PipelineError> {
        let dir = temp_dir("from-vorbis")?;
        let mkv = dir.join("in.mkv");
        make_source(
            &mkv,
            VideoContainerFormat::Mkv,
            VideoEncodeMode::CpuH264 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            TargetAudioCodec::Vorbis,
        )?;
        let cases = [
            (
                "out.mp4",
                VideoContainerFormat::Mp4,
                VideoTrackAction::Encode(VideoEncodeMode::CpuH264 {
                    rate_control: fastconv_domain::VideoRateControl::DEFAULT,
                }),
                TargetAudioCodec::Aac,
                VideoCodec::H264,
                VideoCodec::Aac,
            ),
            (
                "out.webm",
                VideoContainerFormat::Webm,
                VideoTrackAction::Encode(VideoEncodeMode::CpuVp9 {
                    rate_control: fastconv_domain::VideoRateControl::DEFAULT,
                }),
                TargetAudioCodec::Opus,
                VideoCodec::Vp9,
                VideoCodec::Opus,
            ),
            (
                "out2.mkv",
                VideoContainerFormat::Mkv,
                VideoTrackAction::Copy,
                TargetAudioCodec::Vorbis,
                VideoCodec::H264,
                VideoCodec::Vorbis,
            ),
        ];
        for (name, format, video, audio, video_codec, audio_codec) in cases {
            let target = dir.join(name);
            transcode(
                &mkv,
                &target,
                VideoContainerFormat::Mkv,
                format,
                video,
                AudioTrackAction::Encode(default_config(audio)),
            )?;
            assert_output(&target, format, video_codec, audio_codec)?;
        }
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn vorbis_options_set_rate_channels_and_bitrate() -> Result<(), PipelineError> {
        let dir = temp_dir("vorbis-options")?;
        let source = dir.join("in.mkv");
        let target = dir.join("out.mkv");
        make_source(
            &source,
            VideoContainerFormat::Mkv,
            VideoEncodeMode::CpuH264 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            TargetAudioCodec::Aac,
        )?;
        let config = audio_config(TargetAudioCodec::Vorbis, 128_000, 44_100, 1)?;
        transcode(
            &source,
            &target,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Mkv,
            VideoTrackAction::Copy,
            AudioTrackAction::Encode(config),
        )?;
        let media = demux_video(&target, VideoContainerFormat::Mkv)?;
        let audio = media
            .tracks
            .iter()
            .find(|track| track.codec == VideoCodec::Vorbis)
            .ok_or_else(|| decode_error(&target, "Vorbis track missing after transcode"))?;
        assert_eq!(audio.sample_rate, Some(44_100));
        assert_eq!(audio.channels, Some(1));
        let (ident, _, _) = unpack_xiph_laced_vorbis_headers(&audio.config).map_err(|reason| {
            decode_error(
                &target,
                format!("invalid Vorbis CodecPrivate in output: {reason}"),
            )
        })?;
        let (ident_channels, ident_rate, _, _) =
            vorbis_ident_info(ident).map_err(|reason| decode_error(&target, reason))?;
        assert_eq!(ident_channels, 1);
        assert_eq!(ident_rate, 44_100);
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn vorbis_bitrate_option_changes_encoded_size() -> Result<(), PipelineError> {
        let dir = temp_dir("vorbis-bitrate")?;
        let samples = sine(9_600, 44_100, 2);
        let encoded_bytes = |bps: u32| -> Result<u64, PipelineError> {
            let (track, packets) = encode_audio_track(
                &dir.join("x"),
                &samples,
                44_100,
                2,
                TargetAudioCodec::Vorbis,
                audio_config(TargetAudioCodec::Vorbis, bps, 44_100, 2)?,
            )?;
            assert_eq!(track.codec, VideoCodec::Vorbis);
            Ok(packets
                .iter()
                .map(|packet| u64::try_from(packet.data.len()).unwrap_or(u64::MAX))
                .sum())
        };
        // `vorbis-enc` ABR templates cap the per-channel bitrate well below
        // the domain's 512 kbps ceiling at 44.1 kHz stereo.
        let low = encoded_bytes(64_000)?;
        let high = encoded_bytes(192_000)?;
        assert!(
            low > 0 && high > low,
            "expected higher bitrate to emit more bytes: {low} vs {high}"
        );
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn vorbis_track_decodes_back_to_pcm() -> Result<(), PipelineError> {
        let dir = temp_dir("vorbis-roundtrip")?;
        let path = dir.join("x");
        let samples = sine(4_800, 48_000, 2);
        let (track, packets) = encode_audio_track(
            &path,
            &samples,
            48_000,
            2,
            TargetAudioCodec::Vorbis,
            default_config(TargetAudioCodec::Vorbis),
        )?;
        let (decoded, rate, channels) = decode_audio_track_pcm(&path, &track, &packets)?;
        assert_ne!(decoded.len(), 0);
        assert_eq!(rate, 48_000);
        assert_eq!(channels, 2);
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn webm_vp9_opus_to_avi_h264_mp3() -> Result<(), PipelineError> {
        let dir = temp_dir("webm-avi")?;
        let webm = dir.join("in.webm");
        let avi = dir.join("out.avi");
        make_source(
            &webm,
            VideoContainerFormat::Webm,
            VideoEncodeMode::CpuVp9 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            TargetAudioCodec::Opus,
        )?;
        transcode(
            &webm,
            &avi,
            VideoContainerFormat::Webm,
            VideoContainerFormat::Avi,
            VideoTrackAction::Encode(VideoEncodeMode::CpuH264 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            }),
            AudioTrackAction::Encode(default_config(TargetAudioCodec::Mp3)),
        )?;
        assert_output(
            &avi,
            VideoContainerFormat::Avi,
            VideoCodec::H264,
            VideoCodec::Mp3,
        )?;
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn avi_h264_mp3_to_ts_av1_aac() -> Result<(), PipelineError> {
        let dir = temp_dir("avi-ts")?;
        let avi = dir.join("in.avi");
        let ts = dir.join("out.ts");
        make_source(
            &avi,
            VideoContainerFormat::Avi,
            VideoEncodeMode::CpuH264 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            TargetAudioCodec::Mp3,
        )?;
        transcode(
            &avi,
            &ts,
            VideoContainerFormat::Avi,
            VideoContainerFormat::Ts,
            VideoTrackAction::Encode(VideoEncodeMode::CpuAv1 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            }),
            AudioTrackAction::Encode(default_config(TargetAudioCodec::Aac)),
        )?;
        assert_output(
            &ts,
            VideoContainerFormat::Ts,
            VideoCodec::Av1,
            VideoCodec::Aac,
        )?;
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn ts_av1_aac_to_mov_copy_video_mp3_audio() -> Result<(), PipelineError> {
        let dir = temp_dir("ts-mov")?;
        let ts = dir.join("in.ts");
        let mov = dir.join("out.mov");
        make_source(
            &ts,
            VideoContainerFormat::Ts,
            VideoEncodeMode::CpuAv1 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            TargetAudioCodec::Aac,
        )?;
        transcode(
            &ts,
            &mov,
            VideoContainerFormat::Ts,
            VideoContainerFormat::Mov,
            VideoTrackAction::Copy,
            AudioTrackAction::Encode(default_config(TargetAudioCodec::Mp3)),
        )?;
        assert_output(
            &mov,
            VideoContainerFormat::Mov,
            VideoCodec::Av1,
            VideoCodec::Mp3,
        )?;
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn streaming_aac_emits_packets_before_finish() -> Result<(), PipelineError> {
        let samples = sine(20_000, 48_000, 2);
        let (mut encoder, _) = StreamingAudioEncoder::start(
            TargetAudioCodec::Aac,
            2,
            48_000,
            default_config(TargetAudioCodec::Aac),
        )?;
        let packets = encoder.push(&samples)?;
        assert!(
            !packets.is_empty(),
            "streaming AAC must emit packets before finish so the bounded queue bounds memory"
        );
        assert!(packets.iter().all(|packet| packet.duration > 0));
        Ok(())
    }

    #[test]
    fn streaming_vorbis_accepts_a_push_larger_than_one_encoded_block() -> Result<(), PipelineError>
    {
        // A resample+remix step can hand over more than the old decoded-frame
        // cap in one push; it must be split, not rejected.
        let channels = 6_u16;
        let frames = VORBIS_FRAME_SAMPLE_CAP / usize::from(channels) + VORBIS_CHUNK_FRAMES;
        let samples = sine(frames, 48_000, channels);
        assert!(samples.len() > VORBIS_FRAME_SAMPLE_CAP);
        let (mut encoder, _) = StreamingAudioEncoder::start(
            TargetAudioCodec::Vorbis,
            channels,
            48_000,
            default_config(TargetAudioCodec::Vorbis),
        )?;
        let packets = encoder.push(&samples)?;
        assert!(
            !packets.is_empty(),
            "streaming Vorbis must emit packets from a multi-block push"
        );
        assert!(packets.iter().all(|packet| packet.duration > 0));
        Ok(())
    }

    #[test]
    fn resamples_the_whole_track_in_blocks() -> Result<(), PipelineError> {
        let samples = sine(8_000, 44_100, 2);
        let out = resample_interleaved(&samples, 2, 44_100, 48_000)?;
        let expected = 8_000_u64
            .saturating_mul(48_000)
            .saturating_div(44_100)
            .saturating_mul(2);
        assert_eq!(out.len(), usize::try_from(expected).unwrap_or(usize::MAX));
        assert!(out.iter().any(|sample| sample.abs() > 0.01));
        Ok(())
    }

    #[test]
    fn streaming_resampler_tail_matches_whole_track_frame_count() -> Result<(), PipelineError> {
        // A final partial block just past a block boundary must not extend the
        // audio timeline beyond the whole-track resampler's exact frame count.
        let input_frames = 209_920_usize;
        let samples = vec![0.25_f32; input_frames.saturating_mul(2)];
        let mut streaming = StreamingResampler::new(2, 44_100, 48_000)?;
        let mut out = streaming.push(&samples)?;
        out.extend(streaming.finish()?);
        let expected = usize::try_from(
            u64::try_from(input_frames)
                .unwrap_or(u64::MAX)
                .saturating_mul(48_000)
                .saturating_div(44_100)
                .saturating_mul(2),
        )
        .unwrap_or(usize::MAX);
        assert_eq!(out.len(), expected);
        assert_eq!(
            resample_interleaved(&samples, 2, 44_100, 48_000)?.len(),
            expected
        );
        Ok(())
    }

    fn expected_resampled_frames(input_frames: usize, from: u32, to: u32) -> usize {
        usize::try_from(
            u64::try_from(input_frames)
                .unwrap_or(u64::MAX)
                .saturating_mul(u64::from(to))
                / u64::from(from),
        )
        .unwrap_or(usize::MAX)
    }

    fn ramp(frames: usize, channels: usize) -> Vec<f32> {
        (0..frames.saturating_mul(channels))
            .map(|index| {
                let step = i16::try_from(index % 997).unwrap_or(0);
                f32::from(step) / 997.0_f32 - 0.5_f32
            })
            .collect()
    }

    /// Feeds the clip in caller-controlled chunk sizes so staging boundaries
    /// never line up with the resampler block by accident.
    fn stream_resampled(
        samples: &[f32],
        channels: usize,
        from: u32,
        to: u32,
        push_frames: &[usize],
    ) -> Result<Vec<f32>, PipelineError> {
        let mut resampler = StreamingResampler::new(channels, from, to)?;
        let total_frames = samples.len() / channels.max(1);
        let mut out = Vec::new();
        let mut offset = 0_usize;
        for &frames in push_frames {
            let count = frames.min(total_frames.saturating_sub(offset));
            let start = offset.saturating_mul(channels);
            let end = start.saturating_add(count.saturating_mul(channels));
            if let Some(chunk) = samples.get(start..end) {
                out.extend(resampler.push(chunk)?);
            }
            offset = offset.saturating_add(count);
        }
        if let Some(chunk) = samples.get(offset.saturating_mul(channels)..) {
            out.extend(resampler.push(chunk)?);
        }
        out.extend(resampler.finish()?);
        Ok(out)
    }

    #[test]
    fn resampling_same_rate_is_a_passthrough() -> Result<(), PipelineError> {
        let samples = ramp(2_000, 2);
        assert_eq!(resample_interleaved(&samples, 2, 48_000, 48_000)?, samples);
        let mut streaming = StreamingResampler::new(2, 48_000, 48_000)?;
        let mut out = streaming.push(&samples[..100])?;
        out.extend(streaming.push(&samples[100..])?);
        out.extend(streaming.finish()?);
        assert_eq!(out, samples);
        Ok(())
    }

    #[test]
    fn resampling_zero_sample_input_is_empty() -> Result<(), PipelineError> {
        for channels in [1_usize, 2, 6] {
            assert_eq!(
                resample_interleaved(&[], channels, 44_100, 48_000)?,
                Vec::<f32>::new(),
                "empty whole-track input must stay empty"
            );
            let mut streaming = StreamingResampler::new(channels, 44_100, 48_000)?;
            assert_eq!(streaming.push(&[])?, Vec::<f32>::new());
            assert_eq!(
                streaming.finish()?,
                Vec::<f32>::new(),
                "empty streamed input must stay empty"
            );
        }
        Ok(())
    }

    #[test]
    fn resampling_short_input_below_one_chunk_matches_whole_track() -> Result<(), PipelineError> {
        for channels in [1_usize, 2, 3, 6] {
            let frames = 137_usize;
            let samples = ramp(frames, channels);
            let whole = resample_interleaved(&samples, channels, 44_100, 48_000)?;
            let streamed = stream_resampled(&samples, channels, 44_100, 48_000, &[frames])?;
            assert_eq!(
                whole.len(),
                expected_resampled_frames(frames, 44_100, 48_000) * channels
            );
            assert_eq!(streamed, whole, "channels={channels}");
        }
        Ok(())
    }

    #[test]
    fn resampling_partial_final_block_has_exact_frame_count() -> Result<(), PipelineError> {
        let chunk = Fft::<f32>::new(44_100, 48_000, 1024, 2, FixedSync::Both)
            .map_err(|err| encode_error(Path::new("resampler"), err.to_string()))?
            .input_frames_next();
        assert!(chunk > 1, "resampler must have a usable input block");
        for frames in [
            chunk - 1,
            chunk,
            chunk + 1,
            chunk.saturating_mul(2),
            chunk.saturating_mul(2).saturating_add(137),
        ] {
            let samples = ramp(frames, 2);
            let whole = resample_interleaved(&samples, 2, 44_100, 48_000)?;
            let streamed = stream_resampled(
                &samples,
                2,
                44_100,
                48_000,
                &[chunk + 1, chunk.saturating_sub(1), 7],
            )?;
            assert_eq!(
                whole.len(),
                expected_resampled_frames(frames, 44_100, 48_000) * 2,
                "frames={frames}"
            );
            assert_eq!(streamed, whole, "frames={frames}");
        }
        Ok(())
    }

    #[test]
    fn resampling_multichannel_streaming_matches_whole_track() -> Result<(), PipelineError> {
        for channels in [1_usize, 2, 3, 5, 6, 8] {
            let frames = 5_000_usize;
            let samples = ramp(frames, channels);
            let whole = resample_interleaved(&samples, channels, 48_000, 44_100)?;
            let streamed =
                stream_resampled(&samples, channels, 48_000, 44_100, &[1, 997, 2_048, 13])?;
            assert_eq!(
                whole.len(),
                expected_resampled_frames(frames, 48_000, 44_100) * channels
            );
            assert_eq!(streamed, whole, "channels={channels}");
        }
        Ok(())
    }

    #[test]
    fn resampling_trims_startup_delay_from_the_front() -> Result<(), PipelineError> {
        let frames = 4_000_usize;
        let mut samples = vec![0.0_f32; frames];
        samples[0] = 1.0;
        let out = resample_interleaved(&samples, 1, 48_000, 96_000)?;
        assert_eq!(out.len(), frames * 2);
        let peak = out
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
            .map_or(0, |(index, _)| index);
        assert!(
            peak < 64,
            "impulse peak landed at frame {peak}; the resampler delay was not trimmed"
        );
        Ok(())
    }

    #[test]
    fn resampling_rejects_invalid_configuration() {
        assert!(resample_interleaved(&[0.0], 0, 44_100, 48_000).is_err());
        assert!(resample_interleaved(&[0.0], 2, 0, 48_000).is_err());
        assert!(resample_interleaved(&[0.0], 2, 44_100, 0).is_err());
        assert!(StreamingResampler::new(0, 44_100, 48_000).is_err());
        assert!(StreamingResampler::new(2, 0, 48_000).is_err());
        assert!(StreamingResampler::new(2, 44_100, 0).is_err());
    }

    #[test]
    fn pcm_decode_budget_tracks_live_samples_not_cumulative_throughput() -> Result<(), PipelineError>
    {
        let path = Path::new("pcm-budget");
        // Just before `emit_ready` drains, `chunk` holds one block and the
        // incoming slice holds another. Repeat for longer than a 48 kHz stereo
        // stream needs to cross 1 GiB cumulatively (~46 minutes); the cap must
        // stay bounded because every handed-off block left `chunk`.
        let live: Vec<f32> = vec![0.0; AUDIO_PCM_CHUNK_FRAMES * 2];
        let incoming = u64::try_from(live.len()).unwrap_or(u64::MAX);
        let slices = 46_u64
            .saturating_mul(60)
            .saturating_mul(48_000)
            .div_ceil(u64::try_from(AUDIO_PCM_CHUNK_FRAMES).unwrap_or(1));
        for _ in 0..slices {
            enforce_live_pcm_budget(path, &live, incoming)?;
        }
        assert!(
            enforce_live_pcm_budget(path, &[], u64::MAX).is_err(),
            "a genuinely oversized live slice must still be rejected"
        );
        Ok(())
    }

    // Non-empty PCM assertion documents the successful decode roundtrip.
    #[allow(clippy::assert_is_empty)]
    #[test]
    fn opus_tail_trims_to_the_input_duration() -> Result<(), PipelineError> {
        let dir = temp_dir("opus-tail")?;
        let samples = sine(1_000, 48_000, 2);
        let (track, packets) = encode_audio_track(
            &dir.join("x"),
            &samples,
            48_000,
            2,
            TargetAudioCodec::Opus,
            default_config(TargetAudioCodec::Opus),
        )?;
        assert!(
            packets.len() >= 2,
            "the encoder delay needs a padded packet"
        );
        let last = &packets[packets.len() - 1];
        assert!(last.duration > 0 && last.duration <= 20_000);
        assert_eq!(
            last.pts,
            20_000_i64 * i64::try_from(packets.len() - 1).unwrap_or(1)
        );
        let (decoded, _, _) = decode_audio_track_pcm(&dir.join("x"), &track, &packets)?;
        assert!(!decoded.is_empty());
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn rebase_copied_and_encoded_tracks_on_one_timeline() {
        let packet = |track: usize, start: i64, index: i64| ContainerPacket {
            track,
            pts: start.saturating_add(index.saturating_mul(20_000)),
            dts: start.saturating_add(index.saturating_mul(20_000)),
            duration: 20_000,
            keyframe: index == 0,
            data: Vec::new(),
        };
        let video_start = 1_000_000_i64;
        let audio_start = 250_000_i64;
        let mut audio = vec![packet(1, audio_start, 0), packet(1, audio_start, 1)];
        let source_video = vec![packet(0, video_start, 0), packet(0, video_start, 1)];
        let origin = audio
            .iter()
            .chain(source_video.iter())
            .map(|p| p.dts)
            .min()
            .unwrap_or(0);
        assert_eq!(origin, audio_start);

        rebase_source_packets(&mut audio, origin);
        let mut nvenc_video = source_video.clone();
        rebase_source_packets(&mut nvenc_video, origin);
        let mut cpu_video = vec![packet(0, 0, 0), packet(0, 0, 1)];
        rebase_encoded_packets(&mut cpu_video, video_start, origin);

        assert_eq!(audio[0].pts, 0);
        assert_eq!(audio[0].dts, 0);
        assert_eq!(nvenc_video[0].pts, 750_000);
        assert_eq!(cpu_video[0].pts, 750_000);
        assert_eq!(cpu_video[0].pts - audio[0].pts, 750_000);
        assert_eq!(cpu_video[1].pts, nvenc_video[1].pts);
    }

    #[test]
    fn copies_timed_text_with_the_packet_copy_path() -> Result<(), PipelineError> {
        let dir = temp_dir("timed-text")?;
        let target = dir.join("out.mov");
        let track = ContainerTrack {
            codec: VideoCodec::TimedText,
            width: None,
            height: None,
            channels: None,
            sample_rate: None,
            config: Vec::new(),
            timescale: US_TIMESCALE,
            presentation_edit: None,
            pcm_format: None,
        };
        let packets = vec![ContainerPacket {
            track: 0,
            pts: 1_000_000,
            dts: 1_000_000,
            duration: 500_000,
            keyframe: true,
            data: b"hello".to_vec(),
        }];
        let mut tracks = vec![None];
        let mut out = Vec::new();
        copy_track(&track, packets, 0, None, &target, &mut tracks, &mut out)?;
        assert_eq!(tracks.len(), 1);
        assert_eq!(
            tracks[0].as_ref().map(|track| track.codec),
            Some(VideoCodec::TimedText)
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].data, b"hello".to_vec());
        assert_eq!(out[0].pts, 1_000_000);
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn staging_name_is_short_and_cleans_up_after_a_failed_commit() -> Result<(), PipelineError> {
        let dir = temp_dir("staging")?;
        let long_target = dir.join(format!("{}.mov", "a".repeat(255)));
        let long_staging = StagingTarget::create_with_handle(&long_target, 3)?;
        let long_temp = long_staging.path().to_path_buf();
        let name = long_temp.file_name().and_then(|n| n.to_str()).unwrap_or("");
        assert!(long_temp.starts_with(&dir));
        assert!(name.len() < 64);
        drop(long_staging);
        assert!(!long_temp.exists());

        let target = dir.join("out.mov");
        let staging = StagingTarget::create_with_handle(&target, 4)?;
        let temp = staging.path().to_path_buf();
        {
            let _file = staging.file()?;
        }
        fs::create_dir(&target).map_err(|source| PipelineError::Io {
            path: target.clone(),
            source,
        })?;
        assert!(staging.commit().is_err());
        assert!(!temp.exists());

        fs::remove_dir(&target).map_err(|source| PipelineError::Io {
            path: target.clone(),
            source,
        })?;
        let staging = StagingTarget::create_with_handle(&target, 5)?;
        staging.commit()?;
        assert!(target.is_file());
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn batch_admission_stops_claiming_after_worker_and_startup_failures() {
        let mut admission = BatchAdmission::default();
        assert_eq!(admission.claim(), Some(0));
        assert_eq!(admission.claim(), Some(1));
        let error = |reason: &str| PipelineError::RemuxFailed {
            path: PathBuf::from("failed.mp4"),
            reason: reason.to_owned(),
        };
        admission.fail(error("first"));
        // A later failure must not overwrite the first, and neither failure
        // may let another worker claim the next item.
        admission.fail(error("second"));
        assert_eq!(admission.claim(), None);
        assert_eq!(admission.next, 2);
        assert!(matches!(
            admission.failure,
            Some(PipelineError::RemuxFailed { ref reason, .. }) if reason == "first"
        ));

        let mut admission = BatchAdmission::default();
        assert_eq!(admission.claim(), Some(0));
        admission.stop(); // spawn failure, no worker error to report
        assert_eq!(admission.claim(), None);
        assert!(admission.failure.is_none());
    }

    #[test]
    fn drift_free_aac_and_mp3_timestamps_stay_accurate_over_long_streams() {
        for rate in [44_100_u32, 48_000] {
            for frame_samples in [1_024_u64, 1_152] {
                let mut worst_error = 0_i128;
                for index in 0..100_000_u64 {
                    let offset = index.saturating_mul(frame_samples);
                    let pts = i128::from(sample_pts_us(offset, rate));
                    let exact = offset.saturating_mul(1_000_000) / u64::from(rate);
                    let error = (pts - i128::from(exact)).abs();
                    worst_error = worst_error.max(error);
                }
                assert!(
                    worst_error <= 1,
                    "drift at {rate} Hz / {frame_samples} samples: {worst_error} us"
                );
            }
        }
    }

    #[test]
    fn streaming_audio_worker_preserves_av_sync_offset_and_track_order() -> Result<(), PipelineError>
    {
        let dir = temp_dir("av-sync")?;
        let source = dir.join("in.webm");
        let target = dir.join("out.webm");
        make_source(
            &source,
            VideoContainerFormat::Webm,
            VideoEncodeMode::CpuVp9 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            TargetAudioCodec::Opus,
        )?;
        let mut media = demux_video(&source, VideoContainerFormat::Webm)?;
        for packet in &mut media.packets {
            let shift = if packet.track == 0 {
                1_000_000
            } else {
                250_000
            };
            packet.pts = packet.pts.saturating_add(shift);
            packet.dts = packet.dts.saturating_add(shift);
        }
        let file = File::create(&source).map_err(|err| PipelineError::Io {
            path: source.clone(),
            source: err,
        })?;
        mux_video(&file, &source, VideoContainerFormat::Webm, &media)?;
        drop(file);

        transcode(
            &source,
            &target,
            VideoContainerFormat::Webm,
            VideoContainerFormat::Webm,
            VideoTrackAction::Encode(VideoEncodeMode::CpuVp9 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            }),
            AudioTrackAction::Encode(default_config(TargetAudioCodec::Opus)),
        )?;
        let out = demux_video(&target, VideoContainerFormat::Webm)?;
        assert_eq!(out.tracks.len(), 2);
        assert_eq!(out.tracks[0].codec, VideoCodec::Vp9);
        assert_eq!(out.tracks[1].codec, VideoCodec::Opus);
        let video_start = out
            .packets
            .iter()
            .filter(|packet| packet.track == 0)
            .map(|packet| packet.pts)
            .min();
        let audio_start = out
            .packets
            .iter()
            .filter(|packet| packet.track == 1)
            .map(|packet| packet.pts)
            .min();
        assert_eq!(audio_start, Some(0));
        assert_eq!(video_start, Some(750_000));
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn audio_worker_resamples_streaming_without_drift() -> Result<(), PipelineError> {
        let dir = temp_dir("stream-resample")?;
        let source = dir.join("in.mp4");
        let target = dir.join("out.webm");
        let samples = sine(4_800, 44_100, 2);
        let (track, packets) = encode_audio_track(
            &source,
            &samples,
            44_100,
            2,
            TargetAudioCodec::Aac,
            default_config(TargetAudioCodec::Aac),
        )?;
        let media = ContainerMedia {
            tracks: vec![track],
            packets,
        };
        let file = File::create(&source).map_err(|err| PipelineError::Io {
            path: source.clone(),
            source: err,
        })?;
        mux_video(&file, &source, VideoContainerFormat::Mp4, &media)?;
        drop(file);

        transcode(
            &source,
            &target,
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Webm,
            VideoTrackAction::Copy,
            AudioTrackAction::Encode(default_config(TargetAudioCodec::Opus)),
        )?;
        let out = demux_video(&target, VideoContainerFormat::Webm)?;
        assert_eq!(out.tracks.len(), 1);
        assert_eq!(out.tracks[0].codec, VideoCodec::Opus);
        assert_eq!(out.tracks[0].sample_rate, Some(OPUS_RATE));
        let (decoded, rate, channels) =
            decode_audio_track_pcm(&target, &out.tracks[0], &out.packets)?;
        assert_eq!(rate, OPUS_RATE);
        assert_eq!(channels, 2);
        assert!(
            decoded.len() > 4_000 * 2 && decoded.len() < 8_000 * 2,
            "unexpected decoded length {}",
            decoded.len()
        );
        assert!(decoded.iter().any(|sample| sample.abs() > 0.01));
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn audio_only_copy_reports_no_video_decode_path() -> Result<(), PipelineError> {
        let dir = temp_dir("audio-only-decode-path")?;
        let source = dir.join("in.mp4");
        let target = dir.join("out.mp4");
        let (track, packets) = encode_audio_track(
            &source,
            &sine(4_800, 44_100, 2),
            44_100,
            2,
            TargetAudioCodec::Aac,
            default_config(TargetAudioCodec::Aac),
        )?;
        let media = ContainerMedia {
            tracks: vec![track],
            packets,
        };
        let file = File::create(&source).map_err(|err| PipelineError::Io {
            path: source.clone(),
            source: err,
        })?;
        mux_video(&file, &source, VideoContainerFormat::Mp4, &media)?;
        drop(file);

        let plan = VideoTranscodePlan::new(
            vec![
                VideoTranscodeItem::new(
                    source.clone(),
                    target.clone(),
                    VideoContainerFormat::Mp4,
                    VideoContainerFormat::Mp4,
                    streams(&source, VideoContainerFormat::Mp4)?,
                    VideoTrackAction::Copy,
                    AudioTrackAction::Copy,
                    VideoDecodeMode::Cpu,
                )
                .map_err(|err| PipelineError::RemuxFailed {
                    path: target.clone(),
                    reason: err.to_string(),
                })?,
            ],
            NonZeroUsize::MIN,
        )
        .map_err(|err| PipelineError::RemuxFailed {
            path: target.clone(),
            reason: err.to_string(),
        })?;
        let report = execute_video_transcode_batch(&plan)?;
        assert_eq!(report.decode_paths(), VideoDecodePaths::NoVideo);
        assert_eq!(report.decode_path(), "no video decoded");
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    const AUDIO_ERROR_CHILD_ENV: &str = "FASTCONV_AUDIO_ERROR_TEST_CHILD";
    const AUDIO_ERROR_SENTINEL_ENV: &str = "FASTCONV_AUDIO_ERROR_TEST_SENTINEL";

    fn read_child_log(log_path: &Path) -> String {
        match fs::read_to_string(log_path) {
            Ok(log) => log,
            Err(err) => format!("<child log unreadable: {err}>"),
        }
    }

    /// Runs the real assertions in-process in the child test binary. The
    /// sentinel is the child's proof that this exact test actually executed:
    /// libtest exits 0 even when `--exact` matches zero tests.
    fn run_audio_error_test_body() -> Result<(), PipelineError> {
        let dir = temp_dir("audio-error")?;
        let source = dir.join("in.mp4");
        let target = dir.join("out.mp4");
        let video = frames(4);
        let encoded = encode_video_cpu(
            &source,
            &video,
            VideoEncodeMode::CpuH264 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            0,
        )?;
        // Five channels are legal AAC but illegal Opus: the worker fails when
        // it initializes the target encoder from the first decoded chunk.
        let samples = sine(1_920, 48_000, 5);
        let (mut audio_track, mut audio_packets) = encode_audio_track(
            &source,
            &samples,
            48_000,
            5,
            TargetAudioCodec::Aac,
            default_config(TargetAudioCodec::Aac),
        )?;
        audio_track.channels = Some(5);
        for packet in &mut audio_packets {
            packet.track = 1;
        }
        let media = ContainerMedia {
            tracks: vec![encoded.track, audio_track],
            packets: encoded.packets.into_iter().chain(audio_packets).collect(),
        };
        let file = File::create(&source).map_err(|err| PipelineError::Io {
            path: source.clone(),
            source: err,
        })?;
        mux_video(&file, &source, VideoContainerFormat::Mp4, &media)?;
        drop(file);

        let result = transcode(
            &source,
            &target,
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mp4,
            VideoTrackAction::Encode(VideoEncodeMode::CpuH264 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            }),
            AudioTrackAction::Encode(default_config(TargetAudioCodec::Opus)),
        );
        assert!(result.is_err(), "unsupported channel count must fail");
        assert!(!target.exists());
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        if let Some(sentinel) = std::env::var_os(AUDIO_ERROR_SENTINEL_ENV) {
            File::create(Path::new(&sentinel)).map_err(|source| PipelineError::Io {
                path: PathBuf::from(&sentinel),
                source,
            })?;
        }
        Ok(())
    }

    fn finish_audio_error_test_child(
        status: ExitStatus,
        dir: &Path,
        log_path: &Path,
        sentinel_path: &Path,
    ) -> Result<(), PipelineError> {
        let sentinel_written = sentinel_path.exists();
        if !status.success() {
            let log = read_child_log(log_path);
            let cleanup = fs::remove_dir_all(dir);
            return Err(encode_error(
                log_path,
                format!(
                    "audio-error test child failed with {status}; sentinel={sentinel_written}, cleanup={cleanup:?}: {log}"
                ),
            ));
        }
        let cleanup = fs::remove_dir_all(dir);
        match cleanup {
            Ok(()) if sentinel_written => Ok(()),
            Ok(()) => Err(encode_error(
                log_path,
                format!(
                    "audio-error test child exited successfully but sentinel {} is missing (the test name or module path no longer matches)",
                    sentinel_path.display()
                ),
            )),
            Err(source) => Err(PipelineError::Io {
                path: dir.to_path_buf(),
                source,
            }),
        }
    }

    fn timeout_audio_error_test_child(
        child: &mut std::process::Child,
        dir: &Path,
        log_path: &Path,
    ) -> Result<(), PipelineError> {
        let kill_result = match child.kill() {
            Ok(()) => "ok".to_owned(),
            Err(err) => format!("failed: {err}"),
        };
        let wait_result = match child.wait() {
            Ok(status) => format!("exited with {status}"),
            Err(err) => format!("failed: {err}"),
        };
        let log = read_child_log(log_path);
        let cleanup = fs::remove_dir_all(dir);
        Err(encode_error(
            log_path,
            format!(
                "audio-error test child exceeded the 30s deadline; kill={kill_result}, wait={wait_result}, cleanup={cleanup:?}: {log}"
            ),
        ))
    }

    /// Parent supervision: a real child process owns the timeout, so a
    /// deadlock kills only that child instead of aborting the test harness.
    fn run_audio_error_test_parent() -> Result<(), PipelineError> {
        let dir = temp_dir("audio-error-parent")?;
        let log_path = dir.join("child.log");
        let sentinel_path = dir.join("child.sentinel");
        // Both child streams go to one file via cloned handles: no pipe buffer
        // can fill up and deadlock a verbose child before it reaches the timeout.
        let log_file = File::create(&log_path).map_err(|source| PipelineError::Io {
            path: log_path.clone(),
            source,
        })?;
        let stdout_file = log_file.try_clone().map_err(|source| PipelineError::Io {
            path: log_path.clone(),
            source,
        })?;
        let current_exe = std::env::current_exe().map_err(|source| PipelineError::Io {
            path: PathBuf::from("current test executable"),
            source,
        })?;
        let mut child = Command::new(current_exe)
            .args([
                "--exact",
                "video_transcode::tests::audio_worker_error_propagates_and_cancels_cleanly",
                "--nocapture",
            ])
            .env(AUDIO_ERROR_CHILD_ENV, "1")
            .env(AUDIO_ERROR_SENTINEL_ENV, &sentinel_path)
            .stdout(Stdio::from(stdout_file))
            .stderr(Stdio::from(log_file))
            .spawn()
            .map_err(|source| PipelineError::Io {
                path: PathBuf::from("audio-error test child"),
                source,
            })?;

        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    return finish_audio_error_test_child(status, &dir, &log_path, &sentinel_path);
                }
                Ok(None) => {}
                Err(source) => {
                    let kill_result = child.kill();
                    let wait_result = child.wait();
                    let log = read_child_log(&log_path);
                    let cleanup = fs::remove_dir_all(&dir);
                    return Err(encode_error(
                        &log_path,
                        format!(
                            "failed to poll audio-error test child: {source}; kill={kill_result:?}, wait={wait_result:?}, cleanup={cleanup:?}: {log}"
                        ),
                    ));
                }
            }
            if Instant::now() >= deadline {
                return timeout_audio_error_test_child(&mut child, &dir, &log_path);
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    #[test]
    fn audio_worker_error_propagates_and_cancels_cleanly() -> Result<(), PipelineError> {
        if std::env::var_os(AUDIO_ERROR_CHILD_ENV).is_some() {
            run_audio_error_test_body()
        } else {
            run_audio_error_test_parent()
        }
    }

    #[test]
    fn nvenc_source_rebase_preserves_inter_track_presentation_offset() {
        // MPEG-TS style timestamps: audio starts at 1.40 s, video presents 80 ms
        // later (PTS 1.48 s) while its decode preroll starts at DTS 1.40 s.
        // Rebasing both tracks by the same file-level origin must keep the
        // 80 ms A/V presentation offset intact.
        let mut audio = vec![ContainerPacket {
            track: 1,
            pts: 1_400_000,
            dts: 1_400_000,
            duration: 21_333,
            keyframe: true,
            data: Vec::new(),
        }];
        let mut nvenc_video = vec![ContainerPacket {
            track: 0,
            pts: 1_480_000,
            dts: 1_446_667,
            duration: 16_667,
            keyframe: true,
            data: Vec::new(),
        }];
        let origin = 1_400_000_i64;
        rebase_source_packets(&mut audio, origin);
        rebase_source_packets(&mut nvenc_video, origin);
        assert_eq!(audio[0].pts, 0);
        assert_eq!(nvenc_video[0].pts, 80_000);
        assert_eq!(nvenc_video[0].dts, 46_667);
    }

    fn write_edited_h264_aac_mp4(
        path: &Path,
        video_edit: &containr::EditList,
    ) -> Result<(), PipelineError> {
        let map_err = |err: containr::Error| PipelineError::RemuxFailed {
            path: path.to_path_buf(),
            reason: err.to_string(),
        };
        let encoded_video = encode_video_cpu(
            path,
            &frames(12),
            VideoEncodeMode::CpuH264 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            1,
        )?;
        let (audio_track, audio_packets) = encode_audio_track(
            path,
            &sine(19_200, 48_000, 2),
            48_000,
            2,
            TargetAudioCodec::Aac,
            default_config(TargetAudioCodec::Aac),
        )?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|err| PipelineError::Io {
                path: path.to_path_buf(),
                source: err,
            })?;
        let options = containr::mp4::MuxOptions {
            brand: containr::mp4::Brand::Isom,
            faststart: true,
            movie_timescale: US_TIMESCALE,
            ..containr::mp4::MuxOptions::default()
        };
        let mut muxer = containr::mp4::Muxer::new(file, options).map_err(map_err)?;
        let audio_id = muxer
            .add_track(containr::mp4::TrackSpec::new(
                containr::TrackKind::Audio,
                US_TIMESCALE,
                containr::mp4::SampleEntrySpec::Audio {
                    codec: containr::mp4::AudioCodec::Aac,
                    config: crate::video_bitstream::build_aac_esds(&audio_track.config),
                    channels: 2,
                    sample_rate: 48_000,
                },
            ))
            .map_err(map_err)?;
        let video_id = muxer
            .add_track(containr::mp4::TrackSpec::new(
                containr::TrackKind::Video,
                US_TIMESCALE,
                containr::mp4::SampleEntrySpec::Visual {
                    codec: containr::mp4::VisualCodec::Avc,
                    config: encoded_video.track.config.clone(),
                    width: encoded_video.track.width.unwrap_or(64),
                    height: encoded_video.track.height.unwrap_or(64),
                },
            ))
            .map_err(map_err)?;
        for (id, list) in [
            (audio_id, &audio_packets),
            (video_id, &encoded_video.packets),
        ] {
            for packet in list {
                muxer
                    .write_packet(
                        id,
                        &containr::mp4::PacketIn {
                            data: &packet.data,
                            pts: packet.pts,
                            dts: packet.dts,
                            duration: Some(packet.duration),
                            sync: packet.keyframe,
                        },
                    )
                    .map_err(map_err)?;
            }
        }
        muxer.set_edit_list(video_id, video_edit).map_err(map_err)?;
        muxer.finish().map_err(map_err)?;
        Ok(())
    }

    #[test]
    fn mp4_empty_edit_preroll_survives_transcode_rebase() -> Result<(), PipelineError> {
        let dir = temp_dir("mp4-edit-rebase")?;
        let source = dir.join("in.mp4");
        let target = dir.join("out.mp4");

        // Video carries an 80 ms empty edit plus a 120 ms media start while raw
        // video packets begin at 0 (preroll).
        let video_edit = containr::EditList {
            movie_timescale: US_TIMESCALE,
            entries: vec![
                containr::EditEntry {
                    segment_duration: 80_000,
                    media_time: containr::EditEntry::EMPTY_MEDIA_TIME,
                    media_rate: containr::EditEntry::NORMAL_RATE,
                },
                containr::EditEntry {
                    segment_duration: 120_000,
                    media_time: 120_000,
                    media_rate: containr::EditEntry::NORMAL_RATE,
                },
            ],
        };
        write_edited_h264_aac_mp4(&source, &video_edit)?;

        // Re-encoding audio triggers `needs_rebase = true` while video is copied
        // with its negative-PTS preroll packets and edit interval.
        transcode(
            &source,
            &target,
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mp4,
            VideoTrackAction::Copy,
            AudioTrackAction::Encode(default_config(TargetAudioCodec::Aac)),
        )?;

        let out_file = File::open(&target).map_err(|err| PipelineError::Io {
            path: target.clone(),
            source: err,
        })?;
        let inspected =
            containr::mp4::Demuxer::open(out_file).map_err(|err| PipelineError::RemuxFailed {
                path: target.clone(),
                reason: err.to_string(),
            })?;
        let out_audio = inspected
            .tracks()
            .iter()
            .find(|t| t.kind == containr::TrackKind::Audio)
            .ok_or_else(|| PipelineError::RemuxFailed {
                path: target.clone(),
                reason: "missing audio track".to_owned(),
            })?;
        let out_video = inspected
            .tracks()
            .iter()
            .find(|t| t.kind == containr::TrackKind::Video)
            .ok_or_else(|| PipelineError::RemuxFailed {
                path: target.clone(),
                reason: "missing video track".to_owned(),
            })?;
        assert!(
            out_audio.edit_list.entries.is_empty(),
            "audio must stay at movie time 0 after transcode rebase"
        );
        assert_eq!(out_video.edit_list.entries.len(), 2);
        assert!(out_video.edit_list.entries[0].is_empty());
        assert_eq!(out_video.edit_list.entries[0].segment_duration, 80_000);
        assert_eq!(out_video.edit_list.entries[1].media_time, 120_000);
        assert_eq!(out_video.edit_list.entries[1].segment_duration, 120_000);

        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[cfg(feature = "nvenc")]
    fn nvenc_sized_frames(width: u32, height: u32, count: usize) -> Vec<Yuv420Frame> {
        let width_usize = usize::try_from(width).unwrap_or(0);
        let height_usize = usize::try_from(height).unwrap_or(0);
        (0..count)
            .map(|index| {
                let y = (0..width_usize.saturating_mul(height_usize))
                    .map(|sample| {
                        u8::try_from(24 + (sample + index.saturating_mul(11)) % 180).unwrap_or(24)
                    })
                    .collect();
                let chroma = (width_usize / 2).saturating_mul(height_usize / 2);
                Yuv420Frame {
                    width,
                    height,
                    y,
                    u: vec![110; chroma],
                    v: vec![140; chroma],
                    pts_us: i64::try_from(index).unwrap_or(0).saturating_mul(40_000),
                    duration_us: 40_000,
                    color: YuvColor::default(),
                }
            })
            .collect()
    }

    #[cfg(feature = "nvenc")]
    fn assert_forced_nvenc_mkv_b_frames_roundtrip(
        source: &Path,
        media: &ContainerMedia,
        forced_target: &Path,
        name: &str,
        codec: VideoCodec,
        mode: VideoEncodeMode,
    ) -> Result<(), PipelineError> {
        if !crate::gpu::nvenc::codec_supports_b_frames(codec)? {
            return Ok(());
        }
        // Verify MKV mux + reopen deterministically with `disableBadapt = 1`
        // so B-frame PTS inversions are guaranteed on B-frame-capable GPUs.
        let (forced_stream, _, _) =
            crate::gpu::nvenc::encode_video_nvenc_streaming_forced_b_frames(
                source,
                &media.tracks[0],
                &media.packets,
                mode,
                0,
                || Ok(()),
            )?;
        let forced_file = File::create(forced_target).map_err(|err| PipelineError::Io {
            path: forced_target.to_path_buf(),
            source: err,
        })?;
        let forced_media = ContainerMedia {
            tracks: vec![forced_stream.track],
            packets: forced_stream.packets,
        };
        mux_video(
            &forced_file,
            forced_target,
            VideoContainerFormat::Mkv,
            &forced_media,
        )?;
        drop(forced_file);
        let forced_reopened = demux_video(forced_target, VideoContainerFormat::Mkv)?;
        assert_eq!(forced_reopened.packets.len(), 24);
        let mut forced_max_pts: Option<i64> = None;
        let mut forced_inversion = false;
        for (index, packet) in forced_reopened.packets.iter().enumerate() {
            assert!(packet.dts <= packet.pts);
            if index > 0 {
                assert!(packet.dts > forced_reopened.packets[index - 1].dts);
            }
            if let Some(prev_max) = forced_max_pts
                && packet.pts < prev_max
            {
                forced_inversion = true;
            }
            forced_max_pts = Some(forced_max_pts.map_or(packet.pts, |m| m.max(packet.pts)));
        }
        assert!(
            forced_inversion,
            "{name}: forced B-frame MKV must retain PTS inversions after reopen"
        );
        let forced_decoded = decode_video_track(
            forced_target,
            &forced_reopened.tracks[0],
            &forced_reopened.packets,
        )?;
        assert_eq!(forced_decoded.len(), 24);
        for (index, frame) in forced_decoded.iter().enumerate() {
            if index > 0 {
                assert!(
                    frame.pts_us > forced_decoded[index - 1].pts_us,
                    "{name}: forced B-frame MKV frames must decode in display order"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "nvenc")]
    #[test]
    fn nvenc_mkv_encode_and_reopen_preserves_decode_order() -> Result<(), PipelineError> {
        if cudarc::driver::CudaContext::new(0).is_err() {
            return Ok(());
        }
        let dir = temp_dir("nvenc-mkv-bframes")?;
        let source = dir.join("in.mp4");
        // NVENC HEVC requires at least 145x49; 256x144 with 24 frames exercises
        // real B-frames across both H.264 and HEVC.
        let video = nvenc_sized_frames(256, 144, 24);
        let encoded = encode_video_cpu(
            &source,
            &video,
            VideoEncodeMode::CpuH264 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            0,
        )?;
        let media = ContainerMedia {
            tracks: vec![encoded.track],
            packets: encoded.packets,
        };
        let file = File::create(&source).map_err(|err| PipelineError::Io {
            path: source.clone(),
            source: err,
        })?;
        mux_video(&file, &source, VideoContainerFormat::Mp4, &media)?;
        drop(file);

        for (name, codec, mode) in [
            (
                "out_h264.mkv",
                VideoCodec::H264,
                VideoEncodeMode::NvencH264 {
                    rate_control: fastconv_domain::VideoRateControl::DEFAULT,
                },
            ),
            (
                "out_hevc.mkv",
                VideoCodec::Hevc,
                VideoEncodeMode::NvencHevc {
                    rate_control: fastconv_domain::VideoRateControl::DEFAULT,
                },
            ),
        ] {
            let target = dir.join(name);
            transcode(
                &source,
                &target,
                VideoContainerFormat::Mp4,
                VideoContainerFormat::Mkv,
                VideoTrackAction::Encode(mode),
                AudioTrackAction::Copy,
            )?;
            let reopened = demux_video(&target, VideoContainerFormat::Mkv)?;
            assert_eq!(reopened.tracks.len(), 1);
            assert_eq!(reopened.packets.len(), 24);
            let mut max_pts_seen: Option<i64> = None;
            let mut has_pts_inversion = false;
            for (index, packet) in reopened.packets.iter().enumerate() {
                assert!(packet.dts <= packet.pts);
                if index > 0 {
                    assert!(packet.dts > reopened.packets[index - 1].dts);
                }
                if let Some(prev_max) = max_pts_seen
                    && packet.pts < prev_max
                {
                    has_pts_inversion = true;
                }
                max_pts_seen = Some(max_pts_seen.map_or(packet.pts, |m| m.max(packet.pts)));
            }
            let any_dts_lag = reopened
                .packets
                .iter()
                .any(|packet| packet.dts < packet.pts);
            assert_eq!(
                any_dts_lag, has_pts_inversion,
                "{name}: reopened MKV DTS lag must match actual PTS inversions"
            );
            let decoded = decode_video_track(&target, &reopened.tracks[0], &reopened.packets)?;
            assert_eq!(decoded.len(), 24);
            for (index, frame) in decoded.iter().enumerate() {
                if index > 0 {
                    assert!(
                        frame.pts_us > decoded[index - 1].pts_us,
                        "{name}: reopened MKV frames must decode in display order"
                    );
                }
            }
            assert_forced_nvenc_mkv_b_frames_roundtrip(
                &source,
                &media,
                &dir.join(format!("forced_{name}")),
                name,
                codec,
                mode,
            )?;
        }
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    fn make_animation_source(
        path: &Path,
        format: VideoContainerFormat,
        width: u32,
        height: u32,
        count: u32,
    ) -> Result<(), PipelineError> {
        let quality = fastconv_domain::VideoQuality::DEFAULT;
        let file = File::create(path).map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        crate::video_anim::encode_animation_stream(
            format,
            quality,
            2,
            AnimationLoop::Infinite,
            &file,
            |sink| {
                for index in 0..count {
                    let color: [u8; 3] = match index % 3 {
                        0 => [220, 24, 24],
                        1 => [24, 220, 24],
                        _ => [24, 24, 220],
                    };
                    let rgb = color.repeat(
                        usize::try_from(width).unwrap_or(0) * usize::try_from(height).unwrap_or(0),
                    );
                    sink(
                        AnimFrameInfo {
                            width,
                            height,
                            pts_us: i64::from(index).saturating_mul(100_000),
                            duration_us: 100_000,
                            channels: 3,
                        },
                        &rgb,
                    )?;
                }
                Ok(())
            },
        )?;
        drop(file);
        Ok(())
    }

    fn assert_animation(
        path: &Path,
        format: VideoContainerFormat,
        width: u32,
        height: u32,
        count: usize,
        expected_duration: u64,
    ) -> Result<(), PipelineError> {
        let (media_codec, media_width, media_height, packets) = match format {
            VideoContainerFormat::Gif => {
                let media = crate::video_anim::demux_gif_metadata(path)?;
                (
                    media.tracks[0].codec,
                    media.tracks[0].width,
                    media.tracks[0].height,
                    media.packets,
                )
            }
            VideoContainerFormat::Webp => {
                let media = crate::video_anim::demux_webp_metadata(path)?;
                (
                    media.tracks[0].codec,
                    media.tracks[0].width,
                    media.tracks[0].height,
                    media.packets,
                )
            }
            _ => return Err(decode_error(path, "expected an animation container")),
        };
        assert_eq!(media_codec, format.default_video_codec().as_video_codec());
        assert_eq!(media_width, u16::try_from(width).ok());
        assert_eq!(media_height, u16::try_from(height).ok());
        assert_eq!(packets.len(), count, "animation packet count");
        let durations: Vec<u64> = packets.iter().map(|p| p.duration).collect();
        assert!(
            durations.iter().all(|&d| d == expected_duration),
            "unexpected animation frame durations: {durations:?}"
        );

        let mut decoded = 0_usize;
        match format {
            VideoContainerFormat::Gif => {
                crate::video_anim::decode_gif_frames_streaming_rgb(path, |info, rgb| {
                    assert_eq!((info.width, info.height), (width, height));
                    assert_eq!(
                        rgb.len(),
                        usize::try_from(width).unwrap_or(0)
                            * usize::try_from(height).unwrap_or(0)
                            * 3
                    );
                    decoded = decoded.saturating_add(1);
                    Ok(())
                })?;
            }
            VideoContainerFormat::Webp => {
                crate::video_anim::decode_webp_frames_streaming_rgb(path, |info, rgb| {
                    assert_eq!((info.width, info.height), (width, height));
                    assert_eq!(
                        rgb.len(),
                        usize::try_from(width).unwrap_or(0)
                            * usize::try_from(height).unwrap_or(0)
                            * 3
                    );
                    decoded = decoded.saturating_add(1);
                    Ok(())
                })?;
            }
            _ => return Err(decode_error(path, "expected an animation container")),
        }
        assert_eq!(decoded, count, "decoded animation frame count");
        Ok(())
    }

    fn cpu_video_mode(container: VideoContainerFormat) -> VideoEncodeMode {
        let rate_control = fastconv_domain::VideoRateControl::DEFAULT;
        if container == VideoContainerFormat::Webm {
            VideoEncodeMode::CpuVp9 { rate_control }
        } else {
            VideoEncodeMode::CpuH264 { rate_control }
        }
    }

    #[test]
    fn all_video_containers_to_gif_and_webp_drop_audio() -> Result<(), PipelineError> {
        let dir = temp_dir("anim-targets")?;
        for container in [
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Webm,
            VideoContainerFormat::Mov,
            VideoContainerFormat::Avi,
            VideoContainerFormat::Ts,
        ] {
            let source = dir.join(format!("src.{}", container.extension()));
            let audio = if container == VideoContainerFormat::Webm {
                TargetAudioCodec::Opus
            } else {
                TargetAudioCodec::Aac
            };
            make_source(&source, container, cpu_video_mode(container), audio)?;
            for (format, mode) in [
                (
                    VideoContainerFormat::Gif,
                    VideoEncodeMode::CpuGif {
                        quality: fastconv_domain::VideoQuality::DEFAULT,
                    },
                ),
                (
                    VideoContainerFormat::Webp,
                    VideoEncodeMode::CpuWebp {
                        quality: fastconv_domain::VideoQuality::DEFAULT,
                    },
                ),
            ] {
                let target = dir.join(format!(
                    "{}_to.{}",
                    container.extension(),
                    format.extension()
                ));
                transcode(
                    &source,
                    &target,
                    container,
                    format,
                    VideoTrackAction::Encode(mode),
                    AudioTrackAction::Drop,
                )?;
                assert_animation(&target, format, 64, 48, 4, 20_000)?;
            }
        }
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn animations_to_all_video_containers_on_cpu() -> Result<(), PipelineError> {
        let dir = temp_dir("anim-sources")?;
        for input in [VideoContainerFormat::Gif, VideoContainerFormat::Webp] {
            let source = dir.join(format!("anim.{}", input.extension()));
            make_animation_source(&source, input, 64, 48, 3)?;
            for container in [
                VideoContainerFormat::Mp4,
                VideoContainerFormat::Mkv,
                VideoContainerFormat::Webm,
                VideoContainerFormat::Mov,
                VideoContainerFormat::Avi,
                VideoContainerFormat::Ts,
            ] {
                let mode = cpu_video_mode(container);
                let target = dir.join(format!(
                    "{}_to.{}",
                    input.extension(),
                    container.extension()
                ));
                transcode(
                    &source,
                    &target,
                    input,
                    container,
                    VideoTrackAction::Encode(mode),
                    AudioTrackAction::Drop,
                )?;
                let media = demux_video(&target, container)?;
                assert_eq!(media.tracks.len(), 1, "audio must not appear");
                assert_eq!(media.tracks[0].codec, mode.codec().as_video_codec());
                let frames = media.packets.iter().filter(|p| p.track == 0).count();
                assert!(
                    frames >= 3,
                    "expected decoded animation frames, got {frames}"
                );
            }
        }
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn gif_and_webp_bidirectional_roundtrip() -> Result<(), PipelineError> {
        let dir = temp_dir("anim-bidirectional")?;
        for (input, output, mode) in [
            (
                VideoContainerFormat::Gif,
                VideoContainerFormat::Webp,
                VideoEncodeMode::CpuWebp {
                    quality: fastconv_domain::VideoQuality::DEFAULT,
                },
            ),
            (
                VideoContainerFormat::Webp,
                VideoContainerFormat::Gif,
                VideoEncodeMode::CpuGif {
                    quality: fastconv_domain::VideoQuality::DEFAULT,
                },
            ),
        ] {
            let source = dir.join(format!("src.{}", input.extension()));
            make_animation_source(&source, input, 32, 24, 3)?;
            let target = dir.join(format!("out.{}", output.extension()));
            transcode(
                &source,
                &target,
                input,
                output,
                VideoTrackAction::Encode(mode),
                AudioTrackAction::Drop,
            )?;
            assert_animation(&target, output, 32, 24, 3, 100_000)?;
        }
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[cfg(feature = "nvenc")]
    #[test]
    fn animation_sources_use_real_nvenc_hardware() -> Result<(), PipelineError> {
        if cudarc::driver::CudaContext::new(0).is_err() {
            return Ok(());
        }
        let dir = temp_dir("anim-nvenc")?;
        for (input, container, mode) in [
            (
                VideoContainerFormat::Gif,
                VideoContainerFormat::Mp4,
                VideoEncodeMode::NvencH264 {
                    rate_control: fastconv_domain::VideoRateControl::DEFAULT,
                },
            ),
            (
                VideoContainerFormat::Webp,
                VideoContainerFormat::Mkv,
                VideoEncodeMode::NvencHevc {
                    rate_control: fastconv_domain::VideoRateControl::DEFAULT,
                },
            ),
        ] {
            let source = dir.join(format!("src.{}", input.extension()));
            make_animation_source(&source, input, 256, 144, 3)?;
            let target = dir.join(format!("out.{}", container.extension()));
            let report = transcode(
                &source,
                &target,
                input,
                container,
                VideoTrackAction::Encode(mode),
                AudioTrackAction::Drop,
            )?;
            assert!(
                report.hardware_device().is_some(),
                "NVENC must report the actual hardware device"
            );
            let media = demux_video(&target, container)?;
            assert_eq!(media.tracks[0].codec, mode.codec().as_video_codec());
            assert!(media.packets.len() >= 3);
        }
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[cfg(feature = "nvenc")]
    #[test]
    fn hwdecode_auto_reports_nvdec_nvenc_hardware_path() -> Result<(), PipelineError> {
        use fastconv_domain::{HardwareBackend, HardwareDecodeBackend};

        if cudarc::driver::CudaContext::new(0).is_err() {
            return Ok(());
        }

        let dir = temp_dir("nvdec-nvenc-cli")?;
        let source = dir.join("in.mp4");
        let target = dir.join("out.mp4");
        // NVENC requires at least 320x240; the shared 64x48 test frames are
        // intentionally below the hardware minimum.
        let video_frames: Vec<Yuv420Frame> = (0..8_u32)
            .map(|index| Yuv420Frame {
                width: 320,
                height: 240,
                y: vec![u8::try_from(48 + index).unwrap_or(48); 320 * 240],
                u: vec![96; 160 * 120],
                v: vec![128; 160 * 120],
                pts_us: i64::from(index) * 40_000,
                duration_us: 40_000,
                color: YuvColor::default(),
            })
            .collect();
        let encoded = encode_video_cpu(
            &source,
            &video_frames,
            VideoEncodeMode::CpuH264 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            0,
        )?;
        let media = ContainerMedia {
            tracks: vec![encoded.track],
            packets: encoded.packets,
        };
        let file = File::create(&source).map_err(|source_err| PipelineError::Io {
            path: source.clone(),
            source: source_err,
        })?;
        mux_video(&file, &source, VideoContainerFormat::Mp4, &media)?;
        drop(file);
        let mode = VideoEncodeMode::NvencH264 {
            rate_control: fastconv_domain::VideoRateControl::DEFAULT,
        };
        let decode_mode = HardwareDecodeBackend::Auto
            .resolve(VideoCodec::H264, HardwareBackend::Gpu(GpuBackend::Nvenc))
            .map_err(|err| PipelineError::DecodeFailed {
                path: source.clone(),
                reason: err.to_string(),
            })?;
        assert_eq!(decode_mode, VideoDecodeMode::Nvdec);
        let plan = VideoTranscodePlan::new(
            vec![
                VideoTranscodeItem::new(
                    source.clone(),
                    target.clone(),
                    VideoContainerFormat::Mp4,
                    VideoContainerFormat::Mp4,
                    streams(&source, VideoContainerFormat::Mp4)?,
                    VideoTrackAction::Encode(mode),
                    AudioTrackAction::Drop,
                    decode_mode,
                )
                .map_err(|err| PipelineError::DecodeFailed {
                    path: source.clone(),
                    reason: err.to_string(),
                })?,
            ],
            NonZeroUsize::MIN,
        )
        .map_err(|err| PipelineError::DecodeFailed {
            path: source.clone(),
            reason: err.to_string(),
        })?;
        let report = execute_video_transcode_batch(&plan)?;
        assert_eq!(report.decode_path(), "gpu:nvdec+nvenc");
        assert!(report.hardware_device().is_some());
        let media = demux_video(&target, VideoContainerFormat::Mp4)?;
        assert_eq!(media.tracks[0].codec, VideoCodec::H264);
        assert!(
            media.packets.len() >= 8,
            "NVDEC -> NVENC output lost frames: {} packets",
            media.packets.len()
        );
        fs::remove_dir_all(&dir).map_err(|source_err| PipelineError::Io {
            path: dir.clone(),
            source: source_err,
        })?;
        Ok(())
    }

    #[test]
    fn presentation_edit_filters_clips_and_rebases_animation_frames() {
        let frame = |pts_us: i64| DecodedFrameInfo {
            width: 16,
            height: 16,
            pts_us,
            duration_us: 40_000,
            color: crate::video_codecs::YuvColor::default(),
        };
        let bounded = Some(TrackPresentation {
            start_us: 0,
            duration_us: Some(50_000),
        });
        let kept: Vec<DecodedFrameInfo> = [0, 40_000, 80_000]
            .into_iter()
            .filter_map(|pts| apply_frame_presentation_edit(bounded, frame(pts)))
            .collect();
        assert_eq!(
            kept.iter()
                .map(|f| (f.pts_us, f.duration_us))
                .collect::<Vec<_>>(),
            vec![(0, 40_000), (40_000, 10_000)],
            "last retained frame duration must be clipped to the edit end"
        );

        let shifted = Some(TrackPresentation {
            start_us: 40_000,
            duration_us: Some(80_000),
        });
        let kept: Vec<DecodedFrameInfo> = [0, 40_000, 80_000, 120_000]
            .into_iter()
            .filter_map(|pts| apply_frame_presentation_edit(shifted, frame(pts)))
            .collect();
        assert_eq!(
            kept.iter()
                .map(|f| (f.pts_us, f.duration_us))
                .collect::<Vec<_>>(),
            vec![(0, 40_000), (40_000, 40_000)],
            "frames outside the visible interval must be dropped and rebased"
        );

        let open_ended = Some(TrackPresentation {
            start_us: 40_000,
            duration_us: None,
        });
        let kept: Vec<DecodedFrameInfo> = [0, 40_000]
            .into_iter()
            .filter_map(|pts| apply_frame_presentation_edit(open_ended, frame(pts)))
            .collect();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].pts_us, 0);
        assert_eq!(kept[0].duration_us, 40_000);
    }

    #[test]
    fn presentation_edit_keeps_pre_edit_h264_reference_packets_for_decoding()
    -> Result<(), PipelineError> {
        let path = Path::new("h264-edit-p-frame.mp4");
        let frames: Vec<Yuv420Frame> = (0..4_u8)
            .map(|idx| Yuv420Frame {
                width: 32,
                height: 32,
                y: vec![40_u8.saturating_add(idx * 40); 32 * 32],
                u: vec![128; 16 * 16],
                v: vec![128; 16 * 16],
                pts_us: i64::from(idx) * 40_000,
                duration_us: 40_000,
                color: YuvColor::default(),
            })
            .collect();
        let stream = encode_video_cpu(
            path,
            &frames,
            VideoEncodeMode::CpuH264 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            0,
        )?;
        assert!(stream.packets[0].keyframe, "first H.264 packet must be IDR");
        assert!(
            !stream.packets[1].keyframe,
            "second H.264 packet must be a P-frame referencing frame 0"
        );

        // Edit starts on P-frame at 40 ms and ends at 90 ms (clipping frame 2 to 10 ms).
        let edit = Some(TrackPresentation {
            start_us: 40_000,
            duration_us: Some(50_000),
        });
        let selected = select_packets_for_presentation_edit(edit, &stream.packets);
        assert_eq!(
            selected.iter().map(|p| p.pts).collect::<Vec<_>>(),
            vec![0, 40_000, 80_000],
            "pre-edit keyframe must stay in the decode span while post-edit tail is excluded"
        );

        let mut emitted = Vec::new();
        decode_video_track_streaming(path, &stream.track, selected, |info, pack| {
            let Some(edited) = apply_frame_presentation_edit(edit, info) else {
                return Ok(());
            };
            let pitch = usize::try_from(edited.width).unwrap_or(0);
            let mut nv12 = vec![0_u8; nv12_buffer_len(edited.width, edited.height, pitch)?];
            pack(&mut nv12, pitch)?;
            emitted.push((edited.pts_us, edited.duration_us, nv12[0]));
            Ok(())
        })?;

        assert_eq!(
            emitted.len(),
            2,
            "pre-edit reference frame must not be emitted"
        );
        assert_eq!(emitted[0].0, 0);
        assert_eq!(emitted[0].1, 40_000);
        assert_eq!(emitted[1].0, 40_000);
        assert_eq!(emitted[1].1, 10_000);
        assert!(
            emitted[0].2.abs_diff(80) <= 10 && emitted[1].2.abs_diff(120) <= 10,
            "decoded P-frames must carry frames 1 and 2 luma, got {emitted:?}"
        );
        Ok(())
    }

    fn vb(bps: u32) -> Result<VideoBitrate, PipelineError> {
        VideoBitrate::new(bps).map_err(|err| PipelineError::RemuxFailed {
            path: PathBuf::new(),
            reason: err.to_string(),
        })
    }

    fn audio_config(
        codec: TargetAudioCodec,
        bps: u32,
        rate: u32,
        channels: u16,
    ) -> Result<AudioEncodeConfig, PipelineError> {
        let bitrate = AudioBitrate::new(bps).map_err(|err| PipelineError::RemuxFailed {
            path: PathBuf::new(),
            reason: err.to_string(),
        })?;
        let options = AudioEncodeOptions::new(
            Some(bitrate),
            Some(
                AudioSampleRate::new(rate).map_err(|err| PipelineError::RemuxFailed {
                    path: PathBuf::new(),
                    reason: err.to_string(),
                })?,
            ),
            Some(
                AudioChannels::new(channels).map_err(|err| PipelineError::RemuxFailed {
                    path: PathBuf::new(),
                    reason: err.to_string(),
                })?,
            ),
        );
        AudioEncodeConfig::new(codec, options).map_err(|err| PipelineError::RemuxFailed {
            path: PathBuf::new(),
            reason: err.to_string(),
        })
    }

    fn video_only_source(
        path: &Path,
        codec: TargetVideoCodec,
        count: usize,
    ) -> Result<(), PipelineError> {
        let mode = match codec {
            TargetVideoCodec::H264 => VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            TargetVideoCodec::Av1 => VideoEncodeMode::CpuAv1 {
                rate_control: VideoRateControl::DEFAULT,
            },
            TargetVideoCodec::Vp9 => VideoEncodeMode::CpuVp9 {
                rate_control: VideoRateControl::DEFAULT,
            },
            other => {
                return Err(PipelineError::RemuxFailed {
                    path: path.to_path_buf(),
                    reason: format!("{other} is not a CPU bitrate-test source"),
                });
            }
        };
        let encoded = encode_video_cpu(path, &frames(count), mode, 0)?;
        let media = ContainerMedia {
            tracks: vec![encoded.track],
            packets: encoded.packets,
        };
        let file = File::create(path).map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        mux_video(&file, path, VideoContainerFormat::Mp4, &media)?;
        drop(file);
        Ok(())
    }

    fn long_audio_source(path: &Path, audio: TargetAudioCodec) -> Result<(), PipelineError> {
        let encoded = encode_video_cpu(
            path,
            &frames(4),
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
        )?;
        let samples = sine(48_000, 48_000, 2);
        let (audio_track, mut audio_packets) =
            encode_audio_track(path, &samples, 48_000, 2, audio, default_config(audio))?;
        for packet in &mut audio_packets {
            packet.track = 1;
        }
        let media = ContainerMedia {
            tracks: vec![encoded.track, audio_track],
            packets: encoded.packets.into_iter().chain(audio_packets).collect(),
        };
        let file = File::create(path).map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        mux_video(&file, path, VideoContainerFormat::Mp4, &media)?;
        drop(file);
        Ok(())
    }

    #[test]
    fn channel_remix_handles_every_supported_ratio() -> Result<(), PipelineError> {
        let mono = [0.25_f32, 0.75];
        let stereo = [0.2_f32, 0.8, 0.4, 0.6];
        let surround = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        assert_eq!(mix_to_channels(&stereo, 2, 2)?.as_ref(), stereo);
        let stereo_from_mono = mix_to_channels(&mono, 1, 2)?;
        assert_eq!(stereo_from_mono.as_ref(), [0.25, 0.25, 0.75, 0.75]);
        let mono_from_stereo = mix_to_channels(&stereo, 2, 1)?;
        assert_eq!(mono_from_stereo.as_ref(), [0.5, 0.5]);
        let six_from_stereo = mix_to_channels(&stereo, 2, 6)?;
        assert_eq!(six_from_stereo.len(), 12);
        assert_eq!(six_from_stereo[0], 0.2);
        assert_eq!(six_from_stereo[1], 0.8);
        assert_eq!(six_from_stereo[2], 0.5);
        let two_from_six = mix_to_channels(&surround, 6, 2)?;
        // The average of all six channels (0.35) is folded into the stereo
        // pair, so channels 2-5 reach the output and the peak stays bounded.
        assert!((two_from_six[0] - 0.225).abs() < 0.000_1);
        assert!((two_from_six[1] - 0.275).abs() < 0.000_1);
        let center_only = [0.0_f32, 0.0, 1.0, 0.0, 0.0, 0.0];
        let folded = mix_to_channels(&center_only, 6, 2)?;
        assert!(folded[0] > 0.0 && folded[1] > 0.0);
        assert_eq!(mix_to_channels(&[1.0_f32; 6], 6, 2)?.as_ref(), [1.0, 1.0]);
        let one_from_six = mix_to_channels(&surround, 6, 1)?;
        assert!((one_from_six[0] - 0.35).abs() < 0.000_1);
        assert!(mix_to_channels(&[1.0], 0, 2).is_err());
        assert!(mix_to_channels(&[1.0], 2, 0).is_err());
        Ok(())
    }

    #[test]
    fn non_stereo_reductions_preserve_every_source_channel() -> Result<(), PipelineError> {
        for to in [3_usize, 5] {
            for source_channel in 0..6 {
                let mut frame = [0.0_f32; 6];
                frame[source_channel] = 1.0;
                let mixed = mix_to_channels(&frame, 6, to)?;
                assert_eq!(mixed.len(), to);
                assert!(
                    mixed.iter().any(|sample| *sample > 0.0),
                    "6->{to} dropped source channel {source_channel}"
                );
            }
        }
        assert_eq!(
            mix_to_channels(&[1.0_f32; 6], 6, 3)?.as_ref(),
            [1.0, 1.0, 1.0]
        );
        Ok(())
    }

    #[test]
    fn mp3_effective_bitrate_band_is_checked_before_encoder_creation() -> Result<(), PipelineError>
    {
        let config =
            |bitrate: Option<u32>, rate: Option<u32>| -> Result<AudioEncodeConfig, PipelineError> {
                let to_err = |err: DomainError| PipelineError::RemuxFailed {
                    path: PathBuf::new(),
                    reason: err.to_string(),
                };
                let bitrate = bitrate
                    .map(|bps| AudioBitrate::new(bps).map_err(to_err))
                    .transpose()?;
                let rate = rate
                    .map(|hz| AudioSampleRate::new(hz).map_err(to_err))
                    .transpose()?;
                AudioEncodeConfig::new(
                    TargetAudioCodec::Mp3,
                    AudioEncodeOptions::new(bitrate, rate, None),
                )
                .map_err(to_err)
            };
        // Rate-only override inherits the 192 kbps default: out of band below 32 kHz.
        let rate_only = config(None, Some(22_050))?;
        assert!(checked_mp3_bitrate_kbps(rate_only, 22_050, Path::new("mp3")).is_err());
        // Bitrate-only override inherits a 48 kHz source: below the floor.
        let bitrate_only = config(Some(16_000), None)?;
        assert!(checked_mp3_bitrate_kbps(bitrate_only, 48_000, Path::new("mp3")).is_err());
        // Valid pairs keep working in both bands.
        let valid_low = config(Some(128_000), Some(22_050))?;
        assert_eq!(
            checked_mp3_bitrate_kbps(valid_low, 22_050, Path::new("mp3"))?,
            128
        );
        let valid_high = config(Some(64_000), Some(48_000))?;
        assert_eq!(
            checked_mp3_bitrate_kbps(valid_high, 48_000, Path::new("mp3"))?,
            64
        );
        Ok(())
    }

    #[test]
    fn cpu_vbitrate_roundtrips_and_scales_stream_size() -> Result<(), PipelineError> {
        let dir = temp_dir("cpu-vbitrate")?;
        let frame_count = 24;
        for codec in [
            TargetVideoCodec::H264,
            TargetVideoCodec::Av1,
            TargetVideoCodec::Vp9,
        ] {
            let source = dir.join(format!("{codec:?}-in.mp4").to_ascii_lowercase());
            video_only_source(&source, codec, frame_count)?;
            let mut sizes = Vec::new();
            for (label, bps) in [("low", 100_000_u32), ("high", 2_000_000)] {
                let target = dir.join(format!("{codec:?}-{label}.mp4").to_ascii_lowercase());
                let mode = match codec {
                    TargetVideoCodec::H264 => VideoEncodeMode::CpuH264 {
                        rate_control: VideoRateControl::Bitrate(vb(bps)?),
                    },
                    TargetVideoCodec::Av1 => VideoEncodeMode::CpuAv1 {
                        rate_control: VideoRateControl::Bitrate(vb(bps)?),
                    },
                    TargetVideoCodec::Vp9 => VideoEncodeMode::CpuVp9 {
                        rate_control: VideoRateControl::Bitrate(vb(bps)?),
                    },
                    other => {
                        return Err(PipelineError::RemuxFailed {
                            path: target,
                            reason: format!("unexpected {other}"),
                        });
                    }
                };
                transcode(
                    &source,
                    &target,
                    VideoContainerFormat::Mp4,
                    VideoContainerFormat::Mp4,
                    VideoTrackAction::Encode(mode),
                    AudioTrackAction::Drop,
                )?;
                let media = demux_video(&target, VideoContainerFormat::Mp4)?;
                let decoded = decode_video_track(&target, &media.tracks[0], &media.packets)?;
                assert_eq!(
                    decoded.len(),
                    frame_count,
                    "{codec:?} at {bps} bps must decode every frame"
                );
                let bytes: usize = media.packets.iter().map(|packet| packet.data.len()).sum();
                assert!(bytes > 0, "{codec:?} at {bps} bps produced no bitstream");
                sizes.push(bytes);
            }
            assert!(
                sizes[0] < sizes[1],
                "{codec:?}: low bitrate produced {} bytes, high produced {} bytes",
                sizes[0],
                sizes[1]
            );
        }
        Ok(())
    }

    #[test]
    fn audio_options_roundtrip_decodable_tracks_and_scale_bitrate() -> Result<(), PipelineError> {
        let dir = temp_dir("audio-options")?;
        for (codec, rate, channels, low, high) in [
            (
                TargetAudioCodec::Aac,
                44_100_u32,
                1_u16,
                64_000_u32,
                256_000_u32,
            ),
            (TargetAudioCodec::Opus, 48_000, 2, 64_000, 256_000),
            (TargetAudioCodec::Mp3, 44_100, 1, 64_000, 256_000),
        ] {
            let source = dir.join(format!("{codec:?}-in.mp4").to_ascii_lowercase());
            long_audio_source(&source, codec)?;
            let mut sizes = Vec::new();
            for (label, bps) in [("low", low), ("high", high)] {
                let target = dir.join(format!("{codec:?}-{label}.mp4").to_ascii_lowercase());
                let config = audio_config(codec, bps, rate, channels)?;
                transcode(
                    &source,
                    &target,
                    VideoContainerFormat::Mp4,
                    VideoContainerFormat::Mp4,
                    VideoTrackAction::Copy,
                    AudioTrackAction::Encode(config),
                )?;
                let media = demux_video(&target, VideoContainerFormat::Mp4)?;
                let index = media
                    .tracks
                    .iter()
                    .position(|track| track.codec.is_audio())
                    .ok_or_else(|| PipelineError::RemuxFailed {
                        path: target.clone(),
                        reason: "output has no audio track".to_owned(),
                    })?;
                let track = &media.tracks[index];
                assert_eq!(
                    track.sample_rate,
                    Some(rate),
                    "{codec:?} at {bps} bps sample rate"
                );
                assert_eq!(
                    track.channels,
                    Some(channels),
                    "{codec:?} at {bps} bps channels"
                );
                let packets: Vec<_> = media
                    .packets
                    .iter()
                    .filter(|packet| packet.track == index)
                    .cloned()
                    .collect();
                let (pcm, decoded_rate, decoded_channels) =
                    decode_audio_track_pcm(&target, track, &packets)?;
                assert_eq!(decoded_rate, rate, "{codec:?} decoded sample rate");
                assert_eq!(decoded_channels, channels, "{codec:?} decoded channels");
                assert!(!pcm.is_empty(), "{codec:?} decoded to no PCM");
                let bytes: usize = packets.iter().map(|packet| packet.data.len()).sum();
                assert!(bytes > 0, "{codec:?} at {bps} bps produced no packets");
                sizes.push(bytes);
            }
            assert!(
                sizes[0] < sizes[1],
                "{codec:?}: {low} bps produced {} bytes, {high} bps produced {} bytes",
                sizes[0],
                sizes[1]
            );
        }
        Ok(())
    }

    fn h264_video() -> VideoTrackAction {
        VideoTrackAction::Encode(VideoEncodeMode::CpuH264 {
            rate_control: fastconv_domain::VideoRateControl::DEFAULT,
        })
    }

    fn audio_facts(path: &Path, format: VideoContainerFormat) -> Result<(u32, u16), PipelineError> {
        let media = demux_video(path, format)?;
        let audio = media
            .tracks
            .iter()
            .find(|track| track.codec.is_audio())
            .ok_or_else(|| decode_error(path, "no audio track after transcode"))?;
        Ok((
            audio
                .sample_rate
                .ok_or_else(|| decode_error(path, "no sample rate"))?,
            audio
                .channels
                .ok_or_else(|| decode_error(path, "no channels"))?,
        ))
    }

    #[test]
    fn aac_source_transcodes_to_flac_in_mkv_mp4_and_mov() -> Result<(), PipelineError> {
        let dir = temp_dir("to-flac")?;
        let source = dir.join("aac.mp4");
        make_source(
            &source,
            VideoContainerFormat::Mp4,
            VideoEncodeMode::CpuH264 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            TargetAudioCodec::Aac,
        )?;
        for (name, format) in [
            ("out.mkv", VideoContainerFormat::Mkv),
            ("out.mp4", VideoContainerFormat::Mp4),
            ("out.mov", VideoContainerFormat::Mov),
        ] {
            let target = dir.join(name);
            transcode(
                &source,
                &target,
                VideoContainerFormat::Mp4,
                format,
                h264_video(),
                AudioTrackAction::Encode(default_config(TargetAudioCodec::Flac)),
            )?;
            assert_output(&target, format, VideoCodec::H264, VideoCodec::Flac)?;
            let (rate, channels) = audio_facts(&target, format)?;
            assert_eq!((rate, channels), (48_000, 2), "FLAC facts in {name}");
            let media = demux_video(&target, format)?;
            let audio = media
                .tracks
                .iter()
                .find(|track| track.codec == VideoCodec::Flac)
                .ok_or_else(|| decode_error(&target, "no FLAC track"))?;
            // Fixed-size frames need equal STREAMINFO bounds; unequal bounds
            // make readers treat frame numbers as absolute sample offsets.
            let bounds: [u8; 4] = audio
                .config
                .get(..4)
                .ok_or_else(|| decode_error(&target, "short FLAC STREAMINFO"))?
                .try_into()
                .map_err(|_| decode_error(&target, "invalid FLAC STREAMINFO bounds"))?;
            let min_block = u16::from_be_bytes([bounds[0], bounds[1]]);
            let max_block = u16::from_be_bytes([bounds[2], bounds[3]]);
            assert_eq!(
                (usize::from(min_block), usize::from(max_block)),
                (FLAC_CHUNK_FRAMES, FLAC_CHUNK_FRAMES),
                "FLAC STREAMINFO block bounds must both be {FLAC_CHUNK_FRAMES} in {name}"
            );
            let packets: Vec<_> = media
                .packets
                .iter()
                .filter(|packet| packet.track == 1)
                .cloned()
                .collect();
            let (pcm, decoded_rate, decoded_channels) =
                decode_audio_track_pcm(&target, audio, &packets)?;
            assert!(!pcm.is_empty(), "FLAC decoded to no PCM in {name}");
            assert_eq!((decoded_rate, decoded_channels), (48_000, 2));
        }
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn aac_source_transcodes_to_pcm_in_mkv_avi_and_mov() -> Result<(), PipelineError> {
        let dir = temp_dir("to-pcm")?;
        let source = dir.join("aac.mp4");
        make_source(
            &source,
            VideoContainerFormat::Mp4,
            VideoEncodeMode::CpuH264 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            TargetAudioCodec::Aac,
        )?;
        for (name, format) in [
            ("out.mkv", VideoContainerFormat::Mkv),
            ("out.avi", VideoContainerFormat::Avi),
            ("out.mov", VideoContainerFormat::Mov),
        ] {
            let target = dir.join(name);
            transcode(
                &source,
                &target,
                VideoContainerFormat::Mp4,
                format,
                h264_video(),
                AudioTrackAction::Encode(default_config(TargetAudioCodec::Pcm)),
            )?;
            assert_output(&target, format, VideoCodec::H264, VideoCodec::Pcm)?;
            let (rate, channels) = audio_facts(&target, format)?;
            assert_eq!((rate, channels), (48_000, 2), "PCM facts in {name}");
            let media = demux_video(&target, format)?;
            let audio = media
                .tracks
                .iter()
                .find(|track| track.codec == VideoCodec::Pcm)
                .ok_or_else(|| decode_error(&target, "no PCM track"))?;
            assert_eq!(
                audio.pcm_format,
                Some(PcmFormat::S16Le),
                "PCM format in {name}"
            );
            let packets: Vec<_> = media
                .packets
                .iter()
                .filter(|packet| packet.track == 1)
                .cloned()
                .collect();
            let (pcm, decoded_rate, decoded_channels) =
                decode_audio_track_pcm(&target, audio, &packets)?;
            assert!(!pcm.is_empty(), "PCM decoded to no samples in {name}");
            assert_eq!((decoded_rate, decoded_channels), (48_000, 2));
        }
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn flac_and_pcm_sources_transcode_back_to_aac_and_opus() -> Result<(), PipelineError> {
        let dir = temp_dir("from-lossless")?;
        let flac = dir.join("flac.mkv");
        let pcm = dir.join("pcm.mkv");
        for (path, codec) in [
            (&flac, TargetAudioCodec::Flac),
            (&pcm, TargetAudioCodec::Pcm),
        ] {
            make_source(
                path,
                VideoContainerFormat::Mkv,
                VideoEncodeMode::CpuH264 {
                    rate_control: fastconv_domain::VideoRateControl::DEFAULT,
                },
                codec,
            )?;
        }

        let aac_target = dir.join("back-aac.mp4");
        transcode(
            &flac,
            &aac_target,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Mp4,
            h264_video(),
            AudioTrackAction::Encode(default_config(TargetAudioCodec::Aac)),
        )?;
        assert_output(
            &aac_target,
            VideoContainerFormat::Mp4,
            VideoCodec::H264,
            VideoCodec::Aac,
        )?;

        let opus_target = dir.join("back-opus.webm");
        transcode(
            &pcm,
            &opus_target,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Webm,
            VideoTrackAction::Encode(VideoEncodeMode::CpuVp9 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            }),
            AudioTrackAction::Encode(default_config(TargetAudioCodec::Opus)),
        )?;
        assert_output(
            &opus_target,
            VideoContainerFormat::Webm,
            VideoCodec::Vp9,
            VideoCodec::Opus,
        )?;
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn flac_and_pcm_honour_sample_rate_and_channel_options() -> Result<(), PipelineError> {
        let dir = temp_dir("lossless-options")?;
        let source = dir.join("aac.mp4");
        make_source(
            &source,
            VideoContainerFormat::Mp4,
            VideoEncodeMode::CpuH264 {
                rate_control: fastconv_domain::VideoRateControl::DEFAULT,
            },
            TargetAudioCodec::Aac,
        )?;
        let options = AudioEncodeOptions::new(
            None,
            Some(
                AudioSampleRate::new(44_100)
                    .map_err(|err| encode_error(&source, err.to_string()))?,
            ),
            Some(AudioChannels::new(1).map_err(|err| encode_error(&source, err.to_string()))?),
        );
        for (name, format, codec, expected) in [
            (
                "flac.mkv",
                VideoContainerFormat::Mkv,
                TargetAudioCodec::Flac,
                VideoCodec::Flac,
            ),
            (
                "pcm.mov",
                VideoContainerFormat::Mov,
                TargetAudioCodec::Pcm,
                VideoCodec::Pcm,
            ),
        ] {
            let target = dir.join(name);
            let config = AudioEncodeConfig::new(codec, options)
                .map_err(|err| encode_error(&target, err.to_string()))?;
            transcode(
                &source,
                &target,
                VideoContainerFormat::Mp4,
                format,
                h264_video(),
                AudioTrackAction::Encode(config),
            )?;
            assert_output(&target, format, VideoCodec::H264, expected)?;
            let (rate, channels) = audio_facts(&target, format)?;
            assert_eq!((rate, channels), (44_100, 1), "options in {name}");
        }
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn lossless_audio_remux_roundtrips() -> Result<(), PipelineError> {
        let dir = temp_dir("lossless-remux")?;
        let flac = dir.join("in.flac.mkv");
        let pcm = dir.join("in.pcm.mkv");
        for (path, codec) in [
            (&flac, TargetAudioCodec::Flac),
            (&pcm, TargetAudioCodec::Pcm),
        ] {
            make_source(
                path,
                VideoContainerFormat::Mkv,
                VideoEncodeMode::CpuH264 {
                    rate_control: fastconv_domain::VideoRateControl::DEFAULT,
                },
                codec,
            )?;
        }
        let flac_mov = dir.join("flac.mov");
        transcode(
            &flac,
            &flac_mov,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Mov,
            VideoTrackAction::Copy,
            AudioTrackAction::Copy,
        )?;
        assert_output(
            &flac_mov,
            VideoContainerFormat::Mov,
            VideoCodec::H264,
            VideoCodec::Flac,
        )?;
        let pcm_avi = dir.join("pcm.avi");
        transcode(
            &pcm,
            &pcm_avi,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Avi,
            VideoTrackAction::Copy,
            AudioTrackAction::Copy,
        )?;
        assert_output(
            &pcm_avi,
            VideoContainerFormat::Avi,
            VideoCodec::H264,
            VideoCodec::Pcm,
        )?;
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn flac_streaminfo_declares_equal_fixed_block_bounds() -> Result<(), PipelineError> {
        let info = flac_streaminfo(48_000, 2, 0)?;
        let bytes = flac_streaminfo_bytes(&info)?;
        let min = u16::from_be_bytes([bytes[0], bytes[1]]);
        let max = u16::from_be_bytes([bytes[2], bytes[3]]);
        assert_eq!(
            (usize::from(min), usize::from(max)),
            (FLAC_CHUNK_FRAMES, FLAC_CHUNK_FRAMES),
            "fixed-block FLAC must declare equal min/max block sizes"
        );
        Ok(())
    }

    #[test]
    fn flac_decode_accepts_full_metadata_chain_config() -> Result<(), PipelineError> {
        let dir = temp_dir("flac-full-metadata")?;
        let path = dir.join("x");
        let samples = sine(4_800, 48_000, 2);
        let (mut track, packets) = encode_audio_track(
            &path,
            &samples,
            48_000,
            2,
            TargetAudioCodec::Flac,
            default_config(TargetAudioCodec::Flac),
        )?;
        let si: [u8; 34] = track
            .config
            .as_slice()
            .try_into()
            .map_err(|_| decode_error(&path, "encode helper emitted no bare STREAMINFO"))?;
        parse_flac_streaminfo(&si).map_err(|reason| decode_error(&path, reason))?;

        // Canonical MKV/MOV configs keep every native block. Put a minimal
        // VORBIS_COMMENT after STREAMINFO: the decoder must still find the
        // validated STREAMINFO body, while remuxing must keep the whole chain.
        let mut config = Vec::new();
        config.extend_from_slice(b"fLaC");
        config.push(0x00); // STREAMINFO, more blocks follow
        config.extend_from_slice(&[0, 0, 34]);
        config.extend_from_slice(&si);
        config.push(0x84); // VORBIS_COMMENT, last block
        config.extend_from_slice(&[0, 0, 8]);
        config.extend_from_slice(&[0_u8; 8]); // empty vendor, zero comments
        track.config = config.clone();

        let (decoded, rate, channels) = decode_audio_track_pcm(&path, &track, &packets)?;
        assert_ne!(decoded, [] as [f32; 0]);
        assert_eq!((rate, channels), (48_000, 2));
        assert_eq!(
            track.config, config,
            "remux must keep the full metadata chain"
        );
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn lossless_whole_track_encoders_resample_rate_overrides() -> Result<(), PipelineError> {
        let dir = temp_dir("lossless-resample")?;
        let path = dir.join("x");
        let input_frames = 4_800_usize;
        let expected = expected_resampled_frames(input_frames, 48_000, 44_100);
        assert_eq!(expected, 4_410);
        let samples = sine(input_frames, 48_000, 2);
        let options = || -> Result<AudioEncodeOptions, PipelineError> {
            Ok(AudioEncodeOptions::new(
                None,
                Some(
                    AudioSampleRate::new(44_100)
                        .map_err(|err| encode_error(&path, err.to_string()))?,
                ),
                None,
            ))
        };

        let flac_config = AudioEncodeConfig::new(TargetAudioCodec::Flac, options()?)
            .map_err(|err| encode_error(&path, err.to_string()))?;
        let (flac, _) = encode_flac_track(&path, &samples, 48_000, 2, flac_config)?;
        assert_eq!(flac.sample_rate, Some(44_100));
        let packed = u64::from_be_bytes(
            flac.config
                .get(10..18)
                .ok_or_else(|| encode_error(&path, "short FLAC STREAMINFO"))?
                .try_into()
                .map_err(|_| encode_error(&path, "invalid FLAC STREAMINFO bounds"))?,
        );
        assert_eq!(
            packed & 0xF_FFFF_FFFF,
            u64::try_from(expected).unwrap_or(u64::MAX),
            "FLAC STREAMINFO must declare the resampled frame count"
        );

        let pcm_config = AudioEncodeConfig::new(TargetAudioCodec::Pcm, options()?)
            .map_err(|err| encode_error(&path, err.to_string()))?;
        let (pcm, packets) = encode_pcm_track(&path, &samples, 48_000, 2, pcm_config)?;
        assert_eq!(pcm.sample_rate, Some(44_100));
        let frames: usize = packets
            .iter()
            .map(|packet| packet.data.len())
            .sum::<usize>()
            / 4;
        assert_eq!(frames, expected, "PCM packets must hold resampled frames");
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn s16_quantization_roundtrips_both_signed_endpoints() {
        // -1.0 decodes from S16Le -32768 and must re-encode to -32768 exactly;
        // the positive endpoint is 32767/32768 (i16 has no +32768).
        let positive = 32_767.0_f32 / 32_768.0_f32;
        assert_eq!(to_s16_i32(-1.0_f32), -32_768);
        assert_eq!(to_s16_i32(positive), 32_767);
        assert_eq!(to_s16_i32(1.0_f32), 32_767);

        // PCM output and FLAC frames both consume these i32 samples directly,
        // so the endpoints survive either output path.
        let mut raw = Vec::with_capacity(4);
        raw.extend_from_slice(&(-32_768_i16).to_le_bytes());
        raw.extend_from_slice(&32_767_i16.to_le_bytes());
        assert_eq!(pcm_to_s16le(&[-1.0_f32, positive]), raw);
        assert_eq!(
            pcm_bytes_to_f32(&raw, PcmFormat::S16Le),
            [-1.0_f32, positive]
        );
    }

    #[test]
    fn pcm_large_packet_streams_bounded_blocks() -> Result<(), PipelineError> {
        let dir = temp_dir("pcm-large-packet")?;
        let path = dir.join("x");
        let channels = 1_u16;
        let blocks = 8_usize;
        let frames = AUDIO_PCM_CHUNK_FRAMES.saturating_mul(blocks) + 13;
        let mut data = Vec::with_capacity(frames.saturating_mul(2));
        for frame in 0..frames {
            let raw = if frame % 2 == 0 {
                -32_768_i16
            } else {
                32_767_i16
            };
            data.extend_from_slice(&raw.to_le_bytes());
        }
        let track = ContainerTrack {
            codec: VideoCodec::Pcm,
            width: None,
            height: None,
            channels: Some(channels),
            sample_rate: Some(48_000),
            config: Vec::new(),
            timescale: US_TIMESCALE,
            presentation_edit: None,
            pcm_format: Some(PcmFormat::S16Le),
        };
        let packets = vec![ContainerPacket {
            track: 0,
            pts: 0,
            dts: 0,
            duration: 0,
            keyframe: true,
            data,
        }];
        let mut block_lens = Vec::new();
        let mut decoded = Vec::new();
        decode_audio_track_chunks(
            &path,
            &track,
            &packets,
            &AtomicBool::new(false),
            |chunk, rate, decoded_channels| {
                assert_eq!(rate, 48_000);
                assert_eq!(decoded_channels, channels);
                block_lens.push(chunk.len());
                decoded.extend_from_slice(chunk);
                Ok(())
            },
        )?;
        let ready = AUDIO_PCM_CHUNK_FRAMES.saturating_mul(usize::from(channels));
        assert_eq!(block_lens.len(), blocks + 1);
        assert!(block_lens[..blocks].iter().all(|&len| len == ready));
        assert_eq!(block_lens[blocks], 13);
        assert_eq!(decoded.len(), frames);
        assert_eq!(decoded[0], -1.0_f32);
        assert_eq!(decoded[1], 32_767.0_f32 / 32_768.0_f32);
        fs::remove_dir_all(&dir).map_err(|source| PipelineError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(())
    }
}
