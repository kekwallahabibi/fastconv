//! Native video + audio transcoding across all supported containers.
//!
//! Source packets are demuxed once into the canonical representation, each
//! track is either copied or decoded/re-encoded by a pure-Rust codec backend,
//! then the tracks are interleaved and muxed through a crash-safe staging
//! file. NVENC modes are routed to the GPU backend with no silent CPU fallback.

use std::fs::{self, File, OpenOptions};
use std::io::Cursor;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

use fastconv_domain::{
    AudioTrackAction, GpuBackend, HardwareBackend, TargetAudioCodec, VideoCodec, VideoEncodeMode,
    VideoTrackAction, VideoTranscodeItem, VideoTranscodePlan, VideoTranscodeReport,
};
use opus_pure::{Application, MAX_PACKET_BYTES, OpusEncoder};
use rubato::{FftFixedInOut, Resampler};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::formats::TrackType;
use symphonia::core::formats::probe::Hint;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

use crate::PipelineError;
use crate::video_bitstream::{default_opus_head, wrap_adts};
use crate::video_codecs::{decode_video_track, encode_video_cpu};
use crate::video_container::{
    ContainerPacket, ContainerTrack, US_TIMESCALE, demux_video, mux_video,
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

/// Linear resampling of interleaved `f32` PCM, used when the source rate is
/// not accepted by the target encoder. One resampler instance per call keeps
/// the working set bounded to a handful of planar chunks.
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
    let mut resampler = FftFixedInOut::<f32>::new(
        usize::try_from(from_rate).unwrap_or(usize::MAX),
        usize::try_from(to_rate).unwrap_or(usize::MAX),
        1024,
        channels,
    )
    .map_err(|err| {
        encode_error(
            Path::new("resampler"),
            format!("failed to create resampler from {from_rate} Hz to {to_rate} Hz: {err}"),
        )
    })?;
    let mut planar = Vec::with_capacity(channels);
    for channel in 0..channels {
        planar.push(
            samples
                .iter()
                .skip(channel)
                .step_by(channels)
                .copied()
                .collect::<Vec<f32>>(),
        );
    }
    let out = resampler
        .process(&planar, None)
        .map_err(|err| encode_error(Path::new("resampler"), format!("resampling failed: {err}")))?;
    let delay = resampler.output_delay();
    let frames = out.first().map_or(0, Vec::len);
    let end = frames.saturating_sub(delay);
    let mut interleaved = Vec::with_capacity(end.saturating_mul(channels));
    for frame in delay..frames {
        for plane in &out {
            if let Some(&sample) = plane.get(frame) {
                interleaved.push(sample);
            }
        }
    }
    Ok(interleaved)
}

/// Decodes one audio track to interleaved `f32` PCM at its native rate.
// One branch per codec keeps rate/channel handling explicit.
#[allow(clippy::too_many_lines)]
pub(crate) fn decode_audio_track_pcm(
    path: &Path,
    track: &ContainerTrack,
    packets: &[ContainerPacket],
) -> Result<(Vec<f32>, u32, u16), PipelineError> {
    let ordered = decode_order(packets);
    let channels = track.channels.unwrap_or(2).max(1);
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
            let mut pcm = Vec::new();
            let mut block = vec![0.0_f32; OPUS_FRAME.saturating_mul(usize::from(channels))];
            for packet in ordered {
                let count = decoder
                    .decode(&packet.data, OPUS_FRAME, &mut block)
                    .map_err(|err| decode_error(path, format!("Opus decode failed: {err}")))?;
                pcm.extend_from_slice(
                    &block[..count.saturating_mul(usize::from(channels)).min(block.len())],
                );
            }
            let skip = pre_skip.saturating_mul(usize::from(channels));
            let pcm = if skip >= pcm.len() {
                Vec::new()
            } else {
                pcm[skip..].to_vec()
            };
            if pcm.is_empty() {
                return Err(decode_error(path, "Opus track decoded to no audio"));
            }
            Ok((pcm, OPUS_RATE, channels))
        }
        VideoCodec::Aac => {
            let mut decoder = rusty_aac::AacDecoder::new();
            let mut pcm = Vec::new();
            let mut rate = 0_u32;
            let mut decoded_channels = 0_u16;
            for packet in ordered {
                let adts = wrap_adts(&track.config, &packet.data)
                    .map_err(|reason| decode_error(path, reason))?;
                let frame = decoder
                    .decode(&adts, None)
                    .map_err(|err| decode_error(path, format!("AAC decode failed: {err}")))?;
                if rate == 0 {
                    rate = frame.sample_rate;
                    decoded_channels = frame.channels;
                }
                pcm.extend_from_slice(&frame.samples);
            }
            if pcm.is_empty() || rate == 0 || decoded_channels == 0 {
                return Err(decode_error(path, "AAC track decoded to no audio"));
            }
            Ok((pcm, rate, decoded_channels))
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
            let rate = params.sample_rate.unwrap_or(0);
            let decoded_channels = params
                .channels
                .as_ref()
                .map_or(0, symphonia::core::audio::Channels::count);
            let track_id = audio_track.id;
            let mut decoder = symphonia::default::get_codecs()
                .make_audio_decoder(&params, &AudioDecoderOptions::default())
                .map_err(|err| {
                    decode_error(path, format!("failed to create MP3 decoder: {err}"))
                })?;
            let mut pcm = Vec::new();
            loop {
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
                let frame = decoder
                    .decode(&packet)
                    .map_err(|err| decode_error(path, format!("MP3 decode failed: {err}")))?;
                let count = frame.samples_interleaved();
                let start = pcm.len();
                pcm.resize(start.saturating_add(count), 0.0);
                if let Some(slot) = pcm.get_mut(start..start.saturating_add(count)) {
                    frame.copy_to_slice_interleaved(slot);
                }
            }
            if pcm.is_empty() || rate == 0 || decoded_channels == 0 {
                return Err(decode_error(path, "MP3 track decoded to no audio"));
            }
            Ok((pcm, rate, u16::try_from(decoded_channels).unwrap_or(0)))
        }
        other => Err(decode_error(
            path,
            format!("audio codec {other} is not supported for transcoding"),
        )),
    }
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

fn encode_aac_track(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<(ContainerTrack, Vec<ContainerPacket>), PipelineError> {
    if !(1..=6).contains(&channels) || channels == 4 {
        return Err(encode_error(
            path,
            format!("AAC output supports 1-3 and 5-6 channels, got {channels}"),
        ));
    }
    let rate = if rusty_aac::sf_index_for_rate(sample_rate).is_some() {
        sample_rate
    } else {
        OPUS_RATE
    };
    let pcm = resample_interleaved(samples, usize::from(channels), sample_rate, rate)?;
    if pcm.is_empty() {
        return Err(encode_error(path, "no samples to encode as AAC"));
    }
    let asc = aac_config(rate, channels)?;
    let mut encoder = rusty_aac::AacEncoder::new(rusty_aac::AacEncoderConfig {
        bitrate_bps: AAC_BITRATE_BPS,
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
    let duration = u64::try_from(AAC_FRAME)
        .unwrap_or(u64::MAX)
        .saturating_mul(1_000_000)
        / u64::from(rate.max(1));
    let packets = raw_frames
        .into_iter()
        .enumerate()
        .map(|(index, data)| {
            let pts = i64::try_from(index as u64)
                .unwrap_or(i64::MAX)
                .saturating_mul(i64::try_from(duration).unwrap_or(1));
            ContainerPacket {
                track: 0,
                pts,
                dts: pts,
                duration,
                keyframe: true,
                data,
            }
        })
        .collect();
    Ok((
        ContainerTrack {
            codec: VideoCodec::Aac,
            width: None,
            height: None,
            channels: Some(channels),
            sample_rate: Some(rate),
            config,
            timescale: US_TIMESCALE,
        },
        packets,
    ))
}

fn encode_opus_track(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<(ContainerTrack, Vec<ContainerPacket>), PipelineError> {
    if channels == 0 || channels > 2 {
        return Err(encode_error(
            path,
            format!("Opus output supports mono/stereo only, got {channels}"),
        ));
    }
    let pcm = resample_interleaved(samples, usize::from(channels), sample_rate, OPUS_RATE)?;
    if pcm.is_empty() {
        return Err(encode_error(path, "no samples to encode as Opus"));
    }
    let mut encoder = OpusEncoder::new(
        i32::try_from(OPUS_RATE).unwrap_or(i32::MAX),
        usize::from(channels),
        Application::Audio,
    )
    .map_err(|err| encode_error(path, format!("failed to create Opus encoder: {err}")))?;
    encoder.bitrate_bps = OPUS_BITRATE;
    encoder.complexity = 9;
    let head = opus_pure::OpusHead::for_encoder(&encoder, OPUS_RATE);
    let pre_skip = head.pre_skip;
    let frame_samples = OPUS_FRAME.saturating_mul(usize::from(channels));
    let frames = pcm.len().div_ceil(frame_samples).max(1);
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
            duration: 20_000,
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
            sample_rate: Some(OPUS_RATE),
            config: default_opus_head(channels, pre_skip),
            timescale: US_TIMESCALE,
        },
        packets,
    ))
}

fn encode_mp3_track(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<(ContainerTrack, Vec<ContainerPacket>), PipelineError> {
    if channels == 0 || channels > 2 {
        return Err(encode_error(
            path,
            format!("MP3 output supports mono/stereo only, got {channels}"),
        ));
    }
    let rate = if MP3_RATES.contains(&sample_rate) {
        sample_rate
    } else {
        OPUS_RATE
    };
    let pcm = resample_interleaved(samples, usize::from(channels), sample_rate, rate)?;
    if pcm.is_empty() {
        return Err(encode_error(path, "no samples to encode as MP3"));
    }
    let mut encoder = rusty_mp3::Mp3Encoder::new(rusty_mp3::Mp3EncoderConfig {
        bitrate_kbps: MP3_BITRATE_KBPS,
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
    let frame_us = frame_samples.saturating_mul(1_000_000) / u64::from(rate.max(1));
    let packets = frames
        .into_iter()
        .enumerate()
        .map(|(index, data)| {
            let pts = i64::try_from(index as u64)
                .unwrap_or(i64::MAX)
                .saturating_mul(i64::try_from(frame_us).unwrap_or(1));
            ContainerPacket {
                track: 0,
                pts,
                dts: pts,
                duration: frame_us,
                keyframe: true,
                data,
            }
        })
        .collect();
    Ok((
        ContainerTrack {
            codec: VideoCodec::Mp3,
            width: None,
            height: None,
            channels: Some(channels),
            sample_rate: Some(rate),
            config: Vec::new(),
            timescale: US_TIMESCALE,
        },
        packets,
    ))
}

/// Encodes decoded PCM into the requested in-container audio codec.
pub(crate) fn encode_audio_track(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    target: TargetAudioCodec,
) -> Result<(ContainerTrack, Vec<ContainerPacket>), PipelineError> {
    match target {
        TargetAudioCodec::Aac => encode_aac_track(path, samples, sample_rate, channels),
        TargetAudioCodec::Opus => encode_opus_track(path, samples, sample_rate, channels),
        TargetAudioCodec::Mp3 => encode_mp3_track(path, samples, sample_rate, channels),
    }
}

/// A crash-safe staging file: exclusively created, written through the owned
/// handle, then published by an atomic rename.
struct StagingTarget {
    temp: PathBuf,
    target: PathBuf,
    file: Option<File>,
}

impl StagingTarget {
    fn new(target: &Path, tag: usize) -> Result<Self, PipelineError> {
        if let Some(parent) = target.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let name = target.file_name().map_or_else(
            || "transcode.mp4".to_owned(),
            |n| n.to_string_lossy().into_owned(),
        );
        let serial = STAGING_SERIAL.fetch_add(1, Ordering::Relaxed);
        let temp = target.with_file_name(format!(
            ".{name}.transcode-{}-{serial}-{tag}.tmp",
            std::process::id()
        ));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(&temp).map_err(|source| PipelineError::Io {
            path: temp.clone(),
            source,
        })?;
        Ok(Self {
            temp,
            target: target.to_path_buf(),
            file: Some(file),
        })
    }

    fn file(&self) -> Result<&File, PipelineError> {
        self.file
            .as_ref()
            .ok_or_else(|| PipelineError::RemuxFailed {
                path: self.target.clone(),
                reason: "staging file is already committed".to_owned(),
            })
    }

    fn commit(mut self) -> Result<(), PipelineError> {
        drop(self.file.take());
        fs::rename(&self.temp, &self.target).map_err(|source| PipelineError::Io {
            path: self.target.clone(),
            source,
        })
    }
}

impl Drop for StagingTarget {
    fn drop(&mut self) {
        if self.file.is_some() {
            let _ = fs::remove_file(&self.temp);
        }
    }
}

static STAGING_SERIAL: AtomicU64 = AtomicU64::new(0);

fn input_size(path: &Path) -> u64 {
    fs::metadata(path).map_or(0, |meta| meta.len())
}

// Track-by-track dispatch keeps copy/encode decisions explicit per codec.
#[allow(clippy::too_many_lines)]
fn transcode_file(
    item: &VideoTranscodeItem,
    tag: usize,
) -> Result<(u64, u64, u64, Option<String>), PipelineError> {
    let source = item.source();
    let target = item.target();
    let format = item.output_format();
    let media = demux_video(source, item.input_format())?;
    // Freshly encoded tracks start at zero; copied tracks must be shifted by
    // the same amount (e.g. a TS PCR offset) or their timestamps diverge.
    let rebase_copied = media.tracks.iter().any(|track| {
        (track.codec.is_video() && matches!(item.video_action(), VideoTrackAction::Encode(_)))
            || (track.codec.is_audio()
                && matches!(item.audio_action(), AudioTrackAction::Encode(_)))
    });
    let source_min_dts = media
        .packets
        .iter()
        .map(|packet| packet.dts)
        .min()
        .unwrap_or(0);
    let mut tracks = Vec::with_capacity(media.tracks.len());
    let mut packets = Vec::new();
    let mut frames_processed = 0_u64;
    let mut hardware_device = None;

    for (index, source_track) in media.tracks.iter().enumerate() {
        let track_packets: Vec<ContainerPacket> = media
            .packets
            .iter()
            .filter(|packet| packet.track == index)
            .cloned()
            .collect();
        let new_index = tracks.len();
        match (source_track.codec.is_video(), source_track.codec.is_audio()) {
            (true, _) => match item.video_action() {
                VideoTrackAction::Copy => {
                    if !source_track.codec.supported_in(format) {
                        return Err(PipelineError::RemuxFailed {
                            path: target.to_path_buf(),
                            reason: format!(
                                "codec {} cannot be carried by {}",
                                source_track.codec,
                                format.label()
                            ),
                        });
                    }
                    // MPEG-TS carries no display geometry; the canonical muxer
                    // supplies a safe default when dimensions are absent.
                    tracks.push(source_track.clone());
                    for mut packet in track_packets {
                        packet.track = new_index;
                        if rebase_copied {
                            packet.pts = packet.pts.saturating_sub(source_min_dts);
                            packet.dts = packet.dts.saturating_sub(source_min_dts);
                        }
                        packets.push(packet);
                    }
                }
                VideoTrackAction::Encode(mode) => {
                    if !mode.codec().supported_in(format) {
                        return Err(PipelineError::RemuxFailed {
                            path: target.to_path_buf(),
                            reason: format!(
                                "codec {} cannot be carried by {}",
                                mode.codec(),
                                format.label()
                            ),
                        });
                    }
                    let frames = decode_video_track(source, source_track, &track_packets)?;
                    frames_processed = frames_processed
                        .saturating_add(u64::try_from(frames.len()).unwrap_or(u64::MAX));
                    let stream = match mode {
                        VideoEncodeMode::CpuH264 { .. }
                        | VideoEncodeMode::CpuAv1 { .. }
                        | VideoEncodeMode::CpuVp9 { .. } => {
                            encode_video_cpu(source, &frames, mode, new_index)?
                        }
                        VideoEncodeMode::NvencH264 { .. } | VideoEncodeMode::NvencHevc { .. } => {
                            let (stream, device) = crate::gpu::nvenc::encode_video_nvenc(
                                source, &frames, mode, new_index,
                            )?;
                            if hardware_device.is_none() {
                                hardware_device = Some(device);
                            }
                            stream
                        }
                    };
                    tracks.push(stream.track);
                    packets.extend(stream.packets);
                }
            },
            (false, true) => match item.audio_action() {
                AudioTrackAction::Copy => {
                    if !source_track.codec.supported_in(format) {
                        return Err(PipelineError::RemuxFailed {
                            path: target.to_path_buf(),
                            reason: format!(
                                "codec {} cannot be carried by {}",
                                source_track.codec,
                                format.label()
                            ),
                        });
                    }
                    tracks.push(source_track.clone());
                    for mut packet in track_packets {
                        packet.track = new_index;
                        if rebase_copied {
                            packet.pts = packet.pts.saturating_sub(source_min_dts);
                            packet.dts = packet.dts.saturating_sub(source_min_dts);
                        }
                        packets.push(packet);
                    }
                }
                AudioTrackAction::Encode(codec) => {
                    if !codec.supported_in(format) {
                        return Err(PipelineError::RemuxFailed {
                            path: target.to_path_buf(),
                            reason: format!(
                                "codec {codec} cannot be carried by {}",
                                format.label()
                            ),
                        });
                    }
                    let (samples, rate, channels) =
                        decode_audio_track_pcm(source, source_track, &track_packets)?;
                    let (mut track, mut encoded) =
                        encode_audio_track(source, &samples, rate, channels, codec)?;
                    for packet in &mut encoded {
                        packet.track = new_index;
                    }
                    if track.channels.is_none() {
                        track.channels = Some(channels);
                    }
                    tracks.push(track);
                    packets.extend(encoded);
                }
            },
            (false, false) => {
                return Err(PipelineError::RemuxFailed {
                    path: target.to_path_buf(),
                    reason: format!(
                        "track {} with codec {} is neither video nor audio and cannot be transcoded",
                        index, source_track.codec
                    ),
                });
            }
        }
    }

    if tracks.is_empty() || packets.is_empty() {
        return Err(PipelineError::RemuxFailed {
            path: target.to_path_buf(),
            reason: "source has no supported tracks to transcode".to_owned(),
        });
    }
    packets.sort_by_key(|packet| (packet.dts, packet.track));
    let out_media = crate::video_container::ContainerMedia { tracks, packets };
    let staging = StagingTarget::new(target, tag)?;
    mux_video(staging.file()?, target, format, &out_media)?;
    staging.commit()?;
    Ok((
        frames_processed,
        input_size(source),
        input_size(target),
        hardware_device,
    ))
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
    let workers = if uses_nvenc {
        1
    } else {
        plan.effective_workers().get().min(MAX_CPU_FILE_WORKERS)
    };
    let next = AtomicUsize::new(0);
    let frames = AtomicU64::new(0);
    let failure: Mutex<Option<PipelineError>> = Mutex::new(None);
    let device: Mutex<Option<String>> = Mutex::new(None);
    let tag = AtomicUsize::new(0);

    if items.len() == 1 {
        let (processed, _in, _out, hw) = transcode_file(&items[0], 0)?;
        frames.store(processed, Ordering::Relaxed);
        if let Some(hw) = hw
            && let Ok(mut guard) = device.lock()
        {
            *guard = Some(hw);
        }
    } else {
        thread::scope(|scope| -> Result<(), PipelineError> {
            let mut handles = Vec::with_capacity(workers);
            for _ in 0..workers {
                let (next, frames, failure, device, tag) =
                    (&next, &frames, &failure, &device, &tag);
                handles.push(scope.spawn(move || {
                    loop {
                        if failure.lock().map_or(true, |guard| guard.is_some()) {
                            break;
                        }
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(index) else { break };
                        let my_tag = tag.fetch_add(1, Ordering::Relaxed);
                        match transcode_file(item, my_tag) {
                            Ok((processed, _in, _out, hw)) => {
                                frames.fetch_add(processed, Ordering::Relaxed);
                                if let Some(hw) = hw
                                    && let Ok(mut guard) = device.lock()
                                    && guard.is_none()
                                {
                                    *guard = Some(hw);
                                }
                            }
                            Err(err) => {
                                if let Ok(mut guard) = failure.lock()
                                    && guard.is_none()
                                {
                                    *guard = Some(err);
                                }
                                break;
                            }
                        }
                    }
                }));
            }
            for handle in handles {
                handle.join().map_err(|_| PipelineError::WorkerPanicked)?;
            }
            Ok(())
        })?;
    }
    if let Some(err) = failure
        .into_inner()
        .map_err(|_| PipelineError::WorkerPanicked)?
    {
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
    Ok(VideoTranscodeReport::new(
        files,
        frames.load(Ordering::Relaxed),
        input_bytes,
        output_bytes,
        started.elapsed(),
        worker_threads,
        backend,
        hardware_device,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_codecs::Yuv420Frame;
    use crate::video_container::{ContainerMedia, mux_video};
    use fastconv_domain::{VideoContainerFormat, VideoStreamKind};
    use std::num::NonZeroUsize;
    use std::sync::atomic::{AtomicU64, Ordering};

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
            encode_audio_track(path, &samples, 48_000, 2, audio)?;
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
            let (track, packets) = encode_audio_track(&dir.join("x"), &samples, 48_000, 2, codec)?;
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
                quality: fastconv_domain::VideoQuality::DEFAULT,
            },
            TargetAudioCodec::Aac,
        )?;
        let report = transcode(
            &mp4,
            &webm,
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Webm,
            VideoTrackAction::Encode(VideoEncodeMode::CpuVp9 {
                quality: fastconv_domain::VideoQuality::DEFAULT,
            }),
            AudioTrackAction::Encode(TargetAudioCodec::Opus),
        )?;
        assert_eq!(report.files_converted().get(), 1);
        assert_eq!(report.frames_processed(), 4);
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
    fn webm_vp9_opus_to_avi_h264_mp3() -> Result<(), PipelineError> {
        let dir = temp_dir("webm-avi")?;
        let webm = dir.join("in.webm");
        let avi = dir.join("out.avi");
        make_source(
            &webm,
            VideoContainerFormat::Webm,
            VideoEncodeMode::CpuVp9 {
                quality: fastconv_domain::VideoQuality::DEFAULT,
            },
            TargetAudioCodec::Opus,
        )?;
        transcode(
            &webm,
            &avi,
            VideoContainerFormat::Webm,
            VideoContainerFormat::Avi,
            VideoTrackAction::Encode(VideoEncodeMode::CpuH264 {
                quality: fastconv_domain::VideoQuality::DEFAULT,
            }),
            AudioTrackAction::Encode(TargetAudioCodec::Mp3),
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
                quality: fastconv_domain::VideoQuality::DEFAULT,
            },
            TargetAudioCodec::Mp3,
        )?;
        transcode(
            &avi,
            &ts,
            VideoContainerFormat::Avi,
            VideoContainerFormat::Ts,
            VideoTrackAction::Encode(VideoEncodeMode::CpuAv1 {
                quality: fastconv_domain::VideoQuality::DEFAULT,
            }),
            AudioTrackAction::Encode(TargetAudioCodec::Aac),
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
                quality: fastconv_domain::VideoQuality::DEFAULT,
            },
            TargetAudioCodec::Aac,
        )?;
        transcode(
            &ts,
            &mov,
            VideoContainerFormat::Ts,
            VideoContainerFormat::Mov,
            VideoTrackAction::Copy,
            AudioTrackAction::Encode(TargetAudioCodec::Mp3),
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
}
