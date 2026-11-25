//! ISO BMFF (M4A) muxing for AAC-LC and ALAC.

use std::fs::OpenOptions;
use std::path::Path;

use containr::mp4::{AudioCodec, Brand, MuxOptions, Muxer, PacketIn, SampleEntrySpec, TrackSpec};
use containr::{EditEntry, EditList, TrackKind};

use crate::PipelineError;

/// Packets, largest encoded packet and cookie from one ALAC shard.
type AlacShard = (Vec<(Vec<u8>, u32)>, u32, Vec<u8>);

/// Encodes one contiguous shard of whole ALAC packets with a private encoder.
fn encode_alac_shard(
    shard: &[i16],
    channels: usize,
    sample_rate: f64,
    chunk_samples: usize,
) -> AlacShard {
    let ch_u32 = u32::try_from(channels).unwrap_or(u32::MAX);
    let output_fmt = alac_encoder::FormatDescription::alac(
        sample_rate,
        alac_encoder::DEFAULT_FRAMES_PER_PACKET,
        ch_u32,
    );
    let input_fmt = alac_encoder::FormatDescription::pcm::<i16>(sample_rate, ch_u32);
    let mut encoder = alac_encoder::AlacEncoder::new(&output_fmt);
    let mut out_buf = vec![0_u8; output_fmt.max_packet_size()];
    let mut bytes = vec![0_u8; chunk_samples.saturating_mul(2)];
    let mut max_frame_bytes = 0_u32;
    let mut packets = Vec::new();
    for chunk in shard.chunks(chunk_samples) {
        for (index, &sample) in chunk.iter().enumerate() {
            let start = index.saturating_mul(2);
            if let Some(dst) = bytes.get_mut(start..start.saturating_add(2)) {
                dst.copy_from_slice(&sample.to_le_bytes());
            }
        }
        let frame_bytes = &bytes[..chunk.len().saturating_mul(2)];
        let written = encoder.encode(&input_fmt, frame_bytes, &mut out_buf);
        max_frame_bytes = max_frame_bytes.max(u32::try_from(written).unwrap_or(u32::MAX));
        let frames = u32::try_from(chunk.len() / channels).unwrap_or(u32::MAX);
        packets.push((out_buf[..written].to_vec(), frames));
    }
    (packets, max_frame_bytes, encoder.magic_cookie())
}

fn mux_alac_packets(
    path: &Path,
    sample_rate: u32,
    entry: Vec<u8>,
    packets: &[(Vec<u8>, u32)],
) -> Result<(), PipelineError> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let mut track = TrackSpec::new(TrackKind::Audio, sample_rate, SampleEntrySpec::Raw(entry));
    track.default_duration = Some(alac_encoder::DEFAULT_FRAMES_PER_PACKET);
    let options = MuxOptions {
        brand: Brand::Isom,
        faststart: true,
        movie_timescale: sample_rate,
        ..MuxOptions::default()
    };
    let mut muxer = Muxer::new(file, options).map_err(|err| encode_err(path, &err))?;
    let track_id = muxer
        .add_track(track)
        .map_err(|err| encode_err(path, &err))?;
    let mut cursor = 0_i64;
    for (data, frames) in packets {
        let packet = PacketIn {
            data,
            pts: cursor,
            dts: cursor,
            duration: Some(u64::from(*frames)),
            sync: true,
        };
        muxer
            .write_packet(track_id, &packet)
            .map_err(|err| encode_err(path, &err))?;
        cursor = cursor.saturating_add(i64::from(*frames));
    }
    muxer
        .finish()
        .map(|_| ())
        .map_err(|err| encode_err(path, &err))
}

/// Writes an ISO/IEC 14496-1 descriptor with the four-byte expanded length form.
fn descriptor(tag: u8, body: &[u8]) -> Vec<u8> {
    let len = u32::try_from(body.len()).unwrap_or(u32::MAX);
    let mut out = Vec::with_capacity(body.len().saturating_add(5));
    out.push(tag);
    out.push(u8::try_from(0x80 | ((len >> 21) & 0x7F)).unwrap_or(0xFF));
    out.push(u8::try_from(0x80 | ((len >> 14) & 0x7F)).unwrap_or(0xFF));
    out.push(u8::try_from(0x80 | ((len >> 7) & 0x7F)).unwrap_or(0xFF));
    out.push(u8::try_from(len & 0x7F).unwrap_or(0x7F));
    out.extend_from_slice(body);
    out
}

fn build_aac_esds(asc: &[u8], bitrate_bps: u32) -> Vec<u8> {
    let mut decoder_cfg = Vec::with_capacity(asc.len().saturating_add(18));
    decoder_cfg.extend_from_slice(&[0x40, 0x15, 0, 0, 0]);
    decoder_cfg.extend_from_slice(&bitrate_bps.to_be_bytes());
    decoder_cfg.extend_from_slice(&bitrate_bps.to_be_bytes());
    decoder_cfg.extend_from_slice(&descriptor(0x05, asc));

    let mut es_desc = Vec::with_capacity(decoder_cfg.len().saturating_add(12));
    es_desc.extend_from_slice(&[0x00, 0x01, 0x00]);
    es_desc.extend_from_slice(&descriptor(0x04, &decoder_cfg));
    es_desc.extend_from_slice(&descriptor(0x06, &[0x02]));

    let mut esds = Vec::with_capacity(es_desc.len().saturating_add(5));
    esds.extend_from_slice(&[0, 0, 0, 0]);
    esds.extend_from_slice(&descriptor(0x03, &es_desc));
    esds
}

/// Builds a complete `alac` `AudioSampleEntry` box around the encoder magic cookie.
fn build_alac_sample_entry(
    channels: u16,
    sample_rate: u32,
    magic_cookie: &[u8],
) -> Result<Vec<u8>, PipelineError> {
    if magic_cookie.len() < 24 {
        return Err(PipelineError::EncodeFailed {
            path: Path::new("alac-magic-cookie").to_path_buf(),
            reason: format!(
                "ALAC magic cookie is {} bytes, expected at least 24",
                magic_cookie.len()
            ),
        });
    }
    let mut inner = Vec::with_capacity(36);
    inner.extend_from_slice(&36_u32.to_be_bytes());
    inner.extend_from_slice(b"alac");
    inner.extend_from_slice(&0_u32.to_be_bytes());
    inner.extend_from_slice(&magic_cookie[..24]);

    let trailing = &magic_cookie[24..];
    let total = 36_u64
        .saturating_add(u64::try_from(inner.len()).unwrap_or(u64::MAX))
        .saturating_add(u64::try_from(trailing.len()).unwrap_or(u64::MAX));
    let size = u32::try_from(total).map_err(|_| PipelineError::EncodeFailed {
        path: Path::new("alac-sample-entry").to_path_buf(),
        reason: "ALAC sample entry exceeds the 32-bit box size field".to_owned(),
    })?;

    let mut entry = Vec::with_capacity(usize::try_from(total).unwrap_or(usize::MAX));
    entry.extend_from_slice(&size.to_be_bytes());
    entry.extend_from_slice(b"alac");
    entry.extend_from_slice(&[0_u8; 6]);
    entry.extend_from_slice(&1_u16.to_be_bytes());
    entry.extend_from_slice(&[0_u8; 8]);
    entry.extend_from_slice(&channels.to_be_bytes());
    entry.extend_from_slice(&16_u16.to_be_bytes());
    entry.extend_from_slice(&0_u32.to_be_bytes());
    let sample_rate_field = if sample_rate <= 0xFFFF {
        sample_rate << 16
    } else {
        0
    };
    entry.extend_from_slice(&sample_rate_field.to_be_bytes());
    entry.extend_from_slice(&inner);
    entry.extend_from_slice(trailing);
    Ok(entry)
}

fn encode_err(path: &Path, err: &containr::Error) -> PipelineError {
    PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason: format!("M4A muxing failed: {err}"),
    }
}

/// Muxes raw AAC-LC access units (without ADTS) into an M4A file.
pub(crate) fn write_m4a_aac(
    path: &Path,
    sample_rate: u32,
    channels: u16,
    bitrate_bps: u32,
    total_input_frames: u64,
    priming_samples: u32,
    packets: &[(Vec<u8>, u32)],
) -> Result<(), PipelineError> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let asc = rusty_aac::audio_specific_config_bytes(sample_rate, channels);
    let esds = build_aac_esds(&asc, bitrate_bps);
    let mut track = TrackSpec::new(
        TrackKind::Audio,
        sample_rate,
        SampleEntrySpec::Audio {
            codec: AudioCodec::Aac,
            config: esds,
            channels,
            sample_rate,
        },
    );
    track.default_duration = Some(1024);
    if total_input_frames > 0 {
        track.edit_list = EditList {
            movie_timescale: sample_rate,
            entries: vec![EditEntry {
                segment_duration: total_input_frames,
                media_time: i64::from(priming_samples),
                media_rate: EditEntry::NORMAL_RATE,
            }],
        };
    }
    let options = MuxOptions {
        brand: Brand::Isom,
        faststart: true,
        movie_timescale: sample_rate,
        ..MuxOptions::default()
    };
    let mut muxer = Muxer::new(file, options).map_err(|err| encode_err(path, &err))?;
    let track_id = muxer
        .add_track(track)
        .map_err(|err| encode_err(path, &err))?;
    let mut cursor = 0_i64;
    for (data, duration) in packets {
        let packet = PacketIn {
            data,
            pts: cursor,
            dts: cursor,
            duration: Some(u64::from(*duration)),
            sync: true,
        };
        muxer
            .write_packet(track_id, &packet)
            .map_err(|err| encode_err(path, &err))?;
        cursor = cursor.saturating_add(i64::from(*duration));
    }
    muxer.finish().map_err(|err| encode_err(path, &err))?;
    Ok(())
}

/// Encodes interleaved 16-bit PCM as ALAC and muxes the result into an M4A file.
///
/// Each worker owns a contiguous run of whole `DEFAULT_FRAME_SIZE` packets, so the
/// ALAC predictor state stays local and only the final packet of the whole stream
/// can be partial. The global `max_frame_bytes` in the magic cookie is patched from
/// all shards; the remaining cookie fields are identical for every worker.
pub(crate) fn encode_alac_m4a(
    path: &Path,
    sample_rate: u32,
    channels: usize,
    samples_i16: &[i16],
    inner_threads: usize,
) -> Result<(), PipelineError> {
    if !(1..=8).contains(&channels) || samples_i16.is_empty() {
        return Err(PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!(
                "ALAC supports 1-8 channels and non-empty PCM, got {channels} channels and {} samples",
                samples_i16.len()
            ),
        });
    }
    // Frame-aligned PCM keeps every shard boundary on a packet boundary.
    if !samples_i16.len().is_multiple_of(channels) {
        return Err(PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!(
                "ALAC PCM sample count {} is not a multiple of {channels} channels",
                samples_i16.len()
            ),
        });
    }
    let ch_u16 = u16::try_from(channels).map_err(|_| PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason: format!("ALAC channel count {channels} exceeds the 16-bit field"),
    })?;
    let total_frames = samples_i16.len() / channels;
    let packet_count = total_frames.div_ceil(alac_encoder::DEFAULT_FRAME_SIZE);
    let requested_workers = inner_threads.max(1).min(packet_count);
    let packets_per_worker = packet_count.div_ceil(requested_workers);
    let workers = packet_count.div_ceil(packets_per_worker);
    let chunk_samples = alac_encoder::DEFAULT_FRAME_SIZE.saturating_mul(channels);
    let sample_rate_f64 = f64::from(sample_rate);

    let mut results: Vec<Option<AlacShard>> = (0..workers).map(|_| None).collect();
    std::thread::scope(|scope| -> Result<(), PipelineError> {
        let mut handles = Vec::with_capacity(workers);
        for (worker_index, slot) in results.iter_mut().enumerate() {
            let start_packet = worker_index.saturating_mul(packets_per_worker);
            if start_packet >= packet_count {
                break;
            }
            let start_frame = start_packet
                .saturating_mul(alac_encoder::DEFAULT_FRAME_SIZE)
                .min(total_frames);
            let end_frame = start_frame
                .saturating_add(packets_per_worker.saturating_mul(alac_encoder::DEFAULT_FRAME_SIZE))
                .min(total_frames);
            let shard = &samples_i16
                [start_frame.saturating_mul(channels)..end_frame.saturating_mul(channels)];
            handles.push((
                slot,
                scope.spawn(move || {
                    encode_alac_shard(shard, channels, sample_rate_f64, chunk_samples)
                }),
            ));
        }
        for (slot, handle) in handles {
            *slot = Some(handle.join().map_err(|_| PipelineError::WorkerPanicked)?);
        }
        Ok(())
    })?;

    let mut packets: Vec<(Vec<u8>, u32)> = Vec::with_capacity(packet_count);
    let mut global_max_frame_bytes = 0_u32;
    let mut cookie: Option<Vec<u8>> = None;
    for (worker_index, result) in results.into_iter().enumerate() {
        let (shard_packets, shard_max, shard_cookie) =
            result.ok_or(PipelineError::WorkerPanicked)?;
        if worker_index == 0 {
            cookie = Some(shard_cookie);
        }
        global_max_frame_bytes = global_max_frame_bytes.max(shard_max);
        packets.extend(shard_packets);
    }
    let mut cookie = cookie.ok_or(PipelineError::WorkerPanicked)?;
    if cookie.len() < 24 {
        return Err(PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!(
                "ALAC magic cookie is {} bytes, expected at least 24",
                cookie.len()
            ),
        });
    }
    // ALACSpecificConfig: frame_length(4), version(1), bit_depth(1), PB/MB/KB(1 each),
    // channels(1), max_run(2), max_frame_bytes(4) at byte offset 12.
    cookie[12..16].copy_from_slice(&global_max_frame_bytes.to_be_bytes());
    let entry = build_alac_sample_entry(ch_u16, sample_rate, &cookie)?;
    mux_alac_packets(path, sample_rate, entry, &packets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alac_sample_entry_has_expected_layout() {
        let cookie = vec![0_u8; 24];
        let entry = build_alac_sample_entry(2, 48_000, &cookie).unwrap_or_default();
        assert_eq!(&entry[..4], &72_u32.to_be_bytes());
        assert_eq!(&entry[4..8], b"alac");
        assert_eq!(&entry[24..26], &2_u16.to_be_bytes());
        assert_eq!(&entry[26..28], &16_u16.to_be_bytes());
        assert_eq!(&entry[32..36], &(48_000_u32 << 16).to_be_bytes());
        assert_eq!(&entry[36..40], &36_u32.to_be_bytes());
        assert_eq!(&entry[40..44], b"alac");
        assert_eq!(entry.len(), 72);
    }

    #[test]
    fn rejects_short_alac_cookie() {
        assert!(build_alac_sample_entry(2, 48_000, &[0_u8; 23]).is_err());
    }

    #[test]
    fn descriptor_encodes_expanded_length() {
        assert_eq!(
            descriptor(0x05, &[1, 2]),
            vec![0x05, 0x80, 0x80, 0x80, 2, 1, 2]
        );
    }

    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::formats::TrackType;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
    use symphonia::core::meta::MetadataOptions;

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = std::env::temp_dir().join(format!("fastconv-alac-{label}-{nanos}"));
        let _ = fs::create_dir_all(&path);
        path
    }

    fn probe(
        path: &Path,
    ) -> Result<Box<dyn symphonia::core::formats::FormatReader>, PipelineError> {
        let file = fs::File::open(path).map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let stream = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
        let mut hint = Hint::new();
        hint.with_extension("m4a");
        symphonia::default::get_probe()
            .probe(
                &hint,
                stream,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(|err| PipelineError::DecodeFailed {
                path: path.to_path_buf(),
                reason: format!("failed to probe M4A: {err}"),
            })
    }

    fn decode_alac(path: &Path) -> Result<Vec<i16>, PipelineError> {
        let mut reader = probe(path)?;
        let track =
            reader
                .default_track(TrackType::Audio)
                .ok_or_else(|| PipelineError::DecodeFailed {
                    path: path.to_path_buf(),
                    reason: "no audio track".to_owned(),
                })?;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|params| params.audio())
            .ok_or_else(|| PipelineError::DecodeFailed {
                path: path.to_path_buf(),
                reason: "no audio parameters".to_owned(),
            })?
            .clone();
        let track_id = track.id;
        let mut decoder = symphonia::default::get_codecs()
            .make_audio_decoder(&params, &AudioDecoderOptions::default())
            .map_err(|err| PipelineError::DecodeFailed {
                path: path.to_path_buf(),
                reason: format!("failed to create ALAC decoder: {err}"),
            })?;
        let mut out = Vec::new();
        loop {
            let packet = match reader.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => break,
                Err(err) => {
                    return Err(PipelineError::DecodeFailed {
                        path: path.to_path_buf(),
                        reason: format!("failed to read M4A packet: {err}"),
                    });
                }
            };
            if packet.track_id != track_id {
                continue;
            }
            let frame = decoder
                .decode(&packet)
                .map_err(|err| PipelineError::DecodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("failed to decode ALAC packet: {err}"),
                })?;
            let start = out.len();
            let end = start.saturating_add(frame.samples_interleaved());
            out.resize(end, 0);
            if let Some(slot) = out.get_mut(start..end) {
                frame.copy_to_slice_interleaved::<i16, _>(slot);
            }
        }
        Ok(out)
    }

    fn cookie_from_file(path: &Path) -> Result<Vec<u8>, PipelineError> {
        let bytes = fs::read(path).map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        // `alac` sample entry: the 36-byte inner box header (36, "alac")
        // precedes the 24-byte ALACSpecificConfig magic cookie.
        let offset = bytes
            .windows(8)
            .position(|window| window[..4] == 36_u32.to_be_bytes() && &window[4..] == b"alac")
            .ok_or_else(|| PipelineError::DecodeFailed {
                path: path.to_path_buf(),
                reason: "no ALAC magic cookie in M4A".to_owned(),
            })?;
        let start = offset.saturating_add(12);
        let cookie = bytes.get(start..start.saturating_add(24)).ok_or_else(|| {
            PipelineError::DecodeFailed {
                path: path.to_path_buf(),
                reason: "truncated ALAC magic cookie".to_owned(),
            }
        })?;
        Ok(cookie.to_vec())
    }

    fn muxed_max_packet(path: &Path) -> Result<usize, PipelineError> {
        let mut reader = probe(path)?;
        let track =
            reader
                .default_track(TrackType::Audio)
                .ok_or_else(|| PipelineError::DecodeFailed {
                    path: path.to_path_buf(),
                    reason: "no audio track".to_owned(),
                })?;
        let track_id = track.id;
        let mut max = 0_usize;
        loop {
            let packet = match reader.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => break,
                Err(err) => {
                    return Err(PipelineError::DecodeFailed {
                        path: path.to_path_buf(),
                        reason: format!("failed to read M4A packet: {err}"),
                    });
                }
            };
            if packet.track_id == track_id {
                max = max.max(packet.data.len());
            }
        }
        Ok(max)
    }

    fn tone(frames: usize, channels: usize) -> Vec<i16> {
        (0..frames)
            .flat_map(|frame| {
                (0..channels).map(move |channel| {
                    let phase = (frame.saturating_mul(137 + channel * 29)) % 4096;
                    i16::try_from(
                        i64::try_from(phase).unwrap_or(0).saturating_mul(24_000) / 4096 - 12_000,
                    )
                    .unwrap_or(0)
                })
            })
            .collect()
    }

    #[test]
    fn parallel_alac_roundtrip_is_bit_exact_with_partial_tail() -> Result<(), PipelineError> {
        let dir = temp_dir("parallel-stereo");
        let path = dir.join("out.m4a");
        let samples = tone(alac_encoder::DEFAULT_FRAME_SIZE * 3 + 17, 2);
        encode_alac_m4a(&path, 48_000, 2, &samples, 4)?;
        assert_eq!(decode_alac(&path)?, samples);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn parallel_alac_mono_and_single_thread_roundtrip() -> Result<(), PipelineError> {
        let dir = temp_dir("mono-single-thread");
        let path = dir.join("out.m4a");
        let samples = tone(alac_encoder::DEFAULT_FRAME_SIZE + 13, 1);
        encode_alac_m4a(&path, 48_000, 1, &samples, 1)?;
        assert_eq!(decode_alac(&path)?, samples);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn parallel_alac_cookie_records_global_max_frame_bytes() -> Result<(), PipelineError> {
        let dir = temp_dir("cookie-max");
        let path = dir.join("out.m4a");
        let samples = tone(alac_encoder::DEFAULT_FRAME_SIZE * 5, 2);
        encode_alac_m4a(&path, 48_000, 2, &samples, 4)?;
        let cookie = cookie_from_file(&path)?;
        assert_eq!(&cookie[0..4], &(4096_u32).to_be_bytes());
        assert_eq!(cookie[4], 0, "ALAC compatible version");
        assert_eq!(cookie[5], 16);
        assert_eq!(cookie[9], 2);
        assert_eq!(&cookie[20..24], &48_000_u32.to_be_bytes());
        assert_eq!(cookie[16..20], [0, 0, 0, 0], "avg_bit_rate stays unset");
        let packet_max = u32::try_from(muxed_max_packet(&path)?).unwrap_or(u32::MAX);
        let cookie_max = u32::from_be_bytes(cookie[12..16].try_into().unwrap_or([0, 0, 0, 0]));
        assert_eq!(cookie_max, packet_max);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn parallel_alac_rejects_misaligned_pcm() {
        let dir = temp_dir("misaligned");
        let path = dir.join("out.m4a");
        let result = encode_alac_m4a(&path, 48_000, 2, &[0_i16; 4097], 4);
        assert!(matches!(result, Err(PipelineError::EncodeFailed { .. })));
        let _ = fs::remove_dir_all(&dir);
    }
}
