//! ISO BMFF (M4A) muxing for AAC-LC and ALAC.

use std::fs::OpenOptions;
use std::path::Path;

use containr::mp4::{AudioCodec, Brand, MuxOptions, Muxer, PacketIn, SampleEntrySpec, TrackSpec};
use containr::{EditEntry, EditList, TrackKind};

use crate::PipelineError;

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
pub(crate) fn encode_alac_m4a(
    path: &Path,
    sample_rate: u32,
    channels: usize,
    samples_i16: &[i16],
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
    let ch_u32 = u32::try_from(channels).map_err(|_| PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason: format!("ALAC channel count {channels} exceeds the 32-bit field"),
    })?;
    let ch_u16 = u16::try_from(channels).map_err(|_| PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason: format!("ALAC channel count {channels} exceeds the 16-bit field"),
    })?;
    let output_fmt = alac_encoder::FormatDescription::alac(
        f64::from(sample_rate),
        alac_encoder::DEFAULT_FRAMES_PER_PACKET,
        ch_u32,
    );
    let input_fmt = alac_encoder::FormatDescription::pcm::<i16>(f64::from(sample_rate), ch_u32);
    let mut encoder = alac_encoder::AlacEncoder::new(&output_fmt);
    let mut out_buf = vec![0_u8; output_fmt.max_packet_size()];
    let mut packets: Vec<(Vec<u8>, u32)> = Vec::new();
    let chunk_samples = alac_encoder::DEFAULT_FRAME_SIZE.saturating_mul(channels);
    let mut bytes = Vec::with_capacity(chunk_samples.saturating_mul(2));
    for chunk in samples_i16.chunks(chunk_samples) {
        bytes.clear();
        for &sample in chunk {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        let written = encoder.encode(&input_fmt, &bytes, &mut out_buf);
        let frames = u32::try_from(chunk.len() / channels).unwrap_or(u32::MAX);
        packets.push((out_buf[..written].to_vec(), frames));
    }
    // The cookie records packet statistics that are only final after all frames are encoded.
    let cookie = encoder.magic_cookie();
    let entry = build_alac_sample_entry(ch_u16, sample_rate, &cookie)?;

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
            data: &data,
            pts: cursor,
            dts: cursor,
            duration: Some(u64::from(frames)),
            sync: true,
        };
        muxer
            .write_packet(track_id, &packet)
            .map_err(|err| encode_err(path, &err))?;
        cursor = cursor.saturating_add(i64::from(frames));
    }
    muxer.finish().map_err(|err| encode_err(path, &err))?;
    Ok(())
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
}
