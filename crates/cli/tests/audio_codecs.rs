//! End-to-end checks for multi-codec containers: probe the source, convert via
//! the same boundary path the CLI uses, probe the result, and decode it back.

use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use fastconv_domain::{
    AudioBatchPlan, AudioInputFormat, AudioOptionBitrate, AudioOutputFormat, AudioRateControl,
    AudioTargetCodec, HardwareBackend,
};

fn temp_dir(label: &str) -> Result<PathBuf, String> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| err.to_string())?
        .as_nanos();
    let path = std::env::temp_dir().join(format!("fastconv-cli-codec-{label}-{nanos}"));
    fs::create_dir_all(&path).map_err(|err| err.to_string())?;
    Ok(path)
}

fn wav_bytes(samples: &[i16], rate: u32, channels: u16) -> Vec<u8> {
    let mut data = Vec::with_capacity(samples.len().saturating_mul(2));
    for &sample in samples {
        data.extend_from_slice(&sample.to_le_bytes());
    }
    let block_align = channels.saturating_mul(2);
    let byte_rate = rate.saturating_mul(u32::from(block_align));
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&1_u16.to_le_bytes());
    fmt.extend_from_slice(&channels.to_le_bytes());
    fmt.extend_from_slice(&rate.to_le_bytes());
    fmt.extend_from_slice(&byte_rate.to_le_bytes());
    fmt.extend_from_slice(&block_align.to_le_bytes());
    fmt.extend_from_slice(&16_u16.to_le_bytes());

    let mut chunks = Vec::new();
    chunks.extend_from_slice(b"fmt ");
    chunks.extend_from_slice(&u32::try_from(fmt.len()).unwrap_or(0).to_le_bytes());
    chunks.extend_from_slice(&fmt);
    chunks.extend_from_slice(b"data");
    chunks.extend_from_slice(&u32::try_from(data.len()).unwrap_or(0).to_le_bytes());
    chunks.extend_from_slice(&data);
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

fn write_wav(path: &Path, samples: &[i16], rate: u32, channels: u16) -> Result<(), String> {
    fs::write(path, wav_bytes(samples, rate, channels)).map_err(|err| err.to_string())
}

fn read_wav_i16(path: &Path) -> Result<Vec<i16>, String> {
    let bytes = fs::read(path).map_err(|err| err.to_string())?;
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("not a RIFF/WAVE file: {}", path.display()));
    }
    let mut offset = 12_usize;
    while offset.saturating_add(8) <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let len = usize::try_from(u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]))
        .unwrap_or(0);
        let start = offset.saturating_add(8);
        if id == b"data" {
            let end = start.saturating_add(len).min(bytes.len());
            let (pairs, _) = bytes[start..end].as_chunks::<2>();
            let samples = pairs
                .iter()
                .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
                .collect();
            return Ok(samples);
        }
        offset = start.saturating_add(len).saturating_add(len % 2);
    }
    Err(format!("no data chunk in {}", path.display()))
}

fn convert(
    source: &Path,
    target: &Path,
    to: Option<AudioOutputFormat>,
    codec: Option<&str>,
    bitrate: Option<&str>,
    bit_depth: Option<&str>,
) -> Result<AudioBatchPlan, String> {
    let request =
        fastconv_probe::parse_audio_encoding_request(codec, bitrate, None, None, None, bit_depth)?;
    let plan = fastconv_probe::prepare_audio_batch(
        source,
        target,
        to,
        Some(NonZeroUsize::MIN),
        HardwareBackend::Cpu,
        request,
    )
    .map_err(|err| err.to_string())?;
    fastconv_pipeline::execute_audio_batch(&plan).map_err(|err| err.to_string())?;
    Ok(plan)
}

#[test]
fn roundtrips_all_container_codec_pairs_with_two_way_probe() -> Result<(), String> {
    let dir = temp_dir("two-way-probe")?;
    let samples: Vec<i16> = (0_usize..16_000)
        .map(|index| {
            let value = i32::try_from(index.saturating_mul(7919) % 32_000).unwrap_or(0) - 16_000;
            i16::try_from(value).unwrap_or(0)
        })
        .collect();
    let source = dir.join("in.wav");
    write_wav(&source, &samples, 48_000, 2)?;
    assert_eq!(
        fastconv_probe::probe_audio(&source)
            .map_err(|err| err.to_string())?
            .format,
        AudioInputFormat::Wav
    );

    for (name, codec, codec_value, probed, lossless) in [
        (
            "ogg",
            "vorbis",
            AudioTargetCodec::Vorbis,
            AudioInputFormat::Vorbis,
            false,
        ),
        (
            "ogg",
            "opus",
            AudioTargetCodec::Opus,
            AudioInputFormat::Opus,
            false,
        ),
        (
            "ogg",
            "flac",
            AudioTargetCodec::Flac,
            AudioInputFormat::OggFlac,
            true,
        ),
        (
            "m4a",
            "aac",
            AudioTargetCodec::Aac,
            AudioInputFormat::M4a,
            false,
        ),
        (
            "m4a",
            "alac",
            AudioTargetCodec::Alac,
            AudioInputFormat::Alac,
            true,
        ),
    ] {
        let encoded = dir.join(format!("{codec}.{name}"));
        let back = dir.join(format!("{codec}-back.wav"));
        let plan = convert(
            &source,
            &encoded,
            None,
            Some(codec),
            None,
            if codec == "alac" { Some("16") } else { None },
        )?;
        assert_eq!(plan.target_codec(), codec_value, "--codec {codec}");

        let probed_encoded =
            fastconv_probe::probe_audio(&encoded).map_err(|err| err.to_string())?;
        assert_eq!(probed_encoded.format, probed, "--codec {codec}");
        assert_eq!(probed_encoded.stream.channels().get(), 2);
        assert_eq!(probed_encoded.stream.sample_rate().get(), 48_000);

        convert(
            &encoded,
            &back,
            Some(AudioOutputFormat::Wav),
            None,
            None,
            None,
        )?;
        let probed_back = fastconv_probe::probe_audio(&back).map_err(|err| err.to_string())?;
        assert_eq!(probed_back.format, AudioInputFormat::Wav);
        assert_eq!(probed_back.stream.channels().get(), 2);
        assert_eq!(probed_back.stream.sample_rate().get(), 48_000);
        let decoded = read_wav_i16(&back)?;
        assert!(!decoded.is_empty(), "--codec {codec} decoded no samples");
        if lossless {
            assert_eq!(decoded, samples, "--codec {codec} must be bit-exact");
        }
    }

    fs::remove_dir_all(&dir).map_err(|err| err.to_string())?;
    Ok(())
}

#[test]
fn bitrate_128k_reaches_opus_and_aac_containers() -> Result<(), String> {
    let dir = temp_dir("container-128k")?;
    let samples: Vec<i16> = (0_usize..16_000)
        .map(|index| {
            let value = i32::try_from(index.saturating_mul(4099) % 30_000).unwrap_or(0) - 15_000;
            i16::try_from(value).unwrap_or(0)
        })
        .collect();
    let source = dir.join("in.wav");
    write_wav(&source, &samples, 48_000, 2)?;

    for (codec, name, expected) in [
        ("opus", "ogg", AudioTargetCodec::Opus),
        ("aac", "m4a", AudioTargetCodec::Aac),
    ] {
        let target = dir.join(format!("out-{codec}.{name}"));
        let plan = convert(&source, &target, None, Some(codec), Some("128k"), None)?;
        assert_eq!(plan.target_codec(), expected);
        assert_eq!(
            plan.options().rate_control(),
            AudioRateControl::Bitrate(
                AudioOptionBitrate::new(128_000).map_err(|err| err.to_string())?
            )
        );
        assert_eq!(
            fastconv_probe::probe_audio(&target)
                .map_err(|err| err.to_string())?
                .format,
            if codec == "opus" {
                AudioInputFormat::Opus
            } else {
                AudioInputFormat::M4a
            }
        );
    }

    fs::remove_dir_all(&dir).map_err(|err| err.to_string())?;
    Ok(())
}
