//! End-to-end checks for multi-codec containers: probe the source, convert via
//! the same boundary path the CLI uses, probe the result, and decode it back.

use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use fastconv_domain::{
    AudioBatchPlan, AudioInputFormat, AudioOptionBitrate, AudioOutputFormat, AudioRateControl,
    AudioTargetCodec, DomainError, HardwareBackend,
};
use fastconv_probe::ProbeError;

/// Owns a uniquely named scratch directory and removes it on every exit path.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(label: &str) -> Result<Self, String> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| err.to_string())?
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fastconv-cli-codec-{label}-{nanos}"));
        // Exclusive creation: never reuse (and then recursively delete) a
        // directory that may belong to another test or process.
        fs::create_dir(&path).map_err(|err| err.to_string())?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if let Err(err) = fs::remove_dir_all(&self.path) {
            eprintln!("failed to clean up temp dir {}: {err}", self.path.display());
        }
    }
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
    let dir = TempDir::new("two-way-probe")?;
    let samples: Vec<i16> = (0_usize..16_000)
        .map(|index| {
            let value = i32::try_from(index.saturating_mul(7919) % 32_000).unwrap_or(0) - 16_000;
            i16::try_from(value).unwrap_or(0)
        })
        .collect();
    let source = dir.path().join("in.wav");
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
        let encoded = dir.path().join(format!("{codec}.{name}"));
        let back = dir.path().join(format!("{codec}-back.wav"));
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
    Ok(())
}

#[test]
fn non_default_bitrates_reach_opus_and_aac_containers() -> Result<(), String> {
    let dir = TempDir::new("container-bitrates")?;
    // ~2 seconds of deterministic high entropy at 48 kHz stereo: a VBR encoder
    // cannot hide an ignored --bitrate behind a near-silent source.
    let mut state: u32 = 0x1357_9BDF;
    let samples: Vec<i16> = std::iter::repeat_with(|| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let centered = i32::try_from(state >> 16).unwrap_or(0) - 32_768;
        i16::try_from(centered).unwrap_or(0)
    })
    .take(96_000_usize.saturating_mul(2))
    .collect();
    let source = dir.path().join("in.wav");
    write_wav(&source, &samples, 48_000, 2)?;

    for (codec, name, expected, probed) in [
        (
            "opus",
            "ogg",
            AudioTargetCodec::Opus,
            AudioInputFormat::Opus,
        ),
        ("aac", "m4a", AudioTargetCodec::Aac, AudioInputFormat::M4a),
    ] {
        let mut sizes = Vec::new();
        // Both values differ from the 128 kbps encoder default, so dropping
        // --bitrate anywhere on the path cannot accidentally satisfy this test.
        for (raw, bps) in [("64k", 64_000_u32), ("256k", 256_000_u32)] {
            let target = dir.path().join(format!("out-{codec}-{raw}.{name}"));
            let plan = convert(&source, &target, None, Some(codec), Some(raw), None)?;
            assert_eq!(
                plan.target_codec(),
                expected,
                "--codec {codec} --bitrate {raw}"
            );
            assert_eq!(
                plan.options().rate_control(),
                AudioRateControl::Bitrate(
                    AudioOptionBitrate::new(bps).map_err(|err| err.to_string())?
                ),
                "--codec {codec} --bitrate {raw}"
            );
            assert_eq!(
                fastconv_probe::probe_audio(&target)
                    .map_err(|err| err.to_string())?
                    .format,
                probed,
                "--codec {codec} --bitrate {raw}"
            );
            sizes.push(fs::metadata(&target).map_err(|err| err.to_string())?.len());
        }
        assert!(
            sizes[1] > sizes[0],
            "--codec {codec}: 256k output ({} bytes) must exceed 64k output ({} bytes)",
            sizes[1],
            sizes[0]
        );
    }
    Ok(())
}

#[test]
fn alac_rejects_sample_rate_above_fixed_point_limit() -> Result<(), String> {
    let dir = TempDir::new("alac-rate-limit")?;
    let samples: Vec<i16> = (0_usize..8_000)
        .map(|index| {
            let value = i32::try_from(index.saturating_mul(4001) % 30_000).unwrap_or(0) - 15_000;
            i16::try_from(value).unwrap_or(0)
        })
        .collect();
    let source = dir.path().join("in.wav");
    write_wav(&source, &samples, 48_000, 2)?;

    // M4A stores ALAC's sample rate in a 16.16 fixed-point field: 96 kHz would
    // truncate to zero, so the boundary must reject it without building a plan.
    let request = fastconv_probe::parse_audio_encoding_request(
        Some("alac"),
        None,
        None,
        Some("96000"),
        None,
        Some("16"),
    )?;
    let target = dir.path().join("rejected.m4a");
    match fastconv_probe::prepare_audio_batch(
        &source,
        &target,
        None,
        Some(NonZeroUsize::MIN),
        HardwareBackend::Cpu,
        request,
    ) {
        Ok(_) => return Err("--codec alac --sample-rate 96000 must be rejected".to_owned()),
        Err(ProbeError::InvalidDomain(DomainError::InvalidAudioOptionSampleRate {
            hz, ..
        })) => assert_eq!(hz, 96_000),
        Err(err) => {
            return Err(format!(
                "expected InvalidAudioOptionSampleRate for 96000 Hz, got {err:?}"
            ));
        }
    }

    let request = fastconv_probe::parse_audio_encoding_request(
        Some("alac"),
        None,
        None,
        Some("48000"),
        None,
        Some("16"),
    )?;
    let target = dir.path().join("accepted.m4a");
    let plan = fastconv_probe::prepare_audio_batch(
        &source,
        &target,
        None,
        Some(NonZeroUsize::MIN),
        HardwareBackend::Cpu,
        request,
    )
    .map_err(|err| err.to_string())?;
    assert_eq!(plan.target_codec(), AudioTargetCodec::Alac);
    Ok(())
}
