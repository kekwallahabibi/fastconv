//! Pure-Rust uncompressed big-endian PCM AIFF / AIFF-C writer.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use fastconv_domain::AudioBitDepth;

use crate::PipelineError;

const PCM_CHUNK_BYTES: usize = 256 * 1024;

/// Encodes an unsigned 32-bit sample rate as an IEEE 754 binary80 extended-precision value.
pub(crate) fn encode_ieee80_u32(value: u32) -> [u8; 10] {
    if value == 0 {
        return [0; 10];
    }
    let mut out = [0_u8; 10];
    let shift = value.leading_zeros();
    let exp: u16 =
        16_383_u16.saturating_add(31_u16.saturating_sub(u16::try_from(shift).unwrap_or(0)));
    let mantissa: u64 = u64::from(value << shift) << 32;
    out[..2].copy_from_slice(&exp.to_be_bytes());
    out[2..].copy_from_slice(&mantissa.to_be_bytes());
    out
}

/// Writes a PCM or IEEE-float AIFF file from the provided interleaved samples.
///
/// Integer targets use the classic `FORM`/`AIFF` layout. `Float32` writes the
/// standard AIFF-C `FORM`/`AIFC` layout with `FVER`, a 38-byte `COMM` carrying
/// the `fl32` compression id, and big-endian IEEE 754 `f32` samples in `SSND`.
pub(crate) fn encode_aiff(
    path: &Path,
    rate: u32,
    channels: usize,
    depth: AudioBitDepth,
    samples_i16: &[i16],
    samples_i32: &[i32],
    float_samples: &[f32],
) -> Result<(), PipelineError> {
    if !(1..=usize::from(u16::MAX)).contains(&channels) {
        return Err(PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("AIFF supports at most 65535 channels, got {channels}"),
        });
    }
    let io_err = |source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    };
    let encode_err = |reason: String| PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason,
    };

    let bits = usize::from(depth.bits());
    let sample_count = match depth {
        AudioBitDepth::Int8 | AudioBitDepth::Int16 => samples_i16.len(),
        AudioBitDepth::Int24 | AudioBitDepth::Int32 => samples_i32.len(),
        AudioBitDepth::Float32 => float_samples.len(),
    };
    let bytes_per_sample = bits / 8;
    let pcm_bytes_u64 = u64::try_from(sample_count)
        .ok()
        .and_then(|count| count.checked_mul(u64::try_from(bytes_per_sample).unwrap_or(u64::MAX)))
        .ok_or_else(|| encode_err("AIFF PCM byte size overflows".to_owned()))?;
    let pad = usize::from((pcm_bytes_u64 & 1) != 0);
    let num_frames = sample_count / channels;
    let ssnd_size_u64 = pcm_bytes_u64
        .checked_add(8)
        .ok_or_else(|| encode_err("AIFF SSND size overflows".to_owned()))?;
    // FORM payload = form type (4) + chunks. AIFF-C carries FVER (12) and a
    // 38-byte COMM instead of the plain 18-byte COMM, hence 78 vs 46.
    let header_bytes = if depth.is_float() { 78_u64 } else { 46_u64 };
    let form_size_u64 = pcm_bytes_u64
        .checked_add(header_bytes)
        .and_then(|size| size.checked_add(u64::try_from(pad).unwrap_or(u64::MAX)))
        .ok_or_else(|| encode_err("AIFF FORM size overflows".to_owned()))?;
    let num_frames = u32::try_from(num_frames)
        .map_err(|_| encode_err("AIFF frame count exceeds the 32-bit IFF field".to_owned()))?;
    let ssnd_size = u32::try_from(ssnd_size_u64)
        .map_err(|_| encode_err("AIFF SSND size exceeds the 32-bit IFF field".to_owned()))?;
    let form_size = u32::try_from(form_size_u64)
        .map_err(|_| encode_err("AIFF FORM size exceeds the 32-bit IFF field".to_owned()))?;
    let channel_count = u16::try_from(channels)
        .map_err(|_| encode_err("AIFF channel count exceeds the 16-bit IFF field".to_owned()))?;
    let bits_u16 = u16::try_from(bits)
        .map_err(|_| encode_err("AIFF sample depth exceeds the 16-bit IFF field".to_owned()))?;

    let file = File::create(path).map_err(io_err)?;
    let mut writer = BufWriter::with_capacity(PCM_CHUNK_BYTES, file);
    writer.write_all(b"FORM").map_err(io_err)?;
    writer.write_all(&form_size.to_be_bytes()).map_err(io_err)?;
    if depth.is_float() {
        writer.write_all(b"AIFC").map_err(io_err)?;
        // AIFF-C version 2, as required for `fl32` compressed-format chunks.
        writer.write_all(b"FVER").map_err(io_err)?;
        writer.write_all(&4_u32.to_be_bytes()).map_err(io_err)?;
        writer
            .write_all(&0xA280_5140_u32.to_be_bytes())
            .map_err(io_err)?;
        writer.write_all(b"COMM").map_err(io_err)?;
        writer.write_all(&38_u32.to_be_bytes()).map_err(io_err)?;
        writer
            .write_all(&channel_count.to_be_bytes())
            .map_err(io_err)?;
        writer
            .write_all(&num_frames.to_be_bytes())
            .map_err(io_err)?;
        writer.write_all(&bits_u16.to_be_bytes()).map_err(io_err)?;
        writer.write_all(&encode_ieee80_u32(rate)).map_err(io_err)?;
        writer.write_all(b"fl32").map_err(io_err)?;
        // Pascal string: 15-byte name padded to an even 16-byte payload.
        writer.write_all(b"\x0f32-bit floating").map_err(io_err)?;
    } else {
        writer.write_all(b"AIFF").map_err(io_err)?;
        writer.write_all(b"COMM").map_err(io_err)?;
        writer.write_all(&18_u32.to_be_bytes()).map_err(io_err)?;
        writer
            .write_all(&channel_count.to_be_bytes())
            .map_err(io_err)?;
        writer
            .write_all(&num_frames.to_be_bytes())
            .map_err(io_err)?;
        writer.write_all(&bits_u16.to_be_bytes()).map_err(io_err)?;
        writer.write_all(&encode_ieee80_u32(rate)).map_err(io_err)?;
    }
    writer.write_all(b"SSND").map_err(io_err)?;
    writer.write_all(&ssnd_size.to_be_bytes()).map_err(io_err)?;
    writer.write_all(&0_u32.to_be_bytes()).map_err(io_err)?;
    writer.write_all(&0_u32.to_be_bytes()).map_err(io_err)?;

    write_aiff_pcm(
        &mut writer,
        path,
        depth,
        samples_i16,
        samples_i32,
        float_samples,
    )?;
    if pad == 1 {
        writer.write_all(&[0_u8]).map_err(io_err)?;
    }
    writer.flush().map_err(io_err)
}

fn write_aiff_pcm(
    writer: &mut BufWriter<File>,
    path: &Path,
    depth: AudioBitDepth,
    samples_i16: &[i16],
    samples_i32: &[i32],
    float_samples: &[f32],
) -> Result<(), PipelineError> {
    let io_err = |source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    };
    let mut chunk: Vec<u8> = Vec::with_capacity(PCM_CHUNK_BYTES);
    let mut flush = |chunk: &mut Vec<u8>| -> Result<(), PipelineError> {
        if !chunk.is_empty() {
            writer.write_all(chunk).map_err(io_err)?;
            chunk.clear();
        }
        Ok(())
    };
    match depth {
        AudioBitDepth::Int8 => {
            for &sample in samples_i16 {
                let value =
                    i8::try_from(sample).unwrap_or(if sample < 0 { i8::MIN } else { i8::MAX });
                chunk.extend_from_slice(&value.to_be_bytes());
                if chunk.len() >= PCM_CHUNK_BYTES {
                    flush(&mut chunk)?;
                }
            }
        }
        AudioBitDepth::Int16 => {
            for &sample in samples_i16 {
                chunk.extend_from_slice(&sample.to_be_bytes());
                if chunk.len() >= PCM_CHUNK_BYTES {
                    flush(&mut chunk)?;
                }
            }
        }
        AudioBitDepth::Int24 => {
            for &sample in samples_i32 {
                let bytes = sample.to_be_bytes();
                chunk.extend_from_slice(&bytes[1..4]);
                if chunk.len() >= PCM_CHUNK_BYTES {
                    flush(&mut chunk)?;
                }
            }
        }
        AudioBitDepth::Int32 => {
            for &sample in samples_i32 {
                chunk.extend_from_slice(&sample.to_be_bytes());
                if chunk.len() >= PCM_CHUNK_BYTES {
                    flush(&mut chunk)?;
                }
            }
        }
        AudioBitDepth::Float32 => {
            for &sample in float_samples {
                chunk.extend_from_slice(&sample.to_be_bytes());
                if chunk.len() >= PCM_CHUNK_BYTES {
                    flush(&mut chunk)?;
                }
            }
        }
    }
    flush(&mut chunk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_aiff(label: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir = std::env::temp_dir().join(format!("fastconv-aiff-{label}-{nanos}"));
        let _ = fs::create_dir_all(&dir);
        dir.join("out.aiff")
    }

    #[test]
    fn encodes_ieee80_sample_rates() {
        assert_eq!(
            encode_ieee80_u32(44_100),
            [0x40, 0x0E, 0xAC, 0x44, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            encode_ieee80_u32(48_000),
            [0x40, 0x0E, 0xBB, 0x80, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            encode_ieee80_u32(96_000),
            [0x40, 0x0F, 0xBB, 0x80, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(encode_ieee80_u32(0), [0; 10]);
    }

    #[test]
    fn writes_aifc_fl32_header_and_big_endian_samples() -> Result<(), PipelineError> {
        let path = temp_aiff("fl32-header");
        let samples = [0.25_f32, -0.5_f32, 1.0_f32];
        encode_aiff(&path, 48_000, 1, AudioBitDepth::Float32, &[], &[], &samples)?;
        let bytes = fs::read(&path).map_err(|source| PipelineError::Io {
            path: path.clone(),
            source,
        })?;
        assert_eq!(&bytes[..4], b"FORM");
        assert_eq!(&bytes[8..12], b"AIFC");
        let form_size = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(form_size, 78 + 12);
        assert!(bytes.windows(4).any(|w| w == b"FVER"));
        assert!(bytes.windows(4).any(|w| w == b"COMM"));
        assert!(bytes.windows(4).any(|w| w == b"fl32"));
        assert!(bytes.windows(4).any(|w| w == b"SSND"));
        assert!(bytes.windows(16).any(|w| w == b"\x0f32-bit floating"));
        // FVER version 2 and the 38-byte float COMM are at fixed offsets.
        assert_eq!(&bytes[12..16], b"FVER");
        assert_eq!(&bytes[20..24], &0xA280_5140_u32.to_be_bytes());
        assert_eq!(&bytes[24..28], b"COMM");
        assert_eq!(&bytes[28..32], &38_u32.to_be_bytes());
        let ssnd = bytes
            .windows(4)
            .position(|w| w == b"SSND")
            .unwrap_or(bytes.len());
        assert!(ssnd < bytes.len());
        for (index, sample) in samples.iter().enumerate() {
            let start = ssnd.saturating_add(16).saturating_add(index * 4);
            let Some(slot) = bytes.get(start..start + 4) else {
                return Err(PipelineError::DecodeFailed {
                    path,
                    reason: "AIFC payload shorter than expected".to_owned(),
                });
            };
            assert_eq!(slot, sample.to_be_bytes());
        }
        if let Some(parent) = path.parent() {
            let _ = fs::remove_dir_all(parent);
        }
        Ok(())
    }

    #[test]
    fn writes_32bit_integer_aiff_with_four_byte_samples() -> Result<(), PipelineError> {
        let path = temp_aiff("int32");
        let samples = [1_i32, -2, i32::MAX - 7];
        encode_aiff(&path, 48_000, 1, AudioBitDepth::Int32, &[], &samples, &[])?;
        let bytes = fs::read(&path).map_err(|source| PipelineError::Io {
            path: path.clone(),
            source,
        })?;
        assert_eq!(&bytes[..4], b"FORM");
        assert_eq!(&bytes[8..12], b"AIFF");
        assert_eq!(&bytes[12..16], b"COMM");
        assert_eq!(&bytes[16..20], &18_u32.to_be_bytes());
        assert_eq!(&bytes[26..28], &32_u16.to_be_bytes());
        let ssnd = bytes
            .windows(4)
            .position(|w| w == b"SSND")
            .unwrap_or(bytes.len());
        for (index, sample) in samples.iter().enumerate() {
            let start = ssnd.saturating_add(16).saturating_add(index * 4);
            let Some(slot) = bytes.get(start..start + 4) else {
                return Err(PipelineError::DecodeFailed {
                    path,
                    reason: "AIFF payload shorter than expected".to_owned(),
                });
            };
            assert_eq!(slot, sample.to_be_bytes());
        }
        if let Some(parent) = path.parent() {
            let _ = fs::remove_dir_all(parent);
        }
        Ok(())
    }
}
