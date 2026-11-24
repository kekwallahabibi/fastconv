//! Pure-Rust uncompressed big-endian PCM AIFF writer.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

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

/// Writes a signed-integer PCM AIFF file from the provided interleaved samples.
pub(crate) fn encode_aiff(
    path: &Path,
    rate: u32,
    channels: usize,
    bits: usize,
    samples_i16: &[i16],
    samples_i32: &[i32],
) -> Result<(), PipelineError> {
    if !(1..=usize::from(u16::MAX)).contains(&channels) {
        return Err(PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("AIFF supports at most 65535 channels, got {channels}"),
        });
    }
    if !matches!(bits, 8 | 16 | 24) {
        return Err(PipelineError::EncodeFailed {
            path: path.to_path_buf(),
            reason: format!("unsupported AIFF sample depth {bits} bits"),
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

    let sample_count = if bits == 24 {
        samples_i32.len()
    } else {
        samples_i16.len()
    };
    let bytes_per_sample = bits / 8;
    let pcm_bytes_u64 = u64::try_from(sample_count).unwrap_or(u64::MAX)
        * u64::try_from(bytes_per_sample).unwrap_or(u64::MAX);
    let pad = usize::from((pcm_bytes_u64 & 1) != 0);
    let num_frames = sample_count / channels;
    let ssnd_size_u64 = pcm_bytes_u64.saturating_add(8);
    let form_size_u64 = pcm_bytes_u64
        .saturating_add(46)
        .saturating_add(u64::try_from(pad).unwrap_or(0));
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
    writer.write_all(b"SSND").map_err(io_err)?;
    writer.write_all(&ssnd_size.to_be_bytes()).map_err(io_err)?;
    writer.write_all(&0_u32.to_be_bytes()).map_err(io_err)?;
    writer.write_all(&0_u32.to_be_bytes()).map_err(io_err)?;

    write_aiff_pcm(&mut writer, path, bits, samples_i16, samples_i32)?;
    if pad == 1 {
        writer.write_all(&[0_u8]).map_err(io_err)?;
    }
    writer.flush().map_err(io_err)
}

fn write_aiff_pcm(
    writer: &mut BufWriter<File>,
    path: &Path,
    bits: usize,
    samples_i16: &[i16],
    samples_i32: &[i32],
) -> Result<(), PipelineError> {
    let io_err = |source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    };
    let mut chunk: Vec<u8> = Vec::with_capacity(PCM_CHUNK_BYTES);
    match bits {
        8 => {
            for &sample in samples_i16 {
                let value =
                    i8::try_from(sample).unwrap_or(if sample < 0 { i8::MIN } else { i8::MAX });
                chunk.extend_from_slice(&value.to_be_bytes());
                if chunk.len() >= PCM_CHUNK_BYTES {
                    writer.write_all(&chunk).map_err(io_err)?;
                    chunk.clear();
                }
            }
        }
        16 => {
            for &sample in samples_i16 {
                chunk.extend_from_slice(&sample.to_be_bytes());
                if chunk.len() >= PCM_CHUNK_BYTES {
                    writer.write_all(&chunk).map_err(io_err)?;
                    chunk.clear();
                }
            }
        }
        _ => {
            for &sample in samples_i32 {
                let bytes = sample.to_be_bytes();
                chunk.extend_from_slice(&bytes[1..4]);
                if chunk.len() >= PCM_CHUNK_BYTES {
                    writer.write_all(&chunk).map_err(io_err)?;
                    chunk.clear();
                }
            }
        }
    }
    writer.write_all(&chunk).map_err(io_err)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
