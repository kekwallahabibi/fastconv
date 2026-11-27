//! Xiph Ogg-FLAC v1.0 mapping for FLAC frames produced by `flacenc`.
//!
//! The identification packet carries the native FLAC `STREAMINFO` block, the
//! second packet carries an empty Vorbis comment, and each following Ogg packet
//! carries exactly one FLAC audio frame. No new dependency is required: `ogg`
//! already frames Vorbis and Opus packets elsewhere in the workspace.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use flacenc::bitsink::ByteSink;
use flacenc::component::{BitRepr, Stream};
use ogg::{PacketWriteEndInfo, PacketWriter};

use crate::PipelineError;

/// Nonzero stream serial; each output file has exactly one logical bitstream.
const STREAM_SERIAL: u32 = 0x4643_4F4E;
const STREAM_INFO_BYTES: usize = 34;
const IDENTIFICATION_BYTES: usize = 51;
const COMMENT_BYTES: usize = 20;
const VENDOR: &[u8] = b"fastconv";

fn encode_err(path: &Path, reason: String) -> PipelineError {
    PipelineError::EncodeFailed {
        path: path.to_path_buf(),
        reason,
    }
}

fn io_err(path: &Path, source: std::io::Error) -> PipelineError {
    PipelineError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Builds the 51-byte BOS packet: Ogg-FLAC signature, mapping version, one
/// header packet, the native `fLaC` marker, and the 34-byte `STREAMINFO` block.
fn identification_packet(path: &Path, stream: &Stream) -> Result<Vec<u8>, PipelineError> {
    let mut info_sink = ByteSink::new();
    stream
        .stream_info()
        .write(&mut info_sink)
        .map_err(|err| encode_err(path, format!("FLAC stream-info write failed: {err}")))?;
    let info_bytes = info_sink.as_slice();
    if info_bytes.len() != STREAM_INFO_BYTES {
        return Err(encode_err(
            path,
            format!(
                "FLAC stream-info is {} bytes, expected {STREAM_INFO_BYTES}",
                info_bytes.len()
            ),
        ));
    }

    let mut packet = Vec::with_capacity(IDENTIFICATION_BYTES);
    packet.push(0x7F);
    packet.extend_from_slice(b"FLAC");
    packet.extend_from_slice(&[0x01, 0x00]);
    packet.extend_from_slice(&1_u16.to_be_bytes());
    packet.extend_from_slice(b"fLaC");
    let info_len = u8::try_from(STREAM_INFO_BYTES).unwrap_or(u8::MAX);
    packet.extend_from_slice(&[0x00, 0x00, 0x00, info_len]);
    packet.extend_from_slice(info_bytes);
    if packet.len() != IDENTIFICATION_BYTES {
        return Err(encode_err(
            path,
            format!(
                "Ogg-FLAC identification packet is {} bytes, expected {IDENTIFICATION_BYTES}",
                packet.len()
            ),
        ));
    }
    Ok(packet)
}

/// Builds the last metadata packet: an empty Vorbis comment with a `fastconv` vendor.
fn comment_packet(path: &Path) -> Result<Vec<u8>, PipelineError> {
    let mut packet = Vec::with_capacity(COMMENT_BYTES);
    packet.extend_from_slice(&[0x84, 0x00, 0x00, 16]);
    packet.extend_from_slice(
        &u32::try_from(VENDOR.len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    packet.extend_from_slice(VENDOR);
    packet.extend_from_slice(&0_u32.to_le_bytes());
    if packet.len() != COMMENT_BYTES {
        return Err(encode_err(
            path,
            format!(
                "Ogg-FLAC comment packet is {} bytes, expected {COMMENT_BYTES}",
                packet.len()
            ),
        ));
    }
    Ok(packet)
}

/// Writes one FLAC frame per Ogg packet; the granule position is the number of
/// decoded samples that have been emitted so far.
fn write_frames<W: Write>(
    path: &Path,
    writer: &mut PacketWriter<'_, W>,
    stream: &Stream,
) -> Result<(), PipelineError> {
    let frame_count = stream.frame_count();
    let mut sink = ByteSink::with_capacity(4096 * 8);
    let mut granule = 0_u64;
    for index in 0..frame_count {
        let frame = stream.frame(index).ok_or_else(|| {
            encode_err(
                path,
                format!("FLAC frame {index} vanished during Ogg serialization"),
            )
        })?;
        sink.clear();
        frame
            .write(&mut sink)
            .map_err(|err| encode_err(path, format!("FLAC frame write failed: {err}")))?;
        let block_size = u64::try_from(frame.header().block_size()).map_err(|_| {
            encode_err(
                path,
                "FLAC frame block size does not fit in an Ogg granule".to_owned(),
            )
        })?;
        granule = granule
            .checked_add(block_size)
            .ok_or_else(|| encode_err(path, "Ogg-FLAC granule position overflowed".to_owned()))?;
        let end = if index + 1 == frame_count {
            PacketWriteEndInfo::EndStream
        } else {
            PacketWriteEndInfo::NormalPacket
        };
        // `PacketWriter` stores packets for the lifetime of the writer, so the
        // frame bytes must be owned; the sink itself is reused across frames.
        writer
            .write_packet(sink.as_slice().to_vec(), STREAM_SERIAL, end, granule)
            .map_err(|source| io_err(path, source))?;
    }
    Ok(())
}

/// Writes an Ogg-FLAC v1.0 stream from an already encoded native FLAC `Stream`.
///
/// # Errors
///
/// Returns [`PipelineError::EmptyOutput`] for a stream without frames and a
/// [`PipelineError`] for any serialization or file I/O failure.
pub(crate) fn write_ogg_flac(path: &Path, stream: &Stream) -> Result<(), PipelineError> {
    if stream.frame_count() == 0 {
        return Err(PipelineError::EmptyOutput(path.to_path_buf()));
    }
    let identification = identification_packet(path, stream)?;
    let comment = comment_packet(path)?;

    let file = File::create(path).map_err(|source| io_err(path, source))?;
    let mut writer = PacketWriter::new(BufWriter::with_capacity(64 * 1024, file));
    writer
        .write_packet(
            identification.as_slice(),
            STREAM_SERIAL,
            PacketWriteEndInfo::EndPage,
            0,
        )
        .map_err(|source| io_err(path, source))?;
    writer
        .write_packet(
            comment.as_slice(),
            STREAM_SERIAL,
            PacketWriteEndInfo::EndPage,
            0,
        )
        .map_err(|source| io_err(path, source))?;
    write_frames(path, &mut writer, stream)?;

    let mut buffered = writer.into_inner();
    buffered.flush().map_err(|source| io_err(path, source))
}
