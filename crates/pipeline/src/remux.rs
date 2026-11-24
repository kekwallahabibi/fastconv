//! Lossless container remuxing: packet/bitstream copy between MP4/MOV and MKV.
//!
//! Uses `containr` as a pure-Rust container library. No codec is decoded or
//! re-encoded. No `ffmpeg`/`ffprobe` subprocess is involved.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

/// Process-wide serial makes staging paths unique across concurrent invocations.
static STAGING_SERIAL: AtomicUsize = AtomicUsize::new(0);

use containr::Packet;
use containr::mkv::DocType;
use containr::mp4::Brand;
use fastconv_domain::{
    HardwareBackend, VideoCodec, VideoContainerFormat, VideoRemuxItem, VideoRemuxPlan,
    VideoRemuxReport,
};

use crate::PipelineError;
use crate::video_container::{demux_video, mux_video};

/// Nanoseconds per Matroska tick used when an exact timestamp plan is not possible.
///
/// A 48 kHz AAC frame and 60 fps video do not land on whole nanoseconds, so
/// `StreamCopy::plan` refuses them. Rounding to 1 µs keeps drift negligible
/// while staying well inside Matroska's tick range.
const ROUNDED_TICK_NS: u64 = 1;

/// Window size used by [`WindowedFileReader`].
const WINDOW_BYTES: usize = 1024 * 1024;

/// Above this source size, MP4 faststart is left on the target file instead of
/// being staged in RAM.
const IN_MEMORY_MP4_FASTSTART_LIMIT: u64 = 64 * 1024 * 1024;

fn remux_failed(path: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::RemuxFailed {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

fn container_err(path: &Path, err: &containr::Error) -> PipelineError {
    remux_failed(path, format!("container error: {err}"))
}

/// Keeps a 1 MiB window of the source in memory across containr's absolute
/// seeks so each EBML/box header and packet does not become a fresh syscall.
struct WindowedFileReader {
    file: File,
    file_len: u64,
    pos: u64,
    buf: Vec<u8>,
    buf_start: u64,
    buf_valid: usize,
}

impl WindowedFileReader {
    fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let window_len = u64::try_from(WINDOW_BYTES).unwrap_or(u64::MAX);
        let capacity = usize::try_from(file_len.min(window_len)).unwrap_or(WINDOW_BYTES);
        Ok(Self {
            file,
            file_len,
            pos: 0,
            buf: vec![0_u8; capacity],
            buf_start: 0,
            buf_valid: 0,
        })
    }
}

impl Seek for WindowedFileReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match pos {
            SeekFrom::Start(offset) => {
                self.pos = offset;
                Ok(self.pos)
            }
            SeekFrom::Current(delta) => {
                self.pos = self.pos.checked_add_signed(delta).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "seek before start of file")
                })?;
                Ok(self.pos)
            }
            SeekFrom::End(delta) => {
                // containr probes the input length with `seek(SeekFrom::End(0))`.
                self.pos = self.file_len.checked_add_signed(delta).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "seek before start of file")
                })?;
                Ok(self.pos)
            }
        }
    }
}

impl Read for WindowedFileReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() || self.pos >= self.file_len {
            return Ok(0);
        }

        if let Some(rel) = self.pos.checked_sub(self.buf_start)
            && let Ok(rel_usize) = usize::try_from(rel)
            && rel_usize < self.buf_valid
        {
            let count = out.len().min(self.buf_valid - rel_usize);
            out[..count].copy_from_slice(&self.buf[rel_usize..rel_usize + count]);
            self.pos = self
                .pos
                .saturating_add(u64::try_from(count).unwrap_or(u64::MAX));
            return Ok(count);
        }

        if out.len() >= self.buf.len() {
            self.file.seek(SeekFrom::Start(self.pos))?;
            let count = self.file.read(out)?;
            self.pos = self
                .pos
                .saturating_add(u64::try_from(count).unwrap_or(u64::MAX));
            return Ok(count);
        }

        self.file.seek(SeekFrom::Start(self.pos))?;
        let mut filled = 0_usize;
        while filled < self.buf.len() {
            match self.file.read(&mut self.buf[filled..]) {
                Ok(0) => break,
                Ok(count) => filled += count,
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) => return Err(err),
            }
        }
        self.buf_start = self.pos;
        self.buf_valid = filled;
        if filled == 0 {
            return Ok(0);
        }
        let count = out.len().min(filled);
        out[..count].copy_from_slice(&self.buf[..count]);
        self.pos = self
            .pos
            .saturating_add(u64::try_from(count).unwrap_or(u64::MAX));
        Ok(count)
    }
}

/// Whether `containr`'s zero-copy `StreamCopy` path can carry every stream.
///
/// MP3 in MP4 needs a rebuilt `mp4a` sample entry, VP9 in MP4 needs a `vp09`
/// sample entry, and neither `StreamCopy` planner models those. Everything
/// else inside the MP4/MOV <-> MKV/WebM family is a pure descriptor copy.
fn stream_copy_supports(item: &VideoRemuxItem) -> bool {
    let input = item.input_format();
    let output = item.output_format();
    let family = |format| {
        matches!(
            format,
            VideoContainerFormat::Mp4 | VideoContainerFormat::Mov
        )
    };
    let direct = (input == VideoContainerFormat::Mp4 && family(output))
        || (matches!(
            input,
            VideoContainerFormat::Mkv | VideoContainerFormat::Webm
        ) && matches!(
            output,
            VideoContainerFormat::Mkv | VideoContainerFormat::Webm
        ))
        || (family(input)
            && matches!(
                output,
                VideoContainerFormat::Mkv | VideoContainerFormat::Webm
            ))
        || (matches!(
            input,
            VideoContainerFormat::Mkv | VideoContainerFormat::Webm
        ) && family(output));
    if !direct {
        return false;
    }
    item.streams().iter().all(|stream| {
        stream.codec() != VideoCodec::Mp3 && !(stream.codec() == VideoCodec::Vp9 && family(output))
    })
}

/// Fails before any path is taken when an MKV source carries attachments that
/// the target container cannot represent.
fn reject_mkv_attachments(source: &Path, target: &Path) -> Result<(), PipelineError> {
    let reader = WindowedFileReader::open(source).map_err(|err| PipelineError::Io {
        path: source.to_path_buf(),
        source: err,
    })?;
    let input = containr::mkv::Demuxer::open(reader).map_err(|err| container_err(source, &err))?;
    if input.truncated() {
        return Err(remux_failed(source, "truncated MKV input"));
    }
    if !input.attachments().is_empty() {
        return Err(remux_failed(
            target,
            "MKV attachments have no representation in the target container; refusing to drop data",
        ));
    }
    Ok(())
}

/// Remuxes a single file without re-encoding, rejecting unsupported codecs.
fn remux_file(item: &VideoRemuxItem, target: &File) -> Result<(), PipelineError> {
    let source = item.source();
    let output = item.output_format();
    if matches!(
        item.input_format(),
        VideoContainerFormat::Mkv | VideoContainerFormat::Webm
    ) && !matches!(
        output,
        VideoContainerFormat::Mkv | VideoContainerFormat::Webm
    ) {
        reject_mkv_attachments(source, item.target())?;
    }
    for stream in item.streams() {
        if !stream.codec().supported_in(output) {
            return Err(remux_failed(
                item.target(),
                format!(
                    "codec {} cannot be carried by {}",
                    stream.codec(),
                    output.label()
                ),
            ));
        }
    }
    if !stream_copy_supports(item) {
        let media = demux_video(source, item.input_format())?;
        return mux_video(target, item.target(), output, &media);
    }
    let brand = if output == VideoContainerFormat::Mov {
        Brand::QuickTime
    } else {
        Brand::Isom
    };
    let doc_type = if output == VideoContainerFormat::Webm {
        DocType::WebM
    } else {
        DocType::Matroska
    };
    match (item.input_format(), output) {
        (
            VideoContainerFormat::Mp4 | VideoContainerFormat::Mov,
            VideoContainerFormat::Mp4 | VideoContainerFormat::Mov,
        ) => mp4_to_mp4(source, target, item.target(), brand),
        (
            VideoContainerFormat::Mp4 | VideoContainerFormat::Mov,
            VideoContainerFormat::Mkv | VideoContainerFormat::Webm,
        ) => mp4_to_mkv(source, target, item.target(), doc_type),
        (
            VideoContainerFormat::Mkv | VideoContainerFormat::Webm,
            VideoContainerFormat::Mp4 | VideoContainerFormat::Mov,
        ) => mkv_to_mp4(source, target, item.target(), brand),
        (
            VideoContainerFormat::Mkv | VideoContainerFormat::Webm,
            VideoContainerFormat::Mkv | VideoContainerFormat::Webm,
        ) => mkv_to_mkv(source, target, item.target(), doc_type),
        _ => Err(remux_failed(
            item.target(),
            format!(
                "{} -> {} remuxing is not implemented yet",
                item.input_format(),
                output
            ),
        )),
    }
}

/// A crash-safe staging target: the temp file is created exclusively, written
/// through the owned handle, then published with a rename.
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
            std::fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let name = target.file_name().map_or_else(
            || "remux.mp4".to_owned(),
            |n| n.to_string_lossy().into_owned(),
        );
        let serial = STAGING_SERIAL.fetch_add(1, Ordering::Relaxed);
        let temp = target.with_file_name(format!(
            ".{name}.remux-{}-{serial}-{tag}.tmp",
            std::process::id()
        ));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
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
            .ok_or_else(|| remux_failed(&self.target, "staging file is already committed"))
    }

    fn commit(mut self) -> Result<(), PipelineError> {
        // Close the owned handle before the rename so no writer holds the inode.
        drop(self.file.take());
        std::fs::rename(&self.temp, &self.target).map_err(|source| PipelineError::Io {
            path: self.target.clone(),
            source,
        })
    }
}

impl Drop for StagingTarget {
    fn drop(&mut self) {
        // Only remove the entry this invocation created exclusively.
        if self.file.is_some() {
            let _ = std::fs::remove_file(&self.temp);
        }
    }
}

fn read_packets(
    packets: impl Iterator<Item = Result<Packet, containr::Error>>,
    path: &Path,
) -> Result<Vec<Packet>, PipelineError> {
    packets
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| container_err(path, &err))
}

fn mp4_to_mkv(
    source: &Path,
    target: &File,
    target_path: &Path,
    doc_type: DocType,
) -> Result<(), PipelineError> {
    let reader = WindowedFileReader::open(source).map_err(|source_err| PipelineError::Io {
        path: source.to_path_buf(),
        source: source_err,
    })?;
    let mut input =
        containr::mp4::Demuxer::open(reader).map_err(|err| container_err(source, &err))?;

    let mut track_ids: Vec<u32> = Vec::new();
    let mut plans: Vec<containr::mkv::StreamCopy> = Vec::new();
    for track in input.tracks().iter().filter(|t| !t.chapter_track) {
        let packets = read_packets(
            input
                .track_packets(track.id)
                .map_err(|e| container_err(source, &e))?,
            source,
        )?;
        let plan = match containr::mkv::StreamCopy::plan(track, &packets) {
            Ok(plan) => plan,
            Err(_exact) => {
                containr::mkv::StreamCopy::plan_rounded(track, &packets, ROUNDED_TICK_NS)
                    .map_err(|err| container_err(source, &err))?
            }
        };
        track_ids.push(track.id);
        plans.push(plan);
    }
    if plans.is_empty() {
        return Err(remux_failed(source, "no remuxable tracks in MP4 source"));
    }

    let options = containr::mkv::MkvOptions {
        doc_type,
        timestamp_scale: containr::mkv::StreamCopy::timestamp_scale(&plans),
        ..containr::mkv::MkvOptions::default()
    };

    let mut writer = BufWriter::with_capacity(512 * 1024, target);
    let mut output = containr::mkv::Muxer::new(&mut writer, options)
        .map_err(|err| container_err(target_path, &err))?;

    let mut ids: Vec<(u32, containr::mkv::TrackId)> = Vec::with_capacity(plans.len());
    for (source_id, plan) in track_ids.iter().zip(&plans) {
        let id = output
            .add_track(plan.spec.clone())
            .map_err(|err| container_err(target_path, &err))?;
        ids.push((*source_id, id));
    }
    let tags = containr::mkv::tags_from_mp4(input.metadata())
        .map_err(|err| container_err(source, &err))?;
    output
        .set_metadata(tags)
        .map_err(|err| container_err(target_path, &err))?;
    output
        .set_chapters(input.chapters().to_vec())
        .map_err(|err| container_err(target_path, &err))?;

    let mut packet_buf: Vec<u8> = Vec::new();
    for packet in input.packets() {
        let packet = packet.map_err(|err| container_err(source, &err))?;
        let Some(slot) = ids.iter().position(|(id, _)| *id == packet.track_id) else {
            continue;
        };
        input
            .read_packet_into(&packet, &mut packet_buf)
            .map_err(|err| container_err(source, &err))?;
        if let Some(block) = plans[slot]
            .packet(&packet, &packet_buf)
            .map_err(|err| container_err(source, &err))?
        {
            output
                .write_packet(ids[slot].1, &block)
                .map_err(|err| container_err(target_path, &err))?;
        }
    }
    output
        .finish()
        .map_err(|err| container_err(target_path, &err))?;
    writer.flush().map_err(|source| PipelineError::Io {
        path: target_path.to_path_buf(),
        source,
    })?;
    Ok(())
}

fn mux_mkv_to_mp4<W: io::Read + io::Write + io::Seek>(
    source: &Path,
    target: W,
    target_path: &Path,
    brand: Brand,
) -> Result<(), PipelineError> {
    let reader = WindowedFileReader::open(source).map_err(|source_err| PipelineError::Io {
        path: source.to_path_buf(),
        source: source_err,
    })?;
    let mut input =
        containr::mkv::Demuxer::open(reader).map_err(|err| container_err(source, &err))?;
    if input.truncated() {
        return Err(remux_failed(source, "truncated MKV input"));
    }
    if !input.attachments().is_empty() {
        return Err(remux_failed(
            source,
            "MKV attachments have no MP4 representation; refusing to drop data",
        ));
    }

    let options = containr::mp4::MuxOptions {
        brand,
        faststart: true,
        ..containr::mp4::MuxOptions::default()
    };
    let movie_timescale = options.movie_timescale;

    let mut output = containr::mp4::Muxer::new(target, options)
        .map_err(|err| container_err(target_path, &err))?;

    let mut plans: HashMap<u32, (containr::mp4::StreamCopy, containr::mp4::TrackId)> =
        HashMap::new();
    for track in input.tracks() {
        let mut packets = read_packets(
            input
                .track_packets(track.id)
                .map_err(|e| container_err(source, &e))?,
            source,
        )?;
        // containr's MP4 edit-list planner ignores DiscardPadding when the last
        // block stores a duration. Only fold the pad into that duration when it
        // still covers a whole frame; some muxers (notably FFmpeg) already trim
        // BlockDuration, and subtracting again would shorten the edit twice.
        let default_duration_ns = track
            .matroska
            .as_ref()
            .and_then(|mkv| mkv.default_duration)
            .map(i128::from);
        if let Some(last) = packets.last_mut()
            && let Some(padding) = last.discard_padding
            && padding > 0
            && let Some(duration) = last.duration
        {
            let tick_ns = i128::from(track.timebase.num.max(1));
            let stored_ns = i128::from(duration) * tick_ns;
            let covers_full_frame =
                default_duration_ns.is_some_and(|frame_ns| stored_ns + tick_ns >= frame_ns);
            if covers_full_frame {
                let pad_ticks =
                    u64::try_from(((i128::from(padding) + tick_ns / 2) / tick_ns).max(1))
                        .unwrap_or(u64::MAX);
                last.duration = Some(duration.saturating_sub(pad_ticks).max(1));
            }
        }
        let plan = containr::mp4::StreamCopy::plan(track, &packets, movie_timescale)
            .map_err(|err| container_err(source, &err))?;
        let id = output
            .add_track(plan.spec.clone())
            .map_err(|err| container_err(target_path, &err))?;
        plans.insert(track.id, (plan, id));
    }
    if plans.is_empty() {
        return Err(remux_failed(source, "no remuxable tracks in MKV source"));
    }
    output
        .set_metadata(input.metadata().to_vec())
        .map_err(|err| container_err(target_path, &err))?;
    output
        .set_chapters(containr::mp4::StreamCopy::chapters(input.chapters()))
        .map_err(|err| container_err(target_path, &err))?;

    let mut packet_buf: Vec<u8> = Vec::new();
    for packet in input.packets() {
        let packet = packet.map_err(|err| container_err(source, &err))?;
        let Some((plan, id)) = plans.get(&packet.track_id) else {
            continue;
        };
        input
            .read_packet_into(&packet, &mut packet_buf)
            .map_err(|err| container_err(source, &err))?;
        let block = plan
            .packet(&packet, &packet_buf)
            .map_err(|err| container_err(source, &err))?;
        output
            .write_packet(*id, &block)
            .map_err(|err| container_err(target_path, &err))?;
    }
    output
        .finish()
        .map_err(|err| container_err(target_path, &err))?;
    Ok(())
}

fn mkv_to_mp4(
    source: &Path,
    target: &File,
    target_path: &Path,
    brand: Brand,
) -> Result<(), PipelineError> {
    let source_len = input_size(source);
    if source_len <= IN_MEMORY_MP4_FASTSTART_LIMIT {
        let mut mem = io::Cursor::new(Vec::with_capacity(
            usize::try_from(source_len)
                .unwrap_or(0)
                .saturating_add(256 * 1024),
        ));
        mux_mkv_to_mp4(source, &mut mem, target_path, brand)?;
        let mut writer: &File = target;
        writer
            .write_all(mem.get_ref())
            .map_err(|source| PipelineError::Io {
                path: target_path.to_path_buf(),
                source,
            })?;
        writer.flush().map_err(|source| PipelineError::Io {
            path: target_path.to_path_buf(),
            source,
        })?;
        Ok(())
    } else {
        mux_mkv_to_mp4(source, target, target_path, brand)
    }
}

fn mux_mp4_to_mp4<W: io::Read + io::Write + io::Seek>(
    source: &Path,
    target: W,
    target_path: &Path,
    brand: Brand,
) -> Result<(), PipelineError> {
    let reader = WindowedFileReader::open(source).map_err(|source_err| PipelineError::Io {
        path: source.to_path_buf(),
        source: source_err,
    })?;
    let mut input =
        containr::mp4::Demuxer::open(reader).map_err(|err| container_err(source, &err))?;
    let options = containr::mp4::MuxOptions {
        brand,
        faststart: true,
        ..containr::mp4::MuxOptions::default()
    };
    let movie_timescale = options.movie_timescale;

    let mut output = containr::mp4::Muxer::new(target, options)
        .map_err(|err| container_err(target_path, &err))?;

    let mut plans: HashMap<u32, (containr::mp4::StreamCopy, containr::mp4::TrackId)> =
        HashMap::new();
    for track in input.tracks().iter().filter(|t| !t.chapter_track) {
        let packets = read_packets(
            input
                .track_packets(track.id)
                .map_err(|e| container_err(source, &e))?,
            source,
        )?;
        let plan = containr::mp4::StreamCopy::plan(track, &packets, movie_timescale)
            .map_err(|err| container_err(source, &err))?;
        let id = output
            .add_track(plan.spec.clone())
            .map_err(|err| container_err(target_path, &err))?;
        plans.insert(track.id, (plan, id));
    }
    output
        .set_metadata(input.metadata().to_vec())
        .map_err(|err| container_err(target_path, &err))?;
    output
        .set_chapters(containr::mp4::StreamCopy::chapters(input.chapters()))
        .map_err(|err| container_err(target_path, &err))?;
    let mut packet_buf: Vec<u8> = Vec::new();
    for packet in input.packets() {
        let packet = packet.map_err(|err| container_err(source, &err))?;
        let Some((plan, id)) = plans.get(&packet.track_id) else {
            continue;
        };
        input
            .read_packet_into(&packet, &mut packet_buf)
            .map_err(|err| container_err(source, &err))?;
        let block = plan
            .packet(&packet, &packet_buf)
            .map_err(|err| container_err(source, &err))?;
        output
            .write_packet(*id, &block)
            .map_err(|err| container_err(target_path, &err))?;
    }
    output
        .finish()
        .map_err(|err| container_err(target_path, &err))?;
    Ok(())
}

fn mp4_to_mp4(
    source: &Path,
    target: &File,
    target_path: &Path,
    brand: Brand,
) -> Result<(), PipelineError> {
    let source_len = input_size(source);
    if source_len <= IN_MEMORY_MP4_FASTSTART_LIMIT {
        let mut mem = io::Cursor::new(Vec::with_capacity(
            usize::try_from(source_len)
                .unwrap_or(0)
                .saturating_add(256 * 1024),
        ));
        mux_mp4_to_mp4(source, &mut mem, target_path, brand)?;
        let mut writer: &File = target;
        writer
            .write_all(mem.get_ref())
            .map_err(|source| PipelineError::Io {
                path: target_path.to_path_buf(),
                source,
            })?;
        writer.flush().map_err(|source| PipelineError::Io {
            path: target_path.to_path_buf(),
            source,
        })?;
        Ok(())
    } else {
        mux_mp4_to_mp4(source, target, target_path, brand)
    }
}

fn mkv_to_mkv(
    source: &Path,
    target: &File,
    target_path: &Path,
    doc_type: DocType,
) -> Result<(), PipelineError> {
    let reader = WindowedFileReader::open(source).map_err(|source_err| PipelineError::Io {
        path: source.to_path_buf(),
        source: source_err,
    })?;
    let mut input =
        containr::mkv::Demuxer::open(reader).map_err(|err| container_err(source, &err))?;
    if input.truncated() {
        return Err(remux_failed(source, "truncated MKV input"));
    }
    let mut track_ids: Vec<u32> = Vec::new();
    let mut plans: Vec<containr::mkv::StreamCopy> = Vec::new();
    for track in input.tracks() {
        let packets = read_packets(
            input
                .track_packets(track.id)
                .map_err(|e| container_err(source, &e))?,
            source,
        )?;
        let plan = containr::mkv::StreamCopy::plan(track, &packets)
            .map_err(|err| container_err(source, &err))?;
        track_ids.push(track.id);
        plans.push(plan);
    }
    let options = containr::mkv::MkvOptions {
        doc_type,
        timestamp_scale: containr::mkv::StreamCopy::timestamp_scale(&plans),
        ..containr::mkv::MkvOptions::default()
    };

    let mut writer = BufWriter::with_capacity(512 * 1024, target);
    let mut output = containr::mkv::Muxer::new(&mut writer, options)
        .map_err(|err| container_err(target_path, &err))?;
    let mut ids: Vec<(u32, containr::mkv::TrackId)> = Vec::with_capacity(plans.len());
    for (source_id, plan) in track_ids.iter().zip(&plans) {
        ids.push((
            *source_id,
            output
                .add_track(plan.spec.clone())
                .map_err(|err| container_err(target_path, &err))?,
        ));
    }
    output
        .set_metadata(input.metadata().to_vec())
        .map_err(|err| container_err(target_path, &err))?;
    output
        .set_chapters(input.chapters().to_vec())
        .map_err(|err| container_err(target_path, &err))?;

    // Copy attachments with their UIDs so tags targeting them stay valid.
    let attachments = input.attachments().to_vec();
    for attachment in &attachments {
        let data = input
            .read_attachment(attachment)
            .map_err(|err| container_err(source, &err))?;
        let item = containr::mkv::AttachmentIn {
            name: &attachment.name,
            mime: &attachment.mime,
            description: attachment.description.as_deref(),
            uid: Some(attachment.uid),
            data: &data,
        };
        output
            .attach(&item)
            .map_err(|err| container_err(target_path, &err))?;
    }

    let mut packet_buf: Vec<u8> = Vec::new();
    for packet in input.packets() {
        let packet = packet.map_err(|err| container_err(source, &err))?;
        let Some(slot) = ids.iter().position(|(id, _)| *id == packet.track_id) else {
            continue;
        };
        input
            .read_packet_into(&packet, &mut packet_buf)
            .map_err(|err| container_err(source, &err))?;
        if let Some(block) = plans[slot]
            .packet(&packet, &packet_buf)
            .map_err(|err| container_err(source, &err))?
        {
            output
                .write_packet(ids[slot].1, &block)
                .map_err(|err| container_err(target_path, &err))?;
        }
    }
    output
        .finish()
        .map_err(|err| container_err(target_path, &err))?;
    writer.flush().map_err(|source| PipelineError::Io {
        path: target_path.to_path_buf(),
        source,
    })?;
    Ok(())
}

fn remux_with_staging(item: &VideoRemuxItem, tag: usize) -> Result<(), PipelineError> {
    let staging = StagingTarget::new(item.target(), tag)?;
    remux_file(item, staging.file()?)?;
    staging.commit()
}

/// Executes a validated [`VideoRemuxPlan`] with parallel file-level workers.
///
/// # Errors
///
/// Returns [`PipelineError`] if any container read/write fails. First failure
/// latches and stops remaining workers.
pub fn execute_video_batch(plan: &VideoRemuxPlan) -> Result<VideoRemuxReport, PipelineError> {
    let started = Instant::now();
    let workers = plan.effective_workers().get();
    let total_input: u64 = plan.items().iter().map(|i| input_size(i.source())).sum();

    let next = AtomicUsize::new(0);
    let failure: Mutex<Option<PipelineError>> = Mutex::new(None);
    let tag = AtomicUsize::new(0);

    // A single item does not benefit from a scoped worker: it avoids a thread
    // spawn and keeps the I/O window and packet buffer on the calling stack.
    if plan.items().len() == 1 {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            remux_with_staging(&plan.items()[0], 0)
        }))
        .map_err(|_| PipelineError::WorkerPanicked)?;
        result?;
    } else {
        thread::scope(|scope| -> Result<(), PipelineError> {
            let mut handles = Vec::with_capacity(workers);
            for _ in 0..workers {
                let next = &next;
                let failure = &failure;
                let tag = &tag;
                let items = plan.items();
                handles.push(scope.spawn(move || {
                    loop {
                        if failure.lock().map_or(true, |guard| guard.is_some()) {
                            break;
                        }
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= items.len() {
                            break;
                        }
                        let my_tag = tag.fetch_add(1, Ordering::Relaxed);
                        if let Err(err) = remux_with_staging(&items[index], my_tag) {
                            if let Ok(mut guard) = failure.lock()
                                && guard.is_none()
                            {
                                *guard = Some(err);
                            }
                            break;
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

    let total_output: u64 = plan.items().iter().map(|i| input_size(i.target())).sum();
    let duration = started.elapsed();
    let files =
        std::num::NonZeroUsize::new(plan.items().len()).unwrap_or(std::num::NonZeroUsize::MIN);
    let input_bytes =
        std::num::NonZeroU64::new(total_input.max(1)).unwrap_or(std::num::NonZeroU64::MIN);
    let output_bytes =
        std::num::NonZeroU64::new(total_output.max(1)).unwrap_or(std::num::NonZeroU64::MIN);
    Ok(VideoRemuxReport::new(
        files,
        input_bytes,
        output_bytes,
        duration,
        plan.effective_workers(),
        HardwareBackend::Cpu,
    ))
}

fn input_size(path: &Path) -> u64 {
    std::fs::metadata(path).map_or(0, |m| m.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_file(label: &str, bytes: &[u8]) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = std::env::temp_dir().join(format!("fastconv-remux-{label}-{nanos}"));
        std::fs::write(&path, bytes)?;
        Ok(path)
    }

    #[test]
    fn windowed_reader_seek_and_eof() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_file("seek", b"0123456789")?;
        let mut reader = WindowedFileReader::open(&path)?;
        assert_eq!(reader.seek(SeekFrom::End(0))?, 10);
        assert_eq!(reader.seek(SeekFrom::End(-4))?, 6);
        let mut out = [0_u8; 4];
        assert_eq!(reader.read(&mut out)?, 4);
        assert_eq!(&out, b"6789");
        reader.seek(SeekFrom::Start(0))?;
        for pos in [SeekFrom::Current(-1), SeekFrom::End(-11)] {
            match reader.seek(pos) {
                Ok(result) => return Err(format!("unexpected seek result: {result}").into()),
                Err(err) => assert_eq!(err.kind(), io::ErrorKind::InvalidInput),
            }
        }
        reader.seek(SeekFrom::Start(100))?;
        assert_eq!(reader.read(&mut out)?, 0);
        Ok(())
    }

    #[test]
    fn windowed_reader_backward_seek_hits_buffer() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_file("backward", b"0123456789")?;
        let mut reader = WindowedFileReader::open(&path)?;
        let mut out = [0_u8; 4];
        assert_eq!(reader.read(&mut out)?, 4);
        assert!(reader.buf_valid >= 4);
        reader.seek(SeekFrom::Start(2))?;
        assert_eq!(reader.read(&mut out)?, 4);
        assert_eq!(&out, b"2345");
        Ok(())
    }

    #[test]
    fn windowed_reader_cross_window_and_large_reads() -> Result<(), Box<dyn std::error::Error>> {
        let len = WINDOW_BYTES + 4096;
        let data: Vec<u8> = (0..len)
            .map(|i| u8::try_from(i % 251).unwrap_or(0))
            .collect();
        let path = temp_file("cross-window", &data)?;
        let mut reader = WindowedFileReader::open(&path)?;
        let mut out = vec![0_u8; 4096];
        let mut total = 0_usize;
        loop {
            let count = reader.read(&mut out)?;
            if count == 0 {
                break;
            }
            assert_eq!(&out[..count], &data[total..total + count]);
            total += count;
        }
        assert_eq!(total, len);

        let mut reader = WindowedFileReader::open(&path)?;
        let mut big = vec![0_u8; len];
        let mut total = 0_usize;
        loop {
            let count = reader.read(&mut big[total..])?;
            if count == 0 {
                break;
            }
            total += count;
        }
        assert_eq!(total, len);
        assert_eq!(big, data);
        Ok(())
    }
}

#[cfg(test)]
mod container_tests {
    use super::*;
    use crate::video_container::{
        ContainerMedia, ContainerPacket, ContainerTrack, US_TIMESCALE, demux_video, mux_video,
    };
    use fastconv_domain::VideoStreamInfo;
    use std::num::NonZeroUsize;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(label: &str, extension: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!("fastconv-container-{label}-{nanos}.{extension}"))
    }

    fn avc_config() -> Vec<u8> {
        crate::video_bitstream::extract_avcc_from_annex_b(&[
            0, 0, 1, 0x67, 0x42, 0x00, 0x1E, 0, 0, 1, 0x68, 0xCE,
        ])
        .unwrap_or_default()
    }

    fn h264_aac_media() -> ContainerMedia {
        ContainerMedia {
            tracks: vec![
                ContainerTrack {
                    codec: VideoCodec::H264,
                    width: Some(320),
                    height: Some(240),
                    channels: None,
                    sample_rate: None,
                    config: avc_config(),
                    timescale: US_TIMESCALE,
                },
                ContainerTrack {
                    codec: VideoCodec::Aac,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: vec![0x11, 0x90],
                    timescale: US_TIMESCALE,
                },
            ],
            packets: vec![
                ContainerPacket {
                    track: 0,
                    pts: 0,
                    dts: 0,
                    duration: 33_333,
                    keyframe: true,
                    data: vec![0, 0, 0, 2, 0x65, 0x88],
                },
                ContainerPacket {
                    track: 1,
                    pts: 0,
                    dts: 0,
                    duration: 21_000,
                    keyframe: true,
                    data: vec![0xAA; 8],
                },
                ContainerPacket {
                    track: 0,
                    pts: 33_333,
                    dts: 33_333,
                    duration: 33_333,
                    keyframe: false,
                    data: vec![0, 0, 0, 2, 0x41, 0x22],
                },
                ContainerPacket {
                    track: 1,
                    pts: 21_000,
                    dts: 21_000,
                    duration: 21_000,
                    keyframe: true,
                    data: vec![0xBB; 8],
                },
            ],
        }
    }

    fn vp9_opus_media() -> ContainerMedia {
        ContainerMedia {
            tracks: vec![
                ContainerTrack {
                    codec: VideoCodec::Vp9,
                    width: Some(640),
                    height: Some(360),
                    channels: None,
                    sample_rate: None,
                    config: Vec::new(),
                    timescale: US_TIMESCALE,
                },
                ContainerTrack {
                    codec: VideoCodec::Opus,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: crate::video_bitstream::default_opus_head(2, 0),
                    timescale: US_TIMESCALE,
                },
            ],
            packets: vec![
                ContainerPacket {
                    track: 0,
                    pts: 0,
                    dts: 0,
                    duration: 33_333,
                    keyframe: true,
                    data: vec![0x82, 0x31, 0x01],
                },
                ContainerPacket {
                    track: 1,
                    pts: 0,
                    dts: 0,
                    duration: 20_000,
                    keyframe: true,
                    data: vec![0xFC, 0x01],
                },
            ],
        }
    }

    fn write_container(
        path: &Path,
        format: VideoContainerFormat,
        media: &ContainerMedia,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        mux_video(&file, path, format, media)?;
        drop(file);
        Ok(())
    }

    fn streams(media: &ContainerMedia) -> Vec<VideoStreamInfo> {
        media
            .tracks
            .iter()
            .map(|track| {
                VideoStreamInfo::new(
                    track.codec,
                    if track.codec.is_video() {
                        fastconv_domain::VideoStreamKind::Video
                    } else {
                        fastconv_domain::VideoStreamKind::Audio
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
            .collect()
    }

    fn remux(
        source: &Path,
        target: &Path,
        input: VideoContainerFormat,
        output: VideoContainerFormat,
    ) -> Result<(), PipelineError> {
        let plan = VideoRemuxPlan::new(
            vec![VideoRemuxItem::new(
                source.to_path_buf(),
                target.to_path_buf(),
                input,
                output,
                streams(&demux_video(source, input)?),
            )],
            NonZeroUsize::MIN,
        )
        .map_err(|err| remux_failed(target, err.to_string()))?;
        execute_video_batch(&plan).map(|_| ())
    }

    fn payloads(media: &ContainerMedia, codec: VideoCodec) -> Vec<Vec<u8>> {
        media
            .packets
            .iter()
            .filter(|packet| media.tracks[packet.track].codec == codec)
            .map(|packet| packet.data.clone())
            .collect()
    }

    #[test]
    fn mp4_to_mov_writes_quicktime_brand_and_preserves_packets()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = temp_path("mp4-mov-source", "mp4");
        let target = temp_path("mp4-mov-target", "mov");
        let media = h264_aac_media();
        write_container(&source, VideoContainerFormat::Mp4, &media)?;
        remux(
            &source,
            &target,
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mov,
        )?;
        let bytes = std::fs::read(&target)?;
        assert_eq!(&bytes[4..8], b"ftyp");
        assert_eq!(&bytes[8..12], b"qt  ");
        let out = demux_video(&target, VideoContainerFormat::Mov)?;
        assert_eq!(out.tracks[0].codec, VideoCodec::H264);
        assert_eq!(
            payloads(&out, VideoCodec::H264),
            payloads(&media, VideoCodec::H264)
        );
        assert_eq!(
            payloads(&out, VideoCodec::Aac),
            payloads(&media, VideoCodec::Aac)
        );
        std::fs::remove_file(source)?;
        std::fs::remove_file(target)?;
        Ok(())
    }

    #[test]
    fn mkv_to_webm_writes_webm_doctype_and_preserves_packets()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = temp_path("mkv-webm-source", "mkv");
        let target = temp_path("mkv-webm-target", "webm");
        let media = vp9_opus_media();
        write_container(&source, VideoContainerFormat::Mkv, &media)?;
        remux(
            &source,
            &target,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Webm,
        )?;
        let bytes = std::fs::read(&target)?;
        assert_eq!(&bytes[..4], &[0x1A, 0x45, 0xDF, 0xA3]);
        assert!(bytes.windows(4).any(|window| window == b"webm"));
        let out = demux_video(&target, VideoContainerFormat::Webm)?;
        assert_eq!(out.tracks[0].codec, VideoCodec::Vp9);
        assert_eq!(
            payloads(&out, VideoCodec::Vp9),
            payloads(&media, VideoCodec::Vp9)
        );
        assert_eq!(
            payloads(&out, VideoCodec::Opus),
            payloads(&media, VideoCodec::Opus)
        );
        std::fs::remove_file(source)?;
        std::fs::remove_file(target)?;
        Ok(())
    }

    #[test]
    fn mp4_avi_ts_mkv_mp4_chain_preserves_codecs_and_payloads()
    -> Result<(), Box<dyn std::error::Error>> {
        let media = h264_aac_media();
        let mp4 = temp_path("chain-mp4", "mp4");
        let avi = temp_path("chain-avi", "avi");
        let ts = temp_path("chain-ts", "ts");
        let mkv = temp_path("chain-mkv", "mkv");
        let final_mp4 = temp_path("chain-final", "mp4");
        write_container(&mp4, VideoContainerFormat::Mp4, &media)?;
        let mut source = mp4.clone();
        for (target, format) in [
            (&avi, VideoContainerFormat::Avi),
            (&ts, VideoContainerFormat::Ts),
            (&mkv, VideoContainerFormat::Mkv),
            (&final_mp4, VideoContainerFormat::Mp4),
        ] {
            let input = VideoContainerFormat::from_extension(
                source.extension().and_then(|e| e.to_str()).unwrap_or("mp4"),
            )
            .unwrap_or(VideoContainerFormat::Mp4);
            remux(&source, target, input, format)?;
            source = target.clone();
        }
        let out = demux_video(&final_mp4, VideoContainerFormat::Mp4)?;
        assert_eq!(out.tracks.len(), 2);
        assert_eq!(out.tracks[0].codec, VideoCodec::H264);
        assert_eq!(out.tracks[1].codec, VideoCodec::Aac);
        for (codec, source_payloads) in [
            (VideoCodec::H264, payloads(&media, VideoCodec::H264)),
            (VideoCodec::Aac, payloads(&media, VideoCodec::Aac)),
        ] {
            let target_payloads = payloads(&out, codec);
            assert_eq!(
                target_payloads.len(),
                source_payloads.len(),
                "codec {codec}"
            );
            for (original, remuxed) in source_payloads.iter().zip(&target_payloads) {
                assert!(
                    remuxed.ends_with(original),
                    "codec {codec}: {original:?} vs {remuxed:?}"
                );
            }
        }
        for path in [mp4, avi, ts, mkv, final_mp4] {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    #[test]
    fn webm_and_ts_reject_unsupported_codecs_explicitly() -> Result<(), Box<dyn std::error::Error>>
    {
        let webm = temp_path("reject-webm", "webm");
        let ts = temp_path("reject-ts", "ts");
        let file = File::create(&webm)?;
        assert!(matches!(
            mux_video(&file, &webm, VideoContainerFormat::Webm, &h264_aac_media()),
            Err(PipelineError::RemuxFailed { .. })
        ));
        drop(file);
        let file = File::create(&ts)?;
        assert!(matches!(
            mux_video(&file, &ts, VideoContainerFormat::Ts, &vp9_opus_media()),
            Err(PipelineError::RemuxFailed { .. })
        ));
        drop(file);
        std::fs::remove_file(webm)?;
        std::fs::remove_file(ts)?;
        Ok(())
    }

    fn one_second_h264_aac_media() -> ContainerMedia {
        let mut packets = Vec::new();
        for i in 0..30_u32 {
            let pts = i64::from(i) * 33_333;
            let nal = if i == 0 { 0x65 } else { 0x41 };
            packets.push(ContainerPacket {
                track: 0,
                pts,
                dts: pts,
                duration: 33_333,
                keyframe: i == 0,
                data: vec![0, 0, 0, 4, nal, 0x88, 0x01, 0x02],
            });
        }
        for i in 0..47_u32 {
            let pts = i64::from(i) * 21_333;
            packets.push(ContainerPacket {
                track: 1,
                pts,
                dts: pts,
                duration: 21_333,
                keyframe: true,
                data: vec![0xA0 + u8::try_from(i % 8).unwrap_or(0); 8],
            });
        }
        ContainerMedia {
            tracks: vec![
                ContainerTrack {
                    codec: VideoCodec::H264,
                    width: Some(320),
                    height: Some(240),
                    channels: None,
                    sample_rate: None,
                    config: avc_config(),
                    timescale: US_TIMESCALE,
                },
                ContainerTrack {
                    codec: VideoCodec::Aac,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: vec![0x11, 0x90],
                    timescale: US_TIMESCALE,
                },
            ],
            packets,
        }
    }

    fn one_second_vp9_opus_media() -> ContainerMedia {
        let mut packets = Vec::new();
        for i in 0..30_u32 {
            let pts = i64::from(i) * 33_333;
            packets.push(ContainerPacket {
                track: 0,
                pts,
                dts: pts,
                duration: 33_333,
                keyframe: i == 0,
                data: vec![0x82, 0x31, u8::try_from(i).unwrap_or(0)],
            });
        }
        for i in 0..50_u32 {
            let pts = i64::from(i) * 20_000;
            packets.push(ContainerPacket {
                track: 1,
                pts,
                dts: pts,
                duration: 20_000,
                keyframe: true,
                data: vec![0xFC, u8::try_from(i).unwrap_or(0)],
            });
        }
        ContainerMedia {
            tracks: vec![
                ContainerTrack {
                    codec: VideoCodec::Vp9,
                    width: Some(640),
                    height: Some(360),
                    channels: None,
                    sample_rate: None,
                    config: Vec::new(),
                    timescale: US_TIMESCALE,
                },
                ContainerTrack {
                    codec: VideoCodec::Opus,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: crate::video_bitstream::default_opus_head(2, 0),
                    timescale: US_TIMESCALE,
                },
            ],
            packets,
        }
    }

    /// Presentation span in microseconds. A span (rather than an absolute
    /// timestamp) is used because MPEG-TS deliberately starts at a PCR offset.
    fn track_span_us(media: &ContainerMedia, codec: VideoCodec) -> i64 {
        let mut start = i64::MAX;
        let mut end = i64::MIN;
        for packet in &media.packets {
            if media.tracks[packet.track].codec == codec {
                start = start.min(packet.pts);
                end = end.max(
                    packet
                        .pts
                        .saturating_add(i64::try_from(packet.duration).unwrap_or(0)),
                );
            }
        }
        if start == i64::MAX {
            0
        } else {
            end.saturating_sub(start)
        }
    }

    fn assert_one_second(media: &ContainerMedia, codec: VideoCodec, hop: &str) {
        let span = track_span_us(media, codec);
        assert!(
            (span - 1_000_000).abs() <= 50_000,
            "{hop}: {codec} span is {span} µs, expected ~1_000_000 µs"
        );
    }

    #[test]
    fn one_second_h264_aac_timing_survives_every_container_hop()
    -> Result<(), Box<dyn std::error::Error>> {
        let media = one_second_h264_aac_media();
        let mp4 = temp_path("timing-h264-source", "mp4");
        write_container(&mp4, VideoContainerFormat::Mp4, &media)?;
        assert_one_second(
            &demux_video(&mp4, VideoContainerFormat::Mp4)?,
            VideoCodec::H264,
            "source mp4",
        );

        let mut source = mp4.clone();
        let mut outputs = Vec::new();
        for (format, ext) in [
            (VideoContainerFormat::Mov, "mov"),
            (VideoContainerFormat::Mkv, "mkv"),
            (VideoContainerFormat::Avi, "avi"),
            (VideoContainerFormat::Ts, "ts"),
            (VideoContainerFormat::Mp4, "mp4"),
        ] {
            let target = temp_path("timing-h264-hop", ext);
            let input = VideoContainerFormat::from_extension(
                source.extension().and_then(|e| e.to_str()).unwrap_or("mp4"),
            )
            .unwrap_or(VideoContainerFormat::Mp4);
            remux(&source, &target, input, format)?;
            let out = demux_video(&target, format)?;
            assert_eq!(out.tracks[0].codec, VideoCodec::H264);
            assert_one_second(&out, VideoCodec::H264, ext);
            source = target.clone();
            outputs.push(target);
        }
        for path in std::iter::once(mp4).chain(outputs) {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    #[test]
    fn one_second_vp9_opus_timing_survives_webm_mkv_mp4() -> Result<(), Box<dyn std::error::Error>>
    {
        let media = one_second_vp9_opus_media();
        let webm = temp_path("timing-vp9-webm", "webm");
        let mkv = temp_path("timing-vp9-mkv", "mkv");
        let mp4 = temp_path("timing-vp9-mp4", "mp4");
        write_container(&webm, VideoContainerFormat::Webm, &media)?;
        assert_one_second(
            &demux_video(&webm, VideoContainerFormat::Webm)?,
            VideoCodec::Vp9,
            "webm",
        );
        remux(
            &webm,
            &mkv,
            VideoContainerFormat::Webm,
            VideoContainerFormat::Mkv,
        )?;
        assert_one_second(
            &demux_video(&mkv, VideoContainerFormat::Mkv)?,
            VideoCodec::Vp9,
            "mkv",
        );
        remux(
            &mkv,
            &mp4,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Mp4,
        )?;
        assert_one_second(
            &demux_video(&mp4, VideoContainerFormat::Mp4)?,
            VideoCodec::Vp9,
            "mp4",
        );
        for path in [webm, mkv, mp4] {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    #[test]
    fn stream_copy_paths_exclude_mp3_and_vp9_to_mp4() {
        let stream = |codec| {
            VideoStreamInfo::new(
                codec,
                fastconv_domain::VideoStreamKind::Video,
                None,
                None,
                None,
                None,
                1,
                1000,
                None,
                None,
            )
        };
        let mp3_item = VideoRemuxItem::new(
            PathBuf::from("in.mkv"),
            PathBuf::from("out.mp4"),
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Mp4,
            vec![stream(VideoCodec::Mp3)],
        );
        assert!(!stream_copy_supports(&mp3_item));
        let vp9_item = VideoRemuxItem::new(
            PathBuf::from("in.mkv"),
            PathBuf::from("out.mp4"),
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Mp4,
            vec![stream(VideoCodec::Vp9)],
        );
        assert!(!stream_copy_supports(&vp9_item));
        let plain = VideoRemuxItem::new(
            PathBuf::from("in.mkv"),
            PathBuf::from("out.mp4"),
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Mp4,
            vec![stream(VideoCodec::H264)],
        );
        assert!(stream_copy_supports(&plain));
    }
}
