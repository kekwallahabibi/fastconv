//! Lossless container remuxing: packet/bitstream copy between MP4/MOV and MKV.
//!
//! Uses `containr` as a pure-Rust container library. No codec is decoded or
//! re-encoded. No `ffmpeg`/`ffprobe` subprocess is involved.

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

/// Up to this source size, MKV-to-MP4 remux reads the MKV source once into
/// memory so `containr::mkv::Demuxer::open` (cluster scan) and packet reading
/// do not scan the file through the kernel twice.
const IN_MEMORY_MKV_READ_LIMIT: u64 = 32 * 1024 * 1024;

/// Upper bound on the source size of an MJPEG remux that must take the
/// canonical `demux_video` -> `mux_video` route.
///
/// That route copies every packet payload into `ContainerMedia` before muxing
/// starts, so its peak RAM grows with the source size. MJPEG frames are already
/// compressed JPEG images and do not shrink on demux, so the route has no
/// natural bound; a multi-gigabyte source would otherwise exhaust memory on
/// constrained hosts. Rejecting it up front keeps the route's footprint
/// bounded. Matroska-to-Matroska MJPEG keeps containr's streaming `StreamCopy`
/// path and is not subject to this limit.
const MAX_CANONICAL_MJPEG_SOURCE_BYTES: u64 = 512 * 1024 * 1024;

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
    let matroska = |format| {
        matches!(
            format,
            VideoContainerFormat::Mkv | VideoContainerFormat::Webm
        )
    };
    let direct = (family(input) && family(output))
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
    // FLAC needs a rebuilt `dfLa` entry and PCM a rebuilt QuickTime sample
    // entry; containr's StreamCopy cannot express either across families.
    item.streams().iter().all(|stream| {
        stream.codec() != VideoCodec::Mp3
            // containr maps neither a QuickTime `jpeg` sample entry to
            // `V_MJPEG` nor `V_MJPEG` back to a sample entry, so cross-family
            // MJPEG must use the canonical route. A Matroska-to-Matroska copy
            // keeps the raw `CodecID`, so MJPEG stays stream-copied there.
            && !(stream.codec() == VideoCodec::Mjpeg
                && !(matroska(input) && matroska(output)))
            && !matches!(stream.codec(), VideoCodec::Flac | VideoCodec::Pcm)
            && !(stream.codec() == VideoCodec::Vp9 && family(output))
    })
}

/// Codecs the canonical `demux_video` -> `mux_video` route can actually
/// preserve. Codecs outside this set are supported by the container format but
/// silently dropped by `mkv_codec`, so they must be rejected before demuxing.
fn canonical_remux_supports(codec: VideoCodec) -> bool {
    matches!(
        codec,
        VideoCodec::H264
            | VideoCodec::Hevc
            | VideoCodec::Av1
            | VideoCodec::Vp9
            | VideoCodec::Mjpeg
            | VideoCodec::Aac
            | VideoCodec::Opus
            | VideoCodec::Vorbis
            | VideoCodec::Mp3
            | VideoCodec::Flac
            | VideoCodec::Pcm
    )
}

/// Rejects an MJPEG source too large for the canonical route before any
/// demuxing happens. `demux_video` holds the whole source in `ContainerMedia`,
/// so failing here avoids an allocator exhaustion deep in the demuxer.
fn check_canonical_mjpeg_source_size(source: &Path, source_len: u64) -> Result<(), PipelineError> {
    if source_len > MAX_CANONICAL_MJPEG_SOURCE_BYTES {
        return Err(remux_failed(
            source,
            format!(
                "MJPEG source is {source_len} bytes, above the \
                 {MAX_CANONICAL_MJPEG_SOURCE_BYTES}-byte limit for the canonical remux route; \
                 Matroska-to-Matroska copies stay streamed, otherwise split the source"
            ),
        ));
    }
    Ok(())
}

/// Fails before any path is taken when an MKV source carries metadata that the
/// canonical `demux_video` -> `mux_video` route cannot preserve.
///
/// `ContainerMedia` has no chapter, tag or attachment field, so that route
/// would silently discard them for every codec it accepts. Stream copy keeps
/// them, so only the canonical route needs this guard.
fn reject_unrepresentable_mkv_metadata(source: &Path, target: &Path) -> Result<(), PipelineError> {
    let reader = WindowedFileReader::open(source).map_err(|err| PipelineError::Io {
        path: source.to_path_buf(),
        source: err,
    })?;
    let input = containr::mkv::Demuxer::open(reader).map_err(|err| container_err(source, &err))?;
    if input.truncated() {
        return Err(remux_failed(source, "truncated MKV input"));
    }
    if !input.chapters().is_empty() {
        return Err(remux_failed(
            target,
            "MKV chapters cannot be preserved on the canonical remux route; refusing to drop data",
        ));
    }
    if !input.metadata().is_empty() {
        return Err(remux_failed(
            target,
            "MKV tags cannot be preserved on the canonical remux route; refusing to drop data",
        ));
    }
    if !input.attachments().is_empty() {
        return Err(remux_failed(
            target,
            "MKV attachments have no representation in the target container; refusing to drop data",
        ));
    }
    Ok(())
}

/// Fails before any path is taken when an MP4/MOV source carries title or
/// chapter metadata that the canonical `demux_video` -> `mux_video` route
/// cannot carry (`ContainerMedia` has no metadata or chapter field). Stream
/// copy keeps both, so only the canonical route needs this guard.
fn reject_unrepresentable_mp4_metadata(source: &Path, target: &Path) -> Result<(), PipelineError> {
    let reader = WindowedFileReader::open(source).map_err(|err| PipelineError::Io {
        path: source.to_path_buf(),
        source: err,
    })?;
    let input = containr::mp4::Demuxer::open(reader).map_err(|err| container_err(source, &err))?;
    if !input.chapters().is_empty() {
        return Err(remux_failed(
            target,
            "MP4/MOV chapters cannot be preserved on the canonical remux route; refusing to drop data",
        ));
    }
    if !input.metadata().is_empty() {
        return Err(remux_failed(
            target,
            "MP4/MOV title/tag metadata cannot be preserved on the canonical remux route; refusing to drop data",
        ));
    }
    Ok(())
}

/// Remuxes a single file without re-encoding, rejecting unsupported codecs.
fn remux_file(item: &VideoRemuxItem, target: &File) -> Result<(), PipelineError> {
    let source = item.source();
    let output = item.output_format();
    // Select the route first: only the canonical path loses attachments, and
    // even a same-family conversion takes it when a codec disables stream copy.
    let direct = stream_copy_supports(item);
    if matches!(
        item.input_format(),
        VideoContainerFormat::Mkv | VideoContainerFormat::Webm
    ) && !direct
    {
        reject_unrepresentable_mkv_metadata(source, item.target())?;
    }
    if matches!(
        item.input_format(),
        VideoContainerFormat::Mp4 | VideoContainerFormat::Mov
    ) && !direct
    {
        reject_unrepresentable_mp4_metadata(source, item.target())?;
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
    if !direct {
        for stream in item.streams() {
            if !canonical_remux_supports(stream.codec()) {
                return Err(remux_failed(
                    item.target(),
                    format!(
                        "{} stream cannot be preserved on the canonical remux route; refusing to drop track",
                        stream.codec()
                    ),
                ));
            }
        }
        if item
            .streams()
            .iter()
            .any(|stream| stream.codec() == VideoCodec::Mjpeg)
        {
            check_canonical_mjpeg_source_size(source, input_size(source))?;
        }
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

    let mut writer = BufWriter::with_capacity(1024 * 1024, target);
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

    let mut packet_buf: Vec<u8> = Vec::with_capacity(256 * 1024);
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

fn mux_mkv_to_mp4<R: io::Read + io::Seek, W: io::Read + io::Write + io::Seek>(
    source: &Path,
    reader: R,
    target: W,
    target_path: &Path,
    brand: Brand,
) -> Result<(), PipelineError> {
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

    // Tracks are few; a linear Vec beats a hashed lookup per packet.
    let mut plans: Vec<(u32, containr::mp4::StreamCopy, containr::mp4::TrackId)> =
        Vec::with_capacity(input.tracks().len());
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
        plans.push((track.id, plan, id));
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

    let mut packet_buf: Vec<u8> = Vec::with_capacity(256 * 1024);
    for packet in input.packets() {
        let packet = packet.map_err(|err| container_err(source, &err))?;
        let Some(entry) = plans.iter().find(|(id, _, _)| *id == packet.track_id) else {
            continue;
        };
        input
            .read_packet_into(&packet, &mut packet_buf)
            .map_err(|err| container_err(source, &err))?;
        let block = entry
            .1
            .packet(&packet, &packet_buf)
            .map_err(|err| container_err(source, &err))?;
        output
            .write_packet(entry.2, &block)
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
        if source_len <= IN_MEMORY_MKV_READ_LIMIT {
            let bytes = std::fs::read(source).map_err(|source_err| PipelineError::Io {
                path: source.to_path_buf(),
                source: source_err,
            })?;
            mux_mkv_to_mp4(source, io::Cursor::new(bytes), &mut mem, target_path, brand)?;
        } else {
            let reader =
                WindowedFileReader::open(source).map_err(|source_err| PipelineError::Io {
                    path: source.to_path_buf(),
                    source: source_err,
                })?;
            mux_mkv_to_mp4(source, reader, &mut mem, target_path, brand)?;
        }
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
        let reader = WindowedFileReader::open(source).map_err(|source_err| PipelineError::Io {
            path: source.to_path_buf(),
            source: source_err,
        })?;
        mux_mkv_to_mp4(source, reader, target, target_path, brand)
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

    // Tracks are few; a linear Vec beats a hashed lookup per packet.
    let mut plans: Vec<(u32, containr::mp4::StreamCopy, containr::mp4::TrackId)> =
        Vec::with_capacity(input.tracks().len());
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
        plans.push((track.id, plan, id));
    }
    output
        .set_metadata(input.metadata().to_vec())
        .map_err(|err| container_err(target_path, &err))?;
    output
        .set_chapters(containr::mp4::StreamCopy::chapters(input.chapters()))
        .map_err(|err| container_err(target_path, &err))?;
    let mut packet_buf: Vec<u8> = Vec::with_capacity(256 * 1024);
    for packet in input.packets() {
        let packet = packet.map_err(|err| container_err(source, &err))?;
        let Some(entry) = plans.iter().find(|(id, _, _)| *id == packet.track_id) else {
            continue;
        };
        input
            .read_packet_into(&packet, &mut packet_buf)
            .map_err(|err| container_err(source, &err))?;
        let block = entry
            .1
            .packet(&packet, &packet_buf)
            .map_err(|err| container_err(source, &err))?;
        output
            .write_packet(entry.2, &block)
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

    let mut writer = BufWriter::with_capacity(1024 * 1024, target);
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

    let mut packet_buf: Vec<u8> = Vec::with_capacity(256 * 1024);
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
                    presentation_edit: None,
                    pcm_format: None,
                },
                ContainerTrack {
                    codec: VideoCodec::Aac,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: vec![0x11, 0x90],
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                    pcm_format: None,
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
                    presentation_edit: None,
                    pcm_format: None,
                },
                ContainerTrack {
                    codec: VideoCodec::Opus,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: crate::video_bitstream::default_opus_head(2, 0),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                    pcm_format: None,
                },
            ],
            packets: vec![
                ContainerPacket {
                    track: 0,
                    pts: 0,
                    dts: 0,
                    duration: 33_333,
                    keyframe: true,
                    data: vec![0x92, 0x49, 0x83, 0x42, 0x00],
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

    fn vorbis_media() -> Result<ContainerMedia, Box<dyn std::error::Error>> {
        let mut ident = vec![0_u8; 30];
        ident[0] = 0x01;
        ident[1..7].copy_from_slice(b"vorbis");
        ident[11] = 2;
        ident[12..16].copy_from_slice(&48_000_u32.to_le_bytes());
        ident[28] = 0xB8;
        ident[29] = 1;
        let config = crate::video_bitstream::pack_xiph_laced_vorbis_headers(
            &ident,
            b"\x03vorbis",
            b"\x05vorbis",
        )?;
        Ok(ContainerMedia {
            tracks: vec![
                ContainerTrack {
                    codec: VideoCodec::Vp9,
                    width: Some(640),
                    height: Some(360),
                    channels: None,
                    sample_rate: None,
                    config: Vec::new(),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                    pcm_format: None,
                },
                ContainerTrack {
                    codec: VideoCodec::Vorbis,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config,
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                    pcm_format: None,
                },
            ],
            packets: vec![
                ContainerPacket {
                    track: 0,
                    pts: 0,
                    dts: 0,
                    duration: 33_333,
                    keyframe: true,
                    data: vec![0x92, 0x49, 0x83, 0x42, 0x00],
                },
                ContainerPacket {
                    track: 1,
                    pts: 5_000,
                    dts: 5_000,
                    duration: 23_000,
                    keyframe: true,
                    data: vec![0x00, 0x11, 0x22],
                },
            ],
        })
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
    fn mkv_webm_vorbis_stream_copy_preserves_codec_private_and_timestamps()
    -> Result<(), Box<dyn std::error::Error>> {
        let media = vorbis_media()?;
        let config = media.tracks[1].config.clone();
        let timings: Vec<(i64, i64, u64)> = media
            .packets
            .iter()
            .map(|packet| (packet.pts, packet.dts, packet.duration))
            .collect();
        let source = temp_path("mkv-vorbis-source", "mkv");
        let webm = temp_path("vorbis-webm-target", "webm");
        let back = temp_path("vorbis-mkv-target", "mkv");
        write_container(&source, VideoContainerFormat::Mkv, &media)?;
        remux(
            &source,
            &webm,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Webm,
        )?;
        let out = demux_video(&webm, VideoContainerFormat::Webm)?;
        assert_eq!(out.tracks[1].codec, VideoCodec::Vorbis);
        assert_eq!(out.tracks[1].config, config);
        assert_eq!(
            payloads(&out, VideoCodec::Vorbis),
            payloads(&media, VideoCodec::Vorbis)
        );
        assert_eq!(
            out.packets
                .iter()
                .map(|packet| (packet.pts, packet.dts, packet.duration))
                .collect::<Vec<_>>(),
            timings
        );

        remux(
            &webm,
            &back,
            VideoContainerFormat::Webm,
            VideoContainerFormat::Mkv,
        )?;
        let back_media = demux_video(&back, VideoContainerFormat::Mkv)?;
        assert_eq!(back_media.tracks[1].codec, VideoCodec::Vorbis);
        assert_eq!(back_media.tracks[1].config, config);
        assert_eq!(
            back_media
                .packets
                .iter()
                .map(|packet| (packet.pts, packet.dts, packet.duration))
                .collect::<Vec<_>>(),
            timings
        );
        std::fs::remove_file(source)?;
        std::fs::remove_file(webm)?;
        std::fs::remove_file(back)?;
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
                    presentation_edit: None,
                    pcm_format: None,
                },
                ContainerTrack {
                    codec: VideoCodec::Aac,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: vec![0x11, 0x90],
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                    pcm_format: None,
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
                data: vec![0x92, 0x49, 0x83, 0x42, 0x00, u8::try_from(i).unwrap_or(0)],
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
                    presentation_edit: None,
                    pcm_format: None,
                },
                ContainerTrack {
                    codec: VideoCodec::Opus,
                    width: None,
                    height: None,
                    channels: Some(2),
                    sample_rate: Some(48_000),
                    config: crate::video_bitstream::default_opus_head(2, 0),
                    timescale: US_TIMESCALE,
                    presentation_edit: None,
                    pcm_format: None,
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

    fn mjpeg_frame(width: u16, height: u16, seed: u8) -> Result<Vec<u8>, String> {
        let pixels: Vec<u8> = (0..usize::from(width) * usize::from(height) * 3)
            .map(|index| u8::try_from(index % 251).unwrap_or(0).wrapping_add(seed))
            .collect();
        let mut jpeg = Vec::new();
        jpeg_encoder::Encoder::new(&mut jpeg, 80)
            .encode(&pixels, width, height, jpeg_encoder::ColorType::Rgb)
            .map_err(|err| format!("MJPEG fixture encode failed: {err}"))?;
        Ok(jpeg)
    }

    fn mjpeg_media() -> Result<ContainerMedia, String> {
        let track = ContainerTrack {
            codec: VideoCodec::Mjpeg,
            width: Some(64),
            height: Some(48),
            channels: None,
            sample_rate: None,
            config: Vec::new(),
            timescale: US_TIMESCALE,
            presentation_edit: None,
            pcm_format: None,
        };
        let mut packets = Vec::new();
        for index in 0_u64..3 {
            packets.push(ContainerPacket {
                track: 0,
                pts: i64::try_from(index).unwrap_or(0) * 40_000,
                dts: i64::try_from(index).unwrap_or(0) * 40_000,
                duration: 40_000,
                keyframe: true,
                data: mjpeg_frame(64, 48, u8::try_from(index).unwrap_or(0))?,
            });
        }
        Ok(ContainerMedia {
            tracks: vec![track],
            packets,
        })
    }

    #[test]
    fn mjpeg_remux_roundtrips_avi_mov_mkv() -> Result<(), Box<dyn std::error::Error>> {
        let media = mjpeg_media()?;
        let expected = payloads(&media, VideoCodec::Mjpeg);
        let avi = temp_path("mjpeg-remux", "avi");
        let mov = temp_path("mjpeg-remux", "mov");
        let mkv = temp_path("mjpeg-remux", "mkv");
        let back = temp_path("mjpeg-remux", "avi");
        write_container(&avi, VideoContainerFormat::Avi, &media)?;
        for (source, target, input, output) in [
            (
                &avi,
                &mov,
                VideoContainerFormat::Avi,
                VideoContainerFormat::Mov,
            ),
            (
                &mov,
                &mkv,
                VideoContainerFormat::Mov,
                VideoContainerFormat::Mkv,
            ),
            (
                &mkv,
                &back,
                VideoContainerFormat::Mkv,
                VideoContainerFormat::Avi,
            ),
        ] {
            remux(source, target, input, output)?;
            let out = demux_video(target, output)?;
            assert_eq!(out.tracks[0].codec, VideoCodec::Mjpeg);
            assert_eq!(payloads(&out, VideoCodec::Mjpeg), expected);
            assert!(
                out.packets.iter().all(|packet| packet.keyframe),
                "MJPEG remux must preserve every keyframe flag"
            );
        }
        for path in [avi, mov, mkv, back] {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    #[test]
    fn mjpeg_rejects_iso_mp4_and_stream_copy() -> Result<(), Box<dyn std::error::Error>> {
        let media = mjpeg_media()?;
        let mp4 = temp_path("mjpeg-mp4-reject", "mp4");
        let err = match write_container(&mp4, VideoContainerFormat::Mp4, &media) {
            Ok(()) => return Err("MJPEG must be rejected in ISO MP4".into()),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("MJPEG") && err.to_string().contains("MP4"),
            "unexpected error: {err}"
        );

        let stream = VideoStreamInfo::new(
            VideoCodec::Mjpeg,
            fastconv_domain::VideoStreamKind::Video,
            Some(64),
            Some(48),
            None,
            None,
            1,
            US_TIMESCALE,
            None,
            None,
        );
        let item = VideoRemuxItem::new(
            PathBuf::from("in.mkv"),
            PathBuf::from("out.mov"),
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Mov,
            vec![stream.clone()],
        );
        assert!(!stream_copy_supports(&item));

        // containr cannot map V_MJPEG to a QuickTime `jpeg` sample entry or
        // back, so both cross-family directions must stay on the bounded
        // canonical route.
        let cross_family_item = VideoRemuxItem::new(
            PathBuf::from("in.mov"),
            PathBuf::from("out.mkv"),
            VideoContainerFormat::Mov,
            VideoContainerFormat::Mkv,
            vec![stream.clone()],
        );
        assert!(!stream_copy_supports(&cross_family_item));

        // A Matroska copy keeps the raw CodecID, so MJPEG stays streamed and
        // never loads the whole source into `ContainerMedia`.
        let matroska_copy_item = VideoRemuxItem::new(
            PathBuf::from("in.mkv"),
            PathBuf::from("out-copy.mkv"),
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Mkv,
            vec![stream],
        );
        assert!(stream_copy_supports(&matroska_copy_item));
        assert!(canonical_remux_supports(VideoCodec::Mjpeg));
        Ok(())
    }

    #[test]
    fn mjpeg_mkv_to_mkv_stream_copy_preserves_payloads() -> Result<(), Box<dyn std::error::Error>> {
        let media = mjpeg_media()?;
        let expected = payloads(&media, VideoCodec::Mjpeg);
        let source = temp_path("mjpeg-mkv-copy-source", "mkv");
        let target = temp_path("mjpeg-mkv-copy-target", "mkv");
        write_container(&source, VideoContainerFormat::Mkv, &media)?;
        remux(
            &source,
            &target,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Mkv,
        )?;
        let out = demux_video(&target, VideoContainerFormat::Mkv)?;
        assert_eq!(out.tracks[0].codec, VideoCodec::Mjpeg);
        assert_eq!(payloads(&out, VideoCodec::Mjpeg), expected);
        assert!(
            out.packets.iter().all(|packet| packet.keyframe),
            "MJPEG stream copy must keep every frame a keyframe"
        );
        std::fs::remove_file(source)?;
        std::fs::remove_file(target)?;
        Ok(())
    }

    #[test]
    fn oversized_mjpeg_canonical_source_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let source = Path::new("oversized.mjpeg");
        assert!(
            check_canonical_mjpeg_source_size(source, MAX_CANONICAL_MJPEG_SOURCE_BYTES).is_ok()
        );
        let err = match check_canonical_mjpeg_source_size(
            source,
            MAX_CANONICAL_MJPEG_SOURCE_BYTES.saturating_add(1),
        ) {
            Ok(()) => return Err("one byte over the bound must be rejected".into()),
            Err(err) => err,
        };
        let message = err.to_string();
        assert!(message.contains("MJPEG"), "unexpected error: {message}");
        assert!(
            message.contains("canonical remux route"),
            "unexpected error: {message}"
        );
        Ok(())
    }

    fn write_mov_with_chapter(
        path: &Path,
        media: &ContainerMedia,
        title: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let base = temp_path("mov-chapter-base", "mp4");
        write_container(&base, VideoContainerFormat::Mp4, media)?;
        let reader = WindowedFileReader::open(&base)?;
        let mut input = containr::mp4::Demuxer::open(reader)?;
        // Faststart reads mdat back through the same handle.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let options = containr::mp4::MuxOptions {
            brand: Brand::QuickTime,
            faststart: true,
            ..containr::mp4::MuxOptions::default()
        };
        let mut output = containr::mp4::Muxer::new(&file, options)?;
        let mut plans = Vec::new();
        for track in input.tracks().iter().filter(|t| !t.chapter_track) {
            let packets = read_packets(input.track_packets(track.id)?, &base)?;
            let plan = containr::mp4::StreamCopy::plan(track, &packets, options.movie_timescale)?;
            let id = output.add_track(plan.spec.clone())?;
            plans.push((track.id, plan, id));
        }
        output.set_chapters(vec![containr::Chapter {
            start: 0,
            duration: Some(33_333),
            timescale: US_TIMESCALE,
            title: title.to_owned(),
            source: containr::ChapterSource::Chpl,
        }])?;
        let mut buf = Vec::new();
        for packet in input.packets() {
            let packet = packet?;
            let Some((_, plan, id)) = plans
                .iter()
                .find(|(track_id, _, _)| *track_id == packet.track_id)
            else {
                continue;
            };
            input.read_packet_into(&packet, &mut buf)?;
            output.write_packet(*id, &plan.packet(&packet, &buf)?)?;
        }
        output.finish()?;
        drop(file);
        std::fs::remove_file(base)?;
        Ok(())
    }

    #[test]
    fn mov_to_mp4_direct_copy_preserves_chapters() -> Result<(), Box<dyn std::error::Error>> {
        let source = temp_path("chapter-mov", "mov");
        let target = temp_path("chapter-mp4", "mp4");
        write_mov_with_chapter(&source, &h264_aac_media(), "Intro")?;
        remux(
            &source,
            &target,
            VideoContainerFormat::Mov,
            VideoContainerFormat::Mp4,
        )?;
        let reader = WindowedFileReader::open(&target)?;
        let demuxer = containr::mp4::Demuxer::open(reader)?;
        assert!(
            demuxer
                .chapters()
                .iter()
                .any(|chapter| chapter.title == "Intro"),
            "chapters were dropped from the MOV -> MP4 remux"
        );
        std::fs::remove_file(source)?;
        std::fs::remove_file(target)?;
        Ok(())
    }

    fn h264_mp3_media() -> ContainerMedia {
        let mut media = h264_aac_media();
        media.tracks[1].codec = VideoCodec::Mp3;
        media.tracks[1].config = Vec::new();
        media
    }

    fn write_mkv_with_attachment(
        path: &Path,
        media: &ContainerMedia,
        name: &str,
        data: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let base = temp_path("mkv-attachment-base", "mkv");
        write_container(&base, VideoContainerFormat::Mkv, media)?;
        let reader = WindowedFileReader::open(&base)?;
        let mut input = containr::mkv::Demuxer::open(reader)?;
        let file = File::create(path)?;
        let mut writer = BufWriter::new(&file);
        let options = containr::mkv::MkvOptions {
            doc_type: DocType::Matroska,
            timestamp_scale: crate::video_container::MKV_TICK_NS,
            ..containr::mkv::MkvOptions::default()
        };
        let mut output = containr::mkv::Muxer::new(&mut writer, options)?;
        let mut plans = Vec::new();
        for track in input.tracks() {
            let packets = read_packets(input.track_packets(track.id)?, &base)?;
            let plan = containr::mkv::StreamCopy::plan(track, &packets)?;
            let id = output.add_track(plan.spec.clone())?;
            plans.push((track.id, plan, id));
        }
        output.attach(&containr::mkv::AttachmentIn {
            name,
            mime: "text/plain",
            description: None,
            uid: Some(7),
            data,
        })?;
        let mut buf = Vec::new();
        for packet in input.packets() {
            let packet = packet?;
            let Some((_, plan, id)) = plans
                .iter()
                .find(|(track_id, _, _)| *track_id == packet.track_id)
            else {
                continue;
            };
            input.read_packet_into(&packet, &mut buf)?;
            if let Some(block) = plan.packet(&packet, &buf)? {
                output.write_packet(*id, &block)?;
            }
        }
        output.finish()?;
        writer.flush()?;
        drop(writer);
        std::fs::remove_file(base)?;
        Ok(())
    }

    #[test]
    fn mkv_to_mkv_with_mp3_rejects_attachments_before_muxing()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = temp_path("mkv-mp3-attachment", "mkv");
        let target = temp_path("mkv-mp3-attachment-out", "mkv");
        write_mkv_with_attachment(&source, &h264_mp3_media(), "note.txt", b"keep me")?;
        // MP3 disables stream copy, so even MKV -> MKV takes the canonical path.
        let result = remux(
            &source,
            &target,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Mkv,
        );
        assert!(matches!(result, Err(PipelineError::RemuxFailed { .. })));
        assert!(!target.exists(), "no target was published after rejection");
        std::fs::remove_file(source)?;
        Ok(())
    }

    fn h264_mp3_vorbis_media() -> Result<ContainerMedia, Box<dyn std::error::Error>> {
        let mut media = h264_mp3_media();
        let vorbis = vorbis_media()?;
        let mut packet = vorbis.packets[1].clone();
        packet.track = media.tracks.len();
        media.tracks.push(vorbis.tracks[1].clone());
        media.packets.push(packet);
        Ok(media)
    }

    fn write_mkv_with_chapter_or_tag(
        path: &Path,
        media: &ContainerMedia,
        chapter: bool,
        tag: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let base = temp_path("mkv-metadata-base", "mkv");
        write_container(&base, VideoContainerFormat::Mkv, media)?;
        let reader = WindowedFileReader::open(&base)?;
        let mut input = containr::mkv::Demuxer::open(reader)?;
        let file = File::create(path)?;
        let mut writer = BufWriter::new(&file);
        let options = containr::mkv::MkvOptions {
            doc_type: DocType::Matroska,
            timestamp_scale: crate::video_container::MKV_TICK_NS,
            ..containr::mkv::MkvOptions::default()
        };
        let mut output = containr::mkv::Muxer::new(&mut writer, options)?;
        let mut plans = Vec::new();
        for track in input.tracks() {
            let packets = read_packets(input.track_packets(track.id)?, &base)?;
            let plan = containr::mkv::StreamCopy::plan(track, &packets)?;
            let id = output.add_track(plan.spec.clone())?;
            plans.push((track.id, plan, id));
        }
        if chapter {
            output.set_chapters(vec![containr::Chapter {
                start: 0,
                duration: Some(1_000_000_000),
                timescale: 1_000_000_000,
                title: "Intro".to_owned(),
                source: containr::ChapterSource::Matroska {
                    edition_uid: None,
                    chapter_uid: 1,
                    depth: 0,
                },
            }])?;
        }
        if tag {
            output.set_metadata(vec![containr::MetadataItem {
                scope: containr::MetadataScope::Tag(containr::TagScope {
                    target_type_value: 50,
                    target: containr::TagTarget::Segment,
                    binary: false,
                }),
                key: containr::MetadataKey::Named(b"TITLE".to_vec()),
                value: b"Sample".to_vec(),
                data: Vec::new(),
            }])?;
        }
        let mut buf = Vec::new();
        for packet in input.packets() {
            let packet = packet?;
            let Some((_, plan, id)) = plans
                .iter()
                .find(|(track_id, _, _)| *track_id == packet.track_id)
            else {
                continue;
            };
            input.read_packet_into(&packet, &mut buf)?;
            if let Some(block) = plan.packet(&packet, &buf)? {
                output.write_packet(*id, &block)?;
            }
        }
        output.finish()?;
        writer.flush()?;
        drop(writer);
        std::fs::remove_file(base)?;
        Ok(())
    }

    fn write_mp4_with_chapter_or_tag(
        path: &Path,
        media: &ContainerMedia,
        chapter: bool,
        tag: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let base = temp_path("mp4-metadata-base", "mp4");
        write_container(&base, VideoContainerFormat::Mp4, media)?;
        let reader = WindowedFileReader::open(&base)?;
        let mut input = containr::mp4::Demuxer::open(reader)?;
        let file = File::create(path)?;
        let options = containr::mp4::MuxOptions {
            brand: Brand::Isom,
            faststart: false,
            ..containr::mp4::MuxOptions::default()
        };
        let movie_timescale = options.movie_timescale;
        let mut output = containr::mp4::Muxer::new(file, options)?;
        let mut plans = Vec::new();
        for track in input.tracks() {
            let packets = read_packets(input.track_packets(track.id)?, &base)?;
            let plan = containr::mp4::StreamCopy::plan(track, &packets, movie_timescale)?;
            let id = output.add_track(plan.spec.clone())?;
            plans.push((track.id, plan, id));
        }
        if chapter {
            output.set_chapters(vec![containr::Chapter {
                start: 0,
                duration: Some(1_000_000_000),
                timescale: 1_000_000_000,
                title: "Intro".to_owned(),
                source: containr::ChapterSource::Chpl,
            }])?;
        }
        if tag {
            output.set_metadata(vec![containr::MetadataItem {
                scope: containr::MetadataScope::Tag(containr::TagScope {
                    target_type_value: 50,
                    target: containr::TagTarget::Segment,
                    binary: false,
                }),
                key: containr::MetadataKey::Named(b"TITLE".to_vec()),
                value: b"Sample".to_vec(),
                data: Vec::new(),
            }])?;
        }
        let mut buf = Vec::new();
        for packet in input.packets() {
            let packet = packet?;
            let Some((_, plan, id)) = plans
                .iter()
                .find(|(track_id, _, _)| *track_id == packet.track_id)
            else {
                continue;
            };
            input.read_packet_into(&packet, &mut buf)?;
            let block = plan.packet(&packet, &buf)?;
            output.write_packet(*id, &block)?;
        }
        output.finish()?;
        std::fs::remove_file(base)?;
        Ok(())
    }

    #[test]
    fn mp4_canonical_route_rejects_chapters_and_tags_with_mp3()
    -> Result<(), Box<dyn std::error::Error>> {
        let media = h264_mp3_media();
        for (name, chapter, tag) in [("chapter", true, false), ("tag", false, true)] {
            let source = temp_path(&format!("mp4-{name}-source"), "mov");
            let target = temp_path(&format!("mp4-{name}-target"), "mp4");
            write_mp4_with_chapter_or_tag(&source, &media, chapter, tag)?;
            let reader = WindowedFileReader::open(&source)?;
            let demuxer = containr::mp4::Demuxer::open(reader)?;
            // containr may emit both a chapter track and a Nero chapter for
            // one logical chapter; only presence matters to the guard.
            assert_eq!(!demuxer.chapters().is_empty(), chapter);
            assert_eq!(!demuxer.metadata().is_empty(), tag);

            // MP3 disables stream copy, so the MOV source takes the canonical
            // route; chapters and tags must fail before a target is published.
            let result = remux(
                &source,
                &target,
                VideoContainerFormat::Mov,
                VideoContainerFormat::Mp4,
            );
            assert!(
                matches!(result, Err(PipelineError::RemuxFailed { .. })),
                "{name} metadata was dropped instead of rejected: {result:?}"
            );
            assert!(
                !target.exists(),
                "no target was published after {name} rejection"
            );
            std::fs::remove_file(source)?;
        }
        Ok(())
    }

    #[test]
    fn mp4_canonical_route_without_metadata_still_remuxes() -> Result<(), Box<dyn std::error::Error>>
    {
        let source = temp_path("mp4-metadata-free-source", "mov");
        let target = temp_path("mp4-metadata-free-target", "mp4");
        write_container(&source, VideoContainerFormat::Mov, &h264_mp3_media())?;
        remux(
            &source,
            &target,
            VideoContainerFormat::Mov,
            VideoContainerFormat::Mp4,
        )?;
        assert!(target.exists());
        std::fs::remove_file(source)?;
        std::fs::remove_file(target)?;
        Ok(())
    }

    #[test]
    fn mkv_canonical_route_rejects_chapters_and_tags_with_mp3_and_vorbis()
    -> Result<(), Box<dyn std::error::Error>> {
        let media = h264_mp3_vorbis_media()?;
        for (name, chapter, tag) in [("chapter", true, false), ("tag", false, true)] {
            let source = temp_path(&format!("mkv-{name}-source"), "mkv");
            let target = temp_path(&format!("mkv-{name}-target"), "mkv");
            write_mkv_with_chapter_or_tag(&source, &media, chapter, tag)?;

            let reader = WindowedFileReader::open(&source)?;
            let demuxer = containr::mkv::Demuxer::open(reader)?;
            assert_eq!(demuxer.chapters().len(), usize::from(chapter));
            assert_eq!(demuxer.metadata().len(), usize::from(tag));

            // MP3 disables stream copy, so the MP3 + Vorbis source takes the
            // canonical route; chapters and tags must fail before demuxing.
            let result = remux(
                &source,
                &target,
                VideoContainerFormat::Mkv,
                VideoContainerFormat::Mkv,
            );
            assert!(
                matches!(result, Err(PipelineError::RemuxFailed { .. })),
                "{name} metadata was dropped instead of rejected: {result:?}"
            );
            assert!(
                !target.exists(),
                "no target was published after {name} rejection"
            );
            std::fs::remove_file(source)?;
        }
        Ok(())
    }

    #[test]
    fn mkv_to_mkv_direct_copy_preserves_attachments() -> Result<(), Box<dyn std::error::Error>> {
        let source = temp_path("mkv-attachment", "mkv");
        let target = temp_path("mkv-attachment-out", "mkv");
        write_mkv_with_attachment(&source, &h264_aac_media(), "note.txt", b"keep me")?;
        remux(
            &source,
            &target,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Mkv,
        )?;
        let reader = WindowedFileReader::open(&target)?;
        let demuxer = containr::mkv::Demuxer::open(reader)?;
        assert_eq!(demuxer.attachments().len(), 1);
        assert_eq!(demuxer.attachments()[0].name, "note.txt");
        std::fs::remove_file(source)?;
        std::fs::remove_file(target)?;
        Ok(())
    }

    fn write_mkv_with_subtitle(
        path: &Path,
        media: &ContainerMedia,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let base = temp_path("mkv-subtitle-base", "mkv");
        write_container(&base, VideoContainerFormat::Mkv, media)?;
        let reader = WindowedFileReader::open(&base)?;
        let mut input = containr::mkv::Demuxer::open(reader)?;
        let file = File::create(path)?;
        let mut writer = BufWriter::new(&file);
        let options = containr::mkv::MkvOptions {
            doc_type: DocType::Matroska,
            timestamp_scale: crate::video_container::MKV_TICK_NS,
            ..containr::mkv::MkvOptions::default()
        };
        let mut output = containr::mkv::Muxer::new(&mut writer, options)?;
        let mut plans = Vec::new();
        for track in input.tracks() {
            let packets = read_packets(input.track_packets(track.id)?, &base)?;
            let plan = containr::mkv::StreamCopy::plan(track, &packets)?;
            let id = output.add_track(plan.spec.clone())?;
            plans.push((track.id, plan, id));
        }
        let timebase = containr::Timebase::from_nanoseconds(u32::try_from(
            crate::video_container::MKV_TICK_NS,
        )?);
        let mut spec =
            containr::mkv::TrackSpec::new(containr::TrackKind::Subtitle, "S_TEXT/UTF8", timebase);
        spec.matroska.language = "eng".to_owned();
        let subtitle_id = output.add_track(spec)?;
        let mut buf = Vec::new();
        for packet in input.packets() {
            let packet = packet?;
            let Some((_, plan, id)) = plans
                .iter()
                .find(|(track_id, _, _)| *track_id == packet.track_id)
            else {
                continue;
            };
            input.read_packet_into(&packet, &mut buf)?;
            if let Some(block) = plan.packet(&packet, &buf)? {
                output.write_packet(*id, &block)?;
            }
        }
        output.write_packet(
            subtitle_id,
            &containr::mkv::PacketIn {
                data: b"Hello",
                pts: 0,
                duration: Some(1_000_000),
                sync: true,
                discardable: false,
                discard_padding: None,
            },
        )?;
        output.finish()?;
        writer.flush()?;
        drop(writer);
        std::fs::remove_file(base)?;
        Ok(())
    }

    #[test]
    fn mkv_canonical_route_rejects_subtitle_instead_of_dropping_it()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = temp_path("mkv-subtitle-source", "mkv");
        let target = temp_path("mkv-subtitle-target", "mkv");
        write_mkv_with_subtitle(&source, &h264_mp3_media())?;

        // Probe sees the subtitle track; the canonical route must reject it
        // before demux_video silently drops S_TEXT/UTF8.
        let mut stream_infos = streams(&demux_video(&source, VideoContainerFormat::Mkv)?);
        stream_infos.push(VideoStreamInfo::new(
            VideoCodec::TimedText,
            fastconv_domain::VideoStreamKind::Subtitle,
            None,
            None,
            None,
            None,
            1,
            US_TIMESCALE,
            None,
            Some("eng".to_owned()),
        ));
        let plan = VideoRemuxPlan::new(
            vec![VideoRemuxItem::new(
                source.clone(),
                target.clone(),
                VideoContainerFormat::Mkv,
                VideoContainerFormat::Mkv,
                stream_infos,
            )],
            NonZeroUsize::MIN,
        )
        .map_err(|err| remux_failed(&target, err.to_string()))?;
        let result = execute_video_batch(&plan);
        assert!(matches!(result, Err(PipelineError::RemuxFailed { .. })));
        assert!(!target.exists(), "no target published after rejection");
        std::fs::remove_file(source)?;
        Ok(())
    }

    #[test]
    fn mkv_direct_copy_preserves_subtitle() -> Result<(), Box<dyn std::error::Error>> {
        let source = temp_path("mkv-subtitle-direct", "mkv");
        let target = temp_path("mkv-subtitle-direct-out", "mkv");
        write_mkv_with_subtitle(&source, &h264_aac_media())?;
        remux(
            &source,
            &target,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Mkv,
        )?;
        let reader = WindowedFileReader::open(&target)?;
        let demuxer = containr::mkv::Demuxer::open(reader)?;
        assert!(
            demuxer
                .tracks()
                .iter()
                .any(|track| track.codec_id() == Some("S_TEXT/UTF8"))
        );
        std::fs::remove_file(source)?;
        std::fs::remove_file(target)?;
        Ok(())
    }
}
