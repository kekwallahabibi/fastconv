use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use containr::Track;
use containr::TrackKind;
use fastconv_domain::{
    DomainError, PathRole, VideoCodec, VideoContainerFormat, VideoRemuxItem, VideoRemuxPlan,
    VideoStreamInfo, VideoStreamKind,
};

use crate::{ProbeError, available_cpu_threads};

const MAGIC_PREFIX_BYTES: usize = 64;

/// Window size used by [`WindowedFileReader`].
const WINDOW_BYTES: usize = 1024 * 1024;

/// Top-level ISO base media box types that identify MP4/MOV files.
const MP4_TOP_LEVEL_BOXES: [&[u8; 4]; 10] = [
    b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"pnot", b"udta", b"cmov", b"rmra",
];

/// Validated video metadata produced at the file-system boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbedVideo {
    pub path: PathBuf,
    pub format: VideoContainerFormat,
    pub streams: Vec<VideoStreamInfo>,
}

fn invalid_video(path: &Path, reason: impl Into<String>) -> ProbeError {
    ProbeError::InvalidVideo {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

/// Detects the container from leading bytes only; file extensions are never trusted.
#[must_use]
pub fn detect_video_magic(bytes: &[u8]) -> Option<VideoContainerFormat> {
    // Matroska / WebM start with the EBML header magic.
    if bytes.len() >= 4 && bytes[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
        return Some(VideoContainerFormat::Mkv);
    }

    // ISO base media (MP4/MOV): a 32-bit box size followed by a top-level box type.
    // QuickTime MOV files can start with any of these top-level boxes.
    if bytes.len() >= 8 {
        let kind = &bytes[4..8];
        if MP4_TOP_LEVEL_BOXES.iter().any(|k| *k == kind) {
            return Some(VideoContainerFormat::Mp4);
        }
    }
    None
}

/// A `Read + Seek` adapter that keeps a 1 MiB window of the file in memory.
///
/// `containr` seeks to an absolute offset before every EBML/box header and
/// packet. `BufReader::seek(Start(..))` discards its buffer on every such call,
/// which turns a clip into thousands of tiny reads; this reader only refills
/// the window when the requested position falls outside it.
struct WindowedFileReader {
    file: File,
    file_len: u64,
    pos: u64,
    buf: Vec<u8>,
    buf_start: u64,
    buf_valid: usize,
}

impl WindowedFileReader {
    #[cfg(test)]
    fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        Ok(Self::from_file(file, file_len))
    }

    fn from_file(file: File, file_len: u64) -> Self {
        let window_len = u64::try_from(WINDOW_BYTES).unwrap_or(u64::MAX);
        let capacity = usize::try_from(file_len.min(window_len)).unwrap_or(WINDOW_BYTES);
        Self {
            file,
            file_len,
            pos: 0,
            buf: vec![0_u8; capacity],
            buf_start: 0,
            buf_valid: 0,
        }
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
                // `containr::*::Demuxer::open` probes the input length with
                // `seek(SeekFrom::End(0))`; resolve it from metadata with no I/O.
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

fn read_magic_prefix(file: &mut File, path: &Path) -> Result<[u8; MAGIC_PREFIX_BYTES], ProbeError> {
    let mut prefix = [0_u8; MAGIC_PREFIX_BYTES];
    let mut filled = 0_usize;
    while filled < prefix.len() {
        match file.read(&mut prefix[filled..]) {
            Ok(0) => break,
            Ok(count) => filled += count,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(source) => {
                return Err(ProbeError::Io {
                    path: path.to_path_buf(),
                    source,
                });
            }
        }
    }
    Ok(prefix)
}

fn stream_kind(track: &Track) -> VideoStreamKind {
    match track.kind {
        TrackKind::Video => VideoStreamKind::Video,
        TrackKind::Audio => VideoStreamKind::Audio,
        TrackKind::Subtitle => VideoStreamKind::Subtitle,
        TrackKind::Data => VideoStreamKind::Data,
        TrackKind::Other => VideoStreamKind::Other,
    }
}

fn stream_codec(track: &Track) -> VideoCodec {
    if let Some(codec_id) = track.codec_id() {
        // Matroska CodecID.
        if codec_id.contains("AVC") || codec_id == "V_MPEG4/ISO/AVC" {
            return VideoCodec::H264;
        }
        if codec_id.contains("HEVC") {
            return VideoCodec::Hevc;
        }
        if codec_id == "V_AV1" {
            return VideoCodec::Av1;
        }
        if codec_id == "V_VP9" || codec_id == "V_VP8" {
            return VideoCodec::Vp9;
        }
        if codec_id == "A_AAC" {
            return VideoCodec::Aac;
        }
        if codec_id == "A_OPUS" {
            return VideoCodec::Opus;
        }
        if codec_id == "A_VORBIS" {
            return VideoCodec::Vorbis;
        }
        if codec_id == "S_TEXT/UTF8" {
            return VideoCodec::TimedText;
        }
        return VideoCodec::Unknown;
    }

    // MP4/MOV sample entry fourcc.
    match track.codec().map(|fourcc| fourcc.0) {
        Some([b'a', b'v', b'c', b'1' | b'3']) => VideoCodec::H264,
        Some([b'h', b'v', b'c', b'1'] | [b'h', b'e', b'v', b'1']) => VideoCodec::Hevc,
        Some([b'a', b'v', b'0', b'1']) => VideoCodec::Av1,
        Some([b'v', b'p', b'0', b'9']) => VideoCodec::Vp9,
        Some([b'm', b'p', b'4', b'a']) => VideoCodec::Aac,
        Some([b'O', b'p', b'u', b's']) => VideoCodec::Opus,
        Some([b't', b'x', b'3', b'g']) => VideoCodec::TimedText,
        _ => VideoCodec::Unknown,
    }
}

fn dimensions(track: &Track) -> (Option<u32>, Option<u32>) {
    if let Some(video) = track.video() {
        return (Some(u32::from(video.width)), Some(u32::from(video.height)));
    }
    if let Some(mkv) = &track.matroska
        && let Some(video) = &mkv.video
    {
        return (
            u32::try_from(video.pixel_width).ok(),
            u32::try_from(video.pixel_height).ok(),
        );
    }
    (None, None)
}

// Matroska stores sampling frequency as f64; the range is validated before conversion.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn audio_facts(track: &Track) -> (Option<u16>, Option<u32>) {
    if let Some(audio) = track.audio() {
        return (u16::try_from(audio.channels).ok(), Some(audio.sample_rate));
    }
    if let Some(mkv) = &track.matroska
        && let Some(audio) = &mkv.audio
    {
        // Matroska records sample rate as f64; canonical rates are integral.
        let rate = audio.sampling_frequency;
        let rate = if rate.is_finite() && rate > 0.0 && rate <= f64::from(u32::MAX) {
            Some(rate as u32)
        } else {
            None
        };
        return (u16::try_from(audio.channels).ok(), rate);
    }
    (None, None)
}

fn stream_info(track: &Track) -> VideoStreamInfo {
    let (width, height) = dimensions(track);
    let (channels, sample_rate) = audio_facts(track);
    let language = track.language.iso639_mapped().map(str::to_owned);
    VideoStreamInfo::new(
        stream_codec(track),
        stream_kind(track),
        width,
        height,
        channels,
        sample_rate,
        track.timebase.num,
        track.timebase.den,
        track.duration,
        language,
    )
}

fn probe_containr(
    path: &Path,
    format: VideoContainerFormat,
    file: File,
    file_len: u64,
) -> Result<Vec<VideoStreamInfo>, ProbeError> {
    let reader = WindowedFileReader::from_file(file, file_len);
    let streams = match format {
        VideoContainerFormat::Mp4 => {
            let demuxer = containr::mp4::Demuxer::open(reader)
                .map_err(|err| invalid_video(path, format!("container parse failed: {err}")))?;
            demuxer.tracks().iter().map(stream_info).collect::<Vec<_>>()
        }
        VideoContainerFormat::Mkv => {
            let demuxer = containr::mkv::Demuxer::open(reader)
                .map_err(|err| invalid_video(path, format!("container parse failed: {err}")))?;
            if demuxer.truncated() {
                return Err(invalid_video(path, "truncated MKV input"));
            }
            demuxer.tracks().iter().map(stream_info).collect::<Vec<_>>()
        }
    };
    Ok(streams)
}

/// Probes a video file with magic-byte detection and extracts stream metadata.
///
/// # Errors
///
/// Returns [`ProbeError`] for I/O failures, unrecognised magic bytes, or malformed containers.
pub fn probe_video(path: &Path) -> Result<ProbedVideo, ProbeError> {
    let metadata = fs::metadata(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(ProbeError::NotAFile(path.to_path_buf()));
    }
    if metadata.len() == 0 {
        return Err(invalid_video(path, "video file is empty"));
    }

    let mut file = File::open(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let prefix = read_magic_prefix(&mut file, path)?;
    file.seek(SeekFrom::Start(0))
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let format = detect_video_magic(&prefix).ok_or_else(|| {
        invalid_video(
            path,
            "unrecognized container magic bytes (expected MP4/MOV or Matroska/MKV)",
        )
    })?;
    let streams = probe_containr(path, format, file, metadata.len())?;
    if streams.is_empty() {
        return Err(invalid_video(path, "container has no tracks"));
    }

    Ok(ProbedVideo {
        path: path.to_path_buf(),
        format,
        streams,
    })
}

/// Rejects targets that name the same file as the source through a different path spelling.
fn reject_alias_collision(source: &Path, target: &Path) -> Result<(), ProbeError> {
    let canonical_source = fs::canonicalize(source).map_err(|source_err| ProbeError::Io {
        path: source.to_path_buf(),
        source: source_err,
    })?;
    let canonical_target = if target.exists() {
        fs::canonicalize(target).map_err(|err| ProbeError::Io {
            path: target.to_path_buf(),
            source: err,
        })?
    } else {
        let parent = target
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let canonical_parent = fs::canonicalize(parent).map_err(|err| ProbeError::Io {
            path: parent.to_path_buf(),
            source: err,
        })?;
        match target.file_name() {
            Some(name) => canonical_parent.join(name),
            None => canonical_parent,
        }
    };
    if canonical_source == canonical_target {
        return Err(DomainError::IdenticalSourceAndTarget(source.to_path_buf()).into());
    }
    Ok(())
}

fn resolve_single_target(
    source: &Path,
    target: &Path,
    to: Option<VideoContainerFormat>,
) -> Result<(PathBuf, VideoContainerFormat), ProbeError> {
    let target_is_dir = target.is_dir()
        || target
            .to_str()
            .is_some_and(|s| s.ends_with('/') || s.ends_with('\\'));
    if target_is_dir {
        let format = to.ok_or_else(ProbeError::requires_to_for_video_directory)?;
        let stem = source
            .file_stem()
            .ok_or_else(|| invalid_video(source, "source path has no file stem"))?;
        let mut name = stem.to_os_string();
        name.push(".");
        name.push(format.extension());
        return Ok((target.join(name), format));
    }

    let ext = target.extension().and_then(|e| e.to_str()).ok_or_else(|| {
        ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: "mp4 or mkv",
        }
    })?;
    let inferred = VideoContainerFormat::from_extension(ext).ok_or_else(|| {
        ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: "mp4 or mkv",
        }
    })?;
    if let Some(explicit) = to
        && explicit != inferred
    {
        return Err(ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: explicit.extension(),
        });
    }
    Ok((target.to_path_buf(), inferred))
}

fn prepare_directory_batch(
    source_dir: &Path,
    target_dir: &Path,
    format: VideoContainerFormat,
    threads: NonZeroUsize,
) -> Result<VideoRemuxPlan, ProbeError> {
    if !target_dir.is_dir() {
        let exists = fs::metadata(target_dir).is_ok();
        if exists {
            return Err(ProbeError::TargetNotADirectory(target_dir.to_path_buf()));
        }
        fs::create_dir_all(target_dir).map_err(|source| ProbeError::Io {
            path: target_dir.to_path_buf(),
            source,
        })?;
    }

    let mut sources: Vec<PathBuf> = fs::read_dir(source_dir)
        .map_err(|source| ProbeError::Io {
            path: source_dir.to_path_buf(),
            source,
        })?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter(|path| {
            detect_video_magic_file(path).is_some_and(|detected| {
                detected == VideoContainerFormat::Mp4 || detected == VideoContainerFormat::Mkv
            })
        })
        .collect();
    sources.sort();

    if sources.is_empty() {
        return Err(ProbeError::EmptyVideoDirectory(source_dir.to_path_buf()));
    }

    let mut items = Vec::with_capacity(sources.len());
    for source in sources {
        let probed = probe_video(&source)?;
        let stem = source
            .file_stem()
            .ok_or_else(|| invalid_video(&source, "source path has no file stem"))?;
        let mut name = stem.to_os_string();
        name.push(".");
        name.push(format.extension());
        let target = target_dir.join(name);
        // Reject an unsupported codec at the boundary before building the plan.
        for stream in &probed.streams {
            if !stream.codec().supported_in(format) {
                return Err(DomainError::UnsupportedVideoCodec {
                    codec: stream.codec(),
                    container: format,
                }
                .into());
            }
        }
        reject_alias_collision(&source, &target)?;
        items.push(VideoRemuxItem::new(
            source,
            target,
            probed.format,
            format,
            probed.streams,
        ));
    }

    Ok(VideoRemuxPlan::new(items, threads)?)
}

fn detect_video_magic_file(path: &Path) -> Option<VideoContainerFormat> {
    let mut file = File::open(path).ok()?;
    read_magic_prefix(&mut file, path)
        .ok()
        .and_then(|prefix| detect_video_magic(&prefix))
}

/// Builds a validated [`VideoRemuxPlan`] from CLI inputs.
///
/// # Errors
///
/// Returns [`ProbeError`] for invalid paths, unsupported target formats, or unsupported codecs.
pub fn prepare_video_batch(
    source: &Path,
    target: &Path,
    to: Option<VideoContainerFormat>,
    worker_threads: Option<NonZeroUsize>,
) -> Result<VideoRemuxPlan, ProbeError> {
    if source.as_os_str().is_empty() {
        return Err(DomainError::EmptyPath(PathRole::Source).into());
    }
    if target.as_os_str().is_empty() {
        return Err(DomainError::EmptyPath(PathRole::Target).into());
    }

    let threads = match worker_threads {
        Some(count) => count,
        None => available_cpu_threads()?,
    };
    let source_meta = fs::metadata(source).map_err(|err| ProbeError::Io {
        path: source.to_path_buf(),
        source: err,
    })?;

    if source_meta.is_dir() {
        let format = to.ok_or_else(ProbeError::requires_to_for_video_directory)?;
        return prepare_directory_batch(source, target, format, threads);
    }
    if !source_meta.is_file() {
        return Err(ProbeError::NotAFileOrDirectory(source.to_path_buf()));
    }

    let (resolved_target, format) = resolve_single_target(source, target, to)?;
    reject_alias_collision(source, &resolved_target)?;
    let probed = probe_video(source)?;
    for stream in &probed.streams {
        if !stream.codec().supported_in(format) {
            return Err(DomainError::UnsupportedVideoCodec {
                codec: stream.codec(),
                container: format,
            }
            .into());
        }
    }
    let item = VideoRemuxItem::new(
        source.to_path_buf(),
        resolved_target,
        probed.format,
        format,
        probed.streams,
    );
    Ok(VideoRemuxPlan::new(vec![item], threads)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_file(label: &str, bytes: &[u8]) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = std::env::temp_dir().join(format!("fastconv-probe-video-{label}-{nanos}"));
        fs::write(&path, bytes)?;
        Ok(path)
    }

    fn ebml_el(id: u32, body: &[u8]) -> Vec<u8> {
        let raw = id.to_be_bytes();
        let skip = raw.iter().take_while(|&&b| b == 0).count();
        let mut out = raw[skip..].to_vec();
        let size = u8::try_from(body.len()).unwrap_or(u8::MAX);
        out.push(0x80_u8 | size);
        out.extend_from_slice(body);
        out
    }

    fn ebml_uint(id: u32, value: u64) -> Vec<u8> {
        let bytes = value.to_be_bytes();
        let skip = bytes.iter().take_while(|&&b| b == 0).count().min(7);
        ebml_el(id, &bytes[skip..])
    }

    /// An EBML/MKV file whose declared Segment size extends past the real data.
    fn truncated_mkv() -> Vec<u8> {
        let mut body = Vec::new();
        body.extend(ebml_uint(0x4286, 1));
        body.extend(ebml_uint(0x42F2, 4));
        body.extend(ebml_uint(0x42F3, 8));
        body.extend(ebml_el(0x4282, b"matroska"));
        body.extend(ebml_uint(0x4287, 4));
        body.extend(ebml_uint(0x4285, 2));
        let mut bytes = ebml_el(0x1A45_DFA3, &body);
        // Segment id followed by a one-byte size declaring 100 bytes of data.
        bytes.extend_from_slice(&[0x18, 0x53, 0x80, 0x67, 0xE4]);
        bytes.extend(ebml_el(0x1549_A966, &ebml_uint(0x002A_D7B1, 1_000_000)));
        let mut entry = Vec::new();
        entry.extend(ebml_uint(0xD7, 1));
        entry.extend(ebml_uint(0x73C5, 1));
        entry.extend(ebml_uint(0x83, 1));
        entry.extend(ebml_el(0x86, b"V_MPEG4/ISO/AVC"));
        bytes.extend(ebml_el(0x1654_AE6B, &ebml_el(0xAE, &entry)));
        bytes
    }

    #[test]
    fn detects_magic_without_extension_help() {
        let mut mkv = vec![0_u8; 8];
        mkv[..4].copy_from_slice(&[0x1A, 0x45, 0xDF, 0xA3]);
        assert_eq!(detect_video_magic(&mkv), Some(VideoContainerFormat::Mkv));

        let mut mp4 = vec![0_u8; 8];
        mp4[4..8].copy_from_slice(b"ftyp");
        assert_eq!(detect_video_magic(&mp4), Some(VideoContainerFormat::Mp4));

        let mut mov = vec![0_u8; 8];
        mov[4..8].copy_from_slice(b"moov");
        assert_eq!(detect_video_magic(&mov), Some(VideoContainerFormat::Mp4));

        assert_eq!(detect_video_magic(b"unknown"), None);
    }

    #[test]
    fn windowed_reader_end_seek_and_current_seek() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_file("end-seek", b"0123456789")?;
        let mut reader = WindowedFileReader::open(&path)?;
        assert_eq!(reader.seek(SeekFrom::End(0))?, 10);
        assert_eq!(reader.seek(SeekFrom::End(-4))?, 6);
        let mut out = [0_u8; 4];
        assert_eq!(reader.read(&mut out)?, 4);
        assert_eq!(&out, b"6789");
        assert_eq!(reader.seek(SeekFrom::Current(-2))?, 8);
        assert_eq!(reader.read(&mut out)?, 2);
        assert_eq!(&out[..2], b"89");
        Ok(())
    }

    #[test]
    fn windowed_reader_rejects_seek_before_start() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_file("negative-seek", b"0123456789")?;
        let mut reader = WindowedFileReader::open(&path)?;
        for pos in [SeekFrom::Current(-1), SeekFrom::End(-11)] {
            match reader.seek(pos) {
                Ok(result) => return Err(format!("unexpected seek result: {result}").into()),
                Err(err) => assert_eq!(err.kind(), io::ErrorKind::InvalidInput),
            }
        }
        Ok(())
    }

    #[test]
    fn windowed_reader_past_eof_read_returns_zero() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_file("past-eof", b"0123456789")?;
        let mut reader = WindowedFileReader::open(&path)?;
        reader.seek(SeekFrom::Start(100))?;
        assert_eq!(reader.read(&mut [0_u8; 4])?, 0);
        Ok(())
    }

    #[test]
    fn windowed_reader_backward_seek_hits_buffer() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_file("backward", b"0123456789")?;
        let mut reader = WindowedFileReader::open(&path)?;
        let mut out = [0_u8; 4];
        assert_eq!(reader.read(&mut out)?, 4);
        assert_eq!(&out, b"0123");
        assert!(reader.buf_valid >= 4);
        assert_eq!(reader.buf_start, 0);
        reader.seek(SeekFrom::Start(2))?;
        assert_eq!(reader.read(&mut out)?, 4);
        assert_eq!(&out, b"2345");
        Ok(())
    }

    #[test]
    fn windowed_reader_crosses_window_boundary() -> Result<(), Box<dyn std::error::Error>> {
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
        Ok(())
    }

    #[test]
    fn windowed_reader_large_read_exceeds_window() -> Result<(), Box<dyn std::error::Error>> {
        let len = WINDOW_BYTES + 4096;
        let data: Vec<u8> = (0..len)
            .map(|i| u8::try_from(i % 251).unwrap_or(0))
            .collect();
        let path = temp_file("large-read", &data)?;
        let mut reader = WindowedFileReader::open(&path)?;
        let mut out = vec![0_u8; len];
        let mut total = 0_usize;
        loop {
            let count = reader.read(&mut out[total..])?;
            if count == 0 {
                break;
            }
            total += count;
        }
        assert_eq!(total, len);
        assert_eq!(out, data);
        Ok(())
    }

    #[test]
    fn truncated_mkv_is_still_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let bytes = truncated_mkv();
        let path = temp_file("truncated-mkv", &bytes)?;
        let reason = match probe_video(&path) {
            Ok(_) => return Err("a truncated MKV must be rejected".into()),
            Err(ProbeError::InvalidVideo { reason, .. }) => reason,
            Err(err) => return Err(format!("unexpected probe error: {err}").into()),
        };
        assert!(reason.contains("truncated MKV input"), "reason: {reason}");
        Ok(())
    }
}
