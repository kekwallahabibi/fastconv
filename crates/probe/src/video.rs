use std::fs::{self, File};
use std::io::{self, Read};
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

fn read_magic_prefix(path: &Path) -> Result<[u8; MAGIC_PREFIX_BYTES], ProbeError> {
    let mut file = File::open(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
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
) -> Result<Vec<VideoStreamInfo>, ProbeError> {
    let streams = match format {
        VideoContainerFormat::Mp4 => {
            let demuxer = containr::mp4::Demuxer::open_path(path)
                .map_err(|err| invalid_video(path, format!("container parse failed: {err}")))?;
            demuxer.tracks().iter().map(stream_info).collect::<Vec<_>>()
        }
        VideoContainerFormat::Mkv => {
            let demuxer = containr::mkv::Demuxer::open_path(path)
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

    let prefix = read_magic_prefix(path)?;
    let format = detect_video_magic(&prefix).ok_or_else(|| {
        invalid_video(
            path,
            "unrecognized container magic bytes (expected MP4/MOV or Matroska/MKV)",
        )
    })?;
    let streams = probe_containr(path, format)?;
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
    read_magic_prefix(path)
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
}
