use std::fmt;
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;

use crate::{DomainError, HardwareBackend};

/// Video container formats supported for lossless remuxing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum VideoContainerFormat {
    Mp4,
    Mkv,
}

impl VideoContainerFormat {
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Mkv => "mkv",
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Mp4 => "MP4",
            Self::Mkv => "MKV",
        }
    }

    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "mp4" | "m4v" | "mov" => Some(Self::Mp4),
            "mkv" | "webm" => Some(Self::Mkv),
            _ => None,
        }
    }
}

impl fmt::Display for VideoContainerFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Video/audio codecs that fastconv recognises in containers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum VideoCodec {
    H264,
    Hevc,
    Av1,
    Vp9,
    Aac,
    Opus,
    Vorbis,
    Unknown,
}

impl VideoCodec {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::Hevc => "HEVC",
            Self::Av1 => "AV1",
            Self::Vp9 => "VP9",
            Self::Aac => "AAC",
            Self::Opus => "Opus",
            Self::Vorbis => "Vorbis",
            Self::Unknown => "unknown",
        }
    }

    /// Whether this codec can be stream-copied into `container`.
    #[must_use]
    pub const fn supported_in(self, container: VideoContainerFormat) -> bool {
        match container {
            VideoContainerFormat::Mp4 => matches!(
                self,
                Self::H264 | Self::Hevc | Self::Av1 | Self::Aac | Self::Opus
            ),
            VideoContainerFormat::Mkv => matches!(
                self,
                Self::H264
                    | Self::Hevc
                    | Self::Av1
                    | Self::Vp9
                    | Self::Aac
                    | Self::Opus
                    | Self::Vorbis
            ),
        }
    }
}

impl fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Kind of a stream inside a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum VideoStreamKind {
    Video,
    Audio,
    Subtitle,
    Data,
    Other,
}

impl fmt::Display for VideoStreamKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Video => "video",
            Self::Audio => "audio",
            Self::Subtitle => "subtitle",
            Self::Data => "data",
            Self::Other => "other",
        };
        f.write_str(label)
    }
}

/// Probed stream metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VideoStreamInfo {
    codec: VideoCodec,
    kind: VideoStreamKind,
    width: Option<NonZeroU32>,
    height: Option<NonZeroU32>,
    channels: Option<NonZeroU16>,
    sample_rate: Option<NonZeroU32>,
    timebase_num: u32,
    timebase_den: u32,
    duration: Option<u64>,
    language: Option<String>,
}

impl VideoStreamInfo {
    // One argument per container fact keeps the boundary mapping explicit.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        codec: VideoCodec,
        kind: VideoStreamKind,
        width: Option<u32>,
        height: Option<u32>,
        channels: Option<u16>,
        sample_rate: Option<u32>,
        timebase_num: u32,
        timebase_den: u32,
        duration: Option<u64>,
        language: Option<String>,
    ) -> Self {
        Self {
            codec,
            kind,
            width: width.and_then(NonZeroU32::new),
            height: height.and_then(NonZeroU32::new),
            channels: channels.and_then(NonZeroU16::new),
            sample_rate: sample_rate.and_then(NonZeroU32::new),
            timebase_num,
            timebase_den,
            duration,
            language,
        }
    }

    #[must_use]
    pub const fn codec(&self) -> VideoCodec {
        self.codec
    }

    #[must_use]
    pub const fn kind(&self) -> VideoStreamKind {
        self.kind
    }

    #[must_use]
    pub const fn width(&self) -> Option<NonZeroU32> {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> Option<NonZeroU32> {
        self.height
    }

    #[must_use]
    pub const fn channels(&self) -> Option<NonZeroU16> {
        self.channels
    }

    #[must_use]
    pub const fn sample_rate(&self) -> Option<NonZeroU32> {
        self.sample_rate
    }

    #[must_use]
    pub const fn timebase(&self) -> (u32, u32) {
        (self.timebase_num, self.timebase_den)
    }

    #[must_use]
    pub const fn duration(&self) -> Option<u64> {
        self.duration
    }

    #[must_use]
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }
}

impl fmt::Display for VideoStreamInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.codec)?;
        if let (Some(width), Some(height)) = (self.width, self.height) {
            write!(f, " {width}x{height}")?;
        }
        if let Some(sample_rate) = self.sample_rate {
            write!(f, " {sample_rate}Hz")?;
        }
        if let Some(channels) = self.channels {
            write!(f, " {channels}ch")?;
        }
        Ok(())
    }
}

/// A single remux item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoRemuxItem {
    source: PathBuf,
    target: PathBuf,
    input_format: VideoContainerFormat,
    output_format: VideoContainerFormat,
    streams: Vec<VideoStreamInfo>,
}

impl VideoRemuxItem {
    #[must_use]
    pub fn new(
        source: PathBuf,
        target: PathBuf,
        input_format: VideoContainerFormat,
        output_format: VideoContainerFormat,
        streams: Vec<VideoStreamInfo>,
    ) -> Self {
        Self {
            source,
            target,
            input_format,
            output_format,
            streams,
        }
    }

    #[must_use]
    pub fn source(&self) -> &Path {
        &self.source
    }

    #[must_use]
    pub fn target(&self) -> &Path {
        &self.target
    }

    #[must_use]
    pub const fn input_format(&self) -> VideoContainerFormat {
        self.input_format
    }

    #[must_use]
    pub const fn output_format(&self) -> VideoContainerFormat {
        self.output_format
    }

    #[must_use]
    pub fn streams(&self) -> &[VideoStreamInfo] {
        &self.streams
    }
}

/// Validated batch plan for remuxing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoRemuxPlan {
    items: Vec<VideoRemuxItem>,
    worker_threads: NonZeroUsize,
}

impl VideoRemuxPlan {
    /// Creates a validated plan.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyBatch`] if `items` is empty, or
    /// [`DomainError::DuplicateTargetPath`] / [`DomainError::TargetOverwritesSource`].
    pub fn new(
        items: Vec<VideoRemuxItem>,
        worker_threads: NonZeroUsize,
    ) -> Result<Self, DomainError> {
        if items.is_empty() {
            return Err(DomainError::EmptyBatch);
        }
        let mut targets = std::collections::HashSet::new();
        let sources: std::collections::HashSet<PathBuf> =
            items.iter().map(|i| i.source().to_path_buf()).collect();
        for item in &items {
            if !targets.insert(item.target().to_path_buf()) {
                return Err(DomainError::DuplicateTargetPath(
                    item.target().to_path_buf(),
                ));
            }
            if sources.contains(item.target()) {
                return Err(DomainError::TargetOverwritesSource(
                    item.target().to_path_buf(),
                ));
            }
        }
        Ok(Self {
            items,
            worker_threads,
        })
    }

    #[must_use]
    pub fn items(&self) -> &[VideoRemuxItem] {
        &self.items
    }

    #[must_use]
    pub const fn worker_threads(&self) -> NonZeroUsize {
        self.worker_threads
    }

    #[must_use]
    pub fn effective_workers(&self) -> NonZeroUsize {
        let n = self.worker_threads.get().min(self.items.len()).max(1);
        NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN)
    }

    #[must_use]
    pub fn output_format(&self) -> VideoContainerFormat {
        self.items[0].output_format()
    }
}

/// Report after a remux batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VideoRemuxReport {
    files_remuxed: NonZeroUsize,
    total_input_bytes: NonZeroU64,
    total_output_bytes: NonZeroU64,
    duration: Duration,
    worker_threads_used: NonZeroUsize,
    backend: HardwareBackend,
}

impl VideoRemuxReport {
    #[must_use]
    pub fn new(
        files_remuxed: NonZeroUsize,
        total_input_bytes: NonZeroU64,
        total_output_bytes: NonZeroU64,
        duration: Duration,
        worker_threads_used: NonZeroUsize,
        backend: HardwareBackend,
    ) -> Self {
        Self {
            files_remuxed,
            total_input_bytes,
            total_output_bytes,
            duration,
            worker_threads_used,
            backend,
        }
    }

    #[must_use]
    pub const fn files_remuxed(&self) -> NonZeroUsize {
        self.files_remuxed
    }

    #[must_use]
    pub const fn total_input_bytes(&self) -> NonZeroU64 {
        self.total_input_bytes
    }

    #[must_use]
    pub const fn total_output_bytes(&self) -> NonZeroU64 {
        self.total_output_bytes
    }

    #[must_use]
    pub const fn duration(&self) -> Duration {
        self.duration
    }

    #[must_use]
    pub const fn worker_threads_used(&self) -> NonZeroUsize {
        self.worker_threads_used
    }

    #[must_use]
    pub const fn backend(&self) -> HardwareBackend {
        self.backend
    }

    #[must_use]
    pub fn megabytes_per_sec(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        (u64_to_f64(self.total_input_bytes.get()) / (1024.0 * 1024.0)) / secs
    }
}

fn u64_to_f64(value: u64) -> f64 {
    let hi = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let lo = u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(hi) * 4_294_967_296.0 + f64::from(lo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    #[test]
    fn codec_support_matrix() {
        assert!(VideoCodec::H264.supported_in(VideoContainerFormat::Mp4));
        assert!(VideoCodec::H264.supported_in(VideoContainerFormat::Mkv));
        assert!(VideoCodec::Aac.supported_in(VideoContainerFormat::Mp4));
        assert!(VideoCodec::Aac.supported_in(VideoContainerFormat::Mkv));
        assert!(!VideoCodec::Vorbis.supported_in(VideoContainerFormat::Mp4));
    }

    #[test]
    fn plan_rejects_empty_and_duplicates() {
        let w = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        assert!(matches!(
            VideoRemuxPlan::new(Vec::new(), w),
            Err(DomainError::EmptyBatch)
        ));
    }
}
