use std::fmt;
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;

use crate::{DomainError, HardwareBackend, PathRole};

/// Audio container/codec formats accepted at the system boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum AudioInputFormat {
    Wav,
    Flac,
    Mp3,
    Vorbis,
    Opus,
}

impl AudioInputFormat {
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Mp3 => "mp3",
            Self::Vorbis => "ogg",
            Self::Opus => "opus",
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Wav => "WAV",
            Self::Flac => "FLAC",
            Self::Mp3 => "MP3",
            Self::Vorbis => "Ogg Vorbis",
            Self::Opus => "Ogg Opus",
        }
    }
}

impl fmt::Display for AudioInputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Native Rust audio encoders available to the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum AudioOutputFormat {
    Wav,
    Flac,
    Opus,
}

impl AudioOutputFormat {
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Opus => "opus",
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Wav => "WAV",
            Self::Flac => "FLAC",
            Self::Opus => "Opus",
        }
    }

    #[must_use]
    pub const fn is_lossless(self) -> bool {
        !matches!(self, Self::Opus)
    }

    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        if ext.eq_ignore_ascii_case("wav") || ext.eq_ignore_ascii_case("wave") {
            Some(Self::Wav)
        } else if ext.eq_ignore_ascii_case("flac") {
            Some(Self::Flac)
        } else if ext.eq_ignore_ascii_case("opus") {
            Some(Self::Opus)
        } else {
            None
        }
    }
}

impl fmt::Display for AudioOutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Validated decoded stream metadata produced by the boundary probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AudioStreamInfo {
    channels: NonZeroU16,
    sample_rate: NonZeroU32,
    bits_per_sample: Option<NonZeroU16>,
    total_samples: Option<u64>,
}

impl AudioStreamInfo {
    /// Creates validated stream metadata.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyField`] when `channels` or `sample_rate` is zero.
    pub fn new(
        channels: NonZeroU16,
        sample_rate: NonZeroU32,
        bits_per_sample: Option<NonZeroU16>,
        total_samples: Option<u64>,
    ) -> Result<Self, DomainError> {
        if channels.get() == 0 || sample_rate.get() == 0 {
            return Err(DomainError::EmptyField("audio.stream_info"));
        }
        Ok(Self {
            channels,
            sample_rate,
            bits_per_sample,
            total_samples,
        })
    }

    #[must_use]
    pub const fn channels(self) -> NonZeroU16 {
        self.channels
    }

    #[must_use]
    pub const fn sample_rate(self) -> NonZeroU32 {
        self.sample_rate
    }

    #[must_use]
    pub const fn bits_per_sample(self) -> Option<NonZeroU16> {
        self.bits_per_sample
    }

    #[must_use]
    pub const fn total_samples(self) -> Option<u64> {
        self.total_samples
    }
}

/// Concrete CPU encoder selected for an audio batch.
///
/// There is no GPU variant: audio conversion on a GPU backend is rejected before a plan is
/// created, so a GPU audio plan cannot be represented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum AudioExecutionMode {
    CpuWav,
    CpuFlac,
    CpuOpus,
}

impl AudioExecutionMode {
    /// Resolves the CPU encoder for `format`.
    ///
    /// Never falls back to CPU when a GPU backend was requested: the caller receives an explicit
    /// [`DomainError::UnsupportedAudioPipeline`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedAudioPipeline`] for any GPU backend.
    pub fn resolve(
        format: AudioOutputFormat,
        backend: HardwareBackend,
    ) -> Result<Self, DomainError> {
        match backend {
            HardwareBackend::Cpu => Ok(match format {
                AudioOutputFormat::Wav => Self::CpuWav,
                AudioOutputFormat::Flac => Self::CpuFlac,
                AudioOutputFormat::Opus => Self::CpuOpus,
            }),
            HardwareBackend::Gpu(backend) => Err(DomainError::UnsupportedAudioPipeline {
                format,
                backend: HardwareBackend::Gpu(backend),
                reason: format!(
                    "{backend} has no audio encoder and audio is never silently routed to CPU; use --backend cpu"
                ),
            }),
        }
    }

    #[must_use]
    pub const fn output_format(self) -> AudioOutputFormat {
        match self {
            Self::CpuWav => AudioOutputFormat::Wav,
            Self::CpuFlac => AudioOutputFormat::Flac,
            Self::CpuOpus => AudioOutputFormat::Opus,
        }
    }

    #[must_use]
    pub const fn backend(self) -> HardwareBackend {
        HardwareBackend::Cpu
    }
}

impl fmt::Display for AudioExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::CpuWav => "wav pcm (cpu)",
            Self::CpuFlac => "flac (cpu)",
            Self::CpuOpus => "opus (cpu)",
        })
    }
}

/// Single validated audio conversion item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AudioBatchItem {
    source: PathBuf,
    target: PathBuf,
    input_format: AudioInputFormat,
    stream: AudioStreamInfo,
}

impl AudioBatchItem {
    /// Creates a validated [`AudioBatchItem`].
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyPath`] for empty paths and
    /// [`DomainError::IdenticalSourceAndTarget`] if both paths are equal.
    pub fn new(
        source: PathBuf,
        target: PathBuf,
        input_format: AudioInputFormat,
        stream: AudioStreamInfo,
    ) -> Result<Self, DomainError> {
        if source.as_os_str().is_empty() {
            return Err(DomainError::EmptyPath(PathRole::Source));
        }
        if target.as_os_str().is_empty() {
            return Err(DomainError::EmptyPath(PathRole::Target));
        }
        if source == target {
            return Err(DomainError::IdenticalSourceAndTarget(source));
        }
        Ok(Self {
            source,
            target,
            input_format,
            stream,
        })
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
    pub const fn input_format(&self) -> AudioInputFormat {
        self.input_format
    }

    #[must_use]
    pub const fn stream(&self) -> AudioStreamInfo {
        self.stream
    }
}

/// Validated non-empty audio batch with conflict-free targets and one explicit CPU encoder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AudioBatchPlan {
    items: Vec<AudioBatchItem>,
    mode: AudioExecutionMode,
    worker_threads: NonZeroUsize,
}

impl AudioBatchPlan {
    /// Creates a validated [`AudioBatchPlan`].
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyBatch`] if `items` is empty,
    /// [`DomainError::DuplicateTargetPath`] for repeated targets, and
    /// [`DomainError::TargetOverwritesSource`] if a target equals any batch source.
    pub fn new(
        items: Vec<AudioBatchItem>,
        mode: AudioExecutionMode,
        worker_threads: NonZeroUsize,
    ) -> Result<Self, DomainError> {
        if items.is_empty() {
            return Err(DomainError::EmptyBatch);
        }
        let mut targets = std::collections::HashSet::with_capacity(items.len());
        for item in &items {
            if !targets.insert(item.target()) {
                return Err(DomainError::DuplicateTargetPath(
                    item.target().to_path_buf(),
                ));
            }
            if items.iter().any(|other| other.source() == item.target()) {
                return Err(DomainError::TargetOverwritesSource(
                    item.target().to_path_buf(),
                ));
            }
        }
        Ok(Self {
            items,
            mode,
            worker_threads,
        })
    }

    #[must_use]
    pub fn items(&self) -> &[AudioBatchItem] {
        &self.items
    }

    #[must_use]
    pub const fn mode(&self) -> AudioExecutionMode {
        self.mode
    }

    #[must_use]
    pub const fn output_format(&self) -> AudioOutputFormat {
        self.mode.output_format()
    }

    #[must_use]
    pub const fn worker_threads(&self) -> NonZeroUsize {
        self.worker_threads
    }

    #[must_use]
    pub fn effective_workers(&self) -> NonZeroUsize {
        let count = self.worker_threads.get().min(self.items.len()).max(1);
        NonZeroUsize::new(count).unwrap_or(NonZeroUsize::MIN)
    }
}

/// Execution summary returned after an [`AudioBatchPlan`] completes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AudioConversionReport {
    files_converted: NonZeroUsize,
    total_input_bytes: NonZeroU64,
    total_output_bytes: NonZeroU64,
    duration: Duration,
    worker_threads_used: NonZeroUsize,
    backend: HardwareBackend,
}

impl AudioConversionReport {
    /// Creates a report from measured batch totals.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyBatch`] when no files were converted or when either byte
    /// total is zero.
    pub fn new(
        files_converted: NonZeroUsize,
        total_input_bytes: NonZeroU64,
        total_output_bytes: NonZeroU64,
        duration: Duration,
        worker_threads_used: NonZeroUsize,
        backend: HardwareBackend,
    ) -> Result<Self, DomainError> {
        if total_input_bytes.get() == 0 || total_output_bytes.get() == 0 {
            return Err(DomainError::EmptyBatch);
        }
        Ok(Self {
            files_converted,
            total_input_bytes,
            total_output_bytes,
            duration,
            worker_threads_used,
            backend,
        })
    }

    #[must_use]
    pub const fn files_converted(&self) -> NonZeroUsize {
        self.files_converted
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
        if secs <= 0.0 {
            return 0.0;
        }
        (u64_to_f64(self.total_input_bytes.get()) / 1_000_000.0) / secs
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
    use crate::GpuBackend;

    fn item(name: &str) -> AudioBatchItem {
        AudioBatchItem::new(
            PathBuf::from(format!("{name}.wav")),
            PathBuf::from(format!("out/{name}.flac")),
            AudioInputFormat::Wav,
            AudioStreamInfo::new(
                NonZeroU16::new(2).unwrap_or(NonZeroU16::MIN),
                NonZeroU32::new(48_000).unwrap_or(NonZeroU32::MIN),
                NonZeroU16::new(16),
                Some(48_000),
            )
            .unwrap_or_else(|_| unreachable!()),
        )
        .unwrap_or_else(|_| unreachable!())
    }

    #[test]
    fn resolves_only_cpu_audio_modes() {
        assert_eq!(
            AudioExecutionMode::resolve(AudioOutputFormat::Flac, HardwareBackend::Cpu),
            Ok(AudioExecutionMode::CpuFlac)
        );
        for format in [
            AudioOutputFormat::Wav,
            AudioOutputFormat::Flac,
            AudioOutputFormat::Opus,
        ] {
            for backend in [GpuBackend::Nvenc, GpuBackend::Vaapi, GpuBackend::Vulkan] {
                assert!(matches!(
                    AudioExecutionMode::resolve(format, HardwareBackend::Gpu(backend)),
                    Err(DomainError::UnsupportedAudioPipeline { .. })
                ));
            }
        }
    }

    #[test]
    fn rejects_invalid_items_and_plans() {
        let path = PathBuf::from("audio.wav");
        assert_eq!(
            AudioBatchItem::new(
                path.clone(),
                path.clone(),
                AudioInputFormat::Wav,
                AudioStreamInfo::new(NonZeroU16::MIN, NonZeroU32::MIN, None, None,)
                    .unwrap_or_else(|_| unreachable!()),
            ),
            Err(DomainError::IdenticalSourceAndTarget(path))
        );

        let workers = NonZeroUsize::new(4).unwrap_or(NonZeroUsize::MIN);
        assert_eq!(
            AudioBatchPlan::new(Vec::new(), AudioExecutionMode::CpuFlac, workers),
            Err(DomainError::EmptyBatch)
        );

        let dup = AudioBatchPlan::new(
            vec![item("a"), item("a")],
            AudioExecutionMode::CpuFlac,
            workers,
        );
        assert!(matches!(dup, Err(DomainError::DuplicateTargetPath(_))));
    }

    #[test]
    fn caps_workers_and_reports_throughput() {
        let workers = NonZeroUsize::new(8).unwrap_or(NonZeroUsize::MIN);
        let plan = AudioBatchPlan::new(vec![item("a")], AudioExecutionMode::CpuOpus, workers)
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(plan.effective_workers().get(), 1);
        assert_eq!(plan.output_format(), AudioOutputFormat::Opus);

        let report = AudioConversionReport::new(
            NonZeroUsize::MIN,
            NonZeroU64::new(2_000_000).unwrap_or(NonZeroU64::MIN),
            NonZeroU64::new(1_000_000).unwrap_or(NonZeroU64::MIN),
            Duration::from_secs(2),
            NonZeroUsize::MIN,
            HardwareBackend::Cpu,
        )
        .unwrap_or_else(|_| unreachable!());
        assert!((report.megabytes_per_sec() - 1.0).abs() < 1e-9);
    }
}
