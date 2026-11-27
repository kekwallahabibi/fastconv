use std::fmt;
use std::num::NonZeroU64;
use std::time::Duration;

#[cfg(feature = "serde")]
use serde::Serialize;

use crate::{DomainError, MediaKind, require_non_empty, u64_to_f64};

const BYTES_PER_MEGABYTE: f64 = 1_000_000.0;

/// Peak resident memory observed during a benchmark run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum PeakMemory {
    MeasuredBytes(NonZeroU64),
    Unavailable,
}

/// Primary work unit used to compute benchmark throughput.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum WorkloadUnit {
    Bytes,
    VideoFrames(NonZeroU64),
}

/// Derived throughput metric guaranteed to match measured duration and size.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum Throughput {
    MegabytesPerSec(f64),
    FramesPerSec { fps: f64, megabytes_per_sec: f64 },
}

impl fmt::Display for Throughput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MegabytesPerSec(mbps) => write!(f, "{mbps:.2} MB/s"),
            Self::FramesPerSec {
                fps,
                megabytes_per_sec,
            } => write!(f, "{fps:.1} fps ({megabytes_per_sec:.2} MB/s)"),
        }
    }
}

/// Raw measured sample passed into [`BenchmarkMetrics::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkSample {
    pub duration: Duration,
    pub input_bytes: NonZeroU64,
    pub output_bytes: NonZeroU64,
    pub unit: WorkloadUnit,
    pub peak_memory: PeakMemory,
}

/// Validated, self-consistent performance record for a benchmark workload.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct BenchmarkMetrics {
    workload_name: String,
    media_kind: MediaKind,
    tool_label: String,
    duration_ms: f64,
    input_bytes: NonZeroU64,
    output_bytes: NonZeroU64,
    throughput: Throughput,
    peak_memory: PeakMemory,
}

impl BenchmarkMetrics {
    /// Creates validated [`BenchmarkMetrics`] and derives consistent throughput metrics.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyField`] if `workload_name` or `tool_label` is blank,
    /// or [`DomainError::ZeroDuration`] if `sample.duration` is zero.
    pub fn new(
        workload_name: &str,
        media_kind: MediaKind,
        tool_label: &str,
        sample: BenchmarkSample,
    ) -> Result<Self, DomainError> {
        let workload_name = require_non_empty(workload_name, "benchmark.workload_name")?;
        let tool_label = require_non_empty(tool_label, "benchmark.tool_label")?;
        if sample.duration.is_zero() {
            return Err(DomainError::ZeroDuration);
        }

        let secs = sample.duration.as_secs_f64();
        let duration_ms = secs * 1_000.0;
        let megabytes_per_sec = (u64_to_f64(sample.input_bytes.get()) / BYTES_PER_MEGABYTE) / secs;

        let throughput = match sample.unit {
            WorkloadUnit::Bytes => Throughput::MegabytesPerSec(megabytes_per_sec),
            WorkloadUnit::VideoFrames(frames) => Throughput::FramesPerSec {
                fps: u64_to_f64(frames.get()) / secs,
                megabytes_per_sec,
            },
        };

        Ok(Self {
            workload_name,
            media_kind,
            tool_label,
            duration_ms,
            input_bytes: sample.input_bytes,
            output_bytes: sample.output_bytes,
            throughput,
            peak_memory: sample.peak_memory,
        })
    }

    #[must_use]
    pub fn workload_name(&self) -> &str {
        &self.workload_name
    }

    #[must_use]
    pub const fn media_kind(&self) -> MediaKind {
        self.media_kind
    }

    #[must_use]
    pub fn tool_label(&self) -> &str {
        &self.tool_label
    }

    #[must_use]
    pub const fn duration_ms(&self) -> f64 {
        self.duration_ms
    }

    #[must_use]
    pub const fn input_bytes(&self) -> NonZeroU64 {
        self.input_bytes
    }

    #[must_use]
    pub const fn output_bytes(&self) -> NonZeroU64 {
        self.output_bytes
    }

    #[must_use]
    pub const fn throughput(&self) -> Throughput {
        self.throughput
    }

    #[must_use]
    pub const fn peak_memory(&self) -> PeakMemory {
        self.peak_memory
    }

    #[must_use]
    pub fn size_ratio(&self) -> f64 {
        u64_to_f64(self.output_bytes.get()) / u64_to_f64(self.input_bytes.get())
    }
}

#[cfg(test)]
mod tests {
    use super::{BenchmarkMetrics, BenchmarkSample, PeakMemory, Throughput, WorkloadUnit};
    use crate::{DomainError, MediaKind};
    use std::num::NonZeroU64;
    use std::time::Duration;

    #[test]
    fn computes_byte_throughput_and_ratio() -> Result<(), DomainError> {
        let input = NonZeroU64::new(20_000_000).ok_or(DomainError::EmptyField("test"))?;
        let output = NonZeroU64::new(5_000_000).ok_or(DomainError::EmptyField("test"))?;
        let mem = NonZeroU64::new(64 * 1024 * 1024).ok_or(DomainError::EmptyField("test"))?;

        let metrics = BenchmarkMetrics::new(
            "audio_60s_wav_to_flac",
            MediaKind::Audio,
            "oracle (flac)",
            BenchmarkSample {
                duration: Duration::from_secs(2),
                input_bytes: input,
                output_bytes: output,
                unit: WorkloadUnit::Bytes,
                peak_memory: PeakMemory::MeasuredBytes(mem),
            },
        )?;

        assert!((metrics.duration_ms() - 2_000.0).abs() < 1e-6);
        assert!((metrics.size_ratio() - 0.25).abs() < 1e-9);
        match metrics.throughput() {
            Throughput::MegabytesPerSec(mbps) => assert!((mbps - 10.0).abs() < 1e-6),
            Throughput::FramesPerSec { .. } => {
                return Err(DomainError::EmptyField("unexpected throughput variant"));
            }
        }
        Ok(())
    }

    #[test]
    fn computes_video_fps_and_megabytes_per_sec() -> Result<(), DomainError> {
        let input = NonZeroU64::new(30_000_000).ok_or(DomainError::EmptyField("test"))?;
        let output = NonZeroU64::new(15_000_000).ok_or(DomainError::EmptyField("test"))?;
        let frames = NonZeroU64::new(600).ok_or(DomainError::EmptyField("test"))?;

        let metrics = BenchmarkMetrics::new(
            "video_1080p60",
            MediaKind::Video,
            "oracle (libx264)",
            BenchmarkSample {
                duration: Duration::from_secs(4),
                input_bytes: input,
                output_bytes: output,
                unit: WorkloadUnit::VideoFrames(frames),
                peak_memory: PeakMemory::Unavailable,
            },
        )?;

        match metrics.throughput() {
            Throughput::FramesPerSec {
                fps,
                megabytes_per_sec,
            } => {
                assert!((fps - 150.0).abs() < 1e-6);
                assert!((megabytes_per_sec - 7.5).abs() < 1e-6);
            }
            Throughput::MegabytesPerSec(_) => {
                return Err(DomainError::EmptyField("unexpected throughput variant"));
            }
        }
        Ok(())
    }

    #[test]
    fn rejects_zero_duration_and_blank_labels() -> Result<(), DomainError> {
        let bytes = NonZeroU64::new(1024).ok_or(DomainError::EmptyField("test"))?;

        let zero = BenchmarkMetrics::new(
            "workload",
            MediaKind::Image,
            "oracle",
            BenchmarkSample {
                duration: Duration::ZERO,
                input_bytes: bytes,
                output_bytes: bytes,
                unit: WorkloadUnit::Bytes,
                peak_memory: PeakMemory::Unavailable,
            },
        );
        assert_eq!(zero, Err(DomainError::ZeroDuration));

        let blank_workload = BenchmarkMetrics::new(
            "   ",
            MediaKind::Image,
            "oracle",
            BenchmarkSample {
                duration: Duration::from_millis(10),
                input_bytes: bytes,
                output_bytes: bytes,
                unit: WorkloadUnit::Bytes,
                peak_memory: PeakMemory::Unavailable,
            },
        );
        assert_eq!(
            blank_workload,
            Err(DomainError::EmptyField("benchmark.workload_name"))
        );

        let blank_tool = BenchmarkMetrics::new(
            "workload",
            MediaKind::Image,
            "   ",
            BenchmarkSample {
                duration: Duration::from_millis(10),
                input_bytes: bytes,
                output_bytes: bytes,
                unit: WorkloadUnit::Bytes,
                peak_memory: PeakMemory::Unavailable,
            },
        );
        assert_eq!(
            blank_tool,
            Err(DomainError::EmptyField("benchmark.tool_label"))
        );
        Ok(())
    }
}
