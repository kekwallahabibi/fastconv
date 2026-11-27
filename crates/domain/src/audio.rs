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
    Aac,
    M4a,
    Alac,
    Aiff,
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
            Self::Aac => "aac",
            Self::M4a | Self::Alac => "m4a",
            Self::Aiff => "aiff",
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
            Self::Aac => "AAC (ADTS)",
            Self::M4a => "M4A",
            Self::Alac => "ALAC",
            Self::Aiff => "AIFF",
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
    Ogg,
    Mp3,
    Aac,
    M4a,
    Alac,
    Aiff,
}

impl AudioOutputFormat {
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Opus => "opus",
            Self::Ogg => "ogg",
            Self::Mp3 => "mp3",
            Self::Aac => "aac",
            Self::M4a | Self::Alac => "m4a",
            Self::Aiff => "aiff",
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Wav => "WAV",
            Self::Flac => "FLAC",
            Self::Opus => "Opus",
            Self::Ogg => "Ogg Vorbis",
            Self::Mp3 => "MP3",
            Self::Aac => "AAC",
            Self::M4a => "M4A (AAC)",
            Self::Alac => "ALAC (M4A)",
            Self::Aiff => "AIFF",
        }
    }

    #[must_use]
    pub const fn is_lossless(self) -> bool {
        matches!(self, Self::Wav | Self::Flac | Self::Alac | Self::Aiff)
    }

    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        if ext.eq_ignore_ascii_case("wav") || ext.eq_ignore_ascii_case("wave") {
            Some(Self::Wav)
        } else if ext.eq_ignore_ascii_case("flac") {
            Some(Self::Flac)
        } else if ext.eq_ignore_ascii_case("opus") {
            Some(Self::Opus)
        } else if ext.eq_ignore_ascii_case("ogg") || ext.eq_ignore_ascii_case("oga") {
            Some(Self::Ogg)
        } else if ext.eq_ignore_ascii_case("mp3") {
            Some(Self::Mp3)
        } else if ext.eq_ignore_ascii_case("aac") {
            Some(Self::Aac)
        } else if ext.eq_ignore_ascii_case("m4a") {
            Some(Self::M4a)
        } else if ext.eq_ignore_ascii_case("alac") {
            Some(Self::Alac)
        } else if ext.eq_ignore_ascii_case("aiff")
            || ext.eq_ignore_ascii_case("aif")
            || ext.eq_ignore_ascii_case("aifc")
        {
            Some(Self::Aiff)
        } else {
            None
        }
    }

    /// Returns true if `ext` is a valid target file extension for this output format.
    ///
    /// ALAC is stored inside an M4A container, so both `.m4a` and `.alac` are accepted when
    /// `--to alac` is explicitly requested.
    #[must_use]
    pub fn accepts_extension(self, ext: &str) -> bool {
        match self {
            Self::Alac => ext.eq_ignore_ascii_case("m4a") || ext.eq_ignore_ascii_case("alac"),
            _ => Self::from_extension(ext) == Some(self),
        }
    }
}

impl fmt::Display for AudioOutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Validated lossy-audio target bitrate in bits per second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct AudioBitrate(NonZeroU32);

impl AudioBitrate {
    /// Creates a bitrate from an exact bits-per-second value.
    ///
    /// CLI syntax parsing belongs at the system boundary in `fastconv-probe`; this
    /// constructor only accepts an already-resolved exact rate.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidAudioOptionBitrate`] when `bps` is zero.
    pub fn new(bps: u32) -> Result<Self, DomainError> {
        NonZeroU32::new(bps)
            .map(Self)
            .ok_or_else(|| DomainError::InvalidAudioOptionBitrate {
                raw: bps.to_string(),
                expected: "a positive bits-per-second value",
            })
    }

    #[must_use]
    pub const fn bps(self) -> u32 {
        self.0.get()
    }

    /// Rounds to whole kbps for encoders whose configuration is expressed in kbps.
    #[must_use]
    pub fn kbps_rounded(self) -> u32 {
        (self.0.get().saturating_add(500) / 1_000).max(1)
    }
}

/// Validated lossy-audio quality on a 0..=100 scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct AudioQuality(u8);

impl AudioQuality {
    pub const MAX: u8 = 100;

    /// # Errors
    ///
    /// Returns [`DomainError::InvalidAudioQuality`] when `value` exceeds 100.
    pub fn new(value: u8) -> Result<Self, DomainError> {
        if value <= Self::MAX {
            Ok(Self(value))
        } else {
            Err(DomainError::InvalidAudioQuality {
                value,
                max: Self::MAX,
            })
        }
    }

    #[must_use]
    pub const fn value(self) -> u8 {
        self.0
    }
}

/// Mutually exclusive rate-control choice for lossy audio encoders.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize)]
pub enum AudioRateControl {
    #[default]
    Default,
    Bitrate(AudioBitrate),
    Quality(AudioQuality),
}

/// Validated audio sample rate in Hz plus the rate tables of the native encoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct AudioSampleRate(NonZeroU32);

impl AudioSampleRate {
    /// The nine MPEG Layer III sample rates accepted by `rusty_mp3`.
    pub const MP3_RATES: [u32; 9] = [
        8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000,
    ];
    /// MPEG-2/2.5 Layer III rates. `rusty_mp3` caps Layer III at 160 kbps here and
    /// silently snaps a higher bitrate down, so the domain rejects that combination.
    pub const MP3_MPEG2_RATES: [u32; 6] = [8_000, 11_025, 12_000, 16_000, 22_050, 24_000];
    /// The AAC sample-rate table (`rusty_aac::SAMPLE_RATES`), including 7.35 kHz.
    pub const AAC_RATES: [u32; 13] = [
        7_350, 8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 64_000,
        88_200, 96_000,
    ];
    pub const OPUS_RATE_HZ: u32 = 48_000;
    pub const MIN_HZ: u32 = 8_000;
    pub const MAX_HZ: u32 = 192_000;

    /// Creates a sample rate. The absolute ceiling is enforced here; per-format
    /// tables and the 8 kHz floor are enforced by [`AudioEncodingOptions::new`].
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidAudioOptionSampleRate`] for zero or rates above 192 kHz.
    pub fn new(hz: u32) -> Result<Self, DomainError> {
        NonZeroU32::new(hz)
            .filter(|rate| rate.get() <= Self::MAX_HZ)
            .map(Self)
            .ok_or(DomainError::InvalidAudioOptionSampleRate {
                hz,
                expected: "1..=192000 Hz",
            })
    }

    #[must_use]
    pub const fn hz(self) -> u32 {
        self.0.get()
    }
}

/// Validated channel count for audio conversion (1..=8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct AudioChannels(NonZeroU16);

impl AudioChannels {
    pub const MAX: u16 = 8;

    /// # Errors
    ///
    /// Returns [`DomainError::InvalidAudioOptionChannels`] for zero or more than eight channels.
    pub fn new(channels: u16) -> Result<Self, DomainError> {
        if (1..=Self::MAX).contains(&channels) {
            NonZeroU16::new(channels)
                .map(Self)
                .ok_or(DomainError::InvalidAudioOptionChannels {
                    channels,
                    expected: "1..=8",
                })
        } else {
            Err(DomainError::InvalidAudioOptionChannels {
                channels,
                expected: "1..=8",
            })
        }
    }

    #[must_use]
    pub const fn get(self) -> u16 {
        self.0.get()
    }
}

/// Validated encoding options carried by an [`AudioBatchPlan`].
///
/// A default instance selects the exact encoder settings used before these
/// options existed, so the type carries no behaviour change on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AudioEncodingOptions {
    rate_control: AudioRateControl,
    sample_rate: Option<AudioSampleRate>,
    channels: Option<AudioChannels>,
}

impl Default for AudioEncodingOptions {
    fn default() -> Self {
        Self {
            rate_control: AudioRateControl::Default,
            sample_rate: None,
            channels: None,
        }
    }
}

impl AudioEncodingOptions {
    /// Highest Layer III bitrate available at MPEG-2/2.5 sample rates.
    const MP3_MPEG2_MAX_BPS: u32 = 160_000;

    /// Creates options and validates every field against the output format.
    ///
    /// # Errors
    ///
    /// Returns a [`DomainError`] for any bitrate, quality, sample-rate, or
    /// channel count the native encoder for `format` cannot represent.
    pub fn new(
        format: AudioOutputFormat,
        rate_control: AudioRateControl,
        sample_rate: Option<AudioSampleRate>,
        channels: Option<AudioChannels>,
    ) -> Result<Self, DomainError> {
        let options = Self {
            rate_control,
            sample_rate,
            channels,
        };
        options.validate(format)?;
        Ok(options)
    }

    #[must_use]
    pub const fn rate_control(&self) -> AudioRateControl {
        self.rate_control
    }

    #[must_use]
    pub const fn sample_rate(&self) -> Option<AudioSampleRate> {
        self.sample_rate
    }

    #[must_use]
    pub const fn channels(&self) -> Option<AudioChannels> {
        self.channels
    }

    fn validate(&self, format: AudioOutputFormat) -> Result<(), DomainError> {
        Self::validate_rate_control(format, self.rate_control)?;
        if let Some(rate) = self.sample_rate {
            Self::validate_sample_rate(format, rate)?;
        }
        if let Some(channels) = self.channels {
            Self::validate_channels(format, channels)?;
        }
        // Cross-field check: an explicit MPEG-2/2.5 rate cannot carry more than
        // 160 kbps through rusty_mp3 without a silent downgrade to 160 kbps.
        // Without an explicit rate the pipeline resolves the effective rate itself.
        if format == AudioOutputFormat::Mp3
            && let AudioRateControl::Bitrate(bitrate) = self.rate_control
            && let Some(rate) = self.sample_rate
            && AudioSampleRate::MP3_MPEG2_RATES.contains(&rate.hz())
            && bitrate.bps() > Self::MP3_MPEG2_MAX_BPS
        {
            return Err(DomainError::IncompatibleAudioBitrateRate {
                bps: bitrate.bps(),
                hz: rate.hz(),
                max_bps: Self::MP3_MPEG2_MAX_BPS,
            });
        }
        Ok(())
    }

    fn validate_rate_control(
        format: AudioOutputFormat,
        control: AudioRateControl,
    ) -> Result<(), DomainError> {
        let unsupported =
            |flag: &'static str, reason: &'static str| DomainError::UnsupportedAudioRateControl {
                format,
                control: flag,
                reason,
            };
        match control {
            AudioRateControl::Default => {}
            AudioRateControl::Bitrate(bitrate) => {
                if format.is_lossless() {
                    return Err(unsupported(
                        "--bitrate",
                        "lossless formats store the source PCM unchanged",
                    ));
                }
                let (min, max, expected) = match format {
                    AudioOutputFormat::Mp3 => (32_000, 320_000, "32000..=320000 bps for MP3"),
                    AudioOutputFormat::Aac | AudioOutputFormat::M4a => {
                        (32_000, 512_000, "32000..=512000 bps for AAC/M4A")
                    }
                    AudioOutputFormat::Ogg => (32_000, 512_000, "32000..=512000 bps for Ogg"),
                    AudioOutputFormat::Opus => (6_000, 510_000, "6000..=510000 bps for Opus"),
                    AudioOutputFormat::Wav
                    | AudioOutputFormat::Flac
                    | AudioOutputFormat::Alac
                    | AudioOutputFormat::Aiff => {
                        return Err(unsupported(
                            "--bitrate",
                            "lossless formats store the source PCM unchanged",
                        ));
                    }
                };
                let bps = bitrate.bps();
                if !(min..=max).contains(&bps) {
                    return Err(DomainError::InvalidAudioOptionBitrate {
                        raw: bps.to_string(),
                        expected,
                    });
                }
            }
            AudioRateControl::Quality(quality) => {
                if format.is_lossless() {
                    return Err(unsupported(
                        "--quality",
                        "lossless formats store the source PCM unchanged",
                    ));
                }
                if matches!(format, AudioOutputFormat::Aac | AudioOutputFormat::M4a) {
                    return Err(unsupported(
                        "--quality",
                        "the AAC encoder exposes only bitrate_bps; use --bitrate",
                    ));
                }
                // 0..=100 is guaranteed by AudioQuality's validated constructor.
                debug_assert!(quality.value() <= AudioQuality::MAX);
            }
        }
        Ok(())
    }

    fn validate_sample_rate(
        format: AudioOutputFormat,
        rate: AudioSampleRate,
    ) -> Result<(), DomainError> {
        let hz = rate.hz();
        if format == AudioOutputFormat::Opus && hz != AudioSampleRate::OPUS_RATE_HZ {
            return Err(DomainError::InvalidAudioOptionSampleRate {
                hz,
                expected: "exactly 48000 Hz (Ogg Opus always encodes at 48 kHz)",
            });
        }
        let valid = match format {
            AudioOutputFormat::Opus => true,
            AudioOutputFormat::Mp3 => AudioSampleRate::MP3_RATES.contains(&hz),
            AudioOutputFormat::Aac | AudioOutputFormat::M4a => {
                AudioSampleRate::AAC_RATES.contains(&hz)
            }
            AudioOutputFormat::Ogg
            | AudioOutputFormat::Wav
            | AudioOutputFormat::Flac
            | AudioOutputFormat::Alac
            | AudioOutputFormat::Aiff => {
                (AudioSampleRate::MIN_HZ..=AudioSampleRate::MAX_HZ).contains(&hz)
            }
        };
        if valid {
            return Ok(());
        }
        let expected = match format {
            AudioOutputFormat::Mp3 => {
                "one of 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, or 48000 Hz"
            }
            AudioOutputFormat::Aac | AudioOutputFormat::M4a => {
                "one of 7350, 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, or 96000 Hz"
            }
            _ => "8000..=192000 Hz",
        };
        Err(DomainError::InvalidAudioOptionSampleRate { hz, expected })
    }

    fn validate_channels(
        format: AudioOutputFormat,
        channels: AudioChannels,
    ) -> Result<(), DomainError> {
        let count = channels.get();
        let (valid, expected) = match format {
            AudioOutputFormat::Mp3 | AudioOutputFormat::Opus => {
                ((1..=2).contains(&count), "1 or 2 channels")
            }
            AudioOutputFormat::Aac | AudioOutputFormat::M4a | AudioOutputFormat::Ogg => (
                (1..=6).contains(&count) && count != 4,
                "1, 2, 3, 5, or 6 channels",
            ),
            AudioOutputFormat::Wav
            | AudioOutputFormat::Flac
            | AudioOutputFormat::Alac
            | AudioOutputFormat::Aiff => ((1..=8).contains(&count), "1..=8 channels"),
        };
        if valid {
            Ok(())
        } else {
            Err(DomainError::InvalidAudioOptionChannels {
                channels: count,
                expected,
            })
        }
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
    CpuOgg,
    CpuMp3,
    CpuAac,
    CpuM4a,
    CpuAlac,
    CpuAiff,
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
                AudioOutputFormat::Ogg => Self::CpuOgg,
                AudioOutputFormat::Mp3 => Self::CpuMp3,
                AudioOutputFormat::Aac => Self::CpuAac,
                AudioOutputFormat::M4a => Self::CpuM4a,
                AudioOutputFormat::Alac => Self::CpuAlac,
                AudioOutputFormat::Aiff => Self::CpuAiff,
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
            Self::CpuOgg => AudioOutputFormat::Ogg,
            Self::CpuMp3 => AudioOutputFormat::Mp3,
            Self::CpuAac => AudioOutputFormat::Aac,
            Self::CpuM4a => AudioOutputFormat::M4a,
            Self::CpuAlac => AudioOutputFormat::Alac,
            Self::CpuAiff => AudioOutputFormat::Aiff,
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
            Self::CpuOgg => "ogg vorbis (cpu)",
            Self::CpuMp3 => "mp3 (cpu)",
            Self::CpuAac => "aac adts (cpu)",
            Self::CpuM4a => "m4a aac (cpu)",
            Self::CpuAlac => "alac m4a (cpu)",
            Self::CpuAiff => "aiff pcm (cpu)",
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
    options: AudioEncodingOptions,
}

impl AudioBatchPlan {
    /// Creates a validated [`AudioBatchPlan`] with default encoder options.
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
        Self::with_options(items, mode, worker_threads, AudioEncodingOptions::default())
    }

    /// Creates a validated plan whose encoder options are checked against the
    /// mode's output format, so a mode/options mismatch cannot be represented.
    ///
    /// # Errors
    ///
    /// Returns all errors of [`Self::new`] plus any [`DomainError`] raised when
    /// revalidating `options` for `mode.output_format()`.
    pub fn with_options(
        items: Vec<AudioBatchItem>,
        mode: AudioExecutionMode,
        worker_threads: NonZeroUsize,
        options: AudioEncodingOptions,
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
        options.validate(mode.output_format())?;
        Ok(Self {
            items,
            mode,
            worker_threads,
            options,
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
    pub const fn options(&self) -> AudioEncodingOptions {
        self.options
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
        for (format, expected) in [
            (AudioOutputFormat::Wav, AudioExecutionMode::CpuWav),
            (AudioOutputFormat::Flac, AudioExecutionMode::CpuFlac),
            (AudioOutputFormat::Opus, AudioExecutionMode::CpuOpus),
            (AudioOutputFormat::Ogg, AudioExecutionMode::CpuOgg),
            (AudioOutputFormat::Mp3, AudioExecutionMode::CpuMp3),
            (AudioOutputFormat::Aac, AudioExecutionMode::CpuAac),
            (AudioOutputFormat::M4a, AudioExecutionMode::CpuM4a),
            (AudioOutputFormat::Alac, AudioExecutionMode::CpuAlac),
            (AudioOutputFormat::Aiff, AudioExecutionMode::CpuAiff),
        ] {
            assert_eq!(
                AudioExecutionMode::resolve(format, HardwareBackend::Cpu),
                Ok(expected)
            );
            assert_eq!(expected.output_format(), format);
            for backend in [GpuBackend::Nvenc, GpuBackend::Vaapi, GpuBackend::Vulkan] {
                assert!(matches!(
                    AudioExecutionMode::resolve(format, HardwareBackend::Gpu(backend)),
                    Err(DomainError::UnsupportedAudioPipeline { .. })
                ));
            }
        }
        assert!(AudioOutputFormat::Alac.accepts_extension("m4a"));
        assert!(AudioOutputFormat::Alac.accepts_extension("alac"));
        assert!(!AudioOutputFormat::Alac.accepts_extension("flac"));
        assert!(AudioOutputFormat::Aiff.is_lossless());
        assert!(AudioOutputFormat::Alac.is_lossless());
        assert!(!AudioOutputFormat::Aac.is_lossless());
        assert!(!AudioOutputFormat::M4a.is_lossless());
        assert!(!AudioOutputFormat::Mp3.is_lossless());
        assert!(!AudioOutputFormat::Ogg.is_lossless());
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

    fn default_options() -> AudioEncodingOptions {
        AudioEncodingOptions::default()
    }

    #[test]
    fn bitrate_constructor_accepts_only_exact_nonzero_bps() -> Result<(), DomainError> {
        assert_eq!(AudioBitrate::new(191_500)?.bps(), 191_500);
        assert_eq!(AudioBitrate::new(191_500)?.kbps_rounded(), 192);
        assert_eq!(AudioBitrate::new(1)?.kbps_rounded(), 1);
        assert!(matches!(
            AudioBitrate::new(0),
            Err(DomainError::InvalidAudioOptionBitrate { .. })
        ));
        Ok(())
    }

    #[test]
    fn validates_quality_sample_rate_and_channel_bounds() -> Result<(), DomainError> {
        assert_eq!(AudioQuality::new(100)?.value(), AudioQuality::MAX);
        assert!(matches!(
            AudioQuality::new(101),
            Err(DomainError::InvalidAudioQuality {
                value: 101,
                max: 100
            })
        ));
        assert!(matches!(
            AudioSampleRate::new(0),
            Err(DomainError::InvalidAudioOptionSampleRate { .. })
        ));
        assert!(matches!(
            AudioSampleRate::new(AudioSampleRate::MAX_HZ + 1),
            Err(DomainError::InvalidAudioOptionSampleRate { .. })
        ));
        assert_eq!(AudioSampleRate::new(44_100)?.hz(), 44_100);
        assert!(matches!(
            AudioChannels::new(0),
            Err(DomainError::InvalidAudioOptionChannels { .. })
        ));
        assert!(matches!(
            AudioChannels::new(9),
            Err(DomainError::InvalidAudioOptionChannels { .. })
        ));
        assert_eq!(AudioChannels::new(8)?.get(), 8);
        Ok(())
    }

    #[test]
    fn rejects_illegal_encoding_options_per_format() -> Result<(), DomainError> {
        let bitrate = |bps| AudioBitrate::new(bps).map(AudioRateControl::Bitrate);
        let quality = |value| AudioQuality::new(value).map(AudioRateControl::Quality);

        for format in [
            AudioOutputFormat::Wav,
            AudioOutputFormat::Flac,
            AudioOutputFormat::Alac,
            AudioOutputFormat::Aiff,
        ] {
            assert!(matches!(
                AudioEncodingOptions::new(format, bitrate(128_000)?, None, None),
                Err(DomainError::UnsupportedAudioRateControl { .. })
            ));
            assert!(matches!(
                AudioEncodingOptions::new(format, quality(80)?, None, None),
                Err(DomainError::UnsupportedAudioRateControl { .. })
            ));
        }
        for format in [AudioOutputFormat::Aac, AudioOutputFormat::M4a] {
            assert!(matches!(
                AudioEncodingOptions::new(format, quality(80)?, None, None),
                Err(DomainError::UnsupportedAudioRateControl { .. })
            ));
        }

        for (format, rejected) in [
            (AudioOutputFormat::Mp3, 31_000),
            (AudioOutputFormat::Mp3, 321_000),
            (AudioOutputFormat::Aac, 31_000),
            (AudioOutputFormat::M4a, 513_000),
            (AudioOutputFormat::Ogg, 5_000),
            (AudioOutputFormat::Opus, 5_999),
            (AudioOutputFormat::Opus, 511_000),
        ] {
            assert!(
                matches!(
                    AudioEncodingOptions::new(format, bitrate(rejected)?, None, None),
                    Err(DomainError::InvalidAudioOptionBitrate { .. })
                ),
                "{format} must reject {rejected} bps"
            );
        }
        assert!(
            AudioEncodingOptions::new(AudioOutputFormat::Mp3, bitrate(32_000)?, None, None).is_ok()
        );
        assert!(
            AudioEncodingOptions::new(AudioOutputFormat::Mp3, bitrate(320_000)?, None, None)
                .is_ok()
        );
        assert!(
            AudioEncodingOptions::new(AudioOutputFormat::Opus, bitrate(6_000)?, None, None).is_ok()
        );
        assert!(
            AudioEncodingOptions::new(AudioOutputFormat::Opus, bitrate(510_000)?, None, None)
                .is_ok()
        );
        for format in [
            AudioOutputFormat::Mp3,
            AudioOutputFormat::Ogg,
            AudioOutputFormat::Opus,
        ] {
            assert!(AudioEncodingOptions::new(format, quality(0)?, None, None).is_ok());
            assert!(AudioEncodingOptions::new(format, quality(100)?, None, None).is_ok());
        }
        Ok(())
    }

    #[test]
    fn rejects_mp3_high_bitrate_at_mpeg2_sample_rates() -> Result<(), DomainError> {
        let bitrate = |bps| AudioBitrate::new(bps).map(AudioRateControl::Bitrate);
        let rate = |hz| AudioSampleRate::new(hz).map(Some);

        for hz in AudioSampleRate::MP3_MPEG2_RATES {
            assert!(
                matches!(
                    AudioEncodingOptions::new(
                        AudioOutputFormat::Mp3,
                        bitrate(320_000)?,
                        rate(hz)?,
                        None
                    ),
                    Err(DomainError::IncompatibleAudioBitrateRate {
                        bps: 320_000,
                        max_bps: 160_000,
                        ..
                    })
                ),
                "MP3 at {hz} Hz must reject 320000 bps"
            );
            assert!(
                AudioEncodingOptions::new(
                    AudioOutputFormat::Mp3,
                    bitrate(160_000)?,
                    rate(hz)?,
                    None
                )
                .is_ok(),
                "MP3 at {hz} Hz must accept 160000 bps"
            );
        }
        for hz in [32_000, 44_100, 48_000] {
            assert!(
                AudioEncodingOptions::new(
                    AudioOutputFormat::Mp3,
                    bitrate(320_000)?,
                    rate(hz)?,
                    None
                )
                .is_ok(),
                "MP3 at {hz} Hz (MPEG-1) must accept 320000 bps"
            );
        }
        assert!(
            AudioEncodingOptions::new(AudioOutputFormat::Mp3, bitrate(320_000)?, None, None)
                .is_ok()
        );
        Ok(())
    }

    #[test]
    fn rejects_illegal_sample_rates_per_format() -> Result<(), DomainError> {
        let rate = |hz| AudioSampleRate::new(hz).map(Some);

        assert!(
            AudioEncodingOptions::new(
                AudioOutputFormat::Opus,
                AudioRateControl::Default,
                rate(48_000)?,
                None
            )
            .is_ok()
        );
        for hz in [8_000, 44_100, 96_000] {
            assert!(
                matches!(
                    AudioEncodingOptions::new(
                        AudioOutputFormat::Opus,
                        AudioRateControl::Default,
                        rate(hz)?,
                        None
                    ),
                    Err(DomainError::InvalidAudioOptionSampleRate { .. })
                ),
                "Opus must reject {hz} Hz"
            );
        }
        for hz in [8_000, 11_025, 48_000] {
            assert!(
                AudioEncodingOptions::new(
                    AudioOutputFormat::Mp3,
                    AudioRateControl::Default,
                    rate(hz)?,
                    None
                )
                .is_ok()
            );
        }
        assert!(matches!(
            AudioEncodingOptions::new(
                AudioOutputFormat::Mp3,
                AudioRateControl::Default,
                rate(7_350)?,
                None
            ),
            Err(DomainError::InvalidAudioOptionSampleRate { .. })
        ));
        assert!(
            AudioEncodingOptions::new(
                AudioOutputFormat::M4a,
                AudioRateControl::Default,
                rate(7_350)?,
                None
            )
            .is_ok()
        );
        for format in [
            AudioOutputFormat::Wav,
            AudioOutputFormat::Flac,
            AudioOutputFormat::Ogg,
        ] {
            assert!(
                AudioEncodingOptions::new(format, AudioRateControl::Default, rate(44_100)?, None)
                    .is_ok()
            );
            assert!(matches!(
                AudioEncodingOptions::new(format, AudioRateControl::Default, rate(7_999)?, None),
                Err(DomainError::InvalidAudioOptionSampleRate { .. })
            ));
        }
        Ok(())
    }

    #[test]
    fn rejects_illegal_channels_per_format() -> Result<(), DomainError> {
        let channels = |count| AudioChannels::new(count).map(Some);
        for format in [AudioOutputFormat::Mp3, AudioOutputFormat::Opus] {
            assert!(
                AudioEncodingOptions::new(format, AudioRateControl::Default, None, channels(2)?)
                    .is_ok()
            );
            assert!(matches!(
                AudioEncodingOptions::new(format, AudioRateControl::Default, None, channels(3)?),
                Err(DomainError::InvalidAudioOptionChannels { .. })
            ));
        }
        for format in [
            AudioOutputFormat::Aac,
            AudioOutputFormat::M4a,
            AudioOutputFormat::Ogg,
        ] {
            for count in [1, 2, 3, 5, 6] {
                assert!(
                    AudioEncodingOptions::new(
                        format,
                        AudioRateControl::Default,
                        None,
                        channels(count)?
                    )
                    .is_ok()
                );
            }
            assert!(matches!(
                AudioEncodingOptions::new(format, AudioRateControl::Default, None, channels(4)?),
                Err(DomainError::InvalidAudioOptionChannels { .. })
            ));
        }
        assert!(
            AudioEncodingOptions::new(
                AudioOutputFormat::Wav,
                AudioRateControl::Default,
                None,
                channels(8)?
            )
            .is_ok()
        );
        Ok(())
    }

    #[test]
    fn plan_rejects_mp3_high_bitrate_at_low_rate_before_io() {
        let workers = NonZeroUsize::MIN;
        let options = AudioEncodingOptions {
            rate_control: AudioRateControl::Bitrate(
                AudioBitrate::new(320_000).unwrap_or_else(|_| unreachable!()),
            ),
            sample_rate: Some(AudioSampleRate::new(24_000).unwrap_or_else(|_| unreachable!())),
            channels: None,
        };
        let result = AudioBatchPlan::with_options(
            vec![item("a")],
            AudioExecutionMode::CpuMp3,
            workers,
            options,
        );
        assert!(matches!(
            result,
            Err(DomainError::IncompatibleAudioBitrateRate { .. })
        ));
    }

    #[test]
    fn plan_revalidates_options_against_its_mode() {
        let workers = NonZeroUsize::MIN;
        let bitrate = AudioBitrate::new(128_000).unwrap_or_else(|_| unreachable!());
        let invalid_for_wav = AudioEncodingOptions::new(
            AudioOutputFormat::Mp3,
            AudioRateControl::Bitrate(bitrate),
            None,
            None,
        )
        .unwrap_or_else(|_| unreachable!());
        let result = AudioBatchPlan::with_options(
            vec![item("a")],
            AudioExecutionMode::CpuWav,
            workers,
            invalid_for_wav,
        );
        assert!(matches!(
            result,
            Err(DomainError::UnsupportedAudioRateControl { .. })
        ));
        let valid = AudioEncodingOptions::new(
            AudioOutputFormat::Mp3,
            AudioRateControl::Bitrate(bitrate),
            None,
            None,
        )
        .unwrap_or_else(|_| unreachable!());
        let plan = AudioBatchPlan::with_options(
            vec![item("a")],
            AudioExecutionMode::CpuMp3,
            workers,
            valid,
        )
        .unwrap_or_else(|_| unreachable!());
        assert_eq!(plan.options(), valid);
        assert_eq!(default_options(), AudioEncodingOptions::default());
    }
}
