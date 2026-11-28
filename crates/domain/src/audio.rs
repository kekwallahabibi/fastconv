use std::fmt;
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[cfg(feature = "serde")]
use serde::Serialize;

use crate::{DomainError, HardwareBackend, PathRole, u64_to_f64};

/// Audio container/codec formats accepted at the system boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum AudioInputFormat {
    Wav,
    Flac,
    Mp3,
    Vorbis,
    Opus,
    OggFlac,
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
            Self::Vorbis | Self::OggFlac => "ogg",
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
            Self::OggFlac => "Ogg FLAC",
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
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
    /// Every audio output container, in CLI help order.
    pub const ALL: [Self; 9] = [
        Self::Wav,
        Self::Flac,
        Self::Opus,
        Self::Ogg,
        Self::Mp3,
        Self::Aac,
        Self::M4a,
        Self::Alac,
        Self::Aiff,
    ];

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

    /// CLI-facing container name, free of any codec qualifier.
    #[must_use]
    pub const fn cli_name(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Opus => "opus",
            Self::Ogg => "ogg",
            Self::Mp3 => "mp3",
            Self::Aac => "aac",
            Self::M4a => "m4a",
            Self::Alac => "alac",
            Self::Aiff => "aiff",
        }
    }

    /// Neutral container label for diagnostics: never misleading pairs such as
    /// "M4A (AAC)" or "Ogg Vorbis" when an explicit codec is rejected.
    #[must_use]
    pub const fn container_label(self) -> &'static str {
        match self {
            Self::Wav => "WAV",
            Self::Flac => "FLAC",
            Self::Opus => "Opus",
            Self::Ogg => "Ogg",
            Self::Mp3 => "MP3",
            Self::Aac => "AAC",
            Self::M4a => "M4A",
            Self::Alac => "ALAC",
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

    /// The codec used when `--codec` is absent.
    #[must_use]
    pub const fn default_codec(self) -> AudioTargetCodec {
        match self {
            Self::Wav | Self::Aiff => AudioTargetCodec::Pcm,
            Self::Flac => AudioTargetCodec::Flac,
            Self::Opus => AudioTargetCodec::Opus,
            Self::Ogg => AudioTargetCodec::Vorbis,
            Self::Mp3 => AudioTargetCodec::Mp3,
            Self::Aac | Self::M4a => AudioTargetCodec::Aac,
            Self::Alac => AudioTargetCodec::Alac,
        }
    }

    /// Codecs this container can carry, in help-message order.
    #[must_use]
    pub const fn supported_codecs(self) -> &'static [AudioTargetCodec] {
        match self {
            Self::Ogg => &[
                AudioTargetCodec::Vorbis,
                AudioTargetCodec::Opus,
                AudioTargetCodec::Flac,
            ],
            Self::M4a => &[AudioTargetCodec::Aac, AudioTargetCodec::Alac],
            Self::Wav | Self::Aiff => &[AudioTargetCodec::Pcm],
            Self::Flac => &[AudioTargetCodec::Flac],
            Self::Opus => &[AudioTargetCodec::Opus],
            Self::Mp3 => &[AudioTargetCodec::Mp3],
            Self::Aac => &[AudioTargetCodec::Aac],
            Self::Alac => &[AudioTargetCodec::Alac],
        }
    }

    /// Label for the resolved container/codec pair, used in diagnostics so an
    /// explicit `--codec` is never reported under the container's default name.
    #[must_use]
    pub const fn resolved_label(self, codec: AudioTargetCodec) -> &'static str {
        match (self, codec) {
            (Self::Ogg, AudioTargetCodec::Vorbis) => "Ogg Vorbis",
            (Self::Ogg, AudioTargetCodec::Opus) => "Ogg Opus",
            (Self::Ogg, AudioTargetCodec::Flac) => "Ogg FLAC",
            (Self::M4a, AudioTargetCodec::Aac) => "M4A (AAC)",
            (Self::M4a, AudioTargetCodec::Alac) => "M4A (ALAC)",
            (Self::Alac, _) => "ALAC (M4A)",
            _ => self.label(),
        }
    }

    /// Resolves `codec` against the codecs this container can carry.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedAudioCodecForFormat`] when an explicit
    /// codec cannot be stored in this container.
    pub fn resolve_codec(
        self,
        codec: Option<AudioTargetCodec>,
    ) -> Result<AudioTargetCodec, DomainError> {
        match codec {
            None => Ok(self.default_codec()),
            Some(codec) if self.supported_codecs().contains(&codec) => Ok(codec),
            Some(codec) => Err(DomainError::UnsupportedAudioCodecForFormat {
                format: self,
                codec,
            }),
        }
    }
}

impl fmt::Display for AudioOutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Narrow codec selector for containers that can carry more than one codec.
///
/// Pairing a codec with an incompatible container is rejected by
/// [`AudioOutputFormat::resolve_codec`] before a plan can be built, so an
/// `(AudioOutputFormat, AudioTargetCodec)` pair inside a plan is always valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum AudioTargetCodec {
    Pcm,
    Flac,
    Opus,
    Vorbis,
    Mp3,
    Aac,
    Alac,
}

impl AudioTargetCodec {
    /// Every audio codec, in CLI help order.
    pub const ALL: [Self; 7] = [
        Self::Pcm,
        Self::Flac,
        Self::Opus,
        Self::Vorbis,
        Self::Mp3,
        Self::Aac,
        Self::Alac,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Pcm => "PCM",
            Self::Flac => "FLAC",
            Self::Opus => "Opus",
            Self::Vorbis => "Vorbis",
            Self::Mp3 => "MP3",
            Self::Aac => "AAC",
            Self::Alac => "ALAC",
        }
    }

    /// CLI-facing codec name, used in diagnostics and help messages.
    #[must_use]
    pub const fn cli_name(self) -> &'static str {
        match self {
            Self::Pcm => "pcm",
            Self::Flac => "flac",
            Self::Opus => "opus",
            Self::Vorbis => "vorbis",
            Self::Mp3 => "mp3",
            Self::Aac => "aac",
            Self::Alac => "alac",
        }
    }

    /// Containers that can carry this codec, in canonical order. This is the
    /// reverse of [`AudioOutputFormat::supported_codecs`]; both must agree.
    #[must_use]
    pub const fn supported_formats(self) -> &'static [AudioOutputFormat] {
        match self {
            Self::Pcm => &[AudioOutputFormat::Wav, AudioOutputFormat::Aiff],
            Self::Flac => &[AudioOutputFormat::Flac, AudioOutputFormat::Ogg],
            Self::Opus => &[AudioOutputFormat::Opus, AudioOutputFormat::Ogg],
            Self::Vorbis => &[AudioOutputFormat::Ogg],
            Self::Mp3 => &[AudioOutputFormat::Mp3],
            Self::Aac => &[AudioOutputFormat::Aac, AudioOutputFormat::M4a],
            Self::Alac => &[AudioOutputFormat::M4a, AudioOutputFormat::Alac],
        }
    }

    #[must_use]
    pub const fn is_lossless(self) -> bool {
        matches!(self, Self::Pcm | Self::Flac | Self::Alac)
    }
}

impl fmt::Display for AudioTargetCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Validated PCM sample depth requested for a lossless audio target.
///
/// This is a lossless-encoder knob only: lossy codecs consume normalized `f32`
/// samples and reject every explicit depth at the domain boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum AudioBitDepth {
    Int8,
    Int16,
    Int24,
    Int32,
    Float32,
}

impl AudioBitDepth {
    #[must_use]
    pub const fn bits(self) -> u16 {
        match self {
            Self::Int8 => 8,
            Self::Int16 => 16,
            Self::Int24 => 24,
            Self::Int32 | Self::Float32 => 32,
        }
    }

    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::Float32)
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Int8 => "8",
            Self::Int16 => "16",
            Self::Int24 => "24",
            Self::Int32 => "32",
            Self::Float32 => "f32",
        }
    }
}

impl fmt::Display for AudioBitDepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Whether a decoded stream carries integer or IEEE float samples.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum AudioSampleFormat {
    #[default]
    Int,
    Float,
}

/// Validated lossy-audio target bitrate in bits per second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct AudioOptionBitrate(NonZeroU32);

impl AudioOptionBitrate {
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

/// Maximum bitrate the AAC frame model can carry at a rate/channel pair.
///
/// `rusty_aac` stores at most 6144 bits per channel in each 1024-sample frame,
/// i.e. 6 bits per sample per channel. Asking for more makes the encoder silently
/// lower the bitrate, so both the domain and the pipeline reject such pairs.
#[must_use]
pub const fn aac_frame_capacity_bps(hz: u32, channels: u16) -> u32 {
    hz.saturating_mul(channels as u32).saturating_mul(6)
}

/// Validated lossy-audio quality on a 0..=100 scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum AudioRateControl {
    #[default]
    Default,
    Bitrate(AudioOptionBitrate),
    Quality(AudioQuality),
}

/// Validated audio sample rate in Hz plus the rate tables of the native encoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct AudioOptionSampleRate(NonZeroU32);

impl AudioOptionSampleRate {
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
    /// The M4A sample entry stores the rate in a 16.16 fixed-point field, so an
    /// integer rate above 65535 would be written as zero and break reopening.
    pub const ALAC_MAX_HZ: u32 = 0xFFFF;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct AudioOptionChannels(NonZeroU16);

impl AudioOptionChannels {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct AudioEncodingOptions {
    rate_control: AudioRateControl,
    sample_rate: Option<AudioOptionSampleRate>,
    channels: Option<AudioOptionChannels>,
    bit_depth: Option<AudioBitDepth>,
}

impl Default for AudioEncodingOptions {
    fn default() -> Self {
        Self {
            rate_control: AudioRateControl::Default,
            sample_rate: None,
            channels: None,
            bit_depth: None,
        }
    }
}

impl AudioEncodingOptions {
    /// Highest Layer III bitrate available at MPEG-2/2.5 sample rates.
    const MP3_MPEG2_MAX_BPS: u32 = 160_000;

    /// Creates options and validates every field against the output format's
    /// default codec.
    ///
    /// # Errors
    ///
    /// Returns a [`DomainError`] for any bitrate, quality, sample-rate, or
    /// channel count the native encoder for `format` cannot represent.
    pub fn new(
        format: AudioOutputFormat,
        rate_control: AudioRateControl,
        sample_rate: Option<AudioOptionSampleRate>,
        channels: Option<AudioOptionChannels>,
        bit_depth: Option<AudioBitDepth>,
    ) -> Result<Self, DomainError> {
        Self::for_codec(
            format,
            format.default_codec(),
            rate_control,
            sample_rate,
            channels,
            bit_depth,
        )
    }

    /// Creates options and validates every field against the resolved codec,
    /// including that `codec` is storable in `format`.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedAudioCodecForFormat`] for an invalid
    /// container/codec pair and a [`DomainError`] for any option the native
    /// encoder for that codec cannot represent.
    pub fn for_codec(
        format: AudioOutputFormat,
        codec: AudioTargetCodec,
        rate_control: AudioRateControl,
        sample_rate: Option<AudioOptionSampleRate>,
        channels: Option<AudioOptionChannels>,
        bit_depth: Option<AudioBitDepth>,
    ) -> Result<Self, DomainError> {
        let options = Self {
            rate_control,
            sample_rate,
            channels,
            bit_depth,
        };
        options.validate(format, codec)?;
        Ok(options)
    }

    #[must_use]
    pub const fn rate_control(&self) -> AudioRateControl {
        self.rate_control
    }

    #[must_use]
    pub const fn sample_rate(&self) -> Option<AudioOptionSampleRate> {
        self.sample_rate
    }

    #[must_use]
    pub const fn channels(&self) -> Option<AudioOptionChannels> {
        self.channels
    }

    #[must_use]
    pub const fn bit_depth(&self) -> Option<AudioBitDepth> {
        self.bit_depth
    }

    fn validate(
        &self,
        format: AudioOutputFormat,
        codec: AudioTargetCodec,
    ) -> Result<(), DomainError> {
        format.resolve_codec(Some(codec))?;
        Self::validate_rate_control(format, codec, self.rate_control)?;
        if let Some(depth) = self.bit_depth {
            Self::validate_bit_depth(format, codec, depth)?;
        }
        if let Some(rate) = self.sample_rate {
            Self::validate_sample_rate(codec, rate)?;
        }
        if let Some(channels) = self.channels {
            Self::validate_channels(codec, channels)?;
        }
        // Cross-field check: an explicit MPEG-2/2.5 rate cannot carry more than
        // 160 kbps through rusty_mp3 without a silent downgrade to 160 kbps.
        // Without an explicit rate the pipeline resolves the effective rate itself.
        if codec == AudioTargetCodec::Mp3
            && let AudioRateControl::Bitrate(bitrate) = self.rate_control
            && let Some(rate) = self.sample_rate
            && AudioOptionSampleRate::MP3_MPEG2_RATES.contains(&rate.hz())
            && bitrate.bps() > Self::MP3_MPEG2_MAX_BPS
        {
            return Err(DomainError::IncompatibleAudioBitrateRate {
                bps: bitrate.bps(),
                hz: rate.hz(),
                max_bps: Self::MP3_MPEG2_MAX_BPS,
            });
        }
        // AAC stores at most 6144 bits per channel per 1024-sample frame. With an
        // explicit rate and channel count the limit is knowable before any I/O;
        // without either one the pipeline checks it against the effective values.
        if codec == AudioTargetCodec::Aac
            && let AudioRateControl::Bitrate(bitrate) = self.rate_control
            && let Some(rate) = self.sample_rate
            && let Some(channels) = self.channels
        {
            let max_bps = aac_frame_capacity_bps(rate.hz(), channels.get());
            if bitrate.bps() > max_bps {
                return Err(DomainError::IncompatibleAacBitrateRate {
                    bps: bitrate.bps(),
                    hz: rate.hz(),
                    channels: channels.get(),
                    max_bps,
                });
            }
        }
        Ok(())
    }

    /// Checks an explicit PCM depth against what the target encoder can actually store.
    ///
    /// The limits come from the native encoders: `flacenc` caps samples at 24-bit
    /// integers, ALAC in this pipeline is wired for 16-bit integer samples, and the
    /// lossy encoders only accept normalized `f32` input.
    fn validate_bit_depth(
        format: AudioOutputFormat,
        codec: AudioTargetCodec,
        depth: AudioBitDepth,
    ) -> Result<(), DomainError> {
        let valid = match codec {
            AudioTargetCodec::Pcm => true,
            AudioTargetCodec::Flac => matches!(
                depth,
                AudioBitDepth::Int8 | AudioBitDepth::Int16 | AudioBitDepth::Int24
            ),
            // `alac-encoder` implements `PcmFormat` only for `i16` and hardcodes
            // `bits_per_channel: 16`, so every other depth must be rejected.
            AudioTargetCodec::Alac => depth == AudioBitDepth::Int16,
            AudioTargetCodec::Mp3
            | AudioTargetCodec::Aac
            | AudioTargetCodec::Opus
            | AudioTargetCodec::Vorbis => false,
        };
        if valid {
            return Ok(());
        }
        let reason = match codec {
            AudioTargetCodec::Flac => "FLAC stores at most 24-bit integer PCM; use 8, 16 or 24",
            AudioTargetCodec::Alac => "ALAC stores 16-bit integer PCM only; use 16",
            AudioTargetCodec::Mp3
            | AudioTargetCodec::Aac
            | AudioTargetCodec::Opus
            | AudioTargetCodec::Vorbis => "lossy codecs consume normalized f32 samples",
            AudioTargetCodec::Pcm => "",
        };
        Err(DomainError::UnsupportedAudioBitDepth {
            format,
            codec,
            bit_depth: depth,
            reason,
        })
    }

    fn validate_rate_control(
        format: AudioOutputFormat,
        codec: AudioTargetCodec,
        control: AudioRateControl,
    ) -> Result<(), DomainError> {
        let unsupported =
            |flag: &'static str, reason: &'static str| DomainError::UnsupportedAudioRateControl {
                format,
                codec,
                control: flag,
                reason,
            };
        match control {
            AudioRateControl::Default => {}
            AudioRateControl::Bitrate(bitrate) => {
                if codec.is_lossless() {
                    return Err(unsupported(
                        "--bitrate",
                        "lossless formats store the source PCM unchanged",
                    ));
                }
                let (min, max, expected) = match codec {
                    AudioTargetCodec::Mp3 => (32_000, 320_000, "32000..=320000 bps for MP3"),
                    AudioTargetCodec::Aac => (32_000, 512_000, "32000..=512000 bps for AAC"),
                    AudioTargetCodec::Vorbis => (32_000, 512_000, "32000..=512000 bps for Vorbis"),
                    AudioTargetCodec::Opus => (6_000, 510_000, "6000..=510000 bps for Opus"),
                    AudioTargetCodec::Pcm | AudioTargetCodec::Flac | AudioTargetCodec::Alac => {
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
                if codec.is_lossless() {
                    return Err(unsupported(
                        "--quality",
                        "lossless formats store the source PCM unchanged",
                    ));
                }
                if codec == AudioTargetCodec::Aac {
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
        codec: AudioTargetCodec,
        rate: AudioOptionSampleRate,
    ) -> Result<(), DomainError> {
        let hz = rate.hz();
        if codec == AudioTargetCodec::Opus && hz != AudioOptionSampleRate::OPUS_RATE_HZ {
            return Err(DomainError::InvalidAudioOptionSampleRate {
                hz,
                expected: "exactly 48000 Hz (Ogg Opus always encodes at 48 kHz)",
            });
        }
        if codec == AudioTargetCodec::Alac && hz > AudioOptionSampleRate::ALAC_MAX_HZ {
            return Err(DomainError::InvalidAudioOptionSampleRate {
                hz,
                expected: "8000..=65535 Hz for ALAC (M4A stores a 16.16 fixed-point rate)",
            });
        }
        let valid = match codec {
            AudioTargetCodec::Opus => true,
            AudioTargetCodec::Mp3 => AudioOptionSampleRate::MP3_RATES.contains(&hz),
            AudioTargetCodec::Aac => AudioOptionSampleRate::AAC_RATES.contains(&hz),
            AudioTargetCodec::Vorbis
            | AudioTargetCodec::Pcm
            | AudioTargetCodec::Flac
            | AudioTargetCodec::Alac => {
                (AudioOptionSampleRate::MIN_HZ..=AudioOptionSampleRate::MAX_HZ).contains(&hz)
            }
        };
        if valid {
            return Ok(());
        }
        let expected = match codec {
            AudioTargetCodec::Mp3 => {
                "one of 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, or 48000 Hz"
            }
            AudioTargetCodec::Aac => {
                "one of 7350, 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, or 96000 Hz"
            }
            _ => "8000..=192000 Hz",
        };
        Err(DomainError::InvalidAudioOptionSampleRate { hz, expected })
    }

    fn validate_channels(
        codec: AudioTargetCodec,
        channels: AudioOptionChannels,
    ) -> Result<(), DomainError> {
        let count = channels.get();
        let (valid, expected) = match codec {
            AudioTargetCodec::Mp3 | AudioTargetCodec::Opus => {
                ((1..=2).contains(&count), "1 or 2 channels")
            }
            AudioTargetCodec::Aac | AudioTargetCodec::Vorbis => (
                (1..=6).contains(&count) && count != 4,
                "1, 2, 3, 5, or 6 channels",
            ),
            AudioTargetCodec::Pcm | AudioTargetCodec::Flac | AudioTargetCodec::Alac => {
                ((1..=8).contains(&count), "1..=8 channels")
            }
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

/// FNV-1a 64-bit hash of a byte slice, used to bind a saved WAV layout to the
/// exact header bytes the probe validated.
#[must_use]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// Validated canonical signed-integer PCM layout of a RIFF/WAVE `data` chunk.
///
/// The boundary probe parses the WAV header once and records the exact payload
/// range here. Downstream crates consume these values without re-parsing the
/// header; they only open the file and read the validated byte range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct PcmWavLayout {
    channels: NonZeroU16,
    bits_per_sample: NonZeroU16,
    sample_rate: NonZeroU32,
    block_align: NonZeroU16,
    /// `dwChannelMask` from WAVEFORMATEXTENSIBLE; zero for plain PCM or an
    /// unspecified layout. Kept so consumers can reject layouts whose speaker
    /// order differs from the one their fast path assumes.
    channel_mask: u32,
    data_offset: u64,
    data_bytes: u64,
    /// FNV-1a of the file bytes before the payload. Re-checked by the pipeline
    /// so a same-length replacement with a moved `data` chunk cannot be read at
    /// a stale offset.
    header_fingerprint: u64,
}

impl PcmWavLayout {
    /// Validates the parsed header values of a canonical signed PCM WAV.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidPcmWavLayout`] when the values cannot
    /// describe a whole number of signed-integer PCM frames.
    // Eight independent header facts must be validated together; introducing a
    // wrapper struct would only move the same arity one level down.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        channels: NonZeroU16,
        bits_per_sample: NonZeroU16,
        sample_rate: NonZeroU32,
        block_align: NonZeroU16,
        channel_mask: u32,
        data_offset: u64,
        data_bytes: u64,
        header_fingerprint: u64,
    ) -> Result<Self, DomainError> {
        let invalid = |reason| DomainError::InvalidPcmWavLayout { reason };
        if !(1..=8).contains(&channels.get()) {
            return Err(invalid("channel count must be 1..=8"));
        }
        if !matches!(bits_per_sample.get(), 16 | 24 | 32) {
            return Err(invalid("bit depth must be 16, 24 or 32"));
        }
        let expected_align =
            usize::from(channels.get()).saturating_mul(usize::from(bits_per_sample.get()) / 8);
        let expected_align = u16::try_from(expected_align).map_err(|_| invalid("frame size"))?;
        if block_align.get() != expected_align {
            return Err(invalid("block alignment does not match channels and depth"));
        }
        if data_bytes == 0 {
            return Err(invalid("empty data chunk"));
        }
        let align = u64::from(block_align.get());
        if !data_bytes.is_multiple_of(align) {
            return Err(invalid("data chunk is not a whole number of frames"));
        }
        data_offset
            .checked_add(data_bytes)
            .ok_or(invalid("data chunk range overflow"))?;
        Ok(Self {
            channels,
            bits_per_sample,
            sample_rate,
            block_align,
            channel_mask,
            data_offset,
            data_bytes,
            header_fingerprint,
        })
    }

    #[must_use]
    pub const fn channels(self) -> NonZeroU16 {
        self.channels
    }

    #[must_use]
    pub const fn bits_per_sample(self) -> NonZeroU16 {
        self.bits_per_sample
    }

    #[must_use]
    pub const fn sample_rate(self) -> NonZeroU32 {
        self.sample_rate
    }

    #[must_use]
    pub const fn block_align(self) -> NonZeroU16 {
        self.block_align
    }

    #[must_use]
    pub const fn channel_mask(self) -> u32 {
        self.channel_mask
    }

    #[must_use]
    pub const fn data_offset(self) -> u64 {
        self.data_offset
    }

    #[must_use]
    pub const fn data_bytes(self) -> u64 {
        self.data_bytes
    }

    #[must_use]
    pub const fn total_samples(self) -> u64 {
        self.data_bytes / self.block_align.get() as u64
    }

    #[must_use]
    pub const fn header_fingerprint(self) -> u64 {
        self.header_fingerprint
    }
}

/// Validated decoded stream metadata produced by the boundary probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct AudioStreamInfo {
    channels: NonZeroU16,
    sample_rate: NonZeroU32,
    bits_per_sample: Option<NonZeroU16>,
    sample_format: AudioSampleFormat,
    total_samples: Option<u64>,
    /// Present only when the source is a canonical signed PCM WAV that the
    /// pipeline may stream directly from the validated `data` chunk.
    pcm_wav_layout: Option<PcmWavLayout>,
}

impl AudioStreamInfo {
    /// Creates validated integer-PCM stream metadata.
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
        Self::with_sample_format(
            channels,
            sample_rate,
            bits_per_sample,
            AudioSampleFormat::Int,
            total_samples,
        )
    }

    /// Creates validated stream metadata carrying an explicit sample format.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyField`] when `channels` or `sample_rate` is zero.
    pub fn with_sample_format(
        channels: NonZeroU16,
        sample_rate: NonZeroU32,
        bits_per_sample: Option<NonZeroU16>,
        sample_format: AudioSampleFormat,
        total_samples: Option<u64>,
    ) -> Result<Self, DomainError> {
        if channels.get() == 0 || sample_rate.get() == 0 {
            return Err(DomainError::EmptyField("audio.stream_info"));
        }
        Ok(Self {
            channels,
            sample_rate,
            bits_per_sample,
            sample_format,
            total_samples,
            pcm_wav_layout: None,
        })
    }

    /// Attaches the validated canonical PCM WAV layout parsed at the boundary.
    ///
    /// The layout is authoritative: every stream field is derived from it so
    /// admission budgets and conversion cannot disagree when symphonia reported
    /// different outer metadata for the same file.
    #[must_use]
    pub const fn with_pcm_wav_layout(mut self, layout: PcmWavLayout) -> Self {
        self.channels = layout.channels();
        self.sample_rate = layout.sample_rate();
        self.bits_per_sample = Some(layout.bits_per_sample());
        self.sample_format = AudioSampleFormat::Int;
        self.total_samples = Some(layout.total_samples());
        self.pcm_wav_layout = Some(layout);
        self
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
    pub const fn sample_format(self) -> AudioSampleFormat {
        self.sample_format
    }

    #[must_use]
    pub const fn total_samples(self) -> Option<u64> {
        self.total_samples
    }

    /// Returns the validated canonical PCM WAV payload layout, if the source is one.
    #[must_use]
    pub const fn pcm_wav_layout(self) -> Option<PcmWavLayout> {
        self.pcm_wav_layout
    }
}

/// Concrete CPU encoder selected for an audio batch.
///
/// There is no GPU variant: audio conversion on a GPU backend is rejected before a plan is
/// created, so a GPU audio plan cannot be represented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum AudioExecutionMode {
    CpuWav,
    CpuFlac,
    CpuOpus,
    CpuOgg,
    CpuOggOpus,
    CpuOggFlac,
    CpuMp3,
    CpuAac,
    CpuM4a,
    CpuM4aAlac,
    CpuAlac,
    CpuAiff,
}

impl AudioExecutionMode {
    /// Resolves the CPU encoder for `format` with its default codec.
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
        Self::resolve_with_codec(format, None, backend)
    }

    /// Resolves the CPU encoder for an explicit container/codec pair. Invalid
    /// pairs and GPU backends are rejected before any plan can be represented.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedAudioPipeline`] for any GPU backend and
    /// [`DomainError::UnsupportedAudioCodecForFormat`] for an incompatible pair.
    pub fn resolve_with_codec(
        format: AudioOutputFormat,
        codec: Option<AudioTargetCodec>,
        backend: HardwareBackend,
    ) -> Result<Self, DomainError> {
        match backend {
            HardwareBackend::Cpu => {
                let codec = format.resolve_codec(codec)?;
                Ok(match (format, codec) {
                    (AudioOutputFormat::Wav, AudioTargetCodec::Pcm) => Self::CpuWav,
                    (AudioOutputFormat::Flac, AudioTargetCodec::Flac) => Self::CpuFlac,
                    (AudioOutputFormat::Opus, AudioTargetCodec::Opus) => Self::CpuOpus,
                    (AudioOutputFormat::Ogg, AudioTargetCodec::Vorbis) => Self::CpuOgg,
                    (AudioOutputFormat::Ogg, AudioTargetCodec::Opus) => Self::CpuOggOpus,
                    (AudioOutputFormat::Ogg, AudioTargetCodec::Flac) => Self::CpuOggFlac,
                    (AudioOutputFormat::Mp3, AudioTargetCodec::Mp3) => Self::CpuMp3,
                    (AudioOutputFormat::Aac, AudioTargetCodec::Aac) => Self::CpuAac,
                    (AudioOutputFormat::M4a, AudioTargetCodec::Aac) => Self::CpuM4a,
                    (AudioOutputFormat::M4a, AudioTargetCodec::Alac) => Self::CpuM4aAlac,
                    (AudioOutputFormat::Alac, AudioTargetCodec::Alac) => Self::CpuAlac,
                    (AudioOutputFormat::Aiff, AudioTargetCodec::Pcm) => Self::CpuAiff,
                    // `format.resolve_codec` already accepts only the pairs above.
                    _ => {
                        return Err(DomainError::UnsupportedAudioCodecForFormat { format, codec });
                    }
                })
            }
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
            Self::CpuOgg | Self::CpuOggOpus | Self::CpuOggFlac => AudioOutputFormat::Ogg,
            Self::CpuMp3 => AudioOutputFormat::Mp3,
            Self::CpuAac => AudioOutputFormat::Aac,
            Self::CpuM4a | Self::CpuM4aAlac => AudioOutputFormat::M4a,
            Self::CpuAlac => AudioOutputFormat::Alac,
            Self::CpuAiff => AudioOutputFormat::Aiff,
        }
    }

    #[must_use]
    pub const fn codec(self) -> AudioTargetCodec {
        match self {
            Self::CpuWav | Self::CpuAiff => AudioTargetCodec::Pcm,
            Self::CpuFlac | Self::CpuOggFlac => AudioTargetCodec::Flac,
            Self::CpuOpus | Self::CpuOggOpus => AudioTargetCodec::Opus,
            Self::CpuOgg => AudioTargetCodec::Vorbis,
            Self::CpuMp3 => AudioTargetCodec::Mp3,
            Self::CpuAac | Self::CpuM4a => AudioTargetCodec::Aac,
            Self::CpuM4aAlac | Self::CpuAlac => AudioTargetCodec::Alac,
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
            Self::CpuOggOpus => "ogg opus (cpu)",
            Self::CpuOggFlac => "ogg flac (cpu)",
            Self::CpuMp3 => "mp3 (cpu)",
            Self::CpuAac => "aac adts (cpu)",
            Self::CpuM4a => "m4a aac (cpu)",
            Self::CpuM4aAlac => "m4a alac (cpu)",
            Self::CpuAlac => "alac m4a (cpu)",
            Self::CpuAiff => "aiff pcm (cpu)",
        })
    }
}

/// Single validated audio conversion item.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
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
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
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
        options.validate(mode.output_format(), mode.codec())?;
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
    pub const fn target_codec(&self) -> AudioTargetCodec {
        self.mode.codec()
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
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
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

    fn audio_options(
        format: AudioOutputFormat,
        rate_control: AudioRateControl,
        sample_rate: Option<AudioOptionSampleRate>,
        channels: Option<AudioOptionChannels>,
    ) -> Result<AudioEncodingOptions, DomainError> {
        AudioEncodingOptions::new(format, rate_control, sample_rate, channels, None)
    }

    #[test]
    fn audio_bit_depth_reports_bits_and_float_kind() {
        for (depth, bits, float) in [
            (AudioBitDepth::Int8, 8, false),
            (AudioBitDepth::Int16, 16, false),
            (AudioBitDepth::Int24, 24, false),
            (AudioBitDepth::Int32, 32, false),
            (AudioBitDepth::Float32, 32, true),
        ] {
            assert_eq!(depth.bits(), bits);
            assert_eq!(depth.is_float(), float);
            assert_eq!(depth.to_string(), depth.label());
        }
        assert_eq!(AudioSampleFormat::default(), AudioSampleFormat::Int);
    }

    #[test]
    fn stream_info_defaults_to_integer_samples() -> Result<(), DomainError> {
        let channels = NonZeroU16::new(2).unwrap_or(NonZeroU16::MIN);
        let rate = NonZeroU32::new(48_000).unwrap_or(NonZeroU32::MIN);
        let bits = NonZeroU16::new(32);
        let plain = AudioStreamInfo::new(channels, rate, bits, None)?;
        assert_eq!(plain.sample_format(), AudioSampleFormat::Int);
        let float = AudioStreamInfo::with_sample_format(
            channels,
            rate,
            bits,
            AudioSampleFormat::Float,
            None,
        )?;
        assert_eq!(float.sample_format(), AudioSampleFormat::Float);
        assert_eq!(float.bits_per_sample(), bits);
        Ok(())
    }

    #[test]
    fn wav_layout_overrides_mismatched_stream_metadata() -> Result<(), DomainError> {
        let layout = PcmWavLayout::new(
            NonZeroU16::new(1).unwrap_or(NonZeroU16::MIN),
            NonZeroU16::new(16).unwrap_or(NonZeroU16::MIN),
            NonZeroU32::new(44_100).unwrap_or(NonZeroU32::MIN),
            NonZeroU16::new(2).unwrap_or(NonZeroU16::MIN),
            0,
            44,
            4_000,
            fnv1a64(b"header"),
        )?;
        // Deliberately wrong outer metadata for the same file: stereo, 48 kHz,
        // float and a different frame count.
        let outer = AudioStreamInfo::with_sample_format(
            NonZeroU16::new(2).unwrap_or(NonZeroU16::MIN),
            NonZeroU32::new(48_000).unwrap_or(NonZeroU32::MIN),
            NonZeroU16::new(32),
            AudioSampleFormat::Float,
            Some(999),
        )?
        .with_pcm_wav_layout(layout);
        assert_eq!(outer.channels().get(), 1);
        assert_eq!(outer.sample_rate().get(), 44_100);
        assert_eq!(
            outer.bits_per_sample(),
            NonZeroU16::new(16),
            "layout bit depth must win"
        );
        assert_eq!(outer.sample_format(), AudioSampleFormat::Int);
        assert_eq!(outer.total_samples(), Some(2_000));
        assert_eq!(outer.pcm_wav_layout(), Some(layout));
        Ok(())
    }

    #[test]
    fn validates_explicit_bit_depth_for_every_output_format() -> Result<(), DomainError> {
        let all = [
            AudioBitDepth::Int8,
            AudioBitDepth::Int16,
            AudioBitDepth::Int24,
            AudioBitDepth::Int32,
            AudioBitDepth::Float32,
        ];
        let options = |format, depth: Option<AudioBitDepth>| {
            AudioEncodingOptions::new(format, AudioRateControl::Default, None, None, depth)
        };
        for format in [AudioOutputFormat::Wav, AudioOutputFormat::Aiff] {
            for depth in all {
                let resolved = options(format, Some(depth))?;
                assert_eq!(resolved.bit_depth(), Some(depth), "{format} {depth}");
            }
        }
        for depth in [
            AudioBitDepth::Int8,
            AudioBitDepth::Int16,
            AudioBitDepth::Int24,
        ] {
            assert_eq!(
                options(AudioOutputFormat::Flac, Some(depth))?.bit_depth(),
                Some(depth)
            );
        }
        for depth in [AudioBitDepth::Int32, AudioBitDepth::Float32] {
            assert!(matches!(
                options(AudioOutputFormat::Flac, Some(depth)),
                Err(DomainError::UnsupportedAudioBitDepth { .. })
            ));
        }
        assert_eq!(
            options(AudioOutputFormat::Alac, Some(AudioBitDepth::Int16))?.bit_depth(),
            Some(AudioBitDepth::Int16)
        );
        for depth in [
            AudioBitDepth::Int8,
            AudioBitDepth::Int24,
            AudioBitDepth::Int32,
            AudioBitDepth::Float32,
        ] {
            assert!(matches!(
                options(AudioOutputFormat::Alac, Some(depth)),
                Err(DomainError::UnsupportedAudioBitDepth { .. })
            ));
        }
        for format in [
            AudioOutputFormat::Mp3,
            AudioOutputFormat::Aac,
            AudioOutputFormat::M4a,
            AudioOutputFormat::Opus,
            AudioOutputFormat::Ogg,
        ] {
            for depth in all {
                assert!(
                    matches!(
                        options(format, Some(depth)),
                        Err(DomainError::UnsupportedAudioBitDepth { .. })
                    ),
                    "{format} must reject {depth}"
                );
            }
            assert_eq!(options(format, None)?.bit_depth(), None);
        }
        Ok(())
    }

    #[test]
    fn bitrate_constructor_accepts_only_exact_nonzero_bps() -> Result<(), DomainError> {
        assert_eq!(AudioOptionBitrate::new(191_500)?.bps(), 191_500);
        assert_eq!(AudioOptionBitrate::new(191_500)?.kbps_rounded(), 192);
        assert_eq!(AudioOptionBitrate::new(1)?.kbps_rounded(), 1);
        assert!(matches!(
            AudioOptionBitrate::new(0),
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
            AudioOptionSampleRate::new(0),
            Err(DomainError::InvalidAudioOptionSampleRate { .. })
        ));
        assert!(matches!(
            AudioOptionSampleRate::new(AudioOptionSampleRate::MAX_HZ + 1),
            Err(DomainError::InvalidAudioOptionSampleRate { .. })
        ));
        assert_eq!(AudioOptionSampleRate::new(44_100)?.hz(), 44_100);
        assert!(matches!(
            AudioOptionChannels::new(0),
            Err(DomainError::InvalidAudioOptionChannels { .. })
        ));
        assert!(matches!(
            AudioOptionChannels::new(9),
            Err(DomainError::InvalidAudioOptionChannels { .. })
        ));
        assert_eq!(AudioOptionChannels::new(8)?.get(), 8);
        Ok(())
    }

    #[test]
    fn rejects_illegal_encoding_options_per_format() -> Result<(), DomainError> {
        let bitrate = |bps| AudioOptionBitrate::new(bps).map(AudioRateControl::Bitrate);
        let quality = |value| AudioQuality::new(value).map(AudioRateControl::Quality);

        for format in [
            AudioOutputFormat::Wav,
            AudioOutputFormat::Flac,
            AudioOutputFormat::Alac,
            AudioOutputFormat::Aiff,
        ] {
            assert!(matches!(
                audio_options(format, bitrate(128_000)?, None, None),
                Err(DomainError::UnsupportedAudioRateControl { .. })
            ));
            assert!(matches!(
                audio_options(format, quality(80)?, None, None),
                Err(DomainError::UnsupportedAudioRateControl { .. })
            ));
        }
        for format in [AudioOutputFormat::Aac, AudioOutputFormat::M4a] {
            assert!(matches!(
                audio_options(format, quality(80)?, None, None),
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
                    audio_options(format, bitrate(rejected)?, None, None),
                    Err(DomainError::InvalidAudioOptionBitrate { .. })
                ),
                "{format} must reject {rejected} bps"
            );
        }
        assert!(audio_options(AudioOutputFormat::Mp3, bitrate(32_000)?, None, None).is_ok());
        assert!(audio_options(AudioOutputFormat::Mp3, bitrate(320_000)?, None, None).is_ok());
        assert!(audio_options(AudioOutputFormat::Opus, bitrate(6_000)?, None, None).is_ok());
        assert!(audio_options(AudioOutputFormat::Opus, bitrate(510_000)?, None, None).is_ok());
        for format in [
            AudioOutputFormat::Mp3,
            AudioOutputFormat::Ogg,
            AudioOutputFormat::Opus,
        ] {
            assert!(audio_options(format, quality(0)?, None, None).is_ok());
            assert!(audio_options(format, quality(100)?, None, None).is_ok());
        }
        Ok(())
    }

    #[test]
    fn rejects_mp3_high_bitrate_at_mpeg2_sample_rates() -> Result<(), DomainError> {
        let bitrate = |bps| AudioOptionBitrate::new(bps).map(AudioRateControl::Bitrate);
        let rate = |hz| AudioOptionSampleRate::new(hz).map(Some);

        for hz in AudioOptionSampleRate::MP3_MPEG2_RATES {
            assert!(
                matches!(
                    audio_options(AudioOutputFormat::Mp3, bitrate(320_000)?, rate(hz)?, None),
                    Err(DomainError::IncompatibleAudioBitrateRate {
                        bps: 320_000,
                        max_bps: 160_000,
                        ..
                    })
                ),
                "MP3 at {hz} Hz must reject 320000 bps"
            );
            assert!(
                audio_options(AudioOutputFormat::Mp3, bitrate(160_000)?, rate(hz)?, None).is_ok(),
                "MP3 at {hz} Hz must accept 160000 bps"
            );
        }
        for hz in [32_000, 44_100, 48_000] {
            assert!(
                audio_options(AudioOutputFormat::Mp3, bitrate(320_000)?, rate(hz)?, None).is_ok(),
                "MP3 at {hz} Hz (MPEG-1) must accept 320000 bps"
            );
        }
        assert!(audio_options(AudioOutputFormat::Mp3, bitrate(320_000)?, None, None).is_ok());
        Ok(())
    }

    #[test]
    fn rejects_aac_bitrate_above_the_frame_capacity() -> Result<(), DomainError> {
        let bitrate = |bps| AudioOptionBitrate::new(bps).map(AudioRateControl::Bitrate);
        let rate = |hz| AudioOptionSampleRate::new(hz).map(Some);
        let channels = |count| AudioOptionChannels::new(count).map(Some);
        for format in [AudioOutputFormat::Aac, AudioOutputFormat::M4a] {
            assert!(matches!(
                audio_options(format, bitrate(512_000)?, rate(8_000)?, channels(1)?),
                Err(DomainError::IncompatibleAacBitrateRate {
                    bps: 512_000,
                    hz: 8_000,
                    channels: 1,
                    max_bps: 48_000,
                })
            ));
            assert!(audio_options(format, bitrate(256_000)?, rate(48_000)?, channels(2)?).is_ok());
        }
        assert_eq!(aac_frame_capacity_bps(8_000, 1), 48_000);
        assert_eq!(aac_frame_capacity_bps(48_000, 2), 576_000);
        Ok(())
    }

    #[test]
    fn rejects_illegal_sample_rates_per_format() -> Result<(), DomainError> {
        let rate = |hz| AudioOptionSampleRate::new(hz).map(Some);

        assert!(
            audio_options(
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
                    audio_options(
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
                audio_options(
                    AudioOutputFormat::Mp3,
                    AudioRateControl::Default,
                    rate(hz)?,
                    None
                )
                .is_ok()
            );
        }
        assert!(matches!(
            audio_options(
                AudioOutputFormat::Mp3,
                AudioRateControl::Default,
                rate(7_350)?,
                None
            ),
            Err(DomainError::InvalidAudioOptionSampleRate { .. })
        ));
        assert!(
            audio_options(
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
            assert!(audio_options(format, AudioRateControl::Default, rate(44_100)?, None).is_ok());
            assert!(matches!(
                audio_options(format, AudioRateControl::Default, rate(7_999)?, None),
                Err(DomainError::InvalidAudioOptionSampleRate { .. })
            ));
        }
        Ok(())
    }

    #[test]
    fn rejects_illegal_channels_per_format() -> Result<(), DomainError> {
        let channels = |count| AudioOptionChannels::new(count).map(Some);
        for format in [AudioOutputFormat::Mp3, AudioOutputFormat::Opus] {
            assert!(audio_options(format, AudioRateControl::Default, None, channels(2)?).is_ok());
            assert!(matches!(
                audio_options(format, AudioRateControl::Default, None, channels(3)?),
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
                    audio_options(format, AudioRateControl::Default, None, channels(count)?)
                        .is_ok()
                );
            }
            assert!(matches!(
                audio_options(format, AudioRateControl::Default, None, channels(4)?),
                Err(DomainError::InvalidAudioOptionChannels { .. })
            ));
        }
        assert!(
            audio_options(
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
                AudioOptionBitrate::new(320_000).unwrap_or_else(|_| unreachable!()),
            ),
            sample_rate: Some(
                AudioOptionSampleRate::new(24_000).unwrap_or_else(|_| unreachable!()),
            ),
            channels: None,
            bit_depth: None,
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
        let bitrate = AudioOptionBitrate::new(128_000).unwrap_or_else(|_| unreachable!());
        let invalid_for_wav = audio_options(
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
        let valid = audio_options(
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

    fn codec_options(
        format: AudioOutputFormat,
        codec: AudioTargetCodec,
        control: AudioRateControl,
        sample_rate: Option<u32>,
        channels: Option<u16>,
        depth: Option<AudioBitDepth>,
    ) -> Result<AudioEncodingOptions, DomainError> {
        let rate = sample_rate.map(AudioOptionSampleRate::new).transpose()?;
        let channels = channels.map(AudioOptionChannels::new).transpose()?;
        AudioEncodingOptions::for_codec(format, codec, control, rate, channels, depth)
    }

    fn error_text(result: Result<AudioEncodingOptions, DomainError>) -> String {
        match result {
            Ok(_) => String::new(),
            Err(err) => err.to_string(),
        }
    }

    #[test]
    fn resolves_default_and_explicit_codecs_per_container() -> Result<(), DomainError> {
        for (format, default, supported) in [
            (
                AudioOutputFormat::Wav,
                AudioTargetCodec::Pcm,
                &[AudioTargetCodec::Pcm][..],
            ),
            (
                AudioOutputFormat::Aiff,
                AudioTargetCodec::Pcm,
                &[AudioTargetCodec::Pcm][..],
            ),
            (
                AudioOutputFormat::Flac,
                AudioTargetCodec::Flac,
                &[AudioTargetCodec::Flac][..],
            ),
            (
                AudioOutputFormat::Opus,
                AudioTargetCodec::Opus,
                &[AudioTargetCodec::Opus][..],
            ),
            (
                AudioOutputFormat::Mp3,
                AudioTargetCodec::Mp3,
                &[AudioTargetCodec::Mp3][..],
            ),
            (
                AudioOutputFormat::Aac,
                AudioTargetCodec::Aac,
                &[AudioTargetCodec::Aac][..],
            ),
            (
                AudioOutputFormat::Alac,
                AudioTargetCodec::Alac,
                &[AudioTargetCodec::Alac][..],
            ),
            (
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Vorbis,
                &[
                    AudioTargetCodec::Vorbis,
                    AudioTargetCodec::Opus,
                    AudioTargetCodec::Flac,
                ],
            ),
            (
                AudioOutputFormat::M4a,
                AudioTargetCodec::Aac,
                &[AudioTargetCodec::Aac, AudioTargetCodec::Alac],
            ),
        ] {
            assert_eq!(format.default_codec(), default, "{format}");
            assert_eq!(format.supported_codecs(), supported, "{format}");
            assert_eq!(format.resolve_codec(None)?, default, "{format}");
            for &codec in supported {
                assert_eq!(
                    format.resolve_codec(Some(codec))?,
                    codec,
                    "{format} {codec}"
                );
            }
            assert!(
                codec_options(format, default, AudioRateControl::Default, None, None, None).is_ok()
            );
        }

        assert_eq!(AudioInputFormat::OggFlac.extension(), "ogg");
        assert_eq!(AudioInputFormat::OggFlac.label(), "Ogg FLAC");
        Ok(())
    }

    #[test]
    fn rejects_invalid_container_codec_pairs() -> Result<(), DomainError> {
        let invalid = [
            (
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Aac,
                "vorbis, opus, flac",
                "aac, m4a",
            ),
            (
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Pcm,
                "vorbis, opus, flac",
                "wav, aiff",
            ),
            (
                AudioOutputFormat::M4a,
                AudioTargetCodec::Opus,
                "aac, alac",
                "opus, ogg",
            ),
            (
                AudioOutputFormat::Flac,
                AudioTargetCodec::Alac,
                "flac",
                "m4a, alac",
            ),
            (
                AudioOutputFormat::Wav,
                AudioTargetCodec::Flac,
                "pcm",
                "flac, ogg",
            ),
        ];
        for (format, codec, expected_codecs, expected_containers) in invalid {
            let err = format
                .resolve_codec(Some(codec))
                .err()
                .ok_or(DomainError::EmptyField("audio.codec"))?;
            assert!(
                matches!(err, DomainError::UnsupportedAudioCodecForFormat { .. }),
                "{format} {codec}: {err}"
            );
            let msg = err.to_string();
            assert!(msg.contains(expected_codecs), "{msg}");
            assert!(msg.contains(expected_containers), "{msg}");
            assert_eq!(
                AudioExecutionMode::resolve_with_codec(format, Some(codec), HardwareBackend::Cpu),
                Err(DomainError::UnsupportedAudioCodecForFormat { format, codec })
            );
        }
        Ok(())
    }

    #[test]
    fn for_codec_validates_every_option_against_the_resolved_codec() -> Result<(), DomainError> {
        let bitrate = |bps| {
            AudioRateControl::Bitrate(
                AudioOptionBitrate::new(bps).unwrap_or_else(|_| unreachable!()),
            )
        };
        let quality = |value| -> Result<AudioRateControl, DomainError> {
            AudioQuality::new(value).map(AudioRateControl::Quality)
        };
        let plain = |format, codec, control, rate, channels, depth| {
            codec_options(format, codec, control, rate, channels, depth)
        };

        // Ogg + FLAC is lossless: 8/16/24-bit only; rate control is rejected and
        // diagnostics name the resolved container/codec, not Ogg Vorbis.
        for depth in [
            AudioBitDepth::Int8,
            AudioBitDepth::Int16,
            AudioBitDepth::Int24,
        ] {
            assert!(
                plain(
                    AudioOutputFormat::Ogg,
                    AudioTargetCodec::Flac,
                    AudioRateControl::Default,
                    None,
                    None,
                    Some(depth)
                )
                .is_ok()
            );
        }
        for depth in [AudioBitDepth::Int32, AudioBitDepth::Float32] {
            assert!(matches!(
                plain(
                    AudioOutputFormat::Ogg,
                    AudioTargetCodec::Flac,
                    AudioRateControl::Default,
                    None,
                    None,
                    Some(depth)
                ),
                Err(DomainError::UnsupportedAudioBitDepth { .. })
            ));
        }
        let err = error_text(plain(
            AudioOutputFormat::Ogg,
            AudioTargetCodec::Flac,
            bitrate(128_000),
            None,
            None,
            None,
        ));
        assert!(err.contains("Ogg FLAC"), "{err}");
        assert!(err.contains("--bitrate"), "{err}");
        let err = error_text(plain(
            AudioOutputFormat::Ogg,
            AudioTargetCodec::Flac,
            quality(80)?,
            None,
            None,
            None,
        ));
        assert!(err.contains("Ogg FLAC"), "{err}");
        assert!(
            plain(
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Flac,
                AudioRateControl::Default,
                Some(192_000),
                Some(8),
                None
            )
            .is_ok()
        );
        assert!(matches!(
            plain(
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Flac,
                AudioRateControl::Default,
                Some(7_999),
                None,
                None
            ),
            Err(DomainError::InvalidAudioOptionSampleRate { .. })
        ));

        Ok(())
    }

    #[test]
    fn ogg_opus_options_enforce_48k_and_rate_bounds() -> Result<(), DomainError> {
        let bitrate = |bps| {
            AudioRateControl::Bitrate(
                AudioOptionBitrate::new(bps).unwrap_or_else(|_| unreachable!()),
            )
        };
        let quality = |value| -> Result<AudioRateControl, DomainError> {
            AudioQuality::new(value).map(AudioRateControl::Quality)
        };
        let plain = |format, codec, control, rate, channels, depth| {
            codec_options(format, codec, control, rate, channels, depth)
        };

        // Ogg + Opus: 48 kHz, 1-2 channels, 6-510 kbps, no integer depth.
        assert!(
            plain(
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Opus,
                AudioRateControl::Default,
                Some(48_000),
                Some(2),
                None
            )
            .is_ok()
        );
        assert!(
            plain(
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Opus,
                bitrate(128_000),
                None,
                None,
                None
            )
            .is_ok()
        );
        assert!(
            plain(
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Opus,
                quality(80)?,
                None,
                None,
                None
            )
            .is_ok()
        );
        for (rate, channels, bps) in [
            (Some(44_100), None, None),
            (None, Some(3), None),
            (None, None, Some(5_999)),
            (None, None, Some(511_000)),
        ] {
            assert!(
                plain(
                    AudioOutputFormat::Ogg,
                    AudioTargetCodec::Opus,
                    bps.map_or(AudioRateControl::Default, bitrate),
                    rate,
                    channels,
                    None
                )
                .is_err(),
                "{rate:?} {channels:?} {bps:?}"
            );
        }
        assert!(matches!(
            plain(
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Opus,
                AudioRateControl::Default,
                None,
                None,
                Some(AudioBitDepth::Int16)
            ),
            Err(DomainError::UnsupportedAudioBitDepth { .. })
        ));

        Ok(())
    }

    #[test]
    fn ogg_vorbis_options_enforce_rate_and_layouts() -> Result<(), DomainError> {
        let bitrate = |bps| {
            AudioRateControl::Bitrate(
                AudioOptionBitrate::new(bps).unwrap_or_else(|_| unreachable!()),
            )
        };
        let quality = |value| -> Result<AudioRateControl, DomainError> {
            AudioQuality::new(value).map(AudioRateControl::Quality)
        };
        let plain = |format, codec, control, rate, channels, depth| {
            codec_options(format, codec, control, rate, channels, depth)
        };

        // Ogg + Vorbis: bitrate/quality, no integer depth, no quad layout.
        assert!(
            plain(
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Vorbis,
                bitrate(128_000),
                None,
                None,
                None
            )
            .is_ok()
        );
        assert!(
            plain(
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Vorbis,
                quality(80)?,
                None,
                None,
                None
            )
            .is_ok()
        );
        for bps in [31_000, 513_000] {
            assert!(matches!(
                plain(
                    AudioOutputFormat::Ogg,
                    AudioTargetCodec::Vorbis,
                    bitrate(bps),
                    None,
                    None,
                    None
                ),
                Err(DomainError::InvalidAudioOptionBitrate { .. })
            ));
        }
        for channels in [1, 2, 3, 5, 6] {
            assert!(
                plain(
                    AudioOutputFormat::Ogg,
                    AudioTargetCodec::Vorbis,
                    AudioRateControl::Default,
                    None,
                    Some(channels),
                    None
                )
                .is_ok()
            );
        }
        assert!(matches!(
            plain(
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Vorbis,
                AudioRateControl::Default,
                None,
                Some(4),
                None
            ),
            Err(DomainError::InvalidAudioOptionChannels { .. })
        ));

        Ok(())
    }

    #[test]
    fn m4a_alac_and_aac_options_enforce_codec_limits() -> Result<(), DomainError> {
        let bitrate = |bps| {
            AudioRateControl::Bitrate(
                AudioOptionBitrate::new(bps).unwrap_or_else(|_| unreachable!()),
            )
        };
        let quality = |value| -> Result<AudioRateControl, DomainError> {
            AudioQuality::new(value).map(AudioRateControl::Quality)
        };
        let plain = |format, codec, control, rate, channels, depth| {
            codec_options(format, codec, control, rate, channels, depth)
        };

        // M4A + ALAC: 16-bit only; rate control and wrong depths are rejected.
        assert!(
            plain(
                AudioOutputFormat::M4a,
                AudioTargetCodec::Alac,
                AudioRateControl::Default,
                None,
                None,
                Some(AudioBitDepth::Int16)
            )
            .is_ok()
        );
        for depth in [
            AudioBitDepth::Int8,
            AudioBitDepth::Int24,
            AudioBitDepth::Int32,
            AudioBitDepth::Float32,
        ] {
            let err = error_text(plain(
                AudioOutputFormat::M4a,
                AudioTargetCodec::Alac,
                AudioRateControl::Default,
                None,
                None,
                Some(depth),
            ));
            assert!(err.contains("M4A (ALAC)"), "{err}");
            assert!(err.contains("16"), "{err}");
        }
        let err = error_text(plain(
            AudioOutputFormat::M4a,
            AudioTargetCodec::Alac,
            bitrate(128_000),
            None,
            None,
            None,
        ));
        assert!(err.contains("M4A (ALAC)"), "{err}");
        assert!(
            error_text(plain(
                AudioOutputFormat::M4a,
                AudioTargetCodec::Alac,
                quality(80)?,
                None,
                None,
                None
            ))
            .contains("M4A (ALAC)")
        );
        // 96 kHz is a valid AAC rate, but the M4A ALAC sample entry stores the
        // rate in a 16.16 fixed-point field: it would write zero and break reopening.
        assert!(matches!(
            plain(
                AudioOutputFormat::M4a,
                AudioTargetCodec::Alac,
                AudioRateControl::Default,
                Some(96_000),
                None,
                Some(AudioBitDepth::Int16)
            ),
            Err(DomainError::InvalidAudioOptionSampleRate { .. })
        ));
        assert!(
            plain(
                AudioOutputFormat::M4a,
                AudioTargetCodec::Alac,
                AudioRateControl::Default,
                Some(AudioOptionSampleRate::ALAC_MAX_HZ),
                None,
                Some(AudioBitDepth::Int16)
            )
            .is_ok()
        );

        Ok(())
    }

    #[test]
    fn m4a_aac_options_enforce_bitrate_and_frame_capacity() -> Result<(), DomainError> {
        let bitrate = |bps| {
            AudioRateControl::Bitrate(
                AudioOptionBitrate::new(bps).unwrap_or_else(|_| unreachable!()),
            )
        };
        let quality = |value| -> Result<AudioRateControl, DomainError> {
            AudioQuality::new(value).map(AudioRateControl::Quality)
        };
        let plain = |format, codec, control, rate, channels, depth| {
            codec_options(format, codec, control, rate, channels, depth)
        };

        // M4A + AAC uses the bitrate path and keeps the frame-capacity check.
        assert!(
            plain(
                AudioOutputFormat::M4a,
                AudioTargetCodec::Aac,
                bitrate(128_000),
                Some(48_000),
                Some(2),
                None
            )
            .is_ok()
        );
        assert!(matches!(
            plain(
                AudioOutputFormat::M4a,
                AudioTargetCodec::Aac,
                bitrate(512_000),
                Some(8_000),
                Some(1),
                None
            ),
            Err(DomainError::IncompatibleAacBitrateRate { .. })
        ));
        let err = error_text(plain(
            AudioOutputFormat::M4a,
            AudioTargetCodec::Aac,
            quality(80)?,
            None,
            None,
            None,
        ));
        assert!(err.contains("M4A (AAC)"), "{err}");
        assert!(matches!(
            plain(
                AudioOutputFormat::M4a,
                AudioTargetCodec::Aac,
                AudioRateControl::Default,
                None,
                None,
                Some(AudioBitDepth::Int16)
            ),
            Err(DomainError::UnsupportedAudioBitDepth { .. })
        ));
        Ok(())
    }

    #[test]
    fn execution_modes_carry_their_codec_without_silent_fallback() -> Result<(), DomainError> {
        for (format, codec, mode, label) in [
            (
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Vorbis,
                AudioExecutionMode::CpuOgg,
                "ogg vorbis (cpu)",
            ),
            (
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Opus,
                AudioExecutionMode::CpuOggOpus,
                "ogg opus (cpu)",
            ),
            (
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Flac,
                AudioExecutionMode::CpuOggFlac,
                "ogg flac (cpu)",
            ),
            (
                AudioOutputFormat::M4a,
                AudioTargetCodec::Aac,
                AudioExecutionMode::CpuM4a,
                "m4a aac (cpu)",
            ),
            (
                AudioOutputFormat::M4a,
                AudioTargetCodec::Alac,
                AudioExecutionMode::CpuM4aAlac,
                "m4a alac (cpu)",
            ),
            (
                AudioOutputFormat::Flac,
                AudioTargetCodec::Flac,
                AudioExecutionMode::CpuFlac,
                "flac (cpu)",
            ),
            (
                AudioOutputFormat::Alac,
                AudioTargetCodec::Alac,
                AudioExecutionMode::CpuAlac,
                "alac m4a (cpu)",
            ),
        ] {
            assert_eq!(
                AudioExecutionMode::resolve_with_codec(format, Some(codec), HardwareBackend::Cpu)?,
                mode
            );
            assert_eq!(mode.codec(), codec);
            assert_eq!(mode.output_format(), format);
            assert_eq!(mode.to_string(), label);
            assert!(matches!(
                AudioExecutionMode::resolve_with_codec(
                    format,
                    Some(codec),
                    HardwareBackend::Gpu(GpuBackend::Nvenc)
                ),
                Err(DomainError::UnsupportedAudioPipeline { .. })
            ));
        }
        assert_eq!(
            AudioExecutionMode::resolve(AudioOutputFormat::Ogg, HardwareBackend::Cpu)?,
            AudioExecutionMode::CpuOgg
        );

        let workers = NonZeroUsize::MIN;
        let plan = AudioBatchPlan::with_options(
            vec![item("a")],
            AudioExecutionMode::CpuOggFlac,
            workers,
            AudioEncodingOptions::for_codec(
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Flac,
                AudioRateControl::Default,
                None,
                None,
                Some(AudioBitDepth::Int24),
            )?,
        )
        .unwrap_or_else(|_| unreachable!());
        assert_eq!(plan.target_codec(), AudioTargetCodec::Flac);

        let bad_options = AudioEncodingOptions::for_codec(
            AudioOutputFormat::M4a,
            AudioTargetCodec::Aac,
            AudioRateControl::Bitrate(
                AudioOptionBitrate::new(128_000).unwrap_or_else(|_| unreachable!()),
            ),
            None,
            None,
            None,
        )?;
        assert!(matches!(
            AudioBatchPlan::with_options(
                vec![item("a")],
                AudioExecutionMode::CpuM4aAlac,
                workers,
                bad_options,
            ),
            Err(DomainError::UnsupportedAudioRateControl { .. })
        ));
        Ok(())
    }

    #[test]
    fn container_codec_matrix_is_symmetric_across_all_pairs() {
        for format in AudioOutputFormat::ALL {
            for codec in AudioTargetCodec::ALL {
                let forward = format.supported_codecs().contains(&codec);
                let reverse = codec.supported_formats().contains(&format);
                let resolved = format.resolve_codec(Some(codec)).is_ok();
                assert_eq!(
                    forward, reverse,
                    "{format}/{codec}: forward and reverse lists disagree"
                );
                assert_eq!(
                    forward, resolved,
                    "{format}/{codec}: resolve_codec disagrees with the matrix"
                );
            }
        }
    }

    #[test]
    fn unsupported_container_codec_error_is_bidirectional() {
        let cases: &[(AudioOutputFormat, AudioTargetCodec, &[&str])] = &[
            (
                AudioOutputFormat::M4a,
                AudioTargetCodec::Mp3,
                &["aac", "alac", "mp3"],
            ),
            (
                AudioOutputFormat::Ogg,
                AudioTargetCodec::Aac,
                &["vorbis", "opus", "flac", "aac", "m4a"],
            ),
        ];
        for &(format, codec, expected) in cases {
            let err = format
                .resolve_codec(Some(codec))
                .err()
                .ok_or(DomainError::EmptyField("audio.codec"))
                .unwrap_or_else(|_| unreachable!());
            let text = err.to_string();
            for &needle in expected {
                assert!(text.contains(needle), "{text}");
            }
        }
        assert_eq!(AudioOutputFormat::Alac.cli_name(), "alac");
        assert_eq!(AudioOutputFormat::Alac.container_label(), "ALAC");
        assert_eq!(AudioOutputFormat::M4a.container_label(), "M4A");
        assert_eq!(AudioTargetCodec::Alac.cli_name(), "alac");
    }
}
