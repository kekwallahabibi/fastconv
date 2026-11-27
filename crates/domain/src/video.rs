use std::fmt;
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[cfg(feature = "serde")]
use serde::Serialize;

use crate::{DomainError, GpuBackend, HardwareBackend, PathRole, u64_to_f64};

/// Video container formats supported for remuxing and transcoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum VideoContainerFormat {
    Mp4,
    Mkv,
    Webm,
    Mov,
    Avi,
    Ts,
    Gif,
    Webp,
}

impl VideoContainerFormat {
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Mkv => "mkv",
            Self::Webm => "webm",
            Self::Mov => "mov",
            Self::Avi => "avi",
            Self::Ts => "ts",
            Self::Gif => "gif",
            Self::Webp => "webp",
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Mp4 => "MP4",
            Self::Mkv => "MKV",
            Self::Webm => "WebM",
            Self::Mov => "MOV",
            Self::Avi => "AVI",
            Self::Ts => "TS",
            Self::Gif => "GIF",
            Self::Webp => "WebP",
        }
    }

    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "mp4" | "m4v" => Some(Self::Mp4),
            "mov" | "qt" => Some(Self::Mov),
            "mkv" => Some(Self::Mkv),
            "webm" => Some(Self::Webm),
            "avi" => Some(Self::Avi),
            "ts" | "m2ts" | "mts" => Some(Self::Ts),
            "gif" => Some(Self::Gif),
            "webp" => Some(Self::Webp),
            _ => None,
        }
    }

    /// Whether this container can carry a stream copy (no re-encoding).
    #[must_use]
    pub const fn supports_remux(self) -> bool {
        !matches!(self, Self::Gif | Self::Webp)
    }

    /// Whether this container can carry an audio track.
    #[must_use]
    pub const fn supports_audio(self) -> bool {
        !matches!(self, Self::Gif | Self::Webp)
    }

    #[must_use]
    pub const fn default_video_codec(self) -> TargetVideoCodec {
        match self {
            Self::Webm => TargetVideoCodec::Vp9,
            Self::Gif => TargetVideoCodec::Gif,
            Self::Webp => TargetVideoCodec::Webp,
            Self::Mp4 | Self::Mkv | Self::Mov | Self::Avi | Self::Ts => TargetVideoCodec::H264,
        }
    }

    /// `None` for animation containers, which have no audio track.
    #[must_use]
    pub const fn default_audio_codec(self) -> Option<TargetAudioCodec> {
        match self {
            Self::Gif | Self::Webp => None,
            Self::Webm => Some(TargetAudioCodec::Opus),
            Self::Mp4 | Self::Mkv | Self::Mov | Self::Avi | Self::Ts => Some(TargetAudioCodec::Aac),
        }
    }
}

impl fmt::Display for VideoContainerFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Video/audio codecs that fastconv recognises in containers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum VideoCodec {
    H264,
    Hevc,
    Av1,
    Vp9,
    Mjpeg,
    Aac,
    Opus,
    Mp3,
    Vorbis,
    Flac,
    Pcm,
    TimedText,
    Gif,
    Webp,
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
            Self::Mjpeg => "MJPEG",
            Self::Aac => "AAC",
            Self::Opus => "Opus",
            Self::Mp3 => "MP3",
            Self::Vorbis => "Vorbis",
            Self::Flac => "FLAC",
            Self::Pcm => "PCM",
            Self::TimedText => "Timed text",
            Self::Gif => "GIF",
            Self::Webp => "WebP",
            Self::Unknown => "unknown",
        }
    }

    /// Whether this codec can be stream-copied into `container`.
    #[must_use]
    pub const fn supported_in(self, container: VideoContainerFormat) -> bool {
        match container {
            VideoContainerFormat::Mp4 => matches!(
                self,
                Self::H264
                    | Self::Hevc
                    | Self::Av1
                    | Self::Vp9
                    | Self::Aac
                    | Self::Opus
                    | Self::Mp3
                    | Self::Flac
                    | Self::TimedText
            ),
            // QuickTime MOV has a v2 `lpcm` sample entry; ISO MP4 does not.
            VideoContainerFormat::Mov => matches!(
                self,
                Self::H264
                    | Self::Hevc
                    | Self::Av1
                    | Self::Vp9
                    | Self::Mjpeg
                    | Self::Aac
                    | Self::Opus
                    | Self::Mp3
                    | Self::Flac
                    | Self::Pcm
                    | Self::TimedText
            ),
            VideoContainerFormat::Mkv => matches!(
                self,
                Self::H264
                    | Self::Hevc
                    | Self::Av1
                    | Self::Vp9
                    | Self::Mjpeg
                    | Self::Aac
                    | Self::Opus
                    | Self::Mp3
                    | Self::Vorbis
                    | Self::Flac
                    | Self::Pcm
                    | Self::TimedText
            ),
            VideoContainerFormat::Webm => {
                matches!(self, Self::Vp9 | Self::Av1 | Self::Opus | Self::Vorbis)
            }
            VideoContainerFormat::Avi => matches!(
                self,
                Self::H264
                    | Self::Hevc
                    | Self::Av1
                    | Self::Vp9
                    | Self::Mjpeg
                    | Self::Aac
                    | Self::Opus
                    | Self::Mp3
                    | Self::Pcm
            ),
            // Standard MPEG-TS carries no VP9 bitstream mapping.
            VideoContainerFormat::Ts => matches!(
                self,
                Self::H264 | Self::Hevc | Self::Av1 | Self::Aac | Self::Opus | Self::Mp3
            ),
            VideoContainerFormat::Gif => matches!(self, Self::Gif),
            VideoContainerFormat::Webp => matches!(self, Self::Webp),
        }
    }

    #[must_use]
    pub const fn is_video(self) -> bool {
        matches!(
            self,
            Self::H264 | Self::Hevc | Self::Av1 | Self::Vp9 | Self::Mjpeg | Self::Gif | Self::Webp
        )
    }

    #[must_use]
    pub const fn is_audio(self) -> bool {
        matches!(
            self,
            Self::Aac | Self::Opus | Self::Mp3 | Self::Vorbis | Self::Flac | Self::Pcm
        )
    }

    /// Whether NVDEC can decode this bitstream on a Turing-or-newer GPU.
    ///
    /// AV1 is deliberately excluded: this host's RTX 2060 (Turing) has no
    /// AV1 decode block, and GIF/WebP are image sequences, not NVDEC inputs.
    #[must_use]
    pub const fn nvdec_decodable(self) -> bool {
        matches!(self, Self::H264 | Self::Hevc | Self::Vp9)
    }
}

impl fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Video codec selected for re-encoding, distinct from probed bitstream codecs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum TargetVideoCodec {
    H264,
    Hevc,
    Av1,
    Vp9,
    Mjpeg,
    Gif,
    Webp,
}

impl TargetVideoCodec {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::Hevc => "HEVC",
            Self::Av1 => "AV1",
            Self::Vp9 => "VP9",
            Self::Mjpeg => "MJPEG",
            Self::Gif => "GIF",
            Self::Webp => "WebP",
        }
    }

    #[must_use]
    pub const fn as_video_codec(self) -> VideoCodec {
        match self {
            Self::H264 => VideoCodec::H264,
            Self::Hevc => VideoCodec::Hevc,
            Self::Av1 => VideoCodec::Av1,
            Self::Vp9 => VideoCodec::Vp9,
            Self::Mjpeg => VideoCodec::Mjpeg,
            Self::Gif => VideoCodec::Gif,
            Self::Webp => VideoCodec::Webp,
        }
    }

    #[must_use]
    pub const fn supported_in(self, container: VideoContainerFormat) -> bool {
        self.as_video_codec().supported_in(container)
    }

    #[must_use]
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "h264" | "avc" | "x264" => Some(Self::H264),
            "hevc" | "h265" | "x265" => Some(Self::Hevc),
            "av1" => Some(Self::Av1),
            "vp9" => Some(Self::Vp9),
            "mjpeg" | "mjpg" => Some(Self::Mjpeg),
            "gif" => Some(Self::Gif),
            "webp" => Some(Self::Webp),
            _ => None,
        }
    }
}

impl fmt::Display for TargetVideoCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Audio codec selected for re-encoding inside a video container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum TargetAudioCodec {
    Aac,
    Opus,
    Mp3,
    Vorbis,
    Flac,
    Pcm,
}

impl TargetAudioCodec {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Aac => "AAC",
            Self::Opus => "Opus",
            Self::Mp3 => "MP3",
            Self::Vorbis => "Vorbis",
            Self::Flac => "FLAC",
            Self::Pcm => "PCM",
        }
    }

    #[must_use]
    pub const fn as_video_codec(self) -> VideoCodec {
        match self {
            Self::Aac => VideoCodec::Aac,
            Self::Opus => VideoCodec::Opus,
            Self::Mp3 => VideoCodec::Mp3,
            Self::Vorbis => VideoCodec::Vorbis,
            Self::Flac => VideoCodec::Flac,
            Self::Pcm => VideoCodec::Pcm,
        }
    }

    #[must_use]
    pub const fn supported_in(self, container: VideoContainerFormat) -> bool {
        self.as_video_codec().supported_in(container)
    }

    #[must_use]
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "aac" => Some(Self::Aac),
            "opus" => Some(Self::Opus),
            "mp3" => Some(Self::Mp3),
            "vorbis" => Some(Self::Vorbis),
            "flac" => Some(Self::Flac),
            "pcm" => Some(Self::Pcm),
            _ => None,
        }
    }
}

impl fmt::Display for TargetAudioCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Validated lossy video quality factor in `0..=100`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct VideoQuality(u8);

impl VideoQuality {
    pub const MAX: u8 = 100;
    pub const DEFAULT: Self = Self(80);

    /// Creates a validated [`VideoQuality`] in `0..=100`.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidVideoQuality`] when `value > 100`.
    pub const fn new(value: u8) -> Result<Self, DomainError> {
        if value > Self::MAX {
            return Err(DomainError::InvalidVideoQuality {
                value,
                max: Self::MAX,
            });
        }
        Ok(Self(value))
    }

    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

impl fmt::Display for VideoQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Parses `500k`, `2.5M` or a raw bit count into bits per second.
///
/// Integer-only on purpose: bitrates are counts of bits, and a float parse
/// would silently round values that the codec configs store as integers.
fn parse_bitrate(raw: &str) -> Result<u32, DomainError> {
    let invalid = || DomainError::InvalidBitrateSyntax {
        raw: raw.to_owned(),
    };
    let trimmed = raw.trim();
    let (digits, divisor) = if let Some(rest) = trimmed
        .strip_suffix('k')
        .or_else(|| trimmed.strip_suffix('K'))
    {
        (rest, 1_000_u64)
    } else if let Some(rest) = trimmed
        .strip_suffix('m')
        .or_else(|| trimmed.strip_suffix('M'))
    {
        // The fractional part is already scaled to millionths of a whole
        // Mbps unit, so no further division is needed for `M`.
        (rest, 1_u64)
    } else {
        (trimmed, 1_000_000_u64)
    };
    if digits.is_empty() {
        return Err(invalid());
    }
    let (whole, fraction) = match digits.split_once('.') {
        Some((whole, fraction)) if !fraction.is_empty() => (whole, Some(fraction)),
        Some(_) => return Err(invalid()),
        None => (digits, None),
    };
    if whole.is_empty() || !whole.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid());
    }
    let whole = whole.parse::<u64>().map_err(|_| invalid())?;
    let fraction = match fraction {
        None => 0_u64,
        Some(fraction) => {
            if fraction.len() > 6 || !fraction.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(invalid());
            }
            let digits = fraction.parse::<u32>().map_err(|_| invalid())?;
            u64::from(digits)
                .checked_mul(10_u64.pow(u32::try_from(6 - fraction.len()).unwrap_or(0)))
                .ok_or_else(invalid)?
        }
    };
    // `micro` holds the requested rate scaled by 1_000_000; the divisor turns
    // that into bps for a raw number, kbps or Mbps input respectively. Reject
    // non-integral bps rather than silently rounding.
    let micro = whole
        .checked_mul(1_000_000)
        .and_then(|scaled| scaled.checked_add(fraction))
        .ok_or_else(invalid)?;
    if !micro.is_multiple_of(divisor) {
        return Err(invalid());
    }
    let bps = micro / divisor;
    u32::try_from(bps).map_err(|_| invalid())
}

const fn bitrate_display(bps: u32) -> (u32, &'static str) {
    if bps.is_multiple_of(1_000) {
        (bps / 1_000, "k")
    } else {
        (bps, "")
    }
}

/// Validated video bitrate in bits per second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct VideoBitrate(u32);

impl VideoBitrate {
    pub const MIN_BPS: u32 = 50_000;
    pub const MAX_BPS: u32 = 500_000_000;

    /// Creates a validated bitrate in `50_000..=500_000_000` bps.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidVideoBitrate`] outside that range.
    pub fn new(bps: u32) -> Result<Self, DomainError> {
        if !(Self::MIN_BPS..=Self::MAX_BPS).contains(&bps) {
            return Err(DomainError::InvalidVideoBitrate {
                value: bps,
                min: Self::MIN_BPS,
                max: Self::MAX_BPS,
            });
        }
        Ok(Self(bps))
    }

    /// Parses `500k`, `2000k`, `5M`, `2.5M` or `5000000` into bps.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidBitrateSyntax`] for malformed input and
    /// [`DomainError::InvalidVideoBitrate`] for an out-of-range rate.
    pub fn parse(raw: &str) -> Result<Self, DomainError> {
        Self::new(parse_bitrate(raw)?)
    }

    #[must_use]
    pub const fn bps(self) -> u32 {
        self.0
    }
}

impl fmt::Display for VideoBitrate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (value, suffix) = bitrate_display(self.0);
        write!(f, "{value}{suffix}")
    }
}

/// Validated audio bitrate in bits per second. Codec-specific limits are
/// applied by [`AudioEncodeConfig::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct AudioBitrate(u32);

impl AudioBitrate {
    pub const MIN_BPS: u32 = 1_000;
    pub const MAX_BPS: u32 = 1_000_000_000;

    /// Creates a validated bitrate in `1_000..=1_000_000_000` bps.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidAudioBitrate`] outside that range.
    pub fn new(bps: u32) -> Result<Self, DomainError> {
        if !(Self::MIN_BPS..=Self::MAX_BPS).contains(&bps) {
            return Err(DomainError::InvalidAudioBitrate {
                value: bps,
                min: Self::MIN_BPS,
                max: Self::MAX_BPS,
            });
        }
        Ok(Self(bps))
    }

    /// Parses `128k`, `1.5M` or a raw bit count into bps.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidBitrateSyntax`] for malformed input and
    /// [`DomainError::InvalidAudioBitrate`] for an out-of-range rate.
    pub fn parse(raw: &str) -> Result<Self, DomainError> {
        Self::new(parse_bitrate(raw)?)
    }

    #[must_use]
    pub const fn bps(self) -> u32 {
        self.0
    }
}

impl fmt::Display for AudioBitrate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (value, suffix) = bitrate_display(self.0);
        write!(f, "{value}{suffix}")
    }
}

/// Validated audio sample rate in Hz.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct AudioSampleRate(NonZeroU32);

impl AudioSampleRate {
    /// Creates a validated, non-zero sample rate.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidAudioSampleRate`] when `hz` is zero.
    pub fn new(hz: u32) -> Result<Self, DomainError> {
        NonZeroU32::new(hz)
            .map(Self)
            .ok_or(DomainError::InvalidAudioSampleRate { value: hz })
    }

    #[must_use]
    pub const fn hz(self) -> u32 {
        self.0.get()
    }
}

impl fmt::Display for AudioSampleRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} Hz", self.0)
    }
}

/// Validated audio channel count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct AudioChannels(NonZeroU16);

impl AudioChannels {
    /// Creates a validated, non-zero channel count.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidAudioChannels`] when `count` is zero.
    pub fn new(count: u16) -> Result<Self, DomainError> {
        NonZeroU16::new(count)
            .map(Self)
            .ok_or(DomainError::InvalidAudioChannels { value: count })
    }

    #[must_use]
    pub const fn count(self) -> u16 {
        self.0.get()
    }
}

impl fmt::Display for AudioChannels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}ch", self.0)
    }
}

/// Optional, user-requested audio encoding settings for tracks inside a video
/// container. Nothing here is applied until it has been checked against a
/// concrete codec by [`AudioEncodeConfig::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct AudioEncodeOptions {
    bitrate: Option<AudioBitrate>,
    sample_rate: Option<AudioSampleRate>,
    channels: Option<AudioChannels>,
}

impl AudioEncodeOptions {
    #[must_use]
    pub const fn new(
        bitrate: Option<AudioBitrate>,
        sample_rate: Option<AudioSampleRate>,
        channels: Option<AudioChannels>,
    ) -> Self {
        Self {
            bitrate,
            sample_rate,
            channels,
        }
    }

    #[must_use]
    pub const fn bitrate(self) -> Option<AudioBitrate> {
        self.bitrate
    }

    #[must_use]
    pub const fn sample_rate(self) -> Option<AudioSampleRate> {
        self.sample_rate
    }

    #[must_use]
    pub const fn channels(self) -> Option<AudioChannels> {
        self.channels
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.bitrate.is_none() && self.sample_rate.is_none() && self.channels.is_none()
    }
}

/// Validated audio encoder settings for one target codec, with all unspecified
/// fields left as `None` so the pipeline can keep its existing defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct AudioEncodeConfig {
    codec: TargetAudioCodec,
    bitrate: Option<AudioBitrate>,
    sample_rate: Option<AudioSampleRate>,
    channels: Option<AudioChannels>,
}

impl AudioEncodeConfig {
    const AAC_RATES: [u32; 13] = [
        7_350, 8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 64_000,
        88_200, 96_000,
    ];
    const MP3_RATES: [u32; 9] = [
        8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000,
    ];

    /// Rate-dependent MP3 band: below 32 kHz the ceiling is 160 kbps, at or
    /// above it the floor is 32 kbps.
    const fn mp3_band(sample_rate_hz: u32) -> (u32, u32) {
        if sample_rate_hz >= 32_000 {
            (32_000, 320_000)
        } else {
            (8_000, 160_000)
        }
    }

    /// Validates `options` against the concrete codec's supported range.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedAudioEncodeOption`] when a requested
    /// value is outside the codec's bitrate/sample-rate/channel limits.
    pub fn new(codec: TargetAudioCodec, options: AudioEncodeOptions) -> Result<Self, DomainError> {
        let unsupported =
            |reason: String| DomainError::UnsupportedAudioEncodeOption { codec, reason };
        if let Some(rate) = options.bitrate() {
            // FLAC and PCM are lossless: a target bitrate has no meaning and
            // could only be silently ignored, so reject it at the boundary.
            if matches!(codec, TargetAudioCodec::Flac | TargetAudioCodec::Pcm) {
                return Err(unsupported(format!(
                    "{codec} is lossless and does not accept --abitrate"
                )));
            }
            let bps = rate.bps();
            let (min, max) = match codec {
                TargetAudioCodec::Aac => (16_000, 512_000),
                TargetAudioCodec::Opus => (6_000, 510_000),
                TargetAudioCodec::Mp3 => (8_000, 320_000),
                TargetAudioCodec::Vorbis => (32_000, 512_000),
                // Rejected above; zero-width range keeps the match total.
                TargetAudioCodec::Flac | TargetAudioCodec::Pcm => (1, 0),
            };
            if !(min..=max).contains(&bps) {
                return Err(unsupported(format!(
                    "bitrate {rate} is outside {codec}'s {min}..={max} bps range"
                )));
            }
            if codec == TargetAudioCodec::Mp3
                && let Some(sample_rate) = options.sample_rate()
            {
                let hz = sample_rate.hz();
                let (min, max) = Self::mp3_band(hz);
                if !(min..=max).contains(&bps) {
                    return Err(unsupported(format!(
                        "MP3 bitrate {rate} is outside {min}..={max} bps at {sample_rate}"
                    )));
                }
            }
        }
        if let Some(sample_rate) = options.sample_rate() {
            let hz = sample_rate.hz();
            let allowed = match codec {
                TargetAudioCodec::Aac => Self::AAC_RATES.contains(&hz),
                TargetAudioCodec::Opus => hz == 48_000,
                TargetAudioCodec::Mp3 => Self::MP3_RATES.contains(&hz),
                TargetAudioCodec::Vorbis => (8_000..=192_000).contains(&hz),
                TargetAudioCodec::Flac => (1..=655_350).contains(&hz),
                TargetAudioCodec::Pcm => (1..=384_000).contains(&hz),
            };
            if !allowed {
                return Err(unsupported(format!(
                    "sample rate {sample_rate} is not supported by {codec}"
                )));
            }
        }
        if let Some(channels) = options.channels() {
            let count = channels.count();
            let allowed = match codec {
                TargetAudioCodec::Aac | TargetAudioCodec::Vorbis => {
                    (1..=6).contains(&count) && count != 4
                }
                TargetAudioCodec::Opus | TargetAudioCodec::Mp3 => (1..=2).contains(&count),
                TargetAudioCodec::Flac | TargetAudioCodec::Pcm => (1..=8).contains(&count),
            };
            if !allowed {
                return Err(unsupported(format!(
                    "channel count {channels} is not supported by {codec}"
                )));
            }
        }
        Ok(Self {
            codec,
            bitrate: options.bitrate(),
            sample_rate: options.sample_rate(),
            channels: options.channels(),
        })
    }

    /// Validates the effective MP3 bitrate/sample-rate pair after the worker
    /// has resolved its defaults. [`Self::new`] can only inspect values the
    /// caller supplied; an omitted field is replaced by a codec default that
    /// could otherwise reach the encoder out of band.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedAudioEncodeOption`] when the effective
    /// pair is outside the rate-dependent MP3 band. Other codecs are unaffected.
    pub fn validate_effective_mp3(
        self,
        bitrate_bps: u32,
        sample_rate_hz: u32,
    ) -> Result<(), DomainError> {
        if self.codec != TargetAudioCodec::Mp3 {
            return Ok(());
        }
        let (min, max) = Self::mp3_band(sample_rate_hz);
        if !(min..=max).contains(&bitrate_bps) {
            return Err(DomainError::UnsupportedAudioEncodeOption {
                codec: self.codec,
                reason: format!(
                    "MP3 bitrate {bitrate_bps} bps is outside {min}..={max} bps at {sample_rate_hz} Hz"
                ),
            });
        }
        Ok(())
    }

    /// Applies no overrides; the pipeline keeps its per-codec defaults.
    #[must_use]
    pub const fn default_for(codec: TargetAudioCodec) -> Self {
        Self {
            codec,
            bitrate: None,
            sample_rate: None,
            channels: None,
        }
    }

    #[must_use]
    pub const fn codec(self) -> TargetAudioCodec {
        self.codec
    }

    #[must_use]
    pub const fn bitrate(self) -> Option<AudioBitrate> {
        self.bitrate
    }

    #[must_use]
    pub const fn sample_rate(self) -> Option<AudioSampleRate> {
        self.sample_rate
    }

    #[must_use]
    pub const fn channels(self) -> Option<AudioChannels> {
        self.channels
    }
}

/// Rejects rate control on a stream-copy request before any encoder is built.
/// `None` means the user did not pass a rate-control flag.
///
/// # Errors
///
/// Returns [`DomainError::ConflictingVideoRateControl`] for `--quality` or
/// `--vbitrate` combined with video stream copy.
pub fn resolve_video_rate_control(
    copy: bool,
    requested: Option<VideoRateControl>,
) -> Result<VideoRateControl, DomainError> {
    if copy && requested.is_some() {
        return Err(DomainError::ConflictingVideoRateControl {
            reason: "video stream copy cannot re-encode; omit --quality and --vbitrate".to_owned(),
        });
    }
    Ok(requested.unwrap_or(VideoRateControl::DEFAULT))
}

/// Rejects audio overrides on a copied or dropped track.
///
/// # Errors
///
/// Returns [`DomainError::ConflictingAudioEncodeOptions`] when options were
/// supplied for a track that is not encoded.
pub fn resolve_audio_encode_options(
    copy_or_drop: bool,
    options: AudioEncodeOptions,
) -> Result<AudioEncodeOptions, DomainError> {
    if copy_or_drop && !options.is_empty() {
        return Err(DomainError::ConflictingAudioEncodeOptions {
            reason: "--abitrate, --asample-rate and --achannels require audio re-encoding"
                .to_owned(),
        });
    }
    Ok(options)
}

/// Animation containers have no audio track, so audio overrides are illegal.
///
/// # Errors
///
/// Returns [`DomainError::ConflictingAudioEncodeOptions`] for a non-empty
/// request against a container that cannot carry audio.
pub fn validate_audio_options_for_container(
    options: AudioEncodeOptions,
    container: VideoContainerFormat,
) -> Result<(), DomainError> {
    if !container.supports_audio() && !options.is_empty() {
        return Err(DomainError::ConflictingAudioEncodeOptions {
            reason: format!("{container} containers have no audio track"),
        });
    }
    Ok(())
}

/// Decode backend requested at the CLI/system boundary.
///
/// `Auto` is resolved exactly once in `fastconv-probe` from the probed source
/// codec and the selected encoder backend. Forced `Nvdec` never silently
/// degrades to CPU decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum HardwareDecodeBackend {
    Auto,
    Cpu,
    Nvdec,
}

impl HardwareDecodeBackend {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Nvdec => "nvdec",
        }
    }

    #[must_use]
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "cpu" | "software" => Some(Self::Cpu),
            "nvdec" | "cuda" | "gpu" => Some(Self::Nvdec),
            _ => None,
        }
    }

    /// Resolves the user request into the concrete decode path that the
    /// pipeline must execute.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedVideoPipeline`] when `Nvdec` was
    /// forced but the encoder backend is not NVENC, or the source codec has
    /// no NVDEC implementation. `Auto` picks NVDEC for hardware-decodable
    /// codecs on NVENC and explicitly picks CPU decode otherwise.
    pub fn resolve(
        self,
        source_codec: VideoCodec,
        encode_backend: HardwareBackend,
    ) -> Result<VideoDecodeMode, DomainError> {
        let nvenc = matches!(encode_backend, HardwareBackend::Gpu(GpuBackend::Nvenc));
        match (self, nvenc, source_codec.nvdec_decodable()) {
            (Self::Nvdec | Self::Auto, true, true) => Ok(VideoDecodeMode::Nvdec),
            (Self::Nvdec, false, _) => Err(DomainError::UnsupportedVideoPipeline {
                codec: source_codec,
                backend: encode_backend,
                reason: "--hwdecode nvdec requires --backend nvenc; no silent CPU decode fallback"
                    .to_owned(),
            }),
            (Self::Nvdec, true, false) => Err(DomainError::UnsupportedVideoPipeline {
                codec: source_codec,
                backend: encode_backend,
                reason: format!(
                    "NVDEC cannot decode {source_codec}; use --hwdecode cpu or --hwdecode auto"
                ),
            }),
            (Self::Cpu | Self::Auto, _, _) => Ok(VideoDecodeMode::Cpu),
        }
    }
}

impl fmt::Display for HardwareDecodeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Concrete, validated decode path carried from probe into the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum VideoDecodeMode {
    Cpu,
    Nvdec,
}

impl VideoDecodeMode {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu decode",
            Self::Nvdec => "nvdec",
        }
    }

    #[must_use]
    pub const fn is_nvdec(self) -> bool {
        matches!(self, Self::Nvdec)
    }

    /// Exact executed hardware path, surfaced in reports and the CLI summary.
    #[must_use]
    pub const fn path(self, encode_backend: HardwareBackend) -> &'static str {
        match (self, encode_backend) {
            (Self::Nvdec, HardwareBackend::Gpu(GpuBackend::Nvenc)) => "gpu:nvdec+nvenc",
            (Self::Cpu, HardwareBackend::Gpu(GpuBackend::Nvenc)) => "gpu:nvenc (cpu decode)",
            (Self::Cpu, HardwareBackend::Cpu) => "cpu",
            // Forced NVDEC with a CPU encoder is rejected at the boundary; these
            // arms only exist to keep the match exhaustive without a panic.
            (Self::Nvdec, _) => "invalid:nvdec+cpu",
            (Self::Cpu, _) => "cpu (unsupported encoder)",
        }
    }
}

impl fmt::Display for VideoDecodeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Decode paths actually executed by a whole batch. Under `--hwdecode auto`
/// each item is resolved separately, so one batch may use both paths. `NoVideo`
/// means no item dispatched a video decode (e.g. an audio-only copy batch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum VideoDecodePaths {
    NoVideo,
    Cpu,
    Nvdec,
    Mixed {
        nvdec_files: NonZeroUsize,
        cpu_files: NonZeroUsize,
    },
}

impl VideoDecodePaths {
    /// Collapses per-item modes into one batch-level value. `None` means the
    /// batch had no items; mixed counts are non-zero by construction.
    #[must_use]
    pub fn from_modes(modes: impl IntoIterator<Item = VideoDecodeMode>) -> Option<Self> {
        let (mut nvdec, mut cpu) = (0_usize, 0_usize);
        for mode in modes {
            match mode {
                VideoDecodeMode::Nvdec => nvdec = nvdec.saturating_add(1),
                VideoDecodeMode::Cpu => cpu = cpu.saturating_add(1),
            }
        }
        match (NonZeroUsize::new(nvdec), NonZeroUsize::new(cpu)) {
            (Some(nvdec_files), Some(cpu_files)) => Some(Self::Mixed {
                nvdec_files,
                cpu_files,
            }),
            (Some(_), None) => Some(Self::Nvdec),
            (None, Some(_)) => Some(Self::Cpu),
            (None, None) => None,
        }
    }

    /// Exact executed paths for reports and the CLI summary. Homogeneous
    /// batches keep the single-path labels; mixed batches name both paths
    /// (counts remain available through [`Self::Mixed`]).
    #[must_use]
    pub const fn path(self, encode_backend: HardwareBackend) -> &'static str {
        match self {
            Self::NoVideo => "no video decoded",
            Self::Cpu => VideoDecodeMode::Cpu.path(encode_backend),
            Self::Nvdec => VideoDecodeMode::Nvdec.path(encode_backend),
            Self::Mixed { .. } => match encode_backend {
                HardwareBackend::Gpu(GpuBackend::Nvenc) => {
                    "mixed: gpu:nvdec+nvenc + gpu:nvenc (cpu decode)"
                }
                _ => "mixed: invalid:nvdec+cpu + cpu",
            },
        }
    }
}

impl From<VideoDecodeMode> for VideoDecodePaths {
    fn from(mode: VideoDecodeMode) -> Self {
        match mode {
            VideoDecodeMode::Nvdec => Self::Nvdec,
            VideoDecodeMode::Cpu => Self::Cpu,
        }
    }
}

/// Explicit rate-control choice for video encoders that support it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum VideoRateControl {
    Quality(VideoQuality),
    Bitrate(VideoBitrate),
}

impl VideoRateControl {
    pub const DEFAULT: Self = Self::Quality(VideoQuality::DEFAULT);

    #[must_use]
    pub const fn quality(self) -> Option<VideoQuality> {
        match self {
            Self::Quality(quality) => Some(quality),
            Self::Bitrate(_) => None,
        }
    }

    #[must_use]
    pub const fn bitrate(self) -> Option<VideoBitrate> {
        match self {
            Self::Bitrate(bitrate) => Some(bitrate),
            Self::Quality(_) => None,
        }
    }
}

impl fmt::Display for VideoRateControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Quality(quality) => write!(f, "quality {quality}"),
            Self::Bitrate(bitrate) => write!(f, "bitrate {bitrate}"),
        }
    }
}

/// Compile-time-legal video encoder choice: a codec/backend pair and its rate control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum VideoEncodeMode {
    CpuH264 { rate_control: VideoRateControl },
    CpuAv1 { rate_control: VideoRateControl },
    CpuVp9 { rate_control: VideoRateControl },
    CpuMjpeg { quality: VideoQuality },
    CpuGif { quality: VideoQuality },
    CpuWebp { quality: VideoQuality },
    NvencH264 { rate_control: VideoRateControl },
    NvencHevc { rate_control: VideoRateControl },
}

impl VideoEncodeMode {
    /// Resolves an encode mode without ever falling back to CPU silently.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::UnsupportedVideoPipeline`] when the requested codec
    /// has no encoder on the selected backend, when bitrate targeting is
    /// requested for GIF/WebP/MJPEG (which have no bitrate-aware encoder), or
    /// when a CPU-only codec is paired with a GPU backend.
    pub fn resolve(
        codec: TargetVideoCodec,
        rate_control: VideoRateControl,
        backend: HardwareBackend,
    ) -> Result<Self, DomainError> {
        if matches!(codec, TargetVideoCodec::Gif | TargetVideoCodec::Webp)
            && matches!(rate_control, VideoRateControl::Bitrate(_))
        {
            return Err(DomainError::UnsupportedVideoPipeline {
                codec: codec.as_video_codec(),
                backend,
                reason: format!(
                    "{codec} animation encoding supports only --quality, not --vbitrate"
                ),
            });
        }
        if matches!(codec, TargetVideoCodec::Mjpeg)
            && matches!(rate_control, VideoRateControl::Bitrate(_))
        {
            return Err(DomainError::UnsupportedVideoPipeline {
                codec: VideoCodec::Mjpeg,
                backend,
                reason: format!("{codec} encoding supports only --quality, not --vbitrate"),
            });
        }
        match (codec, backend) {
            (TargetVideoCodec::H264, HardwareBackend::Cpu) => Ok(Self::CpuH264 { rate_control }),
            (TargetVideoCodec::Av1, HardwareBackend::Cpu) => Ok(Self::CpuAv1 { rate_control }),
            (TargetVideoCodec::Vp9, HardwareBackend::Cpu) => Ok(Self::CpuVp9 { rate_control }),
            (TargetVideoCodec::Mjpeg, HardwareBackend::Cpu) => Ok(Self::CpuMjpeg {
                quality: rate_control.quality().unwrap_or(VideoQuality::DEFAULT),
            }),
            (TargetVideoCodec::Gif, HardwareBackend::Cpu) => Ok(Self::CpuGif {
                quality: rate_control.quality().unwrap_or(VideoQuality::DEFAULT),
            }),
            (TargetVideoCodec::Webp, HardwareBackend::Cpu) => Ok(Self::CpuWebp {
                quality: rate_control.quality().unwrap_or(VideoQuality::DEFAULT),
            }),
            (TargetVideoCodec::Gif | TargetVideoCodec::Webp, HardwareBackend::Gpu(_)) => {
                Err(DomainError::UnsupportedVideoPipeline {
                    codec: codec.as_video_codec(),
                    backend,
                    reason:
                        "animation encoding (GIF/WebP) runs on CPU only; no silent CPU fallback"
                            .to_owned(),
                })
            }
            (TargetVideoCodec::Mjpeg, HardwareBackend::Gpu(_)) => {
                Err(DomainError::UnsupportedVideoPipeline {
                    codec: VideoCodec::Mjpeg,
                    backend,
                    reason: format!("{codec} runs on CPU only; no silent CPU fallback"),
                })
            }
            (TargetVideoCodec::Hevc, HardwareBackend::Cpu) => {
                Err(DomainError::UnsupportedVideoPipeline {
                    codec: VideoCodec::Hevc,
                    backend,
                    reason:
                        "HEVC (H.265) encoding requires hardware acceleration via --backend nvenc"
                            .to_owned(),
                })
            }
            (TargetVideoCodec::H264, HardwareBackend::Gpu(GpuBackend::Nvenc)) => {
                Ok(Self::NvencH264 { rate_control })
            }
            (TargetVideoCodec::Hevc, HardwareBackend::Gpu(GpuBackend::Nvenc)) => {
                Ok(Self::NvencHevc { rate_control })
            }
            (
                TargetVideoCodec::Av1 | TargetVideoCodec::Vp9,
                HardwareBackend::Gpu(GpuBackend::Nvenc),
            ) => Err(DomainError::UnsupportedVideoPipeline {
                codec: codec.as_video_codec(),
                backend,
                reason: "NVENC supports H.264 and HEVC encoding; use --backend cpu for AV1/VP9"
                    .to_owned(),
            }),
            (_, HardwareBackend::Gpu(other)) => Err(DomainError::UnsupportedVideoPipeline {
                codec: codec.as_video_codec(),
                backend: HardwareBackend::Gpu(other),
                reason: format!(
                    "GPU backend '{other}' video encoding is not implemented; no silent CPU fallback"
                ),
            }),
        }
    }

    #[must_use]
    pub const fn codec(self) -> TargetVideoCodec {
        match self {
            Self::CpuH264 { .. } | Self::NvencH264 { .. } => TargetVideoCodec::H264,
            Self::CpuAv1 { .. } => TargetVideoCodec::Av1,
            Self::CpuVp9 { .. } => TargetVideoCodec::Vp9,
            Self::CpuMjpeg { .. } => TargetVideoCodec::Mjpeg,
            Self::CpuGif { .. } => TargetVideoCodec::Gif,
            Self::CpuWebp { .. } => TargetVideoCodec::Webp,
            Self::NvencHevc { .. } => TargetVideoCodec::Hevc,
        }
    }

    /// Rate control of a bitrate-aware encoder. GIF/WebP have no bitrate
    /// control, so they expose only [`Self::quality`].
    #[must_use]
    pub const fn rate_control(self) -> Option<VideoRateControl> {
        match self {
            Self::CpuH264 { rate_control }
            | Self::CpuAv1 { rate_control }
            | Self::CpuVp9 { rate_control }
            | Self::NvencH264 { rate_control }
            | Self::NvencHevc { rate_control } => Some(rate_control),
            Self::CpuMjpeg { .. } | Self::CpuGif { .. } | Self::CpuWebp { .. } => None,
        }
    }

    /// Quality of a quality-driven or animation mode. Bitrate-driven modes
    /// report [`VideoQuality::DEFAULT`] only so animation callers keep a valid
    /// quality; their encoder must use [`Self::rate_control`] instead.
    #[must_use]
    pub const fn quality(self) -> VideoQuality {
        match self {
            Self::CpuMjpeg { quality } | Self::CpuGif { quality } | Self::CpuWebp { quality } => {
                quality
            }
            Self::CpuH264 { rate_control }
            | Self::CpuAv1 { rate_control }
            | Self::CpuVp9 { rate_control }
            | Self::NvencH264 { rate_control }
            | Self::NvencHevc { rate_control } => match rate_control {
                VideoRateControl::Quality(quality) => quality,
                VideoRateControl::Bitrate(_) => VideoQuality::DEFAULT,
            },
        }
    }

    #[must_use]
    pub const fn backend(self) -> HardwareBackend {
        match self {
            Self::CpuH264 { .. }
            | Self::CpuAv1 { .. }
            | Self::CpuVp9 { .. }
            | Self::CpuMjpeg { .. }
            | Self::CpuGif { .. }
            | Self::CpuWebp { .. } => HardwareBackend::Cpu,
            Self::NvencH264 { .. } | Self::NvencHevc { .. } => {
                HardwareBackend::Gpu(GpuBackend::Nvenc)
            }
        }
    }
}

impl fmt::Display for VideoEncodeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.rate_control() {
            Some(rate_control) => {
                write!(f, "{} {} ({rate_control})", self.backend(), self.codec())
            }
            None => write!(
                f,
                "{} {} (quality {})",
                self.backend(),
                self.codec(),
                self.quality()
            ),
        }
    }
}

/// Per-track decision for a video stream during transcoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum VideoTrackAction {
    Copy,
    Encode(VideoEncodeMode),
}

impl VideoTrackAction {
    #[must_use]
    pub const fn backend(self) -> HardwareBackend {
        match self {
            Self::Copy => HardwareBackend::Cpu,
            Self::Encode(mode) => mode.backend(),
        }
    }
}

impl fmt::Display for VideoTrackAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Copy => f.write_str("copy"),
            Self::Encode(mode) => write!(f, "encode {mode}"),
        }
    }
}

/// Per-track decision for an audio stream during video transcoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum AudioTrackAction {
    Copy,
    Encode(AudioEncodeConfig),
    Drop,
}

impl AudioTrackAction {
    /// Target codec of an encode action; copied/dropped tracks have none.
    #[must_use]
    pub const fn codec(self) -> Option<TargetAudioCodec> {
        match self {
            Self::Encode(config) => Some(config.codec()),
            Self::Copy | Self::Drop => None,
        }
    }

    #[must_use]
    pub const fn config(self) -> Option<AudioEncodeConfig> {
        match self {
            Self::Encode(config) => Some(config),
            Self::Copy | Self::Drop => None,
        }
    }
}

impl fmt::Display for AudioTrackAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Copy => f.write_str("copy"),
            Self::Encode(config) => write!(f, "encode {}", config.codec()),
            Self::Drop => f.write_str("drop"),
        }
    }
}

/// Kind of a stream inside a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
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

/// Playback loop count signalled by an animation source. `Times(1)` plays once;
/// there is deliberately no `Times(0)` (that bit pattern means "forever").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub enum AnimationLoop {
    #[default]
    Infinite,
    Times(NonZeroU16),
}

/// Animation facts validated once at the probe boundary and carried unchanged
/// into execution, so the pipeline never re-parses container metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct AnimationMeta {
    loop_count: AnimationLoop,
    has_alpha: bool,
}

impl AnimationMeta {
    #[must_use]
    pub const fn new(loop_count: AnimationLoop, has_alpha: bool) -> Self {
        Self {
            loop_count,
            has_alpha,
        }
    }

    #[must_use]
    pub const fn loop_count(self) -> AnimationLoop {
        self.loop_count
    }

    #[must_use]
    pub const fn has_alpha(self) -> bool {
        self.has_alpha
    }
}

/// Probed stream metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
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
    animation: Option<AnimationMeta>,
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
            animation: None,
        }
    }

    /// Attaches animation metadata validated by the probe boundary.
    #[must_use]
    pub const fn with_animation(mut self, animation: AnimationMeta) -> Self {
        self.animation = Some(animation);
        self
    }

    #[must_use]
    pub const fn animation(&self) -> Option<AnimationMeta> {
        self.animation
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
            if !item.input_format().supports_remux() || !item.output_format().supports_remux() {
                return Err(DomainError::UnsupportedVideoPipeline {
                    codec: VideoCodec::Unknown,
                    backend: HardwareBackend::Cpu,
                    reason: "GIF and WebP are animation containers and cannot be stream-copied; use `fastconv video`"
                        .to_owned(),
                });
            }
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
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
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

/// A single transcode item with validated per-track actions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoTranscodeItem {
    source: PathBuf,
    target: PathBuf,
    input_format: VideoContainerFormat,
    output_format: VideoContainerFormat,
    streams: Vec<VideoStreamInfo>,
    video_action: VideoTrackAction,
    audio_action: AudioTrackAction,
    decode_mode: VideoDecodeMode,
}

impl VideoTranscodeItem {
    /// Creates a transcode item after rejecting codec/container mismatches.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyPath`] or [`DomainError::IdenticalSourceAndTarget`]
    /// for invalid paths, [`DomainError::UnsupportedVideoCodec`] when a copied or
    /// encoded track cannot be carried by `output_format`, and
    /// [`DomainError::UnsupportedVideoPipeline`] when `decode_mode` is NVDEC but
    /// the encode backend or source codec cannot support it.
    // Every validated input is passed explicitly at the probe boundary; the
    // decode-mode invariant belongs with the other track actions.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn new(
        source: PathBuf,
        target: PathBuf,
        input_format: VideoContainerFormat,
        output_format: VideoContainerFormat,
        streams: Vec<VideoStreamInfo>,
        video_action: VideoTrackAction,
        audio_action: AudioTrackAction,
        decode_mode: VideoDecodeMode,
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

        if decode_mode.is_nvdec() {
            if video_action.backend() != HardwareBackend::Gpu(GpuBackend::Nvenc) {
                return Err(DomainError::UnsupportedVideoPipeline {
                    codec: VideoCodec::Unknown,
                    backend: video_action.backend(),
                    reason: "NVDEC decode requires --backend nvenc; no silent CPU fallback"
                        .to_owned(),
                });
            }
            let video_streams: Vec<&VideoStreamInfo> = streams
                .iter()
                .filter(|s| s.kind() == VideoStreamKind::Video)
                .collect();
            let unsupported = video_streams
                .iter()
                .find(|s| !s.codec().nvdec_decodable())
                .map(|s| s.codec());
            if video_streams.is_empty() || unsupported.is_some() {
                return Err(DomainError::UnsupportedVideoPipeline {
                    codec: unsupported.unwrap_or(VideoCodec::Unknown),
                    backend: video_action.backend(),
                    reason: "NVDEC supports H.264, HEVC and VP9 sources only; use --hwdecode cpu"
                        .to_owned(),
                });
            }
        }

        match video_action {
            VideoTrackAction::Copy => {
                if !input_format.supports_remux() || !output_format.supports_remux() {
                    return Err(DomainError::UnsupportedVideoPipeline {
                        codec: VideoCodec::Unknown,
                        backend: HardwareBackend::Cpu,
                        reason: "stream copy requires remuxable containers; animation formats must be re-encoded"
                            .to_owned(),
                    });
                }
                for stream in streams
                    .iter()
                    .filter(|s| s.kind() == VideoStreamKind::Video)
                {
                    if !stream.codec().supported_in(output_format) {
                        return Err(DomainError::UnsupportedVideoCodec {
                            codec: stream.codec(),
                            container: output_format,
                        });
                    }
                }
            }
            VideoTrackAction::Encode(mode) => {
                if !mode.codec().supported_in(output_format) {
                    return Err(DomainError::UnsupportedVideoCodec {
                        codec: mode.codec().as_video_codec(),
                        container: output_format,
                    });
                }
            }
        }

        match audio_action {
            AudioTrackAction::Drop => {}
            AudioTrackAction::Copy => {
                if !output_format.supports_audio() {
                    return Err(DomainError::UnsupportedVideoPipeline {
                        codec: VideoCodec::Unknown,
                        backend: HardwareBackend::Cpu,
                        reason: "GIF and WebP containers have no audio track; drop audio streams for animation output"
                            .to_owned(),
                    });
                }
                for stream in streams
                    .iter()
                    .filter(|s| s.kind() == VideoStreamKind::Audio)
                {
                    if !stream.codec().supported_in(output_format) {
                        return Err(DomainError::UnsupportedVideoCodec {
                            codec: stream.codec(),
                            container: output_format,
                        });
                    }
                }
            }
            AudioTrackAction::Encode(config) => {
                let codec = config.codec();
                if !codec.supported_in(output_format) {
                    return Err(DomainError::UnsupportedVideoCodec {
                        codec: codec.as_video_codec(),
                        container: output_format,
                    });
                }
            }
        }

        if !output_format.supports_audio()
            && !streams.is_empty()
            && !streams.iter().any(|s| s.kind() == VideoStreamKind::Video)
        {
            return Err(DomainError::UnsupportedVideoPipeline {
                codec: VideoCodec::Unknown,
                backend: HardwareBackend::Cpu,
                reason: "GIF and WebP containers require at least one video stream".to_owned(),
            });
        }

        Ok(Self {
            source,
            target,
            input_format,
            output_format,
            streams,
            video_action,
            audio_action,
            decode_mode,
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

    #[must_use]
    pub const fn video_action(&self) -> VideoTrackAction {
        self.video_action
    }

    #[must_use]
    pub const fn audio_action(&self) -> AudioTrackAction {
        self.audio_action
    }

    /// Concrete decode path that the pipeline must execute for this item.
    #[must_use]
    pub const fn decode_mode(&self) -> VideoDecodeMode {
        self.decode_mode
    }
}

/// Validated batch plan for video transcoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoTranscodePlan {
    items: Vec<VideoTranscodeItem>,
    worker_threads: NonZeroUsize,
}

impl VideoTranscodePlan {
    /// Creates a validated transcode batch plan.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyBatch`] if `items` is empty, or
    /// [`DomainError::DuplicateTargetPath`] / [`DomainError::TargetOverwritesSource`].
    /// A batch that mixes encoder backends (e.g. CPU and NVENC items) is rejected
    /// with [`DomainError::UnsupportedVideoPipeline`]: the batch report carries a
    /// single encoder backend, so a mixed batch would mislabel the other items.
    pub fn new(
        items: Vec<VideoTranscodeItem>,
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
        // Stream-copy items have no encoder, so only Encode actions define the
        // batch's reported encoder backend. NVDEC and CPU decode stay valid
        // together as long as every encoder is NVENC.
        let mut encoder_backend: Option<HardwareBackend> = None;
        for item in &items {
            let VideoTrackAction::Encode(mode) = item.video_action() else {
                continue;
            };
            let backend = mode.backend();
            if let Some(existing) = encoder_backend
                && existing != backend
            {
                return Err(DomainError::UnsupportedVideoPipeline {
                    codec: VideoCodec::Unknown,
                    backend,
                    reason: format!(
                        "a batch cannot mix {existing} and {backend} video encoders; split the batch by backend"
                    ),
                });
            }
            encoder_backend = Some(backend);
        }
        Ok(Self {
            items,
            worker_threads,
        })
    }

    #[must_use]
    pub fn items(&self) -> &[VideoTranscodeItem] {
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

    #[must_use]
    pub fn backend(&self) -> HardwareBackend {
        self.items[0].video_action().backend()
    }

    /// Decode path of the first item. All items in a batch share the requested
    /// decode backend; mixed source codecs may resolve differently under `auto`,
    /// which the pipeline handles per item.
    #[must_use]
    pub fn decode_mode(&self) -> VideoDecodeMode {
        self.items[0].decode_mode()
    }
}

/// Report after a transcode batch.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct VideoTranscodeReport {
    files_converted: NonZeroUsize,
    frames_processed: u64,
    total_input_bytes: NonZeroU64,
    total_output_bytes: NonZeroU64,
    duration: Duration,
    worker_threads_used: NonZeroUsize,
    backend: HardwareBackend,
    hardware_device: Option<String>,
    decode_paths: VideoDecodePaths,
}

impl VideoTranscodeReport {
    // One argument per reported metric keeps the report construction explicit.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        files_converted: NonZeroUsize,
        frames_processed: u64,
        total_input_bytes: NonZeroU64,
        total_output_bytes: NonZeroU64,
        duration: Duration,
        worker_threads_used: NonZeroUsize,
        backend: HardwareBackend,
        hardware_device: Option<String>,
        decode_paths: impl Into<VideoDecodePaths>,
    ) -> Self {
        Self {
            files_converted,
            frames_processed,
            total_input_bytes,
            total_output_bytes,
            duration,
            worker_threads_used,
            backend,
            hardware_device,
            decode_paths: decode_paths.into(),
        }
    }

    #[must_use]
    pub const fn files_converted(&self) -> NonZeroUsize {
        self.files_converted
    }

    #[must_use]
    pub const fn frames_processed(&self) -> u64 {
        self.frames_processed
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
    pub fn hardware_device(&self) -> Option<&str> {
        self.hardware_device.as_deref()
    }

    /// Concrete decode paths executed across the batch.
    #[must_use]
    pub const fn decode_paths(&self) -> VideoDecodePaths {
        self.decode_paths
    }

    /// Exact executed hardware path for the CLI summary,
    /// e.g. `gpu:nvdec+nvenc`, `gpu:nvenc (cpu decode)` or a mixed-path label.
    #[must_use]
    pub const fn decode_path(&self) -> &'static str {
        self.decode_paths.path(self.backend)
    }

    #[must_use]
    pub fn megabytes_per_sec(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        (u64_to_f64(self.total_input_bytes.get()) / (1024.0 * 1024.0)) / secs
    }

    #[must_use]
    pub fn fps(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        u64_to_f64(self.frames_processed) / secs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    fn stream(codec: VideoCodec, kind: VideoStreamKind) -> VideoStreamInfo {
        VideoStreamInfo::new(codec, kind, None, None, None, None, 1, 1000, None, None)
    }

    fn item(
        source: &str,
        target: &str,
        input: VideoContainerFormat,
        output: VideoContainerFormat,
        streams: Vec<VideoStreamInfo>,
        video: VideoTrackAction,
        audio: AudioTrackAction,
    ) -> Result<VideoTranscodeItem, DomainError> {
        VideoTranscodeItem::new(
            PathBuf::from(source),
            PathBuf::from(target),
            input,
            output,
            streams,
            video,
            audio,
            VideoDecodeMode::Cpu,
        )
    }

    fn audio_encode(codec: TargetAudioCodec) -> AudioTrackAction {
        AudioTrackAction::Encode(AudioEncodeConfig::default_for(codec))
    }

    // Direct construction keeps test fixtures independent of the validation
    // under test; the child module may access the parent's private fields.
    fn ab(bps: u32) -> AudioBitrate {
        AudioBitrate(bps)
    }

    fn vb(bps: u32) -> VideoBitrate {
        VideoBitrate(bps)
    }

    fn cpu_encode(codec: TargetVideoCodec) -> VideoTrackAction {
        let rate_control = VideoRateControl::DEFAULT;
        VideoTrackAction::Encode(match codec {
            TargetVideoCodec::H264 => VideoEncodeMode::CpuH264 { rate_control },
            TargetVideoCodec::Av1 => VideoEncodeMode::CpuAv1 { rate_control },
            TargetVideoCodec::Vp9 => VideoEncodeMode::CpuVp9 { rate_control },
            TargetVideoCodec::Mjpeg => VideoEncodeMode::CpuMjpeg {
                quality: VideoQuality::DEFAULT,
            },
            TargetVideoCodec::Hevc => VideoEncodeMode::NvencHevc { rate_control },
            TargetVideoCodec::Gif => VideoEncodeMode::CpuGif {
                quality: VideoQuality::DEFAULT,
            },
            TargetVideoCodec::Webp => VideoEncodeMode::CpuWebp {
                quality: VideoQuality::DEFAULT,
            },
        })
    }

    #[test]
    fn codec_support_matrix() {
        let all = [
            VideoCodec::H264,
            VideoCodec::Hevc,
            VideoCodec::Av1,
            VideoCodec::Vp9,
            VideoCodec::Aac,
            VideoCodec::Opus,
            VideoCodec::Mp3,
            VideoCodec::Vorbis,
            VideoCodec::Flac,
            VideoCodec::Pcm,
            VideoCodec::TimedText,
        ];
        let webm = [
            VideoCodec::Vp9,
            VideoCodec::Av1,
            VideoCodec::Opus,
            VideoCodec::Vorbis,
        ];
        let ts = [
            VideoCodec::H264,
            VideoCodec::Hevc,
            VideoCodec::Av1,
            VideoCodec::Aac,
            VideoCodec::Opus,
            VideoCodec::Mp3,
        ];
        let rich_audio = [
            VideoCodec::H264,
            VideoCodec::Hevc,
            VideoCodec::Av1,
            VideoCodec::Vp9,
            VideoCodec::Aac,
            VideoCodec::Opus,
            VideoCodec::Mp3,
            VideoCodec::Pcm,
        ];
        let mp4 = [
            VideoCodec::H264,
            VideoCodec::Hevc,
            VideoCodec::Av1,
            VideoCodec::Vp9,
            VideoCodec::Aac,
            VideoCodec::Opus,
            VideoCodec::Mp3,
            VideoCodec::Flac,
            VideoCodec::TimedText,
        ];
        let mkv = [
            VideoCodec::H264,
            VideoCodec::Hevc,
            VideoCodec::Av1,
            VideoCodec::Vp9,
            VideoCodec::Aac,
            VideoCodec::Opus,
            VideoCodec::Mp3,
            VideoCodec::Vorbis,
            VideoCodec::Flac,
            VideoCodec::Pcm,
            VideoCodec::TimedText,
        ];

        // PCM is a QuickTime sample-entry codec: MOV carries it, ISO MP4 does not.
        assert!(VideoCodec::Pcm.supported_in(VideoContainerFormat::Mov));
        assert!(!VideoCodec::Pcm.supported_in(VideoContainerFormat::Mp4));
        assert!(!VideoCodec::Flac.supported_in(VideoContainerFormat::Avi));
        assert!(!VideoCodec::Flac.supported_in(VideoContainerFormat::Webm));
        for codec in all {
            assert_eq!(
                codec.supported_in(VideoContainerFormat::Webm),
                webm.contains(&codec)
            );
            assert_eq!(
                codec.supported_in(VideoContainerFormat::Ts),
                ts.contains(&codec)
            );
            assert_eq!(
                codec.supported_in(VideoContainerFormat::Avi),
                rich_audio.contains(&codec)
            );
            assert_eq!(
                codec.supported_in(VideoContainerFormat::Mp4),
                mp4.contains(&codec)
            );
            assert_eq!(
                codec.supported_in(VideoContainerFormat::Mov),
                mp4.contains(&codec) || codec == VideoCodec::Pcm
            );
            assert_eq!(
                codec.supported_in(VideoContainerFormat::Mkv),
                mkv.contains(&codec)
            );
        }
    }

    #[test]
    fn mjpeg_validation_matrix() {
        assert_eq!(VideoCodec::Mjpeg.label(), "MJPEG");
        assert!(VideoCodec::Mjpeg.is_video());
        assert!(!VideoCodec::Mjpeg.is_audio());
        assert!(!VideoCodec::Mjpeg.nvdec_decodable());

        for container in [
            VideoContainerFormat::Avi,
            VideoContainerFormat::Mov,
            VideoContainerFormat::Mkv,
        ] {
            assert!(
                VideoCodec::Mjpeg.supported_in(container),
                "MJPEG must be allowed in {container}"
            );
            assert!(TargetVideoCodec::Mjpeg.supported_in(container));
        }
        for container in [
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Webm,
            VideoContainerFormat::Ts,
            VideoContainerFormat::Gif,
            VideoContainerFormat::Webp,
        ] {
            assert!(
                !VideoCodec::Mjpeg.supported_in(container),
                "MJPEG must be rejected in {container}"
            );
            assert!(!TargetVideoCodec::Mjpeg.supported_in(container));
        }

        assert_eq!(TargetVideoCodec::Mjpeg.label(), "MJPEG");
        assert_eq!(TargetVideoCodec::Mjpeg.as_video_codec(), VideoCodec::Mjpeg);
        assert_eq!(
            TargetVideoCodec::from_name("mjpeg"),
            Some(TargetVideoCodec::Mjpeg)
        );
        assert_eq!(
            TargetVideoCodec::from_name("MJPG"),
            Some(TargetVideoCodec::Mjpeg)
        );
        assert_eq!(TargetVideoCodec::from_name("jpe"), None);
    }

    #[test]
    fn mjpeg_encode_mode_is_cpu_quality_only() {
        let quality = VideoQuality::new(55).unwrap_or(VideoQuality::DEFAULT);
        let mode = VideoEncodeMode::resolve(
            TargetVideoCodec::Mjpeg,
            VideoRateControl::Quality(quality),
            HardwareBackend::Cpu,
        );
        assert_eq!(mode, Ok(VideoEncodeMode::CpuMjpeg { quality }));
        let mode = mode.unwrap_or(VideoEncodeMode::CpuMjpeg {
            quality: VideoQuality::DEFAULT,
        });
        assert_eq!(mode.codec(), TargetVideoCodec::Mjpeg);
        assert_eq!(mode.quality(), quality);
        assert_eq!(mode.backend(), HardwareBackend::Cpu);
        assert_eq!(mode.rate_control(), None);

        let bitrate = vb(2_000_000);
        let result = VideoEncodeMode::resolve(
            TargetVideoCodec::Mjpeg,
            VideoRateControl::Bitrate(bitrate),
            HardwareBackend::Cpu,
        );
        assert!(result.is_err(), "MJPEG must reject --vbitrate");
        let err = result
            .err()
            .unwrap_or(DomainError::EmptyField("expected error"));
        assert!(err.to_string().contains("only --quality"));

        let result = VideoEncodeMode::resolve(
            TargetVideoCodec::Mjpeg,
            VideoRateControl::DEFAULT,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
        );
        assert!(result.is_err(), "MJPEG must reject NVENC");
        let err = result
            .err()
            .unwrap_or(DomainError::EmptyField("expected error"));
        assert!(err.to_string().contains("CPU only"));

        // Forced NVDEC is rejected for an MJPEG source at the domain boundary.
        let err = HardwareDecodeBackend::Nvdec
            .resolve(VideoCodec::Mjpeg, HardwareBackend::Gpu(GpuBackend::Nvenc));
        assert!(err.is_err());
        assert!(!VideoCodec::Mjpeg.nvdec_decodable());
    }

    #[test]
    fn codec_support_specific_rules() {
        // WebM is restricted to VP9/AV1 video and Opus/Vorbis audio.
        for codec in [
            VideoCodec::H264,
            VideoCodec::Hevc,
            VideoCodec::Aac,
            VideoCodec::Mp3,
        ] {
            assert!(!codec.supported_in(VideoContainerFormat::Webm));
        }
        // MPEG-TS has no VP9 mapping.
        assert!(!VideoCodec::Vp9.supported_in(VideoContainerFormat::Ts));
        for container in [
            VideoContainerFormat::Avi,
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mov,
            VideoContainerFormat::Mkv,
        ] {
            for codec in [
                VideoCodec::H264,
                VideoCodec::Hevc,
                VideoCodec::Av1,
                VideoCodec::Vp9,
                VideoCodec::Aac,
                VideoCodec::Opus,
                VideoCodec::Mp3,
            ] {
                assert!(codec.supported_in(container));
            }
        }
        assert!(!VideoCodec::Vorbis.supported_in(VideoContainerFormat::Mp4));
        assert!(!VideoCodec::TimedText.supported_in(VideoContainerFormat::Avi));
        assert!(!VideoCodec::Unknown.supported_in(VideoContainerFormat::Mkv));
        assert!(VideoCodec::H264.is_video());
        assert!(VideoCodec::Hevc.is_video());
        assert!(VideoCodec::Av1.is_video());
        assert!(VideoCodec::Vp9.is_video());
        assert!(VideoCodec::Gif.is_video());
        assert!(VideoCodec::Webp.is_video());
        assert!(VideoCodec::Aac.is_audio());
        assert!(VideoCodec::Opus.is_audio());
        assert!(VideoCodec::Mp3.is_audio());
        assert!(VideoCodec::Vorbis.is_audio());
        assert!(VideoCodec::Flac.is_audio());
        assert!(VideoCodec::Pcm.is_audio());
        assert!(!VideoCodec::Flac.is_video());
        assert!(!VideoCodec::Pcm.is_video());
        assert!(!VideoCodec::Flac.nvdec_decodable());
        assert!(!VideoCodec::Pcm.nvdec_decodable());
        assert!(!VideoCodec::H264.is_audio());
        assert!(!VideoCodec::Mp3.is_video());
    }

    #[test]
    fn animation_codecs_are_confined_to_their_own_container() {
        assert!(VideoCodec::Gif.supported_in(VideoContainerFormat::Gif));
        assert!(VideoCodec::Webp.supported_in(VideoContainerFormat::Webp));
        for container in [
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Webm,
            VideoContainerFormat::Mov,
            VideoContainerFormat::Avi,
            VideoContainerFormat::Ts,
        ] {
            assert!(!VideoCodec::Gif.supported_in(container), "{container}");
            assert!(!VideoCodec::Webp.supported_in(container), "{container}");
        }
        assert!(!VideoCodec::Gif.supported_in(VideoContainerFormat::Webp));
        assert!(!VideoCodec::Webp.supported_in(VideoContainerFormat::Gif));
        assert!(!VideoCodec::H264.supported_in(VideoContainerFormat::Gif));
        assert!(!VideoCodec::Aac.supported_in(VideoContainerFormat::Webp));
    }

    #[test]
    fn animation_containers_are_video_only_and_cpu_encoded() {
        let gif = VideoContainerFormat::Gif;
        let webp = VideoContainerFormat::Webp;
        assert_eq!(gif.extension(), "gif");
        assert_eq!(webp.extension(), "webp");
        assert_eq!(gif.label(), "GIF");
        assert_eq!(webp.label(), "WebP");
        assert!(!gif.supports_remux());
        assert!(!webp.supports_remux());
        assert!(!gif.supports_audio());
        assert!(!webp.supports_audio());
        for container in [
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Webm,
            VideoContainerFormat::Mov,
            VideoContainerFormat::Avi,
            VideoContainerFormat::Ts,
        ] {
            assert!(container.supports_remux(), "{container}");
            assert!(container.supports_audio(), "{container}");
        }
        assert_eq!(gif.default_video_codec(), TargetVideoCodec::Gif);
        assert_eq!(webp.default_video_codec(), TargetVideoCodec::Webp);
        assert_eq!(gif.default_audio_codec(), None);
        assert_eq!(webp.default_audio_codec(), None);
        assert_eq!(
            VideoContainerFormat::Mp4.default_audio_codec(),
            Some(TargetAudioCodec::Aac)
        );
        assert_eq!(
            VideoContainerFormat::Webm.default_audio_codec(),
            Some(TargetAudioCodec::Opus)
        );
        assert_eq!(AudioTrackAction::Drop.to_string(), "drop");
    }

    #[test]
    fn animation_meta_carries_validated_playback_facts() {
        let times = NonZeroU16::new(3).unwrap_or(NonZeroU16::MIN);
        let meta = AnimationMeta::new(AnimationLoop::Times(times), true);
        assert_eq!(meta.loop_count(), AnimationLoop::Times(times));
        assert!(meta.has_alpha());
        assert_eq!(
            AnimationMeta::default().loop_count(),
            AnimationLoop::Infinite
        );
        assert!(!AnimationMeta::default().has_alpha());
        let with_meta = stream(VideoCodec::Webp, VideoStreamKind::Video).with_animation(meta);
        assert_eq!(with_meta.animation(), Some(meta));
        assert_eq!(
            stream(VideoCodec::Webp, VideoStreamKind::Video).animation(),
            None
        );
    }

    #[test]
    fn from_extension_maps_all_aliases() {
        let cases = [
            ("mp4", VideoContainerFormat::Mp4),
            ("m4v", VideoContainerFormat::Mp4),
            ("mov", VideoContainerFormat::Mov),
            ("qt", VideoContainerFormat::Mov),
            ("mkv", VideoContainerFormat::Mkv),
            ("webm", VideoContainerFormat::Webm),
            ("avi", VideoContainerFormat::Avi),
            ("ts", VideoContainerFormat::Ts),
            ("m2ts", VideoContainerFormat::Ts),
            ("mts", VideoContainerFormat::Ts),
            ("gif", VideoContainerFormat::Gif),
            ("webp", VideoContainerFormat::Webp),
            ("MP4", VideoContainerFormat::Mp4),
            ("WEBM", VideoContainerFormat::Webm),
            ("GIF", VideoContainerFormat::Gif),
            ("WEBP", VideoContainerFormat::Webp),
        ];
        for (ext, expected) in cases {
            assert_eq!(
                VideoContainerFormat::from_extension(ext),
                Some(expected),
                "extension {ext}"
            );
        }
        assert_eq!(VideoContainerFormat::from_extension("flv"), None);
    }

    #[test]
    fn container_defaults_and_codec_names() {
        assert_eq!(VideoContainerFormat::Mp4.extension(), "mp4");
        assert_eq!(VideoContainerFormat::Mkv.extension(), "mkv");
        assert_eq!(VideoContainerFormat::Webm.extension(), "webm");
        assert_eq!(VideoContainerFormat::Mov.extension(), "mov");
        assert_eq!(VideoContainerFormat::Avi.extension(), "avi");
        assert_eq!(VideoContainerFormat::Ts.extension(), "ts");
        assert_eq!(VideoContainerFormat::Webm.label(), "WebM");
        assert_eq!(VideoContainerFormat::Ts.label(), "TS");
        assert_eq!(
            VideoContainerFormat::Webm.default_video_codec(),
            TargetVideoCodec::Vp9
        );
        assert_eq!(
            VideoContainerFormat::Mp4.default_video_codec(),
            TargetVideoCodec::H264
        );
        assert_eq!(
            VideoContainerFormat::Webm.default_audio_codec(),
            Some(TargetAudioCodec::Opus)
        );
        assert_eq!(
            VideoContainerFormat::Mp4.default_audio_codec(),
            Some(TargetAudioCodec::Aac)
        );
        assert_eq!(
            TargetVideoCodec::from_name("x264"),
            Some(TargetVideoCodec::H264)
        );
        assert_eq!(
            TargetVideoCodec::from_name("HEVC"),
            Some(TargetVideoCodec::Hevc)
        );
        assert_eq!(
            TargetVideoCodec::from_name("h265"),
            Some(TargetVideoCodec::Hevc)
        );
        assert_eq!(
            TargetVideoCodec::from_name("av1"),
            Some(TargetVideoCodec::Av1)
        );
        assert_eq!(
            TargetVideoCodec::from_name("vp9"),
            Some(TargetVideoCodec::Vp9)
        );
        assert_eq!(
            TargetVideoCodec::from_name("gif"),
            Some(TargetVideoCodec::Gif)
        );
        assert_eq!(
            TargetVideoCodec::from_name("WebP"),
            Some(TargetVideoCodec::Webp)
        );
        assert_eq!(TargetVideoCodec::from_name("vp8"), None);
        assert_eq!(
            TargetAudioCodec::from_name("aac"),
            Some(TargetAudioCodec::Aac)
        );
        assert_eq!(
            TargetAudioCodec::from_name("opus"),
            Some(TargetAudioCodec::Opus)
        );
        assert_eq!(
            TargetAudioCodec::from_name("mp3"),
            Some(TargetAudioCodec::Mp3)
        );
        assert_eq!(
            TargetAudioCodec::from_name("Vorbis"),
            Some(TargetAudioCodec::Vorbis)
        );
        assert_eq!(
            TargetAudioCodec::from_name("flac"),
            Some(TargetAudioCodec::Flac)
        );
        assert_eq!(
            TargetAudioCodec::from_name("PCM"),
            Some(TargetAudioCodec::Pcm)
        );
        assert!(TargetVideoCodec::Vp9.supported_in(VideoContainerFormat::Webm));
        assert!(!TargetVideoCodec::H264.supported_in(VideoContainerFormat::Webm));
        assert!(!TargetAudioCodec::Mp3.supported_in(VideoContainerFormat::Webm));
    }

    #[test]
    fn vorbis_is_supported_only_in_matroska_containers() {
        assert_eq!(TargetAudioCodec::Vorbis.label(), "Vorbis");
        assert_eq!(
            TargetAudioCodec::Vorbis.as_video_codec(),
            VideoCodec::Vorbis
        );
        assert_eq!(TargetAudioCodec::Vorbis.to_string(), "Vorbis");
        assert!(TargetAudioCodec::Vorbis.supported_in(VideoContainerFormat::Mkv));
        assert!(TargetAudioCodec::Vorbis.supported_in(VideoContainerFormat::Webm));
        for container in [
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mov,
            VideoContainerFormat::Avi,
            VideoContainerFormat::Ts,
            VideoContainerFormat::Gif,
            VideoContainerFormat::Webp,
        ] {
            assert!(
                !TargetAudioCodec::Vorbis.supported_in(container),
                "Vorbis must be rejected in {container}"
            );
        }
        // Defaults must remain unchanged.
        assert_eq!(
            VideoContainerFormat::Mkv.default_audio_codec(),
            Some(TargetAudioCodec::Aac)
        );
        assert_eq!(
            VideoContainerFormat::Webm.default_audio_codec(),
            Some(TargetAudioCodec::Opus)
        );
    }

    #[test]
    fn encode_mode_resolve_accepts_supported_pairs() -> Result<(), DomainError> {
        let q = VideoQuality::DEFAULT;
        let rc = VideoRateControl::Quality(q);
        assert_eq!(
            VideoEncodeMode::resolve(TargetVideoCodec::H264, rc, HardwareBackend::Cpu)?,
            VideoEncodeMode::CpuH264 { rate_control: rc }
        );
        assert_eq!(
            VideoEncodeMode::resolve(TargetVideoCodec::Av1, rc, HardwareBackend::Cpu)?,
            VideoEncodeMode::CpuAv1 { rate_control: rc }
        );
        assert_eq!(
            VideoEncodeMode::resolve(TargetVideoCodec::Vp9, rc, HardwareBackend::Cpu)?,
            VideoEncodeMode::CpuVp9 { rate_control: rc }
        );
        assert_eq!(
            VideoEncodeMode::resolve(TargetVideoCodec::Gif, rc, HardwareBackend::Cpu)?,
            VideoEncodeMode::CpuGif { quality: q }
        );
        assert_eq!(
            VideoEncodeMode::resolve(TargetVideoCodec::Webp, rc, HardwareBackend::Cpu)?,
            VideoEncodeMode::CpuWebp { quality: q }
        );
        let nvenc = HardwareBackend::Gpu(GpuBackend::Nvenc);
        assert_eq!(
            VideoEncodeMode::resolve(TargetVideoCodec::H264, rc, nvenc)?,
            VideoEncodeMode::NvencH264 { rate_control: rc }
        );
        assert_eq!(
            VideoEncodeMode::resolve(TargetVideoCodec::Hevc, rc, nvenc)?,
            VideoEncodeMode::NvencHevc { rate_control: rc }
        );
        assert_eq!(
            VideoEncodeMode::CpuH264 { rate_control: rc }.codec(),
            TargetVideoCodec::H264
        );
        assert_eq!(
            VideoEncodeMode::NvencHevc { rate_control: rc }.codec(),
            TargetVideoCodec::Hevc
        );
        assert_eq!(VideoEncodeMode::CpuAv1 { rate_control: rc }.quality(), q);
        assert_eq!(
            VideoEncodeMode::CpuGif { quality: q }.codec(),
            TargetVideoCodec::Gif
        );
        assert_eq!(
            VideoEncodeMode::CpuWebp { quality: q }.codec(),
            TargetVideoCodec::Webp
        );
        assert_eq!(VideoEncodeMode::CpuGif { quality: q }.quality(), q);
        assert_eq!(VideoEncodeMode::CpuWebp { quality: q }.quality(), q);
        assert_eq!(
            VideoEncodeMode::CpuVp9 { rate_control: rc }.backend(),
            HardwareBackend::Cpu
        );
        assert_eq!(
            VideoEncodeMode::CpuWebp { quality: q }.backend(),
            HardwareBackend::Cpu
        );
        assert_eq!(
            VideoEncodeMode::NvencH264 { rate_control: rc }.backend(),
            nvenc
        );
        assert_eq!(VideoTrackAction::Copy.backend(), HardwareBackend::Cpu);
        Ok(())
    }

    #[test]
    fn encode_mode_resolve_rejects_unsupported_pairs_without_fallback() {
        let q = VideoQuality::DEFAULT;
        let rc = VideoRateControl::Quality(q);
        assert_eq!(
            VideoEncodeMode::resolve(TargetVideoCodec::Hevc, rc, HardwareBackend::Cpu),
            Err(DomainError::UnsupportedVideoPipeline {
                codec: VideoCodec::Hevc,
                backend: HardwareBackend::Cpu,
                reason: "HEVC (H.265) encoding requires hardware acceleration via --backend nvenc"
                    .to_owned(),
            })
        );
        for codec in [TargetVideoCodec::Av1, TargetVideoCodec::Vp9] {
            assert_eq!(
                VideoEncodeMode::resolve(codec, rc, HardwareBackend::Gpu(GpuBackend::Nvenc)),
                Err(DomainError::UnsupportedVideoPipeline {
                    codec: codec.as_video_codec(),
                    backend: HardwareBackend::Gpu(GpuBackend::Nvenc),
                    reason: "NVENC supports H.264 and HEVC encoding; use --backend cpu for AV1/VP9"
                        .to_owned(),
                })
            );
        }
        for codec in [TargetVideoCodec::Gif, TargetVideoCodec::Webp] {
            for backend in [
                HardwareBackend::Gpu(GpuBackend::Nvenc),
                HardwareBackend::Gpu(GpuBackend::Vaapi),
            ] {
                assert_eq!(
                    VideoEncodeMode::resolve(codec, rc, backend),
                    Err(DomainError::UnsupportedVideoPipeline {
                        codec: codec.as_video_codec(),
                        backend,
                        reason:
                            "animation encoding (GIF/WebP) runs on CPU only; no silent CPU fallback"
                                .to_owned(),
                    })
                );
            }
        }
        for gpu in [
            GpuBackend::Vaapi,
            GpuBackend::Vulkan,
            GpuBackend::Qsv,
            GpuBackend::Amf,
            GpuBackend::VideoToolbox,
        ] {
            let backend = HardwareBackend::Gpu(gpu);
            assert_eq!(
                VideoEncodeMode::resolve(TargetVideoCodec::H264, rc, backend),
                Err(DomainError::UnsupportedVideoPipeline {
                    codec: VideoCodec::H264,
                    backend,
                    reason: format!(
                        "GPU backend '{gpu}' video encoding is not implemented; no silent CPU fallback"
                    ),
                })
            );
        }
    }

    #[test]
    fn video_quality_rejects_out_of_range() {
        assert_eq!(VideoQuality::MAX, 100);
        assert_eq!(VideoQuality::DEFAULT.get(), 80);
        assert_eq!(
            VideoQuality::new(101),
            Err(DomainError::InvalidVideoQuality {
                value: 101,
                max: 100
            })
        );
        assert_eq!(VideoQuality::new(100).map(VideoQuality::get), Ok(100));
    }

    #[test]
    fn transcode_item_rejects_webm_codec_mismatch() -> Result<(), DomainError> {
        let webm = VideoContainerFormat::Webm;
        let mp4 = VideoContainerFormat::Mp4;
        assert_eq!(
            item(
                "in.mp4",
                "out.webm",
                mp4,
                webm,
                Vec::new(),
                cpu_encode(TargetVideoCodec::H264),
                audio_encode(TargetAudioCodec::Opus)
            ),
            Err(DomainError::UnsupportedVideoCodec {
                codec: VideoCodec::H264,
                container: webm
            })
        );
        assert_eq!(
            item(
                "in.mp4",
                "out.webm",
                mp4,
                webm,
                vec![stream(VideoCodec::H264, VideoStreamKind::Video)],
                VideoTrackAction::Copy,
                AudioTrackAction::Copy
            ),
            Err(DomainError::UnsupportedVideoCodec {
                codec: VideoCodec::H264,
                container: webm
            })
        );
        assert_eq!(
            item(
                "in.mp4",
                "out.webm",
                mp4,
                webm,
                Vec::new(),
                cpu_encode(TargetVideoCodec::Vp9),
                audio_encode(TargetAudioCodec::Aac)
            ),
            Err(DomainError::UnsupportedVideoCodec {
                codec: VideoCodec::Aac,
                container: webm
            })
        );
        assert_eq!(
            item(
                "in.mkv",
                "out.webm",
                VideoContainerFormat::Mkv,
                webm,
                vec![stream(VideoCodec::Aac, VideoStreamKind::Audio)],
                VideoTrackAction::Copy,
                AudioTrackAction::Copy
            ),
            Err(DomainError::UnsupportedVideoCodec {
                codec: VideoCodec::Aac,
                container: webm
            })
        );

        let valid = item(
            "in.mkv",
            "out.webm",
            VideoContainerFormat::Mkv,
            webm,
            Vec::new(),
            cpu_encode(TargetVideoCodec::Vp9),
            audio_encode(TargetAudioCodec::Opus),
        )?;
        assert_eq!(valid.output_format(), webm);
        let copied = item(
            "in.webm",
            "out2.webm",
            webm,
            webm,
            vec![
                stream(VideoCodec::Vp9, VideoStreamKind::Video),
                stream(VideoCodec::Opus, VideoStreamKind::Audio),
            ],
            VideoTrackAction::Copy,
            AudioTrackAction::Copy,
        )?;
        assert_eq!(copied.streams().len(), 2);
        Ok(())
    }

    #[test]
    fn transcode_item_rejects_vorbis_outside_matroska() -> Result<(), DomainError> {
        for (container, video_codec) in [
            (VideoContainerFormat::Mp4, TargetVideoCodec::H264),
            (VideoContainerFormat::Mov, TargetVideoCodec::H264),
            (VideoContainerFormat::Avi, TargetVideoCodec::H264),
            (VideoContainerFormat::Ts, TargetVideoCodec::H264),
            (VideoContainerFormat::Gif, TargetVideoCodec::Gif),
            (VideoContainerFormat::Webp, TargetVideoCodec::Webp),
        ] {
            assert_eq!(
                item(
                    "in.mkv",
                    "out.media",
                    VideoContainerFormat::Mkv,
                    container,
                    Vec::new(),
                    cpu_encode(video_codec),
                    audio_encode(TargetAudioCodec::Vorbis),
                ),
                Err(DomainError::UnsupportedVideoCodec {
                    codec: VideoCodec::Vorbis,
                    container,
                }),
                "Vorbis must be rejected in {container}"
            );
        }
        for (container, video_codec) in [
            (VideoContainerFormat::Mkv, TargetVideoCodec::H264),
            (VideoContainerFormat::Webm, TargetVideoCodec::Vp9),
        ] {
            let accepted = item(
                "in.mkv",
                "out.media",
                VideoContainerFormat::Mkv,
                container,
                Vec::new(),
                cpu_encode(video_codec),
                audio_encode(TargetAudioCodec::Vorbis),
            )?;
            assert_eq!(accepted.output_format(), container);
        }
        Ok(())
    }

    #[test]
    fn transcode_item_rejects_ts_vp9() -> Result<(), DomainError> {
        let ts = VideoContainerFormat::Ts;
        assert_eq!(
            item(
                "in.webm",
                "out.ts",
                VideoContainerFormat::Webm,
                ts,
                Vec::new(),
                cpu_encode(TargetVideoCodec::Vp9),
                audio_encode(TargetAudioCodec::Opus)
            ),
            Err(DomainError::UnsupportedVideoCodec {
                codec: VideoCodec::Vp9,
                container: ts
            })
        );
        assert_eq!(
            item(
                "in.webm",
                "out.ts",
                VideoContainerFormat::Webm,
                ts,
                vec![stream(VideoCodec::Vp9, VideoStreamKind::Video)],
                VideoTrackAction::Copy,
                AudioTrackAction::Copy
            ),
            Err(DomainError::UnsupportedVideoCodec {
                codec: VideoCodec::Vp9,
                container: ts
            })
        );
        let valid = item(
            "in.mp4",
            "out.ts",
            VideoContainerFormat::Mp4,
            ts,
            Vec::new(),
            cpu_encode(TargetVideoCodec::H264),
            audio_encode(TargetAudioCodec::Mp3),
        )?;
        assert_eq!(valid.output_format(), ts);
        Ok(())
    }

    #[test]
    fn transcode_item_rejects_invalid_paths() {
        let target = vec![
            stream(VideoCodec::Vp9, VideoStreamKind::Video),
            stream(VideoCodec::Opus, VideoStreamKind::Audio),
        ];
        assert_eq!(
            item(
                "",
                "out.webm",
                VideoContainerFormat::Webm,
                VideoContainerFormat::Webm,
                target.clone(),
                VideoTrackAction::Copy,
                AudioTrackAction::Copy
            ),
            Err(DomainError::EmptyPath(PathRole::Source))
        );
        assert_eq!(
            item(
                "in.webm",
                "",
                VideoContainerFormat::Webm,
                VideoContainerFormat::Webm,
                target.clone(),
                VideoTrackAction::Copy,
                AudioTrackAction::Copy
            ),
            Err(DomainError::EmptyPath(PathRole::Target))
        );
        assert_eq!(
            item(
                "in.webm",
                "in.webm",
                VideoContainerFormat::Webm,
                VideoContainerFormat::Webm,
                target,
                VideoTrackAction::Copy,
                AudioTrackAction::Copy
            ),
            Err(DomainError::IdenticalSourceAndTarget(PathBuf::from(
                "in.webm"
            )))
        );
    }

    #[test]
    fn transcode_plan_rejects_empty_duplicates_and_overwrites() -> Result<(), DomainError> {
        let w = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        assert!(matches!(
            VideoTranscodePlan::new(Vec::new(), w),
            Err(DomainError::EmptyBatch)
        ));

        let make = |source: &str, target: &str| {
            item(
                source,
                target,
                VideoContainerFormat::Mp4,
                VideoContainerFormat::Webm,
                Vec::new(),
                cpu_encode(TargetVideoCodec::Vp9),
                audio_encode(TargetAudioCodec::Opus),
            )
        };
        assert!(matches!(
            VideoTranscodePlan::new(
                vec![make("a.mp4", "out.webm")?, make("b.mp4", "out.webm")?],
                w
            ),
            Err(DomainError::DuplicateTargetPath(_))
        ));
        assert!(matches!(
            VideoTranscodePlan::new(vec![make("a.mp4", "b.mp4")?, make("b.mp4", "out.webm")?], w),
            Err(DomainError::TargetOverwritesSource(_))
        ));

        let plan =
            VideoTranscodePlan::new(vec![make("a.mp4", "a.webm")?, make("b.mp4", "b.webm")?], w)?;
        assert_eq!(plan.items().len(), 2);
        assert_eq!(plan.worker_threads(), w);
        assert_eq!(plan.effective_workers().get(), 2);
        assert_eq!(plan.output_format(), VideoContainerFormat::Webm);
        assert_eq!(plan.backend(), HardwareBackend::Cpu);
        Ok(())
    }

    #[test]
    fn transcode_report_metrics() {
        let w = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        let input = NonZeroU64::new(2 * 1024 * 1024).unwrap_or(NonZeroU64::MIN);
        let output = NonZeroU64::new(1024 * 1024).unwrap_or(NonZeroU64::MIN);
        let files = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        let report = VideoTranscodeReport::new(
            files,
            600,
            input,
            output,
            Duration::from_secs(2),
            w,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            Some("cuda:0".to_owned()),
            VideoDecodeMode::Nvdec,
        );
        assert_eq!(report.files_converted(), files);
        assert_eq!(report.frames_processed(), 600);
        assert_eq!(report.total_input_bytes(), input);
        assert_eq!(report.total_output_bytes(), output);
        assert_eq!(report.duration(), Duration::from_secs(2));
        assert_eq!(report.worker_threads_used(), w);
        assert_eq!(report.backend(), HardwareBackend::Gpu(GpuBackend::Nvenc));
        assert_eq!(report.hardware_device(), Some("cuda:0"));
        assert_eq!(report.megabytes_per_sec(), 1.0);
        assert_eq!(report.fps(), 300.0);
    }

    #[test]
    fn remux_plan_rejects_empty_and_duplicates() {
        let w = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        assert!(matches!(
            VideoRemuxPlan::new(Vec::new(), w),
            Err(DomainError::EmptyBatch)
        ));
    }

    #[test]
    fn remux_plan_rejects_animation_formats() -> Result<(), String> {
        let w = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        let gif_source = VideoRemuxItem::new(
            PathBuf::from("in.gif"),
            PathBuf::from("out.mkv"),
            VideoContainerFormat::Gif,
            VideoContainerFormat::Mkv,
            vec![stream(VideoCodec::Gif, VideoStreamKind::Video)],
        );
        assert!(matches!(
            VideoRemuxPlan::new(vec![gif_source], w),
            Err(DomainError::UnsupportedVideoPipeline { .. })
        ));
        let webp_target = VideoRemuxItem::new(
            PathBuf::from("in.mp4"),
            PathBuf::from("out.webp"),
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Webp,
            vec![stream(VideoCodec::H264, VideoStreamKind::Video)],
        );
        assert!(matches!(
            VideoRemuxPlan::new(vec![webp_target], w),
            Err(DomainError::UnsupportedVideoPipeline { .. })
        ));
        let valid = VideoRemuxItem::new(
            PathBuf::from("in.mp4"),
            PathBuf::from("out.mkv"),
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mkv,
            vec![stream(VideoCodec::H264, VideoStreamKind::Video)],
        );
        assert_eq!(
            VideoRemuxPlan::new(vec![valid], w)
                .map_err(|err| err.to_string())?
                .items()
                .len(),
            1
        );
        Ok(())
    }

    #[test]
    fn transcode_item_accepts_animation_with_dropped_audio()
    -> Result<(), Box<dyn std::error::Error>> {
        let video = stream(VideoCodec::Gif, VideoStreamKind::Video);
        let into_gif = item(
            "in.mp4",
            "out.gif",
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Gif,
            vec![video.clone()],
            cpu_encode(TargetVideoCodec::Gif),
            AudioTrackAction::Drop,
        )?;
        assert_eq!(into_gif.output_format(), VideoContainerFormat::Gif);
        assert_eq!(into_gif.audio_action(), AudioTrackAction::Drop);
        let into_webp = item(
            "in.gif",
            "out.webp",
            VideoContainerFormat::Gif,
            VideoContainerFormat::Webp,
            vec![video],
            cpu_encode(TargetVideoCodec::Webp),
            AudioTrackAction::Drop,
        )?;
        assert_eq!(into_webp.video_action(), cpu_encode(TargetVideoCodec::Webp));
        // An animation container with no streams has nothing to reject.
        let empty = item(
            "in.mp4",
            "empty.gif",
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Gif,
            Vec::new(),
            cpu_encode(TargetVideoCodec::Gif),
            AudioTrackAction::Drop,
        )?;
        assert_eq!(empty.streams(), []);
        Ok(())
    }

    #[test]
    fn transcode_item_rejects_illegal_animation_audio_and_copy() {
        let video = stream(VideoCodec::Gif, VideoStreamKind::Video);
        let audio = stream(VideoCodec::Aac, VideoStreamKind::Audio);
        let result = item(
            "in.gif",
            "out.gif",
            VideoContainerFormat::Gif,
            VideoContainerFormat::Gif,
            vec![video.clone()],
            VideoTrackAction::Copy,
            AudioTrackAction::Drop,
        );
        assert!(
            matches!(result, Err(DomainError::UnsupportedVideoPipeline { .. })),
            "animation streams cannot be copied"
        );
        let result = item(
            "in.mp4",
            "out.gif",
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Gif,
            vec![video.clone(), audio.clone()],
            cpu_encode(TargetVideoCodec::Gif),
            AudioTrackAction::Copy,
        );
        assert!(
            matches!(result, Err(DomainError::UnsupportedVideoPipeline { .. })),
            "animation containers cannot carry copied audio"
        );
        let result = item(
            "in.mp4",
            "out.gif",
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Gif,
            vec![audio],
            cpu_encode(TargetVideoCodec::Gif),
            AudioTrackAction::Drop,
        );
        assert!(
            matches!(result, Err(DomainError::UnsupportedVideoPipeline { .. })),
            "animation containers require at least one video stream"
        );
    }

    #[test]
    fn hwdecode_auto_picks_nvdec_only_for_supported_codecs() {
        let nvenc = HardwareBackend::Gpu(GpuBackend::Nvenc);
        for codec in [VideoCodec::H264, VideoCodec::Hevc, VideoCodec::Vp9] {
            assert_eq!(
                HardwareDecodeBackend::Auto.resolve(codec, nvenc),
                Ok(VideoDecodeMode::Nvdec),
                "auto must use NVDEC for {codec}"
            );
        }
        for codec in [VideoCodec::Av1, VideoCodec::Gif, VideoCodec::Webp] {
            assert_eq!(
                HardwareDecodeBackend::Auto.resolve(codec, nvenc),
                Ok(VideoDecodeMode::Cpu),
                "auto must use CPU decode for {codec}"
            );
        }
        assert_eq!(
            HardwareDecodeBackend::Auto.resolve(VideoCodec::H264, HardwareBackend::Cpu),
            Ok(VideoDecodeMode::Cpu)
        );
        assert_eq!(
            HardwareDecodeBackend::Cpu.resolve(VideoCodec::Hevc, nvenc),
            Ok(VideoDecodeMode::Cpu)
        );
    }

    #[test]
    fn hwdecode_nvdec_rejects_unsupported_pipelines_without_fallback() {
        let nvenc = HardwareBackend::Gpu(GpuBackend::Nvenc);
        assert!(matches!(
            HardwareDecodeBackend::Nvdec.resolve(VideoCodec::H264, HardwareBackend::Cpu),
            Err(DomainError::UnsupportedVideoPipeline { .. })
        ));
        for codec in [VideoCodec::Av1, VideoCodec::Gif, VideoCodec::Webp] {
            assert!(
                matches!(
                    HardwareDecodeBackend::Nvdec.resolve(codec, nvenc),
                    Err(DomainError::UnsupportedVideoPipeline { .. })
                ),
                "forced NVDEC must reject {codec}"
            );
        }
        assert_eq!(
            HardwareDecodeBackend::Nvdec.resolve(VideoCodec::Hevc, nvenc),
            Ok(VideoDecodeMode::Nvdec)
        );
    }

    #[test]
    fn video_item_rejects_nvdec_with_cpu_or_unsupported_stream() {
        let quality = VideoQuality::DEFAULT;
        let h264 = stream(VideoCodec::H264, VideoStreamKind::Video);
        let av1 = stream(VideoCodec::Av1, VideoStreamKind::Video);
        let nvenc_action = VideoTrackAction::Encode(VideoEncodeMode::NvencH264 {
            rate_control: VideoRateControl::Quality(quality),
        });
        assert!(
            item(
                "in.mp4",
                "out.mp4",
                VideoContainerFormat::Mp4,
                VideoContainerFormat::Mp4,
                vec![h264.clone()],
                nvenc_action,
                AudioTrackAction::Drop,
            )
            .is_ok()
        );

        let result = VideoTranscodeItem::new(
            PathBuf::from("in.mp4"),
            PathBuf::from("out.mp4"),
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mp4,
            vec![h264],
            cpu_encode(TargetVideoCodec::H264),
            AudioTrackAction::Drop,
            VideoDecodeMode::Nvdec,
        );
        assert!(
            matches!(result, Err(DomainError::UnsupportedVideoPipeline { .. })),
            "NVDEC decode with CPU encoding must be rejected"
        );

        let result = VideoTranscodeItem::new(
            PathBuf::from("in.mp4"),
            PathBuf::from("out.mp4"),
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mp4,
            vec![av1],
            nvenc_action,
            AudioTrackAction::Drop,
            VideoDecodeMode::Nvdec,
        );
        assert!(
            matches!(result, Err(DomainError::UnsupportedVideoPipeline { .. })),
            "NVDEC decode of AV1 must be rejected"
        );
    }

    #[test]
    fn report_surfaces_exact_hardware_path() {
        let one = NonZeroUsize::MIN;
        let bytes = NonZeroU64::MIN;
        let report = VideoTranscodeReport::new(
            one,
            1,
            bytes,
            bytes,
            Duration::from_secs(1),
            one,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            Some("RTX 2060".to_owned()),
            VideoDecodeMode::Nvdec,
        );
        assert_eq!(report.decode_path(), "gpu:nvdec+nvenc");
        let cpu_decode = VideoTranscodeReport::new(
            one,
            1,
            bytes,
            bytes,
            Duration::from_secs(1),
            one,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            None,
            VideoDecodeMode::Cpu,
        );
        assert_eq!(cpu_decode.decode_path(), "gpu:nvenc (cpu decode)");
    }

    #[test]
    fn report_surfaces_mixed_decode_paths() {
        let one = NonZeroUsize::MIN;
        let two = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        let bytes = NonZeroU64::MIN;
        let modes = [
            VideoDecodeMode::Nvdec,
            VideoDecodeMode::Cpu,
            VideoDecodeMode::Nvdec,
        ];
        assert_eq!(
            VideoDecodePaths::from_modes(modes),
            Some(VideoDecodePaths::Mixed {
                nvdec_files: two,
                cpu_files: one,
            })
        );
        assert_eq!(
            VideoDecodePaths::from_modes([VideoDecodeMode::Cpu; 2]),
            Some(VideoDecodePaths::Cpu)
        );
        assert_eq!(VideoDecodePaths::from_modes([]), None);

        let report = VideoTranscodeReport::new(
            NonZeroUsize::new(3).unwrap_or(NonZeroUsize::MIN),
            3,
            bytes,
            bytes,
            Duration::from_secs(1),
            one,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            None,
            VideoDecodePaths::Mixed {
                nvdec_files: two,
                cpu_files: one,
            },
        );
        assert_eq!(
            report.decode_paths(),
            VideoDecodePaths::Mixed {
                nvdec_files: two,
                cpu_files: one,
            }
        );
        assert_eq!(
            report.decode_path(),
            "mixed: gpu:nvdec+nvenc + gpu:nvenc (cpu decode)"
        );
    }

    #[test]
    fn report_without_video_decode_has_no_video_path() {
        let one = NonZeroUsize::MIN;
        let bytes = NonZeroU64::MIN;
        let report = VideoTranscodeReport::new(
            one,
            0,
            bytes,
            bytes,
            Duration::from_secs(1),
            one,
            HardwareBackend::Cpu,
            None,
            VideoDecodePaths::NoVideo,
        );
        assert_eq!(report.decode_paths(), VideoDecodePaths::NoVideo);
        assert_eq!(report.decode_path(), "no video decoded");
    }

    #[test]
    fn transcode_plan_rejects_mixed_encoder_backends() -> Result<(), DomainError> {
        let w = NonZeroUsize::MIN;
        let quality = VideoQuality::DEFAULT;
        let stream = vec![stream(VideoCodec::H264, VideoStreamKind::Video)];
        let cpu = VideoTranscodeItem::new(
            PathBuf::from("cpu.mp4"),
            PathBuf::from("cpu.out.mp4"),
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mp4,
            stream.clone(),
            VideoTrackAction::Encode(VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::Quality(quality),
            }),
            AudioTrackAction::Drop,
            VideoDecodeMode::Cpu,
        )?;
        let nvenc = VideoTranscodeItem::new(
            PathBuf::from("gpu.mp4"),
            PathBuf::from("gpu.out.mp4"),
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mp4,
            stream,
            VideoTrackAction::Encode(VideoEncodeMode::NvencH264 {
                rate_control: VideoRateControl::Quality(quality),
            }),
            AudioTrackAction::Drop,
            VideoDecodeMode::Cpu,
        )?;
        assert!(matches!(
            VideoTranscodePlan::new(vec![cpu, nvenc], w),
            Err(DomainError::UnsupportedVideoPipeline { .. })
        ));
        Ok(())
    }

    #[test]
    fn transcode_plan_keeps_nvenc_backend_with_both_decode_paths() -> Result<(), DomainError> {
        let w = NonZeroUsize::MIN;
        let quality = VideoQuality::DEFAULT;
        let stream = vec![stream(VideoCodec::H264, VideoStreamKind::Video)];
        let nvdec = VideoTranscodeItem::new(
            PathBuf::from("nvdec.mp4"),
            PathBuf::from("nvdec.out.mp4"),
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mp4,
            stream.clone(),
            VideoTrackAction::Encode(VideoEncodeMode::NvencH264 {
                rate_control: VideoRateControl::Quality(quality),
            }),
            AudioTrackAction::Drop,
            VideoDecodeMode::Nvdec,
        )?;
        let cpu_decode = VideoTranscodeItem::new(
            PathBuf::from("cpu.mp4"),
            PathBuf::from("cpu.out.mp4"),
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mp4,
            stream,
            VideoTrackAction::Encode(VideoEncodeMode::NvencH264 {
                rate_control: VideoRateControl::Quality(quality),
            }),
            AudioTrackAction::Drop,
            VideoDecodeMode::Cpu,
        )?;
        let plan = VideoTranscodePlan::new(vec![nvdec, cpu_decode], w)?;
        assert_eq!(plan.backend(), HardwareBackend::Gpu(GpuBackend::Nvenc));
        assert_eq!(plan.items()[0].decode_mode(), VideoDecodeMode::Nvdec);
        assert_eq!(plan.items()[1].decode_mode(), VideoDecodeMode::Cpu);
        Ok(())
    }

    #[test]
    fn video_bitrate_parses_all_forms() -> Result<(), DomainError> {
        for (raw, bps) in [
            ("500k", 500_000),
            ("2000K", 2_000_000),
            ("5M", 5_000_000),
            ("2.5m", 2_500_000),
            ("0.5M", 500_000),
            ("5000000", 5_000_000),
            ("  1.25M  ", 1_250_000),
        ] {
            assert_eq!(VideoBitrate::parse(raw)?.bps(), bps, "parse {raw}");
        }
        assert_eq!(VideoBitrate::new(50_000)?.bps(), 50_000);
        assert_eq!(VideoBitrate::new(500_000_000)?.bps(), 500_000_000);
        assert_eq!(VideoBitrate::new(2_000_000)?.to_string(), "2000k");
        assert_eq!(
            VideoRateControl::Bitrate(VideoBitrate::new(2_000_000)?).to_string(),
            "bitrate 2000k"
        );
        Ok(())
    }

    #[test]
    fn bitrate_parsing_rejects_illegal_input() {
        for raw in [
            "",
            "   ",
            "k",
            "0",
            "-1k",
            "1.2.3M",
            "2.M",
            "1000.5",
            "1.0005k",
            "1.0000001M",
            "8M8",
            "999999999999M",
            "abc",
        ] {
            assert!(
                matches!(
                    VideoBitrate::parse(raw),
                    Err(DomainError::InvalidBitrateSyntax { .. }
                        | DomainError::InvalidVideoBitrate { .. })
                ),
                "must reject {raw:?}"
            );
        }
        assert_eq!(
            VideoBitrate::new(49_999),
            Err(DomainError::InvalidVideoBitrate {
                value: 49_999,
                min: 50_000,
                max: 500_000_000,
            })
        );
        assert_eq!(
            VideoBitrate::new(500_000_001),
            Err(DomainError::InvalidVideoBitrate {
                value: 500_000_001,
                min: 50_000,
                max: 500_000_000,
            })
        );
    }

    #[test]
    fn audio_bitrate_parses_and_enforces_codec_limits() -> Result<(), DomainError> {
        assert_eq!(AudioBitrate::parse("128k")?.bps(), 128_000);
        assert_eq!(AudioBitrate::parse("1.5M")?.bps(), 1_500_000);
        assert_eq!(AudioBitrate::parse("96000")?.bps(), 96_000);
        assert_eq!(AudioBitrate::new(1_000)?, AudioBitrate::parse("1k")?);
        assert!(AudioBitrate::new(999).is_err());
        for (codec, bps, accepted) in [
            (TargetAudioCodec::Aac, 16_000, true),
            (TargetAudioCodec::Aac, 15_999, false),
            (TargetAudioCodec::Aac, 512_000, true),
            (TargetAudioCodec::Aac, 512_001, false),
            (TargetAudioCodec::Opus, 6_000, true),
            (TargetAudioCodec::Opus, 5_999, false),
            (TargetAudioCodec::Opus, 510_000, true),
            (TargetAudioCodec::Opus, 510_001, false),
            (TargetAudioCodec::Mp3, 8_000, true),
            (TargetAudioCodec::Mp3, 7_999, false),
            (TargetAudioCodec::Mp3, 320_000, true),
            (TargetAudioCodec::Mp3, 320_001, false),
            (TargetAudioCodec::Vorbis, 32_000, true),
            (TargetAudioCodec::Vorbis, 31_999, false),
            (TargetAudioCodec::Vorbis, 512_000, true),
            (TargetAudioCodec::Vorbis, 512_001, false),
        ] {
            let options = AudioEncodeOptions::new(Some(AudioBitrate::new(bps)?), None, None);
            assert_eq!(
                AudioEncodeConfig::new(codec, options).is_ok(),
                accepted,
                "{codec} at {bps} bps"
            );
        }
        Ok(())
    }

    #[test]
    fn lossless_audio_codecs_reject_bitrate_and_validate_rate_channels() -> Result<(), DomainError>
    {
        let bitrate = AudioBitrate::new(128_000)?;
        for codec in [TargetAudioCodec::Flac, TargetAudioCodec::Pcm] {
            let options = AudioEncodeOptions::new(Some(bitrate), None, None);
            let result = AudioEncodeConfig::new(codec, options);
            let Err(err) = result else {
                return Err(DomainError::UnsupportedAudioEncodeOption {
                    codec,
                    reason: "lossless codecs must reject --abitrate".to_owned(),
                });
            };
            assert!(matches!(
                err,
                DomainError::UnsupportedAudioEncodeOption { .. }
            ));
            assert!(
                err.to_string().contains("lossless"),
                "error must explain lossless rejection: {err}"
            );
            // Empty options keep the codec's defaults legal.
            assert!(AudioEncodeConfig::new(codec, AudioEncodeOptions::default()).is_ok());
        }
        for (codec, rate, accepted) in [
            (TargetAudioCodec::Flac, 1, true),
            (TargetAudioCodec::Flac, 44_100, true),
            (TargetAudioCodec::Flac, 655_350, true),
            (TargetAudioCodec::Flac, 655_351, false),
            (TargetAudioCodec::Pcm, 1, true),
            (TargetAudioCodec::Pcm, 384_000, true),
            (TargetAudioCodec::Pcm, 384_001, false),
        ] {
            let options = AudioEncodeOptions::new(None, Some(AudioSampleRate::new(rate)?), None);
            assert_eq!(
                AudioEncodeConfig::new(codec, options).is_ok(),
                accepted,
                "{codec} at {rate} Hz"
            );
        }
        for codec in [TargetAudioCodec::Flac, TargetAudioCodec::Pcm] {
            for count in [1, 2, 8] {
                let options = AudioEncodeOptions::new(None, None, Some(AudioChannels::new(count)?));
                assert!(AudioEncodeConfig::new(codec, options).is_ok());
            }
            for count in [0, 9, 16] {
                if count == 0 {
                    continue; // AudioChannels::new(0) is already invalid.
                }
                let options = AudioEncodeOptions::new(None, None, Some(AudioChannels::new(count)?));
                assert!(AudioEncodeConfig::new(codec, options).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn flac_and_pcm_are_confined_to_their_containers() -> Result<(), DomainError> {
        assert_eq!(TargetAudioCodec::Flac.label(), "FLAC");
        assert_eq!(TargetAudioCodec::Pcm.label(), "PCM");
        assert_eq!(TargetAudioCodec::Flac.as_video_codec(), VideoCodec::Flac);
        assert_eq!(TargetAudioCodec::Pcm.as_video_codec(), VideoCodec::Pcm);
        for container in [
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Mov,
        ] {
            assert!(TargetAudioCodec::Flac.supported_in(container));
            let item = VideoTranscodeItem::new(
                PathBuf::from("in.mp4"),
                PathBuf::from("out.mkv"),
                VideoContainerFormat::Mp4,
                container,
                vec![VideoStreamInfo::new(
                    VideoCodec::H264,
                    VideoStreamKind::Video,
                    Some(64),
                    Some(48),
                    None,
                    None,
                    1,
                    1_000,
                    None,
                    None,
                )],
                VideoTrackAction::Encode(VideoEncodeMode::resolve(
                    TargetVideoCodec::H264,
                    VideoRateControl::DEFAULT,
                    HardwareBackend::Cpu,
                )?),
                AudioTrackAction::Encode(AudioEncodeConfig::default_for(TargetAudioCodec::Flac)),
                VideoDecodeMode::Cpu,
            )?;
            assert_eq!(item.output_format(), container);
        }
        for container in [
            VideoContainerFormat::Webm,
            VideoContainerFormat::Avi,
            VideoContainerFormat::Ts,
            VideoContainerFormat::Gif,
            VideoContainerFormat::Webp,
        ] {
            assert!(!TargetAudioCodec::Flac.supported_in(container));
        }
        for container in [
            VideoContainerFormat::Mkv,
            VideoContainerFormat::Avi,
            VideoContainerFormat::Mov,
        ] {
            assert!(TargetAudioCodec::Pcm.supported_in(container));
        }
        for container in [
            VideoContainerFormat::Webm,
            VideoContainerFormat::Mp4,
            VideoContainerFormat::Ts,
            VideoContainerFormat::Gif,
            VideoContainerFormat::Webp,
        ] {
            assert!(!TargetAudioCodec::Pcm.supported_in(container));
        }
        Ok(())
    }

    #[test]
    fn mp3_bitrate_follows_sample_rate_band() -> Result<(), DomainError> {
        let low = AudioSampleRate::new(22_050)?;
        let high = AudioSampleRate::new(48_000)?;
        // 200 kbps exceeds the low-rate band's 160 kbps ceiling but fits the
        // high-rate band; 16 kbps is the mirror image.
        let bps = AudioBitrate::new(200_000)?;
        let low_options = AudioEncodeOptions::new(Some(bps), Some(low), None);
        let high_options = AudioEncodeOptions::new(Some(bps), Some(high), None);
        assert!(AudioEncodeConfig::new(TargetAudioCodec::Mp3, low_options).is_err());
        assert!(AudioEncodeConfig::new(TargetAudioCodec::Mp3, high_options).is_ok());
        let bps = AudioBitrate::new(16_000)?;
        let low_options = AudioEncodeOptions::new(Some(bps), Some(low), None);
        let high_options = AudioEncodeOptions::new(Some(bps), Some(high), None);
        assert!(AudioEncodeConfig::new(TargetAudioCodec::Mp3, low_options).is_ok());
        assert!(AudioEncodeConfig::new(TargetAudioCodec::Mp3, high_options).is_err());
        Ok(())
    }

    #[test]
    fn effective_mp3_pair_checks_worker_defaults() -> Result<(), DomainError> {
        let config = AudioEncodeConfig::default_for(TargetAudioCodec::Mp3);
        // A rate-only override inherits 192 kbps, too high below 32 kHz.
        assert!(config.validate_effective_mp3(192_000, 22_050).is_err());
        // A bitrate-only override inherits a 48 kHz source, below the floor.
        assert!(config.validate_effective_mp3(16_000, 48_000).is_err());
        // Valid pairs stay valid in both bands.
        config.validate_effective_mp3(128_000, 22_050)?;
        config.validate_effective_mp3(64_000, 48_000)?;
        // Other codecs have no rate-dependent band.
        AudioEncodeConfig::default_for(TargetAudioCodec::Aac)
            .validate_effective_mp3(16_000, 22_050)?;
        Ok(())
    }

    #[test]
    fn audio_sample_rates_and_channels_follow_codec_limits() -> Result<(), DomainError> {
        assert_eq!(
            AudioSampleRate::new(0),
            Err(DomainError::InvalidAudioSampleRate { value: 0 })
        );
        assert_eq!(
            AudioChannels::new(0),
            Err(DomainError::InvalidAudioChannels { value: 0 })
        );
        assert_eq!(AudioSampleRate::new(48_000)?.hz(), 48_000);
        assert_eq!(AudioChannels::new(6)?.count(), 6);
        assert_eq!(AudioSampleRate::new(48_000)?.to_string(), "48000 Hz");
        assert_eq!(AudioChannels::new(2)?.to_string(), "2ch");

        for (codec, rate, accepted) in [
            (TargetAudioCodec::Aac, 7_350, true),
            (TargetAudioCodec::Aac, 96_000, true),
            (TargetAudioCodec::Aac, 44_100, true),
            (TargetAudioCodec::Aac, 44_200, false),
            (TargetAudioCodec::Opus, 48_000, true),
            (TargetAudioCodec::Opus, 44_100, false),
            (TargetAudioCodec::Mp3, 8_000, true),
            (TargetAudioCodec::Mp3, 48_000, true),
            (TargetAudioCodec::Mp3, 96_000, false),
            (TargetAudioCodec::Vorbis, 8_000, true),
            (TargetAudioCodec::Vorbis, 44_100, true),
            (TargetAudioCodec::Vorbis, 192_000, true),
            (TargetAudioCodec::Vorbis, 7_999, false),
            (TargetAudioCodec::Vorbis, 192_001, false),
        ] {
            let options = AudioEncodeOptions::new(None, Some(AudioSampleRate::new(rate)?), None);
            assert_eq!(
                AudioEncodeConfig::new(codec, options).is_ok(),
                accepted,
                "{codec} at {rate} Hz"
            );
        }
        for (codec, channels, accepted) in [
            (TargetAudioCodec::Aac, 1, true),
            (TargetAudioCodec::Aac, 2, true),
            (TargetAudioCodec::Aac, 3, true),
            (TargetAudioCodec::Aac, 4, false),
            (TargetAudioCodec::Aac, 5, true),
            (TargetAudioCodec::Aac, 6, true),
            (TargetAudioCodec::Aac, 7, false),
            (TargetAudioCodec::Opus, 2, true),
            (TargetAudioCodec::Opus, 3, false),
            (TargetAudioCodec::Mp3, 1, true),
            (TargetAudioCodec::Mp3, 2, true),
            (TargetAudioCodec::Mp3, 3, false),
            (TargetAudioCodec::Vorbis, 1, true),
            (TargetAudioCodec::Vorbis, 2, true),
            (TargetAudioCodec::Vorbis, 3, true),
            (TargetAudioCodec::Vorbis, 4, false),
            (TargetAudioCodec::Vorbis, 5, true),
            (TargetAudioCodec::Vorbis, 6, true),
            (TargetAudioCodec::Vorbis, 7, false),
        ] {
            let options = AudioEncodeOptions::new(None, None, Some(AudioChannels::new(channels)?));
            assert_eq!(
                AudioEncodeConfig::new(codec, options).is_ok(),
                accepted,
                "{codec} with {channels} channels"
            );
        }
        Ok(())
    }

    #[test]
    fn audio_encode_config_defaults_and_options() {
        let options = AudioEncodeOptions::new(None, None, None);
        assert!(options.is_empty());
        let config = AudioEncodeConfig::default_for(TargetAudioCodec::Aac);
        assert_eq!(config.codec(), TargetAudioCodec::Aac);
        assert!(config.bitrate().is_none());
        assert!(config.sample_rate().is_none());
        assert!(config.channels().is_none());
        let rate = ab(192_000);
        let options = AudioEncodeOptions::new(Some(rate), None, None);
        assert!(!options.is_empty());
        assert_eq!(options.bitrate(), Some(rate));
        let config = AudioEncodeConfig::new(TargetAudioCodec::Mp3, options);
        assert!(config.is_ok());
        assert_eq!(config.map_or(None, AudioEncodeConfig::bitrate), Some(rate));
    }

    #[test]
    fn domain_resolution_rejects_conflicting_options_without_fallback() {
        assert_eq!(
            resolve_video_rate_control(true, Some(VideoRateControl::DEFAULT)),
            Err(DomainError::ConflictingVideoRateControl {
                reason: "video stream copy cannot re-encode; omit --quality and --vbitrate"
                    .to_owned()
            })
        );
        assert_eq!(
            resolve_video_rate_control(true, None).map(|value| value.to_string()),
            Ok("quality 80".to_owned())
        );
        let bitrate = vb(2_000_000);
        assert_eq!(
            resolve_video_rate_control(false, Some(VideoRateControl::Bitrate(bitrate)))
                .map(|value| value.to_string()),
            Ok("bitrate 2000k".to_owned())
        );

        let options = AudioEncodeOptions::new(Some(ab(128_000)), None, None);
        assert!(resolve_audio_encode_options(true, options).is_err());
        assert_eq!(resolve_audio_encode_options(false, options), Ok(options));
        assert!(validate_audio_options_for_container(options, VideoContainerFormat::Gif).is_err());
        assert!(validate_audio_options_for_container(options, VideoContainerFormat::Mp4).is_ok());
    }

    #[test]
    fn animation_modes_cannot_hold_bitrate_and_display_is_stable() {
        let bitrate = vb(2_000_000);
        let rate_control = VideoRateControl::Bitrate(bitrate);
        for codec in [TargetVideoCodec::Gif, TargetVideoCodec::Webp] {
            assert_eq!(
                VideoEncodeMode::resolve(codec, rate_control, HardwareBackend::Cpu)
                    .err()
                    .map(|err| matches!(err, DomainError::UnsupportedVideoPipeline { .. })),
                Some(true)
            );
        }
        let rc = VideoRateControl::Quality(VideoQuality::DEFAULT);
        assert_eq!(
            VideoEncodeMode::CpuH264 { rate_control: rc }.to_string(),
            "cpu H.264 (quality 80)"
        );
        assert_eq!(
            VideoEncodeMode::NvencHevc { rate_control: rc }.to_string(),
            "gpu:nvenc HEVC (quality 80)"
        );
        assert_eq!(
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::Bitrate(bitrate)
            }
            .to_string(),
            "cpu H.264 (bitrate 2000k)"
        );
    }
}
