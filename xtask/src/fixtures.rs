use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use jpeg_encoder::{
    ChromaSubsamplingMethod, ColorType as JpegColorType, Encoder as JpegEncoder, SamplingFactor,
};

pub const IMAGE_BATCH_COUNT: usize = 20;
pub const IMAGE_WIDTH: u32 = 3840;
pub const IMAGE_HEIGHT: u32 = 2160;

pub const AUDIO_SAMPLE_RATE: u32 = 48_000;
pub const AUDIO_CHANNELS: u16 = 2;
pub const AUDIO_DURATION_SECS: u32 = 60;

pub const VIDEO_FPS: u64 = 60;
pub const VIDEO_DURATION_SECS: u64 = 10;
pub const VIDEO_TOTAL_FRAMES: u64 = VIDEO_FPS * VIDEO_DURATION_SECS;

#[must_use]
pub fn images_dir(root: &Path) -> PathBuf {
    root.join("fixtures/images_4k")
}

#[must_use]
pub fn image_path(root: &Path, index: usize) -> PathBuf {
    images_dir(root).join(format!("img_{index:02}.png"))
}

#[must_use]
pub fn jpeg_images_dir(root: &Path) -> PathBuf {
    root.join("fixtures/images_4k_jpeg")
}

#[must_use]
pub fn jpeg_image_path(root: &Path, index: usize) -> PathBuf {
    jpeg_images_dir(root).join(format!("img_{index:02}.jpg"))
}

#[must_use]
pub fn audio_wav_path(root: &Path) -> PathBuf {
    root.join("fixtures/audio_60s_stereo.wav")
}

#[must_use]
pub fn video_mp4_path(root: &Path) -> PathBuf {
    root.join("fixtures/video_1080p60.mp4")
}

pub fn generate_all(root: &Path, force: bool) -> Result<(), String> {
    generate_image_batch(root, force)?;
    generate_jpeg_batch(root, force)?;
    generate_audio_wav(root, force)?;
    generate_video_mp4(root, force)?;
    Ok(())
}

#[must_use]
pub fn is_non_empty_file(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.len() > 0)
}

fn generate_image_batch(root: &Path, force: bool) -> Result<(), String> {
    let dir = images_dir(root);
    fs::create_dir_all(&dir)
        .map_err(|err| format!("failed to create '{}': {err}", dir.display()))?;

    let all_present = (1..=IMAGE_BATCH_COUNT).all(|idx| is_non_empty_file(&image_path(root, idx)));
    if all_present && !force {
        println!(
            "==> skip 4K PNG batch ({IMAGE_BATCH_COUNT} files already exist in {})",
            dir.display()
        );
        return Ok(());
    }

    run_parallel_4k_generation("PNG", |index| write_single_4k_png(root, index))
}

fn generate_jpeg_batch(root: &Path, force: bool) -> Result<(), String> {
    let dir = jpeg_images_dir(root);
    fs::create_dir_all(&dir)
        .map_err(|err| format!("failed to create '{}': {err}", dir.display()))?;

    let all_present =
        (1..=IMAGE_BATCH_COUNT).all(|idx| is_non_empty_file(&jpeg_image_path(root, idx)));
    if all_present && !force {
        println!(
            "==> skip 4K JPEG batch ({IMAGE_BATCH_COUNT} files already exist in {})",
            dir.display()
        );
        return Ok(());
    }

    run_parallel_4k_generation("JPEG", |index| write_single_4k_jpeg(root, index))
}

fn run_parallel_4k_generation<F>(label: &str, generate_one: F) -> Result<(), String>
where
    F: Fn(usize) -> Result<(), String> + Sync,
{
    let workers = thread::available_parallelism()
        .map_or(4, NonZeroUsize::get)
        .min(IMAGE_BATCH_COUNT);

    println!(
        "==> generating {IMAGE_BATCH_COUNT}x {IMAGE_WIDTH}x{IMAGE_HEIGHT} {label} fixtures using {workers} threads..."
    );

    let next_index = AtomicUsize::new(1);
    let results: Vec<Result<(), String>> = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            handles.push(scope.spawn(|| {
                loop {
                    let index = next_index.fetch_add(1, Ordering::Relaxed);
                    if index > IMAGE_BATCH_COUNT {
                        break;
                    }
                    generate_one(index)?;
                }
                Ok(())
            }));
        }

        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|_| Err(format!("{label} generator worker thread panicked")))
            })
            .collect()
    });

    for result in results {
        result?;
    }

    println!("==> generated {IMAGE_BATCH_COUNT} 4K {label} fixtures");
    Ok(())
}

fn generate_4k_rgb_frame(index: usize) -> Result<Vec<u8>, String> {
    let width = usize::try_from(IMAGE_WIDTH).map_err(|err| err.to_string())?;
    let height = usize::try_from(IMAGE_HEIGHT).map_err(|err| err.to_string())?;
    let row_bytes = width * 3;
    let mut rgb = vec![0_u8; row_bytes * height];

    let mut prng: u32 =
        0x9e37_79b9_u32 ^ u32::try_from(index).unwrap_or(1).wrapping_mul(0x85eb_ca6b);
    let frame_seed = index.wrapping_mul(17);

    for (y, row) in rgb.chunks_exact_mut(row_bytes).enumerate() {
        let y_shift = y >> 3;
        let y_tile = y >> 6;
        let (pixels, _) = row.as_chunks_mut::<3>();
        for (x, pixel) in pixels.iter_mut().enumerate() {
            prng ^= prng << 13;
            prng ^= prng >> 17;
            prng ^= prng << 5;
            let grain = prng.to_le_bytes()[0] & 0x0f;
            let checker = ((x >> 6) ^ y_tile ^ index) & 1;

            let r = ((x >> 4).wrapping_add(frame_seed) & 0xff).to_le_bytes()[0];
            let g = (y_shift.wrapping_add(frame_seed.wrapping_mul(3)) & 0xff).to_le_bytes()[0];
            let b = (((x ^ y) >> 4).wrapping_add(checker * 48) & 0xff).to_le_bytes()[0];

            pixel[0] = r.saturating_add(grain);
            pixel[1] = g.saturating_add(grain >> 1);
            pixel[2] = b.saturating_add(grain);
        }
    }

    Ok(rgb)
}

fn write_single_4k_png(root: &Path, index: usize) -> Result<(), String> {
    let target = image_path(root, index);
    let tmp = images_dir(root).join(format!(".img_{index:02}.png.tmp"));
    let rgb = generate_4k_rgb_frame(index)?;

    {
        let file = File::create(&tmp)
            .map_err(|err| format!("failed to create '{}': {err}", tmp.display()))?;
        let writer = BufWriter::with_capacity(256 * 1024, file);
        let mut encoder = png::Encoder::new(writer, IMAGE_WIDTH, IMAGE_HEIGHT);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);

        let mut png_writer = encoder
            .write_header()
            .map_err(|err| format!("failed to write PNG header for {}: {err}", tmp.display()))?;
        png_writer
            .write_image_data(&rgb)
            .map_err(|err| format!("failed to encode PNG '{}': {err}", tmp.display()))?;
    }

    fs::rename(&tmp, &target).map_err(|err| {
        format!(
            "failed to finalize PNG '{}' -> '{}': {err}",
            tmp.display(),
            target.display()
        )
    })?;

    Ok(())
}

fn write_single_4k_jpeg(root: &Path, index: usize) -> Result<(), String> {
    let target = jpeg_image_path(root, index);
    let tmp = jpeg_images_dir(root).join(format!(".img_{index:02}.jpg.tmp"));
    let rgb = generate_4k_rgb_frame(index)?;
    let width = u16::try_from(IMAGE_WIDTH).map_err(|err| err.to_string())?;
    let height = u16::try_from(IMAGE_HEIGHT).map_err(|err| err.to_string())?;

    {
        let file = File::create(&tmp)
            .map_err(|err| format!("failed to create '{}': {err}", tmp.display()))?;
        let mut writer = BufWriter::with_capacity(256 * 1024, file);
        let mut encoder = JpegEncoder::new(&mut writer, 90);
        encoder.set_sampling_factor(SamplingFactor::F_2_2);
        encoder.set_chroma_subsampling_method(ChromaSubsamplingMethod::Average);
        encoder
            .encode(&rgb, width, height, JpegColorType::Rgb)
            .map_err(|err| format!("failed to encode JPEG '{}': {err}", tmp.display()))?;
        writer
            .flush()
            .map_err(|err| format!("failed to flush JPEG '{}': {err}", tmp.display()))?;
    }

    fs::rename(&tmp, &target).map_err(|err| {
        format!(
            "failed to finalize JPEG '{}' -> '{}': {err}",
            tmp.display(),
            target.display()
        )
    })?;

    Ok(())
}

fn generate_audio_wav(root: &Path, force: bool) -> Result<(), String> {
    let target = audio_wav_path(root);
    if is_non_empty_file(&target) && !force {
        println!("==> skip WAV fixture ({} already exists)", target.display());
        return Ok(());
    }

    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create '{}': {err}", parent.display()))?;
    }

    let tmp = root.join("fixtures/.audio_60s_stereo.wav.tmp");
    let total_frames = AUDIO_SAMPLE_RATE * AUDIO_DURATION_SECS;

    println!(
        "==> generating {AUDIO_DURATION_SECS}s stereo WAV fixture ({AUDIO_SAMPLE_RATE} Hz, 16-bit PCM)..."
    );

    {
        let file = File::create(&tmp)
            .map_err(|err| format!("failed to create '{}': {err}", tmp.display()))?;
        let mut writer = BufWriter::with_capacity(256 * 1024, file);
        write_synthetic_wav(&mut writer, total_frames, AUDIO_SAMPLE_RATE, AUDIO_CHANNELS)?;
        writer
            .flush()
            .map_err(|err| format!("failed to flush '{}': {err}", tmp.display()))?;
    }

    fs::rename(&tmp, &target).map_err(|err| {
        format!(
            "failed to finalize WAV '{}' -> '{}': {err}",
            tmp.display(),
            target.display()
        )
    })?;

    println!("==> generated {}", target.display());
    Ok(())
}

fn build_wav_header(
    total_frames: u32,
    sample_rate: u32,
    channels: u16,
) -> Result<[u8; 44], String> {
    if sample_rate == 0 {
        return Err("WAV sample rate must be greater than zero".to_owned());
    }
    if !matches!(channels, 1 | 2) {
        return Err(format!(
            "unsupported WAV channel count {channels}; expected 1 or 2"
        ));
    }
    let bytes_per_sample: u32 = 2;
    let block_align = u32::from(channels) * bytes_per_sample;
    let byte_rate = sample_rate
        .checked_mul(block_align)
        .ok_or_else(|| "WAV byte rate overflow".to_owned())?;
    let data_bytes = total_frames
        .checked_mul(block_align)
        .ok_or_else(|| "WAV data size overflow".to_owned())?;
    let riff_size = data_bytes
        .checked_add(36)
        .ok_or_else(|| "WAV RIFF size overflow".to_owned())?;
    let block_align_u16 = u16::try_from(block_align).map_err(|err| err.to_string())?;

    let mut header = [0_u8; 44];
    header[0..4].copy_from_slice(b"RIFF");
    header[4..8].copy_from_slice(&riff_size.to_le_bytes());
    header[8..12].copy_from_slice(b"WAVE");
    header[12..16].copy_from_slice(b"fmt ");
    header[16..20].copy_from_slice(&16_u32.to_le_bytes());
    header[20..22].copy_from_slice(&1_u16.to_le_bytes());
    header[22..24].copy_from_slice(&channels.to_le_bytes());
    header[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    header[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    header[32..34].copy_from_slice(&block_align_u16.to_le_bytes());
    header[34..36].copy_from_slice(&16_u16.to_le_bytes());
    header[36..40].copy_from_slice(b"data");
    header[40..44].copy_from_slice(&data_bytes.to_le_bytes());
    Ok(header)
}

fn parabolic_wave(phase: u16) -> i64 {
    let signed = i64::from(phase) - 32_768;
    let abs = signed.abs();
    (signed * (32_768 - abs)) >> 14
}

fn write_synthetic_wav<W: Write>(
    writer: &mut W,
    total_frames: u32,
    sample_rate: u32,
    channels: u16,
) -> Result<(), String> {
    let header = build_wav_header(total_frames, sample_rate, channels)?;
    writer
        .write_all(&header)
        .map_err(|err| format!("failed to write WAV header: {err}"))?;

    let mut phase_l: u16 = 0;
    let mut phase_r: u16 = 8_192;
    let mut phase_bass: u16 = 0;
    let mut prng: u32 = 0x1234_5678;

    let mut chunk = Vec::with_capacity(16_384 * 4);
    for frame in 0..total_frames {
        let sweep = u16::try_from((frame / 12_000) % 240).unwrap_or(0);
        phase_l = phase_l.wrapping_add(300 + sweep);
        phase_r = phase_r.wrapping_add(450 + sweep);
        phase_bass = phase_bass.wrapping_add(150);

        prng ^= prng << 13;
        prng ^= prng >> 17;
        prng ^= prng << 5;
        let dither = i64::from(prng & 0x0f) - 8;

        let bass = parabolic_wave(phase_bass) / 3;
        let left = (parabolic_wave(phase_l) + bass + dither).clamp(-32_767, 32_767);
        let right = (parabolic_wave(phase_r) + bass - dither).clamp(-32_767, 32_767);

        let left_i16 = i16::try_from(left).map_err(|err| err.to_string())?;
        let right_i16 = i16::try_from(right).map_err(|err| err.to_string())?;

        chunk.extend_from_slice(&left_i16.to_le_bytes());
        if channels >= 2 {
            chunk.extend_from_slice(&right_i16.to_le_bytes());
        }

        if chunk.len() >= 16_384 * 4 {
            writer
                .write_all(&chunk)
                .map_err(|err| format!("failed to write WAV PCM data: {err}"))?;
            chunk.clear();
        }
    }

    if !chunk.is_empty() {
        writer
            .write_all(&chunk)
            .map_err(|err| format!("failed to write final WAV PCM chunk: {err}"))?;
    }

    Ok(())
}

fn generate_video_mp4(root: &Path, force: bool) -> Result<(), String> {
    let target = video_mp4_path(root);
    if is_non_empty_file(&target) && !force {
        println!(
            "==> skip video fixture ({} already exists)",
            target.display()
        );
        return Ok(());
    }

    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create '{}': {err}", parent.display()))?;
    }

    let tmp = root.join("fixtures/.video_1080p60.mp4.tmp");
    let tmp_str = tmp
        .to_str()
        .ok_or_else(|| "video temp fixture path is not valid UTF-8".to_owned())?;

    println!("==> generating 1080p60 10s MP4 fixture via ffmpeg...");
    let output = Command::new("ffmpeg")
        .stdin(Stdio::null())
        .args([
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=60:duration=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=10",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ac",
            "2",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            tmp_str,
        ])
        .output()
        .map_err(|err| format!("failed to launch ffmpeg for video fixture generation: {err}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "ffmpeg video fixture generation failed ({}): {}",
            output.status,
            stderr.trim()
        ));
    }

    fs::rename(&tmp, &target).map_err(|err| {
        format!(
            "failed to finalize video '{}' -> '{}': {err}",
            tmp.display(),
            target.display()
        )
    })?;

    println!("==> generated {}", target.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{build_wav_header, write_synthetic_wav};

    #[test]
    fn writes_valid_wav_header_and_payload_length() -> Result<(), String> {
        let header = build_wav_header(100, 48_000, 2)?;
        assert_eq!(&header[0..4], b"RIFF");
        assert_eq!(&header[8..12], b"WAVE");
        assert_eq!(&header[36..40], b"data");
        assert_eq!(
            u32::from_le_bytes([header[40], header[41], header[42], header[43]]),
            400
        );

        let mut buf = Vec::new();
        write_synthetic_wav(&mut buf, 100, 48_000, 2)?;
        assert_eq!(buf.len(), 44 + 400);

        let mut mono_buf = Vec::new();
        write_synthetic_wav(&mut mono_buf, 100, 48_000, 1)?;
        assert_eq!(mono_buf.len(), 44 + 200);
        Ok(())
    }

    #[test]
    fn rejects_invalid_wav_parameters() {
        assert!(build_wav_header(100, 0, 2).is_err());
        assert!(build_wav_header(100, 48_000, 0).is_err());
        assert!(build_wav_header(100, 48_000, 4).is_err());
    }
}
