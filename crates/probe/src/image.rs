use std::fs::{self, File};
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use fastconv_domain::{
    DomainError, HardwareBackend, HostHardware, ImageBatchItem, ImageBatchPlan, ImageCompression,
    ImageDimensions, ImageExecutionMode, ImageInputFormat, ImageOutputFormat, PathRole,
};
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::colorspace::ColorSpace;

use crate::{ProbeError, available_cpu_threads, detect_host_hardware};

const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
const JPEG_MAGIC: [u8; 3] = [0xFF, 0xD8, 0xFF];
const UNRECOGNIZED_MAGIC_REASON: &str = "unrecognized image magic bytes (expected PNG or JPEG)";

/// Detected input container format and validated pixel dimensions from image headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbedImage {
    pub format: ImageInputFormat,
    pub dimensions: ImageDimensions,
}

fn sniff_magic_format(header: &[u8]) -> Option<ImageInputFormat> {
    if header.starts_with(&PNG_MAGIC) {
        Some(ImageInputFormat::Png)
    } else if header.starts_with(&JPEG_MAGIC) {
        Some(ImageInputFormat::Jpeg)
    } else {
        None
    }
}

fn read_magic_prefix(file: &mut File, path: &Path) -> Result<([u8; 8], usize), ProbeError> {
    let mut buf = [0_u8; 8];
    let mut filled = 0_usize;
    while filled < buf.len() {
        let slot = buf
            .get_mut(filled..)
            .ok_or_else(|| ProbeError::InvalidImage {
                path: path.to_path_buf(),
                reason: "internal magic-byte buffer offset overflow".to_owned(),
            })?;
        match file.read(slot) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(source) => {
                return Err(ProbeError::Io {
                    path: path.to_path_buf(),
                    source,
                });
            }
        }
    }
    Ok((buf, filled))
}

fn is_supported_jpeg_colorspace(cs: Option<ColorSpace>) -> bool {
    matches!(
        cs,
        Some(
            ColorSpace::Luma
                | ColorSpace::YCbCr
                | ColorSpace::RGB
                | ColorSpace::CMYK
                | ColorSpace::YCCK
        )
    )
}

fn probe_opened_image(
    mut file: File,
    path: &Path,
    format: ImageInputFormat,
) -> Result<ProbedImage, ProbeError> {
    file.seek(SeekFrom::Start(0))
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;

    let dimensions = match format {
        ImageInputFormat::Png => {
            let mut decoder = png::Decoder::new(BufReader::new(file));
            decoder.set_ignore_text_chunk(true);
            let reader = decoder
                .read_info()
                .map_err(|err| ProbeError::InvalidImage {
                    path: path.to_path_buf(),
                    reason: err.to_string(),
                })?;
            let info = reader.info();
            ImageDimensions::new(info.width, info.height)?
        }
        ImageInputFormat::Jpeg => {
            let mut decoder = JpegDecoder::new(BufReader::new(file));
            decoder
                .decode_headers()
                .map_err(|err| ProbeError::InvalidImage {
                    path: path.to_path_buf(),
                    reason: err.to_string(),
                })?;
            let info = decoder.info().ok_or_else(|| ProbeError::InvalidImage {
                path: path.to_path_buf(),
                reason: "missing JPEG SOF header metadata".to_owned(),
            })?;
            let input_cs = decoder.input_colorspace();
            if !is_supported_jpeg_colorspace(input_cs) {
                return Err(ProbeError::InvalidImage {
                    path: path.to_path_buf(),
                    reason: format!("unsupported JPEG color space: {input_cs:?}"),
                });
            }
            ImageDimensions::new(u32::from(info.width), u32::from(info.height))?
        }
    };

    Ok(ProbedImage { format, dimensions })
}

/// Probes image format via magic bytes and reads header dimensions without decoding pixels.
///
/// # Errors
///
/// Returns [`ProbeError::Io`] if `path` cannot be opened/seeked, [`ProbeError::InvalidImage`]
/// if magic bytes, headers, or JPEG color space are not valid/supported, or
/// [`ProbeError::InvalidDomain`] if dimensions are zero.
pub fn probe_image(path: &Path) -> Result<ProbedImage, ProbeError> {
    let mut file = File::open(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let (prefix, len) = read_magic_prefix(&mut file, path)?;
    let header_slice = prefix.get(..len).unwrap_or(&prefix);
    let format = sniff_magic_format(header_slice).ok_or_else(|| ProbeError::InvalidImage {
        path: path.to_path_buf(),
        reason: UNRECOGNIZED_MAGIC_REASON.to_owned(),
    })?;

    probe_opened_image(file, path, format)
}

/// Validates a single image file or mixed PNG/JPEG folder at the boundary against live host
/// hardware and returns a compile-time-legal [`ImageBatchPlan`].
///
/// Avoids initializing GPU probing (`wgpu`/Vulkan) when [`HardwareBackend::Cpu`] is explicitly
/// requested so CPU invocations incur zero loader overhead. Never performs silent CPU fallbacks.
///
/// # Errors
///
/// Returns [`ProbeError`] on missing/corrupt inputs, incompatible target paths or extensions,
/// empty source directories, or backends unavailable/unsupported for the chosen output format.
pub fn prepare_image_batch(
    source: &Path,
    target: &Path,
    compression: ImageCompression,
    output_format: Option<ImageOutputFormat>,
    worker_threads: Option<NonZeroUsize>,
    backend: HardwareBackend,
) -> Result<ImageBatchPlan, ProbeError> {
    prepare_image_batch_inner(
        source,
        target,
        compression,
        output_format,
        worker_threads,
        backend,
        None,
    )
}

fn prepare_image_batch_inner(
    source: &Path,
    target: &Path,
    compression: ImageCompression,
    output_format: Option<ImageOutputFormat>,
    worker_threads: Option<NonZeroUsize>,
    backend: HardwareBackend,
    hardware: Option<&HostHardware>,
) -> Result<ImageBatchPlan, ProbeError> {
    if source.as_os_str().is_empty() {
        return Err(ProbeError::InvalidDomain(DomainError::EmptyPath(
            PathRole::Source,
        )));
    }
    if target.as_os_str().is_empty() {
        return Err(ProbeError::InvalidDomain(DomainError::EmptyPath(
            PathRole::Target,
        )));
    }

    let threads = match worker_threads {
        Some(count) => count,
        None => available_cpu_threads()?,
    };

    let source_meta = fs::metadata(source).map_err(|err| ProbeError::Io {
        path: source.to_path_buf(),
        source: err,
    })?;

    if source_meta.is_file() {
        let (resolved_target, resolved_format) =
            resolve_single_target(source, target, output_format)?;
        let mode = resolve_execution_mode(resolved_format, compression, backend, hardware)?;
        let probed = probe_image(source)?;
        reject_canonical_same_path(source, &resolved_target)?;
        let item = ImageBatchItem::new(
            source.to_path_buf(),
            resolved_target,
            probed.format,
            probed.dimensions,
        )?;
        return Ok(ImageBatchPlan::new(vec![item], mode, threads)?);
    }

    if source_meta.is_dir() {
        let resolved_format = output_format.unwrap_or(ImageOutputFormat::Webp);
        let mode = resolve_execution_mode(resolved_format, compression, backend, hardware)?;
        return prepare_directory_batch(source, target, mode, threads);
    }

    Err(ProbeError::NotAFileOrDirectory(source.to_path_buf()))
}

fn resolve_execution_mode(
    format: ImageOutputFormat,
    compression: ImageCompression,
    backend: HardwareBackend,
    hardware: Option<&HostHardware>,
) -> Result<ImageExecutionMode, ProbeError> {
    match (backend, hardware) {
        (HardwareBackend::Cpu, _) => Ok(ImageExecutionMode::resolve_cpu(format, compression)?),
        (HardwareBackend::Gpu(_), Some(hw)) => Ok(ImageExecutionMode::resolve(
            format,
            compression,
            backend,
            hw,
        )?),
        (HardwareBackend::Gpu(_), None) => {
            let detected = detect_host_hardware()?;
            Ok(ImageExecutionMode::resolve(
                format,
                compression,
                backend,
                &detected,
            )?)
        }
    }
}

fn prepare_directory_batch(
    source_dir: &Path,
    target_dir: &Path,
    mode: ImageExecutionMode,
    threads: NonZeroUsize,
) -> Result<ImageBatchPlan, ProbeError> {
    match fs::metadata(target_dir) {
        Ok(meta) if meta.is_dir() => {}
        Ok(_) => {
            return Err(ProbeError::TargetNotADirectory(target_dir.to_path_buf()));
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => {
            return Err(ProbeError::Io {
                path: target_dir.to_path_buf(),
                source: err,
            });
        }
    }

    let entries = fs::read_dir(source_dir).map_err(|err| ProbeError::Io {
        path: source_dir.to_path_buf(),
        source: err,
    })?;

    let mut probed_entries = Vec::new();
    for entry_result in entries {
        let entry = entry_result.map_err(|err| ProbeError::Io {
            path: source_dir.to_path_buf(),
            source: err,
        })?;
        let path = entry.path();
        if let Some(probed) = probe_directory_entry(&path)? {
            probed_entries.push((path, probed));
        }
    }
    probed_entries.sort_by(|(a, _), (b, _)| a.cmp(b));

    if probed_entries.is_empty() {
        return Err(ProbeError::EmptySourceDirectory(source_dir.to_path_buf()));
    }

    let format = mode.output_format();
    let mut items = Vec::with_capacity(probed_entries.len());
    for (src_path, probed) in probed_entries {
        let out_path = derive_target_in_dir(&src_path, target_dir, format)?;
        reject_canonical_same_path(&src_path, &out_path)?;
        items.push(ImageBatchItem::new(
            src_path,
            out_path,
            probed.format,
            probed.dimensions,
        )?);
    }

    Ok(ImageBatchPlan::new(items, mode, threads)?)
}

fn probe_directory_entry(path: &Path) -> Result<Option<ProbedImage>, ProbeError> {
    let is_hidden = path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'));
    if is_hidden {
        return Ok(None);
    }

    let meta = fs::metadata(path).map_err(|err| ProbeError::Io {
        path: path.to_path_buf(),
        source: err,
    })?;
    if !meta.is_file() {
        return Ok(None);
    }

    let mut file = File::open(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let (prefix, len) = read_magic_prefix(&mut file, path)?;
    let header_slice = prefix.get(..len).unwrap_or(&prefix);
    if let Some(format) = sniff_magic_format(header_slice) {
        return probe_opened_image(file, path, format).map(Some);
    }

    if has_any_extension(path, &["png", "jpg", "jpeg"]) {
        return Err(ProbeError::InvalidImage {
            path: path.to_path_buf(),
            reason: UNRECOGNIZED_MAGIC_REASON.to_owned(),
        });
    }

    Ok(None)
}

fn resolve_single_target(
    source: &Path,
    target: &Path,
    explicit_format: Option<ImageOutputFormat>,
) -> Result<(PathBuf, ImageOutputFormat), ProbeError> {
    match fs::metadata(target) {
        Ok(meta) if meta.is_dir() => {
            let format = explicit_format.unwrap_or(ImageOutputFormat::Webp);
            let derived = derive_target_in_dir(source, target, format)?;
            return Ok((derived, format));
        }
        Ok(meta) if !meta.is_file() => {
            return Err(ProbeError::NotAFileOrDirectory(target.to_path_buf()));
        }
        Ok(_) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => {
            return Err(ProbeError::Io {
                path: target.to_path_buf(),
                source: err,
            });
        }
    }

    let inferred = infer_format_from_extension(target);
    match (explicit_format, inferred) {
        (Some(expected), Some(actual)) if expected == actual => {
            Ok((target.to_path_buf(), expected))
        }
        (Some(expected), _) => Err(ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: match expected {
                ImageOutputFormat::Webp => ".webp",
                ImageOutputFormat::Heic => ".heic",
                ImageOutputFormat::Jpeg => ".jpg or .jpeg",
            },
        }),
        (None, Some(actual)) => Ok((target.to_path_buf(), actual)),
        (None, None) => Err(ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: ".webp, .heic, .jpg, or .jpeg",
        }),
    }
}

fn infer_format_from_extension(path: &Path) -> Option<ImageOutputFormat> {
    if has_extension(path, "webp") {
        Some(ImageOutputFormat::Webp)
    } else if has_extension(path, "heic") {
        Some(ImageOutputFormat::Heic)
    } else if has_any_extension(path, &["jpg", "jpeg"]) {
        Some(ImageOutputFormat::Jpeg)
    } else {
        None
    }
}

fn derive_target_in_dir(
    source: &Path,
    target_dir: &Path,
    format: ImageOutputFormat,
) -> Result<PathBuf, ProbeError> {
    let stem = source.file_stem().ok_or_else(|| ProbeError::InvalidImage {
        path: source.to_path_buf(),
        reason: "source path has no file stem".to_owned(),
    })?;
    let mut name = stem.to_os_string();
    name.push(".");
    name.push(format.extension());
    Ok(target_dir.join(name))
}

fn has_extension(path: &Path, expected: &str) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case(expected))
}

fn has_any_extension(path: &Path, candidates: &[&str]) -> bool {
    candidates.iter().any(|ext| has_extension(path, ext))
}

fn reject_canonical_same_path(source: &Path, target: &Path) -> Result<(), ProbeError> {
    let canon_target = match fs::canonicalize(target) {
        Ok(path) => path,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => {
            return Err(ProbeError::Io {
                path: target.to_path_buf(),
                source: err,
            });
        }
    };
    let canon_source = fs::canonicalize(source).map_err(|err| ProbeError::Io {
        path: source.to_path_buf(),
        source: err,
    })?;
    if canon_source == canon_target {
        return Err(ProbeError::InvalidDomain(
            DomainError::IdenticalSourceAndTarget(source.to_path_buf()),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ProbedImage, prepare_image_batch, prepare_image_batch_inner, probe_image};
    use crate::ProbeError;
    use fastconv_domain::{
        CpuInfo, DetectedGpuBackend, DomainError, GpuBackend, GpuCapability, GpuDeviceType,
        GpuVendor, GraphicsApi, HardwareBackend, HostHardware, ImageCompression, ImageDimensions,
        ImageExecutionMode, ImageInputFormat, ImageOutputFormat, ImageQuality, PathRole, SimdLevel,
        WgpuAdapterInfo,
    };
    use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder};
    use std::env;
    use std::fs::{self, File};
    use std::io::BufWriter;
    use std::num::{NonZeroU64, NonZeroUsize};
    use std::path::{Path, PathBuf};
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn create_temp_dir(label: &str) -> Result<PathBuf, ProbeError> {
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("fastconv-probe-{label}-{}-{id}", process::id()));
        fs::create_dir_all(&dir).map_err(|source| ProbeError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(dir)
    }

    fn write_tiny_png(path: &Path, width: u32, height: u32) -> Result<(), ProbeError> {
        let file = File::create(path).map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|err| ProbeError::InvalidImage {
                path: path.to_path_buf(),
                reason: err.to_string(),
            })?;
        let data_len = usize::try_from(width * height * 3).unwrap_or(0);
        writer
            .write_image_data(&vec![128_u8; data_len])
            .map_err(|err| ProbeError::InvalidImage {
                path: path.to_path_buf(),
                reason: err.to_string(),
            })?;
        Ok(())
    }

    fn write_tiny_jpeg(path: &Path, width: u16, height: u16) -> Result<(), ProbeError> {
        let data_len = usize::from(width) * usize::from(height) * 3;
        let encoder = JpegEncoder::new_file(path, 85).map_err(|err| ProbeError::InvalidImage {
            path: path.to_path_buf(),
            reason: err.to_string(),
        })?;
        encoder
            .encode(&vec![140_u8; data_len], width, height, JpegColorType::Rgb)
            .map_err(|err| ProbeError::InvalidImage {
                path: path.to_path_buf(),
                reason: err.to_string(),
            })?;
        Ok(())
    }

    fn synthetic_rtx_hardware() -> Result<HostHardware, ProbeError> {
        let cores =
            NonZeroUsize::new(4).ok_or(ProbeError::InvalidDomain(DomainError::EmptyBatch))?;
        let threads =
            NonZeroUsize::new(8).ok_or(ProbeError::InvalidDomain(DomainError::EmptyBatch))?;
        let cpu = CpuInfo::new(
            "Test CPU",
            "x86_64",
            cores,
            threads,
            SimdLevel::X86Avx2Fma,
            NonZeroU64::new(16 * 1024 * 1024 * 1024),
        )?;
        let adapter = WgpuAdapterInfo::new(
            "NVIDIA GeForce RTX 2060",
            GpuVendor::Nvidia,
            0x1e89,
            GpuDeviceType::Discrete,
            GraphicsApi::Vulkan,
            "NVIDIA",
            "620.73.04",
        )?;
        let backends = vec![
            DetectedGpuBackend::new(
                GpuBackend::Nvenc,
                GpuCapability::VideoEncodeAndDecode,
                "NVIDIA GeForce RTX 2060",
                "/usr/lib/libnvidia-encode.so.1",
            )?,
            DetectedGpuBackend::new(
                GpuBackend::Vulkan,
                GpuCapability::ComputeOnly,
                "NVIDIA GeForce RTX 2060",
                "wgpu vulkan",
            )?,
            DetectedGpuBackend::new(
                GpuBackend::Vaapi,
                GpuCapability::VideoDecodeOnly,
                "NVIDIA GeForce RTX 2060",
                "/usr/lib/libva.so.2 + /usr/lib/dri/nvidia_drv_video.so",
            )?,
        ];
        Ok(HostHardware::new(cpu, vec![adapter], backends))
    }

    #[test]
    fn prepares_single_image_targets() -> Result<(), ProbeError> {
        let root = create_temp_dir("single")?;
        let png_path = root.join("alpha.png");
        let jpg_path = root.join("beta.JPG");
        write_tiny_png(&png_path, 16, 12)?;
        write_tiny_jpeg(&jpg_path, 32, 24)?;

        assert_eq!(
            probe_image(&png_path)?,
            ProbedImage {
                format: ImageInputFormat::Png,
                dimensions: ImageDimensions::new(16, 12)?,
            }
        );
        assert_eq!(
            probe_image(&jpg_path)?,
            ProbedImage {
                format: ImageInputFormat::Jpeg,
                dimensions: ImageDimensions::new(32, 24)?,
            }
        );

        let rtx = synthetic_rtx_hardware()?;
        let lossy = ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        };
        let single_jpeg = prepare_image_batch(
            &png_path,
            &root.join("single.jpeg"),
            lossy,
            Some(ImageOutputFormat::Jpeg),
            None,
            HardwareBackend::Cpu,
        )?;
        assert_eq!(
            single_jpeg.mode(),
            ImageExecutionMode::CpuJpeg {
                quality: ImageQuality::DEFAULT
            }
        );

        let heic_single = prepare_image_batch_inner(
            &jpg_path,
            &root.join("nvenc.heic"),
            lossy,
            None,
            None,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            Some(&rtx),
        )?;
        assert_eq!(
            heic_single.items()[0].input_format(),
            ImageInputFormat::Jpeg
        );
        assert!(matches!(
            heic_single.mode(),
            ImageExecutionMode::NvencHeic { .. }
        ));

        let out_dir = root.join("out");
        fs::create_dir_all(&out_dir).map_err(|source| ProbeError::Io {
            path: out_dir.clone(),
            source,
        })?;
        let into_dir =
            prepare_image_batch(&jpg_path, &out_dir, lossy, None, None, HardwareBackend::Cpu)?;
        assert_eq!(into_dir.items()[0].target(), out_dir.join("beta.webp"));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn prepares_mixed_directory_batches() -> Result<(), ProbeError> {
        let root = create_temp_dir("batch")?;
        let src_dir = root.join("src");
        let out_dir = root.join("out");
        fs::create_dir_all(&src_dir).map_err(|source| ProbeError::Io {
            path: src_dir.clone(),
            source,
        })?;

        let img_a = src_dir.join("alpha.png");
        let img_b = src_dir.join("beta.JPG");
        let hidden = src_dir.join(".hidden.jpg");
        let non_img = src_dir.join("notes.txt");
        write_tiny_png(&img_a, 16, 12)?;
        write_tiny_jpeg(&img_b, 32, 24)?;
        write_tiny_jpeg(&hidden, 8, 8)?;
        fs::write(&non_img, b"ignore").map_err(|source| ProbeError::Io {
            path: non_img.clone(),
            source,
        })?;

        let rtx = synthetic_rtx_hardware()?;
        let lossy = ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        };
        let heic_batch = prepare_image_batch_inner(
            &src_dir,
            &out_dir,
            lossy,
            Some(ImageOutputFormat::Heic),
            None,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            Some(&rtx),
        )?;
        assert_eq!(heic_batch.items().len(), 2);
        assert_eq!(heic_batch.items()[0].input_format(), ImageInputFormat::Png);
        assert_eq!(heic_batch.items()[0].target(), out_dir.join("alpha.heic"));
        assert_eq!(heic_batch.items()[1].input_format(), ImageInputFormat::Jpeg);
        assert_eq!(heic_batch.items()[1].target(), out_dir.join("beta.heic"));

        let jpeg_batch = prepare_image_batch(
            &src_dir,
            &out_dir,
            lossy,
            Some(ImageOutputFormat::Jpeg),
            None,
            HardwareBackend::Cpu,
        )?;
        assert_eq!(jpeg_batch.items().len(), 2);
        assert_eq!(jpeg_batch.items()[0].target(), out_dir.join("alpha.jpg"));
        assert_eq!(jpeg_batch.items()[1].target(), out_dir.join("beta.jpg"));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn probes_jpeg_via_magic_bytes_even_when_misnamed_and_rejects_corrupt_files()
    -> Result<(), ProbeError> {
        let root = create_temp_dir("magic")?;
        let misnamed = root.join("actually_jpeg.png");
        write_tiny_jpeg(&misnamed, 24, 16)?;

        let probed = probe_image(&misnamed)?;
        assert_eq!(probed.format, ImageInputFormat::Jpeg);
        assert_eq!(probed.dimensions, ImageDimensions::new(24, 16)?);

        let corrupt_jpg = root.join("corrupt.jpg");
        fs::write(&corrupt_jpg, b"\xff\xd8\xffnot-a-valid-jpeg").map_err(|source| {
            ProbeError::Io {
                path: corrupt_jpg.clone(),
                source,
            }
        })?;
        assert!(matches!(
            probe_image(&corrupt_jpg),
            Err(ProbeError::InvalidImage { .. })
        ));

        let empty_jpg = root.join("empty.jpg");
        fs::write(&empty_jpg, b"").map_err(|source| ProbeError::Io {
            path: empty_jpg.clone(),
            source,
        })?;
        assert!(matches!(
            probe_image(&empty_jpg),
            Err(ProbeError::InvalidImage { .. })
        ));

        let dir = root.join("dir_with_corrupt");
        fs::create_dir_all(&dir).map_err(|source| ProbeError::Io {
            path: dir.clone(),
            source,
        })?;
        write_tiny_png(&dir.join("good.png"), 8, 8)?;
        fs::write(dir.join("bad.jpeg"), b"not jpeg magic").map_err(|source| ProbeError::Io {
            path: dir.join("bad.jpeg"),
            source,
        })?;
        let batch_err = prepare_image_batch(
            &dir,
            &root.join("out"),
            ImageCompression::Lossy {
                quality: ImageQuality::DEFAULT,
            },
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(batch_err, Err(ProbeError::InvalidImage { .. })));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn rejects_incompatible_gpu_backends_and_format_mismatches() -> Result<(), ProbeError> {
        let root = create_temp_dir("gpu-errors")?;
        let valid = root.join("good.png");
        write_tiny_png(&valid, 8, 8)?;

        let rtx = synthetic_rtx_hardware()?;
        let lossy = ImageCompression::Lossy {
            quality: ImageQuality::DEFAULT,
        };

        let nvenc_webp = prepare_image_batch_inner(
            &valid,
            &root.join("out.webp"),
            lossy,
            None,
            None,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            Some(&rtx),
        );
        assert!(matches!(
            nvenc_webp,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedImagePipeline {
                    format: ImageOutputFormat::Webp,
                    backend: HardwareBackend::Gpu(GpuBackend::Nvenc),
                    ..
                }
            ))
        ));

        let vulkan_jpeg = prepare_image_batch_inner(
            &valid,
            &root.join("out.jpg"),
            lossy,
            None,
            None,
            HardwareBackend::Gpu(GpuBackend::Vulkan),
            Some(&rtx),
        );
        assert!(matches!(
            vulkan_jpeg,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedImagePipeline {
                    format: ImageOutputFormat::Jpeg,
                    backend: HardwareBackend::Gpu(GpuBackend::Vulkan),
                    ..
                }
            ))
        ));

        let vaapi_err = prepare_image_batch_inner(
            &valid,
            &root.join("out.webp"),
            lossy,
            None,
            None,
            HardwareBackend::Gpu(GpuBackend::Vaapi),
            Some(&rtx),
        );
        assert!(matches!(
            vaapi_err,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedImagePipeline {
                    backend: HardwareBackend::Gpu(GpuBackend::Vaapi),
                    ..
                }
            ))
        ));

        let format_mismatch = prepare_image_batch_inner(
            &valid,
            &root.join("out.webp"),
            lossy,
            Some(ImageOutputFormat::Jpeg),
            None,
            HardwareBackend::Cpu,
            Some(&rtx),
        );
        assert!(matches!(
            format_mismatch,
            Err(ProbeError::UnsupportedTargetFormat {
                expected: ".jpg or .jpeg",
                ..
            })
        ));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn rejects_corrupt_png_bad_extension_and_empty_dir() -> Result<(), ProbeError> {
        let root = create_temp_dir("errors")?;
        let valid = root.join("good.png");
        let corrupt = root.join("corrupt.png");
        let empty_dir = root.join("empty");
        fs::create_dir_all(&empty_dir).map_err(|source| ProbeError::Io {
            path: empty_dir.clone(),
            source,
        })?;
        write_tiny_png(&valid, 8, 8)?;
        fs::write(&corrupt, b"not a png").map_err(|source| ProbeError::Io {
            path: corrupt.clone(),
            source,
        })?;

        let corrupt_err = prepare_image_batch(
            &corrupt,
            &root.join("out.webp"),
            ImageCompression::Lossless,
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(corrupt_err, Err(ProbeError::InvalidImage { .. })));

        let ext_err = prepare_image_batch(
            &valid,
            &root.join("out.avif"),
            ImageCompression::Lossless,
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            ext_err,
            Err(ProbeError::UnsupportedTargetFormat { .. })
        ));

        let jpeg_lossless_err = prepare_image_batch(
            &valid,
            &root.join("out.jpg"),
            ImageCompression::Lossless,
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            jpeg_lossless_err,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedImagePipeline {
                    format: ImageOutputFormat::Jpeg,
                    ..
                }
            ))
        ));

        let empty_err = prepare_image_batch(
            &empty_dir,
            &root.join("out"),
            ImageCompression::Lossless,
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            empty_err,
            Err(ProbeError::EmptySourceDirectory(_))
        ));

        let dir_to_file = prepare_image_batch(
            &root,
            &valid,
            ImageCompression::Lossless,
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            dir_to_file,
            Err(ProbeError::TargetNotADirectory(_))
        ));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn rejects_empty_paths_duplicate_targets_and_same_file_overwrite() -> Result<(), ProbeError> {
        let root = create_temp_dir("collisions")?;
        let compression = ImageCompression::Lossless;

        assert!(matches!(
            prepare_image_batch(
                Path::new(""),
                &root.join("out.webp"),
                compression,
                None,
                None,
                HardwareBackend::Cpu,
            ),
            Err(ProbeError::InvalidDomain(DomainError::EmptyPath(
                PathRole::Source
            )))
        ));
        assert!(matches!(
            prepare_image_batch(
                &root.join("in.png"),
                Path::new(""),
                compression,
                None,
                None,
                HardwareBackend::Cpu,
            ),
            Err(ProbeError::InvalidDomain(DomainError::EmptyPath(
                PathRole::Target
            )))
        ));

        let dup_dir = root.join("dup");
        fs::create_dir_all(&dup_dir).map_err(|source| ProbeError::Io {
            path: dup_dir.clone(),
            source,
        })?;
        write_tiny_png(&dup_dir.join("photo.png"), 8, 8)?;
        write_tiny_jpeg(&dup_dir.join("photo.jpg"), 8, 8)?;
        let dup_err = prepare_image_batch(
            &dup_dir,
            &root.join("out"),
            compression,
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            dup_err,
            Err(ProbeError::InvalidDomain(DomainError::DuplicateTargetPath(
                _
            )))
        ));

        let same_webp = root.join("same.webp");
        write_tiny_png(&same_webp, 8, 8)?;
        let same_err = prepare_image_batch(
            &same_webp,
            &root.join("./same.webp"),
            compression,
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            same_err,
            Err(ProbeError::InvalidDomain(
                DomainError::IdenticalSourceAndTarget(_)
            ))
        ));

        let in_place_dir = root.join("in_place_jpeg");
        fs::create_dir_all(&in_place_dir).map_err(|source| ProbeError::Io {
            path: in_place_dir.clone(),
            source,
        })?;
        write_tiny_jpeg(&in_place_dir.join("img.jpg"), 8, 8)?;
        let in_place_err = prepare_image_batch(
            &in_place_dir,
            &in_place_dir,
            ImageCompression::Lossy {
                quality: ImageQuality::DEFAULT,
            },
            Some(ImageOutputFormat::Jpeg),
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            in_place_err,
            Err(ProbeError::InvalidDomain(
                DomainError::IdenticalSourceAndTarget(_)
            ))
        ));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }
}
