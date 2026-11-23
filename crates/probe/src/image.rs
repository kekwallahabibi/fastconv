use std::fs::{self, File};
use std::io::{self, BufReader};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use fastconv_domain::{
    DomainError, HardwareBackend, HostHardware, ImageBatchItem, ImageBatchPlan, ImageDimensions,
    ImageExecutionMode, ImageOutputFormat, PathRole, WebpEncoding,
};

use crate::{ProbeError, available_cpu_threads, detect_host_hardware};

/// Probes the PNG header at `path` without decoding pixel data.
///
/// # Errors
///
/// Returns [`ProbeError::Io`] if `path` cannot be opened, [`ProbeError::InvalidImage`]
/// if the header is not valid PNG, or [`ProbeError::InvalidDomain`] if dimensions are zero.
pub fn probe_png_dimensions(path: &Path) -> Result<ImageDimensions, ProbeError> {
    let file = File::open(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_ignore_text_chunk(true);

    let reader = decoder
        .read_info()
        .map_err(|err| ProbeError::InvalidImage {
            path: path.to_path_buf(),
            reason: err.to_string(),
        })?;
    let info = reader.info();
    Ok(ImageDimensions::new(info.width, info.height)?)
}

/// Validates a single PNG file or folder of PNGs at the boundary against live host hardware
/// and returns a compile-time-legal [`ImageBatchPlan`].
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
    encoding: WebpEncoding,
    output_format: Option<ImageOutputFormat>,
    worker_threads: Option<NonZeroUsize>,
    backend: HardwareBackend,
) -> Result<ImageBatchPlan, ProbeError> {
    prepare_image_batch_inner(
        source,
        target,
        encoding,
        output_format,
        worker_threads,
        backend,
        None,
    )
}

fn prepare_image_batch_inner(
    source: &Path,
    target: &Path,
    encoding: WebpEncoding,
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
        let mode = resolve_execution_mode(resolved_format, encoding, backend, hardware)?;
        let dims = probe_png_dimensions(source)?;
        reject_canonical_same_path(source, &resolved_target)?;
        let item = ImageBatchItem::new(source.to_path_buf(), resolved_target, dims)?;
        return Ok(ImageBatchPlan::new(vec![item], mode, threads)?);
    }

    if source_meta.is_dir() {
        let resolved_format = output_format.unwrap_or(ImageOutputFormat::Webp);
        let mode = resolve_execution_mode(resolved_format, encoding, backend, hardware)?;
        return prepare_directory_batch(source, target, mode, threads);
    }

    Err(ProbeError::NotAFileOrDirectory(source.to_path_buf()))
}

fn resolve_execution_mode(
    format: ImageOutputFormat,
    encoding: WebpEncoding,
    backend: HardwareBackend,
    hardware: Option<&HostHardware>,
) -> Result<ImageExecutionMode, ProbeError> {
    match (backend, hardware) {
        (HardwareBackend::Cpu, _) => Ok(ImageExecutionMode::resolve_cpu(format, encoding)?),
        (HardwareBackend::Gpu(_), Some(hw)) => {
            Ok(ImageExecutionMode::resolve(format, encoding, backend, hw)?)
        }
        (HardwareBackend::Gpu(_), None) => {
            let detected = detect_host_hardware()?;
            Ok(ImageExecutionMode::resolve(
                format, encoding, backend, &detected,
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

    let mut png_paths = Vec::new();
    for entry_result in entries {
        let entry = entry_result.map_err(|err| ProbeError::Io {
            path: source_dir.to_path_buf(),
            source: err,
        })?;
        let path = entry.path();
        if is_visible_png_file(&path)? {
            png_paths.push(path);
        }
    }
    png_paths.sort();

    if png_paths.is_empty() {
        return Err(ProbeError::EmptySourceDirectory(source_dir.to_path_buf()));
    }

    let format = mode.output_format();
    let mut items = Vec::with_capacity(png_paths.len());
    for src_path in png_paths {
        let dims = probe_png_dimensions(&src_path)?;
        let out_path = derive_target_in_dir(&src_path, target_dir, format)?;
        reject_canonical_same_path(&src_path, &out_path)?;
        items.push(ImageBatchItem::new(src_path, out_path, dims)?);
    }

    Ok(ImageBatchPlan::new(items, mode, threads)?)
}

fn is_visible_png_file(path: &Path) -> Result<bool, ProbeError> {
    let is_hidden = path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'));
    if is_hidden || !has_extension(path, "png") {
        return Ok(false);
    }

    let meta = fs::metadata(path).map_err(|err| ProbeError::Io {
        path: path.to_path_buf(),
        source: err,
    })?;
    Ok(meta.is_file())
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
            },
        }),
        (None, Some(actual)) => Ok((target.to_path_buf(), actual)),
        (None, None) => Err(ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: ".webp or .heic",
        }),
    }
}

fn infer_format_from_extension(path: &Path) -> Option<ImageOutputFormat> {
    if has_extension(path, "webp") {
        Some(ImageOutputFormat::Webp)
    } else if has_extension(path, "heic") {
        Some(ImageOutputFormat::Heic)
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
    use super::{prepare_image_batch, prepare_image_batch_inner, probe_png_dimensions};
    use crate::ProbeError;
    use fastconv_domain::{
        CpuInfo, DetectedGpuBackend, DomainError, GpuBackend, GpuCapability, GpuDeviceType,
        GpuVendor, GraphicsApi, HardwareBackend, HostHardware, ImageDimensions, ImageExecutionMode,
        ImageOutputFormat, ImageQuality, PathRole, SimdLevel, WebpEncoding, WgpuAdapterInfo,
    };
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
    fn prepares_single_image_and_directory_batches_for_webp_and_heic() -> Result<(), ProbeError> {
        let root = create_temp_dir("batch")?;
        let src_dir = root.join("src");
        let out_dir = root.join("out");
        fs::create_dir_all(&src_dir).map_err(|source| ProbeError::Io {
            path: src_dir.clone(),
            source,
        })?;

        let img_a = src_dir.join("alpha.png");
        let img_b = src_dir.join("beta.PNG");
        let hidden = src_dir.join(".hidden.png");
        let non_png = src_dir.join("notes.txt");
        write_tiny_png(&img_a, 16, 12)?;
        write_tiny_png(&img_b, 32, 24)?;
        write_tiny_png(&hidden, 8, 8)?;
        fs::write(&non_png, b"ignore").map_err(|source| ProbeError::Io {
            path: non_png.clone(),
            source,
        })?;

        assert_eq!(probe_png_dimensions(&img_a)?, ImageDimensions::new(16, 12)?);

        let rtx = synthetic_rtx_hardware()?;
        let encoding = WebpEncoding::Lossy {
            quality: ImageQuality::DEFAULT,
        };
        let single = prepare_image_batch(
            &img_a,
            &root.join("single.webp"),
            encoding,
            None,
            None,
            HardwareBackend::Cpu,
        )?;
        assert_eq!(single.items().len(), 1);
        assert_eq!(single.items()[0].target(), root.join("single.webp"));
        assert_eq!(single.mode(), ImageExecutionMode::CpuWebp(encoding));

        let heic_single = prepare_image_batch_inner(
            &img_a,
            &root.join("nvenc.heic"),
            encoding,
            None,
            None,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            Some(&rtx),
        )?;
        assert_eq!(
            heic_single.mode(),
            ImageExecutionMode::NvencHeic {
                quality: ImageQuality::DEFAULT
            }
        );

        fs::create_dir_all(&out_dir).map_err(|source| ProbeError::Io {
            path: out_dir.clone(),
            source,
        })?;
        let into_dir =
            prepare_image_batch(&img_a, &out_dir, encoding, None, None, HardwareBackend::Cpu)?;
        assert_eq!(into_dir.items()[0].target(), out_dir.join("alpha.webp"));

        let heic_batch = prepare_image_batch_inner(
            &src_dir,
            &out_dir,
            encoding,
            Some(ImageOutputFormat::Heic),
            None,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            Some(&rtx),
        )?;
        assert_eq!(heic_batch.items().len(), 2);
        assert_eq!(heic_batch.items()[0].target(), out_dir.join("alpha.heic"));
        assert_eq!(heic_batch.items()[1].target(), out_dir.join("beta.heic"));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn rejects_incompatible_gpu_backends_and_format_mismatches() -> Result<(), ProbeError> {
        let root = create_temp_dir("gpu-errors")?;
        let valid = root.join("good.png");
        write_tiny_png(&valid, 8, 8)?;

        let rtx = synthetic_rtx_hardware()?;
        let lossy = WebpEncoding::Lossy {
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

        let vulkan_webp = prepare_image_batch_inner(
            &valid,
            &root.join("out.webp"),
            lossy,
            None,
            None,
            HardwareBackend::Gpu(GpuBackend::Vulkan),
            Some(&rtx),
        );
        assert!(matches!(
            vulkan_webp,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedImagePipeline {
                    format: ImageOutputFormat::Webp,
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
            Some(ImageOutputFormat::Heic),
            None,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            Some(&rtx),
        );
        assert!(matches!(
            format_mismatch,
            Err(ProbeError::UnsupportedTargetFormat {
                expected: ".heic",
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
            WebpEncoding::Lossless,
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(corrupt_err, Err(ProbeError::InvalidImage { .. })));

        let ext_err = prepare_image_batch(
            &valid,
            &root.join("out.jpg"),
            WebpEncoding::Lossless,
            None,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            ext_err,
            Err(ProbeError::UnsupportedTargetFormat { .. })
        ));

        let empty_err = prepare_image_batch(
            &empty_dir,
            &root.join("out"),
            WebpEncoding::Lossless,
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
            WebpEncoding::Lossless,
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
        let encoding = WebpEncoding::Lossless;

        assert!(matches!(
            prepare_image_batch(
                Path::new(""),
                &root.join("out.webp"),
                encoding,
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
                encoding,
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
        write_tiny_png(&dup_dir.join("dup.png"), 8, 8)?;
        write_tiny_png(&dup_dir.join("dup.PNG"), 8, 8)?;
        let dup_err = prepare_image_batch(
            &dup_dir,
            &root.join("out"),
            encoding,
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
            encoding,
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

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }
}
