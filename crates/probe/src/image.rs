use std::fs::{self, File};
use std::io::{self, BufReader};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use fastconv_domain::{
    DomainError, HardwareBackend, ImageBatchPlan, ImageDimensions, MediaKind, PathRole,
    PngToWebpItem, WebpEncoding,
};

use crate::{ProbeError, available_cpu_threads};

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

/// Validates a single PNG file or folder of PNGs at the boundary and returns an [`ImageBatchPlan`].
///
/// Never performs silent CPU fallbacks: requesting any GPU backend fails explicitly.
///
/// # Errors
///
/// Returns [`ProbeError`] on missing/corrupt inputs, incompatible target paths, empty source
/// directories, or unsupported backends.
pub fn prepare_image_batch(
    source: &Path,
    target: &Path,
    encoding: WebpEncoding,
    worker_threads: Option<NonZeroUsize>,
    backend: HardwareBackend,
) -> Result<ImageBatchPlan, ProbeError> {
    if matches!(backend, HardwareBackend::Gpu(_)) {
        return Err(ProbeError::InvalidDomain(DomainError::UnsupportedBackend {
            media_kind: MediaKind::Image,
            backend,
        }));
    }
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
        let dims = probe_png_dimensions(source)?;
        let resolved_target = resolve_single_target(source, target)?;
        reject_canonical_same_path(source, &resolved_target)?;
        let item = PngToWebpItem::new(source.to_path_buf(), resolved_target, dims)?;
        return Ok(ImageBatchPlan::new(vec![item], encoding, threads)?);
    }

    if source_meta.is_dir() {
        return prepare_directory_batch(source, target, encoding, threads);
    }

    Err(ProbeError::NotAFileOrDirectory(source.to_path_buf()))
}

fn prepare_directory_batch(
    source_dir: &Path,
    target_dir: &Path,
    encoding: WebpEncoding,
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

    let mut items = Vec::with_capacity(png_paths.len());
    for src_path in png_paths {
        let dims = probe_png_dimensions(&src_path)?;
        let out_path = derive_webp_in_dir(&src_path, target_dir)?;
        reject_canonical_same_path(&src_path, &out_path)?;
        items.push(PngToWebpItem::new(src_path, out_path, dims)?);
    }

    Ok(ImageBatchPlan::new(items, encoding, threads)?)
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

fn resolve_single_target(source: &Path, target: &Path) -> Result<PathBuf, ProbeError> {
    match fs::metadata(target) {
        Ok(meta) if meta.is_dir() => return derive_webp_in_dir(source, target),
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

    if !has_extension(target, "webp") {
        return Err(ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: "webp",
        });
    }
    Ok(target.to_path_buf())
}

fn derive_webp_in_dir(source: &Path, target_dir: &Path) -> Result<PathBuf, ProbeError> {
    let stem = source.file_stem().ok_or_else(|| ProbeError::InvalidImage {
        path: source.to_path_buf(),
        reason: "source path has no file stem".to_owned(),
    })?;
    let mut name = stem.to_os_string();
    name.push(".webp");
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
    use super::{prepare_image_batch, probe_png_dimensions};
    use crate::ProbeError;
    use fastconv_domain::{
        DomainError, GpuBackend, HardwareBackend, ImageDimensions, PathRole, WebpEncoding,
        WebpQuality,
    };
    use std::env;
    use std::fs::{self, File};
    use std::io::BufWriter;
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

    #[test]
    fn prepares_single_image_and_directory_batch() -> Result<(), ProbeError> {
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

        let encoding = WebpEncoding::Lossy {
            quality: WebpQuality::DEFAULT,
        };
        let single = prepare_image_batch(
            &img_a,
            &root.join("single.webp"),
            encoding,
            None,
            HardwareBackend::Cpu,
        )?;
        assert_eq!(single.items().len(), 1);
        assert_eq!(single.items()[0].target(), root.join("single.webp"));

        fs::create_dir_all(&out_dir).map_err(|source| ProbeError::Io {
            path: out_dir.clone(),
            source,
        })?;
        let into_dir = prepare_image_batch(&img_a, &out_dir, encoding, None, HardwareBackend::Cpu)?;
        assert_eq!(into_dir.items()[0].target(), out_dir.join("alpha.webp"));

        let batch = prepare_image_batch(&src_dir, &out_dir, encoding, None, HardwareBackend::Cpu)?;
        assert_eq!(batch.items().len(), 2);
        assert_eq!(batch.items()[0].target(), out_dir.join("alpha.webp"));
        assert_eq!(batch.items()[1].target(), out_dir.join("beta.webp"));

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io { path: root, source })?;
        Ok(())
    }

    #[test]
    fn rejects_corrupt_png_bad_targets_and_gpu_backend() -> Result<(), ProbeError> {
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

        let encoding = WebpEncoding::Lossless;

        let gpu_err = prepare_image_batch(
            &valid,
            &root.join("out.webp"),
            encoding,
            None,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
        );
        assert!(matches!(
            gpu_err,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedBackend { .. }
            ))
        ));

        let corrupt_err = prepare_image_batch(
            &corrupt,
            &root.join("out.webp"),
            encoding,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(corrupt_err, Err(ProbeError::InvalidImage { .. })));

        let ext_err = prepare_image_batch(
            &valid,
            &root.join("out.jpg"),
            encoding,
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
            encoding,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            empty_err,
            Err(ProbeError::EmptySourceDirectory(_))
        ));

        let dir_to_file = prepare_image_batch(&root, &valid, encoding, None, HardwareBackend::Cpu);
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
