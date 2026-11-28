//! System boundary for `fastconv stack`: directory/argument parsing, natural
//! frame ordering, path-safety checks and target-format resolution. Every
//! external fact is validated here and converted into a validated
//! [`FrameSequencePlan`]; the pipeline never re-parses user input.

use std::cmp::Ordering;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use fastconv_domain::{
    AnimationLoop, DomainError, FrameSequencePlan, HardwareBackend, ImageDimensions,
    StackExecutionMode, StackFps, StackFrame, StackTargetFormat, VideoQuality,
};

use crate::{
    ProbeError, available_cpu_threads, canonical_or_parent,
    image::probe_directory_entry_with_alpha, image::probe_image_with_alpha,
};

/// Bounds of the significant part of a digit run (leading zeros stripped).
#[must_use]
fn significant_digits(bytes: &[u8], start: usize) -> (usize, usize) {
    let mut end = start;
    while bytes.get(end).is_some_and(u8::is_ascii_digit) {
        end += 1;
    }
    let mut first = start;
    while first + 1 < end && bytes[first] == b'0' {
        first += 1;
    }
    (first, end)
}

/// Natural byte-wise comparison: ASCII digit runs compare numerically without
/// any integer conversion, so arbitrarily long runs cannot overflow. Leading
/// zeros and non-UTF-8 bytes are preserved by a final raw-byte tie-break, which
/// keeps this a total order (reflexive, antisymmetric and transitive).
#[must_use]
pub fn natural_cmp_bytes(a: &[u8], b: &[u8]) -> Ordering {
    let mut i = 0_usize;
    let mut j = 0_usize;
    while i < a.len() && j < b.len() {
        if a[i].is_ascii_digit() && b[j].is_ascii_digit() {
            let (a_first, a_end) = significant_digits(a, i);
            let (b_first, b_end) = significant_digits(b, j);
            let significant = &a[a_first..a_end];
            let other = &b[b_first..b_end];
            let numeric = significant
                .len()
                .cmp(&other.len())
                .then_with(|| significant.cmp(other));
            if numeric != Ordering::Equal {
                return numeric;
            }
            i = a_end;
            j = b_end;
        } else {
            match a[i].cmp(&b[j]) {
                Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
                other => return other,
            }
        }
    }
    a[i..].cmp(&b[j..]).then_with(|| a.cmp(b))
}

/// Natural comparison for UTF-8 strings; see [`natural_cmp_bytes`].
#[must_use]
pub fn natural_cmp_str(a: &str, b: &str) -> Ordering {
    natural_cmp_bytes(a.as_bytes(), b.as_bytes())
}

fn resolved_extension(format: StackTargetFormat) -> &'static str {
    format.extension()
}

fn resolve_target_format(
    target: &Path,
    explicit: Option<StackTargetFormat>,
) -> Result<StackTargetFormat, ProbeError> {
    let extension = target
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default();
    let inferred = StackTargetFormat::from_extension(extension);
    match (explicit, inferred) {
        (Some(requested), Some(in_view)) if requested != in_view => {
            Err(ProbeError::ConflictingStackTargetFormat {
                path: target.to_path_buf(),
                explicit: resolved_extension(requested),
                inferred: resolved_extension(in_view),
            })
        }
        (Some(requested), _) | (None, Some(requested)) => Ok(requested),
        (None, None) => Err(ProbeError::UnsupportedTargetFormat {
            path: target.to_path_buf(),
            expected: "gif or webp (pass --to <gif|webp>)",
        }),
    }
}

fn probe_single_file(path: &Path) -> Result<(StackFrame, ImageDimensions), ProbeError> {
    let metadata = fs::metadata(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(ProbeError::NotAFile(path.to_path_buf()));
    }
    let probed = probe_image_with_alpha(path)?;
    Ok((
        StackFrame::new(path.to_path_buf(), probed.format, probed.has_alpha)?,
        probed.dimensions,
    ))
}

fn probe_source_directory(
    source_dir: &Path,
) -> Result<Vec<(StackFrame, ImageDimensions)>, ProbeError> {
    let entries = fs::read_dir(source_dir).map_err(|source| ProbeError::Io {
        path: source_dir.to_path_buf(),
        source,
    })?;
    let mut probed_entries = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| ProbeError::Io {
            path: source_dir.to_path_buf(),
            source,
        })?;
        if let Some(probed) = probe_directory_entry_with_alpha(&entry.path())? {
            probed_entries.push((entry.path(), probed));
        }
    }
    probed_entries.sort_by(|(path_a, _), (path_b, _)| {
        let name_a = path_a
            .file_name()
            .map_or(<&[u8]>::default(), std::ffi::OsStr::as_encoded_bytes);
        let name_b = path_b
            .file_name()
            .map_or(<&[u8]>::default(), std::ffi::OsStr::as_encoded_bytes);
        natural_cmp_bytes(name_a, name_b).then_with(|| {
            path_a
                .as_os_str()
                .as_encoded_bytes()
                .cmp(path_b.as_os_str().as_encoded_bytes())
        })
    });
    if probed_entries.is_empty() {
        return Err(ProbeError::EmptySourceDirectory(source_dir.to_path_buf()));
    }
    probed_entries
        .into_iter()
        .map(|(path, probed)| {
            Ok((
                StackFrame::new(path, probed.format, probed.has_alpha)?,
                probed.dimensions,
            ))
        })
        .collect()
}

fn reject_aliases(
    sources: &[StackFrame],
    source_dir: Option<&Path>,
    target: &Path,
) -> Result<(), ProbeError> {
    let canonical_target = canonical_or_parent(target)?;
    for frame in sources {
        if canonical_or_parent(frame.source())? == canonical_target {
            return Err(ProbeError::InvalidDomain(
                DomainError::StackTargetOverlapsSource(target.to_path_buf()),
            ));
        }
    }
    if let Some(dir) = source_dir
        && canonical_or_parent(dir)? == canonical_target
    {
        return Err(ProbeError::InvalidDomain(
            DomainError::StackTargetOverlapsSource(target.to_path_buf()),
        ));
    }
    Ok(())
}

/// Validates stack inputs once at the boundary and returns a compile-time-legal
/// [`FrameSequencePlan`].
///
/// A single source path is either a directory (scanned in natural numeric
/// order, skipping hidden entries and subdirectories) or a one-frame image.
/// Two or more paths are treated as an explicit, ordered frame list where
/// duplicate paths are intentional. GPU backends are rejected with no silent
/// CPU fallback.
///
/// # Errors
///
/// Returns [`ProbeError`] for missing/corrupt/animated inputs, mixed frame
/// dimensions, an empty source directory, an existing directory target, a
/// target that aliases any source, or an unsupported output format/backend.
#[allow(clippy::too_many_arguments)] // Flat boundary input: one argument per validated CLI option.
pub fn prepare_frame_sequence(
    sources: &[PathBuf],
    target: &Path,
    fps: StackFps,
    target_format: Option<StackTargetFormat>,
    quality: VideoQuality,
    loop_count: AnimationLoop,
    worker_threads: Option<NonZeroUsize>,
    backend: HardwareBackend,
) -> Result<FrameSequencePlan, ProbeError> {
    if sources.is_empty() {
        return Err(DomainError::EmptyFrameSequence.into());
    }
    if let HardwareBackend::Gpu(gpu) = backend {
        return Err(DomainError::UnsupportedStackBackend {
            backend: HardwareBackend::Gpu(gpu),
        }
        .into());
    }
    if target.as_os_str().is_empty() {
        return Err(DomainError::EmptyPath(fastconv_domain::PathRole::Target).into());
    }
    let resolved_format = resolve_target_format(target, target_format)?;
    match fs::metadata(target) {
        Ok(metadata) if metadata.is_dir() => {
            return Err(ProbeError::StackTargetIsDirectory(target.to_path_buf()));
        }
        Ok(_) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(ProbeError::Io {
                path: target.to_path_buf(),
                source,
            });
        }
    }

    let threads = match worker_threads {
        Some(count) => count,
        None => available_cpu_threads()?,
    };

    let (probed_frames, source_dir) = if sources.len() == 1 {
        let source = &sources[0];
        let metadata = fs::metadata(source).map_err(|err| ProbeError::Io {
            path: source.clone(),
            source: err,
        })?;
        if metadata.is_dir() {
            (probe_source_directory(source)?, Some(source.as_path()))
        } else if metadata.is_file() {
            (vec![probe_single_file(source)?], None)
        } else {
            return Err(ProbeError::NotAFileOrDirectory(source.clone()));
        }
    } else {
        let frames = sources
            .iter()
            .map(|path| probe_single_file(path))
            .collect::<Result<Vec<_>, _>>()?;
        (frames, None)
    };

    let expected = probed_frames[0].1;
    for (frame, actual) in probed_frames.iter().skip(1) {
        if *actual != expected {
            return Err(ProbeError::StackFrameDimensionMismatch {
                path: frame.source().to_path_buf(),
                expected,
                actual: *actual,
            });
        }
    }
    let frames = probed_frames
        .into_iter()
        .map(|(frame, _)| frame)
        .collect::<Vec<_>>();

    reject_aliases(&frames, source_dir, target)?;

    let mode = StackExecutionMode::cpu_animation(resolved_format, quality);
    Ok(FrameSequencePlan::new(
        frames,
        expected,
        fps,
        target.to_path_buf(),
        mode,
        loop_count,
        threads,
    )?)
}

#[cfg(test)]
mod tests {
    use super::{natural_cmp_bytes, natural_cmp_str, prepare_frame_sequence};
    use crate::ProbeError;
    use fastconv_domain::{
        AnimationLoop, DomainError, HardwareBackend, StackFps, StackTargetFormat, VideoQuality,
    };
    use std::fs;
    use std::num::NonZeroUsize;
    use std::path::{Path, PathBuf};
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(label: &str) -> Result<PathBuf, ProbeError> {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("fastconv-stack-{label}-{}-{id}", process::id()));
        fs::create_dir_all(&dir).map_err(|source| ProbeError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(dir)
    }

    fn invalid_alpha_image(path: &Path, reason: String) -> ProbeError {
        ProbeError::InvalidImage {
            path: path.to_path_buf(),
            reason,
        }
    }

    fn write_png(path: &Path, width: u32, height: u32) -> Result<(), ProbeError> {
        let file = fs::File::create(path).map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|err| ProbeError::InvalidImage {
                path: path.to_path_buf(),
                reason: err.to_string(),
            })?;
        writer
            .write_image_data(&vec![64_u8; (width * height * 3) as usize])
            .map_err(|err| ProbeError::InvalidImage {
                path: path.to_path_buf(),
                reason: err.to_string(),
            })
    }

    #[test]
    fn natural_sort_orders_numbers_without_overflow() {
        assert_eq!(
            natural_cmp_str("frame_2.png", "frame_10.png"),
            std::cmp::Ordering::Less
        );
        assert_eq!(natural_cmp_str("01", "1"), std::cmp::Ordering::Less);
        assert_eq!(natural_cmp_str("a", "a"), std::cmp::Ordering::Equal);
        let huge = "9".repeat(200);
        assert_eq!(natural_cmp_str(&huge, "10"), std::cmp::Ordering::Greater);
        assert_eq!(
            natural_cmp_bytes(b"p\xFF2", b"p\xFF10"),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn natural_sort_is_a_total_order() {
        let items = [
            "01", "1", "001", "2", "10", "0002", "a01b2", "a1b02", "", "\u{FF}9",
        ];
        for a in items {
            assert_eq!(natural_cmp_str(a, a), std::cmp::Ordering::Equal);
            for b in items {
                let ab = natural_cmp_str(a, b);
                let ba = natural_cmp_str(b, a);
                assert_eq!(ab, ba.reverse(), "antisymmetry for {a:?} vs {b:?}");
                for c in items {
                    let bc = natural_cmp_str(b, c);
                    let ac = natural_cmp_str(a, c);
                    if ab != std::cmp::Ordering::Greater && bc != std::cmp::Ordering::Greater {
                        assert_ne!(
                            ac,
                            std::cmp::Ordering::Greater,
                            "transitivity for {a:?} {b:?} {c:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn probes_alpha_signals_across_image_formats() -> Result<(), ProbeError> {
        let root = temp_dir("alpha")?;
        let rgb_path = root.join("rgb.png");
        write_png(&rgb_path, 4, 4)?;
        assert!(!crate::probe_image_with_alpha(&rgb_path)?.has_alpha);

        let rgba_path = root.join("rgba.png");
        {
            let file = fs::File::create(&rgba_path).map_err(|source| ProbeError::Io {
                path: rgba_path.clone(),
                source,
            })?;
            let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), 4, 4);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder
                .write_header()
                .map_err(|err| invalid_alpha_image(&rgba_path, err.to_string()))?;
            writer
                .write_image_data(&[255_u8; 4 * 4 * 4])
                .map_err(|err| invalid_alpha_image(&rgba_path, err.to_string()))?;
        }
        assert!(crate::probe_image_with_alpha(&rgba_path)?.has_alpha);

        // QOI stores its channel count in the 14-byte header.
        let qoi_path = root.join("alpha.qoi");
        let mut qoi = b"qoif".to_vec();
        qoi.extend_from_slice(&4_u32.to_be_bytes());
        qoi.extend_from_slice(&4_u32.to_be_bytes());
        qoi.extend_from_slice(&[4, 0]);
        qoi.push(0xC1); // one encoded pixel (minimum viable QOI stream)
        qoi.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]);
        fs::write(&qoi_path, &qoi).map_err(|source| ProbeError::Io {
            path: qoi_path.clone(),
            source,
        })?;
        assert!(crate::probe_image_with_alpha(&qoi_path)?.has_alpha);

        // GIF's graphic-control extension carries a transparency flag.
        let gif_path = root.join("transparent.gif");
        {
            let file = fs::File::create(&gif_path).map_err(|source| ProbeError::Io {
                path: gif_path.clone(),
                source,
            })?;
            let mut encoder =
                gif::Encoder::new(std::io::BufWriter::new(file), 1, 1, &[255, 0, 0, 0, 255, 0])
                    .map_err(|err| invalid_alpha_image(&gif_path, err.to_string()))?;
            encoder
                .write_frame(&gif::Frame::from_rgba_speed(1, 1, &mut [0_u8, 0, 0, 0], 10))
                .map_err(|err| invalid_alpha_image(&gif_path, err.to_string()))?;
        }
        assert!(crate::probe_image_with_alpha(&gif_path)?.has_alpha);

        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn prepares_ordered_multi_file_stack() -> Result<(), ProbeError> {
        let root = temp_dir("ordered")?;
        let a = root.join("a.png");
        let b = root.join("b.png");
        let target = root.join("out.webp");
        write_png(&a, 16, 12)?;
        write_png(&b, 16, 12)?;
        let plan = prepare_frame_sequence(
            &[b.clone(), a.clone(), b.clone()],
            &target,
            StackFps::DEFAULT,
            None,
            VideoQuality::DEFAULT,
            AnimationLoop::Infinite,
            NonZeroUsize::new(2),
            HardwareBackend::Cpu,
        )?;
        assert_eq!(plan.frames().len(), 3);
        assert_eq!(plan.frames()[0].source(), b);
        assert_eq!(plan.frames()[1].source(), a);
        assert_eq!(plan.frames()[2].source(), b);
        assert_eq!(plan.target_format(), StackTargetFormat::Webp);
        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn rejects_gpu_backend_and_conflicting_format() -> Result<(), ProbeError> {
        let root = temp_dir("reject")?;
        let a = root.join("a.png");
        write_png(&a, 16, 12)?;
        let target = root.join("out.gif");
        let gpu = prepare_frame_sequence(
            std::slice::from_ref(&a),
            &target,
            StackFps::DEFAULT,
            None,
            VideoQuality::DEFAULT,
            AnimationLoop::Infinite,
            None,
            HardwareBackend::Gpu(fastconv_domain::GpuBackend::Nvenc),
        );
        assert!(matches!(
            gpu,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedStackBackend { .. }
            ))
        ));
        let conflict = prepare_frame_sequence(
            std::slice::from_ref(&a),
            &target,
            StackFps::DEFAULT,
            Some(StackTargetFormat::Webp),
            VideoQuality::DEFAULT,
            AnimationLoop::Infinite,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            conflict,
            Err(ProbeError::ConflictingStackTargetFormat { .. })
        ));
        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn rejects_mixed_dimensions_and_aliasing_target() -> Result<(), ProbeError> {
        let root = temp_dir("mixed")?;
        let a = root.join("a.png");
        let b = root.join("b.png");
        write_png(&a, 16, 12)?;
        write_png(&b, 20, 12)?;
        let target = root.join("out.gif");
        let mixed = prepare_frame_sequence(
            &[a.clone(), b.clone()],
            &target,
            StackFps::DEFAULT,
            None,
            VideoQuality::DEFAULT,
            AnimationLoop::Infinite,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            mixed,
            Err(ProbeError::StackFrameDimensionMismatch { .. })
        ));
        let alias = prepare_frame_sequence(
            std::slice::from_ref(&a),
            &a,
            StackFps::DEFAULT,
            Some(StackTargetFormat::Gif),
            VideoQuality::DEFAULT,
            AnimationLoop::Infinite,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(
            alias,
            Err(ProbeError::InvalidDomain(
                DomainError::StackTargetOverlapsSource(_)
            ))
        ));
        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }
}
