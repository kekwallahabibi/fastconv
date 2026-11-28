//! System boundary for `fastconv stack`: directory/argument parsing, natural
//! frame ordering, path-safety checks and target-format resolution. Every
//! external fact is validated here and converted into a validated
//! [`FrameSequencePlan`]; the pipeline never re-parses user input.

use std::cmp::Ordering;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use fastconv_domain::{
    AnimationLoop, DomainError, FrameSequencePlan, GpuBackend, HardwareBackend, HostHardware,
    ImageDimensions, StackBackendRequest, StackExecutionMode, StackFps, StackFrame,
    TargetVideoCodec, VideoContainerFormat, VideoRateControl,
};

use crate::{
    ProbeError, available_cpu_threads, canonical_or_parent, detect_host_hardware,
    image::probe_directory_entry_with_alpha_for_stack, image::probe_image_with_alpha_for_stack,
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

fn resolved_extension(format: VideoContainerFormat) -> &'static str {
    format.extension()
}

fn resolve_target_format(
    target: &Path,
    explicit: Option<VideoContainerFormat>,
) -> Result<VideoContainerFormat, ProbeError> {
    let extension = target
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .trim();
    let inferred = VideoContainerFormat::from_extension(extension);
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
            expected: "gif, webp, mp4, mkv, webm, mov, avi, or ts (pass --to <format>)",
        }),
    }
}

/// Probes host hardware only when the request can actually require NVENC:
/// an explicit NVENC backend on a video container, or `auto` for H.264/HEVC.
fn hardware_needed(
    backend: StackBackendRequest,
    codec: Option<TargetVideoCodec>,
    container: VideoContainerFormat,
) -> bool {
    if matches!(
        container,
        VideoContainerFormat::Gif | VideoContainerFormat::Webp
    ) {
        return false;
    }
    let effective = codec.unwrap_or_else(|| container.default_video_codec());
    match backend {
        StackBackendRequest::Explicit(HardwareBackend::Gpu(GpuBackend::Nvenc)) => {
            matches!(effective, TargetVideoCodec::H264 | TargetVideoCodec::Hevc)
        }
        StackBackendRequest::Auto => {
            matches!(effective, TargetVideoCodec::H264 | TargetVideoCodec::Hevc)
        }
        StackBackendRequest::Explicit(_) => false,
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
    let probed = probe_image_with_alpha_for_stack(path)?;
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
        if let Some(probed) = probe_directory_entry_with_alpha_for_stack(&entry.path())? {
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
    target_format: Option<VideoContainerFormat>,
    rate_control: VideoRateControl,
    requested_codec: Option<TargetVideoCodec>,
    explicit_loop: Option<AnimationLoop>,
    worker_threads: Option<NonZeroUsize>,
    backend: StackBackendRequest,
    nvenc_compiled: bool,
) -> Result<FrameSequencePlan, ProbeError> {
    if sources.is_empty() {
        return Err(DomainError::EmptyFrameSequence.into());
    }
    if target.as_os_str().is_empty() {
        return Err(DomainError::EmptyPath(fastconv_domain::PathRole::Target).into());
    }
    let resolved_format = resolve_target_format(target, target_format)?;
    // Reject hardware-independent option errors before touching the system, so
    // e.g. `--loop` on a video target is never masked by a DRM/procfs I/O error.
    StackExecutionMode::validate_stack_request(
        resolved_format,
        requested_codec,
        rate_control,
        explicit_loop,
        backend,
        nvenc_compiled,
    )?;
    // A CPU-only build can never select NVENC; do not probe the host at all.
    let hardware = (nvenc_compiled && hardware_needed(backend, requested_codec, resolved_format))
        .then(detect_host_hardware);
    let hardware = match hardware {
        Some(result) => Some(result?),
        None => None,
    };
    prepare_frame_sequence_with_hardware(
        sources,
        target,
        fps,
        target_format,
        rate_control,
        requested_codec,
        explicit_loop,
        worker_threads,
        backend,
        nvenc_compiled,
        hardware.as_ref(),
    )
}

/// Same validation as [`prepare_frame_sequence`] with an injected host snapshot.
/// This keeps the domain resolver's hardware decisions unit-testable without
/// touching real NVIDIA drivers.
///
/// # Errors
///
/// Returns [`ProbeError`] for the same invalid inputs as
/// [`prepare_frame_sequence`], including a domain error when the injected
/// hardware snapshot cannot satisfy an explicit NVENC request.
#[allow(clippy::too_many_arguments)] // Flat boundary input: one argument per validated CLI option.
pub fn prepare_frame_sequence_with_hardware(
    sources: &[PathBuf],
    target: &Path,
    fps: StackFps,
    target_format: Option<VideoContainerFormat>,
    rate_control: VideoRateControl,
    requested_codec: Option<TargetVideoCodec>,
    explicit_loop: Option<AnimationLoop>,
    worker_threads: Option<NonZeroUsize>,
    backend: StackBackendRequest,
    nvenc_compiled: bool,
    hardware: Option<&HostHardware>,
) -> Result<FrameSequencePlan, ProbeError> {
    if sources.is_empty() {
        return Err(DomainError::EmptyFrameSequence.into());
    }
    if target.as_os_str().is_empty() {
        return Err(DomainError::EmptyPath(fastconv_domain::PathRole::Target).into());
    }
    let resolved_format = resolve_target_format(target, target_format)?;
    // Validate codec/container/rate-control/loop/backend before probing any
    // source image headers so bad CLI flag combinations fail immediately.
    let mode = StackExecutionMode::resolve_with_hardware(
        resolved_format,
        requested_codec,
        rate_control,
        explicit_loop,
        backend,
        nvenc_compiled,
        hardware,
    )?;
    match fs::metadata(target) {
        Ok(metadata) if metadata.is_dir() => {
            return Err(ProbeError::StackTargetIsDirectory(target.to_path_buf()));
        }
        Ok(metadata) if !metadata.is_file() => {
            return Err(ProbeError::NotAFileOrDirectory(target.to_path_buf()));
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

    Ok(FrameSequencePlan::new(
        frames,
        expected,
        fps,
        target.to_path_buf(),
        mode,
        threads,
    )?)
}

#[cfg(test)]
mod tests {
    use super::{natural_cmp_bytes, natural_cmp_str};
    use crate::ProbeError;
    use crate::stack::prepare_frame_sequence_with_hardware;
    use fastconv_domain::{
        AnimationLoop, DomainError, FrameSequencePlan, HardwareBackend, StackBackendRequest,
        StackFps, StackTargetFormat, TargetVideoCodec, VideoContainerFormat, VideoQuality,
        VideoRateControl,
    };
    use std::fs;
    use std::num::{NonZeroU16, NonZeroUsize};
    use std::path::{Path, PathBuf};
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[allow(clippy::too_many_arguments)] // Mirrors the boundary function's flat argument list.
    fn prepare(
        sources: &[PathBuf],
        target: &Path,
        fps: StackFps,
        format: Option<VideoContainerFormat>,
        quality: VideoQuality,
        loop_count: AnimationLoop,
        threads: Option<NonZeroUsize>,
        backend: HardwareBackend,
    ) -> Result<FrameSequencePlan, ProbeError> {
        prepare_frame_sequence_with_hardware(
            sources,
            target,
            fps,
            format,
            VideoRateControl::Quality(quality),
            None,
            Some(loop_count),
            threads,
            StackBackendRequest::Explicit(backend),
            true,
            None,
        )
    }

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
        let plan = prepare(
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
        assert_eq!(plan.animation_format(), Some(StackTargetFormat::Webp));
        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn rejects_gpu_animation_and_conflicting_format() -> Result<(), ProbeError> {
        let root = temp_dir("reject")?;
        let a = root.join("a.png");
        write_png(&a, 16, 12)?;
        let target = root.join("out.gif");
        let gpu = prepare(
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
                DomainError::UnsupportedVideoPipeline { .. }
            ))
        ));
        let conflict = prepare(
            std::slice::from_ref(&a),
            &target,
            StackFps::DEFAULT,
            Some(VideoContainerFormat::Webp),
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
        let mixed = prepare(
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
        let alias = prepare(
            std::slice::from_ref(&a),
            &a,
            StackFps::DEFAULT,
            Some(VideoContainerFormat::Gif),
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

    #[cfg(unix)]
    #[test]
    fn rejects_non_regular_stack_target() -> Result<(), ProbeError> {
        let root = temp_dir("nonregular")?;
        let source = root.join("a.png");
        write_png(&source, 16, 12)?;
        // A Unix socket is an existing, non-regular entry that `metadata` accepts
        // but must not be replaced by the staging rename.
        let socket_path = root.join("out.gif");
        let listener = std::os::unix::net::UnixListener::bind(&socket_path).map_err(|source| {
            ProbeError::Io {
                path: socket_path.clone(),
                source,
            }
        })?;
        let result = prepare(
            std::slice::from_ref(&source),
            &socket_path,
            StackFps::DEFAULT,
            None,
            VideoQuality::DEFAULT,
            AnimationLoop::Infinite,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(result, Err(ProbeError::NotAFileOrDirectory(_))));
        drop(listener);
        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn resolves_every_video_container_and_rejects_loop() -> Result<(), ProbeError> {
        let root = temp_dir("containers")?;
        let source = root.join("frame.png");
        write_png(&source, 16, 16)?;
        for container in VideoContainerFormat::ALL {
            let target = root.join(format!("out.{}", container.extension()));
            let plan = prepare_frame_sequence_with_hardware(
                std::slice::from_ref(&source),
                &target,
                StackFps::DEFAULT,
                None,
                VideoRateControl::DEFAULT,
                None,
                None,
                None,
                StackBackendRequest::Auto,
                true,
                None,
            )?;
            assert_eq!(plan.target_container(), container);
            assert!(plan.animation_format().is_none() || !container.supports_remux());
        }
        let target = root.join("out.mp4");
        let loop_request = prepare_frame_sequence_with_hardware(
            std::slice::from_ref(&source),
            &target,
            StackFps::DEFAULT,
            None,
            VideoRateControl::DEFAULT,
            Some(TargetVideoCodec::H264),
            Some(AnimationLoop::Times(NonZeroU16::MIN)),
            None,
            StackBackendRequest::Auto,
            true,
            None,
        );
        assert!(matches!(
            loop_request,
            Err(ProbeError::InvalidDomain(
                DomainError::UnsupportedStackLoop { .. }
            ))
        ));
        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }
}
