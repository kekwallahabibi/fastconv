//! System-boundary validation for still-frame extraction requests.

use std::fs;
use std::num::NonZeroUsize;
use std::path::Path;

use fastconv_domain::{
    DomainError, FrameExtractionPlan, FrameRange, FrameSelection, HardwareBackend,
    ImageCompression, ImageDimensions, ImageExecutionMode, ImageOutputFormat, VideoCodec,
    VideoStreamKind,
};

use crate::{
    ProbeError, available_cpu_threads, canonical_or_parent, probe_video, validate_source_file,
};

fn reject(path: &Path, reason: impl Into<String>) -> ProbeError {
    ProbeError::InvalidVideo {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

/// True when `source` itself sits at a path the extractor would generate
/// (`frame_<digits>.<ext>` inside the output directory). The alias check below
/// compares the source with the *directory*; this closes the gap where a file
/// named like an output frame would be overwritten by its own extraction.
fn is_generated_frame_path(
    source: &Path,
    output_dir: &Path,
    extension: &str,
) -> Result<bool, ProbeError> {
    let Some(name) = source.file_name().and_then(|name| name.to_str()) else {
        return Ok(false);
    };
    let Some(parent) = source.parent() else {
        return Ok(false);
    };
    if canonical_or_parent(parent)? != canonical_or_parent(output_dir)? {
        return Ok(false);
    }
    let Some(rest) = name.strip_prefix("frame_") else {
        return Ok(false);
    };
    let Some((digits, ext)) = rest.rsplit_once('.') else {
        return Ok(false);
    };
    Ok(ext == extension && !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
}

/// Validates a frame-extraction request once at the filesystem/media boundary.
///
/// GPUs are rejected outright: extraction is a CPU-only pipeline and a silent
/// fallback would hide an explicitly requested backend.
///
/// # Errors
///
/// Returns [`ProbeError`] for invalid sources, unsupported codecs/containers,
/// invalid output directories, alias conflicts, or incompatible encoder options.
// The probe boundary contract fixes all eight inputs (paths, format,
// compression, selection, range, threading, backend) in one validated call.
#[allow(clippy::too_many_arguments)]
pub fn prepare_frame_extraction(
    source: &Path,
    output_dir: &Path,
    output_format: ImageOutputFormat,
    compression: Option<ImageCompression>,
    selection: FrameSelection,
    range: FrameRange,
    worker_threads: Option<NonZeroUsize>,
    backend: HardwareBackend,
) -> Result<FrameExtractionPlan, ProbeError> {
    if let HardwareBackend::Gpu(backend) = backend {
        return Err(reject(
            source,
            format!(
                "frame extraction runs on CPU only; backend 'gpu:{backend}' is rejected with no silent CPU fallback"
            ),
        ));
    }
    if source.as_os_str().is_empty() {
        return Err(DomainError::EmptyPath(fastconv_domain::PathRole::Source).into());
    }
    if output_dir.as_os_str().is_empty() {
        return Err(DomainError::EmptyPath(fastconv_domain::PathRole::Target).into());
    }

    let validated_source = validate_source_file(source)?;
    let probed = probe_video(&validated_source)?;
    let video_stream = probed
        .streams
        .iter()
        .find(|stream| stream.kind() == VideoStreamKind::Video)
        .ok_or_else(|| reject(source, "source has no video stream"))?;
    if !matches!(
        video_stream.codec(),
        VideoCodec::H264
            | VideoCodec::Hevc
            | VideoCodec::Av1
            | VideoCodec::Vp9
            | VideoCodec::Mjpeg
            | VideoCodec::Gif
            | VideoCodec::Webp
    ) {
        return Err(reject(
            source,
            format!(
                "frame extraction cannot decode video codec '{}'",
                video_stream.codec()
            ),
        ));
    }
    let dimensions = match (video_stream.width(), video_stream.height()) {
        (Some(width), Some(height)) => Some(ImageDimensions::new(width.get(), height.get())?),
        _ => None,
    };

    // A supplied output directory must not itself be a regular file or a
    // symlink to somewhere that is not a directory. `metadata` follows links
    // while `symlink_metadata` sees the link itself, so both are checked.
    if let Ok(link_meta) = fs::symlink_metadata(output_dir) {
        if link_meta.file_type().is_symlink() {
            let followed = fs::metadata(output_dir).map_err(|source| ProbeError::Io {
                path: output_dir.to_path_buf(),
                source,
            })?;
            if !followed.is_dir() {
                return Err(ProbeError::TargetNotADirectory(output_dir.to_path_buf()));
            }
        } else if !link_meta.is_dir() {
            return Err(ProbeError::TargetNotADirectory(output_dir.to_path_buf()));
        }
    }

    let canonical_source = canonical_or_parent(&validated_source)?;
    let canonical_output = canonical_or_parent(output_dir)?;
    if canonical_source == canonical_output {
        return Err(reject(
            source,
            "output directory aliases the source file (in-place extraction is not allowed)",
        ));
    }
    if is_generated_frame_path(&validated_source, output_dir, output_format.extension())? {
        return Err(reject(
            source,
            "source path matches a generated frame_<n>.<ext> name in the output directory",
        ));
    }

    let mode = ImageExecutionMode::resolve_cpu(output_format, compression)?;
    let threads = match worker_threads {
        Some(threads) => threads,
        None => available_cpu_threads()?,
    };

    FrameExtractionPlan::new(
        validated_source,
        probed.format,
        dimensions,
        output_dir.to_path_buf(),
        output_format,
        mode,
        selection,
        range,
        threads,
    )
    .map_err(ProbeError::from)
}

#[cfg(test)]
mod tests {
    use super::prepare_frame_extraction;
    use crate::ProbeError;
    use fastconv_domain::{
        FrameRange, FrameSelection, GpuBackend, HardwareBackend, ImageOutputFormat,
        VideoContainerFormat,
    };
    use std::env;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process;

    /// Minimal valid 1x1 `GIF89a` with one frame (no external tools involved).
    const TINY_GIF: &[u8] = &[
        0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00, 0x80, 0x00, 0x00, 0xFF, 0xFF,
        0xFF, 0x00, 0x00, 0x00, 0x21, 0xF9, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2C, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x02, 0x02, 0x44, 0x01, 0x00, 0x3B,
    ];

    fn temp_root(label: &str) -> Result<PathBuf, ProbeError> {
        let path = env::temp_dir().join(format!("fastconv-probe-frames-{label}-{}", process::id()));
        fs::create_dir_all(&path).map_err(|source| ProbeError::Io {
            path: path.clone(),
            source,
        })?;
        Ok(path)
    }

    #[test]
    fn prepares_extraction_from_animated_gif() -> Result<(), ProbeError> {
        let root = temp_root("valid")?;
        let source = root.join("in.gif");
        fs::write(&source, TINY_GIF).map_err(|err| ProbeError::Io {
            path: source.clone(),
            source: err,
        })?;
        let out = root.join("frames");
        let plan = prepare_frame_extraction(
            &source,
            &out,
            ImageOutputFormat::Png,
            None,
            FrameSelection::All,
            FrameRange::new(0, None).map_err(ProbeError::from)?,
            None,
            HardwareBackend::Cpu,
        )?;
        assert_eq!(plan.source_container(), VideoContainerFormat::Gif);
        assert_eq!(plan.output_format(), ImageOutputFormat::Png);
        assert_eq!(plan.output_dir(), out.as_path());
        assert!(plan.worker_threads().get() >= 1);
        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn rejects_gpu_backend_before_any_io() -> Result<(), ProbeError> {
        let result = prepare_frame_extraction(
            Path::new("missing.mp4"),
            Path::new("out"),
            ImageOutputFormat::Png,
            None,
            FrameSelection::All,
            FrameRange::new(0, None).map_err(ProbeError::from)?,
            None,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
        );
        assert!(matches!(result, Err(ProbeError::InvalidVideo { .. })));
        Ok(())
    }

    #[test]
    fn rejects_non_video_inputs() -> Result<(), ProbeError> {
        let root = temp_root("reject")?;
        let png = root.join("in.png");
        fs::write(&png, b"\x89PNG\r\n\x1a\n").map_err(|source| ProbeError::Io {
            path: png.clone(),
            source,
        })?;
        let wav = root.join("in.wav");
        fs::write(&wav, b"RIFF\x00\x00\x00\x00WAVEfmt ").map_err(|source| ProbeError::Io {
            path: wav.clone(),
            source,
        })?;
        for source in [&png, &wav] {
            let result = prepare_frame_extraction(
                source,
                &root.join("out"),
                ImageOutputFormat::Png,
                None,
                FrameSelection::All,
                FrameRange::new(0, None).map_err(ProbeError::from)?,
                None,
                HardwareBackend::Cpu,
            );
            assert!(matches!(result, Err(ProbeError::InvalidVideo { .. })));
        }
        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn rejects_file_output_directory_and_source_alias() -> Result<(), ProbeError> {
        let root = temp_root("paths")?;
        let source = root.join("in.gif");
        fs::write(&source, TINY_GIF).map_err(|err| ProbeError::Io {
            path: source.clone(),
            source: err,
        })?;
        let out_file = root.join("not-a-dir");
        fs::write(&out_file, b"x").map_err(|source| ProbeError::Io {
            path: out_file.clone(),
            source,
        })?;
        let result = prepare_frame_extraction(
            &source,
            &out_file,
            ImageOutputFormat::Png,
            None,
            FrameSelection::All,
            FrameRange::new(0, None).map_err(ProbeError::from)?,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(result, Err(ProbeError::TargetNotADirectory(_))));

        let aliased = prepare_frame_extraction(
            &source,
            &source,
            ImageOutputFormat::Png,
            None,
            FrameSelection::All,
            FrameRange::new(0, None).map_err(ProbeError::from)?,
            None,
            HardwareBackend::Cpu,
        );
        // The file-valued output directory is rejected either as "not a
        // directory" or by the alias guard, never accepted.
        assert!(matches!(
            aliased,
            Err(ProbeError::InvalidVideo { .. } | ProbeError::TargetNotADirectory(_))
        ));
        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn rejects_source_named_like_a_generated_frame() -> Result<(), ProbeError> {
        let root = temp_root("frame-name")?;
        let source = root.join("frame_000001.gif");
        fs::write(&source, TINY_GIF).map_err(|err| ProbeError::Io {
            path: source.clone(),
            source: err,
        })?;
        let result = prepare_frame_extraction(
            &source,
            &root,
            ImageOutputFormat::Gif,
            None,
            FrameSelection::All,
            FrameRange::new(0, None).map_err(ProbeError::from)?,
            None,
            HardwareBackend::Cpu,
        );
        assert!(matches!(result, Err(ProbeError::InvalidVideo { .. })));

        // A different extension cannot collide with the .gif output frame.
        let other_ext = prepare_frame_extraction(
            &source,
            &root,
            ImageOutputFormat::Png,
            None,
            FrameSelection::All,
            FrameRange::new(0, None).map_err(ProbeError::from)?,
            None,
            HardwareBackend::Cpu,
        );
        assert!(other_ext.is_ok());

        // An ordinary name in the same directory stays valid.
        let ordinary = root.join("clip.gif");
        fs::write(&ordinary, TINY_GIF).map_err(|err| ProbeError::Io {
            path: ordinary.clone(),
            source: err,
        })?;
        let valid = prepare_frame_extraction(
            &ordinary,
            &root,
            ImageOutputFormat::Gif,
            None,
            FrameSelection::All,
            FrameRange::new(0, None).map_err(ProbeError::from)?,
            None,
            HardwareBackend::Cpu,
        );
        assert!(valid.is_ok());
        fs::remove_dir_all(&root).map_err(|source| ProbeError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(())
    }
}
