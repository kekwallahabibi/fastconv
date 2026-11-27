//! Exclusive, crash-safe staging files shared by every pipeline backend.
//!
//! Each writer reserves a temporary file next to the destination with
//! `create_new` and owner-only permissions, then publishes it with one atomic
//! rename. The short name deliberately omits the destination basename so a
//! user-supplied 255-byte path cannot overflow a filesystem component limit.

use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use crate::PipelineError;

/// Process-wide serial; combined with the PID it keeps concurrent invocations
/// from colliding even when several workers target the same directory.
static STAGING_SERIAL: AtomicU64 = AtomicU64::new(0);

/// A reserved temporary file that is either published or removed on drop.
pub(crate) struct StagingTarget {
    temp: PathBuf,
    target: PathBuf,
    file: Option<File>,
    published: bool,
}

impl StagingTarget {
    fn reserve(target: &Path, tag: usize) -> Result<(Self, File), PipelineError> {
        if let Some(parent) = target.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        for _ in 0..64 {
            let serial = STAGING_SERIAL.fetch_add(1, Ordering::Relaxed);
            let temp =
                target.with_file_name(format!(".fastconv-{}-{serial}-{tag}.tmp", process::id()));
            let mut options = OpenOptions::new();
            options.read(true).write(true).create_new(true);
            // Owner-only regardless of the process umask: staged media must not
            // be exposed before the final rename replaces a restrictive target.
            #[cfg(unix)]
            options.mode(0o600);
            match options.open(&temp) {
                Ok(file) => {
                    return Ok((
                        Self {
                            temp,
                            target: target.to_path_buf(),
                            file: None,
                            published: false,
                        },
                        file,
                    ));
                }
                Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {}
                Err(source) => {
                    return Err(PipelineError::Io { path: temp, source });
                }
            }
        }
        Err(PipelineError::Io {
            path: target.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                "failed to reserve a unique staging file",
            ),
        })
    }

    /// Image encoders own the `File` (possibly through a `BufWriter`).
    pub(crate) fn create_for_image(target: &Path) -> Result<(Self, File), PipelineError> {
        Self::reserve(target, 0)
    }

    /// Video backends write through the staging handle.
    pub(crate) fn create_with_handle(target: &Path, tag: usize) -> Result<Self, PipelineError> {
        let (mut staging, file) = Self::reserve(target, tag)?;
        staging.file = Some(file);
        Ok(staging)
    }

    /// Audio encoders reopen the staging path by name, so the reserved handle
    /// is closed before their closure runs.
    pub(crate) fn create_path_only(target: &Path) -> Result<Self, PipelineError> {
        let (staging, file) = Self::reserve(target, 0)?;
        drop(file);
        Ok(staging)
    }

    pub(crate) fn file(&self) -> Result<&File, PipelineError> {
        self.file
            .as_ref()
            .ok_or_else(|| PipelineError::RemuxFailed {
                path: self.target.clone(),
                reason: "staging file is already committed".to_owned(),
            })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.temp
    }

    /// Publish bytes written through `file`, preserving the destination's
    /// existing permissions when an image is being replaced.
    pub(crate) fn commit_with_permissions(self, file: File) -> Result<(), PipelineError> {
        preserve_existing_permissions(&file, &self.target)?;
        drop(file);
        self.publish()
    }

    /// Publish bytes written through the staging handle.
    pub(crate) fn commit(mut self) -> Result<(), PipelineError> {
        drop(self.file.take());
        self.publish()
    }

    fn publish(mut self) -> Result<(), PipelineError> {
        fs::rename(&self.temp, &self.target).map_err(|source| PipelineError::Io {
            path: self.target.clone(),
            source,
        })?;
        self.published = true;
        Ok(())
    }
}

impl Drop for StagingTarget {
    fn drop(&mut self) {
        // Only remove the entry this invocation created exclusively.
        if !self.published && fs::remove_file(&self.temp).is_err() {
            // Best-effort cleanup of the staging file when encoding or rename failed.
        }
    }
}

fn preserve_existing_permissions(file: &File, target: &Path) -> Result<(), PipelineError> {
    match fs::symlink_metadata(target) {
        Ok(meta) if meta.file_type().is_file() => {
            file.set_permissions(meta.permissions())
                .map_err(|source| PipelineError::Io {
                    path: target.to_path_buf(),
                    source,
                })
        }
        Ok(_) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(PipelineError::Io {
            path: target.to_path_buf(),
            source,
        }),
    }
}
