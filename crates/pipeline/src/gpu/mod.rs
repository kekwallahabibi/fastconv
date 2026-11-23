pub mod heic;
pub mod nvenc;

use fastconv_domain::{ExecutionPlan, GpuBackend, HardwareBackend};

use crate::PipelineError;

/// Concrete GPU zero-copy execution backend entry point for video plans.
///
/// # Errors
///
/// Returns [`PipelineError::BackendNotImplemented`] until video GPU pipelines are
/// integrated. Never silently falls back to CPU.
pub const fn execute(backend: GpuBackend, _plan: &ExecutionPlan) -> Result<(), PipelineError> {
    Err(PipelineError::BackendNotImplemented(HardwareBackend::Gpu(
        backend,
    )))
}
