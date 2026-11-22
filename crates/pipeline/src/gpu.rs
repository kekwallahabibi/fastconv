use fastconv_domain::{ExecutionPlan, GpuBackend, HardwareBackend};

use crate::PipelineError;

/// Concrete GPU zero-copy execution backend entry point.
///
/// # Errors
///
/// Returns [`PipelineError::BackendNotImplemented`] until the requested GPU backend is
/// integrated. Never silently falls back to CPU.
pub const fn execute(backend: GpuBackend, _plan: &ExecutionPlan) -> Result<(), PipelineError> {
    Err(PipelineError::BackendNotImplemented(HardwareBackend::Gpu(
        backend,
    )))
}
