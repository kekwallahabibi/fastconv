use std::fmt;
use std::num::{NonZeroU64, NonZeroUsize};

use serde::Serialize;

use crate::{DomainError, GpuBackend, require_non_empty};

/// Highest SIMD instruction set detected on the host CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum SimdLevel {
    Scalar,
    X86Sse2,
    X86Sse42,
    X86Avx,
    X86Avx2,
    X86Avx2Fma,
    X86Avx512,
    ArmNeon,
}

impl fmt::Display for SimdLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Scalar => "scalar",
            Self::X86Sse2 => "x86:sse2",
            Self::X86Sse42 => "x86:sse4.2",
            Self::X86Avx => "x86:avx",
            Self::X86Avx2 => "x86:avx2",
            Self::X86Avx2Fma => "x86:avx2+fma",
            Self::X86Avx512 => "x86:avx512",
            Self::ArmNeon => "arm:neon",
        };
        f.write_str(label)
    }
}

/// Validated CPU topology, SIMD level, and host system memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CpuInfo {
    model_name: String,
    architecture: String,
    physical_cores: NonZeroUsize,
    logical_threads: NonZeroUsize,
    simd_level: SimdLevel,
    total_memory_bytes: Option<NonZeroU64>,
}

impl CpuInfo {
    /// Creates a validated [`CpuInfo`] descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyField`] if `model_name` or `architecture` is blank,
    /// or [`DomainError::InvalidCpuTopology`] if `logical_threads < physical_cores`.
    pub fn new(
        model_name: &str,
        architecture: &str,
        physical_cores: NonZeroUsize,
        logical_threads: NonZeroUsize,
        simd_level: SimdLevel,
        total_memory_bytes: Option<NonZeroU64>,
    ) -> Result<Self, DomainError> {
        let model_name = require_non_empty(model_name, "cpu.model_name")?;
        let architecture = require_non_empty(architecture, "cpu.architecture")?;
        if logical_threads < physical_cores {
            return Err(DomainError::InvalidCpuTopology {
                physical_cores,
                logical_threads,
            });
        }

        Ok(Self {
            model_name,
            architecture,
            physical_cores,
            logical_threads,
            simd_level,
            total_memory_bytes,
        })
    }

    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    #[must_use]
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    #[must_use]
    pub const fn physical_cores(&self) -> NonZeroUsize {
        self.physical_cores
    }

    #[must_use]
    pub const fn logical_threads(&self) -> NonZeroUsize {
        self.logical_threads
    }

    #[must_use]
    pub const fn simd_level(&self) -> SimdLevel {
        self.simd_level
    }

    #[must_use]
    pub const fn total_memory_bytes(&self) -> Option<NonZeroU64> {
        self.total_memory_bytes
    }
}

/// GPU hardware vendor derived from PCI vendor IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other(u32),
}

impl GpuVendor {
    /// Maps a PCI vendor ID into a [`GpuVendor`].
    #[must_use]
    pub const fn from_pci_id(id: u32) -> Self {
        match id {
            0x10de => Self::Nvidia,
            0x1002 => Self::Amd,
            0x8086 => Self::Intel,
            0x106b => Self::Apple,
            other => Self::Other(other),
        }
    }
}

impl fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nvidia => f.write_str("nvidia"),
            Self::Amd => f.write_str("amd"),
            Self::Intel => f.write_str("intel"),
            Self::Apple => f.write_str("apple"),
            Self::Other(id) => write!(f, "other(0x{id:04x})"),
        }
    }
}

/// Validated description of a GPU adapter discovered from host sysfs/DRM data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GpuAdapterInfo {
    name: String,
    vendor: GpuVendor,
    device_id: u32,
    driver: String,
    driver_info: String,
}

impl GpuAdapterInfo {
    /// Creates a validated [`GpuAdapterInfo`] record.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyField`] if `name` is blank.
    pub fn new(
        name: &str,
        vendor: GpuVendor,
        device_id: u32,
        driver: &str,
        driver_info: &str,
    ) -> Result<Self, DomainError> {
        let name = require_non_empty(name, "gpu_adapter.name")?;
        Ok(Self {
            name,
            vendor,
            device_id,
            driver: driver.trim().to_owned(),
            driver_info: driver_info.trim().to_owned(),
        })
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub const fn vendor(&self) -> GpuVendor {
        self.vendor
    }

    #[must_use]
    pub const fn device_id(&self) -> u32 {
        self.device_id
    }

    #[must_use]
    pub fn driver(&self) -> &str {
        &self.driver
    }

    #[must_use]
    pub fn driver_info(&self) -> &str {
        &self.driver_info
    }
}

/// Verified video/compute capability level of a detected GPU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum GpuCapability {
    ComputeOnly,
    VideoDecodeOnly,
    VideoEncodeOnly,
    VideoEncodeAndDecode,
}

impl GpuCapability {
    #[must_use]
    pub const fn supports_video_encode(self) -> bool {
        matches!(self, Self::VideoEncodeOnly | Self::VideoEncodeAndDecode)
    }

    #[must_use]
    pub const fn supports_video_decode(self) -> bool {
        matches!(self, Self::VideoDecodeOnly | Self::VideoEncodeAndDecode)
    }
}

impl fmt::Display for GpuCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::ComputeOnly => "compute",
            Self::VideoDecodeOnly => "decode-only",
            Self::VideoEncodeOnly => "encode-only",
            Self::VideoEncodeAndDecode => "encode+decode",
        };
        f.write_str(label)
    }
}

/// Validated GPU acceleration backend detected on the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DetectedGpuBackend {
    backend: GpuBackend,
    capability: GpuCapability,
    device_name: String,
    evidence: String,
}

impl DetectedGpuBackend {
    /// Creates a validated [`DetectedGpuBackend`].
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyField`] if `device_name` or `evidence` is blank.
    pub fn new(
        backend: GpuBackend,
        capability: GpuCapability,
        device_name: &str,
        evidence: &str,
    ) -> Result<Self, DomainError> {
        let device_name = require_non_empty(device_name, "gpu_backend.device_name")?;
        let evidence = require_non_empty(evidence, "gpu_backend.evidence")?;
        Ok(Self {
            backend,
            capability,
            device_name,
            evidence,
        })
    }

    #[must_use]
    pub const fn backend(&self) -> GpuBackend {
        self.backend
    }

    #[must_use]
    pub const fn capability(&self) -> GpuCapability {
        self.capability
    }

    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    #[must_use]
    pub fn evidence(&self) -> &str {
        &self.evidence
    }
}

/// Complete, validated snapshot of the host CPU, GPU adapters, and native GPU backends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HostHardware {
    cpu: CpuInfo,
    gpu_adapters: Vec<GpuAdapterInfo>,
    gpu_backends: Vec<DetectedGpuBackend>,
}

impl HostHardware {
    #[must_use]
    pub const fn new(
        cpu: CpuInfo,
        gpu_adapters: Vec<GpuAdapterInfo>,
        gpu_backends: Vec<DetectedGpuBackend>,
    ) -> Self {
        Self {
            cpu,
            gpu_adapters,
            gpu_backends,
        }
    }

    #[must_use]
    pub const fn cpu(&self) -> &CpuInfo {
        &self.cpu
    }

    #[must_use]
    pub fn gpu_adapters(&self) -> &[GpuAdapterInfo] {
        &self.gpu_adapters
    }

    #[must_use]
    pub fn gpu_backends(&self) -> &[DetectedGpuBackend] {
        &self.gpu_backends
    }

    #[must_use]
    pub fn has_hardware_gpu(&self) -> bool {
        !self.gpu_adapters.is_empty() || !self.gpu_backends.is_empty()
    }

    #[must_use]
    pub fn supports_video_encode(&self, backend: GpuBackend) -> bool {
        self.gpu_backends
            .iter()
            .any(|entry| entry.backend() == backend && entry.capability().supports_video_encode())
    }

    /// Returns the best matching detected entry for `backend`, preferring encode-capable
    /// devices on hybrid multi-GPU hosts.
    #[must_use]
    pub fn find_gpu_backend(&self, backend: GpuBackend) -> Option<&DetectedGpuBackend> {
        let mut fallback = None;
        for entry in &self.gpu_backends {
            if entry.backend() == backend {
                if entry.capability().supports_video_encode() {
                    return Some(entry);
                }
                if fallback.is_none() {
                    fallback = Some(entry);
                }
            }
        }
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CpuInfo, DetectedGpuBackend, GpuAdapterInfo, GpuCapability, GpuVendor, HostHardware,
        SimdLevel,
    };
    use crate::{DomainError, GpuBackend};
    use std::num::{NonZeroU64, NonZeroUsize};

    #[test]
    fn constructs_valid_host_hardware() -> Result<(), DomainError> {
        let physical = NonZeroUsize::new(4).ok_or(DomainError::EmptyField("test"))?;
        let logical = NonZeroUsize::new(8).ok_or(DomainError::EmptyField("test"))?;
        let mem = NonZeroU64::new(32 * 1024 * 1024 * 1024);

        let cpu = CpuInfo::new(
            "Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz",
            "x86_64",
            physical,
            logical,
            SimdLevel::X86Avx2Fma,
            mem,
        )?;

        let adapter = GpuAdapterInfo::new(
            "NVIDIA GeForce RTX 2060",
            GpuVendor::from_pci_id(0x10de),
            0x1e89,
            "nvidia",
            "590.48.01",
        )?;

        let nvenc = DetectedGpuBackend::new(
            GpuBackend::Nvenc,
            GpuCapability::VideoEncodeAndDecode,
            "NVIDIA GeForce RTX 2060",
            "libnvidia-encode.so.1 + libnvcuvid.so.1",
        )?;

        let vaapi = DetectedGpuBackend::new(
            GpuBackend::Vaapi,
            GpuCapability::VideoDecodeOnly,
            "NVIDIA GeForce RTX 2060",
            "nvidia_drv_video.so",
        )?;

        let host = HostHardware::new(cpu, vec![adapter], vec![nvenc, vaapi]);
        assert!(host.has_hardware_gpu());
        assert!(host.supports_video_encode(GpuBackend::Nvenc));
        assert!(!host.supports_video_encode(GpuBackend::Vaapi));
        assert!(!host.supports_video_encode(GpuBackend::Qsv));
        Ok(())
    }

    #[test]
    fn rejects_invalid_cpu_topology() -> Result<(), DomainError> {
        let physical = NonZeroUsize::new(8).ok_or(DomainError::EmptyField("test"))?;
        let logical = NonZeroUsize::new(4).ok_or(DomainError::EmptyField("test"))?;

        let result = CpuInfo::new(
            "Test CPU",
            "x86_64",
            physical,
            logical,
            SimdLevel::X86Avx2,
            None,
        );

        assert_eq!(
            result,
            Err(DomainError::InvalidCpuTopology {
                physical_cores: physical,
                logical_threads: logical,
            })
        );
        Ok(())
    }

    #[test]
    fn rejects_blank_fields() {
        let cores = NonZeroUsize::MIN;
        let blank_model = CpuInfo::new("   ", "x86_64", cores, cores, SimdLevel::Scalar, None);
        assert_eq!(blank_model, Err(DomainError::EmptyField("cpu.model_name")));

        let blank_arch = CpuInfo::new("CPU", "  ", cores, cores, SimdLevel::Scalar, None);
        assert_eq!(blank_arch, Err(DomainError::EmptyField("cpu.architecture")));

        let blank_adapter = GpuAdapterInfo::new("", GpuVendor::Nvidia, 0, "", "");
        assert_eq!(
            blank_adapter,
            Err(DomainError::EmptyField("gpu_adapter.name"))
        );

        let blank_device_name =
            DetectedGpuBackend::new(GpuBackend::Vulkan, GpuCapability::ComputeOnly, "   ", "ok");
        assert_eq!(
            blank_device_name,
            Err(DomainError::EmptyField("gpu_backend.device_name"))
        );

        let blank_evidence =
            DetectedGpuBackend::new(GpuBackend::Vulkan, GpuCapability::ComputeOnly, "GPU", "  ");
        assert_eq!(
            blank_evidence,
            Err(DomainError::EmptyField("gpu_backend.evidence"))
        );
    }

    #[test]
    fn has_hardware_gpu_accounts_for_headless_gpu_backends() -> Result<(), DomainError> {
        let cores = NonZeroUsize::MIN;
        let cpu = CpuInfo::new(
            "Server CPU",
            "x86_64",
            cores,
            cores,
            SimdLevel::X86Avx2,
            None,
        )?;
        let nvenc = DetectedGpuBackend::new(
            GpuBackend::Nvenc,
            GpuCapability::VideoEncodeAndDecode,
            "NVIDIA GPU",
            "/usr/lib/libnvidia-encode.so.1",
        )?;

        let headless_host = HostHardware::new(cpu, Vec::new(), vec![nvenc]);
        assert!(headless_host.has_hardware_gpu());
        assert!(headless_host.supports_video_encode(GpuBackend::Nvenc));
        Ok(())
    }

    #[test]
    fn has_hardware_gpu_reflects_detected_adapters_and_backends() -> Result<(), DomainError> {
        let cores = NonZeroUsize::new(2).ok_or(DomainError::EmptyField("test"))?;
        let cpu_a = CpuInfo::new("CI CPU", "x86_64", cores, cores, SimdLevel::X86Avx2, None)?;
        let cpu_b = CpuInfo::new("CI CPU", "x86_64", cores, cores, SimdLevel::X86Avx2, None)?;
        let adapter = GpuAdapterInfo::new(
            "NVIDIA GeForce RTX 2060",
            GpuVendor::Nvidia,
            0x1e89,
            "nvidia",
            "590.48.01",
        )?;

        assert!(!HostHardware::new(cpu_a, Vec::new(), Vec::new()).has_hardware_gpu());
        assert!(HostHardware::new(cpu_b, vec![adapter], Vec::new()).has_hardware_gpu());
        assert_eq!(GpuVendor::from_pci_id(0x10de), GpuVendor::Nvidia);
        assert_eq!(GpuVendor::from_pci_id(0x1002), GpuVendor::Amd);
        assert_eq!(GpuVendor::from_pci_id(0x8086), GpuVendor::Intel);
        assert_eq!(GpuVendor::from_pci_id(0x106b), GpuVendor::Apple);
        Ok(())
    }
}
