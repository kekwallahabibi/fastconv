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

/// Adapter device category reported by `wgpu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum GpuDeviceType {
    Discrete,
    Integrated,
    Virtual,
    CpuFallback,
    Other,
}

impl fmt::Display for GpuDeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Discrete => "discrete",
            Self::Integrated => "integrated",
            Self::Virtual => "virtual",
            Self::CpuFallback => "cpu",
            Self::Other => "other",
        };
        f.write_str(label)
    }
}

/// Graphics/compute backend API exposed by a `wgpu` adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum GraphicsApi {
    Vulkan,
    Metal,
    Dx12,
    Gl,
}

impl fmt::Display for GraphicsApi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
            Self::Dx12 => "dx12",
            Self::Gl => "gl",
        };
        f.write_str(label)
    }
}

/// Validated `wgpu` adapter description enumerated without display handles.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WgpuAdapterInfo {
    name: String,
    vendor: GpuVendor,
    device_id: u32,
    device_type: GpuDeviceType,
    graphics_api: GraphicsApi,
    driver: String,
    driver_info: String,
}

impl WgpuAdapterInfo {
    /// Creates a validated [`WgpuAdapterInfo`] record.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::EmptyField`] if `name` is blank.
    pub fn new(
        name: &str,
        vendor: GpuVendor,
        device_id: u32,
        device_type: GpuDeviceType,
        graphics_api: GraphicsApi,
        driver: &str,
        driver_info: &str,
    ) -> Result<Self, DomainError> {
        let name = require_non_empty(name, "wgpu_adapter.name")?;
        Ok(Self {
            name,
            vendor,
            device_id,
            device_type,
            graphics_api,
            driver: driver.trim().to_owned(),
            driver_info: driver_info.trim().to_owned(),
        })
    }

    #[must_use]
    pub const fn is_hardware(&self) -> bool {
        matches!(
            self.device_type,
            GpuDeviceType::Discrete | GpuDeviceType::Integrated
        )
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
    pub const fn device_type(&self) -> GpuDeviceType {
        self.device_type
    }

    #[must_use]
    pub const fn graphics_api(&self) -> GraphicsApi {
        self.graphics_api
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

/// Complete, validated snapshot of the host CPU, `wgpu` adapters, and native GPU backends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HostHardware {
    cpu: CpuInfo,
    wgpu_adapters: Vec<WgpuAdapterInfo>,
    gpu_backends: Vec<DetectedGpuBackend>,
}

impl HostHardware {
    #[must_use]
    pub const fn new(
        cpu: CpuInfo,
        wgpu_adapters: Vec<WgpuAdapterInfo>,
        gpu_backends: Vec<DetectedGpuBackend>,
    ) -> Self {
        Self {
            cpu,
            wgpu_adapters,
            gpu_backends,
        }
    }

    #[must_use]
    pub const fn cpu(&self) -> &CpuInfo {
        &self.cpu
    }

    #[must_use]
    pub fn wgpu_adapters(&self) -> &[WgpuAdapterInfo] {
        &self.wgpu_adapters
    }

    #[must_use]
    pub fn gpu_backends(&self) -> &[DetectedGpuBackend] {
        &self.gpu_backends
    }

    #[must_use]
    pub fn has_hardware_gpu(&self) -> bool {
        self.wgpu_adapters.iter().any(WgpuAdapterInfo::is_hardware) || !self.gpu_backends.is_empty()
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
        CpuInfo, DetectedGpuBackend, GpuCapability, GpuDeviceType, GpuVendor, GraphicsApi,
        HostHardware, SimdLevel, WgpuAdapterInfo,
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

        let adapter = WgpuAdapterInfo::new(
            "NVIDIA GeForce RTX 2060",
            GpuVendor::from_pci_id(0x10de),
            0x1e89,
            GpuDeviceType::Discrete,
            GraphicsApi::Vulkan,
            "NVIDIA",
            "620.73.04",
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

        let blank_adapter = WgpuAdapterInfo::new(
            "",
            GpuVendor::Nvidia,
            0,
            GpuDeviceType::Discrete,
            GraphicsApi::Vulkan,
            "",
            "",
        );
        assert_eq!(
            blank_adapter,
            Err(DomainError::EmptyField("wgpu_adapter.name"))
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
    fn distinguishes_software_rasterizer_from_hardware_gpu() -> Result<(), DomainError> {
        let cores = NonZeroUsize::new(2).ok_or(DomainError::EmptyField("test"))?;
        let cpu = CpuInfo::new("CI CPU", "x86_64", cores, cores, SimdLevel::X86Avx2, None)?;
        let llvmpipe = WgpuAdapterInfo::new(
            "llvmpipe",
            GpuVendor::Other(0x10005),
            0,
            GpuDeviceType::CpuFallback,
            GraphicsApi::Vulkan,
            "llvmpipe",
            "Mesa",
        )?;

        let host = HostHardware::new(cpu, vec![llvmpipe], Vec::new());
        assert!(!host.has_hardware_gpu());
        assert_eq!(GpuVendor::from_pci_id(0x10de), GpuVendor::Nvidia);
        assert_eq!(GpuVendor::from_pci_id(0x1002), GpuVendor::Amd);
        assert_eq!(GpuVendor::from_pci_id(0x8086), GpuVendor::Intel);
        assert_eq!(GpuVendor::from_pci_id(0x106b), GpuVendor::Apple);
        Ok(())
    }
}
