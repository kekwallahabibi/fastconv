use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};

use fastconv_domain::{
    CpuInfo, DetectedGpuBackend, DomainError, GpuAdapterInfo, GpuBackend, GpuCapability, GpuVendor,
    HostHardware, SimdLevel,
};

use crate::{ProbeError, available_cpu_threads};

/// Probes the local host for CPU topology, SIMD level, GPU adapters, and native GPU media backends.
///
/// Never spawns `ffmpeg`, `ffprobe`, or any external subprocess.
///
/// # Errors
///
/// Returns [`ProbeError`] if OS CPU parallelism, `/proc` topology files, or domain
/// validation fail.
pub fn detect_host_hardware() -> Result<HostHardware, ProbeError> {
    let cpu = detect_cpu()?;
    let gpu_adapters = detect_gpu_adapters()?;
    let gpu_backends = detect_gpu_backends(&gpu_adapters)?;
    Ok(HostHardware::new(cpu, gpu_adapters, gpu_backends))
}

fn detect_cpu() -> Result<CpuInfo, ProbeError> {
    let logical_threads = available_cpu_threads()?;
    let simd_level = detect_simd_level();
    let architecture = std::env::consts::ARCH.to_owned();

    #[cfg(target_os = "linux")]
    let (model_name, physical_cores, total_memory_bytes) = {
        let cpuinfo_path = Path::new("/proc/cpuinfo");
        let cpuinfo = fs::read_to_string(cpuinfo_path).map_err(|source| ProbeError::Io {
            path: cpuinfo_path.to_path_buf(),
            source,
        })?;
        let (model_name, physical_cores) = parse_linux_cpuinfo(&cpuinfo, logical_threads);

        let meminfo_path = Path::new("/proc/meminfo");
        let meminfo = fs::read_to_string(meminfo_path).map_err(|source| ProbeError::Io {
            path: meminfo_path.to_path_buf(),
            source,
        })?;
        let total_memory_bytes = parse_linux_meminfo_bytes(&meminfo);
        (model_name, physical_cores, total_memory_bytes)
    };

    #[cfg(not(target_os = "linux"))]
    let (model_name, physical_cores, total_memory_bytes) = (
        format!("{architecture} CPU"),
        logical_threads,
        None::<NonZeroU64>,
    );

    Ok(CpuInfo::new(
        &model_name,
        &architecture,
        physical_cores,
        logical_threads,
        simd_level,
        total_memory_bytes,
    )?)
}

fn detect_simd_level() -> SimdLevel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            SimdLevel::X86Avx512
        } else if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            SimdLevel::X86Avx2Fma
        } else if std::arch::is_x86_feature_detected!("avx2") {
            SimdLevel::X86Avx2
        } else if std::arch::is_x86_feature_detected!("avx") {
            SimdLevel::X86Avx
        } else if std::arch::is_x86_feature_detected!("sse4.2") {
            SimdLevel::X86Sse42
        } else if std::arch::is_x86_feature_detected!("sse2") {
            SimdLevel::X86Sse2
        } else {
            SimdLevel::Scalar
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            SimdLevel::ArmNeon
        } else {
            SimdLevel::Scalar
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        SimdLevel::Scalar
    }
}

fn parse_linux_cpuinfo(content: &str, logical_threads: NonZeroUsize) -> (String, NonZeroUsize) {
    let mut model_name: Option<String> = None;
    let mut core_pairs = BTreeSet::new();
    let mut max_cpu_cores: Option<usize> = None;

    let mut current_physical_id: Option<u32> = None;
    let mut current_core_id: Option<u32> = None;

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            if let (Some(phys), Some(core)) = (current_physical_id, current_core_id) {
                core_pairs.insert((phys, core));
            }
            current_physical_id = None;
            current_core_id = None;
            continue;
        }

        let Some((raw_key, raw_value)) = line.split_once(':') else {
            continue;
        };
        let key = raw_key.trim();
        let value = raw_value.trim();

        match key {
            "processor" => {
                if let (Some(phys), Some(core)) = (current_physical_id, current_core_id) {
                    core_pairs.insert((phys, core));
                }
                current_physical_id = None;
                current_core_id = None;
            }
            "model name" | "Hardware" | "Processor"
                if model_name.is_none() && !value.is_empty() =>
            {
                model_name = Some(value.to_owned());
            }
            "physical id" => {
                if let Ok(id) = value.parse::<u32>() {
                    current_physical_id = Some(id);
                }
            }
            "core id" => {
                if let Ok(id) = value.parse::<u32>() {
                    current_core_id = Some(id);
                }
            }
            "cpu cores" => {
                if let Ok(count) = value.parse::<usize>()
                    && count > 0
                {
                    max_cpu_cores = Some(max_cpu_cores.map_or(count, |prev| prev.max(count)));
                }
            }
            _ => {}
        }
    }

    if let (Some(phys), Some(core)) = (current_physical_id, current_core_id) {
        core_pairs.insert((phys, core));
    }

    let raw_physical = if !core_pairs.is_empty() {
        core_pairs.len()
    } else if let Some(cores) = max_cpu_cores {
        cores
    } else {
        logical_threads.get()
    };

    // Clamp to logical_threads so cgroup/affinity limits never violate logical_threads >= physical_cores.
    let clamped = raw_physical.clamp(1, logical_threads.get());
    let physical_cores = NonZeroUsize::new(clamped).unwrap_or(logical_threads);
    let resolved_model = model_name.unwrap_or_else(|| format!("{} CPU", std::env::consts::ARCH));

    (resolved_model, physical_cores)
}

fn parse_linux_meminfo_bytes(content: &str) -> Option<NonZeroU64> {
    for line in content.lines() {
        let Some(rest) = line.strip_prefix("MemTotal:") else {
            continue;
        };
        let kb_token = rest.split_whitespace().next()?;
        let kb = kb_token.parse::<u64>().ok()?;
        return NonZeroU64::new(kb.checked_mul(1024)?);
    }
    None
}

#[cfg(target_os = "linux")]
struct NvidiaGpuInfo {
    slot: String,
    model: String,
    firmware: String,
}

/// Enumerates DRM card adapters once; vendor facts for backend classification reuse this scan.
#[cfg(target_os = "linux")]
fn detect_gpu_adapters() -> Result<Vec<GpuAdapterInfo>, ProbeError> {
    let drm_dir = Path::new("/sys/class/drm");
    let entries = match fs::read_dir(drm_dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => {
            return Err(ProbeError::Io {
                path: drm_dir.to_path_buf(),
                source,
            });
        }
    };

    let nvidia_info = read_nvidia_gpu_information();
    let mut adapters = Vec::new();
    for entry_result in entries {
        let entry = entry_result.map_err(|source| ProbeError::Io {
            path: drm_dir.to_path_buf(),
            source,
        })?;
        let file_name = entry.file_name();
        let Some(card) = file_name.to_str() else {
            continue;
        };
        if !card.starts_with("card") || card.contains('-') {
            continue;
        }

        let device_dir = entry.path().join("device");
        let Some(vendor_id) = read_pci_id(&device_dir.join("vendor")) else {
            continue;
        };
        let device_id = read_pci_id(&device_dir.join("device")).unwrap_or(0);
        let vendor = GpuVendor::from_pci_id(vendor_id);
        let slot = device_dir.canonicalize().ok().and_then(|resolved| {
            resolved
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_owned)
        });
        let (name, driver_info) = if vendor == GpuVendor::Nvidia {
            nvidia_adapter_details(&nvidia_info, slot.as_deref(), device_id)
        } else {
            (vendor_fallback_name(vendor, device_id), String::new())
        };
        let driver = read_kernel_driver(&device_dir.join("uevent"));

        adapters.push(GpuAdapterInfo::new(
            &name,
            vendor,
            device_id,
            &driver,
            &driver_info,
        )?);
    }

    Ok(adapters)
}

#[cfg(not(target_os = "linux"))]
fn detect_gpu_adapters() -> Result<Vec<GpuAdapterInfo>, ProbeError> {
    Ok(Vec::new())
}

#[cfg(target_os = "linux")]
fn read_pci_id(path: &Path) -> Option<u32> {
    let content = fs::read_to_string(path).ok()?;
    let trimmed = content.trim();
    let hex = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
        .unwrap_or(trimmed);
    u32::from_str_radix(hex, 16).ok()
}

#[cfg(target_os = "linux")]
fn read_kernel_driver(uevent_path: &Path) -> String {
    let Ok(content) = fs::read_to_string(uevent_path) else {
        return String::new();
    };
    for line in content.lines() {
        if let Some(driver) = line.strip_prefix("DRIVER=") {
            return driver.trim().to_owned();
        }
    }
    String::new()
}

#[cfg(target_os = "linux")]
fn read_nvidia_gpu_information() -> Vec<NvidiaGpuInfo> {
    let Ok(entries) = fs::read_dir(Path::new("/proc/driver/nvidia/gpus")) else {
        return Vec::new();
    };

    let mut result = Vec::new();
    for entry in entries.flatten() {
        let Ok(content) = fs::read_to_string(entry.path().join("information")) else {
            continue;
        };
        let mut slot = None;
        let mut model = None;
        let mut firmware = None;
        for line in content.lines() {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            match key.trim() {
                "Bus Location" => slot = Some(value.trim().to_owned()),
                "Model" => model = Some(value.trim().to_owned()),
                "GPU Firmware" | "Firmware Version" => firmware = Some(value.trim().to_owned()),
                _ => {}
            }
        }
        if let (Some(slot), Some(model)) = (slot, model) {
            result.push(NvidiaGpuInfo {
                slot,
                model,
                firmware: firmware.unwrap_or_default(),
            });
        }
    }
    result
}

#[cfg(target_os = "linux")]
fn nvidia_adapter_details(
    info: &[NvidiaGpuInfo],
    slot: Option<&str>,
    device_id: u32,
) -> (String, String) {
    let Some(slot) = slot else {
        return (
            vendor_fallback_name(GpuVendor::Nvidia, device_id),
            String::new(),
        );
    };
    match info
        .iter()
        .find(|entry| entry.slot.eq_ignore_ascii_case(slot))
    {
        Some(entry) => {
            let driver_info = if entry.firmware.is_empty() {
                format!("PCI 0x{device_id:04x}")
            } else {
                entry.firmware.clone()
            };
            (entry.model.clone(), driver_info)
        }
        None => (
            vendor_fallback_name(GpuVendor::Nvidia, device_id),
            String::new(),
        ),
    }
}

#[cfg(target_os = "linux")]
fn vendor_fallback_name(vendor: GpuVendor, device_id: u32) -> String {
    let label = match vendor {
        GpuVendor::Nvidia => "NVIDIA GPU",
        GpuVendor::Amd => "AMD GPU",
        GpuVendor::Intel => "Intel GPU",
        GpuVendor::Apple => "Apple GPU",
        GpuVendor::Other(_) => "GPU",
    };
    format!("{label} (PCI 0x{device_id:04x})")
}

#[cfg(target_os = "linux")]
fn detect_gpu_backends(adapters: &[GpuAdapterInfo]) -> Result<Vec<DetectedGpuBackend>, ProbeError> {
    let facts = gather_linux_probe_facts(adapters)?;
    Ok(classify_linux_gpu_backends(&facts, adapters)?)
}

// Off Linux there is no DRM sysfs to scan, so no backend can be confirmed without guessing.
#[cfg(not(target_os = "linux"))]
fn detect_gpu_backends(
    _adapters: &[GpuAdapterInfo],
) -> Result<Vec<DetectedGpuBackend>, ProbeError> {
    Ok(Vec::new())
}

#[cfg(any(target_os = "linux", test))]
#[derive(Debug, Default)]
struct LinuxProbeFacts {
    drm_vendors: Vec<GpuVendor>,
    has_render_node: bool,
    has_nvidia_ctl: bool,
    nvenc_lib: Option<PathBuf>,
    nvcuvid_lib: Option<PathBuf>,
    vaapi_lib: Option<PathBuf>,
    vpl_or_mfx_lib: Option<PathBuf>,
    amf_lib: Option<PathBuf>,
    intel_vaapi_driver: Option<PathBuf>,
    amd_vaapi_driver: Option<PathBuf>,
    nvidia_vaapi_driver: Option<PathBuf>,
}

#[cfg(target_os = "linux")]
fn gather_linux_probe_facts(adapters: &[GpuAdapterInfo]) -> Result<LinuxProbeFacts, ProbeError> {
    let mut drm_vendors = Vec::new();
    for adapter in adapters {
        let vendor = adapter.vendor();
        if !drm_vendors.contains(&vendor) {
            drm_vendors.push(vendor);
        }
    }
    let has_render_node = has_drm_render_node(Path::new("/dev/dri"))?;
    let has_nvidia_ctl = Path::new("/dev/nvidiactl").exists();

    let lib_dirs = library_search_dirs();
    let dri_dirs = vaapi_driver_search_dirs();

    Ok(LinuxProbeFacts {
        drm_vendors,
        has_render_node,
        has_nvidia_ctl,
        nvenc_lib: find_first_file(&lib_dirs, &["libnvidia-encode.so.1", "libnvidia-encode.so"]),
        nvcuvid_lib: find_first_file(&lib_dirs, &["libnvcuvid.so.1", "libnvcuvid.so"]),
        vaapi_lib: find_first_file(&lib_dirs, &["libva.so.2", "libva.so"]),
        vpl_or_mfx_lib: find_first_file(&lib_dirs, &["libvpl.so.2", "libmfx.so.1"]),
        amf_lib: find_first_file(&lib_dirs, &["libamfrt64.so.1", "libamfrt64.so"]),
        intel_vaapi_driver: find_first_file(&dri_dirs, &["iHD_drv_video.so", "i965_drv_video.so"]),
        amd_vaapi_driver: find_first_file(&dri_dirs, &["radeonsi_drv_video.so"]),
        nvidia_vaapi_driver: find_first_file(&dri_dirs, &["nvidia_drv_video.so"]),
    })
}

#[cfg(target_os = "linux")]
fn has_drm_render_node(dri_dir: &Path) -> Result<bool, ProbeError> {
    let entries = match fs::read_dir(dri_dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(source) => {
            return Err(ProbeError::Io {
                path: dri_dir.to_path_buf(),
                source,
            });
        }
    };

    for entry_result in entries {
        let entry = entry_result.map_err(|source| ProbeError::Io {
            path: dri_dir.to_path_buf(),
            source,
        })?;
        if entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.starts_with("renderD"))
        {
            return Ok(true);
        }
    }

    Ok(false)
}

#[cfg(target_os = "linux")]
fn library_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(ld_paths) = env::var_os("LD_LIBRARY_PATH") {
        dirs.extend(env::split_paths(&ld_paths));
    }
    dirs.extend(
        [
            "/usr/lib",
            "/usr/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            "/lib64",
            "/lib/x86_64-linux-gnu",
            "/run/opengl-driver/lib",
        ]
        .into_iter()
        .map(PathBuf::from),
    );
    dirs
}

#[cfg(target_os = "linux")]
fn vaapi_driver_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(env_paths) = env::var_os("LIBVA_DRIVERS_PATH") {
        dirs.extend(env::split_paths(&env_paths));
    }
    dirs.extend(
        [
            "/usr/lib/dri",
            "/usr/lib64/dri",
            "/usr/lib/x86_64-linux-gnu/dri",
            "/usr/lib/aarch64-linux-gnu/dri",
        ]
        .into_iter()
        .map(PathBuf::from),
    );
    dirs
}

#[cfg(target_os = "linux")]
fn find_first_file(dirs: &[PathBuf], candidates: &[&str]) -> Option<PathBuf> {
    for dir in dirs {
        for candidate in candidates {
            let full = dir.join(candidate);
            if full.is_file() {
                return Some(full);
            }
        }
    }
    None
}

#[cfg(any(target_os = "linux", test))]
fn classify_linux_gpu_backends(
    facts: &LinuxProbeFacts,
    adapters: &[GpuAdapterInfo],
) -> Result<Vec<DetectedGpuBackend>, DomainError> {
    let has_vendor = |target: GpuVendor| {
        adapters.iter().any(|adapter| adapter.vendor() == target)
            || facts.drm_vendors.contains(&target)
    };

    let has_nvidia = has_vendor(GpuVendor::Nvidia) || facts.has_nvidia_ctl;
    let has_intel = has_vendor(GpuVendor::Intel);
    let has_amd = has_vendor(GpuVendor::Amd);

    let mut backends = Vec::new();

    if has_nvidia && let Some(nvenc_path) = &facts.nvenc_lib {
        let (capability, evidence) = match &facts.nvcuvid_lib {
            Some(cuvid_path) => (
                GpuCapability::VideoEncodeAndDecode,
                format!("{} + {}", nvenc_path.display(), cuvid_path.display()),
            ),
            None => (
                GpuCapability::VideoEncodeOnly,
                nvenc_path.display().to_string(),
            ),
        };
        backends.push(DetectedGpuBackend::new(
            GpuBackend::Nvenc,
            capability,
            &device_name_for(adapters, GpuVendor::Nvidia, "NVIDIA GPU"),
            &evidence,
        )?);
    }

    push_vaapi_backends(
        facts,
        adapters,
        has_intel,
        has_amd,
        has_nvidia,
        &mut backends,
    )?;

    if has_intel
        && facts.has_render_node
        && let Some(vpl_path) = &facts.vpl_or_mfx_lib
    {
        backends.push(DetectedGpuBackend::new(
            GpuBackend::Qsv,
            GpuCapability::VideoEncodeAndDecode,
            &device_name_for(adapters, GpuVendor::Intel, "Intel GPU"),
            &vpl_path.display().to_string(),
        )?);
    }

    if has_amd && let Some(amf_path) = &facts.amf_lib {
        backends.push(DetectedGpuBackend::new(
            GpuBackend::Amf,
            GpuCapability::VideoEncodeAndDecode,
            &device_name_for(adapters, GpuVendor::Amd, "AMD GPU"),
            &amf_path.display().to_string(),
        )?);
    }

    Ok(backends)
}

#[cfg(any(target_os = "linux", test))]
fn device_name_for(adapters: &[GpuAdapterInfo], target: GpuVendor, fallback: &str) -> String {
    adapters
        .iter()
        .find(|adapter| adapter.vendor() == target)
        .map_or_else(|| fallback.to_owned(), |adapter| adapter.name().to_owned())
}

#[cfg(any(target_os = "linux", test))]
fn push_vaapi_backends(
    facts: &LinuxProbeFacts,
    adapters: &[GpuAdapterInfo],
    has_intel: bool,
    has_amd: bool,
    has_nvidia: bool,
    backends: &mut Vec<DetectedGpuBackend>,
) -> Result<(), DomainError> {
    if !facts.has_render_node {
        return Ok(());
    }
    let Some(vaapi_path) = &facts.vaapi_lib else {
        return Ok(());
    };

    // libva-nvidia-driver wraps NVDEC for decode only and exposes no VAAPI encode entrypoints.
    let candidates = [
        (
            has_intel,
            GpuVendor::Intel,
            "Intel GPU",
            GpuCapability::VideoEncodeAndDecode,
            &facts.intel_vaapi_driver,
        ),
        (
            has_amd,
            GpuVendor::Amd,
            "AMD GPU",
            GpuCapability::VideoEncodeAndDecode,
            &facts.amd_vaapi_driver,
        ),
        (
            has_nvidia,
            GpuVendor::Nvidia,
            "NVIDIA GPU",
            GpuCapability::VideoDecodeOnly,
            &facts.nvidia_vaapi_driver,
        ),
    ];

    for (present, vendor, fallback, capability, driver_path_opt) in candidates {
        if present && let Some(driver_path) = driver_path_opt {
            backends.push(DetectedGpuBackend::new(
                GpuBackend::Vaapi,
                capability,
                &device_name_for(adapters, vendor, fallback),
                &format!("{} + {}", vaapi_path.display(), driver_path.display()),
            )?);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        LinuxProbeFacts, classify_linux_gpu_backends, detect_host_hardware, parse_linux_cpuinfo,
        parse_linux_meminfo_bytes,
    };
    use crate::ProbeError;
    use fastconv_domain::{DomainError, GpuAdapterInfo, GpuBackend, GpuCapability, GpuVendor};
    use std::num::{NonZeroU64, NonZeroUsize};
    use std::path::PathBuf;

    #[test]
    fn detects_live_host_hardware() -> Result<(), ProbeError> {
        let host = detect_host_hardware()?;
        assert_ne!(host.cpu().model_name(), "");
        assert_ne!(host.cpu().architecture(), "");
        assert!(host.cpu().logical_threads() >= host.cpu().physical_cores());
        Ok(())
    }

    #[test]
    fn parses_linux_cpuinfo_physical_and_logical_cores() -> Result<(), DomainError> {
        let sample = "\
processor\t: 0
model name\t: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
physical id\t: 0
siblings\t: 8
core id\t\t: 0
cpu cores\t: 4

processor\t: 1
model name\t: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
physical id\t: 0
siblings\t: 8
core id\t\t: 1
cpu cores\t: 4

processor\t: 4
model name\t: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
physical id\t: 0
siblings\t: 8
core id\t\t: 0
cpu cores\t: 4
";
        let logical = NonZeroUsize::new(8).ok_or(DomainError::EmptyField("test"))?;
        let (model, physical) = parse_linux_cpuinfo(sample, logical);
        assert_eq!(model, "Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz");
        assert_eq!(physical.get(), 2);
        Ok(())
    }

    #[test]
    fn parses_linux_meminfo() -> Result<(), DomainError> {
        let sample = "\
MemTotal:       32791816 kB
MemFree:          840080 kB
";
        let expected = NonZeroU64::new(32_791_816 * 1024).ok_or(DomainError::EmptyField("test"))?;
        assert_eq!(parse_linux_meminfo_bytes(sample), Some(expected));
        assert_eq!(parse_linux_meminfo_bytes("MemFree: 100 kB"), None);
        Ok(())
    }

    #[test]
    fn classifies_nvidia_system_without_qsv_false_positive() -> Result<(), DomainError> {
        let adapter = GpuAdapterInfo::new(
            "NVIDIA GeForce RTX 2060",
            GpuVendor::Nvidia,
            0x1e89,
            "nvidia",
            "590.48.01",
        )?;

        let facts = LinuxProbeFacts {
            drm_vendors: vec![GpuVendor::Nvidia],
            has_render_node: true,
            has_nvidia_ctl: true,
            nvenc_lib: Some(PathBuf::from("/usr/lib/libnvidia-encode.so.1")),
            nvcuvid_lib: Some(PathBuf::from("/usr/lib/libnvcuvid.so.1")),
            vaapi_lib: Some(PathBuf::from("/usr/lib/libva.so.2")),
            vpl_or_mfx_lib: Some(PathBuf::from("/usr/lib/libvpl.so.2")),
            amf_lib: None,
            intel_vaapi_driver: None,
            amd_vaapi_driver: None,
            nvidia_vaapi_driver: Some(PathBuf::from("/usr/lib/dri/nvidia_drv_video.so")),
        };

        let backends = classify_linux_gpu_backends(&facts, &[adapter])?;
        assert!(backends.iter().any(|b| {
            b.backend() == GpuBackend::Nvenc
                && b.capability() == GpuCapability::VideoEncodeAndDecode
                && b.device_name() == "NVIDIA GeForce RTX 2060"
        }));
        assert!(!backends.iter().any(|b| b.backend() == GpuBackend::Vulkan));
        assert!(backends.iter().any(|b| {
            b.backend() == GpuBackend::Vaapi && b.capability() == GpuCapability::VideoDecodeOnly
        }));
        assert!(!backends.iter().any(|b| b.backend() == GpuBackend::Qsv));
        assert!(!backends.iter().any(|b| b.backend() == GpuBackend::Amf));
        Ok(())
    }

    #[test]
    fn classifies_intel_and_amd_backends() -> Result<(), DomainError> {
        let intel_facts = LinuxProbeFacts {
            drm_vendors: vec![GpuVendor::Intel],
            has_render_node: true,
            vaapi_lib: Some(PathBuf::from("/usr/lib/libva.so.2")),
            vpl_or_mfx_lib: Some(PathBuf::from("/usr/lib/libvpl.so.2")),
            intel_vaapi_driver: Some(PathBuf::from("/usr/lib/dri/iHD_drv_video.so")),
            ..LinuxProbeFacts::default()
        };
        let intel_backends = classify_linux_gpu_backends(&intel_facts, &[])?;
        assert!(intel_backends.iter().any(|b| {
            b.backend() == GpuBackend::Qsv && b.capability() == GpuCapability::VideoEncodeAndDecode
        }));
        assert!(intel_backends.iter().any(|b| {
            b.backend() == GpuBackend::Vaapi
                && b.capability() == GpuCapability::VideoEncodeAndDecode
        }));

        let amd_facts = LinuxProbeFacts {
            drm_vendors: vec![GpuVendor::Amd],
            has_render_node: true,
            vaapi_lib: Some(PathBuf::from("/usr/lib/libva.so.2")),
            amf_lib: Some(PathBuf::from("/usr/lib/libamfrt64.so.1")),
            amd_vaapi_driver: Some(PathBuf::from("/usr/lib/dri/radeonsi_drv_video.so")),
            ..LinuxProbeFacts::default()
        };
        let amd_backends = classify_linux_gpu_backends(&amd_facts, &[])?;
        assert!(amd_backends.iter().any(|b| {
            b.backend() == GpuBackend::Amf && b.capability() == GpuCapability::VideoEncodeAndDecode
        }));
        assert!(amd_backends.iter().any(|b| {
            b.backend() == GpuBackend::Vaapi
                && b.capability() == GpuCapability::VideoEncodeAndDecode
        }));
        Ok(())
    }

    #[test]
    fn clamps_physical_cores_to_logical_threads_and_handles_empty_cpuinfo()
    -> Result<(), DomainError> {
        let cgroup_limited = "\
processor\t: 0
physical id\t: 0
core id\t\t: 0

processor\t: 1
physical id\t: 0
core id\t\t: 1

processor\t: 2
physical id\t: 0
core id\t\t: 2

processor\t: 3
physical id\t: 0
core id\t\t: 3
";
        let two_threads = NonZeroUsize::new(2).ok_or(DomainError::EmptyField("test"))?;
        let (model, physical) = parse_linux_cpuinfo(cgroup_limited, two_threads);
        assert_eq!(model, format!("{} CPU", std::env::consts::ARCH));
        assert_eq!(physical.get(), 2);

        let four_threads = NonZeroUsize::new(4).ok_or(DomainError::EmptyField("test"))?;
        let (_, empty_physical) = parse_linux_cpuinfo("", four_threads);
        assert_eq!(empty_physical.get(), 4);
        Ok(())
    }

    #[test]
    fn classifies_hybrid_intel_and_nvidia_without_dropping_vaapi() -> Result<(), DomainError> {
        let facts = LinuxProbeFacts {
            drm_vendors: vec![GpuVendor::Intel, GpuVendor::Nvidia],
            has_render_node: true,
            has_nvidia_ctl: true,
            nvenc_lib: Some(PathBuf::from("/usr/lib/libnvidia-encode.so.1")),
            nvcuvid_lib: Some(PathBuf::from("/usr/lib/libnvcuvid.so.1")),
            vaapi_lib: Some(PathBuf::from("/usr/lib/libva.so.2")),
            vpl_or_mfx_lib: Some(PathBuf::from("/usr/lib/libvpl.so.2")),
            amf_lib: None,
            intel_vaapi_driver: Some(PathBuf::from("/usr/lib/dri/iHD_drv_video.so")),
            amd_vaapi_driver: None,
            nvidia_vaapi_driver: Some(PathBuf::from("/usr/lib/dri/nvidia_drv_video.so")),
        };

        let backends = classify_linux_gpu_backends(&facts, &[])?;
        assert!(backends.iter().any(|b| {
            b.backend() == GpuBackend::Vaapi
                && b.capability() == GpuCapability::VideoEncodeAndDecode
                && b.device_name() == "Intel GPU"
        }));
        assert!(backends.iter().any(|b| {
            b.backend() == GpuBackend::Vaapi
                && b.capability() == GpuCapability::VideoDecodeOnly
                && b.device_name() == "NVIDIA GPU"
        }));
        assert!(backends.iter().any(|b| b.backend() == GpuBackend::Qsv));
        assert!(backends.iter().any(|b| b.backend() == GpuBackend::Nvenc));
        Ok(())
    }

    #[test]
    fn classifies_nvenc_encode_only_without_nvcuvid() -> Result<(), DomainError> {
        let facts = LinuxProbeFacts {
            drm_vendors: vec![GpuVendor::Nvidia],
            nvenc_lib: Some(PathBuf::from("/usr/lib/libnvidia-encode.so.1")),
            nvcuvid_lib: None,
            ..LinuxProbeFacts::default()
        };

        let backends = classify_linux_gpu_backends(&facts, &[])?;
        assert!(backends.iter().any(|b| {
            b.backend() == GpuBackend::Nvenc && b.capability() == GpuCapability::VideoEncodeOnly
        }));
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn resolves_nvidia_adapter_details_by_pci_slot() {
        use super::{NvidiaGpuInfo, nvidia_adapter_details, vendor_fallback_name};

        let info = [NvidiaGpuInfo {
            slot: "0000:01:00.0".to_owned(),
            model: "NVIDIA GeForce RTX 2060".to_owned(),
            firmware: "90.15.117.0.9".to_owned(),
        }];
        let (name, firmware) = nvidia_adapter_details(&info, Some("0000:01:00.0"), 0x1e89);
        assert_eq!(name, "NVIDIA GeForce RTX 2060");
        assert_eq!(firmware, "90.15.117.0.9");

        let (fallback, firmware) = nvidia_adapter_details(&info, Some("0000:02:00.0"), 0x1e89);
        assert_eq!(fallback, vendor_fallback_name(GpuVendor::Nvidia, 0x1e89));
        assert_eq!(firmware, "");
        assert!(fallback.contains("0x1e89"));
    }
}
