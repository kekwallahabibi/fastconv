#[cfg(feature = "nvenc")]
mod enabled {
    use std::cell::Cell;
    use std::collections::{HashMap, VecDeque};
    use std::ffi::{CStr, c_int, c_void};
    use std::fs;
    use std::num::{NonZeroU64, NonZeroUsize};
    use std::path::{Path, PathBuf};
    use std::ptr;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::mpsc::{self, Receiver, SyncSender};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    use cudarc::driver::{CudaContext, CudaStream, HostSlice, PinnedHostSlice, result, sys};
    use fastconv_domain::{
        GpuBackend, HardwareBackend, ImageBatchItem, ImageBatchPlan, ImageConversionReport,
        ImageInputFormat, ImageQuality, VideoCodec, VideoEncodeMode, VideoRateControl,
    };
    use nvidia_video_codec_sdk::{
        Bitstream, Buffer, ENCODE_API, EncodePictureParams, Encoder, EncoderInitParams,
        EncoderInput, Session,
        sys::nvEncodeAPI::{
            GUID, NV_ENC_BFRAME_REF_MODE, NV_ENC_BUFFER_FORMAT, NV_ENC_CAPS, NV_ENC_CAPS_PARAM,
            NV_ENC_CAPS_PARAM_VER, NV_ENC_CODEC_H264_GUID, NV_ENC_CODEC_HEVC_GUID, NV_ENC_CONFIG,
            NV_ENC_CONFIG_VER, NV_ENC_CREATE_BITSTREAM_BUFFER, NV_ENC_CREATE_BITSTREAM_BUFFER_VER,
            NV_ENC_DEVICE_TYPE, NV_ENC_INITIALIZE_PARAMS, NV_ENC_INITIALIZE_PARAMS_VER,
            NV_ENC_INPUT_RESOURCE_TYPE, NV_ENC_LOCK_BITSTREAM, NV_ENC_LOCK_BITSTREAM_VER,
            NV_ENC_MAP_INPUT_RESOURCE, NV_ENC_MAP_INPUT_RESOURCE_VER,
            NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER,
            NV_ENC_PARAMS_RC_MODE, NV_ENC_PIC_FLAGS, NV_ENC_PIC_PARAMS, NV_ENC_PIC_PARAMS_VER,
            NV_ENC_PIC_STRUCT, NV_ENC_PIC_TYPE, NV_ENC_PRESET_CONFIG, NV_ENC_PRESET_CONFIG_VER,
            NV_ENC_PRESET_P4_GUID, NV_ENC_RC_PARAMS, NV_ENC_REGISTER_RESOURCE,
            NV_ENC_REGISTER_RESOURCE_VER, NV_ENC_TUNING_INFO, NV_ENC_VUI_COLOR_PRIMARIES,
            NV_ENC_VUI_MATRIX_COEFFS, NV_ENC_VUI_TRANSFER_CHARACTERISTIC, NV_ENC_VUI_VIDEO_FORMAT,
            NVENCAPI_VERSION, NVENCSTATUS,
        },
    };

    use super::{
        NVENC_AQ_STRENGTH, NvencCodecCaps, NvencVideoPlan, assign_reordered_timestamps,
        quality_to_nvenc_cq,
    };

    use crate::PipelineError;
    use crate::decode::{
        DecodeTargetHint, DecodedFrameMeta, DecodedPixelFormat, decode_image_into,
        precreate_target_directories,
    };
    use crate::gpu::heic::{HeicColorInfo, mux_annex_b_hevc_to_heic};
    use crate::video_bitstream::{
        annex_b_to_avcc, annex_b_to_hvcc, extract_avcc_from_annex_b, extract_hvcc_from_annex_b,
        is_h264_keyframe_annex_b, is_hevc_keyframe_annex_b,
    };
    use crate::video_codecs::{
        DecodedFrameInfo, EncodedVideoStream, decode_video_track_streaming,
        estimate_video_fps_from_packets, nv12_buffer_len,
    };
    use crate::video_container::{ContainerPacket, ContainerTrack, US_TIMESCALE};

    const DECODE_QUEUE_CAPACITY: usize = 2;
    /// Pinned host slots shared between the decode thread and the NVENC thread.
    const VIDEO_RING_SLOTS: usize = 4;

    /// A fully decoded NV12 frame in a recyclable pinned host slot.
    struct FilledPinnedFrame {
        pinned: PinnedHostSlice<u8>,
        info: DecodedFrameInfo,
        pitch: u32,
    }

    /// A synchronous `cuMemAlloc` device buffer owned by the NVENC ring.
    ///
    /// NVENC refuses CUDA pointers from cudarc's async memory pools
    /// (`NV_ENC_ERR_RESOURCE_REGISTER_FAILED`), so `CudaStream::alloc_zeros`
    /// cannot be used here despite being safe.
    struct DeviceBuffer {
        ptr: sys::CUdeviceptr,
        ctx: Arc<CudaContext>,
    }

    impl DeviceBuffer {
        #[allow(unsafe_code)]
        fn zeroed(
            ctx: &Arc<CudaContext>,
            stream: &CudaStream,
            len: usize,
        ) -> Result<Self, PipelineError> {
            ctx.bind_to_thread().map_err(|err| {
                video_error(format!(
                    "failed to bind CUDA context for device buffer: {err}"
                ))
            })?;
            let ptr = unsafe { result::malloc_sync(len) }.map_err(|err| {
                video_error(format!(
                    "failed to allocate CUDA device buffer ({len} bytes): {err}"
                ))
            })?;
            let buffer = Self {
                ptr,
                ctx: Arc::clone(ctx),
            };
            // SAFETY: `ptr` is a fresh `cuMemAlloc` allocation of `len` bytes.
            // On error, `buffer` is dropped and its `Drop` impl frees `ptr`.
            if let Err(err) = unsafe { result::memset_d8_async(ptr, 0, len, stream.cu_stream()) } {
                return Err(video_error(format!(
                    "failed to clear CUDA device buffer: {err}"
                )));
            }
            Ok(buffer)
        }

        fn ptr(&self) -> sys::CUdeviceptr {
            self.ptr
        }
    }

    impl Drop for DeviceBuffer {
        #[allow(unsafe_code)]
        fn drop(&mut self) {
            // `free_sync` waits for queued work that referenced the allocation.
            self.ctx.record_err(unsafe { result::free_sync(self.ptr) });
        }
    }

    // `CudaContext::alloc_pinned` returns uninitialized page-locked write-combined memory;
    // zero the whole slice before exposing it so padding bytes are deterministic.
    #[allow(unsafe_code)]
    fn alloc_zeroed_pinned(
        cuda_ctx: &Arc<CudaContext>,
        len: usize,
    ) -> Result<PinnedHostSlice<u8>, PipelineError> {
        // SAFETY: the caller frees this allocation through `PinnedHostSlice::drop`.
        // `alloc_pinned` documents the memory as unset, so it must be initialized
        // through a raw pointer before any `&mut [u8]` is created over it.
        let mut slice = unsafe { cuda_ctx.alloc_pinned::<u8>(len) }.map_err(|err| {
            video_error(format!(
                "failed to allocate CUDA pinned buffer ({len} bytes): {err}"
            ))
        })?;
        // `as_mut_ptr` synchronizes the slice's event and yields the raw pointer.
        // Writing all `len` bytes through it establishes the invariant required
        // by every later `as_mut_slice()` call on this slot.
        let ptr = slice
            .as_mut_ptr()
            .map_err(|err| video_error(format!("failed to access CUDA pinned buffer: {err}")))?;
        unsafe { std::ptr::write_bytes(ptr, 0_u8, len) };
        Ok(slice)
    }

    struct DecodedRgbaFrame {
        item_index: usize,
        width: u32,
        height: u32,
        rgba_bytes: Vec<u8>,
        input_file_bytes: u64,
    }

    struct ActiveHevcSession {
        width: u32,
        height: u32,
        session: Session,
    }

    /// Maps quality (`0..=100`) to HEVC `ConstQP` (`1..=51`).
    fn quality_to_hevc_qp(quality: ImageQuality) -> u32 {
        let inv = u32::from(100_u8.saturating_sub(quality.get()));
        let qp = 8 + (inv * 43 + 50) / 100;
        qp.clamp(1, 51)
    }

    pub(super) fn execute_heic_batch(
        plan: &ImageBatchPlan,
        quality: ImageQuality,
    ) -> Result<ImageConversionReport, PipelineError> {
        let items = plan.items();
        let Some(first_item) = items.first() else {
            return Err(PipelineError::EmptyOutput(PathBuf::new()));
        };

        let cuda_ctx = CudaContext::new(0).map_err(|err| PipelineError::GpuBackendFailed {
            backend: GpuBackend::Nvenc,
            reason: format!("failed to initialize CUDA device 0 for NVENC: {err}"),
        })?;
        let device_name = cuda_ctx
            .name()
            .map_err(|err| PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                reason: format!("failed to query CUDA device name: {err}"),
            })?;

        let first_dims = first_item.dimensions();
        let qp = quality_to_hevc_qp(quality);
        // Open and verify the HEVC session up front so unsupported GPUs fail before touching I/O,
        // and reuse this live session directly for the first resolution run.
        let initial_session = open_hevc_session(
            &cuda_ctx,
            first_dims.width().get(),
            first_dims.height().get(),
            qp,
        )?;
        precreate_target_directories(items)?;

        let worker_threads = plan.effective_workers();
        let files_converted = NonZeroUsize::new(items.len())
            .ok_or_else(|| PipelineError::EmptyOutput(first_item.target().to_path_buf()))?;

        let (total_input, total_output) = if items.len() == 1 || worker_threads.get() == 1 {
            execute_sequential(&cuda_ctx, items, qp, Some(initial_session))?
        } else {
            execute_overlapped(
                &cuda_ctx,
                items,
                qp,
                worker_threads.get(),
                Some(initial_session),
            )?
        };

        let input_nz = NonZeroU64::new(total_input)
            .ok_or_else(|| PipelineError::EmptyOutput(first_item.source().to_path_buf()))?;
        let output_nz = NonZeroU64::new(total_output)
            .ok_or_else(|| PipelineError::EmptyOutput(first_item.target().to_path_buf()))?;

        Ok(ImageConversionReport::new(
            files_converted,
            input_nz,
            output_nz,
            worker_threads,
            HardwareBackend::Gpu(GpuBackend::Nvenc),
            Some(device_name),
        ))
    }

    fn open_hevc_session(
        cuda_ctx: &Arc<CudaContext>,
        width: u32,
        height: u32,
        qp: u32,
    ) -> Result<ActiveHevcSession, PipelineError> {
        let encoder = Encoder::initialize_with_cuda(Arc::clone(cuda_ctx)).map_err(|err| {
            PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                reason: format!("failed to open NVENC encoder: {err}"),
            }
        })?;
        let guids = encoder
            .get_encode_guids()
            .map_err(|err| PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                reason: format!("failed to query NVENC codecs: {err}"),
            })?;
        if !guids.contains(&NV_ENC_CODEC_HEVC_GUID) {
            return Err(PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                reason: "this NVIDIA GPU does not expose hardware HEVC (H.265) NVENC encoding"
                    .to_owned(),
            });
        }
        let formats = encoder
            .get_supported_input_formats(NV_ENC_CODEC_HEVC_GUID)
            .map_err(|err| PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                reason: format!("failed to query NVENC HEVC input formats: {err}"),
            })?;
        if !formats.contains(&NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ABGR) {
            return Err(PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                reason: "NVENC HEVC on this GPU does not support ABGR (RGBA8) input buffers"
                    .to_owned(),
            });
        }
        let mut preset_cfg = encoder
            .get_preset_config(
                NV_ENC_CODEC_HEVC_GUID,
                NV_ENC_PRESET_P4_GUID,
                NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_HIGH_QUALITY,
            )
            .map_err(|err| PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                reason: format!("failed to query NVENC HEVC P4 preset config: {err}"),
            })?;

        preset_cfg.presetCfg.gopLength = 1;
        preset_cfg.presetCfg.frameIntervalP = 1;
        preset_cfg.presetCfg.rcParams.rateControlMode =
            NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CONSTQP;
        preset_cfg.presetCfg.rcParams.constQP.qpIntra = qp;
        preset_cfg.presetCfg.rcParams.constQP.qpInterP = qp;
        preset_cfg.presetCfg.rcParams.constQP.qpInterB = qp;

        configure_hevc_codec_config(&mut preset_cfg.presetCfg);

        let mut init = EncoderInitParams::new(NV_ENC_CODEC_HEVC_GUID, width, height);
        init.preset_guid(NV_ENC_PRESET_P4_GUID)
            .tuning_info(NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_HIGH_QUALITY)
            // NVENC divides by frameRateDen internally; set 1/1 for still images.
            .framerate(1, 1)
            .display_aspect_ratio(width, height)
            .encode_config(&mut preset_cfg.presetCfg);

        let session = encoder
            .start_session(NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ABGR, init)
            .map_err(|err| PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                reason: format!("failed to start NVENC HEVC session for {width}x{height}: {err}"),
            })?;

        Ok(ActiveHevcSession {
            width,
            height,
            session,
        })
    }

    // NVENC stores codec-specific config in a C union (`NV_ENC_CODEC_CONFIG`).
    // Accessing `hevcConfig` is valid because `preset_cfg` was initialized for `NV_ENC_CODEC_HEVC_GUID`.
    #[allow(unsafe_code)]
    fn configure_hevc_codec_config(cfg: &mut NV_ENC_CONFIG) {
        // SAFETY: `cfg` was populated by `get_preset_config(NV_ENC_CODEC_HEVC_GUID, ...)`,
        // so the active union member of `encodeCodecConfig` is `hevcConfig`.
        let hevc = unsafe { &mut cfg.encodeCodecConfig.hevcConfig };
        hevc.idrPeriod = 1;
        hevc.maxNumRefFramesInDPB = 1;
        hevc.set_repeatSPSPPS(1);
        hevc.set_outputAUD(0);
        hevc.set_chromaFormatIDC(1);
        hevc.set_pixelBitDepthMinus8(0);

        let vui = &mut hevc.hevcVUIParameters;
        vui.videoSignalTypePresentFlag = 1;
        vui.videoFormat = NV_ENC_VUI_VIDEO_FORMAT::NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
        vui.videoFullRangeFlag = 0;
        vui.colourDescriptionPresentFlag = 1;
        vui.colourPrimaries = NV_ENC_VUI_COLOR_PRIMARIES::NV_ENC_VUI_COLOR_PRIMARIES_BT709;
        vui.transferCharacteristics =
            NV_ENC_VUI_TRANSFER_CHARACTERISTIC::NV_ENC_VUI_TRANSFER_CHARACTERISTIC_SRGB;
        vui.colourMatrix = NV_ENC_VUI_MATRIX_COEFFS::NV_ENC_VUI_MATRIX_COEFFS_SMPTE170M;
    }

    fn take_or_open_session(
        cuda_ctx: &Arc<CudaContext>,
        preopened: &mut Option<ActiveHevcSession>,
        width: u32,
        height: u32,
        qp: u32,
    ) -> Result<ActiveHevcSession, PipelineError> {
        if let Some(existing) = preopened.take() {
            if existing.width == width && existing.height == height {
                return Ok(existing);
            }
            drop(existing);
        }
        open_hevc_session(cuda_ctx, width, height, qp)
    }

    fn execute_sequential(
        cuda_ctx: &Arc<CudaContext>,
        items: &[ImageBatchItem],
        qp: u32,
        mut preopened: Option<ActiveHevcSession>,
    ) -> Result<(u64, u64), PipelineError> {
        let mut file_bytes = Vec::new();
        let mut pixel_buf = Vec::new();
        let mut rgba_buf = Vec::new();
        let mut pitched_buf = Vec::new();
        let mut total_in = 0_u64;
        let mut total_out = 0_u64;

        let mut idx = 0_usize;
        while let Some(first_item) = items.get(idx) {
            let first_dims = first_item.dimensions();
            let width = first_dims.width().get();
            let height = first_dims.height().get();
            let active = take_or_open_session(cuda_ctx, &mut preopened, width, height, qp)?;
            let (mut input, mut bitstream, pitch) = allocate_session_buffers(&active.session)?;

            while let Some(item) = items.get(idx) {
                if item.dimensions() != first_dims {
                    break;
                }
                let frame = decode_to_rgba_frame(
                    idx,
                    item,
                    &mut file_bytes,
                    &mut pixel_buf,
                    std::mem::take(&mut rgba_buf),
                )?;
                let out_bytes = encode_into_session(
                    &active,
                    &mut input,
                    &mut bitstream,
                    pitch,
                    item,
                    &frame,
                    &mut pitched_buf,
                )?;
                rgba_buf = frame.rgba_bytes;
                total_in = total_in.saturating_add(frame.input_file_bytes);
                total_out = total_out.saturating_add(out_bytes);
                idx += 1;
            }
        }

        Ok((total_in, total_out))
    }

    fn execute_overlapped(
        cuda_ctx: &Arc<CudaContext>,
        items: &[ImageBatchItem],
        qp: u32,
        worker_count: usize,
        preopened: Option<ActiveHevcSession>,
    ) -> Result<(u64, u64), PipelineError> {
        let next_decode = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let total_in = AtomicU64::new(0);
        let total_out = AtomicU64::new(0);
        let rgba_pool: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

        let decode_workers = worker_count.min(items.len()).clamp(1, 4);
        let (tx_to_nvenc, rx_for_nvenc) =
            mpsc::sync_channel::<DecodedRgbaFrame>(DECODE_QUEUE_CAPACITY);

        let (decode_results, nvenc_result) = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(decode_workers);
            for _ in 0..decode_workers {
                let tx = tx_to_nvenc.clone();
                let next_ref = &next_decode;
                let failed_ref = &failed;
                let pool_ref = &rgba_pool;
                handles.push(scope.spawn(move || -> Result<(), PipelineError> {
                    run_decode_worker(items, next_ref, failed_ref, pool_ref, &tx)
                }));
            }
            drop(tx_to_nvenc);

            let enc_ctx = NvencStageContext {
                cuda_ctx,
                items,
                qp,
                failed: &failed,
                rgba_pool: &rgba_pool,
                total_in: &total_in,
                total_out: &total_out,
            };
            let enc_res = run_nvenc_stage(&enc_ctx, &rx_for_nvenc, preopened);
            drop(rx_for_nvenc);

            let dec_res: Vec<Result<(), PipelineError>> = handles
                .into_iter()
                .map(|h| h.join().unwrap_or(Err(PipelineError::WorkerPanicked)))
                .collect();

            (dec_res, enc_res)
        });

        nvenc_result?;
        for res in decode_results {
            res?;
        }

        Ok((
            total_in.load(Ordering::Relaxed),
            total_out.load(Ordering::Relaxed),
        ))
    }

    fn run_decode_worker(
        items: &[ImageBatchItem],
        next_decode: &AtomicUsize,
        failed: &AtomicBool,
        rgba_pool: &Mutex<Vec<Vec<u8>>>,
        tx_to_nvenc: &SyncSender<DecodedRgbaFrame>,
    ) -> Result<(), PipelineError> {
        let mut file_bytes = Vec::new();
        let mut pixel_buf = Vec::new();

        while !failed.load(Ordering::Relaxed) {
            let idx = next_decode.fetch_add(1, Ordering::Relaxed);
            let Some(item) = items.get(idx) else {
                break;
            };
            let rgba_buf = match pop_pooled_buf(rgba_pool) {
                Ok(buf) => buf,
                Err(err) => {
                    failed.store(true, Ordering::Relaxed);
                    return Err(err);
                }
            };
            let frame =
                match decode_to_rgba_frame(idx, item, &mut file_bytes, &mut pixel_buf, rgba_buf) {
                    Ok(frame) => frame,
                    Err(err) => {
                        failed.store(true, Ordering::Relaxed);
                        return Err(err);
                    }
                };
            if tx_to_nvenc.send(frame).is_err() {
                break;
            }
        }
        Ok(())
    }

    struct NvencStageContext<'a> {
        cuda_ctx: &'a Arc<CudaContext>,
        items: &'a [ImageBatchItem],
        qp: u32,
        failed: &'a AtomicBool,
        rgba_pool: &'a Mutex<Vec<Vec<u8>>>,
        total_in: &'a AtomicU64,
        total_out: &'a AtomicU64,
    }

    fn run_nvenc_stage(
        ctx: &NvencStageContext<'_>,
        rx_for_nvenc: &Receiver<DecodedRgbaFrame>,
        mut preopened: Option<ActiveHevcSession>,
    ) -> Result<(), PipelineError> {
        let mut pitched_buf = Vec::new();
        let mut pending = rx_for_nvenc.recv().ok();

        while let Some(first_frame) = pending.take() {
            if ctx.failed.load(Ordering::Relaxed) {
                break;
            }
            let session_w = first_frame.width;
            let session_h = first_frame.height;
            let active = match take_or_open_session(
                ctx.cuda_ctx,
                &mut preopened,
                session_w,
                session_h,
                ctx.qp,
            ) {
                Ok(sess) => sess,
                Err(err) => {
                    ctx.failed.store(true, Ordering::Relaxed);
                    return Err(err);
                }
            };
            let (mut input, mut bitstream, pitch) = match allocate_session_buffers(&active.session)
            {
                Ok(bufs) => bufs,
                Err(err) => {
                    ctx.failed.store(true, Ordering::Relaxed);
                    return Err(err);
                }
            };

            let mut current = Some(first_frame);
            while let Some(frame) = current.take() {
                if ctx.failed.load(Ordering::Relaxed) {
                    return Ok(());
                }
                if frame.width != session_w || frame.height != session_h {
                    pending = Some(frame);
                    break;
                }
                let Some(item) = ctx.items.get(frame.item_index) else {
                    ctx.failed.store(true, Ordering::Relaxed);
                    return Err(PipelineError::GpuBackendFailed {
                        backend: GpuBackend::Nvenc,
                        reason: "internal batch item index out of range".to_owned(),
                    });
                };
                match encode_into_session(
                    &active,
                    &mut input,
                    &mut bitstream,
                    pitch,
                    item,
                    &frame,
                    &mut pitched_buf,
                ) {
                    Ok(out_bytes) => {
                        ctx.total_in
                            .fetch_add(frame.input_file_bytes, Ordering::Relaxed);
                        ctx.total_out.fetch_add(out_bytes, Ordering::Relaxed);
                        if let Err(err) = push_pooled_buf(ctx.rgba_pool, frame.rgba_bytes) {
                            ctx.failed.store(true, Ordering::Relaxed);
                            return Err(err);
                        }
                    }
                    Err(err) => {
                        ctx.failed.store(true, Ordering::Relaxed);
                        return Err(err);
                    }
                }
                current = rx_for_nvenc.recv().ok();
            }
        }
        Ok(())
    }

    fn allocate_session_buffers(
        session: &Session,
    ) -> Result<(Buffer<'_>, Bitstream<'_>, u32), PipelineError> {
        let mut input =
            session
                .create_input_buffer()
                .map_err(|err| PipelineError::GpuBackendFailed {
                    backend: GpuBackend::Nvenc,
                    reason: format!("failed to allocate NVENC input buffer: {err}"),
                })?;
        // `nvidia-video-codec-sdk` initializes `Buffer.pitch` to pixel width until the first
        // `lock()` populates the actual byte stride from `NV_ENC_LOCK_INPUT_BUFFER.pitch`.
        {
            let _probe_lock = input
                .lock()
                .map_err(|err| PipelineError::GpuBackendFailed {
                    backend: GpuBackend::Nvenc,
                    reason: format!("failed to probe NVENC input buffer pitch: {err}"),
                })?;
        }
        let pitch = input.pitch();
        let bitstream =
            session
                .create_output_bitstream()
                .map_err(|err| PipelineError::GpuBackendFailed {
                    backend: GpuBackend::Nvenc,
                    reason: format!("failed to allocate NVENC output bitstream: {err}"),
                })?;
        Ok((input, bitstream, pitch))
    }

    fn decode_to_rgba_frame(
        item_index: usize,
        item: &ImageBatchItem,
        file_bytes: &mut Vec<u8>,
        pixel_buf: &mut Vec<u8>,
        mut rgba_bytes: Vec<u8>,
    ) -> Result<DecodedRgbaFrame, PipelineError> {
        match item.input_format() {
            ImageInputFormat::Jpeg
            | ImageInputFormat::Qoi
            | ImageInputFormat::Bmp
            | ImageInputFormat::Heic => {
                let decoded = decode_image_into(
                    item,
                    DecodeTargetHint::OpaqueRgba8,
                    file_bytes,
                    &mut rgba_bytes,
                )?;
                if decoded.format == DecodedPixelFormat::Rgba8 {
                    rgba_bytes.truncate(decoded.byte_len);
                    return Ok(DecodedRgbaFrame {
                        item_index,
                        width: decoded.width,
                        height: decoded.height,
                        rgba_bytes,
                        input_file_bytes: decoded.input_file_bytes,
                    });
                }
                std::mem::swap(pixel_buf, &mut rgba_bytes);
                finish_rgba_expansion(item_index, item, decoded, pixel_buf, rgba_bytes)
            }
            ImageInputFormat::Tiff => {
                let decoded =
                    decode_image_into(item, DecodeTargetHint::OpaqueRgba8, file_bytes, pixel_buf)?;
                finish_rgba_expansion(item_index, item, decoded, pixel_buf, rgba_bytes)
            }
            ImageInputFormat::Png
            | ImageInputFormat::Webp
            | ImageInputFormat::Jxl
            | ImageInputFormat::Avif
            | ImageInputFormat::Gif
            | ImageInputFormat::Ico
            | ImageInputFormat::Pnm => {
                let decoded =
                    decode_image_into(item, DecodeTargetHint::Native, file_bytes, pixel_buf)?;
                finish_rgba_expansion(item_index, item, decoded, pixel_buf, rgba_bytes)
            }
        }
    }

    fn finish_rgba_expansion(
        item_index: usize,
        item: &ImageBatchItem,
        decoded: DecodedFrameMeta,
        pixel_buf: &[u8],
        mut rgba_bytes: Vec<u8>,
    ) -> Result<DecodedRgbaFrame, PipelineError> {
        if decoded.has_non_opaque_native_alpha {
            return Err(PipelineError::UnsupportedPixelFormat {
                path: item.source().to_path_buf(),
                format: "TIFF alpha sample is not fully opaque at native bit depth".to_owned(),
            });
        }
        let pixels =
            pixel_buf
                .get(..decoded.byte_len)
                .ok_or_else(|| PipelineError::DecodeFailed {
                    path: item.source().to_path_buf(),
                    reason: "decoded frame exceeded allocated pixel buffer".to_owned(),
                })?;

        let pixel_count = usize::try_from(decoded.width)
            .ok()
            .and_then(|w| {
                usize::try_from(decoded.height)
                    .ok()
                    .and_then(|h| w.checked_mul(h))
            })
            .ok_or_else(|| PipelineError::DecodeFailed {
                path: item.source().to_path_buf(),
                reason: "decoded image pixel count overflow".to_owned(),
            })?;
        expand_opaque_pixels_to_rgba(
            item.source(),
            pixels,
            decoded.format,
            pixel_count,
            &mut rgba_bytes,
        )?;

        Ok(DecodedRgbaFrame {
            item_index,
            width: decoded.width,
            height: decoded.height,
            rgba_bytes,
            input_file_bytes: decoded.input_file_bytes,
        })
    }

    fn expand_opaque_pixels_to_rgba(
        source: &std::path::Path,
        pixels: &[u8],
        format: DecodedPixelFormat,
        pixel_count: usize,
        rgba_bytes: &mut Vec<u8>,
    ) -> Result<(), PipelineError> {
        let src_bpp = match format {
            DecodedPixelFormat::L8 => 1_usize,
            DecodedPixelFormat::La8 => 2,
            DecodedPixelFormat::Rgb8 => 3,
            DecodedPixelFormat::Rgba8 => 4,
        };
        let expected_src_len =
            pixel_count
                .checked_mul(src_bpp)
                .ok_or_else(|| PipelineError::DecodeFailed {
                    path: source.to_path_buf(),
                    reason: "source pixel buffer size overflow".to_owned(),
                })?;
        if pixels.len() != expected_src_len {
            return Err(PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: format!(
                    "decoded image pixel byte count {} does not match expected {expected_src_len}",
                    pixels.len()
                ),
            });
        }

        let rgba_len = pixel_count
            .checked_mul(4)
            .ok_or_else(|| PipelineError::DecodeFailed {
                path: source.to_path_buf(),
                reason: "RGBA buffer size overflow".to_owned(),
            })?;
        rgba_bytes.resize(rgba_len, 255);

        match format {
            DecodedPixelFormat::Rgb8 => {
                let (triplets, _) = pixels.as_chunks::<3>();
                let (dst_quads, _) = rgba_bytes.as_chunks_mut::<4>();
                for (&[r, g, b], dst) in triplets.iter().zip(dst_quads.iter_mut()) {
                    *dst = [r, g, b, 255];
                }
            }
            DecodedPixelFormat::Rgba8 => {
                let (quads, _) = pixels.as_chunks::<4>();
                if quads.iter().any(|&[_, _, _, a]| a < 255) {
                    return Err(PipelineError::UnsupportedPixelFormat {
                        path: source.to_path_buf(),
                        format: "non-opaque alpha channel (NVENC HEVC-Intra HEIC currently supports opaque images)".to_owned(),
                    });
                }
                rgba_bytes.copy_from_slice(pixels);
            }
            DecodedPixelFormat::L8 => {
                let (dst_quads, _) = rgba_bytes.as_chunks_mut::<4>();
                for (&v, dst) in pixels.iter().zip(dst_quads.iter_mut()) {
                    *dst = [v, v, v, 255];
                }
            }
            DecodedPixelFormat::La8 => {
                let (pairs, _) = pixels.as_chunks::<2>();
                let (dst_quads, _) = rgba_bytes.as_chunks_mut::<4>();
                for (&[v, a], dst) in pairs.iter().zip(dst_quads.iter_mut()) {
                    if a < 255 {
                        return Err(PipelineError::UnsupportedPixelFormat {
                            path: source.to_path_buf(),
                            format: "non-opaque alpha channel (NVENC HEVC-Intra HEIC currently supports opaque images)".to_owned(),
                        });
                    }
                    *dst = [v, v, v, 255];
                }
            }
        }
        Ok(())
    }

    fn prepare_pitched_payload<'a>(
        rgba_bytes: &'a [u8],
        width: u32,
        height: u32,
        pitch_u32: u32,
        pitched_buf: &'a mut Vec<u8>,
    ) -> Result<(&'a [u8], usize), PipelineError> {
        let row_bytes = usize::try_from(width)
            .ok()
            .and_then(|w| w.checked_mul(4))
            .ok_or_else(|| PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                reason: "row byte stride overflow".to_owned(),
            })?;
        let height_usize =
            usize::try_from(height).map_err(|_| PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                reason: "height exceeds usize".to_owned(),
            })?;
        let expected_rgba_len =
            row_bytes
                .checked_mul(height_usize)
                .ok_or_else(|| PipelineError::GpuBackendFailed {
                    backend: GpuBackend::Nvenc,
                    reason: "RGBA input buffer size overflow".to_owned(),
                })?;
        if rgba_bytes.len() != expected_rgba_len {
            return Err(PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                reason: format!(
                    "RGBA input buffer size mismatch: got {}, expected {expected_rgba_len}",
                    rgba_bytes.len()
                ),
            });
        }
        let pitch = usize::try_from(pitch_u32).map_err(|_| PipelineError::GpuBackendFailed {
            backend: GpuBackend::Nvenc,
            reason: "NVENC buffer pitch exceeds usize".to_owned(),
        })?;
        if pitch < row_bytes {
            return Err(PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                reason: format!(
                    "NVENC buffer pitch ({pitch}) is smaller than row width ({row_bytes})"
                ),
            });
        }
        let expected_len =
            pitch
                .checked_mul(height_usize)
                .ok_or_else(|| PipelineError::GpuBackendFailed {
                    backend: GpuBackend::Nvenc,
                    reason: "pitched NVENC buffer size overflow".to_owned(),
                })?;

        if pitch == row_bytes {
            return Ok((rgba_bytes, expected_len));
        }

        pitched_buf.clear();
        pitched_buf.resize(expected_len, 0);
        for row in 0..height_usize {
            let src_start = row * row_bytes;
            let dst_start = row * pitch;
            let src = rgba_bytes
                .get(src_start..src_start + row_bytes)
                .ok_or_else(|| PipelineError::GpuBackendFailed {
                    backend: GpuBackend::Nvenc,
                    reason: "internal pitched source row slice out of bounds".to_owned(),
                })?;
            let dst = pitched_buf
                .get_mut(dst_start..dst_start + row_bytes)
                .ok_or_else(|| PipelineError::GpuBackendFailed {
                    backend: GpuBackend::Nvenc,
                    reason: "internal pitched destination row slice out of bounds".to_owned(),
                })?;
            dst.copy_from_slice(src);
        }
        Ok((pitched_buf.as_slice(), expected_len))
    }

    fn encode_into_session(
        active: &ActiveHevcSession,
        input: &mut Buffer<'_>,
        bitstream: &mut Bitstream<'_>,
        pitch: u32,
        item: &ImageBatchItem,
        frame: &DecodedRgbaFrame,
        pitched_buf: &mut Vec<u8>,
    ) -> Result<u64, PipelineError> {
        let (upload_slice, expected_len) = prepare_pitched_payload(
            &frame.rgba_bytes,
            frame.width,
            frame.height,
            pitch,
            pitched_buf,
        )?;
        write_pitched_nvenc_buffer(input, expected_len, upload_slice)?;

        active
            .session
            .encode_picture(
                input,
                bitstream,
                EncodePictureParams {
                    picture_type: NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_IDR,
                    ..Default::default()
                },
            )
            .map_err(|err| PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                reason: format!(
                    "NVENC HEVC encode failed for '{}': {err}",
                    item.source().display()
                ),
            })?;

        let lock = bitstream
            .lock()
            .map_err(|err| PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                reason: format!("failed to lock NVENC output bitstream: {err}"),
            })?;
        // NVENC converts ABGR input internally to BT.601 limited-range NV12, so match `colr` metadata.
        let heic_bytes = mux_annex_b_hevc_to_heic(
            lock.data(),
            frame.width,
            frame.height,
            HeicColorInfo::BT601_LIMITED,
        )
        .map_err(|err| PipelineError::EncodeFailed {
            path: item.target().to_path_buf(),
            reason: err.to_string(),
        })?;
        drop(lock);

        if heic_bytes.is_empty() {
            return Err(PipelineError::EmptyOutput(item.target().to_path_buf()));
        }

        fs::write(item.target(), &heic_bytes).map_err(|source| PipelineError::Io {
            path: item.target().to_path_buf(),
            source,
        })?;

        Ok(u64::try_from(heic_bytes.len()).unwrap_or(u64::MAX))
    }

    // `nvidia-video-codec-sdk`'s `BufferLock::write` is marked `unsafe` because it does not
    // validate buffer bounds itself; we verify `data.len() == expected_len == pitch * height`.
    #[allow(unsafe_code)]
    fn write_pitched_nvenc_buffer(
        input: &mut Buffer<'_>,
        expected_len: usize,
        data: &[u8],
    ) -> Result<(), PipelineError> {
        if data.len() != expected_len {
            return Err(PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                reason: format!(
                    "NVENC input payload size mismatch: got {}, expected {expected_len}",
                    data.len()
                ),
            });
        }
        let mut lock = input
            .lock()
            .map_err(|err| PipelineError::GpuBackendFailed {
                backend: GpuBackend::Nvenc,
                reason: format!("failed to lock NVENC input buffer: {err}"),
            })?;
        // SAFETY: `input` was allocated by `Session::create_input_buffer` for `ABGR`
        // (`pitch * height` bytes = `expected_len`), and we verified `data.len() == expected_len`.
        unsafe {
            lock.write(data);
        }
        Ok(())
    }

    fn pop_pooled_buf(pool: &Mutex<Vec<Vec<u8>>>) -> Result<Vec<u8>, PipelineError> {
        let mut guard = pool.lock().map_err(|_| PipelineError::WorkerPanicked)?;
        Ok(guard.pop().unwrap_or_default())
    }

    fn push_pooled_buf(pool: &Mutex<Vec<Vec<u8>>>, mut buf: Vec<u8>) -> Result<(), PipelineError> {
        buf.clear();
        let mut guard = pool.lock().map_err(|_| PipelineError::WorkerPanicked)?;
        if guard.len() < 4 {
            guard.push(buf);
        }
        Ok(())
    }

    fn video_error(reason: impl Into<String>) -> PipelineError {
        PipelineError::GpuBackendFailed {
            backend: GpuBackend::Nvenc,
            reason: reason.into(),
        }
    }

    /// VBR mode in both cases: with zero bitrate/VBV constraints NVENC treats
    /// `targetQuality` as Constant Quality; with `averageBitRate` set it runs
    /// true average-bitrate rate control and ignores `targetQuality`.
    fn configure_video_rc(
        rc: &mut NV_ENC_RC_PARAMS,
        rate_control: VideoRateControl,
        temporal_aq: bool,
        lookahead_depth: u16,
        disable_b_adapt: bool,
    ) {
        match rate_control {
            VideoRateControl::Quality(quality) => {
                let (target, lsb) = quality_to_nvenc_cq(quality);
                rc.averageBitRate = 0;
                rc.maxBitRate = 0;
                rc.vbvBufferSize = 0;
                rc.vbvInitialDelay = 0;
                rc.targetQuality = target;
                rc.targetQualityLSB = lsb;
            }
            VideoRateControl::Bitrate(bitrate) => {
                let bps = bitrate.bps();
                rc.averageBitRate = bps;
                rc.maxBitRate = bps.saturating_mul(3) / 2;
                rc.vbvBufferSize = 0;
                rc.vbvInitialDelay = 0;
                rc.targetQuality = 0;
                rc.targetQualityLSB = 0;
            }
        }
        rc.rateControlMode = NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_VBR;
        rc.set_enableMinQP(0);
        rc.set_enableMaxQP(0);
        rc.set_enableInitialRCQP(0);
        rc.set_enableAQ(1);
        rc.set_aqStrength(NVENC_AQ_STRENGTH);
        rc.set_enableTemporalAQ(u32::from(temporal_aq));
        rc.set_enableLookahead(u32::from(lookahead_depth > 0));
        rc.lookaheadDepth = lookahead_depth;
        rc.set_disableBadapt(u32::from(disable_b_adapt));
    }

    /// One value drives the encode GUID, the codec-config union member, and
    /// the output codec, so a mismatched combination cannot be constructed.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum NvencVideoCodec {
        H264,
        Hevc,
    }

    impl NvencVideoCodec {
        fn from_mode(mode: VideoEncodeMode) -> Result<(Self, VideoRateControl), PipelineError> {
            match mode {
                VideoEncodeMode::NvencH264 { rate_control } => Ok((Self::H264, rate_control)),
                VideoEncodeMode::NvencHevc { rate_control } => Ok((Self::Hevc, rate_control)),
                other => Err(video_error(format!(
                    "NVENC video encoding rejects mode {other}; only H.264 and HEVC are supported"
                ))),
            }
        }

        fn guid(self) -> GUID {
            match self {
                Self::H264 => NV_ENC_CODEC_H264_GUID,
                Self::Hevc => NV_ENC_CODEC_HEVC_GUID,
            }
        }

        fn output_codec(self) -> VideoCodec {
            match self {
                Self::H264 => VideoCodec::H264,
                Self::Hevc => VideoCodec::Hevc,
            }
        }

        /// NVENC exposes codec-specific config in a C union (`NV_ENC_CODEC_CONFIG`).
        /// The preset was queried with `self.guid()`, so this selects its one
        /// valid member.
        #[allow(unsafe_code)]
        fn configure_codec_config(
            self,
            cfg: &mut NV_ENC_CONFIG,
            gop_length: u32,
            bframe_ref_mode: NV_ENC_BFRAME_REF_MODE,
        ) {
            match self {
                Self::H264 => {
                    // SAFETY: `cfg` came from `get_preset_config(NV_ENC_CODEC_H264_GUID, ...)`.
                    let h264_cfg = unsafe { &mut cfg.encodeCodecConfig.h264Config };
                    h264_cfg.set_repeatSPSPPS(1);
                    h264_cfg.set_outputAUD(0);
                    h264_cfg.chromaFormatIDC = 1;
                    h264_cfg.idrPeriod = gop_length;
                    h264_cfg.useBFramesAsRef = bframe_ref_mode;
                }
                Self::Hevc => {
                    // SAFETY: `cfg` came from `get_preset_config(NV_ENC_CODEC_HEVC_GUID, ...)`.
                    let hevc_cfg = unsafe { &mut cfg.encodeCodecConfig.hevcConfig };
                    hevc_cfg.set_repeatSPSPPS(1);
                    hevc_cfg.set_outputAUD(0);
                    hevc_cfg.set_chromaFormatIDC(1);
                    hevc_cfg.set_pixelBitDepthMinus8(0);
                    hevc_cfg.idrPeriod = gop_length;
                    hevc_cfg.useBFramesAsRef = bframe_ref_mode;
                }
            }
        }

        fn is_keyframe(self, data: &[u8]) -> bool {
            match self {
                Self::H264 => is_h264_keyframe_annex_b(data),
                Self::Hevc => is_hevc_keyframe_annex_b(data),
            }
        }

        fn extract_config(self, data: &[u8]) -> Option<Vec<u8>> {
            match self {
                Self::H264 => extract_avcc_from_annex_b(data),
                Self::Hevc => extract_hvcc_from_annex_b(data),
            }
        }

        fn to_sample(self, data: &[u8]) -> Vec<u8> {
            match self {
                Self::H264 => annex_b_to_avcc(data),
                Self::Hevc => annex_b_to_hvcc(data),
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum SubmitOutcome {
        Ready,
        NeedMoreInput,
    }

    // NVENC SDK 0.4.0 hides `Encoder.ptr` (so `nvEncGetEncodeCaps` cannot be
    // called on it) and panics inside `Session::drop` when a second EOS fails.
    // This thin RAII wrapper calls `ENCODE_API` directly for the video path so
    // capability queries work on every GPU generation and EOS/teardown errors
    // return cleanly without panicking.
    #[allow(unsafe_code)]
    fn nvenc_last_error(ptr: *mut c_void) -> Option<String> {
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ptr` is a live encoder handle opened via `open_encode_session_ex`.
        let raw = unsafe { (ENCODE_API.get_last_error_string)(ptr) };
        if raw.is_null() {
            return None;
        }
        // SAFETY: NVENC returns a NUL-terminated static or session-owned C string.
        let text = unsafe { CStr::from_ptr(raw) }.to_string_lossy();
        (!text.is_empty()).then(|| text.into_owned())
    }

    fn nvenc_check(
        ptr: *mut c_void,
        status: NVENCSTATUS,
        context: &str,
    ) -> Result<(), PipelineError> {
        if status == NVENCSTATUS::NV_ENC_SUCCESS {
            return Ok(());
        }
        let detail = nvenc_last_error(ptr)
            .map_or_else(|| format!("{status:?}"), |msg| format!("{status:?}: {msg}"));
        Err(video_error(format!("{context}: {detail}")))
    }

    struct RawNvencEncoder {
        ptr: *mut c_void,
        _cuda_ctx: Arc<CudaContext>,
    }

    impl RawNvencEncoder {
        #[allow(unsafe_code)]
        fn open(cuda_ctx: Arc<CudaContext>) -> Result<Self, PipelineError> {
            let mut ptr = ptr::null_mut();
            let mut params = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {
                version: NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER,
                deviceType: NV_ENC_DEVICE_TYPE::NV_ENC_DEVICE_TYPE_CUDA,
                apiVersion: NVENCAPI_VERSION,
                device: cuda_ctx.cu_ctx().cast::<c_void>(),
                ..Default::default()
            };
            // SAFETY: `params` is properly versioned and `cuda_ctx` stays alive for the session.
            let status =
                unsafe { (ENCODE_API.open_encode_session_ex)(&raw mut params, &raw mut ptr) };
            if status != NVENCSTATUS::NV_ENC_SUCCESS {
                if !ptr.is_null() {
                    // SAFETY: NVENC requires destroying a failed handle if non-null.
                    let _destroy_status = unsafe { (ENCODE_API.destroy_encoder)(ptr) };
                }
                return Err(video_error(format!(
                    "failed to open NVENC encoder: {status:?}"
                )));
            }
            Ok(Self {
                ptr,
                _cuda_ctx: cuda_ctx,
            })
        }

        #[allow(unsafe_code)]
        fn get_encode_guids(&self) -> Result<Vec<GUID>, PipelineError> {
            let mut count = 0_u32;
            nvenc_check(
                self.ptr,
                unsafe { (ENCODE_API.get_encode_guid_count)(self.ptr, &raw mut count) },
                "failed to query NVENC codec count",
            )?;
            let cap = usize::try_from(count).unwrap_or(0);
            let mut guids = vec![GUID::default(); cap];
            let mut actual = 0_u32;
            nvenc_check(
                self.ptr,
                unsafe {
                    (ENCODE_API.get_encode_guids)(
                        self.ptr,
                        guids.as_mut_ptr(),
                        count,
                        &raw mut actual,
                    )
                },
                "failed to query NVENC codecs",
            )?;
            guids.truncate(usize::try_from(actual).unwrap_or(0));
            Ok(guids)
        }

        #[allow(unsafe_code)]
        fn get_supported_input_formats(
            &self,
            guid: GUID,
        ) -> Result<Vec<NV_ENC_BUFFER_FORMAT>, PipelineError> {
            let mut count = 0_u32;
            nvenc_check(
                self.ptr,
                unsafe { (ENCODE_API.get_input_format_count)(self.ptr, guid, &raw mut count) },
                "failed to query NVENC input format count",
            )?;
            let cap = usize::try_from(count).unwrap_or(0);
            let mut formats = vec![NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_UNDEFINED; cap];
            let mut actual = 0_u32;
            nvenc_check(
                self.ptr,
                unsafe {
                    (ENCODE_API.get_input_formats)(
                        self.ptr,
                        guid,
                        formats.as_mut_ptr(),
                        count,
                        &raw mut actual,
                    )
                },
                "failed to query NVENC input formats",
            )?;
            formats.truncate(usize::try_from(actual).unwrap_or(0));
            Ok(formats)
        }

        #[allow(unsafe_code)]
        fn get_encode_cap(&self, guid: GUID, cap: NV_ENC_CAPS) -> Result<i32, PipelineError> {
            let mut param = NV_ENC_CAPS_PARAM {
                version: NV_ENC_CAPS_PARAM_VER,
                capsToQuery: cap,
                ..Default::default()
            };
            let mut value: c_int = 0;
            nvenc_check(
                self.ptr,
                unsafe {
                    (ENCODE_API.get_encode_caps)(self.ptr, guid, &raw mut param, &raw mut value)
                },
                "failed to query NVENC codec capability",
            )?;
            Ok(value)
        }

        fn query_codec_caps(&self, guid: GUID) -> Result<NvencCodecCaps, PipelineError> {
            Ok(NvencCodecCaps {
                max_b_frames: self
                    .get_encode_cap(guid, NV_ENC_CAPS::NV_ENC_CAPS_NUM_MAX_BFRAMES)?,
                bframe_ref_mode: self
                    .get_encode_cap(guid, NV_ENC_CAPS::NV_ENC_CAPS_SUPPORT_BFRAME_REF_MODE)?,
                lookahead_supported: self
                    .get_encode_cap(guid, NV_ENC_CAPS::NV_ENC_CAPS_SUPPORT_LOOKAHEAD)?
                    > 0,
                temporal_aq_supported: self
                    .get_encode_cap(guid, NV_ENC_CAPS::NV_ENC_CAPS_SUPPORT_TEMPORAL_AQ)?
                    > 0,
            })
        }

        #[allow(unsafe_code)]
        fn get_preset_config(
            &self,
            guid: GUID,
            preset: GUID,
            tuning: NV_ENC_TUNING_INFO,
        ) -> Result<NV_ENC_PRESET_CONFIG, PipelineError> {
            let mut preset_config = NV_ENC_PRESET_CONFIG {
                version: NV_ENC_PRESET_CONFIG_VER,
                presetCfg: NV_ENC_CONFIG {
                    version: NV_ENC_CONFIG_VER,
                    ..Default::default()
                },
                ..Default::default()
            };
            nvenc_check(
                self.ptr,
                unsafe {
                    (ENCODE_API.get_encode_preset_config_ex)(
                        self.ptr,
                        guid,
                        preset,
                        tuning,
                        &raw mut preset_config,
                    )
                },
                "failed to query NVENC P4 preset",
            )?;
            Ok(preset_config)
        }

        #[allow(unsafe_code)]
        fn start_session(
            self,
            width: u32,
            height: u32,
            buffer_format: NV_ENC_BUFFER_FORMAT,
            init_params: &mut NV_ENC_INITIALIZE_PARAMS,
        ) -> Result<RawNvencSession, PipelineError> {
            nvenc_check(
                self.ptr,
                unsafe { (ENCODE_API.initialize_encoder)(self.ptr, init_params) },
                "failed to start NVENC video session",
            )?;
            Ok(RawNvencSession {
                encoder: self,
                width,
                height,
                buffer_format,
                eos_sent: Cell::new(false),
            })
        }
    }

    impl Drop for RawNvencEncoder {
        #[allow(unsafe_code)]
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                // SAFETY: all bitstreams and mapped resources borrow `RawNvencSession`
                // and therefore drop before this encoder handle. Never panic on drop.
                let _status = unsafe { (ENCODE_API.destroy_encoder)(self.ptr) };
            }
        }
    }

    struct RawBitstream<'a> {
        ptr: *mut c_void,
        encoder_ptr: *mut c_void,
        /// Set between a successful `NvEncLockBitstream` and a successful
        /// `NvEncUnlockBitstream`. A locked bitstream must never be destroyed.
        locked: Cell<bool>,
        _session: std::marker::PhantomData<&'a RawNvencSession>,
    }

    impl Drop for RawBitstream<'_> {
        #[allow(unsafe_code)]
        fn drop(&mut self) {
            if self.ptr.is_null() {
                return;
            }
            if self.locked.get() {
                // Retry the unlock while the encoder is still alive. If the
                // driver keeps refusing, leak this buffer rather than destroy
                // one it may still consider locked.
                let status = unsafe { (ENCODE_API.unlock_bitstream)(self.encoder_ptr, self.ptr) };
                if status != NVENCSTATUS::NV_ENC_SUCCESS {
                    return;
                }
                self.locked.set(false);
            }
            // SAFETY: `self.ptr` was allocated by `create_bitstream_buffer` on `encoder_ptr`.
            let _status =
                unsafe { (ENCODE_API.destroy_bitstream_buffer)(self.encoder_ptr, self.ptr) };
        }
    }

    /// A CUDA device pointer registered once for the lifetime of the session.
    ///
    /// Registration is expensive, so all NVENC ring slots are registered at
    /// session init and reused; only the lightweight map/unmap pair is per frame.
    struct RawRegisteredResource<'a> {
        reg_ptr: *mut c_void,
        encoder_ptr: *mut c_void,
        _session: std::marker::PhantomData<&'a RawNvencSession>,
    }

    impl Drop for RawRegisteredResource<'_> {
        #[allow(unsafe_code)]
        fn drop(&mut self) {
            // SAFETY: all mappings borrow this registration and drop first.
            if !self.reg_ptr.is_null() {
                let _unreg =
                    unsafe { (ENCODE_API.unregister_resource)(self.encoder_ptr, self.reg_ptr) };
            }
        }
    }

    /// A short-lived mapping of a registered slot, held only from `encode_picture`
    /// until the matching output bitstream has been drained. Unmapping before a
    /// CUDA copy reuses the slot prevents NVENC from reading a surface being written.
    struct RawMappedResource<'a> {
        map_ptr: *mut c_void,
        pitch: u32,
        encoder_ptr: *mut c_void,
        _registered: &'a RawRegisteredResource<'a>,
    }

    impl Drop for RawMappedResource<'_> {
        #[allow(unsafe_code)]
        fn drop(&mut self) {
            // SAFETY: unmapping while the encoder and registration are alive.
            if !self.map_ptr.is_null() {
                let _unmap =
                    unsafe { (ENCODE_API.unmap_input_resource)(self.encoder_ptr, self.map_ptr) };
            }
        }
    }

    struct RawNvencSession {
        encoder: RawNvencEncoder,
        width: u32,
        height: u32,
        buffer_format: NV_ENC_BUFFER_FORMAT,
        eos_sent: Cell<bool>,
    }

    impl RawNvencSession {
        #[allow(unsafe_code)]
        fn create_bitstream(&self) -> Result<RawBitstream<'_>, PipelineError> {
            let mut params = NV_ENC_CREATE_BITSTREAM_BUFFER {
                version: NV_ENC_CREATE_BITSTREAM_BUFFER_VER,
                bitstreamBuffer: ptr::null_mut(),
                ..Default::default()
            };
            nvenc_check(
                self.encoder.ptr,
                unsafe { (ENCODE_API.create_bitstream_buffer)(self.encoder.ptr, &raw mut params) },
                "failed to allocate NVENC output bitstream",
            )?;
            Ok(RawBitstream {
                ptr: params.bitstreamBuffer,
                encoder_ptr: self.encoder.ptr,
                locked: Cell::new(false),
                _session: std::marker::PhantomData,
            })
        }

        #[allow(unsafe_code)]
        fn register_cuda_resource(
            &self,
            resource_ptr: *mut c_void,
            pitch: u32,
        ) -> Result<RawRegisteredResource<'_>, PipelineError> {
            let mut reg_params = NV_ENC_REGISTER_RESOURCE {
                version: NV_ENC_REGISTER_RESOURCE_VER,
                resourceType: NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                width: self.width,
                height: self.height,
                pitch,
                resourceToRegister: resource_ptr,
                bufferFormat: self.buffer_format,
                ..Default::default()
            };
            nvenc_check(
                self.encoder.ptr,
                unsafe { (ENCODE_API.register_resource)(self.encoder.ptr, &raw mut reg_params) },
                "failed to register NVENC CUDA device buffer",
            )?;
            Ok(RawRegisteredResource {
                reg_ptr: reg_params.registeredResource,
                encoder_ptr: self.encoder.ptr,
                _session: std::marker::PhantomData,
            })
        }

        #[allow(unsafe_code)]
        fn map_cuda_resource<'a>(
            &self,
            registered: &'a RawRegisteredResource<'a>,
            pitch: u32,
        ) -> Result<RawMappedResource<'a>, PipelineError> {
            let mut map_params = NV_ENC_MAP_INPUT_RESOURCE {
                version: NV_ENC_MAP_INPUT_RESOURCE_VER,
                registeredResource: registered.reg_ptr,
                mappedResource: ptr::null_mut(),
                mappedBufferFmt: NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_UNDEFINED,
                ..Default::default()
            };
            nvenc_check(
                self.encoder.ptr,
                unsafe { (ENCODE_API.map_input_resource)(self.encoder.ptr, &raw mut map_params) },
                "failed to map NVENC CUDA device buffer",
            )?;
            Ok(RawMappedResource {
                map_ptr: map_params.mappedResource,
                pitch,
                encoder_ptr: self.encoder.ptr,
                _registered: registered,
            })
        }

        /// Submits one picture, retrying transient `EncoderBusy` and reporting
        /// `NeedMoreInput` cleanly without locking the bitstream.
        #[allow(unsafe_code)]
        fn encode_picture(
            &self,
            input: &RawMappedResource<'_>,
            bitstream: &mut RawBitstream<'_>,
            input_timestamp: u64,
        ) -> Result<SubmitOutcome, PipelineError> {
            let mut pic_params = NV_ENC_PIC_PARAMS {
                version: NV_ENC_PIC_PARAMS_VER,
                inputWidth: self.width,
                inputHeight: self.height,
                inputPitch: input.pitch,
                inputBuffer: input.map_ptr,
                outputBitstream: bitstream.ptr,
                bufferFmt: self.buffer_format,
                pictureStruct: NV_ENC_PIC_STRUCT::NV_ENC_PIC_STRUCT_FRAME,
                inputTimeStamp: input_timestamp,
                pictureType: NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_UNKNOWN,
                ..Default::default()
            };
            for _ in 0..1000 {
                let status =
                    unsafe { (ENCODE_API.encode_picture)(self.encoder.ptr, &raw mut pic_params) };
                match status {
                    NVENCSTATUS::NV_ENC_SUCCESS => return Ok(SubmitOutcome::Ready),
                    NVENCSTATUS::NV_ENC_ERR_NEED_MORE_INPUT => {
                        return Ok(SubmitOutcome::NeedMoreInput);
                    }
                    NVENCSTATUS::NV_ENC_ERR_ENCODER_BUSY => {
                        thread::sleep(Duration::from_millis(1));
                    }
                    other => {
                        nvenc_check(self.encoder.ptr, other, "NVENC video encode failed")?;
                    }
                }
            }
            Err(video_error(
                "NVENC encoder stayed busy while submitting a picture",
            ))
        }

        /// Flushes the NVENC pipeline with an EOS packet. Marks `eos_sent` on
        /// attempt so `Drop` never retries EOS or panics if retries were exhausted.
        #[allow(unsafe_code)]
        fn flush_eos(&self) -> Result<(), PipelineError> {
            self.eos_sent.set(true);
            let mut eos = NV_ENC_PIC_PARAMS {
                version: NV_ENC_PIC_PARAMS_VER,
                encodePicFlags: NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_EOS as u32,
                ..Default::default()
            };
            for _ in 0..1000 {
                let status = unsafe { (ENCODE_API.encode_picture)(self.encoder.ptr, &raw mut eos) };
                match status {
                    NVENCSTATUS::NV_ENC_SUCCESS => return Ok(()),
                    NVENCSTATUS::NV_ENC_ERR_ENCODER_BUSY => {
                        thread::sleep(Duration::from_millis(1));
                    }
                    other => {
                        nvenc_check(self.encoder.ptr, other, "NVENC EOS failed")?;
                    }
                }
            }
            Err(video_error("NVENC encoder stayed busy while flushing"))
        }

        /// Locks a ready bitstream synchronously (`doNotWait = 0`), copies its
        /// Annex-B payload and output timestamp, and unlocks it before returning.
        #[allow(unsafe_code)]
        fn lock_bitstream(
            &self,
            bitstream: &mut RawBitstream<'_>,
        ) -> Result<(Vec<u8>, u64), PipelineError> {
            let mut lock_params = NV_ENC_LOCK_BITSTREAM {
                version: NV_ENC_LOCK_BITSTREAM_VER,
                outputBitstream: bitstream.ptr,
                ..Default::default()
            };
            // Synchronous NVENC mode (Linux) requires blocking lock (`doNotWait = 0`).
            nvenc_check(
                self.encoder.ptr,
                unsafe { (ENCODE_API.lock_bitstream)(self.encoder.ptr, &raw mut lock_params) },
                "failed to lock NVENC bitstream",
            )?;
            bitstream.locked.set(true);
            let size = usize::try_from(lock_params.bitstreamSizeInBytes).unwrap_or(0);
            let data = if size == 0 || lock_params.bitstreamBufferPtr.is_null() {
                Vec::new()
            } else {
                // SAFETY: `lock_bitstream` succeeded and guarantees `size` valid readable bytes.
                unsafe {
                    std::slice::from_raw_parts(lock_params.bitstreamBufferPtr.cast::<u8>(), size)
                }
                .to_vec()
            };
            let timestamp = lock_params.outputTimeStamp;
            let unlock = nvenc_check(
                self.encoder.ptr,
                unsafe { (ENCODE_API.unlock_bitstream)(self.encoder.ptr, bitstream.ptr) },
                "failed to unlock NVENC bitstream",
            );
            if unlock.is_ok() {
                bitstream.locked.set(false);
            }
            unlock?;
            Ok((data, timestamp))
        }

        /// Synchronously locks and immediately unlocks a ready bitstream
        /// without copying its payload, used while aborting an in-flight stage
        /// before its buffers are released.
        #[allow(unsafe_code)]
        fn discard_bitstream(&self, bitstream: &mut RawBitstream<'_>) -> Result<(), PipelineError> {
            let mut lock_params = NV_ENC_LOCK_BITSTREAM {
                version: NV_ENC_LOCK_BITSTREAM_VER,
                outputBitstream: bitstream.ptr,
                ..Default::default()
            };
            // Synchronous NVENC mode (Linux) requires blocking lock (`doNotWait = 0`).
            nvenc_check(
                self.encoder.ptr,
                unsafe { (ENCODE_API.lock_bitstream)(self.encoder.ptr, &raw mut lock_params) },
                "failed to lock NVENC bitstream during abort",
            )?;
            bitstream.locked.set(true);
            let unlock = nvenc_check(
                self.encoder.ptr,
                unsafe { (ENCODE_API.unlock_bitstream)(self.encoder.ptr, bitstream.ptr) },
                "failed to unlock NVENC bitstream during abort",
            );
            if unlock.is_ok() {
                bitstream.locked.set(false);
            }
            unlock
        }
    }

    impl Drop for RawNvencSession {
        #[allow(unsafe_code)]
        fn drop(&mut self) {
            if !self.eos_sent.get() {
                let mut eos = NV_ENC_PIC_PARAMS {
                    version: NV_ENC_PIC_PARAMS_VER,
                    encodePicFlags: NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_EOS as u32,
                    ..Default::default()
                };
                // Best-effort flush on early abort; never panic and never mask the caller's error.
                let _status =
                    unsafe { (ENCODE_API.encode_picture)(self.encoder.ptr, &raw mut eos) };
            }
        }
    }

    // Mirrors cudarc's own `memcpy_htod`: the pinned slot's event is awaited and
    // re-recorded via `sync_guard`, so the decode thread cannot overwrite the
    // buffer while the asynchronous copy is still in flight.
    #[allow(unsafe_code)]
    fn dma_pinned_to_device(
        cuda_stream: &CudaStream,
        pinned: &PinnedHostSlice<u8>,
        dev_ptr: sys::CUdeviceptr,
        dev_len: usize,
    ) -> Result<(), PipelineError> {
        let (host, sync_guard) = unsafe { pinned.stream_synced_slice(cuda_stream) };
        if host.len() > dev_len {
            return Err(video_error("NV12 frame exceeds CUDA device slot capacity"));
        }
        let dma = unsafe { result::memcpy_htod_async(dev_ptr, host, cuda_stream.cu_stream()) };
        drop(sync_guard);
        dma.map_err(|err| video_error(format!("failed to upload NV12 frame to GPU: {err}")))
    }

    /// Drains the bitstream slots whose output is *confirmed* signalled in
    /// `ready_bitstreams` using a blocking synchronous lock, releasing each
    /// completed picture's input surface only once its own output timestamp has
    /// been retrieved. Slots that have not been signalled stay in
    /// `pending_bitstreams` so teardown never blocks on deferred NVENC output.
    #[allow(clippy::too_many_arguments)]
    fn drain_ready_bitstreams(
        session: &RawNvencSession,
        codec: NvencVideoCodec,
        bitstreams: &mut [RawBitstream<'_>],
        mapped: &mut [Option<RawMappedResource<'_>>],
        ready_bitstreams: &mut VecDeque<usize>,
        pending_bitstreams: &mut VecDeque<usize>,
        free_bitstreams: &mut VecDeque<usize>,
        free_inputs: &mut VecDeque<usize>,
        frame_to_input_slot: &mut HashMap<usize, usize>,
        track: &mut ContainerTrack,
        encoded_units: &mut Vec<(Vec<u8>, bool, usize)>,
        on_frame_encoded: &mut impl FnMut() -> Result<(), PipelineError>,
    ) -> Result<(), PipelineError> {
        while let Some(bs_slot) = ready_bitstreams.pop_front() {
            let (data, timestamp) = match session.lock_bitstream(&mut bitstreams[bs_slot]) {
                Ok(result) => result,
                Err(err) => {
                    // A failed lock does not prove NVENC is done with this
                    // picture, so it must not stay in the confirmed-ready queue.
                    // Keep it tracked in the deferred queue instead (the earlier
                    // fix pushed it back into `ready_bitstreams`): the slot is
                    // never dropped here, and `abort_pending_nvenc_work` either
                    // blocking-discards it after a confirmed EOS or quarantines
                    // its buffers when no successful submission is available.
                    pending_bitstreams.push_back(bs_slot);
                    return Err(err);
                }
            };
            if data.is_empty() {
                return Err(video_error("NVENC produced an empty access unit"));
            }
            let display_idx = usize::try_from(timestamp).map_err(|_| {
                video_error(format!("NVENC output timestamp {timestamp} exceeds usize"))
            })?;
            let finished_in_slot = frame_to_input_slot.remove(&display_idx).ok_or_else(|| {
                video_error(format!(
                    "NVENC returned unexpected or duplicate output timestamp {display_idx}"
                ))
            })?;
            // With B-frame reordering, the finished picture's input surface is
            // identified by `display_idx`, not by `bs_slot`. Unmap that exact
            // surface before recycling its device slot; the registration stays
            // alive for the whole session.
            mapped[finished_in_slot] = None;
            free_inputs.push_back(finished_in_slot);
            free_bitstreams.push_back(bs_slot);

            let keyframe = codec.is_keyframe(&data);
            if track.config.is_empty()
                && keyframe
                && let Some(config) = codec.extract_config(&data)
            {
                track.config = config;
            }
            encoded_units.push((codec.to_sample(&data), keyframe, display_idx));
            on_frame_encoded()?;
        }
        Ok(())
    }

    /// Completes every NVENC picture whose output is confirmed before the ring
    /// buffers drop on an early stage error, and reports whether deferred work
    /// had to be quarantined instead.
    ///
    /// `ready_bitstreams` holds slots whose output was signalled by a
    /// successfully submitted later picture (the SDK's output-delay contract),
    /// so a synchronous lock on them is safe. `pending_bitstreams` holds
    /// submitted but unsignalled slots — including any picture that returned
    /// `NV_ENC_ERR_NEED_MORE_INPUT` and any slot whose earlier lock failed. If
    /// EOS was never confirmed, no further picture will be submitted, so a
    /// blocking `NvEncLockBitstream` on those slots would wait forever. They
    /// are handed to the caller's bounded quarantine rather than freed while
    /// the driver can still write into them.
    ///
    /// Returns `true` when at least one slot was quarantined, i.e. the caller
    /// must leak the owning buffers instead of dropping them.
    fn abort_pending_nvenc_work(
        session: &RawNvencSession,
        eos_flushed: bool,
        bitstreams: &mut [RawBitstream<'_>],
        ready_bitstreams: &mut VecDeque<usize>,
        pending_bitstreams: &mut VecDeque<usize>,
    ) -> bool {
        if ready_bitstreams.is_empty() && pending_bitstreams.is_empty() {
            return false;
        }
        let mut eos_confirmed = eos_flushed;
        if !eos_confirmed {
            // `flush_eos` sets `eos_sent` before submitting, so a failed attempt
            // leaves a stale flag. Retry regardless: a successful EOS signals
            // all deferred output, which makes the blocking discards safe.
            eos_confirmed = session.flush_eos().is_ok();
        }

        // Best-effort: each lock is independent, so a failure on one slot must
        // not abandon the slots behind it. The caller returns the original
        // stage error rather than masking it.
        while let Some(bs_slot) = ready_bitstreams.pop_front() {
            let discard_result = session.discard_bitstream(&mut bitstreams[bs_slot]);
            drop(discard_result);
        }

        if eos_confirmed {
            while let Some(bs_slot) = pending_bitstreams.pop_front() {
                let discard_result = session.discard_bitstream(&mut bitstreams[bs_slot]);
                drop(discard_result);
            }
            return false;
        }

        // EOS was never confirmed: NVENC requires a successful submission
        // before it signals deferred output, and none is coming. Do not block
        // on these slots and do not destroy their bitstreams; the caller keeps
        // every owning buffer alive in a bounded quarantine instead.
        let quarantined = !pending_bitstreams.is_empty();
        pending_bitstreams.clear();
        quarantined
    }

    /// GPU side of the streaming pipeline. Runs on the calling thread while the
    /// scoped decode thread refills pinned slots. `Ok(None)` means the channel
    /// closed before the first frame; the caller then surfaces the decode error.
    // Every stage input is passed explicitly so the GPU thread owns no hidden state.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    fn run_streaming_nvenc_stage<F>(
        encoder: RawNvencEncoder,
        cuda_ctx: &Arc<CudaContext>,
        codec: NvencVideoCodec,
        rate_control: VideoRateControl,
        fps: u32,
        plan: NvencVideoPlan,
        disable_b_adapt: bool,
        track_index: usize,
        first: FilledPinnedFrame,
        ready_rx: &Receiver<FilledPinnedFrame>,
        free_tx: &SyncSender<PinnedHostSlice<u8>>,
        on_frame_encoded: &mut F,
    ) -> Result<Option<(EncodedVideoStream, u64)>, PipelineError>
    where
        F: FnMut() -> Result<(), PipelineError>,
    {
        let guid = codec.guid();
        let (width, height, pitch) = (first.info.width, first.info.height, first.pitch);
        let pitch_usize =
            usize::try_from(pitch).map_err(|_| video_error("NVENC frame pitch exceeds usize"))?;
        // Align each slot to 256 bytes: one contiguous slab replaces a ring of
        // synchronous cuMemAlloc calls while keeping every registered surface
        // at the alignment NVENC's DMA expects.
        let slot_bytes = nv12_buffer_len(width, height, pitch_usize)?.next_multiple_of(256);

        let mut preset_cfg = encoder.get_preset_config(
            guid,
            NV_ENC_PRESET_P4_GUID,
            NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_HIGH_QUALITY,
        )?;
        configure_video_rc(
            &mut preset_cfg.presetCfg.rcParams,
            rate_control,
            plan.temporal_aq,
            plan.lookahead_depth,
            disable_b_adapt,
        );
        preset_cfg.presetCfg.gopLength = plan.gop_length;
        preset_cfg.presetCfg.frameIntervalP = plan.frame_interval_p();
        codec.configure_codec_config(
            &mut preset_cfg.presetCfg,
            plan.gop_length,
            plan.bframe_ref_mode(),
        );

        let mut init_params = NV_ENC_INITIALIZE_PARAMS {
            version: NV_ENC_INITIALIZE_PARAMS_VER,
            encodeGUID: guid,
            presetGUID: NV_ENC_PRESET_P4_GUID,
            encodeWidth: width,
            encodeHeight: height,
            darWidth: width,
            darHeight: height,
            frameRateNum: fps,
            frameRateDen: 1,
            enablePTD: 1,
            encodeConfig: &raw mut preset_cfg.presetCfg,
            tuningInfo: NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_HIGH_QUALITY,
            ..Default::default()
        };
        let session = encoder.start_session(
            width,
            height,
            NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12,
            &mut init_params,
        )?;

        let slot_count = plan.ring_slots();
        let output_delay = plan.output_delay();
        let total_slab_bytes = slot_bytes
            .checked_mul(slot_count)
            .ok_or_else(|| video_error("NVENC VRAM slab size overflow"))?;
        let cuda_stream = cuda_ctx.default_stream();
        // Declaration order fixes drop order: live mappings and bitstreams drop
        // before the pre-registered resources, which drop before the one VRAM
        // slab, and all drop before `session`. Registration happens once per
        // slot here instead of once per frame, removing NVENC registration churn.
        let vram_slab = DeviceBuffer::zeroed(cuda_ctx, &cuda_stream, total_slab_bytes)?;
        let slot_ptrs: Vec<sys::CUdeviceptr> = (0..slot_count)
            .map(|slot| {
                let offset = u64::try_from(slot.saturating_mul(slot_bytes)).unwrap_or(u64::MAX);
                vram_slab.ptr().saturating_add(offset)
            })
            .collect();
        let mut reg_resources: Vec<RawRegisteredResource<'_>> = Vec::with_capacity(slot_count);
        for &slot_ptr in &slot_ptrs {
            let ptr = usize::try_from(slot_ptr)
                .map_err(|_| video_error("CUDA device pointer exceeds usize"))?
                as *mut c_void;
            reg_resources.push(session.register_cuda_resource(ptr, pitch)?);
        }
        let mut mapped: Vec<Option<RawMappedResource<'_>>> =
            (0..slot_count).map(|_| None).collect();
        let mut bitstreams: Vec<RawBitstream<'_>> = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            bitstreams.push(session.create_bitstream()?);
        }

        let mut track = ContainerTrack {
            codec: codec.output_codec(),
            width: u16::try_from(width).ok(),
            height: u16::try_from(height).ok(),
            channels: None,
            sample_rate: None,
            config: Vec::new(),
            timescale: US_TIMESCALE,
            presentation_edit: None,
        };
        let mut free_inputs: VecDeque<usize> = (0..slot_count).collect();
        let mut free_bitstreams: VecDeque<usize> = (0..slot_count).collect();
        let mut pending_bitstreams: VecDeque<usize> = VecDeque::with_capacity(slot_count);
        let mut ready_bitstreams: VecDeque<usize> = VecDeque::with_capacity(slot_count);
        let mut frame_to_input_slot: HashMap<usize, usize> = HashMap::with_capacity(slot_count);
        let mut frame_timings: Vec<(i64, u64)> = Vec::new();
        let mut encoded_units: Vec<(Vec<u8>, bool, usize)> = Vec::new();
        let mut frames_encoded = 0_u64;
        let mut next_frame = Some(first);
        let mut eos_flushed = false;

        let stage_result: Result<(), PipelineError> = (|| {
            while let Some(filled) = next_frame.take().or_else(|| ready_rx.recv().ok()) {
                let in_slot = free_inputs.pop_front().ok_or_else(|| {
                    video_error("NVENC input ring exhausted while awaiting encoder output")
                })?;
                let bs_slot = free_bitstreams.pop_front().ok_or_else(|| {
                    video_error("NVENC bitstream ring exhausted while awaiting encoder output")
                })?;

                dma_pinned_to_device(&cuda_stream, &filled.pinned, slot_ptrs[in_slot], slot_bytes)?;
                if free_tx.send(filled.pinned).is_err() {
                    // The decode thread already exited; its error is surfaced after
                    // the join, and dropping the send error recycles this slot.
                }
                // Only the lightweight map is per frame; the CUDA pointer was
                // registered once at session init with the rest of the ring.
                mapped[in_slot] = Some(session.map_cuda_resource(&reg_resources[in_slot], pitch)?);
                let resource = mapped[in_slot]
                    .as_ref()
                    .ok_or_else(|| video_error("NVENC input slot was not mapped"))?;

                let frame_idx = frame_timings.len();
                let frame_idx_u64 =
                    u64::try_from(frame_idx).map_err(|_| video_error("frame index exceeds u64"))?;
                frame_timings.push((filled.info.pts_us, filled.info.duration_us.max(1)));
                frame_to_input_slot.insert(frame_idx, in_slot);

                let outcome =
                    session.encode_picture(resource, &mut bitstreams[bs_slot], frame_idx_u64)?;
                pending_bitstreams.push_back(bs_slot);
                frames_encoded = frames_encoded.saturating_add(1);

                if outcome == SubmitOutcome::Ready {
                    while pending_bitstreams.len() > output_delay {
                        if let Some(eligible) = pending_bitstreams.pop_front() {
                            ready_bitstreams.push_back(eligible);
                        }
                    }
                    drain_ready_bitstreams(
                        &session,
                        codec,
                        &mut bitstreams,
                        &mut mapped,
                        &mut ready_bitstreams,
                        &mut pending_bitstreams,
                        &mut free_bitstreams,
                        &mut free_inputs,
                        &mut frame_to_input_slot,
                        &mut track,
                        &mut encoded_units,
                        on_frame_encoded,
                    )?;
                }
            }

            session.flush_eos()?;
            eos_flushed = true;
            ready_bitstreams.extend(pending_bitstreams.drain(..));
            drain_ready_bitstreams(
                &session,
                codec,
                &mut bitstreams,
                &mut mapped,
                &mut ready_bitstreams,
                &mut pending_bitstreams,
                &mut free_bitstreams,
                &mut free_inputs,
                &mut frame_to_input_slot,
                &mut track,
                &mut encoded_units,
                on_frame_encoded,
            )?;
            Ok(())
        })();

        let mut quarantined = false;
        if stage_result.is_err() {
            quarantined = abort_pending_nvenc_work(
                &session,
                eos_flushed,
                &mut bitstreams,
                &mut ready_bitstreams,
                &mut pending_bitstreams,
            );
        }
        if quarantined {
            // WHY: EOS was never confirmed, so NVENC may still write into the
            // deferred pictures' bitstreams and mapped input surfaces. Keep this
            // bounded set — one ring's buffers, registrations, the VRAM slab and
            // the encoder handle — alive until process teardown instead of
            // destroying memory the driver may still access.
            std::mem::forget(bitstreams);
            std::mem::forget(mapped);
            std::mem::forget(reg_resources);
            std::mem::forget(vram_slab);
            std::mem::forget(session);
        }

        // Reverse declaration order drops bitstreams, live mappings,
        // registrations and device buffers before the NVENC session on every
        // normal exit path, including early `?` errors. Explicit drops are
        // avoided because `mapped` borrows `reg_resources`, which borrows
        // `session`; the quarantine above is the only exception.
        stage_result?;

        if encoded_units.len() != frame_timings.len() {
            return Err(video_error(format!(
                "NVENC returned {} packets for {} input frames",
                encoded_units.len(),
                frame_timings.len()
            )));
        }
        let display_indices: Vec<usize> = encoded_units.iter().map(|(_, _, idx)| *idx).collect();
        let timestamps = assign_reordered_timestamps(&frame_timings, &display_indices)?;
        let mut packets = Vec::with_capacity(encoded_units.len());
        for ((payload, keyframe, _), (pts, dts, duration)) in
            encoded_units.into_iter().zip(timestamps)
        {
            packets.push(ContainerPacket {
                track: track_index,
                pts,
                dts,
                duration,
                keyframe,
                data: payload,
            });
        }
        if packets.is_empty() {
            return Err(video_error(
                "NVENC produced no video packets (no silent CPU fallback)",
            ));
        }
        Ok(Some((
            EncodedVideoStream { track, packets },
            frames_encoded,
        )))
    }

    /// GPU-resident NVDEC -> NVENC path: luma and chroma of each decoded surface
    /// are copied device-to-device into a separate, pre-registered NVENC slot.
    /// This avoids CPU readback and pinned host staging but does not directly
    /// share the original NVDEC surface. Any NVDEC setup or caps failure is a
    /// hard error (no CPU fallback).
    #[allow(clippy::too_many_lines)]
    pub(super) fn encode_video_nvdec_nvenc_streaming_enabled<F>(
        source_track: &ContainerTrack,
        track_packets: &[ContainerPacket],
        source_codec: VideoCodec,
        mode: VideoEncodeMode,
        track_index: usize,
        mut on_frame_encoded: F,
    ) -> Result<(EncodedVideoStream, u64, String), PipelineError>
    where
        F: FnMut() -> Result<(), PipelineError>,
    {
        if !source_codec.nvdec_decodable() {
            return Err(video_error(format!(
                "--hwdecode nvdec cannot decode {source_codec}; no silent CPU fallback"
            )));
        }
        let (codec, rate_control) = NvencVideoCodec::from_mode(mode)?;
        let guid = codec.guid();
        let width = u32::from(
            source_track
                .width
                .ok_or_else(|| video_error("NVDEC -> NVENC requires a known source width"))?,
        );
        let height = u32::from(
            source_track
                .height
                .ok_or_else(|| video_error("NVDEC -> NVENC requires a known source height"))?,
        );
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(video_error(format!(
                "NVENC requires even, non-zero dimensions; got {width}x{height}"
            )));
        }
        let cuda_ctx = CudaContext::new(0)
            .map_err(|err| video_error(format!("failed to initialise CUDA device 0: {err}")))?;
        // NVCUVID capability queries require a current CUDA context.
        crate::gpu::nvdec::validate_decode_caps(source_codec, 8, width, height)?;
        let device_name = cuda_ctx
            .name()
            .map_err(|err| video_error(format!("failed to query CUDA device name: {err}")))?;
        let encoder = RawNvencEncoder::open(Arc::clone(&cuda_ctx))?;
        if !encoder.get_encode_guids()?.contains(&guid) {
            return Err(video_error(
                "this NVIDIA GPU does not expose the requested NVENC video codec",
            ));
        }
        if !encoder
            .get_supported_input_formats(guid)?
            .contains(&NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12)
        {
            return Err(video_error(
                "NVENC on this GPU does not support NV12 input buffers",
            ));
        }
        let caps = encoder.query_codec_caps(guid)?;
        let fps = estimate_video_fps_from_packets(track_packets);
        let plan = NvencVideoPlan::from_caps(fps, caps);
        let mut preset_cfg = encoder.get_preset_config(
            guid,
            NV_ENC_PRESET_P4_GUID,
            NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_HIGH_QUALITY,
        )?;
        configure_video_rc(
            &mut preset_cfg.presetCfg.rcParams,
            rate_control,
            plan.temporal_aq,
            plan.lookahead_depth,
            false,
        );
        preset_cfg.presetCfg.gopLength = plan.gop_length;
        preset_cfg.presetCfg.frameIntervalP = plan.frame_interval_p();
        codec.configure_codec_config(
            &mut preset_cfg.presetCfg,
            plan.gop_length,
            plan.bframe_ref_mode(),
        );
        let mut init_params = NV_ENC_INITIALIZE_PARAMS {
            version: NV_ENC_INITIALIZE_PARAMS_VER,
            encodeGUID: guid,
            presetGUID: NV_ENC_PRESET_P4_GUID,
            encodeWidth: width,
            encodeHeight: height,
            darWidth: width,
            darHeight: height,
            frameRateNum: fps,
            frameRateDen: 1,
            enablePTD: 1,
            encodeConfig: &raw mut preset_cfg.presetCfg,
            tuningInfo: NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_HIGH_QUALITY,
            ..Default::default()
        };
        let session = encoder.start_session(
            width,
            height,
            NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12,
            &mut init_params,
        )?;

        // 256-byte pitch alignment gives coalesced VRAM-to-VRAM copies and NVENC DMA.
        let pitch_usize = usize::try_from(width)
            .unwrap_or(usize::MAX)
            .next_multiple_of(256);
        let pitch =
            u32::try_from(pitch_usize).map_err(|_| video_error("NVENC frame pitch exceeds u32"))?;
        let slot_bytes = nv12_buffer_len(width, height, pitch_usize)?.next_multiple_of(256);
        let output_delay = plan.output_delay();
        let slot_count = plan.ring_slots().max(output_delay + 4);
        let total_slab_bytes = slot_bytes
            .checked_mul(slot_count)
            .ok_or_else(|| video_error("NVENC VRAM slab size overflow"))?;
        let cuda_stream = cuda_ctx.default_stream();
        // Reverse declaration order gives the required teardown order. One
        // contiguous VRAM slab avoids 29 synchronous cuMemAlloc/cuMemFree stalls.
        let vram_slab = DeviceBuffer::zeroed(&cuda_ctx, &cuda_stream, total_slab_bytes)?;
        let slot_ptrs: Vec<sys::CUdeviceptr> = (0..slot_count)
            .map(|slot| {
                let offset = u64::try_from(slot.saturating_mul(slot_bytes)).unwrap_or(u64::MAX);
                vram_slab.ptr().saturating_add(offset)
            })
            .collect();
        let mut reg_resources: Vec<RawRegisteredResource<'_>> = Vec::with_capacity(slot_count);
        for &slot_ptr in &slot_ptrs {
            let ptr = usize::try_from(slot_ptr)
                .map_err(|_| video_error("CUDA device pointer exceeds usize"))?
                as *mut c_void;
            reg_resources.push(session.register_cuda_resource(ptr, pitch)?);
        }
        let mut mapped: Vec<Option<RawMappedResource<'_>>> =
            (0..slot_count).map(|_| None).collect();
        let mut bitstreams: Vec<RawBitstream<'_>> = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            bitstreams.push(session.create_bitstream()?);
        }

        let mut track = ContainerTrack {
            codec: codec.output_codec(),
            width: u16::try_from(width).ok(),
            height: u16::try_from(height).ok(),
            channels: None,
            sample_rate: None,
            config: Vec::new(),
            timescale: US_TIMESCALE,
            presentation_edit: source_track.presentation_edit,
        };
        let mut free_inputs: VecDeque<usize> = (0..slot_count).collect();
        let mut free_bitstreams: VecDeque<usize> = (0..slot_count).collect();
        let mut pending_bitstreams: VecDeque<usize> = VecDeque::with_capacity(slot_count);
        let mut ready_bitstreams: VecDeque<usize> = VecDeque::with_capacity(slot_count);
        let mut frame_to_input_slot: HashMap<usize, usize> = HashMap::with_capacity(slot_count);
        let mut frame_timings: Vec<(i64, u64)> = Vec::new();
        let mut encoded_units: Vec<(Vec<u8>, bool, usize)> = Vec::new();
        let mut frames_encoded = 0_u64;
        let mut eos_flushed = false;

        let decode_result = crate::gpu::nvdec::decode_packets_to_gpu(
            &cuda_ctx,
            &cuda_stream,
            source_codec,
            &source_track.config,
            track_packets,
            |frame, pts_us, duration_us| {
                if free_inputs.is_empty() {
                    // Recycle whatever the encoder has already produced before
                    // refusing a new surface; never overwrite a mapped slot.
                    while pending_bitstreams.len() > output_delay {
                        if let Some(bs) = pending_bitstreams.pop_front() {
                            ready_bitstreams.push_back(bs);
                        }
                    }
                    drain_ready_bitstreams(
                        &session,
                        codec,
                        &mut bitstreams,
                        &mut mapped,
                        &mut ready_bitstreams,
                        &mut pending_bitstreams,
                        &mut free_bitstreams,
                        &mut free_inputs,
                        &mut frame_to_input_slot,
                        &mut track,
                        &mut encoded_units,
                        &mut on_frame_encoded,
                    )?;
                }
                let in_slot = free_inputs.pop_front().ok_or_else(|| {
                    video_error("NVDEC -> NVENC input ring exhausted while awaiting encoder output")
                })?;
                let bs_slot = free_bitstreams.pop_front().ok_or_else(|| {
                    video_error("NVENC bitstream ring exhausted while awaiting encoder output")
                })?;
                frame.copy_to_nvenc_slot(
                    &cuda_stream,
                    slot_ptrs[in_slot],
                    pitch_usize,
                    width,
                    height,
                )?;
                mapped[in_slot] = Some(session.map_cuda_resource(&reg_resources[in_slot], pitch)?);
                let resource = mapped[in_slot]
                    .as_ref()
                    .ok_or_else(|| video_error("NVENC input slot was not mapped"))?;
                let frame_idx = frame_timings.len();
                let frame_idx_u64 =
                    u64::try_from(frame_idx).map_err(|_| video_error("frame index exceeds u64"))?;
                frame_timings.push((pts_us, duration_us.max(1)));
                frame_to_input_slot.insert(frame_idx, in_slot);
                let outcome =
                    session.encode_picture(resource, &mut bitstreams[bs_slot], frame_idx_u64)?;
                pending_bitstreams.push_back(bs_slot);
                frames_encoded = frames_encoded.saturating_add(1);
                if outcome == SubmitOutcome::Ready {
                    while pending_bitstreams.len() > output_delay {
                        if let Some(bs) = pending_bitstreams.pop_front() {
                            ready_bitstreams.push_back(bs);
                        }
                    }
                    drain_ready_bitstreams(
                        &session,
                        codec,
                        &mut bitstreams,
                        &mut mapped,
                        &mut ready_bitstreams,
                        &mut pending_bitstreams,
                        &mut free_bitstreams,
                        &mut free_inputs,
                        &mut frame_to_input_slot,
                        &mut track,
                        &mut encoded_units,
                        &mut on_frame_encoded,
                    )?;
                }
                Ok(())
            },
        );

        let stage_result = decode_result.and_then(|_| {
            session.flush_eos()?;
            eos_flushed = true;
            ready_bitstreams.extend(pending_bitstreams.drain(..));
            drain_ready_bitstreams(
                &session,
                codec,
                &mut bitstreams,
                &mut mapped,
                &mut ready_bitstreams,
                &mut pending_bitstreams,
                &mut free_bitstreams,
                &mut free_inputs,
                &mut frame_to_input_slot,
                &mut track,
                &mut encoded_units,
                &mut on_frame_encoded,
            )
        });
        let mut quarantined = false;
        if stage_result.is_err() {
            quarantined = abort_pending_nvenc_work(
                &session,
                eos_flushed,
                &mut bitstreams,
                &mut ready_bitstreams,
                &mut pending_bitstreams,
            );
        }
        if quarantined {
            // WHY: EOS was never confirmed, so NVENC may still write into the
            // deferred pictures' bitstreams and mapped input surfaces. Keep this
            // bounded set — one ring's buffers, registrations, the VRAM slab and
            // the encoder handle — alive until process teardown instead of
            // destroying memory the driver may still access.
            std::mem::forget(bitstreams);
            std::mem::forget(mapped);
            std::mem::forget(reg_resources);
            std::mem::forget(vram_slab);
            std::mem::forget(session);
        }
        stage_result?;

        if encoded_units.len() != frame_timings.len() {
            return Err(video_error(format!(
                "NVENC returned {} packets for {} NVDEC frames",
                encoded_units.len(),
                frame_timings.len()
            )));
        }
        let display_indices: Vec<usize> = encoded_units.iter().map(|(_, _, idx)| *idx).collect();
        let timestamps = super::assign_reordered_timestamps(&frame_timings, &display_indices)?;
        let mut packets = Vec::with_capacity(encoded_units.len());
        for ((payload, keyframe, _), (pts, dts, duration)) in
            encoded_units.into_iter().zip(timestamps)
        {
            packets.push(ContainerPacket {
                track: track_index,
                pts,
                dts,
                duration,
                keyframe,
                data: payload,
            });
        }
        if packets.is_empty() {
            return Err(video_error("NVDEC -> NVENC produced no video packets"));
        }
        Ok((
            EncodedVideoStream { track, packets },
            frames_encoded,
            device_name.clone(),
        ))
    }

    #[cfg(test)]
    pub(super) fn codec_supports_b_frames_enabled(
        codec: VideoCodec,
    ) -> Result<bool, PipelineError> {
        let nvenc_codec = match codec {
            VideoCodec::H264 => NvencVideoCodec::H264,
            VideoCodec::Hevc => NvencVideoCodec::Hevc,
            other => {
                return Err(video_error(format!(
                    "NVENC does not support video codec {other}"
                )));
            }
        };
        let cuda_ctx = CudaContext::new(0)
            .map_err(|err| video_error(format!("failed to initialize CUDA device 0: {err}")))?;
        let encoder = RawNvencEncoder::open(cuda_ctx)?;
        let caps = encoder.query_codec_caps(nvenc_codec.guid())?;
        Ok(caps.max_b_frames > 0)
    }

    /// Decodes source packets straight into pinned host memory while NVENC
    /// uploads and encodes from a separate CUDA device ring.
    #[allow(clippy::too_many_lines)]
    pub(super) fn encode_video_nvenc_streaming_enabled<F>(
        path: &Path,
        source_track: &ContainerTrack,
        track_packets: &[ContainerPacket],
        mode: VideoEncodeMode,
        track_index: usize,
        disable_b_adapt: bool,
        mut on_frame_encoded: F,
    ) -> Result<(EncodedVideoStream, u64, String), PipelineError>
    where
        F: FnMut() -> Result<(), PipelineError>,
    {
        let (codec, rate_control) = NvencVideoCodec::from_mode(mode)?;
        let guid = codec.guid();

        // Only the CUDA context is needed before the decode thread starts.
        // NVENC session setup runs on the calling thread in parallel with the
        // CPU/GIF/WebP decode so both halves overlap instead of serialising.
        let cuda_ctx = CudaContext::new(0)
            .map_err(|err| video_error(format!("failed to initialize CUDA device 0: {err}")))?;
        let device_name = cuda_ctx
            .name()
            .map_err(|err| video_error(format!("failed to query CUDA device name: {err}")))?;
        let cancelled = AtomicBool::new(false);

        thread::scope(|scope| {
            let decode_ctx = Arc::clone(&cuda_ctx);
            let (ready_tx, ready_rx) = mpsc::sync_channel::<FilledPinnedFrame>(VIDEO_RING_SLOTS);
            let (free_tx, free_rx) = mpsc::sync_channel::<PinnedHostSlice<u8>>(VIDEO_RING_SLOTS);
            let seed_free_tx = free_tx.clone();
            let decode_cancelled = &cancelled;

            let decode_handle = thread::Builder::new()
                .name("fastconv-video-decode".to_owned())
                .spawn_scoped(scope, move || -> Result<(), PipelineError> {
                    decode_ctx.bind_to_thread().map_err(|err| {
                        video_error(format!(
                            "failed to bind CUDA context on decode thread: {err}"
                        ))
                    })?;
                    let mut ring_geometry: Option<(u32, u32, usize, u32)> = None;
                    let mut first_pinned: Option<PinnedHostSlice<u8>> = None;
                    let mut seed_free_tx = Some(seed_free_tx);
                    let mut stopped_by_consumer = false;

                    let decode_result = decode_video_track_streaming(
                        path,
                        source_track,
                        track_packets,
                        |info, pack_nv12| {
                            if decode_cancelled.load(Ordering::Acquire) {
                                stopped_by_consumer = true;
                                return Err(video_error(
                                    "NVENC consumer stopped the decode pipeline",
                                ));
                            }
                            let (expected_w, expected_h, pitch_usize, pitch_u32) =
                                if let Some(geometry) = ring_geometry {
                                    geometry
                                } else {
                                    if info.width == 0
                                        || info.height == 0
                                        || !info.width.is_multiple_of(2)
                                        || !info.height.is_multiple_of(2)
                                    {
                                        return Err(video_error(format!(
                                            "NVENC requires even, non-zero dimensions; got {}x{}",
                                            info.width, info.height
                                        )));
                                    }
                                    // CUDADEVICEPTR slots come from `cuMemAlloc`, so NVENC
                                    // needs the allocation's row width in bytes, which must
                                    // be a multiple of 4. The allocation is pitched at
                                    // `width.next_multiple_of(4)`, so that is its row width:
                                    // padding only the stride keeps 2-mod-4 picture widths
                                    // (854, 1366, ...) encodable. Packing, upload and
                                    // registration all use this one stride.
                                    let pitch_usize = usize::try_from(info.width)
                                        .map_err(|_| {
                                            video_error("NVENC frame width exceeds usize")
                                        })?
                                        .next_multiple_of(4);
                                    let pitch_u32 = u32::try_from(pitch_usize).map_err(|_| {
                                        video_error("NVENC frame pitch exceeds u32")
                                    })?;
                                    let slot_bytes =
                                        nv12_buffer_len(info.width, info.height, pitch_usize)?;
                                    let mut buffers = Vec::with_capacity(VIDEO_RING_SLOTS);
                                    for _ in 0..VIDEO_RING_SLOTS {
                                        buffers.push(alloc_zeroed_pinned(&decode_ctx, slot_bytes)?);
                                    }
                                    let mut buffers = buffers.into_iter();
                                    let pinned = buffers.next().ok_or_else(|| {
                                        video_error("internal pinned ring buffer underflow")
                                    })?;
                                    // Seed the free ring, then drop this sender so a GPU
                                    // failure closes `free_rx` and unblocks our `recv`.
                                    let seed = seed_free_tx.take().ok_or_else(|| {
                                        video_error("internal pinned ring already seeded")
                                    })?;
                                    for buffer in buffers {
                                        seed.send(buffer).map_err(|_| {
                                            video_error("NVENC consumer stopped during ring setup")
                                        })?;
                                    }
                                    drop(seed);
                                    first_pinned = Some(pinned);
                                    ring_geometry =
                                        Some((info.width, info.height, pitch_usize, pitch_u32));
                                    (info.width, info.height, pitch_usize, pitch_u32)
                                };
                            if info.width != expected_w || info.height != expected_h {
                                return Err(video_error(
                                    "video frame dimensions change mid-stream",
                                ));
                            }
                            let mut pinned = match first_pinned.take() {
                                Some(buffer) => buffer,
                                None => {
                                    if let Ok(buffer) = free_rx.recv() {
                                        buffer
                                    } else {
                                        stopped_by_consumer = true;
                                        return Err(video_error(
                                            "NVENC consumer stopped the decode pipeline",
                                        ));
                                    }
                                }
                            };
                            let dst = pinned.as_mut_slice().map_err(|err| {
                                video_error(format!("failed to sync CUDA pinned buffer: {err}"))
                            })?;
                            pack_nv12(dst, pitch_usize)?;
                            if ready_tx
                                .send(FilledPinnedFrame {
                                    pinned,
                                    info,
                                    pitch: pitch_u32,
                                })
                                .is_err()
                            {
                                stopped_by_consumer = true;
                                return Err(video_error(
                                    "NVENC consumer stopped the decode pipeline",
                                ));
                            }
                            Ok(())
                        },
                    );

                    if stopped_by_consumer {
                        Ok(())
                    } else {
                        decode_result
                    }
                })
                .map_err(PipelineError::WorkerStartFailed)?;

            // NVENC encoder open/capability queries overlap with the decode
            // thread refilling the bounded pinned ring. Any failure here still
            // reaches the shared cancellation path below, which drops both
            // channel ends before joining so the decoder never blocks forever.
            let init = (|| -> Result<(RawNvencEncoder, u32, NvencVideoPlan), PipelineError> {
                let encoder = RawNvencEncoder::open(Arc::clone(&cuda_ctx))?;
                let guids = encoder.get_encode_guids()?;
                if !guids.contains(&guid) {
                    return Err(video_error(
                        "this NVIDIA GPU does not expose the requested NVENC video codec",
                    ));
                }
                let formats = encoder.get_supported_input_formats(guid)?;
                if !formats.contains(&NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12) {
                    return Err(video_error(
                        "NVENC on this GPU does not support NV12 input buffers",
                    ));
                }
                let caps = encoder.query_codec_caps(guid)?;
                let fps = estimate_video_fps_from_packets(track_packets);
                let plan = NvencVideoPlan::from_caps(fps, caps);
                Ok((encoder, fps, plan))
            })();

            // The first decoded frame determines the session and ring geometry;
            // subsequent frames decode while NVENC initialises and encodes.
            let gpu_result = match init {
                Ok((encoder, fps, plan)) => {
                    let first = ready_rx.recv().ok();
                    match first {
                        Some(first) => run_streaming_nvenc_stage(
                            encoder,
                            &cuda_ctx,
                            codec,
                            rate_control,
                            fps,
                            plan,
                            disable_b_adapt,
                            track_index,
                            first,
                            &ready_rx,
                            &free_tx,
                            &mut on_frame_encoded,
                        ),
                        None => Ok(None),
                    }
                }
                Err(err) => Err(err),
            };

            // On every exit path (including early `?` inside the stage) cancel the
            // decoder and drop both channel ends before joining, so it cannot block
            // in `recv`/`send` while we wait for it.
            cancelled.store(true, Ordering::Release);
            drop(ready_rx);
            drop(free_tx);
            let decode_result = decode_handle
                .join()
                .unwrap_or(Err(PipelineError::WorkerPanicked));

            match gpu_result {
                Ok(Some((stream, frames_encoded))) => {
                    decode_result?;
                    Ok((stream, frames_encoded, device_name.clone()))
                }
                Ok(None) => match decode_result {
                    Err(decode_err) => Err(decode_err),
                    Ok(()) => Err(video_error("no video frames to encode")),
                },
                Err(gpu_err) => Err(gpu_err),
            }
        })
    }

    #[cfg(test)]
    mod heic_decode_tests {
        use super::decode_to_rgba_frame;
        use crate::PipelineError;
        use fastconv_domain::{ImageBatchItem, ImageDimensions, ImageInputFormat};
        use std::env;
        use std::fs;
        use std::path::Path;
        use std::process;

        fn write_heic(
            path: &Path,
            width: u32,
            height: u32,
            pixels: &[u8],
        ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
            let bytes = heifer::encode_rgba8(
                width,
                height,
                pixels,
                &heifer::EncodeOptions::new().with_max_threads(1),
            )?;
            fs::write(path, &bytes)?;
            Ok(bytes)
        }

        #[test]
        fn heic_feeds_opaque_rgba_directly_and_rejects_transparency()
        -> Result<(), Box<dyn std::error::Error>> {
            let dir = env::temp_dir().join(format!("fastconv-nvenc-heic-{}-decode", process::id()));
            fs::create_dir_all(&dir)?;
            let (width, height) = (32_u32, 24_u32);
            let mut rgba = Vec::with_capacity((width * height * 4) as usize);
            for index in 0..width * height {
                let value = u8::try_from(index % 251).unwrap_or(0);
                rgba.extend_from_slice(&[value, value.wrapping_mul(3), value.wrapping_mul(7), 255]);
            }
            let opaque = dir.join("opaque.heic");
            write_heic(&opaque, width, height, &rgba)?;
            let item = ImageBatchItem::new(
                opaque,
                dir.join("out.heic"),
                ImageInputFormat::Heic,
                ImageDimensions::new(width, height)?,
            )?;
            let mut file_bytes = Vec::new();
            let mut pixel_buf = Vec::new();
            let frame =
                decode_to_rgba_frame(0, &item, &mut file_bytes, &mut pixel_buf, Vec::new())?;
            assert_eq!((frame.width, frame.height), (width, height));
            assert_eq!(frame.rgba_bytes.len(), (width * height * 4) as usize);

            rgba[3] = 96;
            let transparent = dir.join("transparent.heic");
            write_heic(&transparent, width, height, &rgba)?;
            let item = ImageBatchItem::new(
                transparent,
                dir.join("out2.heic"),
                ImageInputFormat::Heic,
                ImageDimensions::new(width, height)?,
            )?;
            assert!(matches!(
                decode_to_rgba_frame(0, &item, &mut file_bytes, &mut pixel_buf, Vec::new()),
                Err(PipelineError::UnsupportedPixelFormat { .. })
            ));

            fs::remove_dir_all(&dir)?;
            Ok(())
        }
    }
}

#[cfg(not(feature = "nvenc"))]
use fastconv_domain::GpuBackend;
#[cfg(any(feature = "nvenc", test))]
use fastconv_domain::VideoQuality;
use fastconv_domain::{ImageBatchPlan, ImageConversionReport, ImageQuality, VideoEncodeMode};
use std::path::Path;

use crate::PipelineError;
use crate::video_codecs::EncodedVideoStream;
use crate::video_container::{ContainerPacket, ContainerTrack};

/// Target B-frame count between anchors on capable GPUs.
#[cfg(any(feature = "nvenc", test))]
const NVENC_B_FRAMES: i32 = 3;
#[cfg(feature = "nvenc")]
const NVENC_AQ_STRENGTH: u32 = 8;
#[cfg(any(feature = "nvenc", test))]
const NVENC_LOOKAHEAD_DEPTH: u16 = 16;

/// Hardware capabilities queried from `nvEncGetEncodeCaps` for one codec GUID.
#[cfg(any(feature = "nvenc", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NvencCodecCaps {
    max_b_frames: i32,
    bframe_ref_mode: i32,
    lookahead_supported: bool,
    temporal_aq_supported: bool,
}

/// Validated NVENC encode configuration derived from hardware capabilities so
/// older GPUs (e.g. Pascal HEVC without B-frames) stay on NVENC with
/// `frameIntervalP = 1` instead of failing initialization or falling back to CPU.
#[cfg(any(feature = "nvenc", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NvencVideoPlan {
    gop_length: u32,
    b_frames: u32,
    use_middle_b_ref: bool,
    lookahead_depth: u16,
    temporal_aq: bool,
}

#[cfg(any(feature = "nvenc", test))]
impl NvencVideoPlan {
    fn from_caps(fps: u32, caps: NvencCodecCaps) -> Self {
        let gop_length = fps.saturating_mul(2).clamp(30, 300);
        let b_frames = u32::try_from(caps.max_b_frames.clamp(0, NVENC_B_FRAMES)).unwrap_or(0);
        let use_middle_b_ref = b_frames >= 2 && caps.bframe_ref_mode > 0;
        let lookahead_depth = if caps.lookahead_supported {
            NVENC_LOOKAHEAD_DEPTH
        } else {
            0
        };
        Self {
            gop_length,
            b_frames,
            use_middle_b_ref,
            lookahead_depth,
            temporal_aq: caps.temporal_aq_supported,
        }
    }

    fn frame_interval_p(self) -> i32 {
        i32::try_from(self.b_frames).unwrap_or(0).saturating_add(1)
    }

    #[cfg(feature = "nvenc")]
    fn bframe_ref_mode(self) -> nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_BFRAME_REF_MODE {
        use nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_BFRAME_REF_MODE;
        if self.use_middle_b_ref {
            NV_ENC_BFRAME_REF_MODE::NV_ENC_BFRAME_REF_MODE_MIDDLE
        } else {
            NV_ENC_BFRAME_REF_MODE::NV_ENC_BFRAME_REF_MODE_DISABLED
        }
    }

    /// Number of submitted pictures kept buffered before blocking-locking the
    /// oldest bitstream on `NV_ENC_SUCCESS`. Matches NVIDIA's `NvEncoder`
    /// output-delay contract so synchronous `lock_bitstream(doNotWait = 0)`
    /// never touches an unready lookahead slot.
    fn output_delay(self) -> usize {
        if self.lookahead_depth == 0 {
            0
        } else {
            usize::from(self.lookahead_depth)
                .saturating_add(usize::try_from(self.b_frames).unwrap_or(0))
                .saturating_add(3)
        }
    }

    /// Device/bitstream ring capacity: at least 3 for overlap when IPPP, and
    /// strictly larger than `output_delay + b_frames` when lookahead/B-frames
    /// are active so `NeedMoreInput` bursts never exhaust free slots.
    fn ring_slots(self) -> usize {
        let b_frames = usize::try_from(self.b_frames).unwrap_or(0);
        self.output_delay()
            .saturating_add(b_frames)
            .saturating_add(4)
            .max(3)
    }
}

/// Maps `VideoQuality` (`0..=100`) to NVENC VBR/CQ `targetQuality` (1/256 units).
///
/// The line is anchored at `100 -> CQ 10`, `80 -> CQ 23` and `0 -> CQ 51`, so
/// higher quality always means a lower (finer) quantization parameter. All
/// arithmetic is integer fixed-point; NVENC treats `targetQuality == 0` as
/// automatic, hence the floor of 1.
#[cfg(any(feature = "nvenc", test))]
fn quality_to_nvenc_cq(quality: VideoQuality) -> (u8, u8) {
    let q = quality.get();
    let cq_256 = if q >= 80 {
        10 * 256 + ((i32::from(100 - q) * (13 * 256) + 10) / 20)
    } else {
        23 * 256 + ((i32::from(80 - q) * (28 * 256) + 40) / 80)
    };
    let cq_256 = cq_256.clamp(256, 51 * 256);
    let mut target = u8::try_from(cq_256 >> 8).unwrap_or(51);
    let mut lsb = u8::try_from(cq_256 & 0xFF).unwrap_or(u8::MAX);
    if target >= 51 {
        target = 51;
        lsb = 0;
    }
    (target, lsb)
}

/// Assigns presentation and decode timestamps to packets emitted in decode order.
///
/// `display_indices[k]` is the input-frame index carried by output packet `k`
/// (NVENC's `input_timestamp`). It must be a permutation of `0..n`. Packet order
/// is decode order, so both PTS and DTS are produced **in packet order**: pairing
/// a packet's PTS with an unrelated frame's DTS (the previous implementation)
/// produces inverted or equal DTS for B-frame/animation sources and is the root
/// cause of `non monotonically increasing dts` warnings on GIF→MP4 NVENC output.
///
/// The DTS sequence advances by the *current packet's own* duration-derived
/// cadence for reordered packets (and by 1 µs only to break a PTS tie), then is
/// uniformly shifted back by the largest deficit `max(g - pts)`. Taking the
/// cadence from the current packet rather than a global median matters when a
/// later segment is faster: a median cadence would keep applying the old, wider
/// frame spacing on every reorder, growing an unbounded decode lead. This still
/// guarantees `dts[k] <= pts[k]` and strictly increasing DTS with one global
/// reorder delay, and gaps wide enough to stay distinct after conversion to the
/// 90 kHz timebase of MP4/MPEG-TS: there a 1 µs step rounds a reordered P/B group
/// onto a single tick, making a decoder sort B-frames ahead of their reference.
#[cfg(any(feature = "nvenc", test))]
pub(crate) fn assign_reordered_timestamps(
    frame_timings: &[(i64, u64)],
    display_indices: &[usize],
) -> Result<Vec<(i64, i64, u64)>, PipelineError> {
    let encode_error = |reason: String| PipelineError::EncodeFailed {
        path: Path::new("nvenc").to_path_buf(),
        reason,
    };
    if display_indices.len() != frame_timings.len() {
        return Err(encode_error(format!(
            "NVENC returned {} frames for {} inputs",
            display_indices.len(),
            frame_timings.len()
        )));
    }
    let mut seen = vec![false; frame_timings.len()];
    let mut pts = Vec::with_capacity(display_indices.len());
    let mut durations = Vec::with_capacity(display_indices.len());
    for (k, &index) in display_indices.iter().enumerate() {
        let (frame_pts, duration) = frame_timings.get(index).ok_or_else(|| {
            encode_error(format!(
                "NVENC output timestamp {index} at packet {k} exceeds the {} input frames",
                frame_timings.len()
            ))
        })?;
        if seen[index] {
            return Err(encode_error(format!(
                "NVENC returned duplicate output timestamp {index}"
            )));
        }
        seen[index] = true;
        pts.push(i128::from(*frame_pts));
        durations.push(*duration);
    }

    // Per-packet decode-order cadence: a reordered packet (PTS below the
    // previous packet's) advances by its own frame duration instead of 1 µs so
    // its DTS stays distinct once the container rescales to 90 kHz (~11.1 µs per
    // tick). A global median would keep the slower cadence of an earlier
    // segment in a later, faster one and stretch the decode timeline. The 12 µs
    // floor is `US_TIMESCALE / 90_000 + 1`, the smallest step that cannot round
    // onto the neighbouring tick.
    //
    // Strictly increasing sequence with g[k] >= pts[k]: keep each packet's own
    // PTS when it moves forward, advance by that packet's cadence on a reorder,
    // and bump by 1 µs only to break an exact PTS tie (the minimal repair for
    // duplicate display delays).
    let mut increasing = Vec::with_capacity(pts.len());
    for (k, &packet_pts) in pts.iter().enumerate() {
        let packet_cadence = i128::from(durations[k]).max(12);
        let next = match increasing.last().copied() {
            Some(prev) if packet_pts < prev => prev.saturating_add(packet_cadence),
            Some(prev) if packet_pts == prev => prev.saturating_add(1),
            None | Some(_) => packet_pts,
        };
        increasing.push(next);
    }
    // Uniform delay covering every deficit keeps `dts <= pts` for all packets.
    let shift = increasing
        .iter()
        .zip(&pts)
        .fold(0_i128, |max, (&g, &p)| max.max(g - p));
    let mut result = Vec::with_capacity(pts.len());
    for ((k, &g), (&p, &duration)) in increasing
        .iter()
        .enumerate()
        .zip(pts.iter().zip(&durations))
    {
        let dts = g - shift;
        let dts = i64::try_from(dts)
            .map_err(|_| encode_error(format!("NVENC DTS of packet {k} exceeds i64 range")))?;
        let pts = i64::try_from(p)
            .map_err(|_| encode_error(format!("NVENC PTS of packet {k} exceeds i64 range")))?;
        debug_assert!(dts <= pts, "reordered DTS must not exceed PTS");
        result.push((pts, dts, duration));
    }
    Ok(result)
}

#[cfg(all(test, feature = "nvenc"))]
pub(crate) fn codec_supports_b_frames(
    codec: fastconv_domain::VideoCodec,
) -> Result<bool, PipelineError> {
    enabled::codec_supports_b_frames_enabled(codec)
}

/// Converts a validated image batch to `.heic` using hardware NVIDIA NVENC HEVC-Intra encoding.
///
/// Never silently falls back to CPU: returns an explicit error when NVENC support was not
/// compiled in (`--features nvenc`) or when CUDA/NVENC hardware initialization fails.
///
/// # Errors
///
/// Returns [`PipelineError`] on missing `nvenc` build feature, CUDA/NVENC failures,
/// non-opaque alpha inputs, or container/I/O failures.
pub fn execute_heic_batch(
    plan: &ImageBatchPlan,
    quality: ImageQuality,
) -> Result<ImageConversionReport, PipelineError> {
    #[cfg(feature = "nvenc")]
    {
        enabled::execute_heic_batch(plan, quality)
    }

    #[cfg(not(feature = "nvenc"))]
    {
        let _ = (plan, quality);
        Err(PipelineError::GpuBackendFailed {
            backend: GpuBackend::Nvenc,
            reason: "this fastconv binary was built without the 'nvenc' Cargo feature (rebuild with `--features nvenc` on an NVIDIA host)".to_owned(),
        })
    }
}

/// Streams decoded source packets through CUDA pinned memory into NVENC H.264
/// or HEVC, reporting progress once per emitted container packet.
///
/// Never falls back to CPU: an explicit error is returned if the `nvenc`
/// feature is missing or if any GPU/NVENC stage fails.
///
/// # Errors
///
/// Returns [`PipelineError::GpuBackendFailed`] for missing hardware support and
/// NVENC failures. Source decode errors and errors returned by
/// `on_frame_encoded` propagate unchanged after the decode thread is joined.
pub(crate) fn encode_video_nvenc_streaming<F>(
    path: &Path,
    source_track: &ContainerTrack,
    track_packets: &[ContainerPacket],
    mode: VideoEncodeMode,
    track_index: usize,
    on_frame_encoded: F,
) -> Result<(EncodedVideoStream, u64, String), PipelineError>
where
    F: FnMut() -> Result<(), PipelineError>,
{
    #[cfg(feature = "nvenc")]
    {
        enabled::encode_video_nvenc_streaming_enabled(
            path,
            source_track,
            track_packets,
            mode,
            track_index,
            false,
            on_frame_encoded,
        )
    }

    #[cfg(not(feature = "nvenc"))]
    {
        let _ = (
            path,
            source_track,
            track_packets,
            mode,
            track_index,
            on_frame_encoded,
        );
        Err(PipelineError::GpuBackendFailed {
            backend: GpuBackend::Nvenc,
            reason: "NVENC video encoding requires building with --features nvenc (no silent CPU fallback)".to_owned(),
        })
    }
}

/// NVDEC hardware decode followed by a GPU-resident handoff into NVENC: luma and
/// chroma are copied device-to-device into a separate NVENC slot. No CPU readback
/// or pinned host staging is involved, but the original NVDEC surface is not
/// directly shared. Every hardware failure is returned instead of a silent CPU
/// fallback.
///
/// # Errors
///
/// Returns [`PipelineError::DecodeFailed`] / [`PipelineError::GpuBackendFailed`]
/// when the `nvenc` feature is missing, NVDEC setup fails, caps reject the
/// source, or NVENC fails.
pub(crate) fn encode_video_nvdec_nvenc_streaming<F>(
    source_track: &crate::video_container::ContainerTrack,
    track_packets: &[crate::video_container::ContainerPacket],
    source_codec: fastconv_domain::VideoCodec,
    mode: VideoEncodeMode,
    track_index: usize,
    on_frame_encoded: F,
) -> Result<(EncodedVideoStream, u64, String), PipelineError>
where
    F: FnMut() -> Result<(), PipelineError>,
{
    #[cfg(feature = "nvenc")]
    {
        enabled::encode_video_nvdec_nvenc_streaming_enabled(
            source_track,
            track_packets,
            source_codec,
            mode,
            track_index,
            on_frame_encoded,
        )
    }

    #[cfg(not(feature = "nvenc"))]
    {
        let _ = (
            source_track,
            track_packets,
            source_codec,
            mode,
            track_index,
            on_frame_encoded,
        );
        Err(PipelineError::GpuBackendFailed {
            backend: GpuBackend::Nvenc,
            reason: "--hwdecode nvdec requires building with --features nvenc".to_owned(),
        })
    }
}

/// Test-only entry point that sets `disableBadapt = 1` so lookahead cannot
/// suppress B-frames on synthetic fixtures when verifying B-frame PTS
/// inversions and reorder roundtrips on B-frame-capable GPUs.
#[cfg(all(test, feature = "nvenc"))]
pub(crate) fn encode_video_nvenc_streaming_forced_b_frames<F>(
    path: &Path,
    source_track: &ContainerTrack,
    track_packets: &[ContainerPacket],
    mode: VideoEncodeMode,
    track_index: usize,
    on_frame_encoded: F,
) -> Result<(EncodedVideoStream, u64, String), PipelineError>
where
    F: FnMut() -> Result<(), PipelineError>,
{
    enabled::encode_video_nvenc_streaming_enabled(
        path,
        source_track,
        track_packets,
        mode,
        track_index,
        true,
        on_frame_encoded,
    )
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::thread;

    fn cq(quality: u8) -> (u8, u8) {
        quality_to_nvenc_cq(VideoQuality::new(quality).unwrap_or(VideoQuality::DEFAULT))
    }

    fn cq_256(quality: u8) -> i32 {
        let (target, lsb) = cq(quality);
        i32::from(target) * 256 + i32::from(lsb)
    }

    #[test]
    fn nvenc_cq_maps_quality_to_known_anchors() {
        assert_eq!(cq(0), (51, 0));
        assert_eq!(cq(50), (33, 128));
        assert_eq!(cq(80), (23, 0));
        assert_eq!(cq(100), (10, 0));
    }

    #[test]
    fn nvenc_cq_is_strictly_monotonic() {
        for q in 1..=100 {
            assert!(
                cq_256(q) < cq_256(q - 1),
                "quality {q} must yield a strictly lower CQ than {}",
                q - 1
            );
        }
        for q in 0..=100 {
            let (target, _) = cq(q);
            assert!((1..=51).contains(&target));
        }
    }

    #[test]
    fn nvenc_streaming_cancel_unblocks_the_decode_producer() -> Result<(), String> {
        // Mirrors the bounded ready/free channel topology of
        // `encode_video_nvenc_streaming_enabled`. When NVENC init or the
        // progress callback fails, both consumer ends are dropped before the
        // decode thread is joined; the producer must observe the closed
        // channels (or the cancel flag) instead of blocking on send/recv.
        let cancelled = AtomicBool::new(false);
        thread::scope(|scope| -> Result<(), String> {
            let (ready_tx, ready_rx) = mpsc::sync_channel::<u32>(2);
            let (free_tx, free_rx) = mpsc::sync_channel::<u32>(2);
            free_tx.send(0).map_err(|err| err.to_string())?;
            free_tx.send(1).map_err(|err| err.to_string())?;

            let cancel_flag = &cancelled;
            let producer = thread::Builder::new()
                .name("fastconv-nvenc-cancel-test".to_owned())
                .spawn_scoped(scope, move || -> u32 {
                    let mut frames = 0_u32;
                    while frames < 64 {
                        if cancel_flag.load(Ordering::Acquire) {
                            break;
                        }
                        if free_rx.recv().is_err() {
                            break;
                        }
                        if ready_tx.send(frames).is_err() {
                            break;
                        }
                        frames = frames.saturating_add(1);
                    }
                    frames
                })
                .map_err(|err| err.to_string())?;

            // Consumer fails after draining a couple of frames: drop both ends
            // exactly as the GPU thread does on an init/callback error.
            ready_rx.recv().map_err(|err| err.to_string())?;
            ready_rx.recv().map_err(|err| err.to_string())?;
            drop(ready_rx);
            drop(free_tx);
            cancelled.store(true, Ordering::Release);

            match producer.join() {
                Ok(frames) if frames < 64 => Ok(()),
                Ok(_) => Err("producer ignored cancellation".to_owned()),
                Err(_) => Err("decode producer panicked after cancellation".to_owned()),
            }
        })
    }

    #[test]
    fn nvenc_plan_disables_b_frames_when_gpu_lacks_support() {
        // Pascal GPUs expose HEVC + NV12 with max_b_frames = 0 and no B-ref.
        let pascal_hevc = NvencCodecCaps {
            max_b_frames: 0,
            bframe_ref_mode: 0,
            lookahead_supported: true,
            temporal_aq_supported: false,
        };
        let plan = NvencVideoPlan::from_caps(60, pascal_hevc);
        assert_eq!(plan.gop_length, 120);
        assert_eq!(plan.b_frames, 0);
        assert_eq!(plan.frame_interval_p(), 1);
        assert!(!plan.use_middle_b_ref);
        assert_eq!(plan.lookahead_depth, 16);
        assert!(!plan.temporal_aq);
        assert!(plan.ring_slots() > plan.output_delay());
    }

    #[test]
    fn nvenc_plan_enables_b_frames_and_lookahead_when_supported() {
        let turing_caps = NvencCodecCaps {
            max_b_frames: 4,
            bframe_ref_mode: 2,
            lookahead_supported: true,
            temporal_aq_supported: true,
        };
        let plan = NvencVideoPlan::from_caps(60, turing_caps);
        assert_eq!(plan.gop_length, 120);
        assert_eq!(plan.b_frames, 3);
        assert_eq!(plan.frame_interval_p(), 4);
        assert!(plan.use_middle_b_ref);
        assert_eq!(plan.lookahead_depth, 16);
        assert!(plan.temporal_aq);
        assert!(plan.ring_slots() > plan.output_delay() + 3);
    }

    #[test]
    fn reordered_timestamps_ippp() -> Result<(), PipelineError> {
        let timings = [
            (0, 40_000),
            (40_000, 40_000),
            (80_000, 40_000),
            (120_000, 40_000),
        ];
        let result = assign_reordered_timestamps(&timings, &[0, 1, 2, 3])?;
        assert_eq!(
            result,
            vec![
                (0, 0, 40_000),
                (40_000, 40_000, 40_000),
                (80_000, 80_000, 40_000),
                (120_000, 120_000, 40_000),
            ]
        );
        Ok(())
    }

    #[test]
    fn reordered_timestamps_ibbp() -> Result<(), PipelineError> {
        // Decode order I0 P3 B1 B2; B1 and B2 present before P3.
        let timings = [
            (0, 25_000),
            (25_000, 25_000),
            (50_000, 25_000),
            (75_000, 25_000),
        ];
        let result = assign_reordered_timestamps(&timings, &[0, 3, 1, 2])?;
        // One uniform 75 ms reorder delay, then 25 ms of decode-order cadence
        // between P3, B1 and B2 (no 1 µs bump that a 90 kHz tick would erase).
        assert_eq!(
            result,
            vec![
                (0, -75_000, 25_000),
                (75_000, 0, 25_000),
                (25_000, 25_000, 25_000),
                (50_000, 50_000, 25_000),
            ]
        );
        for (k, &(pts, dts, _)) in result.iter().enumerate() {
            assert!(
                dts <= pts,
                "packet {k}: dts {dts} must not exceed pts {pts}"
            );
            if k > 0 {
                assert!(dts > result[k - 1].1, "packet {k}: dts must increase");
            }
        }
        // MPEG-TS/MP4 video uses 90 kHz (1 tick ≈ 11.1 µs): the reordered P3/B1/B2
        // DTS must not collapse onto one tick, or a decoder sorts B1/B2 before
        // their P3 reference.
        let dts_90k = |dts: i64| -> i64 {
            i64::try_from((i128::from(dts) * 90_000 + 500_000) / 1_000_000).unwrap_or(i64::MAX)
        };
        let ticks: Vec<i64> = result[1..]
            .iter()
            .map(|&(_, dts, _)| dts_90k(dts))
            .collect();
        assert!(
            ticks.windows(2).all(|pair| pair[0] < pair[1]),
            "reordered DTS collapsed at 90 kHz: {ticks:?}"
        );
        Ok(())
    }

    #[test]
    fn reordered_timestamps_cadence_change_bounds_decode_lead() -> Result<(), PipelineError> {
        const SLOW: i64 = 40_000;
        const FAST: i64 = 10_000;
        let mut timings: Vec<(i64, u64)> = (0..1_000_i64).map(|k| (k * SLOW, 40_000)).collect();
        let fast_start = 1_000_i64 * SLOW;
        timings.extend((0..600_i64).map(|k| (fast_start + k * FAST, 10_000)));

        let mut display_indices: Vec<usize> = (0..timings.len()).collect();
        // One swapped pair inside the fast segment: packet 1_500 presents frame
        // 1_501, then 98 more 10 ms frames follow in presentation order.
        display_indices.swap(1_500, 1_501);

        let result = assign_reordered_timestamps(&timings, &display_indices)?;
        assert_eq!(result.len(), timings.len());
        for (k, &(pts, dts, _)) in result.iter().enumerate() {
            assert!(dts <= pts, "packet {k}: dts {dts} > pts {pts}");
            if k > 0 {
                assert!(dts > result[k - 1].1, "packet {k}: dts must increase");
            }
        }

        // A global 40 ms median keeps applying the old cadence to every later
        // 10 ms packet, adding 30 ms of decode lead per frame. Using the current
        // packet's own cadence settles at two fast frames instead.
        let lead = result[result.len() - 1].0 - result[result.len() - 1].1;
        assert!(
            (2 * FAST..=3 * FAST).contains(&lead),
            "decode lead {lead} us escaped the 10 ms segment cadence"
        );
        Ok(())
    }

    #[test]
    fn reordered_timestamps_b_pyramid() -> Result<(), PipelineError> {
        // Decode order 0, 4, 2, 1, 3 (B-pyramid).
        let timings = (0..5).map(|k| (k * 40_000, 40_000)).collect::<Vec<_>>();
        let result = assign_reordered_timestamps(&timings, &[0, 4, 2, 1, 3])?;
        for (k, &(pts, dts, _)) in result.iter().enumerate() {
            assert!(dts <= pts);
            if k > 0 {
                assert!(dts > result[k - 1].1);
            }
        }
        assert!(result[2].0 < result[1].0);
        Ok(())
    }

    #[test]
    fn reordered_timestamps_variable_gif_delays_stay_monotonic() -> Result<(), PipelineError> {
        // GIF delays are stored in centiseconds: 7 cs/9 cs here. NVENC emits
        // I0 P5 B2 B1 B4 B3 with two B-frames, so packet order differs from
        // presentation order and durations are not uniform.
        let timings: Vec<(i64, u64)> = [70_000, 90_000, 70_000, 90_000, 70_000, 90_000]
            .into_iter()
            .scan(0_i64, |pts, duration| {
                let current = *pts;
                *pts = pts.saturating_add(i64::try_from(duration).unwrap_or(0));
                Some((current, duration))
            })
            .collect();
        let result = assign_reordered_timestamps(&timings, &[0, 5, 2, 1, 4, 3])?;
        assert_eq!(result.len(), timings.len());
        for (k, &(pts, dts, duration)) in result.iter().enumerate() {
            assert!(dts <= pts, "packet {k}: dts {dts} > pts {pts}");
            if k > 0 {
                assert!(
                    dts > result[k - 1].1,
                    "packet {k}: DTS must strictly increase"
                );
            }
            let frame = timings[[0, 5, 2, 1, 4, 3][k]];
            assert_eq!(
                (pts, duration),
                frame,
                "packet {k}: PTS/duration must follow display index"
            );
        }
        Ok(())
    }

    #[test]
    fn reordered_timestamps_rejects_duplicate_display_indices() {
        let timings = [(0, 1), (1, 1), (2, 1)];
        assert!(assign_reordered_timestamps(&timings, &[0, 0, 2]).is_err());
    }

    #[test]
    fn reordered_timestamps_non_zero_origin_and_duplicate_pts() -> Result<(), PipelineError> {
        let timings = [
            (1_000_000, 33_333),
            (1_000_000, 33_333),
            (1_033_333, 33_333),
        ];
        let result = assign_reordered_timestamps(&timings, &[0, 1, 2])?;
        for (k, &(pts, dts, _)) in result.iter().enumerate() {
            assert!(dts <= pts);
            if k > 0 {
                assert!(dts > result[k - 1].1);
            }
        }
        let shifted = assign_reordered_timestamps(&[(0, 1), (1, 1)], &[0, 1])?;
        assert_eq!(shifted[0].1, 0);
        assert_eq!(shifted[1].1, 1);
        Ok(())
    }

    #[test]
    fn reordered_timestamps_rejects_bad_indices() {
        let timings = [(0, 1), (1, 1)];
        assert!(assign_reordered_timestamps(&timings, &[0]).is_err());
        assert!(assign_reordered_timestamps(&timings, &[0, 2]).is_err());
    }

    #[test]
    fn reordered_timestamps_repairs_paired_ties_in_linear_pass() -> Result<(), PipelineError> {
        let count = 20_000_usize;
        let timings: Vec<(i64, u64)> = (0..count)
            .map(|idx| (i64::try_from(idx / 2).unwrap_or(0).saturating_mul(10), 10))
            .collect();
        let display_indices: Vec<usize> = (0..count).collect();
        let result = assign_reordered_timestamps(&timings, &display_indices)?;
        assert_eq!(result.len(), count);
        for (k, &(pts, dts, _)) in result.iter().enumerate() {
            assert!(dts <= pts, "packet {k}: dts {dts} > pts {pts}");
            if k > 0 {
                assert!(dts > result[k - 1].1, "packet {k}: dts must increase");
            }
        }
        // Only the tied element of each pair moves back by 1; strictly increasing
        // slots stay untouched instead of shifting the entire prefix.
        assert_eq!(result[0].1, -1);
        assert_eq!(result[1].1, 0);
        assert_eq!(result[2].1, 9);
        assert_eq!(result[3].1, 10);
        Ok(())
    }

    #[test]
    fn reordered_timestamps_rejects_signed_range_exhaustion() {
        let timings = [(i64::MIN, 1), (i64::MIN, 1)];
        assert!(assign_reordered_timestamps(&timings, &[0, 1]).is_err());
    }
}

#[cfg(all(test, feature = "nvenc"))]
mod tests {
    use std::path::Path;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use fastconv_domain::{VideoCodec, VideoEncodeMode, VideoRateControl};

    use super::encode_video_nvenc_streaming;
    use crate::PipelineError;
    use crate::video_codecs::{Yuv420Frame, decode_video_track, encode_video_cpu};
    use crate::video_container::{ContainerPacket, ContainerTrack, US_TIMESCALE};

    fn synthetic_frames(width: u32, height: u32, count: usize) -> Vec<Yuv420Frame> {
        let width_usize = usize::try_from(width).unwrap_or(0);
        let height_usize = usize::try_from(height).unwrap_or(0);
        (0..count)
            .map(|index| {
                let luma = (0..width_usize.saturating_mul(height_usize))
                    .map(|sample_index| {
                        let shade = 20_usize.saturating_add(
                            sample_index.saturating_add(index.saturating_mul(13)) % 180,
                        );
                        u8::try_from(shade).unwrap_or(20)
                    })
                    .collect();
                let chroma = width_usize
                    .div_ceil(2)
                    .saturating_mul(height_usize.div_ceil(2));
                let u_value =
                    u8::try_from(96_usize.saturating_add(index.saturating_mul(3))).unwrap_or(128);
                let v_value =
                    u8::try_from(150_usize.saturating_sub(index.saturating_mul(2))).unwrap_or(120);
                Yuv420Frame {
                    width,
                    height,
                    y: luma,
                    u: vec![u_value; chroma],
                    v: vec![v_value; chroma],
                    pts_us: i64::try_from(index).unwrap_or(0).saturating_mul(40_000),
                    duration_us: 40_000,
                }
            })
            .collect()
    }

    /// Decodes the actual NVENC output to prove the pinned upload, registered
    /// resource pitch and bitstream drain produced valid, non-blank pictures.
    #[test]
    fn streaming_nvenc_encodes_av1_to_hevc_and_h264_roundtrip() -> Result<(), PipelineError> {
        // No CUDA device is an expected environment condition; a real device
        // must surface every encode or decode error.
        if cudarc::driver::CudaContext::new(0).is_err() {
            return Ok(());
        }
        let path = Path::new("nvenc-streaming-test");
        let frames = synthetic_frames(320, 240, 10);
        let av1 = encode_video_cpu(
            path,
            &frames,
            VideoEncodeMode::CpuAv1 {
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
        )?;
        let h264 = encode_video_cpu(
            path,
            &frames,
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
        )?;
        let expected_av1 = decode_video_track(path, &av1.track, &av1.packets)?;
        let expected_h264 = decode_video_track(path, &h264.track, &h264.packets)?;
        assert_eq!(expected_av1.len(), 10);
        assert_eq!(expected_h264.len(), 10);

        for (source, expected, mode, codec) in [
            (
                &av1,
                &expected_av1,
                VideoEncodeMode::NvencHevc {
                    rate_control: VideoRateControl::DEFAULT,
                },
                VideoCodec::Hevc,
            ),
            (
                &av1,
                &expected_av1,
                VideoEncodeMode::NvencH264 {
                    rate_control: VideoRateControl::DEFAULT,
                },
                VideoCodec::H264,
            ),
            (
                &h264,
                &expected_h264,
                VideoEncodeMode::NvencH264 {
                    rate_control: VideoRateControl::DEFAULT,
                },
                VideoCodec::H264,
            ),
        ] {
            let mut callbacks = 0_u32;
            let (stream, frames_encoded, device) = encode_video_nvenc_streaming(
                path,
                &source.track,
                &source.packets,
                mode,
                0,
                || {
                    callbacks = callbacks.saturating_add(1);
                    Ok(())
                },
            )?;
            assert_eq!(stream.track.codec, codec);
            assert_eq!(frames_encoded, 10);
            assert_eq!(stream.packets.len(), 10);
            assert_ne!(stream.track.config, Vec::<u8>::new());
            assert!(callbacks >= 10);
            assert_ne!(device, String::new());
            for (index, packet) in stream.packets.iter().enumerate() {
                assert!(packet.dts <= packet.pts);
                if index > 0 {
                    assert!(packet.dts > stream.packets[index - 1].dts);
                }
            }

            let decoded = decode_video_track(path, &stream.track, &stream.packets)?;
            assert_eq!(decoded.len(), 10);
            for (frame, decoded_source) in decoded.iter().zip(expected.iter()) {
                assert_eq!((frame.width, frame.height), (320, 240));
                assert_eq!(frame.pts_us, decoded_source.pts_us);
                assert_eq!(frame.duration_us, decoded_source.duration_us);
                let min = frame.y.iter().copied().min().unwrap_or(0);
                let max = frame.y.iter().copied().max().unwrap_or(0);
                assert!(
                    max.saturating_sub(min) > 8,
                    "NVENC produced a blank or corrupted frame"
                );
            }
        }
        Ok(())
    }

    /// A width congruent to 2 mod 4 must still encode: the registered pitch
    /// is the row width of the `cuMemAlloc` allocation (rounded to a multiple
    /// of 4), not the coded picture width.
    #[test]
    fn streaming_nvenc_roundtrip_at_width_854() -> Result<(), PipelineError> {
        if cudarc::driver::CudaContext::new(0).is_err() {
            return Ok(());
        }
        let path = Path::new("nvenc-streaming-width-854");
        let frames = synthetic_frames(854, 480, 6);
        let source = encode_video_cpu(
            path,
            &frames,
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
        )?;
        let expected = decode_video_track(path, &source.track, &source.packets)?;
        assert_eq!(expected.len(), 6);
        let (stream, frames_encoded, _) = encode_video_nvenc_streaming(
            path,
            &source.track,
            &source.packets,
            VideoEncodeMode::NvencHevc {
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
            || Ok(()),
        )?;
        assert_eq!(frames_encoded, 6);
        let decoded = decode_video_track(path, &stream.track, &stream.packets)?;
        assert_eq!(decoded.len(), 6);
        for frame in &decoded {
            assert_eq!((frame.width, frame.height), (854, 480));
            let min = frame.y.iter().copied().min().unwrap_or(0);
            let max = frame.y.iter().copied().max().unwrap_or(0);
            assert!(
                max.saturating_sub(min) > 8,
                "NVENC produced a blank or corrupted frame at width 854"
            );
        }
        Ok(())
    }

    #[test]
    fn streaming_nvenc_surfaces_decode_error_before_first_frame() {
        if cudarc::driver::CudaContext::new(0).is_err() {
            return;
        }
        let path = Path::new("nvenc-corrupt-first-packet");
        let track = ContainerTrack {
            codec: VideoCodec::H264,
            width: Some(320),
            height: Some(240),
            channels: None,
            sample_rate: None,
            config: Vec::new(),
            timescale: US_TIMESCALE,
            presentation_edit: None,
        };
        let corrupt = vec![ContainerPacket {
            track: 0,
            pts: 0,
            dts: 0,
            duration: 40_000,
            keyframe: true,
            data: vec![0xFF, 0xFF, 0xFF, 0xFF],
        }];
        let result = encode_video_nvenc_streaming(
            path,
            &track,
            &corrupt,
            VideoEncodeMode::NvencH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
            || Ok(()),
        );
        assert!(
            matches!(result, Err(PipelineError::DecodeFailed { .. })),
            "decode failure before the first frame must surface DecodeFailed, got {result:?}"
        );
    }

    #[test]
    fn streaming_nvenc_propagates_progress_callback_error_without_deadlock()
    -> Result<(), PipelineError> {
        if cudarc::driver::CudaContext::new(0).is_err() {
            return Ok(());
        }
        let path = Path::new("nvenc-streaming-error");
        let frames = synthetic_frames(320, 240, 6);
        let source = encode_video_cpu(
            path,
            &frames,
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
        )?;
        let source_for_thread = source.clone();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let mut calls = 0_u32;
            let result = encode_video_nvenc_streaming(
                path,
                &source_for_thread.track,
                &source_for_thread.packets,
                VideoEncodeMode::NvencH264 {
                    rate_control: VideoRateControl::DEFAULT,
                },
                0,
                || {
                    calls = calls.saturating_add(1);
                    Err(PipelineError::EncodeFailed {
                        path: path.to_path_buf(),
                        reason: "progress callback stopped encoding".to_owned(),
                    })
                },
            );
            if tx.send((result, calls)).is_err() {
                // The main thread gave up waiting; the test already failed.
            }
        });
        let (result, calls) = rx.recv_timeout(Duration::from_secs(30)).map_err(|err| {
            PipelineError::EncodeFailed {
                path: path.to_path_buf(),
                reason: format!("streaming NVENC deadlocked: {err}"),
            }
        })?;
        assert!(matches!(result, Err(PipelineError::EncodeFailed { .. })));
        assert_eq!(calls, 1);
        Ok(())
    }

    /// Exercises an early progress-callback error during the *first non-EOS*
    /// drain of a long lookahead encode (more frames than `output_delay`), so
    /// pending NVENC pictures must be flushed and drained before GPU buffers
    /// drop while preserving the original callback error.
    #[test]
    fn streaming_nvenc_aborts_pending_lookahead_work_before_buffer_drop()
    -> Result<(), PipelineError> {
        if cudarc::driver::CudaContext::new(0).is_err() {
            return Ok(());
        }
        let path = Path::new("nvenc-streaming-lookahead-abort");
        let frames = synthetic_frames(320, 240, 32);
        let source = encode_video_cpu(
            path,
            &frames,
            VideoEncodeMode::CpuH264 {
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
        )?;
        let source_for_thread = source.clone();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let mut calls = 0_u32;
            let result = encode_video_nvenc_streaming(
                path,
                &source_for_thread.track,
                &source_for_thread.packets,
                VideoEncodeMode::NvencH264 {
                    rate_control: VideoRateControl::DEFAULT,
                },
                0,
                || {
                    calls = calls.saturating_add(1);
                    Err(PipelineError::EncodeFailed {
                        path: path.to_path_buf(),
                        reason: "abort during lookahead drain".to_owned(),
                    })
                },
            );
            if tx.send((result, calls)).is_err() {
                // The main thread gave up waiting; the test already failed.
            }
        });
        let (result, calls) = rx.recv_timeout(Duration::from_secs(30)).map_err(|err| {
            PipelineError::EncodeFailed {
                path: path.to_path_buf(),
                reason: format!("streaming NVENC lookahead abort deadlocked: {err}"),
            }
        })?;
        match result {
            Err(PipelineError::EncodeFailed { reason, .. }) => {
                assert_eq!(reason, "abort during lookahead drain");
            }
            other => {
                return Err(PipelineError::EncodeFailed {
                    path: path.to_path_buf(),
                    reason: format!("expected preserved callback error, got {other:?}"),
                });
            }
        }
        assert_eq!(calls, 1);
        Ok(())
    }
}
