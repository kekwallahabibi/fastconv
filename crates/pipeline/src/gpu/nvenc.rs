#[cfg(feature = "nvenc")]
mod enabled {
    use std::collections::VecDeque;
    use std::ffi::c_void;
    use std::fs;
    use std::num::{NonZeroU64, NonZeroUsize};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::mpsc::{self, Receiver, SyncSender};
    use std::sync::{Arc, Mutex};
    use std::thread;

    use cudarc::driver::{CudaContext, CudaStream, HostSlice, PinnedHostSlice, result, sys};
    use fastconv_domain::{
        GpuBackend, HardwareBackend, ImageBatchItem, ImageBatchPlan, ImageConversionReport,
        ImageInputFormat, ImageQuality, VideoCodec, VideoEncodeMode, VideoQuality,
    };
    use nvidia_video_codec_sdk::{
        Bitstream, Buffer, EncodePictureParams, Encoder, EncoderInitParams, EncoderInput,
        RegisteredResource, Session,
        sys::nvEncodeAPI::{
            GUID, NV_ENC_BUFFER_FORMAT, NV_ENC_CODEC_H264_GUID, NV_ENC_CODEC_HEVC_GUID,
            NV_ENC_CONFIG, NV_ENC_INPUT_RESOURCE_TYPE, NV_ENC_PARAMS_RC_MODE, NV_ENC_PIC_TYPE,
            NV_ENC_PRESET_P4_GUID, NV_ENC_TUNING_INFO, NV_ENC_VUI_COLOR_PRIMARIES,
            NV_ENC_VUI_MATRIX_COEFFS, NV_ENC_VUI_TRANSFER_CHARACTERISTIC, NV_ENC_VUI_VIDEO_FORMAT,
        },
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
    /// Device buffers registered with NVENC; three deep so upload, encode and drain overlap.
    const GPU_RING_SLOTS: usize = 3;

    /// A fully decoded NV12 frame in a recyclable pinned host slot.
    struct FilledPinnedFrame {
        pinned: PinnedHostSlice<u8>,
        info: DecodedFrameInfo,
        pitch: u32,
    }

    /// One submitted NVENC picture waiting for its bitstream to be drained.
    #[derive(Clone, Copy)]
    struct InFlightFrame {
        slot: usize,
        pts_us: i64,
        duration_us: u64,
    }

    /// A synchronous `cuMemAlloc` device buffer owned by the NVENC ring.
    ///
    /// NVENC refuses CUDA pointers from cudarc's async memory pools
    /// (`NV_ENC_ERR_RESOURCE_REGISTER_FAILED`), so `CudaStream::alloc_zeros`
    /// cannot be used here despite being safe.
    struct DeviceBuffer {
        ptr: sys::CUdeviceptr,
        len: usize,
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
                len,
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

        fn len(&self) -> usize {
            self.len
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
            ImageInputFormat::Jpeg | ImageInputFormat::Qoi | ImageInputFormat::Bmp => {
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

    /// Maps video quality (`0..=100`) to NVENC `ConstQP` (`1..=51`).
    fn quality_to_video_qp(quality: VideoQuality) -> u32 {
        let inv = u32::from(100_u8.saturating_sub(quality.get()));
        (8 + (inv.saturating_mul(43).saturating_add(50) / 100)).clamp(1, 51)
    }

    fn video_error(reason: impl Into<String>) -> PipelineError {
        PipelineError::GpuBackendFailed {
            backend: GpuBackend::Nvenc,
            reason: reason.into(),
        }
    }

    /// One value drives the encode GUID, the codec-config union member, and
    /// the output codec, so a mismatched combination cannot be constructed.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum NvencVideoCodec {
        H264,
        Hevc,
    }

    impl NvencVideoCodec {
        fn from_mode(mode: VideoEncodeMode) -> Result<(Self, VideoQuality), PipelineError> {
            match mode {
                VideoEncodeMode::NvencH264 { quality } => Ok((Self::H264, quality)),
                VideoEncodeMode::NvencHevc { quality } => Ok((Self::Hevc, quality)),
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
        fn configure_codec_config(self, cfg: &mut NV_ENC_CONFIG) {
            match self {
                Self::H264 => {
                    // SAFETY: `cfg` came from `get_preset_config(NV_ENC_CODEC_H264_GUID, ...)`.
                    let h264_cfg = unsafe { &mut cfg.encodeCodecConfig.h264Config };
                    h264_cfg.set_repeatSPSPPS(1);
                    h264_cfg.set_outputAUD(0);
                    h264_cfg.chromaFormatIDC = 1;
                }
                Self::Hevc => {
                    // SAFETY: `cfg` came from `get_preset_config(NV_ENC_CODEC_HEVC_GUID, ...)`.
                    let hevc_cfg = unsafe { &mut cfg.encodeCodecConfig.hevcConfig };
                    hevc_cfg.set_repeatSPSPPS(1);
                    hevc_cfg.set_outputAUD(0);
                    hevc_cfg.set_chromaFormatIDC(1);
                    hevc_cfg.set_pixelBitDepthMinus8(0);
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

    // Mirrors cudarc's own `memcpy_htod`: the pinned slot's event is awaited and
    // re-recorded via `sync_guard`, so the decode thread cannot overwrite the
    // buffer while the asynchronous copy is still in flight.
    #[allow(unsafe_code)]
    fn dma_pinned_to_device(
        cuda_stream: &CudaStream,
        pinned: &PinnedHostSlice<u8>,
        dev_buf: &DeviceBuffer,
    ) -> Result<(), PipelineError> {
        let (host, sync_guard) = unsafe { pinned.stream_synced_slice(cuda_stream) };
        if host.len() > dev_buf.len() {
            return Err(video_error("NV12 frame exceeds CUDA device slot capacity"));
        }
        let dma =
            unsafe { result::memcpy_htod_async(dev_buf.ptr(), host, cuda_stream.cu_stream()) };
        drop(sync_guard);
        dma.map_err(|err| video_error(format!("failed to upload NV12 frame to GPU: {err}")))
    }

    /// Drains one encoded picture from a bitstream into a container packet.
    fn drain_encoded_picture(
        codec: NvencVideoCodec,
        bitstream: &mut Bitstream<'_>,
        track: &mut ContainerTrack,
        packets: &mut Vec<ContainerPacket>,
        frame: InFlightFrame,
        track_index: usize,
        on_frame_encoded: &mut impl FnMut() -> Result<(), PipelineError>,
    ) -> Result<(), PipelineError> {
        let lock = bitstream
            .lock()
            .map_err(|err| video_error(format!("failed to lock NVENC bitstream: {err}")))?;
        let data = lock.data();
        if data.is_empty() {
            return Err(video_error("NVENC produced an empty access unit"));
        }
        let keyframe = codec.is_keyframe(data);
        if track.config.is_empty()
            && keyframe
            && let Some(config) = codec.extract_config(data)
        {
            track.config = config;
        }
        let payload = codec.to_sample(data);
        drop(lock);
        packets.push(ContainerPacket {
            track: track_index,
            pts: frame.pts_us,
            dts: frame.pts_us,
            duration: frame.duration_us.max(1),
            keyframe,
            data: payload,
        });
        on_frame_encoded()?;
        Ok(())
    }

    /// GPU side of the streaming pipeline. Runs on the calling thread while the
    /// scoped decode thread refills pinned slots. `Ok(None)` means the channel
    /// closed before the first frame; the caller then surfaces the decode error.
    // Every stage input is passed explicitly so the GPU thread owns no hidden state.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    fn run_streaming_nvenc_stage<F>(
        encoder: Encoder,
        cuda_ctx: &Arc<CudaContext>,
        codec: NvencVideoCodec,
        fps: u32,
        qp: u32,
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
        let slot_bytes = nv12_buffer_len(width, height, pitch_usize)?;

        let mut preset_cfg = encoder
            .get_preset_config(
                guid,
                NV_ENC_PRESET_P4_GUID,
                NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_HIGH_QUALITY,
            )
            .map_err(|err| video_error(format!("failed to query NVENC P4 preset: {err}")))?;
        preset_cfg.presetCfg.gopLength = 60;
        preset_cfg.presetCfg.frameIntervalP = 1;
        preset_cfg.presetCfg.rcParams.rateControlMode =
            NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CONSTQP;
        preset_cfg.presetCfg.rcParams.constQP.qpIntra = qp;
        preset_cfg.presetCfg.rcParams.constQP.qpInterP = qp;
        preset_cfg.presetCfg.rcParams.constQP.qpInterB = qp;
        codec.configure_codec_config(&mut preset_cfg.presetCfg);

        let mut init = EncoderInitParams::new(guid, width, height);
        init.preset_guid(NV_ENC_PRESET_P4_GUID)
            .tuning_info(NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_HIGH_QUALITY)
            .framerate(fps, 1)
            .display_aspect_ratio(width, height)
            .enable_picture_type_decision()
            .encode_config(&mut preset_cfg.presetCfg);
        let session = encoder
            .start_session(NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12, init)
            .map_err(|err| video_error(format!("failed to start NVENC video session: {err}")))?;

        let cuda_stream = cuda_ctx.default_stream();
        // Declaration order fixes drop order: bitstreams, registrations, then device buffers
        // must all go before the session ends.
        let mut dev_bufs: Vec<DeviceBuffer> = Vec::with_capacity(GPU_RING_SLOTS);
        for _ in 0..GPU_RING_SLOTS {
            dev_bufs.push(DeviceBuffer::zeroed(cuda_ctx, &cuda_stream, slot_bytes)?);
        }
        // NVENC requires a mapped input resource to be unmapped once its
        // bitstream has been locked, and forbids writing device memory while a
        // mapping is active. Register (which also maps) a slot only after its
        // upload is queued, then drop the mapping after the oldest picture has
        // been drained. The map call also synchronizes the pending upload.
        let mut reg_resources: Vec<Option<RegisteredResource<'_, ()>>> =
            (0..GPU_RING_SLOTS).map(|_| None).collect();
        let mut bitstreams: Vec<Bitstream<'_>> = Vec::with_capacity(GPU_RING_SLOTS);
        for _ in 0..GPU_RING_SLOTS {
            bitstreams.push(session.create_output_bitstream().map_err(|err| {
                video_error(format!("failed to allocate NVENC output bitstream: {err}"))
            })?);
        }

        let mut track = ContainerTrack {
            codec: codec.output_codec(),
            width: u16::try_from(width).ok(),
            height: u16::try_from(height).ok(),
            channels: None,
            sample_rate: None,
            config: Vec::new(),
            timescale: US_TIMESCALE,
        };
        let mut packets = Vec::new();
        let mut in_flight: VecDeque<InFlightFrame> = VecDeque::with_capacity(GPU_RING_SLOTS);
        let mut frames_encoded = 0_u64;
        let mut next_slot = 0_usize;
        let mut next_frame = Some(first);

        while let Some(filled) = next_frame.take().or_else(|| ready_rx.recv().ok()) {
            let slot = next_slot % GPU_RING_SLOTS;
            next_slot = next_slot.saturating_add(1);

            // Once all device slots are busy, the oldest picture must complete
            // before its buffer can be safely overwritten by the next DMA.
            if in_flight.len() == GPU_RING_SLOTS
                && let Some(oldest) = in_flight.pop_front()
            {
                drain_encoded_picture(
                    codec,
                    &mut bitstreams[oldest.slot],
                    &mut track,
                    &mut packets,
                    oldest,
                    track_index,
                    on_frame_encoded,
                )?;
                // The bitstream is locked, so the encoder no longer reads this
                // mapping. Unmap and unregister before the slot is reused.
                reg_resources[oldest.slot] = None;
            }

            dma_pinned_to_device(&cuda_stream, &filled.pinned, &dev_bufs[slot])?;
            if free_tx.send(filled.pinned).is_err() {
                // The decode thread already exited; its error is surfaced after
                // the join, and dropping the send error recycles this slot.
            }
            if reg_resources[slot].is_none() {
                let ptr = usize::try_from(dev_bufs[slot].ptr())
                    .map_err(|_| video_error("CUDA device pointer exceeds usize"))?
                    as *mut c_void;
                reg_resources[slot] = Some(
                    session
                        .register_generic_resource(
                            (),
                            NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                            ptr,
                            pitch,
                        )
                        .map_err(|err| {
                            video_error(format!(
                                "failed to register NVENC CUDA device buffer: {err}"
                            ))
                        })?,
                );
            }
            let resource = reg_resources[slot]
                .as_mut()
                .ok_or_else(|| video_error("NVENC input slot was not registered"))?;
            let pts_us = filled.info.pts_us;
            let duration_us = filled.info.duration_us;
            session
                .encode_picture(
                    resource,
                    &mut bitstreams[slot],
                    EncodePictureParams {
                        input_timestamp: u64::try_from(pts_us.max(0)).unwrap_or(0),
                        ..Default::default()
                    },
                )
                .map_err(|err| video_error(format!("NVENC video encode failed: {err}")))?;
            in_flight.push_back(InFlightFrame {
                slot,
                pts_us,
                duration_us,
            });
            frames_encoded = frames_encoded.saturating_add(1);
        }

        session
            .end_of_stream()
            .map_err(|err| video_error(format!("NVENC EOS failed: {err}")))?;
        while let Some(frame) = in_flight.pop_front() {
            drain_encoded_picture(
                codec,
                &mut bitstreams[frame.slot],
                &mut track,
                &mut packets,
                frame,
                track_index,
                on_frame_encoded,
            )?;
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

    /// Decodes source packets straight into pinned host memory while NVENC
    /// uploads and encodes from a separate CUDA device ring.
    #[allow(clippy::too_many_lines)]
    pub(super) fn encode_video_nvenc_streaming_enabled<F>(
        path: &Path,
        source_track: &ContainerTrack,
        track_packets: &[ContainerPacket],
        mode: VideoEncodeMode,
        track_index: usize,
        mut on_frame_encoded: F,
    ) -> Result<(EncodedVideoStream, u64, String), PipelineError>
    where
        F: FnMut() -> Result<(), PipelineError>,
    {
        let (codec, quality) = NvencVideoCodec::from_mode(mode)?;
        let guid = codec.guid();

        // Validate everything that can fail without a decoded frame before any
        // thread is spawned, so unsupported hardware fails fast.
        let cuda_ctx = CudaContext::new(0)
            .map_err(|err| video_error(format!("failed to initialize CUDA device 0: {err}")))?;
        let device_name = cuda_ctx
            .name()
            .map_err(|err| video_error(format!("failed to query CUDA device name: {err}")))?;
        let encoder = Encoder::initialize_with_cuda(Arc::clone(&cuda_ctx))
            .map_err(|err| video_error(format!("failed to open NVENC encoder: {err}")))?;
        let guids = encoder
            .get_encode_guids()
            .map_err(|err| video_error(format!("failed to query NVENC codecs: {err}")))?;
        if !guids.contains(&guid) {
            return Err(video_error(
                "this NVIDIA GPU does not expose the requested NVENC video codec",
            ));
        }
        let formats = encoder
            .get_supported_input_formats(guid)
            .map_err(|err| video_error(format!("failed to query NVENC input formats: {err}")))?;
        if !formats.contains(&NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12) {
            return Err(video_error(
                "NVENC on this GPU does not support NV12 input buffers",
            ));
        }
        let fps = estimate_video_fps_from_packets(track_packets);
        let qp = quality_to_video_qp(quality);
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
                                    // CUDADEVICEPTR slots come from `cuMemAlloc`, so the
                                    // registered pitch must be the row width in bytes and a
                                    // multiple of 4. Round only the stride, not the frame
                                    // width: even widths like 854 or 1366 still encode.
                                    // Packing, upload and registration all use this one stride.
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
                .map_err(|err| {
                    video_error(format!("failed to spawn video decode thread: {err}"))
                })?;

            // Receive the first frame before configuring the session: its geometry
            // determines both the NVENC session and the ring buffer sizes.
            let first = ready_rx.recv().ok();
            let gpu_result = match first {
                Some(first) => run_streaming_nvenc_stage(
                    encoder,
                    &cuda_ctx,
                    codec,
                    fps,
                    qp,
                    track_index,
                    first,
                    &ready_rx,
                    &free_tx,
                    &mut on_frame_encoded,
                ),
                None => Ok(None),
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
}

#[cfg(not(feature = "nvenc"))]
use fastconv_domain::GpuBackend;
use fastconv_domain::{ImageBatchPlan, ImageConversionReport, ImageQuality, VideoEncodeMode};
use std::path::Path;

use crate::PipelineError;
use crate::video_codecs::EncodedVideoStream;
use crate::video_container::{ContainerPacket, ContainerTrack};

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

#[cfg(all(test, feature = "nvenc"))]
mod tests {
    use std::path::Path;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use fastconv_domain::{VideoCodec, VideoEncodeMode, VideoQuality};

    use super::encode_video_nvenc_streaming;
    use crate::PipelineError;
    use crate::video_codecs::{Yuv420Frame, decode_video_track, encode_video_cpu};

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
                quality: VideoQuality::DEFAULT,
            },
            0,
        )?;
        let h264 = encode_video_cpu(
            path,
            &frames,
            VideoEncodeMode::CpuH264 {
                quality: VideoQuality::DEFAULT,
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
                    quality: VideoQuality::DEFAULT,
                },
                VideoCodec::Hevc,
            ),
            (
                &av1,
                &expected_av1,
                VideoEncodeMode::NvencH264 {
                    quality: VideoQuality::DEFAULT,
                },
                VideoCodec::H264,
            ),
            (
                &h264,
                &expected_h264,
                VideoEncodeMode::NvencH264 {
                    quality: VideoQuality::DEFAULT,
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
            for (packet, decoded_source) in stream.packets.iter().zip(expected.iter()) {
                assert_eq!(packet.pts, decoded_source.pts_us);
                assert_eq!(packet.duration, decoded_source.duration_us);
            }

            let decoded = decode_video_track(path, &stream.track, &stream.packets)?;
            assert_eq!(decoded.len(), 10);
            for frame in &decoded {
                assert_eq!((frame.width, frame.height), (320, 240));
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
                quality: VideoQuality::DEFAULT,
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
                    quality: VideoQuality::DEFAULT,
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
}
