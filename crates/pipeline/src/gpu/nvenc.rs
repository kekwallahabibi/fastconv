#[cfg(feature = "nvenc")]
mod enabled {
    use std::fs;
    use std::num::{NonZeroU64, NonZeroUsize};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::mpsc::{self, Receiver, SyncSender};
    use std::sync::{Arc, Mutex};
    use std::thread;

    use cudarc::driver::CudaContext;
    use fastconv_domain::{
        GpuBackend, HardwareBackend, ImageBatchItem, ImageBatchPlan, ImageConversionReport,
        ImageInputFormat, ImageQuality,
    };
    use nvidia_video_codec_sdk::{
        Bitstream, Buffer, EncodePictureParams, Encoder, EncoderInitParams, EncoderInput, Session,
        sys::nvEncodeAPI::{
            NV_ENC_BUFFER_FORMAT, NV_ENC_CODEC_HEVC_GUID, NV_ENC_CONFIG, NV_ENC_PARAMS_RC_MODE,
            NV_ENC_PIC_TYPE, NV_ENC_PRESET_P4_GUID, NV_ENC_TUNING_INFO, NV_ENC_VUI_COLOR_PRIMARIES,
            NV_ENC_VUI_MATRIX_COEFFS, NV_ENC_VUI_TRANSFER_CHARACTERISTIC, NV_ENC_VUI_VIDEO_FORMAT,
        },
    };

    use crate::PipelineError;
    use crate::decode::{
        DecodeTargetHint, DecodedFrameMeta, DecodedPixelFormat, decode_image_into,
        precreate_target_directories,
    };
    use crate::gpu::heic::{HeicColorInfo, mux_annex_b_hevc_to_heic};

    const DECODE_QUEUE_CAPACITY: usize = 2;

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
            ImageInputFormat::Png
            | ImageInputFormat::Webp
            | ImageInputFormat::Tiff
            | ImageInputFormat::Jxl => {
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
}

#[cfg(not(feature = "nvenc"))]
use fastconv_domain::GpuBackend;
use fastconv_domain::{ImageBatchPlan, ImageConversionReport, ImageQuality};

use crate::PipelineError;

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
