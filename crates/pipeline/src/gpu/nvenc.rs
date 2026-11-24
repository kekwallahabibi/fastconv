#[cfg(feature = "nvenc")]
mod enabled {
    use std::fs;
    use std::num::{NonZeroU64, NonZeroUsize};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::mpsc::{self, Receiver, SyncSender};
    use std::sync::{Arc, Mutex};
    use std::thread;

    use cudarc::driver::CudaContext;
    use fastconv_domain::{
        GpuBackend, HardwareBackend, ImageBatchItem, ImageBatchPlan, ImageConversionReport,
        ImageInputFormat, ImageQuality, VideoCodec, VideoEncodeMode, VideoQuality,
    };
    use nvidia_video_codec_sdk::{
        Bitstream, Buffer, EncodePictureParams, Encoder, EncoderInitParams, EncoderInput, Session,
        sys::nvEncodeAPI::{
            GUID, NV_ENC_BUFFER_FORMAT, NV_ENC_CODEC_H264_GUID, NV_ENC_CODEC_HEVC_GUID,
            NV_ENC_CONFIG, NV_ENC_PARAMS_RC_MODE, NV_ENC_PIC_TYPE, NV_ENC_PRESET_P4_GUID,
            NV_ENC_TUNING_INFO, NV_ENC_VUI_COLOR_PRIMARIES, NV_ENC_VUI_MATRIX_COEFFS,
            NV_ENC_VUI_TRANSFER_CHARACTERISTIC, NV_ENC_VUI_VIDEO_FORMAT,
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
    use crate::video_codecs::{EncodedVideoStream, Yuv420Frame};
    use crate::video_container::{ContainerPacket, ContainerTrack, US_TIMESCALE};

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

    fn estimate_video_fps(frames: &[Yuv420Frame]) -> u32 {
        let mut durations: Vec<u64> = frames
            .iter()
            .map(|frame| frame.duration_us.max(1))
            .collect();
        durations.sort_unstable();
        let duration = if durations.is_empty() {
            1_000_000 / 30
        } else {
            durations[durations.len() / 2]
        };
        (1_000_000_u64 / duration.max(1)).clamp(1, 240) as u32
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

    /// Converts one 8-bit YUV 4:2:0 frame into a pitched NV12 upload buffer.
    fn prepare_nv12_payload<'a>(
        frame: &Yuv420Frame,
        pitch_u32: u32,
        buf: &'a mut Vec<u8>,
    ) -> Result<(&'a [u8], usize), PipelineError> {
        let width = usize::try_from(frame.width)
            .map_err(|_| video_error("NVENC frame width exceeds usize"))?;
        let height = usize::try_from(frame.height)
            .map_err(|_| video_error("NVENC frame height exceeds usize"))?;
        if width == 0 || height == 0 || width % 2 != 0 || height % 2 != 0 {
            return Err(video_error(format!(
                "NVENC requires even, non-zero dimensions; got {}x{}",
                frame.width, frame.height
            )));
        }
        let (cw, ch) = (width / 2, height / 2);
        if frame.y.len() != width.saturating_mul(height)
            || frame.u.len() != cw.saturating_mul(ch)
            || frame.v.len() != cw.saturating_mul(ch)
        {
            return Err(video_error("NVENC input frame has invalid plane sizes"));
        }
        let pitch = usize::try_from(pitch_u32)
            .map_err(|_| video_error("NVENC buffer pitch exceeds usize"))?;
        if pitch < width {
            return Err(video_error(format!(
                "NVENC buffer pitch ({pitch}) is smaller than frame width ({width})"
            )));
        }
        let expected = pitch
            .checked_mul(height)
            .and_then(|value| value.checked_add(pitch.checked_mul(ch)?))
            .ok_or_else(|| video_error("NVENC NV12 buffer size overflow"))?;
        buf.clear();
        buf.resize(expected, 0);
        for row in 0..height {
            let start = row.saturating_mul(pitch);
            let src = frame
                .y
                .get(row.saturating_mul(width)..row.saturating_mul(width).saturating_add(width))
                .ok_or_else(|| video_error("NVENC Y plane row out of bounds"))?;
            let dst = buf
                .get_mut(start..start.saturating_add(width))
                .ok_or_else(|| video_error("NVENC Y pitch row out of bounds"))?;
            dst.copy_from_slice(src);
        }
        for row in 0..ch {
            let start = height.saturating_add(row).saturating_mul(pitch);
            let dst = buf
                .get_mut(start..start.saturating_add(width))
                .ok_or_else(|| video_error("NVENC UV pitch row out of bounds"))?;
            for x in 0..cw {
                let u = frame.u[row.saturating_mul(cw).saturating_add(x)];
                let v = frame.v[row.saturating_mul(cw).saturating_add(x)];
                let u_index = x.saturating_mul(2);
                let v_index = u_index.saturating_add(1);
                if let Some(slot) = dst.get_mut(u_index) {
                    *slot = u;
                }
                if let Some(slot) = dst.get_mut(v_index) {
                    *slot = v;
                }
            }
        }
        Ok((buf.as_slice(), expected))
    }

    // The encoder, session and bitstream lifetimes must stay in one scope.
    #[allow(clippy::too_many_lines)]
    pub(super) fn encode_video_nvenc_enabled(
        path: &Path,
        frames: &[Yuv420Frame],
        mode: VideoEncodeMode,
        track_index: usize,
    ) -> Result<(EncodedVideoStream, String), PipelineError> {
        let (codec, quality) = NvencVideoCodec::from_mode(mode)?;
        let guid = codec.guid();
        let first = frames
            .first()
            .ok_or_else(|| video_error("no video frames to encode"))?;
        let (width, height) = (first.width, first.height);
        if width == 0 || height == 0 || width % 2 != 0 || height % 2 != 0 {
            return Err(video_error(format!(
                "NVENC requires even, non-zero dimensions; got {width}x{height}"
            )));
        }
        for frame in frames {
            if frame.width != width || frame.height != height {
                return Err(video_error("video frame dimensions change mid-stream"));
            }
        }
        let fps = estimate_video_fps(frames);
        let qp = quality_to_video_qp(quality);

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

        let (mut input, mut bitstream, pitch) = allocate_session_buffers(&session)?;
        let mut pitched_buf = Vec::new();
        let mut annex_b_frames: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
        for frame in frames {
            let (upload, expected_len) = prepare_nv12_payload(frame, pitch, &mut pitched_buf)?;
            write_pitched_nvenc_buffer(&mut input, expected_len, upload)?;
            session
                .encode_picture(&mut input, &mut bitstream, EncodePictureParams::default())
                .map_err(|err| video_error(format!("NVENC video encode failed: {err}")))?;
            let lock = bitstream
                .lock()
                .map_err(|err| video_error(format!("failed to lock NVENC bitstream: {err}")))?;
            annex_b_frames.push(lock.data().to_vec());
        }
        session
            .end_of_stream()
            .map_err(|err| video_error(format!("NVENC EOS failed: {err}")))?;
        // With IPPP GOPs the EOS flush normally produces no additional
        // bitstream; a lock failure here only means there is nothing to drain.
        if let Ok(lock) = bitstream.lock()
            && !lock.data().is_empty()
        {
            if let Some(last) = annex_b_frames.last_mut() {
                last.extend_from_slice(lock.data());
            } else {
                annex_b_frames.push(lock.data().to_vec());
            }
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
        let mut packets = Vec::with_capacity(annex_b_frames.len());
        for (index, data) in annex_b_frames.into_iter().enumerate() {
            if data.is_empty() {
                return Err(video_error("NVENC produced an empty access unit"));
            }
            let keyframe = codec.is_keyframe(&data);
            if track.config.is_empty()
                && keyframe
                && let Some(config) = codec.extract_config(&data)
            {
                track.config = config;
            }
            let payload = codec.to_sample(&data);
            let frame = frames.get(index).or_else(|| frames.last());
            let (pts, duration) = frame.map_or((0, 1_000_000 / u64::from(fps.max(1))), |frame| {
                (frame.pts_us, frame.duration_us.max(1))
            });
            packets.push(ContainerPacket {
                track: track_index,
                pts,
                dts: pts,
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
        let _ = path;
        Ok((EncodedVideoStream { track, packets }, device_name))
    }
}

#[cfg(not(feature = "nvenc"))]
use fastconv_domain::GpuBackend;
use fastconv_domain::{ImageBatchPlan, ImageConversionReport, ImageQuality, VideoEncodeMode};
use std::path::Path;

use crate::PipelineError;
use crate::video_codecs::{EncodedVideoStream, Yuv420Frame};

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

/// Encodes video frames with NVIDIA NVENC H.264 or HEVC.
///
/// Never falls back to CPU: an explicit error is returned if the `nvenc`
/// feature is missing or the selected mode is not an NVENC video mode.
///
/// # Errors
///
/// Returns [`PipelineError::GpuBackendFailed`] for missing hardware support or
/// an unsupported mode, and [`PipelineError::EncodeFailed`] if the encoder fails.
pub(crate) fn encode_video_nvenc(
    path: &Path,
    frames: &[Yuv420Frame],
    mode: VideoEncodeMode,
    track_index: usize,
) -> Result<(EncodedVideoStream, String), PipelineError> {
    #[cfg(feature = "nvenc")]
    {
        enabled::encode_video_nvenc_enabled(path, frames, mode, track_index)
    }

    #[cfg(not(feature = "nvenc"))]
    {
        let _ = (path, frames, mode, track_index);
        Err(PipelineError::GpuBackendFailed {
            backend: GpuBackend::Nvenc,
            reason: "NVENC video encoding requires building with --features nvenc (no silent CPU fallback)".to_owned(),
        })
    }
}
