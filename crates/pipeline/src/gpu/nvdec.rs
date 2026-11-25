//! NVDEC (`libnvcuvid.so.1`) hardware decoding with GPU-resident surface handoff.
//!
//! The library is loaded dynamically so a build with `--features nvenc` still
//! starts on hosts without the Video Decode SDK; a forced NVDEC request then
//! fails loudly instead of silently falling back to CPU decode.
//!
//! RAII drop order is parser -> decoder -> context lock -> CUDA context, and a
//! mapped surface is always released by [`MappedCuvidFrame::drop`], including
//! on error paths. No panic may unwind across an `extern "C"` callback.

#[allow(unsafe_code)]
mod enabled {
    use std::collections::HashMap;
    use std::ffi::{c_int, c_uint, c_ulonglong, c_void};
    use std::ptr;
    use std::sync::OnceLock;

    use cudarc::driver::{CudaContext, CudaStream, sys};
    use fastconv_domain::VideoCodec;
    use libloading::Library;
    use nvidia_video_codec_sdk::sys::{cuviddec, nvcuvid};

    use crate::PipelineError;
    use crate::video_container::ContainerPacket;

    const US_CLOCK_RATE: c_uint = 1_000_000;
    const NV12_OUTPUT_BIT: u16 =
        1 << (cuviddec::cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12 as u16);

    type CuResult = sys::CUresult;
    type GetCapsFn = unsafe extern "C" fn(*mut cuviddec::CUVIDDECODECAPS) -> CuResult;
    type CreateDecoderFn = unsafe extern "C" fn(
        *mut cuviddec::CUvideodecoder,
        *mut cuviddec::CUVIDDECODECREATEINFO,
    ) -> CuResult;
    type DestroyDecoderFn = unsafe extern "C" fn(cuviddec::CUvideodecoder) -> CuResult;
    type DecodePictureFn =
        unsafe extern "C" fn(cuviddec::CUvideodecoder, *mut cuviddec::CUVIDPICPARAMS) -> CuResult;
    type MapFrameFn = unsafe extern "C" fn(
        cuviddec::CUvideodecoder,
        c_int,
        *mut c_ulonglong,
        *mut c_uint,
        *mut cuviddec::CUVIDPROCPARAMS,
    ) -> CuResult;
    type UnmapFrameFn = unsafe extern "C" fn(cuviddec::CUvideodecoder, c_ulonglong) -> CuResult;
    type CreateParserFn = unsafe extern "C" fn(
        *mut nvcuvid::CUvideoparser,
        *mut nvcuvid::CUVIDPARSERPARAMS,
    ) -> CuResult;
    type ParseDataFn = unsafe extern "C" fn(
        nvcuvid::CUvideoparser,
        *mut nvcuvid::CUVIDSOURCEDATAPACKET,
    ) -> CuResult;
    type DestroyParserFn = unsafe extern "C" fn(nvcuvid::CUvideoparser) -> CuResult;
    type LockCreateFn =
        unsafe extern "C" fn(*mut cuviddec::CUvideoctxlock, sys::CUcontext) -> CuResult;
    type LockDestroyFn = unsafe extern "C" fn(cuviddec::CUvideoctxlock) -> CuResult;
    type Memcpy2DFn = unsafe extern "C" fn(*const sys::CUDA_MEMCPY2D, sys::CUstream) -> CuResult;

    fn nvdec_error(reason: impl Into<String>) -> PipelineError {
        PipelineError::DecodeFailed {
            path: std::path::PathBuf::from("nvdec"),
            reason: reason.into(),
        }
    }

    fn cu_result(status: CuResult, context: &str) -> Result<(), PipelineError> {
        if status == CuResult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(nvdec_error(format!("{context}: {status:?}")))
        }
    }

    /// All NVCUVID and CUDA driver entry points used by this module.
    struct CuvidApi {
        _library: Library,
        _cuda: Library,
        get_caps: GetCapsFn,
        create_decoder: CreateDecoderFn,
        destroy_decoder: DestroyDecoderFn,
        decode_picture: DecodePictureFn,
        map_frame: MapFrameFn,
        unmap_frame: UnmapFrameFn,
        create_parser: CreateParserFn,
        parse_data: ParseDataFn,
        destroy_parser: DestroyParserFn,
        lock_create: LockCreateFn,
        lock_destroy: LockDestroyFn,
        memcpy2d: Memcpy2DFn,
    }

    /// Resolves one exported symbol using its exact declared signature.
    // SAFETY: every caller names a function whose C ABI matches `T` as declared
    // in nvcuvid.h / cuviddec.h / cuda_runtime_api.h.
    #[allow(unsafe_code)]
    fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, String> {
        let mut bytes = name.to_vec();
        bytes.push(0);
        unsafe {
            library
                .get::<T>(&bytes)
                .map(|symbol| *symbol)
                .map_err(|err| {
                    format!(
                        "libnvcuvid.so.1 is missing {}: {err}",
                        String::from_utf8_lossy(name)
                    )
                })
        }
    }

    impl CuvidApi {
        #[allow(unsafe_code)]
        fn load() -> Result<Self, String> {
            // SAFETY: loading a system codec/driver library has no Rust safety
            // contract; all resolved symbols are checked against SDK signatures.
            let (library, cuda) = unsafe {
                (
                    Library::new("libnvcuvid.so.1").or_else(|_| Library::new("libnvcuvid.so")),
                    Library::new("libcuda.so.1").or_else(|_| Library::new("libcuda.so")),
                )
            };
            let library =
                library.map_err(|err| format!("failed to load libnvcuvid.so.1: {err}"))?;
            let cuda = cuda.map_err(|err| format!("failed to load libcuda.so.1: {err}"))?;
            let memcpy2d = symbol::<Memcpy2DFn>(&cuda, b"cuMemcpy2DAsync_v2")
                .map_err(|err| format!("libcuda.so.1: {err}"))?;
            Ok(Self {
                get_caps: symbol(&library, b"cuvidGetDecoderCaps")?,
                create_decoder: symbol(&library, b"cuvidCreateDecoder")?,
                destroy_decoder: symbol(&library, b"cuvidDestroyDecoder")?,
                decode_picture: symbol(&library, b"cuvidDecodePicture")?,
                map_frame: symbol(&library, b"cuvidMapVideoFrame64")?,
                unmap_frame: symbol(&library, b"cuvidUnmapVideoFrame64")?,
                create_parser: symbol(&library, b"cuvidCreateVideoParser")?,
                parse_data: symbol(&library, b"cuvidParseVideoData")?,
                destroy_parser: symbol(&library, b"cuvidDestroyVideoParser")?,
                lock_create: symbol(&library, b"cuvidCtxLockCreate")?,
                lock_destroy: symbol(&library, b"cuvidCtxLockDestroy")?,
                memcpy2d,
                // Moved last: all symbol lookups above borrow `library` and `cuda`.
                _library: library,
                _cuda: cuda,
            })
        }
    }

    static CUVID_API: OnceLock<Result<CuvidApi, String>> = OnceLock::new();

    fn api() -> Result<&'static CuvidApi, PipelineError> {
        CUVID_API
            .get_or_init(CuvidApi::load)
            .as_ref()
            .map_err(|err| nvdec_error(err.clone()))
    }

    fn codec_type(codec: VideoCodec) -> Result<cuviddec::cudaVideoCodec, PipelineError> {
        match codec {
            VideoCodec::H264 => Ok(cuviddec::cudaVideoCodec::cudaVideoCodec_H264),
            VideoCodec::Hevc => Ok(cuviddec::cudaVideoCodec::cudaVideoCodec_HEVC),
            VideoCodec::Vp9 => Ok(cuviddec::cudaVideoCodec::cudaVideoCodec_VP9),
            other => Err(nvdec_error(format!("NVDEC cannot decode {other}"))),
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Geometry {
        coded_width: u32,
        coded_height: u32,
        display_width: u32,
        display_height: u32,
        bit_depth: u8,
        surfaces: u32,
    }

    /// Exhaustive `cuvidGetDecoderCaps` check done before any decoder is created.
    ///
    /// Rejects unsupported codec/chroma/bit-depth pairs (e.g. 10-bit H.264 on
    /// Turing), dimensions outside the hardware limits and surfaces lacking
    /// NV12 output. A hard error is returned; there is no CPU fallback.
    fn check_caps(
        codec: cuviddec::cudaVideoCodec,
        bit_depth: u8,
        width: u32,
        height: u32,
    ) -> Result<u32, PipelineError> {
        let api = api()?;
        // NVCUVID reports CUDA_ERROR_INVALID_DEVICE until the CUDA driver is
        // initialised, even for a pure capability query.
        cudarc::driver::result::init()
            .map_err(|err| nvdec_error(format!("failed to initialise CUDA driver: {err}")))?;
        let mut caps = cuviddec::CUVIDDECODECAPS {
            eCodecType: codec,
            eChromaFormat: cuviddec::cudaVideoChromaFormat::cudaVideoChromaFormat_420,
            nBitDepthMinus8: u32::from(bit_depth.saturating_sub(8)),
            ..Default::default()
        };
        // SAFETY: `caps` is a fully initialised SDK struct passed by pointer.
        #[allow(unsafe_code)]
        let status = unsafe { (api.get_caps)(&raw mut caps) };
        cu_result(status, "cuvidGetDecoderCaps failed")?;
        if caps.bIsSupported == 0 {
            return Err(nvdec_error(format!(
                "NVDEC does not support codec {codec:?} at {bit_depth}-bit 4:2:0"
            )));
        }
        if caps.nBitDepthMinus8 + 8 != u32::from(bit_depth) {
            return Err(nvdec_error(format!(
                "NVDEC reported {}-bit support for a {bit_depth}-bit request",
                caps.nBitDepthMinus8 + 8
            )));
        }
        if width < u32::from(caps.nMinWidth)
            || width > caps.nMaxWidth
            || height < u32::from(caps.nMinHeight)
            || height > caps.nMaxHeight
        {
            return Err(nvdec_error(format!(
                "NVDEC supports {}-bit frames of {}x{}..{}x{}, got {width}x{height}",
                bit_depth, caps.nMinWidth, caps.nMinHeight, caps.nMaxWidth, caps.nMaxHeight
            )));
        }
        let macroblocks = u64::from(width.div_ceil(16)) * u64::from(height.div_ceil(16));
        if macroblocks > u64::from(caps.nMaxMBCount) {
            return Err(nvdec_error(format!(
                "NVDEC macroblock limit {} exceeded by {width}x{height} ({macroblocks})",
                caps.nMaxMBCount
            )));
        }
        if caps.nOutputFormatMask & NV12_OUTPUT_BIT == 0 {
            return Err(nvdec_error(
                "NVDEC device has no NV12 output format; NVENC requires NV12 surfaces".to_owned(),
            ));
        }
        Ok(u32::from(caps.nNumNVDECs.max(1)))
    }

    /// One decode surface mapped for the duration of this guard. `Drop` always
    /// calls `cuvidUnmapVideoFrame64`, even when the consumer returns an error.
    pub(crate) struct MappedCuvidFrame<'a> {
        decoder: &'a CuvidDecoder,
        ptr: sys::CUdeviceptr,
        pitch: u32,
        geometry: Geometry,
    }

    impl MappedCuvidFrame<'_> {
        /// Device-to-device copy of the cropped NV12 surfaces into a registered
        /// NVENC slot. Luma and chroma are copied separately so odd display
        /// widths/heights, crop offsets and differing pitches are handled
        /// without touching rows outside the destination allocation.
        #[allow(unsafe_code)]
        pub(crate) fn copy_to_nvenc_slot(
            &self,
            stream: &CudaStream,
            dst: sys::CUdeviceptr,
            dst_pitch: usize,
            dst_width: u32,
            dst_height: u32,
        ) -> Result<(), PipelineError> {
            let g = self.geometry;
            if dst_width != g.display_width || dst_height != g.display_height {
                return Err(nvdec_error(format!(
                    "NVDEC surface {}x{} does not match NVENC slot {}x{}",
                    g.display_width, g.display_height, dst_width, dst_height
                )));
            }
            if dst_pitch < usize::try_from(dst_width).unwrap_or(usize::MAX) {
                return Err(nvdec_error("NVENC slot pitch is narrower than the frame"));
            }
            let src_pitch = usize::try_from(self.pitch).unwrap_or(usize::MAX);
            let chroma_bytes = usize::try_from(g.display_width.div_ceil(2) * 2).unwrap_or(0);
            let src_chroma = self.ptr.saturating_add(
                u64::try_from(
                    src_pitch.saturating_mul(usize::try_from(g.display_height).unwrap_or(0)),
                )
                .unwrap_or(u64::MAX),
            );
            let dst_chroma = dst.saturating_add(
                u64::try_from(dst_pitch.saturating_mul(usize::try_from(dst_height).unwrap_or(0)))
                    .unwrap_or(u64::MAX),
            );
            let api = self.decoder.api;
            let mut copy_accepted = false;
            for (src, dst_ptr, width_bytes, height) in [
                (
                    self.ptr,
                    dst,
                    usize::try_from(g.display_width).unwrap_or(0),
                    usize::try_from(g.display_height).unwrap_or(0),
                ),
                (
                    src_chroma,
                    dst_chroma,
                    chroma_bytes,
                    usize::try_from(g.display_height.div_ceil(2)).unwrap_or(0),
                ),
            ] {
                if width_bytes == 0 || height == 0 {
                    continue;
                }
                let copy = sys::CUDA_MEMCPY2D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    srcHost: ptr::null(),
                    srcDevice: src,
                    srcArray: ptr::null_mut(),
                    srcPitch: src_pitch,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    dstHost: ptr::null_mut(),
                    dstDevice: dst_ptr,
                    dstArray: ptr::null_mut(),
                    dstPitch: dst_pitch,
                    WidthInBytes: width_bytes,
                    Height: height,
                };
                // SAFETY: both rectangles stay inside their respective NV12
                // allocations; the stream remains valid for the stage lifetime.
                let status = unsafe { (api.memcpy2d)(&raw const copy, stream.cu_stream()) };
                if let Err(err) = cu_result(status, "cuMemcpy2DAsync (NVDEC -> NVENC) failed") {
                    if copy_accepted {
                        // The earlier copy is already queued on this stream. Drain
                        // it before the caller unmaps the NVDEC surface; the
                        // submission error stays the authoritative cause.
                        let drained = stream.synchronize().map_err(|sync| {
                            nvdec_error(format!(
                                "CUDA stream synchronize after failed NVDEC -> NVENC copy: {sync}"
                            ))
                        });
                        if let Err(sync_err) = drained {
                            return Err(nvdec_error(format!("{err}; {sync_err}")));
                        }
                    }
                    return Err(err);
                }
                copy_accepted = true;
            }
            // NVENC runs on a separate hardware engine that does not order
            // against CUDA streams; synchronize before mapping/encoding the
            // slot or unmapping the NVDEC surface.
            stream.synchronize().map_err(|err| {
                nvdec_error(format!(
                    "CUDA stream synchronize after NVDEC -> NVENC copy failed: {err}"
                ))
            })?;
            Ok(())
        }
    }

    impl Drop for MappedCuvidFrame<'_> {
        #[allow(unsafe_code)]
        fn drop(&mut self) {
            // SAFETY: the decoder outlives this guard by construction and the
            // mapped pointer came from `cuvidMapVideoFrame64`. Never panic here.
            let _status = unsafe { (self.decoder.api.unmap_frame)(self.decoder.handle, self.ptr) };
        }
    }

    struct CuvidDecoder {
        api: &'static CuvidApi,
        handle: cuviddec::CUvideodecoder,
    }

    impl Drop for CuvidDecoder {
        #[allow(unsafe_code)]
        fn drop(&mut self) {
            // SAFETY: the decoder was created by `cuvidCreateDecoder` on this API.
            if !self.handle.is_null() {
                let _status = unsafe { (self.api.destroy_decoder)(self.handle) };
            }
        }
    }

    struct CuvidCtxLock {
        api: &'static CuvidApi,
        handle: cuviddec::CUvideoctxlock,
    }

    impl Drop for CuvidCtxLock {
        #[allow(unsafe_code)]
        fn drop(&mut self) {
            // SAFETY: the lock was created by `cuvidCtxLockCreate` on this API.
            if !self.handle.is_null() {
                let _status = unsafe { (self.api.lock_destroy)(self.handle) };
            }
        }
    }

    /// Parser handle. Declared after the decoder/lock state it borrows so it
    /// drops first; no callback may run after the parser is destroyed.
    struct CuvidParser {
        api: &'static CuvidApi,
        handle: nvcuvid::CUvideoparser,
    }

    impl Drop for CuvidParser {
        #[allow(unsafe_code)]
        fn drop(&mut self) {
            // SAFETY: the parser was created by `cuvidCreateVideoParser`.
            if !self.handle.is_null() {
                let _status = unsafe { (self.api.destroy_parser)(self.handle) };
            }
        }
    }

    type FrameCallback<'a> =
        &'a mut dyn FnMut(MappedCuvidFrame<'_>, i64, u64) -> Result<(), PipelineError>;

    struct CallbackState<'a> {
        api: &'static CuvidApi,
        codec: cuviddec::cudaVideoCodec,
        source: VideoCodec,
        ctx_lock: cuviddec::CUvideoctxlock,
        stream: sys::CUstream,
        decoder: Option<CuvidDecoder>,
        geometry: Option<Geometry>,
        error: Option<PipelineError>,
        durations: HashMap<i64, u64>,
        displayed_frames: u64,
        on_frame: FrameCallback<'a>,
    }

    impl CallbackState<'_> {
        fn fail(&mut self, error: PipelineError) -> c_int {
            self.error.get_or_insert(error);
            0
        }
    }

    // Every capability and reconfigure case is validated inline because no
    // error may unwind across the C callback boundary.
    #[allow(unsafe_code, clippy::too_many_lines)]
    unsafe extern "C" fn sequence_callback(
        user: *mut c_void,
        format: *mut nvcuvid::CUVIDEOFORMAT,
    ) -> c_int {
        // SAFETY: `user` points at a live `CallbackState` and `format` at a
        // sequence header supplied by NVCUVID for the duration of this call.
        unsafe {
            if let (Some(state), Some(format)) =
                (user.cast::<CallbackState<'_>>().as_mut(), format.as_ref())
            {
                let bit_depth = format.bit_depth_luma_minus8.saturating_add(8);
                if format.chroma_format
                    != cuviddec::cudaVideoChromaFormat::cudaVideoChromaFormat_420
                {
                    return state.fail(nvdec_error(format!(
                        "NVDEC requires 4:2:0 chroma, got {:?}",
                        format.chroma_format
                    )));
                }
                let width = format.coded_width;
                let height = format.coded_height;
                let result = (|| -> Result<c_int, PipelineError> {
                    check_caps(state.codec, bit_depth, width, height)?;
                    if let Some(existing) = &state.geometry {
                        let next = Geometry {
                            coded_width: width,
                            coded_height: height,
                            display_width: u32::try_from(format.display_area.right)
                                .unwrap_or(0)
                                .saturating_sub(
                                    u32::try_from(format.display_area.left).unwrap_or(0),
                                ),
                            display_height: u32::try_from(format.display_area.bottom)
                                .unwrap_or(0)
                                .saturating_sub(
                                    u32::try_from(format.display_area.top).unwrap_or(0),
                                ),
                            bit_depth,
                            surfaces: u32::from(format.min_num_decode_surfaces).max(8),
                        };
                        if *existing != next {
                            return Err(nvdec_error(
                                "NVDEC stream geometry changed mid-sequence; reinitialise the decoder",
                            ));
                        }
                        return Ok(1);
                    }
                    let display_width = u32::try_from(format.display_area.right)
                        .unwrap_or(0)
                        .saturating_sub(u32::try_from(format.display_area.left).unwrap_or(0));
                    let display_height = u32::try_from(format.display_area.bottom)
                        .unwrap_or(0)
                        .saturating_sub(u32::try_from(format.display_area.top).unwrap_or(0));
                    if display_width == 0 || display_height == 0 {
                        return Err(nvdec_error("NVDEC reported an empty display area"));
                    }
                    let surfaces = u32::from(format.min_num_decode_surfaces).max(8);
                    let mut info = cuviddec::CUVIDDECODECREATEINFO {
                        ulWidth: u64::from(width),
                        ulHeight: u64::from(height),
                        ulNumDecodeSurfaces: u64::from(surfaces),
                        CodecType: state.codec,
                        ChromaFormat: cuviddec::cudaVideoChromaFormat::cudaVideoChromaFormat_420,
                        ulCreationFlags: cuviddec::cudaVideoCreateFlags::cudaVideoCreate_PreferCUVID
                            as u64,
                        bitDepthMinus8: u64::from(bit_depth.saturating_sub(8)),
                        OutputFormat: cuviddec::cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12,
                        DeinterlaceMode:
                            cuviddec::cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Weave,
                        ulTargetWidth: u64::from(display_width),
                        ulTargetHeight: u64::from(display_height),
                        ulNumOutputSurfaces: 4,
                        vidLock: state.ctx_lock,
                        ..Default::default()
                    };
                    let display_left = i16::try_from(format.display_area.left).unwrap_or(0);
                    let display_top = i16::try_from(format.display_area.top).unwrap_or(0);
                    let target_right = i16::try_from(display_width).unwrap_or(i16::MAX);
                    let target_bottom = i16::try_from(display_height).unwrap_or(i16::MAX);
                    info.display_area.left = display_left;
                    info.display_area.top = display_top;
                    info.display_area.right = display_left.saturating_add(target_right);
                    info.display_area.bottom = display_top.saturating_add(target_bottom);
                    info.target_rect.left = 0;
                    info.target_rect.top = 0;
                    info.target_rect.right = target_right;
                    info.target_rect.bottom = target_bottom;
                    let mut decoder = ptr::null_mut();
                    cu_result(
                        (state.api.create_decoder)(&raw mut decoder, &raw mut info),
                        "cuvidCreateDecoder failed",
                    )?;
                    state.decoder = Some(CuvidDecoder {
                        api: state.api,
                        handle: decoder,
                    });
                    state.geometry = Some(Geometry {
                        coded_width: width,
                        coded_height: height,
                        display_width,
                        display_height,
                        bit_depth,
                        surfaces,
                    });
                    // Return the exact decode surface count so the parser updates
                    // its DPB limit to match `ulNumDecodeSurfaces`.
                    Ok(c_int::try_from(surfaces).unwrap_or(1))
                })();
                return match result {
                    Ok(ret) => ret,
                    Err(err) => state.fail(err),
                };
            }
            0
        }
    }

    #[allow(unsafe_code)]
    unsafe extern "C" fn decode_callback(
        user: *mut c_void,
        picture: *mut cuviddec::CUVIDPICPARAMS,
    ) -> c_int {
        // SAFETY: `picture` is a valid picture parameter block supplied by the
        // parser; the decoder was created in the sequence callback.
        unsafe {
            if let (Some(state), Some(picture)) =
                (user.cast::<CallbackState<'_>>().as_mut(), picture.as_ref())
            {
                if let Some(decoder) = state.decoder.as_ref() {
                    let result = cu_result(
                        (state.api.decode_picture)(
                            decoder.handle,
                            std::ptr::from_ref(picture).cast_mut(),
                        ),
                        "cuvidDecodePicture failed",
                    );
                    return match result {
                        Ok(()) => 1,
                        Err(err) => state.fail(err),
                    };
                }
                return state.fail(nvdec_error("NVDEC decoder was not created"));
            }
            0
        }
    }

    #[allow(unsafe_code)]
    unsafe extern "C" fn display_callback(
        user: *mut c_void,
        display: *mut nvcuvid::CUVIDPARSERDISPINFO,
    ) -> c_int {
        // SAFETY: `display` is a valid parser display info and the state lives
        // for the whole `cuvidParseVideoData` call.
        unsafe {
            if let (Some(state), Some(display)) =
                (user.cast::<CallbackState<'_>>().as_mut(), display.as_ref())
            {
                let Some(decoder) = state.decoder.as_ref() else {
                    return state.fail(nvdec_error("NVDEC decoder was not created"));
                };
                let Some(geometry) = state.geometry else {
                    return state.fail(nvdec_error("NVDEC sequence was not parsed"));
                };
                let mut ptr = 0_u64;
                let mut pitch = 0_u32;
                let mut params = cuviddec::CUVIDPROCPARAMS {
                    progressive_frame: display.progressive_frame,
                    top_field_first: display.top_field_first,
                    second_field: 0,
                    output_stream: state.stream,
                    ..Default::default()
                };
                if let Err(err) = cu_result(
                    (state.api.map_frame)(
                        decoder.handle,
                        display.picture_index,
                        &raw mut ptr,
                        &raw mut pitch,
                        &raw mut params,
                    ),
                    "cuvidMapVideoFrame64 failed",
                ) {
                    return state.fail(err);
                }
                let frame = MappedCuvidFrame {
                    decoder,
                    ptr,
                    pitch,
                    geometry,
                };
                let pts = display.timestamp;
                let duration = state.durations.get(&pts).copied().unwrap_or(1).max(1);
                // The frame guard is dropped before `state` is used again, so
                // the surface is unmapped even when the consumer fails.
                let result = (state.on_frame)(frame, pts, duration);
                return match result {
                    Ok(()) => {
                        state.displayed_frames = state.displayed_frames.saturating_add(1);
                        1
                    }
                    Err(err) => state.fail(err),
                };
            }
            0
        }
    }

    /// Validates that this host can decode the given codec/bit-depth pair,
    /// independent of any source file. Used by real-GPU tests and by the
    /// runtime pipeline before creating a session.
    pub(crate) fn validate_decode_caps(
        codec: VideoCodec,
        bit_depth: u8,
        width: u32,
        height: u32,
    ) -> Result<(), PipelineError> {
        check_caps(codec_type(codec)?, bit_depth, width, height).map(|_| ())
    }

    /// Byte ranges of one Annex-B NAL unit inside the original packet.
    struct NalRange {
        start: usize,
        payload_start: usize,
        end: usize,
    }

    fn annex_b_nal_ranges(data: &[u8]) -> Vec<NalRange> {
        let mut starts = Vec::new();
        let mut idx = 0_usize;
        while idx.saturating_add(2) < data.len() {
            if data[idx] == 0 && data[idx + 1] == 0 && data[idx + 2] == 1 {
                starts.push((idx, idx + 3));
                idx = idx.saturating_add(3);
            } else {
                idx = idx.saturating_add(1);
            }
        }
        let mut ranges = Vec::with_capacity(starts.len());
        for (position, &(start, payload_start)) in starts.iter().enumerate() {
            let next = starts
                .get(position + 1)
                .map_or(data.len(), |&(next_start, _)| next_start);
            let mut end = next;
            while end > payload_start && data[end - 1] == 0 {
                end -= 1;
            }
            if end > payload_start {
                ranges.push(NalRange {
                    start,
                    payload_start,
                    end,
                });
            }
        }
        ranges
    }

    /// MSB of the first RBSP byte, skipping emulation-prevention bytes. Both
    /// H.264 `first_mb_in_slice` and HEVC `first_slice_segment_in_pic_flag` are
    /// the first Exp-Golomb / u(1) value, so a set MSB marks a picture start.
    fn rbsp_starts_with_one(nal: &[u8], header_len: usize) -> bool {
        let mut zeros = 0_u8;
        for &byte in nal.iter().skip(header_len) {
            if zeros >= 2 && byte == 0x03 {
                zeros = 0;
                continue;
            }
            return byte & 0x80 != 0;
        }
        false
    }

    /// `(is_vcl, starts_access_unit, starts_picture)` for one NAL unit.
    fn nal_role(nal: &[u8], codec: VideoCodec) -> (bool, bool, bool) {
        match codec {
            VideoCodec::H264 => {
                let kind = nal.first().copied().unwrap_or(0) & 0x1F;
                let first = rbsp_starts_with_one(nal, 1);
                let vcl = (1..=5).contains(&kind);
                (vcl, matches!(kind, 6..=9) || (vcl && first), vcl && first)
            }
            VideoCodec::Hevc => {
                let kind = nal
                    .first()
                    .copied()
                    .map_or(0, |header| (header >> 1) & 0x3F);
                let first = rbsp_starts_with_one(nal, 2);
                let vcl = kind <= 31;
                (
                    vcl,
                    matches!(kind, 32..=35 | 39) || (vcl && first),
                    vcl && first,
                )
            }
            _ => (false, false, false),
        }
    }

    /// Splits Annex-B H.264/HEVC into access units. A unit is marked complete
    /// only when it contains the first slice of a picture; that is the only
    /// case where `CUVID_PKT_ENDOFPICTURE` is legal for the submitted packet.
    fn annex_b_access_units(data: &[u8], codec: VideoCodec) -> Vec<(&[u8], bool)> {
        let mut units = Vec::new();
        let mut au_start = 0_usize;
        let mut au_end = 0_usize;
        let mut has_picture_start = false;
        let mut has_vcl = false;
        for range in annex_b_nal_ranges(data) {
            let nal = &data[range.payload_start..range.end];
            let (is_vcl, starts_au, starts_picture) = nal_role(nal, codec);
            if starts_au && has_vcl {
                units.push((&data[au_start..au_end], has_picture_start));
                au_start = range.start;
                has_picture_start = false;
                has_vcl = false;
            }
            au_end = range.end;
            has_picture_start |= starts_picture;
            has_vcl |= is_vcl;
        }
        if au_end > au_start {
            units.push((&data[au_start..au_end], has_picture_start));
        }
        units
    }

    /// Decodes `packets` with NVDEC and invokes `on_frame` once per displayed
    /// frame while its GPU surface is still mapped. Copying into NVENC here
    /// keeps the frame resident in VRAM: no CUDA host staging and no CPU decode.
    // Setup, packet conversion, EOS handling and teardown stay in one place so
    // no RAII guard can be dropped out of order.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn decode_packets_to_gpu<F>(
        cuda_ctx: &CudaContext,
        stream: &CudaStream,
        codec: VideoCodec,
        config: &[u8],
        packets: &[ContainerPacket],
        mut on_frame: F,
    ) -> Result<u64, PipelineError>
    where
        F: FnMut(MappedCuvidFrame<'_>, i64, u64) -> Result<(), PipelineError>,
    {
        let api = api()?;
        let kind = codec_type(codec)?;
        let first = packets
            .first()
            .ok_or_else(|| nvdec_error("no packets to decode"))?;
        let mut durations = HashMap::with_capacity(packets.len());
        for packet in packets {
            durations.insert(packet.pts, packet.duration.max(1));
        }

        let mut lock = ptr::null_mut();
        // SAFETY: the shared CUDA context handle outlives `ctx_lock`.
        #[allow(unsafe_code)]
        let status = unsafe { (api.lock_create)(&raw mut lock, cuda_ctx.cu_ctx()) };
        cu_result(status, "cuvidCtxLockCreate failed")?;
        let ctx_lock = CuvidCtxLock { api, handle: lock };

        let mut state = CallbackState {
            api,
            codec: kind,
            source: codec,
            ctx_lock: ctx_lock.handle,
            stream: stream.cu_stream(),
            decoder: None,
            geometry: None,
            error: None,
            durations,
            displayed_frames: 0,
            on_frame: &mut on_frame,
        };
        let mut parser = ptr::null_mut();
        let mut parser_params = nvcuvid::CUVIDPARSERPARAMS {
            CodecType: kind,
            ulMaxNumDecodeSurfaces: 32,
            ulClockRate: US_CLOCK_RATE,
            ulErrorThreshold: 0,
            // Allow NVDEC to keep 4 decoded frames pipelined ahead of display
            // so the hardware decoder stays saturated alongside NVENC.
            ulMaxDisplayDelay: 4,
            pUserData: (&raw mut state).cast::<c_void>(),
            pfnSequenceCallback: Some(sequence_callback),
            pfnDecodePicture: Some(decode_callback),
            pfnDisplayPicture: Some(display_callback),
            ..Default::default()
        };
        // SAFETY: callbacks only touch `state`, which outlives the parser.
        #[allow(unsafe_code)]
        let status = unsafe { (api.create_parser)(&raw mut parser, &raw mut parser_params) };
        cu_result(status, "cuvidCreateVideoParser failed")?;
        let parser = CuvidParser {
            api,
            handle: parser,
        };

        for packet in packets {
            // NVDEC consumes Annex-B H.264/HEVC; the demuxer stores AVCC/HVCC
            // configuration. Convert each access unit before submitting it.
            let payload = match state.source {
                // MP4/MKV store length-prefixed AVCC/HVCC samples; MPEG-TS
                // already carries Annex-B. An absent config means the latter.
                VideoCodec::H264 if !config.is_empty() => {
                    crate::video_bitstream::avcc_to_annex_b(config, &packet.data, packet.keyframe)
                        .map_err(|err| nvdec_error(format!("AVCC to Annex-B failed: {err}")))?
                }
                VideoCodec::Hevc if !config.is_empty() => {
                    crate::video_bitstream::hvcc_to_annex_b(config, &packet.data, packet.keyframe)
                        .map_err(|err| nvdec_error(format!("HVCC to Annex-B failed: {err}")))?
                }
                VideoCodec::H264 | VideoCodec::Hevc | VideoCodec::Vp9 => packet.data.clone(),
                other => return Err(nvdec_error(format!("NVDEC cannot decode {other}"))),
            };
            // MPEG-TS reassembles a whole PES into one ContainerPacket, which
            // may hold several access units. Submit each one separately so
            // ENDOFPICTURE only marks a packet with exactly one complete
            // frame/field, as NVCUVID requires.
            let units: Vec<(&[u8], bool)> = match state.source {
                VideoCodec::H264 | VideoCodec::Hevc => {
                    let split = annex_b_access_units(&payload, state.source);
                    if split.is_empty() && !payload.is_empty() {
                        // Unparsable Annex-B cannot be proven complete; never
                        // claim an end-of-picture boundary for it.
                        vec![(payload.as_slice(), false)]
                    } else {
                        split
                    }
                }
                VideoCodec::Vp9 => vec![(payload.as_slice(), true)],
                other => return Err(nvdec_error(format!("NVDEC cannot decode {other}"))),
            };
            let step = packet.duration.max(1);
            let step_us = i64::try_from(step).unwrap_or(i64::MAX);
            for (index, (bytes, complete)) in units.into_iter().enumerate() {
                let offset = step_us.saturating_mul(i64::try_from(index).unwrap_or(i64::MAX));
                let timestamp = packet.pts.saturating_add(offset);
                state.durations.insert(timestamp, step);
                let mut flags = nvcuvid::CUvideopacketflags::CUVID_PKT_TIMESTAMP as u32;
                if complete {
                    flags |= nvcuvid::CUvideopacketflags::CUVID_PKT_ENDOFPICTURE as u32;
                }
                let mut data = nvcuvid::CUVIDSOURCEDATAPACKET {
                    flags: u64::from(flags),
                    payload_size: u64::try_from(bytes.len())
                        .map_err(|_| nvdec_error("NVDEC packet exceeds u64 size"))?,
                    payload: bytes.as_ptr(),
                    timestamp,
                };
                // SAFETY: `parser` outlives the call and `bytes` stays alive in
                // `payload` for the duration of this submission.
                #[allow(unsafe_code)]
                let status = unsafe { (api.parse_data)(parser.handle, &raw mut data) };
                // NVCUVID propagates callback failures, so surface the specific
                // cause recorded by the callback before the parser status.
                if let Some(err) = state.error.take() {
                    return Err(err);
                }
                cu_result(status, "cuvidParseVideoData failed")?;
            }
        }

        let mut eos = nvcuvid::CUVIDSOURCEDATAPACKET {
            flags: nvcuvid::CUvideopacketflags::CUVID_PKT_ENDOFSTREAM as u64
                | u64::from(nvcuvid::CUvideopacketflags::CUVID_PKT_TIMESTAMP as u32),
            payload_size: 0,
            payload: ptr::null(),
            timestamp: first.pts,
        };
        // SAFETY: the empty EOS packet is valid and the parser is still alive.
        #[allow(unsafe_code)]
        let status = unsafe { (api.parse_data)(parser.handle, &raw mut eos) };
        if let Some(err) = state.error.take() {
            return Err(err);
        }
        cu_result(status, "cuvidParseVideoData EOS failed")?;

        let frames = state.displayed_frames;
        // Explicit drop order: parser -> state/decoder -> ctx lock (while caller's
        // shared CUDA context remains alive).
        drop(parser);
        drop(state);
        drop(ctx_lock);
        Ok(frames)
    }
}

pub(crate) use enabled::{decode_packets_to_gpu, validate_decode_caps};

#[cfg(all(test, feature = "nvenc"))]
mod tests {
    use std::path::Path;

    use cudarc::driver::CudaContext;
    use fastconv_domain::{VideoCodec, VideoEncodeMode, VideoQuality};

    use super::{decode_packets_to_gpu, validate_decode_caps};
    use crate::PipelineError;
    use crate::gpu::nvenc::{encode_video_nvdec_nvenc_streaming, encode_video_nvenc_streaming};
    use crate::video_codecs::{Yuv420Frame, decode_video_track, encode_video_cpu};

    fn synthetic_frames(width: u32, height: u32, count: usize) -> Vec<Yuv420Frame> {
        let w = usize::try_from(width).unwrap_or(0);
        let h = usize::try_from(height).unwrap_or(0);
        (0..count)
            .map(|index| Yuv420Frame {
                width,
                height,
                y: vec![
                    u8::try_from(32_usize.saturating_add(index.saturating_mul(7) % 160))
                        .unwrap_or(32);
                    w.saturating_mul(h)
                ],
                u: vec![96; w.div_ceil(2).saturating_mul(h.div_ceil(2))],
                v: vec![128; w.div_ceil(2).saturating_mul(h.div_ceil(2))],
                pts_us: i64::try_from(index).unwrap_or(0).saturating_mul(40_000),
                duration_us: 40_000,
            })
            .collect()
    }

    #[test]
    fn nvdec_caps_check_covers_turing_codecs_and_depths() {
        // NVCUVID capability queries require a current CUDA context.
        let Ok(_ctx) = CudaContext::new(0) else {
            return;
        };
        // 8-bit H.264/HEVC/VP9 must be supported on this RTX 2060; 10-bit
        // HEVC/VP9 must get an explicit answer (supported or hard rejection),
        // never a silent CPU fallback.
        for (codec, depth, must_be_supported) in [
            (VideoCodec::H264, 8_u8, true),
            (VideoCodec::Hevc, 8, true),
            (VideoCodec::Hevc, 10, false),
            (VideoCodec::Vp9, 8, true),
            (VideoCodec::Vp9, 10, false),
        ] {
            let result = validate_decode_caps(codec, depth, 320, 240);
            if must_be_supported {
                assert!(
                    result.is_ok(),
                    "{codec} {depth}-bit must decode on RTX 2060: {result:?}"
                );
            } else {
                assert!(
                    result.is_ok() || matches!(result, Err(PipelineError::DecodeFailed { .. })),
                    "NVDEC caps must answer cleanly for {codec} {depth}-bit"
                );
            }
        }
        assert!(validate_decode_caps(VideoCodec::Av1, 8, 320, 240).is_err());
    }

    #[test]
    fn nvdec_decodes_h264_surfaces_with_preserved_timestamps() -> Result<(), PipelineError> {
        let Ok(ctx) = CudaContext::new(0) else {
            return Ok(());
        };
        let stream = ctx.default_stream();
        let path = Path::new("nvdec-h264-surface-test");
        let frames = synthetic_frames(320, 240, 8);
        let source = encode_video_cpu(
            path,
            &frames,
            VideoEncodeMode::CpuH264 {
                quality: VideoQuality::DEFAULT,
            },
            0,
        )?;
        // Assertions must live outside the `extern "C"` display callback: a
        // panic there aborts the whole test process instead of failing this
        // test, so record the timings and compare them after decoding.
        let mut displayed: Vec<(i64, u64)> = Vec::new();
        decode_packets_to_gpu(
            &ctx,
            &stream,
            VideoCodec::H264,
            &source.track.config,
            &source.packets,
            |_frame, pts_us, duration_us| {
                displayed.push((pts_us, duration_us));
                Ok(())
            },
        )?;
        let expected: Vec<(i64, u64)> = frames
            .iter()
            .map(|frame| (frame.pts_us, frame.duration_us))
            .collect();
        assert_eq!(
            displayed.len(),
            expected.len(),
            "every H.264 frame must be displayed once"
        );
        assert_eq!(
            displayed, expected,
            "NVDEC must preserve every distinct timestamp and 40,000 µs duration"
        );
        Ok(())
    }

    #[test]
    fn nvdec_to_nvenc_vp9_roundtrip_decodes_cleanly() -> Result<(), PipelineError> {
        if CudaContext::new(0).is_err() {
            return Ok(());
        }
        let path = Path::new("nvdec-nvenc-vp9-test");
        let frames = synthetic_frames(320, 240, 8);
        let source = encode_video_cpu(
            path,
            &frames,
            VideoEncodeMode::CpuVp9 {
                quality: VideoQuality::DEFAULT,
            },
            0,
        )?;
        let (stream, encoded, _) = encode_video_nvdec_nvenc_streaming(
            &source.track,
            &source.packets,
            VideoCodec::Vp9,
            VideoEncodeMode::NvencH264 {
                quality: VideoQuality::DEFAULT,
            },
            0,
            || Ok(()),
        )?;
        assert_eq!(encoded, frames.len() as u64);
        let decoded = decode_video_track(path, &stream.track, &stream.packets)?;
        assert_eq!(decoded.len(), frames.len());
        Ok(())
    }

    #[test]
    fn nvdec_to_nvenc_hevc_roundtrip_decodes_cleanly() -> Result<(), PipelineError> {
        if CudaContext::new(0).is_err() {
            return Ok(());
        }
        let path = Path::new("nvdec-nvenc-hevc-test");
        let frames = synthetic_frames(320, 240, 8);
        let h264 = encode_video_cpu(
            path,
            &frames,
            VideoEncodeMode::CpuH264 {
                quality: VideoQuality::DEFAULT,
            },
            0,
        )?;
        let (hevc, _, _) = encode_video_nvenc_streaming(
            path,
            &h264.track,
            &h264.packets,
            VideoEncodeMode::NvencHevc {
                quality: VideoQuality::DEFAULT,
            },
            0,
            || Ok(()),
        )?;
        let (stream, encoded, _) = encode_video_nvdec_nvenc_streaming(
            &hevc.track,
            &hevc.packets,
            VideoCodec::Hevc,
            VideoEncodeMode::NvencH264 {
                quality: VideoQuality::DEFAULT,
            },
            0,
            || Ok(()),
        )?;
        assert_eq!(encoded, frames.len() as u64);
        let decoded = decode_video_track(path, &stream.track, &stream.packets)?;
        assert_eq!(decoded.len(), frames.len());
        Ok(())
    }

    #[test]
    fn nvdec_rejects_av1_without_cpu_fallback() {
        let err = encode_video_nvdec_nvenc_streaming(
            &crate::video_container::ContainerTrack {
                codec: VideoCodec::Av1,
                width: Some(320),
                height: Some(240),
                channels: None,
                sample_rate: None,
                config: Vec::new(),
                timescale: crate::video_container::US_TIMESCALE,
                presentation_edit: None,
            },
            &[],
            VideoCodec::Av1,
            VideoEncodeMode::NvencH264 {
                quality: VideoQuality::DEFAULT,
            },
            0,
            || Ok(()),
        );
        assert!(
            matches!(err, Err(PipelineError::GpuBackendFailed { .. })),
            "forced NVDEC with AV1 must be a hard error: {err:?}"
        );
    }

    #[test]
    fn nvdec_to_nvenc_zero_copy_roundtrip_decodes_cleanly() -> Result<(), PipelineError> {
        if CudaContext::new(0).is_err() {
            return Ok(());
        }
        let path = Path::new("nvdec-nvenc-zero-copy-test");
        let frames = synthetic_frames(320, 240, 10);
        let source = encode_video_cpu(
            path,
            &frames,
            VideoEncodeMode::CpuH264 {
                quality: VideoQuality::DEFAULT,
            },
            0,
        )?;
        let mut callbacks = 0_u32;
        let (stream, encoded, device) = encode_video_nvdec_nvenc_streaming(
            &source.track,
            &source.packets,
            VideoCodec::H264,
            VideoEncodeMode::NvencH264 {
                quality: VideoQuality::DEFAULT,
            },
            0,
            || {
                callbacks = callbacks.saturating_add(1);
                Ok(())
            },
        )?;
        assert_eq!(encoded, frames.len() as u64);
        assert_eq!(u64::from(callbacks), encoded);
        assert!(!device.is_empty(), "real NVIDIA device must be reported");
        let decoded = decode_video_track(path, &stream.track, &stream.packets)?;
        assert_eq!(decoded.len(), frames.len(), "zero-copy output must decode");
        let mut previous = None;
        for packet in &stream.packets {
            assert!(
                packet.dts <= packet.pts,
                "NVDEC -> NVENC DTS {} exceeds PTS {}",
                packet.dts,
                packet.pts
            );
            if let Some(prev) = previous {
                assert!(
                    packet.dts > prev,
                    "NVDEC -> NVENC DTS must strictly increase"
                );
            }
            previous = Some(packet.dts);
        }
        Ok(())
    }
}
