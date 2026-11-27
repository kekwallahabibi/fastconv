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
    use std::cell::{Cell, RefCell};
    use std::collections::{HashMap, VecDeque};
    use std::ffi::{c_int, c_uint, c_ulonglong, c_void};
    use std::ptr;
    use std::sync::Arc;
    use std::sync::OnceLock;

    use cudarc::driver::{CudaContext, CudaEvent, CudaStream, sys};
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
        // libloading 0.9 accepts a byte slice directly and null-terminates it
        // internally; a `&Vec<u8>` no longer implements `AsSymbolName`.
        unsafe {
            library.get::<T>(name).map(|symbol| *symbol).map_err(|err| {
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

    /// Bounds the number of NVDEC surfaces whose device-to-device copy is still
    /// queued on the CUDA stream. A CUDA event is recorded immediately after
    /// each copy; the oldest surface is unmapped once the ring fills and every
    /// remaining surface is unmapped on `Drop`, which runs before the decoder
    /// is destroyed by the enclosing `CallbackState`.
    ///
    /// Events are recycled so the steady state performs no per-frame CUDA
    /// object allocation. This replaces a blocking `cuStreamSynchronize` on
    /// every frame while keeping unmaps race-free.
    struct UnmapTracker {
        api: &'static CuvidApi,
        cuda_ctx: Arc<CudaContext>,
        decoder: Cell<cuviddec::CUvideodecoder>,
        pending: VecDeque<(sys::CUdeviceptr, CudaEvent)>,
        free_events: Vec<CudaEvent>,
        max_in_flight: usize,
    }

    impl UnmapTracker {
        fn new(api: &'static CuvidApi, cuda_ctx: &Arc<CudaContext>) -> Self {
            Self {
                api,
                cuda_ctx: Arc::clone(cuda_ctx),
                decoder: Cell::new(ptr::null_mut()),
                pending: VecDeque::new(),
                free_events: Vec::new(),
                // Well below `ulNumOutputSurfaces` (8), so NVCUVID always has
                // an unmapped output surface available for the next frame.
                max_in_flight: 4,
            }
        }

        fn set_decoder(&self, decoder: cuviddec::CUvideodecoder) {
            self.decoder.set(decoder);
        }

        /// Records `stream` into an event and queues `ptr` for deferred unmap.
        /// The ring is drained *before* taking ownership of `ptr`, so a failure
        /// never leaves the caller's surface in two places at once.
        #[allow(unsafe_code)]
        fn defer(
            &mut self,
            ptr: sys::CUdeviceptr,
            stream: &CudaStream,
        ) -> Result<(), PipelineError> {
            while self.pending.len() >= self.max_in_flight {
                self.release_oldest()?;
            }
            let event = match self.free_events.pop() {
                Some(event) => event,
                None => self
                    .cuda_ctx
                    .new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
                    .map_err(|err| nvdec_error(format!("cuEventCreate failed: {err}")))?,
            };
            if let Err(err) = event.record(stream) {
                // Return the event to the pool before propagating so teardown
                // does not leak a CUDA event on this hard error path.
                self.free_events.push(event);
                return Err(nvdec_error(format!("cuEventRecord failed: {err}")));
            }
            self.pending.push_back((ptr, event));
            Ok(())
        }

        /// Waits for one queued copy, then releases its surface only once the
        /// wait confirms completion. The event is recycled on every path. A
        /// failed wait leaves the surface owned by the decoder (which destroys
        /// it at teardown) rather than unmapping memory the device may still read.
        #[allow(unsafe_code)]
        fn finish_one(
            &mut self,
            ptr: sys::CUdeviceptr,
            event: CudaEvent,
        ) -> Result<(), PipelineError> {
            let sync = event
                .synchronize()
                .map_err(|err| nvdec_error(format!("cuEventSynchronize failed: {err}")));
            self.free_events.push(event);
            sync?;
            // SAFETY: the event proves the queued copy finished, and the decoder
            // outlives this tracker, so the mapping is still valid here.
            cu_result(
                unsafe { (self.api.unmap_frame)(self.decoder.get(), ptr) },
                "cuvidUnmapVideoFrame64 failed",
            )
        }

        /// Synchronises the oldest copy and releases its surface.
        fn release_oldest(&mut self) -> Result<(), PipelineError> {
            let Some((ptr, event)) = self.pending.pop_front() else {
                return Ok(());
            };
            self.finish_one(ptr, event)
        }

        /// Drains every deferred copy, returning the first failure only after
        /// all confirmed surfaces are released. Kept separate from `Drop` so a
        /// completion error can still reach the caller on the success path.
        fn drain_all(&mut self) -> Result<(), PipelineError> {
            let mut first_error = None;
            while let Some((ptr, event)) = self.pending.pop_front() {
                if let Err(err) = self.finish_one(ptr, event)
                    && first_error.is_none()
                {
                    first_error = Some(err);
                }
            }
            match first_error {
                Some(err) => Err(err),
                None => Ok(()),
            }
        }
    }

    impl Drop for UnmapTracker {
        fn drop(&mut self) {
            if self.decoder.get().is_null() {
                return;
            }
            // Every remaining surface must be released before the decoder is
            // destroyed (field order in `CallbackState` guarantees this).
            // Errors cannot be propagated from `Drop`; never panic here.
            let _drained = self.drain_all();
        }
    }

    /// One decode surface mapped for the duration of this guard. `Drop` always
    /// releases it, either immediately (before a copy could be queued) or via
    /// the [`UnmapTracker`] once the queued device-to-device copy has completed.
    pub(crate) struct MappedCuvidFrame<'a> {
        decoder: &'a CuvidDecoder,
        tracker: &'a RefCell<UnmapTracker>,
        ptr: sys::CUdeviceptr,
        pitch: u32,
        geometry: Geometry,
        deferred: Cell<bool>,
    }

    impl MappedCuvidFrame<'_> {
        /// Device-to-device copy of the NV12 surface into a registered NVENC
        /// slot. When the row pitch and geometry match and the width is even,
        /// luma and chroma are contiguous with identical strides in both
        /// allocations, so one `cuMemcpy2DAsync_v2` replaces two submissions.
        /// Otherwise the planes are copied separately so crop offsets, odd
        /// dimensions and differing pitches never touch rows outside the
        /// destination allocation.
        ///
        /// The copy is asynchronous: a CUDA event is recorded after it and the
        /// source surface is handed to the tracker. NVENC reads the destination
        /// on the same stream (bound via `NvEncSetIOCudaStreams`), so no
        /// per-frame host synchronize is required.
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
            let luma_bytes = usize::try_from(g.display_width).unwrap_or(0);
            let luma_height = usize::try_from(g.display_height).unwrap_or(0);
            let chroma_height = usize::try_from(g.display_height.div_ceil(2)).unwrap_or(0);
            let src_chroma = self.ptr.saturating_add(
                u64::try_from(src_pitch.saturating_mul(luma_height)).unwrap_or(u64::MAX),
            );
            let dst_chroma = dst.saturating_add(
                u64::try_from(dst_pitch.saturating_mul(luma_height)).unwrap_or(u64::MAX),
            );
            // An even width makes `chroma_bytes == luma_bytes`; with equal
            // pitches the chroma plane starts exactly after `luma_height`
            // rows, so a single rectangle reproduces both separate copies.
            let coalesce = src_pitch == dst_pitch
                && luma_height == usize::try_from(dst_height).unwrap_or(0)
                && luma_bytes == chroma_bytes
                && luma_bytes != 0
                && chroma_height != 0;
            let rects: [(sys::CUdeviceptr, sys::CUdeviceptr, usize, usize); 2] = if coalesce {
                [
                    (self.ptr, dst, luma_bytes, luma_height + chroma_height),
                    (0, 0, 0, 0),
                ]
            } else {
                [
                    (self.ptr, dst, luma_bytes, luma_height),
                    (src_chroma, dst_chroma, chroma_bytes, chroma_height),
                ]
            };
            let api = self.decoder.api;
            let mut copy_accepted = false;
            for (src, dst_ptr, width_bytes, height) in rects {
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
            if !copy_accepted {
                return Err(nvdec_error("NVDEC produced an empty copy rectangle"));
            }
            // Record after every queued copy and defer the surface release;
            // the tracker synchronises the event before unmapping.
            {
                let mut tracker = self.tracker.try_borrow_mut().map_err(|_| {
                    nvdec_error("NVDEC unmap tracker was already borrowed during frame handoff")
                })?;
                if let Err(err) = tracker.defer(self.ptr, stream) {
                    drop(tracker);
                    // The copy is still queued on this stream; drain it so the
                    // guard's immediate unmap cannot race with the device read.
                    let drained = stream.synchronize().map_err(|sync| {
                        nvdec_error(format!(
                            "CUDA stream synchronize after failed NVDEC surface handoff: {sync}"
                        ))
                    });
                    return Err(match drained {
                        Err(sync_err) => nvdec_error(format!("{err}; {sync_err}")),
                        Ok(()) => err,
                    });
                }
            }
            self.deferred.set(true);
            Ok(())
        }
    }

    impl Drop for MappedCuvidFrame<'_> {
        #[allow(unsafe_code)]
        fn drop(&mut self) {
            if self.deferred.get() {
                return;
            }
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
        // Declared before `decoder` so its `Drop` releases every deferred
        // surface while the decoder is still alive.
        unmap_tracker: RefCell<UnmapTracker>,
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
                        // Eight mapped-output slots leave headroom for the
                        // four-in-flight deferred-unmap ring.
                        ulNumOutputSurfaces: 8,
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
                    state.unmap_tracker.borrow().set_decoder(decoder);
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
                    tracker: &state.unmap_tracker,
                    ptr,
                    pitch,
                    geometry,
                    deferred: Cell::new(false),
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
            } else if data[idx + 2] > 1 {
                // A start code needs `00 00 01` at some offset j. When
                // `data[idx + 2] > 1`, no start code can begin at idx, idx+1
                // or idx+2; skip all three without changing the matches.
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
    /// Returns `(bytes, has_picture_start, has_vcl)` per unit. The VCL flag
    /// matters for timing: header-only tails are submitted but must not be
    /// counted as pictures when the PES interval is divided.
    pub(super) fn annex_b_access_units(data: &[u8], codec: VideoCodec) -> Vec<(&[u8], bool, bool)> {
        let mut units = Vec::new();
        let mut au_start = 0_usize;
        let mut au_end = 0_usize;
        let mut has_picture_start = false;
        let mut has_vcl = false;
        for range in annex_b_nal_ranges(data) {
            let nal = &data[range.payload_start..range.end];
            let (is_vcl, starts_au, starts_picture) = nal_role(nal, codec);
            if starts_au && has_vcl {
                units.push((&data[au_start..au_end], has_picture_start, has_vcl));
                au_start = range.start;
                has_picture_start = false;
                has_vcl = false;
            }
            au_end = range.end;
            has_picture_start |= starts_picture;
            has_vcl |= is_vcl;
        }
        if au_end > au_start {
            units.push((&data[au_start..au_end], has_picture_start, has_vcl));
        }
        units
    }

    /// Same access-unit grouping as [`annex_b_access_units`] but driven by the
    /// NALU ranges produced while emitting a length-prefixed sample, so the
    /// Annex-B buffer is never rescanned for start codes.
    fn access_units_from_ranges<'a>(
        data: &'a [u8],
        ranges: &[(usize, usize)],
        codec: VideoCodec,
    ) -> Vec<(&'a [u8], bool, bool)> {
        let mut units = Vec::new();
        let mut au_start = 0_usize;
        let mut au_end = 0_usize;
        let mut has_picture_start = false;
        let mut has_vcl = false;
        for &(payload_start, end) in ranges {
            // Every emitted NALU is preceded by a four-byte start code.
            let start = payload_start.saturating_sub(4);
            let (is_vcl, starts_au, starts_picture) = nal_role(&data[payload_start..end], codec);
            if starts_au && has_vcl {
                units.push((&data[au_start..au_end], has_picture_start, has_vcl));
                au_start = start;
                has_picture_start = false;
                has_vcl = false;
            }
            au_end = end;
            has_picture_start |= starts_picture;
            has_vcl |= is_vcl;
        }
        if au_end > au_start {
            units.push((&data[au_start..au_end], has_picture_start, has_vcl));
        }
        units
    }

    /// Splits one PES presentation interval evenly across its access units.
    ///
    /// MPEG-TS reassembles a PES carrying several pictures into one packet with
    /// a single timestamp. Stamping every picture with the whole interval pushes
    /// the last picture onto the next PES's timestamp; dividing instead keeps
    /// every picture distinct and ordered. A one-AU packet keeps its interval.
    pub(super) fn au_timing(pts: i64, duration: u64, au_count: usize, index: usize) -> (i64, u64) {
        let divisor = u64::try_from(au_count).unwrap_or(u64::MAX).max(1);
        let per_au = (duration / divisor).max(1);
        let step_us = i64::try_from(per_au).unwrap_or(i64::MAX);
        let offset = step_us.saturating_mul(i64::try_from(index).unwrap_or(i64::MAX));
        (pts.saturating_add(offset), per_au)
    }

    /// How packet payloads reach NVCUVID: raw Annex-B (MPEG-TS) or one of the
    /// length-prefixed MP4/MKV formats.
    enum PacketFormat {
        AnnexB,
        Avcc,
        Hvcc,
    }

    /// Decodes `packets` with NVDEC and invokes `on_frame` once per displayed
    /// frame while its GPU surface is still mapped. Copying into NVENC here
    /// keeps the frame resident in VRAM: no CUDA host staging and no CPU decode.
    // Setup, packet conversion, EOS handling and teardown stay in one place so
    // no RAII guard can be dropped out of order.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn decode_packets_to_gpu<F>(
        cuda_ctx: &Arc<CudaContext>,
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
            unmap_tracker: RefCell::new(UnmapTracker::new(api, cuda_ctx)),
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

        // avcC/hvcC parameter sets never change within a track. Parse them once
        // and reuse both the Annex-B scratch buffer and the NALU range list
        // across packets instead of reparsing and reallocating per frame.
        let format = match state.source {
            VideoCodec::H264 if !config.is_empty() => PacketFormat::Avcc,
            VideoCodec::Hevc if !config.is_empty() => PacketFormat::Hvcc,
            VideoCodec::H264 | VideoCodec::Hevc | VideoCodec::Vp9 => PacketFormat::AnnexB,
            other => return Err(nvdec_error(format!("NVDEC cannot decode {other}"))),
        };
        let prepared = match format {
            PacketFormat::Avcc => Some(
                crate::video_bitstream::LengthPrefixedConfig::avcc(config)
                    .map_err(|err| nvdec_error(format!("invalid avcC config: {err}")))?,
            ),
            PacketFormat::Hvcc => Some(
                crate::video_bitstream::LengthPrefixedConfig::hvcc(config)
                    .map_err(|err| nvdec_error(format!("invalid hvcC config: {err}")))?,
            ),
            PacketFormat::AnnexB => None,
        };
        let mut annex_b_scratch = Vec::with_capacity(64 * 1024);
        let mut nalu_ranges: Vec<(usize, usize)> = Vec::with_capacity(64);

        for packet in packets {
            // MPEG-TS reassembles a whole PES into one ContainerPacket, which
            // may hold several access units. Submit each one separately so
            // ENDOFPICTURE only marks a packet with exactly one complete
            // frame/field, as NVCUVID requires. Length-prefixed samples emit
            // their NALU ranges as they are written, so no start-code rescan
            // is necessary for MP4/MKV inputs.
            let units: Vec<(&[u8], bool, bool)> = if let Some(config) = prepared.as_ref() {
                annex_b_scratch.clear();
                nalu_ranges.clear();
                crate::video_bitstream::length_prefixed_to_annex_b_into(
                    config,
                    &packet.data,
                    packet.keyframe,
                    &mut annex_b_scratch,
                    &mut nalu_ranges,
                )
                .map_err(|err| nvdec_error(format!("length-prefixed sample conversion: {err}")))?;
                match state.source {
                    VideoCodec::H264 | VideoCodec::Hevc => {
                        access_units_from_ranges(&annex_b_scratch, &nalu_ranges, state.source)
                    }
                    VideoCodec::Vp9 => vec![(annex_b_scratch.as_slice(), true, true)],
                    other => return Err(nvdec_error(format!("NVDEC cannot decode {other}"))),
                }
            } else {
                match state.source {
                    VideoCodec::H264 | VideoCodec::Hevc => {
                        let split = annex_b_access_units(&packet.data, state.source);
                        if split.is_empty() && !packet.data.is_empty() {
                            // Unparsable Annex-B cannot be proven complete;
                            // never claim an end-of-picture boundary for it.
                            vec![(packet.data.as_slice(), false, false)]
                        } else {
                            split
                        }
                    }
                    VideoCodec::Vp9 => vec![(packet.data.as_slice(), true, true)],
                    other => return Err(nvdec_error(format!("NVDEC cannot decode {other}"))),
                }
            };
            // Only units carrying a VCL NAL produce a picture; a trailing AUD,
            // SPS or PPS is still submitted but must not inflate the divisor.
            let au_count = units
                .iter()
                .filter(|(_, _, has_vcl)| *has_vcl)
                .count()
                .max(1);
            let mut picture_index = 0_usize;
            for (bytes, complete, has_vcl) in units {
                let (timestamp, step) =
                    au_timing(packet.pts, packet.duration, au_count, picture_index);
                if has_vcl {
                    picture_index = picture_index.saturating_add(1);
                    // Header-only units still need a submission timestamp, but
                    // registering a duration for them would overwrite the
                    // duration of the next picture at the same PTS.
                    state.durations.insert(timestamp, step);
                }
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
        // A copy can queue successfully and fail only when its event is awaited;
        // `Drop` has no error channel, so drain explicitly before reporting
        // success. Remaining surfaces are still released by `Drop` afterwards.
        let drained = state.unmap_tracker.borrow_mut().drain_all();
        // Explicit drop order: parser -> state/decoder -> ctx lock (while caller's
        // shared CUDA context remains alive).
        drop(parser);
        drop(state);
        drop(ctx_lock);
        drained?;
        Ok(frames)
    }
}

pub(crate) use enabled::{decode_packets_to_gpu, validate_decode_caps};

#[cfg(all(test, feature = "nvenc"))]
mod tests {
    use std::path::Path;

    use cudarc::driver::CudaContext;
    use fastconv_domain::{VideoCodec, VideoEncodeMode, VideoRateControl};

    use super::enabled::{annex_b_access_units, au_timing};
    use super::{decode_packets_to_gpu, validate_decode_caps};
    use crate::PipelineError;
    use crate::gpu::nvenc::{encode_video_nvdec_nvenc_streaming, encode_video_nvenc_streaming};
    use crate::video_codecs::{Yuv420Frame, YuvColor, decode_video_track, encode_video_cpu};

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
                color: YuvColor::default(),
            })
            .collect()
    }

    #[test]
    fn two_access_units_share_the_pes_interval_without_duplicate_pts() {
        // 25 fps with two pictures per PES: consecutive PES timestamps are
        // 80 ms apart. Each PES's pictures must land 40 ms apart so the last
        // picture does not collide with the next PES's first picture.
        let first: Vec<(i64, u64)> = (0..2).map(|index| au_timing(0, 80_000, 2, index)).collect();
        assert_eq!(first, [(0_i64, 40_000_u64), (40_000, 40_000)]);
        let second: Vec<(i64, u64)> = (0..2)
            .map(|index| au_timing(80_000, 80_000, 2, index))
            .collect();
        assert_eq!(second, [(80_000, 40_000), (120_000, 40_000)]);
        // A packet holding a single AU keeps its timestamp and duration exactly.
        assert_eq!(au_timing(80_000, 40_000, 1, 0), (80_000, 40_000));
    }

    #[test]
    fn header_only_tail_does_not_inflate_the_picture_timing_divisor() {
        // AUD, first IDR slice, a later slice of the same picture, trailing AUD.
        let annex_b = [
            0, 0, 0, 1, 0x09, 0xF0, //
            0, 0, 0, 1, 0x65, 0x80, //
            0, 0, 0, 1, 0x65, 0x40, //
            0, 0, 0, 1, 0x09, 0xF0, //
        ];
        let units = annex_b_access_units(&annex_b, VideoCodec::H264);
        assert_eq!(units.len(), 2);
        assert_eq!(units.iter().filter(|(_, _, has_vcl)| *has_vcl).count(), 1);
        assert!(!units[1].1, "a header-only tail cannot end a picture");
        // The single picture gets the full 80 ms interval, not 80/3 ms.
        assert_eq!(au_timing(0, 80_000, 1, 0), (0, 80_000));
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
                rate_control: VideoRateControl::DEFAULT,
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
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
        )?;
        let (stream, encoded, _) = encode_video_nvdec_nvenc_streaming(
            &source.track,
            &source.packets,
            VideoCodec::Vp9,
            VideoEncodeMode::NvencH264 {
                rate_control: VideoRateControl::DEFAULT,
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
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
        )?;
        let (hevc, _, _) = encode_video_nvenc_streaming(
            path,
            &h264.track,
            &h264.packets,
            VideoEncodeMode::NvencHevc {
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
            || Ok(()),
        )?;
        let (stream, encoded, _) = encode_video_nvdec_nvenc_streaming(
            &hevc.track,
            &hevc.packets,
            VideoCodec::Hevc,
            VideoEncodeMode::NvencH264 {
                rate_control: VideoRateControl::DEFAULT,
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
                pcm_format: None,
            },
            &[],
            VideoCodec::Av1,
            VideoEncodeMode::NvencH264 {
                rate_control: VideoRateControl::DEFAULT,
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
                rate_control: VideoRateControl::DEFAULT,
            },
            0,
        )?;
        let mut callbacks = 0_u32;
        let (stream, encoded, device) = encode_video_nvdec_nvenc_streaming(
            &source.track,
            &source.packets,
            VideoCodec::H264,
            VideoEncodeMode::NvencH264 {
                rate_control: VideoRateControl::DEFAULT,
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
