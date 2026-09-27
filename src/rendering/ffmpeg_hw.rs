// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2021-2022 Adrian <adrian.eddy at gmail>

use ffmpeg_next::{ ffi, format, codec, encoder };

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ffi::{ CStr, CString };
use std::ptr;
use std::sync::LazyLock;
use parking_lot::Mutex;

type DeviceType = ffi::AVHWDeviceType;

// FFmpeg's Vulkan hwaccel is opt-in - never let FFmpeg create a Vulkan device unless the user explicitly enabled it
pub fn is_vulkan_enabled() -> bool { gyroflow_core::settings::get_bool("useVulkanEncoder", false) }

#[derive(Debug)]
pub struct HWDevice {
    type_: DeviceType,
    device_ref: *mut ffi::AVBufferRef,
    device_name: Option<String>,

    pub hw_formats: Vec<format::Pixel>,
    pub sw_formats: Vec<format::Pixel>,
    pub min_size: (i32, i32),
    pub max_size: (i32, i32)
}
impl HWDevice {
    pub fn from_type(type_: DeviceType, device_name: Option<&str>) -> Result<Self, super::FFmpegError> {
        log::debug!("HWDevice::from_type {type_:?}, device: {device_name:?}");
        if type_ == ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN && !is_vulkan_enabled() {
            log::debug!("Vulkan hwaccel is disabled");
            return Err(super::FFmpegError::CannotCreateGPUDecoding);
        }
        unsafe {
            let dev = device_name.and_then(|x| if x.is_empty() { None } else { CString::new(x).ok() });

            let mut device_ref = ptr::null_mut();
            let err = ffi::av_hwdevice_ctx_create(&mut device_ref, type_, dev.as_ref().map_or(ptr::null(), |x| x.as_ptr()), ptr::null_mut(), 0);
            if err >= 0 && !device_ref.is_null() {
                Ok(Self {
                    type_,
                    device_name: device_name.map(|x| x.to_string()),
                    device_ref,
                    hw_formats: Vec::new(),
                    sw_formats: Vec::new(),
                    min_size: (0, 0),
                    max_size: (0, 0),
                })
            } else {
                super::append_log(&format!("Failed to create specified HW device: {:?}\n", type_));
                Err(super::FFmpegError::CannotCreateGPUDecoding)
            }
        }
    }

    pub fn add_ref(&self) -> *mut ffi::AVBufferRef {
        unsafe { ffi::av_buffer_ref(self.device_ref) }
    }
    pub fn as_mut_ptr(&self) -> *mut ffi::AVBufferRef { self.device_ref }
    pub fn device_type(&self) -> DeviceType { self.type_ }
    pub fn device_name(&self) -> Option<&str> { self.device_name.as_deref() }
    pub fn name(&self) -> String {
        unsafe {
            let name_ptr = ffi::av_hwdevice_get_type_name(self.type_);
            CStr::from_ptr(name_ptr).to_string_lossy().into()
        }
    }
}
impl Drop for HWDevice {
    fn drop(&mut self) {
        unsafe { ffi::av_buffer_unref(&mut self.device_ref); }
    }
}
unsafe impl Sync for HWDevice { }
unsafe impl Send for HWDevice { }

static DEVICES: LazyLock<Mutex<HashMap<u64, HWDevice>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn clear_devices() {
    DEVICES.lock().clear();
}
pub fn initialize_ctx(type_: ffi::AVHWDeviceType) {
    let mut devices = DEVICES.lock();
    if let Entry::Vacant(e) = devices.entry(type_ as u64) {
        ::log::debug!("create {:?}", type_);
        if let Ok(dev) = HWDevice::from_type(type_, None) {
            ::log::debug!("created ok {:?}", type_);
            e.insert(dev);
        }
    }
}

pub fn supported_gpu_backends() -> Vec<String> {
    let mut ret = Vec::new();
    let mut hw_type = ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE;
    for _ in 0..20 { // Better 20 than infinity
        unsafe {
            hw_type = ffi::av_hwdevice_iterate_types(hw_type);
            if hw_type == ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE {
                break;
            }
            // returns a pointer to static string, shouldn't be freed
            let name_ptr = ffi::av_hwdevice_get_type_name(hw_type);
            ret.push(CStr::from_ptr(name_ptr).to_string_lossy().into());
        }
    }
    ret
}

pub unsafe fn pix_formats_to_vec(formats: *const ffi::AVPixelFormat) -> Vec<format::Pixel> {
    if formats.is_null() {
        return Vec::new();
    }
    let mut ret = Vec::new();
    for i in 0..100 {
        let p = unsafe { *formats.offset(i) };
        if p == ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            break;
        }
        ret.push(p.into());
    }
    ret
}

pub unsafe fn codec_pix_formats(codec_ctx: *const ffi::AVCodecContext, codec: *const ffi::AVCodec) -> Vec<format::Pixel> {
    unsafe {
        if codec.is_null() && codec_ctx.is_null() {
            return Vec::new();
        }
        let mut formats: *const ffi::AVPixelFormat = ptr::null();
        let mut count: std::os::raw::c_int = 0;
        let ret = ffi::avcodec_get_supported_config(codec_ctx, codec, ffi::AVCodecConfig::AV_CODEC_CONFIG_PIX_FORMAT, 0, (&mut formats as *mut *const ffi::AVPixelFormat).cast(), &mut count);
        if ret < 0 || formats.is_null() || count <= 0 {
            return Vec::new();
        }
        std::slice::from_raw_parts(formats, count as usize).iter().map(|p| (*p).into()).collect()
    }
}

pub fn init_device_for_decoding(index: usize, codec: *const ffi::AVCodec, decoder_ctx: &mut codec::context::Context, device: Option<&str>) -> Result<(usize, ffi::AVHWDeviceType, String, Option<ffi::AVPixelFormat>), super::FFmpegError> {
    let mut decoders = Vec::<(usize, ffi::AVHWDeviceType)>::new();

    for i in 0..20 {
        unsafe {
            let config = ffi::avcodec_get_hw_config(codec, i as i32);
            if !config.is_null() && (*config).device_type != ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE {
                decoders.push((i, (*config).device_type));
            }
        }
    }
    if let Some(pos) = decoders.iter().position(|(_, type_)| *type_ == ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA) {
        // Move CUDA to first position
        let cuda = decoders.remove(pos);
        decoders.insert(0, cuda);
    }
    if is_vulkan_enabled() {
        if let Some(pos) = decoders.iter().position(|(_, type_)| *type_ == ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN) {
            let x = decoders.remove(pos);
            decoders.insert(0, x);
        }
    } else {
        decoders.retain(|(_, type_)| *type_ != ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN);
    }
    if gyroflow_core::settings::get_bool("useD3D12Encoder", false) {
        if let Some(pos) = decoders.iter().position(|(_, type_)| *type_ == ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D12VA) {
            let x = decoders.remove(pos);
            decoders.insert(0, x);
        }
    }

    log::debug!("Available decoders: {decoders:?}");

    for (i, _) in decoders.iter().skip(index) {
        unsafe {
            let config = ffi::avcodec_get_hw_config(codec, *i as i32);
            if config.is_null() {
                ::log::debug!("config null for {}", i);
                continue;
            }
            let type_ = (*config).device_type;
            if type_ == ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE {
                continue;
            }
            if cfg!(target_os = "windows") && type_ == ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI {
                continue;
            }
            ::log::debug!("[dec] codec type {:?} {}", type_, i);
            let mut devices = DEVICES.lock();
            let mut device_hash = 0;
            if let Some(dev_name) = device {
                let mut hasher = crc32fast::Hasher::new();
                hasher.update(dev_name.as_bytes());
                device_hash = hasher.finalize() as u64;
            }
            if let Entry::Vacant(e) = devices.entry(type_ as u64 + device_hash) {
                if let Ok(dev) = HWDevice::from_type(type_, device) {
                    e.insert(dev);
                }
            }
            if let Some(dev) = devices.get(&(type_ as u64 + device_hash)) {
                (*decoder_ctx.as_mut_ptr()).hw_device_ctx = dev.add_ref();
                return Ok((*i, type_, dev.name(), Some((*config).pix_fmt)));
            }
        }
    }
    Ok((0, ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE, String::new(), None))
}

pub fn find_working_encoder(encoders: &[(&'static str, bool)], device: Option<&str>) -> (&'static str, bool, Option<DeviceType>) {
    if encoders.is_empty() { return ("", false, None); } // TODO: should be Result<>

    let mut device_hash = 0;
    if let Some(dev_name) = device {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(dev_name.as_bytes());
        device_hash = hasher.finalize() as u64;
    }

    for x in encoders {
        if let Some(enc) = encoder::find_by_name(x.0) {
            if !x.1 { return (x.0, x.1, None); } // If not HW encoder

            for i in 0..20 {
                unsafe {
                    let type_ = if !x.0.contains("videotoolbox") {
                        let config = ffi::avcodec_get_hw_config(enc.as_ptr(), i);
                        if config.is_null() {
                            println!("config is null {}", x.0);
                            break;
                        }
                        let type_ = (*config).device_type;
                        ::log::debug!("[enc] codec type {:?} {}, for: {}", type_, i, x.0);
                        let mut devices = DEVICES.lock();
                        if let Entry::Vacant(e) = devices.entry(type_ as u64 + device_hash) {
                            ::log::debug!("create {:?}", type_);
                            if let Ok(dev) = HWDevice::from_type(type_, device) {
                                ::log::debug!("created ok {:?}", type_);
                                e.insert(dev);
                            }
                        }
                        type_
                    } else {
                        ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX
                    };
                    let mut devices = DEVICES.lock();
                    if let Some(dev) = devices.get_mut(&(type_ as u64 + device_hash)) {
                        let mut constraints = ffi::av_hwdevice_get_hwframe_constraints(dev.as_mut_ptr(), ptr::null());
                        if !constraints.is_null() {
                            dev.hw_formats = pix_formats_to_vec((*constraints).valid_hw_formats);
                            dev.sw_formats = pix_formats_to_vec((*constraints).valid_sw_formats);
                            dev.min_size = ((*constraints).min_width, (*constraints).min_height);
                            dev.max_size = ((*constraints).max_width, (*constraints).max_height);

                            log::debug!("HW formats: {:?}", &dev.hw_formats);
                            log::debug!("SW formats: {:?}", &dev.sw_formats);

                            ffi::av_hwframe_constraints_free(&mut constraints);
                        }
                        return (x.0, x.1, Some(dev.device_type()));
                    }
                }
            }
        } else {
            super::append_log(&format!("Codec not found: {:?}\n", x.0));
        }
    }
    let x = encoders.last().unwrap();
    (x.0, x.1, None)
}

pub unsafe fn get_transfer_formats_from_gpu(frame: *mut ffi::AVFrame) -> Vec<format::Pixel> {
    unsafe {
        let mut formats = ptr::null_mut();
        if !frame.is_null() && !(*frame).hw_frames_ctx.is_null() {
            ffi::av_hwframe_transfer_get_formats((*frame).hw_frames_ctx, ffi::AVHWFrameTransferDirection::AV_HWFRAME_TRANSFER_DIRECTION_FROM, &mut formats, 0);
        }
        if formats.is_null() {
            Vec::new()
        } else {
            pix_formats_to_vec(formats)
        }
    }
}
pub unsafe fn get_transfer_formats_to_gpu(frame: *mut ffi::AVFrame) -> Vec<format::Pixel> {
    unsafe {
        let mut formats = ptr::null_mut();
        if !frame.is_null() && !(*frame).hw_frames_ctx.is_null() {
            ffi::av_hwframe_transfer_get_formats((*frame).hw_frames_ctx, ffi::AVHWFrameTransferDirection::AV_HWFRAME_TRANSFER_DIRECTION_TO, &mut formats, 0);
        }
        if formats.is_null() {
            Vec::new()
        } else {
            pix_formats_to_vec(formats)
        }
    }
}

pub fn is_hardware_format(format: ffi::AVPixelFormat) -> bool {
    format == ffi::AVPixelFormat::AV_PIX_FMT_CUDA ||
    format == ffi::AVPixelFormat::AV_PIX_FMT_DXVA2_VLD ||
    format == ffi::AVPixelFormat::AV_PIX_FMT_VDPAU ||
    format == ffi::AVPixelFormat::AV_PIX_FMT_D3D11 ||
    format == ffi::AVPixelFormat::AV_PIX_FMT_D3D11VA_VLD ||
    format == ffi::AVPixelFormat::AV_PIX_FMT_D3D12 ||
    format == ffi::AVPixelFormat::AV_PIX_FMT_VIDEOTOOLBOX ||
    format == ffi::AVPixelFormat::AV_PIX_FMT_MEDIACODEC ||
    format == ffi::AVPixelFormat::AV_PIX_FMT_OPENCL ||
    format == ffi::AVPixelFormat::AV_PIX_FMT_QSV ||
    format == ffi::AVPixelFormat::AV_PIX_FMT_MMAL ||
    format == ffi::AVPixelFormat::AV_PIX_FMT_VULKAN ||
    format == ffi::AVPixelFormat::AV_PIX_FMT_VAAPI
}

pub fn initialize_hwframes_context(encoder_ctx: *mut ffi::AVCodecContext, _frame_ctx: *mut ffi::AVFrame, type_: DeviceType, pixel_format: ffi::AVPixelFormat, size: (u32, u32), init_hwframes: bool, device_name: Option<&str>) -> Result<(), ()> {
    let mut devices = DEVICES.lock();
    let mut device_hash = 0;
    if let Some(dev_name) = device_name {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(dev_name.as_bytes());
        device_hash = hasher.finalize() as u64;
    }
    if let Some(dev) = devices.get_mut(&(type_ as u64 + device_hash)) {
        unsafe {
            if (*encoder_ctx).hw_device_ctx.is_null() && type_ != ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_QSV {
                (*encoder_ctx).hw_device_ctx = dev.add_ref();
                log::debug!("Setting hw_device_ctx {:?}", (*encoder_ctx).hw_device_ctx);
            }

            if init_hwframes {
                if dev.sw_formats.is_empty() && !(*encoder_ctx).codec.is_null() {
                    dev.sw_formats = codec_pix_formats(encoder_ctx, (*encoder_ctx).codec);
                    log::debug!("Setting codec formats: {:?}", dev.sw_formats);
                }

                if !dev.hw_formats.is_empty() {
                    let target_format: ffi::AVPixelFormat = {
                        if !dev.sw_formats.contains(&pixel_format.into()) {
                            log::warn!("Encoder doesn't support the desired pixel format ({:?})\n", pixel_format);
                            log::debug!("dev.sw_formats: {:?}", &dev.sw_formats);
                            let formats = get_transfer_formats_to_gpu(_frame_ctx);
                            if formats.is_empty() {
                                super::append_log(&format!("No frame transfer formats. Desired format: {:?}\n", pixel_format));
                                ffi::AVPixelFormat::AV_PIX_FMT_NONE
                            } else if formats.contains(&pixel_format.into()) {
                                pixel_format
                            } else {
                                // Just pick the first format.
                                // TODO: this should maybe take into consideration if the frame is 8 bit or more
                                format::Pixel::into(*formats.first().unwrap())
                            }
                        } else {
                            pixel_format
                        }
                    };
                    log::debug!("target_format: {:?}", &target_format);

                    if target_format != ffi::AVPixelFormat::AV_PIX_FMT_NONE {
                        let hw_format = *dev.hw_formats.first().unwrap(); // Safe because we check !is_empty() above

                        if (*encoder_ctx).hw_frames_ctx.is_null() {
                            let mut hw_frames_ref = ffi::av_hwframe_ctx_alloc(dev.as_mut_ptr());
                            if hw_frames_ref.is_null() {
                                super::append_log(&format!("Failed to create GPU frame context {:?}.\n", type_));
                                return Err(());
                            }
                            (*encoder_ctx).hw_frames_ctx = ffi::av_buffer_ref(hw_frames_ref);
                            ffi::av_buffer_unref(&mut hw_frames_ref);
                        } else {
                            log::debug!("hwframes already exists");
                        }
                        let mut frames_ctx_ref = (*encoder_ctx).hw_frames_ctx;

                        let frames_ctx = (*frames_ctx_ref).data as *mut ffi::AVHWFramesContext;
                        if (*frames_ctx).format    == ffi::AVPixelFormat::AV_PIX_FMT_NONE { (*frames_ctx).format    = hw_format.into(); }
                        if (*frames_ctx).sw_format == ffi::AVPixelFormat::AV_PIX_FMT_NONE { (*frames_ctx).sw_format = target_format; }
                        (*frames_ctx).width     = size.0 as i32;
                        (*frames_ctx).height    = size.1 as i32;
                        if type_ == ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_QSV || type_ == ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI {
                            (*frames_ctx).initial_pool_size = 20;
                        }

                        let err = ffi::av_hwframe_ctx_init(frames_ctx_ref);
                        if err < 0 {
                            super::append_log(&format!("Failed to initialize frame context. Error code: {}\n", err));
                            ffi::av_buffer_unref(&mut frames_ctx_ref);
                            return Err(());
                        } else {
                            log::debug!("inited hwframe ctx");
                        }
                        log::debug!("frames_ctx.format: {:?}", &(*frames_ctx).format);
                        log::debug!("frames_ctx.sw_format: {:?}", &(*frames_ctx).sw_format);
                        (*encoder_ctx).pix_fmt = (*frames_ctx).format;
                    }
                }

            }
            return Ok(());
        }
    } else {
        log::warn!("DEVICES didn't have {:?}", type_);
    }
    Ok(())
}

/// Answers, by actually trying it, whether this machine's encoder will open with `pix_fmt`.
///
/// An encoder's advertised `pix_fmts` list belongs to the ffmpeg build, not to the hardware. On
/// this machine `hevc_nvenc` lists `P210LE`, but an Ampere card has no 4:2:2 NVENC block at all
/// (that arrived with Blackwell), and the truth only surfaces inside `avcodec_open2`:
///
/// ```text
///   [hevc_nvenc] YUV422P not supported
///   [hevc_nvenc] Provided device doesn't support required NVENC features
/// ```
///
/// That is a hard failure at render time, not the recoverable "unsupported pixel format" dialog,
/// so a warning built on the advertised list alone would send the user straight into it. Opening
/// a throwaway 640x480 encoder is the same question the renderer asks, asked early.
///
/// Cached: opening an NVENC session costs on the order of 100 ms, and this is called from the UI
/// thread while the user is picking a codec. The answer cannot change while the app is running -
/// it is a property of the hardware and the driver.
pub fn encoder_accepts_format(encoder_name: &str, pix_fmt: format::Pixel) -> bool {
    // Keyed on the raw AVPixelFormat value: `format::Pixel` is comparable but not hashable.
    static PROBES: LazyLock<Mutex<HashMap<(String, i32), bool>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

    let key = (encoder_name.to_string(), Into::<ffi::AVPixelFormat>::into(pix_fmt) as i32);
    if let Some(&cached) = PROBES.lock().get(&key) { return cached; }

    let accepted = unsafe {
        let Ok(name) = CString::new(encoder_name) else { return false; };
        let codec = ffi::avcodec_find_encoder_by_name(name.as_ptr());
        if codec.is_null() { return false; }

        let mut ctx = ffi::avcodec_alloc_context3(codec);
        if ctx.is_null() { return false; }

        // Big enough to clear every encoder's minimum (NVENC's is 33x17) and small enough that
        // allocating the session costs nothing. Dimensions are not what is being tested here.
        (*ctx).width  = 640;
        (*ctx).height = 480;
        (*ctx).pix_fmt = pix_fmt.into();
        (*ctx).time_base = ffi::AVRational { num: 1, den: 30 };
        (*ctx).framerate = ffi::AVRational { num: 30, den: 1 };
        (*ctx).bit_rate = 10_000_000;

        // The probe deliberately provokes encoder errors, and those go through the same ffmpeg
        // log callback that fills the buffer shown in the export error dialog. Silence ffmpeg
        // around the call and put the level back, so a successful export never carries the noise
        // from a format we merely asked about.
        let prev_level = ffi::av_log_get_level();
        ffi::av_log_set_level(ffi::AV_LOG_QUIET);
        let ret = ffi::avcodec_open2(ctx, codec, std::ptr::null_mut());
        ffi::av_log_set_level(prev_level);

        ffi::avcodec_free_context(&mut ctx);
        ret >= 0
    };

    log::debug!("Encoder probe: {encoder_name} + {pix_fmt:?} -> {}", if accepted { "ok" } else { "rejected" });
    PROBES.lock().insert(key, accepted);
    accepted
}

/// The pixel format an encoder should be asked for when it cannot take the source's own.
///
/// Ranked, not first-match. The order below is what "closest" means, and it matters because the
/// answer is offered to the user as *the* alternative:
///
/// 1. The source format itself, when the encoder takes it. Nothing to choose.
/// 2. The hand-written pairs. These are the same layout under two names - `P210LE` is
///    `YUV422P10LE` with the chroma planes interleaved - so they are exact, not approximations,
///    and they stay ahead of the scoring so this function's long-standing answers cannot drift.
/// 3. Highest score, where the score prefers keeping the source's bit depth first and its chroma
///    subsampling second. Depth first because banding in a gradient is more visible than chroma
///    softening on any real footage, and because a 10-bit source dropped to 8 bits cannot be
///    recovered by a later grade.
///
/// Hardware and non-YUV formats are excluded from the scoring: an encoder advertises `CUDA` and
/// `D3D11` as "supported", but they are surfaces rather than layouts, and offering `BGRA` for a
/// YUV source would silently add a colour conversion the user did not ask for. Both remain
/// reachable through the exact-pair rules if a source genuinely is RGB.
pub fn find_best_matching_codec(codec: format::Pixel, supported: &[format::Pixel]) -> Option<format::Pixel> {
    if supported.is_empty() { return None; }

    if supported.contains(&codec) { return Some(codec); }

    let pairs = vec![
        (format::Pixel::P210LE, format::Pixel::YUV422P10LE),
        (format::Pixel::P010LE, format::Pixel::YUV420P10LE),
        (format::Pixel::NV12,   format::Pixel::YUV420P),
        (format::Pixel::NV21,   format::Pixel::YUV420P),
    ];
    for (a, b) in pairs {
        if codec == a && supported.contains(&b) { return Some(b); }
        if codec == b && supported.contains(&a) { return Some(a); }
    }

    if let Some(best) = best_by_layout(codec, supported) {
        return Some(best);
    }

    super::append_log(&format!("No matching codec, we need {:?} and supported are: {:?}
", codec, supported));

    None
}

/// Bit depth, chroma subsampling as (log2 width, log2 height), and whether the format is a plain
/// software YUV layout - the only kind worth offering as a substitute for another YUV format.
///
/// `None` for anything ffmpeg has no descriptor for, which excludes it from the scoring rather
/// than letting it score as a zero-depth match.
fn layout_of(format: format::Pixel) -> Option<(i32, (i32, i32), bool)> {
    unsafe {
        let desc = ffi::av_pix_fmt_desc_get(format.into());
        if desc.is_null() { return None; }
        let flags = (*desc).flags;
        let is_hw    = (flags & ffi::AV_PIX_FMT_FLAG_HWACCEL as u64) != 0;
        let is_rgb   = (flags & ffi::AV_PIX_FMT_FLAG_RGB     as u64) != 0;
        let is_pal   = (flags & ffi::AV_PIX_FMT_FLAG_PAL     as u64) != 0;
        let is_float = (flags & ffi::AV_PIX_FMT_FLAG_FLOAT   as u64) != 0;
        Some((
            (*desc).comp[0].depth,
            ((*desc).log2_chroma_w as i32, (*desc).log2_chroma_h as i32),
            !is_hw && !is_rgb && !is_pal && !is_float && (*desc).nb_components >= 3,
        ))
    }
}

/// The best software YUV substitute for `codec` among `supported`, or `None` when there is none.
///
/// Scored rather than sorted so the tie-breaks are visible: an exact depth match is worth more
/// than any chroma match, a deeper format is preferred to a shallower one at equal distance
/// (rounding up never destroys source information), and chroma is compared on both axes.
fn best_by_layout(codec: format::Pixel, supported: &[format::Pixel]) -> Option<format::Pixel> {
    let (want_depth, want_chroma, want_yuv) = layout_of(codec)?;
    if !want_yuv { return None; }

    supported.iter()
        .filter_map(|&f| {
            let (depth, chroma, is_yuv) = layout_of(f)?;
            if !is_yuv { return None; }

            // Depth dominates: the worst possible chroma penalty (4 below) can never outweigh
            // one bit of depth (100 per bit), so no amount of chroma gain buys a depth loss.
            let depth_penalty = (depth - want_depth).abs() * 100
                              + if depth < want_depth { 50 } else { 0 }; // prefer rounding up
            let chroma_penalty = (chroma.0 - want_chroma.0).abs()
                               + (chroma.1 - want_chroma.1).abs();
            Some((f, -(depth_penalty + chroma_penalty)))
        })
        // `max_by_key` returns the *last* maximum; the encoder lists its own preferred formats
        // first, so reverse the iteration to keep the earliest on a tie.
        .collect::<Vec<_>>().into_iter().rev()
        .max_by_key(|&(_, score)| score)
        .map(|(f, _)| f)
}

// pub fn get_supported_pixel_formats(name: &str) -> Vec<ffi::AVPixelFormat> {
//     if let Some(mut codec) = encoder::find_by_name(name) {
//         unsafe {
//             pix_formats_to_vec((*codec.as_mut_ptr()).pix_fmts)
//         }
//     } else {
//         Vec::new()
//     }
// }
