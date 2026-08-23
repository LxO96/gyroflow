// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2021-2022 Adrian <adrian.eddy at gmail>

mod ffmpeg_audio;
mod ffmpeg_video;
mod ffmpeg_video_converter;
mod audio_resampler;
pub mod ffmpeg_processor;
pub mod ffmpeg_hw;
pub mod render_queue;
pub mod mdk_processor;
pub mod video_processor;
pub mod zero_copy;
use gyroflow_core::settings;
use zero_copy::*;
#[cfg(target_os = "android")]
pub mod ffmpeg_android;

pub use self::video_processor::VideoProcessor;
pub use self::ffmpeg_processor::{ FfmpegProcessor, FFmpegError };
use render_queue::RenderOptions;
use crate::core::{ StabilizationManager, stabilization::* };
use crate::core::color::OutputColorSpace;
use ffmpeg_next::{ format::Pixel, frame::Video, codec, Error, ffi };
use std::cell::RefCell;
use std::ffi::c_void;
use std::os::raw::c_char;
use std::os::raw::c_int;
use std::rc::Rc;
use std::sync::{ Arc, atomic::AtomicBool };
use parking_lot::RwLock;
use gyroflow_core::gpu::Buffers;

#[derive(Debug, PartialEq, Clone, Copy)]
enum GpuType {
    Nvidia, Amd, Intel, AppleSilicon, Android, WindowsArm, Unknown
}
lazy_static::lazy_static! {
    static ref GPU_TYPE: RwLock<GpuType> = RwLock::new(GpuType::Unknown);
}
pub fn set_gpu_type_from_name(name: &str) {
    if name.is_empty() { return; }
    ffmpeg_hw::clear_devices();
    let name = name.to_ascii_lowercase();
         if name.contains("nvidia") || name.contains("quadro") || name.contains("grid") { *GPU_TYPE.write() = GpuType::Nvidia; }
    else if name.contains("amd") || name.contains("advanced micro devices") { *GPU_TYPE.write() = GpuType::Amd; }
    else if name.contains("intel") && !name.contains("intel(r) core(tm)") { *GPU_TYPE.write() = GpuType::Intel; }
    else if name.contains("apple m") { *GPU_TYPE.write() = GpuType::AppleSilicon; }
    else if name.contains("adreno") { *GPU_TYPE.write() = if cfg!(target_os = "android") { GpuType::Android } else { GpuType::WindowsArm }; }
    else {
        log::warn!("Unknown GPU {}", name);
    }

    let gpu_type = *GPU_TYPE.read();
    if gpu_type == GpuType::Nvidia {
        ffmpeg_hw::initialize_ctx(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA);
    }

    if settings::get_bool("useVulkanEncoder", false) { ffmpeg_hw::initialize_ctx(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN); }
    if settings::get_bool("useD3D12Encoder",  false) { ffmpeg_hw::initialize_ctx(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D12VA); }
    #[cfg(target_os = "windows")]
    if gpu_type == GpuType::Amd {
        ffmpeg_hw::initialize_ctx(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA);
    }
    #[cfg(target_os = "android")]
    {
        ffmpeg_hw::initialize_ctx(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_MEDIACODEC);
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        #[cfg(target_os = "macos")]
        if !name.contains("apple m") {
            // Disable GPU decoding on Intel macOS by default
            gyroflow_core::settings::set("gpudecode", false.into());
        }
        ffmpeg_hw::initialize_ctx(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX);
    }

    ::log::debug!("GPU type: {:?}, from name: {}", gpu_type, name);
}

pub fn get_possible_encoders(codec: &str, use_gpu: bool) -> Vec<(&'static str, bool)> { // -> (name, is_gpu)
    if codec.contains("PNG") || codec.contains("png") { return vec![("png", false)]; }
    if codec.contains("EXR") || codec.contains("exr") { return vec![("exr", false)]; }

    let mut encoders = if use_gpu {
        match codec {
            "H.264/AVC" => vec![
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                ("h264_videotoolbox", true),
                #[cfg(any(target_os = "windows", target_os = "linux"))]
                ("h264_nvenc",        true),
                #[cfg(any(target_os = "windows", target_os = "linux"))]
                ("h264_amf",          true),
                #[cfg(any(target_os = "linux"))]
                ("h264_vaapi",        true),
                #[cfg(any(target_os = "windows", target_os = "linux"))]
                ("h264_qsv",          true),
                #[cfg(target_os = "windows")]
                ("h264_mf",           true),
                #[cfg(any(target_os = "windows", target_os = "linux"))]
                ("h264_vulkan",       true),
                #[cfg(target_os = "linux")]
                ("h264_v4l2m2m",      true),
                #[cfg(target_os = "android")]
                ("h264_mediacodec",   true),
                ("libx264",           false),
            ],
            "H.265/HEVC" => vec![
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                ("hevc_videotoolbox", true),
                #[cfg(any(target_os = "windows", target_os = "linux"))]
                ("hevc_nvenc",        true),
                #[cfg(any(target_os = "windows", target_os = "linux"))]
                ("hevc_amf",          true),
                #[cfg(any(target_os = "linux"))]
                ("hevc_vaapi",        true),
                #[cfg(any(target_os = "windows", target_os = "linux"))]
                ("hevc_qsv",          true),
                #[cfg(target_os = "windows")]
                ("hevc_mf",           true),
                #[cfg(any(target_os = "windows", target_os = "linux"))]
                ("hevc_vulkan",       true),
                #[cfg(target_os = "windows")]
                ("hevc_d3d12va",      true),
                #[cfg(target_os = "linux")]
                ("hevc_v4l2m2m",      true),
                #[cfg(target_os = "android")]
                ("hevc_mediacodec",   true),
                ("libx265",           false),
            ],
            "AV1" => vec![
                #[cfg(any(target_os = "windows", target_os = "linux"))]
                ("av1_nvenc",        true),
                #[cfg(any(target_os = "windows", target_os = "linux"))]
                ("av1_amf",          true),
                #[cfg(any(target_os = "windows", target_os = "linux"))]
                ("av1_qsv",          true),
                #[cfg(any(target_os = "linux"))]
                ("av1_vaapi",         true),
                #[cfg(target_os = "android")]
                ("av1_mediacodec",   true),
                ("librav1e",         false),
                ("libaom-av1",       false),
                ("libsvtav1",        false),
            ],
            "ProRes" => vec![
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                ("prores_videotoolbox", true),
                //#[cfg(any(target_os = "windows", target_os = "linux", target_os = "android"))]
                //("prores_ks_vulkan", true),
                ("prores_ks", false)
            ],
            "DNxHD"    => vec![("dnxhd", false)],
            "CineForm" => vec![("cfhd", false)],
            _          => vec![]
        }
    } else {
        match codec {
            "H.264/AVC"  => vec![("libx264", false)],
            "H.265/HEVC" => vec![("libx265", false)],
            "ProRes"     => vec![("prores_ks", false)],
            "DNxHD"      => vec![("dnxhd", false)],
            "CineForm"   => vec![("cfhd", false)],
            "AV1"        => vec![("librav1e", false), ("libaom-av1", false), ("libsvtav1", false)],
            _            => vec![]
        }
    };

    let gpu_type = *GPU_TYPE.read();
    if gpu_type != GpuType::Nvidia {
        encoders.retain(|x| !x.0.contains("nvenc"));
    }

    if !settings::get_bool("useVulkanEncoder", false) { encoders.retain(|x| !x.0.contains("_vulkan")); }
    if !settings::get_bool("useD3D12Encoder",  false) { encoders.retain(|x| !x.0.contains("_d3d12")); }

    if gpu_type != GpuType::Amd {
        encoders.retain(|x| !x.0.contains("_amf"));
    }
    if gpu_type != GpuType::Intel {
        encoders.retain(|x| !x.0.contains("qsv"));
    }
    log::debug!("Possible encoders with {:?}: {:?}", gpu_type, encoders);
    encoders
}

/// `true` if swscale in the linked ffmpeg can read this format.
fn sws_supports_input(format: Pixel) -> bool {
    unsafe { ffi::sws_isSupportedInput(format.into()) != 0 }
}
/// `true` if swscale in the linked ffmpeg can write this format.
fn sws_supports_output(format: Pixel) -> bool {
    unsafe { ffi::sws_isSupportedOutput(format.into()) != 0 }
}

/// Component depth of the first component, and whether the format holds floats.
/// Falls back to `(16, false)` for formats ffmpeg has no descriptor for, which is the
/// conservative answer: it never loses source precision.
fn format_depth(format: Pixel) -> (i32, bool) {
    unsafe {
        let desc = ffi::av_pix_fmt_desc_get(format.into());
        if desc.is_null() { return (16, false); }
        let is_float = ((*desc).flags & ffi::AV_PIX_FMT_FLAG_FLOAT as u64) != 0;
        ((*desc).comp[0].depth, is_float)
    }
}

/// The packed **RGB** format a source must be converted to before the colour stage can
/// run on it, or `None` when no conversion is possible or needed.
///
/// The colour stage (`apply_color`) grades whatever three channel values it is handed as
/// if they were R, G and B - no kernel performs any colour-space conversion. So the frame
/// reaching it must be packed RGB: planar sources would give the kernel one channel at a
/// time, and YUV sources (`AYUV64LE` included) would have exposure act as a hue rotation
/// and saturation as a chroma shift, because U and V are offset by 0.5.
///
/// `None` means "do not convert": either the source is already dispatched as packed RGB
/// and can be graded where it is, or colour cannot be baked for it at all. In the latter
/// case the plane dispatch is left untouched and the per-plane gate in `create_planes_proc!`
/// switches the colour stage off, so `None` is always safe.
///
/// Only `RGB24` and `RGBA` qualify as "already packed RGB". `RGB48BE` and `RGBA64BE` are
/// dispatched to the `RGB16` / `RGBA16` `PixelType`s, whose fields are plain native-endian
/// `u16` read straight out of the frame by `bytemuck`, so on a little-endian host the kernel
/// sees every sample byte-swapped. Untouched bytes round-trip, which is why the non-baking
/// dispatch gets away with it, but colour arithmetic on byte-swapped values produces
/// garbage - so those two are converted like any other unsuitable source.
///
/// Source precision is preserved: 8-bit in, 8-bit out; anything deeper goes to 16-bit.
/// 32-bit float sources are refused - see the note on `Pixel::RGBAF32LE` below.
fn packed_color_format(format: Pixel) -> Option<Pixel> {
    match format {
        // Already dispatched as packed RGB (`RGB8` / `RGBA8`), in the host's byte order.
        Pixel::RGB24 | Pixel::RGBA => None,
        _ => {
            let (depth, is_float) = format_depth(format);
            if is_float {
                // `Pixel::RGBAF32LE` exists and `RGBAf` is a `PixelType`, but swscale in
                // ffmpeg 8.1 cannot *output* it ("rgbaf32le is not supported as output
                // pixel format", "128bpp not supported by yuv2rgb"), so there is no way to
                // get a float source into a packed RGB frame. Converting to 16-bit instead
                // would silently throw away the source's precision, which is the one thing
                // a float source is for. Baking is refused for these; see
                // `swscale_still_cannot_output_packed_float`.
                return None;
            }
            let target = if depth <= 8 { Pixel::RGBA } else { Pixel::RGBA64LE };
            // A source swscale cannot read would turn a renderable file into a hard error,
            // so fall back to not baking rather than to failing the render.
            if !sws_supports_input(format) || !sws_supports_output(target) { return None; }
            Some(target)
        }
    }
}

/// Companion to the prores videotoolbox workaround in `create_planes_proc!`, for the baking
/// path only.
///
/// `ffmpeg_video.rs` tags the *encoder* limited-range for NV12 + `prores_videotoolbox` (see
/// the matching workaround there), so the samples handed to it have to be limited-range too.
/// Without baking the kernel does that itself, with `FIX_COLOR_RANGE` / `remap_colorrange`.
/// Under baking the kernel only ever sees packed RGB, where a YUV range remap would be
/// meaningless, and the conversion back to NV12 is swscale's job instead. swscale will apply
/// exactly that range as part of the RGB->YUV matrix - but only if the destination frame
/// declares it, so declare it. The frame then also describes what it really holds, which is
/// what `ffmpeg_video.rs` reads if it converts again on the way to the encoder.
///
/// `prores_videotoolbox` only exists on macOS, so this is unreachable elsewhere. Note the
/// encoder-side workaround also covers `dnxhd`, which never had a kernel-side counterpart;
/// that pre-existing asymmetry is left alone here.
fn tag_limited_range_for_prores_videotoolbox(out_frame: &mut Video, is_prores_videotoolbox: bool) {
    if is_prores_videotoolbox && out_frame.format() == Pixel::NV12 {
        out_frame.set_color_range(ffmpeg_next::util::color::Range::MPEG);
    }
}

/// Declares on the output what the baked grade actually produced.
///
/// Only the tags change - not one pixel. The export otherwise inherits the source's colour
/// tags verbatim (`ffmpeg_video.rs`, which reads all four off the output frame), and Sony XAVC
/// leaves matrix, transfer and primaries unspecified at the container level. The result was a
/// file declaring `color_range=pc` and nothing else; with no matrix declared, players fall back
/// to their own defaults and enough of them assume limited range for 10-bit YUV that a correct
/// full-range export gets visibly stretched on playback.
///
/// Called only from the baking path, and that is the point: the tag describes what the LUT
/// produced, so it may only be written when the grade was actually baked in. Tagging Rec.709
/// on a file whose pixels are still S-Log3 would be a worse lie than saying nothing.
fn tag_output_color_space(out_frame: &mut Video, space: OutputColorSpace) {
    use ffmpeg_next::util::color::{ Primaries, TransferCharacteristic, Space };
    let (primaries, transfer, matrix) = match space {
        // The user has not said, so keep inheriting the source as before.
        OutputColorSpace::Unspecified => return,
        OutputColorSpace::Rec709      => (Primaries::BT709,    TransferCharacteristic::BT709,        Space::BT709),
        OutputColorSpace::Srgb        => (Primaries::BT709,    TransferCharacteristic::IEC61966_2_1, Space::BT709),
        // P3 D65 shares the 709 transfer and matrix; only the primaries differ.
        OutputColorSpace::DciP3D65    => (Primaries::SMPTE432, TransferCharacteristic::BT709,        Space::BT709),
        OutputColorSpace::Rec2020Pq   => (Primaries::BT2020,   TransferCharacteristic::SMPTE2084,    Space::BT2020NCL),
        OutputColorSpace::Rec2020Hlg  => (Primaries::BT2020,   TransferCharacteristic::ARIB_STD_B67, Space::BT2020NCL),
    };
    out_frame.set_color_primaries(primaries);
    out_frame.set_color_transfer_characteristic(transfer);
    out_frame.set_color_space(matrix);
}

/// The colour stage is RGB-only, so it must never run on a plane that cannot carry RGB:
/// a single-channel `Luma8` / `Luma16` / `R32f` plane or a two-channel `UV8` / `UV16` one.
/// This is applied to every plane `create_planes_proc!` builds, so no dispatch arm can
/// bypass it whatever the flags say.
fn should_bake_color<T: PixelType>(bake_color: bool) -> bool {
    bake_color && T::COUNT >= 3
}

pub fn render<F, F2>(stab: Arc<StabilizationManager>, progress: F, input_file: &gyroflow_core::InputFile, render_options: &RenderOptions, gpu_decoder_index: i32, trim_range_ind: Option<usize>, cancel_flag: Arc<AtomicBool>, pause_flag: Arc<AtomicBool>, encoder_initialized: F2) -> Result<(), FFmpegError>
    where F: Fn((f64, usize, usize, bool, bool)) + Send + Sync + Clone,
          F2: Fn(String) + Send + Sync + Clone
{
    log::debug!("ffmpeg_hw::supported_gpu_backends: {:?}", ffmpeg_hw::supported_gpu_backends());

    let params = stab.params.read();
    let org_trim_ranges = params.trim_ranges.clone();
    let trim_ranges = trim_range_ind.map(|x| vec![params.trim_ranges[x]]).unwrap_or_else(|| params.trim_ranges.clone());
    let trim_ratio = if !render_options.pad_with_black && !render_options.preserve_other_tracks {
        params.get_trim_ratio()
    } else {
        1.0
    };
    let total_frame_count = params.frame_count;
    let fps_scale = params.fps_scale;
    let has_alpha = params.background[3] < 1.0;

    let mut pixel_format = render_options.pixel_format.clone();

    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    let _prevent_system_sleep = keep_awake::inhibit_system("Gyroflow", "Rendering video");
    #[cfg(any(target_os = "ios", target_os = "android"))]
    let _prevent_system_sleep = keep_awake::inhibit_display("Gyroflow", "Rendering video");

    let output_width = render_options.output_width;
    let output_height = render_options.output_height;

    let duration_ms = params.duration_ms;
    let fps = params.fps;
    let video_speed = params.video_speed;

    let render_duration = params.duration_ms * trim_ratio;
    let render_frame_count = (total_frame_count as f64 * trim_ratio).round() as usize;

    // Only use post-conversion processing when background is not opaque
    let order = if params.background[3] < 1.0 {
        ffmpeg_video::ProcessingOrder::PostConversion
    } else {
        ffmpeg_video::ProcessingOrder::PreConversion
    };

    drop(params);

    let lens_checksum = stab.lens.read().checksum.clone();

    let mut decoder_options = ffmpeg_next::Dictionary::new();
    if input_file.image_sequence_fps > 0.0 {
        let fps = fps_to_rational(input_file.image_sequence_fps);
        decoder_options.set("framerate", &format!("{}/{}", fps.numerator(), fps.denominator()));
    }
    if input_file.image_sequence_start > 0 {
        decoder_options.set("start_number", &format!("{}", input_file.image_sequence_start));
    }
    if cfg!(target_os = "android") {
        decoder_options.set("ndk_codec", "1");
    }
    let gpu_decoding = stab.gpu_decoding.load(std::sync::atomic::Ordering::SeqCst);
    let mut proc = FfmpegProcessor::from_file(&input_file.url, gpu_decoding && gpu_decoder_index >= 0, gpu_decoder_index as usize, Some(decoder_options))?;

    let render_options_dict = render_options.get_encoder_options_dict();
    let hwaccel_device = render_options_dict.get("hwaccel_device");

    match render_options.audio_codec.as_ref() {
        "AAC"         => proc.audio_codec = ffmpeg_next::codec::Id::AAC,
        "PCM (s16le)" => proc.audio_codec = ffmpeg_next::codec::Id::PCM_S16LE,
        "PCM (s16be)" => proc.audio_codec = ffmpeg_next::codec::Id::PCM_S16BE,
        "PCM (s24le)" => proc.audio_codec = ffmpeg_next::codec::Id::PCM_S24LE,
        "PCM (s24be)" => proc.audio_codec = ffmpeg_next::codec::Id::PCM_S24BE,
        _ => { }
    }

    let interpolation: Interpolation = render_options.interpolation.as_str().into();
    let ffmpeg_interpolation = match interpolation {
        Interpolation::Bilinear => ffmpeg_next::software::scaling::flag::Flags::BILINEAR,
        Interpolation::Bicubic  => ffmpeg_next::software::scaling::flag::Flags::BICUBIC,
       _ => ffmpeg_next::software::scaling::flag::Flags::LANCZOS,
    };

    log::debug!("interpolation: {:?}", &interpolation);
    log::debug!("proc.gpu_device: {:?}", &proc.gpu_device);
    let encoder = ffmpeg_hw::find_working_encoder(&get_possible_encoders(&render_options.codec, render_options.use_gpu), hwaccel_device);
    proc.video_codec = Some(encoder.0.to_owned());
    proc.video.gpu_encoding = encoder.1;
    proc.video.ffmpeg_interpolation = ffmpeg_interpolation.bits();
    proc.video.encoder_params.hw_device_type = encoder.2;
    proc.video.encoder_params.options.set("threads", "auto");
    proc.video.encoder_params.metadata = render_options.get_metadata_dict();
    proc.video.processing_order = order;
    log::debug!("video_codec: {:?}, processing_order: {:?}", &proc.video_codec, proc.video.processing_order);

    if !render_options.pad_with_black && !render_options.preserve_other_tracks && !trim_ranges.is_empty() {
        proc.ranges_ms = trim_ranges.iter().map(|x| (if x.0 > 0.0 { Some(x.0 * duration_ms) } else { None }, if x.1 < 1.0 { Some(x.1 * duration_ms) } else { None })).collect();
    }

    match proc.video_codec.as_deref() {
        Some("prores_ks") | Some("prores_videotoolbox") => {
            let profiles = ["Proxy", "LT", "Standard", "HQ", "4444", "4444XQ"];
            let pix_fmts = [Pixel::YUV422P10LE, Pixel::YUV422P10LE, Pixel::YUV422P10LE, Pixel::YUV422P10LE, Pixel::YUVA444P10LE, Pixel::YUVA444P10LE];
            if let Some(profile) = profiles.iter().position(|&x| x == render_options.codec_options) {
                proc.video.encoder_params.options.set("profile", &format!("{}", profile));
                if proc.video_codec.as_deref() == Some("prores_ks") {
                    proc.video.encoder_params.pixel_format = Some(pix_fmts[profile]);
                }
            }
            proc.video.clone_frames = proc.video_codec.as_deref() == Some("prores_ks");
        }
        Some("dnxhd") => {
            let profiles = ["DNxHD", "DNxHR LB", "DNxHR SQ", "DNxHR HQ", "DNxHR HQX", "DNxHR 444"];
            let pix_fmts = [Pixel::YUV422P, Pixel::YUV422P, Pixel::YUV422P, Pixel::YUV422P, Pixel::YUV422P10LE, Pixel::YUV444P10LE];
            if let Some(profile) = profiles.iter().position(|&x| x == render_options.codec_options) {
                proc.video.encoder_params.options.set("profile", &format!("{}", profile));
                proc.video.encoder_params.pixel_format = Some(pix_fmts[profile]);
            }
            proc.video.clone_frames = true;
        }
        Some("cfhd") => {
            proc.video.encoder_params.pixel_format = Some(Pixel::YUV422P10LE);
            proc.video.clone_frames = true;
        }
        Some("png") => {
            if render_options.codec_options.contains("16-bit") {
                proc.video.encoder_params.pixel_format = Some(if has_alpha { Pixel::RGBA64BE } else { Pixel::RGB48BE });
            } else {
                proc.video.encoder_params.pixel_format = Some(if has_alpha { Pixel::RGBA } else { Pixel::RGB24 });
            }
            proc.video.clone_frames = true;
        }
        Some("exr") => {
            proc.video.clone_frames = true;
            proc.video.encoder_params.options.set("compression", "1"); // RLE compression
            proc.video.encoder_params.options.set("gamma", "1.0");
            proc.video.encoder_params.pixel_format = Some(if has_alpha { Pixel::GBRAPF32LE } else { Pixel::GBRPF32LE });
            /*Decoder options:
                -layer             <string>     .D.V....... Set the decoding layer (default "")
                -part              <int>        .D.V....... Set the decoding part (from 0 to INT_MAX) (default 0)
                -gamma             <float>      .D.V....... Set the float gamma value when decoding (from 0.001 to FLT_MAX) (default 1)
                -apply_trc         <int>        .D.V....... color transfer characteristics to apply to EXR linear input (from 1 to 18) (default gamma)
                    bt709           1            .D.V....... BT.709
                    gamma           2            .D.V....... gamma
                    gamma22         4            .D.V....... BT.470 M
                    gamma28         5            .D.V....... BT.470 BG
                    smpte170m       6            .D.V....... SMPTE 170 M
                    smpte240m       7            .D.V....... SMPTE 240 M
                    linear          8            .D.V....... Linear
                    log             9            .D.V....... Log
                    log_sqrt        10           .D.V....... Log square root
                    iec61966_2_4    11           .D.V....... IEC 61966-2-4
                    bt1361          12           .D.V....... BT.1361
                    iec61966_2_1    13           .D.V....... IEC 61966-2-1
                    bt2020_10bit    14           .D.V....... BT.2020 - 10 bit
                    bt2020_12bit    15           .D.V....... BT.2020 - 12 bit
                    smpte2084       16           .D.V....... SMPTE ST 2084
                    smpte428_1      17           .D.V....... SMPTE ST 428-1
            */
        }
        _ => { }
    }

    let is_sequence = matches!(proc.video_codec.as_deref(), Some("png") | Some("exr"));

    if cfg!(any(target_os = "macos", target_os = "ios")) {
        proc.video.encoder_params.options.set("allow_sw", "1");
        proc.video.encoder_params.options.set("realtime", "0");

        // Test if `constant_bit_rate` is supported
        let vt_cbr = VT_SUPPORTS_CONSTANT_BIT_RATE.read().clone();
        if vt_cbr.is_none() {
            let log = FFMPEG_LOG.read().clone();
            if let Some(enc) = ffmpeg_next::encoder::find_by_name("h264_videotoolbox") {
                let ctx_ptr = unsafe { ffi::avcodec_alloc_context3(enc.as_ptr()) };
                let context = unsafe { codec::context::Context::wrap(ctx_ptr, Some(Rc::new(0))) };
                let mut encoder = context.encoder().video()?;
                encoder.set_width(1920);
                encoder.set_height(1080);
                encoder.set_format(ffmpeg_next::format::Pixel::NV12);
                encoder.set_time_base(ffmpeg_next::Rational::new(30, 1));
                let mut options = ffmpeg_next::Dictionary::new();
                options.set("allow_sw", "1");
                options.set("constant_bit_rate", "1");
                if encoder.open_with(options).is_ok() {
                    VT_SUPPORTS_CONSTANT_BIT_RATE.write().replace(true);
                } else {
                    VT_SUPPORTS_CONSTANT_BIT_RATE.write().replace(false);
                }
            }
            *FFMPEG_LOG.write() = log;
        }
        if VT_SUPPORTS_CONSTANT_BIT_RATE.read().is_some_and(|x| x) {
            proc.video.encoder_params.options.set("constant_bit_rate", "1");
        }
    }
    if encoder.0.contains("nvenc") {
        proc.video.encoder_params.options.set("b_ref_mode", "disabled");
    }

    if cfg!(target_os = "android") {
        proc.video.encoder_params.options.set("ndk_codec", "1");
    }

    proc.video.encoder_params.keyframe_distance_s = render_options.keyframe_distance.max(0.0001);

    proc.preserve_other_tracks = render_options.preserve_other_tracks && !crate::util::is_insta360(&input_file.url);

    for (key, value) in render_options_dict.iter() {
        log::info!("Setting encoder option {}: {}", key, value);
        if key == "pix_fmt" {
            pixel_format = value.to_string();
            continue;
        }
        proc.video.encoder_params.options.set(key, value);
    }

    if encoder.0.contains("nvenc") {
        if proc.video.encoder_params.options.get("rc").is_none() {
            proc.video.encoder_params.options.set("rc", "cbr");
        }
    }

    if !pixel_format.is_empty() {
        use std::str::FromStr;
        match Pixel::from_str(&pixel_format.to_ascii_lowercase()) {
            Ok(px) => { proc.video.encoder_params.pixel_format = Some(px); },
            Err(e) => { ::log::debug!("Unknown requested pixel format: {}, {:?}", pixel_format, e); }
        }
    }

    let start_us = (proc.ranges_ms.first().and_then(|x| x.0).unwrap_or_default() * 1000.0) as i64;

    if !render_options.audio {
        proc.audio_codec = codec::Id::None;
    }

    log::debug!("start_us: {}, render_duration: {}, render_frame_count: {}", start_us, render_duration, render_frame_count);

    let mut planes = Vec::<Box<dyn FnMut(i64, &mut Video, &mut Video, usize, bool)>>::new();

    // Colour adjustments are baked into the render only when the user asked for it.
    // Read once, so the decision (and the pixel format it implies) cannot change between
    // the frame that builds the plane closures and the frames that run them.
    // Cleared below if the source turns out not to be bakeable, so that the decision is
    // carried by this one flag rather than by `color_convert_to` being `None` - the plane
    // dispatch reads this, and `None` there does not by itself mean "do not grade".
    let mut bake_color = {
        let c = stab.color.read();
        c.is_active() && c.bake_in_render
    };
    // Read once, next to `bake_color`, because the two must describe the same decision: the
    // tag is only ever written where the grade was actually baked in.
    let output_color_space = stab.color.read().output_color_space;
    // The packed RGB format the source must be converted to for the colour stage, decided
    // once when the planes are built and reused by the per-frame dispatch below, so the two
    // cannot disagree about the layout the closures were built for.
    let mut color_convert_to: Option<Pixel> = None;

    let is_prores_videotoolbox = proc.video_codec.as_deref() == Some("prores_videotoolbox");

    let progress2 = progress.clone();
    let mut process_frame = 0;
    if let Some(i) = trim_range_ind {
        for x in 0..i {
            let x = org_trim_ranges[x];
            process_frame += ((x.1 - x.0) * total_frame_count as f64).round() as usize;
        }
    }

    proc.on_encoder_initialized(|enc: &ffmpeg_next::encoder::video::Video| {
        encoder_initialized(enc.codec().map(|x| x.name().to_string()).unwrap_or_default());
        Ok(())
    });

    let mut prev_real_ts = 0;
    let mut ramped_ts = 0.0;
    let mut final_ts = 0;
    let interval = (1_000_000.0 / fps).round() as i64;
    let is_speed_changed = video_speed != 1.0 || stab.keyframes.read().is_keyframed(&gyroflow_core::keyframes::KeyframeType::VideoSpeed);
    if is_speed_changed {
        proc.audio_codec = codec::Id::None; // Audio not supported when changing speed
    }

    let render_globals = Rc::new(RefCell::new(zero_copy::RenderGlobals::default()));

    proc.on_frame(move |mut timestamp_us, input_frame, output_frame, converter, rate_control| {
        let fill_with_background = render_options.pad_with_black && !trim_ranges.is_empty() &&
            !trim_ranges.iter().any(|x| timestamp_us >= (x.0 * duration_ms * 1000.0).round() as i64 &&
                                        timestamp_us <= (x.1 * duration_ms * 1000.0).round() as i64);

        if let Some(scale) = fps_scale {
            timestamp_us = (timestamp_us as f64 / scale).round() as i64;
        }

        if is_speed_changed {
            let vid_speed = stab.keyframes.read().value_at_video_timestamp(&gyroflow_core::keyframes::KeyframeType::VideoSpeed, timestamp_us as f64 / 1000.0).unwrap_or(video_speed);
            let current_interval = ((rate_control.out_timestamp_us - prev_real_ts) as f64) / vid_speed;
            ramped_ts += current_interval;
            prev_real_ts = rate_control.out_timestamp_us;
            if ramped_ts < (final_ts as f64 + interval as f64 / 2.0) { // interval/2 because we want frame in the middle of the range, not in the end
                rate_control.repeat_times = 0; // skip this frame
                process_frame += 1;
                return Ok(());
            } else {
                let repeat_times = current_interval / interval as f64;
                if repeat_times >= 1.5 {
                    // Need to duplicate the frames
                    rate_control.repeat_times = repeat_times.round() as i64;
                    rate_control.repeat_interval = interval;
                }
            }
            rate_control.out_timestamp_us = final_ts;
            final_ts += interval * rate_control.repeat_times;
        }

        let output_frame = output_frame.unwrap();

        macro_rules! create_planes_proc {
            ($planes:ident, $(($t:tt, $in_frame:expr, $out_frame:expr, $ind:expr, $yuvi:expr, $max_val:expr), )*) => {
                $({
                    let in_size = zero_copy::get_plane_size($in_frame, $ind);
                    let out_size = zero_copy::get_plane_size($out_frame, $ind);

                    let org_sizes = {
                        let params = stab.params.read();
                        (params.size, params.output_size)
                    };

                    let mut plane = Stabilization::default();
                    plane.interpolation = interpolation;
                    plane.share_wgpu_instances = true;
                    plane.set_device(stab.params.read().current_device as isize);

                    // Workaround for a bug in prores videotoolbox encoder.
                    // Under baking `$in_frame` is the converted packed RGB frame, so this is
                    // false and the kernel-side remap does not run - deliberately: it is a
                    // YUV range remap and RGB has no such range. The equivalent for that path
                    // is `tag_limited_range_for_prores_videotoolbox`, which makes swscale do
                    // it during the conversion back to NV12.
                    if $in_frame.format() == ffmpeg_next::format::Pixel::NV12 && is_prores_videotoolbox {
                        plane.kernel_flags.set(KernelParamsFlags::FIX_COLOR_RANGE, true);
                    }

                    let mut compute_params = ComputeParams::from_manager(&stab);
                    // Structural gate, applied to every plane this macro builds: the colour
                    // stage runs only when the user asked to bake it *and* this plane can
                    // actually carry RGB. See `should_bake_color`.
                    compute_params.color.enabled &= should_bake_color::<$t>(bake_color);

                    let is_limited_range = $out_frame.color_range() == ffmpeg_next::util::color::Range::MPEG;
                    compute_params.background = <$t as PixelType>::from_rgb_color(compute_params.background, &$yuvi, is_limited_range);

                    let offset_us = {
                        let params = stab.params.read();
                        (params.frame_offset as f64 / params.fps * 1000000.0).round() as i64
                    };

                    plane.init_size(org_sizes.0, org_sizes.1);
                    plane.set_compute_params(compute_params);
                    let render_globals = render_globals.clone();
                    $planes.push(Box::new(move |timestamp_us: i64, in_frame_data: &mut Video, out_frame_data: &mut Video, plane_index: usize, fill_with_background: bool| {
                        let mut g = render_globals.borrow_mut();
                        let wgpu_format = $t::wgpu_format().map(|x| x.0);
                        let timestamp_us = timestamp_us + offset_us;

                        let mut buffers = Buffers {
                            input:  get_plane_buffer(in_frame_data, in_size, plane_index, &mut g, wgpu_format),
                            output: get_plane_buffer(out_frame_data, out_size, plane_index, &mut g, wgpu_format)
                        };

                        if plane.initialized_backend.is_none() || plane.pending_device_change.is_some() {
                            plane.ensure_ready_for_processing::<$t>(timestamp_us, None, &mut buffers);
                            plane.stab_data.clear();
                        }
                        let mut transform = plane.get_frame_transform_at::<$t>(timestamp_us, None, &mut buffers);
                        transform.kernel_params.pixel_value_limit = $max_val;
                        transform.kernel_params.max_pixel_value = $max_val;
                        if plane.initialized_backend.is_wgpu() && $t::wgpu_format().map(|x| x.2).unwrap_or_default() {
                            transform.kernel_params.pixel_value_limit = 1.0;
                            transform.kernel_params.max_pixel_value = 1.0;
                        }
                        transform.kernel_params.plane_index = plane_index as i32;
                        if fill_with_background {
                            transform.kernel_params.flags |= KernelParamsFlags::FILL_WITH_BACKGROUND.bits();
                        }
                        if let Err(e) = plane.process_pixels::<$t>(timestamp_us, None, &mut buffers, Some(&transform)) {
                            ::log::error!("Failed to process pixels: {e:?}");
                        }
                    }));
                })*
            };
        }

        if planes.is_empty() {
            // Good reference about video formats: https://source.chromium.org/chromium/chromium/src/+/master:media/base/video_frame.cc
            // https://gist.github.com/Jim-Bar/3cbba684a71d1a9d468a6711a6eddbeb

            let mut format = input_frame.format();
            if input_frame.width() == 0 || input_frame.height() == 0 || output_frame.width() == 0 || output_frame.height() == 0 {
                return Err(FFmpegError::FrameEmpty);
            }

            let is_hardware_frame = if let Some(underlying_format) = zero_copy::map_hardware_format(format, input_frame) {
                log::debug!("HW frame ({:?}) underlying format: {:?}", format, underlying_format);
                format = underlying_format;
                true
            } else {
                false
            };

            if bake_color {
                if is_hardware_frame {
                    // A hardware frame's pixels are not in system memory, so swscale cannot
                    // convert it; the zero-copy plane path is the only way to reach them.
                    ::log::warn!("Colour baking is not supported for hardware frames ({:?}); rendering without colour adjustments", input_frame.format());
                    bake_color = false;
                } else {
                    color_convert_to = packed_color_format(format);
                    if color_convert_to.is_none() && !matches!(format, Pixel::RGB24 | Pixel::RGBA) {
                        // `packed_color_format` also returns `None` when swscale cannot
                        // handle the format, and those sources dispatch to 3- and 4-channel
                        // plane types that `should_bake_color` would happily grade - as YUV,
                        // or byte-swapped. Make the warning true instead of relying on the
                        // per-plane channel-count gate to catch it.
                        ::log::warn!("Colour baking is not supported for {:?} sources; rendering without colour adjustments", format);
                        bake_color = false;
                    }
                }
            }

            if let Some(target) = color_convert_to {
                // Baking: the colour stage grades R, G and B of one pixel at once, so the
                // frame it sees must be packed RGB. `packed_color_format` picks the target
                // and guarantees it is one of these two.
                ::log::info!("Baking colour adjustments: converting {:?} to {:?}", format, target);
                // Must happen before the first conversion: the swscale contexts read the
                // frame's range when they are built, and they are cached from then on.
                tag_limited_range_for_prores_videotoolbox(output_frame, is_prores_videotoolbox);
                tag_output_color_space(output_frame, output_color_space);
                converter.convert_pixel_format_color_aware(input_frame, output_frame, target, ffmpeg_interpolation, |converted_frame, converted_output| {
                    if target == Pixel::RGBA {
                        create_planes_proc!(planes, (RGBA8,  converted_frame, converted_output, 0, [], 255.0), );
                    } else {
                        create_planes_proc!(planes, (RGBA16, converted_frame, converted_output, 0, [], 65535.0), );
                    }
                })?;
            } else {
                match format {
                    Pixel::NV12 => {
                        create_planes_proc!(planes,
                            (Luma8, input_frame, output_frame, 0, [0], 255.0),
                            (UV8,   input_frame, output_frame, 1, [1,2], 255.0),
                        );
                    },
                    Pixel::NV21 => {
                        create_planes_proc!(planes,
                            (Luma8, input_frame, output_frame, 0, [0], 255.0),
                            (UV8,   input_frame, output_frame, 1, [2,1], 255.0),
                        );
                    },
                    Pixel::P010LE | Pixel::P016LE |
                    Pixel::P210LE | Pixel::P216LE |
                    Pixel::P410LE | Pixel::P416LE => {
                        let max_val = match input_frame.format() {
                            // I'm not sure if this is correct but it appears that P010LE uses 16-bit values, even though it's 10-bit
                            //Pixel::P010LE | Pixel::P210LE | Pixel::P410LE => 1023.0,
                            _ => 65535.0
                        };
                        create_planes_proc!(planes,
                            (Luma16, input_frame, output_frame, 0, [0], max_val),
                            (UV16,   input_frame, output_frame, 1, [1,2], max_val),
                        );
                    },
                    Pixel::YUV420P | Pixel::YUVJ420P => {
                        create_planes_proc!(planes,
                            (Luma8, input_frame, output_frame, 0, [0], 255.0),
                            (Luma8, input_frame, output_frame, 1, [1], 255.0),
                            (Luma8, input_frame, output_frame, 2, [2], 255.0),
                        );
                    },
                    Pixel::YUV420P10LE | Pixel::YUV420P12LE | Pixel::YUV420P14LE | Pixel::YUV420P16LE |
                    Pixel::YUV422P10LE | Pixel::YUV422P12LE | Pixel::YUV422P14LE | Pixel::YUV422P16LE |
                    Pixel::YUV444P10LE | Pixel::YUV444P12LE | Pixel::YUV444P14LE | Pixel::YUV444P16LE => {
                        let max_val = match input_frame.format() {
                            Pixel::YUV420P10LE | Pixel::YUV422P10LE | Pixel::YUV444P10LE => 1023.0,
                            Pixel::YUV420P12LE | Pixel::YUV422P12LE | Pixel::YUV444P12LE => 4095.0,
                            Pixel::YUV420P14LE | Pixel::YUV422P14LE | Pixel::YUV444P14LE => 16383.0,
                            _ => 65535.0
                        };
                        create_planes_proc!(planes,
                            (Luma16, input_frame, output_frame, 0, [0], max_val),
                            (Luma16, input_frame, output_frame, 1, [1], max_val),
                            (Luma16, input_frame, output_frame, 2, [2], max_val),
                        );
                    },
                    Pixel::YUVA444P10LE | Pixel::YUVA444P12LE | Pixel::YUVA444P16LE => {
                        let max_val = match input_frame.format() {
                            Pixel::YUVA444P10LE => 1023.0,
                            Pixel::YUVA444P12LE => 4095.0,
                            _ => 65535.0
                        };
                        create_planes_proc!(planes,
                            (Luma16, input_frame, output_frame, 0, [0], max_val),
                            (Luma16, input_frame, output_frame, 1, [1], max_val),
                            (Luma16, input_frame, output_frame, 2, [2], max_val),
                            (Luma16, input_frame, output_frame, 3, [3], max_val),
                        );
                    },
                    Pixel::GBRAPF32LE => { create_planes_proc!(planes,
                        (R32f,  input_frame, output_frame, 0, [2], 255.0),
                        (R32f,  input_frame, output_frame, 0, [0], 255.0),
                        (R32f,  input_frame, output_frame, 0, [1], 255.0),
                        (R32f,  input_frame, output_frame, 0, [3], 255.0),
                    ); },
                    Pixel::GBRPF32LE => { create_planes_proc!(planes,
                        (R32f,  input_frame, output_frame, 0, [2], 255.0),
                        (R32f,  input_frame, output_frame, 0, [0], 255.0),
                        (R32f,  input_frame, output_frame, 0, [1], 255.0),
                    ); },
                    Pixel::AYUV64LE => { create_planes_proc!(planes, (AYUV16, input_frame, output_frame, 0, [3,0,1,2], 65535.0), ); },
                    Pixel::RGB24    => { create_planes_proc!(planes, (RGB8,   input_frame, output_frame, 0, [], 255.0), ); },
                    Pixel::RGBA     => { create_planes_proc!(planes, (RGBA8,  input_frame, output_frame, 0, [], 255.0), ); },
                    Pixel::RGB48BE  => { create_planes_proc!(planes, (RGB16,  input_frame, output_frame, 0, [], 65535.0), ); },
                    Pixel::RGBA64BE => { create_planes_proc!(planes, (RGBA16, input_frame, output_frame, 0, [], 65535.0), ); },
                    format => { // All other convert to YUV444P16LE
                        ::log::info!("Unknown format {:?}, converting to YUV444P16LE", format);
                        // Go through 4:4:4 because of even plane dimensions
                        converter.convert_pixel_format(input_frame, output_frame, Pixel::YUV444P16LE, ffmpeg_interpolation, |converted_frame, converted_output| {
                            create_planes_proc!(planes,
                                (Luma16, converted_frame, converted_output, 0, [0], 65535.0),
                                (Luma16, converted_frame, converted_output, 1, [1], 65535.0),
                                (Luma16, converted_frame, converted_output, 2, [2], 65535.0),
                            );
                        })?;
                    }
                }
            }
        }
        if planes.is_empty() {
            return Err(FFmpegError::UnknownPixelFormat(input_frame.format()));
        }

        let mut undistort_frame = |frame: &mut Video, out_frame: &mut Video| {
            for (i, cb) in planes.iter_mut().enumerate() {
                (*cb)(timestamp_us, frame, out_frame, i, fill_with_background);
            }
            progress2((process_frame as f64 / render_frame_count as f64, process_frame, render_frame_count, false, false));
        };

        if let Some(target) = color_convert_to {
            // The plane closures above were built for this converted packed RGB frame, so
            // the frame handed to them must go through the same conversion. `color_convert_to`
            // is the one value both decisions read, so they cannot disagree.
            tag_limited_range_for_prores_videotoolbox(output_frame, is_prores_videotoolbox);
            tag_output_color_space(output_frame, output_color_space);
            converter.convert_pixel_format_color_aware(input_frame, output_frame, target, ffmpeg_interpolation, |converted_frame, converted_output| {
                undistort_frame(converted_frame, converted_output);
            })?;
        } else {
            match input_frame.format() {
                Pixel::VIDEOTOOLBOX | // Pixel::D3D11 |
                Pixel::NV12 | Pixel::NV21 | Pixel::YUV420P | Pixel::YUVJ420P |
                Pixel::P010LE | Pixel::P016LE | Pixel::P210LE | Pixel::P216LE | Pixel::P410LE | Pixel::P416LE |
                Pixel::YUV420P10LE | Pixel::YUV420P12LE | Pixel::YUV420P14LE | Pixel::YUV420P16LE |
                Pixel::YUV422P10LE | Pixel::YUV422P12LE | Pixel::YUV422P14LE | Pixel::YUV422P16LE |
                Pixel::YUV444P10LE | Pixel::YUV444P12LE | Pixel::YUV444P14LE | Pixel::YUV444P16LE |
                Pixel::YUVA444P10LE | Pixel::YUVA444P12LE | Pixel::YUVA444P16LE |
                Pixel::AYUV64LE | Pixel::GBRAPF32LE | Pixel::GBRPF32LE |
                Pixel::RGB24 | Pixel::RGBA | Pixel::RGB48BE | Pixel::RGBA64BE => {
                    undistort_frame(input_frame, output_frame)
                },
                _ => {
                    converter.convert_pixel_format(input_frame, output_frame, Pixel::YUV444P16LE, ffmpeg_interpolation, |converted_frame, converted_output| {
                        undistort_frame(converted_frame, converted_output);
                    })?;
                }
            }
        }

        process_frame += 1;
        // log::debug!("process_frame: {}, timestamp_us: {}", process_frame, timestamp_us);

        Ok(())
    });

    let mut filename = render_options.output_filename.clone();
    let folder = &render_options.output_folder;
    if cfg!(not(any(target_os = "android", target_os = "ios"))) && !gyroflow_core::filesystem::exists(folder) {
        let path = gyroflow_core::filesystem::url_to_path(folder);
        if !path.is_empty() {
            let _ = std::fs::create_dir_all(path);
        }
    }
    if org_trim_ranges.len() > 1 {
        if let Some(ind) = trim_range_ind {
            if let Some(pos) = filename.rfind('.') {
                filename.insert_str(pos, &format!("-{:0>3}", ind + 1));
            }
        }
    }

    let start_ms = proc.ranges_ms.first().and_then(|x| x.0);
    let mut render_filename = filename.clone();
    if cfg!(not(any(target_os = "android", target_os = "ios"))) && !is_sequence {
        render_filename = format!("{filename}.tmp");
    }
    proc.render(folder, &render_filename, (output_width as u32, output_height as u32), if render_options.bitrate > 0.0 { Some(render_options.bitrate) } else { None }, cancel_flag, pause_flag)?;

    drop(proc);

    let output_url = gyroflow_core::filesystem::get_file_url(folder, &filename, false);

    if render_filename != filename {
        let output_url_temp = gyroflow_core::filesystem::get_file_url(folder, &render_filename, false);
        if let Err(e) = std::fs::rename(&gyroflow_core::filesystem::url_to_path(&output_url_temp), &gyroflow_core::filesystem::url_to_path(&output_url)) {
            ::log::error!("Failed to rename file from {output_url_temp} to {output_url}: {e:?}");
        }
    }

    let re = regex::Regex::new(r#"%[0-9]+d"#).unwrap();
    if re.is_match(&filename) {
        ::log::debug!("Removing {output_url}");
        let _ = gyroflow_core::filesystem::remove_file(&output_url);
    }
    if trim_range_ind.is_none() || trim_range_ind == Some(org_trim_ranges.len() - 1) {
        progress((1.0, render_frame_count, render_frame_count, true, false));
    }

    crate::util::update_file_times(&output_url, &input_file.url, start_ms);

    if render_options.preserve_other_tracks {
        if let Err(e) = crate::util::copy_insta360_metadata(&output_url, &input_file.url) {
            ::log::error!("Failed to copy Insta360 metadata: {e:?}");
        }
    }

    crate::util::report_lens_profile_usage(lens_checksum);

    Ok(())
}

pub fn init_log() {
	unsafe {
        ffi::av_log_set_level(ffi::AV_LOG_INFO);
        ffi::av_log_set_callback(Some(ffmpeg_log));
    }
}

pub fn fps_to_rational(fps: f64) -> ffmpeg_next::Rational {
    if fps.fract() > 0.1 {
        ffmpeg_next::Rational::new((fps * 1001.0).round() as i32, 1001)
    } else {
        ffmpeg_next::Rational::new(fps.round() as i32, 1)
    }
}

lazy_static::lazy_static! {
    pub static ref FFMPEG_LOG: Arc<RwLock<String>> = Arc::new(RwLock::new(String::new()));
    pub static ref LAST_PREFIX: Arc<RwLock<i32>> = Arc::new(RwLock::new(1));
    pub static ref VT_SUPPORTS_CONSTANT_BIT_RATE: RwLock<Option<bool>> = RwLock::new(None);
}

#[cfg(not(any(all(target_os = "linux", target_arch = "x86_64"), all(target_os = "macos", target_arch = "x86_64"))))]
type VaList = ffi::va_list;
#[cfg(any(all(target_os = "linux", target_arch = "x86_64"), all(target_os = "macos", target_arch = "x86_64")))]
type VaList = *mut ffi::__va_list_tag;

#[allow(improper_ctypes_definitions)]
unsafe extern "C" fn ffmpeg_log(avcl: *mut c_void, level: i32, fmt: *const c_char, vl: VaList) {
    unsafe {
        if level <= ffi::av_log_get_level() {
            let mut line = vec![0u8; 2048];
            let mut prefix: i32 = *LAST_PREFIX.read();

            #[cfg(any(target_os = "android", all(target_os = "linux", target_arch = "aarch64")))]
            let written = ffi::av_log_format_line2(avcl, level, fmt, vl, line.as_mut_ptr() as *mut u8, line.len() as i32, &mut prefix);
            #[cfg(not(any(target_os = "android", all(target_os = "linux", target_arch = "aarch64"))))]
            let written = ffi::av_log_format_line2(avcl, level, fmt, vl, line.as_mut_ptr() as *mut i8, line.len() as i32, &mut prefix);
            if written > 0 {
                line.resize(written as usize, 0u8);

                if let Ok(mut line) = String::from_utf8(line) {
                    if line.contains("Unknown profile bitstream") ||
                    line.contains("infe version < 2") ||
                    line.contains("Update your FFmpeg version to the newest one from Git.") ||
                    line.contains("Cannot find an index entry before timestamp") ||
                    line.contains("stream set to be discarded by default.") ||
                    line.contains("Missing key frame while searching for timestamp") {
                        return;
                    }
                    *LAST_PREFIX.write() = prefix;

                    // ffi::av_log_default_callback(avcl, level, fmt, vl);

                    match level {
                        ffi::AV_LOG_PANIC | ffi::AV_LOG_FATAL | ffi::AV_LOG_ERROR => {
                            ::log::error!("{}", line.trim());
                            line = format!("<font color=\"#d82626\">{}</font>", line);
                        },
                        ffi::AV_LOG_WARNING => {
                            ::log::warn!("{}", line.trim());
                            line = format!("<font color=\"#f6a10c\">{}</font>", line);
                        },
                        _ => { ::log::debug!("{}", line.trim()); }
                    }
                    FFMPEG_LOG.write().push_str(&line);
                }
            }
        }
    }
}

pub fn append_log(msg: &str) { ::log::debug!("{}", msg); FFMPEG_LOG.write().push_str(msg); }
pub fn get_log() -> String { FFMPEG_LOG.read().clone() }
pub fn clear_log() { FFMPEG_LOG.write().clear() }

unsafe fn to_str<'a>(ptr: *const c_char) -> std::borrow::Cow<'a, str> {
    if ptr.is_null() { return std::borrow::Cow::Borrowed(""); }
    unsafe { std::ffi::CStr::from_ptr(ptr) }.to_string_lossy()
}
unsafe fn codec_options(c: *const ffi::AVCodec) {
    if c.is_null() {
        log::warn!("Null pointer to encoder.");
        return;
    }

    unsafe {
        use std::fmt::Write;
        let mut ret = String::new();
        let _ = writeln!(ret, "{} <b>{}</b>:\n", ["Decoder", "Encoder"][ffi::av_codec_is_encoder(c) as usize], to_str((*c).name));

        if !(*c).pix_fmts.is_null() {
            ret.push_str("Supported pixel formats (-pix_fmt): ");
            for i in 0..100 {
                let fmt = (*c).pix_fmts.offset(i);
                if fmt.is_null() { break; }
                let p = *fmt;
                if p == ffi::AVPixelFormat::AV_PIX_FMT_NONE {
                    break;
                }
                if i > 0 { ret.push_str(", "); }
                ret.push_str(&to_str(ffi::av_get_pix_fmt_name(p)));
            }
        }

        if !(*c).priv_class.is_null() {
            ret.push_str("<pre>");
            FFMPEG_LOG.write().push_str(&ret);
            show_help_children((*c).priv_class, ffi::AV_OPT_FLAG_ENCODING_PARAM | ffi::AV_OPT_FLAG_DECODING_PARAM);
            FFMPEG_LOG.write().push_str("</pre>Additional supported flags:<pre>-hwaccel_device\n-qscale</pre>");
        }
    }
}

unsafe fn show_help_children(mut class: *const ffi::AVClass, flags: c_int) {
    unsafe {
        if !(*class).option.is_null() {
            let ptr = std::ptr::null_mut();
            ffi::av_opt_show2((&mut class) as *mut *const _ as *mut _, ptr, flags, 0);
        }
    }
    // let mut iter = std::ptr::null_mut();
    // loop {
    //     let child = ffi::av_opt_child_class_iterate(class, &mut iter);
    //     if child.is_null() {
    //         break;
    //     }
    //     show_help_children(child, flags);
    // }
}

pub fn get_default_encoder(codec: &str, gpu: bool) -> String {
    let encoder = ffmpeg_hw::find_working_encoder(&get_possible_encoders(codec, gpu), None);
    encoder.0.to_string()
}
pub fn get_encoder_options(name: &str) -> String {
	init_log();
    clear_log();
    match ffmpeg_next::encoder::find_by_name(name) {
        Some(encoder) => { unsafe { codec_options(encoder.as_ptr()); } },
        None => log::warn!("Failed to find codec by name: {name}")
    }
    let ret = get_log().replace("E..V.......", "").replace('\n', "<br>");
    clear_log();
    ret
}

/*
pub fn test() {
    log::debug!("FfmpegProcessor::supported_gpu_backends: {:?}", ffmpeg_hw::supported_gpu_backends());

    let stab = StabilizationManager::<crate::core::stabilization::RGBA8>::default();
    let duration_ms = 15015.0;
    let frame_count = 900;
    let fps = 60000.0/1001.0;
    //let video_size = (3840, 2160);
    let video_size = (5120, 3840);

    let vid = "/Users/eddy/Downloads/colors-GX029349.MP4";

    stab.init_from_video_data(duration_ms, fps, frame_count, video_size).unwrap();
    stab.load_gyro_data(vid, |_|(), Arc::new(AtomicBool::new(false)));
    {
        let mut gyro = stab.gyro.write();

        //gyro.set_offset(0, -26.0);
        gyro.integration_method = 1;
        gyro.integrate();
    }
    // stab.load_lens_profile("E:/clips/GoPro/GoPro_Hero_7_Black_4K_60_wide_16by9_1_120.json").unwrap();
    stab.set_size(video_size.0, video_size.1);
    stab.set_smoothing_method(0);
    //stab.smoothing_id = 1;
    //stab.smoothing_algs[1].as_mut().set_parameter("time_constant", 0.4);
    {
        let mut params = stab.params.write();
        // params.frame_readout_time = 8.9;
        params.fov = 1.0;
        params.background = nalgebra::Vector4::new(0.0, 0.0, 0.0, 255.0);
        params.lens_correction_amount = 0.0;
    }
    stab.recompute_blocking();

    render(
        Arc::new(stab),
        move |_params: (f64, usize, usize, bool)| {
            // ::log::debug!("frame {}/{}", params.1, params.2);
        },
        vid.into(),
        &RenderOptions {
            codec: "ProRes".into(),
            codec_options: "Standard".into(),
            output_width: video_size.0,
            output_height: video_size.1,
            bitrate: 100.0,
            use_gpu: true,
            audio: true,
            pixel_format: "".into(),
        },
        -1,
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false))
    ).unwrap();
}
// use opencv::core::{Mat, Size, CV_8UC1};
// use std::os::raw::c_void;

pub fn test_decode() {
    let mut proc = FfmpegProcessor::from_file("/storage/self/primary/Download/gf/h8.MP4", true, 0, None).unwrap();

    let recv = proc.android_handles.as_mut().unwrap().receiver.take().unwrap().0.acquire();
    // TODO: gpu scaling in filters, example here https://github.com/zmwangx/rust-ffmpeg/blob/master/examples/transcode-audio.rs, filter scale_cuvid or scale_npp
    proc.on_frame(move |timestamp_us, input_frame, _output_frame, converter, _rate_control| {

        ffmpeg_android::release_frame(input_frame);
        let hw_buf = recv.recv().unwrap();

        unsafe {
            let desc = unsafe {
                let mut result = std::mem::MaybeUninit::uninit();
                ndk_sys::AHardwareBuffer_describe(hw_buf.as_ptr(), result.as_mut_ptr());
                result.assume_init()
            };
            ::log::debug!("recv: {:x}", desc.format);
        }

        ::log::debug!("recv: {:?}", hw_buf.as_ptr());

        ::log::debug!("ts: {} width: {}, format: {:?}", timestamp_us, input_frame.width(), input_frame.format());

        /*let (w, h) = (small_frame.plane_width(0) as i32, small_frame.plane_height(0) as i32);
        let mut bytes = small_frame.data_mut(0);
        let inp = unsafe { Mat::new_size_with_data(Size::new(w, h), CV_8UC1, bytes.as_mut_ptr() as *mut c_void, w as usize) }.unwrap();
        opencv::imgcodecs::imwrite("D:/test.jpg", &inp, &opencv::types::VectorOfi32::new());*/
        Ok(())
    });
    let _time = std::time::Instant::now();
    let _ = proc.start_decoder_only(vec![(0.0, 1000.0)], Arc::new(AtomicBool::new(false)));
    ::log::debug!("Done in {:.3} ms", _time.elapsed().as_micros() as f64 / 1000.0);
}*/

#[cfg(test)]
mod packed_color_format_tests {
    use super::*;

    /// The two formats the baking path converts into must be writable by swscale, and a
    /// real conversion into each must run - `sws_isSupportedOutput` alone is only a claim.
    #[test]
    fn swscale_can_output_the_packed_rgb_formats_we_bake_into() {
        for target in [Pixel::RGBA, Pixel::RGBA64LE] {
            assert!(sws_supports_output(target), "swscale cannot output {target:?}");

            let mut ctx = ffmpeg_next::software::scaling::Context::get(
                Pixel::YUV420P, 64, 64, target, 64, 64,
                ffmpeg_next::software::scaling::Flags::BILINEAR
            ).unwrap_or_else(|e| panic!("YUV420P -> {target:?} Context::get failed: {e:?}"));

            let mut src = Video::new(Pixel::YUV420P, 64, 64);
            let mut dst = Video::new(target, 64, 64);
            for p in 0..3 { for b in src.data_mut(p).iter_mut() { *b = 128; } }
            ctx.run(&mut src, &mut dst).unwrap_or_else(|e| panic!("YUV420P -> {target:?} run() failed: {e:?}"));
        }
    }

    /// OQ1. `Pixel::RGBAF32LE` exists and `RGBAf` is a `PixelType`, but swscale in ffmpeg
    /// 8.1 refuses it as an *output* format ("rgbaf32le is not supported as output pixel
    /// format" / "128bpp not supported by yuv2rgb"), which is why `packed_color_format`
    /// refuses to bake colour for float sources. If this test ever starts failing, swscale
    /// has gained support and the float path can be implemented.
    #[test]
    fn swscale_still_cannot_output_packed_float() {
        assert!(!sws_supports_output(Pixel::RGBAF32LE));
        assert!(ffmpeg_next::software::scaling::Context::get(
            Pixel::YUV420P, 64, 64, Pixel::RGBAF32LE, 64, 64,
            ffmpeg_next::software::scaling::Flags::BILINEAR
        ).is_err());
    }

    #[test]
    fn packed_color_format_preserves_source_precision() {
        // 8-bit sources -> 8-bit packed RGB.
        for f in [Pixel::NV12, Pixel::NV21, Pixel::YUV420P, Pixel::YUVJ420P, Pixel::YUYV422] {
            assert_eq!(packed_color_format(f), Some(Pixel::RGBA), "{f:?}");
        }
        // Deeper integer sources -> 16-bit packed RGB. AYUV64LE is packed, but it is YUV,
        // so it must be converted too. RGB48BE / RGBA64BE are packed RGB already, but in the
        // wrong byte order for the native-endian `RGB16` / `RGBA16` kernel types, so they are
        // converted as well - swscale does the byte swap and the kernel grades real values.
        for f in [Pixel::P010LE, Pixel::P016LE, Pixel::YUV420P10LE, Pixel::YUV444P12LE, Pixel::YUVA444P16LE, Pixel::AYUV64LE, Pixel::RGB48BE, Pixel::RGBA64BE] {
            assert_eq!(packed_color_format(f), Some(Pixel::RGBA64LE), "{f:?}");
        }
        // Already dispatched as packed RGB in native byte order - graded in place.
        for f in [Pixel::RGB24, Pixel::RGBA] {
            assert_eq!(packed_color_format(f), None, "{f:?}");
        }
        // Float sources - refused, see `swscale_still_cannot_output_packed_float`.
        for f in [Pixel::GBRPF32LE, Pixel::GBRAPF32LE] {
            assert_eq!(packed_color_format(f), None, "{f:?}");
        }
    }

    /// The gate that keeps the RGB colour stage off planes that cannot carry RGB.
    #[test]
    fn color_stage_is_refused_on_planes_that_cannot_carry_rgb() {
        assert!(!should_bake_color::<Luma8>(true));
        assert!(!should_bake_color::<Luma16>(true));
        assert!(!should_bake_color::<R32f>(true));
        assert!(!should_bake_color::<UV8>(true));
        assert!(!should_bake_color::<UV16>(true));

        assert!(should_bake_color::<RGB8>(true));
        assert!(should_bake_color::<RGBA8>(true));
        assert!(should_bake_color::<RGB16>(true));
        assert!(should_bake_color::<RGBA16>(true));

        // And nothing is graded when the user did not ask for baking.
        assert!(!should_bake_color::<RGBA8>(false));
        assert!(!should_bake_color::<RGBA16>(false));
    }
}
