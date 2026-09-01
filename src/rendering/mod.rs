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
pub mod temporal_window;
use gyroflow_core::settings;
use zero_copy::*;
#[cfg(target_os = "android")]
pub mod ffmpeg_android;

pub use self::video_processor::VideoProcessor;
pub use self::ffmpeg_processor::{ FfmpegProcessor, FFmpegError };
use render_queue::RenderOptions;
use crate::core::{ StabilizationManager, stabilization::* };
use crate::core::color::{ OutputColorSpace, OutputRange };
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

/// Chooses the range the output is encoded in, converting the pixels to match.
///
/// This is the one place in this file that changes samples rather than tags, and the mechanism
/// is the same one `tag_limited_range_for_prores_videotoolbox` relies on: swscale applies the
/// range as part of the RGB->YUV matrix, but only if the destination frame declares it, so
/// declaring it here is what performs the conversion. `ffmpeg_video.rs` then reads the range
/// back off the frame and tags the encoder to match, so the file describes what it holds.
///
/// Why anyone would want it: a baked log-to-display LUT expands contrast into the full code
/// range, and a great many players ignore `color_range=pc` for YUV and assume video range
/// anyway - stretching the picture by 1023/876 about a pivot at code 445, which reads as
/// crushed blacks and clipped highlights. Encoding video range instead means those players are
/// right by construction. The cost is ~15% of the code range, which matters if the file is
/// going back into a grading suite and does not if it is going to a viewer.
///
/// Baking-only, and unlike the colour-space tag that is a hard technical limit rather than a
/// policy: without baking the frame never becomes RGB, swscale never builds an RGB->YUV matrix,
/// and the only range machinery in play is the kernels' own `remap_colorrange`, which is driven
/// by `FIX_COLOR_RANGE` per plane. Routing this through that instead would mean touching all
/// five hand-maintained kernels. The UI gates on baking for the same reason.
///
/// Called before the first conversion, next to the other two, because the swscale contexts read
/// the frame's range when they are built and are cached from then on - and *before*
/// `tag_limited_range_for_prores_videotoolbox`, deliberately: that one works around an encoder
/// bug and has to win, so a user asking for `Full` on prores NV12 still gets the limited-range
/// samples that encoder requires rather than a broken file.
fn set_output_range(out_frame: &mut Video, range: OutputRange) {
    use ffmpeg_next::util::color::Range;
    match range {
        // Keep whatever the decoded source carried, which is what happened before this existed.
        OutputRange::SameAsSource => { },
        OutputRange::Limited => out_frame.set_color_range(Range::MPEG),
        OutputRange::Full    => out_frame.set_color_range(Range::JPEG),
    }
}

/// The colour stage is RGB-only, so it must never run on a plane that cannot carry RGB:
/// a single-channel `Luma8` / `Luma16` / `R32f` plane or a two-channel `UV8` / `UV16` one.
/// This is applied to every plane `create_planes_proc!` builds, so no dispatch arm can
/// bypass it whatever the flags say.
fn should_bake_color<T: PixelType>(bake_color: bool) -> bool {
    bake_color && T::COUNT >= 3
}

/// Whether `timestamp_us` falls outside every trim range, i.e. this output frame should be
/// replaced with the configured background rather than real (trimmed-away) content.
/// Factored out so the trailing frames a temporal window drains after decoding ends
/// (`timestamp_us` no longer arriving from the decode loop) can compute the same thing the
/// live per-frame path does.
fn compute_fill_with_background(pad_with_black: bool, trim_ranges: &[(f64, f64)], duration_ms: f64, timestamp_us: i64) -> bool {
    pad_with_black && !trim_ranges.is_empty() &&
        !trim_ranges.iter().any(|x| timestamp_us >= (x.0 * duration_ms * 1000.0).round() as i64 &&
                                    timestamp_us <= (x.1 * duration_ms * 1000.0).round() as i64)
}

/// A decoded frame plus the two per-frame values that must travel with it through the
/// temporal window.
///
/// Emission is delayed by `k` frames, so anything derived from the frame's own position in
/// the timeline has to be captured at push time and restored at emit time. Recomputing
/// either from the *centre* timestamp at emit time is wrong:
///
/// - `out_ts` is the normalised output PTS (`ffmpeg_video.rs` builds it as
///   `ts - frame_ts.first + frame_ts.add_video`, then applies the speed ramp on top). The
///   window's `centre_ts` is the raw decoder timestamp - a *different domain*. Substituting
///   one for the other is a no-op on an untrimmed, unscaled, single-range clip and silently
///   shifts or breaks every other export.
/// - `fill_bg` is compared against `trim_ranges`, which are fractions of the *unscaled*
///   duration, so it must be computed before the `fps_scale` division.
struct WindowFrame {
    out_ts: i64,
    fill_bg: bool,
    frame: Video,
}

/// Reborrows a shared frame reference as mutable, for APIs (`data_mut`, `swscale::Context::run`
/// as the source) that only ever read through it despite requiring `&mut` in their signature.
///
/// Needed because the temporal window used for denoising hands out `&Video` for the centre
/// frame and its references alike - the centre frame doubles as a reference for its
/// neighbours, so no genuine `&mut` can coexist with those shared borrows. Every call site
/// here only *reads* through the result; the window's contract (no `push` while an `Emission`
/// is alive) is what makes that sound despite going through `&mut`.
///
/// `rustc`'s `invalid_reference_casting` lint flags the direct `&T as *const T as *mut T`
/// shape as always-UB regardless of usage, so this is deliberately exempted rather than
/// worked around - the point of centralising it in one function is that the actual safety
/// argument (read-only use, no concurrent mutation) lives in exactly one place to audit.
#[allow(invalid_reference_casting)]
fn as_mut_video(v: &Video) -> &mut Video {
    unsafe { &mut *(v as *const Video as *mut Video) }
}

/// Converts `src` to `target` format, into a freshly-allocated frame. Used only for
/// temporal-window reference frames that need format conversion before undistortion: unlike
/// the centre frame (which can reuse `Converter`'s single cached scratch buffer across
/// calls, since only one is ever alive at a time), every reference has to keep its own
/// independent buffer alive simultaneously alongside the centre and every other reference -
/// reusing one shared scratch buffer across them would make each conversion silently
/// overwrite the last.
///
/// Not colour-managed to the same degree as `Converter::convert_pixel_format_color_aware`
/// (which additionally sets swscale's colourspace/range coefficients): reference frames only
/// feed the denoise gather kernel's averaging, never the visible output directly, so a small
/// colour mismatch there is judged an acceptable simplification versus threading that whole
/// mechanism through a one-off per-reference conversion.
fn convert_frame_to_format(src: &Video, target: Pixel, interpolation: ffmpeg_next::software::scaling::flag::Flags, full_range: bool) -> Result<Video, FFmpegError> {
    if src.format() == target {
        return Ok(src.clone());
    }
    let mut out = Video::new(target, src.width(), src.height());
    if full_range {
        out.set_color_range(ffmpeg_next::util::color::Range::JPEG);
    } else {
        out.set_color_range(src.color_range());
    }
    let mut conv = ffmpeg_next::software::scaling::Context::get(src.format(), src.width(), src.height(), target, src.width(), src.height(), interpolation)?;
    conv.run(as_mut_video(src), &mut out)?;
    Ok(out)
}

/// Runs every plane closure over one frame (plus its temporal references) into `output_frame`.
type PlaneClosures = Vec<Box<dyn FnMut(i64, &Video, &[&Video], &[i64], &mut Video, usize, bool)>>;
fn run_planes(planes: &mut PlaneClosures, timestamp_us: i64, frame: &Video, references: &[&Video], reference_timestamps: &[i64], out_frame: &mut Video, fill_with_background: bool) {
    for (i, cb) in planes.iter_mut().enumerate() {
        (*cb)(timestamp_us, frame, references, reference_timestamps, out_frame, i, fill_with_background);
    }
}

/// The per-frame dispatch that used to live inline as the `undistort_frame` closure plus the
/// `color_convert_to`/format-match block around it. Pulled out to a free function, parameterised
/// over everything it used to capture, so both the live decode path (`on_frame`) and the
/// temporal window's drain (`on_finish`, run once decoding has already finished) can call the
/// exact same logic on a centre frame plus its references.
#[allow(clippy::too_many_arguments)]
fn process_and_undistort(
    planes: &mut PlaneClosures,
    converter: &mut ffmpeg_video_converter::Converter,
    color_convert_to: Option<Pixel>,
    output_range: OutputRange,
    output_color_space: OutputColorSpace,
    is_prores_videotoolbox: bool,
    ffmpeg_interpolation: ffmpeg_next::software::scaling::flag::Flags,
    timestamp_us: i64,
    frame: &Video,
    references: &[&Video],
    reference_timestamps: &[i64],
    output_frame: &mut Video,
    fill_with_background: bool,
) -> Result<(), FFmpegError> {
    if let Some(target) = color_convert_to {
        set_output_range(output_frame, output_range);
        tag_limited_range_for_prores_videotoolbox(output_frame, is_prores_videotoolbox);
        tag_output_color_space(output_frame, output_color_space);
        let mut owned_refs: Vec<Video> = Vec::with_capacity(references.len());
        for r in references {
            owned_refs.push(convert_frame_to_format(r, target, ffmpeg_interpolation, true)?);
        }
        let ref_slices: Vec<&Video> = owned_refs.iter().collect();
        converter.convert_pixel_format_color_aware(as_mut_video(frame), output_frame, target, ffmpeg_interpolation, |converted_frame, converted_output| {
            run_planes(planes, timestamp_us, converted_frame, &ref_slices, reference_timestamps, converted_output, fill_with_background);
        })?;
    } else {
        match frame.format() {
            Pixel::VIDEOTOOLBOX | // Pixel::D3D11 |
            Pixel::NV12 | Pixel::NV21 | Pixel::YUV420P | Pixel::YUVJ420P |
            Pixel::P010LE | Pixel::P016LE | Pixel::P210LE | Pixel::P216LE | Pixel::P410LE | Pixel::P416LE |
            Pixel::YUV420P10LE | Pixel::YUV420P12LE | Pixel::YUV420P14LE | Pixel::YUV420P16LE |
            Pixel::YUV422P10LE | Pixel::YUV422P12LE | Pixel::YUV422P14LE | Pixel::YUV422P16LE |
            Pixel::YUV444P10LE | Pixel::YUV444P12LE | Pixel::YUV444P14LE | Pixel::YUV444P16LE |
            Pixel::YUVA444P10LE | Pixel::YUVA444P12LE | Pixel::YUVA444P16LE |
            Pixel::AYUV64LE | Pixel::GBRAPF32LE | Pixel::GBRPF32LE |
            Pixel::RGB24 | Pixel::RGBA | Pixel::RGB48BE | Pixel::RGBA64BE => {
                run_planes(planes, timestamp_us, frame, references, reference_timestamps, output_frame, fill_with_background);
            },
            _ => {
                let mut owned_refs: Vec<Video> = Vec::with_capacity(references.len());
                for r in references {
                    owned_refs.push(convert_frame_to_format(r, Pixel::YUV444P16LE, ffmpeg_interpolation, false)?);
                }
                let ref_slices: Vec<&Video> = owned_refs.iter().collect();
                converter.convert_pixel_format(as_mut_video(frame), output_frame, Pixel::YUV444P16LE, ffmpeg_interpolation, |converted_frame, converted_output| {
                    run_planes(planes, timestamp_us, converted_frame, &ref_slices, reference_timestamps, converted_output, fill_with_background);
                })?;
            }
        }
    }
    Ok(())
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
    // How many past/future neighbours each output frame is denoised against. Computed from
    // the *input* frame size (not the output/render size) because that's what the cost of
    // gathering references scales with. `0` means denoising is off - the window below then
    // emits every frame immediately, on push, with no reference gathering at all.
    let denoise_k = {
        let (w, h) = stab.params.read().size;
        stab.denoise.read().radius_for_size(w, h)
    };

    let mut gpu_decoding = stab.gpu_decoding.load(std::sync::atomic::Ordering::SeqCst);
    if denoise_k > 0 && gpu_decoding {
        // Ruling 12: reference upload (Task 7) only handles `BufferSource::Cpu`. The
        // zero-copy GPU-interop sources (D3D11, CUDA, Metal, Vulkan) log and skip
        // references, which would make denoising silently do nothing on the majority of
        // exports (GPU decoding is the default) - the user waits through a whole render and
        // gets an unchanged file back. Forcing CPU decoding here is a modest extra cost
        // (denoising already pays for a multi-frame gather) in exchange for the feature
        // actually working.
        log::info!("Denoising requires CPU-decoded frames (radius {denoise_k}); forcing software decoding for this render.");
        gpu_decoding = false;
    }
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

    // Shared between the live decode path (`on_frame`) and the drain of the temporal
    // window's tail (`on_finish`), which runs as a separate closure after decoding has
    // already finished - see `process_and_undistort`.
    let planes: Rc<RefCell<PlaneClosures>> = Rc::new(RefCell::new(Vec::new()));

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
    // Same reasoning, and the same gate: converting range is only possible on the baking path,
    // because that is the only one that produces a full-range RGB intermediate for swscale to
    // convert *from*. See `set_output_range`.
    let output_range = stab.color.read().output_range;
    // The packed RGB format the source must be converted to for the colour stage, decided
    // once when the planes are built and reused by the per-frame dispatch below, so the two
    // cannot disagree about the layout the closures were built for. `Cell` (not a plain
    // `Option`) for the same reason `planes` is behind an `Rc<RefCell<_>>`: `on_finish`'s
    // drain needs to read the value the first `on_frame` call settled on.
    let color_convert_to: Rc<std::cell::Cell<Option<Pixel>>> = Rc::new(std::cell::Cell::new(None));

    let is_prores_videotoolbox = proc.video_codec.as_deref() == Some("prores_videotoolbox");

    let progress2 = progress.clone();
    let progress_finish = progress.clone();
    // `Rc<Cell<_>>` rather than a plain local: the drain (`on_finish`) needs to keep counting
    // from wherever the live decode path left off, and to remain accurate for the progress
    // report on both sides, it counts *emitted* frames (see the `on_frame`/`on_finish` bodies
    // below) rather than decoded ones - with output delayed by `denoise_k`, counting decoded
    // frames would run `denoise_k` ahead and finish short of 100%.
    let process_frame = Rc::new(std::cell::Cell::new(0usize));
    if let Some(i) = trim_range_ind {
        for x in 0..i {
            let x = org_trim_ranges[x];
            process_frame.set(process_frame.get() + ((x.1 - x.0) * total_frame_count as f64).round() as usize);
        }
    }

    // Delays output by `denoise_k` frames so each denoised frame sees both its past and
    // future neighbours; see `temporal_window`. At `denoise_k == 0` the window is bypassed
    // entirely (see `on_frame`), so denoising off costs neither a clone nor a delay.
    let window: Rc<RefCell<temporal_window::TemporalWindow<WindowFrame>>> = Rc::new(RefCell::new(temporal_window::TemporalWindow::new(denoise_k)));
    let window_finish = window.clone();
    let planes_finish = planes.clone();
    let color_convert_to_finish = color_convert_to.clone();
    let process_frame_finish = process_frame.clone();
    // Owns the buffer the drain writes denoised pixels into before encoding. Deliberately
    // *not* `vt.buffers.output_frame_pre`/`output_frame_post` (the live path's buffer):
    // `encode_extra_frame` needs `&mut VideoTranscoder` to reach the encoder, and that
    // buffer is itself a field of `VideoTranscoder` - passing a borrow derived from `vt`
    // back into a method that also wants `&mut vt` doesn't borrow-check. A fully independent
    // owned frame avoids that self-borrow; it's (re)sized to match the live buffer's
    // format/dimensions the first time the drain runs.
    let drain_output_frame: Rc<RefCell<Option<Video>>> = Rc::new(RefCell::new(None));

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
        // Computed before the `fps_scale` division below, because `trim_ranges` are
        // fractions of the unscaled duration.
        let fill_with_background = compute_fill_with_background(render_options.pad_with_black, &trim_ranges, duration_ms, timestamp_us);

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
                process_frame.set(process_frame.get() + 1);
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
                    let denoise = stab.denoise.clone();
                    $planes.borrow_mut().push(Box::new(move |timestamp_us: i64, in_frame_data: &Video, references: &[&Video], reference_timestamps: &[i64], out_frame_data: &mut Video, plane_index: usize, fill_with_background: bool| {
                        let mut g = render_globals.borrow_mut();
                        let wgpu_format = $t::wgpu_format().map(|x| x.0);
                        let timestamp_us = timestamp_us + offset_us;

                        let mut buffers = Buffers {
                            input:  get_plane_buffer_ref(in_frame_data, in_size, plane_index, &mut g, wgpu_format),
                            output: get_plane_buffer(out_frame_data, out_size, plane_index, &mut g, wgpu_format),
                            references: references.iter().map(|r| {
                                let ref_size = zero_copy::get_plane_size(*r, plane_index);
                                get_plane_buffer_ref(*r, ref_size, plane_index, &mut g, wgpu_format)
                            }).collect(),
                        };

                        if plane.initialized_backend.is_none() || plane.pending_device_change.is_some() {
                            plane.ensure_ready_for_processing::<$t>(timestamp_us, None, &mut buffers);
                            plane.stab_data.clear();
                        }
                        let mut transform = plane.get_frame_transform_with_references::<$t>(timestamp_us, reference_timestamps, None, &buffers);
                        transform.kernel_params.pixel_value_limit = $max_val;
                        transform.kernel_params.max_pixel_value = $max_val;
                        if plane.initialized_backend.is_wgpu() && $t::wgpu_format().map(|x| x.2).unwrap_or_default() {
                            transform.kernel_params.pixel_value_limit = 1.0;
                            transform.kernel_params.max_pixel_value = 1.0;
                        }
                        transform.kernel_params.plane_index = plane_index as i32;
                        transform.kernel_params.denoise_sigma = denoise.read().sigma_for_plane(plane_index as i32);
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

        if planes.borrow().is_empty() {
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
                    color_convert_to.set(packed_color_format(format));
                    if color_convert_to.get().is_none() && !matches!(format, Pixel::RGB24 | Pixel::RGBA) {
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

            if let Some(target) = color_convert_to.get() {
                // Baking: the colour stage grades R, G and B of one pixel at once, so the
                // frame it sees must be packed RGB. `packed_color_format` picks the target
                // and guarantees it is one of these two.
                ::log::info!("Baking colour adjustments: converting {:?} to {:?}", format, target);
                // Must happen before the first conversion: the swscale contexts read the
                // frame's range when they are built, and they are cached from then on.
                set_output_range(output_frame, output_range);
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
        if planes.borrow().is_empty() {
            return Err(FFmpegError::UnknownPixelFormat(input_frame.format()));
        }

        if denoise_k == 0 {
            // Denoising off: render the frame as it arrives, exactly as before this feature
            // existed. No window, and - importantly - no `input_frame.clone()`, which would
            // otherwise cost a full `av_frame_get_buffer` + `av_frame_copy` (~12 MB/frame at
            // 4K 8-bit 4:2:0) on every export for nothing, and cannot copy a hardware frame
            // at all. `rate_control.out_timestamp_us` keeps the value `ffmpeg_video.rs`
            // computed for it.
            process_and_undistort(
                &mut planes.borrow_mut(),
                converter,
                color_convert_to.get(),
                output_range,
                output_color_space,
                is_prores_videotoolbox,
                ffmpeg_interpolation,
                timestamp_us,
                input_frame,
                &[],
                &[],
                output_frame,
                fill_with_background,
            )?;
            process_frame.set(process_frame.get() + 1);
            progress2((process_frame.get() as f64 / render_frame_count as f64, process_frame.get(), render_frame_count, false, false));
            return Ok(());
        }

        // Delayed by `denoise_k` frames so each output frame sees both its past and future
        // neighbours (see `temporal_window`). `out_ts` and `fill_bg` are captured here and
        // restored on emission - see `WindowFrame` for why recomputing them from the centre
        // timestamp is wrong.
        let item = WindowFrame { out_ts: rate_control.out_timestamp_us, fill_bg: fill_with_background, frame: input_frame.clone() };
        match window.borrow_mut().push(timestamp_us, item) {
            Some(emission) => {
                rate_control.out_timestamp_us = emission.centre.out_ts;
                let references: Vec<&Video> = emission.references.iter().map(|r| &r.frame).collect();
                process_and_undistort(
                    &mut planes.borrow_mut(),
                    converter,
                    color_convert_to.get(),
                    output_range,
                    output_color_space,
                    is_prores_videotoolbox,
                    ffmpeg_interpolation,
                    emission.centre_ts,
                    &emission.centre.frame,
                    &references,
                    &emission.reference_timestamps,
                    output_frame,
                    emission.centre.fill_bg,
                )?;
                process_frame.set(process_frame.get() + 1);
                progress2((process_frame.get() as f64 / render_frame_count as f64, process_frame.get(), render_frame_count, false, false));
            }
            None => {
                // The window is still filling: nothing new was written into `output_frame`
                // this call, so suppress the encode for this decoded frame - the same lever
                // the frame-rate conversion path above uses to skip a frame.
                rate_control.repeat_times = 0;
            }
        }

        Ok(())
    });

    proc.on_finish(move |vt, rate_control, octx, ost_time_bases| {
        // The last `denoise_k` frames are still in the window's ring: without this they are
        // silently dropped and the output is `denoise_k` frames short of the input - a
        // failure that is very easy to miss, since the file still plays fine.
        let mut result = Ok(());
        window_finish.borrow_mut().drain_each(|emission| {
            if result.is_err() { return; }
            let outcome = (|| -> Result<(), FFmpegError> {
                let mut buf_cell = drain_output_frame.borrow_mut();
                if buf_cell.is_none() {
                    let template = vt.output_frame_template().ok_or(FFmpegError::FrameEmpty)?;
                    let mut nf = Video::new(template.format(), template.width(), template.height());
                    nf.set_color_range(template.color_range());
                    *buf_cell = Some(nf);
                }
                let output_frame = buf_cell.as_mut().unwrap();

                rate_control.out_timestamp_us = emission.centre.out_ts;
                rate_control.repeat_times = 1;
                let references: Vec<&Video> = emission.references.iter().map(|r| &r.frame).collect();
                process_and_undistort(
                    &mut planes_finish.borrow_mut(),
                    &mut vt.converter,
                    color_convert_to_finish.get(),
                    output_range,
                    output_color_space,
                    is_prores_videotoolbox,
                    ffmpeg_interpolation,
                    emission.centre_ts,
                    &emission.centre.frame,
                    &references,
                    &emission.reference_timestamps,
                    output_frame,
                    emission.centre.fill_bg,
                )?;
                vt.encode_extra_frame(output_frame, rate_control, octx, ost_time_bases)?;
                process_frame_finish.set(process_frame_finish.get() + 1);
                progress_finish((process_frame_finish.get() as f64 / render_frame_count as f64, process_frame_finish.get(), render_frame_count, false, false));
                Ok(())
            })();
            if let Err(e) = outcome {
                result = Err(e);
            }
        });
        result
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

/// What the pixel format warning in Export settings needs to know, decided before any render.
///
/// The check the renderer performs at `ffmpeg_video.rs` - "is the frame's format in the encoder's
/// `pix_fmts`?" - needs nothing that only exists mid-render: the list is a static property of the
/// encoder, read straight out of ffmpeg. Running it up front is what lets the answer be a warning
/// beside the codec picker rather than a modal with twenty-one buttons after the export has
/// already started and failed.
///
/// `source_format` is the raw ffmpeg name from the file (`stream.video[0].codec.format_name`),
/// e.g. `"yuv422p10le"`. Fields in the returned object:
///
/// - `ok` - the export can run as configured. Everything below is then absent.
/// - `encoder` - which encoder the codec/GPU choice actually resolves to, for the message.
/// - `source` - the source format, echoed back so QML does not have to normalise it.
/// - `suggestion` - the format to encode as instead, ready to put in `RenderOptions::pixel_format`.
/// - `same_quality` - the suggestion keeps the source's bit depth *and* chroma subsampling, so
///   accepting it costs nothing. False means it is still the best available, but it does lose
///   something, and the message has to say so.
/// - `cpu_only` - no format works on this encoder at all; the only way out is CPU encoding.
///
/// `ok` is also returned when the source format is unknown to ffmpeg or the encoder cannot be
/// resolved. Neither is a reason to warn: there is nothing to suggest, and the runtime dialog is
/// still there to catch it.
/// The answer, as plain data. A struct rather than a JSON value so the decision can be tested
/// without Qt in the picture; `render_queue.rs` does the conversion for QML.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct PixelFormatCheck {
    pub ok: bool,
    pub encoder: String,
    pub source: String,
    pub suggestion: String,
    pub same_quality: bool,
    pub cpu_only: bool,
}

pub fn check_export_pixel_format(codec: &str, gpu: bool, source_format: &str) -> PixelFormatCheck {
    use std::str::FromStr;
    let ok = PixelFormatCheck { ok: true, ..Default::default() };

    let Ok(source) = Pixel::from_str(&source_format.to_ascii_lowercase()) else {
        ::log::debug!("check_export_pixel_format: unknown source format {source_format:?}");
        return ok;
    };

    let encoder_name = get_default_encoder(codec, gpu);
    let Some(enc) = ffmpeg_next::encoder::find_by_name(&encoder_name) else { return ok; };
    let supported = unsafe {
        if enc.as_ptr().is_null() { return ok; }
        ffmpeg_hw::pix_formats_to_vec((*enc.as_ptr()).pix_fmts)
    };
    // An encoder that advertises nothing takes anything as far as this check is concerned - the
    // renderer's own test is `!supported.contains(..)`, which is false for an empty list too.
    if supported.is_empty() { return ok; }

    // The advertised list is necessary but not sufficient: it comes from the ffmpeg build, while
    // the refusal comes from the hardware. Both questions have to be asked, in that order - the
    // list is free, the probe is not.
    if supported.contains(&source) && ffmpeg_hw::encoder_accepts_format(&encoder_name, source) { return ok; }

    // Rank, probe the winner, and on refusal drop it and rank again. Ranking the whole list
    // through the probe instead would mean opening an encoder session per candidate - twenty-one
    // of them for NVENC, at ~100 ms each, on the UI thread. In practice this settles in one or
    // two rounds; the cap is there so a thoroughly broken encoder cannot spin.
    let mut remaining = supported.clone();
    let mut suggestion = None;
    for _ in 0..4 {
        let Some(candidate) = ffmpeg_hw::find_best_matching_codec(source, &remaining) else { break };
        if ffmpeg_hw::encoder_accepts_format(&encoder_name, candidate) {
            suggestion = Some(candidate);
            break;
        }
        remaining.retain(|&f| f != candidate);
    }
    ::log::debug!("check_export_pixel_format: {encoder_name} cannot take {source:?}, suggesting {suggestion:?}");

    PixelFormatCheck {
        ok:           false,
        encoder:      encoder_name,
        source:       format!("{source:?}").to_ascii_lowercase(),
        suggestion:   suggestion.map(|f| format!("{f:?}").to_ascii_lowercase()).unwrap_or_default(),
        same_quality: suggestion.map(|f| same_visual_quality(source, f)).unwrap_or(false),
        cpu_only:     suggestion.is_none(),
    }
}

/// Whether swapping `from` for `to` keeps every bit the source carried: same component depth and
/// same chroma subsampling. Layout differences (planar versus semi-planar) do not count - the
/// pixels are identical, only the memory order differs.
fn same_visual_quality(from: Pixel, to: Pixel) -> bool {
    unsafe {
        let (a, b) = (ffi::av_pix_fmt_desc_get(from.into()), ffi::av_pix_fmt_desc_get(to.into()));
        if a.is_null() || b.is_null() { return false; }
        (*a).comp[0].depth   == (*b).comp[0].depth &&
        (*a).log2_chroma_w   == (*b).log2_chroma_w &&
        (*a).log2_chroma_h   == (*b).log2_chroma_h
    }
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
mod pixel_format_check_tests {
    use super::*;
    use ffmpeg_hw::find_best_matching_codec;

    /// What `hevc_nvenc` reports on the machine this was written against, verbatim from
    /// `Codec formats:` in gyroflow.log. Hardcoded so the ranking tests below are the same
    /// everywhere, including on a machine with no NVIDIA GPU at all.
    const NVENC_HEVC: &[Pixel] = &[
        Pixel::YUV420P, Pixel::NV12, Pixel::P010LE, Pixel::YUV444P, Pixel::P016LE, Pixel::NV16,
        Pixel::P210LE, Pixel::P216LE, Pixel::YUV444P16LE, Pixel::BGRA, Pixel::RGBA,
        Pixel::X2RGB10LE, Pixel::X2BGR10LE, Pixel::GBRP, Pixel::GBRP16LE, Pixel::CUDA, Pixel::D3D11,
    ];

    /// The case the warning exists for: Sony 4:2:2 10-bit into the GPU H.265 encoder. `P210LE` is
    /// the same layout with interleaved chroma, so the substitution is free - which is exactly
    /// what makes it worth offering up front instead of a twenty-one button dialog after failure.
    #[test]
    fn a_422_10bit_source_is_offered_the_free_substitution() {
        let picked = find_best_matching_codec(Pixel::YUV422P10LE, NVENC_HEVC);
        assert_eq!(picked, Some(Pixel::P210LE));
        assert!(same_visual_quality(Pixel::YUV422P10LE, Pixel::P210LE), "P210LE should cost nothing");
    }

    #[test]
    fn a_supported_format_is_returned_unchanged() {
        for f in [Pixel::YUV420P, Pixel::P010LE, Pixel::NV12] {
            assert_eq!(find_best_matching_codec(f, NVENC_HEVC), Some(f));
        }
    }

    /// Depth outranks chroma. A 10-bit 4:2:2 source offered a choice between "same chroma, 8 bit"
    /// and "same depth, 4:2:0" must take the depth: chroma softening is recoverable-looking on
    /// real footage, two lost bits in a gradient are not.
    #[test]
    fn keeping_bit_depth_beats_keeping_chroma() {
        let supported = [Pixel::YUV422P, Pixel::YUV420P10LE];
        assert_eq!(find_best_matching_codec(Pixel::YUV422P10LE, &supported), Some(Pixel::YUV420P10LE));
    }

    /// ...and when depth cannot be kept exactly, round up rather than down.
    #[test]
    fn depth_rounds_up_when_it_cannot_be_matched() {
        let supported = [Pixel::YUV420P, Pixel::YUV420P12LE];
        assert_eq!(find_best_matching_codec(Pixel::YUV420P10LE, &supported), Some(Pixel::YUV420P12LE));
    }

    /// An encoder advertises its hardware surfaces and its RGB layouts alongside real formats.
    /// Neither is a substitute for a YUV source: `CUDA` is not a pixel layout, and `BGRA` would
    /// silently insert a colour conversion. With nothing else on offer the honest answer is none,
    /// which is what drives the "CPU encoding" wording in the warning.
    #[test]
    fn hardware_and_rgb_formats_are_not_offered_for_a_yuv_source() {
        for supported in [
            vec![Pixel::CUDA, Pixel::D3D11],
            vec![Pixel::BGRA, Pixel::RGBA, Pixel::X2RGB10LE],
            vec![Pixel::CUDA, Pixel::BGRA],
        ] {
            assert_eq!(find_best_matching_codec(Pixel::YUV422P10LE, &supported), None, "{supported:?}");
        }
    }

    #[test]
    fn an_encoder_that_advertises_nothing_yields_nothing() {
        assert_eq!(find_best_matching_codec(Pixel::YUV422P10LE, &[]), None);
    }

    /// `same_visual_quality` is what decides whether the message promises the swap is free, so it
    /// has to be strict about both axes, and indifferent to memory layout.
    #[test]
    fn same_visual_quality_compares_depth_and_chroma_only() {
        assert!(same_visual_quality(Pixel::YUV420P10LE, Pixel::P010LE));   // planar vs semi-planar
        assert!(same_visual_quality(Pixel::YUV420P,     Pixel::NV12));
        assert!(!same_visual_quality(Pixel::YUV422P10LE, Pixel::P010LE));  // 4:2:2 -> 4:2:0
        assert!(!same_visual_quality(Pixel::YUV420P10LE, Pixel::YUV420P)); // 10 bit -> 8 bit
    }

    /// A format string ffmpeg does not recognise must not produce a warning: there is nothing to
    /// suggest, and a scary message about a file we failed to parse is worse than silence.
    #[test]
    fn an_unrecognised_source_format_never_warns() {
        for s in ["", "not_a_format", "yuv422p10le_typo"] {
            assert!(check_export_pixel_format("H.265/HEVC", true, s).ok, "{s:?} produced a warning");
        }
    }

    /// The probe has to be able to say yes as well as no, or the warning would decide every
    /// encoder is useless and always recommend CPU. libx264 is present in every build here and
    /// takes yuv420p; nothing takes a planar float layout, and it is not in x264's list either.
    #[test]
    fn the_probe_distinguishes_a_usable_format_from_an_impossible_one() {
        if ffmpeg_next::encoder::find_by_name("libx264").is_none() { return; }
        assert!( ffmpeg_hw::encoder_accepts_format("libx264", Pixel::YUV420P));
        assert!(!ffmpeg_hw::encoder_accepts_format("libx264", Pixel::GBRPF32LE));
    }

    /// A name ffmpeg has never heard of must be a clean `false`, not a panic or a null deref -
    /// `get_default_encoder` can return one when a codec has no encoder on this machine.
    #[test]
    fn probing_an_encoder_that_does_not_exist_is_false() {
        assert!(!ffmpeg_hw::encoder_accepts_format("no_such_encoder", Pixel::YUV420P));
        assert!(!ffmpeg_hw::encoder_accepts_format("", Pixel::YUV420P));
    }

    /// Machine-independent invariant, run against whatever encoders this machine really has: if
    /// the check says no, the way out it offers must actually exist. Suggesting a format the
    /// encoder cannot take would replace one failed export with another.
    #[test]
    fn any_suggestion_is_one_the_resolved_encoder_accepts() {
        for codec in ["H.264/AVC", "H.265/HEVC", "AV1", "ProRes", "DNxHD", "PNG Sequence"] {
            for gpu in [true, false] {
                for source in ["yuv422p10le", "yuv420p", "yuv444p12le", "gbrpf32le"] {
                    let c = check_export_pixel_format(codec, gpu, source);
                    if c.ok { continue; }

                    assert_eq!(c.suggestion.is_empty(), c.cpu_only, "{codec} gpu={gpu} {source}: cpu_only must mean no suggestion");
                    if c.cpu_only { continue; }

                    let enc = ffmpeg_next::encoder::find_by_name(&c.encoder).expect("resolved encoder must exist");
                    let supported = unsafe { ffmpeg_hw::pix_formats_to_vec((*enc.as_ptr()).pix_fmts) };
                    let suggested = supported.iter().find(|f| format!("{f:?}").to_ascii_lowercase() == c.suggestion);
                    assert!(suggested.is_some(), "{codec} gpu={gpu} {source}: suggested {} which {} does not advertise", c.suggestion, c.encoder);

                    // The stronger half, and the one that matters: advertised is not the same as
                    // usable. This is what stops the warning offering P210LE on a card with no
                    // 4:2:2 encode block.
                    assert!(ffmpeg_hw::encoder_accepts_format(&c.encoder, *suggested.unwrap()),
                        "{codec} gpu={gpu} {source}: suggested {} which {} advertises but will not open with", c.suggestion, c.encoder);
                }
            }
        }
    }
}

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
