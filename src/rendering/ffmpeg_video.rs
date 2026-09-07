// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2021-2022 Adrian <adrian.eddy at gmail>

use ffmpeg_next::{ ffi, codec, decoder, encoder, format, frame, picture, software, util, Dictionary, Packet, Rational, Error, rescale::Rescale };

use super::ffmpeg_processor::Status;
use super::ffmpeg_processor::FFmpegError;
use super::ffmpeg_processor::FrameTimestamps;
use super::ffmpeg_video_converter::Converter;

pub struct FrameBuffers {
    pub sw_frame: frame::Video,
    pub converted_frame: frame::Video,

    pub output_frame_pre: Option<frame::Video>,
    pub output_frame_post: Option<frame::Video>,
    pub output_frame_hw: Option<frame::Video>,
}
impl Default for FrameBuffers {
    fn default() -> Self { Self {
        sw_frame: frame::Video::empty(),
        converted_frame: frame::Video::empty(),
        output_frame_pre: None,
        output_frame_post: None,
        output_frame_hw: None,
    } }
}

#[derive(Default, Eq, PartialEq, Debug)]
pub enum ProcessingOrder {
    #[default]
    PreConversion,
    PostConversion
}

#[derive(Default)]
pub struct EncoderParams<'a> {
    pub codec: Option<codec::codec::Codec>,
    pub hw_device_type: Option<ffi::AVHWDeviceType>,
    pub options: Dictionary<'a>,
    pub metadata: Dictionary<'a>,
    pub pixel_format: Option<format::Pixel>,
    pub frame_rate: Option<Rational>,
    pub time_base: Option<Rational>,
    pub keyframe_distance_s: f64,
}
#[derive(Default)]
pub struct VideoTranscoder<'a> {
    pub input_index: usize,
    pub output_index: Option<usize>,
    pub decoder: Option<decoder::Video>,
    pub encoder: Option<encoder::video::Video>,
    pub encoder_name: String,

    pub encoder_params: EncoderParams<'a>,

    pub codec_supported_formats: Vec<format::Pixel>,

    pub encoder_converter: Option<software::scaling::Context>,

    pub decode_only: bool,
    pub gpu_decoding: bool,
    pub gpu_encoding: bool,
    pub clone_frames: bool,

    pub converter: Converter,

    pub buffers: FrameBuffers,

    pub on_frame_callback: Option<Box<dyn FnMut(i64, &mut frame::Video, Option<&mut frame::Video>, &mut Converter, &mut RateControl) -> Result<(), FFmpegError> + 'a>>,
    /// Called once after the decode loop ends, so a processor holding frames back
    /// (temporal denoising delays output by `k`) can emit its tail. Without this,
    /// those frames are silently dropped and the output is `k` frames short.
    ///
    /// Takes `&mut VideoTranscoder` itself (not just `&mut Converter`) because emitting a
    /// tail frame has to reach the encoder too - `Converter` alone has no path to it. The
    /// `octx`/`ost_time_bases` parameters are the same ones `render()` already threads
    /// through `receive_and_process_video_frames`, needed here for the same reason: writing
    /// encoded packets out.
    pub on_finish_callback: Option<Box<dyn FnMut(&mut VideoTranscoder<'a>, &mut RateControl, &mut format::context::Output, &mut Vec<Rational>) -> Result<(), FFmpegError> + 'a>>,
    /// Output timestamp of the last frame `encode_extra_frame` wrote, for the caller to fold
    /// into `FrameTimestamps` - that path never sees `frame_ts` itself.
    pub last_extra_video_ts: Option<i64>,
    pub on_encoder_initialized: Option<Box<dyn FnMut(&encoder::video::Video) -> Result<(), FFmpegError> + 'a>>,

    pub processing_order: ProcessingOrder,

    pub ffmpeg_interpolation: i32,
}

pub struct RateControl {
    pub out_timestamp_us: i64,
    pub repeat_times: i64,
    pub repeat_interval: i64,
}
impl Default for RateControl { fn default() -> Self { Self { out_timestamp_us: 0, repeat_times: 1, repeat_interval: 0 } } }

macro_rules! ffmpeg {
    ($func:stmt; $err:ident) => {
        let err = unsafe { $func };
        if err < 0 { return Err(FFmpegError::$err(err)); }
    };
}

impl<'a> VideoTranscoder<'a> {
    fn init_encoder(frame: &mut frame::Video, params: &EncoderParams, decoder: &mut decoder::Video, size: (u32, u32), bitrate_mbps: Option<f64>, octx: &mut format::context::Output, output_index: usize, hw_upload_format: &Option<format::Pixel>) -> Result<encoder::video::Video, FFmpegError> {
        let global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);
        let mut ost = octx.stream_mut(output_index).unwrap();
        let encoder_codec = params.codec.unwrap();

        let options = params.options.to_owned();

        let ctx_ptr = unsafe { ffi::avcodec_alloc_context3(encoder_codec.as_ptr()) };
        let context = unsafe { codec::context::Context::wrap(ctx_ptr, Some(std::rc::Rc::new(0))) };
        let mut encoder = context.encoder().video()?;
        let codec_name = encoder.codec().map(|x| x.name().to_string()).unwrap_or_default();
        let pixel_format = params.pixel_format.unwrap_or_else(|| frame.format());
        let mut color_range = frame.color_range();

        // Workaround for a bug in prores videotoolbox encoder
        if cfg!(any(target_os = "macos", target_os = "ios")) && pixel_format == format::Pixel::NV12 && (codec_name == "prores_videotoolbox" || codec_name == "dnxhd") {
            color_range = util::color::Range::MPEG;
        }

        log::debug!("Setting output pixel format: {:?}, color range: {:?}", pixel_format, color_range);

        encoder.set_width(size.0);
        encoder.set_height(size.1);
        encoder.set_aspect_ratio(frame.aspect_ratio());
        encoder.set_format(pixel_format);
        encoder.set_frame_rate(params.frame_rate);
        encoder.set_time_base(params.time_base.unwrap());
        let bitrate = bitrate_mbps.map(|x| (x * 1024.0*1024.0) as usize).unwrap_or_else(|| decoder.bit_rate());
        encoder.set_bit_rate(bitrate);
        if !codec_name.contains("videotoolbox") {
            encoder.set_max_bit_rate(bitrate);
        }
        unsafe {
            (*encoder.as_mut_ptr()).rc_min_rate = bitrate as i64;
        }
        encoder.set_color_range(color_range);
        encoder.set_colorspace(frame.color_space());
        let gop: f64 = params.frame_rate.unwrap_or(Rational::new(30, 1)).into();
        encoder.set_gop(((gop * params.keyframe_distance_s) as u32).max(1));

        unsafe {
            if !codec_name.contains("videotoolbox") {
                (*encoder.as_mut_ptr()).color_trc = (*frame.as_ptr()).color_trc;
            }
            (*encoder.as_mut_ptr()).color_primaries = (*frame.as_ptr()).color_primaries;
        }

        if global_header {
            encoder.set_flags(codec::Flags::GLOBAL_HEADER);
        }
        if let Some(qscale) = options.get("qscale").and_then(|x| x.parse::<i32>().ok()) {
            if qscale >= 0 {
                unsafe {
                    (*encoder.as_mut_ptr()).flags |= ffi::AV_CODEC_FLAG_QSCALE as i32;
                    (*encoder.as_mut_ptr()).global_quality = ffi::FF_QP2LAMBDA * qscale;
                }
            }
        }

        log::debug!("hw_device_type {:?}", params.hw_device_type);
        if let Some(hw_type) = params.hw_device_type {
            unsafe {
                if super::ffmpeg_hw::initialize_hwframes_context(encoder.as_mut_ptr(), frame.as_mut_ptr(), hw_type, pixel_format.into(), size, hw_upload_format.is_some(), params.options.get("hwaccel_device")).is_err() {
                    super::append_log("Failed to create encoder HW context.\n");
                }
            }
        }

        let mut new_options = Dictionary::new();
        log::debug!("Encoder options: {options:?}");

        for (k, v) in options.iter() {
            if !(k == "profile" && v == "main42210") {
                new_options.set(k, v);
            }
            unsafe {
                let k = std::ffi::CString::new(k).unwrap_or_default();
                let v = std::ffi::CString::new(v).unwrap_or_default();
                ffmpeg_next::ffi::av_opt_set((*ctx_ptr).priv_data, k.as_ptr(), v.as_ptr(), 0);
            }
        }

        let encoder = encoder.open_with(new_options)?;
        ost.set_parameters(&encoder);
        let context = unsafe { codec::context::Context::wrap(ctx_ptr, None) };

        if codec_name.contains("hevc") || codec_name.contains("x265") {
            let hvc1_tag: u32 = (b'h' as u32) | ((b'v' as u32) << 8) | ((b'c' as u32) << 16) | ((b'1' as u32) << 24);
            unsafe { (*ost.parameters().as_mut_ptr()).codec_tag = hvc1_tag; }
        }

        Ok(context.encoder().video()?)
    }

    pub fn receive_and_process_video_frames(&mut self, size: (u32, u32), bitrate: Option<f64>, mut octx: Option<&mut format::context::Output>, ost_time_bases: &mut Vec<Rational>, start_ms: Option<f64>, end_ms: Option<f64>, frame_ts: &mut FrameTimestamps) -> Result<Status, FFmpegError> {
        let mut status = Status::Continue;

        let decoder = self.decoder.as_mut().ok_or(FFmpegError::DecoderNotFound)?;

        let mut frame = frame::Video::empty();
        let mut sw_frame = &mut self.buffers.sw_frame;

        while decoder.receive_frame(&mut frame).is_ok() {
            let time_base = self.encoder_params.time_base.unwrap();

            if let Some(mut ts) = frame.timestamp() {
                let timestamp_us = ts;
                let timestamp_ms = timestamp_us as f64 / 1000.0;

                if start_ms.is_none() || timestamp_ms >= start_ms.unwrap() {
                    if frame_ts.first.is_none() {
                        frame_ts.first = Some(timestamp_us);
                    }
                    ts -= frame_ts.first.unwrap();
                    ts += frame_ts.add_video;

                    let mut rate_control = RateControl {
                        out_timestamp_us: ts,
                        ..Default::default()
                    };

                    let mut hw_formats = None;
                    let input_frame =
                        if unsafe { !(*frame.as_mut_ptr()).hw_frames_ctx.is_null() } {
                            hw_formats = Some(unsafe { super::ffmpeg_hw::get_transfer_formats_from_gpu(frame.as_mut_ptr()) });
                            // log::debug!("Hardware transfer formats from GPU: {:?}", hw_formats);
                            // retrieve data from GPU to CPU
                            ffmpeg!(ffi::av_hwframe_transfer_data(sw_frame.as_mut_ptr(), frame.as_mut_ptr(), 0); FromHWTransferError);
                            ffmpeg!(ffi::av_frame_copy_props(sw_frame.as_mut_ptr(), frame.as_mut_ptr()); FromHWTransferError);
                            &mut sw_frame
                        } else {
                            &mut frame
                        };

                    if input_frame.format() == format::Pixel::YUVJ420P {
                        input_frame.set_format(format::Pixel::YUV420P);
                        input_frame.set_color_range(util::color::Range::JPEG);
                    }

                    if !self.decode_only {
                        if self.encoder_name.is_empty() {
                            self.encoder_name = self.encoder_params.codec.map(|x| x.name().to_string()).unwrap_or_default();
                        }

                        // Videotoolbox doesn't support YUV420P, Use NV12 instead
                        if self.encoder_name.contains("videotoolbox") && input_frame.format() == format::Pixel::YUV420P {
                            self.encoder_params.pixel_format = Some(format::Pixel::NV12);
                            self.processing_order = ProcessingOrder::PostConversion;
                        }
                        if self.encoder_name == "hevc_videotoolbox" && input_frame.format() == format::Pixel::P210LE {
                            self.encoder_params.options.set("profile", "main42210");
                        }

                        if input_frame.format() == format::Pixel::RGB24 || input_frame.format() == format::Pixel::RGB48 {
                            self.processing_order = ProcessingOrder::PostConversion;
                        }

                        if self.processing_order == ProcessingOrder::PreConversion && self.buffers.output_frame_pre.is_none()  {
                            let mut out_frame = frame::Video::new(input_frame.format(), size.0, size.1);
                            unsafe { Self::copy_frame_props(out_frame.as_mut_ptr(), input_frame.as_ptr()) }
                            self.buffers.output_frame_pre = Some(out_frame);
                        }
                    }

                    // Process frame
                    if self.decode_only || self.processing_order == ProcessingOrder::PreConversion {
                        if let Some(ref mut cb) = self.on_frame_callback {
                            cb(timestamp_us, input_frame, self.buffers.output_frame_pre.as_mut(), &mut self.converter, &mut rate_control)?;
                        }
                    }

                    let mut hw_upload_format = None;

                    // Encode output frame
                    if !self.decode_only {
                        let in_format = input_frame.format();
                        let mut final_frame = if self.processing_order == ProcessingOrder::PreConversion {
                            self.buffers.output_frame_pre.as_mut().unwrap()
                        } else {
                            input_frame
                        };

                        if self.gpu_decoding && self.encoder_params.pixel_format.is_none() {
                            log::debug!("Hardware transfer formats from GPU: {:?}", hw_formats);
                            if let Some(hw_formats) = &hw_formats {
                                if !hw_formats.is_empty() {
                                    let dl_format = *hw_formats.first().ok_or(FFmpegError::NoHWTransferFormats)?;
                                    let picked = super::ffmpeg_hw::find_best_matching_codec(dl_format, &self.codec_supported_formats)
                                        .unwrap_or_else(|| *self.codec_supported_formats.first().unwrap_or(&format::Pixel::None));
                                    if super::ffmpeg_hw::is_hardware_format(picked.into()) {
                                        hw_upload_format = Some(picked);
                                        self.encoder_params.pixel_format = Some(dl_format);
                                    } else if picked != format::Pixel::None {
                                        self.encoder_params.pixel_format = Some(picked);
                                    }
                                }
                            }
                        }
                        if self.codec_supported_formats.len() == 1 {
                            if let Some(first_format) = self.codec_supported_formats.first() {
                                if super::ffmpeg_hw::is_hardware_format(first_format.clone().into()) {
                                    hw_upload_format = Some(*first_format);
                                }
                            }
                        }

                        let mut target_format = self.encoder_params.pixel_format.unwrap_or(in_format);
                        if super::ffmpeg_hw::is_hardware_format(target_format.into()) {
                            let sw_format = if self.gpu_decoding {
                                if let Some(hw_formats) = &hw_formats {
                                    *hw_formats.first().ok_or(FFmpegError::NoHWTransferFormats)?
                                } else {
                                    in_format
                                }
                            } else {
                                in_format
                            };
                            hw_upload_format = Some(target_format);
                            target_format = sw_format;
                            self.encoder_params.pixel_format = Some(target_format);
                        }

                        if in_format != target_format {
                            if self.encoder_converter.is_none() {
                                log::debug!("Converting from {:?} to {:?}", final_frame.format(), target_format);
                                self.buffers.converted_frame = frame::Video::new(target_format, final_frame.width(), final_frame.height());

                                unsafe { Self::copy_frame_props(self.buffers.converted_frame.as_mut_ptr(), final_frame.as_ptr()) }
                                let mut conv = software::scaling::Context::get(
                                    final_frame.format(), // input
                                    final_frame.width(),
                                    final_frame.height(),
                                    self.buffers.converted_frame.format(), // output
                                    self.buffers.converted_frame.width(),
                                    self.buffers.converted_frame.height(),
                                    software::scaling::flag::Flags::from_bits_truncate(self.ffmpeg_interpolation),
                                )?;

                                unsafe {
                                    use std::os::raw::c_int;
                                    // let mut dummy: [c_int; 4] = [0; 4];
                                    let mut src_range: c_int = 0;
                                    let mut dst_range: c_int = 0;
                                    // let mut brightness: c_int = 0;
                                    // let mut contrast: c_int = 0;
                                    // let mut saturation: c_int = 0;
                                    // ffi::sws_getColorspaceDetails(conv.as_mut_ptr(), &mut dummy.as_mut_ptr(), &mut src_range, &mut dummy.as_mut_ptr(), &mut dst_range, &mut brightness, &mut contrast, &mut saturation);
                                    let coefs = ffi::sws_getCoefficients(ffi::SWS_CS_ITU709);
                                    if final_frame.color_range() == util::color::Range::JPEG {
                                        src_range |= 1;
                                    }
                                    if self.buffers.converted_frame.color_range() == util::color::Range::JPEG {
                                        dst_range |= 1;
                                    }
                                    ffi::sws_setColorspaceDetails(conv.as_mut_ptr(), coefs, src_range, coefs, dst_range, 0, 1 << 16, 1 << 16);
                                    //self.encoder.as_mut().ok_or(FFmpegError::EncoderNotFound)?.set_color_range(self.buffers.converted_frame.color_range());
                                }
                                self.encoder_converter = Some(conv);
                            }
                            let conv = self.encoder_converter.as_mut().ok_or(FFmpegError::EncoderConverterEmpty)?;
                            let buff = &mut self.buffers.converted_frame;
                            conv.run(final_frame, buff)?;
                            final_frame = buff;
                        }

                        if self.processing_order == ProcessingOrder::PostConversion {
                            if let Some(ref mut cb) = self.on_frame_callback {
                                if self.buffers.output_frame_post.is_none()  {
                                    let mut out_frame = frame::Video::new(target_format, size.0, size.1);
                                    unsafe { Self::copy_frame_props(out_frame.as_mut_ptr(), final_frame.as_ptr()) }
                                    self.buffers.output_frame_post = Some(out_frame);
                                }

                                cb(timestamp_us, final_frame, self.buffers.output_frame_post.as_mut(), &mut self.converter, &mut rate_control)?;

                                final_frame = self.buffers.output_frame_post.as_mut().unwrap();
                            }
                        }

                        if self.encoder.is_none() {
                            let octx = octx.as_deref_mut().ok_or(FFmpegError::NoOutputContext)?;

                            log::debug!("hw_device_type: {:?}, encoder_pixel_format: {:?}", self.encoder_params.hw_device_type, self.encoder_params.pixel_format);
                            let pixel_format = self.encoder_params.pixel_format.unwrap_or_else(|| final_frame.format());
                            if !self.codec_supported_formats.contains(&pixel_format) && hw_upload_format.is_none() {
                                return Err(FFmpegError::PixelFormatNotSupported((pixel_format, self.codec_supported_formats.clone(), super::ffmpeg_hw::find_best_matching_codec(pixel_format, &self.codec_supported_formats))));
                            }

                            // let mut stderr_buf  = gag::BufferRedirect::stderr().unwrap();

                            let result = Self::init_encoder(final_frame, &self.encoder_params, decoder, size, bitrate, octx, self.output_index.unwrap_or_default(), &hw_upload_format);

                            // let mut output = String::new();
                            // std::io::Read::read_to_string(stderr_buf, &mut output).unwrap();
                            // drop(stderr_buf);
                            // println!("output: {:?}", output);

                            self.encoder = Some(result?);

                            octx.write_header()?;
                            // format::context::output::dump(&octx, 0, Some(&output_path));

                            for (ost_index, _) in octx.streams().enumerate() {
                                ost_time_bases[ost_index] = octx.stream(ost_index as _).ok_or(Error::StreamNotFound)?.time_base();
                            }

                            if let Some(ref mut cb) = self.on_encoder_initialized {
                                cb(self.encoder.as_ref().unwrap())?;
                            }
                        }

                        let encoder = self.encoder.as_mut().ok_or(FFmpegError::EncoderNotFound)?;
                        encoder.set_format(final_frame.format());
                        encoder.set_color_range(final_frame.color_range());

                        ts = rate_control.out_timestamp_us;

                        let mut output_hw_frame;

                        if let Some(hw_upload_format) = hw_upload_format {
                            log::debug!("Uploading frame to the device, hw_upload_format {:?}, final_frame.format: {:?}", hw_upload_format, final_frame.format());

                            output_hw_frame = Some(frame::Video::empty());

                            // Upload back to GPU
                            unsafe {
                                let frame_ptr = output_hw_frame.as_mut().ok_or(FFmpegError::FrameEmpty)?.as_mut_ptr();
                                let err = ffi::av_hwframe_get_buffer((*encoder.as_mut_ptr()).hw_frames_ctx, frame_ptr, 0);
                                if err < 0 {
                                    return Err(FFmpegError::ToHWBufferError(err));
                                }
                                if (*frame_ptr).hw_frames_ctx.is_null() {
                                    return Err(FFmpegError::NoFramesContext);
                                }
                                let err = ffi::av_hwframe_transfer_data(frame_ptr, final_frame.as_mut_ptr(), 0);
                                if err < 0 {
                                    return Err(FFmpegError::ToHWTransferError(err));
                                }
                                Self::copy_frame_props(frame_ptr, final_frame.as_ptr());
                            }
                            final_frame = output_hw_frame.as_mut().ok_or(FFmpegError::FrameEmpty)?;
                        }

                        for _ in 0..rate_control.repeat_times {
                            let timestamp = Some(ts.rescale((1, 1000000), time_base));
                            final_frame.set_pts(timestamp);
                            final_frame.set_kind(picture::Type::None);

                            if self.clone_frames {
                                // TODO: ideally this should be a buffer pool per thread, but we need to figure out which thread ffmpeg actually used for that frame
                                encoder.send_frame(&final_frame.clone())?;
                            } else {
                                encoder.send_frame(final_frame)?;
                            }
                            ts += rate_control.repeat_interval;

                            // Copy of receive_and_process_encoded_packets
                            let ost_time_base = ost_time_bases[self.output_index.unwrap_or_default()];
                            let octx = octx.as_mut().unwrap();
                            let time_base = self.encoder_params.time_base.unwrap();//self.decoder.as_ref().ok_or(FFmpegError::DecoderNotFound)?.time_base();
                            let mut encoded = Packet::empty();
                            while encoder.receive_packet(&mut encoded).is_ok() {
                                encoded.set_stream(self.output_index.unwrap_or_default());
                                encoded.rescale_ts(time_base, ost_time_base);
                                if octx.format().name().contains("image") {
                                    encoded.write(octx)?;
                                } else {
                                    encoded.write_interleaved(octx)?;
                                }
                            }
                        }
                    }
                    if let Some(last_ts) = frame_ts.last_video {
                        frame_ts.last_duration_video = ts - last_ts;
                    }
                    frame_ts.last_video = Some(ts);
                    if end_ms.is_some() && timestamp_ms > end_ms.unwrap() {
                        status = Status::Finish;
                        break;
                    }
                }
            }
        }

        // if !self.decode_only && self.encoder.is_some() {
        //     let ost_time_base = ost_time_bases[self.output_index.unwrap_or_default()];
        //     let octx = octx.unwrap();
        //     self.receive_and_process_encoded_packets(octx, ost_time_base)?;
        // }

        Ok(status)
    }

    /// The buffer the normal decode loop writes processed pixels into before encoding -
    /// whichever of `output_frame_pre`/`output_frame_post` is live for the current
    /// `processing_order`. A delayed processor's drain hook reads this only to learn the
    /// format/size/colour tags an *extra* frame needs to match; it must not write into this
    /// buffer directly; see `encode_extra_frame`, which takes its own owned frame instead
    /// specifically to avoid aliasing this one.
    pub fn output_frame_template(&self) -> Option<&frame::Video> {
        match self.processing_order {
            ProcessingOrder::PreConversion  => self.buffers.output_frame_pre.as_ref(),
            ProcessingOrder::PostConversion => self.buffers.output_frame_post.as_ref(),
        }
    }

    /// Encodes a single already-processed frame that did not arrive through the normal
    /// decode loop - used for the trailing `k` frames a temporal window is still holding
    /// once decoding has finished (`on_finish_callback`).
    ///
    /// By the time this runs, at least one real frame has already gone through
    /// `receive_and_process_video_frames`, so the encoder exists and `encoder_params.pixel_format`
    /// / `encoder_converter` are already whatever that path settled on - this only has to
    /// match that target format and hand off to the encoder. It does not re-detect a codec,
    /// initialise the encoder, or handle hardware upload: denoising forces CPU decoding
    /// (Ruling 12), so a delayed frame is never a hardware frame.
    /// Returns whether a frame was actually encoded. `false` is not an error - there is
    /// simply no encoder to hand this to - but the caller must not count it as progress, or
    /// a render that encoded nothing would still report 100%.
    pub fn encode_extra_frame(&mut self, final_frame_in: &mut frame::Video, rate_control: &mut RateControl, octx: &mut format::context::Output, ost_time_bases: &mut Vec<Rational>) -> Result<bool, FFmpegError> {
        if self.decode_only || self.encoder.is_none() {
            // decode_only never encodes at all; encoder.is_none() means nothing was ever
            // encoded (e.g. a zero-frame input) and there is no pipeline to hand this to.
            return Ok(false);
        }

        let in_format = final_frame_in.format();
        let target_format = self.encoder_params.pixel_format.unwrap_or(in_format);
        let mut final_frame: &mut frame::Video = final_frame_in;
        if in_format != target_format {
            if self.encoder_converter.is_none() {
                let mut conv = software::scaling::Context::get(
                    final_frame.format(), final_frame.width(), final_frame.height(),
                    target_format, final_frame.width(), final_frame.height(),
                    software::scaling::flag::Flags::from_bits_truncate(self.ffmpeg_interpolation),
                )?;
                // Same coefficients the live path sets when it builds this converter. Without
                // them a drained tail frame built through this fallback would be converted
                // with swscale's BT.601/limited-range defaults while every other frame in the
                // file got 709 - a visible shift on the last k frames.
                unsafe {
                    use std::os::raw::c_int;
                    let mut src_range: c_int = 0;
                    let mut dst_range: c_int = 0;
                    let coefs = ffi::sws_getCoefficients(ffi::SWS_CS_ITU709);
                    if final_frame.color_range() == util::color::Range::JPEG { src_range |= 1; }
                    if self.buffers.converted_frame.color_range() == util::color::Range::JPEG { dst_range |= 1; }
                    ffi::sws_setColorspaceDetails(conv.as_mut_ptr(), coefs, src_range, coefs, dst_range, 0, 1 << 16, 1 << 16);
                }
                self.encoder_converter = Some(conv);
            }
            if self.buffers.converted_frame.format() != target_format || self.buffers.converted_frame.width() != final_frame.width() || self.buffers.converted_frame.height() != final_frame.height() {
                self.buffers.converted_frame = frame::Video::new(target_format, final_frame.width(), final_frame.height());
            }
            unsafe { Self::copy_frame_props(self.buffers.converted_frame.as_mut_ptr(), final_frame.as_ptr()) }
            let conv = self.encoder_converter.as_mut().ok_or(FFmpegError::EncoderConverterEmpty)?;
            let buff = &mut self.buffers.converted_frame;
            conv.run(final_frame, buff)?;
            final_frame = buff;
        }

        let time_base = self.encoder_params.time_base.unwrap();
        let mut ts = rate_control.out_timestamp_us;
        let clone_frames = self.clone_frames;
        let output_index = self.output_index.unwrap_or_default();
        // Owns the uploaded hardware frame, if one is needed, so `final_frame` can borrow it
        // for the rest of this function. The initial `None` is never read - it exists purely
        // to give the binding a lifetime that outlives the block below.
        #[allow(unused_assignments)]
        let mut output_hw_frame: Option<frame::Video> = None;
        {
            let encoder = self.encoder.as_mut().ok_or(FFmpegError::EncoderNotFound)?;
            encoder.set_format(final_frame.format());
            encoder.set_color_range(final_frame.color_range());

            // Some encoders only accept hardware frames, and that is a property of the
            // ENCODER, not of how the input was decoded - Ruling 12 forces software decoding
            // when denoising, which says nothing about what the encoder wants. Without this
            // upload a drained tail frame reaches `send_frame` as a plain software frame and
            // is rejected, failing the whole export after all the work is done. Mirrors the
            // live path's upload; keyed off the encoder's own frames context rather than
            // re-deriving the format, so it cannot disagree with what the encoder expects.
            if !unsafe { (*encoder.as_mut_ptr()).hw_frames_ctx.is_null() } {
                let mut hw = frame::Video::empty();
                unsafe {
                    let frame_ptr = hw.as_mut_ptr();
                    let err = ffi::av_hwframe_get_buffer((*encoder.as_mut_ptr()).hw_frames_ctx, frame_ptr, 0);
                    if err < 0 { return Err(FFmpegError::ToHWBufferError(err)); }
                    if (*frame_ptr).hw_frames_ctx.is_null() { return Err(FFmpegError::NoFramesContext); }
                    let err = ffi::av_hwframe_transfer_data(frame_ptr, final_frame.as_mut_ptr(), 0);
                    if err < 0 { return Err(FFmpegError::ToHWTransferError(err)); }
                    Self::copy_frame_props(frame_ptr, final_frame.as_ptr());
                }
                output_hw_frame = Some(hw);
                final_frame = output_hw_frame.as_mut().ok_or(FFmpegError::FrameEmpty)?;
            }

            for _ in 0..rate_control.repeat_times.max(1) {
                let timestamp = Some(ts.rescale((1, 1000000), time_base));
                final_frame.set_pts(timestamp);
                final_frame.set_kind(picture::Type::None);
                if clone_frames {
                    encoder.send_frame(&final_frame.clone())?;
                } else {
                    encoder.send_frame(final_frame)?;
                }
                ts += rate_control.repeat_interval;
            }
        }
        // The live path updates `frame_ts.last_video` after every encode; this path cannot
        // reach `frame_ts`, so it records the last timestamp it wrote and the caller folds it
        // in. Without that, a trim seam computes `add_video` from the last *live* frame and
        // the next range's timestamps collide with the frames the drain just emitted - the
        // muxer silently drops one frame per collision.
        self.last_extra_video_ts = Some(ts - rate_control.repeat_interval);

        let ost_time_base = ost_time_bases[output_index];
        self.receive_and_process_encoded_packets(octx, ost_time_base)?;
        Ok(true)
    }

    pub fn receive_and_process_encoded_packets(&mut self, octx: &mut format::context::Output, ost_time_base: Rational) -> Result<(), FFmpegError> {
        if !self.decode_only {
            let time_base = self.encoder_params.time_base.unwrap();//self.decoder.as_ref().ok_or(FFmpegError::DecoderNotFound)?.time_base();
            let mut encoded = Packet::empty();
            while self.encoder.as_mut().ok_or(FFmpegError::EncoderNotFound)?.receive_packet(&mut encoded).is_ok() {
                encoded.set_stream(self.output_index.unwrap_or_default());
                encoded.rescale_ts(time_base, ost_time_base);
                if octx.format().name().contains("image") {
                    encoded.write(octx)?;
                } else {
                    encoded.write_interleaved(octx)?;
                }
            }
        }
        Ok(())
    }

    /*fn get_format_range(format: format::Pixel) -> (bool, format::Pixel) {
        match format {
            format::Pixel::YUVJ420P => (true, format::Pixel::YUV420P),
            format::Pixel::YUVJ411P => (true, format::Pixel::YUV411P),
            format::Pixel::YUVJ422P => (true, format::Pixel::YUV422P),
            format::Pixel::YUVJ444P => (true, format::Pixel::YUV444P),
            format::Pixel::YUVJ440P => (true, format::Pixel::YUV440P),
            format::Pixel::GRAY8 |
            format::Pixel::YA8 |
            format::Pixel::GRAY16LE |
            format::Pixel::GRAY16BE |
            format::Pixel::YA16BE |
            format::Pixel::YA16LE => (true, format),
            _ => (false, format)
        }
    }*/

    pub(crate) unsafe fn copy_frame_props(dst: *mut ffi::AVFrame, src: *const ffi::AVFrame) {
        unsafe {
            // (*dst).key_frame              = (*src).key_frame;
            (*dst).pict_type              = (*src).pict_type;
            (*dst).sample_aspect_ratio    = (*src).sample_aspect_ratio;
            // (*dst).crop_top               = (*src).crop_top;
            // (*dst).crop_bottom            = (*src).crop_bottom;
            // (*dst).crop_left              = (*src).crop_left;
            // (*dst).crop_right             = (*src).crop_right;
            (*dst).pts                    = (*src).pts;
            (*dst).repeat_pict            = (*src).repeat_pict;
            //(*dst).interlaced_frame       = (*src).interlaced_frame;
            //(*dst).top_field_first        = (*src).top_field_first;
            //(*dst).palette_has_changed    = (*src).palette_has_changed;
            (*dst).sample_rate            = (*src).sample_rate;
            (*dst).opaque                 = (*src).opaque;
            // (*dst).pkt_dts                = (*src).pkt_dts;
            // (*dst).pkt_pos                = (*src).pkt_pos;
            // (*dst).pkt_size               = (*src).pkt_size;
            // (*dst).pkt_duration           = (*src).pkt_duration;
            // (*dst).time_base              = (*src).time_base; // TODO
            // (*dst).reordered_opaque       = (*src).reordered_opaque;
            (*dst).quality                = (*src).quality;
            // (*dst).best_effort_timestamp  = (*src).best_effort_timestamp;
            // (*dst).coded_picture_number   = (*src).coded_picture_number;
            // (*dst).display_picture_number = (*src).display_picture_number;
            (*dst).flags                  = (*src).flags;
            (*dst).decode_error_flags     = (*src).decode_error_flags;
            (*dst).color_primaries        = (*src).color_primaries;
            (*dst).color_trc              = (*src).color_trc;
            (*dst).colorspace             = (*src).colorspace;
            (*dst).color_range            = (*src).color_range;
            (*dst).chroma_location        = (*src).chroma_location;
        }
    }
}
