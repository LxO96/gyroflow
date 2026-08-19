// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2021-2022 Adrian <adrian.eddy at gmail>

use ffmpeg_next::{ ffi, format, frame, software, util };
use crate::rendering::FFmpegError;

#[derive(Default)]
pub struct Converter {
    pub convert_to: Option<software::scaling::Context>,
    pub convert_from: Option<software::scaling::Context>,
    pub sw_frame_converted: Option<frame::Video>,
    pub sw_frame_converted_out: Option<frame::Video>,
}
impl<'a> Converter {
    /// The `SWS_CS_*` constant `sws_getCoefficients` expects, for the colour space of the
    /// **YUV** side of a conversion. `sws_getCoefficients` falls back to `SWS_CS_DEFAULT`
    /// (= ITU601) for anything it does not recognise, which would be a silent downgrade for
    /// an unspecified BT.709 source, so every unknown space is mapped to 709 here instead -
    /// the same assumption `ffmpeg_video.rs` hardcodes.
    fn sws_colorspace(space: util::color::Space) -> std::os::raw::c_int {
        use util::color::Space;
        match space {
            Space::BT709                    => ffi::SWS_CS_ITU709,
            Space::FCC                      => ffi::SWS_CS_FCC,
            Space::BT470BG | Space::SMPTE170M => ffi::SWS_CS_ITU601,
            Space::SMPTE240M                => ffi::SWS_CS_SMPTE240M,
            // swscale has no constant-luminance support at all; BT2020 NCL coefficients are
            // the closest thing it can express for either.
            Space::BT2020NCL | Space::BT2020CL => ffi::SWS_CS_BT2020,
            _                               => ffi::SWS_CS_ITU709,
        }
    }

    /// Tells swscale the actual ranges of the two frames a context converts between, instead
    /// of letting it fall back to its defaults. Must be called after `Context::get` and is
    /// not retroactive, so it belongs next to the construction of the context it configures.
    ///
    /// `yuv_space` is the colour space of whichever side is YUV; it is ignored for the RGB
    /// side, which has no matrix of its own.
    unsafe fn set_color_details(ctx: &mut software::scaling::Context, yuv_space: util::color::Space, src_range: util::color::Range, dst_range: util::color::Range) {
        use std::os::raw::c_int;
        let src_range: c_int = if src_range == util::color::Range::JPEG { 1 } else { 0 };
        let dst_range: c_int = if dst_range == util::color::Range::JPEG { 1 } else { 0 };
        unsafe {
            let coefs = ffi::sws_getCoefficients(Self::sws_colorspace(yuv_space));
            ffi::sws_setColorspaceDetails(ctx.as_mut_ptr(), coefs, src_range, coefs, dst_range, 0, 1 << 16, 1 << 16);
        }
    }

    pub fn convert_pixel_format<F>(&mut self, frame: &mut frame::Video, out_frame: &mut frame::Video, format: format::Pixel, interpolation: software::scaling::flag::Flags, cb: F) -> Result<(), FFmpegError> where F: FnMut(&mut frame::Video, &mut frame::Video) + 'a {
        self.convert_pixel_format_impl(frame, out_frame, format, interpolation, false, cb)
    }

    /// `convert_pixel_format`, but with the colour range of both conversions pinned to what
    /// the frames actually declare (see `set_color_details`).
    ///
    /// This exists as a separate entry point rather than as a change to `convert_pixel_format`
    /// so that the "unknown format -> `YUV444P16LE`" fallback, which is YUV->YUV and
    /// round-trips through swscale's defaults today, keeps its exact current behaviour.
    /// Getting the range wrong only becomes visible when one side is RGB, which is the case
    /// for the colour-baking conversion and only that one.
    pub fn convert_pixel_format_color_aware<F>(&mut self, frame: &mut frame::Video, out_frame: &mut frame::Video, format: format::Pixel, interpolation: software::scaling::flag::Flags, cb: F) -> Result<(), FFmpegError> where F: FnMut(&mut frame::Video, &mut frame::Video) + 'a {
        self.convert_pixel_format_impl(frame, out_frame, format, interpolation, true, cb)
    }

    fn convert_pixel_format_impl<F>(&mut self, frame: &mut frame::Video, out_frame: &mut frame::Video, format: format::Pixel, interpolation: software::scaling::flag::Flags, color_range_aware: bool, mut cb: F) -> Result<(), FFmpegError> where F: FnMut(&mut frame::Video, &mut frame::Video) + 'a {
        if frame.format() != format {
            if self.sw_frame_converted.is_none() {
                let mut converted = frame::Video::new(format, frame.width(), frame.height());
                //self.convert_from = Some(software::converter((frame.width(), frame.height()), frame.format(), format)?);
                let mut conv = software::scaling::Context::get(
                    frame.format(), // input
                    frame.width(),
                    frame.height(),
                    format, // output
                    frame.width(),
                    frame.height(),
                    interpolation,
                )?;
                if color_range_aware {
                    // The intermediate is packed RGB, and RGB in ffmpeg is full range by
                    // convention - there is no such thing as a 16-235 RGB frame here.
                    converted.set_color_range(util::color::Range::JPEG);
                    unsafe { Self::set_color_details(&mut conv, frame.color_space(), frame.color_range(), converted.color_range()); }
                }
                self.sw_frame_converted = Some(converted);
                self.convert_from = Some(conv);
            }

            if self.sw_frame_converted_out.is_none() {
                let mut converted_out = frame::Video::new(format, out_frame.width(), out_frame.height());
                //self.convert_to = Some(software::converter((out_frame.width(), out_frame.height()), format, out_frame.format())?);
                let mut conv = software::scaling::Context::get(
                    format, // input
                    out_frame.width(),
                    out_frame.height(),
                    out_frame.format(), // output
                    out_frame.width(),
                    out_frame.height(),
                    interpolation,
                )?;
                if color_range_aware {
                    converted_out.set_color_range(util::color::Range::JPEG);
                    // Here the YUV side is the output frame, so the matrix comes from it.
                    unsafe { Self::set_color_details(&mut conv, out_frame.color_space(), converted_out.color_range(), out_frame.color_range()); }
                }
                self.sw_frame_converted_out = Some(converted_out);
                self.convert_to = Some(conv);
            }

            let sw_frame_converted = self.sw_frame_converted.as_mut().ok_or(FFmpegError::FrameEmpty)?;
            let sw_frame_converted_out = self.sw_frame_converted_out.as_mut().ok_or(FFmpegError::FrameEmpty)?;
            let convert_from = self.convert_from.as_mut().ok_or(FFmpegError::ConverterEmpty)?;
            let convert_to = self.convert_to.as_mut().ok_or(FFmpegError::ConverterEmpty)?;

            convert_from.run(frame, sw_frame_converted)?;

            cb(sw_frame_converted, sw_frame_converted_out);

            convert_to.run(sw_frame_converted_out, out_frame)?;
        } else {
            cb(frame, out_frame);
        }
        Ok(())
    }

    // Scale is only used for autosync
    pub fn scale(&mut self, frame: &mut frame::Video, format: format::Pixel, width: u32, height: u32) -> Result<frame::Video, FFmpegError> {
        if frame.width() != width || frame.height() != height || frame.format() != format {
            if self.sw_frame_converted.is_none() {
                self.sw_frame_converted = Some(frame::Video::new(format, width, height));
                self.convert_to = Some(
                    software::scaling::Context::get(
                        frame.format(), frame.width(), frame.height(), format, width, height, software::scaling::Flags::BILINEAR,
                    )?
                );
            }

            let sw_frame_converted = self.sw_frame_converted.as_mut().ok_or(FFmpegError::FrameEmpty)?;
            let convert_to = self.convert_to.as_mut().ok_or(FFmpegError::ConverterEmpty)?;

            convert_to.run(frame, sw_frame_converted)?;

            Ok(unsafe { frame::Video::wrap(ffi::av_frame_clone(sw_frame_converted.as_ptr())) })
        } else {
            Ok(unsafe { frame::Video::wrap(ffi::av_frame_clone(frame.as_ptr())) })
        }
    }
}
