// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2022 Adrian <adrian.eddy at gmail>

use super::*;
use super::mdk_processor::*;
use super::ffmpeg_video::RateControl;
use ffmpeg_next::{ frame, Dictionary };
use std::{ sync::{ Arc, atomic::AtomicBool }, rc::Rc, cell::RefCell };

pub enum Processor<'a> {
    Ffmpeg(FfmpegProcessor<'a>),
    Mdk(MDKProcessor)
}
pub struct VideoProcessor<'a> {
    inner: Processor<'a>
}

impl<'a> VideoProcessor<'a> {
    pub fn from_file( url: &str, gpu_decoding: bool, gpu_decoder_index: usize, decoder_options: Option<Dictionary>) -> Result<Self, FFmpegError> {
        let filename = gyroflow_core::filesystem::get_filename(url);
        if filename.to_lowercase().ends_with(".braw") || filename.to_lowercase().ends_with(".r3d") || filename.to_lowercase().ends_with(".nev") {
            Ok(Self { inner: Processor::Mdk(MDKProcessor::from_file(url, decoder_options, gpu_decoding)) })
        } else {
            Ok(Self { inner: Processor::Ffmpeg(FfmpegProcessor::from_file(url, gpu_decoding, gpu_decoder_index, decoder_options)?) })
        }
    }

    pub fn get_video_info(url: &str) -> Result<crate::rendering::ffmpeg_processor::VideoInfo, ffmpeg_next::Error> {
        let filename = gyroflow_core::filesystem::get_filename(url);
        if filename.to_lowercase().ends_with(".braw") || filename.to_lowercase().ends_with(".r3d") || filename.to_lowercase().ends_with(".nev") {
            let mut mdk = MDKProcessor::from_file(url, None, false);

            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();

            let info = Rc::new(RefCell::new(crate::rendering::ffmpeg_processor::VideoInfo::default()));

            let info2 = info.clone();
            mdk.mdk.startProcessing(0, 0, 0, false, &mdk.custom_decoder, vec![], move |frame_num, _, _, _, org_width, org_height, fps, duration_ms, frame_count, data| {
                if fps > 0.0 && org_width > 0 {
                    let mut info2 = info2.borrow_mut();
                    info2.duration_ms = duration_ms;
                    info2.frame_count = frame_count as usize;
                    info2.fps = fps;
                    info2.width = org_width;
                    info2.height = org_height;
                }
                if frame_num == -1 || data.is_empty() {
                    let _ = tx.send(());
                    return true;
                }
                false
            });
            pollster::block_on(rx.receive());
            info.borrow_mut().rotation = mdk.mdk.getRotation();

            std::thread::sleep(std::time::Duration::from_millis(100));
            mdk.mdk.stopProcessing(0);

            let info = info.borrow().clone();
            Ok(info)
        } else {
            FfmpegProcessor::get_video_info(url)
        }
    }

    pub fn on_frame<F>(&mut self, cb: F) where F: FnMut(i64, &mut frame::Video, Option<&mut frame::Video>, &mut ffmpeg_video_converter::Converter, &mut RateControl) -> Result<(), FFmpegError> + 'static {
        match &mut self.inner {
            Processor::Ffmpeg(x) => x.on_frame(cb),
            Processor::Mdk(x) => x.on_frame(cb),
        }
    }
    pub fn start_decoder_only(&mut self, ranges: Vec<(f64, f64)>, cancel_flag: Arc<AtomicBool>) -> Result<(), FFmpegError> {
        match &mut self.inner {
            Processor::Ffmpeg(x) => x.start_decoder_only(ranges, cancel_flag),
            Processor::Mdk(x) => x.start_decoder_only(ranges, cancel_flag)
        }
    }

    /// Decode at most `max_frames` frames from `start_ms` to `end_ms` and stop.
    ///
    /// A thin wrapper over `start_decoder_only`, which seeks and then runs to the end of its
    /// range: the denoised preview only ever wants the handful of frames immediately around
    /// the playhead, so the decode loop is stopped through the same cancel flag as soon as
    /// the callback has seen `max_frames` of them. Without the frame budget a paused preview
    /// near the start of a long clip would decode the whole file.
    ///
    /// `start_ms` is a *filter* as well as a seek target - `receive_and_process_video_frames`
    /// drops everything before it - so the frames handed to `cb` are exactly the first
    /// `max_frames` presented at or after `start_ms`. The trailing frame past `end_ms` that
    /// the range check lets through before finishing is the caller's to discard.
    ///
    /// Lives here rather than on `FfmpegProcessor` so the .braw/.r3d MDK decoder gets it too;
    /// both back-ends already share `on_frame` and `start_decoder_only`.
    pub fn decode_only_range<F>(&mut self, start_ms: f64, end_ms: f64, max_frames: usize, mut cb: F) -> Result<(), FFmpegError>
        where F: FnMut(i64, &mut frame::Video, &mut ffmpeg_video_converter::Converter) + 'static
    {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag2 = stop_flag.clone();
        let mut decoded = 0usize;
        self.on_frame(move |timestamp_us, input_frame, _output_frame, converter, _rate_control| {
            if decoded < max_frames {
                cb(timestamp_us, input_frame, converter);
                decoded += 1;
            }
            if decoded >= max_frames {
                stop_flag2.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            Ok(())
        });
        self.start_decoder_only(vec![(start_ms, end_ms)], stop_flag)
    }
}
