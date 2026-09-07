// SPDX-License-Identifier: GPL-3.0-or-later

//! Decodes the first frame of a video file and writes it to stdout as raw BGRA.
//!
//! Spawned by `gyroflow_thumbnail.dll` (the Explorer thumbnail handler) for formats Windows
//! cannot preview itself — 10-bit 4:2:2 H.264 above all, which is what prompted this.
//!
//! It exists as a separate process rather than as code inside the DLL for one reason above
//! the others: **it can be killed.** A shell extension that hangs on a corrupt file is a
//! classic way to make Explorer unusable. The handler gives this process a hard wall-clock
//! budget and terminates it on overrun; that guarantee is not available to a decoder linked
//! into the shell host.
//!
//! Usage: `gyroflow-thumbnailer <input-file> <max-edge>` — writes a `protocol` header plus
//! `w * h * 4` bytes of BGRA to stdout. Any failure exits non-zero having written nothing,
//! which the handler reports to Explorer as "no thumbnail" so it draws the generic icon.

// Without this a console window flashes on every single thumbnail Explorer draws.
#![windows_subsystem = "windows"]

mod protocol;
mod rotate;

use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: gyroflow-thumbnailer <input-file> <max-edge>");
        std::process::exit(2);
    }
    let max_edge = match args[2].parse::<u32>() {
        Ok(v) if v > 0 && v <= protocol::MAX_EDGE => v,
        // Clamp rather than fail: Explorer occasionally asks for a very large box, and a
        // slightly smaller thumbnail is a far better answer than none.
        Ok(_) => protocol::MAX_EDGE,
        Err(_) => { eprintln!("max-edge must be a positive integer"); std::process::exit(2); }
    };

    match render(&args[1], max_edge) {
        Ok((w, h, bgra)) => {
            let mut out = std::io::stdout().lock();
            if out.write_all(&protocol::encode_header(w, h)).is_err() || out.write_all(&bgra).is_err() || out.flush().is_err() {
                // A broken pipe means the handler gave up on us, most likely on its timeout.
                // Nothing to report to and nothing to clean up.
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("thumbnail extraction failed: {e}");
            std::process::exit(1);
        }
    }
}

/// Decode the first usable frame, scale it to fit `max_edge`, and apply rotation metadata.
fn render(path: &str, max_edge: u32) -> Result<(u32, u32, Vec<u8>), String> {
    ffmpeg_next::init().map_err(|e| format!("ffmpeg init: {e}"))?;

    let mut ictx = ffmpeg_next::format::input(&path).map_err(|e| format!("open: {e}"))?;
    let stream = ictx.streams().best(ffmpeg_next::media::Type::Video).ok_or("no video stream")?;
    let stream_index = stream.index();

    // Display rotation lives in stream side data. Without applying it, a phone or action-cam
    // clip produces a sideways thumbnail, which is worse than no thumbnail.
    let rotation = rotate::rotation_from_stream(&stream);

    let ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters()).map_err(|e| format!("codec: {e}"))?;
    let mut decoder = ctx.decoder().video().map_err(|e| format!("decoder: {e}"))?;

    let (sw, sh) = (decoder.width(), decoder.height());
    if sw == 0 || sh == 0 { return Err("zero-sized video".into()); }

    // Fit the long edge, never upscale: enlarging a small frame to fill a big request only
    // produces a blurrier picture.
    let scale = (max_edge as f64 / sw.max(sh) as f64).min(1.0);
    let tw = ((sw as f64 * scale).round() as u32).max(1);
    let th = ((sh as f64 * scale).round() as u32).max(1);

    // BGRA is what a Windows DIB wants, so the handler can memcpy our output straight in.
    let mut scaler = ffmpeg_next::software::scaling::Context::get(
        decoder.format(), sw, sh,
        ffmpeg_next::format::Pixel::BGRA, tw, th,
        ffmpeg_next::software::scaling::Flags::BILINEAR,
    ).map_err(|e| format!("scaler: {e}"))?;

    let mut decoded = ffmpeg_next::frame::Video::empty();
    let mut rgba = ffmpeg_next::frame::Video::empty();

    for (s, packet) in ictx.packets() {
        if s.index() != stream_index { continue; }
        if decoder.send_packet(&packet).is_err() { continue; }
        while decoder.receive_frame(&mut decoded).is_ok() {
            scaler.run(&decoded, &mut rgba).map_err(|e| format!("scale: {e}"))?;
            return Ok(finish(&rgba, tw, th, rotation));
        }
    }

    // Flush: some codecs deliver nothing until end of input, and a very short clip can hit
    // this path even though it decodes perfectly well.
    let _ = decoder.send_eof();
    while decoder.receive_frame(&mut decoded).is_ok() {
        scaler.run(&decoded, &mut rgba).map_err(|e| format!("scale: {e}"))?;
        return Ok(finish(&rgba, tw, th, rotation));
    }

    Err("no decodable frame".into())
}

/// Pack a scaled frame into a tight BGRA buffer, rotating if the stream asked for it.
fn finish(frame: &ffmpeg_next::frame::Video, w: u32, h: u32, rotation: i64) -> (u32, u32, Vec<u8>) {
    let stride = frame.stride(0) as u32;
    let data = frame.data(0);

    if rotation != 0 {
        return rotate::rotate_bgra(data, w, h, stride, rotation);
    }

    // Even with no rotation the rows must be repacked: a decoded frame's stride is usually
    // padded, and the handler expects exactly `w * 4` bytes per row.
    let row = (w * 4) as usize;
    let mut out = vec![0u8; row * h as usize];
    for y in 0..h as usize {
        let src = y * stride as usize;
        if src + row <= data.len() {
            out[y * row..(y + 1) * row].copy_from_slice(&data[src..src + row]);
        }
    }
    (w, h, out)
}
