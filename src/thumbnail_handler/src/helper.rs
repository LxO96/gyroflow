// SPDX-License-Identifier: GPL-3.0-or-later

//! Runs `gyroflow-thumbnailer.exe` and reads a decoded frame back from it.
//!
//! The subprocess exists so it can be *killed*. A shell extension that hangs on a corrupt
//! file is one of the standard ways to make Explorer unusable, and a decoder linked into the
//! shell host offers no way to bound that. Here the budget is enforced by simply terminating
//! the process.

use std::io::Read;
use std::path::PathBuf;
use std::os::windows::io::AsRawHandle;
use std::process::{ Command, Stdio };
use std::time::Duration;

use windows::Win32::Foundation::HANDLE;
use windows::Win32::System::Threading::TerminateProcess;

use crate::protocol;

/// Wall-clock budget for one decode. Measured cost on a 193 MB 10-bit 4K clip is ~270 ms, so
/// this is generous by an order of magnitude while still being far below the point at which
/// Explorer gives up on a folder.
const TIMEOUT: Duration = Duration::from_secs(3);

/// Ceiling on the pixel payload, independent of `protocol::MAX_EDGE`. Two bounds rather than
/// one because this is the number that decides an allocation inside a shell host.
const MAX_PAYLOAD: u64 = (protocol::MAX_EDGE as u64) * (protocol::MAX_EDGE as u64) * 4;

/// A decoded frame: tightly packed BGRA, `w * h * 4` bytes.
pub struct Frame {
    pub w: u32,
    pub h: u32,
    pub bgra: Vec<u8>,
}

/// Absolute path to the helper, resolved from this DLL's own location.
///
/// Never from `PATH`: `dllhost.exe` runs with the user's token but not their environment, so
/// a bare name would resolve unpredictably or not at all. The helper is installed beside the
/// DLL, which is also where the ffmpeg DLLs it imports live.
fn helper_path() -> Option<PathBuf> {
    let dll = crate::register::this_dll_path().ok()?;
    let dir = PathBuf::from(dll).parent()?.to_path_buf();
    let exe = dir.join("gyroflow-thumbnailer.exe");
    if exe.is_file() { Some(exe) } else { None }
}

/// Decode one frame of `input`, scaled to fit `max_edge`.
///
/// `None` for every failure - helper missing, spawn refused, timeout, non-zero exit, bad
/// header, short read. The caller reports that to Explorer as "no thumbnail", which is
/// exactly what it did before this handler existed.
pub fn decode_frame(input: &str, max_edge: u32) -> Option<Frame> {
    let exe = helper_path()?;

    let mut child = Command::new(exe)
        .arg(input)
        .arg(max_edge.min(protocol::MAX_EDGE).to_string())
        .stdin(Stdio::null())   // it must never block waiting for input
        .stdout(Stdio::piped())
        .stderr(Stdio::null())  // a stray ffmpeg warning must not reach the pixel stream
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .ok()?;

    let mut stdout = child.stdout.take()?;

    // A WATCHDOG THREAD, not a check inside the read loop.
    //
    // The obvious implementation - measure elapsed time between reads - does not work, and
    // was measured not working: a child that writes nothing and never exits leaves `read`
    // blocked in the kernel forever, so the loop never comes back round to look at the clock.
    // A hung helper hung the shell request with it, which is precisely the failure this whole
    // subprocess design exists to prevent.
    //
    // Terminating the child is what unblocks the read: the pipe's write end closes and `read`
    // returns 0. The thread waits on a channel so a normal, fast decode cancels it instead of
    // sleeping out the full budget.
    let raw_handle = child.as_raw_handle() as usize;
    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    let watchdog = std::thread::spawn(move || {
        if done_rx.recv_timeout(TIMEOUT).is_err() {
            // Timed out (or the sender vanished). Kill it; `read` then returns 0 and the
            // caller discards whatever partial bytes arrived.
            unsafe { let _ = TerminateProcess(HANDLE(raw_handle as *mut _), 1); }
            return true;
        }
        false
    });

    let mut buf = Vec::with_capacity(protocol::HEADER_LEN + 512 * 512 * 4);
    let mut chunk = [0u8; 64 * 1024];
    let mut read_failed = false;
    loop {
        match stdout.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if buf.len() as u64 > protocol::HEADER_LEN as u64 + MAX_PAYLOAD {
                    read_failed = true;
                    break;
                }
            }
            Err(_) => { read_failed = true; break; }
        }
    }

    // Stop the watchdog before waiting, so a slow-but-successful decode is not killed while
    // we are reaping it.
    let _ = done_tx.send(());
    let timed_out = watchdog.join().unwrap_or(false);

    if read_failed || timed_out {
        let _ = child.kill();
        let _ = child.wait();
        return None;
    }

    // A non-zero exit means it reported failure; trust that over whatever is in the buffer.
    match child.wait() {
        Ok(status) if status.success() => {}
        _ => return None,
    }

    let (w, h) = protocol::decode_header(&buf)?;
    let expected = protocol::payload_len(w, h);
    let payload = &buf[protocol::HEADER_LEN..];
    // Exact, not "at least": a short read means a truncated picture, and a long one means the
    // header and the data disagree. Neither is safe to display.
    if payload.len() as u64 != expected {
        return None;
    }

    Some(Frame { w, h, bgra: payload.to_vec() })
}

/// Keeps the helper from flashing a console window. It is already built as a windows-subsystem
/// binary, so this is belt and braces - but a thumbnail that pops up a black rectangle on a
/// user's desktop is bad enough to be worth both.
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

use std::os::windows::process::CommandExt;

/// Extensions handled by decoding, as opposed to `.gyroflow`'s embedded image.
///
/// `.braw`, `.r3d` and `.nev` are deliberately absent: they decode only through mdk with
/// vendor SDKs, so claiming them would mean shipping a handler that silently fails on exactly
/// the formats those users care about.
pub const VIDEO_EXTENSIONS: &[&str] = &[".mp4", ".mov", ".mxf", ".360", ".insv", ".exr"];

/// Whether a path is one we decode rather than parse.
pub fn is_video_extension(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    VIDEO_EXTENSIONS.iter().any(|e| lower.ends_with(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_extensions_are_matched_case_insensitively() {
        assert!(is_video_extension("C:/x/CLIP.MP4"));
        assert!(is_video_extension("C:/x/clip.mp4"));
        assert!(is_video_extension("C:/x/a.MoV"));
        assert!(is_video_extension("C:/x/render.exr"));
    }

    /// The project path must NOT go through the decoder - it has its own embedded image and a
    /// much faster path that involves no subprocess at all.
    #[test]
    fn project_files_are_not_treated_as_video() {
        assert!(!is_video_extension("C:/x/project.gyroflow"));
    }

    /// Vendor-SDK formats stay out. If these ever appear here, the helper will fail on them
    /// and users will get a blank icon where Windows might otherwise have managed something.
    #[test]
    fn vendor_sdk_formats_are_excluded() {
        for f in ["a.braw", "b.r3d", "c.nev"] {
            assert!(!is_video_extension(f), "{f} should not be claimed");
        }
    }

    /// A name that merely contains an extension string is not a match.
    #[test]
    fn a_substring_is_not_an_extension() {
        assert!(!is_video_extension("C:/x/mp4notes.txt"));
        assert!(!is_video_extension("C:/my.mov.backup"));
    }
}
