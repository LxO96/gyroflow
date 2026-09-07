// SPDX-License-Identifier: GPL-3.0-or-later

//! The wire format between `gyroflow-thumbnailer.exe` and the Explorer thumbnail handler.
//!
//! Deliberately tiny and fixed-size: the handler runs inside a shell host, so parsing must be
//! trivial and impossible to get subtly wrong. Raw BGRA rather than an encoded image, because
//! a JPEG round-trip between two processes we control would cost time and quality for nothing
//! — the handler feeds these bytes straight into a DIB.
//!
//! This module is compiled into BOTH the helper and the handler, so the two can never drift
//! apart on layout. It has no dependencies on purpose.

/// Identifies our output and catches a stream that is actually an error message or a stray
/// ffmpeg warning. "GFTH" = GyroFlow THumbnail.
pub const MAGIC: [u8; 4] = *b"GFTH";

/// Bytes preceding the pixel data: magic, width, height, all little-endian.
pub const HEADER_LEN: usize = 12;

/// Upper bound on a decoded thumbnail's edge, enforced on BOTH sides.
///
/// The handler must never size an allocation from a number a subprocess told it without a
/// bound — a corrupt or hostile header claiming 40000x40000 would otherwise ask for 6 GB
/// inside a shell host. 4096 is far above anything Explorer requests (96–256 typically,
/// rarely over 1024).
pub const MAX_EDGE: u32 = 4096;

/// Serialise a header. Pixel data follows immediately, `w * h * 4` bytes of BGRA.
pub fn encode_header(w: u32, h: u32) -> [u8; HEADER_LEN] {
    let mut out = [0u8; HEADER_LEN];
    out[0..4].copy_from_slice(&MAGIC);
    out[4..8].copy_from_slice(&w.to_le_bytes());
    out[8..12].copy_from_slice(&h.to_le_bytes());
    out
}

/// Parse a header, returning `(width, height)`.
///
/// `None` for anything not recognisably ours or not within `MAX_EDGE`. The caller's only
/// correct response is to give up on the thumbnail — never to guess at a size.
pub fn decode_header(buf: &[u8]) -> Option<(u32, u32)> {
    if buf.len() < HEADER_LEN || buf[0..4] != MAGIC {
        return None;
    }
    let w = u32::from_le_bytes(buf[4..8].try_into().ok()?);
    let h = u32::from_le_bytes(buf[8..12].try_into().ok()?);
    if w == 0 || h == 0 || w > MAX_EDGE || h > MAX_EDGE {
        return None;
    }
    Some((w, h))
}

/// Byte length of the pixel payload for a given size. `u64` so the multiplication cannot
/// overflow on 32-bit, even though `decode_header` has already bounded the inputs.
pub fn payload_len(w: u32, h: u32) -> u64 {
    w as u64 * h as u64 * 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_header_round_trips() {
        assert_eq!(decode_header(&encode_header(288, 512)), Some((288, 512)));
    }

    /// The handler reads whatever the helper wrote. If ffmpeg printed a warning to the wrong
    /// stream, or the helper died mid-write, the result must be rejected rather than
    /// interpreted as dimensions.
    #[test]
    fn junk_is_not_mistaken_for_a_header() {
        assert_eq!(decode_header(b""), None);
        assert_eq!(decode_header(b"short"), None);
        assert_eq!(decode_header(b"[warning] deprecated pixel format"), None);
        // Right length, wrong magic.
        assert_eq!(decode_header(&[0u8; HEADER_LEN]), None);
    }

    /// The bound is the whole reason the handler can size an allocation from this.
    #[test]
    fn absurd_and_zero_dimensions_are_rejected() {
        assert_eq!(decode_header(&encode_header(0, 512)), None);
        assert_eq!(decode_header(&encode_header(512, 0)), None);
        assert_eq!(decode_header(&encode_header(MAX_EDGE + 1, 512)), None);
        assert_eq!(decode_header(&encode_header(512, u32::MAX)), None);
        // The bound itself is allowed.
        assert_eq!(decode_header(&encode_header(MAX_EDGE, MAX_EDGE)), Some((MAX_EDGE, MAX_EDGE)));
    }

    #[test]
    fn payload_length_does_not_overflow_at_the_bound() {
        assert_eq!(payload_len(MAX_EDGE, MAX_EDGE), 4096u64 * 4096 * 4);
    }
}
