// SPDX-License-Identifier: GPL-3.0-or-later

//! Display rotation for decoded frames.
//!
//! Decoded frames are in sensor orientation; a phone or action-cam clip carries rotation
//! metadata that a player applies at display time. Ignoring it puts a sideways picture in
//! Explorer, which is worse than no picture.
//!
//! Note the sign convention, because getting it wrong yields a correctly-*shaped* but
//! upside-down image — easy to miss at icon size, and exactly the bug that occurred while
//! building the `.gyroflow` handler. The stream's `rotate` metadata (and the display matrix)
//! express a counter-clockwise angle; `rotate_bgra` works clockwise. `rotation_from_stream`
//! returns the clockwise angle to apply, having already negated.

/// The clockwise rotation, in degrees, that this stream's frames need to display upright.
/// Always one of 0, 90, 180, 270.
pub fn rotation_from_stream(stream: &ffmpeg_next::format::stream::Stream) -> i64 {
    // The DISPLAY MATRIX is where modern containers keep this - not the legacy `rotate` tag.
    // Reading only the tag returns 0 for files that plainly are rotated: the clip this was
    // built against reports `rotation=-90` in side data and carries no `rotate` tag at all,
    // and produced a sideways thumbnail until this was fixed. The tag is still checked as a
    // fallback for older files that only have it.
    let ccw = display_matrix_rotation(stream).or_else(|| {
        stream.metadata().get("rotate")
            .and_then(|v| v.trim().parse::<f64>().ok())
            .map(|v| v.round() as i64)
    }).unwrap_or(0);

    normalise(-ccw)
}

/// The counter-clockwise angle encoded in a stream's 3x3 display matrix, if it has one.
///
/// Equivalent to ffmpeg's `av_display_rotation_get`, computed here so the helper needs no
/// extra linkage: the matrix is nine 16.16 fixed-point ints, and the angle is
/// `-atan2(m[1], m[0])`.
fn display_matrix_rotation(stream: &ffmpeg_next::format::stream::Stream) -> Option<i64> {
    for sd in stream.side_data() {
        if sd.kind() != ffmpeg_next::codec::packet::side_data::Type::DisplayMatrix { continue; }
        let bytes = sd.data();
        if bytes.len() < 36 { continue; }
        let at = |i: usize| -> f64 {
            i32::from_ne_bytes([bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]]) as f64 / 65536.0
        };
        let deg = -at(1).atan2(at(0)).to_degrees();
        return Some(deg.round() as i64);
    }
    None
}

/// Snap to the nearest right angle in 0..360. Video rotation metadata is only ever a right
/// angle in practice, and an arbitrary angle would need resampling for no benefit here.
fn normalise(deg: i64) -> i64 {
    let d = ((deg % 360) + 360) % 360;
    match d {
        45..=134 => 90,
        135..=224 => 180,
        225..=314 => 270,
        _ => 0,
    }
}

/// Rotate a BGRA image clockwise by 90, 180 or 270 degrees, returning the new dimensions and
/// a tightly packed buffer. `stride` is the source's row pitch, which is usually padded.
///
/// Any angle other than those three is treated as no rotation, but the rows are still
/// repacked so callers get a predictable `w * 4` pitch either way.
pub fn rotate_bgra(src: &[u8], w: u32, h: u32, stride: u32, degrees: i64) -> (u32, u32, Vec<u8>) {
    let (dw, dh) = if degrees == 90 || degrees == 270 { (h, w) } else { (w, h) };
    let mut out = vec![0u8; (dw as usize) * (dh as usize) * 4];

    for y in 0..h {
        for x in 0..w {
            let si = (y * stride + x * 4) as usize;
            if si + 4 > src.len() { continue; }
            let (dx, dy) = match degrees {
                90 => (h - 1 - y, x),
                180 => (w - 1 - x, h - 1 - y),
                270 => (y, w - 1 - x),
                _ => (x, y),
            };
            let di = ((dy * dw + dx) * 4) as usize;
            if di + 4 <= out.len() {
                out[di..di + 4].copy_from_slice(&src[si..si + 4]);
            }
        }
    }
    (dw, dh, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2x1: blue pixel at (0,0), green at (1,0). BGRA, packed.
    fn src() -> Vec<u8> { vec![255, 0, 0, 255, 0, 255, 0, 255] }
    const BLUE: [u8; 4] = [255, 0, 0, 255];
    const GREEN: [u8; 4] = [0, 255, 0, 255];
    fn px(buf: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * w + x) * 4) as usize;
        [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]
    }

    #[test]
    fn rotating_90_clockwise_puts_the_left_pixel_on_top() {
        let (w, h, out) = rotate_bgra(&src(), 2, 1, 8, 90);
        assert_eq!((w, h), (1, 2));
        assert_eq!(px(&out, w, 0, 0), BLUE);
        assert_eq!(px(&out, w, 0, 1), GREEN);
    }

    /// The opposite of 90. A sign error makes these two identical, which is the failure this
    /// pins down.
    #[test]
    fn rotating_270_clockwise_puts_the_left_pixel_on_the_bottom() {
        let (w, h, out) = rotate_bgra(&src(), 2, 1, 8, 270);
        assert_eq!((w, h), (1, 2));
        assert_eq!(px(&out, w, 0, 0), GREEN);
        assert_eq!(px(&out, w, 0, 1), BLUE);
    }

    #[test]
    fn rotating_180_keeps_the_shape_and_swaps_the_ends() {
        let (w, h, out) = rotate_bgra(&src(), 2, 1, 8, 180);
        assert_eq!((w, h), (2, 1));
        assert_eq!(px(&out, w, 0, 0), GREEN);
        assert_eq!(px(&out, w, 1, 0), BLUE);
    }

    /// Decoded frames are nearly always padded; reading the padding as pixels smears the
    /// picture sideways by a pixel per row.
    #[test]
    fn padded_rows_are_not_read_as_pixels() {
        let src = vec![255, 0, 0, 255, 9, 9, 9, 9, 0, 255, 0, 255, 9, 9, 9, 9];
        let (w, h, out) = rotate_bgra(&src, 1, 2, 8, 180);
        assert_eq!((w, h), (1, 2));
        assert_eq!(px(&out, w, 0, 0), GREEN);
        assert_eq!(px(&out, w, 0, 1), BLUE);
        assert!(!out.windows(4).any(|c| c == [9, 9, 9, 9]), "padding leaked into the image");
    }

    /// The metadata is counter-clockwise and we rotate clockwise, so the two must be
    /// opposites. A clip tagged 270 needs a quarter turn clockwise to come out upright.
    #[test]
    fn the_metadata_angle_is_negated() {
        assert_eq!(normalise(-270), 90);
        assert_eq!(normalise(-90), 270);
        assert_eq!(normalise(-180), 180);
        assert_eq!(normalise(0), 0);
    }

    #[test]
    fn odd_angles_snap_to_a_right_angle() {
        assert_eq!(normalise(89), 90);
        assert_eq!(normalise(-361), 0);
        assert_eq!(normalise(719), 0);
    }
}
