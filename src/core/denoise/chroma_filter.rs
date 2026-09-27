// Copyright © 2026 Gyroflow

//! Chroma smoothing: a luma-guided joint bilateral on chroma, run after the temporal gather.
//! This file is the definition of the maths. `opencl_chroma_filter.cl` and
//! `wgpu_chroma_filter.wgsl` mirror it and are tested against it (`tests_gpu.rs`).
//!
//! Why: temporal denoising cannot average subjects that move relative to the camera, and the
//! chroma noise it leaves there is blotchy (8-16 px, from 4:2:2 capture plus H.264). Measured
//! on Sony S-Log3 4K, this filter took the V noise on such a patch from 3.23 to 1.84 while
//! keeping colour edges on luma edges (see docs/superpowers/specs/2026-09-27-chroma-smoothing-design.md).

pub const RADIUS: i32 = 8;
pub const STEP: i32 = 2;
pub const SIGMA_SPATIAL: f32 = 4.0;
/// Luma difference, as a fraction of full scale, at which a tap's weight falls to exp(-1/2).
pub const SIGMA_LUMA: f32 = 0.015;

pub const MODE_OFF: i32 = 0;
/// A chroma plane (U, V, or interleaved UV) with the luma plane as the guide.
pub const MODE_PLANAR: i32 = 1;
/// Packed RGB: the guide is each pixel's own Rec.709 luma; only `rgb - Y` is filtered.
pub const MODE_RGB: i32 = 2;

/// The pass's uniform block. Mirrored by `ChromaFilterParams` in both shaders; 48 bytes.
/// Strides are in bytes. The backends fill the sizes from `Buffers`; settings fill the rest.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ChromaFilterParams {
    pub width: i32, pub height: i32, pub stride: i32, pub channels: i32,
    pub guide_width: i32, pub guide_height: i32, pub guide_stride: i32, pub mode: i32,
    pub amount: f32, pub sigma_y: f32, pub max_value: f32, pub _pad: f32,
}
unsafe impl bytemuck::Zeroable for ChromaFilterParams {}
unsafe impl bytemuck::Pod for ChromaFilterParams {}

impl ChromaFilterParams {
    pub fn is_active(&self) -> bool { self.mode != MODE_OFF && self.amount > 0.0 }
}

/// Luma samples per chroma sample along one axis. Rounded, because a plane of odd size has a
/// chroma plane of ceil(n/2): 3841 / 1921 must be 2, not 1.
pub fn guide_ratio(guide: usize, plane: usize) -> usize {
    ((guide + plane / 2) / plane.max(1)).max(1)
}

fn tap_weight(dx: i32, dy: i32, dg: f32, sigma_y: f32) -> f32 {
    (-((dx * dx + dy * dy) as f32) / (2.0 * SIGMA_SPATIAL * SIGMA_SPATIAL) - dg * dg / (2.0 * sigma_y * sigma_y)).exp()
}

/// The luma guide at chroma sample (x, y): the average of the luma block it covers, with
/// guide coordinates clamped so odd-sized planes never read past the guide.
fn guide_at(guide: &[f32], gw: usize, gh: usize, rx: usize, ry: usize, x: usize, y: usize) -> f32 {
    let mut s = 0.0;
    for j in 0..ry {
        for i in 0..rx {
            let gx = (x * rx + i).min(gw - 1);
            let gy = (y * ry + j).min(gh - 1);
            s += guide[gy * gw + gx];
        }
    }
    s / (rx * ry) as f32
}

/// `plane`: `w * h * channels` interleaved samples; `guide`: `gw * gh` luma samples. All
/// normalised to 0..1, no row padding.
pub fn reference_planar(plane: &[f32], w: usize, h: usize, channels: usize, guide: &[f32], gw: usize, gh: usize, amount: f32) -> Vec<f32> {
    if amount <= 0.0 { return plane.to_vec(); }
    let (rx, ry) = (guide_ratio(gw, w), guide_ratio(gh, h));
    let g: Vec<f32> = (0..w * h).map(|i| guide_at(guide, gw, gh, rx, ry, i % w, i / w)).collect();
    let mut out = plane.to_vec();
    for y in 0..h {
        for x in 0..w {
            let g0 = g[y * w + x];
            let mut acc = [0.0f32; 4];
            let mut wsum = 0.0;
            for dy in (-RADIUS..=RADIUS).step_by(STEP as usize) {
                for dx in (-RADIUS..=RADIUS).step_by(STEP as usize) {
                    let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let wt = tap_weight(dx, dy, g[sy * w + sx] - g0, SIGMA_LUMA);
                    for c in 0..channels { acc[c] += plane[(sy * w + sx) * channels + c] * wt; }
                    wsum += wt;
                }
            }
            for c in 0..channels {
                let p = plane[(y * w + x) * channels + c];
                out[(y * w + x) * channels + c] = p + amount * (acc[c] / wsum - p);
            }
        }
    }
    out
}

/// `pixels`: `w * h * channels` (3 or 4) normalised RGB(A). Channel 3 (alpha) is copied.
pub fn reference_rgb(pixels: &[f32], w: usize, h: usize, channels: usize, amount: f32) -> Vec<f32> {
    if amount <= 0.0 { return pixels.to_vec(); }
    let luma = |i: usize| 0.2126 * pixels[i * channels] + 0.7152 * pixels[i * channels + 1] + 0.0722 * pixels[i * channels + 2];
    let mut out = pixels.to_vec();
    for y in 0..h {
        for x in 0..w {
            let i0 = y * w + x;
            let y0 = luma(i0);
            let mut acc = [0.0f32; 3];
            let mut wsum = 0.0;
            for dy in (-RADIUS..=RADIUS).step_by(STEP as usize) {
                for dx in (-RADIUS..=RADIUS).step_by(STEP as usize) {
                    let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let i = sy * w + sx;
                    let yi = luma(i);
                    let wt = tap_weight(dx, dy, yi - y0, SIGMA_LUMA);
                    for c in 0..3 { acc[c] += (pixels[i * channels + c] - yi) * wt; }
                    wsum += wt;
                }
            }
            for c in 0..3 {
                let c0 = pixels[i0 * channels + c] - y0;
                out[i0 * channels + c] = y0 + c0 + amount * (acc[c] / wsum - c0);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(i: usize, seed: u64) -> f32 {
        let mut z = (i as u64).wrapping_add(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn std(v: &[f32]) -> f32 {
        let m = v.iter().sum::<f32>() / v.len() as f32;
        (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32).sqrt()
    }

    #[test]
    fn params_block_is_48_bytes() {
        assert_eq!(std::mem::size_of::<ChromaFilterParams>(), 48);
    }

    #[test]
    fn guide_ratio_rounds_for_odd_sizes() {
        assert_eq!(guide_ratio(3840, 1920), 2);
        assert_eq!(guide_ratio(3841, 1921), 2); // odd luma width: chroma is ceil(w/2)
        assert_eq!(guide_ratio(2160, 2160), 1); // 4:2:2 vertical, 4:4:4
        assert_eq!(guide_ratio(127, 64), 2);
    }

    #[test]
    fn amount_zero_is_the_identity() {
        let (w, h) = (20, 12);
        let plane: Vec<f32> = (0..w * h * 2).map(|i| 0.5 + 0.1 * noise(i, 1)).collect();
        let guide = vec![0.4; w * 2 * h * 2];
        assert_eq!(reference_planar(&plane, w, h, 2, &guide, w * 2, h * 2, 0.0), plane);
        let rgb: Vec<f32> = (0..w * h * 4).map(|i| 0.5 + 0.1 * noise(i, 2)).collect();
        assert_eq!(reference_rgb(&rgb, w, h, 4, 0.0), rgb);
    }

    #[test]
    fn flat_chroma_noise_is_reduced() {
        let (w, h) = (40, 40);
        let plane: Vec<f32> = (0..w * h * 2).map(|i| 0.5 + 0.02 * noise(i, 3)).collect();
        let guide = vec![0.3; w * 2 * h * 2]; // flat luma: every tap has full range weight
        let out = reference_planar(&plane, w, h, 2, &guide, w * 2, h * 2, 1.0);
        assert!(std(&out) < std(&plane) / 4.0, "{} -> {}", std(&plane), std(&out));
    }

    #[test]
    fn colour_does_not_bleed_across_a_luma_edge() {
        // Left half: dark luma, chroma 0.3. Right half: bright luma, chroma 0.7.
        let (w, h) = (32, 16);
        let plane: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 0.3 } else { 0.7 }).collect();
        let guide: Vec<f32> = (0..w * 2 * h * 2).map(|i| if i % (w * 2) < w { 0.1 } else { 0.9 }).collect();
        let out = reference_planar(&plane, w, h, 1, &guide, w * 2, h * 2, 1.0);
        for y in 0..h {
            assert!((out[y * w + w / 2 - 1] - 0.3).abs() < 1e-3, "left of the edge bled: {}", out[y * w + w / 2 - 1]);
            assert!((out[y * w + w / 2] - 0.7).abs() < 1e-3, "right of the edge bled: {}", out[y * w + w / 2]);
        }
    }

    #[test]
    fn rgb_mode_keeps_every_pixels_luma_and_alpha() {
        let (w, h) = (24, 24);
        let px: Vec<f32> = (0..w * h * 4).map(|i| if i % 4 == 3 { 0.8 } else { 0.5 + 0.05 * noise(i, 4) }).collect();
        let out = reference_rgb(&px, w, h, 4, 1.0);
        let y = |p: &[f32]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
        for (a, b) in px.chunks_exact(4).zip(out.chunks_exact(4)) {
            assert!((y(a) - y(b)).abs() < 1e-5, "luma moved");
            assert_eq!(a[3], b[3], "alpha moved");
        }
        // And the chroma did get smoother.
        let cb = |v: &[f32]| v.chunks_exact(4).map(|p| p[2] - y(p)).collect::<Vec<_>>();
        assert!(std(&cb(&out)) < std(&cb(&px)) / 2.0);
    }

    #[test]
    fn odd_sized_plane_reads_stay_inside_the_guide() {
        // 64x64 chroma for a 127x127 luma: the last block's second column/row does not exist.
        let (w, h, gw, gh) = (64, 64, 127, 127);
        let plane = vec![0.5; w * h];
        let guide = vec![0.5; gw * gh];
        let out = reference_planar(&plane, w, h, 1, &guide, gw, gh, 1.0); // must not panic
        assert!(out.iter().all(|v| (v - 0.5).abs() < 1e-6));
    }
}
