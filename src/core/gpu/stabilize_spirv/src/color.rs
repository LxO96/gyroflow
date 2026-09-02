// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Adrian <adrian.eddy at gmail>

//! Colour adjustments for the rust-gpu backend.
//!
//! This arithmetic is a deliberate, ratified duplicate of `crate::color::apply_color`
//! (src/core/color/mod.rs) and `crate::color::sample_trilinear` (src/core/color/lut.rs) in
//! `gyroflow-core`. Every operation, and their order, must stay identical to that reference:
//!
//! - the lerps are written out as `a + (b - a) * t` and must NOT be replaced with
//!   `glam::Vec3::lerp`, which is `self * (1.0 - s) + rhs * s` and is not bit-identical;
//! - the luma is an explicit left-associated sum, not `Vec3::dot`, which may contract to an
//!   FMA chain with different rounding;
//! - the interpolation fraction is derived from the *clamped* base index, matching
//!   `frac[c] = pos - i0 as f32` in `sample_trilinear`.
//!
//! `gyroflow-core`'s `gpu::spirv_color_tests` asserts exact `f32` equality against the
//! reference; any divergence introduced here will fail there.

use spirv_std::glam::{ Vec3, Vec4, Vec4Swizzles };
use crate::types::KernelParams;
#[cfg(target_arch = "spirv")]
#[allow(unused_imports)]
use crate::types::Float;

/// Mirrored from `LUT_DATA_OFFSET` in src/core/stabilization/mod.rs (it equals
/// `COEFFS.len()`); a change there must be made here too.
pub const LUT_DATA_OFFSET: usize = 488;

fn lerp3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

fn lut_fetch(coeffs: &[f32], r: i32, g: i32, b: i32, n: i32) -> Vec3 {
    let rc = r.clamp(0, n - 1);
    let gc = g.clamp(0, n - 1);
    let bc = b.clamp(0, n - 1);
    let idx = LUT_DATA_OFFSET + ((bc * n * n + gc * n + rc) * 3) as usize;
    Vec3::new(coeffs[idx], coeffs[idx + 1], coeffs[idx + 2])
}

fn lut_sample(coeffs: &[f32], rgb: Vec3, n: i32, domain_min: f32, domain_max: f32) -> Vec3 {
    let scale = 1.0 / (domain_max - domain_min).max(1e-9);
    let normalized = ((rgb - Vec3::splat(domain_min)) * scale).clamp(Vec3::ZERO, Vec3::ONE);
    let pos = normalized * (n - 1) as f32;
    let floored = pos.floor();
    // The fraction is taken from the clamped base, not from `pos - floored`, so it matches
    // `sample_trilinear`'s `frac[c] = pos - i0 as f32` by construction.
    let bx = (floored.x as i32).clamp(0, n - 1);
    let by = (floored.y as i32).clamp(0, n - 1);
    let bz = (floored.z as i32).clamp(0, n - 1);
    let f = pos - Vec3::new(bx as f32, by as f32, bz as f32);

    let c000 = lut_fetch(coeffs, bx,     by,     bz,     n);
    let c100 = lut_fetch(coeffs, bx + 1, by,     bz,     n);
    let c010 = lut_fetch(coeffs, bx,     by + 1, bz,     n);
    let c110 = lut_fetch(coeffs, bx + 1, by + 1, bz,     n);
    let c001 = lut_fetch(coeffs, bx,     by,     bz + 1, n);
    let c101 = lut_fetch(coeffs, bx + 1, by,     bz + 1, n);
    let c011 = lut_fetch(coeffs, bx,     by + 1, bz + 1, n);
    let c111 = lut_fetch(coeffs, bx + 1, by + 1, bz + 1, n);

    let c00 = lerp3(c000, c100, f.x);
    let c10 = lerp3(c010, c110, f.x);
    let c01 = lerp3(c001, c101, f.x);
    let c11 = lerp3(c011, c111, f.x);
    let c0  = lerp3(c00, c10, f.y);
    let c1  = lerp3(c01, c11, f.y);
    lerp3(c0, c1, f.z)
}

pub fn apply_color(px: Vec4, coeffs: &[f32], params: &KernelParams) -> Vec4 {
    let max_value = params.max_pixel_value;
    let inv_max = if max_value > 0.0 { 1.0 / max_value } else { 1.0 };
    let mut c = px.xyz() * inv_max;

    // 1. LUT, blended by strength
    let lut_size = params.color_params_2.w as i32;
    let lut_strength = params.color_params_1.w;
    // The length check mirrors the CPU path, which slices the LUT out of the coeffs buffer
    // with `coeffs.get(start..start + count)` and falls back to no LUT if it is too short.
    // It never fires for the wgpu/OpenCL buffers, which are always built by
    // `build_coeffs_buffer`; it exists because the Qt RHI entry point passes an empty
    // slice for `coeffs` until Task 11 gives it a real LUT binding.
    // `lut_size` comes from a saturating float cast, so bound it before cubing: in i32 the
    // product overflows above ~860 (panic on a debug host build, wrap in SPIR-V, and a
    // wrapped-negative product would make this guard trivially true). 64 is the maximum
    // `parse_cube` accepts (src/core/color/lut.rs).
    let n = lut_size.clamp(0, 64) as usize;
    let lut_fits = lut_size >= 2 && lut_size <= 64 && coeffs.len() >= LUT_DATA_OFFSET + n * n * n * 3;
    if (params.flags & 8192) != 0 && lut_fits && lut_strength > 0.0 {
        let graded = lut_sample(coeffs, c, lut_size, params.color_params_3.x, params.color_params_3.y);
        c = c + (graded - c) * lut_strength;
    }

    // 2. Exposure
    c = c * params.color_params_1.x;

    // 3. White balance
    c = c * Vec3::new(params.color_params_2.x, params.color_params_2.y, params.color_params_2.z);

    // 4. Contrast around mid grey
    c = (c - Vec3::splat(0.5)) * params.color_params_1.y + Vec3::splat(0.5);

    // 5. Shadows / highlights: a cubic Bezier tone curve on luma, applied as a single gain.
    // A step-for-step duplicate of `apply_color` in src/core/color/mod.rs, which carries the
    // derivation and the monotonicity bound. Operation order must not be rearranged.
    let tone_luma = (0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z).clamp(0.0, 1.0);
    let inv_luma = 1.0 - tone_luma;
    let tone = 1.0 + params.color_params_3.z * inv_luma * inv_luma + params.color_params_3.w * tone_luma * inv_luma;
    c = c * tone;

    // 6. Saturation around luma
    let luma = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
    c = Vec3::splat(luma) + (c - Vec3::splat(luma)) * params.color_params_1.z;

    Vec4::new(c.x * max_value, c.y * max_value, c.z * max_value, px.w)
}
