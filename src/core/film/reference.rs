// Copyright © 2026 Gyroflow

//! What the GPU film chain does, in f32, pass by pass. Both kernels are tested against this.
//! Values are 0..1; `max_value` is ignored here.

use super::params::*;
use super::noise::{ gaussian_f32, hash4, NOISE_SALT };

fn decode(v: f32) -> f32 { if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) } }
fn encode(x: f32) -> f32 { if x <= 0.0031308 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 } }
fn smoothstep(a: f32, b: f32, x: f32) -> f32 { let t = ((x - a) / (b - a)).clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) }
/// In-shader weights: `exp(-k²/2σ²)`, normalised by their sum over [-r, r].
fn weights(r: i32, sigma: f32) -> Vec<f32> {
    let w: Vec<f32> = (-r..=r).map(|k| (-((k * k) as f32) / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = w.iter().sum();
    w.into_iter().map(|x| x / s).collect()
}
/// `std(u)`, linearly interpolated between the 201 table entries. Interpolated rather than
/// nearest (Photo360 rounds): a nearest lookup flips at each step boundary, so f32 differences
/// between backends showed up as whole-step jumps in grain amplitude.
pub(crate) fn std_of(p: &FilmParams, u: f32) -> f32 {
    let f = u.clamp(0.0, 1.0) * 200.0;
    let i = (f.floor() as usize).min(199);
    let t = f - i as f32;
    p.std_table[i] * (1.0 - t) + p.std_table[i + 1] * t
}

/// The export pixel under render pixel `x`.
fn export_px(x: i32, inv_p: f32) -> u32 { ((x as f32 + 0.5) * inv_p).floor() as i32 as u32 }

/// Pass G1 then the vertical half of G2: blurred unit noise per layer.
fn noise_field(w: usize, h: usize, p: &FilmParams) -> Vec<[f32; 3]> {
    let layers = if p.flags & FLAG_BW != 0 { 1 } else { 3 };
    let r = p.grain_radius;
    let k = weights(r, p.grain_sigma);
    let mut g1 = vec![[0f32; 3]; w * h];
    for y in 0..h { for x in 0..w { for l in 0..layers {
        let mut s = 0.0;
        for i in -r..=r {
            let xx = (x as i32 + i).clamp(0, w as i32 - 1);
            s += k[(i + r) as usize] * gaussian_f32(hash4(export_px(xx, p.inv_p), export_px(y as i32, p.inv_p), p.seeds[l], NOISE_SALT));
        }
        g1[y * w + x][l] = s;
    } } }
    let mut out = vec![[0f32; 3]; w * h];
    for y in 0..h { for x in 0..w { for l in 0..layers {
        let mut s = 0.0;
        for i in -r..=r { s += k[(i + r) as usize] * g1[(y as i32 + i).clamp(0, h as i32 - 1) as usize * w + x][l]; }
        out[y * w + x][l] = s;
    } } }
    out
}

/// Passes H1-H3: the halation field at 1/f resolution.
fn halo_field(src: &[[f32; 4]], w: usize, h: usize, p: &FilmParams) -> Vec<f32> {
    let (f, hw, hh) = (p.halo_f as usize, p.halo_w as usize, p.halo_h as usize);
    let mut a = vec![0f32; hw * hh];
    for j in 0..hh { for i in 0..hw {
        let mut s = 0.0;
        for b in 0..f { for c in 0..f {
            let px = src[(j * f + b).min(h - 1) * w + (i * f + c).min(w - 1)];
            let y = 0.2126 * decode(px[0]) + 0.7152 * decode(px[1]) + 0.0722 * decode(px[2]);
            s += smoothstep(0.75, 1.0, y) * y;
        } }
        a[j * hw + i] = s / (f * f) as f32;
    } }
    let r = p.halo_radius;
    let k = weights(r, p.halo_sigma);
    let mut b = vec![0f32; hw * hh];
    for j in 0..hh { for i in 0..hw {
        b[j * hw + i] = (-r..=r).map(|t| k[(t + r) as usize] * a[j * hw + (i as i32 + t).clamp(0, hw as i32 - 1) as usize]).sum();
    } }
    for j in 0..hh { for i in 0..hw {
        a[j * hw + i] = (-r..=r).map(|t| k[(t + r) as usize] * b[(j as i32 + t).clamp(0, hh as i32 - 1) as usize * hw + i]).sum();
    } }
    a
}

/// Bilinear, clamp-to-edge, at halo-texel coordinates (texel centres at +0.5).
fn sample_halo(field: &[f32], p: &FilmParams, hx: f32, hy: f32) -> f32 {
    let (fx, fy) = (hx - 0.5, hy - 0.5);
    let (x0, y0) = (fx.floor() as i32, fy.floor() as i32);
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let at = |x: i32, y: i32| field[y.clamp(0, p.halo_h - 1) as usize * p.halo_w as usize + x.clamp(0, p.halo_w - 1) as usize];
    (at(x0, y0) * (1.0 - tx) + at(x0 + 1, y0) * tx) * (1.0 - ty) + (at(x0, y0 + 1) * (1.0 - tx) + at(x0 + 1, y0 + 1) * tx) * ty
}

pub fn render(src: &[[f32; 4]], w: usize, h: usize, p: &FilmParams) -> Vec<[f32; 4]> {
    let noise = (p.flags & FLAG_GRAIN != 0).then(|| noise_field(w, h, p));
    let halo = (p.flags & FLAG_HALATION != 0).then(|| halo_field(src, w, h, p));
    let bw = p.flags & FLAG_BW != 0;
    let mut out = vec![[0f32; 4]; w * h];
    for y in 0..h { for x in 0..w {
        let c = src[y * w + x];
        let mut lin = [decode(c[0]), decode(c[1]), decode(c[2])];
        if let Some(field) = &halo {
            let f = p.halo_f as f32;
            let hv = sample_halo(field, p, (x as f32 + 0.5) / f, (y as f32 + 0.5) / f);
            for i in 0..3 { lin[i] += p.halo_strength * hv * p.halo_tint[i]; }
        }
        let mut u = [0f32; 3];
        if p.flags & FLAG_TONE != 0 {
            for i in 0..3 {
                let e = p.sens[i][0] * lin[0] + p.sens[i][1] * lin[1] + p.sens[i][2] * lin[2];
                let log_e = e.max(1e-5).log10() + p.fit_b;
                let d = p.u_min[i] + p.range[i] / (1.0 + (-p.k[i] * (log_e - p.log_e0[i])).exp());
                u[i] = p.fit_umin_a + p.fit_s * (d - p.fit_umin_a);
            }
        } else {
            for i in 0..3 { u[i] = encode(lin[i].max(0.0)); }
        }
        if let Some(n) = &noise {
            let n = n[y * w + x];
            for i in 0..3 {
                let sd = std_of(p, u[i]);
                u[i] = (u[i] + p.grain_k * sd * n[if bw { 0 } else { i }]).clamp(0.0, 1.0);
            }
        }
        let o = if p.flags & FLAG_TONE != 0 {
            [0, 1, 2].map(|i| (p.cross[i][0] * u[0] + p.cross[i][1] * u[1] + p.cross[i][2] * u[2]).clamp(0.0, 1.0))
        } else {
            u.map(|v| v.clamp(0.0, 1.0))
        };
        out[y * w + x] = [o[0], o[1], o[2], c[3]];
    } }
    out
}
