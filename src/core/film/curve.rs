// Copyright © 2026 Gyroflow

//! Wigglecam's characteristic curve and the per-stock re-centring fit (handover "Tone curves").

use super::stocks::{ Layer, Stock };

pub fn srgb_decode(v: f64) -> f64 { if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) } }
pub fn srgb_encode(x: f64) -> f64 { if x <= 0.0031308 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 } }

pub const MID: f64 = 0.18;
/// `srgb_encode(0.18)`: mid-grey stays put.
pub const MID_OUT: f64 = 0.46135612950044164;
pub const WHITE_OUT: f64 = 0.96;

/// Developed fraction (display lightness) for log10 exposure: a logistic with slope `gamma` at `log_e0`.
pub fn develop(log_e: f64, l: &Layer) -> f64 {
    let range = l.u_max - l.u_min;
    let k = 4.0 * l.gamma / range;
    l.u_min + range / (1.0 + (-k * (log_e - l.log_e0)).exp())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fit { pub b: f64, pub s: f64, pub u_min_a: f64 }

/// One exposure shift `b` and stretch `s`, fitted on the anchor layer (green; layer 0 for B&W)
/// so linear 0.18 -> `MID_OUT` and 1.0 -> `WHITE_OUT`. 80-step bisection: the mid rises with b.
pub fn fit_curve(stock: &Stock) -> Fit {
    let l = &stock.layers[if stock.bw { 0 } else { 1 }];
    let at = |e: f64, b: f64| develop(e.log10() + b, l);
    let mid_for = |b: f64| {
        let s = (WHITE_OUT - l.u_min) / (at(1.0, b) - l.u_min);
        (s, l.u_min + s * (at(MID, b) - l.u_min))
    };
    let (mut lo, mut hi) = (-4.0f64, 4.0f64);
    for _ in 0..80 {
        let m = (lo + hi) / 2.0;
        if mid_for(m).1 < MID_OUT { lo = m; } else { hi = m; }
    }
    let b = (lo + hi) / 2.0;
    Fit { b, s: mid_for(b).0, u_min_a: l.u_min }
}

/// Linear rgb -> the three developed layers (display values, before grain).
pub fn curve_pixel(lin: [f64; 3], stock: &Stock, fit: &Fit) -> [f64; 3] {
    let m = &stock.sensitivity;
    let mut out = [0.0; 3];
    for i in 0..3 {
        let e = m[i * 3] * lin[0] + m[i * 3 + 1] * lin[1] + m[i * 3 + 2] * lin[2];
        let u = develop(e.max(1e-5).log10() + fit.b, &stock.layers[i]);
        out[i] = fit.u_min_a + fit.s * (u - fit.u_min_a);
    }
    out
}
