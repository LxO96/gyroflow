// Copyright © 2026 Gyroflow

//! Wigglecam's PCG hash and Box-Muller, integer-exact on CPU and GPU (handover "Noise").

/// Salt of the per-pixel noise hash: `hash4(x, y, layer_seed, NOISE_SALT)`.
pub const NOISE_SALT: u32 = 0x3C6E_F372;

pub fn pcg(v: u32) -> u32 {
    let state = v.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}
pub fn hash4(a: u32, b: u32, c: u32, d: u32) -> u32 { pcg(a ^ pcg(b ^ pcg(c ^ pcg(d)))) }

const U24: f64 = 1.0 / 16777216.0;
/// Standard normal from a hash, from two 24-bit uniforms (exact in fp32).
pub fn gaussian(h: u32) -> f64 {
    let h1 = pcg(h);
    let h2 = pcg(h1);
    let u1 = ((h1 >> 8) as f64 * U24).max(U24);
    let u2 = (h2 >> 8) as f64 * U24;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
/// `gaussian` in the precision the GPU kernels compute it, for the reference renderer.
pub fn gaussian_f32(h: u32) -> f32 {
    let h1 = pcg(h);
    let h2 = pcg(h1);
    let u1 = ((h1 >> 8) as f32 / 16777216.0).max(1.0 / 16777216.0);
    let u2 = (h2 >> 8) as f32 / 16777216.0;
    (-2.0 * u1.ln()).sqrt() * (6.283185307179586f32 * u2).cos()
}

/// FNV-1a over UTF-16 code units, as JS `charCodeAt` sees a string: the clip's grain seed.
pub fn fnv1a_utf16(s: &str) -> u32 {
    s.encode_utf16().fold(0x811c_9dc5u32, |h, c| (h ^ c as u32).wrapping_mul(0x0100_0193))
}

/// Wigglecam `FilmPipeline.layerSeed`: one independent noise field per layer and frame.
pub fn layer_seeds(seed: u32, frame: u32) -> [u32; 3] { [0, 1, 2].map(|l| hash4(seed, frame, l, 0x51)) }
