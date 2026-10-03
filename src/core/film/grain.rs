// Copyright © 2026 Gyroflow

//! Wigglecam's silver-halide grain model and its calibration to blurred noise (handover "Grain").

use std::sync::Arc;
use super::stocks::GrainSpec;

const MAX_U: f64 = 0.995;
const QUANTILES: usize = 32;
pub const MAX_KERNEL_RADIUS: usize = 31;
pub const STD_TABLE: usize = 200;

/// Inverse standard normal CDF (Acklam, |error| < 1.2e-9).
pub fn inverse_normal(p: f64) -> f64 {
    const A: [f64; 6] = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const B: [f64; 5] = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const C: [f64; 6] = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const D: [f64; 4] = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const LOW: f64 = 0.02425;
    let tail = |q: f64| (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    if p < LOW { return tail((-2.0 * p.ln()).sqrt()); }
    if p <= 1.0 - LOW {
        let q = p - 0.5;
        let r = q * q;
        return (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0);
    }
    -tail((-2.0 * (1.0 - p).ln()).sqrt())
}

/// e^-z I0(z), z >= 0 (Abramowitz & Stegun 9.8.1-9.8.2).
pub fn i0e(z: f64) -> f64 {
    if z <= 3.75 {
        let t = (z / 3.75) * (z / 3.75);
        let i0 = 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
        return i0 * (-z).exp();
    }
    let t = 3.75 / z;
    (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))))) / z.sqrt()
}

/// Overlap area of two disks of radius r whose centres are d apart.
pub fn lens_area(r: f64, d: f64) -> f64 {
    if d >= 2.0 * r { return 0.0; }
    2.0 * r * r * (d / (2.0 * r)).acos() - d / 2.0 * (4.0 * r * r - d * d).sqrt()
}

/// Boolean model of disks with log-normal radii at Poisson centres, seen through a Gaussian
/// filter. Lengths in render pixels.
#[derive(Clone, Debug)]
pub struct GrainModel { radius_median: f64, radius_log_sigma: f64, filter_sigma: f64, mean_area: f64, radii: Vec<f64>, r_max: f64 }

impl GrainModel {
    pub fn new(radius_median: f64, radius_log_sigma: f64, filter_sigma: f64) -> Self {
        let mean_area = std::f64::consts::PI * radius_median * radius_median * (2.0 * radius_log_sigma * radius_log_sigma).exp();
        let radii: Vec<f64> = (0..QUANTILES).map(|i| radius_median * (radius_log_sigma * inverse_normal((i as f64 + 0.5) / QUANTILES as f64)).exp()).collect();
        let r_max = radii.iter().cloned().fold(f64::MIN, f64::max);
        Self { radius_median, radius_log_sigma, filter_sigma, mean_area, radii, r_max }
    }
    fn lambda(&self, u: f64) -> f64 { -(1.0 - u.clamp(0.0, MAX_U)).ln() / self.mean_area }
    fn covariance(&self, u: f64, d: f64) -> f64 {
        if u <= 0.0 { return 0.0; }
        let q = 1.0 - u.clamp(0.0, MAX_U);
        let overlap = self.radii.iter().map(|&r| lens_area(r, d)).sum::<f64>() / self.radii.len() as f64;
        q * q * ((self.lambda(u) * overlap).exp() - 1.0)
    }
    /// Covariance after the Gaussian filter at offset `delta` (Simpson, 400 steps on [0, 2 r_max]).
    pub fn filtered_covariance(&self, u: f64, delta: f64) -> f64 {
        if u <= 0.0 { return 0.0; }
        let (upper, n) = (2.0 * self.r_max, 400);
        let h = upper / n as f64;
        let s2 = self.filter_sigma * self.filter_sigma;
        let f = |r: f64| self.covariance(u, r) * (-(r - delta) * (r - delta) / (4.0 * s2)).exp() * i0e(r * delta / (2.0 * s2)) * r / (2.0 * s2);
        let mut sum = f(0.0) + f(upper);
        for i in 1..n { sum += if i % 2 == 1 { 4.0 } else { 2.0 } * f(i as f64 * h); }
        sum * h / 3.0
    }
    pub fn filtered_variance(&self, u: f64) -> f64 { self.filtered_covariance(u, 0.0) }
    pub fn lag_one_correlation(&self, u: f64) -> f64 {
        let v = self.filtered_covariance(u, 0.0);
        if v <= 0.0 { 0.0 } else { self.filtered_covariance(u, 1.0) / v }
    }
    /// The same film rendered `s`x larger (s < 1 adds the pixel-box term 1/12).
    pub fn for_render_scale(&self, s: f64) -> Self {
        Self::new(self.radius_median * s, self.radius_log_sigma,
            (self.filter_sigma * self.filter_sigma * s * s + (1.0 - s * s).max(0.0) / 12.0).sqrt())
    }
}

pub fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let r = ((3.0 * sigma).ceil() as usize).clamp(1, MAX_KERNEL_RADIUS);
    let mut k: Vec<f64> = (0..2 * r + 1).map(|i| { let d = i as f64 - r as f64; (-(d * d) / (2.0 * sigma * sigma)).exp() }).collect();
    let sum: f64 = k.iter().sum();
    k.iter_mut().for_each(|w| *w /= sum);
    k
}
/// Lag-1 autocorrelation of white noise after this separable kernel.
pub fn kernel_lag_one(sigma: f64) -> f64 {
    let k = gaussian_kernel(sigma);
    let den: f64 = k.iter().map(|w| w * w).sum();
    let num: f64 = k.windows(2).map(|w| w[0] * w[1]).sum();
    num / den
}
/// Sigma whose blurred noise has the target lag-1 correlation (60-step bisection).
pub fn calibrate_noise_sigma(target: f64) -> f64 {
    let (mut lo, mut hi) = (0.05f64, 20.0f64);
    for _ in 0..60 {
        let m = (lo + hi) / 2.0;
        if kernel_lag_one(m) < target { lo = m; } else { hi = m; }
    }
    (lo + hi) / 2.0
}
/// Scales are cached at 1/8-octave steps.
pub fn quantize_scale(v: f64) -> f64 { 2f64.powf((8.0 * v.log2()).round() / 8.0) }

#[derive(Clone, Debug)]
pub struct GrainSetup {
    pub sigma: f64,
    pub radius: usize,
    /// `1 / Σw²` of the 1-D kernel: restores unit std after the separable blur.
    pub norm: f64,
    pub std: [f32; STD_TABLE + 1],
    /// The model's lag-1 correlation at u = 0.5, for the statistics tests.
    pub lag_one: f64,
}

/// Everything a render needs for one stock's grain at export scale `scale_export`, seen at
/// `z` render px per export px (1 for the export). Tens of ms to build, so cached.
pub fn grain_setup(spec: &GrainSpec, scale_export: f64, z: f64) -> Arc<GrainSetup> {
    type Key = [u64; 5];
    static CACHE: std::sync::OnceLock<parking_lot::Mutex<Vec<(Key, Arc<GrainSetup>)>>> = std::sync::OnceLock::new();
    let key: Key = [spec.radius_median_px.to_bits(), spec.radius_log_sigma.to_bits(), spec.filter_sigma_px.to_bits(), scale_export.to_bits(), z.to_bits()];
    let cache = CACHE.get_or_init(Default::default);
    if let Some((_, s)) = cache.lock().iter().find(|(k, _)| *k == key) { return s.clone(); }

    let mut model = GrainModel::new(spec.radius_median_px, spec.radius_log_sigma, spec.filter_sigma_px).for_render_scale(scale_export);
    if z != 1.0 { model = model.for_render_scale(z); }
    let lag_one = model.lag_one_correlation(0.5);
    let sigma = calibrate_noise_sigma(lag_one);
    let k = gaussian_kernel(sigma);
    let norm = 1.0 / k.iter().map(|w| w * w).sum::<f64>();
    let mut std = [0f32; STD_TABLE + 1];
    for (i, s) in std.iter_mut().enumerate() { *s = model.filtered_variance(i as f64 / STD_TABLE as f64).max(0.0).sqrt() as f32; }
    let setup = Arc::new(GrainSetup { sigma, radius: (k.len() - 1) / 2, norm, std, lag_one });

    let mut c = cache.lock();
    if c.len() > 16 { c.remove(0); }
    c.push((key, setup.clone()));
    setup
}
