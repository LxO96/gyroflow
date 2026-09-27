// Copyright © 2026 Gyroflow

//! Gyro-guided temporal denoising settings.
//!
//! The camera rotation Gyroflow already computes makes every reference frame
//! land in the right place, so the only knobs here are how many frames to
//! gather (`quality`) and how readily a disagreeing sample is rejected
//! (`luma_strength` / `chroma_strength`).

pub const MIN_STRENGTH: f64 = 0.0;
pub const MAX_STRENGTH: f64 = 1.0;

/// Sigma of the rejection Gaussian at `strength == 0`, in normalised pixel
/// units (the sample values are divided by `max_pixel_value` before comparison).
/// Never zero: it is a divisor.
const SIGMA_MIN: f64 = 0.002;
/// Sigma at `strength == 1`. Above roughly this, genuine detail starts averaging
/// away along with the noise.
const SIGMA_MAX: f64 = 0.050;
/// Chroma's own range, mapped geometrically. Chroma noise is far smaller than luma noise
/// (measured on Sony S-Log3 4K: ~0.9 vs ~15 ten-bit codes frame to frame), so sharing luma's
/// linear range left most of the chroma slider doing nothing: every sigma above ~0.008 gave an
/// identical render. The response is also steep near zero (sigma 0.0002 / 0.002 / 0.004 left
/// 100% / 46% / 32% of the chroma noise), which a linear slider bunches into its first few
/// percent. Geometric spreads it out: 0.5 -> 0.0022, 0.7 -> 0.006 (that clip's saturation),
/// 1.0 -> 0.025 as headroom for footage with coarser chroma noise.
const SIGMA_CHROMA_MIN: f64 = 0.0002;
const SIGMA_CHROMA_MAX: f64 = 0.025;

/// How much memory the reference-frame ring is allowed to occupy, across both
/// host and GPU. `radius_for_size` shrinks `k` rather than exceed it.
const FRAME_BUDGET_BYTES: usize = 512 * 1024 * 1024;
/// Worst-case bytes per pixel across the formats the kernels accept
/// (`AYUV64LE`, `RGBA64BE` and friends are 8; 4:2:2 10-bit is 4).
const WORST_BYTES_PER_PIXEL: usize = 8;

/// The reference rejection weight. Both kernels mirror this arithmetic exactly,
/// the way every kernel mirrors `color::apply_color`; the GPU tests in
/// `tests_gpu.rs` compare against it.
///
/// `distance` is the difference between a reference sample and the centre
/// sample, in the same normalised units as `sigma`. A sample that disagrees
/// strongly contributes almost nothing, which is what keeps a moving subject
/// from ghosting across the temporal average.
pub fn rejection_weight(distance: f32, sigma: f32) -> f32 {
    let s = sigma.max(f32::MIN_POSITIVE); // never divide by zero
    (-(distance * distance) / (2.0 * s * s)).exp()
}

/// The (luma, chroma) weights for one packed RGB reference sample whose difference from the
/// centre is `d`. The kernels split every RGB sample as `Y*(1,1,1) + c` (the Rec.709 weights
/// sum to 1, so `c` has zero luma) and gather the two parts separately, luma with the first
/// weight and chroma with the second - exactly what planar YUV gets by running once per plane.
/// One shared weight would let a chroma rejection throw the luma away too, which is what made
/// a low chroma strength switch luma denoising off in the preview. Both kernels mirror this.
pub fn rgb_gather_weights(d: [f32; 3], sigma_luma: f32, sigma_chroma: f32) -> (f32, f32) {
    let dy = 0.2126 * d[0] + 0.7152 * d[1] + 0.0722 * d[2];
    let dcb = (d[2] - dy) / 1.8556;
    let dcr = (d[0] - dy) / 1.5748;
    (rejection_weight(dy, sigma_luma), rejection_weight((dcb * dcb + dcr * dcr).sqrt(), sigma_chroma))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenoiseQuality {
    /// `k = 1` — three frames. Roughly halves noise power, cheap enough to keep
    /// export throughput close to undenoised.
    Fast,
    /// `k = 3` — seven frames.
    High,
}

impl Default for DenoiseQuality { fn default() -> Self { Self::Fast } }

impl DenoiseQuality {
    pub fn radius(&self) -> usize {
        match self { Self::Fast => 1, Self::High => 3 }
    }
    pub fn id(&self) -> &'static str {
        match self { Self::Fast => "fast", Self::High => "high" }
    }
    pub fn from_id(id: &str) -> Option<Self> {
        match id { "fast" => Some(Self::Fast), "high" => Some(Self::High), _ => None }
    }
}

#[derive(Debug, Clone)]
pub struct DenoiseSettings {
    pub enabled: bool,
    pub quality: DenoiseQuality,
    /// 0..1. Maps to the rejection sigma for plane 0.
    pub luma_strength: f64,
    /// 0..1. Maps to the rejection sigma for planes 1 and above.
    pub chroma_strength: f64,
    /// 0..1. Blend of the spatial chroma pass (`chroma_filter`) into the temporal result;
    /// 0 skips the pass entirely.
    pub chroma_smoothing: f64,
}

impl Default for DenoiseSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            quality: DenoiseQuality::default(),
            luma_strength: 0.5,
            chroma_strength: 0.5,
            chroma_smoothing: 0.5,
        }
    }
}

impl DenoiseSettings {
    /// Reference radius `k`, ignoring resolution. Zero when disabled, which is
    /// what every consumer branches on to skip the work entirely.
    pub fn radius(&self) -> usize {
        if self.enabled { self.quality.radius() } else { 0 }
    }

    /// `radius()` reduced so that `2k+1` frames fit inside `FRAME_BUDGET_BYTES`.
    /// A 4K clip keeps the full radius; 8K drops to `k = 1` rather than
    /// allocating a gigabyte of reference frames.
    pub fn radius_for_size(&self, width: usize, height: usize) -> usize {
        let k = self.radius();
        if k == 0 { return 0; }
        let frame_bytes = width.saturating_mul(height).saturating_mul(WORST_BYTES_PER_PIXEL).max(1);
        let affordable_frames = FRAME_BUDGET_BYTES / frame_bytes;
        if affordable_frames < 3 { return 0; }
        k.min((affordable_frames - 1) / 2)
    }

    /// Rejection sigma for a plane. Plane 0 is luma, everything above is chroma
    /// — `KernelParams::plane_index` runs the kernel once per plane, so a single
    /// invocation only ever needs one of the two.
    pub fn sigma_for_plane(&self, plane_index: i32) -> f32 {
        if plane_index == 0 { self.luma_sigma() } else { self.chroma_sigma() }
    }
    pub fn luma_sigma(&self) -> f32 {
        (SIGMA_MIN + self.luma_strength.clamp(MIN_STRENGTH, MAX_STRENGTH) * (SIGMA_MAX - SIGMA_MIN)) as f32
    }
    /// Also what packed RGB planes (the preview, RGB exports) use for the chroma part of each
    /// difference - see `KernelParams::denoise_rgb`.
    pub fn chroma_sigma(&self) -> f32 {
        let t = self.chroma_strength.clamp(MIN_STRENGTH, MAX_STRENGTH);
        (SIGMA_CHROMA_MIN * (SIGMA_CHROMA_MAX / SIGMA_CHROMA_MIN).powf(t)) as f32
    }

    /// The single definition of every scalar's valid range. Both entry points
    /// that write these values — the project importer and the QML slider path
    /// (`Controller::set_denoise_param`) — go through here, mirroring
    /// `ColorAdjustments::set_param`.
    pub fn set_param(&mut self, name: &str, val: f64) -> bool {
        if !val.is_finite() { return false; }
        match name {
            "luma_strength"   => self.luma_strength   = val.clamp(MIN_STRENGTH, MAX_STRENGTH),
            "chroma_strength" => self.chroma_strength = val.clamp(MIN_STRENGTH, MAX_STRENGTH),
            "chroma_smoothing" => self.chroma_smoothing = val.clamp(MIN_STRENGTH, MAX_STRENGTH),
            _ => return false,
        }
        true
    }

    /// Read back a scalar by the same name `set_param` accepts. Test support for
    /// the clamping test, and used by the QML getter.
    pub fn set_param_value(&self, name: &str) -> f64 {
        match name {
            "luma_strength"   => self.luma_strength,
            "chroma_strength" => self.chroma_strength,
            "chroma_smoothing" => self.chroma_smoothing,
            _ => f64::NAN,
        }
    }

    /// The chroma pass's parameters for a plane of the given mode. Sizes are left at 0: the
    /// backend fills them from the buffers it is actually given.
    pub fn chroma_filter_params(&self, mode: i32, max_value: f32) -> chroma_filter::ChromaFilterParams {
        if !self.enabled || self.chroma_smoothing <= 0.0 { return Default::default(); }
        chroma_filter::ChromaFilterParams {
            mode,
            amount: self.chroma_smoothing.clamp(MIN_STRENGTH, MAX_STRENGTH) as f32,
            sigma_y: chroma_filter::SIGMA_LUMA,
            max_value,
            ..Default::default()
        }
    }

    /// Mirrors `ColorAdjustments::get_checksum`. Every field here changes
    /// rendered pixels, so every field is hashed.
    pub fn get_checksum(&self) -> u64 {
        use std::hash::Hasher;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        h.write_u8(self.enabled as u8);
        h.write_u8(self.quality.radius() as u8);
        for v in [self.luma_strength, self.chroma_strength, self.chroma_smoothing] {
            h.write_u64(v.to_bits());
        }
        h.finish()
    }
}

/// End-to-end GPU tests for the temporal gather. `#[ignore]`d - they need a real device.
pub mod chroma_filter;

#[cfg(test)] mod tests_gpu;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_maps_to_radius() {
        assert_eq!(DenoiseQuality::Fast.radius(), 1);
        assert_eq!(DenoiseQuality::High.radius(), 3);
    }

    #[test]
    fn radius_is_zero_when_disabled() {
        let mut d = DenoiseSettings::default();
        d.quality = DenoiseQuality::High;
        d.enabled = false;
        assert_eq!(d.radius(), 0);
        d.enabled = true;
        assert_eq!(d.radius(), 3);
    }

    #[test]
    fn radius_is_capped_by_resolution() {
        let mut d = DenoiseSettings::default();
        d.enabled = true;
        d.quality = DenoiseQuality::High;
        // 1080p: 1920*1080*4 = ~8.3 MB/frame, budget allows all 7 frames.
        assert_eq!(d.radius_for_size(1920, 1080), 3);
        // 8K: 7680*4320*8 = ~253 MB/frame (WORST_BYTES_PER_PIXEL is 8, not 4 as an
        // earlier draft of this comment assumed). Only 2 frames fit in the 512 MB
        // budget, short of the 3 needed for k=1, so radius drops all the way to 0.
        assert_eq!(d.radius_for_size(7680, 4320), 0);
        // Absurd resolution still leaves a usable k=0 rather than panicking.
        assert_eq!(d.radius_for_size(60000, 40000), 0);
    }

    #[test]
    fn radius_for_size_is_zero_when_disabled() {
        let d = DenoiseSettings::default(); // enabled: false
        assert_eq!(d.radius_for_size(1920, 1080), 0);
    }

    #[test]
    fn set_param_clamps_every_scalar_to_its_range() {
        let mut d = DenoiseSettings::default();
        for name in ["luma_strength", "chroma_strength", "chroma_smoothing"] {
            assert!(d.set_param(name, -1e6));
            assert_eq!(d.set_param_value(name), MIN_STRENGTH);
            assert!(d.set_param(name, 1e6));
            assert_eq!(d.set_param_value(name), MAX_STRENGTH);
        }
    }

    #[test]
    fn set_param_rejects_unknown_names_and_non_finite_values() {
        let mut d = DenoiseSettings::default();
        d.luma_strength = 0.5;
        assert!(!d.set_param("radius", 3.0));
        assert!(!d.set_param("luma_strength", f64::NAN));
        assert!(!d.set_param("luma_strength", f64::INFINITY));
        assert_eq!(d.luma_strength, 0.5);
    }

    #[test]
    fn quality_id_round_trips() {
        for q in [DenoiseQuality::Fast, DenoiseQuality::High] {
            assert_eq!(DenoiseQuality::from_id(q.id()), Some(q));
        }
        assert_eq!(DenoiseQuality::from_id("nonsense"), None);
    }

    #[test]
    fn every_pixel_affecting_field_changes_the_checksum() {
        let base = DenoiseSettings::default();
        for (name, m) in [
            ("enabled",         { let mut d = base.clone(); d.enabled = !d.enabled; d }),
            ("quality",         { let mut d = base.clone(); d.quality = DenoiseQuality::High; d }),
            ("luma_strength",   { let mut d = base.clone(); d.luma_strength = 0.123; d }),
            ("chroma_strength", { let mut d = base.clone(); d.chroma_strength = 0.456; d }),
            ("chroma_smoothing", { let mut d = base.clone(); d.chroma_smoothing = 0.9; d }),
        ] {
            assert_ne!(base.get_checksum(), m.get_checksum(), "{name} must invalidate the preview");
        }
    }

    #[test]
    fn rejection_weight_is_one_at_zero_distance_and_falls_off() {
        let sigma = 0.05;
        assert_eq!(rejection_weight(0.0, sigma), 1.0);
        // One sigma away is exp(-0.5).
        assert!((rejection_weight(sigma, sigma) - (-0.5f32).exp()).abs() < 1e-6);
        // Far-away samples are rejected hard, which is what stops moving
        // subjects from ghosting.
        assert!(rejection_weight(sigma * 6.0, sigma) < 1e-7);
    }

    #[test]
    fn rejection_weight_is_symmetric_and_monotonic() {
        let sigma = 0.05;
        assert_eq!(rejection_weight(0.03, sigma), rejection_weight(-0.03, sigma));
        let mut prev = f32::INFINITY;
        for i in 0..20 {
            let w = rejection_weight(i as f32 * 0.01, sigma);
            assert!(w < prev, "weight must fall monotonically with distance");
            prev = w;
        }
    }

    #[test]
    fn chroma_smoothing_defaults_clamps_and_changes_the_checksum() {
        let mut d = DenoiseSettings::default();
        assert_eq!(d.chroma_smoothing, 0.5);
        assert!(d.set_param("chroma_smoothing", 7.0));
        assert_eq!(d.set_param_value("chroma_smoothing"), MAX_STRENGTH);
        let before = d.get_checksum();
        d.chroma_smoothing = 0.2;
        assert_ne!(before, d.get_checksum());
    }

    #[test]
    fn chroma_filter_params_are_inactive_unless_enabled_and_nonzero() {
        let mut d = DenoiseSettings::default();
        d.enabled = false;
        assert!(!d.chroma_filter_params(chroma_filter::MODE_RGB, 1.0).is_active(), "disabled denoise must not smooth");
        d.enabled = true;
        d.chroma_smoothing = 0.0;
        assert!(!d.chroma_filter_params(chroma_filter::MODE_RGB, 1.0).is_active());
        d.chroma_smoothing = 0.5;
        let p = d.chroma_filter_params(chroma_filter::MODE_PLANAR, 65535.0);
        assert!(p.is_active());
        assert_eq!((p.mode, p.amount, p.sigma_y, p.max_value), (chroma_filter::MODE_PLANAR, 0.5, chroma_filter::SIGMA_LUMA, 65535.0));
    }

    #[test]
    fn rgb_weight_judges_luma_and_chroma_by_their_own_sigma() {
        let (loose, tight) = (0.5, 0.002);
        let chroma_only = [0.05, 0.0, -0.05 * 0.2126 / 0.0722]; // dY = 0
        let luma_only = [0.05, 0.05, 0.05];                     // dCb = dCr = 0
        // A chroma-only change is judged on chroma alone, and never costs the luma its weight.
        let (wl, wc) = rgb_gather_weights(chroma_only, loose, tight);
        assert!(wc < 1e-6 && wl > 0.999, "chroma change: luma {wl}, chroma {wc}");
        let (wl, wc) = rgb_gather_weights(chroma_only, tight, loose);
        assert!(wl > 0.999 && wc > 0.9, "chroma change, loose chroma: luma {wl}, chroma {wc}");
        // And the reverse.
        let (wl, wc) = rgb_gather_weights(luma_only, tight, loose);
        assert!(wl < 1e-6 && wc > 0.999, "luma change: luma {wl}, chroma {wc}");
        assert_eq!(rgb_gather_weights([0.0; 3], tight, tight), (1.0, 1.0));
    }

    #[test]
    fn chroma_range_ends_well_below_luma() {
        // Measured on Sony S-Log3 4K: every chroma sigma above ~0.008 gave an identical result,
        // so a slider sharing luma's range (up to 0.05) did nothing over most of its travel.
        let mut d = DenoiseSettings::default();
        d.chroma_strength = 1.0;
        assert!(d.chroma_sigma() <= 0.03, "chroma max {}", d.chroma_sigma());
        d.chroma_strength = 0.7;
        assert!((d.chroma_sigma() - 0.006).abs() < 0.001, "the measured saturation point sits at 0.7: {}", d.chroma_sigma());
        d.chroma_strength = 0.0;
        assert!(d.chroma_sigma() > 0.0 && d.chroma_sigma() < 0.001, "chroma min {}", d.chroma_sigma());
        // Geometric: each step multiplies sigma by the same factor, and it never decreases.
        let at = |s: f64| { let mut d = DenoiseSettings::default(); d.chroma_strength = s; d.chroma_sigma() as f64 };
        assert!(((at(0.5) / at(0.25)) - (at(0.75) / at(0.5))).abs() < 1e-3);
        assert!((0..10).all(|i| at(i as f64 / 10.0) < at((i + 1) as f64 / 10.0)));
        // Planes 1+ use exactly this.
        assert_eq!(d.sigma_for_plane(1), d.chroma_sigma());
        assert_eq!(d.sigma_for_plane(2), d.chroma_sigma());
    }

    #[test]
    fn sigma_rises_with_strength_and_selects_by_plane() {
        let mut d = DenoiseSettings::default();
        d.luma_strength = 0.0;
        d.chroma_strength = 1.0;
        // plane 0 is luma, planes 1+ are chroma
        assert!(d.sigma_for_plane(0) < d.sigma_for_plane(1));
        assert!(d.sigma_for_plane(1) == d.sigma_for_plane(2));
        assert!(d.sigma_for_plane(0) > 0.0, "sigma must never be zero - it divides");
    }
}
