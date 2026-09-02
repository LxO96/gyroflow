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
}

impl Default for DenoiseSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            quality: DenoiseQuality::default(),
            luma_strength: 0.5,
            chroma_strength: 0.5,
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
        let strength = if plane_index == 0 { self.luma_strength } else { self.chroma_strength };
        (SIGMA_MIN + strength.clamp(MIN_STRENGTH, MAX_STRENGTH) * (SIGMA_MAX - SIGMA_MIN)) as f32
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
            _ => f64::NAN,
        }
    }

    /// Mirrors `ColorAdjustments::get_checksum`. Every field here changes
    /// rendered pixels, so every field is hashed.
    pub fn get_checksum(&self) -> u64 {
        use std::hash::Hasher;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        h.write_u8(self.enabled as u8);
        h.write_u8(self.quality.radius() as u8);
        for v in [self.luma_strength, self.chroma_strength] {
            h.write_u64(v.to_bits());
        }
        h.finish()
    }
}

/// End-to-end GPU tests for the temporal gather. `#[ignore]`d - they need a real device.
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
        for name in ["luma_strength", "chroma_strength"] {
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
