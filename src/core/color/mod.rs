// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Adrian <adrian.eddy at gmail>

pub mod lut;
pub mod output_range;
pub mod output_space;
pub mod source_metadata;
pub mod white_balance;

pub use lut::{ Lut3D, LutError, parse_cube, sample_trilinear };
pub use output_range::OutputRange;
pub use output_space::OutputColorSpace;
pub use source_metadata::SourceExposure;
pub use white_balance::kelvin_to_rgb_gains;

use std::sync::Arc;

pub const DEFAULT_KELVIN: f64 = 5500.0;
/// The range over which `white_balance::cct_to_xy` is defined.
pub const MIN_KELVIN: f64 = 1667.0;
pub const MAX_KELVIN: f64 = 25000.0;

/// Exposure bounds, in stops. Shared by `set_param` and `to_kernel_params` so the static
/// and keyframe paths cannot drift apart.
pub const MIN_EXPOSURE: f64 = -5.0;
pub const MAX_EXPOSURE: f64 =  5.0;

#[derive(Debug, Clone)]
pub struct ColorAdjustments {
    pub enabled: bool,
    pub bake_in_render: bool,
    pub exposure: f64,               // stops, -5..5, 0 neutral
    pub contrast: f64,               // -1..1, 0 neutral
    pub saturation: f64,             // -1..1, 0 neutral
    pub source_kelvin: f64,
    pub source_kelvin_detected: bool,
    pub target_kelvin: f64,
    pub tint: f64,                   // -1..1, 0 neutral
    pub lut_strength: f64,           // 0..1
    pub lut: Option<Arc<Lut3D>>,
    pub lut_path: Option<String>,
    pub lut_name: Option<String>,
    /// What the *exported file* declares itself to be. Tags only - never affects a pixel,
    /// and deliberately absent from `get_checksum` so changing it cannot invalidate the
    /// preview or trigger a recompute.
    pub output_color_space: OutputColorSpace,
    /// What range the *exported file* is encoded in. Unlike `output_color_space` this one does
    /// change pixels - but only during encoding, never in the stabilization kernels, so it is
    /// likewise absent from `get_checksum`: the preview is unaffected and must not be
    /// invalidated by it.
    pub output_range: OutputRange,
}

impl Default for ColorAdjustments {
    fn default() -> Self {
        Self {
            enabled: false,
            bake_in_render: false,
            exposure: 0.0,
            contrast: 0.0,
            saturation: 0.0,
            source_kelvin: DEFAULT_KELVIN,
            source_kelvin_detected: false,
            target_kelvin: DEFAULT_KELVIN,
            tint: 0.0,
            lut_strength: 1.0,
            lut: None,
            lut_path: None,
            lut_name: None,
            output_color_space: OutputColorSpace::default(),
            output_range: OutputRange::default(),
        }
    }
}

/// Ready-to-multiply values handed to every kernel. No colour science here —
/// it has all been done in Rust by the time this struct is built.
#[derive(Debug, Clone, Copy, Default)]
pub struct ColorKernelParams {
    pub exposure_gain: f32,
    pub contrast_factor: f32,
    pub saturation_factor: f32,
    pub lut_strength: f32,
    pub wb_gains: [f32; 3],
    pub lut_size: u32,
    pub domain_min: f32,
    pub domain_max: f32,
}

impl ColorAdjustments {
    /// The single definition of every scalar's valid range. Both entry points that can
    /// write these values - the project importer (`StabilizationManager::import_gyroflow_data`)
    /// and the QML slider path (`Controller::set_color_param`) - go through here, so a value
    /// can't enter the struct out of range whichever way it arrives.
    ///
    /// `exposure` is the one that matters: nothing downstream bounds it, and `1e6` gives an
    /// infinite gain and a white frame. The others are clamped again further down
    /// (`to_kernel_params`, `kelvin_to_rgb_gains`); doing it here too keeps the stored value
    /// honest, so the UI shows what is actually applied.
    ///
    /// Non-finite values are rejected rather than clamped - `f64::clamp` passes NaN through,
    /// and a NaN scalar NaNs every pixel silently. JSON has no NaN literal so the importer
    /// can't produce one; QML can.
    ///
    /// Returns false, having changed nothing, if the name is unknown or the value is not finite.
    pub fn set_param(&mut self, name: &str, val: f64) -> bool {
        if !val.is_finite() { return false; }
        match name {
            "exposure"      => self.exposure      = val.clamp(MIN_EXPOSURE, MAX_EXPOSURE),
            "contrast"      => self.contrast      = val.clamp(-1.0, 1.0),
            "saturation"    => self.saturation    = val.clamp(-1.0, 1.0),
            "tint"          => self.tint          = val.clamp(-1.0, 1.0),
            "source_kelvin" => self.source_kelvin = val.clamp(MIN_KELVIN, MAX_KELVIN),
            "target_kelvin" => self.target_kelvin = val.clamp(MIN_KELVIN, MAX_KELVIN),
            "lut_strength"  => self.lut_strength  = val.clamp(0.0, 1.0),
            _ => return false,
        }
        true
    }

    pub fn is_active(&self) -> bool {
        self.enabled && (
            self.lut.is_some()
            || self.exposure != 0.0 || self.contrast != 0.0 || self.saturation != 0.0
            || self.tint != 0.0
            || (self.target_kelvin - self.source_kelvin).abs() > f64::EPSILON
        )
    }

    /// Scalars come in already resolved for this frame (keyframes applied by the caller).
    ///
    /// Every scalar is clamped here, including exposure. These values arrive from the
    /// *keyframe* path, which never passes through `set_param`, so the import clamps and the
    /// slider bounds are both bypassed - a hand-edited `keyframes` section, or a value typed
    /// into the unbounded `NumberField`, lands here raw. Unclamped, `exposure.exp2()` reaches
    /// `+inf` around 1024 and renders a white frame with no diagnostic.
    pub fn to_kernel_params(&self, exposure: f64, contrast: f64, saturation: f64, target_kelvin: f64, tint: f64, lut_strength: f64) -> ColorKernelParams {
        ColorKernelParams {
            exposure_gain:     exposure.clamp(MIN_EXPOSURE, MAX_EXPOSURE).exp2() as f32,
            contrast_factor:   (1.0 + contrast.clamp(-1.0, 1.0)) as f32,
            saturation_factor: (1.0 + saturation.clamp(-1.0, 1.0)) as f32,
            lut_strength:      lut_strength.clamp(0.0, 1.0) as f32,
            wb_gains:          kelvin_to_rgb_gains(self.source_kelvin, target_kelvin, tint),
            lut_size:          self.lut.as_ref().map(|l| l.size).unwrap_or(0),
            domain_min:        self.lut.as_ref().map(|l| l.domain_min).unwrap_or(0.0),
            domain_max:        self.lut.as_ref().map(|l| l.domain_max).unwrap_or(1.0),
        }
    }

    pub fn get_checksum(&self) -> u64 {
        use std::hash::Hasher;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        h.write_u8(self.enabled as u8);
        h.write_u8(self.bake_in_render as u8);
        for v in [self.exposure, self.contrast, self.saturation, self.source_kelvin, self.target_kelvin, self.tint, self.lut_strength] {
            h.write_u64(v.to_bits());
        }
        h.write_u32(self.lut.as_ref().map(|l| l.checksum).unwrap_or(0));
        h.finish()
    }
}

/// The reference colour stage. Every kernel implementation mirrors this
/// arithmetic exactly; cross-backend tests compare against it.
///
/// Input and output are display-referred RGB in 0..1.
pub fn apply_color(rgb: [f32; 3], p: &ColorKernelParams, lut: Option<&Lut3D>) -> [f32; 3] {
    let mut c = rgb;

    // 1. LUT, blended by strength
    if let Some(lut) = lut {
        if p.lut_strength > 0.0 {
            let graded = sample_trilinear(lut, c);
            for i in 0..3 { c[i] += (graded[i] - c[i]) * p.lut_strength; }
        }
    }

    // 2. Exposure
    for i in 0..3 { c[i] *= p.exposure_gain; }

    // 3. White balance
    for i in 0..3 { c[i] *= p.wb_gains[i]; }

    // 4. Contrast around mid grey
    for i in 0..3 { c[i] = (c[i] - 0.5) * p.contrast_factor + 0.5; }

    // 5. Saturation around luma
    let luma = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    for i in 0..3 { c[i] = luma + (c[i] - luma) * p.saturation_factor; }

    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neutral() -> ColorKernelParams {
        ColorAdjustments::default().to_kernel_params(0.0, 0.0, 0.0, 5500.0, 0.0, 1.0)
    }

    #[test]
    fn neutral_params_are_identity() {
        let p = neutral();
        for probe in [[0.0f32, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 0.25, 0.75]] {
            let out = apply_color(probe, &p, None);
            for c in 0..3 { assert!((out[c] - probe[c]).abs() < 1e-6, "channel {c}: {} != {}", out[c], probe[c]); }
        }
    }

    #[test]
    fn one_stop_exposure_doubles_linear_value() {
        let p = ColorAdjustments::default().to_kernel_params(1.0, 0.0, 0.0, 5500.0, 0.0, 1.0);
        let out = apply_color([0.25, 0.25, 0.25], &p, None);
        assert!((out[0] - 0.5).abs() < 1e-6, "was {}", out[0]);
    }

    #[test]
    fn contrast_pivots_around_mid_grey() {
        let p = ColorAdjustments::default().to_kernel_params(0.0, 0.5, 0.0, 5500.0, 0.0, 1.0);
        let mid = apply_color([0.5, 0.5, 0.5], &p, None);
        assert!((mid[0] - 0.5).abs() < 1e-6, "mid grey must not move, was {}", mid[0]);
        assert!(apply_color([0.75, 0.75, 0.75], &p, None)[0] > 0.75);
        assert!(apply_color([0.25, 0.25, 0.25], &p, None)[0] < 0.25);
    }

    #[test]
    fn full_negative_saturation_yields_grey() {
        let p = ColorAdjustments::default().to_kernel_params(0.0, 0.0, -1.0, 5500.0, 0.0, 1.0);
        let out = apply_color([0.8, 0.2, 0.4], &p, None);
        assert!((out[0] - out[1]).abs() < 1e-6 && (out[1] - out[2]).abs() < 1e-6, "expected grey, got {out:?}");
    }

    #[test]
    fn saturation_preserves_luma() {
        let p = ColorAdjustments::default().to_kernel_params(0.0, 0.0, 0.6, 5500.0, 0.0, 1.0);
        let input = [0.8f32, 0.2, 0.4];
        let luma = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let out = apply_color(input, &p, None);
        assert!((luma(out) - luma(input)).abs() < 1e-5);
    }

    #[test]
    fn zero_lut_strength_ignores_the_lut() {
        // A LUT that maps everything to black; at strength 0 it must do nothing.
        let black = Lut3D { size: 2, domain_min: 0.0, domain_max: 1.0, data: vec![0.0; 8 * 3], checksum: 0 };
        let p = ColorAdjustments::default().to_kernel_params(0.0, 0.0, 0.0, 5500.0, 0.0, 0.0);
        let out = apply_color([0.7, 0.7, 0.7], &p, Some(&black));
        assert!((out[0] - 0.7).abs() < 1e-6, "was {}", out[0]);
    }

    #[test]
    fn half_lut_strength_blends_evenly() {
        let black = Lut3D { size: 2, domain_min: 0.0, domain_max: 1.0, data: vec![0.0; 8 * 3], checksum: 0 };
        let p = ColorAdjustments::default().to_kernel_params(0.0, 0.0, 0.0, 5500.0, 0.0, 0.5);
        let out = apply_color([0.8, 0.8, 0.8], &p, Some(&black));
        assert!((out[0] - 0.4).abs() < 1e-6, "was {}", out[0]);
    }

    #[test]
    fn defaults_are_inactive() {
        let c = ColorAdjustments::default();
        assert!(!c.enabled);
        assert!(!c.is_active());
    }

    #[test]
    fn set_param_clamps_every_scalar_to_its_range() {
        let mut c = ColorAdjustments::default();
        for (name, too_low, too_high, min, max) in [
            ("exposure",      -1e6, 1e6, -5.0,       5.0),
            ("contrast",      -1e6, 1e6, -1.0,       1.0),
            ("saturation",    -1e6, 1e6, -1.0,       1.0),
            ("tint",          -1e6, 1e6, -1.0,       1.0),
            ("source_kelvin",  0.0, 1e6, MIN_KELVIN, MAX_KELVIN),
            ("target_kelvin",  0.0, 1e6, MIN_KELVIN, MAX_KELVIN),
            ("lut_strength",  -1e6, 1e6,  0.0,       1.0),
        ] {
            assert!(c.set_param(name, too_low), "{name} should be a known parameter");
            assert_eq!(read_param(&c, name), min, "{name} not clamped at the low end");
            assert!(c.set_param(name, too_high));
            assert_eq!(read_param(&c, name), max, "{name} not clamped at the high end");
        }
    }

    #[test]
    fn set_param_stores_in_range_values_untouched() {
        let mut c = ColorAdjustments::default();
        assert!(c.set_param("exposure", 1.25));
        assert!(c.set_param("target_kelvin", 3200.0));
        assert_eq!(c.exposure, 1.25);
        assert_eq!(c.target_kelvin, 3200.0);
    }

    #[test]
    fn set_param_rejects_unknown_names_and_non_finite_values() {
        let mut c = ColorAdjustments::default();
        c.exposure = 1.0;
        assert!(!c.set_param("gamma", 2.2));
        assert!(!c.set_param("exposure", f64::NAN));
        assert!(!c.set_param("exposure", f64::INFINITY));
        assert_eq!(c.exposure, 1.0, "a rejected value must leave the field alone");
    }

    fn read_param(c: &ColorAdjustments, name: &str) -> f64 {
        match name {
            "exposure"      => c.exposure,
            "contrast"      => c.contrast,
            "saturation"    => c.saturation,
            "tint"          => c.tint,
            "source_kelvin" => c.source_kelvin,
            "target_kelvin" => c.target_kelvin,
            "lut_strength"  => c.lut_strength,
            _ => unreachable!(),
        }
    }

    #[test]
    fn checksum_changes_when_any_parameter_changes() {
        let base = ColorAdjustments::default();
        let mut modified = base.clone();
        modified.exposure = 0.5;
        assert_ne!(base.get_checksum(), modified.get_checksum());

        let mut with_lut = base.clone();
        with_lut.lut = Some(std::sync::Arc::new(Lut3D { size: 2, domain_min: 0.0, domain_max: 1.0, data: vec![0.5; 24], checksum: 12345 }));
        assert_ne!(base.get_checksum(), with_lut.get_checksum());
    }

    /// The keyframe path never goes through `set_param`, so a hand-edited project or the
    /// unbounded numeric field can hand `to_kernel_params` anything at all. Unclamped,
    /// `exp2` reaches `+inf` around 1024 stops and renders a white frame with no diagnostic.
    #[test]
    fn keyframed_exposure_cannot_blow_up_the_gain() {
        let c = ColorAdjustments::default();

        let huge = c.to_kernel_params(100_000.0, 0.0, 0.0, DEFAULT_KELVIN, 0.0, 1.0);
        assert!(huge.exposure_gain.is_finite(), "exposure gain was {}", huge.exposure_gain);
        assert_eq!(huge.exposure_gain, (MAX_EXPOSURE as f32).exp2());

        let tiny = c.to_kernel_params(-100_000.0, 0.0, 0.0, DEFAULT_KELVIN, 0.0, 1.0);
        assert!(tiny.exposure_gain.is_finite());
        assert_eq!(tiny.exposure_gain, (MIN_EXPOSURE as f32).exp2());

        // In-range values are untouched: one stop up is exactly a doubling.
        let one_stop = c.to_kernel_params(1.0, 0.0, 0.0, DEFAULT_KELVIN, 0.0, 1.0);
        assert_eq!(one_stop.exposure_gain, 2.0);
    }

    /// `set_param` and `to_kernel_params` must agree on the bounds, or the static and
    /// keyframe paths clamp differently.
    #[test]
    fn static_and_keyframe_exposure_paths_share_bounds() {
        let mut c = ColorAdjustments::default();
        c.set_param("exposure", 100.0);
        assert_eq!(c.exposure, MAX_EXPOSURE);
        let via_keyframe = c.to_kernel_params(100.0, 0.0, 0.0, DEFAULT_KELVIN, 0.0, 1.0);
        let via_static = c.to_kernel_params(c.exposure, 0.0, 0.0, DEFAULT_KELVIN, 0.0, 1.0);
        assert_eq!(via_keyframe.exposure_gain, via_static.exposure_gain);
    }
}
