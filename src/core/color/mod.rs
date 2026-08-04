// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Adrian <adrian.eddy at gmail>

pub mod lut;
pub mod source_metadata;
pub mod white_balance;

pub use lut::{ Lut3D, LutError, parse_cube, sample_trilinear };
pub use source_metadata::SourceExposure;
pub use white_balance::kelvin_to_rgb_gains;

use std::sync::Arc;

pub const DEFAULT_KELVIN: f64 = 5500.0;

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
    pub fn is_active(&self) -> bool {
        self.enabled && (
            self.lut.is_some()
            || self.exposure != 0.0 || self.contrast != 0.0 || self.saturation != 0.0
            || self.tint != 0.0
            || (self.target_kelvin - self.source_kelvin).abs() > f64::EPSILON
        )
    }

    /// Scalars come in already resolved for this frame (keyframes applied by the caller).
    pub fn to_kernel_params(&self, exposure: f64, contrast: f64, saturation: f64, target_kelvin: f64, tint: f64, lut_strength: f64) -> ColorKernelParams {
        ColorKernelParams {
            exposure_gain:     exposure.exp2() as f32,
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
    fn checksum_changes_when_any_parameter_changes() {
        let base = ColorAdjustments::default();
        let mut modified = base.clone();
        modified.exposure = 0.5;
        assert_ne!(base.get_checksum(), modified.get_checksum());

        let mut with_lut = base.clone();
        with_lut.lut = Some(std::sync::Arc::new(Lut3D { size: 2, domain_min: 0.0, domain_max: 1.0, data: vec![0.5; 24], checksum: 12345 }));
        assert_ne!(base.get_checksum(), with_lut.get_checksum());
    }
}
