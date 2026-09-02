// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Adrian <adrian.eddy at gmail>

pub mod lut;
pub mod output_range;
pub mod picker;
pub mod output_space;
pub mod source_metadata;
pub mod white_balance;

pub use lut::{ Lut3D, LutError, parse_cube, sample_trilinear };
pub use output_range::OutputRange;
pub use picker::{ solve_white_balance, average_patch, PickError };
pub use output_space::OutputColorSpace;
pub use source_metadata::SourceExposure;
pub use white_balance::kelvin_to_rgb_gains;

use std::sync::Arc;

pub const DEFAULT_KELVIN: f64 = 5500.0;
/// The range over which the white balance model is usable.
///
/// The top is where `white_balance::cct_to_xy` stops being defined. The bottom is *not* that
/// function's lower limit of 1667 K, but the point below which sRGB cannot represent the
/// Planckian locus at all: the white point's blue component goes negative around 1902 K, and
/// `kelvin_to_rgb_gains` then clamps it to a floor. The result is a discontinuity you can see -
/// the blue gain is 1e-6 at 1900 K and 13.8 at 1905 K - so every temperature below the crossing
/// rendered a broken, blue-less frame. 2000 K is the nearest round number safely above it.
/// `white_point_stays_in_gamut_across_the_slider_range` is the guard.
pub const MIN_KELVIN: f64 = 2000.0;
pub const MAX_KELVIN: f64 = 25000.0;

/// Exposure bounds, in stops. Shared by `set_param` and `to_kernel_params` so the static
/// and keyframe paths cannot drift apart.
pub const MIN_EXPOSURE: f64 = -5.0;
pub const MAX_EXPOSURE: f64 =  5.0;

/// The transfer function the colour stage assumes its input carries, used to turn a slider
/// reading in stops into a gain on the encoded signal.
///
/// The stage works on display-referred RGB, so a channel value is roughly `linear^(1/2.2)`.
/// Multiplying an encoded value by `g` therefore multiplies *light* by `g^2.2`, which is why
/// applying `2^EV` directly made one slider stop worth 2.2 photographic stops: at +1 EV a
/// quarter of a typical frame clipped to flat white, which is what read as a sudden loss of
/// contrast. Dividing the stops by the gamma first makes a stop a stop.
///
/// It is an assumption, and it is only right for display-referred input. On ungraded log
/// footage a multiply acts as a gamma change rather than an exposure change whatever constant
/// is used; fixing that means reading the source's transfer function, which the recording
/// metadata does carry ("Gamma equation") but nothing here reads yet.
pub const DISPLAY_GAMMA: f64 = 2.2;

#[derive(Debug, Clone)]
pub struct ColorAdjustments {
    pub enabled: bool,
    pub bake_in_render: bool,
    pub exposure: f64,               // stops, -5..5, 0 neutral
    pub contrast: f64,               // -1..1, 0 neutral
    pub saturation: f64,             // -1..1, 0 neutral
    pub shadows: f64,                // -1..1, 0 neutral
    pub highlights: f64,             // -1..1, 0 neutral
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
            shadows: 0.0,
            highlights: 0.0,
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

/// The keyframable scalars, resolved for a single frame by the caller.
///
/// A struct rather than a positional argument list because there are eight of them and every
/// one is an `f64`: transposing `contrast` and `saturation` at a call site would compile
/// cleanly and silently mis-grade every pixel.
///
/// `Default` is the *neutral* grade, not all-zeroes - `target_kelvin` at
/// `DEFAULT_KELVIN` and `lut_strength` at 1.0 are what "no adjustment" means for those two.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedScalars {
    pub exposure: f64,
    pub contrast: f64,
    pub saturation: f64,
    pub shadows: f64,
    pub highlights: f64,
    pub target_kelvin: f64,
    pub tint: f64,
    pub lut_strength: f64,
}

impl Default for ResolvedScalars {
    fn default() -> Self {
        Self {
            exposure: 0.0, contrast: 0.0, saturation: 0.0,
            shadows: 0.0, highlights: 0.0,
            target_kelvin: DEFAULT_KELVIN, tint: 0.0, lut_strength: 1.0,
        }
    }
}

/// How far a full-scale shadows or highlights slider bends the tone curve.
///
/// **Must stay below 1/3.** The curve is monotonic for any strength up to exactly 1/3 and folds
/// back on itself above it - at 0.35 a brighter input already comes out darker than a dimmer
/// one by 6e-4, which is a visible solarisation artefact on a large smooth gradient like a sky.
/// `the_tone_curve_never_inverts` is the guard; do not raise this without re-reading it.
pub const TONE_REGION_STRENGTH: f64 = 0.30;

/// Ready-to-multiply values handed to every kernel. No colour science here —
/// it has all been done in Rust by the time this struct is built.
#[derive(Debug, Clone, Copy, Default)]
pub struct ColorKernelParams {
    pub exposure_gain: f32,
    pub contrast_factor: f32,
    pub saturation_factor: f32,
    /// Bezier control weights, already multiplied by 3 * `TONE_REGION_STRENGTH`. The kernels
    /// form `1 + shadows*(1-l)^2 + highlights*l*(1-l)` and multiply by it.
    pub shadows: f32,
    pub highlights: f32,
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
            "shadows"       => self.shadows       = val.clamp(-1.0, 1.0),
            "highlights"    => self.highlights    = val.clamp(-1.0, 1.0),
            "tint"          => self.tint          = val.clamp(-1.0, 1.0),
            "source_kelvin" => self.source_kelvin = val.clamp(MIN_KELVIN, MAX_KELVIN),
            "target_kelvin" => self.target_kelvin = val.clamp(MIN_KELVIN, MAX_KELVIN),
            "lut_strength"  => self.lut_strength  = val.clamp(0.0, 1.0),
            _ => return false,
        }
        true
    }

    /// Point the white balance at a new clip's own neutral, keeping the correction the user
    /// chose rather than the number it happened to produce.
    ///
    /// `target_kelvin` is stored absolute, but what the user actually chose is the *offset*
    /// from this clip's neutral - "400 K warmer than the camera saw it". Carrying the absolute
    /// number to a clip the camera metered differently applies a different correction, which
    /// is what made a batch of mixed-white-balance clips come out inconsistent: every queued
    /// job inherited the on-screen clip's `source_kelvin` and graded against the wrong anchor.
    ///
    /// Shifting the target by the same delta as the source keeps the offset and moves the
    /// result, which is what travels correctly across files.
    ///
    /// With a zero offset - the common case, colour switched off or temperature untouched -
    /// the target lands exactly on the new source, which is what this did before it preserved
    /// anything. An offset that was clamped at either Kelvin bound does not survive a round
    /// trip through the bound; that is a deliberate consequence of storing the value absolute.
    pub fn reanchor_source_kelvin(&mut self, new_source: f64, detected: bool) {
        if !new_source.is_finite() { return; }
        let previous = self.source_kelvin;
        self.set_param("source_kelvin", new_source);
        self.source_kelvin_detected = detected;

        // Read back rather than using `new_source`: `set_param` clamps, and the target has to
        // move by the distance the source actually moved.
        let delta = self.source_kelvin - previous;
        if delta != 0.0 {
            self.set_param("target_kelvin", self.target_kelvin + delta);
        }
    }

    pub fn is_active(&self) -> bool {
        self.enabled && (
            self.lut.is_some()
            || self.exposure != 0.0 || self.contrast != 0.0 || self.saturation != 0.0
            || self.shadows != 0.0 || self.highlights != 0.0
            || self.tint != 0.0
            || (self.target_kelvin - self.source_kelvin).abs() > f64::EPSILON
        )
    }

    /// Scalars come in already resolved for this frame (keyframes applied by the caller).
    ///
    /// Every scalar is clamped here, including exposure. These values arrive from the
    /// *keyframe* path, which never passes through `set_param`, so the import clamps and the
    /// slider bounds are both bypassed - a hand-edited `keyframes` section, or a value typed
    /// into the unbounded `NumberField`, lands here raw. Unclamped, the exposure gain reaches
    /// `+inf` around 2250 stops and renders a white frame with no diagnostic.
    pub fn to_kernel_params(&self, s: ResolvedScalars) -> ColorKernelParams {
        ColorKernelParams {
            exposure_gain:     (s.exposure.clamp(MIN_EXPOSURE, MAX_EXPOSURE) / DISPLAY_GAMMA).exp2() as f32,
            contrast_factor:   (1.0 + s.contrast.clamp(-1.0, 1.0)) as f32,
            saturation_factor: (1.0 + s.saturation.clamp(-1.0, 1.0)) as f32,
            shadows:           (s.shadows.clamp(-1.0, 1.0)    * 3.0 * TONE_REGION_STRENGTH) as f32,
            highlights:        (s.highlights.clamp(-1.0, 1.0) * 3.0 * TONE_REGION_STRENGTH) as f32,
            lut_strength:      s.lut_strength.clamp(0.0, 1.0) as f32,
            wb_gains:          kelvin_to_rgb_gains(self.source_kelvin, s.target_kelvin, s.tint),
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
        for v in [self.exposure, self.contrast, self.saturation, self.shadows, self.highlights, self.source_kelvin, self.target_kelvin, self.tint, self.lut_strength] {
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

    // 5. Shadows / highlights: a cubic tone curve on luma, applied as a single gain.
    //
    // The curve is `l + shadows*3l(1-l)^2 + highlights*3l^2(1-l)` - the two middle Bernstein
    // basis functions, i.e. a cubic Bezier whose endpoints are pinned at 0 and 1. Both terms
    // carry a factor of `l`, so dividing through to get the gain removes it analytically and
    // leaves the expression below: no division, no singularity at black, and exactly 1.0 when
    // both sliders are centred.
    //
    // Three properties follow from that shape, and all three are what "natural" means here:
    //
    // - *Black stays black and white stays white.* The endpoints are pinned, so no setting can
    //   lift the black point off zero or pull white off one. An earlier version added a
    //   luma-weighted offset instead; it raised black to 0.5 at full shadows, which is exactly
    //   what a washed-out, milky picture is.
    // - *Saturation is untouched.* One scalar multiplies all three channels, so their ratios -
    //   and therefore the saturation - come out bit-identical. The offset version dragged the
    //   channels together: a dark blue at saturation 0.72 came out at 0.22, which is what
    //   "dull" was.
    // - *Each slider owns its end.* The deltas peak at l = 1/3 and l = 2/3 respectively, the
    //   textbook placement, and fall to zero at both extremes.
    //
    // Luma is clamped to 0..1 first because the input is not guaranteed to be in range - a LUT
    // or an exposure push can carry a channel past either end - and an unclamped basis
    // function would change sign there and invert the correction.
    //
    // Computed on the post-contrast pixel on purpose: the curve has to describe the image the
    // user is looking at, or the sliders stop feeling predictable.
    let tone_luma = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]).clamp(0.0, 1.0);
    let inv_luma = 1.0 - tone_luma;
    let tone = 1.0 + p.shadows * inv_luma * inv_luma + p.highlights * tone_luma * inv_luma;
    for i in 0..3 { c[i] *= tone; }

    // 6. Saturation around luma
    let luma = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    for i in 0..3 { c[i] = luma + (c[i] - luma) * p.saturation_factor; }

    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neutral() -> ColorKernelParams {
        ColorAdjustments::default().to_kernel_params(ResolvedScalars::default())
    }

    #[test]
    fn neutral_params_are_identity() {
        let p = neutral();
        for probe in [[0.0f32, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 0.25, 0.75]] {
            let out = apply_color(probe, &p, None);
            for c in 0..3 { assert!((out[c] - probe[c]).abs() < 1e-6, "channel {c}: {} != {}", out[c], probe[c]); }
        }
    }

    /// One slider stop is one *photographic* stop - linear light doubles. On a display-referred
    /// signal that is a gain of 2^(1/2.2) on the encoded value, not 2.
    #[test]
    fn one_stop_exposure_doubles_linear_light() {
        let p = ColorAdjustments::default().to_kernel_params(ResolvedScalars { exposure: 1.0, ..Default::default() });
        let encoded = 0.25f32;
        let out = apply_color([encoded, encoded, encoded], &p, None)[0];

        let g = DISPLAY_GAMMA as f32;
        let ratio = out.powf(g) / encoded.powf(g);
        assert!((ratio - 2.0).abs() < 1e-4, "linear light went up {ratio}x, expected 2x");
    }

    /// The symptom that prompted the calibration: at a raw `2^EV`, +1 EV multiplied the encoded
    /// signal by 2, so everything above 0.5 - about a quarter of a real frame - slammed into
    /// pure white and lost all detail.
    #[test]
    fn one_stop_does_not_clip_the_upper_midtones() {
        let p = ColorAdjustments::default().to_kernel_params(ResolvedScalars { exposure: 1.0, ..Default::default() });
        for probe in [0.6f32, 0.7, 0.72] {
            let out = apply_color([probe, probe, probe], &p, None)[0];
            assert!(out < 1.0, "{probe} clipped to {out}");
        }
    }

    #[test]
    fn contrast_pivots_around_mid_grey() {
        let p = ColorAdjustments::default().to_kernel_params(ResolvedScalars { contrast: 0.5, ..Default::default() });
        let mid = apply_color([0.5, 0.5, 0.5], &p, None);
        assert!((mid[0] - 0.5).abs() < 1e-6, "mid grey must not move, was {}", mid[0]);
        assert!(apply_color([0.75, 0.75, 0.75], &p, None)[0] > 0.75);
        assert!(apply_color([0.25, 0.25, 0.25], &p, None)[0] < 0.25);
    }

    #[test]
    fn full_negative_saturation_yields_grey() {
        let p = ColorAdjustments::default().to_kernel_params(ResolvedScalars { saturation: -1.0, ..Default::default() });
        let out = apply_color([0.8, 0.2, 0.4], &p, None);
        assert!((out[0] - out[1]).abs() < 1e-6 && (out[1] - out[2]).abs() < 1e-6, "expected grey, got {out:?}");
    }

    #[test]
    fn saturation_preserves_luma() {
        let p = ColorAdjustments::default().to_kernel_params(ResolvedScalars { saturation: 0.6, ..Default::default() });
        let input = [0.8f32, 0.2, 0.4];
        let luma = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let out = apply_color(input, &p, None);
        assert!((luma(out) - luma(input)).abs() < 1e-5);
    }

    #[test]
    fn zero_lut_strength_ignores_the_lut() {
        // A LUT that maps everything to black; at strength 0 it must do nothing.
        let black = Lut3D { size: 2, domain_min: 0.0, domain_max: 1.0, data: vec![0.0; 8 * 3], checksum: 0 };
        let p = ColorAdjustments::default().to_kernel_params(ResolvedScalars { lut_strength: 0.0, ..Default::default() });
        let out = apply_color([0.7, 0.7, 0.7], &p, Some(&black));
        assert!((out[0] - 0.7).abs() < 1e-6, "was {}", out[0]);
    }

    #[test]
    fn half_lut_strength_blends_evenly() {
        let black = Lut3D { size: 2, domain_min: 0.0, domain_max: 1.0, data: vec![0.0; 8 * 3], checksum: 0 };
        let p = ColorAdjustments::default().to_kernel_params(ResolvedScalars { lut_strength: 0.5, ..Default::default() });
        let out = apply_color([0.8, 0.8, 0.8], &p, Some(&black));
        assert!((out[0] - 0.4).abs() < 1e-6, "was {}", out[0]);
    }

    #[test]
    fn reanchoring_keeps_the_users_offset_not_the_number() {
        let mut c = ColorAdjustments::default();
        c.set_param("source_kelvin", 6000.0);
        c.set_param("target_kelvin", 6400.0);   // the user asked for +400 K

        c.reanchor_source_kelvin(3200.0, true); // a tungsten clip from the same shoot

        assert_eq!(c.source_kelvin, 3200.0);
        assert_eq!(c.target_kelvin, 3600.0, "the +400 K offset should have travelled");
        assert!(c.source_kelvin_detected);
    }

    /// The case that must not regress: with no temperature offset the target follows the
    /// source exactly, which is what happened before re-anchoring preserved anything.
    #[test]
    fn reanchoring_with_no_offset_lands_on_the_new_source() {
        let mut c = ColorAdjustments::default();
        c.set_param("source_kelvin", 6000.0);
        c.set_param("target_kelvin", 6000.0);

        c.reanchor_source_kelvin(4300.0, true);

        assert_eq!(c.source_kelvin, 4300.0);
        assert_eq!(c.target_kelvin, 4300.0);
    }

    #[test]
    fn reanchoring_clamps_and_survives_a_missing_reading() {
        let mut c = ColorAdjustments::default();
        c.set_param("source_kelvin", 6000.0);
        c.set_param("target_kelvin", 6000.0);

        // A camera that reported nothing: the caller passes the default, still anchored.
        c.reanchor_source_kelvin(DEFAULT_KELVIN, false);
        assert_eq!(c.source_kelvin, DEFAULT_KELVIN);
        assert!(!c.source_kelvin_detected);

        // Way out of range, and a non-finite reading must change nothing at all.
        c.reanchor_source_kelvin(1e9, true);
        assert_eq!(c.source_kelvin, MAX_KELVIN);
        let before = (c.source_kelvin, c.target_kelvin);
        c.reanchor_source_kelvin(f64::NAN, true);
        assert_eq!((c.source_kelvin, c.target_kelvin), before, "a NaN reading must be ignored");
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
            ("shadows",       -1e6, 1e6, -1.0,       1.0),
            ("highlights",    -1e6, 1e6, -1.0,       1.0),
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

    /// Each slider owns one end of the tone range. Cubic weighting is what makes that true:
    /// at the far end the weight is ~0, so the two controls do not fight each other.
    #[test]
    fn shadows_lift_the_toe_and_leave_the_shoulder_alone() {
        let p = ColorAdjustments::default().to_kernel_params(ResolvedScalars { shadows: 1.0, ..Default::default() });
        let toe      = apply_color([0.05, 0.05, 0.05], &p, None);
        let shoulder = apply_color([0.95, 0.95, 0.95], &p, None);
        assert!(toe[0] > 0.05 * 1.5, "toe should lift hard, was {}", toe[0]);
        assert!((shoulder[0] - 0.95).abs() < 0.01, "shoulder should barely move, was {}", shoulder[0]);
        assert_eq!(apply_color([1.0, 1.0, 1.0], &p, None)[0], 1.0, "white is pinned");
    }

    #[test]
    fn highlights_pull_the_shoulder_and_leave_the_toe_alone() {
        let p = ColorAdjustments::default().to_kernel_params(ResolvedScalars { highlights: -1.0, ..Default::default() });
        // Probed at 2/3, where the highlight term peaks - not at 0.95, because white is pinned
        // and the correction necessarily falls back to nothing as it approaches it.
        let toe   = apply_color([0.05, 0.05, 0.05], &p, None);
        let upper = apply_color([0.667, 0.667, 0.667], &p, None);
        assert!(upper[0] < 0.667 * 0.85, "upper mids should come down hard, was {}", upper[0]);
        assert!((toe[0] - 0.05).abs() < 0.01, "toe should barely move, was {}", toe[0]);
        assert_eq!(apply_color([1.0, 1.0, 1.0], &p, None)[0], 1.0, "white is pinned");
    }

    /// The numbers that decide whether these sliders feel controllable. At full scale the
    /// shadow term reaches `3 * TONE_REGION_STRENGTH` of extra gain as luma approaches zero,
    /// and a quarter of that at mid grey, where `(1-l)^2` is 0.25.
    #[test]
    fn a_full_slider_reaches_its_declared_strength() {
        let p = ColorAdjustments::default().to_kernel_params(ResolvedScalars { shadows: 1.0, ..Default::default() });
        let gain = |v: f32| apply_color([v, v, v], &p, None)[0] / v;
        let k = (3.0 * TONE_REGION_STRENGTH) as f32;
        assert!((gain(0.001) - (1.0 + k)).abs() < 1e-2, "near black: {}", gain(0.001));
        assert!((gain(0.5)   - (1.0 + k * 0.25)).abs() < 1e-6, "mid grey: {}", gain(0.5));
        assert!((gain(1.0)   - 1.0).abs() < 1e-6, "white: {}", gain(1.0));
    }

    /// Where each control does its most work. The textbook placement, and the thing that makes
    /// them read as "shadows" and "highlights" rather than two more exposure sliders.
    #[test]
    fn the_two_deltas_peak_at_one_third_and_two_thirds() {
        for (name, scalars) in [
            ("shadows",    ResolvedScalars { shadows: 1.0, ..Default::default() }),
            ("highlights", ResolvedScalars { highlights: 1.0, ..Default::default() }),
        ] {
            let p = ColorAdjustments::default().to_kernel_params(scalars);
            let mut best = (0.0f32, 0.0f32);
            for i in 1..1000 {
                let v = i as f32 / 1000.0;
                let d = apply_color([v, v, v], &p, None)[0] - v;
                if d > best.1 { best = (v, d); }
            }
            let expected = if name == "shadows" { 1.0 / 3.0 } else { 2.0 / 3.0 };
            assert!((best.0 - expected).abs() < 0.02, "{name} peaked at {} not {expected}", best.0);
        }
    }

    /// Half of why this is a gain and not an offset. Adding a constant to R, G and B drags the
    /// channels together and washes the picture out; multiplying leaves their ratios, and so
    /// the saturation, exactly where they were.
    #[test]
    fn tone_regions_preserve_saturation_exactly() {
        let sat = |c: [f32; 3]| {
            let mx = c[0].max(c[1]).max(c[2]);
            let mn = c[0].min(c[1]).min(c[2]);
            if mx == 0.0 { 0.0 } else { (mx - mn) / mx }
        };
        for (shadows, highlights) in [(1.0, 0.0), (0.0, -1.0), (-0.6, 0.8)] {
            let p = ColorAdjustments::default().to_kernel_params(ResolvedScalars { shadows, highlights, ..Default::default() });
            for probe in [[0.05f32, 0.07, 0.18], [0.22, 0.15, 0.12], [0.18, 0.30, 0.14], [0.9, 0.7, 0.2]] {
                let out = apply_color(probe, &p, None);
                assert!((sat(out) - sat(probe)).abs() < 1e-6,
                    "saturation moved: {:?} -> {:?} at ({shadows}, {highlights})", sat(probe), sat(out));
            }
        }
    }

    /// The other half: an offset lifts the black point off zero, which is what reads as milky.
    /// A gain cannot, at any setting.
    #[test]
    fn the_endpoints_are_pinned_however_hard_the_sliders_are_pushed() {
        for shadows in [0.25, 0.5, 1.0] {
            for highlights in [-1.0, 0.0, 1.0] {
                let p = ColorAdjustments::default().to_kernel_params(ResolvedScalars { shadows, highlights, ..Default::default() });
                assert_eq!(apply_color([0.0, 0.0, 0.0], &p, None), [0.0, 0.0, 0.0],
                    "black lifted at ({shadows}, {highlights})");
                assert_eq!(apply_color([1.0, 1.0, 1.0], &p, None), [1.0, 1.0, 1.0],
                    "white moved at ({shadows}, {highlights})");
            }
        }
    }

    /// A tone curve that folds back on itself makes a brighter input come out darker than a
    /// dimmer one - solarisation, most visible on a large smooth gradient like a sky. The
    /// curve is monotonic for any `TONE_REGION_STRENGTH` up to exactly 1/3; this is the guard
    /// on that bound, and it is why the constant is 0.30 rather than something rounder.
    #[test]
    fn the_tone_curve_never_inverts() {
        for (shadows, highlights) in [(1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)] {
            let p = ColorAdjustments::default().to_kernel_params(ResolvedScalars { shadows, highlights, ..Default::default() });
            let mut prev = f32::NEG_INFINITY;
            for i in 0..=1000 {
                let v = i as f32 / 1000.0;
                let out = apply_color([v, v, v], &p, None)[0];
                assert!(out >= prev - 1e-7, "inverted at {v} with ({shadows}, {highlights}): {out} < {prev}");
                prev = out;
            }
        }
    }

    /// Equal and opposite pushes cancel exactly at mid grey - the weights are symmetric
    /// about 0.5, so this is a cheap check that neither curve has drifted.
    #[test]
    fn opposite_pushes_cancel_at_mid_grey() {
        let p = ColorAdjustments::default().to_kernel_params(ResolvedScalars { shadows: 1.0, highlights: -1.0, ..Default::default() });
        let mid = apply_color([0.5, 0.5, 0.5], &p, None);
        assert!((mid[0] - 0.5).abs() < 1e-6, "was {}", mid[0]);
    }

    #[test]
    fn shadows_or_highlights_alone_make_the_grade_active() {
        for name in ["shadows", "highlights"] {
            let mut c = ColorAdjustments::default();
            c.enabled = true;
            assert!(!c.is_active(), "nothing set yet");
            assert!(c.set_param(name, 0.5));
            assert!(c.is_active(), "{name} alone should activate the grade");
        }
    }

    fn read_param(c: &ColorAdjustments, name: &str) -> f64 {
        match name {
            "exposure"      => c.exposure,
            "contrast"      => c.contrast,
            "saturation"    => c.saturation,
            "shadows"       => c.shadows,
            "highlights"    => c.highlights,
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

        for name in ["shadows", "highlights"] {
            let mut m = base.clone();
            assert!(m.set_param(name, 0.5));
            assert_ne!(base.get_checksum(), m.get_checksum(), "{name} must invalidate the preview");
        }

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

        let huge = c.to_kernel_params(ResolvedScalars { exposure: 100_000.0, target_kelvin: DEFAULT_KELVIN, ..Default::default() });
        assert!(huge.exposure_gain.is_finite(), "exposure gain was {}", huge.exposure_gain);
        assert_eq!(huge.exposure_gain, ((MAX_EXPOSURE / DISPLAY_GAMMA) as f32).exp2());

        let tiny = c.to_kernel_params(ResolvedScalars { exposure: -100_000.0, target_kelvin: DEFAULT_KELVIN, ..Default::default() });
        assert!(tiny.exposure_gain.is_finite());
        assert_eq!(tiny.exposure_gain, ((MIN_EXPOSURE / DISPLAY_GAMMA) as f32).exp2());

        // In-range values are untouched: one stop up is a doubling of light, which on the
        // encoded signal is 2^(1/2.2).
        let one_stop = c.to_kernel_params(ResolvedScalars { exposure: 1.0, target_kelvin: DEFAULT_KELVIN, ..Default::default() });
        assert_eq!(one_stop.exposure_gain, ((1.0 / DISPLAY_GAMMA) as f32).exp2());
    }

    /// `set_param` and `to_kernel_params` must agree on the bounds, or the static and
    /// keyframe paths clamp differently.
    #[test]
    fn static_and_keyframe_exposure_paths_share_bounds() {
        let mut c = ColorAdjustments::default();
        c.set_param("exposure", 100.0);
        assert_eq!(c.exposure, MAX_EXPOSURE);
        let via_keyframe = c.to_kernel_params(ResolvedScalars { exposure: 100.0, target_kelvin: DEFAULT_KELVIN, ..Default::default() });
        let via_static = c.to_kernel_params(ResolvedScalars { exposure: c.exposure, target_kelvin: DEFAULT_KELVIN, ..Default::default() });
        assert_eq!(via_keyframe.exposure_gain, via_static.exposure_gain);
    }
}
