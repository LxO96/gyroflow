// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Adrian <adrian.eddy at gmail>

//! Solving temperature and tint from a patch the user says should be grey.
//!
//! # Why this can work from a screenshot
//!
//! The eyedropper samples what is on screen, which is the fully graded pixel, not the source.
//! That sounds like it should be useless - the LUT alone is not invertible - but the stage order
//! in [`apply_color`](super::apply_color) rescues it:
//!
//! ```text
//!   LUT -> exposure -> WHITE BALANCE -> contrast -> tone -> saturation
//! ```
//!
//! Everything *below* white balance preserves neutrality in both directions. Contrast is the same
//! affine map on each channel, the tone curve is one scalar gain on all three, and saturation
//! collapses toward luma. So a pixel is neutral on screen exactly when it is neutral immediately
//! after white balance - which means the picker never has to invert the LUT or the exposure gain.
//! It only has to undo three stages, all of which invert exactly, and then solve for the gains
//! that would have made that value grey.
//!
//! # Why temperature and tint separate
//!
//! [`kelvin_to_rgb_gains`](super::kelvin_to_rgb_gains) builds tint as a trim on green alone, so
//! the *red-to-blue* ratio of the gains is a function of temperature by itself. That makes the
//! solve two independent 1-D problems instead of a 2-D search: bisect temperature against the
//! red/blue ratio (monotone across the whole Planckian range), then read tint off the leftover
//! green in closed form.

use super::{ ColorKernelParams, MIN_KELVIN, MAX_KELVIN, white_balance::kelvin_to_rgb_gains };

/// Rec.709 luma weights - the same ones `apply_color` uses. They must match, or the saturation
/// and tone inverses below would be undoing a slightly different function than the one applied.
const LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];

fn luma(c: [f64; 3]) -> f64 { LUMA[0] * c[0] + LUMA[1] * c[1] + LUMA[2] * c[2] }

/// Why a patch could not be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickError {
    /// Too dark to carry a colour. The channel ratios of a near-black pixel are mostly sensor
    /// noise, and dividing by them throws the sliders to an arbitrary place.
    TooDark,
    /// One or more channels are at or near the top of the range. A clipped channel has lost the
    /// very ratio the solve depends on: it reads the same whatever the real value was.
    Clipped,
    /// The grade cannot be undone: a saturation or contrast slider is at the bottom of its range,
    /// where the stage maps every input to the same output and no inverse exists.
    GradeNotInvertible,
}

impl PickError {
    /// Shown to the user, so keep it about what to do next rather than about the maths.
    pub fn message(self) -> &'static str {
        match self {
            Self::TooDark   => "That patch is too dark to read a colour from. Pick a brighter grey.",
            Self::Clipped   => "That patch is clipped. Pick a grey that isn't blown out.",
            Self::GradeNotInvertible => "Saturation or contrast is at its minimum, so the picked colour can't be traced back.",
        }
    }
}

/// Below this the ratios are noise rather than colour.
const MIN_USABLE: f64 = 0.02;
/// At or above this a channel may be clipped, and its true value is unknowable.
const MAX_USABLE: f64 = 0.98;

/// Undo saturation, the tone curve and contrast, recovering the value white balance produced.
///
/// The inverse of stages 4-6 of [`apply_color`](super::apply_color), in reverse order. Exposed
/// (and tested) on its own because getting it wrong is silent: the solve would still return a
/// plausible-looking temperature that simply does not neutralise the patch.
pub fn undo_stages_below_white_balance(displayed: [f64; 3], p: &ColorKernelParams) -> Result<[f64; 3], PickError> {
    let mut c = displayed;

    // 6. Saturation: `out = L + (c - L) * s`, and because the weights sum to 1 this map leaves
    //    luma alone - `luma(out) == luma(c)`. So L is readable straight off the sampled pixel and
    //    the inverse is a division, with no need to know the original.
    let s = p.saturation_factor as f64;
    if s.abs() < 1e-4 { return Err(PickError::GradeNotInvertible); }
    let l = luma(c);
    for v in &mut c { *v = l + (*v - l) / s; }

    // 5. Tone: `out = c * tone(luma(c))`, one gain on all three channels, so `luma(out) =
    //    luma(c) * tone(luma(c))`. That is a single equation in the single unknown `luma(c)`;
    //    solve it and the gain follows. Bisection rather than algebra because `tone` is a cubic
    //    in the unknown and the closed form is not worth the reading.
    let y_out = luma(c);
    let gain = solve_tone_gain(y_out, p.shadows as f64, p.highlights as f64);
    if gain.abs() < 1e-6 { return Err(PickError::GradeNotInvertible); }
    for v in &mut c { *v /= gain; }

    // 4. Contrast around mid grey, the same affine map on every channel.
    let k = p.contrast_factor as f64;
    if k.abs() < 1e-4 { return Err(PickError::GradeNotInvertible); }
    for v in &mut c { *v = (*v - 0.5) / k + 0.5; }

    Ok(c)
}

/// Given the luma the tone stage *produced*, recover the gain it applied.
///
/// `apply_color` computes `tone` from the luma going in, clamped to 0..1, then multiplies. So
/// `y_out = y_in * tone(y_in)`, which is monotone in `y_in` for any settings the sliders can
/// reach (`TONE_REGION_STRENGTH` is held below 1/3 precisely to keep it so), and bisection is
/// both simple and exact enough.
fn solve_tone_gain(y_out: f64, shadows: f64, highlights: f64) -> f64 {
    let tone_at = |y: f64| { let y = y.clamp(0.0, 1.0); let inv = 1.0 - y; 1.0 + shadows * inv * inv + highlights * y * inv };

    if shadows == 0.0 && highlights == 0.0 { return 1.0; }

    // Outside 0..1 the stage clamps its luma, so the gain is constant there and there is
    // nothing to solve.
    if y_out <= 0.0 { return tone_at(0.0); }
    if y_out >= tone_at(1.0) { return tone_at(1.0); }

    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if mid * tone_at(mid) < y_out { lo = mid; } else { hi = mid; }
    }
    tone_at(0.5 * (lo + hi))
}

/// Temperature and tint that would turn `patch` grey, where `patch` is a pixel sampled from the
/// screen and `p` is the grade currently on it.
///
/// `source_kelvin` is the clip's own white point - the anchor the temperature slider is an offset
/// from - and is passed through unchanged; the picker moves the *target*, never the anchor.
pub fn solve_white_balance(patch: [f64; 3], p: &ColorKernelParams, source_kelvin: f64) -> Result<(f64, f64), PickError> {
    if patch.iter().any(|v| !v.is_finite()) { return Err(PickError::TooDark); }
    if patch.iter().cloned().fold(f64::MAX, f64::min) < MIN_USABLE { return Err(PickError::TooDark); }
    if patch.iter().cloned().fold(f64::MIN, f64::max) > MAX_USABLE { return Err(PickError::Clipped); }

    // What white balance was handed, before the stages below it dressed it up.
    let a = undo_stages_below_white_balance(patch, p)?;
    if a.iter().any(|v| *v <= 1e-6) { return Err(PickError::TooDark); }

    // We want gains `g` with `a[i] * g[i]` equal for all i, i.e. `g` proportional to `1/a`.
    // `kelvin_to_rgb_gains` is luma-normalised, so only ratios matter and proportionality is
    // exactly the right relation to solve for.
    let want_rb = a[2] / a[0]; // g_r/g_b must be a_b/a_r

    // Temperature from the red/blue ratio alone. Tint only trims green, so it cannot affect this
    // ratio - which is what lets the two be solved one after the other instead of together.
    // Raising the target warms the image: red gain up, blue gain down, so the ratio rises
    // monotonically and bisection is safe over the whole defined range.
    let rb_at = |k: f64| { let g = kelvin_to_rgb_gains(source_kelvin, k, 0.0); g[0] as f64 / g[2].max(1e-6) as f64 };

    let (mut lo, mut hi) = (MIN_KELVIN, MAX_KELVIN);
    // A patch whose cast is stronger than 1667..25000 K can express: clamp rather than fail, so
    // an extreme pick still lands at the nearest reachable white point.
    let kelvin = if want_rb <= rb_at(lo) { lo } else if want_rb >= rb_at(hi) { hi } else {
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if rb_at(mid) < want_rb { lo = mid; } else { hi = mid; }
        }
        0.5 * (lo + hi)
    };

    // Tint from the green that is left over. At this temperature and zero tint the gains are
    // `g`; green must instead be `a_r*g_r/a_g` for the three products to match, and green is
    // scaled by `1 - 0.2t`, so `t` reads straight off. (The luma normalisation divides all three
    // by the same number and cancels out of the ratio.)
    let g = kelvin_to_rgb_gains(source_kelvin, kelvin, 0.0);
    let want_g = a[0] * g[0] as f64 / a[1];
    let tint = ((1.0 - want_g / g[1] as f64) / 0.2).clamp(-1.0, 1.0);

    Ok((kelvin, tint))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{ ColorAdjustments, ResolvedScalars, apply_color, DEFAULT_KELVIN };

    fn params(f: impl FnOnce(&mut ResolvedScalars)) -> ColorKernelParams {
        let mut s = ResolvedScalars::default();
        f(&mut s);
        ColorAdjustments::default().to_kernel_params(s)
    }
    fn f32x3(c: [f64; 3]) -> [f32; 3] { [c[0] as f32, c[1] as f32, c[2] as f32] }

    /// Every stage the inverse claims to undo, undone. Run against `apply_color` itself rather
    /// than against a re-derivation of it, so the test fails if the forward stage ever changes.
    #[test]
    fn the_inverse_round_trips_against_apply_color() {
        // A cast pixel, so a bug that only happens to work on greys would show.
        let probe = [0.42f32, 0.38, 0.30];
        for (contrast, saturation, shadows, highlights) in [
            ( 0.0,  0.0,  0.0,  0.0),
            ( 0.5,  0.0,  0.0,  0.0),
            (-0.5,  0.0,  0.0,  0.0),
            ( 0.0,  0.6,  0.0,  0.0),
            ( 0.0, -0.6,  0.0,  0.0),
            ( 0.0,  0.0,  0.8, -0.8),
            ( 0.0,  0.0, -1.0,  1.0),
            ( 0.4, -0.3,  0.6,  0.5),
        ] {
            let p = params(|s| { s.contrast = contrast; s.saturation = saturation; s.shadows = shadows; s.highlights = highlights; });
            // `apply_color` with neutral white balance, so what comes out is exactly what the
            // three invertible stages did to `probe`.
            let displayed = apply_color(probe, &p, None);
            let back = undo_stages_below_white_balance([displayed[0] as f64, displayed[1] as f64, displayed[2] as f64], &p).unwrap();
            for c in 0..3 {
                assert!((back[c] - probe[c] as f64).abs() < 1e-4,
                    "contrast {contrast} sat {saturation} sh {shadows} hi {highlights}: channel {c} came back {} not {}", back[c], probe[c]);
            }
        }
    }

    /// The property that actually matters, tested end to end: solve from a patch, apply the
    /// solved grade, and the patch must come out grey. This covers the inverse and the solver
    /// together, and it is the one test that fails if `apply_color`'s stage *order* changes.
    #[test]
    fn solving_then_grading_neutralises_the_patch() {
        for cast in [
            [0.50f64, 0.45, 0.38], // warm
            [0.34, 0.40, 0.52],    // cool
            [0.40, 0.46, 0.41],    // green
            [0.46, 0.40, 0.45],    // magenta
            [0.20, 0.19, 0.17],    // dark but usable
            [0.80, 0.76, 0.70],    // bright but not clipped
        ] {
            for (contrast, saturation, shadows, highlights) in [(0.0, 0.0, 0.0, 0.0), (0.4, -0.3, 0.6, 0.5)] {
                let before = params(|s| { s.contrast = contrast; s.saturation = saturation; s.shadows = shadows; s.highlights = highlights; });
                let (kelvin, tint) = solve_white_balance(cast, &before, DEFAULT_KELVIN).unwrap();

                let after = params(|s| {
                    s.contrast = contrast; s.saturation = saturation; s.shadows = shadows; s.highlights = highlights;
                    s.target_kelvin = kelvin; s.tint = tint;
                });
                // The sampled patch is post-grade, so to re-grade it we start from the value the
                // grade was handed - which is what the inverse recovers.
                let a = undo_stages_below_white_balance(cast, &before).unwrap();
                let out = apply_color(f32x3(a), &after, None);

                let spread = out.iter().cloned().fold(f32::MIN, f32::max) - out.iter().cloned().fold(f32::MAX, f32::min);
                assert!(spread < 2e-3, "{cast:?} at contrast {contrast}: solved {kelvin:.0}K tint {tint:.3} left {out:?} (spread {spread})");
            }
        }
    }

    /// A patch that is already neutral must not move the sliders off the clip's own white point.
    #[test]
    fn a_neutral_patch_solves_to_the_source_white_point() {
        let p = params(|_| {});
        let (kelvin, tint) = solve_white_balance([0.5, 0.5, 0.5], &p, DEFAULT_KELVIN).unwrap();
        assert!((kelvin - DEFAULT_KELVIN).abs() < 20.0, "moved to {kelvin}K");
        assert!(tint.abs() < 1e-3, "tint moved to {tint}");
    }

    /// Direction check, in the terms a user would describe: a patch that looks too blue means the
    /// shot was lit warmer than the grade assumes, so the correction warms it - the target goes up.
    #[test]
    fn a_blue_patch_warms_and_an_orange_patch_cools() {
        let p = params(|_| {});
        let (blue, _)   = solve_white_balance([0.34, 0.40, 0.52], &p, DEFAULT_KELVIN).unwrap();
        let (orange, _) = solve_white_balance([0.52, 0.40, 0.34], &p, DEFAULT_KELVIN).unwrap();
        assert!(blue > DEFAULT_KELVIN,   "a blue patch should raise the target, got {blue}K");
        assert!(orange < DEFAULT_KELVIN, "an orange patch should lower the target, got {orange}K");
    }

    /// Same, for the other axis: a green patch is corrected toward magenta, which is positive tint.
    #[test]
    fn a_green_patch_pushes_tint_toward_magenta() {
        let p = params(|_| {});
        let (_, green)   = solve_white_balance([0.40, 0.46, 0.41], &p, DEFAULT_KELVIN).unwrap();
        let (_, magenta) = solve_white_balance([0.46, 0.40, 0.45], &p, DEFAULT_KELVIN).unwrap();
        assert!(green > 0.0,   "a green patch should push magenta, got {green}");
        assert!(magenta < 0.0, "a magenta patch should push green, got {magenta}");
    }

    /// The solve is only as good as the patch. Refusing is the useful answer here - a near-black
    /// or clipped sample has no ratio to read, and silently returning something plausible would
    /// throw the sliders somewhere the user cannot explain.
    #[test]
    fn unusable_patches_are_refused_rather_than_guessed_at() {
        let p = params(|_| {});
        assert_eq!(solve_white_balance([0.004, 0.005, 0.003], &p, DEFAULT_KELVIN), Err(PickError::TooDark));
        assert_eq!(solve_white_balance([0.99, 0.99, 0.99],    &p, DEFAULT_KELVIN), Err(PickError::Clipped));
        assert_eq!(solve_white_balance([0.5, 0.99, 0.5],      &p, DEFAULT_KELVIN), Err(PickError::Clipped));
        assert_eq!(solve_white_balance([f64::NAN, 0.5, 0.5],  &p, DEFAULT_KELVIN), Err(PickError::TooDark));

        // Saturation at its minimum maps every colour to the same grey, so there is genuinely no
        // way back and the honest answer is to say so.
        let flat = params(|s| s.saturation = -1.0);
        assert_eq!(solve_white_balance([0.5, 0.45, 0.4], &flat, DEFAULT_KELVIN), Err(PickError::GradeNotInvertible));
    }

    /// A cast stronger than the Planckian range can express clamps to the nearest reachable white
    /// point instead of failing - an extreme pick should still do the most it can.
    #[test]
    fn an_extreme_cast_clamps_to_the_ends_of_the_range() {
        let p = params(|_| {});
        let (hot,  _) = solve_white_balance([0.10, 0.30, 0.90], &p, DEFAULT_KELVIN).unwrap();
        let (cold, _) = solve_white_balance([0.90, 0.30, 0.10], &p, DEFAULT_KELVIN).unwrap();
        assert!(hot  <= MAX_KELVIN && hot  > 10000.0, "got {hot}K");
        assert!(cold >= MIN_KELVIN && cold < 3000.0,  "got {cold}K");
    }

    /// The picker moves the target, never the clip's anchor, so the same patch under a different
    /// anchor must still solve to something that neutralises it.
    #[test]
    fn the_solve_follows_the_source_anchor() {
        let p = params(|_| {});
        for source in [3200.0, 5000.0, DEFAULT_KELVIN, 7500.0] {
            let (kelvin, tint) = solve_white_balance([0.50, 0.45, 0.38], &p, source).unwrap();
            let g = kelvin_to_rgb_gains(source, kelvin, tint);
            let out = [0.50 * g[0] as f64, 0.45 * g[1] as f64, 0.38 * g[2] as f64];
            let spread = out.iter().cloned().fold(f64::MIN, f64::max) - out.iter().cloned().fold(f64::MAX, f64::min);
            assert!(spread < 2e-3, "source {source}K left {out:?} (spread {spread})");
        }
    }
}

