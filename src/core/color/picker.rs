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
//! It only has to undo four stages - the three below white balance, all of which invert exactly,
//! and then white balance itself, which is three divisions - and solve for the gains that would
//! have made the result grey.
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

/// Whether one sample is worth averaging into a patch: not so dark that its channel ratios are
/// noise, not so bright that any channel may have clipped.
///
/// The same bounds [`solve_white_balance`] applies to the finished average, applied per sample so
/// that dragging a region across a grey wall does not quietly average in a blown window.
pub fn is_usable_sample(rgb: [f64; 3]) -> bool {
    rgb.iter().all(|v| v.is_finite() && *v >= MIN_USABLE && *v <= MAX_USABLE)
}

/// A click, as a square of this many pixels a side, centred on the point.
///
/// Video is noisy: at 10-bit H.264 a single sample can sit a couple of codes off its neighbours,
/// which is enough to move the solved temperature by a few hundred kelvin on a low-contrast grey.
const CLICK_SPAN: i32 = 5;

/// Most samples any one pick reads, per axis.
///
/// A drag across a 1900x1200 preview covers 2.3 million pixels and every read crosses into Qt, so
/// reading them all would stall the UI for seconds. An even grid of at most 64x64 points spread
/// over the region is both fast and no worse statistically - the answer is a mean, and 4096
/// samples pin a mean far tighter than the frame's own noise floor.
const MAX_SAMPLES_PER_AXIS: i32 = 64;

/// The colour a pick averages out of an image region.
///
/// `rect` is `[x0, y0, x1, y1]` in 0..1 of the image, in any order - a click is the degenerate
/// case where the two corners coincide, and is widened to [`CLICK_SPAN`] square about its centre
/// so that clicking and dragging share one code path.
///
/// `read` is handed integer pixel coordinates, always inside the image.
///
/// Unusable samples are dropped rather than averaged in. If too few survive to be worth trusting
/// the *unfiltered* mean is returned instead, so that [`solve_white_balance`] sees the patch the
/// user actually pointed at and refuses it with the right reason, rather than this function
/// inventing a failure of its own.
pub fn average_patch(width: i32, height: i32, rect: [f64; 4], read: impl Fn(u32, u32) -> [f64; 3]) -> [f64; 3] {
    let px = |v: f64, max: i32| (v.clamp(0.0, 1.0) * (max - 1).max(0) as f64).round() as i32;
    let (mut x0, mut x1) = (px(rect[0], width),  px(rect[2], width));
    let (mut y0, mut y1) = (px(rect[1], height), px(rect[3], height));
    if x0 > x1 { std::mem::swap(&mut x0, &mut x1); }
    if y0 > y1 { std::mem::swap(&mut y0, &mut y1); }

    // Widen anything narrower than a click to click size, about its own centre. Done on both axes
    // independently so a thin horizontal drag keeps its length and only gains height.
    let widen = |a: &mut i32, b: &mut i32, max: i32| {
        let short = CLICK_SPAN - (*b - *a + 1);
        if short > 0 {
            *a -= short / 2;
            *b += short - short / 2;
        }
        // Shift rather than clip at the edges, so a pick near a border still averages the full
        // count and cannot be biased by whichever side fell off.
        if *a < 0 { *b -= *a; *a = 0; }
        if *b > max - 1 { *a -= *b - (max - 1); *b = max - 1; }
        *a = (*a).max(0);
        *b = (*b).min(max - 1).max(*a);
    };
    widen(&mut x0, &mut x1, width);
    widen(&mut y0, &mut y1, height);

    let step = |a: i32, b: i32| {
        let span = b - a + 1;
        (span, span.min(MAX_SAMPLES_PER_AXIS).max(1))
    };
    let (span_x, count_x) = step(x0, x1);
    let (span_y, count_y) = step(y0, y1);

    let mut all = [0.0f64; 3];
    let mut usable = [0.0f64; 3];
    let (mut n_all, mut n_usable) = (0.0f64, 0.0f64);

    for iy in 0..count_y {
        // Sample at the centre of each cell rather than its corner, so a grid coarser than the
        // region does not bias toward the top-left.
        let y = y0 + ((iy as i64 * 2 + 1) * span_y as i64 / (count_y as i64 * 2)) as i32;
        for ix in 0..count_x {
            let x = x0 + ((ix as i64 * 2 + 1) * span_x as i64 / (count_x as i64 * 2)) as i32;
            let s = read(x.clamp(0, width - 1) as u32, y.clamp(0, height - 1) as u32);
            for c in 0..3 { all[c] += s[c]; }
            n_all += 1.0;
            if is_usable_sample(s) {
                for c in 0..3 { usable[c] += s[c]; }
                n_usable += 1.0;
            }
        }
    }

    if n_all == 0.0 { return [0.0; 3]; }

    // A handful of surviving samples out of thousands is not a reading, it is whatever happened to
    // sit at the edge of a highlight. Below that, hand back what was actually pointed at.
    let enough = (n_all * 0.05).max(9.0);
    if n_usable >= enough {
        [usable[0] / n_usable, usable[1] / n_usable, usable[2] / n_usable]
    } else {
        [all[0] / n_all, all[1] / n_all, all[2] / n_all]
    }
}

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

    // What white balance *produced*, before the stages below it dressed it up.
    let mut a = undo_stages_below_white_balance(patch, p)?;

    // ...and then back through white balance itself, to what it was handed.
    //
    // Easy to leave out, and wrong in a way that only shows on the second pick: the sampled pixel
    // already carries whatever correction is on the sliders now. Solving from it directly answers
    // "what would neutralise a pixel that has already been neutralised", which is "nothing" - so
    // a second pick on the same patch would throw the temperature back to the source anchor and
    // undo the first. Dividing the gains out makes the solve absolute and idempotent: pick twice,
    // get the same answer, whatever the sliders were at when you started.
    for i in 0..3 { a[i] /= (p.wb_gains[i] as f64).max(1e-9); }

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
            // The last two start from a white balance that is already off neutral, which is what
            // a second pick - or a pick after nudging the temperature by hand - actually sees.
            for (contrast, saturation, shadows, highlights, k0, t0) in [
                (0.0, 0.0, 0.0, 0.0, DEFAULT_KELVIN, 0.0),
                (0.4, -0.3, 0.6, 0.5, DEFAULT_KELVIN, 0.0),
                (0.0, 0.0, 0.0, 0.0, 3200.0, 0.3),
                (0.4, -0.3, 0.6, 0.5, 8000.0, -0.4),
            ] {
                let before = params(|s| { s.contrast = contrast; s.saturation = saturation; s.shadows = shadows; s.highlights = highlights; s.target_kelvin = k0; s.tint = t0; });
                let (kelvin, tint) = solve_white_balance(cast, &before, DEFAULT_KELVIN).unwrap();

                let after = params(|s| {
                    s.contrast = contrast; s.saturation = saturation; s.shadows = shadows; s.highlights = highlights;
                    s.target_kelvin = kelvin; s.tint = tint;
                });
                // The sampled patch is post-grade, so to re-grade it we start from the value the
                // grade was handed: undo the stages below white balance, then white balance too.
                let mut a = undo_stages_below_white_balance(cast, &before).unwrap();
                for i in 0..3 { a[i] /= before.wb_gains[i] as f64; }
                let out = apply_color(f32x3(a), &after, None);

                let spread = out.iter().cloned().fold(f32::MIN, f32::max) - out.iter().cloned().fold(f32::MAX, f32::min);

                // Neutral, unless the correction needed more than the sliders can express - a
                // strong cast on top of an already-shifted starting point can want past tint
                // +/-1. There the promise is only that it gets closer, which is what the clamp is
                // for; asserting neutrality would mean dropping the extreme cases from the test.
                let clamped = tint.abs() >= 1.0 - 1e-9
                           || kelvin <= crate::color::MIN_KELVIN + 1.0
                           || kelvin >= crate::color::MAX_KELVIN - 1.0;
                let cast_spread = (cast.iter().cloned().fold(f64::MIN, f64::max) - cast.iter().cloned().fold(f64::MAX, f64::min)) as f32;
                let limit = if clamped { cast_spread * 0.5 } else { 2e-3 };
                assert!(spread < limit, "{cast:?} at contrast {contrast}, starting {k0}K/{t0}: solved {kelvin:.0}K tint {tint:.3} left {out:?} (spread {spread}, limit {limit})");
            }
        }
    }

    /// Picking twice must give the same answer.
    ///
    /// The regression this pins: the eyedropper samples the *displayed* pixel, which already
    /// carries whatever white balance is on the sliders. A solve that does not divide those gains
    /// back out reads an already-corrected patch as neutral and resets the temperature to the
    /// source anchor - so a second pick silently undoes the first. Found by picking twice in the
    /// app: a region pick set 4534 K, and clicking the same fabric again jumped it back to 5059 K.
    #[test]
    fn picking_a_second_time_does_not_undo_the_first() {
        let cast = [0.50, 0.45, 0.38];
        let first = params(|_| {});
        let (k1, t1) = solve_white_balance(cast, &first, DEFAULT_KELVIN).unwrap();

        // What the screen shows once that pick is applied: the same source pixel, re-graded.
        let applied = params(|s| { s.target_kelvin = k1; s.tint = t1; });
        let mut a = undo_stages_below_white_balance(cast, &first).unwrap();
        for i in 0..3 { a[i] /= first.wb_gains[i] as f64; }
        let displayed = apply_color(f32x3(a), &applied, None);

        let (k2, t2) = solve_white_balance([displayed[0] as f64, displayed[1] as f64, displayed[2] as f64], &applied, DEFAULT_KELVIN).unwrap();
        assert!((k2 - k1).abs() < 25.0,  "second pick moved {k1:.0}K to {k2:.0}K");
        assert!((t2 - t1).abs() < 0.01,  "second pick moved tint {t1:.3} to {t2:.3}");
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

    // ---- average_patch ----

    /// A grid of solid colour, so any correct sampling returns that colour and any indexing bug
    /// shows up as an out-of-bounds panic rather than a slightly wrong number.
    fn solid(c: [f64; 3]) -> impl Fn(u32, u32) -> [f64; 3] { move |_, _| c }

    /// A click is the degenerate rectangle. It must keep reading the 5x5 block it always did -
    /// one pixel is noise, and the whole point of the drag is that the click still works.
    #[test]
    fn a_click_still_reads_a_five_by_five_block() {
        let seen = std::cell::RefCell::new(Vec::new());
        let out = average_patch(100, 100, [0.5, 0.5, 0.5, 0.5], |x, y| {
            seen.borrow_mut().push((x, y));
            [0.4, 0.4, 0.4]
        });
        for c in out { assert!((c - 0.4).abs() < 1e-9, "{out:?}"); }
        let seen = seen.into_inner();
        assert_eq!(seen.len(), 25, "read {} pixels, wanted a 5x5 block", seen.len());
        let xs: std::collections::BTreeSet<u32> = seen.iter().map(|p| p.0).collect();
        let ys: std::collections::BTreeSet<u32> = seen.iter().map(|p| p.1).collect();
        assert_eq!(xs.len(), 5);
        assert_eq!(ys.len(), 5);
    }

    /// The property the feature rests on: over a uniform region, dragging and clicking agree.
    #[test]
    fn dragging_over_a_uniform_region_matches_clicking_inside_it() {
        let c = [0.46, 0.42, 0.38];
        let click = average_patch(400, 300, [0.5, 0.5, 0.5, 0.5], solid(c));
        let drag  = average_patch(400, 300, [0.2, 0.3, 0.8, 0.9], solid(c));
        for i in 0..3 { assert!((click[i] - drag[i]).abs() < 1e-9, "{click:?} vs {drag:?}"); }
    }

    /// Corners in any order describe the same region - a drag up and to the left is still a drag.
    #[test]
    fn the_corners_may_be_given_in_any_order() {
        let c = [0.5, 0.45, 0.4];
        let a = average_patch(400, 300, [0.2, 0.3, 0.8, 0.9], solid(c));
        let b = average_patch(400, 300, [0.8, 0.9, 0.2, 0.3], solid(c));
        for i in 0..3 { assert!((a[i] - b[i]).abs() < 1e-12, "{a:?} vs {b:?}"); }
    }

    /// The reason per-sample filtering exists: drag across a grey wall, catch a window, and the
    /// blown pixels must not drag the answer with them. Half the region is clipped white here,
    /// which would shift a naive mean by a third.
    #[test]
    fn clipped_samples_inside_a_region_do_not_move_the_result() {
        let grey = [0.45, 0.42, 0.39];
        let out = average_patch(400, 300, [0.0, 0.0, 1.0, 1.0], |x, _| {
            if x > 200 { [1.0, 1.0, 1.0] } else { grey }
        });
        for i in 0..3 { assert!((out[i] - grey[i]).abs() < 1e-9, "{out:?} should be {grey:?}"); }
    }

    /// ...and the same for crushed blacks at the other end.
    #[test]
    fn near_black_samples_inside_a_region_do_not_move_the_result() {
        let grey = [0.45, 0.42, 0.39];
        let out = average_patch(400, 300, [0.0, 0.0, 1.0, 1.0], |_, y| {
            if y > 150 { [0.001, 0.001, 0.001] } else { grey }
        });
        for i in 0..3 { assert!((out[i] - grey[i]).abs() < 1e-9, "{out:?} should be {grey:?}"); }
    }

    /// When almost nothing survives the filter, the honest answer is the patch the user pointed
    /// at - so that `solve_white_balance` refuses it with the right reason instead of this
    /// function silently reporting whatever few pixels happened to sit inside the bounds.
    #[test]
    fn an_entirely_unusable_region_returns_what_was_pointed_at() {
        let dark = [0.004, 0.005, 0.003];
        let out = average_patch(400, 300, [0.0, 0.0, 1.0, 1.0], solid(dark));
        for i in 0..3 { assert!((out[i] - dark[i]).abs() < 1e-9, "{out:?} should be {dark:?}"); }

        let p = params(|_| {});
        assert_eq!(solve_white_balance(out, &p, DEFAULT_KELVIN), Err(PickError::TooDark));
    }

    /// A full-frame drag must not read every pixel: each read crosses into Qt, and 2.3 million of
    /// them would stall the UI for seconds.
    #[test]
    fn a_large_region_is_subsampled_rather_than_read_whole() {
        let count = std::cell::Cell::new(0usize);
        average_patch(1920, 1080, [0.0, 0.0, 1.0, 1.0], |_, _| {
            count.set(count.get() + 1);
            [0.5, 0.5, 0.5]
        });
        assert!(count.get() <= 64 * 64, "read {} samples", count.get());
        assert!(count.get() >= 32 * 32, "read only {} samples, too coarse", count.get());
    }

    /// Picks at the very edge, and regions larger than the image, must stay in bounds - `read`
    /// asserts it, and a 1x1 image is the meanest case there is.
    #[test]
    fn every_read_stays_inside_the_image() {
        for (w, h) in [(1, 1), (3, 3), (7, 200), (640, 480)] {
            for rect in [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, 2.0, 2.0], [0.99, 0.0, 1.0, 0.01]] {
                average_patch(w, h, rect, |x, y| {
                    assert!((x as i32) < w && (y as i32) < h, "{w}x{h} {rect:?} read ({x},{y})");
                    [0.5, 0.5, 0.5]
                });
            }
        }
    }

    #[test]
    fn usable_samples_are_the_ones_solve_would_accept() {
        assert!(is_usable_sample([0.5, 0.5, 0.5]));
        assert!(!is_usable_sample([0.5, 0.99, 0.5]));
        assert!(!is_usable_sample([0.5, 0.001, 0.5]));
        assert!(!is_usable_sample([f64::NAN, 0.5, 0.5]));
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

