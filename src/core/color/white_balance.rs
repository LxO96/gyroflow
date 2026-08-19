// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Adrian <adrian.eddy at gmail>

/// Correlated colour temperature to CIE 1931 xy, using Kim et al.'s cubic
/// approximation of the Planckian locus. Valid 1667–25000 K; inputs are clamped.
fn cct_to_xy(kelvin: f64) -> (f64, f64) {
    let t = kelvin.clamp(1667.0, 25000.0);
    let t2 = t * t;
    let t3 = t2 * t;

    let x = if t <= 4000.0 {
        -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.179910
    } else {
        -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390
    };
    let x2 = x * x;
    let x3 = x2 * x;

    let y = if t <= 2222.0 {
        -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
    } else if t <= 4000.0 {
        -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
    } else {
         3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483
    };
    (x, y)
}

/// White point at the given temperature as linear sRGB, normalised to Y = 1.
fn white_point_rgb(kelvin: f64) -> [f64; 3] {
    let (x, y) = cct_to_xy(kelvin);
    let y = if y.abs() < 1e-9 { 1e-9 } else { y };

    // xyY -> XYZ with Y = 1
    let big_x = x / y;
    let big_y = 1.0;
    let big_z = (1.0 - x - y) / y;

    // XYZ -> linear sRGB (D65)
    [
         3.2404542 * big_x - 1.5371385 * big_y - 0.4985314 * big_z,
        -0.9692660 * big_x + 1.8760108 * big_y + 0.0415560 * big_z,
         0.0556434 * big_x - 0.2040259 * big_y + 1.0572252 * big_z,
    ]
}

/// Per-channel gains that re-balance an image shot at `source_k` so that
/// `target_k` reads as white. Raising `target_k` warms the image.
///
/// `tint` is -1..1 along the green–magenta axis: positive pushes magenta
/// (green down), negative pushes green.
///
/// The result is luma-normalised, so white balance never changes brightness.
pub fn kelvin_to_rgb_gains(source_k: f64, target_k: f64, tint: f64) -> [f32; 3] {
    let src = white_point_rgb(source_k);
    let dst = white_point_rgb(target_k);

    let mut gains = [0.0f64; 3];
    for c in 0..3 {
        let d = if dst[c].abs() < 1e-6 { 1e-6 } else { dst[c] };
        gains[c] = (src[c] / d).max(1e-6);
    }

    // Green-magenta trim, scaled so ±1 is a visible but usable range.
    let t = tint.clamp(-1.0, 1.0);
    gains[1] *= 1.0 - 0.2 * t;

    // Normalise so luma is unchanged.
    let luma = 0.2126 * gains[0] + 0.7152 * gains[1] + 0.0722 * gains[2];
    let luma = if luma.abs() < 1e-9 { 1.0 } else { luma };

    [(gains[0] / luma) as f32, (gains[1] / luma) as f32, (gains[2] / luma) as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn luma(g: [f32; 3]) -> f32 { 0.2126 * g[0] + 0.7152 * g[1] + 0.0722 * g[2] }

    #[test]
    fn neutral_when_target_equals_source() {
        let g = kelvin_to_rgb_gains(5500.0, 5500.0, 0.0);
        for c in 0..3 { assert!((g[c] - 1.0).abs() < 1e-5, "channel {c} was {}", g[c]); }
    }

    #[test]
    fn neutral_at_several_temperatures() {
        for k in [2500.0, 4000.0, 5500.0, 6500.0, 9000.0] {
            let g = kelvin_to_rgb_gains(k, k, 0.0);
            for c in 0..3 { assert!((g[c] - 1.0).abs() < 1e-5, "{k}K channel {c} was {}", g[c]); }
        }
    }

    #[test]
    fn raising_target_warms_the_image() {
        let g = kelvin_to_rgb_gains(5500.0, 8000.0, 0.0);
        assert!(g[0] > 1.0, "red gain should rise, was {}", g[0]);
        assert!(g[2] < 1.0, "blue gain should fall, was {}", g[2]);
    }

    #[test]
    fn lowering_target_cools_the_image() {
        let g = kelvin_to_rgb_gains(5500.0, 3200.0, 0.0);
        assert!(g[0] < 1.0, "red gain should fall, was {}", g[0]);
        assert!(g[2] > 1.0, "blue gain should rise, was {}", g[2]);
    }

    #[test]
    fn luma_is_preserved_across_the_range() {
        for target in [2000.0, 3200.0, 5500.0, 6500.0, 10000.0, 15000.0] {
            let g = kelvin_to_rgb_gains(5500.0, target, 0.0);
            assert!((luma(g) - 1.0).abs() < 1e-4, "{target}K luma was {}", luma(g));
        }
    }

    #[test]
    fn positive_tint_pushes_magenta_negative_pushes_green() {
        let magenta = kelvin_to_rgb_gains(5500.0, 5500.0, 1.0);
        let green   = kelvin_to_rgb_gains(5500.0, 5500.0, -1.0);
        assert!(magenta[1] < 1.0, "green channel should drop for magenta tint");
        assert!(green[1]   > 1.0, "green channel should rise for green tint");
        assert!((luma(magenta) - 1.0).abs() < 1e-4);
        assert!((luma(green)   - 1.0).abs() < 1e-4);
    }

    #[test]
    fn gains_stay_finite_and_positive_at_extremes() {
        for (s, t) in [(1500.0, 20000.0), (20000.0, 1500.0), (1000.0, 1000.0)] {
            let g = kelvin_to_rgb_gains(s, t, 0.0);
            for c in 0..3 { assert!(g[c].is_finite() && g[c] > 0.0, "({s},{t}) channel {c} was {}", g[c]); }
        }
    }
}
