// Copyright © 2026 Gyroflow

//! Film simulation: per-stock tone and colour, silver-halide grain and halation, applied to the
//! graded output by a GPU chain after chroma smoothing.
//!
//! The maths is a line-for-line port of Photo360's `film.js` (see
//! `docs/superpowers/specs/2026-10-04-film-handover.md`); `test_vectors.json` holds numbers
//! produced by that file, and the tests below hold the port to them.

pub mod stocks;
pub mod curve;
pub mod noise;
pub mod grain;

/// Grain stocks are sized for a 3060 px short side taken as a Super 8 gate.
pub const REF_SHORT_PX: f64 = 3060.0;
pub const REF_SHORT_MM: f64 = 4.01;
/// Halation keeps its own reference: 0.8 % of the frame on 35 mm (24 mm gate).
pub const HALATION_RADIUS: f64 = 0.008;
pub const HALATION_REF_MM: f64 = 24.0;

/// Halation blur radius (sigma) in output px for a `d` px film-frame short side.
pub fn halation_radius(d: f64, gate_mm: f64) -> f64 { HALATION_RADIUS * d * HALATION_REF_MM / gate_mm }

/// The halation field's downsample factor: its blur sigma `r / f` stays at most 8 texels.
pub fn halo_factor(r: f64) -> u32 {
    if r <= 8.0 { 1 } else { (2f64.powf((r / 8.0).log2().ceil())).min(64.0) as u32 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn vectors() -> Value { serde_json::from_str(include_str!("test_vectors.json")).unwrap() }
    fn f(v: &Value) -> f64 { v.as_f64().unwrap() }
    fn u(v: &Value) -> u32 { v.as_u64().unwrap() as u32 }
    fn close(a: f64, b: f64, tol: f64, what: &str) { assert!((a - b).abs() <= tol * b.abs().max(1e-12), "{what}: {a} vs {b}"); }

    #[test]
    fn pcg_and_hash4_match_film_js() {
        let v = vectors();
        for p in v["pcg"].as_array().unwrap() { assert_eq!(noise::pcg(u(&p[0])), u(&p[1]), "pcg({})", p[0]); }
        for p in v["hash4"].as_array().unwrap() {
            let a: Vec<u32> = p[0].as_array().unwrap().iter().map(u).collect();
            assert_eq!(noise::hash4(a[0], a[1], a[2], a[3]), u(&p[1]), "hash4({a:?})");
        }
    }

    #[test]
    fn gaussian_fnv_and_seeds_match_film_js() {
        let v = vectors();
        for p in v["gaussian"].as_array().unwrap() { close(noise::gaussian(u(&p[0])), f(&p[1]), 1e-12, "gaussian"); }
        for p in v["fnv"].as_array().unwrap() { assert_eq!(noise::fnv1a_utf16(p[0].as_str().unwrap()), u(&p[1]), "fnv {}", p[0]); }
        let ls = &v["layer_seeds"][0];
        let want: Vec<u32> = ls[1].as_array().unwrap().iter().map(u).collect();
        assert_eq!(noise::layer_seeds(u(&ls[0]), 0).to_vec(), want);
    }

    #[test]
    fn special_functions_match_film_js() {
        let v = vectors();
        for p in v["inverse_normal"].as_array().unwrap() { close(grain::inverse_normal(f(&p[0])), f(&p[1]), 1e-12, "inverse_normal"); }
        for p in v["i0e"].as_array().unwrap() { close(grain::i0e(f(&p[0])), f(&p[1]), 1e-12, "i0e"); }
        for p in v["kernel_lag_one"].as_array().unwrap() { close(grain::kernel_lag_one(f(&p[0])), f(&p[1]), 1e-12, "kernel_lag_one"); }
        for p in v["quantize"].as_array().unwrap() { close(grain::quantize_scale(f(&p[0])), f(&p[1]), 1e-12, "quantize"); }
        for p in v["halo_factor"].as_array().unwrap() { assert_eq!(halo_factor(f(&p[0])) as f64, f(&p[1]), "halo_factor({})", p[0]); }
    }

    #[test]
    fn grain_model_matches_film_js() {
        for m in vectors()["model"].as_array().unwrap() {
            let s = stocks::stock(m["id"].as_str().unwrap()).unwrap();
            let model = grain::GrainModel::new(s.grain.radius_median_px, s.grain.radius_log_sigma, s.grain.filter_sigma_px).for_render_scale(f(&m["scale"]));
            close(model.filtered_variance(0.1), f(&m["var10"]), 1e-9, "var10");
            close(model.filtered_variance(0.5), f(&m["var50"]), 1e-9, "var50");
            close(model.filtered_variance(0.9), f(&m["var90"]), 1e-9, "var90");
            close(model.lag_one_correlation(0.5), f(&m["lag1"]), 1e-9, "lag1");
        }
    }

    #[test]
    fn grain_setup_matches_film_js() {
        for m in vectors()["setup"].as_array().unwrap() {
            let s = stocks::stock(m["id"].as_str().unwrap()).unwrap();
            let g = grain::grain_setup(&s.grain, f(&m["scale"]), f(&m["z"]));
            close(g.sigma, f(&m["sigma"]), 1e-9, "sigma");
            assert_eq!(g.radius as f64, f(&m["radius"]));
            close(g.norm, f(&m["norm"]), 1e-9, "norm");
            close(g.std[50] as f64, f(&m["std50"]), 1e-6, "std50");
            close(g.std[100] as f64, f(&m["std100"]), 1e-6, "std100");
            close(g.std[180] as f64, f(&m["std180"]), 1e-6, "std180");
        }
    }

    /// The handover's reference table: fit and output after crosstalk for neutral linear 0, 0.18, 1.
    #[test]
    fn curve_fit_matches_the_handover_table() {
        #[rustfmt::skip]
        let table: [(&str, f64, f64, [f64; 3], [f64; 3], [f64; 3]); 8] = [
            ("portrait160",  -0.4134, 1.4131, [0.040, 0.040, 0.047], [0.462, 0.462, 0.434], [0.960, 0.961, 0.923]),
            ("golden200",    -0.2339, 1.2222, [0.036, 0.029, 0.050], [0.475, 0.463, 0.388], [0.970, 0.962, 0.890]),
            ("vivid100",     -0.1826, 1.1521, [0.030, 0.027, 0.037], [0.472, 0.461, 0.454], [0.966, 0.961, 0.942]),
            ("pastel400",    -0.7714, 1.7700, [0.082, 0.090, 0.098], [0.447, 0.470, 0.459], [0.942, 0.971, 0.943]),
            ("verdant400",   -0.3414, 1.2715, [0.021, 0.057, 0.035], [0.359, 0.473, 0.416], [0.888, 0.968, 0.937]),
            ("tungsten800",  -0.2799, 1.2744, [0.047, 0.039, 0.054], [0.399, 0.461, 0.536], [0.911, 0.961, 0.996]),
            ("classic400bw", -0.0974, 1.1005, [0.020, 0.020, 0.020], [0.461, 0.461, 0.461], [0.960, 0.960, 0.960]),
            ("smooth400bw",  -0.2853, 1.2998, [0.040, 0.040, 0.040], [0.461, 0.461, 0.461], [0.960, 0.960, 0.960]),
        ];
        for (id, b, s, black, mid, white) in table {
            let st = stocks::stock(id).unwrap();
            let fit = curve::fit_curve(st);
            close(fit.b, b, 0.00006 / b.abs(), &format!("{id} b"));
            close(fit.s, s, 0.00006 / s, &format!("{id} s"));
            for (lin, want) in [(0.0, black), (0.18, mid), (1.0, white)] {
                let u = curve::curve_pixel([lin; 3], st, &fit);
                for c in 0..3 {
                    let m = &st.crosstalk;
                    let o = (m[c * 3] * u[0] + m[c * 3 + 1] * u[1] + m[c * 3 + 2] * u[2]).clamp(0.0, 1.0);
                    assert!((o - want[c]).abs() <= 0.0006, "{id} lin {lin} channel {c}: {o} vs {}", want[c]);
                }
            }
        }
    }

    #[test]
    fn formats_round_trip_and_scale() {
        for f in stocks::FilmFormat::ALL { assert_eq!(stocks::FilmFormat::from_id(f.id()), Some(f)); }
        // Handover table: D = 5100, Super 16 -> 0.902, 35 mm -> 0.278.
        let s16 = 5100.0 / REF_SHORT_PX * REF_SHORT_MM / stocks::FilmFormat::Super16.gate_short_mm();
        assert!((s16 - 0.902).abs() < 0.001);
        assert!((halation_radius(5100.0, 24.0) - 40.8).abs() < 0.1);
    }
}
