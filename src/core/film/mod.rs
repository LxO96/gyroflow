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

pub mod params;
pub mod reference;
pub use params::FilmParams;
use params::*;
use stocks::FilmFormat;

#[derive(Clone, Debug, PartialEq)]
pub struct FilmSettings {
    /// `None` = Off.
    pub stock: Option<String>,
    pub format: FilmFormat,
    pub tone: bool,
    pub grain: bool,
    pub halation: bool,
    /// 0..2, 1 = the stock's own grain.
    pub grain_amount: f64,
    /// 0..1.
    pub halation_amount: f64,
}
impl Default for FilmSettings {
    fn default() -> Self { Self { stock: None, format: FilmFormat::Mm35, tone: true, grain: true, halation: true, grain_amount: 1.0, halation_amount: 0.5 } }
}
impl FilmSettings {
    pub fn set_param(&mut self, name: &str, v: f64) -> bool {
        if !v.is_finite() { return false; }
        match name {
            "grain_amount" => self.grain_amount = v.clamp(0.0, 2.0),
            "halation_amount" => self.halation_amount = v.clamp(0.0, 1.0),
            _ => return false,
        }
        true
    }
    pub fn set_flag(&mut self, name: &str, v: bool) -> bool {
        match name { "tone" => self.tone = v, "grain" => self.grain = v, "halation" => self.halation = v, _ => return false }
        true
    }
    /// An unknown stock id is Off: a newer project opened in an older build must not crash or guess.
    pub fn set_stock(&mut self, id: &str) { self.stock = stocks::stock(id).map(|s| s.id.to_string()); }
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "stock": self.stock.clone().unwrap_or_else(|| "off".into()),
            "format": self.format.id(),
            "tone": self.tone, "grain": self.grain, "halation": self.halation,
            "grain_amount": self.grain_amount, "halation_amount": self.halation_amount,
        })
    }
    /// Missing keys keep their current value; out-of-range values are clamped.
    pub fn apply_json(&mut self, v: &serde_json::Value) {
        if let Some(s) = v.get("stock").and_then(|x| x.as_str()) { self.set_stock(s); }
        if let Some(f) = v.get("format").and_then(|x| x.as_str()).and_then(FilmFormat::from_id) { self.format = f; }
        for k in ["tone", "grain", "halation"] { if let Some(b) = v.get(k).and_then(|x| x.as_bool()) { self.set_flag(k, b); } }
        for k in ["grain_amount", "halation_amount"] { if let Some(x) = v.get(k).and_then(|x| x.as_f64()) { self.set_param(k, x); } }
    }
}

/// Everything per-clip the film chain needs; `frame_params` makes one frame's block.
#[derive(Clone, Debug)]
pub struct FilmContext { pub settings: FilmSettings, pub export_short: usize, pub fps: f64, pub seed: u32 }

impl FilmContext {
    /// `None` when the chain would do nothing: stock Off or unknown, or every part off.
    pub fn new(settings: FilmSettings, export_short: usize, fps: f64, file_name: &str) -> Option<Self> {
        let stock = stocks::stock(settings.stock.as_deref()?)?;
        let grain = settings.grain && settings.grain_amount > 0.0;
        let halo = settings.halation && settings.halation_amount > 0.0 && stock.halation.gain > 0.0;
        if !(settings.tone || grain || halo) || export_short == 0 { return None; }
        Some(Self { settings, export_short, fps, seed: noise::fnv1a_utf16(file_name) })
    }

    /// One frame's parameters for a `buf_w` x `buf_h` buffer. Sizes and stride are filled by the backend.
    pub fn frame_params(&self, buf_w: usize, buf_h: usize, source_timestamp_us: i64, max_value: f32) -> FilmParams {
        let s = &self.settings;
        let stock = stocks::stock(s.stock.as_deref().unwrap_or_default()).expect("FilmContext::new checked the stock");
        let gate = s.format.gate_short_mm();
        let d = self.export_short as f64;
        let p = (buf_w.min(buf_h) as f64 / d).min(1.0);
        let mut fp = FilmParams { max_value, inv_p: (1.0 / p) as f32, ..Default::default() };

        if s.tone {
            fp.flags |= FLAG_TONE;
            let fit = curve::fit_curve(stock);
            fp.fit_b = fit.b as f32; fp.fit_s = fit.s as f32; fp.fit_umin_a = fit.u_min_a as f32;
            for i in 0..3 {
                let l = &stock.layers[i];
                let range = l.u_max - l.u_min;
                fp.log_e0[i] = l.log_e0 as f32; fp.k[i] = (4.0 * l.gamma / range) as f32;
                fp.u_min[i] = l.u_min as f32; fp.range[i] = range as f32;
                for j in 0..3 { fp.sens[i][j] = stock.sensitivity[i * 3 + j] as f32; fp.cross[i][j] = stock.crosstalk[i * 3 + j] as f32; }
            }
        }
        if stock.bw { fp.flags |= FLAG_BW; }
        if s.grain && s.grain_amount > 0.0 {
            fp.flags |= FLAG_GRAIN;
            let scale = grain::quantize_scale(d / REF_SHORT_PX * REF_SHORT_MM / gate);
            let z = if p >= 1.0 { 1.0 } else { grain::quantize_scale(p) };
            let g = grain::grain_setup(&stock.grain, scale, z);
            fp.grain_sigma = g.sigma as f32;
            fp.grain_radius = g.radius as i32;
            fp.grain_k = (s.grain_amount * g.norm) as f32;
            fp.std_table[..=grain::STD_TABLE].copy_from_slice(&g.std);
            let frame = (source_timestamp_us as f64 * self.fps / 1_000_000.0).round().max(0.0) as u32;
            let ls = noise::layer_seeds(self.seed, frame);
            fp.seeds = [ls[0], ls[1], ls[2], 0];
        }
        if s.halation && s.halation_amount > 0.0 && stock.halation.gain > 0.0 {
            fp.flags |= FLAG_HALATION;
            let r = halation_radius(d, gate) * p;
            let f = halo_factor(r);
            fp.halo_f = f as i32;
            fp.halo_w = buf_w.div_ceil(f as usize) as i32;
            fp.halo_h = buf_h.div_ceil(f as usize) as i32;
            fp.halo_sigma = (r / f as f64) as f32;
            fp.halo_radius = ((grain::gaussian_kernel(r / f as f64).len() - 1) / 2) as i32;
            fp.halo_strength = (s.halation_amount * stock.halation.gain) as f32;
            fp.halo_tint = [stock.halation.tint[0] as f32, stock.halation.tint[1] as f32, stock.halation.tint[2] as f32, 0.0];
        }
        fp
    }
}

/// The film chain runs on packed RGBA only, in R, G, B order, 8 or 16 bits. By type name rather
/// than `TypeId`: the generic callers (`process_pixels_with_references::<T>`) carry no `'static`
/// bound, and BGRA8 / AYUV16 share these types' OpenCL and wgpu formats, so formats cannot tell.
pub fn supported<T: crate::stabilization::PixelType>() -> bool {
    let n = std::any::type_name::<T>();
    n.ends_with("::RGBA8") || n.ends_with("::RGBA16")
}

/// Test support: a halo grid of factor `f` regardless of the radius the settings give.
#[cfg(test)]
pub(crate) fn force_halo_grid(p: &mut FilmParams, f: i32, w: usize, h: usize) {
    p.flags |= FLAG_HALATION;
    p.halo_f = f;
    p.halo_w = w.div_ceil(f as usize) as i32;
    p.halo_h = h.div_ceil(f as usize) as i32;
    p.halo_sigma = 2.0;
    p.halo_radius = 6;
    if p.halo_strength == 0.0 { p.halo_strength = 1.0; p.halo_tint = [1.0, 0.28, 0.05, 0.0]; }
}

static LAST_SKIP: parking_lot::Mutex<Option<String>> = parking_lot::Mutex::new(None);
/// Why the GPU last skipped a film frame it was asked to render (build failure, unsupported
/// output). Shown in the Film panel; the render itself continues without film.
pub fn set_last_skip(why: String) { ::log::warn!("Film simulation skipped: {why}"); *LAST_SKIP.lock() = Some(why); }
pub fn last_skip() -> Option<String> { LAST_SKIP.lock().clone() }

#[cfg(test)] mod tests_gpu;

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
    fn ctx(stock: &str) -> FilmContext {
        let s = FilmSettings { stock: Some(stock.into()), ..Default::default() };
        FilmContext::new(s, 256, 25.0, "C0001.MP4").unwrap()
    }
    fn grey(w: usize, h: usize, v: f32) -> Vec<[f32; 4]> { vec![[v, v, v, 1.0]; w * h] }
    fn mean_std(xs: &[f32]) -> (f64, f64) {
        let n = xs.len() as f64;
        let m = xs.iter().map(|&x| x as f64).sum::<f64>() / n;
        (m, (xs.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / n).sqrt())
    }

    #[test]
    fn film_params_has_the_layout_every_kernel_mirrors() {
        assert_eq!(std::mem::size_of::<params::FilmParams>(), 1296);
        assert_eq!(std::mem::offset_of!(params::FilmParams, seeds), 80);
        assert_eq!(std::mem::offset_of!(params::FilmParams, std_table), 272);
    }

    #[test]
    fn off_or_nothing_enabled_is_no_context() {
        assert!(FilmContext::new(FilmSettings::default(), 256, 25.0, "a").is_none(), "stock Off");
        let s = FilmSettings { stock: Some("golden200".into()), tone: false, grain: false, halation: false, ..Default::default() };
        assert!(FilmContext::new(s, 256, 25.0, "a").is_none(), "all toggles off");
        let s = FilmSettings { stock: Some("no-such-stock".into()), ..Default::default() };
        assert!(FilmContext::new(s, 256, 25.0, "a").is_none(), "unknown stock");
    }

    #[test]
    fn grain_moves_every_frame_and_repeats_per_frame() {
        let c = ctx("golden200");
        let src = grey(64, 64, 0.46);
        let a = reference::render(&src, 64, 64, &c.frame_params(64, 64, 0, 1.0));
        let a2 = reference::render(&src, 64, 64, &c.frame_params(64, 64, 0, 1.0));
        let b = reference::render(&src, 64, 64, &c.frame_params(64, 64, 40_000, 1.0)); // next frame at 25 fps
        assert_eq!(a, a2, "same frame must render identically");
        assert!(a.iter().zip(&b).filter(|(x, y)| x != y).count() > 64 * 64 / 2, "grain did not move between frames");
    }

    /// Handover verification: on a flat patch the measured std is amount·std(u) and the lag-1
    /// correlation the model's, at full size and at a 1/4 preview.
    #[test]
    fn grain_statistics_match_the_model() {
        for (w, export_short) in [(256usize, 256usize), (256, 1024)] {
            let s = FilmSettings { stock: Some("golden200".into()), tone: false, halation: false, ..Default::default() };
            let c = FilmContext::new(s, export_short, 25.0, "x").unwrap();
            let p = c.frame_params(w, w, 0, 1.0);
            let out = reference::render(&grey(w, w, 0.46), w, w, &p);
            let d: Vec<f32> = out.iter().map(|px| px[1] - 0.46).collect();
            let (_, sd) = mean_std(&d);
            let want = reference::std_of(&p, 0.46) as f64;
            assert!((sd / want - 1.0).abs() < 0.05, "std {sd} vs model {want} (export_short {export_short})");
            let num: f64 = (0..w).flat_map(|y| (0..w - 1).map(move |x| (y, x))).map(|(y, x)| d[y * w + x] as f64 * d[y * w + x + 1] as f64).sum::<f64>() / (w * (w - 1)) as f64;
            let lag1 = num / (sd * sd);
            let setup = grain::grain_setup(&stocks::stock("golden200").unwrap().grain, p_scale(&c), p_z(&c, w));
            assert!((lag1 - setup.lag_one).abs() < 0.05, "lag1 {lag1} vs model {} (export_short {export_short})", setup.lag_one);
        }
    }
    fn p_scale(c: &FilmContext) -> f64 { grain::quantize_scale(c.export_short as f64 / REF_SHORT_PX * REF_SHORT_MM / c.settings.format.gate_short_mm()) }
    fn p_z(c: &FilmContext, buf_short: usize) -> f64 { let p = (buf_short as f64 / c.export_short as f64).min(1.0); if p >= 1.0 { 1.0 } else { grain::quantize_scale(p) } }

    #[test]
    fn bw_grain_stays_neutral() {
        let s = FilmSettings { stock: Some("classic400bw".into()), tone: false, halation: false, ..Default::default() };
        let c = FilmContext::new(s, 64, 25.0, "x").unwrap();
        for px in reference::render(&grey(64, 64, 0.4), 64, 64, &c.frame_params(64, 64, 0, 1.0)) {
            assert!(px[0] == px[1] && px[1] == px[2], "B&W grain tinted a pixel: {px:?}");
        }
    }

    #[test]
    fn zero_amounts_equal_tone_only() {
        let src: Vec<[f32; 4]> = (0..64 * 64).map(|i| { let v = (i % 64) as f32 / 63.0; [v, 0.5 * v, 1.0 - v, 1.0] }).collect();
        let tone_only = FilmSettings { stock: Some("golden200".into()), grain: false, halation: false, ..Default::default() };
        let zeroed = FilmSettings { stock: Some("golden200".into()), grain_amount: 0.0, halation_amount: 0.0, ..Default::default() };
        let a = reference::render(&src, 64, 64, &FilmContext::new(tone_only, 64, 25.0, "x").unwrap().frame_params(64, 64, 0, 1.0));
        let b = reference::render(&src, 64, 64, &FilmContext::new(zeroed, 64, 25.0, "x").unwrap().frame_params(64, 64, 0, 1.0));
        assert_eq!(a, b);
    }

    #[test]
    fn halation_only_glows_around_highlights() {
        // Super 8 at a 64 px frame: R = 0.008·64·24/4.01 ≈ 3.1 px, f = 1.
        let s = FilmSettings { stock: Some("tungsten800".into()), format: stocks::FilmFormat::Super8, tone: false, grain: false, halation_amount: 1.0, ..Default::default() };
        let c = FilmContext::new(s, 64, 25.0, "x").unwrap();
        // Nothing near clipping: only the decode/encode round trip may differ.
        let dark = grey(64, 64, 0.5);
        let out = reference::render(&dark, 64, 64, &c.frame_params(64, 64, 0, 1.0));
        assert!(out.iter().zip(&dark).all(|(a, b)| (0..4).all(|i| (a[i] - b[i]).abs() < 1e-5)), "a frame without highlights changed");
        let mut src = grey(64, 64, 0.3);
        for y in 28..36 { for x in 28..36 { src[y * 64 + x] = [1.0, 1.0, 1.0, 1.0]; } }
        let out = reference::render(&src, 64, 64, &c.frame_params(64, 64, 0, 1.0));
        let near = out[32 * 64 + 38];
        assert!(near[0] > 0.3 + 0.01, "no red glow next to the highlight: {near:?}");
        assert!(near[0] - 0.3 > 3.0 * (near[2] - 0.3), "glow is not red-dominant: {near:?}");
        let far = out[2 * 64 + 2];
        assert!((far[0] - 0.3).abs() < 1e-4, "glow reached the far corner: {far:?}");
    }

    /// Review focus 1: a halo grid whose last block hangs off the frame. R = 0.008·D·24/gate·p
    /// cannot exceed 8 px on a frame this small (`D·p` is the buffer's short side), so the
    /// grid is forced to f = 4; the arithmetic is the same at any size.
    #[test]
    fn halo_grid_on_an_odd_frame_reads_inside_the_frame() {
        let s = FilmSettings { stock: Some("golden200".into()), tone: false, grain: false, ..Default::default() };
        let mut p = FilmContext::new(s, 45, 25.0, "x").unwrap().frame_params(67, 45, 0, 1.0);
        force_halo_grid(&mut p, 4, 67, 45);
        assert_eq!((p.halo_w, p.halo_h), (17, 12));
        let mut src = grey(67, 45, 0.3);
        for y in 40..45 { for x in 64..67 { src[y * 67 + x] = [1.0; 4]; } } // highlight in the clipped corner block
        let out = reference::render(&src, 67, 45, &p);
        assert!(out[44 * 67 + 60][0] > 0.3, "the corner block's highlight did not glow");
    }

    #[test]
    fn settings_round_trip_and_clamp() {
        let s = FilmSettings { stock: Some("vivid100".into()), format: stocks::FilmFormat::Super16, tone: false, grain_amount: 1.5, ..Default::default() };
        let mut t = FilmSettings::default();
        t.apply_json(&s.to_json());
        assert_eq!(t, s);
        let mut u = FilmSettings::default();
        u.apply_json(&serde_json::json!({ "stock": "bogus", "grain_amount": 9.0, "halation_amount": -1.0, "format": "70mm" }));
        assert_eq!(u.stock, None);
        assert_eq!(u.grain_amount, 2.0);
        assert_eq!(u.halation_amount, 0.0);
        assert_eq!(u.format, stocks::FilmFormat::Mm35);
        assert!(super::supported::<crate::stabilization::RGBA8>() && super::supported::<crate::stabilization::RGBA16>());
        assert!(!super::supported::<crate::stabilization::BGRA8>() && !super::supported::<crate::stabilization::RGBAf>() && !super::supported::<crate::stabilization::Luma16>());
    }
}
