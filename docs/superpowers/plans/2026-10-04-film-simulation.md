# Film Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-develop the graded output as film (per-stock tone and colour, frame-varying silver-halide grain, halation) as a GPU chain after chroma smoothing, on OpenCL and wgpu, with a Film panel, project persistence and CLI flags.

**Architecture:** `gyroflow-core` gets a `film` module: a line-for-line Rust port of the Photo360 handover (stocks, curve fit, PCG noise, crystal grain model, halation helpers), a `#[repr(C)]` `FilmParams` uniform block, a `FilmContext` that turns settings into per-frame params, and a pure-Rust reference renderer. Each GPU backend gets a lazily built film chain (halo source, halo blur X/Y, noise+blur X, apply) that runs inside `undistort_image` after the chroma pass and before readback; both are tested against the reference. The export closure and the preview attach `FrameTransform::film` only for graded packed RGBA output.

**Tech Stack:** Rust (gyroflow-core, gyroflow app), OpenCL C (`ocl` 0.19), WGSL (`wgpu` 30), QML, node (test vectors from Photo360 `film.js`).

**Spec:** `docs/superpowers/specs/2026-10-04-film-simulation-design.md` (maths authority: `docs/superpowers/specs/2026-10-04-film-handover.md`).

## Global Constraints

- Pipeline: main kernel → chroma smoothing → film chain → readback. Film never runs before denoise/chroma.
- Gate: only when the grade is baked into the render (export: `bake_color`; preview: preview colour on and `ColorAdjustments::is_active()`), never for `OutputColorSpace::Rec2020Pq`/`Rec2020Hlg`, only on packed `RGBA8`/`RGBA16` planes with `BufferSource::Cpu` output. Anything else: chain skipped, output byte-identical.
- Stock `None` (Off) or all three toggles off → `FilmContext` is `None` → byte-identical output.
- Controls: stock (Off + 8 handover stocks, ids verbatim), format `s8`/`s16`/`35`/`120` (gate short sides 4.01/7.41/24/56 mm, default `35`), toggles `tone`/`grain`/`halation` (default all true), `grain_amount` 0..2 (default 1.0), `halation_amount` 0..1 (default 0.5).
- Scale: `D` = export output short side; `scale = q((D/3060)·(4.01/gate))`; preview factor `p = min(1, buffer_short/D)`; preview grain model gets a second `for_render_scale(q(p))`; halation `R = 0.008·D·24/gate·p`; noise hashed at export pixel `floor((x+0.5)/p)`.
- Seeds: `seed = FNV-1a(UTF-16 code units of the video file name)`, `layerSeed[l] = hash4(seed, frame_index, l, 0x51)` (Wigglecam `FilmPipeline.layerSeed`; frame 0 equals Photo360's `layerSeeds`). `frame_index = round(source_timestamp_us · fps / 1e6)`, timestamp taken before `frame_offset` is applied.
- Precision: intermediates f32; GPU within 1 code value of the quantised reference (8-bit and 16-bit).
- A failed film build (either backend) logs a warning, records the reason in `film::last_skip()`, and the frame renders without film; the backend survives.
- Never push. Commit locally on `Dev-sony`. Line endings differ per file: detect per file (Python byte count); never `sed -i` or PowerShell `Set-Content` on repo files. New files: LF.

## Build and test commands (Windows, this machine)

Every cargo/just command needs the MSVC environment. In PowerShell:

```powershell
$vc = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vc`" >nul 2>nul && set" | % { if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($matches[1])" $matches[2] } }
```

- Core unit tests: `cd src/core; cargo test --lib film`
- GPU tests (need a GPU and the OpenCL link path; never set this RUSTFLAGS for app builds):
  `$env:RUSTFLAGS = "-L native=C:/Users/Leo/Gits/gyroflow/ext/vcpkg/installed/x64-windows/lib"; cd src/core; cargo test --lib --features use-opencl -- --ignored film_gpu --nocapture --test-threads 1`
- Full core suite: `cd src/core; cargo test --lib --features use-opencl` (with the RUSTFLAGS above), then `$env:RUSTFLAGS = $null`.
- App: `$env:RUSTFLAGS = $null; just build`, `just test` (repo root). A running `gyroflow.exe` locks the build: ask the user before closing it.
- Test vectors: `node _scripts/tests/film_vectors.js C:/Users/Leo/Gits/Photo360/film.js > src/core/film/test_vectors.json`

## Review Focus

1. **Odd and non-multiple sizes for the halo grid** (W=1921, f=4 → `halo_w = 481`, last block partly outside): block reads must clamp to the frame, never read past it. Pinned in Task 2 (reference) and Tasks 4/5 (GPU at 67×45 with f=4).
2. **16-bit output near full scale**: grain plus halation can push a channel above 1.0; it must clamp before scaling or an `Rgba16Uint` target wraps (the chroma bug). Pinned in Tasks 4/5 (bright 16-bit frame, assert no channel below 60000 where input was 65535 and tone off).
3. **Frame-to-frame motion vs determinism**: frame N and N+1 differ, frame N twice is identical, preview and export use the same hash lattice at p=1. Pinned in Task 2.
4. **Settings change while a GPU wrapper is cached**: film buffers are allocated lazily inside the wrapper at full output size, so toggling film or changing stock/format/halation never needs a new wrapper and never reads a stale size. Halo textures (wgpu) are recreated when `(halo_w, halo_h)` changes. Pinned in Task 5 (two frames with different halation radius through one wrapper).
5. **Gate holes**: film must not run on ungraded log footage, PQ/HLG output, planar/YUV planes, BGRA, float formats, or non-CPU outputs. Pinned in Task 3 (`film_context` gate tests) and Task 6 (`film_plane_ok` tests); non-CPU outputs are enforced by construction in Tasks 4/5 (film source only prepared for `BufferSource::Cpu`).

---

### Task 1: Film maths (port of the handover)

**Files:**
- Create: `src/core/film/mod.rs`, `src/core/film/stocks.rs`, `src/core/film/curve.rs`, `src/core/film/noise.rs`, `src/core/film/grain.rs`, `src/core/film/test_vectors.json` (generated), `_scripts/tests/film_vectors.js`
- Modify: `src/core/lib.rs` (add `pub mod film;` after `pub mod denoise;`)

**Interfaces:**
- Produces:
  - `film::stocks::{Layer, GrainSpec, Halation, Stock, STOCKS: [Stock; 8], stock(id: &str) -> Option<&'static Stock>, FilmFormat { Super8, Super16, Mm35, Medium120 }}` with `FilmFormat::{id(), from_id(&str) -> Option<Self>, gate_short_mm() -> f64, label(), ALL}`
  - `film::curve::{srgb_decode(f64)->f64, srgb_encode(f64)->f64, develop(log_e: f64, l: &Layer) -> f64, Fit { b, s, u_min_a }, fit_curve(&Stock) -> Fit, curve_pixel(lin: [f64;3], &Stock, &Fit) -> [f64;3], MID_OUT, WHITE_OUT}`
  - `film::noise::{pcg(u32)->u32, hash4(u32,u32,u32,u32)->u32, gaussian(u32)->f64, gaussian_f32(u32)->f32, fnv1a_utf16(&str)->u32, layer_seeds(seed: u32, frame: u32) -> [u32; 3], NOISE_SALT: u32 = 0x3C6EF372}`
  - `film::grain::{inverse_normal, i0e, lens_area, GrainModel, gaussian_kernel(sigma) -> Vec<f64>, kernel_lag_one, calibrate_noise_sigma, quantize_scale, GrainSetup { sigma: f64, radius: usize, norm: f64, std: [f32; 201], lag_one: f64 }, grain_setup(&GrainSpec, scale_export: f64, z: f64) -> Arc<GrainSetup>, STD_TABLE: usize = 200, MAX_KERNEL_RADIUS: usize = 31}`
  - `film::halo_factor(r: f64) -> u32`, `film::halation_radius(d: f64, gate_mm: f64) -> f64`, `film::REF_SHORT_PX = 3060.0`, `film::REF_SHORT_MM = 4.01`

- [ ] **Step 1: Write the vector generator and generate the fixture**

`_scripts/tests/film_vectors.js`:

```js
// Reference numbers for gyroflow-core's film port, computed by Photo360's film.js itself.
// Usage: node _scripts/tests/film_vectors.js <path to Photo360/film.js> > src/core/film/test_vectors.json
'use strict';
const Film = require(process.argv[2] || 'C:/Users/Leo/Gits/Photo360/film.js');
const u = (v) => v >>> 0;
const pcgIn = [0, 1, 2, 3, 0x51, 123456789, 0x7fffffff, 0x80000000 | 0, -1, 0x3C6EF372];
const hashIn = [[0, 0, 0, 0], [1, 2, 3, 4], [-5, 7, 0x51, 0x3C6EF372], [4000, 3000, 12345, 0x3C6EF372]];
const golden = Film.STOCKS.find((s) => s.id === 'golden200');
const tungsten = Film.STOCKS.find((s) => s.id === 'tungsten800');
const model = (st, s) => Film.grainModel(st.grain.radiusMedianPx, st.grain.radiusLogSigma, st.grain.filterSigmaPx).forRenderScale(s);
const setup = (st, s, z) => { const g = Film.grainSetup(st.grain, s, z); return { sigma: g.sigma, radius: g.radius, norm: g.norm, std50: g.std[50], std100: g.std[100], std180: g.std[180] }; };
const out = {
  pcg: pcgIn.map((v) => [u(v), u(Film.pcg(v))]),
  hash4: hashIn.map((a) => [a.map(u), u(Film.hash4(...a))]),
  gaussian: [0, 1, 99, 0x3C6EF372, 0xdeadbeef | 0].map((h) => [u(h), Film.gaussian(h)]),
  fnv: ['', 'C0001.MP4', 'clip ä.mov'].map((s) => [s, u(Film.seedOf(s))]),
  layer_seeds: [[u(Film.seedOf('C0001.MP4')), Film.layerSeeds(Film.seedOf('C0001.MP4')).map(u)]],
  inverse_normal: [0.001, 0.02, 0.3, 0.5, 0.9, 0.99].map((p) => [p, Film.inverseNormal(p)]),
  i0e: [0, 0.5, 3.75, 4, 20].map((z) => [z, Film.i0e(z)]),
  model: [[golden, 1], [golden, 0.278], [tungsten, 1.667]].map(([st, s]) => {
    const m = model(st, s);
    return { id: st.id, scale: s, var10: m.filteredVariance(0.1), var50: m.filteredVariance(0.5), var90: m.filteredVariance(0.9), lag1: m.lagOneCorrelation() };
  }),
  kernel_lag_one: [0.5, 1, 2.5].map((s) => [s, Film.kernelLagOne(s)]),
  setup: [
    { id: 'golden200', scale: Film.quantizeScale(0.5), z: 1, ...setup(golden, Film.quantizeScale(0.5), 1) },
    { id: 'golden200', scale: Film.quantizeScale(0.5), z: Film.quantizeScale(0.3), ...setup(golden, Film.quantizeScale(0.5), Film.quantizeScale(0.3)) },
    { id: 'tungsten800', scale: Film.quantizeScale(1.2), z: 1, ...setup(tungsten, Film.quantizeScale(1.2), 1) },
  ],
  quantize: [0.1, 0.278, 0.5, 1, 1.667].map((v) => [v, Film.quantizeScale(v)]),
  halo_factor: [3, 8, 9, 41, 244, 2000].map((r) => [r, Film.haloFactor(r)]),
};
process.stdout.write(JSON.stringify(out, null, 1) + '\n');
```

Run: `node _scripts/tests/film_vectors.js C:/Users/Leo/Gits/Photo360/film.js > src/core/film/test_vectors.json`
Expected: a JSON file with keys `pcg`, `hash4`, … `halo_factor`; no error. (If `Film.haloFactor` is undefined, film.js exports it under `haloFactor` per its `return { … }` block; check before continuing.)

- [ ] **Step 2: Write the failing tests**

`src/core/film/mod.rs` (module skeleton + tests; the submodules are empty files at this step):

```rust
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
```

Add `pub mod film;` to `src/core/lib.rs` after `pub mod denoise;`. Create the four submodule files empty.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd src/core; cargo test --lib film::tests`
Expected: compile FAIL — `cannot find function pcg in module noise` (and the other missing items).

- [ ] **Step 4: Implement `stocks.rs`**

```rust
// Copyright © 2026 Gyroflow

//! The eight stocks of the handover, verbatim (Photo360 `film.js` STOCKS), and the film formats.

#[derive(Clone, Copy, Debug)]
pub struct Layer { pub gamma: f64, pub log_e0: f64, pub u_min: f64, pub u_max: f64 }
#[derive(Clone, Copy, Debug)]
pub struct GrainSpec { pub radius_median_px: f64, pub radius_log_sigma: f64, pub filter_sigma_px: f64 }
#[derive(Clone, Copy, Debug)]
pub struct Halation { pub gain: f64, pub tint: [f64; 3] }
#[derive(Clone, Copy, Debug)]
pub struct Stock {
    pub id: &'static str,
    pub name: &'static str,
    pub bw: bool,
    /// 3x3 row-major: linear rgb -> layer exposure.
    pub sensitivity: [f64; 9],
    pub layers: [Layer; 3],
    pub grain: GrainSpec,
    /// 3x3 row-major: developed layers -> output rgb.
    pub crosstalk: [f64; 9],
    pub halation: Halation,
}

const fn l(gamma: f64, log_e0: f64, u_min: f64, u_max: f64) -> Layer { Layer { gamma, log_e0, u_min, u_max } }
const fn g(radius_median_px: f64, radius_log_sigma: f64) -> GrainSpec { GrainSpec { radius_median_px, radius_log_sigma, filter_sigma_px: 0.8 } }
const HALO_STD: Halation = Halation { gain: 0.6, tint: [1.0, 0.28, 0.05] };

pub const STOCKS: [Stock; 8] = [
    Stock { id: "portrait160", name: "Portrait 160", bw: false,
        sensitivity: [1.02, -0.01, -0.01, 0.00, 1.00, 0.00, -0.01, -0.02, 0.96],
        layers: [l(0.50, -0.82, 0.040, 0.960), l(0.50, -0.82, 0.040, 0.960), l(0.49, -0.80, 0.045, 0.955)],
        grain: g(0.09, 0.30), crosstalk: [1.04, -0.03, -0.01, -0.02, 1.04, -0.02, -0.01, -0.03, 1.04], halation: HALO_STD },
    Stock { id: "golden200", name: "Golden 200", bw: false,
        sensitivity: [1.06, -0.02, -0.04, -0.01, 1.00, 0.01, -0.03, -0.05, 0.92],
        layers: [l(0.62, -0.80, 0.035, 0.965), l(0.62, -0.79, 0.030, 0.965), l(0.60, -0.76, 0.045, 0.950)],
        grain: g(0.12, 0.35), crosstalk: [1.12, -0.08, -0.04, -0.06, 1.10, -0.04, -0.03, -0.09, 1.12], halation: HALO_STD },
    Stock { id: "vivid100", name: "Vivid 100", bw: false,
        sensitivity: [1.08, -0.05, -0.03, -0.04, 1.08, -0.04, -0.03, -0.06, 1.09],
        layers: [l(0.70, -0.80, 0.030, 0.970), l(0.70, -0.79, 0.028, 0.970), l(0.68, -0.78, 0.035, 0.960)],
        grain: g(0.07, 0.30), crosstalk: [1.16, -0.10, -0.06, -0.07, 1.15, -0.08, -0.04, -0.11, 1.15], halation: HALO_STD },
    Stock { id: "pastel400", name: "Pastel 400", bw: false,
        sensitivity: [0.96, 0.02, 0.02, 0.00, 0.99, 0.05, 0.02, 0.04, 0.98],
        layers: [l(0.41, -0.90, 0.085, 0.952), l(0.41, -0.92, 0.090, 0.952), l(0.40, -0.88, 0.095, 0.948)],
        grain: g(0.15, 0.35), crosstalk: [0.95, 0.03, 0.02, 0.03, 0.95, 0.02, 0.02, 0.03, 0.95], halation: HALO_STD },
    Stock { id: "verdant400", name: "Verdant 400", bw: false,
        sensitivity: [0.97, -0.01, -0.01, -0.01, 1.04, -0.01, -0.01, 0.00, 1.04],
        layers: [l(0.60, -0.76, 0.030, 0.960), l(0.58, -0.85, 0.055, 0.960), l(0.60, -0.80, 0.040, 0.955)],
        grain: g(0.16, 0.35), crosstalk: [1.08, -0.05, -0.03, -0.04, 1.08, -0.04, -0.02, -0.06, 1.08], halation: HALO_STD },
    Stock { id: "tungsten800", name: "Tungsten 800", bw: false,
        sensitivity: [0.80, -0.01, -0.01, -0.02, 1.00, 0.02, 0.00, 0.05, 1.22],
        layers: [l(0.58, -0.82, 0.045, 0.955), l(0.58, -0.80, 0.040, 0.955), l(0.56, -0.78, 0.050, 0.950)],
        grain: g(0.20, 0.40), crosstalk: [1.10, -0.06, -0.04, -0.05, 1.09, -0.04, -0.03, -0.08, 1.11],
        halation: Halation { gain: 1.8, tint: [1.0, 0.15, 0.02] } },
    Stock { id: "classic400bw", name: "Classic 400 B&W", bw: true,
        sensitivity: [0.30, 0.59, 0.11, 0.30, 0.59, 0.11, 0.30, 0.59, 0.11],
        layers: [l(0.78, -0.74, 0.020, 0.980), l(0.78, -0.74, 0.020, 0.980), l(0.78, -0.74, 0.020, 0.980)],
        grain: g(0.22, 0.40), crosstalk: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], halation: HALO_STD },
    Stock { id: "smooth400bw", name: "Smooth 400 B&W", bw: true,
        sensitivity: [0.25, 0.62, 0.13, 0.25, 0.62, 0.13, 0.25, 0.62, 0.13],
        layers: [l(0.56, -0.78, 0.040, 0.960), l(0.56, -0.78, 0.040, 0.960), l(0.56, -0.78, 0.040, 0.960)],
        grain: g(0.13, 0.30), crosstalk: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], halation: HALO_STD },
];

pub fn stock(id: &str) -> Option<&'static Stock> { STOCKS.iter().find(|s| s.id == id) }

/// The film gauge the frame stands for: a smaller gate means coarser grain and wider halation
/// at the same output size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FilmFormat { Super8, Super16, #[default] Mm35, Medium120 }

impl FilmFormat {
    pub const ALL: [FilmFormat; 4] = [Self::Super8, Self::Super16, Self::Mm35, Self::Medium120];
    /// Stable key for the project file, CLI and QML. Never localise.
    pub fn id(self) -> &'static str { match self { Self::Super8 => "s8", Self::Super16 => "s16", Self::Mm35 => "35", Self::Medium120 => "120" } }
    pub fn from_id(id: &str) -> Option<Self> { Self::ALL.into_iter().find(|f| f.id() == id) }
    pub fn label(self) -> &'static str { match self { Self::Super8 => "Super 8", Self::Super16 => "Super 16", Self::Mm35 => "35 mm", Self::Medium120 => "120 (6×6)" } }
    pub fn gate_short_mm(self) -> f64 { match self { Self::Super8 => 4.01, Self::Super16 => 7.41, Self::Mm35 => 24.0, Self::Medium120 => 56.0 } }
}
```

- [ ] **Step 5: Implement `curve.rs`**

```rust
// Copyright © 2026 Gyroflow

//! Wigglecam's characteristic curve and the per-stock re-centring fit (handover "Tone curves").

use super::stocks::{ Layer, Stock };

pub fn srgb_decode(v: f64) -> f64 { if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) } }
pub fn srgb_encode(x: f64) -> f64 { if x <= 0.0031308 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 } }

pub const MID: f64 = 0.18;
/// `srgb_encode(0.18)`: mid-grey stays put.
pub const MID_OUT: f64 = 0.46135612950044164;
pub const WHITE_OUT: f64 = 0.96;

/// Developed fraction (display lightness) for log10 exposure: a logistic with slope `gamma` at `log_e0`.
pub fn develop(log_e: f64, l: &Layer) -> f64 {
    let range = l.u_max - l.u_min;
    let k = 4.0 * l.gamma / range;
    l.u_min + range / (1.0 + (-k * (log_e - l.log_e0)).exp())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fit { pub b: f64, pub s: f64, pub u_min_a: f64 }

/// One exposure shift `b` and stretch `s`, fitted on the anchor layer (green; layer 0 for B&W)
/// so linear 0.18 -> `MID_OUT` and 1.0 -> `WHITE_OUT`. 80-step bisection: the mid rises with b.
pub fn fit_curve(stock: &Stock) -> Fit {
    let l = &stock.layers[if stock.bw { 0 } else { 1 }];
    let at = |e: f64, b: f64| develop(e.log10() + b, l);
    let mid_for = |b: f64| {
        let s = (WHITE_OUT - l.u_min) / (at(1.0, b) - l.u_min);
        (s, l.u_min + s * (at(MID, b) - l.u_min))
    };
    let (mut lo, mut hi) = (-4.0f64, 4.0f64);
    for _ in 0..80 {
        let m = (lo + hi) / 2.0;
        if mid_for(m).1 < MID_OUT { lo = m; } else { hi = m; }
    }
    let b = (lo + hi) / 2.0;
    Fit { b, s: mid_for(b).0, u_min_a: l.u_min }
}

/// Linear rgb -> the three developed layers (display values, before grain).
pub fn curve_pixel(lin: [f64; 3], stock: &Stock, fit: &Fit) -> [f64; 3] {
    let m = &stock.sensitivity;
    let mut out = [0.0; 3];
    for i in 0..3 {
        let e = m[i * 3] * lin[0] + m[i * 3 + 1] * lin[1] + m[i * 3 + 2] * lin[2];
        let u = develop(e.max(1e-5).log10() + fit.b, &stock.layers[i]);
        out[i] = fit.u_min_a + fit.s * (u - fit.u_min_a);
    }
    out
}
```

- [ ] **Step 6: Implement `noise.rs`**

```rust
// Copyright © 2026 Gyroflow

//! Wigglecam's PCG hash and Box-Muller, integer-exact on CPU and GPU (handover "Noise").

/// Salt of the per-pixel noise hash: `hash4(x, y, layer_seed, NOISE_SALT)`.
pub const NOISE_SALT: u32 = 0x3C6E_F372;

pub fn pcg(v: u32) -> u32 {
    let state = v.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}
pub fn hash4(a: u32, b: u32, c: u32, d: u32) -> u32 { pcg(a ^ pcg(b ^ pcg(c ^ pcg(d)))) }

const U24: f64 = 1.0 / 16777216.0;
/// Standard normal from a hash, from two 24-bit uniforms (exact in fp32).
pub fn gaussian(h: u32) -> f64 {
    let h1 = pcg(h);
    let h2 = pcg(h1);
    let u1 = ((h1 >> 8) as f64 * U24).max(U24);
    let u2 = (h2 >> 8) as f64 * U24;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
/// `gaussian` in the precision the GPU kernels compute it, for the reference renderer.
pub fn gaussian_f32(h: u32) -> f32 {
    let h1 = pcg(h);
    let h2 = pcg(h1);
    let u1 = ((h1 >> 8) as f32 / 16777216.0).max(1.0 / 16777216.0);
    let u2 = (h2 >> 8) as f32 / 16777216.0;
    (-2.0 * u1.ln()).sqrt() * (6.283185307179586f32 * u2).cos()
}

/// FNV-1a over UTF-16 code units, as JS `charCodeAt` sees a string: the clip's grain seed.
pub fn fnv1a_utf16(s: &str) -> u32 {
    s.encode_utf16().fold(0x811c_9dc5u32, |h, c| (h ^ c as u32).wrapping_mul(0x0100_0193))
}

/// Wigglecam `FilmPipeline.layerSeed`: one independent noise field per layer and frame.
pub fn layer_seeds(seed: u32, frame: u32) -> [u32; 3] { [0, 1, 2].map(|l| hash4(seed, frame, l, 0x51)) }
```

- [ ] **Step 7: Implement `grain.rs`**

```rust
// Copyright © 2026 Gyroflow

//! Wigglecam's silver-halide grain model and its calibration to blurred noise (handover "Grain").

use std::sync::Arc;
use super::stocks::GrainSpec;

const MAX_U: f64 = 0.995;
const QUANTILES: usize = 32;
pub const MAX_KERNEL_RADIUS: usize = 31;
pub const STD_TABLE: usize = 200;

/// Inverse standard normal CDF (Acklam, |error| < 1.2e-9).
pub fn inverse_normal(p: f64) -> f64 {
    const A: [f64; 6] = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const B: [f64; 5] = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const C: [f64; 6] = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const D: [f64; 4] = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const LOW: f64 = 0.02425;
    let tail = |q: f64| (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    if p < LOW { return tail((-2.0 * p.ln()).sqrt()); }
    if p <= 1.0 - LOW {
        let q = p - 0.5;
        let r = q * q;
        return (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0);
    }
    -tail((-2.0 * (1.0 - p).ln()).sqrt())
}

/// e^-z I0(z), z >= 0 (Abramowitz & Stegun 9.8.1-9.8.2).
pub fn i0e(z: f64) -> f64 {
    if z <= 3.75 {
        let t = (z / 3.75) * (z / 3.75);
        let i0 = 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
        return i0 * (-z).exp();
    }
    let t = 3.75 / z;
    (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))))) / z.sqrt()
}

/// Overlap area of two disks of radius r whose centres are d apart.
pub fn lens_area(r: f64, d: f64) -> f64 {
    if d >= 2.0 * r { return 0.0; }
    2.0 * r * r * (d / (2.0 * r)).acos() - d / 2.0 * (4.0 * r * r - d * d).sqrt()
}

/// Boolean model of disks with log-normal radii at Poisson centres, seen through a Gaussian
/// filter. Lengths in render pixels.
#[derive(Clone, Debug)]
pub struct GrainModel { radius_median: f64, radius_log_sigma: f64, filter_sigma: f64, mean_area: f64, radii: Vec<f64>, r_max: f64 }

impl GrainModel {
    pub fn new(radius_median: f64, radius_log_sigma: f64, filter_sigma: f64) -> Self {
        let mean_area = std::f64::consts::PI * radius_median * radius_median * (2.0 * radius_log_sigma * radius_log_sigma).exp();
        let radii: Vec<f64> = (0..QUANTILES).map(|i| radius_median * (radius_log_sigma * inverse_normal((i as f64 + 0.5) / QUANTILES as f64)).exp()).collect();
        let r_max = radii.iter().cloned().fold(f64::MIN, f64::max);
        Self { radius_median, radius_log_sigma, filter_sigma, mean_area, radii, r_max }
    }
    fn lambda(&self, u: f64) -> f64 { -(1.0 - u.clamp(0.0, MAX_U)).ln() / self.mean_area }
    fn covariance(&self, u: f64, d: f64) -> f64 {
        if u <= 0.0 { return 0.0; }
        let q = 1.0 - u.clamp(0.0, MAX_U);
        let overlap = self.radii.iter().map(|&r| lens_area(r, d)).sum::<f64>() / self.radii.len() as f64;
        q * q * ((self.lambda(u) * overlap).exp() - 1.0)
    }
    /// Covariance after the Gaussian filter at offset `delta` (Simpson, 400 steps on [0, 2 r_max]).
    pub fn filtered_covariance(&self, u: f64, delta: f64) -> f64 {
        if u <= 0.0 { return 0.0; }
        let (upper, n) = (2.0 * self.r_max, 400);
        let h = upper / n as f64;
        let s2 = self.filter_sigma * self.filter_sigma;
        let f = |r: f64| self.covariance(u, r) * (-(r - delta) * (r - delta) / (4.0 * s2)).exp() * i0e(r * delta / (2.0 * s2)) * r / (2.0 * s2);
        let mut sum = f(0.0) + f(upper);
        for i in 1..n { sum += if i % 2 == 1 { 4.0 } else { 2.0 } * f(i as f64 * h); }
        sum * h / 3.0
    }
    pub fn filtered_variance(&self, u: f64) -> f64 { self.filtered_covariance(u, 0.0) }
    pub fn lag_one_correlation(&self, u: f64) -> f64 {
        let v = self.filtered_covariance(u, 0.0);
        if v <= 0.0 { 0.0 } else { self.filtered_covariance(u, 1.0) / v }
    }
    /// The same film rendered `s`x larger (s < 1 adds the pixel-box term 1/12).
    pub fn for_render_scale(&self, s: f64) -> Self {
        Self::new(self.radius_median * s, self.radius_log_sigma,
            (self.filter_sigma * self.filter_sigma * s * s + (1.0 - s * s).max(0.0) / 12.0).sqrt())
    }
}

pub fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let r = ((3.0 * sigma).ceil() as usize).clamp(1, MAX_KERNEL_RADIUS);
    let mut k: Vec<f64> = (0..2 * r + 1).map(|i| { let d = i as f64 - r as f64; (-(d * d) / (2.0 * sigma * sigma)).exp() }).collect();
    let sum: f64 = k.iter().sum();
    k.iter_mut().for_each(|w| *w /= sum);
    k
}
/// Lag-1 autocorrelation of white noise after this separable kernel.
pub fn kernel_lag_one(sigma: f64) -> f64 {
    let k = gaussian_kernel(sigma);
    let den: f64 = k.iter().map(|w| w * w).sum();
    let num: f64 = k.windows(2).map(|w| w[0] * w[1]).sum();
    num / den
}
/// Sigma whose blurred noise has the target lag-1 correlation (60-step bisection).
pub fn calibrate_noise_sigma(target: f64) -> f64 {
    let (mut lo, mut hi) = (0.05f64, 20.0f64);
    for _ in 0..60 {
        let m = (lo + hi) / 2.0;
        if kernel_lag_one(m) < target { lo = m; } else { hi = m; }
    }
    (lo + hi) / 2.0
}
/// Scales are cached at 1/8-octave steps.
pub fn quantize_scale(v: f64) -> f64 { 2f64.powf((8.0 * v.log2()).round() / 8.0) }

#[derive(Clone, Debug)]
pub struct GrainSetup {
    pub sigma: f64,
    pub radius: usize,
    /// `1 / Σw²` of the 1-D kernel: restores unit std after the separable blur.
    pub norm: f64,
    pub std: [f32; STD_TABLE + 1],
    /// The model's lag-1 correlation at u = 0.5, for the statistics tests.
    pub lag_one: f64,
}

/// Everything a render needs for one stock's grain at export scale `scale_export`, seen at
/// `z` render px per export px (1 for the export). Tens of ms to build, so cached.
pub fn grain_setup(spec: &GrainSpec, scale_export: f64, z: f64) -> Arc<GrainSetup> {
    type Key = [u64; 5];
    static CACHE: std::sync::OnceLock<parking_lot::Mutex<Vec<(Key, Arc<GrainSetup>)>>> = std::sync::OnceLock::new();
    let key: Key = [spec.radius_median_px.to_bits(), spec.radius_log_sigma.to_bits(), spec.filter_sigma_px.to_bits(), scale_export.to_bits(), z.to_bits()];
    let cache = CACHE.get_or_init(Default::default);
    if let Some((_, s)) = cache.lock().iter().find(|(k, _)| *k == key) { return s.clone(); }

    let mut model = GrainModel::new(spec.radius_median_px, spec.radius_log_sigma, spec.filter_sigma_px).for_render_scale(scale_export);
    if z != 1.0 { model = model.for_render_scale(z); }
    let lag_one = model.lag_one_correlation(0.5);
    let sigma = calibrate_noise_sigma(lag_one);
    let k = gaussian_kernel(sigma);
    let norm = 1.0 / k.iter().map(|w| w * w).sum::<f64>();
    let mut std = [0f32; STD_TABLE + 1];
    for (i, s) in std.iter_mut().enumerate() { *s = model.filtered_variance(i as f64 / STD_TABLE as f64).max(0.0).sqrt() as f32; }
    let setup = Arc::new(GrainSetup { sigma, radius: (k.len() - 1) / 2, norm, std, lag_one });

    let mut c = cache.lock();
    if c.len() > 16 { c.remove(0); }
    c.push((key, setup.clone()));
    setup
}
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `cd src/core; cargo test --lib film::tests`
Expected: 7 passed. If `curve_fit_matches_the_handover_table` fails by a rounding hair on one value, print the actual value: the table has 3 decimals, tolerance 0.0006 covers rounding; a larger miss is a port bug.

- [ ] **Step 9: Commit**

```bash
git add _scripts/tests/film_vectors.js src/core/film src/core/lib.rs
git commit -m "feat(film): port the film maths (stocks, curve fit, PCG noise, grain model)"
```

---

### Task 2: `FilmParams`, `FilmContext` and the reference renderer

**Files:**
- Create: `src/core/film/params.rs`, `src/core/film/reference.rs`
- Modify: `src/core/film/mod.rs` (add modules, `FilmSettings`, re-exports, tests)

**Interfaces:**
- Consumes: everything Task 1 produces.
- Produces:
  - `film::FilmSettings { stock: Option<String>, format: FilmFormat, tone: bool, grain: bool, halation: bool, grain_amount: f64, halation_amount: f64 }` with `Default` (stock None, Mm35, all true, 1.0, 0.5), `set_param(name, f64) -> bool` (`grain_amount` 0..2, `halation_amount` 0..1), `set_flag(name, bool) -> bool` (`tone`/`grain`/`halation`), `to_json() -> serde_json::Value`, `apply_json(&serde_json::Value)`.
  - `film::params::{FilmParams (#[repr(C)], Pod, 1296 bytes), FLAG_TONE=1, FLAG_GRAIN=2, FLAG_HALATION=4, FLAG_BW=8}`, `FilmParams::is_active(&self) -> bool` (`flags != 0`)
  - `film::FilmContext { settings: FilmSettings, export_short: usize, fps: f64, seed: u32 }` with `FilmContext::new(settings, export_short, fps, file_name: &str) -> Option<Self>` (None when inactive) and `frame_params(&self, buf_w: usize, buf_h: usize, source_timestamp_us: i64, max_value: f32) -> FilmParams`
  - `film::reference::render(src: &[[f32; 4]], w: usize, h: usize, p: &FilmParams) -> Vec<[f32; 4]>` (values 0..1, `p.max_value` ignored)
  - `film::supported::<T: PixelType>() -> bool` (RGBA8 or RGBA16 only)
  - `film::set_last_skip(String)`, `film::last_skip() -> Option<String>`

- [ ] **Step 1: Write the failing tests** (append to `mod tests` in `film/mod.rs`)

```rust
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
            let want = p.std_table[(0.46f32 * 200.0 + 0.5) as usize] as f64;
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
```

Run: `cd src/core; cargo test --lib film::tests`
Expected: compile FAIL — `FilmContext`, `FilmSettings`, `params`, `reference` not found.

- [ ] **Step 2: Implement `params.rs`**

```rust
// Copyright © 2026 Gyroflow

//! The film chain's per-frame uniform block. Mirrored field for field by `FilmParams` in
//! `gpu/opencl_film.cl` and `gpu/wgpu_film.wgsl`: 80 bytes of scalars, then 16-byte rows.

pub const FLAG_TONE: u32 = 1;
pub const FLAG_GRAIN: u32 = 2;
pub const FLAG_HALATION: u32 = 4;
pub const FLAG_BW: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FilmParams {
    /// Filled by the backend from the buffers it is given.
    pub width: i32, pub height: i32, pub stride: i32, pub flags: u32,
    pub halo_f: i32, pub halo_w: i32, pub halo_h: i32, pub halo_radius: i32,
    pub grain_radius: i32, pub max_value: f32, pub grain_k: f32, pub grain_sigma: f32,
    pub halo_sigma: f32, pub halo_strength: f32, pub inv_p: f32, pub fit_b: f32,
    pub fit_s: f32, pub fit_umin_a: f32, pub _pad0: f32, pub _pad1: f32,
    pub seeds: [u32; 4],
    pub halo_tint: [f32; 4],
    pub log_e0: [f32; 4],
    pub k: [f32; 4],
    pub u_min: [f32; 4],
    pub range: [f32; 4],
    pub sens: [[f32; 4]; 3],
    pub cross: [[f32; 4]; 3],
    /// `std(u)` at `u = i / 200`, i in 0..=200; the rest is padding (WGSL wants vec4 rows).
    pub std_table: [f32; 256],
}

impl Default for FilmParams { fn default() -> Self { bytemuck::Zeroable::zeroed() } }

impl FilmParams {
    pub fn is_active(&self) -> bool { self.flags & (FLAG_TONE | FLAG_GRAIN | FLAG_HALATION) != 0 }
}
```

- [ ] **Step 3: Implement `FilmSettings`, `FilmContext`, `supported`, `last_skip` in `film/mod.rs`**

Add after the constants:

```rust
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
```

(If `parking_lot::Mutex::new` is not `const` in this parking_lot version, use `std::sync::Mutex` with `.lock().unwrap()` — record a ruling.)

- [ ] **Step 4: Implement `reference.rs`**

```rust
// Copyright © 2026 Gyroflow

//! What the GPU film chain does, in f32, pass by pass. Both kernels are tested against this.
//! Values are 0..1; `max_value` is ignored here.

use super::params::*;
use super::noise::{ gaussian_f32, hash4, NOISE_SALT };

fn decode(v: f32) -> f32 { if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) } }
fn encode(x: f32) -> f32 { if x <= 0.0031308 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 } }
fn smoothstep(a: f32, b: f32, x: f32) -> f32 { let t = ((x - a) / (b - a)).clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) }
/// In-shader weights: `exp(-k²/2σ²)`, normalised by their sum over [-r, r].
fn weights(r: i32, sigma: f32) -> Vec<f32> {
    let w: Vec<f32> = (-r..=r).map(|k| (-((k * k) as f32) / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = w.iter().sum();
    w.into_iter().map(|x| x / s).collect()
}
/// The export pixel under render pixel `x`.
fn export_px(x: i32, inv_p: f32) -> u32 { ((x as f32 + 0.5) * inv_p).floor() as i32 as u32 }

/// Pass G1 then the vertical half of G2: blurred unit noise per layer.
fn noise_field(w: usize, h: usize, p: &FilmParams) -> Vec<[f32; 3]> {
    let layers = if p.flags & FLAG_BW != 0 { 1 } else { 3 };
    let r = p.grain_radius;
    let k = weights(r, p.grain_sigma);
    let mut g1 = vec![[0f32; 3]; w * h];
    for y in 0..h { for x in 0..w { for l in 0..layers {
        let mut s = 0.0;
        for i in -r..=r {
            let xx = (x as i32 + i).clamp(0, w as i32 - 1);
            s += k[(i + r) as usize] * gaussian_f32(hash4(export_px(xx, p.inv_p), export_px(y as i32, p.inv_p), p.seeds[l], NOISE_SALT));
        }
        g1[y * w + x][l] = s;
    } } }
    let mut out = vec![[0f32; 3]; w * h];
    for y in 0..h { for x in 0..w { for l in 0..layers {
        let mut s = 0.0;
        for i in -r..=r { s += k[(i + r) as usize] * g1[(y as i32 + i).clamp(0, h as i32 - 1) as usize * w + x][l]; }
        out[y * w + x][l] = s;
    } } }
    out
}

/// Passes H1-H3: the halation field at 1/f resolution.
fn halo_field(src: &[[f32; 4]], w: usize, h: usize, p: &FilmParams) -> Vec<f32> {
    let (f, hw, hh) = (p.halo_f as usize, p.halo_w as usize, p.halo_h as usize);
    let mut a = vec![0f32; hw * hh];
    for j in 0..hh { for i in 0..hw {
        let mut s = 0.0;
        for b in 0..f { for c in 0..f {
            let px = src[(j * f + b).min(h - 1) * w + (i * f + c).min(w - 1)];
            let y = 0.2126 * decode(px[0]) + 0.7152 * decode(px[1]) + 0.0722 * decode(px[2]);
            s += smoothstep(0.75, 1.0, y) * y;
        } }
        a[j * hw + i] = s / (f * f) as f32;
    } }
    let r = p.halo_radius;
    let k = weights(r, p.halo_sigma);
    let mut b = vec![0f32; hw * hh];
    for j in 0..hh { for i in 0..hw {
        b[j * hw + i] = (-r..=r).map(|t| k[(t + r) as usize] * a[j * hw + (i as i32 + t).clamp(0, hw as i32 - 1) as usize]).sum();
    } }
    for j in 0..hh { for i in 0..hw {
        a[j * hw + i] = (-r..=r).map(|t| k[(t + r) as usize] * b[(j as i32 + t).clamp(0, hh as i32 - 1) as usize * hw + i]).sum();
    } }
    a
}

/// Bilinear, clamp-to-edge, at halo-texel coordinates (texel centres at +0.5).
fn sample_halo(field: &[f32], p: &FilmParams, hx: f32, hy: f32) -> f32 {
    let (fx, fy) = (hx - 0.5, hy - 0.5);
    let (x0, y0) = (fx.floor() as i32, fy.floor() as i32);
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let at = |x: i32, y: i32| field[y.clamp(0, p.halo_h - 1) as usize * p.halo_w as usize + x.clamp(0, p.halo_w - 1) as usize];
    (at(x0, y0) * (1.0 - tx) + at(x0 + 1, y0) * tx) * (1.0 - ty) + (at(x0, y0 + 1) * (1.0 - tx) + at(x0 + 1, y0 + 1) * tx) * ty
}

pub fn render(src: &[[f32; 4]], w: usize, h: usize, p: &FilmParams) -> Vec<[f32; 4]> {
    let noise = (p.flags & FLAG_GRAIN != 0).then(|| noise_field(w, h, p));
    let halo = (p.flags & FLAG_HALATION != 0).then(|| halo_field(src, w, h, p));
    let bw = p.flags & FLAG_BW != 0;
    let mut out = vec![[0f32; 4]; w * h];
    for y in 0..h { for x in 0..w {
        let c = src[y * w + x];
        let mut lin = [decode(c[0]), decode(c[1]), decode(c[2])];
        if let Some(field) = &halo {
            let f = p.halo_f as f32;
            let hv = sample_halo(field, p, (x as f32 + 0.5) / f, (y as f32 + 0.5) / f);
            for i in 0..3 { lin[i] += p.halo_strength * hv * p.halo_tint[i]; }
        }
        let mut u = [0f32; 3];
        if p.flags & FLAG_TONE != 0 {
            for i in 0..3 {
                let e = p.sens[i][0] * lin[0] + p.sens[i][1] * lin[1] + p.sens[i][2] * lin[2];
                let log_e = e.max(1e-5).log10() + p.fit_b;
                let d = p.u_min[i] + p.range[i] / (1.0 + (-p.k[i] * (log_e - p.log_e0[i])).exp());
                u[i] = p.fit_umin_a + p.fit_s * (d - p.fit_umin_a);
            }
        } else {
            for i in 0..3 { u[i] = encode(lin[i].max(0.0)); }
        }
        if let Some(n) = &noise {
            let n = n[y * w + x];
            for i in 0..3 {
                let sd = p.std_table[(u[i].clamp(0.0, 1.0) * 200.0 + 0.5) as usize];
                u[i] = (u[i] + p.grain_k * sd * n[if bw { 0 } else { i }]).clamp(0.0, 1.0);
            }
        }
        let o = if p.flags & FLAG_TONE != 0 {
            [0, 1, 2].map(|i| (p.cross[i][0] * u[0] + p.cross[i][1] * u[1] + p.cross[i][2] * u[2]).clamp(0.0, 1.0))
        } else {
            u.map(|v| v.clamp(0.0, 1.0))
        };
        out[y * w + x] = [o[0], o[1], o[2], c[3]];
    } }
    out
}
```

Note `halation_only_glows_around_highlights` first case expects a dark frame unchanged with halation only and tone off: decode→encode round trip in f32 is not bit-exact. If that assertion fails on round-trip error only (diff < 1e-6), change it to a tolerance of `1e-5` and ledger the ruling — the GPU tests compare quantised values anyway.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd src/core; cargo test --lib film::tests`
Expected: 16 passed (7 from Task 1 + 9 new).

- [ ] **Step 6: Commit**

```bash
git add src/core/film
git commit -m "feat(film): FilmParams, FilmContext and the reference renderer"
```

---

### Task 3: Settings on the manager: persistence, gate, CLI

**Files:**
- Modify: `src/core/lib.rs` (field `film`, constructor, clone site near line 1220, save near line 1365, load near line 1960, new methods, tests)
- Modify: `src/cli.rs` (flags near line 106, preset merge near line 204)

**Interfaces:**
- Consumes: `film::{FilmSettings, FilmContext}`.
- Produces:
  - `StabilizationManager::film: Arc<RwLock<film::FilmSettings>>`
  - `StabilizationManager::film_context(&self, graded: bool) -> Option<film::FilmContext>` — `None` unless `graded`, the output colour space is not PQ/HLG, and settings are active.
  - `StabilizationManager::film_preview_graded(&self) -> bool` — `params.preview_color_enabled && color.is_active()`.
  - `StabilizationManager::film_gate_message(&self) -> String` — empty when film would run in the preview; otherwise one sentence.
  - Project key `"film"` (object from `FilmSettings::to_json`).
  - CLI: `--film-stock <id|off>`, `--film-format <s8|s16|35|120>`, `--film-grain <0..2>`, `--film-halation <0..1>`, `--film-no-tone`.

- [ ] **Step 1: Write the failing tests** (in `src/core/lib.rs`, in the same `#[cfg(test)]` module as `color_section_round_trips_without_a_lut_path`; reuse that module's `apply_as_preset` helper and its way of creating a manager)

```rust
    #[test]
    fn film_section_round_trips_and_old_projects_load_with_film_off() {
        let mgr = configured_manager(None);
        *mgr.film.write() = film::FilmSettings { stock: Some("golden200".into()), format: film::stocks::FilmFormat::Super16, grain_amount: 1.4, ..Default::default() };
        let restored = round_trip(&mgr);
        assert_eq!(*restored.film.read(), *mgr.film.read());

        let old = StabilizationManager::default();
        apply_as_preset(&old, r#"{"title":"Gyroflow data file","version":4}"#);
        assert_eq!(old.film.read().stock, None);
    }

    #[test]
    fn film_runs_only_on_graded_display_output() {
        let mgr = StabilizationManager::default();
        mgr.params.write().output_size = (3840, 2160);
        mgr.params.write().fps = 25.0;
        mgr.film.write().stock = Some("golden200".into());
        assert!(mgr.film_context(false).is_none(), "ungraded output must not get film");
        assert!(mgr.film_context(true).is_some());
        assert_eq!(mgr.film_context(true).unwrap().export_short, 2160);
        for cs in [color::OutputColorSpace::Rec2020Pq, color::OutputColorSpace::Rec2020Hlg] {
            mgr.color.write().output_color_space = cs;
            assert!(mgr.film_context(true).is_none(), "{cs:?} must not get film");
        }
        mgr.color.write().output_color_space = color::OutputColorSpace::Rec709;
        assert!(mgr.film_context(true).is_some());
        mgr.film.write().stock = None;
        assert!(mgr.film_context(true).is_none());
    }

    #[test]
    fn film_is_cloned_into_the_export_manager() {
        let mgr = StabilizationManager::default();
        mgr.film.write().stock = Some("vivid100".into());
        // The export renders from a clone; see the denoise line next to it.
        let cloned = mgr.get_cloned();
        assert_eq!(cloned.film.read().stock.as_deref(), Some("vivid100"));
    }
```

`configured_manager`, `round_trip` and `apply_as_preset` are the existing helpers of that test module (`lib.rs:2418`, `2441`, `2722`).

Run: `cd src/core; cargo test --lib film_`
Expected: compile FAIL — no field `film`.

- [ ] **Step 2: Implement**

In `StabilizationManager` (next to `pub denoise: …` at ~line 90):

```rust
    /// Film simulation (`film`). Render-only, like `denoise`: read per frame by the export
    /// closure and the preview, never by the stabilization maths.
    pub film: Arc<RwLock<film::FilmSettings>>,
```

Constructor (~line 132): `film: Arc::new(RwLock::new(film::FilmSettings::default())),`
Clone site (~line 1220): `film:       Arc::new(RwLock::new(self.film.read().clone())),`
Save (~line 1365, after `let denoise = …;`): `let film = self.film.read().to_json();` and add `"film": film,` next to `"denoise": denoise,`.
Load (after the `obj.get("denoise")` arm, ~line 1974):

```rust
            if let Some(v @ serde_json::Value::Object(_)) = obj.get("film") {
                self.film.write().apply_json(v);
            }
```

Methods (next to `set_preview_color_enabled`):

```rust
    /// Whether the preview shows a grade, which is what the film chain needs as input.
    pub fn film_preview_graded(&self) -> bool {
        self.params.read().preview_color_enabled && self.color.read().is_active()
    }

    /// The film chain's per-clip context, or `None` when it must not run: not graded
    /// (`graded` is the caller's bake/preview decision), HDR output, or nothing enabled.
    pub fn film_context(&self, graded: bool) -> Option<film::FilmContext> {
        use color::OutputColorSpace::{ Rec2020Pq, Rec2020Hlg };
        if !graded || matches!(self.color.read().output_color_space, Rec2020Pq | Rec2020Hlg) { return None; }
        let (output_size, fps) = { let p = self.params.read(); (p.output_size, p.fps) };
        let name = filesystem::get_filename(&self.input_file.read().url);
        film::FilmContext::new(self.film.read().clone(), output_size.0.min(output_size.1), fps, &name)
    }

    /// Why the preview shows no film, or empty when it does (or film is Off).
    pub fn film_gate_message(&self) -> String {
        if self.film.read().stock.is_none() { return String::new(); }
        use color::OutputColorSpace::{ Rec2020Pq, Rec2020Hlg };
        if matches!(self.color.read().output_color_space, Rec2020Pq | Rec2020Hlg) { return "Film is not available for HDR (PQ/HLG) output.".into(); }
        if !self.film_preview_graded() { return "Film needs a colour grade: enable colour adjustments (with a LUT for log footage).".into(); }
        film::last_skip().map(|w| format!("Film was skipped: {w}")).unwrap_or_default()
    }
```

CLI (`src/cli.rs`): add to the options struct after `denoise_chroma_smoothing`:

```rust
    /// film stock id (portrait160, golden200, vivid100, pastel400, verdant400, tungsten800, classic400bw, smooth400bw) or "off"
    #[argh(option)]
    film_stock: Option<String>,

    /// film format: s8, s16, 35, 120
    #[argh(option)]
    film_format: Option<String>,

    /// film grain amount, 0.0 to 2.0
    #[argh(option)]
    film_grain: Option<f64>,

    /// film halation amount, 0.0 to 1.0
    #[argh(option)]
    film_halation: Option<f64>,

    /// keep the grade's own tone and colour (grain and halation only)
    #[argh(switch)]
    film_no_tone: bool,
```

(Match the attribute style the existing `denoise_*` fields use — copy one of them and rename.)

After the denoise preset block, a second block in the same style:

```rust
        {
            let mut film = serde_json::Map::new();
            if let Some(ref s) = opts.film_stock {
                if s != "off" && gyroflow_core::film::stocks::stock(s).is_none() {
                    eprintln!("Unknown film stock: {s}.");
                    std::process::exit(1);
                }
                film.insert("stock".into(), s.clone().into());
            }
            if let Some(ref f) = opts.film_format {
                if gyroflow_core::film::stocks::FilmFormat::from_id(f).is_none() { eprintln!("Unknown film format: {f}. Expected s8, s16, 35 or 120."); std::process::exit(1); }
                film.insert("format".into(), f.clone().into());
            }
            if let Some(v) = opts.film_grain { film.insert("grain_amount".into(), v.into()); }
            if let Some(v) = opts.film_halation { film.insert("halation_amount".into(), v.into()); }
            if opts.film_no_tone { film.insert("tone".into(), false.into()); }
            if !film.is_empty() {
                // identical merge into `open_preset` as the denoise block above: copy its body verbatim
            }
        }
```

Fill the `if !film.is_empty()` body by copying the denoise block's merge code with `"denoise"` → `"film"` (the merge logic must be identical; if it is long, extract `fn merge_preset_section(open_preset: &mut String, key: &str, map: serde_json::Map<…>)` and use it from both — record the refactor as a ruling).

- [ ] **Step 3: Run the tests**

Run: `cd src/core; cargo test --lib film`
Expected: all film tests pass (16 + 3).
Then (app crate, for the CLI): `$env:RUSTFLAGS = $null; just build` — Expected: builds.

- [ ] **Step 4: Commit**

```bash
git add src/core/lib.rs src/cli.rs
git commit -m "feat(film): settings on the manager, project persistence, gate and CLI flags"
```

---

### Task 4: OpenCL film chain

**Files:**
- Create: `src/core/gpu/opencl_film.cl`, `src/core/film/tests_gpu.rs`
- Modify: `src/core/gpu/opencl.rs`, `src/core/stabilization/frame_transform.rs` (field), `src/core/stabilization/mod.rs` and `src/core/denoise/tests_gpu.rs` (struct literals), `src/core/film/mod.rs` (`#[cfg(test)] mod tests_gpu;`)

**Interfaces:**
- Consumes: `film::FilmParams`, `film::reference::render`, `film::set_last_skip`.
- Produces: `FrameTransform::film: crate::film::FilmParams` (inactive by default); `OclWrapper` runs the chain when `itm.film.is_active()`; test hook `opencl::FAIL_FILM_BUILD` (thread-local, `#[cfg(test)]`).

- [ ] **Step 1: Add the field** (no behaviour yet)

`frame_transform.rs` struct, after `chroma_filter`:

```rust
    /// The film chain to run after the chroma pass. Inactive (`flags == 0`) by default.
    pub film: crate::film::FilmParams,
```

Every `FrameTransform { chroma_filter: Default::default(), …` literal (3 sites: `frame_transform.rs:528`, `stabilization/mod.rs`, `denoise/tests_gpu.rs`) gets `film: Default::default(),` after it.

- [ ] **Step 2: Write the failing GPU tests** — `src/core/film/tests_gpu.rs`

```rust
// Copyright © 2026 Gyroflow

//! The film chain on real backends, against `film::reference`. Fixture: the identity transform
//! of `denoise::tests_gpu` on RGBA16 (the format exports bake into) and RGBA8 (the preview).
//! Run: cd src/core && cargo test --features use-opencl -- --ignored film_gpu --nocapture

use crate::gpu::{ Buffers, BufferDescription, BufferSource };
use crate::stabilization::{ FrameTransform, KernelParams, PixelType, RGBA8, RGBA16 };
use crate::stabilization::distortion_models::DistortionModel;
use super::{ FilmContext, FilmSettings, FilmParams, reference };

pub(super) const W: usize = 67; // odd, and not a multiple of the halo factor
pub(super) const H: usize = 45;

pub(super) fn scene() -> Vec<[f32; 4]> {
    (0..W * H).map(|i| {
        let (x, y) = (i % W, i / W);
        if (30..38).contains(&x) && (18..26).contains(&y) { return [1.0, 1.0, 1.0, 1.0]; } // highlight for halation
        let t = x as f32 / (W - 1) as f32;
        [0.1 + 0.8 * t, 0.5 - 0.3 * t, 0.3 + 0.4 * (y as f32 / (H - 1) as f32), 1.0]
    }).collect()
}

/// Golden 200 on Super 8 with everything on: export short 45 → p = 1, R = 0.008·45·24/4.01 ≈ 2.2 px
/// (f = 1). `preview_params` views an export 4x larger (p = 0.25) to exercise the preview lattice.
pub(super) fn params(max_value: f32) -> FilmParams {
    let s = FilmSettings { stock: Some("golden200".into()), format: super::stocks::FilmFormat::Super8, halation_amount: 1.0, ..Default::default() };
    let mut p = FilmContext::new(s, H, 25.0, "C0001.MP4").unwrap().frame_params(W, H, 40_000, max_value);
    p.width = W as i32; p.height = H as i32;
    p
}
pub(super) fn preview_params(max_value: f32) -> FilmParams {
    let s = FilmSettings { stock: Some("vivid100".into()), format: super::stocks::FilmFormat::Super16, ..Default::default() };
    let mut p = FilmContext::new(s, 4 * H, 25.0, "C0001.MP4").unwrap().frame_params(W, H, 80_000, max_value);
    p.width = W as i32; p.height = H as i32;
    assert!((p.inv_p - 4.0).abs() < 1e-6);
    p
}
/// Same, with the halo grid forced to f = 4 (67x45 → 17x12, last blocks clipped): see
/// `force_halo_grid` for why a frame this small cannot reach it by itself.
pub(super) fn halo_f_params(max_value: f32) -> FilmParams {
    let s = FilmSettings { stock: Some("tungsten800".into()), format: super::stocks::FilmFormat::Super8, halation_amount: 1.0, ..Default::default() };
    let mut p = FilmContext::new(s, H, 25.0, "x").unwrap().frame_params(W, H, 0, max_value);
    p.width = W as i32; p.height = H as i32;
    super::force_halo_grid(&mut p, 4, W, H);
    p
}

pub(super) fn identity(bpp: usize, max_value: f32, film: FilmParams) -> FrameTransform {
    let mut p = KernelParams::default();
    let (w, h, s) = (W as i32, H as i32, (W * bpp) as i32);
    p.width = w; p.height = h; p.stride = s;
    p.output_width = w; p.output_height = h; p.output_stride = s;
    p.matrix_count = 1; p.interpolation = 2;
    p.bytes_per_pixel = bpp as i32; p.pix_element_count = 4;
    p.f = [1.0, 1.0]; p.c = [0.0, 0.0]; p.fov = 1.0; p.lens_correction_amount = 1.0;
    p.source_rect = [0, 0, w, h]; p.output_rect = [0, 0, w, h];
    p.safe_area_rect = [0.0, 0.0, w as f32, h as f32];
    p.max_pixel_value = max_value; p.pixel_value_limit = max_value;
    p.light_refraction_coefficient = 1.0;
    let identity = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    FrameTransform { chroma_filter: Default::default(), film, matrices: vec![identity], reference_matrices: vec![], kernel_params: p, fov: 1.0, minimal_fov: 1.0, focal_length: None, mesh_data: vec![] }
}

pub(super) fn to16(px: &[[f32; 4]]) -> Vec<u8> { px.iter().flat_map(|p| p.iter().flat_map(|v| ((v * 65535.0).round() as u16).to_le_bytes())).collect() }
pub(super) fn from16(b: &[u8]) -> Vec<[f32; 4]> { b.chunks_exact(8).map(|c| [0, 1, 2, 3].map(|i| u16::from_le_bytes([c[i * 2], c[i * 2 + 1]]) as f32 / 65535.0)).collect() }
pub(super) fn to8(px: &[[f32; 4]]) -> Vec<u8> { px.iter().flat_map(|p| p.map(|v| (v * 255.0).round() as u8)).collect() }
pub(super) fn from8(b: &[u8]) -> Vec<[f32; 4]> { b.chunks_exact(4).map(|c| [0, 1, 2, 3].map(|i| c[i] as f32 / 255.0)).collect() }
pub(super) fn desc(buf: &mut [u8], bpp: usize) -> BufferDescription<'_> {
    BufferDescription { size: (W, H, W * bpp), rect: None, rotation: None, data: BufferSource::Cpu { buffer: buf }, texture_copy: false }
}
/// The reference on the *quantised* input, so only the film chain differs.
pub(super) fn expected(input: &[[f32; 4]], p: &FilmParams) -> Vec<[f32; 4]> { reference::render(input, W, H, p) }
pub(super) fn max_diff(a: &[[f32; 4]], b: &[[f32; 4]]) -> f32 { a.iter().zip(b).flat_map(|(x, y)| (0..4).map(move |i| (x[i] - y[i]).abs())).fold(0.0, f32::max) }

#[cfg(feature = "use-opencl")]
pub(super) fn run_opencl16(input: &[u8], film: FilmParams) -> Result<Vec<u8>, String> {
    if crate::gpu::opencl::OclWrapper::list_devices().is_empty() { return Err("UNAVAILABLE: no OpenCL device".into()); }
    let itm = identity(8, 65535.0, film);
    let mut inp = input.to_vec();
    let mut out = vec![0u8; W * H * 8];
    let mut buffers = Buffers { input: desc(&mut inp, 8), output: desc(&mut out, 8), references: vec![], guide: None };
    let inst = crate::gpu::opencl::OclWrapper::new(&itm.kernel_params, RGBA16::ocl_names(), DistortionModel::default(), None, &buffers, 0, None).map_err(|e| format!("OclWrapper::new failed: {e:?}"))?;
    inst.undistort_image(&mut buffers, &itm, &[]).map_err(|e| format!("undistort_image failed: {e:?}"))?;
    drop(buffers);
    Ok(out)
}
#[cfg(feature = "use-opencl")]
pub(super) fn run_opencl8(input: &[u8], film: FilmParams) -> Result<Vec<u8>, String> {
    if crate::gpu::opencl::OclWrapper::list_devices().is_empty() { return Err("UNAVAILABLE: no OpenCL device".into()); }
    let itm = identity(4, 255.0, film);
    let mut inp = input.to_vec();
    let mut out = vec![0u8; W * H * 4];
    let mut buffers = Buffers { input: desc(&mut inp, 4), output: desc(&mut out, 4), references: vec![], guide: None };
    let inst = crate::gpu::opencl::OclWrapper::new(&itm.kernel_params, RGBA8::ocl_names(), DistortionModel::default(), None, &buffers, 0, None).map_err(|e| format!("OclWrapper::new failed: {e:?}"))?;
    inst.undistort_image(&mut buffers, &itm, &[]).map_err(|e| format!("undistort_image failed: {e:?}"))?;
    drop(buffers);
    Ok(out)
}
#[cfg(not(feature = "use-opencl"))] pub(super) fn run_opencl16(_: &[u8], _: FilmParams) -> Result<Vec<u8>, String> { Err("UNAVAILABLE: built without use-opencl".into()) }
#[cfg(not(feature = "use-opencl"))] pub(super) fn run_opencl8(_: &[u8], _: FilmParams) -> Result<Vec<u8>, String> { Err("UNAVAILABLE: built without use-opencl".into()) }

pub(super) fn ok<T>(name: &str, r: Result<T, String>) -> Option<T> {
    match r { Ok(v) => Some(v), Err(e) if e.starts_with("UNAVAILABLE") => { println!("SKIPPED {name}: {e}"); None }, Err(e) => panic!("{name} failed: {e}") }
}

#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored film_gpu"]
fn film_gpu_opencl_matches_the_reference() {
    for (name, p) in [("full", params(65535.0)), ("halo f=4", halo_f_params(65535.0)), ("preview p=0.25", preview_params(65535.0))] {
        let input = to16(&scene());
        let Some(out) = ok("OpenCL 16", run_opencl16(&input, p)) else { return; };
        let d = max_diff(&from16(&out), &expected(&from16(&input), &p));
        println!("OpenCL RGBA16 {name}: max diff {:.2} codes", d * 65535.0);
        assert!(d <= 1.0 / 65535.0 + 1e-6, "OpenCL RGBA16 {name} differs from the reference by {} codes", d * 65535.0);
    }
    let input = to8(&scene());
    if let Some(out) = ok("OpenCL 8", run_opencl8(&input, params(255.0))) {
        let d = max_diff(&from8(&out), &expected(&from8(&input), &params(255.0)));
        println!("OpenCL RGBA8: max diff {:.2} codes", d * 255.0);
        assert!(d <= 1.0 / 255.0 + 1e-6, "OpenCL RGBA8 differs by {} codes", d * 255.0);
    }
}

#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored film_gpu"]
fn film_gpu_opencl_inactive_is_bit_exact() {
    let input = to16(&scene());
    if let Some(out) = ok("OpenCL", run_opencl16(&input, FilmParams::default())) { assert_eq!(out, input); }
}

/// Review focus 2: grain and halation on a full-scale 16-bit frame must clamp, never wrap.
#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored film_gpu"]
fn film_gpu_opencl_full_scale_does_not_wrap() {
    let s = FilmSettings { stock: Some("tungsten800".into()), tone: false, grain_amount: 2.0, halation_amount: 1.0, ..Default::default() };
    let mut p = FilmContext::new(s, H, 25.0, "x").unwrap().frame_params(W, H, 0, 65535.0);
    p.width = W as i32; p.height = H as i32;
    let input = to16(&vec![[1.0, 1.0, 1.0, 1.0]; W * H]);
    if let Some(out) = ok("OpenCL", run_opencl16(&input, p)) {
        let min = from16(&out).iter().map(|px| px[0]).fold(1.0, f32::min);
        assert!(min > 60000.0 / 65535.0, "a clipped pixel wrapped to {}", min * 65535.0);
    }
}

#[cfg(feature = "use-opencl")]
#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored film_gpu"]
fn film_gpu_opencl_build_failure_skips_film_and_keeps_the_backend() {
    crate::gpu::opencl::FAIL_FILM_BUILD.with(|f| f.set(true));
    let input = to16(&scene());
    let r = run_opencl16(&input, params(65535.0));
    crate::gpu::opencl::FAIL_FILM_BUILD.with(|f| f.set(false));
    if let Some(out) = ok("OpenCL", r) {
        assert_eq!(out, input, "with film failing to build the frame must come out unfilmed");
        assert!(super::last_skip().is_some(), "the skip reason must be recorded for the panel");
    }
}
```

Add `#[cfg(test)] mod tests_gpu;` to `film/mod.rs`.

Run: `$env:RUSTFLAGS = "-L native=C:/Users/Leo/Gits/gyroflow/ext/vcpkg/installed/x64-windows/lib"; cd src/core; cargo test --lib --features use-opencl -- --ignored film_gpu_opencl --nocapture --test-threads 1`
Expected: compile FAIL on `FAIL_FILM_BUILD`; after adding a stub `FAIL_FILM_BUILD` thread-local, `film_gpu_opencl_matches_the_reference` FAILS (output equals input: no chain yet) and the other three pass or fail as expected (`build_failure` fails on `last_skip`).

- [ ] **Step 3: Write `opencl_film.cl`**

```c
// Copyright © 2026 Gyroflow
// Film chain. Mirror of `film::reference` (Rust); tested against it in film/tests_gpu.rs.
// DATA_TYPE / DATA_TYPEF / DATA_CONVERT / DATA_CONVERTF / PIXEL_BYTES / ROUND_OFFSET
// are substituted by opencl.rs.

typedef struct {
    int width, height, stride; uint flags;
    int halo_f, halo_w, halo_h, halo_radius;
    int grain_radius; float max_value, grain_k, grain_sigma;
    float halo_sigma, halo_strength, inv_p, fit_b;
    float fit_s, fit_umin_a, pad0, pad1;
    uint4 seeds;
    float4 halo_tint, log_e0, k, u_min, range;
    float4 sens[3];
    float4 cross[3];
    float std_table[256];
} FilmParams;

#define FILM_TONE 1u
#define FILM_GRAIN 2u
#define FILM_HALO 4u
#define FILM_BW 8u
#define NOISE_SALT 0x3C6EF372u

inline uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
inline uint hash4(uint a, uint b, uint c, uint d) { return pcg(a ^ pcg(b ^ pcg(c ^ pcg(d)))); }
inline float film_gauss(uint h) {
    uint h1 = pcg(h), h2 = pcg(h1);
    float u1 = max((float)(h1 >> 8u) / 16777216.0f, 1.0f / 16777216.0f);
    float u2 = (float)(h2 >> 8u) / 16777216.0f;
    return sqrt(-2.0f * log(u1)) * cos(6.283185307179586f * u2);
}
inline float srgb_dec(float v) { return v <= 0.04045f ? v / 12.92f : pow((v + 0.055f) / 1.055f, 2.4f); }
inline float srgb_enc(float x) { return x <= 0.0031308f ? x * 12.92f : 1.055f * pow(x, 1.0f / 2.4f) - 0.055f; }
inline float smooth_step(float a, float b, float x) { float t = clamp((x - a) / (b - a), 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }
inline float gw(int k, float s) { return exp(-(float)(k * k) / (2.0f * s * s)); }
inline uint export_px(int x, float inv_p) { return (uint)(int)floor(((float)x + 0.5f) * inv_p); }
inline uint seed_of(__global const FilmParams *p, int l) { return l == 0 ? p->seeds.x : (l == 1 ? p->seeds.y : p->seeds.z); }

inline float4 load_px(__global const uchar *buf, int x, int y, __global const FilmParams *p) {
    float4 v = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    *(DATA_TYPEF *)&v = DATA_CONVERTF(*(__global const DATA_TYPE *)(buf + y * p->stride + x * PIXEL_BYTES));
    return v / p->max_value;
}

// H1: f x f box average of the highlight source.
__kernel void film_halo_source(__global const uchar *src, __global float *halo, __global const FilmParams *p) {
    int i = get_global_id(0), j = get_global_id(1);
    if (i >= p->halo_w || j >= p->halo_h) return;
    int f = p->halo_f;
    float s = 0.0f;
    for (int b = 0; b < f; b++) {
        int y = min(j * f + b, p->height - 1);
        for (int a = 0; a < f; a++) {
            float4 c = load_px(src, min(i * f + a, p->width - 1), y, p);
            float Y = 0.2126f * srgb_dec(c.x) + 0.7152f * srgb_dec(c.y) + 0.0722f * srgb_dec(c.z);
            s += smooth_step(0.75f, 1.0f, Y) * Y;
        }
    }
    halo[j * p->halo_w + i] = s / (float)(f * f);
}

// H2 / H3: separable clamp-edge Gaussian on the halo grid.
__kernel void film_halo_blur(__global const float *in, __global float *out, __global const FilmParams *p, int horizontal) {
    int i = get_global_id(0), j = get_global_id(1);
    if (i >= p->halo_w || j >= p->halo_h) return;
    int r = p->halo_radius;
    float s = 0.0f, ws = 0.0f;
    for (int t = -r; t <= r; t++) {
        float w = gw(t, p->halo_sigma);
        int x = horizontal ? clamp(i + t, 0, p->halo_w - 1) : i;
        int y = horizontal ? j : clamp(j + t, 0, p->halo_h - 1);
        s += w * in[y * p->halo_w + x];
        ws += w;
    }
    out[j * p->halo_w + i] = s / ws;
}

// G1: per-layer unit noise, blurred horizontally.
__kernel void film_noise_blur_x(__global float4 *g1, __global const FilmParams *p) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= p->width || y >= p->height) return;
    int r = p->grain_radius, layers = (p->flags & FILM_BW) ? 1 : 3;
    uint ey = export_px(y, p->inv_p);
    float n[3] = { 0.0f, 0.0f, 0.0f };
    float ws = 0.0f;
    for (int t = -r; t <= r; t++) {
        float w = gw(t, p->grain_sigma);
        uint ex = export_px(clamp(x + t, 0, p->width - 1), p->inv_p);
        for (int l = 0; l < layers; l++) n[l] += w * film_gauss(hash4(ex, ey, seed_of(p, l), NOISE_SALT));
        ws += w;
    }
    g1[y * p->width + x] = (float4)(n[0] / ws, n[1] / ws, n[2] / ws, 0.0f);
}

inline float sample_halo(__global const float *h, float hx, float hy, __global const FilmParams *p) {
    float fx = hx - 0.5f, fy = hy - 0.5f;
    int x0 = (int)floor(fx), y0 = (int)floor(fy);
    float tx = fx - (float)x0, ty = fy - (float)y0;
    int xa = clamp(x0, 0, p->halo_w - 1), xb = clamp(x0 + 1, 0, p->halo_w - 1);
    int ya = clamp(y0, 0, p->halo_h - 1), yb = clamp(y0 + 1, 0, p->halo_h - 1);
    return (h[ya * p->halo_w + xa] * (1.0f - tx) + h[ya * p->halo_w + xb] * tx) * (1.0f - ty)
         + (h[yb * p->halo_w + xa] * (1.0f - tx) + h[yb * p->halo_w + xb] * tx) * ty;
}

// G2: vertical blur of the noise, then halation -> curve -> grain -> crosstalk.
__kernel void film_apply(__global const uchar *src, __global const float4 *g1, __global const float *halo, __global uchar *dst, __global const FilmParams *p) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= p->width || y >= p->height) return;
    float4 c = load_px(src, x, y, p);
    float lin[3] = { srgb_dec(c.x), srgb_dec(c.y), srgb_dec(c.z) };
    if (p->flags & FILM_HALO) {
        float f = (float)p->halo_f;
        float hv = sample_halo(halo, ((float)x + 0.5f) / f, ((float)y + 0.5f) / f, p);
        lin[0] += p->halo_strength * hv * p->halo_tint.x;
        lin[1] += p->halo_strength * hv * p->halo_tint.y;
        lin[2] += p->halo_strength * hv * p->halo_tint.z;
    }
    float u[3];
    float lE0[3] = { p->log_e0.x, p->log_e0.y, p->log_e0.z }, kk[3] = { p->k.x, p->k.y, p->k.z };
    float um[3] = { p->u_min.x, p->u_min.y, p->u_min.z }, rg[3] = { p->range.x, p->range.y, p->range.z };
    for (int i = 0; i < 3; i++) {
        if (p->flags & FILM_TONE) {
            float e = p->sens[i].x * lin[0] + p->sens[i].y * lin[1] + p->sens[i].z * lin[2];
            float logE = log10(max(e, 1e-5f)) + p->fit_b;
            float d = um[i] + rg[i] / (1.0f + exp(-kk[i] * (logE - lE0[i])));
            u[i] = p->fit_umin_a + p->fit_s * (d - p->fit_umin_a);
        } else {
            u[i] = srgb_enc(max(lin[i], 0.0f));
        }
    }
    if (p->flags & FILM_GRAIN) {
        int r = p->grain_radius;
        float4 n = (float4)(0.0f);
        float ws = 0.0f;
        for (int t = -r; t <= r; t++) {
            float w = gw(t, p->grain_sigma);
            n += w * g1[clamp(y + t, 0, p->height - 1) * p->width + x];
            ws += w;
        }
        n /= ws;
        float nn[3] = { n.x, n.y, n.z };
        int bw = (p->flags & FILM_BW) != 0;
        for (int i = 0; i < 3; i++) {
            float sd = p->std_table[(int)(clamp(u[i], 0.0f, 1.0f) * 200.0f + 0.5f)];
            u[i] = clamp(u[i] + p->grain_k * sd * nn[bw ? 0 : i], 0.0f, 1.0f);
        }
    }
    float o[3];
    for (int i = 0; i < 3; i++) {
        o[i] = (p->flags & FILM_TONE) ? clamp(p->cross[i].x * u[0] + p->cross[i].y * u[1] + p->cross[i].z * u[2], 0.0f, 1.0f)
                                       : clamp(u[i], 0.0f, 1.0f);
    }
    float4 out = (float4)(o[0], o[1], o[2], c.w) * p->max_value + ROUND_OFFSET;
    *(__global DATA_TYPE *)(dst + y * p->stride + x * PIXEL_BYTES) = DATA_CONVERT(out);
}
```

Note the reference divides by the weight sum per-kernel (`weights()` normalises first, then sums); the kernel accumulates then divides. Both are the normalised Gaussian; float differences are far below one code.

- [ ] **Step 4: Wire the chain into `opencl.rs`**

Fields on `OclWrapper` (after `chroma_warned`):

```rust
    /// The film chain's program source, prepared at creation when this output and format can
    /// run it (CPU output, packed RGBA8/16). The chain itself is built on first use: its
    /// buffers are big (16 B/px) and most renders never ask for film.
    film_src: Option<String>,
    film: std::sync::OnceLock<Option<OclFilm>>,
```

Thread-local test hook next to `FAIL_CHROMA_BUILD`:

```rust
    /// Test-only: makes building the film chain fail.
    pub(crate) static FAIL_FILM_BUILD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
```

Struct:

```rust
struct OclFilm {
    halo_source: Kernel,
    halo_blur_x: Kernel,
    halo_blur_y: Kernel,
    noise_blur_x: Kernel,
    apply: Kernel,
    params: Buffer<u8>,
    dst: Buffer<u8>,
    /// g1, halo_a, halo_b: bound as kernel args, owned here.
    _scratch: [Buffer<f32>; 3],
}
```

In `new()`, after the `chroma` block:

```rust
            let film_src = (matches!(buffers.output.data, BufferSource::Cpu { .. }) && matches!(ocl_names.0, "uchar4" | "ushort4")).then(|| {
                include_str!("opencl_film.cl")
                    .replace("DATA_CONVERTF", ocl_names.3)
                    .replace("DATA_TYPEF", ocl_names.2)
                    .replace("DATA_CONVERT", ocl_names.1)
                    .replace("DATA_TYPE", ocl_names.0)
                    .replace("PIXEL_BYTES", &format!("{}", params.bytes_per_pixel))
                    .replace("ROUND_OFFSET", "0.5f")
            });
```

and `film_src, film: Default::default(),` in the `Ok(Self { … })`.

Methods:

```rust
    /// Builds the film chain on first use, for `pixels` output pixels. `None` (reason kept for
    /// the Film panel) when this output cannot run it or the driver rejects it; the wrapper
    /// keeps working either way. The source buffer is bound per frame in `run_film`.
    fn film_chain(&self, pixels: usize) -> Option<&OclFilm> {
        self.film.get_or_init(|| {
            let Some(code) = &self.film_src else { crate::film::set_last_skip("this output or pixel format is not supported by the film chain".into()); return None; };
            let build = || -> ocl::Result<OclFilm> {
                #[cfg(test)]
                if FAIL_FILM_BUILD.with(|f| f.get()) { return Err(ocl::Error::from(String::from("forced film build failure"))); }
                let program = Program::builder().src(code.as_str()).devices(self.queue.device()).build(&self.queue.context())?;
                let scratch = |len: usize| Buffer::<f32>::builder().queue(self.queue.clone()).len(len.max(1)).flags(MemFlags::new().read_write().host_no_access()).build();
                // Halo grids are at most the frame (f = 1); g1 holds one float4 per pixel.
                let (g1, halo_a, halo_b) = (scratch(pixels * 4)?, scratch(pixels)?, scratch(pixels)?);
                let params = Buffer::<u8>::builder().queue(self.queue.clone()).len(std::mem::size_of::<crate::film::FilmParams>()).flags(MemFlags::new().read_only().host_write_only()).build()?;
                let dst = Buffer::<u8>::builder().queue(self.queue.clone()).len(self.dst.len()).flags(MemFlags::new().write_only().host_read_only().alloc_host_ptr()).build()?;
                let kernel = |name: &str| { let mut b = Kernel::builder(); b.program(&program).name(name).queue(self.queue.clone()).global_work_size(1).disable_arg_type_check(); b };
                let halo_source  = unsafe { kernel("film_halo_source").arg(&self.dst).arg(&halo_a).arg(&params).build()? };
                let halo_blur_x  = unsafe { kernel("film_halo_blur").arg(&halo_a).arg(&halo_b).arg(&params).arg(1i32).build()? };
                let halo_blur_y  = unsafe { kernel("film_halo_blur").arg(&halo_b).arg(&halo_a).arg(&params).arg(0i32).build()? };
                let noise_blur_x = unsafe { kernel("film_noise_blur_x").arg(&g1).arg(&params).build()? };
                let apply        = unsafe { kernel("film_apply").arg(&self.dst).arg(&g1).arg(&halo_a).arg(&dst).arg(&params).build()? };
                Ok(OclFilm { halo_source, halo_blur_x, halo_blur_y, noise_blur_x, apply, params, dst, _scratch: [g1, halo_a, halo_b] })
            };
            match build() {
                Ok(f) => Some(f),
                Err(e) => { crate::film::set_last_skip(format!("the OpenCL film chain failed to build: {e:?}")); None }
            }
        }).as_ref()
    }

    /// Runs the film chain on `src` (the chroma pass's output when it ran, else the main
    /// kernel's) if it applies. Returns whether it ran.
    fn run_film(&self, buffers: &Buffers, src: &Buffer<u8>, p: &crate::film::FilmParams) -> ocl::Result<bool> {
        if !p.is_active() { return Ok(false); }
        let (w, h) = (buffers.output.size.0, buffers.output.size.1);
        let Some(f) = self.film_chain(w * h) else { return Ok(false); };
        // Per frame: the chroma pass may run on one frame and not the next.
        f.halo_source.set_arg(0, src)?;
        f.apply.set_arg(0, src)?;
        let mut kp = *p;
        kp.width = w as i32;
        kp.height = h as i32;
        kp.stride = buffers.output.size.2 as i32;
        f.params.write(bytemuck::bytes_of(&kp)).enq()?;
        unsafe {
            if kp.flags & crate::film::params::FLAG_HALATION != 0 {
                let g = (kp.halo_w as usize, kp.halo_h as usize);
                f.halo_source.cmd().global_work_size(g).enq()?;
                f.halo_blur_x.cmd().global_work_size(g).enq()?;
                f.halo_blur_y.cmd().global_work_size(g).enq()?;
            }
            if kp.flags & crate::film::params::FLAG_GRAIN != 0 {
                f.noise_blur_x.cmd().global_work_size((w, h)).enq()?;
            }
            f.apply.cmd().global_work_size((w, h)).enq()?;
        }
        Ok(true)
    }
```

If `ocl`'s `KernelBuilder::arg` chaining does not return `&mut Self` usable as above, build each kernel with a `let mut b = kernel(..); unsafe { b.arg(..)…; } b.build()?` block as the chroma pass does.

In `undistort_image`, replace the readback selection:

```rust
        let chroma_ran = self.run_chroma(buffers, &itm.chroma_filter)?;
        let film_src = match (&self.chroma, chroma_ran) { (Some(c), true) => &c.dst, _ => &self.dst };
        let film_ran = self.run_film(buffers, film_src, &itm.film)?;

        match &mut buffers.output.data {
            BufferSource::None => { },
            BufferSource::Cpu { buffer, .. } => {
                if film_ran {
                    self.film.get().and_then(|f| f.as_ref()).unwrap().dst.read(&mut **buffer).enq()?;
                } else {
                    film_src.read(&mut **buffer).enq()?;
                }
            },
```

`c.dst` is created `write_only()` in the chroma builder; the film kernels read it, so change its flags to `MemFlags::new().read_write().host_read_only().alloc_host_ptr()` (same change chroma made to the main `dst`).

- [ ] **Step 5: Run the GPU tests**

Run: `$env:RUSTFLAGS = "-L native=C:/Users/Leo/Gits/gyroflow/ext/vcpkg/installed/x64-windows/lib"; cd src/core; cargo test --lib --features use-opencl -- --ignored film_gpu_opencl --nocapture --test-threads 1`
Expected: 4 passed; printed max diffs ≤ 1 code.

- [ ] **Step 6: Run the whole core suite** (regressions in chroma/denoise readback)

Run: same RUSTFLAGS; `cargo test --lib --features use-opencl` then `cargo test --lib --features use-opencl -- --ignored --test-threads 1`
Expected: all pass (unit count = previous 208 + new film tests; GPU 11 + 4).

- [ ] **Step 7: Commit**

```bash
git add src/core/gpu/opencl_film.cl src/core/gpu/opencl.rs src/core/film src/core/stabilization src/core/denoise/tests_gpu.rs
git commit -m "feat(film): OpenCL film chain after the chroma pass"
```

---

### Task 5: wgpu film chain

**Files:**
- Create: `src/core/gpu/wgpu_film.wgsl`
- Modify: `src/core/gpu/wgpu.rs`, `src/core/film/tests_gpu.rs`

**Interfaces:**
- Consumes: `FrameTransform::film`, `film::set_last_skip`, Task 4's test helpers.
- Produces: `WgpuWrapper` runs the chain when `itm.film.is_active()` on the render (texture) path with CPU output and an `Rgba8Unorm`/`Rgba16Uint` target.

- [ ] **Step 1: Write the failing tests** (append to `film/tests_gpu.rs`)

```rust
fn run_wgpu(input: &[u8], bpp: usize, films: &[FilmParams]) -> Result<Vec<Vec<u8>>, String> {
    if crate::gpu::wgpu::WgpuWrapper::initialize_context().is_none() { return Err("UNAVAILABLE: no usable wgpu adapter".into()); }
    let max = if bpp == 8 { 65535.0 } else { 1.0 }; // Rgba8Unorm is normalised on wgpu
    let fmt = if bpp == 8 { RGBA16::wgpu_format() } else { RGBA8::wgpu_format() }.ok_or("no format")?;
    let mut itm = identity(bpp, max, FilmParams::default());
    itm.kernel_params.translation2d = [-0.5, -0.5]; // fragment path samples at pixel centres; see denoise::tests_gpu::run_wgpu
    let mut inp = input.to_vec();
    let mut out = vec![0u8; W * H * bpp];
    let mut buffers = Buffers { input: desc(&mut inp, bpp), output: desc(&mut out, bpp), references: vec![], guide: None };
    let inst = crate::gpu::wgpu::WgpuWrapper::new(&itm.kernel_params, fmt, DistortionModel::default(), None, &buffers, 0, None).map_err(|e| format!("WgpuWrapper::new failed: {e:?}"))?;
    let mut results = vec![];
    for f in films {
        itm.film = *f;
        if !inst.undistort_image(&mut buffers, &itm, &[]) { return Err("undistort_image returned false".into()); }
        if let BufferSource::Cpu { buffer } = &buffers.output.data { results.push(buffer.to_vec()); }
    }
    Ok(results)
}

#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored film_gpu"]
fn film_gpu_wgpu_matches_the_reference_and_opencl() {
    let input = to16(&scene());
    // One wrapper, three frames: full chain, a different halo grid (review focus 4), then off.
    let frames = [params(65535.0), halo_f_params(65535.0), preview_params(65535.0), FilmParams::default()];
    let out = ok("wgpu 16", run_wgpu(&input, 8, &frames)).expect("wgpu is required for this test");
    for (i, name) in ["full", "halo f=4", "preview p=0.25"].iter().enumerate() {
        let d = max_diff(&from16(&out[i]), &expected(&from16(&input), &frames[i]));
        println!("wgpu RGBA16 {name}: max diff {:.2} codes", d * 65535.0);
        assert!(d <= 1.0 / 65535.0 + 1e-6, "wgpu RGBA16 {name} differs by {} codes", d * 65535.0);
        if let Some(o) = ok("OpenCL 16", run_opencl16(&input, frames[i])) {
            let d = max_diff(&from16(&o), &from16(&out[i]));
            println!("wgpu vs OpenCL {name}: {:.2} codes", d * 65535.0);
            assert!(d <= 1.0 / 65535.0 + 1e-6);
        }
    }
    assert_eq!(out[3], input, "film off must be bit-exact after film frames on the same wrapper");

    let input8 = to8(&scene());
    let mut p8 = params(1.0);
    p8.max_value = 1.0;
    let out8 = ok("wgpu 8", run_wgpu(&input8, 4, &[p8])).unwrap();
    let d = max_diff(&from8(&out8[0]), &expected(&from8(&input8), &p8));
    println!("wgpu RGBA8: max diff {:.2} codes", d * 255.0);
    assert!(d <= 1.0 / 255.0 + 1e-6, "wgpu RGBA8 differs by {} codes", d * 255.0);
}

#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored film_gpu"]
fn film_gpu_wgpu_full_scale_does_not_wrap() {
    let s = FilmSettings { stock: Some("tungsten800".into()), tone: false, grain_amount: 2.0, halation_amount: 1.0, ..Default::default() };
    let mut p = FilmContext::new(s, H, 25.0, "x").unwrap().frame_params(W, H, 0, 65535.0);
    p.width = W as i32; p.height = H as i32;
    let input = to16(&vec![[1.0, 1.0, 1.0, 1.0]; W * H]);
    if let Some(out) = ok("wgpu", run_wgpu(&input, 8, &[p])) {
        let min = from16(&out[0]).iter().map(|px| px[0]).fold(1.0, f32::min);
        assert!(min > 60000.0 / 65535.0, "a clipped pixel wrapped to {}", min * 65535.0);
    }
}
```

Run: `… cargo test --lib --features use-opencl -- --ignored film_gpu_wgpu --nocapture --test-threads 1`
Expected: FAIL — wgpu output equals the input (no chain).

- [ ] **Step 2: Write `wgpu_film.wgsl`**

```wgsl
// Copyright © 2026 Gyroflow
// Film chain. Mirror of `film::reference` (Rust) and opencl_film.cl.
// SCALAR (f32 or u32) and FILM_ROUND (round, or nothing for f32) are substituted by wgpu.rs.

struct FilmParams {
    width: i32, height: i32, stride: i32, flags: u32,
    halo_f: i32, halo_w: i32, halo_h: i32, halo_radius: i32,
    grain_radius: i32, max_value: f32, grain_k: f32, grain_sigma: f32,
    halo_sigma: f32, halo_strength: f32, inv_p: f32, fit_b: f32,
    fit_s: f32, fit_umin_a: f32, pad0: f32, pad1: f32,
    seeds: vec4<u32>,
    halo_tint: vec4<f32>, log_e0: vec4<f32>, k: vec4<f32>, u_min: vec4<f32>, range: vec4<f32>,
    sens: array<vec4<f32>, 3>,
    cross: array<vec4<f32>, 3>,
    std_table: array<vec4<f32>, 64>,
}

@group(0) @binding(0) var<uniform> p: FilmParams;
@group(0) @binding(1) var src: texture_2d<SCALAR>;
@group(0) @binding(2) var aux_a: texture_2d<f32>;
@group(0) @binding(3) var aux_b: texture_2d<f32>;

const FILM_TONE: u32 = 1u;
const FILM_GRAIN: u32 = 2u;
const FILM_HALO: u32 = 4u;
const FILM_BW: u32 = 8u;

@vertex
fn film_vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 4>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0));
    return vec4<f32>(pos[i], 0.0, 1.0);
}

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn hash4(a: u32, b: u32, c: u32, d: u32) -> u32 { return pcg(a ^ pcg(b ^ pcg(c ^ pcg(d)))); }
fn film_gauss(h: u32) -> f32 {
    let h1 = pcg(h);
    let h2 = pcg(h1);
    let u1 = max(f32(h1 >> 8u) / 16777216.0, 1.0 / 16777216.0);
    let u2 = f32(h2 >> 8u) / 16777216.0;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}
fn dec(v: f32) -> f32 { if (v <= 0.04045) { return v / 12.92; } return pow((v + 0.055) / 1.055, 2.4); }
fn enc(x: f32) -> f32 { if (x <= 0.0031308) { return x * 12.92; } return 1.055 * pow(x, 1.0 / 2.4) - 0.055; }
fn sstep(a: f32, b: f32, x: f32) -> f32 { let t = clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
fn gw(k: i32, s: f32) -> f32 { return exp(-f32(k * k) / (2.0 * s * s)); }
fn export_px(x: i32) -> u32 { return u32(i32(floor((f32(x) + 0.5) * p.inv_p))); }
fn load_px(x: i32, y: i32) -> vec4<f32> { return vec4<f32>(textureLoad(src, vec2<i32>(x, y), 0)) / p.max_value; }

// H1
@fragment
fn film_halo_source(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let i = i32(pos.x); let j = i32(pos.y); let f = p.halo_f;
    var s = 0.0;
    for (var b = 0; b < f; b = b + 1) {
        for (var a = 0; a < f; a = a + 1) {
            let c = load_px(min(i * f + a, p.width - 1), min(j * f + b, p.height - 1));
            let y = 0.2126 * dec(c.x) + 0.7152 * dec(c.y) + 0.0722 * dec(c.z);
            s = s + sstep(0.75, 1.0, y) * y;
        }
    }
    return vec4<f32>(s / f32(f * f), 0.0, 0.0, 0.0);
}

fn halo_blur(pos: vec4<f32>, horizontal: bool) -> vec4<f32> {
    let i = i32(pos.x); let j = i32(pos.y); let r = p.halo_radius;
    var s = 0.0; var ws = 0.0;
    for (var t = -r; t <= r; t = t + 1) {
        let w = gw(t, p.halo_sigma);
        var c = vec2<i32>(i, clamp(j + t, 0, p.halo_h - 1));
        if (horizontal) { c = vec2<i32>(clamp(i + t, 0, p.halo_w - 1), j); }
        s = s + w * textureLoad(aux_a, c, 0).x;
        ws = ws + w;
    }
    return vec4<f32>(s / ws, 0.0, 0.0, 0.0);
}
// H2, H3
@fragment fn film_halo_blur_x(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> { return halo_blur(pos, true); }
@fragment fn film_halo_blur_y(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> { return halo_blur(pos, false); }

// G1
@fragment
fn film_noise_blur_x(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = i32(pos.x); let y = i32(pos.y); let r = p.grain_radius;
    var layers = 3;
    if ((p.flags & FILM_BW) != 0u) { layers = 1; }
    let ey = export_px(y);
    var n = vec3<f32>(0.0); var ws = 0.0;
    for (var t = -r; t <= r; t = t + 1) {
        let w = gw(t, p.grain_sigma);
        let ex = export_px(clamp(x + t, 0, p.width - 1));
        for (var l = 0; l < layers; l = l + 1) {
            n[l] = n[l] + w * film_gauss(hash4(ex, ey, p.seeds[l], 0x3C6EF372u));
        }
        ws = ws + w;
    }
    return vec4<f32>(n / ws, 0.0);
}

fn sample_halo(hx: f32, hy: f32) -> f32 {
    let fx = hx - 0.5; let fy = hy - 0.5;
    let x0 = i32(floor(fx)); let y0 = i32(floor(fy));
    let tx = fx - f32(x0); let ty = fy - f32(y0);
    let xa = clamp(x0, 0, p.halo_w - 1); let xb = clamp(x0 + 1, 0, p.halo_w - 1);
    let ya = clamp(y0, 0, p.halo_h - 1); let yb = clamp(y0 + 1, 0, p.halo_h - 1);
    let h = aux_b;
    return (textureLoad(h, vec2<i32>(xa, ya), 0).x * (1.0 - tx) + textureLoad(h, vec2<i32>(xb, ya), 0).x * tx) * (1.0 - ty)
         + (textureLoad(h, vec2<i32>(xa, yb), 0).x * (1.0 - tx) + textureLoad(h, vec2<i32>(xb, yb), 0).x * tx) * ty;
}

// G2: aux_a = G1 noise, aux_b = halo field.
@fragment
fn film_apply(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<SCALAR> {
    let x = i32(pos.x); let y = i32(pos.y);
    let c = load_px(x, y);
    var lin = vec3<f32>(dec(c.x), dec(c.y), dec(c.z));
    if ((p.flags & FILM_HALO) != 0u) {
        let f = f32(p.halo_f);
        lin = lin + p.halo_strength * sample_halo((f32(x) + 0.5) / f, (f32(y) + 0.5) / f) * p.halo_tint.xyz;
    }
    var u = vec3<f32>(0.0);
    for (var i = 0; i < 3; i = i + 1) {
        if ((p.flags & FILM_TONE) != 0u) {
            let e = dot(p.sens[i].xyz, lin);
            let log_e = log(max(e, 1e-5)) / log(10.0) + p.fit_b;
            let d = p.u_min[i] + p.range[i] / (1.0 + exp(-p.k[i] * (log_e - p.log_e0[i])));
            u[i] = p.fit_umin_a + p.fit_s * (d - p.fit_umin_a);
        } else {
            u[i] = enc(max(lin[i], 0.0));
        }
    }
    if ((p.flags & FILM_GRAIN) != 0u) {
        let r = p.grain_radius;
        var n = vec4<f32>(0.0); var ws = 0.0;
        for (var t = -r; t <= r; t = t + 1) {
            let w = gw(t, p.grain_sigma);
            n = n + w * textureLoad(aux_a, vec2<i32>(x, clamp(y + t, 0, p.height - 1)), 0);
            ws = ws + w;
        }
        n = n / ws;
        let bw = (p.flags & FILM_BW) != 0u;
        for (var i = 0; i < 3; i = i + 1) {
            let idx = i32(clamp(u[i], 0.0, 1.0) * 200.0 + 0.5);
            let sd = p.std_table[idx / 4][idx % 4];
            var nl = n[i];
            if (bw) { nl = n.x; }
            u[i] = clamp(u[i] + p.grain_k * sd * nl, 0.0, 1.0);
        }
    }
    var o = clamp(u, vec3<f32>(0.0), vec3<f32>(1.0));
    if ((p.flags & FILM_TONE) != 0u) {
        o = clamp(vec3<f32>(dot(p.cross[0].xyz, u), dot(p.cross[1].xyz, u), dot(p.cross[2].xyz, u)), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    return vec4<SCALAR>(FILM_ROUND(vec4<f32>(o, c.w) * p.max_value));
}
```

`log10` does not exist in WGSL; `log(x)/log(10.0)` matches the reference's `log10` to f32 rounding (tolerance covers it).

- [ ] **Step 3: Wire the chain into `wgpu.rs`**

Fields on `WgpuWrapper` (after `chroma_warned`):

```rust
    /// The film chain (`wgpu_film.wgsl`), built on first use: render path, CPU output,
    /// Rgba8Unorm/Rgba16Uint only. `None` inside = unsupported or failed to build.
    film_format: Option<(wgpu::TextureFormat, &'static str)>,
    film: parking_lot::Mutex<Option<Option<WgpuFilm>>>,
```

```rust
struct WgpuFilm {
    layout: wgpu::BindGroupLayout,
    halo_source: wgpu::RenderPipeline,
    halo_blur_x: wgpu::RenderPipeline,
    halo_blur_y: wgpu::RenderPipeline,
    noise_blur_x: wgpu::RenderPipeline,
    apply: wgpu::RenderPipeline,
    params: wgpu::Buffer,
    g1: wgpu::Texture,
    out: wgpu::Texture,
    dummy: wgpu::Texture,
    /// (halo_w, halo_h, a, b): recreated when the halo grid changes size.
    halo: Option<(u32, u32, wgpu::Texture, wgpu::Texture)>,
}
```

In `new()`, next to the chroma block: `let film_format = (uses_textures && matches!(buffers.output.data, BufferSource::Cpu { .. }) && matches!(wgpu_format.0, wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba16Uint)).then_some((wgpu_format.0, wgpu_format.1));` and `film_format, film: Default::default(),` in `Ok(Self { … })`. The chroma `out` texture gains `wgpu::TextureUsages::TEXTURE_BINDING` (the film reads it).

Builder (method on `WgpuWrapper`):

```rust
    fn build_film(&self, (format, scalar): (wgpu::TextureFormat, &'static str), w: u32, h: u32) -> Result<WgpuFilm, String> {
        let device = &self.device;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let src = include_str!("wgpu_film.wgsl").replace("SCALAR", scalar).replace("FILM_ROUND", if scalar == "u32" { "round" } else { "" });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("film"), source: wgpu::ShaderSource::Wgsl(Cow::Owned(src)) });
        let src_sample = if scalar == "u32" { wgpu::TextureSampleType::Uint } else { wgpu::TextureSampleType::Float { filterable: false } };
        let tex = |binding, sample_type| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None };
        let float = wgpu::TextureSampleType::Float { filterable: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("film"), entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            tex(1, src_sample), tex(2, float), tex(3, float),
        ] });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipe = |entry: &str, target: wgpu::TextureFormat| device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(entry), layout: Some(&pl),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("film_vertex"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some(entry), targets: &[Some(wgpu::ColorTargetState { format: target, blend: None, write_mask: wgpu::ColorWrites::default() })], compilation_options: Default::default() }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            multiview_mask: None, depth_stencil: None, multisample: wgpu::MultisampleState::default(), cache: Default::default(),
        });
        let r32 = wgpu::TextureFormat::R32Float;
        let tex2d = |w: u32, h: u32, f: wgpu::TextureFormat, usage: wgpu::TextureUsages| device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: f, usage, view_formats: &[] });
        let rw = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let film = WgpuFilm {
            halo_source: pipe("film_halo_source", r32), halo_blur_x: pipe("film_halo_blur_x", r32), halo_blur_y: pipe("film_halo_blur_y", r32),
            noise_blur_x: pipe("film_noise_blur_x", wgpu::TextureFormat::Rgba32Float), apply: pipe("film_apply", format),
            params: device.create_buffer(&wgpu::BufferDescriptor { label: Some("film params"), size: std::mem::size_of::<crate::film::FilmParams>() as u64, usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST, mapped_at_creation: false }),
            g1: tex2d(w, h, wgpu::TextureFormat::Rgba32Float, rw),
            out: tex2d(w, h, format, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC),
            dummy: tex2d(1, 1, r32, wgpu::TextureUsages::TEXTURE_BINDING),
            halo: None, layout,
        };
        #[cfg(test)]
        if FAIL_FILM_BUILD_WGPU.with(|f| f.get()) { return Err("forced film build failure".into()); }
        match pollster::block_on(scope.pop()) { Some(e) => Err(format!("{e}")), None => Ok(film) }
    }

    /// Records the film chain into `encoder` if it applies; returns whether it did.
    fn encode_film(&self, buffers: &Buffers, src: &wgpu::Texture, p: &crate::film::FilmParams, encoder: &mut wgpu::CommandEncoder) -> bool {
        use crate::film::params::{ FLAG_GRAIN, FLAG_HALATION };
        if !p.is_active() { return false; }
        let (w, h) = (buffers.output.size.0 as u32, buffers.output.size.1 as u32);
        let mut guard = self.film.lock();
        if guard.is_none() {
            *guard = Some(match self.film_format {
                None => { crate::film::set_last_skip("this output, pixel format or pipeline is not supported by the film chain".into()); None },
                Some(fmt) => match self.build_film(fmt, w, h) {
                    Ok(f) => Some(f),
                    Err(e) => { crate::film::set_last_skip(format!("the wgpu film chain failed to build: {e}")); None },
                },
            });
        }
        let Some(Some(film)) = guard.as_mut() else { return false; };
        let mut kp = *p;
        kp.width = w as i32; kp.height = h as i32; kp.stride = buffers.output.size.2 as i32;
        self.queue.write_buffer(&film.params, 0, bytemuck::bytes_of(&kp));

        if kp.flags & FLAG_HALATION != 0 {
            let (hw, hh) = (kp.halo_w as u32, kp.halo_h as u32);
            if !matches!(&film.halo, Some((a, b, _, _)) if *a == hw && *b == hh) {
                let mk = || self.device.create_texture(&wgpu::TextureDescriptor { label: Some("film halo"), size: wgpu::Extent3d { width: hw, height: hh, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::R32Float, usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[] });
                film.halo = Some((hw, hh, mk(), mk()));
            }
        }
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let src_v = view(src);
        let dummy_v = view(&film.dummy);
        let bind = |a: &wgpu::TextureView, b: &wgpu::TextureView| self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &film.layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: film.params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src_v) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(a) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(b) },
        ] });
        let pass = |encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, pipeline: &wgpu::RenderPipeline, group: &wgpu::BindGroup| {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None, depth_stencil_attachment: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: target, resolve_target: None, depth_slice: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
            });
            rp.set_pipeline(pipeline);
            rp.set_bind_group(0, group, &[]);
            rp.draw(0..4, 0..1);
        };
        let g1_v = view(&film.g1);
        let (halo_a_v, halo_b_v) = match &film.halo { Some((_, _, a, b)) => (view(a), view(b)), None => (view(&film.dummy), view(&film.dummy)) };
        if kp.flags & FLAG_HALATION != 0 {
            pass(encoder, &halo_a_v, &film.halo_source, &bind(&dummy_v, &dummy_v));
            pass(encoder, &halo_b_v, &film.halo_blur_x, &bind(&halo_a_v, &dummy_v));
            pass(encoder, &halo_a_v, &film.halo_blur_y, &bind(&halo_b_v, &dummy_v));
        }
        if kp.flags & FLAG_GRAIN != 0 {
            pass(encoder, &g1_v, &film.noise_blur_x, &bind(&dummy_v, &dummy_v));
        }
        let halo_for_apply = if kp.flags & FLAG_HALATION != 0 { &halo_a_v } else { &dummy_v };
        pass(encoder, &view(&film.out), &film.apply, &bind(&g1_v, halo_for_apply));
        true
    }
```

Test hook next to the wgpu module's other statics:

```rust
#[cfg(test)]
thread_local! {
    pub(crate) static FAIL_FILM_BUILD_WGPU: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
```

`dummy` is R32Float but binding 2 in G1/halo passes is declared `texture_2d<f32>` — compatible. Binding 2 in `film_apply` is the Rgba32Float g1 — also `texture_2d<f32>`, non-filterable: matches the layout.

A texture cannot be read and written in the same pass; each pass above reads only textures it does not render to (halo_a→halo_b→halo_a across passes).

In `undistort_image` replace the chroma copy block:

```rust
        let chroma_ran = self.encode_chroma(buffers, &itm.chroma_filter, &mut encoder);
        let film_src: Option<&wgpu::Texture> = if chroma_ran { Some(&self.chroma.as_ref().unwrap().out) } else { self.out_texture.wgpu_texture.as_ref() };
        let film_ran = match film_src { Some(t) => self.encode_film(buffers, t, &itm.film, &mut encoder), None => false };
        let copy_from: Option<wgpu::Texture> = if film_ran {
            self.film.lock().as_ref().and_then(|f| f.as_ref()).map(|f| f.out.clone())
        } else if chroma_ran { Some(self.chroma.as_ref().unwrap().out.clone()) } else { None };
        let _temp_texture2 = if let Some(tex) = copy_from {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: self.staging_buffer.as_ref().unwrap(), layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.padded_out_stride), rows_per_image: None } },
                wgpu::Extent3d { width: buffers.output.size.0 as u32, height: buffers.output.size.1 as u32, depth_or_array_layers: 1 },
            );
            None
        } else {
            handle_output_texture(&self.device, &buffers.output, &self.queue, &mut encoder, &self.out_texture, self.pixel_format, self.staging_buffer.as_ref().unwrap(), self.padded_out_stride)
        };
```

(`wgpu::Texture` is cheaply `Clone` in wgpu 30; if it is not, hold the lock guard across the copy instead and ledger it.) `WgpuWrapper::drop` must drop `film` before `device.poll()`: add `*self.film.lock() = None;` at the top of `drop`.

- [ ] **Step 4: Add the wgpu build-failure test and run all film GPU tests**

```rust
#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored film_gpu"]
fn film_gpu_wgpu_build_failure_skips_film_and_keeps_the_backend() {
    crate::gpu::wgpu::FAIL_FILM_BUILD_WGPU.with(|f| f.set(true));
    let input = to16(&scene());
    let r = run_wgpu(&input, 8, &[params(65535.0)]);
    crate::gpu::wgpu::FAIL_FILM_BUILD_WGPU.with(|f| f.set(false));
    if let Some(out) = ok("wgpu", r) { assert_eq!(out[0], input); }
}
```

Run: `… cargo test --lib --features use-opencl -- --ignored film_gpu --nocapture --test-threads 1`
Expected: 7 passed (4 OpenCL + 3 wgpu); printed diffs ≤ 1 code; wgpu vs OpenCL ≤ 1 code.

- [ ] **Step 5: Full core suite** (unit + all `--ignored`), as in Task 4 Step 6. Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/core/gpu/wgpu_film.wgsl src/core/gpu/wgpu.rs src/core/film/tests_gpu.rs
git commit -m "feat(film): wgpu film chain after the chroma pass"
```

---

### Task 6: Wire into exports, the preview and the Film panel

**Files:**
- Modify: `src/rendering/mod.rs` (plane closure, ~lines 1036–1110), `src/core/lib.rs` (`process_pixels_with_references`, ~lines 890–967), `src/controller.rs` (texture callback ~line 1286, new qt methods), `src/ui/App.qml`, `src/ui/VideoArea.qml`, `src/ui/menu/qmldir`, `src/resources_qml.rs`
- Create: `src/ui/menu/Film.qml`

**Interfaces:**
- Consumes: `StabilizationManager::{film_context, film_preview_graded, film_gate_message}`, `film::{supported, FilmContext::frame_params}`, `FrameTransform::film`.
- Produces: `fn film_plane_ok<T: PixelType + 'static>(packed_rgb: bool, bake_color: bool) -> bool` in `rendering/mod.rs`; controller qt methods `set_film_stock(QString)`, `set_film_format(QString)`, `set_film_flag(QString, bool)`, `set_film_param(QString, f64)`, `get_film_stocks() -> QJsonArray`, `get_film_formats() -> QJsonArray`, `get_film_gate_message() -> QString`.

- [ ] **Step 1: Write the failing export-gate test** (in `rendering/mod.rs`'s test module, next to `chroma_plane_tests`)

```rust
#[cfg(test)]
mod film_plane_tests {
    use super::*;
    use gyroflow_core::stabilization::*;
    #[test]
    fn film_runs_only_on_baked_packed_rgba() {
        assert!(film_plane_ok::<RGBA8>(true, true));
        assert!(film_plane_ok::<RGBA16>(true, true));
        assert!(!film_plane_ok::<RGBA16>(true, false), "not baked");
        assert!(!film_plane_ok::<RGBA16>(false, true), "packed YUV (AYUV16 dispatch has yuvi)");
        assert!(!film_plane_ok::<BGRA8>(true, true), "BGRA would swap R and B in OpenCL");
        assert!(!film_plane_ok::<RGBAf>(true, true));
        assert!(!film_plane_ok::<Luma16>(true, true));
        assert!(!film_plane_ok::<UV16>(true, true));
    }
}
```

Run: `$env:RUSTFLAGS = $null; just test` (or `cargo test film_plane` in the app crate)
Expected: compile FAIL — `film_plane_ok` not found.

- [ ] **Step 2: Implement the export wiring** (`rendering/mod.rs`)

Next to `chroma_plane_mode`:

```rust
/// Whether the film chain may run on this plane: only the packed RGBA planes colour baking
/// produces (see `packed_color_format`), and only when the grade is actually baked in.
fn film_plane_ok<T: PixelType + 'static>(packed_rgb: bool, bake_color: bool) -> bool {
    packed_rgb && bake_color && gyroflow_core::film::supported::<T>()
}
```

Inside `create_planes_proc!`, next to `let packed_rgb = …`:

```rust
                    // Read once per plane: the film chain's settings, gate and seed for this render.
                    let film_ctx = if film_plane_ok::<$t>(packed_rgb, bake_color) { stab.film_context(true) } else { None };
```

Inside the per-frame closure, at its top (before `let timestamp_us = timestamp_us + offset_us;`): `let source_timestamp_us = timestamp_us;`. After the denoise block sets `transform.chroma_filter`:

```rust
                        if let Some(ctx) = &film_ctx {
                            // `max_pixel_value` is final for this backend by now (1.0 on wgpu unorm).
                            transform.film = ctx.frame_params(out_size.0, out_size.1, source_timestamp_us, transform.kernel_params.max_pixel_value);
                        }
```

Check `out_size` is `(width, height, stride)` from `zero_copy::get_plane_size` (it is used for `get_plane_buffer` above); if it is a different shape, use `out_frame_data.plane_width(plane_index)`/`plane_height`.

- [ ] **Step 3: Implement the preview wiring** (`lib.rs`, `process_pixels_with_references`)

At the very top, before `timestamp_us += offset_us;`: `let source_timestamp_us = timestamp_us;`. Replace the body of `if let Some(undist) = self.stabilization.try_read_for(…) { … }` with:

```rust
            let film_ctx = if T::COUNT >= 3 && film::supported::<T>() { self.film_context(self.film_preview_graded()) } else { None };
            if film_ctx.is_none() && (reference_timestamps.is_empty() || buffers.references.is_empty()) {
                buffers.references.clear();
                undist.process_pixels::<T>(timestamp_us, frame, buffers, None)
            } else {
                if buffers.references.is_empty() { reference_timestamps.clear(); }
                if reference_timestamps.is_empty() { buffers.references.clear(); }
                let mut transform = undist.get_frame_transform_with_references::<T>(timestamp_us, &reference_timestamps, frame, buffers);
                let denoise = self.denoise.read();
                transform.kernel_params.denoise_sigma = denoise.luma_sigma();
                if T::COUNT >= 3 && !reference_timestamps.is_empty() {
                    transform.kernel_params.denoise_rgb = [denoise.chroma_sigma(), 0.0, 0.0, 0.0];
                    transform.chroma_filter = denoise.chroma_filter_params(denoise::chroma_filter::MODE_RGB, transform.kernel_params.max_pixel_value);
                }
                drop(denoise);
                if let Some(ctx) = &film_ctx {
                    transform.film = ctx.frame_params(buffers.output.size.0, buffers.output.size.1, source_timestamp_us, transform.kernel_params.max_pixel_value);
                }
                undist.process_pixels::<T>(timestamp_us, frame, buffers, Some(&transform))
            }
```

Keep the two existing explanatory comments (the "Clear rather than merely branch" and the "Preview pixels are one interleaved RGBA plane" ones) on the lines they describe. Verify, before relying on it, that `get_frame_transform_with_references` with an empty slice yields the same matrices as the `None` path: read its body; if it treats an empty slice specially or differs, add a test `preview_transform_without_references_matches_the_plain_path` comparing `transform.matrices` from both on a default manager, and ledger the finding.

- [ ] **Step 4: Controller**

Texture callback (after the denoise `UsePixels` line, ~1286):

```rust
                // The film chain runs only on CPU outputs (see film_plane_ok); hand film frames
                // to the pixel callback like denoised ones. During playback too: grain moves.
                if stab.film_context(stab.film_preview_graded()).is_some() { return TextureResult::UsePixels; }
```

Qt methods (declare next to `set_denoise_param` at ~line 236, implement next to it):

```rust
    set_film_stock: qt_method!(fn(&mut self, id: QString)),
    set_film_format: qt_method!(fn(&mut self, id: QString)),
    set_film_flag: qt_method!(fn(&mut self, name: QString, v: bool)),
    set_film_param: qt_method!(fn(&mut self, name: QString, v: f64)),
    get_film_stocks: qt_method!(fn(&self) -> QJsonArray),
    get_film_formats: qt_method!(fn(&self) -> QJsonArray),
    get_film_gate_message: qt_method!(fn(&self) -> QString),
```

```rust
    // Film is render-only like denoise: `request_recompute()` redraws the preview.
    fn set_film_stock(&mut self, id: QString) {
        let id = id.to_string();
        { let mut f = self.stabilizer.film.write(); if id == "off" { f.stock = None; } else { f.set_stock(&id); } }
        self.request_recompute();
    }
    fn set_film_format(&mut self, id: QString) {
        match core::film::stocks::FilmFormat::from_id(&id.to_string()) {
            Some(f) => self.stabilizer.film.write().format = f,
            None => { ::log::warn!("Unknown film format: {id}"); return; }
        }
        self.request_recompute();
    }
    fn set_film_flag(&mut self, name: QString, v: bool) {
        if !self.stabilizer.film.write().set_flag(&name.to_string(), v) { ::log::warn!("Unknown film flag: {name}"); return; }
        self.request_recompute();
    }
    fn set_film_param(&mut self, name: QString, v: f64) {
        if !self.stabilizer.film.write().set_param(&name.to_string(), v) { ::log::warn!("Rejected film parameter: {name} = {v}"); return; }
        self.request_recompute();
    }
    fn get_film_stocks(&self) -> QJsonArray {
        let mut a = QJsonArray::default();
        for s in core::film::stocks::STOCKS.iter() {
            a.push(QJsonValue::from(QJsonObject::from(std::collections::HashMap::from([("id", QJsonValue::from(QString::from(s.id))), ("name", QJsonValue::from(QString::from(s.name)))]))));
        }
        a
    }
    fn get_film_formats(&self) -> QJsonArray {
        let mut a = QJsonArray::default();
        for f in core::film::stocks::FilmFormat::ALL {
            a.push(QJsonValue::from(QJsonObject::from(std::collections::HashMap::from([("id", QJsonValue::from(QString::from(f.id()))), ("name", QJsonValue::from(QString::from(f.label())))]))));
        }
        a
    }
    fn get_film_gate_message(&self) -> QString { QString::from(self.stabilizer.film_gate_message()) }
```

Copy the QJsonArray/QJsonObject construction style from `get_output_color_spaces` (`controller.rs:1763`), which builds the same id/name list for a ComboBox; the code above is a sketch of that and must follow whatever that function does.

- [ ] **Step 5: The Film panel** — `src/ui/menu/Film.qml`

```qml
// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Adrian <adrian.eddy at gmail>

import QtQuick

import "../components/"

MenuItem {
    id: root;
    text: qsTr("Film");
    iconName: "grid"; // see Denoise.qml: no film icon in the packaged subset; pick another existing one if "grid" reads badly next to Denoise
    innerItem.enabled: window.videoArea.vid.loaded;
    objectName: "film";
    opened: false;

    readonly property var stocks: controller.get_film_stocks();
    readonly property var formats: controller.get_film_formats();
    property bool filmOn: stockCb.currentIndex > 0;

    // Mirrors the core importer (`obj.get("film")`): a file with no `film` section leaves the panel alone.
    function loadGyroflow(obj: var): void {
        const f = obj.film;
        if (!f || typeof f !== "object") return;
        if (typeof f.stock !== "undefined") stockCb.selectId(f.stock);
        if (typeof f.format !== "undefined") formatCb.selectId(f.format);
        if (typeof f.tone !== "undefined") toneCb.checked = !!f.tone;
        if (typeof f.grain !== "undefined") grainCb.checked = !!f.grain;
        if (typeof f.halation !== "undefined") haloCb.checked = !!f.halation;
        if (typeof f.grain_amount !== "undefined") grainSlider.value = +f.grain_amount;
        if (typeof f.halation_amount !== "undefined") haloSlider.value = +f.halation_amount;
    }

    Label {
        text: qsTr("Stock");
        ComboBox {
            id: stockCb;
            width: parent.width;
            font.pixelSize: 12 * dpiScale;
            model: [qsTr("Off")].concat(root.stocks.map(s => s.name));
            currentIndex: 0;
            property bool preventChange: false;
            function selectId(id: string): void {
                const i = id === "off" ? 0 : root.stocks.findIndex(s => s.id === id) + 1;
                preventChange = true; currentIndex = Math.max(0, i); preventChange = false;
                controller.set_film_stock(currentIndex > 0 ? root.stocks[currentIndex - 1].id : "off");
            }
            onCurrentIndexChanged: if (!preventChange) controller.set_film_stock(currentIndex > 0 ? root.stocks[currentIndex - 1].id : "off");
        }
    }

    BasicText {
        id: gateText;
        width: parent.width;
        wrapMode: Text.WordWrap;
        visible: root.filmOn && text.length > 0;
        opacity: 0.6;
        text: "";
        // The gate depends on the colour panel and the output colour space, which change elsewhere.
        Timer { interval: 1000; running: root.filmOn && root.opened; repeat: true; triggeredOnStart: true; onTriggered: gateText.text = controller.get_film_gate_message(); }
    }

    Label {
        text: qsTr("Format");
        tooltip: qsTr("The film gauge the frame stands for. Smaller formats have coarser grain and wider halation at the same output size.");
        visible: root.filmOn;
        ComboBox {
            id: formatCb;
            width: parent.width;
            font.pixelSize: 12 * dpiScale;
            model: root.formats.map(f => f.name);
            currentIndex: Math.max(0, root.formats.findIndex(f => f.id === "35"));
            function selectId(id: string): void { const i = root.formats.findIndex(f => f.id === id); if (i >= 0) currentIndex = i; }
            onCurrentIndexChanged: if (currentIndex >= 0) controller.set_film_format(root.formats[currentIndex].id);
        }
    }

    CheckBox { id: toneCb; visible: root.filmOn; text: qsTr("Tone & colour"); checked: true; onCheckedChanged: controller.set_film_flag("tone", checked);
               tooltip: qsTr("Re-develop the grade with the stock's curve and colour. Off keeps your grade and adds only grain and halation."); }
    CheckBox { id: grainCb; visible: root.filmOn; text: qsTr("Grain"); checked: true; onCheckedChanged: controller.set_film_flag("grain", checked); }
    CheckBox { id: haloCb; visible: root.filmOn; text: qsTr("Halation"); checked: true; onCheckedChanged: controller.set_film_flag("halation", checked);
               tooltip: qsTr("A red glow around clipped highlights, from light reflecting off the film base."); }

    Label {
        text: qsTr("Grain amount");
        visible: root.filmOn && grainCb.checked;
        SliderWithField {
            id: grainSlider;
            width: parent.width;
            from: 0; to: 200; value: 1.0; defaultValue: 1.0; scaler: 100.0; unit: "%"; precision: 0;
            onValueChanged: controller.set_film_param("grain_amount", value);
        }
    }
    Label {
        text: qsTr("Halation");
        visible: root.filmOn && haloCb.checked;
        SliderWithField {
            id: haloSlider;
            width: parent.width;
            from: 0; to: 100; value: 0.5; defaultValue: 0.5; scaler: 100.0; unit: "%"; precision: 0;
            onValueChanged: controller.set_film_param("halation_amount", value);
        }
    }
}
```

`CheckBox` here may not have a `tooltip` property; check `src/ui/components/CheckBox.qml` and drop the `tooltip:` lines if it has none.

Registration:
- `src/ui/menu/qmldir`: `Film 1.0 Film.qml` (alphabetical, after `Export`).
- `src/resources_qml.rs`: `"src/ui/menu/Film.qml",` after the Denoise line.
- `src/ui/App.qml`: after the denoise loader and Hr: `ItemLoader { id: film; sourceComponent: Component { Menu.Film { } } }` and `Hr { id: filmHr; }`; alias next to `property alias denoise: denoise.item;`: `property alias film: film.item;`.
- `src/ui/VideoArea.qml`: after `window.denoise.loadGyroflow(obj);`: `window.film.loadGyroflow(obj);`.

- [ ] **Step 6: Build, test, run**

Run: `$env:RUSTFLAGS = $null; just build` then `just test`
Expected: build OK; app tests pass (139 + `film_plane_tests`).
Then the core suites from Task 5 Step 5 once more.

Visual check (ask the user before closing a running Gyroflow; copy `gyroflow.log` aside first): `just run`, open a clip with a LUT, pick Golden 200 → the preview shows grain moving during playback; toggle Tone & colour; Halation on a clipped highlight; Stock Off → identical to before. Report what was seen; the user verifies the look.

- [ ] **Step 7: Commit**

```bash
git add src/rendering/mod.rs src/core/lib.rs src/controller.rs src/ui src/resources_qml.rs
git commit -m "feat(film): film chain in exports and the preview, Film panel"
```

---

### Task 7: Verify on real footage, then tidy up

**Files:**
- Modify: `docs/superpowers/specs/2026-10-04-film-simulation-design.md` (Results section; rulings that changed the design: lazy buffers instead of a checksum key, render passes instead of compute on wgpu, the Wigglecam seed form, panel message instead of export-dialog notice)

- [ ] **Step 1: Export timing and grain statistics**

Copy `gyroflow.log` aside first. Using the CLI on the same C2520 clip used for chroma smoothing (graded with its LUT, 4K, HEVC P010 as before):
1. Export with film Off; note the time from the log.
2. Export with `--film-stock golden200 --film-format s16`; note the time.
3. Extract frame 8 from each with ffmpeg (`-ss`/`-frames:v`), crop a flat patch (sky or wall, at least 64×64), and measure the std of (filmed − unfilmed) in Python with numpy; compare with `amount·std(u)` printed by a one-off `cargo test -- --nocapture` helper or computed from `FilmContext::frame_params` (`std_table[round(u·200)]` at the patch's mean value).
Expected: export time increase ≤ 15 %; measured std within 15 % of the model (P010 quantisation and chroma subsampling soften it, so this is looser than the unit test).

- [ ] **Step 2: Write the Results section**, listing timings, the measured vs model std, the GPU parity numbers printed by Tasks 4/5, and the rulings above.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-10-04-film-simulation-design.md
git commit -m "docs: film simulation results on real footage"
```
