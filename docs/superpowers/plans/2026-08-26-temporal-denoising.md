# Temporal Denoising Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Denoise video temporally by gathering each output pixel from `2k+1` source frames, motion-compensated using the camera rotation Gyroflow already computes from gyro data.

**Architecture:** The denoise is *fused into the existing undistort kernel* rather than added as a separate pass. `undistort_coord()` already maps an output pixel back into source space through a `matrices` array of 14-float entries indexed per scanline for rolling shutter. Holding `2k+1` matrix blocks instead of one makes each reference frame sample land exactly where that pixel is in that frame. Samples are combined with a Gaussian weight on their difference from the centre sample, so disagreeing samples (parallax, moving subjects) drop out.

**Tech Stack:** Rust 2024, wgpu + WGSL, OpenCL C, Qt/QML via `qmetaobject`, `ffmpeg-next` 8.1, `bytemuck`.

**Spec:** `docs/superpowers/specs/2026-08-26-temporal-denoising-design.md`

## Global Constraints

- Build and test with `just`, never bare `cargo`. Test: `just test`. Build: `just build`.
- Line endings are mixed per file in this repo. Detect per file before editing; never use `sed -i` or PowerShell `Set-Content`.
- `KernelParams` is byte-compatible across **five** files and every change must be applied to all five in the same commit: `src/core/stabilization/mod.rs`, `src/core/gpu/opencl_undistort.cl`, `src/core/gpu/wgpu_undistort.wgsl`, `src/qt_gpu/undistort.frag`, `src/core/gpu/stabilize_spirv/src/types.rs`. Each field aligns to 4 bytes, the whole struct to 16.
- Matrix blocks are exactly **14 floats** per matrix (`wgpu_undistort.wgsl:624`: `min(sy, matrix_count-1) * 14u`).
- `k = 0` must cost nothing measurable: every new shader branch is guarded on `reference_count <= 1`.
- Quality tiers are named **Fast** (`k=1`) and **High** (`k=3`). Never "Preview"/"Export" — the paused preview renders at whatever quality is selected.
- No CPU denoise path. Where no GPU backend is available, denoise is pass-through.

---

### Task 1: `DenoiseSettings` core type

**Files:**
- Create: `src/core/denoise/mod.rs`
- Modify: `src/core/lib.rs` (add `pub mod denoise;` beside the other `pub mod` lines; add the field to `StabilizationManager` at `:87` and its initialiser at `:121`)

**Interfaces:**
- Consumes: nothing.
- Produces: `denoise::DenoiseSettings { enabled: bool, quality: DenoiseQuality, luma_strength: f64, chroma_strength: f64 }`; `DenoiseQuality::{Fast, High}` with `radius() -> usize`, `id() -> &'static str`, `from_id(&str) -> Option<Self>`; `DenoiseSettings::radius() -> usize`, `radius_for_size(usize, usize) -> usize`, `sigma_for_plane(i32) -> f32`, `set_param(&str, f64) -> bool`, `set_param_value(&str) -> f64`, `get_checksum() -> u64`; the free function `denoise::rejection_weight(distance: f32, sigma: f32) -> f32`. On `StabilizationManager`: `pub denoise: Arc<RwLock<denoise::DenoiseSettings>>`.

- [ ] **Step 1: Write the failing tests**

Create `src/core/denoise/mod.rs` containing only a `mod tests` block for now:

```rust
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
        // 8K: 7680*4320*4 = ~132 MB/frame, budget allows 3 frames -> k=1.
        assert_eq!(d.radius_for_size(7680, 4320), 1);
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `just test denoise`
Expected: FAIL — `cannot find type DenoiseSettings in this scope`.

- [ ] **Step 3: Write the implementation**

Prepend to `src/core/denoise/mod.rs`, above the `mod tests` block:

```rust
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
```

- [ ] **Step 4: Register the module and the manager field**

In `src/core/lib.rs`, add `pub mod denoise;` alongside the existing `pub mod color;` declaration.

In the `StabilizationManager` struct (near `:87`, next to `pub color: Arc<RwLock<color::ColorAdjustments>>`):

```rust
    pub denoise: Arc<RwLock<denoise::DenoiseSettings>>,
```

In its initialiser (near `:121`, next to `color: Arc::new(RwLock::new(color::ColorAdjustments::default())),`):

```rust
            denoise: Arc::new(RwLock::new(denoise::DenoiseSettings::default())),
```

If the compiler reports a third construction site (`lib.rs:1191` clones the manager for rendering), add `denoise: Arc::new(RwLock::new(self.denoise.read().clone())),` there too, matching how `color` is handled on that line.

- [ ] **Step 5: Run tests to verify they pass**

Run: `just test denoise`
Expected: PASS, 8 tests.

- [ ] **Step 6: Commit**

```bash
git add src/core/denoise/mod.rs src/core/lib.rs
git commit -m "feat(denoise): settings type for temporal denoising"
```

---

### Task 2: Project file serialization

**Files:**
- Modify: `src/core/lib.rs:1301` (build the JSON object), `:1401` (insert into the project object), `:1792` (read it back)
- Test: `src/core/lib.rs` (the existing `mod tests` that holds the `apply_as_preset` tests at `:2488`)

**Interfaces:**
- Consumes: `denoise::DenoiseSettings`, `DenoiseQuality::{id, from_id}`, `DenoiseSettings::set_param` from Task 1.
- Produces: a `"denoise"` object in the `.gyroflow` file with keys `enabled`, `quality`, `luma_strength`, `chroma_strength`.

- [ ] **Step 1: Write the failing tests**

Add to the existing `mod tests` block in `src/core/lib.rs`, beside the colour serialization tests:

```rust
    #[test]
    fn denoise_settings_round_trip_through_the_project_file() {
        let mgr = StabilizationManager::default();
        {
            let mut d = mgr.denoise.write();
            d.enabled = true;
            d.quality = denoise::DenoiseQuality::High;
            d.luma_strength = 0.25;
            d.chroma_strength = 0.75;
        }
        let json = mgr.export_gyroflow_data(gyro_export::ExportSettings::default(), "{}", None).unwrap();

        let restored = StabilizationManager::default();
        restored.import_gyroflow_data(json.as_bytes(), true, None, |_| (), Arc::new(AtomicBool::new(false)), false).unwrap();

        let a = mgr.denoise.read();
        let b = restored.denoise.read();
        assert_eq!(a.enabled, b.enabled);
        assert_eq!(a.quality, b.quality);
        assert_eq!(a.luma_strength, b.luma_strength);
        assert_eq!(a.chroma_strength, b.chroma_strength);
    }

    #[test]
    fn denoise_import_clamps_hand_edited_values() {
        let mgr = StabilizationManager::default();
        apply_as_preset(&mgr, r#"{"title":"Gyroflow data file","version":4,"denoise":{"enabled":true,"luma_strength":99.0,"chroma_strength":-99.0}}"#);
        let d = mgr.denoise.read();
        assert_eq!(d.luma_strength, denoise::MAX_STRENGTH);
        assert_eq!(d.chroma_strength, denoise::MIN_STRENGTH);
    }

    #[test]
    fn denoise_import_falls_back_on_an_unknown_quality_id() {
        let mgr = StabilizationManager::default();
        mgr.denoise.write().quality = denoise::DenoiseQuality::High;
        // An id written by a newer build must leave the current value alone
        // rather than fail the import, matching how output_color_space behaves.
        apply_as_preset(&mgr, r#"{"title":"Gyroflow data file","version":4,"denoise":{"quality":"ultra"}}"#);
        assert_eq!(mgr.denoise.read().quality, denoise::DenoiseQuality::High);
    }
```

Note: match the exact `export_gyroflow_data` / `import_gyroflow_data` signatures used by the neighbouring colour round-trip test at `lib.rs:2371` — copy that test's call shape rather than the sketch above if it differs.

- [ ] **Step 2: Run tests to verify they fail**

Run: `just test denoise_`
Expected: FAIL — no `"denoise"` key is written, so the restored settings are all defaults.

- [ ] **Step 3: Build the JSON object**

In `src/core/lib.rs`, immediately after the `let color = { ... };` block that ends at `:1326`:

```rust
        let denoise = {
            let d = self.denoise.read();
            serde_json::json!({
                "enabled":         d.enabled,
                "quality":         d.quality.id(),
                "luma_strength":   d.luma_strength,
                "chroma_strength": d.chroma_strength,
            })
        };
```

- [ ] **Step 4: Insert it into the project object**

In the `serde_json::json!({ ... })` at `:1401`, directly below the `"color": color,` line:

```rust
            "denoise": denoise,
```

- [ ] **Step 5: Read it back**

In `src/core/lib.rs`, directly after the closing brace of the `if let Some(serde_json::Value::Object(obj)) = obj.get("color") { ... }` block that begins at `:1792`:

```rust
            if let Some(serde_json::Value::Object(obj)) = obj.get("denoise") {
                let mut d = self.denoise.write();
                if let Some(v) = obj.get("enabled").and_then(|x| x.as_bool()) { d.enabled = v; }
                // Clamped through set_param so a hand-edited or corrupted project
                // cannot produce a nonsensical sigma. The ranges live in
                // `DenoiseSettings::set_param`, shared with the QML entry point.
                for key in ["luma_strength", "chroma_strength"] {
                    if let Some(v) = obj.get(key).and_then(|x| x.as_f64()) { d.set_param(key, v); }
                }
                // An unknown id (an older or newer build, a hand-edited file) falls
                // back to the current value rather than failing the import.
                if let Some(v) = obj.get("quality").and_then(|x| x.as_str()) {
                    if let Some(q) = denoise::DenoiseQuality::from_id(v) { d.quality = q; }
                }
            }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `just test denoise_`
Expected: PASS, 3 tests.

- [ ] **Step 7: Commit**

```bash
git add src/core/lib.rs
git commit -m "feat(denoise): persist settings in the project file"
```

---

### Task 3: `TemporalWindow` ring buffer

**Files:**
- Create: `src/rendering/temporal_window.rs`
- Modify: `src/rendering/mod.rs` (add `pub mod temporal_window;` beside the other module declarations)

**Interfaces:**
- Consumes: nothing from earlier tasks (deliberately free of `ffmpeg` types so it can be tested headlessly).
- Produces: `TemporalWindow<T>` with `new(k: usize) -> Self`, `push(&mut self, ts: i64, item: T) -> Option<Emission<'_, T>>`, `drain(&mut self) -> Vec<i64>`, `radius() -> usize`; `Emission { centre_ts: i64, centre: &T, references: Vec<&T> }`.

The window is generic over the payload so the tests can drive it with `i64` markers instead of decoded `frame::Video`s.

- [ ] **Step 1: Write the failing tests**

Create `src/rendering/temporal_window.rs` with only a test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Drive the window with timestamps as payloads and collect what it emits.
    fn run(k: usize, count: i64) -> Vec<i64> {
        let mut w = TemporalWindow::new(k);
        let mut emitted = Vec::new();
        for ts in 0..count {
            if let Some(e) = w.push(ts, ts) { emitted.push(e.centre_ts); }
        }
        emitted.extend(w.drain());
        emitted
    }

    #[test]
    fn k_zero_emits_every_frame_immediately() {
        assert_eq!(run(0, 5), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn every_frame_is_emitted_exactly_once_and_in_order() {
        for k in 0..=3 {
            for count in 1..12 {
                let emitted = run(k, count);
                let expected: Vec<i64> = (0..count).collect();
                assert_eq!(emitted, expected, "k={k}, count={count}");
            }
        }
    }

    #[test]
    fn output_is_delayed_by_k_while_priming() {
        let mut w = TemporalWindow::new(2);
        // The first k pushes emit nothing at all.
        assert!(w.push(0, 0).is_none());
        assert!(w.push(1, 1).is_none());
        // The (k+1)th push emits the centre, which is frame 0.
        assert_eq!(w.push(2, 2).unwrap().centre_ts, 0);
        assert_eq!(w.push(3, 3).unwrap().centre_ts, 1);
    }

    #[test]
    fn drain_emits_exactly_the_trailing_k_frames() {
        let mut w = TemporalWindow::new(3);
        for ts in 0..10 { w.push(ts, ts); }
        // Frames 0..=6 came out during push; 7, 8, 9 are still held.
        assert_eq!(w.drain(), vec![7, 8, 9]);
    }

    #[test]
    fn drain_is_idempotent() {
        let mut w = TemporalWindow::new(2);
        for ts in 0..5 { w.push(ts, ts); }
        let first = w.drain();
        assert!(!first.is_empty());
        assert!(w.drain().is_empty(), "a second drain must not re-emit");
    }

    #[test]
    fn a_clip_shorter_than_the_window_still_emits_every_frame() {
        // Two frames with k=3 never fills the ring, so everything comes from drain.
        assert_eq!(run(3, 2), vec![0, 1]);
    }

    #[test]
    fn references_shrink_at_the_head_and_tail() {
        let mut w = TemporalWindow::new(1);
        // Frame 0 is emitted with only its forward neighbour available.
        let e = w.push(0, 0).map(|e| e.references.len());
        assert_eq!(e, None);
        let e = w.push(1, 1).unwrap();
        assert_eq!(e.centre_ts, 0);
        assert_eq!(e.references.len(), 1, "head frame has no backward reference");
        let e = w.push(2, 2).unwrap();
        assert_eq!(e.centre_ts, 1);
        assert_eq!(e.references.len(), 2, "interior frame sees both sides");
    }

    #[test]
    fn references_never_include_the_centre() {
        let mut w = TemporalWindow::new(1);
        w.push(0, 0);
        w.push(1, 1);
        let e = w.push(2, 2).unwrap();
        assert_eq!(*e.centre, 1);
        assert!(!e.references.iter().any(|r| **r == 1));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `just test temporal_window`
Expected: FAIL — `cannot find type TemporalWindow in this scope`.

- [ ] **Step 3: Write the implementation**

Prepend to `src/rendering/temporal_window.rs`:

```rust
// Copyright © 2026 Gyroflow

//! A fixed ring of decoded frames that hands back a centre frame plus its
//! temporal neighbours.
//!
//! The decoder is strictly forward-only, so a symmetric window is only possible
//! by delaying output: frame `N+1` does not exist when `N` arrives. Pushing
//! frame `N` therefore emits frame `N-k`, and the trailing `k` frames come out
//! of [`TemporalWindow::drain`] once decoding has finished.
//!
//! Frames at both ends of a clip get fewer references than the interior. That
//! is deliberate and standard for temporal filters — the alternative is either
//! trimming the clip or duplicating edge frames, and duplicating them biases the
//! average toward the edge frame's own noise.

use std::collections::VecDeque;

/// One frame ready to be processed, together with the neighbours it should be
/// denoised against.
pub struct Emission<'a, T> {
    pub centre_ts: i64,
    pub centre: &'a T,
    /// Neighbours in temporal order, excluding the centre. Between `0` and `2k`
    /// entries: fewer at the start and end of the clip.
    pub references: Vec<&'a T>,
}

pub struct TemporalWindow<T> {
    k: usize,
    ring: VecDeque<(i64, T)>,
    /// Index within `ring` of the frame to emit next. Grows during priming,
    /// then sticks at `k` once the ring is full.
    centre: usize,
    drained: bool,
}

impl<T> TemporalWindow<T> {
    pub fn new(k: usize) -> Self {
        Self { k, ring: VecDeque::with_capacity(2 * k + 1), centre: 0, drained: false }
    }

    pub fn radius(&self) -> usize { self.k }

    /// Push a newly decoded frame. Returns the frame that is now ready, which is
    /// `k` frames behind the one just pushed — or `None` while the window is
    /// still filling.
    pub fn push(&mut self, ts: i64, item: T) -> Option<Emission<'_, T>> {
        self.ring.push_back((ts, item));

        // Drop frames that have fallen out of the back of the window, so the ring
        // never holds more than 2k+1 and the centre index stays put.
        if self.ring.len() > 2 * self.k + 1 {
            self.ring.pop_front();
        } else if self.ring.len() > self.k + 1 {
            // Still filling the forward half; the centre walks forward with it.
            self.centre += 1;
        }

        // Nothing is ready until the forward half is populated.
        if self.ring.len() <= self.k { return None; }

        Some(Self::emit(&self.ring, self.centre))
    }

    /// Emit the frames still held after the decoder has finished, oldest first.
    /// Returns their timestamps in emission order.
    ///
    /// Callers that need the frames themselves should use [`TemporalWindow::drain_each`];
    /// this form exists because the render loop only needs to know how many
    /// frames remain and at what timestamps.
    pub fn drain(&mut self) -> Vec<i64> {
        let mut out = Vec::new();
        self.drain_each(|e| out.push(e.centre_ts));
        out
    }

    /// Emit each remaining frame in turn, oldest first. Idempotent: a second
    /// call after the window has been drained yields nothing.
    pub fn drain_each<F: FnMut(Emission<'_, T>)>(&mut self, mut f: F) {
        if self.drained { return; }
        self.drained = true;
        // Everything after the current centre is still unemitted.
        let mut c = self.centre + 1;
        while c < self.ring.len() {
            f(Self::emit(&self.ring, c));
            c += 1;
        }
    }

    fn emit(ring: &VecDeque<(i64, T)>, centre: usize) -> Emission<'_, T> {
        let (centre_ts, ref centre_item) = ring[centre];
        let references = ring.iter()
            .enumerate()
            .filter(|(i, _)| *i != centre)
            .map(|(_, (_, item))| item)
            .collect();
        Emission { centre_ts, centre: centre_item, references }
    }
}
```

- [ ] **Step 4: Register the module**

In `src/rendering/mod.rs`, add beside the other module declarations:

```rust
pub mod temporal_window;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `just test temporal_window`
Expected: PASS, 8 tests.

If `every_frame_is_emitted_exactly_once_and_in_order` fails for a specific `(k, count)` where `count <= k`, the priming branch in `push` needs the ring length compared against `self.k + 1` rather than `self.k`; fix it against the failing case rather than by guessing.

- [ ] **Step 6: Commit**

```bash
git add src/rendering/temporal_window.rs src/rendering/mod.rs
git commit -m "feat(denoise): temporal frame window with delayed emission"
```

---

### Task 4: Drain hook in the decode loop

**Files:**
- Modify: `src/rendering/ffmpeg_video.rs:70` (add the field), `:470-484` (call the hook after the loop)
- Modify: `src/rendering/ffmpeg_processor.rs:602` (expose a setter beside `on_frame`)
- Test: `src/rendering/temporal_window.rs` covers the ring; this task's own verification is a real render, described in Step 5.

**Interfaces:**
- Consumes: `TemporalWindow` from Task 3.
- Produces: `VideoTranscoder::on_finish_callback: Option<Box<dyn FnMut(&mut Converter, &mut RateControl) -> Result<(), FFmpegError> + 'a>>` and `FfmpegProcessor::on_finish<F>(&mut self, cb: F)`.

This is the one genuinely invasive change to existing code. The decode loop at `ffmpeg_video.rs:470` breaks and returns `Ok(status)` with no further callback, so a window holding `k` frames would drop them silently — the failure mode is a rendered file `k` frames short, which is easy to miss.

- [ ] **Step 1: Add the callback field**

In `src/rendering/ffmpeg_video.rs`, directly below the `on_frame_callback` field at `:70`:

```rust
    /// Called once after the decode loop ends, so a processor holding frames back
    /// (temporal denoising delays output by `k`) can emit its tail. Without this,
    /// those frames are silently dropped and the output is `k` frames short.
    pub on_finish_callback: Option<Box<dyn FnMut(&mut Converter, &mut RateControl) -> Result<(), FFmpegError> + 'a>>,
```

Add `on_finish_callback: None` to the same initialiser that sets `on_frame_callback: None`.

- [ ] **Step 2: Call it after the loop**

In `src/rendering/ffmpeg_video.rs`, between the end of the decode loop (the closing brace after the `break` at `:471`) and `Ok(status)` at `:483`:

```rust
        // Emit any frames a delayed processor is still holding. Only on a clean
        // finish: if the user cancelled or an error broke the loop, the tail
        // frames belong to a render that is not happening.
        if status == Status::Finish {
            if let Some(ref mut cb) = self.on_finish_callback {
                let mut rate_control = RateControl::default();
                cb(&mut self.converter, &mut rate_control)?;
            }
        }
```

Check the surrounding code for how `status` is named and compared at `:470`; use the same form.

- [ ] **Step 3: Expose it on the processor**

In `src/rendering/ffmpeg_processor.rs`, directly below the `on_frame` method at `:602`:

```rust
    pub fn on_finish<F>(&mut self, cb: F) where F: FnMut(&mut ffmpeg_video_converter::Converter, &mut ffmpeg_video::RateControl) -> Result<(), FFmpegError> + 'a {
        self.video.on_finish_callback = Some(Box::new(cb));
    }
```

- [ ] **Step 4: Build**

Run: `just build`
Expected: compiles. A borrow-checker complaint about `self.converter` being borrowed alongside `self.on_finish_callback` is likely; resolve it the same way the existing `on_frame_callback` call at `:259` does — take the callback out of `self` with `.take()`, call it, then put it back.

- [ ] **Step 5: Verify against a real render**

Run a short export from the CLI on a clip in `testfiles/` and confirm the output frame count equals the input frame count. With denoise not yet wired in, `on_finish_callback` is `None` and this is a regression check that the hook changed nothing:

```bash
just run -- --help
```

Then export a test clip and compare frame counts with `ffprobe -count_frames`.
Expected: input and output frame counts match exactly, as they did before this task.

- [ ] **Step 6: Commit**

```bash
git add src/rendering/ffmpeg_video.rs src/rendering/ffmpeg_processor.rs
git commit -m "feat(rendering): add an end-of-decode hook for delayed processors"
```

---

### Task 5: `KernelParams` denoise fields

**Files:**
- Modify: `src/core/stabilization/mod.rs:120-170` (the struct)
- Modify: `src/core/gpu/wgpu_undistort.wgsl:9` (the mirrored struct)
- Modify: `src/core/gpu/opencl_undistort.cl` (the mirrored struct)
- Modify: `src/qt_gpu/undistort.frag` (the mirrored struct)
- Modify: `src/core/gpu/stabilize_spirv/src/types.rs` (the mirrored struct)

**Interfaces:**
- Consumes: nothing.
- Produces: two new `KernelParams` fields — `reference_count: i32` (total frames gathered, `2k+1`; `1` means denoise off) and `denoise_sigma: f32` (the rejection sigma for the plane being processed).

`reserved1` and `reserved2` already exist at `stabilization/mod.rs:161-162` as `f32` in a 16-byte-aligned group. Reuse them rather than growing the struct, which keeps every mirrored copy byte-identical with no realignment.

- [ ] **Step 1: Repurpose the reserved fields in Rust**

In `src/core/stabilization/mod.rs`, replace the two reserved lines:

```rust
    pub reserved1:                f32, // 12
    pub reserved2:                f32, // 16
```

with:

```rust
    /// Total source frames gathered per output pixel: `2k+1`. `1` disables the
    /// temporal path entirely — every kernel branches on this before doing any
    /// extra work, so `k = 0` costs nothing.
    /// Stored as f32 to keep this slot byte-identical to the `reserved1` it
    /// replaces; the kernels round it back to an integer count.
    pub reference_count:          f32, // 12
    /// Sigma of the rejection Gaussian, in normalised pixel units, for the plane
    /// currently being processed. Selected per plane by `plane_index`.
    pub denoise_sigma:            f32, // 16
```

- [ ] **Step 2: Mirror in the four kernel copies**

In each of `wgpu_undistort.wgsl`, `opencl_undistort.cl`, `undistort.frag` and `stabilize_spirv/src/types.rs`, find the two reserved members in the `KernelParams` struct and rename them to `reference_count` and `denoise_sigma`, keeping their existing type (`f32` / `float`) and position exactly. Do not reorder, do not change types — the byte layout must not move.

- [ ] **Step 3: Fix every construction site**

Run: `just build`
Expected: errors at each place that names `reserved1` / `reserved2`. Update each to the new names. If `KernelParams` is built with `..Default::default()` there may be none.

- [ ] **Step 4: Assert the layout did not move**

Add to the `mod tests` block in `src/core/stabilization/mod.rs` (create one if absent):

```rust
    #[test]
    fn kernel_params_layout_is_unchanged() {
        // Mirrored byte-for-byte in opencl_undistort.cl, wgpu_undistort.wgsl,
        // qt_gpu/undistort.frag and stabilize_spirv/src/types.rs. If this size
        // changes, all four must change with it.
        assert_eq!(std::mem::size_of::<KernelParams>() % 16, 0, "struct must stay 16-byte aligned");
        assert_eq!(std::mem::align_of::<KernelParams>(), 4);
    }
```

Pin the exact size too, so a later field addition cannot silently desynchronise the five copies. Get the number by letting the assertion fail once — write it as `0`, run the test, and copy the actual value the failure prints into place:

```rust
        // Pinned so a new field cannot be added to one copy and not the other four.
        assert_eq!(std::mem::size_of::<KernelParams>(), 0); // replace 0 with the value this prints on first run
```

Because this task only *renames* two existing fields, that number must equal the size on the parent commit. Verify with `git stash && just test kernel_params_layout && git stash pop` — if the size differs, the rename changed the layout and something was typed wrong.

- [ ] **Step 5: Run the tests**

Run: `just test kernel_params_layout`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/core/stabilization/mod.rs src/core/gpu/wgpu_undistort.wgsl src/core/gpu/opencl_undistort.cl src/qt_gpu/undistort.frag src/core/gpu/stabilize_spirv/src/types.rs
git commit -m "feat(denoise): reserve kernel params for the temporal path"
```

---

### Task 6: WGSL multi-frame gather

**Files:**
- Modify: `src/core/gpu/wgpu_undistort.wgsl:560-660` (`undistort_coord`), `:663-738` (`undistort`), `:63-67` (bindings)

**Interfaces:**
- Consumes: `KernelParams::reference_count`, `KernelParams::denoise_sigma` from Task 5.
- Produces: WGSL `undistort_coord(position: vec2<f32>, frame_base: u32) -> vec2<f32>`, `sample_bilinear(uv: vec2<f32>, frame: u32) -> vec4<f32>`, and `gather_denoised(position: vec2<f32>, centre: vec4<f32>) -> vec4<f32>`.

Matrix blocks are 14 floats. Frame `f`'s block starts at `f * matrix_count * 14`, so the existing per-scanline index simply gains that offset.

- [ ] **Step 1: Give `undistort_coord` a frame offset**

Change the signature at `:561`:

```wgsl
fn undistort_coord(position: vec2<f32>, frame_base: u32) -> vec2<f32> {
```

Inside it, the three `idx` computations become frame-relative. At `:612`:

```wgsl
        let idx: u32 = frame_base + u32((params.matrix_count / 2) * 14); // Use middle matrix
```

At `:624`:

```wgsl
    let idx: u32 = frame_base + min(sy, u32(params.matrix_count - 1)) * 14u;
```

- [ ] **Step 2: Update the existing call sites**

In `undistort` at `:682` and `:686-687`, pass `0u` — the centre frame is always block zero:

```wgsl
    var uv = undistort_coord(position, 0u);
```

```wgsl
        let xyx = undistort_coord(position + vec2<f32>(eps, 0.0), 0u) - uv;
        let xyy = undistort_coord(position + vec2<f32>(0.0, eps), 0u) - uv;
```

- [ ] **Step 3: Add the reference-frame bindings**

Replace the binding-5 declarations at `:63-67`. The texture path gains an array layer dimension; the buffer path keeps one buffer and indexes it by frame stride:

```wgsl
#ifdef TEXTURE_PATH
@group(0) @binding(5) @fragment var input_texture: texture_2d_array<SCALAR>;
#else
// All reference frames concatenated. Frame f starts at f * params.height * params.stride.
@group(0) @binding(5) @fragment var<storage, read> input_buffer: array<SCALAR>;
@group(0) @binding(6) @fragment var<storage, read_write> output_buffer: array<SCALAR>;
#endif
```

Match the file's existing conditional-compilation style — the shader is preprocessed by `shader_builder`, so use whatever directive form the surrounding code already uses rather than the `#ifdef` sketched here.

Every existing read of `input_texture` must gain a layer index of `0` for the centre frame; every existing read of `input_buffer` must gain the frame's base offset. Find them with:

```bash
grep -n "input_texture\|input_buffer" src/core/gpu/wgpu_undistort.wgsl
```

- [ ] **Step 4: Add the gather**

Add above `undistort` at `:663`:

```wgsl
// Combine the centre sample with its temporal neighbours.
//
// Each reference frame has its own matrix block, so running the same
// output->source mapping against block f gives exactly where this output pixel
// lands in frame f. That is motion compensation for free: the rotation came
// from the gyro, not from an estimator.
//
// Samples that disagree with the centre are rejected by a Gaussian on their
// difference. This is load-bearing rather than a refinement - rotation aligns
// frames for rotational camera motion only, so translation creates
// depth-dependent parallax that nothing here can compensate, and moving
// subjects never align at all.
fn gather_denoised(position: vec2<f32>, centre: vec4<f32>) -> vec4<f32> {
    let count = u32(round(params.reference_count));
    if (count <= 1u) { return centre; }

    let sigma = params.denoise_sigma * params.max_pixel_value;
    let inv_2sigma_sq = 1.0 / (2.0 * sigma * sigma);

    var acc: vec4<f32> = centre;
    var weight_sum: f32 = 1.0;

    let block = u32(params.matrix_count) * 14u;
    for (var f: u32 = 1u; f < count; f = f + 1u) {
        let ruv = undistort_coord(position, f * block);
        if (ruv.x < -99998.0) { continue; }

        // References are sampled bilinearly regardless of `interpolation`.
        // At k=3 the configured sampler would otherwise run seven times per
        // pixel, and the extra accuracy is wasted on samples that are about to
        // be averaged anyway.
        let s = sample_bilinear(ruv, f);

        let d = s - centre;
        let dist_sq = dot(d, d);
        let w = exp(-dist_sq * inv_2sigma_sq);

        acc = acc + s * w;
        weight_sum = weight_sum + w;
    }

    return acc / weight_sum;
}
```

- [ ] **Step 5: Add the bilinear sampler for reference frames**

The file already samples bilinearly for `interpolation == 2`; find that code with `grep -n "bilinear\|interpolation == 2" src/core/gpu/wgpu_undistort.wgsl` and factor the existing body into:

```wgsl
fn sample_bilinear(uv: vec2<f32>, frame: u32) -> vec4<f32>
```

taking the frame index so it reads from the right texture layer or buffer offset. Reuse the existing arithmetic verbatim — do not write a second bilinear implementation, or the centre and reference samples will disagree at sub-pixel level and the rejection weights will be wrong everywhere.

- [ ] **Step 6: Call the gather**

In `undistort`, after `pixel` has been computed from the centre frame and before it is returned, wrap it:

```wgsl
    pixel = gather_denoised(position, pixel);
```

Place this *before* any colour stage (`apply_color`) so denoising happens on the source values, not on graded ones — grading changes the noise distribution the sigma was calibrated against. Confirm the ordering by reading the tail of `undistort`.

- [ ] **Step 7: Build**

Run: `just build`
Expected: compiles, and the shader validates. `naga` reports WGSL errors at build time with line numbers.

- [ ] **Step 8: Verify k=0 is unchanged**

Render a short clip with denoise disabled and confirm the output is byte-identical to a render from the previous commit:

```bash
ffmpeg -i before.mp4 -i after.mp4 -filter_complex "[0:v][1:v]psnr" -f null -
```

Expected: `inf` PSNR — identical. If not, `reference_count` is not defaulting to `1`.

- [ ] **Step 9: Commit**

```bash
git add src/core/gpu/wgpu_undistort.wgsl
git commit -m "feat(denoise): gather reference frames in the wgpu kernel"
```

---

### Task 7: wgpu reference frame upload

**Files:**
- Modify: `src/core/gpu/wgpu.rs:178` (`params_size`), `:309` (`buf_matrices`), `:321-355` (bind group layout), `:409-452` (bind group), `:466-500` (`undistort_image`)
- Modify: `src/core/gpu/mod.rs:241-260` (`BufferDescription` / `Buffers`)
- Modify: `src/core/stabilization/mod.rs:291` (`get_frame_transform_at`), `:656` (`process_pixels`)

**Interfaces:**
- Consumes: `FrameTransform { matrices: Vec<[f32; 14]>, .. }`, `Buffers`, Task 6's shader.
- Produces: `Buffers::references: Vec<BufferDescription<'a>>` (empty means no denoising); `FrameTransform::reference_matrices: Vec<Vec<[f32; 14]>>`.

- [ ] **Step 1: Carry reference frames on `Buffers`**

In `src/core/gpu/mod.rs`, add to `Buffers` at `:248`:

```rust
    /// Temporal reference frames, in temporal order, excluding the centre frame
    /// which is `input`. Empty when denoising is off.
    pub references: Vec<BufferDescription<'a>>,
```

Add `references: Vec::new()` to its `Default` impl. Include the reference frame count in `Buffers::get_checksum` at `:367`, so a change in reference count forces the OpenCL context to rebuild:

```rust
        hasher.write_usize(self.references.len());
```

- [ ] **Step 2: Carry reference matrices on `FrameTransform`**

In `src/core/stabilization/frame_transform.rs:12`, add:

```rust
    /// One matrix block per reference frame, in the same order as
    /// `Buffers::references`. Each block has the same length as `matrices`.
    pub reference_matrices: Vec<Vec<[f32; 14]>>,
```

Initialise it to `Vec::new()` at every construction site the compiler flags.

- [ ] **Step 3: Grow the matrix buffer**

In `src/core/gpu/wgpu.rs:178`, `params_size` currently sizes for one frame's matrices. Multiply by the maximum reference count so the buffer never needs reallocating mid-render:

```rust
        // 7 = the largest 2k+1 any quality tier asks for (DenoiseQuality::High).
        const MAX_REFERENCE_FRAMES: usize = 7;
        let params_size = (max_matrix_count * MAX_REFERENCE_FRAMES * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
```

- [ ] **Step 4: Upload the reference matrices**

In `undistort_image` at `:467`, the upload is currently:

```rust
        let matrices = bytemuck::cast_slice(&itm.matrices);
```

Replace with a concatenation so the centre block is first and each reference follows:

```rust
        // Block layout the shader indexes into: frame f starts at
        // f * matrix_count * 14 floats. Centre first, then references in
        // temporal order.
        let mut all_matrices: Vec<[f32; 14]> = Vec::with_capacity(itm.matrices.len() * (1 + itm.reference_matrices.len()));
        all_matrices.extend_from_slice(&itm.matrices);
        for block in &itm.reference_matrices {
            debug_assert_eq!(block.len(), itm.matrices.len(), "every reference block must match the centre block length or the shader's stride arithmetic is wrong");
            all_matrices.extend_from_slice(block);
        }
        let matrices = bytemuck::cast_slice(&all_matrices);
```

- [ ] **Step 5: Bind the reference frames**

For the **texture** path, change binding 5's layout entry at `:334` from `view_dimension: wgpu::TextureViewDimension::D2` to `D2Array`, create the input texture with `depth_or_array_layers: 1 + references.len()`, and copy each reference into its own layer.

For the **buffer** path, binding 5 at `:346` already takes one storage buffer. Size it to `in_size * (1 + references.len())` and write each reference at offset `i * in_size`, so the shader's `f * params.height * params.stride` stride arithmetic lands correctly.

Set the params before dispatch:

```rust
        params.reference_count = (1 + buffers.references.len()) as f32;
```

- [ ] **Step 6: Populate the reference matrices**

In `src/core/stabilization/mod.rs`, `get_frame_transform_at` at `:291` already computes a `FrameTransform` for one timestamp. Add a wrapper that fills the reference blocks by calling it again per neighbour:

```rust
    /// Build the centre transform plus one matrix block per reference timestamp.
    /// Each block comes from the same `get_frame_transform_at` the centre frame
    /// uses, so a reference frame's rolling-shutter matrices are computed
    /// exactly as that frame's own render would compute them.
    pub fn get_frame_transform_with_references<T: PixelType>(&self, timestamp_us: i64, reference_timestamps: &[i64], frame: Option<usize>, buffers: &Buffers) -> FrameTransform {
        let mut t = self.get_frame_transform_at::<T>(timestamp_us, frame, buffers);
        t.reference_matrices = reference_timestamps.iter()
            .map(|ts| self.get_frame_transform_at::<T>(*ts, None, buffers).matrices)
            .collect();
        t
    }
```

- [ ] **Step 7: Build and verify k=0 is still unchanged**

Run: `just build`, then repeat Task 6 Step 8's PSNR check.
Expected: compiles; PSNR `inf` with denoise disabled.

- [ ] **Step 8: Commit**

```bash
git add src/core/gpu/wgpu.rs src/core/gpu/mod.rs src/core/stabilization/mod.rs src/core/stabilization/frame_transform.rs
git commit -m "feat(denoise): upload reference frames and matrices to wgpu"
```

---

### Task 8: OpenCL mirror

**Files:**
- Modify: `src/core/gpu/opencl_undistort.cl`
- Modify: `src/core/gpu/opencl.rs`

**Interfaces:**
- Consumes: everything from Tasks 5-7.
- Produces: the same behaviour on the OpenCL backend.

- [ ] **Step 1: Give `undistort_coord` a frame offset**

In `opencl_undistort.cl`, find `undistort_coord` and add a `uint frame_base` parameter. Offset all three matrix index computations by it — the same three the WGSL has, identifiable by the `* 14` stride:

```c
    // Middle matrix, used to resolve the rolling-shutter scanline
    uint idx = frame_base + (uint)((params->matrix_count / 2) * 14);
```

```c
    // Per-scanline matrix for this pixel
    uint idx = frame_base + min(sy, (uint)(params->matrix_count - 1)) * 14u;
```

Update every existing call to pass `0u` for the centre frame.

- [ ] **Step 2: Factor out the bilinear sampler**

Find the existing bilinear branch (`grep -n "bilinear\|interpolation == 2" src/core/gpu/opencl_undistort.cl`) and lift its body into:

```c
float4 sample_bilinear(__global const SCALAR* input, float2 uv, uint frame, __constant KernelParams* params)
```

reading from `frame * params->height * params->stride`. Reuse the existing arithmetic verbatim rather than writing a second implementation — if the centre and reference samplers disagree at sub-pixel level, every rejection weight is computed against a slightly wrong centre and the filter misbehaves everywhere.

- [ ] **Step 3: Add the gather**

```c
// Mirror of gather_denoised in wgpu_undistort.wgsl and of
// denoise::rejection_weight in Rust. All three must agree: the GPU tests
// compare this kernel's output against the wgpu kernel's.
float4 gather_denoised(__global const SCALAR* input, float2 position, float4 centre, __constant KernelParams* params) {
    uint count = (uint)round(params->reference_count);
    if (count <= 1u) { return centre; }

    float sigma = params->denoise_sigma * params->max_pixel_value;
    float inv_2sigma_sq = 1.0f / (2.0f * sigma * sigma);

    float4 acc = centre;
    float weight_sum = 1.0f;

    uint block = (uint)params->matrix_count * 14u;
    for (uint f = 1u; f < count; f++) {
        float2 ruv = undistort_coord(position, f * block);
        if (ruv.x < -99998.0f) { continue; }

        float4 s = sample_bilinear(input, ruv, f, params);
        float4 d = s - centre;
        float dist_sq = dot(d, d);
        float w = native_exp(-dist_sq * inv_2sigma_sq);

        acc += s * w;
        weight_sum += w;
    }
    return acc / weight_sum;
}
```

Call it in `undistort` at the same point the WGSL does — after the centre pixel is resolved, before any colour stage.

**Watch out:** `native_exp` is faster but lower precision than `exp`. If the Task 13 cross-backend comparison fails on tolerance, switch to `exp` before assuming the logic is wrong.

- [ ] **Step 4: Mirror the host-side upload**

In `opencl.rs`, apply the same matrix concatenation as Task 7 Step 4 — centre block first, then each reference block, all contiguous. Set `params.reference_count = (1 + buffers.references.len()) as f32` before enqueueing.

References go in as offsets into one enlarged input buffer, matching the `frame * height * stride` stride the kernel in Step 2 reads. Size the input buffer to `in_size * (1 + references.len())` and write each reference at `i * in_size`.

`Buffers::get_checksum` now includes the reference count (Task 7 Step 1), which matters here specifically: `opencl.rs:218` rebuilds the context when the checksum changes, so switching quality tiers correctly forces a rebuild rather than reusing a context sized for the old reference count.

- [ ] **Step 5: Build with the OpenCL feature**

Run: `just build --features opencl`
Expected: compiles, and the kernel compiles at runtime.

- [ ] **Step 6: Verify k=0 is unchanged on OpenCL**

Repeat Task 6 Step 8's PSNR check with the OpenCL backend selected.
Expected: PSNR `inf`.

- [ ] **Step 7: Commit**

```bash
git add src/core/gpu/opencl_undistort.cl src/core/gpu/opencl.rs
git commit -m "feat(denoise): mirror the temporal gather in the OpenCL kernel"
```

---

### Task 9: Wire the window into the render loop

**Files:**
- Modify: `src/rendering/mod.rs:409` (processor setup), `:895-905` (`undistort_frame`), `:899` (progress), `:640-670` (rate control)

**Interfaces:**
- Consumes: `TemporalWindow` (Task 3), `on_finish` (Task 4), `Buffers::references` (Task 7), `DenoiseSettings::radius_for_size` (Task 1).
- Produces: denoised exports.

- [ ] **Step 1: Create the window**

In `src/rendering/mod.rs`, before the `on_frame` callback is installed near `:409`:

```rust
    let denoise_k = stab.denoise.read().radius_for_size(input_size.0, input_size.1);
    let mut window = temporal_window::TemporalWindow::new(denoise_k);
```

Use whatever the surrounding code already calls the input dimensions; check the names near `:409` rather than assuming `input_size`.

- [ ] **Step 2: Extend `undistort_frame` to take references**

`undistort_frame` at `:895` is a closure taking `(frame, out_frame)` and calling each plane closure in turn. Give it the reference frames and the centre timestamp, since neither is implicit any more once output is delayed:

```rust
        let mut undistort_frame = |frame: &mut Video, references: &[&Video], centre_ts: i64, out_frame: &mut Video| {
            for (i, cb) in planes.iter_mut().enumerate() {
                // The plane closure builds the `Buffers`; `references` becomes
                // `Buffers::references`, and an empty slice leaves
                // reference_count at 1, i.e. the existing undenoised path.
                (*cb)(centre_ts, frame, references, out_frame, i, fill_with_background);
            }
        };
```

The `create_planes_proc!` macro at `:881` generates those closures; add the `references` parameter to the macro's closure signature and pass it into the `Buffers` it constructs, alongside the existing `input` and `output`. Note that `progress2` moves out of this closure in Step 3.

- [ ] **Step 3: Push through the window in the callback**

Inside the `on_frame` closure, replace the direct call to `undistort_frame` with a push. When the window returns nothing, suppress the encode:

```rust
        // Delayed by k frames so each output frame sees both its past and its
        // future neighbours. `repeat_times = 0` is the existing lever the
        // frame-rate conversion path uses at :652 to skip encoding a frame.
        match window.push(timestamp_us, input_frame.clone()) {
            Some(emission) => {
                rate_control.out_timestamp_us = emission.centre_ts;
                undistort_frame(emission.centre, &emission.references, emission.centre_ts, output_frame);
                process_frame += 1;
                progress2((process_frame as f64 / render_frame_count as f64, process_frame, render_frame_count, false, false));
            }
            None => {
                rate_control.repeat_times = 0;
            }
        }
```

Two things will not compile as written, both requiring a decision rather than a guess:

- `emission.centre` is `&Video` but `undistort_frame` wants `&mut Video`. The window hands out shared references because the centre frame is also a reference frame for its neighbours. Either make `undistort_frame` take `&Video` (it only reads the input), or have `Emission` expose the centre mutably; prefer the former, since the input frame genuinely is read-only here.
- If `frame::Video` is not `Clone`, the window must own its frames another way — allocate a fresh `frame::Video` per push and copy into it, reusing the allocation pattern at `ffmpeg_video.rs:251`.

- [ ] **Step 4: Fix progress reporting**

`progress2` at `:899` currently sits inside `undistort_frame` and divides `process_frame` (decoded frames) by `render_frame_count`. With output delayed it would run `k` frames ahead of what has actually been encoded, reaching 100% while `k` frames are still pending.

Remove the `progress2` call from `undistort_frame` (Step 2 already drops it) and place it where Step 3 shows — in the `Some(emission)` arm — plus in the drain callback in Step 5. Progress then counts *emitted* frames, which is what the user is waiting for.

- [ ] **Step 5: Install the drain**

After the `on_frame` call near `:409`:

```rust
    proc.on_finish(move |_converter, rate_control| {
        // The last k frames are still in the ring. Without this they are
        // silently dropped and the file is k frames short - a failure that is
        // very easy to miss, because the video still plays.
        window.borrow_mut().drain_each(|emission| {
            rate_control.out_timestamp_us = emission.centre_ts;
            undistort_frame(emission.centre, &emission.references, emission.centre_ts, output_frame);
            process_frame += 1;
            progress2((process_frame as f64 / render_frame_count as f64, process_frame, render_frame_count, false, false));
        });
        Ok(())
    });
```

`window` is borrowed by both closures, which will not compile as written. Wrap it in `Rc<RefCell<_>>` and clone the handle into each, matching how the surrounding code shares mutable state across callbacks — `process_frame` and `progress2` are already shared this way, so follow whatever they do.

Note the drain writes into the same `output_frame` and expects the encoder still to be alive. Confirm against Task 4 Step 2 that `on_finish_callback` fires *before* the encoder is flushed, not after; if it fires after, the drained frames go nowhere and Step 6 will catch it.

- [ ] **Step 6: Verify the frame count**

Export a clip from `testfiles/` with denoise enabled at High quality and compare frame counts:

```bash
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 input.mp4
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 output.mp4
```

Expected: identical counts. A shortfall of exactly `k` means the drain is not running.

- [ ] **Step 7: Verify progress reaches 100%**

Run the same export through the UI and watch the progress bar.
Expected: reaches 100% at completion, not `(n-k)/n`.

- [ ] **Step 8: Commit**

```bash
git add src/rendering/mod.rs
git commit -m "feat(denoise): denoise exports through the temporal window"
```

---

### Task 10: Controller and QML UI

**Files:**
- Create: `src/ui/menu/Denoise.qml`
- Modify: `src/ui/menu/qmldir`, `src/controller.rs:81-93` (method declarations), `:1251` (setter neighbourhood), `src/resources_qml.rs` if QML files are listed there

**Interfaces:**
- Consumes: `DenoiseSettings::set_param`, `DenoiseQuality::{id, from_id}` (Task 1).
- Produces: `Controller::set_denoise_param(name, val)`, `set_denoise_enabled(enabled)`, `set_denoise_quality(id)`, `get_denoise_quality() -> QString`.

- [ ] **Step 1: Declare the QML methods**

In `src/controller.rs`, beside the colour declarations at `:81-93`:

```rust
    set_denoise_param: qt_method!(fn(&self, name: QString, val: f64)),
    set_denoise_enabled: qt_method!(fn(&self, enabled: bool)),
    set_denoise_quality: qt_method!(fn(&self, id: QString)),
    get_denoise_quality: qt_method!(fn(&self) -> QString),
```

- [ ] **Step 2: Implement them**

Beside `set_color_param` at `:1251`:

```rust
    fn set_denoise_param(&mut self, name: QString, val: f64) {
        let name = name.to_string();
        if !self.stabilizer.denoise.write().set_param(&name, val) {
            ::log::warn!("Rejected denoise parameter: {name} = {val}");
            return;
        }
        self.request_recompute();
    }

    fn set_denoise_enabled(&mut self, enabled: bool) {
        self.stabilizer.denoise.write().enabled = enabled;
        self.request_recompute();
    }

    fn set_denoise_quality(&mut self, id: QString) {
        let id = id.to_string();
        match gyroflow_core::denoise::DenoiseQuality::from_id(&id) {
            Some(q) => { self.stabilizer.denoise.write().quality = q; }
            None => { ::log::warn!("Unknown denoise quality: {id}"); return; }
        }
        self.request_recompute();
    }

    fn get_denoise_quality(&self) -> QString {
        QString::from(self.stabilizer.denoise.read().quality.id())
    }
```

`request_recompute()` is the live invalidation path — `set_color_param` calls it at `:1257` and the preview will not refresh without it.

- [ ] **Step 3: Build the QML panel**

Create `src/ui/menu/Denoise.qml` modelled on `src/ui/menu/ColorAdjustments.qml`. Read that file first and copy its `MenuItem` wrapper, its `SliderWithField` usage and its property-binding style rather than inventing a layout. The panel needs:

- A checkbox bound to `set_denoise_enabled`.
- A `ComboBox` with entries "Fast" and "High", writing `"fast"` / `"high"` through `set_denoise_quality`, initialised from `get_denoise_quality()`.
- Two sliders, 0..1, calling `set_denoise_param("luma_strength", value)` and `set_denoise_param("chroma_strength", value)`.
- A short note that denoising appears in the preview only when playback is paused, so a user who sees no change while playing is not left guessing.

Register it in `src/ui/menu/qmldir` following the existing entries, and add it to `src/resources_qml.rs` if QML files are enumerated there (check how `ColorAdjustments.qml` appears).

- [ ] **Step 4: Run the app**

Run: `just run`
Expected: the Denoise panel appears, all controls move, and no QML errors appear in `gyroflow.log`. Note that `gyroflow.log` is truncated on each launch — copy it aside before relaunching if you need to compare.

- [ ] **Step 5: Commit**

```bash
git add src/ui/menu/Denoise.qml src/ui/menu/qmldir src/controller.rs src/resources_qml.rs
git commit -m "feat(denoise): denoise panel and controller bindings"
```

---

### Task 11: Pause-only preview

**Files:**
- Modify: `src/controller.rs:1150` and `:1179` (the two `process_pixels` callbacks)

**Interfaces:**
- Consumes: `Buffers::references` (Task 7), `VideoProcessor::from_file` (the pattern at `controller.rs:524`).
- Produces: a paused preview that matches the export exactly.

The mdk player owns decoding and seeks arbitrarily, so no forward window exists during playback. Denoise therefore renders only when paused, fetching the true ±k window with a second decoder.

- [ ] **Step 1: Add a reference fetcher**

Add to `src/controller.rs`:

```rust
/// The ±k source frames around `timestamp_us`, fetched with a second decoder.
///
/// The mdk player owns preview decoding and seeks arbitrarily, so there is no
/// forward window during playback. Pausing is the only moment a symmetric
/// window can be assembled, and it is also the only moment the user is actually
/// judging the result.
struct ReferenceCache {
    /// Keyed by the centre timestamp and radius. The preview re-renders on every
    /// unrelated slider move, and re-seeking on each one would make the UI
    /// unusable, so a hit here is the common case while tuning.
    key: Option<(i64, usize)>,
    frames: Vec<Vec<u8>>,
}

impl ReferenceCache {
    fn get(&mut self, url: &str, timestamp_us: i64, k: usize, frame_interval_us: i64) -> &[Vec<u8>] {
        let key = (timestamp_us, k);
        if self.key == Some(key) { return &self.frames; }

        self.frames.clear();
        if k > 0 {
            let start = (timestamp_us - k as i64 * frame_interval_us).max(0);
            // Same construction as the autosync path at controller.rs:524.
            if let Ok(mut proc) = VideoProcessor::from_file(url, false, 0, None) {
                let wanted = 2 * k + 1;
                let mut collected = Vec::with_capacity(wanted);
                let _ = proc.decode_only_range(start, wanted, |ts, frame| {
                    // Skip the centre frame: the player already supplied it as
                    // `input`, and including it twice would weight it double.
                    if ts != timestamp_us { collected.push(frame.to_owned()); }
                });
                self.frames = collected;
            }
        }
        self.key = Some(key);
        &self.frames
    }
}
```

`decode_only_range` does not exist yet. `FfmpegProcessor` already supports `decode_only` (`ffmpeg_processor.rs:533`) and seeking; add a thin wrapper beside it that seeks to a timestamp, runs the decode loop, and stops after `n` frames. Read how the autosync path at `controller.rs:524` drives its `decode_only` processor and follow that shape — it is the same job.

If the decoder cannot be opened or the seek lands short, `frames` stays empty and denoising silently does not apply to the preview. That is the right failure: a preview that shows the undenoised image is recoverable, a preview that shows a wrong image is not.

- [ ] **Step 2: Use it only when paused**

In both `onProcessPixels` callbacks, populate `buffers.references` only when the player is paused and `denoise.radius_for_size(..) > 0`. During playback leave it empty, which sets `reference_count = 1` and takes the existing undenoised path at zero cost.

- [ ] **Step 3: Verify preview matches export**

Pause on a frame with denoise at High, screenshot the preview, then export a single frame at that timestamp and compare.
Expected: visually identical. A difference means the preview and the render disagree about which reference timestamps to use.

- [ ] **Step 4: Verify playback is not slowed**

Play a clip with denoise enabled.
Expected: playback framerate matches denoise-disabled playback — the references are never fetched while playing.

- [ ] **Step 5: Commit**

```bash
git add src/controller.rs
git commit -m "feat(denoise): show denoised preview when paused"
```

---

### Task 12: CLI flags

**Files:**
- Modify: `src/cli.rs`

**Interfaces:**
- Consumes: `DenoiseSettings::set_param`, `DenoiseQuality::from_id` (Task 1).
- Produces: `--denoise <fast|high>`, `--denoise-luma <0..1>`, `--denoise-chroma <0..1>`.

- [ ] **Step 1: Add the arguments**

In `src/cli.rs`, add to the `argh`-derived options struct, matching the doc-comment style of the neighbouring fields:

```rust
    /// enable temporal denoising at the given quality: fast or high
    #[argh(option)]
    denoise: Option<String>,

    /// denoise luma strength, 0.0 to 1.0 (default 0.5)
    #[argh(option)]
    denoise_luma: Option<f64>,

    /// denoise chroma strength, 0.0 to 1.0 (default 0.5)
    #[argh(option)]
    denoise_chroma: Option<f64>,
```

- [ ] **Step 2: Apply them**

Where the CLI configures the stabilizer before rendering:

```rust
    if let Some(ref q) = opts.denoise {
        match gyroflow_core::denoise::DenoiseQuality::from_id(q) {
            Some(quality) => {
                let mut d = stab.denoise.write();
                d.enabled = true;
                d.quality = quality;
            }
            None => {
                eprintln!("Unknown denoise quality: {q}. Expected 'fast' or 'high'.");
                std::process::exit(1);
            }
        }
    }
    if let Some(v) = opts.denoise_luma   { stab.denoise.write().set_param("luma_strength", v); }
    if let Some(v) = opts.denoise_chroma { stab.denoise.write().set_param("chroma_strength", v); }
```

- [ ] **Step 3: Verify**

Run: `just run -- --denoise high --denoise-luma 0.8 <a test clip>`
Expected: renders, and the output differs from an undenoised render of the same clip.

Run: `just run -- --denoise nonsense <a test clip>`
Expected: exits with the error message, not a panic.

- [ ] **Step 4: Commit**

```bash
git add src/cli.rs
git commit -m "feat(denoise): CLI flags for temporal denoising"
```

---

### Task 13: GPU end-to-end tests

**Files:**
- Create: `src/core/denoise/tests_gpu.rs`
- Modify: `src/core/denoise/mod.rs` (add `#[cfg(test)] mod tests_gpu;`)

**Interfaces:**
- Consumes: everything.
- Produces: `#[ignore]`d tests run by hand.

These assert against real wgpu output and are `#[ignore]`d so CI skips them. **Accepted risk, recorded in the spec:** the shaders therefore get no automated check, and wgpu/OpenCL divergence will not be caught automatically.

- [ ] **Step 1: Write the tests**

Create `src/core/denoise/tests_gpu.rs`:

```rust
// Run these by hand: `just test -- --ignored denoise_gpu`
//
// They need a real GPU, so CI skips them. That means the kernels have no
// automated guard - a deliberate trade recorded in the design doc.

use super::*;

/// A synthetic clip: a fixed gradient plus per-frame Gaussian noise of known
/// sigma, with an identity gyro so every reference frame aligns exactly.
fn synthetic_clip(frames: usize, width: usize, height: usize, noise_sigma: f32) -> Vec<Vec<f32>> {
    let mut rng_state: u32 = 0x2545F491;
    let mut next = || {
        // xorshift, then Box-Muller against a second draw. Deterministic so a
        // failure reproduces.
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        (rng_state as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    (0..frames).map(|_| {
        (0..width * height).map(|i| {
            let base = (i % width) as f32 / width as f32;
            base + next() * noise_sigma
        }).collect()
    }).collect()
}

fn variance(v: &[f32]) -> f32 {
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32
}

#[test]
#[ignore = "needs a GPU"]
fn denoise_gpu_reduces_variance_by_root_n_on_a_static_scene() {
    // With identity motion and independent noise, averaging n frames should
    // divide noise variance by n. Allow generous slack: the rejection weights
    // discard some samples, and the interpolator correlates neighbours slightly.
    let k = 3;
    let n = 2 * k + 1;
    let clip = synthetic_clip(n, 256, 256, 0.05);

    let denoised = render_through_gpu(&clip, k, /* sigma */ 0.05);

    let before = variance(&clip[k]);
    let after = variance(&denoised);
    let ratio = before / after;
    assert!(ratio > (n as f32) * 0.5, "expected roughly {n}x variance reduction, got {ratio:.2}x");
}

#[test]
#[ignore = "needs a GPU"]
fn denoise_gpu_is_a_no_op_at_k_zero() {
    let clip = synthetic_clip(1, 256, 256, 0.05);
    let out = render_through_gpu(&clip, 0, 0.05);
    for (a, b) in clip[0].iter().zip(out.iter()) {
        assert_eq!(a, b, "k=0 must not touch a single pixel");
    }
}

#[test]
#[ignore = "needs a GPU"]
fn denoise_gpu_rejects_a_moving_subject_instead_of_ghosting() {
    // A bright square present in the centre frame and absent from every
    // reference. Rejection should keep it; a naive average would fade it to
    // 1/n of its contrast.
    let k = 3;
    let n = 2 * k + 1;
    let (w, h) = (256usize, 256usize);
    let mut clip = synthetic_clip(n, w, h, 0.01);
    for y in 100..150 { for x in 100..150 { clip[k][y * w + x] = 1.0; } }

    let out = render_through_gpu(&clip, k, /* sigma */ 0.02);

    let subject: f32 = (110..140).flat_map(|y| (110..140).map(move |x| (y, x)))
        .map(|(y, x)| out[y * w + x]).sum::<f32>() / (30.0 * 30.0);
    assert!(subject > 0.9, "moving subject ghosted away to {subject:.3}; rejection is not working");
}
```

- [ ] **Step 2: Write the GPU harness**

Implement `render_through_gpu(clip: &[Vec<f32>], k: usize, sigma: f32) -> Vec<f32>`: build a `Buffers` with `clip[k]` as `input` and the rest as `references`, set an identity `FrameTransform` with `reference_matrices` filled with identity blocks, run `undistort_image`, and read the output back. Model it on however the existing stabilization tests construct a `Buffers`; find them with `grep -rn "BufferSource::Cpu" src/core/ --include=*.rs`.

- [ ] **Step 3: Run them**

Run: `just test -- --ignored denoise_gpu`
Expected: PASS, 3 tests.

- [ ] **Step 4: Run the same three against OpenCL**

Select the OpenCL backend and re-run. This is the only check that the two kernels agree, and it is manual.
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/core/denoise/tests_gpu.rs src/core/denoise/mod.rs
git commit -m "test(denoise): GPU end-to-end tests for the temporal gather"
```

---

## Verification

Before considering the feature done:

- [ ] `just test` passes.
- [ ] `just test -- --ignored denoise_gpu` passes on wgpu.
- [ ] `just test -- --ignored denoise_gpu` passes on OpenCL.
- [ ] `just clippy` is clean.
- [ ] An export with denoise disabled is byte-identical to one from before this work (PSNR `inf`).
- [ ] An export with denoise enabled has the same frame count as its input.
- [ ] The paused preview matches a single-frame export at the same timestamp.
- [ ] A project saved with denoise settings reloads with them intact.
