# Chroma Smoothing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the chroma blotches left on moving subjects after temporal denoising, with a luma-guided joint-bilateral chroma pass on the GPU (OpenCL and wgpu), controlled by one "Colour smoothing" slider.

**Architecture:** A second GPU pass runs inside each backend's `undistort_image`, on the finished output of the existing kernel, before readback. A Rust reference implementation defines the exact maths; both GPU passes are tested against it. The export closure and the preview decide per plane whether the pass applies (planar chroma with a luma guide, or packed RGB) and hand the parameters over through `FrameTransform::chroma_filter` and the guide through `Buffers::guide`.

**Tech Stack:** Rust (gyroflow-core, gyroflow app), OpenCL C (`ocl` crate), WGSL (`wgpu`), QML.

**Spec:** `docs/superpowers/specs/2026-09-27-chroma-smoothing-design.md`

## Global Constraints

- Filter: joint bilateral, taps `d ∈ {-8,-6,…,8}²` (81), spatial sigma 4, luma sigma `0.015` of full scale; `out = p + amount·(F − p)`; taps clamped to the plane edge.
- `amount == 0` (or denoising disabled) must skip the pass entirely: bit-exact with no pass.
- The pass runs only when the output is `BufferSource::Cpu`; anything else skips it and warns once per backend instance.
- Planar mode: chroma planes only, guide = luma plane, averaged down to chroma resolution by the rounded size ratio. RGB mode: filter `c = rgb − Y·(1,1,1)` and keep each pixel's own `Y`; alpha untouched.
- Skipped: planar RGB float planes (`R32f`), packed YUV (`AYUV16`), alpha planes, half-float (`RGBA16F`) on OpenCL, the wgpu buffer (compute) path, the CPU backend, the Qt RHI pipeline.
- `KernelParams` and its five mirrors and the `.qsb` shaders are **not** changed.
- Setting: `DenoiseSettings::chroma_smoothing`, 0..1, default `0.5`, key `denoise.chroma_smoothing`, CLI `--denoise-chroma-smoothing`.
- Never push. Commit locally on `Dev-sony`. Line endings differ per file: detect per file (Python byte count), never `sed -i` or PowerShell `Set-Content` on repo files.

## Build and test commands (Windows, this machine)

Every cargo/just command needs the MSVC environment. In PowerShell:

```powershell
$vc = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vc`" >nul 2>nul && set" | % { if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($matches[1])" $matches[2] } }
```

- Core unit tests: `cd src/core; cargo test --lib <filter>`
- GPU tests (need a GPU, OpenCL link path):
  `$env:RUSTFLAGS = "-L native=C:/Users/Leo/Gits/gyroflow/ext/vcpkg/installed/x64-windows/lib"; cd src/core; cargo test --lib --features use-opencl -- --ignored <filter> --nocapture --test-threads 1`
- App: `just build`, `just test` (repo root). A running `gyroflow.exe` locks the build: ask the user before closing it.
- A new `.rs` file containing `cpp!` needs `build.rs` touched before building (stale cpp_build metadata). Not expected here.

## Review Focus

1. **Odd-sized planes** (odd luma width, e.g. 3841 → chroma 1921): the guide ratio must round (`2`, not `1`), and guide reads must clamp to the guide's edge instead of reading past it. Pinned in Task 1 (reference) and Task 4/5 (GPU, guide 127×127 for a 64×64 plane).
2. **Padded strides** on the plane and the guide (FFmpeg pads rows): every read must step by the stride, not `width × bytes`. Pinned in Task 4/5 (guide stride padded by 64 bytes).
3. **`amount == 0`** must produce exactly the temporal-only output, not a re-quantised copy. Pinned in Task 4/5.
4. **Non-CPU output** (zero-copy GPU textures): the pass must skip, not panic or read garbage. Not testable in the harness (it cannot build a GPU-interop output), so it is enforced by construction in Tasks 4/5: the pass objects are only created when the output is `BufferSource::Cpu`, and `run_chroma`/`encode_chroma` return `false` with a one-time warning when they are absent. Reviewers of Tasks 4/5: check exactly that.
5. **Denoise disabled with `chroma_smoothing > 0`** must not run the pass. Pinned in Task 2 (`chroma_filter_params` returns an inactive block).

---

### Task 1: Reference filter and parameter block

**Files:**
- Create: `src/core/denoise/chroma_filter.rs`
- Modify: `src/core/denoise/mod.rs` (add `pub mod chroma_filter;`)

**Interfaces:**
- Produces:
  - `pub const RADIUS: i32 = 8; pub const STEP: i32 = 2; pub const SIGMA_SPATIAL: f32 = 4.0; pub const SIGMA_LUMA: f32 = 0.015;`
  - `pub const MODE_OFF: i32 = 0; pub const MODE_PLANAR: i32 = 1; pub const MODE_RGB: i32 = 2;`
  - `#[repr(C)] pub struct ChromaFilterParams { width, height, stride, channels, guide_width, guide_height, guide_stride, mode: i32, amount, sigma_y, max_value, _pad: f32 }` (48 bytes, `Pod`, `Default`, `Copy`, `PartialEq`, `Debug`), with `fn is_active(&self) -> bool`.
  - `pub fn guide_ratio(guide: usize, plane: usize) -> usize`
  - `pub fn reference_planar(plane: &[f32], w: usize, h: usize, channels: usize, guide: &[f32], gw: usize, gh: usize, amount: f32) -> Vec<f32>` (all values normalised 0..1, interleaved, no stride)
  - `pub fn reference_rgb(pixels: &[f32], w: usize, h: usize, channels: usize, amount: f32) -> Vec<f32>` (`channels` 3 or 4; channel 3 is alpha and is copied)

- [ ] **Step 1: Write the failing tests** at the bottom of the new file.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn noise(i: usize, seed: u64) -> f32 {
        let mut z = (i as u64).wrapping_add(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn std(v: &[f32]) -> f32 {
        let m = v.iter().sum::<f32>() / v.len() as f32;
        (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32).sqrt()
    }

    #[test]
    fn params_block_is_48_bytes() {
        assert_eq!(std::mem::size_of::<ChromaFilterParams>(), 48);
    }

    #[test]
    fn guide_ratio_rounds_for_odd_sizes() {
        assert_eq!(guide_ratio(3840, 1920), 2);
        assert_eq!(guide_ratio(3841, 1921), 2); // odd luma width: chroma is ceil(w/2)
        assert_eq!(guide_ratio(2160, 2160), 1); // 4:2:2 vertical, 4:4:4
        assert_eq!(guide_ratio(127, 64), 2);
    }

    #[test]
    fn amount_zero_is_the_identity() {
        let (w, h) = (20, 12);
        let plane: Vec<f32> = (0..w * h * 2).map(|i| 0.5 + 0.1 * noise(i, 1)).collect();
        let guide = vec![0.4; w * 2 * h * 2];
        assert_eq!(reference_planar(&plane, w, h, 2, &guide, w * 2, h * 2, 0.0), plane);
        let rgb: Vec<f32> = (0..w * h * 4).map(|i| 0.5 + 0.1 * noise(i, 2)).collect();
        assert_eq!(reference_rgb(&rgb, w, h, 4, 0.0), rgb);
    }

    #[test]
    fn flat_chroma_noise_is_reduced() {
        let (w, h) = (40, 40);
        let plane: Vec<f32> = (0..w * h * 2).map(|i| 0.5 + 0.02 * noise(i, 3)).collect();
        let guide = vec![0.3; w * 2 * h * 2]; // flat luma: every tap has full range weight
        let out = reference_planar(&plane, w, h, 2, &guide, w * 2, h * 2, 1.0);
        assert!(std(&out) < std(&plane) / 4.0, "{} -> {}", std(&plane), std(&out));
    }

    #[test]
    fn colour_does_not_bleed_across_a_luma_edge() {
        // Left half: dark luma, chroma 0.3. Right half: bright luma, chroma 0.7.
        let (w, h) = (32, 16);
        let plane: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 0.3 } else { 0.7 }).collect();
        let guide: Vec<f32> = (0..w * 2 * h * 2).map(|i| if i % (w * 2) < w { 0.1 } else { 0.9 }).collect();
        let out = reference_planar(&plane, w, h, 1, &guide, w * 2, h * 2, 1.0);
        for y in 0..h {
            assert!((out[y * w + w / 2 - 1] - 0.3).abs() < 1e-3, "left of the edge bled: {}", out[y * w + w / 2 - 1]);
            assert!((out[y * w + w / 2] - 0.7).abs() < 1e-3, "right of the edge bled: {}", out[y * w + w / 2]);
        }
    }

    #[test]
    fn rgb_mode_keeps_every_pixels_luma_and_alpha() {
        let (w, h) = (24, 24);
        let px: Vec<f32> = (0..w * h * 4).map(|i| if i % 4 == 3 { 0.8 } else { 0.5 + 0.05 * noise(i, 4) }).collect();
        let out = reference_rgb(&px, w, h, 4, 1.0);
        let y = |p: &[f32]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
        for (a, b) in px.chunks_exact(4).zip(out.chunks_exact(4)) {
            assert!((y(a) - y(b)).abs() < 1e-5, "luma moved");
            assert_eq!(a[3], b[3], "alpha moved");
        }
        // And the chroma did get smoother.
        let cb = |v: &[f32]| v.chunks_exact(4).map(|p| p[2] - y(p)).collect::<Vec<_>>();
        assert!(std(&cb(&out)) < std(&cb(&px)) / 2.0);
    }

    #[test]
    fn odd_sized_plane_reads_stay_inside_the_guide() {
        // 64x64 chroma for a 127x127 luma: the last block's second column/row does not exist.
        let (w, h, gw, gh) = (64, 64, 127, 127);
        let plane = vec![0.5; w * h];
        let guide = vec![0.5; gw * gh];
        let out = reference_planar(&plane, w, h, 1, &guide, gw, gh, 1.0); // must not panic
        assert!(out.iter().all(|v| (v - 0.5).abs() < 1e-6));
    }
}
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd src/core; cargo test --lib chroma_filter`
Expected: compile error, `chroma_filter` module / items not found.

- [ ] **Step 3: Implement**

```rust
// Copyright © 2026 Gyroflow

//! Chroma smoothing: a luma-guided joint bilateral on chroma, run after the temporal gather.
//! This file is the definition of the maths. `opencl_chroma_filter.cl` and
//! `wgpu_chroma_filter.wgsl` mirror it and are tested against it (`tests_gpu.rs`).
//!
//! Why: temporal denoising cannot average subjects that move relative to the camera, and the
//! chroma noise it leaves there is blotchy (8-16 px, from 4:2:2 capture plus H.264). Measured
//! on Sony S-Log3 4K, this filter took the V noise on such a patch from 3.23 to 1.84 while
//! keeping colour edges on luma edges (see docs/superpowers/specs/2026-09-27-chroma-smoothing-design.md).

pub const RADIUS: i32 = 8;
pub const STEP: i32 = 2;
pub const SIGMA_SPATIAL: f32 = 4.0;
/// Luma difference, as a fraction of full scale, at which a tap's weight falls to exp(-1/2).
pub const SIGMA_LUMA: f32 = 0.015;

pub const MODE_OFF: i32 = 0;
/// A chroma plane (U, V, or interleaved UV) with the luma plane as the guide.
pub const MODE_PLANAR: i32 = 1;
/// Packed RGB: the guide is each pixel's own Rec.709 luma; only `rgb - Y` is filtered.
pub const MODE_RGB: i32 = 2;

/// The pass's uniform block. Mirrored by `ChromaFilterParams` in both shaders; 48 bytes.
/// Strides are in bytes. The backends fill the sizes from `Buffers`; settings fill the rest.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ChromaFilterParams {
    pub width: i32, pub height: i32, pub stride: i32, pub channels: i32,
    pub guide_width: i32, pub guide_height: i32, pub guide_stride: i32, pub mode: i32,
    pub amount: f32, pub sigma_y: f32, pub max_value: f32, pub _pad: f32,
}
unsafe impl bytemuck::Zeroable for ChromaFilterParams {}
unsafe impl bytemuck::Pod for ChromaFilterParams {}

impl ChromaFilterParams {
    pub fn is_active(&self) -> bool { self.mode != MODE_OFF && self.amount > 0.0 }
}

/// Luma samples per chroma sample along one axis. Rounded, because a plane of odd size has a
/// chroma plane of ceil(n/2): 3841 / 1921 must be 2, not 1.
pub fn guide_ratio(guide: usize, plane: usize) -> usize {
    ((guide + plane / 2) / plane.max(1)).max(1)
}

fn tap_weight(dx: i32, dy: i32, dg: f32, sigma_y: f32) -> f32 {
    (-((dx * dx + dy * dy) as f32) / (2.0 * SIGMA_SPATIAL * SIGMA_SPATIAL) - dg * dg / (2.0 * sigma_y * sigma_y)).exp()
}

/// The luma guide at chroma sample (x, y): the average of the luma block it covers, with
/// guide coordinates clamped so odd-sized planes never read past the guide.
fn guide_at(guide: &[f32], gw: usize, gh: usize, rx: usize, ry: usize, x: usize, y: usize) -> f32 {
    let mut s = 0.0;
    for j in 0..ry {
        for i in 0..rx {
            let gx = (x * rx + i).min(gw - 1);
            let gy = (y * ry + j).min(gh - 1);
            s += guide[gy * gw + gx];
        }
    }
    s / (rx * ry) as f32
}

pub fn reference_planar(plane: &[f32], w: usize, h: usize, channels: usize, guide: &[f32], gw: usize, gh: usize, amount: f32) -> Vec<f32> {
    if amount <= 0.0 { return plane.to_vec(); }
    let (rx, ry) = (guide_ratio(gw, w), guide_ratio(gh, h));
    let g: Vec<f32> = (0..w * h).map(|i| guide_at(guide, gw, gh, rx, ry, i % w, i / w)).collect();
    let mut out = plane.to_vec();
    for y in 0..h {
        for x in 0..w {
            let g0 = g[y * w + x];
            let mut acc = [0.0f32; 4];
            let mut wsum = 0.0;
            for dy in (-RADIUS..=RADIUS).step_by(STEP as usize) {
                for dx in (-RADIUS..=RADIUS).step_by(STEP as usize) {
                    let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let wt = tap_weight(dx, dy, g[sy * w + sx] - g0, SIGMA_LUMA);
                    for c in 0..channels { acc[c] += plane[(sy * w + sx) * channels + c] * wt; }
                    wsum += wt;
                }
            }
            for c in 0..channels {
                let p = plane[(y * w + x) * channels + c];
                out[(y * w + x) * channels + c] = p + amount * (acc[c] / wsum - p);
            }
        }
    }
    out
}

pub fn reference_rgb(pixels: &[f32], w: usize, h: usize, channels: usize, amount: f32) -> Vec<f32> {
    if amount <= 0.0 { return pixels.to_vec(); }
    let luma = |i: usize| 0.2126 * pixels[i * channels] + 0.7152 * pixels[i * channels + 1] + 0.0722 * pixels[i * channels + 2];
    let mut out = pixels.to_vec();
    for y in 0..h {
        for x in 0..w {
            let i0 = y * w + x;
            let y0 = luma(i0);
            let mut acc = [0.0f32; 3];
            let mut wsum = 0.0;
            for dy in (-RADIUS..=RADIUS).step_by(STEP as usize) {
                for dx in (-RADIUS..=RADIUS).step_by(STEP as usize) {
                    let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let i = sy * w + sx;
                    let yi = luma(i);
                    let wt = tap_weight(dx, dy, yi - y0, SIGMA_LUMA);
                    for c in 0..3 { acc[c] += (pixels[i * channels + c] - yi) * wt; }
                    wsum += wt;
                }
            }
            for c in 0..3 {
                let c0 = pixels[i0 * channels + c] - y0;
                out[i0 * channels + c] = y0 + c0 + amount * (acc[c] / wsum - c0);
            }
        }
    }
    out
}
```

Add to `src/core/denoise/mod.rs`, next to the existing `#[cfg(test)] mod tests_gpu;` line:

```rust
pub mod chroma_filter;
```

- [ ] **Step 4: Run the tests**

Run: `cd src/core; cargo test --lib chroma_filter`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/core/denoise/chroma_filter.rs src/core/denoise/mod.rs
git commit -m "feat(denoise): reference implementation of luma-guided chroma smoothing"
```

---

### Task 2: The setting, its persistence, CLI flag and slider

**Files:**
- Modify: `src/core/denoise/mod.rs` (`DenoiseSettings`)
- Modify: `src/core/lib.rs` (export ~line 1453 `"denoise": denoise`, import ~line 1957, tests ~line 2513)
- Modify: `src/cli.rs` (~lines 94-104, ~196-200)
- Modify: `src/ui/menu/Denoise.qml`

**Interfaces:**
- Consumes: `chroma_filter::{ChromaFilterParams, MODE_OFF, SIGMA_LUMA}` (Task 1)
- Produces: `DenoiseSettings::chroma_smoothing: f64`; `pub fn chroma_filter_params(&self, mode: i32, max_value: f32) -> chroma_filter::ChromaFilterParams` (sizes left 0 for the backend to fill; inactive when disabled or `chroma_smoothing == 0`).

- [ ] **Step 1: Write the failing tests** in `src/core/denoise/mod.rs` `mod tests`:

```rust
    #[test]
    fn chroma_smoothing_defaults_clamps_and_changes_the_checksum() {
        let mut d = DenoiseSettings::default();
        assert_eq!(d.chroma_smoothing, 0.5);
        assert!(d.set_param("chroma_smoothing", 7.0));
        assert_eq!(d.set_param_value("chroma_smoothing"), MAX_STRENGTH);
        let before = d.get_checksum();
        d.chroma_smoothing = 0.2;
        assert_ne!(before, d.get_checksum());
    }

    #[test]
    fn chroma_filter_params_are_inactive_unless_enabled_and_nonzero() {
        let mut d = DenoiseSettings::default();
        d.enabled = false;
        assert!(!d.chroma_filter_params(chroma_filter::MODE_RGB, 1.0).is_active(), "disabled denoise must not smooth");
        d.enabled = true;
        d.chroma_smoothing = 0.0;
        assert!(!d.chroma_filter_params(chroma_filter::MODE_RGB, 1.0).is_active());
        d.chroma_smoothing = 0.5;
        let p = d.chroma_filter_params(chroma_filter::MODE_PLANAR, 65535.0);
        assert!(p.is_active());
        assert_eq!((p.mode, p.amount, p.sigma_y, p.max_value), (chroma_filter::MODE_PLANAR, 0.5, chroma_filter::SIGMA_LUMA, 65535.0));
    }
```

and in `src/core/lib.rs` tests, extend `denoise_settings_round_trip_through_the_project_file` (set `d.chroma_smoothing = 0.3;` in the setup block and add `assert_eq!(a.chroma_smoothing, b.chroma_smoothing);`), and add:

```rust
    #[test]
    fn denoise_import_clamps_chroma_smoothing() {
        let mgr = StabilizationManager::default();
        apply_as_preset(&mgr, r#"{"title":"Gyroflow data file","version":4,"denoise":{"chroma_smoothing":-3.0}}"#);
        assert_eq!(mgr.denoise.read().chroma_smoothing, denoise::MIN_STRENGTH);
    }
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd src/core; cargo test --lib denoise`
Expected: compile errors (`chroma_smoothing`, `chroma_filter_params` missing).

- [ ] **Step 3: Implement**

In `DenoiseSettings` add the field (after `chroma_strength`) and its default:

```rust
    /// 0..1. Blend of the spatial chroma pass (`chroma_filter`) into the temporal result;
    /// 0 skips the pass entirely.
    pub chroma_smoothing: f64,
```
```rust
            chroma_smoothing: 0.5,
```

In `set_param` add `"chroma_smoothing" => self.chroma_smoothing = val.clamp(MIN_STRENGTH, MAX_STRENGTH),`; in `set_param_value` add `"chroma_smoothing" => self.chroma_smoothing,`; in `get_checksum` change the loop to `for v in [self.luma_strength, self.chroma_strength, self.chroma_smoothing]`. Add:

```rust
    /// The pass's parameters for a plane of the given mode. Sizes are left at 0: the backend
    /// fills them from the buffers it is actually given.
    pub fn chroma_filter_params(&self, mode: i32, max_value: f32) -> chroma_filter::ChromaFilterParams {
        if !self.enabled || self.chroma_smoothing <= 0.0 { return Default::default(); }
        chroma_filter::ChromaFilterParams {
            mode,
            amount: self.chroma_smoothing.clamp(MIN_STRENGTH, MAX_STRENGTH) as f32,
            sigma_y: chroma_filter::SIGMA_LUMA,
            max_value,
            ..Default::default()
        }
    }
```

Also extend the existing `set_param_clamps_every_scalar_to_its_range` and `every_pixel_affecting_field_changes_the_checksum` tests: add `"chroma_smoothing"` to the name list, and add `("chroma_smoothing", { let mut d = base.clone(); d.chroma_smoothing = 0.9; d }),`.

In `src/core/lib.rs` export, add `"chroma_smoothing": d.chroma_smoothing,` to the `denoise` json. In the import loop change the key list to `["luma_strength", "chroma_strength", "chroma_smoothing"]`.

In `src/cli.rs`, after the `denoise_chroma` option:

```rust
    /// denoise chroma smoothing (spatial colour-blotch reduction), 0.0 to 1.0
    #[argh(option)]
    denoise_chroma_smoothing: Option<f64>,
```
and after the `denoise_chroma` insert:
```rust
            if let Some(v) = opts.denoise_chroma_smoothing { denoise.insert("chroma_smoothing".into(), v.into()); }
```

In `src/ui/menu/Denoise.qml`, in `loadGyroflow` add
`if (typeof d.chroma_smoothing !== "undefined") smoothingSlider.value = +d.chroma_smoothing;`
and after the Chroma strength `Label`, add:

```qml
    Label {
        text: qsTr("Colour smoothing");
        tooltip: qsTr("Smooths colour blotches that temporal denoising leaves on moving subjects, following brightness edges so colours stay inside their outlines. 0% turns it off.");
        Column {
            width: parent.width;
            spacing: 3 * dpiScale;
            SliderWithField {
                id: smoothingSlider;
                width: parent.width;
                from: 0; to: 100;              // core: 0 .. 1
                value: 0.5; defaultValue: 0.5;
                scaler: 100.0;
                unit: "%"; precision: 0;
                onValueChanged: controller.set_denoise_param("chroma_smoothing", value);
            }
            SliderSteppers {
                target: smoothingSlider;
                step: 5;
            }
        }
    }
```

- [ ] **Step 4: Run the tests**

Run: `cd src/core; cargo test --lib denoise` then `cargo test --lib`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/core/denoise/mod.rs src/core/lib.rs src/cli.rs src/ui/menu/Denoise.qml
git commit -m "feat(denoise): chroma smoothing setting, project key, CLI flag and slider"
```

---

### Task 3: Plumbing: `Buffers::guide` and `FrameTransform::chroma_filter`

**Files:**
- Modify: `src/core/gpu/mod.rs:286` (`Buffers`)
- Modify: `src/core/stabilization/frame_transform.rs:12` (`FrameTransform`)
- Modify: every `Buffers { … }` literal (16 today; the compiler lists them) and the two `FrameTransform { … }` literals in tests (`denoise/tests_gpu.rs:156`, `stabilization/mod.rs:1083`).

**Interfaces:**
- Produces: `Buffers { …, pub guide: Option<BufferDescription<'a>> }` (luma plane for planar chroma, CPU only); `FrameTransform { …, pub chroma_filter: crate::denoise::chroma_filter::ChromaFilterParams }`.

- [ ] **Step 1: Add the fields**

```rust
pub struct Buffers<'a> {
    pub input: BufferDescription<'a>,
    pub output: BufferDescription<'a>,
    /// Temporal reference frames, in temporal order, excluding the centre frame
    /// which is `input`. Empty when denoising is off.
    pub references: Vec<BufferDescription<'a>>,
    /// The luma plane guiding the chroma pass on a chroma plane (see
    /// `denoise::chroma_filter`). `None` everywhere else, and whenever the pass is off.
    pub guide: Option<BufferDescription<'a>>,
}
```

```rust
    /// The spatial chroma pass to run after this frame's main kernel. Inactive by default.
    pub chroma_filter: crate::denoise::chroma_filter::ChromaFilterParams,
```

- [ ] **Step 2: Build and fix every literal**

Run: `cd src/core; cargo test --lib --no-run` then `just build` from the root.
Each error `missing field guide` / `missing field chroma_filter`: add `guide: None,` / `chroma_filter: Default::default(),`. Do not change anything else in those literals.

- [ ] **Step 3: Run all tests**

Run: `cd src/core; cargo test --lib` and `just test`
Expected: unchanged pass counts (196 core, 138 app at time of writing).

- [ ] **Step 4: Commit**

```bash
git add -A src
git commit -m "refactor(gpu): carry a chroma-pass guide buffer and parameters through Buffers and FrameTransform"
```

---

### Task 4: OpenCL chroma pass

**Files:**
- Create: `src/core/gpu/opencl_chroma_filter.cl`
- Modify: `src/core/gpu/opencl.rs` (`OclWrapper` struct, `new`, `undistort_image`)
- Test: `src/core/denoise/tests_gpu.rs`

**Interfaces:**
- Consumes: `ChromaFilterParams`, `reference_planar`, `reference_rgb` (Task 1); `Buffers::guide`, `FrameTransform::chroma_filter` (Task 3).
- Produces: OpenCL runs the pass when `itm.chroma_filter.is_active()`, the output is `BufferSource::Cpu`, and (planar) a guide is present. Test helpers `run_opencl_chroma_rgb`, `run_opencl_chroma_planar` used again in Task 5.

- [ ] **Step 1: Write the failing GPU tests** (append to `tests_gpu.rs`). They reuse the file's `identity_transform`, `desc`, `to_bytes`, `read_pixels`, `resolve`, `noise`, `W`, `H`, `STRIDE`, `N`.

```rust
// ---------------------------------------------------------------------------------------
// Chroma pass. No references: the gather is a no-op, so the output is exactly
// geometry(identity) + chroma pass, which is compared against the Rust reference.

use crate::denoise::chroma_filter::{ self, ChromaFilterParams };
use crate::stabilization::{ UV16 };

const CHROMA_TOLERANCE: f32 = 1.0 / 1024.0;

/// A flat grey RGBAf frame with a vertical luma edge and chroma noise on both sides.
fn rgb_chroma_scene() -> Vec<[f32; 4]> {
    (0..N).map(|i| {
        let y = if i % W as usize >= W as usize / 2 { 0.7 } else { 0.3 };
        let (cb, cr) = (0.02 * noise(1, i, 1.0), 0.02 * noise(2, i, 1.0));
        // Build rgb with luma y and small chroma offsets.
        let r = y + 1.5748 * cr;
        let b = y + 1.8556 * cb;
        let g = (y - 0.2126 * r - 0.0722 * b) / 0.7152;
        [r, g, b, 1.0]
    }).collect()
}

fn rgb_params(amount: f32) -> ChromaFilterParams {
    ChromaFilterParams { mode: chroma_filter::MODE_RGB, amount, sigma_y: chroma_filter::SIGMA_LUMA, max_value: 1.0, ..Default::default() }
}

#[cfg(feature = "use-opencl")]
fn run_opencl_chroma_rgb(px: &[[f32; 4]], amount: f32) -> Result<Vec<[f32; 4]>, String> {
    if crate::gpu::opencl::OclWrapper::list_devices().is_empty() { return Err("UNAVAILABLE: no OpenCL device".into()); }
    let mut itm = identity_transform(0, 0.02);
    itm.chroma_filter = rgb_params(amount);
    let mut input = to_bytes(px);
    let mut output = vec![0u8; (STRIDE * H) as usize];
    let mut buffers = Buffers { input: desc(&mut input), output: desc(&mut output), references: vec![], guide: None };
    let inst = crate::gpu::opencl::OclWrapper::new(&itm.kernel_params, RGBAf::ocl_names(), DistortionModel::default(), None, &buffers, 0, None)
        .map_err(|e| format!("OclWrapper::new failed: {e:?}"))?;
    inst.undistort_image(&mut buffers, &itm, &[]).map_err(|e| format!("undistort_image failed: {e:?}"))?;
    drop(buffers);
    Ok(read_pixels(&output))
}
#[cfg(not(feature = "use-opencl"))]
fn run_opencl_chroma_rgb(_px: &[[f32; 4]], _amount: f32) -> Result<Vec<[f32; 4]>, String> { Err("UNAVAILABLE: built without use-opencl".into()) }

/// A 64x64 UV16 chroma plane (4:2:0) and its 127x127 Luma16 guide with a row stride padded by
/// 64 bytes: odd luma size and padded stride are both review-focus cases.
struct PlanarScene { uv: Vec<u16>, guide: Vec<u8>, gw: usize, gh: usize, gstride: usize }
fn planar_chroma_scene() -> PlanarScene {
    let (gw, gh) = (127usize, 127usize);
    let gstride = gw * 2 + 64;
    let mut guide = vec![0u8; gstride * gh];
    for y in 0..gh { for x in 0..gw {
        let v: u16 = if x < gw / 2 { 20000 } else { 45000 };
        guide[y * gstride + x * 2..y * gstride + x * 2 + 2].copy_from_slice(&v.to_le_bytes());
    } }
    let uv = (0..N * 2).map(|i| (32768.0 + 1500.0 * noise(3, i, 1.0)) as u16).collect();
    PlanarScene { uv, guide, gw, gh, gstride }
}
fn planar_reference(s: &PlanarScene, amount: f32) -> Vec<f32> {
    let plane: Vec<f32> = s.uv.iter().map(|&v| v as f32 / 65535.0).collect();
    let guide: Vec<f32> = (0..s.gw * s.gh).map(|i| {
        let (x, y) = (i % s.gw, i / s.gw);
        u16::from_le_bytes([s.guide[y * s.gstride + x * 2], s.guide[y * s.gstride + x * 2 + 1]]) as f32 / 65535.0
    }).collect();
    chroma_filter::reference_planar(&plane, W as usize, H as usize, 2, &guide, s.gw, s.gh, amount)
}
/// Identity transform for a UV16 plane.
fn uv16_transform(amount: f32) -> FrameTransform {
    let mut itm = identity_transform(0, 0.02);
    let p = &mut itm.kernel_params;
    p.bytes_per_pixel = 4; p.pix_element_count = 2; p.stride = W * 4; p.output_stride = W * 4;
    p.max_pixel_value = 65535.0; p.pixel_value_limit = 65535.0;
    itm.chroma_filter = ChromaFilterParams { mode: chroma_filter::MODE_PLANAR, amount, sigma_y: chroma_filter::SIGMA_LUMA, max_value: 65535.0, ..Default::default() };
    itm
}
fn uv_desc(buf: &mut [u8]) -> BufferDescription<'_> {
    BufferDescription { size: (W as usize, H as usize, (W * 4) as usize), rect: None, rotation: None, data: BufferSource::Cpu { buffer: buf }, texture_copy: false }
}
fn uv_bytes(uv: &[u16]) -> Vec<u8> { uv.iter().flat_map(|v| v.to_le_bytes()).collect() }
fn uv_read(b: &[u8]) -> Vec<f32> { b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as f32 / 65535.0).collect() }

#[cfg(feature = "use-opencl")]
fn run_opencl_chroma_planar(s: &PlanarScene, amount: f32) -> Result<Vec<f32>, String> {
    if crate::gpu::opencl::OclWrapper::list_devices().is_empty() { return Err("UNAVAILABLE: no OpenCL device".into()); }
    let itm = uv16_transform(amount);
    let mut input = uv_bytes(&s.uv);
    let mut output = vec![0u8; (W * 4 * H) as usize];
    let mut guide = s.guide.clone();
    let mut buffers = Buffers {
        input: uv_desc(&mut input), output: uv_desc(&mut output), references: vec![],
        guide: Some(BufferDescription { size: (s.gw, s.gh, s.gstride), rect: None, rotation: None, data: BufferSource::Cpu { buffer: &mut guide }, texture_copy: false }),
    };
    let inst = crate::gpu::opencl::OclWrapper::new(&itm.kernel_params, UV16::ocl_names(), DistortionModel::default(), None, &buffers, 0, None)
        .map_err(|e| format!("OclWrapper::new failed: {e:?}"))?;
    inst.undistort_image(&mut buffers, &itm, &[]).map_err(|e| format!("undistort_image failed: {e:?}"))?;
    drop(buffers);
    Ok(uv_read(&output))
}
#[cfg(not(feature = "use-opencl"))]
fn run_opencl_chroma_planar(_s: &PlanarScene, _amount: f32) -> Result<Vec<f32>, String> { Err("UNAVAILABLE: built without use-opencl".into()) }

fn max_diff(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }
fn flat(px: &[[f32; 4]]) -> Vec<f32> { px.iter().flat_map(|p| p.iter().copied()).collect() }

#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored chroma_gpu"]
fn chroma_gpu_opencl_matches_the_reference() {
    let scene = rgb_chroma_scene();
    let reference = chroma_filter::reference_rgb(&flat(&scene), W as usize, H as usize, 4, 0.5);
    let ps = planar_chroma_scene();
    let preference = planar_reference(&ps, 0.5);
    let mut ran = false;
    if let Some(out) = resolve("OpenCL rgb", run_opencl_chroma_rgb(&scene, 0.5).map(|v| v)) {
        ran = true;
        let d = max_diff(&flat(&out), &reference);
        println!("OpenCL rgb: max diff vs reference {d:e}");
        assert!(d < CHROMA_TOLERANCE, "OpenCL rgb differs from the reference by {d}");
    }
    match run_opencl_chroma_planar(&ps, 0.5) {
        Ok(out) => { let d = max_diff(&out, &preference); println!("OpenCL planar: max diff {d:e}"); assert!(d < CHROMA_TOLERANCE, "OpenCL planar differs by {d}"); }
        Err(e) if e.starts_with("UNAVAILABLE") => println!("SKIPPED OpenCL planar: {e}"),
        Err(e) => panic!("OpenCL planar failed: {e}"),
    }
    assert!(ran, "no backend ran");
}

#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored chroma_gpu"]
fn chroma_gpu_amount_zero_is_bit_exact() {
    let scene = rgb_chroma_scene();
    if let Some(out) = resolve("OpenCL", run_opencl_chroma_rgb(&scene, 0.0)) {
        for i in 0..N { assert_eq!(out[i], scene[i], "pixel {i} changed with amount 0"); }
    }
}
```

Note: `resolve` takes `Result<Vec<[f32;4]>, String>`; the planar helper returns `Vec<f32>`, which is why it is matched by hand.

- [ ] **Step 2: Run to confirm failure**

Run: GPU test command with filter `chroma_gpu`.
Expected: `chroma_gpu_opencl_matches_the_reference` fails. The pass does not exist yet, so the output equals the input and the diff against the reference is well above 1/1024.

- [ ] **Step 3: Write the kernel** `src/core/gpu/opencl_chroma_filter.cl`. The placeholders are replaced in Rust exactly like the main kernel's.

```c
// Copyright © 2026 Gyroflow
// Chroma smoothing pass. Mirror of `denoise::chroma_filter` (Rust); tested against it.
// DATA_TYPE / DATA_TYPEF / DATA_CONVERT / DATA_CONVERTF / PIXEL_BYTES / GUIDE_TYPE / ROUND_OFFSET
// are substituted by opencl.rs.

typedef struct {
    int width, height, stride, channels;
    int guide_width, guide_height, guide_stride, mode;
    float amount, sigma_y, max_value, pad;
} ChromaFilterParams;

#define CF_RADIUS 8
#define CF_STEP 2
#define CF_SIGMA_SPATIAL 4.0f

inline float4 cf_load(__global const uchar *buf, int x, int y, __global const ChromaFilterParams *p) {
    x = clamp(x, 0, p->width - 1);
    y = clamp(y, 0, p->height - 1);
    float4 v = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    *(DATA_TYPEF *)&v = DATA_CONVERTF(*(__global const DATA_TYPE *)(buf + y * p->stride + x * PIXEL_BYTES));
    return v / p->max_value;
}

inline int cf_ratio(int g, int n) { return max(1, (g + n / 2) / n); }

inline float cf_guide(__global const uchar *g, int x, int y, __global const ChromaFilterParams *p) {
    x = clamp(x, 0, p->width - 1);
    y = clamp(y, 0, p->height - 1);
    int rx = cf_ratio(p->guide_width, p->width), ry = cf_ratio(p->guide_height, p->height);
    float s = 0.0f;
    for (int j = 0; j < ry; j++) {
        int gy = min(y * ry + j, p->guide_height - 1);
        for (int i = 0; i < rx; i++) {
            int gx = min(x * rx + i, p->guide_width - 1);
            s += (float)(((__global const GUIDE_TYPE *)(g + gy * p->guide_stride))[gx]);
        }
    }
    return s / (float)(rx * ry) / p->max_value;
}

inline float cf_luma(float4 v) { return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z; }

__kernel void chroma_filter(__global const uchar *src, __global uchar *dst, __global const uchar *guide, __global const void *params_buf) {
    __global const ChromaFilterParams *p = (__global const ChromaFilterParams *)params_buf;
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= p->width || y >= p->height) return;

    bool rgb = p->mode == 2;
    float4 c0 = cf_load(src, x, y, p);
    float g0 = rgb? cf_luma(c0) : cf_guide(guide, x, y, p);
    float4 v0 = rgb? c0 - (float4)(g0, g0, g0, 0.0f) : c0;

    float inv_s = 1.0f / (2.0f * CF_SIGMA_SPATIAL * CF_SIGMA_SPATIAL);
    float inv_y = 1.0f / (2.0f * p->sigma_y * p->sigma_y);
    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float wsum = 0.0f;
    for (int dy = -CF_RADIUS; dy <= CF_RADIUS; dy += CF_STEP) {
        for (int dx = -CF_RADIUS; dx <= CF_RADIUS; dx += CF_STEP) {
            float4 c = cf_load(src, x + dx, y + dy, p);
            float g = rgb? cf_luma(c) : cf_guide(guide, x + dx, y + dy, p);
            float4 v = rgb? c - (float4)(g, g, g, 0.0f) : c;
            float w = exp(-(float)(dx * dx + dy * dy) * inv_s - (g - g0) * (g - g0) * inv_y);
            acc += v * w;
            wsum += w;
        }
    }
    float4 out = v0 + p->amount * (acc / wsum - v0);
    if (rgb) { out.xyz += g0; out.w = c0.w; }
    out = out * p->max_value + ROUND_OFFSET;
    DATA_TYPEF o = *(DATA_TYPEF *)&out;
    *(__global DATA_TYPE *)(dst + y * p->stride + x * PIXEL_BYTES) = DATA_CONVERT(o);
}
```

- [ ] **Step 4: Wire it into `OclWrapper`.**

Add fields to the struct:

```rust
    /// The chroma smoothing pass (`opencl_chroma_filter.cl`), when this format supports it.
    chroma: Option<OclChroma>,
    chroma_warned: std::sync::atomic::AtomicBool,
```
```rust
struct OclChroma {
    kernel: Kernel,
    dst: Buffer<u8>,
    guide: Option<Buffer<u8>>,
    params: Buffer<u8>,
}
```

In `new()`: make the CPU output buffer readable by the second pass. Change the `is_in == false` flags in the `BufferSource::Cpu` arm to:

```rust
                                    else     { MemFlags::new().read_write().host_read_only().alloc_host_ptr() };
```

After `let kernel = builder.build()?;`, build the chroma pass for CPU outputs (half-float is skipped):

```rust
            let chroma = if matches!(buffers.output.data, BufferSource::Cpu { .. }) && ocl_names.1 != "convert_half4" {
                let guide_type = ocl_names.0.trim_end_matches(|c: char| c.is_ascii_digit());
                let src = include_str!("opencl_chroma_filter.cl")
                    .replace("DATA_CONVERTF", ocl_names.3)
                    .replace("DATA_TYPEF", ocl_names.2)
                    .replace("DATA_CONVERT", ocl_names.1)
                    .replace("DATA_TYPE", ocl_names.0)
                    .replace("GUIDE_TYPE", guide_type)
                    .replace("PIXEL_BYTES", &format!("{}", params.bytes_per_pixel))
                    .replace("ROUND_OFFSET", if ocl_names.0.starts_with('u') { "0.5f" } else { "0.0f" });
                let program = Program::builder().src(&src).devices(ctx.device).build(&ctx.context)?;
                let out_len = dest_buffer.len();
                let cdst = Buffer::<u8>::builder().queue(ocl_queue.clone()).len(out_len).flags(MemFlags::new().write_only().host_read_only().alloc_host_ptr()).build()?;
                let cguide = match &buffers.guide {
                    Some(g) => Some(Buffer::<u8>::builder().queue(ocl_queue.clone()).len(g.size.1 * g.size.2).flags(MemFlags::new().read_only().host_write_only()).build()?),
                    None => None,
                };
                let cparams = Buffer::<u8>::builder().queue(ocl_queue.clone()).len(std::mem::size_of::<crate::denoise::chroma_filter::ChromaFilterParams>()).flags(MemFlags::new().read_only().host_write_only()).build()?;
                let mut b = Kernel::builder();
                unsafe {
                    b.program(&program).name("chroma_filter").queue(ocl_queue.clone())
                        .global_work_size((buffers.output.size.0, buffers.output.size.1))
                        .disable_arg_type_check()
                        .arg(&dest_buffer)                      // 0: the main pass's output
                        .arg(&cdst)                             // 1
                        .arg(cguide.as_ref().unwrap_or(&dest_buffer)) // 2: unused in RGB mode
                        .arg(&cparams);                         // 3
                }
                Some(OclChroma { kernel: b.build()?, dst: cdst, guide: cguide, params: cparams })
            } else { None };
```

and add `chroma, chroma_warned: Default::default(),` to the `Ok(Self { … })`.

In `undistort_image`, right after `unsafe { self.kernel.enq()?; }`:

```rust
        let chroma_ran = self.run_chroma(buffers, &itm.chroma_filter)?;
```

and in the readback `BufferSource::Cpu` arm replace `self.dst.read(&mut **buffer).enq()?;` with:

```rust
                match (&self.chroma, chroma_ran) {
                    (Some(c), true) => c.dst.read(&mut **buffer).enq()?,
                    _ => self.dst.read(&mut **buffer).enq()?,
                }
```

Add the method:

```rust
    /// Runs the chroma pass if it applies. Returns whether it ran; never fails the frame for
    /// a reason that is ours to skip (see the spec: skipping is the only failure mode).
    fn run_chroma(&self, buffers: &Buffers, p: &crate::denoise::chroma_filter::ChromaFilterParams) -> ocl::Result<bool> {
        use crate::denoise::chroma_filter::MODE_PLANAR;
        if !p.is_active() { return Ok(false); }
        let skip = |why: &str| {
            if !self.chroma_warned.swap(true, std::sync::atomic::Ordering::Relaxed) { log::warn!("Chroma smoothing skipped: {why}"); }
            Ok(false)
        };
        let Some(c) = &self.chroma else { return skip("this output or pixel format is not supported"); };
        let mut kp = *p;
        kp.width = buffers.output.size.0 as i32;
        kp.height = buffers.output.size.1 as i32;
        kp.stride = buffers.output.size.2 as i32;
        if p.mode == MODE_PLANAR {
            let (Some(g), Some(gbuf)) = (&buffers.guide, &c.guide) else { return skip("no luma guide for a chroma plane"); };
            let BufferSource::Cpu { buffer } = &g.data else { return skip("the luma guide is not a CPU buffer"); };
            if buffer.len() < g.size.1 * g.size.2 || gbuf.len() < g.size.1 * g.size.2 { return skip("the luma guide is smaller than its size says"); }
            gbuf.write(&buffer[..g.size.1 * g.size.2]).enq()?;
            kp.guide_width = g.size.0 as i32;
            kp.guide_height = g.size.1 as i32;
            kp.guide_stride = g.size.2 as i32;
        }
        c.params.write(bytemuck::bytes_of(&kp)).enq()?;
        unsafe { c.kernel.enq()?; }
        Ok(true)
    }
```

- [ ] **Step 5: Run the tests**

Run: GPU test command with filter `chroma_gpu`, then with filter `denoise_gpu` (the existing temporal tests must still pass).
Expected: `chroma_gpu_*` pass with a printed max diff below `9.8e-4`; the 6 `denoise_gpu_*` tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/core/gpu/opencl_chroma_filter.cl src/core/gpu/opencl.rs src/core/denoise/tests_gpu.rs
git commit -m "feat(opencl): chroma smoothing pass after the main kernel"
```

---

### Task 5: wgpu chroma pass

**Files:**
- Create: `src/core/gpu/wgpu_chroma_filter.wgsl`
- Modify: `src/core/gpu/wgpu.rs` (`WgpuWrapper` struct, `new`, `undistort_image`)
- Modify: `src/core/gpu/wgpu_interop.rs:44` (`init_texture`: output textures also `TEXTURE_BINDING`)
- Test: `src/core/denoise/tests_gpu.rs`

**Interfaces:**
- Consumes: Task 1, Task 3, and Task 4's test helpers (`rgb_chroma_scene`, `planar_chroma_scene`, `planar_reference`, `uv16_transform`, `uv_desc`, `uv_bytes`, `uv_read`, `rgb_params`, `max_diff`, `flat`, `run_opencl_chroma_rgb`, `run_opencl_chroma_planar`, `CHROMA_TOLERANCE`).
- Produces: wgpu runs the pass on the render (texture) path for CPU outputs.

- [ ] **Step 1: Write the failing tests** (append to `tests_gpu.rs`):

```rust
fn run_wgpu_chroma_rgb(px: &[[f32; 4]], amount: f32) -> Result<Vec<[f32; 4]>, String> {
    if crate::gpu::wgpu::WgpuWrapper::initialize_context().is_none() { return Err("UNAVAILABLE: no usable wgpu adapter".into()); }
    let mut itm = identity_transform(0, 0.02);
    itm.kernel_params.translation2d = [-0.5, -0.5]; // see run_wgpu
    itm.chroma_filter = rgb_params(amount);
    let mut input = to_bytes(px);
    let mut output = vec![0u8; (STRIDE * H) as usize];
    let mut buffers = Buffers { input: desc(&mut input), output: desc(&mut output), references: vec![], guide: None };
    let inst = crate::gpu::wgpu::WgpuWrapper::new(&itm.kernel_params, RGBAf::wgpu_format().ok_or("no format")?, DistortionModel::default(), None, &buffers, 0, None)
        .map_err(|e| format!("WgpuWrapper::new failed: {e:?}"))?;
    if !inst.undistort_image(&mut buffers, &itm, &[]) { return Err("undistort_image returned false".into()); }
    drop(buffers);
    Ok(read_pixels(&output))
}

fn run_wgpu_chroma_planar(s: &PlanarScene, amount: f32) -> Result<Vec<f32>, String> {
    if crate::gpu::wgpu::WgpuWrapper::initialize_context().is_none() { return Err("UNAVAILABLE: no usable wgpu adapter".into()); }
    let mut itm = uv16_transform(amount);
    itm.kernel_params.translation2d = [-0.5, -0.5];
    let mut input = uv_bytes(&s.uv);
    let mut output = vec![0u8; (W * 4 * H) as usize];
    let mut guide = s.guide.clone();
    let mut buffers = Buffers {
        input: uv_desc(&mut input), output: uv_desc(&mut output), references: vec![],
        guide: Some(BufferDescription { size: (s.gw, s.gh, s.gstride), rect: None, rotation: None, data: BufferSource::Cpu { buffer: &mut guide }, texture_copy: false }),
    };
    let inst = crate::gpu::wgpu::WgpuWrapper::new(&itm.kernel_params, UV16::wgpu_format().ok_or("no format")?, DistortionModel::default(), None, &buffers, 0, None)
        .map_err(|e| format!("WgpuWrapper::new failed: {e:?}"))?;
    if !inst.undistort_image(&mut buffers, &itm, &[]) { return Err("undistort_image returned false".into()); }
    drop(buffers);
    Ok(uv_read(&output))
}

#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored chroma_gpu"]
fn chroma_gpu_wgpu_matches_the_reference_and_opencl() {
    let scene = rgb_chroma_scene();
    let reference = chroma_filter::reference_rgb(&flat(&scene), W as usize, H as usize, 4, 0.5);
    let ps = planar_chroma_scene();
    let preference = planar_reference(&ps, 0.5);

    let w_rgb = resolve("wgpu rgb", run_wgpu_chroma_rgb(&scene, 0.5)).expect("wgpu is required for this test");
    let d = max_diff(&flat(&w_rgb), &reference);
    println!("wgpu rgb: max diff vs reference {d:e}");
    assert!(d < CHROMA_TOLERANCE, "wgpu rgb differs from the reference by {d}");

    let w_planar = run_wgpu_chroma_planar(&ps, 0.5).expect("wgpu planar failed");
    let d = max_diff(&w_planar, &preference);
    println!("wgpu planar: max diff vs reference {d:e}");
    assert!(d < CHROMA_TOLERANCE, "wgpu planar differs from the reference by {d}");

    // The two backends against each other, per pixel.
    if let Ok(o) = run_opencl_chroma_rgb(&scene, 0.5) {
        let d = max_diff(&flat(&o), &flat(&w_rgb));
        println!("wgpu vs OpenCL rgb: {d:e}");
        assert!(d < CHROMA_TOLERANCE);
    }
    if let Ok(o) = run_opencl_chroma_planar(&ps, 0.5) {
        let d = max_diff(&o, &w_planar);
        println!("wgpu vs OpenCL planar: {d:e}");
        assert!(d < CHROMA_TOLERANCE);
    }

    // amount 0: bit-exact with no pass.
    let zero = run_wgpu_chroma_rgb(&scene, 0.0).unwrap();
    for i in 0..N { assert_eq!(zero[i], scene[i], "wgpu pixel {i} changed with amount 0"); }
}
```

- [ ] **Step 2: Run to confirm failure**

Run: GPU test command with filter `chroma_gpu_wgpu`.
Expected: FAIL. The wgpu output is unfiltered, so the diff against the reference is above tolerance.

- [ ] **Step 3: Write the shader** `src/core/gpu/wgpu_chroma_filter.wgsl` (`SCALAR` is `f32` or `u32`; `CF_ROUND` is `round` for `u32`, empty for `f32`; both substituted by `wgpu.rs`):

```wgsl
// Copyright © 2026 Gyroflow
// Chroma smoothing pass. Mirror of `denoise::chroma_filter` (Rust) and opencl_chroma_filter.cl.

struct ChromaFilterParams {
    width: i32, height: i32, stride: i32, channels: i32,
    guide_width: i32, guide_height: i32, guide_stride: i32, mode: i32,
    amount: f32, sigma_y: f32, max_value: f32, pad: f32,
}

@group(0) @binding(0) var<uniform> p: ChromaFilterParams;
@group(0) @binding(1) var src: texture_2d<SCALAR>;
@group(0) @binding(2) var guide: texture_2d<SCALAR>;

@vertex
fn cf_vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 4>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0));
    return vec4<f32>(pos[i], 0.0, 1.0);
}

fn cf_load(x: i32, y: i32) -> vec4<f32> {
    let c = vec2<i32>(clamp(x, 0, p.width - 1), clamp(y, 0, p.height - 1));
    return vec4<f32>(textureLoad(src, c, 0)) / p.max_value;
}
fn cf_ratio(g: i32, n: i32) -> i32 { return max(1, (g + n / 2) / n); }
fn cf_guide(x0: i32, y0: i32) -> f32 {
    let x = clamp(x0, 0, p.width - 1);
    let y = clamp(y0, 0, p.height - 1);
    let rx = cf_ratio(p.guide_width, p.width);
    let ry = cf_ratio(p.guide_height, p.height);
    var s = 0.0;
    for (var j = 0; j < ry; j = j + 1) {
        for (var i = 0; i < rx; i = i + 1) {
            let g = vec2<i32>(min(x * rx + i, p.guide_width - 1), min(y * ry + j, p.guide_height - 1));
            s = s + f32(textureLoad(guide, g, 0).x);
        }
    }
    return s / f32(rx * ry) / p.max_value;
}
fn cf_luma(v: vec4<f32>) -> f32 { return 0.2126 * v.x + 0.7152 * v.y + 0.0722 * v.z; }

@fragment
fn cf_fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<SCALAR> {
    let x = i32(position.x);
    let y = i32(position.y);
    let rgb = p.mode == 2;
    let c0 = cf_load(x, y);
    var g0 = 0.0;
    if (rgb) { g0 = cf_luma(c0); } else { g0 = cf_guide(x, y); }
    var v0 = c0;
    if (rgb) { v0 = c0 - vec4<f32>(g0, g0, g0, 0.0); }

    let inv_s = 1.0 / (2.0 * 4.0 * 4.0);
    let inv_y = 1.0 / (2.0 * p.sigma_y * p.sigma_y);
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var dy = -8; dy <= 8; dy = dy + 2) {
        for (var dx = -8; dx <= 8; dx = dx + 2) {
            let c = cf_load(x + dx, y + dy);
            var g = 0.0;
            if (rgb) { g = cf_luma(c); } else { g = cf_guide(x + dx, y + dy); }
            var v = c;
            if (rgb) { v = c - vec4<f32>(g, g, g, 0.0); }
            let w = exp(-f32(dx * dx + dy * dy) * inv_s - (g - g0) * (g - g0) * inv_y);
            acc = acc + v * w;
            wsum = wsum + w;
        }
    }
    var out = v0 + p.amount * (acc / wsum - v0);
    if (rgb) { out = vec4<f32>(out.x + g0, out.y + g0, out.z + g0, c0.w); }
    out = out * p.max_value;
    return vec4<SCALAR>(CF_ROUND(out));
}
```

- [ ] **Step 4: Wire it into `WgpuWrapper`.**

In `wgpu_interop.rs::init_texture`, give output textures `TEXTURE_BINDING` so the pass can read them:

```rust
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING
```

Add to the struct:

```rust
    chroma: Option<WgpuChroma>,
    chroma_warned: std::sync::atomic::AtomicBool,
```
```rust
struct WgpuChroma {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    out: wgpu::Texture,
    guide: Option<wgpu::Texture>,
}
```

In `new()`, before `Ok(Self { … })` (and add `chroma, chroma_warned: Default::default(),` to it):

```rust
            // Chroma smoothing: render path + CPU output only (see the spec).
            let chroma = if uses_textures && matches!(buffers.output.data, BufferSource::Cpu { .. }) {
                let src_tex = out_texture.wgpu_texture.as_ref().unwrap();
                let size = wgpu::Extent3d { width: buffers.output.size.0 as u32, height: buffers.output.size.1 as u32, depth_or_array_layers: 1 };
                let out = device.create_texture(&wgpu::TextureDescriptor { label: None, size, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu_format.0, usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, view_formats: &[] });
                // The guide is the same scalar type as the plane, one channel.
                let guide_format = match wgpu_format.0 {
                    wgpu::TextureFormat::R8Unorm | wgpu::TextureFormat::Rg8Unorm => Some(wgpu::TextureFormat::R8Unorm),
                    wgpu::TextureFormat::R16Uint | wgpu::TextureFormat::Rg16Uint => Some(wgpu::TextureFormat::R16Uint),
                    _ => None,
                };
                let guide = match (&buffers.guide, guide_format) {
                    (Some(g), Some(f)) => Some(device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: g.size.0 as u32, height: g.size.1 as u32, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: f, usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, view_formats: &[] })),
                    _ => None,
                };
                let params = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: std::mem::size_of::<crate::denoise::chroma_filter::ChromaFilterParams>() as u64, usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST, mapped_at_creation: false });
                let src = include_str!("wgpu_chroma_filter.wgsl")
                    .replace("SCALAR", wgpu_format.1)
                    .replace("CF_ROUND", if wgpu_format.1 == "u32" { "round" } else { "" });
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Owned(src)) });
                let sample_type = if wgpu_format.1 == "u32" { wgpu::TextureSampleType::Uint } else { wgpu::TextureSampleType::Float { filterable: false } };
                let tex_entry = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None };
                let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[
                    wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    tex_entry(1), tex_entry(2),
                ] });
                let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
                let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None, layout: Some(&pl),
                    vertex: wgpu::VertexState { module: &module, entry_point: Some("cf_vertex"), buffers: &[], compilation_options: Default::default() },
                    fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("cf_fragment"), targets: &[Some(wgpu::ColorTargetState { format: wgpu_format.0, blend: None, write_mask: wgpu::ColorWrites::default() })], compilation_options: Default::default() }),
                    primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
                    multiview_mask: None, depth_stencil: None, multisample: wgpu::MultisampleState::default(), cache: Default::default(),
                });
                let src_view = src_tex.create_view(&Default::default());
                // RGB mode never reads the guide; bind the source again so the layout is satisfied.
                let guide_view = guide.as_ref().map(|g| g.create_view(&Default::default())).unwrap_or_else(|| src_tex.create_view(&Default::default()));
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&guide_view) },
                ] });
                Some(WgpuChroma { pipeline, bind_group, params, out, guide })
            } else { None };
```

If the guide texture's sample type differs from the plane's (it does not for the four mapped formats above: `R8Unorm`/`Rg8Unorm` are both float, `R16Uint`/`Rg16Uint` both uint), the shared `sample_type` holds.

In `undistort_image`, replace the single `handle_output_texture(...)` line with:

```rust
        let chroma_ran = self.encode_chroma(buffers, &itm.chroma_filter, &mut encoder);
        let _temp_texture2 = if chroma_ran {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &self.chroma.as_ref().unwrap().out, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: self.staging_buffer.as_ref().unwrap(), layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.padded_out_stride), rows_per_image: None } },
                wgpu::Extent3d { width: buffers.output.size.0 as u32, height: buffers.output.size.1 as u32, depth_or_array_layers: 1 },
            );
            None
        } else {
            handle_output_texture(&self.device, &buffers.output, &self.queue, &mut encoder, &self.out_texture, self.pixel_format, self.staging_buffer.as_ref().unwrap(), self.padded_out_stride)
        };
```

Add the method:

```rust
    /// Records the chroma pass into `encoder` if it applies; returns whether it did.
    fn encode_chroma(&self, buffers: &Buffers, p: &crate::denoise::chroma_filter::ChromaFilterParams, encoder: &mut wgpu::CommandEncoder) -> bool {
        use crate::denoise::chroma_filter::MODE_PLANAR;
        if !p.is_active() { return false; }
        let skip = |why: &str| {
            if !self.chroma_warned.swap(true, std::sync::atomic::Ordering::Relaxed) { log::warn!("Chroma smoothing skipped: {why}"); }
            false
        };
        let Some(c) = &self.chroma else { return skip("this output, pixel format or pipeline is not supported"); };
        let mut kp = *p;
        kp.width = buffers.output.size.0 as i32;
        kp.height = buffers.output.size.1 as i32;
        kp.stride = buffers.output.size.2 as i32;
        if p.mode == MODE_PLANAR {
            let (Some(g), Some(gtex)) = (&buffers.guide, &c.guide) else { return skip("no luma guide for a chroma plane"); };
            let BufferSource::Cpu { buffer } = &g.data else { return skip("the luma guide is not a CPU buffer"); };
            if buffer.len() < g.size.1 * g.size.2 { return skip("the luma guide is smaller than its size says"); }
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: gtex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &buffer[..g.size.1 * g.size.2],
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(g.size.2 as u32), rows_per_image: None },
                wgpu::Extent3d { width: g.size.0 as u32, height: g.size.1 as u32, depth_or_array_layers: 1 },
            );
            kp.guide_width = g.size.0 as i32;
            kp.guide_height = g.size.1 as i32;
            kp.guide_stride = g.size.2 as i32;
        }
        self.queue.write_buffer(&c.params, 0, bytemuck::bytes_of(&kp));
        let view = c.out.create_view(&Default::default());
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None, depth_stencil_attachment: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, resolve_target: None, depth_slice: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
        });
        rpass.set_pipeline(&c.pipeline);
        rpass.set_bind_group(0, &c.bind_group, &[]);
        rpass.draw(0..4, 0..1);
        true
    }
```

- [ ] **Step 5: Run the tests**

Run: GPU test command with filter `chroma_gpu`, then `denoise_gpu`.
Expected: all `chroma_gpu_*` pass (printed diffs below `9.8e-4`, backends agree); the 6 `denoise_gpu_*` still pass.

- [ ] **Step 6: Commit**

```bash
git add src/core/gpu/wgpu_chroma_filter.wgsl src/core/gpu/wgpu.rs src/core/gpu/wgpu_interop.rs src/core/denoise/tests_gpu.rs
git commit -m "feat(wgpu): chroma smoothing pass on the render path"
```

---

### Task 6: Wire it into exports and the preview

**Files:**
- Modify: `src/rendering/mod.rs` (the `create_planes_proc!` closure, ~lines 1020-1075; a new classification function near `should_bake_color` ~line 348)
- Modify: `src/core/lib.rs` (~line 950, `process_pixels_with_references`)
- Test: `src/rendering/mod.rs` (existing `#[cfg(test)]` module in that file, or a new `mod chroma_plane_tests`)

**Interfaces:**
- Consumes: `DenoiseSettings::chroma_filter_params` (Task 2), `Buffers::guide`, `FrameTransform::chroma_filter` (Task 3), both passes (Tasks 4-5).
- Produces: `fn chroma_plane_mode<T: PixelType>(plane_index: usize, yuvi: &[usize]) -> i32` returning a `chroma_filter::MODE_*`.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod chroma_plane_tests {
    use super::*;
    use gyroflow_core::denoise::chroma_filter::{ MODE_OFF, MODE_PLANAR, MODE_RGB };
    use gyroflow_core::stabilization::{ Luma8, Luma16, UV8, UV16, RGBA8, RGBA16, R32f, AYUV16 };

    #[test]
    fn classifies_every_dispatched_plane() {
        // NV12 / P010
        assert_eq!(chroma_plane_mode::<Luma8>(0, &[0]), MODE_OFF);
        assert_eq!(chroma_plane_mode::<UV8>(1, &[1, 2]), MODE_PLANAR);
        assert_eq!(chroma_plane_mode::<UV16>(1, &[1, 2]), MODE_PLANAR);
        // NV21: swapped order is still chroma
        assert_eq!(chroma_plane_mode::<UV8>(1, &[2, 1]), MODE_PLANAR);
        // YUV420P / YUV444P16
        assert_eq!(chroma_plane_mode::<Luma16>(1, &[1]), MODE_PLANAR);
        assert_eq!(chroma_plane_mode::<Luma16>(2, &[2]), MODE_PLANAR);
        // YUVA: alpha plane
        assert_eq!(chroma_plane_mode::<Luma16>(3, &[3]), MODE_OFF);
        // packed RGB (preview-like and baked colour)
        assert_eq!(chroma_plane_mode::<RGBA8>(0, &[]), MODE_RGB);
        assert_eq!(chroma_plane_mode::<RGBA16>(0, &[]), MODE_RGB);
        // planar RGB float: plane 2 carries [1], which must not be mistaken for Cb
        assert_eq!(chroma_plane_mode::<R32f>(2, &[1]), MODE_OFF);
        // packed YUV
        assert_eq!(chroma_plane_mode::<AYUV16>(0, &[3, 0, 1, 2]), MODE_OFF);
    }
}
```

If any of these types is not exported from `gyroflow_core::stabilization`, import it from where `create_planes_proc!` already gets it (same `use` as at the top of `rendering/mod.rs`).

- [ ] **Step 2: Run to confirm failure**

Run: `just test`
Expected: compile error, `chroma_plane_mode` not found.

- [ ] **Step 3: Implement the classifier** next to `should_bake_color`:

```rust
/// Which chroma smoothing mode applies to a dispatched plane (`denoise::chroma_filter`).
/// `yuvi` is the plane's YUV component list from the dispatch table: empty for packed RGB,
/// `[0]` for luma, `[1]`/`[2]`/`[1,2]`/`[2,1]` for chroma, `[3]` for alpha. Planar RGB float
/// planes (`R32f`) reuse those indices for G/B/R, so they are excluded by type.
fn chroma_plane_mode<T: PixelType>(plane_index: usize, yuvi: &[usize]) -> i32 {
    use gyroflow_core::denoise::chroma_filter::{ MODE_OFF, MODE_PLANAR, MODE_RGB };
    if yuvi.is_empty() {
        return if T::COUNT >= 3 { MODE_RGB } else { MODE_OFF };
    }
    let planar_rgb_float = T::COUNT == 1 && T::SCALAR_BYTES == 4;
    if !planar_rgb_float && plane_index >= 1 && T::COUNT <= 2 && yuvi.iter().all(|&c| c == 1 || c == 2) {
        MODE_PLANAR
    } else {
        MODE_OFF
    }
}
```

- [ ] **Step 4: Use it in the export closure.** Beside `let packed_rgb = …` (outside the closure, where `$yuvi` is in scope), add:

```rust
                    let yuvi_owned: Vec<usize> = { let yuvi: &[usize] = &$yuvi; yuvi.to_vec() };
```

Inside the closure, before `let mut buffers = Buffers {`, compute the mode for this call and take the guide. `plane_index` is the `run_planes` enumeration index, and plane 0 of `out_frame_data` has already been written, since planes run in order:

```rust
                        let mode = chroma_plane_mode::<$t>(plane_index, &yuvi_owned);
                        // Only to decide whether a guide is needed; the real parameters are set
                        // below, once `max_pixel_value` is final for the chosen backend.
                        let cf = denoise.read().chroma_filter_params(mode, 1.0);
                        // SAFETY: plane 0 and `plane_index` (>= 1 when `guide` is used) are distinct
                        // allocations inside the frame; the guide is only read, the output plane only written.
                        let guide_slice: Option<(&[u8], (usize, usize, usize))> = if cf.is_active() && mode == gyroflow_core::denoise::chroma_filter::MODE_PLANAR {
                            let (w, h, s) = (out_frame_data.plane_width(0) as usize, out_frame_data.plane_height(0) as usize, out_frame_data.stride(0));
                            let data = out_frame_data.data(0);
                            Some((unsafe { std::slice::from_raw_parts(data.as_ptr(), data.len()) }, (w, h, s)))
                        } else { None };
```

Add `guide:` to the `Buffers` literal (replacing the `guide: None` from Task 3):

```rust
                            guide: guide_slice.map(|(b, size)| BufferDescription {
                                size,
                                // The CPU upload path only reads it; the `&mut` is what the type requires.
                                data: BufferSource::Cpu { buffer: unsafe { std::slice::from_raw_parts_mut(b.as_ptr() as *mut u8, b.len()) } },
                                ..Default::default()
                            }),
```

After the `denoise_sigma` block (where `transform.kernel_params.max_pixel_value` is final for this backend):

```rust
                        transform.chroma_filter = denoise.read().chroma_filter_params(mode, transform.kernel_params.max_pixel_value);
```

- [ ] **Step 5: Preview.** In `src/core/lib.rs` `process_pixels_with_references`, inside the `if T::COUNT >= 3 { … }` block that sets `denoise_rgb`, add:

```rust
                    transform.chroma_filter = denoise.chroma_filter_params(denoise::chroma_filter::MODE_RGB, transform.kernel_params.max_pixel_value);
```

This covers only the paused preview (the path with references), which matches the temporal pass.

- [ ] **Step 6: Build and run**

Run: `just build`, `just test`, `cd src/core; cargo test --lib`, then the GPU test command with filter `gpu`.
Expected: all green; `classifies_every_dispatched_plane` passes.

- [ ] **Step 7: Commit**

```bash
git add src/rendering/mod.rs src/core/lib.rs
git commit -m "feat(denoise): run chroma smoothing on export chroma planes and the paused preview"
```

---

### Task 7: Verify on real footage, then tidy up

**Files:**
- Modify: `docs/superpowers/specs/2026-09-27-chroma-smoothing-design.md` (record the results)
- Scratch: `%TEMP%\claude\…\scratchpad\dn\` (existing prototype scripts: `frames.py`, `proto.py`)

- [ ] **Step 1: Save the user's log, then render.** Copy `%LOCALAPPDATA%\Gyroflow\gyroflow.log` into the scratchpad first (CLI launches truncate it). Render `testfiles\C2520.MP4` three times via the CLI (`Start-Process … -Wait`, `-p "{ 'codec': 'H.265/HEVC', 'bitrate': 200, 'use_gpu': true, 'audio': false, 'pixel_format': 'P010LE', 'output_folder': '<scratch>/renders' }"`), with `--denoise high` and `--denoise-chroma-smoothing 0`, `0.5`, `1`, suffixes `_cs0`, `_cs50`, `_cs100`. Record each `Done in` time from the log.

- [ ] **Step 2: Measure.** In `dn/`, add to a copy of `proto.py`'s `noise_stats` usage: frame 8, the trousers patch `(slice(2560,2760), slice(760,1000))`. Print U and V std for `cs0`, `cs50`, `cs100`.
Expected: `cs0` ≈ temporal-only (U 2.31, V 3.23); `cs100` V ≤ 2.0. If not, stop and report, do not tune the constants to pass.

- [ ] **Step 3: Visual check.** Produce the shoe-edge sheet (as in `chroma_only.py`: chroma-plane crop `y=1300, x=420, n=110`, chroma ×6 over stretched luma) for `cs0` and `cs100`. Look at it: colour must stay inside the shoe, no fringe.

- [ ] **Step 4: Record the results** in the spec under a new `## Results` section: the table of U/V per setting, the three export times, and one line on the visual check.

- [ ] **Step 5: Clean up** the scratch renders (`renders\*.mp4`) and the `dn` prototypes. They are throwaway.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-09-27-chroma-smoothing-design.md
git commit -m "docs: chroma smoothing results on real footage"
```
