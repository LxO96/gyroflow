# Colour adjustments and 3D LUT support — design

**Date:** 2026-08-03
**Status:** Approved, ready for implementation planning

## Summary

Add a colour stage to Gyroflow: a 3D `.cube` LUT plus exposure, contrast, saturation and white balance. Adjustments always affect the preview; a per-project toggle controls whether they are baked into the rendered file. All values except the LUT file itself are keyframable.

The colour maths lives in `gyroflow-core`, so the CLI, the render queue and the NLE plugins inherit it.

## Requirements

| # | Requirement |
|---|---|
| R1 | Load Adobe/IRIDAS `.cube` 3D LUTs; 1D LUTs are rejected with a clear message |
| R2 | Exposure (stops), contrast, saturation, white balance (absolute Kelvin) and tint |
| R3 | Adjustments always apply to the preview |
| R4 | A user toggle controls whether they are baked into the render |
| R5 | Preview and baked render must agree visually |
| R6 | All scalars keyframable, including LUT strength; the LUT file is not |
| R7 | Order of operations: LUT first, then all adjustments |
| R8 | Projects store the LUT path *and* an embedded copy as fallback |
| R9 | Source white balance / exposure metadata is displayed, and the temperature slider is anchored to the source Kelvin |
| R10 | Zero behavioural or performance change when the feature is disabled |

## Constraints discovered in the codebase

These drive the whole design and should be re-verified before implementation.

1. **The render path processes YUV planes separately.** `src/rendering/mod.rs:517` dispatches the undistort kernel once per plane (Y as `Luma8`, chroma separately), so the export kernel never sees R, G and B together. A 3D LUT is impossible in that configuration.
2. **The preview path is different.** `src/qt_gpu/qrhi_undistort.cpp:69` operates on a full RGBA8 texture from MDK, so preview always has full colour regardless of source format.
3. **The pixel kernel exists in five hand-maintained implementations:** `core/gpu/wgpu_undistort.wgsl`, `core/gpu/opencl_undistort.cl`, `core/gpu/stabilize_spirv/` (rust-gpu), `core/stabilization/cpu_undistort.rs`, and `qt_gpu/undistort.frag`. There is no shared codegen; `shader_builder` only compiles the rust-gpu crate to SPIR-V. Any shader-side maths is written five times.
4. **A format-conversion wrapper already exists.** `rendering/mod.rs:676` already routes unknown formats through `Converter::convert_pixel_format`, and a packed single-plane `AYUV16` dispatch already exists at `mod.rs:635`.
5. **Camera metadata is already extracted.** `gyro_source/mod.rs:320` builds `additional_data.recording_settings` with White balance, ISO, shutter, iris and colour primaries, delivered to QML via `telemetry_loaded` (`controller.rs:74`).
6. **`src/core/` has no tests.** Zero `#[test]` functions; the `test-case` dev-dependency is unused.

## Approach

**One RGB colour stage in core; the render path switches to a packed pixel format when baking is enabled.**

Rejected alternatives:

- *YUV-native adjustments per plane, packed format only for the LUT.* Faster for the no-LUT case, but requires two independent colour implementations that cannot agree exactly — luma-only contrast desaturates differently than RGB contrast, so toggling a LUT would visibly shift the image. Fails R5.
- *App-layer only: ffmpeg `lut3d`/`eq` filters for export, custom shader for preview.* Least code, but ffmpeg filters cannot be keyframed per frame (fails R6), its `lut3d` interpolation would not match ours (fails R5), and nothing would reach the NLE plugins or the CLI.

## Architecture

### Data model

`src/core/color/lut.rs`
```rust
pub struct Lut3D { pub size: u32, pub data: Vec<f32> }  // size³ × 3, RGB-major
pub fn parse_cube(text: &str) -> Result<Lut3D, LutError>
```
Handles `LUT_3D_SIZE`, `DOMAIN_MIN`/`DOMAIN_MAX`, `TITLE` and comments.

`src/core/color/mod.rs`
```rust
pub struct ColorAdjustments {
    pub enabled: bool,
    pub bake_in_render: bool,
    pub exposure: f64,               // stops, -5..5, 0 neutral
    pub contrast: f64,               // -1..1, 0 neutral
    pub saturation: f64,             // -1..1, 0 neutral
    pub source_kelvin: f64,          // anchor, from metadata or 5500 default
    pub source_kelvin_detected: bool,
    pub target_kelvin: f64,          // what the slider shows
    pub tint: f64,                   // -1..1
    pub lut_strength: f64,           // 0..1
    pub lut: Option<Arc<Lut3D>>,
    pub lut_path: Option<String>,
}
```
With a `get_checksum()` mirroring `SmoothingAlgorithm`, so existing cache invalidation picks up colour changes.

`src/core/color/source_metadata.rs`
```rust
pub struct SourceExposure {
    pub white_balance_k: Option<f64>,
    pub wb_mode: Option<String>,
    pub iso: Option<f64>,
    pub shutter_s: Option<f64>,
    pub aperture: Option<f64>,
    pub ev: Option<f64>,             // computed only when iso+shutter+aperture all present
}
```
Parsing takes the leading numeric run and returns `None` on failure, so `"5500"`, `"5500K"` and `"5500 K"` all work while `"Auto"` yields `None` for Kelvin but still sets `wb_mode`.
EV is `log2(N²/t) − log2(ISO/100)`, omitted rather than guessed when any input is missing.

`StabilizationManager` gains `pub color: Arc<RwLock<ColorAdjustments>>` alongside `gyro`/`lens`/`smoothing` (`core/lib.rs:83`). `ComputeParams::from_manager` copies the scalars out.

All defaults are zero-effect with `enabled: false`, satisfying R10.

### Keyframes

Six new `KeyframeType` variants via the macro at `core/keyframes.rs:26`: `ColorExposure`, `ColorContrast`, `ColorSaturation`, `ColorTemperature`, `ColorTint`, `ColorLutStrength`. Evaluated per frame in `FrameTransform::at_timestamp`, as `LensCorrectionStrength` already is.

### White balance maths (Rust, not shaders)

The shader never sees Kelvin. Once per frame, in `FrameTransform::at_timestamp`:

```
(target_kelvin, tint) ──► CCT→xy (Planckian approximation) ──► XYZ
                      ──► linear sRGB white points
                      ──► gains = white(source_kelvin) / white(target_kelvin), luma-normalised
                      ──► KernelParams.wb_gains: vec3
```

This keeps the colour science in one testable Rust location instead of five shader languages, and keyframing works unchanged because gains are recomputed per frame regardless.

When the source has no parsed Kelvin, the anchor defaults to 5500 K and the UI labels it as assumed. The slider behaves identically in both cases — one code path, not two.

### Pixel pipeline

**Insert point:** end of the kernel, after the resampling loop produces `sum`, before `remap_colorrange` and the final clamp (`wgpu_undistort.wgsl:290` and its four siblings). Colour-range remapping stays last. Cost is one evaluation per *output* pixel, not per sample tap.

**Parameters:** two `vec4`s appended to the end of `KernelParams` — `color_params_1` (exposure, contrast_factor, saturation_factor, lut_strength) and `color_params_2` (wb_gains.rgb, lut_size). Appending preserves the required 16-byte alignment. Two new flags: `COLOR_ENABLED = 1 << 12`, `HAS_LUT = 1 << 13`.

The sync comment at `stabilization/mod.rs:102` currently lists only three files and must be corrected to include `stabilize_spirv/src/types.rs`.

**LUT delivery** differs by shader family because the available resources differ:

- *Core backends* (wgpu, rust-gpu, CPU): append the LUT to the existing `coeffs` buffer. It is currently a fixed 488-float const uploaded immutably (`gpu/wgpu.rs:306`); it becomes a sized `COPY_DST` buffer of `488 + size³ × 3` floats, rewritten only when the LUT changes. These three need no new binding — `coeffs` is already plumbed through them (wgpu bind group entry 2; rust-gpu `stabilize_spirv/src/lib.rs:46` storage buffer binding 2; CPU path passes the slice directly).
- *OpenCL* is the exception, discovered during implementation and corrected here: it has **no `coeffs` buffer at all**. The interpolation tables are `__constant` literals compiled into `opencl_undistort.cl`, never uploaded at runtime, and `undistort_image` (`opencl_undistort.cl:582`) takes no `coeffs` argument. OpenCL therefore needs a genuinely new kernel argument plus host-side upload, which is done in the same task that adds its colour stage rather than alongside the other backends' buffer work.
- *Qt RHI preview*: `undistort.frag` has no storage buffers, only `sampler2D`s (`undistort.frag:61-63`). The LUT arrives as a new `texLut` with the 3D grid tiled into 2D, read with `texelFetch`.

**Trilinear interpolation is manual in all five implementations**, never hardware sampler filtering, which differs between vendors and would not match the CPU path. Eight fetches and seven lerps, identical arithmetic everywhere.

**Per-pixel maths:**
```
rgb = mix(rgb, trilinear(lut, rgb), lut_strength)
rgb *= exp2(exposure)
rgb *= wb_gains
rgb  = (rgb - 0.5) * contrast_factor + 0.5
luma = dot(rgb, [0.2126, 0.7152, 0.0722])
rgb  = mix(luma, rgb, saturation_factor)
```

`contrast` and `saturation` are stored as −1..1 with 0 neutral and both map to a multiplier identically: `factor = 1 + value`, giving 0× at −1 (flat grey / fully desaturated) and 2× at +1. Both mappings are applied in Rust so the shaders receive ready multipliers, consistent with how `wb_gains` is handled. The contrast pivot is 0.5, display-referred, which is appropriate because contrast runs after the LUT.

YUV pixel types (notably `AYUV16`) convert to RGB at the top of this block and back at the bottom; `pixel_formats.rs:26` already holds the Rec709 matrix on the Rust side and the shaders use the same constants.

### Render path

When colour is enabled and baking is on, `rendering/mod.rs` selects the plane dispatch as follows:

| Source | Behaviour |
|---|---|
| Planar YUV (NV12, YUV420P, P010, YUV444P10…) | convert → `AYUV64LE`, single `AYUV16` plane |
| Already packed and full-colour (`AYUV64LE`, `RGB24`, `RGBA`, `RGB48BE`, `RGBA64BE`) | unchanged, no conversion |
| Planar RGB (`GBRPF32LE`, `GBRAPF32LE`) | convert → packed float; see open question OQ1 |
| Baking off | today's per-plane path, byte-for-byte unchanged |

`AYUV64LE` is chosen over an RGB target because it is 16-bit (no banding from the grade) and stays in YUV, so swscale only upsamples chroma and packs rather than doing a full colour-space round trip that can clip out-of-gamut values.

The preview path needs none of this — MDK already provides RGBA8.

### UI

New `src/ui/menu/ColorAdjustments.qml`, registered in `ui/menu/qmldir`, instantiated as an `ItemLoader` in the right-hand `paramsTab` next to `Menu.Stabilization` (`App.qml:510`). Uses the existing `SliderWithField`, `CheckBoxWithContent`, `FileDialog` and `InfoMessageSmall` components.

```
▼ Color adjustments                    [ ] Enable
    Source   5500 K · ISO 400 · 1/50 · f/2.8 · +1.0 EV
    ────────────────────────────────────────
    LUT      [ Choose file… ] name.cube  [×]
    Strength ▬▬▬▬▬▬▬●───  100%
    ────────────────────────────────────────
    Exposure     ───●───   0.00 EV
    Contrast     ───●───   0
    Saturation   ───●───   0
    Temperature  ───●───   5500 K
    Tint         ───●───   0
    [ ] Apply to rendered video
    [Reset]
```

The source row hides entirely when nothing parsed (the common GoPro/DJI case). When baking is off, an inline `InfoMessageSmall` states the grade is preview-only.

Controller surface, following `set_smoothing_param`:
```rust
set_color_param:   qt_method!(fn(&self, name: QString, val: f64)),
set_color_enabled: qt_method!(fn(&self, enabled: bool, bake_in_render: bool)),
load_lut:          qt_method!(fn(&mut self, url: QUrl) -> QString),  // "" or error message
clear_lut:         qt_method!(fn(&mut self)),
```
`load_lut` returns the error message rather than a bool so the panel can explain *why* a file was rejected.

### Project file

A new top-level `"color"` object in `export_gyroflow_data` (`core/lib.rs:1285`), read back in `import_gyroflow_data`. Import already uses tolerant `obj.get(...)` lookups, so the file version stays at **4**: older builds ignore the key and projects without it load with colour disabled.

```json
"color": {
  "enabled": true,
  "bake_in_render": false,
  "exposure": 0.0, "contrast": 0.0, "saturation": 0.0,
  "source_kelvin": 5500, "source_kelvin_detected": true,
  "target_kelvin": 6200, "tint": 0.0,
  "lut_strength": 1.0,
  "lut_path": "D:/luts/gopro_flat_to_709.cube",
  "lut_name": "gopro_flat_to_709.cube",
  "lut_data": "<compressed>"
}
```

LUT embedding reuses `util::compress_to_base91_cbor` / `decompress_from_base91_cbor` (`lib.rs:1386`). A 33³ LUT is ~107k floats, roughly 100–200 KB compressed — comparable to the gyro payloads already present.

On load: try `lut_path` first, fall back to `lut_data`. If the path resolves but differs from the embedded copy, prefer the file and log it.

Storing `source_kelvin` and `source_kelvin_detected` means a project renders identically even if metadata parsing later changes or the source file is swapped.

Presets need no extra work — they use the same export/import path.

The CLI needs no new flags. The existing `--preset` option (`cli.rs:88`) accepts a file or inline JSON and already merges arbitrary project sections, so `{"color": {…}}` works with no CLI changes; baking obeys the project.

## Error handling

`LutError` via `thiserror`, folded into `GyroflowCoreError`:

| Condition | Behaviour |
|---|---|
| `LUT_1D_SIZE` present | Rejected: "1D LUTs are not supported" |
| Row count ≠ `size³` | Rejected, reporting expected vs actual |
| `size` outside 2–64 | Rejected (64³ is already 3 MB of floats) |
| NaN or infinity in data | Rejected |
| Unparseable line | Rejected, reporting the line number |

LUT loading is **atomic**: parse fully into a new `Lut3D`, then swap the `Arc`. A failed load leaves the previous LUT and the preview untouched.

Non-parser failures:

- **Project references a missing LUT with no embedded copy** → import succeeds with colour disabled plus a UI warning; never a failed import.
- **Float source with no packed float format available** (OQ1) → baking refused for that render with an explicit message, rather than silently clipping HDR range.

## Testing

`src/core/` has no tests today, so this introduces the first ones. Scope is limited to what pays for itself.

Runs anywhere (pure functions, no GPU):

- `.cube` parser: valid 2³/17³/33³, `DOMAIN_MIN`/`DOMAIN_MAX`, comments, CRLF, whitespace variance, and each rejection above. Fixtures in `src/core/color/tests/`; small LUTs inline as strings.
- Kelvin→gains: source == target yields exactly `(1,1,1)`; warming raises red and lowers blue; luma stays normalised across 2000–15000 K.
- EV computation: known triples (ISO 100, 1/100, f/2.8 → EV 9.5) and `None` when any input is missing.
- Metadata parsing: `"5500"`, `"5500K"`, `"5500 K"`, `"Auto"`, `""`, garbage.
- Identity LUT is a no-op through the CPU pixel path within 1e-6.
- Project round-trip: export → import preserves every scalar and the embedded LUT bit-for-bit.

Requires hardware, `#[ignore]`d and run explicitly with `cargo test -- --ignored`:

- Cross-backend equivalence: the same synthetic frame and params through CPU, wgpu and OpenCL, asserting max per-channel difference under 1/1024. CI runners have no usable GPU, so this is a pre-release check rather than a gate.

Manual acceptance (no automated test can assert it): load a log clip, set a LUT and a grade, screenshot the preview, render with baking on, compare a frame. This is the acceptance criterion for R5 and belongs in the plan as an explicit step.

Performance: measure a 4K render before and after with baking on, so the documented cost is measured rather than estimated.

## Open questions

**OQ1 — packed float format.** Resolved: unavailable; baking is refused for `GBRPF32LE`/`GBRAPF32LE` sources with the message 'Colour baking is not supported for 32-bit float sources'.

Investigation found that while FFmpeg 8.1's libavutil defines `AV_PIX_FMT_RGBAF32LE` and `AV_PIX_FMT_RGBF32LE`, the ffmpeg-sys-next Rust bindings used by this project do not expose these formats. A repository search confirms no RGBF32 variants are defined in ffmpeg-sys-next.

## Out of scope

Lift/gamma/gain per channel, highlights/shadows recovery, 1D LUTs, LUT formats other than `.cube`, scopes or histograms, colour-managed working spaces, and per-clip colour in the render queue beyond what the project file already carries.
