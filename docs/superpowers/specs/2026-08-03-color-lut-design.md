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

The colour stage assumes its input is RGB, and no kernel converts colour spaces to make that true. This was an error in an earlier draft of this spec, which claimed YUV pixel types (notably `AYUV16`) convert to RGB at the top of the block and back at the bottom. They do not, and never did: the only colour-space-aware code in the whole pixel path is `remap_colorrange`, a full↔limited *range* remap, and `PixelType::from_rgb_color`, which converts the *background colour* into the frame's native space. The Rec709 matrix at `pixel_formats.rs:26` is used only by the latter.

Feeding a YUV pixel type to the colour stage would therefore grade Y, U and V as if they were R, G and B — an exposure gain becomes a hue rotation, because U and V are offset by 0.5. This is prevented structurally rather than by convention: the render path converts to packed RGB before baking (see the table below), and `create_planes_proc!` forces `color.enabled` off for any plane whose `PixelType::COUNT` is under 3.

### Render path

When colour is enabled and baking is on, `rendering/mod.rs` selects the plane dispatch as follows:

| Source | Behaviour |
|---|---|
| 8-bit (NV12, NV21, YUV420P, YUVJ420P…) | convert → `Pixel::RGBA`, single `RGBA8` plane |
| Deeper integer (P010LE, YUV444P12LE, YUVA444P16LE, `AYUV64LE`…) | convert → `Pixel::RGBA64LE`, single `RGBA16` plane |
| Already packed RGB (`RGB24`, `RGBA`, `RGB48BE`, `RGBA64BE`) | unchanged, no conversion; graded in place |
| Float (`GBRPF32LE`, `GBRAPF32LE`) | baking refused, warning logged; see open question OQ1 |
| Hardware frames, or formats swscale cannot read | baking refused, warning logged |
| Baking off | today's per-plane path, byte-for-byte unchanged |

The target is packed **RGB**, not `AYUV64LE`. No kernel performs any colour-space conversion — `apply_color` grades whatever three channel values it is handed — so a YUV target would run exposure, white balance, contrast and saturation over Y, U and V as if they were R, G and B. U and V are offset by 0.5, which turns an exposure gain into a hue rotation and a saturation change into a chroma shift on the wrong axis. It would also make the preview/export match impossible by construction, since the Qt preview grades RGBA8 in RGB.

The target must be packed rather than planar for the other half of the reason: a planar dispatch hands the kernel one channel per plane, and the colour stage needs all three at once.

Source precision is preserved — 8-bit sources stay 8-bit, deeper sources go to 16-bit — and the decision is made once, from the source format, and reused by both the plane construction and the per-frame dispatch so the two cannot disagree about the layout the plane closures were built for.

Independently of the format choice, `create_planes_proc!` forces the colour stage off for any plane whose `PixelType::COUNT` is below 3. A single-channel (`Luma8`/`Luma16`/`R32f`) or two-channel (`UV8`/`UV16`) plane therefore cannot have an RGB grade run on it whatever the flags say, which is what makes the "baking refused" rows above safe rather than merely intended.

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

### R5 acceptance — executed, export half only

Run headless through the CLI on Windows, against a 4K 25p Sony Cinema Line clip (`YUV422P10LE`, `color_range=pc`, `color_space=unknown`) and a 33³ S-Log3/S-Gamut3.Cine→Rec.709 `.cube`:

```
gyroflow CLIP.MP4 --preset color-on.json -t _lutbaked -f \
    -p "{'pixel_format':'P010LE','codec':'H.265/HEVC','bitrate':60}"
```

with a matching `color-off.json` control. The log confirms the dispatch:

```
[INFO] Baking colour adjustments: converting P010LE to RGBA64LE
```

The graded frame is a correct log-to-display conversion: black point restored, warm tone recovered in the highlights, saturation plausible, no hue rotation. A hue rotation is the signature failure of grading YUV as RGB, which is what the plan originally specified (`AYUV64LE`) and what the packed-RGB switch exists to prevent, so its absence is the load-bearing observation here.

This exercises, for the first time on real data: the packed-RGB conversion and its precision choice (10-bit → `RGBA64LE`), the LUT upload at `LUT_DATA_OFFSET` with 33³ trilinear interpolation, colour deserialised from a preset, and `bake_in_render` being honoured. The source is full-range with an unspecified matrix, so it also traverses the swscale colour-range path and its BT.709 fallback without visible level error.

**Not covered by this run, and still owed:** the preview half of R5 (the CLI opens no preview, so preview/export match remains unverified) and numerical comparison against a reference implementation — the result is visibly right, not measured.

### Cross-backend equivalence — the planned test cannot be run through full renders

`NO_OPENCL` and `NO_WGPU` (`gpu/mod.rs:375`) select the backend, so the same clip and preset were rendered through each and compared frame-for-frame as 16-bit RGB.

First, the pipeline is **bit-exact deterministic**: two OpenCL renders of the same input produce byte-identical output (equal SHA-256 over the decoded frame). The lossy encoder contributes no noise, so any difference between backends is real signal rather than compression.

Measured on one frame (frame 72 of 144), per-channel absolute difference in 16-bit terms:

| comparison | max | mean | samples differing |
|---|---|---|---|
| OpenCL vs OpenCL (same backend, rerun) | 0 | 0 | 0% |
| OpenCL vs wgpu, **colour disabled** | 4822 (≈1/14 scale) | 180 | 92.9% |
| OpenCL vs wgpu, **colour enabled** | 9469 (≈1/7 scale) | 300 | 96.3% |

The colour-disabled row is the load-bearing one: **the two backends already disagree by roughly 75× the 1/1024 threshold before any colour code runs.** That divergence is pre-existing — different undistortion and Lanczos4 resampling implementations — and is not attributable to this branch.

Colour roughly doubles the divergence, which is what an S-Log3→Rec.709 LUT should do to a pre-existing difference: the curve is steep through shadows and midtones, so it amplifies whatever disagreement it is handed. That is the most plausible reading, but **this measurement cannot separate amplification from a genuine colour-stage divergence**, because the baseline is already far above the threshold.

The consequence for the planned test is concrete: cross-backend equivalence must be asserted on **a synthetic frame through the colour stage in isolation**, exactly as this spec's testing section originally specified, and never by diffing full renders. A full-render diff can only ever measure the pipeline's pre-existing backend disagreement. That isolated test remains unwritten and unrun.

**CPU-only rendering does not work at all.** Setting both `NO_OPENCL=1` and `NO_WGPU=1` fails with `No cached wgpu found for key: …` rather than falling back to the CPU pixel path, so the CPU backend could not be included above. Reproduced with colour disabled, so pre-existing and not attributable to this branch — but it does mean the CPU colour stage (Task 8) has never been executed through a real render, only through unit tests.

Two pre-existing obstacles found while doing this, neither caused by colour and both reproducible with colour disabled:

- `YUV422P10LE` is not among the supported render output formats, so a stock Sony Cinema Line clip cannot be exported without overriding `pixel_format`. Worth its own issue.
- The CLI aborts on a debug-only null-pointer assertion: `RenderQueue` is a QObject constructed at `cli.rs:222`, but `QCoreApplication` is not created until `cli.rs:645`. Release builds do not trip it. Also worth its own issue.

## Known kernel divergence — colour stage placement

Found by review of the finished branch, confirmed by reading, **not fixed**. The CPU path applies the colour stage once to the final pixel (`cpu_undistort.rs:641,655`). The wgpu and OpenCL kernels apply it inside `sample_input_at` (`wgpu_undistort.wgsl:370`, `opencl_undistort.cl:483`), which places it before compositing rather than after. Two consequences:

- **Background pixels.** When `undistort_coord` returns its sentinel — the `r_limit`-clipped region on fisheye profiles — the GPU kernels leave the background ungraded, while the CPU path and the Qt preview (`qt_gpu/undistort.frag:416`) grade it. With colour enabled those arcs differ between a preview and a GPU render.
- **The feather band.** In `background_mode == 3`, the GPU kernels grade the two samples and then blend; the CPU path blends and then grades. Exposure, white balance, contrast and saturation are affine and commute with the blend, so this is invisible for them — but `sample_trilinear` is not, so **with a LUT loaded the feather band renders differently on GPU than on CPU.**

This breaks the invariant the five-way duplication rests on: the implementations must be arithmetically identical, and reviewers verify that rather than flagging the duplication. The correct target is the CPU order — grade the composited pixel once — because that is what the reference `apply_color` describes and what the preview already does.

It is deliberately left for a change that can be verified. Moving the call site in two GPU kernels is exactly the kind of edit that needs the cross-backend equivalence test to confirm, and that test's wgpu identity control is not yet solved (see below), so a fix today could not be checked by anything except reading. The edge cases involved — clipped fisheye arcs, and the feather band with a LUT — are narrow enough to be worth less than a silently wrong "fix".

## Manual verification checklist

Everything below needs a human at a running GUI. The export path has been verified headlessly through the CLI (see the R5 section above), but **the CLI opens no preview and no dialogs, so the entire UI layer is unobserved** — the panel has never been rendered, no `.cube` has been loaded through a file dialog, and no `messageBox` or notification in this feature has ever fired.

1. **Panel appears and is neutral.** Load any video. "Color adjustments" appears in the right panel below "Stabilization", opens, and every slider sits at neutral (exposure 0 EV; contrast/saturation/tint 0; temperature 5500 K or the clip's source Kelvin). Preview unchanged. The `pencil` icon renders and is not blank.
2. **Exposure changes the preview.** Tick Enable, drag exposure — the preview must brighten immediately. This is the first real proof that `request_recompute` reaches the *preview*; Task 14 could only argue it from the call graph, and the CLI render does not exercise it.
3. **Load a valid `.cube`.** The preview changes, the button label becomes the filename, the "LUT strength" slider appears and visibly scales the effect.
4. **Load a truncated `.cube`.** The parse error appears in a dialog and the *previous* LUT stays active with the preview untouched.
5. **Source Kelvin.** On a clip with white-balance metadata the temperature slider and its reset default both read the source Kelvin, and the "Source: …" line shows it with ISO/shutter/aperture/EV. A clip without it reads 5500 K and shows no Source line. Then, with the panel **enabled** and a temperature dialled in, load a second clip and confirm the dialled-in value is not overwritten — this is the ratified anchoring behaviour and the case flagged as a rough edge.
6. **Project round-trip.** Distinct non-default values on every slider plus a LUT, save, restart, reopen — everything returns exactly.
7. **Preset without a LUT (F3).** Applying a preset carrying a `color` section but no LUT must leave the loaded LUT alone while applying the scalars.
8. **Preset with an unresolvable LUT (F3).** A preset whose `lut_path` does not exist and whose `lut_data` is absent must keep the current LUT and show a non-blocking warning naming both — not an error dialog.
9. **Keyframes.** Right-click each slider → Enable keyframing → Add keyframe. The timeline value must match the slider. A mismatch shows as 100× on contrast, saturation, tint and LUT strength — the `scaler` mapping the plan got wrong.
10. **Preset authoring.** Save a preset and confirm the new "Color adjustments" category appears with its six rows, that the LUT row is unticked by default, and that ticking it embeds the LUT.
11. **Mobile layout.** With `forceMobileLayout`, the panel and its `Hr` appear in the Parameters tab below Stabilization.
12. **Localisation/RTL**, including the "Choose file…" ellipsis, in a non-English locale.

On macOS, Android and iOS the panel is compiled in through `qmlcachegen` rather than the `resources_qml.rs` qrc list. Both lists are updated, but that is a different code path from the one exercised on Windows and has never been run.

## Verification owed on other platforms

Everything below is inert on Windows and Linux and was written without any way to execute it. It is shipped deliberately, on the reasoning recorded here, and needs a human on the named platform to confirm.

**ProRes VideoToolbox range under baking (macOS/iOS).** `prores_videotoolbox` is tagged limited-range at the encoder (`ffmpeg_video.rs:108`, `cfg!`-gated to Apple), and the kernel-side `FIX_COLOR_RANGE` remap is what makes the samples match that tag. Under baking the kernel only ever sees packed RGB, where a YUV range remap is meaningless, so `tag_limited_range_for_prores_videotoolbox` (`rendering/mod.rs:271`) declares the range on the output frame instead and lets swscale fold it into the RGB→YUV matrix. Gated at runtime on the codec name, so it compiles and ships everywhere and is simply unreachable off Apple.

To confirm: render a full-range NV12 source to ProRes twice, once with "apply to rendered video" on and once off, and check the black and white levels match. If they do not, the baking path is writing full-range samples into a frame the encoder declares limited. CI will not catch this — `release.yml` has a macOS runner, but it builds and packages rather than rendering and inspecting.

**Note a pre-existing asymmetry, out of scope here:** the same encoder-side workaround also covers `dnxhd`, which never had a kernel-side counterpart at all. That predates this branch and is unchanged by it, but suggests a latent levels bug for `dnxhd` on Apple worth its own issue.

**Android/Apple LUT paths.** `lut_path` stores an encoded URL rather than a filesystem path, because content URIs and security-scoped URLs have no path. The round-trip through the project importer is unit-tested, but loading a `.cube` through the platform file picker on Android or iOS has never been executed.

## Open questions

**OQ1 — packed float format.** Resolved: swscale cannot output it, so baking is refused for float sources; they render with colour adjustments skipped and a warning in the log.

The earlier answer — that the format does not exist in the bindings — was wrong, and was reached by reading ffmpeg-sys-next on GitHub with no toolchain available. `Pixel::RGBAF32LE` *is* in the vendored ffmpeg-next 8.1 (`util/format/pixel.rs:402`), and `RGBAf` has been a first-class `PixelType` all along (`src/core/stabilization/pixel_formats.rs:58`).

The real obstacle is that enum membership is not converter support. Re-verified empirically against the linked ffmpeg 8.1 in Task 12: `sws_isSupportedOutput(AV_PIX_FMT_RGBAF32LE)` returns 0, and building a `YUV420P → RGBAF32LE` scaling context fails with `128bpp not supported by yuv2rgb` / `rgbaf32le is not supported as output pixel format`. With no way to get a float source into a packed RGB frame, and 16-bit being the one thing a float source exists to avoid, `packed_color_format` returns `None` for any format ffmpeg marks `AV_PIX_FMT_FLAG_FLOAT`. The per-plane gate then keeps the colour stage off the `R32f` planes those sources dispatch to. The test `swscale_still_cannot_output_packed_float` (`src/rendering/mod.rs`) records this, and will fail if a future ffmpeg gains the support — at which point the float path can simply be implemented.

## Out of scope

Lift/gamma/gain per channel, highlights/shadows recovery, 1D LUTs, LUT formats other than `.cube`, scopes or histograms, colour-managed working spaces, and per-clip colour in the render queue beyond what the project file already carries.
