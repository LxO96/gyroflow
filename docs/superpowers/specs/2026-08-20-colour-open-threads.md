# Colour / LUT — open threads

> **2026-08-22: §1 is RESOLVED — Gyroflow was not wrong.** The exported file, decoded
> honouring its own `color_range=pc` tag, matches the preview to within 0.0007 mean and
> 0.0008 std (fit: `file = 0.954 × preview + 0.022`). Decoded *as if limited-range*, it
> reproduces the reported difference exactly: `1.108 × preview − 0.047`, pivot 0.433, against
> `1.133 / 0.417` measured independently from the user's own screenshots. The viewer was
> expanding a full-range file.
>
> The real defect is that the export is **under-described**. `color_range=pc` is written, but
> `color_space`, `color_transfer` and `color_primaries` are all `unknown`, inherited from the
> source (`ffmpeg_video.rs:128-137`, `548-551`). With no matrix declared, players fall back to
> defaults and many assume limited range for 10-bit YUV, ignoring the range tag. Fix is a
> tagging policy decision, not a colour-maths change — see §1 below.

Handoff note, 2026-08-20; updated 2026-08-24. Branch `wgpu-identity-investigation`, 13 commits
on top of `master`.
`master` (`beb392a5`) already carries the merged colour work and is 50 commits ahead of
`origin/master` — nothing here has been pushed.

The design spec is `2026-08-03-color-lut-design.md` in this directory; it holds the manual
verification checklist, the cross-backend results and the platform verification debt. This file
is only the *open* work, written so it can be picked up cold.

---

## 1. The live bug: the grade looks stronger in the export than in the preview

**Status: root cause not found. Localised, with one cheap diagnostic left that needs a human.**

### What is established

The export is arithmetically correct. This was measured, not argued. The test preset has neutral
exposure/contrast/saturation and `source_kelvin == target_kelvin`, so `apply_color` collapses to
*just the LUT* — which makes graded output exactly predictable from ungraded input.

Rendered `testfiles/C3105.MP4` twice (graded and ungraded), extracted the same frame from each as
16-bit RGB, and applied the reference trilinear LUT to the ungraded pixels in Python:

| quantity | 16-bit LSB |
|---|---|
| predicted vs actual graded, mean | 239 |
| predicted vs actual graded, max | 2271 |
| magnitude of the grade itself, mean | 7258 |

About 97% of the effect is accounted for; the residual is consistent with 10-bit source
quantisation and lossy encoding. **The export applies the LUT as specified.** If one of the two
sides is wrong, it is the preview.

### Ruled out, with the reason

Each of these was a live candidate. None of them is the cause — do not re-tread them.

- **The two `apply_color` implementations differing.** `qt_gpu/undistort.frag:132` and
  `core/gpu/wgpu_undistort.wgsl:173` are step-for-step identical to the reference at
  `core/color/mod.rs:152`, including the 0.5 contrast pivot and the luma weights.
- **`max_pixel_value` mismatch.** Preview grades RGBA8 with 255 (`qrhi_undistort.rs` pins
  `RGBA8`), export grades RGBA16 with 65535 (`rendering/mod.rs:719`). Both correct; both
  normalise to 0–1 before grading and rescale after.
- **Preview LUT precision fallback.** `qrhi_undistort.cpp:137-165` has a
  RGBA32F → RGBA16F → RGBA8 ladder, and rung 2 silently refuses real LUT data. This machine is
  D3D11, which supports RGBA32F, so it sits on rung 0 at full precision. **Still worth confirming
  on the affected machine** — rung 2 presents as "no LUT in the preview at all", which is one of
  the two shapes the next step below is trying to distinguish. It logs when it drops to rung 1 or
  2, and rung 0 logs nothing, so absence of a message is weak evidence either way.
- **`remap_colorrange`.** This fit the symptom well — it compresses contrast by 0.859, exactly a
  "flatter preview" signature. But the preview shader does not contain it at all, and
  `FIX_COLOR_RANGE` is only ever set for prores/NV12 (`rendering/mod.rs:619`).

### The "different input values" theory — measured, and dead

The obvious theory was that the two paths grade *different pixels*: the preview grades an RGBA8
texture from **MDK**, the export grades RGB from **swscale**, and the clips are `color_range=pc`
with `color_space=unknown`, so a disagreement about range or transfer looked likely.

It was measured on the real frame rather than argued, and **it does not hold**. Every variant
fails on *direction* — each one makes the grade equal or **stronger**, never weaker:

| input the LUT is handed | grade magnitude, mean LSB16 | vs correct |
|---|---|---|
| correct full-range log (what the export gets) | 6763 | — |
| wrongly range-expanded (full read as limited) | 7217 | 1.07× |
| already de-logged (as if MDK tone-mapped it) | 9314 | 1.38× |
| 8-bit quantised log | 6763 | 1.00× |

So a range mismatch shifts grade strength by 7% *upward*, an already-converted input by 38%
*upward*, and 8-bit preview quantisation not at all. None of them produces a weaker preview.

Also checked and clean: the export preserves the source's `color_range=pc` tag, so this is not a
player misreading an untagged file; `lut_sample` in `undistort.frag:106` matches `sample_trilinear`
(`core/color/lut.rs:111`) operation for operation, including the `(g*n + r, b)` texel addressing;
and `uploadLut` (`qrhi_undistort.cpp:284`) writes .cube order verbatim in all three format rungs.

### Where this points instead

The export is a provably correct, single, full-strength LUT application. For the preview to look
*weaker* than that, the preview must be applying **less than the full LUT** — not grading
different pixels. That inverts the search.

The concrete mechanism to check first is the `lut_usable` guard in `undistort.frag:147`:

```glsl
bool lut_usable = lut_tex_size.x == lut_size * lut_size && lut_tex_size.y == lut_size;
if (bool(params.flags & 8192) && lut_size >= 2 && lut_strength > 0.0 && lut_usable) { … }
```

`lut_size` arrives inside `KernelParams` (`color_params_2.w`), baked when the frame transform was
computed; the *texture* is uploaded separately from an independent read of `stab.color`
(`qt_gpu/qrhi_undistort.rs`, the `lut_ptr` / `lut_size` / `lut_checksum` block). The two come from
different snapshots. The shader is written to skip the LUT for a frame when they disagree, which is
right for a one-frame transient — but if they disagree *persistently* (e.g. `color_params_2.w`
stuck at 0 because the compute-params snapshot never refreshed), the preview drops the LUT
permanently while still applying exposure, contrast, saturation and white balance. A user with
both a LUT and slider adjustments would see exactly a partial, weaker-looking grade.

### Resolution

Settled by rendering `C3147.MP4` with the grade the user actually had on screen (LUT at 100%,
every scalar neutral, target 6000 K = source 6000 K, so white balance a no-op), screen-capturing
the preview, and fitting the decoded file against it:

| decode of the export | fit vs preview | mean (prev / file) | std (prev / file) |
|---|---|---|---|
| honouring `color_range=pc` | `0.954 × preview + 0.022` | 0.4558 / 0.4565 | 0.2104 / 0.2112 |
| as limited-range | `1.108 × preview − 0.047`, pivot 0.433 | — | 0.2104 / 0.2456 |

Decoded correctly the file *is* the preview. Decoded as limited-range it reproduces the reported
difference, matching the `1.133 / 0.417` fit taken independently from the user's screenshots. The
colour maths, on both sides, was right the whole time.

### What to actually fix: output colour tagging

The export writes `color_range=pc` but leaves `color_space`, `color_transfer` and
`color_primaries` at `unknown`, copied from the source (`ffmpeg_video.rs:128-137`, `548-551`).
An undeclared matrix is what sends players to their defaults, and for 10-bit YUV many default to
limited range and ignore the range tag.

**Resolved: option 3 was chosen and is implemented** (see the note under the list).

The options as they stood:

1. **Declare the matrix, keep full range.** Write BT.709 primaries/transfer/matrix when the
   source leaves them unspecified. Most players assume BT.709 for HD/UHD anyway, so this mostly
   removes the ambiguity that makes them guess. Does not *guarantee* they honour `pc`.
2. **Encode limited range.** Convert full→limited on output. Virtually every player assumes
   limited for video, so this is the most interoperable, but it is a real behaviour change for
   every user and costs a little headroom.
3. **Tag from the LUT.** Tempting after baking a log→709 LUT, but Gyroflow cannot know what a
   given `.cube` outputs — a Rec.2020 LUT would be mistagged. Only sound if the user declares it.

**What was built.** Option 3, with the objection above answered by making it the user's
declaration rather than an inference. `OutputColorSpace` (`src/core/color/output_space.rs`) is a
new field on `ColorAdjustments`, exposed in the Color adjustments panel as an "Output color space"
dropdown offering *Same as source* (the default, i.e. the old behaviour), Rec. 709, sRGB,
DCI-P3 D65, Rec. 2020 PQ and Rec. 2020 HLG. `from_lut_name` supplies the initial guess from the
LUT's filename - `SL3SG3Ctos709.cube` lands on Rec. 709 - and the dropdown shows that guess before
the user exports, so a wrong guess is visible and one click from being corrected. The guess is only
seeded when the value is still *Same as source*, so it never overwrites a deliberate choice.

Two rules keep the tag honest:

- It is written only when the grade is actually baked (`bake_color`). Declaring a conversion on an
  export whose pixels were never converted would be worse than saying nothing, so the render side
  and the UI both gate on it.
- It writes tags only. Verified: the YUV sample hash of a render is byte-identical before and after
  tagging (`2B02EE18...`), while `ffprobe` goes from `unknown` on all three fields to
  `bt709/bt709/bt709` alongside the `color_range=pc` that was already being written.

The field is deliberately absent from `get_checksum`, so changing it cannot invalidate the preview -
it has no effect on any pixel to invalidate. It round-trips through the `.gyroflow` project file
under `"output_color_space"`.

**Then option 2 as well, because option 3 did not fix the symptom.** Declaring the matrix removed
a real ambiguity but left the actual mechanism untouched: `color_range=pc` was already being
written and is precisely the field players ignore for YUV. Measured on a fresh export of C3147
with the LUT baked - luma spans codes 31..1016 with 0.65% of pixels *below* 64, so the file
genuinely uses the full range, and a player assuming video range applies a gain of 1023/876 =
1.168 about a pivot at code 445. That matches the `1.133 / 0.417` fit from the user's screenshots
and the `1.108 / 0.433` fit from decoding the file as limited-range: three independent numbers,
one mechanism.

`OutputRange` (`src/core/color/output_range.rs`) is therefore a second field on
`ColorAdjustments`, with an "Output range" dropdown beside the colour-space one: *Same as source*
(default, the old behaviour), *Limited (video, 16-235)*, *Full (0-255)*. It is deliberately a
separate module from `output_space.rs` so that file's "only the tags change, never a pixel"
invariant stays literally true - this one *does* convert.

The conversion is free: swscale already applies range as part of the RGB->YUV matrix if the
destination frame declares it, which is the same mechanism
`tag_limited_range_for_prores_videotoolbox` relies on, so `set_output_range` is a one-line
`set_color_range` on the output frame. It runs *before* the prores workaround, so that
encoder-bug fix still wins over a user asking for Full.

Baking-only, and here that is a hard technical limit rather than a policy choice: without baking
the frame never becomes RGB, swscale never builds an RGB->YUV matrix, and the only range
machinery in play is the kernels' own `remap_colorrange` behind `FIX_COLOR_RANGE` - routing
through that would mean touching all five hand-maintained kernels.

Seeding ties the two together: a LUT whose name resolves to a *display* colour space is a
deliverable-grade LUT, so `load_lut` seeds range to Limited alongside the space. Verified in the
GUI - loading `SL3SG3Ctos709.cube` moves both dropdowns on its own, to Rec. 709 and
Limited (video, 16-235). Both stay overridable, and neither overwrites a deliberate choice.

Measured end to end on C3147:

| | tags | Y min | Y max | % below 64 |
|---|---|---|---|---|
| before | `pc` + `unknown`x3 | 31 | 1016 | 0.65 |
| space only | `pc` + `bt709`x3 | 31 | 1016 | 0.65 |
| space + limited range | `tv` + `bt709`x3 | 90 | 934 | 0.00 |

And decoded the way a player actually decodes each: the new limited-range file reads back as
`0.997x` the intended picture (offset -0.007), against `1.163x / -0.079` for the old one. The
disagreement is gone.

Option 1 remains available as a future default for the untagged-source case, and is not ruled out
by this.

---

## 2. Colour-stage placement divergence — now actionable, still unfixed

Documented in the design spec under "Known kernel divergence". Recorded here because its blocker
was removed today and that is easy to miss.

The CPU path grades the composited pixel once (`cpu_undistort.rs:641,655`). The wgpu and OpenCL
kernels grade inside `sample_input_at` (`wgpu_undistort.wgsl:370`, `opencl_undistort.cl:483`),
which is before compositing. Two visible consequences:

- **Background pixels** on the `r_limit`-clipped region of fisheye profiles are left ungraded by
  the GPU kernels but graded by the CPU path and the Qt preview.
- **The feather band** (`background_mode == 3`): the GPU kernels grade two samples then blend, the
  CPU blends then grades. The affine adjustments commute with the blend, so this is invisible for
  them — but `sample_trilinear` does not, so **with a LUT loaded the feather band renders
  differently on GPU than on CPU.**

This was parked because no test could verify a fix. **That is no longer true** — the cross-backend
equivalence harness now works (see §4). The correct target is the CPU order: grade the composited
pixel once, matching the reference `apply_color` and what the preview already does.

**Important:** the existing harness will *not* catch a regression here on its own. It runs an
identity transform with no background region and no feather band, so it exercises neither
consequence. Extending it with a case that hits the sentinel path and one with
`background_mode == 3` plus a LUT is part of this work, not a follow-up to it.

---

## 3. Verification owed

- **The GUI export path, end to end.** Today's fix `0b2f5a96` makes `get_cloned` carry the colour
  grade. Unit tests prove the clone, but nothing has exercised the actual Export button — and that
  is precisely the path that was broken. Load a clip, set a LUT, tick "apply to rendered video",
  export, confirm the output is graded.
- **The preview LUT lifetime crash** (`df6c7c64`). Item 13 of the checklist in the design spec.
  Needs two clips loaded through the preview by hand, repeatedly, plus two different LUT sizes
  back to back. Note it reproduces **with colour disabled**, so it is not a colour-only concern.

## 4. What was resolved today, so it is not re-investigated

- **The wgpu identity divergence is solved.** Cause was `draw_safe_area`, which is not behind a
  flag in the wgpu/OpenCL/SPIR-V kernels: it halves rgb outside `safe_area_rect` and halves it
  again within 5px of the edge. The test harness built `KernelParams` by hand and left the rect at
  `[0,0,0,0]`. Production sets it from the output size, so the darkening is inert in the app — this
  was a harness bug. Identity is now bit-exact; the colour stage measures 5.96e-8 against a 1/1024
  tolerance.
- **A trap worth remembering:** the sampling-map probes decode a source position *from channel
  values*, so a multiplicative scale on the values is indistinguishable from a scale on the sampled
  position. That is how the above was misread as "wgpu maps output x to a nonlinear curve" and
  chased as geometry. Rule out value scaling before re-deriving geometry.
- **Exports silently dropped the grade** (`0b2f5a96`). `StabilizationManager::get_cloned`
  (`core/lib.rs:1171`) did not clone `color`, so the Export button rendered with
  `ColorAdjustments::default()`. The preview uses the live manager, so it kept showing the grade,
  and no warning fired because they all sit downstream of `bake_color` already being true. Also
  fixed in `RenderQueue::add_file`.

## 5. Pre-existing defects found along the way — each worth its own issue

None of these are caused by the colour work; all were reproduced with colour disabled.

- **CPU-only rendering does not work.** `NO_OPENCL=1 NO_WGPU=1` fails with
  `No cached wgpu found for key: …` instead of falling back to the CPU pixel path. This means the
  CPU colour stage has never run through a real render, only unit tests.
- **`YUV422P10LE` is not a supported render output format**, so a stock Sony Cinema Line clip
  cannot be exported without overriding `pixel_format`.
- **The CLI trips a debug-only null-pointer assertion.** `RenderQueue` is a QObject constructed at
  `cli.rs:222`, but `QCoreApplication` is not created until `cli.rs:645`. Release builds are fine.

---

## Working notes for this repo

- **Build and run with `just`**, never bare cargo — `just build`, `just test`, `just run`. Bare
  `cargo build --release` fails on qmake and `cargo run` on OpenCV, because only `just` exports the
  needed PATH (`_scripts/common.just:70`). `cargo test` *inside* `src/core/` is fine.
- **Calling `target\release\gyroflow.exe` directly** needs three `ext/` dirs on PATH or it dies
  with `0xC0000135`: Qt `bin`, ffmpeg `bin`, and `ext\vcpkg\installed\x64-windows\bin` (OpenCV —
  the easy one to miss).
- **A positional file argument puts the CLI in headless mode**, so it never opens the GUI. Launch
  with no arguments to get the app.
- **`gyroflow.log` is truncated on every launch** (`%LOCALAPPDATA%\Gyroflow\`). Copy it aside
  before reproducing anything from an earlier session. Its timestamps are UTC while the file mtime
  is local. `settings.json` beside it is not truncated and is a useful secondary source.
- **Line endings are mixed, per file.** `src/core/color/mod.rs` is LF; `src/core/lib.rs`,
  `src/controller.rs`, `src/rendering/mod.rs` and the QML files are CRLF. Check with `file` before
  any scripted edit — assuming either one silently makes every replacement fail to match. Never
  `sed -i` (rewrites the whole file to LF), and note PowerShell `Set-Content` re-encodes: it added
  a BOM and mojibaked every non-ASCII character in `ColorAdjustments.qml`. Appending with `>>` can
  also concatenate onto a missing trailing newline. Python in binary mode preserves both.
- **Editing `src/qt_gpu/undistort.frag` does nothing until you recompile it.** The preview loads
  precompiled `.qsb` files from `src/qt_gpu/compiled/`, baked into a Qt resource. Run
  `bash src/qt_gpu/compiled/compile_shaders.sh` (needs `qsb.exe` from the Qt `bin` dir, path is
  hardcoded in the script) and rebuild; it regenerates 21 permutations of distortion model x
  digital lens. Nothing warns you - the preview simply keeps running the old shader, which reads
  as "the feature does not work on the preview path".
- **In PowerShell, never pipe a running process through `Select-Object -First N`** — it terminates
  the pipeline and kills the process. This aborted two renders before it was recognised.
- The GPU backend is selected by `NO_OPENCL` / `NO_WGPU` (`core/gpu/mod.rs:375`).
- Cross-backend tests need a GPU and deliberately fail rather than skip when none is present, so
  they are `#[ignore]`d: `cd src/core && cargo test cross_backend -- --ignored --nocapture`.
