# Colour / LUT — open threads

Handoff note, 2026-08-20. Branch `wgpu-identity-investigation`, 7 commits on top of `master`.
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
  D3D11, which supports RGBA32F, so it sits on rung 0 at full precision. Worth re-checking on a
  different GPU or backend, because rung 2 would present as "no LUT in preview at all".
- **`remap_colorrange`.** This fit the symptom well — it compresses contrast by 0.859, exactly a
  "flatter preview" signature. But the preview shader does not contain it at all, and
  `FIX_COLOR_RANGE` is only ever set for prores/NV12 (`rendering/mod.rs:619`).

### Leading hypothesis

The divergence is in **the input to the colour stage, not the colour stage**.

The preview grades an RGBA8 texture produced by **MDK**. The export grades RGB produced by
**swscale**. The test clips are `color_range=pc` (full range), `color_space=unknown`.

The export explicitly honours that flag — that is what
`convert_pixel_format_color_aware` (`rendering/ffmpeg_video_converter.rs:62`) exists to do. If MDK
sees `unknown` and assumes limited range, it expands 64–940 to full, so the *ungraded* preview is
already contrast-stretched, and a LUT whose job is adding contrast has visibly less left to do.
That reads exactly as "the grade is weaker in the preview".

This is unconfirmed. `MDKPlayer` is only forward-declared in our tree
(`qt_gpu/qrhi_undistort.cpp:18`); the implementation is in the MDK SDK under `ext/`. The preview
framebuffer also could not be captured headlessly.

### Next step

**The diagnostic that settles it, and it needs eyes on the app:** with the colour panel disabled
entirely, compare the ungraded preview against an ungraded export of the same frame.

- If they **already differ** (preview punchier, less flat) — the colour code is innocent and this
  is a decode-range bug in the preview path. Investigate what MDK is told about range and
  colourspace, and whether Gyroflow can pass the source's `color_range` through to it.
- If they **match** — the hypothesis is wrong. Move to the LUT sampling path: compare
  `lut_sample` in `undistort.frag` against `sample_trilinear` in `core/color/lut.rs`, and check the
  domain values in `color_params_3` and the 8-bit RGBA staging of the preview LUT upload.

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
- **The repo is CRLF.** Never `sed -i` — it rewrites the whole file to LF. Appending with `>>` can
  also concatenate onto a missing trailing newline.
- **In PowerShell, never pipe a running process through `Select-Object -First N`** — it terminates
  the pipeline and kills the process. This aborted two renders before it was recognised.
- The GPU backend is selected by `NO_OPENCL` / `NO_WGPU` (`core/gpu/mod.rs:375`).
- Cross-backend tests need a GPU and deliberately fail rather than skip when none is present, so
  they are `#[ignore]`d: `cd src/core && cargo test cross_backend -- --ignored --nocapture`.
