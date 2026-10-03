# Film simulation (grain, tone, halation) — design

Date: 2026-10-04. Status: approved in conversation, awaiting spec review.

Reference: `2026-10-04-film-handover.md` (copied verbatim from Photo360
`docs/film-handover.md`). It is the authority for every formula, constant and
stock table below; this spec only says how it lands in Gyroflow. Where this
spec and the handover disagree on maths, the handover wins.

## Goal

Re-develop the graded output as if it were shot on film: per-stock tone and
colour, silver-halide grain that moves every frame, and halation. Fast on a
good GPU: target 1–3 ms per 4K frame for the whole film chain.

## Decisions taken

1. **Graded output only.** The film pass runs only when the colour grade is
   baked into the render (`bake_color` in `src/rendering/mod.rs`), so its input
   is always packed RGB(A) display values (u8 or u16; colour baking already
   converts every source with `packed_color_format`). It never runs for PQ/HLG
   output colour spaces, for hardware frames (which already skip baking), or on
   log footage without a grade. When the gate fails the Film panel says why.
2. **Stock plus toggles.** Controls: Stock (Off + 8 stocks), Format (Super 8,
   Super 16, 35 mm, 120), toggles *Tone & colour*, *Grain*, *Halation*, Grain
   amount 0–200 % (default 100 %), Halation 0–100 (default 50). With *Tone &
   colour* off, `u` is the display value itself (handover: "If your project has
   no film curve…") and no sensitivity/crosstalk is applied.
3. **Approach: separate film passes after chroma smoothing, noise fused into
   blur X.** Fusing into the undistort kernel is impossible (the blur needs the
   finished neighbours); a repeating noise tile breaks the statistics.
4. **Both GPU backends** (OpenCL and wgpu), same structure as the chroma pass.

## Pipeline

```
main kernel (undistort → temporal denoise → grade/LUT)
  → chroma smoothing (if active)
  → film chain (if active)          ← new
  → readback → encoder
```

Last, so denoising never removes grain.

## CPU side: `src/core/film/`

Line-for-line Rust port of the handover (f64 maths):

- `stocks.rs`: the 8 stocks (`id`, name, `bw`, sensitivity, layers, grain,
  crosstalk, halation), ids exactly as the handover.
- `curve.rs`: `develop`, `fit_curve` (cached per stock).
- `grain.rs`: `inverse_normal`, `i0e`, `lens_area`, `GrainModel`
  (`filtered_variance`, `lag_one_correlation`, `for_render_scale`),
  `gaussian_kernel` (max radius 31), `kernel_lag_one`, `calibrate_noise_sigma`,
  `grain_setup` → `{ sigma, radius, norm, std: [f32; 201] }`, cached by
  `(stock, q(scale))` with `q(v) = 2^(round(8·log2 v)/8)`.
- `noise.rs`: `pcg`, `hash4`, `gaussian` (wrapping u32 arithmetic), FNV-1a.
- `halation.rs`: `halation_radius(D, gate)`, `halo_factor(R)`.
- `mod.rs`: `FilmSettings` (serde, project file), `FilmFormat` with gate short
  sides 4.01 / 7.41 / 24 / 56 mm, and
  `film_params(settings, out_w, out_h, render_scale, frame_index, seed) -> Option<FilmParams>`
  (`None` = chain skipped). `FilmParams` is `#[repr(C)]`, holds sensitivity,
  crosstalk, per-layer `logE0`, `k = 4·gamma/range`, `uMin`, `range`, fit
  `(b, s, uMinA)`, layer seeds, grain `sigma`, `radius`, `amount·norm`,
  halation `R`, `f`, `gain·slider`, tint, flags (TONE, GRAIN, HALATION, BW),
  plus the 201-entry std table. A pure-Rust reference renderer of the whole
  chain (`reference.rs`) is used by tests only.

### Scale

`D` = output short side in output px. Export: `scale = (D / 3060)·(4.01 /
gateShortMm)`. Preview renders at reduced resolution `p = preview_short /
output_short ≤ 1`; per the handover's zoom-out rule its grain model is
`model.for_render_scale(q(scale)).for_render_scale(q(p))` (applied on top of
the export scale, not multiplied into it), and halation uses `R·p`. `p = 1`
(export, or Full preview) skips the second step. Noise is hashed on the **preview** pixel
lattice (no export-lattice trick: Gyroflow's viewer zoom magnifies the preview
texture, it never re-renders). To see export-exact grain the user sets Preview
resolution to Full; the viewer zoom then shows it 1:1 or magnified.

### Seed

`seed = FNV-1a(video file name)`, `frameSeed = hash4(seed, frame_index, 0, 0x51)`,
`layerSeed[l] = hash4(frameSeed, 0, l, 0x51)`. Grain moves every frame; the
same frame always renders identically, in preview and export.

## GPU side: the film chain

Each pass is skipped when its feature is off; when `film_params` is `None` the
chain is not dispatched at all and output is byte-identical to today.

| Pass | When | Work | Output |
|---|---|---|---|
| H1 halo source | halation on | f×f box average of `smoothstep(0.75,1,Y)·Y`, `Y` = Rec.709 luma of the sRGB-decoded pixel | R32F ⌈W/f⌉×⌈H/f⌉ |
| H2, H3 halo blur | halation on | separable Gaussian σ = R/f, clamp-to-edge, radius ⌈3σ⌉ | same size |
| G1 noise + blur X | grain on | per layer: Σₖ wₖ·`gaussian(hash4(clamp(x+k), y, layerSeed[l], 0x3C6EF372))` (one layer for B&W) | RGBA32F W×H |
| G2 apply | chain active | blur Y of G1 (clamp-to-edge), then: sRGB decode → `+ halation·gain·halo·tint` (halo bilinear at `(x+0.5)/f`) → if TONE: sensitivity → curve → fit, else sRGB encode → `u + amount·norm·std(u)·n`, clamp → if TONE: crosstalk → clamp → write in frame format, alpha untouched | frame |

- Blur weights computed in-shader exactly as `gaussian_kernel` (normalised by
  their sum).
- `std(u)` read at `int(clamp(u,0,1)·200 + 0.5)`.
- Intermediates are allocated once per resolution and reused (G1 at 4K:
  133 MB).
- `FilmParams` (with the std table) is one small uniform/constant buffer per
  frame.
- The GPU wrapper cache key (`Buffers::get_checksum`) includes the film
  presence and buffer sizes, so toggling film never reuses a stale wrapper.

### OpenCL (`src/core/gpu/opencl_film.cl`)

A separate program with kernels `film_halo_source`, `film_halo_blur`,
`film_noise_blur_x`, `film_apply`, built next to the chroma program inside a
fallible closure: a build failure logs a warning and sets `film = None`; the
backend stays alive (test hook `FAIL_FILM_BUILD`, like `FAIL_CHROMA_BUILD`).
Runs after `run_chroma`; its output buffer becomes the readback source.

### wgpu (`src/core/gpu/wgpu_film.wgsl`)

Compute pipelines for the four passes; G2 writes a storage buffer/texture that
is copied to staging in place of the chroma/main output. Same fallible
construction and logging as OpenCL. Like the chroma pass, render path with CPU
output only.

### No GPU

CPU-only processing skips the film chain with a warning.

## Settings, UI, CLI

- `FilmSettings { stock: Option<String>, format, tone, grain, halation,
  grain_amount: f64, halation_amount: f64 }` in the project file under
  `"film"`; missing section → off.
- `src/ui/menu/Film.qml`: new collapsible section next to Denoise, with the
  controls above and a gate message line ("Film needs a baked colour grade",
  "Not available for HDR (PQ/HLG) output").
- CLI: `--film-stock <id|off>`, `--film-format`, `--film-grain`,
  `--film-halation`, `--film-no-tone`.
- When the chain is skipped because the GPU program failed, the log warns and
  the export dialog shows it once.

## Testing

CPU (default `cargo test`):

- `fit_curve` b, s and black/mid/white RGB for all 8 stocks vs the handover's
  reference table (±0.001).
- `pcg`, `hash4`, `gaussian` vs values produced by running Photo360's
  `film.js` in node (script kept in `_scripts/tests/`).
- `grain_model` values vs `film.js` for two stocks at two scales.
- Flat grey frame through `reference.rs`: measured std ≈ `amount·std(u)`
  (±5 %), lag-1 ≈ `lag_one_correlation()` at scale 1 and preview 0.25.
- Frame N ≠ frame N+1; frame N twice identical; B&W stays neutral.
- Stock Off → `film_params` is `None`; grain 0 / halation 0 skip their passes
  and equal the tone-only result.
- Halation: zero below 0.75 luma; reddens next to a clipped highlight.
- Gate: off without baked grade, off for PQ/HLG, off for hardware frames.
- Project save/load round trip; old project loads with film off; CLI parse.
- Cache checksum changes with film on/off.

GPU (`-- --ignored`, both backends):

- Each pass and the whole chain vs `reference.rs`: within 1/1023 (16-bit) and
  1/255 (8-bit); OpenCL vs wgpu agree.
- Forced film build failure: backend survives, film skipped.

Real footage: graded 4K export with Golden 200 / Super 16, report flat-area
grain std vs model and export time with vs without film.

## Out of scope

Keyframing film parameters; HDR (PQ/HLG) film; export-lattice preview grain;
custom stocks; CPU fallback renderer.
