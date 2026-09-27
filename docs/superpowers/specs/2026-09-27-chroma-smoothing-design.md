# Chroma smoothing: a spatial pass after temporal denoising

Date: 2026-09-27 · Status: approved in conversation, awaiting spec review

## Problem

After gyro-guided temporal denoising, "quite large blotches" remain. Investigation on
`testfiles/C2520.MP4` (Sony ILCE-6700, 4K 25p, XAVC H.264 4:2:2 10-bit, S-Log3):

- The blotches are **chroma** noise, 8–16 px across at full resolution, elongated vertically
  (4:2:2 capture plus H.264 macroblock compression).
- They survive on **moving subjects**. Gyro alignment compensates camera rotation only, so a
  subject that moves relative to the camera disagrees with its neighbouring frames and the
  temporal rejection (correctly) refuses to average it. Static areas are already clean.

Measured on a flat patch of moving dark trousers, frame 8, 10-bit codes (std of U and V):

| | U | V |
|---|---|---|
| no denoise | 2.63 | 3.51 |
| temporal High (current) | 2.31 | 3.23 |

## Candidates evaluated (throwaway prototypes, Python/OpenCV)

| Method | U | V | Verdict |
|---|---|---|---|
| luma-guided filter r8 (He et al.) | 0.81 | 1.62 | blotches gone, colour edges intact; ~6 box-filter stages |
| **joint bilateral, luma range term, r8 step 2 (81 taps)** | **1.06** | **1.84** | visually indistinguishable from guided; single pass |
| joint bilateral r12 step 3 | 0.92 | 1.75 | slightly stronger |
| chroma-only bilateral (no luma guide) | 1.20 | 2.01 | blotches remain, red fringing at colour edges |
| spatial luma (bilateral / NLM) | — | — | plasticky, smears static texture; out of scope |
| motion-compensated temporal (DIS optical flow) | 1.38 | 2.07 | ghosts colour at edges, 3.4 s/frame CPU; out of scope |

Luma guidance is required: without it colour bleeds across edges. The single-pass joint
bilateral is chosen over the guided filter because it needs one GPU dispatch and no
intermediate buffers.

## Design

### Filter

For each chroma sample `p` at chroma-plane position `x`, with luma guide `G` at chroma resolution:

```
out(x) = p(x) + amount * (F(x) - p(x))
F(x)   = sum_d w(d) p(x+d) / sum_d w(d),   d in {-8, -6, ..., 8}^2   (81 taps)
w(d)   = exp(-|d|^2 / (2 * 4^2)  -  (G(x+d) - G(x))^2 / (2 * sigma_y^2))
```

- `sigma_y` = 0.015 of full scale (≈ 16 ten-bit codes, the prototype's value).
- `amount` = the user's "Colour smoothing" slider, 0..1. `amount == 0` is a bit-exact no-op.
- Taps outside the plane are clamped to the edge.

### Where it runs

A **second GPU pass**, run by the same backend instance right after the existing
`undistort_image` kernel, on its output, before readback. Temporal first, then spatial.

- **Planar / semi-planar YUV** (NV12, P010, YUV420P, NV16, P210, YUV444P, …): chroma planes only.
  The guide is the luma plane in the output frame, already written because `run_planes`
  (`rendering/mod.rs`) runs planes in order, luma first. It is uploaded like a reference frame.
  The kernel averages it down to the chroma plane's resolution from the size ratio (2×2 for
  4:2:0, 2×1 for 4:2:2, 1×1 for 4:4:4). Two-channel UV planes filter both channels with the
  same weights; single-channel U or V planes filter their one channel.
- **Packed RGB** (RGBA8/16, RGB8/16: the paused preview and colour-baked exports): the guide is
  each pixel's own Rec.709 luma. Only the chroma component `c = rgb − Y·(1,1,1)` is filtered,
  and the pixel keeps its own `Y`, so luma is never blurred. Alpha is untouched.
- **Everything else** (planar RGB planes such as GBRP float, packed YUV such as AYUV, alpha
  planes, half-float on OpenCL): skipped. Packed float RGB (RGBAf) is supported like any
  other packed RGB.

Plane classification reuses what the export dispatch already knows: the plane index and the
YUV index list `$yuvi` (empty = packed RGB), as the `denoise_rgb` work does.

The pass runs only when the output is a CPU buffer (`BufferSource::Cpu`), which is how both
the denoise preview (pixel path) and denoised exports hand frames over. Any other output
source skips the pass and warns once.

### Backends

- **OpenCL:** a second kernel `chroma_filter` in the existing program source
  (`opencl_undistort.cl`), a scratch output buffer and a guide buffer on `OclWrapper`, and the
  readback taken from the scratch buffer when the pass ran.
- **wgpu:** a second compute pipeline from its own shader, `wgpu_chroma_filter.wgsl` (kept out
  of `wgpu_undistort.wgsl`, which is specialised per pixel format through overrides), reading
  the output storage buffer and writing a scratch buffer that is read back.
- **Not covered:** the CPU fallback backend and the Qt RHI preview pipeline. Neither has
  temporal denoising today, and the denoise preview always goes through the pixel path.

### Parameters

A new, separate uniform block for the pass. `KernelParams` and its five mirrors (and the
compiled `.qsb` shaders) are **not** changed.

```
ChromaFilterParams {
  width, height, stride            // the plane being filtered (bytes for stride)
  guide_width, guide_height, guide_stride
  channels                          // 1, 2 (UV) or 3/4 (packed RGB)
  mode                              // 0 = planar chroma with luma guide, 1 = packed RGB
  amount, sigma_y, max_pixel_value
}
```

`Buffers` gains an optional guide buffer description for the planar case.

### Settings and UI

- `DenoiseSettings::chroma_smoothing: f64`, 0..1, default **0.5**, clamped in `set_param`
  like the strengths, included in `get_checksum`, saved and loaded as
  `denoise.chroma_smoothing` in projects and presets.
- CLI: `--denoise-chroma-smoothing <0..1>`, following `--denoise-chroma`.
- Denoise panel: a "Colour smoothing" slider (0–100%) with a tooltip, gated by the existing
  Enable checkbox like the other controls. Fast/High remains the temporal pass only.
- The pass runs whenever denoising is enabled and `chroma_smoothing > 0`; it does not depend
  on the temporal pass having any references.

Changing the default means projects with denoising already on look slightly different when
reopened; setting the slider to 0 reproduces the previous output exactly.

## Error handling

Skipping the pass is the only failure mode: never a corrupted frame. A missing or mismatched
guide, a failed scratch allocation, or a kernel error leaves the temporal result in place and
logs one warning (once per backend instance, not per frame).

## Testing

1. **Rust reference** `denoise::chroma_filter_reference` (planar and RGB modes), unit tested:
   `amount == 0` is bit-exact; flat chroma noise is reduced; a colour edge coinciding with a
   luma edge does not bleed; RGB mode leaves each pixel's luma unchanged; 4:2:0 and 4:2:2
   guide downsampling is correct.
2. **GPU tests** (`denoise/tests_gpu.rs` style, `#[ignore]`d, wgpu and OpenCL): each backend
   matches the reference within 1/1024 of full scale; the two backends agree pixel for pixel;
   both modes covered.
3. **Real footage:** renders of `C2520.MP4` with the feature on. On the trousers patch the
   target is V ≤ 2.0 (from 3.23) with U reduced proportionally, and a visual check of the shoe
   colour edge matching the prototype. Export time measured and reported.
4. Existing core (`cargo test` in `src/core`) and app (`just test`) suites stay green.

## Out of scope

Spatial luma denoising, motion-compensated temporal denoising, a CPU implementation of the
pass, and exposing radius or sigma as user controls.
