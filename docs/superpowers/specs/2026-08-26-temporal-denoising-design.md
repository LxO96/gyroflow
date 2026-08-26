# Temporal denoising — design

**Date:** 2026-08-26
**Status:** Awaiting review

## Summary

Add gyro-guided temporal denoising to Gyroflow. For each output pixel, the undistort kernel gathers not just from the current source frame but from `2k+1` source frames, each sampled at the position that pixel lands in *that* frame, derived from the rotation Gyroflow already computed. Samples that disagree with the centre are rejected by weight, so parallax and moving subjects do not ghost.

The distinguishing idea: a motion-compensated temporal denoiser normally spends most of its budget estimating motion. Gyroflow already knows the camera motion exactly. That estimate is free here, and no other NLE-side denoiser has it.

## Why not FFmpeg

The bundled build (`ext/ffmpeg-8.1-windows-desktop-vs2026-gpl-lite/config.txt`) configures `--disable-filters` with a narrow allow-list — `drawtext`, `null`, `format`, `crop`, `eq*`, `scale*`, `yadif*`, `overlay` and hardware variants. Every temporal denoiser (`hqdn3d`, `atadenoise`, `nlmeans`, `vaguedenoiser`, `fftdnoiz`, `bm3d`, `mcdeint`) is compiled out. Enabling any of them means rebuilding FFmpeg for all five target platforms, and those binaries come from upstream avbuild artifacts rather than from this repo.

Two further obstacles: `src/rendering/` never builds a filtergraph for video (only the TODO at `mod.rs:1303`), and hardware frames (`Pixel::D3D11`, `VIDEOTOOLBOX`, CUDA via `zero_copy.rs`) would have to be round-tripped through system memory to feed one.

Alternatives considered and rejected: OpenCV `photo::fastNlMeansDenoisingColoredMulti` (CPU-only, seconds per frame at 4K, cannot serve a preview tier, ignores the gyro data); VapourSynth mvtools / KNLMeansCL (external runtime, not embeddable); BM3D/VBM4D (no practical embeddable implementation).

## Requirements

| # | Requirement |
|---|---|
| R1 | Temporal denoise using `2k+1` source frames, motion-compensated from gyro data |
| R2 | Per-pixel rejection so parallax and moving subjects do not ghost |
| R3 | Two quality tiers behind one control, trading render speed against noise reduction |
| R4 | Works on the wgpu and OpenCL backends |
| R5 | Preview shows the true denoised result when paused, matching the export exactly |
| R6 | Settings persist in the `.gyroflow` project file and are reachable from the CLI |
| R7 | Disabled (`k = 0`) costs nothing measurable |

### Non-goals

- CPU fallback. Where no GPU backend is available, denoise degrades to pass-through.
- NLE plugins (`src/external_sdk`). Hosts deliver one frame at a time with no seek, so temporal work there needs its own design.
- Keyframable denoise strength.
- Cross-channel (luma-guided chroma) rejection.

## Architecture

### Frame supply — export

`ffmpeg_video.rs` runs decode → callback → encode as one straight line per frame (`:258`, `:356`), with no queue between process and encode. A symmetric window therefore requires delaying output by `k` frames; frame `N+1` does not exist when `N` arrives.

The lever already exists. `mod.rs:652` sets `rate_control.repeat_times = 0` to suppress encoding of a frame, for frame-rate conversion. So:

- A new `TemporalWindow` holds a ring of `2k+1` decoded source frames.
- Each callback pushes the incoming frame, then emits the **centre** frame.
- While priming (the first `k` calls) it sets `repeat_times = 0` and emits nothing.
- `rate_control.out_timestamp_us` is set to the centre frame's timestamp, not the incoming frame's.

Frames at both ends of the clip get a naturally shrinking window, which is standard for temporal filters and needs no special handling.

**The one invasive change:** the decode loop breaks at `ffmpeg_video.rs:470` and returns with no post-loop callback, so the final `k` frames would be silently dropped. A drain hook must be added after the loop.

**Memory:** `2k+1` full-res source frames, resident in host memory and uploaded to the GPU. 4K 10-bit 4:2:2 is roughly 35 MB/frame, so `k = 3` is about 250 MB; 8K approaches 1 GB. `k` is capped by resolution rather than allowed to exhaust memory.

**Progress reporting:** with output delayed by `k`, emitted frame count no longer matches decoded frame count. The progress callback at `mod.rs:899` must count emitted frames, or it will run `k` behind and finish short of 100%.

### Frame supply — preview

`controller.rs:1150` receives a single frame from the mdk player, which owns decoding and seeks arbitrarily. No forward window is available during playback.

Therefore: **denoise renders in preview only when paused.** During playback the preview shows the undistorted but undenoised image. On pause, a second decoder seeks and fetches the true ±k window, so the still frame matches the export exactly. Opening a second decoder on the same file is established practice — `controller.rs:524` (autosync) and `render_queue.rs:1253` (thumbnails).

This was chosen over causal-only preview because a slider is judged on a still frame, and a preview that silently disagrees with the export is worse than one that visibly waits.

### The denoise pass

Fused into the existing undistort kernel rather than run as a separate pre-pass.

`undistort_coord()` (`wgpu_undistort.wgsl:561`) already maps an output pixel back to a source coordinate using rotation matrices held in the `matrices` array — an array that is already variable-length, indexed per scanline for rolling shutter (`matrix_count`, `stabilization/mod.rs:129`).

Extend `matrices` to hold `2k+1` sets, one per reference frame. For each output pixel, run the same mapping once per reference frame. Each result is exactly where that output pixel lands in that frame's source image — motion-compensated per pixel, rolling shutter included, using maths that already exists and is tested.

Rejected alternative: a separate pre-pass warping neighbours into the centre frame's source space. It would denoise strictly before resampling, but requires a new source→source homography, an extra full-res intermediate buffer, and a new mapping to get wrong.

**Accepted trade-off:** the fused form denoises *during* resampling rather than strictly before it. Each reference sample is still an independent interpolation of an independent noise realisation, so variance still falls by roughly `√(2k+1)`, but the interpolator does slightly correlate noise before averaging.

**Sampling cost mitigation:** reference frames are sampled bilinearly; only the centre frame uses the configured interpolation (EWA/bicubic). Without this, `k = 3` multiplies the expensive sampler by seven.

**Combination rule.** Per pixel, take the centre sample `s_c` as reference. For each neighbour `j`, compute `d = |s_j − s_c|` and weight `w_j = exp(−(d / σ)²)`, where `σ` is the strength knob; `w_c = 1`. Normalise by `Σw`. Samples that disagree — parallax, moving subjects, occlusion — fall out of the average automatically. This rejection is load-bearing, not a refinement: gyro rotation aligns frames for *rotational* camera motion only, and translation (walking, a moving drone) creates depth-dependent parallax that no rotation can compensate.

**Binding.** Binding 5 is currently one texture *or* one storage buffer (`wgpu_undistort.wgsl:63`, `:66`). References need `texture_2d_array` on the texture path and one concatenated buffer with per-frame offsets on the storage path. The same shape is needed in `opencl_undistort.cl`. This is the bulk of the plumbing, and it is paid twice, once per backend.

**Per-plane processing.** `KernelParams::plane_index` (`stabilization/mod.rs:160`) means luma and chroma go through separate kernel invocations. Rejection is therefore decided one channel at a time, which suits separate luma and chroma strength controls. Cross-channel rejection would require binding the luma plane into the chroma pass and is out of scope.

### Settings, UI, persistence

Mirrors `ColorAdjustments`, which is recent and proven.

- New `src/core/denoise/mod.rs` holding `DenoiseSettings { enabled, quality, luma_strength, chroma_strength }`, hung off `Stabilizer` as `Arc<RwLock<_>>` following `lib.rs:87`. `quality` alone determines `k`; there is no independent radius field, so the two cannot disagree. `luma_strength` and `chroma_strength` are the `σ` of the combination rule, selected by `plane_index`.
- A `set_param` entry for QML, with ranges clamped core-side per `lib.rs:1798`.
- **Included in `get_checksum`.** The colour code documents the trap: fields that never touch preview pixels (`output_color_space`, `output_range`) are deliberately excluded, and everything that does change them must be in. Omitting denoise would leave a stale preview after every slider move.
- Serialised as a `"denoise"` sibling of `"color"` at `lib.rs:1401` and `:1792`, with the same preset-merge semantics.
- New `src/ui/menu/Denoise.qml` plus a `qmldir` entry, modelled on `ColorAdjustments.qml`.
- Flags in `src/cli.rs`, since CLI renders bypass the UI.

The two tiers collapse to one **Quality** control mapping to `k`: *Fast* = `k=1`, *High* = `k=3`. One algorithm, one knob — not two implementations.

The tiers are named *Fast* and *High*, deliberately not *Preview* and *Export*. They are quality levels, not contexts: the paused preview always renders at whatever quality is selected, which is what makes R5 hold. A tier named "Preview" would imply the preview ignores the setting, and it does not.

`KernelParams` is duplicated across five files, as the comment at `stabilization/mod.rs:118` states: the Rust struct, `opencl_undistort.cl`, `wgpu_undistort.wgsl`, `qt_gpu/undistort.frag` and `stabilize_spirv/src/types.rs`. New parameters cost five byte-compatible edits.

## Testing

- Pure-Rust unit tests for the weight and rejection function. It is a scalar function; no GPU required.
- Ring-buffer and drain logic tested headlessly: feed a timestamp sequence, assert emitted timestamps are centre-shifted and that drain emits exactly the final `k`.
- Serialization round-trip and preset merge, mirroring the tests at `lib.rs:2488`.
- End-to-end on a synthetic clip with known Gaussian noise and known gyro: assert variance falls by roughly `√(2k+1)` in static regions, and that a moving-subject region stays within a ghosting bound.

The end-to-end tests assert against real wgpu output and are marked `#[ignore]`, so CI skips them and they are run by hand.

**Accepted risk:** consequently the shaders get no automated check, and divergence between the wgpu and OpenCL kernels will not be caught automatically. This was chosen over maintaining a scalar reference implementation as a test oracle.

## Open questions

None outstanding. The resolution cap on `k` and the exact `σ` ranges are calibration work for implementation, not design decisions.
