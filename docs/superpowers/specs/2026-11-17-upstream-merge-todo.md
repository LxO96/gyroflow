# TODO: merge upstream gyroflow into Dev-sony

Status (2026-11-21): merged on `merge/upstream-v1.7` (upstream `45ba865f`, v1.7.1). Conflicts resolved, core tests
and GPU tests (OpenCL + wgpu) green, app builds. Real-footage testing and the merge into `Dev-sony` are still open.

- Upstream: `upstream` remote = https://github.com/gyroflow/gyroflow, `upstream/master` at `c69173ad` (v1.7.0 + fixes)
- Merge base: `d918ab35`
- Incoming: 97 commits, 273 files changed (+70.9k / -26.0k)
- Biggest upstream changes: lens-profile math rework, output projections, optical-only stabilization,
  GPU KLT optical analysis, gmflow (AI) optical flow, "Optimal path" smoothing, CRM playback,
  AV1 export, ffmpeg 9.1, parking_lot 0.13, burn/cubecl updates, BRAW/R3D SDK updates

## Plan

- [x] Branch off: `git switch -c merge/upstream-v1.7 Dev-sony`, then `git merge upstream/master`
- [x] Resolve the GPU kernel interface (the hard part, see below)
- [x] Resolve the remaining conflicts
- [x] Rebuild all `src/qt_gpu/compiled/*.qsb` from the merged `src/qt_gpu/undistort.frag` (don't pick a side)
- [x] Build with `just` and fix compile errors
- [ ] Test on real footage, on both OpenCL and wgpu, plus the CPU fallback:
  - [ ] plain stabilization (also the new output projections and optical-only mode)
  - [ ] temporal denoise, including trim seams
  - [ ] chroma smoothing
  - [ ] colour / LUT
  - [ ] film simulation (preview + export)
  - [ ] render queue and export (Export panel, pixel format memory, export folder)
- [ ] Merge `merge/upstream-v1.7` into `Dev-sony` and push

## The hard part: GPU kernel interface

Upstream changed the same structures that denoise / colour / LUT / film depend on.
The layout must stay byte-identical across Rust, SPIR-V types, OpenCL, WGSL and GLSL.

- [x] **Matrix stride 14 -> 16 floats.** `FrameTransform::matrices` is now `Vec<[f32; 16]>` upstream.
  The fork's `reference_matrices: Vec<Vec<[f32; 14]>>` and the per-frame block indexing
  (`f * matrix_count * 14`) must move to 16 in every backend:
  `opencl.rs` + `opencl_undistort.cl`, `wgpu.rs` + `wgpu_undistort.wgsl`, `undistort.frag`, `cpu_undistort.rs`.
- [x] **KernelParams slot clash.** Upstream put `output_projection: i32` + `reserved2` in the slot the fork
  uses for `reference_count: f32` + `denoise_sigma: f32`. Keep upstream's fields there and move the fork's
  two fields into a new 16-byte slot (e.g. into `denoise_rgb.yz`, which are unused, or a new vec4).
  Update `stabilization/mod.rs`, `stabilize_spirv/src/types.rs` and every kernel's struct definition.
- [x] `stabilization/mod.rs` CPU path: keep both upstream's `NoBackendForBuffers` early return and the
  fork's film skip notice + coeffs/LUT buffer.
- [x] `frame_transform.rs` (5 conflicts), `core/lib.rs` (3): merge upstream's `FocalSource` / projection
  work with the fork's reference-frame and film fields.

## Other conflicts

- [x] `Cargo.toml`: the fork vendors a patched `qml-video-rs` (path `src/qml-video-rs`, from rev `855130d`,
  `TextureResult` callback patch). Upstream moved to rev `baca6be`. Re-apply the patch on top of `baca6be`
  and keep the path dependency. Then regenerate `Cargo.lock`.
- [x] `src/rendering/`: `mod.rs` (2), `render_queue.rs` (3), `ffmpeg_processor.rs` (1), `ffmpeg_video.rs` (1)
- [x] QML: `VideoArea.qml` (1), `menu/Advanced.qml` (2), `menu/Export.qml` (1)

## Alternative if the full merge is too much right now

Cherry-pick standalone upstream fixes onto `Dev-sony` and leave the big rework for later, e.g.:
`01b59ec4` (exported audio padding), `69c46052` (rendering to NAS), `b2a359d7` (accelerometer bias),
`4cd232d3` / `78cfb312` (qml-video-rs / preview texture races), `5a75ac85` (VAAPI HEVC/H.264).

## Decisions taken during the merge (check these on real footage)

- **wgpu half pixel.** Upstream 58c7e429 subtracts 0.5 px in `undistort_fragment`. On this fork that put wgpu half a
  pixel off the CPU/OpenCL (11 GPU tests failed; `Dev-sony` passes them), so it was left out. Compare wgpu vs OpenCL
  preview/export on a sharp frame.
- **Stabilization-off preview** rebuilt for upstream's ray kernel (`lens_correction_amount = 0`, the lens model's own
  round trip). Check the toggle shows the untouched frame with fisheye and Sony profiles.
- **COEFFS grew** (two new interpolation kernels): `LUT_DATA_OFFSET` is now 496 in every backend.
- **CUDA host-copy path** (upstream, for GPUs wgpu can't share CUDA memory with) drops denoise references.
- **qml-video-rs** vendored copy rebased onto `baca6be` (clean 3-way merge).
- New build deps fetched by `just install-deps`: ffmpeg 9.1, `ext/mdk-sdk` + CRM plugin.
