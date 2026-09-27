// Copyright © 2026 Gyroflow

//! End-to-end GPU tests for the temporal gather.
//!
//! These run a real frame plus real reference frames through a real backend and read the
//! pixels back, which is the only thing that checks the two gather kernels
//! (`wgpu_undistort.wgsl::gather_denoised` and `opencl_undistort.cl::gather_denoised`)
//! against each other and against what the feature claims to do.
//!
//! Run them:
//! ```text
//! cd src/core && cargo test -- --ignored denoise_gpu                     # wgpu
//! cd src/core && cargo test --features use-opencl -- --ignored denoise_gpu  # + OpenCL
//! ```
//! `just test` runs *no* core tests at all — `gyroflow-core` is a path dependency, not a
//! workspace member — so bare `cargo test` from `src/core` is the only thing that works
//! here (ledger Ruling 4).
//!
//! # How this is built
//!
//! The fixture is the one `stabilization::cross_backend_color_tests` established and
//! proved: an exact identity transform (k = 0 distortion, f = [1,1], c = [0,0],
//! `lens_correction_amount` 1.0, 1:1 rects, identity matrix, a `safe_area_rect` that
//! actually covers the frame) at `RGBAf`. **`identity_geometry_is_bit_exact_on_every_backend`
//! in that module is the control for everything here.** If it ever fails, the geometry is
//! resampling and every number below is measuring that rather than the gather; stop and fix
//! it first.
//!
//! `denoise_gpu_k_zero_is_a_bit_exact_no_op` re-proves the identity per backend inside this
//! module, so the OpenCL path — which that older module never exercised — is pinned here too.
//!
//! Same discipline as that module: a backend that is genuinely absent is *reported and
//! skipped*, anything else is a hard failure, and every test ends by asserting that at
//! least one backend actually ran.

use crate::gpu::{ Buffers, BufferDescription, BufferSource };
use crate::stabilization::{ FrameTransform, KernelParams, RGBAf, PixelType };
use crate::stabilization::distortion_models::DistortionModel;

// wgpu's `ParamCheck` refuses width < 16, height < 4 or stride < 16 (`gpu/wgpu.rs:169`),
// and OpenCL refuses height < 4 (`gpu/opencl.rs:181`). 64x64 clears both with room to
// spare and still gives 4096 pixels to measure a variance over, at 256 KB per frame.
const W: i32 = 64;
const H: i32 = 64;
const N: usize = (W * H) as usize;
const BPP: i32 = 16; // RGBAf: 4 x f32
const STRIDE: i32 = W * BPP;

/// Reference radius under test: `DenoiseQuality::High`, the largest window either backend
/// allocates for (`gpu::MAX_GATHER_FRAMES`).
const K: usize = 3;
/// Frames in the gather, centre included. This is what `reference_count` becomes.
const GATHER_FRAMES: usize = 2 * K + 1;

/// Deterministic pseudo-random noise for pixel `i` of frame `f`, **uniform on
/// `[-amplitude, amplitude]` — not Gaussian**. (The plan's sketch called its xorshift
/// "Box-Muller" while doing no such thing; nothing here needs a normal distribution, so
/// the comment is corrected rather than the code.) A uniform variate on `[-a, a]` has
/// variance `a^2 / 3`, which is what the variance test's "before" figure should land near.
///
/// splitmix64 over `(frame, index)` rather than a running xorshift state: it is
/// order-independent, so a test can ask for one pixel of one frame without generating the
/// whole clip, and it decorrelates frames far better than one xorshift round would — which
/// matters, because the variance claim below rests on the noise being independent across
/// frames.
fn noise(frame: usize, i: usize, amplitude: f32) -> f32 {
    let mut z = (((frame as u64) << 32) | i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Top 24 bits -> [0, 1), then to [-1, 1).
    let u = (z >> 40) as f32 / (1u32 << 24) as f32;
    (u * 2.0 - 1.0) * amplitude
}

/// The noiseless scene: a horizontal ramp in red, flat green/blue, opaque.
///
/// Only red carries noise and only red carries the subject, so the rejection distance
/// (`dot` over the lanes `pix_element_count` selects — all four here) is exactly the red
/// difference. That keeps every number below one-dimensional and directly interpretable.
/// Values stay well inside 0..1 so nothing is decided by a clamp.
fn base_pixel(i: usize) -> [f32; 4] {
    let x = (i % W as usize) as f32 / (W - 1) as f32;
    [0.30 + 0.30 * x, 0.5, 0.25, 1.0]
}

fn to_bytes(pixels: &[[f32; 4]]) -> Vec<u8> {
    let mut b = vec![0u8; (STRIDE * H) as usize];
    for (i, p) in pixels.iter().enumerate() {
        b[i * 16..i * 16 + 16].copy_from_slice(bytemuck::bytes_of(p));
    }
    b
}

fn read_pixels(buf: &[u8]) -> Vec<[f32; 4]> {
    (0..N).map(|i| *bytemuck::from_bytes::<[f32; 4]>(&buf[i * 16..i * 16 + 16])).collect()
}

/// A clip: the centre frame plus its temporal neighbours, already laid out as bytes.
struct Scene {
    centre: Vec<[f32; 4]>,
    references: Vec<Vec<[f32; 4]>>,
}

impl Scene {
    /// `GATHER_FRAMES` frames of the same static scene, each with its own independent
    /// noise. Frame `K` is the centre; the rest are the references, in temporal order.
    fn noisy_static(amplitude: f32) -> Self {
        let frame = |f: usize| -> Vec<[f32; 4]> {
            (0..N).map(|i| {
                let mut p = base_pixel(i);
                p[0] += noise(f, i, amplitude);
                p
            }).collect()
        };
        Scene {
            centre: frame(K),
            references: (0..GATHER_FRAMES).filter(|&f| f != K).map(frame).collect(),
        }
    }
}

/// The identity fixture, extended with the gather.
///
/// - `reference_matrices` gets one identity block per reference, each the same length as
///   `matrices`; both backends index block `f` at `f * matrix_count * 14` floats, so a
///   missing or short block reads another frame's matrix.
/// - `denoise_sigma` is set explicitly. Nothing derives it (ledger Ruling 3), and at 0.0
///   `inv_2sigma_sq` is an infinity.
/// - `reference_count` is deliberately *not* set: both backends compute it themselves from
///   `buffers.references.len()`, and overwriting it here would test the fixture, not them.
fn identity_transform(references: usize, sigma: f32) -> FrameTransform {
    let mut p = KernelParams::default();
    p.width = W; p.height = H; p.stride = STRIDE;
    p.output_width = W; p.output_height = H; p.output_stride = STRIDE;
    p.matrix_count = 1;
    p.interpolation = 2;
    p.bytes_per_pixel = BPP;
    p.pix_element_count = 4;
    p.f = [1.0, 1.0];
    p.c = [0.0, 0.0];
    p.fov = 1.0;
    p.lens_correction_amount = 1.0;
    p.source_rect = [0, 0, W, H];
    p.output_rect = [0, 0, W, H];
    // Not neutral at its default [0,0,0,0]: `draw_safe_area` runs unconditionally and would
    // halve rgb on every pixel. See the long note in `cross_backend_color_tests`.
    p.safe_area_rect = [0.0, 0.0, W as f32, H as f32];
    p.max_pixel_value = 1.0;
    p.pixel_value_limit = f32::MAX;
    p.light_refraction_coefficient = 1.0;
    p.flags = 0; // no colour stage, no range remap - the gather is what is under test
    p.denoise_sigma = sigma;

    let identity = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    FrameTransform { chroma_filter: Default::default(),
        matrices: vec![identity],
        reference_matrices: vec![vec![identity]; references],
        kernel_params: p,
        fov: 1.0,
        minimal_fov: 1.0,
        focal_length: None,
        mesh_data: vec![],
    }
}

fn desc(buf: &mut [u8]) -> BufferDescription<'_> {
    BufferDescription {
        size: (W as usize, H as usize, STRIDE as usize),
        rect: None,
        rotation: None,
        data: BufferSource::Cpu { buffer: buf },
        texture_copy: false,
    }
}

/// Runs `scene` through wgpu and returns the output pixels.
///
/// `Err("UNAVAILABLE: ...")` means the machine has no usable adapter, which is a legitimate
/// skip. Any other `Err` is a failure: collapsing the two is how a test ends up asserting
/// nothing while looking green.
///
/// The `translation2d` half-pixel correction is the one from `run_wgpu` in
/// `cross_backend_color_tests` and is essential, not a fudge: the wgpu *fragment* path is
/// handed `@builtin(position)`, the pixel centre, while the CPU path is handed the integer
/// index. Without it both the centre sampler and `sample_bilinear_at` land between texels
/// and every number here would be measuring a resampling.
fn run_wgpu(scene: &Scene, sigma: f32) -> Result<Vec<[f32; 4]>, String> { run_wgpu_rgb(scene, sigma, 0.0) }
/// `chroma_sigma` > 0 marks the plane as packed RGB (`KernelParams::denoise_rgb`).
fn run_wgpu_rgb(scene: &Scene, sigma: f32, chroma_sigma: f32) -> Result<Vec<[f32; 4]>, String> {
    if crate::gpu::wgpu::WgpuWrapper::initialize_context().is_none() {
        return Err("UNAVAILABLE: no usable wgpu adapter".into());
    }

    let mut itm = identity_transform(scene.references.len(), sigma);
    itm.kernel_params.translation2d = [-0.5, -0.5];
    itm.kernel_params.denoise_rgb = [chroma_sigma, 0.0, 0.0, 0.0];

    let mut input = to_bytes(&scene.centre);
    let mut refs: Vec<Vec<u8>> = scene.references.iter().map(|r| to_bytes(r)).collect();
    let mut output = vec![0u8; (STRIDE * H) as usize];

    let mut buffers = Buffers { guide: None,
        input: desc(&mut input),
        output: desc(&mut output),
        references: refs.iter_mut().map(|r| desc(r.as_mut_slice())).collect(),
    };

    let inst = crate::gpu::wgpu::WgpuWrapper::new(
        &itm.kernel_params,
        RGBAf::wgpu_format().ok_or("RGBAf has no wgpu format")?,
        DistortionModel::default(),
        None,
        &buffers,
        0,
        None,
    ).map_err(|e| format!("WgpuWrapper::new failed: {e:?}"))?;

    if !inst.undistort_image(&mut buffers, &itm, &[]) {
        return Err("undistort_image returned false".into());
    }
    drop(buffers);
    Ok(read_pixels(&output))
}

/// Same scene through OpenCL. This is the only check that the two gather kernels agree, so
/// it exists even though nothing else in the repo drives `OclWrapper` from a test.
///
/// No half-pixel correction here, deliberately: the OpenCL kernel takes its coordinates
/// from `get_global_id` (`opencl_undistort.cl:806`), i.e. the integer pixel index, exactly
/// like the CPU path. Adding wgpu's offset would shift *this* backend off the sampling grid.
fn run_opencl(scene: &Scene, sigma: f32) -> Result<Vec<[f32; 4]>, String> { run_opencl_rgb(scene, sigma, 0.0) }
#[cfg(feature = "use-opencl")]
fn run_opencl_rgb(scene: &Scene, sigma: f32, chroma_sigma: f32) -> Result<Vec<[f32; 4]>, String> {
    if crate::gpu::opencl::OclWrapper::list_devices().is_empty() {
        return Err("UNAVAILABLE: no OpenCL device".into());
    }

    let mut itm = identity_transform(scene.references.len(), sigma);
    itm.kernel_params.denoise_rgb = [chroma_sigma, 0.0, 0.0, 0.0];

    let mut input = to_bytes(&scene.centre);
    let mut refs: Vec<Vec<u8>> = scene.references.iter().map(|r| to_bytes(r)).collect();
    let mut output = vec![0u8; (STRIDE * H) as usize];

    let mut buffers = Buffers { guide: None,
        input: desc(&mut input),
        output: desc(&mut output),
        references: refs.iter_mut().map(|r| desc(r.as_mut_slice())).collect(),
    };

    let inst = crate::gpu::opencl::OclWrapper::new(
        &itm.kernel_params,
        RGBAf::ocl_names(),
        DistortionModel::default(),
        None,
        &buffers,
        0,
        None,
    ).map_err(|e| format!("OclWrapper::new failed: {e:?}"))?;

    inst.undistort_image(&mut buffers, &itm, &[]).map_err(|e| format!("undistort_image failed: {e:?}"))?;
    drop(buffers);
    Ok(read_pixels(&output))
}

#[cfg(not(feature = "use-opencl"))]
fn run_opencl_rgb(_scene: &Scene, _sigma: f32, _chroma_sigma: f32) -> Result<Vec<[f32; 4]>, String> {
    Err("UNAVAILABLE: built without the use-opencl feature (cargo test --features use-opencl)".into())
}

/// Usable output, a skip, or a hard failure. Only a literal `UNAVAILABLE` may skip.
fn resolve(name: &str, r: Result<Vec<[f32; 4]>, String>) -> Option<Vec<[f32; 4]>> {
    match r {
        Ok(v) => Some(v),
        Err(e) if e.starts_with("UNAVAILABLE") => { println!("SKIPPED {name}: {e}"); None }
        Err(e) => panic!("{name} is available but the run failed: {e}"),
    }
}

/// Variance of the *residual* — the difference from the known noiseless scene — over the
/// selected pixels' red channel. Measuring the raw red channel instead would be dominated
/// by the ramp, which denoising does not and should not touch.
fn residual_variance(pixels: &[[f32; 4]], selected: &dyn Fn(usize) -> bool) -> f32 {
    let residuals: Vec<f32> = (0..N).filter(|&i| selected(i)).map(|i| pixels[i][0] - base_pixel(i)[0]).collect();
    let n = residuals.len() as f32;
    let mean = residuals.iter().sum::<f32>() / n;
    residuals.iter().map(|r| (r - mean) * (r - mean)).sum::<f32>() / n
}

fn all(_i: usize) -> bool { true }

// ---------------------------------------------------------------------------------------

/// Averaging `n` frames of independent noise divides its variance by `n`. Generous slack:
/// the rejection Gaussian discards part of every sample, so the realised factor is always
/// below `n`.
///
/// The printed "ideal" figure is the same average computed in Rust from the same noise
/// values with weight 1.0 throughout. It is not asserted on — it is there so that a bad RNG
/// (correlated frames would push it far below `n`) is visible in the output rather than
/// silently making the GPU figure look good or bad for the wrong reason.
#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test -- --ignored denoise_gpu"]
fn denoise_gpu_reduces_variance_on_a_static_scene() {
    const AMPLITUDE: f32 = 0.05;
    // Comfortably above the largest frame-to-frame difference (2 * AMPLITUDE), so genuine
    // noise is accepted. Rejection is what the moving-subject test exercises.
    const SIGMA: f32 = 0.2;

    let scene = Scene::noisy_static(AMPLITUDE);
    let before = residual_variance(&scene.centre, &all);

    let ideal: Vec<[f32; 4]> = (0..N).map(|i| {
        let mut p = base_pixel(i);
        p[0] += (0..GATHER_FRAMES).map(|f| noise(f, i, AMPLITUDE)).sum::<f32>() / GATHER_FRAMES as f32;
        p
    }).collect();
    println!(
        "centre residual variance {before:.3e} (uniform +-{AMPLITUDE} predicts {:.3e}); \
         ideal {GATHER_FRAMES}-frame mean reduces it {:.2}x",
        AMPLITUDE * AMPLITUDE / 3.0,
        before / residual_variance(&ideal, &all)
    );

    let mut ran_any = false;
    for (name, result) in [("wgpu", run_wgpu(&scene, SIGMA)), ("OpenCL", run_opencl(&scene, SIGMA))] {
        if let Some(out) = resolve(name, result) {
            ran_any = true;
            let after = residual_variance(&out, &all);
            let ratio = before / after;
            println!("{name}: residual variance {before:.3e} -> {after:.3e} ({ratio:.2}x reduction, ideal {GATHER_FRAMES}x)");
            assert!(ratio > GATHER_FRAMES as f32 * 0.5,
                "{name}: expected roughly {GATHER_FRAMES}x variance reduction from a {GATHER_FRAMES}-frame gather, got {ratio:.2}x. \
                 Report this, do not weaken the bound.");
        }
    }
    assert!(ran_any, "no backend was available, so this test asserted nothing");
}

/// With no reference frames the gather must not touch a single pixel: `reference_count`
/// comes out as 1 and both kernels return the centre sample unchanged.
///
/// This is also the per-backend identity control for this module. `cross_backend_color_tests`
/// proves it for wgpu and the CPU; OpenCL had no such test until here, and every figure the
/// other two tests produce on OpenCL is meaningless without it.
#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test -- --ignored denoise_gpu"]
fn denoise_gpu_k_zero_is_a_bit_exact_no_op() {
    let mut scene = Scene::noisy_static(0.05);
    scene.references.clear();

    let mut ran_any = false;
    // A non-zero sigma, so a kernel that failed to guard on `reference_count <= 1` would
    // still be caught rather than being saved by an infinity.
    for (name, result) in [("wgpu", run_wgpu(&scene, 0.02)), ("OpenCL", run_opencl(&scene, 0.02))] {
        if let Some(out) = resolve(name, result) {
            ran_any = true;
            for i in 0..N {
                assert_eq!(out[i], scene.centre[i],
                    "{name}: k=0 must not touch a single pixel, but pixel {i} changed \
                     ({:?} -> {:?})", scene.centre[i], out[i]);
            }
        }
    }
    assert!(ran_any, "no backend was available, so this test asserted nothing");
}

/// A subject present only in the centre frame must survive. This is what separates the
/// rejection weighting from a naive temporal average: at `GATHER_FRAMES = 7` a plain mean
/// would fade it to roughly a seventh of its contrast above the background.
///
/// The same run also checks that the *background* was denoised, because "the subject
/// survived" is equally true of a gather that did nothing at all.
#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test -- --ignored denoise_gpu"]
fn denoise_gpu_preserves_a_subject_the_references_do_not_have() {
    // Small enough that the background is accepted at this sigma, while the subject's
    // ~0.5 contrast step is many sigma away and is rejected outright.
    const AMPLITUDE: f32 = 0.005;
    const SIGMA: f32 = 0.02;
    const SUBJECT: std::ops::Range<usize> = 20..44;
    // The interior of the subject, and the background well away from it, so nothing here
    // depends on how an edge pixel is treated.
    let inside = |i: usize| { let (x, y) = (i % W as usize, i / W as usize); (24..40).contains(&x) && (24..40).contains(&y) };
    let outside = |i: usize| { let (x, y) = (i % W as usize, i / W as usize); !(16..48).contains(&x) || !(16..48).contains(&y) };

    let mut scene = Scene::noisy_static(AMPLITUDE);
    for y in SUBJECT { for x in SUBJECT { scene.centre[y * W as usize + x][0] = 1.0; } }

    let background_before = residual_variance(&scene.centre, &outside);
    let naive = {
        let subject_sum: f32 = (0..N).filter(|&i| inside(i)).map(|i| scene.centre[i][0]).sum();
        let refs_sum: f32 = scene.references.iter()
            .map(|r| (0..N).filter(|&i| inside(i)).map(|i| r[i][0]).sum::<f32>())
            .sum();
        (subject_sum + refs_sum) / ((0..N).filter(|&i| inside(i)).count() * GATHER_FRAMES) as f32
    };
    let expected: f32 = (0..N).filter(|&i| inside(i)).map(|i| scene.centre[i][0]).sum::<f32>()
                      / (0..N).filter(|&i| inside(i)).count() as f32;
    println!("subject mean in the centre frame {expected:.4}; an unweighted {GATHER_FRAMES}-frame average would give {naive:.4}");

    let mut ran_any = false;
    for (name, result) in [("wgpu", run_wgpu(&scene, SIGMA)), ("OpenCL", run_opencl(&scene, SIGMA))] {
        if let Some(out) = resolve(name, result) {
            ran_any = true;
            let count = (0..N).filter(|&i| inside(i)).count() as f32;
            let subject = (0..N).filter(|&i| inside(i)).map(|i| out[i][0]).sum::<f32>() / count;
            let background_after = residual_variance(&out, &outside);
            let ratio = background_before / background_after;
            println!("{name}: subject mean {subject:.4} (centre {expected:.4}, naive average {naive:.4}); \
                      background residual variance {background_before:.3e} -> {background_after:.3e} ({ratio:.2}x)");

            assert!((subject - expected).abs() < 0.01,
                "{name}: the subject ghosted away to {subject:.4} from {expected:.4} (an unweighted average \
                 would give {naive:.4}); the rejection weighting is not working. Report this, do not weaken the bound.");
            assert!(ratio > 2.0,
                "{name}: the background was not denoised ({ratio:.2}x variance reduction), so the subject surviving \
                 proves nothing - the gather may not be running at all.");
        }
    }
    assert!(ran_any, "no backend was available, so this test asserted nothing");
}

/// The two gather kernels, compared **per pixel** rather than through a summary statistic.
///
/// The three tests above can all pass on a backend whose kernel is subtly wrong — a
/// variance ratio and a subject mean average thousands of pixels, and two kernels can land
/// on the same aggregate by different routes. This is the only assertion in the repo that
/// `wgpu_undistort.wgsl::gather_denoised` and `opencl_undistort.cl::gather_denoised`
/// produce the same *pixels*, which is the thing Task 13 exists to check.
///
/// Bit-exactness is not required and not expected: `exp()` is implementation-defined on
/// both, and the two accumulate through different compilers. The bound is the same
/// `1/1024` of full scale that `cross_backend_color_tests` holds the colour stage to.
///
/// Requires *both* backends, so it is compiled out entirely without `use-opencl` rather
/// than skipping at runtime. A test that printed `ok` while comparing nothing would let a
/// wgpu-only run look like it had checked the kernels against each other; this way the
/// absence shows up as a smaller test count, which is not mistakable for a pass.
#[cfg(feature = "use-opencl")]
#[test]
#[ignore = "needs a GPU and OpenCL; run with: cd src/core && cargo test --features use-opencl -- --ignored denoise_gpu"]
fn denoise_gpu_wgpu_and_opencl_agree_pixel_for_pixel() {
    /// Max per-channel difference allowed between the two backends, as a fraction of full
    /// scale. Same figure `cross_backend_color_tests::TOLERANCE` uses.
    const TOLERANCE: f32 = 1.0 / 1024.0;

    // A scene that exercises both halves of the gather at once: noise the rejection should
    // accept and average away, plus a subject it should reject outright.
    const AMPLITUDE: f32 = 0.005;
    const SIGMA: f32 = 0.02;
    let mut scene = Scene::noisy_static(AMPLITUDE);
    for y in 20..44 { for x in 20..44 { scene.centre[y * W as usize + x][0] = 1.0; } }

    let (a, b) = match (
        resolve("wgpu", run_wgpu(&scene, SIGMA)),
        resolve("OpenCL", run_opencl(&scene, SIGMA)),
    ) {
        (Some(a), Some(b)) => (a, b),
        (a, b) => panic!("this comparison needs both backends, but wgpu={} OpenCL={} - a build with use-opencl that cannot reach a device must fail, not quietly assert nothing", a.is_some(), b.is_some()),
    };

    let mut max_diff = 0.0f32;
    let mut worst = 0usize;
    let mut exact = 0usize;
    for i in 0..N {
        if a[i] == b[i] { exact += 1; }
        for c in 0..4 {
            let d = (a[i][c] - b[i][c]).abs();
            if d > max_diff { max_diff = d; worst = i; }
        }
    }
    println!("wgpu vs OpenCL: {exact}/{N} pixels bit-identical, max per-channel difference {max_diff:e} \
              at pixel {worst} (wgpu {:?}, OpenCL {:?})", a[worst], b[worst]);

    assert!(max_diff < TOLERANCE,
        "the two gather kernels disagree by {max_diff} at pixel {worst} (wgpu {:?}, OpenCL {:?}), over the \
         {TOLERANCE} tolerance. Report this - do not patch either kernel to make it pass.",
        a[worst], b[worst]);
}

/// Packed RGB is judged as luma and chroma separately, each against its own sigma - the same
/// split the planar YUV path gets from running once per plane. Before this, one distance over
/// R, G and B used the luma sigma alone, so the preview (RGBA) ignored chroma strength.
///
/// Every reference differs from the centre by one fixed offset, with no noise, so each case
/// has an exact expected outcome: rejected (output == centre) or accepted (output moves
/// 6/7 of the way to the references, since the centre keeps weight 1 among 7 frames).
#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test -- --ignored denoise_gpu"]
fn denoise_gpu_rgb_judges_luma_and_chroma_separately() {
    const A: f32 = 0.05;
    const LOOSE: f32 = 0.5;
    const TIGHT: f32 = 0.002;
    // Pure chroma: raises red, lowers blue by exactly as much luma as red added (dY = 0).
    let chroma_offset = [A, 0.0, -A * 0.2126 / 0.0722, 0.0];
    // Pure luma: equal R, G, B, so Cb and Cr do not move.
    let luma_offset = [A, A, A, 0.0];

    let scene_with = |offset: [f32; 4]| -> Scene {
        let centre: Vec<[f32; 4]> = (0..N).map(base_pixel).collect();
        let shifted: Vec<[f32; 4]> = centre.iter().map(|p| [p[0] + offset[0], p[1] + offset[1], p[2] + offset[2], p[3]]).collect();
        Scene { centre, references: vec![shifted; GATHER_FRAMES - 1] }
    };
    // Fraction of the way from the centre to the references, on the channel that moved most.
    let moved = |out: &[[f32; 4]], offset: [f32; 4]| -> f32 {
        let c = (0..3).max_by(|&a, &b| offset[a].abs().total_cmp(&offset[b].abs())).unwrap();
        (0..N).map(|i| (out[i][c] - base_pixel(i)[c]) / offset[c]).sum::<f32>() / N as f32
    };

    let cases = [
        ("chroma change, tight chroma sigma", chroma_offset, LOOSE, TIGHT, false),
        ("chroma change, loose chroma sigma", chroma_offset, TIGHT, LOOSE, true),
        ("luma change, tight luma sigma",     luma_offset,   TIGHT, LOOSE, false),
        ("luma change, loose luma sigma",     luma_offset,   LOOSE, TIGHT, true),
    ];
    let mut ran_any = false;
    for (case, offset, luma_sigma, chroma_sigma, accepted) in cases {
        let scene = scene_with(offset);
        for (name, result) in [("wgpu", run_wgpu_rgb(&scene, luma_sigma, chroma_sigma)), ("OpenCL", run_opencl_rgb(&scene, luma_sigma, chroma_sigma))] {
            if let Some(out) = resolve(name, result) {
                ran_any = true;
                let m = moved(&out, offset);
                println!("{name}: {case}: moved {m:.3} of the way to the references");
                if accepted {
                    assert!(m > 0.8, "{name}: {case}: the references should be accepted (expected ~0.857), moved {m:.3}");
                } else {
                    assert!(m < 0.05, "{name}: {case}: the references should be rejected (expected ~0), moved {m:.3}");
                }
            }
        }
    }
    assert!(ran_any, "no backend was available, so this test asserted nothing");
}

/// A tight chroma sigma must not stop *luma* from being denoised. With one weight per sample,
/// rejecting a sample for its chroma threw its luma away too, so on RGB (the preview) turning
/// chroma strength down also switched luma denoising off. Planar YUV never had this: each
/// plane is weighted on its own. RGB now accumulates luma and chroma separately to match.
#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test -- --ignored denoise_gpu"]
fn denoise_gpu_rgb_tight_chroma_still_denoises_luma() {
    const AMPLITUDE: f32 = 0.02;
    // Every reference: independent grey (luma-only) noise, plus a fixed chroma offset that the
    // tight chroma sigma must reject.
    let chroma_offset = [0.05, 0.0, -0.05 * 0.2126 / 0.0722];
    let frame = |f: usize, offset: bool| -> Vec<[f32; 4]> {
        (0..N).map(|i| {
            let mut p = base_pixel(i);
            let n = noise(f, i, AMPLITUDE);
            for c in 0..3 { p[c] += n + if offset { chroma_offset[c] } else { 0.0 }; }
            p
        }).collect()
    };
    let scene = Scene { centre: frame(K, false), references: (0..GATHER_FRAMES).filter(|&f| f != K).map(|f| frame(f, true)).collect() };
    // Luma residual: the green channel carries luma noise one-to-one here and no chroma offset.
    let green_var = |px: &[[f32; 4]]| -> f32 {
        let r: Vec<f32> = (0..N).map(|i| px[i][1] - base_pixel(i)[1]).collect();
        let m = r.iter().sum::<f32>() / N as f32;
        r.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / N as f32
    };
    let before = green_var(&scene.centre);

    let mut ran_any = false;
    for (name, result) in [("wgpu", run_wgpu_rgb(&scene, 0.5, 0.002)), ("OpenCL", run_opencl_rgb(&scene, 0.5, 0.002))] {
        if let Some(out) = resolve(name, result) {
            ran_any = true;
            let ratio = before / green_var(&out);
            // Blue carries the chroma offset: it must not have moved towards the references.
            let blue_moved = (0..N).map(|i| (out[i][2] - scene.centre[i][2]) / chroma_offset[2]).sum::<f32>() / N as f32;
            println!("{name}: luma variance reduced {ratio:.2}x; chroma moved {blue_moved:.3} towards the rejected offset");
            assert!(ratio > GATHER_FRAMES as f32 * 0.5, "{name}: luma was not denoised ({ratio:.2}x) - a chroma rejection threw the luma away");
            assert!(blue_moved.abs() < 0.05, "{name}: the chroma offset should be rejected, moved {blue_moved:.3}");
        }
    }
    assert!(ran_any, "no backend was available, so this test asserted nothing");
}

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
    if let Some(out) = resolve("OpenCL rgb", run_opencl_chroma_rgb(&scene, 0.5)) {
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
