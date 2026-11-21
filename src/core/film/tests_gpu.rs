// Copyright © 2026 Gyroflow

//! The film chain on real backends, against `film::reference`. Fixture: the identity transform
//! of `denoise::tests_gpu` on RGBA16 (the format exports bake into) and RGBA8 (the preview).
//! Run: cd src/core && cargo test --features use-opencl -- --ignored film_gpu --nocapture

use crate::gpu::{ Buffers, BufferDescription, BufferSource };
use crate::stabilization::{ FrameTransform, KernelParams, PixelType, RGBA8, RGBA16 };
use crate::stabilization::distortion_models::DistortionModel;
use super::{ FilmContext, FilmSettings, FilmParams, reference };

pub(super) const W: usize = 67; // odd, and not a multiple of the halo factor
pub(super) const H: usize = 45;

pub(super) fn scene() -> Vec<[f32; 4]> {
    (0..W * H).map(|i| {
        let (x, y) = (i % W, i / W);
        if (30..38).contains(&x) && (18..26).contains(&y) { return [1.0, 1.0, 1.0, 1.0]; } // highlight for halation
        let t = x as f32 / (W - 1) as f32;
        [0.1 + 0.8 * t, 0.5 - 0.3 * t, 0.3 + 0.4 * (y as f32 / (H - 1) as f32), 1.0]
    }).collect()
}

/// Golden 200 on Super 8 with everything on: export short 45 → p = 1, R = 0.008·45·24/4.01 ≈ 2.2 px
/// (f = 1). `preview_params` views an export 4x larger (p = 0.25) to exercise the preview lattice.
pub(super) fn params(max_value: f32) -> FilmParams {
    let s = FilmSettings { stock: Some("golden200".into()), format: super::stocks::FilmFormat::Super8, halation_amount: 1.0, ..Default::default() };
    let mut p = FilmContext::new(s, H, 25.0, "C0001.MP4").unwrap().frame_params(W, H, 40_000, max_value);
    p.width = W as i32; p.height = H as i32;
    p
}
pub(super) fn preview_params(max_value: f32) -> FilmParams {
    let s = FilmSettings { stock: Some("vivid100".into()), format: super::stocks::FilmFormat::Super16, ..Default::default() };
    let mut p = FilmContext::new(s, 4 * H, 25.0, "C0001.MP4").unwrap().frame_params(W, H, 80_000, max_value);
    p.width = W as i32; p.height = H as i32;
    assert!((p.inv_p - 4.0).abs() < 1e-6);
    p
}
/// Same, with the halo grid forced to f = 4 (67x45 → 17x12, last blocks clipped): see
/// `force_halo_grid` for why a frame this small cannot reach it by itself.
pub(super) fn halo_f_params(max_value: f32) -> FilmParams {
    let s = FilmSettings { stock: Some("tungsten800".into()), format: super::stocks::FilmFormat::Super8, halation_amount: 1.0, ..Default::default() };
    let mut p = FilmContext::new(s, H, 25.0, "x").unwrap().frame_params(W, H, 0, max_value);
    p.width = W as i32; p.height = H as i32;
    super::force_halo_grid(&mut p, 4, W, H);
    p
}

pub(super) fn identity(bpp: usize, max_value: f32, film: FilmParams) -> FrameTransform {
    let mut p = KernelParams::default();
    let (w, h, s) = (W as i32, H as i32, (W * bpp) as i32);
    p.width = w; p.height = h; p.stride = s;
    p.output_width = w; p.output_height = h; p.output_stride = s;
    p.matrix_count = 1; p.interpolation = 2;
    p.bytes_per_pixel = bpp as i32; p.pix_element_count = 4;
    // Upstream's kernel builds the output plane itself, `n = (p - out_c) / f`, projects it rectilinearly and the lens model images it again: with `f` = the frame width the rays stay well inside the field (GPU trig is too coarse for the tiny angles a huge `f` gives), the round trip is exact to well under the 1/32 px the interpolation tables resolve, and `uv = n * f + c` hands back `p` when `c` is the frame centre.
    p.f = [w as f32, w as f32]; p.c = [w as f32 / 2.0, h as f32 / 2.0]; p.fov = 1.0; p.lens_correction_amount = 1.0;
    p.source_rect = [0, 0, w, h]; p.output_rect = [0, 0, w, h];
    p.safe_area_rect = [0.0, 0.0, w as f32, h as f32];
    p.max_pixel_value = max_value; p.pixel_value_limit = max_value;
    p.light_refraction_coefficient = 1.0;
    let identity = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    FrameTransform { chroma_filter: Default::default(), film, matrices: vec![identity], reference_matrices: vec![], kernel_params: p, fov: 1.0, minimal_fov: 1.0, focal_length: None, mesh_data: vec![] }
}

pub(super) fn to16(px: &[[f32; 4]]) -> Vec<u8> { px.iter().flat_map(|p| p.iter().flat_map(|v| ((v * 65535.0).round() as u16).to_le_bytes())).collect() }
pub(super) fn from16(b: &[u8]) -> Vec<[f32; 4]> { b.chunks_exact(8).map(|c| [0, 1, 2, 3].map(|i| u16::from_le_bytes([c[i * 2], c[i * 2 + 1]]) as f32 / 65535.0)).collect() }
pub(super) fn to8(px: &[[f32; 4]]) -> Vec<u8> { px.iter().flat_map(|p| p.map(|v| (v * 255.0).round() as u8)).collect() }
pub(super) fn from8(b: &[u8]) -> Vec<[f32; 4]> { b.chunks_exact(4).map(|c| [0, 1, 2, 3].map(|i| c[i] as f32 / 255.0)).collect() }
pub(super) fn desc(buf: &mut [u8], bpp: usize) -> BufferDescription<'_> {
    BufferDescription { size: (W, H, W * bpp), rect: None, rotation: None, data: BufferSource::Cpu { buffer: buf }, texture_copy: false }
}
/// The reference on the *quantised* input, so only the film chain differs.
pub(super) fn expected(input: &[[f32; 4]], p: &FilmParams) -> Vec<[f32; 4]> { reference::render(input, W, H, p) }
pub(super) fn max_diff(a: &[[f32; 4]], b: &[[f32; 4]]) -> f32 { a.iter().zip(b).flat_map(|(x, y)| (0..4).map(move |i| (x[i] - y[i]).abs())).fold(0.0, f32::max) }

#[cfg(feature = "use-opencl")]
pub(super) fn run_opencl16(input: &[u8], film: FilmParams) -> Result<Vec<u8>, String> {
    if crate::gpu::opencl::OclWrapper::list_devices().is_empty() { return Err("UNAVAILABLE: no OpenCL device".into()); }
    let itm = identity(8, 65535.0, film);
    let mut inp = input.to_vec();
    let mut out = vec![0u8; W * H * 8];
    let mut buffers = Buffers { input: desc(&mut inp, 8), output: desc(&mut out, 8), references: vec![], guide: None };
    let inst = crate::gpu::opencl::OclWrapper::new(&itm.kernel_params, RGBA16::ocl_names(), DistortionModel::default(), None, &buffers, 0, None).map_err(|e| format!("OclWrapper::new failed: {e:?}"))?;
    inst.undistort_image(&mut buffers, &itm, &[]).map_err(|e| format!("undistort_image failed: {e:?}"))?;
    drop(buffers);
    Ok(out)
}
#[cfg(feature = "use-opencl")]
pub(super) fn run_opencl8(input: &[u8], film: FilmParams) -> Result<Vec<u8>, String> {
    if crate::gpu::opencl::OclWrapper::list_devices().is_empty() { return Err("UNAVAILABLE: no OpenCL device".into()); }
    let itm = identity(4, 255.0, film);
    let mut inp = input.to_vec();
    let mut out = vec![0u8; W * H * 4];
    let mut buffers = Buffers { input: desc(&mut inp, 4), output: desc(&mut out, 4), references: vec![], guide: None };
    let inst = crate::gpu::opencl::OclWrapper::new(&itm.kernel_params, RGBA8::ocl_names(), DistortionModel::default(), None, &buffers, 0, None).map_err(|e| format!("OclWrapper::new failed: {e:?}"))?;
    inst.undistort_image(&mut buffers, &itm, &[]).map_err(|e| format!("undistort_image failed: {e:?}"))?;
    drop(buffers);
    Ok(out)
}
#[cfg(not(feature = "use-opencl"))] pub(super) fn run_opencl16(_: &[u8], _: FilmParams) -> Result<Vec<u8>, String> { Err("UNAVAILABLE: built without use-opencl".into()) }
#[cfg(not(feature = "use-opencl"))] pub(super) fn run_opencl8(_: &[u8], _: FilmParams) -> Result<Vec<u8>, String> { Err("UNAVAILABLE: built without use-opencl".into()) }

pub(super) fn ok<T>(name: &str, r: Result<T, String>) -> Option<T> {
    match r { Ok(v) => Some(v), Err(e) if e.starts_with("UNAVAILABLE") => { println!("SKIPPED {name}: {e}"); None }, Err(e) => panic!("{name} failed: {e}") }
}

#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored film_gpu"]
fn film_gpu_opencl_matches_the_reference() {
    for (name, p) in [("full", params(65535.0)), ("halo f=4", halo_f_params(65535.0)), ("preview p=0.25", preview_params(65535.0))] {
        let input = to16(&scene());
        let Some(out) = ok("OpenCL 16", run_opencl16(&input, p)) else { return; };
        let d = max_diff(&from16(&out), &expected(&from16(&input), &p));
        println!("OpenCL RGBA16 {name}: max diff {:.2} codes", d * 65535.0);
        assert!(d <= 1.0 / 65535.0 + 1e-6, "OpenCL RGBA16 {name} differs from the reference by {} codes", d * 65535.0);
    }
    let input = to8(&scene());
    if let Some(out) = ok("OpenCL 8", run_opencl8(&input, params(255.0))) {
        let d = max_diff(&from8(&out), &expected(&from8(&input), &params(255.0)));
        println!("OpenCL RGBA8: max diff {:.2} codes", d * 255.0);
        assert!(d <= 1.0 / 255.0 + 1e-6, "OpenCL RGBA8 differs by {} codes", d * 255.0);
    }
}

#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored film_gpu"]
fn film_gpu_opencl_inactive_is_bit_exact() {
    let input = to16(&scene());
    if let Some(out) = ok("OpenCL", run_opencl16(&input, FilmParams::default())) { assert_eq!(out, input); }
}

/// Review focus 2: grain and halation on a full-scale 16-bit frame must clamp, never wrap.
#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored film_gpu"]
fn film_gpu_opencl_full_scale_does_not_wrap() {
    let s = FilmSettings { stock: Some("tungsten800".into()), tone: false, grain_amount: 2.0, halation_amount: 1.0, ..Default::default() };
    let mut p = FilmContext::new(s, H, 25.0, "x").unwrap().frame_params(W, H, 0, 65535.0);
    p.width = W as i32; p.height = H as i32;
    let input = to16(&vec![[1.0, 1.0, 1.0, 1.0]; W * H]);
    if let Some(out) = ok("OpenCL", run_opencl16(&input, p)) {
        let min = from16(&out).iter().map(|px| px[0]).fold(1.0, f32::min);
        assert!(min > 60000.0 / 65535.0, "a clipped pixel wrapped to {}", min * 65535.0);
    }
}

#[cfg(feature = "use-opencl")]
#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored film_gpu"]
fn film_gpu_opencl_build_failure_skips_film_and_keeps_the_backend() {
    crate::gpu::opencl::FAIL_FILM_BUILD.with(|f| f.set(true));
    let input = to16(&scene());
    let r = run_opencl16(&input, params(65535.0));
    crate::gpu::opencl::FAIL_FILM_BUILD.with(|f| f.set(false));
    if let Some(out) = ok("OpenCL", r) {
        assert_eq!(out, input, "with film failing to build the frame must come out unfilmed");
        assert!(super::last_skip().is_some(), "the skip reason must be recorded for the panel");
    }
}

fn run_wgpu(input: &[u8], bpp: usize, films: &[FilmParams]) -> Result<Vec<Vec<u8>>, String> {
    if crate::gpu::wgpu::WgpuWrapper::initialize_context().is_none() { return Err("UNAVAILABLE: no usable wgpu adapter".into()); }
    let max = if bpp == 8 { 65535.0 } else { 1.0 }; // Rgba8Unorm is normalised on wgpu
    let fmt = if bpp == 8 { RGBA16::wgpu_format() } else { RGBA8::wgpu_format() }.ok_or("no format")?;
    let mut itm = identity(bpp, max, FilmParams::default());
    itm.kernel_params.translation2d = [-0.5, -0.5]; // fragment path samples at pixel centres; see denoise::tests_gpu::run_wgpu
    let mut inp = input.to_vec();
    let mut out = vec![0u8; W * H * bpp];
    let mut buffers = Buffers { input: desc(&mut inp, bpp), output: desc(&mut out, bpp), references: vec![], guide: None };
    let inst = crate::gpu::wgpu::WgpuWrapper::new(&itm.kernel_params, fmt, DistortionModel::default(), None, &buffers, 0, None).map_err(|e| format!("WgpuWrapper::new failed: {e:?}"))?;
    let mut results = vec![];
    for f in films {
        itm.film = *f;
        if !inst.undistort_image(&mut buffers, &itm, &[]) { return Err("undistort_image returned false".into()); }
        if let BufferSource::Cpu { buffer } = &buffers.output.data { results.push(buffer.to_vec()); }
    }
    Ok(results)
}

#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored film_gpu"]
fn film_gpu_wgpu_matches_the_reference_and_opencl() {
    let input = to16(&scene());
    // One wrapper, three frames: full chain, a different halo grid (review focus 4), then off.
    let frames = [params(65535.0), halo_f_params(65535.0), preview_params(65535.0), FilmParams::default()];
    let out = ok("wgpu 16", run_wgpu(&input, 8, &frames)).expect("wgpu is required for this test");
    for (i, name) in ["full", "halo f=4", "preview p=0.25"].iter().enumerate() {
        let d = max_diff(&from16(&out[i]), &expected(&from16(&input), &frames[i]));
        println!("wgpu RGBA16 {name}: max diff {:.2} codes", d * 65535.0);
        assert!(d <= 1.0 / 65535.0 + 1e-6, "wgpu RGBA16 {name} differs by {} codes", d * 65535.0);
        if let Some(o) = ok("OpenCL 16", run_opencl16(&input, frames[i])) {
            let d = max_diff(&from16(&o), &from16(&out[i]));
            println!("wgpu vs OpenCL {name}: {:.2} codes", d * 65535.0);
            assert!(d <= 1.0 / 65535.0 + 1e-6);
        }
    }
    assert_eq!(out[3], input, "film off must be bit-exact after film frames on the same wrapper");

    let input8 = to8(&scene());
    // 255, as the preview passes it: an Rgba8Unorm texture reads 0..1 regardless, so the
    // wgpu chain must normalise by itself rather than trust the caller's scale.
    let p8 = params(255.0);
    let out8 = ok("wgpu 8", run_wgpu(&input8, 4, &[p8])).unwrap();
    let d = max_diff(&from8(&out8[0]), &expected(&from8(&input8), &p8));
    println!("wgpu RGBA8: max diff {:.2} codes", d * 255.0);
    assert!(d <= 1.0 / 255.0 + 1e-6, "wgpu RGBA8 differs by {} codes", d * 255.0);
}

#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored film_gpu"]
fn film_gpu_wgpu_full_scale_does_not_wrap() {
    let s = FilmSettings { stock: Some("tungsten800".into()), tone: false, grain_amount: 2.0, halation_amount: 1.0, ..Default::default() };
    let mut p = FilmContext::new(s, H, 25.0, "x").unwrap().frame_params(W, H, 0, 65535.0);
    p.width = W as i32; p.height = H as i32;
    let input = to16(&vec![[1.0, 1.0, 1.0, 1.0]; W * H]);
    if let Some(out) = ok("wgpu", run_wgpu(&input, 8, &[p])) {
        let min = from16(&out[0]).iter().map(|px| px[0]).fold(1.0, f32::min);
        assert!(min > 60000.0 / 65535.0, "a clipped pixel wrapped to {}", min * 65535.0);
    }
}

#[test]
#[ignore = "needs a GPU; run with: cd src/core && cargo test --features use-opencl -- --ignored film_gpu"]
fn film_gpu_wgpu_build_failure_skips_film_and_keeps_the_backend() {
    crate::gpu::wgpu::FAIL_FILM_BUILD_WGPU.with(|f| f.set(true));
    let input = to16(&scene());
    let r = run_wgpu(&input, 8, &[params(65535.0)]);
    crate::gpu::wgpu::FAIL_FILM_BUILD_WGPU.with(|f| f.set(false));
    if let Some(out) = ok("wgpu", r) { assert_eq!(out[0], input); }
}

