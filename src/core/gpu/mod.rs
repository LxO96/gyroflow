// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2021-2022 Adrian <adrian.eddy at gmail>

#[cfg(feature = "use-opencl")]
pub mod opencl;
pub mod wgpu;

pub mod wgpu_interop;
#[cfg(not(any(target_os = "macos", target_os = "ios")))] pub mod wgpu_interop_vulkan;
#[cfg(any(target_os = "macos", target_os = "ios"))]      pub mod wgpu_interop_metal;
#[cfg(target_os = "windows")]                            pub mod wgpu_interop_directx;
#[cfg(any(target_os = "windows", target_os = "linux"))]  pub mod wgpu_interop_cuda;

pub mod drawing;
use std::hash::Hasher;

/// The rust-gpu colour stage must be arithmetically identical to `crate::color::apply_color`.
/// `stabilize_spirv` is a path dependency with `crate-type = ["lib", "dylib"]`, so the very
/// same source that is compiled to SPIR-V also compiles for the host and can be called here.
#[cfg(test)]
mod spirv_color_tests {
    use crate::color::{ ColorKernelParams, Lut3D, ResolvedScalars, apply_color };
    use stabilize_spirv::glam::{ Vec4, vec4 };
    use stabilize_spirv::KernelParams as SpirvParams;

    const COLOR_ENABLED: i32 = 1 << 12;
    const HAS_LUT: i32 = 1 << 13;

    /// A deliberately non-identity 3x3x3 LUT: channel-swapped, gamma-ish and offset, so a
    /// wrong index order, a wrong interpolation fraction or a `mix()`-style lerp all show up.
    fn test_lut() -> Lut3D {
        let n = 3usize;
        let mut data = Vec::with_capacity(n * n * n * 3);
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    let (rf, gf, bf) = (r as f32 / 2.0, g as f32 / 2.0, b as f32 / 2.0);
                    data.push(bf * 0.9 + 0.05);
                    data.push(rf * rf);
                    data.push(1.0 - gf * 0.7);
                }
            }
        }
        Lut3D { size: n as u32, domain_min: 0.0, domain_max: 1.0, data, checksum: 1 }
    }

    fn params_and_spirv(cp: &ColorKernelParams, has_lut: bool, max_pixel_value: f32) -> SpirvParams {
        let mut p = SpirvParams::default();
        p.flags = COLOR_ENABLED | if has_lut { HAS_LUT } else { 0 };
        p.max_pixel_value = max_pixel_value;
        p.color_params_1 = vec4(cp.exposure_gain, cp.contrast_factor, cp.saturation_factor, cp.lut_strength);
        p.color_params_2 = vec4(cp.wb_gains[0], cp.wb_gains[1], cp.wb_gains[2], cp.lut_size as f32);
        p.color_params_3 = vec4(cp.domain_min, cp.domain_max, cp.shadows, cp.highlights);
        p
    }

    fn kernel_params(lut: Option<&Lut3D>, s: crate::color::ResolvedScalars) -> ColorKernelParams {
        let mut adj = crate::color::ColorAdjustments::default();
        adj.lut = lut.map(|l| std::sync::Arc::new(l.clone()));
        adj.to_kernel_params(s)
    }

    /// Exact f32 equality. Any tolerance here would mean the expressions diverged.
    #[track_caller]
    fn assert_identical(got: Vec4, expected: [f32; 3], alpha: f32) {
        assert_eq!(got.x.to_bits(), expected[0].to_bits(), "red: {} != {}", got.x, expected[0]);
        assert_eq!(got.y.to_bits(), expected[1].to_bits(), "green: {} != {}", got.y, expected[1]);
        assert_eq!(got.z.to_bits(), expected[2].to_bits(), "blue: {} != {}", got.z, expected[2]);
        assert_eq!(got.w.to_bits(), alpha.to_bits(), "alpha must pass through untouched");
    }

    const PROBES: &[[f32; 3]] = &[
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
        [0.25, 0.5, 0.75],
        [0.9, 0.1, 0.6],
        [0.123456, 0.789012, 0.345678],
        // Outside the 0..1 domain, on both sides - exercises the clamps.
        [-0.5, 1.5, 0.5],
        [-2.0, -0.001, 3.0],
        [2.0, 2.0, 2.0],
    ];

    #[test]
    fn matches_reference_without_lut() {
        for &(exposure, contrast, saturation, shadows, highlights, kelvin, tint) in &[
            (0.0, 0.0, 0.0, 0.0, 0.0, 5500.0, 0.0),
            (1.5, 0.4, -0.3, 0.6, -0.45, 7200.0, 0.25),
            (-2.25, -0.8, 0.9, -1.0, 1.0, 3200.0, -0.4),
        ] {
            let cp = kernel_params(None, ResolvedScalars { exposure, contrast, saturation, shadows, highlights, target_kelvin: kelvin, tint, ..Default::default() });
            let sp = params_and_spirv(&cp, false, 1.0);
            for probe in PROBES {
                let expected = apply_color(*probe, &cp, None);
                let got = stabilize_spirv::color::apply_color(vec4(probe[0], probe[1], probe[2], 0.75), &[], &sp);
                assert_identical(got, expected, 0.75);
            }
        }
    }

    #[test]
    fn matches_reference_with_a_real_lut() {
        let lut = test_lut();
        let coeffs = crate::stabilization::build_coeffs_buffer(Some(&lut));
        assert_eq!(coeffs.len(), crate::stabilization::LUT_DATA_OFFSET + 27 * 3);

        for strength in [0.0f64, 0.5, 1.0] {
            for &(exposure, contrast, saturation, shadows, highlights) in &[(0.0, 0.0, 0.0, 0.0, 0.0), (0.75, 0.3, -0.5, -0.35, 0.8)] {
                let cp = kernel_params(Some(&lut), ResolvedScalars { exposure, contrast, saturation, shadows, highlights, target_kelvin: 6500.0, tint: 0.1, lut_strength: strength });
                let sp = params_and_spirv(&cp, true, 1.0);
                for probe in PROBES {
                    // The reference skips the LUT entirely at strength 0, which is what the
                    // kernel's `lut_strength > 0.0` guard reproduces.
                    let expected = apply_color(*probe, &cp, Some(&lut));
                    let got = stabilize_spirv::color::apply_color(vec4(probe[0], probe[1], probe[2], 1.0), &coeffs, &sp);
                    assert_identical(got, expected, 1.0);
                }
            }
        }
    }

    #[test]
    fn matches_reference_with_a_non_unit_lut_domain() {
        let mut lut = test_lut();
        lut.domain_min = -0.5;
        lut.domain_max = 2.0;
        let coeffs = crate::stabilization::build_coeffs_buffer(Some(&lut));
        let cp = kernel_params(Some(&lut), ResolvedScalars { exposure: 0.25, contrast: 0.2, saturation: 0.35, shadows: 0.5, highlights: -0.6, target_kelvin: 4800.0, tint: -0.2, ..Default::default() });
        let sp = params_and_spirv(&cp, true, 1.0);
        for probe in PROBES {
            let expected = apply_color(*probe, &cp, Some(&lut));
            let got = stabilize_spirv::color::apply_color(vec4(probe[0], probe[1], probe[2], 1.0), &coeffs, &sp);
            assert_identical(got, expected, 1.0);
        }
    }

    /// The kernel works in raw pixel units and normalises by `max_pixel_value`, exactly the
    /// way `cpu_undistort::apply_color_stage` does.
    #[test]
    fn normalises_by_max_pixel_value_like_the_cpu_path() {
        let lut = test_lut();
        let coeffs = crate::stabilization::build_coeffs_buffer(Some(&lut));
        let cp = kernel_params(Some(&lut), ResolvedScalars { exposure: 0.5, contrast: -0.25, saturation: 0.4, shadows: -0.7, highlights: 0.3, target_kelvin: 6000.0, tint: 0.15, lut_strength: 0.75 });
        let max = 255.0f32;
        let sp = params_and_spirv(&cp, true, max);
        let inv_max = 1.0 / max;
        for probe in PROBES {
            let raw = [probe[0] * max, probe[1] * max, probe[2] * max];
            let expected_norm = apply_color([raw[0] * inv_max, raw[1] * inv_max, raw[2] * inv_max], &cp, Some(&lut));
            let expected = [expected_norm[0] * max, expected_norm[1] * max, expected_norm[2] * max];
            let got = stabilize_spirv::color::apply_color(vec4(raw[0], raw[1], raw[2], 200.0), &coeffs, &sp);
            assert_identical(got, expected, 200.0);
        }
    }

    /// Without HAS_LUT the coeffs buffer must never be read, even when one is bound.
    #[test]
    fn has_lut_flag_gates_the_lut() {
        let lut = test_lut();
        let coeffs = crate::stabilization::build_coeffs_buffer(Some(&lut));
        let cp = kernel_params(Some(&lut), ResolvedScalars::default());
        let sp = params_and_spirv(&cp, false, 1.0);
        for probe in PROBES {
            let expected = apply_color(*probe, &cp, None);
            let got = stabilize_spirv::color::apply_color(vec4(probe[0], probe[1], probe[2], 1.0), &coeffs, &sp);
            assert_identical(got, expected, 1.0);
        }
    }

    /// glam's `Vec3::lerp` is `self * (1.0 - s) + rhs * s`, which is NOT bit-identical to the
    /// reference's `a + (b - a) * t`. This pins the difference so nobody "simplifies" the
    /// kernel into using it.
    #[test]
    fn glam_lerp_is_not_the_reference_expression() {
        use stabilize_spirv::glam::Vec3;
        let mut disagreements = 0;
        for a in [0.1f32, 0.333333, 0.7, 0.987654] {
            for b in [0.2f32, 0.456789, 0.8, 1.234567] {
                for t in [0.1f32, 0.3, 0.5, 0.618034, 0.9] {
                    let reference = Vec3::splat(a) + (Vec3::splat(b) - Vec3::splat(a)) * t;
                    let glam_lerp = Vec3::splat(a).lerp(Vec3::splat(b), t);
                    if reference.x.to_bits() != glam_lerp.x.to_bits() { disagreements += 1; }
                }
            }
        }
        assert!(disagreements > 0, "if these ever agree everywhere, the rationale in color.rs needs revisiting");
    }
}

/// Source-text checks on the OpenCL kernel. The kernel itself is only compiled by an
/// OpenCL driver at runtime (and `opencl.rs` only exists under `--features use-opencl`),
/// so these run unconditionally to catch the two things that would otherwise fail
/// silently on a user's machine: a stale LUT offset, and a kernel-argument index that no
/// longer matches the host-side `set_arg` order.
#[cfg(test)]
mod opencl_source_tests {
    const CL: &str = include_str!("opencl_undistort.cl");
    const HOST: &str = include_str!("opencl.rs");

    /// The offset the kernel reads LUT data at must match the Rust constant it mirrors.
    #[test]
    fn lut_data_offset_matches_rust() {
        let expected = format!("#define LUT_DATA_OFFSET {}", crate::stabilization::LUT_DATA_OFFSET);
        assert!(CL.contains(&expected), "opencl_undistort.cl must contain `{expected}`");
    }

    /// A kernel-argument index mismatch is a silent runtime failure, so pin both sides.
    #[test]
    fn kernel_argument_order_matches_host() {
        let device_side = "__kernel void undistort_image(__global const uchar *srcptr, __global uchar *dstptr, __global const void *params_buf, __global const float *matrices, __global const uchar *drawing, __global const float *mesh_data, __global const float *lut_coeffs)";
        assert!(CL.contains(device_side), "undistort_image signature changed; update the host-side arg order in opencl.rs too:\n{device_side}");

        // Host side: the `.arg()` calls must appear in exactly this order, because kernel
        // argument indices are positional. Presence alone is not enough - two swapped
        // `.arg()` calls would still bind, just to the wrong parameters.
        let mut prev: Option<(&str, usize)> = None;
        for (idx, name) in ["source_buffer", "dest_buffer", "buf_params", "buf_matrices", "buf_drawing", "buf_mesh_data", "buf_coeffs"].iter().enumerate() {
            let expected = format!(".arg(&{name})");
            let at = HOST.find(&expected).unwrap_or_else(|| panic!("opencl.rs must bind `{name}` as kernel argument {idx} via `{expected}`"));
            if let Some((prev_name, prev_at)) = prev {
                assert!(at > prev_at, "`{name}` must be bound after `{prev_name}`: kernel argument {idx} is positional, and opencl_undistort.cl expects this exact order");
            }
            prev = Some((name, at));
        }
    }

    /// The colour arithmetic must stay bit-identical to `crate::color::apply_color`;
    /// `mix()` and `dot()` are not (see the comment block in the .cl).
    #[test]
    fn colour_stage_avoids_non_identical_builtins() {
        let start = CL.find("float3 lerp3(").expect("colour block missing from opencl_undistort.cl");
        let end = CL.find("float map_coord(").expect("colour block end marker missing");
        let block = &CL[start..end];
        assert!(!block.contains("mix("), "the OpenCL colour stage must not use mix()");
        assert!(!block.contains("dot("), "the OpenCL colour stage must not use dot()");
    }
}

#[derive(Debug, Default)]
pub struct BufferDescription<'a> {
    pub size: (usize, usize, usize), // width, height, stride
    pub rect: Option<(usize, usize, usize, usize)>, // x, y, width, height
    pub rotation: Option<f32>, // pixels rotation in degrees
    pub data: BufferSource<'a>,
    pub texture_copy: bool
}
pub struct Buffers<'a> {
    pub input: BufferDescription<'a>,
    pub output: BufferDescription<'a>,
    /// Temporal reference frames, in temporal order, excluding the centre frame
    /// which is `input`. Empty when denoising is off.
    pub references: Vec<BufferDescription<'a>>,
}

#[derive(Debug, Default)]
pub enum BufferSource<'a> {
    #[default]
    None,
    Cpu { buffer: &'a mut [u8] },
    #[cfg(feature = "use-opencl")]
    OpenCL {
        texture: ocl::ffi::cl_mem,
        queue: ocl::ffi::cl_command_queue
    },
    #[cfg(target_os = "windows")]
    DirectX11 {
        texture: *mut std::ffi::c_void, // ID3D11Texture2D*
        device: *mut std::ffi::c_void, // ID3D11Device*
        device_context: *mut std::ffi::c_void, // ID3D11DeviceContext*
    },
    OpenGL {
        texture: u32, // GLuint
        context: *mut std::ffi::c_void, // OpenGL context pointer
    },
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    Vulkan {
        texture: u64,
        device: u64,
        physical_device: u64,
        instance: u64,
    },
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    Metal {
        texture: *mut std::ffi::c_void,
        command_queue: *mut std::ffi::c_void,
    },
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    MetalBuffer {
        buffer: *mut std::ffi::c_void,
        command_queue: *mut std::ffi::c_void,
    },
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    CUDABuffer {
        buffer: *mut std::ffi::c_void // Cudeviceptr
    },
}
impl<'a> BufferDescription<'a> {
    pub fn get_checksum(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.write_usize(self.size.0);
        hasher.write_usize(self.size.1);
        hasher.write_usize(self.size.2);
        if let Some(r) = self.rect {
            hasher.write_usize(r.0);
            hasher.write_usize(r.1);
            hasher.write_usize(r.2);
            hasher.write_usize(r.3);
        }
        hasher.write_u32(self.rotation.unwrap_or_default().to_bits());
        match &self.data {
            BufferSource::None => { }
            BufferSource::Cpu { .. } => { }
            #[cfg(feature = "use-opencl")]
            BufferSource::OpenCL { texture: _, queue } => {
                // if !self.texture_copy {
                //     hasher.write_u64(*texture as u64);
                // }
                hasher.write_u64(*queue as u64);
            }
            BufferSource::OpenGL { texture, context } => {
                if !self.texture_copy {
                    hasher.write_u32(*texture);
                }
                hasher.write_u64(*context as u64);
            }
            #[cfg(target_os = "windows")]
            BufferSource::DirectX11 { texture, device, device_context } => {
                if !self.texture_copy {
                    hasher.write_u64(*texture as u64);
                }
                hasher.write_u64(*device as u64);
                hasher.write_u64(*device_context as u64);
            },
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            BufferSource::Vulkan { texture, instance, device, physical_device } => {
                if !self.texture_copy {
                    hasher.write_u64(*texture);
                }
                hasher.write_u64(*instance);
                hasher.write_u64(*device);
                hasher.write_u64(*physical_device);
            },
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            BufferSource::CUDABuffer { buffer } => {
                if !self.texture_copy {
                    hasher.write_u64(*buffer as u64);
                }
                hasher.write_i32(wgpu_interop_cuda::get_current_cuda_device());
            },
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            BufferSource::Metal { texture, command_queue } => {
                if !self.texture_copy {
                    hasher.write_u64(*texture as u64);
                }
                hasher.write_u64(*command_queue as u64);
            },
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            BufferSource::MetalBuffer { buffer, command_queue } => {
                if !self.texture_copy {
                    hasher.write_u64(*buffer as u64);
                }
                hasher.write_u64(*command_queue as u64);
            },
        }
        hasher.finalize()
    }
}
impl<'a> Buffers<'a> {
    pub fn get_checksum(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.write_u32(self.input.get_checksum());
        hasher.write_u32(self.output.get_checksum());
        hasher.write_usize(self.references.len());
        hasher.finalize()
    }
}

pub fn initialize_contexts() -> Option<(String, String)> {
    #[cfg(feature = "use-opencl")]
    if std::env::var("NO_OPENCL").unwrap_or_default().is_empty() {
        let cl = std::panic::catch_unwind(|| {
            opencl::OclWrapper::initialize_context(None)
        });
        match cl {
            Ok(Ok(names)) => { return Some(names); },
            Ok(Err(e)) => { log::error!("OpenCL error init: {:?}", e); },
            Err(e) => {
                if let Some(s) = e.downcast_ref::<&str>() {
                    log::error!("Failed to initialize OpenCL {}", s);
                } else if let Some(s) = e.downcast_ref::<String>() {
                    log::error!("Failed to initialize OpenCL {}", s);
                } else {
                    log::error!("Failed to initialize OpenCL {:?}", e);
                }
            }
        }
    }

    if std::env::var("NO_WGPU").unwrap_or_default().is_empty() {
        let wgpu = std::panic::catch_unwind(|| {
            wgpu::WgpuWrapper::initialize_context()
        });
        match wgpu {
            Ok(Some(names)) => { return Some(names); },
            Ok(None) => { log::error!("wgpu init error"); },
            Err(e) => {
                if let Some(s) = e.downcast_ref::<&str>() {
                    log::error!("Failed to initialize wgpu {}", s);
                } else if let Some(s) = e.downcast_ref::<String>() {
                    log::error!("Failed to initialize wgpu {}", s);
                } else {
                    log::error!("Failed to initialize wgpu {:?}", e);
                }
            }
        }
    }

    None
}
