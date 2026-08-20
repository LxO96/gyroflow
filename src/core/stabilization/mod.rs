// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2021-2022 Adrian <adrian.eddy at gmail>

use std::collections::BTreeMap;
use std::cell::RefCell;

use crate::GyroflowCoreError;

use super::gpu::*;
use super::StabilizationManager;
use drawing::DrawCanvas;

mod compute_params;
mod frame_transform;
mod cpu_undistort;
mod pixel_formats;
// mod interpolation;
pub mod distortion_models;
pub use pixel_formats::*;
pub use compute_params::ComputeParams;
pub use frame_transform::FrameTransform;
pub use cpu_undistort::*;

/// The LUT is appended to the coeffs buffer after the fixed interpolation
/// coefficients, so no backend needs an extra binding.
pub const LUT_DATA_OFFSET: usize = 64 + 128 + 256 + 9 * 4 + 4; // = 488, must equal COEFFS.len()

/// `COEFFS` followed by the LUT table, which is what every backend uploads as
/// the `coeffs` buffer. Returns just `COEFFS` when there is no LUT.
pub fn build_coeffs_buffer(lut: Option<&crate::color::Lut3D>) -> Vec<f32> {
    let mut out = Vec::with_capacity(LUT_DATA_OFFSET + lut.map(|l| l.data.len()).unwrap_or(0));
    out.extend_from_slice(&COEFFS);
    debug_assert_eq!(out.len(), LUT_DATA_OFFSET);
    if let Some(lut) = lut { out.extend_from_slice(&lut.data); }
    out
}

#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum Interpolation {
    #[default]
    Bilinear = 2,
    Bicubic  = 4,
    Lanczos4 = 8,
    RobidouxSharp = 10,
    Robidoux = 11,
    Mitchell = 12,
    CatmullRom = 13
}
impl From<&str> for Interpolation {
    fn from(s: &str) -> Self {
        match s {
            "Bilinear"           => Interpolation::Bilinear,
            "Bicubic"            => Interpolation::Bicubic,
            "Lanczos4"           => Interpolation::Lanczos4,
            "EWA: RobidouxSharp" => Interpolation::RobidouxSharp,
            "EWA: Robidoux"      => Interpolation::Robidoux,
            "EWA: Mitchell"      => Interpolation::Mitchell,
            "EWA: Catmull-Rom"   => Interpolation::CatmullRom,
            _ => Interpolation::Lanczos4
        }
    }
}

struct ThreadLocalWgpuCache(RefCell<lru::LruCache<u32, wgpu::WgpuWrapper>>);
impl Drop for ThreadLocalWgpuCache {
    fn drop(&mut self) {
        // Workaround for a Vulkan hang on device destroy (https://github.com/gfx-rs/wgpu/issues/4973)
        let inner = self.0.replace(lru::LruCache::new(std::num::NonZeroUsize::new(1).unwrap()));
        std::thread::spawn(move || drop(inner));
    }
}

lazy_static::lazy_static! {
    pub static ref GPU_LIST: parking_lot::RwLock<Vec<String>> = parking_lot::RwLock::new(Vec::new());
}
thread_local! {
    static CACHED_WGPU: ThreadLocalWgpuCache = ThreadLocalWgpuCache(RefCell::new(lru::LruCache::new(std::num::NonZeroUsize::new(15).unwrap())));
    #[cfg(feature = "use-opencl")]
    static CACHED_OPENCL: RefCell<lru::LruCache<u32, opencl::OclWrapper>> = RefCell::new(lru::LruCache::new(std::num::NonZeroUsize::new(15).unwrap()));
}

/// Drop the current thread's cached GPU pipelines to reclaim VRAM.
///
/// Uses the same cross-thread-drop pattern as `ThreadLocalWgpuCache::drop` to
/// avoid the Vulkan-on-destroy hang (gfx-rs/wgpu#4973).
pub fn clear_gpu_cache_current_thread() {
    CACHED_WGPU.with(|x| {
        let inner = x.0.replace(lru::LruCache::new(std::num::NonZeroUsize::new(15).unwrap()));
        std::thread::spawn(move || drop(inner));
    });
    #[cfg(feature = "use-opencl")]
    CACHED_OPENCL.with(|x| {
        x.borrow_mut().clear();
    });
}

bitflags::bitflags! {
    #[derive(Default, Clone)]
    pub struct KernelParamsFlags: i32 {
        const FIX_COLOR_RANGE      = 1 << 0; // 1
        const HAS_DIGITAL_LENS     = 1 << 1; // 2
        const FILL_WITH_BACKGROUND = 1 << 2; // 4
        const DRAWING_ENABLED      = 1 << 3; // 8
        const HORIZONTAL_RS        = 1 << 4; // 16, right-to-left or left-to-right rolling shutter
        const HAS_SOURCE_RECT      = 1 << 5; // 32
        const HAS_OUTPUT_RECT      = 1 << 6; // 64
        const FRAMEBUFFER_INVERTED = 1 << 7; // 128
        const HAS_IBIS_DATA        = 1 << 8; // 256
        const HAS_MESH_DATA        = 1 << 9; // 512
        const HAS_FPD_DATA         = 1 << 10; // 1024
        const ANY_UNDERWATER       = 1 << 11; // 2048
        const COLOR_ENABLED        = 1 << 12; // 4096
        const HAS_LUT              = 1 << 13; // 8192
    }
}

// Each parameter must be aligned to 4 bytes and whole struct to 16 bytes
// Must be kept in sync with: opencl_undistort.cl, wgpu_undistort.wgsl,
// qt_gpu/undistort.frag and gpu/stabilize_spirv/src/types.rs
#[repr(C, packed(4))]
#[derive(Default, Copy, Clone)]
pub struct KernelParams {
    pub width:             i32, // 4
    pub height:            i32, // 8
    pub stride:            i32, // 12
    pub output_width:      i32, // 16
    pub output_height:     i32, // 4
    pub output_stride:     i32, // 8
    pub matrix_count:      i32, // 12 - for rolling shutter correction. 1 = no correction, only main matrix
    pub interpolation:     i32, // 16
    pub background_mode:   i32, // 4
    pub flags:             i32, // 8
    pub bytes_per_pixel:   i32, // 12
    pub pix_element_count: i32, // 16
    pub background:        [f32; 4], // 16
    pub f:                 [f32; 2], // 8  - focal length in pixels
    pub c:                 [f32; 2], // 16 - lens center
    pub k:                 [f32; 12], // 16,16,16 - distortion coefficients
    pub fov:               f32, // 4
    pub r_limit:           f32, // 8
    pub lens_correction_amount:   f32, // 12
    pub input_vertical_stretch:   f32, // 16
    pub input_horizontal_stretch: f32, // 4
    pub background_margin:        f32, // 8
    pub background_margin_feather:f32, // 12
    pub canvas_scale:             f32, // 16
    pub input_rotation:           f32, // 4
    pub output_rotation:          f32, // 8
    pub translation2d:            [f32; 2], // 16
    pub translation3d:            [f32; 4], // 16
    pub source_rect:              [i32; 4], // 16 - x, y, w, h
    pub output_rect:              [i32; 4], // 16 - x, y, w, h
    pub digital_lens_params:      [f32; 16], // 16,16,16,16
    pub safe_area_rect:           [f32; 4], // 16
    pub max_pixel_value:          f32, // 4
    pub distortion_model:         stabilize_spirv::DistortionModel, // 8
    pub digital_lens:             stabilize_spirv::DistortionModel, // 12
    pub pixel_value_limit:        f32, // 16
    pub light_refraction_coefficient: f32, // 4
    pub plane_index:              i32, // 8
    pub reserved1:                f32, // 12
    pub reserved2:                f32, // 16
    pub ewa_coeffs_p:             [f32; 4], // 16
    pub ewa_coeffs_q:             [f32; 4], // 16
    pub color_params_1:           [f32; 4], // 16 - exposure_gain, contrast_factor, saturation_factor, lut_strength
    pub color_params_2:           [f32; 4], // 16 - wb_gain_r, wb_gain_g, wb_gain_b, lut_size
    pub color_params_3:           [f32; 4], // 16 - lut_domain_min, lut_domain_max, unused, unused
}
unsafe impl bytemuck::Zeroable for KernelParams {}
unsafe impl bytemuck::Pod for KernelParams {}

#[derive(Default, Debug)]
pub enum BackendType {
    #[default]
    None,
    OpenCL(u32),
    Wgpu(u32),
    Cpu(u32)
}
impl BackendType {
    pub fn get_hash(&self) -> u32 {
        match self { BackendType::Cpu(x) => *x, BackendType::OpenCL(x) => *x, BackendType::Wgpu(x) => *x, _ => 0 }
    }
    pub fn is_none(&self) -> bool { matches!(self, Self::None) }
    pub fn is_wgpu(&self) -> bool { matches!(self, Self::Wgpu(_)) }
}

#[derive(Default)]
pub struct Stabilization {
    pub stab_data: BTreeMap<i64, FrameTransform>,

    pub size:        (usize, usize), // width, height
    pub output_size: (usize, usize), // width, height

    pub interpolation: Interpolation,
    pub kernel_flags: KernelParamsFlags,

    #[cfg(feature = "use-opencl")]
    cl: Option<opencl::OclWrapper>,

    pub wgpu: Option<wgpu::WgpuWrapper>,

    pub initialized_backend: BackendType,

    compute_params: ComputeParams,

    pub drawing: DrawCanvas,
    pub pending_device_change: Option<isize>,

    pub share_wgpu_instances: bool,
    pub cache_frame_transform: bool,
    next_backend: Option<&'static str>
}

#[derive(Debug)]
pub struct ProcessedInfo {
    pub fov: f64,
    pub minimal_fov: f64,
    pub focal_length: Option<f64>,
    pub backend: &'static str,
}

impl Stabilization {
    pub fn set_compute_params(&mut self, params: ComputeParams) {
        self.stab_data.clear();
        self.compute_params = params;
    }

    fn get_rect(desc: &BufferDescription) -> [i32; 4] {
        let mut ret = [0i32; 4];
        if let Some(r) = desc.rect {
            ret[0] = r.0 as i32;
            ret[1] = r.1 as i32;
            ret[2] = r.2 as i32;
            ret[3] = r.3 as i32;
        } else {
            // Stretch to the buffer by default
            ret[0] = 0;
            ret[1] = 0;
            ret[2] = desc.size.0 as i32;
            ret[3] = desc.size.1 as i32;
        }
        ret
    }

    pub fn get_kernel_flags(&self, frame: usize, buffers: &Buffers) -> KernelParamsFlags {
        let mut kernel_flags = self.kernel_flags.clone();
        kernel_flags.set(KernelParamsFlags::HAS_DIGITAL_LENS, self.compute_params.digital_lens.is_some());
        kernel_flags.set(KernelParamsFlags::HORIZONTAL_RS, self.compute_params.frame_readout_direction.is_horizontal());
        kernel_flags.set(KernelParamsFlags::HAS_SOURCE_RECT, buffers.input.rect.is_some() || self.size.0 != buffers.input.size.0 || self.size.1 != buffers.input.size.1);
        kernel_flags.set(KernelParamsFlags::HAS_OUTPUT_RECT, buffers.output.rect.is_some() || self.output_size.0 != buffers.output.size.0 || self.output_size.1 != buffers.output.size.1);
        kernel_flags.set(KernelParamsFlags::FRAMEBUFFER_INVERTED, self.compute_params.framebuffer_inverted);
        kernel_flags.set(KernelParamsFlags::ANY_UNDERWATER, (self.compute_params.light_refraction_coefficient != 1.0 && self.compute_params.light_refraction_coefficient > 0.0) || self.compute_params.keyframes.is_keyframed(&crate::KeyframeType::LightRefractionCoeff));

        // Colour flags must live here, not only in `FrameTransform::at_timestamp`, for two reasons:
        // 1. `get_frame_transform_at` *assigns* (not ORs) `get_kernel_flags(..).bits()` over
        //    `kernel_params.flags`, so anything set only in `at_timestamp` is discarded.
        // 2. `get_current_key` derives the GPU-instance cache key from these flags. Without the
        //    colour bits, toggling colour on or off produces an identical key, so the cached wgpu
        //    instance and its already-specialized pipeline (`flags` is an `@id(103) override`)
        //    are reused and the colour stage never switches on or off.
        // The condition mirrors `frame_transform.rs` exactly (`color.is_active()`, and HAS_LUT
        // only when a LUT is present); `at_timestamp` keeps setting them for its direct callers,
        // which have no `Stabilization` to ask, and the two paths agree by construction.
        let color = &self.compute_params.color;
        let color_active = color.enabled && (color.is_active() || self.compute_params.keyframes.is_any_color_keyframed());
        kernel_flags.set(KernelParamsFlags::COLOR_ENABLED, color_active);
        kernel_flags.set(KernelParamsFlags::HAS_LUT, color_active && self.compute_params.color.lut.is_some());

        {
            let gyro = self.compute_params.gyro.read();
            let file_metadata = gyro.file_metadata.read();
            if let Some(mc) = file_metadata.mesh_correction.get(frame) {
                if mc.1[0] > 10.0 {
                    kernel_flags.set(KernelParamsFlags::HAS_MESH_DATA, true);
                }
                if mc.1[0] > 0.0 && mc.1[mc.1[0] as usize] > 0.0 {
                    kernel_flags.set(KernelParamsFlags::HAS_FPD_DATA, true);
                }
            }
            if file_metadata.camera_stab_data.len() > frame {
                kernel_flags.set(KernelParamsFlags::HAS_IBIS_DATA, true);
            }
        }
        kernel_flags
    }

    pub fn get_frame_transform_at<T: PixelType>(&self, timestamp_us: i64, frame: Option<usize>, buffers: &Buffers) -> FrameTransform {
        let timestamp_ms = (timestamp_us as f64) / 1000.0;
        let frame = frame.unwrap_or_else(|| crate::frame_at_timestamp(timestamp_ms, self.compute_params.scaled_fps) as usize);

        let mut transform = FrameTransform::at_timestamp(&self.compute_params, timestamp_ms, frame);
        transform.kernel_params.pixel_value_limit = T::default_max_value().unwrap_or(f32::MAX);
        transform.kernel_params.max_pixel_value = T::default_max_value().unwrap_or(1.0);
        // If the pixel format gets converted to normalized 0-1 float in shader
        if self.initialized_backend.is_wgpu() && T::wgpu_format().map(|x| x.2).unwrap_or_default() {
            transform.kernel_params.pixel_value_limit = 1.0;
            transform.kernel_params.max_pixel_value = 1.0;
        }
        transform.kernel_params.interpolation = self.interpolation as i32;
        transform.kernel_params.width  = self.size.0 as i32;
        transform.kernel_params.height = self.size.1 as i32;
        transform.kernel_params.output_width  = self.output_size.0 as i32;
        transform.kernel_params.output_height = self.output_size.1 as i32;
        transform.kernel_params.background = [self.compute_params.background[0], self.compute_params.background[1], self.compute_params.background[2], self.compute_params.background[3]];
        transform.kernel_params.bytes_per_pixel = (T::COUNT * T::SCALAR_BYTES) as i32;
        transform.kernel_params.pix_element_count = T::COUNT as i32;
        transform.kernel_params.canvas_scale = self.drawing.scale as f32;
        transform.kernel_params.flags = self.get_kernel_flags(frame, buffers).bits();

        transform.kernel_params.stride        = buffers.input.size.2 as i32;
        transform.kernel_params.output_stride = buffers.output.size.2 as i32;

        if transform.kernel_params.interpolation > 8 {
            let (b, c) = match self.interpolation {
                Interpolation::RobidouxSharp => (0.2620145, 0.3689927),
                Interpolation::Robidoux      => (0.3782157, 0.3108921),
                Interpolation::Mitchell      => (0.3333333, 0.3333333),
                Interpolation::CatmullRom    => (0.0000000, 0.5000000),
                _ => (0.0, 0.0)
            };
            transform.kernel_params.ewa_coeffs_p[0] = (6.0 - 2.0 * b) / 6.0;
            transform.kernel_params.ewa_coeffs_p[1] = 0.0;
            transform.kernel_params.ewa_coeffs_p[2] = (-18.0 + 12.0 * b + 6.0 * c) / 6.0;
            transform.kernel_params.ewa_coeffs_p[3] = (12.0 - 9.0 * b - 6.0 * c) / 6.0;
            transform.kernel_params.ewa_coeffs_q[0] = (8.0 * b + 24.0 * c) / 6.0;
            transform.kernel_params.ewa_coeffs_q[1] = (-12.0 * b - 48.0 * c) / 6.0;
            transform.kernel_params.ewa_coeffs_q[2] = (6.0 * b + 30.0 * c) / 6.0;
            transform.kernel_params.ewa_coeffs_q[3] = (-1.0 * b - 6.0 * c) / 6.0;
        }

        let sa_fov =
            if self.compute_params.show_safe_area || self.compute_params.fov_overview  {
                let fov = self.compute_params.keyframes.value_at_video_timestamp(&crate::keyframes::KeyframeType::Fov, timestamp_ms).unwrap_or(self.compute_params.fov_scale) as f32;
                if self.compute_params.fov_overview {
                    (if self.compute_params.adaptive_zoom_window == 0.0 { 1.0 } else { 1.0 / fov }) + 1.0
                } else {
                    fov / (if self.compute_params.adaptive_zoom_window == 0.0 { transform.minimal_fov as f32 } else { 1.0 })
                }
            } else {
                1.0
            };
        let pos_x = (transform.kernel_params.output_width as f32 - (transform.kernel_params.output_width as f32 / sa_fov)) / 2.0;
        let pos_y = (transform.kernel_params.output_height as f32 - (transform.kernel_params.output_height as f32 / sa_fov)) / 2.0;
        transform.kernel_params.safe_area_rect[0] = pos_x;
        transform.kernel_params.safe_area_rect[1] = pos_y;
        transform.kernel_params.safe_area_rect[2] = transform.kernel_params.output_width as f32 - pos_x;
        transform.kernel_params.safe_area_rect[3] = transform.kernel_params.output_height as f32 - pos_y;

        if let Some(r) = buffers.input.rotation {
            transform.kernel_params.input_rotation = r;
        }
        if let Some(r) = buffers.output.rotation {
            transform.kernel_params.output_rotation = r;
        }

        transform.kernel_params.source_rect = Self::get_rect(&buffers.input);
        transform.kernel_params.output_rect = Self::get_rect(&buffers.output);

        transform
    }

    pub fn ensure_stab_data_at_timestamp<T: PixelType>(&mut self, timestamp_us: i64, frame: Option<usize>, buffers: &mut Buffers, is_pixel_normalized: bool) {
        let mut insert = true;
        if let Some(itm) = self.stab_data.get(&timestamp_us) {
            insert = false;
            if itm.kernel_params.stride        != buffers.input.size.2 as i32 ||
               itm.kernel_params.output_stride != buffers.output.size.2 as i32 {
                log::warn!("Stride mismatch ({} != {} || {} != {})", itm.kernel_params.stride, buffers.input.size.2, itm.kernel_params.output_stride, buffers.output.size.2);
                insert = true;
            }
            if itm.kernel_params.input_rotation != buffers.input.rotation.unwrap_or(0.0) ||
               itm.kernel_params.output_rotation != buffers.output.rotation.unwrap_or(0.0) ||
               itm.kernel_params.source_rect != Self::get_rect(&buffers.input) ||
               itm.kernel_params.output_rect != Self::get_rect(&buffers.output) {
                log::warn!("Updating stab params at {timestamp_us}");
                insert = true;
            }
        }
        if insert {
            let mut transform = self.get_frame_transform_at::<T>(timestamp_us, frame, buffers);
            if is_pixel_normalized {
                transform.kernel_params.max_pixel_value = 1.0;
                transform.kernel_params.pixel_value_limit = 1.0;
            }
            self.stab_data.insert(timestamp_us, transform);
        }
    }

    pub fn get_current_key(&self, buffers: &Buffers) -> String {
        let mut flags = self.get_kernel_flags(0, buffers);
        flags.set(KernelParamsFlags::FILL_WITH_BACKGROUND, false);
        format!(
            "{}|{}|{}|{}|{}|{:?}|{:?}|{:?}|{:?}|{}",
            buffers.get_checksum(),
            self.compute_params.distortion_model.id(),
            self.compute_params.digital_lens.as_ref().map(|x| x.id()).unwrap_or_default(),
            self.interpolation as u32,
            flags.bits(),
            self.size,
            self.output_size,
            self.interpolation,
            std::thread::current().id(),
            // A LUT change alters the size of the coeffs buffer every backend uploads, so it
            // must be part of the cache key: otherwise a differently-sized LUT could reuse a
            // cached GPU instance and write past the end of its (smaller) coeffs buffer.
            self.compute_params.color.lut.as_ref().map(|l| l.checksum).unwrap_or(0),
        )
    }
    pub fn get_current_checksum(&self, buffers: &Buffers) -> u32 {
        crc32fast::hash(self.get_current_key(buffers).as_bytes())
    }

    pub fn init_size(&mut self, size: (usize, usize), output_size: (usize, usize)) {
        self.initialized_backend = BackendType::None;
        #[cfg(feature = "use-opencl")]
        { self.cl = None; }
        self.wgpu = None;

        self.size = size;
        self.output_size = output_size;

        if self.kernel_flags.contains(KernelParamsFlags::DRAWING_ENABLED) {
            self.drawing = DrawCanvas::new(size.0, size.1, output_size.0, output_size.1, (size.1 as f64 / 720.0).max(1.0) as usize);
        }

        self.stab_data.clear();
    }

    pub fn clear_stab_data(&mut self) {
        self.stab_data.clear();
    }

    pub fn get_undistortion_data(&self, timestamp_us: i64) -> Option<&FrameTransform> {
        self.stab_data.get(&timestamp_us)
    }

    pub fn list_devices(&self) -> Vec<String> {
        let mut ret = Vec::new();

        #[cfg(feature = "use-opencl")]
        if std::env::var("NO_OPENCL").unwrap_or_default().is_empty() {
            ret.extend(opencl::OclWrapper::list_devices().into_iter().map(|x| format!("[OpenCL] {x}")));
        }
        if std::env::var("NO_WGPU").unwrap_or_default().is_empty() {
            ret.extend(wgpu::WgpuWrapper::list_devices().into_iter().map(|x| format!("[wgpu] {x}")));
        }
        ret
    }

    pub fn set_device(&mut self, i: isize) {
        self.pending_device_change = Some(i);
    }

    pub fn update_device(&mut self, i: isize, buffers: &Buffers) -> bool {
        self.stab_data.clear();
        self.next_backend = None;
        self.initialized_backend = BackendType::None;
        #[cfg(feature = "use-opencl")]
        { self.cl = None; }
        self.wgpu = None;

        let hash = self.get_current_checksum(buffers);
        if i < 0 { // CPU
            CACHED_WGPU.with(|x| x.0.borrow_mut().clear());
            #[cfg(feature = "use-opencl")] {
                CACHED_OPENCL.with(|x| x.borrow_mut().clear());
            }
            self.initialized_backend = BackendType::Cpu(hash);
            return true;
        }
        let gpu_list = GPU_LIST.read();
        if let Some(name) = gpu_list.get(i as usize) {
            if name.starts_with("[OpenCL]") {
                self.initialized_backend = BackendType::None;
                #[cfg(feature = "use-opencl")]
                match opencl::OclWrapper::set_device(i as usize, buffers) {
                    Ok(_) => {
                        self.next_backend = Some("opencl");
                        return true;
                    },
                    Err(e) => {
                        log::error!("Failed to set OpenCL device {}: {:?}", name, e);
                    }
                }
            } else if name.starts_with("[wgpu]") {
                self.initialized_backend = BackendType::None;
                let first_ind = gpu_list.iter().enumerate().find(|(_, m)| m.starts_with("[wgpu]")).map(|(idx, _)| idx).unwrap_or(0);
                let wgpu_ind = i - first_ind as isize;
                if wgpu_ind >= 0 {
                    match wgpu::WgpuWrapper::set_device(wgpu_ind as usize) {
                        Some(_) => {
                            self.next_backend = Some("wgpu");
                            return true;
                        },
                        None => {
                            log::error!("Failed to set wgpu device {}", name);
                        }
                    }
                }
            }
        }
        false
    }

    pub fn init_backends<T: PixelType>(&mut self, timestamp_us: i64, frame: Option<usize>, buffers: &Buffers) {
        let hash = self.get_current_checksum(buffers);
        let current_hash = self.initialized_backend.get_hash();

        if current_hash != hash {
            self.initialized_backend = BackendType::None;
            let canvas_len = self.drawing.get_buffer_len();
            #[allow(unused_mut)]
            let mut next_backend = self.next_backend.take().unwrap_or_default();

            #[cfg(feature = "use-opencl")]
            if std::env::var("NO_OPENCL").unwrap_or_default().is_empty() && next_backend != "wgpu" && opencl::is_buffer_supported(buffers) {
                if self.share_wgpu_instances && CACHED_OPENCL.with(|x| x.borrow().contains(&hash)) {
                    self.cl = None;
                    self.initialized_backend = BackendType::OpenCL(hash);
                } else {
                    self.cl = None;
                    let transform = self.get_frame_transform_at::<T>(timestamp_us, frame, buffers);
                    let params = transform.kernel_params;
                    let distortion_model = self.compute_params.distortion_model.clone();
                    let digital_lens = self.compute_params.digital_lens.clone();
                    let lut = self.compute_params.color.lut.clone();
                    let cl = std::panic::catch_unwind(|| {
                        opencl::OclWrapper::new(&params, T::ocl_names(), distortion_model, digital_lens, buffers, canvas_len, lut)
                    });
                    match cl {
                        Ok(Ok(cl)) => {
                            if self.share_wgpu_instances {
                                CACHED_OPENCL.with(|x| x.borrow_mut().put(hash, cl));
                            } else {
                                self.cl = Some(cl);
                            }
                            self.initialized_backend = BackendType::OpenCL(hash);
                            log::info!("Initialized OpenCL for {:?} -> {:?}, key: {}", buffers.input.size, buffers.output.size, self.get_current_key(buffers));
                            CACHED_WGPU.with(|x| x.0.borrow_mut().clear());
                        },
                        Ok(Err(e)) => {
                            next_backend = "";
                            log::error!("OpenCL error init_backends: {:?}", e);
                            if self.share_wgpu_instances { CACHED_OPENCL.with(|x| x.borrow_mut().clear()) }
                        },
                        Err(e) => {
                            next_backend = "";
                            if let Some(s) = e.downcast_ref::<&str>() {
                                log::error!("Failed to initialize OpenCL {}", s);
                            } else if let Some(s) = e.downcast_ref::<String>() {
                                log::error!("Failed to initialize OpenCL {}", s);
                            } else {
                                log::error!("Failed to initialize OpenCL {:?}", e);
                            }
                            if self.share_wgpu_instances { CACHED_OPENCL.with(|x| x.borrow_mut().clear()); }
                        }
                    }
                }
            }
            if self.initialized_backend.is_none() && T::wgpu_format().is_some() && next_backend != "opencl" && std::env::var("NO_WGPU").unwrap_or_default().is_empty() && wgpu::is_buffer_supported(buffers) {
                if self.share_wgpu_instances && CACHED_WGPU.with(|x| x.0.borrow().contains(&hash)) {
                    self.wgpu = None;
                    self.initialized_backend = BackendType::Wgpu(hash);
                } else {
                    self.wgpu = None;
                    let transform = self.get_frame_transform_at::<T>(timestamp_us, frame, buffers);
                    let params = transform.kernel_params;
                    let distortion_model = self.compute_params.distortion_model.clone();
                    let digital_lens = self.compute_params.digital_lens.clone();
                    let lut = self.compute_params.color.lut.clone();
                    let wgpu = std::panic::catch_unwind(|| {
                        wgpu::WgpuWrapper::new(&params, T::wgpu_format().unwrap(), distortion_model, digital_lens, buffers, canvas_len, lut)
                    });
                    match wgpu {
                        Ok(Ok(wgpu)) => {
                            if self.share_wgpu_instances {
                                CACHED_WGPU.with(|x| x.0.borrow_mut().put(hash, wgpu));
                            } else {
                                self.wgpu = Some(wgpu);
                            }
                            self.initialized_backend = BackendType::Wgpu(hash);
                            log::info!("Initialized wgpu for {:?} -> {:?} | key: {}", buffers.input.size, buffers.output.size, self.get_current_key(buffers));
                            #[cfg(feature = "use-opencl")] {
                                CACHED_OPENCL.with(|x| x.borrow_mut().clear());
                            }
                        },
                        Ok(Err(e)) => {
                            log::error!("Failed to initialize wgpu {:?}", e);
                            if self.share_wgpu_instances { CACHED_WGPU.with(|x| x.0.borrow_mut().clear()) }
                        },
                        Err(e) => {
                            if let Some(s) = e.downcast_ref::<&str>() {
                                log::error!("Failed to initialize wgpu {}", s);
                            } else if let Some(s) = e.downcast_ref::<String>() {
                                log::error!("Failed to initialize wgpu {}", s);
                            } else {
                                log::error!("Failed to initialize wgpu {:?}", e);
                            }
                            if self.share_wgpu_instances { CACHED_WGPU.with(|x| x.0.borrow_mut().clear()); }
                        },
                    }
                }
            }
        }
    }

    pub fn ensure_ready_for_processing<T: PixelType>(&mut self, timestamp_us: i64, frame: Option<usize>, buffers: &mut Buffers) {
        let pending_dev = self.pending_device_change.clone();
        if let Some(dev) = self.pending_device_change.take() {
            log::debug!("Setting device {dev}");
            self.update_device(dev, buffers);
        }

        self.init_backends::<T>(timestamp_us, frame, buffers);
        self.ensure_stab_data_at_timestamp::<T>(timestamp_us, frame, buffers, false);

        if self.share_wgpu_instances {
            if wgpu::is_buffer_supported(buffers) && CACHED_WGPU.with(|x| !x.0.borrow().is_empty()) {
                let hash = self.get_current_checksum(buffers);
                let has_cached = CACHED_WGPU.with(|x| x.0.borrow().contains(&hash));
                if !has_cached {
                    log::warn!("Cached wgpu not found, reinitializing. Key: {}", self.get_current_key(buffers));
                    self.initialized_backend = BackendType::None;
                    if let Some(dev) = pending_dev {
                        log::debug!("Setting device {dev}");
                        self.update_device(dev, buffers);
                    }
                    self.init_backends::<T>(timestamp_us, frame, buffers);
                } else {
                    self.initialized_backend = BackendType::Wgpu(hash);
                }
            } else {
                #[cfg(feature = "use-opencl")]
                if opencl::is_buffer_supported(buffers) && CACHED_OPENCL.with(|x| !x.borrow().is_empty()) {
                    let hash = self.get_current_checksum(buffers);
                    let has_cached = CACHED_OPENCL.with(|x| x.borrow().contains(&hash));
                    if !has_cached {
                        log::warn!("Cached OpenCL not found, reinitializing. Key: {}", self.get_current_key(buffers));
                        self.initialized_backend = BackendType::None;
                        if let Some(dev) = pending_dev {
                            log::debug!("Setting device {dev}");
                            self.update_device(dev, buffers);
                        }
                        self.init_backends::<T>(timestamp_us, frame, buffers);
                    } else {
                        self.initialized_backend = BackendType::OpenCL(hash);
                    }
                }
            }
        }
    }
    pub fn process_pixels<T: PixelType>(&self, timestamp_us: i64, frame: Option<usize>, buffers: &mut Buffers, frame_transform: Option<&FrameTransform>) -> Result<ProcessedInfo, GyroflowCoreError> {
        if buffers.input.size.1 < 4 || buffers.output.size.1 < 4 { return Err(GyroflowCoreError::SizeTooSmall); }

        let mut _tmp_transform = None;
        if frame_transform.is_none() && !self.cache_frame_transform {
            _tmp_transform = Some(self.get_frame_transform_at::<T>(timestamp_us, frame, buffers));
        }
        let itm = frame_transform.map(|x| Some(x)).unwrap_or_else(||
            if !self.cache_frame_transform {
                _tmp_transform.as_ref()
            } else {
                self.stab_data.get(&timestamp_us)
            }
        );

        if let Some(itm) = itm {
            let mut ret = ProcessedInfo {
                fov: itm.fov,
                minimal_fov: itm.minimal_fov,
                focal_length: itm.focal_length,
                backend: ""
            };
            let drawing_buffer = self.drawing.get_buffer();

            if self.size        != (itm.kernel_params.width        as usize, itm.kernel_params.height        as usize) { return Err(GyroflowCoreError::SizeMismatch(self.size, (itm.kernel_params.width        as usize, itm.kernel_params.height        as usize))); }
            if self.output_size != (itm.kernel_params.output_width as usize, itm.kernel_params.output_height as usize) { return Err(GyroflowCoreError::SizeMismatch(self.size, (itm.kernel_params.output_width as usize, itm.kernel_params.output_height as usize))); }

            if buffers.input.size.0  as i32 > itm.kernel_params.stride        { return Err(GyroflowCoreError::InvalidStride(itm.kernel_params.stride, buffers.input.size.0 as i32)); }
            if buffers.output.size.0 as i32 > itm.kernel_params.output_stride { return Err(GyroflowCoreError::InvalidStride(itm.kernel_params.output_stride, buffers.output.size.0 as i32)); }

            // OpenCL path
            #[cfg(feature = "use-opencl")]
            if !matches!(self.initialized_backend, BackendType::Cpu(_)) && opencl::is_buffer_supported(buffers) {
                if self.share_wgpu_instances {
                    let hash = self.get_current_checksum(buffers);
                    let has_cache = CACHED_OPENCL.with(|lru| lru.borrow().contains(&hash));
                    if has_cache {
                        return CACHED_OPENCL.with(|x| {
                            let mut cached = x.borrow_mut();
                            if let Some(cl) = cached.get(&hash) {
                                if let Err(err) = cl.undistort_image(buffers, &itm, drawing_buffer) {
                                    log::error!("OpenCL error undistort: {:?}", err);
                                }
                                ret.backend = "OpenCL";
                                Ok(ret)
                            } else {
                                Err(GyroflowCoreError::NoCachedWgpuInstance(self.get_current_key(buffers)))
                            }
                        });
                    }
                } else {
                    if let Some(ref cl) = self.cl {
                        if let Err(err) = cl.undistort_image(buffers, &itm, drawing_buffer) {
                            log::error!("OpenCL error undistort: {:?}", err);
                        } else {
                            ret.backend = "OpenCL";
                            return Ok(ret);
                        }
                    }
                }
            }

            // wgpu path
            if !matches!(self.initialized_backend, BackendType::Cpu(_)) && wgpu::is_buffer_supported(buffers) {
                if self.share_wgpu_instances {
                    let hash = self.get_current_checksum(buffers);
                    let has_any_cache = CACHED_WGPU.with(|x| !x.0.borrow().is_empty());
                    if has_any_cache {
                        return CACHED_WGPU.with(|x| {
                            let mut cached = x.0.borrow_mut();
                            if let Some(wgpu) = cached.get(&hash) {
                                wgpu.undistort_image(buffers, &itm, drawing_buffer);
                                ret.backend = "wgpu";
                                Ok(ret)
                            } else {
                                Err(GyroflowCoreError::NoCachedWgpuInstance(self.get_current_key(buffers)))
                            }
                        });
                    } else {
                        log::error!("No cached wgpu found for key: {}", self.get_current_key(buffers));
                    }
                } else {
                    if let Some(ref wgpu) = self.wgpu {
                        wgpu.undistort_image(buffers, &itm, drawing_buffer);
                        ret.backend = "wgpu";
                        return Ok(ret);
                    } else {
                        log::error!("No wgpu instance!");
                    }
                }
            }

            //let ok = Self::undistort_image_cpu_spirv::<T>(buffers, &itm.kernel_params, &self.compute_params.distortion_model, self.compute_params.digital_lens.as_ref(), &itm.matrices, drawing_buffer);
            // CPU path
            // Only build the combined coeffs+LUT buffer (and allocate) when colour is actually
            // enabled for this frame; otherwise reuse the static COEFFS table with no extra
            // allocation. `enabled` and `lut` are independent fields (Task 15's UI and Task 13's
            // project deserialisation both let a LUT be selected while colour is toggled off), so
            // gating on `lut.is_some()` alone would allocate on every frame even when
            // COLOR_ENABLED is unset and the pixel path is a no-op. Gate on the flag actually
            // consulted by the pixel path instead.
            let color_enabled = (itm.kernel_params.flags & KernelParamsFlags::COLOR_ENABLED.bits()) != 0;
            let coeffs_buf;
            let coeffs: &[f32] = if color_enabled {
                if let Some(lut) = self.compute_params.color.lut.as_deref() {
                    coeffs_buf = build_coeffs_buffer(Some(lut));
                    &coeffs_buf
                } else {
                    &COEFFS
                }
            } else {
                &COEFFS
            };
            let ok = match self.interpolation {
                Interpolation::Bilinear => { Self::undistort_image_cpu::<2, T>(buffers, &itm.kernel_params, &self.compute_params.distortion_model, self.compute_params.digital_lens.as_ref(), &itm.matrices, drawing_buffer, &itm.mesh_data, coeffs) },
                Interpolation::Bicubic  => { Self::undistort_image_cpu::<4, T>(buffers, &itm.kernel_params, &self.compute_params.distortion_model, self.compute_params.digital_lens.as_ref(), &itm.matrices, drawing_buffer, &itm.mesh_data, coeffs) },
                Interpolation::Lanczos4 => { Self::undistort_image_cpu::<8, T>(buffers, &itm.kernel_params, &self.compute_params.distortion_model, self.compute_params.digital_lens.as_ref(), &itm.matrices, drawing_buffer, &itm.mesh_data, coeffs) },
                Interpolation::RobidouxSharp => { Self::undistort_image_cpu::<10, T>(buffers, &itm.kernel_params, &self.compute_params.distortion_model, self.compute_params.digital_lens.as_ref(), &itm.matrices, drawing_buffer, &itm.mesh_data, coeffs) },
                Interpolation::Robidoux      => { Self::undistort_image_cpu::<11, T>(buffers, &itm.kernel_params, &self.compute_params.distortion_model, self.compute_params.digital_lens.as_ref(), &itm.matrices, drawing_buffer, &itm.mesh_data, coeffs) },
                Interpolation::Mitchell      => { Self::undistort_image_cpu::<12, T>(buffers, &itm.kernel_params, &self.compute_params.distortion_model, self.compute_params.digital_lens.as_ref(), &itm.matrices, drawing_buffer, &itm.mesh_data, coeffs) },
                Interpolation::CatmullRom    => { Self::undistort_image_cpu::<13, T>(buffers, &itm.kernel_params, &self.compute_params.distortion_model, self.compute_params.digital_lens.as_ref(), &itm.matrices, drawing_buffer, &itm.mesh_data, coeffs) },
            };
            if ok {
                ret.backend = "CPU";
                return Ok(ret);
            }
        } else {
            log::warn!("No stab data at {timestamp_us}");
            return Err(GyroflowCoreError::NoStabilizationData(timestamp_us));
        }
        Err(GyroflowCoreError::Unknown)
    }
}

unsafe impl Send for Stabilization { }

#[cfg(test)]
mod coeffs_buffer_tests {
    use super::*;

    #[test]
    fn offset_matches_coeffs_length() {
        assert_eq!(LUT_DATA_OFFSET, COEFFS.len());
    }

    #[test]
    fn buffer_without_lut_is_just_coeffs() {
        assert_eq!(build_coeffs_buffer(None).len(), COEFFS.len());
    }

    #[test]
    fn lut_data_starts_at_the_offset() {
        let lut = crate::color::Lut3D { size: 2, domain_min: 0.0, domain_max: 1.0, data: (0..24).map(|i| i as f32).collect(), checksum: 0 };
        let buf = build_coeffs_buffer(Some(&lut));
        assert_eq!(buf.len(), LUT_DATA_OFFSET + 24);
        assert_eq!(buf[LUT_DATA_OFFSET], 0.0);
        assert_eq!(buf[LUT_DATA_OFFSET + 23], 23.0);
    }
}

/// The GPU-instance cache key must change when colour is toggled or a LUT is added/removed,
/// otherwise a wgpu pipeline specialized on the old `flags` value is silently reused.
#[cfg(test)]
mod color_cache_key_tests {
    use super::*;

    fn buffers() -> Buffers<'static> {
        Buffers { input: BufferDescription::default(), output: BufferDescription::default() }
    }

    fn stab_with(color: crate::color::ColorAdjustments) -> Stabilization {
        let mut stab = Stabilization::default();
        let mut params = ComputeParams::default();
        params.color = color;
        stab.set_compute_params(params);
        stab
    }

    fn lut() -> std::sync::Arc<crate::color::Lut3D> {
        std::sync::Arc::new(crate::color::Lut3D { size: 2, domain_min: 0.0, domain_max: 1.0, data: vec![0.5; 24], checksum: 12345 })
    }

    #[test]
    fn toggling_enabled_changes_the_key() {
        let mut off = crate::color::ColorAdjustments::default();
        off.exposure = 1.0; // something for `is_active` to see
        let mut on = off.clone();
        on.enabled = true;

        let b = buffers();
        assert_ne!(stab_with(off).get_current_key(&b), stab_with(on).get_current_key(&b));
    }

    #[test]
    fn adding_or_removing_a_lut_changes_the_key() {
        let mut without = crate::color::ColorAdjustments::default();
        without.enabled = true;
        let mut with = without.clone();
        with.lut = Some(lut());

        let b = buffers();
        assert_ne!(stab_with(without).get_current_key(&b), stab_with(with).get_current_key(&b));
    }

    #[test]
    fn untouched_colour_leaves_the_flags_clear() {
        // No colour usage must not perturb the key for everyone else.
        let flags = stab_with(crate::color::ColorAdjustments::default()).get_kernel_flags(0, &buffers());
        assert!(!flags.contains(KernelParamsFlags::COLOR_ENABLED));
        assert!(!flags.contains(KernelParamsFlags::HAS_LUT));
    }

    #[test]
    fn kernel_flags_reflect_colour_state() {
        let mut c = crate::color::ColorAdjustments::default();
        c.enabled = true;
        c.lut = Some(lut());
        let flags = stab_with(c).get_kernel_flags(0, &buffers());
        assert!(flags.contains(KernelParamsFlags::COLOR_ENABLED));
        assert!(flags.contains(KernelParamsFlags::HAS_LUT));
    }
}

/// Cross-backend colour equivalence.
///
/// The point of this module is to compare what each backend's colour stage *does to a
/// pixel*, and nothing else. That is only meaningful if the geometry is an exact identity:
/// the backends disagree substantially on undistortion and Lanczos resampling — measured at
/// roughly 75x the tolerance asserted here on a real 4K frame with colour disabled — so any
/// comparison that lets geometry vary measures that pre-existing disagreement instead, and
/// says nothing about colour. See the spec's "Cross-backend equivalence" section.
///
/// The identity setup (k=0 distortion, f=[1,1], c=[0,0], `lens_correction_amount` 1.0,
/// 1:1 rects, identity matrix) is the same one
/// `color_stage_applied_end_to_end_through_undistort_image_cpu` established for the CPU
/// path, and `identity_geometry_is_bit_exact_on_every_backend` re-proves it here per
/// backend before any colour claim is made. If that control ever fails, every tolerance in
/// this module becomes meaningless, so it asserts bit-exact equality rather than a bound.
///
/// GPU-dependent, so `#[ignore]`d: CI runners have no usable GPU. Run explicitly with
/// `cargo test -- --ignored`. A backend that is unavailable at runtime is reported and
/// skipped, never silently passed.
#[cfg(test)]
mod cross_backend_color_tests {
    use super::*;
    use crate::color::{ apply_color, ColorKernelParams, Lut3D };
    use super::distortion_models::DistortionModel;
    use std::sync::Arc;

    // wgpu refuses anything smaller than 16 wide / 4 high (`WgpuError::ParamCheck`,
    // `gpu/wgpu.rs:169`), so this is the smallest frame every backend will accept.
    const W: i32 = 16;
    const H: i32 = 4;
    const N: usize = (W * H) as usize;
    const BPP: i32 = 16; // RGBAf: 4 x f32
    const STRIDE: i32 = W * BPP;

    /// Max per-channel difference allowed between a backend and the `apply_color`
    /// reference, as a fraction of full scale. From the spec's testing section.
    const TOLERANCE: f32 = 1.0 / 1024.0;

    const COLOR_PARAMS_1: [f32; 4] = [1.2, 1.1, 0.8, 0.5]; // exposure, contrast, saturation, lut_strength
    const COLOR_PARAMS_2: [f32; 4] = [1.05, 1.0, 0.95, 2.0]; // wb r/g/b, lut_size
    const COLOR_PARAMS_3: [f32; 4] = [0.0, 1.0, 0.0, 0.0]; // domain_min, domain_max

    /// Inverts every channel; bilinear interpolation of a linear function is exact, so
    /// `sample_trilinear` returns exactly `[1-c0, 1-c1, 1-c2]`. Same LUT the CPU test uses.
    fn inverting_lut() -> Lut3D {
        Lut3D {
            size: 2,
            domain_min: 0.0,
            domain_max: 1.0,
            data: vec![
                1.0, 1.0, 1.0,   0.0, 1.0, 1.0,   1.0, 0.0, 1.0,   0.0, 0.0, 1.0,
                1.0, 1.0, 0.0,   0.0, 1.0, 0.0,   1.0, 0.0, 0.0,   0.0, 0.0, 0.0,
            ],
            checksum: 0,
        }
    }

    /// 64 distinct, non-trivial pixels spanning the 0..1 range on each channel, with the
    /// channels deliberately out of phase so no two are equal and a backend cannot pass by
    /// coincidence. Values stay strictly inside 0..1 so nothing is decided by clamping.
    fn input_pixels() -> Vec<[f32; 4]> {
        (0..N).map(|i| {
            let t = i as f32 / (N - 1) as f32; // 0..=1
            [
                0.02 + 0.96 * t,
                0.02 + 0.96 * ((t * 3.0) % 1.0),
                0.02 + 0.96 * ((t * 7.0 + 0.31) % 1.0),
                1.0,
            ]
        }).collect()
    }

    fn input_bytes() -> Vec<u8> {
        let mut b = vec![0u8; (STRIDE * H) as usize];
        for (i, p) in input_pixels().iter().enumerate() {
            b[i * 16..i * 16 + 16].copy_from_slice(bytemuck::bytes_of(p));
        }
        b
    }

    fn identity_transform(flags: i32) -> FrameTransform {
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
        p.lens_correction_amount = 1.0; // skips the lens-correction block entirely
        p.source_rect = [0, 0, W, H];
        p.output_rect = [0, 0, W, H];
        // Must cover the whole frame, and leaving it at the default [0,0,0,0] is not neutral.
        // `draw_safe_area` in the wgpu/OpenCL/SPIR-V kernels is *not* behind a flag: it runs on
        // every pixel and multiplies rgb by 0.5 outside the rect, then by 0.5 again within 5px
        // of its edge. At [0,0,0,0] every pixel is "outside", and every pixel with x < 5 is also
        // "border", which is exactly the 1/2-and-1/4 attenuation this harness was measuring and
        // misreading as a geometric divergence. Production sets this rect from the output size
        // (see `undistort_image`), so the darkening is inert there - only a hand-built
        // KernelParams like this one can trip it. The CPU path has no safe-area code at all.
        p.safe_area_rect = [0.0, 0.0, W as f32, H as f32];
        p.max_pixel_value = 1.0;
        p.pixel_value_limit = f32::MAX;
        p.light_refraction_coefficient = 1.0;
        p.flags = flags;
        p.color_params_1 = COLOR_PARAMS_1;
        p.color_params_2 = COLOR_PARAMS_2;
        p.color_params_3 = COLOR_PARAMS_3;

        FrameTransform {
            // Identity 3x3 in the first 9 fields; the trailing 5 stay zero, which disables
            // the extra shift/angle adjustment.
            matrices: vec![[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            kernel_params: p,
            fov: 1.0,
            minimal_fov: 1.0,
            focal_length: None,
            mesh_data: vec![],
        }
    }

    fn read_pixels(buf: &[u8]) -> Vec<[f32; 4]> {
        (0..N).map(|i| *bytemuck::from_bytes::<[f32; 4]>(&buf[i * 16..i * 16 + 16])).collect()
    }

    /// What the colour stage is supposed to produce, from the Rust reference every kernel
    /// was hand-ported from. Comparing backends only against each other would let a shared
    /// misreading of the params pass, so each backend is checked against this.
    fn reference() -> Vec<[f32; 4]> {
        let lut = inverting_lut();
        let cp = ColorKernelParams {
            exposure_gain:     COLOR_PARAMS_1[0],
            contrast_factor:   COLOR_PARAMS_1[1],
            saturation_factor: COLOR_PARAMS_1[2],
            lut_strength:      COLOR_PARAMS_1[3],
            wb_gains:          [COLOR_PARAMS_2[0], COLOR_PARAMS_2[1], COLOR_PARAMS_2[2]],
            lut_size:          COLOR_PARAMS_2[3] as u32,
            domain_min:        COLOR_PARAMS_3[0],
            domain_max:        COLOR_PARAMS_3[1],
        };
        input_pixels().iter().map(|p| {
            let out = apply_color([p[0], p[1], p[2]], &cp, Some(&lut));
            [out[0], out[1], out[2], p[3]]
        }).collect()
    }

    /// Runs the identity transform through wgpu.
    ///
    /// `Err` distinguishes "wgpu is genuinely unavailable on this machine", which is a
    /// legitimate skip, from "wgpu is present but the run failed", which is a failure and
    /// must not be mistaken for a skip. Collapsing the two is how a test ends up asserting
    /// nothing while looking green.
    ///
    /// **The half-pixel correction is essential and is not a fudge.** The CPU path feeds the
    /// transform the integer pixel index (`cpu_undistort.rs:597`, `Vector2::new(x, y)`),
    /// while the wgpu *fragment* path feeds `@builtin(position)`, which is the pixel centre
    /// `(x + 0.5, y + 0.5)`. Verified directly by instrumenting the shader to emit
    /// `@builtin(position)` and `undistort_coord`: the coordinate maths is an exact identity
    /// on both, and the only *geometric* difference is that half pixel — at `uv = 0.5` the sampler
    /// lands on the half-pixel coefficient row and blends two texels, where the CPU lands
    /// exactly on one.
    ///
    /// `translation2d` is added to `out_pos` at the top of `undistort_coord`
    /// (`wgpu_undistort.wgsl:561`), so -0.5 puts both backends on the same sampling grid.
    /// Without it this test would be comparing two different resamplings and could say
    /// nothing about colour. See the note on the backend divergence in the design spec: the
    /// offset itself is a real, pre-existing, non-colour inconsistency between backends.
    ///
    /// The offset was necessary but *not sufficient*, and on its own it left the identity
    /// control still failing. The rest of the original divergence was not geometric at all:
    /// it was `draw_safe_area` attenuating rgb because `identity_transform` left
    /// `safe_area_rect` at its default — see the note there. The two were easy to conflate
    /// because the sampling-map probes decode a position *from channel values*, so a
    /// multiplicative scale on the values is indistinguishable from one on the position. If
    /// this control ever regresses, rule out value scaling before re-deriving the geometry.
    fn run_wgpu(flags: i32, lut: Option<Arc<Lut3D>>) -> Result<Vec<[f32; 4]>, String> {
        if crate::gpu::wgpu::WgpuWrapper::initialize_context().is_none() {
            return Err("UNAVAILABLE: no usable wgpu adapter".into());
        }

        let mut itm = identity_transform(flags);
        itm.kernel_params.translation2d = [-0.5, -0.5];
        let mut input = input_bytes();
        let mut output = vec![0u8; (STRIDE * H) as usize];
        let mut buffers = Buffers {
            input:  BufferDescription { size: (W as usize, H as usize, STRIDE as usize), rect: None, rotation: None, data: BufferSource::Cpu { buffer: &mut input }, texture_copy: false },
            output: BufferDescription { size: (W as usize, H as usize, STRIDE as usize), rect: None, rotation: None, data: BufferSource::Cpu { buffer: &mut output }, texture_copy: false },
        };

        let inst = crate::gpu::wgpu::WgpuWrapper::new(
            &itm.kernel_params,
            RGBAf::wgpu_format().unwrap(),
            DistortionModel::default(),
            None,
            &buffers,
            0,
            lut,
        ).map_err(|e| format!("WgpuWrapper::new failed: {e:?}"))?;

        if !inst.undistort_image(&mut buffers, &itm, &[]) {
            return Err("undistort_image returned false".into());
        }
        drop(buffers);
        Ok(read_pixels(&output))
    }

    /// Same identity setup through the CPU pixel path, for comparison.
    fn run_cpu(flags: i32) -> Vec<[f32; 4]> {
        let itm = identity_transform(flags);
        let coeffs = crate::stabilization::build_coeffs_buffer(Some(&inverting_lut()));
        let mut input = input_bytes();
        let mut output = vec![0u8; (STRIDE * H) as usize];
        {
            let mut buffers = Buffers {
                input:  BufferDescription { size: (W as usize, H as usize, STRIDE as usize), rect: None, rotation: None, data: BufferSource::Cpu { buffer: &mut input }, texture_copy: false },
                output: BufferDescription { size: (W as usize, H as usize, STRIDE as usize), rect: None, rotation: None, data: BufferSource::Cpu { buffer: &mut output }, texture_copy: false },
            };
            let ok = Stabilization::undistort_image_cpu::<2, RGBAf>(&mut buffers, &itm.kernel_params, &DistortionModel::default(), None, &itm.matrices, &[], &[], &coeffs);
            assert!(ok, "undistort_image_cpu returned false");
        }
        read_pixels(&output)
    }

    /// Turns a backend result into either usable output, or a skip, or a hard failure.
    /// Only a genuine "UNAVAILABLE" is allowed to skip.
    fn resolve(name: &str, r: Result<Vec<[f32; 4]>, String>) -> Option<Vec<[f32; 4]>> {
        match r {
            Ok(v) => Some(v),
            Err(e) if e.starts_with("UNAVAILABLE") => { println!("SKIPPED {name}: {e}"); None }
            Err(e) => panic!("{name} is available but the run failed: {e}"),
        }
    }

    /// The CPU half of the control, and the reason the wgpu half is known to be a real
    /// divergence rather than a bad test fixture: the *same* `KernelParams` and the *same*
    /// identity matrix produce an exactly bit-exact pass-through here at 16x4, 0/64 pixels
    /// differing. This is not an artifact of the 2x2 frame the older CPU test used.
    ///
    /// Not `#[ignore]`d - it needs no GPU, and it is what pins the fixture as correct.
    #[test]
    fn identity_geometry_is_bit_exact_on_cpu() {
        let out = run_cpu(0);
        for (i, expected) in input_pixels().iter().enumerate() {
            assert_eq!(out[i], *expected, "CPU: identity geometry is not bit-exact at pixel {i}");
        }
    }

    /// Diagnostic, not an assertion: what wgpu can see, and how its identity output actually
    /// differs. Kept because it is the entry point for finishing
    /// `identity_geometry_is_bit_exact_on_every_backend` - see that test's note.
    #[test]
    #[ignore = "diagnostic; run with: cargo test -- --ignored"]
    fn wgpu_identity_divergence_diagnostic() {
        println!("wgpu adapters: {:?}", crate::gpu::wgpu::WgpuWrapper::list_devices());
        match run_wgpu(0, None) {
            Ok(out) => {
                let inp = input_pixels();
                let bad = (0..N).filter(|&i| out[i] != inp[i]).count();
                println!("wgpu identity at {W}x{H}: {bad}/{N} pixels differ from input");
                for i in 0..6.min(N) {
                    println!("  pixel {i}: in={:?} out={:?}", inp[i], out[i]);
                }
            }
            Err(e) => println!("wgpu run did not produce output: {e}"),
        }
    }

    /// Same probe in `RGBA8`. `RGBAf` is 16 bytes per pixel and is flagged non-unorm in
    /// `wgpu_format`, while `RGBA8`/`RGBA16` are 4/8 bytes and unorm — and the shader is
    /// generated per format. `RGBAf` is also a format the render path never produces (float
    /// sources are refused, see OQ1), so if the divergence is `RGBAf`-specific the fixture
    /// was testing a configuration that cannot occur.
    #[test]
    #[ignore = "diagnostic; run with: cargo test -- --ignored"]
    fn wgpu_sampling_map_probe_rgba8() {
        const BPP8: i32 = 4;
        const STRIDE8: i32 = W * BPP8;

        // red ramps across x, green ramps down y, both 0..=255.
        let mut input = vec![0u8; (STRIDE8 * H) as usize];
        for i in 0..N {
            let x = i % W as usize;
            let y = i / W as usize;
            let o = i * 4;
            input[o]     = (x * 255 / (W as usize - 1)) as u8;
            input[o + 1] = (y * 255 / (H as usize - 1)) as u8;
            input[o + 2] = 0;
            input[o + 3] = 255;
        }

        let mut itm = identity_transform(0);
        itm.kernel_params.stride = STRIDE8;
        itm.kernel_params.output_stride = STRIDE8;
        itm.kernel_params.bytes_per_pixel = BPP8;
        itm.kernel_params.max_pixel_value = 255.0;

        let mut inp_buf = input.clone();
        let mut output = vec![0u8; (STRIDE8 * H) as usize];
        {
            let mut buffers = Buffers {
                input:  BufferDescription { size: (W as usize, H as usize, STRIDE8 as usize), rect: None, rotation: None, data: BufferSource::Cpu { buffer: &mut inp_buf }, texture_copy: false },
                output: BufferDescription { size: (W as usize, H as usize, STRIDE8 as usize), rect: None, rotation: None, data: BufferSource::Cpu { buffer: &mut output }, texture_copy: false },
            };
            if crate::gpu::wgpu::WgpuWrapper::initialize_context().is_none() { println!("SKIPPED: no wgpu adapter"); return; }
            let inst = match crate::gpu::wgpu::WgpuWrapper::new(&itm.kernel_params, RGBA8::wgpu_format().unwrap(), DistortionModel::default(), None, &buffers, 0, None) {
                Ok(i) => i,
                Err(e) => { println!("WgpuWrapper::new failed: {e:?}"); return; }
            };
            if !inst.undistort_image(&mut buffers, &itm, &[]) { println!("undistort_image false"); return; }
        }

        let mut exact = 0;
        println!("RGBA8: out (x,y) -> decoded src (col,row)");
        for i in 0..N {
            let x = i % W as usize;
            let y = i / W as usize;
            let col = output[i * 4]     as f32 * (W - 1) as f32 / 255.0;
            let row = output[i * 4 + 1] as f32 * (H - 1) as f32 / 255.0;
            if output[i * 4] == input[i * 4] && output[i * 4 + 1] == input[i * 4 + 1] { exact += 1; }
            if i < 6 || x == 0 { println!("  {i:2} ({x:2},{y}) -> ({col:6.3},{row:6.3})"); }
        }
        println!("RGBA8 exact matches: {exact}/{N}");
    }

    /// Evidence-gathering, not a fix: the input's red channel encodes its own linear index
    /// (`red = i / (N-1)`), so decoding the output's red channel says exactly which input
    /// position each output pixel was sampled from. Guessing at the ratio from a handful of
    /// values was not converging; this shows the mapping directly.
    #[test]
    #[ignore = "diagnostic; run with: cargo test -- --ignored"]
    fn wgpu_sampling_map_probe() {
        // red = i/(N-1) exactly; green/blue mark the row and column so a transposed or
        // wrapped read is visible rather than being decoded as some fractional index.
        let mut input = vec![0u8; (STRIDE * H) as usize];
        for i in 0..N {
            let x = (i % W as usize) as f32;
            let y = (i / W as usize) as f32;
            let p: [f32; 4] = [i as f32 / (N - 1) as f32, x / W as f32, y / H as f32, 1.0];
            input[i * 16..i * 16 + 16].copy_from_slice(bytemuck::bytes_of(&p));
        }

        let mut itm = identity_transform(0);
        itm.kernel_params.translation2d = [-0.5, -0.5];
        let mut inp_buf = input.clone();
        let mut output = vec![0u8; (STRIDE * H) as usize];
        {
            let mut buffers = Buffers {
                input:  BufferDescription { size: (W as usize, H as usize, STRIDE as usize), rect: None, rotation: None, data: BufferSource::Cpu { buffer: &mut inp_buf }, texture_copy: false },
                output: BufferDescription { size: (W as usize, H as usize, STRIDE as usize), rect: None, rotation: None, data: BufferSource::Cpu { buffer: &mut output }, texture_copy: false },
            };
            if crate::gpu::wgpu::WgpuWrapper::initialize_context().is_none() {
                println!("SKIPPED: no wgpu adapter"); return;
            }
            let inst = match crate::gpu::wgpu::WgpuWrapper::new(&itm.kernel_params, RGBAf::wgpu_format().unwrap(), DistortionModel::default(), None, &buffers, 0, None) {
                Ok(i) => i,
                Err(e) => { println!("WgpuWrapper::new failed: {e:?}"); return; }
            };
            if !inst.undistort_image(&mut buffers, &itm, &[]) { println!("undistort_image false"); return; }
        }

        // The same frame through the CPU path, so the two maps can be compared directly
        // rather than one being compared against an assumption.
        let coeffs = crate::stabilization::build_coeffs_buffer(None);
        let mut cpu_in = input.clone();
        let mut cpu_out = vec![0u8; (STRIDE * H) as usize];
        {
            let mut b = Buffers {
                input:  BufferDescription { size: (W as usize, H as usize, STRIDE as usize), rect: None, rotation: None, data: BufferSource::Cpu { buffer: &mut cpu_in }, texture_copy: false },
                output: BufferDescription { size: (W as usize, H as usize, STRIDE as usize), rect: None, rotation: None, data: BufferSource::Cpu { buffer: &mut cpu_out }, texture_copy: false },
            };
            // The CPU path must NOT get the wgpu half-pixel correction - it already feeds
            // integer pixel coordinates. Applying it here would shift the CPU instead.
            let cpu_itm = identity_transform(0);
            assert!(Stabilization::undistort_image_cpu::<2, RGBAf>(&mut b, &cpu_itm.kernel_params, &DistortionModel::default(), None, &cpu_itm.matrices, &[], &[], &coeffs));
        }

        let out = read_pixels(&output);
        let cpu = read_pixels(&cpu_out);
        println!("out (x,y) -> RAW wgpu channels [r,g,b,a] | cpu src (col,row)");
        for i in 0..N {
            let x = i % W as usize;
            let y = i / W as usize;
            let (cc, cr) = (cpu[i][1] * W as f32, cpu[i][2] * H as f32);
            let (gc, gr) = (out[i][1] * W as f32, out[i][2] * H as f32);
            println!("  {i:2} ({x:2},{y}) -> wgpu({gc:7.3},{gr:6.3}) | cpu({cc:6.2},{cr:5.2})");
        }
    }

    /// The control. Every tolerance below is only meaningful while this passes, which is why
    /// it demands bit-exact equality with the input rather than a bound.
    ///
    /// **This does not pass yet, and the gap is deliberate rather than unnoticed.** With
    /// byte-identical `KernelParams` and the same identity matrix, the CPU path is exactly
    /// bit-exact (`identity_geometry_is_bit_exact_on_cpu`, which does pass) while wgpu
    /// resamples: its output is a smooth ramp at roughly a quarter of the input's rate.
    /// Ruled out **with evidence**, by `wgpu_sampling_map_probe` (which decodes, per output
    /// pixel, exactly which input position each backend sampled):
    ///
    /// - *Not* the pixel format or bytes-per-pixel. `RGBA8` (4 bytes, unorm) diverges the
    ///   same way as `RGBAf` (16 bytes, non-unorm) — see `wgpu_sampling_map_probe_rgba8`.
    /// - *Not* frame size. The divergence is unchanged when only `H` is doubled.
    /// - *Not* lens distortion. `KernelParams::default()` leaves `k` zero, and the injected
    ///   `opencv_fisheye.wgsl:49` early-outs to the identity for all-zero `k`, as do the
    ///   OpenCL and GLSL copies. Only the rust-gpu copy
    ///   (`stabilize_spirv/src/distortion_models/opencv_fisheye.rs:71`) lacks that early-out,
    ///   and it is not on this path. The WGSL params struct layout was checked against
    ///   `KernelParams` field by field, so `k1` really is zero rather than misaligned.
    /// - *Not* the output-rect flag (bit 64, unset), the lens-correction branch (skipped at
    ///   `lens_correction_amount` 1.0), or `canvas_scale`.
    ///
    /// **Found and corrected: a half-pixel offset.** Instrumenting the shader to emit
    /// `@builtin(position)` and `undistort_coord` directly showed the coordinate maths is an
    /// exact identity on both backends, and that the fragment path (which is what runs for a
    /// `BufferSource::Cpu` buffer, since `init_texture` makes `uses_textures` true) is handed
    /// the pixel *centre* `(x + 0.5, y + 0.5)` while the CPU path is handed the integer index
    /// `(x, y)` (`cpu_undistort.rs:597`). `run_wgpu` cancels it via `translation2d`, which
    /// made the **y axis exactly correct** and fixed pixel 0.
    ///
    /// **Residual, still unexplained:** with y exact, x is compressed 4x — output x of
    /// 0,1,2,3,4 samples input x of 0.00,0.25,0.50,0.75,1.00, then breaks pattern at x=5.
    /// A factor of 4 on x alone, with y unaffected, is the shape of a stride/element-count
    /// error rather than a geometric one, and 4 is exactly
    /// `bytes_per_pixel / pix_element_count` for this format — but the texture branch of
    /// `read_input_at` uses `textureLoad` and never touches those. Worth checking next
    /// whether the `interpolation`, `bytes_per_pixel` and `pix_element_count` pipeline
    /// override constants actually arrive with the values passed to `WgpuWrapper::new`.
    ///
    /// Until it passes, no cross-backend colour claim can be made - which is exactly what
    /// this test existing and failing is for. Run it with `cargo test -- --ignored`.
    #[test]
    #[ignore = "needs a GPU; run with: cargo test -- --ignored"]
    fn identity_geometry_is_bit_exact_on_every_backend() {
        let mut ran_any = false;

        if let Some(out) = resolve("wgpu", run_wgpu(0, None)) {
            ran_any = true;
            for (i, expected) in input_pixels().iter().enumerate() {
                assert_eq!(out[i], *expected,
                    "wgpu: identity geometry is not bit-exact at pixel {i} - every colour \
                     assertion in this module is invalid until this passes");
            }
        }

        assert!(ran_any, "no backend was available, so this test asserted nothing");
    }

    /// **Read `identity_geometry_is_bit_exact_on_every_backend` first.** It is the control:
    /// while it fails, any number this test produces is measuring wgpu's resampling rather
    /// than its colour stage, and must be treated as unread. It currently passes bit-exactly,
    /// so the figure below is a genuine measurement of the colour kernel.
    #[test]
    #[ignore = "needs a GPU; run with: cargo test -- --ignored"]
    fn color_stage_matches_the_reference_on_every_backend() {
        let expected = reference();
        let lut = Some(Arc::new(inverting_lut()));
        let mut ran_any = false;

        // COLOR_ENABLED | HAS_LUT
        let flags = (KernelParamsFlags::COLOR_ENABLED | KernelParamsFlags::HAS_LUT).bits();

        if let Some(out) = resolve("wgpu", run_wgpu(flags, lut.clone())) {
            ran_any = true;
            let mut max_diff = 0.0f32;
            for (i, exp) in expected.iter().enumerate() {
                for c in 0..3 {
                    max_diff = max_diff.max((out[i][c] - exp[c]).abs());
                }
            }
            println!("wgpu: max per-channel difference from apply_color = {max_diff} (1/{:.0} scale)", 1.0 / max_diff.max(f32::EPSILON));
            assert!(max_diff < TOLERANCE,
                "wgpu colour stage differs from the apply_color reference by {max_diff}, \
                 over the {TOLERANCE} tolerance. Report this, do not patch the kernel.");
        }

        assert!(ran_any, "no backend was available, so this test asserted nothing");
    }
}
unsafe impl Sync for Stabilization { }
