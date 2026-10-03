// Copyright © 2026 Gyroflow

//! The film chain's per-frame uniform block. Mirrored field for field by `FilmParams` in
//! `gpu/opencl_film.cl` and `gpu/wgpu_film.wgsl`: 80 bytes of scalars, then 16-byte rows.

pub const FLAG_TONE: u32 = 1;
pub const FLAG_GRAIN: u32 = 2;
pub const FLAG_HALATION: u32 = 4;
pub const FLAG_BW: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FilmParams {
    /// Filled by the backend from the buffers it is given.
    pub width: i32, pub height: i32, pub stride: i32, pub flags: u32,
    pub halo_f: i32, pub halo_w: i32, pub halo_h: i32, pub halo_radius: i32,
    pub grain_radius: i32, pub max_value: f32, pub grain_k: f32, pub grain_sigma: f32,
    pub halo_sigma: f32, pub halo_strength: f32, pub inv_p: f32, pub fit_b: f32,
    pub fit_s: f32, pub fit_umin_a: f32, pub _pad0: f32, pub _pad1: f32,
    pub seeds: [u32; 4],
    pub halo_tint: [f32; 4],
    pub log_e0: [f32; 4],
    pub k: [f32; 4],
    pub u_min: [f32; 4],
    pub range: [f32; 4],
    pub sens: [[f32; 4]; 3],
    pub cross: [[f32; 4]; 3],
    /// `std(u)` at `u = i / 200`, i in 0..=200; the rest is padding (WGSL wants vec4 rows).
    pub std_table: [f32; 256],
}

impl Default for FilmParams { fn default() -> Self { bytemuck::Zeroable::zeroed() } }

impl FilmParams {
    pub fn is_active(&self) -> bool { self.flags & (FLAG_TONE | FLAG_GRAIN | FLAG_HALATION) != 0 }
}
