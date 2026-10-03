// Copyright © 2026 Gyroflow
// Film chain. Mirror of `film::reference` (Rust) and opencl_film.cl.
// SCALAR (f32 or u32) and FILM_ROUND (round, or nothing for f32) are substituted by wgpu.rs.

struct FilmParams {
    width: i32, height: i32, stride: i32, flags: u32,
    halo_f: i32, halo_w: i32, halo_h: i32, halo_radius: i32,
    grain_radius: i32, max_value: f32, grain_k: f32, grain_sigma: f32,
    halo_sigma: f32, halo_strength: f32, inv_p: f32, fit_b: f32,
    fit_s: f32, fit_umin_a: f32, pad0: f32, pad1: f32,
    seeds: vec4<u32>,
    halo_tint: vec4<f32>, log_e0: vec4<f32>, k: vec4<f32>, u_min: vec4<f32>, range: vec4<f32>,
    sens: array<vec4<f32>, 3>,
    cross: array<vec4<f32>, 3>,
    std_table: array<vec4<f32>, 64>,
}

@group(0) @binding(0) var<uniform> p: FilmParams;
@group(0) @binding(1) var src: texture_2d<SCALAR>;
@group(0) @binding(2) var aux_a: texture_2d<f32>;
@group(0) @binding(3) var aux_b: texture_2d<f32>;

const FILM_TONE: u32 = 1u;
const FILM_GRAIN: u32 = 2u;
const FILM_HALO: u32 = 4u;
const FILM_BW: u32 = 8u;

@vertex
fn film_vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 4>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0));
    return vec4<f32>(pos[i], 0.0, 1.0);
}

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn hash4(a: u32, b: u32, c: u32, d: u32) -> u32 { return pcg(a ^ pcg(b ^ pcg(c ^ pcg(d)))); }
fn film_gauss(h: u32) -> f32 {
    let h1 = pcg(h);
    let h2 = pcg(h1);
    let u1 = max(f32(h1 >> 8u) / 16777216.0, 1.0 / 16777216.0);
    let u2 = f32(h2 >> 8u) / 16777216.0;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}
fn dec(v: f32) -> f32 { if (v <= 0.04045) { return v / 12.92; } return pow((v + 0.055) / 1.055, 2.4); }
fn enc(x: f32) -> f32 { if (x <= 0.0031308) { return x * 12.92; } return 1.055 * pow(x, 1.0 / 2.4) - 0.055; }
fn sstep(a: f32, b: f32, x: f32) -> f32 { let t = clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
fn gw(k: i32, s: f32) -> f32 { return exp(-f32(k * k) / (2.0 * s * s)); }
fn export_px(x: i32) -> u32 { return u32(i32(floor((f32(x) + 0.5) * p.inv_p))); }
fn load_px(x: i32, y: i32) -> vec4<f32> { return vec4<f32>(textureLoad(src, vec2<i32>(x, y), 0)) / p.max_value; }

// H1
@fragment
fn film_halo_source(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let i = i32(pos.x); let j = i32(pos.y); let f = p.halo_f;
    var s = 0.0;
    for (var b = 0; b < f; b = b + 1) {
        for (var a = 0; a < f; a = a + 1) {
            let c = load_px(min(i * f + a, p.width - 1), min(j * f + b, p.height - 1));
            let y = 0.2126 * dec(c.x) + 0.7152 * dec(c.y) + 0.0722 * dec(c.z);
            s = s + sstep(0.75, 1.0, y) * y;
        }
    }
    return vec4<f32>(s / f32(f * f), 0.0, 0.0, 0.0);
}

fn halo_blur(pos: vec4<f32>, horizontal: bool) -> vec4<f32> {
    let i = i32(pos.x); let j = i32(pos.y); let r = p.halo_radius;
    var s = 0.0; var ws = 0.0;
    for (var t = -r; t <= r; t = t + 1) {
        let w = gw(t, p.halo_sigma);
        var c = vec2<i32>(i, clamp(j + t, 0, p.halo_h - 1));
        if (horizontal) { c = vec2<i32>(clamp(i + t, 0, p.halo_w - 1), j); }
        s = s + w * textureLoad(aux_a, c, 0).x;
        ws = ws + w;
    }
    return vec4<f32>(s / ws, 0.0, 0.0, 0.0);
}
// H2, H3
@fragment fn film_halo_blur_x(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> { return halo_blur(pos, true); }
@fragment fn film_halo_blur_y(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> { return halo_blur(pos, false); }

// G1
@fragment
fn film_noise_blur_x(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = i32(pos.x); let y = i32(pos.y); let r = p.grain_radius;
    var layers = 3;
    if ((p.flags & FILM_BW) != 0u) { layers = 1; }
    let ey = export_px(y);
    var n = vec3<f32>(0.0); var ws = 0.0;
    for (var t = -r; t <= r; t = t + 1) {
        let w = gw(t, p.grain_sigma);
        let ex = export_px(clamp(x + t, 0, p.width - 1));
        for (var l = 0; l < layers; l = l + 1) {
            n[l] = n[l] + w * film_gauss(hash4(ex, ey, p.seeds[l], 0x3C6EF372u));
        }
        ws = ws + w;
    }
    return vec4<f32>(n / ws, 0.0);
}

fn sample_halo(hx: f32, hy: f32) -> f32 {
    let fx = hx - 0.5; let fy = hy - 0.5;
    let x0 = i32(floor(fx)); let y0 = i32(floor(fy));
    let tx = fx - f32(x0); let ty = fy - f32(y0);
    let xa = clamp(x0, 0, p.halo_w - 1); let xb = clamp(x0 + 1, 0, p.halo_w - 1);
    let ya = clamp(y0, 0, p.halo_h - 1); let yb = clamp(y0 + 1, 0, p.halo_h - 1);
    let h = aux_b;
    return (textureLoad(h, vec2<i32>(xa, ya), 0).x * (1.0 - tx) + textureLoad(h, vec2<i32>(xb, ya), 0).x * tx) * (1.0 - ty)
         + (textureLoad(h, vec2<i32>(xa, yb), 0).x * (1.0 - tx) + textureLoad(h, vec2<i32>(xb, yb), 0).x * tx) * ty;
}

// G2: aux_a = G1 noise, aux_b = halo field.
@fragment
fn film_apply(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<SCALAR> {
    let x = i32(pos.x); let y = i32(pos.y);
    let c = load_px(x, y);
    var lin = vec3<f32>(dec(c.x), dec(c.y), dec(c.z));
    if ((p.flags & FILM_HALO) != 0u) {
        let f = f32(p.halo_f);
        lin = lin + p.halo_strength * sample_halo((f32(x) + 0.5) / f, (f32(y) + 0.5) / f) * p.halo_tint.xyz;
    }
    var u = vec3<f32>(0.0);
    for (var i = 0; i < 3; i = i + 1) {
        if ((p.flags & FILM_TONE) != 0u) {
            let e = dot(p.sens[i].xyz, lin);
            let log_e = log(max(e, 1e-5)) / log(10.0) + p.fit_b;
            let d = p.u_min[i] + p.range[i] / (1.0 + exp(-p.k[i] * (log_e - p.log_e0[i])));
            u[i] = p.fit_umin_a + p.fit_s * (d - p.fit_umin_a);
        } else {
            u[i] = enc(max(lin[i], 0.0));
        }
    }
    if ((p.flags & FILM_GRAIN) != 0u) {
        let r = p.grain_radius;
        var n = vec4<f32>(0.0); var ws = 0.0;
        for (var t = -r; t <= r; t = t + 1) {
            let w = gw(t, p.grain_sigma);
            n = n + w * textureLoad(aux_a, vec2<i32>(x, clamp(y + t, 0, p.height - 1)), 0);
            ws = ws + w;
        }
        n = n / ws;
        let bw = (p.flags & FILM_BW) != 0u;
        for (var i = 0; i < 3; i = i + 1) {
            // Linear between table entries; mirrors `reference::std_of`.
            let fu = clamp(u[i], 0.0, 1.0) * 200.0;
            let i0 = min(i32(floor(fu)), 199);
            let i1 = i0 + 1;
            let tt = fu - f32(i0);
            let sd = p.std_table[i0 / 4][i0 % 4] * (1.0 - tt) + p.std_table[i1 / 4][i1 % 4] * tt;
            var nl = n[i];
            if (bw) { nl = n.x; }
            u[i] = clamp(u[i] + p.grain_k * sd * nl, 0.0, 1.0);
        }
    }
    var o = clamp(u, vec3<f32>(0.0), vec3<f32>(1.0));
    if ((p.flags & FILM_TONE) != 0u) {
        o = clamp(vec3<f32>(dot(p.cross[0].xyz, u), dot(p.cross[1].xyz, u), dot(p.cross[2].xyz, u)), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    return vec4<SCALAR>(FILM_ROUND(vec4<f32>(o, c.w) * p.max_value));
}
