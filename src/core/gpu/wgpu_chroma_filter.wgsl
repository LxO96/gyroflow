// Copyright © 2026 Gyroflow
// Chroma smoothing pass. Mirror of `denoise::chroma_filter` (Rust) and opencl_chroma_filter.cl.
// SCALAR (f32 or u32) and CF_ROUND (round, or nothing for f32) are substituted by wgpu.rs.

struct ChromaFilterParams {
    width: i32, height: i32, stride: i32, channels: i32,
    guide_width: i32, guide_height: i32, guide_stride: i32, mode: i32,
    amount: f32, sigma_y: f32, max_value: f32, pad: f32,
}

@group(0) @binding(0) var<uniform> p: ChromaFilterParams;
@group(0) @binding(1) var src: texture_2d<SCALAR>;
@group(0) @binding(2) var guide: texture_2d<SCALAR>;

@vertex
fn cf_vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 4>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0));
    return vec4<f32>(pos[i], 0.0, 1.0);
}

fn cf_load(x: i32, y: i32) -> vec4<f32> {
    let c = vec2<i32>(clamp(x, 0, p.width - 1), clamp(y, 0, p.height - 1));
    return vec4<f32>(textureLoad(src, c, 0)) / p.max_value;
}
// Rounded: an odd-sized plane has a chroma plane of ceil(n/2), and 3841 / 1921 must be 2.
fn cf_ratio(g: i32, n: i32) -> i32 { return max(1, (g + n / 2) / n); }
fn cf_guide(x0: i32, y0: i32) -> f32 {
    let x = clamp(x0, 0, p.width - 1);
    let y = clamp(y0, 0, p.height - 1);
    let rx = cf_ratio(p.guide_width, p.width);
    let ry = cf_ratio(p.guide_height, p.height);
    var s = 0.0;
    for (var j = 0; j < ry; j = j + 1) {
        for (var i = 0; i < rx; i = i + 1) {
            let g = vec2<i32>(min(x * rx + i, p.guide_width - 1), min(y * ry + j, p.guide_height - 1));
            s = s + f32(textureLoad(guide, g, 0).x);
        }
    }
    return s / f32(rx * ry) / p.max_value;
}
fn cf_luma(v: vec4<f32>) -> f32 { return 0.2126 * v.x + 0.7152 * v.y + 0.0722 * v.z; }

@fragment
fn cf_fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<SCALAR> {
    let x = i32(position.x);
    let y = i32(position.y);
    let rgb = p.mode == 2;
    let c0 = cf_load(x, y);
    var g0 = 0.0;
    if (rgb) { g0 = cf_luma(c0); } else { g0 = cf_guide(x, y); }
    var v0 = c0;
    if (rgb) { v0 = c0 - vec4<f32>(g0, g0, g0, 0.0); }

    let inv_s = 1.0 / (2.0 * 4.0 * 4.0);
    let inv_y = 1.0 / (2.0 * p.sigma_y * p.sigma_y);
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var dy = -8; dy <= 8; dy = dy + 2) {
        for (var dx = -8; dx <= 8; dx = dx + 2) {
            let c = cf_load(x + dx, y + dy);
            var g = 0.0;
            if (rgb) { g = cf_luma(c); } else { g = cf_guide(x + dx, y + dy); }
            var v = c;
            if (rgb) { v = c - vec4<f32>(g, g, g, 0.0); }
            let w = exp(-f32(dx * dx + dy * dy) * inv_s - (g - g0) * (g - g0) * inv_y);
            acc = acc + v * w;
            wsum = wsum + w;
        }
    }
    var out = v0 + p.amount * (acc / wsum - v0);
    if (rgb) { out = vec4<f32>(out.x + g0, out.y + g0, out.z + g0, c0.w); }
    out = out * p.max_value;
    return vec4<SCALAR>(CF_ROUND(out));
}
