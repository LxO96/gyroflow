// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2021-2022 Adrian <adrian.eddy at gmail>

// Adapted from OpenCV: initUndistortRectifyMap
// https://github.com/opencv/opencv/blob/2b60166e5c65f1caccac11964ad760d847c536e4/modules/calib3d/src/fisheye.cpp#L465-L567

#version 420

layout(location = 0) in vec2 v_texcoord;
layout(location = 0) out vec4 fragColor;

layout(binding = 1) uniform sampler2D texIn;

layout(std140, binding = 2) uniform KernelParams {
    int width;             // 4
    int height;            // 8
    int stride;            // 12
    int output_width;      // 16
    int output_height;     // 4
    int output_stride;     // 8
    int matrix_count;      // 12 - for rolling shutter correction. 1 = no correction, only main matrix
    int interpolation;     // 16
    int background_mode;   // 4
    int flags;             // 8
    int bytes_per_pixel;   // 12
    int pix_element_count; // 16
    vec4 background;    // 16
    vec2 f;             // 8  - focal length in pixels
    vec2 c;             // 16 - lens center
    vec4 k1, k2, k3;    // 16, 16, 16 - distortion coefficients
    float fov;          // 4
    float r_limit;      // 8
    float lens_correction_amount;   // 12
    float input_vertical_stretch;   // 16
    float input_horizontal_stretch; // 4
    float background_margin;        // 8
    float background_margin_feather;// 12
    float canvas_scale;             // 16
    float input_rotation;           // 4
    float output_rotation;          // 8
    vec2 translation2d;             // 16
    vec4 translation3d;             // 16
    ivec4 source_rect;              // 16 - x, y, w, h - unused in this kernel
    ivec4 output_rect;              // 16 - x, y, w, h - unused in this kernel
    vec4 digital_lens_params[4];    // 16,16,16,16
    vec4 safe_area_rect;            // 16
    float max_pixel_value;          // 4
    int distortion_model;           // 8
    int digital_lens;               // 12
    float pixel_value_limit;        // 16
    float light_refraction_coefficient; // 4
    int plane_index;                // 8
    float reserved1;                // 12
    float reserved2;                // 16
    vec4 ewa_coefs_p;               // 16
    vec4 ewa_coefs_q;               // 16
    vec4 color_params_1;            // 16 - exposure_gain, contrast_factor, saturation_factor, lut_strength
    vec4 color_params_2;            // 16 - wb_gain_r, wb_gain_g, wb_gain_b, lut_size
    vec4 color_params_3;            // 16 - lut_domain_min, lut_domain_max, unused, unused
} params;

LENS_MODEL_FUNCTIONS;

layout(binding = 3) uniform sampler2D texParams;
layout(binding = 4) uniform sampler2D texCanvas;
layout(binding = 5) uniform sampler2D texMeshData;
layout(binding = 6) uniform sampler2D texLut;

////////////////////////////// Colour adjustments //////////////////////////////
// This arithmetic is a deliberate, ratified duplicate of `crate::color::apply_color`
// (src/core/color/mod.rs) and `crate::color::sample_trilinear` (src/core/color/lut.rs).
// Every operation, and their order, must stay identical to the Rust reference - the
// explicit `a + (b - a) * t` lerps below mirror the Rust source exactly and must not be
// rewritten as `mix()`, which evaluates `a * (1 - t) + b * t` and is not bit-identical.
// Likewise the luma sum must stay an explicit left-associated sum, not `dot()`, which may
// contract to an FMA chain with different rounding.
//
// Unlike the OpenCL and wgpu kernels, this backend does not carry the LUT in the coeffs
// buffer - it has none. The LUT is a 2D-tiled RGBA32F texture instead: the `size³` entries
// are uploaded verbatim in .cube order (red fastest, then green, then blue) into an image
// `size*size` wide and `size` tall. A row-major image `n*n` wide puts texel (x, y) at linear
// entry `y*n*n + x`, and `sample_trilinear` wants entry `b*n*n + g*n + r` (lut.rs:107), so
// the blue index selects the ROW and (g, r) select the column: entry (r, g, b) lives at
// texel `(g*size + r, b)`.
//
// Note this is not the (b*size + r, g) addressing one might expect from a "blue selects the
// tile column" layout - that would need the upload to be reordered, and it is not.
//
// The sampler is Nearest/ClampToEdge and every read goes through texelFetch, because the
// trilinear interpolation below is done by hand - any hardware filtering would make the
// preview diverge from the export.
vec3 lerp3(vec3 a, vec3 b, float t) {
    return a + (b - a) * t;
}

vec3 lut_fetch(int r, int g, int b, int n) {
    int rc = clamp(r, 0, n - 1);
    int gc = clamp(g, 0, n - 1);
    int bc = clamp(b, 0, n - 1);
    // Belt and braces: apply_color() already refuses to call this unless the texture size
    // agrees with `n`, so the min() below never bites. It stays because an out-of-range
    // texelFetch is undefined behaviour and this is the only place that could produce one.
    ivec2 sz = textureSize(texLut, 0);
    return texelFetch(texLut, min(ivec2(gc * n + rc, bc), sz - ivec2(1)), 0).rgb;
}

vec3 lut_sample(vec3 rgb, int n, float domain_min, float domain_max) {
    float scale = 1.0 / max(domain_max - domain_min, 1e-9);
    vec3 normalized = clamp((rgb - vec3(domain_min)) * scale, vec3(0.0), vec3(1.0));
    vec3 pos = normalized * float(n - 1);
    ivec3 base = clamp(ivec3(floor(pos)), ivec3(0), ivec3(n - 1));
    vec3 f = pos - vec3(base);

    vec3 c000 = lut_fetch(base.x,     base.y,     base.z,     n);
    vec3 c100 = lut_fetch(base.x + 1, base.y,     base.z,     n);
    vec3 c010 = lut_fetch(base.x,     base.y + 1, base.z,     n);
    vec3 c110 = lut_fetch(base.x + 1, base.y + 1, base.z,     n);
    vec3 c001 = lut_fetch(base.x,     base.y,     base.z + 1, n);
    vec3 c101 = lut_fetch(base.x + 1, base.y,     base.z + 1, n);
    vec3 c011 = lut_fetch(base.x,     base.y + 1, base.z + 1, n);
    vec3 c111 = lut_fetch(base.x + 1, base.y + 1, base.z + 1, n);

    vec3 c00 = lerp3(c000, c100, f.x);
    vec3 c10 = lerp3(c010, c110, f.x);
    vec3 c01 = lerp3(c001, c101, f.x);
    vec3 c11 = lerp3(c011, c111, f.x);
    vec3 c0  = lerp3(c00, c10, f.y);
    vec3 c1  = lerp3(c01, c11, f.y);
    return lerp3(c0, c1, f.z);
}

vec4 apply_color(vec4 px) {
    float max_value = params.max_pixel_value;
    float inv_max = max_value > 0.0 ? 1.0 / max_value : 1.0;
    vec3 c = px.rgb * inv_max;

    // 1. LUT, blended by strength
    int lut_size = int(params.color_params_2.w);
    float lut_strength = params.color_params_1.w;
    // `lut_size` arrives in the uniform block, baked into KernelParams when the frame
    // transform was computed, while the texture is uploaded from an independently acquired
    // read of the colour state. The two can disagree for a single frame after the LUT
    // changes. Skipping the LUT for that frame is strictly better than grading with
    // clamped-garbage LUT values; the next frame corrects itself either way.
    ivec2 lut_tex_size = textureSize(texLut, 0);
    bool lut_usable = lut_tex_size.x == lut_size * lut_size && lut_tex_size.y == lut_size;
    if (bool(params.flags & 8192) && lut_size >= 2 && lut_strength > 0.0 && lut_usable) {
        vec3 graded = lut_sample(c, lut_size, params.color_params_3.x, params.color_params_3.y);
        c = c + (graded - c) * lut_strength;
    }

    // 2. Exposure
    c = c * params.color_params_1.x;

    // 3. White balance
    c = c * params.color_params_2.xyz;

    // 4. Contrast around mid grey
    c = (c - vec3(0.5)) * params.color_params_1.y + vec3(0.5);

    // 5. Shadows / highlights, each weighted to its own end of the tone range.
    // Cubic weights, clamped at zero - a step-for-step duplicate of `apply_color` in
    // src/core/color/mod.rs. Operation order must not be rearranged.
    float tone_luma = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    float sw = max(1.0 - tone_luma, 0.0);
    float hw = max(tone_luma, 0.0);
    float shadow_w = sw * sw * sw;
    float highlight_w = hw * hw * hw;
    float tone = params.color_params_3.z * shadow_w + params.color_params_3.w * highlight_w;
    c = c + vec3(tone);

    // 6. Saturation around luma
    float luma = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    c = vec3(luma) + (c - vec3(luma)) * params.color_params_1.z;

    return vec4(c * max_value, px.a);
}

// Called at every site that finishes a fragment through the sampling path, always before
// the drawing and safe-area overlays are composited - the overlays are UI, not image, and
// must never be graded. Background pixels produced by that path are graded along with image
// pixels, which matches the wgpu kernel where the background is part of the accumulated sum
// that apply_color() receives.
//
// The FILL_WITH_BACKGROUND (flags & 4) early-out in main() is deliberately NOT graded: it
// never enters the accumulation path at all, and both approved sibling backends skip colour
// there too - wgpu_undistort.wgsl returns the background before reaching apply_color, and
// cpu_undistort.rs early-returns the background before apply_color_stage.
void apply_color_if_enabled(inout vec4 out_pix) {
    if (bool(params.flags & 4096)) {
        out_pix = apply_color(out_pix);
    }
}
////////////////////////////// Colour adjustments //////////////////////////////

const vec4 colors[9] = vec4[9](
    vec4(0.0,   0.0,   0.0,     0.0), // None
    vec4(255.0, 0.0,   0.0,   255.0), // Red
    vec4(0.0,   255.0, 0.0,   255.0), // Green
    vec4(0.0,   0.0,   255.0, 255.0), // Blue
    vec4(254.0, 251.0, 71.0,  255.0), // Yellow
    vec4(200.0, 200.0, 0.0,   255.0), // Yellow2
    vec4(255.0, 0.0,   255.0, 255.0), // Magenta
    vec4(0.0,   128.0, 255.0, 255.0), // Blue2
    vec4(0.0,   200.0, 200.0, 255.0)  // Blue3
);
const float alphas[4] = float[4](1.0, 0.75, 0.50, 0.25);
void draw_pixel(inout vec4 out_pix, float x, float y, bool isInput) {
    if (!bool(params.flags & 8)) { // Drawing not enabled
        return;
    }
    int width = max(params.width, params.output_width);
    int height = max(params.height, params.output_height);

    int data = int(ceil(texture(texCanvas, vec2(x / width, y / height)).r * 255.0));
    if (data > 0) {
        int color = (data & 0xF8) >> 3;
        int alpha = (data & 0x06) >> 1;
        int stage = data & 1;
        if (((stage == 0 && isInput) || (stage == 1 && !isInput)) && color < 9) {
            vec4 colorf = colors[color] / 255.0;
            float alphaf = alphas[alpha];
            out_pix = colorf * alphaf + out_pix * (1.0 - alphaf);
            out_pix.a = 1.0;
        }
    }
}
void draw_safe_area(inout vec4 out_pix, float x, float y) {
    bool isSafeArea = x >= params.safe_area_rect.x && x <= params.safe_area_rect.z &&
                      y >= params.safe_area_rect.y && y <= params.safe_area_rect.w;
    if (!isSafeArea) {
        out_pix.x *= 0.5;
        out_pix.y *= 0.5;
        out_pix.z *= 0.5;
        bool isBorder = x >= params.safe_area_rect.x - 5.0 && x <= params.safe_area_rect.z + 5.0 &&
                        y >= params.safe_area_rect.y - 5.0 && y <= params.safe_area_rect.w + 5.0;
        if (isBorder) {
            out_pix.x *= 0.5;
            out_pix.y *= 0.5;
            out_pix.z *= 0.5;
        }
    }
}

float get_param(float row, float idx) {
    int size = bool(params.flags & 16)? params.width : params.height;
    return texture(texParams, vec2(idx / 13.0, row / float(size - 1))).r;
}

float map_coord(float x, float in_min, float in_max, float out_min, float out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

vec2 rotate_and_distort(vec2 pos, float idx) {
    float _x = (float(pos.x) * get_param(idx, 0)) + (float(pos.y) * get_param(idx, 1)) + get_param(idx, 2) + params.translation3d.x;
    float _y = (float(pos.x) * get_param(idx, 3)) + (float(pos.y) * get_param(idx, 4)) + get_param(idx, 5) + params.translation3d.y;
    float _w = (float(pos.x) * get_param(idx, 6)) + (float(pos.y) * get_param(idx, 7)) + get_param(idx, 8) + params.translation3d.z;

    if (_w > 0.0) {
        if (params.r_limit > 0.0 && length(vec2(_x, _y) / _w) > params.r_limit) {
            return vec2(-99999.0, -99999.0);
        }

        if (params.light_refraction_coefficient != 1.0 && params.light_refraction_coefficient > 0.0) {
            float r = length(vec2(_x, _y)) / _w;
            float sin_theta_d = (r / sqrt(1.0 + r * r)) * params.light_refraction_coefficient;
            float r_d = sin_theta_d / sqrt(1.0 - sin_theta_d * sin_theta_d);
            if (r_d != 0.0) {
                _w *= r / r_d;
            }
        }

        vec2 uv = params.f * distort_point(_x, _y, _w) + params.c;

        if (get_param(idx, 9) != 0.0 || get_param(idx, 10) != 0.0 || get_param(idx, 11) != 0.0 || get_param(idx, 12) != 0.0 || get_param(idx, 13) != 0.0) {
            float ang_rad = get_param(idx, 11);
            float cos_a = cos(-ang_rad);
            float sin_a = sin(-ang_rad);
            uv -= params.c;
            uv = vec2(
                cos_a * uv.x - sin_a * uv.y - get_param(idx, 9)  + get_param(idx, 12),
                sin_a * uv.x + cos_a * uv.y - get_param(idx, 10) + get_param(idx, 13)
            );
            uv += params.c;
        }
        uv = process_coord(uv, idx);

        if (bool(params.flags & 2)) { // Has digital lens
            uv = digital_distort_point(uv);
        }

        if (params.input_horizontal_stretch > 0.001) { uv.x /= params.input_horizontal_stretch; }
        if (params.input_vertical_stretch   > 0.001) { uv.y /= params.input_vertical_stretch; }

        return uv;
    }
    return vec2(-99999.0, -99999.0);
}

vec2 rotate_point(vec2 pos, float angle, vec2 origin, vec2 origin2) {
     return vec2(cos(angle) * (pos.x - origin.x) - sin(angle) * (pos.y - origin.y) + origin2.x,
                 sin(angle) * (pos.x - origin.x) + cos(angle) * (pos.y - origin.y) + origin2.y);
}
void main() {
    vec2 texPos = v_texcoord.xy * vec2(params.output_width, params.output_height) + params.translation2d;
    vec2 outPos = v_texcoord.xy * vec2(params.output_width, params.output_height);

    if (bool(params.flags & 4)) { // Fill with background
        fragColor = params.background;
        return;
    }

    ///////////////////////////////////////////////////////////////////
    // Add lens distortion back
    if (params.lens_correction_amount < 1.0) {
        float factor = max(1.0 - params.lens_correction_amount, 0.001); // FIXME: this is close but wrong
        vec2 out_c = vec2(params.output_width / 2.0, params.output_height / 2.0);
        vec2 out_f = (params.f / params.fov) / factor;

        vec2 new_out_pos = texPos;

        if (bool(params.flags & 2)) { // Has digital lens
            // Apply the digital warp in the UN-zoomed (fov=1) frame so it's FOV-independent. The warp is a
            // frame-relative pixel map; evaluating it on post-zoom pixels made the corrected shape (and the
            // adaptive-zoom bounding box from it) depend on the zoom. Un-zoom -> warp -> re-zoom.
            new_out_pos = (new_out_pos - out_c) * params.fov + out_c;
            new_out_pos = digital_undistort_point(new_out_pos);
            new_out_pos = (new_out_pos - out_c) / params.fov + out_c;
        }

        new_out_pos = (new_out_pos - out_c) / out_f;
        new_out_pos = undistort_point(new_out_pos);
        if (params.light_refraction_coefficient != 1.0 && params.light_refraction_coefficient > 0.0) {
            float r = length(new_out_pos);
            if (r != 0.0) {
                float sin_theta_d = (r / sqrt(1.0 + r * r)) / params.light_refraction_coefficient;
                float r_d = sin_theta_d / sqrt(1.0 - sin_theta_d * sin_theta_d);
                new_out_pos *= r_d / r;
            }
        }
        new_out_pos = out_f * new_out_pos + out_c;

        texPos = new_out_pos * (1.0 - params.lens_correction_amount) + (texPos * params.lens_correction_amount);
    }
    ///////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////
    // Calculate source `y` for rolling shutter
    float sy = texPos.y;
    if (bool(params.flags & 16)) { // Horizontal RS
        sy = min(params.width, max(0, floor(0.5 + texPos.x)));
    } else {
        sy = min(params.height, max(0, floor(0.5 + texPos.y)));
    }
    if (params.matrix_count > 1) {
        float idx = params.matrix_count / 2.0; // Use middle matrix
        vec2 uv = rotate_and_distort(texPos, idx);
        if (uv.x > -99998.0) {
            if (bool(params.flags & 16)) { // Horizontal RS
                sy = min(params.width, max(0, floor(0.5 + uv.x)));
            } else {
                sy = min(params.height, max(0, floor(0.5 + uv.y)));
            }
        }
    }
    ///////////////////////////////////////////////////////////////////

    float idx = min(sy, params.matrix_count - 1.0);

    vec2 uv = rotate_and_distort(texPos, idx);
    vec2 frame_size = vec2(params.width, params.height);
    if (params.input_rotation != 0.0) {
        float rotation = params.input_rotation * (3.1415926535897 / 180.0);
        vec2 size = frame_size;
        frame_size = abs(round(rotate_point(size, rotation, vec2(0.0, 0.0), vec2(0.0, 0.0))));
        uv = rotate_point(uv, rotation, size / vec2(2.0), frame_size / vec2(2.0));
    }

    if (uv.x > -99998.0) {
        if (params.background_mode == 1) { // edge repeat
            uv = max(vec2(0, 0), min(vec2(params.width - 1, params.height - 1), uv));
        } else if (params.background_mode == 2) { // edge mirror
            float width3 = (params.width - 2);
            float height3 = (params.height - 2);
            if (uv.x > width3)  uv.x = width3  - (uv.x - width3);
            if (uv.x < 2)       uv.x = 2 + params.width - (width3  + uv.x);
            if (uv.y > height3) uv.y = height3 - (uv.y - height3);
            if (uv.y < 2)       uv.y = 2 + params.height - (height3 + uv.y);
        } else if (params.background_mode == 3) { // margin with feather
            float widthf  = (params.width  - 1);
            float heightf = (params.height - 1);

            float feather = max(0.0001, params.background_margin_feather * heightf);
            vec2 pt2 = uv;
            float alpha = 1.0;
            if ((uv.x > widthf - feather) || (uv.x < feather) || (uv.y > heightf - feather) || (uv.y < feather)) {
                alpha = max(0.0, min(1.0, min(min(widthf - uv.x, heightf - uv.y), min(uv.x, uv.y)) / feather));
                pt2 /= vec2(widthf, heightf);
                pt2 = ((pt2 - 0.5) * (1.0 - params.background_margin)) + 0.5;
                pt2 *= vec2(widthf, heightf);
            }

            vec4 c1 = texture(texIn, vec2(uv.x / params.width, uv.y / params.height));
            vec4 c2 = texture(texIn, vec2(pt2.x / params.width, pt2.y / params.height));
            fragColor = c1 * alpha + c2 * (1.0 - alpha);
            fragColor.a = 1.0;
            if (!((pt2.x >= 0 && pt2.x < params.width) && (pt2.y >= 0 && pt2.y < params.height))) {
                fragColor = params.background;
            }
            apply_color_if_enabled(fragColor);
            draw_pixel(fragColor, uv.x, uv.y, true);
            draw_pixel(fragColor, outPos.x, outPos.y, false);
            draw_safe_area(fragColor, outPos.x, outPos.y);
            return;
        }

        if ((uv.x >= 0 && uv.x < frame_size.x) && (uv.y >= 0 && uv.y < frame_size.y)) {
            fragColor = texture(texIn, vec2(uv.x / frame_size.x, uv.y / frame_size.y));
            apply_color_if_enabled(fragColor);
            draw_pixel(fragColor, uv.x, uv.y, true);
            draw_pixel(fragColor, outPos.x, outPos.y, false);
            draw_safe_area(fragColor, outPos.x, outPos.y);
            return;
        }
    }

    fragColor = params.background;
    apply_color_if_enabled(fragColor);
    draw_pixel(fragColor, outPos.x, outPos.y, false);
    draw_safe_area(fragColor, outPos.x, outPos.y);
}
