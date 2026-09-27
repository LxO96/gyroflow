// Copyright © 2026 Gyroflow
// Chroma smoothing pass. Mirror of `denoise::chroma_filter` (Rust); tested against it.
// DATA_TYPE / DATA_TYPEF / DATA_CONVERT / DATA_CONVERTF / PIXEL_BYTES / GUIDE_TYPE / ROUND_OFFSET
// are substituted by opencl.rs.

typedef struct {
    int width, height, stride, channels;
    int guide_width, guide_height, guide_stride, mode;
    float amount, sigma_y, max_value, pad;
} ChromaFilterParams;

#define CF_RADIUS 8
#define CF_STEP 2
#define CF_SIGMA_SPATIAL 4.0f

inline float4 cf_load(__global const uchar *buf, int x, int y, __global const ChromaFilterParams *p) {
    x = clamp(x, 0, p->width - 1);
    y = clamp(y, 0, p->height - 1);
    float4 v = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    *(DATA_TYPEF *)&v = DATA_CONVERTF(*(__global const DATA_TYPE *)(buf + y * p->stride + x * PIXEL_BYTES));
    return v / p->max_value;
}

// Rounded: an odd-sized plane has a chroma plane of ceil(n/2), and 3841 / 1921 must be 2.
inline int cf_ratio(int g, int n) { return max(1, (g + n / 2) / n); }

inline float cf_guide(__global const uchar *g, int x, int y, __global const ChromaFilterParams *p) {
    x = clamp(x, 0, p->width - 1);
    y = clamp(y, 0, p->height - 1);
    int rx = cf_ratio(p->guide_width, p->width), ry = cf_ratio(p->guide_height, p->height);
    float s = 0.0f;
    for (int j = 0; j < ry; j++) {
        int gy = min(y * ry + j, p->guide_height - 1);
        for (int i = 0; i < rx; i++) {
            int gx = min(x * rx + i, p->guide_width - 1);
            s += (float)(((__global const GUIDE_TYPE *)(g + gy * p->guide_stride))[gx]);
        }
    }
    return s / (float)(rx * ry) / p->max_value;
}

inline float cf_luma(float4 v) { return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z; }

__kernel void chroma_filter(__global const uchar *src, __global uchar *dst, __global const uchar *guide, __global const void *params_buf) {
    __global const ChromaFilterParams *p = (__global const ChromaFilterParams *)params_buf;
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= p->width || y >= p->height) return;

    bool rgb = p->mode == 2;
    float4 c0 = cf_load(src, x, y, p);
    float g0 = rgb? cf_luma(c0) : cf_guide(guide, x, y, p);
    float4 v0 = rgb? c0 - (float4)(g0, g0, g0, 0.0f) : c0;

    float inv_s = 1.0f / (2.0f * CF_SIGMA_SPATIAL * CF_SIGMA_SPATIAL);
    float inv_y = 1.0f / (2.0f * p->sigma_y * p->sigma_y);
    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float wsum = 0.0f;
    for (int dy = -CF_RADIUS; dy <= CF_RADIUS; dy += CF_STEP) {
        for (int dx = -CF_RADIUS; dx <= CF_RADIUS; dx += CF_STEP) {
            float4 c = cf_load(src, x + dx, y + dy, p);
            float g = rgb? cf_luma(c) : cf_guide(guide, x + dx, y + dy, p);
            float4 v = rgb? c - (float4)(g, g, g, 0.0f) : c;
            float w = exp(-(float)(dx * dx + dy * dy) * inv_s - (g - g0) * (g - g0) * inv_y);
            acc += v * w;
            wsum += w;
        }
    }
    float4 out = v0 + p->amount * (acc / wsum - v0);
    if (rgb) { out.xyz += g0; out.w = c0.w; }
    out = out * p->max_value + ROUND_OFFSET;
    DATA_TYPEF o = *(DATA_TYPEF *)&out;
    *(__global DATA_TYPE *)(dst + y * p->stride + x * PIXEL_BYTES) = DATA_CONVERT(o);
}
