// Copyright © 2026 Gyroflow
// Film chain. Mirror of `film::reference` (Rust); tested against it in film/tests_gpu.rs.
// DATA_TYPE / DATA_TYPEF / DATA_CONVERT / DATA_CONVERTF / PIXEL_BYTES / ROUND_OFFSET
// are substituted by opencl.rs.

typedef struct {
    int width, height, stride; uint flags;
    int halo_f, halo_w, halo_h, halo_radius;
    int grain_radius; float max_value, grain_k, grain_sigma;
    float halo_sigma, halo_strength, inv_p, fit_b;
    float fit_s, fit_umin_a, pad0, pad1;
    uint4 seeds;
    float4 halo_tint, log_e0, k, u_min, range;
    float4 sens[3];
    float4 cross[3];
    float std_table[256];
} FilmParams;

#define FILM_TONE 1u
#define FILM_GRAIN 2u
#define FILM_HALO 4u
#define FILM_BW 8u
#define NOISE_SALT 0x3C6EF372u

inline uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
inline uint hash4(uint a, uint b, uint c, uint d) { return pcg(a ^ pcg(b ^ pcg(c ^ pcg(d)))); }
inline float film_gauss(uint h) {
    uint h1 = pcg(h), h2 = pcg(h1);
    float u1 = max((float)(h1 >> 8u) / 16777216.0f, 1.0f / 16777216.0f);
    float u2 = (float)(h2 >> 8u) / 16777216.0f;
    return sqrt(-2.0f * log(u1)) * cos(6.283185307179586f * u2);
}
inline float srgb_dec(float v) { return v <= 0.04045f ? v / 12.92f : pow((v + 0.055f) / 1.055f, 2.4f); }
inline float srgb_enc(float x) { return x <= 0.0031308f ? x * 12.92f : 1.055f * pow(x, 1.0f / 2.4f) - 0.055f; }
inline float smooth_step(float a, float b, float x) { float t = clamp((x - a) / (b - a), 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }
inline float gw(int k, float s) { return exp(-(float)(k * k) / (2.0f * s * s)); }
inline uint export_px(int x, float inv_p) { return (uint)(int)floor(((float)x + 0.5f) * inv_p); }
inline uint seed_of(__global const FilmParams *p, int l) { return l == 0 ? p->seeds.x : (l == 1 ? p->seeds.y : p->seeds.z); }

inline float4 load_px(__global const uchar *buf, int x, int y, __global const FilmParams *p) {
    float4 v = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    *(DATA_TYPEF *)&v = DATA_CONVERTF(*(__global const DATA_TYPE *)(buf + y * p->stride + x * PIXEL_BYTES));
    return v / p->max_value;
}

// H1: f x f box average of the highlight source.
__kernel void film_halo_source(__global const uchar *src, __global float *halo, __global const FilmParams *p) {
    int i = get_global_id(0), j = get_global_id(1);
    if (i >= p->halo_w || j >= p->halo_h) return;
    int f = p->halo_f;
    float s = 0.0f;
    for (int b = 0; b < f; b++) {
        int y = min(j * f + b, p->height - 1);
        for (int a = 0; a < f; a++) {
            float4 c = load_px(src, min(i * f + a, p->width - 1), y, p);
            float Y = 0.2126f * srgb_dec(c.x) + 0.7152f * srgb_dec(c.y) + 0.0722f * srgb_dec(c.z);
            s += smooth_step(0.75f, 1.0f, Y) * Y;
        }
    }
    halo[j * p->halo_w + i] = s / (float)(f * f);
}

// H2 / H3: separable clamp-edge Gaussian on the halo grid.
__kernel void film_halo_blur(__global const float *in, __global float *out, __global const FilmParams *p, int horizontal) {
    int i = get_global_id(0), j = get_global_id(1);
    if (i >= p->halo_w || j >= p->halo_h) return;
    int r = p->halo_radius;
    float s = 0.0f, ws = 0.0f;
    for (int t = -r; t <= r; t++) {
        float w = gw(t, p->halo_sigma);
        int x = horizontal ? clamp(i + t, 0, p->halo_w - 1) : i;
        int y = horizontal ? j : clamp(j + t, 0, p->halo_h - 1);
        s += w * in[y * p->halo_w + x];
        ws += w;
    }
    out[j * p->halo_w + i] = s / ws;
}

// G1: per-layer unit noise, blurred horizontally.
__kernel void film_noise_blur_x(__global float4 *g1, __global const FilmParams *p) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= p->width || y >= p->height) return;
    int r = p->grain_radius, layers = (p->flags & FILM_BW) ? 1 : 3;
    uint ey = export_px(y, p->inv_p);
    float n[3] = { 0.0f, 0.0f, 0.0f };
    float ws = 0.0f;
    for (int t = -r; t <= r; t++) {
        float w = gw(t, p->grain_sigma);
        uint ex = export_px(clamp(x + t, 0, p->width - 1), p->inv_p);
        for (int l = 0; l < layers; l++) n[l] += w * film_gauss(hash4(ex, ey, seed_of(p, l), NOISE_SALT));
        ws += w;
    }
    g1[y * p->width + x] = (float4)(n[0] / ws, n[1] / ws, n[2] / ws, 0.0f);
}

inline float sample_halo(__global const float *h, float hx, float hy, __global const FilmParams *p) {
    float fx = hx - 0.5f, fy = hy - 0.5f;
    int x0 = (int)floor(fx), y0 = (int)floor(fy);
    float tx = fx - (float)x0, ty = fy - (float)y0;
    int xa = clamp(x0, 0, p->halo_w - 1), xb = clamp(x0 + 1, 0, p->halo_w - 1);
    int ya = clamp(y0, 0, p->halo_h - 1), yb = clamp(y0 + 1, 0, p->halo_h - 1);
    return (h[ya * p->halo_w + xa] * (1.0f - tx) + h[ya * p->halo_w + xb] * tx) * (1.0f - ty)
         + (h[yb * p->halo_w + xa] * (1.0f - tx) + h[yb * p->halo_w + xb] * tx) * ty;
}

// G2: vertical blur of the noise, then halation -> curve -> grain -> crosstalk.
__kernel void film_apply(__global const uchar *src, __global const float4 *g1, __global const float *halo, __global uchar *dst, __global const FilmParams *p) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= p->width || y >= p->height) return;
    float4 c = load_px(src, x, y, p);
    float lin[3] = { srgb_dec(c.x), srgb_dec(c.y), srgb_dec(c.z) };
    if (p->flags & FILM_HALO) {
        float f = (float)p->halo_f;
        float hv = sample_halo(halo, ((float)x + 0.5f) / f, ((float)y + 0.5f) / f, p);
        lin[0] += p->halo_strength * hv * p->halo_tint.x;
        lin[1] += p->halo_strength * hv * p->halo_tint.y;
        lin[2] += p->halo_strength * hv * p->halo_tint.z;
    }
    float u[3];
    float lE0[3] = { p->log_e0.x, p->log_e0.y, p->log_e0.z }, kk[3] = { p->k.x, p->k.y, p->k.z };
    float um[3] = { p->u_min.x, p->u_min.y, p->u_min.z }, rg[3] = { p->range.x, p->range.y, p->range.z };
    for (int i = 0; i < 3; i++) {
        if (p->flags & FILM_TONE) {
            float e = p->sens[i].x * lin[0] + p->sens[i].y * lin[1] + p->sens[i].z * lin[2];
            float logE = log10(max(e, 1e-5f)) + p->fit_b;
            float d = um[i] + rg[i] / (1.0f + exp(-kk[i] * (logE - lE0[i])));
            u[i] = p->fit_umin_a + p->fit_s * (d - p->fit_umin_a);
        } else {
            u[i] = srgb_enc(max(lin[i], 0.0f));
        }
    }
    if (p->flags & FILM_GRAIN) {
        int r = p->grain_radius;
        float4 n = (float4)(0.0f);
        float ws = 0.0f;
        for (int t = -r; t <= r; t++) {
            float w = gw(t, p->grain_sigma);
            n += w * g1[clamp(y + t, 0, p->height - 1) * p->width + x];
            ws += w;
        }
        n /= ws;
        float nn[3] = { n.x, n.y, n.z };
        int bw = (p->flags & FILM_BW) != 0;
        for (int i = 0; i < 3; i++) {
            // Linear between table entries; mirrors `reference::std_of`.
            float fu = clamp(u[i], 0.0f, 1.0f) * 200.0f;
            int i0 = min((int)floor(fu), 199);
            float tt = fu - (float)i0;
            float sd = p->std_table[i0] * (1.0f - tt) + p->std_table[i0 + 1] * tt;
            u[i] = clamp(u[i] + p->grain_k * sd * nn[bw ? 0 : i], 0.0f, 1.0f);
        }
    }
    float o[3];
    for (int i = 0; i < 3; i++) {
        o[i] = (p->flags & FILM_TONE) ? clamp(p->cross[i].x * u[0] + p->cross[i].y * u[1] + p->cross[i].z * u[2], 0.0f, 1.0f)
                                       : clamp(u[i], 0.0f, 1.0f);
    }
    float4 out = (float4)(o[0], o[1], o[2], c.w) * p->max_value + ROUND_OFFSET;
    *(__global DATA_TYPE *)(dst + y * p->stride + x * PIXEL_BYTES) = DATA_CONVERT(out);
}
