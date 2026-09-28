/* llmengine_moe_qwen3.c — dequant MLX (MLX affine on-read) + SwiGLU esperto. */
#include "llmengine_moe_qwen3.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t* gate_w; const uint8_t* gate_s; const uint8_t* gate_b;
    const uint8_t* up_w;   const uint8_t* up_s;   const uint8_t* up_b;
    const uint8_t* down_w; const uint8_t* down_s; const uint8_t* down_b;
    uint32_t hidden, inter, group, bits, per;
    int sc_f16;   /* scales/biases F16 (flag bit0 nel header MLX) invece di BF16 */
} Q3;

static float bf16_to_f32(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, u;
    if (e == 0) { if (m == 0) u = s; else { int sh = 0; while (!(m & 0x400u)) { m <<= 1; sh++; } m &= 0x3FFu; u = s | ((uint32_t)(127 - 15 - sh + 1) << 23) | (m << 13); } }
    else if (e == 31) u = s | 0x7F800000u | (m << 13);
    else u = s | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}

static uint32_t get_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

LlmEngineMoeStatus llmengine_moe_q3_blob_dims(const void* blob, size_t size,
                                  uint32_t* hidden, uint32_t* inter,
                                  uint32_t* group, uint32_t* bits) {
    if (blob == NULL || size < 32) return LLMENGINE_MOE_ERR_FORMAT;
    const uint8_t* p = (const uint8_t*)blob;
    if (memcmp(p, "MLX", 4) != 0) return LLMENGINE_MOE_ERR_FORMAT;
    if (hidden) *hidden = get_u32(p + 16);
    if (inter)  *inter  = get_u32(p + 20);
    if (group)  *group  = get_u32(p + 8);
    if (bits)   *bits   = get_u32(p + 12);
    return LLMENGINE_MOE_OK;
}

static int q3_layout(const void* blob, size_t size, Q3* q) {
    if (llmengine_moe_q3_blob_dims(blob, size, &q->hidden, &q->inter, &q->group, &q->bits) != LLMENGINE_MOE_OK)
        return 0;
    if (q->bits < 1 || q->bits > 16) return 0;   /* 1..16 (flusso contiguo LSB-first) */
    if (q->group == 0 || (q->hidden % q->group) || (q->inter % q->group)) return 0;
    q->sc_f16 = (get_u32((const uint8_t*)blob + 28) & 1u) ? 1 : 0;
    q->per = 32 / q->bits;
    uint32_t H = q->hidden, I = q->inter, g = q->group;
    size_t gw = (size_t)I * ((size_t)H * q->bits / 32) * 4, gsb = (size_t)I * (H / g) * 2;
    size_t dw = (size_t)H * ((size_t)I * q->bits / 32) * 4, dsb = (size_t)H * (I / g) * 2;
    size_t need = 32 + 2 * (gw + 2 * gsb) + (dw + 2 * dsb);
    if (size < need) return 0;
    const uint8_t* p = (const uint8_t*)blob + 32;
    q->gate_w = p; p += gw; q->gate_s = p; p += gsb; q->gate_b = p; p += gsb;
    q->up_w   = p; p += gw; q->up_s   = p; p += gsb; q->up_b   = p; p += gsb;
    q->down_w = p; p += dw; q->down_s = p; p += dsb; q->down_b = p; p += dsb;
    return 1;
}

static void q3_gemv_ex(const void* w, const void* scales, const void* biases,
                       uint32_t rows, uint32_t cols, uint32_t group, uint32_t bits, int sc_f16,
                       const float* x, float* out) {
    uint32_t packed_cols = (uint32_t)((uint64_t)cols * bits / 32u);
    const uint32_t* ww = (const uint32_t*)w;
    const uint16_t* sc = (const uint16_t*)scales;
    const uint16_t* bi = (const uint16_t*)biases;
    uint32_t gcols = cols / group;
    uint32_t mask = (bits >= 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    for (uint32_t o = 0; o < rows; ++o) {
        const uint32_t* wrow = ww + (size_t)o * packed_cols;
        const uint16_t* srow = sc + (size_t)o * gcols;
        const uint16_t* brow = bi + (size_t)o * gcols;
        double acc = 0.0;
        for (uint32_t k = 0; k < cols; ++k) {
            size_t bitpos = (size_t)k * bits; uint32_t q = 0;
            for (uint32_t b = 0; b < bits; ++b) { q |= ((wrow[bitpos >> 5] >> (bitpos & 31u)) & 1u) << b; bitpos++; }
            float s = sc_f16 ? f16_to_f32(srow[k / group]) : bf16_to_f32(srow[k / group]);
            float b = sc_f16 ? f16_to_f32(brow[k / group]) : bf16_to_f32(brow[k / group]);
            acc += ((double)s * (double)q + (double)b) * (double)x[k];
        }
        out[o] = (float)acc;
    }
}

void llmengine_moe_q3_gemv(const void* w, const void* scales, const void* biases,
                     uint32_t rows, uint32_t cols, uint32_t group, uint32_t bits,
                     const float* x, float* out) {
    q3_gemv_ex(w, scales, biases, rows, cols, group, bits, 0, x, out);
}

LlmEngineMoeStatus llmengine_moe_q3_expert_swiglu(const void* blob, size_t size,
                                      const float* x, float* out) {
    Q3 q;
    if (x == NULL || out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (!q3_layout(blob, size, &q)) return LLMENGINE_MOE_ERR_FORMAT;
    float* g = (float*)malloc((size_t)q.inter * sizeof(float));
    float* u = (float*)malloc((size_t)q.inter * sizeof(float));
    if (g == NULL || u == NULL) { free(g); free(u); return LLMENGINE_MOE_ERR_NO_MEMORY; }
    q3_gemv_ex(q.gate_w, q.gate_s, q.gate_b, q.inter, q.hidden, q.group, q.bits, q.sc_f16, x, g);
    q3_gemv_ex(q.up_w,   q.up_s,   q.up_b,   q.inter, q.hidden, q.group, q.bits, q.sc_f16, x, u);
    for (uint32_t i = 0; i < q.inter; ++i) {
        float silu = g[i] / (1.0f + expf(-g[i]));
        g[i] = silu * u[i];
    }
    q3_gemv_ex(q.down_w, q.down_s, q.down_b, q.hidden, q.inter, q.group, q.bits, q.sc_f16, g, out);
    free(g); free(u);
    return LLMENGINE_MOE_OK;
}

/* M17 — GEMM batched dequant-on-read: OUT[n][rows] = X[n][cols] · W^T (W = rows×cols quantizzato).
 * Il peso è DECODIFICATO una volta per (riga, colonna) e riusato per le n posizioni: è questo che
 * ammortizza il compute del prefill. L'ordine di accumulo per output è identico al gemv → per n=1
 * è bit-identico. */
static void q3_gemm(const void* w, const void* scales, const void* biases,
                    uint32_t rows, uint32_t cols, uint32_t group, uint32_t bits, int sc_f16,
                    const float* X, uint32_t n, float* out) {
    uint32_t per = 32 / bits;
    uint32_t packed_cols = cols / per;
    const uint32_t* ww = (const uint32_t*)w;
    const uint16_t* sc = (const uint16_t*)scales;
    const uint16_t* bi = (const uint16_t*)biases;
    uint32_t gcols = cols / group;
    uint32_t mask = (bits >= 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    double* acc = (double*)malloc((size_t)n * sizeof(double));
    if (acc == NULL) return;
    for (uint32_t o = 0; o < rows; ++o) {
        const uint32_t* wrow = ww + (size_t)o * packed_cols;
        const uint16_t* srow = sc + (size_t)o * gcols;
        const uint16_t* brow = bi + (size_t)o * gcols;
        for (uint32_t j = 0; j < n; ++j) acc[j] = 0.0;
        for (uint32_t k = 0; k < cols; ++k) {
            uint32_t qv = (wrow[k / per] >> (bits * (k % per))) & mask;
            float s = sc_f16 ? f16_to_f32(srow[k / group]) : bf16_to_f32(srow[k / group]);
            float b = sc_f16 ? f16_to_f32(brow[k / group]) : bf16_to_f32(brow[k / group]);
            double wval = (double)s * (double)qv + (double)b;
            for (uint32_t j = 0; j < n; ++j) acc[j] += wval * (double)X[(size_t)j * cols + k];
        }
        for (uint32_t j = 0; j < n; ++j) out[(size_t)j * rows + o] = (float)acc[j];
    }
    free(acc);
}

LlmEngineMoeStatus llmengine_moe_q3_expert_swiglu_batch(const void* blob, size_t size,
                                            const float* X, uint32_t n, float* out) {
    Q3 q;
    if (X == NULL || out == NULL || n == 0) return LLMENGINE_MOE_ERR_INVALID;
    if (!q3_layout(blob, size, &q)) return LLMENGINE_MOE_ERR_FORMAT;
    uint32_t H = q.hidden, I = q.inter;
    float* g = (float*)malloc((size_t)n * I * sizeof(float));
    float* u = (float*)malloc((size_t)n * I * sizeof(float));
    if (g == NULL || u == NULL) { free(g); free(u); return LLMENGINE_MOE_ERR_NO_MEMORY; }
    q3_gemm(q.gate_w, q.gate_s, q.gate_b, I, H, q.group, q.bits, q.sc_f16, X, n, g);
    q3_gemm(q.up_w,   q.up_s,   q.up_b,   I, H, q.group, q.bits, q.sc_f16, X, n, u);
    for (size_t i = 0; i < (size_t)n * I; ++i) {
        float silu = g[i] / (1.0f + expf(-g[i]));
        g[i] = silu * u[i];
    }
    q3_gemm(q.down_w, q.down_s, q.down_b, H, I, q.group, q.bits, q.sc_f16, g, n, out);
    free(g); free(u);
    return LLMENGINE_MOE_OK;
}
