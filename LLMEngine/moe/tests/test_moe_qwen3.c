/* test_moe_qwen3.c — M15.3 gate: dequant MLX (MLX affine on-read) + SwiGLU esperto.
 * Costruisce un blob MLX sintetico e confronta llmengine_moe_q3_expert_swiglu con un riferimento
 * naive (dequant in double). */
#include "llmengine_moe_qwen3.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put_f32_as_bf16(uint8_t* p, float f) { uint32_t u; memcpy(&u, &f, 4); uint16_t h = (uint16_t)(u >> 16); memcpy(p, &h, 2); }
static uint16_t rd_bf16(const uint8_t* p) { uint16_t h; memcpy(&h, p, 2); return h; }
static float bf16_to_f32(uint16_t h) { uint32_t u = ((uint32_t)h) << 16; float f; memcpy(&f, &u, 4); return f; }

/* Riempie un sotto-tensore quantizzato (rows×cols) con valori determinini e scrive il riferimento
 * dequant in ref[rows][cols]. Restituisce i byte scritti. */
static size_t fill_quant(uint8_t* p, uint32_t rows, uint32_t cols, uint32_t group, uint32_t bits,
                         float* ref_out) {
    uint32_t per = 32 / bits;
    uint32_t packed_cols = cols / per;
    size_t wbytes = (size_t)rows * packed_cols * 4;
    uint32_t gcols = cols / group;
    size_t sbytes = (size_t)rows * gcols * 2;
    uint32_t mask = (1u << bits) - 1u;
    /* interleave: weight, scales, biases */
    uint8_t* wp = p;
    uint8_t* sp = p + wbytes;
    uint8_t* bp = p + wbytes + sbytes;
    for (uint32_t o = 0; o < rows; ++o) {
        for (uint32_t k = 0; k < cols; ++k) {
            uint32_t q = (uint32_t)((o * 7 + k * 13 + 3) & mask);
            float s = 0.01f + 0.001f * (float)((o + k / group) % 9);
            float b = -0.02f + 0.001f * (float)((o * 2 + k / group) % 5);
            uint32_t* word = (uint32_t*)(wp + (size_t)o * packed_cols * 4) + (k / per);
            *word |= q << (bits * (k % per));
            if (k % group == 0) { put_f32_as_bf16(sp + ((size_t)o * gcols + k / group) * 2, s); put_f32_as_bf16(bp + ((size_t)o * gcols + k / group) * 2, b); }
            ref_out[(size_t)o * cols + k] = bf16_to_f32(rd_bf16(sp + ((size_t)o * gcols + k / group) * 2)) * (float)q
                                          + bf16_to_f32(rd_bf16(bp + ((size_t)o * gcols + k / group) * 2));
        }
    }
    return wbytes + 2 * sbytes;
}

int main(void) {
    const uint32_t H = 8, I = 4, group = 4, bits = 8, per = 32 / bits;
    uint32_t gw = I * (H / per) * 4, gsb = I * (H / group) * 2;
    uint32_t dw = H * (I / per) * 4, dsb = H * (I / group) * 2;
    size_t need = 32 + 2 * (gw + 2 * gsb) + (dw + 2 * dsb);
    uint8_t* blob = (uint8_t*)calloc(1, need);
    assert(blob != NULL);
    memcpy(blob, "MLX", 4);
    uint32_t hdr[7] = { 1, group, bits, H, I, 9, 0 };
    memcpy(blob + 4, hdr, sizeof(hdr));

    uint32_t* ref_gate = malloc(sizeof(float) * I * H);
    uint32_t* ref_up = malloc(sizeof(float) * I * H);
    uint32_t* ref_down = malloc(sizeof(float) * H * I);
    float* rg = (float*)ref_gate; float* ru = (float*)ref_up; float* rd = (float*)ref_down;

    uint8_t* p = blob + 32;
    p += fill_quant(p, I, H, group, bits, rg);   /* gate */
    p += fill_quant(p, I, H, group, bits, ru);   /* up */
    p += fill_quant(p, H, I, group, bits, rd);   /* down */
    assert((size_t)(p - blob) == need);

    uint32_t hh, ii, gg, bb;
    assert(llmengine_moe_q3_blob_dims(blob, need, &hh, &ii, &gg, &bb) == LLMENGINE_MOE_OK);
    assert(hh == H && ii == I && gg == group && bb == bits);

    float x[8] = {0.5f, -1.2f, 0.3f, 2.0f, -0.7f, 0.1f, 1.1f, -0.4f};
    float out[8];
    assert(llmengine_moe_q3_expert_swiglu(blob, need, x, out) == LLMENGINE_MOE_OK);

    /* riferimento naive (double): g=silu(gate·x); u=up·x; h=g*u; out=down·h */
    double g[4], u[4], hv[4], o[8];
    for (uint32_t r = 0; r < I; ++r) { double a = 0; for (uint32_t k = 0; k < H; ++k) a += (double)rg[r * H + k] * x[k]; g[r] = a; }
    for (uint32_t r = 0; r < I; ++r) { double a = 0; for (uint32_t k = 0; k < H; ++k) a += (double)ru[r * H + k] * x[k]; u[r] = a; }
    for (uint32_t i = 0; i < I; ++i) hv[i] = (g[i] / (1.0 + exp(-g[i]))) * u[i];
    for (uint32_t r = 0; r < H; ++r) { double a = 0; for (uint32_t k = 0; k < I; ++k) a += (double)rd[r * I + k] * hv[k]; o[r] = a; }

    double maxdiff = 0;
    for (uint32_t r = 0; r < H; ++r) { double d = fabs((double)out[r] - o[r]); if (d > maxdiff) maxdiff = d; }
    printf("[m15.3] SwiGLU esperto vs riferimento: maxdiff=%.3e\n", maxdiff);
    assert(maxdiff < 1e-4);

    /* blob malformato → FORMAT */
    assert(llmengine_moe_q3_expert_swiglu(blob, 16, x, out) == LLMENGINE_MOE_ERR_FORMAT);

    free(blob); free(ref_gate); free(ref_up); free(ref_down);
    printf("test_moe_qwen3: PASS\n");
    return 0;
}
